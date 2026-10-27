"""Render-and-compare utilities for the bottle trajectory solver (used by solve.py)."""
import json
import numpy as np
import torch
import torch.nn.functional as Fn
import trimesh
import h5py
import cv2
import nvdiffrast.torch as dr

dev = 'cuda'


def load_mesh(path):
    return trimesh.load(path, force="scene", process=False).to_geometry()


class Renderer:
    def __init__(self, mesh_path):
        m = load_mesh(mesh_path)
        self.mesh = m
        # NOTE: the raw (unwelded) triangle soup is used on purpose: with the welded watertight mesh the
        # dr.antialias silhouette gradients made the pose optimisation diverge (tested), with the soup it converges.
        self.v = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
        self.f = torch.tensor(m.faces, dtype=torch.int32, device=dev)
        self.fuv = self.f
        uv = np.array(m.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=dev)
        tex = np.array(m.visual.material.baseColorTexture.convert('RGB'), dtype=np.float32) / 255.
        self.tex = torch.tensor(tex, device=dev)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext()


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def blur(x, s):
    """separable gaussian blur of (B,H,W,C) or (B,H,W)"""
    sq = x.ndim == 3
    if sq: x = x[..., None]
    k = int(3 * s) * 2 + 1
    g = torch.exp(-(torch.arange(k, device=x.device, dtype=x.dtype) - k // 2) ** 2 / (2 * s * s)); g = g / g.sum()
    x = x.permute(0, 3, 1, 2); C = x.shape[1]
    x = Fn.conv2d(x, g.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    x = Fn.conv2d(x, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)
    x = x.permute(0, 2, 3, 1)
    return x[..., 0] if sq else x


def Rz(a_deg):
    c, s = np.cos(np.radians(a_deg)), np.sin(np.radians(a_deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def rot_angle(Ra, Rb):
    """pairwise geodesic angle (deg) between (n,3,3) and (m,3,3)"""
    tr = np.einsum('aij,bij->ab', Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


class FrameData:
    """Per-frame crops (image, object mask, valid mask) centred on the SAM2 object mask."""

    def __init__(self, video, masks_h5, S=192):
        mf = h5py.File(masks_h5, 'r')
        OM = mf['object']; PM = mf['person']
        self.F = OM.shape[0]; self.S = S
        cap = cv2.VideoCapture(video)
        self.img = np.zeros((self.F, S, S, 3), np.uint8)
        self.om = np.zeros((self.F, S, S), bool)
        self.val = np.zeros((self.F, S, S), bool)
        self.xy0 = np.zeros((self.F, 2), np.float32)
        self.area = np.zeros(self.F)
        self.cent = np.zeros((self.F, 2))
        for i in range(self.F):
            ok, fr = cap.read()
            assert ok, i
            o = OM[i]; p = PM[i]
            ys, xs = np.nonzero(o)
            self.area[i] = len(xs)
            c = (xs.mean(), ys.mean())
            self.cent[i] = c
            x0 = int(np.clip(c[0] - S / 2, 0, fr.shape[1] - S)); y0 = int(np.clip(c[1] - S / 2, 0, fr.shape[0] - S))
            self.img[i] = fr[y0:y0 + S, x0:x0 + S, ::-1]
            oc = o[y0:y0 + S, x0:x0 + S]; pc = p[y0:y0 + S, x0:x0 + S]
            self.om[i] = oc
            self.val[i] = ~(pc & ~oc)   # pixels covered by the person (not object) cannot be judged
            self.xy0[i] = (x0, y0)

    def batch(self, idx):
        idx = np.asarray(idx)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
        return (t(self.img[idx]).float() / 255., t(self.om[idx]).float(), t(self.val[idx]).float(),
                t(self.xy0[idx, 0]), t(self.xy0[idx, 1]))


class Refiner:
    def __init__(self, R, K, S, wc=0.6, wb=4.0, wd=4.0, ws=0.3):
        self.R = R; self.S = S; self.wc = wc; self.wb = wb; self.wd = wd; self.ws = ws
        self.K = torch.tensor(K, dtype=torch.float32, device=dev)

    def render(self, Rm, t, x0, y0):
        R = self.R; S = self.S
        pc = torch.einsum('bij,vj->bvi', Rm, R.v) + t[:, None]
        z = pc[..., 2]
        u = self.K[0, 0] * pc[..., 0] / z + self.K[0, 2] - x0[:, None]
        v = self.K[1, 1] * pc[..., 1] / z + self.K[1, 2] - y0[:, None]
        n, f = 0.05, 20.
        clip = torch.stack([(2 * u / S - 1) * z, (2 * v / S - 1) * z, z * (f + n) / (f - n) - 2 * f * n / (f - n), z], -1).contiguous()
        rast, _ = dr.rasterize(R.glctx, clip, R.f, (S, S))
        texc, _ = dr.interpolate(R.uv, rast, R.fuv)
        col = dr.texture(R.tex, texc, filter_mode='linear')
        a = (rast[..., 3:] > 0).float()
        col = col * a
        out = dr.antialias(torch.cat([col, a], -1).contiguous(), rast, clip, R.f)
        return out[..., :3], out[..., 3]

    def losses(self, Rm, t, img, om, valid, x0, y0):
        """silhouette loss (multi-scale, person-occluded pixels ignored) and colour loss (rendered pixels)"""
        col, a = self.render(Rm, t, x0, y0)
        A = a[:, None]; O = om[:, None]; V = valid[:, None]
        lsil = 0
        for k in [1, 2, 4, 8]:
            if k > 1:
                Ak = Fn.avg_pool2d(A, k); Ok = Fn.avg_pool2d(O, k); Vk = (Fn.avg_pool2d(V, k) > 0.99).float()
            else:
                Ak, Ok, Vk = A, O, V
            lsil = lsil + (((Ak - Ok) ** 2) * Vk).sum((1, 2, 3)) / (Ok.sum((1, 2, 3)) + 1)
        w = (a * valid)[..., None]
        g = ((img * col * w).sum((1, 2)) / ((col * col * w).sum((1, 2)) + 1e-6)).detach().clamp(0.7, 1.5)
        lcol = ((col * g[:, None, None] - img).abs() * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) + 1)
        # appearance features robust to shading: blueness (B-G, the blue strap near the cap, the only strong
        # cue for the spin about the long axis) and darkness (the dark cap, cue for the end-over-end flip),
        # compared after a light blur so that 1-2 px misalignment is tolerated.
        cb = blur(col, 1.5); ib = blur(img, 1.5); w0 = w[..., 0]; nw = w0.sum((1, 2)) + 1
        blue = lambda x: x[..., 2] - x[..., 1]
        dark = lambda x: (0.35 - x.mean(-1)).clamp(min=0)
        lblue = ((blue(cb) - blue(ib)).abs() * w0).sum((1, 2)) / nw
        ldark = ((dark(cb * g[:, None, None]) - dark(ib)).abs() * w0).sum((1, 2)) / nw
        # strap overlap: soft-thresholded blue maps (strap ~1, cyan body ~0), mismatch normalised by the total strap
        # energy -> ~0 if both show no strap, ~1 if only one of them shows it or they are disjoint
        sb = lambda x: torch.sigmoid((blue(x) - 0.08) / 0.02)
        reg = valid * torch.clamp(om + a, max=1)
        br = sb(cb) * a; bi = sb(ib) * reg
        lstrap = (((br - bi) ** 2) * reg).sum((1, 2)) / ((br ** 2 * reg).sum((1, 2)) + (bi ** 2).sum((1, 2)) + 20.)
        return lsil, lcol, lblue, ldark, lstrap

    def unary(self, ls, lc, lb, ld, lst):
        return ls + self.wc * lc + self.wb * lb + self.wd * ld + self.ws * lst

    def optimize(self, R0, t0, img, om, valid, x0, y0, iters=40, lr_r=0.03, lr_t=0.01):
        B = R0.shape[0]
        w = torch.zeros(B, 3, device=dev, requires_grad=True)
        dt = torch.zeros(B, 3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
        for it in range(iters):
            Rm = R0 @ so3_exp(w)
            t = t0 + dt * t0[:, 2:3] / 3
            loss = self.unary(*self.losses(Rm, t, img, om, valid, x0, y0)).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Rm = R0 @ so3_exp(w); t = t0 + dt * t0[:, 2:3] / 3
            L = self.losses(Rm, t, img, om, valid, x0, y0)
        return Rm.detach(), t.detach(), torch.stack(L, -1)

    @torch.no_grad()
    def evaluate(self, Rm, t, img, om, valid, x0, y0):
        return torch.stack(self.losses(Rm, t, img, om, valid, x0, y0), -1)
