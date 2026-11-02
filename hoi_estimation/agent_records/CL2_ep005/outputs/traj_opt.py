"""Joint render-and-compare trajectory optimisation over all frames (crop-based, chunked)."""
import numpy as np
import torch
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
from hoi_lib import so3_exp


def gblur(x, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-((torch.arange(k, device=x.device) - k // 2) ** 2) / (2 * s * s))
    g = g / g.sum()
    x = Fn.conv2d(x[:, None], g[None, None, None, :], padding=(0, k // 2))
    x = Fn.conv2d(x, g[None, None, :, None], padding=(k // 2, 0))
    return x[:, 0]


def make_crops(Tc_init, obj_mask_fn, per_mask_fn, K, H, W, R, S=192, pad=0.25):
    """Square crop per frame around union of projected init pose and the object mask.
    Returns crop params (F,3)=(x0,y0,size) and cropped masks (F,S,S) object / valid (float16, cpu)."""
    import cv2
    F = len(Tc_init)
    crops = np.zeros((F, 3))
    om = np.zeros((F, S, S), np.float16)
    vm = np.zeros((F, S, S), np.float16)
    for s in range(0, F, 64):
        rm = R.render(Tc_init[s:s + 64], K, H, W, scale=0.25, need_rgb=False)["mask"].cpu().numpy()
        for j in range(len(rm)):
            i = s + j
            o = obj_mask_fn(i)
            p = per_mask_fn(i)
            ys, xs = np.where(rm[j])
            ys, xs = ys * 4, xs * 4
            yo, xo = np.where(o)
            if len(yo):
                ys, xs = np.concatenate([ys, yo]), np.concatenate([xs, xo])
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            c = max(y1 - y0, x1 - x0) * (1 + 2 * pad)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            crops[i] = [cx - c / 2, cy - c / 2, c]
            M = np.array([[S / c, 0, -(cx - c / 2) * S / c], [0, S / c, -(cy - c / 2) * S / c]])
            om[i] = cv2.warpAffine(o.astype(np.float32), M, (S, S), flags=cv2.INTER_AREA)
            valid = (~(p & ~o)).astype(np.float32)
            vm[i] = cv2.warpAffine(valid, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=1.0)
    return crops, om, vm


def crop_proj(K, crops, S, dev):
    """Per-frame OpenGL projection for the crop (B,4,4)."""
    B = len(crops)
    P = np.zeros((B, 4, 4), np.float32)
    n, fr = 0.05, 50.0
    for b, (x0, y0, c) in enumerate(crops):
        s = S / c
        fx, fy = K[0, 0] * s, K[1, 1] * s
        cx, cy = (K[0, 2] - x0 + 0.5) * s - 0.5, (K[1, 2] - y0 + 0.5) * s - 0.5
        P[b] = [[2 * fx / S, 0, 2 * (cx + 0.5) / S - 1, 0],
                [0, 2 * fy / S, 2 * (cy + 0.5) / S - 1, 0],
                [0, 0, (fr + n) / (fr - n), -2 * fr * n / (fr - n)],
                [0, 0, 1, 0]]
    return torch.tensor(P, device=dev)


def render_crop_sil(R, T, P, S):
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("bij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (S, S))
    sil = (rast[..., 3:4] > 0).float()
    sil = dr.antialias(sil, rast, clip, R.f)
    return sil[..., 0]


class TrajOpt:
    """Poses: T_i = [exp(w_n) R0_n | (xy0_n + dxy_n) * z0_n*exp(dlz_n)] with node n = node_of[i].
    Static groups share one node."""

    def __init__(self, R, K, Tc_init_nodes, node_of, crops, om, vm, S, dev="cuda"):
        self.R, self.K, self.S, self.dev = R, K, S, dev
        self.node_of = torch.tensor(node_of, device=dev, dtype=torch.long)
        T0 = torch.tensor(Tc_init_nodes, dtype=torch.float32, device=dev)
        self.R0 = T0[:, :3, :3]
        t0 = T0[:, :3, 3]
        self.xy0 = t0[:, :2] / t0[:, 2:]
        self.z0 = t0[:, 2:]
        N = len(T0)
        self.w = torch.zeros(N, 3, device=dev, requires_grad=True)
        self.dxy = torch.zeros(N, 2, device=dev, requires_grad=True)
        self.dlz = torch.zeros(N, 1, device=dev, requires_grad=True)
        self.P = crop_proj(K, crops, S, dev)
        self.om = torch.tensor(om, device=dev)
        self.vm = torch.tensor(vm, device=dev)
        self.F = len(node_of)
        self.crops = crops

    def node_poses(self):
        Rn = so3_exp(self.w) @ self.R0
        z = self.z0 * torch.exp(self.dlz)
        t = torch.cat([(self.xy0 + self.dxy) * z, z], 1)
        return Rn, t

    def frame_T(self, idx=None):
        Rn, t = self.node_poses()
        nid = self.node_of if idx is None else self.node_of[idx]
        T = torch.zeros(len(nid), 4, 4, device=self.dev)
        T[:, :3, :3] = Rn[nid]
        T[:, :3, 3] = t[nid]
        T[:, 3, 3] = 1
        return T

    def sil_loss(self, idx, frame_w):
        T = self.frame_T(idx)
        sil = render_crop_sil(self.R, T, self.P[idx], self.S)
        m = self.om[idx].float()
        v = self.vm[idx].float()
        loss = ((sil - m) ** 2 * v).mean((1, 2))
        for s in (4.0, 1.5):
            loss = loss + ((gblur(sil, s) - gblur(m, s)) ** 2 * v).mean((1, 2))
        return (loss * frame_w[idx]).sum(), sil.detach()

    def run(self, iters, frame_w, smooth_fn, lr=2e-3, chunk=96, log=None, edge=None, edge_w=None, reextract=50):
        """edge: dict(dts, pms, E, fn_obb, mu) enables the crease-edge chamfer term (weights edge_w per frame)."""
        opt = torch.optim.Adam([self.w, self.dxy, self.dlz], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.1)
        frame_w = torch.as_tensor(frame_w, dtype=torch.float32, device=self.dev)
        if edge is not None:
            E = edge["E"]
            Pe = crop_proj(self.K, self.crops, E, self.dev)
            edge_w = torch.as_tensor(edge_w, dtype=torch.float32, device=self.dev)
            gen = torch.Generator().manual_seed(0)
        for it in range(iters):
            if edge is not None and it % reextract == 0:
                pts, val = [], []
                for s in range(0, self.F, chunk):
                    idx = torch.arange(s, min(s + chunk, self.F), device=self.dev)
                    p, v = extract_edge_points(self.R, edge["fn_obb"], self.frame_T(idx).detach(), Pe[idx], E, gen=gen)
                    pts.append(p)
                    val.append(v)
                pts, val = torch.cat(pts), torch.cat(val)
            opt.zero_grad()
            tot, tote = 0.0, 0.0
            for s in range(0, self.F, chunk):
                idx = torch.arange(s, min(s + chunk, self.F), device=self.dev)
                l, _ = self.sil_loss(idx, frame_w)
                if edge is not None:
                    el, _ = edge_loss(self.frame_T(idx), Pe[idx], pts[idx], val[idx],
                                      edge["dts"][idx].float(), edge["pms"][idx].float(), E)
                    le = edge["mu"] * (el * edge_w[idx]).sum()
                    tote += le.item()
                    l = l + le
                l.backward()
                tot += l.item()
            Rn, t = self.node_poses()
            ls = smooth_fn(Rn, t)
            ls.backward()
            opt.step()
            sched.step()
            if log is not None and (it % 25 == 0 or it == iters - 1):
                log(f"  it {it} data {tot:.4f} (edge {tote:.4f}) smooth {ls.item():.4f}")

    @torch.no_grad()
    def ious(self, chunk=128):
        out = []
        for s in range(0, self.F, chunk):
            idx = torch.arange(s, min(s + chunk, self.F), device=self.dev)
            sil = render_crop_sil(self.R, self.frame_T(idx), self.P[idx], self.S) > 0.5
            m = self.om[idx] > 0.5
            v = self.vm[idx] > 0.5
            out.append((((sil & m & v).sum((1, 2)).float()) / ((sil | m) & v).sum((1, 2)).float().clamp_min(1)).cpu().numpy())
        return np.concatenate(out)


def make_edge_maps(crops, gray_fn, person_fn, E=320, trunc=10.0):
    """Per-frame distance transform (crop px at res E) to Canny edges of the contrast-equalised frame crop,
    plus a person mask (dilated) at the same res. Returns float16 arrays (F,E,E)."""
    import cv2
    F = len(crops)
    dts = np.zeros((F, E, E), np.float16)
    pms = np.zeros((F, E, E), np.uint8)
    for i in range(F):
        x0, y0, c = crops[i]
        M = np.array([[E / c, 0, -x0 * E / c], [0, E / c, -y0 * E / c]])
        g = cv2.warpAffine(gray_fn(i), M, (E, E), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        g = cv2.equalizeHist(g)
        ed = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.2), 40, 100)
        dts[i] = np.minimum(cv2.distanceTransform((ed == 0).astype(np.uint8), cv2.DIST_L2, 3), trunc)
        p = cv2.warpAffine(person_fn(i).astype(np.uint8), M, (E, E), flags=cv2.INTER_NEAREST)
        pms[i] = cv2.dilate(p, np.ones((7, 7))) > 0
    return dts, pms


@torch.no_grad()
def extract_edge_points(SM, fn_obb, T, P, E, n_pts=256, gen=None):
    """Internal crease edges of the box at pose T, as mesh-frame 3D points (B,n_pts,3) + validity (B,n_pts).
    fn_obb: (Fc,) label of each face = dominant OBB face direction (1..6)."""
    vh = torch.cat([SM.v, torch.ones_like(SM.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("bij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(SM.glctx, clip, SM.f, (E, E))
    tri = rast[..., 3].long() - 1
    ok = tri >= 0
    lab = torch.where(ok, fn_obb[tri.clamp_min(0)], torch.zeros_like(tri))
    e = torch.zeros_like(ok)
    e[:, :, 1:] |= (lab[:, :, 1:] != lab[:, :, :-1]) & ok[:, :, 1:] & ok[:, :, :-1]
    e[:, 1:, :] |= (lab[:, 1:, :] != lab[:, :-1, :]) & ok[:, 1:, :] & ok[:, :-1, :]
    pos, _ = dr.interpolate(SM.v[None].contiguous(), rast, SM.f)
    B = len(T)
    pts = torch.zeros(B, n_pts, 3, device=T.device)
    val = torch.zeros(B, n_pts, dtype=torch.bool, device=T.device)
    for b in range(B):
        idx = torch.nonzero(e[b].flatten()).squeeze(1)
        if len(idx) < 20:
            continue
        sel = idx[torch.randperm(len(idx), generator=gen, device="cpu").to(idx.device)[:n_pts]] if len(idx) > n_pts else idx
        pts[b, :len(sel)] = pos[b].reshape(-1, 3)[sel]
        val[b, :len(sel)] = True
    return pts, val


def edge_loss(T, P, pts, val, dts, pms, E):
    """Mean truncated DT at projected edge points (excluding points on the person)."""
    pc = torch.einsum("bij,bnj->bni", T[:, :3, :3], pts) + T[:, None, :3, 3]
    clip = torch.einsum("bij,bnj->bni", P, torch.cat([pc, torch.ones_like(pc[..., :1])], -1))
    ndc = clip[..., :2] / clip[..., 3:4]  # [-1,1] over the crop, y down
    grid = ndc[:, :, None, :]
    dv = Fn.grid_sample(dts[:, None], grid, mode="bilinear", align_corners=False, padding_mode="border")[:, 0, :, 0]
    pv = Fn.grid_sample(pms[:, None], grid.detach(), mode="nearest", align_corners=False)[:, 0, :, 0]
    w = val.float() * (1 - pv)
    return (dv * w).sum(1) / w.sum(1).clamp_min(1.0), w.sum(1)
