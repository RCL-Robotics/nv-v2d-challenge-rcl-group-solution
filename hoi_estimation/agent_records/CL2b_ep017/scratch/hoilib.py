"""Shared helpers (data loading, rendering, scoring) for the traffic-cone trajectory solver.

Imported by solve.py; kept in /work/outputs next to it.
"""
import json
import os

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot


# ----------------------------------------------------------------------------- data
class Episode:
    def __init__(self, inputs):
        self.inputs = inputs
        cam = json.load(open(os.path.join(inputs, "camera.json")))
        self.K = np.array(cam["K"], dtype=np.float64)
        self.W, self.H = int(cam["width"]), int(cam["height"])
        self.T_world_cam = np.array(cam["T_world_cam"], dtype=np.float64)
        self.T_cam_world = np.linalg.inv(self.T_world_cam)
        gp = json.load(open(os.path.join(inputs, "ground_plane.json")))
        self.plane = np.array(gp["plane"], dtype=np.float64)
        rec = np.load(os.path.join(inputs, "reconstruction.npz"))
        self.T_base = rec["T_world_mesh"].astype(np.float64)
        self.F = len(self.T_base)
        self.Tc_base = self.T_cam_world[None] @ self.T_base
        self.kp = rec["human_keypoints_world"].astype(np.float64)
        self.contact = rec["contact_prob_right_left"].astype(np.float64)
        self.T_fp = rec["T_cam_mesh_foundationpose"].astype(np.float64)
        mh = h5py.File(os.path.join(inputs, "masks.h5"), "r")
        self.obj_mask = mh["object"][:]
        self.person_mask = mh["person"][:]
        mesh = trimesh.load(os.path.join(inputs, "object_mesh.glb"), force="scene", process=False).to_geometry()
        self.mesh = mesh
        self._frames = None

    def frames(self):
        if self._frames is None:
            cap = cv2.VideoCapture(os.path.join(self.inputs, "video.mp4"))
            out = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                out.append(f[:, :, ::-1].copy())  # RGB
            self._frames = np.stack(out)
        return self._frames


# ----------------------------------------------------------------------------- geometry
def twist_swing(Rrel):
    """Decompose rotation Rrel (object frame) into twist about object z (deg) and swing magnitude (deg)."""
    q = Rot.from_matrix(Rrel).as_quat()
    tw = np.array([0, 0, q[2], q[3]])
    tw /= np.linalg.norm(tw)
    ang = 2 * np.degrees(np.arctan2(tw[2], tw[3]))
    sw = Rot.from_quat(q) * Rot.from_quat(tw).inv()
    return (ang + 180) % 360 - 180, np.degrees(sw.magnitude())


def Rz(deg):
    return Rot.from_euler("z", deg, degrees=True).as_matrix()


def proj(K, p):
    uv = (K @ p.T).T
    return uv[:, :2] / uv[:, 2:3]


# ----------------------------------------------------------------------------- renderer
class Renderer:
    """Batched nvdiffrast renderer of the textured mesh into a square crop."""

    def __init__(self, mesh, device="cuda", axis_center=None):
        self.device = device
        self.ctx = dr.RasterizeCudaContext(device=device)
        v = np.asarray(mesh.vertices, dtype=np.float32)
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces, dtype=np.int32), device=device)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()

    def render(self, Tc, K, crop, size, color=False):
        """Tc: (B,4,4) camera-from-mesh. crop=(x0,y0,s) in full-res pixels, output size x size.
        Returns mask (B,S,S) float [0,1] (antialiased) and optionally rgb (B,S,S,3)."""
        B = Tc.shape[0]
        x0, y0, s = crop
        sc = size / s
        fx, fy = K[0, 0] * sc, K[1, 1] * sc
        cx, cy = (K[0, 2] - x0) * sc, (K[1, 2] - y0) * sc
        n, fz = 0.05, 20.0
        # OpenCV camera -> clip space (y down in image => flip)
        P = torch.tensor([[2 * fx / size, 0, 2 * cx / size - 1, 0],
                          [0, 2 * fy / size, 2 * cy / size - 1, 0],
                          [0, 0, (fz + n) / (fz - n), -2 * fz * n / (fz - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=self.device)
        Tc = torch.as_tensor(Tc, dtype=torch.float32, device=self.device)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        pc = torch.einsum("bij,vj->bvi", Tc, vh)
        clip = torch.einsum("ij,bvj->bvi", P, pc).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (size, size))
        cov = (rast[..., 3:] > 0).float()
        m = dr.antialias(cov, rast, clip, self.f)[..., 0]
        if not color:
            return m, None
        texc, _ = dr.interpolate(self.uv[None], rast, self.f)
        rgb = dr.texture(self.tex, texc, filter_mode="linear")
        rgb = rgb * cov
        return m, rgb


def crop_box(center, half):
    c = np.round(center).astype(int)
    return (int(c[0] - half), int(c[1] - half), int(2 * half))


def cut(img, crop, size, interp=cv2.INTER_AREA):
    """Cut crop (x0,y0,s) out of full image (pad with 0) and resize to size."""
    x0, y0, s = crop
    H, W = img.shape[:2]
    out = np.zeros((s, s) + img.shape[2:], dtype=img.dtype)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + s), min(H, y0 + s)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    if out.dtype == bool:
        out = out.astype(np.uint8)
    return cv2.resize(out, (size, size), interpolation=interp)


# ----------------------------------------------------------------------------- differentiable fitting
AXIS_C = np.array([0.0026, -0.003, 0.0])  # cone axis centre in mesh-file coords (OBB centre, from oriented_bounds)


def axis_rot(T, deg):
    """Rotate pose T (cam/world-from-mesh) by deg about the cone's own axis (mesh z through AXIS_C)."""
    A = np.eye(4); A[:3, 3] = AXIS_C
    B = np.eye(4); B[:3, :3] = Rz(deg)
    C = np.eye(4); C[:3, 3] = -AXIS_C
    return T @ A @ B @ C


def so3_exp(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2]; Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


class MaskTargets:
    """Per-frame crops of object/person masks at fixed size, stored on GPU."""

    def __init__(self, ep, frames, Tc_init, half=None, size=160):
        self.frames = list(frames)
        self.size = size
        crops, om, val = [], [], []
        for k, i in enumerate(self.frames):
            ctr = proj(ep.K, Tc_init[k][None, :3, 3])[0]
            z = Tc_init[k][2, 3]
            h = half if half is not None else max(48, int(0.33 * ep.K[0, 0] / z))
            crop = crop_box(ctr, h)
            o = cut(ep.obj_mask[i].astype(np.uint8) * 255, crop, size) / 255.0
            p = cut(ep.person_mask[i].astype(np.uint8) * 255, crop, size) / 255.0
            crops.append(crop); om.append(o); val.append(1 - p * (1 - o))
        self.crops = crops
        self.om = torch.tensor(np.stack(om), device="cuda", dtype=torch.float32)
        self.valid = torch.tensor(np.stack(val), device="cuda", dtype=torch.float32)
        self.area = self.om.sum((1, 2))


def render_masks(rd, Tc, K, crops, size):
    """Differentiable render of B poses Tc (torch (B,4,4)) each with its own crop."""
    B = Tc.shape[0]
    P = []
    for (x0, y0, s) in crops:
        sc = size / s
        fx, fy = K[0, 0] * sc, K[1, 1] * sc
        cx, cy = (K[0, 2] - x0) * sc, (K[1, 2] - y0) * sc
        n, fz = 0.05, 20.0
        P.append([[2 * fx / size, 0, 2 * cx / size - 1, 0],
                  [0, 2 * fy / size, 2 * cy / size - 1, 0],
                  [0, 0, (fz + n) / (fz - n), -2 * fz * n / (fz - n)],
                  [0, 0, 1, 0]])
    P = torch.tensor(np.array(P), dtype=torch.float32, device="cuda")
    vh = torch.cat([rd.v, torch.ones_like(rd.v[:, :1])], 1)
    M = P @ Tc
    clip = torch.einsum("bij,vj->bvi", M, vh).contiguous()
    rast, _ = dr.rasterize(rd.ctx, clip, rd.f, (size, size))
    cov = (rast[..., 3:] > 0).float()
    return dr.antialias(cov, rast, clip, rd.f)[..., 0]


def soft_iou(m, tg, idx=None):
    om = tg.om if idx is None else tg.om[idx]
    va = tg.valid if idx is None else tg.valid[idx]
    inter = (m * om * va).sum((1, 2))
    uni = ((m + om - m * om) * va).sum((1, 2))
    return inter / uni.clamp_min(1.0)


def compose(T0, w, t, zscale=0.0, fix_twist=False):
    """Apply rotation exp(w) about the cone centre (in camera frame) and translation t to poses T0 (B,4,4).
    fix_twist: remove the component of w along the cone axis (twist is then held at its initial value)."""
    if fix_twist:
        a = T0[:, :3, 2]
        w = w - (w * a).sum(-1, keepdim=True) * a
    R = so3_exp(w)
    cl = torch.tensor(list(AXIS_C[:2]) + [0.09, 1.0], device=T0.device, dtype=T0.dtype)
    cc = (T0 @ cl)[:, :3]
    Rn = R @ T0[:, :3, :3]
    tm = torch.tensor([1.0, 1.0, zscale], device=T0.device, dtype=T0.dtype)
    tn = (R @ (T0[:, :3, 3] - cc)[..., None])[..., 0] + cc + t * tm
    top = torch.cat([Rn, tn[..., None]], 2)
    bot = torch.tensor([0, 0, 0, 1.0], device=T0.device, dtype=T0.dtype).expand(T0.shape[0], 1, 4)
    return torch.cat([top, bot], 1)


def fit_frames(rd, ep, tg, T0, idx, iters=120, lr=0.01, zscale=0.0, chunk=96, fix_twist=False):
    """Independent per-hypothesis silhouette fit. T0 (N,4,4) numpy init, idx (N,) index into tg frames.
    Returns refined poses (N,4,4) and IoU (N,)."""
    outT, outI = [], []
    for s in range(0, len(T0), chunk):
        T0c = torch.tensor(T0[s:s + chunk], device="cuda", dtype=torch.float32)
        ic = idx[s:s + chunk]
        crops = [tg.crops[k] for k in ic]
        n = len(T0c)
        w = torch.zeros(n, 3, device="cuda", requires_grad=True)
        t = torch.zeros(n, 3, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([w, t], lr=lr)
        for it in range(iters):
            T = compose(T0c, w, t, zscale, fix_twist)
            m = render_masks(rd, T, ep.K, crops, tg.size)
            iou = soft_iou(m, tg, ic)
            loss = (1 - iou).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if it == int(iters * 0.7):
                for g in opt.param_groups:
                    g["lr"] = lr * 0.3
        with torch.no_grad():
            T = compose(T0c, w, t, zscale, fix_twist)
            iou = soft_iou(render_masks(rd, T, ep.K, crops, tg.size), tg, ic)
        outT.append(T.cpu().numpy().astype(np.float64)); outI.append(iou.cpu().numpy())
    return np.concatenate(outT), np.concatenate(outI)
