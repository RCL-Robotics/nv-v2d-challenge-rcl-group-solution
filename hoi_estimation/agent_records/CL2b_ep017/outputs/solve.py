#!/usr/bin/env python
"""Trajectory-level re-estimation of the traffic-cone 6-DoF poses for one HOI video.

Usage: python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

Pipeline (see REPORT.md for the reasoning):
 1. phase segmentation from the SAM2 object mask (static on floor / carried / static on table);
 2. static phases: ONE shared pose per phase, fitted by render-and-compare (silhouette IoU against
    the object mask, person pixels ignored) pooled over the whole phase; depth fixed to the baseline;
 3. carried phase: per-frame silhouette fits on a grid of twist angles about the cone axis
    (the cone is ~4-fold symmetric, so the silhouette only measures twist modulo 90 deg);
 4. an HMM over the absolute twist decodes the trajectory jointly: data term = per-frame
    silhouette profile, transition = small rotation per frame with the person's torso yaw change as
    the expected twist change (object held in the hands co-rotates with the body), anchors = the two
    static-phase fits. This resolves the 90-deg branch per frame and in the final static phase;
 5. swing / translation of the carried phase from the per-frame fits, robustly smoothed; frames without
    visible object are bridged by a hand-attached model (offset to the wrist midpoint);
 6. depth re-anchoring: camera-frame positions are scaled along their viewing rays so that the
    median camera depth per phase equals the baseline's.
Deterministic up to GPU floating-point atomics in nvdiffrast's antialias gradients: fixed seeds,
deterministic grid initialisations, no sampling. Needs only the inputs (no /pipeline, no network).
"""
import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation as Rot

# ============================================================================= library (data, rendering, fitting)


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


# ============================================================================= solver

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------------------------- constants
STATIC_IOU = 0.85        # object-mask IoU (person pixels ignored) w.r.t. the phase reference mask
CONTACT_OFF = 0.5        # contact probability below which the table phase may start
MIN_AREA = 150           # full-res object-mask pixels needed for a frame to give silhouette evidence
TWIST_GRID = np.arange(0, 90, 5.0)
HMM_STATES = np.arange(-360, 360, 2.5)
SIG_IOU = 0.01           # IoU differences of ~0.01 are treated as 1 sigma in the twist data term
SIG_STEP = 6.0           # deg/frame std of twist change w.r.t. the body-yaw prediction
BRANCH_SWITCH_P = 0.7    # table-phase 90-deg branch differs from the baseline's only if its posterior exceeds this


def log(*a):
    print(*a, flush=True)


def wrap90(a):
    return (a + 45.0) % 90.0 - 45.0


# ----------------------------------------------------------------------------- phases
def segment_phases(ep):
    om, pm = ep.obj_mask, ep.person_mask
    F = ep.F

    def iou_to(ref, t):
        v = ~pm[t] | om[t]
        u = ((om[t] | ref) & v).sum()
        return (om[t] & ref & v).sum() / max(u, 1)

    refA = np.median(om[:15].astype(np.uint8), 0) > 0
    a_end = 0
    while a_end + 1 < F and iou_to(refA, a_end + 1) >= STATIC_IOU:
        a_end += 1
    refB = np.median(om[-15:].astype(np.uint8), 0) > 0
    b_start = F - 1
    while b_start - 1 > a_end and iou_to(refB, b_start - 1) >= STATIC_IOU and \
            ep.contact[b_start - 1].max() < CONTACT_OFF:
        b_start -= 1
    return a_end, b_start


# ----------------------------------------------------------------------------- static fits
def fit_static(ep, rd, frames, offsets, fix_twist=False):
    """One shared pose for all `frames`. Returns list of (iou, Tc) for each initial twist offset."""
    Tm = ep.Tc_base[frames[len(frames) // 2]].copy()
    Tm[:3, 3] = np.median(ep.Tc_base[frames][:, :3, 3], 0)
    tg = MaskTargets(ep, frames, np.repeat(Tm[None], len(frames), 0), size=192)
    out = []
    for o in offsets:
        T0c = torch.tensor(axis_rot(Tm, o)[None], device="cuda", dtype=torch.float32)
        w = torch.zeros(1, 3, device="cuda", requires_grad=True)
        t = torch.zeros(1, 3, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([w, t], lr=0.01)
        for it in range(150):
            T = compose(T0c, w, t, 0.0, fix_twist).expand(len(frames), 4, 4)
            iou = soft_iou(render_masks(rd, T, ep.K, tg.crops, tg.size), tg)
            loss = 1 - iou.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it == 100:
                for g in opt.param_groups:
                    g["lr"] = 0.003
        with torch.no_grad():
            T = compose(T0c, w, t, 0.0, fix_twist)
            iou = soft_iou(render_masks(rd, T.expand(len(frames), 4, 4), ep.K, tg.crops, tg.size), tg).mean()
        out.append((float(iou), T[0].cpu().numpy().astype(np.float64)))
    return out, Tm


# ----------------------------------------------------------------------------- helpers
def body_yaw(ep):
    """Torso facing yaw (deg, about world +y which points down), unwrapped and lightly smoothed."""
    K = ep.kp
    up = np.array([0, -1.0, 0])
    fc = np.cross(up, (K[:, 5] - K[:, 6]) + (K[:, 9] - K[:, 10]))
    yy = np.degrees(np.unwrap(np.arctan2(fc[:, 0], fc[:, 2])))
    return uniform_filter1d(yy, 5)


def periodic_interp(x, tws, vals):
    m = tws % 90.0
    o = np.argsort(m)
    xs = np.concatenate([m[o] - 90, m[o], m[o] + 90])
    ys = np.tile(vals[o], 3)
    return np.interp(x % 90.0, xs, ys)


def set_twist(T_world, R0, target):
    """Rotate world pose about the cone axis so that its twist w.r.t. R0 equals target (deg)."""
    cur = twist_swing(R0.T @ T_world[:3, :3])[0]
    d = (target - cur + 180) % 360 - 180
    return axis_rot(T_world, d)


def depth_anchor(Tc, Tc_ref, idx):
    """Scale camera-frame positions along their rays so median z over idx equals the reference median z."""
    Tc = Tc.copy()
    s = np.median(Tc_ref[idx, 2, 3]) / np.median(Tc[idx, 2, 3])
    Tc[idx, :3, 3] *= s
    return Tc, s


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--debug", default="")
    args = ap.parse_args()
    t_start = time.time()
    ep = Episode(args.inputs)
    rd = Renderer(ep.mesh)
    F = ep.F
    R0 = ep.T_base[0, :3, :3]          # twist reference: the baseline orientation at frame 0
    Twc, Tcw = ep.T_world_cam, ep.T_cam_world

    # 1. phases ------------------------------------------------------------------------------
    a_end, b_start = segment_phases(ep)
    log(f"phases: static-floor [0,{a_end}]  carried [{a_end + 1},{b_start - 1}]  static-table [{b_start},{F - 1}]")
    area = ep.obj_mask.reshape(F, -1).sum(1)

    # 2. static phases -------------------------------------------------------------------------
    framesA = list(range(0, a_end + 1, 3))
    framesB = list(range(b_start, F, 3))
    resA, TmA = fit_static(ep, rd, framesA, np.arange(-40, 41, 10))
    iouA, TA = max(resA, key=lambda r: r[0])
    # floor phase keeps the baseline's 90-deg branch (no independent evidence for another branch):
    twA_base = twist_swing(R0.T @ (Twc @ TmA)[:3, :3])[0]
    twA = twist_swing(R0.T @ (Twc @ TA)[:3, :3])[0]
    twA = twA_base + wrap90(twA - twA_base)
    TA = Tcw @ set_twist(Twc @ TA, R0, twA)
    log(f"static floor: IoU {iouA:.4f}, twist {twA:.1f} (baseline {twA_base:.1f})")
    # table phase: free-twist pooled fit from several initial twists; the silhouette only fixes twist mod 90
    resB, TmB = fit_static(ep, rd, framesB, np.arange(-40, 41, 10))
    iouB, TB = max(resB, key=lambda r: r[0])
    twB_mod = twist_swing(R0.T @ (Twc @ TB)[:3, :3])[0]
    twB_all = np.array([twist_swing(R0.T @ (Twc @ T)[:3, :3])[0] for _, T in resB])
    log(f"static table: IoU {iouB:.4f}, twist mod 90 = {twB_mod % 90:.1f} "
        f"(all inits: {np.round(twB_all % 90, 1).tolist()})")

    # 3. carried phase: per-frame twist profiles ------------------------------------------------
    carry = list(range(a_end + 1, b_start))
    # smoothed baseline depth (the baseline's per-frame z is noisy during the carry)
    zb = ep.Tc_base[:, 2, 3]
    z_s = zb.copy()
    z_s[carry] = gaussian_filter1d(zb[carry], 3.0, mode="nearest")
    vis = [t for t in carry if area[t] >= MIN_AREA]
    Tinit = []
    for t in vis:
        T = ep.Tc_base[t].copy()
        T[:3, 3] *= z_s[t] / T[2, 3]
        Tinit.append(T)
    Tinit = np.array(Tinit)
    tg = MaskTargets(ep, vis, Tinit, size=128)
    T0, idx = [], []
    for k, t in enumerate(vis):
        tw = twist_swing(R0.T @ (Twc @ Tinit[k])[:3, :3])[0]
        for g in TWIST_GRID:
            T0.append(axis_rot(Tinit[k], g - tw)); idx.append(k)
    Tf, iouf = fit_frames(rd, ep, tg, np.array(T0), np.array(idx), iters=80, fix_twist=True, chunk=128)
    G = len(TWIST_GRID)
    Tf = Tf.reshape(len(vis), G, 4, 4)
    iouf = iouf.reshape(len(vis), G)
    tw_f = np.array([[twist_swing(R0.T @ (Twc @ Tf[k, j])[:3, :3])[0] for j in range(G)] for k in range(len(vis))])

    # 4. HMM over absolute twist -----------------------------------------------------------------
    X = HMM_STATES
    seq = list(range(a_end, b_start + 1))           # from last floor frame to first table frame
    n = len(seq)
    unary = np.zeros((n, len(X)))
    vis_k = {t: k for k, t in enumerate(vis)}
    for s, t in enumerate(seq):
        if t in vis_k:
            k = vis_k[t]
            prof = periodic_interp(X, tw_f[k], iouf[k])
            unary[s] = -0.5 * ((prof - prof.max()) / SIG_IOU) ** 2
    unary[0] += -0.5 * ((X - twA) / 3.0) ** 2                       # floor anchor (absolute)
    unary[-1] += -0.5 * (wrap90(X - twB_mod) / 4.0) ** 2           # table anchor (mod 90 only)
    yaw = body_yaw(ep)
    dpred = -np.diff(yaw, prepend=yaw[0])        # twist about "up" = - yaw change about +y (down)
    D = X[None, :] - X[:, None]

    def trans(t):
        return -0.5 * ((D - dpred[t]) / SIG_STEP) ** 2

    # forward pass (sum-product) for the posterior of the table branch
    alpha = unary[0] - unary[0].max()
    for s in range(1, n):
        m = alpha[:, None] + trans(seq[s])
        mx = m.max(0)
        alpha = mx + np.log(np.exp(m - mx).sum(0)) + unary[s]
        alpha -= alpha.max()
    post = np.exp(alpha); post /= post.sum()
    branches = {}
    for c in np.arange(0, 360, 90.0):
        ang = (twB_mod + c) % 360
        sel = np.abs(((X - ang) + 180) % 360 - 180) < 45
        branches[(ang + 180) % 360 - 180] = post[sel].sum()
    log("table-phase twist branch posterior: " + ", ".join(f"{k:.1f}: {v:.3f}" for k, v in sorted(branches.items())))
    base_twB = twist_swing(R0.T @ ep.T_base[F - 1, :3, :3])[0]
    log(f"(baseline table twist {base_twB:.1f})")
    # table-phase branch: keep the baseline's branch unless the HMM clearly favours another one
    best_b = max(branches, key=branches.get)
    base_b = min(branches, key=lambda k: abs(((k - base_twB) + 180) % 360 - 180))
    chosen = best_b if branches[best_b] > BRANCH_SWITCH_P else base_b
    log(f"table branch: HMM-best {best_b:.1f} (p={branches[best_b]:.3f}), baseline-branch {base_b:.1f} "
        f"(p={branches[base_b]:.3f}) -> using {chosen:.1f}")
    unary[-1] += -0.5 * ((((X - chosen) + 180) % 360 - 180) / 3.0) ** 2
    # Viterbi (max-product) for the MAP twist path
    delta = unary[0].copy(); BP = []
    for s in range(1, n):
        m = delta[:, None] + trans(seq[s])
        BP.append(m.argmax(0)); delta = m.max(0) + unary[s]
    path = [int(delta.argmax())]
    for b in BP[::-1]:
        path.append(int(b[path[-1]]))
    tw_path = X[np.array(path[::-1])]
    tw_path_s = gaussian_filter1d(tw_path, 1.5, mode="nearest")
    twB = tw_path[-1]
    log(f"decoded table twist {twB:.1f}")

    # 5. assemble poses --------------------------------------------------------------------------
    Tw = np.zeros((F, 4, 4))
    TA_w = Twc @ TA
    TB_w = set_twist(Twc @ TB, R0, twB)
    Tw[:a_end + 1] = TA_w
    Tw[b_start:] = TB_w
    # per-frame carried poses at the decoded twist
    good = {}
    for s, t in enumerate(seq[1:-1], start=1):
        if t not in vis_k:
            continue
        k = vis_k[t]
        j = int(np.argmin(np.abs(wrap90(tw_f[k] - tw_path_s[s]))))
        if iouf[k, j] < 0.6:
            continue
        good[t] = set_twist(Twc @ Tf[k, j], R0, tw_path_s[s])
    # robust smoothing of the cone axis direction and of the position (camera frame)
    ts = np.array(sorted(good))
    cl = np.array(list(AXIS_C[:2]) + [0.09])      # cone centre (mesh coords); positions below are centres
    ctr = lambda T: T[:3, :3] @ cl + T[:3, 3]      # noqa: E731
    axis_c = np.array([(Tcw @ good[t])[:3, 2] for t in ts])
    pos_c = np.array([ctr(Tcw @ good[t]) for t in ts])
    # anchor points at the phase boundaries so the carry connects to the static poses
    ts_all = np.concatenate([[a_end], ts, [b_start]])
    axis_all = np.concatenate([[TA[:3, 2]], axis_c, [(Tcw @ TB_w)[:3, 2]]])
    pos_all = np.concatenate([[ctr(TA)], pos_c, [ctr(Tcw @ TB_w)]])
    # bridge: hand-attached model for long gaps (object follows the wrist midpoint)
    wm = 0.5 * (ep.kp[:, 41] + ep.kp[:, 62])
    wm_c = (Tcw[:3, :3] @ wm.T).T + Tcw[:3, 3]
    full_t = np.arange(a_end, b_start + 1)
    off = pos_all - wm_c[ts_all]
    off_i = np.stack([np.interp(full_t, ts_all, off[:, d]) for d in range(3)], 1)
    pos_i = wm_c[full_t] + off_i
    # for observed frames use the measured position; smooth everything
    meas = np.isin(full_t, ts_all)
    pos_i[meas] = pos_all[np.searchsorted(ts_all, full_t[meas])]
    for d in range(3):
        pos_i[1:-1, d] = gaussian_filter1d(pos_i[:, d], 2.0, mode="nearest")[1:-1]
    # axis: interpolate on the sphere (slerp of swing rotations), then smooth
    ax_i = np.stack([np.interp(full_t, ts_all, axis_all[:, d]) for d in range(3)], 1)
    for d in range(3):
        ax_i[1:-1, d] = gaussian_filter1d(ax_i[:, d], 2.0, mode="nearest")[1:-1]
    ax_i /= np.linalg.norm(ax_i, axis=1, keepdims=True)
    # compose rotation: minimal rotation taking the frame-0 cone axis onto ax_i, then set twist
    Rc0 = (Tcw @ TA_w)[:3, :3]
    for s, t in enumerate(full_t):
        if t <= a_end or t >= b_start:
            continue
        a0 = Rc0[:, 2]
        v = np.cross(a0, ax_i[s]); c = float(a0 @ ax_i[s])
        if np.linalg.norm(v) < 1e-9:
            Rs = np.eye(3)
        else:
            Rs = Rot.from_rotvec(v / np.linalg.norm(v) * np.arccos(np.clip(c, -1, 1))).as_matrix()
        Tc = np.eye(4); Tc[:3, :3] = Rs @ Rc0
        Tc[:3, 3] = pos_i[s] - Tc[:3, :3] @ cl     # place the cone centre at pos_i
        Tw_t = Twc @ Tc
        Tw[t] = set_twist(Tw_t, R0, tw_path_s[s])

    # 6. depth re-anchoring per phase --------------------------------------------------------------
    Tc_all = Tcw[None] @ Tw
    scales = {}
    for name, idx in [("floor", np.arange(0, a_end + 1)), ("carried", np.arange(a_end + 1, b_start)),
                      ("table", np.arange(b_start, F))]:
        Tc_all, s = depth_anchor(Tc_all, ep.Tc_base, idx)
        scales[name] = s
    log("depth re-anchoring scale factors: " + ", ".join(f"{k}: {v:.4f}" for k, v in scales.items()))
    Tw = Twc[None] @ Tc_all
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    if args.debug:
        np.savez(args.debug, a_end=a_end, b_start=b_start, tw_path=tw_path, seq=np.array(seq), vis=np.array(vis),
                 iouf=iouf, tw_f=tw_f, branches=np.array(list(branches.items())), twA=twA, twB=twB,
                 iouA=iouA, iouB=iouB)
    log(f"done in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
