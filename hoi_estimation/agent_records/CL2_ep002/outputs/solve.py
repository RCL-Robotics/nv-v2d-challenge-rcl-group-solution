#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF pose of the blue easter basket (episode 2).

Pipeline (see REPORT.md for the reasoning and the validation):
  1. phases (static on table / carried / static on floor / carried+handover / static on floor, occluded)
  2. per-frame silhouette fit (nvdiffrast, person pixels ignored) starting from the baseline -> image position + tilt
  3. yaw evidence: the video is unwrapped onto the bucket's cylindrical surface for every frame and correlated with
     the (polarity-inverted) mesh texture (45 deg periodic bunny pattern) + physical bail-lug cue + weak baseline vote
  4. Viterbi over absolute world yaw for the whole episode (static phases: constant yaw; carried: <= 6 deg/frame)
  5. final rotations: static phases upright on the floor normal; carried: silhouette tilt (smoothed) + decoded yaw
  6. final silhouette fit of image position with temporal smoothing; depth from the (smoothed) baseline;
     occluded carry: object attached to the holding hand; final occluded rest: one constant pose
  7. per-phase re-anchoring of the camera depth to the baseline median (along viewing rays)
Deterministic: fixed seeds, no sampling.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import cv2
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import (load_inputs, load_mesh, decimated_mesh, SilRenderer, so3_exp, load_mask_crops, mask_stats,
                    crop_K, fit_lm, CovRenderer, rot_angle_deg)
import nvdiffrast.torch as dr

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

# ------------------------------------------------------------------------------------------------------------------
# Constants
# Phase boundaries (inclusive), verified by eye on the video and by the object-mask centroid / bbox (static phases:
# bbox constant to +-1 px). S1: basket on the round table; C1: carried with one hand to the floor; S2: resting on
# the floor; C2: picked up, hand-over, carried (visible until ~742, then hidden in front of the person's body) and put
# down on the floor at ~846; S3: resting on the floor, occluded by the person's legs (only 847-867 partly visible).
PHASES = [("S1_table_static", 0, 124, "static"),
          ("C1_carry", 125, 213, "carry"),
          ("S2_floor_static", 214, 436, "static"),
          ("C2_carry_handover", 437, 846, "carry"),
          ("S3_floor_static_occluded", 847, 979, "static")]
# Within C2 the object mask disappears at 743 (object hidden in front of the body) until it is put down.
OCCLUDED_CARRY = (743, 846)
# Bucket geometry in mesh coordinates: symmetry axis is mesh +y through (0.004, *, 0.003) (circle fits of the wall);
# PC is a point on the axis near the middle of the bucket used as the translation anchor.
AXC = np.array([0.004, 0.0, 0.003])
PC = np.array([0.004, 0.011, 0.003])
BODY_R_BOTTOM, BODY_R_TOP, Y_BOTTOM, Y_TOP = 0.089, 0.114, -0.0666, 0.089
# The wire bail is attached to two lugs on the rim at mesh azimuth atan2(z,x) = -60 / +120 deg (mesh analysis).
LUG_AZIMUTH_DEG = 120.0
# Frame ranges in which the bail lugs / wire loops are seen at the left AND right silhouette sides (verified by eye
# on zoomed crops: frames 0-100 wire loop at the left edge; 350 lug at the left edge; 555-700 wire loops on both
# sides). There the lug axis must be perpendicular to the viewing ray (yaw known modulo 180 deg).
LUG_SIDE_RANGES = [(0, 124), (214, 436), (545, 742)]
# Yaw decoding parameters
NB = 120                  # yaw bins (3 deg)
W_LUG = 3.0               # weight of the lug-side cost (cos^2 of the angle lug-axis / viewing ray)
W_BASE = 0.3              # weight of the baseline yaw vote (only where the baseline silhouette IoU is good)
MAXD = 2                  # max yaw change per frame while carried (bins)
W_TRANS = 0.3             # quadratic transition cost per bin^2
STATIC_TOTAL_W = 10.0     # total data weight of one static phase (frames are near-identical images)
TEMP = 3.0                # posterior temperature: per-frame evidence is strongly correlated between frames
RES = 128                 # silhouette crop resolution
# ------------------------------------------------------------------------------------------------------------------


def wyaw(R):
    """world yaw of the mesh x axis (deg), R (...,3,3) world rotation."""
    x = R[..., :, 0]
    return np.degrees(np.arctan2(x[..., 2], x[..., 0]))


def yawrot(a):
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    M[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    M[:3, 3] = AXC - M[:3, :3] @ AXC
    return M


def rot_from_axis_yaw(axis, yaw_deg):
    """world rotation with mesh +y along `axis` (unit, world) and world yaw of mesh x approx `yaw_deg`."""
    th = -np.radians(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R0 = np.diag([1.0, -1.0, -1.0])           # mesh +y -> world -y (up), mesh x -> world x
    R = Ry @ R0                               # upright, yaw = yaw_deg
    up = np.array([0.0, -1.0, 0.0])
    v = np.cross(up, axis)
    sn, cs = np.linalg.norm(v), float(up @ axis)
    if sn < 1e-9:
        return R
    k = v / sn
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    Ra = np.eye(3) + sn * Kx + (1 - cs) * Kx @ Kx
    return Ra @ R


def circ_smooth_deg(y, sigma):
    if sigma <= 0 or len(y) < 3:
        return y
    from scipy.ndimage import gaussian_filter1d
    r = np.radians(y)
    c = gaussian_filter1d(np.cos(r), sigma, mode="nearest")
    s = gaussian_filter1d(np.sin(r), sigma, mode="nearest")
    return np.degrees(np.arctan2(s, c))


def smooth(x, sigma):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(x, sigma, axis=0, mode="nearest")


# ------------------------------------------------------------------------------------------------------------------
# texture unwrapping onto the bucket wall
class Unwrapper:
    NPHI, NH, HB = 120, 12, (-0.055, 0.08)

    def __init__(self, V, F, UV, TEX, K):
        self.K = K
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        fc = V[F].mean(1)
        rad = fc[:, [0, 2]] - AXC[[0, 2]]
        r = np.linalg.norm(rad, axis=1)
        rad /= r[:, None] + 1e-12
        radial = fn[:, 0] * rad[:, 0] + fn[:, 2] * rad[:, 1]
        y = fc[:, 1]
        rb = BODY_R_BOTTOM + (y - Y_BOTTOM) / (Y_TOP - Y_BOTTOM) * (BODY_R_TOP - BODY_R_BOTTOM)
        outer = (np.abs(r - rb) < 0.006) & (y > -0.06) & (y < 0.085) & (radial > 0.5)
        Vu = V[F].reshape(-1, 3)
        Fu = np.arange(len(Vu)).reshape(-1, 3)
        phi = np.arctan2(Vu[:, 2] - AXC[2], Vu[:, 0] - AXC[0])
        attr = np.stack([np.cos(phi), np.sin(phi), Vu[:, 1], np.repeat(outer.astype(np.float32), 3)], 1)
        dev = "cuda"
        self.glctx = dr.RasterizeCudaContext()
        self.vt = torch.tensor(Vu, dtype=torch.float32, device=dev)
        self.ft = torch.tensor(Fu, dtype=torch.int32, device=dev)
        self.at = torch.tensor(attr, dtype=torch.float32, device=dev)
        self.nf = torch.tensor(np.repeat(fn, 3, 0), dtype=torch.float32, device=dev)
        # textured renderer for the reference texture panorama
        uvt = UV.copy()
        uvt[:, 1] = 1 - uvt[:, 1]
        self.uv = torch.tensor(uvt[F].reshape(-1, 2), dtype=torch.float32, device=dev)
        self.tex = torch.tensor(TEX, dtype=torch.float32, device=dev)[None].contiguous()
        self.txp = None

    def _raster(self, T, x0, y0, S, scale):
        K = self.K
        res = int(S * scale)
        fx, fy = K[0, 0] * scale, K[1, 1] * scale
        cx, cy = (K[0, 2] - x0 + 0.5) * scale, (K[1, 2] - y0 + 0.5) * scale
        Tt = torch.tensor(T, dtype=torch.float32, device="cuda")
        vc = self.vt @ Tt[:3, :3].T + Tt[:3, 3]
        near, far = 0.05, 20.0
        x, yy, z = vc[:, 0], vc[:, 1], vc[:, 2]
        clip = torch.stack([(2 * fx * x + (2 * cx - res) * z) / res, (-2 * fy * yy - (2 * cy - res) * z) / res,
                            (far + near) / (far - near) * z - 2 * far * near / (far - near), z], -1)[None].contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.ft, resolution=[res, res])
        nc = self.nf @ Tt[:3, :3].T
        vd = -vc / torch.linalg.norm(vc, dim=1, keepdim=True)
        cosv = (nc * vd).sum(1, keepdim=True)
        a, _ = dr.interpolate(torch.cat([self.at, cosv], 1)[None].contiguous(), rast, self.ft)
        return rast, a

    def pano_from_gray(self, gimg, x0, y0, S, T, permask=None, scale=4):
        g = cv2.resize(gimg, (S * scale, S * scale), interpolation=cv2.INTER_CUBIC)
        rast, a = self._raster(T, x0, y0, S, scale)
        a = a[0].flip(0).cpu().numpy()
        m = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()
        ok = m & (a[..., 3] > 0.99) & (a[..., 4] > 0.35)
        if permask is not None:
            ok &= ~cv2.resize(permask.astype(np.uint8), (S * scale, S * scale),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        ph = np.degrees(np.arctan2(a[..., 1], a[..., 0])) % 360
        h = a[..., 2]
        w = ok.astype(np.float32)
        k = int(6 * scale) | 1
        mu = cv2.GaussianBlur(g * w, (k, k), 0) / (cv2.GaussianBlur(w, (k, k), 0) + 1e-6)
        hp = g - mu
        pi = (ph[ok] / 360 * self.NPHI).astype(int) % self.NPHI
        hi = np.clip(((h[ok] - self.HB[0]) / (self.HB[1] - self.HB[0]) * self.NH).astype(int), 0, self.NH - 1)
        sm = np.zeros((self.NH, self.NPHI))
        wt = np.zeros((self.NH, self.NPHI))
        np.add.at(sm, (hi, pi), hp[ok])
        np.add.at(wt, (hi, pi), 1)
        return sm, wt

    def texture_pano(self, dist=2.6, nview=12):
        from scipy.spatial.transform import Rotation as Rot
        K = self.K
        sm = np.zeros((self.NH, self.NPHI))
        wt = np.zeros((self.NH, self.NPHI))
        for k in range(nview):
            T = np.eye(4)
            T[:3, :3] = Rot.from_euler("x", -10, degrees=True).as_matrix() @ \
                Rot.from_euler("y", k * 360 / nview, degrees=True).as_matrix()
            T[:3, 3] = [0, 0, dist]
            T[:3, 3] -= T[:3, :3] @ AXC
            p = K @ T[:3, 3]
            u, v = p[:2] / p[2]
            S = int(K[0, 0] * 0.4 / p[2]) + 8
            x0, y0 = int(u - S / 2), int(v - S / 2)
            rast, _ = self._raster(T, x0, y0, S, 1)
            uvi, _ = dr.interpolate(self.uv[None].contiguous(), rast, self.ft)
            col = dr.texture(self.tex, uvi, filter_mode="linear")
            col = torch.where(rast[..., 3:4] > 0, col, torch.zeros_like(col))[0].flip(0).cpu().numpy()
            gimg = (col @ np.array([0.299, 0.587, 0.114]) * 255).astype(np.float32)
            s, w = self.pano_from_gray(gimg, x0, y0, S, T)
            sm += s
            wt += w
        self.txp = np.where(wt > 0, sm / np.maximum(wt, 1), 0)

    def frame_pano(self, img_bgr, permask_full, T):
        K = self.K
        p = K @ T[:3, 3]
        u, v = p[:2] / p[2]
        S = int(K[0, 0] * 0.4 / p[2]) + 8
        x0, y0 = int(u - S / 2), int(v - S / 2)
        H, W = img_bgr.shape[:2]
        crop = np.zeros((S, S, 3), np.uint8)
        pm = np.zeros((S, S), bool)
        ya, yb, xa, xb = max(y0, 0), min(y0 + S, H), max(x0, 0), min(x0 + S, W)
        if yb > ya and xb > xa:
            crop[ya - y0:yb - y0, xa - x0:xb - x0] = img_bgr[ya:yb, xa:xb]
            pm[ya - y0:yb - y0, xa - x0:xb - x0] = permask_full[ya:yb, xa:xb]
        pm = cv2.dilate(pm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return self.pano_from_gray(g, x0, y0, S, T, permask=pm)

    def corr(self, sm, wt):
        """score per shift s (bins): video bunny outlines are bright, texture outlines dark -> sign -1."""
        P = np.where(wt > 0, sm / np.maximum(wt, 1), 0)
        Wm = (wt > 0).astype(float)
        P = P - (P * Wm).sum(1, keepdims=True) / np.maximum(Wm.sum(1, keepdims=True), 1)
        P *= Wm
        sc = np.zeros(self.NPHI)
        pn = (P ** 2).sum()
        for s in range(self.NPHI):
            Ts = np.roll(self.txp, s, axis=1)
            sc[s] = -(P * Ts).sum() / (np.sqrt(pn * ((Ts ** 2) * Wm).sum()) + 1e-9)
        return sc


# ------------------------------------------------------------------------------------------------------------------
def read_frames(path, F):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(F):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


def phase_of(F):
    ph = np.zeros(F, int)
    for k, (_, a, b, _) in enumerate(PHASES):
        ph[a:b + 1] = k
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    args = ap.parse_args()
    t_start = time.time()
    inp = args.inputs
    K, Twc, rec, cam = load_inputs(inp)
    Tcw = np.linalg.inv(Twc)
    Tb_w = rec["T_world_mesh"].astype(np.float64)
    F = len(Tb_w)
    Tb_c = Tcw @ Tb_w
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], np.float64)
    n_up = plane[:3] / np.linalg.norm(plane[:3])          # points to -y (up) in this y-down world
    if n_up[1] > 0:
        n_up = -n_up
    kp = rec["human_keypoints_world"].astype(np.float64)
    g, V, Fc, UV, TEX = load_mesh(inp)
    Vd, Fd = decimated_mesh(V, Fc)
    cov = CovRenderer(Vd, Fd)
    ph = phase_of(F)
    log = {}

    # ---------------- masks, crops ----------------
    area, bbox, cen = mask_stats(inp)
    visible = area > 300
    anc_b = np.einsum("fij,j->fi", Tb_c[:, :3, :3], PC) + Tb_c[:, :3, 3]
    proj = (K @ anc_b.T).T
    proj = proj[:, :2] / proj[:, 2:]
    # reference depth for crop sizes (baseline is garbage during the occluded carry, but no mask there anyway)
    zref_crop = np.clip(anc_b[:, 2], 1.5, 6.0)
    diam = K[0, 0] * 0.26 / zref_crop
    centers = np.where((area > 200)[:, None], (bbox[:, :2] + bbox[:, 2:]) / 2, proj)
    bsz = np.nan_to_num(np.max(bbox[:, 2:] - bbox[:, :2], 1))
    sizes = np.maximum(2.0 * diam, 1.5 * bsz)
    obj, per, inim, boxes = load_mask_crops(inp, centers, sizes, RES)
    Kc = crop_K(K, boxes, RES)
    valid = inim * (1 - per)

    # ---------------- pass 1: per-frame silhouette fit from the baseline ----------------
    # static phases: start upright (support = floor / table parallel to the floor) with the baseline's median yaw,
    # at the mask centre with the baseline's median depth (the baseline tilt in S3 is up to 80 deg off)
    T0_c = Tb_c.copy()
    anc0 = anc_b.copy()
    for nm, a, b, kind in PHASES:
        if kind != "static":
            continue
        ym = np.degrees(np.angle(np.exp(1j * np.radians(wyaw(Tb_w[a:b + 1, :3, :3]))).mean()))
        Rw0 = rot_from_axis_yaw(n_up, ym)
        zm = np.median(anc_b[a:b + 1, 2])
        for i in range(a, b + 1):
            T0_c[i, :3, :3] = Tcw[:3, :3] @ Rw0
            if area[i] > 200:
                c = (bbox[i, :2] + bbox[i, 2:]) / 2
                anc0[i] = np.array([(c[0] - K[0, 2]) / K[0, 0], (c[1] - K[1, 2]) / K[1, 1], 1.0]) * zm
            else:
                anc0[i] = anc_b[i] / anc_b[i, 2] * zm
    fr = np.nonzero(visible)[0]
    T1_c = Tb_c.copy()
    iou_base = np.full(F, np.nan)
    iou1 = np.full(F, np.nan)
    for s in range(0, len(fr), 200):
        f = fr[s:s + 200]
        R, A, iou = fit_lm(cov, T0_c[f, :3, :3], anc0[f], Kc[f], obj[f], valid[f], PC, K, iters=15,
                           w_rot_prior=0.02, res=RES)
        _, _, iou0 = fit_lm(cov, Tb_c[f, :3, :3], anc_b[f], Kc[f], obj[f], valid[f], PC, K, iters=0, res=RES,
                            init_centroid=False)
        for k, i in enumerate(f):
            T1_c[i, :3, :3] = R[k]
            T1_c[i, :3, 3] = A[k] - R[k] @ PC
        iou_base[f] = iou0
        iou1[f] = iou
    T1_w = Twc @ T1_c
    # reliability: visible object-mask area relative to the unoccluded rendered silhouette area of the pass-1 pose
    exp_area = np.zeros(F)
    with torch.no_grad():
        for s in range(0, len(fr), 200):
            f = fr[s:s + 200]
            Rt = torch.tensor(T1_c[f, :3, :3], dtype=torch.float32, device="cuda")
            tt = torch.tensor(T1_c[f, :3, 3], dtype=torch.float32, device="cuda")
            sl = cov.render(Rt, tt, torch.tensor(Kc[f], dtype=torch.float32, device="cuda"), RES)
            exp_area[f] = sl.sum((1, 2)).cpu().numpy() * (boxes[f, 2] / RES) ** 2
    vis_frac = np.where(exp_area > 0, area / np.maximum(exp_area, 1), 0)
    reliable = visible & (vis_frac > 0.6) & (iou1 > 0.6)
    print(f"[pass1] reliable frames: {reliable.sum()} / visible {visible.sum()}")
    print(f"[pass1] mean IoU baseline {np.nanmean(iou_base):.3f} -> {np.nanmean(iou1):.3f}  ({time.time()-t_start:.0f}s)")

    # ---------------- yaw evidence ----------------
    frames = read_frames(f"{inp}/video.mp4", F)
    unw = Unwrapper(V, Fc, UV, TEX, K)
    unw.texture_pano()
    pm_all = h5py.File(f"{inp}/masks.h5", "r")["person"]
    yaws = np.arange(NB) * 3.0
    Ztex = np.zeros((F, NB))
    has_tex = np.zeros(F, bool)
    for i in fr:
        sm, wt = unw.frame_pano(frames[i], pm_all[i], T1_c[i])
        if (wt.sum(0) > 0).mean() < 0.15:
            continue
        sc = unw.corr(sm, wt)
        y0 = wyaw(T1_w[i, :3, :3])
        # shift s means: true pose = estimate o yawrot(-s)
        ays = np.array([wyaw((T1_w[i] @ yawrot(np.radians(-s_ * 3.0)))[:3, :3]) % 360 for s_ in range(NB)])
        o = np.argsort(ays)
        row = np.interp(yaws, ays[o], sc[o], period=360)       # scores on the absolute-yaw grid
        from scipy.ndimage import gaussian_filter1d
        row = gaussian_filter1d(row, 1.0, mode="wrap")         # remove 3-deg binning comb
        Ztex[i] = (row - row.mean()) / (row.std() + 1e-9)
        has_tex[i] = True
    print(f"[yaw] texture scores for {has_tex.sum()} frames ({time.time()-t_start:.0f}s)")

    # per frame: axis (tilt) and anchor from pass 1; static phases upright
    axis1 = T1_w[:, :3, 1].copy()
    anc1_w = np.einsum("fij,j->fi", T1_w[:, :3, :3], PC) + T1_w[:, :3, 3]
    cam_c = Twc[:3, 3]
    yaws = np.arange(NB) * 3.0
    lugm = np.array([np.cos(np.radians(LUG_AZIMUTH_DEG)), 0, np.sin(np.radians(LUG_AZIMUTH_DEG))])
    lug_on = np.zeros(F, bool)
    for a, b in LUG_SIDE_RANGES:
        lug_on[a:b + 1] = True
    base_ok = visible & (iou_base > 0.6)
    base_yaw = wyaw(Tb_w[:, :3, :3])
    data = np.zeros((F, NB))
    for i in range(F):
        d = np.zeros(NB)
        if has_tex[i]:
            d -= Ztex[i]
        if lug_on[i] and visible[i]:
            ray = anc1_w[i] - cam_c
            ray /= np.linalg.norm(ray)
            ax = axis1[i] if PHASES[ph[i]][3] == "carry" else n_up
            for j, ay in enumerate(yaws):
                Rw = rot_from_axis_yaw(ax, ay)
                d[j] += W_LUG * float((Rw @ lugm) @ ray) ** 2
        if base_ok[i] and reliable[i]:
            d -= W_BASE * np.cos(np.radians(yaws - base_yaw[i]))
        data[i] = d
    for k, (_, a, b, kind) in enumerate(PHASES):
        if kind == "static":
            data[a:b + 1] *= STATIC_TOTAL_W / (b - a + 1)
    # Forward-backward over yaw (costs act as negative log-likelihoods). The posterior per frame is summarised by the
    # circular mean of its dominant mode (+-60 deg around the MAP bin): minimum expected error, and stable against
    # near-ties (a hard Viterbi argmax jumped by 20-40 deg between near-identical runs in static phase S2).
    data = data / TEMP

    def trans_list(i):
        static_step = PHASES[ph[i]][3] == "static" and ph[i] == ph[i - 1]
        md = 0 if static_step else MAXD
        return [(dd, W_TRANS * dd * dd / TEMP) for dd in range(-md, md + 1)]

    def lse(x, axis=0):
        m = x.max(axis)
        return m + np.log(np.exp(x - np.expand_dims(m, axis)).sum(axis))

    la = np.zeros((F, NB))
    la[0] = -data[0]
    la[0] -= lse(la[0])
    for i in range(1, F):
        tl = trans_list(i)
        st = np.stack([np.roll(la[i - 1], dd) - c for dd, c in tl])
        la[i] = lse(st) - data[i]
        la[i] -= lse(la[i])
    lb = np.zeros((F, NB))
    for i in range(F - 2, -1, -1):
        tl = trans_list(i + 1)
        st = np.stack([np.roll(lb[i + 1] - data[i + 1], -dd) - c for dd, c in tl])
        lb[i] = lse(st)
        lb[i] -= lse(lb[i])
    post = la + lb
    post = np.exp(post - post.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    yaw_dp = np.zeros(F)
    for i in range(F):
        jm = int(np.argmax(post[i]))
        dlt = ((yaws - yaws[jm] + 180) % 360) - 180
        w = post[i] * (np.abs(dlt) <= 60)
        yaw_dp[i] = (yaws[jm] + np.degrees(np.arctan2((w * np.sin(np.radians(dlt))).sum(),
                                                       (w * np.cos(np.radians(dlt))).sum()))) % 360
    log["yaw_posterior_mass_in_mode_per_phase"] = {
        nm: float(np.mean([post[i][np.abs(((yaws - yaws[int(np.argmax(post[i]))] + 180) % 360) - 180) <= 60].sum()
                           for i in range(a_, b_ + 1)])) for nm, a_, b_, _ in PHASES}
    yaw_final = yaw_dp.copy()
    for k, (_, a, b, kind) in enumerate(PHASES):
        if kind == "carry":
            yaw_final[a:b + 1] = circ_smooth_deg(yaw_dp[a:b + 1], 4.0)
    print("[yaw] decoded yaw per phase:",
          [(nm, float(np.round(np.degrees(np.angle(np.exp(1j * np.radians(yaw_final[a:b + 1])).mean())) % 360, 1)))
           for nm, a, b, _ in PHASES])

    # ---------------- rotations ----------------
    R_w = np.zeros((F, 3, 3))
    # carried: tilt axis from pass 1 (visible frames), smoothed, interpolated over invisible frames
    axis_s = np.tile(n_up, (F, 1))
    for k, (_, a, b, kind) in enumerate(PHASES):
        if kind != "carry":
            continue
        idx = np.arange(a, b + 1)
        ax = axis1[idx].copy()
        # reliable frames only; reject axes tilted more than 45 deg from up (flips)
        good = reliable[idx] & (ax @ n_up > np.cos(np.radians(45)))
        # the basket is picked up from / put down on a horizontal support: upright at both ends of the carry
        ti = np.concatenate([[a - 1], idx[good], [b + 1]])
        tv = np.concatenate([n_up[None], ax[good], n_up[None]], 0)
        axi = np.stack([np.interp(idx, ti, tv[:, c]) for c in range(3)], 1)
        axi = smooth(axi, 3.0)
        axi /= np.linalg.norm(axi, axis=1, keepdims=True)
        axis_s[a:b + 1] = axi
    for i in range(F):
        R_w[i] = rot_from_axis_yaw(axis_s[i], yaw_final[i])

    # ---------------- translations ----------------
    # reference depth of the anchor: baseline (smoothed within phases); per phase re-anchored at the end
    zb = anc_b[:, 2].copy()
    zref = zb.copy()
    for k, (_, a, b, kind) in enumerate(PHASES):
        if kind == "static":
            zref[a:b + 1] = np.median(zb[a:b + 1])
        else:
            from scipy.ndimage import median_filter
            zref[a:b + 1] = smooth(median_filter(zb[a:b + 1], 9, mode="nearest"), 2.0)
    R_c = Tcw[:3, :3] @ R_w
    anc_c = np.zeros((F, 3))
    final_iou = np.full(F, np.nan)
    # carried, visible frames: fit image position with temporal smoothing, rotations fixed
    for k, (_, a, b, kind) in enumerate(PHASES):
        idx = np.arange(a, b + 1)
        if kind == "static":
            f = idx[visible[idx]]
            if len(f) == 0:
                continue
            # one pose for all visible frames of the phase
            a0 = anc1_w[f].mean(0)
            a0c = Tcw[:3, :3] @ a0 + Tcw[:3, 3]
            a0c = a0c / a0c[2] * zref[a]
            Rs = np.repeat(R_c[a:a + 1], len(f), 0)
            A0 = np.repeat(a0c[None], len(f), 0)
            R_, A_, iou = fit_lm(cov, Rs, A0, Kc[f], obj[f], valid[f], PC, K, iters=15,
                                 fit_rot=False, static=True, res=RES)
            anc_c[a:b + 1] = A_[0]
            final_iou[f] = iou
        else:
            if OCCLUDED_CARRY[0] >= a and OCCLUDED_CARRY[1] <= b:
                seg = idx[idx < OCCLUDED_CARRY[0]]
            else:
                seg = idx
            a1c = (Tcw[:3, :3] @ anc1_w[seg].T).T + Tcw[:3, 3]
            # invisible frames inside the segment: interpolate the pass-1 image position
            vis = visible[seg]
            uv = a1c[:, :2] / a1c[:, 2:]
            uvi = np.stack([np.interp(seg, seg[vis], uv[vis, c]) for c in range(2)], 1)
            A0 = np.concatenate([uvi * zref[seg][:, None], zref[seg][:, None]], 1)
            R_, A_, iou = fit_lm(cov, R_c[seg], A0, Kc[seg], obj[seg], valid[seg], PC, K, iters=15,
                                 fit_rot=False, res=RES)
            # temporal smoothing of the image position, weighted by how much of the object is visible
            wv = np.clip(vis_frac[seg], 0, 1) * vis * (iou > 0.5)
            uvf = A_[:, :2] / A_[:, 2:]
            from scipy.ndimage import gaussian_filter1d
            num = gaussian_filter1d(uvf * wv[:, None], 2.0, axis=0, mode="nearest")
            den = gaussian_filter1d(wv, 2.0, mode="nearest")[:, None]
            uvs = np.where(den > 1e-3, num / np.maximum(den, 1e-9), uvi)
            A_ = np.concatenate([uvs * A_[:, 2:], A_[:, 2:]], 1)
            anc_c[seg] = A_
            final_iou[seg[vis]] = iou[vis]
    print(f"[fit] final mean IoU (visible) {np.nanmean(final_iou[visible]):.3f}  ({time.time()-t_start:.0f}s)")

    # ---------------- re-anchor depth per phase (along the viewing rays) ----------------
    o0, o1 = OCCLUDED_CARRY
    depth_scale = {}
    for nm, a, b, kind in PHASES:
        sl = np.arange(a, b + 1)
        if nm.startswith("C2"):
            # the occluded part follows the hand (the baseline has lost the object there)
            sl = sl[(sl < o0) | (sl > o1)]
        org_z = anc_c[sl, 2] - np.einsum("fj,j->f", R_c[sl, 2, :], PC)   # camera z of the mesh origin
        s = np.median(Tb_c[sl, 2, 3]) / np.median(org_z)
        depth_scale[nm] = float(s)
        anc_c[sl] *= s

    # ---------------- occluded carry: attach to the holding hand ----------------
    anc_w = (Twc[:3, :3] @ anc_c.T).T + Twc[:3, 3]
    hands = {b: kp[:, b:b + 21].mean(1) for b in (21, 42)}
    pre = np.arange(o0 - 20, o0)
    dist = {b: np.linalg.norm(hands[b][pre] - anc_w[pre], axis=1).mean() for b in hands}
    hb = min(dist, key=dist.get)
    hand = smooth(hands[hb], 2.0)
    off0 = (anc_w[pre] - hand[pre]).mean(0)                   # object - hand before the occlusion
    after = np.arange(o1 + 1, o1 + 6)
    off1 = (anc_w[after] - hand[after]).mean(0)                  # object (resting, S3) - hand at release
    for i in range(o0, o1 + 1):
        w = (i - (o0 - 1)) / ((o1 + 1) - (o0 - 1))
        anc_w[i] = hand[i] + (1 - w) * off0 + w * off1
    log["occluded_carry_hand"] = int(hb)
    log["occluded_hand_offset_before_cm"] = (off0 * 100).round(1).tolist()
    log["occluded_hand_offset_after_cm"] = (off1 * 100).round(1).tolist()
    anc_c = (Tcw[:3, :3] @ anc_w.T).T + Tcw[:3, 3]

    # ---------------- assemble ----------------
    T_c = np.tile(np.eye(4), (F, 1, 1))
    T_c[:, :3, :3] = R_c
    T_c[:, :3, 3] = anc_c - np.einsum("fij,j->fi", R_c, PC)
    log["depth_rescale_per_phase"] = depth_scale
    T_w = Twc @ T_c
    np.savez(args.out, T_world_mesh=T_w.astype(np.float64))
    np.savez(os.path.join(os.path.dirname(args.out), "solve_diag.npz"), data=data, Ztex=Ztex, has_tex=has_tex,
             post=post, yaw_dp=yaw_dp, yaw_final=yaw_final, iou_base=iou_base, iou1=iou1, final_iou=final_iou,
             reliable=reliable, vis_frac=vis_frac, T1_w=T1_w, axis_s=axis_s)
    log["runtime_s"] = time.time() - t_start
    log["torch_max_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(log, open(os.path.join(os.path.dirname(args.out), "solve_log.json"), "w"), indent=1)
    print(f"[done] wrote {args.out}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
