#!/usr/bin/env python3
"""Deterministic trajectory recovery for a monocular rigid-object video.

The method starts from the released frame-wise reconstruction, enforces the two
visually verified rest phases, robustly filters the manipulation interval,
re-centres each silhouette in the image, decodes smooth multi-frame rotation
corrections with Viterbi, and finally resolves spin about the bottle's long axis
using the textured mesh.  It does not read any hidden/ground-truth annotation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

SEED = 20261020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = 'cuda'
H, W = 1152, 1536


def projection_matrix(K: np.ndarray, h: int, w: int, near=0.001, far=100.0) -> np.ndarray:
    """OpenCV intrinsic -> OpenGL clip projection used by nvdiffrast (y down)."""
    depth = float(far - near)
    q = -(far + near) / depth
    qn = -2.0 * far * near / depth
    return np.array([
        [2.0 * K[0, 0] / w, -2.0 * K[0, 1] / w, (-2.0 * K[0, 2] + w) / w, 0.0],
        [0.0, 2.0 * K[1, 1] / h, (2.0 * K[1, 2] - h) / h, 0.0],
        [0.0, 0.0, q, qn],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float64)


def qcontinuous(q: np.ndarray) -> np.ndarray:
    out = [q[0]]
    for x in q[1:]:
        out.append(-x if np.dot(out[-1], x) < 0 else x)
    return np.asarray(out)


def qmed(q: np.ndarray, radius: int, iters: int = 2) -> np.ndarray:
    q = q.copy()
    n = len(q)
    for _ in range(iters):
        nxt = []
        for i in range(n):
            ids = np.arange(max(0, i - radius), min(n, i + radius + 1))
            cand = q[ids]
            angles = np.arccos(np.clip(np.abs(cand @ cand.T), -1, 1))
            nxt.append(q[ids[np.argmin(angles.sum(1))]])
        q = qcontinuous(np.asarray(nxt))
    return q


def robust_quadratic_series(y: np.ndarray, radius: int, threshold: float) -> np.ndarray:
    """Local robust quadratic regression, used as a zero-lag temporal filter."""
    n = len(y)
    out = np.empty_like(y)
    for i in range(n):
        ids = np.arange(max(0, i - radius), min(n, i + radius + 1))
        x = (ids - i).astype(np.float64)
        yy = y[ids]
        w = np.ones(len(ids), dtype=np.float64)
        coef = None
        for _ in range(2):
            A = np.stack([np.ones_like(x), x, x * x], 1)
            Aw = A * np.sqrt(w[:, None])
            yw = yy * np.sqrt(w[:, None])
            coef = np.linalg.lstsq(Aw, yw, rcond=None)[0]
            pred = A @ coef
            err = np.linalg.norm(yy - pred, axis=1) / threshold
            w = 1.0 / np.maximum(1.0, err * err)
        out[i] = coef[0]
    return out


def robust_filter_pose(T: np.ndarray, radius=9, rot_threshold_deg=20.0,
                       trans_threshold_m=0.12) -> np.ndarray:
    """Remove one-frame network flips while retaining low-frequency motion."""
    F = len(T)
    q = qcontinuous(Rot.from_matrix(T[:, :3, :3]).as_quat())
    q = qmed(q, 2, 1)
    qf = []
    for i in range(F):
        ids = np.arange(max(0, i - radius), min(F, i + radius + 1))
        x = (ids - i).astype(np.float64)
        ref = Rot.from_quat(q[i])
        rel = (ref.inv() * Rot.from_quat(q[ids])).as_rotvec()
        w = np.ones(len(ids), dtype=np.float64)
        for _ in range(2):
            A = np.stack([np.ones_like(x), x, x * x], 1)
            coef = np.linalg.lstsq(A * np.sqrt(w[:, None]),
                                   rel * np.sqrt(w[:, None]), rcond=None)[0]
            pred = A @ coef
            err = np.linalg.norm(rel - pred, axis=1) / np.deg2rad(rot_threshold_deg)
            w = 1.0 / np.maximum(1.0, err * err)
        qf.append((ref * Rot.from_rotvec(coef[0])).as_quat())
    q = qcontinuous(np.asarray(qf))
    tf = robust_quadratic_series(T[:, :3, 3], radius, trans_threshold_m)
    out = np.repeat(np.eye(4, dtype=np.float64)[None], F, axis=0)
    out[:, :3, :3] = Rot.from_quat(q).as_matrix()
    out[:, :3, 3] = tf
    return out


class Renderer:
    def __init__(self, mesh: trimesh.Trimesh, Tcw: np.ndarray, K: np.ndarray):
        self.mesh = mesh
        self.Tcw = Tcw.astype(np.float32)
        self.K = K.astype(np.float64)
        self.verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=DEVICE)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32, device=DEVICE)
        self.homo = torch.cat([self.verts, torch.ones((len(self.verts), 1), device=DEVICE)], 1)
        self.glctx = dr.RasterizeCudaContext()
        self.P = torch.tensor(projection_matrix(self.K, H, W),
                              dtype=torch.float32, device=DEVICE)
        self.G = torch.tensor(np.diag([1.0, -1.0, -1.0, 1.0]),
                              dtype=torch.float32, device=DEVICE)

    def raster_ids(self, poses_world: np.ndarray):
        poses_world = np.asarray(poses_world, dtype=np.float32).reshape(-1, 4, 4)
        cam = torch.tensor(np.stack([self.Tcw @ p for p in poses_world]),
                           dtype=torch.float32, device=DEVICE)
        mtx = self.P[None] @ self.G[None] @ cam
        clip = (mtx[:, None] @ self.homo[None, :, :, None])[..., 0]
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.faces,
                               resolution=(H, W))
        # nvdiffrast triangle IDs are 1-based; background is 0.
        return rast[..., 3].flip(1)

    def masks(self, poses_world: np.ndarray) -> np.ndarray:
        return self.raster_ids(poses_world).cpu().numpy() > 0

    def color_crop(self, pose_world: np.ndarray, Kcrop: np.ndarray,
                   size: int, tex_tensors) -> np.ndarray:
        from nvdiffrast.torch import texture  # local import keeps startup small
        Pc = torch.tensor(projection_matrix(Kcrop, size, size),
                          dtype=torch.float32, device=DEVICE)
        cam = torch.tensor((self.Tcw @ pose_world)[None], dtype=torch.float32,
                           device=DEVICE)
        mtx = Pc[None] @ self.G[None] @ cam
        clip = (mtx[:, None] @ self.homo[None, :, :, None])[..., 0]
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.faces,
                               resolution=(size, size))
        texc, _ = dr.interpolate(tex_tensors['uv'].contiguous(), rast,
                                 self.faces.contiguous())
        col = texture(tex_tensors['tex'], texc, filter_mode='linear').clip(0, 1)
        col = col * torch.clamp(rast[..., -1:], 0, 1)
        return col[0].flip(0).cpu().numpy()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + 1e-9))


def best_anchor(T: np.ndarray, candidate_ids: np.ndarray, target_ids: np.ndarray,
                renderer: Renderer, obj: np.ndarray, perd: np.ndarray) -> int:
    """Pick the rest-frame pose that best explains all other rest frames."""
    cand = np.stack([T[i] for i in candidate_ids])
    scores = np.zeros(len(candidate_ids), dtype=np.float64)
    for i in target_ids:
        rms = renderer.masks(cand)
        tv = obj[i] & ~perd[i]
        for j, r in enumerate(rms):
            rv = r & ~perd[i]
            scores[j] += iou(rv, tv)
    return int(candidate_ids[np.argmax(scores)])


def refine_static_rotation(T0: np.ndarray, segment_ids: np.ndarray,
                           sample_ids: np.ndarray, renderer: Renderer,
                           obj: np.ndarray, perd: np.ndarray,
                           angles_deg: np.ndarray) -> np.ndarray:
    """Pool one constant rest pose over several frames (exact mask IoU)."""
    deltas = []
    for ax in angles_deg:
        for ay in angles_deg:
            deltas.append(Rot.from_euler('xy', [ax, ay], degrees=True).as_matrix())
    deltas = np.asarray(deltas)
    score = np.zeros(len(deltas), dtype=np.float64)
    Rc0 = T0[:3, :3]
    for i in sample_ids:
        poses = []
        for d in deltas:
            p = T0.copy()
            p[:3, :3] = d @ Rc0
            poses.append(p)
        rms = renderer.masks(np.asarray(poses))
        tv = obj[i] & ~perd[i]
        for j, r in enumerate(rms):
            score[j] += iou(r & ~perd[i], tv)
    best = int(np.argmax(score))
    out = T0.copy()
    out[:3, :3] = deltas[best] @ Rc0
    return out


def shift_static_pose(T: np.ndarray, segment_ids: np.ndarray, renderer: Renderer,
                      obj: np.ndarray, perd: np.ndarray, Tcw: np.ndarray,
                      Twc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Re-centre a constant pose using the median visible-mask centroid."""
    out = T.copy()
    for _ in range(3):
        r = renderer.masks(out[None])[0]
        shifts = []
        for i in segment_ids:
            rv = r & ~perd[i]
            tv = obj[i] & ~perd[i]
            if rv.sum() > 50 and tv.sum() > 50:
                ys, xs = np.where(rv)
                yt, xt = np.where(tv)
                shifts.append((xt.mean() - xs.mean(), yt.mean() - ys.mean()))
        if not shifts:
            break
        dx, dy = np.median(np.asarray(shifts), axis=0)
        zc = (Tcw @ out)[2, 3]
        dc = np.array([dx * zc / K[0, 0], dy * zc / K[1, 1], 0.0])
        out[:3, 3] += Twc[:3, :3] @ dc
    return out


def centroid_refine(T: np.ndarray, ids: np.ndarray, renderer: Renderer,
                    obj: np.ndarray, perd: np.ndarray, Tcw: np.ndarray,
                    Twc: np.ndarray, K: np.ndarray, iterations=5) -> np.ndarray:
    """Translate each frame laterally so the rendered mask centroid tracks SAM."""
    out = T.copy()
    for _ in range(iterations):
        shifts = np.zeros((len(out), 2), dtype=np.float64)
        for b0 in range(0, len(ids), 12):
            bid = ids[b0:b0 + 12]
            rms = renderer.masks(out[bid])
            for i, r in zip(bid, rms):
                rv = r & ~perd[i]
                tv = obj[i] & ~perd[i]
                ys, xs = np.where(rv)
                yt, xt = np.where(tv)
                if len(xs) and len(yt):
                    shifts[i] = (xt.mean() - xs.mean(), yt.mean() - ys.mean())
        shifts = np.stack([gaussian_filter1d(shifts[:, k], 1.0) for k in range(2)], 1)
        for i in ids:
            zc = (Tcw @ out[i])[2, 3]
            dc = np.array([shifts[i, 0] * zc / K[0, 0],
                           shifts[i, 1] * zc / K[1, 1], 0.0])
            out[i, :3, 3] += Twc[:3, :3] @ dc
    return out


def rotation_viterbi(T: np.ndarray, s: int, e: int, angle_values: np.ndarray,
                     renderer: Renderer, obj: np.ndarray, perd: np.ndarray,
                     data_weight=18.0, prior_weight=0.0015,
                     transition_weight=0.004) -> np.ndarray:
    """Decode smooth camera-frame tilt corrections over the manipulation phase."""
    ds = []
    for ax in angle_values:
        for ay in angle_values:
            ds.append(Rot.from_euler('xy', [ax, ay], degrees=True).as_matrix())
    ds = np.asarray(ds)
    na = len(ds)
    ax = angle_values[np.arange(na) // len(angle_values)]
    ay = angle_values[np.arange(na) % len(angle_values)]
    data = np.empty((e - s, na), dtype=np.float32)
    out = T.copy()
    for k, i in enumerate(range(s, e)):
        cur = out[i]
        poses = []
        for d in ds:
            p = cur.copy()
            p[:3, :3] = d @ cur[:3, :3]
            poses.append(p)
        ids = renderer.raster_ids(np.asarray(poses))
        yy, xx = np.where(obj[i])
        pad = 36
        x0 = max(0, int(xx.min()) - pad)
        x1 = min(W, int(xx.max()) + pad + 1)
        y0 = max(0, int(yy.min()) - pad)
        y1 = min(H, int(yy.max()) + pad + 1)
        rmc = ids[:, y0:y1, x0:x1] > 0
        pvc = torch.tensor(perd[i, y0:y1, x0:x1], dtype=torch.bool, device=DEVICE)
        tvc = torch.tensor(obj[i, y0:y1, x0:x1] & ~perd[i, y0:y1, x0:x1],
                           dtype=torch.bool, device=DEVICE)
        rvc = rmc & ~pvc
        inter = (rvc & tvc).sum((1, 2)).float()
        union = (rvc | tvc).sum((1, 2)).float()
        vals = (1.0 - inter / (union + 1e-9)).cpu().numpy()
        tv_area = int(tvc.sum().item())
        wgt = float(np.clip(np.sqrt(max(1, tv_area) / 2000.0), 0.25, 1.0))
        data[k] = vals * data_weight * wgt
        if k % 80 == 0:
            print(f'  rotation search frame {i}/{e}', flush=True)
    prior = prior_weight * (ax * ax + ay * ay)
    transition = transition_weight * ((ax[:, None] - ax[None, :]) ** 2 +
                                      (ay[:, None] - ay[None, :]) ** 2)
    dp = data[0] + prior
    zero_idx = int(np.argmin(np.abs(angle_values)))
    dp[zero_idx] = data[0, zero_idx] + prior[zero_idx]
    dp[np.arange(na) != zero_idx] += 1000.0  # first frame still matches rest pose
    back = np.zeros((e - s, na), dtype=np.int16)
    for k in range(1, e - s):
        vals = dp[None, :] + transition
        back[k] = np.argmin(vals, 1)
        dp = data[k] + np.min(vals, 1) + prior
    dp += 1000.0
    dp[zero_idx] = data[-1, zero_idx] + prior[zero_idx]  # returns to final rest pose
    path = np.zeros(e - s, dtype=np.int16)
    path[-1] = zero_idx
    for k in range(len(path) - 2, -1, -1):
        path[k] = back[k + 1, path[k + 1]]
    for k, i in enumerate(range(s, e)):
        d = ds[path[k]]
        out[i, :3, :3] = d @ T[i, :3, :3]
    return out


def spin_viterbi(T: np.ndarray, s: int, e: int, renderer: Renderer,
                 obj: np.ndarray, per: np.ndarray, K: np.ndarray,
                 tex_tensors) -> np.ndarray:
    """Resolve texture/spin about the bottle's local z axis."""
    import cv2
    angles = np.arange(0, 360, 15)
    Rz = []
    for a in angles:
        r = np.eye(4)
        r[:3, :3] = Rot.from_euler('z', a, degrees=True).as_matrix()
        Rz.append(r)
    Rz = np.asarray(Rz)
    cc = np.full((e - s, len(angles)), -1.0, dtype=np.float32)
    hc = np.full_like(cc, -1.0)
    areas = np.zeros(e - s, dtype=np.int32)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    for k, i in enumerate(range(s, e)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError(f'cannot read video frame {i}')
        yy, xx = np.where(obj[i])
        side = int(max(xx.max() - xx.min(), yy.max() - yy.min()) + 36)
        side = int(np.clip(side, 128, 320))
        cx = int((xx.min() + xx.max()) // 2)
        cy = int((yy.min() + yy.max()) // 2)
        x0 = int(np.clip(cx - side // 2, 0, W - side))
        y0 = int(np.clip(cy - side // 2, 0, H - side))
        Kc = K.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        obs = fr[y0:y0 + side, x0:x0 + side, ::-1].astype(np.float32) / 255.0
        tgt = obj[i, y0:y0 + side, x0:x0 + side] & ~per[i, y0:y0 + side, x0:x0 + side]
        areas[k] = tgt.sum()
        hp_obs = (obs - cv2.GaussianBlur(obs, (0, 0), 2.0)).reshape(-1)
        cols = []
        for a in angles:
            p = T[i].copy()
            p[:3, :3] = p[:3, :3] @ Rz[int(a // 15), :3, :3]
            cols.append(renderer.color_crop(p, Kc, side, tex_tensors))
        cols = np.asarray(cols)
        for j, c in enumerate(cols):
            rm = c.max(2) > 0.01
            vv = rm & tgt
            if vv.sum() < 30:
                continue
            x = c[vv]
            y = obs[vv]
            if x.std() < 1e-6 or y.std() < 1e-6:
                continue
            cc[k, j] = np.corrcoef(x.ravel(), y.ravel())[0, 1]
            hp = (c - cv2.GaussianBlur(c, (0, 0), 2.0))[vv]
            if hp.std() > 1e-6:
                hc[k, j] = np.corrcoef(hp.ravel(), hp_obs.reshape(obs.shape)[vv].ravel())[0, 1]
        if k % 60 == 0:
            print(f'  spin search frame {i}/{e}', flush=True)
    cap.release()
    ccf = np.maximum(cc, 0)
    hcf = np.maximum(hc, 0)
    weight = np.clip(np.sqrt(np.maximum(areas, 1) / 1800.0), 0.35, 1.0)
    data = ((1.0 - ccf) * 14.0 + (1.0 - hcf) * 6.0) * weight[:, None]
    bad = cc < 0
    if bad.any():
        data[bad] = np.median(data[~bad])
    rad = np.deg2rad(angles)
    sharp = np.abs((rad + np.pi) % (2 * np.pi) - np.pi) * 180.0 / np.pi
    prior = 0.0008 * (sharp / 15.0) ** 2
    D = np.abs((angles[:, None] - angles[None, :] + 180) % 360 - 180)
    trans = 0.2 * (D / 15.0) ** 2
    trans[D > 15] = 1e6  # no physically implausible one-frame spin flips
    dp = data[0] + 20.0 * prior
    dp[1:] += 1000.0
    back = np.zeros((e - s, len(angles)), dtype=np.int16)
    for k in range(1, e - s):
        vals = dp[None, :] + trans
        back[k] = np.argmin(vals, 1)
        dp = data[k] + np.min(vals, 1) + prior
    path = np.zeros(e - s, dtype=np.int16)
    path[-1] = 0
    for k in range(len(path) - 2, -1, -1):
        path[k] = back[k + 1, path[k + 1]]
    sel = angles[path]
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(sel)))
    smoothed = gaussian_filter1d(unwrapped, 1.0, mode='nearest')
    smoothed[0] = 0.0
    smoothed[-1] = 360.0 * np.round(smoothed[-2] / 360.0)
    smoothed[:5] *= np.linspace(0, 1, 5)
    corr = angles[np.rint((smoothed % 360) / 15).astype(int) % len(angles)]
    out = T.copy()
    for k, i in enumerate(range(s, e)):
        r = np.eye(4)
        r[:3, :3] = Rot.from_euler('z', corr[k], degrees=True).as_matrix()
        out[i, :3, :3] = T[i, :3, :3] @ r[:3, :3]
    return out


def slerp(q0, q1, t):
    r0 = Rot.from_quat(q0)
    r1 = Rot.from_quat(q1)
    return (r0 * Rot.from_rotvec((r0.inv() * r1).as_rotvec() * t)).as_quat()


def blend_rest_boundaries(T: np.ndarray, s: int, e: int,
                          initial_pose: np.ndarray, final_pose: np.ndarray,
                          width=8) -> np.ndarray:
    out = T.copy()
    qs = Rot.from_matrix(initial_pose[:3, :3]).as_quat()
    qe = Rot.from_matrix(T[s + width][:3, :3]).as_quat()
    for i in range(s, s + width):
        t = (i - s + 1) / width
        out[i, :3, :3] = Rot.from_quat(slerp(qs, qe, t)).as_matrix()
        out[i, :3, 3] = (1 - t) * initial_pose[:3, 3] + t * T[s + width, :3, 3]
    qs = Rot.from_matrix(T[e - width][:3, :3]).as_quat()
    qe = Rot.from_matrix(final_pose[:3, :3]).as_quat()
    for i in range(e - width + 1, e):
        t = (i - (e - width)) / (width - 1)
        out[i, :3, :3] = Rot.from_quat(slerp(qs, qe, t)).as_matrix()
        out[i, :3, 3] = (1 - t) * T[e - width, :3, 3] + t * final_pose[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    global VIDEO_PATH
    VIDEO_PATH = Path(args.inputs) / 'video.mp4'
    t_start = time.time()

    with np.load(Path(args.inputs) / 'reconstruction.npz', allow_pickle=True) as z:
        Tbase = z['T_world_mesh'].astype(np.float64)
        contact = z['contact_prob_right_left'].max(1)
    cam = json.load(open(Path(args.inputs) / 'camera.json'))
    K = np.asarray(cam['K'], dtype=np.float64)
    Twc = np.asarray(cam['T_world_cam'], dtype=np.float64)
    Tcw = np.linalg.inv(Twc)
    mesh = trimesh.load(Path(args.inputs) / 'object_mesh.glb', force='scene',
                        process=False).to_geometry()
    with h5py.File(Path(args.inputs) / 'masks.h5', 'r') as f:
        obj = f['object'][:]
        per = f['person'][:]
    kernel = np.ones((5, 5), np.uint8)
    perd = np.asarray([cv2.dilate(x.astype(np.uint8), kernel) > 0 for x in per])
    renderer = Renderer(mesh, Tcw, K)
    F = len(Tbase)

    # Hand contact brackets the manipulation interval.  The object can remain
    # at rest while first touched, so the exact rest end is found below.
    first_contact = int(np.argmax(contact > 0.5))
    last_contact = int(F - 1 - np.argmax(contact[::-1] > 0.5))
    print(f'contact brackets: {first_contact}..{last_contact}', flush=True)

    # ---- rest-pose evidence pooling -------------------------------------
    cand0 = np.arange(0, first_contact, 3)
    tgt0 = np.arange(0, first_contact, 3)
    a0 = best_anchor(Tbase, cand0, tgt0, renderer, obj, perd)
    S0 = Tbase[a0].copy()
    S0 = refine_static_rotation(S0, np.arange(0, first_contact),
                                np.arange(0, first_contact, 8),
                                renderer, obj, perd, np.arange(-12, 12.1, 2.0))
    S0 = shift_static_pose(S0, np.arange(0, first_contact), renderer, obj, perd,
                           Tcw, Twc, K)
    print(f'initial rest anchor/refinement: frame {a0}', flush=True)

    cand1 = np.arange(last_contact + 1, F, 3)
    tgt1 = np.arange(last_contact + 1, F, 3)
    a1 = best_anchor(Tbase, cand1, tgt1, renderer, obj, perd)
    S1 = Tbase[a1].copy()
    S1 = refine_static_rotation(S1, np.arange(last_contact + 1, F),
                                np.arange(last_contact + 1, F, 5),
                                renderer, obj, perd, np.arange(-12, 12.1, 2.0))
    S1 = shift_static_pose(S1, np.arange(last_contact + 1, F), renderer, obj, perd,
                           Tcw, Twc, K)
    print(f'final rest anchor/refinement: frame {a1}', flush=True)

    # Extend the initial rest while its pooled pose still explains the mask.
    r0 = renderer.masks(S0[None])[0]
    ious0 = []
    for i in range(0, min(F, first_contact + 60)):
        ious0.append(iou(r0 & ~perd[i], obj[i] & ~perd[i]))
    ious0 = np.asarray(ious0)
    low = np.where((np.arange(len(ious0)) >= first_contact) & (ious0 < 0.94))[0]
    s = int(low[0]) if len(low) else first_contact
    s = max(s, first_contact)

    # Earliest sustained high-IoU run at the end is the final rest phase.
    r1 = renderer.masks(S1[None])[0]
    ious1 = np.empty(F, dtype=np.float64)
    for i in range(max(0, last_contact - 90), F):
        ious1[i] = iou(r1 & ~perd[i], obj[i] & ~perd[i])
    e = last_contact + 1
    for i in range(max(0, last_contact - 90), last_contact + 1):
        if i + 8 <= F and np.all(ious1[i:i + 8] > 0.95):
            e = i
            break
    print(f'rest phases extended to [0,{s}) and [{e},{F})', flush=True)

    # ---- robust trajectory initialisation --------------------------------
    T = robust_filter_pose(Tbase)
    T[:s] = S0
    T[e:] = S1
    T = centroid_refine(T, np.arange(s, e), renderer, obj, perd, Tcw, Twc, K,
                        iterations=5)
    T[:s] = S0
    T[e:] = S1

    # ---- smooth multi-frame orientation decoding -------------------------
    T = rotation_viterbi(T, s, e, np.arange(-36, 36.1, 6.0),
                         renderer, obj, perd)
    T = rotation_viterbi(T, s, e, np.arange(-6, 6.1, 1.0),
                         renderer, obj, perd)
    T = rotation_viterbi(T, s, e, np.arange(-3, 3.01, 0.5),
                         renderer, obj, perd)

    # ---- textured spin recovery ------------------------------------------
    mat = mesh.visual.material
    tex_img = np.asarray(mat.baseColorTexture.convert('RGB'), dtype=np.uint8)
    tex = torch.tensor(tex_img, dtype=torch.float32, device=DEVICE)[None] / 255.0
    uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=DEVICE)
    uv[:, 1] = 1.0 - uv[:, 1]
    tex_tensors = {'tex': tex, 'uv': uv,
                   'uv_idx': torch.tensor(mesh.faces, dtype=torch.int32, device=DEVICE)}
    T = spin_viterbi(T, s, e, renderer, obj, per, K, tex_tensors)

    # Blend rest/manipulation transitions over 8 frames.  This removes the
    # residual depth jump at pickup/release without changing the rest phases.
    T = blend_rest_boundaries(T, s, e, S0, S1, width=8)
    T[:s] = S0
    T[e:] = S1

    # Final kinematics cleanup: keep rotation matrices orthonormal and enforce
    # float64 output expected by the evaluator.
    for i in range(F):
        u, _, vt = np.linalg.svd(T[i, :3, :3])
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt
        T[i, :3, :3] = R
        T[i, 3] = [0, 0, 0, 1]
    T = T.astype(np.float64)
    assert T.shape == (F, 4, 4) and np.isfinite(T).all()
    os.makedirs(Path(args.out).parent, exist_ok=True)
    np.savez(args.out, T_world_mesh=T)
    print(f'wrote {args.out}; elapsed {time.time() - t_start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
