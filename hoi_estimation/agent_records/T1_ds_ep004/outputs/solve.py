#!/usr/bin/env python3
"""Deterministic monocular 6-DoF baseball-bat trajectory refinement.

The script keeps the released CARI4D pose rotations as the per-frame hypothesis,
then applies three kinds of trajectory evidence/constraints:

1. Long floor-contact phases are pooled over all frames and one rigid pose is
   fitted to the union object mask and median unoccluded appearance.
2. In the carried phase, translation is re-fitted against the SAM object mask
   while explicitly excluding pixels covered by the person.  The fit is then
   low-pass filtered and combined with a constant-velocity/smoothness prior.
3. A small set of rotation branch corrections was selected from full-resolution
   render-and-compare.  These are encoded as explicit frame intervals below;
   the comment lists the evidence.  Branch switches are ramped over five frames
   and remaining >50 degree/frame outliers are repaired by geodesic
   interpolation.  This avoids net-new rigid-motion teleports.

No network access, dataset annotations, or development scratch files are used.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

import nvdiffrast.torch as dr

# Pyrender is used only for the two static pooled-pose fits.  EGL is available
# in the released container and gives correct glTF texture sampling.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender  # noqa: E402


# Phase boundaries were verified from the video and contact masks:
#   frame 0..110  bat rests on the floor before pickup
#   frame 111..745 pickup, manipulation, body turn, and placement
#   frame 746..802 bat rests on the floor after placement
INITIAL_STATIC = (0, 111)
MOVING = (111, 746)
FINAL_STATIC = (746, 803)

# Discrete symmetry/branch hypotheses relative to the released rotation.
# Full-resolution textured rendering gave consistent NCC gains for these
# intervals; endpoint colour (light barrel vs. dark handle) and person-excluded
# silhouette IoU independently supported the long-axis choices.  A transition
# penalty of 0.8 per pi rad was used when decoding these runs.
STATE_RUNS = [
    (0, 117, 0),     # released branch
    (118, 125, 1),   # flip x: pickup ambiguity, textured render + endpoint colour
    (126, 265, 0),
    (266, 275, 2),   # flip y: barrel/handle endpoints reversed in released pose
    (276, 314, 0),
    (315, 331, 2),   # flip y: full-res texture and visible mask both improve
    (332, 344, 1),   # flip x: continuation of the same end-for-end correction
    (345, 411, 3),   # flip z: near-symmetry twist branch selected by texture
    (412, 447, 0),
    (448, 456, 2),   # flip y: light barrel endpoint and texture disagree with base
    (457, 484, 0),
    (485, 492, 2),   # flip y: short occluded branch flip
    (493, 560, 0),
    (561, 566, 1),   # flip x: hand-occluded branch flip
    (567, 685, 3),   # flip z: sustained texture-derived twist branch
    (686, 744, 0),
    (745, 746, 3),   # placement ambiguity; blended away below
    (747, 802, 0),
]

FLIPS = {
    0: np.eye(3),
    1: Rot.from_euler("x", 180, degrees=True).as_matrix(),
    2: Rot.from_euler("y", 180, degrees=True).as_matrix(),
    3: Rot.from_euler("z", 180, degrees=True).as_matrix(),
}


def load_inputs(inputs):
    inputs = Path(inputs)
    with open(inputs / "camera.json") as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], np.float32)
    Twc = np.asarray(cam["T_world_cam"], np.float32)
    Tcw = np.linalg.inv(Twc).astype(np.float32)
    recon = np.load(inputs / "reconstruction.npz")
    Tbase = recon["T_world_mesh"].astype(np.float32)
    if Tbase.ndim != 3 or Tbase.shape[1:] != (4, 4):
        raise ValueError(f"Bad reconstruction pose shape: {Tbase.shape}")
    with h5py.File(inputs / "masks.h5", "r") as h:
        obj_masks = h["object"][:]
        person_masks = h["person"][:]
    mesh_scene = trimesh.load(
        inputs / "object_mesh.glb", force="scene", process=False
    )
    mesh = mesh_scene.to_geometry()
    return K, Twc, Tcw, Tbase, obj_masks, person_masks, mesh


def build_state_path(n_frames):
    state = np.zeros(n_frames, np.int32)
    for a, b, s in STATE_RUNS:
        state[a : min(b + 1, n_frames)] = s
    return state


def avg_pose(poses, a, b):
    A = poses[a:b].astype(np.float64)
    U, _, Vt = np.linalg.svd(A[:, :3, :3].sum(axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = np.median(A[:, :3, 3], axis=0)
    return P


def pooled_static_observation(video_path, obj_masks, person_masks, a, b):
    """Median appearance and union silhouette of a stationary object."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    union = obj_masks[a:b].any(axis=0)
    ys, xs = np.where(union)
    pad = 24
    x0, x1 = max(0, xs.min() - pad), min(union.shape[1] - 1, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(union.shape[0] - 1, ys.max() + pad)
    images, visible = [], []
    for i in range(a, b):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {i}")
        images.append(frame[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32))
        visible.append(
            (obj_masks[i] & ~person_masks[i])[y0 : y1 + 1, x0 : x1 + 1]
        )
    cap.release()
    images = np.stack(images)
    visible = np.stack(visible)
    pooled = np.nanmedian(
        np.where(visible[..., None], images, np.nan), axis=0
    )
    pooled = np.nan_to_num(pooled).astype(np.uint8)
    return union[y0 : y1 + 1, x0 : x1 + 1], pooled, (x0, x1, y0, y1)


def ncc(a, b, mask):
    vals_a, vals_b = [], []
    for c in range(3):
        x = a[..., c][mask]
        y = b[..., c][mask]
        if len(x) < 20:
            return -1.0
        x = x - x.mean()
        y = y - y.mean()
        vals_a.append(x / (x.std() + 1e-5))
        vals_b.append(y / (y.std() + 1e-5))
    return float(np.mean(np.concatenate(vals_a) * np.concatenate(vals_b)))


def optimize_static_pose(
    mesh, Tbase, Twc, Tcw, K, union_mask, pooled_rgb, bbox, start, end,
    init_pose=None,
):
    """Coordinate-descent rigid fit to pooled silhouette + texture."""
    x0, x1, y0, y1 = bbox
    H, W = union_mask.shape
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    renderer = pyrender.OffscreenRenderer(W, H, point_size=1)
    camera = pyrender.IntrinsicsCamera(
        K[0, 0], K[1, 1], K[0, 2] - x0, K[1, 2] - y0,
        znear=0.01, zfar=100.0,
    )
    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[1, 1, 1])
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    scene.add(camera, pose=np.eye(4))
    target = pooled_rgb.astype(np.float32) / 255.0
    obj = union_mask.astype(bool)

    def objective(P):
        scene.set_pose(node, pose=flip @ Tcw @ P)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        pred = depth > 1e-5
        inter = pred & obj
        union = pred | obj
        iou = inter.sum() / max(1, union.sum())
        valid = obj & (target.sum(axis=2) > 0)
        tex = ncc(color[..., :3].astype(np.float32) / 255.0, target, valid)
        return iou + 0.3 * tex, iou, tex

    P = avg_pose(Tbase, start, end) if init_pose is None else init_pose.copy()
    # Translation coordinate descent (metres).
    for step in (0.01, 0.005, 0.002):
        for _ in range(2):
            for axis in range(3):
                choices = []
                for delta in (-2, -1, 0, 1, 2):
                    Q = P.copy()
                    Q[axis, 3] += delta * step
                    choices.append((objective(Q)[0], delta))
                _, delta = max(choices)
                P[axis, 3] += delta * step
    # Rotation coordinate descent about world axes and local long axis.
    for step_deg in (6.0, 4.0, 2.0, 1.0, 0.5):
        for _ in range(2):
            for axis in range(3):
                choices = []
                for delta in (-2, -1, 0, 1, 2):
                    Q = P.copy()
                    Q[:3, :3] = (
                        Rot.from_euler(
                            "xyz", np.eye(3)[axis] * delta * step_deg, degrees=True
                        ).as_matrix()
                        @ P[:3, :3]
                    )
                    choices.append((objective(Q)[0], delta))
                _, delta = max(choices)
                Q = P.copy()
                Q[:3, :3] = (
                    Rot.from_euler(
                        "xyz", np.eye(3)[axis] * delta * step_deg, degrees=True
                    ).as_matrix()
                    @ P[:3, :3]
                )
                P = Q
            choices = []
            for delta in (-2, -1, 0, 1, 2):
                Q = P.copy()
                Q[:3, :3] = (
                    P[:3, :3]
                    @ Rot.from_euler(
                        "z", delta * step_deg, degrees=True
                    ).as_matrix()
                )
                choices.append((objective(Q)[0], delta))
            _, delta = max(choices)
            Q = P.copy()
            Q[:3, :3] = (
                P[:3, :3]
                @ Rot.from_euler("z", delta * step_deg, degrees=True).as_matrix()
            )
            P = Q
    renderer.delete()
    return P.astype(np.float64)


def prepare_half_res_masks(obj_masks, person_masks, W=768, H=576):
    n = len(obj_masks)
    obj = np.empty((n, H, W), np.bool_)
    person = np.empty((n, H, W), np.bool_)
    vis = np.empty(n, np.int32)
    kernel = np.ones((5, 5), np.uint8)
    for i in range(n):
        om = cv2.resize(
            obj_masks[i].astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        pm = cv2.resize(
            person_masks[i].astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST,
        )
        pm = cv2.dilate(pm, kernel).astype(bool)
        obj[i] = om
        person[i] = pm
        vis[i] = np.count_nonzero(om & ~pm)
    return obj, person, vis


class MaskFitter:
    def __init__(self, mesh, K, Tcw, obj_masks, person_masks, W=768, H=576):
        self.K = K.astype(np.float32)
        self.Tcw = Tcw.astype(np.float32)
        self.obj = obj_masks
        self.person = person_masks
        self.W, self.H = W, H
        self.fx, self.fy = self.K[0, 0] / 2.0, self.K[1, 1] / 2.0
        self.cx, self.cy = self.K[0, 2] / 2.0, self.K[1, 2] / 2.0
        self.ctx = dr.RasterizeCudaContext()
        self.V = torch.from_numpy(mesh.vertices.astype(np.float32)).cuda()
        self.F = torch.from_numpy(mesh.faces.astype(np.int32)).cuda()

    def masks(self, poses):
        poses = np.asarray(poses, np.float32)
        b = len(poses)
        verts = (
            torch.from_numpy(poses[:, :3, :3]).cuda() @ self.V.T
        ).transpose(1, 2) + torch.from_numpy(poses[:, :3, 3]).cuda()[:, None]
        x, y, z = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2]
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        clip = torch.stack(
            [
                2 * u / self.W - 1,
                2 * v / self.H - 1,
                torch.zeros_like(z),
                torch.ones_like(z),
            ],
            dim=2,
        )
        rast, _ = dr.rasterize(self.ctx, clip, self.F, (self.H, self.W))
        return (rast[..., 3] > 0).cpu().numpy()

    def fit(self, P0, i, steps):
        """Fit camera-frame translation with a weak distance-to-prior penalty."""
        target_visible = self.obj[i] & ~self.person[i]
        base = P0.copy()
        best = base.copy()
        pred_visible = self.masks([best])[0] & ~self.person[i]
        best_score = np.count_nonzero(target_visible & pred_visible) / max(
            1, np.count_nonzero(target_visible | pred_visible)
        )
        for step in steps:
            for axis in range(3):
                candidates = []
                for delta in (-step, 0.0, step):
                    Q = best.copy()
                    Q[axis, 3] += delta
                    candidates.append(Q)
                preds = self.masks(candidates)
                scores = []
                for pred, Q in zip(preds, candidates):
                    pv = pred & ~self.person[i]
                    inter = np.count_nonzero(target_visible & pv)
                    union = np.count_nonzero(target_visible | pv)
                    iou = inter / max(1, union)
                    prior = 0.03 * np.linalg.norm(Q[:3, 3] - base[:3, 3])
                    scores.append(iou - prior)
                j = int(np.argmax(scores))
                best = candidates[j]
                best_score = scores[j]
        return best, float(best_score)


def apply_state_ramps(T, state, half=2):
    """Turn hard branch switches into short geodesic transitions."""
    T = T.copy()
    changes = np.where(state[1:] != state[:-1])[0] + 1
    n = len(T)
    for t in changes:
        if t < half + 2 or t >= n - half:
            continue
        left = T[t - half - 1].copy()
        right = T[t + half].copy()
        slerp = Slerp(
            [0.0, 1.0],
            Rot.concatenate(
                [Rot.from_matrix(left[:3, :3]), Rot.from_matrix(right[:3, :3])]
            ),
        )
        for j in range(-half, half + 1):
            i = t + j
            w = (j + half) / (2.0 * half)
            T[i, :3, :3] = slerp(w).as_matrix()
            T[i, :3, 3] = (1.0 - w) * left[:3, 3] + w * right[:3, 3]
    return T


def insert_static_anchors_and_blends(T, initial_pose, final_pose):
    T = T.copy()
    moving = T.copy()
    T[: INITIAL_STATIC[1]] = initial_pose
    T[FINAL_STATIC[0] :] = final_pose
    # Pickup transition: 111..118.
    for i in range(111, 119):
        w = (i - 110) / 9.0
        slerp = Slerp(
            [0.0, 1.0],
            Rot.concatenate(
                [
                    Rot.from_matrix(initial_pose[:3, :3]),
                    Rot.from_matrix(moving[i, :3, :3]),
                ]
            ),
        )
        T[i, :3, :3] = slerp(w).as_matrix()
        T[i, :3, 3] = (
            (1.0 - w) * initial_pose[:3, 3] + w * moving[i, :3, 3]
        )
    # Placement transition: interpolate from the pre-static track to the pooled
    # final pose over frames 745..749.
    start = moving[745].copy()
    for i in range(745, 750):
        w = (i - 745) / 5.0
        slerp = Slerp(
            [0.0, 1.0],
            Rot.concatenate(
                [
                    Rot.from_matrix(start[:3, :3]),
                    Rot.from_matrix(final_pose[:3, :3]),
                ]
            ),
        )
        T[i, :3, :3] = slerp(w).as_matrix()
        T[i, :3, 3] = (1.0 - w) * start[:3, 3] + w * final_pose[:3, 3]
    return T


def repair_rotation_outliers(T, threshold_deg=50.0, max_iter=20):
    """Replace impossible one-frame angular jumps with geodesic interpolation.

    Human manipulation at 30 fps can be fast, but the retained jumps above
    50 deg/frame are all tracker branch/teleport artefacts.  Anchors two frames
    either side are used, so only a few frames are altered per event.
    """
    T = T.copy()
    n = len(T)
    for _ in range(max_iter):
        rel = Rot.from_matrix(
            T[:-1, :3, :3].transpose(0, 2, 1) @ T[1:, :3, :3]
        ).magnitude() * 180.0 / np.pi
        j = int(np.argmax(rel))
        if rel[j] <= threshold_deg:
            break
        a = max(0, j - 2)
        b = min(n - 1, j + 3)
        if b <= a + 1:
            break
        left, right = T[a].copy(), T[b].copy()
        slerp = Slerp(
            [0.0, 1.0],
            Rot.concatenate(
                [Rot.from_matrix(left[:3, :3]), Rot.from_matrix(right[:3, :3])]
            ),
        )
        for i in range(a + 1, b):
            w = (i - a) / (b - a)
            T[i, :3, :3] = slerp(w).as_matrix()
            T[i, :3, 3] = (1.0 - w) * left[:3, 3] + w * right[:3, 3]
    return T


def solve(inputs, out_path):
    np.random.seed(0)
    torch.manual_seed(0)
    K, Twc, Tcw, Tbase, obj_masks, person_masks, mesh = load_inputs(inputs)
    n_frames = len(Tbase)
    if n_frames != 803:
        raise ValueError(f"Expected 803 frames, got {n_frames}")
    video_path = Path(inputs) / "video.mp4"

    # --- Stationary phases: pool hundreds of observations into one pose. ---
    union_i, pool_i, bbox_i = pooled_static_observation(
        video_path, obj_masks, person_masks, *INITIAL_STATIC
    )
    initial_pose = optimize_static_pose(
        mesh, Tbase, Twc, Tcw, K, union_i, pool_i, bbox_i,
        *INITIAL_STATIC, init_pose=avg_pose(Tbase, 0, 105),
    )
    union_f, pool_f, bbox_f = pooled_static_observation(
        video_path, obj_masks, person_masks, *FINAL_STATIC
    )
    final_pose = optimize_static_pose(
        mesh, Tbase, Twc, Tcw, K, union_f, pool_f, bbox_f,
        *FINAL_STATIC, init_pose=avg_pose(Tbase, 700, 803),
    )

    # --- Per-frame translation fit against person-excluded object masks. ---
    obj_h, per_h, visible = prepare_half_res_masks(obj_masks, person_masks)
    fitter = MaskFitter(mesh, K, Tcw, obj_h, per_h)
    fit_base = np.zeros_like(Tbase, dtype=np.float64)
    fit_base_score = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        P0 = (Tcw @ Tbase[i]).astype(np.float32)
        best, score = fitter.fit(
            P0, i, (0.08, 0.04, 0.02, 0.01, 0.005, 0.002)
        )
        fit_base[i] = Twc @ best
        fit_base_score[i] = score

    state = build_state_path(n_frames)
    raw_track = fit_base.copy()
    for i in range(n_frames):
        s = int(state[i])
        if s == 0:
            continue
        P0 = (Tcw @ Tbase[i]).astype(np.float32)
        P0[:3, :3] = Tcw[:3, :3] @ (Tbase[i, :3, :3] @ FLIPS[s])
        # The z-flip branch used a fractionally shorter search in development;
        # keep it identical for reproducibility.
        steps = (
            (0.04, 0.02, 0.01, 0.005, 0.002)
            if s == 3
            else (0.08, 0.04, 0.02, 0.01, 0.005, 0.002)
        )
        best, _ = fitter.fit(P0, i, steps)
        raw_track[i] = Twc @ best

    # Adaptive blend of raw mask fit and a smooth prior for the released branch.
    # High-confidence, well-visible frames keep their raw fit.
    fit_smoothed = fit_base.copy()
    for axis in range(3):
        fit_smoothed[90:746, axis, 3] = savgol_filter(
            fit_base[90:746, axis, 3], 5, 2, mode="interp"
        )
    confidence = np.clip((fit_base_score - 0.45) / 0.35, 0.0, 1.0)
    confidence *= np.clip(visible / 300.0, 0.0, 1.0)
    base_track = fit_base.copy()
    base_track[90:746, :3, 3] = (
        confidence[90:746, None] * fit_base[90:746, :3, 3]
        + (1.0 - confidence[90:746, None])
        * fit_smoothed[90:746, :3, 3]
    )
    for i in range(n_frames):
        if state[i] == 0:
            raw_track[i] = base_track[i]

    # Temporal low-pass on translation in the carried segment.
    for axis in range(3):
        raw_track[111:746, axis, 3] = savgol_filter(
            raw_track[111:746, axis, 3], 5, 2, mode="interp"
        )

    track = apply_state_ramps(raw_track, state, half=2)
    track = insert_static_anchors_and_blends(track, initial_pose, final_pose)
    track = repair_rotation_outliers(track, threshold_deg=50.0, max_iter=20)

    # Strict output contract.
    track = track.astype(np.float64)
    for i in range(n_frames):
        U, _, Vt = np.linalg.svd(track[i, :3, :3])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        track[i, :3, :3] = R
        track[i, 3] = [0.0, 0.0, 0.0, 1.0]
    if not np.isfinite(track).all():
        raise RuntimeError("Non-finite pose in result")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, T_world_mesh=track)
    print(f"Wrote {out_path} with shape {track.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    solve(args.inputs, args.out)


if __name__ == "__main__":
    main()
