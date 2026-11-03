#!/usr/bin/env python3
"""Trajectory recovery for the big black box episode.

The released reconstruction is used as the motion/rotation prior.  This solver
adds the temporal structure that a per-frame tracker misses:

* two visually verified rest phases are represented by one pooled rigid pose;
* rotations are averaged in SO(3) on each rest phase and gently smoothed while
  the object is manipulated;
* translations at sparse, deterministic keyframes are re-registered against
  the actual image silhouette (Canny edges + SAM object mask), pooling evidence
  over the static phases and interpolating a smooth absolute trajectory while
  manipulated.

Only files below --inputs and the object mesh supplied there are required.
No network access, random sampling, or evaluation annotations are used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt, median_filter
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# Scene boundaries were checked on the video and against the baseline speed
# profile.  The box is untouched through frame 159, is manipulated from 160
# through 899, and is again at rest from frame 900 onward (0-based indices).
STATIC_PHASES = ((0, 160), (900, None))
MOTION_PHASE = (160, 900)

# Deterministic keyframe spacing.  The motion is slow relative to 30 fps, so
# 10-frame spacing (1/3 s) captures its accelerations while suppressing the
# baseline's frame-wise depth noise.  Static evidence is taken only from frames
# where the person does not cover the box's silhouette.
MOTION_KEY_STEP = 10
STATIC_KEYFRAMES = {
    0: tuple(range(0, 100, 10)),
    1: tuple(range(940, 1003, 10)),
}


def load_inputs(root: Path):
    with open(root / "camera.json", "r", encoding="utf-8") as f:
        camera = json.load(f)
    K = np.asarray(camera["K"], dtype=np.float64)
    T_world_cam = np.asarray(camera["T_world_cam"], dtype=np.float64)
    T_cam_world = np.linalg.inv(T_world_cam)

    rec = np.load(root / "reconstruction.npz")
    T_base = np.asarray(rec["T_world_mesh"], dtype=np.float64)
    F = len(T_base)

    with h5py.File(root / "masks.h5", "r") as h:
        object_masks = h["object"][:]
        person_masks = h["person"][:]

    scene = trimesh.load(root / "object_mesh.glb", force="scene", process=False)
    mesh = scene.to_geometry()
    # A fixed, dense vertex subset is sufficient for the convex silhouette and
    # keeps the automatic registration inexpensive and deterministic.
    vertices = np.asarray(mesh.vertices[::50], dtype=np.float64)
    return K, T_world_cam, T_cam_world, T_base, object_masks, person_masks, vertices


def chordal_rotation_mean(rotations: np.ndarray) -> np.ndarray:
    """SO(3) chordal mean via SVD projection."""
    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def project_contour(vertices, R_mw, t_mw, K, T_cam_world):
    """Project the mesh's convex silhouette into the image."""
    vc = (vertices @ R_mw.T + t_mw) @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    z = vc[:, 2]
    uv = vc[:, :2] / z[:, None]
    uv[:, 0] = K[0, 0] * uv[:, 0] + K[0, 2]
    uv[:, 1] = K[1, 1] * uv[:, 1] + K[1, 2]
    hull = cv2.convexHull(uv.astype(np.float32))
    return hull


def contour_mask(hull, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(hull).astype(np.int32)], 1)
    return mask.astype(bool)


def image_evidence(frame_bgr, object_mask, person_mask):
    """Build evidence used by the silhouette translation registration."""
    visible = object_mask & ~person_mask
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 80) > 0
    near = cv2.dilate(visible.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    edges = edges & near & ~person_mask
    edge_distance = distance_transform_edt(~edges)
    return visible, edge_distance


def refine_translation(
    R_mw: np.ndarray,
    t_prior: np.ndarray,
    visible: np.ndarray,
    edge_distance: np.ndarray,
    vertices: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
):
    """Re-register translation while holding the supplied rotation fixed.

    The cost combines outer-contour alignment to real image edges with
    silhouette agreement against the SAM mask.  A weak prior keeps ambiguous,
    heavily occluded keyframes close to the calibrated baseline.
    """
    height, width = visible.shape
    first = np.argwhere(visible)[0]
    # This is only to avoid OpenCV's contour ordering dependence in tie cases.
    del first

    def objective(delta):
        t = t_prior + delta
        hull = project_contour(vertices, R_mw, t, K, T_cam_world)
        pred = contour_mask(hull, width, height)
        contours, _ = cv2.findContours(
            pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contour = np.zeros_like(pred, dtype=np.uint8)
        cv2.drawContours(contour, contours, -1, 1, 1)
        contour = contour.astype(bool)
        contour_cost = float(edge_distance[contour].mean()) if contour.any() else 30.0
        union = np.count_nonzero(pred | visible)
        iou = np.count_nonzero(pred & visible) / max(union, 1)
        prior = 10.0 * float(delta @ delta)
        return contour_cost + 10.0 * (1.0 - iou) + prior

    result = minimize(
        objective,
        np.zeros(3, dtype=np.float64),
        method="Powell",
        bounds=((-0.20, 0.20), (-0.20, 0.20), (-0.30, 0.30)),
        options={"maxiter": 60, "xtol": 1e-4, "ftol": 1e-5},
    )
    delta = np.asarray(result.x, dtype=np.float64)
    # Reject catastrophic registrations; in that case the calibrated baseline
    # is the safer evidence source.
    if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) > 0.32:
        delta = np.zeros(3, dtype=np.float64)
    return t_prior + delta


def quaternion_smooth(rotations: np.ndarray, start: int, stop: int):
    """Smooth a motion segment in quaternion space with sign continuity."""
    q = Rotation.from_matrix(rotations).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] *= -1.0
    seg = q[start:stop].copy()
    if len(seg) >= 7:
        seg = savgol_filter(seg, window_length=7, polyorder=2, axis=0, mode="interp")
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    q[start:stop] = seg
    return Rotation.from_quat(q).as_matrix()


def quat_slerp(q0, q1, alpha):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = (1.0 - alpha) * q0 + alpha * q1
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(theta)
        q = np.sin((1.0 - alpha) * theta) / s * q0 + np.sin(alpha * theta) / s * q1
    return q / np.linalg.norm(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    started = time.time()
    np.random.seed(0)
    cv2.setRNGSeed(0)
    cv2.setNumThreads(2)

    K, T_world_cam, T_cam_world, T_base, object_masks, person_masks, vertices = load_inputs(
        args.inputs
    )
    F = len(T_base)

    # Pooled rest-phase rotations.  This removes frame-wise tracker noise from
    # long intervals in which a rigid object cannot change pose.
    static_rotations = []
    static_translations = []
    for a, b in STATIC_PHASES:
        b_eff = F if b is None else b
        static_rotations.append(chordal_rotation_mean(T_base[a:b_eff, :3, :3]))
        static_translations.append(np.median(T_base[a:b_eff, :3, 3], axis=0))
    static_rotations = np.asarray(static_rotations)
    static_translations = np.asarray(static_translations)

    # Smooth the tracked rotations only during manipulation.  Static rotations
    # are not filtered: they are replaced by the exact pooled rigid pose.
    R_result = np.array(T_base[:, :3, :3], copy=True)
    m0, m1 = MOTION_PHASE
    R_result = quaternion_smooth(R_result, m0, m1)
    q_sm = Rotation.from_matrix(R_result).as_quat()
    for i in range(1, F):
        if np.dot(q_sm[i], q_sm[i - 1]) < 0.0:
            q_sm[i] *= -1.0
    q_sm = Rotation.from_quat(q_sm).as_matrix()
    R_result = q_sm

    # Rest poses are exact and constant.
    a0, b0 = STATIC_PHASES[0]
    R_result[a0:b0] = static_rotations[0]
    a1, _ = STATIC_PHASES[1]
    R_result[a1:F] = static_rotations[1]

    # Gently blend smoothed motion rotations toward the pooled rest anchors at
    # the two transition boundaries.
    qm = Rotation.from_matrix(R_result).as_quat()
    for i in range(1, F):
        if np.dot(qm[i], qm[i - 1]) < 0.0:
            qm[i] *= -1.0
    q0 = Rotation.from_matrix(static_rotations[0]).as_quat()
    q1 = Rotation.from_matrix(static_rotations[1]).as_quat()
    for i in range(m0, min(m0 + 10, m1)):
        alpha = (i - m0) / 10.0
        qm[i] = quat_slerp(q0, qm[i], alpha)
    for i in range(max(m0, m1 - 10), m1):
        alpha = (m1 - i) / 10.0
        qm[i] = quat_slerp(q1, qm[i], alpha)
    R_result = Rotation.from_quat(qm).as_matrix()

    # Collect independently useful keyframes and re-register translation.
    motion_keys = list(range(m0, m1 + 1, MOTION_KEY_STEP))
    if motion_keys[-1] != m1:
        motion_keys.append(m1)
    all_evidence_frames = set(motion_keys)
    for frames in STATIC_KEYFRAMES.values():
        all_evidence_frames.update(frames)

    cap = cv2.VideoCapture(str(args.inputs / "video.mp4"))
    if not cap.isOpened():
        raise RuntimeError("could not open video.mp4")

    t_motion_prior = {}
    t_motion_refined = {}
    t_static_samples = [[], []]
    last = -1
    for frame_idx in range(F):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"video ended before frame {frame_idx}")
        if frame_idx not in all_evidence_frames:
            continue

        visible, edge_dist = image_evidence(
            frame, object_masks[frame_idx], person_masks[frame_idx]
        )
        if visible.sum() < 500:
            continue

        if frame_idx <= b0:
            phase = 0
            R_use = static_rotations[0]
        elif frame_idx >= a1:
            phase = 1
            R_use = static_rotations[1]
        else:
            phase = -1
            R_use = R_result[frame_idx]

        t_prior = (
            static_translations[phase]
            if phase >= 0
            else np.asarray(T_base[frame_idx, :3, 3], dtype=np.float64)
        )
        t_ref = refine_translation(
            R_use,
            t_prior,
            visible,
            edge_dist,
            vertices,
            K,
            T_cam_world,
        )
        if phase >= 0:
            t_static_samples[phase].append(t_ref)
        else:
            t_motion_prior[frame_idx] = t_prior
            t_motion_refined[frame_idx] = t_ref
        last = frame_idx
    cap.release()
    if last < 0:
        raise RuntimeError("no usable object masks")

    # Robust pooling of repeated static evidence.  The same world pose should
    # explain every rest frame, so per-frame registration noise is discarded.
    static_t_final = []
    for phase in (0, 1):
        samples = np.asarray(t_static_samples[phase], dtype=np.float64)
        if len(samples) == 0:
            static_t_final.append(static_translations[phase])
        else:
            static_t_final.append(np.median(samples, axis=0))
    static_t_final = np.asarray(static_t_final)

    # Absolute translation keys over the manipulated segment.
    motion_times = np.asarray([m0] + sorted(t_motion_refined) + [m1], dtype=np.float64)
    motion_values = np.vstack(
        [static_t_final[0]]
        + [t_motion_refined[int(k)] for k in sorted(t_motion_refined)]
        + [static_t_final[1]]
    ).astype(np.float64)

    # Remove isolated registration failures before interpolation.
    if len(motion_values) >= 5:
        motion_values = median_filter(motion_values, size=(3, 1), mode="nearest")
        if len(motion_values) >= 7:
            motion_values = savgol_filter(
                motion_values, window_length=5, polyorder=2, axis=0, mode="interp"
            )

    T_result = np.repeat(np.eye(4, dtype=np.float64)[None], F, axis=0)
    T_result[:, :3, :3] = R_result

    # Rest intervals use one rigid pose, not a jittering filtered trajectory.
    T_result[0:b0, :3, 3] = static_t_final[0]
    T_result[a1:F, :3, 3] = static_t_final[1]

    # PCHIP gives a deterministic C1 trajectory through the robustified keys.
    for axis in range(3):
        interp = PchipInterpolator(motion_times, motion_values[:, axis], extrapolate=True)
        T_result[m0:m1, axis, 3] = interp(np.arange(m0, m1))
    # Exact continuity at both rest boundaries.
    T_result[m0, :3, 3] = static_t_final[0]
    T_result[m1, :3, 3] = static_t_final[1]

    # Finite, right-handed transforms.
    for i in range(F):
        u, _, vt = np.linalg.svd(T_result[i, :3, :3])
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt
        T_result[i, :3, :3] = R
        T_result[i, 3] = (0.0, 0.0, 0.0, 1.0)
    if not np.isfinite(T_result).all():
        raise RuntimeError("non-finite pose generated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, T_world_mesh=T_result.astype(np.float64))
    elapsed = time.time() - started
    print(f"wrote {args.out} with {F} poses in {elapsed:.1f}s")
    print("static translations:", static_t_final.tolist())
    print("motion translation key count:", len(t_motion_refined))


if __name__ == "__main__":
    main()
