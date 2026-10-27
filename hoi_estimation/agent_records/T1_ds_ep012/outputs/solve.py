#!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF world trajectory for one rigid object.

The released baseline contains useful per-frame appearance evidence but has
tracker jitter and depth drift while the object is carried.  This script:

  1. pools the two static phases into one rigid pose per phase;
  2. smoothly interpolates the baseline motion (SE(3)-compatible quaternion
     smoothing + translation smoothing);
  3. aligns the projected mesh silhouette to the SAM2 object mask, excluding
     pixels covered by the person, to remove 2-D trajectory drift;
  4. searches per-frame metric depth along the camera ray, using a mask eroded
     by one pixel to compensate SAM2's approximately one-pixel dilation;
  5. tests small local rotation corrections on a sparse set of anchors and
     applies the resulting smooth rotation curve.

Only the supplied inputs are read.  Rendering is deterministic and uses the
installed nvdiffrast rasterizer.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr


# Phase boundaries were verified by frame differencing inside the visible
# object region (excluding the person): the image is static from frames 0-195
# (median consecutive abs-difference < 1 grayscale level), begins moving at
# frame 196, settles at frame 582, and remains static through frame 699.
INITIAL_STATIC_END = 196       # exclusive
FINAL_STATIC_START = 582       # inclusive
# Pooling windows avoid the periods when the approaching/departing person
# occludes much of the object while retaining all clean, unambiguous frames.
INITIAL_POOL = (20, 150)
FINAL_POOL = (590, 700)


class MaskRenderer:
    """Small nvdiffrast wrapper returning a binary silhouette."""

    def __init__(self, mesh_path: str, width: int, height: int,
                 device: str = "cuda") -> None:
        self.device = device
        self.width = int(width)
        self.height = int(height)
        mesh = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
        self.vertices = torch.as_tensor(
            np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.faces = torch.as_tensor(
            np.asarray(mesh.faces), dtype=torch.int32, device=device)
        self.ctx = dr.RasterizeCudaContext()

    def mask(self, T_cam_mesh: np.ndarray, K: np.ndarray) -> np.ndarray:
        if isinstance(T_cam_mesh, np.ndarray):
            T_cam_mesh = torch.as_tensor(
                T_cam_mesh, dtype=torch.float32, device=self.device)
        if isinstance(K, np.ndarray):
            K = torch.as_tensor(K, dtype=torch.float32, device=self.device)
        R = T_cam_mesh[:3, :3]
        t = T_cam_mesh[:3, 3]
        vc = self.vertices @ R.T + t
        x, y, z = vc[:, 0], vc[:, 1], vc[:, 2]
        u = K[0, 0] * x + K[0, 1] * y + K[0, 2] * z
        v = K[1, 0] * x + K[1, 1] * y + K[1, 2] * z
        # OpenCV camera coordinates (x right, y down) -> nvdiffrast clip.
        clip = torch.stack(
            [2.0 * u / self.width - z,
             2.0 * v / self.height - z,
             0.5 * z,
             z], dim=-1)[None]
        rast, _ = dr.rasterize(
            self.ctx, clip, self.faces, resolution=(self.height, self.width))
        return (rast[0, :, :, 3] > 0).detach().cpu().numpy()


def mean_pose(T: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Chordal mean rotation plus robust median translation."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rot.from_matrix(T[ids, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(T[ids, :3, 3], axis=0)
    return out


def qlerp(qa: np.ndarray, qb: np.ndarray, w: float) -> np.ndarray:
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = (1.0 - w) * qa + w * qb
    return q / np.linalg.norm(q)


def apply_camera_xy_shift(T_world: np.ndarray, T_wc: np.ndarray,
                          z_camera: float, dx_full_pix: float,
                          dy_full_pix: float, K: np.ndarray) -> np.ndarray:
    """Move a world pose by the camera x/y translation that induces a desired
    image shift, keeping depth along the viewing ray unchanged to first order.
    """
    dx = -dx_full_pix * z_camera / K[0, 0]
    dy = -dy_full_pix * z_camera / K[1, 1]
    D = np.eye(4, dtype=np.float64)
    D[:3, 3] = T_wc[:3, :3] @ np.array([dx, dy, 0.0])
    return D @ T_world


def visibility_iou(pred: np.ndarray, target: np.ndarray) -> float:
    inter = np.count_nonzero(pred & target)
    union = np.count_nonzero(pred | target)
    return float(inter) / float(union) if union else 1.0


def centroid_correction(C: np.ndarray, object_half: np.ndarray,
                        person_half: np.ndarray, renderer: MaskRenderer,
                        K: np.ndarray, K_half: np.ndarray, T_wc: np.ndarray,
                        T_wc_inv: np.ndarray, start: int, stop: int,
                        sigma: float) -> np.ndarray:
    """Estimate and remove smooth 2-D silhouette-centroid drift."""
    count = stop - start
    ds = np.zeros((count, 2), dtype=np.float64)
    zs = np.zeros(count, dtype=np.float64)
    for j, i in enumerate(range(start, stop)):
        pred = renderer.mask(T_wc_inv @ C[i], K_half) & ~person_half[i]
        target = object_half[i] & ~person_half[i]
        ys, xs = np.where(pred)
        yt, xt = np.where(target)
        if xs.size and xt.size:
            # Convert half-resolution pixel offsets back to full resolution.
            ds[j] = (np.array([xs.mean() - xt.mean(),
                               ys.mean() - yt.mean()]) / 0.5)
        zs[j] = (T_wc_inv @ C[i])[2, 3]
    ds = gaussian_filter1d(ds, sigma=sigma, axis=0, mode="nearest")
    out = C.copy()
    for j, i in enumerate(range(start, stop)):
        out[i] = apply_camera_xy_shift(
            C[i], T_wc, zs[j], ds[j, 0], ds[j, 1], K)
    return out


def estimate_depth_curve(C: np.ndarray, h5: h5py.File,
                         object_half: np.ndarray, person_half: np.ndarray,
                         renderer: MaskRenderer, K: np.ndarray,
                         K_half: np.ndarray, T_wc: np.ndarray,
                         T_wc_inv: np.ndarray, start: int, stop: int,
                         step: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Find the camera-ray depth scale that best explains eroded object masks.

    A scale > 1 means move the pose toward the camera (smaller metric depth).
    SAM2 masks are about one pixel dilated, so one full-resolution erosion is
    applied before downsampling; this avoids a systematic ~2% depth bias.
    """
    scales = np.arange(1.0, 1.0501, 0.005, dtype=np.float64)
    kernel = np.ones((3, 3), np.uint8)
    anchors, best_scales = [], []
    for i in range(start, stop, step):
        obj = cv2.erode(h5["object"][i].astype(np.uint8), kernel, iterations=1)
        obj = cv2.resize(obj, (renderer.width, renderer.height),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        target = obj & ~person_half[i]
        vals = []
        for scale in scales:
            Tc = T_wc_inv @ C[i]
            Tc = Tc.copy()
            Tc[:3, 3] /= scale
            pred = renderer.mask(Tc, K_half) & ~person_half[i]
            vals.append(visibility_iou(pred, target))
        anchors.append(i)
        best_scales.append(float(scales[int(np.argmax(vals))]))
    return np.asarray(anchors, dtype=np.float64), np.asarray(best_scales)


def apply_depth_curve(C: np.ndarray, anchors: np.ndarray, scales: np.ndarray,
                      T_wc: np.ndarray, T_wc_inv: np.ndarray,
                      start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.ones(len(C), dtype=np.float64)
    s_curve = np.interp(np.arange(start, stop), anchors, scales)
    s[start:stop] = s_curve
    s = gaussian_filter1d(s, sigma=5.0, mode="nearest")
    s = np.clip(s, 1.0, 1.05)
    # Taper to the pooled static pose at both phase boundaries.
    for i in range(start, min(start + 6, stop)):
        s[i] = 1.0 + (s[i] - 1.0) * (i - start + 1) / 7.0
    for i in range(max(start, stop - 6), stop):
        s[i] = 1.0 + (s[i] - 1.0) * (stop - i) / 6.0
    out = C.copy()
    for i in range(start, stop):
        Tc = T_wc_inv @ C[i]
        Tc = Tc.copy()
        Tc[:3, 3] /= s[i]
        out[i] = T_wc @ Tc
    return out, s


def probe_rotation_corrections(C: np.ndarray, h5: h5py.File,
                               person_half: np.ndarray,
                               renderer: MaskRenderer, K_half: np.ndarray,
                               T_wc_inv: np.ndarray, start: int,
                               stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Sparse coordinate search for small local rotation errors.

    The search is deliberately coarse and sparse.  The small, smoothly
    varying correction curve it produces is much less prone to mask overfit
    than independently optimising every frame.
    """
    kernel = np.ones((3, 3), np.uint8)
    frames, deltas = [], []
    for i in range(start, stop, 20):
        obj = cv2.erode(h5["object"][i].astype(np.uint8), kernel, iterations=1)
        obj = cv2.resize(obj, (renderer.width, renderer.height),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        target = obj & ~person_half[i]
        T = C[i].copy()
        total = np.zeros(3, dtype=np.float64)
        for _ in range(2):
            for axis in range(3):
                best_iou, best_delta = -1.0, 0.0
                for delta in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    R = np.eye(4, dtype=np.float64)
                    R[:3, :3] = Rot.from_euler(
                        "xyz", np.eye(3)[axis] * delta,
                        degrees=True).as_matrix()
                    value = visibility_iou(
                        renderer.mask(T_wc_inv @ (T @ R), K_half) & ~person_half[i],
                        target)
                    if value > best_iou:
                        best_iou, best_delta = value, delta
                if best_delta != 0.0:
                    R = np.eye(4, dtype=np.float64)
                    R[:3, :3] = Rot.from_euler(
                        "xyz", np.eye(3)[axis] * best_delta,
                        degrees=True).as_matrix()
                    T = T @ R
                    total[axis] += best_delta
        frames.append(i)
        deltas.append(total)
    return np.asarray(frames, dtype=np.float64), np.asarray(deltas)


def apply_rotation_curve(C: np.ndarray, anchors: np.ndarray,
                         deltas: np.ndarray, start: int, stop: int,
                         sigma: float = 10.0) -> np.ndarray:
    angles = np.zeros((len(C), 3), dtype=np.float64)
    xx = np.concatenate(([start - 6.0], anchors, [stop + 8.0]))
    dd = np.vstack((np.zeros((1, 3)), deltas, np.zeros((1, 3))))
    for axis in range(3):
        angles[start:stop, axis] = np.interp(
            np.arange(start, stop), xx, dd[:, axis])
    angles = gaussian_filter1d(angles, sigma=sigma, axis=0, mode="nearest")
    out = C.copy()
    for i in range(start, stop):
        R = np.eye(4, dtype=np.float64)
        R[:3, :3] = Rot.from_euler("xyz", angles[i], degrees=True).as_matrix()
        out[i] = C[i] @ R
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    t_start = time.time()
    np.random.seed(0)
    torch.manual_seed(0)
    cv2.setNumThreads(1)

    inputs = os.path.abspath(args.inputs)
    with open(os.path.join(inputs, "camera.json"), "r") as f:
        camera = json.load(f)
    K = np.asarray(camera["K"], dtype=np.float64)
    T_wc = np.asarray(camera["T_world_cam"], dtype=np.float64)
    T_wc_inv = np.linalg.inv(T_wc)
    recon = np.load(os.path.join(inputs, "reconstruction.npz"))
    baseline = recon["T_world_mesh"].astype(np.float64)
    frame_count = len(baseline)

    # --- Static phases and smooth baseline trajectory ---------------------
    S0 = mean_pose(baseline, np.arange(*INITIAL_POOL))
    S1 = mean_pose(baseline, np.arange(*FINAL_POOL))

    q = Rot.from_matrix(baseline[:, :3, :3]).as_quat().astype(np.float64)
    for i in range(1, frame_count):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] *= -1
    p_smooth = gaussian_filter1d(
        baseline[:, :3, 3], sigma=2.0, axis=0, mode="nearest")
    q_smooth = gaussian_filter1d(q, sigma=2.0, axis=0, mode="nearest")
    q_smooth /= np.linalg.norm(q_smooth, axis=1, keepdims=True)

    C = np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1))
    C[:, :3, :3] = Rot.from_quat(q_smooth).as_matrix()
    C[:, :3, 3] = p_smooth
    C[:INITIAL_STATIC_END] = S0
    q0 = Rot.from_matrix(S0[:3, :3]).as_quat()
    for i in range(INITIAL_STATIC_END, INITIAL_STATIC_END + 6):
        w = (i - INITIAL_STATIC_END + 1) / 7.0
        anchor = min(INITIAL_STATIC_END + 6, frame_count - 1)
        C[i, :3, :3] = Rot.from_quat(qlerp(
            q0, Rot.from_matrix(C[anchor, :3, :3]).as_quat(), w)).as_matrix()
        C[i, :3, 3] = (1.0 - w) * S0[:3, 3] + w * C[anchor, :3, 3]
    C[FINAL_STATIC_START:] = S1
    q1 = Rot.from_matrix(S1[:3, :3]).as_quat()
    for i in range(FINAL_STATIC_START - 5, FINAL_STATIC_START):
        w = (FINAL_STATIC_START - i) / 5.0
        C[i, :3, :3] = Rot.from_quat(qlerp(
            Rot.from_matrix(C[i, :3, :3]).as_quat(), q1, w)).as_matrix()
        C[i, :3, 3] = (1.0 - w) * C[i, :3, 3] + w * S1[:3, 3]

    # --- Downsample masks once for the repeated silhouette passes ----------
    half_scale = 0.5
    W = int(round(camera["width"] * half_scale))
    H = int(round(camera["height"] * half_scale))
    K_half = K.copy()
    K_half[:2] *= half_scale
    object_half = np.empty((frame_count, H, W), dtype=bool)
    person_half = np.empty_like(object_half)
    masks_path = os.path.join(inputs, "masks.h5")
    with h5py.File(masks_path, "r") as f:
        for i in range(frame_count):
            object_half[i] = cv2.resize(
                f["object"][i].astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            person_half[i] = cv2.resize(
                f["person"][i].astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST).astype(bool)

    renderer = MaskRenderer(
        os.path.join(inputs, "object_mesh.glb"), W, H, device="cuda")

    # --- Pass 1: silhouette-centroid drift, then metric depth --------------
    C = centroid_correction(
        C, object_half, person_half, renderer, K, K_half, T_wc, T_wc_inv,
        INITIAL_STATIC_END, FINAL_STATIC_START, sigma=5.0)
    with h5py.File(masks_path, "r") as f:
        depth_anchors, depth_scales = estimate_depth_curve(
            C, f, object_half, person_half, renderer, K, K_half, T_wc,
            T_wc_inv, INITIAL_STATIC_END, FINAL_STATIC_START)
    C, depth_curve = apply_depth_curve(
        C, depth_anchors, depth_scales, T_wc, T_wc_inv,
        INITIAL_STATIC_END, FINAL_STATIC_START)
    C = centroid_correction(
        C, object_half, person_half, renderer, K, K_half, T_wc, T_wc_inv,
        INITIAL_STATIC_END, FINAL_STATIC_START, sigma=2.0)

    # --- Pass 2: sparse rotation correction, then re-fit translation/depth -
    with h5py.File(masks_path, "r") as f:
        rot_anchors, rot_deltas = probe_rotation_corrections(
            C, f, person_half, renderer, K_half, T_wc_inv,
            INITIAL_STATIC_END + 4, FINAL_STATIC_START)
    C = apply_rotation_curve(
        C, rot_anchors, rot_deltas, INITIAL_STATIC_END,
        FINAL_STATIC_START, sigma=10.0)
    C = centroid_correction(
        C, object_half, person_half, renderer, K, K_half, T_wc, T_wc_inv,
        INITIAL_STATIC_END, FINAL_STATIC_START, sigma=5.0)
    with h5py.File(masks_path, "r") as f:
        depth_anchors, depth_scales = estimate_depth_curve(
            C, f, object_half, person_half, renderer, K, K_half, T_wc,
            T_wc_inv, INITIAL_STATIC_END, FINAL_STATIC_START)
    C, _ = apply_depth_curve(
        C, depth_anchors, depth_scales, T_wc, T_wc_inv,
        INITIAL_STATIC_END, FINAL_STATIC_START)
    C = centroid_correction(
        C, object_half, person_half, renderer, K, K_half, T_wc, T_wc_inv,
        INITIAL_STATIC_END, FINAL_STATIC_START, sigma=2.0)

    # Re-assert the pooled rigid poses after all moving-frame refinements.
    C[:INITIAL_STATIC_END] = S0
    C[FINAL_STATIC_START:] = S1

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, T_world_mesh=C.astype(np.float64))
    print(f"wrote {out_path}: {C.shape}, elapsed {time.time() - t_start:.1f}s")
    print("initial static pose:\n", S0)
    print("final static pose:\n", S1)


if __name__ == "__main__":
    main()
