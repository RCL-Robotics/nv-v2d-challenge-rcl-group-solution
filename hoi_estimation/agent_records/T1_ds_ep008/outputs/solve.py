#!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF chair trajectory.

The released baseline is a useful per-frame estimate but it jitters while the
chair is at rest and accumulates a depth offset.  This script:

1. detects the two long static object phases from the interaction timeline,
2. fits one pooled pose to each static phase with a differentiable silhouette
   rasterizer (one-pixel-eroded SAM masks are used for the final boundary fit,
   which compensates for the mask's outward boundary bias),
3. smooths the tracked dynamic segment and transfers the endpoint static-pose
   corrections smoothly through it.

Only the files under --inputs and the installed software are used.  Runtime is
approximately 3-4 minutes on the provided GPU.
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

# Object phases verified from the video, object-mask motion, and contact
# probabilities.  The chair is stationary through the grasp, starts sliding at
# frame 400, stops by frame 596, and remains stationary while the person leaves.
INITIAL_END = 399       # last frame of the initial static phase
DYNAMIC_START = 400     # first frame with sustained image motion
DYNAMIC_END = 595       # last frame with sustained image motion
FINAL_START = 596       # first frame of the final static phase
SMOOTH_WINDOW = 17      # ~0.57 s; suppresses tracker noise without lagging motion


def mean_pose(transforms, indices):
    """Robust pooled pose for a static segment."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rot.from_matrix(transforms[indices, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(transforms[indices, :3, 3], axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t_start = time.time()

    np.random.seed(0)
    torch.manual_seed(0)

    with open(os.path.join(args.inputs, "camera.json"), "r") as f:
        cam = json.load(f)
    K = np.asarray(cam["K"], dtype=np.float32)
    T_world_cam = np.asarray(cam["T_world_cam"], dtype=np.float64)
    T_cam_world = np.linalg.inv(T_world_cam)

    rec = np.load(os.path.join(args.inputs, "reconstruction.npz"))
    baseline = rec["T_world_mesh"].astype(np.float64)
    F = len(baseline)

    mesh = trimesh.load(
        os.path.join(args.inputs, "object_mesh.glb"),
        force="scene",
        process=False,
    ).to_geometry()
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device="cuda")

    device = torch.device("cuda")
    Tcw = torch.tensor(T_cam_world, dtype=torch.float32, device=device)
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    fx, fy = Kt[0, 0], Kt[1, 1]
    cx, cy = Kt[0, 2], Kt[1, 2]
    width, height = int(cam["width"]), int(cam["height"])
    resolution = torch.tensor([height, width], device=device)
    raster_ctx = dr.RasterizeCudaContext()

    mask_h5 = h5py.File(os.path.join(args.inputs, "masks.h5"), "r")
    object_masks = mask_h5["object"]
    person_masks = mask_h5["person"]

    def render_silhouette(P):
        """Rasterize the exact mesh; returns a full-resolution bool mask."""
        P = torch.tensor(P, dtype=torch.float32, device=device)
        xyz = P[:3, :3] @ vertices.T + P[:3, 3:4]
        xyz = Tcw[:3, :3] @ xyz + Tcw[:3, 3:4]
        x = 2.0 * (fx * xyz[0] / xyz[2] + cx) / width - 1.0
        y = 2.0 * (fy * xyz[1] / xyz[2] + cy) / height - 1.0
        clip = torch.stack(
            [x, y, torch.full_like(x, 0.5), torch.ones_like(x)], dim=-1
        )[None]
        rast, _ = dr.rasterize(raster_ctx, clip, faces, resolution)
        return rast[0, :, :, 3] > 0

    def prepare_masks(indices, erode=0):
        """Targets for visible object pixels; ignore uncertain person boundary."""
        indices = np.asarray(indices, dtype=np.int64)
        obj = object_masks[indices].astype(np.uint8)
        person = person_masks[indices].astype(np.uint8)
        if erode:
            kernel = np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)
            obj = np.asarray([cv2.erode(x, kernel) for x in obj], dtype=np.uint8)
        person = np.asarray(
            [cv2.dilate(x, np.ones((7, 7), np.uint8)) for x in person],
            dtype=np.uint8,
        )
        valid = person == 0
        target = (obj > 0) & valid
        return (
            torch.tensor(target, dtype=torch.bool, device=device),
            torch.tensor(valid, dtype=torch.bool, device=device),
        )

    def make_static_objective(target, valid, start_pose):
        """Mean visible-mask IoU over all pooled static frames."""
        def objective(delta):
            P = np.eye(4, dtype=np.float64)
            P[:3, :3] = Rot.from_rotvec(delta[:3]).as_matrix() @ start_pose[:3, :3]
            P[:3, 3] = start_pose[:3, 3] + delta[3:]
            loss = 0.0
            for tgt, val in zip(target, valid):
                rendered = render_silhouette(P) & val
                inter = (rendered & tgt).sum().float()
                union = rendered.sum().float() + tgt.sum().float() - inter
                loss = loss + (1.0 - inter / union)
            return float((loss / len(target)).item())
        return objective

    def fit_static(start_pose, indices, erode, bounds, tag):
        target, valid = prepare_masks(indices, erode=erode)
        objective = make_static_objective(target, valid, start_pose)
        print(f"[fit] {tag}: start loss={objective(np.zeros(6)):.6f}", flush=True)
        t0 = time.time()
        result = minimize(
            objective,
            np.zeros(6, dtype=np.float64),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 100, "xtol": 1e-5, "ftol": 1e-7, "disp": False},
        )
        P = np.eye(4, dtype=np.float64)
        P[:3, :3] = Rot.from_rotvec(result.x[:3]).as_matrix() @ start_pose[:3, :3]
        P[:3, 3] = start_pose[:3, 3] + result.x[3:]
        print(
            f"[fit] {tag}: end loss={result.fun:.6f}, "
            f"dt={result.x[3:]}, dr={result.x[:3]}, {time.time()-t0:.1f}s",
            flush=True,
        )
        del target, valid
        torch.cuda.empty_cache()
        return P

    print("[phase] fitting initial static pose", flush=True)
    init_start = mean_pose(baseline, np.arange(378, 400))
    wide = [(-0.15, 0.15)] * 3 + [(-0.08, 0.08)] * 3
    narrow = [(-0.04, 0.04)] * 6
    # First pool the whole static phase with the released mask, then refine on
    # a 1-pixel-eroded mask over the unobstructed portion.  Erosion removes the
    # consistent outward halo of the SAM2 mask and prevents a depth bias.
    P_initial = fit_static(
        init_start, np.arange(0, 400, 4), 0, wide, "initial/raw"
    )
    P_initial = fit_static(
        P_initial, np.arange(0, 347, 6), 1, narrow, "initial/eroded"
    )

    print("[phase] fitting final static pose", flush=True)
    final_start = mean_pose(baseline, np.arange(640, F))
    P_final = fit_static(
        final_start, np.arange(610, F, 2), 0, wide, "final/raw"
    )
    P_final = fit_static(
        P_final, np.arange(610, F, 3), 1, narrow, "final/eroded"
    )

    # Smooth the per-frame tracker only inside the moving segment.  Quaternions
    # are sign-unwrapped before filtering and renormalised afterwards.
    out = baseline.copy()
    smooth_ids = np.arange(390, 607)
    q = Rot.from_matrix(baseline[smooth_ids, :3, :3]).as_quat()
    for j in range(1, len(q)):
        if np.dot(q[j], q[j - 1]) < 0.0:
            q[j] *= -1.0
    q = savgol_filter(q, SMOOTH_WINDOW, 2, axis=0, mode="interp")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    smoothed_rot = Rot.from_quat(q).as_matrix()
    smoothed_trans = savgol_filter(
        baseline[smooth_ids, :3, 3], SMOOTH_WINDOW, 2, axis=0, mode="interp"
    )

    # Transfer the pooled endpoint corrections smoothly through the slide.
    # Rotation is applied about the mesh origin (its centroid); this preserves
    # the rigid shape while removing the baseline's static depth/pose drift.
    Rcorr0 = P_initial[:3, :3] @ baseline[INITIAL_END, :3, :3].T
    Rcorr1 = P_final[:3, :3] @ baseline[FINAL_START, :3, :3].T
    tcorr0 = P_initial[:3, 3] - baseline[INITIAL_END, :3, 3]
    tcorr1 = P_final[:3, 3] - baseline[FINAL_START, :3, 3]
    correction_slerp = Slerp(
        [0.0, 1.0], Rot.concatenate([Rot.from_matrix(Rcorr0), Rot.from_matrix(Rcorr1)])
    )
    span = float(FINAL_START - 1 - INITIAL_END)
    for i in range(DYNAMIC_START, DYNAMIC_END + 1):
        s = np.clip((i - INITIAL_END) / span, 0.0, 1.0)
        s = s * s * (3.0 - 2.0 * s)  # C1-continuous endpoint correction
        Rcorr = correction_slerp(s).as_matrix()
        j = i - smooth_ids[0]
        out[i, :3, :3] = Rcorr @ smoothed_rot[j]
        out[i, :3, 3] = smoothed_trans[j] + (1.0 - s) * tcorr0 + s * tcorr1

    # The object is exactly static in these phases.
    out[: DYNAMIC_START] = P_initial
    out[FINAL_START:] = P_final

    assert out.shape == (F, 4, 4)
    assert np.isfinite(out).all()
    np.savez(args.out, T_world_mesh=out.astype(np.float64))
    print(
        f"[done] wrote {args.out} ({F} frames) in {time.time()-t_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
