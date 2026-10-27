#!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF monitor trajectory.

The method uses the released CARI4D/CoCoNet camera-frame track as a proposal
track.  It (1) pools all frames in the two static phases to estimate one pose
per phase, (2) repairs the long, physically impossible yaw reversal during the
fast edge-on turn by propagating yaw about the table normal, and (3) smooths or
replaces tracker translation detours while retaining the baseline's human/floor
calibrated world depth.  Static rotations are finally refined under camera K and
placed on the shared table-support plane.  Silhouette scoring uses nvdiffrast and the SAM2 masks;
no ground-truth data is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# Phase boundaries were verified visually from the video and contact signal.
# The monitor is untouched through frame 190, is turned quickly while edge-on
# around frames 345-360, and is resting again from frame 400 onward.
STATIC_INITIAL_END = 190
TURN_START = 344
TURN_END = 380
SETTLE_END = 399
STATIC_FINAL_START = 400


def robust_mean_rotation(mats: np.ndarray) -> Rotation:
    return Rotation.from_matrix(mats).mean()


def load_inputs(inputs: Path):
    episode = json.loads((inputs / "episode.json").read_text())
    camera = json.loads((inputs / "camera.json").read_text())
    ground = json.loads((inputs / "ground_plane.json").read_text())
    rec = np.load(inputs / "reconstruction.npz")
    with h5py.File(inputs / "masks.h5", "r") as h:
        object_masks = h["object"][:]
        person_masks = h["person"][:]
    scene = trimesh.load(inputs / "object_mesh.glb", force="scene", process=False)
    mesh = scene.to_geometry()
    return episode, camera, ground, rec, object_masks, person_masks, mesh


class SilhouetteRenderer:
    def __init__(self, mesh: trimesh.Trimesh, K: np.ndarray, height: int, width: int):
        self.H = int(height)
        self.W = int(width)
        self.K = np.asarray(K, dtype=np.float64)
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32, device="cuda")
        self.glctx = dr.RasterizeCudaContext()

    def mask(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        Rt = torch.tensor(R, dtype=torch.float32, device="cuda")
        tt = torch.tensor(t, dtype=torch.float32, device="cuda")
        v = self.vertices @ Rt.T + tt
        x, y, z = v.unbind(1)
        # K_pipeline is the camera convention used by FoundationPose/CoCoNet.
        # nvdiffrast's rasterizer uses a top-left image origin for this setup.
        clip_x = (2.0 * self.fx / self.W) * x + (2.0 * self.cx / self.W - 1.0) * z
        clip_y = (2.0 * self.fy / self.H) * y + (1.0 - 2.0 * self.cy / self.H) * z
        clip = torch.stack([clip_x, clip_y, torch.zeros_like(z), z], dim=1)[None]
        rast = dr.rasterize(self.glctx, clip, self.faces, [self.H, self.W])[0]
        return (rast[..., 3] > 0).reshape(self.H, self.W).cpu().numpy()


def mean_visible_iou(renderer: SilhouetteRenderer, R, t, indices, object_masks, person_masks):
    pred = renderer.mask(R, t)
    vals = []
    for i in indices:
        visible_pred = pred & ~person_masks[i]
        target = object_masks[i]
        vals.append(float((visible_pred & target).sum()) / max(1, int((visible_pred | target).sum())))
    return float(np.mean(vals))


def optimize_static_pose(renderer, proposal, indices, object_masks, person_masks):
    """Pool all frames in a static phase into one camera pose.

    A deterministic bounded Powell search is robust to the small mask boundary
    changes between frames.  Rotation perturbations are mesh-local; translation
    perturbations are camera-frame metres.
    """
    R0 = robust_mean_rotation(proposal[indices, :3, :3]).as_matrix()
    t0 = np.median(proposal[indices, :3, 3], axis=0)

    def objective(x):
        delta_R = Rotation.from_rotvec(x[:3]).as_matrix()
        R = R0 @ delta_R
        t = t0 + x[3:]
        return 1.0 - mean_visible_iou(renderer, R, t, indices, object_masks, person_masks)

    result = minimize(
        objective,
        np.zeros(6, dtype=np.float64),
        method="Powell",
        bounds=[(-0.15, 0.15)] * 6,
        options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": 8, "disp": False},
    )
    R = R0 @ Rotation.from_rotvec(result.x[:3]).as_matrix()
    t = t0 + result.x[3:]
    return R, t, 1.0 - float(result.fun)


def optimize_static_rotation_world_K(renderer, rec, camera, ground, mesh, indices, support_target, object_masks, person_masks):
    """Refine only rotation under the released camera K, then place the pose on the table.

    CoCoNet and FoundationPose use K_pipeline, but the released evaluation frame
    uses camera.json K.  Optimising all six DoF under K can trade a good mask fit
    for an object hovering above the table.  Here translation is held at the
    pooled baseline value during silhouette optimisation, and only the component
    normal to the table is adjusted afterwards to preserve table support.
    """
    T = rec["T_world_mesh"].astype(np.float64)
    T_world_cam = np.asarray(camera["T_world_cam"], dtype=np.float64)
    T_cam_world = np.linalg.inv(T_world_cam)
    cam_pose = np.array([T_cam_world @ T[i] for i in range(len(T))])
    R0 = robust_mean_rotation(cam_pose[indices, :3, :3]).as_matrix()
    t0 = np.median(cam_pose[indices, :3, 3], axis=0)

    def objective(x):
        R = Rotation.from_rotvec(x).as_matrix() @ R0
        return 1.0 - mean_visible_iou(renderer, R, t0, indices, object_masks, person_masks)

    result = minimize(
        objective,
        np.zeros(3, dtype=np.float64),
        method="Powell",
        bounds=[(-0.25, 0.25)] * 3,
        options={"xtol": 1e-5, "ftol": 1e-6, "maxiter": 10, "disp": False},
    )
    R = Rotation.from_rotvec(result.x).as_matrix() @ R0
    world = T_world_cam @ np.block(
        [[R, t0[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]
    )
    n = np.asarray(ground["plane"][:3], dtype=np.float64)
    n /= np.linalg.norm(n)
    verts = mesh.vertices
    support = float(np.percentile((verts @ world[:3, :3].T + world[:3, 3]) @ n, 0.1))
    world[:3, 3] += (support_target - support) * n
    return world, 1.0 - float(result.fun)


def stabilize_static_world_poses(P, initial_pose, final_pose):
    """Replace the two static phases and blend smoothly into the moving segment."""
    P = P.copy()
    P[:191] = initial_pose
    P[400:] = final_pose
    Ri = Rotation.from_matrix(initial_pose[:3, :3])
    Rf = Rotation.from_matrix(final_pose[:3, :3])
    for i in range(191, 211):
        u = (i - 190) / 20.0
        s = u * u * (3.0 - 2.0 * u)
        q = Rotation.from_matrix(P[i, :3, :3]) * Ri.inv()
        P[i, :3, :3] = (Rotation.from_rotvec(s * q.as_rotvec()) * Ri).as_matrix()
        P[i, :3, 3] = (1.0 - s) * initial_pose[:3, 3] + s * P[i, :3, 3]
    for i in range(380, 400):
        u = (i - 380) / 20.0
        s = u * u * (3.0 - 2.0 * u)
        q = Rf * Rotation.from_matrix(P[i, :3, :3]).inv()
        P[i, :3, :3] = (Rotation.from_rotvec(s * q.as_rotvec()) * Rotation.from_matrix(P[i, :3, :3])).as_matrix()
        P[i, :3, 3] = (1.0 - s) * P[i, :3, 3] + s * final_pose[:3, 3]
    return P


def build_camera_trajectory(rec, camera, static_init, static_final):
    C = rec["T_cam_mesh_coconet"].astype(np.float64)
    F = len(C)
    Ri, ti = static_init[0], static_init[1]
    Rf, tf = static_final[0], static_final[1]

    # Table-normal axis expressed in the pipeline camera frame.
    T_world_cam = np.asarray(camera["T_world_cam"], dtype=np.float64)
    axis = T_world_cam[:3, :3].T @ np.array([0.0, 1.0, 0.0])
    axis /= np.linalg.norm(axis)
    Ri_rot = Rotation.from_matrix(Ri)

    def yaw_angle(R):
        q = Rotation.from_matrix(R @ Ri.T)
        return float(np.degrees(q.as_rotvec() @ axis))

    yaw = np.array([yaw_angle(C[i, :3, :3]) for i in range(F)])

    # Baseline camera translations are already close except in the tracked
    # left/right detour during the edge-on turn.  Smooth the reliable part.
    ts = C[:, :3, 3].copy()
    for j in range(3):
        ts[190:345, j] = savgol_filter(C[190:345, :3, 3][:, j], 21, 2)

    for i in range(345, 380):
        u = (i - TURN_START) / (TURN_END - TURN_START)
        s = u * u * (3.0 - 2.0 * u)
        ts[i] = (1.0 - s) * ts[TURN_START] + s * ts[TURN_END]

    for i in range(380, 400):
        u = (i - 380) / 20.0
        s = u * u * (3.0 - 2.0 * u)
        ts[i] = (1.0 - s) * ts[380] + s * tf
    ts[:191] = ti
    ts[400:] = tf

    turn_end_R = (
        Rotation.from_rotvec(axis * np.radians(yaw[TURN_END] - yaw[TURN_START])).as_matrix()
        @ C[TURN_START, :3, :3]
    )

    rotations = np.empty((F, 3, 3), dtype=np.float64)
    for i in range(F):
        if i <= STATIC_INITIAL_END:
            R = Ri
        elif i <= TURN_START:
            R_raw = C[i, :3, :3]
            u = min(1.0, (i - STATIC_INITIAL_END) / 20.0)
            s = u * u * (3.0 - 2.0 * u)
            delta = s * (Rotation.from_matrix(R_raw) * Ri_rot.inv()).as_rotvec()
            R = (Rotation.from_rotvec(delta) * Ri_rot).as_matrix()
        elif i < TURN_END:
            u = (i - TURN_START) / (TURN_END - TURN_START)
            # The display crosses edge-on almost immediately after frame 344
            # and then slowly settles.  This sqrt easing matches the observed
            # brightness/width transition; a smoothstep turns too slowly.
            s = np.sqrt(max(0.0, u))
            target_yaw = (1.0 - s) * yaw[TURN_START] + s * yaw[TURN_END]
            turn = Rotation.from_rotvec(axis * np.radians(target_yaw - yaw[TURN_START]))
            R = turn.as_matrix() @ C[TURN_START, :3, :3]
        elif i <= SETTLE_END:
            u = (i - TURN_END) / 20.0
            s = u * u * (3.0 - 2.0 * u)
            delta = s * (Rotation.from_matrix(Rf) * Rotation.from_matrix(turn_end_R).inv()).as_rotvec()
            R = (Rotation.from_rotvec(delta) * Rotation.from_matrix(turn_end_R)).as_matrix()
        else:
            R = Rf
        rotations[i] = R
    return rotations, ts


def build_world_trajectory(rec, camera, ground, clean_rot_cam, clean_t_cam, mesh):
    C = rec["T_cam_mesh_coconet"].astype(np.float64)
    T = rec["T_world_mesh"].astype(np.float64)
    F = len(T)

    # Rotation relation between the calibrated world track and CoCoNet is
    # constant up to depth correction.
    rel = np.array([T[i, :3, :3] @ C[i, :3, :3].T for i in range(F)])
    R_world_cam = robust_mean_rotation(rel).as_matrix()
    Rw = np.array([R_world_cam @ clean_rot_cam[i] for i in range(F)])

    tw = T[:, :3, 3].copy()
    for j in range(3):
        tw[190:345, j] = savgol_filter(T[190:345, :3, 3][:, j], 21, 2)
    for i in range(345, 380):
        u = (i - TURN_START) / (TURN_END - TURN_START)
        s = u * u * (3.0 - 2.0 * u)
        tw[i] = (1.0 - s) * tw[TURN_START] + s * tw[TURN_END]

    ti = np.median(T[:191, :3, 3], axis=0)
    tf = np.median(T[400:, :3, 3], axis=0)
    Rwi = robust_mean_rotation(Rw[:191]).as_matrix()
    Rwf = robust_mean_rotation(Rw[400:]).as_matrix()

    # Both static phases rest on the same table.  Use the untouched initial
    # pose to define the table support, then place the final pose on that same
    # support height along the released ground-plane normal.  This is a support
    # constraint, not a snap to the floor.
    n = np.asarray(ground["plane"][:3], dtype=np.float64)
    n /= np.linalg.norm(n)
    verts = mesh.vertices
    def support(R, t):
        return float(np.percentile((verts @ R.T + t) @ n, 0.1))
    initial_support = support(Rwi, ti)
    final_support = support(Rwf, tf)
    tf = tf + (initial_support - final_support) * n

    tw[:191] = ti
    tw[400:] = tf
    for i in range(380, 400):
        u = (i - 380) / 20.0
        s = u * u * (3.0 - 2.0 * u)
        tw[i] = (1.0 - s) * tw[380] + s * tf

    Rw[:191] = Rwi
    Rw[400:] = Rwf
    for i in range(191, 211):
        u = (i - 190) / 20.0
        s = u * u * (3.0 - 2.0 * u)
        q = Rotation.from_matrix(Rw[i]) * Rotation.from_matrix(Rwi).inv()
        Rw[i] = (Rotation.from_rotvec(s * q.as_rotvec()) * Rotation.from_matrix(Rwi)).as_matrix()
    for i in range(380, 400):
        u = (i - 380) / 20.0
        s = u * u * (3.0 - 2.0 * u)
        q = Rotation.from_matrix(Rwf) * Rotation.from_matrix(Rw[i]).inv()
        Rw[i] = (Rotation.from_rotvec(s * q.as_rotvec()) * Rotation.from_matrix(Rw[i])).as_matrix()

    out = np.zeros((F, 4, 4), dtype=np.float64)
    out[:, :3, :3] = Rw
    out[:, :3, 3] = tw
    out[:, 3, 3] = 1.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    episode, camera, ground, rec, object_masks, person_masks, mesh = load_inputs(args.inputs)
    F = int(episode["frames"])
    assert len(rec["T_world_mesh"]) == F
    assert object_masks.shape[0] == F

    proposal = rec["T_cam_mesh_coconet"].astype(np.float64)
    renderer = SilhouetteRenderer(mesh, np.asarray(camera["K_pipeline"]), camera["height"], camera["width"])

    init_idx = list(range(0, STATIC_INITIAL_END + 1, 5))
    final_idx = list(range(STATIC_FINAL_START, F, 5))
    static_init = optimize_static_pose(renderer, proposal, init_idx, object_masks, person_masks)
    static_final = optimize_static_pose(renderer, proposal, final_idx, object_masks, person_masks)
    print(f"static initial pooled mask IoU: {static_init[2]:.4f}")
    print(f"static final pooled mask IoU:   {static_final[2]:.4f}")

    clean_rot_cam, clean_t_cam = build_camera_trajectory(rec, camera, static_init, static_final)
    world = build_world_trajectory(rec, camera, ground, clean_rot_cam, clean_t_cam, mesh)

    # Final static pose refinement and support constraint under the released K.
    renderer_k = SilhouetteRenderer(mesh, np.asarray(camera["K"]), camera["height"], camera["width"])
    T_base = rec["T_world_mesh"].astype(np.float64)
    n = np.asarray(ground["plane"][:3], dtype=np.float64)
    n /= np.linalg.norm(n)
    supports = [
        float(np.percentile((mesh.vertices @ T_base[i, :3, :3].T + T_base[i, :3, 3]) @ n, 0.1))
        for i in range(0, STATIC_INITIAL_END + 1, 5)
    ]
    support_target = float(np.median(supports))
    static_world_init, init_k_iou = optimize_static_rotation_world_K(
        renderer_k, rec, camera, ground, mesh, init_idx, support_target, object_masks, person_masks
    )
    static_world_final, final_k_iou = optimize_static_rotation_world_K(
        renderer_k, rec, camera, ground, mesh, final_idx, support_target, object_masks, person_masks
    )
    print(f"K-space static initial rotation IoU: {init_k_iou:.4f}")
    print(f"K-space static final rotation IoU:   {final_k_iou:.4f}")
    world = stabilize_static_world_poses(world, static_world_init, static_world_final)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T_world_mesh=world.astype(np.float64))


if __name__ == "__main__":
    main()
