#!/usr/bin/env python3
"""Recover a temporally consistent world-space trajectory for the step ladder.

The reconstruction in reconstruction.npz is already a strong per-frame source of
evidence.  This solver changes it only where the video supplies stronger,
whole-trajectory evidence:

* the ladder is untouched and static for frames 0..200;
* it is dragged between frames 201..555;
* it is static again from frame 556 onward.

The boundaries are explicit constants because they were verified visually and
by the static-vs-baseline silhouette residual curves.  Within each static phase
one 6-DoF pose is fitted jointly to many frames (all frames are equivalent under
a static camera and object).  The dynamic part uses the baseline as evidence,
is lightly smoothed, and is joined continuously to the two static anchors.
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
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# nvdiffrast is part of the supplied environment.  Importing it here keeps all
# rendering deterministic and offline.
import nvdiffrast.torch as dr

SEED = 0
START_REST_END = 201       # first moving frame; verified on 200/201/202 boundary
FINAL_REST_START = 556     # first final-rest frame; verified on 555/556/557
SMOOTH_WINDOW = 7          # 0.23 s; removes tracker jitter without motion lag
SMOOTH_POLY = 2
START_FADE = 20            # frames used to leave the initial static anchor
END_FADE = 22              # frames used to enter the final static anchor


def load_mesh(path):
    return trimesh.load(path, force="scene", process=False).to_geometry()


def inv_rigid(T):
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = T[:3, :3].T
    X[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return X


def pool_poses(T):
    """Chordal rotation mean and arithmetic translation mean."""
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = Rotation.from_matrix(T[:, :3, :3]).mean().as_matrix()
    X[:3, 3] = T[:, :3, 3].mean(axis=0)
    return X


def update_pose(T, mesh_center, axis, value, is_rotation):
    """Apply a small world-frame perturbation to a mesh-to-world pose."""
    G = np.eye(4, dtype=np.float64)
    if is_rotation:
        R = Rotation.from_euler(axis, value, degrees=True).as_matrix()
        G[:3, :3] = R
        G[:3, 3] = mesh_center - R @ mesh_center
    else:
        G[:3, 3] = value
    return G @ T


def scaled_transform(Z, weight):
    """Scale an identity-to-Z rigid transform without changing its axis."""
    X = np.eye(4, dtype=np.float64)
    X[:3, :3] = Rotation.from_rotvec(
        Rotation.from_matrix(Z[:3, :3]).as_rotvec() * weight
    ).as_matrix()
    X[:3, 3] = Z[:3, 3] * weight
    return X


class MaskRasterizer:
    def __init__(self, mesh, K, T_cam_world, height, width, chunk=8):
        self.device = "cuda"
        self.vertices = torch.as_tensor(
            mesh.vertices, dtype=torch.float32, device=self.device
        )
        self.faces = torch.as_tensor(mesh.faces, dtype=torch.int32, device=self.device)
        self.Tcw = torch.as_tensor(
            T_cam_world, dtype=torch.float32, device=self.device
        )
        self.K = torch.as_tensor(K, dtype=torch.float32, device=self.device).clone()
        self.H, self.W = int(height), int(width)
        self.K[0] *= self.W / 1536.0
        self.K[1] *= self.H / 1152.0
        self.rastctx = dr.RasterizeCudaContext()

    def one(self, T_world_mesh):
        T = torch.as_tensor(
            T_world_mesh, dtype=torch.float32, device=self.device
        )[None]
        vw = torch.einsum("bij,vj->bvi", T[:, :3, :3], self.vertices)
        vw = vw + T[:, None, :3, 3]
        vc = torch.einsum("ij,bvj->bvi", self.Tcw[:3, :3], vw)
        vc = vc + self.Tcw[None, None, :3, 3]
        z = vc[..., 2].clamp(min=1e-5)
        px = self.K[0, 0] * vc[..., 0] / z + self.K[0, 2]
        py = self.K[1, 1] * vc[..., 1] / z + self.K[1, 2]
        clip = torch.stack(
            [
                (2.0 * px / self.W - 1.0) * z,
                (2.0 * py / self.H - 1.0) * z,
                z,
                z,
            ],
            dim=-1,
        )
        rast, _ = dr.rasterize(
            self.rastctx, clip, self.faces, resolution=[self.H, self.W]
        )
        return (rast[..., 3] > 0)[0].cpu().numpy()


def outer_edge(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    out = np.zeros_like(mask, dtype=np.uint8)
    if contours:
        cv2.drawContours(out, contours, -1, 1, 1)
    return out.astype(bool)


def static_pose_score(T, rasterizer, object_masks, person_masks):
    """Mean symmetric outer-contour chamfer distance over static frames."""
    rendered = rasterizer.one(T)
    scores = []
    for obs, person in zip(object_masks, person_masks):
        pd = cv2.dilate(
            person.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
        vis = rendered & ~pd
        ob = obs & ~pd
        ev = outer_edge(vis)
        eo = outer_edge(ob)
        dt_obs = cv2.distanceTransform(
            (~obs).astype(np.uint8), cv2.DIST_L2, 5
        )
        dt_vis = cv2.distanceTransform(
            (~vis).astype(np.uint8), cv2.DIST_L2, 5
        )
        d1 = float(dt_obs[ev].mean()) if ev.any() else 99.0
        d2 = float(dt_vis[eo].mean()) if eo.any() else 99.0
        scores.append(d1 + d2)
    return float(np.mean(scores))


def refine_static_pose(seed, rasterizer, object_masks, person_masks, rounds):
    """Deterministic coordinate descent on the multi-frame mask objective."""
    T = seed.copy()
    best_score = static_pose_score(T, rasterizer, object_masks, person_masks)
    print(f"  seed static score: {best_score:.4f}", flush=True)
    for rot_step, trans_step in rounds:
        for axis in ("y", "x", "z"):
            chosen = (best_score, T.copy())
            for sign in (-1.0, 1.0):
                candidate = update_pose(
                    T, rasterizer.mesh_center, axis, sign * rot_step, True
                )
                score = static_pose_score(
                    candidate, rasterizer, object_masks, person_masks
                )
                if score < chosen[0]:
                    chosen = (score, candidate)
            best_score, T = chosen
        for axis in range(3):
            chosen = (best_score, T.copy())
            for sign in (-1.0, 1.0):
                delta = np.zeros(3, dtype=np.float64)
                delta[axis] = sign * trans_step
                candidate = update_pose(T, rasterizer.mesh_center, None, delta, False)
                score = static_pose_score(
                    candidate, rasterizer, object_masks, person_masks
                )
                if score < chosen[0]:
                    chosen = (score, candidate)
            best_score, T = chosen
        print(f"  static score: {best_score:.4f}", flush=True)
    return T


def read_mask_stack(path, key, frame_ids):
    frame_ids = list(frame_ids)
    with h5py.File(path, "r") as h:
        return np.asarray(h[key][frame_ids], dtype=bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    t_start = time.time()

    rec = np.load(os.path.join(args.inputs, "reconstruction.npz"))
    baseline = np.asarray(rec["T_world_mesh"], dtype=np.float64)
    F = baseline.shape[0]
    if F != 719:
        raise RuntimeError(f"Expected 719 poses in this episode, got {F}")

    with open(os.path.join(args.inputs, "camera.json"), "r") as f:
        camera = json.load(f)
    K = np.asarray(camera["K"], dtype=np.float64)
    T_world_cam = np.asarray(camera["T_world_cam"], dtype=np.float64)
    T_cam_world = np.linalg.inv(T_world_cam)
    mesh = load_mesh(os.path.join(args.inputs, "object_mesh.glb"))
    masks_path = os.path.join(args.inputs, "masks.h5")

    # Initial static pose.  Frames 88..150 are unoccluded and therefore carry
    # the cleanest silhouette evidence.  Fitting is done at half resolution to
    # avoid over-weighting the SAM boundary dilation.
    start_ids = list(range(88, 151, 4))
    start_obs = read_mask_stack(masks_path, "object", start_ids)[:, ::2, ::2]
    start_per = read_mask_stack(masks_path, "person", start_ids)[:, ::2, ::2]
    start_seed = pool_poses(baseline[:START_REST_END])
    rast_start = MaskRasterizer(mesh, K, T_cam_world, 576, 768)
    rast_start.mesh_center = np.asarray(mesh.bounds.mean(axis=0), dtype=np.float64)
    print("Fitting initial static pose...", flush=True)
    start_pose = refine_static_pose(
        start_seed,
        rast_start,
        start_obs,
        start_per,
        rounds=[(1.0, 0.010), (0.5, 0.005), (0.25, 0.0025), (0.1, 0.001)],
    )

    # Final static pose.  By this point the person is out of the way, so fit at
    # native resolution and use every sixth frame.
    end_ids = list(range(660, 719, 6))
    end_obs = read_mask_stack(masks_path, "object", end_ids)
    end_per = read_mask_stack(masks_path, "person", end_ids)
    end_seed = pool_poses(baseline[660:719])
    rast_end = MaskRasterizer(mesh, K, T_cam_world, 1152, 1536)
    rast_end.mesh_center = np.asarray(mesh.bounds.mean(axis=0), dtype=np.float64)
    print("Fitting final static pose...", flush=True)
    end_pose = refine_static_pose(
        end_seed,
        rast_end,
        end_obs,
        end_per,
        rounds=[(0.2, 0.002), (0.1, 0.001), (0.05, 0.0005)],
    )

    # Light trajectory smoothing.  Rotations are filtered as sign-consistent
    # quaternions so the interpolation does not cross the quaternion double
    # cover.  A 7-frame window reduces frame-wise tracking noise while adding
    # no observable silhouette bias.
    quat = Rotation.from_matrix(baseline[:, :3, :3]).as_quat()
    signs = np.where(quat[0] @ quat.T < 0.0, -1.0, 1.0)
    quat *= signs[:, None]
    quat = savgol_filter(
        quat, SMOOTH_WINDOW, SMOOTH_POLY, axis=0, mode="interp"
    )
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    trajectory = baseline.copy()
    trajectory[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    trajectory[:, :3, 3] = savgol_filter(
        baseline[:, :3, 3],
        SMOOTH_WINDOW,
        SMOOTH_POLY,
        axis=0,
        mode="interp",
    )

    # Pool the whole static phases into one pose each.
    trajectory[:START_REST_END] = start_pose[None]
    trajectory[FINAL_REST_START:] = end_pose[None]

    # Join the smoothed dynamic evidence to the two static anchors.  The
    # correction is local to the two transitions; the middle of the drag stays
    # with the per-frame evidence rather than being globally warped.
    correction_start = start_pose @ inv_rigid(trajectory[START_REST_END])
    weights_in = np.linspace(1.0, 0.0, START_FADE)
    for k, frame in enumerate(range(START_REST_END, START_REST_END + START_FADE)):
        trajectory[frame] = (
            scaled_transform(correction_start, weights_in[k]) @ trajectory[frame]
        )

    correction_end = end_pose @ inv_rigid(trajectory[FINAL_REST_START - 1])
    weights_out = np.linspace(0.0, 1.0, END_FADE)
    for k, frame in enumerate(
        range(FINAL_REST_START - END_FADE, FINAL_REST_START)
    ):
        trajectory[frame] = (
            scaled_transform(correction_end, weights_out[k]) @ trajectory[frame]
        )

    # Numerical hygiene: enforce proper rigid transforms and float64 output.
    for i in range(F):
        U, _, Vt = np.linalg.svd(trajectory[i, :3, :3])
        trajectory[i, :3, :3] = U @ Vt
        trajectory[i, :3, 3] = trajectory[i, :3, 3].astype(np.float64)
        trajectory[i, 3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=trajectory.astype(np.float64))
    print(f"Wrote {args.out}: {trajectory.shape}, {trajectory.dtype}")
    print(f"Elapsed: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
