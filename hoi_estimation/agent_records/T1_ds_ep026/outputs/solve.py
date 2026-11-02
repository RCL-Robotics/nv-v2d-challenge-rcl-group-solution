#!/usr/bin/env python3
"""Trajectory-level skateboard pose recovery for a single monocular video.

The release reconstruction is used as a strong per-frame prior.  The method
below adds evidence that the per-frame tracker cannot exploit by itself:
repeated observations of the rigid object in three static phases, silhouette
agreement over the whole hand-object sequence, and smooth SE(3) motion through
the two manipulated intervals.  The object mesh is rasterized with nvdiffrast;
all fitting is deterministic and runs offline from the released inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

try:
    import nvdiffrast.torch as dr
except Exception as exc:  # pragma: no cover - environment diagnostic
    raise RuntimeError("nvdiffrast is required to reproduce the result") from exc


# The following phase boundaries were verified from the video/masks and the
# baseline pose derivatives.  Frames are inclusive.  The board is stationary
# while the person approaches, during the hand repositioning between the two
# pushes, and after the final push.  These are explicit constants rather than
# a per-video threshold so the method remains deterministic at phase edges.
REST_PHASES = ((0, 285), (355, 404), (505, 668))
MOTION_PHASES = ((286, 354), (405, 504))
STATIC_FIT_FRAMES = ((0, 230), (355, 405), (530, 669))

# Crop containing the object in every frame.  Its constant camera projection
# makes batched rasterization fast and avoids fitting irrelevant background.
CROP_X0, CROP_Y0, CROP_W, CROP_H = 450, 740, 500, 250
SMOOTH_WINDOW = 7


def set_deterministic(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[0])
    return torch.stack(
        (
            torch.stack((z, -v[2], v[1])),
            torch.stack((v[2], z, -v[0])),
            torch.stack((-v[1], v[0], z)),
        )
    )


def pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse_pose(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def rotation_mean(mats: np.ndarray) -> np.ndarray:
    # Quaternion chordal mean with sign alignment.  The within-phase spread is
    # below a degree, so this is numerically stable and avoids Euler singularities.
    q = Rot.from_matrix(mats).as_quat()
    q0 = q[0]
    for _ in range(5):
        q = q * np.sign(q @ q0)[:, None]
        q0 = q.mean(axis=0)
        q0 /= np.linalg.norm(q0)
    return Rot.from_quat(q0).as_matrix()


def mean_pose(T: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return pose(rotation_mean(T[idx, :3, :3]), T[idx, :3, 3].mean(axis=0))


def interpolate_pose(A: np.ndarray, B: np.ndarray, a: float) -> np.ndarray:
    """Geodesic rotation and linear translation interpolation."""
    dR = A[:3, :3].T @ B[:3, :3]
    R = A[:3, :3] @ Rot.from_rotvec(Rot.from_matrix(dR).as_rotvec() * a).as_matrix()
    t = (1.0 - a) * A[:3, 3] + a * B[:3, 3]
    return pose(R, t)


def continuous_rotvecs(T: np.ndarray, start: int, end: int) -> np.ndarray:
    """Unrolled log map relative to the pose at `start`.

    Integrating small adjacent relative rotations preserves several complete
    turns, unlike direct axis-angle extraction, and avoids filter wrap jumps.
    """
    out = np.zeros((end - start, 3), dtype=np.float64)
    for i in range(start + 1, end):
        dR = T[i - 1, :3, :3].T @ T[i, :3, :3]
        out[i - start] = out[i - start - 1] + Rot.from_matrix(dR).as_rotvec()
    return out


class MeshRenderer:
    def __init__(self, mesh_path: Path, camera: dict, device: torch.device):
        self.device = device
        mesh = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
        # The exported skateboard has mixed winding.  Duplicating each triangle
        # with reversed winding gives the union silhouette but still preserves
        # front-most depth ordering in the rasterizer.
        self.faces = torch.cat((faces, faces.flip(1)), dim=0)
        self.uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
        texture = np.asarray(mesh.visual.material.baseColorTexture, dtype=np.float32) / 255.0
        self.texture = torch.tensor(texture, dtype=torch.float32, device=device)

        T_cam_world = np.linalg.inv(np.asarray(camera["T_world_cam"], dtype=np.float64))
        self.R_cam_world = torch.tensor(T_cam_world[:3, :3], dtype=torch.float32, device=device)
        self.t_cam_world = torch.tensor(T_cam_world[:3, 3], dtype=torch.float32, device=device)
        self.K = torch.tensor(np.asarray(camera["K"], dtype=np.float32), device=device)
        self.ctx = dr.RasterizeCudaContext(device=device)

    def alpha_and_color(self, R: torch.Tensor, t: torch.Tensor):
        camera_xyz = (
            (self.vertices @ R.T + t) @ self.R_cam_world.T + self.t_cam_world
        )
        x = self.K[0, 0] * camera_xyz[:, 0] / camera_xyz[:, 2] + self.K[0, 2]
        y = self.K[1, 1] * camera_xyz[:, 1] / camera_xyz[:, 2] + self.K[1, 2]
        z = camera_xyz[:, 2]
        pos = torch.stack(
            (
                ((x - CROP_X0) / CROP_W * 2.0 - 1.0) * z,
                ((y - CROP_Y0) / CROP_H * 2.0 - 1.0) * z,
                z,
                z,
            ),
            dim=1,
        )[None].contiguous()
        rast, _ = dr.rasterize(
            self.ctx, pos, self.faces, resolution=(CROP_H, CROP_W)
        )
        hard_alpha = (rast[..., 3] > 0).float()
        alpha = dr.antialias(hard_alpha[..., None], rast, pos, self.faces)[0, ..., 0]
        uv = dr.interpolate(self.uv[None].contiguous(), rast, self.faces)[0]
        # glTF's V axis is top-down while nvdiffrast samples bottom-up.
        rgb = dr.texture(
            self.texture.flip(0)[None],
            uv,
            filter_mode="linear",
            boundary_mode="wrap",
        )[0]
        return alpha, rgb

    def alpha(self, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        camera_xyz = (
            (self.vertices @ R.T + t) @ self.R_cam_world.T + self.t_cam_world
        )
        x = self.K[0, 0] * camera_xyz[:, 0] / camera_xyz[:, 2] + self.K[0, 2]
        y = self.K[1, 1] * camera_xyz[:, 1] / camera_xyz[:, 2] + self.K[1, 2]
        z = camera_xyz[:, 2]
        pos = torch.stack(
            (
                ((x - CROP_X0) / CROP_W * 2.0 - 1.0) * z,
                ((y - CROP_Y0) / CROP_H * 2.0 - 1.0) * z,
                z,
                z,
            ),
            dim=1,
        )[None].contiguous()
        rast, _ = dr.rasterize(
            self.ctx, pos, self.faces, resolution=(CROP_H, CROP_W)
        )
        hard_alpha = (rast[..., 3] > 0).float()
        return dr.antialias(hard_alpha[..., None], rast, pos, self.faces)[0, ..., 0]


def soft_iou(alpha: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (alpha * target).sum() / (alpha + target - alpha * target).sum().clamp_min(1e-8)


def optimize_static(
    renderer: MeshRenderer,
    baseline: np.ndarray,
    mask_crop: torch.Tensor,
    person_crop: torch.Tensor,
    fit_idx: np.ndarray,
    validation_frames: list[tuple[int, torch.Tensor]],
    steps: int = 180,
) -> np.ndarray:
    target = mask_crop[fit_idx].mean(dim=0)
    valid = 1.0 - person_crop[fit_idx].mean(dim=0)
    target = target * valid
    R_init = torch.tensor(
        rotation_mean(baseline[fit_idx, :3, :3]), dtype=torch.float32, device=renderer.device
    )
    t_init = torch.tensor(
        np.median(baseline[fit_idx, :3, 3], axis=0), dtype=torch.float32, device=renderer.device
    )
    dt = torch.zeros(3, device=renderer.device, requires_grad=True)
    drv = torch.zeros(3, device=renderer.device, requires_grad=True)
    opt = torch.optim.Adam((dt, drv), lr=3e-3)
    valid_sum = valid.sum().clamp_min(1.0)

    # The silhouette objective is weakly textured and has a shallow depth
    # valley.  Checkpoints are scored with an independent image-texture term;
    # this resolves the residual yaw/depth ambiguity without trusting either
    # cue alone.  The 10:1 scale keeps silhouette the dominant requirement.
    best_score = -1e9
    best_dt = dt.detach().clone()
    best_drv = drv.detach().clone()

    def validation_score() -> float:
        nonlocal best_dt, best_drv
        ious, nccs = [], []
        with torch.no_grad():
            R_cur = R_init @ torch.matrix_exp(skew(drv))
            t_cur = t_init + dt
            for frame_idx, gray_image in validation_frames:
                hard_alpha, rgb = renderer.alpha_and_color(R_cur, t_cur)
                target_i = mask_crop[frame_idx] * (1.0 - person_crop[frame_idx])
                hit = (hard_alpha > 0.5) & (target_i > 0.5)
                ious.append(
                    (hit.sum().float() / ((hard_alpha > 0.5) | (target_i > 0.5)).sum().float().clamp_min(1.0)).item()
                )
                weight = hit
                if weight.sum() > 100:
                    gray_render = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
                    x = gray_render[weight]
                    y = gray_image[weight]
                    x = x - x.mean()
                    y = y - y.mean()
                    nccs.append(
                        ((x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-8)).item()
                    )
        return float(np.mean(ious) + 0.10 * np.mean(nccs))

    for it in range(steps):
        if it % 10 == 0:
            score = validation_score()
            if score > best_score:
                best_score = score
                best_dt = dt.detach().clone()
                best_drv = drv.detach().clone()
        opt.zero_grad(set_to_none=True)
        R = R_init @ torch.matrix_exp(skew(drv))
        alpha = renderer.alpha(R, t_init + dt)
        mse = (((alpha - target) ** 2) * valid).sum() / valid_sum
        bce = -(
            (target * torch.log(alpha + 1e-5) + (1.0 - target) * torch.log(1.0 - alpha + 1e-5))
            * valid
        ).sum() / valid_sum
        reg = ((dt / 0.03) ** 2).mean() + ((drv / 0.03) ** 2).mean()
        (mse + 0.02 * bce + 0.01 * reg).backward()
        opt.step()
    with torch.no_grad():
        R = R_init @ torch.matrix_exp(skew(best_drv))
        print(f"  static checkpoint score={best_score:.4f}", flush=True)
        return pose(R.cpu().numpy(), (t_init + best_dt).cpu().numpy())


def optimize_dynamic_frame(
    renderer: MeshRenderer,
    initial: np.ndarray,
    mask: torch.Tensor,
    person: torch.Tensor,
    steps: int = 55,
) -> np.ndarray:
    valid = 1.0 - person
    target = mask * valid
    R_init = torch.tensor(initial[:3, :3], dtype=torch.float32, device=renderer.device)
    t_init = torch.tensor(initial[:3, 3], dtype=torch.float32, device=renderer.device)
    dt = torch.zeros(3, device=renderer.device, requires_grad=True)
    drv = torch.zeros(3, device=renderer.device, requires_grad=True)
    opt = torch.optim.Adam((dt, drv), lr=2e-3)
    valid_sum = valid.sum().clamp_min(1.0)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        R = R_init @ torch.matrix_exp(skew(drv))
        alpha = renderer.alpha(R, t_init + dt)
        mse = (((alpha - target) ** 2) * valid).sum() / valid_sum
        bce = -(
            (target * torch.log(alpha + 1e-5) + (1.0 - target) * torch.log(1.0 - alpha + 1e-5))
            * valid
        ).sum() / valid_sum
        reg = ((dt / 0.03) ** 2).mean() + ((drv / 0.035) ** 2).mean()
        (mse + 0.02 * bce + 0.015 * reg).backward()
        opt.step()
    with torch.no_grad():
        R = R_init @ torch.matrix_exp(skew(drv))
        return pose(R.cpu().numpy(), (t_init + dt).detach().cpu().numpy())


def smooth_motion(T: np.ndarray) -> np.ndarray:
    out = T.copy()
    for start, end in MOTION_PHASES:
        # end is inclusive
        a, b = start, end + 1
        out[a:b, :3, 3] = savgol_filter(
            T[a:b, :3, 3], SMOOTH_WINDOW, 2, axis=0, mode="interp"
        )
        rv = continuous_rotvecs(T, a, b)
        rv = savgol_filter(rv, SMOOTH_WINDOW, 2, axis=0, mode="interp")
        R0 = T[a, :3, :3]
        for i in range(a, b):
            out[i, :3, :3] = R0 @ Rot.from_rotvec(rv[i - a]).as_matrix()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    set_deterministic(0)

    inputs = args.inputs
    with np.load(inputs / "reconstruction.npz") as rec:
        baseline = rec["T_world_mesh"].astype(np.float64)
    camera = json.loads((inputs / "camera.json").read_text())
    episode = json.loads((inputs / "episode.json").read_text())
    F = int(episode["frames"])
    if baseline.shape != (F, 4, 4):
        raise ValueError(f"Unexpected baseline shape {baseline.shape}, expected {(F, 4, 4)}")

    crop_slice = (slice(None), slice(CROP_Y0, CROP_Y0 + CROP_H), slice(CROP_X0, CROP_X0 + CROP_W))
    with h5py.File(inputs / "masks.h5", "r") as h5:
        masks_crop = h5["object"][crop_slice]
        people_crop = h5["person"][crop_slice]
    if masks_crop.shape[0] != F or people_crop.shape[0] != F:
        raise ValueError("Mask frame count does not match episode metadata")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = MeshRenderer(inputs / "object_mesh.glb", camera, device)
    mask_crop = torch.tensor(masks_crop, dtype=torch.float32, device=device)
    person_crop = torch.tensor(people_crop, dtype=torch.float32, device=device)

    # Load only frames used to score static-pose candidates.  Grayscale texture
    # correlation is an independent check on the silhouette fit.
    all_validation_indices = []
    for fit_a, fit_b in STATIC_FIT_FRAMES:
        idx = np.arange(fit_a, min(fit_b, F), dtype=np.int64)
        all_validation_indices.extend(idx[np.linspace(0, len(idx) - 1, min(10, len(idx))).astype(int)].tolist())
    video = cv2.VideoCapture(str(inputs / "video.mp4"))
    validation_images: dict[int, torch.Tensor] = {}
    for frame_idx in sorted(set(all_validation_indices)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = video.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {frame_idx}")
        gray = cv2.cvtColor(
            frame[CROP_Y0 : CROP_Y0 + CROP_H, CROP_X0 : CROP_X0 + CROP_W],
            cv2.COLOR_BGR2GRAY,
        ).astype(np.float32) / 255.0
        validation_images[frame_idx] = torch.tensor(gray, dtype=torch.float32, device=device)
    video.release()

    T = baseline.copy()

    # Pool hundreds of repeated observations into one pose per rest interval.
    static_poses = []
    for (rest_a, rest_b), (fit_a, fit_b) in zip(REST_PHASES, STATIC_FIT_FRAMES):
        fit_idx = np.arange(fit_a, min(fit_b, F), dtype=np.int64)
        val_idx = fit_idx[np.linspace(0, len(fit_idx) - 1, min(10, len(fit_idx))).astype(int)]
        validation_frames = [(int(i), validation_images[int(i)]) for i in val_idx]
        S = optimize_static(
            renderer, baseline, mask_crop, person_crop, fit_idx, validation_frames
        )
        T[rest_a : min(rest_b + 1, F)] = S
        static_poses.append(S)
        print(f"rest {rest_a:03d}-{rest_b:03d}: fitted on {fit_idx[0]:03d}-{fit_idx[-1]:03d}", flush=True)

    # Correct the baseline's slowly varying depth/translation bias by carrying
    # the static-pose correction smoothly between adjacent rest anchors.
    m1 = mean_pose(baseline, np.arange(0, 286))
    m3 = mean_pose(baseline, np.arange(355, 405))
    m5 = mean_pose(baseline, np.arange(505, 669))
    c1 = static_poses[0] @ inverse_pose(m1)
    c3 = static_poses[1] @ inverse_pose(m3)
    c5 = static_poses[2] @ inverse_pose(m5)

    base = baseline.copy()
    base[0:286] = static_poses[0]
    base[355:405] = static_poses[1]
    base[505:669] = static_poses[2]
    for (start, end), c_left, c_right, anchor_left, anchor_right in (
        ((286, 354), c1, c3, static_poses[0], static_poses[1]),
        ((405, 504), c3, c5, static_poses[1], static_poses[2]),
    ):
        count = end - start + 1
        for k, i in enumerate(range(start, end + 1)):
            alpha = k / max(count - 1, 1)
            correction = interpolate_pose(c_left, c_right, alpha)
            base[i] = correction @ baseline[i]
            # Force continuity at the phase boundary, then let the image fit
            # take over over the next three frames.
            if k < 3:
                base[i] = interpolate_pose(anchor_left, base[i], (k + 1) / 3.0)
        for k, i in enumerate(range(end - 2, end + 1)):
            base[i] = interpolate_pose(base[i], anchor_right, (k + 1) / 3.0)

    for start, end in MOTION_PHASES:
        for i in range(start, end + 1):
            T[i] = optimize_dynamic_frame(
                renderer, base[i], mask_crop[i], person_crop[i]
            )
        print(f"motion {start:03d}-{end:03d}: mask-refined", flush=True)

    T = smooth_motion(T)
    for i in range(F):
        U, _, Vt = np.linalg.svd(T[i, :3, :3])
        T[i, :3, :3] = U @ Vt
        T[i, 3] = (0.0, 0.0, 0.0, 1.0)
    if not np.isfinite(T).all():
        raise RuntimeError("Non-finite pose produced")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T_world_mesh=T.astype(np.float64))
    print(
        f"wrote {args.out} with T_world_mesh {T.shape} "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
