#!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF watering-can trajectory.

The method uses the supplied FoundationPose/CoCoNet world trajectory only as an
initialisation.  It fits the exact textured mesh to the object mask and to
gradient/texture detail in a fixed image ROI, pools the two rest phases into a
single rigid pose, then smooths and re-fits the manipulation interval.  No
network access and no episode-specific pose table is used.

Usage:
    python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
import nvdiffrast.torch as dr

# ---------------------------------------------------------------------------
# Fixed choices verified from the video/masks (see REPORT.md).
# ROI contains the object in every frame.  Frame boundaries are annotated
# explicitly because decisions made from visual inspection must be reproducible.
# ---------------------------------------------------------------------------
ROI_X0, ROI_Y0, ROI_W, ROI_H = 650, 745, 210, 180
INITIAL_REST_END = 129       # hand begins lifting/turning the can after this frame
INITIAL_BLEND_END = 146
FINAL_REST_START = 615       # can has settled; no visible motion or contact afterwards
FINAL_BLEND_START = 595
PER_FRAME_STEPS = 55
REFINE_STEPS = 40

DEVICE = torch.device("cuda")


def set_deterministic(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rodrigues(r: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(r) + 1e-12
    k = r / theta
    kx = torch.zeros(3, 3, device=r.device, dtype=r.dtype)
    kx[0, 1] = -k[2]
    kx[0, 2] = k[1]
    kx[1, 0] = k[2]
    kx[1, 2] = -k[0]
    kx[2, 0] = -k[1]
    kx[2, 1] = k[0]
    return (
        torch.eye(3, device=r.device, dtype=r.dtype)
        + torch.sin(theta) * kx
        + (1.0 - torch.cos(theta)) * (kx @ kx)
    )


def gaussian_blur(x: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    radius = 3
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k * k) / (2.0 * sigma * sigma))
    k = k / k.sum()
    y = F.conv2d(
        x.permute(0, 3, 1, 2),
        k[None, None, :, None].repeat(3, 1, 1, 1),
        padding=(radius, 0),
        groups=3,
    )
    y = F.conv2d(
        y,
        k[None, None, None, :].repeat(3, 1, 1, 1),
        padding=(0, radius),
        groups=3,
    )
    return y.permute(0, 2, 3, 1)


class MeshRenderer:
    def __init__(self, inputs: Path):
        mesh = trimesh.load(
            str(inputs / "object_mesh.glb"), force="scene", process=False
        ).to_geometry()
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=DEVICE)
        self.faces = torch.tensor(
            np.asarray(mesh.faces, dtype=np.int32), dtype=torch.int32, device=DEVICE
        ).contiguous()
        self.uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=DEVICE)
        tex = np.asarray(mesh.visual.material.baseColorTexture).astype(np.float32)
        tex = torch.tensor(tex / 255.0, dtype=torch.float32, device=DEVICE)
        self.texture = tex.permute(2, 0, 1)[None]
        with open(inputs / "camera.json", "r", encoding="utf-8") as f:
            camera = json.load(f)
        self.K = np.asarray(camera["K"], dtype=np.float64)
        self.T_world_cam = np.asarray(camera["T_world_cam"], dtype=np.float64)
        self.T_cam_world = np.linalg.inv(self.T_world_cam)
        self.T_cam_world_t = torch.tensor(
            self.T_cam_world, dtype=torch.float32, device=DEVICE
        )
        self.glctx = dr.RasterizeCudaContext()

    def render(self, T_world_mesh: torch.Tensor, ss: int = 2):
        """Render ROI RGB and antialiased silhouette (both top-down, h x w)."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        projection = torch.tensor(
            [
                [2 * fx / ROI_W, 0, 2 * (cx - ROI_X0) / ROI_W - 1, 0],
                [0, -2 * fy / ROI_H, 1 - 2 * (cy - ROI_Y0) / ROI_H, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ],
            dtype=torch.float32,
            device=DEVICE,
        )
        vertices_h = torch.cat(
            [self.vertices, torch.ones_like(self.vertices[:, :1])], dim=1
        )
        clip = (
            projection
            @ (self.T_cam_world_t @ T_world_mesh @ vertices_h.T)
        ).T[None].contiguous()
        hr, wr = ROI_H * ss, ROI_W * ss
        rast, _ = dr.rasterize(self.glctx, clip, self.faces, resolution=[hr, wr])
        uv, _ = dr.interpolate(self.uv[None], rast, self.faces)
        grid = uv * 2.0 - 1.0
        # glTF UV v=0 is at image top; nvdiffrast's image is bottom-up.
        grid = torch.stack([grid[..., 0], 1.0 - grid[..., 1]], dim=-1)
        rgb = F.grid_sample(
            self.texture, grid, mode="bilinear", align_corners=False
        )[0].permute(1, 2, 0)
        alpha_hard = (rast[0, :, :, 3] > 0).float()[..., None]
        rgba = torch.cat([rgb * alpha_hard, alpha_hard], dim=-1)[None]
        rgba = dr.antialias(rgba.contiguous(), rast, clip, self.faces)
        rgb = rgba[0, :, :, :3].flip(0).permute(2, 0, 1)[None]
        alpha = rgba[0, :, :, 3].flip(0)[None, None]
        if ss > 1:
            rgb = F.avg_pool2d(rgb, ss)
            alpha = F.avg_pool2d(alpha, ss)
        return rgb[0].permute(1, 2, 0), alpha[0, 0]


def perturb_pose(T0: torch.Tensor, r: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    T = T0.clone()
    T[:3, :3] = rodrigues(r) @ T0[:3, :3]
    T[:3, 3] = T0[:3, 3] + dt
    return T


def photo_cosine(rgb, img_highpass, pv, target):
    hp = rgb - gaussian_blur(rgb[None])[0]
    m = pv * target
    den = m.sum().clamp_min(1.0)
    hpz = hp - (hp * m[..., None]).sum((0, 1), keepdim=True) / den
    iz = img_highpass - (
        img_highpass * m[..., None]
    ).sum((0, 1), keepdim=True) / den
    return (hpz * iz * m[..., None]).sum() / torch.sqrt(
        (hpz * hpz * m[..., None]).sum()
        * (iz * iz * m[..., None]).sum()
        + 1e-6
    )


def optimize_frame(
    renderer: MeshRenderer,
    T0_np: np.ndarray,
    image: np.ndarray,
    object_mask: np.ndarray,
    person_mask: np.ndarray,
    steps: int,
    lr: float,
    loss_kind: str,
    target_roi=None,
):
    """Optimize a small SE(3) correction using mask + texture/photo cues."""
    T0 = torch.tensor(T0_np, dtype=torch.float32, device=DEVICE)
    if target_roi is None:
        image_t = torch.tensor(image, dtype=torch.float32, device=DEVICE)
        target = torch.tensor(object_mask, dtype=torch.float32, device=DEVICE)
        person = torch.tensor(person_mask, dtype=torch.float32, device=DEVICE)
        person = F.max_pool2d(person[None, None], 5, 1, 2)[0, 0]
        vis = (1.0 - person).clamp(0.0, 1.0)
        target = target * vis
    else:
        image_t, target = target_roi
        vis = torch.ones_like(target)
    img_highpass = image_t - gaussian_blur(image_t[None])[0]

    r = torch.zeros(3, device=DEVICE, requires_grad=True)
    dt = torch.zeros(3, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([r, dt], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        T = perturb_pose(T0, r, dt)
        rgb, alpha = renderer.render(T)
        pv = alpha * vis
        if loss_kind == "dice":
            dice = (2.0 * (pv * target).sum() + 1.0) / (
                pv.sum() + target.sum() + 1.0
            )
            data_loss = 3.0 * (1.0 - dice)
            prior = 0.02 * (r * r).sum() + 1.0 * (dt * dt).sum()
        elif loss_kind == "soft_iou":
            inter = (pv * target).sum()
            union = (pv * pv).sum() + (target * target).sum() - inter
            iou = inter / (union + 1e-9)
            data_loss = 2.0 * (1.0 - iou)
            prior = 0.05 * (r * r).sum() + 2.0 * (dt * dt).sum()
        else:
            raise ValueError(loss_kind)
        cos = photo_cosine(rgb, img_highpass, pv, target)
        loss = data_loss + 0.1 * (1.0 - cos) + prior
        loss.backward()
        opt.step()
    with torch.no_grad():
        return perturb_pose(T0, r, dt).detach().cpu().numpy().astype(np.float64)


def rotation_average(rotations: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(rotations).as_quat()
    q0 = q[0].copy()
    for i in range(len(q)):
        if np.dot(q[i], q0) < 0:
            q[i] *= -1
    qm = q.mean(axis=0)
    qm /= np.linalg.norm(qm)
    return Rotation.from_quat(qm).as_matrix()


def pool_rest_pose(
    renderer,
    per_frame: np.ndarray,
    images: np.ndarray,
    object_masks: np.ndarray,
    start: int,
    end: int,
    init_pose: np.ndarray,
    steps: int,
    lr: float,
    loss_kind: str,
    hull_vertices: torch.Tensor,
    plane_n: torch.Tensor,
    plane_d: float,
):
    """Fit one constant world pose to all frames of a static phase."""
    image = torch.tensor(
        np.median(images[start:end].astype(np.float32) / 255.0, axis=0),
        dtype=torch.float32,
        device=DEVICE,
    )
    target = torch.tensor(
        object_masks[start:end].mean(axis=0), dtype=torch.float32, device=DEVICE
    )
    img_highpass = image - gaussian_blur(image[None])[0]
    T0 = torch.tensor(init_pose, dtype=torch.float32, device=DEVICE)
    r = torch.zeros(3, device=DEVICE, requires_grad=True)
    dt = torch.zeros(3, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([r, dt], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        T = perturb_pose(T0, r, dt)
        rgb, pv = renderer.render(T)
        if loss_kind == "dice":
            dice = 2.0 * (pv * target).sum() / (
                pv.sum() + target.sum() + 1e-9
            )
            data_loss = 3.0 * (1.0 - dice)
            prior = 0.02 * (r * r).sum() + 1.0 * (dt * dt).sum()
        else:
            inter = (pv * target).sum()
            union = (pv * pv).sum() + (target * target).sum() - inter
            iou = inter / (union + 1e-9)
            data_loss = 2.0 * (1.0 - iou)
            prior = 0.05 * (r * r).sum() + 2.0 * (dt * dt).sum()
        cos = photo_cosine(rgb, img_highpass, pv, target)
        vw = hull_vertices @ T[:3, :3].T + T[:3, 3]
        signed = vw @ plane_n + plane_d
        floor = torch.relu(-signed.min() + 0.005) + torch.relu(
            signed.min() - 0.04
        )
        loss = data_loss + 0.1 * (1.0 - cos) + 20.0 * floor * floor + prior
        loss.backward()
        opt.step()
    with torch.no_grad():
        return perturb_pose(T0, r, dt).detach().cpu().numpy().astype(np.float64)


def smooth_trajectory(T: np.ndarray, window: int) -> np.ndarray:
    quat = Rotation.from_matrix(T[:, :3, :3]).as_quat()
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1
    quat = savgol_filter(quat, window, 2, axis=0, mode="interp")
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    trans = savgol_filter(T[:, :3, 3], window, 2, axis=0, mode="interp")
    out = np.tile(np.eye(4, dtype=np.float64), (len(T), 1, 1))
    out[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
    out[:, :3, 3] = trans
    return out


def smoothstep(x: float) -> float:
    return x * x * (3.0 - 2.0 * x)


def blend_pose(A: np.ndarray, B: np.ndarray, w: float) -> np.ndarray:
    slerp = Slerp(
        [0.0, 1.0], Rotation.from_matrix(np.stack([A[:3, :3], B[:3, :3]]))
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = slerp(w).as_matrix()
    out[:3, 3] = (1.0 - w) * A[:3, 3] + w * B[:3, 3]
    return out


def apply_rest_constraints(T, initial_pose, final_pose):
    out = T.copy()
    out[:INITIAL_REST_END] = initial_pose
    for i in range(INITIAL_REST_END, INITIAL_BLEND_END):
        w = smoothstep((i - (INITIAL_REST_END - 1)) /
                       (INITIAL_BLEND_END - (INITIAL_REST_END - 1)))
        out[i] = blend_pose(initial_pose, T[i], w)
    for i in range(FINAL_BLEND_START, FINAL_REST_START):
        w = smoothstep((i - (FINAL_BLEND_START - 1)) /
                       (FINAL_REST_START - (FINAL_BLEND_START - 1)))
        out[i] = blend_pose(T[i], final_pose, w)
    out[FINAL_REST_START:] = final_pose
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path("/work/inputs"))
    parser.add_argument(
        "--out", type=Path, default=Path("/work/outputs/object_poses_world.npz")
    )
    args = parser.parse_args()
    set_deterministic(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.empty_cache()
    started = time.time()

    inputs = args.inputs
    with np.load(inputs / "reconstruction.npz") as z:
        baseline = z["T_world_mesh"].astype(np.float64)
        human_keypoints = z["human_keypoints_world"]
        contact_prob = z["contact_prob_right_left"][:, 1].astype(np.float64)
    frame_count = len(baseline)

    cap = cv2.VideoCapture(str(inputs / "video.mp4"))
    images = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        images.append(
            frame[ROI_Y0 : ROI_Y0 + ROI_H, ROI_X0 : ROI_X0 + ROI_W].copy()
        )
    cap.release()
    images = np.stack(images)
    if len(images) != frame_count:
        raise RuntimeError(f"video has {len(images)} frames, expected {frame_count}")

    with h5py.File(inputs / "masks.h5", "r") as h5:
        object_masks = h5["object"][
            :, ROI_Y0 : ROI_Y0 + ROI_H, ROI_X0 : ROI_X0 + ROI_W
        ]
        person_masks = h5["person"][
            :, ROI_Y0 : ROI_Y0 + ROI_H, ROI_X0 : ROI_X0 + ROI_W
        ]

    renderer = MeshRenderer(inputs)
    print(f"loaded {frame_count} frames, mesh={len(renderer.vertices)} vertices", flush=True)

    # 1) Independent mask/photo refinement seeded by the supplied trajectory.
    per_frame = np.empty_like(baseline)
    for i in range(frame_count):
        per_frame[i] = optimize_frame(
            renderer,
            baseline[i],
            images[i].astype(np.float32) / 255.0,
            object_masks[i],
            person_masks[i],
            PER_FRAME_STEPS,
            0.0015,
            "dice",
        )
        if i % 25 == 0:
            print(f"per-frame {i}/{frame_count} ({time.time()-started:.1f}s)", flush=True)

    # 2) Pool each static phase into one rigid pose.  Static evidence is much
    # stronger than a single-frame silhouette, so this also removes tracker drift.
    with open(inputs / "ground_plane.json", "r", encoding="utf-8") as f:
        ground = json.load(f)
    plane = np.asarray(ground["plane"], dtype=np.float64)
    process_mesh = trimesh.load(
        str(inputs / "object_mesh.glb"), force="scene", process=True
    ).to_geometry()
    hull_vertices = torch.tensor(
        process_mesh.convex_hull.vertices, dtype=torch.float32, device=DEVICE
    )
    plane_n = torch.tensor(plane[:3], dtype=torch.float32, device=DEVICE)
    plane_d = float(plane[3])

    coarse = {}
    for name, a, b in (("initial", 0, 110), ("final", 640, frame_count)):
        init = np.eye(4, dtype=np.float64)
        init[:3, :3] = rotation_average(per_frame[a:b, :3, :3])
        init[:3, 3] = np.median(per_frame[a:b, :3, 3], axis=0)
        coarse[name] = pool_rest_pose(
            renderer, per_frame, images, object_masks, a, b, init,
            200, 0.0015, "dice", hull_vertices, plane_n, plane_d,
        )
        print(f"pooled {name} ({time.time()-started:.1f}s)", flush=True)

    pooled = {}
    for name, a, b in (
        ("initial", 0, INITIAL_REST_END),
        ("final", FINAL_REST_START, frame_count),
    ):
        pooled[name] = pool_rest_pose(
            renderer, per_frame, images, object_masks, a, b, coarse[name],
            120, 0.0007, "soft_iou", hull_vertices, plane_n, plane_d,
        )
    initial_pose, final_pose = pooled["initial"], pooled["final"]

    # 3) Smooth the per-frame dynamic estimates, then blend into the pooled rest
    # poses so phase changes are continuous rather than one-frame jumps.
    smooth9 = smooth_trajectory(per_frame, 9)
    candidate = apply_rest_constraints(smooth9, initial_pose, final_pose)

    # 4) Re-fit the manipulation frames with a soft-IoU objective to recover the
    # silhouette area lost by the smoother, while retaining the smooth initialisation.
    refined = candidate.copy()
    for i in range(INITIAL_REST_END, FINAL_REST_START):
        refined[i] = optimize_frame(
            renderer,
            candidate[i],
            images[i].astype(np.float32) / 255.0,
            object_masks[i],
            person_masks[i],
            REFINE_STEPS,
            0.0008,
            "soft_iou",
        )
        if i % 25 == 0:
            print(f"refine {i}/{FINAL_REST_START} ({time.time()-started:.1f}s)", flush=True)

    # A short quadratic Savitzky-Golay filter removes remaining observation
    # noise.  Mask IoU changes by <0.002 on the validation set while frame-to-
    # frame rotation/translation jitter falls by roughly 3x.
    smooth7 = smooth_trajectory(refined, 7)
    final_poses = apply_rest_constraints(smooth7, initial_pose, final_pose)
    final_poses = final_poses.astype(np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T_world_mesh=final_poses)
    print(
        f"wrote {args.out} shape={final_poses.shape} "
        f"in {time.time()-started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
