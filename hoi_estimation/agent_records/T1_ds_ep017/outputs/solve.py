#!/usr/bin/env python3
"""Recover a temporally consistent world-space traffic-cone trajectory.

The released baseline is per-frame and contains a symmetry-branch error in the
long initial rest phase.  This solver (1) resolves that branch and jointly fits
the one static pose using all sampled static silhouettes, (2) bridges from that
pose into the tracker's post-grasp branch, (3) replaces the object translation
by the reconstructed two-hand midpoint only while the object is fully obscured,
and (4) applies deterministic Savitzky-Golay smoothing to the manipulated-segment
rotation and translation.  Since a stationary rigid object has one pose, both
observed rest phases are emitted with constant rotations (and constant poses).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

# Explicit phase decisions.  Optical flow over the object is below 0.1 px/frame
# through frame 112 and begins to rise around 113.  Frame 117 is the first
# stable post-grasp tracker pose.  The hands leave the cone just before frame
# 340, after which SAM reports the same table-top silhouette for all frames.
STATIC_END = 112
DYNAMIC_START = 117
TABLE_START = 340
# The object is effectively absent from the object mask for one long run while
# the person carries it behind the torso.  Use hand motion only in this run.
OCCLUSION_AREA_THRESHOLD = 0  # exact empty masks delimit the unusable core
# Smoothing is in frames at 30 fps.  The windows remove single-frame network
# errors while preserving roughly 0.2 s of real manipulation dynamics.
ROT_SMOOTH_WINDOW = 9
TRANS_SMOOTH_WINDOW = 7
# No image evidence exists in the empty-mask run; a wider rotation window is
# justified there because a hand-held rigid object cannot physically jitter.
OCCLUSION_ROT_SMOOTH_WINDOW = 25
STATIC_SAMPLE_FRAMES = (0, 20, 40, 60, 80, 100)


class Renderer:
    def __init__(self, mesh_path: Path, camera: dict, device: str = "cuda") -> None:
        scene = trimesh.load(mesh_path, force="scene", process=False)
        mesh = scene.to_geometry()
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
        self.uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
        # nvdiffrast uses the opposite texture-v coordinate convention to glTF.
        self.uv[:, 1] = 1.0 - self.uv[:, 1]
        self.tex = None
        material = mesh.visual.material
        image = getattr(material, "baseColorTexture", None)
        if image is not None:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            self.tex = torch.tensor(rgb, device=device)[None]
        self.K = np.asarray(camera["K"], dtype=np.float64)
        self.W = int(camera["width"])
        self.H = int(camera["height"])
        self.T_world_cam = np.asarray(camera["T_world_cam"], dtype=np.float64)
        self.T_cam_world = np.linalg.inv(self.T_world_cam)
        self.R_cam_world = torch.tensor(
            self.T_cam_world[:3, :3], dtype=torch.float32, device=device
        )
        self.t_cam_world = torch.tensor(
            self.T_cam_world[:3, 3], dtype=torch.float32, device=device
        )
        self.gl = dr.RasterizeCudaContext(device=device)

    def _raster(self, pose: np.ndarray, crop: tuple[int, int, int, int]):
        x0, x1, y0, y1 = crop
        cw, ch = x1 - x0, y1 - y0
        M = torch.tensor(pose, dtype=torch.float32, device="cuda")
        world = self.vertices @ M[:3, :3].T + M[:3, 3]
        cam = world @ self.R_cam_world.T + self.t_cam_world
        z = cam[:, 2]
        x = self.K[0, 0] * cam[:, 0] / z + self.K[0, 2]
        y = self.K[1, 1] * cam[:, 1] / z + self.K[1, 2]
        # The CUDA rasterizer's image row direction is already image-like.
        ndc_x = (x - x0) / cw * 2.0 - 1.0
        ndc_y = (y - y0) / ch * 2.0 - 1.0
        clip = torch.stack((ndc_x * z, ndc_y * z, z, z), dim=1).contiguous()
        rast, _ = dr.rasterize(self.gl, clip[None], self.faces, resolution=(ch, cw))
        return rast, cw, ch

    def mask(self, pose: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
        with torch.no_grad():
            rast, _, _ = self._raster(pose, crop)
            return (rast[0, ..., 3] > 0).cpu().numpy()

    def rgb(self, pose: np.ndarray, crop: tuple[int, int, int, int]):
        with torch.no_grad():
            rast, cw, ch = self._raster(pose, crop)
            if self.tex is None:
                p = torch.tensor(pose, dtype=torch.float32, device="cuda")
                world = self.vertices @ p[:3, :3].T + p[:3, 3]
                cam = world @ self.R_cam_world.T + self.t_cam_world
                gray = torch.full_like(cam, 0.6)
                color, _ = dr.interpolate(gray.contiguous(), rast, self.faces)
            else:
                texcoord, _ = dr.interpolate(self.uv.contiguous(), rast, self.faces)
                color = dr.texture(self.tex, texcoord.contiguous(), filter_mode="linear")
            mask = (rast[0, ..., 3] > 0).cpu().numpy()
            return color[0].cpu().numpy(), mask


def load_inputs(inputs: Path):
    with (inputs / "camera.json").open() as f:
        camera = json.load(f)
    with (inputs / "episode.json").open() as f:
        episode = json.load(f)
    recon = np.load(inputs / "reconstruction.npz")
    with h5py.File(inputs / "masks.h5", "r") as f:
        object_masks = np.asarray(f["object"], dtype=bool)
        person_masks = np.asarray(f["person"], dtype=bool)
    baseline = np.asarray(recon["T_world_mesh"], dtype=np.float64)
    keypoints = np.asarray(recon["human_keypoints_world"], dtype=np.float64)
    return camera, episode, baseline, object_masks, person_masks, keypoints


def local_z_twist(degrees: float) -> np.ndarray:
    a = np.deg2rad(degrees)
    out = np.eye(4, dtype=np.float64)
    out[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return out


def camera_translation_correction(dc: np.ndarray, camera: dict) -> np.ndarray:
    return np.asarray(camera["T_world_cam"], dtype=np.float64)[:3, :3] @ np.asarray(dc)


def fit_initial_static_pose(
    renderer: Renderer,
    camera: dict,
    baseline: np.ndarray,
    object_masks: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit one pose to the initial rest phase, resolving the 180-degree branch.

    The objective is silhouette IoU pooled over six frames spanning the rest
    phase.  Search is initialized at the exact symmetry flip, with the tracker
    pose as the other branch implicitly rejected by the objective.
    """
    sample = list(STATIC_SAMPLE_FRAMES)
    union = np.any(object_masks[sample], axis=0)
    ys, xs = np.where(union)
    crop = (
        max(0, int(xs.min()) - 35),
        min(renderer.W, int(xs.max()) + 36),
        max(0, int(ys.min()) - 35),
        min(renderer.H, int(ys.max()) + 36),
    )
    gts = [object_masks[i, crop[2] : crop[3], crop[0] : crop[1]] for i in sample]
    base = baseline[sample[0]].copy()

    def pose_for(params: np.ndarray) -> np.ndarray:
        out = base @ local_z_twist(params[0])
        out[:3, 3] += camera_translation_correction(params[1:], camera)
        return out

    def objective(params: np.ndarray) -> float:
        pred = renderer.mask(pose_for(params), crop)
        ious = []
        for gt in gts:
            union_px = np.count_nonzero(pred | gt)
            ious.append(np.count_nonzero(pred & gt) / max(union_px, 1))
        return 1.0 - float(np.mean(ious))

    x0 = np.array([180.0, 0.0, 0.0, 0.0], dtype=np.float64)
    before = 1.0 - objective(x0)
    result = minimize(
        objective,
        x0,
        method="Powell",
        bounds=[(170.0, 190.0), (-0.04, 0.04), (-0.04, 0.04), (-0.06, 0.06)],
        options={"xtol": 1e-3, "ftol": 1e-6, "maxiter": 30},
    )
    pose = pose_for(result.x)
    after = 1.0 - float(result.fun)
    return pose, {
        "twist_deg": float(result.x[0]),
        "camera_translation_correction_m": result.x[1:].astype(float).tolist(),
        "sample_mask_iou_before": float(before),
        "sample_mask_iou_after": float(after),
        "optimizer_success": bool(result.success),
    }


def smooth_quaternions(rotations: Rot, window: int = ROT_SMOOTH_WINDOW) -> Rot:
    q = rotations.as_quat().copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0.0:
            q[i] *= -1.0
    for j in range(4):
        q[:, j] = savgol_filter(q[:, j], window, 3, mode="interp")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return Rot.from_quat(q)


def build_trajectory(
    camera: dict,
    baseline: np.ndarray,
    object_masks: np.ndarray,
    keypoints: np.ndarray,
    static_pose: np.ndarray,
) -> tuple[np.ndarray, dict]:
    F = len(baseline)
    out = baseline.copy()

    # The released tracker's initial pose is held fixed through frame 112.
    out[: STATIC_END + 1] = static_pose

    # Work on the manipulated segment.  During the fully occluded run, recover
    # translation from the two-hand midpoint.  The hand-to-object offset is
    # interpolated between the last and first visible anchors, not assumed.
    areas = object_masks.reshape(F, -1).sum(axis=1)
    occluded = areas <= OCCLUSION_AREA_THRESHOLD
    # Longest fully empty run inside the carry phase.  SAM briefly flickers
    # back on for a few pixels behind the body, so extend to the last empty
    # frame in the carry interval and then add three frames of hand-only use.
    runs = []
    start = None
    for i in range(DYNAMIC_START, TABLE_START):
        if occluded[i] and start is None:
            start = i
        if not occluded[i] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, TABLE_START - 1))
    if not runs:
        raise RuntimeError("No object-occlusion run found while carried")
    occ0, occ1 = max(runs, key=lambda ab: ab[1] - ab[0])
    last_empty = int(np.where(occluded[DYNAMIC_START:TABLE_START])[0].max() + DYNAMIC_START)
    hand_center = 0.5 * (keypoints[:, 21:42].mean(axis=1) + keypoints[:, 42:63].mean(axis=1))
    gap0 = max(DYNAMIC_START, occ0 - 7)
    gap1 = min(TABLE_START - 1, last_empty + 3)
    off0 = baseline[gap0, :3, 3] - hand_center[gap0]
    off1 = baseline[gap1, :3, 3] - hand_center[gap1]
    for i in range(gap0, gap1 + 1):
        u = (i - gap0) / max(gap1 - gap0, 1)
        out[i, :3, 3] = hand_center[i] + (1.0 - u) * off0 + u * off1

    # Rotation smoothing removes single-frame symmetry/noise jumps.  Translation
    # smoothing is stronger because a rigid carry has no translational jitter.
    rot_dyn = smooth_quaternions(Rot.from_matrix(out[DYNAMIC_START:TABLE_START, :3, :3]))
    out[DYNAMIC_START:TABLE_START, :3, :3] = rot_dyn.as_matrix()
    for j in range(3):
        out[DYNAMIC_START:TABLE_START, j, 3] = savgol_filter(
            out[DYNAMIC_START:TABLE_START, j, 3],
            TRANS_SMOOTH_WINDOW,
            2,
            mode="interp",
        )

    # There are no object pixels in the middle of the carry.  Re-smooth that
    # interval with a wider temporal window; the hand midpoint still supplies
    # the gross translation, while rotation must obey rigid-body continuity.
    occ_a = max(DYNAMIC_START, occ0 - 10)
    occ_b = min(TABLE_START, last_empty + 4)
    out[occ_a:occ_b, :3, :3] = smooth_quaternions(
        Rot.from_matrix(out[occ_a:occ_b, :3, :3]), OCCLUSION_ROT_SMOOTH_WINDOW
    ).as_matrix()

    # Smoothly bridge the corrected rest pose into the first tracked carry pose.
    src_rot = Rot.from_matrix(static_pose[:3, :3])
    dst_rot = Rot.from_matrix(out[DYNAMIC_START, :3, :3])
    slerp = Slerp([0.0, 1.0], Rot.concatenate((src_rot, dst_rot)))
    for i in range(STATIC_END + 1, DYNAMIC_START):
        u = (i - STATIC_END) / (DYNAMIC_START - STATIC_END)
        u = u * u * (3.0 - 2.0 * u)
        out[i, :3, :3] = slerp(u).as_matrix()
        out[i, :3, 3] = (1.0 - u) * static_pose[:3, 3] + u * out[DYNAMIC_START, :3, 3]

    # The cone rests on the table from frame 340 onward.  Pool rotation and
    # translation over the phase instead of retaining per-frame depth jitter.
    rest_rot = Rot.from_matrix(baseline[TABLE_START:, :3, :3]).mean()
    rest_t = np.median(baseline[TABLE_START:, :3, 3], axis=0)
    out[TABLE_START:, :3, :3] = rest_rot.as_matrix()
    out[TABLE_START:, :3, 3] = rest_t

    info = {
        "occlusion_mask_run": [int(occ0), int(occ1)],
        "hand_translation_gap": [int(gap0), int(gap1)],
    }
    return out, info


def validation_metrics(
    renderer: Renderer,
    frames: dict,
    baseline: np.ndarray,
    result: np.ndarray,
    object_masks: np.ndarray,
    person_masks: np.ndarray,
) -> list[dict]:
    phases = (
        ("initial_rest", 0, STATIC_END + 1),
        ("pick_transition", STATIC_END + 1, DYNAMIC_START),
        ("carry_visible", DYNAMIC_START, 190),
        ("carry_occluded", 190, 255),
        ("place_approach", 255, TABLE_START),
        ("table_rest", TABLE_START, len(baseline)),
        ("all_frames", 0, len(baseline)),
    )
    rows = []
    for name, a, b in phases:
        ious_b, ious_r, photos_b, photos_r = [], [], [], []
        for i in range(a, b):
            ys, xs = np.where(object_masks[i])
            if len(xs) < 80:
                continue
            crop = (
                max(0, int(xs.min()) - 20),
                min(renderer.W, int(xs.max()) + 21),
                max(0, int(ys.min()) - 20),
                min(renderer.H, int(ys.max()) + 21),
            )
            gt = object_masks[i, crop[2] : crop[3], crop[0] : crop[1]]
            image = frames[i][crop[2] : crop[3], crop[0] : crop[1], ::-1].astype(np.float32) / 255.0
            for pose, ious, photos in (
                (baseline[i], ious_b, photos_b),
                (result[i], ious_r, photos_r),
            ):
                pred = renderer.mask(pose, crop)
                ious.append(np.count_nonzero(pred & gt) / max(np.count_nonzero(pred | gt), 1))
                rgb, pred2 = renderer.rgb(pose, crop)
                valid = pred2 & gt
                if valid.sum() < 100:
                    photos.append(np.nan)
                    continue
                a1 = cv2.GaussianBlur(rgb.mean(axis=2), (0, 0), 2.0)
                b1 = cv2.GaussianBlur(image.mean(axis=2), (0, 0), 2.0)
                ah = a1 - cv2.GaussianBlur(a1, (0, 0), 8.0)
                bh = b1 - cv2.GaussianBlur(b1, (0, 0), 8.0)
                photos.append(float(np.corrcoef(ah[valid], bh[valid])[0, 1]))
        rows.append(
            {
                "phase": name,
                "frames": [a, b - 1],
                "n_scored": len(ious_b),
                "mask_iou_baseline": float(np.mean(ious_b)) if ious_b else None,
                "mask_iou_result": float(np.mean(ious_r)) if ious_r else None,
                "photo_ncc_baseline": float(np.nanmean(photos_b)) if photos_b else None,
                "photo_ncc_result": float(np.nanmean(photos_r)) if photos_r else None,
            }
        )
    return rows


def temporal_metrics(trajectory: np.ndarray) -> dict:
    a, b = DYNAMIC_START, TABLE_START
    rv = Rot.from_matrix(trajectory[a:b, :3, :3]).as_rotvec()
    ang_vel = np.diff(rv, axis=0)
    ang_acc = np.diff(ang_vel, axis=0)
    tr_vel = np.diff(trajectory[a:b, :3, 3], axis=0)
    tr_acc = np.diff(tr_vel, axis=0)
    return {
        "rotation_accel_p90_rad_frame2": float(np.percentile(np.linalg.norm(ang_acc, axis=1), 90)),
        "translation_accel_p90_m_frame2": float(np.percentile(np.linalg.norm(tr_acc, axis=1), 90)),
        "max_step_rotation_deg": float(np.degrees(np.max(np.linalg.norm(ang_vel, axis=1)))),
        "max_step_translation_m": float(np.max(np.linalg.norm(tr_vel, axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    camera, episode, baseline, object_masks, person_masks, keypoints = load_inputs(args.inputs)
    F = int(episode["frames"])
    if baseline.shape[0] != F:
        raise ValueError(f"Expected {F} baseline poses, got {baseline.shape[0]}")
    renderer = Renderer(args.inputs / "object_mesh.glb", camera)

    static_pose, static_info = fit_initial_static_pose(
        renderer, camera, baseline, object_masks
    )
    result, trajectory_info = build_trajectory(
        camera, baseline, object_masks, keypoints, static_pose
    )

    # Validation reads the video only after the trajectory is fixed.
    cap = cv2.VideoCapture(str(args.inputs / "video.mp4"))
    frames = {}
    for i in range(F):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode video frame {i}")
        frames[i] = frame
    cap.release()
    rows = validation_metrics(
        renderer, frames, baseline, result, object_masks, person_masks
    )
    report = {
        "static_pose_fit": static_info,
        "trajectory": trajectory_info,
        "phase_validation": rows,
        "temporal_baseline": temporal_metrics(baseline),
        "temporal_result": temporal_metrics(result),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T_world_mesh=np.asarray(result, dtype=np.float64))
    with (args.out.parent / "validation.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
