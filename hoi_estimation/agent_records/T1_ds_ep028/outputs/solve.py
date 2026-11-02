#!/usr/bin/env python3
"""Deterministic trajectory reconstruction for the squeegee episode.

The released reconstruction is used as a frame-wise proposal.  The method then
uses three forms of temporal evidence that the proposal does not enforce:
  * one constant pose during the two visually stationary phases;
  * robust medoid smoothing to remove isolated tracker inversion jumps;
  * mask-aligned roll hypotheses on unambiguous keyframes around the difficult
    carry/return segment, interpolated in SE(3).

Only the supplied inputs are used.  The renderer is silhouette/UV only; it is
used to score hypotheses, not to train anything.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

import nvdiffrast.torch as dr


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pca_features(mask: np.ndarray):
    ys, xs = np.where(mask)
    pts = np.column_stack((xs, ys)).astype(np.float64)
    center = pts.mean(0)
    q = pts - center
    cov = q.T @ q / max(1, len(q))
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, -1]
    if axis[0] < 0:
        axis = -axis
    angle = np.arctan2(axis[1], axis[0])
    proj = q @ axis
    length = np.percentile(proj, 98) - np.percentile(proj, 2)
    return center, angle, length, len(xs)


class MeshRenderer:
    def __init__(self, mesh_path: Path, camera: dict, device: str = "cuda"):
        self.device = torch.device(device)
        self.width = int(camera["width"])
        self.height = int(camera["height"])
        self.K = np.asarray(camera["K"], np.float32)
        self.Wc = np.asarray(camera["T_world_cam"], np.float64)
        self.Twc = np.linalg.inv(self.Wc)

        mesh = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
        self.mesh = mesh
        self.vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.int32, device=self.device)
        uv = np.asarray(mesh.visual.uv).copy()
        # trimesh loads the image top-left; nvdiffrast samples lower-left.
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, dtype=torch.float32, device=self.device)
        texture = np.asarray(mesh.visual.material.baseColorTexture).astype(np.float32) / 255.0
        self.texture = torch.tensor(texture[..., :3], device=self.device)
        self.ctx = dr.RasterizeCudaContext()
        W, H = self.width, self.height
        k = self.K
        # NDC projection (nvdiffrast's y axis is top-down for this setup).
        self.proj = torch.tensor(
            [[2 * k[0, 0] / W, 0, 2 * k[0, 2] / W - 1, 0],
             [0, 2 * k[1, 1] / H, 2 * k[1, 2] / H - 1, 0],
             [0, 0, 1, 0],
             [0, 0, 1, 0]],
            dtype=torch.float32,
            device=self.device,
        )

    def render(self, T_world_mesh: np.ndarray):
        T_cam_mesh = torch.tensor(self.Twc, dtype=torch.float32, device=self.device) @ torch.tensor(
            T_world_mesh, dtype=torch.float32, device=self.device
        )
        A = T_cam_mesh[:3, :]
        vh = torch.cat([self.vertices, torch.ones_like(self.vertices[:, :1])], 1)
        vc = (A @ vh.T).T
        vclip = self.proj @ torch.cat([vc, torch.ones_like(vc[:, :1])], 1).T
        rast, _ = dr.rasterize(
            self.ctx, vclip.T[None].contiguous(), self.faces.contiguous(),
            resolution=[self.height, self.width],
        )
        uv, _ = dr.interpolate(self.uv[None], rast, self.faces)
        rgb = dr.texture(self.texture[None], uv.contiguous(), filter_mode="linear", boundary_mode="wrap")[0]
        return rgb.cpu().numpy(), (rast[0, ..., 3] > 0).cpu().numpy()


def camera_adjust(T_cam: np.ndarray, K: np.ndarray, theta=0.0, scale_z=1.0, dx=0.0, dy=0.0):
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ T_cam[:3, :3]
    t = Rz @ T_cam[:3, 3]
    t[2] = t[2] / scale_z
    t[0] += dx * t[2] / K[0, 0]
    t[1] += dy * t[2] / K[1, 1]
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def mask_score(rendered: np.ndarray, observed: np.ndarray) -> float:
    if not rendered.any():
        return 999.0
    d_obs = distance_transform_edt(~observed)
    d_rendered = distance_transform_edt(~rendered)
    area = max(1, observed.sum())
    iou = (rendered & observed).sum() / max(1, (rendered | observed).sum())
    return (
        d_obs[rendered].mean()
        + d_rendered[observed].mean()
        + 5.0 * abs(rendered.sum() - area) / area
        - 2.0 * iou
    )


def consensus_pose(renderer: MeshRenderer, baseline: np.ndarray, object_masks: np.ndarray,
                   person_masks: np.ndarray, frame_lo: int, frame_hi: int,
                   candidate_lo: int, candidate_hi: int, crop, candidate_step: int = 2):
    """Pick one static-phase pose by pooling a person-excluded mask over time."""
    x0, x1, y0, y1 = crop
    obj_freq = object_masks[frame_lo:frame_hi].mean(0)
    person_freq = person_masks[frame_lo:frame_hi].mean(0)
    observed = (obj_freq > 0.18) & (person_freq < 0.35)
    observed = observed[y0:y1, x0:x1]
    d_obs = distance_transform_edt(~observed)
    area = max(1, observed.sum())
    best = None
    for j in range(candidate_lo, candidate_hi, candidate_step):
        _, mask = renderer.render(baseline[j].astype(np.float32))
        mask = mask[y0:y1, x0:x1]
        d_mask = distance_transform_edt(~mask)
        iou = (mask & observed).sum() / max(1, (mask | observed).sum())
        score = d_obs[mask].mean() + d_mask[observed].mean()
        score += 5.0 * abs(mask.sum() - area) / area - 2.0 * iou
        if best is None or score < best[0]:
            best = (score, j)
    return best[1], best[0]


def refine_static_translation(renderer: MeshRenderer, world_pose: np.ndarray, baseline: np.ndarray,
                              object_masks: np.ndarray, person_masks: np.ndarray,
                              frame_lo: int, frame_hi: int, crop):
    """Pooled silhouette search for the small residual camera-space depth/shift."""
    x0, x1, y0, y1 = crop
    obj_freq = object_masks[frame_lo:frame_hi].mean(0)
    person_freq = person_masks[frame_lo:frame_hi].mean(0)
    observed = ((obj_freq > 0.18) & (person_freq < 0.35))[y0:y1, x0:x1]
    d_obs = distance_transform_edt(~observed)
    area = max(1, observed.sum())
    T_base = renderer.Twc @ world_pose
    best = None

    def evaluate(delta):
        Tc = T_base.copy()
        Tc[:3, 3] += np.asarray(delta, np.float64)
        _, mask = renderer.render((renderer.Wc @ Tc).astype(np.float32))
        mask = mask[y0:y1, x0:x1]
        d_mask = distance_transform_edt(~mask)
        iou = (mask & observed).sum() / max(1, (mask | observed).sum())
        score = d_obs[mask].mean() + d_mask[observed].mean()
        score += 5.0 * abs(mask.sum() - area) / area - 2.0 * iou
        return score, Tc

    for dz in np.arange(-0.10, 0.101, 0.02):
        for dy in np.arange(-0.04, 0.041, 0.01):
            for dx in np.arange(-0.04, 0.041, 0.01):
                score, Tc = evaluate((dx, dy, dz))
                if best is None or score < best[0]:
                    best = (score, np.array([dx, dy, dz]), Tc)
    coarse = best[1]
    for ddz in np.arange(coarse[2] - 0.015, coarse[2] + 0.0151, 0.005):
        for ddy in np.arange(coarse[1] - 0.0075, coarse[1] + 0.0076, 0.0025):
            for ddx in np.arange(coarse[0] - 0.0075, coarse[0] + 0.0076, 0.0025):
                score, Tc = evaluate((ddx, ddy, ddz))
                if score < best[0]:
                    best = (score, np.array([ddx, ddy, ddz]), Tc)
    return (renderer.Wc @ best[2]).astype(np.float64), best[0]


def smooth_trajectory(baseline: np.ndarray, window: int = 13, threshold_deg: float = 8.0):
    """Robust medoid average in SO(3) plus median translation filtering."""
    quats = Rotation.from_matrix(baseline[:, :3, :3]).as_quat()
    trans = baseline[:, :3, 3].copy()
    out_q = np.empty_like(quats)
    half = window // 2
    for i in range(len(quats)):
        a, b = max(0, i - half), min(len(quats), i + half + 1)
        qq = quats[a:b].copy()
        # A medoid, rather than the current frame, is the robust local reference.
        medoid = qq[np.argmax(np.abs(qq @ qq.T).sum(1))]
        signs = np.sign(qq @ medoid)
        signs[signs == 0] = 1.0
        qq *= signs[:, None]
        ref = medoid * np.sign(medoid @ quats[i])
        angles = 2.0 * np.arccos(np.clip(qq @ ref, -1.0, 1.0))
        weights = np.ones(len(qq))
        bad = angles > np.radians(threshold_deg)
        weights[bad] = np.radians(threshold_deg) / angles[bad]
        mean_q = (qq * weights[:, None]).sum(0)
        out_q[i] = mean_q / np.linalg.norm(mean_q)
    out_t = np.array([
        np.median(trans[max(0, i - half):min(len(trans), i + half + 1)], axis=0)
        for i in range(len(trans))
    ])
    out = baseline.copy()
    out[:, :3, :3] = Rotation.from_quat(out_q).as_matrix()
    out[:, :3, 3] = out_t
    return out


def slerp_pose(A: np.ndarray, B: np.ndarray, u: float) -> np.ndarray:
    qa = Rotation.from_matrix(A[:3, :3]).as_quat()
    qb = Rotation.from_matrix(B[:3, :3]).as_quat()
    if qa @ qb < 0:
        qb = -qb
    q = qa * (1.0 - u) + qb * u
    q /= np.linalg.norm(q)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(q).as_matrix()
    out[:3, 3] = A[:3, 3] * (1.0 - u) + B[:3, 3] * u
    return out


def roll_world(T_world: np.ndarray, mesh_center: np.ndarray, degrees: float) -> np.ndarray:
    C = np.eye(4)
    C[:3, 3] = mesh_center
    Ci = np.eye(4)
    Ci[:3, 3] = -mesh_center
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    Rz = np.eye(4)
    Rz[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T_world @ C @ Rz @ Ci


def select_anchor_pose(renderer: MeshRenderer, baseline_pose: np.ndarray, frame_rgb: np.ndarray,
                       object_mask: np.ndarray, person_mask: np.ndarray,
                       mesh_center: np.ndarray, crop=(350, 1050, 500, 1120)):
    """Search the otherwise ambiguous roll about the long axis on a keyframe.

    The long-axis direction is kept from the baseline.  Each roll is centered and
    rotated in the image to match the observed mask PCA, then scored with a
    visibility-aware symmetric mask distance and a weak texture correlation.
    """
    x0, x1, y0, y1 = crop
    observed_all = object_mask
    observed_vis = object_mask & ~person_mask
    obs_crop = observed_all[y0:y1, x0:x1]
    vis_crop = observed_vis[y0:y1, x0:x1]
    d_obs = distance_transform_edt(~vis_crop)
    co = pca_features(observed_all)[0]
    ao = pca_features(observed_all)[1]
    T_base_cam = renderer.Twc @ baseline_pose

    def evaluate(roll):
        Tcam = renderer.Twc @ roll_world(baseline_pose, mesh_center, roll)
        _, m0 = renderer.render((renderer.Wc @ Tcam).astype(np.float32))
        cm = pca_features(m0)[0]
        am = pca_features(m0)[1]
        dtheta = (ao - am + np.pi / 2.0) % np.pi - np.pi / 2.0
        Tc = camera_adjust(Tcam, renderer.K, dtheta)
        _, m1 = renderer.render((renderer.Wc @ Tc).astype(np.float32))
        c1 = pca_features(m1)[0]
        Tc = camera_adjust(Tc, renderer.K, 0.0, 1.0, co[0] - c1[0], co[1] - c1[1])
        rgb, mask = renderer.render((renderer.Wc @ Tc).astype(np.float32))
        vm = (mask & ~person_mask)[y0:y1, x0:x1]
        if not vm.any():
            return 999.0, Tc
        d_mask = distance_transform_edt(~vm)
        area = max(1, vis_crop.sum())
        iou = (vm & vis_crop).sum() / max(1, (vm | vis_crop).sum())
        ad = abs(vm.sum() - area) / area
        score = d_obs[vm].mean() + 0.5 * d_mask[vis_crop].mean() + 3.0 * ad - 2.0 * iou
        valid = vm & vis_crop
        if valid.sum() > 50:
            a = rgb[y0:y1, x0:x1][valid]
            b = frame_rgb[y0:y1, x0:x1][valid]
            aa, bb = a - a.mean(0), b - b.mean(0)
            ncc = ((aa * bb).sum(0) /
                   (np.sqrt((aa * aa).sum(0) * (bb * bb).sum(0)) + 1e-9)).mean()
            score += 0.05 * max(0.0, ncc)
        return score, Tc

    candidates = []
    for roll in range(-180, 180, 30):
        score, Tc = evaluate(roll)
        candidates.append((score, roll, Tc))
    # Include the unmodified proposal and locally refine the two best rolls.
    for score, roll, Tc in sorted(candidates, key=lambda z: z[0])[:2]:
        for rr in range(int(roll) - 15, int(roll) + 16, 5):
            s2, T2 = evaluate(rr)
            if s2 < score:
                score, roll, Tc = s2, rr, T2
    best = min(candidates, key=lambda z: z[0])
    return (renderer.Wc @ best[2]).astype(np.float64), best[0], best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    t0 = time.time()
    set_seed(0)

    inp = Path(args.inputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    camera = json.loads((inp / "camera.json").read_text())
    episode = json.loads((inp / "episode.json").read_text())
    n_frames = int(episode["frames"])
    rec = np.load(inp / "reconstruction.npz")
    baseline = rec["T_world_mesh"].astype(np.float64)
    assert baseline.shape == (n_frames, 4, 4)

    with h5py.File(inp / "masks.h5", "r") as f:
        object_masks = f["object"][:]
        person_masks = f["person"][:]
    assert object_masks.shape[0] == n_frames

    renderer = MeshRenderer(inp / "object_mesh.glb", camera)
    mesh_center = np.asarray(renderer.mesh.vertices, np.float64).mean(0)

    # Phase boundaries were verified visually and with mask statistics:
    # the object is stationary while leaning on the stool until the reach at
    # ~frame 110, and is stationary again after it has been leaned back down at
    # ~frame 840.  Contact probabilities and mask bboxes support both ends.
    START_STATIC_END = 110
    END_STATIC_START = 840

    # Pool all observations inside each static phase.  The start phase has a
    # near-symmetric long axis; the pooled mask selects the stable orientation
    # cluster around frames 60-75 (the final choice is automatic).
    sj, sscore = consensus_pose(
        renderer, baseline, object_masks, person_masks,
        0, 111, 0, 111, (380, 850, 580, 960), 2,
    )
    ej, escore = consensus_pose(
        renderer, baseline, object_masks, person_masks,
        850, 988, 850, 988, (680, 930, 560, 1080), 2,
    )
    start_pose = baseline[sj].copy()
    start_pose[:3, 3] = np.median(baseline[0:111, :3, 3], axis=0)
    end_pose = baseline[ej].copy()
    end_pose[:3, 3] = np.median(baseline[850:988, :3, 3], axis=0)
    start_pose, _ = refine_static_translation(
        renderer, start_pose, baseline, object_masks, person_masks,
        0, 111, (380, 850, 580, 960),
    )
    end_pose, _ = refine_static_translation(
        renderer, end_pose, baseline, object_masks, person_masks,
        850, 988, (680, 930, 560, 1080),
    )
    print(f"static anchors: start frame {sj} (score {sscore:.3f}), end frame {ej} (score {escore:.3f})", flush=True)

    # Remove isolated tracker inversions while preserving the measured motion.
    trajectory = smooth_trajectory(baseline, window=13, threshold_deg=8.0)
    trajectory[:START_STATIC_END + 1] = start_pose
    trajectory[END_STATIC_START:] = end_pose

    # Difficult carry/turn/place segment: recompute selected roll hypotheses on
    # keyframes.  Frames 760 and 780 are intentionally interpolated because the
    # object is heavily occluded there; 800-830 are interpolated into the end
    # static pose because the object is already resting against the stool.
    anchor_frames = [690, 700, 710, 720, 730, 740, 750, 770, 790]
    cap = cv2.VideoCapture(str(inp / "video.mp4"))
    frame_idx = -1
    selected = {}
    for target in anchor_frames:
        while frame_idx < target:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("video ended before all anchors were read")
            frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pose, score, roll = select_anchor_pose(
            renderer, baseline[target], rgb, object_masks[target], person_masks[target],
            mesh_center,
        )
        selected[target] = pose
        print(f"anchor {target}: roll={roll:4d} deg, mask score={score:.3f}", flush=True)
    cap.release()

    # Smoothstep interpolation between keyframes in SE(3), with the pre-carry
    # smoothed pose and the static end pose as boundary conditions.
    keys = {670: trajectory[670].copy(), **selected, END_STATIC_START: end_pose}
    key_frames = sorted(keys)
    for a, b in zip(key_frames[:-1], key_frames[1:]):
        for i in range(a, b + 1):
            u = (i - a) / (b - a)
            w = u * u * (3.0 - 2.0 * u)
            trajectory[i] = slerp_pose(keys[a], keys[b], w)

    # Gentle start transition: the object begins moving around frame 115-120.
    A, Bend = start_pose, trajectory[130].copy()
    qa = Rotation.from_matrix(A[:3, :3]).as_quat()
    qb = Rotation.from_matrix(Bend[:3, :3]).as_quat()
    qb *= np.sign(qa @ qb)
    for i in range(START_STATIC_END + 1, 130):
        u = (i - START_STATIC_END) / 20.0
        w = u * u * (3.0 - 2.0 * u)
        q = qa * (1.0 - w) + qb * w
        q /= np.linalg.norm(q)
        trajectory[i, :3, :3] = Rotation.from_quat(q).as_matrix()
        trajectory[i, :3, 3] = A[:3, 3] * (1.0 - w) + Bend[:3, 3] * w

    # Project rotations back to SO(3) to suppress numerical drift.
    trajectory[:, :3, :3] = Rotation.from_matrix(trajectory[:, :3, :3]).as_matrix()
    trajectory[:, 3, :] = [0.0, 0.0, 0.0, 1.0]
    np.savez(out, T_world_mesh=trajectory.astype(np.float64))
    print(f"wrote {out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
