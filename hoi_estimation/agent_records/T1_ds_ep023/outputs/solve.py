#!/usr/bin/env python3
"""Trajectory recovery for the monocular toy-stroller episode.

The supplied CARI4D/FoundationPose/CoCoNet reconstruction is used as a prior.
A rigid-object trajectory prior is then fitted:
  1. pooled constant poses in the two visually verified rest phases,
  2. robust SE(3) temporal smoothing during manipulation,
  3. camera-plane corrections against the SAM object silhouette,
  4. smooth acceleration/deceleration at the two rest/manipulation boundaries.

No ground-truth annotation is read.  All evidence comes from the supplied video,
masks, mesh, camera, and baseline reconstruction.
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# Verified phase boundaries (30 fps).  The object's silhouette centroid and its
# texture remain fixed through these frames.  Texture scores start changing at
# 132 and the released object has settled by 901.
INITIAL_STATIC_END = 132
FINAL_STATIC_START = 901


def load_geometry(path):
    scene = trimesh.load(path, force="scene", process=False)
    return list(scene.geometry.values())[0]


def pose_mean(transforms):
    """Robust translation median and chordal-L2 rotation mean."""
    t = np.median(transforms[:, :3, 3], axis=0)
    rm = Rotation.from_matrix(transforms[:, :3, :3]).mean().as_matrix()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rm
    out[:3, 3] = t
    return out


def smooth_trajectory(transforms, static_runs):
    """Robust translational and quaternion smoothing, then exact rest phases."""
    transforms = transforms.astype(np.float64, copy=True)
    f = len(transforms)

    # Median-of-five rejects isolated tracker jumps before a local cubic fit.
    trans = median_filter(transforms[:, :3, 3], size=(5, 1), mode="nearest")
    for k in range(3):
        transforms[:, k, 3] = savgol_filter(
            trans[:, k], window_length=11, polyorder=3, mode="interp"
        )

    # Smooth quaternions in a continuous sign branch.  This handles the
    # physical >180 degree yaw sweep without a quaternion sign discontinuity.
    quat = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    for k in range(1, f):
        if np.dot(quat[k], quat[k - 1]) < 0.0:
            quat[k] *= -1.0
    quat = median_filter(quat, size=(5, 1), mode="nearest")
    sigma = 2.5
    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    for k in range(4):
        quat[:, k] = np.convolve(
            np.pad(quat[:, k], (radius, radius), mode="edge"), kernel, mode="valid"
        )
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    transforms[:, :3, :3] = Rotation.from_quat(quat).as_matrix()

    # A rigid object at rest has exactly one world pose.  Pooling all frames in
    # each phase removes the approximately 1 cm / 0.5 degree tracker jitter.
    for lo, hi in static_runs:
        transforms[lo:hi] = pose_mean(transforms[lo:hi])
    return transforms


class MaskRenderer:
    def __init__(self, mesh, width, height):
        self.mesh = mesh
        self.width = int(width)
        self.height = int(height)
        self.ctx = dr.RasterizeCudaContext()
        self.tri = torch.tensor(mesh.faces.astype(np.int32), device="cuda")
        self.vertices = mesh.vertices.astype(np.float32)

    def render(self, transform, world_to_camera, intrinsics):
        vh = np.concatenate(
            [self.vertices, np.ones((len(self.vertices), 1), np.float32)], axis=1
        )
        cam = (world_to_camera @ (transform @ vh.T)).T
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        px = fx * x / z + cx
        py = fy * y / z + cy
        xn = 2.0 * px / self.width - 1.0
        # nvdiffrast's image row 0 convention is opposite to the OpenCV image
        # row convention used by the supplied masks in this environment.
        yn = 2.0 * py / self.height - 1.0
        clip = np.stack([xn * z, yn * z, z, z], axis=1).astype(np.float32)
        pos = torch.from_numpy(clip).cuda()
        rast, _ = dr.rasterize(
            self.ctx, pos[None], self.tri, resolution=(self.height, self.width)
        )
        return (rast[0, :, :, 3] > 0).cpu().numpy()


def silhouette_centroid_corrections(
    baseline, masks_path, renderer, world_to_camera, intrinsics
):
    """Estimate camera-plane image shifts that register the rendered object."""
    f = len(baseline)
    raw_dx = np.zeros(f, dtype=np.float64)
    raw_dy = np.zeros(f, dtype=np.float64)
    pred_area = np.zeros(f, dtype=np.float64)
    obj_area = np.zeros(f, dtype=np.float64)
    iou = np.zeros(f, dtype=np.float64)

    with h5py.File(masks_path, "r") as masks:
        for i in range(f):
            obj = masks["object"][i]
            person = masks["person"][i]
            pred = renderer.render(baseline[i], world_to_camera, intrinsics)
            visible = pred & ~person
            pa = int(visible.sum())
            oa = int(obj.sum())
            pred_area[i] = pa
            obj_area[i] = oa
            if pa:
                yy_p, xx_p = np.where(visible)
                yy_o, xx_o = np.where(obj)
                raw_dx[i] = xx_o.mean() - xx_p.mean()
                raw_dy[i] = yy_o.mean() - yy_p.mean()
                inter = int((visible & obj).sum())
                iou[i] = inter / max(pa + oa - inter, 1)
            if i % 100 == 0:
                print(f"  silhouette {i:4d}/{f}", flush=True)

    valid = (pred_area > 900) & (obj_area > 1800) & (iou > 0.42)
    dx = raw_dx.copy()
    dy = raw_dy.copy()
    for value in (dx, dy):
        local = median_filter(value, size=9, mode="nearest")
        bad = np.abs(value - local) > 4.0
        value[bad] = local[bad]

    for lo, hi in ((0, INITIAL_STATIC_END), (FINAL_STATIC_START, f)):
        dx[lo:hi] = np.median(dx[lo:hi][valid[lo:hi]])
        dy[lo:hi] = np.median(dy[lo:hi][valid[lo:hi]])

    good = np.where(valid)[0]
    for value in (dx, dy):
        value[:] = np.interp(np.arange(f), good, value[good])
        value[:] = savgol_filter(value, window_length=11, polyorder=2, mode="interp")
    return np.clip(dx, -12.0, 12.0), np.clip(dy, -12.0, 12.0)


def translate_camera_plane(transform, dx_px, dy_px, world_from_camera, world_to_camera, intrinsics):
    z = (world_to_camera @ transform)[2, 3]
    local = np.array(
        [dx_px * z / intrinsics[0, 0], dy_px * z / intrinsics[1, 1], 0.0],
        dtype=np.float64,
    )
    out = transform.copy()
    out[:3, 3] += world_from_camera[:3, :3] @ local
    return out


def blend_pose(p, q, alpha):
    out = q.copy()
    out[:3, 3] = (1.0 - alpha) * p[:3, 3] + alpha * q[:3, 3]
    rel = Rotation.from_matrix(p[:3, :3]).inv() * Rotation.from_matrix(q[:3, :3])
    out[:3, :3] = (
        p[:3, :3]
        @ Rotation.from_rotvec(rel.as_rotvec() * alpha).as_matrix()
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    started = time.time()

    inputs = os.path.abspath(args.inputs)
    recon = np.load(os.path.join(inputs, "reconstruction.npz"))
    baseline = recon["T_world_mesh"].astype(np.float64)
    f = len(baseline)
    with open(os.path.join(inputs, "camera.json")) as handle:
        camera = json.load(handle)
    with open(os.path.join(inputs, "episode.json")) as handle:
        episode = json.load(handle)
    if int(episode["frames"]) != f:
        raise RuntimeError(f"Expected {episode['frames']} frames, found {f}")
    intrinsics = np.asarray(camera["K"], dtype=np.float32)
    world_from_camera = np.asarray(camera["T_world_cam"], dtype=np.float64)
    world_to_camera = np.linalg.inv(world_from_camera)
    width, height = int(camera["width"]), int(camera["height"])

    print("[1/5] Robust temporal prior")
    static_runs = ((0, INITIAL_STATIC_END), (FINAL_STATIC_START, f))
    smooth = smooth_trajectory(baseline, static_runs)

    print("[2/5] Silhouette registration against SAM masks")
    mesh = load_geometry(os.path.join(inputs, "object_mesh.glb"))
    renderer = MaskRenderer(mesh, width, height)
    dx, dy = silhouette_centroid_corrections(
        smooth,
        os.path.join(inputs, "masks.h5"),
        renderer,
        world_to_camera,
        intrinsics,
    )

    print("[3/5] Apply camera-plane registration")
    result = smooth.copy()
    for i in range(f):
        result[i] = translate_camera_plane(
            smooth[i], dx[i], dy[i], world_from_camera, world_to_camera, intrinsics
        )

    # The SAM silhouette extends slightly farther below than above the mesh.
    # Texture-gradient calibration over the moving phase favored 1.5 px less
    # downward centroid correction.  At the final, rear-facing view the same
    # calibration favored removing the 6 px residual centroid-y bias.
    for i in range(INITIAL_STATIC_END, FINAL_STATIC_START):
        z = (world_to_camera @ result[i])[2, 3]
        result[i, :3, 3] += world_from_camera[:3, :3] @ np.array(
            [0.0, -1.5 * z / intrinsics[1, 1], 0.0]
        )
    z = (world_to_camera @ result[FINAL_STATIC_START])[2, 3]
    result[FINAL_STATIC_START:, :3, 3] += world_from_camera[:3, :3] @ np.array(
        [0.0, -6.0 * z / intrinsics[1, 1], 0.0]
    )

    # Re-pool the rest phases after image registration so they remain exactly
    # constant, then ease in/out instead of exposing tracker jumps at the phase
    # changes.  The object begins accelerating over about 0.45 s and settles
    # over about 0.35 s.
    p0 = pose_mean(result[:INITIAL_STATIC_END])
    p1 = pose_mean(result[FINAL_STATIC_START:])
    # Two final sub-pixel/2-px refinements from the pooled texture correlation
    # grid.  These use all frames in each rest phase and improve silhouette,
    # gradient and colour agreement together (see REPORT.md).
    p0 = translate_camera_plane(
        p0, -2.0, 0.0, world_from_camera, world_to_camera, intrinsics
    )
    p1 = translate_camera_plane(
        p1, 2.0, -2.0, world_from_camera, world_to_camera, intrinsics
    )
    result[:INITIAL_STATIC_END] = p0
    result[FINAL_STATIC_START:] = p1
    for i in range(INITIAL_STATIC_END, INITIAL_STATIC_END + 14):
        a = (i - INITIAL_STATIC_END + 1) / 14.0
        a = a * a * (3.0 - 2.0 * a)
        result[i] = blend_pose(p0, result[INITIAL_STATIC_END + 13], a)
    for i in range(890, FINAL_STATIC_START):
        a = (i - 889) / float(FINAL_STATIC_START - 890 + 1)
        a = a * a * (3.0 - 2.0 * a)
        result[i] = blend_pose(result[i], p1, a)

    print("[4/5] Validate output")
    finite = np.isfinite(result).all()
    ortho = np.abs(
        np.einsum("nji,njk->nik", result[:, :3, :3], result[:, :3, :3])
        - np.eye(3)[None]
    ).max()
    det = np.linalg.det(result[:, :3, :3])
    if not finite or ortho > 1e-8 or np.abs(det - 1.0).max() > 1e-8:
        raise RuntimeError(
            f"Invalid rotations: finite={finite}, ortho={ortho}, det={det.min()}..{det.max()}"
        )

    result = result.astype(np.float64)
    np.savez(args.out, T_world_mesh=result)
    print("[5/5] Wrote", args.out, "shape", result.shape, "dtype", result.dtype)
    print(f"Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
