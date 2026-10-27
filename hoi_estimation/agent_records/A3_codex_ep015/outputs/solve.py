#!/usr/bin/env python3
"""Conservative texture-based pose refinement at the released camera distance.

Only supplied data are used. Occluded/featureless/ambiguous frames retain their
baseline poses. Run: python solve.py --inputs /work/inputs --out ...npz
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# These conservative constants deliberately restrict updates to the basin of the
# released pose. The yellow sign has similar faces; no symmetry flip is inferred.
RESOLUTION = 672
MAX_ROTATION = np.deg2rad(8)
MAX_CENTER_MOVE = .025
ROTATION_UNIT = .08
TRANSLATION_UNIT = .015


def pose_from_parameters(baseline, center, parameters):
    """Rotate about the OBB center and keep its original camera range exactly."""
    rotation = Rotation.from_rotvec(parameters[:3] * ROTATION_UNIT).as_matrix() @ baseline[:3, :3]
    old_center = baseline[:3, :3] @ center + baseline[:3, 3]
    new_center = old_center + np.r_[parameters[3:] * TRANSLATION_UNIT, 0.]
    new_center *= np.linalg.norm(old_center) / np.linalg.norm(new_center)
    pose = baseline.copy()
    pose[:3, :3] = rotation
    pose[:3, 3] = new_center - rotation @ center
    return pose


def project(points, pose, intrinsics):
    points_camera = points @ pose[:3, :3].T + pose[:3, 3]
    pixels = points_camera @ intrinsics.T
    return pixels[:, :2] / pixels[:, 2:]


class Renderer:
    """Unlit supplied texture, plus mesh coordinates at each rasterized pixel."""
    def __init__(self, mesh):
        self.device = 'cuda'
        self.context = dr.RasterizeCudaContext()
        self.vertices = torch.tensor(np.asarray(mesh.vertices), device=self.device, dtype=torch.float32)
        self.faces = torch.tensor(np.asarray(mesh.faces), device=self.device, dtype=torch.int32)
        self.uv = torch.tensor(np.asarray(mesh.visual.uv), device=self.device, dtype=torch.float32)
        texture = np.asarray(mesh.visual.material.baseColorTexture.convert('RGB')).astype(np.float32) / 255
        texture = cv2.resize(texture, (2048, 2048), interpolation=cv2.INTER_AREA)
        self.texture = torch.tensor(texture[::-1].copy(), device=self.device)[None]

    @torch.no_grad()
    def render(self, pose, intrinsics):
        size = RESOLUTION
        rotation = torch.tensor(pose[:3, :3], device=self.device, dtype=torch.float32)
        translation = torch.tensor(pose[:3, 3], device=self.device, dtype=torch.float32)
        camera_points = self.vertices @ rotation.T + translation
        x, y, z = camera_points.unbind(-1)
        clip = torch.stack([
            (2 * intrinsics[0, 0] * x + (2 * intrinsics[0, 2] - size) * z) / size,
            (-2 * intrinsics[1, 1] * y + (size - 2 * intrinsics[1, 2]) * z) / size,
            1.002 * z - .2002, z], -1)[None]
        raster, _ = dr.rasterize(self.context, clip, self.faces, resolution=[size, size])
        uv, _ = dr.interpolate(self.uv[None], raster, self.faces)
        rgb = dr.texture(self.texture, uv, filter_mode='linear') * (raster[..., 3:] > 0)
        xyz, _ = dr.interpolate(self.vertices[None], raster, self.faces)
        return (rgb[0].flip(0).cpu().numpy() * 255).astype(np.uint8), xyz[0].flip(0).cpu().numpy()


def extract_correspondences(image, object_mask, baseline, mesh, camera_matrix, renderer, sift):
    # Include the entire baseline projection even when only a sliver is visible.
    corners = trimesh.bounds.corners(mesh.bounds)
    pixels = project(corners, baseline, camera_matrix)
    ys, xs = np.where(object_mask)
    if len(xs) < 100:
        return None
    xy = np.r_[pixels, np.c_[xs, ys]]
    low, high = xy.min(0), xy.max(0)
    side = int(np.ceil(np.max(high - low) + 50))
    x0, y0 = np.floor((low + high - side) / 2).astype(int)
    height, width = image.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + side > width or y0 + side > height:
        return None
    scale = RESOLUTION / side
    intrinsics = camera_matrix.copy()
    intrinsics[:2, 2] -= [x0, y0]
    intrinsics[:2] *= scale
    real = cv2.resize(image[y0:y0 + side, x0:x0 + side], (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(object_mask[y0:y0 + side, x0:x0 + side].astype(np.uint8), (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_NEAREST)
    # SAM sometimes fills the handle hole. Masks only select observed features;
    # neither silhouette scale nor a filled hole determines the fitted rotation.
    feature_mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) * 255
    rendered, xyz = renderer.render(baseline, intrinsics)
    rendered = cv2.GaussianBlur(rendered, (0, 0), 1.3)
    rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
    key_render, descriptor_render = sift.detectAndCompute(rendered_gray, None)
    key_real, descriptor_real = sift.detectAndCompute(real_gray, feature_mask)
    if descriptor_render is None or descriptor_real is None or len(descriptor_real) < 2:
        return None
    matches = cv2.BFMatcher().knnMatch(descriptor_render, descriptor_real, k=2)
    points, observations = [], []
    for pair in matches:
        if len(pair) != 2:
            continue
        match, second = pair
        if match.distance >= .8 * second.distance:
            continue
        source = np.array(key_render[match.queryIdx].pt)
        target = np.array(key_real[match.trainIdx].pt)
        if np.linalg.norm(source - target) / scale > 18:
            continue
        ix, iy = np.rint(source).astype(int)
        if not (0 <= ix < RESOLUTION and 0 <= iy < RESOLUTION):
            continue
        point = xyz[iy, ix].astype(float)
        # A SIFT keypoint can lie in background even if its descriptor sees the
        # sign. Discard it; an unrendered pixel has no valid 3-D correspondence.
        if np.linalg.norm(project(point[None], baseline, intrinsics)[0] - source) > 1.5:
            continue
        observation = (target / scale) + [x0, y0]
        if np.linalg.norm(project(point[None], baseline, camera_matrix)[0] - observation) > 10:
            continue
        # SIFT creates several orientations at one feature. Count it only once.
        if any(np.linalg.norm(observation - previous) < 1.5 for previous in observations):
            continue
        points.append(point)
        observations.append(observation)
    if len(points) < 10:
        return None
    points = np.asarray(points)
    observations = np.asarray(observations)
    # A robust image-space check removes repeated-letter / triangle mismatches.
    source = project(points, baseline, camera_matrix)
    _, inliers = cv2.estimateAffinePartial2D(source, observations, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=1000, confidence=.995, refineIters=10)
    if inliers is None:
        return None
    inliers = inliers.ravel().astype(bool)
    points, observations = points[inliers], observations[inliers]
    if len(points) < 10:
        return None
    hull_area = cv2.contourArea(cv2.convexHull(observations.astype(np.float32)))
    if hull_area < 70 or np.linalg.norm(np.ptp(observations, axis=0)) < 25:
        return None
    return points, observations


def fit_parameters(points, observations, baseline, center, intrinsics, indices=None):
    if indices is not None:
        points, observations = points[indices], observations[indices]

    def residual(parameters):
        pose = pose_from_parameters(baseline, center, parameters)
        errors = project(points, pose, intrinsics) - observations
        # Priors suppress poorly constrained tilt and small-object focal errors.
        return np.r_[errors.ravel(), .8 * parameters]

    result = least_squares(residual, np.zeros(5), loss='soft_l1', f_scale=1., max_nfev=60, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    return result.x


def pixel_errors(points, observations, pose, intrinsics):
    return np.linalg.norm(project(points, pose, intrinsics) - observations, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    data = np.load(args.inputs / 'reconstruction.npz')
    baseline_world = data['T_world_mesh'].astype(np.float64)
    camera = json.loads((args.inputs / 'camera.json').read_text())
    camera_matrix = np.array(camera['K'], dtype=np.float64)
    world_from_camera = np.array(camera['T_world_cam'], dtype=np.float64)
    baseline_camera = np.linalg.inv(world_from_camera)[None] @ baseline_world
    count = len(baseline_world)
    mesh = trimesh.load(args.inputs / 'object_mesh.glb', force='scene', process=False).to_geometry()
    mesh.merge_vertices()
    center_transform, _ = trimesh.bounds.oriented_bounds(mesh)
    center = np.linalg.inv(center_transform)[:3, 3]
    renderer = Renderer(mesh)
    sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=.015, edgeThreshold=12)
    parameters = np.zeros((count, 5))
    raw_accept = np.zeros(count, dtype=bool)
    correspondences = [None] * count
    records = []
    capture = cv2.VideoCapture(str(args.inputs / 'video.mp4'))
    with h5py.File(args.inputs / 'masks.h5', 'r') as masks:
        for frame in range(count):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'Missing video frame {frame}')
            baseline = baseline_camera[frame]
            correspondence = extract_correspondences(image, masks['object'][frame], baseline, mesh, camera_matrix, renderer, sift)
            record = {'frame': frame, 'features': 0, 'raw_accepted': False}
            if correspondence is not None:
                points, observations = correspondence
                correspondences[frame] = correspondence
                original_error = pixel_errors(points, observations, baseline, camera_matrix)
                fitted = fit_parameters(points, observations, baseline, center, camera_matrix)
                candidate = pose_from_parameters(baseline, center, fitted)
                new_error = pixel_errors(points, observations, candidate, camera_matrix)
                angle = np.linalg.norm(fitted[:3] * ROTATION_UNIT)
                old_center = baseline[:3, :3] @ center + baseline[:3, 3]
                new_center = candidate[:3, :3] @ center + candidate[:3, 3]
                center_move = np.linalg.norm(new_center - old_center)
                # Every third feature is held out in turn. These matches are
                # measured on the real printed pattern, never synthesized labels.
                order = np.lexsort((observations[:, 0], observations[:, 1]))
                improvements = []
                fold_parameters = []
                for fold in range(3):
                    heldout = order[np.arange(len(order)) % 3 == fold]
                    training = order[np.arange(len(order)) % 3 != fold]
                    fitted_fold = fit_parameters(points, observations, baseline, center, camera_matrix, training)
                    fold_pose = pose_from_parameters(baseline, center, fitted_fold)
                    fold_error = pixel_errors(points[heldout], observations[heldout], fold_pose, camera_matrix)
                    improvements.append(float(np.median(original_error[heldout]) - np.median(fold_error)))
                    fold_parameters.append(fitted_fold)
                fold_parameters = np.array(fold_parameters)
                consistency = np.max(np.linalg.norm((fold_parameters[:, :3] - fitted[:3]) * ROTATION_UNIT, axis=1))
                accepted = (
                    len(points) >= 10 and angle <= MAX_ROTATION and center_move <= MAX_CENTER_MOVE
                    and np.median(new_error) < .8 * np.median(original_error)
                    and np.median(original_error) - np.median(new_error) > .35
                    and sum(value > .15 for value in improvements) >= 2
                    and np.mean(improvements) > .3
                    and consistency < np.deg2rad(4)
                )
                parameters[frame] = fitted
                raw_accept[frame] = accepted
                record.update(features=len(points), baseline_median_px=float(np.median(original_error)), fitted_median_px=float(np.median(new_error)), correction_degrees=float(np.rad2deg(angle)), center_move_m=float(center_move), heldout_improvement_px=improvements, raw_accepted=bool(accepted))
            records.append(record)
            if frame % 50 == 0:
                print(f'{frame}/{count}: {raw_accept[:frame+1].sum()} supported fits, {time.perf_counter()-started:.1f}s', flush=True)
    capture.release()

    # Smooth only the small corrections, not the calibrated trajectory. Require
    # actual feature evidence in each output frame; do not propagate a guess
    # across an occlusion. A short taper avoids isolated one-frame edits.
    weights = raw_accept.astype(float)
    density = gaussian_filter1d(weights, sigma=2., mode='nearest')
    smooth_parameters = gaussian_filter1d(parameters * weights[:, None], sigma=2., axis=0, mode='nearest') / np.maximum(density[:, None], 1e-12)
    result = baseline_world.copy()
    changed = []
    validations = []
    for frame in range(count):
        correspondence = correspondences[frame]
        if correspondence is None or not raw_accept[frame] or density[frame] < .55:
            continue
        points, observations = correspondence
        baseline = baseline_camera[frame]
        # Use a partial correction when neighboring frames give weaker support.
        strength = min(1., max(0., (density[frame] - .4) / .4))
        proposed_parameters = smooth_parameters[frame] * strength
        candidate = pose_from_parameters(baseline, center, proposed_parameters)
        errors_before = pixel_errors(points, observations, baseline, camera_matrix)
        errors_after = pixel_errors(points, observations, candidate, camera_matrix)
        if np.median(errors_after) > .85 * np.median(errors_before) or np.median(errors_before) - np.median(errors_after) < .3:
            continue
        if np.quantile(errors_after, .8) > np.quantile(errors_before, .8) + .1:
            continue
        old_center = baseline[:3, :3] @ center + baseline[:3, 3]
        new_center = candidate[:3, :3] @ center + candidate[:3, 3]
        if np.linalg.norm(proposed_parameters[:3] * ROTATION_UNIT) > MAX_ROTATION or np.linalg.norm(new_center - old_center) > MAX_CENTER_MOVE:
            continue
        result[frame] = world_from_camera @ candidate
        changed.append(frame)
        validations.append([float(np.median(errors_before)), float(np.median(errors_after))])
        records[frame].update(output_changed=True, output_median_px=float(np.median(errors_after)), output_correction_degrees=float(np.rad2deg(np.linalg.norm(proposed_parameters[:3] * ROTATION_UNIT))), strength=float(strength))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert result.dtype == np.float64 and result.shape == (count, 4, 4)
    assert np.isfinite(result).all()
    assert np.allclose(result[:, 3], [0, 0, 0, 1])
    # Preserve untouched frames bit-for-bit after the required float64 conversion.
    unchanged = np.ones(count, dtype=bool)
    unchanged[changed] = False
    assert np.array_equal(result[unchanged], baseline_world[unchanged])
    np.savez_compressed(args.out, T_world_mesh=result)
    result_camera = np.linalg.inv(world_from_camera)[None] @ result
    old_centers = baseline_camera[:, :3, :3] @ center + baseline_camera[:, :3, 3]
    new_centers = result_camera[:, :3, :3] @ center + result_camera[:, :3, 3]
    range_error = np.max(np.abs(np.linalg.norm(old_centers, axis=1) - np.linalg.norm(new_centers, axis=1)))
    rotation_error = np.max(np.abs(result[:, :3, :3].transpose(0, 2, 1) @ result[:, :3, :3] - np.eye(3)))
    summary = {
        'frames': count, 'changed_frames': changed, 'changed_count': len(changed),
        'unchanged_count': int(unchanged.sum()), 'raw_supported_count': int(raw_accept.sum()),
        'elapsed_seconds': time.perf_counter() - started,
        'max_camera_center_range_change_m': float(range_error),
        'max_rotation_orthogonality_error': float(rotation_error),
        'center_mesh_coordinates': center.tolist(),
        'mean_frame_median_error_before_after_px': np.mean(validations, axis=0).tolist() if validations else [],
        'records': records,
    }
    (args.out.parent / 'refinement_diagnostics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ['records', 'changed_frames']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
