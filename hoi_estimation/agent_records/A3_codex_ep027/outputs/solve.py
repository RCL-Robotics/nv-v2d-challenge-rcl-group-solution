#!/usr/bin/env python3
"""Conservative silhouette translation refinement; preserves rotations and camera range.

Run offline:
 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
No exploratory files, fitted parameters, network access, or downloaded assets are used.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyrender
import trimesh
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import minimize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    random.seed(0)
    np.random.seed(0)
    cv2.setNumThreads(1)
    root = args.inputs
    with np.load(root / 'reconstruction.npz') as data:
        original = data['T_world_mesh'].astype(np.float64).copy()
    with open(root / 'camera.json') as file:
        calibration = json.load(file)
    K = np.asarray(calibration['K'], dtype=np.float64)
    world_from_cam = np.asarray(calibration['T_world_cam'], dtype=np.float64)
    cam_poses = np.linalg.inv(world_from_cam) @ original
    mesh = trimesh.load(root / 'object_mesh.glb', force='scene', process=False).to_geometry()
    # The centre of the mesh-file axis-aligned bounds is explicit and reproducible.
    # All corrections preserve its Euclidean distance from the camera, not just Z.
    centre_mesh = mesh.bounds.mean(axis=0)
    scene = pyrender.Scene()
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    size = 256
    camera = pyrender.IntrinsicsCamera(K[0, 0], K[1, 1], size/2, size/2,
                                       znear=.01, zfar=20.)
    scene.add(camera, pose=np.diag([1., -1., -1., 1.]))
    renderer = pyrender.OffscreenRenderer(size, size)
    count = len(original)
    metrics = np.zeros((count, 7), dtype=np.float64)
    crop_origins = np.zeros((count, 2), dtype=int)
    prediction = np.zeros((count, size, size), dtype=np.uint8)
    observation = np.zeros_like(prediction)
    valid_pixels = np.zeros_like(prediction)
    signed_distance = np.zeros((count, size, size), dtype=np.float32)

    def render(frame, pose):
        x0, y0 = crop_origins[frame]
        camera.cx = K[0, 2] - x0
        camera.cy = K[1, 2] - y0
        scene.set_pose(mesh_node, pose)
        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        return (depth > 0).astype(np.uint8)

    def cost(frame, pred):
        return float(np.sum(pred * signed_distance[frame] * valid_pixels[frame]) /
                     max(int(observation[frame].sum()), 1))

    def iou(a, b):
        return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))

    try:
        with h5py.File(root / 'masks.h5', 'r') as masks:
            for frame, pose in enumerate(cam_poses):
                c = pose[:3, :3] @ centre_mesh + pose[:3, 3]
                pixel = K @ c
                crop_origins[frame] = np.rint(pixel[:2] / pixel[2]).astype(int) - size//2
                x0, y0 = crop_origins[frame]
                if not (0 <= x0 <= calibration['width']-size and
                        0 <= y0 <= calibration['height']-size):
                    raise ValueError('Basket crop falls outside the image.')
                pred = render(frame, pose)
                obs = masks['object'][frame, y0:y0+size, x0:x0+size].astype(np.uint8)
                person = masks['person'][frame, y0:y0+size, x0:x0+size].astype(np.uint8)
                # Hands/body hide the basket: those pixels provide no negative evidence.
                valid = 1 - cv2.dilate(person, np.ones((5, 5), dtype=np.uint8))
                valid[obs > 0] = 1
                distance = (cv2.distanceTransform(1-obs, cv2.DIST_L2, 5) -
                            cv2.distanceTransform(obs, cv2.DIST_L2, 5))
                prediction[frame], observation[frame], valid_pixels[frame] = pred, obs, valid
                # Cap distances at five pixels so isolated segmentation errors cannot
                # dominate. The dimensions/scale/orientation of the mesh stay fixed.
                signed_distance[frame] = np.clip(distance, -5., 5.)
                def shift_cost(shift):
                    affine = np.array([[1., 0., shift[0]], [0., 1., shift[1]]], dtype=np.float32)
                    translated = cv2.warpAffine(pred.astype(np.float32), affine, (size, size),
                                                flags=cv2.INTER_LINEAR)
                    return cost(frame, translated)
                result = minimize(shift_cost, [0., 0.], method='Powell',
                                  bounds=[(-8., 8.), (-8., 8.)],
                                  options={'xtol': .05, 'ftol': .0001})
                metrics[frame] = [*result.x, shift_cost([0., 0.])-result.fun,
                                  (pred*valid).sum()/max(int(pred.sum()), 1),
                                  iou(pred, obs), obs.sum(), pred.sum()]
                if frame % 100 == 0:
                    print(f'Measured frame {frame}/{count}', flush=True)

        # Thresholds deliberately exclude heavy occlusion, poor shape agreement,
        # large displacements and weak evidence. Inspection showed a stable ~4 px
        # vertical error in the early static segment, whereas carrying frames can
        # have ambiguous masked contours. No frame-index-specific edits are used.
        eligible = ((metrics[:, 3] >= .85) & (metrics[:, 4] >= .65) &
                    (metrics[:, 2] >= .12) &
                    (np.max(np.abs(metrics[:, :2]), axis=1) < 6.))
        med = median_filter(metrics[:, :2], size=(9, 1), mode='nearest')
        coherent = np.linalg.norm(metrics[:, :2]-med, axis=1) <= 1.5
        supported = np.convolve(np.pad((eligible & coherent).astype(float),
                                      (4, 4), mode='edge'), np.ones(9), mode='valid') >= 5
        accepted = eligible & coherent & supported
        # Temporal smoothing and 15% shrinkage reduce sensitivity to rasterization
        # and small SAM boundary errors. Fade at the edges of reliable intervals.
        strength = .85 * np.minimum(accepted.astype(float),
                                    gaussian_filter1d(accepted.astype(float), 1.5, mode='nearest'))
        shifts = gaussian_filter1d(med, 1., axis=0, mode='nearest') * strength[:, None]
        output = original.copy()
        final_iou = metrics[:, 4].copy()
        improvements = np.zeros(count)
        range_errors = np.zeros(count)
        for frame in np.flatnonzero(strength > 0):
            pose = cam_poses[frame].copy()
            centre = pose[:3, :3] @ centre_mesh + pose[:3, 3]
            pixel = K @ centre
            uv = pixel[:2] / pixel[2] + shifts[frame]
            ray = np.array([(uv[0]-K[0, 2])/K[0, 0],
                            (uv[1]-K[1, 2])/K[1, 1], 1.])
            centre_new = ray * (np.linalg.norm(centre) / np.linalg.norm(ray))
            pose[:3, 3] += centre_new - centre
            # Verify the actual perspective render, rather than just the shifted
            # raster approximation, before accepting each translation.
            pred_new = render(frame, pose)
            gain = cost(frame, prediction[frame]) - cost(frame, pred_new)
            new_iou = iou(pred_new, observation[frame])
            if gain < .08 or new_iou - metrics[frame, 4] < .025:
                shifts[frame] = 0.
                continue
            # Assign translation only: baseline rotation and homogeneous row are
            # copied bit-for-bit, avoiding numerical changes on unedited frames.
            output[frame, :3, 3] = (world_from_cam @ pose)[:3, 3]
            final_iou[frame] = new_iou
            improvements[frame] = gain
            range_errors[frame] = abs(np.linalg.norm(centre_new)-np.linalg.norm(centre))
    finally:
        renderer.delete()
    changed = np.any(output != original, axis=(1, 2))
    assert output.shape == (count, 4, 4) and output.dtype == np.float64
    assert np.isfinite(output).all()
    assert np.array_equal(output[:, :3, :3], original[:, :3, :3])
    assert np.array_equal(output[~changed], original[~changed])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, T_world_mesh=output)
    diagnostics = {
        'frames': count, 'changed_frames': int(changed.sum()),
        'changed_indices': np.flatnonzero(changed).tolist(),
        'elapsed_seconds': time.perf_counter()-start,
        'mean_baseline_iou_changed': float(metrics[changed, 4].mean()) if changed.any() else None,
        'mean_output_iou_changed': float(final_iou[changed].mean()) if changed.any() else None,
        'median_shift_pixels_changed': np.median(shifts[changed], axis=0).tolist() if changed.any() else [],
        'max_translation_change_m': float(np.linalg.norm(output[:, :3, 3]-original[:, :3, 3], axis=1).max()),
        'max_camera_range_error_m': float(range_errors.max()),
        'rotations_unchanged': bool(np.array_equal(output[:, :3, :3], original[:, :3, :3])),
    }
    with open(args.out.with_suffix('.diagnostics.json'), 'w') as file:
        json.dump(diagnostics, file, indent=2)
    np.savez_compressed(args.out.with_suffix('.alignment.npz'), measurements=metrics,
                        applied_shifts_pixels=shifts, changed=changed,
                        baseline_iou=metrics[:, 4], final_iou=final_iou,
                        signed_distance_gain=improvements)
    print(json.dumps({k: v for k, v in diagnostics.items() if k != 'changed_indices'}, indent=2))


if __name__ == '__main__':
    main()
