#!/usr/bin/env python3
"""Conservative fixed-rotation translation refinement from masks and table support.

Only episode-specific physical choice: this rice cooker stays on the round table
while being turned (video inspection, corroborated by stable mask bottom). The
support plane is parallel to the released floor, but its height is fitted freely.
No image-derived per-frame estimates or fitted correction tables are embedded.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
import time
from pathlib import Path
import cv2
import h5py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    start = time.perf_counter()
    np.random.seed(0)
    cv2.setNumThreads(1)
    root = args.inputs
    rec = np.load(root / 'reconstruction.npz')
    baseline = np.asarray(rec['T_world_mesh'], dtype=np.float64)
    camera = json.loads((root / 'camera.json').read_text())
    K = np.array(camera['K'], dtype=np.float64)
    W = np.array(camera['T_world_cam'], dtype=np.float64)
    cam = np.linalg.inv(W)[None] @ baseline
    F = len(baseline)
    mesh = trimesh.load(root / 'object_mesh.glb', force='scene', process=False).to_geometry()
    # Projected support of the convex hull equals support of the full exact mesh.
    vertices = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
    rotated = np.einsum('fij,vj->fvi', cam[:, :3, :3], vertices)
    normal = np.array(json.loads((root / 'ground_plane.json').read_text())['plane'][:3])
    normal /= np.linalg.norm(normal)
    nc = W[:3, :3].T @ normal
    world_rotated = np.einsum('fij,vj->fvi', baseline[:, :3, :3], vertices)
    support_level = np.min(world_rotated @ normal, axis=1) + baseline[:, :3, 3] @ normal
    level0 = np.median(support_level)
    del world_rotated
    angles = np.arange(32) * (2 * np.pi / 32)
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    observations = np.zeros((F, 32))
    weights = np.zeros_like(observations)
    occlusions = np.zeros(F)
    mask_bounds = np.full((F, 4), np.nan)
    with h5py.File(root / 'masks.h5') as masks:
        for f in range(F):
            ys, xs = np.where(masks['object'][f])
            if len(xs) < 100:
                continue
            mask_bounds[f] = [xs.min(), xs.max(), ys.min(), ys.max()]
            points = np.column_stack([xs, ys])
            # Pixel masks represent square areas, not only pixel centres.
            observations[f] = (points @ directions.T).max(axis=0) + .5 * np.abs(directions).sum(axis=1)
            near_person = cv2.dilate(masks['person'][f].astype(np.uint8), np.ones((7, 7), np.uint8))
            occlusions[f] = near_person[ys, xs].mean()
            w = np.full(32, .10)
            w[[0, 8, 16, 24]] = 1.
            # A hand overlapping the handle removes the true upper silhouette.
            # Suppress those directions using measured person proximity.
            if occlusions[f] > .01:
                w[17:] *= .15
            # Supports are correlated: retain only four effective observations.
            weights[f] = w * (4. / w.sum())

    # Long untouched intervals are static only when image position, size AND
    # baseline orientation agree. No frame boundaries are hard-coded.
    untouched = np.max(rec['contact_prob_right_left'], axis=1) < .1
    transitions = np.diff(np.r_[False, untouched, False].astype(int))
    static_segments = []
    for first, end in zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0]):
        if end - first < 30 or not np.isfinite(mask_bounds[first:end]).all():
            continue
        bb = mask_bounds[first:end]
        centers = (bb[:, [0, 2]] + bb[:, [1, 3]]) / 2
        sizes = bb[:, [1, 3]] - bb[:, [0, 2]]
        relative = baseline[first:end, :3, :3] @ baseline[first, :3, :3].T
        angular = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1))
        if (np.max(np.ptp(centers, axis=0)) <= 2. and
                np.max(np.ptp(sizes, axis=0)) <= 3. and np.max(angular) < np.deg2rad(5)):
            static_segments.append((int(first), int(end)))
    nvars = F * 3 + 2 + 3 * len(static_segments)
    eta_index, table_index = F * 3, F * 3 + 1
    solution = np.zeros(nvars)
    base_t = cam[:, :3, 3]
    initial_rms = None
    records = []
    for iteration in range(10):
        delta = solution[:F * 3].reshape(F, 3)
        eta = solution[eta_index]
        rows, cols, vals, rhs = [], [], [], []
        row = 0
        all_errors = []
        def add(indices, values, target, sigma=1.):
            nonlocal row
            rows.extend([row] * len(indices))
            cols.extend(indices)
            vals.extend(np.asarray(values) / sigma)
            rhs.append(target / sigma)
            row += 1
        for f in range(F):
            pc = rotated[f] + base_t[f] + delta[f]
            uv0 = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]]
            uv = uv0 * (1 + eta) + K[:2, 2]
            index = np.argmax(uv @ directions.T, axis=0)
            selected = pc[index]
            pred = np.sum(uv[index] * directions, axis=1)
            error = pred - observations[f]
            all_errors.extend(error.tolist())
            for j, direction in enumerate(directions):
                if weights[f, j] == 0:
                    continue
                x, y, z = selected[j]
                dx, dy = direction
                jac = np.array([dx * K[0, 0] / z,
                                dy * K[1, 1] / z,
                                -(dx * K[0, 0] * x + dy * K[1, 1] * y) / (z * z)]) * (1 + eta)
                jf = np.dot(uv0[index[j]], direction)
                # Huber robustness to imperfect masks and locked rotation errors.
                robust = min(1., 1. / max(abs(error[j]), 1.e-8))
                sigma = 1. / np.sqrt(weights[f, j] * robust)
                target = -error[j] + jac @ delta[f] + jf * eta
                add([3*f, 3*f+1, 3*f+2, eta_index], [*jac, jf], target, sigma)
            # Baseline prior: a 1-pixel silhouette ambiguity is already several cm
            # in depth. Stronger lateral priors prevent unsupported sideways moves.
            for a, sigma in enumerate([.020, .020, .060]):
                add([3*f+a], [1.], 0., sigma)
            # Parallel table plane, unknown height. Eight mm tolerates mesh-bottom
            # and baseline tilt errors; floor offset itself is never used as table.
            add([3*f, 3*f+1, 3*f+2, table_index], [*nc, -1.],
                -(support_level[f] - level0), .008)
        # Gentle acceleration regularization, preserving sustained physical motion.
        for f in range(1, F-1):
            for a in range(3):
                add([3*(f-1)+a, 3*f+a, 3*(f+1)+a], [1., -2., 1.],
                    -(base_t[f-1, a] - 2*base_t[f, a] + base_t[f+1, a]), .003)
        for segment, (first, end) in enumerate(static_segments):
            center = base_t[first:end].mean(axis=0)
            for f in range(first, end):
                for a in range(3):
                    add([3*f+a, F*3+2+3*segment+a], [1., -1.],
                        -(base_t[f, a] - center[a]), .003)
        # A common focal ambiguity is explicitly allowed, so noisy monocular K
        # cannot by itself force a uniform metric-depth correction.
        add([eta_index], [1.], 0., .025)
        add([table_index], [1.], 0., .020)
        A = sp.coo_matrix((vals, (rows, cols)), shape=(row, nvars)).tocsr()
        new_solution = lsqr(A, np.asarray(rhs), atol=1.e-10, btol=1.e-10, iter_lim=2000)[0]
        # A deterministic trust region keeps this a conservative refinement.
        new_solution[eta_index] = np.clip(new_solution[eta_index], -.05, .05)
        change = np.max(np.abs(new_solution - solution))
        solution = new_solution
        rms = float(np.sqrt(np.mean(np.square(all_errors))))
        if initial_rms is None:
            initial_rms = rms
        records.append({'iteration': iteration, 'support_rms_px': rms, 'max_step': float(change)})
        if iteration >= 3 and change < 1.e-6:
            break
    delta = solution[:F * 3].reshape(F, 3)
    # Hard evidence gate: large departures require stronger evidence than these
    # uncertain masks and estimated intrinsics. Normal limit: 8 cm.
    lengths = np.linalg.norm(delta, axis=1)
    limits = np.full(F, .080)
    # Static segments have additional independent evidence: retain up to 12 cm
    # to correct tracker spikes without breaking their established fixed position.
    for first, end in static_segments:
        limits[first:end] = .120
    delta *= np.minimum(1., limits / np.maximum(lengths, 1.e-12))[:, None]
    result = baseline.copy()
    result[:, :3, 3] += delta @ W[:3, :3].T
    assert np.array_equal(result[:, :3, :3], baseline[:, :3, :3])
    assert result.shape == (F, 4, 4) and np.isfinite(result).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, T_world_mesh=result.astype(np.float64))
    final_cam = np.linalg.inv(W)[None] @ result
    def image_metrics(translations, focal_fraction):
        errors = []
        for f in range(F):
            pc = rotated[f] + translations[f]
            uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] * (1 + focal_fraction) + K[:2, 2]
            errors.append((uv @ directions.T).max(axis=0) - observations[f])
        errors = np.asarray(errors)
        return {'all_direction_rms_px': float(np.sqrt(np.mean(errors**2))),
                'weighted_rms_px': float(np.sqrt(np.sum(weights * errors**2) / weights.sum()))}
    static_metrics = []
    for first, end in static_segments:
        static_metrics.append({'first': first, 'end_exclusive': end,
            'baseline_depth_std_m': float(base_t[first:end, 2].std()),
            'refined_depth_std_m': float(final_cam[first:end, 2, 3].std()),
            'baseline_depth_mean_m': float(base_t[first:end, 2].mean()),
            'refined_depth_mean_m': float(final_cam[first:end, 2, 3].mean())})
    metrics = {
        'baseline_image_nominal_K': image_metrics(base_t, 0.),
        'refined_image_nominal_K': image_metrics(final_cam[:, :3, 3], 0.),
        'refined_image_nuisance_K': image_metrics(final_cam[:, :3, 3], float(solution[eta_index])),
        'static_interval_metrics': static_metrics,
        'baseline_depth_step_rms_m': float(np.sqrt(np.mean(np.diff(base_t[:, 2])**2))),
        'refined_depth_step_rms_m': float(np.sqrt(np.mean(np.diff(final_cam[:, 2, 3])**2))),
        'frames': F, 'runtime_seconds': time.perf_counter() - start,
        'static_segments_start_end_exclusive': static_segments,
        'rotation_exactly_preserved': True, 'focal_nuisance_fraction': float(solution[eta_index]),
        'translation_change_mean_m': float(np.linalg.norm(delta, axis=1).mean()),
        'translation_change_max_m': float(np.linalg.norm(delta, axis=1).max()),
        'camera_depth_change_mean_m': float(delta[:, 2].mean()),
        'camera_depth_change_range_m': [float(delta[:, 2].min()), float(delta[:, 2].max())],
        'baseline_camera_depth_range_m': [float(base_t[:, 2].min()), float(base_t[:, 2].max())],
        'refined_camera_depth_range_m': [float(final_cam[:, 2, 3].min()), float(final_cam[:, 2, 3].max())],
        'baseline_support_plane_std_m': float(np.std(support_level)),
        'refined_support_plane_std_m': float(np.std(support_level + delta @ nc)),
        'iterations': records,
    }
    (args.out.parent / 'diagnostics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
