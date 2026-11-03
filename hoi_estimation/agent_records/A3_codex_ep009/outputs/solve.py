#!/usr/bin/env python3
"""Conservative, offline refinement of the supplied red-bowl trajectory.

The calibrated distance of the mesh centroid from the camera is held fixed.
Most rotations are copied exactly. Only brief heading spikes with a tight
four-frame surrounding consensus and both hands in contact are repaired.
Silhouette translations are accepted only with substantial visible evidence
and an improvement measured after rendering the actual new 3D pose.

Run: python /work/outputs/solve.py --inputs /work/inputs \
        --out /work/outputs/object_poses_world.npz
"""
import os
# Fix CPU parallelism before importing numeric libraries.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import median_filter
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

# Explicit conservative limits, selected from the supplied imagery, not labels:
# the bowl is only about 60--100 pixels wide, and the mesh/video appearance
# cannot securely resolve broad heading ambiguities. A 5-pixel cap limits
# image-plane corrections to approximately 2 cm at the released distance.
SIDE = 160
BATCH = 24
SEARCH_LIMIT = 12.0
MAX_SHIFT = 5.0
SHRINK = 0.75
MIN_VISIBLE_AREA_RATIO = 0.75
MIN_CONTOUR_COVERAGE = 0.70
MIN_FIT_GAIN = 0.35  # pixels of mean symmetric boundary distance
MAX_FIT_COST = 2.50
MIN_FINAL_GAIN = 0.20
MIN_FINAL_IOU_GAIN = 0.005


def sample(a, p, outside=10):
    if len(p) == 0:
        return np.empty(0, np.float32)
    return cv2.remap(a, p[:, 0, None].astype(np.float32),
                     p[:, 1, None].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=outside)[:, 0]


def edge_points(mask):
    edge = (mask > 0).astype(np.uint8)
    edge -= cv2.erode(edge, np.ones((3, 3), np.uint8))
    ys, xs = np.where(edge)
    points = np.stack([xs, ys], 1).astype(np.float32)
    dist = cv2.distanceTransform(1 - edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return points, dist


def prepare_observation(obs, person, rgb):
    ys, xs = np.where(obs)
    hull = np.zeros_like(obs, dtype=np.uint8)
    if len(xs):
        cv2.fillConvexPoly(hull, cv2.convexHull(np.stack([xs, ys], 1)), 1)
    # This bowl has a convex outer silhouette. Its hull fills the thin chair
    # leg stripe and small finger cutouts; it does not infer hidden bowl sides.
    # SAM's person mask includes both the visible held bowl and the torso
    # behind it. Excluding that entire mask would discard real bowl edges.
    # Skin-coloured person pixels instead identify hand occlusion. These
    # colour bounds were checked on the supplied crops (tan hands, red bowl).
    r, g, b = np.moveaxis(rgb.astype(np.float32), -1, 0)
    skin = ((r > 45) & (g > 30) & (b > 15) & (r > 1.08*g)
            & (r < 1.9*g) & (g > 1.05*b) & (r-b > 10))
    human = (person.astype(bool) & skin & ~obs.astype(bool)).astype(np.uint8)
    valid = (1 - cv2.dilate(human, np.ones((5, 5), np.uint8))).astype(np.float32)
    po, dto = edge_points(hull)
    po = po[sample(valid, po, outside=0) > .5]
    return hull, valid, po, dto


class SilhouetteRenderer:
    def __init__(self, mesh):
        self.v = torch.as_tensor(np.array(mesh.vertices), dtype=torch.float32, device='cuda')
        self.f = torch.as_tensor(np.array(mesh.faces), dtype=torch.int32, device='cuda')
        self.ctx = dr.RasterizeCudaContext(device='cuda')

    def render(self, Rs, ts, Ks):
        R = torch.as_tensor(np.asarray(Rs), dtype=torch.float32, device='cuda')
        t = torch.as_tensor(np.asarray(ts), dtype=torch.float32, device='cuda')
        K = torch.as_tensor(np.asarray(Ks), dtype=torch.float32, device='cuda')
        v = self.v[None] @ R.transpose(1, 2) + t[:, None, :]
        z = v[:, :, 2]
        # nvdiffrast array row zero is the -1 clip-y end. Use image-down y.
        # +0.5 maps OpenCV integer pixel centres to raster pixel centres.
        x = 2*(K[:, 0, 0, None]*v[:, :, 0] + (K[:, 0, 2, None]+.5)*z)/SIDE-z
        y = 2*(K[:, 1, 1, None]*v[:, :, 1] + (K[:, 1, 2, None]+.5)*z)/SIDE-z
        near, far = .01, 20.
        clip = torch.stack([x, y, (far+near)/(far-near)*z
                            - 2*far*near/(far-near), z], -1).contiguous()
        raster, _ = dr.rasterize(self.ctx, clip, self.f, resolution=(SIDE, SIDE))
        mask = (raster[:, :, :, 3:] > 0).float()
        mask = dr.antialias(mask, raster, clip, self.f)
        # Quantize insignificant CUDA antialias accumulation noise.
        return np.round(mask[:, :, :, 0].cpu().numpy(), 4)

    def render_all(self, poses, Ks):
        out = []
        with torch.no_grad():
            for a in range(0, len(poses), BATCH):
                out.append(self.render(poses[a:a+BATCH, :3, :3],
                                       poses[a:a+BATCH, :3, 3], Ks[a:a+BATCH]))
        return np.concatenate(out)


def contour_objective(render, prepared):
    _, valid, po, dto = prepared
    pr, dtr = edge_points(render > .5)
    if len(po) < 10 or len(pr) < 10:
        return lambda v: 10.

    def cost(v):
        v = np.asarray(v, np.float32)
        pts = pr + v
        weights = sample(valid, pts, outside=0)
        d1 = np.minimum(sample(dto, pts), 8)
        d2 = np.minimum(sample(dtr, po-v), 8)
        if weights.sum() < 10:
            return 10.
        return float(.5*((d1*weights).sum()/weights.sum() + d2.mean()))
    return cost


def fit_translation(render, prepared):
    cost = contour_objective(render, prepared)
    grid = np.array([[x, y] for y in [-8, -4, 0, 4, 8] for x in [-8, -4, 0, 4, 8]])
    values = np.array([cost(v) for v in grid])
    initial = grid[np.argmin(values)].astype(float)
    fit = minimize(cost, initial, method='Powell',
                   bounds=[(-SEARCH_LIMIT, SEARCH_LIMIT)]*2,
                   options={'xtol': .03, 'ftol': 1e-5, 'maxiter': 15})
    shift = fit.x if fit.fun < min(values) else initial
    return shift, cost


def visible_iou(render, prepared):
    target, valid, _, _ = prepared
    a, b, v = render > .5, target > .5, valid > .5
    return float((a & b & v).sum()/max(((a | b) & v).sum(), 1))


def repair_heading_spikes(poses, contacts):
    result = poses.copy()
    R = poses[:, :3, :3]
    changed = np.zeros(len(R), bool)
    evidence = []
    # A heading spike lasting <= 2 frames, while both hands keep gripping,
    # cannot be supported by four nearby, mutually consistent orientations.
    # Longer ambiguities are deliberately left exactly as supplied.
    for length in (1, 2):
        for a in range(2, len(R)-length-1):
            b = a + length
            if changed[a:b].any() or contacts[a:b].min() < .95:
                continue
            anchors = [a-2, a-1, b, b+1]
            surrounding = Rotation.from_matrix(R[anchors])
            centre = surrounding.mean()
            spread = np.rad2deg((centre.inv()*surrounding).magnitude())
            difference = np.rad2deg((centre.inv()*Rotation.from_matrix(R[a:b])).magnitude())
            if spread.max() >= 8 or difference.min() <= 18 or difference.max() >= 45:
                continue
            rv = (Rotation.from_matrix(R[a:b]).inv()*centre).as_rotvec()
            if np.any(np.abs(rv[:, 1])/np.maximum(np.linalg.norm(rv, axis=1), 1e-12) < .98):
                continue
            target = Slerp([a-1, b], Rotation.from_matrix(R[[a-1, b]]))(np.arange(a, b)).as_matrix()
            # Preserve the opening axis, repairing only local-y heading.
            for j in range(a, b):
                rel = R[j].T @ target[j-a]
                theta = np.arctan2(rel[0, 2]-rel[2, 0], rel[0, 0]+rel[2, 2])
                result[j, :3, :3] = R[j] @ Rotation.from_euler('y', theta).as_matrix()
            changed[a:b] = True
            evidence.append({'first_frame': a, 'last_frame': b-1,
                             'surrounding_max_degrees': float(spread.max()),
                             'original_deviation_degrees': difference.tolist()})
    return result, changed, evidence


def make_translation(poses, shift, K, mesh_centre):
    # Distance means distance of the mesh's surface centroid, not its origin.
    # The supplied mesh centroid is ~1e-10 m from its file origin, but retaining
    # this explicit term makes the camera-distance constraint unambiguous.
    centres = np.einsum('nij,j->ni', poses[:, :3, :3], mesh_centre) + poses[:, :3, 3]
    pixels = centres @ K.T
    pixels = pixels[:, :2]/pixels[:, 2:3] + shift
    rays = np.c_[(pixels[:, 0]-K[0, 2])/K[0, 0],
                 (pixels[:, 1]-K[1, 2])/K[1, 1], np.ones(len(pixels))]
    rays *= (np.linalg.norm(centres, axis=1)/np.linalg.norm(rays, axis=1))[:, None]
    result = poses.copy()
    result[:, :3, 3] = rays-np.einsum('nij,j->ni', poses[:, :3, :3], mesh_centre)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('The supplied CUDA/nvdiffrast environment is required.')
    camera = json.loads((args.inputs/'camera.json').read_text())
    K, C = np.array(camera['K'], np.float64), np.array(camera['T_world_cam'], np.float64)
    with np.load(args.inputs/'reconstruction.npz') as data:
        baseline = data['T_world_mesh'].astype(np.float64)
        contacts = data['contact_prob_right_left']
    n = len(baseline)
    assert n == json.loads((args.inputs/'episode.json').read_text())['frames']
    poses = np.linalg.inv(C) @ baseline
    poses, rotation_changed, rotation_evidence = repair_heading_spikes(poses, contacts)
    mesh = trimesh.load(args.inputs/'object_mesh.glb', force='scene', process=False).to_geometry()
    mesh_centre = np.asarray(mesh.centroid)
    renderer = SilhouetteRenderer(mesh)
    pixels = poses[:, :3, 3] @ K.T
    pixels = pixels[:, :2]/pixels[:, 2:3]
    origins = np.floor(pixels-SIDE/2).astype(int)
    Ks = np.repeat(K[None], n, 0)
    Ks[:, :2, 2] -= origins
    gpu_start = time.perf_counter()
    masks = renderer.render_all(poses, Ks)
    torch.cuda.synchronize()
    render_seconds = time.perf_counter()-gpu_start
    print(f'Rendered {n} baseline silhouettes in {render_seconds:.2f} seconds.', flush=True)
    cap = cv2.VideoCapture(str(args.inputs/'video.mp4'))
    prepared = []
    raw_shift, costs, coverage, area_ratio, iou_before = [], [], [], [], []
    with h5py.File(args.inputs/'masks.h5', 'r') as h5:
        for i in range(n):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Video ended at frame {i}, expected {n} frames.')
            x, y = origins[i]
            assert x >= 0 and y >= 0 and x+SIDE <= bgr.shape[1] and y+SIDE <= bgr.shape[0]
            rgb = cv2.cvtColor(bgr[y:y+SIDE, x:x+SIDE], cv2.COLOR_BGR2RGB)
            obs = h5['object'][i, y:y+SIDE, x:x+SIDE].astype(np.uint8)
            person = h5['person'][i, y:y+SIDE, x:x+SIDE]
            pp = prepare_observation(obs, person, rgb)
            prepared.append(pp)
            shift, cost = fit_translation(masks[i], pp)
            raw_shift.append(shift)
            costs.append(cost)
            coverage.append((masks[i]*pp[1]).sum()/max(masks[i].sum(), 1))
            area_ratio.append(obs.sum()/max(masks[i].sum(), 1))
            iou_before.append(visible_iou(masks[i], pp))
    cap.release()
    # The masks are integer pixels and interpolation uses 1/32-pixel bins.
    # A 0.01-pixel grid removes irrelevant optimiser roundoff between runs.
    raw_shift = np.round(np.asarray(raw_shift), 2)
    coverage, area_ratio, iou_before = map(np.asarray, (coverage, area_ratio, iou_before))
    smooth_shift = median_filter(raw_shift, size=(5, 1), mode='nearest')
    score_before = np.array([c([0, 0]) for c in costs])
    score_fit = np.array([c(s) for c, s in zip(costs, smooth_shift)])
    good = ((area_ratio >= MIN_VISIBLE_AREA_RATIO) & (area_ratio <= 1.35)
            & (coverage >= MIN_CONTOUR_COVERAGE) & (score_fit < MAX_FIT_COST)
            & (score_before-score_fit >= MIN_FIT_GAIN)
            & (np.linalg.norm(raw_shift-smooth_shift, axis=1) < 1.5))
    # Require supporting evidence in at least three of five adjacent frames.
    support = np.convolve(np.pad(good.astype(int), 2, mode='edge'), np.ones(5), mode='valid')
    good &= support >= 3
    proposed_shift = np.round(SHRINK*smooth_shift, 2)
    proposed_shift *= np.minimum(1., MAX_SHIFT/np.maximum(np.linalg.norm(proposed_shift, axis=1), 1e-12))[:, None]
    proposed_shift[~good] = 0
    moved = make_translation(poses, proposed_shift, K, mesh_centre)
    candidates = np.where(good)[0]
    final_score, final_iou = score_before.copy(), iou_before.copy()
    if len(candidates):
        gpu_start = time.perf_counter()
        rerender = renderer.render_all(moved[candidates], Ks[candidates])
        torch.cuda.synchronize()
        render_seconds += time.perf_counter()-gpu_start
        for j, i in enumerate(candidates):
            score = contour_objective(rerender[j], prepared[i])([0, 0])
            iou = visible_iou(rerender[j], prepared[i])
            if score_before[i]-score < MIN_FINAL_GAIN or iou-iou_before[i] < MIN_FINAL_IOU_GAIN:
                good[i] = False
                proposed_shift[i] = 0
            else:
                final_score[i], final_iou[i] = score, iou
    result = baseline.copy()
    result[rotation_changed, :3, :3] = C[:3, :3] @ poses[rotation_changed, :3, :3]
    result[good, :3, 3] = moved[good, :3, 3] @ C[:3, :3].T+C[:3, 3]
    result[:, 3, :] = [0., 0., 0., 1.]
    before_cam = np.linalg.inv(C) @ baseline
    after_cam = np.linalg.inv(C) @ result
    before_centre = np.einsum('nij,j->ni', before_cam[:, :3, :3], mesh_centre)+before_cam[:, :3, 3]
    after_centre = np.einsum('nij,j->ni', after_cam[:, :3, :3], mesh_centre)+after_cam[:, :3, 3]
    distance_error = np.max(np.abs(np.linalg.norm(before_centre, axis=1)-np.linalg.norm(after_centre, axis=1)))
    assert distance_error < 1e-8
    assert result.shape == (n, 4, 4) and result.dtype == np.float64 and np.isfinite(result).all()
    assert np.allclose(np.linalg.det(result[:, :3, :3]), 1., atol=1e-5)
    assert np.max(np.abs(result[:, :3, :3].transpose(0, 2, 1) @ result[:, :3, :3]-np.eye(3))) < 1e-5
    unchanged = ~(good | rotation_changed)
    assert np.array_equal(result[unchanged], baseline[unchanged])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, T_world_mesh=result)
    elapsed = time.perf_counter()-start
    translation_m = np.linalg.norm(result[:, :3, 3]-baseline[:, :3, 3], axis=1)
    def distribution(x):
        return {'mean': float(np.mean(x)), 'median': float(np.median(x)),
                'min': float(np.min(x)), 'max': float(np.max(x))} if len(x) else {}
    metrics = {
        'frames': n, 'dtype': str(result.dtype),
        'translation_changed_frames': int(good.sum()),
        'rotation_changed_frames': np.where(rotation_changed)[0].tolist(),
        'rotation_evidence': rotation_evidence,
        'frames_copied_exactly': int(unchanged.sum()),
        'camera_centre_distance_max_error_m': float(distance_error),
        'translation_correction_m': distribution(translation_m[good]),
        'pixel_correction': distribution(np.linalg.norm(proposed_shift[good], axis=1)),
        'accepted_contour_distance_before_px': distribution(score_before[good]),
        'accepted_contour_distance_after_px': distribution(final_score[good]),
        'accepted_iou_before': distribution(iou_before[good]),
        'accepted_iou_after': distribution(final_iou[good]),
        'render_seconds': render_seconds, 'elapsed_seconds': elapsed,
        'uses_network': False, 'uses_models_or_weights': False,
    }
    (args.out.parent/'validation.json').write_text(json.dumps(metrics, indent=2)+'\n')
    with open(args.out.parent/'frame_diagnostics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'translation_changed', 'heading_changed', 'shift_x_px', 'shift_y_px',
                         'visible_area_ratio', 'coverage', 'contour_before_px', 'contour_after_px',
                         'iou_before', 'iou_after'])
        for i in range(n):
            writer.writerow([i, int(good[i]), int(rotation_changed[i]), *proposed_shift[i],
                             area_ratio[i], coverage[i], score_before[i], final_score[i],
                             iou_before[i], final_iou[i]])
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
