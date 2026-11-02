"""Conservative fixed-camera-distance silhouette registration for a tracked object."""
import os
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/work/tools/torch_extensions')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import json
import time
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

class Renderer:
    def __init__(self, mesh, K):
        self.K = K
        self.ctx = dr.RasterizeCudaContext(device='cuda:0')
        self.v = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device='cuda:0')
        self.f = torch.as_tensor(np.asarray(mesh.faces), dtype=torch.int32, device='cuda:0')
        self.uv = torch.as_tensor(np.asarray(mesh.visual.uv), dtype=torch.float32, device='cuda:0')[None]
        tex = np.array(mesh.visual.material.baseColorTexture.convert('RGB'))[::-1].copy()/255.
        self.tex = torch.as_tensor(tex, dtype=torch.float32, device='cuda:0')[None]

    @torch.no_grad()
    def render(self, pose, left, top, size=160, scale=1, color=False):
        T = torch.as_tensor(pose, dtype=torch.float32, device='cuda:0')
        v = self.v @ T[:3,:3].T + T[:3,3]
        z = v[:,2]
        x = 2*self.K[0,0]/size*v[:,0] + (2*(self.K[0,2]-left)/size-1)*z
        y = 2*self.K[1,1]/size*v[:,1] + (2*(self.K[1,2]-top)/size-1)*z
        clip = torch.stack([x,y,1.002002*z-.2002002,z],-1)[None]
        rast,_ = dr.rasterize(self.ctx, clip, self.f, resolution=[size*scale,size*scale])
        mask = (rast[0,:,:,3]>0).cpu().numpy().astype(np.uint8)
        if not color:
            return mask
        uv,_ = dr.interpolate(self.uv, rast, self.f)
        rgb = dr.texture(self.tex, uv, filter_mode='linear')[0]
        rgb = (rgb.cpu().numpy()*255).clip(0,255).astype(np.uint8)
        rgb[mask==0]=35
        return mask,rgb


# All image thresholds are in original-video pixels. The opening suppresses the
# thin, nonrigid handle and SAM's dangling contour fragments; it leaves the rigid
# basket body. No frame-specific or manually selected pose corrections are used.
CROP_SIZE = 224
SEARCH_RADIUS = 48
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def clean_observation(mask):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, KERNEL)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 150:
        return np.zeros_like(mask)
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def clean_render(mask):
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)


def overlap(render, obs, person):
    render = render.astype(bool)
    obs = obs.astype(bool)
    visible = render & ~(person.astype(bool) & ~obs)
    intersection = np.count_nonzero(visible & obs)
    union = np.count_nonzero(visible | obs)
    return intersection / max(union, 1)


def register(render, obs, person):
    """Translation search with foreground-person occlusion, without depth fitting."""
    yy, xx = np.where(render)
    if not len(xx) or obs.sum() < 150:
        return np.zeros(7, dtype=np.float64)
    x0, x1, y0, y1 = xx.min(), xx.max()+1, yy.min(), yy.max()+1
    template = render[y0:y1, x0:x1].astype(np.float32)
    area = template.sum()
    observed = obs.astype(np.float32)
    intersection = cv2.matchTemplate(observed, template, cv2.TM_CCORR).clip(0, area)
    occluded = cv2.matchTemplate(person.astype(np.float32)*(1-observed), template,
                                cv2.TM_CCORR).clip(0, area)
    scores = intersection / np.maximum(area + observed.sum() - intersection - occluded, 1)
    original = float(scores[y0, x0])
    ys, xs = np.indices(scores.shape)
    allowed = ((abs(xs-x0) <= SEARCH_RADIUS) & (abs(ys-y0) <= SEARCH_RADIUS)
               & (occluded <= .60*area))
    scores[~allowed] = -1
    y, x = np.unravel_index(np.argmax(scores), scores.shape)
    if scores[y, x] <= 0:
        return np.zeros(7, dtype=np.float64)
    # Average only the immediate near-optimal plateau, avoiding pixel-grid jitter.
    plateau = ((scores >= scores[y, x]-.003) & (abs(xs-x) <= 2) & (abs(ys-y) <= 2))
    dx = float(xs[plateau].mean()-x0)
    dy = float(ys[plateau].mean()-y0)
    return np.array([dx, dy, scores[y, x], original,
                     occluded[y, x]/area, observed.sum()/area, area])


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def choose_corrections(measured, centers):
    from scipy.ndimage import maximum_filter1d, median_filter
    dxdy = measured[:, :2]
    fit, before, occlusion, area = measured[:, 2:6].T
    # High confidence needs most of the basket visible. Slightly weaker frames
    # are accepted only near strong frames, to avoid threshold flicker.
    strong = ((fit >= .84) & (occlusion <= .18) & (area >= .80) & (area <= 1.25))
    support = ((fit >= .75) & (occlusion <= .35) & (area >= .65) & (area <= 1.35))
    eligible = support & maximum_filter1d(strong.astype(np.uint8), size=9).astype(bool)
    for a, b in runs(eligible):
        if b-a < 7 or strong[a:b].sum() < 3:
            eligible[a:b] = False
    corrections = np.zeros_like(dxdy)
    for a, b in runs(eligible):
        # Filter the registered image position, not the correction: filtering
        # corrections would preserve the baseline's own frame-to-frame jitter.
        target = centers[a:b] + dxdy[a:b]
        smoothed = .5*target + .5*median_filter(target, size=(3, 1), mode='nearest')
        delta = smoothed - centers[a:b]
        gain = fit[a:b]-before[a:b]
        strength = np.clip((gain-.015)/.04, 0, 1)
        strength *= np.clip((np.linalg.norm(delta, axis=1)-.75)/1.75, 0, 1)
        # Ease back into the baseline at visibility boundaries. Never extend a
        # correction into unseen frames or across an unsupported interval.
        edge = np.ones(b-a)
        if a > 0:
            edge = np.minimum(edge, np.arange(1, b-a+1)/3)
        if b < len(eligible):
            edge = np.minimum(edge, np.arange(b-a, 0, -1)/3)
        corrections[a:b] = delta * (strength*edge)[:, None]
    return corrections, strong, eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.inputs/'reconstruction.npz') as data:
        baseline = data['T_world_mesh'].astype(np.float64)
    camera = json.loads((args.inputs/'camera.json').read_text())
    K = np.array(camera['K'], dtype=np.float64)
    world_from_cam = np.array(camera['T_world_cam'], dtype=np.float64)
    cam_from_world = np.linalg.inv(world_from_cam)
    cam_poses = cam_from_world @ baseline
    mesh = trimesh.load(args.inputs/'object_mesh.glb', force='scene', process=False).to_geometry()
    # The supplied mesh is surface-centred (centroid essentially zero). Explicit
    # centroid handling keeps the distance invariant independent of that detail.
    centroid = np.asarray(mesh.centroid)
    cam_centers = cam_poses[:, :3, :3] @ centroid + cam_poses[:, :3, 3]
    distances = np.linalg.norm(cam_centers, axis=1)
    image_centers = cam_centers @ K.T
    image_centers = image_centers[:, :2]/image_centers[:, 2:]
    origins = np.rint(image_centers-CROP_SIZE/2).astype(int)
    renderer = Renderer(mesh, K)
    F = len(baseline)
    measured = np.zeros((F, 7), dtype=np.float64)
    observations, persons = [], []
    before_iou = np.full(F, np.nan)
    with h5py.File(args.inputs/'masks.h5', 'r', rdcc_nbytes=256*1024**2) as masks:
        assert len(masks['object']) == F
        for i, (left, top) in enumerate(origins):
            # Out-of-image crops are unsupported rather than guessed.
            if (left < 0 or top < 0 or left+CROP_SIZE > camera['width']
                    or top+CROP_SIZE > camera['height']):
                observations.append(None); persons.append(None)
                continue
            obs = clean_observation(masks['object'][i, top:top+CROP_SIZE, left:left+CROP_SIZE])
            person = masks['person'][i, top:top+CROP_SIZE, left:left+CROP_SIZE].astype(np.uint8)
            observations.append(obs); persons.append(person)
            if obs.sum() < 150:
                continue
            rendered = clean_render(renderer.render(cam_poses[i], left, top, CROP_SIZE))
            measured[i] = register(rendered, obs, person)
            before_iou[i] = overlap(rendered, obs, person)
    corrections, strong, eligible = choose_corrections(measured, image_centers)
    result = baseline.copy()
    after_iou = before_iou.copy()
    accepted = np.zeros(F, dtype=bool)
    for i in np.flatnonzero(np.linalg.norm(corrections, axis=1) > .1):
        uv = image_centers[i] + corrections[i]
        ray = np.linalg.solve(K, np.r_[uv, 1.])
        new_center = ray/np.linalg.norm(ray)*distances[i]
        candidate = cam_poses[i].copy()
        candidate[:3, 3] += new_center - cam_centers[i]
        left, top = origins[i]
        rendered = clean_render(renderer.render(candidate, left, top, CROP_SIZE))
        score = overlap(rendered, observations[i], persons[i])
        # Re-render the actual fixed-range 3D result: a 2D shift is only an
        # approximation. Keep the baseline if the verified improvement vanishes.
        if score < before_iou[i] + .01:
            corrections[i] = 0
            continue
        result[i, :3, 3] += world_from_cam[:3, :3] @ (new_center-cam_centers[i])
        after_iou[i] = score
        accepted[i] = True
    corrections[~accepted] = 0
    final_cam = cam_from_world @ result
    final_centers = final_cam[:, :3, :3] @ centroid + final_cam[:, :3, 3]
    range_error = float(np.max(np.abs(np.linalg.norm(final_centers, axis=1)-distances)))
    assert result.shape == (F, 4, 4) and result.dtype == np.float64
    assert np.isfinite(result).all()
    assert np.array_equal(result[:, :3, :3], baseline[:, :3, :3])
    assert np.array_equal(result[~accepted], baseline[~accepted])
    assert np.array_equal(result[:, 3, :], baseline[:, 3, :])
    assert range_error < 1e-10
    np.savez_compressed(args.out, T_world_mesh=result)
    np.savez_compressed(args.out.parent/'alignment_audit.npz',
                        measurements=measured, image_corrections=corrections,
                        before_iou=before_iou, after_iou=after_iou,
                        strong=strong, eligible=eligible, accepted=accepted)
    changes = np.linalg.norm(result[:, :3, 3]-baseline[:, :3, 3], axis=1)
    metrics = {
        'frames': F, 'changed_frames': int(accepted.sum()),
        'unchanged_frames': int((~accepted).sum()),
        'rotation_matrices_bitwise_unchanged': True,
        'max_camera_center_distance_change_m': range_error,
        'mean_iou_before_changed_frames': float(np.mean(before_iou[accepted])),
        'mean_iou_after_changed_frames': float(np.mean(after_iou[accepted])),
        'median_translation_change_m': float(np.median(changes[accepted])),
        'max_translation_change_m': float(changes.max()),
        'accepted_intervals_end_exclusive': runs(accepted),
        'elapsed_seconds': time.perf_counter()-start,
        'example_frames': {str(i): {'accepted': bool(accepted[i]),
                          'before_iou': float(before_iou[i]), 'after_iou': float(after_iou[i]),
                          'image_correction_px': corrections[i].tolist(),
                          'translation_change_m': float(changes[i])}
                          for i in [0, 200, 300, 595, 638, 681, 700, 723, 736, 851] if i < F},
    }
    # Cast numpy integers in interval tuples for a portable JSON audit.
    metrics['accepted_intervals_end_exclusive'] = [[int(a), int(b)] for a, b in runs(accepted)]
    (args.out.parent/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k not in
                      ['accepted_intervals_end_exclusive','example_frames']}, indent=2))


if __name__ == '__main__':
    main()
