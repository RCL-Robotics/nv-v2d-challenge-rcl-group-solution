#!/usr/bin/env python3
"""Conservative silhouette refinement at fixed baseline camera-centre distance.

Uses only the supplied mesh, masks, video, camera and baseline. There are no
hand-selected frame intervals or symmetry flips. All thresholds below are in
pixels, radians or dimensionless silhouette discrepancy, as documented.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import gaussian_filter1d, median_filter, distance_transform_edt
from scipy.spatial.transform import Rotation

SIZE = 256
BATCH = 20
STEPS = 140
ROT_SCALE = 0.03


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cv2.setNumThreads(1)
    inp = args.inputs
    camera = json.loads((inp/'camera.json').read_text())
    K = np.array(camera['K'], dtype=np.float64)
    world_cam = np.array(camera['T_world_cam'], dtype=np.float64)
    with np.load(inp/'reconstruction.npz') as data:
        baseline_world = data['T_world_mesh'].astype(np.float64)
    baseline = np.linalg.inv(world_cam)[None] @ baseline_world
    count = len(baseline)
    mesh = trimesh.load(inp/'object_mesh.glb', force='scene', process=False).to_geometry()
    ctx = dr.RasterizeCudaContext()
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device='cuda')
    kt = torch.tensor(K.T, dtype=torch.float32, device='cuda')
    principal = torch.tensor(K[:2, 2], dtype=torch.float32, device='cuda')
    uv = baseline[:, :3, 3] @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    origins = np.rint(uv - SIZE/2).astype(int)
    target = np.zeros((count, SIZE, SIZE), np.uint8)
    valid = np.zeros_like(target)
    distances = np.zeros_like(target, dtype=np.float32)
    video = cv2.VideoCapture(str(inp/'video.mp4'))
    with h5py.File(inp/'masks.h5', 'r') as masks:
        for f in range(count):
            x, y = origins[f]
            if x < 0 or y < 0 or x+SIZE > camera['width'] or y+SIZE > camera['height']:
                raise ValueError('Object crop extends beyond image')
            target[f] = masks['object'][f, y:y+SIZE, x:x+SIZE]
            person = masks['person'][f, y:y+SIZE, x:x+SIZE].astype(np.uint8)
            # Ignore the human and a two-pixel guard band around their silhouette.
            valid[f] = 1-cv2.dilate(person, np.ones((5, 5), np.uint8))
            ok, image = video.read()
            if not ok:
                raise ValueError(f'Missing video frame {f}')
            gray = cv2.cvtColor(image[y:y+SIZE, x:x+SIZE], cv2.COLOR_BGR2GRAY)
            edge = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), .6), 25, 65)
            distances[f] = cv2.distanceTransform(255-edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    video.release()

    def transforms(params, base):
        w = params[:, :3]*ROT_SCALE
        wx, wy, wz = w.unbind(-1)
        z = wx*0
        skew = torch.stack([z, -wz, wy, wz, z, -wx, -wy, wx, z], -1).reshape(-1, 3, 3)
        r = torch.matrix_exp(skew) @ base[:, :3, :3]
        t = base[:, :3, 3]
        projected = t @ kt
        xy = projected[:, :2]/projected[:, 2:] + params[:, 3:]
        rays = torch.cat([(xy-principal)/K[0, 0], torch.ones_like(xy[:, :1])], 1)
        # Radial distance is fixed, not fitted from the object's image size.
        t = rays/torch.linalg.norm(rays, dim=1, keepdim=True)*torch.linalg.norm(t, dim=1, keepdim=True)
        return r, t

    def render(params, base, origin):
        r, t = transforms(params, base)
        vc = vertices[None] @ r.transpose(1, 2) + t[:, None]
        projected = vc @ kt
        px = projected[:, :, 0]/projected[:, :, 2]-origin[:, None, 0]
        py = projected[:, :, 1]/projected[:, :, 2]-origin[:, None, 1]
        depth = vc[:, :, 2]
        clip = torch.stack([(2*px/SIZE-1)*depth, (2*py/SIZE-1)*depth,
                            ((depth-.1)/9.9*2-1)*depth, depth], -1)
        rast, _ = dr.rasterize(ctx, clip, faces, [SIZE, SIZE])
        alpha = (rast[..., 3:] > 0).float()
        return dr.antialias(alpha, rast, clip, faces)[..., 0]

    proposals = np.zeros((count, 5), np.float64)
    for first in range(0, count, BATCH):
        sl = slice(first, min(first+BATCH, count))
        base = torch.tensor(baseline[sl], dtype=torch.float32, device='cuda')
        origin = torch.tensor(origins[sl], device='cuda')
        gt = torch.tensor(target[sl], dtype=torch.float32, device='cuda')
        va = torch.tensor(valid[sl], dtype=torch.float32, device='cuda')
        area = (gt*va).sum((1, 2)).clamp_min(1)
        params = torch.nn.Parameter(torch.zeros((len(base), 5), device='cuda'))
        optim = torch.optim.Adam([params], lr=.12)
        for step in range(STEPS):
            optim.zero_grad()
            alpha = render(params, base, origin)
            data = ((alpha-gt).square()*va).sum((1, 2))/area
            # The baseline is a strong prior. These costs make ambiguous
            # rotations and large translations less attractive.
            prior = .0003*params[:, :3].square().sum(1)+.00008*params[:, 3:].square().sum(1)
            (data+prior).sum().backward()
            optim.step()
            with torch.no_grad():
                params[:, :3].clamp_(-2, 2)
                params[:, 3:].clamp_(-9, 9)
        proposals[sl] = params.detach().cpu().numpy()
        if first % 100 == 0:
            print(f'Fitted {sl.stop}/{count} frames', flush=True)
    # Only smooth small corrections, leaving the original motion intact.
    proposals = gaussian_filter1d(median_filter(proposals, size=(5, 1), mode='nearest'),
                                 sigma=3, axis=0, mode='nearest')
    rot = np.linalg.norm(proposals[:, :3]*ROT_SCALE, axis=1)
    proposals[:, :3] *= np.minimum(1, np.deg2rad(4)/np.maximum(rot, 1e-12))[:, None]

    def evaluate(params):
        silhouette = np.zeros(count)
        edge_error = np.zeros(count)
        for first in range(0, count, BATCH):
            sl = slice(first, min(first+BATCH, count))
            with torch.no_grad():
                alpha = render(torch.tensor(params[sl], dtype=torch.float32, device='cuda'),
                               torch.tensor(baseline[sl], dtype=torch.float32, device='cuda'),
                               torch.tensor(origins[sl], device='cuda')).cpu().numpy()
            for i, f in enumerate(range(sl.start, sl.stop)):
                silhouette[f] = (np.abs(alpha[i]-target[f])*valid[f]).sum()/max(1, (target[f]*valid[f]).sum())
                contours, _ = cv2.findContours((alpha[i] > .5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                outline = np.zeros((SIZE, SIZE), np.uint8)
                cv2.drawContours(outline, contours, -1, 1, 1)
                keep = (outline > 0) & (valid[f] > 0)
                edge_error[f] = np.minimum(distances[f][keep], 5).mean() if keep.any() else 5
        return silhouette, edge_error

    initial_sil, initial_edge = evaluate(np.zeros_like(proposals))
    proposed_sil, proposed_edge = evaluate(proposals)
    # Require agreement from both segmentation and actual video edges over a
    # 21-frame window. This rejects small silhouette gains caused by mask bias.
    sil_gain = (initial_sil-proposed_sil)/np.maximum(initial_sil, 1e-9)
    edge_gain = initial_edge-proposed_edge
    robust_sil = median_filter(sil_gain, size=21, mode='nearest')
    robust_edge = median_filter(edge_gain, size=21, mode='nearest')
    visible = (target*valid).sum((1, 2))/np.maximum(1, target.sum((1, 2)))
    eligible = (robust_sil > .07) & (robust_edge > .08) & (visible > .55)
    # A soft transition over about one third of a second avoids abrupt switches.
    weight = gaussian_filter1d(eligible.astype(float), sigma=5, mode='nearest')
    weight *= np.clip((robust_sil-.03)/.04, 0, 1)
    weight *= np.clip((robust_edge-.02)/.06, 0, 1)
    weight[weight < .05] = 0
    selected = proposals*weight[:, None]
    final_sil, final_edge = evaluate(selected)
    # Fail closed for individual frames where the smoothed proposal has clear
    # contrary image evidence. Exact baseline values are retained in that case.
    rejected = (final_sil > initial_sil+.002) | (final_edge > initial_edge+.08)
    # Taper to every rejected frame over six frames to prevent single-frame
    # jumps between a corrected pose and the trusted baseline.
    for _ in range(3):
        if not rejected.any():
            break
        taper = np.clip(distance_transform_edt(~rejected)/6, 0, 1)
        taper = taper*taper*(3-2*taper)
        selected *= taper[:, None]
        final_sil, final_edge = evaluate(selected)
        more = (final_sil > initial_sil+.002) | (final_edge > initial_edge+.08)
        if not more.any():
            break
        rejected |= more
    selected[rejected] = 0
    final_sil[rejected] = initial_sil[rejected]
    final_edge[rejected] = initial_edge[rejected]

    # Compute the released pose in float64; avoid round-tripping unchanged poses.
    changed = np.linalg.norm(selected, axis=1) > 1e-10
    result = baseline_world.copy()
    r = Rotation.from_rotvec(selected[:, :3]*ROT_SCALE).as_matrix() @ baseline[:, :3, :3]
    xy = uv+selected[:, 3:]
    rays = np.column_stack([(xy-K[:2, 2])/K[0, 0], np.ones(count)])
    translation = rays/np.linalg.norm(rays, axis=1)[:, None]*np.linalg.norm(baseline[:, :3, 3], axis=1)[:, None]
    refined = baseline.copy()
    refined[:, :3, :3] = r
    refined[:, :3, 3] = translation
    result[changed] = (world_cam[None] @ refined)[changed]
    assert result.shape == (count, 4, 4) and np.isfinite(result).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, T_world_mesh=result.astype(np.float64))
    np.savez_compressed(args.out.parent/'refinement_diagnostics.npz',
                        proposals=proposals, selected=selected, changed=changed,
                        initial_silhouette=initial_sil, final_silhouette=final_sil,
                        initial_edge=initial_edge, final_edge=final_edge,
                        visible_fraction=visible, weight=weight)
    final_camera = np.linalg.inv(world_cam)[None] @ result
    radius_error = np.abs(np.linalg.norm(final_camera[:, :3, 3], axis=1)-np.linalg.norm(baseline[:, :3, 3], axis=1))
    summary = dict(frames=count, changed_frames=int(changed.sum()),
                   silhouette_before=float(initial_sil.mean()), silhouette_after=float(final_sil.mean()),
                   edge_pixels_before=float(initial_edge.mean()), edge_pixels_after=float(final_edge.mean()),
                   max_rotation_degrees=float(np.rad2deg(np.linalg.norm(selected[:, :3]*ROT_SCALE, axis=1)).max()),
                   max_image_shift_pixels=float(np.linalg.norm(selected[:, 3:], axis=1).max()),
                   max_translation_change_metres=float(np.linalg.norm(result[:, :3, 3]-baseline_world[:, :3, 3], axis=1).max()),
                   max_camera_distance_change_metres=float(radius_error.max()),
                   runtime_seconds=time.perf_counter()-start)
    (args.out.parent/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
