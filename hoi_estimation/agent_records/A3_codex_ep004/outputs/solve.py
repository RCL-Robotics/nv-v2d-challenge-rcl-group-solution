"""Conservative, offline refinement of the released baseball-bat trajectory.

The renderer estimates only the long-axis direction and the image position.
Camera distance is a hard constraint. Axial twist is inherited from the input
rotation by applying the shortest rotation between the old and new long axes.
See REPORT.md and TOOLS.md for evidence, thresholds, and artifact provenance.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TORCH_EXTENSIONS_DIR', '/work/tools/torch_extensions')

import cv2
import h5py
import numpy as np
import torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation

RESOLUTION = 128
BATCH = 48
SEEDS = 9
STEPS = 160
RENDER_MESH = Path('/work/tools/bat_render_mesh.npz')
RENDER_MESH_SHA256 = '5814d443720e06b41fe7b60b5d0d833c718477d17fb91f8f0f167d5b03362c67'
SOURCE_MESH_SHA256 = '5aa2294d698eb370d0d1435963e42022cc1064faa355b547e492ffe893e51cd7'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(inputs, camera_poses, K):
    targets, occluders, crops, axes, areas = [], [], [], [], []
    with h5py.File(inputs / 'masks.h5', 'r') as masks:
        for frame, pose in enumerate(camera_poses):
            mask = masks['object'][frame].astype(np.uint8)
            y, x = np.where(mask)
            if len(x) < 10:
                raise ValueError(f'Insufficient object mask at frame {frame}')
            points = np.column_stack((x, y))
            centre = points.mean(axis=0)
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
            axis = eigenvectors[:, -1]
            along = (points - centre) @ axis
            normal = np.array([-axis[1], axis[0]])
            widths = [np.std((points - centre)[along < np.percentile(along, 20)] @ normal),
                      np.std((points - centre)[along > np.percentile(along, 80)] @ normal)]
            # The +z mesh end is the narrow handle. Both polarities are tried,
            # so an occluded or foreshortened barrel cannot decide the result alone.
            if widths[1] > widths[0]:
                axis = -axis
            # These bounds enclose this mesh, with a small radial margin.
            bounds = np.array([[0, 0, -.21], [0, 0, .333], [.05, .05, 0], [-.05, -.05, 0]])
            projected = (bounds @ pose[:3, :3].T + pose[:3, 3]) @ K.T
            projected = projected[:, :2] / projected[:, 2:]
            lower = np.minimum(points.min(axis=0), projected.min(axis=0))
            upper = np.maximum(points.max(axis=0), projected.max(axis=0))
            length = np.max(upper - lower) * 1.3 + 12
            origin = (lower + upper - length) / 2
            warp = np.array([[RESOLUTION / length, 0, -origin[0] * RESOLUTION / length],
                             [0, RESOLUTION / length, -origin[1] * RESOLUTION / length]])
            targets.append(cv2.warpAffine(mask.astype(np.float32), warp,
                                          (RESOLUTION, RESOLUTION), flags=cv2.INTER_AREA))
            person = cv2.dilate(masks['person'][frame].astype(np.uint8), np.ones((5, 5), np.uint8))
            occluders.append(cv2.warpAffine(person.astype(np.float32), warp,
                                            (RESOLUTION, RESOLUTION), flags=cv2.INTER_NEAREST))
            crops.append([*origin, length])
            axes.append([*axis, eigenvalues[1] / max(eigenvalues[0], 1e-8),
                         max(widths) / max(min(widths), .01), *centre])
            areas.append(len(x))
    return tuple(np.asarray(x) for x in (targets, occluders, crops, axes, areas))


def swing_rotation(theta, alpha, baseline_rotation):
    """Shortest swing: do not independently optimize the unobservable axial roll."""
    new_axis = torch.stack((torch.cos(alpha) * torch.cos(theta),
                            torch.cos(alpha) * torch.sin(theta), torch.sin(alpha)), dim=1)
    old_axis = baseline_rotation[:, :, 2]
    old_axis = old_axis / old_axis.norm(dim=1, keepdim=True)
    cross = torch.cross(old_axis, new_axis, dim=1)
    dot = (old_axis * new_axis).sum(dim=1)
    quaternion = torch.cat(((1 + dot)[:, None], cross), dim=1)
    norm = quaternion.norm(dim=1, keepdim=True)
    # At an exact antipode, camera +z supplies a deterministic perpendicular
    # swing axis. This branch is needed only for the alternate initialization.
    camera_z = torch.zeros_like(old_axis)
    camera_z[:, 2] = 1
    perpendicular = camera_z - old_axis * old_axis[:, 2:3]
    perpendicular = perpendicular / perpendicular.norm(dim=1, keepdim=True).clamp_min(1e-8)
    fallback = torch.cat((torch.zeros_like(dot[:, None]), perpendicular), dim=1)
    quaternion = torch.where(norm > 1e-5, quaternion / norm.clamp_min(1e-8), fallback)
    w, x, y, z = quaternion.unbind(dim=1)
    matrix = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                          2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                          2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), dim=1).reshape(-1, 3, 3)
    return matrix @ baseline_rotation


def fit_candidates(inputs, baseline, foundation, K, output_dir):
    start = time.perf_counter()
    target, allow, crops, axes, areas = prepare(inputs, baseline, K)
    preparation_seconds = time.perf_counter() - start
    print(f'Masks prepared in {preparation_seconds:.1f}s', flush=True)
    mesh = np.load(RENDER_MESH)
    device = torch.device('cuda:0')
    vertices = torch.tensor(mesh['vertices'], dtype=torch.float32, device=device)
    triangles = torch.tensor(mesh['faces'].astype(np.int32), device=device)
    context = dr.RasterizeCudaContext(device=device)
    principal = torch.tensor(K[:2, 2], dtype=torch.float32, device=device)
    results, losses, original_losses = [], [], []
    for lower in range(0, len(baseline), BATCH):
        upper = min(lower + BATCH, len(baseline))
        count = upper - lower
        ids = np.tile(np.arange(lower, upper), SEEDS)
        total = len(ids)
        original_rotation = torch.tensor(baseline[ids, :3, :3], dtype=torch.float32, device=device)
        old_axis = original_rotation[:, :, 2]
        original_theta = torch.atan2(old_axis[:, 1], old_axis[:, 0])
        original_alpha = torch.asin(old_axis[:, 2].clamp(-1, 1))
        original_translation = torch.tensor(baseline[ids, :3, 3], dtype=torch.float32, device=device)
        distance = original_translation.norm(dim=1)
        original_uv = original_translation[:, :2] / original_translation[:, 2:3] * K[0, 0] + principal
        crop = torch.tensor(crops[ids], dtype=torch.float32, device=device)
        observed = torch.tensor(target[ids], device=device)
        occluded = torch.tensor(allow[ids], device=device)
        observed_theta = torch.tensor(np.arctan2(axes[ids, 1], axes[ids, 0]), device=device, dtype=torch.float32)
        theta, alpha = original_theta.clone(), original_alpha.clone()
        theta[count:] = observed_theta[count:]
        for seed in (2, 4, 6):
            theta[seed*count:(seed+1)*count] += np.pi
        alpha[3*count:5*count] = 0
        alpha[5*count:7*count] = torch.sign(original_alpha[5*count:7*count]) * .6
        fp_axis = foundation[lower:upper, :3, 2]
        theta[7*count:8*count] = torch.tensor(np.arctan2(fp_axis[:, 1], fp_axis[:, 0]), device=device)
        alpha[7*count:8*count] = torch.tensor(np.arcsin(np.clip(fp_axis[:, 2], -1, 1)), device=device)
        theta[8*count:] = original_theta[8*count:]
        alpha[8*count:] = original_alpha[8*count:]
        parameters = torch.stack((theta, alpha, torch.zeros_like(theta), torch.zeros_like(theta)), dim=1)
        # Baseline-centred and mask-centred starts avoid disjoint-silhouette minima.
        parameters[count:, 2:] = (torch.tensor(axes[ids[count:], 4:6], dtype=torch.float32, device=device)
                                   - original_uv[count:]) / crop[count:, 2:3]
        parameters = parameters.detach()
        optimizer = torch.optim.Adam([parameters], lr=.025)

        def render(params):
            rotation = swing_rotation(params[:, 0], params[:, 1], original_rotation)
            uv = original_uv + params[:, 2:] * crop[:, 2:3]
            ray = torch.cat(((uv - principal) / K[0, 0], torch.ones((total, 1), device=device)), dim=1)
            translation = ray / ray.norm(dim=1, keepdim=True) * distance[:, None]
            points = vertices[None] @ rotation.transpose(1, 2) + translation[:, None]
            clip_x = (2*K[0, 0]/crop[:, 2:3] * points[:, :, 0]
                      + (2*(K[0, 2]-crop[:, 0])/crop[:, 2]-1)[:, None] * points[:, :, 2])
            clip_y = (2*K[1, 1]/crop[:, 2:3] * points[:, :, 1]
                      + (2*(K[1, 2]-crop[:, 1])/crop[:, 2]-1)[:, None] * points[:, :, 2])
            clip = torch.stack((clip_x, clip_y, 1.0002*points[:, :, 2]-.020002, points[:, :, 2]), dim=-1).contiguous()
            raster, _ = dr.rasterize(context, clip, triangles, resolution=[RESOLUTION, RESOLUTION])
            # Hard coverage avoids nondeterministic scatter additions in the
            # antialias backward pass. Central numerical differences below
            # supply a deterministic, pixel-scale silhouette derivative.
            coverage = (raster[:, :, :, 3] > 0).float()
            return coverage, rotation, translation

        def image_loss(coverage):
            missing = (observed * (1-coverage)).sum(dim=(1, 2))
            extra = (coverage * (1-observed) * (1-occluded)).sum(dim=(1, 2))
            return (missing + extra) / (observed.sum(dim=(1, 2)) + 1)

        with torch.no_grad():
            old_parameters = torch.stack((original_theta, original_alpha,
                                          torch.zeros_like(theta), torch.zeros_like(theta)), dim=1)
            base_loss = image_loss(render(old_parameters)[0]).cpu().numpy()[:count]
        best = np.full(total, np.inf)
        best_parameters = np.zeros((total, 4), dtype=np.float32)
        for iteration in range(STEPS):
            optimizer.zero_grad()
            coverage, _, _ = render(parameters)
            # No camera-distance variable exists. The tilt prior discourages
            # fitting the uncertain scale/depth sign from the silhouette alone.
            objective = (image_loss(coverage) + .06*(parameters[:, 1]-original_alpha).square()
                         + .04*parameters[:, 2:].square().sum(dim=1))
            values = objective.detach().cpu().numpy()
            better = values < best
            best[better] = values[better]
            best_parameters[better] = parameters.detach().cpu().numpy()[better]
            deltas = [.006, .012, .0035, .0035] if iteration <= 100 else [.003, .006, .002, .002]
            gradient = torch.zeros_like(parameters)
            with torch.no_grad():
                for component, epsilon in enumerate(deltas):
                    plus, minus = parameters.clone(), parameters.clone()
                    plus[:, component] += epsilon
                    minus[:, component] -= epsilon
                    gradient[:, component] = (image_loss(render(plus)[0]) - image_loss(render(minus)[0])) / (2*epsilon)
                gradient[:, 1] += .12*(parameters[:, 1]-original_alpha)
                gradient[:, 2:] += .08*parameters[:, 2:]
            parameters.grad = gradient
            optimizer.step()
            with torch.no_grad():
                parameters[:, 1].clamp_(-1.48, 1.48)
                parameters[:, 2:].clamp_(-.5, .5)
            if iteration == 100:
                optimizer.param_groups[0]['lr'] = .008
        with torch.no_grad():
            coverage, rotation, translation = render(torch.tensor(best_parameters, device=device))
            errors = image_loss(coverage).cpu().numpy()
        poses = np.tile(np.eye(4), (total, 1, 1))
        poses[:, :3, :3] = rotation.cpu().numpy()
        poses[:, :3, 3] = translation.cpu().numpy()
        results.append(poses.reshape(SEEDS, count, 4, 4).transpose(1, 0, 2, 3))
        losses.append(errors.reshape(SEEDS, count).T)
        original_losses.append(base_loss)
        print(f'Fitted {upper}/{len(baseline)} frames; {time.perf_counter()-start:.1f}s', flush=True)
    candidates = np.concatenate(results)
    errors = np.concatenate(losses)
    original_errors = np.concatenate(original_losses)
    gpu_seconds = time.perf_counter() - start - preparation_seconds
    np.savez(output_dir / 'fit_diagnostics.npz', candidates=candidates, losses=errors,
             base_losses=original_errors, crops=crops, axes=axes, areas=areas)
    return candidates, errors, original_errors, preparation_seconds, gpu_seconds


def appearance(inputs, candidates, K, output_dir):
    """Check that the inferred barrel is brighter than the inferred dark handle.

    The least-squares intersection of each pixel ray with the bat axis gives
    approximate mesh-local z. Only well separated axial regions are compared;
    if a hand hides the handle, appearance provides no polarity evidence.
    """
    count, alternatives = candidates.shape[:2]
    contrast = np.zeros((count, alternatives))
    valid = np.zeros((count, alternatives), dtype=bool)
    video = cv2.VideoCapture(str(inputs / 'video.mp4'))
    with h5py.File(inputs / 'masks.h5', 'r') as masks:
        for frame in range(count):
            ok, image = video.read()
            if not ok:
                raise ValueError(f'Video ended before frame {frame}')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            interior = cv2.erode(masks['object'][frame].astype(np.uint8), np.ones((3, 3), np.uint8))
            y, x = np.where(interior)
            if len(x) < 40:
                continue
            values = gray[y, x].astype(float)
            ray = np.column_stack(((x-K[0, 2])/K[0, 0], (y-K[1, 2])/K[1, 1], np.ones(len(x))))
            ray /= np.linalg.norm(ray, axis=1, keepdims=True)
            scale = max(np.percentile(values, 90)-np.percentile(values, 10), 30)
            for alternative in range(alternatives):
                pose = candidates[frame, alternative]
                axis, translation = pose[:3, 2], pose[:3, 3]
                axis_perpendicular = axis - ray * (ray @ axis)[:, None]
                translation_perpendicular = translation - ray * (ray @ translation)[:, None]
                z = -(translation_perpendicular * axis_perpendicular).sum(axis=1) / np.maximum((axis_perpendicular**2).sum(axis=1), 1e-12)
                barrel = (z < .02) & (z > -.22)
                handle = (z > .16) & (z < .34)
                if barrel.sum() > 20 and handle.sum() > 20:
                    contrast[frame, alternative] = (np.median(values[barrel])-np.median(values[handle])) / scale
                    valid[frame, alternative] = True
    video.release()
    np.savez(output_dir / 'appearance_diagnostics.npz', contrast=contrast, valid=valid)
    return contrast, valid


def choose(baseline_world, baseline, foundation, world_from_camera, candidates,
           errors, original_errors, contrast, appearance_valid, output_dir):
    frames = np.arange(len(baseline))
    original_alpha = np.arcsin(np.clip(baseline[:, 2, 2], -1, 1))
    alpha = np.arcsin(np.clip(candidates[:, :, 2, 2], -1, 1))
    axis_change = np.arccos(np.clip((baseline[:, None, :3, 2]*candidates[:, :, :3, 2]).sum(axis=-1), -1, 1))
    shift = np.linalg.norm(candidates[:, :, :3, 3]-baseline[:, None, :3, 3], axis=-1)
    photo_penalty = np.where(appearance_valid, np.maximum(0, .1-contrast), 0)
    cost = errors + .12*(alpha-original_alpha[:, None])**2 + .02*(axis_change/np.pi)**2 + .2*photo_penalty
    choice = cost.argmin(axis=1)
    selected = candidates[frames, choice].copy()
    selected_error = errors[frames, choice]
    tilt_change = np.abs(alpha[frames, choice]-original_alpha)
    selected_contrast = contrast[frames, choice]
    selected_valid = appearance_valid[frames, choice]
    changed = ((original_errors > .34) & (original_errors-selected_error > .14)
               & (selected_error < .38) & (selected_error < .72*original_errors)
               & (shift[frames, choice] < .11)
               & ~(selected_valid & (selected_contrast < -.15)))
    fp_alpha = np.arcsin(np.clip(foundation[:, 2, 2], -1, 1))
    fp_xy = foundation[:, :2, 2]
    fitted_xy = selected[:, :2, 2]
    fp_cosine = (fp_xy*fitted_xy).sum(axis=1) / np.maximum(np.linalg.norm(fp_xy, axis=1)*np.linalg.norm(fitted_xy, axis=1), 1e-9)
    # A large tilt needs strong silhouette evidence and agreement with a
    # separately estimated FoundationPose axis. Otherwise retain the baseline.
    large_tilt_supported = ((original_errors-selected_error > .4) & (selected_error < .30)
                            & (np.abs(alpha[frames, choice]-fp_alpha) < np.deg2rad(25))
                            & (fp_cosine > np.cos(np.deg2rad(30))))
    changed &= (tilt_change < np.deg2rad(25)) | large_tilt_supported
    # Explicit visual decisions: in frames 0..112 and 690..802 the resting
    # bat's baseline handle/barrel ends and projected axis already match the
    # video. Preserve those entire intervals rather than fit mask thickness.
    changed[:113] = False
    changed[690:] = False
    result = baseline_world.astype(np.float64).copy()
    for frame in np.flatnonzero(changed):
        pose = selected[frame]
        pose[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
        # Reimpose the hard camera-centre distance constraint in float64.
        pose[:3, 3] *= np.linalg.norm(baseline[frame, :3, 3]) / np.linalg.norm(pose[:3, 3])
        result[frame] = world_from_camera @ pose
    np.savez(output_dir / 'selection_diagnostics.npz', changed=changed, candidate=choice,
             base_loss=original_errors, selected_loss=selected_error, tilt_change=tilt_change,
             axis_change=axis_change[frames, choice], shift=shift[frames, choice],
             contrast=selected_contrast, contrast_valid=selected_valid,
             large_tilt_supported=large_tilt_supported)
    return result, changed, selected_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    if sha256(args.inputs / 'object_mesh.glb') != SOURCE_MESH_SHA256:
        raise ValueError('The stored rendering proxy belongs to a different input mesh')
    if sha256(RENDER_MESH) != RENDER_MESH_SHA256:
        raise ValueError('Rendering proxy checksum mismatch')
    reconstruction = np.load(args.inputs / 'reconstruction.npz')
    baseline_world = reconstruction['T_world_mesh'].astype(np.float64)
    foundation = reconstruction['T_cam_mesh_foundationpose'].astype(np.float64)
    camera = json.loads((args.inputs / 'camera.json').read_text())
    K, world_from_camera = np.array(camera['K']), np.array(camera['T_world_cam'])
    baseline = np.linalg.inv(world_from_camera) @ baseline_world
    candidates, errors, original_errors, mask_seconds, gpu_seconds = fit_candidates(
        args.inputs, baseline, foundation, K, args.out.parent)
    contrast, valid = appearance(args.inputs, candidates, K, args.out.parent)
    result, changed, selected_error = choose(baseline_world, baseline, foundation,
        world_from_camera, candidates, errors, original_errors, contrast, valid, args.out.parent)
    assert result.shape == (len(baseline_world), 4, 4) and result.dtype == np.float64
    assert np.isfinite(result).all()
    assert np.array_equal(result[:, 3, :], np.tile([0., 0., 0., 1.], (len(result), 1)))
    assert np.array_equal(result[~changed], baseline_world[~changed])
    camera_result = np.linalg.inv(world_from_camera) @ result
    distance_error = np.max(np.abs(np.linalg.norm(camera_result[:, :3, 3], axis=1)
                                  - np.linalg.norm(baseline[:, :3, 3], axis=1)))
    assert distance_error < 1e-10
    np.savez(args.out, T_world_mesh=result)
    statistics = dict(frames=len(result), changed_frames=int(changed.sum()),
        unchanged_frames=int((~changed).sum()), dtype=str(result.dtype),
        baseline_mean_visible_silhouette_error=float(original_errors.mean()),
        output_mean_visible_silhouette_error=float(np.where(changed, selected_error, original_errors).mean()),
        baseline_error_on_changed=float(original_errors[changed].mean()),
        output_error_on_changed=float(selected_error[changed].mean()),
        max_camera_distance_change_metres=float(distance_error),
        mask_preparation_seconds=mask_seconds, gpu_fit_wall_seconds=gpu_seconds,
        total_seconds=time.perf_counter()-start)
    (args.out.parent / 'validation.json').write_text(json.dumps(statistics, indent=2)+'\n')
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == '__main__':
    main()
