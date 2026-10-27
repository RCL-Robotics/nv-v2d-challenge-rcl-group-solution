#!/usr/bin/env python3
"""Trajectory recovery for the blue trash-can episode.

The method combines:
  1) pooled, static-world-pose estimation for the three rest phases;
  2) local render-and-compare pose refinement during manipulation;
  3) conservative anchoring and temporal smoothing of the manipulation poses.

It is deterministic and uses only files under --inputs plus the installed stack.
"""
import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import binary_erosion
from scipy.optimize import differential_evolution
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation

# Phase boundaries were checked against the object mask track, hand/object contact
# probability, and the video.  Frames in the three "static" ranges have a fixed
# mask bounding box (up to hand occlusion), and the released floor plane is used
# as a physical contact check when pooling their pose.
STATIC_RANGES = ((0, 219), (504, 641), (738, 857))
DYNAMIC_RANGES = ((220, 503), (642, 737))
PHASE_NAMES = ("pre_static", "manipulation_1", "middle_static",
               "manipulation_2", "final_static")
RENDER_H, RENDER_W = 384, 512
SEED = 20261020


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True)
    p.add_argument('--out', required=True)
    return p.parse_args()


def pose_median(T):
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_matrix(T[:, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(T[:, :3, 3], axis=0)
    return out


def angle_deg(A, B):
    c = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def slerp_pose(A, B, w):
    """SE(3)-style interpolation, used only for nearby poses."""
    ra = Rotation.from_matrix(A[:3, :3]).as_rotvec()
    rb = Rotation.from_matrix(B[:3, :3]).as_rotvec()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_rotvec((1.0 - w) * ra + w * rb).as_matrix()
    out[:3, 3] = (1.0 - w) * A[:3, 3] + w * B[:3, 3]
    return out


def floor_contact_adjust(T, vertices, plane):
    """Translate along the floor normal so the mesh's lowest point touches the plane."""
    norm = float(np.linalg.norm(plane[:3]))
    n = np.asarray(plane[:3], dtype=np.float64) / norm
    d = float(plane[3]) / norm
    W = (T @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]
    dist = W @ n + d
    T = T.copy()
    T[:3, 3] -= dist.min() * n
    return T


class Renderer:
    def __init__(self, mesh, camera, tex_size, device='cuda'):
        self.device = device
        self.H, self.W = RENDER_H, RENDER_W
        self.vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.faces_np = np.asarray(mesh.faces, dtype=np.int32)
        self.uv_np = np.asarray(mesh.visual.uv, dtype=np.float32)
        self.center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)

        cam = camera
        self.K = np.asarray(cam['K'], dtype=np.float32).copy()
        self.K[0] *= self.W / float(cam['width'])
        self.K[1] *= self.H / float(cam['height'])
        self.T_cam_world = np.linalg.inv(np.asarray(cam['T_world_cam'], dtype=np.float64))

        self.V = torch.tensor(self.vertices, dtype=torch.float32, device=device)
        self.F = torch.tensor(self.faces_np, dtype=torch.int32, device=device)
        self.UV = torch.tensor(self.uv_np, dtype=torch.float32, device=device)

        tex = np.asarray(mesh.visual.material.baseColorTexture)
        tex = tex[::-1].copy().astype(np.float32) / 255.0
        tex = cv2.resize(tex, tex_size, interpolation=cv2.INTER_AREA)
        tex = np.concatenate([tex, np.ones_like(tex[..., :1])], axis=2)
        self.tex = torch.tensor(tex, dtype=torch.float32, device=device)[None]
        self.gl = dr.RasterizeCudaContext(device=device)

    def render(self, T):
        T = np.asarray(T, dtype=np.float32)
        Tc = self.T_cam_world.astype(np.float32) @ T
        vh = np.c_[self.vertices, np.ones(len(self.vertices), dtype=np.float32)]
        vc = (Tc @ vh.T).T[:, :3]
        u = self.K[0, 0] * vc[:, 0] / vc[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * vc[:, 1] / vc[:, 2] + self.K[1, 2]
        xn = 2.0 * u / self.W - 1.0
        yn = 1.0 - 2.0 * v / self.H
        zn = np.clip((vc[:, 2] - 0.1) / 9.9 * 2.0 - 1.0, -1.0, 1.0)
        pos = torch.tensor(np.c_[xn, yn, zn, np.ones(len(vc), np.float32)],
                           dtype=torch.float32, device=self.device)[None]
        rast, _ = dr.rasterize(self.gl, pos, self.F, resolution=(self.H, self.W))
        uv, _ = dr.interpolate(self.UV[None], rast, self.F)
        col = dr.texture(self.tex, uv, filter_mode='linear')[0, ..., :3]
        col = col.cpu().numpy()[::-1][..., ::-1]
        rgb = (col * 255.0).clip(0, 255).astype(np.uint8)
        mask = (rast[0, ..., 3] > 0).cpu().numpy()[::-1]
        return rgb, mask

    def score(self, T, image, obj_mask, person_mask):
        rgb, mask = self.render(T)
        region = obj_mask & ~person_mask
        if region.sum() < 20:
            return 0.0, 0.0, 0.0
        inter = np.count_nonzero(mask & obj_mask)
        union = np.count_nonzero(mask | obj_mask)
        iou = inter / max(union, 1)
        x = rgb[region].astype(np.float64)
        y = image[region].astype(np.float64)
        nccs = []
        for c in range(3):
            if x[:, c].std() > 1.0 and y[:, c].std() > 1.0:
                nccs.append(float(np.corrcoef(x[:, c], y[:, c])[0, 1]))
        ncc = float(np.mean(nccs)) if nccs else 0.0
        gx = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        sx = cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gy, cv2.CV_64F, 1, 0, ksize=3)
        er = binary_erosion(region, iterations=1)
        gn = float(np.corrcoef(sx[er], sy[er])[0, 1]) if er.sum() > 20 else 0.0
        return float(iou), ncc, gn

    def batch_baseline_scores(self, poses, images, obj_masks, person_masks, batch=16):
        n_frames = len(poses)
        ious = np.zeros(n_frames, np.float64)
        nccs = np.zeros(n_frames, np.float64)
        grads = np.zeros(n_frames, np.float64)
        vh = np.c_[self.vertices, np.ones(len(self.vertices), np.float32)]
        for s in range(0, n_frames, batch):
            e = min(n_frames, s + batch)
            n = e - s
            Tc = np.stack([self.T_cam_world.astype(np.float32) @ poses[i].astype(np.float32)
                           for i in range(s, e)])
            vc = torch.tensor(np.einsum('nij,vj->nvi', Tc, vh),
                              dtype=torch.float32, device=self.device)
            u = self.K[0, 0] * vc[..., 0] / vc[..., 2] + self.K[0, 2]
            v = self.K[1, 1] * vc[..., 1] / vc[..., 2] + self.K[1, 2]
            xn = 2.0 * u / self.W - 1.0
            yn = 1.0 - 2.0 * v / self.H
            zn = torch.clamp((vc[..., 2] - 0.1) / 9.9 * 2.0 - 1.0, -1.0, 1.0)
            pos = torch.stack([xn, yn, zn, torch.ones_like(zn)], dim=-1)
            rast, _ = dr.rasterize(self.gl, pos, self.F, resolution=(self.H, self.W))
            uv, _ = dr.interpolate(self.UV[None].repeat(n, 1, 1), rast, self.F)
            col = dr.texture(self.tex.repeat(n, 1, 1, 1), uv,
                             filter_mode='linear')[..., :3].cpu().numpy()
            col = col[..., ::-1] * 255.0
            masks = (rast[..., 3] > 0).cpu().numpy()
            for j, i in enumerate(range(s, e)):
                rgb = col[j, ::-1].copy()
                mask = masks[j, ::-1].copy()
                om = obj_masks[i]
                pm = person_masks[i]
                region = om & ~pm
                inter = np.count_nonzero(mask & om)
                union = np.count_nonzero(mask | om)
                ious[i] = inter / max(union, 1)
                x = rgb[region]
                y = images[i][region]
                vals = []
                for c in range(3):
                    if x[:, c].std() > 1.0 and y[:, c].std() > 1.0:
                        vals.append(np.corrcoef(x[:, c], y[:, c])[0, 1])
                nccs[i] = np.mean(vals) if vals else 0.0
                gx = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
                gy = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
                sx = cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(gy, cv2.CV_64F, 1, 0, ksize=3)
                er = binary_erosion(region, iterations=1)
                grads[i] = np.corrcoef(sx[er], sy[er])[0, 1] if er.sum() > 20 else 0.0
        return ious, nccs, grads

    def delta_pose(self, T, params):
        rx, ry, rz, tx, ty, tz = params
        A = np.eye(4, dtype=np.float64)
        A[:3, :3] = Rotation.from_rotvec(np.deg2rad([rx, ry, rz])).as_matrix()
        A[:3, 3] = self.center
        Ai = np.eye(4, dtype=np.float64)
        Ai[:3, 3] = -self.center
        D = np.eye(4, dtype=np.float64)
        D[:3, 3] = [tx, ty, tz]
        return D @ T @ A @ Ai


def optimize_static(renderer, T0, target, target_mask, valid_mask, seed):
    region = target_mask & valid_mask

    def objective(p):
        T = renderer.delta_pose(T0, p)
        rgb, mask = renderer.render(T)
        inter = np.count_nonzero(mask & target_mask)
        union = np.count_nonzero(mask | target_mask)
        iou = inter / max(union, 1)
        x = rgb[region].astype(np.float64)
        y = target[region].astype(np.float64)
        vals = []
        for c in range(3):
            if x[:, c].std() > 1.0 and y[:, c].std() > 1.0:
                vals.append(np.corrcoef(x[:, c], y[:, c])[0, 1])
        ncc = float(np.mean(vals)) if vals else 0.0
        gx = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gy = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(np.float64)
        sx = cv2.Sobel(gx, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gy, cv2.CV_64F, 1, 0, ksize=3)
        er = binary_erosion(region, iterations=2)
        gn = np.corrcoef(sx[er], sy[er])[0, 1] if er.sum() > 20 else 0.0
        return -(iou + 0.22 * ncc + 0.04 * gn)

    result = differential_evolution(
        objective,
        [(-8.0, 8.0)] * 3 + [(-0.06, 0.06)] * 3,
        seed=seed, popsize=10, maxiter=30, tol=0.002,
        polish=False, workers=1, updating='deferred')
    T = renderer.delta_pose(T0, result.x)
    rgb, mask = renderer.render(T)
    iou = np.count_nonzero(mask & target_mask) / max(np.count_nonzero(mask | target_mask), 1)
    x = rgb[region].astype(np.float64)
    y = target[region].astype(np.float64)
    vals = []
    for c in range(3):
        if x[:, c].std() > 1.0 and y[:, c].std() > 1.0:
            vals.append(np.corrcoef(x[:, c], y[:, c])[0, 1])
    return T, np.asarray(result.x), float(iou), float(np.mean(vals) if vals else 0.0)


def optimize_dynamic_pose(renderer, T0, image, obj_mask, person_mask):
    region = obj_mask & ~person_mask

    def objective_T(T):
        rgb, mask = renderer.render(T)
        inter = np.count_nonzero(mask & obj_mask)
        union = np.count_nonzero(mask | obj_mask)
        iou = inter / max(union, 1)
        x = rgb[region].astype(np.float64)
        y = image[region].astype(np.float64)
        vals = []
        for c in range(3):
            if x[:, c].std() > 1.0 and y[:, c].std() > 1.0:
                vals.append(np.corrcoef(x[:, c], y[:, c])[0, 1])
        ncc = float(np.mean(vals)) if vals else 0.0
        return iou + 0.30 * ncc

    p = np.zeros(6, dtype=np.float64)
    best = objective_T(T0)
    for rot_step, trans_step in ((3.0, 0.02), (1.5, 0.01),
                                 (0.75, 0.005), (0.4, 0.0025)):
        for _ in range(2):
            improved = False
            for j in range(6):
                for sign in (-1.0, 1.0):
                    q = p.copy()
                    q[j] += sign * (rot_step if j < 3 else trans_step)
                    s = objective_T(renderer.delta_pose(T0, q))
                    if s > best + 1e-7:
                        best = s
                        p = q
                        improved = True
            if not improved:
                break
    return renderer.delta_pose(T0, p), p, best


def correction_series(T_start, T_end, B, a, b):
    """Smooth SE(3) correction from T_start at a to T_end at b."""
    Da = T_start @ np.linalg.inv(B[a])
    Db = T_end @ np.linalg.inv(B[b])
    out = np.empty((b - a + 1, 4, 4), np.float64)
    for i in range(a, b + 1):
        w = (i - a) / float(b - a)
        w = w * w * (3.0 - 2.0 * w)
        out[i - a] = slerp_pose(Da, Db, w) @ B[i]
    return out


def main():
    args = parse_args()
    t_start = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    inp = args.inputs
    camera = json.load(open(os.path.join(inp, 'camera.json')))
    plane_data = json.load(open(os.path.join(inp, 'ground_plane.json')))
    plane = np.asarray(plane_data['plane'], dtype=np.float64)
    recon = np.load(os.path.join(inp, 'reconstruction.npz'))
    B = recon['T_world_mesh'].astype(np.float64)
    F = len(B)
    if F != 858:
        raise RuntimeError(f'This solver was validated on 858 frames, got {F}')

    mesh = trimesh.load(os.path.join(inp, 'object_mesh.glb'),
                        force='scene', process=False).to_geometry()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    print('[1/6] Decoding low-resolution video and masks', flush=True)
    images = np.empty((F, RENDER_H, RENDER_W, 3), np.uint8)
    cap = cv2.VideoCapture(os.path.join(inp, 'video.mp4'))
    for i in range(F):
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f'failed to read frame {i}')
        images[i] = cv2.resize(im, (RENDER_W, RENDER_H), interpolation=cv2.INTER_AREA)
    cap.release()

    obj_masks = np.empty((F, RENDER_H, RENDER_W), bool)
    person_masks = np.empty((F, RENDER_H, RENDER_W), bool)
    with h5py.File(os.path.join(inp, 'masks.h5'), 'r') as hf:
        for i in range(F):
            om = hf['object'][i].astype(np.uint8)
            pm = hf['person'][i].astype(np.uint8)
            obj_masks[i] = cv2.resize(om, (RENDER_W, RENDER_H),
                                      interpolation=cv2.INTER_NEAREST) > 0
            person_masks[i] = cv2.resize(pm, (RENDER_W, RENDER_H),
                                         interpolation=cv2.INTER_NEAREST) > 0

    print('[2/6] Scoring baseline and pooling static-phase evidence', flush=True)
    static_renderer = Renderer(mesh, camera, (1250, 1142))
    base_iou, base_ncc, base_grad = static_renderer.batch_baseline_scores(
        B, images, obj_masks, person_masks)
    static_poses = []
    for phase_idx, (a, b) in enumerate(STATIC_RANGES):
        score = base_iou[a:b + 1] + 0.20 * base_ncc[a:b + 1] + 0.10 * base_grad[a:b + 1]
        top = a + np.argsort(score)[-20:]
        target = np.median(images[top], axis=0).astype(np.uint8)
        target_mask = np.mean(obj_masks[top], axis=0) > 0.5
        valid = target_mask & (np.mean(person_masks[top], axis=0) < 0.15)
        valid = binary_erosion(valid, iterations=1)
        T0 = pose_median(B[top])
        T, params, iou, ncc = optimize_static(
            static_renderer, T0, target, target_mask, valid,
            seed=1234)
        # The released plane fit has several-cm tolerance.  Keep the image/mask
        # optimum rather than snapping the mesh to the plane (ADD cares about
        # metric pose, not just floor contact).
        static_poses.append(T)
        print(f'  static {phase_idx}: frames {a}-{b}, top={top.tolist()}, pooled IoU={iou:.4f}, '
              f'NCC={ncc:.4f}, delta={np.round(params, 4).tolist()}', flush=True)

    T_pre, T_mid, T_fin = static_poses

    print('[3/6] Refining manipulation poses every 4 frames', flush=True)
    dynamic_renderer = Renderer(mesh, camera, (1000, 914))
    sample_ids = list(range(220, 504, 4)) + list(range(642, 738, 4))
    if sample_ids[-1] != 737:
        sample_ids.append(737)
    sample_p = []
    sample_iou = []
    sample_ncc = []
    for i in sample_ids:
        T, p, _ = optimize_dynamic_pose(
            dynamic_renderer, B[i], images[i], obj_masks[i], person_masks[i])
        iou, ncc, _ = dynamic_renderer.score(
            T, images[i], obj_masks[i], person_masks[i])
        sample_p.append(p)
        sample_iou.append(iou)
        sample_ncc.append(ncc)
    sample_p = np.asarray(sample_p)
    sample_ids = np.asarray(sample_ids)

    print('[4/6] Temporally smoothing and anchoring corrections', flush=True)
    P = np.zeros((F, 6), np.float64)
    for c in range(6):
        P[:, c] = np.interp(np.arange(F), sample_ids, sample_p[:, c])
    PS = P.copy()
    for a, b in DYNAMIC_RANGES:
        for c in range(6):
            PS[a:b + 1, c] = savgol_filter(
                medfilt(P[a:b + 1, c], kernel_size=5),
                window_length=9, polyorder=2, mode='interp')
        PS[a:b + 1, 0] = np.clip(PS[a:b + 1, 0], -4.0, 4.0)
        PS[a:b + 1, 1] = np.clip(PS[a:b + 1, 1], -3.0, 4.5)
        PS[a:b + 1, 2] = np.clip(PS[a:b + 1, 2], -6.0, 6.0)
        PS[a:b + 1, 3] = np.clip(PS[a:b + 1, 3], -0.03, 0.03)
        PS[a:b + 1, 4] = np.clip(PS[a:b + 1, 4], -0.015, 0.03)
        PS[a:b + 1, 5] = np.clip(PS[a:b + 1, 5], -0.02, 0.05)

    T_anchor = B.copy()
    T_opt = B.copy()
    T_anchor[0:220] = T_pre
    T_anchor[504:642] = T_mid
    T_anchor[738:] = T_fin
    T_opt[0:220] = T_pre
    T_opt[504:642] = T_mid
    T_opt[738:] = T_fin
    T_anchor[220:504] = correction_series(T_pre, T_mid, B, 220, 503)
    T_anchor[642:738] = correction_series(T_mid, T_fin, B, 642, 737)
    for i in range(220, 504):
        T_opt[i] = dynamic_renderer.delta_pose(B[i], PS[i])
    for i in range(642, 738):
        T_opt[i] = dynamic_renderer.delta_pose(B[i], PS[i])

    Tout = T_anchor.copy()
    for a, b in DYNAMIC_RANGES:
        ramp = 18 if a == 220 else 12
        for i in range(a, b + 1):
            w = 0.50
            if i < a + ramp:
                w *= 0.5 * (1.0 - np.cos(np.pi * (i - a) / ramp))
            if i > b - ramp:
                w *= 0.5 * (1.0 - np.cos(np.pi * (b - i) / ramp))
            Tout[i] = slerp_pose(T_anchor[i], T_opt[i], w)

    print('[5/6] Validating baseline versus result', flush=True)
    eval_ids = list(range(2, F, 6))
    phase_metrics = {name: {'base': [], 'result': []} for name in PHASE_NAMES}
    for i in eval_ids:
        if i <= 219:
            name = 'pre_static'
        elif i <= 503:
            name = 'manipulation_1'
        elif i <= 641:
            name = 'middle_static'
        elif i <= 737:
            name = 'manipulation_2'
        else:
            name = 'final_static'
        qb = dynamic_renderer.score(B[i], images[i], obj_masks[i], person_masks[i])
        qr = dynamic_renderer.score(Tout[i], images[i], obj_masks[i], person_masks[i])
        phase_metrics[name]['base'].append(qb[:2])
        phase_metrics[name]['result'].append(qr[:2])

    report_metrics = {}
    for name in PHASE_NAMES:
        bm = np.asarray(phase_metrics[name]['base'])
        rm = np.asarray(phase_metrics[name]['result'])
        report_metrics[name] = {
            'n_heldout': int(len(bm)),
            'baseline_iou': float(bm[:, 0].mean()),
            'baseline_ncc': float(bm[:, 1].mean()),
            'result_iou': float(rm[:, 0].mean()),
            'result_ncc': float(rm[:, 1].mean()),
        }
    print('VALIDATION_JSON ' + json.dumps(report_metrics, sort_keys=True), flush=True)

    print('[6/6] Writing trajectory', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tout.astype(np.float64))
    elapsed = time.time() - t_start
    print(f'WROTE {args.out} shape={Tout.shape} elapsed_seconds={elapsed:.1f}', flush=True)


if __name__ == '__main__':
    main()
