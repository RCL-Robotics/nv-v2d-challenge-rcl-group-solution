#!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF world trajectory for the bar stool.

The released per-frame reconstruction is a useful, but noisy, observation.  This
solver keeps it during manipulation and replaces the two genuinely static phases
(and mildly smooths the manipulated phase) using pooled silhouette evidence.

Usage:
    python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

All phase boundaries below were verified from the video and masks:
  * 0..124     stool is on the floor, untouched/held but not moving
  * 125..591   person drags the stool; heavy hand/body occlusion near 420..560
  * 592..677   stool has been released and is static
The boundaries are deliberately fixed constants so this script is offline and
does not rely on any result produced outside the solver.
"""
import argparse
import json
import os
import random
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


INIT_END = 125       # first moving frame is 125 (verified in frames/masks)
FINAL_START = 592    # stool is stationary from here to the end
BLEND = 10
SEED = 0

# Coarse-to-fine silhouette fitting. Each stage uses a single shared pose.
INIT_STAGES = (
    # (frame stride/selection, resolution divisor, iterations, learning rate, reg)
    (6, 4, 200, 3.0e-4, 1.0e-3),   # 21 frames at 384x288
    (8, 2, 100, 2.0e-4, 1.0e-4),   # 16 frames at 768x576
    (1, 1, 40, 3.0e-5, 0.0),       # all 125 frames at full resolution
)
FINAL_STAGES = (
    (3, 4, 200, 3.0e-4, 1.0e-3),   # 29 frames at 384x288
    (6, 2, 100, 5.0e-5, 1.0e-4),   # 15 frames at 768x576
    (1, 1, 40, 3.0e-5, 0.0),       # all 86 frames at full resolution
)


def set_determinism():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cv2.setNumThreads(1)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def rodrigues(w):
    """Batched/small-angle-safe Rodrigues formula for a 3-vector."""
    theta = torch.linalg.norm(w) + 1e-12
    a = w / theta
    Kx = torch.zeros((3, 3), dtype=w.dtype, device=w.device)
    Kx[0, 1] = -a[2]
    Kx[0, 2] = a[1]
    Kx[1, 0] = a[2]
    Kx[1, 2] = -a[0]
    Kx[2, 0] = -a[1]
    Kx[2, 1] = a[0]
    return (torch.eye(3, dtype=w.dtype, device=w.device)
            + torch.sin(theta) * Kx
            + (1.0 - torch.cos(theta)) * (Kx @ Kx))


def apply_delta(base, center, dt, dw):
    """Apply a world-space rotation about the mesh center plus translation."""
    base = torch.as_tensor(base, dtype=torch.float32, device=dt.device)
    center = torch.as_tensor(center, dtype=torch.float32, device=dt.device)
    Rd = rodrigues(dw)
    cw = base[:3, :3] @ center + base[:3, 3]
    R = Rd @ base[:3, :3]
    t = Rd @ (base[:3, 3] - cw) + cw + dt
    out = torch.eye(4, dtype=torch.float32, device=dt.device)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def load_visible_masks(mask_path, ids, scale):
    """Object masks with the person (and a small anti-aliasing fringe) removed."""
    out = []
    with h5py.File(mask_path, "r") as f:
        for start in range(0, len(ids), 16):
            chunk = ids[start:start + 16]
            obj = f["object"][chunk]
            per = f["person"][chunk]
            for j in range(len(chunk)):
                p = cv2.dilate(per[j].astype(np.uint8),
                               np.ones((7, 7), np.uint8)).astype(bool)
                vis = obj[j] & ~p
                if scale != 1:
                    vis = cv2.resize(
                        vis.astype(np.uint8),
                        (1536 // scale, 1152 // scale),
                        interpolation=cv2.INTER_AREA,
                    ) > 0.5
                out.append(vis.astype(np.uint8))
    return np.asarray(out, dtype=np.uint8)


def fit_shared_pose(base_pose, ids, stages, vertices, faces, K, T_world_cam,
                    masks, center, log_prefix):
    """Fit one constant mesh pose to all selected frames.

    The objective is soft silhouette Dice after removing person pixels. A soft
    silhouette is obtained by interpolating a constant map over the rasterizer
    barycentrics and running nvdiffrast antialiasing, so the pose receives
    gradients at image-space silhouette edges.
    """
    device = torch.device("cuda")
    V = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    F = torch.as_tensor(faces, dtype=torch.int32, device=device)
    ones = torch.ones((1, len(vertices), 1), dtype=torch.float32, device=device)
    Twc_inv = np.linalg.inv(T_world_cam)
    pose = np.asarray(base_pose, dtype=np.float32).copy()
    n_frames = len(ids)

    for stride, scale, steps, lr, reg_weight in stages:
        if stride == 1:
            stage_ids = np.arange(ids[0], ids[-1] + 1, dtype=np.int64)
        else:
            stage_ids = np.asarray(ids[::stride], dtype=np.int64)
        W, H = 1536 // scale, 1152 // scale
        K_stage = np.asarray(K, dtype=np.float32).copy()
        K_stage[:2, :] /= scale
        K_stage[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        Kt = torch.as_tensor(K_stage, dtype=torch.float32, device=device)
        Twci = torch.as_tensor(Twc_inv, dtype=torch.float32, device=device)
        glctx = dr.RasterizeCudaContext()

        gt_np = load_visible_masks(os.path.join(args.inputs, 'masks.h5'), stage_ids.tolist(), scale)
        base_t = torch.as_tensor(pose, dtype=torch.float32, device=device)
        dt = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
        dw = torch.zeros(3, dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam([dt, dw], lr=lr)
        batch_size = 8
        n_batches = (len(stage_ids) + batch_size - 1) // batch_size

        def render_soft(current_pose, B):
            T = current_pose[None].repeat(B, 1, 1)
            M = Twci[None] @ T
            v = torch.einsum("bij,bvj->bvi", M[:, :3, :3], V[None]) + M[:, None, :3, 3]
            z = v[..., 2].clamp_min(1e-5)
            u = Kt[0, 0] * v[..., 0] / z + Kt[0, 2]
            vv = Kt[1, 1] * v[..., 1] / z + Kt[1, 2]
            clip = torch.stack(
                [2.0 * u / W - 1.0, 2.0 * vv / H - 1.0,
                 z / 10.0, torch.ones_like(z)], dim=-1)
            rast, _ = dr.rasterize(glctx, clip, F, resolution=[H, W])
            col = dr.interpolate(ones.repeat(B, 1, 1), rast, F)[0]
            return dr.antialias(col, rast, clip, F)[..., 0]

        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for start in range(0, len(stage_ids), batch_size):
                gt = torch.as_tensor(
                    gt_np[start:start + batch_size], dtype=torch.float32,
                    device=device)
                soft = render_soft(apply_delta(base_t, center, dt, dw),
                                   len(gt_np[start:start + batch_size]))
                inter = (soft * gt).sum(dim=(1, 2))
                denom = soft.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
                loss = (1.0 - 2.0 * inter / (denom + 1e-6)).mean()
                (loss / n_batches).backward()
                total += loss.item()
            if reg_weight:
                reg = reg_weight * (dt.square().mean() + dw.square().mean())
                reg.backward()
            opt.step()
            if it in (0, steps - 1) or it % 40 == 39:
                print(f"[{log_prefix}] scale={scale} it={it + 1}/{steps} "
                      f"dice={1.0 - total / n_batches:.5f} "
                      f"dt={dt.detach().cpu().numpy().round(5).tolist()} "
                      f"dw_deg={np.degrees(dw.detach().cpu().numpy()).round(3).tolist()}")
        with torch.no_grad():
            pose = apply_delta(base_t, center, dt, dw).detach().cpu().numpy()
        del Kt, Twci, glctx, gt_np, base_t, dt, dw, opt
        torch.cuda.empty_cache()
    return pose


def smooth_manipulation(base, start, end):
    """Causal-free local linear smoothing in rotation/translation, blended at ends.

    A 5-frame window is intentionally weak: the per-frame tracker already follows
    the visible object well, while this removes one-frame jitter and softens the
    most occluded transitions without inventing a long constant-velocity path.
    """
    out = base.copy()
    seg = base[start:end].copy()
    q = Rotation.from_matrix(seg[:, :3, :3]).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] *= -1.0
    q_s = np.empty_like(q)
    for j in range(4):
        q_s[:, j] = savgol_filter(q[:, j], 5, 1, mode="interp")
    q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    R_s = Rotation.from_quat(q_s).as_matrix()
    p_s = savgol_filter(seg[:, :3, 3], 5, 1, axis=0, mode="interp")

    blend = np.ones(len(seg), dtype=np.float32)
    nb = min(BLEND, len(seg) // 2)
    blend[:nb] = np.linspace(0.0, 1.0, nb, endpoint=False)
    blend[-nb:] = np.linspace(1.0, 0.0, nb, endpoint=False)

    # Blend rotations in matrix space, then project back to SO(3).
    R = seg[:, :3, :3] * (1.0 - blend[:, None, None]) + R_s * blend[:, None, None]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    p = seg[:, :3, 3] * (1.0 - blend[:, None]) + p_s * blend[:, None]
    out[start:end, :3, :3] = R
    out[start:end, :3, 3] = p
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args_cli = parser.parse_args()
    global args
    args = args_cli

    t_start = time.time()
    set_determinism()
    if not torch.cuda.is_available():
        raise RuntimeError("nvdiffrast CUDA rasterizer is required")

    inp = args.inputs
    cam = json.load(open(os.path.join(inp, "camera.json"), "r"))
    K = np.asarray(cam["K"], dtype=np.float32)
    T_world_cam = np.asarray(cam["T_world_cam"], dtype=np.float32)
    recon_path = os.path.join(inp, "reconstruction.npz")
    recon = np.load(recon_path)
    T_base = np.asarray(recon["T_world_mesh"], dtype=np.float32)
    mesh = trimesh.load(os.path.join(inp, "object_mesh.glb"),
                        force="scene", process=False).to_geometry()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    center = ((mesh.bounds[0] + mesh.bounds[1]) * 0.5).astype(np.float32)

    print(f"Loaded {len(T_base)} frames, {len(vertices)} vertices, {len(faces)} faces")
    init_ids = np.arange(0, INIT_END, dtype=np.int64)
    final_ids = np.arange(FINAL_START, len(T_base), dtype=np.int64)

    init_pose = fit_shared_pose(
        T_base[0], init_ids, INIT_STAGES, vertices, faces, K, T_world_cam,
        os.path.join(inp, "masks.h5"), center, "initial-static")
    torch.cuda.empty_cache()
    final_pose = fit_shared_pose(
        T_base[FINAL_START], final_ids, FINAL_STAGES, vertices, faces, K,
        T_world_cam, os.path.join(inp, "masks.h5"), center, "final-static")
    torch.cuda.empty_cache()

    result = T_base.astype(np.float64).copy()
    result[:INIT_END] = init_pose.astype(np.float64)
    result[FINAL_START:] = final_pose.astype(np.float64)
    result = smooth_manipulation(result.astype(np.float32),
                                 INIT_END, FINAL_START).astype(np.float64)

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, T_world_mesh=result)
    print(f"Saved {out_path} with T_world_mesh {result.shape} {result.dtype}")
    print(f"Initial static pose:\n{init_pose}")
    print(f"Final static pose:\n{final_pose}")
    print(f"Runtime: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
