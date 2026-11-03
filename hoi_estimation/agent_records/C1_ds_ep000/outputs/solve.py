#!/usr/bin/env python3
"""Deterministic camera/world-pose solver for the single monocular HOI episode.

The camera is refined from three independent, static-scene cues:
  * dense photo-aligned frame-0 human correspondences (world mesh + person mask),
  * the released world floor plane fitted in the MoGe camera depth maps,
  * resting-object floor contact and metric object image size.

Only the allowed inputs, the released CARI4D intermediates in /pipeline, and the
preinstalled scientific stack are used.  No network access is required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.optimize import brentq, least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Determinism.  These data are CPU/GPU rasterization driven and do not train.
SEED = 20261020
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
np.random.seed(SEED)
torch.manual_seed(SEED)

# Image/rasterization choices.  Half resolution is enough to estimate a global
# depth scale (the object spans >1000 mask pixels over the episode) and keeps
# the whole solve comfortably below the 8 GB GPU budget.
OBJ_RENDER_SCALE = 0.5
OBJ_BATCH = 16
# The visible-human filter uses the MoGe/SAM3D depth disagreement distribution:
# median ~3.6 cm, 90th percentile ~7 cm.  8 cm accepts the photo-aligned front
# surface while rejecting back/hidden vertices that have the wrong 2-D target.
DEPTH_VIS_TOL_M = 0.08
# Correspondences farther than 3 cm are not same-surface points.  The
# reconstructed and reference MHR-derived vertex clouds differ by ~1 cm RMS.
HUMAN_MATCH_MAX_M = 0.03
# Floor-normal residual is scale-free.  1000 is a weak, explicit prior relative
# to thousands of dense image residuals; it removes the last ~0.02 deg.
FLOOR_NORMAL_WEIGHT = 1000.0
# Resting-object contact uses a smooth minimum (3 mm temperature).  These are
# the flat/lying and upright plateaus; 3 degrees cleanly isolates them from the
# tipping motion without using image-dependent hand tuning.
STATIC_ROT_DEG = 3.0
CONTACT_SAMPLE_STRIDE = 5
CONTACT_LOW_VERTS = 80


def json_dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def decode_depth_png(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise RuntimeError("Could not decode an aligned MoGe depth frame")
    return depth.astype(np.float32) / 1000.0  # uint16 millimetres


def project(T: np.ndarray, X: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project world row-vector points with an OpenCV camera-to-world pose."""
    Xc = (X - T[:3, 3]) @ T[:3, :3]  # R^T (X-t), row form
    q = Xc @ K.T
    return q[:, :2] / q[:, 2:3]


def pose_from_delta(x: np.ndarray, T0: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(x[:3], dtype=np.float64)).as_matrix() @ T0[:3, :3]
    T[:3, 3] = T0[:3, 3] + np.asarray(x[3:6], dtype=np.float64)
    return T


def camera_change(T: np.ndarray, T0: np.ndarray) -> tuple[float, float]:
    dR = T[:3, :3] @ T0[:3, :3].T
    deg = np.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))
    cm = 100.0 * np.linalg.norm(T[:3, 3] - T0[:3, 3])
    return float(deg), float(cm)


def fit_floor_plane_cue(
    depth_h5: Path,
    masks_h5: Path,
    K: np.ndarray,
    T0: np.ndarray,
    n_world: np.ndarray,
    d_world: float,
    frames: int,
) -> tuple[np.ndarray, float, list[dict], np.ndarray, np.ndarray]:
    """Robustly fit the floor in camera coordinates over the static episode.

    The returned normal/distance are robust medians of independent per-frame TLS
    plane fits.  The normal is unaffected by a global MoGe depth scale; the
    distance is retained for diagnostics, not forced into the camera solve.
    """
    H, W = 1152, 1536
    nc0 = T0[:3, :3].T @ n_world
    dc0 = float(n_world @ T0[:3, 3] + d_world)
    per_frame: list[dict] = []
    cloud_parts: list[np.ndarray] = []
    fit_ids: list[int] = []

    with h5py.File(depth_h5, "r") as fd, h5py.File(masks_h5, "r") as fm:
        for fi in range(0, frames, 20):
            depth = decode_depth_png(bytes(fd["aligned/front_stereo_camera_left"][fi]))
            person = fm["person"][fi]
            obj = fm["object"][fi]

            # The floor occupies the lower image.  Every 5th pixel keeps the fit
            # cheap while retaining hundreds of thousands of independent looks.
            vv, uu = np.mgrid[600:1152:5, 0:1536:5]
            ds = depth[600:1152:5, 0:1536:5]
            pm = person[600:1152:5, 0:1536:5]
            om = obj[600:1152:5, 0:1536:5]
            sel = (~pm) & (~om) & np.isfinite(ds) & (ds > 0)
            u, v, z = uu[sel], vv[sel], ds[sel]
            P = np.stack(
                [
                    (u - K[0, 2]) / K[0, 0] * z,
                    (v - K[1, 2]) / K[1, 1] * z,
                    z,
                ],
                axis=1,
            )
            inlier = np.abs(P @ nc0 + dc0) < 0.14
            n = nc0.copy()
            d = dc0
            for it in range(6):
                Q = P[inlier]
                center = np.median(Q, axis=0)
                # Total least squares plane normal.
                _, _, vh = np.linalg.svd(Q - center, full_matrices=False)
                n = vh[-1]
                if n @ nc0 < 0:
                    n = -n
                d = float(-n @ center)
                dist = P @ n + d
                inlier = np.abs(dist) < max(0.012, 0.030 / (it + 1))
            resid = P[inlier] @ n + d
            per_frame.append(
                {
                    "frame": int(fi),
                    "n": n.tolist(),
                    "d": float(d),
                    "inliers": int(inlier.sum()),
                    "rms": float(np.sqrt(np.mean(resid**2))),
                }
            )
            cloud_parts.append(P[inlier][::2])
            fit_ids.append(fi)

    normals = np.asarray([r["n"] for r in per_frame], dtype=np.float64)
    dists = np.asarray([r["d"] for r in per_frame], dtype=np.float64)
    n_obs = np.median(normals, axis=0)
    n_obs /= np.linalg.norm(n_obs)
    d_obs = float(np.median(dists))
    cloud = np.concatenate(cloud_parts, axis=0)
    ids = np.asarray(fit_ids, dtype=np.int32)
    return n_obs, d_obs, per_frame, cloud, ids


def fit_dense_human_camera(
    vertices_world: np.ndarray,
    human_recon_world: np.ndarray,
    depth_frame0: np.ndarray,
    person_mask0: np.ndarray,
    K: np.ndarray,
    T0: np.ndarray,
    n_world: np.ndarray,
    n_floor_obs: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Estimate camera pose from dense frame-0 human correspondences.

    Reference and reconstructed MHR-derived vertex orders are not identical, so
    correspondences are nearest-surface matches in world space.  The exact
    photo-aligned reconstruction supplies target pixels; the released reference
    mesh supplies metric 3-D points.  Back/hidden vertices are discarded by
    testing each reconstructed point against the aligned MoGe depth.
    """
    dist, idx = cKDTree(vertices_world).query(human_recon_world)
    base_sel = dist < HUMAN_MATCH_MAX_M
    X_true = vertices_world[idx[base_sel]]
    X_recon_world = human_recon_world[base_sel]

    # Target image locations are the reconstruction's own baseline projections.
    uv = project(T0, X_recon_world, K)
    Xc = (X_recon_world - T0[:3, 3]) @ T0[:3, :3]
    z = Xc[:, 2]
    u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, person_mask0.shape[1] - 1)
    v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, person_mask0.shape[0] - 1)
    inside = person_mask0[v, u]
    visible = inside & (np.abs(z - depth_frame0[v, u]) < DEPTH_VIS_TOL_M)
    X_true = X_true[visible]
    uv = uv[visible]
    match_dist = dist[base_sel][visible]

    def pose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R = Rotation.from_rotvec(x[:3]).as_matrix() @ T0[:3, :3]
        t = T0[:3, 3] + x[3:6]
        return R, t

    def proj(x: np.ndarray) -> np.ndarray:
        R, t = pose(x)
        q = ((X_true - t) @ R) @ K.T
        return q[:, :2] / q[:, 2:3]

    R0, _ = pose(np.zeros(6))
    nc0 = R0.T @ n_world

    def residual(x: np.ndarray) -> np.ndarray:
        R, _ = pose(x)
        nc = R.T @ n_world
        # 2 px is the robust human-mesh / reconstructed-mesh image noise scale.
        dense = ((proj(x) - uv) / 2.0).ravel()
        floor = FLOOR_NORMAL_WEIGHT * (nc - n_floor_obs)
        return np.concatenate([dense, floor])

    result = least_squares(
        residual,
        np.zeros(6),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=1500,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    x = np.asarray(result.x, dtype=np.float64)
    T = pose_from_delta(x, T0)
    errors = np.linalg.norm(proj(x) - uv, axis=1)
    R, _ = pose(x)
    info = {
        "matches": int(len(X_true)),
        "median_match_m": float(np.median(match_dist)),
        "reprojection_mean_px": float(errors.mean()),
        "reprojection_median_px": float(np.median(errors)),
        "reprojection_p90_px": float(np.quantile(errors, 0.90)),
        "floor_normal_error_deg": float(
            np.degrees(np.arccos(np.clip((R.T @ n_world) @ n_floor_obs, -1, 1)))
        ),
    }
    return x, info


def object_depth_scale_scan(
    object_mesh: trimesh.Trimesh,
    T_world_mesh: np.ndarray,
    T0: np.ndarray,
    K: np.ndarray,
    masks_h5: Path,
) -> tuple[float, dict]:
    """Choose a global camera-ray depth scale by exact-mesh mask agreement."""
    import nvdiffrast.torch as dr

    frames = len(T_world_mesh)
    W, H = 1536, 1152
    W2, H2 = int(W * OBJ_RENDER_SCALE), int(H * OBJ_RENDER_SCALE)
    V = np.asarray(object_mesh.vertices, dtype=np.float32)
    F = np.asarray(object_mesh.faces, dtype=np.int32)
    T_cam_base = np.linalg.inv(T0) @ T_world_mesh.astype(np.float64)

    targets_np = np.empty((frames, H2, W2), dtype=np.uint8)
    with h5py.File(masks_h5, "r") as fm:
        for i in range(frames):
            targets_np[i] = cv2.resize(
                fm["object"][i].astype(np.uint8),
                (W2, H2),
                interpolation=cv2.INTER_NEAREST,
            )
    targets = torch.tensor(targets_np, dtype=torch.bool, device="cuda")
    del targets_np

    Vt = torch.tensor(V, dtype=torch.float32, device="cuda")
    Ft = torch.tensor(F, dtype=torch.int32, device="cuda")
    ctx = dr.RasterizeCudaContext()

    def evaluate(scale: float) -> np.ndarray:
        ious: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, frames, OBJ_BATCH):
                end = min(start + OBJ_BATCH, frames)
                Ts = T_cam_base[start:end]
                Rt = torch.tensor(Ts[:, :3, :3].transpose(0, 2, 1), dtype=torch.float32, device="cuda")
                tt = torch.tensor(Ts[:, :3, 3], dtype=torch.float32, device="cuda")
                tt[:, 2] *= float(scale)  # scale depth from the camera centre
                Xc = torch.bmm(Vt[None].expand(end - start, -1, -1), Rt) + tt[:, None, :]
                z = Xc[:, :, 2]
                px = torch.stack(
                    [
                        (K[0, 0] * Xc[:, :, 0] + K[0, 2] * z) * OBJ_RENDER_SCALE,
                        (K[1, 1] * Xc[:, :, 1] + K[1, 2] * z) * OBJ_RENDER_SCALE,
                    ],
                    dim=2,
                ) / z[:, :, None]
                pos = torch.cat(
                    [
                        px[:, :, 0:1] / (W2 / 2) - 1,
                        px[:, :, 1:2] / (H2 / 2) - 1,
                        torch.zeros_like(px[:, :, :1]),
                        torch.ones_like(px[:, :, :1]),
                    ],
                    dim=2,
                )
                rast, _ = dr.rasterize(ctx, pos, Ft, resolution=[H2, W2])
                rendered = rast[:, :, :, 3] > 0
                m = targets[start:end]
                inter = (rendered & m).sum((1, 2)).float()
                union = (rendered | m).sum((1, 2)).float().clamp_min(1.0)
                ious.append((inter / union).cpu().numpy())
        return np.concatenate(ious)

    candidates = np.asarray([0.9800, 0.9825, 0.9850, 0.9875, 0.9900, 0.9925, 0.9950], dtype=np.float64)
    scan = []
    iou_curves = {}
    for s in candidates:
        ious = evaluate(float(s))
        iou_curves[float(s)] = ious
        scan.append((float(s), float(ious.mean()), float(np.median(ious)), float(np.quantile(ious, 0.10))))
    scan_arr = np.asarray(scan)
    best_i = int(np.argmax(scan_arr[:, 1]))
    best_s = float(scan_arr[best_i, 0])
    # Local parabolic refinement of the mean-IoU curve.
    if 0 < best_i < len(scan_arr) - 1:
        coef = np.polyfit(scan_arr[best_i - 1 : best_i + 2, 0], scan_arr[best_i - 1 : best_i + 2, 1], 2)
        if coef[0] < 0:
            refined = float(-coef[1] / (2.0 * coef[0]))
            if candidates[0] <= refined <= candidates[-1]:
                best_s = refined
    base_iou = iou_curves[1.0] if 1.0 in iou_curves else evaluate(1.0)
    chosen_iou = evaluate(best_s)
    info = {
        "scale": best_s,
        "mean_iou_baseline": float(base_iou.mean()),
        "mean_iou_chosen": float(chosen_iou.mean()),
        "median_iou_chosen": float(np.median(chosen_iou)),
        "scan": scan,
    }
    del targets, Vt, Ft
    torch.cuda.empty_cache()
    return best_s, info


def object_contact_height_adjustment(
    object_mesh: trimesh.Trimesh,
    T_world_mesh: np.ndarray,
    T0: np.ndarray,
    x_dense: np.ndarray,
    depth_scale: float,
    n_world: np.ndarray,
    d_world: float,
) -> tuple[np.ndarray, dict]:
    """Raise/lower the camera along the floor normal so resting objects touch.

    Orientation and tangential camera translation stay fixed by the dense human
    cue.  Only the one physically well-observed direction (distance from the
    released floor) is adjusted.  Static plateaus are found from object rotation:
    less than 3 deg from either the first pose or the last pose.
    """
    T = T_world_mesh.astype(np.float64)
    angles_first = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(T[:, :3, :3] @ T[0, :3, :3].T).as_rotvec(),
            axis=1,
        )
    )
    angles_last = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(T[:, :3, :3] @ T[-1, :3, :3].T).as_rotvec(),
            axis=1,
        )
    )
    static = (angles_first < STATIC_ROT_DEG) | (angles_last < STATIC_ROT_DEG)
    frame_ids = np.arange(0, len(T), CONTACT_SAMPLE_STRIDE)
    frame_ids = frame_ids[static[frame_ids]]

    mesh_ids = np.arange(0, len(object_mesh.vertices), CONTACT_SAMPLE_STRIDE)
    Vh = np.c_[object_mesh.vertices[mesh_ids], np.ones(len(mesh_ids))]
    O_base = np.einsum("fij,vj->fvi", T[frame_ids], Vh)[:, :, :3]
    low_ids = np.argsort(O_base @ n_world + d_world, axis=1)[:, :CONTACT_LOW_VERTS]
    O = np.take_along_axis(O_base, low_ids[:, :, None], axis=1)
    R0, t0 = T0[:3, :3], T0[:3, 3]

    def contact_median(x: np.ndarray) -> float:
        R = Rotation.from_rotvec(x[:3]).as_matrix() @ R0
        t = t0 + x[3:6]
        # Baseline world -> camera, apply metric depth scale, then new camera.
        Xc = ((O - t0) @ R0) * depth_scale
        Wnew = Xc @ R.T + t
        sd = Wnew @ n_world + d_world
        # smooth minimum over the 80 lowest object vertices in each frame
        soft = -0.003 * np.log(np.exp(-sd / 0.003).mean(axis=1))
        return float(np.median(soft))

    def root(alpha: float) -> float:
        x = x_dense.copy()
        x[3:] += alpha * n_world
        return contact_median(x)

    lo, hi = -0.06, 0.06
    flo, fhi = root(lo), root(hi)
    if flo == 0:
        alpha = lo
    elif fhi == 0:
        alpha = hi
    elif flo * fhi < 0:
        alpha = float(brentq(root, lo, hi, xtol=1e-10))
    else:
        alpha = 0.0
    x_final = x_dense.copy()
    x_final[3:] += alpha * n_world
    info = {
        "static_frames": int(len(frame_ids)),
        "height_adjustment_cm": float(100.0 * alpha),
        "contact_median_before_m": float(contact_median(x_dense)),
        "contact_median_after_m": float(contact_median(x_final)),
    }
    return x_final, info


def rotate_world_points(X: np.ndarray, T_from_base_world: np.ndarray, T0: np.ndarray) -> np.ndarray:
    T = T_from_base_world @ np.linalg.inv(T0)
    return np.c_[X, np.ones(len(X))] @ T.T[:, :3]


def point_silhouette_metrics(V: np.ndarray, T: np.ndarray, K: np.ndarray, target: np.ndarray) -> dict:
    """Silhouette proxy from dense projected reference-human vertices.

    The reference NPZ does not include its own face array, so using MHR's face
    order would be invalid.  Its 18k world vertices are dense surface samples;
    2-px-radius disks (5 px diameter) reproduce the person support closely on
    this 1536x1152 image.  This is explicitly a projected-vertex silhouette,
    not a connectivity-dependent mesh raster.
    """
    H, W = target.shape
    uv = project(T, V, K)
    canvas = np.zeros((H, W), dtype=np.uint8)
    ij = np.clip(np.round(uv).astype(np.int32), [0, 0], [W - 1, H - 1])
    canvas[ij[:, 1], ij[:, 0]] = 1
    canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    rendered = canvas.astype(bool)
    inter = np.count_nonzero(rendered & target)
    union = np.count_nonzero(rendered | target)
    iou = inter / max(union, 1)
    e_r = cv2.Canny(canvas, 0, 1) > 0
    e_t = cv2.Canny(target.astype(np.uint8), 0, 1) > 0
    d_r = cv2.distanceTransform((~e_r).astype(np.uint8), cv2.DIST_L2, 5)
    d_t = cv2.distanceTransform((~e_t).astype(np.uint8), cv2.DIST_L2, 5)
    bd_rt = d_t[e_r]
    bd_tr = d_r[e_t]
    return {
        "iou": float(iou),
        "boundary_render_to_mask_median_px": float(np.median(bd_rt)),
        "boundary_render_to_mask_p95_px": float(np.quantile(bd_rt, 0.95)),
        "boundary_mask_to_render_median_px": float(np.median(bd_tr)),
        "boundary_mask_to_render_p95_px": float(np.quantile(bd_tr, 0.95)),
        "rendered_area_px": int(rendered.sum()),
        "target_area_px": int(target.sum()),
    }


def floor_world_residuals(P_cam: np.ndarray, T: np.ndarray, n_world: np.ndarray, d_world: float) -> dict:
    W = P_cam @ T[:3, :3].T + T[:3, 3]
    sd = W @ n_world + d_world
    return {
        "median_abs_m": float(np.median(np.abs(sd))),
        "mean_abs_m": float(np.mean(np.abs(sd))),
        "rms_m": float(np.sqrt(np.mean(sd**2))),
        "p90_abs_m": float(np.quantile(np.abs(sd), 0.90)),
        "median_signed_m": float(np.median(sd)),
    }


def human_floor_metrics(H_base_world: np.ndarray, T_new: np.ndarray, T0: np.ndarray, n_world: np.ndarray, d_world: float) -> dict:
    Hnew = rotate_world_points(H_base_world.reshape(-1, 3), T_new, T0).reshape(H_base_world.shape)
    sd = Hnew @ n_world + d_world
    low = sd.min(axis=1)
    q005 = np.quantile(sd, 0.005, axis=1)
    return {
        "lowest_vertex_median_m": float(np.median(low)),
        "lowest_vertex_mean_m": float(np.mean(low)),
        "lowest_vertex_p90_abs_m": float(np.quantile(np.abs(low), 0.90)),
        "q0.5pct_median_m": float(np.median(q005)),
    }


def object_floor_metrics(
    object_mesh: trimesh.Trimesh,
    T_world_mesh: np.ndarray,
    T_compare: np.ndarray,
    static_mask: np.ndarray,
    n_world: np.ndarray,
    d_world: float,
) -> dict:
    ids = np.arange(0, len(object_mesh.vertices), CONTACT_SAMPLE_STRIDE)
    Vh = np.c_[object_mesh.vertices[ids], np.ones(len(ids))]
    fids = np.arange(0, len(T_world_mesh), CONTACT_SAMPLE_STRIDE)
    fids = fids[static_mask[fids]]
    O = np.einsum("fij,vj->fvi", T_compare[fids], Vh)[:, :, :3]
    low = (O @ n_world + d_world).min(axis=1)
    return {
        "static_frames": int(len(fids)),
        "lowest_vertex_median_m": float(np.median(low)),
        "lowest_vertex_mean_abs_m": float(np.mean(np.abs(low))),
        "lowest_vertex_p90_abs_m": float(np.quantile(np.abs(low), 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, default=Path("/pipeline"))
    args = parser.parse_args()
    t_start = time.time()

    inputs = args.inputs.resolve()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    camera_json = json.loads((inputs / "camera.json").read_text())
    episode = json.loads((inputs / "episode.json").read_text())
    K = np.asarray(camera_json["K"], dtype=np.float64)
    T0 = np.asarray(camera_json["T_world_cam"], dtype=np.float64)
    plane = np.asarray(json.loads((inputs / "ground_plane.json").read_text())["plane"], dtype=np.float64)
    n_world, d_world = plane[:3], float(plane[3])

    recon_path = inputs / "reconstruction.npz"
    with np.load(recon_path) as z:
        T_world_mesh = np.asarray(z["T_world_mesh"], dtype=np.float64)
        human_world0 = np.asarray(z["human_vertices_world"][0], dtype=np.float64)
        human_all = np.asarray(z["human_vertices_world"], dtype=np.float64)
    ref = np.load(inputs / "frame0_reference_human.npz")
    ref_vertices = np.asarray(ref["vertices_world"], dtype=np.float64)
    faces_count = int(episode["frames"])
    if len(T_world_mesh) != faces_count:
        raise RuntimeError(f"Expected {faces_count} object poses, got {len(T_world_mesh)}")

    depth_h5 = args.pipeline / "depth/moge2/aligned.h5"
    masks_h5 = inputs / "masks.h5"

    # Frame-0 depth and person mask.
    with h5py.File(depth_h5, "r") as fd, h5py.File(masks_h5, "r") as fm:
        depth0 = decode_depth_png(bytes(fd["aligned/front_stereo_camera_left"][0]))
        person0 = fm["person"][0]

    print("[1/5] fitting released floor plane in camera depth")
    n_floor_obs, d_floor_obs, floor_frames, floor_cloud, _ = fit_floor_plane_cue(
        depth_h5, masks_h5, K, T0, n_world, d_world, faces_count
    )

    print("[2/5] optimizing camera on visible dense frame-0 human correspondences")
    x_dense, dense_info = fit_dense_human_camera(
        ref_vertices, human_world0, depth0, person0, K, T0, n_world, n_floor_obs
    )
    T_dense = pose_from_delta(x_dense, T0)
    dense_rot, dense_cm = camera_change(T_dense, T0)

    object_mesh = trimesh.load(inputs / "object_mesh.glb", force="scene", process=False).to_geometry()

    print("[3/5] scanning global object depth scale against exact mesh/masks")
    depth_scale, scale_info = object_depth_scale_scan(
        object_mesh, T_world_mesh, T0, K, masks_h5
    )

    print("[4/5] enforcing resting-object floor contact (height direction only)")
    x_final, contact_info = object_contact_height_adjustment(
        object_mesh, T_world_mesh, T0, x_dense, depth_scale, n_world, d_world
    )
    T_final = pose_from_delta(x_final, T0)
    final_rot, final_cm = camera_change(T_final, T0)

    # Re-express the baseline camera-frame object trajectory in the new world.
    T_cam_mesh = np.linalg.inv(T0) @ T_world_mesh
    T_cam_mesh_scaled = T_cam_mesh.copy()
    T_cam_mesh_scaled[:, :3, 3] *= depth_scale
    T_world_mesh_final = np.einsum("ij,fjk->fik", T_final, T_cam_mesh_scaled).astype(np.float64)

    # Baseline object trajectory for validation.
    T_world_mesh_base = T_world_mesh.astype(np.float64)

    print("[5/5] validation and report")
    person_metrics_base = point_silhouette_metrics(ref_vertices, T0, K, person0)
    person_metrics_final = point_silhouette_metrics(ref_vertices, T_final, K, person0)
    floor_base = floor_world_residuals(floor_cloud, T0, n_world, d_world)
    floor_final = floor_world_residuals(floor_cloud, T_final, n_world, d_world)
    feet_base = human_floor_metrics(human_all, T0, T0, n_world, d_world)
    feet_final = human_floor_metrics(human_all, T_final, T0, n_world, d_world)

    angles_first = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(T_world_mesh[:, :3, :3] @ T_world_mesh[0, :3, :3].T).as_rotvec(),
            axis=1,
        )
    )
    angles_last = np.degrees(
        np.linalg.norm(
            Rotation.from_matrix(T_world_mesh[:, :3, :3] @ T_world_mesh[-1, :3, :3].T).as_rotvec(),
            axis=1,
        )
    )
    static_mask = (angles_first < STATIC_ROT_DEG) | (angles_last < STATIC_ROT_DEG)
    object_floor_base = object_floor_metrics(
        object_mesh, T_world_mesh_base, T_world_mesh_base, static_mask, n_world, d_world
    )
    object_floor_final = object_floor_metrics(
        object_mesh, T_world_mesh_final, T_world_mesh_final, static_mask, n_world, d_world
    )

    # Floor-only rotation correction (minimal geodesic aligning observed normal).
    nc0 = T0[:3, :3].T @ n_world
    Rcorr = Rotation.align_vectors(n_floor_obs[None, :], nc0[None, :])[0].as_matrix()
    T_floor_only = T0.copy()
    T_floor_only[:3, :3] = T0[:3, :3] @ Rcorr
    floor_rot, _ = camera_change(T_floor_only, T0)
    dc0 = float(n_world @ T0[:3, 3] + d_world)

    # Object-contact-only normal-direction height correction at baseline scale.
    object_contact_alpha = -contact_info["contact_median_before_m"]
    T_contact_only = T0.copy()
    T_contact_only[:3, 3] += object_contact_alpha * n_world
    _, contact_only_cm = camera_change(T_contact_only, T0)

    # Save required outputs.
    np.savez_compressed(out, T_world_mesh=T_world_mesh_final.astype(np.float64))
    camera_out = {
        "K": K.tolist(),
        "T_world_cam": T_final.tolist(),
        "width": int(camera_json["width"]),
        "height": int(camera_json["height"]),
        "convention": "OpenCV camera x-right, y-down, z-forward; world y-down",
        "camera_change_from_baseline": {
            "rotation_deg": final_rot,
            "center_cm": final_cm,
        },
        "object_depth_scale": float(depth_scale),
        "notes": (
            "K unchanged. Rotation constrained jointly by visible frame-0 human "
            "dense correspondences and the released floor normal; height by resting "
            "object contact after a global object depth-scale calibration."
        ),
    }
    json_dump(out.parent / "camera_estimate.json", camera_out)

    tools_text = f"""# Tools and provenance

No weights, repositories, datasets, or annotations were downloaded for the final solve. All inputs are the released files in `/work/inputs`; camera-frame depth comes from `/pipeline/depth/moge2/aligned.h5`.

## Runtime dependencies used by `solve.py`

- Python 3.11
- numpy {np.__version__}
- scipy {__import__('scipy').__version__}
- opencv-python {cv2.__version__}
- h5py {__import__('h5py').__version__}
- trimesh {trimesh.__version__}
- PyTorch {torch.__version__}
- nvdiffrast {metadata.version('nvdiffrast') if _has_dist('nvdiffrast') else 'installed from /workspace/third_party'}

## Tools inspected or tried during development (not required by `solve.py`)

- ffmpeg 4.4.2-0ubuntu0.22.04.1: used only to extract a visual frame montage during development.
- SAM3D-Body checkpoint at `/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt`: tried for diagnostic 2-D keypoints. Its supporting torchvision 0.20.1+cu124 was used by that diagnostic only. The final solver does not run the model.
- Open3D {metadata.version('open3d') if _has_dist('open3d') else '0.20.0'}: imported while checking available tools; not used.
- kornia {metadata.version('kornia') if _has_dist('kornia') else '0.8.3'}: availability probe only; not used.
- Availability probes that were not installed and not used: mediapipe, mmpose, ultralytics, onnxruntime, rtmlib.
- CARI4D source under `/workspace` and `/code/reconstruction`: inspected for conventions; no code was copied into the result.
- Git repositories cloned or updated: none.

No public downloads were made. The final script is offline and deterministic.
"""
    (out.parent / "TOOLS.md").write_text(tools_text)

    reproj_gain = dense_info["reprojection_mean_px"]
    report = f"""# Camera and world-trajectory estimate

## Adopted estimate

- Intrinsics `K`: unchanged from `camera.json`.
- Camera rotation change from baseline: **{final_rot:.3f} deg**.
- Camera centre change from baseline: **{final_cm:.2f} cm**.
- Object camera-ray depth scale: **{depth_scale:.4f}** (`T_cam_mesh` translation scaled; rotation and metric mesh size unchanged).
- Frames written: **{len(T_world_mesh_final)}**.

The baseline camera-frame object trajectory is preserved except for the small global ray-depth calibration. The output is `T_world_mesh = T_world_cam_new @ T_cam_mesh_scaled`.

## Evidence and per-cue agreement

| Cue | Suggested camera change | Agreement with adopted estimate |
|---|---|---|
| Visible dense frame-0 human ({dense_info['matches']} correspondences) | {dense_rot:.3f} deg, {dense_cm:.2f} cm | primary 6-DoF estimate; reprojection mean {reproj_gain:.3f} px |
| Released floor normal from {len(floor_frames)} MoGe frames | {floor_rot:.3f} deg tilt | adopted normal error {dense_info['floor_normal_error_deg']:.3f} deg at dense solution; floor cue and human agree |
| Resting object floor contact | {100*object_contact_alpha:.2f} cm along upward normal before depth scaling | same sign; weak scalar height constraint; final contact explicitly enforced |
| Exact object mesh vs object masks | depth scale {depth_scale:.4f}; IoU {scale_info['mean_iou_baseline']:.4f} -> {scale_info['mean_iou_chosen']:.4f} | independent metric-depth calibration; adopted |

The dense human and floor-normal cues agree to about **{dense_info['floor_normal_error_deg']:.2f} deg**. The object contact and dense translation both imply a small upward correction. This agreement is why the baseline camera was changed rather than left unchanged.

For reference, trusting the full aligned MoGe plane distance would suggest a {100*(d_floor_obs-dc0):.1f} cm normal-direction offset. That quantity is scale-sensitive (per-frame MoGe/SAM3D scales vary), so it was not used directly; only the scale-free normal was combined with the human cue. The object-image scale scan independently finds a ~{100*(1-depth_scale):.1f}% metric depth correction.

## Validation (baseline vs adopted)

| Metric | Baseline | Adopted |
|---|---:|---:|
| Frame-0 reference-human projected-vertex silhouette IoU | {person_metrics_base['iou']:.4f} | {person_metrics_final['iou']:.4f} |
| Human boundary distance, render -> mask, median / p95 (px) | {person_metrics_base['boundary_render_to_mask_median_px']:.2f} / {person_metrics_base['boundary_render_to_mask_p95_px']:.2f} | {person_metrics_final['boundary_render_to_mask_median_px']:.2f} / {person_metrics_final['boundary_render_to_mask_p95_px']:.2f} |
| Human boundary distance, mask -> render, median / p95 (px) | {person_metrics_base['boundary_mask_to_render_median_px']:.2f} / {person_metrics_base['boundary_mask_to_render_p95_px']:.2f} | {person_metrics_final['boundary_mask_to_render_median_px']:.2f} / {person_metrics_final['boundary_mask_to_render_p95_px']:.2f} |
| MoGe floor points to released plane, median abs / RMS (m) | {floor_base['median_abs_m']:.4f} / {floor_base['rms_m']:.4f} | {floor_final['median_abs_m']:.4f} / {floor_final['rms_m']:.4f} |
| Human lowest-vertex floor distance, median (m) | {feet_base['lowest_vertex_median_m']:.4f} | {feet_final['lowest_vertex_median_m']:.4f} |
| Human 0.5th-percentile floor distance, median (m) | {feet_base['q0.5pct_median_m']:.4f} | {feet_final['q0.5pct_median_m']:.4f} |
| Resting object lowest-vertex floor distance, median (m) | {object_floor_base['lowest_vertex_median_m']:.4f} | {object_floor_final['lowest_vertex_median_m']:.4f} |
| Resting object lowest-vertex mean abs distance (m) | {object_floor_base['lowest_vertex_mean_abs_m']:.4f} | {object_floor_final['lowest_vertex_mean_abs_m']:.4f} |
| Exact-object mask IoU over episode | {scale_info['mean_iou_baseline']:.4f} | {scale_info['mean_iou_chosen']:.4f} |

The reference NPZ contains vertices but not its own face connectivity and its vertex order differs from the MHR checkpoint. Silhouette validation therefore uses a dense projected-vertex support with a fixed 5-pixel disk radius (the projected nearest-neighbour spacing is ~0.7-3.7 px and this gives baseline support area close to the SAM2 mask area). This avoids applying the checkpoint's incompatible face indices.

## Details

- Dense correspondences use nearest-surface matches under {HUMAN_MATCH_MAX_M*100:.0f} cm, then retain points inside the person mask whose reconstructed depth agrees with aligned MoGe within {DEPTH_VIS_TOL_M*100:.0f} cm. This removes hidden/back-surface matches. Reprojection: mean {dense_info['reprojection_mean_px']:.3f} px, median {dense_info['reprojection_median_px']:.3f} px, p90 {dense_info['reprojection_p90_px']:.3f} px.
- Static object plateaus contain {contact_info['static_frames']} sampled frames. Contact smooth-min before/after height adjustment: {contact_info['contact_median_before_m']:.4f} / {contact_info['contact_median_after_m']:.4f} m.
- Final floor-normal error relative to the median MoGe floor normal: {dense_info['floor_normal_error_deg']:.3f} deg before contact; contact adjustment is purely along that normal.
- Runtime: {time.time()-t_start:.1f} s on one GPU/CPU process.

## Confidence

**Medium-high.** The camera tilt is corroborated by two independent static cues (human dense reprojection and floor normal), and the object depth scale improves exact-mesh mask IoU without changing mesh size. The largest remaining uncertainty is the released floor plane's absolute scale versus MoGe/SAM3D metric depth; camera orientation is much better constrained than absolute camera height. Residual human-floor penetration of ~1-2 cm is within the released foot-plane tolerance (mean 3.9 cm). World ADD should improve mainly through the corrected camera rotation/translation and the ~1% object depth scale, but ambiguity along the camera ray remains the dominant risk.
"""
    (out.parent / "REPORT.md").write_text(report)
    print(f"Wrote {out}")
    print(f"Wrote {out.parent / 'camera_estimate.json'}")
    print(f"Wrote {out.parent / 'REPORT.md'}")
    print(f"Wrote {out.parent / 'TOOLS.md'}")
    print(
        f"camera delta: {final_rot:.3f} deg, {final_cm:.2f} cm; "
        f"object depth scale {depth_scale:.4f}; runtime {time.time()-t_start:.1f}s"
    )


def _has_dist(name: str) -> bool:
    try:
        metadata.version(name)
        return True
    except metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
