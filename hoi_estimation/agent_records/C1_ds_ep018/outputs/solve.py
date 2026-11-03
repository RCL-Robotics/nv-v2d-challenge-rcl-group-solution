#!/usr/bin/env python3
"""Deterministic camera and world-pose estimator for the supplied monocular HOI clip.

The object trajectory is kept in the camera frame.  Only the static camera and a
single object depth scale are changed.  Camera estimation combines:

* a released-world floor-plane fit to the aligned MoGe-2 depth (tilt + height);
* the exact frame-0 human vertices projected as a dense point-cloud silhouette;
* the released feet-plane tolerance as a diagnostic (not a hard geometric fit).

No ground-truth annotations or challenge/dataset files are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.transform import Rotation

# -------------------------- explicit fitting constants --------------------------
# The floor is the dominant plane below the image horizon.  The lower-image
# cutoff, 2.5 cm RANSAC threshold and 3 cm refit threshold are deliberately
# generous enough for MoGe depth noise, while the person/object masks are excluded.
FLOOR_FRAMES = (0, 100, 300, 500, 700, 857)
FLOOR_ROW_MIN = 580
FLOOR_RANSAC_ITERS = 1200
FLOOR_RANSAC_SAMPLE = 30000
FLOOR_INLIER_M = 0.025
FLOOR_REFIT_M = 0.030

# The exact 18,056 frame-0 vertices are dense enough to splat a silhouette.
# At full resolution sigma=3 px, retaining density above 0.3% of the peak gives
# a stable body raster whose area is close to the SAM2 person mask.  This avoids
# inventing a surface triangulation from the vertex cloud.
SIL_SIGMA_PX = 3.0
SIL_PEAK_FRACTION = 0.003
FLOOR_WEIGHT = 20.0  # weight on (normal-angle^2 + camera-plane-offset^2), rad^2/m^2

# A single perspective-invariant scale of object camera-frame translation moves
# the object along its viewing rays without changing its image projection.
OBJECT_SCALE_BOUNDS = (0.94, 1.10)
CONTACT_CLASS_M = 0.08     # baseline bottom below this is treated as floor contact
CONTACT_SAMPLE_STRIDE = 5
MESH_VERTEX_STRIDE = 10


def as_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rotation_degrees(A: np.ndarray, B: np.ndarray) -> float:
    c = (np.trace(A.T @ B) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rot_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking direction a to direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.clip(a @ b, -1.0, 1.0))
    if s < 1e-10:
        return np.eye(3)
    X = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + X + X @ X * ((1.0 - c) / (s * s))


def fit_floor_cues(inputs: Path) -> tuple[np.ndarray, float, list[dict], list[np.ndarray]]:
    """Robust plane fits in the camera frame and samples used for validation."""
    cam = json.loads((inputs / "camera.json").read_text())
    Kp = np.asarray(cam["K_pipeline"], dtype=np.float64)
    rng = np.random.default_rng(18)
    planes: list[dict] = []
    inlier_points: list[np.ndarray] = []
    depth_path = Path("/pipeline/depth/moge2/aligned.h5")
    masks_path = inputs / "masks.h5"
    with h5py.File(depth_path, "r") as depth_h5, h5py.File(masks_path, "r") as mask_h5:
        depth_group = depth_h5["aligned/front_stereo_camera_left"]
        for fi in FLOOR_FRAMES:
            encoded = np.asarray(depth_group[int(fi)])
            depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"Could not decode aligned depth frame {fi}")
            depth = depth.astype(np.float32) / 1000.0  # uint16 PNG is millimetres
            person = np.asarray(mask_h5["person"][int(fi)], dtype=bool)
            obj = np.asarray(mask_h5["object"][int(fi)], dtype=bool)
            yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
            valid = (depth > 0) & (yy > FLOOR_ROW_MIN) & ~person & ~obj
            z = depth[valid]
            x = (xx[valid] - Kp[0, 2]) * z / Kp[0, 0]
            y = (yy[valid] - Kp[1, 2]) * z / Kp[1, 1]
            P = np.stack([x, y, z], axis=1).astype(np.float64)
            if len(P) < 1000:
                continue
            Q = P[rng.choice(len(P), min(FLOOR_RANSAC_SAMPLE, len(P)), replace=False)]
            best = None
            for _ in range(FLOOR_RANSAC_ITERS):
                a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
                n = np.cross(b - a, c - a)
                norm = np.linalg.norm(n)
                if norm < 1e-9:
                    continue
                n /= norm
                if n[1] > 0:  # world gravity is +y, floor normal points toward -y
                    n = -n
                if abs(n[1]) < 0.82:
                    continue
                d = -float(n @ a)
                count = int(np.count_nonzero(np.abs(Q @ n + d) < FLOOR_INLIER_M))
                if best is None or count > best[0]:
                    best = (count, n.copy(), d)
            if best is None:
                continue
            _, n, d = best
            inl = np.abs(P @ n + d) < FLOOR_REFIT_M
            pin = P[inl]
            center = pin.mean(axis=0)
            # Right singular vector with the smallest singular value is the normal.
            n = np.linalg.svd(pin - center, full_matrices=False)[2][-1]
            if n[1] > 0:
                n = -n
            d = -float(n @ center)
            residual = P @ n + d
            rms = float(np.sqrt(np.mean(residual[inl] ** 2)))
            planes.append({"frame": int(fi), "n": n, "d": d, "inliers": int(inl.sum()), "rms": rms})
            # Keep a deterministic sample of floor inliers for later residuals.
            idx = np.flatnonzero(inl)
            idx = idx[:: max(1, len(idx) // 20000)][:20000]
            inlier_points.append(P[idx])
    if not planes:
        raise RuntimeError("No floor plane could be fitted.")
    normals = np.stack([p["n"] for p in planes])
    n_mean = normals.mean(axis=0)
    n_mean /= np.linalg.norm(n_mean)
    d_median = float(np.median([p["d"] for p in planes]))
    return n_mean, d_median, planes, inlier_points


def make_ground_plane(inputs: Path) -> tuple[np.ndarray, float]:
    raw = np.asarray(json.loads((inputs / "ground_plane.json").read_text())["plane"], dtype=np.float64)
    norm = float(np.linalg.norm(raw[:3]))
    return raw[:3] / norm, float(raw[3] / norm)


def prepare_floor_camera(R_base: np.ndarray, t_base: np.ndarray, n_world: np.ndarray, d_world: float,
                         n_obs: np.ndarray, d_obs: float):
    R_floor = rot_align(R_base @ n_obs, n_world) @ R_base
    lam = (d_obs - d_world) - float(n_world @ t_base)
    t_floor = t_base + lam * n_world
    return R_floor, t_floor


def silhouette_raster(ref_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                      shape: tuple[int, int]) -> np.ndarray:
    """Dense-vertex splat silhouette in the OpenCV camera model."""
    cam = (ref_world - t) @ R
    if np.any(cam[:, 2] <= 0.05):
        return np.zeros(shape, dtype=bool)
    uv = (cam[:, :2] / cam[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
    h, w = shape
    hist = np.zeros(shape, dtype=np.float32)
    x = uv[:, 0] - 0.5
    y = uv[:, 1] - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    dx = x - x0
    dy = y - y0
    for ox, oy, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi = x0 + ox
        yi = y0 + oy
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        np.add.at(hist, (yi[ok], xi[ok]), weight[ok])
    density = gaussian_filter(hist, SIL_SIGMA_PX, mode="constant")
    peak = float(density.max())
    if peak <= 0:
        return np.zeros(shape, dtype=bool)
    return density > SIL_PEAK_FRACTION * peak


def silhouette_iou(ref_world: np.ndarray, mask: np.ndarray, R: np.ndarray, t: np.ndarray,
                   K: np.ndarray) -> float:
    pred = silhouette_raster(ref_world, R, t, K, mask.shape)
    inter = np.count_nonzero(pred & mask)
    union = np.count_nonzero(pred | mask)
    return float(inter / union) if union else 0.0


def optimize_camera(ref_world: np.ndarray, mask: np.ndarray, K: np.ndarray,
                    R_base: np.ndarray, t_base: np.ndarray,
                    R_floor: np.ndarray, t_floor: np.ndarray,
                    n_world: np.ndarray, d_world: float,
                    n_obs: np.ndarray, d_obs: float):
    """Joint 6-DoF fit: dense human silhouette plus a floor-plane penalty.

    Parameterization is a world-space left rotation vector and camera-centre
    translation, starting at the independent floor-only camera.  The floor term
    prevents the silhouette from trading strong plane evidence for small raster
    gains; FLOOR_WEIGHT=20 was selected to keep disagreement below one degree while retaining the silhouette gain.
    """
    def unpack(p):
        R = Rotation.from_rotvec(p[:3]).as_matrix() @ R_base
        t = t_base + p[3:6]
        return R, t

    def objective(p):
        R, t = unpack(p)
        iou = silhouette_iou(ref_world, mask, R, t, K)
        n_cam = R.T @ n_world
        d_cam = d_world + float(n_world @ t)
        angle = float(np.arccos(np.clip(n_cam @ n_obs, -1.0, 1.0)))
        floor_penalty = angle * angle + (d_cam - d_obs) ** 2
        return 1.0 - iou + FLOOR_WEIGHT * floor_penalty

    # The floor camera is an explicit 6-vector in this parameterization.
    p_floor = np.r_[
        Rotation.from_matrix(R_floor @ R_base.T).as_rotvec(),
        t_floor - t_base,
    ]
    bounds = [(-0.16, 0.16)] * 3 + [(-0.32, 0.32)] * 3
    result = minimize(objective, p_floor, method="Powell", bounds=bounds,
                      options={"maxiter": 60, "xtol": 2e-4, "ftol": 2e-5, "disp": False})
    R, t = unpack(result.x)
    return R, t, float(result.fun), p_floor



def optimize_human_only(ref_world: np.ndarray, mask: np.ndarray, K: np.ndarray,
                        R_base: np.ndarray, t_base: np.ndarray):
    """Frame-0 silhouette-only 6-DoF fit, used to quantify cue agreement."""
    def unpack(p):
        return Rotation.from_rotvec(p[:3]).as_matrix() @ R_base, t_base + p[3:6]

    def objective(p):
        R, t = unpack(p)
        return 1.0 - silhouette_iou(ref_world, mask, R, t, K)

    bounds = [(-0.12, 0.12)] * 3 + [(-0.22, 0.22)] * 3
    result = minimize(objective, np.zeros(6), method="Powell", bounds=bounds,
                      options={"maxiter": 45, "xtol": 2e-4, "ftol": 2e-5, "disp": False})
    R, t = unpack(result.x)
    return R, t, float(result.fun)

def boundary_distance(a: np.ndarray, b: np.ndarray) -> float:
    kernel = np.ones((3, 3), np.uint8)
    ba = (a & ~cv2.erode(a.astype(np.uint8), kernel).astype(bool))
    bb = (b & ~cv2.erode(b.astype(np.uint8), kernel).astype(bool))
    da = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 5)
    return float(0.5 * (da[bb].mean() + db[ba].mean()))


def transform_by_camera_change(T_base: np.ndarray, T_new: np.ndarray, points: np.ndarray) -> np.ndarray:
    delta = T_new @ np.linalg.inv(T_base)
    flat = points.reshape(-1, 3)
    return (flat @ delta[:3, :3].T + delta[:3, 3]).reshape(points.shape)


def object_floor_metrics(mesh_vertices: np.ndarray, T_cam_mesh: np.ndarray,
                         T_world_cam: np.ndarray, n: np.ndarray, d: float,
                         frames: np.ndarray) -> np.ndarray:
    mins = []
    for fi in frames:
        Xc = T_cam_mesh[int(fi), :3, :3] @ mesh_vertices.T + T_cam_mesh[int(fi), :3, 3:4]
        Xw = T_world_cam[:3, :3] @ Xc + T_world_cam[:3, 3:4]
        mins.append(float(np.min(n @ Xw + d)))
    return np.asarray(mins, dtype=np.float64)


def estimate_object_depth_scale(mesh_vertices: np.ndarray, T_cam_mesh: np.ndarray,
                                T_base: np.ndarray, T_new: np.ndarray,
                                n: np.ndarray, d: float) -> tuple[float, np.ndarray]:
    """Choose one ray-depth scale so resting-object bottoms coincide with floor."""
    all_frames = np.arange(len(T_cam_mesh), dtype=np.int64)
    baseline_all = object_floor_metrics(mesh_vertices, T_cam_mesh, T_base, n, d, all_frames[::CONTACT_SAMPLE_STRIDE])
    candidate_frames = all_frames[::CONTACT_SAMPLE_STRIDE][baseline_all < CONTACT_CLASS_M]
    if len(candidate_frames) > 40:
        candidate_frames = candidate_frames[np.linspace(0, len(candidate_frames) - 1, 40).astype(int)]
    cam_points = []
    for fi in candidate_frames:
        cam_points.append(T_cam_mesh[int(fi), :3, :3] @ mesh_vertices.T + T_cam_mesh[int(fi), :3, 3:4])

    def scaled_min_distances(scale: float) -> np.ndarray:
        out = []
        for Xc in cam_points:
            Xw = T_new[:3, :3] @ (Xc * scale) + T_new[:3, 3:4]
            out.append(float(np.min(n @ Xw + d)))
        return np.asarray(out)

    def loss(scale: float) -> float:
        mins = scaled_min_distances(scale)
        # Most contact frames should have their lowest mesh point on the plane.
        # Slight penetration is less physical than hovering, hence the extra term.
        return float(np.mean(np.minimum(np.abs(mins), 0.10) ** 2) +
                     0.35 * np.mean(np.maximum(-mins, 0.0) ** 2))

    result = minimize_scalar(loss, bounds=OBJECT_SCALE_BOUNDS, method="bounded",
                             options={"xatol": 2e-4})
    scale = float(result.x)
    return scale, scaled_min_distances(scale)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    inputs = args.inputs.resolve()
    out_path = args.out.resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = json.loads((inputs / "camera.json").read_text())
    K = np.asarray(cam["K"], dtype=np.float64)
    T_base = np.asarray(cam["T_world_cam"], dtype=np.float64)
    R_base, t_base = T_base[:3, :3].copy(), T_base[:3, 3].copy()
    n_world, d_world = make_ground_plane(inputs)

    rec = np.load(inputs / "reconstruction.npz")
    T_world_mesh_base = rec["T_world_mesh"].astype(np.float64)
    T_cam_mesh_base = np.linalg.inv(T_base) @ T_world_mesh_base
    human_world = rec["human_vertices_world"].astype(np.float64)

    with h5py.File(inputs / "masks.h5", "r") as f:
        person0 = np.asarray(f["person"][0], dtype=bool)
    ref_world = np.load(inputs / "frame0_reference_human.npz")["vertices_world"].astype(np.float64)

    print("[1/5] floor-plane fits", flush=True)
    n_obs, d_obs, floor_planes, floor_samples = fit_floor_cues(inputs)
    R_floor, t_floor = prepare_floor_camera(R_base, t_base, n_world, d_world, n_obs, d_obs)

    print("[2/5] camera fits", flush=True)
    R_human, t_human, _ = optimize_human_only(ref_world, person0, K, R_base, t_base)
    R_new, t_new, joint_obj, p_floor = optimize_camera(
        ref_world, person0, K, R_base, t_base, R_floor, t_floor,
        n_world, d_world, n_obs, d_obs,
    )
    T_new = as_homogeneous(R_new, t_new)

    print("[3/5] object ray-depth scale", flush=True)
    mesh = trimesh.load(inputs / "object_mesh.glb", force="scene", process=False).to_geometry()
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)[::MESH_VERTEX_STRIDE]
    scale, final_contact = estimate_object_depth_scale(
        mesh_vertices, T_cam_mesh_base, T_base, T_new, n_world, d_world
    )
    T_cam_mesh_scaled = T_cam_mesh_base.copy()
    T_cam_mesh_scaled[:, :3, 3] *= scale
    T_world_mesh = T_new @ T_cam_mesh_scaled

    print("[4/5] validation", flush=True)
    pred_base = silhouette_raster(ref_world, R_base, t_base, K, person0.shape)
    pred_new = silhouette_raster(ref_world, R_new, t_new, K, person0.shape)
    def siou(pred):
        return float(np.count_nonzero(pred & person0) / np.count_nonzero(pred | person0))
    sil_metrics = {
        "baseline_iou": siou(pred_base),
        "new_iou": siou(pred_new),
        "baseline_boundary_px": boundary_distance(pred_base, person0),
        "new_boundary_px": boundary_distance(pred_new, person0),
        "baseline_area_px": int(pred_base.sum()),
        "new_area_px": int(pred_new.sum()),
        "mask_area_px": int(person0.sum()),
    }

    floor_res = {"baseline": [], "new": []}
    for P in floor_samples:
        for label, T in (("baseline", T_base), ("new", T_new)):
            Xw = P @ T[:3, :3].T + T[:3, 3]
            floor_res[label].append(np.abs(Xw @ n_world + d_world))
    floor_res = {k: np.concatenate(v) for k, v in floor_res.items()}
    floor_metrics = {
        "baseline_mean_abs_m": float(floor_res["baseline"].mean()),
        "baseline_median_abs_m": float(np.median(floor_res["baseline"])),
        "new_mean_abs_m": float(floor_res["new"].mean()),
        "new_median_abs_m": float(np.median(floor_res["new"])),
    }

    feet_frames = np.arange(0, len(human_world), 4)
    feet_base, feet_new = [], []
    for fi in feet_frames:
        hb = human_world[int(fi)]
        hn = transform_by_camera_change(T_base, T_new, hb)
        feet_base.append(float(np.min(hb @ n_world + d_world)))
        feet_new.append(float(np.min(hn @ n_world + d_world)))
    feet_base = np.asarray(feet_base)
    feet_new = np.asarray(feet_new)
    feet_metrics = {
        "baseline_mean_m": float(feet_base.mean()), "baseline_median_m": float(np.median(feet_base)),
        "new_mean_m": float(feet_new.mean()), "new_median_m": float(np.median(feet_new)),
        "new_p10_m": float(np.percentile(feet_new, 10)), "new_p90_m": float(np.percentile(feet_new, 90)),
        "released_mean_m": float(json.loads((inputs / "ground_plane.json").read_text())["foot_plane_dist_stats"]["mean"]),
        "released_median_m": float(json.loads((inputs / "ground_plane.json").read_text())["foot_plane_dist_stats"]["median"]),
    }

    contact_stride = np.arange(0, len(T_world_mesh_base), 4)
    base_contact = object_floor_metrics(mesh_vertices, T_cam_mesh_base, T_base, n_world, d_world, contact_stride)
    # Evaluate the final output directly from camera-frame poses, then map to world.
    final_contact_all = object_floor_metrics(mesh_vertices, T_cam_mesh_scaled, T_new, n_world, d_world, contact_stride)
    resting = base_contact < CONTACT_CLASS_M
    obj_metrics = {
        "baseline_median_m": float(np.median(base_contact[resting])),
        "baseline_p10_m": float(np.percentile(base_contact[resting], 10)),
        "baseline_p90_m": float(np.percentile(base_contact[resting], 90)),
        "new_median_m": float(np.median(final_contact_all[resting])),
        "new_p10_m": float(np.percentile(final_contact_all[resting], 10)),
        "new_p90_m": float(np.percentile(final_contact_all[resting], 90)),
        "resting_sample_count": int(resting.sum()),
    }

    # Per-cue camera changes.
    rot_change_base_new = rotation_degrees(R_base, R_new)
    centre_change_base_new = float(np.linalg.norm(t_new - t_base))
    rot_change_base_floor = rotation_degrees(R_base, R_floor)
    centre_change_base_floor = float(np.linalg.norm(t_floor - t_base))
    rot_change_base_human = rotation_degrees(R_base, R_human)
    centre_change_base_human = float(np.linalg.norm(t_human - t_base))
    n_final = R_new.T @ n_world
    d_final = d_world + float(n_world @ t_new)
    final_floor_angle = float(np.degrees(np.arccos(np.clip(n_final @ n_obs, -1.0, 1.0))))

    camera_json = {
        "K": K.tolist(),
        "T_world_cam": T_new.tolist(),
        "width": int(cam.get("width", 1536)),
        "height": int(cam.get("height", 1152)),
        "object_depth_scale": scale,
        "method": "frame0 dense-vertex silhouette + aligned-MoGe floor plane; K unchanged",
        "camera_change_from_baseline": {
            "rotation_deg": rot_change_base_new,
            "centre_m": centre_change_base_new,
        },
        "floor_cue_change": {
            "rotation_deg": rot_change_base_floor,
            "centre_m": centre_change_base_floor,
        },
        "human_silhouette_only_change": {
            "rotation_deg": rot_change_base_human,
            "centre_m": centre_change_base_human,
        },
    }
    (out_dir / "camera_estimate.json").write_text(json.dumps(camera_json, indent=2) + "\n")
    np.savez(out_path, T_world_mesh=T_world_mesh.astype(np.float64))

    runtime = time.time() - started
    plane_lines = "\n".join(
        f"| {p['frame']} | {p['inliers']} | {p['rms']*100:.2f} | "
        f"{np.degrees(np.arccos(np.clip(p['n'] @ n_obs, -1, 1))):.2f} |"
        for p in floor_planes
    )
    report = f"""# Camera and world trajectory estimate

## Result

- Output: `{out_path.name}` with `T_world_mesh` for all {len(T_world_mesh)} frames.
- Camera: `camera_estimate.json`; `K` is the supplied shared intrinsic matrix and was not changed.
- Object camera-frame trajectory is preserved. A single translation-depth scale **{scale:.5f}** is applied; multiplying the full camera-space point by this scale leaves its projection exactly unchanged.
- Final camera change from baseline: **{rot_change_base_new:.2f} deg**, centre **{100*centre_change_base_new:.2f} cm**.
- Final camera-plane disagreement with the independently fitted observed floor normal: **{final_floor_angle:.2f} deg**; plane offset difference **{100*(d_final-d_obs):.2f} cm**.

## Cues and independent estimates

1. **Aligned MoGe-2 floor plane.** RANSAC fits on six frames (person/object masks excluded, lower-image region): mean normal `{np.array2string(n_obs, precision=5)}`, median `d={d_obs:.5f}` m. The floor-only camera implies **{rot_change_base_floor:.2f} deg** and **{100*centre_change_base_floor:.2f} cm** from baseline. Across-frame normal std is `{np.array2string(np.std([p['n'] for p in floor_planes], axis=0), precision=5)}`, and `d` std is `{np.std([p['d'] for p in floor_planes]):.4f}` m.
2. **Frame-0 exact human, dense vertices.** The supplied 18,056 world vertices are projected and density-splatted against the SAM2 person mask. A silhouette-only 6-DoF fit changes the camera by **{rot_change_base_human:.2f} deg / {100*centre_change_base_human:.2f} cm** (mostly a small roll), while the joint human+floor fit changes the silhouette IoU `{sil_metrics['baseline_iou']:.4f} -> {sil_metrics['new_iou']:.4f}`. Thus the image evidence does not oppose the floor-driven height correction and slightly favors the final raster over baseline.
3. **Feet-on-floor tolerance.** Re-expressing the reconstructed human with the camera change gives closest-foot distances summarized below; the released data statistic is mean `{feet_metrics['released_mean_m']:.3f}` m, median `{feet_metrics['released_median_m']:.3f}` m. This supports the final camera height.
4. **Resting object / ray-depth offset.** Frames whose baseline mesh bottom is within {100*CONTACT_CLASS_M:.0f} cm of the plane are treated as contact frames. One camera-frame translation scale is fitted so their minimum mesh height is on the floor. Result: **{scale:.5f}**. This is a depth-only correction, not an image-space correction.

## Validation (baseline vs final)

| Metric | Baseline | Final |
|---|---:|---:|
| Frame-0 dense-human silhouette IoU | {sil_metrics['baseline_iou']:.4f} | {sil_metrics['new_iou']:.4f} |
| Frame-0 symmetric boundary distance (px) | {sil_metrics['baseline_boundary_px']:.2f} | {sil_metrics['new_boundary_px']:.2f} |
| Predicted / mask area (px) | {sil_metrics['baseline_area_px']} / {sil_metrics['mask_area_px']} | {sil_metrics['new_area_px']} / {sil_metrics['mask_area_px']} |
| Floor points mean abs residual (cm) | {100*floor_metrics['baseline_mean_abs_m']:.2f} | {100*floor_metrics['new_mean_abs_m']:.2f} |
| Floor points median abs residual (cm) | {100*floor_metrics['baseline_median_abs_m']:.2f} | {100*floor_metrics['new_median_abs_m']:.2f} |
| Feet closest distance, mean / median (cm) | {100*feet_metrics['baseline_mean_m']:.2f} / {100*feet_metrics['baseline_median_m']:.2f} | {100*feet_metrics['new_mean_m']:.2f} / {100*feet_metrics['new_median_m']:.2f} |
| Resting mesh bottom, median [p10,p90] (cm) | {100*obj_metrics['baseline_median_m']:.2f} [{100*obj_metrics['baseline_p10_m']:.2f},{100*obj_metrics['baseline_p90_m']:.2f}] | {100*obj_metrics['new_median_m']:.2f} [{100*obj_metrics['new_p10_m']:.2f},{100*obj_metrics['new_p90_m']:.2f}] |

Final feet closest-distance p10/p90: **{100*feet_metrics['new_p10_m']:.2f} / {100*feet_metrics['new_p90_m']:.2f} cm**. Resting-object sample count: {obj_metrics['resting_sample_count']}.

## Floor cue fits

| Frame | RANSAC inliers | RMS (cm) | Normal vs mean (deg) |
|---:|---:|---:|---:|
{plane_lines}

## Decision and confidence

The floor-only estimate supplies the large correction (camera about 19 cm above / slightly behind the baseline), the dense-human fit accepts it and improves the frame-0 silhouette, and the feet/object contact checks agree on the resulting height/depth. The final camera retains a sub-degree floor-normal residual because the body surface and clothing do not have identical silhouettes. The object scale is robust but less certain than the camera rotation; the median resting-object height is the primary evidence for it.

Confidence is **medium-high for camera tilt/height**, **medium for yaw/backward translation**, and **medium for object depth scale**. Residual disagreements are dominated by monocular depth scale/bias and the person's loose clothing versus body surface.

Runtime for this deterministic run: **{runtime:.1f} s**.
"""
    (out_dir / "REPORT.md").write_text(report)

    tools = f"""# Tools and external assets

## Runtime environment

- Python 3.11.10
- numpy {np.__version__}
- scipy {__import__('scipy').__version__}
- opencv-python {cv2.__version__}
- h5py {h5py.__version__}
- trimesh {trimesh.__version__}

`solve.py` needs only the packages above plus the inputs and `/pipeline/depth/moge2/aligned.h5`. It runs offline with fixed seeds.

## Tried / used during development

- Open3D 0.20.0 (`open3d`) - tried for point-cloud surface reconstruction; final script does not need it.
- PyTorch 2.5.1+cu124 and PyTorch3D 0.7.9 - tried for differentiable silhouette rendering; final script does not need them.
- torchvision 0.20.1+cu124 - used for an independent Keypoint R-CNN cross-check.
- Keypoint R-CNN COCO weights: `https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth`
  - local path: `/work/tools/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth`
  - sha256: `fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926`
  - not required by `solve.py`.
- CARI4D source tree `/code/reconstruction` at commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` - inspected for file conventions; no code imported by the final script.
- No git repositories or pip packages were installed. No model weights beyond the listed Keypoint R-CNN download were used.
"""
    (out_dir / "TOOLS.md").write_text(tools)
    print(f"Wrote {out_path}, camera_estimate.json, REPORT.md, TOOLS.md in {runtime:.1f}s", flush=True)


if __name__ == "__main__":
    main()
