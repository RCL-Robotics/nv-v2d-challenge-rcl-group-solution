#!/usr/bin/env python3
"""Diagnostic evaluation of CARI4D object poses against the Track 2 Tier 1 reference.

Runs inside the v2d_cari4d image (needs torch, trimesh, pandas, h5py).

Our poses live in the camera frame and the reference in the released Track 2 world frame,
and no camera calibration is released. For diagnosis we therefore fit one camera->world
transform per episode on the evaluated trajectory itself (optimistic, NOT a submission
procedure) and report the official Track 2 metric definitions after alignment:
  se3    rigid alignment of the trajectory as produced
  sim3   rigid + scale; the fitted scale shows the depth-scale error
  ground per-episode depth scale from the released ground plane, then rigid alignment
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
import trimesh

ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
MPPE_OFFSETS = 0.05 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)


def quat_wxyz_to_R(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (xs ** 2).sum(1).mean()) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def metrics(R_a, t_a, R_r, t_r, verts):
    pos = np.linalg.norm(t_a - t_r, axis=-1)
    cos = np.clip((np.einsum("tij,tij->t", R_a, R_r) - 1) / 2, -1, 1)
    rot = np.arccos(cos)
    add = np.linalg.norm(np.einsum("tij,vj->tvi", R_a - R_r, verts) + (t_a - t_r)[:, None], axis=-1).mean(1)
    trap = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trap([np.mean(add < t) for t in ADD_THRESHOLDS_M], x=np.linspace(0, 1, ADD_THRESHOLDS_M.size)))
    kp_a = t_a[:, None] + np.einsum("tij,kj->tki", R_a, MPPE_OFFSETS)
    kp_r = t_r[:, None] + np.einsum("tij,kj->tki", R_r, MPPE_OFFSETS)
    mppe = float(np.linalg.norm(kp_a - kp_r, axis=-1).mean() * 100)
    return {"AUC": auc, "MPPE_cm": mppe, "pos_cm": float(pos.mean() * 100), "rot_deg": float(np.degrees(rot.mean())),
            "SP": bool(pos.mean() <= 0.10 and rot.mean() <= 0.5), "MP": bool(pos.mean() < 0.03 and np.degrees(rot.mean()) < 30)}


def align_and_score(T_cam, R_r, t_r, verts, mode, depth_scale=1.0):
    R_c, t_c = T_cam[:, :3, :3], T_cam[:, :3, 3] * depth_scale
    src = (t_c[:, None] + np.einsum("tij,kj->tki", R_c, MPPE_OFFSETS)).reshape(-1, 3)
    dst = (t_r[:, None] + np.einsum("tij,kj->tki", R_r, MPPE_OFFSETS)).reshape(-1, 3)
    s, R, t = umeyama(src, dst, with_scale=(mode == "sim3"))
    out = metrics(R @ R_c, s * t_c @ R.T + t, R_r, t_r, verts)
    out["fit_scale"] = s * depth_scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=os.environ["V2D_ROOT"] + "/outputs/cari4d")
    ap.add_argument("--dataset", default=os.environ["V2D_ROOT"] + "/dataset/track_2/tier_1_multiview_caption")
    ap.add_argument("--ref-poses", default=os.environ["V2D_ROOT"] + "/ref_poses")
    ap.add_argument("--ground-scales", default=None, help="json {episode_index: aligned-depth correction factor}")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{args.dataset}/meta/episodes_metadata.jsonl"))}
    gscale = {int(k): v for k, v in json.load(open(args.ground_scales)).items()} if args.ground_scales else {}
    rows = []
    for ep_dir in sorted(glob.glob(f"{args.outputs}/episode_*")):
        idx = int(ep_dir.rsplit("_", 1)[1])
        refined = f"{ep_dir}/inference/refined.pth"
        if not os.path.isfile(refined):
            continue
        # reference poses pre-exported from the Tier 1 parquet (the image has no parquet engine)
        pose = np.load(f"{args.ref_poses}/episode_{idx:06d}.npz")["pose_xyz_wxyz"].astype(np.float64)
        ref_ok = np.linalg.norm(pose[:, 3:], axis=1) > 0.5  # a few reference frames carry an all-zero quaternion
        pose[~ref_ok, 3] = 1.0
        t_r, R_r = pose[:, :3], quat_wxyz_to_R(pose[:, 3:])
        glb = trimesh.load(f"{args.dataset}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        src_to_aligned, _ = trimesh.bounds.oriented_bounds(glb)
        verts = np.asarray(glb.sample(1000, seed=0) if "seed" in trimesh.sample.sample_surface.__code__.co_varnames else glb.sample(1000), dtype=np.float64)
        b = torch.load(refined, map_location="cpu", weights_only=False)
        stages = {"foundationpose": b["in"]["pose_abs"], "coconet": b["pr"]["pose_abs"], "final": b["pr"]["pose_abs_postopt"]}
        for stage, P in stages.items():
            T = np.asarray(P, dtype=np.float64) @ np.asarray(src_to_aligned, dtype=np.float64)  # cam_T_glb
            n = min(len(T), len(t_r))
            keep = ref_ok[:n] & np.isfinite(T[:n]).all(axis=(1, 2))
            modes = [("se3", 1.0), ("sim3", 1.0)] + ([("ground", gscale[idx])] if idx in gscale else [])
            for mode, ds in modes:
                m = align_and_score(T[:n][keep], R_r[:n][keep], t_r[:n][keep], verts, "sim3" if mode == "sim3" else "se3", ds)
                rows.append({"episode": idx, "object": meta[idx]["object"], "stage": stage, "align": mode, **m})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    print("\nmean over episodes:")
    print(df.groupby(["stage", "align"])[["AUC", "MPPE_cm", "pos_cm", "rot_deg", "SP", "MP", "fit_scale"]].mean().round(3))
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
