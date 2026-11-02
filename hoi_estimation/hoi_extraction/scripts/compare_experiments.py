#!/usr/bin/env python3
"""Compare object poses of the baseline CARI4D run against FoundationPose-mode experiments.

Runs inside the v2d_cari4d image. For every (episode, variant, stage) it reports
  official-style metrics after per-episode SE3 alignment (eval_object_pose.align_and_score)
  robust diagnostics (diagnose_episode.robust_offset): median rotation error and the
  fraction of frames under 30 deg after a RANSAC camera->world rotation.
Diagnostic only: alignment uses the Tier 1 reference.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from diagnose_episode import ang, q2R, robust_offset  # noqa: E402
from eval_object_pose import align_and_score  # noqa: E402


def load_variant(path_fp, path_coco, path_refined=None):
    out = {}
    if path_fp and os.path.isfile(path_fp):
        out["FP"] = np.asarray(pickle.load(open(path_fp, "rb"))["fp_poses_cam"])[:, 0].astype(np.float64)
    if path_coco and os.path.isfile(path_coco):
        out["CoCoNet"] = np.asarray(torch.load(path_coco, map_location="cpu", weights_only=False)["pr"]["pose_abs"], dtype=np.float64)
    if path_refined and os.path.isfile(path_refined):
        out["Refine"] = np.asarray(torch.load(path_refined, map_location="cpu", weights_only=False)["pr"]["pose_abs_postopt"], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", help="experiment names under <v2d>/experiments")
    ap.add_argument("--v2d", default="/mnt/nas/Group/hanzhou/v2d")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()
    ds = f"{args.v2d}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    eps = sorted({int(p.rsplit("_", 1)[1]) for e in args.experiments for p in glob.glob(f"{args.v2d}/experiments/{e}/episode_*")
                  if any(os.path.isfile(f"{p}/{f}") for f in ("fp.pkl", "coconet.pth", "refined.pth"))})
    rows = []
    for idx in eps:
        ep = f"episode_{idx:06d}"
        base = f"{args.v2d}/outputs/cari4d/{ep}"
        base_fp = glob.glob(f"{base}/foundationpose/*.pkl")  # absent when the baseline run failed (e.g. ep 25)
        variants = {"baseline": load_variant(base_fp[0], f"{base}/inference/coconet.pth", f"{base}/inference/refined.pth")} if base_fp else {}
        for e in args.experiments:
            variants[e] = load_variant(f"{args.v2d}/experiments/{e}/{ep}/fp.pkl", f"{args.v2d}/experiments/{e}/{ep}/coconet.pth", f"{args.v2d}/experiments/{e}/{ep}/refined.pth")
        pose = np.load(f"{args.v2d}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
        ok = np.linalg.norm(pose[:, 3:], axis=1) > 0.5
        pose[~ok, 3] = 1
        t_r, R_r = pose[:, :3], q2R(pose[:, 3:])
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], dtype=np.float64)
        verts = np.asarray(glb.sample(1000), dtype=np.float64)
        for vname, stages in variants.items():
            for stage, P in stages.items():
                T = P @ S
                n = min(len(T), len(t_r))
                keep = ok[:n] & np.isfinite(T[:n]).all(axis=(1, 2))
                m = align_and_score(T[:n][keep], R_r[:n][keep], t_r[:n][keep], verts, "se3")
                C, inl = robust_offset(T[:n][keep][:, :3, :3], R_r[:n][keep])
                rot = ang(C @ T[:n][keep][:, :3, :3], R_r[:n][keep])
                rows.append({"episode": idx, "object": meta[idx]["object"], "variant": vname, "stage": stage,
                             "AUC": m["AUC"], "MPPE_cm": m["MPPE_cm"], "pos_cm": m["pos_cm"], "rot_deg": m["rot_deg"],
                             "SP": m["SP"], "MP": m["MP"], "robust_rot_med": float(np.median(rot)), "frac_rot_lt30": float((rot < 30).mean())})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    for stage in ("FP", "CoCoNet", "Refine"):
        sub = df[df.stage == stage]
        if sub.empty:
            continue
        print(f"\n=== {stage}: frac of frames with rotation error < 30 deg (robust) ===")
        print(sub.pivot_table(index=["episode", "object"], columns="variant", values="frac_rot_lt30").round(2).to_string())
        print(f"\n=== {stage}: MPPE cm (SE3-aligned) ===")
        print(sub.pivot_table(index=["episode", "object"], columns="variant", values="MPPE_cm").round(1).to_string())
        print(f"\n=== {stage}: mean over episodes ===")
        print(sub.groupby("variant")[["AUC", "MPPE_cm", "pos_cm", "rot_deg", "SP", "MP", "robust_rot_med", "frac_rot_lt30"]].mean().round(3).to_string())
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
