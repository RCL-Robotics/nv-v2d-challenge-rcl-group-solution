#!/usr/bin/env python3
"""Diagnostic (uses GT): ADD AUC 1-9 cm of our best Tier 3 trajectory before and after the support snap."""
import json, sys, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
from common import *
DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; TRAJ = f"{V}/experiments/t3_frame0/opus_f3_c2_shift"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}; out = {}
for i in SUPPORT:
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ok = np.linalg.norm(ref[:, 3:], axis=1) > .5
    np.random.seed(0); v = np.asarray(trimesh.load(f"{DS}/{meta[i]['mesh']}", force="scene", process=False).to_geometry().sample(1000))
    r = []
    for p in (f"{TRAJ}/episode_{i:06d}.npz", f"{OUT}/ep{i:03d}/object_poses_world_snapped.npz"):
        T = np.load(p)["T_world_mesh"].astype(float); n = min(len(T), len(ref)); k = ok[:n]
        r.append(metrics(T[:n][k][:, :3, :3], T[:n][k][:, :3, 3], quat_wxyz_to_R(ref[:n][k][:, 3:]), ref[:n][k][:, :3], v)["AUC"])
    out[i] = r; print(f"{i:2d} {meta[i]['object'][:18]:18s} AUC {r[0]:.3f} -> snapped {r[1]:.3f}")
A = np.array(list(out.values())); print(f"MEAN {A[:,0].mean():.3f} -> {A[:,1].mean():.3f}")
json.dump({str(k): v for k, v in out.items()}, open(f"{OUT}/snap_scores_diag.json", "w"))
