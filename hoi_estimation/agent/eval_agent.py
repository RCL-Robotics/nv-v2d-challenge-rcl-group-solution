#!/usr/bin/env python3
"""Score an agent's T_world_mesh (F,4,4) with the official object-pose protocol (same as eval_official.py). Runs OUTSIDE the agent sandbox."""
import os
import argparse, json, sys
import numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R

V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"); DS = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
ap = argparse.ArgumentParser(); ap.add_argument("episode", type=int); ap.add_argument("poses", help="npz with T_world_mesh"); ap.add_argument("--key", default="T_world_mesh"); ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
ep = f"episode_{a.episode:06d}"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}[a.episode]
glb = trimesh.load(f"{DS}/{meta['mesh']}", force="scene", process=False).dump(concatenate=True)
S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], np.float64)
Tw = np.asarray(np.load(a.poses)[a.key], np.float64)  # world_T_glb (mesh-file frame), as the reference
ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
n = min(len(Tw), len(ref)); ok = (np.linalg.norm(ref[:n, 3:], axis=1) > 0.5) & np.isfinite(Tw[:n]).all(axis=(1, 2))
ms = []
for seed in range(a.seeds):  # official script samples 1000 surface points at random; fix and average the samples
    np.random.seed(seed)
    verts = np.asarray(glb.sample(1000), np.float64)
    ms.append(metrics(Tw[:n][ok][:, :3, :3], Tw[:n][ok][:, :3, 3], quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3], verts))
print(json.dumps({k: float(np.mean([m[k] for m in ms])) for k in ("AUC", "MPPE_cm", "pos_cm", "rot_deg")} | {"AUC_per_seed": [round(float(m["AUC"]), 4) for m in ms]}))
