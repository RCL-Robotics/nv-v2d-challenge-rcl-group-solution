#!/usr/bin/env python3
"""Score the released Tier 2 (synthetic-noise) object trajectories against the Tier 1 reference with
the same metric code as eval_object_pose.py: raw (as the leaderboard scores it) and SE3-aligned."""
import json, numpy as np, pandas as pd, trimesh, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from eval_object_pose import quat_wxyz_to_R, metrics, align_and_score
V2D = "/mnt/nas/Group/hanzhou/v2d"
ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
rows = []
for idx in range(30):
    ep = f"episode_{idx:06d}"
    r = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(float)
    t2 = np.load(f"{V2D}/tier2_poses/{ep}.npz")["pose_xyz_wxyz"].astype(float)
    ok = (np.linalg.norm(r[:, 3:], axis=1) > 0.5) & (np.linalg.norm(t2[:, 3:], axis=1) > 0.5)
    r, t2 = r[ok], t2[ok]
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    verts = np.asarray(glb.sample(1000), dtype=float)
    Rr, tr, R2, tt2 = quat_wxyz_to_R(r[:, 3:]), r[:, :3], quat_wxyz_to_R(t2[:, 3:]), t2[:, :3]
    raw = metrics(R2, tt2, Rr, tr, verts)
    T = np.tile(np.eye(4), (len(r), 1, 1)); T[:, :3, :3] = R2; T[:, :3, 3] = tt2
    se3 = align_and_score(T, Rr, tr, verts, "se3")
    rows += [{"episode": idx, "object": meta[idx]["object"], "align": "raw", **raw},
             {"episode": idx, "object": meta[idx]["object"], "align": "se3", **se3}]
pd.DataFrame(rows).to_csv(f"{V2D}/logs/eval_tier2.csv", index=False)
print("done")
