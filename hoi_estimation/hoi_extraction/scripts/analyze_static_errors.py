#!/usr/bin/env python3
"""Where do the remaining E2 rotation errors happen? Split frames into reference-static vs moving
(diagnostic: uses Tier 1 motion) and report rotation error, its within-segment variation, and
how often our pose jumps (>45 deg between consecutive frames)."""
import glob, json, os, sys
import numpy as np, torch, trimesh
sys.path.insert(0, os.path.dirname(__file__))
from diagnose_episode import ang, q2R, robust_offset
V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"); ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
for idx in map(int, sys.argv[1:]):
    ep = f"episode_{idx:06d}"
    P = np.asarray(torch.load(f"{V2D}/experiments/E2_reregister/{ep}/coconet.pth", map_location="cpu", weights_only=False)["pr"]["pose_abs"], dtype=float)
    pose = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(float)
    ok = np.linalg.norm(pose[:, 3:], axis=1) > 0.5; pose[~ok, 3] = 1
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], dtype=float)
    n = min(len(P), len(pose)); T = (P @ S)[:n]; R_r = q2R(pose[:n, 3:]); t_r = pose[:n, :3]
    C, inl = robust_offset(T[:, :3, :3], R_r)
    rot = ang(C @ T[:, :3, :3], R_r)
    # reference motion: static if the object moves < 1 cm and < 2 deg over +-5 frames
    k = 5; sp = np.zeros(n, bool)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        sp[i] = np.linalg.norm(t_r[b] - t_r[a]) < 0.01 and ang(R_r[b], R_r[a]) < 2
    jumps = ang(T[1:, :3, :3], T[:-1, :3, :3]) > 45
    # rotation drift of OUR pose inside reference-static frames (would be ~0 if we were stable)
    our_step = np.r_[0, ang(T[1:, :3, :3], T[:-1, :3, :3])]
    print(f"ep{idx:<3}{meta[idx]['object'][:16]:<17} static {sp.mean():.2f} | rot med static {np.median(rot[sp]) if sp.any() else np.nan:6.1f} moving {np.median(rot[~sp]) if (~sp).any() else np.nan:6.1f} "
          f"| our step in static med {np.median(our_step[sp]) if sp.any() else np.nan:5.2f} p95 {np.percentile(our_step[sp],95) if sp.any() else np.nan:5.1f} | jumps>45 {int(jumps.sum())} | frac<30 static {np.mean(rot[sp]<30) if sp.any() else np.nan:.2f} moving {np.mean(rot[~sp]<30):.2f}")
