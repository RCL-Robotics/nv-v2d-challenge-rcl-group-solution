"""Blend CoCoNet depth with the FoundationPose input depth: |t| *= (|t_in|/|t_pr|)^w (per-episode median or filtered per-frame)."""
import json, sys
import numpy as np, torch, trimesh
from scipy.ndimage import median_filter
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
V2D = "/mnt/nas/Group/hanzhou/v2d"; ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/JC1_joint_camera_align.json"))
t2 = {int(l.split(",")[0]): float(l.split(",")[10]) for l in open(f"{V2D}/logs/officialJC1.csv").read().splitlines()[1:]}
WS = [0.0, 0.25, 0.5, 0.75]
res = {}
for idx in range(30):
    ep = f"episode_{idx:06d}"
    b = torch.load(f"{V2D}/experiments/JC1_joint_camera/{ep}/coconet.pth", map_location="cpu", weights_only=False)
    P = np.asarray(b["pr"]["pose_abs"], np.float64); Pi = np.asarray(b["in"]["pose_abs"], np.float64)
    Pp = np.asarray(b["pr_initial"]["pose_abs"], np.float64)  # unscaled CoCoNet output (pr before JC1 is pr_initial~pr)
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); verts = np.asarray(glb.sample(1000))
    W = np.asarray(al[str(idx)]["sim"]); Rw, C = W[:3, :3], W[:3, 3]
    T = P @ S; tc = T[:, :3, 3]
    r = np.linalg.norm((Pi @ S)[:, :3, 3], axis=1) / np.linalg.norm((Pp @ S)[:, :3, 3], axis=1)
    r = np.where(np.isfinite(r) & (r > 0.5) & (r < 2), r, 1.0)
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
    n = min(len(P), len(ref)); ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    ev = lambda t: metrics((Rw[None] @ T[:, :3, :3])[:n][ok], (t @ Rw.T + C)[:n][ok], Rr, tr, verts)["AUC"]
    row = {}
    for w in WS:
        row[f"c{w}"] = ev(tc * np.median(r) ** w)
        row[f"f{w}"] = ev(tc * (median_filter(r, size=31, mode="nearest") ** w)[:, None])
    res[idx] = row
    print(f"{idx:2d} {meta[idx]['object'][:10]:10s} r_med {np.median(r):.3f} " + " ".join(f"{k}:{v:.3f}" for k, v in row.items()) + f" t2 {t2[idx]:.3f}", flush=True)
for k in res[0]:
    a = np.array([res[i][k] for i in range(30)])
    print(k, f"mean {a.mean():.3f} beat {sum(res[i][k] > t2[i] for i in range(30))}/30")
