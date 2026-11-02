#!/usr/bin/env python3
"""Fit the static exo camera of each episode to the Tier 2 (noisy, world-frame) object trajectory, using our camera-frame
object trajectory (s1: T_cam_mesh = inv(T_world_cam_s1) @ poses) as the image-side track. No GT.
  rotation: chordal average of R_t2 R_cam^T over visible frames (Tier 2 rotation noise is slow but varies over time)
  translation + scale: least squares t_t2 = s * Rc t_cam + c   (s absorbs our depth-scale error)
Then evaluates (diagnostic, GT) rotation error / ADD AUC of: Tier 2, s1 and Opus mapped with this camera, and the hybrid
Tier 2 translation + mapped rotation.
Usage: cam_from_t2.py <out.json> [eps]"""
import json, sys, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
from scipy.spatial.transform import Rotation as Rot
V = "/mnt/nas/Group/hanzhou/v2d"; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
def opus(i): return f"{V}/agent_runs/{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
def proj_so3(M): u, _, vt = np.linalg.svd(M); return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
def roterr(Ra, Rb): return np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tji,tjk->tik", Ra, Rb)).as_rotvec(), axis=1))
eps = list(map(int, sys.argv[2].split(","))) if len(sys.argv) > 2 else range(30); res = {}
for i in eps:
    m = meta[i]; sid = m["sequence_id"]
    W = np.array(json.load(open(f"{REL}/{sid}/camera.json"))["T_world_cam"]); Wi = np.linalg.inv(W)
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    ok = np.linalg.norm(t2[:, 3:], axis=1) > 0.5; F = len(t2)
    R2 = np.tile(np.eye(3), (F, 1, 1)); R2[ok] = quat_wxyz_to_R(t2[ok, 3:]); p2 = t2[:, :3]
    tracks = {"s1": Wi[None] @ np.load(f"{REL}/{sid}/poses.npy"), "opus": Wi[None] @ np.load(opus(i))["T_world_mesh"]}
    cams = {}
    for k, Tc in tracks.items():
        Tc = Tc[:F]; o = ok[:len(Tc)]
        Rc = proj_so3(np.einsum("tij,tkj->ik", R2[:len(Tc)][o], Tc[o, :3, :3]))
        a = Tc[o, :3, 3] @ Rc.T; b = p2[:len(Tc)][o]; am, bm = a.mean(0), b.mean(0)
        s = ((a - am) * (b - bm)).sum() / ((a - am) ** 2).sum(); c = bm - s * am
        cams[k] = (Rc, s, c)
    # diagnostics with GT
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n = min(F, len(ref), *(len(t) for t in tracks.values())); g = ok[:n] & (np.linalg.norm(ref[:n, 3:], axis=1) > 0.5)
    Rr = quat_wxyz_to_R(ref[:n][g][:, 3:]); tr = ref[:n][g][:, :3]
    np.random.seed(0); v = np.asarray(trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry().sample(1000))
    row = {"t2": [metrics(R2[:n][g], p2[:n][g], Rr, tr, v)["AUC"], float(np.median(roterr(Rr, R2[:n][g])))]}
    for k, (Rc, s, c) in cams.items():
        Tc = tracks[k][:n][g]; Rm = Rc[None] @ Tc[:, :3, :3]; pm = s * Tc[:, :3, 3] @ Rc.T + c
        row[k] = [metrics(Rm, pm, Rr, tr, v)["AUC"], float(np.median(roterr(Rr, Rm)))]
        row[k + "+t2pos"] = [metrics(Rm, p2[:n][g], Rr, tr, v)["AUC"], row[k][1]]
        row[k + "_scale"] = s
    res[i] = row
    print(i, m["object"][:14].ljust(14), " ".join(f"{k} {val[0]:.3f}/{val[1]:.0f}d" if isinstance(val, list) else f"{k} {val:.2f}" for k, val in row.items()), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
ks = [k for k in res[next(iter(res))] if not k.endswith("_scale")]
print("MEAN", " ".join(f"{k} {np.mean([res[i][k][0] for i in res]):.3f}/{np.mean([res[i][k][1] for i in res]):.1f}d" for k in ks))
