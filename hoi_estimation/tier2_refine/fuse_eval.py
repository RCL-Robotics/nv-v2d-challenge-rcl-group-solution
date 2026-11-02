#!/usr/bin/env python3
"""Map our camera-frame object trajectories (s1, Opus) into the Tier 2 world with the human-fitted camera and fuse them with
the Tier 2 trajectory. Variants (no GT used to build them):
  t2            Tier 2 as released
  map_X         our trajectory X fully mapped (rotation and position)
  t2pos_X       Tier 2 position + mapped rotation of X
  gate_X        Tier 2 position; mapped rotation of X where it is within TAU deg of the Tier 2 rotation, else Tier 2's
The diagnostics at the end compare against the Tier 1 reference (GT; evaluation only).
Usage: fuse_eval.py <cam.json> <out.json> [tau=45]"""
import os
import json, sys, numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R
from scipy.spatial.transform import Rotation as Rot
V = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"); DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
cams = json.load(open(sys.argv[1])); TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
def opus(i): return f"{V}/agent_runs/{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
def ang(A, B): return np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tji,tjk->tik", A, B)).as_rotvec(), axis=1))
res = {}
for i in range(30):
    m = meta[i]; sid = m["sequence_id"]; c = cams[str(i)]; s, Rc, tc = c["s"], np.array(c["R"]), np.array(c["t"])
    Wi = np.linalg.inv(np.array(json.load(open(f"{REL}/{sid}/camera.json"))["T_world_cam"]))
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    tr = {"s1": Wi[None] @ np.load(f"{REL}/{sid}/poses.npy"), "opus": Wi[None] @ np.load(opus(i))["T_world_mesh"]}
    n = min(len(t2), len(ref), *(len(x) for x in tr.values()))
    g = (np.linalg.norm(t2[:n, 3:], axis=1) > .5) & (np.linalg.norm(ref[:n, 3:], axis=1) > .5)
    R2 = quat_wxyz_to_R(t2[:n][g][:, 3:]); p2 = t2[:n][g][:, :3]; Rr = quat_wxyz_to_R(ref[:n][g][:, 3:]); pr = ref[:n][g][:, :3]
    np.random.seed(0); v = np.asarray(trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry().sample(1000))
    row = {"t2": [metrics(R2, p2, Rr, pr, v)["AUC"], float(np.median(ang(Rr, R2)))]}
    for k, T in tr.items():
        T = T[:n][g]; Rm = Rc[None] @ T[:, :3, :3]; pm = s * T[:, :3, 3] @ Rc.T + tc
        dis = ang(R2, Rm); use = dis < TAU; Rg = np.where(use[:, None, None], Rm, R2)
        row["map_" + k] = [metrics(Rm, pm, Rr, pr, v)["AUC"], float(np.median(ang(Rr, Rm)))]
        row["t2pos_" + k] = [metrics(Rm, p2, Rr, pr, v)["AUC"], row["map_" + k][1]]
        row["gate_" + k] = [metrics(Rg, p2, Rr, pr, v)["AUC"], float(np.median(ang(Rr, Rg))), float(use.mean())]
        row["posdiff_cm_" + k] = float(np.median(np.linalg.norm(pm - p2, axis=1)) * 100)
    res[i] = row
    print(i, m["object"][:14].ljust(14), " ".join(f"{k} {x[0]:.3f}/{x[1]:.0f}d" + (f"({x[2]:.0%})" if len(x) > 2 else "") if isinstance(x, list) else f"{k} {x:.1f}" for k, x in row.items()), flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
ks = [k for k in res[0] if isinstance(res[0][k], list)]
print("MEAN", " | ".join(f"{k} {np.mean([res[i][k][0] for i in res]):.3f}/{np.mean([res[i][k][1] for i in res]):.1f}d" for k in ks))
print("beats t2 (AUC +0.01):", {k: sum(res[i][k][0] > res[i]["t2"][0] + .01 for i in res) for k in ks if k != "t2"}, "worse:", {k: sum(res[i][k][0] < res[i]["t2"][0] - .01 for i in res) for k in ks if k != "t2"})
