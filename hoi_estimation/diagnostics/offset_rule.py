# Diagnostic (scored with GT): GT-free rule "agent trajectory, shifted so its mean position equals s1's" (median also).
import sys, json, os, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
V = "/mnt/nas/Group/hanzhou/v2d"; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
out = {}
for i in range(30):
    m = meta[i]; s1 = np.load(f"{REL}/{m['sequence_id']}/poses.npy")
    p = f"{V}/agent_runs/T1_ds_ep{i:03d}/outputs/object_poses_world.npz"
    if not os.path.isfile(p): continue
    t1 = np.load(p)["T_world_mesh"]; n = min(len(s1), len(t1))
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); n = min(n, len(ref))
    ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    g = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry(); np.random.seed(0); v = np.asarray(g.sample(1000))
    row = {}
    for k, f in (("mean", np.mean), ("median", np.median)):
        h = t1[:n].copy(); h[:, :3, 3] += f(s1[:n, :3, 3] - t1[:n, :3, 3], axis=0)
        row[k] = metrics(h[ok][:, :3, :3], h[ok][:, :3, 3], Rr, tr, v)["AUC"]
    out[i] = row; print(i, row, flush=True)
json.dump(out, open(sys.argv[1], "w"))
