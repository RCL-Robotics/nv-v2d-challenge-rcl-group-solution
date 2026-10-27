# Diagnostic only (uses GT): upper bound from an ideal camera extrinsic. One rigid (or similarity = rigid + depth
# scale) world transform per episode, fitted to the GT over the whole episode, applied on top of each trajectory.
import sys, json, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, align_and_score, quat_wxyz_to_R
V = "/mnt/nas/Group/hanzhou/v2d"; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
def opus(i): return f"{V}/agent_runs/{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
src = {"opus_v2": opus, "frame0_best": lambda i: f"{V}/experiments/t3_frame0/opus_f3_c2_shift/episode_{i:06d}.npz"}
res = {}
for i in range(30):
    m = meta[i]; ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    g = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry(); np.random.seed(0); v = np.asarray(g.sample(1000))
    for k, f in src.items():
        Tw = np.load(f(i))["T_world_mesh"].astype(float); n = min(len(Tw), len(ref)); ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
        Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]; T = Tw[:n][ok]
        a = metrics(T[:, :3, :3], T[:, :3, 3], Rr, tr, v)
        r = align_and_score(T, Rr, tr, v, "rigid"); s = align_and_score(T, Rr, tr, v, "sim3")
        res.setdefault(i, {})[k] = dict(raw=a["AUC"], rigid=r["AUC"], sim3=s["AUC"], rot_raw=a["rot_deg"], rot_rigid=r["rot_deg"], pos_raw=a["pos_cm"], pos_rigid=r["pos_cm"], scale=s["fit_scale"])
    print(i, m["object"][:14], " ".join(f"{k} {x['raw']:.3f}/{x['rigid']:.3f}/{x['sim3']:.3f}" for k, x in res[i].items()), flush=True)
for k in src:
    print(k, "mean raw/rigid/sim3", *[round(np.mean([res[i][k][c] for i in res]), 3) for c in ("raw", "rigid", "sim3")])
json.dump(res, open(sys.argv[1], "w"), indent=1)
