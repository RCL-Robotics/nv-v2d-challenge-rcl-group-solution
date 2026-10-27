# Diagnostic only (uses GT): raw / oracle-rigid / oracle-sim3 AUC for s1 vs agent outputs (camera-frame part kept, one transform fitted per episode).
import sys, json, os, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, align_and_score, quat_wxyz_to_R
V = "/mnt/nas/Group/hanzhou/v2d"; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
def agent(ws):
    p = f"{V}/agent_runs/{ws}/outputs/object_poses_world.npz"
    return np.load(p)["T_world_mesh"] if os.path.isfile(p) else None
res = {}
for i in [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(30):
    m = meta[i]; d = f"{REL}/{m['sequence_id']}"
    s1 = np.load(f"{d}/poses.npy"); W = np.array(json.load(open(f"{d}/camera.json"))["T_world_cam"]); Wi = np.linalg.inv(W)
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    g = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry()
    np.random.seed(0); v = np.asarray(g.sample(1000))
    meth = {"s1": s1}
    t1 = agent(f"T1_ds_ep{i:03d}")
    if t1 is not None:
        meth["T1"] = t1
        h = s1.copy(); h[:len(t1), :3, :3] = t1[:len(s1), :3, :3]; meth["T1rot"] = h
    for k, ws in (("C1", f"C1_ds_ep{i:03d}"), ("CL1", f"CL1_ep{i:03d}")):
        a = agent(ws)
        if a is not None: meth[k] = a
    for k, Tw in meth.items():
        n = min(len(Tw), len(ref)); ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
        Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
        Tc = Wi[None] @ Tw[:n][ok]
        a = metrics(Tw[:n][ok][:, :3, :3], Tw[:n][ok][:, :3, 3], Rr, tr, v)
        r = align_and_score(Tc, Rr, tr, v, "rigid"); s = align_and_score(Tc, Rr, tr, v, "sim3")
        res.setdefault(i, {})[k] = (a["AUC"], r["AUC"], s["AUC"], a["rot_deg"], r["pos_cm"])
    print(i, {k: "%.3f/%.3f/%.3f" % x[:3] for k, x in res[i].items()}, flush=True)
json.dump(res, open(sys.argv[1], "w"))
