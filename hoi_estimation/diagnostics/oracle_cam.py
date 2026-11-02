# Diagnostic only (uses GT): how much AUC would s1 gain with an oracle camera (one rigid / similarity per episode)?
import os, sys, json, numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, align_and_score, quat_wxyz_to_R
V = os.environ["V2D_ROOT"]; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
rows = []
for i in range(30):
    m = meta[i]; d = f"{REL}/{m['sequence_id']}"
    Tw = np.load(f"{d}/poses.npy"); W = np.array(json.load(open(f"{d}/camera.json"))["T_world_cam"])
    Tc = np.linalg.inv(W)[None] @ Tw
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n = min(len(Tw), len(ref)); ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    g = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry()
    np.random.seed(0); v = np.asarray(g.sample(1000))
    a = metrics(Tw[:n][ok][:, :3, :3], Tw[:n][ok][:, :3, 3], Rr, tr, v)
    r = align_and_score(Tc[:n][ok], Rr, tr, v, "rigid"); s = align_and_score(Tc[:n][ok], Rr, tr, v, "sim3")
    # camera error of the estimate vs the oracle rigid camera: rotation (deg) and centre (cm)
    rows.append((i, m["object"][:18], a["AUC"], a["pos_cm"], a["rot_deg"], r["AUC"], r["pos_cm"], r["rot_deg"], s["AUC"], s["pos_cm"], s["fit_scale"]))
    print("%2d %-18s s1 %.3f (%5.1fcm %5.1fdeg) | oracle-rigid %.3f (%5.1fcm %5.1fdeg) | oracle-sim %.3f (%5.1fcm, scale %.3f)" % rows[-1], flush=True)
A = np.array([r[2:] for r in rows], float)
print("mean s1 %.3f  oracle-rigid %.3f  oracle-sim %.3f | pos s1 %.1f rigid %.1f sim %.1f cm" % (A[:, 0].mean(), A[:, 3].mean(), A[:, 6].mean(), A[:, 1].mean(), A[:, 4].mean(), A[:, 7].mean()))
