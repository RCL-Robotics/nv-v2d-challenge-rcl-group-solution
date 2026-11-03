# Diagnostic (uses GT): compare floor-based camera corrections with the oracle camera; AUC of each variant.
import os, sys, json, numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R, umeyama, MPPE_OFFSETS
V = os.environ["V2D_ROOT"]; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
cf = json.load(open(sys.argv[1]))
acc = {k: [] for k in ["s1", "rot_cam", "rot_hum", "rot_hum0", "oracle"]}
for i in range(30):
    m = meta[i]; d = f"{REL}/{m['sequence_id']}"
    Tw = np.load(f"{d}/poses.npy"); W = np.array(json.load(open(f"{d}/camera.json"))["T_world_cam"]); Tc = np.linalg.inv(W)[None] @ Tw
    gp = np.array(json.load(open(f"{d}/ground_plane.json"))["plane"]); nw = gp[:3]
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n = min(len(Tw), len(ref)); ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    g = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry(); np.random.seed(0); v = np.asarray(g.sample(1000))
    T = Tc[:n][ok]
    src = (T[:, None, :3, 3] + np.einsum("tij,kj->tki", T[:, :3, :3], MPPE_OFFSETS)).reshape(-1, 3)
    dst = (tr[:, None] + np.einsum("tij,kj->tki", Rr, MPPE_OFFSETS)).reshape(-1, 3)
    _, Ro, to = umeyama(src, dst, with_scale=False); Wo = np.eye(4); Wo[:3, :3] = Ro; Wo[:3, 3] = to
    # oracle correction expressed like the floor one: tilt of the camera "down" axis and height of the camera centre
    tilt_o = np.degrees(np.arccos(np.clip((W[:3, :3].T @ nw) @ (Wo[:3, :3].T @ nw), -1, 1)))
    h_o = nw @ (Wo[:3, 3] - W[:3, 3])
    c = cf[str(i)]; Wf = np.array(c["rot_cam"])
    tilt_f = np.degrees(np.arccos(np.clip((W[:3, :3].T @ nw) @ (Wf[:3, :3].T @ nw), -1, 1)))
    h_f = nw @ (Wf[:3, 3] - W[:3, 3])
    # agreement of the tilt directions (angle between the two rotation corrections' axes)
    def axis(Wn):
        a = np.cross(W[:3, :3].T @ nw, Wn[:3, :3].T @ nw); return a / (np.linalg.norm(a) + 1e-12)
    cosax = float(axis(Wo) @ axis(Wf))
    sc = {}
    for k, Wn in [("s1", W), ("rot_cam", Wf), ("rot_hum", np.array(c["rot_hum"])), ("rot_hum0", np.array(c["rot_hum0"])), ("oracle", Wo)]:
        Tn = Wn[None] @ T
        sc[k] = metrics(Tn[:, :3, :3], Tn[:, :3, 3], Rr, tr, v)["AUC"]; acc[k].append(sc[k])
    print("%2d tilt oracle %4.1f floor %4.1f axis-cos %+.2f | height oracle %+6.1f floor %+6.1f cm | AUC " % (i, tilt_o, tilt_f, cosax, h_o * 100, h_f * 100)
          + " ".join(f"{k} {x:.3f}" for k, x in sc.items()), flush=True)
print("mean", " ".join(f"{k} {np.mean(x):.3f}" for k, x in acc.items()))
