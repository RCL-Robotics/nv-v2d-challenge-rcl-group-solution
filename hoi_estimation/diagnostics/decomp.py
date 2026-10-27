# Diagnostic (uses GT): s1 world error of the object centre, expressed in camera axes (x right, y down, z depth) and as
# a per-episode constant + time-varying part; also AUC if only the constant depth (z) offset were removed.
import sys, json, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
V = "/mnt/nas/Group/hanzhou/v2d"; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
A = []
for i in range(30):
    m = meta[i]; d = f"{REL}/{m['sequence_id']}"
    Tw = np.load(f"{d}/poses.npy"); W = np.array(json.load(open(f"{d}/camera.json"))["T_world_cam"])
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n = min(len(Tw), len(ref)); ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    g = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry(); np.random.seed(0); v = np.asarray(g.sample(1000))
    T = Tw[:n][ok]; c = g.centroid
    pa = T[:, :3, :3] @ c + T[:, :3, 3]; pr = Rr @ c + tr          # object centroid, ours / GT (world)
    Rcw = W[:3, :3].T; ea = (pa - pr) @ Rcw.T                       # error in camera axes
    # along-ray (radial) error per frame: |p_ours - C| - |p_gt - C|
    C = W[:3, 3]; rad = np.linalg.norm(pa - C, axis=1) - np.linalg.norm(pr - C, axis=1)
    mean_e = ea.mean(0); var_e = np.sqrt(((ea - mean_e) ** 2).sum(1).mean())
    # remove only a constant radial (along-ray) offset: scale each frame's position about C
    k = 1 - rad.mean() / np.linalg.norm(pa - C, axis=1)
    Tz = T.copy(); Tz[:, :3, 3] = C + (T[:, :3, 3] - C) * k[:, None]
    # remove the full constant offset (in world) of the centroid
    To = T.copy(); To[:, :3, 3] -= (pa - pr).mean(0)
    a0 = metrics(T[:, :3, :3], T[:, :3, 3], Rr, tr, v)["AUC"]; az = metrics(Tz[:, :3, :3], Tz[:, :3, 3], Rr, tr, v)["AUC"]
    ao = metrics(To[:, :3, :3], To[:, :3, 3], Rr, tr, v)["AUC"]
    A.append((a0, az, ao, *np.abs(mean_e), var_e))
    print("%2d %-16s err cam-axes mean x %+5.1f y %+5.1f z %+5.1f cm | radial mean %+5.1f sd %4.1f | varying %4.1f cm | AUC s1 %.3f -radial %.3f -const %.3f"
          % (i, m["object"][:16], *(mean_e * 100), rad.mean() * 100, rad.std() * 100, var_e * 100, a0, az, ao), flush=True)
A = np.array(A)
print("mean AUC s1 %.3f | minus const radial %.3f | minus const 3D offset %.3f | |mean err| x %.1f y %.1f z %.1f cm, varying %.1f cm"
      % (*A[:, :3].mean(0), *(A[:, 3:6].mean(0) * 100), A[:, 6].mean() * 100))
