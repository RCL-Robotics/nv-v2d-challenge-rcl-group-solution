"""Diagnostic (uses GT, analysis only): split object position error into along-ray depth vs lateral."""
import json, os, sys
import numpy as np, torch, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
from eval_official import bundle_path

V2D = "/mnt/nas/Group/hanzhou/v2d"
ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/pnp_forE1.json"))
pipe = sys.argv[1].split(",")
eps = [int(x) for x in sys.argv[2].split(",")]
print("ep obj        AUC  AUCk  AUCd  AUCl | pos  lat  dep | k*    r_std r_f0  r_trend")
for idx in eps:
    ep = f"episode_{idx:06d}"
    v = next(p for p in pipe if os.path.isfile(bundle_path(p, ep)))
    P = np.asarray(torch.load(bundle_path(v, ep), map_location="cpu", weights_only=False)["pr"]["pose_abs"], dtype=np.float64)
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); verts = np.asarray(glb.sample(1000))
    T = P @ S
    W = np.asarray(al[str(idx)]["sim"]); s = np.cbrt(np.linalg.det(W[:3, :3])); Rw = W[:3, :3] / s; tw = W[:3, 3]
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
    n = min(len(T), len(ref)); ok = (np.linalg.norm(ref[:n, 3:], axis=1) > 0.5) & np.isfinite(T[:n]).all(axis=(1, 2))
    T, ref = T[:n][ok], ref[:n][ok]
    Rr, tr = quat_wxyz_to_R(ref[:, 3:]), ref[:, :3]
    # reference in our (scaled) camera frame: x_cam = Rw^T (x_w - tw) / s
    tr_c = (tr - tw) @ Rw / s
    tc = T[:, :3, 3]
    Rw_est = Rw[None] @ T[:, :3, :3]
    def score(tcam):
        return metrics(Rw_est, s * tcam @ Rw.T + tw, Rr, tr, verts)
    r = np.linalg.norm(tc, axis=1) / np.linalg.norm(tr_c, axis=1)
    k = np.median(1 / r)
    u = tc / np.linalg.norm(tc, axis=1, keepdims=True)
    e = tc - tr_c
    dep = np.abs((e * u).sum(1)); lat = np.linalg.norm(e - (e * u).sum(1, keepdims=True) * u, axis=1)
    m0 = score(tc); mk = score(tc * k); md = score(tc / r[:, None])
    ml = score(tr_c)  # perfect position, our rotation
    tt = np.arange(len(r)); slope = np.polyfit(tt / len(r), r, 1)[0]
    print(f"{idx:2d} {meta[idx]['object'][:10]:10s} {m0['AUC']:.3f} {mk['AUC']:.3f} {md['AUC']:.3f} {ml['AUC']:.3f} | "
          f"{m0['pos_cm']:4.1f} {lat.mean()*100*s:4.1f} {dep.mean()*100*s:4.1f} | {k:.3f} {r.std():.3f} {r[:5].mean():.3f} {slope:+.3f}")
