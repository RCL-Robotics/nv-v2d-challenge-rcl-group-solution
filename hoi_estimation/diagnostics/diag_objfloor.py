"""DIAGNOSTIC: lowest object vertex height above the released plane, GT vs JC1."""
import os, json, sys
import numpy as np, torch, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import quat_wxyz_to_R
V2D = os.environ["V2D_ROOT"]; ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/JC1_joint_camera_align.json"))
for idx in range(30):
    ep = f"episode_{idx:06d}"
    b = torch.load(f"{V2D}/experiments/JC1_joint_camera/{ep}/coconet.pth", map_location="cpu", weights_only=False)
    P = np.asarray(b["pr"]["pose_abs"], np.float64)
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); V = np.asarray(glb.vertices)[::max(1, len(glb.vertices)//3000)]
    Vh = np.c_[V, np.ones(len(V))]
    W = np.asarray(al[str(idx)]["sim"]); C = W[:3, 3]
    gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"]); nrm = np.linalg.norm(gp[:3]); n, d = gp[:3]/nrm, gp[3]/nrm
    if n @ C + d < 0: n, d = -n, -d
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
    m = min(len(P), len(ref)); ok = np.linalg.norm(ref[:m, 3:], axis=1) > 0.5
    Rr, tr = quat_wxyz_to_R(ref[:m][ok][:, 3:]), ref[:m][ok][:, :3]
    # GT pose maps the canonical (oriented-bounds) frame? test both conventions
    Vs = (Vh @ S.T)[:, :3]  # mesh in S-frame?  eval uses T = P@S, i.e. object frame = S^-1 mesh
    Vc = (Vh @ np.linalg.inv(S).T)[:, :3]
    hg = lambda Vx: (np.einsum("tij,vj->tvi", Rr, Vx) + tr[:, None]) @ n + d
    hgt = hg(Vc).min(1)
    Tw = W[None] @ (P @ S); ho = ((np.einsum("tij,vj->tvi", Tw[:m][ok][:, :3, :3], Vc) + Tw[:m][ok][:, None, :3, 3]) @ n + d).min(1)
    q = lambda h: f"p5 {np.percentile(h,5)*100:+5.1f} med {np.median(h)*100:+5.1f} p95 {np.percentile(h,95)*100:+5.1f}"
    print(f"{idx:2d} {meta[idx]['object'][:12]:12s} GT {q(hgt)} | ours {q(ho)} | diff_med {(np.median(ho-hgt))*100:+5.1f}", flush=True)
