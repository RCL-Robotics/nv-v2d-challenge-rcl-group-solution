"""DIAGNOSTIC: how much of JC1's object error is per-frame noise vs per-episode bias, and what temporal filters buy."""
import json, sys
import numpy as np, torch, trimesh
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
V2D = "/mnt/nas/Group/hanzhou/v2d"; ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/JC1_joint_camera_align.json"))
print("#ep obj        | raw  | med15 med45 | dep15 dep45 | static | bias-free(oracle) | pos_cm raw, bias_cm, resid_cm, speed_cm/f")
for idx in range(30):
    ep = f"episode_{idx:06d}"
    b = torch.load(f"{V2D}/experiments/JC1_joint_camera/{ep}/coconet.pth", map_location="cpu", weights_only=False)
    P = np.asarray(b["pr"]["pose_abs"], np.float64); F = len(P)
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); verts = np.asarray(glb.sample(1000))
    W = np.asarray(al[str(idx)]["sim"]); Rw, C = W[:3, :3], W[:3, 3]
    T = P @ S; tc = T[:, :3, 3]; Rc = T[:, :3, :3]
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
    n = min(F, len(ref)); ok = (np.linalg.norm(ref[:n, 3:], axis=1) > 0.5)
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    def ev(tcam, Rcam=Rc):
        tw = tcam @ Rw.T + C; Rwo = Rw[None] @ Rcam
        return metrics(Rwo[:n][ok], tw[:n][ok], Rr, tr, verts)
    tw = tc @ Rw.T + C; bias = np.median(tw[:n][ok] - tr, axis=0)
    bc = bias @ Rw  # camera frame
    dep = np.median(np.linalg.norm(tc, axis=1)); u = np.median(tc / np.linalg.norm(tc,axis=1)[:,None], axis=0); u/=np.linalg.norm(u)
    along = bc @ u; perp = np.linalg.norm(bc - along * u)
    # per-frame along-ray ratio ours/GT
    tg = (tr - C) @ Rw; r = np.linalg.norm(tc[:n][ok],axis=1)/np.linalg.norm(tg,axis=1)
    # height bias (world up = plane normal)
    gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"]); nn = gp[:3]/np.linalg.norm(gp[:3])
    if nn @ C + gp[3]/np.linalg.norm(gp[:3]) < 0: nn = -nn
    print(f"{idx:2d} {meta[idx]['object'][:10]:10s} bias {np.linalg.norm(bias)*100:5.1f}cm along-ray {along*100:+6.1f} perp {perp*100:5.1f} height {bias@nn*100:+6.1f} depth {dep:.2f}m ratio {np.median(r):.3f} cam_xyz {np.round(bc*100,1)}", flush=True)
