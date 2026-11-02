#!/usr/bin/env python3
"""Tier 2 refinement v2 (no GT used to build the result). Per frame:
  candidates  = our camera-frame rotations (Opus first, then s1) mapped to the Tier 2 world with the human-fitted camera,
                each also composed with the 180 deg turns about the mesh's principal axes (symmetric objects);
  selection   = the first source whose closest candidate lies within TAU deg of the Tier 2 rotation;
  rejected    = Tier 2 rotation corrected by the offset R_sel R2^T interpolated from the nearest accepted frames
                (the Tier 2 noise varies slowly), or Tier 2 as is when no frame of the episode is accepted;
  smoothing   = sliding-window chordal mean of the rotations (WIN frames);
  position    = Tier 2 position.
Writes per-episode npz (T_world_mesh, F x 4 x 4; zeros where Tier 2 marks the object invisible) and, as a diagnostic,
scores against the Tier 1 reference.
Usage: fuse_v2.py <cam.json> <out_dir> [tau=45] [win=5] [flips=1] [interp=1]"""
import json, os, sys, numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R
from scipy.spatial.transform import Rotation as Rot
V = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"); DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
T2DS = f"{V}/dataset/track_2/tier_2_synthetic_noise"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{T2DS}/meta/episodes_metadata.jsonl"))}
cams = json.load(open(sys.argv[1])); OUT = sys.argv[2]; a = sys.argv[3:]
TAU = float(a[0]) if len(a) > 0 else 45.0; WIN = int(a[1]) if len(a) > 1 else 5
FLIPS = int(a[2]) if len(a) > 2 else 1; INTERP = int(a[3]) if len(a) > 3 else 1
os.makedirs(OUT, exist_ok=True)
def opus(i): return f"{V}/agent_runs/{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
def ang(A, B):  # geodesic angle (deg), broadcasting over leading dims
    M = np.einsum("...ji,...jk->...ik", *np.broadcast_arrays(A, B)); tr = np.trace(M, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
def proj(M): u, _, vt = np.linalg.svd(M); d = np.sign(np.linalg.det(u @ vt)); u[..., :, -1] *= d[..., None]; return u @ vt

def fuse(i):
    m = meta[i]; sid = m["sequence_id"]; c = cams[str(i)]; Rc = np.array(c["R"])
    mesh = trimesh.load(f"{T2DS}/{m['mesh']}", force="scene", process=False).to_geometry()
    vv = np.asarray(mesh.vertices); E = np.linalg.eigh(np.cov((vv - vv.mean(0)).T))[1]  # principal axes (columns)
    S = [np.eye(3)] + ([E @ np.diag(d) @ E.T for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])] if FLIPS else [])
    Wi = np.linalg.inv(np.array(json.load(open(f"{REL}/{sid}/camera.json"))["T_world_cam"]))
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); F = len(t2)
    vis = np.linalg.norm(t2[:, 3:], axis=1) > .5
    R2 = np.tile(np.eye(3), (F, 1, 1)); R2[vis] = quat_wxyz_to_R(t2[vis, 3:])
    srcs = []
    for T in (np.load(opus(i))["T_world_mesh"], np.load(f"{REL}/{sid}/poses.npy")):
        Rm = np.tile(np.eye(3), (F, 1, 1)); n = min(F, len(T)); Rm[:n] = Rc[None] @ (Wi[None] @ T[:n])[:, :3, :3]
        cand = np.stack([Rm @ s for s in S], 1)                                   # F x K x 3 x 3
        d = ang(R2[:, None], cand); k = d.argmin(1); best = cand[np.arange(F), k]; dm = d[np.arange(F), k]
        ok = (dm < TAU) & vis; ok[n:] = False; srcs.append((best, ok))
    R = R2.copy(); acc = np.zeros(F, bool); which = np.full(F, -1)
    for j, (best, ok) in enumerate(srcs):
        sel = ok & ~acc; R[sel] = best[sel]; acc |= sel; which[sel] = j
    if INTERP and acc.any():  # rejected frames: Tier 2 rotated by the offset of the nearest accepted frames
        idx = np.where(acc)[0]; off = Rot.from_matrix(np.einsum("tij,tkj->tik", R[idx], R2[idx])).as_rotvec()
        for t in np.where(vis & ~acc)[0]:
            p = np.searchsorted(idx, t); nb = [q for q in (p - 1, p) if 0 <= q < len(idx)]
            w = np.array([1.0 / (1 + abs(idx[q] - t)) for q in nb]); o = (w[:, None] * off[nb]).sum(0) / w.sum()
            R[t] = Rot.from_rotvec(o).as_matrix() @ R2[t]
    if WIN > 1:  # sliding chordal mean over visible frames
        Rs = R.copy(); h = WIN // 2
        for t in np.where(vis)[0]:
            lo, hi = max(0, t - h), min(F, t + h + 1); w = vis[lo:hi]; Rs[t] = proj(R[lo:hi][w].sum(0))
        R = Rs
    T = np.zeros((F, 4, 4)); T[vis, :3, :3] = R[vis]; T[vis, :3, 3] = t2[vis, :3]; T[vis, 3, 3] = 1
    return T, vis, acc, which

res = {}
for i in range(30):
    T, vis, acc, which = fuse(i); np.savez(f"{OUT}/episode_{i:06d}.npz", T_world_mesh=T, visible=vis, accepted=acc, source=which)
    # diagnostic (GT)
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); n = min(len(T), len(ref))
    g = vis[:n] & (np.linalg.norm(ref[:n, 3:], axis=1) > .5); Rr = quat_wxyz_to_R(ref[:n][g][:, 3:]); pr = ref[:n][g][:, :3]
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    np.random.seed(0); v = np.asarray(trimesh.load(f"{DS}/{meta[i]['mesh']}", force="scene", process=False).to_geometry().sample(1000))
    a2 = metrics(quat_wxyz_to_R(t2[:n][g][:, 3:]), t2[:n][g][:, :3], Rr, pr, v)["AUC"]; af = metrics(T[:n][g][:, :3, :3], T[:n][g][:, :3, 3], Rr, pr, v)["AUC"]
    re = float(np.median(ang(Rr, T[:n][g][:, :3, :3])))
    res[i] = (a2, af, re, float(acc[vis].mean()), float((which[vis] == 1).mean()))
    print(f"{i:2d} {meta[i]['object'][:16]:16s} t2 {a2:.3f} -> fused {af:.3f}  rot {re:4.1f}d  accepted {acc[vis].mean():4.0%} (s1 {res[i][4]:.0%})", flush=True)
A = np.array(list(res.values()))
print(f"MEAN t2 {A[:,0].mean():.3f} fused {A[:,1].mean():.3f} rot {A[:,2].mean():.1f}d | better {int((A[:,1]>A[:,0]+.01).sum())} worse {int((A[:,1]<A[:,0]-.01).sum())}")
json.dump({str(k): v for k, v in res.items()}, open(f"{OUT}/diag_scores.json", "w"))
