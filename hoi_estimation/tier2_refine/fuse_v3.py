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
v3: the frame-0 object pose (allowed input) corrects the camera rotation. Over the static start (frames from 0 while the
Tier 2 position stays within 2 cm of frame 0 and our mapped rotation within 5 deg of its frame-0 value, at most KMAX frames)
the object is assumed at its frame-0 pose, and the camera rotation is left-multiplied by the chordal mean of
R0 (Rc R_cam(t) S)^T, with S the flip closest to R0 at frame 0. The correction is skipped when it exceeds MAXFIX deg.
Axisymmetric objects (mesh test: mean nearest-point distance after rotating about a principal axis by 30..330 deg is
below SYM % of the bounding-box diagonal): the spin about that axis is not observable in the image, so the gate uses
the axis angle and the result is Tier 2 swung (minimal rotation) onto our axis direction, keeping the Tier 2 spin.
Usage: fuse_v3.py <cam.json> <out_dir> [tau=45] [win=5] [flips=1] [interp=0] [camfix=1] [kmax=60] [maxfix=15] [sym=0]"""
import json, os, sys, numpy as np, trimesh
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R
from scipy.spatial.transform import Rotation as Rot
V = os.environ["V2D_ROOT"]; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
T2DS = f"{V}/dataset/track_2/tier_2_synthetic_noise"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{T2DS}/meta/episodes_metadata.jsonl"))}
cams = json.load(open(sys.argv[1])); OUT = sys.argv[2]; a = sys.argv[3:]
TAU = float(a[0]) if len(a) > 0 else 45.0; WIN = int(a[1]) if len(a) > 1 else 5
FLIPS = int(a[2]) if len(a) > 2 else 1; INTERP = int(a[3]) if len(a) > 3 else 0
CAMFIX = int(a[4]) if len(a) > 4 else 1; KMAX = int(a[5]) if len(a) > 5 else 60; MAXFIX = float(a[6]) if len(a) > 6 else 15.0
SYM = float(a[7]) if len(a) > 7 else 0.0
REF = f"{V}/ref_poses"  # only frame 0 is read in fuse(): the frame-0 object pose is an allowed input
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
    ax = None
    if SYM > 0:
        np.random.seed(0); q = np.asarray(mesh.sample(8000)); q = q - q.mean(0); tr = cKDTree(q); dg = np.linalg.norm(q.max(0) - q.min(0))
        sc = [np.mean([tr.query(Rot.from_rotvec(E[:, k] * np.radians(d_)).apply(q))[0].mean() for d_ in range(30, 360, 30)]) / dg * 100 for k in range(3)]
        if min(sc) < SYM: ax = E[:, int(np.argmin(sc))]
    S = [np.eye(3)] + ([E @ np.diag(d) @ E.T for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])] if FLIPS else [])
    Wi = np.linalg.inv(np.array(json.load(open(f"{REL}/{sid}/camera.json"))["T_world_cam"]))
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); F = len(t2)
    vis = np.linalg.norm(t2[:, 3:], axis=1) > .5
    R2 = np.tile(np.eye(3), (F, 1, 1)); R2[vis] = quat_wxyz_to_R(t2[vis, 3:])
    f0 = np.load(f"{REF}/episode_{i:06d}.npz")["pose_xyz_wxyz"][0].astype(float); R0 = quat_wxyz_to_R(f0[None, 3:])[0]
    Ts = [np.load(opus(i))["T_world_mesh"], np.load(f"{REL}/{sid}/poses.npy")]
    fix = np.eye(3); info = dict(fix_deg=0.0, K=0, applied=False)
    if CAMFIX and ax is None:  # axisymmetric: the frame-0 match would mix in our unobservable spin
        Rm = Rc[None] @ (Wi[None] @ Ts[0][:KMAX])[:, :3, :3]
        k0 = ang(R0[None], np.stack([Rm[0] @ s_ for s_ in S]))
        S0 = S[int(k0.argmin())]; Rk = Rm @ S0
        st = (np.linalg.norm(t2[:len(Rk), :3] - t2[0, :3], axis=1) < .02) & (ang(Rk[:1], Rk) < 5)
        K = int(np.argmin(st)) if not st.all() else len(st); K = max(K, 1)
        D = proj(np.einsum("ij,tkj->tik", R0, Rk[:K]).sum(0)); fd = float(ang(D, np.eye(3)))
        info = dict(fix_deg=fd, K=K, applied=fd < MAXFIX, k0_deg=float(k0.min()))
        if fd < MAXFIX: fix = D
    Rc = fix @ Rc
    srcs = []
    for T in Ts:
        Rm = np.tile(np.eye(3), (F, 1, 1)); n = min(F, len(T)); Rm[:n] = Rc[None] @ (Wi[None] @ T[:n])[:, :3, :3]
        cand = np.stack([Rm @ s for s in S], 1)                                   # F x K x 3 x 3
        if ax is None: d = ang(R2[:, None], cand)
        else: d = np.degrees(np.arccos(np.clip(np.einsum("ti,tki->tk", R2 @ ax, cand @ ax), -1, 1)))  # axis angle
        k = d.argmin(1); best = cand[np.arange(F), k]; dm = d[np.arange(F), k]
        ok = (dm < TAU) & vis; ok[n:] = False; srcs.append((best, ok))
    R = R2.copy(); acc = np.zeros(F, bool); which = np.full(F, -1)
    for j, (best, ok) in enumerate(srcs):
        sel = ok & ~acc; R[sel] = best[sel]; acc |= sel; which[sel] = j
    if ax is not None:  # keep the Tier 2 spin: swing R2 so that its axis matches ours
        for t in np.where(acc)[0]:
            u, v = R[t] @ ax, R2[t] @ ax; c_ = np.cross(v, u); sn = np.linalg.norm(c_)
            Q = np.eye(3) if sn < 1e-9 else Rot.from_rotvec(c_ / sn * np.arctan2(sn, v @ u)).as_matrix(); R[t] = Q @ R2[t]
    info["sym"] = ax is not None
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
    return T, vis, acc, which, info

res = {}
for i in range(30):
    T, vis, acc, which, info = fuse(i); np.savez(f"{OUT}/episode_{i:06d}.npz", T_world_mesh=T, visible=vis, accepted=acc, source=which)
    # diagnostic (GT)
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); n = min(len(T), len(ref))
    g = vis[:n] & (np.linalg.norm(ref[:n, 3:], axis=1) > .5); Rr = quat_wxyz_to_R(ref[:n][g][:, 3:]); pr = ref[:n][g][:, :3]
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    np.random.seed(0); v = np.asarray(trimesh.load(f"{DS}/{meta[i]['mesh']}", force="scene", process=False).to_geometry().sample(1000))
    a2 = metrics(quat_wxyz_to_R(t2[:n][g][:, 3:]), t2[:n][g][:, :3], Rr, pr, v)["AUC"]; af = metrics(T[:n][g][:, :3, :3], T[:n][g][:, :3, 3], Rr, pr, v)["AUC"]
    re = float(np.median(ang(Rr, T[:n][g][:, :3, :3])))
    res[i] = (a2, af, re, float(acc[vis].mean()), float((which[vis] == 1).mean()), info)
    print(f"{i:2d} {meta[i]['object'][:16]:16s} t2 {a2:.3f} -> fused {af:.3f}  rot {re:4.1f}d  accepted {acc[vis].mean():4.0%} (s1 {res[i][4]:.0%})  fix {info['fix_deg']:.1f}d K={info['K']} {'ok' if info['applied'] else 'SKIP'}{' SYM' if info['sym'] else ''}", flush=True)
A = np.array([v[:5] for v in res.values()])
print(f"MEAN t2 {A[:,0].mean():.3f} fused {A[:,1].mean():.3f} rot {A[:,2].mean():.1f}d | better {int((A[:,1]>A[:,0]+.01).sum())} worse {int((A[:,1]<A[:,0]-.01).sum())}")
json.dump({str(k): v for k, v in res.items()}, open(f"{OUT}/diag_scores.json", "w"))
