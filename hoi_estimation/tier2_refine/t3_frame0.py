#!/usr/bin/env python3
"""Tier 3: refine our world object trajectory (Opus v2 or s1) with the frame-0 object pose (allowed input).
Steps (each switchable, applied in this order; flip=2 replaces flip by a general object-frame anchor R <- R C fitted on
the static start, flip=3 runs flip and camrot and then applies that anchor only when the remaining offset exceeds 15 deg):
  flip   : at frame 0 pick the mesh 180-deg principal-axis turn S (or none) closest to the frame-0 rotation, then track
           the turn through the episode by continuity (per frame, the S that makes R(t) S closest to the previous output);
  camrot : over the static start (frames from 0 while our pose stays within 2 cm / 5 deg of frame 0, at most 60) the
           object is at its frame-0 pose; the chordal mean D of R0 R(t)^T is taken as the camera rotation error and
           applied about the camera centre c: R <- D R, p <- c + D (p - c) (camrot=2: rotations only). Skipped when D exceeds 15 deg;
  trans  : "shift" adds the constant p0 - mean_static(p);  "scale" scales distances from c so that the static mean
           reaches |p0 - c|, then shifts the remaining lateral offset.
No GT beyond frame 0 is used to build the result; the score against the reference is a diagnostic.
Usage: t3_frame0.py <src: opus|s1> <out_dir> <flip 0/1/2/3> <camrot 0/1/2> <trans: none|shift|scale>"""
import json, os, sys, numpy as np, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R, MPPE_OFFSETS
V = os.environ["V2D_ROOT"]; DS = f"{V}/dataset/track_2/tier_1_multiview_caption"; REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
SRC, OUT, FLIP, CAMROT, TRANS = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
os.makedirs(OUT, exist_ok=True)
def opus(i): return f"{V}/agent_runs/{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
def ang(A, B):
    M = np.einsum("...ji,...jk->...ik", *np.broadcast_arrays(A, B)); return np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
def proj(M): u, _, vt = np.linalg.svd(M); u[:, -1] *= np.sign(np.linalg.det(u @ vt)); return u @ vt

def refine(i):
    m = meta[i]; sid = m["sequence_id"]
    T = (np.load(opus(i))["T_world_mesh"] if SRC == "opus" else np.load(f"{REL}/{sid}/poses.npy")).astype(float).copy()
    R, p = T[:, :3, :3].copy(), T[:, :3, 3].copy(); F = len(T)
    c = np.array(json.load(open(f"{REL}/{sid}/camera.json"))["T_world_cam"])[:3, 3]
    f0 = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"][0].astype(float); R0 = quat_wxyz_to_R(f0[None, 3:])[0]; p0 = f0[:3]
    mesh = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry()
    vv = np.asarray(mesh.vertices); E = np.linalg.eigh(np.cov((vv - vv.mean(0)).T))[1]
    S = np.stack([np.eye(3)] + [E @ np.diag(d) @ E.T for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])])
    info = {}
    st = (np.linalg.norm(p[:60] - p[0], axis=1) < .02) & (ang(R[:1], R[:60]) < 5)
    K = max(int(np.argmin(st)) if not st.all() else len(st), 1); info["K"] = K
    if FLIP == 2:  # general object-frame anchor: R <- R C with C the static-start mean of R(t)^T R0
        C = proj(np.einsum("tji,jk->ik", R[:K], R0)); info["anchor_deg"] = float(ang(C, np.eye(3))); R = R @ C
    elif FLIP:
        k = int(ang(R0, R[0] @ S).argmin()); out = R.copy(); out[0] = R[0] @ S[k]; nsw = 0
        for t in range(1, F):
            kt = int(ang(out[t - 1], R[t] @ S).argmin()); nsw += kt != k; k = kt; out[t] = R[t] @ S[k]
        info["flip0"] = int(ang(R0, R[0] @ S).argmin()); info["switches"] = nsw; R = out
    if CAMROT:
        D = proj(np.einsum("ij,tkj->ik", R0, R[:K])); fd = float(ang(D, np.eye(3))); info["fix_deg"] = fd
        if fd < 15:
            R = D[None] @ R
            if CAMROT == 1: p = c + (p - c) @ D.T  # camrot=2: rotations only
    if FLIP == 3:  # hybrid: after flip tracking and the camera fix, a residual above 15 deg at the static start is an
        C = proj(np.einsum("tji,jk->ik", R[:K], R0)); info["anchor_deg"] = float(ang(C, np.eye(3)))  # object misorientation
        if info["anchor_deg"] > 15: R = R @ C
    pm = p[:K].mean(0)
    if TRANS == "shift": p = p + (p0 - pm)
    elif TRANS == "scale":
        s = np.linalg.norm(p0 - c) / np.linalg.norm(pm - c); p = c + s * (p - c); p = p + (p0 - (c + s * (pm - c))); info["scale"] = float(s)
    T[:, :3, :3], T[:, :3, 3] = R, p
    return T, info

res = {}
for i in range(30):
    T, info = refine(i); np.savez(f"{OUT}/episode_{i:06d}.npz", T_world_mesh=T)
    # diagnostic (GT)
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ok = np.linalg.norm(ref[:, 3:], axis=1) > .5
    np.random.seed(0); v = np.asarray(trimesh.load(f"{DS}/{meta[i]['mesh']}", force="scene", process=False).to_geometry().sample(1000))
    n = min(len(T), len(ref)); k = ok[:n]; Rr = quat_wxyz_to_R(ref[:n][k][:, 3:]); tr = ref[:n][k][:, :3]; Tk = T[:n][k]
    pe = (Tk[:, None, :3, 3] + np.einsum("tij,kj->tki", Tk[:, :3, :3], MPPE_OFFSETS)).reshape(-1, 3); pg = (tr[:, None] + np.einsum("tij,kj->tki", Rr, MPPE_OFFSETS)).reshape(-1, 3)
    d = (pg - pe).mean(0); a = metrics(Tk[:, :3, :3], Tk[:, :3, 3], Rr, tr, v)["AUC"]; at = metrics(Tk[:, :3, :3], Tk[:, :3, 3] + d, Rr, tr, v)["AUC"]
    re = float(np.median(ang(Rr, Tk[:, :3, :3]))); te = float(np.median(np.linalg.norm(tr - Tk[:, :3, 3], axis=1)) * 100)
    res[i] = dict(auc=a, auc_t=at, rot=re, pos_cm=te, **info)
    print(f"{i:2d} {meta[i]['object'][:16]:16s} auc {a:.3f} (t-only {at:.3f}) rot {re:5.1f}d pos {te:5.1f}cm  {info}", flush=True)
A = np.array([[r["auc"], r["auc_t"], r["rot"], r["pos_cm"]] for r in res.values()])
print(f"MEAN auc {A[:,0].mean():.3f} t-only {A[:,1].mean():.3f} rot(med) {A[:,2].mean():.1f}d pos(med) {A[:,3].mean():.1f}cm")
json.dump({str(k): v for k, v in res.items()}, open(f"{OUT}/diag_scores.json", "w"))
