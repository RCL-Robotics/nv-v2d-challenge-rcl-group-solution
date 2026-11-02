"""Floor-contact depth factor: k such that our object's lowest vertex lies on the released ground plane.
Compliant estimate uses only our trajectory + released plane + first-frame alignment W; GT is printed for diagnosis only."""
import json, os, sys
import numpy as np, torch, trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R
from eval_official import bundle_path

V2D = os.environ["V2D_ROOT"]
ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/pnp_forE1.json"))
pipe = sys.argv[1].split(","); eps = [int(x) for x in sys.argv[2].split(",")]
TOL = float(os.environ.get("TOL", 0.10)); VEL = float(os.environ.get("VEL", 0.005))
out = {}
print("ep obj         k*    k_flr  nfr  gt_gap  AUC   AUCflr  t2")
t2 = {int(l.split(",")[0]): l for l in []}
for idx in eps:
    ep = f"episode_{idx:06d}"
    v = next(p for p in pipe if os.path.isfile(bundle_path(p, ep)))
    P = np.asarray(torch.load(bundle_path(v, ep), map_location="cpu", weights_only=False)["pr"]["pose_abs"], dtype=np.float64)
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); verts = np.asarray(glb.sample(1000))
    hull = np.asarray(glb.convex_hull.vertices); hull_o = (np.c_[hull, np.ones(len(hull))] @ np.linalg.inv(S).T)[:, :3]
    T = P @ S
    W = np.asarray(al[str(idx)]["sim"]); s = np.cbrt(np.linalg.det(W[:3, :3])); Rw = W[:3, :3] / s; C = W[:3, 3]
    pl = json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"]; n = np.array(pl[:3]); d = pl[3]
    nw = n / np.linalg.norm(n); d = d / np.linalg.norm(n)   # plane is in the world frame; height = n.x + d
    if nw[1] > 0: nw, d = -nw, -d
    p0 = -d * nw                                           # a point on the plane, world
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
    nn = min(len(T), len(ref)); ok = (np.linalg.norm(ref[:nn, 3:], axis=1) > 0.5) & np.isfinite(T[:nn]).all(axis=(1, 2))
    T, ref = T[:nn][ok], ref[:nn][ok]
    Rr, tr = quat_wxyz_to_R(ref[:, 3:]), ref[:, :3]
    Rwo = Rw[None] @ T[:, :3, :3]; tw_unit = s * T[:, :3, 3] @ Rw.T       # world pos = C + k*tw_unit
    low = np.einsum("tij,vj->tvi", Rwo, hull_o) @ nw                         # per vertex height offset (rotation only)
    lowmin = low.min(1)
    h = (C - p0) @ nw + tw_unit @ nw + lowmin                                # lowest-point height at k=1
    k_t = ((p0 - C) @ nw - lowmin) / (tw_unit @ nw)                          # k making lowest point touch floor
    vel = np.r_[0, np.linalg.norm(np.diff(tw_unit, axis=0), axis=1)]
    sel = (np.abs(h) < TOL) & (vel < VEL)
    k_floor = float(np.median(k_t[sel])) if sel.sum() > 10 else np.nan
    # GT gap: lowest GT vertex height at the same frames (should be ~0 if resting on floor)
    gl = (np.einsum("tij,vj->tvi", Rr, hull_o) @ nw).min(1) + (tr - p0) @ nw
    m0 = metrics(Rwo, C + tw_unit, Rr, tr, verts)
    mf = metrics(Rwo, C + (k_floor if np.isfinite(k_floor) else 1) * tw_unit, Rr, tr, verts)
    r = np.linalg.norm(T[:, :3, 3], axis=1) / np.linalg.norm((tr - C) @ Rw / s, axis=1); kstar = np.median(1 / r)
    print(f"{idx:2d} {meta[idx]['object'][:10]:10s} {kstar:.3f} {k_floor:.3f} {sel.sum():4d} {np.median(gl[sel])*100 if sel.any() else np.nan:6.1f} {m0['AUC']:.3f} {mf['AUC']:.3f}")
    out[idx] = {"k_floor": k_floor, "n": int(sel.sum())}
json.dump(out, open(os.environ.get("OUT", "/dev/null"), "w"))
