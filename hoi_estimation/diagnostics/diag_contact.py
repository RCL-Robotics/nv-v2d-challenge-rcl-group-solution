"""Per-episode object depth factor from hand-object contact (estimate uses only our reconstruction; GT only for scoring).

Human depth is anchored by the first-frame alignment and the feet-on-floor correction (FF2 factor f_t, recovered
from the FF2 vs source object translations). During contact frames the object surface should touch the hand, so
k_contact = argmin_k median_t min_dist(hand keypoints, R_t x + k t_t).
"""
import json, os, sys
import numpy as np, torch, trimesh
sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
from eval_official import bundle_path
from lib_mhr.mhr_layer import MHRLayer
from tools.render_mhr_wild_inference import _params

V2D = "/mnt/nas/Group/hanzhou/v2d"
ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/pnp_forE1.json"))
ff = json.load(open(f"{V2D}/experiments/FF2_feet_ref/feet_floor.json"))
dev = torch.device("cuda")
mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
HANDS = {0: list(range(42, 63)), 1: list(range(21, 42))}  # contact_logits columns assumed (left, right); both tried below
KS = np.linspace(0.85, 1.20, 141)
eps = [int(x) for x in sys.argv[1].split(",")]
CM = float(os.environ.get("CM", 0.06))
out = {}
print("ep obj         k*    k_con  ncon  d@1  d@k  | AUC   AUCcon  t2")
t2 = {int(r.split(",")[0]): float(r.strip().split(",")[10]) for r in open(f"{V2D}/logs/officialPnP_FF2.csv").readlines()[1:]}
for idx in eps:
    ep = f"episode_{idx:06d}"
    b = torch.load(bundle_path("FF2_feet_ref", ep), map_location="cpu", weights_only=False)
    bs = torch.load(bundle_path(ff[str(idx)]["src"], ep), map_location="cpu", weights_only=False)
    P = np.asarray(b["pr"]["pose_abs"], np.float64); Ps = np.asarray(bs["pr"]["pose_abs"], np.float64)
    F = len(P)
    f = np.einsum("ti,ti->t", P[:, :3, 3], Ps[:, :3, 3]) / np.einsum("ti,ti->t", Ps[:, :3, 3], Ps[:, :3, 3])
    kp = []
    with torch.no_grad():
        for a in range(0, F, 128):
            kp.append(mhr.mhr_forward(_params(b["pr"], slice(a, min(a + 128, F)), dev)).keypoints.cpu().numpy().astype(np.float64))
    kp = np.concatenate(kp) * f[:, None, None]
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); verts = np.asarray(glb.sample(1000))
    surf = np.asarray(glb.sample(1500)); surf_o = (np.c_[surf, np.ones(len(surf))] @ np.linalg.inv(S).T)[:, :3]
    T = P @ S
    Xr = np.einsum("tij,vj->tvi", T[:, :3, :3], surf_o)  # rotated surface, camera frame (no translation)
    tc = T[:, :3, 3]
    # contact frames: either hand within CM of the surface at k=1 (and CoCoNet logit agrees if available)
    logit = np.asarray(b["pr"]["contact_logits"], np.float64)
    best = None
    dist1 = {}
    for h, ids in HANDS.items():
        H = kp[:, ids]
        dd = np.linalg.norm(H[:, :, None] - (Xr + tc[:, None])[:, None], axis=-1).min(axis=(1, 2))
        dist1[h] = dd
    dmin = np.minimum(dist1[0], dist1[1]); hsel = np.where(dist1[0] < dist1[1], 0, 1)
    con = (dmin < CM) & ((logit > 0).any(1))
    if con.sum() < 15:
        con = dmin < CM
    sel = np.flatnonzero(con)
    if len(sel) < 15:
        print(f"{idx:2d} {meta[idx]['object'][:10]:10s} too few contact frames ({len(sel)})"); out[idx] = None; continue
    sel = sel[:: max(1, len(sel) // 150)]
    cost = []
    for k in KS:
        dk = [np.linalg.norm(kp[t, HANDS[hsel[t]]][:, None] - (Xr[t] + k * tc[t])[None], axis=-1).min() for t in sel]
        cost.append(np.median(dk))
    cost = np.array(cost); kc = float(KS[np.argmin(cost)])
    # scoring (GT)
    W = np.asarray(al[str(idx)]["sim"]); s = np.cbrt(np.linalg.det(W[:3, :3])); Rw = W[:3, :3] / s; C = W[:3, 3]
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
    n = min(F, len(ref)); ok = (np.linalg.norm(ref[:n, 3:], axis=1) > 0.5) & np.isfinite(T[:n]).all(axis=(1, 2))
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    Rwo = Rw[None] @ T[:n][ok][:, :3, :3]; tu = s * tc[:n][ok] @ Rw.T
    m0 = metrics(Rwo, C + tu, Rr, tr, verts); mc = metrics(Rwo, C + kc * tu, Rr, tr, verts)
    r = np.linalg.norm(tc[:n][ok], axis=1) / np.linalg.norm((tr - C) @ Rw / s, axis=1); kstar = np.median(1 / r)
    print(f"{idx:2d} {meta[idx]['object'][:10]:10s} {kstar:.3f} {kc:.3f} {con.sum():4d} {cost[np.argmin(np.abs(KS-1))]*100:4.1f} {cost.min()*100:4.1f} | "
          f"{m0['AUC']:.3f} {mc['AUC']:.3f} {t2[idx]:.3f}", flush=True)
    out[idx] = {"k_contact": kc, "n_contact": int(con.sum()), "cost_min": float(cost.min()), "cost_at1": float(cost[np.argmin(np.abs(KS - 1))])}
json.dump(out, open(os.environ.get("OUT", "/dev/null"), "w"), indent=1)
