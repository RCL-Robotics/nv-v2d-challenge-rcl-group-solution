#!/usr/bin/env python3
"""Physical consistency of our object trajectory with the reconstructed support, and an optional vertical snap.
Rest frames: the object moves < 3 mm/frame (5-frame window) and its lowest point is > 15 cm above the floor.
For each rest frame, the support height under the object is found by casting rays straight down from object surface
samples; gap = object lowest point - highest support hit below it (positive: floating, negative: penetrating).
Leaning objects (no hit under the object) get the closest object-support distance instead.
--snap writes a corrected trajectory: every rest segment with hits is shifted along the up axis by -median(gap),
blended in/out over 10 frames, to OUT/epXXX/object_poses_world_snapped.npz.
--gt (diagnostic only) measures the reference object poses against our support instead.
Usage: contact.py [eps] [--snap] [--gt]"""
import json, sys, numpy as np, trimesh
from common import *
TRAJ = f"{V}/experiments/t3_frame0/opus_f3_c2_shift"
args = [a for a in sys.argv[1:] if not a.startswith("--")]; SNAP = "--snap" in sys.argv
GT = "--gt" in sys.argv  # diagnostic: measure the gap of the reference (GT) object poses to our support instead
res = {}
for i in (map(int, args[0].split(",")) if args else SUPPORT):
    d = f"{OUT}/ep{i:03d}"; s = json.load(open(f"{d}/support.json")); g = np.array(s["ground_plane"]); n = g[:3] / np.linalg.norm(g[:3]); g0 = g[3] / np.linalg.norm(g[:3])
    sup = trimesh.load(f"{MESH}/{s['support']}/{s['support']}.glb", force="scene", process=False).to_geometry(); sup.apply_transform(np.array(s["T_world_mesh"]))
    m = json.load(open(f"{inp(i)}/episode.json"))
    obj = trimesh.load(f"{inp(i)}/object_mesh.glb", force="scene", process=False).to_geometry()
    np.random.seed(0); P = np.asarray(obj.sample(400))
    if GT:
        from scipy.spatial.transform import Rotation as Rot
        ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ok = np.linalg.norm(ref[:, 3:], axis=1) > .5
        ref[~ok] = ref[np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))][~ok]  # hold the last valid pose
        T = np.tile(np.eye(4), (len(ref), 1, 1)); T[:, :3, :3] = Rot.from_quat(ref[:, [4, 5, 6, 3]]).as_matrix(); T[:, :3, 3] = ref[:, :3]
    else: T = np.load(f"{TRAJ}/episode_{i:06d}.npz")["T_world_mesh"].astype(float)
    F = len(T)
    X = np.einsum("tij,pj->tpi", T[:, :3, :3], P) + T[:, None, :3, 3]          # F x 400 x 3 world samples
    hgt = X @ n + g0; low = hgt.min(1)
    c = T[:, :3, 3]; sp = np.zeros(F); sp[2:-2] = np.linalg.norm(c[4:] - c[:-4], axis=1) / 4; sp[:2], sp[-2:] = sp[2], sp[-3]
    rest = (sp < .003) & (low > .15)
    gaps = np.full(F, np.nan); dist = np.full(F, np.nan); idx = np.where(rest)[0]
    for t in idx[::5]:  # every 5th rest frame
        q = X[t][np.argsort(hgt[t])[:60]]                                       # 60 lowest samples
        loc, ri, _ = sup.ray.intersects_location(q + .3 * n, np.tile(-n, (len(q), 1)), multiple_hits=True)
        loc = np.asarray(loc).reshape(-1, 3); hit = loc @ n + g0
        below = hit <= (q[ri] @ n + g0) + .05 if len(ri) else np.zeros(0, bool)
        if below.any(): gaps[t] = low[t] - hit[below].max()
        else: dist[t] = float(trimesh.proximity.closest_point(sup, X[t][::4])[1].min())
    ok = ~np.isnan(gaps)
    r = dict(object=m["object"], support=s["support"], rest_frames=int(rest.sum()), rest_with_support_below=int(ok.sum()),
             gap_median_cm=float(np.nanmedian(gaps) * 100) if ok.any() else None, gap_p10_p90_cm=[float(np.nanpercentile(gaps, 10) * 100), float(np.nanpercentile(gaps, 90) * 100)] if ok.any() else None,
             leaning_min_dist_cm=float(np.nanmedian(dist) * 100) if (~np.isnan(dist)).any() else None)
    res[i] = r; print(i, r, flush=True)
    if SNAP and not GT:
        off = np.zeros(F)
        # rest segments: contiguous runs of rest frames (gaps of < 10 frames bridged)
        seg = []; st = None
        for t in range(F + 1):
            on = t < F and rest[t]
            if on and st is None: st = t
            if not on and st is not None:
                if seg and st - seg[-1][1] < 10: seg[-1] = (seg[-1][0], t)
                else: seg.append((st, t))
                st = None
        for a, b in seg:
            gg = gaps[a:b]; gg = gg[~np.isnan(gg)]
            if len(gg) == 0: continue
            w = np.zeros(F); w[a:b] = 1
            for k in range(1, 11):
                if a - k >= 0: w[a - k] = max(w[a - k], 1 - k / 11)
                if b - 1 + k < F: w[b - 1 + k] = max(w[b - 1 + k], 1 - k / 11)
            off -= w * np.median(gg)
        T2 = T.copy(); T2[:, :3, 3] += off[:, None] * n
        np.savez(f"{d}/object_poses_world_snapped.npz", T_world_mesh=T2.astype(np.float32), vertical_offset_m=off.astype(np.float32))
json.dump({str(k): v for k, v in res.items()}, open(f"{OUT}/contact_report{'_gt_diag' if GT else ''}.json", "w"), indent=1)
