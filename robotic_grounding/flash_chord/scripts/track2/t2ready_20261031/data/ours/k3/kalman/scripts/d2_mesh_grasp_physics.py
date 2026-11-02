"""DESIGN STEP 2 (NOISY-ONLY; Tier-2 input, Tier-2 retarget output, released meshes, physics).

A. Stable resting poses per object from the released mesh (convex hull + centre of mass; trimesh.poses.compute_stable_poses,
   deterministic: sigma=0, n_samples=1). COM rule: mesh.center_mass if mesh.is_volume else area-weighted surface centroid.
B. Human hand points (wrist + 5 finger ends per side) from the Tier-2 SOMA joints, mapped to the OpenCV world with the
   d1 frame maps; per-frame min distance to the object surface under the NOISY object pose. Grasp threshold = Otsu on
   log10(distance) over object-moving visible frames, pooled over episodes.
C. Carry runs (object not at rest, a hand within the threshold; ZUPT-style gap bridging and min length) and the
   hand->object relative-pose variogram on carry runs (slip budget after the object and root random walks).
D. Physics residuals at rest runs (hand-free midpoint): tilt angle to the nearest stable pose (all episodes) and height
   residual above the fitted floor (ground-only episodes); robust intercept with the random-walk growth fixed from d1.
Writes KOUT/design/d2_*.json, d2_stable_poses.npz, d2_hand_dist.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from kcommon import (FINGER_ENDS, H, J, KOUT, LMIN, keep_runs, load_g1, load_mesh_vertices, metas, object_track,  # noqa: E402
                     runs, support_is_empty, wxyz_to_rot)

OUT = KOUT / "design"
D1 = json.loads((OUT / "d1_variograms.json").read_text())
FR = {r["episode"]: r for r in json.loads((OUT / "d1_frames.json").read_text())}
M1 = np.load(OUT / "d1_masks.npz")
Q_BP2 = D1["object"]["pos_slope_cm2_per_frame"] / 3            # cm^2/frame/axis
Q_BR2 = D1["object"]["rot_slope_deg2_per_frame"] / 3            # deg^2/frame/axis
Q_HR2 = D1["feet_ge80"]["rot_slope"] / 3                        # deg^2/frame/axis (root rotation)
Q_HT2 = D1["hips_double_stance_ge80"]["gamma_pos_cm2"][0] / 3   # cm^2/frame/axis upper bound (Hips lag-1, double stance)
CHI2_1_99, CHI2_2_99 = 6.634897, 9.210340
LAGS = np.array([1, 2, 4, 8, 16])


def com_rule(mesh):
    if mesh.is_volume:
        return np.asarray(mesh.center_mass, float), "volume"
    tri = mesh.triangles
    a = mesh.area_faces
    return (tri.mean(1) * a[:, None]).sum(0) / a.sum(), "surface"


def stable_poses(mesh):
    import trimesh
    np.random.seed(0)
    com, kind = com_rule(mesh)
    T, p = trimesh.poses.compute_stable_poses(mesh, center_mass=com, sigma=0.0, n_samples=1, threshold=0.0)
    n = np.stack([t[:3, :3].T @ np.array([0, 0, -1.0]) for t in T])          # object-frame outward contact normal
    h = np.array([t[2, 3] for t in T])                                      # origin height above the support (m)
    hv = mesh.convex_hull.vertices
    h_chk = np.array([np.max(hv @ nk) for nk in n])
    return {"n": n, "h_m": h, "prob": np.asarray(p, float), "com": com, "com_kind": kind, "h_check_maxdiff_m": float(np.abs(h - h_chk).max())}


def otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    c = 0.5 * (edges[1:] + edges[:-1])
    w = hist.astype(float) / hist.sum()
    best, thr = -1, None
    for i in range(1, bins):
        w0, w1 = w[:i].sum(), w[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (w[:i] * c[:i]).sum() / w0, (w[i:] * c[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, edges[i]
    return float(thr)


def bridge(s, maxgap=2 * H + 1):
    s = s.copy()
    idx = np.flatnonzero(s)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= maxgap:
            s[a:b + 1] = True
    return s


def human_cv(g, fr):
    """SOMA joints -> OpenCV world (cm) and joint world rotations in OpenCV."""
    Rs2r, ts2r = np.array(fr["Rs2r"]), np.array(fr["ts2r"])
    B, b = np.array(fr["Bcv2r"]), np.array(fr["bcv2r"])
    pr = g["soma_joints"] @ Rs2r.T + ts2r
    pcv = ((pr - b) @ B) * 100.0
    Mrot = B.T @ Rs2r
    T, Jn = g["soma_joints_wxyz"].shape[:2]
    rot = wxyz_to_rot(g["soma_joints_wxyz"].reshape(-1, 4))
    rcv = (R.from_matrix(np.repeat(Mrot[None], T * Jn, 0)) * rot).as_quat().reshape(T, Jn, 4)    # xyzw
    return pcv, rcv, T, Jn


def main():
    ms = metas()
    # ---------------- A. stable poses
    sp_store, sp_rows, trees = {}, {}, {}
    for obj in sorted({m["object"] for m in ms}):
        mesh = load_mesh_vertices(obj)
        s = stable_poses(mesh)
        sp_store[obj] = s
        v = np.asarray(mesh.vertices, float)
        step = max(1, len(v) // 60000)
        trees[obj] = cKDTree(v[::step] * 100.0)                              # cm, object frame
        ext = (v.max(0) - v.min(0)) * 100
        sp_rows[obj] = {"n_vertices": int(len(v)), "is_volume": bool(mesh.is_volume), "com_kind": s["com_kind"], "n_stable": int(len(s["h_m"])),
                        "probs_top5": np.round(np.sort(s["prob"])[::-1][:5], 3).tolist(), "h_check_maxdiff_m": s["h_check_maxdiff_m"],
                        "extent_cm": np.round(ext, 1).tolist()}
        print(f"{obj:<20} V {len(v):>7} vol {mesh.is_volume!s:<5} stable {len(s['h_m']):>3} top {sp_rows[obj]['probs_top5']} ext {np.round(ext,1)}", flush=True)
    np.savez_compressed(OUT / "d2_stable_poses.npz", **{f"{o}__n": s["n"] for o, s in sp_store.items()},
                        **{f"{o}__h_m": s["h_m"] for o, s in sp_store.items()}, **{f"{o}__prob": s["prob"] for o, s in sp_store.items()},
                        **{f"{o}__com": s["com"] for o, s in sp_store.items()})
    # ---------------- B. hand-object distances
    per_ep, dist_store, moving_d, rest_d = {}, {}, [], []
    for m in ms:
        ep, seq, obj = m["episode_index"], m["sequence_id"], m["object"]
        p, rot, vis, _ = object_track(ep)
        g = load_g1(seq)
        pcv, rcv, T, Jn = human_cv(g, FR[ep])
        dmin = np.zeros((2, T))
        for si, side in enumerate(("left", "right")):
            ids = [J["LeftHand" if side == "left" else "RightHand"]] + FINGER_ENDS[side]
            pts = pcv[:, ids]                                                   # (T, 6, 3) cm
            loc = np.einsum("tji,tkj->tki", rot.as_matrix(), pts - p[:, None])  # R^T (x - p)
            d, _ = trees[obj].query(loc.reshape(-1, 3))
            dmin[si] = d.reshape(T, -1).min(1)
        st = M1[f"obj_{ep}"]
        mov = (~st) & vis
        moving_d.append(np.log10(np.maximum(dmin[:, mov].min(0), 0.1)))
        rest_d.append(np.log10(np.maximum(dmin[:, st & vis].min(0), 0.1)))
        dist_store[f"d_{ep}"] = dmin
        per_ep[ep] = (p, rot, vis, st, pcv, rcv, dmin, g)
    md = np.concatenate(moving_d)
    thr_log = otsu(md)
    d_g = 10 ** thr_log
    np.savez_compressed(OUT / "d2_hand_dist.npz", **dist_store)
    print(f"Otsu grasp threshold on moving frames: {d_g:.2f} cm (log10 {thr_log:.3f}); moving frames {len(md)}; "
          f"frac below {np.mean(md < thr_log):.3f}; rest frames frac below {np.mean(np.concatenate(rest_d) < thr_log):.3f}", flush=True)
    # ---------------- C. carries + relative pose variograms
    rel_r, rel_p, lev2, carry_rows = [], [], [], []
    for ep, (p, rot, vis, st, pcv, rcv, dmin, g) in per_ep.items():
        T = len(p)
        near = dmin.min(0) < d_g
        carry = keep_runs(bridge((~st) & near), LMIN) & (~st)
        carry = keep_runs(carry, LMIN)
        lab = g["contact"].max(0) > 0
        nrun = 0
        for a, b in runs(carry):
            side = int(np.argmin(np.median(dmin[:, a:b + 1], axis=1)))
            nrun += 1
            if b - a + 1 >= 40 and vis[a:b + 1].all():
                hj = J["LeftHand" if side == 0 else "RightHand"]
                Rh = R.from_quat(rcv[a:b + 1, hj])
                ph = pcv[a:b + 1, hj]
                Rrel = Rh.inv() * rot[a:b + 1]
                prel = np.einsum("tji,tj->ti", Rh.as_matrix(), p[a:b + 1] - ph)
                rel_r.append([np.mean(np.degrees((Rrel[l:] * Rrel[:-l].inv()).magnitude()) ** 2) for l in LAGS])
                rel_p.append([np.mean(np.sum((prel[l:] - prel[:-l]) ** 2, 1)) for l in LAGS])
                lev2.append(np.mean(np.sum((p[a:b + 1] - pcv[a:b + 1, J["Hips"]]) ** 2, 1)))
        carry_rows.append({"episode": ep, "carry_frac": float(carry.mean()), "n_carry_runs": nrun, "moving_frac": float((~st).mean()),
                           "contact_label_frac": float(lab.mean()), "agree_carry_vs_label": float(np.mean(carry == lab)),
                           "label_frac_within_carry": float(lab[carry].mean()) if carry.any() else None})
    A = np.stack([np.ones(4), LAGS[:4]], 1)
    gr, gp = np.mean(rel_r, 0), np.mean(rel_p, 0)
    ar, sr = np.linalg.lstsq(A, gr[:4], rcond=None)[0]
    ap, spp = np.linalg.lstsq(A, gp[:4], rcond=None)[0]
    pred_r = 3 * (Q_BR2 + Q_HR2)
    L2 = float(np.mean(lev2))
    pred_p = 3 * Q_BP2 + 3 * Q_HT2 + 2 * np.radians(np.sqrt(Q_HR2)) ** 2 * L2
    carry_res = {"grasp_threshold_cm": d_g, "otsu_log10": thr_log, "n_carry_runs_ge40": len(rel_r),
                 "rel_rot_gamma_deg2": gr.tolist(), "rel_pos_gamma_cm2": gp.tolist(), "lags": LAGS.tolist(),
                 "rel_rot_intercept": float(ar), "rel_rot_slope": float(sr), "rel_pos_intercept": float(ap), "rel_pos_slope": float(spp),
                 "pred_rot_slope_from_object_plus_root": pred_r, "pred_pos_slope_from_object_plus_root": float(pred_p),
                 "mean_sq_obj_minus_hips_cm2": L2,
                 "slip_rw_rot_deg2_per_frame_per_axis": float(max(0.0, (sr - pred_r) / 3)),
                 "slip_rw_pos_cm2_per_frame_per_axis": float(max(0.0, (spp - pred_p) / 3)),
                 "slip_white_rot_deg2_per_axis": float(max(0.0, ar / 6)), "slip_white_pos_cm2_per_axis": float(max(0.0, ap / 6)),
                 "episodes": carry_rows}
    print(json.dumps({k: v for k, v in carry_res.items() if k != "episodes"}, indent=1), flush=True)
    # ---------------- D. physics residuals at hand-free rest-run midpoints
    rows = []
    for m in ms:
        ep, obj = m["episode_index"], m["object"]
        p, rot, vis, st, pcv, rcv, dmin, g = per_ep[ep]
        up, e = np.array(FR[ep]["up_cv"]), FR[ep]["e_cv"]
        s = sp_store[obj]
        floor_only = bool(support_is_empty(m))
        free = st & (dmin.min(0) >= d_g) & vis
        for a, b in runs(st):
            sub = [(x, y) for (x, y) in runs(free[a:b + 1]) if y - x + 1 >= LMIN]
            if not sub:
                continue
            x, y = max(sub, key=lambda r: r[1] - r[0])
            tm = a + (x + y) // 2
            gdown = rot[tm].inv().apply(-up)                                    # world down in object frame
            ang = np.degrees(np.arccos(np.clip(s["n"] @ gdown, -1, 1)))
            order = np.argsort(ang)
            k1 = order[0]
            r = {"episode": ep, "object": obj, "t": int(tm), "run_len": int(b - a + 1), "free_len": int(y - x + 1), "floor_only": floor_only,
                 "tilt_res_deg": float(ang[k1]), "tilt_2nd_deg": float(ang[order[1]]) if len(order) > 1 else None,
                 "pose_prob": float(s["prob"][k1])}
            if floor_only:
                r["height_res_cm"] = float(p[tm] @ up + 100 * e - 100 * s["h_m"][k1])
            rows.append(r)
    # robust intercepts with the random-walk growth fixed from d1 (b0 = 0 convention): r^2 = A + B t
    res = {"n_rest_anchor_candidates": len(rows)}
    t = np.array([r["t"] for r in rows], float)
    tilt = np.array([r["tilt_res_deg"] for r in rows])
    B_tilt = 2 * Q_BR2                                                         # 2 tilt axes, deg^2/frame
    # median of a scaled chi2_2 is 1.386 * sigma^2 ; trimmed at 99% then mean
    sig2 = np.maximum(np.median(np.maximum(tilt ** 2 - B_tilt * t, 0)) / 1.386 / 2, 1e-6)
    for _ in range(5):
        keep = tilt ** 2 <= CHI2_2_99 * (2 * sig2 + B_tilt * t)
        sig2 = max(np.mean(np.maximum(tilt[keep] ** 2 - B_tilt * t[keep], 0)) / 2, 1e-6)
    Af = np.stack([np.ones(keep.sum()), t[keep]], 1)
    a_free, b_free = np.linalg.lstsq(Af, tilt[keep] ** 2, rcond=None)[0]
    res["tilt"] = {"sigma_model_deg_per_axis": float(np.sqrt(sig2)), "n_kept": int(keep.sum()), "n_total": int(len(tilt)),
                   "B_fixed_deg2_per_frame": B_tilt, "free_fit_intercept_deg2": float(a_free), "free_fit_slope_deg2_per_frame": float(b_free),
                   "median_tilt_res_deg": float(np.median(tilt)), "frac_2nd_within_gate": None}
    fl = np.array([("height_res_cm" in r) for r in rows])
    hres = np.array([r["height_res_cm"] for r in rows if "height_res_cm" in r])
    th = t[fl]
    B_h = Q_BP2
    s2h = max(np.median(np.maximum(hres ** 2 - B_h * th, 0)) / 0.455, 1e-4)
    for _ in range(5):
        kh = hres ** 2 <= CHI2_1_99 * (s2h + B_h * th)
        s2h = max(np.mean(np.maximum(hres[kh] ** 2 - B_h * th[kh], 0)), 1e-4)
    Ah = np.stack([np.ones(kh.sum()), th[kh]], 1)
    ah, bh = np.linalg.lstsq(Ah, hres[kh] ** 2, rcond=None)[0]
    res["height"] = {"sigma_model_cm": float(np.sqrt(s2h)), "n_kept": int(kh.sum()), "n_total": int(len(hres)), "B_fixed_cm2_per_frame": B_h,
                     "free_fit_intercept_cm2": float(ah), "free_fit_slope_cm2_per_frame": float(bh), "mean_res_kept_cm": float(np.mean(hres[kh])),
                     "median_abs_res_cm": float(np.median(np.abs(hres)))}
    res["rows"] = rows
    (OUT / "d2_physics_residuals.json").write_text(json.dumps(res, indent=1))
    (OUT / "d2_carry.json").write_text(json.dumps(carry_res, indent=1))
    (OUT / "d2_stable_poses.json").write_text(json.dumps(sp_rows, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
