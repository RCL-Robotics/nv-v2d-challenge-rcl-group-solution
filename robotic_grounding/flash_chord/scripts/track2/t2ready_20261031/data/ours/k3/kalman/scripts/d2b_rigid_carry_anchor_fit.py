"""DESIGN STEP 2b (NOISY-ONLY). Builds on d1/d2 outputs.

1. Rigid-carry detection = "ZUPT in the hand frame": the verbatim ZUPT v1 static rule applied to the hand->object relative
   pose (relative position in the hand frame, relative rotation), with per-frame noise sigmas PREDICTED from the d1
   random-walk steps (object bias + root bias incl. its lever arm), AND a hand within the d2 Otsu grasp threshold, AND the
   object not at rest, AND visible. Slip budget re-estimated from the relative-pose variogram on rigid runs >= 40 frames.
2. Anchor model-error fits (rest-run hand-free midpoints from d2):
   tilt residual r ~ (1-eps) Rayleigh(s(t)) + eps U(0, 90 deg),  s(t)^2 = sig^2 + q_br^2 min(t, t_s)   (per-axis)
   height residual r ~ (1-eps) N(mu, sig_h^2 + q_bp^2 min(t, t_s)) + eps U(-20, 20 cm)
   q_br, q_bp fixed from the d1 rest-run variogram; (sig, t_s, eps) [+ mu] by maximum likelihood on a grid.
Writes KOUT/design/d2b_*.json, d2b_rigid_masks.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from kcommon import J, KOUT, LMIN, keep_runs, load_g1, metas, object_track, runs, static_mask  # noqa: E402
from d2_mesh_grasp_physics import human_cv  # noqa: E402

OUT = KOUT / "design"
D1 = json.loads((OUT / "d1_variograms.json").read_text())
FR = {r["episode"]: r for r in json.loads((OUT / "d1_frames.json").read_text())}
M1 = np.load(OUT / "d1_masks.npz")
DIST = np.load(OUT / "d2_hand_dist.npz")
CARRY = json.loads((OUT / "d2_carry.json").read_text())
PHYS = json.loads((OUT / "d2_physics_residuals.json").read_text())
D_G = CARRY["grasp_threshold_cm"]
Q_BP2 = D1["object"]["pos_slope_cm2_per_frame"] / 3
Q_BR2 = D1["object"]["rot_slope_deg2_per_frame"] / 3
Q_HR2 = D1["feet_ge80"]["rot_slope"] / 3
Q_HT2 = D1["hips_double_stance_ge80"]["gamma_pos_cm2"][0] / 3
LAGS = np.array([1, 2, 4, 8, 16])


def rigid_masks():
    out, rr, rp, lev, rows = {}, [], [], [], []
    for m in metas():
        ep, seq = m["episode_index"], m["sequence_id"]
        p, rot, vis, _ = object_track(ep)
        g = load_g1(seq)
        pcv, rq, T, _ = human_cv(g, FR[ep])
        dmin = DIST[f"d_{ep}"]
        st = M1[f"obj_{ep}"]
        hips = pcv[:, J["Hips"]]
        L2 = np.sum((p - hips) ** 2, 1)
        sig_r = np.sqrt(Q_BR2 + Q_HR2)                                                 # deg/frame/axis
        sig_p = np.sqrt(Q_BP2 + Q_HT2 + (2.0 / 3.0) * np.radians(np.sqrt(Q_HR2)) ** 2 * np.median(L2))   # cm/frame/axis
        side_masks = []
        for si, hj in enumerate((J["LeftHand"], J["RightHand"])):
            Rh = R.from_quat(rq[:, hj])
            Rrel = Rh.inv() * rot
            prel = np.einsum("tji,tj->ti", Rh.as_matrix(), p - pcv[:, hj])
            rel_static = static_mask(prel, Rrel, sig_p, np.radians(sig_r))
            msk = keep_runs(rel_static & (dmin[si] < D_G) & (~st) & vis, LMIN)
            side_masks.append((msk, Rrel, prel))
        order = np.argsort([-sm[0].sum() for sm in side_masks])                      # dominant hand first
        used = np.zeros(T, bool)
        sidearr = -np.ones(T, int)
        for si in order:
            msk = keep_runs(side_masks[si][0] & ~used, LMIN)
            sidearr[msk] = si
            used |= msk
            Rrel, prel = side_masks[si][1], side_masks[si][2]
            for a, b in runs(msk):
                if b - a + 1 >= 40:
                    seg_r, seg_p = Rrel[a:b + 1], prel[a:b + 1]
                    rr.append([np.mean(np.degrees((seg_r[l:] * seg_r[:-l].inv()).magnitude()) ** 2) for l in LAGS])
                    rp.append([np.mean(np.sum((seg_p[l:] - seg_p[:-l]) ** 2, 1)) for l in LAGS])
                    lev.append(np.mean(L2[a:b + 1]))
        out[f"side_{ep}"] = sidearr
        rows.append({"episode": ep, "object": m["object"], "rigid_carry_frac": float(used.mean()), "moving_frac": float((~st).mean()),
                     "n_runs": len(runs(used)), "sig_p_rel": float(sig_p), "sig_r_rel": float(sig_r)})
        print(rows[-1], flush=True)
    A = np.stack([np.ones(4), LAGS[:4]], 1)
    gr, gp = np.mean(rr, 0), np.mean(rp, 0)
    ar, sr = np.linalg.lstsq(A, gr[:4], rcond=None)[0]
    ap, spp = np.linalg.lstsq(A, gp[:4], rcond=None)[0]
    pred_r = 3 * (Q_BR2 + Q_HR2)
    pred_p = 3 * Q_BP2 + 3 * Q_HT2 + 2 * np.radians(np.sqrt(Q_HR2)) ** 2 * float(np.mean(lev))
    res = {"n_rigid_runs_ge40": len(rr), "rel_rot_gamma_deg2": gr.tolist(), "rel_pos_gamma_cm2": gp.tolist(),
           "rel_rot_intercept": float(ar), "rel_rot_slope": float(sr), "rel_pos_intercept": float(ap), "rel_pos_slope": float(spp),
           "pred_rot_slope": pred_r, "pred_pos_slope": float(pred_p),
           "slip_rw_rot_deg2_per_frame_per_axis": float(max(0.0, (sr - pred_r) / 3)),
           "slip_rw_pos_cm2_per_frame_per_axis": float(max(0.0, (spp - pred_p) / 3)),
           "slip_white_rot_deg2_per_axis": float(max(0.0, ar / 6)), "slip_white_pos_cm2_per_axis": float(max(0.0, ap / 6)),
           "rigid_carry_frac_mean": float(np.mean([r["rigid_carry_frac"] for r in rows])),
           "moving_frac_mean": float(np.mean([r["moving_frac"] for r in rows])), "episodes": rows}
    np.savez_compressed(OUT / "d2b_rigid_masks.npz", **out)
    return res


def fit_tilt(t, r, q2):
    best = (-np.inf, None)
    for sig in np.arange(0.25, 15.01, 0.25):
        for ts in np.concatenate([np.arange(25, 1001, 25), [1e9]]):
            s2 = sig ** 2 + q2 * np.minimum(t, ts)
            ray = r / s2 * np.exp(-r ** 2 / (2 * s2))
            for eps in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
                ll = np.sum(np.log((1 - eps) * ray + eps / 90.0))
                if ll > best[0]:
                    best = (ll, (sig, ts, eps))
    return best


def fit_height(t, r, q2):
    best = (-np.inf, None)
    for mu in np.arange(-2.0, 2.01, 0.05):
        for sig in np.arange(0.1, 5.01, 0.05):
            for ts in np.concatenate([np.arange(25, 1001, 25), [1e9]]):
                s2 = sig ** 2 + q2 * np.minimum(t, ts)
                gau = np.exp(-(r - mu) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)
                for eps in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
                    ll = np.sum(np.log((1 - eps) * gau + eps / 40.0))
                    if ll > best[0]:
                        best = (ll, (mu, sig, ts, eps))
    return best


def main():
    carry = rigid_masks()
    (OUT / "d2b_rigid_carry.json").write_text(json.dumps(carry, indent=1))
    print(json.dumps({k: v for k, v in carry.items() if k != "episodes"}, indent=1), flush=True)
    rows = PHYS["rows"]
    t = np.array([r["t"] for r in rows], float)
    tilt = np.array([r["tilt_res_deg"] for r in rows])
    llt, (sig, ts, eps) = fit_tilt(t, tilt, Q_BR2)
    fl = np.array(["height_res_cm" in r for r in rows])
    hres = np.array([r["height_res_cm"] for r in rows if "height_res_cm" in r])
    llh, (mu, sh, tsh, epsh) = fit_height(t[fl], hres, Q_BP2)
    res = {"tilt": {"sigma_model_deg_per_axis": float(sig), "t_sat_frames": float(ts), "outlier_eps": float(eps), "loglik": float(llt),
                    "implied_sat_std_per_axis_deg": float(np.sqrt(sig ** 2 + Q_BR2 * min(ts, 1e9))), "n": int(len(tilt))},
           "height": {"mu_cm": float(mu), "sigma_model_cm": float(sh), "t_sat_frames": float(tsh), "outlier_eps": float(epsh),
                      "loglik": float(llh), "n": int(len(hres))}}
    (OUT / "d2b_anchor_fit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
