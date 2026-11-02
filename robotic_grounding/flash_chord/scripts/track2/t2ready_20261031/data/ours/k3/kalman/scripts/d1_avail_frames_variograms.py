"""DESIGN STEP 1 (NOISY-ONLY; reads Tier-2 input + Tier-2 retarget output only).

1. Data availability per episode (mesh, ground plane, support stage empty or not, SOMA joints, contact labels).
2. Frame maps: SOMA-leveled -> G1 robot (fit on Hips), OpenCV -> G1 robot (fit on the object track, lift removed).
3. Object rest masks (ZUPT v1 rule) and the object bias variogram on rest runs (pos, rot, per world axis).
4. Human-root bias variogram from stance feet (foot orientation -> root rotation step; toe position -> root translation
   step after removing the rotation lever arm about the Hips), plus a direct Hips variogram on double-stance runs.
Writes KOUT/design/d1_*.json, d1_masks.npz
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from kcommon import (FPS, J, KOUT, T2, ground_plane_cv, kabsch, load_g1, metas, noise_sigmas, object_rest_mask,  # noqa: E402
                     object_track, runs, static_mask, support_is_empty, wxyz_to_rot)

OUT = KOUT / "design"
OUT.mkdir(parents=True, exist_ok=True)
LAGS = np.array([1, 2, 4, 8, 16, 32])
FIT_LAGS = 4   # fit gamma(l) = a + s*l on the first 4 lags (1,2,4,8)


def vario(segs_p, segs_r, lags):
    """segs_p: list of (n,3) cm arrays; segs_r: list of Rotation. Returns per-run variograms (pos cm^2, rot deg^2, per-axis)."""
    gp, gr, gpa, gra = [], [], [], []
    for sp, sr in zip(segs_p, segs_r):
        if sp is not None:
            d = [sp[l:] - sp[:-l] for l in lags]
            gp.append([np.mean(np.sum(x ** 2, 1)) for x in d])
            gpa.append([np.mean(x ** 2, 0) for x in d])
        if sr is not None:
            w = [np.degrees((sr[l:] * sr[:-l].inv()).as_rotvec()) for l in lags]      # world-frame increments
            gr.append([np.mean(np.sum(x ** 2, 1)) for x in w])
            gra.append([np.mean(x ** 2, 0) for x in w])
    return (np.array(gp) if gp else None, np.array(gr) if gr else None, np.array(gpa) if gpa else None,
            np.array(gra) if gra else None)


def fit_line(g, lags=LAGS, k=FIT_LAGS):
    A = np.stack([np.ones(k), lags[:k]], 1)
    a, s = np.linalg.lstsq(A, g[:k], rcond=None)[0]
    return float(a), float(s)


def boot_ci(per_run, n=2000, seed=0):
    """Bootstrap over runs of the fitted slope and intercept of the mean variogram."""
    rng = np.random.default_rng(seed)
    m = len(per_run)
    sl, ic = [], []
    for _ in range(n):
        g = per_run[rng.integers(0, m, m)].mean(0)
        a, s = fit_line(g)
        sl.append(s); ic.append(a)
    return [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))], [float(np.percentile(ic, 2.5)), float(np.percentile(ic, 97.5))]


def main():
    avail, frames_fit, masks, ep_rows = [], [], {}, []
    obj_p_runs, obj_r_runs = {80: [], 40: []}, {80: []}
    foot_p_runs, foot_r_runs, foot_lever2 = {80: [], 40: []}, {80: [], 40: []}, {80: [], 40: []}
    hip_p_runs, hip_r_runs = [], []
    for m in metas():
        ep, seq, obj = m["episode_index"], m["sequence_id"], m["object"]
        p, rot, vis, raw = object_track(ep)
        g = load_g1(seq)
        names_ok = True
        # ---------- availability
        mesh = T2 / m["mesh"]
        a = {"episode": ep, "object": obj, "frames": int(len(p)), "visible_frac": float(vis.mean()),
             "mesh_exists": mesh.exists(), "ground_plane_exists": (T2 / m["ground_plane"]).exists(),
             "support_stage_empty": bool(support_is_empty(m)), "g1_frames": int(len(g["obj_p"])),
             "soma_joints": list(g["soma_joints"].shape), "contact_label_frames_LR": g["contact"].sum(1).tolist(),
             "object_lift_m": float(g["ground_alignment"]["object_ground_correction"]["applied_lift"])}
        # ---------- frame maps
        hips_s = g["soma_joints"][:, J["Hips"]]
        Rs2r, ts2r = kabsch(hips_s, g["src_root_t"])
        res_s2r = float(np.abs(hips_s @ Rs2r.T + ts2r - g["src_root_t"]).max())
        lift = np.array([0, 0, a["object_lift_m"]])
        ok = vis & (len(g["obj_p"]) == len(p))
        Bcv2r, bcv2r = kabsch(p[ok] / 100.0, g["obj_p"][ok] - lift)
        res_pos = float(np.linalg.norm((p[ok] / 100.0) @ Bcv2r.T + bcv2r - (g["obj_p"][ok] - lift), axis=1).max())
        Rr = wxyz_to_rot(g["obj_q"][ok])
        C = (R.from_matrix(np.repeat(Bcv2r[None], ok.sum(), 0)) * rot[ok]).inv() * Rr      # R_robot = B R_cv C
        Cm = C.mean()
        res_rot = float(np.degrees((Cm.inv() * C).magnitude()).max())
        hips_r = hips_s @ Rs2r.T + ts2r
        hips_cv = (hips_r - bcv2r) @ Bcv2r          # inverse rigid map robot -> cv (metres)
        up, e = ground_plane_cv(m)
        sgn = np.sign(np.median(hips_cv @ up + e))
        up, e = up * sgn, e * sgn
        frames_fit.append({"episode": ep, "soma_to_robot_maxres_m": res_s2r, "soma_to_robot_rot_is_fixed": True,
                           "cv_to_robot_pos_maxres_m": res_pos, "cv_to_robot_rot_maxres_deg": res_rot,
                           "object_frame_offset_C_deg": float(np.degrees(Cm.magnitude())),
                           "Rs2r": Rs2r.tolist(), "ts2r": ts2r.tolist(), "Bcv2r": Bcv2r.tolist(), "bcv2r": bcv2r.tolist(),
                           "C_xyzw": Cm.as_quat().tolist(), "up_cv": up.tolist(), "e_cv": float(e),
                           "hips_height_median_m": float(np.median(hips_cv @ up + e))})
        avail.append(a)
        # ---------- object rest mask + variogram
        st, sp, sr = object_rest_mask(p, rot)
        masks[f"obj_{ep}"] = st
        for (s0, s1) in runs(st):
            n = s1 - s0 + 1
            if n >= 80:
                obj_p_runs[80].append(p[s0:s1 + 1]); obj_r_runs[80].append(rot[s0:s1 + 1])
            if n >= 40:
                obj_p_runs[40].append(p[s0:s1 + 1])
        # ---------- feet stance (in SOMA-leveled frame, cm; rigid maps do not change variograms)
        sj = g["soma_joints"] * 100.0
        foot_stance = {}
        for side, (jf, jt) in {"L": (J["LeftFoot"], J["LeftToeBase"]), "R": (J["RightFoot"], J["RightToeBase"])}.items():
            fp = sj[:, jt]
            fr = wxyz_to_rot(g["soma_joints_wxyz"][:, jf])
            fsp, fsr = noise_sigmas(fp, fr)
            fst = static_mask(fp, fr, fsp, np.radians(fsr))
            foot_stance[side] = fst
            masks[f"foot{side}_{ep}"] = fst
            for (s0, s1) in runs(fst):
                n = s1 - s0 + 1
                lev = np.mean(np.sum((sj[s0:s1 + 1, jt] - sj[s0:s1 + 1, J["Hips"]]) ** 2, 1))
                for L in (80, 40):
                    if n >= L:
                        foot_p_runs[L].append(fp[s0:s1 + 1]); foot_r_runs[L].append(fr[s0:s1 + 1]); foot_lever2[L].append(lev)
        both = foot_stance["L"] & foot_stance["R"]
        hr = wxyz_to_rot(g["soma_joints_wxyz"][:, J["Hips"]])
        for (s0, s1) in runs(both):
            if s1 - s0 + 1 >= 80:
                hip_p_runs.append(sj[s0:s1 + 1, J["Hips"]]); hip_r_runs.append(hr[s0:s1 + 1])
        ep_rows.append({"episode": ep, "object": obj, "rest_frac": float(st.mean()), "n_rest_runs": len(runs(st)),
                        "sig_p_floor": sp, "sig_r_floor": sr, "footL_stance_frac": float(foot_stance["L"].mean()),
                        "footR_stance_frac": float(foot_stance["R"].mean()), "double_stance_frac": float(both.mean())})
        print(f"ep{ep:02d} {obj:<20} res s2r {res_s2r:.1e} cv2r pos {res_pos:.1e} rot {res_rot:.2e} C {np.degrees(Cm.magnitude()):.2f}deg "
              f"up.hips {frames_fit[-1]['hips_height_median_m']:.2f} rest {st.mean():.2f} stance L {foot_stance['L'].mean():.2f} "
              f"R {foot_stance['R'].mean():.2f} empty_support {a['support_stage_empty']}", flush=True)

    res = {"lags": LAGS.tolist(), "fit": f"gamma(l) = a + s*l least squares on lags {LAGS[:FIT_LAGS].tolist()}"}
    # object
    gp, gr, gpa, gra = vario(obj_p_runs[80], obj_r_runs[80], LAGS)
    ap, sp_ = fit_line(gp.mean(0)); ar, sr_ = fit_line(gr.mean(0))
    res["object"] = {"n_runs_ge80": len(obj_p_runs[80]), "gamma_pos_cm2": gp.mean(0).tolist(), "gamma_rot_deg2": gr.mean(0).tolist(),
                     "pos_intercept_cm2": ap, "pos_slope_cm2_per_frame": sp_, "rot_intercept_deg2": ar, "rot_slope_deg2_per_frame": sr_,
                     "pos_slope_ci95": boot_ci(gp)[0], "pos_intercept_ci95": boot_ci(gp)[1],
                     "rot_slope_ci95": boot_ci(gr)[0], "rot_intercept_ci95": boot_ci(gr)[1],
                     "pos_slope_per_world_axis": [fit_line(gpa.mean(0)[:, k])[1] for k in range(3)],
                     "rot_slope_per_world_axis": [fit_line(gra.mean(0)[:, k])[1] for k in range(3)],
                     "q_b_pos_cm_per_frame_per_axis": float(np.sqrt(sp_ / 3)), "q_b_rot_deg_per_frame_per_axis": float(np.sqrt(sr_ / 3))}
    gp40 = vario(obj_p_runs[40], [None] * len(obj_p_runs[40]), LAGS[:5])[0]
    res["object"]["n_runs_ge40"] = len(obj_p_runs[40])
    # feet -> root
    for L in (80, 40):
        lags = LAGS if L == 80 else LAGS[:5]
        fp_, fr_, _, fra = vario(foot_p_runs[L], foot_r_runs[L], lags)
        if fp_ is None:
            res[f"feet_ge{L}"] = {"n_runs": 0}
            continue
        a_p, s_p = fit_line(fp_.mean(0), lags); a_r, s_r = fit_line(fr_.mean(0), lags)
        q_hr_deg = np.sqrt(max(s_r, 0) / 3)
        lever = float(np.mean(foot_lever2[L]))
        rot_contrib = 2 * np.radians(q_hr_deg) ** 2 * lever
        q_ht = np.sqrt(max(0.0, (s_p - rot_contrib) / 3))
        res[f"feet_ge{L}"] = {"n_runs": len(foot_p_runs[L]), "gamma_pos_cm2": fp_.mean(0).tolist(), "gamma_rot_deg2": fr_.mean(0).tolist(),
                              "pos_intercept_cm2": a_p, "pos_slope": s_p, "rot_intercept_deg2": a_r, "rot_slope": s_r,
                              "rot_slope_ci95": boot_ci(fr_)[0] if L == 80 else None, "pos_slope_ci95": boot_ci(fp_)[0] if L == 80 else None,
                              "rot_slope_per_world_axis": [fit_line(fra.mean(0)[:, k], lags)[1] for k in range(3)],
                              "mean_sq_lever_toe_minus_hips_cm2": lever, "rotation_lever_contribution_cm2_per_frame": float(rot_contrib),
                              "q_root_rot_deg_per_frame_per_axis": float(q_hr_deg), "q_root_trans_cm_per_frame_per_axis": float(q_ht)}
    if hip_p_runs:
        hp, hr_, _, _ = vario(hip_p_runs, hip_r_runs, LAGS)
        res["hips_double_stance_ge80"] = {"n_runs": len(hip_p_runs), "pos_slope": fit_line(hp.mean(0))[1], "rot_slope": fit_line(hr_.mean(0))[1],
                                          "gamma_pos_cm2": hp.mean(0).tolist(), "gamma_rot_deg2": hr_.mean(0).tolist()}
    (OUT / "d1_variograms.json").write_text(json.dumps(res, indent=1))
    (OUT / "d1_availability.json").write_text(json.dumps(avail, indent=1))
    (OUT / "d1_frames.json").write_text(json.dumps(frames_fit, indent=1))
    (OUT / "d1_episode_rows.json").write_text(json.dumps(ep_rows, indent=1))
    np.savez_compressed(OUT / "d1_masks.npz", **masks)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
