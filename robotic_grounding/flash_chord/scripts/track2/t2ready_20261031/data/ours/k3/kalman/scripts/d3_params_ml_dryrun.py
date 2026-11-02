"""DESIGN STEP 3 (NOISY-ONLY).
1. Assemble the global parameters from d1/d2/d2b (noisy-only derivations) -> frozen_params.json (draft until frozen).
2. Maximum-likelihood q_a (position) and q_al (rotation) of the constant-velocity true-motion prior under the K1 model
   (bias random walk fixed, ZV pseudo-measurements on rest frames), grid search on the innovation log-likelihood.
   Per-episode curves are computed; the value USED is the POOLED ML (argmax of the sum of the 30 per-episode
   log-likelihood curves), one global q_a and one global q_al. Reason (noisy-only, decided before any clean scoring):
   per-episode ML picked q_al ~1e-3 rad^2/s^3 for slow large rotations (ep00 116 deg tip, ep08, ep10, ep23) and K1 then
   attributed up to 71 deg of real rotation to the bias, inconsistent with the bias level seen at rest anchors (d2b).
3. NOISY-ONLY dry run of K0-K3: diagnostics (anchor/hand acceptance, correction sizes); writes the variant poses to
   KOUT/outputs/kalman_poses.npz. No clean data, no scores.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from kalman_core import BP, BT, NX, kf_rts  # noqa: E402
from kcommon import DT, KOUT  # noqa: E402

DES = KOUT / "design"
D1 = json.loads((DES / "d1_variograms.json").read_text())
D2C = json.loads((DES / "d2_carry.json").read_text())
D2B = json.loads((DES / "d2b_rigid_carry.json").read_text())
AF = json.loads((DES / "d2b_anchor_fit.json").read_text())
GRID_QA = 10 ** np.arange(0.0, 7.01, 0.25)        # cm^2/s^3
GRID_QAL = 10 ** np.arange(-3.0, 4.01, 0.25)      # rad^2/s^3
POS_IDX = np.r_[0:6, 12:15]
ROT_IDX = np.r_[6:12, 15:18]


def global_params():
    q_bp = float(np.sqrt(D1["object"]["pos_slope_cm2_per_frame"] / 3))
    q_br = float(np.sqrt(D1["object"]["rot_slope_deg2_per_frame"] / 3))
    q_hr = float(np.sqrt(D1["feet_ge80"]["rot_slope"] / 3))
    q_ht = float(np.sqrt(D1["hips_double_stance_ge80"]["gamma_pos_cm2"][0] / 3))
    sw_p, sw_r = 0.01, 0.01
    return {
        "q_bp_cm": q_bp, "q_br_deg": q_br, "q_root_rot_deg": q_hr, "q_root_trans_cm": q_ht,
        "q_d_t_cm": float(np.sqrt(q_ht ** 2 + D2B["slip_rw_pos_cm2_per_frame_per_axis"])),
        "q_d_th_deg": float(np.sqrt(q_hr ** 2 + D2B["slip_rw_rot_deg2_per_frame_per_axis"])),
        "slip_white_pos_cm": float(max(np.sqrt(D2B["slip_white_pos_cm2_per_axis"]), sw_p)),
        "slip_white_rot_deg": float(max(np.sqrt(D2B["slip_white_rot_deg2_per_axis"]), sw_r)),
        "sigma_w_pos_cm": sw_p, "sigma_w_rot_deg": sw_r,
        "sigma_zv_pos_cm_s": float(np.sqrt(2) * sw_p / DT), "sigma_zv_rot_deg_s": float(np.sqrt(2) * sw_r / DT),
        "sigma_tilt_deg": AF["tilt"]["sigma_model_deg_per_axis"], "sigma_height_cm": AF["height"]["sigma_model_cm"],
        "height_mu_cm": AF["height"]["mu_cm"],
        "P_b0_rot_deg": AF["tilt"]["sigma_model_deg_per_axis"], "P_b0_pos_cm": AF["height"]["sigma_model_cm"],
        "grasp_threshold_cm": D2C["grasp_threshold_cm"],
    }


def ml_episode(args):
    ep, prm = args
    from kfilter import Episode, load_params  # noqa
    from kcommon import metas
    m = [x for x in metas() if x["episode_index"] == ep][0]
    prm = dict(prm); prm["q_a_cm2_s3"] = {str(ep): 1.0}; prm["q_al_rad2_s3"] = {str(ep): 1.0}
    E = Episode(m, prm)
    meas = E.base_meas(zv=True)
    out = {}
    for name, idx, grid in (("pos", POS_IDX, GRID_QA), ("rot", ROT_IDX, GRID_QAL)):
        mg = [[mm for mm in lst if np.any(mm[1][:, idx] != 0)] for lst in meas]
        lls = []
        for q in grid:
            E.qa, E.qal = (q, 1.0) if name == "pos" else (1.0, q)
            F, Q = E.FQ()
            x0, P0 = E.prior()
            _, _, ll, _ = kf_rts(E.T, F, Q, x0, P0, mg, active=idx)
            lls.append(ll)
        lls = np.array(lls)
        k = int(np.argmax(lls))
        out[name] = {"q_ml": float(grid[k]), "at_grid_edge": bool(k in (0, len(grid) - 1)), "ll": lls.tolist()}
    return ep, out


def dry_episode(args):
    ep, prm = args
    from kfilter import Episode
    from kcommon import metas
    m = [x for x in metas() if x["episode_index"] == ep][0]
    E = Episode(m, prm)
    res, poses = {"episode": ep, "object": E.obj, "floor_only": E.floor_only}, {}
    poses[f"NONE__{ep}"] = np.concatenate([E.p / 100.0, E.rot.as_quat()], 1)[E.vis]     # frame-alignment check at scoring
    for v in ("K0", "K1", "K2", "K3"):
        t0 = time.time()
        pose, X, diag = E.variant(v)
        poses[f"{v}__{ep}"] = pose
        vis = E.vis
        res[v] = {"mean_abs_bias_rot_deg": float(np.degrees(np.linalg.norm(X[vis][:, BT], axis=1)).mean()),
                  "max_abs_bias_rot_deg": float(np.degrees(np.linalg.norm(X[vis][:, BT], axis=1)).max()),
                  "mean_abs_bias_pos_cm": float(np.linalg.norm(X[vis][:, BP], axis=1).mean()),
                  "secs": round(time.time() - t0, 2), **diag}
    return res, poses


def main():
    from kcommon import metas
    eps = [m["episode_index"] for m in metas()]
    prm = global_params()
    with Pool(8) as pool:
        ml = dict(pool.map(ml_episode, [(ep, prm) for ep in eps]))
    Lp = np.sum([ml[ep]["pos"]["ll"] for ep in eps], 0)
    Lr = np.sum([ml[ep]["rot"]["ll"] for ep in eps], 0)
    qa_pool, qal_pool = float(GRID_QA[int(np.argmax(Lp))]), float(GRID_QAL[int(np.argmax(Lr))])
    prm["q_a_pooled_ml_cm2_s3"], prm["q_al_pooled_ml_rad2_s3"] = qa_pool, qal_pool
    prm["q_a_cm2_s3"] = {str(ep): qa_pool for ep in eps}
    prm["q_al_rad2_s3"] = {str(ep): qal_pool for ep in eps}
    (DES / "d3_ml.json").write_text(json.dumps({str(k): v for k, v in ml.items()}, indent=1))
    (DES / "frozen_params.json").write_text(json.dumps(prm, indent=1))
    for ep in eps:
        print(f"ep{ep:02d} q_a {ml[ep]['pos']['q_ml']:.3g} edge {ml[ep]['pos']['at_grid_edge']}  q_al {ml[ep]['rot']['q_ml']:.3g} "
              f"edge {ml[ep]['rot']['at_grid_edge']}", flush=True)
    with Pool(8) as pool:
        out = pool.map(dry_episode, [(ep, prm) for ep in eps])
    rows, store = [], {}
    for r, p in out:
        rows.append(r); store.update(p)
    (KOUT / "outputs").mkdir(exist_ok=True)
    np.savez_compressed(KOUT / "outputs" / "kalman_poses.npz", **store)
    (DES / "d3_dryrun_diag.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r), flush=True)
    agg = {}
    for v in ("K0", "K1", "K2", "K3"):
        agg[v] = {k: float(np.mean([r[v][k] for r in rows if k in r[v]])) for k in rows[0][v] if isinstance(rows[0][v][k], (int, float))}
        agg[v].update({k: int(np.sum([r[v].get(k, 0) for r in rows])) for k in
                       ("rest_runs", "candidates", "tilt_accepted", "height_accepted", "tilt_no_pose_in_gate", "carry_runs", "hand_frames",
                        "hand_frames_gated") if any(k in r[v] for r in rows)})
    (DES / "d3_dryrun_summary.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
