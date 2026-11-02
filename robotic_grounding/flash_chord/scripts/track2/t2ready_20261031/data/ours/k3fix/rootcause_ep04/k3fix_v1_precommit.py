"""Proposed minimal fix of the frozen K3 smoother (NOISY-ONLY design; no clean data, no new free constants).

FIX-A  (root cause) rotational zero-velocity only on rest runs whose RAW orientation stays inside the bias random-walk
       envelope of the frozen model: stat = max over lags k of |rot(a+k) rot(a)^-1|^2 / (k q_br^2) must be below the 99%
       quantile of that same max statistic for a pure random walk of the run's length (mode 'calibrated', model-only
       Monte Carlo, s6_null_q99.npz; ~15-19). q_br is the frozen d1 rotation step. Mode 'pointwise' (first draft) uses
       chi2_3(99%) = 11.34, which ignores the max over lags and rejects 5-19% of pure-walk runs. A run that fails keeps its
       POSITION zero-velocity (v=0) but not w=0, so a real rotation at rest (e.g. a bat rolling about its long axis while
       the position is static) is no longer pushed into the unbounded bias state b_th.
FIX-C  (guard) a stable-pose tilt candidate is accepted only if the correction it implies relative to the RAW track,
       z_k, lies inside the noisy-only bias envelope: z_k^T z_k / (s_sat^2 + sigma_tilt^2) < chi2_2(99%), with
       s_sat = 14.92 deg/axis (d2b implied saturation std, noisy-only). Stops the gate, which is centred on a possibly
       corrupted K1 estimate, from locking on a pose ~90 deg from the raw track (e.g. an upright bat).
Everything else is the frozen kfilter.Episode code path.
"""
import json
import sys
from pathlib import Path

import numpy as np

KOUT = Path("/root/autodl-tmp/haoran.zhang/tmp_analysis/tier2/kalman")
sys.path.insert(0, str(KOUT / "scripts"))
from kalman_core import BT, NX, P_, V_, W_, PH, BP  # noqa: E402
from kfilter import CHI2_1_99, CHI2_2_99, I3, Episode, Hm, horiz_basis, min_rotvec  # noqa: E402
from kcommon import LMIN, runs  # noqa: E402

CHI2_3_99 = 11.344867
SAT_DEG = json.loads((KOUT / "design/d2b_anchor_fit.json").read_text())["tilt"]["implied_sat_std_per_axis_deg"]
# model-only null (s6_null_calibration.py): 99% quantile of max_k |S_k|^2/k for a pure 3D random walk, per run length n
_Q99 = np.load(Path(__file__).parent / "s6_null_q99.npz")["q99"]


def run_threshold(n, mode):
    """'pointwise': chi2_3(99%) = 11.34 (first draft, over-rejects 5-19% of pure-walk runs because of the max over lags);
    'calibrated': the 99% quantile of the max-over-lags statistic under the frozen random-walk model (1% per run)."""
    return CHI2_3_99 if mode == "pointwise" else float(_Q99[min(max(n, 2), len(_Q99) - 1)])


def rest_run_rotation_tests(rot, rest, q_br_rad):
    """Per rest run: max_k |rot(a+k) rot(a)^-1|^2 / (k q^2) (chi2_3 under the frozen bias random walk)."""
    out = []
    for a, b in runs(rest):
        k = np.arange(1, b - a + 1)
        if len(k) == 0:
            out.append((a, b, 0.0, 0)); continue
        ang = (rot[a + k] * rot[a].inv()).magnitude()
        st = ang ** 2 / (k * q_br_rad ** 2)
        out.append((a, b, float(st.max()), int(k[np.argmax(st)])))
    return out


class EpisodeFix(Episode):
    def __init__(self, m, prm, fix_zupt=True, fix_anchor=True, mode="calibrated"):
        super().__init__(m, prm)
        self.fix_zupt, self.fix_anchor, self.mode = fix_zupt, fix_anchor, mode
        self.run_tests = rest_run_rotation_tests(self.rot, self.rest, np.radians(prm["q_br_deg"]))
        self.rest_rot = self.rest.copy()
        self.run_fail = []
        for a, b, st, _ in self.run_tests:
            f = st >= run_threshold(b - a + 1, mode)
            self.run_fail.append(bool(f))
            if fix_zupt and f:
                self.rest_rot[a:b + 1] = False
        self.anchor_rejected = []

    def base_meas(self, zv=True, ll_from=2):
        pr = self.prm
        Rp, Rr = pr["sigma_w_pos_cm"] ** 2 * I3, np.radians(pr["sigma_w_rot_deg"]) ** 2 * I3
        Rzv_p, Rzv_r = pr["sigma_zv_pos_cm_s"] ** 2 * I3, np.radians(pr["sigma_zv_rot_deg_s"]) ** 2 * I3
        Hp, Hr = Hm([(P_, I3), (BP, I3)]), Hm([(PH, I3), (BT, I3)])
        Hv, Hw = Hm([(V_, I3)]), Hm([(W_, I3)])
        meas = [[] for _ in range(self.T)]
        for t in range(self.T):
            if zv and self.rest[t]:
                meas[t].append((np.zeros(3), Hv, Rzv_p, False, None))
                if self.rest_rot[t]:                                   # FIX-A
                    meas[t].append((np.zeros(3), Hw, Rzv_r, False, None))
            if self.vis[t]:
                meas[t].append((self.p[t], Hp, Rp, t >= ll_from, None))
                meas[t].append((self.phi[t], Hr, Rr, t >= ll_from, None))
        return meas

    def anchors(self, X1, P1):
        pr = self.prm
        st_t = np.radians(pr["sigma_tilt_deg"])
        s_env2 = np.radians(SAT_DEG) ** 2 + st_t ** 2
        E = horiz_basis(self.up)
        d_g = pr["grasp_threshold_cm"]
        free = self.rest_raw & (self.dmin.min(0) >= d_g) & self.vis
        out, diag = [], {"rest_runs": 0, "candidates": 0, "tilt_accepted": 0, "height_accepted": 0, "tilt_no_pose_in_gate": 0,
                         "tilt_no_pose_in_envelope": 0}
        for a, b in runs(self.rest_raw):
            diag["rest_runs"] += 1
            sub = [(x, y) for (x, y) in runs(free[a:b + 1]) if y - x + 1 >= LMIN]
            if not sub:
                continue
            x, y = max(sub, key=lambda r: r[1] - r[0])
            tm = a + (x + y) // 2
            diag["candidates"] += 1
            S = E @ P1[tm][BT, BT] @ E.T + st_t ** 2 * np.eye(2)
            bhat = E @ X1[tm, BT]
            zs, ms = [], []
            for k in range(len(self.sp_h)):
                dk = self.rot[tm].apply(self.sp_n[k])
                zk = E @ (-min_rotvec(dk, -self.up))
                nu = zk - bhat
                zs.append(zk); ms.append(nu @ np.linalg.solve(S, nu))
            zs, ms = np.array(zs), np.array(ms)
            inn = ms < CHI2_2_99
            if self.fix_anchor:                                         # FIX-C
                env = np.einsum("ki,ki->k", zs, zs) / s_env2 < CHI2_2_99
                if (inn & ~env).any():
                    self.anchor_rejected.append({"t": int(tm), "n_gate": int(inn.sum()), "n_gate_and_env": int((inn & env).sum()),
                                                 "min_|z|_in_gate_deg": float(np.degrees(np.sqrt((zs[inn] ** 2).sum(1)).min())) if inn.any() else None})
                if inn.any() and not (inn & env).any():
                    diag["tilt_no_pose_in_envelope"] += 1
                    continue
                inn = inn & env
            if not inn.any():
                diag["tilt_no_pose_in_gate"] += 1
                continue
            w = self.sp_prob[inn] * np.exp(-0.5 * ms[inn])
            if w.sum() <= 0:
                w = np.exp(-0.5 * ms[inn])
            w = w / w.sum()
            z = (w[:, None] * zs[inn]).sum(0)
            dz = zs[inn] - z
            Rt = st_t ** 2 * np.eye(2) + np.einsum("k,ki,kj->ij", w, dz, dz)
            H = np.zeros((2, NX)); H[:, BT] = E
            out.append((tm, (z, H, Rt, False, None)))
            diag["tilt_accepted"] += 1
            if self.floor_only:
                hk = 100.0 * (self.sp_h[inn] - self.e) + pr["height_mu_cm"]
                zh = float((w * hk).sum())
                Rh = pr["sigma_height_cm"] ** 2 + float((w * (hk - zh) ** 2).sum())
                Hh = np.zeros((1, NX)); Hh[0, P_] = self.up
                nu = zh - self.up @ X1[tm, P_]
                Sh = self.up @ P1[tm][P_, P_] @ self.up + Rh
                if nu ** 2 / Sh < CHI2_1_99:
                    out.append((tm, (np.array([zh]), Hh, np.array([[Rh]]), False, None)))
                    diag["height_accepted"] += 1
        return out, diag
