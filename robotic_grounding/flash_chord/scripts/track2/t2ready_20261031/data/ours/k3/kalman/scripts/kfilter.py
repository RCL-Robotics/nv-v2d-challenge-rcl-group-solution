"""Variants K0-K3 (RTS smoothers over the whole clip) for the Tier-2 object pose. NOISY-ONLY inputs.

Parameters come from KOUT/design/frozen_params.json (written by d3 from the d1/d2/d2b noisy-only derivations + ML q_a).
Output per episode: (T_visible, 7) [x y z qx qy qz qw], metres, source OpenCV world (same layout as the ZUPT v1 study).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from kalman_core import BP, BT, DT_, DTH, NX, P_, PH, V_, W_, build_FQ, kf_rts, skew  # noqa: E402
from kcommon import DT, J, KOUT, LMIN, load_g1, metas, object_track, runs, support_is_empty  # noqa: E402

DES = KOUT / "design"
CHI2_1_99, CHI2_2_99, CHI2_6_99 = 6.634897, 9.210340, 16.811894
I3 = np.eye(3)


def Hm(blocks, k=3):
    H = np.zeros((k, NX))
    for sl, M in blocks:
        H[:, sl] += M
    return H


def horiz_basis(up):
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return np.stack([e1, e2])


def min_rotvec(d, u):
    """Rotation vector of the minimal rotation taking unit d onto unit u."""
    c = np.cross(d, u)
    s = np.linalg.norm(c)
    ang = np.arctan2(s, float(np.clip(d @ u, -1, 1)))
    return np.zeros(3) if s < 1e-12 else c / s * ang


class Episode:
    def __init__(self, m, prm):
        self.m, self.prm = m, prm
        self.ep, self.obj = m["episode_index"], m["object"]
        self.p, self.rot, self.vis, _ = object_track(self.ep)
        self.T = len(self.p)
        inc = (self.rot[1:] * self.rot[:-1].inv()).as_rotvec()                  # world-frame increments (rad)
        self.phi = np.vstack([np.zeros(3), np.cumsum(inc, 0)])
        masks = np.load(DES / "d1_masks.npz")
        self.rest = masks[f"obj_{self.ep}"] & self.vis
        self.rest_raw = masks[f"obj_{self.ep}"]
        fr = {r["episode"]: r for r in json.loads((DES / "d1_frames.json").read_text())}[self.ep]
        self.fr = fr
        self.up, self.e = np.array(fr["up_cv"]), fr["e_cv"]
        self.floor_only = bool(support_is_empty(m))
        self.dmin = np.load(DES / "d2_hand_dist.npz")[f"d_{self.ep}"]
        self.side = np.load(DES / "d2b_rigid_masks.npz")[f"side_{self.ep}"]
        sp = np.load(DES / "d2_stable_poses.npz")
        self.sp_n, self.sp_h, self.sp_prob = sp[f"{self.obj}__n"], sp[f"{self.obj}__h_m"], sp[f"{self.obj}__prob"]
        self.qa = prm["q_a_cm2_s3"][str(self.ep)]
        self.qal = prm["q_al_rad2_s3"][str(self.ep)]

    # ------------------------------------------------------------------ measurement lists
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
                meas[t].append((np.zeros(3), Hw, Rzv_r, False, None))
            if self.vis[t]:
                meas[t].append((self.p[t], Hp, Rp, t >= ll_from, None))
                meas[t].append((self.phi[t], Hr, Rr, t >= ll_from, None))
        return meas

    def FQ(self, with_hand=False):
        pr = self.prm
        return build_FQ(DT, self.qa, self.qal, pr["q_bp_cm"] ** 2, np.radians(pr["q_br_deg"]) ** 2,
                        pr["q_d_t_cm"] ** 2 if with_hand else 1.0, np.radians(pr["q_d_th_deg"]) ** 2 if with_hand else 1.0)

    def prior(self):
        pr = self.prm
        x0 = np.zeros(NX); x0[P_] = self.p[0]
        P0 = np.zeros((NX, NX))
        P0[P_, P_] = 1e4 * I3; P0[V_, V_] = 1e4 * I3; P0[PH, PH] = 1.0 * I3; P0[W_, W_] = 10.0 * I3
        P0[BP, BP] = pr["P_b0_pos_cm"] ** 2 * I3
        P0[BT, BT] = np.radians(pr["P_b0_rot_deg"]) ** 2 * I3
        P0[DT_, DT_] = pr["P_b0_pos_cm"] ** 2 * I3
        P0[DTH, DTH] = np.radians(pr["P_b0_rot_deg"]) ** 2 * I3
        return x0, P0

    def run(self, meas, with_hand=False, resets=None):
        F, Q = self.FQ(with_hand)
        x0, P0 = self.prior()
        active = np.arange(NX) if with_hand else np.arange(18)
        return kf_rts(self.T, F, Q, x0, P0, meas, resets=resets, active=active)

    def pose_out(self, X):
        rot = R.from_rotvec(-X[:, BT]) * self.rot
        pos = X[:, P_] / 100.0
        out = np.concatenate([pos, rot.as_quat()], 1)
        return out[self.vis]

    # ------------------------------------------------------------------ K2 anchors (from the K1 smoothed state)
    def anchors(self, X1, P1):
        pr = self.prm
        st_t = np.radians(pr["sigma_tilt_deg"])
        E = horiz_basis(self.up)
        d_g = pr["grasp_threshold_cm"]
        free = self.rest_raw & (self.dmin.min(0) >= d_g) & self.vis
        out, diag = [], {"rest_runs": 0, "candidates": 0, "tilt_accepted": 0, "height_accepted": 0, "tilt_no_pose_in_gate": 0}
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

    # ------------------------------------------------------------------ K3 hand measurements (from the K2 smoothed state)
    def human_hands(self):
        """Hand (wrist joint) positions (cm) / rotations and the Hips position in the OpenCV world, from the Tier-2 SOMA joints."""
        g = load_g1(self.m["sequence_id"])
        Rs2r, ts2r = np.array(self.fr["Rs2r"]), np.array(self.fr["ts2r"])
        B, b = np.array(self.fr["Bcv2r"]), np.array(self.fr["bcv2r"])
        pcv = (((g["soma_joints"] @ Rs2r.T + ts2r) - b) @ B) * 100.0
        Mrot = R.from_matrix(B.T @ Rs2r)
        hp, hr = {}, {}
        for si in (0, 1):
            hj = J["LeftHand"] if si == 0 else J["RightHand"]
            hp[si] = pcv[:, hj]
            hr[si] = Mrot * R.from_quat(g["soma_joints_wxyz"][:, hj][:, [1, 2, 3, 0]])
        return hp, hr, pcv[:, J["Hips"]]

    def hand_meas(self, X2, P2, hands=None):
        pr = self.prm
        hp, hr, hips = self.human_hands() if hands is None else hands
        Rsw = pr["slip_white_rot_deg"]; Psw = pr["slip_white_pos_cm"]
        Rc = np.diag([Psw ** 2] * 3 + [np.radians(Rsw) ** 2] * 3)
        meas, resets, diag = {}, {}, {"carry_runs": 0, "hand_frames": 0}
        for si in (0, 1):
            Rh_all, ph_all = hr[si], hp[si]
            for a, bb in runs(self.side == si):
                diag["carry_runs"] += 1
                Rk2 = R.from_rotvec(-X2[a, BT]) * self.rot[a]
                R_ho = Rh_all[a].inv() * Rk2
                r_ho = Rh_all[a].inv().apply(X2[a, P_] - ph_all[a])
                resets[a] = (slice(18, 24), P2[a][12:18, 12:18].copy())
                for t in range(a, bb + 1):
                    if not self.vis[t]:
                        continue
                    Rhp = Rh_all[t] * R_ho
                    php = ph_all[t] + Rh_all[t].apply(r_ho)
                    z = np.concatenate([php - self.p[t], (Rhp * self.rot[t].inv()).as_rotvec()])
                    r = self.p[t] - hips[t]
                    H = np.zeros((6, NX))
                    H[0:3, BP] = -I3; H[0:3, DT_] = I3; H[0:3, DTH] = -skew(r)
                    H[3:6, BT] = -I3; H[3:6, DTH] = I3
                    meas.setdefault(t, []).append((z, H, Rc, False, CHI2_6_99))
                    diag["hand_frames"] += 1
        return meas, resets, diag

    # ------------------------------------------------------------------ variants
    def variant(self, name, hands=None):
        diag = {}
        if name == "K0":
            X, P, ll, ng = self.run(self.base_meas(zv=False))
            return self.pose_out(X), X, diag
        meas1 = self.base_meas(zv=True)
        X1, P1, ll, _ = self.run(meas1)
        if name == "K1":
            return self.pose_out(X1), X1, diag
        anc, diag = self.anchors(X1, P1)
        meas2 = [list(m) for m in meas1]
        for t, mm in anc:
            meas2[t].append(mm)
        X2, P2, _, _ = self.run(meas2)
        if name == "K2":
            return self.pose_out(X2), X2, diag
        hm, resets, dh = self.hand_meas(X2, P2, hands)
        meas3 = [list(m) for m in meas2]
        for t, lst in hm.items():
            meas3[t].extend(lst)
        X3, P3, _, ng = self.run(meas3, with_hand=True, resets=resets)
        diag.update(dh); diag["hand_frames_gated"] = int(ng)
        return self.pose_out(X3), X3, diag


def load_params():
    return json.loads((DES / "frozen_params.json").read_text())


def episodes():
    return metas()
