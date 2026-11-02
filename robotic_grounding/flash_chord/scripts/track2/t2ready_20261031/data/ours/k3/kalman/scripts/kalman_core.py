"""Linear Kalman filter + Rauch-Tung-Striebel smoother for the Tier-2 object-pose bias model (NOISY-ONLY).

State (24): p(0:3) cm, v(3:6) cm/s, phi(6:9) rad, w(9:12) rad/s, b_p(12:15) cm, b_th(15:18) rad, d_t(18:21) cm, d_th(21:24) rad
  p, v      true object position (OpenCV world) and velocity, constant-velocity (white acceleration q_a cm^2/s^3)
  phi, w    true orientation in the cumulative world-increment chart and its rate, constant-velocity (q_al rad^2/s^3)
  b_p,b_th  measurement bias (random walk, steps q_bp cm/frame/axis, q_br rad/frame/axis); R_meas = exp([b_th]) R_true
  d_t,d_th  hand-chain error since the current carry onset (random walk; reset at each onset)  [K3 only]
Measurements are lists of (z, H, Rcov, use_for_loglik, gate) per frame; gate=None or a chi2 threshold on the
innovation Mahalanobis distance (forward pass; a gated-out measurement is skipped and counted).
"""
from __future__ import annotations

import numpy as np

NX = 24
P_, V_, PH, W_, BP, BT, DT_, DTH = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15), slice(15, 18),
                                    slice(18, 21), slice(21, 24))


def cv_Q(q, dt):
    return q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])


def build_FQ(dt, q_a, q_al, q_bp2, q_br2, q_d_t2, q_d_th2):
    F = np.eye(NX)
    F[P_, V_] = dt * np.eye(3)
    F[PH, W_] = dt * np.eye(3)
    Q = np.zeros((NX, NX))
    for (pos, vel, q) in ((P_, V_, q_a), (PH, W_, q_al)):
        c = cv_Q(q, dt)
        Q[pos, pos] = c[0, 0] * np.eye(3); Q[pos, vel] = c[0, 1] * np.eye(3)
        Q[vel, pos] = c[1, 0] * np.eye(3); Q[vel, vel] = c[1, 1] * np.eye(3)
    Q[BP, BP] = q_bp2 * np.eye(3)
    Q[BT, BT] = q_br2 * np.eye(3)
    Q[DT_, DT_] = q_d_t2 * np.eye(3)
    Q[DTH, DTH] = q_d_th2 * np.eye(3)
    return F, Q


def kf_rts(T, F, Q, x0, P0, meas, resets=None, active=None):
    """meas[t]: list of (z, H, Rcov, loglik_flag). resets: {t: (idx_slice, cov)} -> state block re-initialised at t
    (mean 0, covariance cov, no correlation with the rest). active: index array of state components to propagate
    (others are frozen and excluded; default all). Returns smoothed x (T,NX), P (T,NX,NX), loglik, n_gated."""
    idx = np.arange(NX) if active is None else np.asarray(active)
    n = len(idx)
    F, Q = F[np.ix_(idx, idx)], Q[np.ix_(idx, idx)]
    x0, P0 = x0[idx], P0[np.ix_(idx, idx)]
    pos = {g: i for i, g in enumerate(idx)}
    xp = np.zeros((T, n)); Pp = np.zeros((T, n, n)); xf = np.zeros((T, n)); Pf = np.zeros((T, n, n))
    Ft = {}
    ll = 0.0
    n_gated = 0
    x, P = x0.copy(), P0.copy()
    for t in range(T):
        Fk = F
        if t > 0:
            if resets and t in resets:
                sl, cov = resets[t]
                loc = np.array([pos[i] for i in range(NX)[sl]])
                Fk = F.copy(); Fk[loc, :] = 0.0
                Qk = Q.copy(); Qk[loc, :] = 0.0; Qk[:, loc] = 0.0; Qk[np.ix_(loc, loc)] = cov
            else:
                Qk = Q
            x = Fk @ x
            P = Fk @ P @ Fk.T + Qk
        Ft[t] = Fk
        xp[t], Pp[t] = x, P
        for (z, Hm, Rc, use, gate) in meas[t]:
            Hs = Hm[:, idx]
            nu = z - Hs @ x
            S = Hs @ P @ Hs.T + Rc
            S = 0.5 * (S + S.T)
            if gate is not None and nu @ np.linalg.solve(S, nu) > gate:
                n_gated += 1
                continue
            K = np.linalg.solve(S, Hs @ P).T
            x = x + K @ nu
            IKH = np.eye(n) - K @ Hs
            P = IKH @ P @ IKH.T + K @ Rc @ K.T
            if use:
                sign, logdet = np.linalg.slogdet(S)
                ll += -0.5 * (nu @ np.linalg.solve(S, nu) + logdet + len(nu) * np.log(2 * np.pi))
        xf[t], Pf[t] = x, P
    xs = xf.copy(); Ps = Pf.copy()
    for t in range(T - 2, -1, -1):
        Fk = Ft[t + 1]
        A = Pf[t] @ Fk.T
        try:
            C = np.linalg.solve(Pp[t + 1].T, A.T).T
        except np.linalg.LinAlgError:
            C = A @ np.linalg.pinv(Pp[t + 1])
        xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + C @ (Ps[t + 1] - Pp[t + 1]) @ C.T
    X = np.zeros((T, NX)); X[:, idx] = xs
    PP = np.zeros((T, NX, NX)); PP[:, idx[:, None], idx[None, :]] = Ps
    return X, PP, ll, n_gated


def H_rows(cols, k=3):
    """Helper: H with identity blocks at the given state slices (sum)."""
    Hm = np.zeros((k, NX))
    for sl, M in cols:
        Hm[:, sl] += M
    return Hm


def skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
