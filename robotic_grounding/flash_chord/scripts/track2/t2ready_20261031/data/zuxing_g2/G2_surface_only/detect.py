"""Contact / state detection from Tier-2 data only (noisy object track, Tier-2 SOMA joints, floor, support stage).
All thresholds are chi-square 99 % quantiles on uncertainty-inflated statistics, with sigmas from the noise model or the
current estimate's posterior (Laplace approximation); the few distances (d0_free, settle) are generic constants."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import model as M
from common import kc, CONST, CHI2_1, CHI2_3

H = CONST['win_h']


def window(T):
    t = np.arange(T)
    lo = np.maximum(t - H, 0); hi = np.minimum(t + H, T - 1)
    return lo, hi, (hi - lo).astype(float)


def sig_abs(E, lever, sbp, sbr, t):
    """per-axis std of an object point at distance `lever` cm from the origin + human drift std at the hands."""
    P, L = E.P, CONST['lever_human']
    hum = np.sqrt(P['h0_p'] ** 2 + t * P['q_ht'] ** 2 + (L ** 2) * (P['h0_r'] ** 2 + t * P['q_hr'] ** 2))
    return np.sqrt(sbp ** 2 + (lever * sbr) ** 2 + hum ** 2)


def far_from_object(E, x, sbp, sbr, groups):
    """True where every joint group (lists of SOMA joints, measured positions) is farther from the object than
    d0 + 3 sqrt(3) sigma, sigma = sig_abs at the closest object point. Object = dense surface samples, or the kit's 500
    vertex samples when opts handfree='pts' (B_hands) and the groups are the two hands."""
    T = E.T; t = np.arange(T); p, _ = E.pose(x)
    free = np.ones(T, bool); dmin = np.full((len(groups), T), np.inf)
    for gi, idx in enumerate(groups):
        if E.opts['handfree'] == 'pts' and groups is M.HAND_IDX:
            _, W = E.heights(x, E.PTS)
            Hj = E.JM[:, idx]
            d = np.linalg.norm(Hj[:, :, None] - W[:, None], axis=3)
            flat = d.reshape(T, -1).argmin(1); k = flat % W.shape[1]
            dm = d.reshape(T, -1).min(1); lever = np.linalg.norm(W[t, k] - p, axis=1)
        else:
            Ho = E.to_obj(x, E.JM[:, idx])
            dd, ii = E.tree.query(Ho.reshape(-1, 3)); dd = dd.reshape(T, -1); ii = ii.reshape(T, -1)
            j = dd.argmin(1); dm = dd[t, j]; lever = np.linalg.norm(E.S[ii[t, j]], axis=1)
        thr = CONST['d0_free'] + 3.0 * np.sqrt(3) * sig_abs(E, lever, sbp, sbr, t)
        free &= dm > thr
        dmin[gi] = dm
    return free, dmin


def handfree(E, x, sbp, sbr):
    """both hands (all 25 measured joints of each) far from the object (B_hands rule; dense surface by default)."""
    return far_from_object(E, x, sbp, sbr, M.HAND_IDX)


def bodyfree(E, x, sbp, sbr):
    """every one of the 77 measured SOMA joints far from the object (no body part, feet included, touches it)."""
    return far_from_object(E, x, sbp, sbr, [list(range(E.JM.shape[1]))])


def static_mask(E, free, bfree):
    """rest (ZUPT rule) or untouched and settled: no contact in the last `settle` frames (an untouched, supported object
    cannot move; a just-released one may still settle). Untouched = body-free (opts settle='body': no SOMA joint near,
    so an object pushed by a foot is not called static) or hand-free (settle='hands', B_hands). Gaps <= 2H bridged when
    the gap is untouched. Runs shorter than 10 frames dropped."""
    T = E.T
    unt = bfree if E.opts['settle'] == 'body' else free
    last = -10 ** 9; settled = np.zeros(T, bool)
    for t in range(T):
        if not unt[t]:
            last = t
        settled[t] = unt[t] and (t - last > CONST['settle'])
    s = E.rest | settled
    for (a0, b0), (a1, b1) in zip(M.runs(s)[:-1], M.runs(s)[1:]):
        if a1 - b0 - 1 <= 2 * H and unt[b0 + 1:a1].all():
            s[b0 + 1:a1] = True
    return M.keep_runs(s, 10)


def grasp_mask(E, x, sbp, sbr, static):
    """per hand, uniform surface model: (near) the median over the 14 grip points of the squared distance to the mesh
    surface passes an inflated chi-square(1) test; (co-moving) the median over the grip points of the squared windowed
    displacement relative to the object, along the local surface normal, passes a chi-square(1) test with the noise
    model's increment variance. Gaps <= 5 frames filled, runs < 10 frames dropped, runs with < 10 moving frames dropped."""
    T = E.T; t = np.arange(T); P = E.P
    lo, hi, span = window(T)
    out = np.zeros((2, T), bool); info = {}
    for s in (0, 1):
        Xo = E.grip_obj(x, s)
        d, idx = E.tree.query(Xo.reshape(-1, 3)); d = d.reshape(T, -1); idx = idx.reshape(T, -1)
        lev = np.linalg.norm(Xo.mean(1), axis=1)
        sa = sig_abs(E, lev, sbp, sbr, t)
        near = np.median(d ** 2, axis=1) < CHI2_1 * (sa ** 2 + P['s_c'] ** 2)
        dn = np.einsum('tkj,tkj->tk', E.N[idx], Xo[hi] - Xo[lo])
        s_inc2 = P['q_bp'] ** 2 + (lev * P['q_br']) ** 2 + P['s_slip'] ** 2 + P['q_ht'] ** 2 + (CONST['lever_human'] * P['q_hr']) ** 2
        com = np.median(dn ** 2, axis=1) < CHI2_1 * span * s_inc2
        g = near & com
        g = M.fill_gaps(g, 5)
        g = M.keep_runs(g, 10)
        for a, b in M.runs(g):              # co-motion is informative only while the object moves
            if (~static[a:b + 1]).sum() < 10:
                g[a:b + 1] = False
        out[s] = g
        info[s] = dict(near=int(near.sum()), com=int(com.sum()), med_d=np.median(d, axis=1))
    return out, info


def stance_masks(E):
    T = E.T
    lo, hi, span = window(T)
    h = E.JM @ E.UP + 100 * E.E0
    out = {}
    for f, (ank, toe) in M.FEET.items():
        ok = np.ones(T, bool)
        for j in (ank, toe):
            _, _, fl = kc.psd_floor(np.diff(E.JM[:, j], axis=0))
            sig = np.sqrt(fl * kc.FPS / 2.0)
            ok &= np.linalg.norm(E.JM[hi, j] - E.JM[lo, j], axis=1) < np.sqrt(CHI2_3 * span) * sig
        ok &= h[:, toe] < np.percentile(h[:, toe], 50) + 10.0
        out[f] = M.keep_runs(ok, 10)
    return out


def support_contacts(E, x, sbp, sbr, sup):
    """Which supports a hand-free static object touches, per run of `sup` frames (opts support='auto'):
      * box contact: a support box is touched in the run when the run's median squared distance of the object points
        to the box passes an inflated chi-square(1) test (object-pose std at the closest point + s_sup_b);
      * resting on a box top: a touched box whose top face is at the height of the object's lowest point (same
        inflated chi-square(1) test on the height difference, lowest convex-hull point, sigma s_sup_b);
      * gravity: if the object rests on no box top it rests on the floor (floor contact on), possibly leaning on the
        touched boxes; if it rests on a box top the floor contact is off.
    Contact terms: floor on the floor frames, the nearest touched box on the box frames.
    support='legacy' (B_hands): floor and all boxes on every sup frame."""
    T, nb = E.T, len(E.BOXES); t = np.arange(T)
    sup_f = np.zeros(T, bool); sup_b = np.zeros((T, nb), bool); log = []
    if E.opts['support'] == 'legacy':
        sup_f[:] = sup; sup_b[:] = sup[:, None]
        return sup_f, sup_b, log
    if not sup.any():
        return sup_f, sup_b, log
    h, _ = E.heights(x, E.HULL); k = h.argmin(1); hmin = h[t, k]; lev_f = np.linalg.norm(E.HULL[k], axis=1)
    if nb:
        _, W = E.heights(x, E.BPTS)
        Wr = (W / 100.0) @ E.BCR.T + E.bCR
        dist = np.zeros((T, nb)); lev_b = np.zeros((T, nb)); htop = np.zeros(nb)
        for b, (cen, half) in enumerate(E.BOXES):
            dd = np.linalg.norm(np.maximum(np.abs(Wr - cen) - half, 0), axis=2) * 100
            kb = dd.argmin(1); dist[:, b] = dd[t, kb]; lev_b[:, b] = np.linalg.norm(E.BPTS[kb], axis=1)
            top_cv = ((np.r_[cen[:2], cen[2] + half[2]] - E.bCR) @ E.BCR) * 100.0     # robot world -> OpenCV (cm)
            htop[b] = top_cv @ E.UP + 100 * E.E0                                     # box top height above the floor
    for a, b_ in M.runs(sup):
        sl = slice(a, b_ + 1)
        s2 = np.median(sbp[sl] ** 2 + (lev_f[sl] * sbr[sl]) ** 2)
        rec = dict(run=(int(a), int(b_)), floor_med_cm=round(float(np.median(hmin[sl])), 2), boxes=[])
        on_top = False
        for bi in range(nb):
            s2b = np.median(sbp[sl] ** 2 + (lev_b[sl, bi] * sbr[sl]) ** 2)
            touch = np.median(dist[sl, bi] ** 2) < CHI2_1 * (s2b + E.P['s_sup_b'] ** 2)
            top = touch and np.median((hmin[sl] - htop[bi]) ** 2) < CHI2_1 * (s2 + E.P['s_sup_b'] ** 2)
            sup_b[sl, bi] = touch; on_top |= top
            rec['boxes'].append(dict(touch=bool(touch), top=bool(top), d_cm=round(float(np.median(dist[sl, bi])), 2),
                                     top_h_cm=round(float(htop[bi]), 1)))
        sup_f[sl] = not on_top
        rec['floor'] = bool(not on_top)
        log.append(rec)
    return sup_f, sup_b, log


def posterior_std(prob, x, every=10):
    """Laplace approximation: per-axis std of b_p (cm) and b_r (rad) at every `every`-th frame, interpolated."""
    T = prob.E.T
    Jm, _ = prob.jacobian(x)
    A = (Jm.T @ Jm).tocsc() + sp.identity(Jm.shape[1], format='csc') * 1e-9
    lu = spla.splu(A)
    fr = np.r_[np.arange(0, T, every), T - 1]
    sbp = []; sbr = []
    for t in fr:
        Em = np.zeros((Jm.shape[1], 6))
        for i in range(6):
            Em[t * 12 + i, i] = 1.0
        X = lu.solve(Em)
        v = np.array([X[t * 12 + i, i] for i in range(6)])
        sbp.append(np.sqrt(v[:3].mean())); sbr.append(np.sqrt(v[3:].mean()))
    tt = np.arange(T)
    return np.interp(tt, fr, sbp), np.interp(tt, fr, sbr)
