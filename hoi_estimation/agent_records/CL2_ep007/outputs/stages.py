"""Trajectory estimation stages (called from solve.py)."""
import numpy as np
import torch
import cv2
import h5py
from scipy.ndimage import median_filter, gaussian_filter1d

from common import rot_angle_deg, project
from fitcore import so3_exp, color_loss
from hyp import SOLE_N, HEEL_N, soft_iou_loss, init_translation, fit_batch, hypothesis_set, rot_a_to_b
from constrained import PlaneModel, fit_plane_model, load_frames

W_COL = 0.1        # weight of the colour term while fitting
S_COL = 0.3        # weight of the colour term when scoring hypotheses
KEY_STEP = 10      # keyframe spacing for the hypothesis search


def unit(x):
    return x / np.linalg.norm(x)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------------ hypothesis search with a plane model
def init_ab(model, psi, box, K, cm):
    R, _ = model.pose(psi, 0, 0)
    x0, y0, x1, y1 = box
    ray = np.linalg.inv(K) @ [(x0 + x1) / 2, (y0 + y1) / 2, 1]
    h = (R[0] @ cm) @ model.n + model.s0
    s = (h - model.d) / (model.n @ ray)
    t = s * ray - R[0] @ cm
    return t @ model.e1, t @ model.e2


def plane_search(fit, model, frames, data, K, cm, n_yaw=36, top=4):
    """For each frame: yaw grid, silhouette-only coarse fit, colour refinement of the best `top`."""
    out = {}
    psis = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)
    for fr in frames:
        ab = np.array([init_ab(model, p, data["box"][fr], K, cm) for p in psis])
        p, a, b, s, c, _ = fit_plane_model(fit, model, [fr] * n_yaw, data, psis, ab[:, 0], ab[:, 1], iters=40,
                                           w_col=0.0, hw=(96, 96))
        o = np.argsort(s)
        sel = []
        for k in o:
            if all(abs(wrap(p[k] - p[j])) > np.radians(20) for j in sel):
                sel.append(k)
            if len(sel) == top:
                break
        p, a, b, s, c, _ = fit_plane_model(fit, model, [fr] * len(sel), data, p[sel], a[sel], b[sel], iters=100,
                                           w_col=W_COL)
        out[fr] = dict(psi=p, a=a, b=b, sil=s, col=c, score=s + S_COL * c)
    return out


def viterbi_plane(cands, frames, w_yaw=1.0, w_pos=1.0, yaw_sigma_deg=10.0, pos_sigma=0.03):
    """One candidate per keyframe; transition penalises yaw and position changes."""
    U = [20 * cands[f]["score"] for f in frames]
    cost = U[0].copy()
    bps = []
    for k in range(1, len(frames)):
        A, B = cands[frames[k - 1]], cands[frames[k]]
        dy = np.degrees(np.abs(wrap(A["psi"][:, None] - B["psi"][None])))
        dp = np.hypot(A["a"][:, None] - B["a"][None], A["b"][:, None] - B["b"][None])
        T = w_yaw * (dy / yaw_sigma_deg) ** 2 + w_pos * (dp / pos_sigma) ** 2
        tot = cost[:, None] + T
        bp = np.argmin(tot, 0)
        bps.append(bp)
        cost = tot[bp, np.arange(tot.shape[1])] + U[k]
    path = [int(np.argmin(cost))]
    for bp in reversed(bps):
        path.append(int(bp[path[-1]]))
    return path[::-1]


def interp_params(kf, vals, frames):
    """Linear interpolation of (psi,a,b) given at keyframes kf; psi unwrapped."""
    kf = np.asarray(kf)
    psi = np.unwrap(vals[:, 0])
    return np.stack([np.interp(frames, kf, psi), np.interp(frames, kf, vals[:, 1]),
                     np.interp(frames, kf, vals[:, 2])], 1)


def dense_plane(fit, model, frames, data, init, batch=48, iters=60):
    res = np.zeros((len(frames), 5))
    for s in range(0, len(frames), batch):
        fr = list(frames[s:s + batch])
        x = init[s:s + batch]
        p, a, b, si, c, _ = fit_plane_model(fit, model, fr, data, x[:, 0], x[:, 1], x[:, 2], iters=iters,
                                            w_col=W_COL)
        res[s:s + batch] = np.stack([p, a, b, si, c], 1)
    return res


def robust_smooth(x, sigma, med=5):
    """Hampel-like: median filter to kill isolated outliers, then Gaussian smoothing."""
    m = median_filter(x, size=med, mode="nearest")
    r = x - m
    mad = np.median(np.abs(r)) * 1.4826 + 1e-9
    y = np.where(np.abs(r) > 3 * mad, m, x)
    return gaussian_filter1d(y, sigma, mode="nearest") if sigma > 0 else y


# ------------------------------------------------------------------ free 6-DoF refinement with contact penalty
def fit_free_contact(fit, R0, t0, frames, data, n, d, verts_t, iters=100, w_col=0.3, w_contact=2000.0,
                     lr_r=0.01, lr_t=0.004):
    nt = torch.tensor(n, dtype=torch.float32, device=fit.dev)

    def extra(Rm, t):
        h = torch.einsum("bij,nj->bni", Rm, verts_t) + t[:, None]
        hgt = h @ nt + d  # height above the table
        mn = -0.002 * torch.logsumexp(-hgt / 0.002, dim=1)  # soft-min
        return w_contact * (mn ** 2).sum()
    return fit_batch(fit, R0, t0, frames, data, iters=iters, lr_r=lr_r, lr_t=lr_t, w_col=w_col, extra_loss=extra)


def contact_normal(fit, model_R, model_t, frames, data, facet_n, n_yaw_frames=None):
    """Free 6-DoF refinement (colour-weighted) from constrained solutions; median facet normal -> table normal."""
    R, t, s, c = fit_batch(fit, model_R, model_t, list(frames), data, iters=120, lr_r=0.006, lr_t=0.002, w_col=0.3)
    fn = unit(facet_n)
    ns = np.array([-(Ri @ fn) for Ri in R])
    good = (c < np.percentile(c, 75))
    m = unit(np.median(ns[good], 0))
    return m, ns, s, c


# ------------------------------------------------------------------ main stage runner
def plane_through(n, X):
    n = unit(n)
    return n, float(-(n @ X))


def run_all(fit, data, K, n_floor, d_floor, V, cm, F, rec, Twc, table, cached, log, info):
    n_floor = unit(n_floor)
    C_phys = np.asarray(table["center"])  # physical table-top centre (camera frame)

    # ---- A0. fit-space plane: the plane through s*C_phys (s scales along the viewing ray) that best explains
    # the object's image size with the given mesh (sigma = 1). Static first frames, iron resting on its sole.
    def scale_search():
        res = {}
        for s in (0.90, 0.93, 0.96, 0.99, 1.02):
            m = PlaneModel(V, SOLE_N, *plane_through(n_floor, s * C_phys))
            c = plane_search(fit, m, [0, 50], data, K, cm, top=2)
            res[s] = float(np.mean([c[f]["score"].min() for f in c]))
        return res
    log("A0: fit-space plane scale")
    sres = cached("A0_scale", scale_search)
    ss = np.array(sorted(sres))
    sc = np.array([sres[s] for s in ss])
    k = int(np.argmin(sc))
    if 0 < k < len(ss) - 1:  # parabolic refinement
        a2, a1, _ = np.polyfit(ss[k - 1:k + 2], sc[k - 1:k + 2], 2)
        s_fit = float(-a1 / (2 * a2))
    else:
        s_fit = float(ss[k])
    info["A0_scale_scores"] = {str(a): b for a, b in sres.items()}
    info["s_fit"] = s_fit
    X_fit = s_fit * C_phys
    log("s_fit = %.4f" % s_fit)

    # ---- A. keyframe search: sole model on all keyframes, heel model after the flat phase
    keys = np.array(list(range(0, F, KEY_STEP)) + ([F - 1] if (F - 1) % KEY_STEP else []))
    sole0 = PlaneModel(V, SOLE_N, *plane_through(n_floor, X_fit))
    heel0 = PlaneModel(V, HEEL_N, *plane_through(n_floor, X_fit))
    log("A: keyframe search, sole model (%d keyframes)" % len(keys))
    cs0 = cached("A_sole", lambda: plane_search(fit, sole0, list(keys), data, K, cm, top=3))
    bs = np.array([cs0[f]["score"].min() for f in keys])
    # flat phase ends where the sole model becomes bad for 4 consecutive keyframes
    med_s = np.median(bs)
    bad = bs > med_s + 0.1
    run = np.array([bad[i:i + 4].all() for i in range(len(bad))])
    first_bad = int(keys[np.nonzero(run)[0][0]]) if run.any() else F
    hk = [int(f) for f in keys if f >= first_bad - KEY_STEP]
    log("A: heel model on %d keyframes" % len(hk))
    ch0 = cached("A_heel", lambda: plane_search(fit, heel0, hk, data, K, cm, top=3)) if hk else {}
    bh = np.array([ch0[f]["score"].min() for f in hk]) if hk else np.zeros(0)
    flat_end = first_bad - 10
    heel_start = F
    if len(bh):
        med_h = np.median(bh[np.array(hk) > hk[-1] - 100])
        okh = bh < med_h + 0.05
        # standing phase: from the first keyframe after which the heel model is good until the end
        idx = len(okh)
        while idx > 0 and okh[idx - 1]:
            idx -= 1
        heel_start = min(F, int(hk[idx]) + 10) if idx < len(hk) else F
    info["A_best_sole"] = bs.tolist()
    info["A_best_heel"] = bh.tolist()
    info["flat_end"], info["heel_start"] = flat_end, heel_start
    log("phases: flat [0,%d) transition [%d,%d) standing [%d,%d)" % (flat_end, flat_end, heel_start, heel_start, F))

    # ---- B. Viterbi over keyframes within each constrained phase
    kf_s = [int(f) for f in keys if f < flat_end + KEY_STEP]
    kf_h = [int(f) for f in hk if f >= heel_start - KEY_STEP]
    ps = viterbi_plane(cs0, kf_s)
    sel_s = np.array([[cs0[f][q][i] for q in ("psi", "a", "b")] for f, i in zip(kf_s, ps)])
    if kf_h:
        ph = viterbi_plane(ch0, kf_h)
        sel_h = np.array([[ch0[f][q][i] for q in ("psi", "a", "b")] for f, i in zip(kf_h, ph)])

    # ---- C. table normal from free 6-DoF refinement of the keyframe solutions (pooled median)
    def model_poses(model, sel):
        return model.pose(sel[:, 0], sel[:, 1], sel[:, 2])
    log("C: contact normals")
    Rk, tk = model_poses(sole0, sel_s)
    n_s = cached("C_ns", lambda: contact_normal(fit, Rk, tk, kf_s, data, SOLE_N))[0]
    sole = PlaneModel(V, SOLE_N, *plane_through(n_s, X_fit))
    info["n_sole"] = n_s.tolist()
    info["n_sole_vs_floor_deg"] = float(np.degrees(np.arccos(n_s @ n_floor)))
    if kf_h:
        Rk2, tk2 = model_poses(heel0, sel_h)
        n_h = cached("C_nh", lambda: contact_normal(fit, Rk2, tk2, kf_h, data, HEEL_N))[0]
        heel = PlaneModel(V, HEEL_N, *plane_through(n_h, X_fit))
        info["n_heel"] = n_h.tolist()
        info["n_heel_vs_floor_deg"] = float(np.degrees(np.arccos(n_h @ n_floor)))
    log("normals vs floor normal: sole %.2f deg, heel %s deg" % (info["n_sole_vs_floor_deg"],
                                                                  info.get("n_heel_vs_floor_deg")))

    # ---- D. dense refinement in the constrained phases
    def dense(model, model0, sel, kf, frames, name):
        R, t = model_poses(model0, sel)
        p, a, b = model.from_pose(R, t)
        init = interp_params(kf, np.stack([p, a, b], 1), frames)
        return cached(name, lambda: dense_plane(fit, model, frames, data, init))
    fr_s = np.arange(0, flat_end)
    fr_h = np.arange(heel_start, F)
    log("D: dense flat (%d frames), standing (%d frames)" % (len(fr_s), len(fr_h)))
    ds = dense(sole, sole0, sel_s, kf_s, fr_s, "D_sole")
    dh = dense(heel, heel0, sel_h, kf_h, fr_h, "D_heel") if kf_h else None

    # ---- F1. static segments + smoothing of the constrained phases
    touch = np.asarray(data["touch"])
    info["first_touch"] = int(np.argmax(touch > 0)) if (touch > 0).any() else F
    info["last_touch"] = int(F - 1 - np.argmax(touch[::-1] > 0)) if (touch > 0).any() else -1

    def smooth_phase(D, frames, model, name):
        psi = np.unwrap(D[:, 0])
        P = np.stack([psi, D[:, 1], D[:, 2]], 1)
        Pm = median_filter(P, size=(9, 1), mode="nearest")
        out = P.copy()
        segs = []
        loc = {f: i for i, f in enumerate(frames)}
        for side in ("start", "end"):
            order = list(range(len(frames))) if side == "start" else list(range(len(frames) - 1, -1, -1))
            # untouched frames are static by physics; extend while the pose stays put (1 deg / 4 mm)
            n0 = 0
            for i in order:
                if touch[frames[i]] == 0:
                    n0 += 1
                else:
                    break
            if n0 < 5:
                continue
            core = order[:n0]
            med = np.median(Pm[core], 0)
            n1 = n0
            for i in order[n0:]:
                dv = np.abs(Pm[i] - med)
                if np.degrees(dv[0]) < 1.0 and dv[1] < 0.004 and dv[2] < 0.004:
                    n1 += 1
                else:
                    break
            seg = sorted(order[:n1])
            w = 1.0 / (D[seg, 3] + S_COL * D[seg, 4])
            med = np.array([np.sum(w * P[seg, j]) / np.sum(w) for j in range(3)])
            out[seg] = med
            segs.append((int(frames[seg[0]]), int(frames[seg[-1]]) + 1))
        moving = np.ones(len(frames), bool)
        for a, b in segs:
            moving[(frames >= a) & (frames < b)] = False
        if moving.any():
            idx = np.nonzero(moving)[0]
            for j in range(3):
                out[idx, j] = robust_smooth(P[idx, j], 1.5)
        info["static_" + name] = segs
        R, t = model.pose(out[:, 0], out[:, 1], out[:, 2])
        return R, t, segs
    Rs_, ts_, seg_s = smooth_phase(ds, fr_s, sole, "flat")
    Rh_, th_, seg_h = smooth_phase(dh, fr_h, heel, "standing") if dh is not None else (None, None, [])
    log("static segments: flat %s standing %s" % (seg_s, seg_h))

    # ---- E. transition
    R_all = np.zeros((F, 3, 3))
    t_all = np.zeros((F, 3))
    R_all[fr_s], t_all[fr_s] = Rs_, ts_
    if dh is not None:
        R_all[fr_h], t_all[fr_h] = Rh_, th_
    fr_t = np.arange(flat_end, heel_start)
    if len(fr_t):
        Re, te, se, ce = transition(fit, data, K, cm, V, sole.n, sole.d, fr_t, Rs_[-1], ts_[-1],
                                    Rh_[0] if Rh_ is not None else Rs_[-1], th_[0] if th_ is not None else ts_[-1],
                                    cached, log)
        # smooth (including a few frames of context on both sides)
        Rc = np.r_[Rs_[-3:], Re, Rh_[:3] if Rh_ is not None else Re[-1:].repeat(3, 0)]
        tc = np.r_[ts_[-3:], te, th_[:3] if th_ is not None else te[-1:].repeat(3, 0)]
        Rc = smooth_rotations(Rc, 1.5)
        tc = np.stack([robust_smooth(tc[:, j], 1.5) for j in range(3)], 1)
        R_all[fr_t], t_all[fr_t] = Rc[3:-3], tc[3:-3]
        info["transition_sil_med"] = float(np.median(se))
        info["transition_col_med"] = float(np.median(ce))
    T = np.tile(np.eye(4), (F, 1, 1))
    T[:, :3, :3] = R_all
    T[:, :3, 3] = t_all
    return dict(T_fit=T, sole=sole, heel=heel if kf_h else None, X_fit=X_fit, s_fit=s_fit, n_s=n_s,
                flat_end=flat_end, heel_start=heel_start, seg_s=seg_s, seg_h=seg_h, ds=ds, dh=dh)


# ------------------------------------------------------------------ rotation utilities
def R_to_q(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def q_to_R(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix()


def slerp_poses(kf, Rs, ts, frames):
    from scipy.spatial.transform import Rotation, Slerp
    sl = Slerp(kf, Rotation.from_matrix(Rs))
    fr = np.clip(frames, kf[0], kf[-1])
    R = sl(fr).as_matrix()
    t = np.stack([np.interp(fr, kf, ts[:, i]) for i in range(3)], 1)
    return R, t


def smooth_rotations(R, sigma):
    """Gaussian smoothing of a rotation sequence via sign-consistent quaternions."""
    q = R_to_q(R)
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    qs = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return q_to_R(qs)


# ------------------------------------------------------------------ E. stand-up transition (free 6-DoF + contact)
def transition(fit, data, K, cm, V, n_s, d_s, frames, R_start, t_start, R_end, t_end, cached, log):
    import torch
    verts_t = torch.tensor(V[::40], dtype=torch.float32, device=fit.dev)
    kf = list(range(int(frames[0]), int(frames[-1]) + 1, 5))
    up = n_s

    def search():
        Rs, _ = hypothesis_set(up, n_yaw=36, tilts=(0, 0.2, 0.4, 0.6, 0.8, 1.0))
        out = {}
        for f in kf:
            t0 = init_translation(fit, Rs, data["box"][f], K, V[::50], cm)
            R, t, s, c = fit_batch(fit, Rs, t0, [f] * len(Rs), data, iters=40, opt_rot=False, lr_t=0.01,
                                   hw=(96, 96))
            o = np.argsort(s)
            sel = []
            for k in o:
                if all(rot_angle_deg(R[k], R[j]) > 20 for j in sel):
                    sel.append(k)
                if len(sel) == 6:
                    break
            R2, t2, s2, c2 = fit_free_contact(fit, R[sel], t[sel], [f] * len(sel), data, n_s, d_s, verts_t,
                                              iters=100, w_col=W_COL)
            sc = s2 + S_COL * c2
            bad = ~(np.isfinite(R2).all((1, 2)) & np.isfinite(t2).all(1) & np.isfinite(sc))
            sc[bad] = np.inf
            R2[bad] = np.eye(3)
            t2[bad] = 0
            out[f] = (R2, t2, sc)
        return out
    log("E: transition keyframe search (%d keyframes)" % len(kf))
    cand = cached("E_cand", search)
    # Viterbi with fixed endpoints (last flat pose, first standing pose)
    Rl = [R_start[None]] + [cand[f][0] for f in kf] + [R_end[None]]
    Ul = [np.zeros(1)] + [np.minimum(20 * cand[f][2], 1e6) for f in kf] + [np.zeros(1)]
    cost = Ul[0]
    bps = []
    for i in range(1, len(Rl)):
        T = np.array([[(rot_angle_deg(a, b) / 15.0) ** 2 for b in Rl[i]] for a in Rl[i - 1]])
        tot = cost[:, None] + T
        bp = np.argmin(tot, 0)
        bps.append(bp)
        cost = tot[bp, np.arange(tot.shape[1])] + Ul[i]
    path = [0]
    for bp in reversed(bps):
        path.append(int(bp[path[-1]]))
    path = path[::-1][1:-1]
    Rk = np.array([cand[f][0][p] for f, p in zip(kf, path)])
    tk = np.array([cand[f][1][p] for f, p in zip(kf, path)])
    kk = np.r_[frames[0] - 1, kf, frames[-1] + 1]
    Ri, ti = slerp_poses(kk, np.r_[R_start[None], Rk, R_end[None]], np.r_[t_start[None], tk, t_end[None]], frames)
    log("E: dense transition refinement (%d frames)" % len(frames))

    def dense():
        R, t, s, c = fit_free_contact(fit, Ri, ti, list(frames), data, n_s, d_s, verts_t, iters=100, w_col=W_COL)
        return R, t, s, c
    return cached("E_dense", dense)


# ------------------------------------------------------------------ G. depth cues (scale k about the camera centre)
GRIP_MESH = np.array([0.065, 0.0, -0.01])  # top of the handle (mesh frame): where the right hand holds the iron


def depth_cues(T_fit, sole, X_fit, s_fit, data, K, Twc, rec, M_crop, crop, table, F, log):
    cues = {}
    n, d_fit = sole.n, sole.d
    # (1) table geometry (casters on the released floor plane + table-top rim): plane through C_phys
    cues["table_geometry"] = 1.0 / s_fit
    # (2) MoGe-2 depth (pipeline, aligned to the reconstructed human) on table-top pixels around the iron
    try:
        f = h5py.File("/pipeline/depth/moge2/aligned.h5")
        A = f["aligned/front_stereo_camera_left"]
        rim = np.array(table["rim_px"], float)
        cx, cy = (rim[0, 0] + rim[1, 0]) / 2, (rim[2, 1] + rim[3, 1]) / 2
        ax, ay = (rim[1, 0] - rim[0, 0]) / 2 * 0.9, (rim[3, 1] - rim[2, 1]) / 2 * 0.8
        yy, xx = np.mgrid[0:1152, 0:1536]
        ell = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 < 1
        Ki = np.linalg.inv(K)
        x0, y0, x1, y1 = crop
        ds_ = []
        for i in range(0, min(F, 100), 4):
            dm = cv2.imdecode(A[i], cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
            iron = np.zeros((1152, 1536), np.uint8)
            iron[y0:y1, x0:x1] = M_crop[i]
            iron = cv2.dilate(iron, np.ones((9, 9), np.uint8)) > 0
            sel = ell & ~iron & (data["touch"][i] == 0)
            if sel.sum() < 100:
                continue
            P = (np.stack([xx[sel], yy[sel], np.ones(sel.sum())], 1) @ Ki.T) * dm[sel][:, None]
            ds_.append(-np.median(P @ n))
        cues["moge_table"] = float(np.median(ds_) / d_fit)
    except Exception as e:  # pragma: no cover
        log("MoGe cue failed", e)
    # (3) right hand (reconstructed human, depth calibrated on body + floor) holding the handle
    Tcw = np.linalg.inv(Twc)
    kp = rec["human_keypoints_world"].astype(np.float64)
    rh = (kp[:, 21:42] @ Tcw[:3, :3].T + Tcw[:3, 3]).mean(1)
    G = np.einsum("fij,j->fi", T_fit[:, :3, :3], GRIP_MESH) + T_fit[:, :3, 3]
    uvh, uvg = project(K, rh), project(K, G)
    near = (np.linalg.norm(uvh - uvg, axis=1) < 25) & (data["touch"] > 0)
    if near.sum() > 20:
        ratio = rh[near, 2] / G[near, 2]
        cues["right_hand"] = float(np.median(ratio))
        cues["right_hand_n"] = int(near.sum())
        cues["right_hand_iqr"] = [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))]
    cues["object_size"] = 1.0
    return cues


def apply_scale(T, k, cm):
    """Scale the camera-frame trajectory by k about the camera centre (keeps image positions), about mesh centre."""
    T = T.copy()
    c = np.einsum("fij,j->fi", T[:, :3, :3], cm) + T[:, :3, 3]
    T[:, :3, 3] = k * c - np.einsum("fij,j->fi", T[:, :3, :3], cm)
    return T
