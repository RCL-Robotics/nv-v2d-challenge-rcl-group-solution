"""G2_surface_only batch MAP model (Tier-2 data only), refactored from B_hands (ep28) to take any episode.

State per frame x[t] (12): b_p (cm), b_r (rad) object measurement bias with R_true = exp(-b_r) R_meas,
p_true = p_meas - b_p; d_t (cm), d_r (rad) human root drift (pivot = measured Hips). Residual blocks and numerics
(Levenberg-Marquardt, grouped finite-difference sparse Jacobian) are those of B_hands except:
  * hand model: on detected grasp frames every grip point of the hand lies on the object's mesh surface (vector to the
    nearest dense surface sample, soft-L1 per point) and its motion relative to the object, normal to that surface,
    follows the grip slip random walk (soft-L1 per point); no handle axis;
  * support boxes come from the episode's support USDA; support contact uses the supports detected as touched;
  * invisible frames carry no bias random-walk link (bridged by one link across interior gaps)."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation as R
from common import kf, kc, CONST, DEFAULT_OPTS

DEG = np.pi / 180
FEET = {'L': (69, 71), 'R': (74, 76)}
HAND_IDX = [list(range(14, 39)), list(range(42, 67))]
FING_L = {'thumb': [15, 16, 17, 18], 'index': [19, 20, 21, 22, 23], 'middle': [24, 25, 26, 27, 28],
          'ring': [29, 30, 31, 32, 33], 'pinky': [34, 35, 36, 37, 38]}


def fingers(side):
    off = 0 if side == 0 else 28
    return {k: [j + off for j in v] for k, v in FING_L.items()}


def grip_pts(side):
    """grip points of a hand (SOMA joints): thumb distal + tip, PIP, DIP, tip of the 4 fingers (14 points)."""
    f = fingers(side); out = [f['thumb'][2], f['thumb'][3]]
    for k in ('index', 'middle', 'ring', 'pinky'):
        out += f[k][2:]
    return out


def runs(s):
    return kc.runs(np.asarray(s, bool))


def keep_runs(s, lmin):
    return kc.keep_runs(np.asarray(s, bool), lmin)


def fill_gaps(s, gmax):
    s = np.asarray(s, bool).copy(); rr = runs(s)
    for (a0, b0), (a1, b1) in zip(rr[:-1], rr[1:]):
        if a1 - b0 - 1 <= gmax:
            s[b0:a1] = True
    return s


def soft_l1(r):
    """row-wise soft-L1 on normalised residual vectors r (n,d): returns r' with |r'|^2 = 2(sqrt(1+|r|^2)-1)."""
    r = np.atleast_2d(r)
    n = np.linalg.norm(r, axis=1, keepdims=True)
    f = np.sqrt(2 * (np.sqrt(1 + n ** 2) - 1)) / np.maximum(n, 1e-12)
    f = np.where(n < 1e-9, 1.0, f)
    return r * f


class Episode:
    """One episode's cached Tier-2 inputs and its model parameters (cm, OpenCV world unless stated)."""

    def __init__(self, cache, opts=None):
        self.opts = {**DEFAULT_OPTS, **(opts or {})}
        c = np.load(cache)
        self.ep, self.obj = int(c['ep']), str(c['obj'])
        self.T = T = len(c['p_m'])
        self.PM = c['p_m'].astype(float); self.RM = R.from_quat(c['q_m']); self.vis = c['vis'].astype(bool)
        self.rest = c['rest'].astype(bool)
        self.JM = c['J'].astype(float); self.UP = c['up'].astype(float); self.E0 = float(c['e'])
        self.BCR = c['Bcv2r']; self.bCR = c['bcv2r']; self.HIPS = self.JM[:, 0]
        V = c['Vs'] * 100.0
        self.HULL = V[ConvexHull(V).vertices]                    # floor test: extreme points of the mesh
        self.PTS = c['pts'] * 100.0                              # the kit's 500 vertex samples
        self.BPTS = np.r_[self.PTS, self.HULL] if self.opts['box_pts'] == 'pts+hull' else self.PTS
        self.BOXES = [((lo + hi) / 2, (hi - lo) / 2) for lo, hi in c['boxes']]
        self.S = c['S'] * 100.0; self.N = c['N'].astype(float); self.tree = cKDTree(self.S)
        self.init = c['init']; self.Pb = c['Pb']
        self.GRIP = [grip_pts(0), grip_pts(1)]
        self.K = len(self.GRIP[0])
        PRM = kf.load_params()
        # hand high-frequency floor (cm/frame/axis): median over the grip joints of both hands of the ZUPT v1 PSD floor
        sig = []
        for j in self.GRIP[0] + self.GRIP[1]:
            _, _, fl = kc.psd_floor(np.diff(self.JM[:, j], axis=0)); sig.append(np.sqrt(fl * kc.FPS / 2.0))
        self.s_hf = float(np.median(sig))
        self.P = dict(
            q_bp=PRM['q_bp_cm'], q_br=PRM['q_br_deg'] * DEG, s0_p=PRM['P_b0_pos_cm'], s0_r=PRM['P_b0_rot_deg'] * DEG,
            q_ht=PRM['q_root_trans_cm'], q_hr=PRM['q_root_rot_deg'] * DEG, h0_p=PRM['P_b0_pos_cm'], h0_r=PRM['P_b0_rot_deg'] * DEG,
            s_a=float(np.sqrt(2 / 3 * float(c['q_a']) * kc.DT ** 3)), s_al=float(np.sqrt(2 / 3 * float(c['q_al']) * kc.DT ** 3)),
            s_zv=PRM['sigma_zv_pos_cm_s'] * kc.DT, s_zvr=PRM['sigma_zv_rot_deg_s'] * kc.DT * DEG,
            s_fs=CONST['s_fs'], s_slip=float(np.sqrt(self.s_hf ** 2 + CONST['slip_rw'])),
            s_c=CONST['s_c'], n_corr=CONST['n_corr'], s_pen=CONST['s_pen'], s_sup_f=CONST['s_sup_f'], s_sup_b=CONST['s_sup_b'])
        # per-point sigmas of the surface hand model: the K grip points of a hand share one grasp offset, so their K
        # 1-D normal residuals together weigh as one 2-D grip-centre residual of B_hands (sqrt(K/2) per point)
        kfac = np.sqrt(self.K / 2.0)
        self.P['s_surf'] = self.P['s_c'] * np.sqrt(self.P['n_corr']) * kfac
        self.P['s_slip_pt'] = self.P['s_slip'] * kfac

    # ---------------------------------------------------------------- geometry
    def pose(self, x):
        return self.PM - x[:, 0:3], R.from_rotvec(-x[:, 3:6]) * self.RM

    def human_true(self, x, idx):
        """true positions (T,K,3) of SOMA joints idx under the drift in x (pivot = measured Hips)."""
        dt, dr = x[:, 6:9], x[:, 9:12]
        rel = self.JM[:, idx] - self.HIPS[:, None]
        Rd = R.from_rotvec(dr).as_matrix()          # x_m - hips_m = R_d (x_true - hips_true)
        return self.HIPS[:, None] - dt[:, None] + np.einsum('tji,tkj->tki', Rd, rel)

    def to_obj(self, x, X):
        """world points X (T,K,3) cm -> object frame (cm) under the object pose of x."""
        p, Rt = self.pose(x)
        return np.einsum('tji,tkj->tki', Rt.as_matrix(), X - p[:, None])

    def grip_obj(self, x, s):
        return self.to_obj(x, self.human_true(x, self.GRIP[s]))

    def heights(self, x, pts_obj):
        p, Rt = self.pose(x)
        W = np.einsum('tij,kj->tki', Rt.as_matrix(), pts_obj) + p[:, None]
        return W @ self.UP + 100 * self.E0, W

    def box_terms(self, W):
        """W (n,K,3) cm OpenCV -> (max penetration depth over boxes (n,) cm, distance to each box (n,nb) cm)."""
        Wr = (W / 100.0) @ self.BCR.T + self.bCR
        pen = np.zeros(len(W)); dist = np.full((len(W), len(self.BOXES)), np.inf)
        for b, (cen, half) in enumerate(self.BOXES):
            q = np.abs(Wr - cen) - half
            inside = np.all(q < 0, axis=2)
            depth = np.where(inside, -q.max(axis=2), 0.0) * 100
            pen = np.maximum(pen, depth.max(1))
            dist[:, b] = (np.linalg.norm(np.maximum(q, 0), axis=2) * 100).min(1)
        return pen, dist

    def to_pose(self, x):
        p, Rt = self.pose(x)
        return np.concatenate([p / 100.0, Rt.as_quat()], 1)

    def x_from_pose(self, pose_vis):
        """bias state from a pose estimate on the visible frames (T_vis,7) metres xyzw; invisible frames take the bias of
        the nearest visible frame (their measurement was filled the same way); human drift 0."""
        vi = np.flatnonzero(self.vis)
        full = np.empty((self.T, 7)); full[vi] = pose_vis
        for t in np.flatnonzero(~self.vis):
            j = vi[np.argmin(np.abs(vi - t))]
            full[t] = full[j]
        x = np.zeros((self.T, 12))
        x[:, 0:3] = self.PM - full[:, :3] * 100.0
        x[:, 3:6] = (self.RM * R.from_quat(full[:, 3:]).inv()).as_rotvec()     # exp(b) = R_m R_true^T
        return x


class Problem:
    """Residual blocks with their state-component dependencies (for a fast grouped finite-difference Jacobian)."""
    OBJ_P, OBJ_R, HUM = (0, 1, 2), (3, 4, 5), tuple(range(6, 12))

    def __init__(self, E, static, grasp, stance, sup_f, sup_b, x_ref, w=None):
        self.E = E; T = E.T
        self.static, self.grasp, self.stance = static, grasp, stance
        self.sup_f, self.sup_b = sup_f, sup_b                     # (T,) floor contact, (T,nb) box contact frames
        self.w = dict(hands=1.0, floor=1.0, box=1.0, sup=1.0, feet=1.0) if w is None else dict(w)
        vi = np.flatnonzero(E.vis)
        self.rw_a, self.rw_b = vi[:-1], vi[1:]                    # bias links between consecutive visible frames
        self.t0 = int(vi[0])
        self.sp_static = np.flatnonzero(static[:-1] & static[1:])
        self.gr_pairs = [np.flatnonzero(grasp[s][:-1] & grasp[s][1:]) for s in (0, 1)]
        self.gr_frames = [np.flatnonzero(grasp[s]) for s in (0, 1)]
        self.st_pairs = {f: np.flatnonzero(stance[f][:-1] & stance[f][1:]) for f in FEET}
        self.supf_frames = np.flatnonzero(sup_f)
        self.supb_frames = np.flatnonzero(sup_b.any(1)) if sup_b.size else np.zeros(0, int)
        near = np.zeros(T, bool)
        if E.BOXES:
            _, W = E.heights(x_ref, E.BPTS)                      # frames where the object can reach a support box
            Wr = (W / 100.0) @ E.BCR.T + E.bCR
            for cen, half in E.BOXES:
                near |= (np.linalg.norm(np.maximum(np.abs(Wr - cen) - half, 0), axis=2).min(1) < CONST['box_near_m'])
            near |= sup_b.any(1)
        self.box_frames = np.flatnonzero(near)
        self.stride = int(max(3, (self.rw_b - self.rw_a).max() + 1))
        self.frozen = None
        P_, R_, H_ = self.OBJ_P, self.OBJ_R, self.HUM
        self.blocks = [('rw_bp', set(P_)), ('rw_br', set(R_)), ('b0', set(P_ + R_)), ('acc_p', set(P_)), ('acc_r', set(R_)),
                       ('zv', set(P_ + R_)), ('rw_dt', set(H_)), ('rw_dr', set(H_)), ('d0', set(H_)), ('feet', set(H_)),
                       ('hands_surf', set(P_ + R_ + H_)), ('hands_slip', set(P_ + R_ + H_)), ('floor', set(P_ + R_)),
                       ('box', set(P_ + R_))]

    # nearest dense-surface sample of every grip point on grasp frames; frozen while the Jacobian is differenced
    def correspondences(self, x):
        out = []
        for s in (0, 1):
            fr = self.gr_frames[s]
            if not len(fr):
                out.append(np.zeros((0, self.E.K), int)); continue
            Xo = self.E.grip_obj(x, s)[fr]
            _, idx = self.E.tree.query(Xo.reshape(-1, 3))
            out.append(idx.reshape(len(fr), -1))
        return out

    def block(self, name, x):
        """returns list of (vals (n,d), anchors (n,), span (scalar or (n,)))"""
        out = self._block(name, x)
        return [b for b in out if len(b[1])]

    def _block(self, name, x):
        E, Pp, w, T = self.E, self.E.P, self.w, self.E.T
        bp, br, dt, dr = x[:, 0:3], x[:, 3:6], x[:, 6:9], x[:, 9:12]
        ts = np.arange(T - 1)
        if name == 'rw_bp':
            a, b = self.rw_a, self.rw_b; g = np.sqrt(b - a)[:, None]
            return [((bp[b] - bp[a]) / (Pp['q_bp'] * g), a, b - a + 1)]
        if name == 'rw_br':
            a, b = self.rw_a, self.rw_b; g = np.sqrt(b - a)[:, None]
            return [((R.from_rotvec(br[b]) * R.from_rotvec(br[a]).inv()).as_rotvec() / (Pp['q_br'] * g), a, b - a + 1)]
        if name == 'b0':
            t0 = self.t0
            return [(np.r_[bp[t0] / np.hypot(Pp['s0_p'], np.sqrt(t0) * Pp['q_bp']),
                           br[t0] / np.hypot(Pp['s0_r'], np.sqrt(t0) * Pp['q_br'])][None], np.array([t0]), 1)]
        if name == 'acc_p':
            p = E.PM - bp
            return [((p[2:] - 2 * p[1:-1] + p[:-2]) / Pp['s_a'], np.arange(T - 2), 3)]
        if name in ('acc_r', 'zv'):
            Rt = R.from_rotvec(-br) * E.RM
            om = (Rt[1:] * Rt[:-1].inv()).as_rotvec()
            if name == 'acc_r':
                return [((om[1:] - om[:-1]) / Pp['s_al'], np.arange(T - 2), 3)]
            i = self.sp_static; p = E.PM - bp
            return [(np.c_[(p[i + 1] - p[i]) / Pp['s_zv'], om[i] / Pp['s_zvr']], i, 2)]
        if name == 'rw_dt':
            return [((dt[1:] - dt[:-1]) / Pp['q_ht'], ts, 2)]
        if name == 'rw_dr':
            return [((R.from_rotvec(dr[1:]) * R.from_rotvec(dr[:-1]).inv()).as_rotvec() / Pp['q_hr'], ts, 2)]
        if name == 'd0':
            return [(np.r_[dt[0] / Pp['h0_p'], dr[0] / Pp['h0_r']][None], np.array([0]), 1)]
        if name == 'feet':
            out = []
            if w['feet'] <= 0:
                return out
            for f, (ank, toe) in FEET.items():
                i = self.st_pairs[f]
                if len(i):
                    X = E.human_true(x, [ank, toe])
                    d = (X[i + 1] - X[i]) / Pp['s_fs']
                    out.append((np.sqrt(w['feet']) * np.c_[soft_l1(d[:, 0]), soft_l1(d[:, 1])], i, 2))
            return out
        if name in ('hands_surf', 'hands_slip'):
            out = []
            if w['hands'] <= 0:
                return out
            corr = self.frozen if self.frozen is not None else self.correspondences(x)
            for s in (0, 1):
                fr = self.gr_frames[s]
                if not len(fr):
                    continue
                Xo = E.grip_obj(x, s)                                    # (T,K,3) object frame, drift-corrected hand
                c = corr[s]
                if name == 'hands_surf':
                    d = (Xo[fr] - E.S[c]) / Pp['s_surf']                 # vector to the nearest surface sample
                    out.append((np.sqrt(w['hands']) * soft_l1(d.reshape(-1, 3)).reshape(len(fr), -1), fr, 1))
                else:
                    i = self.gr_pairs[s]
                    if len(i):
                        n = E.N[c[np.searchsorted(fr, i)]]               # (n,K,3) surface normals at frame i
                        dn = np.einsum('nkj,nkj->nk', n, Xo[i + 1] - Xo[i]) / Pp['s_slip_pt']
                        out.append((np.sqrt(w['hands']) * soft_l1(dn.reshape(-1, 1)).reshape(len(i), -1), i, 2))
            return out
        if name == 'floor':
            h, _ = E.heights(x, E.HULL)
            hmin = h.min(1)
            out = [(np.sqrt(w['floor']) * np.maximum(-hmin, 0)[:, None] / Pp['s_pen'], np.arange(T), 1)]
            fr = self.supf_frames
            if len(fr) and w['sup'] > 0:
                out.append((np.sqrt(w['sup']) * soft_l1((hmin[fr] / Pp['s_sup_f'])[:, None]), fr, 1))
            return out
        if name == 'box':
            fr = self.box_frames
            if not len(fr):
                return []
            p, Rt = E.pose(x)
            W = np.einsum('tij,kj->tki', Rt[fr].as_matrix(), E.BPTS) + p[fr][:, None]
            pen, dist = E.box_terms(W)
            out = [(np.sqrt(w['box']) * pen[:, None] / Pp['s_pen'], fr, 1)]
            if w['sup'] > 0:
                cm = self.sup_b[fr]
                m = cm.any(1)
                if m.any():
                    dmin = np.where(cm[m], dist[m], np.inf).min(1)       # nearest of the boxes in contact
                    out.append((np.sqrt(w['sup']) * soft_l1((dmin / Pp['s_sup_b'])[:, None]), fr[m], 1))
            return out
        raise KeyError(name)

    def all_blocks(self, x):
        return {name: self.block(name, x) for name, _ in self.blocks}

    @staticmethod
    def flatten(bl):
        r, anc, span = [], [], []
        for name in bl:
            for vals, a, sp_ in bl[name]:
                vals = np.asarray(vals, float).reshape(len(a), -1)
                r.append(vals.ravel()); anc.append(np.repeat(a, vals.shape[1]))
                span.append(np.repeat(np.broadcast_to(sp_, len(a)), vals.shape[1]))
        return np.concatenate(r), np.concatenate(anc), np.concatenate(span)

    def residuals(self, x):
        return self.flatten(self.all_blocks(x))

    def cost_parts(self, x):
        bl = self.all_blocks(x)
        return {k: float(sum((np.asarray(v[0]) ** 2).sum() for v in bl[k])) for k in bl}

    def jacobian(self, x):
        NV, T, G = 12, self.E.T, self.stride
        self.frozen = self.correspondences(x)
        try:
            bl0 = self.all_blocks(x)
            offs, o = {}, 0
            for name, _ in self.blocks:
                n = sum(np.asarray(v[0]).size for v in bl0[name]); offs[name] = (o, n); o += n
            r0, anc, span = self.flatten(bl0)
            steps = np.array([1e-4] * 3 + [1e-6] * 3 + [1e-4] * 3 + [1e-6] * 3)
            rows, cols, vals = [], [], []
            tt = np.arange(T)
            for comp in range(NV):
                dep = [name for name, cs in self.blocks if comp in cs]
                for k in range(G):
                    xp = x.copy(); xp[tt % G == k, comp] += steps[comp]
                    for name in dep:
                        o, n = offs[name]
                        if n == 0:
                            continue
                        r1 = np.concatenate([np.asarray(v[0], float).ravel() for v in self.block(name, xp)])
                        d = (r1 - r0[o:o + n]) / steps[comp]
                        nz = np.flatnonzero(d)
                        if not len(nz):
                            continue
                        a = anc[o + nz]; sp_ = span[o + nz]
                        off = (k - a) % G
                        ok = off < sp_
                        rows.append(o + nz[ok]); cols.append((a[ok] + off[ok]) * NV + comp); vals.append(d[nz[ok]])
        finally:
            self.frozen = None
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(r0), T * NV)), r0

    def solve(self, x0, iters=60, log=print, tol=1e-6):
        """Levenberg-Marquardt (as B_hands). Returns x and info: status 'rel_tol' (relative decrease < tol),
        'no_decrease' (10 damping increases without a decrease: a local minimum to numerical precision) or 'max_iter'."""
        T = self.E.T
        x = x0.copy(); lam = 1e-3
        r = self.residuals(x)[0]; cost = 0.5 * r @ r
        log(f'  start cost {cost:.2f}')
        status, it = 'max_iter', -1
        for it in range(iters):
            t0 = time.time()
            Jm, r = self.jacobian(x)
            JtJ = (Jm.T @ Jm).tocsc(); g = Jm.T @ r
            dg = JtJ.diagonal()
            improved = False; rel = 0.0
            for _ in range(10):
                A = JtJ + sp.diags(lam * np.maximum(dg, 1e-9) + 1e-9)
                dx = -spla.spsolve(A, g)
                xn = x + dx.reshape(T, 12)
                rn = self.residuals(xn)[0]; cn = 0.5 * rn @ rn
                if cn < cost:
                    rel = (cost - cn) / cost
                    x, cost = xn, cn
                    lam = max(lam / 3, 1e-8); improved = True
                    break
                lam *= 4
            log(f'  it {it:2d} cost {cost:.3f} lam {lam:.1e} rel {rel:.2e} |dx|max {np.abs(dx).max():.3e} ({time.time()-t0:.1f}s)')
            if not improved:
                status = 'no_decrease'; break
            if rel < tol:
                status = 'rel_tol'; break
        return x, dict(status=status, iters=it + 1, cost=float(cost), converged=status != 'max_iter')
