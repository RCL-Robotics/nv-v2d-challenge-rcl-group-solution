#!/usr/bin/env python
"""Trajectory-level re-estimation of the round-table pose for every video frame.

Pipeline (all estimates recomputed from the inputs; see REPORT.md for the reasoning):
  1. Load baseline (CARI4D) poses, masks, video.
  2. Mesh analysis: fit the 5-fold symmetry axis of the leg star (the table axis).
  3. Phase segmentation: static phases from in-mask image differences (+ contact for the first/last phase);
     tracker-slip windows from the baseline's spin rate about the table axis.
  4. Spin hypotheses: the leg star is 5-fold symmetric, so every segment between slips is tested against
     the 5 hypotheses  T_base * Spin(72 k).  Evidence = pooled, rectified texture of the table underside
     (mounting plate, label, lever) and of the table top (wood grain) over all frames of the segment,
     correlated with the mesh texture.  The top texture of this mesh is found to be offset from the
     underside by a constant grid angle, which is calibrated automatically from segments where both are seen.
  5. Apply the per-segment spin, bridge slip windows by interpolation.
  6. Joint silhouette refinement of all frames (nvdiffrast, SAM2 masks, person pixels ignored) with
     second-order temporal smoothness, one shared pose per static phase.
  7. Re-anchor depth: per phase, scale camera-frame positions along the viewing rays so that the median
     camera depth of the table centre equals the baseline's.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn
import cv2
import h5py
import scipy.ndimage as nd
from scipy.spatial import cKDTree
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_inputs, load_mesh, Renderer, rot_angle_deg, axis_angle, dr  # noqa: E402
from rectify import plane_grid, rectify, rotate_grid  # noqa: E402

GRID = np.array([0, 72, 144, -144, -72])
Z_TOP, Z_UNDER = -0.115, -0.092  # mesh z of the top surface (normal -z) and of the underside (normal +z)


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


# ----------------------------------------------------------------------------------------------
def fit_symmetry_axis(v):
    """Fit the 5-fold symmetry axis (point in z=0 plane + direction) of the leg arms."""
    r = np.hypot(v[:, 0], v[:, 1])
    legs = v[(v[:, 2] > 0.5) & (v[:, 2] < 0.588) & (r > 0.06) & (r < 0.30)][::5]
    tree = cKDTree(legs)

    def cost(p):
        c = np.array([p[0], p[1], 0]); ax = np.array([p[2], p[3], 1.]); ax /= np.linalg.norm(ax)
        tot = 0
        for k in (1, 2):
            Rk = axis_angle(ax, np.radians(72 * k))
            d, _ = tree.query((legs - c) @ Rk.T + c)
            tot += np.mean(np.minimum(d, 0.03) ** 2)
        return tot
    p = minimize(cost, [0.006, 0.004, 0, 0], method='Nelder-Mead', options=dict(maxiter=400, xatol=1e-5, fatol=1e-9)).x
    ax = np.array([p[2], p[3], 1.]); ax /= np.linalg.norm(ax)
    return np.array([p[0], p[1], 0.]), ax


def spin_mat(c, ax, deg):
    """4x4 mesh-frame rotation by deg about the table axis (point c, direction ax)."""
    S = np.eye(4)
    S[:3, :3] = axis_angle(ax, np.radians(deg))
    S[:3, 3] = c - S[:3, :3] @ c
    return S


def baseline_spin_rate(T, ax):
    """Per-step rotation of the baseline about the body-frame table axis (deg)."""
    R = T[:, :3, :3]
    rel = np.einsum('fji,fjk->fik', R[:-1], R[1:])
    out = []
    for M in rel:
        rv = cv2.Rodrigues(M)[0][:, 0]
        out.append(np.degrees(rv @ ax))
    return np.array(out)


def find_slips(spin, win=20, thr=45.0, pad=4):
    """Windows where the baseline's accumulated spin changes by > thr within `win` frames."""
    cs = np.r_[0, np.cumsum(spin)]
    s = cs[win:] - cs[:-win]
    bad = np.abs(s) > thr
    lab, n = nd.label(bad)
    slips = []
    for i in range(1, n + 1):
        w = np.where(lab == i)[0]
        c0, c1 = w[0], w[-1] + win
        pre = np.median(cs[max(c0 - 30, 0):c0 + 1]); post = np.median(cs[c1:c1 + 30])
        idx = np.arange(max(c0 - 10, 0), min(c1 + 10, len(cs) - 1))
        dev = (np.abs(cs[idx] - pre) > 6) & (np.abs(cs[idx] - post) > 6)
        if dev.any():
            a, b = idx[dev][0] - pad, idx[dev][-1] + pad
        else:
            a, b = c0, c1
        slips.append([int(max(a, 0)), int(min(b, len(cs) - 1)), float(post - pre)])
    merged = []  # merge overlapping windows
    for sl in slips:
        if merged and sl[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], sl[1])
            merged[-1][2] = float(cs[min(merged[-1][1] + 15, len(cs) - 1)] - cs[max(merged[-1][0] - 15, 0)])
        else:
            merged.append(sl)
    return [tuple(m) for m in merged]


# ----------------------------------------------------------------------------------------------
class TextureEvidence:
    """Pooled rectified texture of the top / underside vs. mesh texture, as a function of spin."""
    n = 400
    angs = np.arange(-180, 180, 2)

    def __init__(self, mesh, renderer, K, dev='cuda'):
        self.K = K
        self.R = renderer
        yy, xx = np.mgrid[:self.n, :self.n]
        rad = np.hypot(xx - self.n / 2 + 0.5, yy - self.n / 2 + 0.5) * 0.002
        self.ann = (rad > 0.05) & (rad < 0.34)
        self.P = {'top': plane_grid(Z_TOP), 'under': plane_grid(Z_UNDER)}
        self.sign = {'top': -1, 'under': 1}
        self.refs = {}
        v, f = mesh['v'], mesh['f']
        for which in ('top', 'under'):
            keep = v[f][:, :, 2].max(1) < 0.0 if which == 'under' else np.ones(len(f), bool)
            m2 = dict(mesh); m2['f'] = f[keep]
            Kr = np.array([[2000, 0, 600], [0, 2000, 600], [0, 0, 1.]])
            Rr = Renderer(m2, Kr, 1200, 1200)
            z0 = Z_UNDER if which == 'under' else Z_TOP
            Tr = np.eye(4)
            if which == 'under':
                Tr[:3, :3] = np.diag([1, -1, -1])  # camera on the +z side looking at the underside
            Tr[:3, 3] = -Tr[:3, :3] @ np.array([0.0, 0.0, z0]) + np.array([0, 0, 2.5])
            o = Rr.render(Tr[None], depth=True)
            ref, wref = rectify(o['rgb'][0].cpu().numpy(), Tr, Kr, self.P[which],
                                depth=o['depth'][0].cpu().numpy(), normal_sign=self.sign[which])
            self.refs[which] = [self.prep(rotate_grid(ref, s), rotate_grid(wref, s)) for s in self.angs]
            del Rr
        self.reset()

    def prep(self, img, w):
        g = img[..., :3].mean(-1)
        m = (w > 0) & self.ann
        g = np.where(m, g, 0)
        b = cv2.GaussianBlur(g, (0, 0), 8); bw = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 8)
        hp = np.where(m, g - b / np.maximum(bw, 1e-3), 0)
        hp = cv2.GaussianBlur(hp, (0, 0), 1.5) / np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5), 1e-3)
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        return hp, m

    def reset(self):
        self.acc = {'top': 0, 'under': 0}
        self.wacc = {'top': 0, 'under': 0}

    def add(self, img, T, obj_mask, person_mask, depth):
        vm = cv2.erode((obj_mask & ~person_mask).astype(np.float32), np.ones((5, 5), np.uint8))
        for which in ('top', 'under'):
            rr, w = rectify(img, T, self.K, self.P[which], depth=depth, valid_mask=vm,
                            normal_sign=self.sign[which], min_facing=0.1)
            if w.sum() < 2000:
                continue
            g = (rr * w[..., None]).sum((0, 1)) / w.sum()
            self.acc[which] = self.acc[which] + rr / g * w[..., None]
            self.wacc[which] = self.wacc[which] + w

    def curve(self, which):
        if np.isscalar(self.wacc[which]):
            return None
        img = self.acc[which] / np.maximum(self.wacc[which], 1e-6)[..., None]
        a, ma = self.prep(img, self.wacc[which])
        cs = []
        for b, mb in self.refs[which]:
            m = ma & mb
            if m.sum() < 2000:
                cs.append(np.nan); continue
            x = a[m] - a[m].mean(); y = b[m] - b[m].mean()
            cs.append((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
        cs = np.array(cs)
        return None if np.all(np.isnan(cs)) else cs


def grid_scores(curve, angs, tol=10):
    """Max correlation within +-tol deg of every grid angle."""
    out = []
    for g in GRID:
        d = np.abs(((angs - g) + 180) % 360 - 180)
        out.append(np.nanmax(np.where(d <= tol, curve, np.nan)))
    return np.nan_to_num(np.array(out), nan=0.0)


# ----------------------------------------------------------------------------------------------
def rodrigues_batch(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th[..., None]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def refine_silhouettes(Tc_init, statics, mesh, K, M, P, centre, epochs=40, ws=1.0, lr=3e-3, seed=0, holdout=None):
    """Joint refinement of all frames: silhouette loss + 2nd-order smoothness; static phases share a pose."""
    torch.manual_seed(seed)
    dev = 'cuda'
    F = len(Tc_init)
    Kh = K.copy(); Kh[:2] *= 0.5
    R = Renderer(mesh, Kh, M.shape[1], M.shape[2])
    R.v = R.v - torch.tensor(centre, dtype=torch.float32, device=dev)
    unit = np.arange(F)
    R0 = Tc_init[:, :3, :3].copy(); p0 = Tc_init[:, :3, :3] @ centre + Tc_init[:, :3, 3]
    for (a, b) in statics:
        idx = np.arange(a, b + 1)
        U, _, Vt = np.linalg.svd(R0[idx].sum(0)); Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        R0[idx] = Rm; p0[idx] = np.median(p0[idx], 0); unit[idx] = a
    R0t = torch.tensor(R0, dtype=torch.float32, device=dev); p0t = torch.tensor(p0, dtype=torch.float32, device=dev)
    ut = torch.tensor(unit, device=dev)
    use = torch.ones(F, device=dev)
    if holdout is not None:
        use[torch.tensor(np.where(holdout)[0], device=dev)] = 0.0
    om = torch.zeros(F, 3, device=dev, requires_grad=True); dp = torch.zeros(F, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [om], 'lr': lr}, {'params': [dp], 'lr': lr * 0.5}])
    Mt = torch.tensor(M, device=dev); Pt = torch.tensor(P, device=dev)
    proj = R.proj(R.near, R.far)
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)

    def poses():
        return rodrigues_batch(om[ut]) @ R0t, p0t + dp[ut]

    def sil(idx, Rf, pf):
        T = torch.zeros(len(idx), 4, 4, device=dev); T[:, :3, :3] = Rf[idx]; T[:, :3, 3] = pf[idx]; T[:, 3, 3] = 1
        clip = torch.einsum('ij,bnj->bni', proj, torch.einsum('bij,nj->bni', T, vh)).contiguous()
        rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
        soft = dr.antialias((rast[..., 3:] > 0).float(), rast, clip, R.f)[..., 0]
        m = Mt[idx].float(); valid = 1 - (Pt[idx] & ~Mt[idx]).float()
        l = (((soft - m) ** 2) * valid).sum((1, 2)) / (m.sum((1, 2)) + 100)
        s8 = Fn.avg_pool2d((soft * valid)[:, None], 8); m8 = Fn.avg_pool2d((m * valid)[:, None], 8)
        l = l + 4 * ((s8 - m8) ** 2).sum((1, 2, 3)) / (m.sum((1, 2)) / 64 + 2)
        return l * use[idx]

    def smooth(Rf, pf):
        return ((Rf[2:] - 2 * Rf[1:-1] + Rf[:-2]) ** 2).sum() + ((pf[2:] - 2 * pf[1:-1] + pf[:-2]) ** 2).sum() / 0.39 ** 2

    bs = 48
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(F, generator=g).to(dev)
        tot = 0
        for b in range(0, F, bs):
            idx = perm[b:b + bs]
            Rf, pf = poses()
            ls = sil(idx, Rf, pf).sum()
            loss = ls + smooth(Rf, pf) * ws * 1e3 * len(idx) / F
            opt.zero_grad(); loss.backward(); opt.step()
            tot += ls.item()
        if ep % 10 == 0 or ep == epochs - 1:
            log('  refine epoch %d  silhouette loss %.4f' % (ep, tot / F))
    with torch.no_grad():
        Rf, pf = poses()
    Rf = Rf.cpu().numpy().astype(np.float64); pf = pf.cpu().numpy().astype(np.float64)
    Tc = np.tile(np.eye(4), (F, 1, 1))
    Tc[:, :3, :3] = Rf; Tc[:, :3, 3] = pf - Rf @ centre
    del R
    torch.cuda.empty_cache()
    return Tc


def slerp_bridge(Tc, a, b):
    """Replace frames a+1..b-1 by interpolation between the poses at a and b (camera frame)."""
    Ra, Rb = Tc[a, :3, :3], Tc[b, :3, :3]
    rv = cv2.Rodrigues(Ra.T @ Rb)[0][:, 0]
    for f in range(a + 1, b):
        t = (f - a) / (b - a)
        Tc[f, :3, :3] = Ra @ cv2.Rodrigues(rv * t)[0]
        Tc[f, :3, 3] = (1 - t) * Tc[a, :3, 3] + t * Tc[b, :3, 3]
    return Tc


def iou_all(Tc, mesh, K, M, P):
    Kh = K.copy(); Kh[:2] *= 0.5
    R = Renderer(mesh, Kh, M.shape[1], M.shape[2])
    out = np.zeros(len(Tc))
    for s in range(0, len(Tc), 32):
        o = R.render(Tc[s:s + 32], rgb=False)['mask'].cpu().numpy()
        for i in range(o.shape[0]):
            f = s + i; r = o[i] & ~(P[f] & ~M[f])
            out[f] = (r & M[f]).sum() / max((r | M[f]).sum(), 1)
    del R
    return out


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--diag', default=None, help='optional path for diagnostics npz')
    ap.add_argument('--ws', type=float, default=0.5, help='smoothness weight (validated by --holdout)')
    ap.add_argument('--holdout', action='store_true', help='validation only: fit odd frames out of the silhouette loss')
    args = ap.parse_args()
    np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    inp = args.inputs
    D = load_inputs(inp); mesh = load_mesh(inp)
    K, Twc = D['K'], D['Twc']
    Tb_w = D['rec']['T_world_mesh'].astype(np.float64)
    F = len(Tb_w)
    Tb = np.linalg.inv(Twc) @ Tb_w  # baseline, camera frame
    H, W = D['H'], D['W']
    log('frames', F)

    # ---------------- 2. table axis
    c_ax, d_ax = fit_symmetry_axis(mesh['v'])
    centre = c_ax + d_ax * 0.27  # point on the axis at mid height (spin invariant reference point)
    log('symmetry axis point', np.round(c_ax, 4), 'dir', np.round(d_ax, 4))

    # ---------------- data at half resolution (masks) + in-mask image differences
    with h5py.File(f'{inp}/masks.h5', 'r') as hf:
        Mfull_o, Mfull_p = hf['object'], hf['person']
        Mh, Ph = [], []
        for s0 in range(0, F, 64):
            for src, dst in ((Mfull_o, Mh), (Mfull_p, Ph)):
                blk = src[s0:s0 + 64]
                dst += [cv2.resize(x.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0 for x in blk]
        Mh, Ph = np.stack(Mh), np.stack(Ph)
    log('masks loaded')

    # ---------------- 3a. static phases: median in-mask grey-level change over 5 frames
    cap = cv2.VideoCapture(f'{inp}/video.mp4')
    grey = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(im, (W // 2, H // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        grey.append(cv2.GaussianBlur(g.astype(np.float32), (5, 5), 0))
    cap.release()
    k5 = np.ones((5, 5), np.uint8)
    pix = np.full(F, np.nan)
    for f in range(5, F):
        m = cv2.erode(Mh[f].astype(np.uint8), k5).astype(bool) & ~cv2.dilate(Ph[f].astype(np.uint8), k5).astype(bool) \
            & ~cv2.dilate(Ph[f - 5].astype(np.uint8), k5).astype(bool)
        if m.sum() >= 50:
            pix[f] = np.median(np.abs(grey[f] - grey[f - 5])[m])
    pix[:5] = pix[5]
    del grey
    pixm = nd.median_filter(np.nan_to_num(pix, nan=9.0), 15)
    contact = D['rec']['contact_prob_right_left'].max(1)
    lab, n = nd.label(pixm < 0.65)  # 0.65 grey levels: noise level of the untouched table is ~0.3
    statics = []
    for i in range(1, n + 1):
        w = np.where(lab == i)[0]
        if len(w) < 25:
            continue
        a, b = w[0], w[-1]
        if a <= 5:
            a = 0
        if b >= F - 6:
            b = F - 1
        # the first/last phase is "untouched at rest": cut it where the hands touch the table
        if a == 0:
            t = np.where(contact[:b + 1] > 0.5)[0]
            b = t[0] - 1 if len(t) else b
        if b == F - 1:
            t = np.where(contact[a:] > 0.5)[0]
            a = a + t[-1] + 1 if len(t) else a
        if b - a >= 20:
            statics.append((int(a), int(b)))
    log('static phases', statics)

    # ---------------- 3b. tracker slips of the baseline about the table axis
    spin = baseline_spin_rate(Tb, d_ax)
    slips = find_slips(spin)
    log('baseline slip windows (a, b, spin change)', [(a, b, round(s, 1)) for a, b, s in slips])
    bounds = [0] + [x for a, b, _ in slips for x in (a, b)] + [F - 1]
    segments = [(bounds[2 * i], bounds[2 * i + 1]) for i in range(len(slips) + 1)]
    log('segments', segments)

    # ---------------- 4. texture evidence per segment (windows of 30 frames, every 2nd frame)
    Rfull = Renderer(mesh, K, H, W)
    TE = TextureEvidence(mesh, Rfull, K)
    win_curves = []  # (which, start, curve)
    cap = cv2.VideoCapture(f'{inp}/video.mp4')
    with h5py.File(f'{inp}/masks.h5', 'r') as hf:
        Mo, Mp = hf['object'], hf['person']
        wstart = 0
        for f in range(F):
            ok, im = cap.read()
            if f % 2 == 0:
                om = Mo[f]
                if om.sum() > 200:
                    dep = Rfull.render(Tb[f][None], rgb=False, depth=True)['depth'][0].cpu().numpy()
                    TE.add(im[:, :, ::-1].astype(np.float32) / 255, Tb[f], om, Mp[f], dep)
            if f % 30 == 29 or f == F - 1:
                for which in ('top', 'under'):
                    cv_ = TE.curve(which)
                    if cv_ is not None:
                        win_curves.append((which, wstart, cv_))
                TE.reset(); wstart = f + 1
    cap.release()
    log('texture windows', len(win_curves))
    seg_ev = []
    for (a, b) in segments:
        ev = {}
        for which in ('top', 'under'):
            sc = np.zeros(len(GRID)); nw = 0
            for wh, s0, cv_ in win_curves:
                if wh == which and s0 >= a and s0 + 29 <= b:
                    sc += grid_scores(cv_, TE.angs); nw += 1
            ev[which] = (sc, nw)
        seg_ev.append(ev)
        log('segment %s: under %s (n=%d)  top %s (n=%d)' % ((a, b), np.round(ev['under'][0], 2), ev['under'][1],
                                                          np.round(ev['top'][0], 2), ev['top'][1]))

    def decide(sc, min_margin=0.3):
        o = np.argsort(-sc)
        return (int(GRID[o[0]]), float(sc[o[0]] - sc[o[1]])) if sc[o[0]] - sc[o[1]] > min_margin else (None, 0.0)

    # calibrate the constant top-vs-underside texture offset from segments where both are decisive
    diffs = []
    for ev in seg_ev:
        gu, _ = decide(ev['under'][0]); gt, _ = decide(ev['top'][0])
        if gu is not None and gt is not None:
            diffs.append(((gu - gt) + 180) % 360 - 180)
    top_offset = 0
    if diffs:
        vals, cnt = np.unique(np.round(np.array(diffs) / 72).astype(int) * 72, return_counts=True)
        top_offset = int(vals[np.argmax(cnt)])
        if cnt.max() < len(diffs):
            log('WARNING: inconsistent top/underside offsets', diffs)
    log('top texture offset vs underside (deg):', top_offset, 'from', diffs)
    seg_spin = []
    for (a, b), ev in zip(segments, seg_ev):
        gu, mu = decide(ev['under'][0])
        if gu is not None:
            g, src = gu, 'underside'
        else:
            gt, mt = decide(ev['top'][0])
            g, src = ((gt + top_offset + 180) % 360 - 180, 'top') if gt is not None else (0, 'none (baseline)')
        seg_spin.append(g)
        log('segment %s -> spin %d deg (%s)' % ((a, b), g, src))

    # ---------------- 5. apply spins, bridge slip windows
    Tc = Tb.copy()
    for (a, b), g in zip(segments, seg_spin):
        S = spin_mat(c_ax, d_ax, g)
        Tc[a:b + 1] = Tb[a:b + 1] @ S
    for (a, b, _) in slips:
        Tc = slerp_bridge(Tc, a, b)

    # ---------------- 6. joint silhouette refinement
    hold = (np.arange(F) % 2 == 1) if args.holdout else None
    Tr = refine_silhouettes(Tc, statics, mesh, K, Mh, Ph, centre, ws=args.ws, holdout=hold)
    if args.holdout:
        i0 = iou_all(Tc, mesh, K, Mh, Ph); i1 = iou_all(Tr, mesh, K, Mh, Ph); ib = iou_all(Tb, mesh, K, Mh, Ph)
        log('HOLDOUT (odd frames, not in loss): IoU baseline %.4f, spin-corrected init %.4f, refined %.4f' % (
            ib[hold].mean(), i0[hold].mean(), i1[hold].mean()))
        log('fitted (even) frames: baseline %.4f init %.4f refined %.4f' % (ib[~hold].mean(), i0[~hold].mean(), i1[~hold].mean()))
        return

    # ---------------- 7. depth re-anchoring to the baseline (along the viewing rays)
    # Static phases: one scale per phase (median z = baseline median z).  Moving phases: a slowly varying
    # scale = running median (+-45 frames, inside the phase) of baseline z / our z, Gaussian-smoothed, so the
    # baseline's calibrated low-frequency depth is kept and only our high-frequency shape is used.
    cuts = sorted(set([0, F] + [x for a, b in statics for x in (a, b + 1)]))
    phases = [(cuts[i], cuts[i + 1] - 1) for i in range(len(cuts) - 1)]
    zb = (Tb[:, :3, :3] @ centre + Tb[:, :3, 3])[:, 2]
    pc_all = Tr[:, :3, :3] @ centre + Tr[:, :3, 3]
    scale = np.ones(F)
    for (a, b) in phases:
        ratio = zb[a:b + 1] / pc_all[a:b + 1, 2]
        if any(a == s0 for s0, _ in statics):
            scale[a:b + 1] = np.median(zb[a:b + 1]) / np.median(pc_all[a:b + 1, 2])
        else:
            rm = np.array([np.median(ratio[max(i - 45, 0):i + 46]) for i in range(len(ratio))])
            scale[a:b + 1] = nd.gaussian_filter1d(rm, 10, mode='nearest')
        log('phase %s depth scale %.4f..%.4f (median shift %.1f cm)' % ((a, b), scale[a:b + 1].min(), scale[a:b + 1].max(),
                                                                       np.median((scale[a:b + 1] - 1) * pc_all[a:b + 1, 2]) * 100))
    Tout = Tr.copy()
    Tout[:, :3, 3] = pc_all * scale[:, None] - Tr[:, :3, :3] @ centre
    Tw = Twc @ Tout
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    log('saved', args.out)

    if args.diag:
        iou_b = iou_all(Tb, mesh, K, Mh, Ph); iou_r = iou_all(Tout, mesh, K, Mh, Ph)
        np.savez(args.diag, iou_base=iou_b, iou_res=iou_r, statics=np.array(statics), phases=np.array(phases),
                 segments=np.array(segments), seg_spin=np.array(seg_spin), slips=np.array(slips),
                 top_offset=top_offset, pix=pix, T_pre_refine=Tc, centre=centre, c_ax=c_ax, d_ax=d_ax)
        log('IoU base %.3f result %.3f' % (iou_b.mean(), iou_r.mean()))


if __name__ == '__main__':
    T0 = time.time()
    main()
