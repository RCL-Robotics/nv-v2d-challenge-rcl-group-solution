#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF object pose for one HOI video.

Pipeline (see REPORT.md):
  1. Phase segmentation from the video: frames where the (unoccluded) object pixels do not change are
     static; maximal static runs >= MIN_STATIC_LEN frames form static phases, the rest is "moving".
  2. Static phases: ONE pose per phase. Initialised at the robust mean of the baseline poses, refined by
     render-and-compare pooled over all frames of the phase (silhouette IoU with a fat-mask model +
     image-edge alignment), with the camera depth of the object centre held fixed (rotation + lateral only).
     Three optimisation schedules are run and their results averaged.
  3. Moving phase: baseline smoothed in time (SE(3) Gaussian), then per-frame silhouette render-and-compare
     refinement of rotation + lateral position (depth of the object centre fixed to the smoothed baseline),
     then light temporal smoothing.
  4. Depth re-anchoring: per phase, camera-frame translations are scaled along their viewing rays so that the
     median camera z equals the baseline's median z over the same frames (no physical depth shift applied).
Deterministic: fixed seeds, deterministic schedules; GPU use < 3 GB.
"""
import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_utils import load_mesh, Renderer  # noqa: E402
from silfit import SilFitter  # noqa: E402
from traj_utils import rot_mean, smooth_poses, reanchor_depth, rot_angle_deg  # noqa: E402

# ---------------- constants (with reasons) ----------------
MOTION_PIX_THR = 15.0     # grey-level change counted as "object pixel changed" (frame-to-frame)
MOTION_FRAC_THR = 0.01    # fraction of changed object pixels below which a frame is static
MIN_STATIC_LEN = 30       # a static phase must last >= 1 s
# A "moving" blip shorter than this between two static runs, whose peak score stays below BLIP_MAX_SCORE, is
# not object motion (frames 79-81: person walking past, score <= 0.012; separate static fits before/after it
# agree within 0.7 deg / 0.7 cm laterally) -> merged into the static phase.
MIN_MOVING_LEN = 10
BLIP_MAX_SCORE = 0.03
# Verified by eye on the video (and reproduced by the automatic criterion above): static 0-201,
# dragged 202-559, static 560-718 (motion starts at 201-202, score 0.005 -> 0.033). Only used as a sanity print, not to override the automatic result.
EXPECTED_PHASES = [(0, 201, 'static'), (202, 559, 'moving'), (560, 718, 'static')]
FAT_PX = 2.5              # SAM2 masks are ~2-3 px wider than the true silhouette of the thin tubes
STATIC_SUBSAMPLE = 2      # frames used for the pooled static fit
STATIC_W_EDGE = 0.05      # edge term made the static optimum well defined (schedules agree within 0.9 deg)
MOVING_INIT_SIGMA = 3.0   # frames, temporal smoothing of the baseline before refinement
MOVING_POST_SIGMA = 1.0   # frames, light smoothing of the refined per-frame poses
MOVING_ROBUST_SIGMA = 2.0  # frames, IRLS smoothing used to initialise pass 2
MOVING_DEV_COST = 0.004   # silhouette-loss units per degree of deviation from the smoothed trajectory
RAMP_AGREE_CM = 2.0       # two depth cues must agree within ~2 cm (task rule) before deviating from the baseline


def log(*a):
    print('[solve %6.1fs]' % (time.time() - T0), *a, flush=True)


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    return frames


def motion_scores(frames, obj, per):
    F = len(frames)
    sc = np.full(F, np.nan)
    prev = cv2.GaussianBlur(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
    k9, k15 = np.ones((9, 9), np.uint8), np.ones((15, 15), np.uint8)
    for i in range(1, F):
        cur = cv2.GaussianBlur(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
        reg = cv2.dilate((obj[i] | obj[i - 1]).astype(np.uint8), k9) > 0
        reg &= ~(cv2.dilate((per[i] | per[i - 1]).astype(np.uint8), k15) > 0)
        if reg.sum() > 200:
            sc[i] = (np.abs(cur - prev)[reg] > MOTION_PIX_THR).mean()
        prev = cur
    sc[0] = sc[1]
    # frames with the object fully hidden: inherit from neighbours
    idx = np.arange(F)
    ok = ~np.isnan(sc)
    sc = np.interp(idx, idx[ok], sc[ok])
    return sc


def segment_phases(sc):
    ph = _segment(sc)
    out = []
    for k, p in enumerate(ph):
        a, b, t = p
        if (t == 'moving' and 0 < k < len(ph) - 1 and b - a + 1 < MIN_MOVING_LEN
                and sc[a:b + 1].max() < BLIP_MAX_SCORE):
            t = 'static'
        if out and out[-1][2] == t:
            out[-1] = (out[-1][0], b, t)
        else:
            out.append((a, b, t))
    return out


def _segment(sc):
    F = len(sc)
    from scipy.ndimage import median_filter
    static = median_filter(sc, size=5, mode='nearest') < MOTION_FRAC_THR
    phases, i = [], 0
    while i < F:
        j = i
        while j + 1 < F and static[j + 1] == static[i]:
            j += 1
        phases.append([i, j, 'static' if static[i] else 'moving'])
        i = j + 1
    # short static runs are not phases -> merge into moving
    for p in phases:
        if p[2] == 'static' and p[1] - p[0] + 1 < MIN_STATIC_LEN:
            p[2] = 'moving'
    merged = []
    for p in phases:
        if merged and merged[-1][2] == p[2]:
            merged[-1][1] = p[1]
        else:
            merged.append(list(p))
    return [tuple(p) for p in merged]


def _fit_chunks(fitter, obj, per, Tinit, **kw):
    T = np.zeros_like(Tinit)
    L = np.zeros(len(Tinit))
    CH = 128  # chunking keeps GPU memory low
    for s in range(0, len(Tinit), CH):
        sl = slice(s, min(len(Tinit), s + CH))
        fitter.setup(obj[sl], per[sl], Tinit[sl])
        T[sl] = fitter.fit(shared=False, fix_depth=True, **kw)
        L[sl] = fitter.score(T[sl], sigma=1.5)
    return T, L


def robust_smooth(T, c, sigma, iters=3, scale_deg=2.0):
    """IRLS temporal smoothing: frames deviating from the smooth trajectory get down-weighted."""
    w = np.ones(len(T))
    for _ in range(iters):
        S = smooth_poses(T, sigma, center=c, weights=w)
        dev = rot_angle_deg(T[:, :3, :3], S[:, :3, :3])
        w = 1.0 / (1.0 + (dev / scale_deg) ** 2)
    return S, dev


def fit_moving(fitter, obj, per, Tc_base, c):
    Tsm = smooth_poses(Tc_base, MOVING_INIT_SIGMA, center=c)
    sched = dict(iters=(80, 60, 60), sigmas=(6, 3, 1.5))
    # pass 1: every frame from the smoothed baseline
    T1, L1 = _fit_chunks(fitter, obj, per, Tsm, **sched)
    # pass 2: every frame again, initialised from the robustly smoothed pass-1 trajectory (fixes frames that
    # fell into a local minimum, e.g. 311-317 where the leg hides the ladder)
    S, dev1 = robust_smooth(T1, c, MOVING_ROBUST_SIGMA)
    T2, L2 = _fit_chunks(fitter, obj, per, S, **sched)
    dev2 = rot_angle_deg(T2[:, :3, :3], S[:, :3, :3])
    LS = frame_losses(fitter, obj, per, S)
    # per frame, 3 hypotheses (pass-1 fit, pass-2 fit, smoothed trajectory = propagation from the neighbours):
    # data cost + temporal-consistency cost; a deviation from the smooth motion must pay for itself in the image
    cost = np.stack([L1 + MOVING_DEV_COST * dev1, L2 + MOVING_DEV_COST * dev2, LS])
    pick = cost.argmin(0)
    T = np.stack([T1, T2, S])[pick, np.arange(len(T1))]
    devc = np.stack([dev1, dev2, np.zeros_like(dev1)])[pick, np.arange(len(T1))]
    log('moving: sil loss smoothed baseline %.4f, pass1 %.4f, pass2 %.4f, smooth %.4f, chosen %.4f; '
        'frames from pass1/pass2/smooth: %d/%d/%d; outliers (>4 deg) pass1 %d -> chosen %d (smooth chosen at %d frames)' % (
            frame_losses(fitter, obj, per, Tsm).mean(), L1.mean(), L2.mean(), LS.mean(),
            cost.min(0).mean(), (pick == 0).sum(), (pick == 1).sum(), (pick == 2).sum(),
            (dev1 > 4).sum(), (devc > 4).sum(), (pick == 2).sum()))
    return smooth_poses(T, MOVING_POST_SIGMA, center=c)


def frame_losses(fitter, obj, per, T):
    out = []
    for s in range(0, len(T), 128):
        sl = slice(s, min(len(T), s + 128))
        fitter.setup(obj[sl], per[sl], T[sl])
        out.append(fitter.score(T[sl], sigma=1.5))
    return np.concatenate(out)


def centre_z(T, c):
    return (T[..., :3, :3] @ c + T[..., :3, 3])[..., 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    args = ap.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    inp = args.inputs
    cam = json.load(open(os.path.join(inp, 'camera.json')))
    K = np.array(cam['K'], np.float64)
    Twc = np.array(cam['T_world_cam'], np.float64)
    Tcw = np.linalg.inv(Twc)
    rec = np.load(os.path.join(inp, 'reconstruction.npz'))
    Pw_base = rec['T_world_mesh'].astype(np.float64)
    F = len(Pw_base)
    Tc_base = Tcw[None] @ Pw_base
    mesh, tex = load_mesh(os.path.join(inp, 'object_mesh.glb'))
    R = Renderer(mesh, tex, K)
    c = R.v.mean(0).double().cpu().numpy()     # rotation centre (mesh coords)
    with h5py.File(os.path.join(inp, 'masks.h5'), 'r') as f:
        obj = f['object'][:]
        per = f['person'][:]
    frames = read_video(os.path.join(inp, 'video.mp4'))
    assert len(frames) == F, (len(frames), F)
    log('loaded', F, 'frames')

    # ---- 1. phases ----
    sc = motion_scores(frames, obj, per)
    phases = segment_phases(sc)
    log('phases:', phases)
    if [(a, b, t) for a, b, t in phases] != EXPECTED_PHASES:
        log('WARNING: automatic phases differ from the ones verified by eye', EXPECTED_PHASES)

    Tc_out = Tc_base.copy()
    fitter = SilFitter(R, K, fat_px=FAT_PX)
    info = {'phases': phases, 'static': {}}

    for (a, b, typ) in phases:
        idx = np.arange(a, b + 1)
        if typ == 'static':
            # ---- 2. one pose per static phase ----
            rm = rot_mean(Tc_base[idx, :3, :3])
            pc = np.median(np.einsum('fij,j->fi', Tc_base[idx, :3, :3], c) + Tc_base[idx, :3, 3], 0)
            Tm = np.eye(4)
            Tm[:3, :3] = rm
            Tm[:3, 3] = pc - rm @ c
            sub = idx[::STATIC_SUBSAMPLE]
            fitter.setup(obj[sub], per[sub], np.repeat(Tm[None], len(sub), 0), imgs=[frames[i] for i in sub])
            cands = []
            for kw in [dict(iters=(60, 60, 60), sigmas=(4, 2, 1)),
                       dict(iters=(80, 60, 60), sigmas=(6, 3, 1.5)),
                       dict(iters=(100, 100, 100, 100), sigmas=(6, 3, 1.5, 0.75), lr_r=0.005, lr_t=0.003)]:
                cands.append(fitter.fit(shared=True, fix_depth=True, w_edge=STATIC_W_EDGE, **kw)[0])
            cands = np.array(cands)
            Rs = rot_mean(cands[:, :3, :3])
            ps = (np.einsum('kij,j->ki', cands[:, :3, :3], c) + cands[:, :3, 3]).mean(0)
            Ts = np.eye(4)
            Ts[:3, :3] = Rs
            Ts[:3, 3] = ps - Rs @ c
            s0 = fitter.score(np.repeat(Tm[None], len(sub), 0)).mean()
            s1 = fitter.score(np.repeat(Ts[None], len(sub), 0)).mean()
            log('static %d-%d: sil loss mean-pose %.4f -> fitted %.4f' % (a, b, s0, s1))
            Tc_out[idx] = Ts[None]
            info['static'][(a, b)] = dict(T_mean=Tm, T_fit=Ts, cands=cands)
        else:
            # ---- 3. moving phase: per-frame refinement around the smoothed baseline ----
            Tc_out[idx] = fit_moving(fitter, obj[idx], per[idx], Tc_base[idx], c)

    # ---- 4. depth re-anchoring per phase (keep baseline median camera z, move along the rays) ----
    for (a, b, typ) in phases:
        idx = np.arange(a, b + 1)
        Tc_out[idx], s = reanchor_depth(Tc_out[idx], Tc_base[idx])
        log('phase %d-%d (%s): depth re-anchor scale %.4f' % (a, b, typ, s))

    # ---- 5. continuity with the static anchors (moving phases only) ----
    # The baseline depth drifts slowly (17 cm inside the last static phase), so the per-phase medians leave
    # non-physical depth jumps at the phase boundaries (object barely moving there). Ramp the along-ray depth
    # scale linearly across a moving phase so that its ends meet the neighbouring static poses. Applied only if
    # the independent floor-contact cue agrees (per third of the phase) within RAMP_AGREE_CM.
    pl = np.array(json.load(open(os.path.join(inp, 'ground_plane.json')))['plane'], np.float64)
    n_c = Twc[:3, :3].T @ pl[:3]
    d_c = pl[3] + pl[:3] @ Twc[:3, 3]
    Vs = np.asarray(mesh.vertices)[::10]
    static_idx = np.concatenate([np.arange(a, b + 1) for a, b, t in phases if t == 'static'] or [np.zeros(0, int)])

    def lowest(Tc):  # lowest mesh point above the floor plane (m); linear in a depth scale k: m + k*q + d_c
        m = (np.einsum('fij,nj->fni', Tc[:, :3, :3], Vs) @ n_c).min(1)
        return m, Tc[:, :3, 3] @ n_c

    info['ramp'] = []
    for k, (a, b, typ) in enumerate(phases):
        if typ != 'moving' or len(static_idx) == 0:
            continue
        idx = np.arange(a, b + 1)
        zc = centre_z(Tc_out[idx], c)
        r0 = centre_z(Tc_out[a - 1], c) / zc[0] if k > 0 and phases[k - 1][2] == 'static' else None
        r1 = centre_z(Tc_out[b + 1], c) / zc[-1] if k + 1 < len(phases) and phases[k + 1][2] == 'static' else None
        r0 = r1 if r0 is None else r0
        r1 = r0 if r1 is None else r1
        ramp = r0 + (r1 - r0) * np.linspace(0, 1, len(idx))
        # floor cue: depth scale making the median lowest point equal to that of the static phases
        ms, qs = lowest(Tc_out[static_idx])
        target = np.median(ms + qs + d_c)
        m, q = lowest(Tc_out[idx])
        agree = True
        for part in np.array_split(np.arange(len(idx)), 3):
            ks = np.linspace(0.9, 1.1, 2001)
            hm = np.array([np.median(m[part] + kk * q[part] + d_c) for kk in ks])
            kf = ks[np.argmin(np.abs(hm - target))]
            zmed = np.median(Tc_out[idx[part], 2, 3])
            sh_floor, sh_ramp = (kf - 1) * zmed * 100, np.median((ramp[part] - 1) * Tc_out[idx[part], 2, 3]) * 100
            dis = abs(sh_floor - sh_ramp)
            # agree within ~2 cm and correction larger than the disagreement (or negligible correction)
            agree &= bool(dis < RAMP_AGREE_CM and abs(sh_ramp) > dis) or abs(sh_ramp) < 1.0
            log('moving %d-%d frames %d-%d: depth shift floor cue %+.2f cm, anchor-continuity ramp %+.2f cm' % (
                a, b, idx[part[0]], idx[part[-1]], sh_floor, sh_ramp))
            info['ramp'].append((idx[part[0]], idx[part[-1]], sh_floor, sh_ramp))
        if agree:
            Tc_out[idx, :3, 3] *= ramp[:, None]
            log('moving %d-%d: anchor ramp applied (scale %.4f -> %.4f)' % (a, b, r0, r1))
        else:
            log('moving %d-%d: cues disagree, ramp NOT applied' % (a, b))

    Pw = Twc[None] @ Tc_out
    Pw[:, 3] = [0, 0, 0, 1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Pw.astype(np.float64),
             phases=np.array([(a, b, 1 if t == 'static' else 0) for a, b, t in phases]), motion_score=sc)
    log('wrote', args.out, Pw.shape)


if __name__ == '__main__':
    T0 = time.time()
    main()
