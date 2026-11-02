#!/usr/bin/env python
"""Trajectory-level re-estimation of the object pose for one HOI video.

Pipeline (see REPORT.md for the reasoning and validation):
  1. load inputs; baseline camera-frame poses  Tb = inv(T_world_cam) @ T_world_mesh
  2. phase segmentation from the video: static-A (untouched / resting) | moving | static-B,
     using the photometric change of the object region w.r.t. reference frames (person pixels excluded)
  3. per-frame occlusion masks: a person pixel only "hides" the object if the reconstructed human is in front
     of the object there (splatted human vertices vs rendered object depth)
  4. per-frame silhouette render-and-compare (nvdiffrast) from the baseline, two mask models
     (raw render, and render dilated by 1 px to mimic SAM2's fattening of the thin legs / foot-rest wires)
  5. static phases: one pose per phase fitted jointly to all frames of the phase
     moving phase: robust temporal smoothing of the per-frame fits, glued to the static poses at the boundaries
  5b. depth (done before smoothing so that static and moving poses share one depth scale): keep the
     baseline placement -> each pose moved along its viewing ray so that z follows the baseline
     (static phases: median baseline z of the phase; moving: fit z * low-pass(baseline z / fit z))
  7. map back to the world, write T_world_mesh (F,4,4)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import (Renderer, load_inputs, read_video, render_sil, so3_exp, rotdeg, mean_rotation,
                    robust_mean_pose, rescale_depth)

torch.manual_seed(0)
np.random.seed(0)

SCALE = 0.5            # working resolution for render-and-compare (768x576)
FIT_ITERS = 200
CHUNK = 32             # frames per GPU batch (keeps GPU memory well below 9 GB)
W_EDGE = 0.3           # weight of the Canny-edge distance term (chosen on 16 frames by full-res edge chamfer)
SIGMA_T = 4.0          # temporal smoothing (frames) of the moving phase
SIGMA_Z = 30.0         # low-pass (frames) of the baseline/fit depth ratio


# ----------------------------------------------------------------------------------------------- data
def load_masks(inp, F):
    import h5py
    f = h5py.File(f"{inp}/masks.h5", "r")
    h, w = int(1152 * SCALE), int(1536 * SCALE)
    om = np.zeros((F, h, w), np.float32)
    pm = np.zeros((F, h, w), bool)
    pfull = np.zeros((F, h, w), bool)
    for i in range(0, F, 50):
        o = f["object"][i:i + 50]
        p = f["person"][i:i + 50]
        n = len(o)
        om[i:i + n] = o.reshape(n, h, 2, w, 2).mean((2, 4))
        pm[i:i + n] = p.reshape(n, h, 2, w, 2).any((2, 4))
    return om, pm


# ------------------------------------------------------------------------------------- phase detection
def detect_phases(video, om, pm, k=6, thr=2.5, persist=8):
    """Return (a_end, b_start): frames [0,a_end] static A, [b_start,F-1] static B.

    Short-term photometric change of the object region: d[i] = mean |I_i - I_{i+k}| over the (dilated) object
    masks of both frames, person pixels (dilated) excluded. While the object rests, d stays at the noise
    floor (1.0-2.3 grey levels in this video, incl. shadows of the person walking around); while it is
    manipulated d is 3-15. A phase boundary is where d exceeds thr=2.5 for >= persist=8 consecutive frames
    (short bumps of <8 frames, e.g. hand shadows at ~253-258, are ignored). Verified by eye on this
    video: the stool starts to be lifted at ~frame 274 and is set down upright at ~frame 621.
    """
    import cv2
    F = len(video)
    d = np.zeros(F)
    ker = np.ones((7, 7), np.uint8)
    for i in range(F):
        j = i + k if i + k < F else i - k
        reg = cv2.dilate(((om[i] > 0.5) | (om[j] > 0.5)).astype(np.uint8), ker) > 0
        p = cv2.dilate((pm[i] | pm[j]).astype(np.uint8), ker) > 0
        v = reg & ~p
        d[i] = np.abs(video[i].astype(np.float32) - video[j].astype(np.float32))[v].mean() if v.sum() > 50 else 0
    bad = d > thr
    run = np.array([bad[i:i + persist].all() for i in range(F - persist + 1)])
    first = int(np.argmax(run))
    last = int(len(run) - 1 - np.argmax(run[::-1])) + persist - 1
    a_end = first + k // 2
    b_start = last + k // 2
    return a_end, b_start, d


# ---------------------------------------------------------------------------------- occlusion masks
def person_in_front(rd, Tc_obj, hum_cam, pm):
    """Boolean (B,h,w): person mask pixels where the reconstructed human is in front of the object.
    Pixels of the person mask without a human depth sample are treated as occluding (conservative)."""
    import cv2
    _, zobj, _ = rd.render(Tc_obj, need_rgb=False)
    zobj = zobj.cpu().numpy()
    out = np.zeros_like(pm)
    K = rd.K
    for b in range(len(Tc_obj)):
        P = hum_cam[b]
        P = P[P[:, 2] > 0.1]
        u = np.round(P[:, 0] / P[:, 2] * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(P[:, 1] / P[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < rd.w) & (v >= 0) & (v < rd.h)
        zb = np.full((rd.h, rd.w), np.inf, np.float32)
        np.minimum.at(zb, (v[ok], u[ok]), P[ok, 2].astype(np.float32))
        zb = cv2.erode(zb, np.ones((9, 9), np.uint8))  # min-filter: fill gaps between sparse vertices
        zo = zobj[b]
        objpix = zo > 0
        front = np.ones_like(pm[b])
        both = objpix & np.isfinite(zb)
        front[both] = zb[both] < zo[both] + 0.05
        # where the object is not rendered, a person pixel cannot hide it -> irrelevant, keep "occluding"
        out[b] = pm[b] & front
    return out


# ------------------------------------------------------------------------------------ fitting
def blur(x, k):
    if k <= 1:
        return x
    return Fn.avg_pool2d(x[:, None], k, 1, k // 2, count_include_pad=False)[:, 0]


def soft_iou(sil, tgt, valid):
    s = (sil > 0.5).float() * valid
    g = (tgt > 0.5).float() * valid
    return (s * g).sum((1, 2)) / ((s + g - s * g).sum((1, 2)) + 1e-6)


def make_T(T0, w, t):
    return torch.cat([torch.cat([so3_exp(w) @ T0[:, :3, :3], (T0[:, :3, 3] + t)[..., None]], 2), T0[:, 3:]], 1)


def edge_dt_maps(frames, occ, trunc=6.0):
    """Truncated distance transform (pixels) to Canny edges for each frame; occluded pixels set to trunc/2."""
    import cv2
    out = np.zeros(frames.shape[:3], np.float32)
    for b, im in enumerate(frames):
        g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), (3, 3), 0.8)
        e = cv2.Canny(g, 30, 80)
        dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)
        dt = np.minimum(dt, trunc)
        dt[occ[b]] = trunc / 2
        out[b] = dt
    return out


def fit_frames(rd, T_init, tgt, valid, dil, shared=False, iters=FIT_ITERS, lr=2e-3, edt=None, w_edge=0.0,
               ks=(9, 5, 3, 1)):
    """Silhouette (+ optional edge distance) render-and-compare. If shared, one pose for all frames."""
    dev = "cuda"
    tgt = torch.as_tensor(tgt, device=dev)
    valid = torch.as_tensor(valid, device=dev, dtype=torch.float32)
    if edt is not None:
        edt = torch.as_tensor(edt, device=dev)
    T0 = torch.as_tensor(T_init, device=dev, dtype=torch.float32)
    n = 1 if shared else len(T0)
    w = torch.zeros(n, 3, device=dev, requires_grad=True)
    t = torch.zeros(n, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, t], lr=lr)
    # cosine decay of the step size: the fit converges to (almost) the same point despite the tiny
    # run-to-run nondeterminism of nvdiffrast's atomic gradient accumulation
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * it / iters)))
    i0 = None
    for it in range(iters):
        T = make_T(T0, w.expand(len(T0), 3), t.expand(len(T0), 3))
        sil0, _, _ = render_sil(rd, T)
        sil = Fn.max_pool2d(sil0[:, None], 2 * dil + 1, 1, dil)[:, 0] if dil > 0 else sil0
        k = ks[min(it * len(ks) // iters, len(ks) - 1)]
        loss = ((blur(sil, k) - blur(tgt, k)) ** 2 * valid).sum((1, 2)) / 1000
        if edt is not None and w_edge > 0:
            gx = (sil0[:, :, 1:] - sil0[:, :, :-1]).abs()
            gy = (sil0[:, 1:, :] - sil0[:, :-1, :]).abs()
            le = (gx * edt[:, :, 1:]).sum((1, 2)) + (gy * edt[:, 1:, :]).sum((1, 2))
            nb = gx.sum((1, 2)) + gy.sum((1, 2)) + 1e-3
            loss = loss + w_edge * le / nb
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        sched.step()
        if it == 0:
            i0 = soft_iou(sil0.detach(), tgt, valid)
    with torch.no_grad():
        T = make_T(T0, w.expand(len(T0), 3), t.expand(len(T0), 3))
        sil, _, _ = render_sil(rd, T)
        i1 = soft_iou(sil, tgt, valid)
    return T.cpu().numpy().astype(np.float64), i0.cpu().numpy(), i1.cpu().numpy()


# ----------------------------------------------------------------------------- temporal smoothing
def smooth_poses(T, weights, sigma, anchors=None):
    """Gaussian-window robust smoothing of camera-frame poses.
    Rotations: weighted chordal mean in a window; translation: weighted mean of (x/z, y/z, log z).
    anchors: dict frame->pose that get a large weight (static-phase poses at the boundaries)."""
    F = len(T)
    out = np.zeros_like(T)
    half = int(3 * sigma)
    u = np.stack([T[:, 0, 3] / T[:, 2, 3], T[:, 1, 3] / T[:, 2, 3], np.log(T[:, 2, 3])], 1)
    for i in range(F):
        lo, hi = max(0, i - half), min(F, i + half + 1)
        g = np.exp(-0.5 * ((np.arange(lo, hi) - i) / sigma) ** 2) * weights[lo:hi]
        out[i] = np.eye(4)
        out[i, :3, :3] = mean_rotation(T[lo:hi, :3, :3], g)
        m = (g[:, None] * u[lo:hi]).sum(0) / g.sum()
        z = np.exp(m[2])
        out[i, :3, 3] = [m[0] * z, m[1] * z, z]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--cache", default=None, help="dev only: cache dir for per-frame fits")
    args = ap.parse_args()
    t_start = time.time()
    inp = args.inputs
    K, Twc, rec, mesh, plane = load_inputs(inp)
    Tcw = np.linalg.inv(Twc)
    Tw_base = rec["T_world_mesh"].astype(np.float64)
    F = len(Tw_base)
    Tb = np.einsum("ij,fjk->fik", Tcw, Tw_base)
    print(f"[load] {F} frames", flush=True)

    # ---- masks (half res) and video
    om, pm = load_masks(inp, F)
    print(f"[masks] {time.time() - t_start:.0f}s", flush=True)

    # ---- phases
    video = read_video(f"{inp}/video.mp4", scale=SCALE)
    a_end, b_start, phase_d = detect_phases(video, om, pm)
    print(f"[phases] static A: 0-{a_end}, moving: {a_end + 1}-{b_start - 1}, static B: {b_start}-{F - 1}",
          flush=True)
    phA = np.arange(0, a_end + 1)
    phM = np.arange(a_end + 1, b_start)
    phB = np.arange(b_start, F)

    # ---- occlusion-aware validity masks and edge distance maps
    rd = Renderer(mesh, K, SCALE)
    hv = rec["human_vertices_world"].astype(np.float64)
    hum_cam = np.einsum("ij,fnj->fni", Tcw[:3, :3], hv) + Tcw[:3, 3]
    valid = np.zeros((F,) + om.shape[1:], np.float32)
    edt = np.zeros((F,) + om.shape[1:], np.float32)
    for c0 in range(0, F, CHUNK):
        sl = slice(c0, min(F, c0 + CHUNK))
        occ = person_in_front(rd, Tb[sl], hum_cam[sl], pm[sl]) & (om[sl] < 0.5)
        valid[sl] = 1.0 - occ
        edt[sl] = edge_dt_maps(video[sl], occ)
    del video
    print(f"[occlusion+edges] {time.time() - t_start:.0f}s", flush=True)

    # ---- per-frame fits: silhouette + edge term, initialised at the baseline
    cache = f"{args.cache}/pf.npz" if args.cache else None
    if cache and os.path.exists(cache):
        z = np.load(cache)
        Tf, i0, i1 = z["T"], z["i0"], z["i1"]
    else:
        Tf = np.zeros_like(Tb)
        i0 = np.zeros(F)
        i1 = np.zeros(F)
        for c0 in range(0, F, CHUNK):
            sl = slice(c0, min(F, c0 + CHUNK))
            Tf[sl], i0[sl], i1[sl] = fit_frames(rd, Tb[sl], om[sl], valid[sl], 0, edt=edt[sl], w_edge=W_EDGE)
        if cache:
            np.savez(cache, T=Tf, i0=i0, i1=i1)
    print(f"[per-frame fit] IoU {i0.mean():.3f}->{i1.mean():.3f}  {time.time() - t_start:.0f}s", flush=True)

    # ---- static phases: one pose per phase, fitted jointly to (up to 48) frames spread over the phase
    # (400 its at lr 1e-3: run-to-run spread of the result < 0.1 deg; 200 its at 2e-3 was up to 1 deg)
    static = {}
    for name, ph in (("A", phA), ("B", phB)):
        init = robust_mean_pose(Tf[ph])
        sel = np.unique(np.linspace(ph[0], ph[-1], min(48, len(ph))).astype(int))
        Ts, _, _ = fit_frames(rd, np.repeat(init[None], len(sel), 0), om[sel], valid[sel], 0, shared=True,
                              edt=edt[sel], w_edge=W_EDGE, ks=(5, 3, 1), iters=400, lr=1e-3)
        static[name] = Ts[0]
        print(f"[static {name}] joint fit vs robust mean of per-frame fits: "
              f"{rotdeg(Ts[0][:3, :3], init[:3, :3]):.2f} deg; vs baseline mean "
              f"{rotdeg(Ts[0][:3, :3], robust_mean_pose(Tb[ph])[:3, :3]):.2f} deg", flush=True)

    # ---- depth: keep the baseline placement. Every pose is moved along its own viewing ray (image position
    # and rotation unchanged) so that its camera-frame depth follows the baseline:
    #   static phases: z = median baseline z of the phase;
    #   moving phase : z = z_fit * lowpass(z_baseline / z_fit) (sigma SIGMA_Z frames), i.e. the slow depth level
    #                  comes from the baseline, only the fast relative changes from the image fit.
    from scipy.ndimage import gaussian_filter1d
    Tseq = Tf.copy()
    Tseq[phA] = rescale_depth(static["A"], np.median(Tb[phA, 2, 3]))
    Tseq[phB] = rescale_depth(static["B"], np.median(Tb[phB, 2, 3]))
    if len(phM):
        qlp = gaussian_filter1d(Tb[phM, 2, 3] / Tf[phM, 2, 3], SIGMA_Z, mode="nearest")
        Tseq[phM] = rescale_depth(Tf[phM], Tf[phM, 2, 3] * qlp)

    # ---- moving phase: robust temporal smoothing, static poses act as anchors at both ends
    wts = np.clip((i1 - 0.5) / 0.3, 0.05, 1.0)
    wts[phA] = wts[phB] = 5.0
    S = smooth_poses(Tseq, wts, SIGMA_T)
    for _ in range(2):  # down-weight per-frame outliers (rotation or image position far from the smooth track)
        r = rotdeg(Tseq[:, :3, :3], S[:, :3, :3])
        px = np.linalg.norm(Tseq[:, :2, 3] / Tseq[:, 2:3, 3] - S[:, :2, 3] / S[:, 2:3, 3], axis=1) * K[0, 0]
        w2 = wts * np.where((r > 4.0) | (px > 8.0), 0.05, 1.0)
        w2[phA] = w2[phB] = 5.0
        S = smooth_poses(Tseq, w2, SIGMA_T)
    n_out = int(((r > 4.0) | (px > 8.0))[phM].sum())
    Tcam = Tseq.copy()
    Tcam[phM] = S[phM]
    print(f"[smooth] moving-phase outlier frames: {n_out}", flush=True)
    if len(phM):
        # continuous bump (zero at both phase ends) so that the moving phase's median z equals the baseline's
        from scipy.optimize import brentq
        bump = np.sin(np.pi * (phM - a_end) / float(b_start - a_end))
        zt = np.median(Tb[phM, 2, 3])
        zm = Tcam[phM, 2, 3]
        c = brentq(lambda c: np.median(zm * (1 + c * bump)) - zt, -0.2, 0.2)
        Tcam[phM] = rescale_depth(Tcam[phM], zm * (1 + c * bump))
        print(f"[depth] moving-phase median correction {c * 100:+.2f}% (peak)", flush=True)
    for name, ph in (("A", phA), ("moving", phM), ("B", phB)):
        dz = (Tcam[ph, 2, 3] - Tb[ph, 2, 3]) * 100
        print(f"[depth] {name}: z(result)-z(baseline) mean {dz.mean():+.2f} cm, median {np.median(dz):+.2f} cm",
              flush=True)

    Tw = np.einsum("ij,fjk->fik", Twc, Tcam)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    if args.cache:
        np.savez(f"{args.cache}/debug.npz", Tf=Tf, i0=i0, i1=i1, Tseq=Tseq, Tcam=Tcam, phase_d=phase_d,
                 a_end=a_end, b_start=b_start, valid_frac=valid.mean((1, 2)))
    print(f"[done] wrote {args.out}  total {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
