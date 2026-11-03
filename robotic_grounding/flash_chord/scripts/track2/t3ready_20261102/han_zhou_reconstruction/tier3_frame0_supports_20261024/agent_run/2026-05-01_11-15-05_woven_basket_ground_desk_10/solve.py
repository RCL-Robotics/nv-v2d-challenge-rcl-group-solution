#!/usr/bin/env python
"""Trajectory-level re-estimation of the object pose for one monocular HOI video.

Pipeline (see REPORT.md for the reasoning and validation):
  1. phase segmentation from the SAM2 object masks: static-on-floor / carried / static-on-desk;
  2. static phases: ONE pose per phase (upright w.r.t. the released floor normal, yaw = circular mean
     of the baseline, image position from a pooled silhouette fit, depth = baseline median depth);
  3. carried phase: per-frame yaw hypotheses (36 bins about the gravity axis, tilt from the smoothed
     baseline), each scored by silhouette IoU (after 2-D alignment) and by colour agreement with a
     per-point colour model of the object built from the video itself; Viterbi decoding with a
     smoothness prior anchored to the two static poses; second pass with a model rebuilt from the
     decoded trajectory; fine yaw search, temporal smoothing, silhouette-aligned translation;
  4. depth re-anchoring: per phase the camera-frame positions are moved along the viewing rays so that
     the median camera depth equals the baseline's median depth over the same frames.
Deterministic (fixed seeds, no sampling at run time other than seeded surface sampling).
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
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import (TexMesh, YawFrame, crop_K, crop_box_from_mask, fit_silhouettes, model_corr,  # noqa: E402
                    resample_crop, sample_point_colors, surface_points)

CROP_R = 160            # silhouette crops are resampled to 160x160 (crop = 2.4x the mask bbox, so that
                        # hypotheses extending behind the person stay inside the crop and pay OCC_PEN)
STATIC_IOU_THR = 0.93   # mask IoU with the phase reference mask above which the object is "at rest"
N_YAW = 36              # yaw hypotheses per carried frame (10 deg bins)
W_IOU, W_PHOTO = 10.0, 1.0   # Viterbi data weights
W_SMOOTH, W_TV = 0.05, 0.15   # transition cost per frame: W_SMOOTH*(dpsi/10deg)^2 + W_TV*|dpsi in deg|. The L1 (total
                              # variation) term removes temporary yaw excursions in frames where the basket is half
                              # hidden behind the hands (verified by eye at frames 236-248: the basket does not turn).
                              # Checked: decoding changes by <20deg on every frame for W_PHOTO in {0,1,2},
                              # W_IOU in {10,20}, W_SMOOTH in {0.05,0.2}, W_TV in {0.1,0.15,0.25}.
OCC_PEN = 0.5          # rendered pixels hidden behind the person count as half a mismatch when scoring yaw
MIN_VIS_IOU = 0.3       # frames whose best hypothesis IoU is below this carry no image evidence


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circmean(a):
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    return frames


def masked_iou(m, ref, person):
    vis = ~(person & ~m)
    return (m & ref).sum() / max(1, ((m | ref) & vis).sum())


def segment_phases(obj, per):
    """Static floor phase [0, a), carried [a, b), static desk phase [b, F) from mask stability."""
    F = obj.shape[0]
    ref0 = obj[:30].mean(0) > 0.5
    ref1 = obj[F - 30:].mean(0) > 0.5
    a = 0
    while a < F and masked_iou(obj[a], ref0, per[a]) >= STATIC_IOU_THR:
        a += 1
    b = F
    while b > a and masked_iou(obj[b - 1], ref1, per[b - 1]) >= STATIC_IOU_THR:
        b -= 1
    return a, b


def viterbi(U, psis, psi_start, psi_end, lam, lam1):
    T, nb = U.shape
    D = np.degrees(np.abs(wrap(psis[:, None] - psis[None, :])))
    trans = lam * D ** 2 / 100.0 + lam1 * D
    e0 = np.degrees(np.abs(wrap(psis - psi_start)))
    e1 = np.degrees(np.abs(wrap(psis - psi_end)))
    cost = U[0] + lam * e0 ** 2 / 100.0 + lam1 * e0
    bp = np.zeros((T, nb), int)
    for t in range(1, T):
        c = cost[:, None] + trans
        bp[t] = c.argmin(0)
        cost = c.min(0) + U[t]
    cost = cost + lam * e1 ** 2 / 100.0 + lam1 * e1
    k = int(cost.argmin())
    path = [k]
    for t in range(T - 1, 0, -1):
        k = bp[t, k]
        path.append(k)
    return np.array(path[::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--debug_dir", default=None, help="optional: dump intermediate arrays here")
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    inp = args.inputs

    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], float)
    Twc = np.array(cam["T_world_cam"], float)
    Tcw = np.linalg.inv(Twc)
    plane = json.load(open(f"{inp}/ground_plane.json"))["plane"]
    nrm = np.array(plane[:3], float)
    down = -nrm / np.linalg.norm(nrm)  # world y is down; the floor normal points up
    rec = np.load(f"{inp}/reconstruction.npz")
    Tw_b = rec["T_world_mesh"].astype(np.float64)
    F = len(Tw_b)
    Tc_b = Tcw[None] @ Tw_b
    mesh = TexMesh(f"{inp}/object_mesh.glb")
    yf = YawFrame(down)   # mesh +y is the basket's bottom direction (verified on the static phases)

    print("[1] reading video and masks", flush=True)
    frames = read_video(f"{inp}/video.mp4")
    assert len(frames) == F, (len(frames), F)
    with h5py.File(f"{inp}/masks.h5", "r") as h:
        obj = h["object"][:].astype(bool)
        per = h["person"][:].astype(bool)

    # silhouette crops (128x128, with person-only pixels marked invalid)
    boxes = np.zeros((F, 3))
    Kc = np.zeros((F, 3, 3))
    obs = np.zeros((F, CROP_R, CROP_R), np.float32)
    val = np.zeros((F, CROP_R, CROP_R), np.float32)
    for i in range(F):
        p = K @ Tc_b[i, :3, 3]
        box = crop_box_from_mask(obj[i], p[:2] / p[2], min_size=96, scale=2.4)
        boxes[i] = box
        Kc[i] = crop_K(K, box, CROP_R)
        o = resample_crop(obj[i].astype(np.float32), box, CROP_R)
        pr = resample_crop(per[i].astype(np.float32), box, CROP_R)
        obs[i] = o
        val[i] = 1 - np.clip(pr - o, 0, 1)

    # blurred colour crops around the object, for the photometric model
    pbox = [None] * F
    blur = [None] * F
    for i in range(F):
        ys, xs = np.nonzero(obj[i])
        if len(xs) < 30:
            continue
        x0, y0 = max(0, xs.min() - 24), max(0, ys.min() - 24)
        x1, y1 = min(obj.shape[2], xs.max() + 25), min(obj.shape[1], ys.max() + 25)
        pbox[i] = (x0, y0, x1, y1)
        blur[i] = cv2.GaussianBlur(frames[i][y0:y1, x0:x1].astype(np.float32) / 255.0, (0, 0), 1.0)
    del frames

    # ------------------------------------------------------------------ phases
    fa, fb = segment_phases(obj, per)
    print(f"[2] phases: floor-static [0,{fa}), carried [{fa},{fb}), desk-static [{fb},{F})", flush=True)

    dec = [yf.decompose(Tw_b[i, :3, :3]) for i in range(F)]
    psi_b = np.array([d[1] for d in dec])
    dvec = np.array([d[2] for d in dec])

    # ------------------------------------------------------------------ static poses
    def static_pose(a, b):
        psi = circmean(psi_b[a:b])
        R = Tcw[:3, :3] @ yf.compose(down, psi)
        tc = np.median(Tc_b[a:b, :3, 3], axis=0)
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = tc
        # pooled silhouette fit of the image position only (rotation, depth fixed)
        fr = np.arange(a, b, 3)
        T, I = fit_silhouettes(mesh, Kc[fr], obs[fr], val[fr], np.repeat(T0[None], len(fr), 0),
                               iters=60, lr_r=0.0, fix_z=True)
        good = I > np.percentile(I, 25)
        uv = []
        for k in np.nonzero(good)[0]:
            p = K @ T[k, :3, 3]
            uv.append(p[:2] / p[2])
        u, v = np.median(np.array(uv), axis=0)
        z = tc[2]
        T0[:3, 3] = z * np.linalg.inv(K) @ np.array([u, v, 1.0])
        return T0, psi, float(np.mean(I))

    T_floor, psi_floor, iou_f = static_pose(0, fa)
    T_desk, psi_desk, iou_d = static_pose(fb, F)
    print(f"    floor yaw {np.degrees(psi_floor):.1f} (mean IoU {iou_f:.3f}), desk yaw {np.degrees(psi_desk):.1f} "
          f"(mean IoU {iou_d:.3f})", flush=True)

    Tout = np.zeros((F, 4, 4))
    Tout[:fa] = T_floor
    Tout[fb:] = T_desk

    # ------------------------------------------------------------------ colour model
    P_np, N_np = surface_points(mesh, 20000, seed=0)
    P = torch.tensor(P_np, device="cuda")
    N = torch.tensor(N_np, device="cuda")

    def sample(i, Th):
        """colours / visibility of the surface points for hypotheses Th (B,4,4) in frame i"""
        return sample_point_colors(mesh, P, N, torch.tensor(Th, dtype=torch.float32, device="cuda"), K, blur[i],
                                   obj[i], per[i], pbox[i])

    def build_model(frame_list, Tcam):
        S = torch.zeros(len(P_np), 3, device="cuda")
        C = torch.zeros(len(P_np), device="cuda")
        for i in frame_list:
            if pbox[i] is None:
                continue
            col, vis = sample(i, Tcam[i][None])
            if vis.sum() < 40:
                continue
            S += col[0] * vis[0, :, None]
            C += vis[0].float()
        return S / C.clamp(min=1)[:, None], C

    static_frames = list(range(0, fa, 2)) + list(range(fb, F, 2))
    model, mcnt = build_model(static_frames, Tout)
    print(f"[3] colour model from static phases: {(mcnt >= 2).sum().item()} points", flush=True)

    # ------------------------------------------------------------------ carried phase: yaw scan
    car = np.arange(fa, fb)
    psis = np.arange(N_YAW) * 2 * np.pi / N_YAW
    dsm = gaussian_filter1d(dvec, 3, axis=0)
    dsm /= np.linalg.norm(dsm, axis=1, keepdims=True)
    tw_b = gaussian_filter1d(Tw_b[:, :3, 3], 2, axis=0)

    def hyp_poses(i, psi_list):
        Th = np.zeros((len(psi_list), 4, 4))
        for k, ps in enumerate(psi_list):
            Twm = np.eye(4)
            Twm[:3, :3] = yf.compose(dsm[i], ps)
            Twm[:3, 3] = tw_b[i]
            Th[k] = Tcw @ Twm
        return Th

    def score_frames(fr_idx, psi_lists, model, mcnt, iters=40):
        """returns aligned hypothesis poses, IoU and photo scores"""
        nh = len(psi_lists[0])
        Th_all = np.concatenate([hyp_poses(i, pl) for i, pl in zip(fr_idx, psi_lists)])
        rep = lambda a: np.repeat(a[fr_idx], nh, 0)  # noqa: E731
        Th_fit, iou = fit_silhouettes(mesh, rep(Kc), rep(obs), rep(val), Th_all, iters=iters, lr_r=0.0,
                                      lr_uv=0.8, fix_z=True, occ_pen=OCC_PEN)
        Th_fit = Th_fit.reshape(len(fr_idx), nh, 4, 4)
        iou = iou.reshape(len(fr_idx), nh)
        ph = photo_scores(fr_idx, Th_fit, model, mcnt)
        return Th_fit, iou, ph

    def photo_scores(fr_idx, Th_fit, model, mcnt):
        ph = np.full(Th_fit.shape[:2], np.nan)
        with torch.no_grad():
            for j, i in enumerate(fr_idx):
                if pbox[i] is None:
                    continue
                col, vis = sample(i, Th_fit[j])
                c, _ = model_corr(col, vis, model, mcnt)
                ph[j] = c.cpu().numpy()
        return ph

    def unary(iou, ph):
        vis = np.nanmax(iou, 1) > MIN_VIS_IOU
        irel = iou - np.nanmax(iou, 1, keepdims=True)
        irel[~vis] = 0
        p = np.nan_to_num(ph, nan=0.0)
        p[~vis] = 0
        return -W_IOU * irel - W_PHOTO * p

    print(f"[4] yaw scan on {len(car)} carried frames x {N_YAW} hypotheses", flush=True)
    Th_fit, IOU, PH = score_frames(car, [psis] * len(car), model, mcnt)
    path = viterbi(unary(IOU, PH), psis, psi_floor, psi_desk, W_SMOOTH, W_TV)

    # second pass: colour model rebuilt from static phases + confidently decoded carried frames
    Tdec = Tout.copy()
    conf = []
    for j, i in enumerate(car):
        Tdec[i] = Th_fit[j, path[j]]
        if IOU[j, path[j]] > 0.7:
            conf.append(i)
    model2, mcnt2 = build_model(static_frames + conf[::2], Tdec)
    PH2 = photo_scores(car, Th_fit, model2, mcnt2)
    path2 = viterbi(unary(IOU, PH2), psis, psi_floor, psi_desk, W_SMOOTH, W_TV)
    nchg = int((np.abs(wrap(psis[path2] - psis[path])) > np.radians(45)).sum())
    print(f"    pass-2 model {(mcnt2 >= 2).sum().item()} points; frames changed by >45deg: {nchg}", flush=True)
    psi_dec = psis[path2]

    # ------------------------------------------------------------------ fine yaw + smoothing
    offs = np.radians(np.arange(-12, 13, 3))
    Th_f, IOU_f, PH_f = score_frames(car, [psi_dec[j] + offs for j in range(len(car))], model2, mcnt2)
    sc = -unary(IOU_f, PH_f)
    best = sc.argmax(1)
    vis = np.nanmax(IOU_f, 1) > MIN_VIS_IOU
    psi_fine = psi_dec + offs[best]
    psi_fine[~vis] = psi_dec[~vis]
    # include the static anchors at both ends, unwrap, robust smoothing
    seq = np.concatenate([[psi_floor] * 5, psi_fine, [psi_desk] * 5])
    seq = np.unwrap(seq)
    # frames without image evidence: linear interpolation between neighbours
    idx = np.arange(len(seq))
    good = np.concatenate([[True] * 5, vis, [True] * 5])
    seq = np.interp(idx, idx[good], seq[good])
    seq = median_filter(seq, size=5, mode="nearest")
    seq = gaussian_filter1d(seq, 2.0, mode="nearest")
    psi_s = seq[5:-5]

    # ------------------------------------------------------------------ translation in the carried phase
    Th0 = np.stack([hyp_poses(i, [psi_s[j]])[0] for j, i in enumerate(car)])
    Tfit, Ifit = fit_silhouettes(mesh, Kc[car], obs[car], val[car], Th0, iters=60, lr_r=0.0, fix_z=True)
    uv = np.zeros((len(car), 2))
    z = np.zeros(len(car))
    for j, i in enumerate(car):
        src = Tfit[j] if Ifit[j] > 0.5 else Th0[j]   # weak evidence -> keep the (smoothed) baseline position
        p = K @ src[:3, 3]
        uv[j] = p[:2] / p[2]
        z[j] = Th0[j, 2, 3]
    # anchor the ends to the static image positions/depths for continuity, then smooth
    def endpoints(T):
        p = K @ T[:3, 3]
        return p[:2] / p[2], T[2, 3]
    (uv_f, z_f), (uv_d, z_d) = endpoints(T_floor), endpoints(T_desk)
    uv_seq = np.concatenate([[uv_f] * 5, uv, [uv_d] * 5])
    z_seq = np.concatenate([[z_f] * 5, z, [z_d] * 5])
    uv_seq = gaussian_filter1d(median_filter(uv_seq, size=(5, 1), mode="nearest"), 1.5, axis=0, mode="nearest")
    z_seq = gaussian_filter1d(median_filter(z_seq, size=5, mode="nearest"), 2.0, mode="nearest")
    uv, z = uv_seq[5:-5], z_seq[5:-5]
    Kinv = np.linalg.inv(K)
    for j, i in enumerate(car):
        T = np.eye(4)
        T[:3, :3] = Tcw[:3, :3] @ yf.compose(dsm[i], psi_s[j])
        T[:3, 3] = z[j] * Kinv @ np.array([uv[j, 0], uv[j, 1], 1.0])
        Tout[i] = T

    # ------------------------------------------------------------------ depth re-anchoring per phase
    for a, b in [(0, fa), (fa, fb), (fb, F)]:
        if b <= a:
            continue
        s = np.median(Tc_b[a:b, 2, 3]) / np.median(Tout[a:b, 2, 3])
        Tout[a:b, :3, 3] *= s   # scaling along the viewing ray keeps the image position
        print(f"    re-anchor [{a},{b}): depth scale {s:.4f}", flush=True)

    T_world = Twc[None] @ Tout
    T_world[:, 3] = [0, 0, 0, 1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        np.savez(f"{args.debug_dir}/solve_debug.npz", IOU=IOU, PH=PH, PH2=PH2, path=path, path2=path2, psis=psis,
                 psi_s=psi_s, psi_b=psi_b, phases=np.array([fa, fb]), Ifit=Ifit)
    print(f"[done] wrote {args.out} ({F} frames) in {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
