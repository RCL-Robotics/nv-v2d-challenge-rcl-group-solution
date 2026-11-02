#!/usr/bin/env python
"""Recover the 6-DoF trajectory of the rice cooker in inputs/video.mp4.

Method (see REPORT.md):
  1. Phase segmentation from the video: frame-to-frame change of the visible (not person-covered) object pixels
     -> static phases (one pose each) and moving phases (one pose per frame).
  2. Physical model: the object stays upright on the table (mesh -z axis == floor normal; the baseline deviates
     from this by <~1.5 deg), so a pose is (yaw theta, image position u,v of the mesh origin, depth z).
  3. Joint render-and-compare optimisation of theta,u,v over ALL frames (nvdiffrast, textured mesh): silhouette IoU
     with the SAM2 object mask (person pixels excluded) + NCC of band-pass filtered colour, with second-order
     temporal smoothness; static phases share one parameter set (evidence pooled over hundreds of frames).
  4. Relative depth profile from two cues: (a) silhouette scale (per-frame 1-D IoU depth sweep), (b) table-plane
     constraint (the object's bottom stays at a constant height above the released floor plane, so its image
     position fixes the depth). The two cues are aligned on the static phases, averaged and smoothed; static
     phases are constant.
  5. Re-anchoring: camera-frame positions are scaled along the viewing rays so that the episode median depth
     equals the baseline's episode median depth (keeps the baseline's calibrated overall placement).
  6. Second optimisation pass of theta,u,v at the final depths; write T_world_mesh.
Deterministic: fixed seeds, deterministic data order, no randomness in the optimisation.
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
from hoi_lib import DiffFitter, Renderer, load_camera, load_mesh, rotvec_to_R, rotz_t  # noqa: E402
from traj_opt import TrajectoryOptimizer, segment_phases  # noqa: E402

CROP_W, CROP_H = 128, 112


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def upright_basis(R0, nup):
    """Minimal rotation of R0 such that mesh -z maps onto the floor normal (world up)."""
    a = R0 @ np.array([0, 0, -1.0])
    ax = np.cross(a, nup); s = np.linalg.norm(ax); c = a @ nup
    Rc = rotvec_to_R(ax / s * np.arctan2(s, c)) if s > 1e-9 else np.eye(3)
    return Rc @ R0


def yaw_of(B, Rwm):
    M = np.einsum("ji,fjk->fik", B, Rwm)
    return np.arctan2(M[:, 1, 0], M[:, 0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--iters1", type=int, default=120)
    ap.add_argument("--iters2", type=int, default=80)
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    inp = args.inputs

    # ---------------- inputs ----------------
    K, Twc, K_pipe, W, H = load_camera(inp)
    Tcw = np.linalg.inv(Twc)
    Rcw = Tcw[:3, :3]
    plane = json.load(open(f"{inp}/ground_plane.json"))["plane"]
    nup = np.array(plane[:3]); d_pl = plane[3] / np.linalg.norm(nup); nup = nup / np.linalg.norm(nup)
    if d_pl < 0:  # make nup point from the floor towards the camera (world origin)
        nup, d_pl = -nup, -d_pl
    rec = np.load(f"{inp}/reconstruction.npz")
    Twm_base = rec["T_world_mesh"].astype(np.float64)
    F = len(Twm_base)
    Tcm_base = Tcw[None] @ Twm_base
    v, f, uv, tex = load_mesh(f"{inp}/object_mesh.glb")
    Rd = Renderer(v, f, uv, tex)

    # crop window around the object (baseline projections; the object barely moves in the image here)
    o = Tcm_base[:, :3, 3]; pu = o @ K.T; pu = pu[:, :2] / pu[:, 2:]
    cu, cv_ = np.median(pu, 0)
    x0 = int(round(cu - CROP_W / 2)); y0 = int(round(cv_ - CROP_H / 2))
    crop = (x0, y0, CROP_W, CROP_H)
    assert pu[:, 0].min() > x0 + 30 and pu[:, 0].max() < x0 + CROP_W - 30, "object leaves crop window"
    log("crop", crop)

    # video frames and masks (full-size arrays, only the crop is used)
    cap = cv2.VideoCapture(f"{inp}/video.mp4")
    imgs = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        full = np.zeros((H, W, 3), np.uint8)
        full[y0:y0 + CROP_H, x0:x0 + CROP_W] = fr[y0:y0 + CROP_H, x0:x0 + CROP_W]
        imgs.append(full)
    assert len(imgs) == F, (len(imgs), F)
    with h5py.File(f"{inp}/masks.h5", "r") as hf:
        om_c = hf["object"][:, y0:y0 + CROP_H, x0:x0 + CROP_W]
        pm_c = hf["person"][:, y0:y0 + CROP_H, x0:x0 + CROP_W]

    def pad(m):
        full = np.zeros((H, W), bool); full[y0:y0 + CROP_H, x0:x0 + CROP_W] = m; return full
    oms = [pad(m) for m in om_c]; pms = [pad(m) for m in pm_c]
    log("loaded", F, "frames")

    # ---------------- 1. phase segmentation ----------------
    d1 = np.zeros(F)
    prev = None
    for i in range(F):
        im = cv2.GaussianBlur(imgs[i][y0:y0 + CROP_H, x0:x0 + CROP_W].astype(np.float32), (5, 5), 1)
        pmd = cv2.dilate(pm_c[i].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        val = om_c[i] & ~pmd
        if prev is not None and val.sum() > 200:
            d1[i] = np.abs(im - prev).mean(-1)[val].mean()
        prev = im
    phases = segment_phases(d1)
    log("phases (start, end, moving):", phases)

    # ---------------- 2. upright model, initialisation from the baseline ----------------
    B = upright_basis(Twm_base[0, :3, :3], nup)
    yaw_base = np.unwrap(yaw_of(B, Twm_base[:, :3, :3]))
    tilt = np.degrees(np.arccos(np.clip(np.einsum("fij,j->fi", Twm_base[:, :3, :3], [0, 0, -1.0]) @ nup, -1, 1)))
    log(f"baseline tilt from upright: median {np.median(tilt):.2f} deg, max {tilt.max():.2f} deg")
    init_uvz = np.concatenate([pu, o[:, 2:]], 1)
    # moving frames: start from a robustly smoothed baseline depth (static phases use the phase median)
    init_uvz[:, 2] = gaussian_filter1d(median_filter(o[:, 2], 9, mode="nearest"), 3, mode="nearest")

    # ---------------- 3. joint optimisation, pass 1 (depth fixed) ----------------
    opt = TrajectoryOptimizer(Rd, K, Rcw, B, crop, imgs, oms, pms, phases, yaw_base, init_uvz)
    opt.optimize(iters=args.iters1, fix_z=True, log=log)
    _, th1, uvz1 = opt.result()

    # ---------------- 4a. silhouette-scale depth (per-frame 1-D IoU sweep) ----------------
    D = opt.D
    dev = "cuda"
    Bc = torch.tensor(Rcw @ B, device=dev, dtype=torch.float32)
    Kinv = np.linalg.inv(K)
    zg = np.arange(-0.15, 0.1501, 0.005)
    z_sil = np.zeros(F)
    with torch.no_grad():
        for s in range(0, F, 96):
            idx = np.arange(s, min(F, s + 96))
            R = Bc[None] @ rotz_t(torch.tensor(th1[idx], device=dev, dtype=torch.float32))
            ray = torch.tensor(np.c_[uvz1[idx, :2], np.ones(len(idx))] @ Kinv.T, device=dev, dtype=torch.float32)
            S = np.zeros((len(zg), len(idx)))
            for k, dz in enumerate(zg):
                t = ray * torch.tensor(uvz1[idx, 2] + dz, device=dev, dtype=torch.float32)[:, None]
                iou, _ = D.losses(R, t, torch.tensor(idx, device=dev), w_ncc=0)
                S[k] = iou.cpu().numpy()
            k = np.clip(S.argmax(0), 1, len(zg) - 2)
            j = np.arange(len(idx))
            y0_, y1_, y2_ = S[k - 1, j], S[k, j], S[k + 1, j]
            den = y0_ - 2 * y1_ + y2_
            off = np.where(den < 0, 0.5 * (y0_ - y2_) / np.where(den < 0, den, -1), 0)
            z_sil[idx] = uvz1[idx, 2] + zg[k] + off * 0.005

    # ---------------- 4b. table-plane depth cues ----------------
    # bottom centre of the cooker in mesh coordinates (mesh -z is up, so the bottom is at max z)
    bot_m = np.array([(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].max()])

    def pose_np(i, z):
        c, s_ = np.cos(th1[i]), np.sin(th1[i])
        Rcm_i = Rcw @ B @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        return Rcm_i, z * (Kinv @ np.r_[uvz1[i, :2], 1])

    def plane_depth(u_, v_, h):
        """camera depth of the point at pixel (u_,v_) lying at height h above the floor plane"""
        ray = Kinv @ np.array([u_, v_, 1.0]); rw = Twc[:3, :3] @ ray
        s_ = (h - d_pl - nup @ Twc[:3, 3]) / (nup @ rw)
        return s_ * ray[2]

    # (i) contact line: the dark contact shadow between the round base and the table top. Its image row at the
    # base centre column is measured to ~0.02 px in static phases. The table top is a plane parallel to the floor
    # (assumption), so the row gives the depth of the base's front point up to a constant (unknown table height,
    # base radius) that the re-anchoring removes.
    gray = [cv2.GaussianBlur(cv2.cvtColor(im[y0:y0 + CROP_H, x0:x0 + CROP_W], cv2.COLOR_BGR2GRAY).astype(np.float32),
                             (3, 3), 0.7) for im in imgs]
    v_ct = np.full(F, np.nan); u_ct = np.zeros(F)
    for i in range(F):
        Rcm_i, t_i = pose_np(i, uvz1[i, 2])
        pb = Rcm_i @ bot_m + t_i; qb = K @ pb; ub, vb = qb[:2] / qb[2]
        u_ct[i] = ub
        rows = []
        for c in range(int(round(ub)) - 6, int(round(ub)) + 7):
            lo = int(round(vb)) - 7
            p = gray[i][lo - y0:lo - y0 + 19, c - x0]
            k = int(np.argmin(p))
            if 0 < k < len(p) - 1:
                a0, a1, a2 = p[k - 1], p[k], p[k + 1]; den = a0 - 2 * a1 + a2
                rows.append(lo + k + (0.5 * (a0 - a2) / den if den > 0 else 0.0))
        if len(rows) >= 7:
            v_ct[i] = np.median(rows)
    good = np.isfinite(v_ct)
    v_ct = np.interp(np.arange(F), np.where(good)[0], v_ct[good])
    v_ct_f = median_filter(v_ct, 7, mode="nearest")
    h_ref = 0.80  # arbitrary reference height; only depth differences are used
    z_ct = np.array([plane_depth(u_ct[i], v_ct_f[i], h_ref) for i in range(F)])

    # (ii) fitted-silhouette table cue: bottom centre of the fitted pose kept at a constant height
    def bottom_height(i, z):
        Rcm_i, t_i = pose_np(i, z)
        pb = Rcm_i @ bot_m + t_i
        return nup @ (Twc[:3, :3] @ pb + Twc[:3, 3]) + d_pl
    z_ref = float(np.median(o[:, 2]))
    hh = np.median([bottom_height(i, z_ref) for i in range(F)])
    z_tab = np.zeros(F)
    for i in range(F):  # height is affine in z along the ray
        h1, h2 = bottom_height(i, z_ref - 0.05), bottom_height(i, z_ref + 0.05)
        z_tab[i] = z_ref - 0.05 + (hh - h1) * 0.1 / (h2 - h1)

    # ---------------- 4c. relative depth profile ----------------
    # primary: contact-line cue (most precise); silhouette-scale and fitted-table cues are reported as cross-checks
    static = np.zeros(F, bool)
    for a, b, mv in phases:
        if not mv:
            static[a:b + 1] = True
    z_rel = gaussian_filter1d(z_ct, 2, mode="nearest")
    for a, b, mv in phases:
        if not mv:
            z_rel[a:b + 1] = np.median(z_ct[a:b + 1])
    for a, b, mv in phases:  # continuity at static/moving boundaries
        if mv:
            left = z_rel[a - 1] if a > 0 else z_rel[a]
            right = z_rel[b + 1] if b < F - 1 else z_rel[b]
            w = np.linspace(0, 1, b - a + 3)[1:-1]
            z_rel[a:b + 1] += (1 - w) * (left - z_rel[a]) + w * (right - z_rel[b])
    ref_phase = [p for p in phases if not p[2]][0]
    cue_rel = {}
    for name, zc in (("contact", z_ct), ("silhouette", z_sil), ("fit_table", z_tab), ("baseline", o[:, 2])):
        r0 = np.median(zc[ref_phase[0]:ref_phase[1] + 1])
        cue_rel[name] = [float(100 * (np.median(zc[a:b + 1]) - r0)) for a, b, mv in phases]
        log(f"cue {name:10s}: phase depth relative to first static phase (cm): "
            + " ".join(f"{x:+.1f}" for x in cue_rel[name]))

    # ---------------- 5. re-anchor to the baseline's episode median depth ----------------
    scale = np.median(o[:, 2]) / np.median(z_rel)
    z_fin = z_rel * scale
    log(f"depth re-anchoring scale {scale:.4f} (shift {100 * (np.median(o[:, 2]) - np.median(z_rel)):.1f} cm)")
    for a, b, mv in phases:
        log(f"  phase {a}-{b} moving={mv}: final z med {np.median(z_fin[a:b+1]):.3f} "
            f"baseline z med {np.median(o[a:b+1, 2]):.3f}")

    # ---------------- 6. pass 2 at the final depth ----------------
    with torch.no_grad():
        opt.uvz0[:, 2] = torch.tensor(
            [np.median(z_fin[a:b + 1]) for (a, b, st) in opt.units], device=dev, dtype=torch.float32)
        opt.raw[:, 3] = 0
    opt.optimize(iters=args.iters2, fix_z=True, lr=(0.25, 0.08, 0.08, 0.5), log=log)
    Tcm, th, uvz = opt.result()
    Twm = Twc[None] @ Tcm
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Twm.astype(np.float64))
    log("wrote", args.out, Twm.shape)

    # ---------------- validation summary (no ground truth) ----------------
    iou_o, ncc_o = opt.eval_frames()
    with torch.no_grad():
        ib, nb = [], []
        for s in range(0, F, 96):
            idx = torch.arange(s, min(F, s + 96), device=dev)
            Rb = torch.tensor(Tcm_base[s:s + 96, :3, :3], device=dev, dtype=torch.float32)
            tb = torch.tensor(Tcm_base[s:s + 96, :3, 3], device=dev, dtype=torch.float32)
            a_, b_ = D.losses(Rb, tb, idx)
            ib.append(a_.cpu().numpy()); nb.append(b_.cpu().numpy())
    iou_b, ncc_b = np.concatenate(ib), np.concatenate(nb)

    def rstep(T):
        R = T[:, :3, :3]
        c = (np.einsum("fij,fij->f", R[1:], R[:-1]) - 1) / 2
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    rs_b, rs_o = rstep(Twm_base), rstep(Twm)
    ts_b = np.linalg.norm(np.diff(Twm_base[:, :3, 3], axis=0), axis=1)
    ts_o = np.linalg.norm(np.diff(Twm[:, :3, 3], axis=0), axis=1)
    dz = (Tcm[:, 2, 3] - Tcm_base[:, 2, 3]) * 100
    val = []
    for a, b, mv in phases:
        sl = slice(a, b + 1); sl2 = slice(a, b)
        row = dict(phase=f"{a}-{b}", moving=bool(mv), iou_base=float(iou_b[sl].mean()), iou_ours=float(iou_o[sl].mean()),
                   ncc_base=float(ncc_b[sl].mean()), ncc_ours=float(ncc_o[sl].mean()),
                   rot_step_base_deg=float(np.median(rs_b[sl2])) if b > a else 0.0,
                   rot_step_ours_deg=float(np.median(rs_o[sl2])) if b > a else 0.0,
                   trans_step_base_mm=float(1000 * np.median(ts_b[sl2])) if b > a else 0.0,
                   trans_step_ours_mm=float(1000 * np.median(ts_o[sl2])) if b > a else 0.0,
                   dz_mean_cm=float(dz[sl].mean()), dz_median_cm=float(np.median(dz[sl])),
                   yaw_diff_median_deg=float(np.degrees(np.median(th[sl] - yaw_base[sl]))))
        val.append(row)
        log(row)
    json.dump(dict(phases=[list(map(int, p[:2])) + [bool(p[2])] for p in phases], validation=val,
                   depth_scale=float(scale), runtime_s=time.time() - t_start,
                   cue_relative_depth_cm=cue_rel, contact_row=v_ct.tolist(),
                   z_sil=z_sil.tolist(), z_tab=z_tab.tolist(), z_contact=z_ct.tolist(), z_final=z_fin.tolist()),
              open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "solve_validation.json"), "w"), indent=1)
    log(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
