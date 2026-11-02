#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF pose of the blue trash can (episode 18).

Method (see REPORT.md):
  1. phases: static S1/S2/S3 and moving M1/M2 (eye-verified boundaries below, checked automatically);
  2. symmetry check: pooled texture NCC of the 4 yaw hypotheses (rotation by k*90 deg about the mesh's
     vertical axis) per phase -> keep the best (it is the baseline's for this episode);
  3. static phases: ONE pose per phase, silhouette render-and-compare fitted jointly over all frames of the phase
     (person-occluded pixels ignored);
  4. moving phases: joint silhouette fit of all frames with 2nd-order smoothness on position and rotation,
     both ends tied to the adjacent static poses;
  5. depth re-anchoring along the viewing rays: static S1/S2 -> baseline median camera depth; S3 -> baseline median
     unless two independent physical cues (calibrated metric silhouette size, floor-contact height consistent
     with S1/S2) agree on a correction; moving phases -> scale factor linearly interpolated between neighbours.
Run: python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_lib import (load_inputs, load_mesh, read_video, read_masks, Renderer, make_crops, fit_poses,  # noqa: E402
                     fit_track, render_crop, ncc_score)
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

# ---------------------------------------------------------------- constants
SCALE = 0.5            # working resolution (768x576)
CROP = (192, 256)      # crop window (w,h) at working resolution around the object mask
# Phase boundaries [start, end) verified by eye on the video and on the object-mask bounding box:
#  S1 0-251   bin untouched / hands resting on rim, bbox constant (x_min 320, y_max 488)
#  M1 252-455 grasp, lift (~27 cm), carry to the right while turning ~40 deg, set down (bottom reaches its final
#             image row at ~454; frames 446-455 still settling)
#  S2 456-658 bin standing on the floor; hand still on the rim until ~495 but no motion (bbox constant)
#  M2 659-719 grasp and slide sideways to the left on the floor (bbox x_min starts to change at ~661)
#  S3 720-857 bin standing still until the end (bbox constant from 720)
PHASES = [("S1", 0, 252), ("M1", 252, 456), ("S2", 456, 659), ("M2", 659, 720), ("S3", 720, 858)]
STATIC_STEP = 3        # frames used for the shared static fit (others are held out for validation)
W_IOU = 5.0            # weight of silhouette IoU vs texture NCC in the static texture refinement
LAM_P, LAM_R = 200.0, 20.0   # smoothness weights (position m^-2, rotation rad^-2) in moving phases
# Two independent depth cues must agree within "~2 cm" to deviate from the baseline depth. We use 2.5 cm: the
# floor cue has a ~1.5 cm depth uncertainty from the plane fit alone (1 mm of height = 2.6 mm of depth here).
CUE_AGREE_M = 0.025


def seed_all(s=0):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    args = ap.parse_args()
    seed_all(0)
    t_start = time.time()
    inp = args.inputs
    D = load_inputs(inp)
    mesh = load_mesh(f"{inp}/object_mesh.glb")
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2   # box centre in mesh coordinates
    V = np.asarray(mesh.vertices)[::10]
    Vall = np.asarray(mesh.vertices)
    bottom_c = Vall[Vall[:, 2] > Vall[:, 2].max() - 0.01].mean(0)   # centre of the bottom face (mesh +z is down)
    n_pl, d_pl = D["plane"][:3], D["plane"][3]
    T_wc = D["T_wc"]
    T_cw = np.linalg.inv(T_wc)
    Tb_w = D["rec"]["T_world_mesh"].astype(np.float64)
    Tb = T_cw[None] @ Tb_w                          # baseline, camera frame
    F = len(Tb)
    log = {}

    # ------------------------------------------------------------ data at working resolution
    OM = read_masks(f"{inp}/masks.h5", "object", SCALE)
    PM = read_masks(f"{inp}/masks.h5", "person", SCALE)
    VID = read_video(f"{inp}/video.mp4", SCALE)
    assert len(OM) == F and len(VID) == F
    BB = np.zeros((F, 4))
    for i in range(F):
        ys, xs = np.nonzero(OM[i])
        if len(xs) == 0:   # fall back to the projected baseline centre
            p = Tb[i, :3, :3] @ center + Tb[i, :3, 3]
            u, v = (D["K"][:2, :2] @ (p[:2] / p[2]) + D["K"][:2, 2]) * SCALE
            BB[i] = [u, u, v, v]
        else:
            BB[i] = [xs.min(), xs.max(), ys.min(), ys.max()]
    R = Renderer(mesh, D["K"], D["W"], D["H"], scale=SCALE)

    def prep(frames, with_img=False):
        bb = BB[frames]
        cen = np.stack([(bb[:, 0] + bb[:, 1]) / 2, (bb[:, 2] + bb[:, 3]) / 2], 1)
        arrs = [OM[frames], PM[frames]] + ([VID[frames]] if with_img else [])
        off, outs = make_crops(arrs, cen, *CROP)
        om, pm = outs[0], outs[1]
        d = dict(tgt=torch.tensor(om, dtype=torch.float32, device="cuda"),
                 valid=torch.tensor(~(pm & ~om), dtype=torch.float32, device="cuda"),
                 vis=torch.tensor(om & ~pm, dtype=torch.float32, device="cuda"),
                 crop=(torch.tensor(off, dtype=torch.float32, device="cuda"), CROP))
        if with_img:
            d["img"] = torch.tensor(outs[2], dtype=torch.float32, device="cuda") / 255
        return d

    def eval_frames(T, frames):
        """per-frame IoU (person pixels ignored) and texture NCC on visible object pixels."""
        iou, ncc = [], []
        for s in range(0, len(frames), 64):
            fr = frames[s:s + 64]
            d = prep(fr, with_img=True)
            rgb, m = render_crop(R, torch.tensor(T[fr], dtype=torch.float32, device="cuda"), d["crop"][0], *CROP)
            tg = d["tgt"] > 0
            val = d["valid"] > 0
            iou.append((((m & tg) & val).sum((1, 2)).float() / ((m | tg) & val).sum((1, 2)).clamp(min=1).float()).cpu().numpy())
            ncc.append(ncc_score(rgb, m, d["img"], d["vis"]).cpu().numpy())
        return np.concatenate(iou), np.concatenate(ncc)

    # ------------------------------------------------------------ automatic check of the phase constants
    for name, s, e in PHASES:
        if name[0] == "S":
            bb = BB[s:e]
            # bottom row and the side not occluded by the person should be constant in a static phase
            log[f"{name}_bbox_bottom_range_px"] = float(np.percentile(bb[:, 3], 95) - np.percentile(bb[:, 3], 5))

    # ------------------------------------------------------------ symmetry hypotheses (k*90 deg about mesh z)
    C = np.eye(4)
    C[:3, 3] = center

    def rz(deg):
        M = np.eye(4)
        c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        M[:2, :2] = [[c, -s_], [s_, c]]
        return C @ M @ np.linalg.inv(C)

    sym = {}
    T_init = Tb.copy()
    for name, s, e in PHASES:
        fr = list(range(s, e, 5))
        d = prep(fr, with_img=True)
        scores = []
        for k in range(4):
            Th = Tb[fr] @ rz(90 * k)
            rgb, m = render_crop(R, torch.tensor(Th, dtype=torch.float32, device="cuda"), d["crop"][0], *CROP)
            scores.append(ncc_score(rgb, m, d["img"], d["vis"]).cpu().numpy())
        scores = np.array(scores)
        best = int(np.argmax(scores.mean(1)))
        sym[name] = dict(mean_ncc=[round(float(x), 3) for x in scores.mean(1)],
                         frac_frames_best=[round(float((scores.argmax(0) == k).mean()), 2) for k in range(4)], best=best)
        if best != 0:
            T_init[s:e] = Tb[s:e] @ rz(90 * best)
    log["symmetry"] = sym
    print("symmetry check:", json.dumps(sym))

    # ------------------------------------------------------------ texture + silhouette refinement of a shared pose
    Kw = R.K

    def perturb(T0, p):
        """p = (rotvec about the box centre in the object frame [deg] (3), image shift of the centre [px] (2));
        camera depth of the centre is kept."""
        T1 = T0.copy()
        Rm = T0[:3, :3] @ Rot.from_rotvec(np.radians(p[:3])).as_matrix()
        pc = T0[:3, :3] @ center + T0[:3, 3]
        u = Kw[0, 0] * pc[0] / pc[2] + Kw[0, 2] + p[3]
        v = Kw[1, 1] * pc[1] / pc[2] + Kw[1, 2] + p[4]
        pc2 = np.array([(u - Kw[0, 2]) / Kw[0, 0] * pc[2], (v - Kw[1, 2]) / Kw[1, 1] * pc[2], pc[2]])
        T1[:3, :3] = Rm
        T1[:3, 3] = pc2 - Rm @ center
        return T1

    def refine_texture(T0, frames):
        """Coordinate descent on J = mean NCC(texture) + W_IOU * mean IoU(silhouette) over the phase frames."""
        d = prep(frames, with_img=True)
        tg = d["tgt"] > 0
        val = d["valid"] > 0

        def J(p):
            Tp = np.repeat(perturb(T0, p)[None], len(frames), 0)
            rgb, m = render_crop(R, torch.tensor(Tp, dtype=torch.float32, device="cuda"), d["crop"][0], *CROP)
            iou = (((m & tg) & val).sum((1, 2)).float() / ((m | tg) & val).sum((1, 2)).clamp(min=1).float()).mean()
            return float(ncc_score(rgb, m, d["img"], d["vis"]).mean() + W_IOU * iou)
        p = np.zeros(5)
        best = j0 = J(p)
        steps = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
        for rnd in range(3):
            for k in range(5):
                for sgn in (-1, 1):
                    for _ in range(12):
                        q = p.copy()
                        q[k] += sgn * steps[k]
                        jq = J(q)
                        if jq > best + 1e-5:
                            p, best = q, jq
                        else:
                            break
            steps /= 2
        return p, j0, best

    # ------------------------------------------------------------ static phases: one pose per phase
    out = T_init.copy()
    static_fit = {}
    for name, s, e in PHASES:
        if name[0] != "S":
            continue
        fr = list(range(s, e, STATIC_STEP))
        d = prep(fr)
        T0 = T_init[(s + e) // 2].copy()
        T0[:3, :3] = Rot.from_matrix(T_init[s:e, :3, :3]).mean().as_matrix()
        pc = T_init[s:e, :3, :3] @ center + T_init[s:e, :3, 3]
        T0[:3, 3] = np.median(pc, 0) - T0[:3, :3] @ center
        T, L = fit_poses(R, T0[None], d["tgt"], d["valid"], center, iters=300, shared=True, crop=d["crop"])
        # texture refinement: the silhouette of a box is flat w.r.t. small rotations; the texture is not
        p, j0, j1 = refine_texture(T[0], fr)
        Tr = perturb(T[0], p)
        out[s:e] = Tr
        static_fit[name] = Tr
        log[f"{name}_texture_refine"] = dict(rotvec_deg=p[:3].tolist(), duv_px=p[3:].tolist(), J_before=j0, J_after=j1)
        print(f"{name}: static fit centre {np.round(Tr[:3, :3] @ center + Tr[:3, 3], 4)}, texture refine {np.round(p, 2)} "
              f"J {j0:.4f}->{j1:.4f} ({time.time() - t_start:.0f}s)")

    # ------------------------------------------------------------ moving phases: smooth joint fit
    for name, s, e in PHASES:
        if name[0] != "M":
            continue
        fr = list(range(s, e))
        d = prep(fr)
        T, L = fit_track(R, T_init[s:e], d["tgt"], d["valid"], center, d["crop"],
                         T_before=out[s - 1] if s > 0 else None, T_after=out[e] if e < F else None,
                         iters=400, lam_p=LAM_P, lam_r=LAM_R)
        out[s:e] = T
        print(f"{name}: track fit done ({time.time() - t_start:.0f}s)")
    T_fit = out.copy()

    # ------------------------------------------------------------ depth re-anchoring along the viewing rays
    def centre_cam(T):
        return T[:, :3, :3] @ center + T[:, :3, 3]

    def scale_along_ray(T, k):
        T = T.copy()
        p = centre_cam(T) * np.asarray(k).reshape(-1, 1)
        T[:, :3, 3] = p - T[:, :3, :3] @ center
        return T

    def bottom_height(Tc):
        """height of the bottom-face centre above the released floor plane (m); insensitive to small tilts."""
        Tw = T_wc[None] @ Tc
        return (Tw[:, :3, :3] @ bottom_c + Tw[:, :3, 3]) @ n_pl + d_pl

    zb = centre_cam(Tb)[:, 2]
    zf = centre_cam(T_fit)[:, 2]
    k_ph = {}
    for name, s, e in PHASES:
        if name in ("S1", "S2"):
            k_ph[name] = float(np.median(zb[s:e]) / np.median(zf[s:e]))
    # S3: default = baseline median; check two independent cues
    s3, e3 = [(s, e) for n, s, e in PHASES if n == "S3"][0]
    zb3 = float(np.median(zb[s3:e3]))
    k_default = zb3 / float(zf[s3])
    # cue A: metric silhouette size, with the silhouette depth bias calibrated on S1/S2 (where baseline is stable)
    k_cal = float(np.mean([k_ph["S1"], k_ph["S2"]]))
    cueA = k_cal * float(zf[s3]) - zb3
    # cue B: an untouched bin standing on the same floor must have the same bottom height above the floor plane as
    # in S1/S2 (only the relative height is used, so a constant offset of the plane fit cancels)
    h_ref = float(np.mean([bottom_height(scale_along_ray(static_fit[n][None], k_ph[n]))[0] for n in ("S1", "S2")]))
    ks = np.linspace(0.95, 1.05, 201) * k_default
    hs = np.array([bottom_height(scale_along_ray(static_fit["S3"][None], k))[0] for k in ks])
    kB = float(np.interp(0.0, (hs - h_ref)[::-1], ks[::-1])) if np.all(np.diff(hs) < 0) else float(ks[np.argmin(np.abs(hs - h_ref))])
    cueB = kB * float(zf[s3]) - zb3
    disagree = abs(cueA - cueB)
    shift = 0.5 * (cueA + cueB)
    apply_s3 = disagree < CUE_AGREE_M and abs(shift) > disagree
    k_ph["S3"] = (zb3 + shift) / float(zf[s3]) if apply_s3 else k_default
    log["depth_cues_S3"] = dict(baseline_median_z=zb3, cueA_silhouette_shift_m=cueA, cueB_floor_shift_m=cueB,
                                disagreement_m=disagree, applied_shift_m=shift if apply_s3 else 0.0,
                                h_ref_S1S2_m=h_ref, k_cal=k_cal)
    print("S3 depth cues:", json.dumps(log["depth_cues_S3"]))
    # per-frame factors: static phases constant, moving phases linear between neighbouring static factors
    kf = np.zeros(F)
    for i, (name, s, e) in enumerate(PHASES):
        if name[0] == "S":
            kf[s:e] = k_ph[name]
        else:
            ka, kb_ = k_ph[PHASES[i - 1][0]], k_ph[PHASES[i + 1][0]]
            a = (np.arange(s, e) - (s - 1)) / (e - (s - 1))
            kf[s:e] = (1 - a) * ka + a * kb_
    log["depth_factors"] = k_ph
    T_final = scale_along_ray(T_fit, kf)

    # ------------------------------------------------------------ validation (no ground truth)
    allf = list(range(F))
    iou_b, ncc_b = eval_frames(Tb, allf)
    iou_r, ncc_r = eval_frames(T_final, allf)
    hb, hr = bottom_height(Tb), bottom_height(T_final)
    zr = centre_cam(T_final)[:, 2]
    pb, pr = centre_cam(Tb), centre_cam(T_final)

    def jitter(T, s, e):
        pc = centre_cam(T[s:e])
        dt = np.linalg.norm(np.diff(pc, axis=0), axis=1)
        dR = [np.degrees(np.linalg.norm(Rot.from_matrix(T[i + 1, :3, :3] @ T[i, :3, :3].T).as_rotvec())) for i in range(s, e - 1)]
        return 1000 * float(np.median(dt)), float(np.median(dR)), float(np.max(dR))

    Vs = V[::10]

    def add(Ta, Tb_):
        return np.array([np.linalg.norm((Vs @ a[:3, :3].T + a[:3, 3]) - (Vs @ b[:3, :3].T + b[:3, 3]), axis=1).mean()
                         for a, b in zip(Ta, Tb_)])
    add_rb = add(T_final, Tb)
    table = []
    for name, s, e in PHASES:
        held = [i for i in range(s, e) if (name[0] == "S" and (i - s) % STATIC_STEP != 0)]
        row = dict(phase=name, frames=f"{s}-{e - 1}",
                   iou_base=float(iou_b[s:e].mean()), iou_res=float(iou_r[s:e].mean()),
                   ncc_base=float(ncc_b[s:e].mean()), ncc_res=float(ncc_r[s:e].mean()),
                   floor_h_base_cm=100 * float(np.median(hb[s:e])), floor_h_res_cm=100 * float(np.median(hr[s:e])),
                   dz_mean_cm=100 * float(np.mean(zr[s:e] - zb[s:e])), dz_median_cm=100 * float(np.median(zr[s:e] - zb[s:e])),
                   jitter_base=jitter(Tb, s, e), jitter_res=jitter(T_final, s, e),
                   ADD_res_vs_base_cm_median=100 * float(np.median(add_rb[s:e])))
        if held:
            row["iou_heldout_res"] = float(iou_r[held].mean())
            row["iou_heldout_base"] = float(iou_b[held].mean())
            row["ncc_heldout_res"] = float(ncc_r[held].mean())
            row["ncc_heldout_base"] = float(ncc_b[held].mean())
        table.append(row)
    log["table"] = table
    for r in table:
        print(json.dumps(r))

    T_world = T_wc[None] @ T_final
    assert T_world.shape == (F, 4, 4) and np.all(np.isfinite(T_world))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))
    log["runtime_s"] = time.time() - t_start
    json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "solve_log.json"), "w"), indent=1,
              default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    print(f"wrote {args.out}  ({log['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
