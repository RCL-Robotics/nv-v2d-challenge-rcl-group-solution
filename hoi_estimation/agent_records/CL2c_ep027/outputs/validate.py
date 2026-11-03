#!/usr/bin/env python
"""Ground-truth-free validation of an object trajectory vs the baseline (run after solve.py).

Measures per phase: silhouette IoU with the SAM2 mask (person-occluded pixels ignored), held-out
photometric consistency (colour model built from even frames of the trajectory itself, evaluated on odd
frames), temporal smoothness, within-phase pose spread, depth difference to the baseline, and physical
cues for the depth along the viewing ray (floor contact, silhouette size, hand contact).
Usage: python validate.py --inputs /work/inputs --result /work/outputs/object_poses_world.npz
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import (TexMesh, crop_K, crop_box_from_mask, fit_silhouettes, model_corr, render_sil_batch,  # noqa
                    resample_crop, rotmat_angle, sample_point_colors, surface_points)
from solve import segment_phases, read_video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--extra", default="", help="comma separated extra trajectories to evaluate")
    args = ap.parse_args()
    torch.manual_seed(0)
    inp = args.inputs
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"])
    Twc = np.array(cam["T_world_cam"])
    Tcw = np.linalg.inv(Twc)
    plane = json.load(open(f"{inp}/ground_plane.json"))["plane"]
    n, d = np.array(plane[:3]), plane[3]
    rec = np.load(f"{inp}/reconstruction.npz")
    trajs = {"baseline": rec["T_world_mesh"].astype(float), "result": np.load(args.result)["T_world_mesh"]}
    for p in [x for x in args.extra.split(",") if x]:
        trajs[os.path.basename(p)] = np.load(p)["T_world_mesh"]
    F = len(trajs["baseline"])
    mesh = TexMesh(f"{inp}/object_mesh.glb")
    with h5py.File(f"{inp}/masks.h5", "r") as h:
        obj = h["object"][:].astype(bool)
        per = h["person"][:].astype(bool)
    frames = read_video(f"{inp}/video.mp4")
    fa, fb = segment_phases(obj, per)
    phases = {"floor-static": (0, fa), "carried": (fa, fb), "desk-static": (fb, F), "all": (0, F)}
    print(f"phases: floor [0,{fa}) carried [{fa},{fb}) desk [{fb},{F})")

    R = 128
    Tc_b = Tcw[None] @ trajs["baseline"]
    Kc = np.zeros((F, 3, 3))
    obs = np.zeros((F, R, R), np.float32)
    val = np.zeros((F, R, R), np.float32)
    for i in range(F):
        p = K @ Tc_b[i, :3, 3]
        box = crop_box_from_mask(obj[i], p[:2] / p[2], min_size=64, scale=1.8)
        Kc[i] = crop_K(K, box, R)
        o = resample_crop(obj[i].astype(np.float32), box, R)
        obs[i] = o
        val[i] = 1 - np.clip(resample_crop(per[i].astype(np.float32), box, R) - o, 0, 1)
    vis_frames = obs.reshape(F, -1).sum(1) > 30

    P_np, N_np = surface_points(mesh, 20000, seed=0)
    P = torch.tensor(P_np, device="cuda")
    N = torch.tensor(N_np, device="cuda")
    pbox, blur = [None] * F, [None] * F
    for i in range(F):
        ys, xs = np.nonzero(obj[i])
        if len(xs) < 30:
            continue
        x0, y0, x1, y1 = max(0, xs.min() - 24), max(0, ys.min() - 24), min(1536, xs.max() + 25), min(1152, ys.max() + 25)
        pbox[i] = (x0, y0, x1, y1)
        blur[i] = cv2.GaussianBlur(frames[i][y0:y1, x0:x1].astype(np.float32) / 255.0, (0, 0), 1.0)
    del frames
    kp = rec["human_keypoints_world"].astype(float)
    hands = np.concatenate([kp[:, 21:42], kp[:, 42:63]], 1)  # right + left hand keypoints
    cprob = rec["contact_prob_right_left"]
    from scipy.spatial import cKDTree
    Vm = mesh.vw.cpu().numpy()[::3]

    rows = {}
    for name, Tw in trajs.items():
        Tc = Tcw[None] @ Tw
        # IoU
        with torch.no_grad():
            Tt = torch.tensor(Tc, dtype=torch.float32, device="cuda")
            ious = np.zeros(F)
            for s in range(0, F, 128):
                e = min(F, s + 128)
                sil = render_sil_batch(mesh, Tt[s:e], torch.tensor(Kc[s:e], dtype=torch.float32, device="cuda"), R)
                h = (sil > 0.5).float()
                O = torch.tensor(obs[s:e], device="cuda")
                V = torch.tensor(val[s:e], device="cuda")
                ious[s:e] = ((h * O * V).sum((1, 2)) / ((h + O - h * O) * V).sum((1, 2)).clamp(min=1)).cpu().numpy()
        # held-out photometric consistency (model from even frames, test on odd frames)
        S = torch.zeros(len(P_np), 3, device="cuda")
        C = torch.zeros(len(P_np), device="cuda")
        cols = {}
        with torch.no_grad():
            for i in range(F):
                if pbox[i] is None:
                    continue
                col, vis = sample_point_colors(mesh, P, N, Tt[i:i + 1], K, blur[i], obj[i], per[i], pbox[i])
                if i % 2 == 0:
                    S += col[0] * vis[0, :, None]
                    C += vis[0].float()
                else:
                    cols[i] = (col, vis)
            model = S / C.clamp(min=1)[:, None]
            photo = np.full(F, np.nan)
            for i, (col, vis) in cols.items():
                photo[i] = model_corr(col, vis, model, C)[0].item()
        # smoothness
        dR = np.zeros(F)
        dT = np.zeros(F)
        for i in range(1, F):
            dR[i] = rotmat_angle(Tw[i - 1][:3, :3], Tw[i][:3, :3])
            dT[i] = np.linalg.norm(Tw[i, :3, 3] - Tw[i - 1, :3, 3]) * 100
        # physical: lowest point vs floor plane; hands to surface
        low = np.array([((Vm @ Tw[i, :3, :3].T + Tw[i, :3, 3]) @ n + d).min() for i in range(F)])
        hd = np.full(F, np.nan)
        for i in range(F):
            if cprob[i].max() > 0.5:
                Pw = Vm @ Tw[i, :3, :3].T + Tw[i, :3, 3]
                hd[i] = np.median(cKDTree(Pw).query(hands[i])[0]) * 100
        rows[name] = dict(iou=ious, photo=photo, dR=dR, dT=dT, low=low, hand=hd, z=Tc[:, 2, 3], Tc=Tc)

    def fmt(x):
        return "%.3f" % x if np.isfinite(x) else "  n/a"
    print("\n| phase | frames | traj | mask IoU (mean / p10) | held-out photo corr | median dR/frame (deg) | "
          "#jumps >15deg | median dT/frame (cm) | rot spread (deg) | dz vs base mean / median (cm) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for ph, (a, b) in phases.items():
        for name, r in rows.items():
            m = vis_frames[a:b]
            iou = r["iou"][a:b][m]
            pc = r["photo"][a:b]
            dR = r["dR"][a + 1:b]
            dT = r["dT"][a + 1:b]
            Rs = trajs[name][a:b, :3, :3]
            Rm = Rs[len(Rs) // 2]
            spread = np.median([rotmat_angle(Rm, x) for x in Rs])
            dz = (r["z"][a:b] - rows["baseline"]["z"][a:b]) * 100
            print(f"| {ph} | {b - a} | {name} | {fmt(iou.mean())} / {fmt(np.percentile(iou, 10))} | "
                  f"{fmt(np.nanmean(pc))} | {np.median(dR):.2f} | {(dR > 15).sum()} | {np.median(dT):.2f} | "
                  f"{spread:.1f} | {dz.mean():+.1f} / {np.median(dz):+.1f} |")
    print("\nphysical cues (result):")
    r = rows["result"]
    print(f"  floor phase: lowest mesh point above floor plane: median {np.median(r['low'][:fa]) * 100:.1f} cm "
          f"(baseline {np.median(rows['baseline']['low'][:fa]) * 100:.1f} cm)")
    print(f"  desk phase: lowest point above floor plane: median {np.median(r['low'][fb:]) * 100:.1f} cm")
    print(f"  hand keypoints to object surface (frames with contact prob>0.5): median {np.nanmedian(r['hand']):.1f} cm "
          f"(baseline {np.nanmedian(rows['baseline']['hand']):.1f} cm)")
    # depth cues: along-ray shift that would satisfy each cue
    Tc = r["Tc"]
    # (1) floor contact: shift along the ray so that the lowest point touches the plane (first order)
    i0 = fa // 2
    Tw0 = trajs["result"][i0]
    cpos = Twc[:3, 3]
    ray = Tw0[:3, 3] - cpos
    ray /= np.linalg.norm(ray)
    dlow = np.median(r["low"][:fa])
    dz_floor = -dlow / (ray @ n) * (Tc[i0, 2, 3] / np.linalg.norm(Tw0[:3, 3] - cpos))
    print(f"  cue floor-contact: lowest point {dlow * 100:.1f} cm above plane -> depth shift {dz_floor * 100:+.1f} cm "
          f"(plane fit tolerance: feet are {json.load(open(f'{inp}/ground_plane.json'))['foot_plane_dist_stats']['median'] * 100:.1f} cm above it)")
    # (2) silhouette size: free-depth silhouette fit on static frames
    for nm, (a, b) in [("floor", (0, fa)), ("desk", (fb, F))]:
        fr = np.arange(a, b, 6)
        T, I = fit_silhouettes(mesh, Kc[fr], obs[fr], val[fr], Tc[fr], iters=100, lr_r=0.0, fix_z=False)
        print(f"  cue silhouette-size ({nm}): depth shift {np.median(T[:, 2, 3] - Tc[fr, 2, 3]) * 100:+.1f} cm "
              f"(IoU {I.mean():.3f})")
    # (3) hands: compare hand-keypoint depth with the depth of the nearest object surface point while carried
    sh = []
    for i in range(fa, fb):
        if cprob[i].max() < 0.5:
            continue
        Pc = (Vm @ Tc[i, :3, :3].T + Tc[i, :3, 3])
        Hc = (hands[i] - Twc[:3, 3]) @ Twc[:3, :3]
        dd, ii = cKDTree(Pc[:, :2] / Pc[:, 2:3]).query(Hc[:, :2] / Hc[:, 2:3])
        sh.append(np.median(Hc[:, 2] - Pc[ii, 2]))
    print(f"  cue hand-contact: median (hand depth - object surface depth at the hand pixel) {np.median(sh) * 100:+.1f} cm")


if __name__ == "__main__":
    main()
