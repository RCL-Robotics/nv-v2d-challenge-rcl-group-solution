#!/usr/bin/env python3
"""Per-episode object depth ratio from the image row of the floor contact (CPU, image + released plane).

Our objects sit 2-6% too close along the camera ray yet still touch the floor, so their floor contact
lies closer to the camera and projects LOWER in the image than the true contact. The SAM2 object
mask shows the true contact: its bottom-most pixels. Casting that pixel's ray onto the released ground
plane (in the camera frame given by the first-frame PnP pose) gives the true contact depth; the ratio
to the depth of our posed mesh's lowest point is the depth correction. We use static, unoccluded,
floor-resting frames and take the per-episode median, then slide the object along its camera rays.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from postproc_temporal import mask_stats, static_frames  # noqa: E402

V2D = "/mnt/nas/Group/hanzhou/v2d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src", nargs="+", default=["M1", "P1_temporal", "E2_reregister", "baseline"])
    ap.add_argument("--align", required=True, help="PnP json (sim = camera->world with depth scale)")
    ap.add_argument("--out-exp", default="CR1_contact_row")
    ap.add_argument("--near-m", type=float, default=0.10, help="our lowest point within this height of the floor")
    ap.add_argument("--min-frames", type=int, default=15)
    ap.add_argument("--clip", type=float, default=0.12)
    args = ap.parse_args()
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    al = json.load(open(args.align))
    summary = {}
    kernel = np.ones((31, 31), np.uint8)
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        variant, src = next(((v, p) for v in args.src for p in [f"{V2D}/outputs/cari4d/{ep}/inference/coconet.pth" if v == "baseline" else f"{V2D}/experiments/{v}/{ep}/coconet.pth"] if os.path.isfile(p)), (None, None))
        if src is None or str(idx) not in al:
            continue
        work = f"{V2D}/experiments/M1_prompt_v2/work/{ep}" if variant == "M1" else f"{V2D}/outputs/cari4d/{ep}/work"
        if not os.path.isdir(f"{work}/masks"):
            work = f"{V2D}/outputs/cari4d_failed/{ep}_work"
        kf = f"{V2D}/outputs/cari4d/{ep}/depth/moge2/intrinsics.pkl"
        if not os.path.isfile(kf):
            kf = f"{V2D}/outputs/cari4d_failed/{ep}_out/depth/moge2/intrinsics.pkl"
        k = pickle.load(open(kf, "rb"))
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
        Kinv = np.linalg.inv(K)
        Sim = np.asarray(al[str(idx)]["sim"], dtype=np.float64)
        s = float(np.cbrt(np.linalg.det(Sim[:3, :3])))
        Rwc, twc = Sim[:3, :3] / s, Sim[:3, 3]
        gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"], dtype=np.float64)
        n_w, d_w = gp[:3] / np.linalg.norm(gp[:3]), gp[3] / np.linalg.norm(gp[:3])
        # plane in our (depth-scaled) camera frame: x_w = Rwc (s x) + twc  ->  n_w.(Rwc s x) + n_w.twc + d_w = 0
        n_c = s * (Rwc.T @ n_w)
        d_c = n_w @ twc + d_w
        if d_c < 0:
            n_c, d_c = -n_c, -d_c  # camera centre on the positive (upper) side
        mesh = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(mesh)[0], dtype=np.float64)
        V = np.asarray(mesh.convex_hull.vertices, dtype=np.float64)
        b = torch.load(src, map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        TG = T @ S
        cen, area = mask_stats(f"{work}/masks/1", len(T))
        stat = static_frames(cen, area)
        ratios = []
        for t in np.flatnonzero(stat)[::2]:
            pc = V @ TG[t, :3, :3].T + TG[t, :3, 3]
            h = pc @ n_c + d_c  # height above floor (scaled camera frame units ~ metres)
            if h.min() > args.near_m or h.min() < -args.near_m:
                continue
            mo = cv2.imread(f"{work}/masks/1/{t:06d}.png", 0)
            mh = cv2.imread(f"{work}/masks/0/{t:06d}.png", 0)
            if mo is None:
                continue
            mo = mo > 0
            rows = np.flatnonzero(mo.any(1))
            if len(rows) < 10:
                continue
            vb = rows.max()
            if vb >= mo.shape[0] - 3:
                continue  # cut by the image border
            cols = np.flatnonzero(mo[max(vb - 2, 0):vb + 1].any(0))
            ub = float(np.median(cols))
            if mh is not None and cv2.dilate((mh > 0).astype(np.uint8), kernel)[vb, int(ub)]:
                continue  # contact region occluded by / adjacent to the human
            ray = Kinv @ np.array([ub, vb, 1.0])
            lam = -d_c / (n_c @ ray)
            if lam <= 0:
                continue
            z_true = (lam * ray)[2]
            z_ours = pc[np.argmin(h), 2]
            ratios.append(z_true / z_ours)
        kfac = float(np.clip(np.median(ratios), 1 - args.clip, 1 + args.clip)) if len(ratios) >= args.min_frames else 1.0
        TG[:, :3, 3] *= kfac
        b["pr"]["pose_abs"] = (TG @ np.linalg.inv(S)).astype(np.float32)
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        torch.save(b, f"{out}/coconet.pth")
        summary[idx] = {"src": variant, "k": kfac, "frames": len(ratios),
                        "iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))] if ratios else None}
        print(ep, variant, round(kfac, 3), len(ratios), flush=True)
    os.makedirs(f"{V2D}/experiments/{args.out_exp}", exist_ok=True)
    json.dump(summary, open(f"{V2D}/experiments/{args.out_exp}/contact_row.json", "w"), indent=1)


if __name__ == "__main__":
    main()
