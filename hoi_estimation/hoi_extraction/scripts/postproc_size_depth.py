#!/usr/bin/env python3
"""Per-episode object depth correction from apparent size (CPU, image + metric mesh only).

Our objects project onto the right image location but sit 3-6% too close along the camera ray: a
5% depth error changes the silhouette size by only 5%, which the pose fit tolerates. The mesh has
metric size, so the ratio between the projected bounding-box diagonal of our posed mesh and that of
the SAM2 object mask equals the needed depth factor. We use frames where the object mask is large
enough, not cut by the image border and not overlapping the (dilated) human mask, take the median
ratio per episode and slide the object along its camera ray by that factor.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import cv2
import numpy as np
import torch
import trimesh

V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d")


def bbox_diag(mask):
    v, u = np.nonzero(mask)
    return np.hypot(u.max() - u.min(), v.max() - v.min()), (u.min(), v.min(), u.max(), v.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src", nargs="+", default=["M1", "P1_temporal", "E2_reregister", "baseline"])
    ap.add_argument("--out-exp", default="Z1_size_depth")
    ap.add_argument("--min-area", type=int, default=800)
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--clip", type=float, default=0.15)
    args = ap.parse_args()
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    summary = {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        variant, src = next(((v, p) for v in args.src for p in [f"{V2D}/outputs/cari4d/{ep}/inference/coconet.pth" if v == "baseline" else f"{V2D}/experiments/{v}/{ep}/coconet.pth"] if os.path.isfile(p)), (None, None))
        if src is None:
            continue
        work = f"{V2D}/experiments/M1_prompt_v2/work/{ep}" if variant == "M1" else f"{V2D}/outputs/cari4d/{ep}/work"
        if not os.path.isdir(f"{work}/masks"):
            work = f"{V2D}/outputs/cari4d_failed/{ep}_work"
        kf = f"{V2D}/outputs/cari4d/{ep}/depth/moge2/intrinsics.pkl"
        if not os.path.isfile(kf):
            kf = f"{V2D}/outputs/cari4d_failed/{ep}_out/depth/moge2/intrinsics.pkl"
        k = pickle.load(open(kf, "rb"))
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
        b = torch.load(src, map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], dtype=np.float64)
        V = np.asarray(glb.convex_hull.vertices, dtype=np.float64)
        TG = T @ S
        ratios = []
        kernel = np.ones((31, 31), np.uint8)
        for t in range(0, len(T), 3):
            mo = cv2.imread(f"{work}/masks/1/{t:06d}.png", 0)
            mh = cv2.imread(f"{work}/masks/0/{t:06d}.png", 0)
            if mo is None or (mo > 0).sum() < args.min_area:
                continue
            mo = mo > 0
            H, Wd = mo.shape
            do, (u0, v0, u1, v1) = bbox_diag(mo)
            if u0 <= 2 or v0 <= 2 or u1 >= Wd - 3 or v1 >= H - 3:
                continue
            if mh is not None and (cv2.dilate((mh > 0).astype(np.uint8), kernel)[v0:v1 + 1, u0:u1 + 1] > 0).any():
                continue  # any human pixel near the object box: occlusion would shrink the observed box
            pc = V @ TG[t, :3, :3].T + TG[t, :3, 3]
            if (pc[:, 2] <= 0.05).any():
                continue
            uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
            dr = np.hypot(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
            ratios.append(dr / do)
        kfac = float(np.clip(np.median(ratios), 1 - args.clip, 1 + args.clip)) if len(ratios) >= args.min_frames else 1.0
        TG[:, :3, 3] *= kfac
        b["pr"]["pose_abs"] = (TG @ np.linalg.inv(S)).astype(np.float32)
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        torch.save(b, f"{out}/coconet.pth")
        summary[idx] = {"src": variant, "k": kfac, "frames": len(ratios), "ratio_iqr": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))] if ratios else None}
        print(ep, variant, round(kfac, 3), len(ratios), flush=True)
    os.makedirs(f"{V2D}/experiments/{args.out_exp}", exist_ok=True)
    json.dump(summary, open(f"{V2D}/experiments/{args.out_exp}/size_depth.json", "w"), indent=1)


if __name__ == "__main__":
    main()
