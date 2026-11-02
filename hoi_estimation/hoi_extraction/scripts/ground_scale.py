#!/usr/bin/env python3
"""Per-episode depth-scale correction from the released ground plane (runs in v2d_cari4d image, CPU).

The released ground plane (ground_plane/<seq>.json, ax+by+cz+d=0) is expressed in the exocentric
camera frame, so |d| is the metric camera height. We fit the floor in our depth (lower image band,
human/object masks removed, RANSAC + SVD refit) on sampled frames and report
    scale_aligned = released height / floor height in CARI4D's human-aligned depth
    scale_raw     = released height / floor height in raw MoGe depth
Multiplying camera-frame translations by scale_aligned corrects CARI4D's depth-scale bias.
"""
import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np


def floor_height(E, work, use_aligned, frames=12, rng=np.random.default_rng(0)):
    import pickle
    raw = np.load(f"{E}/depth/moge2/raw.npy", mmap_mode="r")
    k = pickle.load(open(f"{E}/depth/moge2/intrinsics.pkl", "rb"))
    sc = h5py.File(f"{E}/depth/moge2/aligned.h5", "r")["alignment/front_stereo_camera_left/scale"][:] if use_aligned else None
    T, H, W = raw.shape
    out = []
    for t in np.linspace(0, T - 1, frames).astype(int):
        d = np.asarray(raw[t], dtype=np.float64) / 1000.0 * (sc[t] if use_aligned else 1.0)
        m = np.zeros((H, W), bool)
        m[int(H * 0.62):] = True
        for oid in (0, 1):
            f = f"{work}/masks/{oid}/{t:06d}.png"
            if os.path.isfile(f):
                m &= ~cv2.dilate((cv2.imread(f, 0) > 0).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        v, u = np.nonzero(m & (d > 0.1))
        if len(v) < 500:
            continue
        idx = rng.choice(len(v), min(20000, len(v)), replace=False)
        v, u = v[idx], u[idx]
        z = d[v, u]
        P = np.stack([(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z], 1)
        best = None
        for _ in range(300):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((P - a) @ n) < 0.02
            if best is None or inl.sum() > best.sum():
                best = inl
        Q = P[best]
        c0 = Q.mean(0)
        n = np.linalg.svd(Q - c0)[2][-1]
        out.append(abs(n @ c0))
    return float(np.median(out)) if out else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--v2d", default=os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"))
    ap.add_argument("--out", default=os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d") + "/logs/ground_scales.json")
    args = ap.parse_args()
    ds = f"{args.v2d}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    res = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    for i in args.episodes:
        E = f"{args.v2d}/outputs/cari4d/episode_{i:06d}"
        work = f"{E}/work"
        if not os.path.isdir(f"{E}/depth"):
            E, work = f"{args.v2d}/outputs/cari4d_failed/episode_{i:06d}_out", f"{args.v2d}/outputs/cari4d_failed/episode_{i:06d}_work"
        gp = np.array(json.load(open(f"{ds}/{meta[i]['ground_plane']}"))["plane"])
        h_gt = abs(gp[3]) / np.linalg.norm(gp[:3])
        ha, hr = floor_height(E, work, True), floor_height(E, work, False)
        res[str(i)] = {"camera_height_m": h_gt, "scale_aligned": h_gt / ha, "scale_raw": h_gt / hr}
        print(i, {k: round(v, 3) for k, v in res[str(i)].items()}, flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
