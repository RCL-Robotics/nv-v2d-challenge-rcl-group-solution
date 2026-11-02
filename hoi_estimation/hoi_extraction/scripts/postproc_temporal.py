#!/usr/bin/env python3
"""Temporal stabilisation of CARI4D object poses (CPU only, no reference data used).

1. Rotation outlier rejection: in a sliding window the medoid rotation (minimum summed geodesic
   distance to the other frames) is the local consensus; frames deviating by more than
   --outlier-deg are replaced by it. Translations deviating by more than --outlier-m from the
   window median are replaced likewise.
2. Static hold: a frame is static when its SAM2 object-mask centroid and area barely change over
   +-k frames. Static runs of at least --min-static frames get one pose: the medoid rotation and
   median translation of the run.
Input/output poses are camera-frame 4x4 (pr.pose_abs of a CoCoNet bundle).
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch


def geodesic(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.einsum("...ij,...ij->...", Ra, Rb) - 1) / 2, -1, 1)))


def medoid(R):
    D = geodesic(R[:, None], R[None])
    return R[int(np.argmin(D.sum(1)))]


def mask_stats(mask_dir, n):
    files = sorted(glob.glob(f"{mask_dir}/*.png"))[:n]
    cen = np.full((n, 2), np.nan)
    area = np.zeros(n)
    for i, f in enumerate(files):
        m = cv2.imread(f, 0) > 0
        area[i] = m.sum()
        if area[i] > 0:
            v, u = np.nonzero(m)
            cen[i] = (u.mean(), v.mean())
    return cen, area


def static_frames(cen, area, k=5, px=3.0, rel_area=0.05, min_area=200):
    n = len(area)
    st = np.zeros(n, bool)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        if area[a] < min_area or area[b] < min_area or np.isnan(cen[[a, b]]).any():
            continue
        st[i] = np.linalg.norm(cen[b] - cen[a]) < px and abs(area[b] - area[a]) / max(area[a], 1) < rel_area
    return st


def runs(mask):
    out, start = [], None
    for i, v in enumerate(np.r_[mask, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def stabilise(T, cen, area, window=15, outlier_deg=30.0, outlier_m=0.10, min_static=15, static_hold=True):
    T = T.copy()
    n = len(T)
    R, t = T[:, :3, :3].copy(), T[:, :3, 3].copy()
    R0, t0 = R.copy(), t.copy()
    for i in range(n):
        a, b = max(0, i - window), min(n, i + window + 1)
        Rm = medoid(R0[a:b])
        if geodesic(R0[i], Rm) > outlier_deg:
            R[i] = Rm
        tm = np.median(t0[a:b], axis=0)
        if np.linalg.norm(t0[i] - tm) > outlier_m:
            t[i] = tm
    st = static_frames(cen, area)
    if static_hold:
        for a, b in runs(st):
            if b - a >= min_static:
                R[a:b] = medoid(R[a:b][:: max(1, (b - a) // 200)])  # subsample long runs for the O(n^2) medoid
                t[a:b] = np.median(t[a:b], axis=0)
    T[:, :3, :3], T[:, :3, 3] = R, t
    return T, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--v2d", default=os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"))
    ap.add_argument("--src-exp", default="E2_reregister")
    ap.add_argument("--out-exp", default="P1_temporal")
    ap.add_argument("--no-static-hold", action="store_true")
    args = ap.parse_args()
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        b = torch.load(f"{args.v2d}/experiments/{args.src_exp}/{ep}/coconet.pth", map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        mdir = f"{args.v2d}/outputs/cari4d/{ep}/work/masks/1"
        if not os.path.isdir(mdir):
            mdir = f"{args.v2d}/outputs/cari4d_failed/{ep}_work/masks/1"
        cen, area = mask_stats(mdir, len(T))
        T2, st = stabilise(T, cen, area, static_hold=not args.no_static_hold)
        out = f"{args.v2d}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        b["pr"]["pose_abs"] = T2.astype(np.float32)
        b.setdefault("metadata", {})["v2d_postproc"] = {"script": "postproc_temporal.py", "src_exp": args.src_exp, "static_hold": not args.no_static_hold}
        torch.save(b, f"{out}/coconet.pth")
        print(f"{ep}: static frames {st.mean():.2f}, rotations changed {(geodesic(T[:, :3, :3], T2[:, :3, :3]) > 1).mean():.2f}")


if __name__ == "__main__":
    main()
