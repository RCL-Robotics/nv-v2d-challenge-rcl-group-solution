#!/usr/bin/env python3
"""Temporal median background of each episode video (static camera; the person moves away), plus the per-pixel
fraction of sampled frames where the person or the object mask covers the pixel. Writes OUT/epXXX/background.png and
occupancy.npy."""
import sys, os, numpy as np, cv2, h5py
from common import *
for i in (map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else SUPPORT):
    d = f"{OUT}/ep{i:03d}"; os.makedirs(d, exist_ok=True)
    cap = cv2.VideoCapture(f"{inp(i)}/video.mp4"); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); idx = np.linspace(0, n - 1, 60).astype(int)
    fr = []
    for t in idx: cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, im = cap.read(); fr.append(im) if ok else None
    with h5py.File(f"{inp(i)}/masks.h5") as f:
        m = np.stack([f["person"][t] | f["object"][t] for t in idx[:len(fr)] if t < len(f["person"])])
    fr = np.stack(fr[:len(m)]).astype(np.float32); w = ~m[..., None]
    # median over frames where the pixel is not covered by person/object (fall back to all frames)
    fm = np.where(w, fr, np.nan); bg = np.nanmedian(fm, 0); bg = np.where(np.isnan(bg), np.median(fr, 0), bg)
    cv2.imwrite(f"{d}/background.png", bg.astype(np.uint8)); np.save(f"{d}/occupancy.npy", m.mean(0).astype(np.float16))
    print(i, "frames", len(m), flush=True)
