#!/usr/bin/env python3
"""Focal-length estimates for the static exo camera from the video only (GPU; third_party_vda venv).

MoGe-2 gives one focal per episode with ~2% noise and a systematic bias. The camera is static, so we
pool several frames per episode and compare single-image calibrators: GeoCalib (pinhole, shared focal)
and DepthPro (focal head). Output: json {episode_index: {"geocalib": [...], "depthpro": [...]}} in px.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d")
VIDEOS = f"{V2D}/dataset/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera"


def frames(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default=f"{V2D}/logs/focal_estimates.json")
    ap.add_argument("--depthpro-ckpt", default=os.path.expanduser("~/Projects/v2d/third_party_vda/checkpoints/depth_pro.pt"))
    args = ap.parse_args()
    dev = torch.device("cuda")
    from geocalib import GeoCalib
    gc = GeoCalib().to(dev)
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    cfg = DEFAULT_MONODEPTH_CONFIG_DICT
    cfg.checkpoint_uri = args.depthpro_ckpt
    dp, dp_tf = depth_pro.create_model_and_transforms(config=cfg, device=dev, precision=torch.half)
    dp.eval()
    res = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    for idx in args.episodes:
        ims = frames(f"{VIDEOS}/episode_{idx:06d}.mp4", args.frames)
        g, d = [], []
        with torch.no_grad():
            for im in ims:
                t = torch.from_numpy(im).permute(2, 0, 1).float().div(255).to(dev)
                g.append(float(gc.calibrate(t, camera_model="pinhole")["camera"].f[0, 0]))
                d.append(float(dp.infer(dp_tf(im).to(dev))["focallength_px"]))
            # GeoCalib can also share one camera over all frames of the static view
            batch = torch.stack([torch.from_numpy(im).permute(2, 0, 1).float().div(255) for im in ims]).to(dev)
            try:
                gs = float(gc.calibrate(batch, camera_model="pinhole", shared_intrinsics=True)["camera"].f[0, 0])
            except Exception as e:  # older API without shared intrinsics
                gs = float("nan")
        res[str(idx)] = {"geocalib": g, "depthpro": d, "geocalib_shared": gs, "h": int(ims[0].shape[0]), "w": int(ims[0].shape[1])}
        print(idx, f"geocalib {np.median(g):.1f} (iqr {np.percentile(g,25):.0f}-{np.percentile(g,75):.0f}) shared {gs:.1f} depthpro {np.median(d):.1f}", flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
