#!/usr/bin/env python3
"""Video Depth Anything (metric, ViT-L) depth for one episode, written like CARI4D's raw MoGe depth.

Output: <out>/raw.npy, uint16 millimetres, [T, H, W] at the source resolution (1152 x 1536).
Run with third_party_vda/.venv/bin/python. Stage 4 of CARI4D rescales it to the human each frame,
so only VDA's relative (and temporally consistent) depth matters.
"""
import argparse
import os
import sys

import numpy as np
import torch

VDA = "/home/hanzhou/Projects/v2d/third_party_vda"
sys.path.insert(0, VDA)
from utils.dc_utils import read_video_frames  # noqa: E402
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-size", type=int, default=518)
    args = ap.parse_args()
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], metric=True)
    model.load_state_dict(torch.load(f"{VDA}/checkpoints/metric_video_depth_anything_vitl.pth", map_location="cpu"), strict=True)
    model = model.to("cuda").eval()
    frames, fps = read_video_frames(args.video, -1, -1, 4096)  # full resolution
    depths, _ = model.infer_video_depth(frames, fps, input_size=args.input_size, device="cuda", fp32=False)
    depths = np.asarray(depths, dtype=np.float32)
    os.makedirs(args.out, exist_ok=True)
    mm = np.clip(np.round(depths * 1000.0), 1, 65535).astype(np.uint16)
    np.save(f"{args.out}/raw.npy", mm)
    print("depth", mm.shape, "median m", float(np.median(depths)))


if __name__ == "__main__":
    main()
