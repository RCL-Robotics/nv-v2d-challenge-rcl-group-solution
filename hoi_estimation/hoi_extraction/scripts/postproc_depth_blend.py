#!/usr/bin/env python3
"""Blend the CoCoNet object depth with the FoundationPose input depth (CPU).

Relative to the human, the FoundationPose input object ("in") sits ~3% too far along the camera ray and
the CoCoNet output ("pr_initial"/"pr") over-corrects to ~1-2% too close. We take the geometric mean:
the object's camera-frame translation of the source bundle (e.g. JC1) is scaled by k = median_t(|t_in|/|t_pr0|)^w
with w = 0.5 (no tuning), a per-episode constant along each ray; episodes where the two stages disagree by
more than 10% (FoundationPose failure) keep k = 1. Rotations and the human are unchanged.
Only our own reconstruction is used. Writes experiments/<out>/<ep>/coconet.pth (same alignment as the source).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from eval_official import bundle_path  # noqa: E402

V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src", default="JC1_joint_camera")
    ap.add_argument("--out-exp", default="DB1_depth_blend")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--max-dev", type=float, default=0.1, help="skip episodes whose |log median(|t_in|/|t_pr|)| exceeds this")
    args = ap.parse_args()
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    log = f"{V2D}/experiments/{args.out_exp}/depth_blend.json"
    res = json.load(open(log)) if os.path.isfile(log) else {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        b = torch.load(bundle_path(args.src, ep), map_location="cpu", weights_only=False)
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], np.float64)
        tc = lambda key: (np.asarray(b[key]["pose_abs"], np.float64) @ S)[:, :3, 3]
        r = np.linalg.norm(tc("in"), axis=1) / np.linalg.norm(tc("pr_initial"), axis=1)
        r = r[np.isfinite(r) & (r > 0.5) & (r < 2.0)]
        rm = float(np.median(r)) if len(r) else 1.0
        k = rm ** args.w if abs(np.log(rm)) < args.max_dev else 1.0  # larger disagreement: FoundationPose failed
        TG = np.asarray(b["pr"]["pose_abs"], np.float64) @ S
        TG[:, :3, 3] *= k
        b["pr"]["pose_abs"] = (TG @ np.linalg.inv(S)).astype(np.float32)
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        torch.save(b, f"{out}/coconet.pth")
        res[str(idx)] = {"src": args.src, "k": k, "r_median": rm, "w": args.w}
        print(ep, f"k {k:.4f}", flush=True)
    json.dump(res, open(log, "w"), indent=1)


if __name__ == "__main__":
    main()
