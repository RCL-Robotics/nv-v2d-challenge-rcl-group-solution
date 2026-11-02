#!/usr/bin/env python3
"""Replace CARI4D's per-frame human-anchored depth scale by one constant scale per episode (CPU).

Stage 4 rescales MoGe depth every frame to the SAM-3D human (scale s_t, stored in aligned.h5).
s_t drifts as the person moves, so the reconstructed scene drifts in depth. Positions in the camera
frame scale with the depth, so re-expressing frame t with the episode median s_bar multiplies its
camera-frame object translation by f_t = s_bar / s_t. Writes experiments/<out>/<ep>/coconet.pth and
<out>/frame0_scale.json ({ep: {"scale_aligned": f_0}}) so firstframe_align.py can rescale the
frame-0 human consistently (--ground-scales).
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
import trimesh

V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src", nargs="+", default=["P1_temporal", "E2_reregister", "baseline"])
    ap.add_argument("--out-exp", default="C1_constscale")
    ap.add_argument("--smooth", type=int, default=0, help="if >0, use a moving median of this half-width instead of the episode median")
    ap.add_argument("--target", choices=["median", "frame0", "ground_raw"], default="median",
                    help="median: episode median of s_t; frame0: s_0; ground_raw: raw MoGe depth times the released-ground-plane factor scale_raw")
    args = ap.parse_args()
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    f0 = {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        src = next((p for v in args.src for p in [f"{V2D}/outputs/cari4d/{ep}/inference/coconet.pth" if v == "baseline" else f"{V2D}/experiments/{v}/{ep}/coconet.pth"] if os.path.isfile(p)), None)
        h5 = f"{V2D}/outputs/cari4d/{ep}/depth/moge2/aligned.h5"
        if src is None or not os.path.isfile(h5):
            continue
        s = h5py.File(h5, "r")["alignment/front_stereo_camera_left/scale"][:].astype(np.float64)
        b = torch.load(src, map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        n = min(len(T), len(s))
        if args.smooth > 0:
            target = np.array([np.median(s[max(0, i - args.smooth):i + args.smooth + 1]) for i in range(len(s))])
        elif args.target == "frame0":
            target = np.full(len(s), s[0])
        elif args.target == "ground_raw":
            gs = json.load(open(f"{V2D}/logs/ground_scales.json"))
            target = np.full(len(s), float(gs[str(idx)]["scale_raw"]))
        else:
            target = np.full(len(s), np.median(s))
        f = target / s
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], dtype=np.float64)
        TG = T @ S
        TG[:n, :3, 3] *= f[:n, None]
        T = TG @ np.linalg.inv(S)
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        b["pr"]["pose_abs"] = T.astype(np.float32)
        torch.save(b, f"{out}/coconet.pth")
        f0[str(idx)] = {"scale_aligned": float(f[0])}
        print(ep, f"scale median {np.median(s):.3f} range {s.min():.3f}-{s.max():.3f} f0 {f[0]:.3f}", flush=True)
    json.dump(f0, open(f"{V2D}/experiments/{args.out_exp}/frame0_scale.json", "w"), indent=1)


if __name__ == "__main__":
    main()
