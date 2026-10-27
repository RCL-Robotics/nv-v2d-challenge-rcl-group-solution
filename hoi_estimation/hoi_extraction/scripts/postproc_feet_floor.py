#!/usr/bin/env python3
"""Per-frame depth correction from the human's feet touching the released ground plane (GPU, MHR decode).

CARI4D anchors each frame's depth scale to the SAM-3D human, whose depth drifts over the episode, so
the whole reconstruction drifts in depth after the first-frame alignment. The released ground plane
is in the released world frame and at least one foot touches it most of the time. For every frame
we find the factor f_t that scales the camera-frame scene about the camera centre so that the
human's lowest vertex lies on the plane (after the episode's first-frame rigid transform W),
median-filter f_t over time, and apply it to the object's camera-frame translation.
Only released data (ground plane) and our own reconstruction are used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh

sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")
V2D = "/mnt/nas/Group/hanzhou/v2d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src", nargs="+", default=["M1", "P1_temporal", "E2_reregister", "baseline"])
    ap.add_argument("--align", nargs="+", required=True, help="variant=alignment.json, default=...")
    ap.add_argument("--out-exp", default="FF1_feet_floor")
    ap.add_argument("--window", type=int, default=15, help="half-width of the median filter on f_t")
    ap.add_argument("--clip", type=float, default=0.15, help="limit |f_t - 1|")
    ap.add_argument("--use-sim", action="store_true", help="alignment json holds 'sim' (PnP camera + depth scale)")
    ap.add_argument("--ref-foot-height", action="store_true",
                    help="target the frame-0 reference human's lowest-vertex height instead of 0 (mesh feet sink ~2 cm)")
    args = ap.parse_args()
    from lib_mhr.mhr_layer import MHRLayer
    from tools.render_mhr_wild_inference import _params

    dev = torch.device("cuda")
    mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    aligns = {k: json.load(open(v)) for k, v in (a.split("=", 1) for a in args.align)}
    summary = {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        variant, src = next(((v, p) for v in args.src for p in [f"{V2D}/outputs/cari4d/{ep}/inference/coconet.pth" if v == "baseline" else f"{V2D}/experiments/{v}/{ep}/coconet.pth"] if os.path.isfile(p)), (None, None))
        al = next((aligns[k][str(idx)] for k in (variant, "default") if k in aligns and str(idx) in aligns[k]), None)
        if src is None or al is None:
            continue
        W = np.asarray(al["sim" if args.use_sim else "rigid"], dtype=np.float64)
        sc = float(np.cbrt(np.linalg.det(W[:3, :3])))
        R, t0 = W[:3, :3] / sc, W[:3, 3]
        gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"], dtype=np.float64)
        n, d = gp[:3] / np.linalg.norm(gp[:3]), gp[3] / np.linalg.norm(gp[:3])
        if n @ t0 + d < 0:  # orient towards the camera (up)
            n, d = -n, -d
        b = torch.load(src, map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        F = len(T)
        low = np.empty(F)
        with torch.no_grad():
            for a in range(0, F, 64):
                v = mhr.mhr_forward(_params(b["pr"], slice(a, min(a + 64, F)), dev)).vertices.detach().cpu().numpy().astype(np.float64)
                # height of every vertex above the floor as a function of k: n.(R (k p) + t0) + d = k (n.R p) + (n.t0 + d)
                a_coef = np.einsum("j,tvj->tv", n @ R, v)
                low_idx = np.argmin(a_coef, axis=1)  # the lowest vertex for k near 1 (a_coef < 0 below the camera)
                low[a:a + len(v)] = a_coef[np.arange(len(v)), low_idx]
        low = low * sc  # PnP depth scale
        c0 = n @ t0 + d  # camera-centre height above the floor (world)
        h0 = 0.0
        if args.ref_foot_height:  # frame-0 reference human only (organiser-approved)
            import soma
            if not hasattr(main, "_soma"):
                main._soma = soma.SOMALayer(identity_model_type="mhr", device="cuda")
            ref = np.load(f"{V2D}/ref_human_f0/{ep}.npz")
            with torch.no_grad():
                rv = main._soma(torch.tensor(ref["pose"]).float().view(1, 77, 3).to(dev), torch.tensor(ref["identity"]).float()[None].to(dev),
                                scale_params=torch.tensor(ref["scale"]).float()[None].to(dev), transl=torch.tensor(ref["transl"]).float()[None].to(dev)).vertices[0].detach().cpu().numpy() @ np.diag([1.0, -1.0, -1.0])
            h0 = float((rv @ n + d).min())
        f_raw = np.where(low < -1e-6, (h0 - c0) / low, 1.0)  # k with k * low + c0 = h0
        f_raw = np.clip(f_raw, 1 - args.clip, 1 + args.clip)
        w = args.window
        f = np.array([np.median(f_raw[max(0, i - w):i + w + 1]) for i in range(F)])
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], dtype=np.float64)
        TG = T @ S
        TG[:, :3, 3] *= f[:, None]  # (the PnP depth scale is re-applied at evaluation)
        b["pr"]["pose_abs"] = (TG @ np.linalg.inv(S)).astype(np.float32)
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        torch.save(b, f"{out}/coconet.pth")
        summary[idx] = {"h0": h0, "src": variant, "f_median": float(np.median(f)), "f_p5": float(np.percentile(f, 5)), "f_p95": float(np.percentile(f, 95)), "f0": float(f[0])}
        print(ep, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[idx].items()}, flush=True)
    os.makedirs(f"{V2D}/experiments/{args.out_exp}", exist_ok=True)
    json.dump(summary, open(f"{V2D}/experiments/{args.out_exp}/feet_floor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
