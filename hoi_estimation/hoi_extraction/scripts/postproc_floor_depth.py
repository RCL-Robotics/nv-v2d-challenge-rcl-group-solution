#!/usr/bin/env python3
"""Per-episode object depth correction from floor contact (CPU, released data only).

The released ground plane and the reference poses share the source OpenCV world frame, which our
first-frame human alignment maps the reconstruction into. Our objects sit systematically too close
to the camera, so when they rest on the floor their bottom floats above it. For frames where the
object is static (SAM2 mask) and its lowest point is within --near-m of the floor, we solve for the
factor k that slides the object along its camera ray (camera-frame translation * k) until its lowest
point touches the plane; the episode's k is the median over those frames and is applied to the
whole trajectory. Episodes with fewer than --min-frames such frames keep k = 1.
Output: experiments/<out-exp>/<ep>/coconet.pth plus floor_depth.json with the per-episode k.
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
from postproc_temporal import mask_stats, static_frames  # noqa: E402

V2D = os.environ["V2D_ROOT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src", nargs="+", default=["M1", "P1_temporal", "E2_reregister", "baseline"], help="variant fallback order")
    ap.add_argument("--align", required=True, help="first-frame alignment json (camera -> world)")
    ap.add_argument("--out-exp", default="F1_floor")
    ap.add_argument("--use-sim", action="store_true", help="use the 'sim' (PnP camera) transform; its scale is applied to camera-frame translations")
    ap.add_argument("--align-by-variant", nargs="*", default=[], help="variant=json overrides for episodes whose source is that variant")
    ap.add_argument("--near-m", type=float, default=0.30)
    ap.add_argument("--min-frames", type=int, default=30)
    args = ap.parse_args()
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    al = json.load(open(args.align))
    summary = {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        src = next((p for v in args.src for p in [f"{V2D}/outputs/cari4d/{ep}/inference/coconet.pth" if v == "baseline" else f"{V2D}/experiments/{v}/{ep}/coconet.pth"] if os.path.isfile(p)), None)
        if src is None or str(idx) not in al:
            continue
        b = torch.load(src, map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], dtype=np.float64)
        V = np.asarray(glb.convex_hull.vertices, dtype=np.float64)  # lowest point lies on the hull
        alv = al
        for item in args.align_by_variant:
            vname, vpath = item.split("=", 1)
            if f"/{vname}/" in src:
                alv = json.load(open(vpath))
        W = np.asarray(alv[str(idx)]["sim" if args.use_sim else "rigid"], dtype=np.float64)
        sc = float(np.cbrt(np.linalg.det(W[:3, :3])))
        W[:3, :3] /= sc  # rotation part; the scale multiplies camera-frame translations below
        gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"], dtype=np.float64)
        nrm = np.linalg.norm(gp[:3])
        n, d = gp[:3] / nrm, gp[3] / nrm
        cam_w = W[:3, 3]  # camera centre in the world frame
        if n @ cam_w + d < 0:
            n, d = -n, -d  # orient the normal towards the camera side (up)
        TG = T @ S
        R_w = W[:3, :3] @ TG[:, :3, :3]  # world rotation of the mesh
        t_c = TG[:, :3, 3] * sc  # camera-frame translation (PnP depth scale applied)

        def heights(k):  # lowest mesh point above the floor, per frame, with camera translation scaled by k
            t_w = (W[:3, :3] @ (t_c * k[:, None]).T).T + W[:3, 3]
            pts = np.einsum("tij,vj->tvi", R_w, V) + t_w[:, None]
            return (pts @ n + d).min(1)

        h1 = heights(np.ones(len(T)))
        mdir = f"{V2D}/outputs/cari4d/{ep}/work/masks/1"
        if not os.path.isdir(mdir):
            mdir = f"{V2D}/outputs/cari4d_failed/{ep}_work/masks/1"
        cen, area = mask_stats(mdir, len(T))
        use = static_frames(cen, area) & (np.abs(h1) < args.near_m)
        k_ep = 1.0
        if use.sum() >= args.min_frames:
            ks = []
            for t in np.flatnonzero(use)[:: max(1, int(use.sum()) // 200)]:
                lo, hi = 0.7, 1.4  # bisection: height decreases as the object moves away along the ray
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    t_w = W[:3, :3] @ (t_c[t] * mid) + W[:3, 3]
                    hm = ((R_w[t] @ V.T).T + t_w) @ n + d
                    if hm.min() > 0:
                        lo = mid
                    else:
                        hi = mid
                ks.append(0.5 * (lo + hi))
            k_ep = float(np.median(ks))
        TG[:, :3, 3] *= k_ep  # slide the mesh-origin (GLB) translation along its camera ray (the PnP scale is re-applied at evaluation)
        T = TG @ np.linalg.inv(S)  # back to the bundle's aligned-mesh pose convention
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        b["pr"]["pose_abs"] = T.astype(np.float32)
        torch.save(b, f"{out}/coconet.pth")
        summary[idx] = {"k": k_ep, "floor_frames": int(use.sum()), "median_height_before_m": float(np.median(h1[use])) if use.any() else None}
        print(ep, summary[idx], flush=True)
    json.dump(summary, open(f"{V2D}/experiments/{args.out_exp}/floor_depth.json", "w"), indent=1)


if __name__ == "__main__":
    main()
