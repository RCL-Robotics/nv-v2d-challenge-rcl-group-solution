#!/usr/bin/env python3
"""Tier 3 evaluation under the organiser-approved protocol (runs in v2d_cari4d image, CPU).

Each episode's camera-frame object trajectory is mapped to the released world frame with ONE rigid
transform fitted on the first-frame human (firstframe_align.py); no further alignment. Optionally
camera-frame translations are first multiplied by the ground-plane depth-scale factor
(ground_scale.py; the alignment must then have been fitted on the equally scaled human).
The released Tier 2 trajectories are scored with the same code (no alignment) for comparison.

--pipeline lists variants in fallback order, e.g. P1_temporal E2_reregister baseline; a variant
name maps to experiments/<name>/<ep>/coconet.pth ("baseline" = outputs/cari4d/<ep>/inference).
--align lists alignment json files in the same fallback order per variant (variant=path).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from eval_object_pose import metrics, quat_wxyz_to_R  # noqa: E402

V2D = os.environ["V2D_ROOT"]


def bundle_path(variant, ep):
    if variant == "baseline":
        return f"{V2D}/outputs/cari4d/{ep}/inference/coconet.pth"
    return f"{V2D}/experiments/{variant}/{ep}/coconet.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", nargs="+", required=True)
    ap.add_argument("--align", nargs="+", required=True, help="variant=alignment.json (use 'default=...' as fallback)")
    ap.add_argument("--ground-scales", default=None, help="ground_scales.json; scale translations by scale_aligned")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--use-sim", action="store_true", help="use the first-frame similarity (CARI4D Table 3) instead of the rigid fit")
    args = ap.parse_args()
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    aligns = {k: json.load(open(v)) for k, v in (a.split("=", 1) for a in args.align)}
    gs = json.load(open(args.ground_scales)) if args.ground_scales else {}
    t2 = pd.read_csv(f"{V2D}/logs/eval_tier2.csv")
    t2 = t2[t2["align"] == "raw"].set_index("episode")
    rows = []
    for idx in range(30):
        ep = f"episode_{idx:06d}"
        variant = next((v for v in args.pipeline if os.path.isfile(bundle_path(v, ep))), None)
        al = None
        for key in (variant, "default"):
            if key in aligns and str(idx) in aligns[key]:
                al = aligns[key][str(idx)]
                break
        if variant is None or al is None:
            rows.append({"ep": idx, "object": meta[idx]["object"], "variant": variant or "-", "t2_AUC": t2.loc[idx, "AUC"]})
            continue
        P = np.asarray(torch.load(bundle_path(variant, ep), map_location="cpu", weights_only=False)["pr"]["pose_abs"], dtype=np.float64)
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], dtype=np.float64)
        verts = np.asarray(glb.sample(1000), dtype=np.float64)
        T = P @ S
        c = gs.get(str(idx), {}).get("scale_aligned", 1.0) if args.ground_scales else 1.0
        T[:, :3, 3] *= c
        W = np.asarray(al["sim" if args.use_sim else "rigid"], dtype=np.float64)
        s = float(np.cbrt(np.linalg.det(W[:3, :3])))
        Tw = W[None] @ T
        Tw[:, :3, :3] /= s  # a similarity scales positions only; keep rotations orthonormal
        ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
        n = min(len(Tw), len(ref))
        ok = (np.linalg.norm(ref[:n, 3:], axis=1) > 0.5) & np.isfinite(Tw[:n]).all(axis=(1, 2))
        m = metrics(Tw[:n][ok][:, :3, :3], Tw[:n][ok][:, :3, 3], quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3], verts)
        rows.append({"ep": idx, "object": meta[idx]["object"], "variant": variant, "align_ok": al.get("reliable", True),
                     **{k: m[k] for k in ("AUC", "MPPE_cm", "pos_cm", "rot_deg", "SP", "MP")},
                     "t2_AUC": t2.loc[idx, "AUC"], "t2_MPPE": t2.loc[idx, "MPPE_cm"]})
    d = pd.DataFrame(rows)
    d["beat"] = d["AUC"] > d["t2_AUC"]
    pd.set_option("display.width", 220)
    print(d.round(3).to_string(index=False))
    print(f"\n{args.pipeline} ground_scale={bool(args.ground_scales)}: beats Tier 2 AUC on {int(d.beat.sum())}/30; "
          f"mean AUC {d.AUC.mean():.3f} (Tier 2 {d.t2_AUC.mean():.3f}); MPPE {d.MPPE_cm.mean():.1f} cm (Tier 2 {d.t2_MPPE.mean():.1f})")
    if args.out_csv:
        d.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
