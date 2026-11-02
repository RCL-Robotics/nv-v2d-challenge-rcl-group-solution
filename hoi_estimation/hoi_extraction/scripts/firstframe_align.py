#!/usr/bin/env python3
"""Camera -> released-world transform from the FIRST-FRAME human only (organiser-approved for Tier 3).

Organiser clarification: "For Tier 3, you may use the first-frame human pose from the clean Tier 1
caption solely to establish the released coordinate frame. Apply a single rigid transform
consistently to your entire reconstruction; subsequent clean trajectory frames must not be used."

Reference: Tier 1 frame-0 SOMA-X human (ref_human_f0/<ep>.npz holds only frame 0), decoded with
soma.SOMALayer and flipped to OpenCV axes. Ours: frame-0 MHR human decoded from a CoCoNet bundle
(camera frame). The two meshes have different topologies, so the transform is fitted by trimmed
point-to-point ICP between the surfaces, initialised at identity (the released world frame is
close to the exocentric camera frame). Writes {episode: {"rigid": 4x4, "sim": 4x4, "rms_m": ...}}.
Runs in the v2d_cari4d image on GPU (SOMA/MHR decoders).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")

FLIP = np.diag([1.0, -1.0, -1.0])  # SOMA/MHR model axes (y up, z back) -> OpenCV (y down, z forward)


def fit(src, dst, with_scale):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (xs ** 2).sum()) if with_scale else 1.0
    return s, R, md - s * R @ ms


def yaw(deg):
    a = np.radians(deg)  # rotation about the camera/world vertical (OpenCV y) axis
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def icp(src, dst, with_scale, iters=60, trim=0.8, max_rot_deg=30.0, wide=False):
    """Multi-start ICP: centroid-aligned starts at several yaw angles; keep the lowest residual whose
    rotation stays within max_rot_deg of identity (the released world is close to the camera frame)."""
    best = None
    yaws = range(-180, 180, 30) if wide else (0, -20, 20, -45, 45)
    if wide:
        max_rot_deg = 180.0  # some sessions' released world frame is far from the camera frame
    for y in yaws:
        R0 = yaw(y)
        T, err = _icp_from(src, dst, with_scale, R0, dst.mean(0) - R0 @ src.mean(0), iters, trim)
        rot = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))) - 1) / 2, -1, 1)))
        if rot <= max_rot_deg and (best is None or err < best[1]):
            best = (T, err)
    if best is None:  # e.g. our human is a different person (wrong person prompt): keep the start-0 fit, flag it
        T, err = _icp_from(src, dst, with_scale, np.eye(3), dst.mean(0) - src.mean(0), iters, trim)
        return T, float("inf") if err == err else err
    return best


def _icp_from(src, dst, with_scale, R, t, iters, trim):
    tree = cKDTree(dst)
    s = 1.0
    for _ in range(iters):
        cur = s * src @ R.T + t
        d, j = tree.query(cur)
        keep = d <= np.quantile(d, trim)
        s, R, t = fit(src[keep], dst[j[keep]], with_scale)
    cur = s * src @ R.T + t
    d, _ = tree.query(cur)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    return T, float(np.sqrt(np.mean(np.minimum(d, np.quantile(d, trim)) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True, help="episode=path/to/coconet.pth")
    ap.add_argument("--v2d", default="/mnt/nas/Group/hanzhou/v2d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wide", action="store_true", help="search all yaw starts without the rotation prior")
    ap.add_argument("--ground-scales", default=None, help="ground_scales.json: scale our camera-frame human by scale_aligned before fitting")
    args = ap.parse_args()
    import soma
    from lib_mhr.mhr_layer import MHRLayer
    from tools.render_mhr_wild_inference import _params

    dev = torch.device("cuda")
    soma_layer = soma.SOMALayer(identity_model_type="mhr", device="cuda")
    mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{args.v2d}/weights/sam3d_body", device="cuda")
    res = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    gs = json.load(open(args.ground_scales)) if args.ground_scales else {}
    for item in args.bundles:
        ep, path = item.split("=", 1)
        ref = np.load(f"{args.v2d}/ref_human_f0/episode_{int(ep):06d}.npz")
        with torch.no_grad():
            out = soma_layer(torch.tensor(ref["pose"]).float().view(1, 77, 3).to(dev), torch.tensor(ref["identity"]).float()[None].to(dev),
                             scale_params=torch.tensor(ref["scale"]).float()[None].to(dev), transl=torch.tensor(ref["transl"]).float()[None].to(dev))
            ref_v = out.vertices[0].detach().cpu().numpy().astype(np.float64) @ FLIP.T
            b = torch.load(path, map_location="cpu", weights_only=False)
            our_v = mhr.mhr_forward(_params(b["pr"], slice(0, 1), dev)).vertices[0].detach().cpu().numpy().astype(np.float64)
        c = float(gs.get(str(int(ep)), {}).get("scale_aligned", 1.0)) if gs else 1.0
        our_v = our_v * c  # depth-scale correction about the camera centre (same factor applied to the object)
        rng = np.random.default_rng(0)
        src = our_v[rng.choice(len(our_v), 6000, replace=False)]
        dst = ref_v[rng.choice(len(ref_v), 6000, replace=False)]
        Tr, er = icp(src, dst, False, wide=args.wide)
        Ts, es = icp(src, dst, True, wide=args.wide)
        res[str(int(ep))] = {"bundle": path, "ground_scale": c, "rigid": Tr.tolist(), "rigid_rms_m": er, "sim": Ts.tolist(), "sim_rms_m": es,
                             "sim_scale": float(np.cbrt(np.linalg.det(Ts[:3, :3]))),
                             "rigid_rot_deg": float(np.degrees(np.arccos(np.clip((np.trace(Tr[:3, :3]) - 1) / 2, -1, 1)))),
                             "rigid_trans_m": float(np.linalg.norm(Tr[:3, 3])),
                             "reliable": bool(np.isfinite(er) and er < 0.03)}
        print(ep, {k: round(v, 3) for k, v in res[str(int(ep))].items() if isinstance(v, float)}, flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
