#!/usr/bin/env python3
"""Camera pose + depth scale from the first-frame human via PnP (organiser-approved data only).

Rigid 3D ICP between our frame-0 human and the Tier 1 frame-0 human moves the camera to absorb our
human's DEPTH error, which then shifts every object. Our human's image projection is reliable while
its depth is not, and the reference human has the true size and position. So:
  1. correspondences: start from the rigid ICP alignment (firstframe_align.py output), match each of
     our frame-0 human vertices p_i (camera frame) to its nearest reference vertex q_i (world);
  2. PnP: world->camera pose from (q_i, K p_i) pixels, initialised from the ICP pose;
  3. depth scale: s = median( depth of q_i in the PnP camera / depth of p_i ).
Our camera-frame reconstruction maps to the world as  x_w = R_wc (s x_c) + t_wc ; stored as a 4x4
similarity under "sim" (use eval_official.py --use-sim).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")
FLIP = np.diag([1.0, -1.0, -1.0])
V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", nargs="+", required=True, help="episode=path/to/coconet.pth")
    ap.add_argument("--icp", required=True, help="firstframe_align.py json (rigid init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-match-m", type=float, default=0.05)
    args = ap.parse_args()
    import soma
    from lib_mhr.mhr_layer import MHRLayer
    from tools.render_mhr_wild_inference import _params

    dev = torch.device("cuda")
    soma_layer = soma.SOMALayer(identity_model_type="mhr", device="cuda")
    mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
    icp = json.load(open(args.icp))
    res = json.load(open(args.out)) if os.path.isfile(args.out) else {}
    for item in args.bundles:
        ep, path = item.split("=", 1)
        idx = int(ep)
        e = f"episode_{idx:06d}"
        ref = np.load(f"{V2D}/ref_human_f0/{e}.npz")
        with torch.no_grad():
            q_all = soma_layer(torch.tensor(ref["pose"]).float().view(1, 77, 3).to(dev), torch.tensor(ref["identity"]).float()[None].to(dev),
                               scale_params=torch.tensor(ref["scale"]).float()[None].to(dev), transl=torch.tensor(ref["transl"]).float()[None].to(dev)).vertices[0].detach().cpu().numpy().astype(np.float64) @ FLIP.T
            b = torch.load(path, map_location="cpu", weights_only=False)
            p_all = mhr.mhr_forward(_params(b["pr"], slice(0, 1), dev)).vertices[0].detach().cpu().numpy().astype(np.float64)
        kfile = f"{V2D}/outputs/cari4d/{e}/depth/moge2/intrinsics.pkl"
        if not os.path.isfile(kfile):
            kfile = f"{V2D}/outputs/cari4d_failed/{e}_out/depth/moge2/intrinsics.pkl"
        k = pickle.load(open(kfile, "rb"))
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
        W = np.asarray(icp[str(idx)]["rigid"], dtype=np.float64)
        pw = p_all @ W[:3, :3].T + W[:3, 3]
        dist, j = cKDTree(q_all).query(pw)
        keep = dist < args.max_match_m
        p, q = p_all[keep], q_all[j[keep]]
        uv = (p @ K.T)[:, :2] / p[:, 2:3]
        Wi = np.linalg.inv(W)  # world -> camera initial guess
        rvec0, _ = cv2.Rodrigues(Wi[:3, :3])
        ok, rvec, tvec = cv2.solvePnP(q.astype(np.float64), uv.astype(np.float64), K, None, rvec0, Wi[:3, 3].reshape(3, 1).copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        Rcw, _ = cv2.Rodrigues(rvec)
        qc = q @ Rcw.T + tvec.ravel()
        s = float(np.median(qc[:, 2] / p[:, 2]))
        reproj = np.linalg.norm((qc @ K.T)[:, :2] / qc[:, 2:3] - uv, axis=1)
        Rwc, twc = Rcw.T, -Rcw.T @ tvec.ravel()
        S = np.eye(4)
        S[:3, :3], S[:3, 3] = s * Rwc, twc
        res[str(idx)] = {"bundle": path, "sim": S.tolist(), "rigid": W.tolist(), "depth_scale": s, "matches": int(keep.sum()),
                         "reproj_px_median": float(np.median(reproj)), "pnp_ok": bool(ok),
                         "cam_shift_vs_icp_m": float(np.linalg.norm(twc - W[:3, 3]))}
        print(ep, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in res[str(idx)].items() if kk not in ("sim", "rigid", "bundle")}, flush=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
