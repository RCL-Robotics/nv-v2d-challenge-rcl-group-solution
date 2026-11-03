#!/usr/bin/env python3
"""Joint camera calibration + feet-on-floor depth (GPU, MHR/SOMA decode).

The first-frame PnP camera (firstframe_pnp.py) is poorly conditioned: for a single human ~3 m away a
1 deg pitch and a ~5 cm vertical camera shift give almost the same projection, and a 1 deg pitch error
moves a floor contact by ~5% along the ray. We therefore fit the camera rotation and centre (and one
global depth factor for the SAM-3D human) to two kinds of evidence:
  * the frame-0 reference human (organiser-approved, frame 0 only) re-projected onto our frame-0 human
    in the image, and
  * at every (3rd) frame, our human's lowest vertex lying at the frame-0 reference foot height h0 above
    the released ground plane.
Then, as in postproc_feet_floor.py, a per-frame factor a_t puts the human's lowest vertex at h0 under the
new camera (median filtered) and the object's camera-frame translation is scaled by a_t.
Only the released plane, the frame-0 reference human and our own reconstruction are used.
Writes experiments/<out>/<ep>/coconet.pth and logs/<out>_align.json (variant=... for eval_official.py,
rigid transform with unit scale).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")
sys.path.insert(0, os.path.dirname(__file__))
from eval_official import bundle_path  # noqa: E402

V2D = "/mnt/nas/Group/hanzhou/v2d"
FLIP = np.diag([1.0, -1.0, -1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src-map", default=f"{V2D}/experiments/FF2_feet_ref/feet_floor.json", help="json {ep: {src: variant}}")
    ap.add_argument("--align", default=f"{V2D}/logs/pnp_forE1.json", help="first-frame PnP alignment (initial camera)")
    ap.add_argument("--out-exp", default="JC1_joint_camera")
    ap.add_argument("--sig-px", type=float, default=2.0)
    ap.add_argument("--sig-h", type=float, default=0.02)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--clip", type=float, default=0.3)
    args = ap.parse_args()
    from lib_mhr.mhr_layer import MHRLayer
    from tools.render_mhr_wild_inference import _params
    import soma

    dev = torch.device("cuda")
    mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
    sl = soma.SOMALayer(identity_model_type="mhr", device="cuda")
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    al = json.load(open(args.align))
    srcs = json.load(open(args.src_map))
    align_out = f"{V2D}/logs/{args.out_exp}_align.json"
    res = json.load(open(align_out)) if os.path.isfile(align_out) else {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        src = srcs[str(idx)]["src"]
        b = torch.load(bundle_path(src, ep), map_location="cpu", weights_only=False)
        P = np.asarray(b["pr"]["pose_abs"], np.float64)
        F = len(P)
        K = np.asarray(b["observations"]["K_full"], np.float64)[0]
        with torch.no_grad():
            H = np.concatenate([mhr.mhr_forward(_params(b["pr"], slice(a, min(a + 128, F)), dev)).vertices.cpu().numpy().astype(np.float64)[:, ::8]
                                for a in range(0, F, 128)])
        W = np.asarray(al[str(idx)]["sim"], np.float64)
        s = float(np.cbrt(np.linalg.det(W[:3, :3])))
        Rw, C = W[:3, :3] / s, W[:3, 3]
        gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"], np.float64)
        n, d = gp[:3] / np.linalg.norm(gp[:3]), gp[3] / np.linalg.norm(gp[:3])
        if n @ C + d < 0:
            n, d = -n, -d
        r0 = np.load(f"{V2D}/ref_human_f0/{ep}.npz")  # frame-0 reference human only
        with torch.no_grad():
            g0 = sl(torch.tensor(r0["pose"]).float().view(1, 77, 3).to(dev), torch.tensor(r0["identity"]).float()[None].to(dev),
                    scale_params=torch.tensor(r0["scale"]).float()[None].to(dev), transl=torch.tensor(r0["transl"]).float()[None].to(dev)
                    ).vertices[0].cpu().numpy().astype(np.float64) @ FLIP
        h0 = float((g0 @ n + d).min())
        _, j0 = cKDTree(g0).query(s * H[0] @ Rw.T + C)
        X0 = g0[j0]
        uv0 = H[0] @ K.T
        uv0 = uv0[:, :2] / uv0[:, 2:3]
        Hs = H[::3]
        w0 = 1.0 / args.sig_px / np.sqrt(len(X0) / 200)

        def resid(q):
            R = Rot.from_rotvec(q[:3]).as_matrix() @ Rw
            Cc = C + q[3:6]
            Xc = (X0 - Cc) @ R
            r_img = (Xc[:, :2] / Xc[:, 2:3] * K[0, 0] + K[:2, 2] - uv0).ravel() * w0
            low = np.einsum("j,tvj->tv", n @ R, Hs).min(1) * np.exp(q[6]) + n @ Cc + d
            return np.r_[r_img, (low - h0) / args.sig_h]

        q = least_squares(resid, np.zeros(7), loss="soft_l1", f_scale=1.0).x
        R = Rot.from_rotvec(q[:3]).as_matrix() @ Rw
        Cc = C + q[3:6]
        low = np.einsum("j,tvj->tv", n @ R, H).min(1)
        c0 = n @ Cc + d
        a = np.where(low < -1e-6, (h0 - c0) / low, 1.0)
        a = np.clip(a, 1 - args.clip, 1 + args.clip)
        w = args.window
        a = np.array([np.median(a[max(0, i - w):i + w + 1]) for i in range(F)])
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], np.float64)
        TG = P @ S
        TG[:, :3, 3] *= a[:, None]
        b["pr"]["pose_abs"] = (TG @ np.linalg.inv(S)).astype(np.float32)
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        torch.save(b, f"{out}/coconet.pth")
        Wn = np.eye(4)
        Wn[:3, :3], Wn[:3, 3] = R, Cc
        res[str(idx)] = {"bundle": f"{out}/coconet.pth", "sim": Wn.tolist(), "rigid": Wn.tolist(), "src": src, "h0": h0,
                         "drot_deg": float(np.degrees(np.linalg.norm(q[:3]))), "dC_cm": float(np.linalg.norm(q[3:6]) * 100),
                         "human_scale": float(np.exp(q[6])), "a_median": float(np.median(a))}
        print(ep, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[str(idx)].items() if k not in ("sim", "rigid", "bundle")}, flush=True)
        json.dump(res, open(align_out, "w"), indent=1)


if __name__ == "__main__":
    main()
