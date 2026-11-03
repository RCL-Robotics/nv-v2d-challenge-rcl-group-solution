#!/usr/bin/env python3
"""Per-episode 180-degree flip selection by textured appearance (GPU; image + released textured mesh).

Flat or near-symmetric objects (platform, A-frame sign, ...) are sometimes tracked with a consistent
180-degree flip: the silhouette is identical, only the texture differs. For each episode we render
the textured mesh under the current pose and under 180-degree flips about the three principal axes
on frames where the object mask is large and away from the human, and compare colours inside the
union of the rendered and observed object masks (mean absolute Lab difference, rendered pixels only).
A flip replaces the trajectory only when it is better by a clear margin on most frames.
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
import trimesh

sys.path.insert(0, "/workspace/v2d_foundation_pose/lib/FoundationPose")
V2D = os.environ["V2D_ROOT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src", nargs="+", default=["M1", "P1_temporal", "E2_reregister", "baseline"])
    ap.add_argument("--out-exp", default="FL1_flip")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--margin", type=float, default=0.15, help="relative colour-error improvement needed to flip")
    ap.add_argument("--win-frac", type=float, default=0.7, help="fraction of frames where the flip must win")
    ap.add_argument("--feature", choices=["lab", "dino"], default="lab", help="appearance distance: Lab colour or DINOv2 patch features")
    args = ap.parse_args()
    import nvdiffrast.torch as dr
    import Utils

    glctx = dr.RasterizeCudaContext()
    dino = None
    if args.feature == "dino":
        os.environ.setdefault("TORCH_HOME", f"{V2D}/weights/sam3d_body/torch_home")
        dino = torch.hub.load(f"{V2D}/weights/sam3d_body/torch_home/hub/facebookresearch_dinov2_main", "dinov2_vitb14", source="local").cuda().eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]

    def dino_dist(img_rgb, ren_rgb, mask):
        """1 - cosine similarity of DINOv2 patch tokens, averaged over patches inside the object mask."""
        v, u = np.nonzero(mask)
        pad = 10
        v0, v1, u0, u1 = max(v.min() - pad, 0), v.max() + pad, max(u.min() - pad, 0), u.max() + pad
        crops = []
        for im in (img_rgb, ren_rgb):
            c = cv2.resize(np.ascontiguousarray(im[v0:v1, u0:u1]), (224, 224))
            crops.append((torch.as_tensor(c, device="cuda").permute(2, 0, 1).float() / 255 - mean) / std)
        with torch.no_grad():
            tok = dino.forward_features(torch.stack(crops))["x_norm_patchtokens"]  # (2, 256, C)
        m = cv2.resize(np.ascontiguousarray(mask[v0:v1, u0:u1]).astype(np.uint8), (16, 16), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        if m.sum() < 4:
            return np.nan
        a, b = torch.nn.functional.normalize(tok[0, m], dim=-1), torch.nn.functional.normalize(tok[1, m], dim=-1)
        return float(1 - (a * b).sum(-1).mean())
    ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    summary = {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        variant, src = next(((v, p) for v in args.src for p in [f"{V2D}/outputs/cari4d/{ep}/inference/coconet.pth" if v == "baseline" else f"{V2D}/experiments/{v}/{ep}/coconet.pth"] if os.path.isfile(p)), (None, None))
        if src is None:
            continue
        work = f"{V2D}/experiments/M1_prompt_v2/work/{ep}" if variant == "M1" else f"{V2D}/outputs/cari4d/{ep}/work"
        if not os.path.isdir(f"{work}/masks"):
            work = f"{V2D}/outputs/cari4d_failed/{ep}_work"
        kf = f"{V2D}/outputs/cari4d/{ep}/depth/moge2/intrinsics.pkl"
        if not os.path.isfile(kf):
            kf = f"{V2D}/outputs/cari4d_failed/{ep}_out/depth/moge2/intrinsics.pkl"
        k = pickle.load(open(kf, "rb"))
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.0]])
        mesh = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="mesh", process=False)
        if hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "to_simple"):
            mesh.visual.material = mesh.visual.material.to_simple()  # FoundationPose expects a texture image
        S = np.asarray(trimesh.bounds.oriented_bounds(mesh)[0], dtype=np.float64)
        A = S[:3, :3]
        flips = {"none": np.eye(3), "x180": A.T @ np.diag([1, -1, -1]) @ A, "y180": A.T @ np.diag([-1, 1, -1]) @ A, "z180": A.T @ np.diag([-1, -1, 1]) @ A}
        mt = Utils.make_mesh_tensors(mesh)
        b = torch.load(src, map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        TG = T @ S  # camera_T_glb
        cap = cv2.VideoCapture(f"{ds}/videos/chunk-000/observation.images.exo_camera/{ep}.mp4")
        cand = []
        kernel = np.ones((31, 31), np.uint8)
        for t in range(0, len(T), 5):
            mo = cv2.imread(f"{work}/masks/1/{t:06d}.png", 0)
            mh = cv2.imread(f"{work}/masks/0/{t:06d}.png", 0)
            if mo is None or (mo > 0).sum() < 1500:
                continue
            overlap = ((cv2.dilate((mh > 0).astype(np.uint8), kernel) > 0) & (mo > 0)).mean() / max((mo > 0).mean(), 1e-9) if mh is not None else 0
            cand.append((overlap, -(mo > 0).sum(), t))
        cand = [c[2] for c in sorted(cand)[: args.frames]]
        errs = {name: [] for name in flips}
        for t in cand:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, bgr = cap.read()
            if not ok:
                continue
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            mo = cv2.imread(f"{work}/masks/1/{t:06d}.png", 0) > 0
            H, Wd = mo.shape
            for name, F in flips.items():
                P = TG[t].copy()
                P[:3, :3] = P[:3, :3] @ F
                color, depth, _ = Utils.nvdiffrast_render(K=K, H=H, W=Wd, ob_in_cams=torch.as_tensor(P[None], dtype=torch.float32, device="cuda"), glctx=glctx, mesh_tensors=mt)
                rgb = (color[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                rmask = depth[0].cpu().numpy() > 0
                m = rmask & mo
                if m.sum() < 300:
                    errs[name].append(np.nan)
                    continue
                if args.feature == "dino":
                    errs[name].append(dino_dist(bgr[..., ::-1], rgb, rmask | mo))
                else:
                    rlab = cv2.cvtColor(rgb[..., ::-1].copy(), cv2.COLOR_BGR2LAB).astype(np.float32)
                    errs[name].append(float(np.abs(rlab[m] - lab[m]).mean()))
        E = {n: np.array(v, dtype=float) for n, v in errs.items()}
        base = E["none"]
        choice, best_gain = "none", 0.0
        for name in ("x180", "y180", "z180"):
            valid = np.isfinite(base) & np.isfinite(E[name])
            if valid.sum() < 5:
                continue
            gain = 1 - np.median(E[name][valid]) / np.median(base[valid])
            wins = np.mean(E[name][valid] < base[valid])
            if gain > args.margin and wins >= args.win_frac and gain > best_gain:
                choice, best_gain = name, float(gain)
        if choice != "none":
            TG[:, :3, :3] = TG[:, :3, :3] @ flips[choice]
            b["pr"]["pose_abs"] = (TG @ np.linalg.inv(S)).astype(np.float32)
        out = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        torch.save(b, f"{out}/coconet.pth")
        summary[idx] = {"src": variant, "choice": choice, "gain": best_gain, "frames": len(cand),
                        "median_err": {n: float(np.nanmedian(v)) if np.isfinite(v).any() else None for n, v in E.items()}}
        print(ep, choice, round(best_gain, 3), len(cand), {n: (round(x, 1) if x else None) for n, x in summary[idx]["median_err"].items()}, flush=True)
    os.makedirs(f"{V2D}/experiments/{args.out_exp}", exist_ok=True)
    json.dump(summary, open(f"{V2D}/experiments/{args.out_exp}/flip_select.json", "w"), indent=1)


if __name__ == "__main__":
    main()
