#!/usr/bin/env python3
"""Symmetry-flip hypothesis test per episode (GPU, nvdiffrast; inside v2d_cari4d image).

Many remaining rotation failures are a constant 180 deg turn about one of the object's own near-symmetry axes: the
silhouette is nearly the same, but texture and 3D shape may differ. For each episode we compare the current
trajectory (hypothesis B) with the same trajectory turned by 180 deg about each principal (oriented-bounds) axis of
the mesh through its centroid, in the object frame (hypotheses F_x, F_y, F_z), on sampled,
little-occluded frames, using only the video, the SAM2 masks and the MoGe-2 depth of our own pipeline:
  photo : normalised cross-correlation between the textured render and the image (object pixels not covered by
          the person), on grey-level gradients;
  depth : spread (MAD) of MoGe depth minus rendered depth over the same pixels (after removing the median offset).
Per frame each cue votes for the better hypothesis. Decision rule (fixed a priori, not tuned on references):
flip to the best F when both cues favour it in >= 60% of the frames and on average in >= 70%; if one cue is unavailable, the other
must favour F in >= 75%. Writes experiments/<out>/<ep>/coconet.pth (flipped or copied) and <out>/yaw_flip.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
from eval_official import bundle_path  # noqa: E402

V2D = "/mnt/nas/Group/hanzhou/v2d"
DS = f"{V2D}/dataset/track_2/tier_1_multiview_caption"


def pipeline_dir(src, ep):
    m1 = f"{V2D}/experiments/M1_prompt_v2"
    return (f"{m1}/out/{ep}", f"{m1}/work/{ep}") if src == "M1" else (f"{V2D}/outputs/cari4d/{ep}", f"{V2D}/outputs/cari4d/{ep}/work")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--src-exp", default="DB1_depth_blend")
    ap.add_argument("--align", default=f"{V2D}/logs/JC1_joint_camera_align.json")
    ap.add_argument("--out-exp", default="YF1_yaw_flip")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-area", type=int, default=800)
    ap.add_argument("--max-occ", type=float, default=0.3)
    args = ap.parse_args()
    import nvdiffrast.torch as dr

    dev = torch.device("cuda")
    ctx = dr.RasterizeCudaContext(device=dev)
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
    al = json.load(open(args.align))
    log = f"{V2D}/experiments/{args.out_exp}/yaw_flip.json"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    res = json.load(open(log)) if os.path.isfile(log) else {}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        b = torch.load(bundle_path(args.src_exp, ep), map_location="cpu", weights_only=False)
        src = al[str(idx)]["src"]
        out_dir, work_dir = pipeline_dir(src, ep)
        g = trimesh.load(f"{DS}/{meta[idx]['mesh']}", force="scene", process=False).to_geometry()
        S = np.asarray(trimesh.bounds.oriented_bounds(g)[0], np.float64)
        T = np.asarray(b["pr"]["pose_abs"], np.float64) @ S  # cam_T_glb
        F = len(T)
        K = np.asarray(b["observations"]["K_full"], np.float64)[0]
        W = np.asarray(al[str(idx)]["sim"], np.float64)
        c = g.centroid
        hyps = {}
        for k, ax in enumerate("xyz"):
            a = np.linalg.inv(S)[:3, k]  # principal axis in mesh coordinates
            X = np.eye(4)
            X[:3, :3] = cv2.Rodrigues(np.pi * a / np.linalg.norm(a))[0]
            X[:3, 3] = c - X[:3, :3] @ c
            hyps[ax] = T @ X
        # mesh tensors
        V = torch.tensor(np.asarray(g.vertices), dtype=torch.float32, device=dev)
        Fc = torch.tensor(np.asarray(g.faces), dtype=torch.int32, device=dev)
        tex = uv = None
        vis = g.visual
        if getattr(vis, "uv", None) is not None and getattr(getattr(vis, "material", None), "baseColorTexture", None) is not None:
            uv = torch.tensor(np.asarray(vis.uv), dtype=torch.float32, device=dev)[None]
            img = np.asarray(vis.material.baseColorTexture.convert("RGB"))[::-1].copy()
            tex = torch.tensor(img, dtype=torch.float32, device=dev)[None] / 255
        H, Wd = 1152, 1536
        cap = cv2.VideoCapture(f"{DS}/videos/chunk-000/observation.images.exo_camera/{ep}.mp4")
        mh = h5py.File(f"{work_dir}/masks_k0.h5", "r")[ep]
        dh = h5py.File(f"{out_dir}/depth/moge2/aligned.h5", "r")["aligned/front_stereo_camera_left"]

        def render(Tc):
            Vc = V @ torch.tensor(Tc[:3, :3].T, dtype=torch.float32, device=dev) + torch.tensor(Tc[:3, 3], dtype=torch.float32, device=dev)
            z = Vc[:, 2:3]
            x = (Vc[:, 0:1] * K[0, 0] / z + K[0, 2]) / Wd * 2 - 1
            y = (Vc[:, 1:2] * K[1, 1] / z + K[1, 2]) / H * 2 - 1
            clip = torch.cat([x * z, y * z, z * 1.0002 - 0.02, z], 1)[None]  # rows: y down (OpenCV order)
            rast, _ = dr.rasterize(ctx, clip, Fc, resolution=(H, Wd))
            m = rast[0, ..., 3] > 0
            d, _ = dr.interpolate(Vc[None, :, 2:3].contiguous(), rast, Fc)
            col = None
            if tex is not None:
                tc, _ = dr.interpolate(uv, rast, Fc)
                col = dr.texture(tex, tc, filter_mode="linear")[0]
            return m, d[0, ..., 0], col

        votes = {ax: {"photo": [], "depth": []} for ax in hyps}
        nf = 0
        for t in range(0, F, args.stride):
            om = mh[f"{t:06d}-k0.obj_rend_mask.png"][()] > 0
            pm = mh[f"{t:06d}-k0.person_mask.png"][()] > 0
            area = om.sum()
            if area < args.min_area or (om & pm).sum() / area > args.max_occ or not np.isfinite(T[t]).all():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, im = cap.read()
            if not ok:
                continue
            nf += 1
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            gimg = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
            dm = cv2.imdecode(np.frombuffer(dh[t].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            sc = {}
            for name, Tc in [("B", T[t])] + [(ax, Th[t]) for ax, Th in hyps.items()]:
                m, d, col = render(Tc)
                m = m.cpu().numpy()
                sel = m & om & ~pm
                if sel.sum() < 200:
                    sc[name] = None
                    continue
                dr_ = d.cpu().numpy()[sel]
                r = dm[sel] - dr_
                r = r[np.isfinite(r) & (dm[sel] > 0)]
                dep = float(np.median(np.abs(r - np.median(r)))) if len(r) > 100 else None
                ph = None
                if col is not None:
                    cg = cv2.cvtColor((col.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
                    gr = np.hypot(cv2.Sobel(cg, cv2.CV_32F, 1, 0), cv2.Sobel(cg, cv2.CV_32F, 0, 1))
                    # ignore render gradients at the silhouette border
                    inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    s2 = sel & inner
                    if s2.sum() > 150:
                        a_, b_ = gimg[s2] - gimg[s2].mean(), gr[s2] - gr[s2].mean()
                        ph = float((a_ * b_).sum() / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-9))
                sc[name] = (ph, dep)
            if sc.get("B") is None:
                continue
            for ax in hyps:
                if sc.get(ax) is None:
                    continue
                if sc["B"][0] is not None and sc[ax][0] is not None:
                    votes[ax]["photo"].append(float(sc[ax][0] > sc["B"][0]))
                if sc["B"][1] is not None and sc[ax][1] is not None:
                    votes[ax]["depth"].append(float(sc[ax][1] < sc["B"][1]))

        def decide(v):
            fr = {k: (float(np.mean(x)) if len(x) >= 10 else None) for k, x in v.items()}
            have = [x for x in fr.values() if x is not None]
            if len(have) == 2:
                return fr, min(have) >= 0.6 and np.mean(have) >= 0.7, float(np.mean(have))
            if len(have) == 1:
                return fr, have[0] >= 0.75, have[0]
            return fr, False, 0.0

        dec = {ax: decide(v) for ax, v in votes.items()}
        best = max(dec, key=lambda ax: dec[ax][2])
        flip = dec[best][1]
        if flip:
            b["pr"]["pose_abs"] = (hyps[best] @ np.linalg.inv(S)).astype(np.float32)
        o = f"{V2D}/experiments/{args.out_exp}/{ep}"
        os.makedirs(o, exist_ok=True)
        torch.save(b, f"{o}/coconet.pth")
        res[str(idx)] = {"src": args.src_exp, "frames": nf, "votes": {ax: d[0] for ax, d in dec.items()},
                         "best": best, "flip": bool(flip)}
        print(ep, meta[idx]["object"], res[str(idx)], flush=True)
        json.dump(res, open(log, "w"), indent=1)


if __name__ == "__main__":
    main()
