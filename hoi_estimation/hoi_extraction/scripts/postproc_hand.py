#!/usr/bin/env python3
"""Hand-attachment prior for grasped objects (CPU, no reference data used).

While a hand holds the object, the object rotates with the hand. For each contact segment we take an
anchor frame where the FoundationPose estimate is well supported (highest silhouette IoU around the
segment start) and propagate the object rotation with the hand rotation:
    R_obj(t) = R_hand(t) R_hand(t0)^T R_obj(t0)
R_hand is the Kabsch rotation of the 21 MHR hand keypoints (wrist-centred) between t0 and t.
Contact = hand keypoints within --contact-m of the object surface (object centre minus half its
bounding-box diagonal), for at least --min-frames frames. With --propagate-translation the object
also keeps its offset in the hand frame.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch
import trimesh

HANDS = {"right": (list(range(21, 41)), 41), "left": (list(range(42, 62)), 62)}


def kabsch(A, B):
    """Rotation R minimising |R A_i - B_i| for centred point sets A, B (N x 3)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def runs(mask, min_len, fill=5):
    m = mask.copy()
    idx = np.flatnonzero(m)
    for a, b in zip(idx[:-1], idx[1:]):  # close short gaps
        if 1 < b - a <= fill:
            m[a:b] = True
    out, start = [], None
    for i, v in enumerate(np.r_[m, False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--v2d", default=os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"))
    ap.add_argument("--src-exp", default="P1_temporal")
    ap.add_argument("--fp-exp", default="E2_reregister", help="experiment holding fp.pkl (silhouette IoU for anchors)")
    ap.add_argument("--out-exp", default="H1_hand")
    ap.add_argument("--contact-m", type=float, default=0.08)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--anchor-window", type=int, default=15)
    ap.add_argument("--propagate-translation", action="store_true")
    args = ap.parse_args()
    ds = f"{args.v2d}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        src = f"{args.v2d}/experiments/{args.src_exp}/{ep}/coconet.pth"
        if not os.path.isfile(src):
            print(f"{ep}: no {args.src_exp} bundle, skip")
            continue
        b = torch.load(src, map_location="cpu", weights_only=False)
        T = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        n = len(T)
        root = f"{args.v2d}/outputs/cari4d/{ep}"
        mi = f"{root}/mhr_init/{ep}.pkl"
        if not os.path.isfile(mi):
            mi = f"{args.v2d}/outputs/cari4d_failed/{ep}_out/mhr_init/{ep}.pkl"
        kp = np.asarray(pickle.load(open(mi, "rb"))["mhr_keypoints"], dtype=np.float64)[:n]
        fpf = f"{args.v2d}/experiments/{args.fp_exp}/{ep}/fp.pkl"
        iou = np.nan_to_num(np.asarray(pickle.load(open(fpf, "rb"))["fp_selected_iou"], dtype=float)[:n]) if os.path.isfile(fpf) else np.zeros(n)
        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        radius = 0.5 * float(np.linalg.norm(glb.extents))
        c = T[:, :3, 3]
        dist = {h: np.linalg.norm(kp[:, ids + [w]] - c[:, None], axis=-1).min(1) - radius for h, (ids, w) in HANDS.items()}
        contact = (dist["right"] < args.contact_m) | (dist["left"] < args.contact_m)
        T2 = T.copy()
        segs = runs(contact, args.min_frames)
        for a, e in segs:
            hand = "right" if np.median(dist["right"][a:e]) <= np.median(dist["left"][a:e]) else "left"
            ids, w = HANDS[hand]
            lo = max(0, a - args.anchor_window)
            t0 = lo + int(np.argmax(iou[lo:a + 1]))
            P0 = kp[t0, ids] - kp[t0, w]
            R0, c0 = T2[t0, :3, :3].copy(), T2[t0, :3, 3].copy()
            for t in range(max(a, t0 + 1), e):
                Rh = kabsch(P0, kp[t, ids] - kp[t, w])
                T2[t, :3, :3] = Rh @ R0
                if args.propagate_translation:
                    T2[t, :3, 3] = kp[t, w] + Rh @ (c0 - kp[t0, w])
        out = f"{args.v2d}/experiments/{args.out_exp}/{ep}"
        os.makedirs(out, exist_ok=True)
        b["pr"]["pose_abs"] = T2.astype(np.float32)
        torch.save(b, f"{out}/coconet.pth")
        print(f"{ep}: contact frames {contact.mean():.2f}, segments {len(segs)}")


if __name__ == "__main__":
    main()
