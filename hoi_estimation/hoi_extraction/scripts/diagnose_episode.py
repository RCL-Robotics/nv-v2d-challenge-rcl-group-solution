#!/usr/bin/env python3
"""Per-episode failure diagnosis for CARI4D object poses (runs inside v2d_cari4d image, CPU only).

For each episode it reports and plots:
  prompt   GroundingDINO prompt frame/box for the object
  masks    SAM2 object-mask area vs the reference visibility flag
  fp       FoundationPose tracking-state / emitted-pose-source statistics
  errors   per-frame rotation/position error after a ROBUST camera->world rotation
           (RANSAC over per-frame rotation offsets, so failed segments do not bias it)
  symmetry rotation error modulo 180 deg flips about the mesh principal axes, and the
           angle between principal axes (small axis error + large rotation error => symmetry)
Diagnostic only: it uses the Tier 1 reference poses.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


def q2R(q):  # wxyz
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def ang(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.einsum("...ij,...ij->...", Ra, Rb) - 1) / 2, -1, 1)))


def project_so3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def robust_offset(R_ours, R_ref, thresh=20.0, iters=400, rng=np.random.default_rng(0)):
    """Camera->world rotation C maximising frames with angle(C R_ours, R_ref) < thresh."""
    cand = R_ref @ np.transpose(R_ours, (0, 2, 1))
    best, best_in = None, -1
    for t in rng.choice(len(cand), min(iters, len(cand)), replace=False):
        inl = ang(cand[t][None] @ R_ours, R_ref) < thresh
        if inl.sum() > best_in:
            best, best_in = inl, inl.sum()
    C = project_so3(cand[best].sum(0))
    return C, best.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", type=int, nargs="+")
    ap.add_argument("--v2d", default=os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"))
    ap.add_argument("--local-run", default=os.path.expanduser("~/Projects/v2d/run"))
    ap.add_argument("--out", default=os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d") + "/diagnosis")
    args = ap.parse_args()
    ds = f"{args.v2d}/dataset/track_2/tier_1_multiview_caption"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
    os.makedirs(args.out, exist_ok=True)
    summary = []
    for idx in args.episodes:
        ep = f"episode_{idx:06d}"
        root = f"{args.v2d}/outputs/cari4d/{ep}"
        work = f"{root}/work"
        if not os.path.isdir(f"{root}/foundationpose"):  # failed episodes stay on local disk
            root, work = f"{args.local_run}/out/{ep}", f"{args.local_run}/work/{ep}"
        obj = meta[idx]["object"]
        ref = np.load(f"{args.v2d}/ref_poses/{ep}.npz")
        pose, vis = ref["pose_xyz_wxyz"].astype(np.float64), ref["visible"].astype(bool)
        ok = np.linalg.norm(pose[:, 3:], axis=1) > 0.5
        pose[~ok, 3] = 1
        t_ref, R_ref = pose[:, :3], q2R(pose[:, 3:])

        glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
        S, ext = trimesh.bounds.oriented_bounds(glb)
        S = np.asarray(S, dtype=np.float64)
        # principal axes of the object in GLB coordinates = rows of S's rotation
        axes_glb = S[:3, :3]
        ext = np.asarray(ext)

        fpk = pickle.load(open(glob.glob(f"{root}/foundationpose/*.pkl")[0], "rb"))
        poses = {"FP": np.asarray(fpk["fp_poses_cam"])[:, 0].astype(np.float64)}
        if os.path.isfile(f"{root}/inference/coconet.pth"):
            b = torch.load(f"{root}/inference/coconet.pth", map_location="cpu", weights_only=False)
            poses["CoCoNet"] = np.asarray(b["pr"]["pose_abs"], dtype=np.float64)
        n = min(len(t_ref), *(len(p) for p in poses.values()))

        # masks
        mfiles = sorted(glob.glob(f"{work}/masks/1/*.png"))
        area = np.array([int((cv2.imread(f, 0) > 0).sum()) for f in mfiles[:n]]) if mfiles else np.zeros(n)
        prompts = json.load(open(f"{work}/prompts.json"))["prompts"] if os.path.isfile(f"{work}/prompts.json") else []
        pobj = [p for p in prompts if p["object_id"] == 1]

        # FP diagnostics
        state = np.asarray(fpk["fp_tracking_state"])[:n]
        src = np.asarray(fpk["fp_emitted_pose_source"])[:n]
        iou = np.asarray(fpk["fp_selected_iou"], dtype=float)[:n]

        rec = {"ep": idx, "object": obj, "frames": n, "extents_m": np.round(np.sort(ext), 3).tolist(),
               "prompt_frame": pobj[0]["frame_index"] if pobj else None,
               "ref_visible_frac": round(float(vis[:n].mean()), 3),
               "mask_empty_when_visible_frac": round(float(((area[:n] == 0) & vis[:n]).sum() / max(vis[:n].sum(), 1)), 3),
               "obj_motion_m": round(float(np.linalg.norm(t_ref[:n] - t_ref[0], axis=1).max()), 2),
               "fp_state_frac": {int(k): round(float((state == k).mean()), 3) for k in np.unique(state)},
               "fp_source_frac": {int(k): round(float((src == k).mean()), 3) for k in np.unique(src)},
               "fp_iou_median": round(float(np.nanmedian(iou)), 3)}
        fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        fr = np.arange(n)
        for name, P in poses.items():
            T = P[:n] @ S  # cam_T_glb
            R_o, t_o = T[:, :3, :3], T[:, :3, 3]
            keep = ok[:n] & np.isfinite(T).all(axis=(1, 2))
            C, inl = robust_offset(R_o[keep], R_ref[:n][keep])
            Rw = C @ R_o
            rot = ang(Rw, R_ref[:n])
            flips = [np.eye(3)] + [axes_glb.T @ np.diag(d) @ axes_glb for d in ([1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
            rot_flip = np.min([ang(Rw @ F, R_ref[:n]) for F in flips], axis=0)
            axis_err = []  # per principal axis: angle between our and reference axis direction (sign-free)
            for k in range(3):
                a = axes_glb[k]
                da = np.abs(np.einsum("tij,j,tik,k->t", Rw, a, R_ref[:n], a))
                axis_err.append(np.degrees(np.arccos(np.clip(da, -1, 1))))
            tw = np.median((t_ref[:n] - np.einsum("ij,tj->ti", C, t_o))[keep], axis=0)
            pos = np.linalg.norm(np.einsum("ij,tj->ti", C, t_o) + tw - t_ref[:n], axis=1)
            k_long = int(np.argmax(ext))
            rec[name] = {"inlier_frac_20deg": round(float(inl), 3),
                         "rot_err_med": round(float(np.median(rot[keep])), 1),
                         "rot_err_flipaware_med": round(float(np.median(rot_flip[keep])), 1),
                         "long_axis_err_med": round(float(np.median(axis_err[k_long][keep])), 1),
                         "pos_err_med_cm": round(float(np.median(pos[keep]) * 100), 1),
                         "frac_rot_lt30": round(float((rot[keep] < 30).mean()), 3)}
            axs[0].plot(fr, rot, label=f"{name} rot err", lw=0.8)
            axs[0].plot(fr, rot_flip, "--", label=f"{name} flip-aware", lw=0.8)
            axs[1].plot(fr, axis_err[k_long], label=f"{name} long-axis err", lw=0.8)
            axs[2].plot(fr, pos * 100, label=f"{name} pos err (cm)", lw=0.8)
        axs[0].set_ylabel("deg"); axs[0].legend(fontsize=7); axs[0].set_title(f"ep{idx} {obj}  extents={rec['extents_m']}")
        axs[1].set_ylabel("deg"); axs[1].legend(fontsize=7)
        axs[2].set_ylabel("cm"); axs[2].set_yscale("symlog", linthresh=10); axs[2].legend(fontsize=7)
        axs[3].plot(fr, area / max(area.max(), 1), label="obj mask area (norm)", lw=0.8)
        axs[3].plot(fr, vis[:n] * 1.05, label="ref visible", lw=0.8)
        axs[3].plot(fr, state / max(state.max(), 1) * 0.9, label="FP tracking state", lw=0.8)
        axs[3].plot(fr, np.nan_to_num(iou), label="FP selected IoU", lw=0.6, alpha=0.7)
        if pobj:
            axs[3].axvline(pobj[0]["frame_index"], color="k", ls=":", label="prompt frame")
        axs[3].legend(fontsize=7); axs[3].set_xlabel("frame")
        fig.tight_layout(); fig.savefig(f"{args.out}/{ep}_{obj}.png", dpi=80); plt.close(fig)
        summary.append(rec)
        print(json.dumps(rec))
    json.dump(summary, open(f"{args.out}/summary_{'_'.join(map(str, args.episodes))}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
