#!/usr/bin/env python3
"""Camera tilt / height from the floor (CPU, inside the v2d_cari4d image).

The baseline camera (JC2s1) is fitted to the frame-0 reference human and feet-on-floor. Here the floor seen by MoGe-2
gives an independent constraint: camera-frame floor points (aligned MoGe depth, back-projected with the pipeline K,
remapped pixel-preservingly to the shared K and scaled by the episode's depth factor, i.e. in the same camera frame as
our object and human) are fitted with a robust plane; that plane, mapped to the world, must coincide with the released
ground plane. This fixes the camera's tilt (2 DoF) and height (1 DoF); yaw and horizontal position stay as before.
Floor pixels: not person / object (dilated), and within --band m of the released plane under the current camera
(two passes, the second with the corrected camera).
Variants written per episode (T_world_cam):
  rot_cam   minimal rotation about the camera centre + height shift along the plane normal
  rot_hum   minimal rotation about the frame-0 reference human centroid + height shift
  rot_hum0  minimal rotation about the frame-0 reference human centroid, no height shift
Usage: camera_floor.py --inputs-root <agent_runs> --out <json> [--episodes ...]
"""
from __future__ import annotations

import argparse
import json

import cv2
import h5py
import numpy as np

V2D = "/mnt/nas/Group/hanzhou/v2d"


def pipeline_dir(src, ep):
    if src == "M1":
        return f"{V2D}/experiments/M1_prompt_v2/out/{ep}"
    import os
    p = f"{V2D}/outputs/cari4d/{ep}"
    return p if os.path.isfile(f"{p}/depth/moge2/aligned.h5") else f"{V2D}/outputs/cari4d_failed/{ep}_out"


def ransac_plane(X, it=400, thr=0.015, seed=0):
    rng = np.random.default_rng(seed)
    best, bi = None, -1
    for _ in range(it):
        p = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        d = -n @ p[0]
        k = (np.abs(X @ n + d) < thr).sum()
        if k > bi:
            best, bi = (n, d), k
    n, d = best
    inl = np.abs(X @ n + d) < thr
    for _ in range(3):  # least-squares refinement on inliers
        c = X[inl].mean(0)
        n = np.linalg.svd(X[inl] - c, full_matrices=False)[2][-1]
        d = -n @ c
        inl = np.abs(X @ n + d) < thr
    return n, d, inl.mean()


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-root", default=f"{V2D}/agent_runs", help="contains T1_ds_epXXX/inputs (exported s1 inputs)")
    ap.add_argument("--align", default=f"{V2D}/logs/JC2s1_shared_f820_pp_align.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--band", type=float, default=0.12)
    ap.add_argument("--step", type=int, default=4, help="pixel subsampling")
    a = ap.parse_args()
    al = json.load(open(a.align))
    res = {}
    for idx in a.episodes:
        ep = f"episode_{idx:06d}"
        inp = f"{a.inputs_root}/T1_ds_ep{idx:03d}/inputs"
        cam = json.load(open(f"{inp}/camera.json"))
        K, K0, W = (np.asarray(cam[k], np.float64) for k in ("K", "K_pipeline", "T_world_cam"))
        gp = json.load(open(f"{inp}/ground_plane.json"))["plane"]
        nw, dw = np.asarray(gp[:3], np.float64), float(gp[3])
        h0 = np.load(f"{inp}/frame0_reference_human.npz")["vertices_world"].astype(np.float64).mean(0)
        amed = float(al[str(idx)]["a_median"])
        pdir = pipeline_dir(al[str(idx)]["src"], ep)
        with h5py.File(f"{pdir}/depth/moge2/aligned.h5", "r") as f, h5py.File(f"{inp}/masks.h5", "r") as mf:
            ds = f["aligned/front_stereo_camera_left"]
            F = len(ds)
            pts = []
            ker = np.ones((25, 25), np.uint8)
            for t in np.linspace(0, F - 1, a.frames).astype(int):
                D = cv2.imdecode(np.frombuffer(ds[t].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000
                bad = cv2.dilate((mf["person"][t] | mf["object"][t]).astype(np.uint8), ker) > 0
                v, u = np.mgrid[0:D.shape[0]:a.step, 0:D.shape[1]:a.step]
                z = D[v, u]
                ok = (z > 0.3) & (z < 15) & ~bad[v, u]
                u, v, z = u[ok], v[ok], z[ok]
                # pixel-preserving K0 -> K remap and episode depth factor (same camera frame as the s1 object / human)
                zn = z * K[0, 0] / K0[0, 0] * amed
                X = np.c_[(u - K[0, 2]) / K[0, 0] * zn, (v - K[1, 2]) / K[1, 1] * zn, zn]
                pts.append(X)
            X = np.concatenate(pts)
        Wc = W.copy()
        for _ in range(2):
            Xw = X @ Wc[:3, :3].T + Wc[:3, 3]
            cand = np.abs(Xw @ nw + dw) < a.band
            if cand.sum() < 500:
                break
            nc, dc, frac = ransac_plane(X[cand])
            if (Wc[:3, :3] @ nc) @ nw < 0:
                nc, dc = -nc, -dc
            # world normal of the observed floor under the baseline camera, and the correction
            n_obs = W[:3, :3] @ nc
            Ra = rot_between(n_obs, nw)
            C = W[:3, 3]
            out = {}
            for name, piv, hfix in (("rot_cam", C, True), ("rot_hum", h0, True), ("rot_hum0", h0, False)):
                Wn = np.eye(4)
                Wn[:3, :3] = Ra @ W[:3, :3]
                Wn[:3, 3] = Ra @ (C - piv) + piv
                if hfix:  # plane n_w.X + (d_c - n_w.C) = 0 must equal n_w.X + d_w = 0
                    Wn[:3, 3] += nw * ((dc - nw @ Wn[:3, 3]) - dw)
                out[name] = Wn
            Wc = out["rot_cam"]
        tilt = float(np.degrees(np.arccos(np.clip(n_obs @ nw, -1, 1))))
        hgt_err = float((dc - nw @ C) - dw)  # >0: observed floor implies the camera should move along +n_w
        res[str(idx)] = {"tilt_deg": tilt, "height_shift_m": hgt_err, "n_points": int(cand.sum()), "inlier_frac": float(frac),
                         "a_median": amed, **{k: v.tolist() for k, v in out.items()}}
        print(f"{idx:2d} floor pts {cand.sum():6d} inl {frac:.2f} | tilt corr {tilt:5.2f} deg | height corr {hgt_err*100:+6.1f} cm", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
