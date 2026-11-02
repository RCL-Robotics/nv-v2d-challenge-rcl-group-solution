#!/usr/bin/env python
"""Ground-truth-free validation of a trajectory vs the baseline (writes a markdown table).

usage: python validate.py --inputs /work/inputs --result object_poses_world.npz --a_end A --b_start B
Metrics per phase:
  IoU      : rendered silhouette vs SAM2 object mask (half res, person-occluded pixels ignored) [fitted cue]
  chamf    : full-res Canny-edge chamfer (px) of rendered outer + depth-discontinuity edges -> image edges,
             every 6th frame [partly independent: fit used half-res outer silhouette only]
  jitR     : median |second difference| of rotation (deg/frame^2) -- temporal noise
  jitT     : median |second difference| of the mesh-centre position (mm/frame^2)
  spread   : static phases only, median ADD (cm) of each frame to the phase's mean pose (0 = rigidly static)
  floor    : static phases only, median over frames of the lowest mesh point's signed height above the
             released floor plane (cm); the true frame-0 human's lowest point is at -2.9 cm
  hand     : contact frames, median distance (cm) of the closest hand keypoint to the mesh surface
  dz       : mean / median camera-frame depth difference result - baseline (cm)
  ADDb     : mean ADD (cm) between result and baseline
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import Renderer, load_inputs, read_video, rotdeg, add_err, robust_mean_pose


def edge_chamfer(rf, T, img, person, trunc=10):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 1.2)
    e = cv2.Canny(g, 30, 80)
    dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 5)
    m, z, _ = rf.render(T[None], need_rgb=False)
    m = m[0].cpu().numpy().astype(np.uint8)
    z = z[0].cpu().numpy()
    c = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    zz = np.where(m > 0, z, 100.0)
    gx = np.abs(np.diff(zz, axis=1, prepend=zz[:, :1]))
    gy = np.abs(np.diff(zz, axis=0, prepend=zz[:1]))
    c |= ((gx > 0.03) | (gy > 0.03)) & (m > 0)
    c &= ~person
    return np.minimum(dt[c], trunc).mean() if c.sum() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", required=True)
    ap.add_argument("--a_end", type=int, required=True)
    ap.add_argument("--b_start", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K, Twc, rec, mesh, plane = load_inputs(args.inputs)
    Tcw = np.linalg.inv(Twc)
    F = len(rec["T_world_mesh"])
    cams = {"baseline": np.einsum("ij,fjk->fik", Tcw, rec["T_world_mesh"].astype(float)),
            "result": np.einsum("ij,fjk->fik", Tcw, np.load(args.result)["T_world_mesh"])}
    phases = {"A static": np.arange(0, args.a_end + 1), "moving": np.arange(args.a_end + 1, args.b_start),
              "B static": np.arange(args.b_start, F), "all": np.arange(F)}
    V = np.asarray(mesh.vertices)
    pts = V[np.random.default_rng(0).choice(len(V), 2000, replace=False)]
    surf, _ = __import__("trimesh").sample.sample_surface_even(mesh, 20000, seed=0)
    tree = cKDTree(surf)

    hf = h5py.File(f"{args.inputs}/masks.h5", "r")
    rd = Renderer(mesh, K, 0.5)
    rf = Renderer(mesh, K, 1.0)
    # IoU every 2nd frame, chamfer every 6th frame
    iou = {k: np.full(F, np.nan) for k in cams}
    cham = {k: np.full(F, np.nan) for k in cams}
    idx6 = list(range(0, F, 6))
    imgs = dict(zip(idx6, read_video(f"{args.inputs}/video.mp4", idx6)))
    for i in range(0, F, 2):
        om = hf["object"][i].reshape(576, 2, 768, 2).mean((1, 3)) > 0.5
        pm = hf["person"][i].reshape(576, 2, 768, 2).any((1, 3))
        v = ~(pm & ~om)
        for k, T in cams.items():
            m, _, _ = rd.render(T[i:i + 1], need_rgb=False)
            m = m[0].cpu().numpy()
            iou[k][i] = ((m & om) & v).sum() / max(((m | om) & v).sum(), 1)
            if i in imgs:
                person = cv2.dilate(hf["person"][i].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                cham[k][i] = edge_chamfer(rf, T[i], imgs[i], person)

    n, d = plane[:3], plane[3]
    kp = rec["human_keypoints_world"].astype(float)
    cp = rec["contact_prob_right_left"]
    rows = []
    for k, T in cams.items():
        Tw = np.einsum("ij,fjk->fik", Twc, T)
        Rr = T[:, :3, :3]
        ang = rotdeg(Rr[1:], Rr[:-1])
        cen = np.einsum("fij,j->fi", T[:, :3, :3], V.mean(0)) + T[:, :3, 3]
        for pn, ph in phases.items():
            ph2 = ph[(ph > 0) & (ph < F - 1)]
            jitR = np.median(np.abs(ang[ph2] - ang[ph2 - 1]))
            acc = cen[ph2 + 1] - 2 * cen[ph2] + cen[ph2 - 1]
            jitT = np.median(np.linalg.norm(acc, axis=1)) * 1000
            spread = floor = np.nan
            if "static" in pn:
                M = robust_mean_pose(T[ph])
                spread = np.median(add_err(T[ph], M[None], pts)) * 100
                floor = np.median([((V @ Tw[i, :3, :3].T + Tw[i, :3, 3]) @ n + d).min() for i in ph[::5]]) * 100
            hd = []
            for i in ph[::3]:
                for h, (lo, hi) in enumerate([(21, 42), (42, 63)]):
                    if cp[i, h] > 0.5:
                        Ti = np.linalg.inv(Tw[i])
                        p = kp[i, lo:hi] @ Ti[:3, :3].T + Ti[:3, 3]
                        hd.append(tree.query(p)[0].min())
            hand = np.median(hd) * 100 if hd else np.nan
            dz = (T[ph, 2, 3] - cams["baseline"][ph, 2, 3]) * 100
            addb = add_err(T[ph], cams["baseline"][ph], pts).mean() * 100
            rows.append((pn, k, np.nanmean(iou[k][ph]), np.nanmean(cham[k][ph]), jitR, jitT, spread, floor, hand,
                         dz.mean(), np.median(dz), addb))
    rows.sort(key=lambda r: list(phases).index(r[0]))
    hdr = "| phase | traj | IoU | chamf px | jitR deg | jitT mm | spread cm | floor cm | hand cm | dz mean cm | dz med cm | ADD to base cm |"
    lines = [hdr, "|" + "---|" * 12]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.2f} | {r[4]:.3f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.1f} | "
                     f"{r[8]:.2f} | {r[9]:+.2f} | {r[10]:+.2f} | {r[11]:.2f} |")
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        open(args.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
