#!/usr/bin/env python
"""Ground-truth-free validation of a trajectory vs the baseline, per phase.
python validate.py --inputs /work/inputs --result /work/outputs/object_poses_world.npz [--diag solve_diag.npz]
Prints a markdown table and the depth cues used for the decision on the depth along the viewing ray."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import load_inputs, load_mesh, decimated_mesh, CovRenderer, mask_stats, load_mask_crops, crop_K, rot_angle_deg
import solve as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--diag", default="/work/outputs/solve_diag.npz")
    args = ap.parse_args()
    inp = args.inputs
    K, Twc, rec, cam = load_inputs(inp)
    Tcw = np.linalg.inv(Twc)
    Tb = rec["T_world_mesh"].astype(np.float64)
    Tr = np.load(args.result)["T_world_mesh"]
    F = len(Tb)
    g, V, Fc, UV, TEX = load_mesh(inp)
    Vd, Fd = decimated_mesh(V, Fc)
    cov = CovRenderer(Vd, Fd)
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"])
    n, d = plane[:3], plane[3]
    if n[1] > 0:
        n, d = -n, -d
    nn = np.linalg.norm(n)
    n, d = n / nn, d / nn
    kp = rec["human_keypoints_world"].astype(np.float64)
    hands = np.stack([kp[:, 21:42].mean(1), kp[:, 42:63].mean(1)], 1)
    area, bbox, cen = mask_stats(inp)
    visible = area > 300
    diag = np.load(args.diag) if os.path.exists(args.diag) else None

    def metrics(T):
        Tc = Tcw @ T
        anc = np.einsum("fij,j->fi", Tc[:, :3, :3], S.PC) + Tc[:, :3, 3]
        proj = (K @ anc.T).T
        proj = proj[:, :2] / proj[:, 2:]
        diam = K[0, 0] * 0.26 / np.clip(anc[:, 2], 1.0, 8.0)
        centers = np.where((area > 200)[:, None], (bbox[:, :2] + bbox[:, 2:]) / 2, proj)
        sizes = np.maximum(2.0 * diam, 1.5 * np.nan_to_num(np.max(bbox[:, 2:] - bbox[:, :2], 1)))
        obj, per, inim, boxes = load_mask_crops(inp, centers, sizes, 128)
        Kc = crop_K(K, boxes, 128)
        valid = inim * (1 - per)
        iou = np.full(F, np.nan)
        cdist = np.full(F, np.nan)
        with torch.no_grad():
            for b in range(0, F, 100):
                idx = np.arange(b, min(F, b + 100))
                s = cov.render(torch.tensor(Tc[idx, :3, :3], dtype=torch.float32, device="cuda"),
                               torch.tensor(Tc[idx, :3, 3], dtype=torch.float32, device="cuda"),
                               torch.tensor(Kc[idx], dtype=torch.float32, device="cuda"), 128)
                o = torch.tensor(obj[idx], device="cuda")
                vm = torch.tensor(valid[idx], device="cuda")
                inter = (torch.minimum(s, o) * vm).sum((1, 2))
                uni = (torch.maximum(s, o) * vm).sum((1, 2)).clamp_min(1e-6)
                iou[idx] = (inter / uni).cpu().numpy()
                yy, xx = torch.meshgrid(torch.arange(128, device="cuda").float(), torch.arange(128, device="cuda").float(),
                                        indexing="ij")
                ws, wo = (s * vm).sum((1, 2)).clamp_min(1), (o * vm).sum((1, 2)).clamp_min(1)
                dx = (o * vm * xx).sum((1, 2)) / wo - (s * vm * xx).sum((1, 2)) / ws
                dy = (o * vm * yy).sum((1, 2)) / wo - (s * vm * yy).sum((1, 2)) / ws
                cdist[idx] = (torch.sqrt(dx ** 2 + dy ** 2).cpu().numpy() * boxes[idx, 2] / 128)
        iou[~visible] = np.nan
        cdist[~visible] = np.nan
        # world quantities
        Rw = T[:, :3, :3]
        dR = np.r_[np.nan, rot_angle_deg(Rw[1:], Rw[:-1])]
        aw = np.einsum("fij,j->fi", Rw, S.PC) + T[:, :3, 3]
        acc = np.r_[np.nan, np.linalg.norm(aw[2:] - 2 * aw[1:-1] + aw[:-2], axis=1) * 100, np.nan]
        # lowest mesh point above the floor plane (cm)
        pts = np.einsum("fij,nj->fni", Rw, Vd[::7]) + T[:, None, :3, 3]
        hfl = (pts @ n + d).min(1) * 100      # n points up: positive = above floor
        tilt = np.degrees(np.arccos(np.clip(Rw[:, :, 1] @ n, -1, 1)))
        hd = np.linalg.norm(hands - aw[:, None], axis=2).min(1) * 100
        yaw = S.wyaw(Rw) % 360
        return dict(iou=iou, cdist=cdist, dR=dR, acc=acc, hfl=hfl, tilt=tilt, hd=hd, z=Tc[:, 2, 3], yaw=yaw, anc=aw)

    mb, mr = metrics(Tb), metrics(Tr)
    lugm = np.array([np.cos(np.radians(S.LUG_AZIMUTH_DEG)), 0, np.sin(np.radians(S.LUG_AZIMUTH_DEG))])

    def lugcos(T, m):
        ray = m["anc"] - Twc[:3, 3]
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        ld = np.einsum("fij,j->fi", T[:, :3, :3], lugm)
        return (ld * ray).sum(1) ** 2

    lb, lr = lugcos(Tb, mb), lugcos(Tr, mr)
    lug_on = np.zeros(F, bool)
    for a, b in S.LUG_SIDE_RANGES:
        lug_on[a:b + 1] = True

    def texz(m):
        if diag is None:
            return np.full(F, np.nan)
        Z = diag["Ztex"]
        ht = diag["has_tex"]
        j = np.round(m["yaw"] / 3).astype(int) % S.NB
        out = Z[np.arange(F), j]
        out[~ht] = np.nan
        return out

    tb, tr = texz(mb), texz(mr)
    nm_ = lambda x: f"{np.nanmean(x):.2f}" if np.isfinite(x).any() else "-"
    md_ = lambda x: f"{np.nanmedian(x):.2f}" if np.isfinite(x).any() else "-"
    rows = []
    hdr = ("| phase | frames | vis | IoU base | IoU ours | centroid err px base/ours | texture z base/ours | "
           "lug cos^2 base/ours | median dR deg/frame base/ours | median acc cm/f^2 base/ours | "
           "floor height cm base/ours | tilt vs floor deg base/ours | hand dist cm base/ours | "
           "dz mean/median cm (ours-base) |")
    rows.append(hdr)
    rows.append("|" + "---|" * hdr.count("|")[:-1] if False else "|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
    for nm, a, b, kind in S.PHASES + [("ALL", 0, F - 1, "")]:
        sl = slice(a, b + 1)
        vis = visible[sl]
        dz = (mr["z"][sl] - mb["z"][sl]) * 100
        fl = f"{md_(mb['hfl'][sl])}/{md_(mr['hfl'][sl])}" if kind == "static" else "-"
        lug = f"{nm_(np.where(lug_on[sl], lb[sl], np.nan))}/{nm_(np.where(lug_on[sl], lr[sl], np.nan))}"
        rows.append(f"| {nm} | {a}-{b} | {vis.sum()} | {nm_(mb['iou'][sl])} | {nm_(mr['iou'][sl])} | "
                    f"{md_(mb['cdist'][sl])}/{md_(mr['cdist'][sl])} | {nm_(tb[sl])}/{nm_(tr[sl])} | {lug} | "
                    f"{md_(mb['dR'][sl])}/{md_(mr['dR'][sl])} | {md_(mb['acc'][sl])}/{md_(mr['acc'][sl])} | {fl} | "
                    f"{md_(mb['tilt'][sl])}/{md_(mr['tilt'][sl])} | {md_(mb['hd'][sl])}/{md_(mr['hd'][sl])} | "
                    f"{np.mean(dz):.2f}/{np.median(dz):.2f} |")
    print("\n".join(rows))
    # extra stats
    print()
    for nm, a, b, kind in S.PHASES:
        sl = slice(a, b + 1)
        print(f"{nm}: yaw base circmean {np.degrees(np.angle(np.exp(1j*np.radians(mb['yaw'][sl])).mean()))%360:.0f} "
              f"(circ std {np.degrees(np.sqrt(-2*np.log(abs(np.exp(1j*np.radians(mb['yaw'][sl])).mean())))):.0f}), "
              f"ours {np.degrees(np.angle(np.exp(1j*np.radians(mr['yaw'][sl])).mean()))%360:.0f} "
              f"(circ std {np.degrees(np.sqrt(-2*np.log(abs(np.exp(1j*np.radians(mr['yaw'][sl])).mean())))):.0f}); "
              f"max dR/frame base {np.nanmax(mb['dR'][sl]):.1f} ours {np.nanmax(mr['dR'][sl]):.1f}; "
              f"ours-base rot diff median {np.median(rot_angle_deg(Tb[sl,:3,:3], Tr[sl,:3,:3])):.0f} deg, "
              f"pos diff median {np.median(np.linalg.norm(Tb[sl,:3,3]-Tr[sl,:3,3],axis=1))*100:.1f} cm")
    # depth cues for static floor phases: shift along the viewing ray that puts the lowest point on the floor
    print()
    for nm, a, b, kind in S.PHASES:
        if kind != "static":
            continue
        sl = slice(a, b + 1)
        T = Tr[a]
        Tc = Tcw @ T
        ray = Tc[:3, 3] / np.linalg.norm(Tc[:3, 3])
        rw = Twc[:3, :3] @ ray
        pts = (T[:3, :3] @ Vd.T).T + T[:3, 3]
        h0 = (pts @ n + d).min()
        dn = rw @ n          # change of height per metre moved along the ray
        shift = -h0 / dn if abs(dn) > 1e-6 else np.nan
        print(f"{nm}: lowest point {h0*100:.1f} cm above the released floor plane -> floor-contact cue: move "
              f"{shift*100:+.1f} cm along the viewing ray (ray . up = {dn:.3f}); "
              f"(floor plane tolerance from foot fit: median {0.0397*100:.1f} cm)")


if __name__ == "__main__":
    main()
