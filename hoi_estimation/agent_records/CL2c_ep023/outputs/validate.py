#!/usr/bin/env python
"""Ground-truth-free validation of trajectories: per-phase silhouette IoU (person pixels excluded), held-out-frame
IoU, smoothness, floor contact / tilt, hand-object contact, and camera-depth differences w.r.t. the baseline.
Usage: python validate.py --cand name=path.npz [...] [--diag diag.npz]"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve import (FloorModel, Renderer, build_crops, load_inputs, load_mesh, render_alpha, RES)


def iou_all(R, Tcw, Tw, M, V, Kc, dil):
    out = []
    with torch.no_grad():
        for s in range(0, len(Tw), 48):
            Tc = torch.tensor(Tcw @ Tw[s:s + 48], dtype=torch.float32, device="cuda")
            al = render_alpha(R, Tc, torch.tensor(Kc[s:s + 48], device="cuda"), RES)
            if dil:
                al = torch.nn.functional.max_pool2d(al[:, None], 2 * dil + 1, 1, dil)[:, 0]
            al = (al > 0.5).float()
            m = torch.tensor(M[s:s + 48], device="cuda") > 0.5; v = torch.tensor(V[s:s + 48], device="cuda") > 0.5
            a = al > 0.5
            out.append((((a & m & v).sum((1, 2))).float() / ((a | m) & v).sum((1, 2)).clamp(min=1).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--cand", nargs="+", default=[])
    ap.add_argument("--diag", default=None)
    ap.add_argument("--phases", default="0,135,340,460,878,946")
    args = ap.parse_args()
    cam, K, Twc, rec, plane = load_inputs(args.inputs)
    mesh = load_mesh(args.inputs); mv = np.asarray(mesh.vertices)
    Tcw = np.linalg.inv(Twc); o = Twc[:3, 3]
    Tb = rec["T_world_mesh"].astype(np.float64); F = len(Tb)
    nrm = plane[:3] / np.linalg.norm(plane[:3]); g = -nrm
    dh = np.mean([T[:3, :3].T @ g for T in Tb], 0); dh /= np.linalg.norm(dh)
    fm = FloorModel(mesh, plane, dh)
    cands = {"baseline": Tb}
    for c in args.cand:
        k, p = c.split("=")
        cands[k] = np.load(p)["T_world_mesh"].astype(np.float64)
    # crops around the baseline (same crops for every candidate)
    with h5py.File(os.path.join(args.inputs, "masks.h5"), "r") as mf:
        M, V, Kc, _ = build_crops(mf, range(F), Tcw @ Tb, K, mv)
    R = Renderer(mesh)
    pb = [int(x) for x in args.phases.split(",")]
    names = ["static_start", "push_away", "turn_occluded", "return_back_forth", "static_end"]
    phases = list(zip(names, pb[:-1], pb[1:])) + [("ALL", 0, F)]
    kp = rec["human_keypoints_world"].astype(np.float64)
    vs = mv[::10]
    zb = (Tcw @ Tb)[:, 2, 3]
    rows = {}
    for name, T in cands.items():
        iou3 = iou_all(R, Tcw, T, M, V, Kc, 3)
        iou0 = iou_all(R, Tcw, T, M, V, Kc, 0)
        Tc = Tcw @ T
        z = Tc[:, 2, 3]
        # smoothness: second differences
        acc_t = np.linalg.norm(T[2:, :3, 3] - 2 * T[1:-1, :3, 3] + T[:-2, :3, 3], axis=1) * 100
        dR = np.array([Rot.from_matrix(T[i, :3, :3].T @ T[i + 1, :3, :3]).as_rotvec() for i in range(F - 1)])
        acc_r = np.degrees(np.linalg.norm(dR[1:] - dR[:-1], axis=1))
        # floor: lowest vertex height above plane, tilt of support normal w.r.t. gravity
        hmin = np.array([(mv[::20] @ T[i, :3, :3].T + T[i, :3, 3]) @ nrm + plane[3] / np.linalg.norm(plane[:3]) for i in range(F)]).min(1)
        tilt = np.array([np.degrees(np.arccos(np.clip((T[i, :3, :3] @ fm.dm) @ g, -1, 1))) for i in range(F)])
        # hand contact (frames with high contact prob)
        hd = np.full(F, np.nan)
        cp = rec["contact_prob_right_left"].max(1)
        for i in range(0, F, 2):
            if cp[i] > 0.5:
                tr = cKDTree(vs @ T[i, :3, :3].T + T[i, :3, 3])
                hd[i] = tr.query(kp[i, 21:63])[0].min()
        rows[name] = []
        for pn, a, b in phases:
            sl = slice(a, b)
            rows[name].append(dict(phase=pn, iou3=iou3[sl].mean(), iou0=iou0[sl].mean(), iou3_p10=np.percentile(iou3[sl], 10),
                                   acc_t=np.median(acc_t[max(0, a):max(0, b - 2)]) if b - a > 3 else np.nan,
                                   acc_r=np.median(acc_r[max(0, a):max(0, b - 2)]) if b - a > 3 else np.nan,
                                   hmin_med=np.median(hmin[sl]) * 100, hmin_std=np.std(hmin[sl]) * 100,
                                   tilt=np.median(tilt[sl]), hand=np.nanmedian(hd[sl]) * 100 if np.isfinite(hd[sl]).any() else np.nan,
                                   dz_mean=np.mean(z[sl] - zb[sl]) * 100, dz_med=np.median(z[sl] - zb[sl]) * 100))
    hdr = "| phase (frames) | trajectory | IoU dil3 | IoU p10 | IoU raw | accel t cm/f^2 | accel R deg/f^2 | lowest pt h cm (med / std) | tilt deg | hand-obj cm | dz vs base mean / med cm |"
    print(hdr); print("|" + "---|" * 11)
    for pi, (pn, a, b) in enumerate(phases):
        for name in cands:
            r = rows[name][pi]
            print(f"| {pn} ({a}-{b - 1}) | {name} | {r['iou3']:.3f} | {r['iou3_p10']:.3f} | {r['iou0']:.3f} | {r['acc_t']:.2f} | {r['acc_r']:.2f} | "
                  f"{r['hmin_med']:.1f} / {r['hmin_std']:.1f} | {r['tilt']:.1f} | {r['hand']:.1f} | {r['dz_mean']:.1f} / {r['dz_med']:.1f} |")

    # depth cues for the last candidate (global scale s along the viewing rays; s=1 = as submitted)
    name = list(cands)[-1]; T = cands[name]; Tc = Tcw @ T
    zmed = np.median(Tc[:, 2, 3])
    Dcam = nrm @ o + plane[3] / np.linalg.norm(plane[:3])
    lowest = np.array([((mv[::5] @ T[i, :3, :3].T + T[i, :3, 3]) @ nrm + plane[3] / np.linalg.norm(plane[:3])).min() for i in range(F)])
    stat = np.r_[0:pb[1], pb[-2]:F]
    h_f = np.median(lowest[stat])
    soles = (np.load(os.path.join(args.inputs, "frame0_reference_human.npz"))["vertices_world"] @ nrm + plane[3] / np.linalg.norm(plane[:3])).min()
    print(f"\nDepth cues for '{name}' (median camera z {zmed:.3f} m; static-phase lowest point {h_f * 100:.1f} cm above the released plane):")
    for cue, ht in [("untouched object rests on released floor plane (h=0)", 0.0), (f"floor height from true frame-0 human soles (h={soles * 100:.1f} cm)", soles)]:
        sc = (Dcam - ht) / (Dcam - h_f)
        print(f"  {cue}: scale {sc:.4f} -> depth shift {100 * (sc - 1) * zmed:+.1f} cm at the median depth")
    a, b = pb[2], pb[3]
    res = []
    for sc in np.arange(0.97, 1.0301, 0.005):
        dd = []
        for i in range(a, b, 2):
            Ti = Twc @ np.vstack([np.hstack([Tc[i, :3, :3], sc * Tc[i, :3, 3:]]), [0, 0, 0, 1]])
            dd.append(cKDTree(vs @ Ti[:3, :3].T + Ti[:3, 3]).query(kp[i, 21:63])[0].min())
        res.append((sc, np.median(dd)))
    ok = [r[0] for r in res if r[1] < 0.01]
    print("  hand contact in the turn phase (human not attached to the baseline object there): median hand-object distance by scale: "
          + ", ".join(f"{r[0]:.3f}:{r[1] * 100:.1f}cm" for r in res))
    if ok:
        print(f"  -> contact (<1 cm) for scales {min(ok):.3f}..{max(ok):.3f}, i.e. depth shift {100 * (min(ok) - 1) * zmed:+.1f}..{100 * (max(ok) - 1) * zmed:+.1f} cm")

    # held-out test: smooth the per-frame registrations of even frames only, predict odd frames
    if args.diag:
        dg = np.load(args.diag)
        from solve import gaussian_smooth
        reg = dg["reg"]; w = dg["w"]
        ev = np.arange(F) % 2 == 0
        P = reg.copy()
        for k in range(3):
            P[:, k] = gaussian_smooth(reg[:, k], w * ev, 2.0)
        Tho = np.array([fm.pose(*p, h0=float(dg["h0"])) for p in P])
        ih = iou_all(R, Tcw, Tho, M, V, Kc, 3)
        ib = iou_all(R, Tcw, Tb, M, V, Kc, 3)
        odd = ~ev
        print("\nHeld-out (odd frames predicted from even-frame registrations only), IoU dil3:")
        for pn, a, b in phases:
            sel = odd.copy(); sel[:a] = False; sel[b:] = False
            print(f"  {pn:18s} held-out {ih[sel].mean():.3f}  baseline {ib[sel].mean():.3f}  frames where held-out better: {np.mean(ih[sel] > ib[sel]) * 100:.0f}%")


if __name__ == "__main__":
    main()
