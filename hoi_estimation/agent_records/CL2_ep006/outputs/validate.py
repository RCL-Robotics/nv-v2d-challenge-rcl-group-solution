#!/usr/bin/env python
"""Ground-truth-free validation: baseline vs result, per phase. Prints a markdown table.
usage: python validate.py --inputs /work/inputs --result /work/outputs/object_poses_world.npz [--extra name=path.npy ...]"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_fit import setup, Evidence, SilRenderer, decimate, motion_score, detect_phases, rot_axis
from hoi_validate import FlowCheck, read_gray_frames, mask_iou


def yaw_of(pm, T):
    return np.array([pm.params_from_pose(t)[0] for t in T])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    S = setup(args.inputs); pm = S["pm"]; F_ = len(S["Tb"])
    E = Evidence(S, scale=0.5, inputs=args.inputs)
    sr = SilRenderer(decimate(S["mesh"]), S["K"], S["Twc"], scale=0.5)
    m = motion_score(E.G, E.MO, E.MP)
    seg = detect_phases(m, S["rec"]["contact_prob_right_left"].max(1))
    s1 = np.nonzero(seg == 0)[0][-1] + 1; s2 = np.nonzero(seg == 1)[0][0]
    bad = np.nonzero(~S["good"])[0]
    phases = [("static-1", 0, s1), ("moving (front)", s1, bad[0]), ("moving (baseline failure)", bad[0], bad[-1] + 1),
              ("moving (back)", bad[-1] + 1, s2), ("static-2", s2, F_), ("all", 0, F_)]
    cands = {"baseline": S["Tb"], "result": np.load(args.result)["T_world_mesh"]}
    for e in args.extra:
        k, v = e.split("="); cands[k] = np.load(v)
    grays = read_gray_frames(f"{args.inputs}/video.mp4")
    fc = FlowCheck(S["mesh"], S["K"], S["Twc"], grays, f"{args.inputs}/masks.h5", lag=4, step=2)
    Tcw = np.linalg.inv(S["Twc"])
    zb = np.einsum("ij,fjk->fik", Tcw, S["Tb"])[:, 2, 3]
    up = S["up"]
    rows = []
    for name, T in cands.items():
        fl = fc.errors(T)
        iou = mask_iou(sr, E, T)
        z = np.einsum("ij,fjk->fik", Tcw, T)[:, 2, 3]
        dn = np.einsum("fij,j->fi", T[:, :3, :3], S["nr"])
        tilt = np.degrees(np.arccos(np.clip(dn @ (-up), -1, 1)))
        yaw = np.unwrap(yaw_of(pm, T))
        dyaw = np.degrees(np.abs(np.diff(yaw)))
        pos = T[:, :3, :3] @ pm.c + T[:, :3, 3]
        acc = np.linalg.norm(pos[2:] - 2 * pos[1:-1] + pos[:-2], axis=1) * 100
        h = pos @ up + S["floor_d"]
        for ph, s, e in phases:
            ks = [t for t in fl if s <= t < e]
            rows.append(dict(traj=name, phase=f"{ph} [{s}-{e - 1}]",
                             flow=np.mean([min(fl[t][0], 10) for t in ks]) if ks else np.nan,
                             iou=iou[s:e].mean(), tilt=np.median(tilt[s:e]),
                             yaw_step=np.percentile(dyaw[s:max(s + 1, e - 1)], 95),
                             acc=np.percentile(acc[s:max(s + 1, e - 2)], 95),
                             dz_mean=(z[s:e] - zb[s:e]).mean() * 100, dz_med=np.median(z[s:e] - zb[s:e]) * 100,
                             h=np.median(h[s:e])))
    print("| trajectory | phase | flow err px (lower better) | mask IoU | tilt deg (median) | 95% |dyaw|/frame deg | 95% accel cm/f^2 | dz vs baseline mean / median cm | base height above floor m |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["phase"], r["traj"])):
        print(f"| {r['traj']} | {r['phase']} | {r['flow']:.2f} | {r['iou']:.3f} | {r['tilt']:.1f} | {r['yaw_step']:.2f} | {r['acc']:.2f} | {r['dz_mean']:.1f} / {r['dz_med']:.1f} | {r['h']:.3f} |")


if __name__ == "__main__":
    main()
