#!/usr/bin/env python
"""Ground-truth-free validation: baseline vs result, per phase.
Usage: python validate.py --inputs /work/inputs --result object_poses_world.npz --diag diag.npz
Metrics: mask IoU of the visible part (render minus person vs SAM2 object mask, 320px crops), photometric residual,
held-out IoU (trajectory fitted with every other frame removed, evaluated on the removed frames), angular jitter,
rotation change, camera-depth difference, floor distance of the lowest mesh point."""
import argparse, json, os, sys
import numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_mesh, Renderer, load_camera, iter_video
from evidence import Evidence, score_and_refine
from trajectory import smooth_se3
from scipy.spatial.transform import Rotation as Rot


def scores(r, ev, Tc):
    iou = np.zeros(len(Tc)); ph = np.zeros(len(Tc))
    for k in range(0, len(Tc), 60):
        ids = np.arange(k, min(len(Tc), k + 60))
        _, i, _, p = score_and_refine(r, ev, ids, Tc[ids], iters=0, lam=0.3)
        iou[ids] = i; ph[ids] = p
    return iou, ph


def jitter(Tc):
    R = Rot.from_matrix(Tc[:, :3, :3]); w = (R[:-1].inv() * R[1:]).as_rotvec()
    return np.degrees(np.linalg.norm(w[1:] - w[:-1], axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--heldout", type=int, default=1)
    a = ap.parse_args()
    K, Kp, Twc = load_camera(a.inputs)
    Tw0 = np.load(f"{a.inputs}/reconstruction.npz")["T_world_mesh"].astype(np.float64)
    Tw1 = np.load(a.result)["T_world_mesh"]
    Tc0 = np.linalg.inv(Twc)[None] @ Tw0; Tc1 = np.linalg.inv(Twc)[None] @ Tw1
    d = np.load(a.diag)
    phases = [tuple(p) for p in d["phases"]]
    mesh = load_mesh(f"{a.inputs}/object_mesh.glb"); r = Renderer(mesh, K)
    ev = Evidence(a.inputs, iter_video(f"{a.inputs}/video.mp4"), r, K, Tc0, S=320)
    iou0, ph0 = scores(r, ev, Tc0); iou1, ph1 = scores(r, ev, Tc1)
    vis = d["vis"]; good = (ev.area > 200) & (vis > 0.3)
    res = {}
    if a.heldout:
        # held-out: drop odd frames, refit the trajectory from the even ones only, score on the odd ones
        F = len(Tc0); odd = np.zeros(F, bool); odd[1::2] = True
        w = d["w"].copy(); w[odd] = 0
        fixed = {}
        (s0, s1), _, (e0, e1) = phases
        for k in range(s0, s1): fixed[k] = d["Ts"][s0]
        for k in range(e0, e1): fixed[k] = d["Ts"][e0]
        lam = float(d["lam"])
        Hb = smooth_se3(Tc0, w, fixed={k: Tc0[k] for k in fixed}, lam_rot=lam, lam_tr=lam, iters=1500)
        Hr = smooth_se3(d["Tref"], w, fixed=fixed, lam_rot=lam, lam_tr=lam, iters=1500)
        ihb, _ = scores(r, ev, Hb); ihr, _ = scores(r, ev, Hr)
    names = ["static start", "carried", "static end"]
    pl = json.load(open(f"{a.inputs}/ground_plane.json"))["plane"]; n = np.array(pl[:3]); dd = pl[3]
    v = np.asarray(mesh.vertices)[::10]
    rows = []
    for name, (p0, p1) in zip(names, phases):
        sel = np.zeros(len(Tc0), bool); sel[p0:p1] = True
        g = sel & good
        dz = 100 * (Tc1[p0:p1, 2, 3] - Tc0[p0:p1, 2, 3])
        dR = Rot.from_matrix(np.einsum('nji,njk->nik', Tc0[p0:p1, :3, :3], Tc1[p0:p1, :3, :3])).magnitude()
        row = dict(phase=name, frames=f"{p0}-{p1-1}", n=int(p1 - p0),
                   iou_base=iou0[g].mean(), iou_res=iou1[g].mean(), photo_base=ph0[g].mean(), photo_res=ph1[g].mean(),
                   jit_base=jitter(Tc0[p0:p1]).mean() if p1 - p0 > 2 else 0, jit_res=jitter(Tc1[p0:p1]).mean() if p1 - p0 > 2 else 0,
                   dR_mean=np.degrees(dR).mean(), dt_mean_cm=100 * np.linalg.norm(Tc1[p0:p1, :3, 3] - Tc0[p0:p1, :3, 3], axis=1).mean(),
                   dz_mean_cm=dz.mean(), dz_median_cm=np.median(dz))
        if a.heldout:
            go = g & odd
            row.update(ho_base=ihb[go].mean(), ho_res=ihr[go].mean())
        # floor distance of the lowest mesh point (static phases)
        fl0 = [((Tw0[i, :3, :3] @ v.T).T + Tw0[i, :3, 3]) @ n + dd for i in range(p0, p1, 5)]
        fl1 = [((Tw1[i, :3, :3] @ v.T).T + Tw1[i, :3, 3]) @ n + dd for i in range(p0, p1, 5)]
        row.update(floor_base_cm=100 * np.mean([x.min() for x in fl0]), floor_res_cm=100 * np.mean([x.min() for x in fl1]))
        rows.append(row)
    # sub-segments of interest
    for name, (p0, p1) in [("carried: occlusion 430-510", (430, 510)), ("carried: edge-on 375-405", (375, 405)),
                           ("carried: pick-up 169-225", (169, 225))]:
        sel = np.zeros(len(Tc0), bool); sel[p0:p1] = True; g = sel & good
        row = dict(phase=name, iou_base=iou0[g].mean(), iou_res=iou1[g].mean(), photo_base=ph0[g].mean(), photo_res=ph1[g].mean(),
                   jit_base=jitter(Tc0[p0:p1]).mean(), jit_res=jitter(Tc1[p0:p1]).mean())
        if a.heldout:
            row.update(ho_base=ihb[g & odd].mean(), ho_res=ihr[g & odd].mean())
        rows.append(row)
    for row in rows:
        print(" | ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    np.save(os.path.splitext(a.diag)[0] + "_val.npy", rows, allow_pickle=True)


if __name__ == "__main__":
    main()
