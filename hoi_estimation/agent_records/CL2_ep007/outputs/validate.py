#!/usr/bin/env python
"""Ground-truth-free validation: baseline vs result, per phase.
python validate.py --inputs /work/inputs --result object_poses_world.npz --masks iron_masks_crop.npz"""
import argparse, json, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_inputs, load_mesh, rot_angle_deg, project
from fitcore import Fitter, color_loss
from prep import build, compute_touch
from hyp import soft_iou_loss


def render_metrics(fit, data, Tc, frames, bs=64):
    iou, col = np.zeros(len(frames)), np.zeros(len(frames))
    for s in range(0, len(frames), bs):
        fr = frames[s:s + bs]
        P = fit.proj_mats(data["roi"][fr], (160, 160))
        R = torch.tensor(Tc[fr, :3, :3], dtype=torch.float32, device="cuda")
        t = torch.tensor(Tc[fr, :3, 3], dtype=torch.float32, device="cuda")
        with torch.no_grad():
            m, rgb, _ = fit.render(R, t, P, (160, 160), rgb=True)
            mo = torch.tensor(data["mask"][fr], device="cuda")
            pe = torch.tensor(data["person"][fr], device="cuda")
            valid = 1 - pe * (1 - mo)
            mh, moh = (m > 0.5).float(), (mo > 0.5).float()
            inter = (valid * mh * moh).sum((1, 2))
            uni = (valid * ((mh + moh) > 0).float()).sum((1, 2)).clamp_min(1)
            iou[s:s + bs] = (inter / uni).cpu().numpy()
            col[s:s + bs] = color_loss(rgb, m, torch.tensor(data["rgb"][fr], device="cuda"), mo, valid).cpu().numpy()
    return iou, col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", default=os.path.join(HERE, "object_poses_world.npz"))
    ap.add_argument("--masks", default=os.path.join(HERE, "iron_masks_crop.npz"))
    ap.add_argument("--info", default=os.path.join(HERE, "object_poses_world_info.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "validation.json"))
    a = ap.parse_args()
    K, Kp, Twc, rec = load_inputs(a.inputs)
    info = json.load(open(a.info))
    Tcw = np.linalg.inv(Twc)
    mesh = load_mesh(f"{a.inputs}/object_mesh.glb")
    V = np.asarray(mesh.vertices)
    cm = mesh.bounds.mean(0)
    fit = Fitter(mesh, K)
    z = np.load(a.masks)
    M, crop = z["mask"], tuple(z["crop"])
    F = len(M)
    x0, y0, x1, y1 = crop

    def full(i):
        o = np.zeros((1152, 1536), bool)
        o[y0:y1, x0:x1] = M[i]
        return o
    data = build(f"{a.inputs}/video.mp4", full, f"{a.inputs}/masks.h5", F)
    touch = compute_touch(M, crop, f"{a.inputs}/masks.h5")
    TB = Tcw[None] @ rec["T_world_mesh"].astype(np.float64)
    TR = Tcw[None] @ np.load(a.result)["T_world_mesh"]
    fe, hs = info["flat_end"], info["heel_start"]
    s0 = info["static_flat"][0] if info["static_flat"] else [0, 0]
    s1 = info["static_standing"][-1] if info["static_standing"] else [F, F]
    phases = {"P1 rest on sole (static)": (s0[0], s0[1]), "P2 slid on table": (s0[1], fe),
              "P3 stand-up transition": (fe, hs), "P4 standing, held": (hs, s1[0]),
              "P5 standing (static)": (s1[0], s1[1]), "all": (0, F)}
    # physical table plane (final, scaled): plane through the iron's soleplate in static phase 1
    n_s = np.array(info["n_sole"])
    sole_pts = V[V @ (np.array([-0.9821, 0.0797, 0.1705]) / np.linalg.norm([-0.9821, 0.0797, 0.1705])) >
                 np.max(V @ (np.array([-0.9821, 0.0797, 0.1705]) / np.linalg.norm([-0.9821, 0.0797, 0.1705]))) - 0.002]
    kp = rec["human_keypoints_world"].astype(np.float64)
    rh = (kp[:, 21:42] @ Tcw[:3, :3].T + Tcw[:3, 3]).mean(1)
    grip = np.array([0.065, 0.0, -0.01])
    out = {}
    res = {}
    for name, T in (("baseline", TB), ("result", TR)):
        iou, col = render_metrics(fit, data, T, np.arange(F))
        c = np.einsum("fij,j->fi", T[:, :3, :3], cm) + T[:, :3, 3]
        rot_v = np.r_[0, [rot_angle_deg(T[i - 1, :3, :3], T[i, :3, :3]) for i in range(1, F)]]
        tr_v = np.r_[0, np.linalg.norm(np.diff(c, axis=0), axis=1)]
        Vw = V[::20]
        low = np.array([np.min((Vw @ T[i, :3, :3].T + T[i, :3, 3]) @ n_s) for i in range(F)])
        G = np.einsum("fij,j->fi", T[:, :3, :3], grip) + T[:, :3, 3]
        res[name] = dict(iou=iou, col=col, c=c, rot_v=rot_v, tr_v=tr_v, low=low, hand=np.linalg.norm(rh - G, axis=1))
    # table offset reference: the result's static phase 1 contact height (the table plane under the iron)
    ref = np.median(res["result"]["low"][s0[0]:s0[1]]) if s0[1] > s0[0] else np.median(res["result"]["low"])
    for ph, (a0, a1) in phases.items():
        if a1 <= a0:
            continue
        sl = slice(a0, a1)
        row = {"frames": [int(a0), int(a1)]}
        for name in ("baseline", "result"):
            r = res[name]
            row[name] = {
                "mask_IoU_mean": float(r["iou"][sl].mean()), "mask_IoU_p10": float(np.percentile(r["iou"][sl], 10)),
                "colour_resid_median": float(np.median(r["col"][sl])),
                "rot_speed_deg_per_frame_med": float(np.median(r["rot_v"][sl])),
                "rot_speed_p95": float(np.percentile(r["rot_v"][sl], 95)),
                "trans_speed_cm_per_frame_med": float(100 * np.median(r["tr_v"][sl])),
                "lowest_point_above_table_cm_med": float(100 * (np.median(r["low"][sl]) - ref)),
                "hand_to_handle_cm_med_touching": float(100 * np.median(r["hand"][sl][touch[sl] > 0])) if (touch[sl] > 0).any() else None,
            }
        dz = res["result"]["c"][sl, 2] - res["baseline"]["c"][sl, 2]
        row["depth_diff_result_minus_baseline_cm"] = {"mean": float(100 * dz.mean()), "median": float(100 * np.median(dz))}
        uvb = project(K, res["baseline"]["c"][sl]); uvr = project(K, res["result"]["c"][sl])
        row["image_offset_result_vs_baseline_px_med"] = float(np.median(np.linalg.norm(uvb - uvr, axis=1)))
        out[ph] = row
    json.dump(out, open(a.out, "w"), indent=1)
    for ph, row in out.items():
        print(ph, row["frames"])
        for name in ("baseline", "result"):
            print("  %-8s" % name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row[name].items()})
        print("  depth diff (cm)", {k: round(v, 1) for k, v in row["depth_diff_result_minus_baseline_cm"].items()},
              "image offset px %.0f" % row["image_offset_result_vs_baseline_px_med"])


if __name__ == "__main__":
    main()
