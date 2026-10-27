"""Ground-truth-free validation of a trajectory: image agreement and physical plausibility, per phase."""
import numpy as np
import torch
import h5py
from bat_lib import Renderer, W_IMG, H_IMG, rotmat_angle
from stages import BAT_CENTER, depth_scores, score_frames


def full_res_iou(ep, Tc, step=1, bs=16):
    R = Renderer(ep.mesh)
    f = h5py.File(f"{ep.inp}/masks.h5", "r")
    ids = list(range(0, ep.F, step)); iou = np.full(ep.F, np.nan)
    for s in range(0, len(ids), bs):
        idx = ids[s:s + bs]
        m, _ = R.render(Tc[idx], ep.K, W_IMG, H_IMG); m = m.cpu().numpy() > 0
        for j, i in enumerate(idx):
            b = f["object"][i]; p = f["person"][i]; valid = ~(p & ~b)
            iou[i] = (m[j] & b).sum() / max(((m[j] | b) & valid).sum(), 1)
    return iou


def hand_dist(ep, Tc):
    kpc = np.einsum("ij,fkj->fki", ep.Tcw[:3, :3], ep.kp_world) + ep.Tcw[:3, 3]
    zs = np.linspace(-0.2, 0.33, 30); seg = np.stack([np.zeros(30), np.zeros(30), zs], 1)
    P = np.einsum("fij,nj->fni", Tc[:, :3, :3], seg) + Tc[:, None, :3, 3]
    H = kpc[:, 21:63]
    return np.linalg.norm(H[:, :, None] - P[:, None], axis=-1).min((1, 2))


def floor_pen(ep, Tc, verts):
    Tw = ep.Twc[None] @ Tc
    P = np.einsum("fij,vj->fvi", Tw[:, :3, :3], verts) + Tw[:, None, :3, 3]
    h = P @ ep.plane[:3] + ep.plane[3]
    return h.min(1)


def evaluate(ep, FR, Tc, phases, verts):
    iou = full_res_iou(ep, Tc)
    iou_ext = 1 - np.array(score_frames(ep, FR, list(range(ep.F)), [T[None] for T in Tc]))[:, 0, 1]
    dres = np.array(depth_scores(ep, FR, list(range(ep.F)), [T[None] for T in Tc])).ravel()
    hd = hand_dist(ep, Tc); fh = floor_pen(ep, Tc, verts)
    rot_step = np.array([rotmat_angle(Tc[i, :3, :3], Tc[i + 1, :3, :3]) for i in range(ep.F - 1)])
    cen = Tc[:, :3, :3] @ BAT_CENTER + Tc[:, :3, 3]
    cb = ep.Tc_base[:, :3, :3] @ BAT_CENTER + ep.Tc_base[:, :3, 3]
    acc = np.linalg.norm(cen[:-2] - 2 * cen[1:-1] + cen[2:], axis=1)
    rows = {}
    for name, (a, b) in phases.items():
        sl = slice(a, b + 1)
        dz = (cen[sl, 2] - cb[sl, 2]) * 100
        rows[name] = dict(
            frames=f"{a}-{b}", iou_mean=np.nanmean(iou[sl]), iou_med=np.nanmedian(iou[sl]),
            iou_lt05=np.mean(iou[sl] < 0.5), iou_ext=np.mean(iou_ext[sl]), depthprof_cm=100 * np.nanmedian(dres[sl]),
            hand_cm=100 * np.median(hd[sl]), floor_min_cm=100 * fh[sl].min(),
            rot_step_med=np.median(rot_step[a:max(a + 1, b)]), rot_step_p95=np.percentile(rot_step[a:max(a + 1, b)], 95),
            acc_mm=1000 * np.median(acc[a:max(a + 1, b - 1)]), dz_mean_cm=dz.mean(), dz_med_cm=np.median(dz))
    return rows, dict(iou=iou, iou_ext=iou_ext, dres=dres, hand=hd, floor=fh, rot_step=rot_step)


def format_table(rows_by_method):
    keys = ["frames", "iou_mean", "iou_med", "iou_lt05", "iou_ext", "depthprof_cm", "hand_cm", "floor_min_cm", "rot_step_med",
            "rot_step_p95", "acc_mm", "dz_mean_cm", "dz_med_cm"]
    lines = ["| phase | method | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 2)]
    phases = list(next(iter(rows_by_method.values())).keys())
    for ph in phases:
        for m, rows in rows_by_method.items():
            r = rows[ph]
            vals = [r[k] if isinstance(r[k], str) else f"{r[k]:.3f}" if abs(r[k]) < 10 else f"{r[k]:.1f}" for k in keys]
            lines.append(f"| {ph} | {m} | " + " | ".join(vals) + " |")
    return "\n".join(lines)
