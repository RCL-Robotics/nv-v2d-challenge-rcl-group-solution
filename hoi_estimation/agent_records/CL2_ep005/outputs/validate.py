"""Ground-truth-free validation: baseline vs result, per phase.
python validate.py --inputs /work/inputs --result object_poses_world.npz [--holdout holdout_run.npz] > table.md
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_lib import Renderer, cam_from_world, load_inputs, load_mesh, rot_angle  # noqa: E402
from solve import read_gray_frames  # noqa: E402


def mask_iou(R, d, Tc, OM, PM, scale=0.5):
    out = []
    for s in range(0, len(Tc), 32):
        r = R.render(Tc[s:s + 32], d["K"], d["H"], d["W"], scale=scale, need_rgb=False)["mask"].cpu().numpy()
        for j in range(len(r)):
            o = OM[s + j][::2, ::2]
            v = ~(PM[s + j][::2, ::2] & ~o)
            out.append((r[j] & o & v).sum() / max(((r[j] | o) & v).sum(), 1))
    return np.array(out)


def crease_labels(R, fn_lab, Tc_batch, d):
    T = torch.tensor(Tc_batch, dtype=torch.float32, device="cuda")
    K, H, W = d["K"], d["H"], d["W"]
    P = torch.tensor([[2 * K[0, 0] / W, 0, 2 * (K[0, 2] + 0.5) / W - 1, 0], [0, 2 * K[1, 1] / H, 2 * (K[1, 2] + 0.5) / H - 1, 0],
                      [0, 0, 1.002, -0.1], [0, 0, 1, 0]], dtype=torch.float32, device="cuda")
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum("ij,bnj->bni", P, torch.einsum("bij,nj->bni", T, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (H, W))
    tri = rast[..., 3].long() - 1
    lab = fn_lab[tri.clamp_min(0)]
    lab[tri < 0] = 0
    return lab.cpu().numpy()


def edge_chamfer(R, fn_lab, d, Tc, idx, gray, PM, trunc=8.0):
    """Mean truncated distance (px, full res) from rendered internal box-crease edges to Canny edges."""
    out = []
    for s in range(0, len(idx), 8):
        ii = idx[s:s + 8]
        labs = crease_labels(R, fn_lab, Tc[ii], d)
        for j, i in enumerate(ii):
            lab = labs[j]
            ok = lab > 0
            e = np.zeros(lab.shape, bool)
            e[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & ok[:, 1:] & ok[:, :-1]
            e[1:, :] |= (lab[1:, :] != lab[:-1, :]) & ok[1:, :] & ok[:-1, :]
            e &= ~(cv2.dilate(PM[i].astype(np.uint8), np.ones((9, 9))) > 0)
            ys, xs = np.where(ok)
            y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 20, max(xs.min() - 20, 0), xs.max() + 20
            crop = cv2.equalizeHist(np.ascontiguousarray(gray[i][y0:y1, x0:x1]))
            ed = cv2.Canny(cv2.GaussianBlur(crop, (5, 5), 1.2), 40, 100)
            dt = cv2.distanceTransform((ed == 0).astype(np.uint8), cv2.DIST_L2, 3)
            ee = e[y0:y1, x0:x1]
            out.append(np.minimum(dt[ee], trunc).mean() if ee.sum() > 20 else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--holdout", default=None, help="result of solve.py --holdout (its _diag.npz next to it)")
    args = ap.parse_args()
    d = load_inputs(args.inputs)
    mesh = load_mesh(args.inputs)
    R = Renderer(mesh)
    diag = json.load(open(os.path.splitext(args.result)[0] + "_diag.json"))
    phases = diag["phases"]
    Tb = cam_from_world(d["T_wc"], d["rec"]["T_world_mesh"].astype(np.float64))
    Tr = cam_from_world(d["T_wc"], np.load(args.result)["T_world_mesh"])
    F = len(Tb)
    f5 = h5py.File(os.path.join(args.inputs, "masks.h5"), "r")
    OM, PM = f5["object"][:], f5["person"][:]
    gray = read_gray_frames(os.path.join(args.inputs, "video.mp4"))
    Sobb, _ = trimesh.bounds.oriented_bounds(mesh)
    Robb = np.linalg.inv(Sobb)[:3, :3]
    fn_lab = torch.tensor(np.argmax(np.asarray(mesh.face_normals) @ np.concatenate([Robb, -Robb], 1), 1) + 1, device="cuda")

    iou_b, iou_r = mask_iou(R, d, Tb, OM, PM), mask_iou(R, d, Tr, OM, PM)
    eidx = list(range(2, F, 5))
    ed_b, ed_r = edge_chamfer(R, fn_lab, d, Tb, eidx, gray, PM), edge_chamfer(R, fn_lab, d, Tr, eidx, gray, PM)
    ed_b_full, ed_r_full = np.full(F, np.nan), np.full(F, np.nan)
    ed_b_full[eidx], ed_r_full[eidx] = ed_b, ed_r
    ho = None
    if args.holdout:
        hz = np.load(os.path.splitext(args.holdout)[0] + "_diag.npz")
        ho = hz["holdout_mask"]
        Th = cam_from_world(d["T_wc"], np.load(args.holdout)["T_world_mesh"])
        hidx = list(np.where(ho)[0])
        iou_h = mask_iou(R, d, Th[hidx], OM[hidx], PM[hidx])
        ed_h = edge_chamfer(R, fn_lab, d, Th, hidx, gray, PM)
        iou_h_full, ed_h_full = np.full(F, np.nan), np.full(F, np.nan)
        iou_h_full[hidx], ed_h_full[hidx] = iou_h, ed_h

    n, dd = d["plane"][:3], d["plane"][3]
    V = np.asarray(mesh.vertices)[::10]
    Vadd = np.asarray(mesh.vertices)[::50]

    def hmin(Tc):
        Tw = d["T_wc"][None] @ Tc
        return ((np.einsum("fij,nj->fni", Tw[:, :3, :3], V) + Tw[:, None, :3, 3]) @ n + dd).min(1)

    def pts(Tc):
        return np.einsum("fij,nj->fni", Tc[:, :3, :3], Vadd) + Tc[:, None, :3, 3]

    hb, hr = hmin(Tb), hmin(Tr)
    add_br = np.linalg.norm(pts(Tb) - pts(Tr), axis=2).mean(1) * 100
    rotstep = lambda T: np.r_[rot_angle(T[1:, :3, :3], T[:-1, :3, :3]), np.nan]
    zacc = lambda T: np.r_[np.nan, np.abs(T[2:, 2, 3] - 2 * T[1:-1, 2, 3] + T[:-2, 2, 3]) * 100, np.nan]
    rb, rr, zb, zr = rotstep(Tb), rotstep(Tr), zacc(Tb), zacc(Tr)
    dz = (Tr[:, 2, 3] - Tb[:, 2, 3]) * 100

    rows = []
    hdr = ("| phase | frames | IoU base | IoU ours | held-out IoU | edge px base | edge px ours | held-out edge px | "
           "rot step deg base/ours | z-accel cm base/ours | static spread cm base/ours | floor h cm base/ours (std) | "
           "dz mean / median cm | diff of median z cm | ADD(base,ours) cm |")
    rows.append(hdr)
    rows.append("|" + "---|" * (hdr.count("|") - 1))
    allph = [(k, a, b) for k, a, b in phases] + [("ALL", 0, F - 1)]
    for k, a, b in allph:
        s = slice(a, b + 1)
        if k == "static":
            pb = pts(Tb[s])
            spread_b = np.linalg.norm(pb - pb.mean(0), axis=2).mean(1).mean() * 100
            pr = pts(Tr[s])
            spread_r = np.linalg.norm(pr - pr.mean(0), axis=2).mean(1).mean() * 100
            spread = f"{spread_b:.2f} / {spread_r:.2f}"
        else:
            spread = "-"
        hcol = f"{np.nanmean(iou_h_full[s]):.3f}" if ho is not None and np.isfinite(iou_h_full[s]).any() else "-"
        hed = f"{np.nanmean(ed_h_full[s]):.2f}" if ho is not None and np.isfinite(ed_h_full[s]).any() else "-"
        rows.append(f"| {k} {a}-{b} | {b - a + 1} | {iou_b[s].mean():.3f} | {iou_r[s].mean():.3f} | {hcol} | "
                    f"{np.nanmean(ed_b_full[s]):.2f} | {np.nanmean(ed_r_full[s]):.2f} | {hed} | "
                    f"{np.nanmedian(rb[s]):.2f} / {np.nanmedian(rr[s]):.2f} | {np.nanmedian(zb[s]):.2f} / {np.nanmedian(zr[s]):.2f} | {spread} | "
                    f"{np.median(hb[s]) * 100:.1f} ({hb[s].std() * 100:.1f}) / {np.median(hr[s]) * 100:.1f} ({hr[s].std() * 100:.1f}) | "
                    f"{dz[s].mean():.2f} / {np.median(dz[s]):.2f} | {(np.median(Tr[s, 2, 3]) - np.median(Tb[s, 2, 3])) * 100:.2f} | "
                    f"{add_br[s].mean():.2f} |")
    print("\n".join(rows))
    if ho is not None:
        print(f"\nheld-out frames ({ho.sum()}): baseline IoU {iou_b[ho].mean():.3f} -> held-out-run IoU {np.nanmean(iou_h_full[ho]):.3f}; "
              f"baseline edge px {np.nanmean(edge_chamfer(R, fn_lab, d, Tb, list(np.where(ho)[0]), gray, PM)):.2f} -> {np.nanmean(ed_h_full[ho]):.2f}")
    print(f"\nmax per-frame rotation change ours vs base: {rot_angle(Tb[:, :3, :3], Tr[:, :3, :3]).max():.2f} deg, "
          f"median {np.median(rot_angle(Tb[:, :3, :3], Tr[:, :3, :3])):.2f} deg")


if __name__ == "__main__":
    main()
