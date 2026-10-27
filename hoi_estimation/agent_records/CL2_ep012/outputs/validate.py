#!/usr/bin/env python
"""No-ground-truth validation: compare baseline and result per phase.
Metrics (per frame, person pixels excluded):
  iou      - silhouette IoU with the SAM2 object mask (static phases were fitted on EVEN frames: odd frames are held out)
  holes    - NCC between rendered silhouette-with-holes and image darkness (not optimised by solve.py)
  gradncc  - NCC between gradient magnitudes of the rendered textured mesh and of the image (not optimised)
  jitter   - mean |2nd difference| of camera-frame translation (cm) and rotation (deg)
  height   - lowest object point above the released floor plane (cm)
  dz       - camera-frame depth difference result - baseline (cm)
usage: python validate.py --inputs /work/inputs --result /work/outputs/object_poses_world.npz [--step 2]
"""
import argparse, json, sys, os
import numpy as np, torch, h5py, cv2, trimesh
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import Renderer
from refine import CropData, render_soft, pool, ncc_t


def gradmag(x):
    gx = x[..., :, 1:] - x[..., :, :-1]; gy = x[..., 1:, :] - x[..., :-1, :]
    return torch.sqrt(gx[..., :-1, :] ** 2 + gy[..., :, :-1] ** 2 + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    cam = json.load(open(f"{a.inputs}/camera.json"))
    K = np.array(cam["K"]); Twc = np.array(cam["T_world_cam"]); Tcw = np.linalg.inv(Twc)
    rec = np.load(f"{a.inputs}/reconstruction.npz")
    res = np.load(a.result)
    TB = Tcw @ rec["T_world_mesh"].astype(float); TR = Tcw @ res["T_world_mesh"]
    ph = res["phases"]; F = TB.shape[0]
    phases = {"S1 static": range(ph[0], ph[1] + 1), "M manip": range(ph[2], ph[3] + 1), "S2 static": range(ph[4], ph[5] + 1)}
    mesh = trimesh.load(f"{a.inputs}/object_mesh.glb", force="scene", process=False).to_geometry()
    pl = json.load(open(f"{a.inputs}/ground_plane.json"))["plane"]; n = np.array(pl[:3]); d = pl[3]
    hull = np.asarray(mesh.convex_hull.vertices)
    C, S = 320, 2
    r = Renderer(mesh, C * S, C * S, tex_max=2048)
    H5 = h5py.File(f"{a.inputs}/masks.h5"); MO = H5["object"]; MP = H5["person"]
    cap = cv2.VideoCapture(f"{a.inputs}/video.mp4")
    met = np.full((F, 2, 3), np.nan)
    for f in range(F):
        ok, im = cap.read()
        if f % a.step: continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        c = K @ TB[f, :3, 3]; uv = c[:2] / c[2]
        cr = CropData(g, MO[f], MP[f], K, uv, C, S)
        with torch.no_grad():
            T = torch.tensor(np.stack([TB[f], TR[f]]), dtype=torch.float32, device="cuda")
            ms, gr = render_soft(r, T, cr.Kc); ms = pool(ms, S); gr = pool(gr, S)
            mo = torch.tensor(cr.mo, device="cuda", dtype=torch.float32)[None]
            va = torch.tensor(~cr.mp, device="cuda", dtype=torch.float32)[None]
            hm = (ms > 0.5).float()
            iou = (hm * mo * va).sum((-1, -2)) / (((hm + mo) > 0).float() * va).sum((-1, -2)).clamp_min(1)
            gi = torch.tensor(cr.g, device="cuda", dtype=torch.float32)[None]
            regn = cv2.dilate(((ms.cpu().numpy() > 0.5).any(0) | cr.mo).astype(np.uint8), np.ones((9, 9)))
            reg = torch.tensor(regn, device="cuda", dtype=torch.float32)[None] * va
            holes = ncc_t(ms, -gi, reg)
            gn = ncc_t(gradmag(gr), gradmag(gi), reg[..., :-1, :-1])
        met[f, :, 0] = iou.cpu().numpy(); met[f, :, 1] = holes.cpu().numpy(); met[f, :, 2] = gn.cpu().numpy()

    def jit(T, fr):
        fr = list(fr); t = T[fr, :3, 3]
        d2t = np.linalg.norm(t[2:] - 2 * t[1:-1] + t[:-2], axis=1) * 100
        q = Rot.from_matrix(T[fr, :3, :3])
        w = (q[1:] * q[:-1].inv()).as_rotvec()
        d2r = np.degrees(np.linalg.norm(w[1:] - w[:-1], axis=1))
        return d2t.mean(), d2r.mean()

    def height(T, f):
        Tw = Twc @ T[f]; w = hull @ Tw[:3, :3].T + Tw[:3, 3]; return (w @ n + d).min() * 100

    rows = []
    hdr = "| phase | frames | IoU base | IoU ours | IoU held-out base/ours | holes-NCC base/ours | grad-NCC base/ours | jitter t cm base/ours | jitter R deg base/ours | floor height cm base/ours (median) | dz mean cm | dz median cm | rot change vs base (median deg) |"
    rows.append(hdr); rows.append("|" + "---|" * hdr.count("|")[:-1] if False else "|---" * (hdr.count("|") - 1) + "|")
    for name, fr in list(phases.items()) + [("ALL", range(F))]:
        fr = list(fr); mm = met[fr]
        ok = ~np.isnan(mm[:, 0, 0])
        odd = np.array([f % 2 == 1 for f in fr]) & ok
        jb, jr = jit(TB, fr), jit(TR, fr)
        hb = np.median([height(TB, f) for f in fr]); hr = np.median([height(TR, f) for f in fr])
        dz = (TR[fr, 2, 3] - TB[fr, 2, 3]) * 100
        dR = [np.degrees(np.linalg.norm(Rot.from_matrix(TR[f, :3, :3] @ TB[f, :3, :3].T).as_rotvec())) for f in fr]
        rows.append(f"| {name} | {fr[0]}-{fr[-1]} | {np.nanmean(mm[ok, 0, 0]):.3f} | {np.nanmean(mm[ok, 1, 0]):.3f} | "
                    f"{np.nanmean(mm[odd, 0, 0]):.3f}/{np.nanmean(mm[odd, 1, 0]):.3f} | "
                    f"{np.nanmean(mm[ok, 0, 1]):.3f}/{np.nanmean(mm[ok, 1, 1]):.3f} | {np.nanmean(mm[ok, 0, 2]):.3f}/{np.nanmean(mm[ok, 1, 2]):.3f} | "
                    f"{jb[0]:.2f}/{jr[0]:.2f} | {jb[1]:.2f}/{jr[1]:.2f} | {hb:.1f}/{hr:.1f} | {dz.mean():+.2f} | {np.median(dz):+.2f} | {np.median(dR):.2f} |")
        frac = (mm[ok, 1, 0] > mm[ok, 0, 0]).mean()
        rows.append(f"|  ↳ frac frames ours better | | IoU {frac:.2f} | holes {(mm[ok, 1, 1] > mm[ok, 0, 1]).mean():.2f} | grad {(mm[ok, 1, 2] > mm[ok, 0, 2]).mean():.2f} | | | | | | | | |")
    txt = "\n".join(rows)
    print(txt)
    if a.save:
        open(a.save, "w").write(txt + "\n"); np.save(a.save.replace(".md", "_perframe.npy"), met)


if __name__ == "__main__":
    main()
