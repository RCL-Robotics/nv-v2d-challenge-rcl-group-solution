"""Validation without ground truth: per-phase table baseline vs result (mask IoU, appearance, smoothness,
floor contact, depth difference). Usage: python validate.py --inputs /work/inputs --res object_poses_world.npz"""
import argparse, json, os, sys
import numpy as np, cv2, h5py, torch, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_inputs, load_video, rot_angle
from fit import CropFitter, FrameData, project, box_from
from traj import eval_frames

ap = argparse.ArgumentParser(); ap.add_argument("--inputs", default="/work/inputs")
ap.add_argument("--res", default="/work/outputs/object_poses_world.npz"); a = ap.parse_args()
cam, K, Twc, rec = load_inputs(a.inputs); Tcw = np.linalg.inv(Twc)
Tb = Tcw @ rec["T_world_mesh"].astype(np.float64)
To = Tcw @ np.load(a.res)["T_world_mesh"]
diag = np.load(os.path.splitext(a.res)[0] + "_diag.npz"); ph = diag["phases"]; gs = float(diag["gscale"])
frames = load_video(a.inputs, 1.0)
with h5py.File(os.path.join(a.inputs, "masks.h5"), "r") as f:
    OBJ = f["object"][:]; PER = f["person"][:]
# note: only the person is excluded here (not the stool) - identical treatment for baseline and result
fit = CropFitter(os.path.join(a.inputs, "object_mesh.glb"), res=160)
corners = trimesh.bounds.corners(fit.mesh.bounds)
pl = np.array(json.load(open(os.path.join(a.inputs, "ground_plane.json")))["plane"])
V = np.asarray(fit.mesh.vertices)
rows = []
names = ["static_start", "carried", "static_end"]
for name, (x0, x1) in zip(names, ph):
    ids = [i for i in range(x0, x1 + 1) if OBJ[i].sum() > 200]
    kern = np.ones((7, 7), np.uint8)
    fd = FrameData([frames[i] for i in ids], [OBJ[i] for i in ids],
                   [cv2.dilate(PER[i].astype(np.uint8), kern) > 0 for i in ids],
                   [box_from(OBJ[i], project(K, Tb[i], corners), pad=0.15) for i in ids], K, 160)
    res = {}
    for tag, T in [("base", Tb), ("ours", To)]:
        Rm, p = fit.to_rp(T[ids])
        E1 = eval_frames(fit, fd, Rm, p, scale=1.0)
        Eg = eval_frames(fit, fd, Rm, p, scale=gs)
        Rr = T[x0:x1 + 1, :3, :3]; step = rot_angle(Rr[1:], Rr[:-1])
        c = T[x0:x1 + 1, :3, 3] + np.einsum("bij,j->bi", Rr, fit.center)
        acc = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1)
        W = Twc @ T[x0:x1 + 1]
        fl = np.array([((V @ w[:3, :3].T + w[:3, 3]) @ pl[:3] + pl[3]).min() for w in W[::5]])
        res[tag] = dict(iou=E1[:, 0].mean(), iou_s=Eg[:, 0].mean(), app=(Eg[:, 2] + Eg[:, 3]).mean(),
                        step_med=np.median(step), step_p95=np.percentile(step, 95), acc=np.median(acc) * 100,
                        floor=np.median(fl) * 100, z=c[:, 2])
    dz = (res["ours"]["z"] - res["base"]["z"]) * 100
    dr = rot_angle(To[x0:x1 + 1, :3, :3], Tb[x0:x1 + 1, :3, :3])
    rows.append((name, x0, x1, res, dz, dr))
print("| phase | frames | IoU base / ours (scale 1) | IoU base / ours (scale %.3f) | appearance dist base / ours | rot step med / p95 deg base | ours | centre accel median cm base / ours | lowest point to floor cm base / ours | depth diff ours-base mean / median cm | rot change vs base median / p90 deg |" % gs)
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, x0, x1, r, dz, dr in rows:
    b, o = r["base"], r["ours"]
    print(f"| {name} | {x0}-{x1} | {b['iou']:.3f} / {o['iou']:.3f} | {b['iou_s']:.3f} / {o['iou_s']:.3f} | {b['app']:.3f} / {o['app']:.3f} | "
          f"{b['step_med']:.2f} / {b['step_p95']:.1f} | {o['step_med']:.2f} / {o['step_p95']:.1f} | {b['acc']:.2f} / {o['acc']:.2f} | "
          f"{b['floor']:.1f} / {o['floor']:.1f} | {dz.mean():.2f} / {np.median(dz):.2f} | {np.median(dr):.1f} / {np.percentile(dr, 90):.1f} |")
