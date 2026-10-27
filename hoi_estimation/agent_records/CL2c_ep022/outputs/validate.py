#!/usr/bin/env python
"""Ground-truth-free validation: baseline vs. result, per phase. Not needed by solve.py.
usage: python validate.py --inputs /work/inputs --result object_poses_world.npz"""
import argparse, json, os, sys
import cv2, h5py, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_utils import load_mesh, Renderer
from silfit import SilFitter
from edgescore import edge_score
from traj_utils import smooth_poses, rot_angle_deg

ap = argparse.ArgumentParser()
ap.add_argument('--inputs', default='/work/inputs')
ap.add_argument('--result', default='/work/outputs/object_poses_world.npz')
args = ap.parse_args()
inp = args.inputs
cam = json.load(open(os.path.join(inp, 'camera.json')))
K, Twc = np.array(cam['K']), np.array(cam['T_world_cam'])
Tcw = np.linalg.inv(Twc)
pl = np.array(json.load(open(os.path.join(inp, 'ground_plane.json')))['plane'])
base = np.load(os.path.join(inp, 'reconstruction.npz'))['T_world_mesh'].astype(np.float64)
res = np.load(args.result)
ours = res['T_world_mesh']
phases = [tuple(p) for p in res['phases']]
mesh, tex = load_mesh(os.path.join(inp, 'object_mesh.glb'))
R = Renderer(mesh, tex, K)
c = R.v.mean(0).double().cpu().numpy()
V = np.asarray(mesh.vertices)[::10]
f = h5py.File(os.path.join(inp, 'masks.h5'), 'r')
cap = cv2.VideoCapture(os.path.join(inp, 'video.mp4'))
frames = []
while True:
    ok, im = cap.read()
    if not ok:
        break
    frames.append(im)
F = len(base)
fitter = SilFitter(R, K)


def metrics(Pw, a, b):
    Tc = Tcw[None] @ Pw[a:b + 1]
    idx = np.arange(a, b + 1)
    sub = idx[::2]
    fitter.setup(f['object'][a:b + 1:2], f['person'][a:b + 1:2], Tc[::2])
    sil = 1 - fitter.score(Tc[::2]).mean()
    fitter.fat_px = 0.0
    sil0 = 1 - fitter.score(Tc[::2]).mean()
    fitter.fat_px = 2.5
    es = []
    for i in idx[::5]:
        _, _, dp = R.render(Tcw @ Pw[i], need_rgb=False)
        es.append(edge_score(dp[0].cpu().numpy(), frames[i], f['person'][i]))
    S = smooth_poses(Tc, 3.0, center=c)
    jit_r = np.median(rot_angle_deg(Tc[:, :3, :3], S[:, :3, :3]))
    pc = np.einsum('fij,j->fi', Tc[:, :3, :3], c) + Tc[:, :3, 3]
    jit_t = np.median(np.linalg.norm(pc - (np.einsum('fij,j->fi', S[:, :3, :3], c) + S[:, :3, 3]), axis=1)) * 100
    step_r = rot_angle_deg(Tc[1:, :3, :3], Tc[:-1, :3, :3]) if b > a else np.zeros(1)
    W = np.einsum('fij,nj->fni', Pw[a:b + 1, :3, :3], V) + Pw[a:b + 1, None, :3, 3]
    h = (W @ pl[:3] + pl[3]).min(1) * 100
    return dict(iou_fat=sil, iou=sil0, edge=np.nanmean(es), jit_r=jit_r, jit_t=jit_t, max_step_r=step_r.max(),
                floor_med=np.median(h), floor_std=h.std(), z=Tc[:, 2, 3])


rows = []
for (a, b, t) in phases + [(0, F - 1, -1)]:
    name = {1: 'static', 0: 'moving', -1: 'ALL'}[int(t)]
    mb, mo = metrics(base, a, b), metrics(ours, a, b)
    dz = (mo['z'] - mb['z']) * 100
    rb = rot_angle_deg(base[a:b + 1, :3, :3], ours[a:b + 1, :3, :3])
    dt = np.linalg.norm(base[a:b + 1, :3, 3] - ours[a:b + 1, :3, 3], axis=1) * 100
    rows.append((f'{a}-{b} {name}', mb, mo, dz, rb, dt))

print('| phase | metric | baseline | result |')
print('|---|---|---|---|')
for nm, mb, mo, dz, rb, dt in rows:
    for k, lab, fmt in [('iou_fat', 'mask IoU (fat-mask model, 2.5px)', '%.3f'), ('iou', 'mask IoU (plain render)', '%.3f'),
                        ('edge', 'edge chamfer px (lower=better)', '%.3f'),
                        ('jit_r', 'rot jitter deg (median dev. from smoothed)', '%.2f'),
                        ('jit_t', 'centre jitter cm', '%.2f'), ('max_step_r', 'max frame-to-frame rot deg', '%.2f'),
                        ('floor_med', 'lowest point above floor plane, median cm', '%.2f'),
                        ('floor_std', 'lowest point above floor plane, std cm', '%.2f')]:
        print(f'| {nm} | {lab} | {fmt % mb[k]} | {fmt % mo[k]} |')
    print(f'| {nm} | cam depth z result-baseline, mean / median cm | 0 | {dz.mean():.2f} / {np.median(dz):.2f} |')
    print(f'| {nm} | change vs baseline: rot deg median / p90; transl cm median / p90 | - | '
          f'{np.median(rb):.2f} / {np.percentile(rb, 90):.2f}; {np.median(dt):.2f} / {np.percentile(dt, 90):.2f} |')
