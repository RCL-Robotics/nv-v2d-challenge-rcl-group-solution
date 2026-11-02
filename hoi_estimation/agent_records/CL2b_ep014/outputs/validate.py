#!/usr/bin/env python
"""Ground-truth-free validation of a trajectory vs. the baseline (used to write REPORT.md).

Per phase: silhouette IoU (half res, person pixels ignored), lever-visibility violations, pooled
underside / top texture correlation at the estimated pose (spin evidence), hand-in-object azimuth
stability while grasping, temporal smoothness, floor clearance, camera-depth difference to the baseline."""
import argparse
import sys
import os
import numpy as np
import torch
import cv2
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_inputs, load_mesh, Renderer, rot_angle_deg, dr  # noqa: E402
from solve import TextureEvidence, fit_symmetry_axis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--inputs', default='/work/inputs')
ap.add_argument('--result', default='/work/outputs/object_poses_world.npz')
ap.add_argument('--phases', default='0:95,96:315,316:390,391:599,600:627,628:937,938:992')
ap.add_argument('--texture', type=int, default=1)
a = ap.parse_args()
inp = a.inputs
D = load_inputs(inp); mesh = load_mesh(inp)
K, Twc = D['K'], D['Twc']
F = len(D['rec']['T_world_mesh'])
Tw = {'base': D['rec']['T_world_mesh'].astype(np.float64), 'result': np.load(a.result)['T_world_mesh']}
Tc = {k: np.linalg.inv(Twc) @ v for k, v in Tw.items()}
phases = [tuple(map(int, p.split(':'))) for p in a.phases.split(',')]
c_ax, d_ax = fit_symmetry_axis(mesh['v'])
centre = c_ax + 0.27 * d_ax
H, W = D['H'], D['W']
hf = h5py.File(f'{inp}/masks.h5', 'r')
Mh, Ph = [], []
for s0 in range(0, F, 64):
    for src, dst in ((hf['object'], Mh), (hf['person'], Ph)):
        dst += [cv2.resize(x.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0 for x in src[s0:s0 + 64]]
Mh, Ph = np.stack(Mh), np.stack(Ph)

# lever faces (geometric lever under the board, pointing at azimuth ~161 deg in the mesh frame)
v, f = mesh['v'], mesh['f']; fc = v[f].mean(1)
rr = np.hypot(fc[:, 0] - c_ax[0], fc[:, 1] - c_ax[1]); ang = np.degrees(np.arctan2(fc[:, 1] - c_ax[1], fc[:, 0] - c_ax[0]))
lever = (rr > 0.12) & (rr < 0.37) & (fc[:, 2] > -0.06) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
lev_t = torch.tensor(np.r_[False, lever], device='cuda')

Kh = K.copy(); Kh[:2] *= 0.5
Rh = Renderer(mesh, Kh, H // 2, W // 2)
stats = {}
for k in Tc:
    iou = np.zeros(F); lev_out = np.zeros(F); lev_px = np.zeros(F)
    for s in range(0, F, 32):
        T = torch.tensor(Tc[k][s:s + 32], dtype=torch.float32, device='cuda')
        vh = torch.cat([Rh.v, torch.ones_like(Rh.v[:, :1])], 1)
        clip = torch.einsum('ij,bnj->bni', Rh.proj(Rh.near, Rh.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
        rast, _ = dr.rasterize(Rh.glctx, clip, Rh.f, (Rh.H, Rh.W))
        msk = (rast[..., 3] > 0).cpu().numpy(); lm = lev_t[rast[..., 3].long()].cpu().numpy()
        for i in range(msk.shape[0]):
            fr = s + i; r_ = msk[i] & ~(Ph[fr] & ~Mh[fr])
            iou[fr] = (r_ & Mh[fr]).sum() / max((r_ | Mh[fr]).sum(), 1)
            l = lm[i] & ~Ph[fr]
            lev_px[fr] = l.sum(); lev_out[fr] = (l & ~Mh[fr]).sum()
    R = Tw[k][:, :3, :3]
    drot = np.r_[0, rot_angle_deg(R[:-1], R[1:])]
    pc = Tc[k][:, :3, :3] @ centre + Tc[k][:, :3, 3]
    dtr = np.r_[0, np.linalg.norm(np.diff(Tw[k][:, :3, :3] @ centre + Tw[k][:, :3, 3], axis=0), axis=1)] * 100
    # floor clearance of the lowest mesh point (world plane n.x + d)
    pl = D['plane']; vs = v[::40]
    low = np.array([((vs @ Tw[k][fr, :3, :3].T + Tw[k][fr, :3, 3]) @ pl[:3] + pl[3]).min() for fr in range(F)]) * 100
    stats[k] = dict(iou=iou, lev_px=lev_px, lev_out=lev_out, drot=drot, dtr=dtr, z=pc[:, 2], low=low)

# hand-in-object azimuth (about the table axis), right/left hand centroid
kp = D['rec']['human_keypoints_world'].astype(np.float64)
hands = {'R': kp[:, 21:42].mean(1), 'L': kp[:, 42:63].mean(1)}
contact = D['rec']['contact_prob_right_left']
for k in Tc:
    Ti = np.linalg.inv(Tw[k])
    az = {}
    for hn, p in hands.items():
        q = np.einsum('fij,fj->fi', Ti[:, :3, :3], p) + Ti[:, :3, 3]
        az[hn] = np.degrees(np.arctan2(q[:, 1] - c_ax[1], q[:, 0] - c_ax[0]))
    stats[k]['az'] = az

# texture evidence per phase at the estimated poses (peak angle of pooled underside / top texture)
tex = {k: {} for k in Tc}
if a.texture:
    Rf = Renderer(mesh, K, H, W)
    TE = TextureEvidence(mesh, Rf, K)
    frames = {}
    cap = cv2.VideoCapture(f'{inp}/video.mp4')
    for fr in range(F):
        ok, im = cap.read()
        if fr % 3:
            continue
        frames[fr] = im
    for k in Tc:
        for (p0, p1) in phases:
            TE.reset()
            for fr in range(p0, p1 + 1):
                if fr not in frames or hf['object'][fr].sum() < 200:
                    continue
                dep = Rf.render(Tc[k][fr][None], rgb=False, depth=True)['depth'][0].cpu().numpy()
                TE.add(frames[fr][:, :, ::-1].astype(np.float32) / 255, Tc[k][fr], hf['object'][fr], hf['person'][fr], dep)
            res = {}
            for which in ('under', 'top'):
                cv_ = TE.curve(which)
                if cv_ is not None:
                    i = np.nanargmax(cv_); i0 = list(TE.angs).index(0)
                    res[which] = (int(TE.angs[i]), float(cv_[i]), float(cv_[i0]))
            tex[k][(p0, p1)] = res


def circ_std(x):
    z = np.exp(1j * np.radians(x)); return np.degrees(np.sqrt(-2 * np.log(max(abs(z.mean()), 1e-9))))


print('| phase | frames | IoU base | IoU result | lever px outside mask base/result | rot step med (deg) base/result | trans step med (cm) base/result | lowest point above floor med (cm) base/result | depth diff result-base mean/median (cm) | underside tex peak (deg, corr) base -> result | top tex peak base -> result |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
for (p0, p1) in phases:
    sl = slice(p0, p1 + 1)
    b, r = stats['base'], stats['result']
    dz = (r['z'][sl] - b['z'][sl]) * 100

    def tx(k, w):
        t = tex[k].get((p0, p1), {}).get(w)
        return '-' if t is None else '%d (%.2f)' % (t[0], t[1])
    print('| %d-%d | %d | %.3f | %.3f | %.0f / %.0f | %.2f / %.2f | %.2f / %.2f | %.1f / %.1f | %.1f / %.1f | %s -> %s | %s -> %s |' % (
        p0, p1, p1 - p0 + 1, b['iou'][sl].mean(), r['iou'][sl].mean(), b['lev_out'][sl].mean(), r['lev_out'][sl].mean(),
        np.median(b['drot'][sl]), np.median(r['drot'][sl]), np.median(b['dtr'][sl]), np.median(r['dtr'][sl]),
        np.median(b['low'][sl]), np.median(r['low'][sl]), dz.mean(), np.median(dz),
        tx('base', 'under'), tx('result', 'under'), tx('base', 'top'), tx('result', 'top')))
b, r = stats['base'], stats['result']
print('| all | %d | %.3f | %.3f | %.0f / %.0f | %.2f / %.2f | %.2f / %.2f | %.1f / %.1f | %.1f / %.1f | | |' % (
    F, b['iou'].mean(), r['iou'].mean(), b['lev_out'].mean(), r['lev_out'].mean(), np.median(b['drot']), np.median(r['drot']),
    np.median(b['dtr']), np.median(r['dtr']), np.median(b['low']), np.median(r['low']),
    ((r['z'] - b['z']) * 100).mean(), np.median((r['z'] - b['z']) * 100)))
print('max rot step (deg) base %.1f result %.1f; max trans step (cm) base %.1f result %.1f' % (
    b['drot'].max(), r['drot'].max(), b['dtr'].max(), r['dtr'].max()))
# hand-azimuth stability inside continuous grasp windows (both hands in contact, window length 40)
print('\nHand-in-object azimuth jumps (deg) across windows of 15 frames while contact>0.9 (median / 95th pct):')
for k in Tc:
    js = []
    for hn, ci in (('R', 0), ('L', 1)):
        az = stats[k]['az'][hn]
        for fr in range(0, F - 15):
            if contact[fr:fr + 16, ci].min() > 0.9:
                js.append(abs(((az[fr + 15] - az[fr]) + 180) % 360 - 180))
    print('  %s: %.1f / %.1f' % (k, np.median(js), np.percentile(js, 95)))
