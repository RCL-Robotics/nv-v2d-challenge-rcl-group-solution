#!/usr/bin/env python
"""Ground-truth-free validation: per-phase table baseline vs result.
usage: python validate.py --inputs /work/inputs --result object_poses_world.npz [--md out.md]"""
import argparse, os, sys, json
import numpy as np, h5py, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoi_lib as H
from solve import STATIC

ap = argparse.ArgumentParser()
ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--result', required=True)
ap.add_argument('--pipeline', default='/pipeline'); ap.add_argument('--md', default=None)
ap.add_argument('--holdout_eval', action='store_true', help='also report odd frames of moving segments only')
args = ap.parse_args()
K, Twc, rec = H.load_inputs(args.inputs)
Tb = H.mesh_to_cam(Twc, rec['T_world_mesh'].astype(np.float64))
Tr = H.mesh_to_cam(Twc, np.load(args.result)['T_world_mesh'])
F = len(Tb)
frames = H.load_video(f'{args.inputs}/video.mp4')
with h5py.File(f'{args.inputs}/masks.h5') as f:
    OM = f['object'][:]; PM = f['person'][:]
with h5py.File(f'{args.pipeline}/depth/moge2/aligned.h5') as fa:
    ds = [cv2.imdecode(np.frombuffer(fa['aligned/front_stereo_camera_left'][i], np.uint8), cv2.IMREAD_UNCHANGED)
          for i in range(0, F, 10)]
bg = (np.median(np.stack(ds), 0) / 1000.0).astype(np.float32)
R = H.MeshRenderer(f'{args.inputs}/object_mesh.glb')
fit = H.PoseFitter(R, H.Obs(frames, OM, PM, bg, K), K)
idx = np.arange(F)
Sb = H.score_traj(fit, Tb, idx); Sr = H.score_traj(fit, Tr, idx)
pl = np.array(json.load(open(f'{args.inputs}/ground_plane.json'))['plane'])
V = R.verts_np[::50]

# phases: static + the moving gaps
is_static = np.zeros(F, bool)
for _, a, b in STATIC: is_static[a:b + 1] = True
ph = [(n, a, b, 'static') for n, a, b in STATIC]
t = 0; k = 0
while t < F:
    if is_static[t]: t += 1; continue
    s = t
    while t < F and not is_static[t]: t += 1
    ph.append((f'M{k}', s, t - 1, 'moving')); k += 1
ph.sort(key=lambda p: p[1])

def rspeed(T):
    return H.rot_angle_deg(T[1:, :3, :3], T[:-1, :3, :3])

def tacc(T):
    p = T[:, :3, 3]
    return np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) * 100

def floor_dist(Tc, a, b):
    Tw = H.cam_to_world(Twc, Tc[a:b + 1])
    P = np.einsum('nij,vj->nvi', Tw[:, :3, :3], V) + Tw[:, None, :3, 3]
    return np.median((P @ pl[:3] + pl[3]).min(1))

rows = []
hdr = ('| phase | frames | type | IoU base | IoU ours | tex R2 base | tex R2 ours | rot speed base (med/p95 deg/fr) '
       '| rot speed ours | trans acc base (cm/fr^2, p95) | ours | dz mean (cm) | dz median (cm) | rot diff to base (mean deg) '
       '| bottom-to-floor base/ours (cm) |')
rows.append(hdr); rows.append('|' + '---|' * hdr.count('|')[:-1] if False else '|' + '---|' * (hdr.count('|') - 1))
for n, a, b, typ in ph + [('ALL', 0, F - 1, '')]:
    sl = slice(a, b + 1)
    vis = OM[a:b + 1].reshape(b - a + 1, -1).sum(1) > 50
    def m(x): return np.mean(x[sl][vis]) if vis.any() else np.nan
    rb = rspeed(Tb[sl]) if b > a else np.zeros(1); rr = rspeed(Tr[sl]) if b > a else np.zeros(1)
    ab = tacc(Tb[sl]) if b - a > 1 else np.zeros(1); ar = tacc(Tr[sl]) if b - a > 1 else np.zeros(1)
    dz = (Tr[sl, 2, 3] - Tb[sl, 2, 3]) * 100
    rd = H.rot_angle_deg(Tb[sl, :3, :3], Tr[sl, :3, :3]).mean()
    rows.append(f'| {n} | {a}-{b} | {typ} | {m(Sb["siou"]):.3f} | {m(Sr["siou"]):.3f} | {m(Sb["ncc"]):.3f} | '
                f'{m(Sr["ncc"]):.3f} | {np.median(rb):.2f}/{np.percentile(rb, 95):.2f} | '
                f'{np.median(rr):.2f}/{np.percentile(rr, 95):.2f} | {np.percentile(ab, 95):.2f} | '
                f'{np.percentile(ar, 95):.2f} | {dz.mean():+.2f} | {np.median(dz):+.2f} | {rd:.1f} | '
                f'{floor_dist(Tb, a, b) * 100:.1f}/{floor_dist(Tr, a, b) * 100:.1f} |')
if args.holdout_eval:
    rows.append('')
    rows.append('| moving segment | held-out (odd) frames | IoU base | IoU ours | tex R2 base | tex R2 ours | chamfer base | ours |')
    rows.append('|---|---|---|---|---|---|---|---|')
    allm = []
    for n, a, b, typ in ph:
        if typ != 'moving': continue
        o = np.arange(a + 1, b + 1, 2)
        o = o[OM[o].reshape(len(o), -1).sum(1) > 50]
        allm.append(o)
        rows.append(f'| {n} | {len(o)} | {Sb["siou"][o].mean():.3f} | {Sr["siou"][o].mean():.3f} | {Sb["ncc"][o].mean():.3f} | '
                    f'{Sr["ncc"][o].mean():.3f} | {Sb["cham"][o].mean():.4f} | {Sr["cham"][o].mean():.4f} |')
    o = np.concatenate(allm)
    rows.append(f'| ALL | {len(o)} | {Sb["siou"][o].mean():.3f} | {Sr["siou"][o].mean():.3f} | {Sb["ncc"][o].mean():.3f} | '
                f'{Sr["ncc"][o].mean():.3f} | {Sb["cham"][o].mean():.4f} | {Sr["cham"][o].mean():.4f} |')
txt = '\n'.join(rows)
print(txt)
if args.md:
    open(args.md, 'w').write(txt + '\n')
np.savez(os.path.splitext(args.result)[0] + '_scores.npz', **{f'b_{k}': v for k, v in Sb.items()},
         **{f'r_{k}': v for k, v in Sr.items()})
