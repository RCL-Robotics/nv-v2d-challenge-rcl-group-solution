#!/usr/bin/env python
"""No-ground-truth validation: baseline vs result, per phase. Usage: python validate.py <result.npz> [diag.npz]"""
import sys, json, os
import numpy as np, torch, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve as S
from scipy.spatial.transform import Rotation as Rot
inp = '/work/inputs'
res = np.load(sys.argv[1])['T_world_mesh']; dg = np.load(sys.argv[2])
cam = json.load(open(f'{inp}/camera.json')); K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
gp = json.load(open(f'{inp}/ground_plane.json'))['plane']; n = np.array(gp[:3]); d = gp[3]
rec = np.load(f'{inp}/reconstruction.npz'); base = rec['T_world_mesh'].astype(np.float64); kp = rec['human_keypoints_world']
mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry(); V = np.asarray(mesh.vertices)[::20]
a_end, b_start = int(dg['a_end']), int(dg['b_start']); kappa = float(dg['kappa']); Fn = len(res)
G, Wt, X0 = S.build_observations(inp, K, Tcw, kp)
R = S.SilRenderer(*S.decimated_mesh(mesh))
dm = np.array([0.9998, -0.0132, -0.0162])   # mesh bottom-face normal (fitted to the flat base, points out of the base)
def iou(Tw, scale_free):
    out = np.zeros(Fn)
    for s in range(0, Fn, 128):
        idx = list(range(s, min(s + 128, Fn))); o = S.Obs(G, Wt, X0, idx)
        Tc = np.array([Tcw @ Tw[i] for i in idx])
        if scale_free: Tc[:, :3, 3] /= kappa
        with torch.no_grad():
            sil = R.sil(torch.tensor(Tc, dtype=torch.float32, device='cuda'), K, o.X0, blur=0) > 0.5
        g = o.G > 0.5; w = o.W > 0.5
        out[idx] = (((sil & g) & w).sum((1, 2)).float() / ((sil | g) & w).sum((1, 2)).float().clamp_min(1)).cpu().numpy()
    return out
def per_frame(Tw):
    P = np.einsum('fij,vj->fvi', Tw[:, :3, :3], V) + Tw[:, None, :3, 3]
    floor = (P @ n + d).min(1)
    tilt = np.degrees(np.arccos(np.clip(np.einsum('fij,j->fi', Tw[:, :3, :3], dm) @ (-n / np.linalg.norm(n)), -1, 1)))
    hand = kp[:, 21:42].mean(1)    # holding hand (projects onto the hand that grips the handle)
    hd = np.linalg.norm(P - hand[:, None], axis=2).min(1)
    dR = np.r_[0, [np.degrees(Rot.from_matrix(Tw[i, :3, :3].T @ Tw[i + 1, :3, :3]).magnitude()) for i in range(Fn - 1)]]
    dt = np.r_[0, np.linalg.norm(np.diff(Tw[:, :3, 3], axis=0), axis=1)] * 100
    zc = np.array([(Tcw @ T)[2, 3] for T in Tw])
    return dict(floor=floor, tilt=tilt, hand=hd, dR=dR, dt=dt, zc=zc)
rows = []
ib = {sf: iou(base, sf) for sf in (False, True)}; ir = {sf: iou(res, sf) for sf in (False, True)}
pb, pr = per_frame(base), per_frame(res)
add = np.linalg.norm(np.einsum('fij,vj->fvi', res[:, :3, :3], V) + res[:, None, :3, 3]
                     - np.einsum('fij,vj->fvi', base[:, :3, :3], V) - base[:, None, :3, 3], axis=2).mean(1) * 100
phases = {'A static': range(0, a_end), 'moving': range(a_end, b_start), 'B static': range(b_start, Fn), 'all': range(Fn)}
print(f'phases A [0,{a_end}) moving [{a_end},{b_start}) B [{b_start},{Fn}); kappa {kappa:.3f}')
hdr = '| metric | ' + ' | '.join(f'{k} base | {k} ours' for k in phases) + ' |'
print(hdr); print('|' + '---|' * (1 + 2 * len(phases)))
def line(name, fb, fr, fmt='{:.3f}'):
    cells = []
    for k, r in phases.items():
        r = list(r); cells += [fmt.format(fb(r)), fmt.format(fr(r))]
    print(f'| {name} | ' + ' | '.join(cells) + ' |')
line('silhouette IoU, scale-free (mean)', lambda r: ib[True][r].mean(), lambda r: ir[True][r].mean())
line('silhouette IoU, scale-free (10th pct)', lambda r: np.percentile(ib[True][r], 10), lambda r: np.percentile(ir[True][r], 10))
line('silhouette IoU, as-is metric depth (mean)', lambda r: ib[False][r].mean(), lambda r: ir[False][r].mean())
ho = lambda r: [i for i in r if i % 3 != 0] or r
line('scale-free IoU on held-out frames (i%3!=0)', lambda r: ib[True][ho(r)].mean(), lambda r: ir[True][ho(r)].mean())
line('frame-to-frame rotation, median (deg)', lambda r: np.median(pb['dR'][r[1:]]), lambda r: np.median(pr['dR'][r[1:]]), '{:.2f}')
line('frame-to-frame rotation, 99th pct (deg)', lambda r: np.percentile(pb['dR'][r[1:]], 99), lambda r: np.percentile(pr['dR'][r[1:]], 99), '{:.2f}')
line('frame-to-frame translation, median (cm)', lambda r: np.median(pb['dt'][r[1:]]), lambda r: np.median(pr['dt'][r[1:]]), '{:.2f}')
line('camera depth std over phase (cm)', lambda r: pb['zc'][r].std() * 100, lambda r: pr['zc'][r].std() * 100, '{:.1f}')
line('base tilt vs floor normal, median (deg)', lambda r: np.median(pb['tilt'][r]), lambda r: np.median(pr['tilt'][r]), '{:.1f}')
line('lowest mesh point above plane, median (cm)', lambda r: np.median(pb['floor'][r]) * 100, lambda r: np.median(pr['floor'][r]) * 100, '{:.1f}')
line('lowest mesh point above plane, min (cm)', lambda r: pb['floor'][r].min() * 100, lambda r: pr['floor'][r].min() * 100, '{:.1f}')
line('hand-keypoint centroid to mesh, median (cm)', lambda r: np.median(pb['hand'][r]) * 100, lambda r: np.median(pr['hand'][r]) * 100, '{:.1f}')
print('\nmean point distance result vs baseline (cm) per phase:', {k: round(float(add[list(r)].mean()), 2) for k, r in phases.items()})
cp = rec['contact_prob_right_left'].max(1) > 0.5
print('hand-to-mesh (cm) on contact frames: base %.1f ours %.1f' % (np.median(pb['hand'][cp]) * 100, np.median(pr['hand'][cp]) * 100))
np.savez(os.path.join(os.path.dirname(sys.argv[1]), 'validation_perframe.npz'), iou_base=ib[True], iou_ours=ir[True], add_vs_base=add, **{f'b_{k}': v for k, v in pb.items()}, **{f'r_{k}': v for k, v in pr.items()})
