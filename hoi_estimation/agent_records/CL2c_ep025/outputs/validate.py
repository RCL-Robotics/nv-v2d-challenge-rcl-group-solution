#!/usr/bin/env python
"""Ground-truth-free validation: how well does a trajectory explain the video?
python validate.py --inputs /work/inputs --traj A.npz [--traj B.npz ...] [--names base,ours]
Per phase: silhouette IoU vs the SAM2 object mask (person-occluded pixels ignored), appearance losses
(colour L1, blueness L1 = strap / spin cue, darkness L1 = cap / end cue), temporal jerkiness, and the
camera-frame depth difference to the baseline."""
import argparse, json, os, sys, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_lib import Renderer, Refiner, FrameData, rot_angle, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--traj', action='append', required=True)
    ap.add_argument('--names', default=None)
    ap.add_argument('--phases', default=None, help='json list of [name,a,b]')
    ap.add_argument('--data_cache', default=None)
    ap.add_argument('--out_json', default=None)
    a = ap.parse_args()
    cam = json.load(open(f'{a.inputs}/camera.json'))
    K = np.array(cam['K']); Tcw = np.linalg.inv(np.array(cam['T_world_cam']))
    D = pickle.load(open(a.data_cache, 'rb')) if a.data_cache else FrameData(f'{a.inputs}/video.mp4', f'{a.inputs}/masks.h5')
    RF = Refiner(Renderer(f'{a.inputs}/object_mesh.glb'), K, D.S)
    F = D.F
    phases = json.loads(a.phases) if a.phases else [['all', 0, F]]
    names = a.names.split(',') if a.names else a.traj
    zb = (Tcw[None] @ np.load(f'{a.inputs}/reconstruction.npz')['T_world_mesh'].astype(np.float64))[:, 2, 3]
    res = {}
    for name, path in zip(names, a.traj):
        T = Tcw[None] @ np.load(path)['T_world_mesh']
        Rm = torch.tensor(T[:, :3, :3], dtype=torch.float32, device=dev)
        t = torch.tensor(T[:, :3, 3], dtype=torch.float32, device=dev)
        L = []; IoU = []
        with torch.no_grad():
            for s in range(0, F, 128):
                idx = np.arange(s, min(F, s + 128))
                img, om, val, x0, y0 = D.batch(idx)
                L.append(torch.stack(RF.losses(Rm[idx], t[idx], img, om, val, x0, y0), -1).cpu().numpy())
                _, al = RF.render(Rm[idx], t[idx], x0, y0)
                m = (al > 0.5).float()
                inter = (m * om).sum((1, 2)); uni = (((m + om) > 0).float() * val).sum((1, 2))
                IoU.append((inter / uni.clamp(min=1)).cpu().numpy())
        L = np.concatenate(L); IoU = np.concatenate(IoU)
        steps = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', T[1:, :3, :3], T[:-1, :3, :3]) - 1) / 2, -1, 1)))
        steps = np.concatenate([[0], steps])
        acc = np.zeros(F); acc[1:-1] = np.linalg.norm(T[2:, :3, 3] - 2 * T[1:-1, :3, 3] + T[:-2, :3, 3], axis=1)
        dz = (T[:, 2, 3] - zb) * 100
        r = {}
        for pn, p0, p1 in phases:
            sl = slice(p0, p1)
            r[pn] = dict(iou=float(IoU[sl].mean()), iou_p10=float(np.percentile(IoU[sl], 10)),
                         sil=float(L[sl, 0].mean()), col=float(L[sl, 1].mean()), blue=float(L[sl, 2].mean()),
                         dark=float(L[sl, 3].mean()), strap=float(L[sl, 4].mean()), step_med=float(np.median(steps[sl])),
                         n_step_gt20=int((steps[sl] > 20).sum()), trans_acc_mm=float(acc[sl].mean() * 1000),
                         dz_mean_cm=float(dz[sl].mean()), dz_median_cm=float(np.median(dz[sl])))
        res[name] = r
    keys = ['iou', 'iou_p10', 'sil', 'col', 'blue', 'dark', 'strap', 'step_med', 'n_step_gt20', 'trans_acc_mm', 'dz_mean_cm', 'dz_median_cm']
    print('| traj | phase | ' + ' | '.join(keys) + ' |')
    print('|' + '---|' * (len(keys) + 2))
    for pn, _, _ in phases:
        for name in names:
            print(f'| {name} | {pn} | ' + ' | '.join('%.3f' % res[name][pn][k] if isinstance(res[name][pn][k], float) else str(res[name][pn][k]) for k in keys) + ' |')
    if a.out_json:
        json.dump(res, open(a.out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
