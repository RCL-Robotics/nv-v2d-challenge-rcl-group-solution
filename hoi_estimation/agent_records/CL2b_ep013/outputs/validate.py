#!/usr/bin/env python
"""Ground-truth-free validation: how well does a trajectory explain the video?  Baseline vs result, per phase.
usage: python validate.py --inputs /work/inputs --result object_poses_world.npz --log object_poses_world_log.json"""
import argparse, json, sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solve
from solve import *

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--result', default='/work/outputs/object_poses_world.npz'); ap.add_argument('--log', default=None)
    ap.add_argument('--step', type=int, default=3)
    a = ap.parse_args()
    D = Data(a.inputs); rc = RC(D.mesh)
    Tw = np.load(a.result)['T_world_mesh']; Tc = np.linalg.inv(D.Twc)[None] @ Tw
    log = json.load(open(a.log or os.path.splitext(a.result)[0] + '_log.json'))
    gp = json.load(open(f'{a.inputs}/ground_plane.json'))['plane']; n = np.array(gp[:3]); d0 = gp[3]
    V = np.asarray(D.mesh.vertices)[::20]
    def evalseq(Tcs, frames):
        r = []
        for t in frames:
            if D.om[t].sum() < 50: continue
            ob = make_obs(D, t, Tcs[t])
            s = score(rc, [ob], Tcs[t][None, :3, :3], Tcs[t][None, :3, 3])[0]
            # hard IoU on valid pixels
            o = rc.render(torch.tensor(Tcs[t][None, :3, :3], dtype=torch.float32, device=dev), torch.tensor(Tcs[t][None, :3, 3], dtype=torch.float32, device=dev), ob.Kc, ob.out, ob.out, color=False)
            h = (o['hard'][0] > 0).float(); m = (ob.om[0] > 0.5).float(); v = ob.valid[0]
            iou = ((h * m * v).sum() / (((h + m) > 0).float() * v).sum().clamp(min=1)).item()
            r.append([iou, 1 - s[2]])
        return np.array(r)
    rows = []
    for (s, e, kind) in log['phases']:
        fr = range(s, e + 1, a.step)
        rb = evalseq(D.Tc_base, fr); ro = evalseq(Tc, fr)
        def smooth(T):
            dr = [rdist(T[t], T[t + 1]) for t in range(s, e)]; dt = [np.linalg.norm(T[t + 1][:3, 3] - T[t][:3, 3]) * 100 for t in range(s, e)]
            return np.median(dr), np.percentile(dr, 95), np.median(dt), np.percentile(dt, 95)
        sb = smooth(D.Tc_base); so = smooth(Tc)
        dz = (Tc[s:e + 1, 2, 3] - D.Tc_base[s:e + 1, 2, 3]) * 100
        rot = np.array([rdist(Tc[t], D.Tc_base[t]) for t in range(s, e + 1)])
        hb = np.array([((V @ Tw0[:3, :3].T + Tw0[:3, 3]) @ n + d0).min() for Tw0 in D.Tw_base[s:e + 1:a.step]]) * 100
        ho = np.array([((V @ Tw0[:3, :3].T + Tw0[:3, 3]) @ n + d0).min() for Tw0 in Tw[s:e + 1:a.step]]) * 100
        rows.append((f'{kind} {s}-{e}', e - s + 1, rb[:, 0].mean(), ro[:, 0].mean(), rb[:, 1].mean(), ro[:, 1].mean(),
                     sb[0], so[0], sb[1], so[1], sb[2], so[2], dz.mean(), np.median(dz), np.median(rot), np.median(hb), np.median(ho)))
    hdr = ('phase', 'n', 'IoU base', 'IoU ours', 'NCC base', 'NCC ours', 'dR med base', 'dR med ours', 'dR p95 base', 'dR p95 ours',
           'dt med base(cm)', 'dt med ours(cm)', 'dz mean(cm)', 'dz med(cm)', 'rot ours-vs-base med(deg)', 'bottom h base(cm)', 'bottom h ours(cm)')
    print('| ' + ' | '.join(hdr) + ' |'); print('|' + '---|' * len(hdr))
    for r in rows: print('| ' + ' | '.join([r[0], str(r[1])] + [f'{x:.2f}' if abs(x) < 10 else f'{x:.1f}' for x in r[2:]]) + ' |')

if __name__ == '__main__':
    main()
