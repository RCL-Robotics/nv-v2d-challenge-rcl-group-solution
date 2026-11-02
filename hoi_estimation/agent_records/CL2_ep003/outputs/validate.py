"""Ground-truth-free validation of a trajectory against the baseline (per phase).
python validate.py --inputs /work/inputs --result object_poses_world.npz [--other name=path ...]
Metrics per phase:
  iou      : silhouette IoU with the SAM2 object mask over pixels not covered by the person (no shift search)
  ncc      : appearance NCC (blurred gray) over the rendered, unoccluded region
  logo     : |logo position across body (render) - observed| (fraction of width), static phases, visible logo
  dR, dt   : mean frame-to-frame rotation (deg) / body-centre translation (cm) change (jitter)
  hand2d   : contact frames: 2-D distance (px) between the projected handle centre and the projected grasping
             hand (mean of its 21 keypoints)
  hand3d   : same in 3-D (cm)
  dz_mean/median: camera-frame depth difference of the body centre vs. the baseline (cm)
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--result', required=True); ap.add_argument('--other', nargs='*', default=[])
    ap.add_argument('--crops', default=None)
    args = ap.parse_args(); inp = args.inputs
    m, v, f, uv, tex = load_mesh(f'{inp}/object_mesh.glb'); cam, K, Twc, rec = load_inputs(inp); Tcw = np.linalg.inv(Twc)
    c0, u = body_axis(v)
    if args.crops and os.path.exists(args.crops):
        C = dict(np.load(args.crops))
    else:
        cr = Crops(inp, 160); C = dict(img=(cr.img * 255).astype(np.uint8), om=cr.om, pm=cr.pm, off=cr.off)
    R = Renderer(v, f, uv, tex); Fi = Fitter(R, K, C, c0)
    trajs = {'baseline': Tcw[None] @ rec['T_world_mesh'].astype(np.float64),
             'result': Tcw[None] @ np.load(args.result)['T_world_mesh']}
    for o in args.other:
        n, p = o.split('='); trajs[n] = Tcw[None] @ np.load(p)['T_world_mesh']
    F = len(trajs['baseline'])
    tr = mask_tracks(inp); st = static_phases(tr)
    segs = []; prev = 0
    for (a, b) in st:
        if a > prev: segs.append(('carry', prev, a))
        segs.append(('static', a, b)); prev = b
    if prev < F: segs.append(('carry', prev, F))
    kp = np.einsum('ij,fkj->fki', Tcw[:3, :3], rec['human_keypoints_world'].astype(np.float64)) + Tcw[:3, 3]
    cp = rec['contact_prob_right_left'].max(1)
    hv = v[v[:, 0] > 0.035].mean(0)          # handle centre (mesh frame)
    res = {}
    for name, T in trajs.items():
        iou = np.zeros(F); ncc = np.zeros(F)
        for s in range(0, F, 64):
            fr = np.arange(s, min(F, s + 64))
            with torch.no_grad():
                i_, n_ = Fi.losses(torch.tensor(T[fr], device='cuda', dtype=torch.float32), fr, Fi.batch(fr))
            iou[fr] = i_.cpu().numpy(); ncc[fr] = n_.cpu().numpy()
        cen = T[:, :3, :3] @ c0 + T[:, :3, 3]
        hc = T[:, :3, :3] @ hv + T[:, :3, 3]
        dR = np.r_[0, geodesic_deg(T[1:, :3, :3], T[:-1, :3, :3])]
        dt = np.r_[0, np.linalg.norm(np.diff(cen, axis=0), axis=1) * 100]
        hands = np.stack([kp[:, 21:42].mean(1), kp[:, 42:63].mean(1)], 1)
        hsel = hands[np.arange(F), np.argmin(np.linalg.norm(hands - cen[:, None], axis=2), 1)]
        proj = lambda X: (X @ K.T)[:, :2] / X[:, 2:3]
        h2d = np.linalg.norm(proj(hc) - proj(hsel), axis=1); h3d = np.linalg.norm(hc - hsel, axis=1) * 100
        dz = (cen[:, 2] - (trajs['baseline'][:, :3, :3] @ c0 + trajs['baseline'][:, :3, 3])[:, 2]) * 100
        res[name] = dict(iou=iou, ncc=ncc, dR=dR, dt=dt, h2d=h2d, h3d=h3d, dz=dz)
    contact = cp > 0.5
    print('%-7s %-9s %-10s %6s %6s %6s %6s %7s %7s %8s %8s' % ('phase', 'frames', 'traj', 'iou', 'ncc', 'dR', 'dt', 'hand2d', 'hand3d', 'dz_mean', 'dz_med'))
    rows = []
    for kind, a, b in segs + [('ALL', 0, F)]:
        sl = slice(a, b); cm = contact[sl]
        for name in trajs:
            r = res[name]
            row = (kind, '%d-%d' % (a, b - 1), name, r['iou'][sl].mean(), r['ncc'][sl].mean(), r['dR'][a + 1:b].mean(), r['dt'][a + 1:b].mean(),
                   r['h2d'][sl][cm].mean() if cm.any() else np.nan, r['h3d'][sl][cm].mean() if cm.any() else np.nan,
                   r['dz'][sl].mean(), np.median(r['dz'][sl]))
            rows.append(row)
            print('%-7s %-9s %-10s %6.3f %6.3f %6.2f %6.2f %7.1f %7.1f %8.2f %8.2f' % row)
    out = os.path.splitext(args.result)[0] + '_validation.json'
    json.dump([list(map(lambda x: x if isinstance(x, str) else float(x), r)) for r in rows], open(out, 'w'), indent=0)


if __name__ == '__main__':
    main()
