"""Validation without ground truth: image-evidence + temporal metrics, per phase, for several trajectories."""
import sys, json
import numpy as np, h5py, torch
sys.path.insert(0, '/work/outputs')
from common import *
from fit import FrameData, fit_poses, sample_points, CENTER
from photo import photo_scores
from decode import centres

def evaluate(trajs, phases, inp='/work/inputs', video=None, step=2):
    """trajs: dict name -> (F,4,4) world poses. Returns dict of per-phase metrics."""
    K, Twc, _ = load_camera(inp); Tcw = np.linalg.inv(Twc)
    f = h5py.File(f'{inp}/masks.h5'); R = Renderer(inp)
    pts = sample_points(load_mesh(inp)[4], 1200)
    F = len(next(iter(trajs.values())))
    frames = list(range(0, F, step))
    cam = {k: Tcw @ v for k, v in trajs.items()}
    base = cam.get('baseline')
    m = {k: {'sil': np.full(F, np.nan), 'iou': np.full(F, np.nan), 'photo': np.full(F, np.nan)} for k in trajs}
    for s in range(0, len(frames), 12):
        fr = frames[s:s + 12]
        ob = [f['object'][i] for i in fr]; pe = [f['person'][i] for i in fr]
        data = {i: FrameData(ob[k], pe[k], n_pix=800) for k, i in enumerate(fr)}
        for name, T in cam.items():
            _, L = fit_poses(K, T[fr], fr, data, pts, iters=0)
            m[name]['sil'][fr] = L['loss']
        for k, i in enumerate(fr):
            if ob[k].sum() < 50: continue
            for name, T in cam.items():
                o = R.render(K, T[i:i + 1], rgb=False)['mask'][0].cpu().numpy()
                valid = ~(pe[k] & ~ob[k])
                m[name]['iou'][i] = ((o & ob[k]) & valid).sum() / max(1, ((o | ob[k]) & valid).sum())
                if video is not None:
                    m[name]['photo'][i] = photo_scores(R, K, video[i], ob[k], pe[k], T[i:i + 1])[0]
    rows = []
    for (a, b, pname) in phases:
        for name, T in cam.items():
            sl = slice(a, b + 1)
            dR = rot_angle(T[a + 1:b + 1, :3, :3], T[a:b, :3, :3])
            c = centres(T[sl]); acc = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1) if b - a > 2 else np.zeros(1)
            row = dict(phase=pname, frames=f'{a}-{b}', traj=name,
                       sil=np.nanmean(m[name]['sil'][sl]), iou=np.nanmean(m[name]['iou'][sl]),
                       photo=np.nanmean(m[name]['photo'][sl]),
                       rot_step_med=np.median(dR) if len(dR) else 0, rot_jumps15=int((dR > 15).sum()),
                       acc_cm=np.median(acc) * 100)
            if base is not None:
                dz = (T[sl, 2, 3] - base[sl, 2, 3]) * 100
                row.update(dz_mean_cm=dz.mean(), dz_med_cm=np.median(dz),
                           rot_vs_base_med=np.median(rot_angle(T[sl, :3, :3], base[sl, :3, :3])))
            rows.append(row)
    return rows, m

def table(rows):
    keys = list(rows[0].keys())
    s = '| ' + ' | '.join(keys) + ' |\n|' + '---|' * len(keys) + '\n'
    for r in rows:
        s += '| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in r.values()) + ' |\n'
    return s
