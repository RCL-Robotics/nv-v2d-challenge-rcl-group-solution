#!/usr/bin/env python
"""Ground-truth-free validation: how well does a trajectory explain the video? baseline vs result, per phase.
usage: python validate.py --inputs /work/inputs --result object_poses_world.npz --phases 0,117,180,690,790,905,988 --out table.md
"""
import argparse, json, sys, os
import numpy as np, h5py, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve import FrameBank, Problem, OrangeRenderer, evaluate, read_frames, rot_angle_deg, dev, W, H
import nvdiffrast.torch as dr

def silhouette_iou(prob, T_cam, h5, frames):
    """IoU of the rendered silhouette with the SAM2 object mask, ignoring pixels of the person mask"""
    m = prob.mesh
    v = torch.tensor(np.asarray(m.vertices, np.float32), device=dev); f = torch.tensor(np.asarray(m.faces, np.int32), device=dev)
    glctx = dr.RasterizeCudaContext(); K = prob.K; n, fa = 0.05, 20.
    P = torch.tensor([[2 * K[0, 0] / W, 0, 2 * K[0, 2] / W - 1, 0], [0, 2 * K[1, 1] / H, 2 * K[1, 2] / H - 1, 0],
                      [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)], [0, 0, 1, 0]], dtype=torch.float32, device=dev)
    vh = torch.cat([v, torch.ones_like(v[:, :1])], 1); out = []
    for i in frames:
        T = torch.tensor(T_cam[i], dtype=torch.float32, device=dev)
        clip = (vh @ T.T @ P.T) * torch.tensor([1, -1, 1, 1], device=dev, dtype=torch.float32)
        rast, _ = dr.rasterize(glctx, clip[None].contiguous(), f, (H, W))
        r = torch.flip(rast[0, ..., 3] > 0, [0]).cpu().numpy()
        om = h5['object'][i]; pm = h5['person'][i]; valid = ~pm | om
        out.append(((r & om).sum()) / max(1, ((r | om) & valid).sum()))
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--result', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--phases', default='0,117,180,690,790,905,988'); ap.add_argument('--out', default=None)
    ap.add_argument('--bank', default=None)
    a = ap.parse_args()
    bank = FrameBank(a.inputs)
    prob = Problem(a.inputs, bank); F = prob.F; imgs = read_frames(a.inputs); orr = OrangeRenderer(prob)
    h5 = h5py.File(f'{a.inputs}/masks.h5', 'r')
    res = np.load(a.result)['T_world_mesh']
    trajs = {'baseline': prob.T_base_c, 'result': np.einsum('ij,fjk->fik', prob.Tcw, res)}
    tip = np.array([0, 0, 1.015])  # handle end cap (mesh coords)
    met = {}
    for name, Tc in trajs.items():
        R, t = prob.T_to_Rt(Tc)
        Ls = []; cts = {}
        for s in range(0, F, 16):
            fr = np.arange(s, min(F, s + 16)); l, c = evaluate(prob, fr, R[fr][:, None], t[fr][:, None], prob.tb[fr, 2:3])
            Ls.append(l[:, 0]); [cts.setdefault(k, []).append(v[:, 0]) for k, v in c.items()]
        cts = {k: np.concatenate(v) for k, v in cts.items()}
        ncc = np.array([orr.ncc(Tc[i:i + 1], imgs[i])[0] for i in range(F)])
        iou = silhouette_iou(prob, Tc, h5, range(F))
        Xh = np.einsum('fij,nj->fni', R, prob.handle.cpu().numpy().astype(np.float64)) + t[:, None]
        dh = np.linalg.norm(prob.hands_c[:, :, None] - Xh[:, None], axis=-1).min(-1)  # F,2
        hand = np.where(prob.contact > 0.5, dh, np.nan)
        P = prob.P.cpu().numpy().astype(np.float64)
        hgt = (np.einsum('fij,nj->fni', R, P) + t[:, None]) @ prob.n_c_np + prob.d_c
        tipc = np.einsum('fij,j->fi', Tc[:, :3, :3], tip) + Tc[:, :3, 3]; tiph = tipc @ prob.n_c_np + prob.d_c
        angv = np.r_[np.nan, rot_angle_deg(R[1:], R[:-1])]; spd = np.r_[np.nan, np.linalg.norm(np.diff(t, axis=0), axis=1)]
        met[name] = dict(iou=iou, ncc=ncc, geo=np.concatenate(Ls), i2m=cts['i2m'], m2i=cts['m2i'], hand=np.where(np.isnan(hand).all(1), np.nan, np.nan_to_num(hand, nan=1e9).min(1)),
                         floor_min=hgt.min(1), tiph=tiph, angv=angv, spd=spd, z=t[:, 2], R=R)
    ph = [int(x) for x in a.phases.split(',')]
    rows = []
    hdr = '| phase | frames | traj | mask IoU | orange NCC | geo cost | hand dist cm (med) | min height above floor cm (med) | handle-tip height cm (med) | ang. vel deg/f (med / p95) | rot. to baseline deg (med) | dz vs baseline cm (mean / med) |'
    rows.append(hdr); rows.append('|' + '---|' * 12)
    for p0, p1 in zip(ph[:-1], ph[1:]):
        sl = slice(p0, p1)
        for name in trajs:
            m = met[name]; b = met['baseline']
            with np.errstate(all='ignore'):
                hd = np.nanmedian(m['hand'][sl]) * 100 if np.isfinite(m['hand'][sl]).any() else np.nan
                rows.append(f"| {p0}-{p1} | {p1 - p0} | {name} | {np.mean(m['iou'][sl]):.3f} | {np.nanmean(m['ncc'][sl]):.3f} | {np.mean(m['geo'][sl]):.3f} | "
                            f"{hd:.1f} | {np.median(m['floor_min'][sl]) * 100:.1f} | {np.median(m['tiph'][sl]) * 100:.1f} | "
                            f"{np.nanmedian(m['angv'][sl]):.2f} / {np.nanpercentile(m['angv'][sl], 95):.1f} | "
                            f"{np.median(rot_angle_deg(m['R'][sl], b['R'][sl])):.1f} | "
                            f"{np.mean(m['z'][sl] - b['z'][sl]) * 100:.1f} / {np.median(m['z'][sl] - b['z'][sl]) * 100:.1f} |")
    txt = '\n'.join(rows); print(txt)
    if a.out: open(a.out, 'w').write(txt + '\n')
    np.savez(os.path.splitext(a.out)[0] + '_perframe.npz' if a.out else '/tmp/val_perframe.npz',
             **{f'{n}_{k}': v for n, m in met.items() for k, v in m.items()})

if __name__ == '__main__':
    main()
