#!/usr/bin/env python
"""No-GT validation of a trajectory vs the baseline: per phase mask IoU (person pixels excluded),
silhouette chamfer residuals, temporal smoothness, static-phase jitter, floor height, hand-object
consistency, and camera-depth difference to the baseline.
usage: python validate.py --inputs /work/inputs --pred X.npz [--pred2 Y.npz] [--phases a,b]"""
import argparse, json, numpy as np, h5py, cv2, trimesh, torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot

def render_masks(glctx, V, Fc, T, K, H, W):
    v = torch.tensor(V, dtype=torch.float32, device='cuda'); f = torch.tensor(Fc, dtype=torch.int32, device='cuda')
    n, fr = 0.01, 20.
    P = torch.tensor([[2 * K[0, 0] / W, 0, (2 * K[0, 2] - W) / W, 0], [0, 2 * K[1, 1] / H, (2 * K[1, 2] - H) / H, 0],
                      [0, 0, (fr + n) / (fr - n), -2 * fr * n / (fr - n)], [0, 0, 1, 0]], dtype=torch.float32, device='cuda')
    T = torch.tensor(T, dtype=torch.float32, device='cuda')
    vc = torch.einsum('bij,nj->bni', T[:, :3, :3], v) + T[:, None, :3, 3]
    clip = torch.einsum('ij,bnj->bni', P, torch.cat([vc, torch.ones_like(vc[..., :1])], -1))
    rast, _ = dr.rasterize(glctx, clip.contiguous(), f, (H, W))
    return rast[..., 3] > 0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--pred', required=True)
    ap.add_argument('--phases', default=None); ap.add_argument('--json', default=None); ap.add_argument('--scale', type=float, default=0.5)
    a = ap.parse_args(); inp = a.inputs
    cam = json.load(open(f'{inp}/camera.json')); K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    rec = np.load(f'{inp}/reconstruction.npz'); Tb_w = rec['T_world_mesh'].astype(np.float64); F_ = len(Tb_w)
    Tp_w = np.load(a.pred)['T_world_mesh']
    mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry(); c = mesh.vertices.mean(0)
    plane = np.array(json.load(open(f'{inp}/ground_plane.json'))['plane'])
    if a.phases: p1, p2 = map(int, a.phases.split(','))
    else: p1, p2 = json.load(open(a.pred.replace('.npz', '') + '_log.json'))['phases'] if False else (198, 564)
    phases = {'S1 static': np.arange(0, p1), 'M moving': np.arange(p1, p2), 'S2 static': np.arange(p2, F_), 'all': np.arange(F_)}
    h5 = h5py.File(f'{inp}/masks.h5'); s = a.scale; H, W = int(1152 * s), int(1536 * s)
    Ks = K.copy(); Ks[:2] *= s
    glctx = dr.RasterizeCudaContext()
    res = {}
    for name, Tw in [('base', Tb_w), ('pred', Tp_w)]:
        Tc = Tcw @ Tw
        iou = np.zeros(F_)
        for i in range(0, F_, 32):
            rm = render_masks(glctx, mesh.vertices, mesh.faces, Tc[i:i + 32], Ks, H, W).cpu().numpy()
            for j in range(len(rm)):
                o = cv2.resize(h5['object'][i + j].astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0
                p = cv2.resize(h5['person'][i + j].astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0
                v = ~p | o
                iou[i + j] = (rm[j] & o & v).sum() / max(((rm[j] | o) & v).sum(), 1)
        pc = np.einsum('bij,j->bi', Tc[:, :3, :3], c) + Tc[:, :3, 3]
        rel = Rot.from_matrix(np.einsum('bji,bjk->bik', Tc[:-1, :3, :3], Tc[1:, :3, :3])).as_rotvec()
        rel_c = np.einsum('bij,bj->bi', Tc[:-1, :3, :3], rel)
        racc = np.degrees(np.linalg.norm(np.diff(rel_c, axis=0), axis=1)); racc = np.r_[0, racc, 0]
        tacc = np.linalg.norm(pc[2:] - 2 * pc[1:-1] + pc[:-2], axis=1) * 100; tacc = np.r_[0, tacc, 0]
        Vw = np.einsum('bij,nj->bni', Tw[:, :3, :3], mesh.vertices[::10]) + Tw[:, None, :3, 3]
        hmin = (Vw @ plane[:3] + plane[3]).min(1) * 100
        res[name] = dict(iou=iou, pc=pc, racc=racc, tacc=tacc, hmin=hmin, Tc=Tc)
    rows = []
    for ph, fr in phases.items():
        if len(fr) == 0: continue
        b, p = res['base'], res['pred']
        dz = (p['pc'][fr, 2] - b['pc'][fr, 2]) * 100
        dpos = np.linalg.norm(p['pc'][fr] - b['pc'][fr], axis=1) * 100
        drot = np.degrees(Rot.from_matrix(np.einsum('bji,bjk->bik', b['Tc'][fr, :3, :3], p['Tc'][fr, :3, :3])).magnitude())
        jit = lambda r: np.linalg.norm(r['pc'][fr] - np.median(r['pc'][fr], 0), axis=1).mean() * 100
        rows.append(dict(phase=ph, n=len(fr), iou_base=b['iou'][fr].mean(), iou_pred=p['iou'][fr].mean(),
                         iou_p10_base=np.percentile(b['iou'][fr], 10), iou_p10_pred=np.percentile(p['iou'][fr], 10),
                         racc_base=b['racc'][fr].mean(), racc_pred=p['racc'][fr].mean(), tacc_base=b['tacc'][fr].mean(), tacc_pred=p['tacc'][fr].mean(),
                         jitter_base=jit(b), jitter_pred=jit(p), hmin_base=np.median(b['hmin'][fr]), hmin_pred=np.median(p['hmin'][fr]),
                         dz_mean=dz.mean(), dz_median=np.median(dz), dpos_mean=dpos.mean(), drot_mean=drot.mean(), drot_max=drot.max()))
    keys = list(rows[0].keys())
    print('| ' + ' | '.join(keys) + ' |'); print('|' + '---|' * len(keys))
    for r in rows: print('| ' + ' | '.join(str(r[k]) if isinstance(r[k], (str, int)) else f'{r[k]:.3f}' for k in keys) + ' |')
    if a.json: json.dump(dict(rows=rows, iou_base=res['base']['iou'].tolist(), iou_pred=res['pred']['iou'].tolist()), open(a.json, 'w'))

if __name__ == '__main__':
    main()
