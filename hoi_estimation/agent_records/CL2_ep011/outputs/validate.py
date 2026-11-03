#!/usr/bin/env python
"""Ground-truth-free checks of a trajectory against the video, compared with the baseline.
usage: python validate.py --inputs /work/inputs --traj result.npz [--holdout result_holdout.npz --k 5] --phases diag.json
Prints a per-phase table:
  IoU      silhouette IoU with the SAM2 object mask (full-res mesh render; person pixels and, unless the
           book rests on the table, table-top pixels ignored)
  hoIoU    IoU on held-out frames (frames whose silhouette was NOT used by the optimiser in the holdout run)
  app      per-face shading-normalised texture fit (R^2 of grey levels) of the textured render
  jitR/jitT median frame-to-frame rotation (deg) / translation (mm) change in world
  h_min    median lowest mesh point above the released floor plane (cm)
  dz       mean / median camera-depth difference to the baseline (cm)
"""
import argparse, json, sys, numpy as np, cv2, h5py, torch, trimesh
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
sys.path.insert(0, '/work/outputs')
from solve import table_mask, read_video, rotdiff_deg

DEV = 'cuda'


class TexRenderer:
    def __init__(self, mesh, K):
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=DEV)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=DEV)
        uv = np.asarray(mesh.visual.uv).copy(); uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, dtype=torch.float32, device=DEV)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert('RGB')).astype(np.float32) / 255.
        self.tex = torch.tensor(tex, device=DEV)[None].contiguous()
        n = mesh.face_normals; ax = np.abs(n).argmax(1); sg = np.sign(n[np.arange(len(n)), ax])
        self.flab = torch.tensor(np.concatenate([[6], ax * 2 + (sg > 0)]), device=DEV)
        self.ctx = dr.RasterizeCudaContext(); self.K = K

    def render(self, T, box, S):
        x0, y0, side = box; K = self.K; s = S / side
        fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, (K[0, 2] - x0) * s, (K[1, 2] - y0) * s
        T = torch.tensor(T, dtype=torch.float32, device=DEV)
        vc = self.v @ T[:3, :3].T + T[:3, 3]
        x, y, z = vc[:, 0], vc[:, 1], vc[:, 2]; n, f = 0.05, 20.
        clip = torch.stack([2 * fx / S * x + (2 * cx / S - 1) * z, 2 * fy / S * y + (2 * cy / S - 1) * z,
                            z * (f + n) / (f - n) - 2 * f * n / (f - n), z], -1)[None].contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (S, S))
        texc, _ = dr.interpolate(self.uv[None], rast, self.f)
        rgb = dr.texture(self.tex, texc, filter_mode='linear')[0]
        lab = self.flab[rast[0, ..., 3].long()]
        return rgb, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--traj', required=True)
    ap.add_argument('--holdout', default=None)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--phases', required=True)
    ap.add_argument('--step', type=int, default=1)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    inp = a.inputs
    cam = json.load(open(f'{inp}/camera.json')); K = np.array(cam['K']); Twc = np.array(cam['T_world_cam'])
    Tw_b = np.load(f'{inp}/reconstruction.npz')['T_world_mesh'].astype(np.float64); N = len(Tw_b)
    trajs = {'base': Tw_b, 'ours': np.load(a.traj)['T_world_mesh']}
    if a.holdout: trajs['hold'] = np.load(a.holdout)['T_world_mesh']
    Tc = {k: np.linalg.inv(Twc)[None] @ v for k, v in trajs.items()}
    mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    ctr = mesh.bounds.mean(0)
    plane = np.array(json.load(open(f'{inp}/ground_plane.json'))['plane']); nrm, d0 = plane[:3], plane[3]
    diag = json.load(open(a.phases)); phases = diag['phases']
    on_table = np.zeros(N, bool)
    dg = np.load(a.phases.replace('_diag.json', '_diag.npz'))
    on_table = dg['on_table']
    _, frames_bg = read_video(f'{inp}/video.mp4'); tmask = table_mask(frames_bg); del frames_bg
    H5 = h5py.File(f'{inp}/masks.h5', 'r')
    tr = TexRenderer(mesh, K)
    corners = np.array(trimesh.bounds.corners(mesh.bounds))
    res = {k: dict(iou=np.full(N, np.nan), app=np.full(N, np.nan)) for k in Tc}
    cap = cv2.VideoCapture(f'{inp}/video.mp4')
    S = 128
    for i in range(N):
        ok, img = cap.read()
        if i % a.step: continue
        om = H5['object'][i]; pm = H5['person'][i]
        if om.sum() < 60: continue
        ys, xs = np.nonzero(om)
        cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
        pc = corners @ Tc['base'][i, :3, :3].T + Tc['base'][i, :3, 3]; pu = (K @ pc.T).T; pu = pu[:, :2] / pu[:, 2:]
        side = 1.3 * (pu.max(0) - pu.min(0)).max() + 40
        box = (cx - side / 2, cy - side / 2, side); s = S / side
        M = np.float32([[s, 0, -box[0] * s], [0, s, -box[1] * s]])
        o = cv2.warpAffine(om.astype(np.float32), M, (S, S)) > 0.5
        occ = cv2.warpAffine(pm.astype(np.float32), M, (S, S)) > 0.5
        if not on_table[i]: occ |= cv2.warpAffine(tmask.astype(np.float32), M, (S, S)) > 0.5
        valid = ~(occ & ~o)
        im = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        gi = cv2.GaussianBlur(im.mean(-1), (0, 0), 0.7)
        for k in Tc:
            rgb, lab = tr.render(Tc[k][i], box, S)
            lab = lab.cpu().numpy(); m = lab < 6
            inter = (m & o & valid).sum(); uni = ((m | o) & valid).sum()
            res[k]['iou'][i] = inter / max(uni, 1)
            gr = cv2.GaussianBlur(rgb.cpu().numpy().mean(-1), (0, 0), 1.0)
            rs = vs = 0.0
            for fl in range(6):
                sel = (lab == fl) & o & valid
                if sel.sum() < 20: continue
                x = gr[sel]; y = gi[sel]; x = x - x.mean(); y = y - y.mean()
                g = max((x * y).sum() / ((x * x).sum() + 1e-9), 0)
                rs += ((y - g * x) ** 2).sum(); vs += (y * y).sum()
            res[k]['app'][i] = 1 - rs / vs if vs > 0 else np.nan
    holdmask = (np.arange(N) % a.k == a.k // 2)
    # temporal / physical checks
    def jit(Tw):
        dR = rotdiff_deg(Tw[1:, :3, :3], Tw[:-1, :3, :3]); dT = np.linalg.norm(np.diff(Tw[:, :3, 3], axis=0), axis=1) * 1000
        return np.r_[dR, dR[-1]], np.r_[dT, dT[-1]]
    V = np.asarray(mesh.vertices)[::40]
    def hmin(Tw):
        P = np.einsum('bij,nj->bni', Tw[:, :3, :3], V) + Tw[:, None, :3, 3]
        return (P @ nrm + d0).min(1) * 100
    rows = []
    hdr = '| frames | phase | IoU base | IoU ours | hoIoU base | hoIoU ours(held-out) | app base | app ours | jitR base/ours (deg) | jitT base/ours (mm) | h_min base/ours (cm) | dz mean / median (cm) |'
    print(hdr); print('|' + '---|' * 12)
    J = {k: jit(trajs[k]) for k in trajs}; Hm = {k: hmin(trajs[k]) for k in trajs}
    pcb = (Tc['base'][:, :3, :3] @ ctr) + Tc['base'][:, :3, 3]; pco = (Tc['ours'][:, :3, :3] @ ctr) + Tc['ours'][:, :3, 3]
    dz = (pco[:, 2] - pcb[:, 2]) * 100
    def mean(x): x = x[~np.isnan(x)]; return np.mean(x) if len(x) else np.nan
    for (s0, s1, kind) in phases + [(0, N, 'ALL')]:
        sl = slice(s0, s1); hm = holdmask[sl]
        r = [f'{s0}-{s1 - 1}', kind,
             '%.3f' % mean(res['base']['iou'][sl]), '%.3f' % mean(res['ours']['iou'][sl]),
             '%.3f' % mean(res['base']['iou'][sl][hm]),
             '%.3f' % mean(res['hold']['iou'][sl][hm]) if 'hold' in res else '-',
             '%.3f' % mean(res['base']['app'][sl]), '%.3f' % mean(res['ours']['app'][sl]),
             '%.2f / %.2f' % (np.median(J['base'][0][sl]), np.median(J['ours'][0][sl])),
             '%.1f / %.1f' % (np.median(J['base'][1][sl]), np.median(J['ours'][1][sl])),
             '%.1f / %.1f' % (np.median(Hm['base'][sl]), np.median(Hm['ours'][sl])),
             '%.2f / %.2f' % (dz[sl].mean(), np.median(dz[sl]))]
        print('| ' + ' | '.join(r) + ' |'); rows.append(r)
    if a.out:
        np.savez(a.out, **{f'{k}_{m}': res[k][m] for k in res for m in res[k]})


if __name__ == '__main__':
    main()
