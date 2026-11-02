import numpy as np, torch, cv2, h5py, time, roma
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
from common import *
from render import Renderer
from fitlib import ms_loss, soft_iou

S = 512  # crop size (full res pixels)
class BatchCropRenderer(Renderer):
    def render_boxes(self, T, Kt, boxes):
        # T (B,4,4); boxes (B,2) int x0,y0; crop SxS
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        pc = torch.einsum('bij,nj->bni', T, vh)[..., :3]
        X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
        cx = Kt[0, 2] - boxes[:, 0:1]; cy = Kt[1, 2] - boxes[:, 1:2]
        u = Kt[0, 0] * X + cx * Z; v = Kt[1, 1] * Y + cy * Z
        n, f = 0.05, 20.
        clip = torch.stack([(2 * u + Z) / S - Z, (2 * v + Z) / S - Z, Z * (f + n) / (f - n) - 2 * f * n / (f - n), Z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (S, S))
        ones = torch.ones_like(clip[..., :1])
        m, _ = dr.interpolate(ones, rast, self.f)
        m = dr.antialias(m, rast, clip, self.f)
        return m[..., 0]

def prep_frames(idxs, dil=7):
    f = h5py.File(f'{INP}/masks.h5', 'r')
    ker = np.ones((dil, dil), np.uint8)
    M, W, boxes = [], [], []
    for i in idxs:
        o = f['object'][i]; p = f['person'][i]
        pd = cv2.dilate(p.astype(np.uint8), ker) > 0
        ys, xs = np.nonzero(o)
        if len(xs) == 0: cx, cy = 768, 576
        else: cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        x0 = int(np.clip(cx - S // 2, 0, 1536 - S)); y0 = int(np.clip(cy - S // 2, 0, 1152 - S))
        M.append(o[y0:y0 + S, x0:x0 + S]); W.append(~pd[y0:y0 + S, x0:x0 + S]); boxes.append((x0, y0))
    return (torch.tensor(np.stack(M), dtype=torch.float32, device='cuda'),
            torch.tensor(np.stack(W), dtype=torch.float32, device='cuda'),
            torch.tensor(np.array(boxes), dtype=torch.float32, device='cuda'))

def poses(R0, t0, c, w, dt):
    Rw = roma.rotvec_to_rotmat(w)
    R = Rw @ R0
    Rc = (R0 @ c[:, None])[..., 0]
    t = Rc - (Rw @ Rc[..., None])[..., 0] + t0 + dt
    return R, t

def fit_traj(Tinit, idxs, iters=400, lam_t=0., lam_r=0., chunk=64, lr=3e-3, verbose=True, data=None, static_mask=None, lam_up=0., shared=None):
    """jointly fit poses of frames idxs (consecutive) to masks. smoothness on 2nd differences (camera frame)."""
    M, W, boxes = data if data is not None else prep_frames(idxs)
    R = BatchCropRenderer(S, S)
    c = R.v.mean(0)
    Kt = torch.tensor(K, dtype=torch.float32, device='cuda')
    B = len(idxs)
    R0 = torch.tensor(Tinit[:, :3, :3], dtype=torch.float32, device='cuda'); t0 = torch.tensor(Tinit[:, :3, 3], dtype=torch.float32, device='cuda')
    w = torch.zeros(B, 3, device='cuda', requires_grad=True); dt = torch.zeros(B, 3, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([w, dt], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        Rm, t = poses(R0, t0, c, w, dt)
        T = torch.zeros(B, 4, 4, device='cuda'); T[:, :3, :3] = Rm; T[:, :3, 3] = t; T[:, 3, 3] = 1
        # data term, chunked; accumulate grads via graph (memory ok for moderate B)
        dl = 0
        for k in range(0, B, chunk):
            r = R.render_boxes(T[k:k + chunk], Kt, boxes[k:k + chunk])
            dl = dl + ms_loss(r, M[k:k + chunk], W[k:k + chunk]).sum()
        loss = dl / B
        if lam_up > 0:
            from floor import LEGN, PN
            upc = torch.tensor(Twc[:3, :3].T @ PN, dtype=torch.float32, device='cuda')
            lg = torch.tensor(LEGN, dtype=torch.float32, device='cuda')
            loss = loss + lam_up * (((Rm @ lg) - upc) ** 2).sum(1).mean()
        if B > 2 and (lam_t > 0 or lam_r > 0):
            # world-ish smoothness on object centre and rotation
            ctr = (Rm @ c[:, None])[..., 0] + t
            acc = ctr[2:] - 2 * ctr[1:-1] + ctr[:-2]
            rel = roma.rotmat_to_rotvec(Rm[:-1].transpose(1, 2) @ Rm[1:])
            racc = rel[1:] - rel[:-1]
            loss = loss + lam_t * (acc ** 2).sum(1).mean() + lam_r * (racc ** 2).sum(1).mean()
        loss.backward(); opt.step()
        if verbose and it % 100 == 0: print(it, loss.item(), dl.item() / B)
    with torch.no_grad():
        Rm, t = poses(R0, t0, c, w, dt)
        T = torch.zeros(B, 4, 4, device='cuda'); T[:, :3, :3] = Rm; T[:, :3, 3] = t; T[:, 3, 3] = 1
        ious = []
        for k in range(0, B, chunk):
            r = R.render_boxes(T[k:k + chunk], Kt, boxes[k:k + chunk])
            ious.append(soft_iou((r > 0.5).float(), M[k:k + chunk], W[k:k + chunk]))
    return T.cpu().double().numpy(), torch.cat(ious).cpu().numpy()

def eval_ious(Tcams, idxs, data=None):
    M, W, boxes = data if data is not None else prep_frames(idxs)
    R = BatchCropRenderer(S, S); Kt = torch.tensor(K, dtype=torch.float32, device='cuda')
    out = []
    with torch.no_grad():
        for k in range(0, len(idxs), 64):
            T = torch.tensor(Tcams[k:k + 64], dtype=torch.float32, device='cuda')
            r = R.render_boxes(T, Kt, boxes[k:k + 64]); out.append(soft_iou((r > 0.5).float(), M[k:k + 64], W[k:k + 64]))
    return torch.cat(out).cpu().numpy()
