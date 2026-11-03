import sys, numpy as np, torch, cv2, time
from common import *
from fitlib import *
torch.manual_seed(0)
def prep(idxs, dil=7):
    O, P = load_masks(idxs)
    ker = np.ones((dil, dil), np.uint8)
    Pd = np.stack([cv2.dilate(p.astype(np.uint8), ker) > 0 for p in P])
    ys, xs = np.nonzero(O.any(0))
    x0, x1, y0, y1 = max(xs.min() - 80, 0), min(xs.max() + 80, 1536), max(ys.min() - 80, 0), min(ys.max() + 80, 1152)
    box = (x0, y0, x1 - x0, y1 - y0)
    m = torch.tensor(O[:, y0:y1, x0:x1], dtype=torch.float32, device='cuda')
    w = torch.tensor(~Pd[:, y0:y1, x0:x1], dtype=torch.float32, device='cuda')
    return m, w, box

def fit_single(T_init, idxs, iters=300, lr=(3e-3, 3e-3), fix_depth=False, verbose=True):
    m, w, box = prep(idxs)
    CR = CropRenderer(box[2], box[3])
    c = CR.v.mean(0)
    Kt = torch.tensor(K, dtype=torch.float32, device='cuda')
    R0 = torch.tensor(T_init[None, :3, :3], dtype=torch.float32, device='cuda'); t0 = torch.tensor(T_init[None, :3, 3], dtype=torch.float32, device='cuda')
    wv = torch.zeros(1, 3, device='cuda', requires_grad=True); dt = torch.zeros(1, 3, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([{'params': [wv], 'lr': lr[0]}, {'params': [dt], 'lr': lr[1]}])
    B = len(idxs)
    for it in range(iters):
        T = pose_from_params(R0, t0, c, wv, dt)
        r, _ = CR.render_crop(T.expand(B, 4, 4).contiguous(), Kt, box)
        loss = ms_loss(r, m, w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it % 50 == 0:
            print(it, loss.item(), soft_iou(r, m, w).mean().item())
    with torch.no_grad():
        T = pose_from_params(R0, t0, c, wv, dt); r, _ = CR.render_crop(T.expand(B, 4, 4).contiguous(), Kt, box)
        iou = soft_iou((r > 0.5).float(), m, w).cpu().numpy()
    return T[0].detach().cpu().double().numpy(), iou

def eval_iou(Tcams, idxs):
    m, w, box = prep(idxs)
    CR = CropRenderer(box[2], box[3]); Kt = torch.tensor(K, dtype=torch.float32, device='cuda')
    out = []
    with torch.no_grad():
        for k in range(0, len(idxs), 16):
            T = torch.tensor(Tcams[k:k + 16], dtype=torch.float32, device='cuda')
            r, _ = CR.render_crop(T, Kt, box); out.append(soft_iou((r > 0.5).float(), m[k:k + 16], w[k:k + 16]).cpu().numpy())
    return np.concatenate(out)
