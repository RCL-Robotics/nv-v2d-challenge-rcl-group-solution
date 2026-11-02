import numpy as np, h5py
from opt import *
def static_phases(inp, thr=0.75, ds=4):
    """static (untouched) phases at the start/end: frames whose object mask overlaps the first/last frame's mask (IoU>thr)"""
    h5 = h5py.File(f'{inp}/masks.h5', 'r'); O = h5['object']; F = O.shape[0]
    m0 = O[0][::ds, ::ds]; m1 = O[F - 1][::ds, ::ds]
    iou = lambda a, b: (a & b).sum() / max(1, (a | b).sum())
    s_end = 0
    while s_end < F and iou(O[s_end][::ds, ::ds], m0) > thr: s_end += 1
    e_start = F - 1
    while e_start > 0 and iou(O[e_start][::ds, ::ds], m1) > thr: e_start -= 1
    return (0, s_end), (e_start + 1, F)

def fit_static(prob, fr, R0s, t0s, zref, steps=150):
    """single pose explaining all frames fr (pooled evidence). R0s/t0s: list of inits; returns best."""
    fr = np.asarray(fr)
    maps = prob.sc.maps(fr)
    mp, mpn = pad_pts([prob.bank.mpts[i] for i in fr], 600)
    op, opn = pad_pts([prob.bank.opts[i] for i in fr], 300)
    G = len(R0s)
    R0t = torch.tensor(np.asarray(R0s), dtype=torch.float32, device=dev); t0t = torch.tensor(np.asarray(t0s), dtype=torch.float32, device=dev)
    zr = torch.tensor(zref, dtype=torch.float32, device=dev)
    om = torch.zeros(G, 3, device=dev, requires_grad=True); dt = torch.zeros(G, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [om], 'lr': 0.02}, {'params': [dt], 'lr': 0.01}])
    B = len(fr)
    for k in range(steps + 1):
        sig = 20. * (8. / 20.) ** (min(k, steps) / steps)
        R = so3_exp(om) @ R0t; t = t0t + dt
        Rb_ = R[None].expand(B, G, 3, 3); tb_ = t[None].expand(B, G, 3)
        ct = prob.costs(fr, Rb_, tb_, maps, mp, mpn, op, opn, zr, sig=sig)
        L = prob.total(ct, W_DEF).mean(0)  # G
        if k == steps: break
        opt.zero_grad(); L.sum().backward(); opt.step()
    L = L.detach().cpu().numpy()
    return R.detach().cpu().numpy().astype(np.float64), t.detach().cpu().numpy().astype(np.float64), L
