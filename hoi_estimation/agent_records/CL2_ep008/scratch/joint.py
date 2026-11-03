import numpy as np, torch, roma, time
from common import *
from traj import BatchCropRenderer, prep_frames, S
from fitlib import ms_loss, soft_iou
from floor import LEGN, PN

def joint_fit(Tinit_params, fmap, data_frames, data, iters=300, lr=3e-3, lam_t=0., lam_r=0., lam_up=0., chunk=48, verbose=True, fix_params=()):
    """Tinit_params (P,4,4) camera-frame initial poses; fmap (F,) param index per frame (time order);
    data_frames: frame indices with mask data (data=prep_frames(data_frames)).
    smoothness: 2nd differences of object centre / rotation across all F frames (static frames share params)."""
    dev = 'cuda'
    M, W, boxes = data
    R = BatchCropRenderer(S, S); c = R.v.mean(0)
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    P = len(Tinit_params)
    R0 = torch.tensor(Tinit_params[:, :3, :3], dtype=torch.float32, device=dev); t0 = torch.tensor(Tinit_params[:, :3, 3], dtype=torch.float32, device=dev)
    w = torch.zeros(P, 3, device=dev, requires_grad=True); dt = torch.zeros(P, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, dt], lr=lr)
    fm = torch.tensor(fmap, device=dev, dtype=torch.long)
    dmap = fm[torch.tensor(data_frames, device=dev, dtype=torch.long)]
    upc = torch.tensor(Twc[:3, :3].T @ PN, dtype=torch.float32, device=dev); lg = torch.tensor(LEGN, dtype=torch.float32, device=dev)
    fixed = torch.zeros(P, 1, device=dev)
    for p in fix_params: fixed[p] = 1
    Bd = len(data_frames)
    def build():
        Rw = roma.rotvec_to_rotmat(w * (1 - fixed)); Rm = Rw @ R0
        Rc = (R0 @ c[:, None])[..., 0]
        t = Rc - (Rw @ Rc[..., None])[..., 0] + t0 + dt * (1 - fixed)
        return Rm, t
    for it in range(iters + 1):
        opt.zero_grad()
        Rm, t = build()
        T = torch.zeros(P, 4, 4, device=dev); T[:, :3, :3] = Rm; T[:, :3, 3] = t; T[:, 3, 3] = 1
        Tleaf = T.detach().requires_grad_(it < iters)
        dl = 0.; ious = []
        for k in range(0, Bd, chunk):
            with torch.set_grad_enabled(it < iters):
                r = R.render_boxes(Tleaf[dmap[k:k + chunk]], Kt, boxes[k:k + chunk])
                l = ms_loss(r, M[k:k + chunk], W[k:k + chunk]).sum() / Bd
                if it < iters: l.backward()
            dl += l.item()
            if it == iters: ious.append(soft_iou((r > 0.5).float(), M[k:k + chunk], W[k:k + chunk]).detach())
        if it == iters: break
        reg = 0.
        if lam_up > 0:
            reg = reg + lam_up * (((Rm @ lg) - upc) ** 2).sum(1).mean()
        if lam_t > 0 or lam_r > 0:
            Rf, tf = Rm[fm], t[fm]
            ctr = (Rf @ c[:, None])[..., 0] + tf
            acc = ctr[2:] - 2 * ctr[1:-1] + ctr[:-2]
            rel = roma.rotmat_to_rotvec(Rf[:-1].transpose(1, 2) @ Rf[1:])
            racc = rel[1:] - rel[:-1]
            reg = reg + lam_t * 1e4 * (acc ** 2).sum(1).mean() + lam_r * 1e2 * (racc ** 2).sum(1).mean()
        if torch.is_tensor(reg):
            (T * Tleaf.grad).sum().add(reg).backward()
        else:
            (T * Tleaf.grad).sum().backward()
        opt.step()
        if verbose and it % 100 == 0:
            print(it, 'data %.5f reg %.5f' % (dl, float(reg)), flush=True)
    return T.detach().cpu().double().numpy(), torch.cat(ious).cpu().numpy(), dl
