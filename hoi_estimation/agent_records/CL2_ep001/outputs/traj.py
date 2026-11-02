"""Joint temporal refinement of a pose trajectory: image (silhouette) terms + SE(3) smoothness."""
import numpy as np
import torch

from fit import so3_exp, losses


def viterbi(U, R, P, g_rot=0.02, free_deg=3.0, g_pos=2.0):
    """U (T,N) unary, R (T,N,3,3), P (T,N,3). transition: g_rot*max(angle-free,0) + g_pos*|dp| (m)."""
    T, N = U.shape
    D = U[0].copy()
    bp = np.zeros((T, N), int)
    for t in range(1, T):
        Rr = np.einsum("aij,bkj->abik", R[t - 1], R[t])
        c = (np.trace(Rr, axis1=-2, axis2=-1) - 1) / 2
        ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
        dp = np.linalg.norm(P[t - 1][:, None] - P[t][None], axis=-1)
        tot = D[:, None] + g_rot * np.maximum(ang - free_deg, 0) + g_pos * dp
        bp[t] = tot.argmin(0)
        D = tot.min(0) + U[t]
    path = np.zeros(T, int)
    path[-1] = D.argmin()
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def joint_refine(fit, fd, R0, p0, img_w, R_fix_l=None, p_fix_l=None, R_fix_r=None, p_fix_r=None,
                 iters=150, lr_r=0.01, lr_t=0.004, lam_acc_r=5.0, lam_vel_r=0.05, lam_acc_p=300.0,
                 lam_prior=0.0, chunk=130, w_dt=2.0, fix_depth=True, global_scale=False, scale=1.0, floor=None, log=None):
    """R0 (T,3,3), p0 (T,3) init for consecutive frames; fd: FrameData for those frames (same order).
    img_w (T,) weight of image term per frame (0 = held out / unreliable).
    R_fix_l/p_fix_l: (k,3,3)/(k,3) fixed poses immediately before the segment (anchors), same for right."""
    dev = fit.dev
    T = len(p0)
    R0t = torch.tensor(R0, dtype=torch.float32, device=dev)
    p0t = torch.tensor(p0, dtype=torch.float32, device=dev)
    w = torch.zeros(T, 3, device=dev, requires_grad=True)
    dp = torch.zeros(T, 3, device=dev, requires_grad=True)
    ls = torch.full((1,), float(np.log(scale)), device=dev, requires_grad=global_scale)
    iw = torch.tensor(img_w, dtype=torch.float32, device=dev)
    groups = [{"params": [w], "lr": lr_r}, {"params": [dp], "lr": lr_t}]
    if global_scale:
        groups.append({"params": [ls], "lr": 0.003})
    opt = torch.optim.Adam(groups)
    tt = lambda a: None if a is None else torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    Rl, pl, Rr, pr = tt(R_fix_l), tt(p_fix_l), tt(R_fix_r), tt(p_fix_r)

    def cur():
        Rm = so3_exp(w) @ R0t
        d = torch.cat([dp[:, :2], torch.zeros_like(dp[:, 2:])], 1) if fix_depth else dp
        return Rm, p0t + d

    for it in range(iters):
        opt.zero_grad()
        tot_img = 0.0
        for c0 in range(0, T, chunk):
            sl = slice(c0, min(c0 + chunk, T))
            Rm, p = cur()
            idx = torch.arange(sl.start, sl.stop, device=dev)
            sub = fd.sub(list(range(sl.start, sl.stop)))
            # global scale: the silhouette sees the trajectory scaled along the viewing rays; the
            # scale absorbs a constant mask-size bias and is discarded afterwards (depth re-anchoring)
            L = losses(fit, Rm[idx], p[idx] * torch.exp(ls), sub, 0.0)
            li = ((1 - L["iou"]) + w_dt * L["dt"]) * iw[idx]
            li.sum().backward()
            tot_img += li.sum().item()
        Rm, p = cur()
        Rs, ps = Rm, p
        if Rl is not None:
            Rs, ps = torch.cat([Rl, Rs]), torch.cat([pl, ps])
        if Rr is not None:
            Rs, ps = torch.cat([Rs, Rr]), torch.cat([ps, pr])
        acc_r = ((Rs[2:] - 2 * Rs[1:-1] + Rs[:-2]) ** 2).sum()
        vel_r = ((Rs[1:] - Rs[:-1]) ** 2).sum()
        acc_p = ((ps[2:] - 2 * ps[1:-1] + ps[:-2]) ** 2).sum()
        sm = lam_acc_r * acc_r + lam_vel_r * vel_r + lam_acc_p * acc_p
        if floor is not None:
            sm = sm + floor(Rm, p).sum()
        if lam_prior > 0:
            sm = sm + lam_prior * ((Rm - R0t) ** 2).sum()
        sm.backward()
        opt.step()
        if log is not None and (it % 25 == 0 or it == iters - 1):
            log(f"  it {it} scale {torch.exp(ls).item():.4f} img {tot_img:.3f} acc_r {acc_r.item():.4f} vel_r {vel_r.item():.4f} acc_p {acc_p.item():.6f}")
    with torch.no_grad():
        Rm, p = cur()
    return Rm.cpu().numpy().astype(np.float64), p.cpu().numpy().astype(np.float64)


def eval_frames(fit, fd, Rm, p, chunk=130, scale=1.0):
    """returns per-frame iou, dt, app_red, app_val."""
    from fit import app_score
    out = []
    dev = fit.dev
    for c0 in range(0, len(p), chunk):
        ids = list(range(c0, min(c0 + chunk, len(p))))
        sub = fd.sub(ids)
        Rt = torch.tensor(Rm[ids], dtype=torch.float32, device=dev)
        pt = torch.tensor(p[ids] * scale, dtype=torch.float32, device=dev)
        with torch.no_grad():
            o = fit.render(Rt, pt, sub.K)
            L = losses(fit, Rt, pt, sub, 0.0)
            a1, a2 = app_score(o["color"], o["hard"], sub.img, sub.M, sub.V)
        out.append(np.stack([L["iou"].cpu().numpy(), L["dt"].cpu().numpy(), a1.cpu().numpy(), a2.cpu().numpy()], 1))
    return np.concatenate(out)
