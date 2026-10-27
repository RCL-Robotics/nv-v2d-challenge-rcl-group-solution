"""Per-frame silhouette registration of the cane (2D chamfer between projected surface samples and the SAM2 mask).

Pose parametrisation for optimisation (camera frame, OpenCV):
  T = [R | t],  R = exp(w) R0  (rotation about the object centre c),  centre in camera = (u,v,log z) param.
"""
import numpy as np
import torch
import cv2
import trimesh

CENTER = np.array([0.0, 0.0, 0.076])  # approx. bbox centre of the cane in mesh coords (pivot for rotations)


def sample_points(mesh, n=3000, seed=0):
    p, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(p, np.float32)


def axis_angle_to_R(w):
    """w (...,3) torch -> (...,3,3), Rodrigues."""
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


class FrameData:
    """Per-frame mask evidence: distance transform of the 'allowed' region and sampled mask pixels."""

    def __init__(self, obj, person, n_pix=1500, seed=0, occl_dilate=5):
        self.obj = obj
        self.area = int(obj.sum())
        allowed = obj.copy()
        if person is not None:
            pd = cv2.dilate(person.astype(np.uint8), np.ones((occl_dilate, occl_dilate), np.uint8)) > 0
            allowed |= pd
        self.dt_allowed = cv2.distanceTransform((~allowed).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        ys, xs = np.nonzero(obj)
        rng = np.random.default_rng(seed)
        if len(xs) > n_pix:
            sel = rng.choice(len(xs), n_pix, replace=False)
            xs, ys = xs[sel], ys[sel]
        self.pix = np.stack([xs, ys], 1).astype(np.float32) + 0.5 if len(xs) else np.zeros((0, 2), np.float32)


def sample_dt(dt, uv):
    """dt (B,H,W) torch, uv (B,N,2) pixel coords -> (B,N) bilinear values (border = large)."""
    B, Hh, Ww = dt.shape
    g = torch.stack([uv[..., 0] / Ww * 2 - 1, uv[..., 1] / Hh * 2 - 1], -1)[:, :, None]
    v = torch.nn.functional.grid_sample(dt[:, None], g, mode='bilinear', padding_mode='border', align_corners=False)
    out = v[:, 0, :, 0]
    outside = (g[..., 0, 0].abs() > 1) | (g[..., 0, 1].abs() > 1)
    return torch.where(outside, out + 50., out)


def huber(x, d):
    return torch.where(x < d, 0.5 * x * x / d, x - 0.5 * d)


def fit_poses(K, T0, frame_ids, data, pts, iters=150, lr_rot=0.02, lr_tr=0.01, z_prior=None, w_z=0.0,
              trunc=40., device='cuda', fix_rot=False, share=None, prior=None, w_prior=(0., 0.), rot_axis=None):
    """Optimise a batch of hypotheses. T0 (B,4,4) camera-frame initial poses; frame_ids (B,) index into data
    (dict fid->FrameData). Returns optimised (B,4,4) poses and per-hypothesis loss components.
    share: optional (B,) int group index; rows of one group share one pose (T0 of its first row is used),
    e.g. one static pose evaluated on many frames. prior: optional (B,4,4) poses with weights
    w_prior=(per deg rotation, per cm centre) squared-penalties."""
    B = len(T0)
    if share is None:
        share = np.arange(B)
    share = np.asarray(share)
    G = int(share.max()) + 1
    first = np.array([np.nonzero(share == g)[0][0] for g in range(G)])
    T0 = np.asarray(T0)[first]
    dev = device
    T0t = torch.tensor(np.asarray(T0), dtype=torch.float32, device=dev)
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    P = torch.tensor(pts, device=dev)  # (N,3)
    c = torch.tensor(CENTER, dtype=torch.float32, device=dev)
    R0 = T0t[:, :3, :3]
    c_cam0 = (R0 @ c) + T0t[:, :3, 3]  # centre in camera
    uv0 = (c_cam0 @ Kt.T)
    uv0 = uv0[:, :2] / uv0[:, 2:3]
    if rot_axis is not None:  # rotation restricted to angles about given (camera-frame) axes, one per group
        ax_t = torch.tensor(np.asarray(rot_axis)[first], dtype=torch.float32, device=dev)
        th = torch.zeros(G, 1, device=dev, requires_grad=True)
    w = torch.zeros(G, 3, device=dev, requires_grad=rot_axis is None)
    duv = torch.zeros(G, 2, device=dev, requires_grad=True)
    dlz = torch.zeros(G, 1, device=dev, requires_grad=True)
    sh = torch.tensor(share, device=dev)
    if prior is not None:
        Pr = torch.tensor(np.asarray(prior), dtype=torch.float32, device=dev)
        Pc = Pr[:, :3, :3] @ c + Pr[:, :3, 3]
    uniq = sorted(set(int(i) for i in frame_ids))
    idx = {f: k for k, f in enumerate(uniq)}
    dts = torch.tensor(np.stack([data[f].dt_allowed for f in uniq]), device=dev)
    bidx = torch.tensor([idx[int(f)] for f in frame_ids], device=dev)
    npix = max(1, max(len(data[f].pix) for f in uniq))
    pix = torch.zeros(len(uniq), npix, 2, device=dev)
    pmask = torch.zeros(len(uniq), npix, device=dev)
    for f in uniq:
        n = len(data[f].pix)
        if n:
            pix[idx[f], :n] = torch.tensor(data[f].pix, device=dev)
            pmask[idx[f], :n] = 1
    pix_b, pmask_b = pix[bidx], pmask[bidx]
    z0 = c_cam0[:, 2:3]
    zp = None if z_prior is None else torch.tensor(np.asarray(z_prior), dtype=torch.float32, device=dev)[:, None]
    params = [{'params': [w if rot_axis is None else th], 'lr': 0.0 if fix_rot else lr_rot}, {'params': [duv], 'lr': lr_tr * 100},
              {'params': [dlz], 'lr': lr_tr}]
    opt = torch.optim.Adam(params)

    def pose():
        R = axis_angle_to_R(w if rot_axis is None else th * ax_t) @ R0
        uvc = uv0 + duv
        z = z0 * torch.exp(dlz)
        cc = torch.cat([(uvc[:, 0:1] - Kt[0, 2]) / Kt[0, 0] * z, (uvc[:, 1:2] - Kt[1, 2]) / Kt[1, 1] * z, z], 1)
        t = cc - R @ c
        return R, t, z

    def losses():
        R, t, z = pose()
        R, t, z = R[sh], t[sh], z[sh]
        pc = torch.einsum('bij,nj->bni', R, P) + t[:, None]
        uvp = pc @ Kt.T
        uvp = uvp[..., :2] / uvp[..., 2:3].clamp_min(0.05)
        d_in = sample_dt(dts[bidx], uvp).clamp(max=trunc)
        l_in = huber(d_in, 2.0).mean(1)
        D = torch.cdist(pix_b, uvp)  # (B,npix,N)
        dmin = D.min(2).values.clamp(max=trunc)
        l_cov = (huber(dmin, 2.0) * pmask_b).sum(1) / pmask_b.sum(1).clamp_min(1)
        loss = l_in + l_cov
        if prior is not None:
            cosang = (torch.einsum('bij,bij->b', R, Pr[:, :3, :3]) - 1) / 2
            ang = torch.sqrt((2 * (1 - cosang)).clamp_min(0) + 1e-12) * (180 / np.pi)  # ~ angle in deg (small angles)
            cc = R @ c + t
            loss = loss + w_prior[0] * ang ** 2 + w_prior[1] * ((cc - Pc).norm(dim=-1) * 100) ** 2
        if zp is not None and w_z > 0:
            loss = loss + w_z * ((z - zp) / 0.05).pow(2)[:, 0]
        return loss, l_in, l_cov

    for it in range(iters):
        opt.zero_grad()
        loss, _, _ = losses()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        loss, l_in, l_cov = losses()
        R, t, z = pose()
        T = torch.eye(4, device=dev).repeat(G, 1, 1)
        T[:, :3, :3] = R
        T[:, :3, 3] = t
    return T.cpu().numpy().astype(np.float64), {'loss': loss.cpu().numpy(), 'in': l_in.cpu().numpy(),
                                                'cov': l_cov.cpu().numpy()}
