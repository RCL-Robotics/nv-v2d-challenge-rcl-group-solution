import sys; sys.path.insert(0, '/work/scratch')
from lib import *


def so3_exp(w):
    th = torch.linalg.norm(w) + 1e-9
    k = w / th
    Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def render_soft(R, K, T):
    """T: (4,4) torch camera-from-mesh. antialiased silhouette"""
    P = torch.tensor(R.proj(K), dtype=torch.float32, device=dev)
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = vh @ T.T
    clip = (vc @ P.T)[None].contiguous()
    rast, _ = dr.rasterize(R.ctx, clip, R.f, (R.H, R.W))
    m = (rast[..., 3:] > 0).float()
    m = dr.antialias(m, rast, clip, R.f)
    return m[0, ..., 0]


def fit_pose(R, K, T0, target, weight, iters=150, lr_r=0.01, lr_t=0.005, fix_depth=False):
    T0t = torch.tensor(T0, dtype=torch.float32, device=dev)
    w = torch.zeros(3, device=dev, requires_grad=True)
    dt = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
    tgt = torch.as_tensor(target, dtype=torch.float32, device=dev)
    wt = torch.as_tensor(weight, dtype=torch.float32, device=dev)
    for it in range(iters):
        T = torch.eye(4, device=dev)
        T[:3, :3] = so3_exp(w) @ T0t[:3, :3]
        d = dt
        if fix_depth:  # move only perpendicular to the viewing ray through the object
            ray = T0t[:3, 3] / torch.linalg.norm(T0t[:3, 3])
            d = dt - (dt @ ray) * ray
        T[:3, 3] = T0t[:3, 3] + d
        m = render_soft(R, K, T)
        loss = ((m - tgt) ** 2 * wt).sum() / wt.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return T.detach().cpu().numpy().astype(np.float64), loss.item()


def dark_prob(img_bgr, thr=38., s=5.):
    g = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return 1 / (1 + np.exp(-(thr - g) / s))
