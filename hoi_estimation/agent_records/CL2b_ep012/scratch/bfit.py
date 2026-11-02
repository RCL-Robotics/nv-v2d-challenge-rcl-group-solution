import sys; sys.path.insert(0, '/work/scratch')
from lib import *

CROP = 320


def so3_exp_b(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    z = torch.zeros_like(k[:, 0])
    Kx = torch.stack([z, -k[:, 2], k[:, 1], k[:, 2], z, -k[:, 0], -k[:, 1], k[:, 0], z], -1).view(-1, 3, 3)
    th = th[..., None]
    return torch.eye(3, device=w.device)[None] + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def crop_K(K, cx, cy, size=CROP):
    """K for a crop whose top-left pixel is (cx-size/2, cy-size/2)"""
    Kc = K.copy()
    Kc[0, 2] -= int(cx) - size // 2
    Kc[1, 2] -= int(cy) - size // 2
    return Kc, int(cx) - size // 2, int(cy) - size // 2


def crop_img(a, x0, y0, size=CROP):
    H, W = a.shape[:2]
    out = np.zeros((size, size) + a.shape[2:], a.dtype)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + size, W), min(y0 + size, H)
    out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
    return out


class BatchFitter:
    def __init__(self, R):
        self.R = R  # renderer with W=H=CROP
        self.vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)

    def proj_b(self, Ks):
        return torch.tensor(np.stack([self.R.proj(k) for k in Ks]), dtype=torch.float32, device=dev)

    def render_soft(self, Pb, T):
        vc = torch.einsum('bij,nj->bni', T, self.vh)
        clip = torch.einsum('bij,bnj->bni', Pb, vc).contiguous()
        rast, _ = dr.rasterize(self.R.ctx, clip, self.R.f, (self.R.H, self.R.W))
        m = (rast[..., 3:] > 0).float()
        m = dr.antialias(m, rast, clip, self.R.f)
        return m[..., 0]

    def compose(self, T0, w, dt, ray=None, fix_depth=False):
        Rm = so3_exp_b(w) @ T0[:, :3, :3]
        if fix_depth:
            dt = dt - (dt * ray).sum(-1, keepdim=True) * ray
        t = T0[:, :3, 3] + dt
        T = torch.zeros_like(T0)
        T[:, :3, :3] = Rm; T[:, :3, 3] = t; T[:, 3, 3] = 1
        return T

    def fit(self, T0, Ks, target, weight, iters=100, lr_r=0.01, lr_t=0.004, fix_depth=False,
            smooth_w=0.0, smooth_t=0.0, static=False):
        """T0 (B,4,4) cam poses (full-image cam); Ks crop intrinsics; target/weight (B,S,S)."""
        T0 = torch.tensor(T0, dtype=torch.float32, device=dev)
        B = T0.shape[0]
        Pb = self.proj_b(Ks)
        tgt = torch.as_tensor(target, dtype=torch.float32, device=dev)
        wt = torch.as_tensor(weight, dtype=torch.float32, device=dev)
        ray = T0[:, :3, 3] / torch.linalg.norm(T0[:, :3, 3], dim=-1, keepdim=True)
        n = 1 if static else B
        w = torch.zeros(n, 3, device=dev, requires_grad=True)
        dt = torch.zeros(n, 3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
        for it in range(iters):
            T = self.compose(T0, w.expand(B, 3), dt.expand(B, 3), ray, fix_depth)
            m = self.render_soft(Pb, T)
            per = ((m - tgt) ** 2 * wt).sum((1, 2)) / wt.sum((1, 2)).clamp_min(1)
            loss = per.sum()
            if smooth_w > 0 and B > 2 and not static:
                # second differences of the absolute pose params (rotation via offsets is approx)
                Rm = T[:, :3, :3]
                dR = (Rm[2:] - 2 * Rm[1:-1] + Rm[:-2]).pow(2).sum((1, 2))
                d2t = (T[2:, :3, 3] - 2 * T[1:-1, :3, 3] + T[:-2, :3, 3]).pow(2).sum(-1)
                loss = loss + smooth_w * dR.sum() + smooth_t * d2t.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            T = self.compose(T0, w.expand(B, 3), dt.expand(B, 3), ray, fix_depth)
            m = self.render_soft(Pb, T)
            per = ((m - tgt) ** 2 * wt).sum((1, 2)) / wt.sum((1, 2)).clamp_min(1)
        return T.cpu().numpy().astype(np.float64), per.cpu().numpy(), m.detach().cpu().numpy()
