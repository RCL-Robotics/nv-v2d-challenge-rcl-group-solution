"""Differentiable silhouette render-and-compare fitting (nvdiffrast)."""
import numpy as np
import torch
import nvdiffrast.torch as dr
from common import proj_matrix


def so3_exp(w):
    """w: (B,3) -> (B,3,3), Rodrigues, differentiable."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1] = -k[:, 2]; Kx[:, 0, 2] = k[:, 1]
    Kx[:, 1, 0] = k[:, 2]; Kx[:, 1, 2] = -k[:, 0]
    Kx[:, 2, 0] = -k[:, 1]; Kx[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


def gblur(x, sigma):
    """separable gaussian blur of (B,h,w)"""
    rad = int(3 * sigma + 0.5)
    t = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2); k = k / k.sum()
    y = torch.nn.functional.conv2d(x[:, None], k.view(1, 1, 1, -1), padding=(0, rad))
    y = torch.nn.functional.conv2d(y, k.view(1, 1, -1, 1), padding=(rad, 0))
    return y[:, 0]


class SilFitter:
    def __init__(self, mesh, K, w, h, device='cuda'):
        self.glctx = dr.RasterizeCudaContext()
        self.dev = device
        self.w, self.h = w, h
        self.P = torch.tensor(proj_matrix(np.asarray(K), w, h), device=device)
        v = np.asarray(mesh.vertices, np.float64)
        self.center = (v.min(0) + v.max(0)) / 2
        self.v = torch.tensor(v - self.center, dtype=torch.float32, device=device)  # centred
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)

    def render_soft(self, R, p):
        """R (B,3,3), p (B,3) centre position in camera. returns antialiased mask (B,h,w)."""
        vc = torch.einsum('bij,nj->bni', R, self.v) + p[:, None, :]
        vh = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
        clip = torch.einsum('ij,bnj->bni', self.P, vh).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (self.h, self.w))
        m = (rast[..., 3:4] > 0).float()
        m = dr.antialias(m, rast, clip, self.f)
        return m[..., 0]

    def split(self, Tc):
        """Tc (N,4,4) mesh->cam. returns R (N,3,3), p (N,3) centre positions."""
        R = Tc[:, :3, :3]
        p = Tc[:, :3, 3] + R @ self.center
        return R, p

    def join(self, R, p):
        T = np.tile(np.eye(4), (len(R), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = p - R @ self.center
        return T

    @staticmethod
    def soft_iou_loss(r, M, V):
        inter = (r * M * V).sum((1, 2))
        uni = ((r + M - r * M) * V).sum((1, 2))
        return 1 - inter / uni.clamp(min=1)

    def fit(self, R0g, p0g, group, obj, valid, iters=150, lr_r=0.005, lr_t=0.005,
            w_smooth_r=0.0, w_smooth_t=0.0, chunk=24, weights=None, seed=0,
            w_prior_r=0.0, w_prior_uv=0.0, w_prior_z=0.0, lr_z=None, sigma=(0.0, 0.0),
            tilt_axis=None, tilt_target=None, w_tilt=None):
        """Group-parametrised fit. R0g (G,3,3), p0g (G,3) initial rotation / centre position per
        parameter group; group (N,) maps each frame to its group (static phases share one group).
        Data term: soft IoU between antialiased render and the object mask over valid pixels
        (pixels covered by the person but not by the object mask are ignored).
        Smoothness: squared second differences over consecutive frames of rotation matrices and centres.
        Translation is parametrised as (du, dv, dlog z): image-plane offset of the centre ray and log depth,
        so that the poorly observed depth direction gets its own learning rate (lr_z) and prior.
        Tilt prior (per group weight w_tilt[g]): w_tilt*|R tilt_axis - tilt_target|^2 (mesh axis -> camera dir).
        Priors (per group): w_prior_r*|w|^2 (rad), w_prior_uv*|duv|^2, w_prior_z*dlogz^2 toward the init.
        Returns per-group R (G,3,3), p (G,3), per-frame final data loss (N,)."""
        torch.manual_seed(seed)
        dev = self.dev
        group = torch.tensor(np.asarray(group), dtype=torch.long, device=dev)
        N = len(group); G = len(R0g)
        R0t = torch.tensor(np.asarray(R0g), dtype=torch.float32, device=dev)
        p0t = torch.tensor(np.asarray(p0g), dtype=torch.float32, device=dev)
        wr = torch.zeros(G, 3, device=dev, requires_grad=True)
        duv = torch.zeros(G, 2, device=dev, requires_grad=True)
        dlz = torch.zeros(G, 1, device=dev, requires_grad=True)
        lr_z = lr_t if lr_z is None else lr_z
        opt = torch.optim.Adam([{'params': [wr], 'lr': lr_r}, {'params': [duv], 'lr': lr_t},
                                {'params': [dlz], 'lr': lr_z}])
        uv0 = p0t[:, :2] / p0t[:, 2:3]; z0 = p0t[:, 2:3]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * min(k, iters) / iters)))
        Mt = torch.tensor(np.asarray(obj), dtype=torch.float32)
        Vt_ = torch.tensor(np.asarray(valid), dtype=torch.float32)
        fw = torch.ones(N, device=dev) if weights is None else torch.tensor(np.asarray(weights), dtype=torch.float32, device=dev)
        fws = fw.sum().clamp(min=1e-6)

        if w_tilt is not None:
            ta = torch.tensor(np.asarray(tilt_axis), dtype=torch.float32, device=dev)
            tt = torch.tensor(np.asarray(tilt_target), dtype=torch.float32, device=dev)
            wt_t = torch.tensor(np.asarray(w_tilt), dtype=torch.float32, device=dev)

        def cur():
            Rg = so3_exp(wr) @ R0t
            z = z0 * torch.exp(dlz)
            pg = torch.cat([(uv0 + duv) * z, z], 1)
            return Rg, pg
        losses = torch.zeros(N)
        for it in range(iters):
            opt.zero_grad()
            Rg, pg = cur()
            R = Rg[group]; p = pg[group]
            reg = w_prior_r * (wr ** 2).sum() + w_prior_uv * (duv ** 2).sum() + w_prior_z * (dlz ** 2).sum()
            if N >= 3 and (w_smooth_r > 0 or w_smooth_t > 0):
                reg = reg + w_smooth_r * ((R[2:] - 2 * R[1:-1] + R[:-2]) ** 2).sum() \
                    + w_smooth_t * ((p[2:] - 2 * p[1:-1] + p[:-2]) ** 2).sum()
            if w_tilt is not None:
                reg = reg + (wt_t * ((Rg @ ta - tt) ** 2).sum(-1)).sum()
            if torch.is_tensor(reg) and reg.requires_grad:
                reg.backward()
            for s in range(0, N, chunk):
                e = min(N, s + chunk)
                if fw[s:e].sum() == 0 and it < iters - 1:
                    continue
                Rg, pg = cur()
                r = self.render_soft(Rg[group[s:e]], pg[group[s:e]])
                M = Mt[s:e].to(dev, non_blocking=True)
                sg = sigma[0] + (sigma[1] - sigma[0]) * it / max(1, iters - 1)
                if sg > 0.3:
                    r = gblur(r, sg); M = gblur(M, sg)
                l = self.soft_iou_loss(r, M, Vt_[s:e].to(dev, non_blocking=True))
                (l * fw[s:e]).sum().div(fws).mul(N).backward()
                losses[s:e] = l.detach().cpu()
            opt.step(); sched.step()
        with torch.no_grad():
            Rg, pg = cur()
        return Rg.cpu().double().numpy(), pg.cpu().double().numpy(), losses.numpy()

    def iou(self, R, p, obj, valid, chunk=32):
        """hard IoU per frame for poses R (N,3,3), p (N,3)."""
        out = []
        with torch.no_grad():
            for s in range(0, len(R), chunk):
                r = self.render_soft(torch.tensor(R[s:s + chunk], dtype=torch.float32, device=self.dev),
                                     torch.tensor(p[s:s + chunk], dtype=torch.float32, device=self.dev)) > 0.5
                M = torch.tensor(np.asarray(obj[s:s + chunk]), device=self.dev)
                V = torch.tensor(np.asarray(valid[s:s + chunk]), device=self.dev)
                inter = (r & M & V).sum((1, 2)).float(); uni = ((r | M) & V).sum((1, 2)).float()
                out.append((inter / uni.clamp(min=1)).cpu().numpy())
        return np.concatenate(out)
