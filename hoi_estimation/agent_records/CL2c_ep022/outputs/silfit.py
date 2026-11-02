"""Silhouette render-and-compare fitting of object poses (camera frame), batched over frames."""
import numpy as np
import torch
import torch.nn.functional as Fn
from render_utils import crop_K, proj_batch, render_crops

S = 192  # crop render resolution


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1], Kx[..., 0, 2] = -k[..., 2], k[..., 1]
    Kx[..., 1, 0], Kx[..., 1, 2] = k[..., 2], -k[..., 0]
    Kx[..., 2, 0], Kx[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def gblur(x, sigma):
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    g = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    x = x[:, None]
    x = Fn.conv2d(Fn.pad(x, (r, r, 0, 0), mode='replicate'), g.view(1, 1, 1, -1))
    x = Fn.conv2d(Fn.pad(x, (0, 0, r, r), mode='replicate'), g.view(1, 1, -1, 1))
    return x[:, 0]


def make_boxes(obj_masks, T_init, K, R, pad=1.35, min_size=96):
    """Square crop boxes covering the mask and the initial render."""
    boxes = []
    for i in range(len(obj_masks)):
        ys, xs = np.nonzero(obj_masks[i])
        c = R.v.mean(0).cpu().numpy()
        # projected mesh bbox at init
        V = R.v[::50].cpu().numpy() @ T_init[i][:3, :3].T + T_init[i][:3, 3]
        u = V[:, 0] / V[:, 2] * K[0, 0] + K[0, 2]
        v = V[:, 1] / V[:, 2] * K[1, 1] + K[1, 2]
        if len(xs):
            x0, x1 = min(xs.min(), u.min()), max(xs.max(), u.max())
            y0, y1 = min(ys.min(), v.min()), max(ys.max(), v.max())
        else:
            x0, x1, y0, y1 = u.min(), u.max(), v.min(), v.max()
        s = max(x1 - x0, y1 - y0, min_size) * pad
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        boxes.append((cx - s / 2, cy - s / 2, s))
    return boxes


def crop_masks(masks, boxes, Sz=S):
    """masks: (B,H,W) bool numpy -> (B,Sz,Sz) float torch via bilinear sampling."""
    import cv2
    out = np.zeros((len(masks), Sz, Sz), np.float32)
    for i, (m, (x0, y0, s)) in enumerate(zip(masks, boxes)):
        M = np.array([[Sz / s, 0, -x0 * Sz / s], [0, Sz / s, -y0 * Sz / s]], np.float32)
        out[i] = cv2.warpAffine(m.astype(np.float32), M, (Sz, Sz), flags=cv2.INTER_AREA if s > Sz else cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


class SilFitter:
    def __init__(self, R, K, fat_px=2.5):
        self.R, self.K = R, np.asarray(K, np.float64)
        self.fat_px = fat_px  # SAM masks are ~2-3 px wider than the true silhouette
        self.c = R.v.mean(0).double()  # rotation centre (mesh coords)

    def setup(self, obj, per, T_init, boxes=None, imgs=None):
        """obj/per: (B,H,W) bool masks; T_init (B,4,4) cam<-mesh numpy.
        imgs: optional list of BGR frames -> enables the edge-alignment term."""
        dev = self.R.dev
        self.B = len(obj)
        self.boxes = boxes if boxes is not None else make_boxes(obj, T_init, self.K, self.R)
        Ks = [crop_K(self.K, b, S) for b in self.boxes]
        self.P = proj_batch(Ks, S, dev)
        tgt = crop_masks(obj, self.boxes)
        occ = crop_masks(per & ~obj, self.boxes)
        self.tgt = torch.tensor(tgt, device=dev)
        # valid = not covered by the person (object may be hidden there)
        self.valid = torch.tensor((occ < 0.5).astype(np.float32), device=dev)
        self.valid = 1 - (gblur(1 - self.valid, 1.0) > 0.05).float()
        self.T0 = torch.tensor(np.asarray(T_init), dtype=torch.float64, device=dev)
        self.dt = None
        if imgs is not None:
            import cv2
            dts = np.zeros((self.B, S, S), np.float32)
            for i, (img, (x0, y0, s)) in enumerate(zip(imgs, self.boxes)):
                g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
                e = cv2.Canny(g, 30, 80)
                dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)
                dt = np.minimum(dt, 8.0)  # truncated distance to nearest image edge (full-res px)
                M = np.array([[S / s, 0, -x0 * S / s], [0, S / s, -y0 * S / s]], np.float32)
                dts[i] = cv2.warpAffine(dt, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=8.0)
            self.dt = torch.tensor(dts, device=dev)

    def poses(self, w, d, idx=None):
        T0 = self.T0 if idx is None else self.T0[idx]
        R0, t0 = T0[:, :3, :3], T0[:, :3, 3]
        Rn = so3_exp(w) @ R0
        c_cam = (R0 @ self.c) + t0 + d
        t = c_cam - Rn @ self.c
        T = torch.zeros_like(T0)
        T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = Rn, t, 1
        return T

    def edge_term(self, T, idx):
        """Mean image-edge distance (px) along the rendered (non-inflated) occluding contour."""
        sil, _, _ = render_crops(self.R, T.float(), self.P[idx], S)
        gx = sil[:, :, 1:] - sil[:, :, :-1]
        gy = sil[:, 1:, :] - sil[:, :-1, :]
        g = Fn.pad(gx.abs(), (0, 1, 0, 0)) + Fn.pad(gy.abs(), (0, 0, 0, 1))
        g = g * self.valid[idx]
        return (g * self.dt[idx]).sum((1, 2)) / g.sum((1, 2)).clamp_min(1e-3)

    def loss_terms(self, T, idx, sigma):
        infl = None
        if self.fat_px > 0:
            z = (T[:, :3, :3] @ self.c + T[:, :3, 3])[:, 2]
            infl = self.fat_px * z.detach() / self.K[0, 0]
        sil, _, _ = render_crops(self.R, T.float(), self.P[idx], S, inflate=infl)
        tgt, val = self.tgt[idx], self.valid[idx]
        a, b = gblur(sil, sigma), gblur(tgt, sigma)
        inter = (a * b * val).sum((1, 2))
        union = ((a + b - a * b) * val).sum((1, 2))
        return 1 - inter / union.clamp_min(1e-6)

    @torch.no_grad()
    def score(self, T_list, sigma=0.0, chunk=48):
        """IoU-loss of given cam poses (B,4,4) numpy for the set-up frames."""
        T = torch.tensor(np.asarray(T_list), dtype=torch.float64, device=self.R.dev)
        out = []
        for s in range(0, self.B, chunk):
            idx = torch.arange(s, min(self.B, s + chunk), device=self.R.dev)
            out.append(self.loss_terms(T[idx], idx, sigma))
        return torch.cat(out).cpu().numpy()

    def fit(self, iters=(60, 60, 60), sigmas=(4.0, 2.0, 1.0), lr_r=0.01, lr_t=0.005, shared=False,
            fix_depth=False, w_prior_r=0.0, w_prior_t=0.0, w_smooth=0.0, chunk=64, w_edge=0.0):
        """Optimise rotation (about the object centre) and translation.
        shared: one delta for all frames (static phase). fix_depth: translation only perpendicular to
        the viewing ray of the object centre (keeps its camera depth)."""
        dev = self.R.dev
        n = 1 if shared else self.B
        w = torch.zeros(n, 3, dtype=torch.float64, device=dev, requires_grad=True)
        d = torch.zeros(n, 3, dtype=torch.float64, device=dev, requires_grad=True)
        opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [d], 'lr': lr_t}])
        for it_n, sg in zip(iters, sigmas):
            for it in range(it_n):
                opt.zero_grad()
                tot = 0.0
                for s in range(0, self.B, chunk):
                    idx = torch.arange(s, min(self.B, s + chunk), device=dev)
                    ww = w.expand(len(idx), 3) if shared else w[idx]
                    dd = d.expand(len(idx), 3) if shared else d[idx]
                    if fix_depth:
                        T0 = self.T0[idx]
                        c = (T0[:, :3, :3] @ self.c) + T0[:, :3, 3]
                        ray = c / c.norm(dim=1, keepdim=True)
                        # move on the plane z = const (keeps camera z of centre)
                        dd = torch.stack([dd[:, 0], dd[:, 1], torch.zeros_like(dd[:, 0])], 1)
                    T = self.poses(ww, dd, idx)
                    L = self.loss_terms(T, idx, sg).sum() / self.B
                    if w_edge > 0 and self.dt is not None:
                        L = L + w_edge * self.edge_term(T, idx).sum() / self.B
                    if not shared:
                        L = L + (w_prior_r * (ww ** 2).sum() + w_prior_t * (dd ** 2).sum()) / self.B
                    L.backward()
                    tot += float(L)
                if w_smooth > 0 and not shared:
                    Ls = w_smooth * (((w[2:] - 2 * w[1:-1] + w[:-2]) ** 2).sum() * 1.0 +
                                     ((d[2:] - 2 * d[1:-1] + d[:-2]) ** 2).sum() * 100.0)
                    Ls.backward()
                opt.step()
        with torch.no_grad():
            ww = w.expand(self.B, 3) if shared else w
            dd = d.expand(self.B, 3) if shared else d
            if fix_depth:
                dd = torch.stack([dd[:, 0], dd[:, 1], torch.zeros_like(dd[:, 0])], 1)
            return self.poses(ww, dd).cpu().numpy()
