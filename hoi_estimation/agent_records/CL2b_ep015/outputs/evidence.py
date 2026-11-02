"""Per-frame image evidence: crops, masks, and a batched render-and-compare scorer."""
import numpy as np, torch, cv2, h5py, torch.nn.functional as F
from common import *

def gblur(x, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-(torch.arange(k, device=x.device, dtype=x.dtype) - k // 2) ** 2 / (2 * s * s)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)

class Evidence:
    """Holds for every frame a square crop around the object (SAM mask bbox, or the baseline render if the mask is
    empty), resampled to S x S: image, object mask, valid mask (= not covered by the person unless object)."""
    def __init__(self, inputs, frames, renderer, K, Tc_ref, S=320, margin=1.3):
        """frames: iterable over the video frames (BGR, in order); Tc_ref: (F,4,4) reference camera poses."""
        self.S = S; self.K = K; self.r = renderer
        f = h5py.File(f"{inputs}/masks.h5", "r")
        F_ = len(Tc_ref)
        self.crops = np.zeros((F_, 4)); self.img = np.zeros((F_, S, S, 3), np.float32)
        self.obj = np.zeros((F_, S, S), np.float32); self.valid = np.zeros((F_, S, S), np.float32)
        self.area = np.zeros(F_); self.occ = np.zeros(F_)
        for i, fr in enumerate(frames):
            if i >= F_: break
            om = f["object"][i]; pm = f["person"][i]
            mk = renderer.render(Tc_ref[i][None], want_tex=False)[0][0].cpu().numpy()
            self.area[i] = om.sum()
            box = om if om.sum() > 200 else (om | mk)
            ys, xs = np.nonzero(box)
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            s = max(xs.max() - xs.min(), ys.max() - ys.min()) * margin + 24
            self.crops[i] = (cx - s / 2, cy - s / 2, s, s)
            M = np.array([[S / s, 0, -(cx - s / 2) * S / s], [0, S / s, -(cy - s / 2) * S / s]])
            fl = cv2.INTER_AREA if s > S else cv2.INTER_LINEAR
            o = cv2.warpAffine(om.astype(np.float32), M, (S, S), flags=fl)
            p = cv2.warpAffine(pm.astype(np.float32), M, (S, S), flags=fl)
            self.img[i] = cv2.warpAffine(fr[:, :, ::-1].astype(np.float32) / 255, M, (S, S), flags=fl)
            self.obj[i] = o; self.valid[i] = ((p < 0.5) | (o > 0.5)).astype(np.float32)
            self.occ[i] = (mk & pm & ~om).sum() / max(mk.sum(), 1)   # fraction of the reference render hidden by the person
        self.P = np.array([proj_from_K(crop_K(K, c, S), S, S) for c in self.crops])

    def batch(self, ids, dev="cuda"):
        t = lambda a: torch.tensor(a[ids], dtype=torch.float32, device=dev)
        return t(self.P), t(self.obj), t(self.valid), t(self.img).permute(0, 3, 1, 2)

def photo_loss(rgb, sil, img, vd, s=1.5):
    """Locally normalised colour difference inside the rendered & visible region."""
    x = rgb.permute(0, 3, 1, 2); w = ((sil > 0.5).float() * vd)[:, None]
    def norm(a):
        a = gblur(a, s)
        n = w.sum((2, 3), keepdim=True).clamp(min=1)
        mu = (a * w).sum((2, 3), keepdim=True) / n
        sd = torch.sqrt((((a - mu) ** 2) * w).sum((2, 3), keepdim=True) / n + 1e-4)
        return (a - mu) / sd
    return ((norm(x) - norm(img)) ** 2 * w).sum((1, 2, 3)) / w.sum((1, 2, 3)).clamp(min=1) / 3

def score_and_refine(r, ev, ids, Tc, iters=0, lam=0.3, fixz=True, lr=(3e-3, 2e-3), shared=False, w_prior=None):
    """Render-and-compare for frames `ids` (indices into ev) at camera poses Tc (B,4,4).
    Optionally refine (rotation about the object centre + translation, z fixed) for `iters` Adam steps.
    Returns refined poses, IoU (visible part), silhouette loss, photometric loss."""
    P, tg, vd, img = ev.batch(ids)
    T0 = torch.tensor(Tc, dtype=torch.float32, device="cuda")
    B = len(ids); nb = 1 if shared else B
    w = torch.zeros(nb, 3, device="cuda", requires_grad=True); t = torch.zeros(nb, 3, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr[0]}, {"params": [t], "lr": lr[1]}])
    c0 = T0[:, :3, 3].clone(); msk = torch.tensor([1., 1., 0. if fixz else 1.], device="cuda")
    S = ev.S
    for it in range(iters + 1):
        Rd = rotvec_to_mat(w).expand(B, 3, 3)
        T = torch.cat([torch.cat([Rd @ T0[:, :3, :3], (c0 + t * msk)[..., None]], 2), T0[:, 3:]], 1)
        with torch.set_grad_enabled(it < iters):
            sil, rgb, _ = render_batch(r, T, P, S, with_tex=lam > 0)
            ls = (((sil - tg) ** 2) * vd).sum((1, 2)) / tg.sum((1, 2)).clamp(min=50)
            lp = photo_loss(rgb, sil, img, vd) if lam > 0 else torch.zeros_like(ls)
            l = ls + lam * lp
            if it < iters:
                opt.zero_grad(); l.sum().backward(); opt.step()
    with torch.no_grad():
        inter = ((sil > 0.5) & (tg > 0.5) & (vd > 0)).sum((1, 2)).float()
        uni = (((sil > 0.5) | (tg > 0.5)) & (vd > 0)).sum((1, 2)).float()
    return (T.detach().cpu().numpy().astype(np.float64), (inter / uni.clamp(min=1)).cpu().numpy(),
            ls.detach().cpu().numpy(), lp.detach().cpu().numpy())

def refine_yaw(r, ev, ids, Tc, iters=100, lam=0.3, tilt_sigma_deg=2.0, tilt_w=0.002, lr=(4e-3, 2e-3)):
    """Refine poses with rotation parametrised in the MESH frame, R = R0 exp([a, b, yaw]): yaw about the mesh's
    vertical axis is free, tilts a, b carry a Gaussian prior (sigma tilt_sigma_deg) towards the input pose;
    image-plane translation free, camera depth fixed. Data term: silhouette + lam * photometric."""
    P, tg, vd, img = ev.batch(ids)
    T0 = torch.tensor(Tc, dtype=torch.float32, device="cuda")
    B = len(ids)
    w = torch.zeros(B, 3, device="cuda", requires_grad=True); t = torch.zeros(B, 2, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr[0]}, {"params": [t], "lr": lr[1]}])
    S = ev.S; sig = np.radians(tilt_sigma_deg)
    for it in range(iters + 1):
        Rd = rotvec_to_mat(w)
        tt = torch.cat([T0[:, :2, 3] + t, T0[:, 2:3, 3]], 1)
        T = torch.cat([torch.cat([T0[:, :3, :3] @ Rd, tt[..., None]], 2), T0[:, 3:]], 1)
        with torch.set_grad_enabled(it < iters):
            sil, rgb, _ = render_batch(r, T, P, S, with_tex=lam > 0)
            ls = (((sil - tg) ** 2) * vd).sum((1, 2)) / tg.sum((1, 2)).clamp(min=50)
            lp = photo_loss(rgb, sil, img, vd) if lam > 0 else torch.zeros_like(ls)
            prior = tilt_w * (w[:, :2] ** 2).sum(1) / sig ** 2
            l = ls + lam * lp + prior
            if it < iters:
                opt.zero_grad(); l.sum().backward(); opt.step()
    with torch.no_grad():
        inter = ((sil > 0.5) & (tg > 0.5) & (vd > 0)).sum((1, 2)).float()
        uni = (((sil > 0.5) | (tg > 0.5)) & (vd > 0)).sum((1, 2)).float()
    return (T.detach().cpu().numpy().astype(np.float64), (inter / uni.clamp(min=1)).cpu().numpy(),
            ls.detach().cpu().numpy(), lp.detach().cpu().numpy(), w.detach().cpu().numpy())
