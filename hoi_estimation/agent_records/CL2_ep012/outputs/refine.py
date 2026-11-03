"""Differentiable render-and-compare refinement of an object pose on a cropped, upsampled window."""
import numpy as np, torch, cv2
import nvdiffrast.torch as dr
import torch.nn.functional as F

def rotvec_to_mat(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)

class CropData:
    """Image crop around a projected point, at native resolution C, with K for an S-times upsampled render."""
    def __init__(self, img_gray, mobj, mperson, K, uv, C=320, S=2):
        x0 = int(round(uv[0] - C / 2)); y0 = int(round(uv[1] - C / 2))
        self.x0, self.y0, self.C, self.S = x0, y0, C, S
        def crop(a, fill=0):
            out = np.full((C, C), fill, a.dtype)
            ys = slice(max(0, y0), min(a.shape[0], y0 + C)); xs = slice(max(0, x0), min(a.shape[1], x0 + C))
            if ys.stop > ys.start and xs.stop > xs.start:
                out[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = a[ys, xs]
            return out
        inside = crop(np.ones_like(mobj, bool), False)
        self.g = crop(img_gray.astype(np.float32)); self.mo = crop(mobj); self.mp = crop(mperson) | ~inside
        Kc = K.copy().astype(float)
        Kc[0, 0] *= S; Kc[1, 1] *= S
        Kc[0, 2] = (K[0, 2] - x0 + 0.5) * S - 0.5; Kc[1, 2] = (K[1, 2] - y0 + 0.5) * S - 0.5
        self.Kc = Kc

def ncc_t(a, b, w):
    s = w.sum(dim=(-1, -2)).clamp_min(1.0)
    am = (a * w).sum(dim=(-1, -2)) / s; bm = (b * w).sum(dim=(-1, -2)) / s
    a = a - am[..., None, None]; b = b - bm[..., None, None]
    return (a * b * w).sum(dim=(-1, -2)) / torch.sqrt((a * a * w).sum(dim=(-1, -2)) * (b * b * w).sum(dim=(-1, -2)) + 1e-6)

def render_soft(r, T, Kc):
    """T (B,4,4) torch, differentiable. returns soft mask (B,H,W), gray texture (B,H,W), depth."""
    B = T.shape[0]
    vh = torch.cat([r.v, torch.ones_like(r.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    P = r.proj(Kc)
    clip = torch.einsum("ij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(r.glctx, clip, r.f, (r.H, r.W))
    uvi, _ = dr.interpolate(r.uv[None].contiguous(), rast, r.f)
    col = dr.texture(r.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
    m = (rast[..., 3:4] > 0).float()
    col = col * m
    both = torch.cat([m, col.mean(-1, keepdim=True)], -1)
    both = dr.antialias(both.contiguous(), rast, clip, r.f)
    return both[..., 0], both[..., 1]

def pool(x, S):
    B, H, W = x.shape
    return x.reshape(B, H // S, S, W // S, S).mean((2, 4))

def refine(r, T0, crop, iters=60, lr_r=0.01, lr_t=0.004, w_tex=1.0, w_sil=1.0, fix_depth=True, dev="cuda"):
    """T0 (B,4,4) numpy camera-frame poses (hypotheses for the same crop). Returns refined poses and scores."""
    B = T0.shape[0]
    T0t = torch.tensor(T0, dtype=torch.float32, device=dev)
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    t = torch.zeros(B, 3, device=dev, requires_grad=True)
    g = torch.tensor(crop.g, device=dev)[None]; g = (g - g.mean()) / (g.std() + 1e-6)
    mo = torch.tensor(crop.mo, device=dev, dtype=torch.float32)[None]
    valid = torch.tensor(~crop.mp, device=dev, dtype=torch.float32)[None]
    opt = torch.optim.Adam([{"params": [w], "lr": lr_r}, {"params": [t], "lr": lr_t}])
    S = crop.S
    def pose():
        Rm = rotvec_to_mat(w)
        T = T0t.clone()
        # rotate about the object's current centre (translation of T0), in camera frame
        T[:, :3, :3] = Rm @ T0t[:, :3, :3]
        tt = t.clone()
        if fix_depth:  # move only perpendicular to the viewing ray: scale x,y offsets keep z
            tt = torch.stack([t[:, 0], t[:, 1], torch.zeros_like(t[:, 2])], 1)
        T[:, :3, 3] = T0t[:, :3, 3] + tt
        return T
    for it in range(iters + 1):
        T = pose()
        ms, gr = render_soft(r, T, crop.Kc)
        ms = pool(ms, S); gr = pool(gr, S)
        sil = (((ms - mo) ** 2) * valid).sum((-1, -2)) / ((torch.maximum(ms, mo) * valid).sum((-1, -2)) + 1)
        wt = (ms > 0.98).float() * valid
        tex = ncc_t(gr, g, wt)
        loss = w_sil * sil - w_tex * tex
        if it == iters: break
        opt.zero_grad(); loss.sum().backward(); opt.step()
    with torch.no_grad():
        hard = (ms > 0.5).float()
        iou = ((hard * mo * valid).sum((-1, -2)) / (((hard + mo) > 0).float() * valid).sum((-1, -2)).clamp_min(1)).cpu().numpy()
    return T.detach().cpu().numpy().astype(float), dict(iou=iou, tex=tex.detach().cpu().numpy(), sil=sil.detach().cpu().numpy())

def render_soft_multi(r, T, Ks):
    """Like render_soft but each batch element has its own intrinsics (list of 3x3). Returns soft masks."""
    vh = torch.cat([r.v, torch.ones_like(r.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    P = torch.stack([r.proj(K) for K in Ks], 0)
    clip = torch.einsum("bij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(r.glctx, clip, r.f, (r.H, r.W))
    m = (rast[..., 3:4] > 0).float()
    m = dr.antialias(m.contiguous(), rast, clip, r.f)
    return m[..., 0]
