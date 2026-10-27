"""Batched crop-based render-and-compare for 6-DoF object pose (nvdiffrast)."""
import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import distance_transform_edt


def so3_exp(w):
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1], Kx[..., 0, 2] = -k[..., 2], k[..., 1]
    Kx[..., 1, 0], Kx[..., 1, 2] = k[..., 2], -k[..., 0]
    Kx[..., 2, 0], Kx[..., 2, 1] = -k[..., 1], k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


class CropFitter:
    def __init__(self, mesh_path, res=192, device="cuda", tex_res=512, max_faces=None):
        m = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
        self.mesh = m
        self.dev = device
        self.R = res
        V = np.asarray(m.vertices, np.float64)
        self.center = (V.min(0) + V.max(0)) / 2  # bbox centre in mesh coords
        self.v = torch.tensor(V - self.center, dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=device)
        uv = np.asarray(m.visual.uv, np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255.0
        tex = cv2.resize(tex, (tex_res, tex_res), interpolation=cv2.INTER_AREA)
        self.tex = torch.tensor(tex[:, :, ::-1].copy(), device=device)[None].contiguous()  # BGR
        self.glctx = dr.RasterizeCudaContext(device=device)

    # pose <-> (Rc, p): X_cam = Rc (x - center) + p
    def to_rp(self, T):
        T = np.asarray(T, np.float64)
        Rm = T[:, :3, :3]
        p = T[:, :3, 3] + np.einsum("bij,j->bi", Rm, self.center)
        return Rm, p

    def to_T(self, Rm, p):
        T = np.tile(np.eye(4), (len(p), 1, 1))
        T[:, :3, :3] = Rm
        T[:, :3, 3] = p - np.einsum("bij,j->bi", Rm, self.center)
        return T

    def render(self, Rm, p, Kc, need_color=True):
        """Rm (B,3,3) p (B,3) Kc (B,3,3) torch. returns soft sil (B,R,R), color (B,R,R,3)."""
        R = self.R
        vc = torch.einsum("bij,nj->bni", Rm, self.v) + p[:, None]
        fx, fy, cx, cy = Kc[:, 0, 0], Kc[:, 1, 1], Kc[:, 0, 2], Kc[:, 1, 2]
        near, far = 0.05, 20.0
        z = vc[..., 2]
        xc = (2 * fx[:, None] * vc[..., 0] + (2 * (cx[:, None] + 0.5) - R) * z) / R
        yc = (2 * fy[:, None] * vc[..., 1] + (2 * (cy[:, None] + 0.5) - R) * z) / R
        zc = (far + near) / (far - near) * z - 2 * far * near / (far - near)
        clip = torch.stack([xc, yc, zc, z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (R, R))
        mask = (rast[..., 3:4] > 0).float()
        if need_color:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            col = dr.texture(self.tex, uvi, filter_mode="linear") * mask
            feat = torch.cat([mask, col], -1)
        else:
            feat = mask
        feat = dr.antialias(feat.contiguous(), rast, clip, self.f)
        out = {"sil": feat[..., 0], "rast": rast}
        if need_color:
            out["color"] = feat[..., 1:]
            out["hard"] = mask[..., 0]
        return out


def crop_K(K, box, R):
    x0, y0, s = box
    Kc = K.copy()
    Kc[0, 0] *= R / s
    Kc[1, 1] *= R / s
    Kc[0, 2] = (K[0, 2] - x0) * R / s
    Kc[1, 2] = (K[1, 2] - y0) * R / s
    return Kc


def crop_img(img, box, R, interp=cv2.INTER_AREA):
    x0, y0, s = box
    A = np.array([[R / s, 0, -x0 * R / s], [0, R / s, -y0 * R / s]], np.float64)
    return cv2.warpAffine(img, A, (R, R), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def box_from(mask_bool, extra_pts=None, pad=0.3, min_size=120):
    ys, xs = np.nonzero(mask_bool)
    pts_x = list(xs[[xs.argmin(), xs.argmax()]]) if len(xs) else []
    pts_y = list(ys[[ys.argmin(), ys.argmax()]]) if len(ys) else []
    if extra_pts is not None:
        pts_x += list(extra_pts[:, 0])
        pts_y += list(extra_pts[:, 1])
    x0, x1, y0, y1 = min(pts_x), max(pts_x), min(pts_y), max(pts_y)
    s = max(x1 - x0, y1 - y0, min_size) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - s / 2, cy - s / 2, s)


def project(K, T, pts):
    X = np.einsum("ij,nj->ni", T[:3, :3], pts) + T[:3, 3]
    uv = X[:, :2] / X[:, 2:3]
    return uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


class FrameData:
    """Crops of image / object mask / ignore-mask / distance transform for a batch of frames."""

    def __init__(self, frames, obj, ign, boxes, K, R, dev="cuda"):
        imgs, Ms, Vs, Ds, Ks = [], [], [], [], []
        for im, m, ig, b in zip(frames, obj, ign, boxes):
            ci = crop_img(im, b, R).astype(np.float32) / 255.0
            cm = crop_img(m.astype(np.float32), b, R, cv2.INTER_LINEAR)
            cg = crop_img(ig.astype(np.float32), b, R, cv2.INTER_LINEAR)
            mb = cm > 0.5
            valid = ((cg < 0.5) | mb).astype(np.float32)
            dt = distance_transform_edt(~mb) if mb.any() else np.full(mb.shape, R, np.float64)
            imgs.append(ci)
            Ms.append(cm)
            Vs.append(valid)
            Ds.append(np.minimum(dt, 40.0) / R)
            Ks.append(crop_K(K, b, R))
        t = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=dev)
        self.img, self.M, self.V, self.D, self.K = t(imgs), t(Ms), t(Vs), t(Ds), t(Ks)
        self.boxes = boxes

    def sub(self, idx):
        o = FrameData.__new__(FrameData)
        for k in ("img", "M", "V", "D", "K"):
            setattr(o, k, getattr(self, k)[idx])
        o.boxes = [self.boxes[i] for i in idx]
        return o


def color_residual(col, img, w):
    """per-frame affine colour normalisation (gain+bias per channel), robust L1 residual in mask w."""
    ws = w.sum((1, 2)).clamp_min(1.0)
    mc = (col * w[..., None]).sum((1, 2)) / ws[:, None]
    mi = (img * w[..., None]).sum((1, 2)) / ws[:, None]
    dc = (col - mc[:, None, None]) * w[..., None]
    di = (img - mi[:, None, None]) * w[..., None]
    sc = (dc * dc).sum((1, 2)) / ws[:, None]
    si = (di * di).sum((1, 2)) / ws[:, None]
    zc = dc / sc.sqrt().clamp_min(1e-3)[:, None, None]
    zi = di / si.sqrt().clamp_min(1e-3)[:, None, None]
    ncc = (zc * zi).sum((1, 2, 3)) / (3 * ws)
    return ncc


def losses(fit, Rm, p, fd, w_col=0.0):
    out = fit.render(Rm, p, fd.K, need_color=w_col > 0)
    S = out["sil"]
    V, M = fd.V, fd.M
    inter = (S * M * V).sum((1, 2))
    uni = ((S + M - S * M) * V).sum((1, 2)).clamp_min(1.0)
    iou = inter / uni
    # chamfer-like pull: rendered pixels (valid) far from the mask
    dtl = (S * V * fd.D).sum((1, 2)) / (S * V).sum((1, 2)).clamp_min(1.0)
    res = {"iou": iou, "dt": dtl, "S": S}
    if w_col > 0:
        w = out["hard"] * M * V
        res["chr"], _ = chroma_score(out["color"], fd.img, w)
        res["color"] = out["color"]
    return res


def refine(fit, R0, p0, fd, iters=60, lr_r=0.02, lr_t=0.01, w_dt=2.0, w_col=0.0,
           fix_depth=False, z_prior=None, w_z=0.0, smooth=None, scale=1.0):
    """Adam refinement of rotation (camera-frame left perturbation) and centre position."""
    dev = fit.dev
    R0t = torch.tensor(R0, dtype=torch.float32, device=dev)
    p0t = torch.tensor(p0, dtype=torch.float32, device=dev)
    B = len(p0)
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    dp = torch.zeros(B, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr_r}, {"params": [dp], "lr": lr_t}])
    for it in range(iters):
        Rm = so3_exp(w) @ R0t
        d = dp
        if fix_depth:
            # move only parallel to the image plane at constant depth
            d = torch.cat([dp[:, :2], torch.zeros_like(dp[:, 2:])], 1)
        p = p0t + d
        L = losses(fit, Rm, p * scale, fd, w_col)
        loss = (1 - L["iou"]) + w_dt * L["dt"]
        if w_col > 0:
            loss = loss + w_col * L["chr"]
        if z_prior is not None and w_z > 0:
            loss = loss + w_z * ((p[:, 2] - torch.as_tensor(z_prior, device=dev, dtype=torch.float32)) ** 2)
        if smooth is not None:
            loss = loss + smooth(Rm, p)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        Rm = so3_exp(w) @ R0t
        d = dp if not fix_depth else torch.cat([dp[:, :2], torch.zeros_like(dp[:, 2:])], 1)
        p = p0t + d
        L = losses(fit, Rm, p * scale, fd, w_col=1.0)
    return Rm.detach().cpu().numpy().astype(np.float64), p.detach().cpu().numpy().astype(np.float64), \
        {k: v.detach().cpu().numpy() for k, v in L.items() if k in ("iou", "dt", "chr")}


def refine_grouped(fit, R0, p0, fd, group, iters=80, lr_r=0.02, lr_t=0.01, w_dt=2.0, w_col=0.0,
                   fix_depth=True, frame_w=None, scale=1.0, floor=None):
    """Like refine but parameters are shared within groups. R0,p0: per-group init (G,..);
    group: (B,) index of the group of each frame. Depth (camera z of the centre) kept fixed if fix_depth."""
    dev = fit.dev
    g = torch.as_tensor(np.asarray(group), device=dev, dtype=torch.long)
    R0t = torch.tensor(R0, dtype=torch.float32, device=dev)
    p0t = torch.tensor(p0, dtype=torch.float32, device=dev)
    G = len(p0)
    w = torch.zeros(G, 3, device=dev, requires_grad=True)
    dp = torch.zeros(G, 3, device=dev, requires_grad=True)
    fw = torch.ones(len(group), device=dev) if frame_w is None else torch.as_tensor(frame_w, device=dev, dtype=torch.float32)
    opt = torch.optim.Adam([{"params": [w], "lr": lr_r}, {"params": [dp], "lr": lr_t}])

    def cur():
        Rm = so3_exp(w) @ R0t
        d = torch.cat([dp[:, :2], torch.zeros_like(dp[:, 2:])], 1) if fix_depth else dp
        # keep image position consistent when moving x,y at fixed z
        return Rm, p0t + d

    for it in range(iters):
        Rm, p = cur()
        L = losses(fit, Rm[g], p[g] * scale, fd, w_col)
        loss = (1 - L["iou"]) + w_dt * L["dt"]
        if w_col > 0:
            loss = loss + w_col * L["chr"]
        if floor is not None:
            loss = loss + floor(Rm[g], p[g])
        opt.zero_grad()
        (loss * fw).sum().backward()
        opt.step()
    with torch.no_grad():
        Rm, p = cur()
        L = losses(fit, Rm[g], p[g] * scale, fd, w_col=1.0)
    return Rm.cpu().numpy().astype(np.float64), p.cpu().numpy().astype(np.float64), \
        {k: v.cpu().numpy() for k, v in L.items() if k in ("iou", "dt", "chr")}


def chroma(x):
    s = x.sum(-1, keepdim=True) + 0.05
    return x / s


def chroma_score(col, img, w):
    """mean L1 distance of rg-chromaticity (gain invariant) between render and image over weight w.
    returns (B,) distance (lower better) and the weight mass."""
    d = (chroma(col) - chroma(img)).abs().sum(-1)
    ws = w.sum((1, 2))
    return (d * w).sum((1, 2)) / ws.clamp_min(1.0), ws


def app_feat(x):
    """2-channel appearance feature from BGR in [0,1]: redness and darkness."""
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    red = (r - torch.maximum(g, b)).clamp(0, 1) / (r + 0.05)
    v = x.max(-1).values
    return torch.stack([red, v], -1)


def blur(x, k=5):
    # x (B,H,W,C) box blur
    xc = x.permute(0, 3, 1, 2)
    xc = torch.nn.functional.avg_pool2d(xc, k, stride=1, padding=k // 2, count_include_pad=False)
    return xc.permute(0, 2, 3, 1)


def app_score(col, hard, img, M, V):
    """appearance distance inside rendered∩mask (valid). Brightness normalised by median ratio per frame."""
    fr = app_feat(col)
    fi = app_feat(img)
    w = hard * M * V
    ws = w.sum((1, 2)).clamp_min(1.0)
    # gain for brightness: ratio of means
    gain = (fi[..., 1] * w).sum((1, 2)) / (fr[..., 1] * w).sum((1, 2)).clamp_min(1e-3)
    fr = torch.stack([fr[..., 0], fr[..., 1] * gain[:, None, None]], -1)
    fr, fi = blur(fr * w[..., None]), blur(fi * w[..., None])
    wb = blur(w[..., None]).clamp_min(1e-3)
    fr, fi = fr / wb, fi / wb
    d = (fr - fi).abs()
    return (d[..., 0] * w).sum((1, 2)) / ws, (d[..., 1] * w).sum((1, 2)) / ws


class FloorTerm:
    """Signed distance of the mesh's lowest point to the floor (camera-frame computation).
    h0: floor level of resting contacts (signed plane distance of the true frame-0 human soles).
    mode 'contact': (d_min - h0)^2 ; mode 'above': relu(h0 - tol - d_min)^2."""

    def __init__(self, fit, plane_w, T_world_cam, h0, w, mode, tol=0.01, n_sub=6000):
        n = np.asarray(plane_w[:3], np.float64)
        d = float(plane_w[3])
        Rwc, twc = T_world_cam[:3, :3], T_world_cam[:3, 3]
        self.n = torch.tensor(Rwc.T @ n, dtype=torch.float32, device=fit.dev)
        self.d = float(n @ twc + d)
        step = max(1, len(fit.v) // n_sub)
        self.v = fit.v[::step]  # centred mesh coords
        self.h0, self.w, self.mode, self.tol = h0, w, mode, tol

    def dmin(self, Rm, p, tau=0.004):
        X = torch.einsum("bij,nj->bni", Rm, self.v) + p[:, None]
        s = X @ self.n + self.d
        return -tau * torch.logsumexp(-s / tau, 1)

    def __call__(self, Rm, p):
        m = self.dmin(Rm, p)
        if self.mode == "contact":
            return self.w * (m - self.h0) ** 2
        return self.w * torch.relu(self.h0 - self.tol - m) ** 2
