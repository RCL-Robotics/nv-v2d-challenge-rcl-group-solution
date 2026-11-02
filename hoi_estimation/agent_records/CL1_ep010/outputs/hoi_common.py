"""Shared helpers: data loading, mesh, nvdiffrast rendering."""
import json
import numpy as np
import torch
import trimesh
import h5py
import cv2
import nvdiffrast.torch as dr

H, W = 1152, 1536


def load_camera(inputs):
    c = json.load(open(f"{inputs}/camera.json"))
    return np.array(c["K"], float), np.array(c["T_world_cam"], float), np.array(c["K_pipeline"], float)


def load_mesh(inputs):
    m = trimesh.load(f"{inputs}/object_mesh.glb", force="scene", process=False).to_geometry()
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int32)
    uv = np.asarray(m.visual.uv, np.float32)
    tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255.0
    return m, v, f, uv, tex


def read_video(path, frames=None, scale=1.0):
    cap = cv2.VideoCapture(path)
    out = {}
    i = 0
    want = None if frames is None else set(frames)
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if want is None or i in want:
            if scale != 1.0:
                im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            out[i] = im[:, :, ::-1].copy()
        i += 1
        if want is not None and i > max(want):
            break
    return out


def proj_matrix(K, w, h, near=0.05, far=20.0):
    """OpenCV K -> OpenGL clip matrix (for camera frame x right, y down, z forward)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.zeros((4, 4))
    P[0, 0] = 2 * fx / w
    P[0, 2] = 2 * (cx + 0.5) / w - 1  # pixel centres at integer coords
    P[1, 1] = 2 * fy / h
    P[1, 2] = 2 * (cy + 0.5) / h - 1
    P[2, 2] = (far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = 1
    return P


class Renderer:
    def __init__(self, v, f, uv=None, tex=None, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        self.uv = None if uv is None else torch.tensor(uv * np.array([1, -1], np.float32) + np.array([0, 1], np.float32), device=device)
        self.tex = None if tex is None else torch.tensor(tex, device=device)[None].contiguous()

    def render(self, T_cam_mesh, K, w, h, want_color=False, near=0.05, far=20.0):
        """T_cam_mesh: (B,4,4) torch float32 (differentiable).  Returns mask (B,h,w) [antialiased],
        depth (B,h,w) (0 = background), color (B,h,w,3) optional."""
        B = T_cam_mesh.shape[0]
        P = torch.tensor(proj_matrix(K, w, h, near, far), dtype=torch.float32, device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)  # (N,4)
        vc = torch.einsum("bij,nj->bni", T_cam_mesh, vh)  # camera coords
        clip = torch.einsum("ij,bnj->bni", P, vc)
        # OpenGL y is up in NDC with origin at bottom; image row 0 = NDC y=-1 for nvdiffrast (row 0 is y=-1)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (h, w))
        mask = (rast[..., 3:] > 0).float()
        mask_aa = dr.antialias(mask, rast, clip.contiguous(), self.f)[..., 0]
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        depth = z[..., 0] * (rast[..., 3] > 0)
        col = None
        if want_color and self.tex is not None:
            uvi, uvd = dr.interpolate(self.uv[None], rast, self.f, rast_db=None)
            col = dr.texture(self.tex, uvi, filter_mode="linear")
            col = col * (rast[..., 3:] > 0)
        return mask_aa, depth, col


def load_masks(inputs, frames=None, scale=1.0, key="object"):
    f = h5py.File(f"{inputs}/masks.h5", "r")
    d = f[key]
    idx = range(d.shape[0]) if frames is None else frames
    out = []
    for i in idx:
        m = d[i]
        if scale != 1.0:
            m = cv2.resize(m.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) > 0
        out.append(m)
    return np.stack(out)


def scale_K(K, s):
    K = K.copy()
    K[0, 0] *= s; K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


def so3_exp(w):
    """w (...,3) -> (...,3,3) rotation, differentiable."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)[..., None]
    k = w / th[..., 0]
    Kx = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2]; Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def compose(T0, w, dt, c):
    """T0 (B,4,4) torch; perturb rotation about mesh point c (3,) (in camera frame, left-multiplied)."""
    R0 = T0[:, :3, :3]; t0 = T0[:, :3, 3]
    Rd = so3_exp(w)
    pc = (R0 @ c[:, None])[..., 0] + t0  # centre in cam
    R = Rd @ R0
    t = pc + dt - (R @ c[:, None])[..., 0]
    T = torch.zeros_like(T0)
    T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
    return T


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.transpose(0, 2, 1) @ Rb, axis1=1, axis2=2) - 1) / 2 if Ra.ndim == 3 else (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


class SilhouetteData:
    """Object / person masks at reduced scale on the GPU (uint8)."""

    def __init__(self, inputs, scale=0.5, device="cuda", cache=None):
        self.s = scale
        import os
        if cache and os.path.exists(cache):
            z = np.load(cache); om, pm = z["om"], z["pm"]
        else:
            om = load_masks(inputs, None, scale)
            pm = load_masks(inputs, None, scale, "person")
            if cache:
                np.savez(cache, om=om, pm=pm)
        self.obj = torch.tensor(om, device=device)
        self.valid = torch.tensor(~pm | om, device=device)
        self.h, self.w = om.shape[1:]

    def loss(self, mk, idx):
        """mk (B,h,w) soft render; idx frame indices. Returns per-frame loss (B,) normalised by mask area."""
        t = self.obj[idx].float(); v = self.valid[idx].float()
        e = ((mk - t) ** 2 * v).sum((1, 2))
        # coarse level widens the basin of attraction
        mk4 = torch.nn.functional.avg_pool2d(mk[:, None], 4)[:, 0]
        t4 = torch.nn.functional.avg_pool2d(t[:, None], 4)[:, 0]
        v4 = torch.nn.functional.avg_pool2d(v[:, None], 4)[:, 0]
        e4 = ((mk4 - t4) ** 2 * (v4 > 0.99)).sum((1, 2)) * 16
        area = t.sum((1, 2)).clamp(min=50)
        return (e + e4) / area

    def iou(self, mk, idx):
        o = self.obj[idx].bool(); v = self.valid[idx].bool(); m = (mk > 0.5) & v
        o = o & v
        return (m & o).sum((1, 2)).float() / (m | o).sum((1, 2)).clamp(min=1).float()


def pooled_counts(SD, frames):
    """For a static phase: per-pixel number of frames where the pixel is visible object / visible background."""
    o = SD.obj[frames].float(); v = SD.valid[frames].float()
    return (o * v).sum(0), ((1 - o) * v).sum(0)


def fit_static_pose(R, Ks, h, w, nobj, nbg, T_init, iters=250, lr=2e-3, extra=None):
    """Fit one pose to pooled silhouette evidence of a static phase. Returns (T (4,4) np, iou)."""
    dev = nobj.device
    T0 = torch.tensor(T_init[None], dtype=torch.float32, device=dev)
    c = R.v.mean(0)
    wv = torch.zeros(1, 3, device=dev, requires_grad=True); dt = torch.zeros(1, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([wv, dt], lr=lr)
    nf = (nobj + nbg).max()
    for it in range(iters):
        T = compose(T0, wv, dt, c)
        mk, _, _ = R.render(T, Ks, w, h)
        m = mk[0]
        L = (nobj * (1 - m) ** 2 + nbg * m ** 2).sum() / nf
        m4 = torch.nn.functional.avg_pool2d(m[None, None], 4)[0, 0]
        o4 = torch.nn.functional.avg_pool2d(nobj[None, None], 4)[0, 0]; b4 = torch.nn.functional.avg_pool2d(nbg[None, None], 4)[0, 0]
        L = L + 16 * (o4 * (1 - m4) ** 2 + b4 * m4 ** 2).sum() / nf
        if extra is not None:
            L = L + extra(T)
        opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        T = compose(T0, wv, dt, c)
        mk, _, _ = R.render(T, Ks, w, h); m = mk[0] > 0.5
        inter = (nobj * m).sum(); iou = (inter / (inter + (nbg * m).sum() + (nobj * ~m).sum())).item()
    return T[0].cpu().numpy().astype(np.float64), iou


def flip_about_axis(T, axis, centre):
    """Rotate the object 180 deg about a mesh-frame axis through `centre` (mesh coords)."""
    Rf = np.eye(3)
    for k in range(3):
        if k != axis:
            Rf[k, k] = -1
    F = np.eye(4); F[:3, :3] = Rf; F[:3, 3] = centre - Rf @ centre
    return T @ F


def klt_tracks(video_frames, om, pm, frames):
    """Pairwise KLT tracks on visible object texture between consecutive frames.
    video_frames: dict i->RGB; om/pm: dict or array indexable by position in `frames`.
    Returns array (n,5): frame i, x_i, y_i, x_{i+1}, y_{i+1} (full-res pixels)."""
    out = []
    for k in range(len(frames) - 1):
        i = frames[k]
        g0 = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY); g1 = cv2.cvtColor(video_frames[i + 1], cv2.COLOR_RGB2GRAY)
        allow = (cv2.erode(om[k].astype(np.uint8), np.ones((11, 11))) > 0) & ~(cv2.dilate(pm[k].astype(np.uint8), np.ones((15, 15))) > 0)
        pts = cv2.goodFeaturesToTrack(g0, 400, 0.01, 6, mask=allow.astype(np.uint8), blockSize=7)
        if pts is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(15, 15), maxLevel=3)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(15, 15), maxLevel=3)
        fb = np.linalg.norm(pts - p0b, axis=-1)[:, 0]
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 0.5)
        q = np.round(p1[:, 0]).astype(int); q[:, 0] = q[:, 0].clip(0, W - 1); q[:, 1] = q[:, 1].clip(0, H - 1)
        ok &= om[k + 1][q[:, 1], q[:, 0]] & ~pm[k + 1][q[:, 1], q[:, 0]]
        for p, qq in zip(pts[ok, 0], p1[ok, 0]):
            out.append((i, p[0], p[1], qq[0], qq[1]))
    return np.array(out, np.float64).reshape(-1, 5)


def klt_lift(R, K, Tcam, tracks, scale=0.5):
    """Lift track starts to mesh coordinates using rendered depth under poses Tcam (dict/array by frame).
    Returns (frame, X_mesh (n,3), q (n,2), valid)."""
    Ks = scale_K(K, scale); h, w = int(H * scale), int(W * scale)
    fr = np.unique(tracks[:, 0]).astype(int)
    Xm = np.zeros((len(tracks), 3)); val = np.zeros(len(tracks), bool)
    for c0 in range(0, len(fr), 32):
        ch = fr[c0:c0 + 32]
        with torch.no_grad():
            _, d, _ = R.render(torch.tensor(np.stack([Tcam[i] for i in ch]), dtype=torch.float32, device=R.dev), Ks, w, h)
        d = d.cpu().numpy()
        for j, i in enumerate(ch):
            sel = np.where(tracks[:, 0] == i)[0]
            u = (tracks[sel, 1] + 0.5) * scale - 0.5; vv = (tracks[sel, 2] + 0.5) * scale - 0.5
            z = d[j, np.round(vv).astype(int).clip(0, h - 1), np.round(u).astype(int).clip(0, w - 1)]
            ok = z > 0
            X = np.stack([(tracks[sel, 1] - K[0, 2]) / K[0, 0] * z, (tracks[sel, 2] - K[1, 2]) / K[1, 1] * z, z], 1)
            T = Tcam[i]
            Xm[sel] = (X - T[:3, 3]) @ T[:3, :3]
            val[sel] = ok
    return Xm, val


def klt_residual(K, Tcam, tracks, Xm, val):
    """Reprojection error (px) of lifted points into frame i+1."""
    i1 = tracks[:, 0].astype(int) + 1
    Tn = np.stack([Tcam[i] for i in i1])
    Xc = np.einsum("nij,nj->ni", Tn[:, :3, :3], Xm) + Tn[:, :3, 3]
    q = Xc @ K.T; q = q[:, :2] / q[:, 2:]
    res = np.linalg.norm(q - tracks[:, 3:5], axis=1)
    res[~val] = np.nan
    return res
