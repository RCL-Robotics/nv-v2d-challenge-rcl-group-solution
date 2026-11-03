"""Shared helpers: data loading, nvdiffrast rendering of the textured object mesh."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], np.float64)
    Twc = np.array(cam["T_world_cam"], np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    return K, Twc, rec, cam


def load_mesh(inp):
    s = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False)
    g = s.to_geometry()
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB")).astype(np.float32) / 255.0
    return g, np.asarray(g.vertices, np.float64), np.asarray(g.faces, np.int64), np.asarray(g.visual.uv, np.float64), tex


class Renderer:
    def __init__(self, verts, faces, uv, tex, K, H, W, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(verts, dtype=torch.float32, device=device)
        self.f = torch.tensor(faces, dtype=torch.int32, device=device)
        uvt = uv.copy()
        uvt[:, 1] = 1.0 - uvt[:, 1]
        self.uv = torch.tensor(uvt, dtype=torch.float32, device=device)
        self.tex = torch.tensor(tex, dtype=torch.float32, device=device)[None].contiguous()
        self.K = K
        self.H, self.W = H, W

    def proj(self, near=0.05, far=20.0):
        K, H, W = self.K, self.H, self.W
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip space (y down in image, nvdiffrast y up in NDC => flip)
        P = np.array([
            [2 * fx / W, 0, (2 * cx - W) / W, 0],
            [0, 2 * fy / H, (2 * cy - H) / H, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], np.float64)
        return P

    def render(self, T_cam_mesh, crop=None, scale=1.0):
        """T_cam_mesh: (B,4,4) torch or np. crop=(x0,y0,w,h) in full-res pixels, scale applied to crop.
        Returns rgb (B,h,w,3), mask (B,h,w), depth (B,h,w)."""
        T = torch.as_tensor(np.asarray(T_cam_mesh) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,
                            dtype=torch.float32, device=self.dev)
        if T.dim() == 2:
            T = T[None]
        B = T.shape[0]
        K = self.K.copy()
        if crop is None:
            x0, y0, w, h = 0, 0, self.W, self.H
        else:
            x0, y0, w, h = crop
        Kc = K.copy()
        Kc[0, 2] -= x0 - 0.5
        Kc[1, 2] -= y0 - 0.5
        Kc[:2] *= scale
        wr, hr = int(round(w * scale)), int(round(h * scale))
        fx, fy, cx, cy = Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]
        near, far = 0.05, 20.0
        P = torch.tensor([
            [2 * fx / wr, 0, (2 * cx - wr) / wr, 0],
            [0, -2 * fy / hr, -(2 * cy - hr) / hr, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        # nvdiffrast: row 0 = bottom (y=-1) -> flip vertical after
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, resolution=[hr, wr])
        uvi, _ = dr.interpolate(self.uv[None].expand(B, -1, -1).contiguous(), rast, self.f)
        col = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
        mask = rast[..., 3] > 0
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        col = torch.where(mask[..., None], col, torch.zeros_like(col))
        z = torch.where(mask, z[..., 0], torch.zeros_like(z[..., 0]))
        return col.flip(1), mask.flip(1), z.flip(1)


# ---------------------------------------------------------------------------------------------
# geometry helpers
def so3_exp(w):
    """w: (...,3) torch -> (...,3,3)"""
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def rot_angle_deg(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def decimated_mesh(verts, faces, n_faces=6000):
    import open3d as o3d
    m = trimesh.Trimesh(verts, faces, process=False)
    m.merge_vertices()
    mo = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    md = mo.simplify_quadric_decimation(n_faces)
    return np.asarray(md.vertices, np.float64), np.asarray(md.triangles, np.int64)


class SilRenderer:
    """Differentiable (antialiased) silhouette renderer for per-frame crops."""

    def __init__(self, verts, faces, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(verts, dtype=torch.float32, device=device)
        self.f = torch.tensor(faces, dtype=torch.int32, device=device)

    def render(self, R, t, Kc, res):
        """R (B,3,3), t (B,3) camera-frame pose, Kc (B,3,3) crop intrinsics (torch), res int.
        Returns soft silhouette (B,res,res) with image row 0 at top, and depth (B,res,res)."""
        vc = torch.einsum("bij,nj->bni", R, self.v) + t[:, None, :]
        near, far = 0.05, 20.0
        fx, fy, cx, cy = Kc[:, 0, 0], Kc[:, 1, 1], Kc[:, 0, 2], Kc[:, 1, 2]
        x, y, z = vc[..., 0], vc[..., 1], vc[..., 2]
        cxn = (2 * fx[:, None] * x + (2 * cx[:, None] - res) * z) / res
        cyn = (-2 * fy[:, None] * y - (2 * cy[:, None] - res) * z) / res
        czn = (far + near) / (far - near) * z - 2 * far * near / (far - near)
        clip = torch.stack([cxn, cyn, czn, z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, resolution=[res, res])
        m = (rast[..., 3:4] > 0).float()
        zz, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        sil = dr.antialias(m, rast, clip, self.f)
        return sil[..., 0].flip(1), (zz[..., 0] * m[..., 0]).flip(1)


def load_mask_crops(inp, centers, sizes, res=128, chunk=62):
    """Crop object/person masks around per-frame centers (u,v) with square side sizes (px), area-resampled to res.
    Returns obj (F,res,res) float32, per (F,res,res) float32, inimg (F,res,res) float32, boxes (F,3)=(x0,y0,size)."""
    import h5py
    import cv2
    f = h5py.File(f"{inp}/masks.h5", "r")
    O, P = f["object"], f["person"]
    F, H, W = O.shape
    obj = np.zeros((F, res, res), np.float32)
    per = np.zeros((F, res, res), np.float32)
    inim = np.zeros((F, res, res), np.float32)
    boxes = np.zeros((F, 3), np.float64)
    for a in range(0, F, chunk):
        oc = O[a:a + chunk]
        pc = P[a:a + chunk]
        for k in range(oc.shape[0]):
            i = a + k
            s = int(np.ceil(sizes[i]))
            x0 = int(np.round(centers[i][0] - s / 2))
            y0 = int(np.round(centers[i][1] - s / 2))
            boxes[i] = (x0, y0, s)
            for src, dst in ((oc[k], obj), (pc[k], per), (None, inim)):
                buf = np.zeros((s, s), np.float32)
                xa, xb = max(x0, 0), min(x0 + s, W)
                ya, yb = max(y0, 0), min(y0 + s, H)
                if xb > xa and yb > ya:
                    buf[ya - y0:yb - y0, xa - x0:xb - x0] = 1.0 if src is None else src[ya:yb, xa:xb]
                dst[i] = cv2.resize(buf, (res, res), interpolation=cv2.INTER_AREA)
    return obj, per, inim, boxes


def mask_stats(inp, chunk=62):
    """Per-frame object-mask area, bbox (x0,y0,x1,y1) and centroid; person-mask area."""
    import h5py
    f = h5py.File(f"{inp}/masks.h5", "r")
    O = f["object"]
    F = O.shape[0]
    area = np.zeros(F)
    bbox = np.full((F, 4), np.nan)
    cen = np.full((F, 2), np.nan)
    for a in range(0, F, chunk):
        oc = O[a:a + chunk]
        for k in range(oc.shape[0]):
            ys, xs = np.nonzero(oc[k])
            area[a + k] = len(ys)
            if len(ys):
                bbox[a + k] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
                cen[a + k] = (xs.mean(), ys.mean())
    return area, bbox, cen


def crop_K(K, boxes, res):
    """Intrinsics of res x res crops (half-pixel convention: rendered pixel j centre <-> crop coord j+0.5)."""
    Kc = np.repeat(K[None], len(boxes), 0).copy()
    s = res / boxes[:, 2]
    Kc[:, 0, 0] *= s
    Kc[:, 1, 1] *= s
    Kc[:, 0, 2] = (K[0, 2] - boxes[:, 0] + 0.5) * s
    Kc[:, 1, 2] = (K[1, 2] - boxes[:, 1] + 0.5) * s
    return Kc


def fit_silhouettes(sil, R0, anchor0, Kc, obj, valid, Pc, K, iters=300, lr=0.01, w_rot_prior=1.0,
                    w_smooth_t=0.0, w_smooth_r=0.0, free_depth=False, fit_rot=True, frame_w=None,
                    static=False, res=128, w_depth_prior=0.0, rot_mask=None):
    """Joint silhouette fit of a sequence of frames (or one shared pose if static=True).
    R0 (F,3,3) init camera rotations, anchor0 (F,3) init camera-frame position of mesh point Pc.
    Optimises a rotation increment w (camera frame, R = exp(w) R0), the image position of the anchor
    (pixel offsets) and optionally log-depth. Returns R (F,3,3), anchor (F,3), per-frame loss."""
    dev = "cuda"
    F = len(R0)
    tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    R0t, a0, Kct, objt, valt = tt(R0), tt(anchor0), tt(Kc), tt(obj), tt(valid)
    Pct = tt(Pc)
    fw = tt(np.ones(F) if frame_w is None else frame_w)
    n = 1 if static else F
    w = torch.zeros(n, 3, device=dev, requires_grad=True)
    duv = torch.zeros(n, 2, device=dev, requires_grad=True)
    ldz = torch.zeros(n, 1, device=dev, requires_grad=True)
    params = [w, duv] + ([ldz] if free_depth else [])
    opt = torch.optim.Adam([{"params": [w], "lr": lr if fit_rot else 0.0},
                            {"params": [duv], "lr": lr * 30},
                            {"params": [ldz], "lr": lr * 0.3 if free_depth else 0.0}])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u0 = a0[:, 0] / a0[:, 2] * fx + cx
    v0 = a0[:, 1] / a0[:, 2] * fy + cy
    rm = None if rot_mask is None else tt(rot_mask)

    def pose():
        ww = w if rm is None else w * rm
        Rd = so3_exp(ww.expand(F, 3) if static else ww)
        R = Rd @ R0t
        u = u0 + duv[:, 0].expand(F)
        v = v0 + duv[:, 1].expand(F)
        z = a0[:, 2] * torch.exp(ldz[:, 0].expand(F))
        anc = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
        t = anc - torch.einsum("bij,j->bi", R, Pct)
        return R, t, anc, ww

    for it in range(iters):
        opt.zero_grad()
        R, t, anc, ww = pose()
        s, _ = sil.render(R, t, Kct, res)
        d = (s - objt) ** 2 * valt
        norm = ((s.detach() + objt) * valt).sum((1, 2)).clamp_min(20.0)
        lf = d.sum((1, 2)) / norm
        loss = (lf * fw).sum()
        loss = loss + w_rot_prior * (ww ** 2).sum() * (F if static else 1)
        if w_depth_prior > 0:
            loss = loss + w_depth_prior * (ldz ** 2).sum()
        if not static and F > 2:
            if w_smooth_t > 0:
                acc = (anc[2:] - 2 * anc[1:-1] + anc[:-2]) * 100.0  # cm
                loss = loss + w_smooth_t * (acc ** 2).sum()
            if w_smooth_r > 0:
                ra = R[2:] - 2 * R[1:-1] + R[:-2]
                loss = loss + w_smooth_r * (ra ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        R, t, anc, _ = pose()
        s, _ = sil.render(R, t, Kct, res)
        inter = (torch.minimum(s, objt) * valt).sum((1, 2))
        uni = (torch.maximum(s, objt) * valt).sum((1, 2)).clamp_min(1e-6)
        iou = (inter / uni).cpu().numpy()
    return R.cpu().numpy().astype(np.float64), anc.cpu().numpy().astype(np.float64), iou


class CovRenderer:
    """Deterministic (forward-only) silhouette coverage renderer: supersampled rasterisation + average pooling."""

    def __init__(self, verts, faces, ss=4, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(verts, dtype=torch.float32, device=device)
        self.f = torch.tensor(faces, dtype=torch.int32, device=device)
        self.ss = ss

    @torch.no_grad()
    def render(self, R, t, Kc, res):
        ss = self.ss
        r2 = res * ss
        vc = torch.einsum("bij,nj->bni", R, self.v) + t[:, None, :]
        near, far = 0.05, 20.0
        fx, fy = Kc[:, 0, 0] * ss, Kc[:, 1, 1] * ss
        # crop pixel j centre <-> crop coordinate j+0.5 (see crop_K); supersampled pixel centres accordingly
        cx, cy = Kc[:, 0, 2] * ss, Kc[:, 1, 2] * ss
        x, y, z = vc[..., 0], vc[..., 1], vc[..., 2]
        cxn = (2 * fx[:, None] * x + (2 * cx[:, None] - r2) * z) / r2
        cyn = (-2 * fy[:, None] * y - (2 * cy[:, None] - r2) * z) / r2
        czn = (far + near) / (far - near) * z - 2 * far * near / (far - near)
        clip = torch.stack([cxn, cyn, czn, z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, resolution=[r2, r2])
        m = (rast[..., 3] > 0).float().flip(1)
        return torch.nn.functional.avg_pool2d(m[:, None], ss)[:, 0]


def fit_lm(cov, R0, anchor0, Kc, obj, valid, Pc, K, iters=12, fit_rot=True, static=False, w_rot_prior=0.02,
           res=128, frame_w=None, batch=96, rot_step=0.01, uv_step=0.5, lam0=1e-2, init_centroid=True):
    """Deterministic Levenberg-Marquardt silhouette fit (finite differences on forward renders).
    Per frame: rotation increment w (camera frame, R = exp(w) R0; optional) and image offset of the anchor (px);
    depth fixed. static=True: one shared image offset for all frames (rotations fixed).
    Returns R (F,3,3), anchor (F,3), IoU (F)."""
    dev = "cuda"
    F = len(R0)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a0 = np.asarray(anchor0, np.float64)
    u0 = a0[:, 0] / a0[:, 2] * fx + cx
    v0 = a0[:, 1] / a0[:, 2] * fy + cy
    z0 = a0[:, 2]
    fw = np.ones(F) if frame_w is None else np.asarray(frame_w, np.float64)
    npar = (3 if fit_rot else 0) + 2
    P = np.zeros((1 if static else F, npar))
    steps = np.array(([rot_step] * 3 if fit_rot else []) + [uv_step, uv_step])

    def poses(par, idx):
        """par (B,npar) for frames idx -> R (B,3,3) torch, t (B,3) torch"""
        par_t = torch.tensor(par, dtype=torch.float64, device=dev)
        R0t = torch.tensor(R0[idx], dtype=torch.float64, device=dev)
        if fit_rot:
            R = so3_exp(par_t[:, :3]) @ R0t
            duv = par_t[:, 3:5]
        else:
            R = R0t
            duv = par_t[:, 0:2]
        u = torch.tensor(u0[idx], dtype=torch.float64, device=dev) + duv[:, 0]
        v = torch.tensor(v0[idx], dtype=torch.float64, device=dev) + duv[:, 1]
        z = torch.tensor(z0[idx], dtype=torch.float64, device=dev)
        anc = torch.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
        t = anc - torch.einsum("bij,j->bi", R, torch.tensor(Pc, dtype=torch.float64, device=dev))
        return R, t, anc

    def residuals(par_frames, idx):
        """returns r (B, res*res) float64 numpy weighted residuals"""
        R, t, _ = poses(par_frames, idx)
        s = cov.render(R.float(), t.float(), torch.tensor(Kc[idx], dtype=torch.float32, device=dev), res)
        o = torch.tensor(obj[idx], dtype=torch.float32, device=dev)
        vm = torch.tensor(valid[idx], dtype=torch.float32, device=dev)
        r = ((s - o) * torch.sqrt(vm)).reshape(len(idx), -1).double()
        return r, s

    # coarse init of the image offset: match centroids of visible object mask and visible silhouette (2 rounds)
    if init_centroid:
        yy, xx = torch.meshgrid(torch.arange(res, device=dev, dtype=torch.float32),
                                torch.arange(res, device=dev, dtype=torch.float32), indexing="ij")
        for _ in range(2):
            dpx_all = np.zeros((F, 2))
            ok_all = np.zeros(F, bool)
            for b in range(0, F, batch):
                idx = np.arange(b, min(F, b + batch))
                par = P[np.zeros(len(idx), int)] if static else P[idx]
                _, s = residuals(par, idx)
                vm = torch.tensor(valid[idx], dtype=torch.float32, device=dev)
                o = torch.tensor(obj[idx], dtype=torch.float32, device=dev)
                ws, wo = (s * vm).sum((1, 2)), (o * vm).sum((1, 2))
                ok_all[idx] = ((ws > 10) & (wo > 10)).cpu().numpy()
                dx = ((o * vm * xx).sum((1, 2)) / wo.clamp_min(1) - (s * vm * xx).sum((1, 2)) / ws.clamp_min(1))
                dy = ((o * vm * yy).sum((1, 2)) / wo.clamp_min(1) - (s * vm * yy).sum((1, 2)) / ws.clamp_min(1))
                scl = Kc[idx, 0, 0] / K[0, 0]                    # crop px per image px
                dpx_all[idx] = np.stack([dx.cpu().numpy(), dy.cpu().numpy()], 1) / scl[:, None]
            if static:
                if ok_all.any():
                    P[0, -2:] += dpx_all[ok_all].mean(0)
            else:
                P[:, -2:] += dpx_all * ok_all[:, None]
    # per-frame normalisation (fixed): visible object + initial silhouette area
    norm = np.zeros(F)
    for b in range(0, F, batch):
        idx = np.arange(b, min(F, b + batch))
        par = P[np.zeros(len(idx), int)] if static else P[idx]
        _, s = residuals(par, idx)
        vm = torch.tensor(valid[idx], dtype=torch.float32, device=dev)
        o = torch.tensor(obj[idx], dtype=torch.float32, device=dev)
        norm[idx] = np.maximum(((s + o) * vm).sum((1, 2)).cpu().numpy(), 20.0)
    sc = np.sqrt(fw / norm)

    def eval_all(Pcur):
        """cost per frame, J^T J and J^T r per frame (or summed if static)."""
        cost = np.zeros(F)
        JTJ = np.zeros((F, npar, npar))
        JTr = np.zeros((F, npar))
        for b in range(0, F, batch):
            idx = np.arange(b, min(F, b + batch))
            par = Pcur[np.zeros(len(idx), int)] if static else Pcur[idx]
            r0, _ = residuals(par, idx)
            r0 = r0 * torch.tensor(sc[idx], dtype=torch.float64, device=dev)[:, None]
            cost[idx] = (r0 ** 2).sum(1).cpu().numpy()
            J = []
            for k in range(npar):
                pp = par.copy()
                pp[:, k] += steps[k]
                rk, _ = residuals(pp, idx)
                rk = rk * torch.tensor(sc[idx], dtype=torch.float64, device=dev)[:, None]
                J.append((rk - r0) / steps[k])
            J = torch.stack(J, -1)                          # B, N, npar
            JTJ[idx] = torch.einsum("bnk,bnl->bkl", J, J).cpu().numpy()
            JTr[idx] = torch.einsum("bnk,bn->bk", J, r0).cpu().numpy()
        return cost, JTJ, JTr

    lam = np.full(len(P), lam0)
    cost, JTJ, JTr = eval_all(P)
    for it in range(iters):
        if static:
            A_ = JTJ.sum(0, keepdims=True)
            g_ = JTr.sum(0, keepdims=True)
            c_ = cost.sum(keepdims=True)
        else:
            A_, g_, c_ = JTJ.copy(), JTr.copy(), cost.copy()
        if fit_rot:
            A_[:, :3, :3] += w_rot_prior * np.eye(3)
            g_[:, :3] += w_rot_prior * P[:, :3]
            c_ = c_ + 0.5 * w_rot_prior * (P[:, :3] ** 2).sum(1)
        D = np.stack([np.diag(np.diag(a)) + 1e-9 * np.eye(npar) for a in A_])
        delta = -np.linalg.solve(A_ + lam[:, None, None] * D, g_[..., None])[..., 0]
        Pn = P + delta
        cost_n, JTJ_n, JTr_n = eval_all(Pn)
        if static:
            cn = cost_n.sum(keepdims=True)
        else:
            cn = cost_n.copy()
        if fit_rot:
            cn = cn + 0.5 * w_rot_prior * (Pn[:, :3] ** 2).sum(1)
        better = cn < c_
        P = np.where(better[:, None], Pn, P)
        lam = np.where(better, lam * 0.3, lam * 5.0)
        if static:
            if better[0]:
                cost, JTJ, JTr = cost_n, JTJ_n, JTr_n
        else:
            cost = np.where(better, cost_n, cost)
            JTJ = np.where(better[:, None, None], JTJ_n, JTJ)
            JTr = np.where(better[:, None], JTr_n, JTr)
    # outputs
    Rs, As, ious = [], [], []
    for b in range(0, F, batch):
        idx = np.arange(b, min(F, b + batch))
        par = P[np.zeros(len(idx), int)] if static else P[idx]
        R, t, anc = poses(par, idx)
        s = cov.render(R.float(), t.float(), torch.tensor(Kc[idx], dtype=torch.float32, device=dev), res)
        o = torch.tensor(obj[idx], dtype=torch.float32, device=dev)
        vm = torch.tensor(valid[idx], dtype=torch.float32, device=dev)
        inter = (torch.minimum(s, o) * vm).sum((1, 2))
        uni = (torch.maximum(s, o) * vm).sum((1, 2)).clamp_min(1e-6)
        ious.append((inter / uni).cpu().numpy())
        Rs.append(R.cpu().numpy())
        As.append(anc.cpu().numpy())
    return np.concatenate(Rs), np.concatenate(As), np.concatenate(ious)
