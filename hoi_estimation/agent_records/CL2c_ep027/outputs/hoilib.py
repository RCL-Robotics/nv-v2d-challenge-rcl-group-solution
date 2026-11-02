"""Shared helpers: mesh loading, nvdiffrast rendering, pose utilities."""
import numpy as np
import cv2
import torch
import trimesh
import nvdiffrast.torch as dr

_GLCTX = None


def glctx():
    global _GLCTX
    if _GLCTX is None:
        _GLCTX = dr.RasterizeCudaContext()
    return _GLCTX


class TexMesh:
    def __init__(self, path, device="cuda"):
        m = trimesh.load(path, force="scene", process=False).to_geometry()
        self.trimesh = m
        self.v = torch.tensor(m.vertices, dtype=torch.float32, device=device)
        self.f = torch.tensor(m.faces, dtype=torch.int32, device=device)
        uv = np.asarray(m.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        mw = trimesh.Trimesh(m.vertices, m.faces, process=False)
        mw.merge_vertices(digits_vertex=6)
        self.vw = torch.tensor(mw.vertices, dtype=torch.float32, device=device)
        self.fw = torch.tensor(mw.faces, dtype=torch.int32, device=device)
        self.device = device


def project_clip(v_cam, K, H, W, near=0.01, far=20.0):
    """v_cam (B,N,3) OpenCV camera coords -> clip coords for nvdiffrast (pixel centres matching OpenCV)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
    # ndc x in [-1,1] maps to pixel [0,W]; OpenCV pixel centre u at integer -> u+0.5 in continuous coords
    u = fx * x / z + cx + 0.5
    vv = fy * y / z + cy + 0.5
    xn = 2 * u / W - 1
    yn = 2 * vv / H - 1  # nvdiffrast: row 0 at y=-1
    zn = (z - near) / (far - near) * 2 - 1
    w = torch.ones_like(z)
    return torch.stack([xn * z, yn * z, zn * z, z], -1)


def render(mesh, T_cam_mesh, K, H, W, crop=None, need_rgb=True):
    """T_cam_mesh: (B,4,4) torch. crop=(x0,y0,w,h) renders only a window (K shifted).
    Returns rgb (B,h,w,3), mask (B,h,w), depth (B,h,w)."""
    K = np.asarray(K, dtype=np.float64).copy()
    if crop is not None:
        x0, y0, W, H = crop
        K[0, 2] -= x0
        K[1, 2] -= y0
    R = T_cam_mesh[:, :3, :3]
    t = T_cam_mesh[:, :3, 3]
    vc = torch.einsum("bij,nj->bni", R, mesh.v) + t[:, None]
    clip = project_clip(vc, K, H, W)
    rast, _ = dr.rasterize(glctx(), clip.contiguous(), mesh.f, (H, W))
    mask = rast[..., 3] > 0
    depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, mesh.f)
    depth = depth[..., 0] * mask
    rgb = None
    if need_rgb:
        texc, _ = dr.interpolate(mesh.uv[None], rast, mesh.f)
        rgb = dr.texture(mesh.tex, texc, filter_mode="linear") * mask[..., None]
    return rgb, mask, depth


def rotmat_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def so3_exp(w):
    """w (B,3) -> (B,3,3) rodrigues, differentiable."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s = torch.sin(th)[..., None]
    c = torch.cos(th)[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


def render_soft(mesh, T_cam_mesh, K, crop, need_rgb=False):
    """Antialiased (differentiable wrt pose) silhouette from the welded mesh, and optional rgb."""
    x0, y0, W, H = crop
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    R = T_cam_mesh[:, :3, :3]
    t = T_cam_mesh[:, :3, 3]
    vc = torch.einsum("bij,nj->bni", R, mesh.vw) + t[:, None]
    clip = project_clip(vc, K, H, W).contiguous()
    rast, _ = dr.rasterize(glctx(), clip, mesh.fw, (H, W))
    m = (rast[..., 3:4] > 0).float()
    sil = dr.antialias(m, rast, clip, mesh.fw)[..., 0]
    rgb = None
    if need_rgb:
        rgb, _, _ = render(mesh, T_cam_mesh.detach(), K, H, W, crop=(0, 0, W, H))
    return sil, rgb


def crop_box_from_mask(mask, fallback_uv=None, min_size=64, scale=1.8):
    ys, xs = np.nonzero(mask)
    if len(xs) >= 20:
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        s = max(xs.max() - xs.min(), ys.max() - ys.min())
    else:
        cx, cy = fallback_uv
        s = min_size
    S = max(min_size, scale * s)
    return np.array([cx - S / 2, cy - S / 2, S], dtype=np.float64)


def resample_crop(img, box, R, interp=1):
    """box=(x0,y0,S) -> RxR image (interp 1=bilinear, 0=nearest)."""
    x0, y0, S = box
    s = S / R
    M = np.array([[s, 0, x0 + 0.5 * s - 0.5], [0, s, y0 + 0.5 * s - 0.5]], dtype=np.float64)
    return cv2.warpAffine(img, M, (R, R), flags=interp | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_K(K, box, R):
    x0, y0, S = box
    s = R / S
    Kc = np.array(K, dtype=np.float64).copy()
    Kc[0, 0] *= s
    Kc[1, 1] *= s
    Kc[0, 2] = (K[0, 2] + 0.5 - x0) * s - 0.5
    Kc[1, 2] = (K[1, 2] + 0.5 - y0) * s - 0.5
    return Kc


def project_clip_batch(v_cam, Kb, H, W, near=0.01, far=20.0):
    """v_cam (B,N,3), Kb (B,3,3) torch."""
    fx, fy = Kb[:, 0, 0][:, None], Kb[:, 1, 1][:, None]
    cx, cy = Kb[:, 0, 2][:, None], Kb[:, 1, 2][:, None]
    x, y, z = v_cam[..., 0], v_cam[..., 1], v_cam[..., 2]
    u = fx * x / z + cx + 0.5
    vv = fy * y / z + cy + 0.5
    xn = 2 * u / W - 1
    yn = 2 * vv / H - 1
    zn = (z - near) / (far - near) * 2 - 1
    return torch.stack([xn * z, yn * z, zn * z, z], -1)


def render_sil_batch(mesh, T, Kb, R):
    vc = torch.einsum("bij,nj->bni", T[:, :3, :3], mesh.vw) + T[:, None, :3, 3]
    clip = project_clip_batch(vc, Kb, R, R).contiguous()
    rast, _ = dr.rasterize(glctx(), clip, mesh.fw, (R, R))
    m = (rast[..., 3:4] > 0).float()
    return dr.antialias(m, rast, clip, mesh.fw)[..., 0]


def fit_silhouettes(mesh, Kb, obs, valid, T_init, iters=120, fix_z=True, lr_r=0.02, lr_uv=0.6, lr_z=0.004,
                    rot_prior=0.0, chunk=256, occ_pen=0.0):
    """Batched silhouette refinement.
    Kb (B,3,3) crop intrinsics (numpy), obs/valid (B,R,R) float numpy, T_init (B,4,4) camera-frame poses.
    Parametrisation: R = exp(w) R0 ; translation = z * K^-1 [u, v, 1] with (u, v) the projected centre.
    Returns refined T (B,4,4) numpy and IoU (B,). With occ_pen>0 the returned score counts rendered pixels that
    fall on person-occluded (invalid) pixels as occ_pen of a mismatch, i.e. hypotheses that need to hide behind the
    person to explain the mask are not scored as perfect."""
    B, R = obs.shape[0], obs.shape[1]
    outT = np.zeros((B, 4, 4))
    outI = np.zeros(B)
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        K_t = torch.tensor(Kb[s:e], dtype=torch.float32, device="cuda")
        O = torch.tensor(obs[s:e], dtype=torch.float32, device="cuda")
        V = torch.tensor(valid[s:e], dtype=torch.float32, device="cuda")
        T0 = torch.tensor(T_init[s:e], dtype=torch.float32, device="cuda")
        t0 = T0[:, :3, 3]
        uv0 = torch.stack([K_t[:, 0, 0] * t0[:, 0] / t0[:, 2] + K_t[:, 0, 2], K_t[:, 1, 1] * t0[:, 1] / t0[:, 2] + K_t[:, 1, 2]], -1)
        w = torch.zeros(e - s, 3, device="cuda", requires_grad=True)
        duv = torch.zeros(e - s, 2, device="cuda", requires_grad=True)
        dz = torch.zeros(e - s, 1, device="cuda", requires_grad=True)
        groups = [{"params": [w], "lr": lr_r}, {"params": [duv], "lr": lr_uv}]
        if not fix_z:
            groups.append({"params": [dz], "lr": lr_z})
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.05 * lr_r)
        norm = (O * V).sum((1, 2)).clamp(min=30)
        blur = torch.nn.AvgPool2d(5, 1, 2)

        def make_T():
            Rm = so3_exp(w) @ T0[:, :3, :3]
            uv = uv0 + duv
            z = t0[:, 2:3] * torch.exp(dz)
            x = (uv[:, 0:1] - K_t[:, 0, 2:3]) / K_t[:, 0, 0:1] * z
            y = (uv[:, 1:2] - K_t[:, 1, 2:3]) / K_t[:, 1, 1:2] * z
            T = torch.zeros_like(T0)
            T[:, :3, :3] = Rm
            T[:, :3, 3] = torch.cat([x, y, z], -1)
            T[:, 3, 3] = 1
            return T

        for it in range(iters):
            T = make_T()
            sil = render_sil_batch(mesh, T, K_t, R)
            l1 = ((sil - O) ** 2 * V).sum((1, 2)) / norm
            if it < iters // 2:  # coarse: also match blurred silhouettes for a wider basin
                l1 = l1 + ((blur(sil[:, None])[:, 0] - blur(O[:, None])[:, 0]) ** 2 * V).sum((1, 2)) / norm
            loss = l1.sum() + rot_prior * (w * w).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            T = make_T()
            sil = render_sil_batch(mesh, T, K_t, R)
            h = (sil > 0.5).float()
            inter = (h * O * V).sum((1, 2))
            uni = ((h + O - h * O) * V).sum((1, 2)) + occ_pen * (h * (1 - V)).sum((1, 2))
            outI[s:e] = (inter / uni.clamp(min=1)).cpu().numpy()
            outT[s:e] = T.cpu().numpy()
    return outT, outI


# ---------------------------------------------------------------- photometric point model
def surface_points(mesh, n=20000, seed=0):
    tm = trimesh.Trimesh(mesh.vw.cpu().numpy(), mesh.fw.cpu().numpy(), process=False)
    pts, fid = trimesh.sample.sample_surface_even(tm, n, seed=seed)
    return pts.astype(np.float32), tm.face_normals[fid].astype(np.float32)


def sample_point_colors(mesh, P, N, T, K, img, objm, perm, box, depth_tol=0.008):
    """P,N (n,3) torch mesh-frame points/normals. T (B,4,4) torch camera-frame poses.
    img float numpy (full frame or the box crop, already blurred), objm/perm bool full masks, box=(x0,y0,x1,y1).
    Returns colours (B,n,3) torch normalised by the mean visible colour, and visibility (B,n) bool."""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    _, msk, dep = render(mesh, T, K, H, W, crop=(x0, y0, W, H), need_rgb=False)
    pc = torch.einsum("bij,nj->bni", T[:, :3, :3], P) + T[:, None, :3, 3]
    nc = torch.einsum("bij,nj->bni", T[:, :3, :3], N)
    u = K[0, 0] * pc[..., 0] / pc[..., 2] + K[0, 2] - x0
    v = K[1, 1] * pc[..., 1] / pc[..., 2] + K[1, 2] - y0
    gx = (u + 0.5) / W * 2 - 1
    gy = (v + 0.5) / H * 2 - 1
    grid = torch.stack([gx, gy], -1)[:, None]  # B,1,n,2
    d_s = torch.nn.functional.grid_sample(dep[:, None], grid, mode="nearest", align_corners=False)[:, 0, 0]
    view = pc / pc.norm(dim=-1, keepdim=True)
    vis = ((d_s - pc[..., 2]).abs() < depth_tol) & ((nc * view).sum(-1) < -0.25)
    om = cv2.erode(objm[y0:y1, x0:x1].astype(np.uint8), np.ones((3, 3)))
    pm = cv2.dilate(perm[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5)))
    ok_img = torch.tensor((om > 0) & (pm == 0), dtype=torch.float32, device="cuda")[None, None].expand(T.shape[0], 1, H, W)
    ok_s = torch.nn.functional.grid_sample(ok_img, grid, mode="nearest", align_corners=False)[:, 0, 0] > 0.5
    vis = vis & ok_s
    if img.shape[0] != H or img.shape[1] != W:
        img = img[y0:y1, x0:x1]
    im = torch.tensor(img, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None].expand(T.shape[0], 3, H, W)
    col = torch.nn.functional.grid_sample(im, grid, mode="bilinear", align_corners=False)[:, :, 0].permute(0, 2, 1)
    cnt = vis.float().sum(1).clamp(min=1)
    mean = (col * vis[..., None].float()).sum(1) / cnt[:, None]
    col = col / mean[:, None].clamp(min=1e-3)
    return col, vis


def model_corr(col, vis, model, model_cnt, min_cnt=2, min_pts=40):
    """Pearson correlation of luminance between sampled colours and the model, per hypothesis."""
    ok = vis & (model_cnt[None] >= min_cnt)
    a = col.mean(-1)
    b = model.mean(-1)[None].expand_as(a)
    n = ok.float().sum(1)
    w = ok.float()
    ma = (a * w).sum(1) / n.clamp(min=1)
    mb = (b * w).sum(1) / n.clamp(min=1)
    da = (a - ma[:, None]) * w
    db = (b - mb[:, None]) * w
    c = (da * db).sum(1) / torch.sqrt((da * da).sum(1) * (db * db).sum(1) + 1e-9)
    c = torch.where(n >= min_pts, c, torch.full_like(c, float("nan")))
    return c, n


# ---------------------------------------------------------------- yaw / tilt decomposition
def rot_between(a, b):
    """Minimal rotation taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999999:
        ax = np.cross(a, [1, 0, 0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0, 1, 0])
        ax /= np.linalg.norm(ax)
        return axis_angle(ax, np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def axis_angle(ax, ang):
    ax = np.asarray(ax, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


class YawFrame:
    """R_world_mesh = S(tilt) @ Rot(down, psi) @ R0, with mesh_down the mesh axis pointing to the basket bottom."""

    def __init__(self, down_world, mesh_down=(0, 1, 0)):
        self.down = np.asarray(down_world, float) / np.linalg.norm(down_world)
        self.md = np.asarray(mesh_down, float)
        self.R0 = rot_between(self.md, self.down)

    def decompose(self, R):
        d = R @ self.md
        S = rot_between(self.down, d)
        Rt = S.T @ R  # maps md -> down
        M = Rt @ self.R0.T  # rotation about self.down
        # angle about axis 'down'
        rv = (np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]) / 2)
        s = rv @ self.down
        c = (np.trace(M) - 1) / 2
        psi = np.arctan2(s, c)
        return S, psi, d

    def compose(self, d, psi):
        S = rot_between(self.down, d)
        return S @ axis_angle(self.down, psi) @ self.R0
