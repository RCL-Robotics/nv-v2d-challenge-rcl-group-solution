"""Shared helpers: data loading, nvdiffrast rendering of the textured mesh, pose utilities."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
import cv2
import h5py


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], dtype=np.float64)
    T_wc = np.array(cam["T_world_cam"], dtype=np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    rec = {k: rec[k] for k in rec.files}
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], dtype=np.float64)
    return dict(K=K, T_wc=T_wc, rec=rec, plane=plane, W=cam["width"], H=cam["height"],
                K_pipe=np.array(cam["K_pipeline"], dtype=np.float64))


def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    return m


def read_video(path, scale=1.0, frames=None):
    cap = cv2.VideoCapture(path)
    out = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if frames is None or i in frames:
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            out.append(f[:, :, ::-1].copy())  # RGB
        i += 1
    return np.stack(out)


def read_masks(path, key, scale=1.0, frames=None):
    with h5py.File(path, "r") as f:
        d = f[key]
        idx = range(d.shape[0]) if frames is None else frames
        out = []
        for i in idx:
            m = d[i]
            if scale != 1.0:
                m = cv2.resize(m.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
            out.append(m)
    return np.stack(out)


class Renderer:
    def __init__(self, mesh, K, W, H, scale=1.0, device="cuda", tex_max=2048):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.W, self.H = int(round(W * scale)), int(round(H * scale))
        Ks = K.copy()
        Ks[:2] *= scale
        self.K = Ks
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.int32)
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        img = mesh.visual.material.baseColorTexture.convert("RGB")
        if max(img.size) > tex_max:
            s = tex_max / max(img.size)
            img = img.resize((int(img.size[0] * s), int(img.size[1] * s)))
        self.tex = torch.tensor(np.asarray(img, dtype=np.float32) / 255.0, device=device)[None].contiguous()
        self.proj = self._proj(Ks, self.W, self.H)

    @staticmethod
    def _proj(K, W, H, n=0.05, f=20.0):
        # OpenCV camera -> clip space (nvdiffrast: y up in NDC, image row 0 at y=-1 => we flip)
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2 * K[0, 0] / W
        P[0, 2] = 2 * (K[0, 2] + 0.5) / W - 1
        P[1, 1] = 2 * K[1, 1] / H
        P[1, 2] = 2 * (K[1, 2] + 0.5) / H - 1
        P[2, 2] = (f + n) / (f - n)
        P[2, 3] = -2 * f * n / (f - n)
        P[3, 2] = 1
        return torch.tensor(P, device="cuda")

    def render(self, T_cam_mesh, need_tex=True):
        """T_cam_mesh: (B,4,4) torch float32 on device. Returns rgb (B,H,W,3), mask (B,H,W), depth (B,H,W)."""
        B = T_cam_mesh.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T_cam_mesh, vh)
        clip = torch.einsum("ij,bnj->bni", self.proj, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (self.H, self.W))
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        depth = depth[..., 0] * mask
        rgb = None
        if need_tex:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex, uvi, filter_mode="linear") * mask[..., None]
        return rgb, mask, depth, rast


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb, axis1=-2, axis2=-1) - 1) / 2 if Ra.ndim == 2 else None
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def rotvec_to_R(rv):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_rotvec(rv).as_matrix()


def R_to_rotvec(R):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_rotvec()


# ---------------------------------------------------------------- pose optimisation
def axis_angle_to_R(w):
    """w (B,3) torch -> (B,3,3) Rodrigues."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


class PoseParam(torch.nn.Module):
    """Pose = (delta rotation about the object centre, image position u,v of the centre, log depth)."""

    def __init__(self, T0, K, center):
        super().__init__()
        T0 = torch.as_tensor(T0, dtype=torch.float32, device="cuda")
        self.R0 = T0[:, :3, :3].clone()
        self.c = torch.as_tensor(center, dtype=torch.float32, device="cuda")
        p = T0[:, :3, :3] @ self.c + T0[:, :3, 3]  # camera coords of object centre
        self.K = torch.as_tensor(K, dtype=torch.float32, device="cuda")
        u = self.K[0, 0] * p[:, 0] / p[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * p[:, 1] / p[:, 2] + self.K[1, 2]
        self.w = torch.nn.Parameter(torch.zeros(T0.shape[0], 3, device="cuda"))
        self.uv = torch.nn.Parameter(torch.stack([u, v], -1))
        self.logz = torch.nn.Parameter(torch.log(p[:, 2]))

    def forward(self):
        R = axis_angle_to_R(self.w) @ self.R0
        z = torch.exp(self.logz)
        x = (self.uv[:, 0] - self.K[0, 2]) / self.K[0, 0] * z
        y = (self.uv[:, 1] - self.K[1, 2]) / self.K[1, 1] * z
        p = torch.stack([x, y, z], -1)
        t = p - R @ self.c
        T = torch.zeros(R.shape[0], 4, 4, device="cuda")
        T[:, :3, :3] = R
        T[:, :3, 3] = t
        T[:, 3, 3] = 1
        return T


def soft_render_mask(R, T, crop=None):
    """crop: None or (offsets (B,2) torch [x0,y0] in renderer pixels, (w,h)). Renders only the crop window."""
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    if crop is None:
        clip = torch.einsum("ij,bnj->bni", R.proj, vc).contiguous()
        H, W = R.H, R.W
    else:
        off, (W, H) = crop
        P = R._proj(R.K, W, H)[None].repeat(T.shape[0], 1, 1)
        P[:, 0, 2] = 2 * (R.K[0, 2] - off[:, 0] + 0.5) / W - 1
        P[:, 1, 2] = 2 * (R.K[1, 2] - off[:, 1] + 0.5) / H - 1
        clip = torch.einsum("bij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (H, W))
    col = (rast[..., 3:4] > 0).float()
    aa = dr.antialias(col, rast, clip, R.f)
    return aa[..., 0], rast, clip


def gauss_blur(x, sigma):
    """x (B,H,W) torch."""
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = x[:, None]
    x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return x[:, 0]


def sil_loss(aa, tgt, valid, sigmas=(4.0, 2.0, 1.0, 0.0)):
    a = aa * valid
    t = tgt * valid
    L = 0
    for s in sigmas:
        L = L + ((gauss_blur(a, s) - gauss_blur(t, s)) ** 2).mean(dim=(1, 2))
    return L  # per item


def fit_poses(R, T0, tgt, valid, center, iters=150, lr_rot=0.01, lr_uv=1.0, lr_z=0.003, shared=False,
              sigmas=(4.0, 2.0, 1.0, 0.0), extra_loss=None, fix_z=False, crop=None):
    """Silhouette fitting. T0 (B,4,4) camera-frame init. tgt, valid (B,H,W) float tensors.
    shared=True: one pose for all B frames (static phase)."""
    B = tgt.shape[0]
    P = PoseParam(T0[:1] if shared else T0, R.K, center)
    groups = [{"params": [P.w], "lr": lr_rot}, {"params": [P.uv], "lr": lr_uv}]
    if not fix_z:
        groups.append({"params": [P.logz], "lr": lr_z})
    opt = torch.optim.Adam(groups)
    for it in range(iters):
        T = P()
        if shared:
            T = T.expand(B, 4, 4)
        aa, _, _ = soft_render_mask(R, T.contiguous(), crop)
        L = sil_loss(aa, tgt, valid, sigmas)
        loss = L.sum()
        if extra_loss is not None:
            loss = loss + extra_loss(P, T)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        T = P()
        aa, _, _ = soft_render_mask(R, (T.expand(B, 4, 4) if shared else T).contiguous(), crop)
        L = sil_loss(aa, tgt, valid, (0.0,))
    return T.detach().cpu().double().numpy(), L.detach().cpu().numpy()


def make_crops(masks_list, centers, w=192, h=256):
    """centers (B,2) pixel coords; returns int offsets (B,2) and cropped arrays for each mask stack."""
    B = len(centers)
    H, W = masks_list[0].shape[1:]
    off = np.zeros((B, 2), dtype=np.int64)
    off[:, 0] = np.clip(np.round(centers[:, 0] - w / 2), 0, W - w)
    off[:, 1] = np.clip(np.round(centers[:, 1] - h / 2), 0, H - h)
    outs = []
    for M in masks_list:
        outs.append(np.stack([M[b, off[b, 1]:off[b, 1] + h, off[b, 0]:off[b, 0] + w] for b in range(B)]))
    return off, outs


def render_crop(R, T, off, W, H, need_tex=True):
    """Hard render (rgb, mask) of crop windows. T (B,4,4) torch, off (B,2) torch float."""
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    P = R._proj(R.K, W, H)[None].repeat(T.shape[0], 1, 1)
    P[:, 0, 2] = 2 * (R.K[0, 2] - off[:, 0] + 0.5) / W - 1
    P[:, 1, 2] = 2 * (R.K[1, 2] - off[:, 1] + 0.5) / H - 1
    clip = torch.einsum("bij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (H, W))
    mask = rast[..., 3] > 0
    rgb = None
    if need_tex:
        uvi, _ = dr.interpolate(R.uv[None], rast, R.f)
        rgb = dr.texture(R.tex, uvi, filter_mode="linear") * mask[..., None]
    return rgb, mask


def ncc_score(rgb, mask, img, valid):
    """Zero-mean NCC of gray-level patterns between render and image over (render mask & valid). Per item."""
    g1 = rgb.mean(-1)
    g2 = img.mean(-1)
    w = (mask.float() * valid)
    n = w.sum((1, 2)).clamp(min=1)
    m1 = (g1 * w).sum((1, 2)) / n
    m2 = (g2 * w).sum((1, 2)) / n
    d1 = (g1 - m1[:, None, None]) * w
    d2 = (g2 - m2[:, None, None]) * w
    return (d1 * d2).sum((1, 2)) / torch.sqrt((d1 ** 2).sum((1, 2)) * (d2 ** 2).sum((1, 2)) + 1e-9)


def rot_log(Rm):
    """(B,3,3) torch -> (B,3) axis-angle (small/medium angles)."""
    c = ((Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2] - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    th = torch.acos(c)
    v = torch.stack([Rm[:, 2, 1] - Rm[:, 1, 2], Rm[:, 0, 2] - Rm[:, 2, 0], Rm[:, 1, 0] - Rm[:, 0, 1]], -1)
    s = torch.sin(th).clamp(min=1e-6)
    return v * (th / (2 * s))[:, None]


def fit_track(R, T0, tgt, valid, center, crop, T_before=None, T_after=None, iters=300, lam_p=30.0, lam_r=2.0,
              lr_rot=0.004, lr_uv=0.5, lr_z=0.002, sigmas=(4.0, 2.0, 1.0, 0.0), frame_w=None, chunk=None):
    """Joint silhouette fit of a whole moving segment with 2nd-order smoothness on camera-frame object-centre
    position (metres) and rotation (rad). T_before/T_after: fixed poses adjacent to the segment (static phases)."""
    B = T0.shape[0]
    P = PoseParam(T0, R.K, center)
    opt = torch.optim.Adam([{"params": [P.w], "lr": lr_rot}, {"params": [P.uv], "lr": lr_uv},
                            {"params": [P.logz], "lr": lr_z}])
    cdev = torch.as_tensor(center, dtype=torch.float32, device="cuda")
    fixed = [torch.as_tensor(x, dtype=torch.float32, device="cuda")[None] if x is not None else None
             for x in (T_before, T_after)]
    fw = torch.ones(B, device="cuda") if frame_w is None else torch.as_tensor(frame_w, dtype=torch.float32, device="cuda")
    off, wh = crop
    chunk = chunk or B
    for it in range(iters):
        T = P()
        Ls = []
        for s in range(0, B, chunk):
            aa, _, _ = soft_render_mask(R, T[s:s + chunk].contiguous(), (off[s:s + chunk], wh))
            Ls.append(sil_loss(aa, tgt[s:s + chunk], valid[s:s + chunk], sigmas))
        L = (torch.cat(Ls) * fw).sum()
        TT = T
        if fixed[0] is not None:
            TT = torch.cat([fixed[0], TT])
        if fixed[1] is not None:
            TT = torch.cat([TT, fixed[1]])
        pc = TT[:, :3, :3] @ cdev + TT[:, :3, 3]
        acc = pc[2:] - 2 * pc[1:-1] + pc[:-2]
        Rm = TT[:, :3, :3]
        dr_ = rot_log(Rm[1:] @ Rm[:-1].transpose(1, 2))
        racc = dr_[1:] - dr_[:-1]
        loss = L + lam_p * (acc ** 2).sum() + lam_r * (racc ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        T = P()
        Ls = []
        for s in range(0, B, chunk):
            aa, _, _ = soft_render_mask(R, T[s:s + chunk].contiguous(), (off[s:s + chunk], wh))
            Ls.append(sil_loss(aa, tgt[s:s + chunk], valid[s:s + chunk], (0.0,)))
    return T.detach().cpu().double().numpy(), torch.cat(Ls).cpu().numpy()
