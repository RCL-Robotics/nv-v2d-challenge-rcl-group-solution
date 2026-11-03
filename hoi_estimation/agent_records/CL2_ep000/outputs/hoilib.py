"""Shared helpers: data loading, nvdiffrast rendering of the textured mesh, pose utilities."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot

W, H = 1536, 1152


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], float)
    Twc = np.array(cam["T_world_cam"], float)
    rec = dict(np.load(f"{inp}/reconstruction.npz"))
    mesh = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], float)
    return K, Twc, rec, mesh, plane


def read_video(path, idx=None, scale=1.0):
    import decord
    vr = decord.VideoReader(path)
    if idx is None:
        idx = list(range(len(vr)))
    out = []
    for i0 in range(0, len(idx), 64):
        b = vr.get_batch(idx[i0:i0 + 64]).asnumpy()
        if scale != 1.0:
            import cv2
            b = np.stack([cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for x in b])
        out.append(b)
    return np.concatenate(out)


class Renderer:
    """Renders the textured mesh (mask, depth, rgb) for camera-frame poses at a given image scale."""

    def __init__(self, mesh, K, scale=0.5, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        vis = mesh.visual
        self.uv = torch.tensor(np.asarray(vis.uv), dtype=torch.float32, device=device)
        tex = np.asarray(vis.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex[::-1].copy(), device=device)[None]  # flip v (uv origin bottom-left)
        self.set_scale(K, scale)

    def set_scale(self, K, scale):
        self.scale = scale
        self.w, self.h = int(round(W * scale)), int(round(H * scale))
        Ks = K.copy()
        Ks[:2] *= scale
        self.K = Ks

    def proj_matrix(self, n=0.05, f=20.0):
        K, w, h = self.K, self.w, self.h
        P = np.array([
            [2 * K[0, 0] / w, 0, (2 * K[0, 2] - w) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h) / h, 0],
            [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
            [0, 0, 1, 0]], np.float32)
        # pixel centre convention: pixel (u,v) centre at u+0.5 handled approximately
        return torch.tensor(P, device=self.dev)

    def render(self, T_cam_mesh, need_rgb=True):
        """T_cam_mesh: (B,4,4) tensor/np. Returns mask (B,h,w), depth (B,h,w), rgb (B,h,w,3). Image row 0 = top."""
        T = torch.as_tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # camera coords
        P = self.proj_matrix()
        clip = torch.einsum("ij,bnj->bni", P, vc)
        # OpenCV pixel v maps to ndc y=(2v-h)/h; nvdiffrast row i <-> ndc y=-1..1, so row 0 = image top
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (self.h, self.w))
        mask = rast[..., 3] > 0
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        z = z[..., 0] * mask
        rgb = None
        if need_rgb:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            rgb = rgb * mask[..., None]
        return mask, z, rgb


def rotdeg(Ra, Rb):
    Ra = np.asarray(Ra)
    Rb = np.asarray(Rb)
    d = np.einsum("...ji,...jk->...ik", Ra, Rb)
    return np.degrees(np.linalg.norm(Rot.from_matrix(d.reshape(-1, 3, 3)).as_rotvec(), axis=1)).reshape(d.shape[:-2])


def add_err(Ta, Tb, pts):
    pa = np.einsum("...ij,nj->...ni", Ta[..., :3, :3], pts) + Ta[..., None, :3, 3]
    pb = np.einsum("...ij,nj->...ni", Tb[..., :3, :3], pts) + Tb[..., None, :3, 3]
    return np.linalg.norm(pa - pb, axis=-1).mean(-1)


def add_score(err):
    th = np.arange(1, 10) / 100.0
    return np.mean([(err < t).mean() for t in th])


def so3_exp(w):
    """axis-angle (B,3) -> rotation (B,3,3), torch, differentiable."""
    th = torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def render_sil(rd, T):
    """Differentiable antialiased silhouette for camera-frame poses T (B,4,4) torch."""
    B = T.shape[0]
    vh = torch.cat([rd.v, torch.ones_like(rd.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("ij,bnj->bni", rd.proj_matrix(), vc).contiguous()
    rast, _ = dr.rasterize(rd.glctx, clip, rd.f, (rd.h, rd.w))
    sil = (rast[..., 3:] > 0).float()
    sil = dr.antialias(sil, rast, clip, rd.f)[..., 0]
    return sil, rast, vc


def mean_rotation(Rs, w=None):
    """Chordal L2 mean of rotations (N,3,3)."""
    w = np.ones(len(Rs)) if w is None else np.asarray(w)
    M = np.einsum("n,nij->ij", w, Rs)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def robust_mean_pose(Ts, iters=3):
    """Robust (IRLS) mean of camera-frame poses: chordal rotation mean + median translation."""
    Rm = mean_rotation(Ts[:, :3, :3])
    for _ in range(iters):
        e = rotdeg(Rm[None], Ts[:, :3, :3])
        w = 1.0 / np.maximum(e, 0.5)
        Rm = mean_rotation(Ts[:, :3, :3], w)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = np.median(Ts[:, :3, 3], 0)
    return T


def rescale_depth(Tc, z_target):
    """Move a camera-frame pose along its viewing ray (through the mesh origin) so that z == z_target."""
    T = Tc.copy()
    T[..., :3, 3] = Tc[..., :3, 3] * (np.asarray(z_target) / Tc[..., 2, 3])[..., None]
    return T
