"""Shared helpers: mesh loading and nvdiffrast rendering (OpenCV camera convention)."""
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

W, H = 1536, 1152


def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
    return m, tex


class Renderer:
    def __init__(self, mesh, tex, K, W=W, H=H, device="cuda", scale=1.0):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        self.f = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.set_K(K, W, H, scale)
        # smooth vertex normals (the file mesh has split vertices; merge by position)
        _, inv = np.unique(np.round(mesh.vertices, 6), axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        fn = np.asarray(mesh.face_normals) * np.asarray(mesh.area_faces)[:, None]
        acc = np.zeros((inv.max() + 1, 3))
        for k in range(3):
            np.add.at(acc, inv[mesh.faces[:, k]], fn)
        acc /= np.linalg.norm(acc, axis=1, keepdims=True).clip(1e-12)
        self.n = torch.tensor(acc[inv], dtype=torch.float32, device=device)

    def set_K(self, K, W, H, scale=1.0):
        K = np.array(K, dtype=np.float64).copy()
        K[:2] *= scale
        self.K = K
        self.W, self.H = int(round(W * scale)), int(round(H * scale))

    def proj(self, near=0.05, far=20.0):
        K, W, H = self.K, self.W, self.H
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip; y flipped so that row 0 is image top
        P = np.array([
            [2 * fx / W, 0, (2 * cx - W) / W, 0],
            [0, 2 * fy / H, (2 * cy - H) / H, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], dtype=np.float32)
        return torch.tensor(P, device=self.dev)

    def render(self, T_cam_mesh, need_rgb=True):
        """T_cam_mesh: (B,4,4) torch or np. Returns rgb (B,H,W,3), mask (B,H,W), depth (B,H,W)."""
        if not torch.is_tensor(T_cam_mesh):
            T_cam_mesh = torch.tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        T = T_cam_mesh.float()
        if T.dim() == 2:
            T = T[None]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T, vh)  # (B,N,4)
        P = self.proj()
        clip = torch.einsum('ij,bnj->bni', P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (self.H, self.W))
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        depth = depth[..., 0] * mask
        rgb = None
        if need_rgb:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex.expand(T.shape[0], -1, -1, -1).contiguous(), uvi, filter_mode='linear')
            rgb = rgb * mask[..., None]
        return rgb, mask, depth


def crop_K(K, box, S):
    """Intrinsics of a crop box=(x0,y0,size) resampled to SxS pixels."""
    x0, y0, s = box
    Kc = np.array(K, dtype=np.float64).copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    Kc[:2] *= S / s
    return Kc


def proj_batch(Ks, S, device, near=0.05, far=20.0):
    P = np.zeros((len(Ks), 4, 4), np.float32)
    for b, K in enumerate(Ks):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        P[b] = [[2 * fx / S, 0, (2 * cx - S) / S, 0],
                [0, 2 * fy / S, (2 * cy - S) / S, 0],
                [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, 1, 0]]
    return torch.tensor(P, device=device)


def render_crops(R, T, P, S, antialias=True, need_rgb=False, inflate=None):
    """Differentiable batched render. T: (B,4,4) torch cam<-mesh; P: (B,4,4) crop projections.
    inflate: optional (B,) metres to push vertices along their normals (models fat masks).
    Returns soft silhouette (B,S,S), depth (B,S,S), rgb or None."""
    if inflate is None:
        vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T, vh)
    else:
        v = R.v[None] + inflate.float().view(-1, 1, 1) * R.n[None]
        vc = torch.einsum('bij,bnj->bni', T[:, :3, :3], v) + T[:, None, :3, 3]
        vc = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
    clip = torch.einsum('bij,bnj->bni', P, vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (S, S))
    sil = (rast[..., 3:4] > 0).float()
    if antialias:
        sil = dr.antialias(sil, rast, clip, R.f)
    depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, R.f)
    rgb = None
    if need_rgb:
        uvi, _ = dr.interpolate(R.uv[None], rast, R.f)
        rgb = dr.texture(R.tex.expand(T.shape[0], -1, -1, -1).contiguous(), uvi, filter_mode='linear')
        rgb = rgb * (rast[..., 3:4] > 0)
    return sil[..., 0], depth[..., 0], rgb
