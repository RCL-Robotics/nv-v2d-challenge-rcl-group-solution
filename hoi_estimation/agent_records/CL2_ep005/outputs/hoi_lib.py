"""Shared utilities for solve.py: loading inputs, nvdiffrast rendering of the textured mesh, SE(3) helpers."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], dtype=np.float64)
    T_wc = np.array(cam["T_world_cam"], dtype=np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], dtype=np.float64)
    return dict(K=K, K_pipe=np.array(cam["K_pipeline"], dtype=np.float64), T_wc=T_wc,
                W=cam["width"], H=cam["height"], rec=rec, plane=plane)


def load_mesh(inp):
    return trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()


class Renderer:
    """Renders textured mesh (RGB, mask, depth) for a batch of camera-frame poses at a given resolution."""

    def __init__(self, mesh, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()

    def render(self, T_cam_mesh, K, H, W, scale=1.0, need_rgb=True):
        """T_cam_mesh: (B,4,4) torch/np; K for full-res; returns dict of (B,h,w,...) tensors at res scale."""
        T = torch.as_tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        if T.ndim == 2:
            T = T[None]
        h, w = int(round(H * scale)), int(round(W * scale))
        fx, fy, cx, cy = K[0, 0] * scale, K[1, 1] * scale, (K[0, 2] + 0.5) * scale - 0.5, (K[1, 2] + 0.5) * scale - 0.5
        n, fr = 0.05, 50.0
        # OpenCV camera -> clip; nvdiffrast row 0 is ndc y=-1, so map pixel row v to ndc y = 2(v+0.5)/h-1
        P = torch.tensor([[2 * fx / w, 0, 2 * (cx + 0.5) / w - 1, 0],
                          [0, 2 * fy / h, 2 * (cy + 0.5) / h - 1, 0],
                          [0, 0, (fr + n) / (fr - n), -2 * fr * n / (fr - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (h, w))
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        out = dict(mask=mask, depth=depth[..., 0] * mask)
        if need_rgb:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex.expand(T.shape[0], -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            out["rgb"] = rgb * mask[..., None]
        return out


def cam_from_world(T_wc, T_wm):
    return np.linalg.inv(T_wc)[None] @ T_wm


def rot_angle(Ra, Rb):
    R = np.swapaxes(Ra, -1, -2) @ Rb
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def so3_exp(w):
    """w: (B,3) torch -> (B,3,3)"""
    th = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def render_sil_soft(R, T, K, H, W, scale):
    """Differentiable antialiased silhouette for torch poses T (B,4,4)."""
    h, w = int(round(H * scale)), int(round(W * scale))
    fx, fy = K[0, 0] * scale, K[1, 1] * scale
    cx, cy = (K[0, 2] + 0.5) * scale - 0.5, (K[1, 2] + 0.5) * scale - 0.5
    n, fr = 0.05, 50.0
    P = torch.tensor([[2 * fx / w, 0, 2 * (cx + 0.5) / w - 1, 0],
                      [0, 2 * fy / h, 2 * (cy + 0.5) / h - 1, 0],
                      [0, 0, (fr + n) / (fr - n), -2 * fr * n / (fr - n)],
                      [0, 0, 1, 0]], dtype=torch.float32, device=R.dev)
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("ij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (h, w))
    sil = (rast[..., 3:4] > 0).float()
    sil = dr.antialias(sil, rast, clip, R.f)
    return sil[..., 0], rast, vc


class SilMesh:
    """Decimated untextured mesh (same mesh-file frame) for fast silhouette rendering."""

    def __init__(self, mesh, n_faces=8000, device="cuda", glctx=None):
        import open3d as o3d
        mm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mm.vertices), o3d.utility.Vector3iVector(mm.faces))
        s = om.simplify_quadric_decimation(n_faces)
        self.dev = device
        self.glctx = glctx if glctx is not None else dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(s.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(s.triangles), dtype=torch.int32, device=device)
