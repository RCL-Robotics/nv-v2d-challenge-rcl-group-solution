"""Shared helpers: data loading, nvdiffrast textured rendering, pose utilities."""
import json
import os

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr


def load_inputs(inp):
    cam = json.load(open(os.path.join(inp, "camera.json")))
    K = np.array(cam["K"], np.float64)
    Twc = np.array(cam["T_world_cam"], np.float64)
    rec = np.load(os.path.join(inp, "reconstruction.npz"))
    return cam, K, Twc, rec


def load_video(inp, scale=0.5):
    cap = cv2.VideoCapture(os.path.join(inp, "video.mp4"))
    fr = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if scale != 1:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        fr.append(im)
    return np.stack(fr)


def load_masks(inp, scale=0.5, idx=None):
    f = h5py.File(os.path.join(inp, "masks.h5"), "r")
    out = {}
    for k in ("object", "person"):
        d = f[k]
        n = d.shape[0]
        ids = range(n) if idx is None else idx
        ms = []
        for i in ids:
            m = d[i].astype(np.uint8)
            if scale != 1:
                m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ms.append(m > 0)
        out[k] = np.stack(ms)
    return out


class Renderer:
    """Textured mesh renderer (OpenCV camera) with nvdiffrast."""

    def __init__(self, mesh_path, K, H, W, device="cuda"):
        m = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
        self.mesh = m
        self.dev = device
        self.H, self.W = H, W
        self.K = np.asarray(K, np.float64)
        self.v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=device)
        uv = np.asarray(m.visual.uv, np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255.0
        tex = cv2.resize(tex, (1024, 1024), interpolation=cv2.INTER_AREA)
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext(device=device)

    def proj(self, near=0.05, far=20.0):
        K, H, W = self.K, self.H, self.W
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip space (y down in image: flip handled by sign)
        P = np.array([
            [2 * fx / W, 0, (2 * cx - W) / W + 1.0 / W * 0, 0],
            [0, 2 * fy / H, (2 * cy - H) / H, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], np.float64)
        # pixel centre convention: pixel i covers [i, i+1); centre at i+0.5
        P[0, 2] = (2 * (cx + 0.5) - W) / W
        P[1, 2] = (2 * (cy + 0.5) - H) / H
        return torch.tensor(P, dtype=torch.float32, device=self.dev)

    def render(self, T_cam_mesh, need_tex=True, need_depth=False):
        """T_cam_mesh: (B,4,4) torch or numpy. returns dict of (B,H,W,...) tensors."""
        if not torch.is_tensor(T_cam_mesh):
            T_cam_mesh = torch.tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        T = T_cam_mesh.float()
        if T.dim() == 2:
            T = T[None]
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # (B,N,4)
        P = self.proj()
        clip = torch.einsum("ij,bnj->bni", P, vc)
        # nvdiffrast clip y points up; our image row increases downward -> raster then flip rows
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (self.H, self.W))
        out = {}
        mask = rast[..., 3] > 0
        out["mask"] = mask
        if need_tex:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            col = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous() if B > 1 else self.tex, uvi, filter_mode="linear")
            col = col * mask[..., None]
            out["color"] = col
        if need_depth:
            z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
            out["depth"] = z[..., 0] * mask
        out["rast"] = rast
        return out


def rot_angle(Ra, Rb):
    Rr = Ra @ np.swapaxes(Rb, -1, -2)
    c = (np.trace(Rr, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
