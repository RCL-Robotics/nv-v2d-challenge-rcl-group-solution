"""Shared utilities: data loading, nvdiffrast rendering of the textured object mesh, SE(3) helpers."""
import json
import os

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot


def load_inputs(inp):
    cam = json.load(open(os.path.join(inp, "camera.json")))
    K = np.array(cam["K"], np.float64)
    Twc = np.array(cam["T_world_cam"], np.float64)
    rec = np.load(os.path.join(inp, "reconstruction.npz"))
    rec = {k: rec[k] for k in rec.files}
    plane = np.array(json.load(open(os.path.join(inp, "ground_plane.json")))["plane"], np.float64)
    return cam, K, Twc, rec, plane


def load_mesh(inp):
    m = trimesh.load(os.path.join(inp, "object_mesh.glb"), force="scene", process=False).to_geometry()
    return m


def read_video(path, idx=None, scale=1.0):
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    want = None if idx is None else set(int(x) for x in idx)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if want is None or i in want:
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(f[:, :, ::-1].copy())  # RGB
        i += 1
    return frames


def rot_angle_deg(Ra, Rb):
    return np.degrees(Rot.from_matrix(Ra.T @ Rb).magnitude())


class Renderer:
    """Renders the textured mesh with an OpenCV pinhole camera (K) at a given resolution."""

    def __init__(self, mesh, K, W, H, device="cuda", max_faces=None):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        v = np.asarray(mesh.vertices, np.float32)
        f = np.asarray(mesh.faces, np.int32)
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        uv = np.asarray(mesh.visual.uv, np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = mesh.visual.material.baseColorTexture
        tex = np.asarray(tex.convert("RGB").resize((4096, 4096)), np.float32) / 255.0  # pow2 for mipmaps
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.K = np.asarray(K, np.float64)
        self.W, self.H = W, H

    def proj(self, near=0.05, far=20.0):
        K, W, H = self.K, self.W, self.H
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip space (nvdiffrast: y up in NDC, image row 0 = y=-1 so we flip)
        P = np.array([
            [2 * fx / W, 0, (2 * cx - W) / W + 0.0, 0],
            [0, 2 * fy / H, (2 * cy - H) / H, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], np.float64)
        # pixel centres: x_ndc = (2*(u+0.5)/W - 1); u = fx*X/Z + cx  -> use cx-0.5 shift
        P[0, 2] = (2 * (cx + 0.5) - W) / W
        P[1, 2] = (2 * (cy + 0.5) - H) / H
        return torch.tensor(P, dtype=torch.float32, device=self.dev)

    def render(self, T_cam_mesh, need_tex=True, need_depth=True):
        """T_cam_mesh: (B,4,4) torch or numpy. Returns dict with mask (B,H,W), rgb (B,H,W,3), depth (B,H,W)."""
        if not torch.is_tensor(T_cam_mesh):
            T_cam_mesh = torch.tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        T = T_cam_mesh.float()
        if T.dim() == 2:
            T = T[None]
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # camera coords (B,N,4)
        P = self.proj()
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, resolution=[self.H, self.W])
        out = {"rast": rast, "mask": rast[..., 3] > 0}
        if need_depth:
            z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
            out["depth"] = z[..., 0] * out["mask"]
        if need_tex:
            uvi, _ = dr.interpolate(self.uv[None].expand(B, -1, -1).contiguous(), rast, self.f)
            col = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            out["rgb"] = col * out["mask"][..., None]
        return out


def proj_batch(Kc, W, H, near=0.05, far=20.0):
    """Kc: (B,3,3) torch -> (B,4,4) OpenCV->clip projection matrices for an image of size W x H."""
    B = Kc.shape[0]
    P = torch.zeros(B, 4, 4, device=Kc.device, dtype=torch.float32)
    P[:, 0, 0] = 2 * Kc[:, 0, 0] / W
    P[:, 0, 2] = (2 * (Kc[:, 0, 2] + 0.5) - W) / W
    P[:, 1, 1] = 2 * Kc[:, 1, 1] / H
    P[:, 1, 2] = (2 * (Kc[:, 1, 2] + 0.5) - H) / H
    P[:, 2, 2] = (far + near) / (far - near)
    P[:, 2, 3] = -2 * far * near / (far - near)
    P[:, 3, 2] = 1
    return P


def render_crops(R, T, Kc, res, tex=True, aa=True):
    """Differentiable batch render. R: Renderer (mesh buffers); T: (B,4,4) cam-from-mesh torch; Kc: (B,3,3) crop
    intrinsics; res: square crop size. Returns alpha (B,res,res), rgb (B,res,res,3) or None, depth (B,res,res)."""
    B = T.shape[0]
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("bij,bnj->bni", proj_batch(Kc, res, res), vc).contiguous()
    rast, rast_db = dr.rasterize(R.glctx, clip, R.f, resolution=[res, res], grad_db=tex)
    mask = (rast[..., 3:4] > 0).float()
    alpha = dr.antialias(mask, rast, clip, R.f) if aa else mask
    z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, R.f)
    rgb = None
    if tex:
        uvi, uvda = dr.interpolate(R.uv[None].expand(B, -1, -1).contiguous(), rast, R.f, rast_db=rast_db, diff_attrs="all")
        rgb = dr.texture(R.tex, uvi, uvda, filter_mode="linear-mipmap-linear", max_mip_level=8) * mask
        if aa:
            rgb = dr.antialias(rgb.contiguous(), rast, clip, R.f)
    return alpha[..., 0], rgb, (z[..., 0] * mask[..., 0])


def axis_angle_torch(axis, ang):
    """Rotation matrices (B,3,3) for rotation by ang (B,) about fixed unit axis (3,)."""
    k = axis
    Kx = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=torch.float32, device=ang.device)
    I = torch.eye(3, device=ang.device)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


class FloorModel:
    """Object resting on the floor on its convex-hull support face: world pose determined by yaw theta about gravity
    and 2D position (a,b) on the floor. h0 = height of the support face above the released plane."""

    def __init__(self, mesh, plane, down_hint, h0=0.0):
        v = np.asarray(mesh.vertices, np.float64)
        hull = trimesh.convex.convex_hull(v)
        fn = hull.face_normals
        cand = np.where(fn @ down_hint > np.cos(np.radians(5)))[0]
        f = cand[np.argmax(hull.area_faces[cand])]
        self.dm = fn[f]                      # mesh-frame down direction (support face outward normal)
        self.s = float((v @ self.dm).max())  # support offset along dm
        self.n, self.d = plane[:3] / np.linalg.norm(plane[:3]), plane[3] / np.linalg.norm(plane[:3])
        self.g = -self.n                     # world down
        ax = np.cross(self.dm, self.g); sn = np.linalg.norm(ax)
        self.B = Rot.from_rotvec(ax / sn * np.arctan2(sn, self.dm @ self.g)).as_matrix()  # B @ dm = g
        e1 = np.array([1.0, 0, 0]); e1 -= (e1 @ self.n) * self.n; e1 /= np.linalg.norm(e1)
        self.e1, self.e2 = e1, np.cross(self.n, e1)
        self.h0 = h0
        self.c = 0.5 * (v.min(0) + v.max(0))

    def t0(self, h0=None):
        h0 = self.h0 if h0 is None else h0
        return self.n * (h0 + self.s - self.d)   # n.t = h0 + s - d

    def pose(self, theta, a, b, h0=None):
        R = Rot.from_rotvec(self.g * theta).as_matrix() @ self.B
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = self.t0(h0) + a * self.e1 + b * self.e2
        return T

    def pose_torch(self, theta, a, b, h0=None):
        dev = theta.device
        R = axis_angle_torch(torch.tensor(self.g, dtype=torch.float32), theta) @ torch.tensor(self.B, dtype=torch.float32, device=dev)
        t = torch.tensor(self.t0(h0), dtype=torch.float32, device=dev) + a[:, None] * torch.tensor(self.e1, dtype=torch.float32, device=dev) \
            + b[:, None] * torch.tensor(self.e2, dtype=torch.float32, device=dev)
        T = torch.zeros(theta.shape[0], 4, 4, device=dev); T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
        return T

    def from_pose(self, T, cam_o, h0=None):
        """Project a world pose onto the floor model: remove tilt (min. rotation), keep the image position of the
        bbox centre by moving along its viewing ray. Returns theta, a, b."""
        Rb = T[:3, :3]
        ax = np.cross(Rb @ self.dm, self.g); sn = np.linalg.norm(ax)
        Rc = np.eye(3) if sn < 1e-12 else Rot.from_rotvec(ax / sn * np.arctan2(sn, (Rb @ self.dm) @ self.g)).as_matrix()
        Rn = Rc @ Rb
        Q = Rn @ self.B.T   # rotation about g
        theta = Rot.from_matrix(Q).as_rotvec() @ self.g
        pc = Rb @ self.c + T[:3, 3]; r = pc - cam_o; r /= np.linalg.norm(r)
        h0 = self.h0 if h0 is None else h0
        lam = (h0 + self.s - self.d - self.n @ cam_o + self.n @ (Rn @ self.c)) / (self.n @ r)
        t = cam_o + lam * r - Rn @ self.c - self.t0(h0)
        return float(theta), float(t @ self.e1), float(t @ self.e2)
