"""Shared helpers: data loading, nvdiffrast rendering of the textured object mesh, SE(3) utils."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
import cv2
import h5py


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], np.float64)
    Twc = np.array(cam["T_world_cam"], np.float64)
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    return dict(K=K, Twc=Twc, Kp=np.array(cam["K_pipeline"], np.float64), plane=plane,
                W=cam["width"], H=cam["height"], rec=rec)


def load_mesh(inp):
    g = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255.0
    return dict(v=np.asarray(g.vertices, np.float64), f=np.asarray(g.faces, np.int64),
                uv=np.asarray(g.visual.uv, np.float64), tex=tex, mesh=g)


def read_video(path, scale=1.0):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if scale != 1.0:
            fr = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(fr[:, :, ::-1].copy())  # RGB
    return np.stack(frames)


def read_masks(inp, scale=1.0, key="object"):
    with h5py.File(f"{inp}/masks.h5", "r") as f:
        d = f[key]
        out = []
        for i in range(0, d.shape[0], 64):
            m = d[i:i + 64]
            if scale != 1.0:
                m = np.stack([cv2.resize(x.astype(np.uint8), None, fx=scale, fy=scale,
                                         interpolation=cv2.INTER_AREA) for x in m]) > 0
            out.append(m.astype(bool))
    return np.concatenate(out)


class Renderer:
    """Batched nvdiffrast renderer of the textured mesh with OpenCV pinhole intrinsics."""

    def __init__(self, mesh, K, H, W, device="cuda", near=0.05, far=20.0):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(mesh["v"], dtype=torch.float32, device=device)
        self.f = torch.tensor(mesh["f"], dtype=torch.int32, device=device)
        uv = mesh["uv"].copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, dtype=torch.float32, device=device)
        self.tex = torch.tensor(mesh["tex"], dtype=torch.float32, device=device)[None]
        self.set_K(K, H, W)
        self.near, self.far = near, far

    def set_K(self, K, H, W):
        self.K = np.asarray(K, np.float64)
        self.H, self.W = int(H), int(W)

    def proj(self, n, f):
        K, W, H = self.K, self.W, self.H
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # clip.x/w = (2u+1)/W - 1 with u = fx x/z + cx; same for y (row 0 <-> v=0)
        P = np.array([[2 * fx / W, 0, (2 * cx + 1) / W - 1, 0],
                      [0, 2 * fy / H, (2 * cy + 1) / H - 1, 0],
                      [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                      [0, 0, 1, 0]], np.float64)
        return torch.tensor(P, dtype=torch.float32, device=self.dev)

    def render(self, T_cam_mesh, rgb=True, depth=False):
        """T_cam_mesh: (B,4,4) numpy or torch. Returns dict mask (B,H,W), rgb (B,H,W,3), depth."""
        T = torch.as_tensor(np.asarray(T_cam_mesh, np.float32) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,
                            device=self.dev, dtype=torch.float32)
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # camera coords
        P = self.proj(self.near, self.far)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (self.H, self.W))
        out = {"mask": rast[..., 3] > 0}
        if rgb:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            col = dr.texture(self.tex, uvi, filter_mode="linear")
            out["rgb"] = col * out["mask"][..., None]
        if depth:
            z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
            out["depth"] = z[..., 0] * out["mask"]
        return out


# ---------------- SE(3) / SO(3) utils ----------------
def rot_angle_deg(Ra, Rb):
    R = Ra.swapaxes(-1, -2) @ Rb
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def axis_angle(axis, ang):
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis)
    return cv2.Rodrigues((axis * ang).reshape(3, 1))[0]
