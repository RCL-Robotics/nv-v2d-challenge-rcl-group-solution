"""Shared helpers: input loading, textured-mesh rendering with nvdiffrast, pose utilities."""
import json
import numpy as np
import torch
import trimesh
import h5py
import cv2

H, W = 1152, 1536


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], dtype=np.float64)
    Kp = np.array(cam["K_pipeline"], dtype=np.float64)
    Twc = np.array(cam["T_world_cam"], dtype=np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    gp = json.load(open(f"{inp}/ground_plane.json"))
    return dict(K=K, Kp=Kp, Twc=Twc, rec=rec, plane=np.array(gp["plane"]))


def load_mesh(inp):
    m = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
    return m


def read_video(path, frames=None):
    cap = cv2.VideoCapture(path)
    out = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if frames is None or i in frames:
            out.append(f)
        i += 1
    return out


def proj(K, Xc):
    """Project camera-frame points (...,3) -> pixels (...,2)."""
    x = Xc[..., :2] / Xc[..., 2:3]
    return np.stack([K[0, 0] * x[..., 0] + K[0, 2], K[1, 1] * x[..., 1] + K[1, 2]], -1)


class Renderer:
    """Textured mesh renderer (OpenCV camera) on a crop window."""

    def __init__(self, mesh, device="cuda"):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext()
        self.dev = device
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        vis = mesh.visual
        uv = np.asarray(vis.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(vis.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.vn = torch.tensor(np.asarray(mesh.vertex_normals), dtype=torch.float32, device=device)

    def render(self, T_cam_mesh, K, box, res):
        """T_cam_mesh: (B,4,4) torch or np; K (3,3) np; box=(x0,y0,w,h) full-image pixel window;
        res=(rh,rw) output size. Returns rgb (B,rh,rw,3) [0,1], mask (B,rh,rw), depth (B,rh,rw)."""
        dev = self.dev
        T = torch.as_tensor(T_cam_mesh, dtype=torch.float32, device=dev)
        if T.dim() == 2:
            T = T[None]
        B = T.shape[0]
        x0, y0, bw, bh = box
        rh, rw = res
        sx, sy = rw / bw, rh / bh
        fx, fy = K[0, 0] * sx, K[1, 1] * sy
        cx, cy = (K[0, 2] - x0) * sx, (K[1, 2] - y0) * sy
        n, fa = 0.05, 20.0
        P = torch.tensor([[2 * fx / rw, 0, 2 * (cx + 0.5) / rw - 1, 0],
                          [0, 2 * fy / rh, 2 * (cy + 0.5) / rh - 1, 0],
                          [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # camera frame
        clip = torch.einsum("ij,bnj->bni", P, vc)
        # nvdiffrast: y up in clip -> flip so row 0 = top
        clip = clip * torch.tensor([1, 1, 1, 1], device=dev, dtype=torch.float32)
        rast, _ = self.dr.rasterize(self.glctx, clip.contiguous(), self.f, (rh, rw))
        uvi, _ = self.dr.interpolate(self.uv[None], rast, self.f)
        rgb = self.dr.texture(self.tex, uvi, filter_mode="linear")
        mask = (rast[..., 3] > 0).float()
        dep, _ = self.dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        rgb = rgb * mask[..., None]
        return rgb, mask, dep[..., 0] * mask
