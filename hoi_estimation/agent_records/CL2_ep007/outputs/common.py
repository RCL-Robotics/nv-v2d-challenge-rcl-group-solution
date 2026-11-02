"""Shared helpers: data loading, nvdiffrast rendering of the textured mesh, SE(3) utilities."""
import json
import numpy as np
import torch
import trimesh
import cv2
import nvdiffrast.torch as dr

H, W = 1152, 1536


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], np.float64)
    Kp = np.array(cam["K_pipeline"], np.float64)
    Twc = np.array(cam["T_world_cam"], np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    return K, Kp, Twc, rec


def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    return m


def read_video(path, every=1, scale=1.0):
    cap = cv2.VideoCapture(path)
    out = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            out.append(f)
        i += 1
    return out


class Renderer:
    """Renders the textured mesh with a pinhole K into an image ROI."""

    def __init__(self, mesh, device="cuda", max_faces=None):
        self.dev = device
        v = np.asarray(mesh.vertices, np.float32)
        f = np.asarray(mesh.faces, np.int32)
        uv = np.asarray(mesh.visual.uv, np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        img = mesh.visual.material.baseColorTexture
        tex = np.asarray(img.convert("RGB"), np.float32) / 255.0
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        self.uv = torch.tensor(uv, device=device)
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.verts_np = v

    def render(self, T_cam_mesh, K, roi, out_hw, need_rgb=True):
        """T_cam_mesh: (B,4,4) numpy. roi=(x0,y0,x1,y1) in image pixels, rendered to out_hw=(h,w).
        Returns rgb (B,h,w,3) [0,1] float (RGB order), mask (B,h,w), depth (B,h,w) camera z (0 outside)."""
        T = torch.tensor(np.asarray(T_cam_mesh, np.float32), device=self.dev)
        B = T.shape[0]
        x0, y0, x1, y1 = roi
        h, w = out_hw
        # pixel mapping: u_out = (u - x0) * w/(x1-x0)
        sx = w / (x1 - x0)
        sy = h / (y1 - y0)
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] - x0) * sx, (K[1, 2] - y0) * sy
        n, far = 0.05, 20.0
        # OpenCV camera -> clip. y down in image: nvdiffrast's row 0 is at clip y=-1.
        P = torch.tensor([
            [2 * fx / w, 0, 2 * cx / w - 1, 0],
            [0, 2 * fy / h, 2 * cy / h - 1, 0],
            [0, 0, (far + n) / (far - n), -2 * far * n / (far - n)],
            [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # camera coords
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, resolution=[h, w])
        mask = rast[..., 3] > 0
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        depth = torch.where(mask, z[..., 0], torch.zeros_like(z[..., 0]))
        rgb = None
        if need_rgb:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            rgb = torch.where(mask[..., None], rgb, torch.zeros_like(rgb))
        return rgb, mask, depth


def rotvec_to_R(rv):
    return cv2.Rodrigues(np.asarray(rv, np.float64).reshape(3, 1))[0]


def R_to_rotvec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def rot_angle_deg(R1, R2):
    R = R1.T @ R2
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def project(K, Pc):
    p = Pc @ K.T
    return p[..., :2] / p[..., 2:3]
