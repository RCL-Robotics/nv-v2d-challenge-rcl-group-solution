"""Shared helpers: mesh loading, nvdiffrast rendering, pose utilities."""
import numpy as np, torch, trimesh, nvdiffrast.torch as dr

def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    return m

class Renderer:
    def __init__(self, mesh, H, W, device="cuda"):
        self.dev = device
        self.H, self.W = H, W
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        vis = mesh.visual
        self.uv = None; self.tex = None
        if hasattr(vis, "uv") and vis.uv is not None:
            self.uv = torch.tensor(np.asarray(vis.uv), dtype=torch.float32, device=device)
            img = vis.material.baseColorTexture if hasattr(vis.material, "baseColorTexture") else vis.material.image
            t = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            t = t[::-1].copy()  # uv origin bottom-left
            self.tex = torch.tensor(t, device=device)[None]

    def render(self, T_cam_mesh, K, crop=None, scale=1.0):
        """T_cam_mesh (B,4,4) numpy/torch. crop=(x0,y0,w,h) in full-res pixels; scale downsample.
        Returns rgb (B,h,w,3), mask (B,h,w), depth (B,h,w) (camera z)."""
        T = torch.as_tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        if T.ndim == 2: T = T[None]
        B = T.shape[0]
        if crop is None: crop = (0, 0, self.W, self.H)
        x0, y0, w, h = crop
        rw, rh = int(round(w * scale)), int(round(h * scale))
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        # crop+scale intrinsics
        fx2, fy2 = fx * scale, fy * scale
        cx2, cy2 = (cx - x0) * scale, (cy - y0) * scale
        n, fa = 0.05, 20.0
        P = torch.tensor([[2 * fx2 / rw, 0, 2 * cx2 / rw - 1, 0],
                          [0, 2 * fy2 / rh, 2 * cy2 / rh - 1, 0],
                          [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)
        # nvdiffrast image row 0 corresponds to ndc y=-1, so opencv v maps directly (no flip)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (rh, rw))
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        if self.tex is not None:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex, uvi, filter_mode="linear")
        else:
            rgb = torch.ones(B, rh, rw, 3, device=self.dev)
        rgb = rgb * mask[..., None]
        return rgb, mask, depth[..., 0] * mask

def rotvec_to_R(r):
    import cv2
    return cv2.Rodrigues(np.asarray(r, dtype=np.float64))[0]

def R_angle(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
