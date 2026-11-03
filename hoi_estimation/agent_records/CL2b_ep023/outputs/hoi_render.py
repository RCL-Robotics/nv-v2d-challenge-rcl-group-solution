"""Textured mesh renderer (nvdiffrast) used by solve.py and the analysis scripts."""
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr


class MeshRenderer:
    def __init__(self, mesh_path, device="cuda", max_tex=1024):
        m = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
        self.mesh = m
        self.device = device
        self.verts = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=device)
        self.faces = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=device)
        uv = np.asarray(m.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        img = m.visual.material.baseColorTexture.convert("RGB")
        s = max_tex / max(img.size)
        if s < 1:
            img = img.resize((int(img.size[0] * s), int(img.size[1] * s)))
        tex = np.asarray(img, dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext(device=device)

    def render(self, T_cam_mesh, K, H, W, crop=None, near=0.05, far=20.0, want_rgb=True):
        """T_cam_mesh: (B,4,4) torch/np. K: 3x3. crop=(x0,y0,w,h,out_w,out_h) renders a sub-window.
        Returns dict with mask (B,h,w), depth (B,h,w) (camera z, 0 outside), rgb (B,h,w,3)."""
        dev = self.device
        T = torch.as_tensor(np.asarray(T_cam_mesh) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,
                            dtype=torch.float32, device=dev)
        if T.ndim == 2:
            T = T[None]
        B = T.shape[0]
        K = np.asarray(K, dtype=np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        if crop is None:
            x0, y0, w, h, ow, oh = 0, 0, W, H, W, H
        else:
            x0, y0, w, h, ow, oh = crop
        # projection to NDC for the crop window [x0,x0+w]x[y0,y0+h]
        sx, sy = ow / w, oh / h
        fx_, fy_ = fx * sx, fy * sy
        cx_, cy_ = (cx - x0 + 0.5) * sx, (cy - y0 + 0.5) * sy  # OpenCV pixel centres
        P = torch.tensor([
            [2 * fx_ / ow, 0, 2 * cx_ / ow - 1, 0],
            [0, 2 * fy_ / oh, 2 * cy_ / oh - 1, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], dtype=torch.float32, device=dev)
        # x_ndc = 2u/ow - 1, y_ndc = 2v/oh - 1; nvdiffrast row 0 is y_ndc=-1 -> OpenCV image rows
        vh = torch.cat([self.verts, torch.ones_like(self.verts[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.faces, resolution=[oh, ow])
        out = {}
        mask = rast[..., 3] > 0
        out["mask"] = mask
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.faces)
        out["depth"] = z[..., 0] * mask
        out["rast"] = rast
        if want_rgb:
            uvi, _ = dr.interpolate(self.uv, rast, self.faces)
            col = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            out["rgb"] = col * mask[..., None]
        return out
