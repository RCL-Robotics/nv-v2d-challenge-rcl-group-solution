"""Differentiable render-and-compare fitting of the iron mesh to per-frame masks / colours."""
import numpy as np
import torch
import torch.nn.functional as Fnn
import nvdiffrast.torch as dr
from common import Renderer


def so3_exp(w):
    """w (B,3) -> R (B,3,3), differentiable."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)  # smooth at 0 (no NaN gradient)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


class Fitter:
    def __init__(self, mesh, K, n_sub_faces=None, device="cuda"):
        self.R = Renderer(mesh, device)
        self.dev = device
        self.K = np.asarray(K, np.float64)
        # simplified copy for fast silhouette rendering
        self.v = self.R.v
        self.f = self.R.f
        self.glctx = self.R.glctx
        # decimated, watertight-ish copy for fast silhouette-only rendering
        import open3d as o3d
        tm = mesh.copy()
        tm.merge_vertices(merge_tex=True, merge_norm=True)
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
                                       o3d.utility.Vector3iVector(np.asarray(tm.faces)))
        om = om.simplify_quadric_decimation(6000)
        self.vs = torch.tensor(np.asarray(om.vertices), dtype=torch.float32, device=device)
        self.fs = torch.tensor(np.asarray(om.triangles), dtype=torch.int32, device=device)
        self.cm = torch.tensor(mesh.bounds.mean(0), dtype=torch.float32, device=device)

    def proj_mats(self, rois, hw):
        """rois (B,4) x0,y0,x1,y1 -> (B,4,4) projection for each crop."""
        K = self.K
        h, w = hw
        P = []
        n, far = 0.05, 20.0
        for (x0, y0, x1, y1) in rois:
            sx, sy = w / (x1 - x0), h / (y1 - y0)
            fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] - x0) * sx, (K[1, 2] - y0) * sy
            P.append([[2 * fx / w, 0, 2 * cx / w - 1, 0],
                      [0, 2 * fy / h, 2 * cy / h - 1, 0],
                      [0, 0, (far + n) / (far - n), -2 * far * n / (far - n)],
                      [0, 0, 1, 0]])
        return torch.tensor(np.array(P), dtype=torch.float32, device=self.dev)

    def scaled(self, v, sigma):
        if sigma is None:
            return v
        return (v - self.cm) * sigma + self.cm

    def render(self, Rm, t, P, hw, rgb=False, sigma=None):
        """Rm (B,3,3), t (B,3) torch camera-frame pose; P (B,4,4). Returns soft mask (B,h,w), rgb, depth.
        sigma: optional global scale of the mesh about its bbox centre (absorbs mesh-size / mask-bias mismatch)."""
        if not rgb:
            return self.render_sil(Rm, t, P, hw, sigma)
        vc = torch.einsum("bij,nj->bni", Rm, self.scaled(self.v, sigma)) + t[:, None, :]
        vh = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
        clip = torch.einsum("bij,bnj->bni", P, vh).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, resolution=list(hw))
        m = (rast[..., 3:4] > 0).float()
        out_rgb = None
        if rgb:
            uvi, _ = dr.interpolate(self.R.uv[None], rast, self.f)
            col = dr.texture(self.R.tex.expand(Rm.shape[0], -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            col = torch.cat([col * m, m], -1)
            col = dr.antialias(col.contiguous(), rast, clip, self.f)
            return col[..., 3], col[..., :3], vc
        m = dr.antialias(m.contiguous(), rast, clip, self.f)[..., 0]
        return m, out_rgb, vc

    def render_sil(self, Rm, t, P, hw, sigma=None):
        vc = torch.einsum("bij,nj->bni", Rm, self.scaled(self.vs, sigma)) + t[:, None, :]
        vh = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
        clip = torch.einsum("bij,bnj->bni", P, vh).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.fs, resolution=list(hw))
        m = (rast[..., 3:4] > 0).float()
        m = dr.antialias(m.contiguous(), rast, clip, self.fs)[..., 0]
        return m, None, vc


def make_rois(masks_bbox, pad=0.35, min_size=60):
    """Square ROI around each bbox (x0,y0,x1,y1)."""
    rois = []
    for (x0, y0, x1, y1) in masks_bbox:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        s = max(x1 - x0, y1 - y0, min_size) * (1 + 2 * pad)
        rois.append((cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))
    return np.array(rois)


def crop_resize(img, roi, hw, interp):
    """Crop float image (H,W,C) or (H,W) to roi via affine warp."""
    import cv2
    x0, y0, x1, y1 = roi
    h, w = hw
    A = np.array([[w / (x1 - x0), 0, -x0 * w / (x1 - x0)], [0, h / (y1 - y0), -y0 * h / (y1 - y0)]])
    return cv2.warpAffine(img, A, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def color_loss(rgb_r, m_r, rgb_o, m_o, valid):
    """Mean-normalised colour discrepancy inside rendered & observed object pixels (B,)."""
    w = (m_r * m_o * valid)
    ws = w.sum((1, 2)).clamp_min(1.0)
    def norm(x):
        mu = (x * w[..., None]).sum((1, 2)) / ws[:, None]
        xc = x - mu[:, None, None, :]
        sd = ((xc ** 2 * w[..., None]).sum((1, 2)) / ws[:, None]).sqrt().mean(-1).clamp_min(1e-3)
        return xc / sd[:, None, None, None]
    a, b = norm(rgb_r), norm(rgb_o)
    return ((a - b).abs().mean(-1) * w).sum((1, 2)) / ws
