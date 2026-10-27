"""Shared utilities: mesh loading, nvdiffrast rendering, data loading."""
import json, numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr

H, W = 1152, 1536

def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    return m

class Renderer:
    def __init__(self, mesh, K, device="cuda", H=H, W=W):
        self.dev = device
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.K = np.asarray(K, dtype=np.float64)
        self.H, self.W = H, W

    def proj_mat(self, K, H, W, n=0.05, f=20.0):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip space (nvdiffrast, y up in NDC -> we flip so row 0 = top)
        P = np.array([[2 * fx / W, 0, (2 * cx - W) / W, 0],
                      [0, 2 * fy / H, (2 * cy - H) / H, 0],
                      [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                      [0, 0, 1, 0]], dtype=np.float64)
        return P

    def render(self, T_cam_mesh, K=None, H=None, W=None, crop=None, want_tex=True):
        """T_cam_mesh: (B,4,4) numpy/torch. crop: (x0,y0,w,h) window in full image coords, rendered at (H,W).
        Returns mask (B,H,W), rgb (B,H,W,3), depth (B,H,W)."""
        K = self.K if K is None else np.asarray(K, dtype=np.float64)
        H = self.H if H is None else H
        W = self.W if W is None else W
        Kc = K.copy()
        if crop is not None:
            x0, y0, cw, ch = crop
            Kc[0, 2] -= x0; Kc[1, 2] -= y0
            Kc[0] *= W / cw; Kc[1] *= H / ch
        P = torch.tensor(self.proj_mat(Kc, H, W), dtype=torch.float32, device=self.dev)
        T = torch.as_tensor(T_cam_mesh, dtype=torch.float32, device=self.dev)
        if T.ndim == 2: T = T[None]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T, vh)
        clip = torch.einsum('ij,bnj->bni', P, vc)
        rast, _ = dr.rasterize(self.ctx, clip.contiguous(), self.f, (H, W))
        mask = rast[..., 3] > 0
        depth = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)[0][..., 0] * mask
        rgb = None
        if want_tex:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex, uvi, filter_mode='linear') * mask[..., None]
        return mask, rgb, depth, rast

def load_camera(inputs):
    cam = json.load(open(f"{inputs}/camera.json"))
    return np.array(cam["K"]), np.array(cam["K_pipeline"]), np.array(cam["T_world_cam"])

def read_video(path):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    return frames

def iter_video(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok: break
        yield f

def crop_K(K, crop, S):
    x0, y0, cw, ch = crop
    Kc = np.asarray(K, dtype=np.float64).copy()
    Kc[0, 2] -= x0; Kc[1, 2] -= y0
    Kc[0] *= S / cw; Kc[1] *= S / ch
    return Kc

def proj_from_K(K, H, W, n=0.05, f=20.0):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([[2 * fx / W, 0, (2 * cx - W) / W, 0],
                     [0, 2 * fy / H, (2 * cy - H) / H, 0],
                     [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                     [0, 0, 1, 0]], dtype=np.float64)

def rotvec_to_mat(w):
    """Rodrigues, torch (B,3) -> (B,3,3), differentiable."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1] = -k[:, 2]; Kx[:, 0, 2] = k[:, 1]
    Kx[:, 1, 0] = k[:, 2]; Kx[:, 1, 2] = -k[:, 0]
    Kx[:, 2, 0] = -k[:, 1]; Kx[:, 2, 1] = k[:, 0]
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)

def render_batch(r, T_cam_mesh, Ps, S, with_tex=False):
    """T_cam_mesh torch (B,4,4), Ps torch (B,4,4) per-frame projection; returns soft silhouette (B,S,S) [, rgb]."""
    vh = torch.cat([r.v, torch.ones_like(r.v[:, :1])], 1)
    vc = torch.einsum('bij,nj->bni', T_cam_mesh, vh)
    clip = torch.einsum('bij,bnj->bni', Ps, vc).contiguous()
    rast, _ = dr.rasterize(r.ctx, clip, r.f, (S, S))
    one = (rast[..., 3:4] > 0).float()
    sil = dr.antialias(one, rast, clip, r.f)[..., 0]
    if not with_tex:
        return sil, None, rast
    uvi, _ = dr.interpolate(r.uv[None], rast, r.f)
    rgb = dr.texture(r.tex, uvi, filter_mode='linear') * one
    rgb = dr.antialias(rgb.contiguous(), rast, clip, r.f)
    return sil, rgb, rast
