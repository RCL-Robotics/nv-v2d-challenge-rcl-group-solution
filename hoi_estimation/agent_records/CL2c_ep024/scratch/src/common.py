import json, numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr

INP = '/work/inputs'
dev = 'cuda'


def load_inputs(inp=INP):
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K'], float)
    Twc = np.array(cam['T_world_cam'], float)
    rec = np.load(f'{inp}/reconstruction.npz')
    return cam, K, Twc, rec


class Mesh:
    def __init__(self, path=f'{INP}/object_mesh.glb'):
        m = trimesh.load(path, force='scene', process=False).to_geometry()
        self.tm = m
        self.v = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
        self.f = torch.tensor(m.faces, dtype=torch.int32, device=dev)
        uv = np.array(m.visual.uv, np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=dev)
        mat = m.visual.material
        img = mat.baseColorTexture if getattr(mat, 'baseColorTexture', None) is not None else mat.image
        tex = np.asarray(img.convert('RGB'), np.float32) / 255.
        self.tex = torch.tensor(tex, device=dev)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext()

    def render(self, Tcm, K, x0, y0, w, h, scale=1.0, need_tex=True):
        """Tcm: (B,4,4) torch camera-from-mesh. Renders crop [x0,x0+w) x [y0,y0+h) of the image at scale.
        returns mask (B,H,W), rgb (B,H,W,3), depth (B,H,W)"""
        B = Tcm.shape[0]
        W, H = int(round(w * scale)), int(round(h * scale))
        fx, fy, cx, cy = K[0, 0] * scale, K[1, 1] * scale, (K[0, 2] - x0) * scale, (K[1, 2] - y0) * scale
        n, f = 0.05, 20.
        P = torch.tensor([[2 * fx / W, 0, 2 * (cx + 0.5 * 0) / W - 1, 0],
                          [0, 2 * fy / H, 2 * cy / H - 1, 0],
                          [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=dev)
        # pixel centres: opencv pixel u covers [u-0.5,u+0.5]; ndc x=(2*(u+0.5)/W -1). projected u=fx X/Z+cx
        P[0, 2] = 2 * (cx + 0.5) / W - 1
        P[1, 2] = 2 * (cy + 0.5) / H - 1
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', Tcm.float(), vh)
        clip = torch.einsum('ij,bnj->bni', P, vc)
        # nvdiffrast: y up in NDC -> row 0 at bottom; we flip later
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (H, W))
        mask = (rast[..., 3] > 0)
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        out_rgb = None
        if need_tex:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            out_rgb = dr.texture(self.tex, uvi, filter_mode='linear')
            out_rgb = out_rgb * mask[..., None]
        # rows: ndc y=-1 is row 0 in nvdiffrast; our projection maps image row v-> ndc y = 2(v+.5)/H-1, so row 0 = top. good
        return mask, out_rgb, depth[..., 0] * mask


def load_masks(frames=None, inp=INP):
    with h5py.File(f'{inp}/masks.h5') as f:
        if frames is None:
            return f['object'][:], f['person'][:]
        return f['object'][frames], f['person'][frames]


def read_frame(i, d='/work/scratch/frames'):
    return cv2.imread(f'{d}/{i + 1:04d}.jpg')


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2]; Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


def render_soft(M, Tcm, K, x0, y0, w, h, scale=1.0, need_tex=False):
    """differentiable (antialiased) silhouette + optional texture"""
    B = Tcm.shape[0]
    W, H = int(round(w * scale)), int(round(h * scale))
    fx, fy, cx, cy = K[0, 0] * scale, K[1, 1] * scale, (K[0, 2] - x0) * scale, (K[1, 2] - y0) * scale
    n, f = 0.05, 20.
    P = torch.tensor([[2 * fx / W, 0, 2 * (cx + 0.5) / W - 1, 0],
                      [0, 2 * fy / H, 2 * (cy + 0.5) / H - 1, 0],
                      [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                      [0, 0, 1, 0]], dtype=torch.float32, device=dev)
    vc = Tcm[:, None, :3, :3].matmul(M.v[None, :, :, None])[..., 0] + Tcm[:, None, :3, 3]
    vh = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
    clip = vh @ P.T
    rast, _ = dr.rasterize(M.glctx, clip.contiguous(), M.f, (H, W))
    sil = (rast[..., 3:4] > 0).float()
    rgb = None
    if need_tex:
        uvi, _ = dr.interpolate(M.uv[None], rast, M.f)
        rgb = dr.texture(M.tex, uvi, filter_mode='linear') * sil
        rgb = dr.antialias(torch.cat([rgb, sil], -1).contiguous(), rast, clip.contiguous(), M.f)
        sil = rgb[..., 3]; rgb = rgb[..., :3]
    else:
        sil = dr.antialias(sil.contiguous(), rast, clip.contiguous(), M.f)[..., 0]
    return sil, rgb
