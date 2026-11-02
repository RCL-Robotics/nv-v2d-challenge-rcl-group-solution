import numpy as np, torch, nvdiffrast.torch as dr
from common import load_mesh
class Renderer:
    def __init__(self, W, H, device='cuda'):
        m = load_mesh()
        self.W, self.H, self.dev = W, H, device
        self.v = torch.tensor(m.vertices, dtype=torch.float32, device=device)
        self.f = torch.tensor(m.faces, dtype=torch.int32, device=device)
        uv = np.array(m.visual.uv, dtype=np.float32).copy(); uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.array(m.visual.material.baseColorTexture.convert('RGB'), dtype=np.float32) / 255.
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext()
    def clip(self, T, K, n=0.05, f=20.):
        # T: (B,4,4) cam_from_mesh torch; K: 3x3 for this resolution (pixel centers at integers)
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        pc = torch.einsum('bij,nj->bni', T, vh)[..., :3]
        X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
        u = K[0, 0] * X + K[0, 2] * Z; v = K[1, 1] * Y + K[1, 2] * Z
        xc = (2 * u + Z) / self.W - Z; yc = (2 * v + Z) / self.H - Z
        zc = Z * (f + n) / (f - n) - 2 * f * n / (f - n)
        return torch.stack([xc, yc, zc, Z], -1).contiguous(), Z
    def render(self, T, K, rgb=True):
        T = torch.as_tensor(T, dtype=torch.float32, device=self.dev)
        K = torch.as_tensor(K, dtype=torch.float32, device=self.dev)
        clip, Z = self.clip(T, K)
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (self.H, self.W))
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(Z[..., None].contiguous(), rast, self.f)
        out = {'mask': mask, 'depth': depth[..., 0] * mask, 'rast': rast, 'clip': clip}
        if rgb:
            uvi, _ = dr.interpolate(self.uv, rast, self.f)
            col = dr.texture(self.tex, uvi, filter_mode='linear')
            out['rgb'] = col * mask[..., None]
        return out
def scaleK(K, s):
    K2 = np.array(K, dtype=np.float64).copy()
    K2[0, 0] *= s; K2[1, 1] *= s
    K2[0, 2] = (K[0, 2] + 0.5) * s - 0.5; K2[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K2
