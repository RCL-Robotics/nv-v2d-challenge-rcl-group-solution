import numpy as np, torch, trimesh, json, h5py, cv2
import nvdiffrast.torch as dr

INP = '/work/inputs'
dev = 'cuda'


def load_mesh(path=INP + '/object_mesh.glb'):
    m = trimesh.load(path, force='scene', process=False).to_geometry()
    tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
    return m, tex


class Renderer:
    def __init__(self, mesh, tex, W, H):
        self.ctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev)
        self.f = torch.tensor(mesh.faces, dtype=torch.int32, device=dev)
        uv = mesh.visual.uv.copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, dtype=torch.float32, device=dev)
        self.tex = torch.tensor(tex, dtype=torch.float32, device=dev)[None] / 255.
        self.W, self.H = W, H

    def proj(self, K, n=0.05, fa=50.):
        W, H = self.W, self.H
        P = np.zeros((4, 4))
        P[0, 0] = 2 * K[0, 0] / W; P[0, 2] = 2 * (K[0, 2] + 0.5) / W - 1
        P[1, 1] = 2 * K[1, 1] / H; P[1, 2] = 2 * (K[1, 2] + 0.5) / H - 1
        P[2, 2] = (fa + n) / (fa - n); P[2, 3] = -2 * fa * n / (fa - n)
        P[3, 2] = 1
        return P

    def render(self, K, T_cam_mesh, rgb=True):
        """T_cam_mesh: (B,4,4) numpy or torch. returns mask (B,H,W), depth (B,H,W), rgb (B,H,W,3)"""
        T = torch.as_tensor(np.asarray(T_cam_mesh) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,
                            dtype=torch.float32, device=dev)
        if T.ndim == 2: T = T[None]
        P = torch.tensor(self.proj(K), dtype=torch.float32, device=dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T, vh)
        clip = torch.einsum('ij,bnj->bni', P, vc)
        # nvdiffrast: y up in clip -> image rows from bottom; we flip later
        rast, _ = dr.rasterize(self.ctx, clip.contiguous(), self.f, (self.H, self.W))
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        out = {'mask': mask, 'depth': depth[..., 0] * mask}
        if rgb:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            col = dr.texture(self.tex, uvi, filter_mode='linear')
            out['rgb'] = col * mask[..., None]
        return out


def load_cam():
    c = json.load(open(INP + '/camera.json'))
    return np.array(c['K']), np.array(c['T_world_cam']), np.array(c['K_pipeline'])


def load_masks(key, sl=slice(None)):
    with h5py.File(INP + '/masks.h5') as f:
        return f[key][sl]


def rotdist(R1, R2):
    R = np.einsum('...ji,...jk->...ik', R1, R2)
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    R = U @ Vt
    if np.linalg.det(R) < 0: R = U @ np.diag([1, 1, -1]) @ Vt
    return R
