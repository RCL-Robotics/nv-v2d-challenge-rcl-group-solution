"""Shared helpers: data loading, mesh, nvdiffrast rendering, SE(3) utilities."""
import json
import numpy as np
import torch
import trimesh

INP = '/work/inputs'
H, W = 1152, 1536


def load_camera(inp=INP):
    c = json.load(open(f'{inp}/camera.json'))
    return np.array(c['K'], float), np.array(c['T_world_cam'], float), np.array(c['K_pipeline'], float)


def load_mesh(inp=INP):
    m = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'), np.float32) / 255.
    return (np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
            np.asarray(m.visual.uv, np.float32), tex, m)


def read_video(path, step=1, scale=1.0):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            out.append(f[:, :, ::-1])
        i += 1
    return np.stack(out)


# ---------------------------------------------------------------- SO(3)/SE(3)
def rotvec_to_R(v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()


def R_to_rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def rot_angle(Ra, Rb):
    """angle (deg) between rotations, batched."""
    M = np.einsum('...ji,...jk->...ik', Ra, Rb)
    return np.degrees(np.arccos(np.clip((np.trace(M, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


# ---------------------------------------------------------------- rendering
class Renderer:
    """nvdiffrast renderer of the textured mesh, with a pinhole K (OpenCV)."""

    def __init__(self, inp=INP, device='cuda', n_sub=None):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext()
        v, f, uv, tex, self.tm = load_mesh(inp)
        self.dev = device
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        self.uv = torch.tensor(uv, device=device)
        self.tex = torch.tensor(tex[None], device=device).flip(1)  # glTF uv origin top-left
        self.verts_np = v

    def render(self, K, T_cam_mesh, w=W, h=H, x0=0, y0=0, rgb=True):
        """T_cam_mesh: (B,4,4) numpy/torch. Returns mask (B,h,w), rgb (B,h,w,3), depth (B,h,w)."""
        T = torch.as_tensor(np.asarray(T_cam_mesh, np.float32), device=self.dev)
        if T.ndim == 2:
            T = T[None]
        B = T.shape[0]
        if np.ndim(x0) == 0:
            x0 = [x0] * B
            y0 = [y0] * B
        Ps = torch.stack([torch.tensor(self._P(K, w, h, a, b), device=self.dev) for a, b in zip(x0, y0)])
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T, vh)  # camera coords (OpenCV)
        clip = torch.einsum('bij,bnj->bni', Ps, vc)
        rast, _ = self.dr.rasterize(self.glctx, clip.contiguous(), self.f, (h, w))
        mask = rast[..., 3] > 0
        depth, _ = self.dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        out = {'mask': mask, 'depth': depth[..., 0] * mask, 'rast': rast}
        if rgb:
            uvi, _ = self.dr.interpolate(self.uv[None], rast, self.f)
            col = self.dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode='linear')
            out['rgb'] = col * mask[..., None]
        return out

    @staticmethod
    def _P(K, w, h, x0, y0, n=0.05, fa=20.):
        # maps OpenCV camera coords (x right, y down, z fwd) directly to clip space, raster row 0 = image top.
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] - x0, K[1, 2] - y0
        # x_ndc = (2*(fx x/z + cx)/w - 1); nvdiffrast row 0 corresponds to y_ndc=-1
        return np.array([[2 * fx / w, 0, 2 * cx / w - 1, 0],
                         [0, 2 * fy / h, 2 * cy / h - 1, 0],
                         [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)],
                         [0, 0, 1, 0]], np.float32)


def project(K, T_cam_mesh, pts):
    pc = pts @ T_cam_mesh[:3, :3].T + T_cam_mesh[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]
