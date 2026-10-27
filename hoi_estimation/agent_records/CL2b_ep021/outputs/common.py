"""Shared utilities: data loading, nvdiffrast rendering, pose helpers."""
import json, os
import numpy as np
import torch
import trimesh
import h5py
import cv2

H, W = 1152, 1536


def load_inputs(inp):
    cam = json.load(open(os.path.join(inp, 'camera.json')))
    K = np.array(cam['K'], np.float64)
    Kp = np.array(cam['K_pipeline'], np.float64)
    Twc = np.array(cam['T_world_cam'], np.float64)
    rec = dict(np.load(os.path.join(inp, 'reconstruction.npz')))
    gp = json.load(open(os.path.join(inp, 'ground_plane.json')))
    return K, Kp, Twc, rec, gp


def load_mesh(inp):
    m = trimesh.load(os.path.join(inp, 'object_mesh.glb'), force='scene', process=False).to_geometry()
    return m


def read_video(path, every=1, scale=1.0):
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            if scale != 1.0:
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(f[:, :, ::-1].copy())  # RGB
        i += 1
    return frames


def load_masks(inp, scale=1.0):
    f = h5py.File(os.path.join(inp, 'masks.h5'), 'r')
    obj = f['object'][:]
    per = f['person'][:]
    if scale != 1.0:
        h, w = int(H * scale), int(W * scale)
        obj = np.stack([cv2.resize(o.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for o in obj]).astype(bool)
        per = np.stack([cv2.resize(o.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) for o in per]).astype(bool)
    return obj, per


# ---------------- pose helpers ----------------
def rotvec_to_R(r):
    return cv2.Rodrigues(np.asarray(r, np.float64).reshape(3, 1))[0]


def R_to_rotvec(R):
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].ravel()


def rot_angle_deg(R1, R2):
    R = R1.T @ R2
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def rot_angles_deg(Ra, Rb):
    """batched geodesic angle between (N,3,3) arrays"""
    R = np.einsum('nji,njk->nik', Ra, Rb)
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def proj_matrix(K, w, h, near=0.05, far=20.0):
    """OpenCV intrinsics -> OpenGL clip matrix for camera with x right, y down, z forward.
    nvdiffrast clip space: y up in NDC maps to row index ... we flip so output image rows match OpenCV."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 2 * fx / w
    P[0, 2] = 2 * (cx + 0.5) / w - 1  # pixel-center convention: pixel i center at i
    P[1, 1] = 2 * fy / h
    P[1, 2] = 2 * (cy + 0.5) / h - 1
    P[2, 2] = (far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = 1.0
    return P


class Renderer:
    """nvdiffrast renderer. Images are in OpenCV row order (row 0 = top)."""

    def __init__(self, mesh, K, w, h, device='cuda', tex_scale=0.25):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext()
        self.dev = device
        self.w, self.h = w, h
        self.K = np.asarray(K, np.float64)
        self.P = torch.tensor(proj_matrix(self.K, w, h), device=device)
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        uv = np.asarray(mesh.visual.uv, np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        img = mesh.visual.material.baseColorTexture
        tex = np.asarray(img.convert('RGB'))
        if tex_scale != 1.0:
            tex = cv2.resize(tex, None, fx=tex_scale, fy=tex_scale, interpolation=cv2.INTER_AREA)
        self.tex = torch.tensor(tex.astype(np.float32) / 255.0, device=device)[None].contiguous()

    def set_K(self, K):
        self.K = np.asarray(K, np.float64)
        self.P = torch.tensor(proj_matrix(self.K, self.w, self.h), device=self.dev)

    def render(self, T_cam_mesh, need_color=True):
        """T_cam_mesh: (B,4,4) numpy or torch. returns dict mask (B,h,w), depth (B,h,w), color (B,h,w,3)."""
        if not torch.is_tensor(T_cam_mesh):
            T_cam_mesh = torch.tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        T = T_cam_mesh.float()
        if T.dim() == 2:
            T = T[None]
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T, vh)  # camera coords
        clip = torch.einsum('ij,bnj->bni', self.P, vc)
        # OpenCV y down: NDC y = 2*(v+0.5)/h - 1; nvdiffrast row 0 corresponds to NDC y=-1 -> rows ordered top-down already
        rast, _ = self.dr.rasterize(self.glctx, clip.contiguous(), self.f, (self.h, self.w))
        mask = rast[..., 3] > 0
        depth, _ = self.dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        out = {'mask': mask, 'depth': depth[..., 0] * mask, 'rast': rast, 'vc': vc}
        if need_color:
            texc, _ = self.dr.interpolate(self.uv[None], rast, self.f)
            col = self.dr.texture(self.tex, texc, filter_mode='linear')
            out['color'] = col * mask[..., None]
        return out


def cam_poses_from_world(Twc, Tw):
    Tcw = np.linalg.inv(Twc)
    return np.einsum('ij,njk->nik', Tcw, Tw)


def world_poses_from_cam(Twc, Tc):
    return np.einsum('ij,njk->nik', Twc, Tc)
