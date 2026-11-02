"""Shared helpers: data loading, nvdiffrast rendering, SE(3) utilities."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

W_IMG, H_IMG = 1536, 1152


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], dtype=np.float64)
    Twc = np.array(cam["T_world_cam"], dtype=np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], dtype=np.float64)
    return K, Twc, rec, plane


def load_mesh(inp):
    m = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
    return m


class Renderer:
    def __init__(self, mesh, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()

    def proj(self, K, W, H, near=0.05, far=20.0):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip space; nvdiffrast pixel row 0 at y_ndc=-1
        P = np.array([
            [2 * fx / W, 0, (2 * cx / W) - 1, 0],
            [0, 2 * fy / H, (2 * cy / H) - 1, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], dtype=np.float64)
        return P

    def render(self, T_cam_mesh, K, W, H, color=False, crop=None):
        """T_cam_mesh: (B,4,4). crop=(x0,y0,w,h) renders only this window of the W x H image.
        Returns mask (B,h,w) float, depth (B,h,w) (0 outside), optional rgb (B,h,w,3)."""
        T = torch.as_tensor(np.asarray(T_cam_mesh), dtype=torch.float32, device=self.dev)
        if crop is not None:
            x0, y0, w, h = crop
            K = K.copy(); K[0, 2] -= x0; K[1, 2] -= y0
            W, H = w, h
        P = torch.tensor(self.proj(K, W, H), dtype=torch.float32, device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, resolution=[H, W])
        mask = (rast[..., 3] > 0).float()
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        depth = z[..., 0] * mask
        out = [mask, depth]
        if color:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            rgb = dr.texture(self.tex, uvi, filter_mode="linear") * mask[..., None]
            out.append(rgb)
        return out


def rotmat_angle(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def so3_log(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def so3_exp(w):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(w).as_matrix()


def simplified_mesh(mesh, n_faces=6000):
    """Watertight-ish decimated copy of the mesh with per-vertex colours sampled from the texture
    (used for fast batched silhouette/colour scoring)."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces)
    uv = np.asarray(mesh.visual.uv)
    tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float64) / 255.0
    H, W = tex.shape[:2]
    px = np.clip((uv[:, 0] % 1.0) * (W - 1), 0, W - 1).astype(int)
    py = np.clip((1 - uv[:, 1] % 1.0) * (H - 1), 0, H - 1).astype(int)
    col = tex[py, px]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    m.merge_close_vertices(1e-6)
    m = m.simplify_quadric_decimation(n_faces)
    vs = np.asarray(m.vertices); fs = np.asarray(m.faces if hasattr(m, 'faces') else m.triangles)
    # colour: average of original vertex colours in a small neighbourhood
    tree = cKDTree(v)
    idx = tree.query(vs, k=16)[1]
    cs = col[idx].mean(1)
    return vs, fs, cs


class FastRenderer:
    """Differentiable batched renderer for the decimated vertex-coloured mesh."""
    def __init__(self, vs, fs, cs, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(vs, dtype=torch.float32, device=device)
        self.f = torch.tensor(fs, dtype=torch.int32, device=device)
        self.c = torch.tensor(cs, dtype=torch.float32, device=device)

    def render(self, R, t, Kc, res):
        """R (B,3,3), t (B,3) torch camera-frame; Kc (B,3,3) numpy/torch crop intrinsics for a res x res image.
        Returns antialiased mask (B,res,res), colour (B,res,res,3), depth (B,res,res)."""
        B = R.shape[0]
        Kc = torch.as_tensor(Kc, dtype=torch.float32, device=self.dev)
        vc = torch.einsum("bij,nj->bni", R, self.v) + t[:, None, :]
        near, far = 0.05, 20.0
        x = vc[..., 0]; y = vc[..., 1]; z = vc[..., 2]
        fx = Kc[:, 0, 0, None]; fy = Kc[:, 1, 1, None]; cx = Kc[:, 0, 2, None]; cy = Kc[:, 1, 2, None]
        cxn = 2 * fx / res * x + (2 * cx / res - 1) * z
        cyn = 2 * fy / res * y + (2 * cy / res - 1) * z
        czn = (far + near) / (far - near) * z - 2 * far * near / (far - near)
        clip = torch.stack([cxn, cyn, czn, z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, resolution=[res, res])
        attr = torch.cat([self.c[None].expand(B, -1, -1), z[..., None], torch.ones_like(z[..., None])], -1).contiguous()
        out, _ = dr.interpolate(attr, rast, self.f)
        out = dr.antialias(out, rast, clip, self.f)
        return out[..., 4], out[..., :3], out[..., 3]
