"""Shared helpers: data loading, nvdiffrast rendering of the textured object mesh."""
import json, numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr

def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], float); Twc = np.array(cam["T_world_cam"], float)
    rec = np.load(f"{inp}/reconstruction.npz")
    return cam, K, Twc, rec

def load_video(path, scale=1.0):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, im = cap.read()
        if not ok: break
        if scale != 1.0: im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        fr.append(im[:, :, ::-1].copy())  # RGB
    return np.stack(fr)

def load_mesh(inp):
    m = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
    return m

class Renderer:
    def __init__(self, mesh, H, W, device="cuda", tex_max=2048):
        self.dev = device; self.H, self.W = H, W
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        uv = np.asarray(mesh.visual.uv, np.float32).copy(); uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        img = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))
        s = tex_max / max(img.shape[:2])
        if s < 1: img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        self.tex = torch.tensor(img.astype(np.float32) / 255., device=device)[None].contiguous()

    def proj(self, K, near=0.05, far=20.):
        H, W = self.H, self.W
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip space (y flipped to nvdiffrast's bottom-up raster convention handled below)
        P = np.array([[2 * fx / W, 0, 2 * (cx + 0.5) / W - 1, 0],
                      [0, 2 * fy / H, 2 * (cy + 0.5) / H - 1, 0],
                      [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
                      [0, 0, 1, 0]], np.float32)
        return torch.tensor(P, device=self.dev)

    def render(self, T_cam_mesh, K, need_color=True):
        """T_cam_mesh: (B,4,4) tensor/array. Returns mask (B,H,W), depth (B,H,W), color (B,H,W,3)."""
        T = torch.as_tensor(np.asarray(T_cam_mesh, np.float32) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,
                            dtype=torch.float32, device=self.dev)
        B = T.shape[0]
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # camera coords
        P = self.proj(K)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (self.H, self.W))
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        depth = depth[..., 0] * mask
        col = None
        if need_color:
            uvi, _ = dr.interpolate(self.uv[None].contiguous(), rast, self.f)
            col = dr.texture(self.tex.expand(1, -1, -1, -1).contiguous() if B == 1 else self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            col = col * mask[..., None]
        return mask, depth, col
