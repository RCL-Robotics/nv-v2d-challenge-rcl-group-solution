"""Shared helpers: data loading, nvdiffrast rendering, SE(3) utilities."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R

H, W = 1152, 1536


def load_inputs(inp):
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], dtype=np.float64)
    Twc = np.array(cam["T_world_cam"], dtype=np.float64)
    rec = np.load(f"{inp}/reconstruction.npz")
    return cam, K, Twc, rec


def load_mesh(inp, max_faces=None):
    m = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
    return m


class Renderer:
    """Renders the textured mesh (colour, mask, depth) into crops with nvdiffrast."""

    def __init__(self, mesh, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext()
        v = np.asarray(mesh.vertices, np.float32)
        f = np.asarray(mesh.faces, np.int32)
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        uv = np.asarray(mesh.visual.uv, np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.verts_np = v

    def render(self, T_cam_mesh, K, box, out_hw):
        """T_cam_mesh (B,4,4) torch; K (3,3) np; box=(x0,y0,x1,y1) image crop rendered at out_hw=(h,w).
        Returns rgb (B,h,w,3), mask (B,h,w), depth (B,h,w)."""
        B = T_cam_mesh.shape[0]
        x0, y0, x1, y1 = box
        h, w = out_hw
        sx, sy = w / (x1 - x0), h / (y1 - y0)
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] - x0) * sx, (K[1, 2] - y0) * sy
        n, fa = 0.05, 20.0
        P = torch.tensor([[2 * fx / w, 0, 2 * cx / w - 1, 0],
                          [0, 2 * fy / h, 2 * cy / h - 1, 0],
                          [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)
        # OpenCV cam -> clip; nvdiffrast row 0 = bottom (y up in NDC). We use y-down flip via P row 1 sign below
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T_cam_mesh.float(), vh)
        clip = torch.einsum("ij,bnj->bni", P, vc)
        # nvdiffrast image row 0 corresponds to ndc y=-1; OpenCV row 0 = top; with P above, y_ndc = 2*v/h-1 so row0 ~ v=0. OK
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (h, w))
        texc, _ = dr.interpolate(self.uv[None], rast, self.f)
        rgb = dr.texture(self.tex, texc, filter_mode="linear")
        mask = rast[..., 3] > 0
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        return rgb, mask, z[..., 0]

    def render_soft_mask(self, T_cam_mesh, K, box, out_hw):
        """Differentiable (antialiased) silhouette."""
        B = T_cam_mesh.shape[0]
        x0, y0, x1, y1 = box
        h, w = out_hw
        sx, sy = w / (x1 - x0), h / (y1 - y0)
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] - x0) * sx, (K[1, 2] - y0) * sy
        n, fa = 0.05, 20.0
        P = torch.tensor([[2 * fx / w, 0, 2 * cx / w - 1, 0],
                          [0, 2 * fy / h, 2 * cy / h - 1, 0],
                          [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T_cam_mesh.float(), vh)
        clip = torch.einsum("ij,bnj->bni", P, vc).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (h, w))
        m = (rast[..., 3:4] > 0).float()
        m = dr.antialias(m, rast, clip, self.f)
        texc, _ = dr.interpolate(self.uv[None], rast, self.f)
        rgb = dr.texture(self.tex, texc, filter_mode="linear")
        rgb = dr.antialias(rgb * m, rast, clip, self.f)
        return m[..., 0], rgb


def rotvec_to_mat_t(rv):
    """Batched Rodrigues in torch."""
    th = rv.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = rv / th
    Kx = torch.zeros(rv.shape[:-1] + (3, 3), device=rv.device, dtype=rv.dtype)
    Kx[..., 0, 1], Kx[..., 0, 2] = -k[..., 2], k[..., 1]
    Kx[..., 1, 0], Kx[..., 1, 2] = k[..., 2], -k[..., 0]
    Kx[..., 2, 0], Kx[..., 2, 1] = -k[..., 1], k[..., 0]
    I = torch.eye(3, device=rv.device, dtype=rv.dtype).expand_as(Kx)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


def rot_angle_deg(Ra, Rb):
    Rr = np.swapaxes(Ra, -1, -2) @ Rb
    tr = np.clip((np.trace(Rr, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def rot_mean(Rs, w=None):
    """Chordal L2 mean of rotations."""
    if w is None:
        w = np.ones(len(Rs))
    M = (Rs * w[:, None, None]).sum(0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    return U @ D @ Vt


class FrameData:
    """Loads video frames + masks lazily, provides crops for scoring."""

    def __init__(self, inp, frames_dir=None):
        import h5py
        import cv2
        self.cv2 = cv2
        self.h5 = h5py.File(f"{inp}/masks.h5", "r")
        self.om = self.h5["object"]
        self.pm = self.h5["person"]
        cap = cv2.VideoCapture(f"{inp}/video.mp4")
        self.frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            self.frames.append(im)
        self.F = len(self.frames)

    def crop(self, i, box, out_hw):
        """Returns float tensors: image rgb (h,w,3) in [0,1], object mask, person mask, resampled to out_hw."""
        cv2 = self.cv2
        x0, y0, x1, y1 = [int(round(b)) for b in box]
        h, w = out_hw
        im = self.frames[i]
        pad = 400
        imp = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        c = imp[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
        om = np.pad(self.om[i], pad)[y0 + pad:y1 + pad, x0 + pad:x1 + pad].astype(np.uint8)
        pm = np.pad(self.pm[i], pad)[y0 + pad:y1 + pad, x0 + pad:x1 + pad].astype(np.uint8)
        c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)[..., ::-1].astype(np.float32) / 255.0
        om = cv2.resize(om, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return c, om, pm


def project(K, p):
    q = K @ p
    return q[:2] / q[2]


def frame_box(K, Tc, om_bbox=None, scale=1.5, minsize=60):
    """Square crop box around the object's projected position."""
    c = project(K, Tc[:3, 3])
    size = 0.32 * K[0, 0] / Tc[2, 3] * scale
    size = max(size, minsize)
    return (c[0] - size / 2, c[1] - size / 2, c[0] + size / 2, c[1] + size / 2)


def score_batch(Rd, K, box, out_hw, Tcs, img, om, pm, dev="cuda"):
    """Score candidate camera-frame poses Tcs (B,4,4 np) against one crop.
    Returns dict of per-candidate numpy arrays: iou (person-aware), col (colour NCC), and rgb renders."""
    T = torch.tensor(Tcs, device=dev, dtype=torch.float32)
    rgb, m, _ = Rd.render(T, K, box, out_hw)
    O = torch.tensor(om, device=dev)
    P = torch.tensor(pm, device=dev) & ~O
    I = torch.tensor(img, device=dev)
    inter = (m & O).flatten(1).sum(1).float()
    union = (O | (m & ~P)).flatten(1).sum(1).float()
    iou = inter / union.clamp_min(1)
    # colour agreement inside the visible overlap: per-channel standardised NCC
    ov = (m & O).float()[..., None]
    n = ov.flatten(1).sum(1).clamp_min(1)[:, None]
    def stdz(x):
        mu = (x * ov).flatten(1, 2).sum(1) / n
        xc = (x - mu[:, None, None]) * ov
        sd = (xc ** 2).flatten(1, 2).sum(1) / n
        return xc / (sd[:, None, None] + 1e-4).sqrt()
    a = stdz(rgb)
    b = stdz(I[None].expand_as(rgb))
    col = (a * b).flatten(1, 2).sum(1) / n
    return dict(iou=iou.cpu().numpy(), col=col.mean(1).cpu().numpy(), n=n[:, 0].cpu().numpy())


def rot_about_axis(axis, ang):
    return R.from_rotvec(np.asarray(axis, float) / np.linalg.norm(axis) * ang).as_matrix()


def refine_batch(Rd, K, box, hw, T0, img, om, pm, iters=80, lr_r=0.02, lr_t=0.004, w_col=0.5,
                 fix_depth=False, opt_rot=True, prior=None, dev="cuda"):
    """Render-and-compare refinement of B camera-frame poses against one crop.
    Parametrisation: T = [R0 exp(w) | t0 + dt]; dt in metres (camera frame).
    Loss: person-aware soft-silhouette (1 - dice) + w_col * colour residual after per-channel affine fit.
    Returns refined poses (B,4,4) np and final loss components."""
    T0t = torch.tensor(T0, device=dev, dtype=torch.float32)
    B = T0t.shape[0]
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    dt = torch.zeros(B, 3, device=dev, requires_grad=True)
    O = torch.tensor(om, device=dev).float()
    Pn = (torch.tensor(pm, device=dev) & ~torch.tensor(om, device=dev)).float()
    I = torch.tensor(img, device=dev)
    opt = torch.optim.Adam([{"params": [w], "lr": lr_r if opt_rot else 0.0}, {"params": [dt], "lr": lr_t}])
    for it in range(iters + 1):
        Rm = T0t[:, :3, :3] @ rotvec_to_mat_t(w)
        d = dt if not fix_depth else torch.cat([dt[:, :2], torch.zeros_like(dt[:, 2:])], 1)
        t = T0t[:, :3, 3] + d
        T = torch.zeros_like(T0t)
        T[:, :3, :3] = Rm
        T[:, :3, 3] = t
        T[:, 3, 3] = 1
        m, rgb = Rd.render_soft_mask(T, K, box, hw)
        mv = m * (1 - Pn)  # rendered pixels that fall on the person can be occluded: ignore
        inter = (mv * O).flatten(1).sum(1)
        dice = 2 * inter / (mv.flatten(1).sum(1) + O.sum() + 1e-6)
        ov = (m * O)[..., None]
        n = ov.flatten(1, 2).sum(1).clamp_min(1)
        r = rgb / m[..., None].clamp_min(1e-3)
        mu_r = (r * ov).flatten(1, 2).sum(1) / n
        mu_i = (I[None] * ov).flatten(1, 2).sum(1) / n
        rc = (r - mu_r[:, None, None]) * ov
        ic = (I[None] - mu_i[:, None, None]) * ov
        g = (rc * ic).flatten(1, 2).sum(1) / ((rc * rc).flatten(1, 2).sum(1) + 1e-4)
        g = g.clamp(0.2, 3.0)
        res = ((g[:, None, None] * rc - ic) ** 2 * ov).flatten(1, 3).sum(1) / n[:, 0] / 3
        sd = ((ic ** 2) * ov).flatten(1, 3).sum(1) / n[:, 0] / 3 + 1e-4
        col = res / sd  # 1 - R^2 like
        loss = (1 - dice) + w_col * col
        if prior is not None:
            loss = loss + prior * (w ** 2).sum(1)
        if it == iters:
            break
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    Tn = T.detach().cpu().numpy().astype(np.float64)
    return Tn, dict(loss=loss.detach().cpu().numpy(), dice=dice.detach().cpu().numpy(),
                    col=col.detach().cpu().numpy())
