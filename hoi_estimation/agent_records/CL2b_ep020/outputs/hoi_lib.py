"""Shared helpers: mesh loading, nvdiffrast rendering, SE(3) utilities."""
import json
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr


def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int32)
    uv = np.asarray(m.visual.uv, np.float32)
    mat = m.visual.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    tex = np.asarray(img.convert("RGB"), np.float32) / 255.0
    return v, f, uv, tex


def load_camera(inputs):
    c = json.load(open(f"{inputs}/camera.json"))
    return np.array(c["K"]), np.array(c["T_world_cam"]), np.array(c["K_pipeline"]), c["width"], c["height"]


class Renderer:
    """Renders the textured mesh in a crop window of the full image (OpenCV camera)."""

    def __init__(self, v, f, uv, tex, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        uv = uv.copy(); uv[:, 1] = 1.0 - uv[:, 1]  # glTF uv (v up from bottom in trimesh) -> texture rows
        self.uv = torch.tensor(uv, device=device)
        # glTF uv origin is top-left; nvdiffrast texture lookup has v=0 at first row
        self.tex = torch.tensor(tex, device=device)[None].contiguous()

    def proj(self, K, x0, y0, w, h, near=0.05, far=10.0):
        # map camera coords -> clip coords for crop window [x0,x0+w) x [y0,y0+h)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] - x0, K[1, 2] - y0
        P = np.array([
            [2 * fx / w, 0, 2 * cx / w - 1, 0],
            [0, 2 * fy / h, 2 * cy / h - 1, 0],
            [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, 1, 0]], np.float32)
        return P

    def render(self, T_cam_mesh, K, x0, y0, w, h, need_rgb=True):
        """T_cam_mesh: (B,4,4) torch or numpy. Returns rgb (B,h,w,3), mask (B,h,w), depth (B,h,w).
        Row 0 of output = image row y0 (top)."""
        T = torch.as_tensor(np.asarray(T_cam_mesh, np.float32) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,
                            device=self.dev).float()
        if T.dim() == 2:
            T = T[None]
        P = torch.tensor(self.proj(K, x0, y0, w, h), device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)  # camera coords
        clip = torch.einsum("ij,bnj->bni", P, vc)
        # nvdiffrast output row 0 corresponds to NDC y=-1, which proj() maps to image row y0 (top): no flip.
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, resolution=[h, w])
        mask = rast[..., 3] > 0
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        depth = depth[..., 0] * mask
        rgb = None
        if need_rgb:
            uvi, uvd = dr.interpolate(self.uv[None], rast, self.f, rast_db=None, diff_attrs=None)
            rgb = dr.texture(self.tex, uvi, filter_mode="linear")
            rgb = rgb * mask[..., None]
        return rgb, mask, depth


# ---------------- SE(3)/SO(3) helpers ----------------
def rotvec_to_R(r):
    from scipy.spatial.transform import Rotation as Rr
    return Rr.from_rotvec(r).as_matrix()


def R_to_rotvec(R):
    from scipy.spatial.transform import Rotation as Rr
    return Rr.from_matrix(R).as_rotvec()


def rot_angle_deg(R1, R2):
    R = np.einsum("...ji,...jk->...ik", R1, R2)
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


# ---------------- image evidence ----------------
import torch.nn.functional as F


def blur_t(x, sigma):
    """x: (B,C,H,W) gaussian blur."""
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2); k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


class Scorer:
    """Scores candidate camera-frame poses against one video frame in a crop window."""

    def __init__(self, renderer, K, crop=(690, 545, 128, 112), ss=2, blur_sigma=1.0):
        self.R = renderer
        self.K = K
        self.x0, self.y0, self.w, self.h = crop
        self.ss = ss
        Ks = K.copy(); Ks[:2] *= ss
        Ks[0, 2] = (K[0, 2] + 0.5) * ss - 0.5; Ks[1, 2] = (K[1, 2] + 0.5) * ss - 0.5
        self.Ks = Ks
        self.blur_sigma = blur_sigma

    def set_frame(self, img_bgr, obj_mask, person_mask):
        x0, y0, w, h = self.x0, self.y0, self.w, self.h
        dev = self.R.dev
        im = img_bgr[y0:y0 + h, x0:x0 + w, ::-1].astype(np.float32) / 255.0
        self.img = torch.tensor(im.copy(), device=dev)  # h,w,3 RGB
        self.om = torch.tensor(obj_mask[y0:y0 + h, x0:x0 + w].copy(), device=dev).float()
        pm = person_mask[y0:y0 + h, x0:x0 + w].astype(np.uint8)
        import cv2
        pm = cv2.dilate(pm, np.ones((5, 5), np.uint8))
        self.valid = torch.tensor(1 - pm, device=dev).float()  # pixels not covered by person

    def render_batch(self, Tcm):
        ss = self.ss
        rgb, m, d = self.R.render(Tcm, self.Ks, self.x0 * ss, self.y0 * ss, self.w * ss, self.h * ss)
        rgb = F.avg_pool2d(rgb.permute(0, 3, 1, 2), ss)  # B,3,h,w
        m = F.avg_pool2d(m[:, None].float(), ss)[:, 0]
        rgb = blur_t(rgb, self.blur_sigma)
        return rgb, m

    def score(self, Tcm, chunk=64):
        out_iou, out_ncc = [], []
        Tcm = np.asarray(Tcm)
        for s in range(0, len(Tcm), chunk):
            rgb, m = self.render_batch(Tcm[s:s + chunk])
            v = self.valid[None]
            inter = (m * self.om[None] * v).sum((1, 2))
            union = ((m + self.om[None] - m * self.om[None]) * v).sum((1, 2))
            iou = inter / union.clamp(min=1)
            # NCC of colours inside (render mask & object mask & valid)
            wgt = ((m > 0.99).float() * self.om[None] * v)  # B,h,w
            img = self.img.permute(2, 0, 1)[None]  # 1,3,h,w
            n = wgt.sum((1, 2)).clamp(min=1)[:, None]
            mu_r = (rgb * wgt[:, None]).sum((2, 3)) / n
            mu_i = (img * wgt[:, None]).sum((2, 3)) / n
            dr_ = (rgb - mu_r[..., None, None]) * wgt[:, None]
            di_ = (img - mu_i[..., None, None]) * wgt[:, None]
            ncc = (dr_ * di_).sum((2, 3)) / ((dr_ ** 2).sum((2, 3)).sqrt() * (di_ ** 2).sum((2, 3)).sqrt() + 1e-6)
            out_iou.append(iou); out_ncc.append(ncc.mean(1))
        return torch.cat(out_iou).cpu().numpy(), torch.cat(out_ncc).cpu().numpy()

    def feat(self, x, kind):
        """x: (B,3,h,w) rgb -> feature maps."""
        if kind == "rgb":
            return x
        if kind == "dog":
            return blur_t(x, 0.7) - blur_t(x, 3.0)
        if kind == "dogg":
            g = x.mean(1, keepdim=True)
            return blur_t(g, 0.7) - blur_t(g, 3.0)
        if kind == "chroma":
            s = x.sum(1, keepdim=True) + 1e-3
            return x / s
        raise ValueError(kind)

    def score_feat(self, Tcm, kind="dog", chunk=64, erode=True):
        out_iou, out_ncc = [], []
        Tcm = np.asarray(Tcm)
        img = self.img.permute(2, 0, 1)[None]
        fi = self.feat(img, kind)
        for s in range(0, len(Tcm), chunk):
            rgb, m = self.render_batch(Tcm[s:s + chunk])
            v = self.valid[None]
            inter = (m * self.om[None] * v).sum((1, 2))
            union = ((m + self.om[None] - m * self.om[None]) * v).sum((1, 2))
            iou = inter / union.clamp(min=1)
            # composite render over the real background so that features at the silhouette are comparable
            comp = rgb + img * (1 - m[:, None])
            fr = self.feat(comp, kind)
            mm = (m > 0.5).float()
            if erode:
                mm = -F.max_pool2d(-mm[:, None], 3, 1, 1)[:, 0]
            wgt = mm * self.om[None] * v
            n = wgt.sum((1, 2)).clamp(min=1)[:, None]
            mu_r = (fr * wgt[:, None]).sum((2, 3)) / n
            mu_i = (fi * wgt[:, None]).sum((2, 3)) / n
            dr_ = (fr - mu_r[..., None, None]) * wgt[:, None]
            di_ = (fi - mu_i[..., None, None]) * wgt[:, None]
            ncc = (dr_ * di_).sum((2, 3)) / ((dr_ ** 2).sum((2, 3)).sqrt() * (di_ ** 2).sum((2, 3)).sqrt() + 1e-6)
            out_iou.append(iou); out_ncc.append(ncc.mean(1))
        return torch.cat(out_iou).cpu().numpy(), torch.cat(out_ncc).cpu().numpy()


def rot_about_axis(axis, theta):
    axis = axis / np.linalg.norm(axis)
    return rotvec_to_R(axis * theta)


def yaw_pose(Tcm, up_cam, center_mesh, theta):
    """Rotate object pose (camera frame) by theta about the vertical axis through its centre."""
    Rz = rot_about_axis(up_cam, theta)
    c = Tcm[:3, :3] @ center_mesh + Tcm[:3, 3]
    T = Tcm.copy()
    T[:3, :3] = Rz @ Tcm[:3, :3]
    T[:3, 3] = Rz @ (Tcm[:3, 3] - c) + c
    return T


# ---------------- differentiable fitting ----------------
def rotz_t(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


class DiffFitter:
    """Differentiable render-and-compare over a set of frames sharing one crop window."""

    def __init__(self, renderer, K, crop=(690, 545, 128, 112), ss=2):
        self.R = renderer
        self.x0, self.y0, self.w, self.h = crop
        self.ss = ss
        Ks = K.copy(); Ks[:2] *= ss
        Ks[0, 2] = (K[0, 2] + 0.5) * ss - 0.5; Ks[1, 2] = (K[1, 2] + 0.5) * ss - 0.5
        self.P = torch.tensor(renderer.proj(Ks, self.x0 * ss, self.y0 * ss, self.w * ss, self.h * ss), device=renderer.dev)

    def load_frames(self, imgs, oms, pms):
        import cv2
        x0, y0, w, h = self.x0, self.y0, self.w, self.h
        dev = self.R.dev
        I, O, V = [], [], []
        for im, om, pm in zip(imgs, oms, pms):
            I.append(im[y0:y0 + h, x0:x0 + w, ::-1].astype(np.float32) / 255.0)
            O.append(om[y0:y0 + h, x0:x0 + w].astype(np.float32))
            p = cv2.dilate(pm[y0:y0 + h, x0:x0 + w].astype(np.uint8), np.ones((5, 5), np.uint8))
            V.append(1.0 - p.astype(np.float32))
        self.img = torch.tensor(np.stack(I), device=dev).permute(0, 3, 1, 2).contiguous()
        self.om = torch.tensor(np.stack(O), device=dev)
        self.valid = torch.tensor(np.stack(V), device=dev)

    def render(self, Rcm, tcm):
        """Rcm (B,3,3), tcm (B,3) torch -> rgb (B,3,h,w), soft mask (B,h,w)."""
        R = self.R
        vc = torch.einsum("bij,nj->bni", Rcm, R.v) + tcm[:, None]
        vh = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
        clip = torch.einsum("ij,bnj->bni", self.P, vh).contiguous()
        H, W = self.h * self.ss, self.w * self.ss
        rast, _ = dr.rasterize(R.glctx, clip, R.f, resolution=[H, W])
        uvi, _ = dr.interpolate(R.uv[None], rast, R.f)
        rgb = dr.texture(R.tex, uvi, filter_mode="linear")
        m = (rast[..., 3:4] > 0).float()
        out = torch.cat([rgb * m, m], -1)
        out = dr.antialias(out, rast, clip, R.f)
        out = F.avg_pool2d(out.permute(0, 3, 1, 2), self.ss)
        return out[:, :3], out[:, 3]

    def losses(self, Rcm, tcm, idx, w_ncc=1.0, feat="dog"):
        rgb, m = self.render(Rcm, tcm)
        img, om, v = self.img[idx], self.om[idx], self.valid[idx]
        inter = (m * om * v).sum((1, 2))
        union = ((m + om - m * om) * v).sum((1, 2))
        iou = inter / union.clamp(min=1)
        comp = blur_t(rgb, 1.0) + img * (1 - m[:, None])
        if feat == "dog":
            fr = blur_t(comp, 0.7) - blur_t(comp, 3.0); fi = blur_t(img, 0.7) - blur_t(img, 3.0)
        else:
            fr, fi = comp, img
        wgt = (m.detach() > 0.5).float()
        wgt = -F.max_pool2d(-wgt[:, None], 3, 1, 1)[:, 0] * om * v
        n = wgt.sum((1, 2)).clamp(min=1)[:, None]
        mu_r = (fr * wgt[:, None]).sum((2, 3)) / n
        mu_i = (fi * wgt[:, None]).sum((2, 3)) / n
        a = (fr - mu_r[..., None, None]) * wgt[:, None]
        b = (fi - mu_i[..., None, None]) * wgt[:, None]
        ncc = (a * b).sum((2, 3)) / ((a ** 2).sum((2, 3)).sqrt() * (b ** 2).sum((2, 3)).sqrt() + 1e-6)
        return iou, ncc.mean(1)
