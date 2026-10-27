"""Render-and-match registration: render the textured mesh at a hypothesis pose, match the render to the
video frame with SuperPoint/ALIKED + LightGlue, lift render keypoints to 3D with the rendered depth, and
solve PnP-RANSAC for the pose that explains the real image."""
import os
import sys
import numpy as np
import torch
import cv2

sys.path.insert(0, "/work/tools/LightGlue")
from lightglue import LightGlue, SuperPoint, ALIKED  # noqa: E402

# offline weights (downloaded once into /work/tools, see TOOLS.md)
torch.hub.set_dir("/work/tools/torch_home/hub")


class Matcher:
    def __init__(self, kind="superpoint", n_kp=2048, dev="cuda"):
        self.dev = dev
        if kind == "superpoint":
            self.ext = SuperPoint(max_num_keypoints=n_kp).eval().to(dev)
        else:
            self.ext = ALIKED(max_num_keypoints=n_kp).eval().to(dev)
        self.lg = LightGlue(features=kind).eval().to(dev)

    @torch.no_grad()
    def match(self, img0, img1):
        """img: float tensors (3,H,W) in [0,1]. Returns (k0, k1, score) numpy."""
        f0 = self.ext.extract(img0.to(self.dev))
        f1 = self.ext.extract(img1.to(self.dev))
        out = self.lg({"image0": f0, "image1": f1})
        m = out["matches"][0].cpu().numpy()
        sc = out["scores"][0].cpu().numpy()
        k0 = f0["keypoints"][0].cpu().numpy()[m[:, 0]]
        k1 = f1["keypoints"][0].cpu().numpy()[m[:, 1]]
        return k0, k1, sc


def crop_box(mask, margin=0.15, min_size=160):
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    s = max(y1 - y0, x1 - x0, min_size) * (1 + 2 * margin)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    return int(cx - s / 2), int(cy - s / 2), int(s)


def to_tensor_crop(img, box, size):
    x0, y0, s = box
    H, W = img.shape[:2]
    # affine crop (pads with zeros outside)
    A = np.array([[size / s, 0, -x0 * size / s], [0, size / s, -y0 * size / s]], np.float32)
    c = cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_AREA if s > size else cv2.INTER_LINEAR)
    return torch.tensor(c, dtype=torch.float32).permute(2, 0, 1), A


def register(matcher, renderer, image, obj_mask, person_mask, T_cam_mesh, K, size=640,
             ransac_px=3.0, min_inl=12):
    """image: (H,W,3) uint8 RGB full-res; masks full-res bool. Returns dict(T, n_inl, n_match, med_disp)."""
    out = renderer.render(T_cam_mesh[None], rgb=True, depth=True)
    rm = out["mask"][0].cpu().numpy()
    rgb = out["rgb"][0].cpu().numpy()
    dep = out["depth"][0].cpu().numpy()
    if rm.sum() < 50 or obj_mask.sum() < 50:
        return None
    box = crop_box(rm | obj_mask)
    real = image.astype(np.float32) / 255.0
    keep = cv2.dilate(obj_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    real = real * keep[..., None]
    t_real, A = to_tensor_crop(real, box, size)
    t_rend, _ = to_tensor_crop(rgb.astype(np.float32), box, size)
    k_rend, k_real, sc = matcher.match(t_rend, t_real)
    if len(k_rend) < min_inl:
        return dict(T=None, n_inl=0, n_match=len(k_rend), med_disp=np.nan)
    s = A[0, 0]
    pr = k_rend / s + np.array([box[0], box[1]])
    pq = k_real / s + np.array([box[0], box[1]])
    # lift render points to camera 3D using rendered depth (nearest pixel)
    ui = np.clip(np.round(pr[:, 0]).astype(int), 0, rm.shape[1] - 1)
    vi = np.clip(np.round(pr[:, 1]).astype(int), 0, rm.shape[0] - 1)
    z = dep[vi, ui]
    ok = (z > 0) & rm[vi, ui]
    # drop real points on the person (occluder) unless also on the object mask
    ok &= ~(person_mask[np.clip(np.round(pq[:, 1]).astype(int), 0, rm.shape[0] - 1),
                        np.clip(np.round(pq[:, 0]).astype(int), 0, rm.shape[1] - 1)]
            & ~obj_mask[np.clip(np.round(pq[:, 1]).astype(int), 0, rm.shape[0] - 1),
                        np.clip(np.round(pq[:, 0]).astype(int), 0, rm.shape[1] - 1)])
    if ok.sum() < min_inl:
        return dict(T=None, n_inl=0, n_match=int(ok.sum()), med_disp=np.nan)
    pr, pq, z = pr[ok], pq[ok], z[ok]
    Ki = np.linalg.inv(K)
    Xc = (np.c_[pr, np.ones(len(pr))] @ Ki.T) * z[:, None]
    Tinv = np.linalg.inv(T_cam_mesh)
    Xm = Xc @ Tinv[:3, :3].T + Tinv[:3, 3]
    rv0, _ = cv2.Rodrigues(T_cam_mesh[:3, :3])
    okp, rv, tv, inl = cv2.solvePnPRansac(Xm.astype(np.float64), pq.astype(np.float64), K, None,
                                          rvec=rv0.copy(), tvec=T_cam_mesh[:3, 3].copy().reshape(3, 1),
                                          useExtrinsicGuess=True, reprojectionError=ransac_px,
                                          iterationsCount=500, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    disp = np.linalg.norm(pr - pq, axis=1)
    if not okp or inl is None or len(inl) < min_inl:
        return dict(T=None, n_inl=0 if inl is None else len(inl), n_match=len(pr), med_disp=float(np.median(disp)))
    inl = inl[:, 0]
    rv, tv = cv2.solvePnPRefineLM(Xm[inl], pq[inl], K, None, rv, tv)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rv)[0]
    T[:3, 3] = tv[:, 0]
    return dict(T=T, n_inl=len(inl), n_match=len(pr), med_disp=float(np.median(disp[inl])),
                Xm=Xm[inl], uv=pq[inl])
