"""Rectify image evidence onto a plane of the mesh (e.g. table top / underside) in mesh coordinates, so that
texture from many frames can be pooled in the object frame and compared with the mesh texture under
different spin hypotheses."""
import numpy as np
import torch
import cv2

AX = np.array([0.006, 0.004])  # table axis (mesh xy), centre of top disk / column


def plane_grid(z0, half=0.40, res=0.002):
    n = int(round(2 * half / res))
    xs = (np.arange(n) + 0.5) * res - half
    X, Y = np.meshgrid(xs + AX[0], xs + AX[1])
    return np.stack([X, Y, np.full_like(X, z0)], -1)  # (n,n,3); rows=y, cols=x


def rectify(img, T_cam_mesh, K, P, depth=None, valid_mask=None, normal_sign=+1, min_facing=0.2):
    """img (H,W,C) float; P (n,n,3) mesh points; returns rectified (n,n,C) and weight (n,n)."""
    H, W = img.shape[:2]
    R, t = T_cam_mesh[:3, :3], T_cam_mesh[:3, 3]
    Pc = P @ R.T + t
    n = R[:, 2] * normal_sign
    facing = -(Pc @ n) / np.linalg.norm(Pc, axis=-1)
    q = Pc @ K.T
    u = (q[..., 0] / q[..., 2]).astype(np.float32)
    v = (q[..., 1] / q[..., 2]).astype(np.float32)
    out = cv2.remap(img.astype(np.float32), u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if out.ndim == 2:
        out = out[..., None]
    w = (facing > min_facing).astype(np.float32) * facing
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    w *= inside
    if depth is not None:  # visibility: rendered depth must equal the plane point depth
        w *= (np.abs(depth[vi, ui] - Pc[..., 2]) < 0.015)
    if valid_mask is not None:
        w *= valid_mask[vi, ui]
    return out, w


def rotate_grid(img, deg):
    """Rotate a rectified (n,n,C) image about the axis by deg (mesh frame, counter-clockwise in x->y)."""
    n = img.shape[0]
    c = (n / 2 - 0.5, n / 2 - 0.5)
    # rectified image: col=x, row=y; a mesh rotation by +deg about z maps (x,y) -> R(x,y)
    M = cv2.getRotationMatrix2D(c, -deg, 1.0)
    return cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_LINEAR)
