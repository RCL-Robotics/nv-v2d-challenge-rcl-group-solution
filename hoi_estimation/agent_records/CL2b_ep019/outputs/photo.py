"""Photometric scoring: render the textured mesh and compare with the video frame (gain-normalised)."""
import numpy as np
import torch
import torch.nn.functional as Fn


def photo_scores(R, K, img, obj_mask, person_mask, Ts, crop=None, margin=40):
    """img (H,W,3) uint8; Ts (B,4,4) camera poses for this frame. Returns (B,) score (lower better):
    mean robust colour residual over the union of rendered mask and object mask (person pixels excluded
    unless inside object mask), after a per-hypothesis least-squares gain per channel."""
    ys, xs = np.nonzero(obj_mask)
    if len(xs) == 0:
        return np.zeros(len(Ts))
    x0, x1 = max(0, xs.min() - margin), min(img.shape[1], xs.max() + margin)
    y0, y1 = max(0, ys.min() - margin), min(img.shape[0], ys.max() + margin)
    w, h = x1 - x0, y1 - y0
    o = R.render(K, Ts, w=w, h=h, x0=x0, y0=y0)
    m = o['mask'].float()
    rgb = o['rgb']
    I = torch.tensor(img[y0:y1, x0:x1], device=m.device).float()[None] / 255.
    om = torch.tensor(obj_mask[y0:y1, x0:x1], device=m.device)[None].float()
    pm = torch.tensor(person_mask[y0:y1, x0:x1], device=m.device)[None].float()
    region = torch.clamp(m + om, 0, 1) * (1 - pm * (1 - om))
    # background model: where not rendered, predict the median floor colour around (ring) -> use image blurred
    # outside the SAM mask (inpainting proxy)
    bg = I.clone()
    k = 15
    msk = 1 - om
    num = Fn.avg_pool2d((I * msk[..., None]).permute(0, 3, 1, 2), k, 1, k // 2)
    den = Fn.avg_pool2d(msk[:, None], k, 1, k // 2).clamp_min(1e-3)
    bg = (num / den).permute(0, 2, 3, 1)
    for _ in range(3):  # fill holes of large masks
        num = Fn.avg_pool2d((bg * (den.permute(0, 2, 3, 1) > 0.05)).permute(0, 3, 1, 2), k, 1, k // 2)
        den2 = Fn.avg_pool2d((den > 0.05).float(), k, 1, k // 2).clamp_min(1e-3)
        bg = torch.where(den.permute(0, 2, 3, 1) > 0.05, bg, (num / den2).permute(0, 2, 3, 1))
        den = torch.maximum(den, den2 * (den2 > 0.05))
    # gain per hypothesis/channel on rendered pixels
    num = (rgb * I * m[..., None]).sum((1, 2))
    den = (rgb * rgb * m[..., None]).sum((1, 2)).clamp_min(1e-6)
    g = (num / den).clamp(0.3, 3.0)
    pred = torch.where(m[..., None] > 0, rgb * g[:, None, None], bg)
    res = (pred - I).abs().sum(-1).clamp(max=0.6)
    s = (res * region).sum((1, 2)) / region.sum((1, 2)).clamp_min(1)
    return s.cpu().numpy()
