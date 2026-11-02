"""Mask-independent validation score: chamfer distance (px) from rendered occluding-contour edges
to image Canny edges, ignoring pixels near the person."""
import numpy as np
import cv2


def image_edge_dt(img_bgr, person, box, trunc=10.0):
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, 30, 80)
    dt = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)
    return np.minimum(dt, trunc)


def render_edges(depth):
    """Occluding contours: pixels where rendered depth jumps (incl. silhouette border)."""
    d = depth.copy()
    d[d <= 0] = 1e3
    gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
    gy = np.abs(np.diff(d, axis=0, prepend=d[:1]))
    return (np.maximum(gx, gy) > 0.05)


def edge_score(depth_full, img_bgr, person, obj=None, trunc=10.0):
    ys, xs = np.nonzero(depth_full > 0)
    if len(xs) < 20:
        return np.nan
    H, W = depth_full.shape
    x0, x1 = max(0, xs.min() - 20), min(W, xs.max() + 21)
    y0, y1 = max(0, ys.min() - 20), min(H, ys.max() + 21)
    dt = image_edge_dt(img_bgr, person, (x0, y0, x1, y1), trunc)
    e = render_edges(depth_full[y0:y1, x0:x1])
    occ = cv2.dilate(person[y0:y1, x0:x1].astype(np.uint8), np.ones((7, 7))) > 0
    if obj is not None:
        occ &= ~obj[y0:y1, x0:x1]
    sel = e & ~occ
    if sel.sum() < 10:
        return np.nan
    return float(dt[sel].mean())
