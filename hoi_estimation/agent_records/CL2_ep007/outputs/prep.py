"""Build per-frame crops (RGB, iron mask, person mask) around the iron."""
import numpy as np
import cv2
import h5py
from fitcore import make_rois, crop_resize

HW = (160, 160)


def bbox(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def build(video, iron_full_fn, person_h5, F, hw=HW):
    """iron_full_fn(i) -> full-res bool mask. Returns dict of arrays."""
    cap = cv2.VideoCapture(video)
    with h5py.File(person_h5) as h:
        P = h["person"]
        rgbs, ms, ps, rois, boxes = [], [], [], [], []
        last = None
        for i in range(F):
            ok, fr = cap.read()
            m = iron_full_fn(i)
            b = bbox(m)
            if b is None:
                b = last
            last = b
            boxes.append(b)
            roi = make_rois([b])[0]
            rois.append(roi)
            pm = P[i]
            rgbs.append(crop_resize(fr[..., ::-1].astype(np.float32) / 255, roi, hw, cv2.INTER_AREA))
            ms.append(crop_resize(m.astype(np.float32), roi, hw, cv2.INTER_AREA))
            ps.append(crop_resize(pm.astype(np.float32), roi, hw, cv2.INTER_AREA))
    return dict(rgb=np.array(rgbs, np.float32), mask=np.array(ms, np.float32), person=np.array(ps, np.float32),
                roi=np.array(rois), box=np.array(boxes, np.float64))


def compute_touch(M, crop, person_h5):
    """Number of person-mask pixels within ~7 px of the iron mask, per frame (0 = untouched)."""
    x0, y0, x1, y1 = crop
    out = np.zeros(len(M), np.int64)
    k = np.ones((15, 15), np.uint8)
    with h5py.File(person_h5) as h:
        P = h["person"]
        for i in range(len(M)):
            dil = cv2.dilate(M[i].astype(np.uint8), k) > 0
            out[i] = int((P[i, y0:y1, x0:x1] & dil).sum())
    return out
