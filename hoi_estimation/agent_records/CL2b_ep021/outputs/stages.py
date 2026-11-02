"""Pipeline stages used by solve.py."""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot


def static_segments(gray, obj, per, k=3, thr=1.2, min_len=15):
    """Photometric static-phase detector.
    For each pair (i, i+k) compute the mean absolute grey-level difference (blurred, half-res) over the
    dilated object-mask region, excluding (dilated) person pixels. A frame is static if every pair
    touching it from the left window (i-k..i) is below `thr` grey levels (noise floor here is 0.3-0.9;
    moving phases give 2-20). Returns per-frame bool and list of (start, end, is_static)."""
    F = len(gray)
    ker = np.ones((7, 7), np.uint8)
    s = np.zeros(F)
    g = gray.astype(np.float32)
    for i in range(F - k):
        reg = cv2.dilate((obj[i] | obj[i + k]).astype(np.uint8), ker) > 0
        pv = cv2.dilate((per[i] | per[i + k]).astype(np.uint8), ker) > 0
        r = reg & ~pv
        s[i] = np.abs(g[i] - g[i + k])[r].mean() if r.sum() > 50 else 99.0
    s[F - k:] = s[F - k - 1]
    mv = s > thr
    st = np.array([not mv[max(0, i - k):i + 1].any() for i in range(F)])
    segs = []
    i = 0
    while i < F:
        j = i
        while j < F and st[j] == st[i]:
            j += 1
        segs.append([i, j, bool(st[i])])
        i = j
    # static runs shorter than min_len are treated as moving; merge neighbours
    for sgm in segs:
        if sgm[2] and sgm[1] - sgm[0] < min_len:
            sgm[2] = False
    merged = []
    for sgm in segs:
        if merged and merged[-1][2] == sgm[2]:
            merged[-1][1] = sgm[1]
        else:
            merged.append(list(sgm))
    st = np.zeros(F, bool)
    for a, b, f in merged:
        st[a:b] = f
    return st, [tuple(m) for m in merged], s


def chordal_mean(R):
    U, _, Vt = np.linalg.svd(R.sum(0))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        M = U @ np.diag([1, 1, -1]) @ Vt
    return M


def rot_about_mesh_axis(axis, deg):
    v = np.zeros(3); v["xyz".index(axis)] = np.radians(deg)
    return Rot.from_rotvec(v).as_matrix()


def depth_anchor(p_new, p_ref):
    """Scale camera-frame centres along their viewing rays so that median z matches the reference.
    p_new, p_ref (N,3) camera-frame points of the same frames. Returns scaled points and factor."""
    f = np.median(p_ref[:, 2]) / np.median(p_new[:, 2])
    return p_new * f, f
