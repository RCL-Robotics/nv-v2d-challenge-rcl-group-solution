import numpy as np, roma, torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
def smooth_poses(T, times, query, sigma):
    """Gaussian kernel regression of poses (camera frame) sampled at `times`, evaluated at `query`.
    translation: weighted mean of object centre; rotation: weighted chordal mean."""
    T = np.asarray(T); out = np.zeros((len(query), 4, 4)); out[:, 3, 3] = 1
    for k, q in enumerate(query):
        w = np.exp(-0.5 * ((np.asarray(times) - q) / sigma) ** 2) if sigma > 0 else (np.asarray(times) == q).astype(float)
        w = w / w.sum()
        Rs = (w[:, None, None] * T[:, :3, :3]).sum(0)
        U, _, Vt = np.linalg.svd(Rs); R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        out[k, :3, :3] = R; out[k, :3, 3] = (w[:, None] * T[:, :3, 3]).sum(0)
    return out
