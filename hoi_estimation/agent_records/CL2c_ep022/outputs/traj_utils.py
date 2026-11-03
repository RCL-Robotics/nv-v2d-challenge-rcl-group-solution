"""Trajectory helpers: SO(3)/SE(3) smoothing, averaging, depth re-anchoring."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mean(Rs, w=None):
    q = Rot.from_matrix(Rs).as_quat()
    w = np.ones(len(q)) if w is None else np.asarray(w, float)
    q = q * np.sign(q @ q[len(q) // 2])[:, None]
    M = (q * w[:, None]).T @ q
    return Rot.from_quat(np.linalg.eigh(M)[1][:, -1]).as_matrix()


def smooth_poses(T, sigma, center=None, weights=None):
    """Gaussian smoothing of a (F,4,4) pose sequence. Rotation: weighted quaternion mean;
    translation of the point `center` (mesh coords) smoothed linearly. weights: per-frame confidence."""
    F = len(T)
    c = np.zeros(3) if center is None else np.asarray(center)
    P = np.einsum('fij,j->fi', T[:, :3, :3], c) + T[:, :3, 3]
    wts = np.ones(F) if weights is None else np.asarray(weights, float)
    out = T.copy()
    r = max(1, int(3 * sigma))
    for i in range(F):
        lo, hi = max(0, i - r), min(F, i + r + 1)
        g = np.exp(-0.5 * ((np.arange(lo, hi) - i) / sigma) ** 2) * wts[lo:hi]
        Rm = rot_mean(T[lo:hi, :3, :3], g)
        pm = (g[:, None] * P[lo:hi]).sum(0) / g.sum()
        out[i, :3, :3] = Rm
        out[i, :3, 3] = pm - Rm @ c
    return out


def rot_angle_deg(Ra, Rb):
    return np.degrees(Rot.from_matrix(np.einsum('...ij,...kj->...ik', Ra, Rb)).magnitude())


def reanchor_depth(Tc, Tc_ref):
    """Scale camera-frame positions along their viewing rays so that median z(Tc) == median z(Tc_ref)."""
    s = np.median(Tc_ref[:, 2, 3]) / np.median(Tc[:, 2, 3])
    out = Tc.copy()
    out[:, :3, 3] *= s
    return out, s
