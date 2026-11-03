import numpy as np
from core import rot_angle_deg
def viterbi(R, t, L, alpha=20., th0=8., tc0=0.06, extra_unary=None):
    """R (F,G,3,3), t (F,G,3), L (F,G). returns best path indices"""
    F, G = L.shape
    U = alpha * L if extra_unary is None else alpha * L + extra_unary
    acc = U[0].copy(); bp = np.zeros((F, G), int)
    for f in range(1, F):
        dth = rot_angle_deg(R[f - 1][:, None], R[f][None, :])  # G,G
        dc = np.linalg.norm(t[f - 1][:, None] - t[f][None, :], axis=-1)
        Tc = dth / th0 + dc / tc0
        tot = acc[:, None] + Tc
        bp[f] = tot.argmin(0); acc = tot.min(0) + U[f]
    path = np.zeros(F, int); path[-1] = acc.argmin()
    for f in range(F - 1, 0, -1): path[f - 1] = bp[f, path[f]]
    return path
