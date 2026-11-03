"""Temporal decoding (Viterbi) over per-frame pose hypotheses."""
import numpy as np
from fit import CENTER


def centres(T):
    return np.einsum('...ij,j->...i', T[..., :3, :3], CENTER) + T[..., :3, 3]


def rot_dist(Ra, Rb):
    """pairwise angle (deg) between rotation sets Ra (A,3,3), Rb (B,3,3) -> (A,B)."""
    M = np.einsum('aji,bjk->abik', Ra, Rb)
    tr = np.trace(M, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def viterbi(cands, data_cost, sig_rot=6.0, sig_tr=0.04, static=None, w_static=10.0, gap=None):
    """cands: list over frames of (n_f,4,4) arrays; data_cost: list of (n_f,) arrays.
    Transition cost = |dR|/sig_rot + |dc|/sig_tr (scaled up w_static times inside static phases).
    gap: optional per-frame multiplier list (frames apart for transitions); returns chosen indices."""
    F = len(cands)
    acc = [np.asarray(data_cost[0], float)]
    back = []
    for f in range(1, F):
        A, B = cands[f - 1], cands[f]
        dr = rot_dist(A[:, :3, :3], B[:, :3, :3])
        dc = np.linalg.norm(centres(A)[:, None] - centres(B)[None], axis=-1)
        s = w_static if (static is not None and static[f] and static[f - 1]) else 1.0
        trans = s * (dr / sig_rot + dc / sig_tr)
        tot = acc[-1][:, None] + trans
        bi = tot.argmin(0)
        back.append(bi)
        acc.append(tot[bi, np.arange(len(B))] + np.asarray(data_cost[f], float))
    path = [int(acc[-1].argmin())]
    for f in range(F - 2, -1, -1):
        path.append(int(back[f][path[-1]]))
    return path[::-1]
