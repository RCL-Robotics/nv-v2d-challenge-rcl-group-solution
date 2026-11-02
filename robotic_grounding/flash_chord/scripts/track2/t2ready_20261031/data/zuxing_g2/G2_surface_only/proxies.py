"""Tier-2-only quality proxies of a pose trajectory (no reference data)."""
import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_proxies(E, x, grasp):
    """floor / support-box penetration on the visible frames and the grasp-frame hand-to-object distance (median over
    the grasp frames of both hands of the median over the 14 grip points of the distance to the mesh surface), with the
    measured hand joints and (hand_d_corr) with the hand joints corrected by the human drift of x."""
    vis = E.vis
    h, _ = E.heights(x, E.HULL); hmin = h.min(1)[vis]
    out = dict(floor_min_cm=float(hmin.min()), floor_share_below_1cm=float((hmin < -1.0).mean()))
    if E.BOXES:
        _, W = E.heights(x, E.BPTS); pen, _ = E.box_terms(W[vis])
        out.update(box_pen_max_cm=float(pen.max()), box_share_pen_gt1cm=float((pen > 1.0).mean()))
    else:
        out.update(box_pen_max_cm=np.nan, box_share_pen_gt1cm=np.nan)
    x_meas = x.copy(); x_meas[:, 6:] = 0.0
    for key, xx in (('hand_d_cm', x_meas), ('hand_d_corr_cm', x)):
        ds = []
        for s in (0, 1):
            fr = np.flatnonzero(grasp[s] & vis)
            if len(fr):
                Xo = E.grip_obj(xx, s)[fr]
                d, _ = E.tree.query(Xo.reshape(-1, 3))
                ds.append(np.median(d.reshape(len(fr), -1), axis=1))
        out[key] = float(np.median(np.concatenate(ds))) if ds else np.nan
    return out


def correction(E, x0, x1):
    """implied correction of x1 relative to x0 on the visible frames: position (cm) and rotation (deg) p50/p95/max and
    the per-frame step RMS of the correction."""
    v = E.vis
    p0, R0 = E.pose(x0); p1, R1 = E.pose(x1)
    dp = p1[v] - p0[v]; dpn = np.linalg.norm(dp, axis=1)
    C = R1[v] * R0[v].inv(); ang = np.degrees(C.magnitude())
    out = {}
    for k, a in (('corr_pos_cm', dpn), ('corr_rot_deg', ang)):
        out[k + '_p50'], out[k + '_p95'], out[k + '_max'] = (float(q) for q in np.percentile(a, [50, 95, 100]))
    out['corr_step_rms_pos_cm'] = float(np.sqrt((np.linalg.norm(np.diff(dp, axis=0), axis=1) ** 2).mean()))
    out['corr_step_rms_rot_deg'] = float(np.sqrt((np.degrees((C[1:] * C[:-1].inv()).magnitude()) ** 2).mean()))
    return out


def pose_diff(pts_cm, a, b):
    """two (T,7) pose arrays (m, xyzw): max rotation difference (deg) and mean vertex distance (cm) over pts_cm."""
    Ra, Rb = R.from_quat(a[:, 3:]), R.from_quat(b[:, 3:])
    ang = np.degrees((Ra.inv() * Rb).magnitude())
    Va = np.einsum('tij,kj->tki', Ra.as_matrix(), pts_cm) + a[:, None, :3] * 100
    Vb = np.einsum('tij,kj->tki', Rb.as_matrix(), pts_cm) + b[:, None, :3] * 100
    vd = np.linalg.norm(Va - Vb, axis=2).mean(1)
    return dict(rot_max_deg=float(ang.max()), rot_p50_deg=float(np.median(ang)), vert_mean_cm=float(vd.mean()),
                vert_p95_cm=float(np.percentile(vd, 95)), vert_max_cm=float(vd.max()),
                pos_max_cm=float(np.linalg.norm(a[:, :3] - b[:, :3], axis=1).max() * 100))
