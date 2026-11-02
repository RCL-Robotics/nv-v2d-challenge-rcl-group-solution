"""Shared NOISY-ONLY helpers for the Tier-2 Kalman (RTS) study.

Reads ONLY: the Tier-2 dataset directory (parquet, meta, mesh, ground_plane, support_surface) and the Tier-2
retargeted G1 data (human_motion_data_tier2, produced by soma_to_g1.py from the Tier-2 input). Never reads Tier 1.
Rest detection (static_mask, psd_floor) is copied verbatim from filter_study/noise/scripts/filter_apply.py (ZUPT v1).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal
from scipy.spatial.transform import Rotation as R

ME = Path("/root/autodl-tmp/haoran.zhang")
T2 = ME / "data/video_to_data_challenge/track_2/tier_2_synthetic_noise"
G1_T2 = ME / "data/human_motion_data_tier2"
KOUT = ME / "tmp_analysis/tier2/kalman"
FPS = 30.0
DT = 1.0 / FPS

# ---- ZUPT v1 constants (filter_apply.py)
CHI2_3_99 = 11.344867
H = 7
LMIN = 10
BAND = (7.5, 15.0)

# ---- SOMA joint indices (verified against the payload joint names at runtime)
J = {"Hips": 0, "LeftHand": 14, "RightHand": 42, "LeftFoot": 69, "LeftToeBase": 70, "RightFoot": 74, "RightToeBase": 75}
FINGER_ENDS = {"left": [18, 23, 28, 33, 38], "right": [46, 51, 56, 61, 66]}


def metas():
    return [json.loads(l) for l in (T2 / "meta/episodes_metadata.jsonl").read_text().splitlines() if l.strip()]


def load_raw(root: Path, ep: int) -> dict:
    t = pq.read_table(root / f"data/chunk-000/episode_{ep:06d}.parquet")
    d = {}
    for n in t.schema.names:
        col = t.column(n).combine_chunks()
        if pa.types.is_fixed_size_list(col.type):
            d[n] = np.asarray(col.flatten().to_numpy(zero_copy_only=False)).reshape(len(col), col.type.list_size)
        else:
            d[n] = np.asarray(col.to_numpy(zero_copy_only=False))
    return d


def wxyz_to_rot(q):
    q = np.asarray(q, float)
    return R.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4))


def fill_invisible(p, q_wxyz, vis):
    p, q = p.copy(), q_wxyz.copy()
    if vis.all():
        return p, q
    good = np.flatnonzero(vis)
    for t in np.flatnonzero(~vis):
        j = good[np.argmin(np.abs(good - t))]
        p[t], q[t] = p[j], q[j]
    return p, q


def object_track(ep):
    """Noisy Tier-2 object track in the source OpenCV world: p (T,3) cm, rot (T), vis (T,) bool (invisible filled)."""
    b = load_raw(T2, ep)
    vis = b["observation.object.visible"].astype(bool)
    p, q = fill_invisible(b["observation.object.pose"][:, :3].astype(float), b["observation.object.pose"][:, 3:].astype(float), vis)
    return p * 100.0, wxyz_to_rot(q), vis, b


def psd_floor(inc):
    f, p = signal.welch(inc, fs=FPS, nperseg=min(128, len(inc)), axis=0, detrend="constant")
    p = p.mean(axis=1)
    sel = (f >= BAND[0]) & (f <= BAND[1])
    return f, p, float(np.median(p[sel]))


def noise_sigmas(p_cm, rot):
    """ZUPT v1 per-episode step sigmas from the velocity-spectrum floor: (cm/frame/axis, deg/frame/axis)."""
    inc_p = np.diff(p_cm, axis=0)
    inc_r = np.degrees((rot[:-1].inv() * rot[1:]).as_rotvec())
    _, _, fl_p = psd_floor(inc_p)
    _, _, fl_r = psd_floor(inc_r)
    return float(np.sqrt(fl_p * FPS / 2.0)), float(np.sqrt(fl_r * FPS / 2.0))


def static_mask(p_cm, rot, sig_p, sig_r_rad, rot_test=True):
    """Verbatim ZUPT v1 rule (filter_apply.static_mask); rot_test=False drops the rotation test (position-only)."""
    T = len(p_cm)
    lo = np.maximum(np.arange(T) - H, 0)
    hi = np.minimum(np.arange(T) + H, T - 1)
    span = (hi - lo).astype(float)
    dpos = np.linalg.norm(p_cm[hi] - p_cm[lo], axis=1)
    thr_p = np.sqrt(CHI2_3_99 * span) * sig_p
    s = dpos < thr_p
    if rot_test:
        dang = (rot[lo].inv() * rot[hi]).magnitude()
        thr_r = np.sqrt(CHI2_3_99 * span) * sig_r_rad
        s &= dang < thr_r
    st_idx = np.flatnonzero(s)
    for a, b in zip(st_idx[:-1], st_idx[1:]):
        if 1 < b - a <= 2 * H + 1:
            span_ = float(b - a)
            ok = np.linalg.norm(p_cm[b] - p_cm[a]) < np.sqrt(CHI2_3_99 * span_) * sig_p
            if rot_test:
                ok = ok and (rot[a].inv() * rot[b]).magnitude() < np.sqrt(CHI2_3_99 * span_) * sig_r_rad
            if ok:
                s[a:b + 1] = True
    return keep_runs(s, LMIN)


def keep_runs(s, lmin):
    out = np.zeros(len(s), bool)
    for a, b in runs(s):
        if b - a + 1 >= lmin:
            out[a:b + 1] = True
    return out


def runs(s):
    """[(start, end_inclusive)] of True runs."""
    s = np.asarray(s, bool)
    out, t, T = [], 0, len(s)
    while t < T:
        if s[t]:
            u = t
            while u + 1 < T and s[u + 1]:
                u += 1
            out.append((t, u))
            t = u + 1
        else:
            t += 1
    return out


def object_rest_mask(p_cm, rot):
    sp, sr = noise_sigmas(p_cm, rot)
    return static_mask(p_cm, rot, sp, np.radians(sr)), sp, sr


def load_g1(seq):
    """Tier-2 retargeted G1 parquet + its source payload (SOMA joints from the Tier-2 human input)."""
    t = pq.read_table(G1_T2 / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet")
    g = lambda n: t.column(n)[0].as_py()
    pay = pickle.loads(g("source_payload"))
    return {"obj_p": np.asarray(g("object_body_position"), float)[:, 0], "obj_q": np.asarray(g("object_body_wxyz"), float)[:, 0],
            "contact": np.asarray(g("hand_contact_active"), float), "hand_sides": list(g("hand_sides")),
            "ee": np.asarray(g("ee_pose_w"), float),
            "soma_joints": np.asarray(pay["soma_joints"], float), "soma_joints_wxyz": np.asarray(pay["soma_joints_wxyz"], float),
            "src_root_t": np.asarray(pay["source_root_translation"], float), "src_root_wxyz": np.asarray(pay["source_root_wxyz"], float),
            "ground_alignment": pay["ground_alignment"]}


def kabsch(A, B):
    """Rigid (Rot, t) minimising |Rot A + t - B| (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, cb - Rm @ ca


def ground_plane_cv(meta):
    """(up unit vector, e) with up.x + e = height above the fitted floor (metres), OpenCV world; sign from the human."""
    pl = np.asarray(json.loads((T2 / meta["ground_plane"]).read_text())["plane"], float)
    n, d = pl[:3], pl[3]
    s = np.linalg.norm(n)
    return n / s, d / s


def support_is_empty(meta):
    txt = (T2 / meta["support_surface"]).read_text()
    return txt.count("def ") <= 1          # only the root Xform -> valid empty (ground-only) stage


def load_mesh_vertices(obj_name):
    """Object-frame vertices exactly as the kit loads them (trimesh force='mesh', process=False), metres."""
    import trimesh
    path = T2 / "mesh" / obj_name / f"{obj_name}.glb"
    m = trimesh.load(str(path), force="mesh", process=False)
    return m
