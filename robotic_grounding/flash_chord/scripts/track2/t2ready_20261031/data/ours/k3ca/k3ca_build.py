#!/usr/bin/env python3
"""k3ca_build.py (NOISY-ONLY, CPU only): K3CA = contact-aware K3fix for the 7 Tier-2 episodes that FAIL the K3fix input
gates (00 05 15 19 22 26 28; tmp_analysis/tier2/k3full/build/gates_all30.json). Steps 1-2 of the K3CA build.
Run:  CUDA_VISIBLE_DEVICES= nice -n 19 conda_envs/v2d/bin/python -I -B scripts/k3ca_build.py tracks|gates

K3CA(t) = the point at weight lam(t) on the geodesic from the K3fix pose to the released (raw) Tier-2 pose:
    position  (1 - lam) p_K3fix + lam p_raw          rotation  R_K3fix exp(lam log(R_K3fix^-1 R_raw))
  lam(t)     = max_k lam_req(k) c(|t - k|),  c(d) = (1 + cos(pi d / (F + 1))) / 2 for d <= F, else 0
               (F = 2H + 1 = 15 frames, the ZUPT v1 bridge constant; lam is continuous, every pull fades in/out over 0.5 s)
  lam_req(k) = smallest value on the grid 0, 0.1, ..., 1 that satisfies BOTH
     contact  if frame k is a raw-contact frame (G4 definition: raw object < 3 cm from a hand point = wrist + 5 finger
              ends, either hand, Tier-2 SOMA), the blended object's hand-point distance <= raw distance + 1.5 cm
              (= half the G4 tolerance of 3 cm)
     support  if frame k is visible, the blended object's penetration depth (floor + support stage, G1 training world,
              G2 recipe) <= raw depth + 2.31 cm (= half the G1/G2 envelope of 4.62 cm)
  verification loop: after smoothing, any frame that still violates either requirement gets lam_req = lam + 0.1
  (capped at 1) and lam is recomputed (lam = 1 reproduces raw exactly, so the loop ends in <= 11 rounds).
  lam = 0 frames are byte-for-byte the K3fix track. No clean data, no new free constant (3 cm and 4.62 cm are the gates'
  own Tier-2-only thresholds, F is the existing ZUPT bridge constant, the 0.5 margins are fixed here before any gate run).

Why not the rigid hand attachment of the design sketch (exploration in tmp_analysis/tier2/k3ca/explore/, noisy-only):
the raw hand->object relative pose is far from constant inside the raw-contact windows (spread up to 50 cm / 116 deg),
so attaching the object with one relative transform per window moves it 9-53 cm from raw (G1 fails on all 7 episodes,
64-392 frames); local hand-frame smoothing fails G1 for half-widths >= 15 frames and doubles the position jerk. The
blend above is the minimal departure from K3fix that keeps the raw hand contact; G1/G2/G4 hold by construction:
  G1  the blend lies on the K3fix->raw geodesic, so |K3CA - raw| <= |K3fix - raw| per frame (K3fix passes G1 on all 7)
  G2  support requirement at half the gate threshold      G4  contact requirement at half the gate tolerance
  G3  the K3fix smoother is used unchanged (nothing new is forced; the 7 episodes have no flagged rest run)

tracks  -> tmp_analysis/tier2/k3ca/build/k3ca_tracks.npz (K3CAfull__<ep> (T,7) [x y z qx qy qz qw] metres, released
           OpenCV world, every frame; K3CApose__<ep> visible frames; vis__<ep>; lam__<ep>), build/tracks.json
           (per episode: raw-contact windows, hands, noisy relative-transform spread, lam statistics, smoothness).
gates   -> build/gates.json, build/gates.txt: G1-G4 for K3CA with the gate formulas of scripts/k3fix_build.py gate_work
           (sha256 364d27ed, imported read-only; its helper functions are called, the per-track formulas are copied
           verbatim because gate_work reads its tracks from fixed files). Replication check: the copied formulas
           applied to the K3fix and frozen-K3 tracks must reproduce gates_all30.json's records for these 7 episodes.
Reads no clean data (no Tier-1 path, no *LOCAL_NOT_FOR_UPLOAD* file). Writes only tmp_analysis/tier2/k3ca/build/.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
OUT = ME / "tmp_analysis/tier2/k3ca/build"
KB_PY = ME / "scripts/k3fix_build.py"
KB_SHA = "364d27ed1b0e0a250456ff2fa42b130c7f73bfb4755f88c75237dc79148dff7b"
GATES30 = ME / "tmp_analysis/tier2/k3full/build/gates_all30.json"
GATES30_SHA = "3a741669592cf62792ef97f6bd0d95a4b0a0f03b25967c7b7ec680dd613ac4ab"
K3FIX_NPZ = ME / "tmp_analysis/tier2/k3fix_pilot/build/k3fix_tracks.npz"
K3FIX_NPZ_SHA = "1a4bb14c9da05ea322c5d04947cbca074905f46596d20a511c3e8fd5a1221f7e"
K3_FROZEN_NPZ = ME / "tmp_analysis/tier2/kalman/eval/k3_fullframe.npz"
EPS = (0, 5, 15, 19, 22, 26, 28)
GATES = ("G1_envelope", "G2_penetration", "G3_rest", "G4_contact")


def sha256(path) -> str | None:
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def load_kb():
    got = sha256(KB_PY)
    if got != KB_SHA:
        raise SystemExit(f"{KB_PY} sha256 {got} != pinned {KB_SHA}")
    spec = importlib.util.spec_from_file_location("k3fix_build", KB_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["k3fix_build"] = mod
    spec.loader.exec_module(mod)
    return mod


kb = load_kb()                                   # also verifies the K3fix file (6f126f3e) and its null table
from kcommon import FINGER_ENDS, H, J, load_g1, runs  # noqa: E402  (path set by k3fix_build)

F = 2 * H + 1                                    # 15 frames
GRID = np.round(np.linspace(0.0, 1.0, 11), 10)
STEP = 0.1
MARGIN_CONTACT_CM = kb.FARTHER_CM / 2.0          # 1.5 cm
TOL_PEN_CM = kb.ENV_P / 2.0                      # 2.31 cm
MAX_ROUNDS = 12
BIMANUAL = 0.5                                   # report only: second hand listed if it has >= half the contact frames


def guard(path: Path) -> Path:
    p = Path(path).resolve()
    if not str(p).startswith(str(OUT.resolve()) + "/"):
        raise SystemExit(f"refusing to write outside {OUT}: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================================== shared pieces
def blend(k3f: np.ndarray, raw: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Point at weight lam on the geodesic K3fix -> raw (per frame); lam == 0 frames are copied from K3fix exactly."""
    lam = np.asarray(lam, float)
    rk, rr = R.from_quat(k3f[:, 3:]), R.from_quat(raw[:, 3:])
    pos = (1.0 - lam)[:, None] * k3f[:, :3] + lam[:, None] * raw[:, :3]
    q = (rk * R.from_rotvec(lam[:, None] * (rk.inv() * rr).as_rotvec())).as_quat()
    q *= np.where(np.sum(q * k3f[:, 3:], 1) < 0, -1.0, 1.0)[:, None]        # same hemisphere as K3fix
    out = np.concatenate([pos, q], 1)
    out[lam == 0] = k3f[lam == 0]
    return out


def bump(d) -> np.ndarray:
    d = np.asarray(d, float)
    return np.where(d <= F, 0.5 * (1.0 + np.cos(np.pi * d / (F + 1))), 0.0)


def smooth_lam(req: np.ndarray) -> np.ndarray:
    T = len(req)
    lam = np.zeros(T)
    for k in np.flatnonzero(req > 0):
        lo, hi = max(0, k - F), min(T, k + F + 1)
        lam[lo:hi] = np.maximum(lam[lo:hi], req[k] * bump(np.abs(np.arange(lo, hi) - k)))
    return np.clip(lam, 0.0, 1.0)


def side_dist(seq, ep, mesh, p_cm, rot):
    """kb.hand_dist split by hand: (2, T) min distance (cm) of wrist + 5 finger ends of each hand to the mesh."""
    fr = kb.FRAMES[ep]
    g = load_g1(seq)
    Rs2r, ts2r, B, b = (np.array(fr[x]) for x in ("Rs2r", "ts2r", "Bcv2r", "bcv2r"))
    pcv = (((g["soma_joints"] @ Rs2r.T + ts2r) - b) @ B) * 100.0
    v = np.asarray(mesh.vertices, float)
    tree = cKDTree(v[::max(1, len(v) // 60000)] * 100.0)
    dm = np.zeros((2, len(p_cm)))
    for si, side in enumerate(("left", "right")):
        pts = pcv[:, [J["LeftHand" if side == "left" else "RightHand"]] + FINGER_ENDS[side]]
        loc = np.einsum("tji,tkj->tki", rot.as_matrix(), pts - p_cm[:, None])
        dm[si] = tree.query(loc.reshape(-1, 3))[0].reshape(len(p_cm), -1).min(1)
    return dm


class Ctx:
    """Everything gate_work computes before its per-track loop (same calls, same order)."""

    def __init__(self, ep: int):
        m = kb.META[ep]
        self.ep, self.m, self.seq = ep, m, m["sequence_id"]
        self.Ef = kb.FIX.EpisodeFix(m, kb.PRM, True, True, "calibrated")
        self.vis = self.Ef.vis
        self.raw = np.concatenate([self.Ef.p / 100.0, self.Ef.rot.as_quat()], 1)
        self.mesh = kb.load_mesh_vertices(m["object"])
        self.hull = np.asarray(self.mesh.convex_hull.vertices, float)
        self.prims = kb.parse_support(kb.T2 / m["support_surface"])
        self.RT, self.tT, self.fit_res = kb.g1_transform(self.seq, self.vis, self.raw)
        self.dep_raw = kb.depth_track(self.raw, self.RT, self.tT, self.hull, self.prims)
        self.d_raw = kb.hand_dist(self.seq, ep, self.mesh, self.Ef.p, self.Ef.rot)
        self.contact = self.d_raw < kb.CONTACT_CM

    def dist(self, a):
        return kb.hand_dist(self.seq, self.ep, self.mesh, a[:, :3] * 100.0, R.from_quat(a[:, 3:]))

    def depth(self, a):
        return kb.depth_track(a, self.RT, self.tT, self.hull, self.prims)


def gate_records(c: Ctx, trk: dict) -> dict:
    """kb.gate_work's per-track G1/G2/G4 formulas and its G3 rule, verbatim, on arbitrary tracks.
    G3 forced frames: 'K3_frozen' -> the frozen rest mask; every other track uses the K3fix smoother -> rest_rot."""
    Ef, vis = c.Ef, c.vis
    out = {"episode": c.ep, "object": c.m["object"], "sequence_id": c.seq, "frames": int(Ef.T), "invisible": int((~vis).sum()),
           "g1_fit_residual_m": c.fit_res, "n_support_prims": len(c.prims), "n_raw_contact_frames": int(c.contact.sum())}
    for name, a in trk.items():
        p, rot = a[:, :3] * 100.0, R.from_quat(a[:, 3:])
        dr = np.degrees((rot * Ef.rot.inv()).magnitude())
        dp = np.linalg.norm(p - Ef.p, axis=1)
        g1 = {"max_rot_dev_deg": float(dr[vis].max()), "max_pos_dev_cm": float(dp[vis].max()),
              "frames_beyond": int(((dr > kb.ENV_R) | (dp > kb.ENV_P))[vis].sum()),
              "frames_beyond_all_frames_incl_invisible": int(((dr > kb.ENV_R) | (dp > kb.ENV_P)).sum())}
        g1["pass"] = g1["frames_beyond"] == 0
        dep = kb.depth_track(a, c.RT, c.tT, c.hull, c.prims)
        excess = dep - c.dep_raw
        g2 = {"frames_deeper_than_raw_by_gt_env": int((excess > kb.ENV_P)[vis].sum()),
              "max_excess_depth_cm": float(excess[vis].max()), "frames_deeper_all_frames": int((excess > kb.ENV_P).sum()),
              "pen_p95_cm": float(np.percentile(dep[vis], 95)), "pen_p95_raw_cm": float(np.percentile(c.dep_raw[vis], 95)),
              "pen_max_cm": float(dep[vis].max()), "pen_max_raw_cm": float(c.dep_raw[vis].max())}
        g2["f5_p95_form_fires"] = bool(g2["pen_p95_cm"] > g2["pen_p95_raw_cm"] + kb.ENV_P)
        g2["pass"] = g2["frames_deeper_than_raw_by_gt_env"] == 0
        d = kb.hand_dist(c.seq, c.ep, c.mesh, p, rot)
        n = int(c.contact.sum())
        share = float(((d[c.contact] - c.d_raw[c.contact]) > kb.FARTHER_CM).mean()) if n else None
        g4 = {"n_raw_contact_frames": n, "share_farther_gt3cm": share, "evaluable": n >= kb.MIN_CONTACT,
              "s12_form_share_d_gt3cm": float((d[c.contact] > kb.CONTACT_CM).mean()) if n else None}
        g4["pass"] = (not g4["evaluable"]) or share <= kb.MAX_SHARE
        out[name] = {"G1_envelope": g1, "G2_penetration": g2, "G4_contact": g4}
    flagged = [(int(a), int(b), float(st)) for (a, b, st, _), f in zip(Ef.run_tests, Ef.run_fail) if f]
    for name in trk:
        mask = Ef.rest if name == "K3_frozen" else Ef.rest_rot
        forced = [[a, b, int(mask[a:b + 1].sum())] for a, b, _ in flagged]
        out[name]["G3_rest"] = {"flagged_runs": [[a, b, round(st, 2)] for a, b, st in flagged],
                                "w0_frames_forced_in_flagged": forced, "pass": all(n == 0 for _, _, n in forced)}
    for name in trk:
        o = out[name]
        o["all_pass"] = all(o[g]["pass"] for g in GATES)
    return out


def same(a, b, tol=1e-9) -> bool:
    """k3full_gates.same: recursive equality (floats within tol; 'seconds' ignored)."""
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) | set(b)) - {"seconds"}
        return all(k in a and k in b and same(a[k], b[k], tol) for k in keys)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def jsonable(x):
    return json.loads(json.dumps(x, default=float))


# ============================================================================================== tracks
def k3ca(c: Ctx, k3f: np.ndarray):
    T = c.Ef.T
    cand = [blend(k3f, c.raw, np.full(T, g)) for g in GRID]
    dc = np.stack([c.dist(a) for a in cand]) - c.d_raw[None]               # (G, T) farther than raw (cm)
    pc = np.stack([c.depth(a) for a in cand]) - c.dep_raw[None]            # (G, T) deeper than raw (cm)
    ok_c = dc <= MARGIN_CONTACT_CM
    ok_s = pc <= TOL_PEN_CM
    assert ok_c[-1].all() and ok_s[-1].all()                               # lam = 1 (raw) always satisfies both
    lam_c = np.where(c.contact, GRID[np.argmax(ok_c, 0)], 0.0)
    lam_s = np.where(c.vis, GRID[np.argmax(ok_s, 0)], 0.0)
    req = np.maximum(lam_c, lam_s)
    rounds = []
    for _ in range(MAX_ROUNDS):
        lam = smooth_lam(req)
        a = blend(k3f, c.raw, lam)
        far = c.dist(a) - c.d_raw
        deep = c.depth(a) - c.dep_raw
        viol = (c.contact & (far > MARGIN_CONTACT_CM)) | (c.vis & (deep > TOL_PEN_CM))
        rounds.append(int(viol.sum()))
        if not viol.any():
            break
        req[viol] = np.minimum(1.0, lam[viol] + STEP)
    ok = rounds[-1] == 0
    return a, lam, lam_c, lam_s, rounds, ok


def contact_windows(contact: np.ndarray):
    """Raw-contact runs merged across gaps of <= F frames (reporting only)."""
    out = []
    for a, b in runs(contact):
        if out and a - out[-1][1] - 1 <= F:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def motion_stats(a: np.ndarray) -> dict:
    p, r = a[:, :3] * 100.0, R.from_quat(a[:, 3:])
    jp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    jr = np.degrees((r[1:] * r[:-1].inv()).magnitude())
    acc = np.linalg.norm(np.diff(p, 2, axis=0), axis=1)
    return {"max_step_cm": float(jp.max()), "max_step_deg": float(jr.max()), "rms_step_deg": float(np.sqrt((jr ** 2).mean())),
            "rms_pos_accel_cm_per_frame2": float(np.sqrt((acc ** 2).mean()))}


def track_work(ep: int) -> dict:
    t0 = time.time()
    c = Ctx(ep)
    k3f = np.load(K3FIX_NPZ)[f"K3fixfull__{ep}"]
    a, lam, lam_c, lam_s, rounds, ok = k3ca(c, k3f)
    T, vis, Ef = c.Ef.T, c.vis, c.Ef
    # ---- report: contact windows, hands, noisy relative-transform spread
    d_side = side_dist(c.seq, ep, c.mesh, Ef.p, Ef.rot)
    assert np.array_equal(d_side.min(0), c.d_raw)
    hp, hr, _ = Ef.human_hands()
    wins = []
    for wa, wb in contact_windows(c.contact):
        sl = slice(wa, wb + 1)
        n_side = [int(((d_side[si, sl] < kb.CONTACT_CM) & vis[sl]).sum()) for si in (0, 1)]
        prim = int(np.argmax(n_side))
        hands = [prim] + ([1 - prim] if n_side[1 - prim] > 0 and n_side[1 - prim] >= BIMANUAL * n_side[prim] else [])
        spread = {}
        for si in hands:
            fr = np.flatnonzero((d_side[si] < kb.CONTACT_CM) & vis)
            fr = fr[(fr >= wa) & (fr <= wb)]
            rp = np.einsum("tji,tj->ti", hr[si][fr].as_matrix(), Ef.p[fr] - hp[si][fr])
            rq = hr[si][fr].inv() * Ef.rot[fr]
            pm, qm = np.median(rp, 0), rq.mean()
            dp = np.linalg.norm(rp - pm, axis=1)
            dq = np.degrees((qm.inv() * rq).magnitude())
            spread["left" if si == 0 else "right"] = {
                "n_frames": int(len(fr)), "pos_dev_median_cm": float(np.median(dp)), "pos_dev_max_cm": float(dp.max()),
                "rot_dev_median_deg": float(np.median(dq)), "rot_dev_max_deg": float(dq.max()),
                "object_origin_to_wrist_cm": float(np.linalg.norm(pm))}
        wins.append({"start": int(wa), "end": int(wb), "frames": int(wb - wa + 1), "raw_contact_frames": int(c.contact[sl].sum()),
                     "contact_frames_left_right": n_side, "hands": ["left" if s == 0 else "right" for s in hands],
                     "lam_mean": float(lam[sl].mean()), "lam_max": float(lam[sl].max()),
                     "relative_transform_spread_noisy": spread})
    p, rot = a[:, :3] * 100.0, R.from_quat(a[:, 3:])
    pk, rk = k3f[:, :3] * 100.0, R.from_quat(k3f[:, 3:])
    dev_k = (np.linalg.norm(p - pk, axis=1), np.degrees((rot * rk.inv()).magnitude()))
    dev_r = (np.linalg.norm(p - Ef.p, axis=1), np.degrees((rot * Ef.rot.inv()).magnitude()))
    rep = {"ep": ep, "object": c.m["object"], "sequence_id": c.seq, "frames": int(T), "invisible": int((~vis).sum()),
           "ok": bool(ok and np.isfinite(a).all()), "rounds_violations": rounds,
           "n_raw_contact_frames": int(c.contact.sum()), "raw_contact_runs": [[int(x), int(y)] for x, y in runs(c.contact)],
           "contact_windows": wins,
           "lam": {"frames_gt0": int((lam > 0).sum()), "share_gt0": float((lam > 0).mean()), "mean": float(lam.mean()),
                   "frames_eq1": int((lam >= 1).sum()), "segments_gt0": [[int(x), int(y)] for x, y in runs(lam > 0)],
                   "req_contact_frames_gt0": int((lam_c > 0).sum()), "req_support_frames_gt0": int((lam_s > 0).sum()),
                   "req_contact_mean_on_contact_frames": float(lam_c[c.contact].mean()) if c.contact.any() else None},
           "dev_vs_k3fix": {"max_cm": float(dev_k[0].max()), "max_deg": float(dev_k[1].max()), "mean_cm": float(dev_k[0].mean()),
                            "mean_deg": float(dev_k[1].mean())},
           "dev_vs_raw": {"max_cm": float(dev_r[0][vis].max()), "max_deg": float(dev_r[1][vis].max()),
                          "mean_cm": float(dev_r[0].mean()), "mean_deg": float(dev_r[1].mean())},
           "motion": {"raw": motion_stats(c.raw), "K3fix": motion_stats(k3f), "K3CA": motion_stats(a)},
           "lam0_frames_identical_to_k3fix": bool(np.array_equal(a[lam == 0], k3f[lam == 0])),
           "sec": round(time.time() - t0, 1)}
    return {"rep": rep, "full": a, "pose": a[vis], "vis": vis.copy(), "lam": lam}


def tracks() -> int:
    if sha256(K3FIX_NPZ) != K3FIX_NPZ_SHA:
        raise SystemExit("k3fix_tracks.npz changed")
    with Pool(len(EPS)) as pool:
        res = sorted(pool.map(track_work, EPS), key=lambda r: r["rep"]["ep"])
    arrays = {}
    rep = {"written": time.strftime("%F %T"), "noisy_only": True, "method": __doc__.split("\n\n")[1],
           "constants": {"F_frames": F, "grid": GRID.tolist(), "margin_contact_cm": MARGIN_CONTACT_CM,
                         "tol_penetration_cm": TOL_PEN_CM, "contact_cm": kb.CONTACT_CM, "envelope_rot_deg": kb.ENV_R,
                         "envelope_pos_cm": kb.ENV_P},
           "inputs": {"k3fix_build_py_sha256": KB_SHA, "k3fix_tracks_npz_sha256": K3FIX_NPZ_SHA,
                      "gates_all30_sha256": GATES30_SHA, "k3fix_precommit_sha256": kb.FIX_SHA},
           "python": sys.executable, "numpy": np.__version__, "episodes": {}}
    ok = True
    for r in res:
        ep = r["rep"]["ep"]
        arrays[f"K3CAfull__{ep}"], arrays[f"K3CApose__{ep}"] = r["full"], r["pose"]
        arrays[f"vis__{ep}"], arrays[f"lam__{ep}"] = r["vis"], r["lam"]
        rep["episodes"][f"{ep:02d}"] = r["rep"]
        ok &= r["rep"]["ok"] and r["rep"]["lam0_frames_identical_to_k3fix"]
        x = r["rep"]
        print(f"ep{ep:02d} {x['object']:<16} contact {x['n_raw_contact_frames']:4d} windows {len(x['contact_windows']):2d} | lam>0 "
              f"{x['lam']['share_gt0']:.2f} mean {x['lam']['mean']:.3f} =1 {x['lam']['frames_eq1']} (req contact "
              f"{x['lam']['req_contact_frames_gt0']}, support {x['lam']['req_support_frames_gt0']}) rounds {x['rounds_violations']} | "
              f"vs K3fix max {x['dev_vs_k3fix']['max_cm']:.2f} cm {x['dev_vs_k3fix']['max_deg']:.1f} deg | ok {x['ok']}", flush=True)
    rep["all_ok"] = bool(ok)
    np.savez(guard(OUT / "k3ca_tracks.npz"), **arrays)
    rep["k3ca_tracks_npz_sha256"] = sha256(OUT / "k3ca_tracks.npz")
    guard(OUT / "tracks.json").write_text(json.dumps(rep, indent=1, default=float))
    print(f"all_ok {rep['all_ok']}; wrote {OUT / 'k3ca_tracks.npz'} ({rep['k3ca_tracks_npz_sha256'][:16]})")
    return 0 if rep["all_ok"] else 1


# ============================================================================================== gates
def gate_work(ep: int) -> dict:
    t0 = time.time()
    c = Ctx(ep)
    trk = {"K3fix": np.load(K3FIX_NPZ)[f"K3fixfull__{ep}"], "K3_frozen": np.load(K3_FROZEN_NPZ)[f"K3full__{ep}"],
           "K3CA": np.load(OUT / "k3ca_tracks.npz")[f"K3CAfull__{ep}"]}
    r = gate_records(c, trk)
    r["seconds"] = round(time.time() - t0, 1)
    return r


def gates() -> int:
    tr = json.loads((OUT / "tracks.json").read_text())
    npz_sha = sha256(OUT / "k3ca_tracks.npz")
    if not tr.get("all_ok") or tr.get("k3ca_tracks_npz_sha256") != npz_sha:
        raise SystemExit("build/tracks.json not ok or k3ca_tracks.npz changed: run 'tracks' first")
    if sha256(GATES30) != GATES30_SHA or sha256(K3FIX_NPZ) != K3FIX_NPZ_SHA:
        raise SystemExit("gates_all30.json or k3fix_tracks.npz changed")
    g30 = json.loads(GATES30.read_text())
    with Pool(len(EPS)) as pool:
        res = sorted(pool.map(gate_work, EPS), key=lambda r: r["episode"])
    checks, rows = [], []
    for r in res:
        ep = r["episode"]
        old = g30["clips"][f"{ep:02d}"]
        top = {k: v for k, v in r.items() if k not in ("K3fix", "K3_frozen", "K3CA", "seconds")}
        rep_ok = (same(jsonable(r["K3fix"]), old["K3fix"]) and same(jsonable(r["K3_frozen"]), old["K3_frozen"])
                  and same(jsonable(top), {k: old[k] for k in top}))
        checks.append({"name": f"ep{ep:02d}_replicates_gates_all30", "ok": bool(rep_ok),
                       "detail": "copied gate formulas on K3fix + frozen K3 = gates_all30.json records"})
        checks.append({"name": f"ep{ep:02d}_k3fix_failed_before", "ok": old["K3fix"]["all_pass"] is False,
                       "detail": "episode is one of the 7 K3fix gate failures"})
        k, f = r["K3CA"], r["K3fix"]
        rows.append({"episode": ep, "object": r["object"], "n_contact": k["G4_contact"]["n_raw_contact_frames"],
                     **{f"{nm}_{g[:2]}": o[g]["pass"] for nm, o in (("K3fix", f), ("K3CA", k)) for g in GATES},
                     "K3fix_all_pass": f["all_pass"], "K3CA_all_pass": k["all_pass"],
                     "g1_rot_max": (f["G1_envelope"]["max_rot_dev_deg"], k["G1_envelope"]["max_rot_dev_deg"]),
                     "g1_pos_max": (f["G1_envelope"]["max_pos_dev_cm"], k["G1_envelope"]["max_pos_dev_cm"]),
                     "g1_beyond": (f["G1_envelope"]["frames_beyond"], k["G1_envelope"]["frames_beyond"]),
                     "g2_frames": (f["G2_penetration"]["frames_deeper_than_raw_by_gt_env"], k["G2_penetration"]["frames_deeper_than_raw_by_gt_env"]),
                     "g2_max_excess": (f["G2_penetration"]["max_excess_depth_cm"], k["G2_penetration"]["max_excess_depth_cm"]),
                     "g3_flagged": len(k["G3_rest"]["flagged_runs"]),
                     "g4_share": (f["G4_contact"]["share_farther_gt3cm"], k["G4_contact"]["share_farther_gt3cm"])})
    bad = [c for c in checks if not c["ok"]]
    passing = [x["episode"] for x in rows if x["K3CA_all_pass"]]
    failing = [x["episode"] for x in rows if not x["K3CA_all_pass"]]
    pf = lambda b: "pass" if b else "FAIL"  # noqa: E731
    lines = [f"K3CA - noisy-only input gates G1-G4 on the 7 K3fix gate failures (same gate formulas and Tier-2-only constants "
             f"as gates_all30: envelope {kb.ENV_R:.2f} deg / {kb.ENV_P:.2f} cm; G4 share <= {kb.MAX_SHARE} with >= {kb.MIN_CONTACT} "
             f"contact frames)", f"k3ca_tracks.npz {npz_sha[:16]}; checks {len(checks) - len(bad)}/{len(checks)} ok"
             + (f"; FAILED: {' '.join(c['name'] for c in bad)}" if bad else ""), "",
             "ep  object             track  G1   G2   G3   G4    rot_max pos_max beyond deeper excess_cm  G4 n/share   -> verdict"]
    for x, r in zip(rows, res):
        for nm in ("K3fix", "K3CA"):
            o = r[nm]
            g1, g2, g4 = o["G1_envelope"], o["G2_penetration"], o["G4_contact"]
            verdict = ("K3CA candidate" if o["all_pass"] else "raw run") if nm == "K3CA" else ("(was K3fix candidate)" if o["all_pass"] else "(K3fix gate fail)")
            lines.append(f"{x['episode']:02d}  {x['object']:<18} {nm:<6} {pf(g1['pass']):<4} {pf(g2['pass']):<4} {pf(o['G3_rest']['pass']):<4} "
                         f"{pf(g4['pass']):<4} {g1['max_rot_dev_deg']:7.1f} {g1['max_pos_dev_cm']:7.2f} {g1['frames_beyond']:6d} "
                         f"{g2['frames_deeper_than_raw_by_gt_env']:6d} {g2['max_excess_depth_cm']:9.2f}  {g4['n_raw_contact_frames']:4d}/"
                         f"{g4['share_farther_gt3cm']:.4f}  -> {verdict}")
    lines += ["", f"K3CA PASS ({len(passing)}): {' '.join(f'{e:02d}' for e in passing)}",
              f"K3CA FAIL ({len(failing)}): {' '.join(f'{e:02d}' for e in failing) or '-'}"]
    for cc in bad:
        lines.append(f"CHECK FAIL {cc['name']}: {cc['detail']}")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out = {"written": time.strftime("%F %T"), "noisy_only": True, "gate_code": str(KB_PY), "gate_code_sha256": KB_SHA,
           "gates_all30_sha256": GATES30_SHA, "k3ca_tracks_npz_sha256": npz_sha, "envelope_rot_deg": kb.ENV_R,
           "envelope_pos_cm": kb.ENV_P, "g4_max_share": kb.MAX_SHARE, "g4_min_contact": kb.MIN_CONTACT,
           "rule": "episode uses K3CA iff it failed the K3fix gates (gates_all30) AND K3CA passes G1-G4; otherwise the raw "
                   "C3r2_tier2 run", "checks": checks, "checks_ok": not bad, "table": rows,
           "pass_episodes": passing, "fail_episodes": failing, "clips": {f"{r['episode']:02d}": jsonable(r) for r in res}}
    for path, body in ((OUT / "gates.json", json.dumps(out, indent=1, default=float)), (OUT / "gates.txt", text)):
        p = guard(path)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(body)
        tmp.replace(p)
    print(f"wrote {OUT / 'gates.json'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "tracks":
        raise SystemExit(tracks())
    if cmd == "gates":
        raise SystemExit(gates())
    print(__doc__)
    raise SystemExit(2)
