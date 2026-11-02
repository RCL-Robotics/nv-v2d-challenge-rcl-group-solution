#!/usr/bin/env python3
"""k3fix_build.py (NOISY-ONLY, CPU only): step 1 of the K3fix pilot build (tmp_analysis/tier2/k3fix_pilot).
Run:  CUDA_VISIBLE_DEVICES= nice -n 19 conda_envs/v2d/bin/python -I -B scripts/k3fix_build.py tracks|gates

K3fix = tmp_analysis/tier2/k3_rootcause/ep04/k3fix_v1_precommit.py (sha256 6f126f3e..., pre-committed noisy-only in
ep04/s7_precommit_noisy_only.json), variant fixAsC = EpisodeFix(fix_zupt=True, fix_anchor=True, mode="calibrated").
Reads no clean data (no Tier-1 path, no *LOCAL_NOT_FOR_UPLOAD* file). Writes only tmp_analysis/tier2/k3fix_pilot/build/.

tracks  K3fix on all 30 Tier-2 episodes in one pool (frozen K3 recomputed in the same process for comparison):
          - K3fix full-frame track (e1_k3_fullframe.py formula) and its visible-frame pose output (pose_out)
          - changed episodes = the smoothed state X3 differs from the frozen K3 (np.array_equal); must be exactly
            {3, 4, 9, 17} (k3_rootcause/ep04/s11_changed_noisy_only.txt); for the other 26 the full track must match
            kalman/eval/k3_fullframe.npz and the pose kalman/outputs/kalman_poses.npz to < 1e-12
        -> build/k3fix_tracks.npz (K3fixfull__<ep>, K3fixpose__<ep>, vis__<ep>), build/tracks.json
tracks also records the calibrated rest-run flags, the released w=0 frames and the anchor rejections per episode.

gates   the four noisy-only pre-training gates on the changed clips (K3fix track vs the released Tier-2 track), each
        also evaluated for the FROZEN K3 track and compared with the root-cause numbers (replication of the recipes):
  G1 envelope: no visible frame with |K3fix - raw| rotation > 49.2 deg or position > 4.6 cm
               (fixes/src/f5_gate2.py G1: SAT_SD * sqrt(chi2_3 99%), d2b saturation fit, Tier-2-only constants)
  G2 penetration: no visible frame where the K3fix object is > 4.6 cm deeper inside the Tier-2 support stage / below
               the floor (G1 training world) than the raw object (literal per-frame rule; decides). The root-cause
               F5 form (p95 depth > raw p95 + 4.6 cm) is reported next to it.
  G3 rest windows: no rest run flagged by the calibrated random-walk test still has zero angular velocity forced
  G4 contact consistency (review/r4_contact_gate_all.py, literal): among frames where the raw object is < 3 cm from a
               hand point (wrist + 5 finger ends, Tier-2 SOMA), the share where the K3fix object is > 3 cm farther than
               raw must be <= 5 %, evaluated when there are >= 50 such frames (fewer: not evaluable, reported, passes as
               in the reviewer's all-30 table)
        -> build/gates.json (clip PASS iff G1-G4 all pass; a failing clip is not trained on K3fix)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
KOUT = ME / "tmp_analysis/tier2/kalman"
RC = ME / "tmp_analysis/tier2/k3_rootcause"
FIXFILE = RC / "ep04/k3fix_v1_precommit.py"
FIX_SHA = "6f126f3e28c9934e4c70bb3a47cc4a8ceb83901f8823400e7d4522e40e793a2c"
Q99_SHA = "936dd129c79993b1025a61d70c36df5e66e8d3c551bb05d0e4be645ed82a5fcc"
BUILD = ME / "tmp_analysis/tier2/k3fix_pilot/build"
G1ROOT = ME / "data/human_motion_data_tier2"
EXPECTED_CHANGED = [3, 4, 9, 17]
CONTACT_CM, FARTHER_CM, MAX_SHARE, MIN_CONTACT = 3.0, 3.0, 0.05, 50

sys.path.insert(0, str(KOUT / "scripts"))
from kalman_core import BT, P_  # noqa: E402
from kfilter import Episode, episodes, load_params  # noqa: E402
from kcommon import CHI2_3_99, FINGER_ENDS, J, T2, load_g1, load_mesh_vertices, runs  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(path: Path) -> Path:
    p = Path(path).resolve()
    if not str(p).startswith(str(BUILD.resolve()) + "/"):
        raise SystemExit(f"refusing to write outside {BUILD}: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_fix():
    got = sha256(FIXFILE)
    if got != FIX_SHA:
        raise SystemExit(f"{FIXFILE} sha256 {got} != pre-committed {FIX_SHA}")
    q = sha256(FIXFILE.parent / "s6_null_q99.npz")
    if q != Q99_SHA:
        raise SystemExit(f"s6_null_q99.npz sha256 {q} != {Q99_SHA}")
    spec = importlib.util.spec_from_file_location("k3fix_v1_precommit", FIXFILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PRM = load_params()
META = {m["episode_index"]: m for m in episodes()}
FIX = load_fix()
D2B = json.loads((KOUT / "design/d2b_anchor_fit.json").read_text())
FRAMES = {r["episode"]: r for r in json.loads((KOUT / "design/d1_frames.json").read_text())}
# envelope (fixes/src/fxcommon.py + f5_gate2.py, Tier-2-only constants)
SAT_SD_ROT_DEG = float(PRM["q_br_deg"]) * np.sqrt(float(D2B["tilt"]["t_sat_frames"]))
SAT_SD_POS_CM = float(PRM["q_bp_cm"]) * np.sqrt(float(D2B["height"]["t_sat_frames"]))
ENV_R = SAT_SD_ROT_DEG * np.sqrt(CHI2_3_99)
ENV_P = SAT_SD_POS_CM * np.sqrt(CHI2_3_99)


def full_track(E, X):
    return np.concatenate([X[:, P_] / 100.0, (R.from_rotvec(-X[:, BT]) * E.rot).as_quat()], 1)


# ============================================================================================== tracks
def track_work(ep: int) -> dict:
    t0 = time.time()
    m = META[ep]
    E0 = Episode(m, PRM)
    pose0, X0, _ = E0.variant("K3")
    E1 = FIX.EpisodeFix(m, PRM, True, True, "calibrated")
    pose1, X1, diag1 = E1.variant("K3")
    full1 = full_track(E1, X1)
    flagged = [[int(a), int(b), round(float(st), 2)] for (a, b, st, _), f in zip(E1.run_tests, E1.run_fail) if f]
    forced_in_flagged = int(sum(E1.rest_rot[a:b + 1].sum() for a, b, _ in flagged))
    return {"ep": ep, "full": full1, "pose": pose1, "vis": E1.vis.copy(), "x3_identical": bool(np.array_equal(X0, X1)),
            "x3_max_abs_diff": float(np.abs(X0 - X1).max()), "full_vs_pose_vis": float(np.abs(full1[E1.vis] - pose1).max()),
            "frozen_pose_replay": pose0, "flagged_runs_calibrated": flagged,
            "w0_frames_released": int((E1.rest & ~E1.rest_rot).sum()), "w0_frames_forced_in_flagged_runs": forced_in_flagged,
            "anchor_rejections": E1.anchor_rejected, "frames": int(E1.T), "invisible": int((~E1.vis).sum()),
            "sec": round(time.time() - t0, 1)}


def tracks() -> int:
    ff = np.load(KOUT / "eval/k3_fullframe.npz")
    kp = np.load(KOUT / "outputs/kalman_poses.npz")
    with Pool(10) as pool:
        res = sorted(pool.map(track_work, range(30)), key=lambda r: r["ep"])
    arrays, rep = {}, {"written": time.strftime("%F %T"), "noisy_only": True, "k3fix_file": str(FIXFILE),
                         "k3fix_sha256": FIX_SHA, "variant": "fixAsC = EpisodeFix(fix_zupt=True, fix_anchor=True, mode='calibrated')",
                         "python": sys.executable, "numpy": np.__version__, "episodes": {}}
    ok = True
    changed = []
    for r in res:
        ep = r["ep"]
        arrays[f"K3fixfull__{ep}"], arrays[f"K3fixpose__{ep}"], arrays[f"vis__{ep}"] = r["full"], r["pose"], r["vis"]
        d = {k: v for k, v in r.items() if k not in ("full", "pose", "vis", "frozen_pose_replay")}
        d["object"] = META[ep]["object"]
        d["frozen_replay_vs_kalman_poses"] = float(np.abs(r["frozen_pose_replay"] - kp[f"K3__{ep}"]).max())
        d["k3fix_full_vs_k3_fullframe_max_abs"] = float(np.abs(r["full"] - ff[f"K3full__{ep}"]).max())
        d["k3fix_pose_vs_kalman_poses_max_abs"] = float(np.abs(r["pose"] - kp[f"K3__{ep}"]).max())
        d["changed"] = not r["x3_identical"]
        if d["changed"]:
            changed.append(ep)
            q_new, q_old = R.from_quat(r["full"][:, 3:]), R.from_quat(ff[f"K3full__{ep}"][:, 3:])
            d["k3fix_vs_k3_max_rot_deg"] = float(np.degrees((q_new * q_old.inv()).magnitude()).max())
            d["k3fix_vs_k3_max_pos_cm"] = float(100 * np.linalg.norm(r["full"][:, :3] - ff[f"K3full__{ep}"][:, :3], axis=1).max())
            d["ok"] = d["full_vs_pose_vis"] < 1e-12 and d["frozen_replay_vs_kalman_poses"] < 1e-12 and d["w0_frames_forced_in_flagged_runs"] == 0
        else:
            d["ok"] = (d["k3fix_full_vs_k3_fullframe_max_abs"] < 1e-12 and d["k3fix_pose_vs_kalman_poses_max_abs"] < 1e-12
                       and d["full_vs_pose_vis"] < 1e-12 and not d["flagged_runs_calibrated"] and d["w0_frames_released"] == 0
                       and not d["anchor_rejections"])
        ok &= d["ok"]
        rep["episodes"][f"{ep:02d}"] = d
        print(f"ep{ep:02d} {d['object']:<20} changed={d['changed']!s:<5} X3 maxdiff {d['x3_max_abs_diff']:.2e} | vs k3_fullframe "
              f"{d['k3fix_full_vs_k3_fullframe_max_abs']:.2e} | flagged {d['flagged_runs_calibrated']} released w0 {d['w0_frames_released']}"
              f" | anchor rejections {len(d['anchor_rejections'])} | invisible {d['invisible']} | ok {d['ok']}", flush=True)
    rep["changed_episodes"] = changed
    rep["changed_as_expected"] = changed == EXPECTED_CHANGED
    rep["all_ok"] = bool(ok and rep["changed_as_expected"])
    np.savez(guard(BUILD / "k3fix_tracks.npz"), **arrays)
    rep["k3fix_tracks_npz_sha256"] = sha256(BUILD / "k3fix_tracks.npz")
    guard(BUILD / "tracks.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"changed episodes {changed} (expected {EXPECTED_CHANGED}); all_ok {rep['all_ok']}; wrote {BUILD / 'k3fix_tracks.npz'}")
    return 0 if rep["all_ok"] else 1


# ============================================================================================== gates
def parse_support(path: Path):
    """Verbatim from k3_rootcause/fixes/src/f3_support_metrics.py."""
    txt = path.read_text()
    prims = []
    for m in re.finditer(r'def (Cube|Cylinder) "[^"]*"[^{]*\{(.*?)\n    \}', txt, re.S):
        kind, body = m.group(1), m.group(2)
        num = lambda k: float(re.search(rf"double {k} = ([-0-9.eE]+)", body).group(1))  # noqa: E731
        vec = lambda k: np.array([float(x) for x in re.search(rf"double3 xformOp:{k} = \(([^)]*)\)", body).group(1).split(",")])  # noqa: E731
        t = vec("translate")
        if kind == "Cube":
            sc = vec("scale") if "xformOp:scale" in body else np.ones(3)
            prims.append(("box", t, 0.5 * num("size") * sc))
        else:
            assert '"Z"' in body
            prims.append(("cyl", t, (num("radius"), 0.5 * num("height"))))
    return prims


def depth_in(prim, x):
    """Verbatim from f3_support_metrics.py: (N,) penetration depth (m) of points x (N,3) into the prim, 0 outside."""
    kind, t, s = prim
    d = x - t
    if kind == "box":
        inside = np.all(np.abs(d) <= s, axis=1)
        dep = np.min(s - np.abs(d), axis=1)
    else:
        r, hh = s
        rad = np.linalg.norm(d[:, :2], axis=1)
        inside = (rad <= r) & (np.abs(d[:, 2]) <= hh)
        dep = np.minimum(r - rad, hh - np.abs(d[:, 2]))
    return np.where(inside, dep, 0.0)


def g1_transform(seq: str, vis, raw):
    """f3_support_metrics.qT_of: rigid T (released OpenCV world -> Tier-2 G1 world) at the first visible frame."""
    import pyarrow.parquet as pq
    tab = pq.read_table(G1ROOT / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet",
                        columns=["object_body_position", "object_body_wxyz"])
    bp = np.asarray(tab.column("object_body_position").to_pylist()[0], float)[:, 0]
    bq = np.asarray(tab.column("object_body_wxyz").to_pylist()[0], float)[:, 0]
    i0 = int(np.flatnonzero(vis)[0])
    RT = R.from_quat(bq[i0][[1, 2, 3, 0]]) * R.from_quat(raw[i0, 3:]).inv()
    tT = bp[i0] - RT.apply(raw[i0, :3])
    res = float(np.abs(RT.apply(raw[vis, :3]) + tT - bp[vis]).max())
    assert res < 1e-4, (seq, res)
    return RT, tT, res


def depth_track(a, RT, tT, hull, prims):
    Rw = RT * R.from_quat(a[:, 3:])
    pw = RT.apply(a[:, :3]) + tT
    X = np.einsum("tij,kj->tki", Rw.as_matrix(), hull) + pw[:, None]
    flat = X.reshape(-1, 3)
    dep_floor = np.maximum(0.0, -flat[:, 2]).reshape(len(a), -1).max(1)
    dep_sup = np.zeros(len(a))
    for pr in prims:
        dep_sup = np.maximum(dep_sup, depth_in(pr, flat).reshape(len(a), -1).max(1))
    return np.maximum(dep_floor, dep_sup) * 100.0


def hand_dist(seq, ep, mesh, p_cm, rot):
    """review/r4_contact_gate_all.py recipe: min distance (cm) of wrist + 5 finger ends (both hands) to the mesh."""
    fr = FRAMES[ep]
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
    return dm.min(0)


def gate_work(ep: int) -> dict:
    m = META[ep]
    seq = m["sequence_id"]
    Ef = FIX.EpisodeFix(m, PRM, True, True, "calibrated")           # raw track + the K3fix rest flags
    vis = Ef.vis
    raw = np.concatenate([Ef.p / 100.0, Ef.rot.as_quat()], 1)
    trk = {"K3fix": np.load(BUILD / "k3fix_tracks.npz")[f"K3fixfull__{ep}"],
           "K3_frozen": np.load(KOUT / "eval/k3_fullframe.npz")[f"K3full__{ep}"]}
    mesh = load_mesh_vertices(m["object"])
    hull = np.asarray(mesh.convex_hull.vertices, float)
    prims = parse_support(T2 / m["support_surface"])
    RT, tT, fit_res = g1_transform(seq, vis, raw)
    dep_raw = depth_track(raw, RT, tT, hull, prims)
    d_raw = hand_dist(seq, ep, mesh, Ef.p, Ef.rot)
    contact = d_raw < CONTACT_CM
    out = {"episode": ep, "object": m["object"], "sequence_id": seq, "frames": int(Ef.T), "invisible": int((~vis).sum()),
           "g1_fit_residual_m": fit_res, "n_support_prims": len(prims), "n_raw_contact_frames": int(contact.sum())}
    for name, a in trk.items():
        p, rot = a[:, :3] * 100.0, R.from_quat(a[:, 3:])
        dr = np.degrees((rot * Ef.rot.inv()).magnitude())
        dp = np.linalg.norm(p - Ef.p, axis=1)
        g1 = {"max_rot_dev_deg": float(dr[vis].max()), "max_pos_dev_cm": float(dp[vis].max()),
              "frames_beyond": int(((dr > ENV_R) | (dp > ENV_P))[vis].sum()),
              "frames_beyond_all_frames_incl_invisible": int(((dr > ENV_R) | (dp > ENV_P)).sum())}
        g1["pass"] = g1["frames_beyond"] == 0
        dep = depth_track(a, RT, tT, hull, prims)
        excess = dep - dep_raw
        g2 = {"frames_deeper_than_raw_by_gt_env": int((excess > ENV_P)[vis].sum()),
              "max_excess_depth_cm": float(excess[vis].max()), "frames_deeper_all_frames": int((excess > ENV_P).sum()),
              "pen_p95_cm": float(np.percentile(dep[vis], 95)), "pen_p95_raw_cm": float(np.percentile(dep_raw[vis], 95)),
              "pen_max_cm": float(dep[vis].max()), "pen_max_raw_cm": float(dep_raw[vis].max())}
        g2["f5_p95_form_fires"] = bool(g2["pen_p95_cm"] > g2["pen_p95_raw_cm"] + ENV_P)
        g2["pass"] = g2["frames_deeper_than_raw_by_gt_env"] == 0
        d = hand_dist(seq, ep, mesh, p, rot)
        n = int(contact.sum())
        share = float(((d[contact] - d_raw[contact]) > FARTHER_CM).mean()) if n else None
        g4 = {"n_raw_contact_frames": n, "share_farther_gt3cm": share, "evaluable": n >= MIN_CONTACT,
              "s12_form_share_d_gt3cm": float((d[contact] > CONTACT_CM).mean()) if n else None}
        g4["pass"] = (not g4["evaluable"]) or share <= MAX_SHARE
        out[name] = {"G1_envelope": g1, "G2_penetration": g2, "G4_contact": g4}
    # G3: rest runs flagged by the calibrated random-walk test (the fix's own test) vs zero angular velocity forced
    flagged = [(int(a), int(b), float(st)) for (a, b, st, _), f in zip(Ef.run_tests, Ef.run_fail) if f]
    forced = {"K3fix": [[a, b, int(Ef.rest_rot[a:b + 1].sum())] for a, b, _ in flagged],
              "K3_frozen": [[a, b, int(Ef.rest[a:b + 1].sum())] for a, b, _ in flagged]}
    for name in trk:
        out[name]["G3_rest"] = {"flagged_runs": [[a, b, round(st, 2)] for a, b, st in flagged],
                                "w0_frames_forced_in_flagged": forced[name],
                                "pass": all(n == 0 for _, _, n in forced[name])}
    for name in trk:
        o = out[name]
        o["all_pass"] = all(o[g]["pass"] for g in ("G1_envelope", "G2_penetration", "G3_rest", "G4_contact"))
    return out


def gates() -> int:
    tr = json.loads((BUILD / "tracks.json").read_text())
    if not tr.get("all_ok") or tr.get("k3fix_tracks_npz_sha256") != sha256(BUILD / "k3fix_tracks.npz"):
        raise SystemExit("build/tracks.json not ok or k3fix_tracks.npz changed: run 'tracks' first")
    eps = tr["changed_episodes"]
    with Pool(len(eps)) as pool:
        res = sorted(pool.map(gate_work, eps), key=lambda r: r["episode"])
    # replication of the root-cause numbers for the FROZEN K3 track (same recipes -> the gate code is the same code)
    g2j = json.loads((RC / "fixes/gate2.json").read_text())["gate"]
    r4 = {int(r[0]): r for r in json.loads((RC / "review/r4_contact_gate_all.json").read_text())}
    repl, rep_ok = {}, True
    for r in res:
        ep = r["episode"]
        k3 = next(c for c in g2j[str(ep)]["K3g2"]["checks"] if c["variant"] == "K3")
        fz = r["K3_frozen"]
        rows = {"rot_max_deg": (fz["G1_envelope"]["max_rot_dev_deg"], k3["bias_rot_max_deg"], 1e-6),
                "pos_max_cm": (fz["G1_envelope"]["max_pos_dev_cm"], k3["bias_pos_max_cm"], 1e-6),
                "pen_p95_cm": (fz["G2_penetration"]["pen_p95_cm"], k3["pen_p95_cm"], 2e-5),
                "pen_p95_raw_cm": (fz["G2_penetration"]["pen_p95_raw_cm"], k3["pen_p95_raw_cm"], 2e-5),
                "r4_n_contact": (fz["G4_contact"]["n_raw_contact_frames"], r4[ep][2], 0),
                "r4_literal_share": (fz["G4_contact"]["share_farther_gt3cm"], r4[ep][4], 1e-12)}
        repl[f"{ep:02d}"] = {k: {"here": a, "root_cause": b, "match": abs(a - b) <= tol} for k, (a, b, tol) in rows.items()}
        rep_ok &= all(v["match"] for v in repl[f"{ep:02d}"].values())
    out = {"written": time.strftime("%F %T"), "noisy_only": True, "k3fix_tracks_npz_sha256": tr["k3fix_tracks_npz_sha256"],
           "envelope_rot_deg": ENV_R, "envelope_pos_cm": ENV_P,
           "rules": {"G1": "no visible frame with |K3fix - raw| rotation > envelope_rot_deg or position > envelope_pos_cm",
                     "G2": "no visible frame where the K3fix object penetrates the support stage / floor (G1 world) more than "
                           "the raw object by > envelope_pos_cm (literal per-frame rule; the F5 p95 form is reported)",
                     "G3": "no rest run flagged by the calibrated random-walk test has zero angular velocity forced",
                     "G4": f"share of raw-contact frames (raw hand-object < {CONTACT_CM} cm) where K3fix is > {FARTHER_CM} cm "
                           f"farther than raw <= {MAX_SHARE}, evaluated with >= {MIN_CONTACT} contact frames",
                     "clip": "PASS iff G1..G4 all pass; a failing clip is not trained on K3fix"},
           "replication_frozen_K3_vs_root_cause": repl, "replication_ok": bool(rep_ok),
           "clips": {f"{r['episode']:02d}": r for r in res},
           "pass_clips": [r["episode"] for r in res if r["K3fix"]["all_pass"]],
           "fail_clips": [r["episode"] for r in res if not r["K3fix"]["all_pass"]]}
    guard(BUILD / "gates.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"envelope {ENV_R:.3f} deg / {ENV_P:.3f} cm; replication of root-cause numbers for frozen K3: {'OK' if rep_ok else 'MISMATCH'}")
    for k, v in repl.items():
        print(f"  ep{k} " + "  ".join(f"{n} {x['here']:.5g}/{x['root_cause']:.5g}{'' if x['match'] else ' MISMATCH'}" for n, x in v.items()))
    for r in res:
        for name in ("K3fix", "K3_frozen"):
            o = r[name]
            g1, g2, g3, g4 = o["G1_envelope"], o["G2_penetration"], o["G3_rest"], o["G4_contact"]
            print(f"ep{r['episode']:02d} {r['object']:<18} {name:<9} G1 {'pass' if g1['pass'] else 'FAIL'} (rot max {g1['max_rot_dev_deg']:.1f} "
                  f"pos max {g1['max_pos_dev_cm']:.2f}, {g1['frames_beyond']} frames beyond) | G2 {'pass' if g2['pass'] else 'FAIL'} "
                  f"({g2['frames_deeper_than_raw_by_gt_env']} frames, max excess {g2['max_excess_depth_cm']:.1f} cm; p95 {g2['pen_p95_cm']:.2f} vs raw "
                  f"{g2['pen_p95_raw_cm']:.2f}, F5 form fires {g2['f5_p95_form_fires']}) | G3 {'pass' if g3['pass'] else 'FAIL'} "
                  f"(flagged {g3['flagged_runs']} w0 forced {g3['w0_frames_forced_in_flagged']}) | G4 {'pass' if g4['pass'] else 'FAIL'} "
                  f"(n {g4['n_raw_contact_frames']}, share {g4['share_farther_gt3cm']:.4f}) -> {'PASS' if o['all_pass'] else 'FAIL'}")
    print(f"K3fix clips passing all gates: {out['pass_clips']}; failing: {out['fail_clips']}; wrote {BUILD / 'gates.json'}")
    return 0 if rep_ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "tracks":
        raise SystemExit(tracks())
    if cmd == "gates":
        raise SystemExit(gates())
    print(__doc__)
    raise SystemExit(2)
