#!/usr/bin/env python3
"""t3f0_rcheck.py: post-retarget check of one Tier-3 f0 episode (no clean / Tier-1 trajectory is read). NEW FILE
(2026-10-24): a copy of scripts/t3opus_rcheck_v2.py (left untouched) for the f0 root data/human_motion_data_tier3_f0
(Han Zhou's tier3_frame0_supports_20261024: opus_v2 + frame-0 object-pose correction). Changes: roots/paths; G6 also
reports the packer result against the opus_v2 frame references (not gating); G7 against the f0 root's SOURCE_FILES.sha256; G8 by the f0 class
(16 floor-only: geometry-free file; 14 support: the support USDA is written AFTER the retarget by t3f0_supports.py, so
G8 only requires that no *_g1_support.usda exists and reports the support file; t3f0_build_root.py finalize gates it);
NEW G9 (robot side = the opus parquet of the same episode: same human, same code, the object never enters the IK;
continuous columns <= 1e-3, the integer ik_num_iterations within 2) and
NEW G10 (object track = the static frame chain of t3f0_frames.py that places the supports: <= 1e-5 m / 1e-3 deg).
Outputs tmp_analysis/tier3/f0/retarget/epNN_rcheck.json and tmp_analysis/tier3/f0/retarget/parquet_lock.json.

Checks the G1 parquet that scripts/retarget/soma_to_g1.py wrote from Han's f0 reconstruction:
  G1  parquet complete and readable (PAR1 magic, pyarrow), exactly 1 row, schema equal to the Tier-3 agent_T1 example
      parquet (data/human_motion_data_tier3_agentT1, ep14, the pilot's validated retarget) and to the Tier-2 parquet of
      the same episode; metadata (object name, fps 30, coord_frame, schema_version) as the Tier-2 parquet;
  G2  frame counts: every per-frame column has T = episodes.json frames = Tier-1/Tier-2 manifest frames (803-style
      check: e.g. ep04 has 803 frames everywhere);
  G3  finite values in every pose column;
  G4  object mesh / URDF paths exist and lie inside this episode's folder of the Tier-3 root (no other root referenced);
  G5  object_body track = ONE rigid map of the input poses.npy (plane levelling + frame-0 anchoring + constant lift):
      residual <= 1e-4 m / 1e-4 rad; object root keeps a constant body-relative pose;
  G6  packer emulation (kit tools/pack_submission.py functions, released Tier-2 scored grid, no clean read): the parquet
      track vs --frame-references frame_references_f0 is ACCEPTED; vs Han's release frame_references (= s1 track) and
      vs the released Tier-2 input it is REJECTED;
  G7  the 7 input files are unchanged (sha256 = SOURCE_FILES.sha256 of the build);
  G8  scene (v1/v2 text below; the f0 rule is in the header above):
      floor-only (released support defines no Cylinder/Cube): as v1, the geometry-free file is in the root and no
      *_g1_support.usda -> scene.support=false;
      support_pending (Tier-1/2 runs used scene.support=true): the Tier-3 root holds NO support file of any name (no
      official Tier-3 support exists; none is taken from Tier-1/2) -> trained with scene.support=false, flagged
      SUPPORT_MISSING_FLOOR_ONLY.
Reported, not gating: IK error, hand-contact frames, robot root height, object lowest-vertex height at frame 0 in the G1
world, the retarget's object ground lift (from its log), the same numbers of the Tier-2 parquet.
All gates pass -> the parquet sha256 is recorded in tmp_analysis/tier3/f0/retarget/parquet_lock.json (training
preflight requires it). Writes retarget/epNN_rcheck.json.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t3f0_rcheck.py --ep N [--log retarget log] [--no-write]
     ... --selftest  (the same checks on the pilot's agent_T1 ep14 parquet with its own frame references; writes nothing)
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, re, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb.trajectory import read_episode, episode_parquet  # noqa: E402

_spec = importlib.util.spec_from_file_location("pack_submission_ro", KIT / "tools/pack_submission.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

DATA = ME / "data/human_motion_data_tier3_f0"
FREF = DATA / "frame_references_f0"
HAN_FREF = Path("/root/autodl-tmp/han.zhou/tier3/release/tier3_frame0_supports_20261024/frame_references")
OPUS = ME / "data/human_motion_data_tier3_opus"
OPUS_FREF = OPUS / "frame_references_opus"
FRAMES = ME / "tmp_analysis/tier3/f0/frames"
SUPPORT_EPS = (1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27, 28)
ROBOT_COLS = ("robot_root_position", "robot_root_wxyz", "robot_joint_positions", "ee_pose_w", "ik_error_per_frame",
              "frame_task_errors")
ROBOT_TOL = 1e-3  # GPU float noise between two retargets of the same human was 1.3e-5 (ep14 agent_T1 vs opus)
ITER_TOL = 2      # ik_num_iterations is an integer count: float noise at the convergence test may move it by 1
EXAMPLE = ME / "data/human_motion_data_tier3_agentT1/whole_body/soma/sequence_id=2026-05-28_14-44-09_round_table_flip_08/robot_name=g1/data.parquet"
T2ROOT = ME / "data/human_motion_data_tier2"
REL_T2 = ME / "data/video_to_data_challenge/track_2/tier_2_synthetic_noise"
MAN_T1 = ME / "data/human_motion_data/whole_body/soma/track2_manifest.json"
A = ME / "tmp_analysis/tier3/f0"
RDIR = A / "retarget"
LOCK = RDIR / "parquet_lock.json"
INPUT_FILES = ["camera.json", "episode.json", "ground_plane.json", "object_pose_xyz_wxyz.npy", "poses.npy",
               "soma_params.npz", "reconstructed_mesh/output_aligned.glb"]
POSE_COLS = ("robot_root_position", "robot_root_wxyz", "robot_joint_positions", "object_body_position", "object_body_wxyz",
             "object_root_position", "object_root_axis_angle", "ee_pose_w")
FRAME_COLS = ("robot_root_position", "robot_joint_positions", "object_body_position", "object_root_position", "ik_error_per_frame")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def qmul(a, b):  # wxyz
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


def qconj(q): return q * np.array([1, -1, -1, -1])


def qrot(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


def qangle(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True); b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    d = qmul(qconj(a), b)
    return 2 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))


def aa_to_q(aa):
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.where(th > 1e-12, aa / np.maximum(th, 1e-12), np.array([1.0, 0, 0]))
    return np.concatenate([np.cos(th / 2), np.sin(th / 2) * axis], -1)


def mat_to_q(Rm):  # wxyz (Shepperd)
    out = np.empty(Rm.shape[:-2] + (4,))
    for i, M in enumerate(Rm.reshape(-1, 3, 3)):
        tr = np.trace(M)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2; q = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2; q = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2; q = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2; q = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
        q = np.asarray(q); out.reshape(-1, 4)[i] = q / np.linalg.norm(q)
    return out


def col(t, name):
    return np.asarray(t.column(name).to_pylist()[0], dtype=np.float64)


def strings(t, name):
    v = t.column(name).to_pylist()[0]
    return [v] if isinstance(v, str) else [str(x) for x in (v or [])]


def summary(t):
    ik = col(t, "ik_error_per_frame")
    hc = col(t, "hand_contact_active")
    rp = col(t, "robot_root_position")
    return {"ik_error_median_p95_max": [float(np.median(ik)), float(np.percentile(ik, 95)), float(ik.max())],
            "hand_contact_frames_left_right": [int((hc[0] > 0.5).sum()), int((hc[1] > 0.5).sum())],
            "robot_root_z_median_min_max": [float(np.median(rp[:, 2])), float(rp[:, 2].min()), float(rp[:, 2].max())]}


def check(ep: int, root: Path, fref: Path, log_path: Path | None, selftest: bool) -> dict:
    man = json.loads((root / "whole_body/soma/track2_manifest.json").read_text()) if not selftest else \
        json.loads((ME / "tmp_analysis/tier3/train90k/track2_manifest_ep14.json").read_text())
    row = next(r for r in man.values() if int(r["episode_index"]) == ep)
    seq, T = row["sequence_id"], int(row["frames"])
    sdir = root / "whole_body/soma" / f"sequence_id={seq}"
    parq = sdir / "robot_name=g1/data.parquet"
    rep = {"episode": ep, "sequence_id": seq, "parquet": str(parq), "written": time.strftime("%F %T"), "selftest": selftest, "gates": {}}

    def gate(name, ok, detail):
        rep["gates"][name] = {"ok": bool(ok), "detail": detail}

    # G1
    try:
        with parq.open("rb") as f:
            head = f.read(4); f.seek(-4, os.SEEK_END); tail = f.read(4)
        t = pq.read_table(parq)
    except Exception as error:  # noqa: BLE001
        gate("G1_readable", False, f"{type(error).__name__}: {str(error)[:200]}")
        rep["all_ok"] = False
        return rep
    t_ex = pq.read_table(EXAMPLE)
    t2p = T2ROOT / "whole_body/soma" / f"sequence_id={seq}" / "robot_name=g1/data.parquet"
    t2 = pq.read_table(t2p)
    meta = {c: str(t.column(c).to_pylist()[0]) for c in ("object_name", "fps", "coord_frame", "source_dataset", "schema_version", "motion_kind")}
    meta2 = {c: str(t2.column(c).to_pylist()[0]) for c in meta}
    gate("G1_parquet", head == b"PAR1" == tail and t.num_rows == 1 and t.schema.equals(t_ex.schema) and t.schema.equals(t2.schema)
         and meta == meta2 and meta["fps"] == "30.0",
         f"PAR1 {head == b'PAR1' == tail}, rows {t.num_rows}, schema = agent_T1 example {t.schema.equals(t_ex.schema)}, "
         f"= Tier-2 ep {t.schema.equals(t2.schema)}; metadata {meta} (Tier-2 equal: {meta == meta2})")
    rep["parquet_sha256"] = sha(parq)
    # G2
    frames = {c: int(len(t.column(c).to_pylist()[0])) for c in FRAME_COLS}
    t1man = {int(r["episode_index"]): int(r["frames"]) for r in json.loads(MAN_T1.read_text()).values()}
    ej = json.loads((sdir / "episode.json").read_text())
    gate("G2_frames", set(frames.values()) == {T} and int(ej["frames"]) == T == t1man[ep] and len(np.load(sdir / "poses.npy")) == T,
         f"columns {frames}; episodes.json {ej['frames']}; manifest {T}; Tier-1 manifest {t1man[ep]}")
    # G3
    fin = {c: bool(np.isfinite(col(t, c)).all()) for c in POSE_COLS}
    gate("G3_finite", all(fin.values()), fin)
    # G4
    paths = strings(t, "object_mesh_paths") + strings(t, "object_urdf_paths")
    gate("G4_assets", paths and all(Path(p).is_file() and Path(p).resolve().is_relative_to(sdir.resolve()) for p in paths), paths)
    # G5
    P = np.load(sdir / "poses.npy")
    pos, q = P[:, :3, 3], mat_to_q(P[:, :3, :3])
    bp, bq = col(t, "object_body_position")[:, 0], col(t, "object_body_wxyz")[:, 0]
    qT = qmul(bq[0], qconj(q[0])); qT /= np.linalg.norm(qT)
    tT = bp[0] - qrot(qT, pos[0])
    rp_ = np.linalg.norm(qrot(np.broadcast_to(qT, q.shape), pos) + tT - bp, axis=-1)
    ra_ = qangle(qmul(np.broadcast_to(qT, q.shape), q), bq)
    rr = aa_to_q(col(t, "object_root_axis_angle")); rpos = col(t, "object_root_position")
    loc_q = qmul(qconj(bq), rr); loc_p = qrot(qconj(bq), rpos - bp)
    dlp, dlq = float(np.abs(loc_p - loc_p[0]).max()), float(qangle(loc_q, np.broadcast_to(loc_q[0], loc_q.shape)).max())
    gate("G5_rigid_map", rp_.max() <= 1e-4 and ra_.max() <= 1e-4 and dlp <= 1e-4 and dlq <= 1e-4,
         f"residual {rp_.max():.2e} m / {ra_.max():.2e} rad; root body-relative drift {dlp:.2e} m / {dlq:.2e} rad; "
         f"map rotation {np.degrees(2 * np.arccos(min(1.0, abs(qT[0])))):.2f} deg, translation {np.round(tT, 4).tolist()}")
    # G6 packer emulation on the released Tier-2 scored grid (participant input; no clean read)
    t2rel = read_episode(episode_parquet(REL_T2, ep), ep)
    names, idx = t2rel.object_names, t2rel.frame_index
    ev = np.concatenate([bp[idx], bq[idx][:, 1:], bq[idx][:, :1]], -1)
    emu = {}
    for tag, rroot in (("frame_references_f0", fref), ("han_release_frame_references_s1", HAN_FREF),
                       ("frame_references_opus_v2", OPUS_FREF), ("released_tier2_input", REL_T2)):
        try:
            ref = ps.input_frame_reference(rroot, ep, idx, 1, names)
            _, pe, ae, _ = ps.convert_track2_frame(ev, ev, ref)
            emu[tag] = {"result": "accepted", "resid_m": pe, "resid_rad": ae}
        except (ValueError, OSError, KeyError) as error:
            emu[tag] = {"result": "REJECTED", "msg": str(error)[:160]}
    rep["packer_emulation"] = emu
    want_ok = emu["frame_references_f0"]["result"] == "accepted" and emu["released_tier2_input"]["result"] == "REJECTED"
    if not selftest:
        want_ok = want_ok and emu["han_release_frame_references_s1"]["result"] == "REJECTED"
    # reported only: the opus_v2 frame references (a policy trained on f0 must not be packed against opus_v2)
    gate("G6_packer", want_ok, {k: v["result"] + (f" ({v['resid_m']:.1e} m)" if v["result"] == "accepted" else "") for k, v in emu.items()})
    # G7 inputs unchanged
    if not selftest:
        recorded = {}
        for line in (root / "SOURCE_FILES.sha256").read_text().splitlines():
            h, _, rel = line.partition("  ")
            if rel.startswith(f"reconstructed_data/{seq}/"):
                recorded[rel.split("/", 2)[2]] = h
        same = {f: sha(sdir / f) == recorded.get(f) for f in INPUT_FILES}
        gate("G7_inputs_unchanged", all(same.values()) and len(recorded) == len(INPUT_FILES), same)
        # G8 scene (f0)
        sup = root / "whole_body/reconstructed_stage" / f"{seq}_support.usda"
        g1sup = root / "whole_body/reconstructed_stage" / f"{seq}_g1_support.usda"
        geo = sup.is_file() and bool(re.search(r"def (Cylinder|Cube)", sup.read_text(errors="replace")))
        if ep not in SUPPORT_EPS:
            rep["scene_class"] = "floor-only"
            gate("G8_scene", sup.is_file() and not geo and not g1sup.exists(),
                 f"floor-only: {sup.name}: {'present' if sup.is_file() else 'MISSING'}, geometry {geo} -> scene.support=false")
        else:
            rep["scene_class"] = "support"
            gate("G8_scene", not g1sup.exists(),
                 f"support episode: {sup.name} {'present, geometry ' + str(geo) if sup.exists() else 'not yet written'} "
                 f"(gated at finalize); {g1sup.name} {'PRESENT' if g1sup.exists() else 'absent'}")
        # G9 robot side = the opus parquet of the same episode (same human, same code; the object never enters the IK)
        op = OPUS / "whole_body/soma" / f"sequence_id={seq}" / "robot_name=g1/data.parquet"
        to = pq.read_table(op)
        diffs = {c: float(np.abs(col(t, c) - col(to, c)).max()) for c in ROBOT_COLS}
        it_d = np.abs(col(t, "ik_num_iterations") - col(to, "ik_num_iterations"))
        hc_n, hc_o = col(t, "hand_contact_active"), col(to, "hand_contact_active")
        rep["vs_opus_parquet"] = {"parquet": str(op), "parquet_sha256": sha(op), "robot_max_abs_diff": diffs,
                                  "ik_num_iterations_max_diff_frames": [float(it_d.max()), int((it_d > 0).sum())],
                                  "hand_contact_frames_f0_left_right": [int((hc_n[0] > 0.5).sum()), int((hc_n[1] > 0.5).sum())],
                                  "hand_contact_frames_opus_left_right": [int((hc_o[0] > 0.5).sum()), int((hc_o[1] > 0.5).sum())],
                                  "hand_contact_frames_changed": int((np.abs(hc_n - hc_o) > 0.5).sum()),
                                  "object_pos_change_cm_max": round(100 * float(np.abs(col(t, "object_body_position") - col(to, "object_body_position")).max()), 2)}
        gate("G9_robot_equal_opus", max(diffs.values()) <= ROBOT_TOL and it_d.max() <= ITER_TOL,
             f"max |robot f0 - robot opus| {max(diffs.values()):.1e} (tol {ROBOT_TOL}); ik_num_iterations differ on "
             f"{int((it_d > 0).sum())} frames by <= {it_d.max():.0f} (tol {ITER_TOL}); hand-contact frames changed "
             f"{rep['vs_opus_parquet']['hand_contact_frames_changed']}")
        # G10 object track = the static frame chain used for the supports (t3f0_frames.py)
        fz = FRAMES / f"ep{ep:02d}.npz"
        if fz.is_file():
            with np.load(fz) as d:
                fpos, fq = d["obj_pos"], mat_to_q(d["obj_R"])
            pe = float(np.abs(bp - fpos).max()); ae = float(np.degrees(qangle(bq, fq).max()))
            gate("G10_frame_chain", pe <= 1e-5 and ae <= 1e-3, f"object track vs t3f0_frames chain: {pe:.1e} m, {ae:.1e} deg")
        else:
            gate("G10_frame_chain", False, f"missing {fz}")
    # reported only
    rep["t3"], rep["t2_same_episode"] = summary(t), summary(t2)
    try:
        import trimesh
        objf = Path(strings(t, "object_mesh_paths")[0])
        V = np.asarray(trimesh.load(str(objf), force="mesh", process=False).vertices, dtype=np.float64)
        z0 = (qrot(np.broadcast_to(bq[0], (len(V), 4)), V) + bp[0])[:, 2]
        rep["object_lowest_vertex_z_frame0_cm"] = float(z0.min() * 100)
    except Exception as error:  # noqa: BLE001
        rep["object_lowest_vertex_z_frame0_cm"] = f"n/a ({type(error).__name__})"
    if log_path and log_path.is_file():
        m = re.search(r"object frame-0 ground correction: (.*)", log_path.read_text(errors="replace"))
        rep["retarget_object_ground_correction"] = m.group(1).strip() if m else None
    rep["all_ok"] = all(g["ok"] for g in rep["gates"].values())
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rep = check(14, ME / "data/human_motion_data_tier3_agentT1", ME / "tmp_analysis/tier3/frame_refs_agentT1", None, True)
        for k, v in rep["gates"].items():
            print(f"SELFTEST {k} {'ok' if v['ok'] else 'FAIL'} {str(v['detail'])[:220]}")
        print(f"SELFTEST all_ok {rep['all_ok']} (agent_T1 ep14 parquet sha {rep.get('parquet_sha256', '?')[:16]}...)")
        return 0 if rep["all_ok"] else 1
    rep = check(a.ep, DATA, FREF, a.log, False)
    for k, v in rep["gates"].items():
        print(f"RCHECK ep{a.ep:02d} {k} {'ok' if v['ok'] else 'FAIL'} {str(v['detail'])[:220]}")
    print(f"RCHECK ep{a.ep:02d} all_ok {rep['all_ok']} parquet sha256 {rep.get('parquet_sha256', '?')}")
    if not a.no_write:
        RDIR.mkdir(parents=True, exist_ok=True)
        (RDIR / f"ep{a.ep:02d}_rcheck.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
        if rep["all_ok"]:
            lock = json.loads(LOCK.read_text()) if LOCK.is_file() else {}
            old = lock.get(f"{a.ep:02d}")
            if old and old["sha256"] != rep["parquet_sha256"]:
                print(f"RCHECK ep{a.ep:02d}: parquet_lock already holds a DIFFERENT sha256 {old['sha256'][:16]}...: not overwritten")
                return 1
            lock[f"{a.ep:02d}"] = {"sha256": rep["parquet_sha256"], "parquet": rep["parquet"], "checked": rep["written"]}
            tmp = LOCK.with_suffix(".tmp"); tmp.write_text(json.dumps(lock, indent=1) + "\n"); os.replace(tmp, LOCK)
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
