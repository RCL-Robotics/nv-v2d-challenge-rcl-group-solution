#!/usr/bin/env python3
"""k3fix_build_data.py (NOISY-ONLY, CPU only): step 2 of the K3fix pilot build. Run with the kit python (pyarrow 26, the
writer of data/human_motion_data_tier2_k3):
    CUDA_VISIBLE_DEVICES= nice -n 19 conda_envs/v2dkit/bin/python -I -B scripts/k3fix_build_data.py

Input: tmp_analysis/tier2/k3fix_pilot/build/k3fix_tracks.npz (scripts/k3fix_build.py tracks; sha256 pinned in
tracks.json). Reads no clean data.

1. Stage (build/stage/epNN_data.parquet, all 30 episodes): the Tier-2 G1 parquet with ONLY object_body_position/wxyz
   and object_root_position/axis_angle replaced by T o K3fix, exactly as tmp_analysis/tier2/kalman/eval/
   e2_build_eval_inputs.py part (a) (the code below is that part, with the K3fix full-frame track as input; the
   k3easy reviewer's regen_step2_parquets.py is the same code). For the 26 episodes K3fix leaves unchanged the staged
   file must be byte-identical to data/human_motion_data_tier2_k3 (proves the build path); for ep03/04/09/17 it must
   differ from both the K3 and the raw Tier-2 parquet.
2. Data root data/human_motion_data_tier2_k3fix (new path; refuses to run if its READY exists):
   - whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet: the staged K3fix file for ep03/04/09/17, a byte copy
     of the K3 root's parquet for the other 26 (sha256 verified)
   - whole_body/soma/sequence_id=<seq>/ground_plane.json, whole_body/reconstructed_stage/<seq>_support.usda and
     whole_body/soma/track2_manifest.json: byte copies of the K3 root's (which equal data/human_motion_data_tier2's;
     verified). Object assets stay referenced in data/human_motion_data_tier2 (absolute paths inside the parquets), as
     for the K3 root.
3. Frame references tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix/episode_XXXXXX.npz (for the LOCAL clean diagnostic's
   --frame-references): kalman/eval/frame_refs_k3's file for the 26 unchanged episodes (byte copy), the K3fix pose
   (visible frames, released Tier-2 world, released frame grid) for ep03/04/09/17, as e2 part (b).
4. Kit packer emulation, read-only (tools/pack_submission.py convert_track2_frame + input_frame_reference), all 30:
   the K3fix parquet's object track vs its frame_refs_k3fix entry must be ACCEPTED (one rigid transform); vs the raw
   released Tier-2 input it must be REJECTED; vs the old frame_refs_k3 it is REJECTED for the 4 changed episodes and
   accepted for the other 26.
5. READY marker (first line "READY ..."; then "sha256  relative path" for every parquet) + build/data_build.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
PILOT = ME / "tmp_analysis/tier2/k3fix_pilot"
BUILD = PILOT / "build"
STAGE = BUILD / "stage"
FREF = PILOT / "frame_refs_k3fix"
FREF_K3 = ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3"
K3ROOT = ME / "data/human_motion_data_tier2_k3"
T2G1 = ME / "data/human_motion_data_tier2"
ROOT = ME / "data/human_motion_data_tier2_k3fix"
CHANGED = (3, 4, 9, 17)
OBJ_COLS = ("object_body_position", "object_body_wxyz", "object_root_position", "object_root_axis_angle")

spec = importlib.util.spec_from_file_location("e2ro", ME / "tmp_analysis/tier2/kalman/eval/e2_build_eval_inputs.py")
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)  # top level only loads npz files + the kit modules (read-only)
ps = e2.ps  # the kit's tools/pack_submission.py module, loaded read-only by e2


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def inside(p: Path, roots) -> Path:
    rp = p.resolve()
    if not any(str(rp).startswith(str(r.resolve()) + "/") for r in roots):
        raise SystemExit(f"refusing to write {rp}")
    return rp


def build_one(ep: int, seq: str, k3_full: np.ndarray, dst: Path) -> dict:
    """e2_build_eval_inputs.py part (a), verbatim maths, with the K3fix full-frame track."""
    raw = e2.read_episode(e2.episode_parquet(e2.T2, ep), ep)
    idx = np.asarray(raw.frame_index, dtype=np.int64)
    rel = raw.pose_xyzw[:, 0]
    src = e2.G1 / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
    tab = pq.read_table(src)
    bp, bq = e2.col(tab, "object_body_position")[:, 0], e2.col(tab, "object_body_wxyz")[:, 0]
    assert len(bp) == len(k3_full), (ep, len(bp), len(k3_full))
    rq = e2.xyzw_to_wxyz(rel[:, 3:])
    qT = e2.qmul(bq[idx[0]], e2.qconj(rq[0])); qT /= np.linalg.norm(qT)
    tT = bp[idx[0]] - e2.qrot(qT, rel[0, :3])
    res_p = float(np.linalg.norm(e2.qrot(np.broadcast_to(qT, rq.shape), rel[:, :3]) + tT - bp[idx], axis=-1).max())
    res_a = float(e2.qangle(e2.qmul(np.broadcast_to(qT, rq.shape), rq), bq[idx]).max())
    assert res_p < 1e-5 and res_a < 1e-5, (ep, res_p, res_a)
    kq = e2.xyzw_to_wxyz(k3_full[:, 3:])
    nq = e2.qmul(np.broadcast_to(qT, kq.shape), kq); nq /= np.linalg.norm(nq, axis=-1, keepdims=True)
    npos = e2.qrot(np.broadcast_to(qT, kq.shape), k3_full[:, :3]) + tT
    rp, rr = e2.col(tab, "object_root_position"), e2.aa_to_q(e2.col(tab, "object_root_axis_angle"))
    loc_q = e2.qmul(e2.qconj(bq), rr)
    loc_p = e2.qrot(e2.qconj(bq), rp - bp)
    new_rq = e2.qmul(nq, loc_q); new_rq /= np.linalg.norm(new_rq, axis=-1, keepdims=True)
    new_rp = npos + e2.qrot(nq, loc_p)
    out = tab
    for cname, arr in (("object_body_position", npos[:, None]), ("object_body_wxyz", nq[:, None]),
                       ("object_root_position", new_rp), ("object_root_axis_angle", e2.q_to_aa(new_rq))):
        i = out.schema.get_field_index(cname)
        out = out.set_column(i, out.schema.field(i), pa.array([arr.astype(np.float32).tolist()], type=out.schema.field(i).type))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, inside(dst, [STAGE]))
    back = pq.read_table(dst)
    assert back.num_rows == 1 and back.schema.equals(tab.schema)
    for cname in tab.column_names:
        if cname not in OBJ_COLS:
            assert back.column(cname).equals(tab.column(cname)), cname
    bpn = e2.col(back, "object_body_position")[:, 0]
    assert np.abs(bpn - npos).max() < 1e-5
    ch_p = np.linalg.norm(npos - bp, axis=-1) * 100
    ch_a = np.degrees(e2.qangle(nq, bq))
    return {"fit_resid_m": res_p, "fit_resid_rad": res_a, "ref_change_vs_raw_mean_cm": float(ch_p.mean()),
            "ref_change_vs_raw_max_cm": float(ch_p.max()), "ref_change_vs_raw_mean_deg": float(ch_a.mean()),
            "ref_change_vs_raw_max_deg": float(ch_a.max())}


def packer_checks(ep: int, parquet: Path, raw) -> dict:
    idx = np.asarray(raw.frame_index, dtype=np.int64)
    rel = raw.pose_xyzw[:, 0]
    back = pq.read_table(parquet)
    bpn, bqn = e2.col(back, "object_body_position")[:, 0], e2.col(back, "object_body_wxyz")[:, 0]
    ev = np.concatenate([bpn[idx], e2.wxyz_to_xyzw(bqn[idx])], -1)
    out = {}
    fr = ps.input_frame_reference(FREF, ep, idx, 1, raw.object_names)
    try:
        _, pe, ae, _ = ps.convert_track2_frame(ev, ev, fr)
        out.update(vs_frame_refs_k3fix="accepted", resid_m=pe, resid_rad=ae)
    except ValueError as error:
        out.update(vs_frame_refs_k3fix="REJECTED", error=str(error)[:200])
    for label, ref in (("vs_released_raw", rel), ("vs_old_frame_refs_k3", ps.input_frame_reference(FREF_K3, ep, idx, 1, raw.object_names))):
        try:
            ps.convert_track2_frame(ev, ev, ref)
            out[label] = "accepted"
        except ValueError:
            out[label] = "REJECTED"
    return out


def main() -> int:
    t0 = time.time()
    if (ROOT / "READY").exists():
        raise SystemExit(f"{ROOT}/READY exists: the root is built (never rewritten)")
    tr = json.loads((BUILD / "tracks.json").read_text())
    if not tr.get("all_ok") or tr["changed_episodes"] != list(CHANGED) or tr["k3fix_tracks_npz_sha256"] != sha(BUILD / "k3fix_tracks.npz"):
        raise SystemExit("build/tracks.json not ok, changed set differs, or k3fix_tracks.npz changed")
    gates = json.loads((BUILD / "gates.json").read_text())
    if gates.get("k3fix_tracks_npz_sha256") != tr["k3fix_tracks_npz_sha256"]:
        raise SystemExit("gates.json was computed on another k3fix_tracks.npz")
    TR = np.load(BUILD / "k3fix_tracks.npz")
    manifest = json.loads((K3ROOT / "whole_body/soma/track2_manifest.json").read_text())
    assert sha(K3ROOT / "whole_body/soma/track2_manifest.json") == sha(T2G1 / "whole_body/soma/track2_manifest.json")
    ep_to_seq = {int(v["episode_index"]): v["sequence_id"] for v in manifest.values()}
    assert sorted(ep_to_seq) == list(range(30))
    rows, ok = {}, True
    STAGE.mkdir(parents=True, exist_ok=True)
    FREF.mkdir(parents=True, exist_ok=True)
    (ROOT / "whole_body/soma").mkdir(parents=True, exist_ok=True)
    (ROOT / "whole_body/reconstructed_stage").mkdir(parents=True, exist_ok=True)
    for ep in range(30):
        seq = ep_to_seq[ep]
        rel_pq = Path(f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet")
        k3p, rawp = K3ROOT / rel_pq, T2G1 / rel_pq
        staged = STAGE / f"ep{ep:02d}_data.parquet"
        r = build_one(ep, seq, TR[f"K3fixfull__{ep}"], staged)
        r.update({"sequence_id": seq, "changed": ep in CHANGED, "staged_sha256": sha(staged), "k3_root_sha256": sha(k3p),
                  "raw_tier2_sha256": sha(rawp)})
        if ep in CHANGED:
            r["stage_ok"] = r["staged_sha256"] != r["k3_root_sha256"] and r["staged_sha256"] != r["raw_tier2_sha256"]
            src = staged
        else:
            r["stage_ok"] = r["staged_sha256"] == r["k3_root_sha256"]          # identical by construction, shown
            src = k3p
        dst = inside(ROOT / rel_pq, [ROOT])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        r["root_sha256"] = sha(dst)
        r["root_ok"] = r["root_sha256"] == (r["staged_sha256"] if ep in CHANGED else r["k3_root_sha256"])
        for name in ("ground_plane.json",):
            s, d = K3ROOT / rel_pq.parent.parent / name, inside(ROOT / rel_pq.parent.parent / name, [ROOT])
            shutil.copy2(s, d)
            r["ground_plane_ok"] = sha(d) == sha(s) == sha(T2G1 / rel_pq.parent.parent / name)
        usda = f"whole_body/reconstructed_stage/{seq}_support.usda"
        shutil.copy2(K3ROOT / usda, inside(ROOT / usda, [ROOT]))
        r["support_ok"] = sha(ROOT / usda) == sha(K3ROOT / usda) == sha(T2G1 / usda)
        # frame references
        raw = e2.read_episode(e2.episode_parquet(e2.T2, ep), ep)
        idx = np.asarray(raw.frame_index, dtype=np.int64)
        vis = TR[f"vis__{ep}"]
        assert np.array_equal(np.flatnonzero(vis), idx), ep
        fdst = inside(FREF / f"episode_{ep:06d}.npz", [FREF])
        if ep in CHANGED:
            pose = TR[f"K3fixpose__{ep}"]
            assert np.abs(TR[f"K3fixfull__{ep}"][idx] - pose).max() < 1e-12
            np.savez(fdst, episode_index=np.asarray(ep), coordinate_frame=np.asarray("released_track2_world"),
                     frame_index=idx, pose_xyzw=pose[:, None, :].astype(np.float64))
        else:
            shutil.copy2(FREF_K3 / f"episode_{ep:06d}.npz", fdst)
            with np.load(fdst) as z:
                r["frame_ref_k3_vs_k3fix_pose_max_abs"] = float(np.abs(z["pose_xyzw"][:, 0] - TR[f"K3fixpose__{ep}"]).max())
        r["frame_ref_sha256"] = sha(fdst)
        r.update(packer_checks(ep, dst, raw))
        r["packer_ok"] = (r["vs_frame_refs_k3fix"] == "accepted" and r["vs_released_raw"] == "REJECTED"
                          and r["vs_old_frame_refs_k3"] == ("REJECTED" if ep in CHANGED else "accepted")
                          and (ep in CHANGED or r["frame_ref_k3_vs_k3fix_pose_max_abs"] < 1e-12))
        r["ok"] = all(r[k] for k in ("stage_ok", "root_ok", "ground_plane_ok", "support_ok", "packer_ok"))
        ok &= r["ok"]
        rows[f"{ep:02d}"] = r
        print(f"ep{ep:02d} {'CHANGED ' if ep in CHANGED else 'unchanged'} staged {r['staged_sha256'][:12]} K3 {r['k3_root_sha256'][:12]} "
              f"root {r['root_sha256'][:12]} | packer vs own refs {r['vs_frame_refs_k3fix']} ({r.get('resid_m', float('nan')):.1e} m) "
              f"vs raw {r['vs_released_raw']} vs old K3 refs {r['vs_old_frame_refs_k3']} | change vs raw mean "
              f"{r['ref_change_vs_raw_mean_cm']:.2f} cm {r['ref_change_vs_raw_mean_deg']:.1f} deg | ok {r['ok']}", flush=True)
    man_dst = inside(ROOT / "whole_body/soma/track2_manifest.json", [ROOT])
    shutil.copy2(K3ROOT / "whole_body/soma/track2_manifest.json", man_dst)
    files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file())
    expect = 1 + 30 * 3
    n_ok = len(files) == expect and sha(man_dst) == sha(T2G1 / "whole_body/soma/track2_manifest.json")
    rep = {"written": time.strftime("%F %T"), "noisy_only": True, "root": str(ROOT), "python": sys.executable,
           "pyarrow": pa.__version__, "numpy": np.__version__, "k3fix_tracks_npz_sha256": tr["k3fix_tracks_npz_sha256"],
           "gates_pass_clips": gates.get("pass_clips"), "changed": list(CHANGED), "n_files": len(files),
           "n_files_expected_before_READY": expect, "manifest_ok": bool(n_ok), "episodes": rows,
           "all_ok": bool(ok and n_ok), "seconds": round(time.time() - t0, 1)}
    (BUILD / "data_build.json").write_text(json.dumps(rep, indent=1, default=float))
    if not rep["all_ok"]:
        print(f"FAILED (n_files {len(files)}/{expect}); no READY written; see {BUILD / 'data_build.json'}")
        return 1
    lines = [f"READY {time.strftime('%F %T')} CST: K3fix root (noisy Tier-2 body + K3fix object; K3fix = "
             f"tmp_analysis/tier2/k3_rootcause/ep04/k3fix_v1_precommit.py sha256 6f126f3e, fixAsC). ep03/04/09/17 rebuilt by "
             f"scripts/k3fix_build_data.py (e2_build_eval_inputs.py part (a) maths) from tmp_analysis/tier2/k3fix_pilot/build/"
             f"k3fix_tracks.npz (sha256 {tr['k3fix_tracks_npz_sha256'][:16]}); the other 26 parquets, ground planes, support "
             f"stages and the manifest are byte copies of data/human_motion_data_tier2_k3 (26 parquets regenerated from K3fix "
             f"byte-identical; ground planes / support / manifest = data/human_motion_data_tier2). Object assets referenced in "
             f"data/human_motion_data_tier2. Kit packer: 30/30 accepted vs tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix, "
             f"30/30 rejected vs the released raw input. Details: tmp_analysis/tier2/k3fix_pilot/build/data_build.json.",
             "# sha256  path (relative to this root) of every parquet"]
    for ep in range(30):
        seq = ep_to_seq[ep]
        lines.append(f"{rows[f'{ep:02d}']['root_sha256']}  whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet")
    tmp = ROOT / "READY.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(ROOT / "READY")
    print(f"all ok; wrote {ROOT / 'READY'} ({round(time.time() - t0, 1)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
