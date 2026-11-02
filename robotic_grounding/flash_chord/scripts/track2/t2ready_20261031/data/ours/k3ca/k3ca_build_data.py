#!/usr/bin/env python3
"""k3ca_build_data.py (NOISY-ONLY, CPU only): step 3 of the K3CA build. Run with the kit python (pyarrow 26, the writer of
data/human_motion_data_tier2_k3 and _k3fix):
    CUDA_VISIBLE_DEVICES= nice -n 19 conda_envs/v2dkit/bin/python -I -B scripts/k3ca_build_data.py

Input: tmp_analysis/tier2/k3ca/build/k3ca_tracks.npz (scripts/k3ca_build.py tracks; sha256 pinned in tracks.json and
gates.json). Reads no clean data.

1. Build-path proof: every one of the 30 K3fix-root parquets is regenerated from the K3fix track
   (k3fix_pilot/build/k3fix_tracks.npz) with the code below (= scripts/k3fix_build_data.py build_one = e2 part (a)
   maths, copied because that function only writes into the K3fix pilot's stage directory); each must be
   byte-identical to data/human_motion_data_tier2_k3fix.
2. Stage (build/stage/k3ca_epNN.parquet): the same code with the K3CA full-frame track for the episodes that pass the
   K3CA gates (gates.json pass_episodes; expected 00 05 15 19 22 26 28). Each must differ from the K3fix-root parquet
   and from the raw Tier-2 parquet; every non-object column is verified unchanged.
3. Data root data/human_motion_data_tier2_k3ca (new path; refuses to run if its READY exists): the staged K3CA parquet
   for the K3CA episodes; for the other 23 a byte copy of the K3fix root's parquet (so the root is a complete 30-episode
   root; those 23 are the K3fix input and are NOT part of the K3CA arm). Ground planes, support stages and the manifest
   are byte copies of the K3fix root's (= data/human_motion_data_tier2's; verified). Object assets stay referenced in
   data/human_motion_data_tier2 (absolute paths inside the parquets), as for the K3 and K3fix roots.
4. Frame references tmp_analysis/tier2/k3ca/frame_refs_k3ca/episode_XXXXXX.npz (for the evaluator's / packer's
   --frame-references): the K3CA pose (visible frames, released Tier-2 world, released frame grid) for the K3CA
   episodes; byte copies of k3fix_pilot/frame_refs_k3fix for the other 23.
5. Kit packer emulation, read-only (tools/pack_submission.py convert_track2_frame + input_frame_reference), all 30:
   the root parquet's object track vs its frame_refs_k3ca entry must be ACCEPTED; vs the released raw input REJECTED;
   vs frame_refs_k3fix REJECTED for the K3CA episodes and accepted for the other 23.
6. READY marker (first line "READY ..."; then "sha256  relative path" for every parquet) + build/data_build.json.
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
K = ME / "tmp_analysis/tier2/k3ca"
BUILD = K / "build"
STAGE = BUILD / "stage"
FREF = K / "frame_refs_k3ca"
FREF_FIX = ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix"
FIXROOT = ME / "data/human_motion_data_tier2_k3fix"
FIXROOT_READY_SHA = "1233398f02e61b51fb9bc13072b1607ec3964556cc20828ac56f571b1640dac8"
K3FIX_NPZ = ME / "tmp_analysis/tier2/k3fix_pilot/build/k3fix_tracks.npz"
K3FIX_NPZ_SHA = "1a4bb14c9da05ea322c5d04947cbca074905f46596d20a511c3e8fd5a1221f7e"
T2G1 = ME / "data/human_motion_data_tier2"
ROOT = ME / "data/human_motion_data_tier2_k3ca"
EXPECTED = [0, 5, 15, 19, 22, 26, 28]
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


def build_one(ep: int, seq: str, full: np.ndarray, dst: Path) -> dict:
    """scripts/k3fix_build_data.py build_one (= e2_build_eval_inputs.py part (a)), verbatim maths, any full-frame track."""
    raw = e2.read_episode(e2.episode_parquet(e2.T2, ep), ep)
    idx = np.asarray(raw.frame_index, dtype=np.int64)
    rel = raw.pose_xyzw[:, 0]
    src = e2.G1 / f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
    tab = pq.read_table(src)
    bp, bq = e2.col(tab, "object_body_position")[:, 0], e2.col(tab, "object_body_wxyz")[:, 0]
    assert len(bp) == len(full), (ep, len(bp), len(full))
    rq = e2.xyzw_to_wxyz(rel[:, 3:])
    qT = e2.qmul(bq[idx[0]], e2.qconj(rq[0])); qT /= np.linalg.norm(qT)
    tT = bp[idx[0]] - e2.qrot(qT, rel[0, :3])
    res_p = float(np.linalg.norm(e2.qrot(np.broadcast_to(qT, rq.shape), rel[:, :3]) + tT - bp[idx], axis=-1).max())
    res_a = float(e2.qangle(e2.qmul(np.broadcast_to(qT, rq.shape), rq), bq[idx]).max())
    assert res_p < 1e-5 and res_a < 1e-5, (ep, res_p, res_a)
    kq = e2.xyzw_to_wxyz(full[:, 3:])
    nq = e2.qmul(np.broadcast_to(qT, kq.shape), kq); nq /= np.linalg.norm(nq, axis=-1, keepdims=True)
    npos = e2.qrot(np.broadcast_to(qT, kq.shape), full[:, :3]) + tT
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


def packer_checks(ep: int, parquet: Path, raw, changed: bool) -> dict:
    idx = np.asarray(raw.frame_index, dtype=np.int64)
    rel = raw.pose_xyzw[:, 0]
    back = pq.read_table(parquet)
    bpn, bqn = e2.col(back, "object_body_position")[:, 0], e2.col(back, "object_body_wxyz")[:, 0]
    ev = np.concatenate([bpn[idx], e2.wxyz_to_xyzw(bqn[idx])], -1)
    out = {}
    fr = ps.input_frame_reference(FREF, ep, idx, 1, raw.object_names)
    try:
        _, pe, ae, _ = ps.convert_track2_frame(ev, ev, fr)
        out.update(vs_frame_refs_k3ca="accepted", resid_m=pe, resid_rad=ae)
    except ValueError as error:
        out.update(vs_frame_refs_k3ca="REJECTED", error=str(error)[:200])
    for label, ref in (("vs_released_raw", rel), ("vs_frame_refs_k3fix", ps.input_frame_reference(FREF_FIX, ep, idx, 1, raw.object_names))):
        try:
            ps.convert_track2_frame(ev, ev, ref)
            out[label] = "accepted"
        except ValueError:
            out[label] = "REJECTED"
    out["packer_ok"] = (out["vs_frame_refs_k3ca"] == "accepted" and out["vs_released_raw"] == "REJECTED"
                        and out["vs_frame_refs_k3fix"] == ("REJECTED" if changed else "accepted"))
    return out


def main() -> int:
    t0 = time.time()
    if (ROOT / "READY").exists():
        raise SystemExit(f"{ROOT}/READY exists: the root is built (never rewritten)")
    tr = json.loads((BUILD / "tracks.json").read_text())
    gates = json.loads((BUILD / "gates.json").read_text())
    npz_sha = sha(BUILD / "k3ca_tracks.npz")
    if not tr.get("all_ok") or tr.get("k3ca_tracks_npz_sha256") != npz_sha or gates.get("k3ca_tracks_npz_sha256") != npz_sha:
        raise SystemExit("tracks.json not ok, or tracks.json / gates.json computed on another k3ca_tracks.npz")
    if not gates.get("checks_ok"):
        raise SystemExit("gates.json checks failed")
    if sha(FIXROOT / "READY") != FIXROOT_READY_SHA or sha(K3FIX_NPZ) != K3FIX_NPZ_SHA:
        raise SystemExit("K3fix root READY or k3fix_tracks.npz changed")
    changed = sorted(gates["pass_episodes"])
    if changed != EXPECTED:
        print(f"NOTE: K3CA pass episodes {changed} != expected {EXPECTED}; only passing episodes are changed")
    TRF = np.load(K3FIX_NPZ)
    TRC = np.load(BUILD / "k3ca_tracks.npz")
    manifest = json.loads((FIXROOT / "whole_body/soma/track2_manifest.json").read_text())
    assert sha(FIXROOT / "whole_body/soma/track2_manifest.json") == sha(T2G1 / "whole_body/soma/track2_manifest.json")
    ep_to_seq = {int(v["episode_index"]): v["sequence_id"] for v in manifest.values()}
    assert sorted(ep_to_seq) == list(range(30))
    rows, ok = {}, True
    for d in (STAGE, FREF, ROOT / "whole_body/soma", ROOT / "whole_body/reconstructed_stage"):
        d.mkdir(parents=True, exist_ok=True)
    for ep in range(30):
        seq = ep_to_seq[ep]
        is_ca = ep in changed
        rel_pq = Path(f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet")
        fixp, rawp = FIXROOT / rel_pq, T2G1 / rel_pq
        r = {"sequence_id": seq, "changed": is_ca, "k3fix_root_sha256": sha(fixp), "raw_tier2_sha256": sha(rawp)}
        # 1. build-path proof
        st_fix = STAGE / f"k3fix_ep{ep:02d}.parquet"
        build_one(ep, seq, TRF[f"K3fixfull__{ep}"], st_fix)
        r["regen_k3fix_sha256"] = sha(st_fix)
        r["build_path_ok"] = r["regen_k3fix_sha256"] == r["k3fix_root_sha256"]
        # 2. K3CA stage
        if is_ca:
            st = STAGE / f"k3ca_ep{ep:02d}.parquet"
            r.update(build_one(ep, seq, TRC[f"K3CAfull__{ep}"], st))
            r["staged_sha256"] = sha(st)
            r["stage_ok"] = r["staged_sha256"] != r["k3fix_root_sha256"] and r["staged_sha256"] != r["raw_tier2_sha256"]
            src = st
        else:
            r["stage_ok"] = True
            src = fixp
        # 3. root
        dst = inside(ROOT / rel_pq, [ROOT])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        r["root_sha256"] = sha(dst)
        r["root_ok"] = r["root_sha256"] == (r["staged_sha256"] if is_ca else r["k3fix_root_sha256"])
        gp = rel_pq.parent.parent / "ground_plane.json"
        shutil.copy2(FIXROOT / gp, inside(ROOT / gp, [ROOT]))
        r["ground_plane_ok"] = sha(ROOT / gp) == sha(FIXROOT / gp) == sha(T2G1 / gp)
        usda = f"whole_body/reconstructed_stage/{seq}_support.usda"
        shutil.copy2(FIXROOT / usda, inside(ROOT / usda, [ROOT]))
        r["support_ok"] = sha(ROOT / usda) == sha(FIXROOT / usda) == sha(T2G1 / usda)
        # 4. frame references
        raw = e2.read_episode(e2.episode_parquet(e2.T2, ep), ep)
        idx = np.asarray(raw.frame_index, dtype=np.int64)
        fdst = inside(FREF / f"episode_{ep:06d}.npz", [FREF])
        if is_ca:
            vis = TRC[f"vis__{ep}"]
            assert np.array_equal(np.flatnonzero(vis), idx), ep
            pose = TRC[f"K3CApose__{ep}"]
            assert np.abs(TRC[f"K3CAfull__{ep}"][idx] - pose).max() < 1e-12
            np.savez(fdst, episode_index=np.asarray(ep), coordinate_frame=np.asarray("released_track2_world"),
                     frame_index=idx, pose_xyzw=pose[:, None, :].astype(np.float64))
        else:
            shutil.copy2(FREF_FIX / f"episode_{ep:06d}.npz", fdst)
            r["frame_ref_same_as_k3fix"] = sha(fdst) == sha(FREF_FIX / f"episode_{ep:06d}.npz")
        r["frame_ref_sha256"] = sha(fdst)
        # 5. packer
        r.update(packer_checks(ep, dst, raw, is_ca))
        r["ok"] = all(r[k] for k in ("build_path_ok", "stage_ok", "root_ok", "ground_plane_ok", "support_ok", "packer_ok")) \
            and r.get("frame_ref_same_as_k3fix", True)
        ok &= r["ok"]
        rows[f"{ep:02d}"] = r
        print(f"ep{ep:02d} {'K3CA    ' if is_ca else 'as K3fix'} regen-K3fix byte-identical {r['build_path_ok']} | root "
              f"{r['root_sha256'][:12]} | packer own refs {r['vs_frame_refs_k3ca']} vs raw {r['vs_released_raw']} vs K3fix refs "
              f"{r['vs_frame_refs_k3fix']}" + (f" | change vs raw mean {r['ref_change_vs_raw_mean_cm']:.2f} cm "
                                                 f"{r['ref_change_vs_raw_mean_deg']:.1f} deg" if is_ca else "") + f" | ok {r['ok']}", flush=True)
    man_dst = inside(ROOT / "whole_body/soma/track2_manifest.json", [ROOT])
    shutil.copy2(FIXROOT / "whole_body/soma/track2_manifest.json", man_dst)
    files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file())
    expect = 1 + 30 * 3
    n_ok = len(files) == expect and sha(man_dst) == sha(T2G1 / "whole_body/soma/track2_manifest.json")
    rep = {"written": time.strftime("%F %T"), "noisy_only": True, "root": str(ROOT), "python": sys.executable,
           "pyarrow": pa.__version__, "numpy": np.__version__, "k3ca_tracks_npz_sha256": npz_sha,
           "k3fix_tracks_npz_sha256": K3FIX_NPZ_SHA, "k3fix_root_ready_sha256": FIXROOT_READY_SHA,
           "k3ca_episodes": changed, "n_files": len(files), "n_files_expected_before_READY": expect, "manifest_ok": bool(n_ok),
           "episodes": rows, "all_ok": bool(ok and n_ok), "seconds": round(time.time() - t0, 1)}
    (BUILD / "data_build.json").write_text(json.dumps(rep, indent=1, default=float))
    if not rep["all_ok"]:
        print(f"FAILED (n_files {len(files)}/{expect}); no READY written; see {BUILD / 'data_build.json'}")
        return 1
    lines = [f"READY {time.strftime('%F %T')} CST: K3CA root (noisy Tier-2 body + K3CA object for episodes "
             f"{' '.join(f'{e:02d}' for e in changed)}; K3CA = scripts/k3ca_build.py: the K3fix track pulled toward the released "
             f"track only where raw hand contact or support penetration requires it). Those parquets are built by "
             f"scripts/k3ca_build_data.py (e2 part (a) maths, the K3fix root's build path, re-verified byte-identical on all 30 "
             f"K3fix parquets) from tmp_analysis/tier2/k3ca/build/k3ca_tracks.npz (sha256 {npz_sha[:16]}); gates G1-G4 pass "
             f"(tmp_analysis/tier2/k3ca/build/gates.json). The other 23 parquets, all ground planes, support stages and the "
             f"manifest are byte copies of data/human_motion_data_tier2_k3fix (READY sha256 {FIXROOT_READY_SHA[:8]}); those 23 "
             f"are the K3fix input, not part of the K3CA arm. Object assets referenced in data/human_motion_data_tier2. Kit packer: "
             f"30/30 accepted vs tmp_analysis/tier2/k3ca/frame_refs_k3ca, 30/30 rejected vs the released raw input. Details: "
             f"tmp_analysis/tier2/k3ca/build/data_build.json.", "# sha256  path (relative to this root) of every parquet"]
    for ep in range(30):
        lines.append(f"{rows[f'{ep:02d}']['root_sha256']}  whole_body/soma/sequence_id={ep_to_seq[ep]}/robot_name=g1/data.parquet")
    tmp = ROOT / "READY.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(ROOT / "READY")
    print(f"all ok; wrote {ROOT / 'READY'} sha256 {sha(ROOT / 'READY')} ({round(time.time() - t0, 1)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
