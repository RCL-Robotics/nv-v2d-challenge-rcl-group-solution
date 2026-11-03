#!/usr/bin/env python3
"""t3f0_build_root.py: build data/human_motion_data_tier3_f0 from Han Zhou's Tier-3 release tier3_frame0_supports_20261024
(opus_v2 + frame-0 object-pose correction, variant opus_v2_frame0_f3_c2_shift; supports for 14 episodes). NEW FILE
(2026-10-24); scripts/t3opus_build_data.py is left untouched.

Read-only on the sources: Han's release (/root/autodl-tmp/han.zhou/tier3/release/tier3_frame0_supports_20261024) and our
opus root data/human_motion_data_tier3_opus (only to show which inputs are unchanged). Writes only under the new root and
tmp_analysis/tier3/f0/build/. No clean (Tier-1) trajectory and no Tier-1/Tier-2 support value is read.

  inputs   (step A)
    1. episodes.json -> episode index <-> sequence id, equal (index, sequence, object, frames, task) to the opus root's
       track2_manifest.json (which was checked against the Tier-1/Tier-2 numbering at the opus build); object_variant
       opus_v2_frame0_f3_c2_shift for all 30; a 'support' entry exactly for the 14 support episodes.
    2. byte copy (sha256 verified) of the 30 reconstructed_data/<seq>/ folders (exactly the 7 input files) into
       whole_body/soma/sequence_id=<seq>/ (the soma_to_g1.py data_folder layout). camera.json, ground_plane.json,
       soma_params.npz and reconstructed_mesh/output_aligned.glb must be byte-identical to the opus root (README: unchanged);
       episode.json may differ only in object_variant and support; poses.npy / object_pose_xyz_wxyz.npy are the new track.
    3. array checks of the new track (shapes, finite, orthonormal, poses.npy == object_pose_xyz_wxyz.npy) + a GT-free summary
       of how far the new track moved from the opus_v2 track.
    4. frame_references_f0/episode_NNNNNN.npz: the packer's --frame-references = the new poses.npy in the released Track-2
       world (same format/method as frame_references_opus: episode_index, coordinate_frame released_track2_world,
       frame_index 0..T-1, pose_xyzw (T,1,7) float64). The release's own frame_references/ are the s1 track (unchanged) and
       are NOT used.
    5. whole_body/soma/track2_manifest.json (the opus root's manifest, only 'source' changed).
    6. whole_body/reconstructed_stage/<seq>_support.usda for the 16 floor-only episodes = byte copy of the opus root's
       geometry-free file (no Cylinder/Cube -> scene.support=false, as in every earlier Tier-3 run). The 14 support files
       are written later by scripts/t3f0_supports.py.
    7. han_release_docs/: README.md, episodes.json, agent_run/** and supports/** (README_supports.md, index.json,
       contact_report.json, fit_overview.jpg, <seq>/{support.json, support_mesh.glb, support_world.obj, fit_overlay.jpg}).
    8. SOURCE_FILES.sha256 (release-relative paths), tmp_analysis/tier3/f0/build/build_report.json.
  finalize (step F, after scripts/t3f0_retarget_all.sh, t3f0_supports.py and t3f0_physcheck.py)
    every parquet passed t3f0_rcheck.py and is in tmp_analysis/tier3/f0/retarget/parquet_lock.json (sha256 = file now);
    SCENE_SUPPORT.json + SUPPORT_FILES.sha256 from the supports step match the files; the physical-check report exists for
    the 14; then FILES.sha256 over the root and READY (write-once).
Run: conda_envs/flashchord/bin/python -I -B scripts/t3f0_build_root.py inputs|finalize [--check-only]
"""
from __future__ import annotations

import hashlib, json, os, re, shutil, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
SRC = Path("/root/autodl-tmp/han.zhou/tier3/release/tier3_frame0_supports_20261024")
OPUS = ME / "data/human_motion_data_tier3_opus"
ROOT = ME / "data/human_motion_data_tier3_f0"
SOMA = ROOT / "whole_body/soma"
STAGE = ROOT / "whole_body/reconstructed_stage"
FREF = ROOT / "frame_references_f0"
DOCS = ROOT / "han_release_docs"
A = ME / "tmp_analysis/tier3/f0"
BUILD = A / "build"
INPUT_FILES = ["camera.json", "episode.json", "ground_plane.json", "object_pose_xyz_wxyz.npy", "poses.npy",
               "soma_params.npz", "reconstructed_mesh/output_aligned.glb"]
UNCHANGED = ["camera.json", "ground_plane.json", "soma_params.npz", "reconstructed_mesh/output_aligned.glb"]
SUPPORT_EPS = (1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27, 28)
VARIANT = "opus_v2_frame0_f3_c2_shift"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def files(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


def copy_verified(src: Path, dst: Path, check_only: bool) -> str:
    s = sha(src)
    if dst.exists():
        if sha(dst) != s:
            raise SystemExit(f"ABORT: {dst} exists and differs from {src}; refusing to overwrite")
        return s
    if check_only:
        return s
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha(dst) != s:
        raise SystemExit(f"ABORT: copy of {src} does not verify")
    return s


def has_geometry(usda: Path) -> bool:
    return bool(re.search(r"def (Cylinder|Cube)", usda.read_text(errors="replace")))


def opus_sha() -> dict[str, str]:
    out = {}
    for line in (OPUS / "SOURCE_FILES.sha256").read_text().splitlines():
        h, _, rel = line.partition("  ")
        out[rel.strip()] = h
    return out


def inputs(check_only: bool) -> int:
    if (ROOT / "READY").exists() and not check_only:
        print(f"{ROOT}/READY exists: write-once root (use --check-only)")
        return 0
    assert (OPUS / "READY").is_file(), "opus root not READY"
    rep: dict = {"written": time.strftime("%F %T"), "source": str(SRC), "root": str(ROOT), "checks": {}, "episodes": {}}
    bad: list = []

    def check(name, ok, detail):
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            bad.append(name)

    eps = json.loads((SRC / "episodes.json").read_text())
    man_o = json.loads((OPUS / "whole_body/soma/track2_manifest.json").read_text())
    by_o = {int(v["episode_index"]): v for v in man_o.values()}
    idx = sorted(int(e["episode_index"]) for e in eps)
    check("episodes_json_0_29", idx == list(range(30)) and len({e["sequence_id"] for e in eps}) == 30, f"{len(eps)} entries")
    mism = [f"ep{int(e['episode_index']):02d}" for e in eps
            if (by_o[int(e["episode_index"])]["sequence_id"], by_o[int(e["episode_index"])]["object"],
                int(by_o[int(e["episode_index"])]["frames"]), by_o[int(e["episode_index"])]["task"])
            != (e["sequence_id"], e["object"], int(e["frames"]), e.get("task", ""))]
    check("numbering_equal_opus_manifest", not mism, mism or "30/30 index, sequence, object, frames, task equal")
    check("object_variant", all(e.get("object_variant") == VARIANT for e in eps), sorted({e.get("object_variant") for e in eps}))
    sup = sorted(int(e["episode_index"]) for e in eps if "support" in e)
    check("support_episodes", tuple(sup) == SUPPORT_EPS, sup)
    osha = opus_sha()
    src_lines: list[str] = []
    for e in sorted(eps, key=lambda x: int(x["episode_index"])):
        i, seq, T = int(e["episode_index"]), e["sequence_id"], int(e["frames"])
        sdir, ddir = SRC / "reconstructed_data" / seq, SOMA / f"sequence_id={seq}"
        row = {"sequence_id": seq, "object": e["object"], "frames": T, "support_dir": e.get("support")}
        fs = files(sdir)
        if fs != sorted(INPUT_FILES):
            check(f"ep{i:02d}_file_set", False, f"source files {fs}")
            continue
        for f in INPUT_FILES:
            s = copy_verified(sdir / f, ddir / f, check_only)
            src_lines.append(f"{s}  reconstructed_data/{seq}/{f}")
            row.setdefault("sha256", {})[f] = s
        same = {f: row["sha256"][f] == osha.get(f"reconstructed_data/{seq}/{f}") for f in INPUT_FILES}
        row["equal_opus"] = same
        check(f"ep{i:02d}_unchanged_inputs", all(same[f] for f in UNCHANGED),
              {f: same[f] for f in UNCHANGED})
        ej = json.loads((sdir / "episode.json").read_text())
        ejo = json.loads((OPUS / "whole_body/soma" / f"sequence_id={seq}" / "episode.json").read_text())
        dk = sorted(k for k in set(ej) | set(ejo) if ej.get(k) != ejo.get(k))
        check(f"ep{i:02d}_episode_json_diff", set(dk) <= {"object_variant", "support"} and ej.get("object_variant") == VARIANT
              and int(ej.get("frames", -1)) == T and int(ej.get("episode_index", -1)) == i and ej.get("sequence_id") == seq,
              f"differs from opus in {dk}")
        P = np.load(sdir / "poses.npy")
        X = np.load(sdir / "object_pose_xyz_wxyz.npy")
        Po = np.load(OPUS / "whole_body/soma" / f"sequence_id={seq}" / "poses.npy")
        sp = np.load(sdir / "soma_params.npz", allow_pickle=False)
        ok_shapes = P.shape == (T, 4, 4) and X.shape == (T, 7) and sp["poses"].shape[0] == T and Po.shape == P.shape
        fin = bool(np.isfinite(P).all() and np.isfinite(X).all())
        orth = float(np.abs(np.einsum("tji,tjk->tik", P[:, :3, :3], P[:, :3, :3]) - np.eye(3)).max())
        bottom = float(np.abs(P[:, 3] - np.array([0, 0, 0, 1.0])).max())
        rot = R.from_matrix(P[:, :3, :3])
        Rx = R.from_quat(X[:, [4, 5, 6, 3]])
        ang = np.degrees((rot.inv() * Rx).magnitude())
        dpos = float(np.abs(X[:, :3] - P[:, :3, 3]).max())
        check(f"ep{i:02d}_arrays", ok_shapes and fin and orth < 1e-4 and bottom < 1e-9 and dpos < 1e-5 and ang.max() < 1e-3,
              f"T={T} shapes {ok_shapes} finite {fin} orth {orth:.1e} last-row {bottom:.0e} xyz_wxyz {dpos:.1e} m {ang.max():.1e} deg")
        # GT-free: how far did the frame-0 correction move the track from opus_v2?
        dp = np.linalg.norm(P[:, :3, 3] - Po[:, :3, 3], axis=1)
        da = np.degrees((R.from_matrix(Po[:, :3, :3]).inv() * rot).magnitude())
        row["vs_opus_v2"] = {"pos_change_cm_median_max": [round(100 * float(np.median(dp)), 2), round(100 * float(dp.max()), 2)],
                             "rot_change_deg_median_max": [round(float(np.median(da)), 2), round(float(da.max()), 2)],
                             "frame0_pos_change_cm": round(100 * float(dp[0]), 2), "frame0_rot_change_deg": round(float(da[0]), 2),
                             "identical": bool(np.array_equal(P, Po))}
        # release frame references = unchanged s1 files (not used)
        rf = SRC / "frame_references" / f"episode_{i:06d}.npz"
        row["release_frame_ref_equal_opus_release_file"] = sha(rf) == sha(Path(str(SRC).replace(
            "tier3_frame0_supports_20261024", "tier3_opus_20261021")) / "frame_references" / rf.name)
        # frame reference from the NEW track (format of frame_references_opus)
        out = FREF / f"episode_{i:06d}.npz"
        xyzw = np.concatenate([P[:, :3, 3], rot.as_quat()], -1)[:, None, :].astype(np.float64)
        if not check_only:
            FREF.mkdir(parents=True, exist_ok=True)
            if out.exists():
                with np.load(out) as d:
                    if not (np.array_equal(d["pose_xyzw"], xyzw) and int(d["episode_index"]) == i):
                        raise SystemExit(f"ABORT: {out} exists with different content")
            else:
                np.savez(out, episode_index=np.int64(i), coordinate_frame=np.asarray("released_track2_world"),
                         frame_index=np.arange(T, dtype=np.int64), pose_xyzw=xyzw)
            with np.load(out) as d, np.load(OPUS / "frame_references_opus" / out.name) as o:
                fmt = {k: (d[k].dtype, d[k].shape) for k in d.files} == {k: (o[k].dtype, o[k].shape) for k in o.files} and \
                    str(d["coordinate_frame"]) == str(o["coordinate_frame"]) == "released_track2_world"
                q2 = d["pose_xyzw"][:, 0, 3:]
                rt_deg = float(np.degrees((R.from_quat(q2).inv() * rot).magnitude()).max())
                rt_pos = float(np.abs(d["pose_xyzw"][:, 0, :3] - P[:, :3, 3]).max())
                # same method: the formula rebuilds frame_references_opus bit-exactly from the opus poses.npy
                xo = np.concatenate([Po[:, :3, 3], R.from_matrix(Po[:, :3, :3]).as_quat()], -1)[:, None, :].astype(np.float64)
                same_method = bool(np.array_equal(o["pose_xyzw"], xo))
            check(f"ep{i:02d}_frame_ref", fmt and rt_pos == 0.0 and rt_deg < 1e-3 and same_method,
                  f"format = frame_references_opus {fmt}; position exact {rt_pos == 0.0}; rotation round trip {rt_deg:.1e} deg; "
                  f"formula reproduces frame_references_opus bit-exactly {same_method}")
            row["frame_ref_sha256"] = sha(out)
        # scene file for floor-only episodes (geometry-free byte copy from the opus root)
        if i not in SUPPORT_EPS:
            osup = OPUS / "whole_body/reconstructed_stage" / f"{seq}_support.usda"
            geo = has_geometry(osup)
            check(f"ep{i:02d}_floor_only_file", osup.is_file() and not geo, f"{osup.name} geometry {geo}")
            row["support"] = "floor-only (geometry-free file; scene.support=false)"
            row["support_sha256"] = copy_verified(osup, STAGE / osup.name, check_only)
        else:
            row["support"] = "support (USDA written by t3f0_supports.py)"
        rep["episodes"][f"{i:02d}"] = row
    check("release_frame_refs_unchanged", all(r["release_frame_ref_equal_opus_release_file"] for r in rep["episodes"].values()),
          "release frame_references/ byte-identical to the opus release (s1 track; not used)")
    check("tracks_changed", all(not r["vs_opus_v2"]["identical"] for r in rep["episodes"].values()),
          [k for k, r in rep["episodes"].items() if r["vs_opus_v2"]["identical"]] or "30/30 tracks differ from opus_v2")
    # manifest
    man = {}
    for k, v in man_o.items():
        v = dict(v)
        v["source"] = ("han.zhou tier3_frame0_supports_20261024 (opus_v2 + frame-0 object-pose correction, variant "
                       f"{VARIANT}; s1 human)")
        man[k] = v
    if not check_only:
        p = SOMA / "track2_manifest.json"
        txt = json.dumps(man, indent=2) + "\n"
        if p.exists() and p.read_text() != txt:
            raise SystemExit(f"ABORT: {p} exists with different content")
        p.write_text(txt)
        doc_list = ["README.md", "episodes.json"] + [f"agent_run/{f}" for f in files(SRC / "agent_run")] + \
                   [f"supports/{f}" for f in files(SRC / "supports")]
        for f in doc_list:
            src_lines.append(f"{copy_verified(SRC / f, DOCS / f, False)}  {f}")
        agent_same = all(osha.get(f) == sha(SRC / f) for f in doc_list if f.startswith("agent_run/") and osha.get(f))
        check("agent_run_equal_opus", agent_same, "agent_run/ files with an opus copy are byte-identical")
        (ROOT / "SOURCE_FILES.sha256").write_text("\n".join(src_lines) + "\n")
        again = [sha(SOMA / ("sequence_id=" + r["sequence_id"]) / f) == sha(SRC / "reconstructed_data" / r["sequence_id"] / f)
                 for r in rep["episodes"].values() for f in INPUT_FILES]
        check("copy_reverified", all(again) and len(again) == 30 * len(INPUT_FILES), f"{sum(again)}/{len(again)} files sha256 = source")
    BUILD.mkdir(parents=True, exist_ok=True)
    rep["bad"] = bad
    (BUILD / ("inputs_check.json" if check_only else "build_report.json")).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for k, v in rep["checks"].items():
        if not k.startswith("ep") or not v["ok"]:
            print(f"CHECK {k} {'ok' if v['ok'] else 'FAIL'} {str(v['detail'])[:300]}")
    print(f"episode checks: {sum(v['ok'] for k, v in rep['checks'].items() if k.startswith('ep'))}/"
          f"{sum(1 for k in rep['checks'] if k.startswith('ep'))} ok")
    for k, r in rep["episodes"].items():
        print(f"ep{k} T={r['frames']} vs_opus_v2 {r['vs_opus_v2']}")
    if bad:
        print(f"FAILED: {bad}")
        return 1
    return 0


def finalize(check_only: bool) -> int:
    if (ROOT / "READY").exists() and not check_only:
        print(f"{ROOT}/READY exists: write-once root")
        return 0
    br = json.loads((BUILD / "build_report.json").read_text())
    assert not br["bad"], "inputs step failed"
    lock = json.loads((A / "retarget/parquet_lock.json").read_text())
    scene = json.loads((ROOT / "SCENE_SUPPORT.json").read_text())
    phys = json.loads((A / "physcheck/physcheck_summary.json").read_text())
    bad, checks = [], {}

    def check(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            bad.append(name)

    for k, r in br["episodes"].items():
        i, seq = int(k), r["sequence_id"]
        parq = SOMA / f"sequence_id={seq}" / "robot_name=g1/data.parquet"
        lk = lock.get(k)
        check(f"ep{k}_parquet_locked", lk is not None and parq.is_file() and sha(parq) == lk["sha256"],
              f"{lk['sha256'][:16] if lk else None}")
        rc = json.loads((A / "retarget" / f"ep{k}_rcheck.json").read_text())
        check(f"ep{k}_rcheck", rc.get("all_ok") is True, {g: v["ok"] for g, v in rc["gates"].items()})
        u = STAGE / f"{seq}_support.usda"
        srow = scene["episodes"][k]
        if i in SUPPORT_EPS:
            check(f"ep{k}_support_file", u.is_file() and has_geometry(u) and srow["scene_support"] is True
                  and sha(u) == srow["support_sha256"], srow.get("support_sha256", "")[:16])
            check(f"ep{k}_physcheck_present", k in phys["episodes"], "physical checks reported")
        else:
            check(f"ep{k}_floor_only", u.is_file() and not has_geometry(u) and srow["scene_support"] is False, u.name)
        check(f"ep{k}_no_g1_named_support", not (STAGE / f"{seq}_g1_support.usda").exists(), "only <seq>_support.usda")
    rep = {"written": time.strftime("%F %T"), "checks": checks, "bad": bad}
    if bad or check_only:
        print(json.dumps({k: v for k, v in checks.items() if not v["ok"]}, indent=1))
        print("FAILED" if bad else "check-only ok")
        return 1 if bad else 0
    fl = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name not in ("READY", "FILES.sha256"))
    (ROOT / "FILES.sha256").write_text("".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in fl))
    fs = sha(ROOT / "FILES.sha256")
    rep.update(files=len(fl), files_sha256_sha256=fs)
    (BUILD / "finalize_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    sup_true = [int(k) for k, v in scene["episodes"].items() if v["scene_support"]]
    (ROOT / "READY").write_text(
        f"READY {time.strftime('%F %T %Z')}: Tier-3 f0 root, 30 episodes. Inputs = byte copies of Han Zhou's "
        f"tier3_frame0_supports_20261024/reconstructed_data (object track = opus_v2 + frame-0 object-pose correction, {VARIANT}; "
        f"human, camera, ground plane, mesh byte-identical to the opus release) + frame_references_f0 (new poses.npy, "
        f"released_track2_world) + track2_manifest.json + han_release_docs (incl. supports/). robot_name=g1 parquets = "
        f"soma_to_g1.py @ f2923d61 re-run on these inputs (the object enters object_*, hand_contact_active and the frame-0 "
        f"object ground lift), each passed scripts/t3f0_rcheck.py (tmp_analysis/tier3/f0/retarget/parquet_lock.json). "
        f"reconstructed_stage: scene.support=true for {sup_true} (Cylinder/Cube primitives from Han's support.json "
        f"T_world_mesh_contact + top_box / support mesh, scripts/t3f0_supports.py, SCENE_SUPPORT.json), geometry-free "
        f"floor-only files for the other 16. {len(fl)} files, FILES.sha256 sha256 {fs}. Reports: {BUILD}/\n")
    print(f"READY written; FILES.sha256 sha256 {fs}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    co = "--check-only" in sys.argv
    if mode == "inputs":
        raise SystemExit(inputs(co))
    if mode == "finalize":
        raise SystemExit(finalize(co))
    raise SystemExit("usage: t3f0_build_root.py inputs|finalize [--check-only]")
