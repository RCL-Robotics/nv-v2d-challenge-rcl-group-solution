#!/usr/bin/env python3
"""t3opus_build_data.py: build data/human_motion_data_tier3_opus from Han Zhou's Tier-3 release tier3_opus_20261021.

Read-only on the source (/root/autodl-tmp/han.zhou/tier3/release/tier3_opus_20261021); writes only under the new root
$ME/data/human_motion_data_tier3_opus and $ME/tmp_analysis/tier3/opus_easy/build/. No clean (Tier-1) trajectory is read.

  1. episodes.json -> episode index <-> sequence id; checked against our Tier-1 and Tier-2 manifests (index, sequence,
     object, frame count) and the named Tier-1 numbering (ep14 round_table_flip_08, ep08 skinny_wood_chair_push_10, ...).
  2. byte copy (shutil.copy2, sha256 verified) of the 30 reconstructed_data/<seq>/ folders into
     whole_body/soma/sequence_id=<seq>/ (the soma_to_g1.py data_folder layout; the retargeter later adds object/ and
     robot_name=g1/ there). The file set must be exactly the 7 input files.
  3. whole_body/soma/track2_manifest.json (eval_all.py / helper format, from Han's episodes.json).
  4. whole_body/reconstructed_stage/<seq>_support.usda: byte copies of the RELEASED Track-2 support surfaces ONLY for
     the 16 floor-only episodes, whose released file defines no Cylinder/Cube (so the floor-only rule gives
     scene.support=false, exactly as for Tier 1/2). The 14 support episodes get NO file (support_pending): their
     released disks sit in the organisers' g1_retarget_world under the clean rest pose; a decision is needed first.
  5. frame_references_opus/episode_NNNNNN.npz: the packer's --frame-references = Han's opus object track (poses.npy)
     in the released Track-2 world (coordinate_frame released_track2_world, all frames). The release's own
     frame_references/ are byte-identical to the s1 release (they hold the s1 track, not opus) and are NOT used.
  6. han_release_docs/: README.md, episodes.json, agent_run/<seq>/{REPORT.md,TOOLS.md,solve.py} (provenance; solve.py
     is reconstruction code that must be in the cited commit).
  7. SOURCE_FILES.sha256, build/build_report.json, READY (only when every check passes).
Run: conda_envs/v2d/bin/python -I -B scripts/t3opus_build_data.py [--check-only]
"""
from __future__ import annotations

import hashlib, json, re, shutil, sys, time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
SRC = Path("/root/autodl-tmp/han.zhou/tier3/release/tier3_opus_20261021")
S1 = Path("/root/autodl-tmp/han.zhou/tier3/release/tier3_s1_20261020")
ROOT = ME / "data/human_motion_data_tier3_opus"
SOMA = ROOT / "whole_body/soma"
STAGE = ROOT / "whole_body/reconstructed_stage"
FREF = ROOT / "frame_references_opus"
DOCS = ROOT / "han_release_docs"
BUILD = ME / "tmp_analysis/tier3/opus_easy/build"
DS = ME / "data/video_to_data_challenge/track_2"
T2REL = DS / "tier_2_synthetic_noise"
MAN_T1 = ME / "data/human_motion_data/whole_body/soma/track2_manifest.json"
MAN_T2 = ME / "data/human_motion_data_tier2/whole_body/soma/track2_manifest.json"
INPUT_FILES = ["camera.json", "episode.json", "ground_plane.json", "object_pose_xyz_wxyz.npy", "poses.npy",
               "soma_params.npz", "reconstructed_mesh/output_aligned.glb"]
NAMED = {14: "round_table_flip_08", 8: "skinny_wood_chair_push_10", 7: "iron_round_table_02",
         20: "rice_cooker_clockwise_04", 23: "toy_stroller_back_forth_ground_04", 18: "blue_trash_can_drag_sideways_08",
         22: "step_ladder_drag_03", 0: "tall_bar_stool_tip_up_04"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def files(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


def copy_verified(src: Path, dst: Path, check_only: bool, log: list) -> str:
    s = sha(src)
    if dst.exists():
        if sha(dst) != s:
            raise SystemExit(f"ABORT: {dst} exists and differs from {src}; refusing to overwrite")
        return s
    if check_only:
        log.append(f"would copy {src} -> {dst}")
        return s
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha(dst) != s:
        raise SystemExit(f"ABORT: copy of {src} does not verify")
    return s


def has_geometry(usda: Path) -> bool:
    return bool(re.search(r"def (Cylinder|Cube)", usda.read_text(errors="replace")))


def main() -> int:
    check_only = "--check-only" in sys.argv
    if (ROOT / "READY").exists() and not check_only:
        print(f"{ROOT}/READY exists: already built (nothing to do; use --check-only to re-verify)")
        return 0
    rep: dict = {"written": time.strftime("%F %T"), "source": str(SRC), "root": str(ROOT), "checks": {}}
    log: list = []
    bad: list = []

    def check(name, ok, detail):
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            bad.append(name)

    eps = json.loads((SRC / "episodes.json").read_text())
    idx = [int(e["episode_index"]) for e in eps]
    check("episodes_json_0_29", sorted(idx) == list(range(30)) and len({e["sequence_id"] for e in eps}) == 30,
          f"{len(eps)} entries, indices {sorted(idx)[:3]}..{sorted(idx)[-1]}")
    check("episodes_json_equal_s1", sha(SRC / "episodes.json") == sha(S1 / "episodes.json"), "byte-identical to tier3_s1_20261020")
    m1, m2 = json.loads(MAN_T1.read_text()), json.loads(MAN_T2.read_text())
    t1 = {int(r["episode_index"]): r for r in m1.values()}
    t2 = {int(r["episode_index"]): r for r in m2.values()}
    mism = []
    for e in eps:
        i = int(e["episode_index"])
        for tag, t in (("t1", t1), ("t2", t2)):
            r = t.get(i, {})
            if (r.get("sequence_id"), r.get("object"), int(r.get("frames", -1))) != (e["sequence_id"], e["object"], int(e["frames"])):
                mism.append(f"ep{i:02d} vs {tag}: {r.get('sequence_id')}/{r.get('object')}/{r.get('frames')} != {e['sequence_id']}/{e['object']}/{e['frames']}")
    check("numbering_matches_tier1_tier2_manifests", not mism, mism or "30/30 index, sequence id, object and frame count equal")
    by_idx = {int(e["episode_index"]): e for e in eps}
    named = {f"ep{i:02d}": by_idx[i]["sequence_id"] for i in NAMED}
    check("named_tier1_numbering", all(by_idx[i]["sequence_id"].endswith(s) for i, s in NAMED.items()), named)
    rep["episodes"] = {}
    src_sha = []
    floor_only, support_pending = [], []
    for e in sorted(eps, key=lambda x: int(x["episode_index"])):
        i, seq = int(e["episode_index"]), e["sequence_id"]
        sdir, ddir = SRC / "reconstructed_data" / seq, SOMA / f"sequence_id={seq}"
        fs = files(sdir)
        row = {"sequence_id": seq, "object": e["object"], "frames": int(e["frames"])}
        if fs != sorted(INPUT_FILES):
            check(f"ep{i:02d}_file_set", False, f"source files {fs}")
            continue
        for f in INPUT_FILES:
            s = copy_verified(sdir / f, ddir / f, check_only, log)
            src_sha.append(f"{s}  reconstructed_data/{seq}/{f}")
        P = np.load(sdir / "poses.npy")
        X = np.load(sdir / "object_pose_xyz_wxyz.npy")
        sp = np.load(sdir / "soma_params.npz", allow_pickle=False)
        ej = json.loads((sdir / "episode.json").read_text())
        T = int(e["frames"])
        rot = R.from_matrix(P[:, :3, :3])
        q = rot.as_quat()  # xyzw
        Rx = R.from_quat(X[:, [4, 5, 6, 3]])
        ang = np.degrees((rot.inv() * Rx).magnitude())
        ok_shapes = (P.shape == (T, 4, 4) and X.shape == (T, 7) and sp["poses"].shape[0] == T and sp["transl"].shape[0] == T
                     and int(ej.get("frames", -1)) == T and int(ej.get("episode_index", -1)) == i and ej.get("sequence_id") == seq)
        fin = bool(np.isfinite(P).all() and np.isfinite(X).all())
        orth = float(np.abs(np.einsum("tji,tjk->tik", P[:, :3, :3], P[:, :3, :3]) - np.eye(3)).max())
        row.update({"shapes_ok": ok_shapes, "finite": fin, "max_orthonormality_err": orth,
                    "poses_vs_xyz_wxyz_max_pos_m": float(np.abs(X[:, :3] - P[:, :3, 3]).max()), "poses_vs_xyz_wxyz_max_deg": float(ang.max())})
        check(f"ep{i:02d}_arrays", ok_shapes and fin and orth < 1e-4 and row["poses_vs_xyz_wxyz_max_pos_m"] < 1e-5 and ang.max() < 1e-3,
              f"T={T} shapes {ok_shapes} finite {fin} orth {orth:.1e} pos/xyz_wxyz {row['poses_vs_xyz_wxyz_max_pos_m']:.1e} m {ang.max():.1e} deg")
        # released scene files (non-trajectory): ground plane and mesh identical to the Track-2 release?
        meta = next(json.loads(l) for l in (T2REL / "meta/episodes_metadata.jsonl").read_text().splitlines() if l.strip() and json.loads(l)["episode_index"] == i)
        row["ground_plane_equal_released"] = sha(sdir / "ground_plane.json") == sha(T2REL / meta["ground_plane"])
        row["mesh_equal_released"] = sha(sdir / "reconstructed_mesh/output_aligned.glb") == sha(T2REL / meta["mesh"])
        s1dir = S1 / "reconstructed_data" / seq
        row["soma_params_equal_s1"] = sha(sdir / "soma_params.npz") == sha(s1dir / "soma_params.npz")
        row["poses_equal_s1"] = sha(sdir / "poses.npy") == sha(s1dir / "poses.npy")
        check(f"ep{i:02d}_human_is_s1", row["soma_params_equal_s1"], "soma_params.npz byte-identical to s1 (README: human unchanged)")
        # release frame reference = s1 track (finding), opus frame reference built here
        rf = SRC / "frame_references" / f"episode_{i:06d}.npz"
        row["release_frame_ref_equal_s1_file"] = sha(rf) == sha(S1 / "frame_references" / f"episode_{i:06d}.npz")
        with np.load(rf) as d:
            fr = d["pose_xyzw"][:, 0]
            row["release_frame_ref_vs_opus_max_pos_m"] = float(np.abs(fr[:, :3] - P[d["frame_index"], :3, 3]).max())
        out = FREF / f"episode_{i:06d}.npz"
        xyzw = np.concatenate([P[:, :3, 3], q], -1)[:, None, :].astype(np.float64)
        if not check_only:
            FREF.mkdir(parents=True, exist_ok=True)
            if out.exists():
                with np.load(out) as d:
                    same = np.array_equal(d["pose_xyzw"], xyzw) and int(d["episode_index"]) == i
                if not same:
                    raise SystemExit(f"ABORT: {out} exists with different content")
            else:
                np.savez(out, episode_index=np.int64(i), coordinate_frame=np.asarray("released_track2_world"),
                         frame_index=np.arange(T, dtype=np.int64), pose_xyzw=xyzw)
            row["frame_ref_sha256"] = sha(out)
        # support rule from the released support surface (identical for Tier 1 and Tier 2)
        rel_sup = T2REL / meta["support_surface"]
        geo = has_geometry(rel_sup)
        row["released_support_has_geometry"] = geo
        if geo:
            support_pending.append(i)
            row["support"] = "support_pending (no file written)"
        else:
            floor_only.append(i)
            copy_verified(rel_sup, STAGE / f"{seq}_support.usda", check_only, log)
            row["support"] = "floor-only (released geometry-free file copied; scene.support=false)"
        rep["episodes"][f"{i:02d}"] = row
    rep["floor_only"], rep["support_pending"] = floor_only, support_pending
    check("ground_planes_released", all(r["ground_plane_equal_released"] for r in rep["episodes"].values()), "30/30 ground_plane.json = released Track-2 file")
    check("meshes_released", all(r["mesh_equal_released"] for r in rep["episodes"].values()), "30/30 output_aligned.glb = released Track-2 mesh")
    check("release_frame_refs_are_s1", all(r["release_frame_ref_equal_s1_file"] for r in rep["episodes"].values()),
          "release frame_references/ byte-identical to s1 (not usable for opus-trained policies)")
    check("floor_only_count", len(floor_only) == 16, f"floor-only {floor_only}; support_pending {support_pending}")
    # manifest
    man = {e["sequence_id"]: {"sequence_id": e["sequence_id"], "episode_index": int(e["episode_index"]), "object": e["object"],
                              "task": e.get("task", ""), "frames": int(e["frames"]), "invisible_object_frames": 0,
                              "longest_invisible_run": 0, "status": "converted",
                              "source": "han.zhou tier3_opus_20261021 (Claude Opus agent-corrected object; s1 human)"}
           for e in sorted(eps, key=lambda x: int(x["episode_index"]))}
    if not check_only:
        p = SOMA / "track2_manifest.json"
        txt = json.dumps(man, indent=2) + "\n"
        if p.exists() and p.read_text() != txt:
            raise SystemExit(f"ABORT: {p} exists with different content")
        p.write_text(txt)
        for f in ["README.md", "episodes.json"] + [f"agent_run/{e['sequence_id']}/{n}" for e in eps for n in ("REPORT.md", "TOOLS.md", "solve.py")]:
            if (SRC / f).is_file():
                src_sha.append(f"{copy_verified(SRC / f, DOCS / f, False, log)}  {f}")
        (ROOT / "SOURCE_FILES.sha256").write_text("\n".join(src_sha) + "\n")
        # verify the whole copy once more from scratch
        again = [sha(SOMA / ("sequence_id=" + r["sequence_id"]) / f) == sha(SRC / "reconstructed_data" / r["sequence_id"] / f)
                 for r in rep["episodes"].values() for f in INPUT_FILES]
        check("copy_reverified", all(again) and len(again) == 30 * len(INPUT_FILES), f"{sum(again)}/{len(again)} files sha256 = source")
    BUILD.mkdir(parents=True, exist_ok=True)
    rep["bad"] = bad
    (BUILD / ("build_check.json" if check_only else "build_report.json")).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for k, v in rep["checks"].items():
        if not k.startswith("ep") or not v["ok"]:
            print(f"CHECK {k} {'ok' if v['ok'] else 'FAIL'} {str(v['detail'])[:300]}")
    print(f"episode checks: {sum(v['ok'] for k, v in rep['checks'].items() if k.startswith('ep'))}/{sum(1 for k in rep['checks'] if k.startswith('ep'))} ok")
    print(f"floor-only {floor_only}\nsupport_pending {support_pending}")
    if bad:
        print(f"FAILED: {bad}")
        return 1
    if not check_only:
        (ROOT / "READY").write_text(
            f"READY {time.strftime('%F %T %Z')}: byte copy of {SRC}/reconstructed_data (30 episodes, 7 files each, sha256 verified) "
            f"+ track2_manifest.json from its episodes.json + frame_references_opus (opus poses.npy, released_track2_world) + "
            f"released geometry-free support files for the 16 floor-only episodes {floor_only}; NO support file for the 14 support "
            f"episodes {support_pending} (support_pending). Human = s1 (byte-identical). Report: {BUILD}/build_report.json\n")
        print(f"wrote {ROOT}/READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
