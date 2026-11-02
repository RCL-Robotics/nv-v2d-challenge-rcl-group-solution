#!/usr/bin/env python3
"""t2cons_r6_pack.py: step R7 of fc_runs/submissions/t2_consensus_r6_20261031 (PREREG_R6 6): build and pack the
R6-CLOSEST file (trial t* of the S6 set, every episode's S6 job's phase-F run t*), then validate it offline with the
kit's uploader, after round 6 froze (SELECTION_R6.sha256). NO clean scoring.

Round 5's pack script scripts/t2cons_r5_pack.py (62f88e6f...) with only the round-6 bookkeeping (PREREG_R6 6):
  W1  one run source: round 6 phase F (runs_f/), through episode_source_r6 = round 4's pinned episode_source_r4 logic
      with ROLLOUTS_F6 / TRACKS_R6 (raw input reference: round 1's inputs/TRACKS.sha256), as round 5 re-used it.
  W2  r6_closest = selection_r6.json r6_closest_file, re-derived from the frozen records (S6 composition, R6-CLOSEST
      trial, the run's ok try, its rollout sha256 in ROLLOUTS_F6, the job's export sha256).
  W3  packing: the kit packer tools/pack_submission.py, UNCHANGED, run directly with the placeholder commit URL and the
      packer arguments of rounds 1-5 (round 4's pack_command). The packer's own "locally computed: ..." line (its
      built-in clean mean) is dropped from the stored log (a "withheld" marker) and never parsed (PREREG_R6 7: clean
      numbers only in the separate LOCAL report).
  W4  validate: the uploader's offline validate_submission (track_2_tier2), row count == sample, uniform provenance
      columns (placeholder commit URL), exactly one checkpoint digest per episode == the chosen export, pose-column
      sha256; plus, read-only, the same uploader validation of round 5's R5-CLOSEST packed file and a per-episode
      comparison of checkpoint digests (unchanged episodes must carry the same digest in both files).
Commands: check (dry) | prepare | pack | validate      (order: check; prepare; pack; validate)
Gate (every command): the pinned round-6 scorer's pin chain (phase F); SELECTION_R6.sha256 exists and every entry holds;
selection_r6.json frozen and written by the pinned scorer; kit pins and KIT_MANIFEST. Every command except check also
needs PREREG_R6_PINS_c.md == its sha file, pinning this script.
Writes only under pack_r6_LOCAL_NOT_FOR_UPLOAD/ (new files only). Nothing uploaded, nothing signalled, CPU only.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r6_pack.py <cmd>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r6_20261031"
R5 = ME / "fc_runs/submissions/t2_consensus_r5_20261029"
R5_CLOSEST_CSV = R5 / "pack_r5_LOCAL_NOT_FOR_UPLOAD/r5_closest/packed_LOCAL_NOT_FOR_UPLOAD.csv"  # read only, after the freeze
R5_CLOSEST_PREFIX = "e8a03a4d"  # PREREG_R6 0 / 5.6
OUT = ROOT / "pack_r6_LOCAL_NOT_FOR_UPLOAD"
R6SC_FILE, R6SC_SHA = ME / "scripts/t2cons_r6_consensus.py", "9b65e612fb289cd8e7f7584ffb7c7ccb40e1d2de9235f0221a70a99c11382d31"
S7_FILE, S7_SHA = ME / "scripts/t2cons_s7_pack.py", "fef326d577af9075c439090b4ca4f8f8cfaedc75dd98bd08752598f03d56eebf"
PINS_C, PINS_C_SHA_FILE = ROOT / "PREREG_R6_PINS_c.md", ROOT / "PREREG_R6_PINS_c.sha256"
SEL, SEL_SHA = ROOT / "selection_r6.json", ROOT / "SELECTION_R6.sha256"
FILES = ["r6_closest"]
WITHHELD = "locally computed: [withheld by t2cons_r6_pack.py: no clean value is recorded in step R7]"


# ================================================================================================ pinned code
def _load(name: str, path: Path, want: str):
    src = path.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != want:
        raise SystemExit(f"{path} sha256 {got} != pinned {want}")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 (verified pinned bytes)
    return mod


R6C = _load("t2cons_r6_consensus_pinned", R6SC_FILE, R6SC_SHA)  # round-6 scorer: chain, load_jobs, load_evals
S7 = _load("t2cons_s7_pack_pinned", S7_FILE, S7_SHA)  # round-1 pack: helpers, kit pins, placeholder, raw npz writer
C = R6C.C  # the round-6 scorer's t2cons_consensus instance (extended guard)
MINE = Path(__file__).resolve()
sha, now, stamp, dump, write_new, copy_new, read_json = S7.sha, S7.now, S7.stamp, S7.dump, S7.write_new, S7.copy_new, \
    S7.read_json  # fmt: skip


def first_token(p: Path) -> str:
    return (p.read_text().split() or [""])[0] if p.is_file() else ""


# ================================================================================================ gate
def gate(need_c: bool) -> tuple[list[str], list[str]]:
    fatal, info = R6C.chain("F", "check")  # PREREG_R6 chain, PINS, DRYRUN, lists, round-1 chain, phase-F heartbeat, integrity
    if not SEL_SHA.is_file():
        fatal.append("SELECTION_R6.sha256 does not exist: step R7 may not run (PREREG_R6 6)")
        return fatal, info
    n, listed = 0, False
    for rel, h0 in R6C.hash_lines(SEL_SHA):
        n += 1
        p = R6C.pin_path(rel)
        listed |= p == SEL
        if (sha(p) if p.is_file() else "missing") != h0:
            fatal.append(f"SELECTION_R6.sha256: {rel} changed")
    if not listed:
        fatal.append("SELECTION_R6.sha256 does not list selection_r6.json")
    info.append(f"SELECTION_R6.sha256 ({sha(SEL_SHA)[:16]}...): {n} entries verify; lists selection_r6.json {listed}")
    sel = read_json(SEL)
    if sel.get("synthetic") is not False or sel.get("noisy_only") is not True or sel.get("binding_variant") != "B" or \
            sel.get("script_sha256") != R6SC_SHA or sel.get("phase") != "F":  # fmt: skip
        fatal.append("selection_r6.json: synthetic / noisy_only / variant / script / phase unexpected")
    info.append(f"selection_r6.json written by t2cons_r6_consensus.py {R6SC_SHA[:8]}; changed episodes "
                f"{sel.get('changed_episodes')}; R6-CLOSEST trial {sel.get('r6_closest_trial')}; replace "
                f"{(sel.get('replace_verdict') or {}).get('replace')}")  # fmt: skip
    for rel, want in S7.KIT_PINS.items():
        hh = sha(S7.KIT / rel) if (S7.KIT / rel).is_file() else "missing"
        if hh != want:
            fatal.append(f"kit {rel} sha256 {hh[:16]} != pinned {want[:16]}")
    man = read_json(S7.KIT / "KIT_MANIFEST.json")["files"]
    bad = [p for p, hh in man.items() if not (S7.KIT / p).is_file() or sha(S7.KIT / p) != hh]
    if bad:
        fatal.append(f"KIT_MANIFEST mismatches: {bad[:5]}")
    info.append(f"kit pins ({len(S7.KIT_PINS)}) hold; KIT_MANIFEST: {len(man)} files, {len(bad)} mismatched")
    if need_c:
        if not PINS_C.is_file():
            fatal.append("PREREG_R6_PINS_c.md is missing: this script must be pinned before it prepares / packs")
            return fatal, info
        g, r = sha(PINS_C), first_token(PINS_C_SHA_FILE)
        if g != r:
            fatal.append(f"PREREG_R6_PINS_c.md {g[:16]} != PREREG_R6_PINS_c.sha256 {r[:16]}")
        pins = R6C.pin_lines(PINS_C.read_text())
        mine = sha(MINE)
        if ("scripts/t2cons_r6_pack.py", mine) not in pins:
            fatal.append(f"PREREG_R6_PINS_c.md does not pin this script ({mine[:16]})")
        for rel, h0 in pins:
            p = R6C.pin_path(rel)
            if (sha(p) if p.is_file() else "missing") != h0:
                fatal.append(f"PREREG_R6_PINS_c pin {rel} changed")
        info.append(f"PREREG_R6_PINS_c.md ({g[:16]}) == its sha file; {len(pins)} pins hold; pins this script {mine[:16]}")
    return fatal, info


def require(need_c: bool) -> bool:
    fatal, info = gate(need_c)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


# ================================================================================================ plan
def plan() -> tuple[dict, dict, dict, list[str]]:
    """episode -> (job, run); re-derived from the frozen selection_r6.json records."""
    sel = read_json(SEL)
    jobs = R6C.load_jobs("F")
    evals, _, lp = R6C.load_evals("F", jobs)
    problems = list(lp)
    t = sel.get("r6_closest_trial")
    if t is None:
        return {}, jobs, evals, ["no R6-CLOSEST trial (no complete S6 trial)"]
    s6 = sel["compositions"]["S6"]
    rows6 = {r["trial"]: r for r in sel["trials"]["S6"]}
    if not rows6[t]["complete"]:
        problems.append(f"trial {t} is not complete for S6")
    roll = C.hash_list(R6C.PH["F"]["rollouts"])
    out = {}
    for e in range(30):
        k = f"{e:02d}"
        x = sel["r6_closest_file"][k]
        job = s6[k]
        ev = evals.get(job, {}).get(t)
        if x["job"] != job or int(x["run"]) != t or ev is None or int(x["try"]) != int(ev["try"]) or \
                x["rollout"] != ev["parquet"] or roll.get(ev["parquet"]) != x["rollout_sha256"] or \
                jobs[job]["export_sha256"] != x["export_sha256"] or jobs[job]["input"] != x["input"]:  # fmt: skip
            problems.append(f"ep{k}: r6_closest_file record {x['job']}#{x['run']} != re-derived {job}#{t}")
        out[e] = (job, t)
    return out, jobs, evals, problems


def episode_source_r6(K, jobs, evals, roll, trk, job: str, k: int, e: int) -> tuple[dict, list[str]]:
    """Round 4's pinned episode_source_r4 logic on round-6 records (ROLLOUTS_F6; frame reference = TRACKS_R6, raw = round
    1's inputs/TRACKS.sha256)."""
    j, rec = jobs[job], evals[job][k]
    pq = Path(rec["parquet"])
    problems = []
    h = C.sha256_file(pq)
    if roll.get(str(pq)) != h:
        problems.append(f"{job} run {k}: rollout sha256 {h[:16]} != ROLLOUTS_F6.sha256")
    js = pq.with_suffix(".json")
    jh = C.sha256_file(js) if js.is_file() else "missing"
    rep = C.read_json(js) if js.is_file() else {}
    if rep.get("checkpoint_sha256") != j["export_sha256"]:
        problems.append(f"{job} run {k}: metrics json checkpoint_sha256 != export")
    traj = K.read_trajectory(C.guard(pq))
    prov = K.provenance(traj)
    if traj.episode_index != e or prov.get("checkpoint_sha256") != j["export_sha256"]:
        problems.append(f"{job} run {k}: parquet episode / checkpoint provenance mismatch")
    if prov.get("eval_script_sha256") != S7.EVAL_SCRIPT_SHA or prov.get("eval_code_sha256", S7.EVAL_CODE_SHA) != S7.EVAL_CODE_SHA:
        problems.append(f"{job} run {k}: evaluator digests differ")
    kind = j["input"]
    if kind == "raw":
        fr_src = C.T2 / "data/chunk-000" / f"episode_{e:06d}.parquet"
        want = K.tracks_sha.get(str(fr_src))  # round 1's inputs/TRACKS.sha256
    else:
        fr_src = Path(j["input_reference"]) / f"episode_{e:06d}.npz"
        want = trk.get(str(fr_src))
    fr_h = C.sha256_file(fr_src) if fr_src.is_file() else "missing"
    if want != fr_h:
        problems.append(f"{job}: frame reference source {fr_src} sha256 {fr_h[:16]} != its frozen record")
    return {"episode": e, "job": job, "run": k, "try": int(rec["try"]), "input": kind, "source": "round6 phase F",
            "role": j["role"], "export_sha256": j["export_sha256"], "rollout_src": str(pq), "rollout_sha256": h,
            "json_src": str(js), "json_sha256": jh,
            "frame_ref_kind": "written_from_released_tier2" if kind == "raw" else "byte_copy",
            "frame_ref_src": str(fr_src), "frame_ref_src_sha256": fr_h, "input_reference": j["input_reference"]}, problems  # fmt: skip


def pack_command(d: Path) -> list[str]:
    """The kit packer, run directly, with the packer arguments of rounds 1-5 (round 4's pinned pack_command)."""
    return ["nice", "-n", "10", str(S7.KPY), "-I", "-B", str(S7.KIT / "tools/pack_submission.py"),
            "--track", "track_2_tier2", "--results", str(d / "results"), "--dataset-root", str(S7.DATA),
            "--frame-references", str(d / "frame_refs"), "--code-commit", S7.PLACEHOLDER,
            "--eval-script", str(S7.KIT / "flash_chord/scripts/evaluate_policy.py"),
            "-o", str(d / "packed_LOCAL_NOT_FOR_UPLOAD.csv")]  # fmt: skip


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    lines = [f"t2cons_r6_pack.py check at {now()} (dry; no clean file opened; nothing written except this file)",
             f"script sha256 {sha(MINE)}"]  # fmt: skip
    fatal, info = gate(need_c=False)
    lines += ["", "GATE"] + ["  " + x for x in info]
    if fatal:
        lines += ["FATAL"] + ["  " + x for x in fatal]
        print("\n".join(lines))
        return 1
    K = C.load_kit()
    pl, jobs, evals, problems = plan()
    roll, trk = C.hash_list(R6C.PH["F"]["rollouts"]), C.hash_list(R6C.TRACKS)
    lines += ["", "FILE r6_closest (episode: job [input, role] run/try)"]
    for e in sorted(pl):
        job, k = pl[e]
        s, p2 = episode_source_r6(K, jobs, evals, roll, trk, job, k, e)
        problems += p2
        lines.append(f"  ep{e:02d}: {s['job']} [{s['input']}, {s['role']}] run {s['run']} try {s['try']}  rollout "
                     f"{s['rollout_sha256'][:12]}  export {s['export_sha256'][:12]}  frame ref {s['frame_ref_kind']} {s['frame_ref_src']}")
    lines += ["", "PROBLEMS: " + ("; ".join(sorted(set(problems))) or "none")]
    text = "\n".join(lines) + "\n"
    write_new(OUT / "dryrun" / f"check_{stamp()}.txt", text)
    print(text)
    return 1 if problems else 0


# ================================================================================================ prepare
def cmd_prepare(a) -> int:
    if not require(need_c=True):
        return 1
    K = C.load_kit()
    pl, jobs, evals, problems = plan()
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    roll, trk = C.hash_list(R6C.PH["F"]["rollouts"]), C.hash_list(R6C.TRACKS)
    sel = read_json(SEL)
    for f in FILES:
        d = OUT / f
        if d.exists():
            print(f"{f}: exists, skipped")
            continue
        tmpd = OUT / f".{f}.preparing"
        if tmpd.exists():
            raise SystemExit(f"stale {tmpd}")
        (tmpd / "results").mkdir(parents=True)
        (tmpd / "frame_refs").mkdir()
        rows, probs = [], []
        for e in range(30):
            job, k = pl[e]
            s0, p2 = episode_source_r6(K, jobs, evals, roll, trk, job, k, e)
            s = dict(s0)
            probs += p2
            s["results_parquet_sha256"] = copy_new(Path(s["rollout_src"]), tmpd / "results" / f"episode_{e:06d}.parquet",
                                                   s["rollout_sha256"])  # fmt: skip
            s["results_json_sha256"] = copy_new(Path(s["json_src"]), tmpd / "results" / f"episode_{e:06d}.json", s["json_sha256"])
            npz = tmpd / "frame_refs" / f"episode_{e:06d}.npz"
            if s["input"] == "raw":
                s.update(S7.write_raw_npz(K, e, npz))
            else:
                copy_new(Path(s["frame_ref_src"]), npz, s["frame_ref_src_sha256"])
            s["frame_ref_sha256"] = sha(npz)
            frames, names = C.grid(K, e)
            mine = K.input_frame_reference(tmpd / "frame_refs", e, frames, 1, names)
            own = K.input_frame_reference(C.guard(Path(s["input_reference"])), e, frames, 1, names)
            s["frame_ref_equals_own_input_on_grid"] = bool(np.array_equal(mine, own))
            if s["input"] == "raw":
                s["frame_ref_equals_packer_default"] = bool(np.array_equal(mine, K.input_frame_reference(C.T2, e, frames, 1, names)))
            if not s["frame_ref_equals_own_input_on_grid"] or s.get("frame_ref_equals_packer_default") is False:
                probs.append(f"{f} ep{e:02d}: frame reference differs from the job's own input on the grid")
            s["grid_frames"] = int(len(frames))
            rows.append(s)
        if probs:
            print(f"{f}: PROBLEMS\n  " + "\n  ".join(probs))
            return 1
        meta = {"file": f, "kind": "R6-CLOSEST", "created_at": now(), "script_sha256": sha(MINE),
                "selection_r6_json_sha256": sha(SEL), "selection_r6_sha256_file_sha256": sha(SEL_SHA),
                "r6_closest_trial": sel["r6_closest_trial"], "replace_verdict": sel["replace_verdict"],
                "upload_eligibility": sel["upload_eligibility"], "episodes": rows,
                "note": "LOCAL_NOT_FOR_UPLOAD: placeholder commit URL; never uploaded"}  # fmt: skip
        write_new(tmpd / "file.json", dump(meta))
        os.rename(tmpd, d)
        print(f"{f}: prepared ({len(rows)} episodes; trial {sel['r6_closest_trial']}; inputs "
              f"{sorted({r['input'] for r in rows})}; finalist rows {[r['episode'] for r in rows if r['role'] == 'finalist']})")
    return 0


# ================================================================================================ pack
def cmd_pack(a) -> int:
    if not require(need_c=True):
        return 1
    before = S7.kit_hashes()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/root"),
           "LANG": "C.UTF-8", "TZ": "Asia/Shanghai", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}  # fmt: skip
    results = {}
    for f in FILES:
        d = OUT / f
        if not (d / "file.json").is_file():
            print(f"{f}: not prepared")
            return 1
        if (d / "packed_LOCAL_NOT_FOR_UPLOAD.csv").exists() or (d / "pack_LOCAL_NOT_FOR_UPLOAD.log").exists():
            print(f"{f}: already packed, skipped")
            continue
        cmd = pack_command(d)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(d), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        kept, n_withheld = [], 0
        for ln in proc.stdout.splitlines():
            if ln.startswith("locally computed:"):
                kept.append(WITHHELD)
                n_withheld += 1
            else:
                kept.append(ln)
        write_new(d / "pack_LOCAL_NOT_FOR_UPLOAD.log", "+ " + " ".join(cmd) + "\n" + "\n".join(kept) + "\n")
        csvp = d / "packed_LOCAL_NOT_FOR_UPLOAD.csv"
        rec = {"file": f, "exit": proc.returncode, "seconds": round(time.time() - t0, 1), "finished_at": now(),
               "command": cmd, "csv_exists": csvp.is_file(), "csv_sha256": sha(csvp) if csvp.is_file() else None,
               "clean_lines_withheld": n_withheld,
               "input_reference_line": next((ln for ln in kept if ln.startswith("input reference check")), None),
               "frame_reference_line": next((ln for ln in kept if ln.startswith("frame conversion reference")), None),
               "note": "kit packer run directly (no clean-score wrapper); its clean 'locally computed' line was not "
                       "recorded (step R7: no clean scoring)"}  # fmt: skip
        rec["ok"] = bool(proc.returncode == 0 and csvp.is_file())
        write_new(d / "pack_LOCAL_NOT_FOR_UPLOAD.json", dump(rec))
        results[f] = rec
        print(f"[{now()}] {f}: exit {proc.returncode} in {rec['seconds']} s; ok {rec['ok']}; csv sha256 {rec['csv_sha256']}", flush=True)
    after = S7.kit_hashes()
    diff = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    run = {"at": now(), "files": list(results), "ok": {f: r["ok"] for f, r in results.items()}, "kit_files": len(after),
           "kit_unchanged": not diff, "kit_diff": diff[:50]}  # fmt: skip
    write_new(OUT / f"pack_run_{stamp()}.json", dump(run))
    print(f"kit unchanged: {not diff} ({len(after)} files)" + (f"; DIFF {diff[:10]}" if diff else ""))
    return 0 if all(r["ok"] for r in results.values()) and not diff else 1


# ================================================================================================ validate
def validate_csv(up, config, csvp: Path, n_sample: int, npose: int, rs, op, man, pd) -> tuple[dict, object]:
    res = {"csv": str(csvp), "csv_sha256": sha(csvp), "problems": []}
    try:
        frame = up.validate_submission(csvp, "track_2_tier2", config)
        res["uploader_validate_submission"] = "pass"
    except SystemExit as err:
        res["uploader_validate_submission"] = f"FAIL: {err}"
        res["problems"].append("uploader validate_submission failed")
        frame = pd.read_csv(csvp)
    res["rows"], res["sample_rows"] = int(len(frame)), n_sample
    if res["rows"] != n_sample:
        res["problems"].append(f"{res['rows']} rows != sample {n_sample}")
    uniform = {c: sorted(frame[c].astype(str).str.strip().unique()) for c in rs.UNIFORM_PROVENANCE_COLUMNS}
    res["uniform_columns"] = uniform
    for c, v in uniform.items():
        if len(v) != 1:
            res["problems"].append(f"{c} not uniform")
    want = {"eval_script_sha256": S7.EVAL_SCRIPT_SHA, "eval_code_sha256": S7.EVAL_CODE_SHA,
            "packer_script_sha256": man["tools/pack_submission.py"], "uploader_script_sha256": man["submit/v2d_submit.py"],
            "code_commit_url": S7.PLACEHOLDER}  # fmt: skip
    for c, v in want.items():
        if uniform.get(c, [None])[0] != v:
            res["problems"].append(f"{c} {uniform.get(c)} != {v}")
    eps = frame["row_id"].astype(str).str.split("/").str[1]
    per_ep = frame.groupby(eps)["checkpoint_sha256"].agg(lambda s: sorted(s.astype(str).str.strip().unique()))
    res["per_episode_checkpoints"] = {ep: v for ep, v in per_ep.items()}
    res["pose_columns_sha256"], header = S7.pose_columns_sha(csvp, npose)
    if header[: 1 + npose] != [rs.ROW_ID_COLUMN, *op._POSE_COLUMNS]:
        res["problems"].append(f"unexpected header {header[: 1 + npose]}")
    res["pose_columns"] = header[: 1 + npose]
    return res, per_ep


def cmd_validate(a) -> int:
    if not require(need_c=True):
        return 1
    C.load_kit()  # kit files == round-1 inputs/KIT.sha256; puts the kit on sys.path
    up = S7.load_uploader()
    from v2dlb import object_pose as op
    from v2dlb import report_schema as rs

    config = read_json(S7.KIT / "config/tracks.json")
    man = read_json(S7.KIT / "KIT_MANIFEST.json")["files"]
    import pandas as pd

    n_sample = len(pd.read_csv(S7.KIT / "data" / "track_2_sample_submission.csv", usecols=[rs.ROW_ID_COLUMN], dtype=str))
    npose = len(op._POSE_COLUMNS)
    sel = read_json(SEL)
    bad = 0
    for f in FILES:
        d = OUT / f
        csvp = d / "packed_LOCAL_NOT_FOR_UPLOAD.csv"
        outp = d / "validate_LOCAL_NOT_FOR_UPLOAD.json"
        if outp.exists():
            print(f"{f}: already validated, skipped")
            continue
        if not (d / "pack_LOCAL_NOT_FOR_UPLOAD.json").is_file() or not read_json(d / "pack_LOCAL_NOT_FOR_UPLOAD.json").get("ok"):
            print(f"{f}: not packed ok")
            return 1
        meta = read_json(d / "file.json")
        res, per_ep = validate_csv(up, config, csvp, n_sample, npose, rs, op, man, pd)
        res.update(file=f, at=now())
        chosen = {f"e{r['episode']:06d}": r["export_sha256"] for r in meta["episodes"]}
        if sorted(per_ep.index) != sorted(chosen):
            res["problems"].append("episode set differs from the file's plan")
        for ep, vals in per_ep.items():
            if len(vals) != 1 or vals[0] != chosen.get(ep):
                res["problems"].append(f"{ep}: checkpoint digest {vals} != chosen export {str(chosen.get(ep))[:16]}")
        res["checkpoint_one_per_episode_and_equal_chosen_export"] = not any("checkpoint digest" in p for p in res["problems"])
        res["distinct_checkpoints"] = int(len({v[0] for v in per_ep}))
        # read only, after the freeze: round 5's R5-CLOSEST packed file (placeholder) through the same uploader check
        r5 = {"file": str(R5_CLOSEST_CSV)}
        if R5_CLOSEST_CSV.is_file():
            r5res, r5ep = validate_csv(up, config, R5_CLOSEST_CSV, n_sample, npose, rs, op, man, pd)
            r5.update(r5res)
            r5["sha256_prefix_matches_prereg"] = r5res["csv_sha256"].startswith(R5_CLOSEST_PREFIX)
            s6, s5 = sel["compositions"]["S6"], sel["compositions"]["S5"]
            same, diff = [], []
            for e in range(30):
                ep = f"e{e:06d}"
                (same if r5ep.get(ep) == per_ep.get(ep) else diff).append(f"{e:02d}")
            r5["episodes_same_checkpoint_as_r6_closest"] = same
            r5["episodes_different_checkpoint"] = diff
            r5["different_equals_changed_episodes"] = sorted(diff) == sorted(f"{e:02d}" for e in range(30)
                                                                               if s6[f"{e:02d}"] != s5[f"{e:02d}"])  # fmt: skip
            if not r5["different_equals_changed_episodes"]:
                res["problems"].append("checkpoint digests differ from R5-CLOSEST on an unchanged episode (or agree on a changed one)")
        else:
            r5["problem"] = "R5-CLOSEST packed file not found"
        res["r5_closest_readonly_check"] = r5
        res["ok"] = not res["problems"]
        bad += 0 if res["ok"] else 1
        write_new(outp, dump(res))
        print(f"{f}: uploader validate_submission {res['uploader_validate_submission']}; {res['rows']:,} rows (sample "
              f"{n_sample:,}); {res['distinct_checkpoints']} distinct checkpoints; csv sha256 {res['csv_sha256']}; pose "
              f"columns sha256 {res['pose_columns_sha256']}; problems: {res['problems'] or 'none'}")  # fmt: skip
        print(f"R5-CLOSEST (read only): {r5.get('csv_sha256')} prefix ok {r5.get('sha256_prefix_matches_prereg')}; uploader "
              f"{r5.get('uploader_validate_submission')}; differing-checkpoint episodes {r5.get('episodes_different_checkpoint')}")
    return 1 if bad else 0


# ================================================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "prepare", "pack", "validate"))
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    return {"check": cmd_check, "prepare": cmd_prepare, "pack": cmd_pack, "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
