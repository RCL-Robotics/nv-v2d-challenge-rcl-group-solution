#!/usr/bin/env python3
"""t2cons_r7_pack.py: step R7 of fc_runs/submissions/t2_consensus_r7_20261102 (PREREG_R7 6): build and pack the
R7-CLOSEST file (trial t* of the S7 set, every episode's S7 job's phase-F run t*), validate it offline with the kit's
uploader, and re-pack it with the real submission commit URL, after round 7 froze (SELECTION_R7.sha256). NO clean scoring.

Round 6's pack script scripts/t2cons_r6_pack.py (da02d1b4...) with only the round-7 bookkeeping (PREREG_R7 6):
  W1  one run source: round 7 phase F (runs_f/), through episode_source_r7 = round 6's episode_source_r6 logic with
      ROLLOUTS_F7 / TRACKS_R7 (raw input reference: round 1's inputs/TRACKS.sha256).
  W2  r7_closest = selection_r7.json r7_closest_file, re-derived from the frozen records (S7 composition, R7-CLOSEST
      trial, the run's ok try, its rollout sha256 in ROLLOUTS_F7, the job's export sha256).
  W3  packing: the kit packer tools/pack_submission.py, UNCHANGED, run directly with the packer arguments of rounds 1-6
      (round 4's pack_command): first with the placeholder commit URL (pack), then with the submission commit URL
      (pack-commit). The packer's own "locally computed: ..." line (its built-in clean mean) is dropped from every stored
      log (a "withheld" marker) and never parsed or printed (PREREG_R7 7).
  W4  validate: the uploader's offline validate_submission (track_2_tier2), row count == sample, uniform provenance
      columns, exactly one checkpoint digest per episode == the chosen export, pose-column sha256; plus, read only,
      the same uploader validation of round 6's R6-CLOSEST packed file and a per-episode comparison of checkpoint
      digests (they must differ exactly on the episodes whose S7 job differs from round 6's S6 job).
  W5  pack-commit --commit-url URL --branch-head-file F: URL = https://github.com/<repo>/commit/<40 hex>; the commit must
      exist in code/v2d_t2ready and F (a git ls-remote listing written after the push) must show
      refs/heads/dev.haoran.t2r7_20261102 at that commit. Output submission_r7/T2_R7_closest_trial<t>_<sha7>.csv: the
      uploader validation passes with code_commit_url == URL and its pose columns are byte-identical to the
      placeholder pack (PREREG_R7 6: a check, not a second file for upload).
Commands: check (dry) | prepare | pack | validate | pack-commit      (in this order)
Gate (every command): the pinned round-7 scorer's pin chain (phase F); SELECTION_R7.sha256 exists and every entry holds;
selection_r7.json frozen and written by the pinned scorer; kit pins and KIT_MANIFEST. Every command except check also
needs PREREG_R7_PINS_c.md == its sha file, pinning this script.
Writes only under pack_r7_LOCAL_NOT_FOR_UPLOAD/ and (pack-commit) submission_r7/ (new files only). Nothing uploaded,
nothing signalled, CPU only.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r7_pack.py <cmd>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r7_20261102"
R6 = ME / "fc_runs/submissions/t2_consensus_r6_20261031"
R6_CLOSEST_CSV = R6 / "pack_r6_LOCAL_NOT_FOR_UPLOAD/r6_closest/packed_LOCAL_NOT_FOR_UPLOAD.csv"  # read only, after the freeze
OUT = ROOT / "pack_r7_LOCAL_NOT_FOR_UPLOAD"
SUB = ROOT / "submission_r7"
R7SC_FILE, R7SC_SHA = ME / "scripts/t2cons_r7_consensus.py", "e7545dfa7fdff0681123d77e2dde3f65ed6a22ead8d4d12057d9b61af9e2708a"
S7_FILE, S7_SHA = ME / "scripts/t2cons_s7_pack.py", "fef326d577af9075c439090b4ca4f8f8cfaedc75dd98bd08752598f03d56eebf"
PINS_C, PINS_C_SHA_FILE = ROOT / "PREREG_R7_PINS_c.md", ROOT / "PREREG_R7_PINS_c.sha256"
SEL, SEL_SHA = ROOT / "selection_r7.json", ROOT / "SELECTION_R7.sha256"
FILES = ["r7_closest"]
WITHHELD = "locally computed: [withheld by t2cons_r7_pack.py: no clean value is recorded in step R7]"
WORKTREE = ME / "code/v2d_t2ready"
BRANCH = "dev.haoran.t2r7_20261102"
URL_RE = re.compile(r"^https://github\.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/([0-9a-f]{40})$")


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


R7C = _load("t2cons_r7_consensus_pinned", R7SC_FILE, R7SC_SHA)  # round-7 scorer: chain, load_jobs, load_evals
S7 = _load("t2cons_s7_pack_pinned", S7_FILE, S7_SHA)  # round-1 pack: helpers, kit pins, placeholder, raw npz writer
C = R7C.C  # the round-7 scorer's t2cons_consensus instance (extended guard)
MINE = Path(__file__).resolve()
sha, now, stamp, dump, write_new, copy_new, read_json = S7.sha, S7.now, S7.stamp, S7.dump, S7.write_new, S7.copy_new, \
    S7.read_json  # fmt: skip


def first_token(p: Path) -> str:
    return (p.read_text().split() or [""])[0] if p.is_file() else ""


# ================================================================================================ gate
def gate(need_c: bool) -> tuple[list[str], list[str]]:
    fatal, info = R7C.chain("F", "check")  # PREREG_R7 chain, PINS, DRYRUN, LOCK, lists, round-1 chain, phase-F heartbeat
    if not SEL_SHA.is_file():
        fatal.append("SELECTION_R7.sha256 does not exist: step R7 may not run (PREREG_R7 6)")
        return fatal, info
    n, listed = 0, False
    for rel, h0 in R7C.hash_lines(SEL_SHA):
        n += 1
        p = R7C.pin_path(rel)
        listed |= p == SEL
        if (sha(p) if p.is_file() else "missing") != h0:
            fatal.append(f"SELECTION_R7.sha256: {rel} changed")
    if not listed:
        fatal.append("SELECTION_R7.sha256 does not list selection_r7.json")
    info.append(f"SELECTION_R7.sha256 ({sha(SEL_SHA)[:16]}...): {n} entries verify; lists selection_r7.json {listed}")
    sel = read_json(SEL)
    if sel.get("synthetic") is not False or sel.get("noisy_only") is not True or sel.get("binding_variant") != "B" or \
            sel.get("script_sha256") != R7SC_SHA or sel.get("phase") != "F":  # fmt: skip
        fatal.append("selection_r7.json: synthetic / noisy_only / variant / script / phase unexpected")
    info.append(f"selection_r7.json written by t2cons_r7_consensus.py {R7SC_SHA[:8]}; replaced {sel.get('replaced_episodes')}; "
                f"R7-CLOSEST trial {sel.get('r7_closest_trial')}")  # fmt: skip
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
            fatal.append("PREREG_R7_PINS_c.md is missing: this script must be pinned before it prepares / packs")
            return fatal, info
        g, r = sha(PINS_C), first_token(PINS_C_SHA_FILE)
        if g != r:
            fatal.append(f"PREREG_R7_PINS_c.md {g[:16]} != PREREG_R7_PINS_c.sha256 {r[:16]}")
        pins = R7C.pin_lines(PINS_C.read_text())
        mine = sha(MINE)
        if ("scripts/t2cons_r7_pack.py", mine) not in pins:
            fatal.append(f"PREREG_R7_PINS_c.md does not pin this script ({mine[:16]})")
        for rel, h0 in pins:
            p = R7C.pin_path(rel)
            if (sha(p) if p.is_file() else "missing") != h0:
                fatal.append(f"PREREG_R7_PINS_c pin {rel} changed")
        info.append(f"PREREG_R7_PINS_c.md ({g[:16]}) == its sha file; {len(pins)} pins hold; pins this script {mine[:16]}")
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
    """episode -> (job, run); re-derived from the frozen selection_r7.json records."""
    sel = read_json(SEL)
    jobs = R7C.load_jobs("F")
    evals, _, lp = R7C.load_evals("F", jobs)
    problems = list(lp)
    t = sel.get("r7_closest_trial")
    if t is None:
        return {}, jobs, evals, ["no R7-CLOSEST trial (no complete S7 trial)"]
    s7 = sel["compositions"]["S7"]
    rows7 = {r["trial"]: r for r in sel["trials"]["S7"]}
    if not rows7[t]["complete"]:
        problems.append(f"trial {t} is not complete for S7")
    roll = C.hash_list(R7C.PH["F"]["rollouts"])
    out = {}
    for e in range(30):
        k = f"{e:02d}"
        x = sel["r7_closest_file"][k]
        job = s7[k]
        ev = evals.get(job, {}).get(t)
        if x["job"] != job or int(x["run"]) != t or ev is None or int(x["try"]) != int(ev["try"]) or \
                x["rollout"] != ev["parquet"] or roll.get(ev["parquet"]) != x["rollout_sha256"] or \
                jobs[job]["export_sha256"] != x["export_sha256"] or jobs[job]["input"] != x["input"]:  # fmt: skip
            problems.append(f"ep{k}: r7_closest_file record {x['job']}#{x['run']} != re-derived {job}#{t}")
        out[e] = (job, t)
    return out, jobs, evals, problems


def episode_source_r7(K, jobs, evals, roll, trk, job: str, k: int, e: int) -> tuple[dict, list[str]]:
    """Round 6's episode_source_r6 logic on round-7 records (ROLLOUTS_F7; frame reference = TRACKS_R7, raw = round 1's
    inputs/TRACKS.sha256)."""
    j, rec = jobs[job], evals[job][k]
    pq = Path(rec["parquet"])
    problems = []
    h = C.sha256_file(pq)
    if roll.get(str(pq)) != h:
        problems.append(f"{job} run {k}: rollout sha256 {h[:16]} != ROLLOUTS_F7.sha256")
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
    return {"episode": e, "job": job, "run": k, "try": int(rec["try"]), "input": kind, "source": "round7 phase F",
            "role": j["role"], "owner": j["owner"], "machine": j["machine"], "export_sha256": j["export_sha256"],
            "rollout_src": str(pq), "rollout_sha256": h, "json_src": str(js), "json_sha256": jh,
            "frame_ref_kind": "written_from_released_tier2" if kind == "raw" else "byte_copy",
            "frame_ref_src": str(fr_src), "frame_ref_src_sha256": fr_h, "input_reference": j["input_reference"]}, problems  # fmt: skip


def pack_command(d: Path, url: str, out_csv: Path) -> list[str]:
    """The kit packer, run directly, with the packer arguments of rounds 1-6 (round 4's pinned pack_command)."""
    return ["nice", "-n", "10", str(S7.KPY), "-I", "-B", str(S7.KIT / "tools/pack_submission.py"),
            "--track", "track_2_tier2", "--results", str(d / "results"), "--dataset-root", str(S7.DATA),
            "--frame-references", str(d / "frame_refs"), "--code-commit", url,
            "--eval-script", str(S7.KIT / "flash_chord/scripts/evaluate_policy.py"),
            "-o", str(out_csv)]  # fmt: skip


PACK_ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/root"), "LANG": "C.UTF-8",
            "TZ": "Asia/Shanghai", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}  # fmt: skip


def run_packer(d: Path, url: str, out_csv: Path, log: Path, tag: str) -> dict:
    cmd = pack_command(d, url, out_csv)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(d), env=PACK_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    kept, n_withheld = [], 0
    for ln in proc.stdout.splitlines():
        if ln.startswith("locally computed:"):
            kept.append(WITHHELD)
            n_withheld += 1
        else:
            kept.append(ln)
    write_new(log, "+ " + " ".join(cmd) + "\n" + "\n".join(kept) + "\n")
    rec = {"file": tag, "exit": proc.returncode, "seconds": round(time.time() - t0, 1), "finished_at": now(),
           "command": cmd, "csv_exists": out_csv.is_file(), "csv_sha256": sha(out_csv) if out_csv.is_file() else None,
           "clean_lines_withheld": n_withheld,
           "input_reference_line": next((ln for ln in kept if ln.startswith("input reference check")), None),
           "frame_reference_line": next((ln for ln in kept if ln.startswith("frame conversion reference")), None),
           "note": "kit packer run directly (no clean-score wrapper); its clean 'locally computed' line was not "
                   "recorded (step R7: no clean scoring)"}  # fmt: skip
    rec["ok"] = bool(proc.returncode == 0 and out_csv.is_file())
    return rec


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    lines = [f"t2cons_r7_pack.py check at {now()} (dry; no clean file opened; nothing written except this file)",
             f"script sha256 {sha(MINE)}"]  # fmt: skip
    fatal, info = gate(need_c=False)
    lines += ["", "GATE"] + ["  " + x for x in info]
    if fatal:
        lines += ["FATAL"] + ["  " + x for x in fatal]
        print("\n".join(lines))
        return 1
    K = C.load_kit()
    pl, jobs, evals, problems = plan()
    roll, trk = C.hash_list(R7C.PH["F"]["rollouts"]), C.hash_list(R7C.TRACKS)
    lines += ["", "FILE r7_closest (episode: job [input, role] run/try)"]
    for e in sorted(pl):
        job, k = pl[e]
        s, p2 = episode_source_r7(K, jobs, evals, roll, trk, job, k, e)
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
    roll, trk = C.hash_list(R7C.PH["F"]["rollouts"]), C.hash_list(R7C.TRACKS)
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
            s0, p2 = episode_source_r7(K, jobs, evals, roll, trk, job, k, e)
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
        meta = {"file": f, "kind": "R7-CLOSEST", "created_at": now(), "script_sha256": sha(MINE),
                "selection_r7_json_sha256": sha(SEL), "selection_r7_sha256_file_sha256": sha(SEL_SHA),
                "r7_closest_trial": sel["r7_closest_trial"], "upload_eligibility": sel["upload_eligibility"],
                "episodes": rows, "note": "LOCAL_NOT_FOR_UPLOAD: placeholder commit URL; never uploaded"}  # fmt: skip
        write_new(tmpd / "file.json", dump(meta))
        os.rename(tmpd, d)
        print(f"{f}: prepared ({len(rows)} episodes; trial {sel['r7_closest_trial']}; inputs "
              f"{sorted({r['input'] for r in rows})}; finalist rows {[r['episode'] for r in rows if r['role'] == 'finalist']})")
    return 0


# ================================================================================================ pack (placeholder)
def cmd_pack(a) -> int:
    if not require(need_c=True):
        return 1
    before = S7.kit_hashes()
    results = {}
    for f in FILES:
        d = OUT / f
        if not (d / "file.json").is_file():
            print(f"{f}: not prepared")
            return 1
        if (d / "packed_LOCAL_NOT_FOR_UPLOAD.csv").exists() or (d / "pack_LOCAL_NOT_FOR_UPLOAD.log").exists():
            print(f"{f}: already packed, skipped")
            continue
        rec = run_packer(d, S7.PLACEHOLDER, d / "packed_LOCAL_NOT_FOR_UPLOAD.csv", d / "pack_LOCAL_NOT_FOR_UPLOAD.log", f)
        write_new(d / "pack_LOCAL_NOT_FOR_UPLOAD.json", dump(rec))
        results[f] = rec
        print(f"[{now()}] {f}: exit {rec['exit']} in {rec['seconds']} s; ok {rec['ok']}; csv sha256 {rec['csv_sha256']}", flush=True)
    after = S7.kit_hashes()
    diff = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    run = {"at": now(), "files": list(results), "ok": {f: r["ok"] for f, r in results.items()}, "kit_files": len(after),
           "kit_unchanged": not diff, "kit_diff": diff[:50]}  # fmt: skip
    write_new(OUT / f"pack_run_{stamp()}.json", dump(run))
    print(f"kit unchanged: {not diff} ({len(after)} files)" + (f"; DIFF {diff[:10]}" if diff else ""))
    return 0 if all(r["ok"] for r in results.values()) and not diff else 1


# ================================================================================================ validate
def validate_csv(up, config, csvp: Path, n_sample: int, npose: int, rs, op, man, pd, url: str) -> tuple[dict, object]:
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
            "code_commit_url": url}  # fmt: skip
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


def uploader_ctx():
    C.load_kit()  # kit files == round-1 inputs/KIT.sha256; puts the kit on sys.path
    up = S7.load_uploader()
    from v2dlb import object_pose as op
    from v2dlb import report_schema as rs

    import pandas as pd

    config = read_json(S7.KIT / "config/tracks.json")
    man = read_json(S7.KIT / "KIT_MANIFEST.json")["files"]
    n_sample = len(pd.read_csv(S7.KIT / "data" / "track_2_sample_submission.csv", usecols=[rs.ROW_ID_COLUMN], dtype=str))
    return up, op, rs, pd, config, man, n_sample, len(op._POSE_COLUMNS)


def digest_checks(res: dict, per_ep, meta: dict) -> None:
    chosen = {f"e{r['episode']:06d}": r["export_sha256"] for r in meta["episodes"]}
    if sorted(per_ep.index) != sorted(chosen):
        res["problems"].append("episode set differs from the file's plan")
    for ep, vals in per_ep.items():
        if len(vals) != 1 or vals[0] != chosen.get(ep):
            res["problems"].append(f"{ep}: checkpoint digest {vals} != chosen export {str(chosen.get(ep))[:16]}")
    res["checkpoint_one_per_episode_and_equal_chosen_export"] = not any("checkpoint digest" in p for p in res["problems"])
    res["distinct_checkpoints"] = int(len({v[0] for v in per_ep}))


def cmd_validate(a) -> int:
    if not require(need_c=True):
        return 1
    up, op, rs, pd, config, man, n_sample, npose = uploader_ctx()
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
        res, per_ep = validate_csv(up, config, csvp, n_sample, npose, rs, op, man, pd, S7.PLACEHOLDER)
        res.update(file=f, at=now())
        digest_checks(res, per_ep, meta)
        # read only, after the freeze: round 6's R6-CLOSEST packed file (placeholder) through the same uploader check
        r6 = {"file": str(R6_CLOSEST_CSV)}
        if R6_CLOSEST_CSV.is_file():
            r6res, r6ep = validate_csv(up, config, R6_CLOSEST_CSV, n_sample, npose, rs, op, man, pd, S7.PLACEHOLDER)
            r6.update(r6res)
            s7, s6 = sel["compositions"]["S7"], sel["compositions"]["S6_round6"]
            same, diff = [], []
            for e in range(30):
                ep = f"e{e:06d}"
                (same if r6ep.get(ep) == per_ep.get(ep) else diff).append(f"{e:02d}")
            r6["episodes_same_checkpoint_as_r7_closest"] = same
            r6["episodes_different_checkpoint"] = diff
            r6["different_equals_changed_vs_r6"] = sorted(diff) == sorted(f"{e:02d}" for e in range(30)
                                                                          if s7[f"{e:02d}"] != s6[f"{e:02d}"])  # fmt: skip
            if not r6["different_equals_changed_vs_r6"]:
                res["problems"].append("checkpoint digests differ from R6-CLOSEST on an unchanged episode (or agree on a changed one)")
        else:
            r6["problem"] = "R6-CLOSEST packed file not found"
        res["r6_closest_readonly_check"] = r6
        res["ok"] = not res["problems"]
        bad += 0 if res["ok"] else 1
        write_new(outp, dump(res))
        print(f"{f}: uploader validate_submission {res['uploader_validate_submission']}; {res['rows']:,} rows (sample "
              f"{n_sample:,}); {res['distinct_checkpoints']} distinct checkpoints; csv sha256 {res['csv_sha256']}; pose "
              f"columns sha256 {res['pose_columns_sha256']}; problems: {res['problems'] or 'none'}")  # fmt: skip
        print(f"R6-CLOSEST (read only): {r6.get('csv_sha256')}; uploader {r6.get('uploader_validate_submission')}; "
              f"differing-checkpoint episodes {r6.get('episodes_different_checkpoint')}")
    return 1 if bad else 0


# ================================================================================================ pack-commit (W5)
def cmd_pack_commit(a) -> int:
    if not require(need_c=True):
        return 1
    m = URL_RE.match(a.commit_url or "")
    if not m:
        print(f"FATAL: --commit-url {a.commit_url!r} is not https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/<40 hex>")
        return 1
    commit = m.group(1)
    chk = subprocess.run(["git", "-C", str(WORKTREE), "cat-file", "-t", commit], capture_output=True, text=True,
                         env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})  # fmt: skip
    if chk.stdout.strip() != "commit":
        print(f"FATAL: {commit} is not a commit in {WORKTREE}")
        return 1
    heads = Path(a.branch_head_file).read_text().splitlines() if a.branch_head_file else []
    if f"{commit}\trefs/heads/{BRANCH}" not in heads:
        print(f"FATAL: {a.branch_head_file} does not show refs/heads/{BRANCH} at {commit}")
        return 1
    d = OUT / "r7_closest"
    vp = d / "validate_LOCAL_NOT_FOR_UPLOAD.json"
    if not vp.is_file() or not read_json(vp).get("ok"):
        print("FATAL: the placeholder pack is not validated ok (run validate first)")
        return 1
    meta = read_json(d / "file.json")
    t = meta["r7_closest_trial"]
    name = f"T2_R7_closest_trial{int(t):02d}_{commit[:7]}.csv"
    out_csv = SUB / name
    if out_csv.exists():
        print(f"{out_csv} exists, skipped")
        return 0
    SUB.mkdir(parents=True, exist_ok=True)
    before = S7.kit_hashes()
    rec = run_packer(d, a.commit_url, out_csv, SUB / f"{name[:-4]}_pack_LOCAL_NOT_FOR_UPLOAD.log", "r7_closest_commit")
    after = S7.kit_hashes()
    diff = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    up, op, rs, pd, config, man, n_sample, npose = uploader_ctx()
    res = {"pack": rec, "kit_unchanged": not diff, "kit_diff": diff[:50], "commit": commit, "commit_url": a.commit_url,
           "branch": BRANCH, "branch_head_file": str(a.branch_head_file), "branch_head_file_sha256": sha(a.branch_head_file)}  # fmt: skip
    if rec["ok"]:
        v, per_ep = validate_csv(up, config, out_csv, n_sample, npose, rs, op, man, pd, a.commit_url)
        digest_checks(v, per_ep, meta)
        ph = read_json(vp)
        v["placeholder_pose_columns_sha256"] = ph["pose_columns_sha256"]
        v["pose_columns_identical_to_placeholder_pack"] = v["pose_columns_sha256"] == ph["pose_columns_sha256"]
        if not v["pose_columns_identical_to_placeholder_pack"]:
            v["problems"].append("pose columns differ from the placeholder pack")
        res["validate"] = v
        res["ok"] = not v["problems"] and not diff
    else:
        res["ok"] = False
    res["at"] = now()
    write_new(SUB / f"{name[:-4]}_report.json", dump(res))
    v = res.get("validate") or {}
    print(f"{name}: pack exit {rec['exit']}; uploader {v.get('uploader_validate_submission')}; {v.get('rows')} rows; csv "
          f"sha256 {v.get('csv_sha256')}; pose columns == placeholder: {v.get('pose_columns_identical_to_placeholder_pack')}; "
          f"problems {v.get('problems')}; kit unchanged {not diff}; ok {res['ok']}")  # fmt: skip
    return 0 if res["ok"] else 1


# ================================================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "prepare", "pack", "validate", "pack-commit"))
    ap.add_argument("--commit-url")
    ap.add_argument("--branch-head-file")
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    return {"check": cmd_check, "prepare": cmd_prepare, "pack": cmd_pack, "validate": cmd_validate,
            "pack-commit": cmd_pack_commit}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
