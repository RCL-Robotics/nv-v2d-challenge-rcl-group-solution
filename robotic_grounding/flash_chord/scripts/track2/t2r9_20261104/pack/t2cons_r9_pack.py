#!/usr/bin/env python3
"""t2cons_r9_pack.py: pack the R9-CLOSEST file of fc_runs/submissions/t2_consensus_r9_20261103 (PREREG_R9 6): every
episode's S9 job's phase-F run t* of the R9-CLOSEST trial t*, validated offline with the kit's uploader, then re-packed
with the submission commit URL. PREREG_R9 6: "packing R9-CLOSEST runs separately with round 8's pack procedure, adapted
and pinned first". This is round 8's pack script t2r8_pack_20261103/tools/t2cons_r8_pack.py (fd0a48c0..., itself round
7's scripts/t2cons_r7_pack.py a4b13eaa... with round-8 bookkeeping) with only the round-9 bookkeeping changed; it is
pinned in PACK_R9_PINS.md before it prepares or packs.

Changes from round 8's script:
  W1  gate: round 9's chain. The round-9 scorer r9_select.py (1bb9c8ba..., pinned in PREREG_R9_PINS) is loaded from its
      verified bytes; it loads round 8's scorer r8_select.py (735149ac...) and through it round 7's scorer (e7545dfa...)
      unchanged. chain9("F9", "check") checks PREREG_R9 / PINS / LOCK, round 8's chain and every line of
      SELECTION_R8.sha256, POOL_R9.sha256, the phase-F heartbeat, the integrity baseline and SELECTION_S9.sha256; then
      SELECTION_R9.sha256 must equal its pinned sha256 (47e5a8d7...) and every line of it must hold, and SELECTION_R9.json
      must be written by the pinned r9_select.py (phase F, no keep-r8 reason, no deadline stop). Kit pins and
      KIT_MANIFEST as in rounds 7-8.
  W2  three run sources: a kept round-8 pick uses its recorded phase-F run t* (round 7's runs_f/, ROLLOUTS_F7.sha256,
      TRACKS_R7.sha256 for a round-7 pick kept in round 8; round 8's runs_f/, ROLLOUTS_F8.sha256, TRACKS_R8.sha256 for
      episode 13's round-8 replacement); a round-9 replacement uses its round-9 phase-F run t* (round 9's runs_f/,
      ROLLOUTS_F9.sha256, TRACKS_R9.sha256). Each run is also checked against round 9's own record of it
      (R8_RECORDS_R9.json phase_f_runs of the incumbent, or the finalist's runs in SELECTION_R9.json). Raw input
      reference: round 1's inputs/TRACKS.sha256 (as rounds 7-8).
  W3  r9_closest = SELECTION_R9.json r9_closest_file, re-derived from the frozen records (S9 composition and run source,
      R9-CLOSEST trial, the run's ok try, its rollout sha256, the job's export sha256 and input).
  W4  outputs under fc_runs/submissions/t2r9_pack_20261104/ (pack_r9_placeholder/, submission_r9/); the round-9, round-8
      and round-7 folders are only read. The read-only comparison is with the R8 file (t2r8_pack_20261103/submission_r8/
      T2_R8_closest_trial07_8387874.csv, pinned in PACK_R9_PINS): its per-episode checkpoint digests must differ from
      this file's exactly on the episodes whose S9 job differs from the round-8 pick.
  W5  pack-commit: unchanged from round 8 (the commit must exist in the clean-history clone
      tmp_analysis/t2_clean_history_20261102/build/repo and the ls-remote listing written after the push must show
      refs/heads/dev.haoran.t2clean_20261102 at it); the file is named T2_R9_closest_trial<t*>_<commit7>.csv.
The kit packer tools/pack_submission.py runs UNCHANGED with round 7's arguments (round 4's pack_command): first with
the placeholder commit URL (pack), then with the submission commit URL (pack-commit). Its own "locally computed: ..."
line is dropped from every stored log (a "withheld" marker) and is never parsed or printed.
Commands: check (dry) | prepare | pack | validate | pack-commit      (in this order)
Writes only under fc_runs/submissions/t2r9_pack_20261104/ (new files only). Nothing uploaded, nothing signalled, CPU
only.
Run: conda_envs/v2dkit/bin/python -I -B fc_runs/submissions/t2r9_pack_20261104/tools/t2cons_r9_pack.py <cmd>
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
R9 = ME / "fc_runs/submissions/t2_consensus_r9_20261103"
R8_FILE = ME / "fc_runs/submissions/t2r8_pack_20261103/submission_r8/T2_R8_closest_trial07_8387874.csv"  # read only
R8_FILE_SHA = "26232e39adb6e0c5e577358d60bb2003193c31363a6d178dfa2d26918e8c3dbb"
WORK = ME / "fc_runs/submissions/t2r9_pack_20261104"
OUT = WORK / "pack_r9_placeholder"
SUB = WORK / "submission_r9"
R9S_FILE, R9S_SHA = R9 / "r9_select.py", "1bb9c8ba784de1a0372f70d60fd5108e49d23a49111ccfc948dc1594eeba2096"
S7_FILE, S7_SHA = ME / "scripts/t2cons_s7_pack.py", "fef326d577af9075c439090b4ca4f8f8cfaedc75dd98bd08752598f03d56eebf"
PINS_C, PINS_C_SHA_FILE = WORK / "PACK_R9_PINS.md", WORK / "PACK_R9_PINS.sha256"
SEL, SEL_SHA = R9 / "SELECTION_R9.json", R9 / "SELECTION_R9.sha256"
SEL_SHA_PINNED = "47e5a8d7d1f00e11d8dca0609d0b02375d173d3dc714c38646998f64cdbf1dbb"  # round 9's final lock
FILES = ["r9_closest"]
WITHHELD = "locally computed: [withheld by t2cons_r9_pack.py: this line is not recorded]"
WORKTREE = ME / "tmp_analysis/t2_clean_history_20261102/build/repo"
BRANCH = "dev.haoran.t2clean_20261102"
URL_RE = re.compile(r"^https://github\.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/([0-9a-f]{40})$")
BASIS = "own-input and consensus-B scores only"
SOURCE_LABEL = {"r7F": "round 7 phase F", "r8F": "round 8 phase F", "r9F": "round 9 phase F"}


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


R9S = _load("r9_select_pinned_pack", R9S_FILE, R9S_SHA)  # round-9 scorer: chain9, load_jobs9, incumbents9
R8S = R9S.R8S  # round-8 scorer (735149ac), as loaded by r9_select.py: load_jobs8, TRACKS (round 8)
R7C = R9S.R7C  # round-7 scorer (e7545dfa), as loaded by r8_select.py: load_jobs, load_evals, PH (with F8 / F9)
S7 = _load("t2cons_s7_pack_pinned_r9", S7_FILE, S7_SHA)  # round-1 pack: helpers, kit pins, placeholder, raw npz writer
C = R7C.C  # the round-7 scorer's t2cons_consensus instance (extended guard)
MINE = Path(__file__).resolve()
sha, now, stamp, dump, write_new, copy_new, read_json = S7.sha, S7.now, S7.stamp, S7.dump, S7.write_new, S7.copy_new, \
    S7.read_json  # fmt: skip


def first_token(p: Path) -> str:
    return (p.read_text().split() or [""])[0] if p.is_file() else ""


# ================================================================================================ gate (W1)
def gate(need_c: bool) -> tuple[list[str], list[str]]:
    fatal, info = R9S.chain9("F9", "check")  # PREREG_R9 / PINS / LOCK, round-8 chain, POOL_R9, phase F, integrity, S9
    if not SEL_SHA.is_file():
        fatal.append("SELECTION_R9.sha256 does not exist: R9-CLOSEST may not be packed (PREREG_R9 5.6 / 6)")
        return fatal, info
    if sha(SEL_SHA) != SEL_SHA_PINNED:
        fatal.append(f"SELECTION_R9.sha256 {sha(SEL_SHA)[:16]} != pinned {SEL_SHA_PINNED[:16]}")
    n, listed = 0, False
    for rel, h0 in R9S.hash_lines(SEL_SHA):
        n += 1
        p = R9S.pin_path(rel)
        listed |= p == SEL
        if R9S.pin_hash(p) != h0:
            fatal.append(f"SELECTION_R9.sha256: {rel} changed")
    if not listed:
        fatal.append("SELECTION_R9.sha256 does not list SELECTION_R9.json")
    info.append(f"SELECTION_R9.sha256 ({sha(SEL_SHA)[:16]}...): {n} entries verify; lists SELECTION_R9.json {listed}")
    sel = read_json(SEL)
    if sel.get("script_sha256") != R9S_SHA or sel.get("phase") != "F" or sel.get("prereg_r9_sha256") != R9S.PREREG_SHA \
            or sel.get("basis") != BASIS or sel.get("keep_r8_reason") is not None or sel.get("deadline_stop") is not None:  # fmt: skip
        fatal.append("SELECTION_R9.json: script / phase / PREREG / basis / keep-r8 reason / deadline stop unexpected")
    info.append(f"SELECTION_R9.json written by r9_select.py {R9S_SHA[:8]}; replaced {sel.get('replaced_episodes')}; "
                f"R9-CLOSEST trial {sel.get('r9_closest_trial')}")  # fmt: skip
    for rel, want in S7.KIT_PINS.items():
        hh = sha(S7.KIT / rel) if (S7.KIT / rel).is_file() else "missing"
        if hh != want:
            fatal.append(f"kit {rel} sha256 {hh[:16]} != pinned {want[:16]}")
    man = read_json(S7.KIT / "KIT_MANIFEST.json")["files"]
    bad = [p for p, hh in man.items() if not (S7.KIT / p).is_file() or sha(S7.KIT / p) != hh]
    if bad:
        fatal.append(f"KIT_MANIFEST mismatches: {bad[:5]}")
    info.append(f"kit pins ({len(S7.KIT_PINS)}) hold; KIT_MANIFEST: {len(man)} files, {len(bad)} mismatched")
    hr8 = sha(R8_FILE) if R8_FILE.is_file() else "missing"
    if hr8 != R8_FILE_SHA:
        fatal.append(f"R8 file {R8_FILE.name} sha256 {hr8[:16]} != {R8_FILE_SHA[:16]}")
    if need_c:
        if not PINS_C.is_file():
            fatal.append("PACK_R9_PINS.md is missing: this script must be pinned before it prepares / packs")
            return fatal, info
        g, r = sha(PINS_C), first_token(PINS_C_SHA_FILE)
        if g != r:
            fatal.append(f"PACK_R9_PINS.md {g[:16]} != PACK_R9_PINS.sha256 {r[:16]}")
        pins = R9S._pins(PINS_C.read_text())
        mine = sha(MINE)
        if (str(MINE), mine) not in pins:
            fatal.append(f"PACK_R9_PINS.md does not pin this script ({mine[:16]})")
        for rel, h0 in pins:
            p = R9S.pin_path(rel)
            if (sha(p) if p.is_file() else "missing") != h0:
                fatal.append(f"PACK_R9_PINS pin {rel} changed")
        info.append(f"PACK_R9_PINS.md ({g[:16]}) == its sha file; {len(pins)} pins hold; pins this script {mine[:16]}")
    return fatal, info


def require(need_c: bool) -> bool:
    fatal, info = gate(need_c)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


# ================================================================================================ plan (W2, W3)
def plan() -> tuple[dict, dict, list[str]]:
    """episode -> (run source, job, run); re-derived from the frozen SELECTION_R9.json records."""
    sel = read_json(SEL)
    t = sel.get("r9_closest_trial")
    if t is None:
        return {}, {}, ["no R9-CLOSEST trial (no complete S9 trial)"]
    s9, src = sel["compositions"]["S9"], sel["compositions"]["S9_run_source"]
    s8 = sel["compositions"]["S8_round8"]
    rows9 = {r["trial"]: r for r in sel["trials"]}
    problems = []
    if not rows9[t]["complete"]:
        problems.append(f"trial {t} is not complete for S9")
    incs = R9S.incumbents9()
    dec = {int(x["episode"]): x for x in sel["episodes"]}
    jobs7 = R7C.load_jobs("F")
    ev7, _, lp7 = R7C.load_evals("F", jobs7)
    problems += lp7
    sources = {"r7F": {"label": SOURCE_LABEL["r7F"], "jobs": jobs7, "evals": ev7, "roll": C.hash_list(R7C.PH["F"]["rollouts"]),
                       "trk": C.hash_list(R7C.TRACKS)}}  # fmt: skip
    if any(v == "r8F" for v in src.values()):
        jobs8 = R8S.load_jobs8("F8")
        ev8, _, lp8 = R7C.load_evals("F8", jobs8)
        problems += lp8
        sources["r8F"] = {"label": SOURCE_LABEL["r8F"], "jobs": jobs8, "evals": ev8,
                          "roll": C.hash_list(R7C.PH["F8"]["rollouts"]), "trk": C.hash_list(R8S.TRACKS)}  # fmt: skip
    if any(v == "r9F" for v in src.values()):
        jobs9 = R9S.load_jobs9("F9")
        ev9, _, lp9 = R7C.load_evals("F9", jobs9)
        problems += lp9
        sources["r9F"] = {"label": SOURCE_LABEL["r9F"], "jobs": jobs9, "evals": ev9,
                          "roll": C.hash_list(R7C.PH["F9"]["rollouts"]), "trk": C.hash_list(R9S.TRACKS)}  # fmt: skip
    replaced = set(sel.get("replaced_episodes") or [])
    out = {}
    for e in range(30):
        k = f"{e:02d}"
        x = sel["r9_closest_file"][k]
        job, s = s9[k], src[k]
        kept = k not in replaced
        if s not in sources or (s in ("r7F", "r8F")) != kept or (kept and (job != incs[e]["job"] or s != incs[e]["phase_f_source"])) \
                or (job == s8[k]) != kept or (not kept and dec[e].get("pick") != job):  # fmt: skip
            problems.append(f"ep{k}: run source {s} / job {job} inconsistent with the replaced episodes {sorted(replaced)}")
            continue
        S = sources[s]
        j, ev = S["jobs"].get(job), S["evals"].get(job, {}).get(t)
        if j is None or x["job"] != job or int(x["run"]) != t or ev is None or int(x["try"]) != int(ev["try"]) or \
                x["rollout"] != ev["parquet"] or S["roll"].get(ev["parquet"]) != x["rollout_sha256"] or \
                j["export_sha256"] != x["export_sha256"] or j["input"] != x["input"]:  # fmt: skip
            problems.append(f"ep{k}: r9_closest_file record {x['job']}#{x['run']} != re-derived {job}#{t} ({s})")
        # round 9's own record of the run (incumbent: R8_RECORDS_R9.json; replacement: SELECTION_R9.json finalist runs)
        rec9 = incs[e]["phase_f_runs"].get(t) if kept else (dec[e].get("finalist_runs") or {}).get(str(t))
        if rec9 is None or not rec9.get("ok") or rec9.get("flagged") or int(rec9.get("try", -1)) != int(x["try"]) or \
                (kept and rec9.get("parquet") != x["rollout"]):  # fmt: skip
            problems.append(f"ep{k}: run {t} of {job} is not an ok, non-flagged run in round 9's records")
        out[e] = (s, job, t)
    return out, sources, problems


def episode_source_r9(K, S: dict, job: str, k: int, e: int) -> tuple[dict, list[str]]:
    """Round 8's episode_source_r8 logic (round 7's episode_source_r7), with the run source's records (W2)."""
    j, rec = S["jobs"][job], S["evals"][job][k]
    pq = Path(rec["parquet"])
    problems = []
    h = C.sha256_file(pq)
    if S["roll"].get(str(pq)) != h:
        problems.append(f"{job} run {k}: rollout sha256 {h[:16]} != the phase's ROLLOUTS list")
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
        want = S["trk"].get(str(fr_src))
    fr_h = C.sha256_file(fr_src) if fr_src.is_file() else "missing"
    if want != fr_h:
        problems.append(f"{job}: frame reference source {fr_src} sha256 {fr_h[:16]} != its frozen record")
    return {"episode": e, "job": job, "run": k, "try": int(rec["try"]), "input": kind, "source": S["label"],
            "role": j["role"], "owner": j["owner"], "machine": j["machine"], "export_sha256": j["export_sha256"],
            "rollout_src": str(pq), "rollout_sha256": h, "json_src": str(js), "json_sha256": jh,
            "frame_ref_kind": "written_from_released_tier2" if kind == "raw" else "byte_copy",
            "frame_ref_src": str(fr_src), "frame_ref_src_sha256": fr_h, "input_reference": j["input_reference"]}, problems  # fmt: skip


def pack_command(d: Path, url: str, out_csv: Path) -> list[str]:
    """The kit packer, run directly, with the packer arguments of rounds 1-8 (round 4's pinned pack_command)."""
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
           "packer_lines_withheld": n_withheld,
           "input_reference_line": next((ln for ln in kept if ln.startswith("input reference check")), None),
           "frame_reference_line": next((ln for ln in kept if ln.startswith("frame conversion reference")), None),
           "note": "kit packer run directly (no score wrapper); its 'locally computed' line was not recorded"}  # fmt: skip
    rec["ok"] = bool(proc.returncode == 0 and out_csv.is_file())
    return rec


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    lines = [f"t2cons_r9_pack.py check at {now()} (dry; nothing written except this file)", f"script sha256 {sha(MINE)}"]
    fatal, info = gate(need_c=False)
    lines += ["", "GATE"] + ["  " + x for x in info]
    if fatal:
        lines += ["FATAL"] + ["  " + x for x in fatal]
        print("\n".join(lines))
        return 1
    K = C.load_kit()
    pl, sources, problems = plan()
    lines += ["", "FILE r9_closest (episode: job [input, role, run source] run/try)"]
    for e in sorted(pl):
        s, job, k = pl[e]
        r, p2 = episode_source_r9(K, sources[s], job, k, e)
        problems += p2
        lines.append(f"  ep{e:02d}: {r['job']} [{r['input']}, {r['role']}, {r['source']}] run {r['run']} try {r['try']}  rollout "
                     f"{r['rollout_sha256'][:12]}  export {r['export_sha256'][:12]}  frame ref {r['frame_ref_kind']} {r['frame_ref_src']}")
    if len(pl) != 30:
        problems.append(f"plan holds {len(pl)} episodes, not 30")
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
    pl, sources, problems = plan()
    if problems or len(pl) != 30:
        print("PROBLEMS:\n  " + "\n  ".join(problems or [f"plan holds {len(pl)} episodes"]))
        return 1
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
            s_key, job, k = pl[e]
            s0, p2 = episode_source_r9(K, sources[s_key], job, k, e)
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
        meta = {"file": f, "kind": "R9-CLOSEST", "created_at": now(), "script_sha256": sha(MINE),
                "selection_r9_json_sha256": sha(SEL), "selection_r9_sha256_file_sha256": sha(SEL_SHA),
                "r9_closest_trial": sel["r9_closest_trial"], "replaced_episodes": sel["replaced_episodes"],
                "basis": sel["basis"], "episodes": rows,
                "note": "placeholder commit URL; never uploaded"}  # fmt: skip
        write_new(tmpd / "file.json", dump(meta))
        os.rename(tmpd, d)
        print(f"{f}: prepared ({len(rows)} episodes; trial {sel['r9_closest_trial']}; inputs "
              f"{sorted({r['input'] for r in rows})}; round-9 rows {[r['episode'] for r in rows if r['source'] == SOURCE_LABEL['r9F']]}; "
              f"round-8 rows {[r['episode'] for r in rows if r['source'] == SOURCE_LABEL['r8F']]})")
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
        if (d / "packed_placeholder.csv").exists() or (d / "pack_placeholder.log").exists():
            print(f"{f}: already packed, skipped")
            continue
        rec = run_packer(d, S7.PLACEHOLDER, d / "packed_placeholder.csv", d / "pack_placeholder.log", f)
        write_new(d / "pack_placeholder.json", dump(rec))
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


def r8_file_check(pd, per_ep, sel: dict) -> dict:
    """Read only: the R8 file's per-episode checkpoint digests; they must differ from this file's exactly on the
    episodes whose S9 job differs from the round-8 pick (W4)."""
    r = {"file": str(R8_FILE), "sha256": sha(R8_FILE)}
    fr = pd.read_csv(R8_FILE, usecols=["row_id", "checkpoint_sha256"], dtype=str)
    eps = fr["row_id"].astype(str).str.split("/").str[1]
    r8ep = fr.groupby(eps)["checkpoint_sha256"].agg(lambda s: sorted(s.astype(str).str.strip().unique()))
    s9, s8 = sel["compositions"]["S9"], sel["compositions"]["S8_round8"]
    same, diff = [], []
    for e in range(30):
        ep = f"e{e:06d}"
        (same if r8ep.get(ep) == per_ep.get(ep) else diff).append(f"{e:02d}")
    r["episodes_same_checkpoint_as_r8_file"] = same
    r["episodes_different_checkpoint"] = diff
    r["different_equals_changed_vs_round8"] = sorted(diff) == sorted(f"{e:02d}" for e in range(30)
                                                                     if s9[f"{e:02d}"] != s8[f"{e:02d}"])  # fmt: skip
    return r


def cmd_validate(a) -> int:
    if not require(need_c=True):
        return 1
    up, op, rs, pd, config, man, n_sample, npose = uploader_ctx()
    sel = read_json(SEL)
    bad = 0
    for f in FILES:
        d = OUT / f
        csvp = d / "packed_placeholder.csv"
        outp = d / "validate_placeholder.json"
        if outp.exists():
            print(f"{f}: already validated, skipped")
            continue
        if not (d / "pack_placeholder.json").is_file() or not read_json(d / "pack_placeholder.json").get("ok"):
            print(f"{f}: not packed ok")
            return 1
        meta = read_json(d / "file.json")
        res, per_ep = validate_csv(up, config, csvp, n_sample, npose, rs, op, man, pd, S7.PLACEHOLDER)
        res.update(file=f, at=now())
        digest_checks(res, per_ep, meta)
        r8 = r8_file_check(pd, per_ep, sel)
        if not r8["different_equals_changed_vs_round8"]:
            res["problems"].append("checkpoint digests differ from the R8 file on an unchanged episode (or agree on a changed one)")
        res["r8_file_readonly_check"] = r8
        res["ok"] = not res["problems"]
        bad += 0 if res["ok"] else 1
        write_new(outp, dump(res))
        print(f"{f}: uploader validate_submission {res['uploader_validate_submission']}; {res['rows']:,} rows (sample "
              f"{n_sample:,}); {res['distinct_checkpoints']} distinct checkpoints; csv sha256 {res['csv_sha256']}; pose "
              f"columns sha256 {res['pose_columns_sha256']}; problems: {res['problems'] or 'none'}")  # fmt: skip
        print(f"R8 file (read only): {r8['sha256']}; differing-checkpoint episodes {r8['episodes_different_checkpoint']}")
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
    d = OUT / "r9_closest"
    vp = d / "validate_placeholder.json"
    if not vp.is_file() or not read_json(vp).get("ok"):
        print("FATAL: the placeholder pack is not validated ok (run validate first)")
        return 1
    meta = read_json(d / "file.json")
    t = meta["r9_closest_trial"]
    name = f"T2_R9_closest_trial{int(t):02d}_{commit[:7]}.csv"
    out_csv = SUB / name
    if out_csv.exists():
        print(f"{out_csv} exists, skipped")
        return 0
    SUB.mkdir(parents=True, exist_ok=True)
    before = S7.kit_hashes()
    rec = run_packer(d, a.commit_url, out_csv, SUB / f"{name[:-4]}_pack.log", "r9_closest_commit")
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
        r8 = r8_file_check(pd, per_ep, read_json(SEL))
        v["r8_file_readonly_check"] = r8
        if not r8["different_equals_changed_vs_round8"]:
            v["problems"].append("checkpoint digests differ from the R8 file on an unchanged episode (or agree on a changed one)")
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
