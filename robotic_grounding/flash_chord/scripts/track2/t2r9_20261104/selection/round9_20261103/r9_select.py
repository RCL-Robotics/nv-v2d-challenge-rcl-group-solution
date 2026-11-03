#!/usr/bin/env python3
"""r9_select.py: scorer and selection of round 9 (fc_runs/submissions/t2_consensus_r9_20261103; PREREG_R9.md 4-7).

Round 9 is round 8's rule (PREREG_R8 sections 4-5 = PREREG_R7 5.1-5.6) with the deltas D1-D6 of PREREG_R9. This script
loads round 8's scorer fc_runs/submissions/t2_consensus_r8_20261103/r8_select.py (735149ac...) from its verified bytes,
and through it round 7's scorer scripts/t2cons_r7_consensus.py (e7545dfa...). The rule functions are called UNCHANGED:
  s_row / select_s      phase-S eligibility and the finalist of an incumbent episode (PREREG_R7 5.3 / 5.4)
  decide_f              phase-F replacement test (PREREG_R7 5.5)
  trial_rows / closest_trial   the complete trials and the closest-to-mean trial (PREREG_R7 5.6)
and, through them, the pinned round-1 / round-2 modules (rollout_on_grid, metrics_vs, load_refs, grid, load_kit, stats,
floor_count, integrity, the own-input gate auc_seed and run_flags). Round 8's frozen records are read through round 8's
own verified loaders (chain8, scored_inputs8, incumbents8, load_r7records, r7_phase_s, r7_phase_f). The gate / score loops
below are round 8's loops over round 9's runs. Selection uses only own-input and consensus-B scores.

Round-9 bookkeeping (PREREG_R9): every episode's incumbent is its round-8 pick (S8, the R8 file, trial 7). Its phase-F
runs are its recorded ones (a kept round-7 pick: its round-7 phase-F runs; the round-8 replacement of episode 13: its
round-8 phase-F runs); m_P_frozen is round 8's value (kept pick) or the mean of its 10 round-8 phase-F consensus-B AUCs
(the round-8 replacement); never re-run. Candidates are the POOL_R9.json rows with role screen (3 new runs), reuse8
(round-8 phase-S results) or reuse7 (round-7 phase-S results); barred rows are never finalists.

Commands (kit Python, CPU only, neutral cwd):  conda_envs/v2dkit/bin/python -I -B r9_select.py <cmd> [--phase S|F]
  r8records          build step: inputs_r9/R8_RECORDS_R9.json (30 incumbents with their recorded phase-F runs and bars;
                     the barred files per episode; the identities and phase-S records of round 8's and round 7's phase-S
                     candidates). New file, pinned.
  selftest           synthetic checks of the round-9 wrappers -> selftest_r9/selftest_r9.json
  table              print the incumbent table (read only)
  check  --phase S|F pin chain, jobs, runs; writes inputs_r9/ROLLOUTS_{S9,F9}.sha256 once
  gate   --phase S|F own-input gate of every round-9 run -> scores_{s9,f9}/own_input_gate_{s9,f9}.{csv,json}
  score  --phase S|F consensus-B scores of every round-9 run -> scores_{s9,f9}/consensus_{scores,job_stats}_*
                     (gate / score: an existing output written by this script from the current inputs is not repeated)
  select --phase S   -> selection_s9.{json,txt,md}, selection_s9_candidates.csv, final_r9.json; SELECTION_S9.sha256 LAST
  select --phase F   -> SELECTION_R9.json, selection_r9.md, selection_r9_episodes.csv, summary_r9_trials.csv;
                     SELECTION_R9.sha256 LAST
  keep-r8 --reason T every episode keeps its round-8 pick (PREREG_R9 7.3) -> the same SELECTION_R9 outputs; refused
                     while an r9_eval.py evaluation driver is alive (PREREG_R9 7.6)
New files only (mode 444). Nothing is signalled. No GPU.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r9_20261103"
R8ROOT = ME / "fc_runs/submissions/t2_consensus_r8_20261103"
R7ROOT = ME / "fc_runs/submissions/t2_consensus_r7_20261102"
R8S_FILE = R8ROOT / "r8_select.py"
R8S_SHA = "735149acfd54eb5db39a776b4775303e16283ca274f34e71215581a94ca2508e"  # PREREG_R8_PINS, PREREG_R9_PINS
SEL_R8_SHA = "e48cc8780a537f464ad72d8ef48577cfabe938decae8b435ede43b3176a4d11e"  # round 8's final lock (PREREG_R9 0)
PREREG, PREREG_SHA = ROOT / "PREREG_R9.md", "13c3b2a61e39cc761633283ae42bf94683df8d1ffd701d1076e3d61212a4c2a0"
PINS, PINS_SHA_FILE = ROOT / "PREREG_R9_PINS.md", ROOT / "PREREG_R9_PINS.sha256"
LOCK = ROOT / "LOCK_R9.sha256"
POOL, POOL_LIST = ROOT / "POOL_R9.json", ROOT / "POOL_R9.sha256"
R8REC = ROOT / "inputs_r9/R8_RECORDS_R9.json"
TRACKS = ROOT / "inputs_r9/TRACKS_R9.sha256"
RELOC_TSV = ROOT / "RELOCATION_R9.tsv"
EXPORTS = ROOT / "exports"
FINAL = ROOT / "final_r9.json"
SEL_S9, SEL_R9 = ROOT / "SELECTION_S9.sha256", ROOT / "SELECTION_R9.sha256"
SELFTEST = ROOT / "selftest_r9/selftest_r9.json"
DEADLINE_F = ROOT / "DEADLINE_F.json"  # written by r9_driver.py when the phase-F deadline stopped new evaluations
BASELINE, BASELINE_SHA = ROOT / "integrity/baseline.json", "ef9b5035c09b8ad40713d9fb08470eeccb2abe8845d4d1cce3037c49da1cd622"
NTRIALS, NRUNS_S = 10, 3
R8_CLOSEST = 7  # the R8 file's trial (SELECTION_R8.json r8_closest_trial)
THIS_ROUND = "r7"  # the tag round 7's load_evals gives rows of the phase's own evaluation CSV (here: round 9's CSV)


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


R8S = _load("r8_select_pinned_r9", R8S_FILE, R8S_SHA)  # round 8's scorer (never re-bound): round 8's records
R7C = R8S.R7C  # round 7's scorer, as loaded and verified by round 8's
C, G = R7C.C, R7C.G
MARGIN, FLOOR_F, MIN_FRESH = R7C.MARGIN, R7C.FLOOR_F, R7C.MIN_FRESH
R7C.PH["S9"] = {"tag": "s9", "name": "screen", "runs": ROOT / "runs_s", "nruns": NRUNS_S, "manifest": ROOT / "manifest_s9.json",
                "evals": ROOT / "own_input_evals_s9.csv", "hb": ROOT / "eval_r9_heartbeat_s9.json",
                "rollouts": ROOT / "inputs_r9/ROLLOUTS_S9.sha256", "sc": ROOT / "scores_s9", "lock": SEL_S9}  # fmt: skip
R7C.PH["F9"] = {"tag": "f9", "name": "final", "runs": ROOT / "runs_f", "nruns": NTRIALS, "manifest": ROOT / "manifest_f9.json",
                "evals": ROOT / "own_input_evals_f9.csv", "hb": ROOT / "eval_r9_heartbeat_f9.json",
                "rollouts": ROOT / "inputs_r9/ROLLOUTS_F9.sha256", "sc": ROOT / "scores_f9", "lock": SEL_R9}  # fmt: skip
PHASES = {"S": "S9", "F": "F9"}
MINE = Path(__file__).resolve()
RULE_S9 = ("PREREG_R9 5.3 / 5.4 = PREREG_R8 5.3 / 5.4 = PREREG_R7 5.3 / 5.4 for incumbent episodes, applied to every "
           "episode with at least one round-9 candidate (role screen, reuse8 or reuse7): a candidate is S-eligible iff its "
           "export is ok, it has 3 ok official runs, none flagged by the own-input gate, and own-input SP-SR = MP-SR = 1 in "
           ">= 2 of them; cons_S = float64 mean of its 3 consensus-B AUCs; finalist = the S-eligible candidate with the "
           "highest cons_S among those with cons_S >= m_P_frozen + 0.01 (exact floats); ties: lower population variance of "
           "the 3 consensus AUCs, lower mean consensus MPPE, job name; barred candidates are never finalists; at most one "
           "finalist per episode. Round-8 / round-7 phase-S results are reused as recorded (roles reuse8 / reuse7).")  # fmt: skip
RULE_F9 = ("PREREG_R9 5.5 / 5.6 = PREREG_R8 5.5 / 5.6: m_F = float64 mean of the finalist's 10 phase-F consensus-B AUCs; "
           "m_P_fresh = the mean of the incumbent's recorded phase-F runs (ok, non-flagged runs, >= 8 required, else +inf; "
           "never re-run): round-7 phase-F runs for a kept round-7 pick, round-8 phase-F runs for the round-8 replacement; "
           "the finalist replaces the round-8 pick iff (a) 10 ok, non-flagged phase-F runs, (b) own-input SP-SR = MP-SR = 1 "
           "in >= 8 of them, (c) m_F >= max(m_P_frozen, m_P_fresh) + 0.01 (exact floats). S9 = per episode the result. "
           "Trial t is complete iff every episode's job has an ok, non-flagged run t (a kept round-8 pick: its recorded "
           "phase-F run t; a replacement: its round-9 phase-F run t); R9-CLOSEST = the complete trial whose 30-episode mean "
           "own-input AUC is closest to the mean over the complete trials (ties: consensus-B distance, then lower t). "
           "Never a best-of.")  # fmt: skip


# ================================================================================================ helpers
def sha(p) -> str:
    return R7C.raw_sha(p)


def pin_hash(p) -> str:
    return R7C.pin_hash(Path(p))


def first_token(p: Path) -> str:
    return (C.guard(p).read_text().split() or [""])[0] if Path(p).is_file() else ""


def write_new(p: Path, text: str) -> None:
    p = C.guard(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp_r9s")
    if tmp.exists():
        raise SystemExit(f"stale temporary {tmp}")
    tmp.write_text(text)
    os.replace(tmp, p)
    os.chmod(p, 0o444)


def pin_path(rel: str) -> Path:
    """PREREG_R9 path rule: '/...' absolute, everything else relative to the round-9 folder."""
    return Path(rel) if rel.startswith("/") else ROOT / rel


def rel_name(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT) + "/") else str(p)


def hash_lines(listing: Path) -> list[tuple[str, str]]:
    return R7C.hash_lines(listing)


def ep_key(e: int) -> str:
    return f"{e:02d}"


def fmt(x, spec=".4f") -> str:
    return "" if x is None else format(float(x), spec)


def csv_text(rows: list[dict], cols: list[str]) -> str:
    return C.csv_text(rows, cols)


runinfo_of = R8S.runinfo_of  # PREREG_R7 5.5 run records (round 7's cmd_select_f runinfo_of, unchanged)
runs_from_json = R8S.runs_from_json
s_info = R8S.s_info  # PREREG_R7 5.3 candidate record (round 7's cmd_select_s info, unchanged)


# ================================================================================================ round-9 rule wrappers
def select_episode_s9(m_P_frozen: float, cands: list[str], info: dict, barred: set) -> tuple[str | None, list[dict]]:
    """PREREG_R9 5.4: round 7's select_s (incumbent episode), unchanged."""
    return R7C.select_s(m_P_frozen, cands, info, barred=barred)


def decide_episode_f9(m_P_frozen: float, inc_runs: dict, fin_job: str | None, fin_runs: dict | None) -> dict:
    """PREREG_R9 5.5: round 7's decide_f with the incumbent's recorded phase-F runs as its fresh runs."""
    return R7C.decide_f(m_P_frozen, inc_runs, fin_job, fin_runs)


def closest9(comp: dict[int, str], runinfo: dict[str, dict[int, dict]]) -> tuple[list[dict], int | None]:
    """PREREG_R9 5.6: round 7's trial_rows / closest_trial on the S9 composite."""
    rows = R7C.trial_rows(comp, runinfo)
    return rows, R7C.closest_trial(rows)


def would_pass(runs: dict, bar: float) -> bool:
    """The round-9 replacement test (a)-(c) applied to recorded phase-F runs (used only to assert that barred files fail
    it, PREREG_R9 2.7)."""
    good = [t for t in sorted(runs) if runs[t]["ok"] and not runs[t]["flagged"]]
    if good != list(range(1, NTRIALS + 1)):
        return False
    mF = C.stats([runs[t]["cons_auc"] for t in good])["mean"]
    return bool(mF >= bar and sum(1 for t in good if runs[t]["own_sp"] == 1.0 and runs[t]["own_mp"] == 1.0) >= FLOOR_F)


# ================================================================================================ pin chain
def _pins(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64:
            out.append((f[1], f[2]))
    return out


def driver_running() -> list[str]:
    """Live r9_eval.py run processes, read from /proc only (a plan-only --dry-run and zombies do not count)."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            args = [x.decode(errors="replace") for x in (d / "cmdline").read_bytes().split(b"\0") if x]
        except OSError:
            continue
        if any(x.endswith("r9_eval.py") for x in args) and "run" in args and "--dry-run" not in args:
            out.append(f"pid {d.name}: {' '.join(args)[:160]}")
    return out


def pidfile_live() -> str | None:
    """driver.pid names a live r9_eval.py process (a stale file of an ended process does not count)."""
    try:
        pid = int((ROOT / "driver.pid").read_text().split()[0])
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError, IndexError):
        return None
    return f"driver.pid {pid}" if b"r9_eval.py" in cmd else None


def r9_docs(require_lock: bool) -> tuple[list[str], list[str]]:
    """PREREG_R9.md == PREREG_R9.sha256 == hard-coded; PREREG_R9_PINS.md == its sha file, every pin holds and pins this
    script; LOCK_R9.sha256 lines hold."""
    fatal, info = [], []
    got, rec = pin_hash(PREREG), first_token(ROOT / "PREREG_R9.sha256")
    if not got == rec == PREREG_SHA:
        fatal.append(f"PREREG_R9.md sha256 {got[:16]} (sha file {rec[:16]}, hard-coded {PREREG_SHA[:16]})")
    else:
        info.append(f"PREREG_R9.md {got[:16]} == PREREG_R9.sha256 == hard-coded")
    mine = sha(MINE)
    if not PINS.is_file():
        fatal.append("PREREG_R9_PINS.md missing")
    else:
        got, rec = sha(PINS), first_token(PINS_SHA_FILE)
        pins = _pins(PINS.read_text())
        if got != rec:
            fatal.append(f"PREREG_R9_PINS.md {got[:16]} != PREREG_R9_PINS.sha256 {rec[:16]}")
        if ("r9_select.py", mine) not in pins:
            fatal.append(f"PREREG_R9_PINS.md does not pin this script {mine[:16]}")
        bad = [rel for rel, h0 in pins if pin_hash(pin_path(rel)) != h0]
        fatal += [f"PREREG_R9_PINS pin {rel} changed" for rel in bad]
        info.append(f"PREREG_R9_PINS.md {got[:16]}: {len(pins)} pins, {len(bad)} changed; pins this script {mine[:16]}")
    if LOCK.is_file():
        bad = [rel for rel, h0 in hash_lines(LOCK) if pin_hash(pin_path(rel)) != h0]
        fatal += [f"LOCK_R9.sha256: {rel} changed" for rel in bad]
        info.append(f"LOCK_R9.sha256: {len(hash_lines(LOCK))} lines, {len(bad)} changed")
    elif require_lock:
        fatal.append("LOCK_R9.sha256 missing")
    return fatal, info


def r8_chain() -> tuple[list[str], list[str]]:
    """Round 8's own verified chain (its chain8 at stage 'check' of phase F: its documents, round 7's chain, its pool,
    heartbeat, integrity, SELECTION_S8) and every line of SELECTION_R8.sha256 (== the pinned e48cc878...)."""
    fatal, info = R8S.chain8("F8", "check")
    fatal = [f"round 8: {x}" for x in fatal]
    if sha(R8S.SEL_R8) != SEL_R8_SHA:
        fatal.append(f"round 8: SELECTION_R8.sha256 {sha(R8S.SEL_R8)[:16]} != pinned {SEL_R8_SHA[:16]}")
    n = 0
    for rel, h0 in hash_lines(R8S.SEL_R8):
        n += 1
        if pin_hash(R8S.pin_path(rel)) != h0:
            fatal.append(f"round 8: SELECTION_R8.sha256 {rel} changed")
    info.append(f"round 8 chain: {len(fatal)} fatal; SELECTION_R8.sha256 {n} lines checked")
    return fatal, info


def chain9(ph: str, stage: str) -> tuple[list[str], list[str]]:
    fatal, info = r9_docs(require_lock=True)
    f8, i8 = r8_chain()
    fatal += f8
    info += i8
    if not R8REC.is_file():
        fatal.append("inputs_r9/R8_RECORDS_R9.json missing")
    for rel, h0 in hash_lines(POOL_LIST) if POOL_LIST.is_file() else []:
        if pin_hash(pin_path(rel)) != h0:
            fatal.append(f"POOL_R9.sha256: {rel} changed")
    if not POOL_LIST.is_file() or not any(pin_path(r) == POOL for r, _ in hash_lines(POOL_LIST)):
        fatal.append("POOL_R9.sha256 missing or does not list POOL_R9.json")
    P = R7C.PH[ph]
    hb = C.read_json(P["hb"]) if P["hb"].is_file() else {}
    ev, ex = hb.get("evals") or {}, hb.get("exports") or {}
    done = hb.get("state") == "done" and ev.get("pending") == 0 and ex.get("pending") == 0
    stopped_at_deadline = ph == "F9" and str(hb.get("state", "")).startswith("stopped") and DEADLINE_F.is_file()
    if not (done or stopped_at_deadline) or str(hb.get("runs_base")) != str(P["runs"]) or hb.get("phase") != P["name"] \
            or hb.get("halted"):  # fmt: skip
        fatal.append(f"phase {ph} evaluations not finished: heartbeat state {hb.get('state')} evals {ev} exports {ex}")
    run = driver_running()
    live_pid = pidfile_live()
    if run or live_pid:
        fatal.append(f"an r9_eval driver is running: {run} {live_pid or ''}")
    info.append(f"phase {ph}: heartbeat state {hb.get('state')}, exports {ex}, evals {ev}; deadline stop recorded: "
                f"{DEADLINE_F.is_file()}")  # fmt: skip
    if pin_hash(BASELINE) != BASELINE_SHA:
        fatal.append("integrity/baseline.json is not the byte copy of round 1's baseline (ef9b5035...)")
    else:
        diff = C.integrity_diff(json.loads(BASELINE.read_text()), C.integrity())
        if diff:
            fatal.append(f"kit / stock trees differ from integrity/baseline.json: {diff[:5]}")
        info.append(f"kit + stock trees == integrity/baseline.json: {not diff}")
    if ph == "F9":
        if not SEL_S9.is_file():
            fatal.append("SELECTION_S9.sha256 missing")
        else:
            listed = False
            for rel, h0 in hash_lines(SEL_S9):
                listed |= pin_path(rel) == FINAL
                if pin_hash(pin_path(rel)) != h0:
                    fatal.append(f"SELECTION_S9.sha256: {rel} changed")
            if not listed:
                fatal.append("SELECTION_S9.sha256 does not list final_r9.json")
    mine = sha(MINE)
    if stage in ("gate", "score", "select"):
        st = C.read_json(SELFTEST) if SELFTEST.is_file() else {}
        if st.get("pass") is not True or st.get("script_sha256") != mine:
            fatal.append("selftest_r9/selftest_r9.json is not a PASS of this script")
        if not P["rollouts"].is_file():
            fatal.append(f"{P['rollouts'].name} missing: run check --phase {ph[0]} first")
    if stage in ("score", "select"):
        F = R7C.files_of(ph)
        g = C.read_json(F["gate_json"]) if F["gate_json"].is_file() else {}
        if g.get("pass") is not True or g.get("script_sha256") != mine or g.get("csv_sha256") != (
                sha(F["gate_csv"]) if F["gate_csv"].is_file() else None):  # fmt: skip
            fatal.append(f"{F['gate_json'].name} is not a PASS of this script")
    if stage in ("gate", "score", "select") and P["lock"].exists():
        fatal.append(f"{P['lock'].name} exists: phase {ph} is frozen")
    if stage in ("gate", "score", "select") and ph == "S9" and SEL_R9.exists():
        fatal.append("SELECTION_R9.sha256 exists")
    return fatal, info


def require(ph: str, stage: str) -> bool:
    fatal, info = chain9(ph, stage)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


# ================================================================================================ round-8 records (D1)
def r8_phase_s():
    """Round 8's verified phase-S records: job -> identity + the PREREG_R7 5.3 candidate record (round 8's own loaders)."""
    js, g, jobs, evals, scores, flagged = R8S.scored_inputs8("S8")
    excluded = set(g.get("validation_failed_jobs") or [])
    return {c: {"job": jobs[c], "info": s_info(c, jobs, evals, excluded, flagged, js["jobs"])} for c in jobs}


def cmd_r8records(a) -> int:
    if R8REC.exists():
        print(f"FATAL: {R8REC} exists (new files only)")
        return 2
    f8, i8 = r8_chain()
    for x in i8:
        print("  " + x)
    if f8:
        print("FATAL:\n  " + "\n  ".join(f8))
        return 2
    sel8 = C.read_json(R8ROOT / "SELECTION_R8.json")
    s8, src8 = sel8["compositions"]["S8"], sel8["compositions"]["S8_run_source"]
    if sel8.get("r8_closest_trial") != R8_CLOSEST:
        raise SystemExit(f"SELECTION_R8.json r8_closest_trial {sel8.get('r8_closest_trial')} != {R8_CLOSEST}")
    eps8 = {int(x["episode"]): x for x in sel8["episodes"]}
    replaced8 = [int(k) for k in sel8["replaced_episodes"]]
    fin8 = C.read_json(R8ROOT / "final_r8.json")
    incs8 = R8S.incumbents8()  # the round-7 picks (round 8's incumbents) with their round-7 phase-F runs
    rec7 = R8S.load_r7records()
    pool8 = {r["job"]: r for r in C.read_json(R8ROOT / "POOL_R8.json")["rows"]}
    jsF, gF, jobsF8, evalsF8, scoresF8, flaggedF8 = R8S.scored_inputs8("F8")
    incs, barred = {}, {}
    for e in range(30):
        k = ep_key(e)
        job, src = s8[k], src8[k]
        x = eps8[e]
        if x.get("pick") != job:
            raise SystemExit(f"ep{k}: SELECTION_R8.json pick {x.get('pick')} != S8 {job}")
        if src == "r7F":
            I = incs8[e]
            if I["job"] != job or x.get("changed"):
                raise SystemExit(f"ep{k}: S8 {job} (r7F) is not round 8's kept incumbent {I['job']}")
            runs = I["r7_phase_f_runs"]
            m_frozen = I["m_P_frozen"]
            d = R7C.decide_f(m_frozen, runs, None, None)
            if d["bar"] != I["bar"] or d["m_P_fresh"] != I["m_P_fresh"] or d["bar"] != x.get("bar"):
                raise SystemExit(f"ep{k}: recomputed bar / m_P_fresh != R7_RECORDS_R8.json / SELECTION_R8.json")
            rec = {kk: v for kk, v in I.items() if kk != "r7_phase_f_runs"}
            rec.update(r8_status="round-7 pick kept in round 8", phase_f_source="r7F", phase_f_runs=runs,
                       phase_f_rollouts=str(R7ROOT / "inputs_r7/ROLLOUTS_F7.sha256"),
                       m_P_frozen_source=f"round 8's m_P_frozen ({I['m_P_frozen_source']})")  # fmt: skip
        elif src == "r8F":
            if e not in replaced8 or not x.get("changed"):
                raise SystemExit(f"ep{k}: S8 {job} (r8F) is not a round-8 replacement")
            j = jobsF8[job]
            runs = runinfo_of(job, evalsF8, scoresF8, flaggedF8)
            good = [t for t in sorted(runs) if runs[t]["ok"] and not runs[t]["flagged"]]
            if good != list(range(1, NTRIALS + 1)):
                raise SystemExit(f"ep{k}: the round-8 replacement {job} has good phase-F runs {good} (want 1..10)")
            m_frozen = C.stats([runs[t]["cons_auc"] for t in good])["mean"]
            if float(x["m_F"]) != m_frozen:
                raise SystemExit(f"ep{k}: m_F {x['m_F']!r} in SELECTION_R8.json != mean {m_frozen!r}")
            d = R7C.decide_f(m_frozen, runs, None, None)
            p8 = pool8[job]
            if p8["role"] == "screen":
                exrec = R8ROOT / "exports" / job / "export.json"
            elif p8["role"] == "reuse7":
                exrec = Path(p8["r7_export_record"])
            else:
                raise SystemExit(f"ep{k}: round-8 pool role {p8['role']} of {job}")
            ex = C.read_json(exrec)
            if ex.get("sha256") != j["export_sha256"] or ex.get("status") != "ok":
                raise SystemExit(f"ep{k}: export record {exrec} != round-8 phase-F job export")
            rec = {"episode": e, "job": job, "input": j["input"], "owner": j["owner"], "machine": j["machine"],
                   "tree": j["tree"], "iteration": j["iteration"], "run_dir": j["run_dir"], "pool_sha256": j["pool_sha256"],
                   "training_sha256": j["training_sha256"], "training_checkpoint": j["training_checkpoint"],
                   "source_checkpoint": j["source_checkpoint"], "export_record": str(exrec), "export_record_sha256": sha(exrec),
                   "export_sha256": j["export_sha256"], "export_file": j["export_file"], "parquet_sha256": j["parquet_sha256"],
                   "input_reference": j["input_reference"], "input_reference_file": j["input_reference_file"],
                   "input_reference_sha256": j["input_reference_sha256"], "selected_in": "round8",
                   "r8_status": "round-8 replacement", "r7_status": None, "phase_f_source": "r8F", "phase_f_runs": runs,
                   "phase_f_rollouts": str(R8ROOT / "inputs_r8/ROLLOUTS_F8.sha256"),
                   "m_P_frozen_source": "mean of its 10 round-8 phase-F consensus-B AUCs (== SELECTION_R8.json m_F)"}  # fmt: skip
        else:
            raise SystemExit(f"ep{k}: unknown S8 run source {src}")
        rec.update(m_P_frozen=m_frozen, m_P_fresh=d["m_P_fresh"], bar=d["bar"])
        incs[k] = rec
    # barred files (PREREG_R9 2.7): round 8's barred (its R7_RECORDS_R8 participants and carried pair), round 8's phase-F
    # participants other than its pick (finalists not picked, replaced incumbents)
    jobsF7, runinfoF7 = R8S.r7_phase_f()
    r7s_rec = rec7["r7_phase_s_candidates"]
    for e in range(30):
        k = ep_key(e)
        bar9 = incs[k]["bar"]
        lst = []
        for x in rec7["r7_phase_f_participants"].get(k, []):
            rr = runinfoF7.get(x["job"])
            if rr is None or would_pass(rr, bar9):
                raise SystemExit(f"ep{k}: barred {x['job']} has no round-7 phase-F runs or would pass the round-9 test")
            lst.append({**{kk: x.get(kk) for kk in ("job", "pool_sha256", "training_sha256", "training_checkpoint",
                                                   "parquet_sha256", "input_reference_sha256")},
                        "why": f"round-8 barred: {x['outcome']} (round-7 phase-F participant)", "r9_bar": bar9})  # fmt: skip
        for j, b in rec7["r7_barred_carried"].items():
            if int(b["episode"]) != e:
                continue
            y = r7s_rec[j]
            lst.append({**{kk: y.get(kk) for kk in ("job", "pool_sha256", "training_sha256", "training_checkpoint",
                                                   "parquet_sha256", "input_reference_sha256")},
                        "why": "round-8 barred: round 7's carried barred candidate (PREREG_R7 5.4)", "r9_bar": bar9})  # fmt: skip
        if e in replaced8:
            I = incs8[e]
            if would_pass(I["r7_phase_f_runs"], bar9):
                raise SystemExit(f"ep{k}: the replaced round-8 incumbent {I['job']} would pass the round-9 test")
            lst.append({**{kk: I.get(kk) for kk in ("job", "pool_sha256", "training_sha256", "training_checkpoint",
                                                   "parquet_sha256", "input_reference_sha256")},
                        "why": "round-8 phase-F participant: the round-8 incumbent replaced in round 8 (its round-7 "
                               "phase-F runs fail the round-9 test)", "r9_bar": bar9})  # fmt: skip
        for f in fin8["finalists"]:
            if int(f["episode"]) == e and f["job"] != s8[k]:
                rr = runinfo_of(f["job"], evalsF8, scoresF8, flaggedF8)
                if would_pass(rr, bar9):
                    raise SystemExit(f"ep{k}: round-8 finalist {f['job']} (not picked) would pass the round-9 test")
                jj = jobsF8[f["job"]]
                lst.append({"job": f["job"], "pool_sha256": jj["pool_sha256"], "training_sha256": jj["training_sha256"],
                            "training_checkpoint": jj["training_checkpoint"], "parquet_sha256": jj["parquet_sha256"],
                            "input_reference_sha256": jj.get("input_reference_sha256"),
                            "why": "round-8 phase-F participant: finalist not picked", "r9_bar": bar9})  # fmt: skip
        barred[k] = lst
    # phase-S candidates with recorded results: round 8's (its phase-S jobs) and round 7's (R7_RECORDS_R8)
    s_out = {}
    s8rec = r8_phase_s()
    for c, v in sorted(s8rec.items()):
        j, p = v["job"], pool8[c]
        row = R7C.s_row(c, v["info"])
        exrec = R8ROOT / "exports" / c / "export.json"
        s_out[c] = {"job": c, "round": "r8", "episode": j["episode"], "input": j["input"], "owner": j["owner"],
                    "machine": j.get("machine"), "tree": j.get("tree"), "iteration": j.get("iteration"),
                    "run_dir": j.get("run_dir"), "pool_sha256": p["sha256"], "training_sha256": p.get("training_sha256"),
                    "training_checkpoint": p.get("training_checkpoint"), "source_checkpoint": p.get("source_checkpoint"),
                    "parquet_sha256": p["parquet_sha256"], "input_reference_sha256": p.get("frame_ref_sha256"),
                    "export_status": j["export_status"], "export_sha256": j.get("export_sha256"),
                    "export_file": j.get("export_file"), "export_record": str(exrec) if j["export_status"] == "ok" else None,
                    "export_record_sha256": sha(exrec) if j["export_status"] == "ok" else None,
                    "s_eligible": row["eligible"], "n_own_ok": row["n_own_ok"], "floor_count": row["floor_count"],
                    "flagged": row["flagged"], "cons_mean": row.get("cons_mean")}  # fmt: skip
    for c, x in sorted(r7s_rec.items()):
        if c in s_out:
            raise SystemExit(f"{c}: a round-7 phase-S job name equals a round-8 phase-S job name")
        s_out[c] = {**x, "round": "r7"}
    out = {"generated_at": C.now(), "script_sha256": sha(MINE), "r8_select_sha256": R8S_SHA,
           "r7_consensus_sha256": R8S.R7C_SHA, "selection_r8_lock_sha256": sha(R8S.SEL_R8),
           "selection_r8_json_sha256": sha(R8ROOT / "SELECTION_R8.json"), "r7_records_r8_sha256": sha(R8S.R7REC),
           "pool_r8_sha256": sha(R8ROOT / "POOL_R8.json"), "r8_closest_trial": R8_CLOSEST,
           "rule": "PREREG_R9 4 (D1): incumbent = the round-8 pick (S8 = the R8 file, trial 7); m_P_frozen as recorded "
                   "(round-8 replacement: the mean of its 10 round-8 phase-F consensus-B AUCs); m_P_fresh and bar from its "
                   "recorded phase-F runs (round 7's decide_f)",
           "incumbents": incs, "barred": barred, "phase_s_candidates": s_out}  # fmt: skip
    write_new(R8REC, C.dump(out))
    print(f"wrote {R8REC} sha256 {sha(R8REC)}: {len(incs)} incumbents, {sum(len(v) for v in barred.values())} barred, "
          f"{sum(1 for v in s_out.values() if v['round'] == 'r8')} round-8 + {sum(1 for v in s_out.values() if v['round'] == 'r7')} "
          "round-7 phase-S candidate records")  # fmt: skip
    return cmd_table(a)


def load_r8records() -> dict:
    if not R8REC.is_file():
        raise SystemExit("inputs_r9/R8_RECORDS_R9.json missing")
    return C.read_json(R8REC)


def incumbents9() -> dict[int, dict]:
    rec = load_r8records()
    out = {}
    for k, x in rec["incumbents"].items():
        x = dict(x)
        x["phase_f_runs"] = runs_from_json(x["phase_f_runs"])
        out[int(k)] = x
    if sorted(out) != list(range(30)):
        raise SystemExit("R8_RECORDS_R9.json does not hold 30 incumbents")
    return out


def cmd_table(a) -> int:
    incs = incumbents9()
    print("| ep | incumbent (round-8 pick) | input | status | runs | m_P_frozen | m_P_fresh | bar |")
    print("|---|---|---|---|---|---:|---:|---:|")
    for e in range(30):
        x = incs[e]
        print(f"| {e:02d} | `{x['job']}` | {x['input']} | {x['r8_status']} | {x['phase_f_source']} | {x['m_P_frozen']:.6f} | "
              f"{x['m_P_fresh']:.6f} | {x['bar']:.6f} |")
    return 0


# ================================================================================================ round-9 jobs and runs
def pool_rows() -> dict[str, dict]:
    data = C.read_json(POOL)
    return {r["job"]: r for r in data["rows"]}


def reloc_rows() -> dict[str, dict]:
    if not RELOC_TSV.is_file():
        return {}
    with C.guard(RELOC_TSV).open(newline="") as fp:
        return {r["job"]: r for r in csv.DictReader(fp, delimiter="\t")}


def load_jobs9(ph: str) -> dict[str, dict]:
    """Phase S9: the screen rows of POOL_R9.json (manifest_s9.json, round 9's single exports). Phase F9: the finalists
    of final_r9.json (screen: round 9's export; reuse8: its round-8 export record; reuse7: its round-7 / round-6 export
    record; never re-exported)."""
    P = R7C.PH[ph]
    man = C.read_json(P["manifest"])
    if man.get("prereg_r9_sha256") != PREREG_SHA or man.get("phase") != P["name"] or man.get("pool_r9_sha256") != sha(POOL):
        raise SystemExit(f"{P['manifest'].name} was not built under PREREG_R9 / POOL_R9.json")
    pool = pool_rows()
    reloc = reloc_rows()
    trk, r1trk = C.hash_list(TRACKS), C.hash_list(R7C.R1_TRACKS)
    man_rows = {r["job"]: r for r in man["rows"]}
    if ph == "S9":
        plan = [(j, "screen", p) for j, p in sorted(pool.items()) if p["role"] == "screen"]
        if sorted(man_rows) != sorted(j for j, _, _ in plan):
            raise SystemExit("manifest_s9.json does not hold exactly the pool's screen jobs")
    else:
        fin = C.read_json(FINAL)
        plan = [(x["job"], "finalist", pool.get(x["job"]) or {}) for x in fin["finalists"]]
        fin_by = {x["job"]: x for x in fin["finalists"]}
        if sorted(man_rows) != sorted(j for j, _, _ in plan):
            raise SystemExit("manifest_f9.json does not hold exactly the finalists of final_r9.json")
    jobs: dict[str, dict] = {}
    for job, role, src in plan:
        if not src:
            raise SystemExit(f"{job}: not in POOL_R9.json")
        e, kind, prole = int(src["episode"]), src["input"], src["role"]
        if prole == "screen":
            rr = reloc.get(job)
            if src.get("relocated_file"):
                if rr is None or rr.get("status") != "ok" or rr["original_sha256"] != src["sha256"]:
                    raise SystemExit(f"{job}: no ok RELOCATION_R9.tsv row")
                tsha = rr["relocated_sha256"]
            else:
                tsha = src["sha256"]
        elif prole == "reuse8" and role == "finalist":
            tsha = src["r8_training_sha256"]
        elif prole == "reuse7" and role == "finalist":
            tsha = src["r7_training_sha256"]
        else:
            raise SystemExit(f"{job}: pool role {prole} cannot be a {role} job")
        d = {"job": job, "episode": e, "role": role, "id": role, "pool_role": prole, "input": kind, "owner": src["owner"],
             "machine": src.get("machine"), "host": src.get("host"), "tree": src.get("tree"), "iteration": src.get("iteration"),
             "run_dir": src.get("run_dir"), "run_label": src.get("run_label"), "slot": src.get("slot"),
             "training_checkpoint": src.get("training_checkpoint"), "source_checkpoint": src.get("source_checkpoint"),
             "selected_in": "round9", "pool_sha256": src["sha256"], "training_sha256": tsha,
             "parquet_sha256": src["parquet_sha256"], "strict_episode": False, "source_round": THIS_ROUND,
             "export_status": None, "runs_base": str(P["runs"]), "evals_csv": str(P["evals"])}  # fmt: skip
        jobs[job] = d
        r = man_rows.get(job)
        if r is None or r.get("status") != "ok":
            d["export_status"] = f"job not built: {None if r is None else r.get('status')}"
            continue
        if r.get("role") != role or r.get("pool_sha256") != src["sha256"]:
            raise SystemExit(f"{job}: manifest row role / pool sha256 != POOL_R9.json")
        if r.get("root_kind") != kind or r.get("training_sha256") != tsha or r.get("parquet_sha256") != src["parquet_sha256"]:
            raise SystemExit(f"{job}: manifest row (input / checkpoint / parquet) != its pool record")
        if prole == "screen":
            p = EXPORTS / job / "export.json"
            if not p.is_file():
                d["export_status"] = "missing"
                continue
            st, want, f = R7C._export_from_record(p, None, None, tsha, EXPORTS / job, job)
        elif prole == "reuse8":
            rp = Path(src["r8_export_record"])
            st, want, f = R7C._export_from_record(rp, src["r8_export_record_sha256"], src["r8_export_sha256"], tsha,
                                                  rp.parent, job)  # fmt: skip
        else:
            rp = Path(src["r7_export_record"])
            st, want, f = R7C._export_from_record(rp, src["r7_export_record_sha256"], src["r7_export_sha256"], tsha,
                                                  rp.parent, job)  # fmt: skip
        if st != "ok":
            d["export_status"] = st
            continue
        if role == "finalist" and fin_by[job].get("export_sha256") != want:
            raise SystemExit(f"{job}: export != final_r9.json")
        d.update(export_status="ok", export_sha256=want, export_file=f)
        ref, fref, rec_h, h = R7C.input_ref(kind, e, trk, r1trk)
        if rec_h != h or (kind != "raw" and src.get("frame_ref_sha256") != h):
            raise SystemExit(f"{job}: input reference {fref} sha256 {h[:16]} != its frozen record")
        d.update(input_reference=str(ref), input_reference_file=str(fref), input_reference_sha256=h)
    return jobs


def scored_inputs9(ph: str):
    """Shared by both selections: verified round-9 scores, gate, jobs, runs (round 7's scored_inputs on round 9's files)."""
    F = R7C.files_of(ph)
    js, g = C.read_json(F["jstats"]), C.read_json(F["gate_json"])
    if js.get("scores_csv_sha256") != sha(F["scores_csv"]) or js.get("script_sha256") != sha(MINE) or \
            js.get("gate_json_sha256") != sha(F["gate_json"]) or not js.get("integrity_kit_stock_unchanged") or \
            js.get("validation_failed_jobs") != (g.get("validation_failed_jobs") or []) or \
            js.get("rollouts_list_sha256") != sha(R7C.PH[ph]["rollouts"]):  # fmt: skip
        raise SystemExit("round-9 scores changed since scoring, scored by another script, or integrity flagged")
    jobs = load_jobs9(ph)
    evals, _, lp = R7C.load_evals(ph, jobs)
    rp = R7C.check_rollouts(ph, evals) + lp
    if rp:
        raise SystemExit("rollouts / runs changed:\n  " + "\n  ".join(rp[:20]))
    scores = {(r["job"], int(r["run"])): r for r in C.read_csv(F["scores_csv"]) if r["status"] == "ok"}
    flagged = {(j, int(k)) for j, k in g.get("flagged_run_keys") or []}
    return js, g, jobs, evals, scores, flagged


# ================================================================================================ check / gate / score
def cmd_check(a) -> int:
    ph = PHASES[a.phase]
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"r9_select CHECK phase {ph} {C.now()}: no metric is computed")
    fatal, info = chain9(ph, "check")
    for x in info:
        p(f"chain: {x}")
    for x in fatal:
        p(f"chain FAILED: {x}")
    jobs = load_jobs9(ph)
    evals, n_not_ok, problems = R7C.load_evals(ph, jobs)
    nr = R7C.PH[ph]["nruns"]
    full = [j for j in sorted(jobs) if sorted(evals.get(j, {})) == list(range(1, nr + 1))]
    p(f"phase {ph} jobs {len(jobs)}; export ok {sum(j['export_status'] == 'ok' for j in jobs.values())}; complete ({nr} ok "
      f"runs) {len(full)}; ok runs {sum(len(v) for v in evals.values())}; non-ok rows {n_not_ok}")
    for j in sorted(jobs):
        if jobs[j]["export_status"] != "ok":
            p(f"NO EXPORT {j}: {jobs[j]['export_status']}")
        elif j not in full:
            p(f"INCOMPLETE {j}: ok runs {sorted(evals.get(j, {}))} of {nr}")
    files = R7C.rollout_files(evals)
    problems += [f"missing rollout {f}" for f in files if not Path(f).is_file()]
    R = R7C.PH[ph]["rollouts"]
    if not R.exists() and not fatal and not problems:
        write_new(R, "".join(f"{sha(f)}  {f}\n" for f in sorted(files)))
        p(f"wrote {rel_name(R)}: {len(files)} rollout parquets, sha256 {sha(R)}")
    elif R.exists():
        rp = R7C.check_rollouts(ph, evals)
        problems += rp
        p(f"{rel_name(R)} exists (sha256 {sha(R)}): {len(files) - len(rp)}/{len(files)} unchanged")
    for x in problems:
        p(f"problem: {x}")
    ok = not fatal and not problems
    p(f"check: {'OK' if ok else 'PROBLEMS'}")
    write_new(ROOT / "dryrun" / f"r9_select_check_{ph}_{C.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


GATE_COLS = R8S.GATE_COLS


def cmd_gate(a) -> int:
    """PREREG_R7 5.1 own-input gate (round 8's cmd_gate loop, unchanged functions) on round 9's runs."""
    ph = PHASES[a.phase]
    if not require(ph, "gate"):
        return 2
    F = R7C.files_of(ph)
    if F["gate_csv"].exists() or F["gate_json"].exists():
        g = C.read_json(F["gate_json"]) if F["gate_json"].is_file() else {}
        if F["gate_csv"].is_file() and g.get("phase") == ph and g.get("script_sha256") == sha(MINE) and \
                g.get("csv_sha256") == sha(F["gate_csv"]) and g.get("rollouts_list_sha256") == sha(R7C.PH[ph]["rollouts"]):  # fmt: skip
            print(f"gate output of phase {ph} exists, written by this script (pass {g.get('pass')}): not repeated "
                  "(PREREG_R9 7.1)")  # fmt: skip
            return 0 if g.get("pass") is True else 1
        print("FATAL: gate output exists (new files only)")
        return 2
    jobs = load_jobs9(ph)
    evals, _, lp = R7C.load_evals(ph, jobs)
    rp = R7C.check_rollouts(ph, evals) + lp
    if rp:
        print("FATAL: rollouts / runs differ:\n  " + "\n  ".join(rp[:20]))
        return 2
    before = C.integrity()
    K = C.load_kit() if evals else None
    cache: dict = {}
    grids: dict = {}
    rows = []
    for job in sorted(evals):
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in sorted(evals[job]):
            rec = evals[job][k]
            row = {"job": job, "episode": e, "run": k, "try": rec["try"], "input": j["input"], "id": j["role"]}
            try:
                conv, inref, inf = C.rollout_on_grid(K, j, rec, frames, names)
                m0 = C.metrics_vs(K, conv, inref, names)
                aucs = [m0["auc"]] + [G.auc_seed(K, cache, conv, inref, names, s) for s in G.SEEDS[1:]]
            except ValueError as err:
                row.update({"status": "failed", "problem": str(err)})
                row["flags"] = "; ".join(R7C.run_flags(row))
                rows.append(row)
                continue
            ev = {m: rec[m] for m in C.METRICS}
            lo, hi = min(aucs), max(aucs)
            row.update({
                "status": "ok", "eval_auc": ev["auc"], **{f"auc_s{s}": x for s, x in zip(G.SEEDS, aucs)},
                "seed_min": lo, "seed_max": hi, "seed_range": hi - lo,
                "excess_outside_range": max(lo - ev["auc"], ev["auc"] - hi, 0.0),
                "band_ok": bool(lo - G.BAND <= ev["auc"] <= hi + G.BAND), "inside_range": bool(lo <= ev["auc"] <= hi),
                "d_auc_seed0": m0["auc"] - ev["auc"], "eval_mppe": ev["mppe"], "mine_mppe": m0["mppe"],
                "d_mppe": m0["mppe"] - ev["mppe"], "eval_sp_sr": ev["sp_sr"], "mine_sp_sr": m0["sp_sr"],
                "eval_mp_sr": ev["mp_sr"], "mine_mp_sr": m0["mp_sr"],
                "conv_pos_residual_m": inf["conv_pos_residual_m"], "conv_ang_residual_rad": inf["conv_ang_residual_rad"],
            })  # fmt: skip
            row["flags"] = "; ".join(R7C.run_flags(row))
            rows.append(row)
        print(f"  {job}: {sum(1 for r in rows if r['job'] == job and r['status'] == 'ok')}/{len(evals[job])} scored against "
              f"its own input; flagged {sum(1 for r in rows if r['job'] == job and r['flags'])}", flush=True)  # fmt: skip
    after = C.integrity()
    diff = C.integrity_diff(before, after) + C.integrity_diff(json.loads(BASELINE.read_text()), after)
    flagged_jobs = sorted({r["job"] for r in rows if r["flags"]})
    ok = [r for r in rows if r["status"] == "ok"]
    passed = not diff
    write_new(F["gate_csv"], csv_text(rows, GATE_COLS))
    out = {
        "written_at": C.now(), "phase": ph, "script_sha256": sha(MINE), "r8_select_sha256": R8S_SHA,
        "prereg_r9_sha256": PREREG_SHA, "rule": "PREREG_R9 5.1 = PREREG_R7 5.1: a run is flagged iff the evaluator AUC is "
        "outside [min_s AUC_s - 0.02, max_s AUC_s + 0.02] (s = 0..9), |dMPPE| > 0.5 cm, SP-SR or MP-SR differs from the "
        "evaluator, or it is a problem run. Phase S: a candidate with a flagged run is validation_failed. Phase F: a "
        "flagged finalist run disqualifies the finalist. A kit / stock-runtime change fails the gate.",
        "thresholds": G.GATE, "pass": passed, "validation_failed_jobs": flagged_jobs if (passed and ph == "S9") else [],
        "flagged_jobs": flagged_jobs, "flagged_run_keys": [[r["job"], r["run"]] for r in rows if r["flags"]],
        "n_jobs": len(evals), "n_runs": len(rows), "n_scored": len(ok), "n_flagged_runs": sum(1 for r in rows if r["flags"]),
        "flagged_runs": [f"{r['job']} r{r['run']}: {r['flags']}" for r in rows if r["flags"]],
        "max_excess_outside_range": max((r["excess_outside_range"] for r in ok), default=None),
        "max_abs_d_mppe": max((abs(r["d_mppe"]) for r in ok), default=None),
        "kit_stock_unchanged": not diff, "integrity_diff": diff[:50],
        "rollouts_list_sha256": sha(R7C.PH[ph]["rollouts"]), "csv_sha256": sha(F["gate_csv"]),
        "note": "own-input check only (each run vs its OWN input track); nothing is selected",
    }  # fmt: skip
    write_new(F["gate_json"], C.dump(out))
    print(json.dumps({k: out[k] for k in ("pass", "n_jobs", "n_runs", "n_scored", "n_flagged_runs", "flagged_jobs",
                                          "kit_stock_unchanged")}, indent=1))  # fmt: skip
    return 0 if passed else 1


def cmd_score(a) -> int:
    """PREREG_R7 5.1 consensus-B scores (round 8's cmd_score loop, unchanged functions) on round 9's runs."""
    ph = PHASES[a.phase]
    if not require(ph, "score"):
        return 2
    F = R7C.files_of(ph)
    if F["scores_csv"].exists() or F["jstats"].exists():
        js = C.read_json(F["jstats"]) if F["jstats"].is_file() else {}
        if F["scores_csv"].is_file() and js.get("phase") == ph and js.get("script_sha256") == sha(MINE) and \
                js.get("scores_csv_sha256") == sha(F["scores_csv"]) and js.get("gate_json_sha256") == sha(F["gate_json"]) and \
                js.get("rollouts_list_sha256") == sha(R7C.PH[ph]["rollouts"]):  # fmt: skip
            print(f"score output of phase {ph} exists, written by this script from the current gate (kit / stock unchanged "
                  f"{js.get('integrity_kit_stock_unchanged')}): not repeated (PREREG_R9 7.1)")  # fmt: skip
            return 0 if js.get("integrity_kit_stock_unchanged") else 1
        print("FATAL: score output exists (scores are computed once)")
        return 2
    g = C.read_json(F["gate_json"])
    excluded = list(g.get("validation_failed_jobs") or [])
    jobs = load_jobs9(ph)
    evals, _, lp = R7C.load_evals(ph, jobs)
    rp = R7C.check_rollouts(ph, evals) + lp
    if rp:
        print("FATAL: rollouts / runs differ:\n  " + "\n  ".join(rp[:20]))
        return 2
    roll = C.hash_list(R7C.PH[ph]["rollouts"])
    before = C.integrity()
    scored_jobs = [j for j in sorted(evals) if j not in excluded]
    K = C.load_kit() if scored_jobs else None
    refs = C.load_refs(K) if scored_jobs else {}
    rows, problems, grids = [], [], {}
    for job in scored_jobs:
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in sorted(evals[job]):
            rec = evals[job][k]
            base = {"job": job, "episode": e, "run": k, "try": rec["try"], "variant": R7C.VARIANT, "input": j["input"],
                    "id": j["role"], "checkpoint_sha256": j["export_sha256"], "parquet": rec["parquet"],
                    "parquet_sha256": roll[rec["parquet"]]}  # fmt: skip
            try:
                conv, _, inf = C.rollout_on_grid(K, j, rec, frames, names)
            except ValueError as err:
                st = "conversion_failed" if "rigid transform" in str(err) else "failed"
                problems.append(f"{job} run {k}: {err}")
                rows.append({**base, "status": st, "problem": str(err)})
                continue
            m = C.metrics_vs(K, conv, refs[R7C.VARIANT][e], names)
            rows.append({**base, "status": "ok", **m, "conv_pos_residual_m": inf["conv_pos_residual_m"],
                         "conv_ang_residual_rad": inf["conv_ang_residual_rad"]})  # fmt: skip
        print(f"  {job}: {sum(1 for r in rows if r['job'] == job and r['status'] == 'ok')} runs scored against refs/B", flush=True)
    write_new(F["scores_csv"], csv_text(rows, R7C.SCORE_COLS))
    jstats = {}
    for job in scored_jobs:
        rs = sorted([r for r in rows if r["job"] == job and r["status"] == "ok"], key=lambda r: r["run"])
        d = {"episode": jobs[job]["episode"], "input": jobs[job]["input"], "role": jobs[job]["role"], "n": len(rs),
             "runs": [r["run"] for r in rs]}  # fmt: skip
        for m in C.METRICS:
            d[m] = [r[m] for r in rs]
            if rs:
                d[m + "_stats"] = C.stats(d[m])
        jstats[job] = d
    diff = C.integrity_diff(before, C.integrity())
    out = {"written_at": C.now(), "phase": ph, "script_sha256": sha(MINE), "variant": R7C.VARIANT, "prereg_r9_sha256": PREREG_SHA,
           "gate_json_sha256": sha(F["gate_json"]), "refs_build_sha256": sha(R7C.R1 / "refs/build.json"),
           "rollouts_list_sha256": sha(R7C.PH[ph]["rollouts"]), "scores_csv_sha256": sha(F["scores_csv"]),
           "validation_failed_jobs": excluded, "scored_jobs": scored_jobs, "n_rows": len(rows),
           "n_ok_rows": sum(r["status"] == "ok" for r in rows), "problems": problems,
           "integrity_kit_stock_unchanged": not diff, "integrity_diff": diff[:50], "jobs": jstats}  # fmt: skip
    write_new(F["jstats"], C.dump(out))
    print(f"phase {ph}: scored {len(scored_jobs)} jobs against refs/{R7C.VARIANT}: {out['n_ok_rows']}/{len(rows)} ok rows; "
          f"problems {len(problems)}; kit/stock unchanged {not diff}")
    return 0 if not diff else 1


# ================================================================================================ select S
def frozen_hashes(names: list[str]) -> str:
    return "".join(f"{sha(pin_path(n))}  {n}\n" for n in names)


def earlier_s_info(pool: dict, need: set[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """The recorded phase-S candidate records (PREREG_R9 2.7 / 5.3) of the reuse8 / reuse7 / barred jobs in `need`:
    round 8's via its own verified phase-S loaders, round 7's via round 8's r7_phase_s (round 7's loaders)."""
    rec = load_r8records()["phase_s_candidates"]
    want8 = {c for c in need if rec.get(c, {}).get("round") == "r8"}
    want7 = {c for c in need if rec.get(c, {}).get("round") == "r7"}
    info, src = {}, {}
    if want8:
        s8 = r8_phase_s()
        for c in want8:
            if c not in s8:
                raise SystemExit(f"{c}: no round-8 phase-S record")
            info[c], src[c] = s8[c]["info"], {**s8[c]["job"], "source_label": "round-8 phase S (reused)"}
    if want7:
        s7 = R8S.r7_phase_s()
        for c in want7:
            if c not in s7:
                raise SystemExit(f"{c}: no round-7 phase-S record")
            info[c], src[c] = s7[c]["info"], {**s7[c]["job"], "source_label": "round-7 phase S (reused)"}
    missing = need - set(info)
    if missing:
        raise SystemExit(f"no recorded phase-S results for {sorted(missing)}")
    return info, src


def cmd_select_s() -> int:
    if not require("S9", "select"):
        return 2
    js, g, jobs, evals, scores, flagged = scored_inputs9("S9")
    excluded = set(g.get("validation_failed_jobs") or [])
    jst = js["jobs"]
    incs = incumbents9()
    pool = pool_rows()
    srec = load_r8records()["phase_s_candidates"]
    reuse_all = {j for j, p in pool.items() if p["role"] in ("reuse8", "reuse7")}
    barred_all = {j for j, p in pool.items() if p["role"] == "barred" and j in srec}
    for j in reuse_all:
        want = "r8" if pool[j]["role"] == "reuse8" else "r7"
        if srec.get(j, {}).get("round") != want:
            raise SystemExit(f"{j}: pool role {pool[j]['role']} without a round-{want[1]} phase-S record")
    einfo, esrc = earlier_s_info(pool, reuse_all | barred_all)
    eps, longrows, lines, finalists = [], [], [], []
    for e in range(30):
        I = incs[e]
        new = sorted(j for j in jobs if jobs[j]["episode"] == e)
        reuse = sorted(j for j in reuse_all if int(pool[j]["episode"]) == e)
        barred = sorted(j for j in barred_all if int(pool[j]["episode"]) == e)
        info = {c: s_info(c, jobs, evals, excluded, flagged, jst) for c in new}
        for c in reuse + barred:
            if int(esrc[c]["episode"]) != e:
                raise SystemExit(f"{c}: recorded episode != pool episode")
            info[c] = einfo[c]
        evaluated = bool(new or reuse)
        if evaluated:
            f, rows = select_episode_s9(I["m_P_frozen"], new + reuse + barred, info, set(barred))
        else:
            f, rows = None, []
        for r in rows:
            c = r["job"]
            src = jobs[c] if c in jobs else esrc[c]
            r.update(source="round-9 phase S" if c in jobs else esrc[c]["source_label"], input=src["input"],
                     owner=src["owner"], machine=src.get("machine"), pool_role=pool[c]["role"], iteration=src.get("iteration"),
                     export_status=src.get("export_status"),
                     own_auc=[info[c]["own"][k]["auc"] for k in sorted(info[c]["own"])])  # fmt: skip
        rows.sort(key=lambda r: (-(r.get("cons_mean") if r.get("cons_mean") is not None else -1.0), r["job"]))
        rec = {"episode": e, "incumbent": I["job"], "incumbent_input": I["input"], "incumbent_status": I["r8_status"],
               "m_P_frozen": I["m_P_frozen"], "bar_s": I["m_P_frozen"] + MARGIN, "evaluated": evaluated,
               "n_new_screen": len(new), "n_reuse": len(reuse), "barred": barred, "finalist": f,
               "n_eligible": sum(r["eligible"] for r in rows), "n_qualifying": sum(r["qualifies"] for r in rows),
               "candidates": rows}  # fmt: skip
        eps.append(rec)
        if f:
            src = jobs[f] if f in jobs else esrc[f]
            fr = next(r for r in rows if r["job"] == f)
            p = pool[f]
            finalists.append({"job": f, "episode": e, "pool_role": p["role"], "input": src["input"], "owner": src["owner"],
                              "machine": src.get("machine"), "tree": src.get("tree"), "iteration": src.get("iteration"),
                              "pool_sha256": p["sha256"], "training_sha256": src["training_sha256"],
                              "export_sha256": src["export_sha256"], "export_file": src["export_file"],
                              "cons_S_mean": fr["cons_mean"], "cons_S_var": fr["cons_var"],
                              "cons_S_mppe_mean": fr["cons_mppe_mean"], "floor_S": fr["floor_count"],
                              "incumbent": I["job"], "m_P_frozen": I["m_P_frozen"], "bar_s": I["m_P_frozen"] + MARGIN})  # fmt: skip
        for r in rows:
            longrows.append({"episode": ep_key(e), "job": r["job"], "source": r["source"], "pool_role": r["pool_role"],
                             "input": r["input"], "owner": r["owner"], "machine": r["machine"], "iteration": r["iteration"],
                             "export_status": r["export_status"], "n_own_ok": r["n_own_ok"], "floor_count": r["floor_count"],
                             "flagged": r["flagged"], "cons_mean": r.get("cons_mean"),
                             "cons_sd": math.sqrt(r["cons_var"]) if "cons_var" in r else None,
                             "cons_mppe_mean": r.get("cons_mppe_mean"), "incumbent": I["job"], "m_P_frozen": I["m_P_frozen"],
                             "delta_vs_m_P": r.get("delta_vs_m_P"), "eligible": r["eligible"], "barred": r.get("barred", False),
                             "qualifies": r["qualifies"], "finalist": r["job"] == f, "why_not": "; ".join(r["why_not"])})  # fmt: skip
        lines.append(f"ep{e:02d} P {I['job']} [{I['input']}] m_P_frozen {I['m_P_frozen']:.6f} (bar_S {I['m_P_frozen'] + MARGIN:.6f}); "
                     + (f"{len(new)} new + {len(reuse)} reused + {len(barred)} barred candidates, {rec['n_eligible']} eligible, "
                        f"{rec['n_qualifying']} qualifying -> finalist {f or 'none'}" if evaluated
                        else "no round-9 candidate: keeps its round-8 pick"))  # fmt: skip
        for r in rows[:3]:
            if r.get("cons_mean") is not None:
                lines.append(f"      {r['job']} [{r['input']}] cons_S {r['cons_mean']:.6f} floor {r['floor_count']}/3 "
                             f"eligible={r['eligible']} qualifies={r['qualifies']} {r['why_not']}")  # fmt: skip
    gen = C.now()
    hashes = {"prereg_r9_sha256": PREREG_SHA, "script_sha256": sha(MINE), "selftest_sha256": sha(SELFTEST),
              "pool_r9_sha256": sha(POOL), "r8_records_sha256": sha(R8REC), "manifest_s9_sha256": sha(R7C.PH["S9"]["manifest"]),
              "rollouts_s9_sha256": sha(R7C.PH["S9"]["rollouts"]), "gate_json_sha256": sha(R7C.files_of("S9")["gate_json"]),
              "scores_csv_sha256": sha(R7C.files_of("S9")["scores_csv"]), "jstats_sha256": sha(R7C.files_of("S9")["jstats"])}  # fmt: skip
    sel = {"generated_at": gen, "phase": "S", "basis": "own-input and consensus-B scores only", **hashes, "rule": RULE_S9,
           "validation_failed_jobs": sorted(excluded), "n_finalists": len(finalists),
           "finalist_episodes": sorted({ep_key(x["episode"]) for x in finalists}),
           "evaluated_episodes": [ep_key(x["episode"]) for x in eps if x["evaluated"]], "episodes": eps}  # fmt: skip
    final = {"generated_at": gen, "prereg_r9_sha256": PREREG_SHA, "script_sha256": hashes["script_sha256"], "rule": RULE_S9,
             "finalists": finalists, "no_finalist_episodes": [ep_key(x["episode"]) for x in eps if not x["finalist"]],
             "note": "phase-F job list (PREREG_R9 3.4): these finalists only, 10 fresh runs each; the incumbents are not "
                     "re-run (their recorded phase-F runs are used); locked by SELECTION_S9.sha256"}  # fmt: skip
    written = []

    def w(name: str, text: str) -> None:
        write_new(ROOT / name, text)
        written.append(name)

    w("selection_s9.json", C.dump(sel))
    w("final_r9.json", C.dump(final))
    w("selection_s9.txt", "\n".join([f"selection_s9 {gen} CST (PREREG_R9 5.3 / 5.4; own-input gate + consensus-B vs round "
                                     f"1's frozen refs/B; finalists {len(finalists)})"] + lines + ["", "finalists: " + (
        ", ".join(f"{x['episode']:02d}:{x['job']} ({x['cons_S_mean']:.4f} vs bar_S {x['bar_s']:.4f})" for x in finalists)
        or "none")]) + "\n")  # fmt: skip
    cols = ["episode", "job", "source", "pool_role", "input", "owner", "machine", "iteration", "export_status", "n_own_ok",
            "floor_count", "flagged", "cons_mean", "cons_sd", "cons_mppe_mean", "incumbent", "m_P_frozen", "delta_vs_m_P",
            "eligible", "barred", "qualifies", "finalist", "why_not"]  # fmt: skip
    w("selection_s9_candidates.csv", csv_text(longrows, cols))
    md = ["# t2_consensus_r9_20261103: phase-S selection (PREREG_R9 5.3 / 5.4)", "",
          f"Generated {gen} CST by `r9_select.py` ({hashes['script_sha256'][:16]}...). Selection uses only own-input and "
          "consensus-B scores. Every episode's incumbent is its round-8 pick; finalist = the S-eligible candidate with the "
          "highest 3-run consensus-B mean, provided it is >= m_P_frozen + 0.01.", "",
          "| ep | incumbent (input) | m_P_frozen | bar_S | new / reused / barred | eligible / qualifying | best: cons_S (floor) | "
          "**finalist** |", "|---|---|---:|---:|---|---|---|---|"]  # fmt: skip
    for x in eps:
        best = next((r for r in x["candidates"] if r.get("cons_mean") is not None), None)
        md.append(f"| {x['episode']:02d} | {x['incumbent']} ({x['incumbent_input']}) | {fmt(x['m_P_frozen'])} | {fmt(x['bar_s'])} | "
                  f"{x['n_new_screen']} / {x['n_reuse']} / {len(x['barred'])} | {x['n_eligible']} / {x['n_qualifying']} | "
                  + (f"{best['job']}: {best['cons_mean']:.4f} ({best['floor_count']}/3)" if best else "")
                  + f" | **{x['finalist'] or ('-' if x['evaluated'] else 'not evaluated')}** |")  # fmt: skip
    w("selection_s9.md", "\n".join(md) + "\n")
    frozen_inputs = [rel_name(p) for p in (R7C.files_of("S9")["scores_csv"], R7C.files_of("S9")["jstats"],
                                           R7C.files_of("S9")["gate_csv"], R7C.files_of("S9")["gate_json"], SELFTEST,
                                           R7C.PH["S9"]["rollouts"], R7C.PH["S9"]["evals"], R7C.PH["S9"]["manifest"],
                                           POOL, POOL_LIST, R8REC, PINS, LOCK, PREREG, MINE)]  # fmt: skip
    write_new(SEL_S9, frozen_hashes(written + frozen_inputs))  # LAST: the phase-S selection is frozen
    print("\n".join(md[4:]))
    print(f"\nSELECTION_S9.sha256 written {C.now()} (frozen): {sha(SEL_S9)}")
    return 0


# ================================================================================================ select F
def rollout_lists() -> dict[str, dict]:
    out = {"r7F": C.hash_list(R7ROOT / "inputs_r7/ROLLOUTS_F7.sha256"), "r8F": C.hash_list(R8ROOT / "inputs_r8/ROLLOUTS_F8.sha256")}
    out["r9F"] = C.hash_list(R7C.PH["F9"]["rollouts"]) if R7C.PH["F9"]["rollouts"].is_file() else {}
    return out


def finish_f(eps: list[dict], comp: dict[int, str], runinfo: dict, extra: dict, reason: str | None) -> int:
    incs = incumbents9()
    rows9, t_star = closest9(comp, runinfo)
    file9 = {}
    if t_star is not None:
        rl = rollout_lists()
        for e in range(30):
            key = comp[e]
            src, job = key.split(":", 1)
            r = runinfo[key][t_star]
            x = next(y for y in eps if y["episode"] == e)
            if src in ("r7F", "r8F"):
                I = incs[e]
                file9[ep_key(e)] = {"job": job, "run": t_star, "try": r["try"],
                                    "source": f"{'round-7' if src == 'r7F' else 'round-8'} phase F (round-8 pick kept)",
                                    "input": I["input"], "owner": I["owner"], "machine": I["machine"],
                                    "frame_reference": I["input_reference"], "export_sha256": I["export_sha256"],
                                    "export_file": I["export_file"], "rollout": r["parquet"],
                                    "rollout_sha256": rl[src][r["parquet"]]}  # fmt: skip
            else:
                file9[ep_key(e)] = {"job": job, "run": t_star, "try": r["try"], "source": "round-9 phase F (replacement)",
                                    "input": x["pick_input"], "owner": x["pick_owner"], "machine": x["pick_machine"],
                                    "frame_reference": x["pick_input_reference"], "export_sha256": x["pick_export_sha256"],
                                    "export_file": x["pick_export_file"], "rollout": r["parquet"],
                                    "rollout_sha256": rl["r9F"][r["parquet"]]}  # fmt: skip
    replaced = [ep_key(x["episode"]) for x in eps if x["changed"]]
    if not replaced and t_star != R8_CLOSEST:
        raise SystemExit(f"no episode changed but the closest trial is {t_star}, not round 8's trial {R8_CLOSEST}")
    gen = C.now()
    sel = {"generated_at": gen, "phase": "F", "basis": "own-input and consensus-B scores only", "prereg_r9_sha256": PREREG_SHA,
           "script_sha256": sha(MINE), "r8_records_sha256": sha(R8REC), "rule": RULE_F9, "keep_r8_reason": reason,
           "replaced_episodes": replaced, "episodes": eps,
           "compositions": {"S9": {ep_key(e): comp[e].split(":", 1)[1] for e in range(30)},
                            "S9_run_source": {ep_key(e): comp[e].split(":", 1)[0] for e in range(30)},
                            "S8_round8": {ep_key(e): incs[e]["job"] for e in range(30)}},
           "trials": rows9, "r9_closest_trial": t_star, "r9_closest_file": file9, **extra}  # fmt: skip
    written = []

    def w(name: str, text: str) -> None:
        write_new(ROOT / name, text)
        written.append(name)

    w("SELECTION_R9.json", C.dump(sel))
    table = []
    for x in eps:
        table.append({"episode": ep_key(x["episode"]), "incumbent_r8_pick": x["incumbent"], "incumbent_input": x["incumbent_input"],
                      "m_P_frozen": x["m_P_frozen"], "m_P_fresh": x.get("m_P_fresh"), "bar": x.get("bar"),
                      "evaluated": x.get("evaluated"), "finalist": x.get("finalist") or "", "finalist_input": x.get("finalist_input") or "",
                      "cons_S_mean": x.get("finalist_cons_S_mean"), "m_F": x.get("m_F"), "finalist_floor": x.get("floor_count"),
                      "changed": x["changed"], "pick": x["pick"], "pick_status": x["pick_status"],
                      "why_not": "; ".join(x.get("why_not") or [])})  # fmt: skip
    cols = list(table[0].keys())
    w("selection_r9_episodes.csv", csv_text(table, cols))
    tcols = ["trial", "complete", "missing_episodes", "own_auc", "own_sp", "own_mp", "own_mppe", "cons_auc", "cons_sp",
             "cons_mp", "cons_mppe"]  # fmt: skip
    w("summary_r9_trials.csv", csv_text([{**r, "missing_episodes": " ".join(map(str, r["missing_episodes"]))} for r in rows9], tcols))
    head = (f"# t2_consensus_r9_20261103: round-9 decision (PREREG_R9 5.5 / 5.6)\n\nGenerated {gen} CST by `r9_select.py` "
            f"({sha(MINE)[:16]}...). Selection uses only own-input and consensus-B scores. Incumbent = the round-8 pick (its "
            "recorded phase-F runs, never re-run).")  # fmt: skip
    md = [head, "", (f"**Kept round-8 picks everywhere: {reason}**" if reason else
                     f"Replaced round-8 picks: {', '.join(replaced) or 'none'}. R9-CLOSEST trial: {t_star}."), "",
          "| ep | round-8 pick (input) | m_P_frozen | m_P_fresh | bar | finalist | m_F | floor | **pick** |",
          "|---|---|---:|---:|---:|---|---:|---:|---|"]  # fmt: skip
    for t in table:
        md.append(f"| {t['episode']} | {t['incumbent_r8_pick']} ({t['incumbent_input']}) | {fmt(t['m_P_frozen'])} | "
                  f"{fmt(t['m_P_fresh'])} | {fmt(t['bar'])} | {t['finalist'] or ('-' if t['evaluated'] else 'not evaluated')} | "
                  f"{fmt(t['m_F'])} | {t['finalist_floor'] if t['finalist'] else ''} | **{t['pick']}** ({t['pick_status']}) |")  # fmt: skip
    md += ["", "| trial | complete | own AUC | own SP-SR | own MP-SR | own MPPE | cons-B AUC |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows9:
        md.append(f"| {r['trial']} | {r['complete']} | {fmt(r.get('own_auc'))} | {fmt(r.get('own_sp'), '.3f')} | "
                  f"{fmt(r.get('own_mp'), '.3f')} | {fmt(r.get('own_mppe'), '.3f')} | {fmt(r.get('cons_auc'))} |")
    md += ["", f"R9-CLOSEST = trial {t_star} of S9 (closest-to-mean; never a best-of). No file is packed by this script."]
    w("selection_r9.md", "\n".join(md) + "\n")
    frozen = [rel_name(p) for p in (POOL, POOL_LIST, R8REC, SELFTEST, PINS, LOCK, PREREG, MINE)]
    for p in (FINAL, SEL_S9, ROOT / "selection_s9.json", R7C.PH["F9"]["rollouts"], R7C.PH["F9"]["evals"], R7C.PH["F9"]["manifest"],
              R7C.files_of("F9")["scores_csv"], R7C.files_of("F9")["jstats"], R7C.files_of("F9")["gate_csv"],
              R7C.files_of("F9")["gate_json"], DEADLINE_F):  # fmt: skip
        if Path(p).is_file():
            frozen.append(rel_name(p))
    write_new(SEL_R9, frozen_hashes(written + frozen))  # LAST: round 9 is final
    print("\n".join(md))
    print(f"\nSELECTION_R9.sha256 written {C.now()} (frozen): {sha(SEL_R9)}")
    return 0


def base_runinfo(incs: dict[int, dict]) -> dict[str, dict]:
    return {f"{incs[e]['phase_f_source']}:{incs[e]['job']}": incs[e]["phase_f_runs"] for e in range(30)}


def cmd_select_f() -> int:
    if not SEL_S9.is_file():
        print("FATAL: SELECTION_S9.sha256 missing")
        return 2
    fin = C.read_json(FINAL)
    finalists = {int(x["episode"]): x for x in fin["finalists"]}
    if finalists:
        if not require("F9", "select"):
            return 2
        js, g, jobs, evals, scores, flagged = scored_inputs9("F9")
    else:  # no finalist: phase F has no job (PREREG_R9 3.4); the chain of the documents and of phase S must hold
        fatal, info = r9_docs(require_lock=True)
        for rel, h0 in hash_lines(SEL_S9):
            if pin_hash(pin_path(rel)) != h0:
                fatal.append(f"SELECTION_S9.sha256: {rel} changed")
        for x in info:
            print("  " + x)
        if fatal:
            print("FATAL:\n  " + "\n  ".join(fatal))
            return 2
        jobs, evals, scores, flagged = {}, {}, {}, set()
    sel_s = C.read_json(ROOT / "selection_s9.json")
    evaluated = {int(x["episode"]): x["evaluated"] for x in sel_s["episodes"]}
    incs = incumbents9()
    runinfo = base_runinfo(incs)
    eps, comp = [], {}
    for e in range(30):
        I = incs[e]
        f = finalists.get(e)
        fr = runinfo_of(f["job"], evals, scores, flagged) if f else None
        d = decide_episode_f9(I["m_P_frozen"], I["phase_f_runs"], f["job"] if f else None, fr)
        pick = f["job"] if d["changed"] else I["job"]
        rec = {"episode": e, "incumbent": I["job"], "incumbent_input": I["input"], "incumbent_status": I["r8_status"],
               "evaluated": evaluated.get(e, False), "finalist_input": f["input"] if f else None,
               "finalist_cons_S_mean": f["cons_S_mean"] if f else None, **d, "pick": pick,
               "pick_status": "replaced" if d["changed"] else "round-8 pick kept",
               "finalist_runs": {k: {x: v for x, v in r.items() if x != "parquet"} for k, r in (fr or {}).items()}}  # fmt: skip
        if d["changed"]:
            j = jobs[pick]
            runinfo[f"r9F:{pick}"] = fr
            comp[e] = f"r9F:{pick}"
            rec.update(pick_input=j["input"], pick_owner=j["owner"], pick_machine=j["machine"],
                       pick_input_reference=j["input_reference"], pick_export_sha256=j["export_sha256"],
                       pick_export_file=j["export_file"], pick_training_checkpoint=j["training_checkpoint"],
                       pick_source_checkpoint=j["source_checkpoint"], pick_iteration=j["iteration"])  # fmt: skip
        else:
            comp[e] = f"{I['phase_f_source']}:{I['job']}"
        eps.append(rec)
    extra = {"final_r9_sha256": sha(FINAL), "selection_s9_lock_sha256": sha(SEL_S9),
             "deadline_stop": C.read_json(DEADLINE_F) if DEADLINE_F.is_file() else None}  # fmt: skip
    return finish_f(eps, comp, runinfo, extra, None)


def cmd_keep_r8(a) -> int:
    """PREREG_R9 7.3: every episode keeps its round-8 pick (phase S not complete by its cut-off, or no usable GPU)."""
    if SEL_R9.exists():
        print("FATAL: SELECTION_R9.sha256 exists")
        return 2
    fatal, info = r9_docs(require_lock=True)
    run, live_pid = driver_running(), pidfile_live()
    if run or live_pid:  # PREREG_R9 7.6: never while an evaluation driver is alive
        fatal.append(f"an r9_eval driver is running: {run} {live_pid or ''}")
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return 2
    incs = incumbents9()
    runinfo = base_runinfo(incs)
    eps = [{"episode": e, "incumbent": incs[e]["job"], "incumbent_input": incs[e]["input"], "m_P_frozen": incs[e]["m_P_frozen"],
            "m_P_fresh": incs[e]["m_P_fresh"], "bar": incs[e]["bar"], "evaluated": False, "finalist": None, "changed": False,
            "pick": incs[e]["job"], "pick_status": "round-8 pick kept", "why_not": [a.reason]} for e in range(30)]  # fmt: skip
    comp = {e: f"{incs[e]['phase_f_source']}:{incs[e]['job']}" for e in range(30)}
    return finish_f(eps, comp, runinfo, {"selection_s9_lock_sha256": sha(SEL_S9) if SEL_S9.is_file() else None}, a.reason)


def cmd_select(a) -> int:
    if a.phase == "S":
        if SEL_S9.exists():
            print("FATAL: SELECTION_S9.sha256 exists; the phase-S selection is frozen")
            return 2
        return cmd_select_s()
    if SEL_R9.exists():
        print("FATAL: SELECTION_R9.sha256 exists; round 9 is frozen")
        return 2
    return cmd_select_f()


# ================================================================================================ selftest
def cmd_selftest(a) -> int:
    if SELFTEST.exists():
        print(f"FATAL: {SELFTEST} exists (new files only)")
        return 2
    res, problems = [], []

    def check(name, ok, detail=""):
        res.append({"test": name, "ok": bool(ok), "detail": str(detail)[:400]})
        if not ok:
            problems.append(f"{name}: {detail}")

    def own3(fl=3):
        return {k: {"sp_sr": 1.0 if k <= fl else 0.0, "mp_sr": 1.0 if k <= fl else 0.0, "auc": 0.5} for k in (1, 2, 3)}

    def cand(aucs, fl=3, flagged=False, export_ok=True, mppe=1.0):
        return {"export_ok": export_ok, "own": own3(fl), "flagged": flagged,
                "cons": {"runs": [1, 2, 3], "auc": list(aucs), "mppe": [mppe] * 3}}  # fmt: skip

    def fr(aucs, sp=None, flagged=(), missing=()):
        out = {}
        for k in range(1, 11):
            if k in missing:
                continue
            ok1 = sp is None or k <= sp
            out[k] = {"ok": True, "flagged": k in flagged, "try": 1, "parquet": f"/x/{k}", "own_auc": 0.5, "own_sp": 1.0 if ok1 else 0.0,
                      "own_mp": 1.0 if ok1 else 0.0, "own_mppe": 1.0, "cons_auc": aucs[k - 1], "cons_sp": 1.0, "cons_mp": 1.0,
                      "cons_mppe": 1.0}  # fmt: skip
        return out

    check("round-8 scorer pinned, round-7 scorer through it, constants: margin 0.01, phase-F floor 8/10, >= 8 fresh runs",
          sha(R8S_FILE) == R8S_SHA and sha(R7C.MINE) == R8S.R7C_SHA and (MARGIN, FLOOR_F, MIN_FRESH) == (0.01, 8, 8))
    info = {"a": cand([0.62, 0.62, 0.62]), "b": cand([0.70, 0.70, 0.70]), "c": cand([0.80, 0.80, 0.80]), "d": cand([0.9] * 3, fl=1)}
    f, rows = select_episode_s9(0.6, ["a", "b", "c", "d"], info, {"c"})
    check("select S: barred c (0.80) never finalist; d fails floor 1/3; b (0.70) is the finalist", f == "b", (f, rows))
    f, _ = select_episode_s9(0.6, ["a", "b"], info, set())
    check("select S: highest qualifying wins", f == "b", f)
    f, _ = select_episode_s9(0.695, ["a", "b"], info, set())
    check("select S: cons_S 0.70 < 0.695 + 0.01 -> no finalist", f is None, f)
    f, _ = select_episode_s9(0.69, ["a", "b"], info, set())
    m_b = C.stats([0.70, 0.70, 0.70])["mean"]
    check("select S: exact floats as round 7: finalist iff the float64 3-run mean >= m_P_frozen + 0.01",
          f == ("b" if m_b >= 0.69 + 0.01 else None), (f, repr(m_b), repr(0.69 + 0.01)))
    info2 = {"x": cand([0.8, 0.8, 0.8], flagged=True), "y": cand([0.8, 0.8, 0.8], export_ok=False)}
    f, _ = select_episode_s9(0.5, ["x", "y"], info2, set())
    check("select S: flagged / failed export never finalists", f is None, f)
    inc = fr([0.50] * 10)
    d = decide_episode_f9(0.49, inc, "f", fr([0.515] * 10))
    check("decide F: bar = max(m_P_frozen 0.49, recorded m_P_fresh 0.50) + 0.01 = 0.51; m_F 0.515 replaces",
          d["changed"] and abs(d["bar"] - 0.51) < 1e-12, d)
    d = decide_episode_f9(0.52, inc, "f", fr([0.525] * 10))
    check("decide F: m_P_frozen 0.52 dominates -> bar 0.53; m_F 0.525 does not replace", not d["changed"], d)
    d = decide_episode_f9(0.40, inc, "f", fr([0.60] * 10, missing=(9, 10)))
    check("decide F: deadline (8 of 10 runs) -> not replaced ((a) fails)", not d["changed"] and d["why_not"][0].startswith("(a)"), d)
    d = decide_episode_f9(0.40, inc, "f", fr([0.60] * 10, sp=7))
    check("decide F: own-input SP=MP=1 in 7/10 -> not replaced ((b))", not d["changed"], d)
    d = decide_episode_f9(0.40, inc, "f", fr([0.60] * 10, flagged=(3,)))
    check("decide F: a flagged run -> not replaced ((a))", not d["changed"], d)
    d = decide_episode_f9(0.40, inc, None, None)
    check("decide F: no finalist -> round-8 pick kept", not d["changed"] and d["why_not"] == ["no finalist (5.4)"], d)
    m_rec = C.stats([0.70 + 0.001 * k for k in range(10)])["mean"]
    d = decide_episode_f9(m_rec, fr([0.70 + 0.001 * k for k in range(10)]), None, None)
    check("round-8 replacement as incumbent: m_P_frozen = mean of its recorded runs -> bar = that mean + 0.01",
          abs(d["bar"] - (m_rec + 0.01)) < 1e-15 and d["m_P_fresh"] == m_rec, d)
    check("would_pass: the round-9 test on recorded runs (10 good, floor 8, mean >= bar)",
          would_pass(fr([0.6] * 10), 0.59) and not would_pass(fr([0.6] * 10), 0.61) and not would_pass(fr([0.6] * 10, sp=7), 0.5)
          and not would_pass(fr([0.6] * 10, missing=(10,)), 0.5))
    rA = fr([0.5] * 10)
    for k, v in rA.items():
        v["own_auc"] = 0.5 + 0.01 * k
    rB, rC = fr([0.6] * 10), fr([0.7] * 10)
    comp = {0: "r7F:A", 1: "r8F:B", 2: "r9F:C"}
    rows, t = closest9(comp, {"r7F:A": rA, "r8F:B": rB, "r9F:C": rC})
    mu = np.mean([(0.5 + 0.01 * k + 0.5 + 0.5) / 3 for k in range(1, 11)])
    want = min(range(1, 11), key=lambda k: (abs((0.5 + 0.01 * k + 1.0) / 3 - mu), k))
    check("closest trial on a round-7 / round-8 / round-9 composite: closest-to-mean own-input AUC, ties lower t", t == want, (t, want))
    rC2 = fr([0.7] * 10, missing=(want,))
    rows, t2 = closest9(comp, {"r7F:A": rA, "r8F:B": rB, "r9F:C": rC2})
    check("closest trial: a trial missing for one episode is not complete", t2 != want and not rows[want - 1]["complete"], (t2, want))
    passed = not problems
    out = {"written_at": C.now(), "script_sha256": sha(MINE), "r8_select_sha256": R8S_SHA, "pass": passed, "tests": res,
           "problems": problems}  # fmt: skip
    write_new(SELFTEST, C.dump(out))
    print(f"selftest: {sum(r['ok'] for r in res)}/{len(res)} ok -> {'PASS' if passed else 'FAIL'}")
    for x in problems:
        print("  " + x)
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("r8records", "selftest", "table", "check", "gate", "score", "select", "keep-r8"))
    ap.add_argument("--phase", choices=("S", "F"))
    ap.add_argument("--reason", default=None)
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    if a.cmd in ("check", "gate", "score", "select") and a.phase is None:
        ap.error("--phase S|F is required")
    if a.cmd == "keep-r8" and not a.reason:
        ap.error("--reason is required")
    return {"r8records": cmd_r8records, "selftest": cmd_selftest, "table": cmd_table, "check": cmd_check, "gate": cmd_gate,
            "score": cmd_score, "select": cmd_select, "keep-r8": cmd_keep_r8}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
