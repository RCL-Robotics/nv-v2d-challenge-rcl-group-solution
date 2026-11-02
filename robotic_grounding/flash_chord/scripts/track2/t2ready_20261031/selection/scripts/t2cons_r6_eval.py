#!/usr/bin/env python3
"""t2cons_r6_eval.py: round 6 of the Tier-2 consensus selection, steps R2/R3/R5 (PREREG_R6.md section 3).

This file is scripts/t2cons_r5_eval.py (68191fbb; round 5's driver, itself round 4's / 3's / 2's driver c) with ONLY the
round-6 bookkeeping changed (full text diff in R6_SCRIPT_DIFFS.txt):
  Q1. ROOT = fc_runs/submissions/t2_consensus_r6_20261031; outputs exports/ (one export per screen job), runs_s/ (phase
      S) and runs_f/ (phase F), manifest_{s6,f6}.{json,csv}, own_input_evals_{s6,f6}.csv,
      summary_{s6,f6}_own_input.{csv,json,md}, seeds_{s6,f6}.tsv, eval_r6_driver.log, eval_r6_heartbeat_{s6,f6}.json,
      eval_r6_driver_stdout_<mode>.log (launcher); ROOT/evallock is our lane-lock folder;
  Q2. two phases (PREREG_R6 3.2 / 3.3): --phase screen = the 250 'screen' rows of inputs_r6/pool_r6.json, 3 runs each;
      --phase final = the 30 incumbents of inputs_r6/incumbents_r6.json + the finalists of final_r6.json (which must be
      listed with its sha256 in SELECTION_S6.sha256), 10 runs (trials) each, trial-major order;
  Q3. designation evidence: a screen job = its pool_r6.json row (rule R1-R5), the file re-hashed (ours: the original in
      our run folder; Zuxing's: our copy, whose COPIES_r6.tsv row says source before == copy == after == the pool sha256);
      an incumbent = its earlier round's export record and exported file (re-hashed; NEVER re-exported) + manifest row;
      there is no relocation in round 6;
  Q4. inputs raw, K3, K3fix, K3CA and G2: the pinned driver's ROOTS / FRAME_REFS / MANIFEST_FOR gain "k3ca"
      (data/human_motion_data_tier2_k3ca, frame refs tmp_analysis/tier2/k3ca/frame_refs_k3ca = round 1's INPUT_REF) by
      rebinding at load time; the export tree is the owner's (fc_r2 ours, b39b69b his, PREREG_R6 3.1);
  Q5. pin chain: PREREG_R6.md == PREREG_R6.sha256 == hard-coded; its section-11 pins and commits; SNAPSHOT_R6 and
      TRACKS_R6 lists; round 1's own chain (pinned t2cons_eval check_pins); PREREG_R6_PINS.md (== its sha file, every pin
      holds) must pin this script and its launcher;
  Q6. GPUs 0, 1, 2, 7 only (PREREG_R6 3.4); SHARED_LOCKDIRS gains round 5's own evallock folder;
  Q7. runs per job: 3 (phase S) / 10 (phase F); dry-run jobs ep26_k3ca_25298 (phase S) and ep26_zx_k3fix_92000 (phase F,
      the incumbent's round-1 export reused) (PREREG_R6 3.6);
  Q8. docstring and log text; the path guard also refuses seed_best, remote_reps, report_r, composites, per_episode_zx.
Exports (1e-4 then one 2e-4 fallback), the evaluation command, every ok check, retries (3 tries, 5 min apart), the caps
(4 in total, 2 per NUMA eval lane), the 35 s cooldown, lane locks once per lock file (all or nothing, never blocking,
never created), never MPS, the seed records, OVERDUE flagging and never signalling are round 5's, unchanged. No clean
diagnostic and no clean path (guard on every record read).

  python -I -B scripts/t2cons_r6_eval.py run --phase screen|final --dry-run       plan only (dryrun/r6_eval_plan_*.txt)
  python -I -B scripts/t2cons_r6_eval.py run --phase screen --dry-run-eval        ONE evaluation of ep26_k3ca_25298
  python -I -B scripts/t2cons_r6_eval.py run --phase final --dry-run-eval         ONE evaluation of ep26_zx_k3fix_92000
  python -I -B scripts/t2cons_r6_eval.py run --phase screen --exports-only        every screen job's single export
  python -I -B scripts/t2cons_r6_eval.py run --phase screen|final                 the evaluations of a phase (resumable)
  python -I -B scripts/t2cons_r6_eval.py summary|status --phase screen|final
Background start: scripts/t2cons_r6_eval_launch.sh --phase screen|final [--dry-run | --dry-run-eval | --exports-only]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r6_20261031"
R1 = ME / "fc_runs/submissions/t2_consensus_20261024"
R2 = ME / "fc_runs/submissions/t2_consensus_r2_20261027"
R3 = ME / "fc_runs/submissions/t2_consensus_r3_20261028"
R4 = ME / "fc_runs/submissions/t2_consensus_r4_20261028"
R5 = ME / "fc_runs/submissions/t2_consensus_r5_20261029"
ZU = Path("/root/autodl-tmp/zuxing.lu")  # READ ONLY (round 6 evaluates our sha256-checked copies of his files)
FC = ME / "fc_runs"
TE_FILE = ME / "scripts/t2cons_eval.py"
TE_SHA = "2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b"  # round-1 PREREG_PINS.md, PREREG_R6 11
PREREG, PREREG_SHA_FILE = ROOT / "PREREG_R6.md", ROOT / "PREREG_R6.sha256"
PREREG_SHA = "0b0fbf9a2b581913ae0da7d36ce76aa785e706c132dc496054f2323a80e487e1"  # PREREG_R6.md (frozen 01:16:30)
PINS, PINS_SHA_FILE = ROOT / "PREREG_R6_PINS.md", ROOT / "PREREG_R6_PINS.sha256"
EXPORTS = ROOT / "exports"  # every round-6 export (one per screen job)
RUNS_S, RUNS_F = ROOT / "runs_s", ROOT / "runs_f"
POOL = ROOT / "inputs_r6/pool_r6.json"
INCS = ROOT / "inputs_r6/incumbents_r6.json"
COPIES = ROOT / "COPIES_r6.tsv"
SNAP = ROOT / "inputs_r6/SNAPSHOT_R6.sha256"
TRACKS = ROOT / "inputs_r6/TRACKS_R6.sha256"
FINAL_LIST = ROOT / "final_r6.json"  # written by the phase-S selection (PREREG_R6 5.4), locked by SELECTION_S6.sha256
SEL_S6 = ROOT / "SELECTION_S6.sha256"
TZ8 = dt.timezone(dt.timedelta(hours=8))
T_C = dt.datetime(2026, 10, 31, 1, 9, 17, tzinfo=TZ8)
NRUNS_S, NRUNS_F = 3, 10  # PREREG_R6 3.2 / 3.3
N_SCREEN = 250
DRY_JOB = {"screen": "ep26_k3ca_25298", "final": "ep26_zx_k3fix_92000"}  # PREREG_R6 3.6
OUR_RUN_DIRS = (ME / "fc_runs/C3r2_tier2", ME / "fc_runs/C3r2_tier2_k3fix", ME / "fc_runs/C3r2_tier2_k3easy",
                ME / "fc_runs/C3r2_tier2_k3_stretch90k", ME / "fc_runs/C3r2_tier2_k3ca", ME / "fc_runs/T2fix/T2B",
                ME / "fc_runs/T2fix/T2L")  # fmt: skip
ZX_COPY_DIR = ROOT / "zuxing_copies_r6"
K3CA_ROOT = ME / "data/human_motion_data_tier2_k3ca"
K3CA_REFS = ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca"  # round 1's t2cons_consensus INPUT_REF["k3ca"]
ALLOWED_GPUS = (0, 1, 2, 7)  # PREREG_R6 3.4: GPUs 0-2 and 7 only, one evaluation of ours per GPU
OWN_LOCKDIR = ROOT / "evallock"
SHARED_LOCKDIRS = (FC / "queue_runin/evallock", FC / "queue_tier2v2/evallock", FC / "queue_t1fix/evallock",
                   FC / "queue_t1fix2/evallock", FC / "queue_t1fix3/evallock", FC / "queue_t1fix4/evallock",
                   FC / "queue_t1fix5/evallock", FC / "queue_t2fix/evallock",
                   FC / "submissions/t1_best20_20261026/evallock", FC / "submissions/t3_f0_final/evallock",
                   ME / "tmp_analysis/tier3/f0/evallock", ME / "tmp_analysis/tier3/opus_easy/evallock",
                   ME / "tmp_analysis/tier2/k3ca/train/evallock", ME / "tmp_analysis/tier2/k3fix_pilot/evallock",
                   ME / "tmp_analysis/tier2/k3full/evallock", ME / "tmp_analysis/tier2/k3easy/evallock",
                   FC / "queue_t1improve/evallock",  # AMENDMENTS_R2_c C2
                   R2 / "evallock",  # round 2's own lane locks (PREREG_R3 3.3)
                   R3 / "evallock", FC / "T1improve/confirm/evallock",  # PREREG_R4 3.3
                   R4 / "evallock",  # PREREG_R5 3.3
                   R5 / "evallock")  # fmt: skip  (PREREG_R6 3.4: round 5's own lane locks)
EXT_MARKERS = ("T2BEST5_JOB", "T3FINAL_JOB", "T1BEST20_JOB", "T1CONFIRM_JOB")  # other drivers' live evals (our caps)
HOST = socket.gethostname()
OUT_LOG = ROOT / "eval_r6_driver.log"
PHASE = {  # per phase: runs base, runs per job, output names
    "screen": {"tag": "s6", "runs": RUNS_S, "nruns": NRUNS_S, "manifest": ROOT / "manifest_s6.json",
               "manifest_csv": ROOT / "manifest_s6.csv", "evals": ROOT / "own_input_evals_s6.csv",
               "summary": ROOT / "summary_s6_own_input", "hb": ROOT / "eval_r6_heartbeat_s6.json",
               "seeds": ROOT / "seeds_s6.tsv"},
    "final": {"tag": "f6", "runs": RUNS_F, "nruns": NRUNS_F, "manifest": ROOT / "manifest_f6.json",
              "manifest_csv": ROOT / "manifest_f6.csv", "evals": ROOT / "own_input_evals_f6.csv",
              "summary": ROOT / "summary_f6_own_input", "hb": ROOT / "eval_r6_heartbeat_f6.json",
              "seeds": ROOT / "seeds_f6.tsv"},
}  # fmt: skip
SEED_COLS = ["env.action.body_delay.seed", "body_delay.group_seeds", "env.action.body_delay.min_steps",
             "env.action.body_delay.max_steps", "env.reset.seed(stored)", "env.reset.seed(effective)",
             "scene.object_scale_seed", "scene.object_scale_min", "scene.object_scale_max",
             "env.objective.wrench_basis_seed", "training.seed", "evaluation.ADD_VERTEX_SEED", "v2dlb.ADD_VERTEX_SEED",
             "seed_signature"]  # fmt: skip
SEED_TSV_COLS = (["job", "episode", "run", "try", "status", "auc_own_input"] + SEED_COLS
                 + ["gpu_index", "gpu_uuid", "host", "start", "end", "seconds", "stock_commit", "eval_script_sha256(parquet)",
                    "eval_code_sha256(parquet)", "checkpoint_sha256(parquet)", "world_count(json)", "exported_file_sha256",
                    "lane_locks", "runner_command"])  # fmt: skip


# ================================================================================================ loading (pinned bytes)
def sha256_file(file) -> str:
    h = hashlib.sha256()
    with Path(file).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_te():
    """scripts/t2cons_eval.py compiled from the exact bytes whose sha256 round 1 pins (it is not run)."""
    src = TE_FILE.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != TE_SHA:
        raise SystemExit(f"{TE_FILE} sha256 {got} != pinned {TE_SHA}")
    mod = types.ModuleType("t2cons_eval_readonly")
    mod.__file__ = str(TE_FILE)
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(TE_FILE), "exec"), mod.__dict__)  # noqa: S102 (pinned bytes)
    return mod


TE = load_te()
_GUARD_EXTRA = ("readme", "train_rl.log", "t2_zx07_20261028", "best_weights", "seed_best", "remote_reps", "report_r",
                "composites", "per_episode_zx")  # fmt: skip


def guard(p) -> Path:
    s = str(p).lower()
    if any(tok in s for tok in _GUARD_EXTRA):
        raise PermissionError(f"t2cons_r6_eval may not open {p}")
    return TE.guard(p)  # round-1 guard: clean, local_not_for_upload, packed_


def guarded_json(p):
    try:
        return json.loads(guard(p).read_text())
    except (OSError, ValueError, TypeError):
        return None


INC_EXPORTS: dict[str, Path] = {}  # incumbent job -> its earlier round's export.json (filled from incumbents_r6.json)


def load_driver():
    D = TE.load_driver()  # t2best5_driver.py from its pinned bytes (re-rooted by TE at round 1) ...
    D.ROOT = ROOT  # ... and re-rooted here: every write of the pinned code goes through inside_root(ROOT)
    D.PREREG, D.PREREG_SHA = PREREG, PREREG_SHA_FILE
    D.PINS = D.PINS_SHA = None
    D.ALLOWED_GPUS = ALLOWED_GPUS  # PREREG_R6 3.4 (the pinned Driver checks --gpus against this module name)
    D.check_pins = lambda require_pins_file=True: check_pins(D, require_pins_file)
    # PREREG_R6 3.1 (Q4): the K3CA input, rebinding only (the pinned functions read these module tables at call time)
    D.ROOTS = {**D.ROOTS, "k3ca": K3CA_ROOT}
    D.FRAME_REFS = {**D.FRAME_REFS, "k3ca": K3CA_REFS}
    D.MANIFEST_FOR = {k: (v / "whole_body/soma/track2_manifest.json") for k, v in D.ROOTS.items()}

    def export_rec_r6(base, job_id):  # a screen job's export lives in exports/; an incumbent's is the earlier round's
        if Path(base) != ROOT:
            raise RuntimeError(f"export_rec called with base {base}, not {ROOT}")
        if job_id in INC_EXPORTS:
            return D.read_json(guard(INC_EXPORTS[job_id]))
        return D.read_json(EXPORTS / job_id / "export.json")

    D.export_rec = export_rec_r6
    return D


# ================================================================================================ pins
def _sha_file_value(p: Path) -> str:
    return (guard(p).read_text().split() or [""])[0] if p.is_file() else ""


def _pin_hash(p: Path) -> str:
    if not p.is_file():
        return "missing"
    return sha256_file(guard(p))


def _pin_path(rel: str) -> Path:
    if rel.startswith("/"):
        return Path(rel)
    return ME / rel if rel.startswith(("scripts/", "tmp_analysis/")) else ROOT / rel


def _pins(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    pins, commits = [], []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64:
            pins.append((f[1], f[2]))
        elif len(f) == 3 and f[0] == "commit" and len(f[2]) == 40:
            commits.append((f[1], f[2]))
    return pins, commits


def _hash_list(listing: Path, base: Path | None) -> list[str]:
    bad = []
    n = 0
    for line in guard(listing).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        p = Path(parts[1].strip()) if base is None else base / parts[1].strip()
        h = sha256_file(guard(p)) if p.is_file() else "missing"
        n += 1
        if h != parts[0]:
            bad.append(f"{listing.name}: {p} {h[:16]} != {parts[0][:16]}")
    return bad if bad else [f"__ok__ {n}"]


def check_pins(D, require_pins_file: bool) -> tuple[list[str], list[str]]:
    fatal, info = [], []
    got, rec = (sha256_file(PREREG) if PREREG.is_file() else "missing"), _sha_file_value(PREREG_SHA_FILE)
    if not (got == rec == PREREG_SHA):
        return [f"PREREG_R6.md sha256 {got[:16]} (sha file {rec[:16]}, pinned here {PREREG_SHA[:16]})"], info
    info.append(f"PREREG_R6.md sha256 {got[:16]} == PREREG_R6.sha256 == pinned")
    pins, commits = _pins(guard(PREREG).read_text().split("## 11. Pins", 1)[-1])
    for rel, want in pins:
        h = _pin_hash(_pin_path(rel))
        if h != want:
            fatal.append(f"PREREG_R6 pin {rel}: {h[:16]} != {want[:16]}")
    for repo, want in commits:
        head = D.git_head(ME / repo)
        (info if head == want else fatal).append(f"commit {repo}: HEAD {head[:12]} (pinned {want[:12]})")
    info.append(f"PREREG_R6.md section 11: {len(pins)} pins checked, {len(commits)} commits")
    for listing, base in ((SNAP, ROOT), (TRACKS, None)):
        res = _hash_list(listing, base)
        if res and res[0].startswith("__ok__"):
            info.append(f"{listing.name}: {res[0].split()[1]} files verify")
        else:
            fatal += res
    f1, i1 = TE.check_pins(D)  # round-1 PREREG.md == its sha file, its section-11 pins + commits, inputs/SNAPSHOT.sha256
    fatal += [f"round 1: {x}" for x in f1]
    info += [f"round 1: {x}" for x in i1 if "t2cons_eval.py sha256" not in x]
    mine = sha256_file(Path(__file__).resolve())
    # PREREG_R6_PINS.md (step R2): == its sha file, every pin holds, and it pins this script and its launcher
    if PINS.is_file():
        got, rec = sha256_file(PINS), _sha_file_value(PINS_SHA_FILE)
        if got != rec:
            fatal.append(f"PREREG_R6_PINS.md sha256 {got[:16]} != PREREG_R6_PINS.sha256 {rec[:16]}")
        else:
            ppins, _ = _pins(guard(PINS).read_text())
            names = {rel for rel, _ in ppins}
            need = ["scripts/t2cons_r6_eval.py", "scripts/t2cons_r6_eval_launch.sh"]
            fatal += [f"PREREG_R6_PINS.md does not pin {rel}" for rel in need if rel not in names]
            if ("scripts/t2cons_r6_eval.py", mine) not in ppins:
                fatal.append(f"PREREG_R6_PINS.md does not pin this script {mine[:16]}")
            nbad = 0
            for rel, want in ppins:
                h = _pin_hash(_pin_path(rel))
                if h != want:
                    nbad += 1
                    fatal.append(f"PREREG_R6_PINS pin {rel}: {h[:16]} != {want[:16]}")
            if not nbad:
                info.append(f"PREREG_R6_PINS.md sha256 {got[:16]} == its sha file; its {len(ppins)} pins hold; pins this "
                            f"script {mine[:16]}")
    elif require_pins_file:
        fatal.append("PREREG_R6_PINS.md missing: this script must be pinned before it exports or evaluates (PREREG_R6 10)")
    else:
        info.append(f"PREREG_R6_PINS.md not written yet (plan-only dry run); this script sha256 {mine}")
    return fatal, info


def final_list_check() -> tuple[dict | None, list[str]]:
    """PREREG_R6 3.3 / 5.4: final_r6.json must be listed, with its current sha256, in SELECTION_S6.sha256 (written last
    by the pinned phase-S selection), and every line of that lock must hold."""
    if not SEL_S6.is_file():
        return None, ["SELECTION_S6.sha256 missing: the phase-S selection is not frozen (finalists unknown)"]
    lines = [ln.split(None, 1) for ln in guard(SEL_S6).read_text().splitlines() if ln.strip()]
    problems, listed = [], False
    for parts in lines:
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        p = _pin_path(parts[1].strip())
        if _pin_hash(p) != parts[0]:
            problems.append(f"SELECTION_S6.sha256: {parts[1].strip()} changed")
        if p == FINAL_LIST:
            listed = True
    if not listed:
        problems.append("SELECTION_S6.sha256 does not list final_r6.json")
    data = guarded_json(FINAL_LIST)
    if data is None:
        problems.append("final_r6.json unreadable")
    return data, problems


# ================================================================================================ manifest
def load_pool() -> list[dict]:
    data = guarded_json(POOL)
    if data is None:
        raise SystemExit(f"cannot read {POOL}")
    return list(data["rows"])


def load_incumbents() -> dict[int, dict]:
    data = guarded_json(INCS)
    if data is None:
        raise SystemExit(f"cannot read {INCS}")
    out = {int(k): v for k, v in data["incumbents"].items()}
    if sorted(out) != list(range(30)):
        raise SystemExit("incumbents_r6.json does not hold the 30 episodes")
    for x in out.values():
        INC_EXPORTS[x["job"]] = Path(x["export_record"])
    return out


def track_records() -> dict[str, str]:
    out = {}
    for line in guard(TRACKS).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[os.path.normpath(parts[1].strip())] = parts[0]
    return out


def copy_rows() -> dict[str, dict]:
    with guard(COPIES).open(newline="") as fp:
        return {os.path.normpath(r["copy"]): r for r in csv.DictReader(fp, delimiter="\t")}


def _input_checks(D, job: dict, problems: list[str], notes: list[str], trk: dict[str, str], ep: int, kind_want: str,
                  parquet: Path, psha_want: str, fr_want: str | None) -> None:
    """Training parquet, data root, manifest and frame reference of a job (round 5's checks, re-keyed on TRACKS_R6)."""
    lay = D.parquet_layout(parquet)
    if lay is None or not parquet.is_file():
        problems.append(f"no / unexpected training parquet {parquet}")
        return
    data_root, seq = lay
    kind = D.root_kind(data_root)
    psha = sha256_file(parquet)
    if kind != kind_want:
        problems.append(f"data root kind {kind} != {kind_want}")
    if psha != psha_want or trk.get(os.path.normpath(str(parquet))) != psha:
        problems.append(f"parquet sha256 {psha[:16]} != record / TRACKS_R6")
    else:
        notes.append(f"training parquet sha256 {psha[:12]} == record == TRACKS_R6.sha256")
    manifest = D.MANIFEST_FOR.get(kind) if kind else None
    if manifest is None or D.manifest_episode_seq(manifest, ep) != seq:
        problems.append(f"manifest {manifest} does not map ep{ep:02d} to {seq}")
    fr = D.FRAME_REFS.get(kind) if kind else None
    if kind == "raw":
        if fr is not None:
            problems.append(f"raw job with frame refs {fr}")
        else:
            notes.append("raw input: no frame-reference npz (the scorer's own-input reference is the released Tier-2 parquet)")
    else:
        npz = Path(fr) / f"episode_{ep:06d}.npz" if fr else None
        if npz is None or not npz.is_file() or sha256_file(npz) != fr_want or trk.get(os.path.normpath(str(npz))) != fr_want:
            problems.append(f"{npz} missing or != record / TRACKS_R6 sha256")
    job.update(sequence_id=seq, parquet=str(parquet), parquet_sha256=psha, data_root=str(data_root), root_kind=kind,
               manifest=str(manifest), frame_refs=str(fr) if fr else None)  # fmt: skip


def build_screen(D, p: dict, trk: dict[str, str], cop: dict[str, dict]) -> dict:
    """A phase-S job = one 'screen' row of pool_r6.json (PREREG_R6 2.1 / 3.2)."""
    jid, ep, it = p["job"], int(p["episode"]), int(p["iteration"])
    problems: list[str] = []
    notes: list[str] = []
    ck = Path(p["training_checkpoint"])
    job = {"job": jid, "episode": ep, "object": None, "family": p["owner"], "owner": p["owner"],
           "arm": f"{p['run_label']} ({p['input']}) {p['slot']}", "stage": "round 6 phase S (PREREG_R6 3.2)",
           "source": p["run_dir"], "variants": "B", "priority": 1, "provisional": False, "role": "screen",
           "runs": NRUNS_S, "iteration": it, "tree": p["tree"], "run_label": p["run_label"], "slot": p["slot"],
           "expected_sha256": p["sha256"], "expected_sha256_from": "inputs_r6/pool_r6.json (PREREG_R6 2)",
           "training_checkpoint": str(ck), "source_checkpoint": p["source_checkpoint"]}  # fmt: skip
    if p.get("role") != "screen":
        problems.append(f"pool role {p.get('role')} is not screen")
    if p["tree"] not in D.TREES or p["tree"] != ("fc_r2" if p["owner"] == "haoran" else "b39b69b"):
        problems.append(f"tree {p['tree']} is not the owner's export tree (PREREG_R6 3.1)")
    if p["owner"] == "haoran":
        if ck != Path(p["source_checkpoint"]) or not any(str(ck).startswith(str(d) + "/") for d in OUR_RUN_DIRS):
            problems.append(f"checkpoint {ck} is not our original in {[str(d) for d in OUR_RUN_DIRS]}")
        des = {"rule": "pool rule R1-R5 (PREREG_R6 2.1): our training run", "run_dir": p["run_dir"], "slot": p["slot"]}
    else:
        row = cop.get(os.path.normpath(str(ck)))
        if not str(ck).startswith(str(ZX_COPY_DIR) + "/") or row is None:
            problems.append(f"checkpoint {ck} is not a recorded copy in zuxing_copies_r6/")
        elif not (row.get("equal") == "True" and row["sha256_copy"] == row["sha256_source_before"] ==
                  row["sha256_source_after"] == p["sha256"] and os.path.normpath(row["source"]) ==
                  os.path.normpath(p["source_checkpoint"])):  # fmt: skip
            problems.append("COPIES_r6.tsv row does not show source before == copy == after == the pool sha256")
        des = {"rule": "pool rule R1-R5 (PREREG_R6 2.1): Zuxing's pro6000 training run, sha256-verified copy",
               "run_dir": p["run_dir"], "slot": p["slot"], "copy_row": row}  # fmt: skip
    tsha = sha256_file(ck) if ck.is_file() else None
    if tsha != p["sha256"]:
        problems.append(f"checkpoint sha256 {str(tsha)[:16]} != pool {p['sha256'][:16]}")
    job["training_sha256"] = tsha
    job["designation_evidence"] = des
    _input_checks(D, job, problems, notes, trk, ep, p["input"], Path(p["parquet"]), p["parquet_sha256"], p.get("frame_ref_sha256"))
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


def build_incumbent(D, x: dict, trk: dict[str, str]) -> dict:
    """A phase-F incumbent job (PREREG_R6 3.3 / 4): its earlier export record and file, re-hashed; never re-exported."""
    jid, ep = x["job"], int(x["episode"])
    problems: list[str] = []
    notes: list[str] = []
    rec = guarded_json(Path(x["export_record"])) or {}
    f = Path(str(rec.get("file")))
    fsha = sha256_file(f) if f.is_file() else None
    if sha256_file(guard(Path(x["export_record"]))) != x["export_record_sha256"]:
        problems.append("export record changed since incumbents_r6.json")
    if rec.get("status") != "ok" or rec.get("stock_verified") is not True or fsha != x["export_sha256"] \
            or rec.get("sha256") != fsha or Path(str(rec.get("file"))) != Path(x["export_file"]):  # fmt: skip
        problems.append(f"export record / file is not the recorded ok export {x['export_sha256'][:16]}")
    tck = Path(x["training_checkpoint"])
    tsha = sha256_file(tck) if tck.is_file() else None
    if tsha != x["training_sha256"] or rec.get("source_sha256") != tsha:
        problems.append("training file sha256 != record")
    job = {"job": jid, "episode": ep, "object": x.get("object"), "family": x.get("owner"), "owner": x.get("owner"),
           "arm": f"incumbent ({x['source_round']}, {x['input']})", "stage": "round 6 phase F (PREREG_R6 3.3)",
           "source": x["source_round"], "variants": "B", "priority": 1, "provisional": False, "role": "incumbent",
           "runs": NRUNS_F, "iteration": int(x["iteration"]), "tree": x["tree"], "expected_sha256": x["training_sha256"],
           "expected_sha256_from": "inputs_r6/incumbents_r6.json (PREREG_R6 4)", "training_checkpoint": str(tck),
           "training_sha256": tsha, "export_record": x["export_record"], "export_sha256": fsha,
           "designation_evidence": {"rule": "round-5 incumbent (selection_r5.json ROUND5); earlier export reused",
                                    "m_P_frozen": x["m_P_frozen"], "r5_closest_run": x["r5_closest_run"]}}  # fmt: skip
    if x["tree"] not in D.TREES:
        problems.append(f"tree {x['tree']} unknown")
    _input_checks(D, job, problems, notes, trk, ep, x["input"], Path(x["parquet"]), x["parquet_sha256"], x.get("frame_ref_sha256"))
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


def build_finalist(D, fl: dict, pool_by: dict[str, dict], trk: dict[str, str], cop: dict[str, dict]) -> dict:
    """A phase-F finalist (PREREG_R6 5.4): the screen job with its phase-S export (exports/<job>/), 10 fresh runs."""
    job = build_screen(D, pool_by[fl["job"]], trk, cop)
    job.update(role="finalist", runs=NRUNS_F, stage="round 6 phase F (PREREG_R6 3.3)")
    exp = guarded_json(EXPORTS / fl["job"] / "export.json") or {}
    if exp.get("status") != "ok" or exp.get("sha256") != fl.get("export_sha256"):
        job["problems"].append("phase-S export record != final_r6.json")
        job["status"] = "refused: " + "; ".join(job["problems"])
    return job


def build_manifest(D, log, phase: str, only: set | None = None) -> list[dict]:
    trk = track_records()
    inc = load_incumbents()
    pool = load_pool()
    cop = copy_rows()
    if phase == "screen":
        rows = [p for p in pool if p.get("role") == "screen"]
        if len(rows) != N_SCREEN:
            raise SystemExit(f"pool_r6.json has {len(rows)} screen rows, PREREG_R6 2.2 says {N_SCREEN}")
        rows.sort(key=lambda p: (int(p["episode"]), p["job"]))
        jobs = [build_screen(D, p, trk, cop) for p in rows if only is None or p["job"] in only]
        log(f"phase S job list from pool_r6.json: {len(jobs)} of {len(rows)} screen jobs (PREREG_R6 3.2)")
    else:
        jobs = [build_incumbent(D, inc[e], trk) for e in range(30) if only is None or inc[e]["job"] in only]
        if only is None:
            data, problems = final_list_check()
            if problems:
                raise SystemExit("phase F needs the frozen phase-S selection: " + "; ".join(problems))
            pool_by = {p["job"]: p for p in pool if p.get("role") == "screen"}
            for fl in data.get("finalists", []):
                jobs.append(build_finalist(D, fl, pool_by, trk, cop))
        jobs.sort(key=lambda j: (int(j["episode"]), j["job"]))
        log(f"phase F job list: {sum(1 for j in jobs if j['role'] == 'incumbent')} incumbents + "
            f"{sum(1 for j in jobs if j['role'] == 'finalist')} finalists (PREREG_R6 3.3)")
    ok = [r for r in jobs if r["status"] == "ok"]
    hulls = D.hull_check(ok, log)
    for r in ok:
        h = hulls.get(r["parquet"], {"ok": False, "missing": ["not checked"]})
        r["hulls"] = h
        if not h["ok"]:
            r["status"] = f"hull_missing: {h['missing']}"
    return jobs


MANIFEST_COLS = ["job", "episode", "role", "runs", "owner", "tree", "source", "arm", "iteration", "training_sha256",
                 "training_checkpoint", "export_record", "export_sha256", "parquet", "parquet_sha256", "root_kind",
                 "data_root", "manifest", "frame_refs", "status"]  # fmt: skip


# ================================================================================================ seeds (records only)
def resolved_config(file: Path) -> dict:
    with Path(file).open("rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(n))
    return json.loads(header["__metadata__"]["user.resolved_config_json"])


def dig(c, *keys):
    for k in keys:
        if not isinstance(c, dict) or k not in c:
            return "MISSING"
        c = c[k]
    return c


def const_from(path: Path, name: str):
    import ast

    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(getattr(t, "id", None) == name for t in targets) and getattr(node, "value", None) is not None:
            return ast.literal_eval(node.value)
    return None


def forced_reset_seed(D):
    text = (D.STOCK / "src/flash_chord/evaluation/harness.py").read_text()
    m = re.search(r'config\["env"\]\["reset"\]\["seed"\]\s*=\s*(\d+)', text)
    return int(m.group(1)) if m else None


def seeds_of(D, file: Path) -> dict:
    """Every seed the official evaluation of this file uses (the t1_best20 audit's list): the file's own resolved config
    (covered by checkpoint_sha256), the harness's forced reset seed and the ADD vertex-sampling constants."""
    c = resolved_config(file)
    bd = dig(c, "env", "action", "body_delay")
    seed = bd.get("seed") if isinstance(bd, dict) else "MISSING"
    out = {
        "env.action.body_delay.seed": seed,
        "body_delay.group_seeds": ",".join(str(seed + g) for g in range(5)) if isinstance(seed, int) else "MISSING",
        "env.action.body_delay.min_steps": bd.get("min_steps") if isinstance(bd, dict) else "MISSING",
        "env.action.body_delay.max_steps": bd.get("max_steps") if isinstance(bd, dict) else "MISSING",
        "env.reset.seed(stored)": dig(c, "env", "reset", "seed"),
        "env.reset.seed(effective)": forced_reset_seed(D),
        "scene.object_scale_seed": dig(c, "scene", "object_scale_seed"),
        "scene.object_scale_min": dig(c, "scene", "object_scale_min"),
        "scene.object_scale_max": dig(c, "scene", "object_scale_max"),
        "env.objective.wrench_basis_seed": dig(c, "env", "objective", "wrench_basis_seed"),
        "training.seed": dig(c, "training", "seed"),
        "evaluation.ADD_VERTEX_SEED": const_from(D.STOCK / "src/flash_chord/evaluation/metrics.py", "ADD_VERTEX_SEED"),
        "v2dlb.ADD_VERTEX_SEED": const_from(D.KIT / "v2dlb/vertices.py", "ADD_VERTEX_SEED"),
    }  # fmt: skip
    out["seed_signature"] = (f"delay={out['env.action.body_delay.seed']}(+g0..4,{out['env.action.body_delay.min_steps']}-"
                             f"{out['env.action.body_delay.max_steps']}) reset={out['env.reset.seed(effective)']} "
                             f"scale_seed={out['scene.object_scale_seed']}({out['scene.object_scale_min']}/"
                             f"{out['scene.object_scale_max']}) wrench={out['env.objective.wrench_basis_seed']} "
                             f"vertex={out['evaluation.ADD_VERTEX_SEED']}/{out['v2dlb.ADD_VERTEX_SEED']} "
                             f"ppo={out['training.seed']}")  # fmt: skip
    return out


def parquet_digests(pp: Path) -> dict:
    try:
        import pyarrow.parquet as pq

        md = {k.decode(errors="replace"): v.decode(errors="replace")
              for k, v in (pq.ParquetFile(guard(pp)).schema_arrow.metadata or {}).items() if not k.startswith(b"pandas")}  # fmt: skip
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error)}
    return {k: md.get(f"flash_chord.{k}") for k in ("eval_script_sha256", "eval_code_sha256", "checkpoint_sha256", "step_count")}


# ================================================================================================ lanes
def try_lock(path: Path, create: bool) -> int | None:
    try:
        fd = os.open(path, (os.O_RDWR | os.O_CREAT) if create else os.O_RDONLY, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _lane_lock_groups(gpu: int) -> list[list[Path]]:
    """AMENDMENTS_R2_c C1: the existing shared lock paths of this GPU, grouped by the file they name ((st_dev, st_ino)
    after following links), groups in first-path SHARED_LOCKDIRS order; group[0] is the path that is opened."""
    groups: dict[tuple[int, int], list[Path]] = {}
    for d in SHARED_LOCKDIRS:
        p = d / f"gpu{gpu}.lock"
        try:
            st = os.stat(p)  # follows links
        except OSError:
            continue
        if not p.is_file():
            continue
        groups.setdefault((st.st_dev, st.st_ino), []).append(p)
    return list(groups.values())


def lane_lock_files(gpu: int) -> list[Path]:
    """Each shared lock FILE once (AMENDMENTS_R2_c C1): flock belongs to the open file description, so a second open of
    the same inode by this process would be refused by our own first lock."""
    return [g[0] for g in _lane_lock_groups(gpu)]


def lane_alias_report(gpus) -> list[str]:
    """AMENDMENTS_R2_c C1: one line per GPU naming the lock files taken and the alias paths merged into them."""
    out = []
    for g in gpus:
        parts = []
        for grp in _lane_lock_groups(g):
            al = [str(x) for x in grp[1:]]
            parts.append(f"{grp[0]}" + (f" (= {', '.join(al)})" if al else ""))
        out.append(f"lane locks gpu{g}: {len(parts)} shared lock file(s): " + ("; ".join(parts) or "none"))
    return out


def take_lane(D, gpu: int):
    """Our lane lock (created in ROOT/evallock) + every shared lane lock file of that GPU (read-only, never created);
    all or nothing, never blocking."""
    OWN_LOCKDIR.mkdir(parents=True, exist_ok=True)
    own = try_lock(D.inside_root(OWN_LOCKDIR / f"gpu{gpu}.lock"), create=True)
    if own is None:
        return f"gpu{gpu}: our own lane lock is held"
    fds, taken = [own], [str(OWN_LOCKDIR / f"gpu{gpu}.lock")]
    for p in lane_lock_files(gpu):
        fd = try_lock(p, create=False)
        if fd is None:
            for x in fds:
                os.close(x)
            return f"gpu{gpu}: {p} is held"
        fds.append(fd)
        taken.append(str(p))
    return fds, taken


def external_evals(D) -> list[dict]:
    """Live GPU evaluations of the t2_best5 / t3final / t1best20 drivers (their job markers), read from /proc only."""
    mine = str(ROOT) + "/"
    seen: dict[tuple, int] = {}
    for pid in D.all_pids():
        env = D.proc_environ(pid)
        for m in EXT_MARKERS:
            v = env.get(m, "")
            if not v or v.startswith(mine):
                continue
            st = D.proc_stat(pid)
            gpu = env.get("CUDA_VISIBLE_DEVICES", "").strip()
            if st is None or st[0] in ("Z", "X") or not gpu.isdigit():
                continue
            seen[(m, v)] = int(gpu)
    return [{"gpu": g, "marker": f"{m}={v}"} for (m, v), g in seen.items()]


# ================================================================================================ summary (own input)
EVAL_COLS = ["job", "episode", "role", "run", "status", "try", "gpu", "seconds", "auc", "sp_sr", "mp_sr", "mppe", "chord_sr",
             "export_sha256", "finished_at", "problems", "export_sha256_full", "results_parquet", "reference_parquet_sha256",
             "variants", "input", "seed_signature"]  # fmt: skip


def write_summary(D, jobs: list[dict], phase: str) -> dict:
    P = PHASE[phase]
    nmax = P["nruns"]
    eval_rows, job_rows = [], []
    for j in jobs:
        jid = j["job"]
        base = {"job": jid, "episode": j["episode"], "role": j.get("role"), "source": j.get("source"),
                "input": j.get("root_kind"), "iteration": j.get("iteration"), "checkpoint_sha256": j.get("training_sha256")}  # fmt: skip
        if j.get("status") != "ok":
            job_rows.append({**base, "export_status": j.get("status")})
            continue
        exp = D.export_rec(ROOT, jid) or {}
        recs, failed = {}, []
        for k in range(1, nmax + 1):
            r = guarded_json(P["runs"] / jid / f"run{k}" / "run.json")
            if r is None:
                continue
            m = r.get("metrics") or {}
            sd = guarded_json(P["runs"] / jid / f"run{k}" / f"try{r.get('try')}" / "seeds.json") or {}
            eval_rows.append({"job": jid, "episode": j["episode"], "role": j.get("role"), "run": k, "status": r.get("status"),
                              "try": r.get("try"), "gpu": r.get("gpu"), "seconds": r.get("seconds"), **m,
                              "chord_sr": r.get("chord_sr"), "export_sha256": str(r.get("export_sha256"))[:16],
                              "finished_at": r.get("finished_at"), "problems": "; ".join(r.get("problems") or []),
                              "export_sha256_full": r.get("export_sha256"), "results_parquet": r.get("results_parquet"),
                              "reference_parquet_sha256": r.get("reference_parquet_sha256"), "variants": j.get("variants"),
                              "input": j.get("root_kind"), "seed_signature": (sd.get("seeds") or {}).get("seed_signature")})  # fmt: skip
            if r.get("status") == "ok":
                recs[k] = m
            else:
                failed.append(k)
        floor = sum(1 for k in recs if recs[k]["sp_sr"] == 1 and recs[k]["mp_sr"] == 1)
        job_rows.append({**base, "parquet_sha256": j.get("parquet_sha256"), "export_status": exp.get("status", "pending"),
                         "export_tolerance": exp.get("tolerance"), "export_fallback": exp.get("fallback"),
                         "export_sha256": exp.get("sha256", ""), "export_max_rel_diff": exp.get("max_rel_diff"),
                         "n_ok": len(recs), "failed_runs": " ".join(map(str, failed)), "floor_sp1_mp1_runs": floor,
                         "runs_planned": nmax, **D.job_stats_row(j, recs, nmax)})  # fmt: skip
    cols = ["job", "episode", "role", "source", "input", "parquet_sha256", "iteration", "checkpoint_sha256", "export_status",
            "export_tolerance", "export_fallback", "export_sha256", "export_max_rel_diff", "runs_planned", "n_ok",
            "failed_runs", "floor_sp1_mp1_runs"]  # fmt: skip
    for m in D.METRICS:
        cols += [f"{m}_r{k}" for k in range(1, nmax + 1)] + [f"{m}_{s}" for s in ("mean", "sd", "min", "max")]
    cols += ["closest_to_mean_run"]
    D.write_csv(P["evals"], eval_rows, EVAL_COLS)
    D.write_csv(P["summary"].with_suffix(".csv"), job_rows, cols)
    D.write_atomic(P["summary"].with_suffix(".json"), D.dump({
        "generated_at": D.now(), "phase": phase, "runs_per_job": nmax, "jobs": job_rows,
        "note": "own input = each policy against its OWN training parquet (official kit evaluator 160d24f5, stock 8c778f9, "
                "frame 0, explicit reset, world count 1). Comparable only within one training parquet sha256. floor = runs "
                "with SP-SR = MP-SR = 1. The selection uses consensus-B scores (PREREG_R6 5), not these values. No clean "
                "number anywhere in this file."}))
    md = [f"# t2_consensus round 6, phase {phase}: official own-input evaluations ({D.now()})", "",
          "Own input = each policy against its OWN training parquet. Not a leaderboard number; no clean value was computed "
          "or read. The selection uses consensus-B scores (PREREG_R6 5), not these values.", "",
          "| job | role | input | ckpt sha256 | export | ok runs | AUC mean | sd | min | max | SP-SR | MP-SR | MPPE cm | floor |",
          "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in job_rows:
        md.append(f"| {r['job']} | {r.get('role', '')} | {r.get('input', '')} | {str(r.get('checkpoint_sha256', ''))[:12]} | "
                  f"{r.get('export_status', '')} {r.get('export_tolerance') or ''} | {r.get('n_ok', '')} | {D.fv(r.get('auc_mean'))} | "
                  f"{D.fv(r.get('auc_sd'))} | {D.fv(r.get('auc_min'))} | {D.fv(r.get('auc_max'))} | "
                  f"{D.fv(r.get('sp_sr_mean'), '.2f')} | {D.fv(r.get('mp_sr_mean'), '.2f')} | {D.fv(r.get('mppe_mean'), '.2f')} | "
                  f"{r.get('floor_sp1_mp1_runs', '')}/{r.get('n_ok', '')} |")  # fmt: skip
    D.write_atomic(P["summary"].with_suffix(".md"), "\n".join(md) + "\n")
    return {"jobs": job_rows}


# ================================================================================================ the driver
def make_driver(D):
    Base = TE.make_driver(D)  # pinned ConsDriver (no Stage B, no clean), itself the pinned t2best5 Driver

    class R6Driver(Base):
        def __init__(self, a, log, phase: str, runs_base: Path, only: set | None, hb_file: Path, seeds_tsv: Path,
                     write_summary_: bool, exports_only: bool = False, manifest_file: Path | None = None):
            super().__init__(a, log)
            self.phase = phase
            self.runs_base, self.only, self.hb_file, self.seeds_tsv = runs_base, only, hb_file, seeds_tsv
            self.write_summary_ = write_summary_
            self.exports_only = exports_only  # PREREG_R6 3.2: every screen job's single export, no evaluation
            self.manifest_file = manifest_file or PHASE[phase]["manifest"]
            self.rest_until: dict[int, float] = {}

        # -------------------------------------------------------------------------------- jobs
        def ok_jobs(self) -> list[dict]:
            return [j for j in self.jobs if j.get("status") == "ok" and (self.only is None or j["job"] in self.only)]

        def nruns(self, job: dict) -> int:  # PREREG_R6 3.2 / 3.3: 3 (screen) or 10 (final), capped by the mode
            return min(int(self.a.runs), int(job.get("runs") or PHASE[self.phase]["nruns"]))

        def run_rec(self, jid: str, k: int):
            return guarded_json(self.runs_base / jid / f"run{k}" / "run.json")

        def build_manifest(self) -> None:
            mf = self.manifest_file
            data = D.read_json(mf)
            if data is not None:
                self.log(f"{mf.name} exists (built {data.get('built_at')}): reused")
            else:
                self.log(f"building {mf.name} (PREREG_R6 3.{2 if self.phase == 'screen' else 3}; phase {self.phase})")
                rows = build_manifest(D, self.log, self.phase, self.only if mf != PHASE[self.phase]["manifest"] else None)
                D.write_csv(mf.with_suffix(".csv"), rows, MANIFEST_COLS)
                D.write_atomic(mf, D.dump({
                    "built_at": D.now(), "phase": self.phase, "prereg_r6_sha256": sha256_file(PREREG),
                    "prereg_r6_pins_sha256": sha256_file(PINS) if PINS.is_file() else None,
                    "pool_r6_sha256": sha256_file(POOL), "incumbents_r6_sha256": sha256_file(INCS),
                    "final_r6_sha256": sha256_file(FINAL_LIST) if (self.phase == "final" and FINAL_LIST.is_file()) else None,
                    "selection_s6_lock_sha256": sha256_file(SEL_S6) if (self.phase == "final" and SEL_S6.is_file()) else None,
                    "t2best5_driver_sha256": TE.DRIVER_SHA, "t2cons_eval_sha256": TE_SHA,
                    "t2cons_r6_eval_sha256": sha256_file(Path(__file__).resolve()), "t_c": "2026-10-31 01:09:17 CST",
                    "runs_per_job": PHASE[self.phase]["nruns"], "rows": rows}))
                for r in rows:
                    self.log(f"  {r['job']}: {r.get('role')} {r.get('root_kind')} {str(r.get('parquet_sha256'))[:8]} ckpt "
                             f"{str(r.get('training_sha256'))[:8]} tree {r.get('tree')}: {r['status']}")
            self.jobs = list((D.read_json(mf) or {}).get("rows", []))
            load_incumbents()  # fills INC_EXPORTS (the incumbents' earlier export records)
            for r in self.jobs:
                if r.get("status") != "ok":
                    self.log.first(("refused", r["job"]), f"{r['job']}: NOT evaluated: {r.get('status')}")

        def all_final(self) -> bool:
            for job in self.ok_jobs():
                exp = D.export_rec(ROOT, job["job"])
                if exp is None:
                    return False
                if exp.get("status") != "ok" or self.exports_only:
                    continue
                if any(self.run_rec(job["job"], k) is None for k in range(1, self.nruns(job) + 1)):
                    return False
            return True

        # -------------------------------------------------------------------------------- GPUs and lanes
        def gpu_rows(self, state: dict, mps: dict, ours: list[dict]) -> list[dict]:
            ours = list(ours) + [{"gpu": e["gpu"]} for e in self.ext]
            busy = {j["gpu"] for j in ours if j.get("gpu") is not None}
            lanes = {0: 0, 1: 0}
            for j in ours:
                if j.get("gpu") is not None:
                    lanes[D.numa(j["gpu"])] += 1
            out = []
            for g in sorted(state["gpus"]):
                st = state["gpus"][g]
                rest = self.rest_until.get(g, 0.0) - time.time()
                row = {"gpu": g, "uuid": st["uuid"], "free": st["free"], "napps": sum(1 for x in state["apps"] if x["gpu"] == g),
                       "lock_file": True, "shared_locks": len(lane_lock_files(g))}  # fmt: skip
                if g not in ALLOWED_GPUS:
                    why = "never (PREREG_R6 3.4: GPUs 0, 1, 2, 7 only)"
                elif g not in self.gpus:
                    why = "not in --gpus"
                elif g in mps:
                    why = f"MPS ({mps[g]})"
                elif g in busy:
                    why = "one of our (or a counted external) evals runs there"
                elif rest > 0:
                    why = f"resting {rest:.0f} s after our last eval (cooldown)"
                elif st["free"] < D.MIN_FREE_MB + D.EVAL_PEAK_MB:
                    why = f"free {st['free']} MiB < {D.MIN_FREE_MB}+{D.EVAL_PEAK_MB}"
                elif lanes[D.numa(g)] >= self.a.lane_max:
                    why = f"NUMA{D.numa(g)} lane has {self.a.lane_max} evals of ours / counted"
                else:
                    why = ""
                row.update(usable=not why, why=why or "usable")
                out.append(row)
            return out

        def pick_gpu(self, state: dict, mps: dict, ours: list[dict], take: bool):
            rows = self.gpu_rows(state, mps, ours)
            usable = sorted((r for r in rows if r["usable"]),
                            key=lambda r: (self.rest_until.get(r["gpu"], 0.0), r["napps"], -r["free"], r["gpu"]))  # fmt: skip
            if not usable:
                return "no usable GPU: " + "; ".join(f"gpu{r['gpu']} {r['why']}" for r in rows if r["gpu"] in self.gpus)
            if not take:
                return usable[0]["gpu"], None
            held = []
            for r in usable:
                got = take_lane(D, r["gpu"])
                if isinstance(got, str):
                    held.append(got)
                    continue
                return r["gpu"], got[0], got[1]
            return "every usable GPU's lane is held: " + "; ".join(held)

        # -------------------------------------------------------------------------------- launching
        def launch(self, kind, job, k, t, d, cmd, cwd, env, gpu, fd, lane, log_file, extra=None) -> None:
            extra = dict(extra or {})
            fds = list(extra.pop("fds", None) or ([] if fd is None else [fd]))
            cpus = D.EVAL_CORES[lane]
            prefix = ["nice", "-n", "10", "taskset", "-c", cpus]
            shown = D.show_command(prefix, cmd, cwd, env)
            with log_file.open("a") as stream:
                stream.write(f"[{D.now()}] {kind} {job['job']}{'' if k is None else f' run {k}'} try {t}"
                             f"{'' if gpu is None else f', gpu {gpu}'}, cpus {cpus}, lane locks {extra.get('lane_locks')}: {shown}\n")  # fmt: skip
                stream.flush()
                proc = subprocess.Popen(prefix + cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=tuple(fds))  # fmt: skip
            D.write_atomic(d / "start.json", D.dump({"kind": kind, "job": job["job"], "run": k, "try": t, "gpu": gpu, "cpus": cpus,
                                                     "lane": lane, "pid": proc.pid, "started_at": D.now(), "started_epoch": time.time(),
                                                     "cwd": str(cwd), "command": prefix + cmd, "shown": shown, "host": HOST,
                                                     **extra}))  # fmt: skip
            key = (kind, job["job"]) if kind == "export" else (kind, job["job"], k)
            self.running[key] = {"kind": kind, "job": job["job"], "run": k, "try": t, "dir": d, "proc": proc, "start": time.time(),
                                 "gpu": gpu, "fd": None, "fds": fds, "lane": lane, "overdue": False,
                                 **{x: extra[x] for x in ("tolerance",) if x in extra}}  # fmt: skip
            self.log(f"start {self.name(self.running[key])}" + ("" if gpu is None else f" on gpu {gpu}") + f" (pid {proc.pid})")

        def launch_exports(self) -> None:  # the pinned Driver.launch_exports with only its base changed (as driver c)
            for job in self.sorted_jobs():
                loads = self.lane_load("export")
                if min(loads) >= 1:
                    return
                jid = job["job"]
                if D.export_rec(ROOT, jid) is not None or self.busy(("export", jid)):
                    continue  # an incumbent's earlier export record always exists: never re-exported (PREREG_R6 3.1)
                if job.get("role") == "incumbent":
                    continue
                base = EXPORTS / jid
                done = D.tries(base)
                tol = D.EXPORT_TOL
                if done:
                    last = D.read_json(done[-1][1] / "try.json") or {}
                    if last.get("status") == "refused_tol" and D.fnum(last.get("tolerance")) == D.EXPORT_TOL:
                        tol = D.FALLBACK_TOL
                    else:
                        tol = D.fnum(last.get("tolerance")) or D.EXPORT_TOL
                        if not self.retry_due(done[-1][1]):
                            continue
                t = done[-1][0] + 1 if done else 1
                d = D.inside_root(base / f"try{t}")
                d.mkdir(parents=True, exist_ok=False)
                cmd, code, extra = D.export_command(job, d, tol)
                env = D.child_env(None, str(d), None)
                env.update(extra)
                lane = 0 if loads[0] <= loads[1] else 1
                self.launch("export", job, None, t, d, cmd, code, env, None, None, lane, d / "export.log", {"tolerance": tol})

        def launch_evals(self) -> None:
            if self.exports_only:  # PREREG_R6 3.2: exports only, never an evaluation
                return
            self.ext = external_evals(D)
            ours = [j for key, j in [*self.running.items(), *self.orphans.items()] if key[0] == "eval"]
            if len(ours) + len(self.ext) >= self.cap:
                if self.ext:
                    self.log.first(("ext", len(self.ext)), f"{len(self.ext)} external eval(s) counted: total cap {self.cap} reached")
                return
            ready = []
            jobs = self.sorted_jobs()
            for k in range(1, self.a.runs + 1):  # run-major (phase S) / trial-major (phase F)
                for job in jobs:
                    jid = job["job"]
                    if k > self.nruns(job):
                        continue
                    exp = D.export_rec(ROOT, jid)
                    if not exp or exp.get("status") != "ok" or self.run_rec(jid, k) is not None or self.busy(("eval", jid, k)):
                        continue
                    done = D.tries(self.runs_base / jid / f"run{k}")
                    if done and not self.retry_due(done[-1][1]):
                        continue
                    ready.append((job, k, exp, done))
            if not ready:
                return
            head = D.pids_headroom()
            if head is not None and head < D.MIN_FREE_PIDS:
                self.log.first(("pids", head // 500), f"pids headroom {head} < {D.MIN_FREE_PIDS}: no new eval")
                return
            try:
                state = D.smi()
                mps = D.mps_gpus(state)
            except Exception as error:  # noqa: BLE001
                self.log.first(("smi", repr(error)), f"nvidia-smi failed ({error!r}): no new eval")
                return
            for job, k, exp, done in ready:
                if len(ours) + len(self.ext) >= self.cap:
                    break
                choice = self.pick_gpu(state, mps, ours, take=True)
                if isinstance(choice, str):
                    self.log.first(("wait", re.sub(r"\d+", "N", choice)), f"{len(ready)} eval(s) ready, waiting: {choice}")
                    break
                gpu, fds, taken = choice
                jid, t = job["job"], (done[-1][0] + 1 if done else 1)
                w = D.inside_root(self.runs_base / jid / f"run{k}" / f"try{t}")
                try:
                    (w / f"ep{int(job['episode']):02d}").mkdir(parents=True, exist_ok=False)
                    (w / f"ep{int(job['episode']):02d}" / f"model_{job['iteration']}.safetensors").symlink_to(exp["file"])
                    cmd, code = D.eval_command(job, w, gpu)
                    env = D.child_env(gpu, str(w), Path(job["data_root"]))
                    extra = {"fds": fds, "lane_locks": taken, "gpu_uuid": state["gpus"][gpu]["uuid"],
                             "cooldown_s": self.a.cooldown, "phase": self.phase}  # fmt: skip
                    self.launch("eval", job, k, t, w, cmd, code, env, gpu, None, D.numa(gpu), w / "eval_all.log", extra)
                except Exception:
                    for fd in fds:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                    raise
                ours.append(self.running[("eval", jid, k)])

        # -------------------------------------------------------------------------------- finishing
        def reap(self) -> None:
            for key, j in list(self.running.items()):
                code = j["proc"].poll()
                if code is None:
                    continue
                del self.running[key]
                for fd in j.get("fds") or []:
                    os.close(fd)
                if j["kind"] == "eval" and j.get("gpu") is not None:
                    self.rest_until[j["gpu"]] = time.time() + self.a.cooldown
                self.finish(j, code, "")
                self.after_job()

        def check_orphans(self) -> None:
            for key, o in list(self.orphans.items()):
                if any(D.proc_alive(p, s) for p, s in o["procs"]):
                    continue
                del self.orphans[key]
                if o["kind"] == "eval" and isinstance(o.get("gpu"), int):
                    self.rest_until[o["gpu"]] = time.time() + self.a.cooldown
                self.finish(o, None, "finished while no driver was running (exit code unknown)")
                self.after_job()

        def recover(self) -> None:  # the pinned recover, on this driver's runs base, without clean slots
            live = D.live_jobs()
            for job in self.ok_jobs():
                jid = job["job"]
                slots = [] if job.get("role") == "incumbent" else [("export", None, EXPORTS / jid)]
                slots += [("eval", k, self.runs_base / jid / f"run{k}") for k in range(1, self.nruns(job) + 1)]
                for kind, k, base in slots:
                    for t, d in D.tries(base):
                        if (d / "try.json").is_file():
                            continue
                        key = (kind, jid) if kind == "export" else (kind, jid, k)
                        start = D.read_json(d / "start.json") or {}
                        if str(d) in live:
                            self.orphans[key] = {"kind": kind, "job": jid, "run": k, "try": t, "dir": d, "gpu": start.get("gpu"),
                                                 "procs": live[str(d)], "lane": start.get("lane", 0),
                                                 "tolerance": start.get("tolerance", D.EXPORT_TOL)}  # fmt: skip
                            self.log(f"{self.name(self.orphans[key])}: still running from a previous driver (pids "
                                     f"{[p for p, _ in live[str(d)]]}); recorded when it ends, never signalled")
                        else:
                            self.finish({"kind": kind, "job": jid, "run": k, "try": t, "dir": d,
                                         "tolerance": start.get("tolerance", D.EXPORT_TOL)}, None,
                                        "recovered after a driver restart (exit code unknown)")  # fmt: skip

        def finish_eval(self, job: dict, k: int, t: int, d: Path, code, remark: str) -> None:
            super().finish_eval(job, k, t, d, code, remark)
            try:
                self.record_seeds(job, k, t, d)
            except Exception as error:  # noqa: BLE001
                self.log(f"WARNING seed record failed for {job['job']} run {k} try {t}: {error!r}")

        def record_seeds(self, job: dict, k: int, t: int, d: Path) -> None:
            tj = D.read_json(d / "try.json") or {}
            sj = D.read_json(d / "start.json") or {}
            exp = D.export_rec(ROOT, job["job"]) or {}
            ef = Path(str(exp.get("file")))
            seeds = seeds_of(D, ef) if ef.is_file() else {"error": f"no exported file {ef}"}
            ep = int(job["episode"])
            js = guarded_json(d / "results" / f"episode_{ep:06d}.json") or {}
            dig_ = parquet_digests(d / "results" / f"episode_{ep:06d}.parquet") if (d / "results" / f"episode_{ep:06d}.parquet").is_file() else {}
            rec = {"job": job["job"], "episode": ep, "run": k, "try": t, "status": tj.get("status"),
                   "auc_own_input": (tj.get("metrics") or {}).get("auc"), "seeds": seeds, "gpu_index": sj.get("gpu"),
                   "gpu_uuid": sj.get("gpu_uuid"), "host": sj.get("host", HOST), "start": sj.get("started_at"),
                   "end": tj.get("finished_at"), "seconds": tj.get("seconds"), "stock_commit": D.STOCK_COMMIT,
                   "parquet_digests": dig_, "world_count(json)": js.get("world_count"), "exported_file": str(ef),
                   "exported_file_sha256": exp.get("sha256"), "lane_locks": sj.get("lane_locks"),
                   "runner_command": sj.get("shown"),
                   "note": "the official path has no selectable seed (PREREG_R6 3.5): every run uses these defaults"}  # fmt: skip
            D.write_atomic(d / "seeds.json", D.dump(rec))
            row = {"job": rec["job"], "episode": ep, "run": k, "try": t, "status": rec["status"], "auc_own_input": rec["auc_own_input"],
                   **{c: seeds.get(c) for c in SEED_COLS}, "gpu_index": rec["gpu_index"], "gpu_uuid": rec["gpu_uuid"],
                   "host": rec["host"], "start": rec["start"], "end": rec["end"], "seconds": rec["seconds"],
                   "stock_commit": rec["stock_commit"], "eval_script_sha256(parquet)": dig_.get("eval_script_sha256"),
                   "eval_code_sha256(parquet)": dig_.get("eval_code_sha256"),
                   "checkpoint_sha256(parquet)": dig_.get("checkpoint_sha256"), "world_count(json)": rec["world_count(json)"],
                   "exported_file_sha256": rec["exported_file_sha256"], "lane_locks": ";".join(rec["lane_locks"] or []),
                   "runner_command": rec["runner_command"]}  # fmt: skip
            new = not self.seeds_tsv.is_file()
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=SEED_TSV_COLS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
            D.append_line(self.seeds_tsv, buf.getvalue().rstrip("\n"))

        # -------------------------------------------------------------------------------- reporting
        def summary(self) -> None:
            if self.write_summary_:
                try:
                    write_summary(D, self.jobs, self.phase)
                except Exception as error:  # noqa: BLE001
                    self.log.first(("summary", repr(error)), f"WARNING summary failed: {error!r}")
            self.last_summary = time.time()

        def progress(self) -> dict:
            ex = {"ok": 0, "refused": 0, "failed": 0, "pending": 0}
            ev = {"ok": 0, "failed": 0, "pending": 0, "no_export": 0}
            for job in self.ok_jobs():
                exp = D.export_rec(ROOT, job["job"])
                st = (exp or {}).get("status")
                ex["ok" if st == "ok" else "pending" if exp is None else "refused" if "refused" in str(st) or "incompatible" in str(st) else "failed"] += 1
                for k in range(1, self.nruns(job) + 1):
                    r = self.run_rec(job["job"], k)
                    ev[("ok" if r.get("status") == "ok" else "failed") if r else ("no_export" if exp and st != "ok" else "pending")] += 1
            return {"phase": self.phase, "jobs": len(self.ok_jobs()),
                    "jobs_refused": len(self.jobs) - len([j for j in self.jobs if j.get("status") == "ok"]),
                    "runs_per_job": min(self.a.runs, PHASE[self.phase]["nruns"]), "exports": ex, "evals": ev,
                    "exports_only": self.exports_only, "clean": "not run (PREREG_R6 7)"}  # fmt: skip

        def heartbeat(self, phase: str) -> None:
            prog = self.progress()
            mean_s = sum(self.durations) / len(self.durations) if self.durations else None
            jobs = [{"kind": j["kind"], "job": j["job"], "run": j.get("run"), "try": j["try"], "gpu": j.get("gpu"),
                     "pid": j["proc"].pid, "elapsed_s": round(time.time() - j["start"]), "overdue": j["overdue"]}
                    for j in self.running.values()]  # fmt: skip
            jobs += [{"kind": o["kind"], "job": o["job"], "run": o.get("run"), "try": o["try"], "gpu": o.get("gpu"),
                      "orphan_pids": [p for p, _ in o["procs"]]} for o in self.orphans.values()]
            par = max(1, min(self.cap, len([1 for k in self.running if k[0] == "eval"]) or 1))
            D.write_atomic(self.hb_file, D.dump({
                "updated_at": D.now(), "updated_epoch": time.time(), "pid": os.getpid(), "state": phase, **prog, "running": jobs,
                "external_evals_counted": self.ext, "resting": {g: round(v - time.time()) for g, v in self.rest_until.items() if v > time.time()},
                "mean_eval_seconds": mean_s, "eta_hours": round(mean_s * prog["evals"]["pending"] / par / 3600, 2) if mean_s else None,
                "halted": self.halted, "stop_file": (ROOT / "STOP").exists(), "runs_base": str(self.runs_base)}))  # fmt: skip

    return R6Driver


# ================================================================================================ integrity baseline
def seed_baseline(D, log) -> None:
    """PREREG_R6 3.1: the round-6 baseline = a byte copy of round-1 integrity/baseline.json if today's trees equal it."""
    dst = ROOT / "integrity" / "baseline.json"
    if dst.is_file():
        return
    r1 = R1 / "integrity" / "baseline.json"
    old = guarded_json(r1)
    diff = D.integrity_diff(old or {}, D.integrity())
    if old is not None and not diff:
        D.inside_root(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r1, dst)
        log(f"integrity baseline: byte copy of round-1 {r1} (kit + stock trees unchanged since round 1)")
    else:
        D.write_atomic(ROOT / "integrity" / "r1_baseline_diff.json", D.dump({"checked_at": D.now(), "differences": diff[:200]}))
        log(f"integrity: today's kit / stock trees differ from round 1's baseline ({len(diff)} entries; "
            "integrity/r1_baseline_diff.json); the pinned driver takes a fresh baseline")


# ================================================================================================ commands
def plan(D, a) -> int:
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"t2cons_r6_eval PLAN-ONLY DRY RUN, phase {a.phase}, {D.now()}: nothing is exported, evaluated, locked or signalled. "
      "Written: dryrun/r6_eval_plan_*.txt and the read-only hull check's preflight/hulls_*.")
    fatal, info = check_pins(D, require_pins_file=False)
    f2, i2 = D.preflight()
    fatal += f2
    for x in info + i2:
        p(f"preflight: {x}")
    for x in fatal:
        p(f"preflight FAILED: {x}")
    if a.phase == "final":
        data, problems = final_list_check()
        p("phase F finalists: " + ("; ".join(problems) + " -> plan covers the 30 incumbents only" if problems else
                                   f"{len(data.get('finalists', []))} from final_r6.json"))  # fmt: skip
        rows = build_manifest(D, D.Logger(None), "final", None if not problems else {x["job"] for x in load_incumbents().values()})
    else:
        rows = build_manifest(D, D.Logger(None), "screen")
    n_ok = 0
    for r in rows:
        n_ok += r["status"] == "ok"
        des = r.get("designation_evidence") or {}
        p(f"{r['job']} [{r.get('role')}, {r.get('runs')} runs, {r.get('owner')}]: {r.get('root_kind')} tree {r.get('tree')} "
          f"parquet {str(r.get('parquet_sha256'))[:12]} ckpt {str(r.get('training_sha256'))[:16]} "
          f"{'export ' + str(r.get('export_sha256'))[:16] + ' (reused)' if r.get('role') == 'incumbent' else ''} "
          f"designation {des.get('rule')}: {r['status']}")
        if r["status"] != "ok":
            for n in r.get("notes", []):
                p(f"    note: {n}")
    p(f"jobs ok {n_ok} of {len(rows)}; hulls ok {sum(1 for r in rows if (r.get('hulls') or {}).get('ok'))}")
    old = guarded_json(R1 / "integrity" / "baseline.json")
    diff = D.integrity_diff(old or {}, D.integrity())
    p(f"integrity: kit + stock trees vs round-1 baseline: {'UNCHANGED' if not diff else f'{len(diff)} differences: {diff[:10]}'}")
    try:
        ok_rows = [r for r in rows if r["status"] == "ok"]
        if ok_rows:
            r0 = ok_rows[0]
            f0 = Path(D.export_rec(ROOT, r0["job"])["file"]) if r0.get("role") == "incumbent" else Path(r0["training_checkpoint"])
            p(f"seed record check on {f0.name} ({r0['job']}): {seeds_of(D, f0)['seed_signature']}")
    except Exception as error:  # noqa: BLE001
        p(f"seed record check failed: {error!r}")
    try:
        state = D.smi()
        mps = D.mps_gpus(state)
        drv = make_driver(D)(a, D.Logger(None), a.phase, PHASE[a.phase]["runs"], None, PHASE[a.phase]["hb"],
                             PHASE[a.phase]["seeds"], False)
        drv.ext = external_evals(D)
        p(f"external evals counted: {drv.ext}")
        for g in drv.gpu_rows(state, mps, []):
            p(f"gpu {g['gpu']}: free {g['free']} MiB, apps {g['napps']}, shared lane lock files {g['shared_locks']}: {g['why']}")
        for x in lane_alias_report(ALLOWED_GPUS):  # AMENDMENTS_R2_c C1
            p(x)
    except Exception as error:  # noqa: BLE001
        p(f"gpu check failed: {error!r}")
    ok = not fatal and all(r["status"] == "ok" for r in rows)
    p(f"plan: {'OK' if ok else 'PROBLEMS'}")
    D.write_atomic(ROOT / "dryrun" / f"r6_eval_plan_{a.phase}_{D.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


def status(D, phase: str) -> int:
    P = PHASE[phase]
    load_incumbents()
    hb = D.read_json(P["hb"]) or {}
    print(f"phase {phase}: heartbeat {hb.get('updated_at')} state {hb.get('state')} exports {hb.get('exports')} evals "
          f"{hb.get('evals')} eta_h {hb.get('eta_hours')} halted {hb.get('halted')} stop {hb.get('stop_file')} external "
          f"{len(hb.get('external_evals_counted') or [])}")
    for j in hb.get("running", []):
        print(f"  running {j}")
    for r in (D.read_json(P["manifest"]) or {}).get("rows", []):
        exp = D.export_rec(ROOT, r["job"]) or {}
        aucs, fl = [], 0
        for k in range(1, int(r.get("runs") or P["nruns"]) + 1):
            rr = guarded_json(P["runs"] / r["job"] / f"run{k}" / "run.json")
            if rr is None:
                aucs.append("-")
            elif rr.get("status") == "ok":
                m = rr["metrics"]
                aucs.append(f"{m['auc']:.3f}")
                fl += int(m["sp_sr"] == 1 and m["mp_sr"] == 1)
            else:
                aucs.append("FAIL")
        print(f"  {r['job']:<26} {str(r.get('role')):<9} {str(r.get('root_kind')):<6} export {exp.get('status', 'pending'):<10} "
              f"{str(exp.get('tolerance') or ''):<7} own-input runs {' '.join(aucs)}  floor {fl}"
              + ("" if r.get("status") == "ok" else f"  ** {r.get('status')}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=("screen", "final"), required=True)
    g = r.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="plan only (no export, no evaluation)")
    g.add_argument("--dry-run-eval", action="store_true", help=f"one official evaluation of {DRY_JOB} (per phase) into dryrun/")
    g.add_argument("--exports-only", action="store_true", help="every screen job's single export, no evaluation (PREREG_R6 3.2)")
    r.add_argument("--gpus", default="0,1,2,7")  # PREREG_R6 3.4
    r.add_argument("--max-parallel", type=int, default=4)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--cooldown", type=float, default=35.0)
    r.add_argument("--poll", type=float, default=10.0)
    for name in ("summary", "status"):
        s = sub.add_parser(name)
        s.add_argument("--phase", choices=("screen", "final"), required=True)
    a = ap.parse_args()
    D = load_driver()
    if a.cmd == "run":
        if not 1 <= a.max_parallel <= 4 or not 1 <= a.lane_max <= 2 or a.cooldown < 35.0:
            ap.error("PREREG_R6 3.4: --max-parallel <= 4, --lane-max <= 2, --cooldown >= 35")
        P = PHASE[a.phase]
        if a.exports_only and a.phase != "screen":
            ap.error("--exports-only is a phase-S mode (incumbents are never re-exported)")
        if a.dry_run:
            a.runs = P["nruns"]
            return plan(D, a)
        if a.dry_run_eval:
            a.runs = 1
            job = DRY_JOB[a.phase]
            base = D.inside_root(ROOT / "dryrun" / f"eval_{a.phase}_{D.stamp()}")
            base.mkdir(parents=True, exist_ok=False)
            log = D.Logger(base / "driver.log")
            seed_baseline(D, log)
            drv = make_driver(D)(a, log, a.phase, base, {job}, base / "heartbeat.json", base / "seeds_dryrun.tsv", False,
                                 manifest_file=base / "manifest_dryrun.json")  # fmt: skip
            rc = drv.run()
            rr = guarded_json(base / job / "run1" / "run.json") or {}
            D.write_atomic(base / "result.json", D.dump({
                "finished_at": D.now(), "driver_rc": rc, "phase": a.phase, "job": job, "export": D.export_rec(ROOT, job),
                "run": {k: rr.get(k) for k in ("status", "try", "gpu", "seconds", "metrics", "problems", "checkpoint_sha256",
                                               "reference_parquet_sha256", "work_dir")},
                "note": "PREREG_R6 3.6: plumbing test (phase S: the K3CA final, first K3CA job of these drivers, fc_r2 export "
                        "kept as the job's single export; phase F: an incumbent with its earlier export reused); a dry run "
                        "is NOT a round-6 run or try and decides nothing"}))  # fmt: skip
            return rc
        a.runs = P["nruns"]
        log = D.Logger(OUT_LOG)
        log(f"t2cons_r6_eval.py (PREREG_R6) start phase {a.phase}{' EXPORTS ONLY' if a.exports_only else ''}: --gpus {a.gpus}; "
            f"allowed {ALLOWED_GPUS}")
        for x in lane_alias_report(ALLOWED_GPUS):  # the merged aliases, once at start (as driver c)
            log(x)
        seed_baseline(D, log)
        return make_driver(D)(a, log, a.phase, P["runs"], None, P["hb"], P["seeds"], True, exports_only=a.exports_only).run()
    if a.cmd == "summary":
        load_incumbents()
        out = write_summary(D, list((D.read_json(PHASE[a.phase]["manifest"]) or {}).get("rows", [])), a.phase)
        print(json.dumps({r["job"]: {"n_ok": r.get("n_ok"), "auc_mean": r.get("auc_mean"), "floor": r.get("floor_sp1_mp1_runs")}
                          for r in out["jobs"]}, indent=1))  # fmt: skip
        return 0
    return status(D, a.phase)


if __name__ == "__main__":
    raise SystemExit(main())
