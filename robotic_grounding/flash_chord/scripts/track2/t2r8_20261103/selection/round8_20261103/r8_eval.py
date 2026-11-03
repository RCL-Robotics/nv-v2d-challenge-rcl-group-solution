#!/usr/bin/env python3
"""r8_eval.py: the evaluation driver of round 8 (fc_runs/submissions/t2_consensus_r8_20261103; PREREG_R8.md section 3).

Round 7's driver scripts/t2cons_r7_eval.py (1d66a2db...) is loaded from its verified bytes and re-bound to the round-8
folder; its code (and through it the pinned t2cons_eval.py 2cf42786 / t2best5_driver.py 24b9af39: export command and
tolerances, the official evaluation command, ok checks, retries, lane locks, never MPS, never signalling, seed records,
OVERDUE flagging, the kit / stock integrity halt) runs unchanged. Round-8 bookkeeping only:
  B1  paths: exports/, runs_s/ (phase S, 3 runs), runs_f/ (phase F, 10 runs), manifest_{s8,f8}, own_input_evals_{s8,f8}.csv,
      summary_{s8,f8}_own_input.*, seeds_{s8,f8}.tsv, eval_r8_driver.log, eval_r8_heartbeat_{s8,f8}.json, evallock/;
      round 7's evallock joins the shared lane-lock list;
  B2  jobs: phase S = the 'screen' rows of POOL_R8.json (round 7's build_screen; copies from other hosts take its
      relocation branch with RELOCATION_R8.tsv); phase F = the finalists of final_r8.json only (listed in
      SELECTION_S8.sha256): a new screen job with its phase-S export, or a reuse7 candidate rebuilt by a second, unbound
      instance of round 7's driver from round 7's records with its round-7 / round-6 export (never re-exported).
      Incumbents are not re-run (PREREG_R8 4);
  B3  GPUs 0, 4, 6 and 7 only (AMENDMENTS_R8 A1, 2026-11-03, before any round-8 evaluation; before A1: 6 and 7), plain,
      one evaluation of ours per GPU, at most 2 per NUMA eval lane, and only while nvidia-smi lists NO compute process
      on that GPU (PREREG_R8 3.5; an added condition on top of round 7's checks);
  B4  pin chain: PREREG_R8.md == PREREG_R8.sha256 == hard-coded; PREREG_R8_PINS.md pins this script and round 7's
      driver and every pin holds; LOCK_R8.sha256 holds; POOL_R8.sha256 / RELOCATION_R8.sha256 / TRACKS_R8.sha256 hold;
      round 7's own chain (its check_pins, unchanged) holds;
  B5  records: round-8 wording; selection uses only own-input and consensus-B scores (r8_select.py), never these
      summaries.

  python -I -B r8_eval.py run --phase screen|final [--gpus 0,4,6,7]        resumable; ROOT/STOP = no new starts
  python -I -B r8_eval.py run --phase screen|final --dry-run [--pool-root DIR]   plan only (dryrun/r8_eval_plan_*.txt)
  python -I -B r8_eval.py status|summary --phase screen|final
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r8_20261103"
R7ROOT = ME / "fc_runs/submissions/t2_consensus_r7_20261102"
E_FILE, E_SHA = ME / "scripts/t2cons_r7_eval.py", "1d66a2dba53287c6a320310c9524fd9ed00e50289852c9eccba74882028a9f2a"
PREREG_SHA = "bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c"
TZ8 = dt.timezone(dt.timedelta(hours=8))
T_C_TEXT = "2026-11-03 12:00:00 CST"
NRUNS_S, NRUNS_F = 3, 10
ALLOWED_GPUS_R8 = (0, 4, 6, 7)  # AMENDMENTS_R8 A1 (operational; was round 7's (6, 7)); never GPUs 1, 2, 3, 5
MINE = Path(__file__).resolve()


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


E = _load("t2cons_r7_eval_pinned_r8", E_FILE, E_SHA)  # re-bound to round 8 below
E7 = _load("t2cons_r7_eval_pinned_r7view", E_FILE, E_SHA)  # never re-bound: round 7's own records (reuse7 finalists)


def sha256_file(p) -> str:
    return E.sha256_file(p)


def _hash_or_missing(p: Path) -> str:
    return sha256_file(p) if Path(p).is_file() else "missing"


def r8_path(rel: str) -> Path:
    """PREREG_R8 path rule: '/...' absolute, everything else relative to the round-8 folder."""
    return Path(rel) if rel.startswith("/") else ROOT / rel


POOL_ROOT = ROOT


def configure(pool_root: Path) -> None:
    """B1: round-8 paths (rebinding of round 7's module names only). pool_root = ROOT, or a dryrun/trial_* folder for the
    plan-only check."""
    global POOL_ROOT
    POOL_ROOT = pool_root
    E.ROOT = ROOT
    E.ALLOWED_GPUS = ALLOWED_GPUS_R8  # AMENDMENTS_R8 A1: round 7's gpu_rows / load_driver read this module name at call time
    E.PREREG, E.PREREG_SHA_FILE, E.PREREG_SHA = ROOT / "PREREG_R8.md", ROOT / "PREREG_R8.sha256", PREREG_SHA
    E.PINS, E.PINS_SHA_FILE = ROOT / "PREREG_R8_PINS.md", ROOT / "PREREG_R8_PINS.sha256"
    E.EXPORTS = ROOT / "exports"
    E.RUNS_S, E.RUNS_F = ROOT / "runs_s", ROOT / "runs_f"
    E.POOL = pool_root / "POOL_R8.json"
    E.COPIES = pool_root / "COPIES_r8.tsv"
    E.SNAP = pool_root / "POOL_R8.sha256"
    E.TRACKS = pool_root / "inputs_r8/TRACKS_R8.sha256"
    E.RELOC_TSV, E.RELOC_LIST = pool_root / "RELOCATION_R8.tsv", pool_root / "RELOCATION_R8.sha256"
    E.FINAL_LIST, E.SEL_S7 = ROOT / "final_r8.json", ROOT / "SELECTION_S8.sha256"
    E.DRY_JOB = {}
    E.ZX_COPY_DIR, E.RELOC_DIR = pool_root / "zuxing_copies_r8", pool_root / "relocated_r8"
    E.PT.COPIES, E.PT.RELOC_DIR = E.ZX_COPY_DIR, E.RELOC_DIR
    E.OWN_LOCKDIR = ROOT / "evallock"
    if R7ROOT / "evallock" not in E.SHARED_LOCKDIRS:
        E.SHARED_LOCKDIRS = tuple(E.SHARED_LOCKDIRS) + (R7ROOT / "evallock",)
    E.OUT_LOG = ROOT / "eval_r8_driver.log"
    E.PHASE = {
        "screen": {"tag": "s8", "runs": ROOT / "runs_s", "nruns": NRUNS_S, "manifest": ROOT / "manifest_s8.json",
                   "manifest_csv": ROOT / "manifest_s8.csv", "evals": ROOT / "own_input_evals_s8.csv",
                   "summary": ROOT / "summary_s8_own_input", "hb": ROOT / "eval_r8_heartbeat_s8.json",
                   "seeds": ROOT / "seeds_s8.tsv"},
        "final": {"tag": "f8", "runs": ROOT / "runs_f", "nruns": NRUNS_F, "manifest": ROOT / "manifest_f8.json",
                  "manifest_csv": ROOT / "manifest_f8.csv", "evals": ROOT / "own_input_evals_f8.csv",
                  "summary": ROOT / "summary_f8_own_input", "hb": ROOT / "eval_r8_heartbeat_f8.json",
                  "seeds": ROOT / "seeds_f8.tsv"},
    }  # fmt: skip
    E.check_pins = check_pins8
    E.load_incumbents = lambda: {}
    E.load_refs = lambda: {}
    E.load_pool = load_pool8
    E.build_manifest = build_manifest8
    E.write_summary = write_summary8


# ================================================================================================ pins (B4)
def _pins(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64:
            out.append((f[1], f[2]))
    return out


def _listing(listing: Path, base: Path | None) -> tuple[int, list[str]]:
    bad, n = [], 0
    for line in listing.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        n += 1
        rel = parts[1].strip()
        p = Path(rel) if rel.startswith("/") or base is None else base / rel
        if _hash_or_missing(p) != parts[0]:
            bad.append(f"{listing.name}: {rel} changed")
    return n, bad


def check_pins8(D, require_pins_file: bool = True) -> tuple[list[str], list[str]]:
    fatal, info = [], []
    got = _hash_or_missing(E.PREREG)
    rec = (E.PREREG_SHA_FILE.read_text().split() or [""])[0] if E.PREREG_SHA_FILE.is_file() else ""
    (info if got == rec == PREREG_SHA else fatal).append(f"PREREG_R8.md {got[:16]} (sha file {rec[:16]}, hard-coded {PREREG_SHA[:16]})")
    mine = sha256_file(MINE)
    if E.PINS.is_file():
        got = sha256_file(E.PINS)
        rec = (E.PINS_SHA_FILE.read_text().split() or [""])[0] if E.PINS_SHA_FILE.is_file() else ""
        pins = _pins(E.PINS.read_text())
        if got != rec:
            fatal.append(f"PREREG_R8_PINS.md {got[:16]} != PREREG_R8_PINS.sha256 {rec[:16]}")
        if ("r8_eval.py", mine) not in pins:
            fatal.append(f"PREREG_R8_PINS.md does not pin this script {mine[:16]}")
        if (str(E_FILE), E_SHA) not in pins:
            fatal.append("PREREG_R8_PINS.md does not pin round 7's driver")
        bad = [rel for rel, h0 in pins if _hash_or_missing(r8_path(rel)) != h0]
        fatal += [f"PREREG_R8_PINS pin {rel} changed" for rel in bad]
        info.append(f"PREREG_R8_PINS.md {got[:16]}: {len(pins)} pins, {len(bad)} changed; pins this script {mine[:16]}")
    elif require_pins_file:
        fatal.append("PREREG_R8_PINS.md missing: this script must be pinned before it exports or evaluates")
    else:
        info.append(f"PREREG_R8_PINS.md not written yet (plan only); this script sha256 {mine}")
    lock = ROOT / "LOCK_R8.sha256"
    if lock.is_file():
        n, bad = _listing(lock, ROOT)
        fatal += bad
        info.append(f"LOCK_R8.sha256: {n} lines, {len(bad)} changed")
    elif require_pins_file:
        fatal.append("LOCK_R8.sha256 missing")
    for listing, base in ((E.SNAP, POOL_ROOT), (E.RELOC_LIST, POOL_ROOT), (E.TRACKS, None)):
        if not listing.is_file():
            (fatal if require_pins_file else info).append(f"{listing.name} missing (pool not frozen)")
            continue
        n, bad = _listing(listing, base)
        fatal += bad
        info.append(f"{listing.name}: {n} files, {len(bad)} changed")
    f7, i7 = E7.check_pins(D, True)  # round 7's own chain, unchanged (incl. round 1's chain)
    fatal += [f"round 7 chain: {x}" for x in f7]
    info.append(f"round 7 chain (its check_pins): {len(f7)} fatal, {len(i7)} lines verified")
    return fatal, info


# ================================================================================================ jobs (B2)
def load_pool8() -> list[dict]:
    data = E.guarded_json(E.POOL)
    if data is None:
        raise SystemExit(f"cannot read {E.POOL}")
    rows = list(data["rows"])
    for p in rows:
        if p.get("role") == "reuse7" and p.get("r7_export_record"):
            E.INC_EXPORTS[p["job"]] = Path(p["r7_export_record"])
    return rows


def build_screen8(D, p: dict, trk: dict, cop: dict, rel: dict) -> dict:
    """A round-8 screen job: round 7's build_screen; a copy from another host takes its relocation branch."""
    q = dict(p)
    if p.get("relocated_file"):
        q["machine"] = "5090"  # round 7's branch for metadata-only relocated copies (RELOCATION_R8.tsv row re-verified)
    job = E.build_screen(D, q, trk, cop, rel)
    des = dict(job.get("designation_evidence") or {})
    des["rule"] = ("round-8 pool (PREREG_R8 2-3): " + ("our training run" if p["owner"] == "haoran" else
                   f"Zuxing's {p['machine']} run, sha256-verified copy" + (" + metadata-only relocated file" if p.get("relocated_file") else "")))  # fmt: skip
    des["routes"] = p.get("routes")
    job.update(machine=p["machine"], host_class=p.get("host_class"), routes=p.get("routes"), designation_evidence=des,
               stage="round 8 phase S (PREREG_R8 3.3)", expected_sha256_from="POOL_R8.json (PREREG_R8 3)",
               arm=f"{p['run_label']} ({p['input']}) {p.get('slot')}")  # fmt: skip
    return job


def build_reused7(D, p: dict, fl: dict) -> dict:
    """A reuse7 finalist (PREREG_R8 3.4): rebuilt by round 7's own driver from round 7's records, its earlier export."""
    r7pool = {r["job"]: r for r in json.loads((R7ROOT / "inputs_r7/pool_r7.json").read_text())["rows"]}
    p7 = r7pool.get(p["r7_job"])
    if p7 is None:
        raise SystemExit(f"{p['job']}: no round-7 pool row {p.get('r7_job')}")
    if p7["role"] == "screen":
        job = E7.build_screen(D, p7, E7.track_records(), E7.copy_rows(E7.COPIES), E7.relocation_rows())
        rec_path = R7ROOT / "exports" / p7["job"] / "export.json"
    elif p7["role"] == "reuse":
        job = E7.build_reused(D, p7, E7.track_records(), E7.copy_rows(E7.R6_COPIES), "finalist")
        rec_path = Path(p7["r6_export_record"])
    else:
        raise SystemExit(f"{p['job']}: round-7 pool role {p7['role']} cannot be reused")
    problems = list(job.get("problems") or [])
    ex = E.guarded_json(rec_path) or {}
    if str(rec_path) != str(p.get("r7_export_record")) or sha256_file(rec_path) != p.get("r7_export_record_sha256"):
        problems.append("round-7 / round-6 export record != POOL_R8.json")
    if ex.get("status") != "ok" or ex.get("sha256") != p.get("r7_export_sha256") or ex.get("sha256") != fl.get("export_sha256"):
        problems.append("earlier export != POOL_R8.json / final_r8.json")
    if job.get("training_sha256") != p.get("r7_training_sha256") or job.get("parquet_sha256") != p.get("parquet_sha256"):
        problems.append("round-7 training file / parquet != POOL_R8.json")
    E.INC_EXPORTS[job["job"]] = rec_path
    job.update(role="finalist", runs=NRUNS_F, stage="round 8 phase F (PREREG_R8 3.4)", pool_sha256=p["sha256"],
               export_record=str(rec_path), export_sha256=ex.get("sha256"), problems=problems,
               status="ok" if not problems else "refused: " + "; ".join(problems))  # fmt: skip
    return job


def build_manifest8(D, log, phase: str, only: set | None = None) -> list[dict]:
    trk = E.track_records()
    pool = load_pool8()
    cop = E.copy_rows(E.COPIES)
    rel = E.relocation_rows()
    if phase == "screen":
        rows = sorted((p for p in pool if p.get("role") == "screen"), key=lambda p: (int(p["episode"]), p["job"]))
        jobs = [build_screen8(D, p, trk, cop, rel) for p in rows if only is None or p["job"] in only]
        log(f"phase S job list from POOL_R8.json: {len(jobs)} new screen jobs x {NRUNS_S} runs (PREREG_R8 3.3)")
    else:
        data, problems = E.final_list_check()
        if problems:
            raise SystemExit("phase F needs the frozen phase-S selection: " + "; ".join(problems))
        by = {p["job"]: p for p in pool}
        jobs = []
        for fl in data.get("finalists", []):
            p = by.get(fl["job"])
            if p is None:
                raise SystemExit(f"finalist {fl['job']} not in POOL_R8.json")
            if p["role"] == "screen":
                job = build_screen8(D, p, trk, cop, rel)
                job.update(role="finalist", runs=NRUNS_F, stage="round 8 phase F (PREREG_R8 3.4)")
                exp = E.guarded_json(E.EXPORTS / fl["job"] / "export.json") or {}
                if exp.get("status") != "ok" or exp.get("sha256") != fl.get("export_sha256"):
                    job["problems"].append("phase-S export record != final_r8.json")
                job["status"] = "ok" if not job["problems"] else "refused: " + "; ".join(job["problems"])
            elif p["role"] == "reuse7":
                job = build_reused7(D, p, fl)
            else:
                raise SystemExit(f"finalist {fl['job']} has pool role {p['role']}")
            jobs.append(job)
        jobs.sort(key=lambda j: (int(j["episode"]), j["job"]))
        log(f"phase F job list: {len(jobs)} finalists x {NRUNS_F} runs (PREREG_R8 3.4; incumbents are not re-run)")
    ok = [r for r in jobs if r["status"] == "ok"]
    hulls = D.hull_check(ok, log)
    for r in ok:
        h = hulls.get(r["parquet"], {"ok": False, "missing": ["not checked"]})
        r["hulls"] = h
        if not h["ok"]:
            r["status"] = f"hull_missing: {h['missing']}"
    return jobs


# ================================================================================================ summary (B5)
def write_summary8(D, jobs: list[dict], phase: str) -> dict:
    """The own-input evaluation table the scorer reads (round 7's EVAL_COLS rows from run.json, unchanged) and a short
    per-job summary."""
    P = E.PHASE[phase]
    nmax = P["nruns"]
    eval_rows, job_rows = [], []
    for j in jobs:
        jid = j["job"]
        base = {"job": jid, "episode": j["episode"], "role": j.get("role"), "source": j.get("source"), "input": j.get("root_kind"),
                "iteration": j.get("iteration"), "checkpoint_sha256": j.get("training_sha256")}  # fmt: skip
        if j.get("status") != "ok":
            job_rows.append({**base, "export_status": j.get("status")})
            continue
        exp = D.export_rec(ROOT, jid) or {}
        recs, failed = {}, []
        for k in range(1, nmax + 1):
            r = E.guarded_json(P["runs"] / jid / f"run{k}" / "run.json")
            if r is None:
                continue
            m = r.get("metrics") or {}
            sd = E.guarded_json(P["runs"] / jid / f"run{k}" / f"try{r.get('try')}" / "seeds.json") or {}
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
                         "export_tolerance": exp.get("tolerance"), "export_sha256": exp.get("sha256", ""), "n_ok": len(recs),
                         "failed_runs": " ".join(map(str, failed)), "floor_sp1_mp1_runs": floor, "runs_planned": nmax,
                         **D.job_stats_row(j, recs, nmax)})  # fmt: skip
    cols = ["job", "episode", "role", "source", "input", "parquet_sha256", "iteration", "checkpoint_sha256", "export_status",
            "export_tolerance", "export_sha256", "runs_planned", "n_ok", "failed_runs", "floor_sp1_mp1_runs"]  # fmt: skip
    for m in D.METRICS:
        cols += [f"{m}_r{k}" for k in range(1, nmax + 1)] + [f"{m}_{s}" for s in ("mean", "sd", "min", "max")]
    D.write_csv(P["evals"], eval_rows, E.EVAL_COLS)
    D.write_csv(P["summary"].with_suffix(".csv"), job_rows, cols)
    D.write_atomic(P["summary"].with_suffix(".json"), D.dump({
        "generated_at": D.now(), "phase": phase, "runs_per_job": nmax, "jobs": job_rows,
        "note": "own input = each policy against its OWN Tier-2-derived training parquet (official kit evaluator 160d24f5, "
                "stock 8c778f9, frame 0, explicit reset, world count 1). floor = runs with SP-SR = MP-SR = 1. The selection "
                "uses own-input gate + consensus-B scores computed by r8_select.py, not this table."}))  # fmt: skip
    md = [f"# t2_consensus round 8, phase {phase}: official own-input evaluations ({D.now()})", "",
          "Own input = each policy against its OWN training parquet. The selection (r8_select.py) uses the own-input gate "
          "and consensus-B scores.", "",
          "| job | role | input | export | ok runs | AUC mean | min | max | SP-SR | MP-SR | MPPE cm | floor |",
          "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in job_rows:
        md.append(f"| {r['job']} | {r.get('role', '')} | {r.get('input', '')} | {r.get('export_status', '')} | {r.get('n_ok', '')} | "
                  f"{D.fv(r.get('auc_mean'))} | {D.fv(r.get('auc_min'))} | {D.fv(r.get('auc_max'))} | "
                  f"{D.fv(r.get('sp_sr_mean'), '.2f')} | {D.fv(r.get('mp_sr_mean'), '.2f')} | {D.fv(r.get('mppe_mean'), '.2f')} | "
                  f"{r.get('floor_sp1_mp1_runs', '')}/{r.get('n_ok', '')} |")  # fmt: skip
    D.write_atomic(P["summary"].with_suffix(".md"), "\n".join(md) + "\n")
    return {"jobs": job_rows}


# ================================================================================================ driver (B2, B3)
PROGRESS_KEYS = ("phase", "jobs", "jobs_refused", "runs_per_job", "exports", "evals", "exports_only")
END_LINE = re.compile(r"every export, evaluation, \w+ diagnostic and Stage-B item is final")


def logger8(D, file: Path | None):
    class Logger8(D.Logger):
        def __call__(self, message: str) -> None:
            super().__call__(END_LINE.sub("every export and evaluation is final", message))

    return Logger8(file)


def make_driver8(D):
    Base = E.make_driver(D)  # round 7's R7Driver (pinned ConsDriver -> t2best5 Driver)

    class R8Driver(Base):
        def build_manifest(self) -> None:
            mf = self.manifest_file
            data = D.read_json(mf)
            if data is not None:
                self.log(f"{mf.name} exists (built {data.get('built_at')}): reused")
            else:
                self.log(f"building {mf.name} (PREREG_R8 3.{3 if self.phase == 'screen' else 4}; phase {self.phase})")
                rows = build_manifest8(D, self.log, self.phase, self.only if mf != E.PHASE[self.phase]["manifest"] else None)
                D.write_csv(mf.with_suffix(".csv"), rows, E.MANIFEST_COLS)
                D.write_atomic(mf, D.dump({
                    "built_at": D.now(), "phase": self.phase, "prereg_r8_sha256": _hash_or_missing(E.PREREG),
                    "prereg_r8_pins_sha256": _hash_or_missing(E.PINS), "pool_r8_sha256": _hash_or_missing(E.POOL),
                    "pool_r8_list_sha256": _hash_or_missing(E.SNAP), "relocation_r8_sha256": _hash_or_missing(E.RELOC_TSV),
                    "final_r8_sha256": _hash_or_missing(E.FINAL_LIST) if self.phase == "final" else None,
                    "selection_s8_lock_sha256": _hash_or_missing(E.SEL_S7) if self.phase == "final" else None,
                    "t2best5_driver_sha256": E.TE.DRIVER_SHA, "t2cons_eval_sha256": E.TE_SHA, "t2cons_r7_eval_sha256": E_SHA,
                    "r8_eval_sha256": sha256_file(MINE), "t_c": T_C_TEXT, "runs_per_job": E.PHASE[self.phase]["nruns"],
                    "rows": rows}))  # fmt: skip
                for r in rows:
                    self.log(f"  {r['job']}: {r.get('role')} {r.get('root_kind')} {str(r.get('parquet_sha256'))[:8]} ckpt "
                             f"{str(r.get('training_sha256'))[:8]} tree {r.get('tree')}: {r['status']}")
            self.jobs = list((D.read_json(mf) or {}).get("rows", []))
            load_pool8()
            for r in self.jobs:
                if r.get("status") != "ok":
                    self.log.first(("refused", r["job"]), f"{r['job']}: NOT evaluated: {r.get('status')}")
            if self.phase == "final":  # reuse7 finalists: their earlier export records (never re-exported)
                for r in self.jobs:
                    if r.get("export_record") and r.get("status") == "ok" and r["job"] not in E.INC_EXPORTS:
                        E.INC_EXPORTS[r["job"]] = Path(r["export_record"])

        def progress(self) -> dict:
            out = super().progress()
            return {k: out[k] for k in PROGRESS_KEYS if k in out}

        def gpu_rows(self, state: dict, mps: dict, ours: list[dict]) -> list[dict]:
            rows = super().gpu_rows(state, mps, ours)
            for r in rows:
                if r["gpu"] not in ALLOWED_GPUS_R8:
                    r.update(usable=False, why="never (PREREG_R8 3.5 / AMENDMENTS_R8 A1: GPUs 0, 4, 6 and 7 only)")
                elif r["usable"] and r["napps"] > 0:
                    r.update(usable=False, why=f"{r['napps']} compute process(es) on this GPU (PREREG_R8 3.5: only while "
                                               "no other compute process is listed)")  # fmt: skip
            return rows

    return R8Driver


# ================================================================================================ commands
def plan8(D, a) -> int:
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"r8_eval PLAN-ONLY DRY RUN, phase {a.phase}, {D.now()}, pool {POOL_ROOT}: nothing is exported, evaluated, locked or "
      "signalled. Written: dryrun/r8_eval_plan_*.txt and the read-only hull check's preflight/hulls_*.")
    fatal, info = check_pins8(D, require_pins_file=False)
    f2, i2 = D.preflight()
    fatal += f2
    for x in info + i2:
        p(f"preflight: {x}")
    for x in fatal:
        p(f"preflight FAILED: {x}")
    rows = []
    if a.phase == "final" and not E.SEL_S7.is_file():
        p("phase F: SELECTION_S8.sha256 absent (expected before phase S): no finalist to plan")
    else:
        rows = build_manifest8(D, D.Logger(None), a.phase)
    n_ok = 0
    for r in rows:
        n_ok += r["status"] == "ok"
        des = r.get("designation_evidence") or {}
        p(f"{r['job']} [{r.get('role')}, {r.get('runs')} runs, {r.get('owner')}, {r.get('machine')}]: {r.get('root_kind')} tree "
          f"{r.get('tree')} parquet {str(r.get('parquet_sha256'))[:12]} ckpt {str(r.get('training_sha256'))[:16]} "
          f"designation {des.get('rule')} {des.get('routes') or ''}: {r['status']}")  # fmt: skip
    p(f"jobs ok {n_ok} of {len(rows)}; hulls ok {sum(1 for r in rows if (r.get('hulls') or {}).get('ok'))}")
    old = E.guarded_json(E.R1 / "integrity" / "baseline.json")
    diff = D.integrity_diff(old or {}, D.integrity())
    p(f"integrity: kit + stock trees vs round-1 baseline: {'UNCHANGED' if not diff else f'{len(diff)} differences: {diff[:10]}'}")
    try:
        ok_rows = [r for r in rows if r["status"] == "ok"]
        if ok_rows:
            f0 = Path(ok_rows[0]["training_checkpoint"])
            p(f"seed record check on {f0.name} ({ok_rows[0]['job']}): {E.seeds_of(D, f0)['seed_signature']}")
    except Exception as error:  # noqa: BLE001
        p(f"seed record check failed: {error!r}")
    try:
        state = D.smi()
        mps = D.mps_gpus(state)
        a.runs = E.PHASE[a.phase]["nruns"]
        drv = make_driver8(D)(a, D.Logger(None), a.phase, E.PHASE[a.phase]["runs"], None, E.PHASE[a.phase]["hb"],
                              E.PHASE[a.phase]["seeds"], False)  # fmt: skip
        drv.ext = E.external_evals(D)
        p(f"external evals counted: {drv.ext}")
        for g in drv.gpu_rows(state, mps, []):
            p(f"gpu {g['gpu']}: free {g['free']} MiB, apps {g['napps']}, shared lane lock files {g['shared_locks']}: {g['why']}")
        for x in E.lane_alias_report(E.ALLOWED_GPUS):
            p(x)
    except Exception as error:  # noqa: BLE001
        p(f"gpu check failed: {error!r}")
    ok = not fatal and all(r["status"] == "ok" for r in rows)
    p(f"plan: {'OK' if ok else 'PROBLEMS'}")
    D.write_atomic(ROOT / "dryrun" / f"r8_eval_plan_{a.phase}_{D.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


def status8(D, phase: str) -> int:
    P = E.PHASE[phase]
    load_pool8()
    hb = D.read_json(P["hb"]) or {}
    print(f"phase {phase}: heartbeat {hb.get('updated_at')} state {hb.get('state')} exports {hb.get('exports')} evals "
          f"{hb.get('evals')} eta_h {hb.get('eta_hours')} halted {hb.get('halted')} stop {hb.get('stop_file')}")
    for j in hb.get("running", []):
        print(f"  running {j}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=("screen", "final"), required=True)
    r.add_argument("--dry-run", action="store_true", help="plan only (no export, no evaluation)")
    r.add_argument("--pool-root", default=None, help="plan only: a dryrun/trial_* pool instead of the frozen POOL_R8")
    r.add_argument("--gpus", default="0,4,6,7")
    r.add_argument("--max-parallel", type=int, default=4)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--cooldown", type=float, default=10.0)
    r.add_argument("--poll", type=float, default=10.0)
    for name in ("summary", "status"):
        s = sub.add_parser(name)
        s.add_argument("--phase", choices=("screen", "final"), required=True)
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with python -I -B")
        return 2
    pool_root = ROOT
    if a.cmd == "run" and a.pool_root:
        pool_root = Path(a.pool_root).resolve()
        if not a.dry_run or not str(pool_root).startswith(str(ROOT / "dryrun") + "/"):
            ap.error("--pool-root is for --dry-run with a folder under dryrun/ only")
    configure(pool_root)
    D = E.load_driver()
    if a.cmd == "run":
        if not 1 <= a.max_parallel <= len(ALLOWED_GPUS_R8) or not 1 <= a.lane_max <= 2 or a.cooldown < 10.0:
            ap.error("PREREG_R8 3.5 / AMENDMENTS_R8 A1: --max-parallel <= 4 (one per allowed GPU), --lane-max <= 2, "
                     "--cooldown >= 10")
        if any(int(x) not in ALLOWED_GPUS_R8 or int(x) not in E.ALLOWED_GPUS for x in str(a.gpus).split(",") if x.strip()):
            ap.error("PREREG_R8 3.5 / AMENDMENTS_R8 A1: --gpus may name GPUs 0, 4, 6 and 7 only")
        P = E.PHASE[a.phase]
        a.runs = P["nruns"]
        a.exports_only = False
        if a.dry_run:
            return plan8(D, a)
        log = logger8(D, E.OUT_LOG)
        log(f"r8_eval.py (PREREG_R8) start phase {a.phase}: --gpus {a.gpus}; allowed {E.ALLOWED_GPUS}; only while no other "
            "compute process is listed on the GPU")
        for x in E.lane_alias_report(E.ALLOWED_GPUS):
            log(x)
        E.seed_baseline(D, log)
        return make_driver8(D)(a, log, a.phase, P["runs"], None, P["hb"], P["seeds"], True).run()
    if a.cmd == "summary":
        load_pool8()
        out = write_summary8(D, list((D.read_json(E.PHASE[a.phase]["manifest"]) or {}).get("rows", [])), a.phase)
        print(json.dumps({r["job"]: {"n_ok": r.get("n_ok"), "auc_mean": r.get("auc_mean")} for r in out["jobs"]}, indent=1))
        return 0
    return status8(D, a.phase)


if __name__ == "__main__":
    raise SystemExit(main())
