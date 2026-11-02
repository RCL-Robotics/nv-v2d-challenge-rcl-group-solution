#!/usr/bin/env python3
"""t2cons_eval.py: step S3 of fc_runs/submissions/t2_consensus_20261024 (PREREG section 6).

Five official own-input evaluations of each of the 5 new Zuxing designated checkpoints of PREREG section 4.2:
  ep05_zx_k3fix_26000, ep26_zx_k3fix_20000   (variants A and B)
  ep04_zx_g2_18000, ep19_zx_g2_20000, ep28_zx_g2_22000   (variant B only)

The protocol is t2_best5 PREREG sections 3.2-3.5 and 3.7, unchanged: this script loads scripts/t2best5_driver.py
read-only (its bytes must hash to the t2_best5 PREREG_PINS value 24b9af39...; it is compiled from those same bytes and
nothing is written next to it) and re-roots it at this folder. The copy, export, evaluation, ok checks, retries,
hull check, kit / stock tree check, GPU locks and lanes are that driver's own code. Differences (all PREREG section 6):
  - outputs here: zuxing_ckpts/, COPIES.tsv, exports/, runs/, manifest_new.json (+ .csv), own_input_evals_new.csv,
    summary_new_own_input.{csv,json,md}, eval_driver.log, eval_heartbeat.json;
  - NO per-run LOCAL clean diagnostic: track2_tier2_clean_scores.py is never called and no clean path is opened;
  - order: the variant-A jobs first (priority 1: run 1 of each before run 2, ...), then the B-only jobs (priority 2);
  - data root and training parquet = t2_best5's manifest row of the same run (snapshot copy inputs/t2_best5/manifest.json),
    re-hashed; Zuxing's run's hydra task.parquet must name the same file;
  - designation re-resolved from the snapshot copy of his metrics.csv (inputs/zuxing_metrics/, hashed in
    inputs/SNAPSHOT.sha256) at T_c = 2026-10-24 23:24:16 CST (status ok, stock_verified True, finished_at <= T_c; highest
    add_auc, ties lower mppe_cm, then earlier iteration); it must equal the PREREG 4.2 table (iteration + full sha256).
Scheduling (decides nothing): GPUs 0,1,2,5,6,7 only; shared flock fc_runs/queue_runin/evallock/gpu<g>.lock opened
read-only; <= 2 per NUMA eval lane and <= 4 in all, counting the t2_best5 driver's live evaluations as ours too; nice +
taskset on the eval-lane cores. Never signals any process; an eval past 2400 s is only flagged OVERDUE; ROOT/STOP = no new
starts.

  python -B scripts/t2cons_eval.py run [--dry-run] [--gpus 0,1,2,5,6,7] [--max-parallel 4] [--lane-max 2]
  python -B scripts/t2cons_eval.py summary | status
Background start: scripts/t2cons_eval_launch.sh
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_20261024"
B5 = ME / "fc_runs/submissions/t2_best5_20261024"  # only its live evals' /proc markers are looked at, never its files
DRIVER = ME / "scripts/t2best5_driver.py"
DRIVER_SHA = "24b9af39a2e2708641e7870447b12b6475c23aec9cd18ba71086768dd69cfc89"  # t2_best5 PREREG_PINS.md
ZU = Path("/root/autodl-tmp/zuxing.lu")  # READ ONLY: only the 5 designated model + rsi files are copied (sha256)
PREREG, PREREG_SHA = ROOT / "PREREG.md", ROOT / "PREREG.sha256"
SNAP = ROOT / "inputs"
TZ8 = dt.timezone(dt.timedelta(hours=8))
T_C = dt.datetime(2026, 10, 24, 23, 24, 16, tzinfo=TZ8).timestamp()
NRUNS = 5
FORBIDDEN = ("clean", "local_not_for_upload", "packed_")  # case-insensitive guard on every record this script reads
NEW = [
    {"job": "ep05_zx_k3fix_26000", "episode": 5, "zuxing_run": "2026-10-24-tier2-runin-k3fix-ep05-ep26", "kind": "k3fix",
     "iteration": 26000, "sha256": "4f925b2a34b93a540cdba278030d104355711d0165deec6f8ac6bcc97be574f6",
     "variants": "A,B", "same_run_job": "ep05_zx_k3fix_18000"},
    {"job": "ep26_zx_k3fix_20000", "episode": 26, "zuxing_run": "2026-10-24-tier2-runin-k3fix-ep05-ep26", "kind": "k3fix",
     "iteration": 20000, "sha256": "249ed85df2fcbf14103e86931cf883ce2f5e6da36aca659c9d0310b14c38d081",
     "variants": "A,B", "same_run_job": "ep26_zx_k3fix_16000"},
    {"job": "ep04_zx_g2_18000", "episode": 4, "zuxing_run": "2026-10-24-tier2-runin-g2", "kind": "g2",
     "iteration": 18000, "sha256": "91e1a200eb75a1b312f10f16019243ca21a1b1627a80129942791fe134579f78",
     "variants": "B", "same_run_job": "ep04_zx_g2_12000"},
    {"job": "ep19_zx_g2_20000", "episode": 19, "zuxing_run": "2026-10-24-tier2-runin-g2", "kind": "g2",
     "iteration": 20000, "sha256": "9f0f090efb6ae2848db74e053ee4a4983bbb2cff690ccc70deb5f0db5940f977",
     "variants": "B", "same_run_job": "ep19_zx_g2_18000"},
    {"job": "ep28_zx_g2_22000", "episode": 28, "zuxing_run": "2026-10-24-tier2-runin-g2", "kind": "g2",
     "iteration": 22000, "sha256": "8750a1427b4bf729ecd2061fe45d07dd756ef4722004a2f5688e84edd35b1b4f",
     "variants": "B", "same_run_job": "ep28_zx_g2_16000"},
]  # fmt: skip


# ================================================================================================ helpers
def sha256_file(file) -> str:
    h = hashlib.sha256()
    with Path(file).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard(p) -> Path:
    s = str(p).lower()
    if any(tok in s for tok in FORBIDDEN):
        raise PermissionError(f"t2cons_eval may not open {p}")
    return Path(p)


def guarded_json(p):
    try:
        return json.loads(guard(p).read_text())
    except (OSError, ValueError, TypeError):
        return None


def load_driver():
    """scripts/t2best5_driver.py compiled from the exact bytes whose sha256 is pinned, re-rooted at this folder."""
    src = DRIVER.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != DRIVER_SHA:
        raise SystemExit(f"{DRIVER} sha256 {got} != pinned {DRIVER_SHA}")
    mod = types.ModuleType("t2best5_driver_readonly")
    mod.__file__ = str(DRIVER)
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(DRIVER), "exec"), mod.__dict__)  # noqa: S102 (pinned bytes)
    mod.ROOT = ROOT
    mod.INVENTORY = SNAP / "t2_best5/inventory.json"
    mod.PREREG, mod.PREREG_SHA = PREREG, PREREG_SHA
    mod.PINS = mod.PINS_SHA = None
    mod.check_pins = lambda require_pins_file=True: check_pins(mod)
    return mod


# ================================================================================================ pins (PREREG 11)
def check_pins(D) -> tuple[list[str], list[str]]:
    fatal, info = [], []
    want = (PREREG_SHA.read_text().split() or [""])[0] if PREREG_SHA.is_file() else ""
    got = sha256_file(PREREG) if PREREG.is_file() else ""
    if not want or got != want:
        return [f"PREREG.md sha256 {got[:16]} != PREREG.sha256 {want[:16]}"], info
    info.append(f"PREREG.md sha256 {got[:16]} == PREREG.sha256")
    for line in PREREG.read_text().split("## 11. Pins", 1)[-1].splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin":
            p = ROOT / f[1] if f[1].startswith("inputs/") else ME / f[1]
        elif len(f) == 4 and f[0] == "pin" and f[1] == "kit":
            p = D.KIT / f[2]
            f = ["pin", "kit:" + f[2], f[3]]
        elif len(f) == 3 and f[0] == "commit":
            head = D.git_head(ME / f[1])
            (info if head == f[2] else fatal).append(f"commit {f[1]}: HEAD {head[:12]} (pinned {f[2][:12]})")
            continue
        else:
            continue
        h = sha256_file(p) if p.is_file() else "missing"
        (info if h == f[2] else fatal).append(f"pin {f[1]}: {h[:16]} (pinned {f[2][:16]})")
    # every file of the input snapshot
    n = 0
    for line in (SNAP / "SNAPSHOT.sha256").read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        p = SNAP / parts[1].strip()
        if p.name == "SNAPSHOT.sha256":
            continue
        h = sha256_file(p) if p.is_file() else "missing"
        n += 1
        if h != parts[0]:
            fatal.append(f"inputs/{parts[1].strip()}: {h[:16]} != SNAPSHOT.sha256 {parts[0][:16]}")
    info.append(f"inputs/SNAPSHOT.sha256: {n} files checked")
    info.append(f"scripts/t2best5_driver.py sha256 {DRIVER_SHA[:16]} (t2_best5 PREREG_PINS) loaded read-only")
    info.append(f"scripts/t2cons_eval.py sha256 {sha256_file(Path(__file__).resolve())}")
    return fatal, info


# ================================================================================================ designation (PREREG 4.2)
def snapshot_metrics(spec: dict) -> Path:
    return SNAP / "zuxing_metrics" / f"{spec['zuxing_run']}_eval_ep{spec['episode']:02d}__metrics.csv"


def designate(D, spec: dict) -> tuple[dict | None, list[str]]:
    problems = []
    src = snapshot_metrics(spec)
    rows = D.load_csv(guard(src))
    ok = [r for r in rows if r.get("status") == "ok" and str(r.get("stock_verified")) == "True"
          and str(r.get("episode")) == str(spec["episode"]) and D.fnum(r.get("add_auc")) is not None
          and (D.parse_time(r.get("finished_at")) or 9e18) <= T_C]  # fmt: skip
    if not ok:
        return None, [f"no ok stock-verified row by T_c in {src.name}"]
    def mppe(r):
        v = D.fnum(r.get("mppe_cm"))
        return 9e18 if v is None else v

    ok.sort(key=lambda r: (-float(r["add_auc"]), mppe(r), int(r["iteration"])))
    best = ok[0]
    if int(best["iteration"]) != spec["iteration"]:
        problems.append(f"snapshot designates model_{best['iteration']}, PREREG 4.2 says model_{spec['iteration']}")
    if best.get("source_sha256") != spec["sha256"]:
        problems.append(f"designated row source_sha256 {str(best.get('source_sha256'))[:16]} != PREREG {spec['sha256'][:16]}")
    text = PREREG.read_text()
    if not any(f"`{spec['job']}` (new)" in ln and spec["sha256"] in ln for ln in text.splitlines()):
        problems.append("PREREG 4.2 table has no line with this job and sha256")
    keep = ("iteration", "add_auc", "sp_sr", "mp_sr", "mppe_cm", "chord_sr", "export_max_rel_diff", "stock_verified",
            "source_sha256", "finished_at")  # fmt: skip
    return {"snapshot_file": str(src), "n_rows": len(rows), "n_eligible": len(ok), "row": {k: best.get(k) for k in keep}}, problems


# ================================================================================================ manifest
def base_rows() -> dict[str, dict]:
    data = guarded_json(SNAP / "t2_best5/manifest.json") or {}
    return {r["job"]: r for r in data.get("rows", [])}


def build_new(D, spec: dict, base: dict[str, dict], copy: bool) -> dict:
    ep, it, run = int(spec["episode"]), int(spec["iteration"]), spec["zuxing_run"]
    run_dir = ZU / "runs" / run / f"ep{ep:02d}"
    train = run_dir / f"model_{it}.safetensors"
    problems: list[str] = []
    notes: list[str] = []
    des, p2 = designate(D, spec)
    problems += p2
    b = base.get(spec["same_run_job"])
    job = {"job": spec["job"], "episode": ep, "family": "zuxing", "owner": "zuxing", "arm": f"Zuxing {run} ({spec['kind']})",
           "stage": "consensus PREREG 4.2 (new)", "variants": spec["variants"], "priority": 1 if "A" in spec["variants"] else 2,
           "provisional": True, "zuxing_run": run, "run_dir": str(run_dir), "iteration": it, "tree": "b39b69b",
           "designation": "Zuxing designated checkpoint at T_c (snapshot metrics.csv; PREREG 4.2)",
           "designation_evidence": des, "expected_sha256": spec["sha256"], "same_run_t2best5_job": spec["same_run_job"],
           "training_checkpoint_original": str(train)}  # fmt: skip
    if b is None:
        problems.append(f"t2_best5 manifest has no row {spec['same_run_job']}")
        b = {}
    elif os.path.normpath(b.get("run_dir", "")) != os.path.normpath(run_dir) or b.get("root_kind") != spec["kind"]:
        problems.append(f"t2_best5 row {spec['same_run_job']} is run {b.get('run_dir')} / {b.get('root_kind')}, not this run")
    job.update(object=b.get("object"), sequence_id=b.get("sequence_id"))
    if not train.is_file():
        problems.append(f"missing {train}")
        tsha = None
    else:
        tsha = D.sha256(train)
        if tsha != spec["sha256"]:
            problems.append(f"checkpoint sha256 {tsha[:16]} != PREREG 4.2 {spec['sha256'][:16]}")
    job["training_sha256"] = tsha
    # training parquet: t2_best5's row of the same run, re-hashed; his hydra task.parquet must name it
    pq = b.get("parquet")
    ov = D.read_overrides(run_dir)
    if ov.get("task.parquet"):
        if not pq or os.path.normpath(ov["task.parquet"]) != os.path.normpath(pq):
            problems.append(f"his overrides task.parquet {ov['task.parquet']} != t2_best5 row parquet {pq}")
        else:
            notes.append("hydra overrides task.parquet == t2_best5 row parquet")
    else:
        notes.append("no hydra/.hydra/overrides.yaml task.parquet: the t2_best5 row's parquet is used")
    lay = D.parquet_layout(Path(pq)) if pq else None
    if lay is None:
        problems.append(f"no / unexpected training parquet {pq}")
        job.update(problems=problems, notes=notes, status="refused: " + "; ".join(problems))
        return job
    parquet = Path(pq)
    data_root, seq = lay
    kind = D.root_kind(data_root)
    psha = D.sha256(parquet) if parquet.is_file() else None
    if kind != spec["kind"]:
        problems.append(f"data root kind {kind} != {spec['kind']}")
    if psha is None or psha != b.get("parquet_sha256"):
        problems.append(f"parquet sha256 {str(psha)[:16]} != t2_best5 row {str(b.get('parquet_sha256'))[:16]}")
    if b.get("sequence_id") and seq != b["sequence_id"]:
        problems.append(f"parquet sequence {seq} != t2_best5 row {b['sequence_id']}")
    manifest = D.MANIFEST_FOR.get(kind) if kind else None
    if manifest is None or D.manifest_episode_seq(manifest, ep) != seq:
        problems.append(f"manifest {manifest} does not map ep{ep:02d} to {seq}")
    fr = D.FRAME_REFS.get(kind) if kind else None
    job.update(parquet=str(parquet), parquet_sha256=psha, data_root=str(data_root), root_kind=kind, manifest=str(manifest),
               frame_refs=str(fr) if fr else None, sequence_id=seq)  # fmt: skip
    # copies (t2_best5 PREREG 3.2)
    dest = ROOT / "zuxing_ckpts" / run / f"ep{ep:02d}"
    rsi = run_dir / f"rsi_{it}.safetensors"
    job.update(training_checkpoint=str(dest / train.name), rsi_original=str(rsi) if rsi.is_file() else "")
    if copy and not problems:
        try:
            got = D.copy_verified(train, dest / train.name, tsha)
            D.record_copy(train, dest / train.name, got)
            sums = [f"{got}  {train.name}"]
            if rsi.is_file():
                rs = D.copy_verified(rsi, dest / rsi.name, None)
                D.record_copy(rsi, dest / rsi.name, rs)
                sums.append(f"{rs}  {rsi.name}")
                job["rsi_sha256"] = rs
            D.write_atomic(dest / "SHA256SUMS", "\n".join(sums) + "\n")
            D.write_atomic(dest / "SOURCE.json", D.dump({"copied_at": D.now(), "from": str(run_dir), "model": str(train),
                                                         "rsi": job["rsi_original"], "model_sha256": got,
                                                         "evidence": des}))  # fmt: skip
        except (OSError, RuntimeError) as error:
            problems.append(f"copy of Zuxing's checkpoint failed ({error})")
    elif not copy:
        notes.append("dry run: the copy is made when the manifest is built")
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


MANIFEST_COLS = ["job", "episode", "object", "family", "owner", "variants", "priority", "provisional", "iteration",
                 "training_sha256", "training_checkpoint", "training_checkpoint_original", "rsi_sha256", "tree", "parquet",
                 "parquet_sha256", "root_kind", "data_root", "manifest", "frame_refs", "same_run_t2best5_job", "status"]  # fmt: skip


def build_manifest(D, log, copy: bool) -> list[dict]:
    base = base_rows()
    rows = [build_new(D, s, base, copy) for s in NEW]
    ok = [r for r in rows if r["status"] == "ok"]
    hulls = D.hull_check(ok, log)
    for r in ok:
        h = hulls.get(r["parquet"], {"ok": False, "missing": ["not checked"]})
        r["hulls"] = h
        if not h["ok"]:
            r["status"] = f"hull_missing: {h['missing']}"
    return rows


# ================================================================================================ own-input summary
EVAL_COLS = ["job", "episode", "run", "status", "try", "gpu", "seconds", "auc", "sp_sr", "mp_sr", "mppe", "chord_sr",
             "export_sha256", "finished_at", "problems", "export_sha256_full", "results_parquet", "reference_parquet_sha256",
             "variants", "input"]  # fmt: skip


def write_summary(D, jobs: list[dict]) -> dict:
    eval_rows, job_rows = [], []
    for j in jobs:
        jid = j["job"]
        if j.get("status") != "ok":
            job_rows.append({"job": jid, "episode": j["episode"], "variants": j.get("variants"), "export_status": j.get("status")})
            continue
        exp = guarded_json(ROOT / "exports" / jid / "export.json") or {}
        recs, failed = {}, []
        for k in range(1, NRUNS + 1):
            r = guarded_json(ROOT / "runs" / jid / f"run{k}" / "run.json")
            if r is None:
                continue
            m = r.get("metrics") or {}
            eval_rows.append({"job": jid, "episode": j["episode"], "run": k, "status": r.get("status"), "try": r.get("try"),
                              "gpu": r.get("gpu"), "seconds": r.get("seconds"), **m, "chord_sr": r.get("chord_sr"),
                              "export_sha256": str(r.get("export_sha256"))[:16], "finished_at": r.get("finished_at"),
                              "problems": "; ".join(r.get("problems") or []), "export_sha256_full": r.get("export_sha256"),
                              "results_parquet": r.get("results_parquet"),
                              "reference_parquet_sha256": r.get("reference_parquet_sha256"), "variants": j.get("variants"),
                              "input": j.get("root_kind")})  # fmt: skip
            if r.get("status") == "ok":
                recs[k] = m
            else:
                failed.append(k)
        floor = sum(1 for k in recs if recs[k]["sp_sr"] == 1 and recs[k]["mp_sr"] == 1)
        his = ((j.get("designation_evidence") or {}).get("row")) or {}
        job_rows.append({"job": jid, "episode": j["episode"], "object": j.get("object"), "variants": j.get("variants"),
                         "input": j.get("root_kind"), "parquet_sha256": j.get("parquet_sha256"), "iteration": j["iteration"],
                         "checkpoint_sha256": j.get("training_sha256"), "export_status": exp.get("status", "pending"),
                         "export_tolerance": exp.get("tolerance"), "export_fallback": exp.get("fallback"),
                         "export_sha256": exp.get("sha256", ""), "export_max_rel_diff": exp.get("max_rel_diff"),
                         "n_ok": len(recs), "failed_runs": " ".join(map(str, failed)), "floor_sp1_mp1_runs": floor,
                         **D.job_stats_row(j, recs, NRUNS),
                         "zuxing_single_auc": his.get("add_auc"), "zuxing_single_sp_sr": his.get("sp_sr"),
                         "zuxing_single_mp_sr": his.get("mp_sr"), "zuxing_single_mppe": his.get("mppe_cm")})  # fmt: skip
    cols = ["job", "episode", "object", "variants", "input", "parquet_sha256", "iteration", "checkpoint_sha256",
            "export_status", "export_tolerance", "export_fallback", "export_sha256", "export_max_rel_diff", "n_ok",
            "failed_runs", "floor_sp1_mp1_runs"]  # fmt: skip
    for m in D.METRICS:
        cols += [f"{m}_r{k}" for k in range(1, NRUNS + 1)] + [f"{m}_{s}" for s in ("mean", "sd", "min", "max")]
    cols += ["closest_to_mean_run", "zuxing_single_auc", "zuxing_single_sp_sr", "zuxing_single_mp_sr", "zuxing_single_mppe"]
    D.write_csv(ROOT / "own_input_evals_new.csv", eval_rows, EVAL_COLS)
    D.write_csv(ROOT / "summary_new_own_input.csv", job_rows, cols)
    D.write_atomic(ROOT / "summary_new_own_input.json", D.dump({
        "generated_at": D.now(), "runs_per_job": NRUNS, "jobs": job_rows,
        "note": "own input = each policy against its OWN training parquet (official kit evaluator, stock 8c778f9, world "
                "count 1). Comparable only within one training parquet sha256. No clean number anywhere in this file."}))
    md = [f"# t2_consensus S3: 5x official own-input evaluations of the new jobs ({D.now()})", "",
          "Own input = each policy against its OWN training parquet (kit run_policy_evaluation.py + evaluate_policy.py "
          "160d24f5 on stock 8c778f9, frame 0, explicit reset, world count 1). Not a leaderboard number. No clean value "
          "was computed or read.", "",
          "| job | variants | input | ckpt sha256 | export | AUC runs 1-5 | mean | sd | min | max | SP-SR | MP-SR | MPPE cm "
          "| SP=MP=1 runs | closest | Zuxing single |",
          "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for r in job_rows:
        runs = " ".join(D.fv(r.get(f"auc_r{k}")) or "-" for k in range(1, NRUNS + 1))
        md.append(f"| {r['job']} | {r.get('variants')} | {r.get('input', '')} {str(r.get('parquet_sha256', ''))[:8]} | "
                  f"{str(r.get('checkpoint_sha256', ''))[:12]} | {r.get('export_status', '')} {r.get('export_tolerance') or ''} | "
                  f"{runs} | {D.fv(r.get('auc_mean'))} | {D.fv(r.get('auc_sd'))} | {D.fv(r.get('auc_min'))} | "
                  f"{D.fv(r.get('auc_max'))} | {D.fv(r.get('sp_sr_mean'), '.2f')} | {D.fv(r.get('mp_sr_mean'), '.2f')} | "
                  f"{D.fv(r.get('mppe_mean'), '.2f')} | {r.get('floor_sp1_mp1_runs', '')} | {r.get('closest_to_mean_run', '')} | "
                  f"{D.fv(r.get('zuxing_single_auc'))} |")
    D.write_atomic(ROOT / "summary_new_own_input.md", "\n".join(md) + "\n")
    return {"jobs": job_rows}


# ================================================================================================ the driver
def external_evals(D) -> list[dict]:
    """Live evaluations of the t2_best5 driver (its T2BEST5_JOB marker under t2_best5/runs/), counted as ours for the
    lane / total caps. Read from /proc only."""
    prefix = str(B5 / "runs") + "/"
    seen: dict[str, int] = {}
    for pid in D.all_pids():
        env = D.proc_environ(pid)
        marker = env.get("T2BEST5_JOB", "")
        if not marker.startswith(prefix):
            continue
        st = D.proc_stat(pid)
        gpu = env.get("CUDA_VISIBLE_DEVICES", "").strip()
        if st is None or st[0] in ("Z", "X") or not gpu.isdigit():
            continue
        seen[marker] = int(gpu)
    return [{"gpu": g, "marker": m} for m, g in seen.items()]


def make_driver(D):
    class ConsDriver(D.Driver):
        def __init__(self, a, log):
            super().__init__(a, log)
            self.cap = a.max_parallel
            self.ext: list[dict] = []

        def build_manifest(self) -> None:
            data = D.read_json(ROOT / "manifest_new.json")
            if data is not None:
                self.log(f"manifest_new.json exists (built {data.get('built_at')}): reused")
            else:
                self.log("building manifest_new.json (PREREG 4.2 / 6) and copying Zuxing's 5 designated checkpoints")
                rows = build_manifest(D, self.log, copy=True)
                D.write_csv(ROOT / "manifest_new.csv", rows, MANIFEST_COLS)
                D.write_atomic(ROOT / "manifest_new.json", D.dump({
                    "built_at": D.now(), "prereg_sha256": sha256_file(PREREG), "t2best5_driver_sha256": DRIVER_SHA,
                    "t2cons_eval_sha256": sha256_file(Path(__file__).resolve()), "t_c": "2026-10-24 23:24:16 CST",
                    "rows": rows}))
                for r in rows:
                    self.log(f"  {r['job']}: P{r.get('priority')} {r.get('variants')} {r.get('root_kind')} "
                             f"{str(r.get('parquet_sha256'))[:8]} ckpt {str(r.get('training_sha256'))[:8]} {r['status']}")
            self.jobs = list((D.read_json(ROOT / "manifest_new.json") or {}).get("rows", []))
            for r in self.jobs:
                if r.get("status") != "ok":
                    self.log.first(("refused", r["job"]), f"{r['job']}: NOT evaluated: {r.get('status')}")

        def stage_b(self) -> None:  # no Stage B here (PREREG 4.3: nothing; later candidates only in Round 2)
            self.last_stageb = time.time()

        def stage_b_done(self) -> bool:
            return True

        def launch_cleans(self) -> None:  # PREREG 6: no per-run LOCAL clean diagnostic before section 7
            return

        def all_final(self) -> bool:
            for job in self.ok_jobs():
                exp = D.export_rec(ROOT, job["job"])
                if exp is None:
                    return False
                if exp.get("status") != "ok":
                    continue
                if any(D.run_rec(ROOT, job["job"], k) is None for k in range(1, self.a.runs + 1)):
                    return False
            return True

        def progress(self) -> dict:
            out = super().progress()
            out["clean"] = "not run (PREREG 6)"
            return out

        def summary(self) -> None:
            try:
                write_summary(D, self.jobs)
            except Exception as error:  # noqa: BLE001
                self.log.first(("summary", repr(error)), f"WARNING summary failed: {error!r}")
            self.last_summary = time.time()

        def gpu_rows(self, state, mps, ours):
            return super().gpu_rows(state, mps, list(ours) + [{"gpu": e["gpu"]} for e in self.ext])

        def launch_evals(self) -> None:
            self.ext = external_evals(D)
            self.a.max_parallel = max(0, self.cap - len(self.ext))
            try:
                if self.a.max_parallel > 0:
                    super().launch_evals()
                elif self.ext:
                    self.log.first(("ext", len(self.ext)), f"{len(self.ext)} t2_best5 eval(s) running: total cap {self.cap} reached")
            finally:
                self.a.max_parallel = self.cap

        def heartbeat(self, phase: str) -> None:
            prog = self.progress()
            mean_s = sum(self.durations) / len(self.durations) if self.durations else None
            jobs = [{"kind": j["kind"], "job": j["job"], "run": j.get("run"), "try": j["try"], "gpu": j.get("gpu"),
                     "pid": j["proc"].pid, "elapsed_s": round(time.time() - j["start"]), "overdue": j["overdue"]}
                    for j in self.running.values()]  # fmt: skip
            jobs += [{"kind": o["kind"], "job": o["job"], "run": o.get("run"), "try": o["try"], "gpu": o.get("gpu"),
                      "orphan_pids": [p for p, _ in o["procs"]]} for o in self.orphans.values()]
            D.write_atomic(ROOT / "eval_heartbeat.json", D.dump({
                "updated_at": D.now(), "updated_epoch": time.time(), "pid": os.getpid(), "phase": phase, **prog,
                "running": jobs, "t2best5_evals_counted": len(self.ext), "mean_eval_seconds": mean_s, "halted": self.halted,
                "stop_file": (ROOT / "STOP").exists()}))

    return ConsDriver


# ================================================================================================ commands
def dry_run(D, a) -> int:
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    fatal, info = check_pins(D)
    f2, i2 = D.preflight()
    fatal += f2
    for x in info + i2:
        p(f"preflight: {x}")
    for x in fatal:
        p(f"preflight FAILED: {x}")
    rows = build_manifest(D, D.Logger(None), copy=False)
    for r in rows:
        des = (r.get("designation_evidence") or {}).get("row") or {}
        p(f"{r['job']}: P{r.get('priority')} variants {r.get('variants')} {r.get('root_kind')} parquet "
          f"{str(r.get('parquet_sha256'))[:12]} ckpt {r.get('training_sha256')} his own-input AUC {des.get('add_auc')} "
          f"(finished {des.get('finished_at')}): {r['status']}")
        for n in r.get("notes", []):
            p(f"    note: {n}")
    try:
        state = D.smi()
        mps = D.mps_gpus(state)
        drv = make_driver(D)(a, D.Logger(None))
        drv.ext = external_evals(D)
        p(f"t2_best5 live evals: {drv.ext}")
        for g in drv.gpu_rows(state, mps, []):
            p(f"gpu {g['gpu']}: free {g['free']} MiB, apps {g['napps']}, lock file {g['lock_file']}: {g['why']}")
    except Exception as error:  # noqa: BLE001
        p(f"gpu check failed: {error!r}")
    D.write_atomic(ROOT / "dryrun" / f"eval_plan_{D.stamp()}.txt", "\n".join(lines) + "\n")
    ok = not fatal and all(r["status"] == "ok" for r in rows)
    p(f"dry run: {'OK' if ok else 'PROBLEMS'}")
    return 0 if ok else 1


def status(D) -> int:
    hb = D.read_json(ROOT / "eval_heartbeat.json") or {}
    print(f"heartbeat {hb.get('updated_at')} phase {hb.get('phase')} exports {hb.get('exports')} evals {hb.get('evals')} "
          f"halted {hb.get('halted')} t2best5 evals counted {hb.get('t2best5_evals_counted')}")
    for j in hb.get("running", []):
        print(f"  running {j}")
    for r in (D.read_json(ROOT / "manifest_new.json") or {}).get("rows", []):
        exp = D.export_rec(ROOT, r["job"]) or {}
        aucs = []
        for k in range(1, NRUNS + 1):
            rr = D.run_rec(ROOT, r["job"], k)
            aucs.append("-" if rr is None else (f"{rr['metrics']['auc']:.4f}" if rr.get("status") == "ok" else "FAIL"))
        print(f"  {r['job']:<22} P{r.get('priority')} {r.get('variants'):<4} {str(r.get('root_kind')):<6} export "
              f"{exp.get('status', 'pending'):<10} {str(exp.get('tolerance') or ''):<7} runs {' '.join(aucs)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--gpus", default="0,1,2,5,6,7")
    r.add_argument("--max-parallel", type=int, default=4)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--runs", type=int, default=NRUNS)
    r.add_argument("--poll", type=float, default=10.0)
    sub.add_parser("summary")
    sub.add_parser("status")
    a = ap.parse_args()
    D = load_driver()
    if a.cmd == "run":
        if a.runs != NRUNS or not 1 <= a.max_parallel <= 4 or not 1 <= a.lane_max <= 2:
            ap.error("PREREG: --runs 5, --max-parallel <= 4, --lane-max <= 2")
        if a.dry_run:
            return dry_run(D, a)
        return make_driver(D)(a, D.Logger(ROOT / "eval_driver.log")).run()
    if a.cmd == "summary":
        out = write_summary(D, list((D.read_json(ROOT / "manifest_new.json") or {}).get("rows", [])))
        print(json.dumps({r["job"]: {"n_ok": r.get("n_ok"), "auc_mean": r.get("auc_mean")} for r in out["jobs"]}, indent=1))
        return 0
    return status(D)


if __name__ == "__main__":
    raise SystemExit(main())
