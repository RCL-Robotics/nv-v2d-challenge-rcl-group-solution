#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""trials_eval.py: repeated official evaluations ("trials") of fixed exported files, packed and scored with the kit.

Portable version of the server driver t1best20_driver.py (the 20-trial evaluation of the Oct-26 best set). Kept:
trial k of an item = one official evaluation of the SAME exported file through scripts/track2/eval_all.py
--export no -> the kit's examples/run_policy_evaluation.py -> flash_chord/scripts/evaluate_policy.py on the
untouched stock runtime (frame 0, one world, explicit reset, no help); trial-major order; one evaluation per GPU; a
completed evaluation is never re-run whatever its score, a technical failure is retried (<= 3 tries); the
acceptance checks of every evaluation (exit 0 and "evaluated", results JSON and parquet for the pinned file's
sha256 and path, the Tier-1 sequence, frame 0 / one world / explicit reset, finite worlds, the line's evaluator
digests in the parquet, environment_valid everywhere, the pinned file unchanged); packing each complete trial with
the kit packer (placeholder commit URL, files named *_LOCAL_NOT_FOR_UPLOAD.csv) and scoring it with the kit's
metric_code (score_csv.py); the trial closest to the mean (never the best) as the file to cite.
Left out: the NUMA eval-core lanes, the shared lock folders and the MPS refusal of the server (EVAL_CPUS,
EVAL_LOCKDIRS and REFUSE_MPS are options), and the PREREG hash lock (write your own protocol before you look).

SETS
  default            the 30 items of the current best set (configs/items/*.json with in_set: ep28alt for episode 28)
  --items ep01,ep25  a subset (packed with a restricted tracks.json, scored per episode; the summary also gives a
                     30-episode ESTIMATE that uses the best set's 20-trial means for the other episodes)
  --extra ep28       a config item evaluated alongside and scored on its own (as ep28alt was in the 20 trials)
  --file ep25=PATH   evaluate PATH instead of the item's pinned export (e.g. a new fork's export)
  --add NAME:ep25=PATH   an extra candidate NAME for episode 25 (scored alone; paired with ep25 in the summary)
Pinned exports are read from <T1_EXPORTS>/<item>/<file> and must match their sha256 in the item config.

  python trials_eval.py plan    --name N [set options] [--trials 20]        write plan.json, print it
  python trials_eval.py run     --name N [--gpus 0,1] [--max-parallel 4] [--cooldown 0] [--dry-run]
  python trials_eval.py summary --name N                                    pack, score, tables, summary.md
  python trials_eval.py status  --name N
  python trials_eval.py export  --out DIR CHECKPOINT [CHECKPOINT ...]      CPU export at tolerance 1e-4
Output: <T1_WORK>/trials/<N>/{plan.json, runs/<item>/tKK/{run.json,tryT/}, scores/tKK/, per_trial.csv,
per_item.csv, summary.md, driver.log}. Stop: touch <T1_WORK>/trials/<N>/STOP (running evaluations finish first).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1common as C  # noqa: E402

PLACEHOLDER_URL = "https://github.com/local-placeholder/not-uploaded/commit/" + "0" * 40
EVAL_TIMEOUT, RETRY_DELAY, EVAL_TRIES, SCORE_TIMEOUT = 2400.0, 300.0, 3, 3600.0
SCORER = Path(__file__).resolve().parent / "score_csv.py"


def root_of(site: C.Site, name: str) -> Path:
    return site.work / "trials" / name


def log_to(root: Path, write: bool = True):
    def log(msg: str) -> None:
        line = f"[{C.now()}] {msg}"
        print(line, flush=True)
        if not write:
            return
        root.mkdir(parents=True, exist_ok=True)
        with (root / "driver.log").open("a") as fp:
            fp.write(line + "\n")
    return log


def iteration_of(path: Path, default: int) -> int:
    m = re.fullmatch(r"model_(\d+)\.safetensors", path.name)
    return int(m.group(1)) if m else int(default)


# ================================================================================================ plan
def make_plan(args, site: C.Site) -> dict:
    items = C.load_items()
    best = C.set_items(items)
    if args.items in (None, "", "all"):
        chosen = best
    else:
        chosen = [C.item_of(x.strip(), items) for x in args.items.split(",") if x.strip()]
    if len({c["episode"] for c in chosen}) != len(chosen):
        raise SystemExit("--items: one item per episode (use --extra / --add for alternatives)")
    files = dict(f.split("=", 1) for f in (args.file or []))
    entries, order = {}, []

    def add(label: str, cfg: dict, role: str, path: Path | None, pair_with: str | None) -> None:
        if label in entries:
            raise SystemExit(f"{label} given twice")
        if path is None:
            path = site.exports / cfg["export"]["file"]
            expect = cfg["export"]["sha256"]
        else:
            expect = None
        path = Path(path).absolute()
        if not path.is_file():
            raise SystemExit(f"{label}: missing exported file {path}")
        sha = C.sha256(path)
        if expect and sha != expect:
            raise SystemExit(f"{label}: {path} sha256 {sha[:16]}... != the config's {expect[:16]}...")
        entries[label] = {"item": label, "config_item": cfg["item"], "role": role, "episode": cfg["episode"],
                          "object": cfg["object"], "sequence_id": cfg["sequence_id"], "file": str(path), "sha256": sha,
                          "iteration": iteration_of(path, cfg["pinned"]["iteration"]),
                          "pinned_export": expect is not None, "pair_with": pair_with,
                          "reference_auc_mean20": cfg["evaluation"]["trials20"]["auc_mean"] if expect else None,
                          "seed_signature": C.seed_signature(path, site.runtime, site.kit)}
        order.append(label)

    for cfg in chosen:
        add(cfg["item"], cfg, "set", Path(files.pop(cfg["item"])) if cfg["item"] in files else None, None)
    for x in (args.extra or "").split(","):
        if x.strip():
            cfg = C.item_of(x.strip(), items)
            base = next((e for e in entries.values() if e["role"] == "set" and e["episode"] == cfg["episode"]), None)
            add(cfg["item"], cfg, "extra", Path(files.pop(cfg["item"])) if cfg["item"] in files else None,
                base["item"] if base else None)
    for spec in args.add or []:
        m = re.fullmatch(r"([A-Za-z0-9_.+-]+):([A-Za-z0-9]+)=(.+)", spec)
        if not m:
            raise SystemExit(f"--add {spec!r}: expected NAME:ITEM=PATH")
        cfg = C.item_of(m.group(2), items)
        base = next((e for e in entries.values() if e["role"] == "set" and e["episode"] == cfg["episode"]), None)
        add(m.group(1), cfg, "extra", Path(m.group(3)), base["item"] if base else None)
    if files:
        raise SystemExit(f"--file for items that are not in the plan: {sorted(files)}")
    set_eps = sorted(e["episode"] for e in entries.values() if e["role"] == "set")
    return {"name": args.name, "created_at": C.now(), "trials": int(args.trials), "full_set": set_eps == list(range(30)),
            "set_episodes": set_eps, "order": order, "items": entries, "code": str(site.code),
            "code_head": C.git_head(site.code), "stock_runtime": str(site.runtime), "kit": str(site.kit)}


def load_or_make_plan(args, site: C.Site, log, write: bool) -> dict:
    root = root_of(site, args.name)
    old = C.read_json(root / "plan.json")
    if old is not None:
        given = any([args.items not in (None, "", "all"), args.extra, args.file, args.add])
        if given:
            new = make_plan(args, site)
            same = {k: new[k] for k in ("order", "trials")} == {k: old[k] for k in ("order", "trials")} and all(
                new["items"][k]["sha256"] == old["items"][k]["sha256"] for k in new["order"])
            if not same:
                raise SystemExit(f"{root}/plan.json exists with another plan: use a new --name")
        for e in old["items"].values():
            if C.sha256(e["file"]) != e["sha256"]:
                raise SystemExit(f"{e['item']}: {e['file']} changed since the plan was written")
        return old
    plan = make_plan(args, site)
    if write:
        C.write_atomic(root / "plan.json", C.dump(plan))
        log(f"plan written: {len(plan['order'])} items x {plan['trials']} trials (full set: {plan['full_set']})")
    return plan


def print_plan(plan: dict) -> None:
    print(f"{plan['name']}: {plan['trials']} trials of {len(plan['order'])} items; set episodes "
          f"{'all 30' if plan['full_set'] else plan['set_episodes']}")
    for k in plan["order"]:
        e = plan["items"][k]
        print(f"  {k:12} {e['role']:5} ep{e['episode']:02d} model_{e['iteration']:<7} {e['sha256'][:12]} "
              f"{'pinned' if e['pinned_export'] else 'given '} {e['file']}"
              f"{'  (pairs with ' + e['pair_with'] + ')' if e.get('pair_with') else ''}")


# ================================================================================================ run
class Runner:
    def __init__(self, args, site: C.Site, plan: dict, log):
        self.a, self.site, self.plan, self.log = args, site, plan, log
        self.root = root_of(site, plan["name"])
        self.keys, self.keys_from = C.kit_metric_keys(site.kit)
        self.lanes = C.Lanes(site)
        self.lock = threading.Lock()
        self.running: dict[tuple, dict] = {}
        self.rested: dict[int, float] = {}

    def run_dir(self, item: str, k: int) -> Path:
        return self.root / "runs" / item / f"t{k:02d}"

    def tries(self, item: str, k: int) -> list[dict]:
        d = self.run_dir(item, k)
        out = []
        for t in sorted(d.glob("try*/try.json")) if d.is_dir() else []:
            rec = C.read_json(t)
            if rec:
                out.append(rec)
        return out

    def final(self, item: str, k: int) -> dict | None:
        return C.read_json(self.run_dir(item, k) / "run.json")

    def jobs(self) -> list[tuple[int, str]]:
        out = []
        for k in range(1, self.plan["trials"] + 1):
            for item in self.plan["order"]:
                if self.a.only and f"t{k:02d}:{item}" not in self.a.only.split(","):
                    continue
                if self.final(item, k) is None:
                    out.append((k, item))
        return out

    def ready(self, k: int, item: str) -> bool:
        tr = self.tries(item, k)
        if not tr:
            return True
        last = max(float(t.get("finished_epoch", 0)) for t in tr)
        return time.time() - last >= RETRY_DELAY

    def command(self, e: dict, work: Path, gpu: int) -> list[str]:
        return C.taskset_prefix(self.site, gpu) + C.eval_all_command(self.site, work, "t1trials", e["episode"],
                                                                     e["iteration"], gpu, export=False)

    def one(self, k: int, item: str, gpu: int, fds: list[int]) -> None:
        e = self.plan["items"][item]
        d = self.run_dir(item, k)
        n = len([p for p in d.glob("try*") if p.is_dir()]) + 1 if d.is_dir() else 1  # a fresh dir for every try
        work = d / f"try{n}"
        try:
            (work / f"ep{e['episode']:02d}").mkdir(parents=True, exist_ok=True)
            link = work / f"ep{e['episode']:02d}" / f"model_{e['iteration']}.safetensors"
            if not link.is_symlink():
                link.symlink_to(e["file"])
            cmd = self.command(e, work, gpu)
            env = C.fc_env(self.site, gpu, {"TMPDIR": self.root / "tmp", "T1TRIALS_JOB": work})
            if self.site.get("EVAL_CACHE_DIR"):
                env["FLASH_CHORD_CACHE_DIR"] = self.site.get("EVAL_CACHE_DIR")
            (self.root / "tmp").mkdir(parents=True, exist_ok=True)
            started, t0, note = C.now(), time.time(), ""
            with (work / "eval_all.log").open("a") as stream:
                stream.write(f"[{started}] t{k:02d} {item} try {n} gpu {gpu}: {shlex.join(cmd)}\n")
                stream.flush()
                proc = subprocess.Popen(cmd, cwd=self.site.code, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True, pass_fds=tuple(fds))
                try:
                    code = proc.wait(timeout=EVAL_TIMEOUT)
                except subprocess.TimeoutExpired:
                    note, code = f"timeout after {EVAL_TIMEOUT:.0f} s", -9
                    for sig in (signal.SIGTERM, signal.SIGKILL):  # this evaluation's own session only
                        try:
                            os.killpg(proc.pid, sig)
                            proc.wait(timeout=60)
                            break
                        except (OSError, subprocess.TimeoutExpired):
                            continue
            problems, info = C.check_results(e["episode"], e["sequence_id"], work / "results", work / "eval_report.json",
                                             code, self.keys, expect_sha=e["sha256"], expect_file=Path(e["file"]))
            if note:
                problems.insert(0, note)
            if C.sha256(e["file"]) != e["sha256"]:
                problems.append("the evaluated file CHANGED")
            rec = {"trial": k, "item": item, "episode": e["episode"], "try": n, "status": "failed" if problems else "ok",
                   "problems": problems, "gpu": gpu, "started_at": started, "finished_at": C.now(),
                   "finished_epoch": time.time(), "seconds": round(time.time() - t0, 1), "file": e["file"],
                   "sha256": e["sha256"], "seed_signature": e["seed_signature"], "command": shlex.join(cmd),
                   "metric_keys": self.keys, **info}
            C.write_atomic(work / "try.json", C.dump(rec))
            if not problems or n >= EVAL_TRIES:
                C.write_atomic(self.run_dir(item, k) / "run.json", C.dump(rec))
            m = rec.get("metrics") or {}
            self.log(f"t{k:02d} {item} try {n} gpu {gpu}: {rec['status']} auc {m.get('auc')} sp {m.get('sp_sr')} "
                     f"mp {m.get('mp_sr')} mppe {m.get('mppe')} {'; '.join(problems)}")
        except Exception as error:  # noqa: BLE001
            self.log(f"t{k:02d} {item} try {n}: driver error {error!r}")
            C.write_atomic(work / "try.json", C.dump({"trial": k, "item": item, "try": n, "status": "failed",
                                                      "problems": [f"driver error {error!r}"],
                                                      "finished_epoch": time.time()}))
        finally:
            C.Lanes.release(fds)
            with self.lock:
                self.running.pop((k, item), None)
                self.rested[gpu] = time.time()

    def loop(self) -> int:
        stop = self.root / "STOP"
        jobs = self.jobs()
        self.log(f"run: {len(jobs)} evaluations to do; metrics from {self.keys_from}")
        if self.a.dry_run:
            for k, item in jobs[:4]:
                e = self.plan["items"][item]
                print(f"DRY t{k:02d} {item}: {shlex.join(self.command(e, self.run_dir(item, k) / 'try1', 0))}")
            return 0
        last_note = 0.0
        while True:
            with self.lock:
                busy = {v["gpu"] for v in self.running.values()}
                n_running = len(self.running)
                taken = set(self.running)
            todo = [j for j in self.jobs() if j not in taken]
            if not todo and not n_running:
                break
            if stop.exists():
                if not n_running:
                    self.log("STOP: exiting (rm STOP and run again to resume)")
                    return 0
                time.sleep(5)
                continue
            state = C.smi()
            gpus, notes = C.usable_gpus(self.site, state)
            if self.a.gpus:
                gpus = [g for g in gpus if g in C.parse_int_list(self.a.gpus)]
            started = False
            for g in sorted(gpus, key=lambda g: -state["gpus"][g]["free_mb"]):
                if n_running >= self.a.max_parallel or g in busy:
                    continue
                if time.time() - self.rested.get(g, 0.0) < self.a.cooldown:
                    continue
                if state["gpus"][g]["free_mb"] < self.site.min_free_mb:
                    continue
                job = next((j for j in todo if self.ready(*j)), None)
                if job is None:
                    break
                fds = self.lanes.take(g)
                if fds is None:
                    continue
                todo.remove(job)
                with self.lock:
                    self.running[job] = {"gpu": g, "start": time.time()}
                threading.Thread(target=self.one, args=(job[0], job[1], g, fds), daemon=False).start()
                n_running += 1
                busy.add(g)
                started = True
            if not started and not n_running and time.time() - last_note > 600:
                self.log(f"waiting: no usable free lane ({'; '.join(notes) or 'all GPUs busy or below EVAL_MIN_FREE_MB'})")
                last_note = time.time()
            time.sleep(self.a.poll)
        failed = [(k, i) for k in range(1, self.plan["trials"] + 1) for i in self.plan["order"]
                  if (self.final(i, k) or {}).get("status") == "failed"]
        self.log(f"run finished; failed evaluations: {failed or 'none'}")
        return 0


# ================================================================================================ summary
def pack(site: C.Site, results: Path, out_csv: Path, log_file: Path, config: Path | None) -> dict:
    """The kit packer (kit python -B, cwd = kit), placeholder commit URL: the CSV is for local scoring only."""
    kit = site.kit
    cmd = ["nice", "-n", "15", site.kit_python, "-B", str(kit / "tools/pack_submission.py"), "--track", "track_2_tier1",
           "--results", str(results), "--dataset-root", str(site.data), "--code-commit", PLACEHOLDER_URL,
           "--eval-script", str(kit / "flash_chord/scripts/evaluate_policy.py"), "-o", str(out_csv)]
    if config is not None:
        cmd += ["--config", str(config)]
    p = subprocess.run(cmd, cwd=kit, env=C.kit_env(), capture_output=True, text=True, timeout=SCORE_TIMEOUT)
    C.write_atomic(log_file, f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    out = {"exit": p.returncode, "csv": str(out_csv), "command": shlex.join(cmd)}
    if p.returncode == 0 and out_csv.is_file():
        out["csv_sha256"] = C.sha256(out_csv)
        out["crosscheck_warning"] = "WARNING: the trajectory and the evaluator's metrics JSON disagree" in p.stdout
    return out


def score(site: C.Site, csv_file: Path, out_json: Path, episodes: str, log_file: Path) -> dict | None:
    cmd = ["nice", "-n", "15", site.kit_python, "-B", str(SCORER), "--csv", str(csv_file), "--out", str(out_json),
           "--episodes", episodes, "--kit", str(site.kit), "--data", str(site.data)]
    p = subprocess.run(cmd, cwd=out_json.parent, env=C.kit_env(), capture_output=True, text=True, timeout=SCORE_TIMEOUT)
    C.write_atomic(log_file, f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return C.read_json(out_json) if p.returncode == 0 else None


def restricted_tracks(site: C.Site, episodes: list[int], dest: Path) -> Path:
    cfg = json.loads((site.kit / "config/tracks.json").read_text())
    cfg["tracks"]["track_2_tier1"]["scored_episodes"] = list(episodes)
    C.write_atomic(dest, json.dumps(cfg, indent=2) + "\n")
    return dest


def copy_results(dest: Path, recs: list[dict]) -> None:
    if dest.is_dir():
        for leftover in dest.glob("episode_*"):
            leftover.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    for r in recs:
        for src in (r["results_json"], r["results_parquet"]):
            tmp = dest / f".{Path(src).name}.tmp"
            shutil.copy2(src, tmp)
            os.replace(tmp, dest / Path(src).name)


def score_group(site: C.Site, base: Path, label: str, recs: list[dict], episodes: list[int], full: bool, log) -> dict:
    inputs = {r["item"]: r.get("parquet_sha256") for r in recs}
    old = C.read_json(base / "score.json") or {}
    if old.get("inputs") == inputs and old.get("score"):
        return old
    copy_results(base / "results", recs)
    config = None if full else restricted_tracks(site, episodes, base / "tracks.json")
    csv_file = base / f"{label}_LOCAL_NOT_FOR_UPLOAD.csv"
    pk = pack(site, base / "results", csv_file, base / "pack.log", config)
    sc = score(site, csv_file, base / "metric_code.json", "all" if full else " ".join(map(str, episodes)),
               base / "score.log") if pk.get("csv_sha256") else None
    out = {"inputs": inputs, "pack": pk, "score": sc, "scored_at": C.now()}
    C.write_atomic(base / "score.json", C.dump(out))
    log(f"{label}: packed (exit {pk['exit']}) and scored: "
        f"{None if sc is None else (sc.get('overall_float32') or sc.get('mean_of_per_episode_float32'))}")
    return out


def mstats(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(vals), "mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


def fv(v, spec=".4f") -> str:
    return "" if v is None else format(v, spec)


def cmd_summary(args, site: C.Site, plan: dict, log) -> int:
    root = root_of(site, plan["name"])
    r = Runner(args, site, plan, log)
    set_labels = [k for k in plan["order"] if plan["items"][k]["role"] == "set"]
    extras = [k for k in plan["order"] if plan["items"][k]["role"] == "extra"]
    full = plan["full_set"]
    per_trial, per_item_vals = [], {k: [] for k in plan["order"]}
    metric_names = {"auc": "AUC", "sp_sr": "SP-SR", "mp_sr": "MP-SR", "mppe": "MPPE"}
    for k in range(1, plan["trials"] + 1):
        recs = {i: r.final(i, k) for i in plan["order"]}
        set_ok = all((recs[i] or {}).get("status") == "ok" for i in set_labels)
        row = {"trial": k, "complete": set_ok, "n_set_ok": sum((recs[i] or {}).get("status") == "ok" for i in set_labels)}
        if set_ok:
            out = score_group(site, root / "scores" / f"t{k:02d}" / "set", f"trial{k:02d}_set", [recs[i] for i in set_labels],
                              plan["set_episodes"], full, log)
            sc = out.get("score") or {}
            pe = sc.get("per_episode_float32") or {}
            for i in set_labels:
                v = pe.get(str(plan["items"][i]["episode"]))
                if v:
                    per_item_vals[i].append(v)
            agg = sc.get("overall_float32") if full else sc.get("mean_of_per_episode_float32")
            for m, km in metric_names.items():
                row[m] = (agg or {}).get(km)
            for m in C.METRICS:
                row[f"json_{m}"] = statistics.fmean(recs[i]["metrics"][m] for i in set_labels)
        set_pe = {}
        if set_ok:
            set_pe = ((C.read_json(root / "scores" / f"t{k:02d}" / "set" / "score.json") or {}).get("score") or {}).get(
                "per_episode_float32") or {}
        for x in extras:
            rec = recs[x]
            if (rec or {}).get("status") != "ok":
                row[f"{x}_auc"] = None
                continue
            e = plan["items"][x]
            out = score_group(site, root / "scores" / f"t{k:02d}" / x, f"trial{k:02d}_{x}_PARTIAL", [rec], [e["episode"]],
                              False, log)
            v = ((out.get("score") or {}).get("per_episode_float32") or {}).get(str(e["episode"]))
            if not v:
                continue
            per_item_vals[x].append(v)
            row[f"{x}_auc"] = v["AUC"]
            base = set_pe.get(str(e["episode"])) if e.get("pair_with") else None
            if base and full and row.get("auc") is not None:
                row[f"set_auc_with_{x}"] = row["auc"] + (v["AUC"] - base["AUC"]) / 30.0
        per_trial.append(row)
    cols = ["trial", "complete", "n_set_ok", "auc", "sp_sr", "mp_sr", "mppe", "json_auc", "json_sp_sr", "json_mp_sr",
            "json_mppe"] + [c for x in extras for c in (f"{x}_auc", f"set_auc_with_{x}")]
    with (root / "per_trial.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in per_trial:
            w.writerow(row)
    items_rows = []
    for i in plan["order"]:
        e = plan["items"][i]
        vals = per_item_vals[i]
        st = mstats([v["AUC"] for v in vals])
        items_rows.append({"item": i, "role": e["role"], "episode": e["episode"], "object": e["object"],
                           "sha256": e["sha256"][:12], "n": st["n"], "auc_mean": st["mean"], "auc_sd": st["sd"],
                           "auc_min": st["min"], "auc_max": st["max"],
                           "sp_pass": sum(1 for v in vals if v["SP-SR"] >= 1.0),
                           "mp_pass": sum(1 for v in vals if v["MP-SR"] >= 1.0),
                           "mppe_mean": mstats([v["MPPE"] for v in vals])["mean"],
                           "best_set_auc_mean20": e.get("reference_auc_mean20"),
                           "pair_with": e.get("pair_with") or ""})
    with (root / "per_item.csv").open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(items_rows[0]))
        w.writeheader()
        for row in items_rows:
            w.writerow(row)
    write_md(root, plan, per_trial, items_rows, extras)
    log(f"summary: {root / 'summary.md'}")
    return 0


def write_md(root: Path, plan: dict, per_trial: list[dict], items_rows: list[dict], extras: list[str]) -> None:
    full = plan["full_set"]
    done = [r for r in per_trial if r["complete"] and r.get("auc") is not None]
    lines = [f"# Trials {plan['name']}", "",
             f"Generated {C.now()} by `trials_eval.py summary`. {plan['trials']} trials planned; {len(done)} complete "
             f"and scored. Set: {'the 30 episodes' if full else 'episodes ' + ' '.join(map(str, plan['set_episodes']))}. "
             "Packed CSVs carry a placeholder commit URL and are named *_LOCAL_NOT_FOR_UPLOAD.csv. "
             f"Code {plan.get('code_head') or '?'}.", "",
             "| trial | AUC | SP-SR | MP-SR | MPPE cm | evaluator-JSON mean AUC |" + "".join(f" {x} AUC |" for x in extras),
             "|---:|---:|---:|---:|---:|---:|" + "---:|" * len(extras)]
    for r in per_trial:
        lines.append(f"| {r['trial']:02d} | {fv(r.get('auc'))} | {fv(r.get('sp_sr'), '.3f')} | {fv(r.get('mp_sr'), '.3f')} | "
                     f"{fv(r.get('mppe'), '.3f')} | {fv(r.get('json_auc'))} |" + "".join(f" {fv(r.get(f'{x}_auc'))} |" for x in extras))
    lines += ["", f"Set metrics are the kit metric_code {'overall' if full else 'means of the per-episode'} float32 scores."]
    if done:
        lines += ["", "| metric | n | mean | sd | min | max |", "|---|---:|---:|---:|---:|---:|"]
        for m in ("auc", "sp_sr", "mp_sr", "mppe"):
            st = mstats([r[m] for r in done])
            lines.append(f"| {m} | {st['n']} | {fv(st['mean'])} | {fv(st['sd'])} | {fv(st['min'])} | {fv(st['max'])} |")
        mean = statistics.fmean(r["auc"] for r in done)
        close = min(done, key=lambda r: (abs(r["auc"] - mean), r["trial"]))
        lines += ["", f"Closest to the mean AUC ({mean:.4f}): trial {close['trial']:02d} (AUC {close['auc']:.4f}). "
                  "Cite a single trial chosen this way, never the best trial."]
        for x in extras:
            hv = [r.get(f"set_auc_with_{x}") for r in done if r.get(f"set_auc_with_{x}") is not None]
            if hv:
                hm = statistics.fmean(hv)
                cands = [r for r in done if r.get(f"set_auc_with_{x}") is not None]
                hc = min(cands, key=lambda r: (abs(r[f"set_auc_with_{x}"] - hm), r["trial"]))
                lines.append(f"With {x} in place of its pair: mean set AUC {hm:.4f} (auc + (alt - base) / 30); closest "
                             f"trial {hc['trial']:02d} ({hc[f'set_auc_with_{x}']:.4f}). Re-pack that trial with {x}'s "
                             "result before citing it.")
    lines += ["", "## Per item (metric_code per episode over scored trials)", "",
              "| item | role | ep | object | file | n | AUC mean | sd | min | max | SP pass | MP pass | MPPE mean | best-set 20-trial AUC |",
              "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in items_rows:
        lines.append(f"| {r['item']} | {r['role']} | {r['episode']} | {r['object']} | {r['sha256']} | {r['n']} | "
                     f"{fv(r['auc_mean'])} | {fv(r['auc_sd'])} | {fv(r['auc_min'])} | {fv(r['auc_max'])} | {r['sp_pass']} | "
                     f"{r['mp_pass']} | {fv(r['mppe_mean'], '.2f')} | {fv(r['best_set_auc_mean20'])} |")
    if not full:
        ref = {c["episode"]: c["evaluation"]["trials20"]["auc_mean"] for c in C.set_items(C.load_items())}
        rows = {r["episode"]: r for r in items_rows if r["role"] == "set" and r["auc_mean"] is not None}
        est = statistics.fmean(rows[e]["auc_mean"] if e in rows else ref[e] for e in range(30))
        lines += ["", f"30-episode AUC ESTIMATE with these set items' means and the best set's 20-trial means for the "
                  f"other episodes: {est:.4f} (best set: {statistics.fmean(ref.values()):.4f}). An estimate, not a packed "
                  "trial; confirm with a full-set run."]
    C.write_atomic(root / "summary.md", "\n".join(lines) + "\n")


def cmd_status(args, site: C.Site, plan: dict, log) -> int:
    r = Runner(args, site, plan, log)
    ok = failed = pending = 0
    for k in range(1, plan["trials"] + 1):
        for i in plan["order"]:
            f = r.final(i, k)
            if f is None:
                pending += 1
            elif f["status"] == "ok":
                ok += 1
            else:
                failed += 1
    print(f"{plan['name']}: ok {ok}, failed {failed}, pending {pending} of {plan['trials'] * len(plan['order'])}")
    return 0


def cmd_export(args, site: C.Site) -> int:
    out = Path(args.out).absolute()
    rc = 0
    for ck in args.checkpoints:
        src = Path(ck).absolute()
        label = args.label or f"{src.parent.name}_{src.stem}"
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        cmd, upd = C.export_command(site, src, d)
        env = C.fc_env(site, None, upd)
        with (d / "export.log").open("a") as fp:
            fp.write(f"[{C.now()}] {shlex.join(cmd)}\n")
            fp.flush()
            code = subprocess.run(["nice", "-n", "10"] + cmd, cwd=site.code, env=env, stdout=fp, stderr=subprocess.STDOUT).returncode
        ok, problems, info = C.check_export_report(d / "export_report.json", C.sha256(src))
        dst = d / src.name
        if code != 0 or not dst.is_file():
            ok, problems = False, problems + [f"exit {code}"]
        rec = {"source": str(src), "source_sha256": C.sha256(src), "file": str(dst), "status": "ok" if ok else "failed",
               "problems": problems, "sha256": C.sha256(dst) if dst.is_file() else None, "code": str(site.code),
               "code_head": C.git_head(site.code), "command": shlex.join(cmd), "exported_at": C.now(), **info}
        C.write_atomic(d / "export.json", C.dump(rec))
        print(f"{'ok' if ok else 'FAILED'} {dst} sha256 {rec['sha256']} max_rel_diff {info.get('max_rel_diff')} "
              f"{'; '.join(problems)}")
        rc |= 0 if ok else 1
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_site_arg(p)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "summary", "status"):
        s = sub.add_parser(name)
        s.add_argument("--name", required=True)
        s.add_argument("--items", help="all (default: the 30 set items) or a comma list")
        s.add_argument("--extra", help="comma list of config items evaluated alongside (scored alone)")
        s.add_argument("--file", action="append", help="ITEM=PATH: evaluate PATH for ITEM")
        s.add_argument("--add", action="append", help="NAME:ITEM=PATH: an extra candidate for ITEM's episode")
        s.add_argument("--trials", type=int, default=20)
        s.add_argument("--gpus", help="restrict to these GPUs (comma list)")
        s.add_argument("--max-parallel", type=int, default=4)
        s.add_argument("--cooldown", type=float, default=0.0, help="seconds a GPU rests after each of our evaluations")
        s.add_argument("--poll", type=float, default=5.0)
        s.add_argument("--only", help="t01:ep04,t02:ep04 style subset of (trial, item) jobs")
        s.add_argument("--no-summary", action="store_true")
        s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("export")
    s.add_argument("--out", required=True)
    s.add_argument("--label", help="output sub-directory (default <run dir name>_<model_N>)")
    s.add_argument("checkpoints", nargs="+")
    args = p.parse_args()
    site = C.load_site(args.site)
    if args.cmd == "export":
        return cmd_export(args, site)
    log = log_to(root_of(site, args.name), write=not args.dry_run)
    plan = load_or_make_plan(args, site, log, write=args.cmd in ("plan", "run") and not args.dry_run)
    if args.cmd == "plan":
        print_plan(plan)
        return 0
    if args.cmd == "status":
        return cmd_status(args, site, plan, log)
    if args.cmd == "summary":
        return cmd_summary(args, site, plan, log)
    rc = Runner(args, site, plan, log).loop()
    if rc == 0 and not args.dry_run and not args.no_summary and not (root_of(site, args.name) / "STOP").exists():
        rc = cmd_summary(args, site, plan, log)
    return rc


if __name__ == "__main__":
    sys.exit(main())
