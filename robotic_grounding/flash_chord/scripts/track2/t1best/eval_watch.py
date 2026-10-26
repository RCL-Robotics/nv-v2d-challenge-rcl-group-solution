#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""eval_watch.py: official Tier-1 evaluation of a fork's checkpoints while it trains, and its best checkpoint.

Portable version of the server watcher runin_eval_watch.py (and its wave-2 copy). Kept: every evaluation goes through
scripts/track2/eval_all.py of the training tree -> E2 export with the stock-runtime equivalence check at tolerance
1e-4 (this tree's exporter) -> the kit's examples/run_policy_evaluation.py + flash_chord/scripts/evaluate_policy.py on
the untouched stock runtime (frame-0 start, no VOC help, one world). Metrics come from the evaluator's own results
JSON with the kit's key mapping (auc = add_auc, sp_sr = spider_sr_uncentered, mp_sr = maniptrans_sr,
mppe = mean_per_point_error_cm). A row is ok only if eval_all exited 0 and says "evaluated", the export is
stock-verified, the results JSON is for the evaluated file, the reference is the episode's Tier-1 sequence, the start
is frame 0 with one world and an explicit reset, and the rollout parquet carries the line's evaluator digests.
Left out: the AUC early stop and the iteration cap (they signalled the trainer), the NUMA eval lanes and the shared
lock folders of the server (EVAL_CPUS and EVAL_LOCKDIRS are options).

WHAT. Checkpoints <run>/model_<N>.safetensors with N a multiple of --every (default 2000), unchanged for 60 s. By
default only the NEWEST not yet evaluated one (older ones are skipped, as the original did); --all evaluates every
one. When the training has ended (final.json, or the launch.json pid is gone) its newest checkpoint is evaluated
even off the grid, then the best checkpoint (highest single-evaluation AUC, ties -> earlier) gets --final-repeats
more evaluations and watch_final.json is written with the mean over its evaluations.
OUTPUT <fork>/eval/ (with --run-dir: <T1_WORK>/watch/<item>-<parent>-<run>/eval/): evals.csv (every try), best.json, watch_final.json, work/it<N>/<tag>/ (eval_all work dir with
the exported stock-format file, its export_report.json and the evaluator results), watcher.log. Stop: touch
<fork>/eval/STOP (the running evaluation finishes first).

  python eval_watch.py <fork name> [--gpu G] [--every 2000] [--all] [--once] [--final-repeats 2] [--dry-run]
  python eval_watch.py --run-dir <dir> --item ep25 ...      a run directory that fork_train.py did not create
Settings: T1_WORK, V2D_MOTION, V2D_RUNTIME, V2D_KIT, V2D_DATA, FC_PYTHON, KIT_PYTHON, T1_CODE, GPUS, EXCLUDE_GPUS,
REFUSE_MPS, EVAL_CPUS, EVAL_LOCKDIRS, EVAL_MIN_FREE_MB.
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1common as C  # noqa: E402

COLUMNS = ["iteration", "purpose", "try", "status", "auc", "sp_sr", "mp_sr", "mppe", "chord_sr", "gpu", "seconds",
           "export_rel_diff", "checkpoint_sha256", "note", "started_at", "finished_at", "work"]
STABLE_AGE = 60.0
EVAL_TIMEOUT = 1800.0


class Watch:
    def __init__(self, args, site: C.Site):
        self.a, self.site = args, site
        items = C.load_items()
        if args.run_dir:
            self.run = Path(args.run_dir).absolute()
            if not args.item:
                raise SystemExit("--run-dir needs --item")
            self.cfg = C.item_of(args.item, items)
            # never write next to a run this tool did not create: <T1_WORK>/watch/<item>-<parent>-<run>/eval
            self.root = site.work / "watch" / f"{self.cfg['item']}-{self.run.parent.name}-{self.run.name}"
            self.launch = {}
        else:
            self.root = site.work / "forks" / args.name
            self.run = self.root / "run"
            fork = C.read_json(self.root / "fork.json")
            if fork is None:
                raise SystemExit(f"{self.root}/fork.json missing: not a fork_train.py fork (use --run-dir)")
            self.cfg = C.item_of(args.item or fork["item"], items)
            self.launch = C.read_json(self.root / "launch.json") or {}
        self.ep = int(self.cfg["episode"])
        self.eval = self.root / "eval"
        self.csv = self.eval / "evals.csv"
        self.keys, self.keys_from = C.kit_metric_keys(site.kit)
        self.lanes = C.Lanes(site)

    # ---------------------------------------------------------------------------------------------- state
    def log(self, msg: str) -> None:
        line = f"[{C.now()}] {msg}"
        print(line, flush=True)
        if not self.a.dry_run:
            self.eval.mkdir(parents=True, exist_ok=True)
            with (self.eval / "watcher.log").open("a") as fp:
                fp.write(line + "\n")

    def rows(self) -> list[dict]:
        try:
            with self.csv.open() as fp:
                return list(csv.DictReader(fp))
        except OSError:
            return []

    def append(self, row: dict) -> None:
        new = not self.csv.is_file()
        self.eval.mkdir(parents=True, exist_ok=True)
        with self.csv.open("a", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})

    def ok_rows(self) -> list[dict]:
        return [r for r in self.rows() if r["status"] == "ok"]

    def tries(self, it: int, purpose: str) -> int:
        return sum(1 for r in self.rows() if int(r["iteration"]) == it and r["purpose"] == purpose)

    def ended(self) -> str | None:
        fin = C.read_json(self.run / "final.json")
        if fin:
            return f"final.json ({fin.get('reason')}, model_{fin.get('iteration')})"
        pid = self.launch.get("pid")
        if pid:
            try:
                os.kill(int(pid), 0)
            except OSError:
                return f"trainer pid {pid} is gone"
        if self.a.run_dir and self.a.ended:
            return "--ended"
        return None

    def stable(self) -> list[int]:
        out, t = [], time.time()
        for n in C.checkpoints_in(self.run):
            try:
                if t - (self.run / f"model_{n}.safetensors").stat().st_mtime >= STABLE_AGE:
                    out.append(n)
            except OSError:
                pass
        return out

    def best(self) -> dict | None:
        cands = [r for r in self.ok_rows() if r["purpose"] == "scan" and C.fnum(r["auc"]) is not None]
        if not cands:
            return None
        return max(cands, key=lambda r: (C.fnum(r["auc"]), -int(r["iteration"])))

    # ---------------------------------------------------------------------------------------------- one evaluation
    def pick_gpu(self) -> tuple[int, list[int]] | None:
        state = C.smi()
        gpus, _ = C.usable_gpus(self.site, state)
        if self.a.gpu is not None:
            gpus = [self.a.gpu] if self.a.gpu in gpus else []
        for g in sorted(gpus, key=lambda g: -state["gpus"][g]["free_mb"]):
            if state["gpus"][g]["free_mb"] < self.site.min_free_mb:
                continue
            fds = self.lanes.take(g)
            if fds is not None:
                return g, fds
        return None

    def evaluate(self, it: int, purpose: str) -> dict:
        tr = self.tries(it, purpose) + 1
        tag = f"{purpose}{tr}"
        work = self.eval / "work" / f"it{it}" / tag
        cmd_preview = C.eval_all_command(self.site, work, f"watch_{self.root.name}", self.ep, it, self.a.gpu or 0, True)
        if self.a.dry_run:
            self.log(f"DRY {purpose} model_{it}: {' '.join(cmd_preview)}")
            return {"status": "dry"}
        got = None
        while got is None:
            got = self.pick_gpu()
            if got is None:
                self.log("no free evaluation lane; waiting 60 s")
                time.sleep(60)
        gpu, fds = got
        try:
            (work / f"ep{self.ep:02d}").mkdir(parents=True, exist_ok=True)
            link = work / f"ep{self.ep:02d}" / f"model_{it}.safetensors"
            src = self.run / f"model_{it}.safetensors"
            if not link.is_symlink():
                link.symlink_to(src)
            (work / "eval_report.json").unlink(missing_ok=True)
            cmd = C.taskset_prefix(self.site, gpu) + C.eval_all_command(self.site, work, f"watch_{self.root.name}",
                                                                        self.ep, it, gpu, True)
            started, t0, note = C.now(), time.time(), ""
            with (work / "eval_all.log").open("a") as stream:
                stream.write(f"[{started}] {purpose} try {tr}, gpu {gpu}: {' '.join(cmd)}\n")
                stream.flush()
                proc = subprocess.Popen(cmd, cwd=self.site.code, env=C.fc_env(self.site, gpu), stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(fds))
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
        finally:
            C.Lanes.release(fds)
        problems, info = C.check_results(self.ep, self.cfg["sequence_id"], work / "results", work / "eval_report.json",
                                         code, self.keys)
        if note:
            problems.insert(0, note)
        rep = C.read_json(work / "eval_report.json") or {}
        row_rep = next((r for r in rep.get("episodes", []) if str(r.get("episode")) == str(self.ep)), {})
        if row_rep and info.get("checkpoint_sha256") and row_rep.get("checkpoint_sha256") not in (None, info["checkpoint_sha256"]):
            problems.append("eval_report checkpoint_sha256 != results JSON")
        m = info.get("metrics") or {}
        row = {"iteration": it, "purpose": purpose, "try": tr, "status": "failed" if problems else "ok", **m,
               "chord_sr": info.get("chord_sr"), "gpu": gpu, "seconds": round(time.time() - t0, 1),
               "export_rel_diff": info.get("export_max_rel_diff"), "checkpoint_sha256": info.get("checkpoint_sha256"),
               "note": "; ".join(problems), "started_at": started, "finished_at": C.now(), "work": str(work)}
        self.append(row)
        self.log(f"{purpose} model_{it} try {tr} on gpu {gpu}: {row['status']} auc {m.get('auc')} sp {m.get('sp_sr')} "
                 f"mp {m.get('mp_sr')} mppe {m.get('mppe')} {row['note']}")
        b = self.best()
        if b is not None:
            C.write_atomic(self.eval / "best.json", C.dump({"item": self.cfg["item"], "episode": self.ep,
                                                            "best_iteration": int(b["iteration"]), "auc": C.fnum(b["auc"]),
                                                            "row": b, "updated_at": C.now()}))
        return row

    # ---------------------------------------------------------------------------------------------- loop
    def pending_scans(self, ended: str | None) -> list[int]:
        done = {int(r["iteration"]) for r in self.rows() if r["purpose"] == "scan" and
                (r["status"] == "ok" or self.tries(int(r["iteration"]), "scan") >= self.a.tries)}
        cks = self.stable()
        grid = [n for n in cks if n % self.a.every == 0 and n > int(self.a.min_iteration)]
        if ended and cks and cks[-1] not in grid:
            grid.append(cks[-1])
        todo = [n for n in grid if n not in done]
        if self.a.all:
            return todo
        return todo[-1:] if todo and (not done or todo[-1] > max(done)) else []

    def final_step(self) -> bool:
        b = self.best()
        if b is None:
            self.log("training ended and no checkpoint evaluated ok: nothing to finalise")
            return True
        it = int(b["iteration"])
        finals = [r for r in self.ok_rows() if int(r["iteration"]) == it and r["purpose"] == "final"]
        if len(finals) < self.a.final_repeats and self.tries(it, "final") < self.a.final_repeats + self.a.tries:
            self.evaluate(it, "final")
            return False
        evals = [r for r in self.ok_rows() if int(r["iteration"]) == it]
        mean = {k: sum(C.fnum(r[k]) for r in evals) / len(evals) for k in C.METRICS}
        C.write_atomic(self.eval / "watch_final.json", C.dump({
            "item": self.cfg["item"], "episode": self.ep, "best_iteration": it, "n_evals": len(evals),
            "mean": mean, "evals": evals, "metric_keys": self.keys, "metric_keys_from": self.keys_from,
            "best_rule": "highest single scan AUC (ties -> earlier iteration)", "written_at": C.now()}))
        self.log(f"final: model_{it} mean over {len(evals)} evaluations {mean}")
        return True

    def loop(self) -> int:
        self.log(f"watching {self.run} ({self.cfg['item']}, episode {self.ep}); metrics from {self.keys_from}")
        while True:
            if (self.eval / "STOP").exists():
                self.log("STOP file: exiting")
                return 0
            ended = self.ended()
            todo = self.pending_scans(ended)
            if todo:
                for it in todo:
                    row = self.evaluate(it, "scan")
                    if self.a.dry_run or (self.eval / "STOP").exists():
                        break
                    if row.get("status") == "failed":
                        time.sleep(60)  # a technical failure is retried (--tries), not at once
                if self.a.dry_run:
                    return 0
                continue
            if ended:
                self.log(f"training ended: {ended}")
                if self.a.dry_run:
                    return 0
                while not self.final_step():
                    if (self.eval / "STOP").exists():
                        return 0
                return 0
            if self.a.once or self.a.dry_run:
                return 0
            time.sleep(self.a.poll)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_site_arg(p)
    p.add_argument("name", nargs="?", help="fork name (<T1_WORK>/forks/<name>)")
    p.add_argument("--run-dir", help="a run directory instead of a fork name (needs --item)")
    p.add_argument("--item", help="item of the run (default: the fork's)")
    p.add_argument("--ended", action="store_true", help="with --run-dir: treat the training as ended")
    p.add_argument("--gpu", type=int, help="evaluate on this GPU only (default: the usable GPU with most free memory)")
    p.add_argument("--every", type=int, default=2000)
    p.add_argument("--min-iteration", type=int, default=0, help="ignore checkpoints at or below this (the fork point)")
    p.add_argument("--all", action="store_true", help="evaluate every grid checkpoint, not only the newest")
    p.add_argument("--final-repeats", type=int, default=2)
    p.add_argument("--tries", type=int, default=2, help="technical tries per evaluation")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--once", action="store_true", help="evaluate what is ready now, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the next evaluation command; write nothing")
    args = p.parse_args()
    if not args.name and not args.run_dir:
        p.error("give a fork name or --run-dir")
    site = C.load_site(args.site)
    return Watch(args, site).loop()


if __name__ == "__main__":
    sys.exit(main())
