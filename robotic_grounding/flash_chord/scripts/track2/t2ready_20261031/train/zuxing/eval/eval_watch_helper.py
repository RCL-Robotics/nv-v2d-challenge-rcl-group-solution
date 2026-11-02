#!/usr/bin/env python3
"""Helper of eval_watch.sh (stdlib only; see EVAL_NOTES.md).

Subcommands:
  todo    Scan <runs>/ep<NN>/ and <out>/metrics.csv. Prints "NOTE ..." lines, one "ITEM <episode> <iteration>
          <attempt>" line per item ready to evaluate (fresh items first, oldest iteration first, then retries), and
          one "SUMMARY complete=<0|1> ..." line. Scheduled = every existing model_<N>.safetensors with
          N % --every == 0, plus the checkpoint named by the run's final.json. complete=1 when every episode has a
          valid final.json, its final checkpoint exists, and every scheduled item is finished (ok, seq_mismatch, or
          failed twice). Read-only.
  record  Read one eval_all.py output dir (<out>/ep<NN>/it<N>/), append one row to <out>/metrics.csv, delete the
          exported stock-format copy after a successful evaluation, print "RESULT <status> <summary>".
  table   Print the ok rows of metrics.csv, with the baseline arm's metrics at the same iteration if given.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import time
from pathlib import Path

CKPT_RE = re.compile(r"^model_(\d+)\.safetensors$")
COLUMNS = [
    "episode", "iteration", "add_auc", "sp_sr", "mp_sr", "mppe_cm", "status", "seconds", "log",
    "object", "attempt", "final", "chord_sr", "export_max_rel_diff", "stock_verified", "source_sha256",
    "finished_at", "note",
]  # fmt: skip
TERMINAL = {"ok", "seq_mismatch"}  # never evaluated again
MAX_ATTEMPTS = 2  # first try + one retry
# ours -> kit leaderboard name; the evaluator key comes from the kit's v2dlb/report_schema.py REPORT_METRIC_KEY
LEADERBOARD = {"add_auc": "add_auc", "sp_sr": "spider_sr", "mp_sr": "maniptrans_sr", "mppe_cm": "mppe_cm"}
METRICS_FALLBACK = {
    "add_auc": "add_auc", "sp_sr": "spider_sr_uncentered", "mp_sr": "maniptrans_sr", "mppe_cm": "mean_per_point_error_cm",
}  # fmt: skip
EXPORT_DELETE_MIN_BYTES = 1 << 20  # delete the exported copy only if it is large (it is ~28 MB)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def episode_list(text: str) -> list[int]:
    return [int(part) for part in re.split(r"[,\s]+", text.strip()) if part]


def load_manifest(path: Path) -> dict[int, dict]:
    data = json.loads(Path(path).read_text())
    return {
        int(r["episode_index"]): {"object": str(r["object"]), "sequence": str(r.get("sequence_id", key))}
        for key, r in data.items()
    }


def read_final(run_dir: Path) -> int | None:
    """Iteration named by <run_dir>/final.json (validated like eval_all.py read_final_marker); None if absent,
    -1 if present but unusable."""
    path = run_dir / "final.json"
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return -1
    iteration = data.get("iteration") if isinstance(data, dict) else None
    if (
        isinstance(iteration, int)
        and not isinstance(iteration, bool)
        and iteration > 0
        and data.get("checkpoint") == f"model_{iteration}.safetensors"
    ):
        return iteration
    return -1


def load_state(csv_path: Path) -> dict[tuple[int, int], dict]:
    state: dict[tuple[int, int], dict] = {}
    if not csv_path.is_file():
        return state
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                key = (int(row["episode"]), int(row["iteration"]))
            except (KeyError, TypeError, ValueError):
                continue
            entry = state.setdefault(key, {"attempts": 0, "terminal": False, "gave_up": False, "last_fail": None})
            entry["attempts"] += 1
            if row.get("status") in TERMINAL:
                entry["terminal"] = True
            else:
                try:
                    entry["last_fail"] = dt.datetime.fromisoformat(row.get("finished_at") or "").timestamp()
                except ValueError:
                    entry["last_fail"] = 0.0
    for entry in state.values():
        if not entry["terminal"] and entry["attempts"] >= MAX_ATTEMPTS:
            entry["terminal"] = entry["gave_up"] = True
    return state


def metric_sources(kit: Path | None) -> tuple[dict[str, str], str]:
    """Ours -> evaluator results key, from the kit's REPORT_METRIC_KEY parsed with ast (importing the module would
    write __pycache__ into the kit); falls back to the 8c778f9 kit's mapping."""
    if kit is not None:
        try:
            for node in ast.parse((kit / "v2dlb/report_schema.py").read_text()).body:
                targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
                if any(getattr(target, "id", None) == "REPORT_METRIC_KEY" for target in targets):
                    table = ast.literal_eval(node.value)
                    return {ours: table[name] for ours, name in LEADERBOARD.items()}, "kit"
        except Exception:  # noqa: BLE001
            pass
    return dict(METRICS_FALLBACK), "fallback"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnum(value):
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, spec: str) -> str:
    value = fnum(value)
    return "n/a" if value is None else format(value, spec)


# ---------------------------------------------------------------------------------------------------- todo
def cmd_todo(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    episodes = episode_list(args.episodes)
    unknown = [ep for ep in episodes if ep not in manifest]
    if unknown:
        print(f"NOTE episodes {unknown} are not in {args.manifest}")
        return 2
    state = load_state(args.out / "metrics.csv")
    t = time.time()
    ready, notes = [], []
    waiting = scheduled = done = gave_up = 0
    complete = True
    for order, ep in enumerate(episodes):
        run = args.runs / f"ep{ep:02d}"
        if not run.is_dir():
            complete = False
            notes.append(f"ep{ep:02d}: no run dir {run} yet")
            continue
        final = read_final(run)
        if final == -1:
            notes.append(f"ep{ep:02d}: unusable {run / 'final.json'} (ignored)")
            final = None
        if final is None:
            complete = False
        checkpoints: dict[int, Path] = {}
        for path in run.iterdir():
            match = CKPT_RE.match(path.name)
            if match:
                iteration = int(match.group(1))
                if iteration % args.every == 0 or iteration == final:
                    checkpoints[iteration] = path
        if final is not None:
            if final in checkpoints:
                notes.append(f"ep{ep:02d}: final.json -> model_{final}")
            else:
                complete = False
                notes.append(f"ep{ep:02d}: final.json names model_{final}, which does not exist")
        for iteration in sorted(checkpoints):
            scheduled += 1
            entry = state.get((ep, iteration))
            if entry and entry["terminal"]:
                done += 1
                gave_up += entry["gave_up"]
                continue
            complete = False
            try:
                age = t - checkpoints[iteration].stat().st_mtime
            except OSError:
                continue
            if age < args.min_age:
                waiting += 1
                continue
            if entry and entry["last_fail"] is not None and t - entry["last_fail"] < args.retry_delay:
                waiting += 1
                continue
            attempt = (entry["attempts"] if entry else 0) + 1
            ready.append((attempt > 1, iteration, order, ep, attempt))
    ready.sort()
    for note in notes:
        print(f"NOTE {note}")
    for _, iteration, _, ep, attempt in ready:
        print(f"ITEM {ep} {iteration} {attempt}")
    print(
        f"SUMMARY complete={int(complete)} ready={len(ready)} waiting={waiting} finished={done}/{scheduled} "
        f"gave_up={gave_up}"
    )
    return 0


# -------------------------------------------------------------------------------------------------- record
def cmd_record(args: argparse.Namespace) -> int:
    ep, it, work = args.episode, args.iteration, args.work
    info = load_manifest(args.manifest).get(ep, {})
    sources, _ = metric_sources(args.kit)
    row: dict = {column: "" for column in COLUMNS}
    row.update(
        episode=ep, iteration=it, status="failed", seconds=args.seconds, log=str(args.log),
        object=info.get("object", ""), attempt=args.attempt, finished_at=now_iso(),
    )  # fmt: skip
    row["final"] = 1 if read_final(args.runs / f"ep{ep:02d}") == it else ""
    problems: list[str] = []
    if args.exit != 0:
        problems.append(f"exit {args.exit}" + (" (timeout)" if args.exit in (124, 137) else ""))

    report_row, data = None, {}
    try:
        rows = json.loads((work / "eval_report.json").read_text()).get("episodes", [])
        report_row = next(r for r in rows if int(r.get("episode", -1)) == ep)
    except (OSError, ValueError, TypeError, AttributeError, StopIteration):
        problems.append("no eval_report.json row for this episode")
    if report_row is not None:
        row["export_max_rel_diff"] = report_row.get("export_max_rel_diff")
        row["stock_verified"] = report_row.get("export_stock_verified")
        if report_row.get("status") != "evaluated":
            problems.append(f"eval_all status {report_row.get('status')}")
        if report_row.get("export_stock_verified") is not True:
            problems.append(f"export stock_verified={report_row.get('export_stock_verified')}")
        if str(report_row.get("iteration")) != str(it):
            problems.append(f"eval_all evaluated iteration {report_row.get('iteration')}, not {it}")
        try:
            data = json.loads((work / "results" / f"episode_{ep:06d}.json").read_text())
        except (OSError, ValueError):
            problems.append("no results JSON")
    try:
        export_report = json.loads((work / "export" / f"ep{ep:02d}" / "export_report.json").read_text())
        row["source_sha256"] = export_report.get("source_sha256", "")
        exported_iteration = export_report.get("iteration")
        if exported_iteration not in (None, "") and str(exported_iteration) != str(it):
            problems.append(f"export_report says source iteration {exported_iteration}, not {it}")
    except (OSError, ValueError):
        pass  # eval_all status 'evaluated' already implies an ok export

    metrics = data.get("metrics") or {} if isinstance(data, dict) else {}
    for ours, key in sources.items():
        row[ours] = fnum(metrics.get(key))
    row["chord_sr"] = fnum(metrics.get("chord_sr"))
    exported = Path(report_row.get("exported_checkpoint") or "") if report_row else Path("")
    evaluated = Path(report_row.get("evaluated_checkpoint") or "") if report_row else Path("")
    if data:
        if evaluated.is_file() and data.get("checkpoint_sha256") != sha256(evaluated):
            problems.append("results JSON is for another checkpoint")
        missing = [ours for ours in sources if row[ours] is None]
        if missing:
            problems.append(f"leaderboard metrics missing: {missing}")

    status = "failed" if problems else "ok"
    # The evaluator takes the motion from the checkpoint's own config, not from the episode number: check that the
    # evaluated parquet is this episode's manifest sequence (a wrong motion in the run dir would be scored silently).
    sequence, parquet = info.get("sequence", ""), str(data.get("parquet", "")) if isinstance(data, dict) else ""
    if status == "ok" and sequence and sequence not in parquet:
        status = "seq_mismatch"
        problems.append(f"evaluated parquet {parquet!r} is not episode {ep}'s sequence {sequence!r}")
    row["status"] = status
    row["note"] = "; ".join(problems)

    deleted = ""
    if status == "ok" and not args.keep_export:
        try:
            if (
                exported.is_file()
                and not exported.is_symlink()
                and exported.resolve().is_relative_to(work.resolve())
                and exported.stat().st_size >= EXPORT_DELETE_MIN_BYTES
            ):
                size = exported.stat().st_size
                exported.unlink()
                deleted = f", deleted export ({size / 2**20:.0f} MB)"
        except OSError as error:
            deleted = f", export not deleted ({error})"

    csv_path = args.out / "metrics.csv"
    new = not csv_path.is_file() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    where = f"ep{ep:02d} ({row['object']}) model_{it}{' (final)' if row['final'] else ''} attempt {args.attempt}"
    if status == "ok":
        text = (
            f"{where}: ok AUC {fmt(row['add_auc'], '.4f')} SP-SR {fmt(row['sp_sr'], '.3f')} "
            f"MP-SR {fmt(row['mp_sr'], '.3f')} MPPE {fmt(row['mppe_cm'], '.2f')} cm | chord_sr "
            f"{fmt(row['chord_sr'], '.3f')}, export rel diff {fmt(row['export_max_rel_diff'], '.1e')}, "
            f"{args.seconds} s{deleted}"
        )
    else:
        text = f"{where}: {status.upper()} ({row['note']}); {args.seconds} s; log {args.log}"
    print(f"RESULT {status} {text}")
    return 0


# --------------------------------------------------------------------------------------------------- table
def cmd_table(args: argparse.Namespace) -> int:
    path = args.out / "metrics.csv"
    if not path.is_file():
        print("no metrics.csv yet")
        return 0
    with path.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("status") == "ok"]
    baseline: dict[tuple[int, int], dict] = {}
    if args.baseline and args.baseline.is_file():
        with args.baseline.open(newline="") as stream:
            for r in csv.DictReader(stream):
                if r.get("arm") == args.baseline_arm and r.get("status", "ok") == "ok":
                    try:
                        baseline[(int(r["episode"]), int(r["iteration"]))] = r
                    except (KeyError, ValueError):
                        continue
    rows.sort(key=lambda r: (int(r["episode"]), int(r["iteration"])))
    head = f"{'ep':>3} {'iter':>6} {'AUC':>7} {'SP-SR':>6} {'MP-SR':>6} {'MPPE':>7}"
    if baseline:
        head += f" | {args.baseline_arm}: {'AUC':>7} {'SP-SR':>6} {'MP-SR':>6} {'MPPE':>7}"
    print(head)
    for r in rows:
        line = (
            f"{int(r['episode']):>3} {int(r['iteration']):>6} {fmt(r['add_auc'], '.4f'):>7} {fmt(r['sp_sr'], '.3f'):>6} "
            f"{fmt(r['mp_sr'], '.3f'):>6} {fmt(r['mppe_cm'], '.2f'):>7}" + (" final" if r.get("final") else "")
        )
        b = baseline.get((int(r["episode"]), int(r["iteration"])))
        if b:
            line += (
                f" | {'':>{len(args.baseline_arm) + 1}} {fmt(b.get('add_auc'), '.4f'):>7} {fmt(b.get('sp_sr'), '.3f'):>6} "
                f"{fmt(b.get('mp_sr'), '.3f'):>6} {fmt(b.get('mppe_cm'), '.2f'):>7}"
            )
        print(line)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    todo = sub.add_parser("todo")
    todo.add_argument("--runs", type=Path, required=True, help="dir holding the ep<NN>/ training run dirs")
    todo.add_argument("--out", type=Path, required=True)
    todo.add_argument("--manifest", type=Path, required=True)
    todo.add_argument("--episodes", required=True, help='e.g. "19 4 1"')
    todo.add_argument("--every", type=int, default=2000)
    todo.add_argument("--min-age", type=float, default=60.0, help="s since the checkpoint's mtime")
    todo.add_argument("--retry-delay", type=float, default=600.0, help="s between a failure and its retry")

    record = sub.add_parser("record")
    record.add_argument("--runs", type=Path, required=True)
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--work", type=Path, required=True, help="the eval_all.py --arm-dir of this item")
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--kit", type=Path)
    record.add_argument("--episode", type=int, required=True)
    record.add_argument("--iteration", type=int, required=True)
    record.add_argument("--attempt", type=int, required=True)
    record.add_argument("--exit", type=int, required=True)
    record.add_argument("--seconds", type=int, required=True)
    record.add_argument("--log", type=Path, required=True)
    record.add_argument("--keep-export", action="store_true")

    table = sub.add_parser("table")
    table.add_argument("--out", type=Path, required=True)
    table.add_argument("--baseline", type=Path)
    table.add_argument("--baseline-arm", default="C3r2_wide")

    args = parser.parse_args(argv)
    return {"todo": cmd_todo, "record": cmd_record, "table": cmd_table}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
