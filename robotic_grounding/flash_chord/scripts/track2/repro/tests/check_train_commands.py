#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that train_all.sh issues exactly the training commands the submitted runs recorded (standard library only).

Compares, per episode, the Hydra argument list printed by ``train_all.sh --dry-run`` with:

* the ``args`` that W&B stored for the run (``<records>/epNN/wandb/run-*/files/wandb-metadata.json``), together with
  the recorded program (``scripts/train_rl.py``) and git commit (the line's TRAINED_COMMIT);
* optionally the queue log of the original run (``--queue-log``): its header line names the arm, experiment and
  iteration cap, and its ``start <ARM>/epNN (<object>)`` lines the object of every episode.

Paths are compared after replacing the motion-data root and the runs root by placeholders (the recorded roots are
read from each record's task.parquet / output_dir; the dry run's from --motion-root / --runs-root). With
``--logging ignore`` the W&B destination keys (logging.mode / logging.entity / logging.project) are dropped on both
sides; with ``--logging exact`` (default) they must match too, so pass the recorded W&B settings to the dry run.

usage: check_train_commands.py --dry-run FILE --records DIR --motion-root DIR --runs-root DIR
                               [--queue-log FILE] [--line FILE] [--logging exact|ignore]
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION_KEYS = ("logging.mode=", "logging.entity=", "logging.project=")


def read_line_spec(path: Path) -> dict[str, str]:
    spec = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        spec[key.strip()] = value.strip().strip('"').strip("'")
    return spec


def normalize(args: list[str], motion_root: str, runs_root: str, logging: str) -> list[str]:
    out = []
    for arg in args:
        if logging == "ignore" and arg.startswith(DESTINATION_KEYS):
            continue
        arg = arg.replace(motion_root.rstrip("/") + "/", "<MOTION>/").replace(runs_root.rstrip("/") + "/", "<RUNS>/")
        out.append(arg)
    return out


def dry_run_commands(path: Path) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^\[train ep(\d\d) gpu[^\]]*\] (.*)$", line)
        if not match:
            continue
        words = shlex.split(match.group(2))
        index = words.index("scripts/train_rl.py")
        if words[index - 1] != "python":
            raise ValueError(f"ep{match.group(1)}: the program is not run as `python scripts/train_rl.py`: {line}")
        commands[match.group(1)] = words[index + 1 :]
    return commands


def recorded_roots(args: list[str], arm: str) -> tuple[str, str]:
    parquet = next(a for a in args if a.startswith("task.parquet="))
    motion = parquet.split("=", 1)[1].strip("'").split("/whole_body/soma/")[0]
    output = next(a for a in args if a.startswith("output_dir=")).split("=", 1)[1]
    runs = output.rsplit(f"/{arm}/", 1)[0]
    return motion, runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", type=Path, required=True, help="output of train_all.sh --dry-run")
    ap.add_argument("--records", type=Path, required=True, help="the original arm run directory (epNN/wandb/...)")
    ap.add_argument("--motion-root", required=True, help="V2D_MOTION used for the dry run")
    ap.add_argument("--runs-root", required=True, help="--runs used for the dry run")
    ap.add_argument("--queue-log", type=Path)
    ap.add_argument("--line", type=Path, default=HERE.parent / "lines" / "c3r2_tier1.env")
    ap.add_argument("--logging", choices=("exact", "ignore"), default="exact")
    a = ap.parse_args()
    spec = read_line_spec(a.line)
    arm = spec["ARM"]
    ours = dry_run_commands(a.dry_run)
    problems: list[str] = []
    checked = 0
    for episode_dir in sorted(p for p in a.records.glob("ep[0-9][0-9]") if p.is_dir()):
        episode = episode_dir.name[2:]
        metas = sorted(episode_dir.glob("wandb/run-*/files/wandb-metadata.json"))
        if not metas:
            problems.append(f"ep{episode}: no wandb-metadata.json under {episode_dir}/wandb")
            continue
        if episode not in ours:
            problems.append(f"ep{episode}: recorded but missing from the dry run")
            continue
        mine = normalize(ours[episode], a.motion_root, a.runs_root, a.logging)
        for meta_path in metas:
            meta = json.loads(meta_path.read_text())
            recorded = meta["args"]
            motion, runs = recorded_roots(recorded, arm)
            theirs = normalize(recorded, motion, runs, a.logging)
            if not str(meta.get("program", "")).endswith("scripts/train_rl.py"):
                problems.append(f"ep{episode}: recorded program {meta.get('program')!r}")
            commit = meta.get("git", {}).get("commit", "")
            if commit != spec["TRAINED_COMMIT"]:
                problems.append(f"ep{episode}: recorded commit {commit} != TRAINED_COMMIT {spec['TRAINED_COMMIT']}")
            if mine != theirs:
                problems.append(f"ep{episode}: arguments differ\n    ours:     {mine}\n    recorded: {theirs}")
            checked += 1
    extra = sorted(set(ours) - {p.name[2:] for p in a.records.glob("ep[0-9][0-9]")})
    if extra:
        problems.append(f"dry run has episodes without a record: {extra}")
    if a.queue_log is not None:
        text = a.queue_log.read_text()
        header = re.search(rf"ARMS=\[{re.escape(arm)}:(\S+)\] ITERS=(\d+)", text)
        if header is None:
            problems.append(f"{a.queue_log}: no queue header for arm {arm}")
        else:
            if header.group(1) != spec["EXPERIMENT"] or header.group(2) != spec["MAX_ITERATIONS"]:
                problems.append(f"queue header experiment/iterations {header.groups()} != line spec")
            code = re.search(rf"ARMS=\[{re.escape(arm)}:[^\n]*CODE=\S+ \(([0-9a-f]+)\)", text)
            if code is None or not spec["TRAINED_COMMIT"].startswith(code.group(1)):
                problems.append(f"queue header code commit {code.group(1) if code else None} != TRAINED_COMMIT")
        started = dict(re.findall(rf"start {re.escape(arm)}/ep(\d\d) \(([^)]+)\)", text))
        for episode, args in sorted(ours.items()):
            run_name = next((x.split("=", 1)[1] for x in args if x.startswith("logging.run_name=")), "")
            if episode in started and run_name != f"ep{episode}_{started[episode]}":
                problems.append(f"ep{episode}: run_name {run_name} but the queue log started object {started[episode]}")
        missing = sorted(set(ours) - set(started))
        if missing:
            problems.append(f"queue log has no start line for {missing}")
    floor_only = sorted(e for e, args in ours.items() if "scene.support=false" in args)
    print(f"compared {checked} recorded runs with {len(ours)} dry-run commands (logging keys: {a.logging})")
    print(f"scene.support=false in the dry run: {' '.join(floor_only)} ({len(floor_only)} episodes)")
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("OK: every per-episode training command equals the recorded one (paths normalized)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
