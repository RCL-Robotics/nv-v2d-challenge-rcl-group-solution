#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the evaluation inputs against the line spec before evaluating or packing (standard library only).

* The submission kit (``--kit``): every file listed in ``KIT_MANIFEST.json`` has its recorded sha256, the manifest's
  evaluator commit is the line's STOCK_RUNTIME_COMMIT, and the kit's ``evaluate_policy.py`` has the line's
  EVAL_SCRIPT_SHA256.
* The stock runtime (``--runtime``, ``.../robotic_grounding/flash_chord``): its ``scripts/evaluate_policy.py`` has
  EVAL_SCRIPT_SHA256 and, when it is a git checkout, its HEAD is STOCK_RUNTIME_COMMIT. Only ``git rev-parse`` is
  run; nothing is written.

usage: check_eval_inputs.py [--line FILE] [--kit DIR] [--runtime DIR]
Exit 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_line_spec(path: Path) -> dict[str, str]:
    spec = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        spec[key.strip()] = value.strip().strip('"').strip("'")
    return spec


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_kit(kit: Path, commit: str, eval_sha: str) -> list[str]:
    problems = []
    manifest_path = kit / "KIT_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        return [f"cannot read {manifest_path}: {error}"]
    files = manifest.get("files") or {}
    for relative, expected in sorted(files.items()):
        path = kit / relative
        if not path.is_file():
            problems.append(f"kit file missing: {relative}")
        elif sha256(path) != expected:
            problems.append(f"kit file differs from KIT_MANIFEST.json: {relative}")
    evaluator = manifest.get("evaluator") or {}
    if evaluator.get("commit") != commit:
        problems.append(f"kit evaluator commit {evaluator.get('commit')} is not the line's {commit}")
    kit_eval = kit / evaluator.get("kit_path", "flash_chord/scripts/evaluate_policy.py")
    if not kit_eval.is_file():
        problems.append(f"kit evaluator missing: {kit_eval}")
    elif sha256(kit_eval) != eval_sha:
        problems.append(f"kit evaluator {kit_eval} does not have the line's EVAL_SCRIPT_SHA256")
    print(f"kit {kit}: {len(files)} manifest files checked, evaluator commit {evaluator.get('commit')}")
    return problems


def check_runtime(runtime: Path, commit: str, eval_sha: str) -> list[str]:
    problems = []
    script = runtime / "scripts" / "evaluate_policy.py"
    if not script.is_file():
        problems.append(f"stock runtime evaluator missing: {script}")
    elif sha256(script) != eval_sha:
        problems.append(f"stock runtime evaluator {script} does not have the line's EVAL_SCRIPT_SHA256")
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(["git", "-C", str(runtime), "rev-parse", "HEAD"], capture_output=True, text=True,
                              env=environment, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        head = None
        print(f"stock runtime {runtime}: HEAD not checked ({error})")
    if head is not None and head.returncode == 0:
        sha = head.stdout.strip()
        if sha != commit:
            problems.append(f"stock runtime HEAD {sha} is not the line's {commit}")
        print(f"stock runtime {runtime}: HEAD {sha}")
    elif head is not None:
        print(f"stock runtime {runtime}: not a git checkout, HEAD not checked (only the evaluator digest)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--line", type=Path, default=HERE / "lines" / "c3r2_tier1.env")
    parser.add_argument("--kit", type=Path, help="extracted submission kit (holds KIT_MANIFEST.json)")
    parser.add_argument("--runtime", type=Path, help="stock runtime .../robotic_grounding/flash_chord")
    args = parser.parse_args()
    if args.kit is None and args.runtime is None:
        parser.error("give --kit and/or --runtime")
    spec = read_line_spec(args.line)
    commit, eval_sha = spec["STOCK_RUNTIME_COMMIT"], spec["EVAL_SCRIPT_SHA256"]
    problems = []
    if args.kit is not None:
        problems += check_kit(args.kit, commit, eval_sha)
    if args.runtime is not None:
        problems += check_runtime(args.runtime, commit, eval_sha)
    for problem in problems:
        print(f"FAIL {problem}")
    print("evaluation inputs: OK" if not problems else f"evaluation inputs: {len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
