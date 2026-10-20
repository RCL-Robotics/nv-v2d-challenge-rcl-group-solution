#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run-directory checks for the Track 2 reproduction drivers (CPU only, numpy + safetensors).

Same logic as the helper the C3r2 training queue, export and final-evaluation drivers used.

  queue_helper.py done <run dir> <iters>
      exit 0 when the job is finished, printing why: <run dir>/final.json (written by the training runner after
      its final checkpoint) says it stopped early (reason help_zero_tail) or reached max_iterations >= <iters>,
      and its checkpoint exists; or model_<iters>.safetensors exists (runs from before final.json). Exit 1 otherwise.
  queue_helper.py resume <run dir> [--quarantine]
      print the newest model_<N>.safetensors that is safe to resume from: readable, every floating-point tensor
      finite, and, when the run keeps RSI sidecars (any rsi_*.safetensors), rsi_<N>.safetensors present, readable
      and finite (the guarded runner refuses non-finite state, and an RSI resume needs the sidecar). Every skipped
      checkpoint is reported on stderr. With --quarantine, newer model_<N>/rsi_<N> files that are unreadable or
      non-finite are MOVED (never deleted) to <run dir>/nonfinite_quarantine/. Exit 0 and print nothing only when the
      run dir holds no model_*.safetensors (after quarantine): a fresh start is safe. Exit 3 (prints nothing) when
      model_* files remain but none is usable: a fresh start would overwrite them, so the job must not start.
  queue_helper.py final <run dir> [--min-age SECONDS]
      print the final iteration named by <run dir>/final.json when the marker is valid and its checkpoint exists
      (and, with --min-age, is at least that old); exit 1 otherwise. This is the checkpoint selection rule of the
      line (pre-registered final checkpoint, no best-of).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

CHECKPOINT = re.compile(r"^model_(\d+)\.safetensors$")


def finite_problem(path: Path) -> str | None:
    """None when every floating-point tensor in ``path`` is finite; otherwise a short reason."""
    import numpy as np
    from safetensors.numpy import load_file

    try:
        tensors = load_file(str(path))
    except Exception as error:  # noqa: BLE001 - truncated or foreign file
        return f"unreadable ({type(error).__name__}: {str(error)[:120]})"
    bad = {
        key: int(value.size - np.count_nonzero(np.isfinite(value)))
        for key, value in tensors.items()
        if value.dtype.kind == "f" and not np.isfinite(value).all()
    }
    if bad:
        shown = ", ".join(f"{key}: {count}" for key, count in sorted(bad.items())[:4])
        return f"non-finite ({len(bad)} tensors: {shown}{', ...' if len(bad) > 4 else ''})"
    return None


def final_marker(run: Path) -> dict | None:
    path = run / "final.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        iteration = data["iteration"]
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or data["checkpoint"] != f"model_{iteration}.safetensors"
        ):
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def done(run: Path, iters: int) -> int:
    final = final_marker(run)
    if final is not None and (run / final["checkpoint"]).is_file():
        if final.get("reason") == "help_zero_tail":
            print(f"final.json: early stop at iteration {final['iteration']}")
            return 0
        if final.get("reason") == "max_iterations" and final["iteration"] >= iters:
            print(f"final.json: max_iterations {final['iteration']}")
            return 0
    if (run / f"model_{iters}.safetensors").is_file():
        print(f"model_{iters}.safetensors")
        return 0
    return 1


def resume(run: Path, quarantine: bool) -> int:
    checkpoints = sorted(
        ((int(m.group(1)), p) for p in run.glob("model_*.safetensors") if (m := CHECKPOINT.match(p.name))),
        reverse=True,
    )
    with_rsi = any(run.glob("rsi_*.safetensors"))
    for iteration, path in checkpoints:
        sidecar = run / f"rsi_{iteration}.safetensors"
        problem = finite_problem(path)
        bad = [path] if problem else []
        if problem is None and with_rsi:
            if not sidecar.is_file():
                problem = "no rsi sidecar (RSI resume needs it)"
            else:
                sidecar_problem = finite_problem(sidecar)
                if sidecar_problem:
                    problem, bad = f"{sidecar.name} {sidecar_problem}", [sidecar]
        if problem is None:
            print(path)
            return 0
        message = f"skip {path.name}: {problem}"
        if quarantine and bad:
            target = run / "nonfinite_quarantine"
            target.mkdir(exist_ok=True)
            moved = [p for p in (path, sidecar) if p.is_file()]
            for p in moved:
                shutil.move(str(p), str(target / p.name))
            message += f"; moved {', '.join(p.name for p in moved)} to {target.name}/"
        print(message, file=sys.stderr)
    remaining = sorted(
        (int(m.group(1)), p.name) for p in run.glob("model_*.safetensors") if (m := CHECKPOINT.match(p.name))
    )
    left = [name for _, name in remaining]
    if left:
        print(
            f"no usable checkpoint, but {len(left)} model_*.safetensors remain ({', '.join(left[-3:])}"
            f"{', ...' if len(left) > 3 else ''}): refusing a fresh start that would overwrite them",
            file=sys.stderr,
        )
        return 3
    if checkpoints:
        print("every checkpoint was quarantined: fresh start (the quarantined files are kept)", file=sys.stderr)
    return 0


def final(run: Path, min_age: float) -> int:
    marker = final_marker(run)
    if marker is None:
        return 1
    checkpoint = run / marker["checkpoint"]
    try:
        age = time.time() - checkpoint.stat().st_mtime
    except OSError:
        return 1
    if age < min_age:
        return 1
    print(marker["iteration"])
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "done":
        return done(Path(argv[1]), int(argv[2]))
    if len(argv) >= 2 and argv[0] == "resume":
        return resume(Path(argv[1]), "--quarantine" in argv[2:])
    if len(argv) >= 2 and argv[0] == "final":
        min_age = float(argv[argv.index("--min-age") + 1]) if "--min-age" in argv[2:] else 0.0
        return final(Path(argv[1]), min_age)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
