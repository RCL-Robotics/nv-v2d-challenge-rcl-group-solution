#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the reference data stored inside exported checkpoints exists on this machine (standard library only).

Every FlashCHORD checkpoint records its training config (safetensors metadata ``user.resolved_config_json``),
including the ABSOLUTE path of the retargeted episode, ``task.parquet``. The official evaluator loads the
reference motion from that path. The stock harness (flash_chord/evaluation/harness.py ``_resolve_parquet`` at
runtime commit 8c778f9) rewrites it only when ``--source-root`` is given AND the stored path is under
``/workspace`` (or relative); any other absolute path is used as is. Checkpoints trained on another machine therefore
evaluate only where that path exists: recreate it (a symlink or bind mount of the derived motion-data root at the
printed prefix), then rerun. See README "Absolute task.parquet paths".

usage: check_task_paths.py <checkpoint or directory> [...]   (directories are searched for *.safetensors)
Exit 0 when every stored path exists, 1 otherwise (each missing path and its data-root prefix are printed).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


MAX_HEADER_BYTES = 100_000_000  # safetensors caps its JSON header at 100 MB


def stored_config(path: Path) -> dict:
    total = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("not a safetensors file (shorter than 8 bytes)")
        (size,) = struct.unpack("<Q", prefix)
        if size > min(MAX_HEADER_BYTES, total - 8):
            raise ValueError(f"not a safetensors file (header length {size} for a file of {total} bytes)")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("not a safetensors file (header is not a JSON object)")
    metadata = header.get("__metadata__") or {}
    text = metadata.get("user.resolved_config_json") if isinstance(metadata, dict) else None
    if not isinstance(text, str):
        raise ValueError("no user.resolved_config_json in the safetensors metadata")
    config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError("user.resolved_config_json is not a JSON object")
    return config


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    files: list[Path] = []
    for item in map(Path, argv):
        files.extend(sorted(item.rglob("*.safetensors")) if item.is_dir() else [item])
    if not files:
        print("no checkpoints found", file=sys.stderr)
        return 1
    missing, prefixes = 0, set()
    for path in files:
        try:
            config = stored_config(path)
        except (OSError, ValueError, struct.error) as error:
            print(f"{path}: unreadable config ({error})")
            missing += 1
            continue
        for key in ("task", "reference"):
            section = config.get(key)
            parquet = section.get("parquet") if isinstance(section, dict) else None
            if not isinstance(parquet, str) or not parquet:
                continue
            ok = Path(parquet).is_file()
            if not ok:
                missing += 1
                marker = "/whole_body/"
                prefixes.add(parquet.split(marker)[0] if marker in parquet else str(Path(parquet).parent))
                print(f"MISSING {path}: {key}.parquet = {parquet}")
    print(f"checked {len(files)} checkpoints: {'all stored task paths exist' if not missing else f'{missing} missing'}")
    for prefix in sorted(prefixes):
        print(f"  the checkpoints expect the derived motion-data root at {prefix} (symlink or bind-mount it there)")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
