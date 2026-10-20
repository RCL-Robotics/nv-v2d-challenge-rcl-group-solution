#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the scene.support rule against the line spec and the recorded runs (standard library only).

Rule (train_all.sh): an episode is floor-only, and trains with ``scene.support=false``, when
``<motion root>/whole_body/reconstructed_stage/<seq>_support.usda`` is missing or contains no ``def Cylinder`` /
``def Cube`` prim. This script applies the rule independently to the converted data and compares the result with
the line's EXPECTED_FLOOR_ONLY and, optionally, with the flags the original runs recorded (W&B args).

usage: check_support_rule.py --motion-root DIR [--records DIR] [--line FILE]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIM = re.compile(r"def (Cylinder|Cube)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-root", type=Path, required=True)
    ap.add_argument("--records", type=Path, help="the original arm run directory (epNN/wandb/...)")
    ap.add_argument("--line", type=Path, default=HERE.parent / "lines" / "c3r2_tier1.env")
    a = ap.parse_args()
    spec = {}
    for raw in a.line.read_text().splitlines():
        if "=" in raw and not raw.lstrip().startswith("#"):
            key, value = raw.split("=", 1)
            spec[key.strip()] = value.strip().strip('"')
    expected = sorted(spec["EXPECTED_FLOOR_ONLY"].split())
    manifest = json.loads((a.motion_root / "whole_body/soma/track2_manifest.json").read_text())
    computed = []
    for seq, record in manifest.items():
        usda = a.motion_root / "whole_body/reconstructed_stage" / f"{seq}_support.usda"
        if not (usda.is_file() and PRIM.search(usda.read_text(errors="replace"))):
            computed.append(f"{int(record['episode_index']):02d}")
    computed.sort()
    ok = computed == expected
    print(f"rule on the data ({len(manifest)} episodes): {' '.join(computed)} ({len(computed)} floor-only)")
    print(f"line spec EXPECTED_FLOOR_ONLY:           {' '.join(expected)} ({len(expected)})")
    if a.records is not None:
        recorded = []
        for meta in sorted(a.records.glob("ep[0-9][0-9]/wandb/run-*/files/wandb-metadata.json")):
            if "scene.support=false" in json.loads(meta.read_text())["args"]:
                recorded.append(meta.relative_to(a.records).parts[0][2:])
        recorded = sorted(set(recorded))
        print(f"recorded runs with scene.support=false:  {' '.join(recorded)} ({len(recorded)})")
        ok = ok and recorded == expected
    print("OK" if ok else "FAIL: the support rule does not reproduce the recorded floor-only episodes")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
