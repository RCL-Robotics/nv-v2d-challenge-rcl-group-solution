#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""show_selection.py: print the decision of the selection that chose a counted file, from its committed records.

  show_selection.py --tier 2|3

Tier 2: the final selection's SELECTION_R9.json (top-level decision fields), the 10 complete trials (summary_r9_trials.csv:
own-input and consensus-B means only) and the 30 items of set_items_r9.json. Tier 3: the 30 items of
set_items.json (kind, iteration, packed run, evaluated file). Read-only, standard library.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
TR = Path(__file__).resolve().parents[3] / "robotic_grounding/flash_chord/scripts/track2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, choices=(2, 3), required=True)
    a = ap.parse_args()
    if a.tier == 2:
        sel = TR / "t2r9_20261104/selection/round9_20261103"
        d = json.loads((sel / "SELECTION_R9.json").read_text())
        print("[SELECTION_R9.json]", ", ".join(f"{k}={json.dumps(v)}" for k, v in d.items()
                                                 if not isinstance(v, (dict, list)) or len(json.dumps(v)) < 120)[:1500])
        print("[summary_r9_trials.csv] (own input = each policy against its own Tier-2 input; consensus-B = refs/B)")
        for r in csv.DictReader((sel / "summary_r9_trials.csv").open()):
            print("   ", ", ".join(f"{k}={v}" for k, v in list(r.items())[:8]))
        s = json.loads((TR / "t2r9_20261104/set/set_items_r9.json").read_text())
        print(f"[set_items_r9.json] {s['what']}")
        print(f"  replaced in the final selection: {s['replaced_episodes']}; inputs: "
              f"{dict(collections.Counter(e['input'] for e in s['episodes']))}")
        for e in s["episodes"]:
            ev = e["evaluated"]
            print(f"  ep{e['episode']:02d} {e['job']:28} {e['input']:6} trained {e['train']['code_commit'][:7]} "
                  f"exported {e['export']['code_commit'][:7]} file {e['export']['sha256'][:12]} rollout {ev['round']} run {ev['run']}")
    else:
        d = json.loads((TR / "t3ready_20261102/set/set_items.json").read_text())
        print("[set_items.json]", d["what"][:200])
        print("  items:", dict(collections.Counter(e["kind"] for e in d["episodes"])),
              "; palm mitigation (training only):", [e["episode"] for e in d["episodes"] if e["mitigated"]])
        for e in d["episodes"]:
            print(f"  ep{e['episode']:02d} {e['object']:20} {e['kind']:6} support {str(e['scene_support']):5} "
                  f"iteration {e['checkpoint']['iteration']:>6} packed run {e['rollout']['run']} file {e['export']['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
