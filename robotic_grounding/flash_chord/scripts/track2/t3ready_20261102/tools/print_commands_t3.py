#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""print_commands_t3.py: the commands that produced each Tier-3 FINAL_r episode, from set/set_items.json.

  python3 print_commands_t3.py [--episode N] [--recorded] [--root <ME>=/path ...] [--repo /path/to/this/checkout]

Per episode, in order:
  1. training stage f0 (every item): scripts/train_rl.py with the run's saved Hydra overrides, from this checkout's
     robotic_grounding/flash_chord (code files identical to fc_r2 @ 61fa5d3, which ran them);
  2. retry items only: the fork (copy model_44922 + rsi_44922 of the f0 run into the retry run dir, sha256 checked),
     then training stage retry_C90 (resume to 90000);
  3. export + official own-input evaluation: scripts/track2/eval_all.py (export tolerance 1e-4, world count 1, the stock
     runtime video_to_data @ 8c778f9 and the unchanged kit evaluator) -- f0 items: the f0 post step exported the eval
     file once and t3_f0_final evaluated it 5 times (--export no); retry items: each of the 5 runs exported and
     evaluated (--export yes); the packed rollout is the run closest to the 5-run mean own-input ADD-AUC.
--recorded also prints the launch / eval lines exactly as the drivers logged them. Paths use the training server's
roots (set_items.json "placeholders") unless --root <ME>=/new/root maps them elsewhere.
Training needs one GPU and Isaac Lab (FlashCHORD env); the run is GPU-non-deterministic, so a rerun reproduces the
policy class, not the bytes; the evaluated files themselves are kept on the training server (paths in set_items.json).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[4]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episode", type=int, action="append")
    p.add_argument("--recorded", action="store_true")
    p.add_argument("--root", action="append", default=[])
    p.add_argument("--repo", default=str(REPO))
    a = p.parse_args()
    data = json.loads((HERE / "set/set_items.json").read_text())
    orig = dict(data.get("placeholders", {}))  # placeholder -> the training server's path
    roots = dict(orig)
    for x in a.root:
        k, v = x.split("=", 1)
        roots[k] = v

    def real(s: str) -> str:
        for k, v in orig.items():  # verbatim records (fork.json) hold the server paths
            s = s.replace(v, k)
        for k, v in roots.items():
            s = s.replace(k, v)
        return s

    fc = f"{a.repo}/robotic_grounding/flash_chord"
    for ent in data["episodes"]:
        e = ent["episode"]
        if a.episode and e not in a.episode:
            continue
        print(f"# ===== ep{e:02d} {ent['object']} ({ent['kind']} item; support {ent['scene_support']}, "
              f"palm mitigation {ent['mitigated']})")
        for st in ent["train"]["stages"]:
            run = real(st["run_dir"])
            if st["stage"] == "retry_C90":
                fork = json.loads((HERE / st["fork_json"]).read_text())
                print(f"mkdir -p {run}")
                for k in ("model", "rsi"):
                    f = fork["files"][k]
                    print(f"cp {real(f['src'])} {real(f['dst'])} && "
                          f"echo '{f['sha256']}  {real(f['dst'])}' | sha256sum -c")
            ovs = [real(o) for o in st["overrides"]]
            print(f"( cd {fc} && env T3F0=1 HUMAN_MOTION_DATA_DIR={real('<ME>/data/human_motion_data_tier3_f0')} "
                  f"WANDB_MODE=offline python scripts/train_rl.py {' '.join(shlex.quote(o) for o in ovs)} "
                  f"hydra.run.dir={run}/hydra )   # stage {st['stage']} -> {st['checkpoint']['name']} "
                  f"sha256 {st['checkpoint']['sha256'][:16]}")
        ex = ent["export"]
        print(f"# export + evaluation -> evaluated file sha256 {ex['sha256'][:16]} (export source "
              f"{ex['export_source_sha256'][:16]}), packed rollout = run {ent['rollout']['run']}")
        if a.recorded:
            for k, v in ent["recorded_commands"].items():
                for line in v:
                    print(f"#   recorded {k}: {real(line)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
