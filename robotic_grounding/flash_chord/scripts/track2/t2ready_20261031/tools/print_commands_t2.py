#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""print_commands_t2.py: print the command lines that rebuild each checkpoint of the Tier-2 set R6-CLOSEST
(train every stage, export, evaluate with the official kit evaluator), from set/set_items.json. Prints only.

  python print_commands_t2.py [--me <path for <ME>>] [--zx <path for <ZX>>] [--zx5090 <path for <ZX5090>>]
                              [--kit KIT] [--stock STOCK] [--episodes 4,11]

Paths keep the placeholders <ME> (Haoran's workdir), <ZX> (Zuxing's workdir) and <ZX5090> (Zuxing's remote RTX 5090
workspace) unless replaced. The training command of a stage is `python scripts/train_rl.py <the stage's saved Hydra
overrides>` run in robotic_grounding/flash_chord of the stage's training commit, with HUMAN_MOTION_DATA_DIR = the
episode's input data root (as the original launchers set it: train/ours/*, train/zuxing/train_t2.sh). Resumed stages
(training.resume=true) continue from the checkpoint their overrides name. Export and evaluation are the commands the
rounds ran (records/exports/<job>.export.json; selection/round6_20261031 drivers).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=HERE / "set/set_items.json")
    p.add_argument("--me")
    p.add_argument("--zx")
    p.add_argument("--zx5090")
    p.add_argument("--kit", default="<KIT>", help="v2d_submission_kit root")
    p.add_argument("--stock", default="<STOCK>/robotic_grounding/flash_chord",
                   help="stock runtime: nvidia-isaac/video_to_data at 8c778f9, its robotic_grounding/flash_chord")
    p.add_argument("--episodes", default="all")
    a = p.parse_args()
    d = json.loads(a.items.read_text())
    sub = {k: v for k, v in (("<ME>", a.me), ("<ZX>", a.zx), ("<ZX5090>", a.zx5090)) if v}

    def r(s):
        if isinstance(s, str):
            for k, v in sub.items():
                s = s.replace(k, v)
        return s

    want = None if a.episodes == "all" else {int(x) for x in a.episodes.split(",")}
    for e in d["episodes"]:
        if want is not None and e["episode"] not in want:
            continue
        t, ck, ex = e["train"], e["checkpoint"], e["export"]
        print(f"# ---- ep{e['episode']:02d} {e['job']} ({e['owner']}, input {e['input']}, {t['machine']})")
        print(f"# training commit: {t['code_commit'] or 'NOT RECORDED'} ({t['code_commit_source']})")
        for st in t["stages"]:
            ovs = [r(o) for o in st["overrides"]]
            # the stage's data root: the task.parquet override up to /whole_body/ (as the launchers exported it)
            pq = next((o.split("=", 1)[1].strip("'\"") for o in ovs if o.startswith("task.parquet=")), "")
            root = pq.split("/whole_body/")[0] if "/whole_body/" in pq else r(e["data_root"])
            print(f"# stage {st['stage']} ({st['record_kind']}; record {st['record']})")
            print(f"cd robotic_grounding/flash_chord && HUMAN_MOTION_DATA_DIR={shlex.quote(root)} PYTHONPATH=$PWD/src "
                  f"python scripts/train_rl.py {' '.join(shlex.quote(o) for o in ovs)}")
        run = r(t["stages"][-1]["run"])
        out = f"<EXPORTS>/{e['job']}/{ck['name']}"
        print(f"# checkpoint {run}/{ck['name']} sha256 {ck['sha256']}"
              + (f" (relocated copy of as-trained sha256 {ck['relocated']['as_trained_sha256']}: task.parquet -> "
                 f"{r(ck['relocated']['new_parquet_path'])})" if ck.get("relocated") else ""))
        print(f"CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src python -B -m flash_chord.training.ppo.coordinate.export "
              f"--input {shlex.quote(r(ck['file']))} --output {out} --stock-runtime {a.stock} "
              f"--report <EXPORTS>/{e['job']}/export_report.json --tolerance {ex['tolerance']}")
        print(f"# evaluated file sha256 {ex['sha256']} (exporter tree {ex['tree']}); the evaluator reads the input parquet "
              f"stored in the checkpoint: {r(e['parquet'])}")
        print(f"echo '{{\"{e['episode']}\": \"{out}\"}}' > <SHARDS>/episode_{e['episode']:06d}.json && "
              f"python {a.kit}/examples/run_policy_evaluation.py --eval-script {a.kit}/flash_chord/scripts/evaluate_policy.py "
              f"--runtime-root {a.stock} --world-count 1 --checkpoints <SHARDS>/episode_{e['episode']:06d}.json --results <RESULTS>")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
