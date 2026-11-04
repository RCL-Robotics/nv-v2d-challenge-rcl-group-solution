#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""print_commands_r9.py: print the command lines that rebuild each checkpoint of the counted Tier-2 file
(train every stage, export, evaluate with the official kit evaluator), from the set record of the final Tier-2 selection
robotic_grounding/flash_chord/scripts/track2/t2r9_20261104/set/set_items_r9.json. Prints only; standard library only.

  python3 -I -B scripts/final/lib/print_commands_r9.py [--me PATH] [--zx PATH] [--zx5090 PATH] [--zxws PATH]
                                                        [--kit KIT] [--stock STOCK] [--episodes 4,11]

It replaces TR/t2ready_20261031/tools/print_commands_t2.py (an earlier selection) for the later set records, whose items name the evaluated
checkpoint `evaluated_file` (a relocated copy) instead of `file`, and whose ep10 stage has no Hydra record (its
overrides are rebuilt from the checkpoint's embedded configuration and the first train_t2 line, as
scripts/final/lib/plan.py does). Paths keep the placeholders <ME>, <ZX>, <ZX5090>, <ZXWS> unless replaced.
scripts/final/train_tier2.sh and eval_tier2.sh run the same steps on this machine (--dry-run shows them).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SET = Path(__file__).resolve().parents[3] / "robotic_grounding/flash_chord/scripts/track2/t2r9_20261104/set/set_items_r9.json"


def overrides(tr: dict, st: dict) -> tuple[list[str], str]:
    if "overrides" in st:
        return list(st["overrides"]), f"{st.get('record_kind', '?')}; record {st.get('record', '?')}"
    ec = st["embedded_config"]
    exp = re.search(r"exp=(\S+)", tr["train_log_first_line"]).group(1)
    ovs = [f"experiment={exp}", f"task.parquet='{ec['task.parquet']}'", f"scene.support={str(ec['scene.support']).lower()}",
           f"reset.seed={ec['reset.seed']}", f"output_dir={ec['output_dir']}", "logging.mode=disabled",
           "+logging.local_jsonl=true", f"scene.world_count={ec['scene.world_count']}",
           f"training.max_iterations={ec['training.max_iterations']}",
           f"training.early_stop_help_zero_tail={ec['training.early_stop_help_zero_tail']}"]
    return ovs, ("no run record on the team server: overrides rebuilt from the checkpoint's embedded configuration "
                 f"({st.get('embedded_config_source', '?')[:80]}...) and the first train_t2 line")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=SET)
    p.add_argument("--me")
    p.add_argument("--zx")
    p.add_argument("--zx5090")
    p.add_argument("--zxws")
    p.add_argument("--kit", default="<KIT>", help="v2d_submission_kit root (the Oct-11 kit)")
    p.add_argument("--stock", default="<STOCK>/robotic_grounding/flash_chord",
                   help="stock runtime: nvidia-isaac/video_to_data at 8c778f9, its robotic_grounding/flash_chord")
    p.add_argument("--episodes", default="all")
    a = p.parse_args()
    d = json.loads(a.items.read_text())
    sub = {k: v for k, v in (("<ME>", a.me), ("<ZX5090>", a.zx5090), ("<ZXWS>", a.zxws), ("<ZX>", a.zx)) if v}

    def r(s):
        if isinstance(s, str):
            for k, v in sub.items():
                s = s.replace(k, v)
        return s

    want = None if a.episodes == "all" else {int(x) for x in a.episodes.split(",")}
    print(f"# {d.get('what', 'Tier-2 set')}; record {a.items.name}")
    for e in d["episodes"]:
        if want is not None and int(e["episode"]) not in want:
            continue
        t, ck, ex = e["train"], e["checkpoint"], e["export"]
        print(f"# ---- ep{int(e['episode']):02d} {e['job']} ({e['owner']}, input {e['input']}, {t.get('machine')})")
        print(f"# training commit: {t['code_commit']} ({t.get('code_in_this_history', '')}); source: "
              f"{t.get('code_commit_source', '')[:160]}")
        if t.get("train_log_first_line"):
            print(f"# first train_t2 line: {t['train_log_first_line']}")
        for st in t["stages"]:
            ovs, note = overrides(t, st)
            ovs = [r(o) for o in ovs]
            pq = next((o.split("=", 1)[1].strip("'\"") for o in ovs if o.startswith("task.parquet=")), "")
            root = pq.split("/whole_body/")[0] if "/whole_body/" in pq else r(e["data_root"])
            print(f"# stage {st['stage']} (run {r(st['run'])}; {note})")
            print(f"cd robotic_grounding/flash_chord && HUMAN_MOTION_DATA_DIR={shlex.quote(root)} PYTHONPATH=$PWD/src "
                  f"python scripts/train_rl.py {' '.join(shlex.quote(o) for o in ovs)}")
        src = ck.get("file") or ck.get("source_checkpoint") or ck.get("evaluated_file")
        line = f"# checkpoint model iteration {ck['iteration']}: {r(src)}"
        if ck.get("evaluated_file"):
            line += f"\n#   evaluated (relocated) copy {r(ck['evaluated_file'])} sha256 {ck['sha256']}"
            if ck.get("pool_sha256"):
                line += f" (as pooled: {ck['pool_sha256']})"
        else:
            line += f" sha256 {ck['sha256']}"
        print(line)
        out = f"<EXPORTS>/{e['job']}/{ck['name']}"
        print(f"CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src python -B -m {ex.get('exporter', 'flash_chord.training.ppo.coordinate.export')} "
              f"--input {shlex.quote(r(ck.get('evaluated_file') or ck['file']))} --output {out} --stock-runtime {a.stock} "
              f"--report <EXPORTS>/{e['job']}/export_report.json --tolerance {ex['tolerance']}"
              f"   # in robotic_grounding/flash_chord of {ex['code_commit'][:7]} (tree {ex['tree']})")
        print(f"# evaluated file {r(ex['file'])} sha256 {ex['sha256']}; the evaluator reads the input parquet stored in "
              f"the checkpoint: {r(e['parquet'])} (sha256 {e.get('parquet_sha256', '?')})")
        print(f"echo '{{\"{int(e['episode'])}\": \"{out}\"}}' > <SHARDS>/episode_{int(e['episode']):06d}.json && "
              f"python {a.kit}/examples/run_policy_evaluation.py --eval-script {a.kit}/flash_chord/scripts/evaluate_policy.py "
              f"--runtime-root {a.stock} --world-count 1 --checkpoints <SHARDS>/episode_{int(e['episode']):06d}.json "
              f"--results <RESULTS>   # own input: HUMAN_MOTION_DATA_DIR={r(e['data_root'])}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
