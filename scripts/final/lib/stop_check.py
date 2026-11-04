#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""stop_check.py: decide the training.max_iterations a stage of train_tier{1,2,3}.sh runs with, and prove that
stopping there does not change the training trajectory up to that iteration.

  cd <worktree>/robotic_grounding/flash_chord && <env of the training command> \\
    <flashchord python> -B stop_check.py --target N [--smoke] [--out DIR] -- <the stage's recorded overrides>

It composes the stage's Hydra configuration twice with the stage's own trainer (scripts/train_rl.py --cfg job
--resolve; nothing is trained, no GPU is used): (A) with the recorded overrides, (B) the same with
training.max_iterations=N. In the five training trees (b39b69b, bc3b1bd, 7fbc227, f2923d6, 82837c8) the trainer's code
reads training.max_iterations only as the end of the iteration loop, to log and save at the last iteration, in
final.json and for the progress bar (training/ppo/runner.py, scripts/train_rl.py); the learning rate, entropy,
curriculum, RSI and help schedules use absolute iterations from their own keys. A configuration file can still
interpolate it (reset.help_final_iteration: ${training.max_iterations} in the *_helpfix experiments of bc3b1bd and
82837c8), so (A) and (B) are compared key by key:
  - only training.max_iterations differs -> prints N (the stage stops at N; the trajectory up to N is the recorded
    configuration's);
  - any other key differs -> prints the recorded value (the stage trains to its recorded max_iterations; the
    checkpoint at N is saved on the way, every save_interval) and lists the keys.
--smoke: smoke tests (train_tier*.sh --iterations K) always get N; the comparison is only reported.
stdout: the number only; the explanation goes to stderr and, with --out, to DIR/{recorded,stop}.yaml and DIR/result.txt.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
KEY = "training.max_iterations"


def compose(ovs: list[str]) -> tuple[dict, str]:
    p = subprocess.run([sys.executable, "-B", "scripts/train_rl.py", "--cfg", "job", "--resolve", *ovs],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"[stop_check] composing the configuration failed (rc {p.returncode}):\n{p.stderr[-3000:]}")
    import yaml  # the flashchord env has PyYAML (Hydra / OmegaConf depend on it)

    return yaml.safe_load(p.stdout), p.stdout


def flat(d, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", type=Path)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    a = p.parse_args()
    ovs = [o for o in a.overrides if o != "--"]
    if not Path("scripts/train_rl.py").is_file():
        raise SystemExit("[stop_check] run it in the stage's robotic_grounding/flash_chord directory")
    stop_ovs = [o for o in ovs if o.split("=", 1)[0].lstrip("+") != KEY] + [f"{KEY}={a.target}"]
    rec, rec_text = compose(ovs)
    new, new_text = compose(stop_ovs)
    fr, fn = flat(rec), flat(new)
    recorded = fr.get(KEY)
    differ = sorted(k for k in set(fr) | set(fn) if fr.get(k, "<absent>") != fn.get(k, "<absent>"))
    others = [k for k in differ if k != KEY]
    if others:
        msg = (f"training.max_iterations {recorded} -> {a.target} also changes {len(others)} other key(s): "
               + ", ".join(f"{k} ({fr.get(k)} -> {fn.get(k)})" for k in others[:8]))
    else:
        msg = (f"training.max_iterations {recorded} -> {a.target}: the resolved configuration is otherwise identical "
               f"({len(fr)} keys), so the stop does not change the trajectory up to {a.target}")
    if a.smoke:
        chosen = a.target
        msg = "smoke run (not a reproduction): " + msg
    elif others:
        if recorded is None:
            raise SystemExit(f"[stop_check] {msg}; and no recorded training.max_iterations to fall back to")
        chosen = int(recorded)
        msg += f"; training to the recorded {recorded} instead (model_{a.target} is saved on the way)"
    else:
        chosen = a.target
    print(f"[stop_check] {msg}", file=sys.stderr)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "recorded.yaml").write_text(rec_text)
        (a.out / "stop.yaml").write_text(new_text)
        (a.out / "result.txt").write_text(f"chosen max_iterations: {chosen}\n{msg}\n")
    print(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
