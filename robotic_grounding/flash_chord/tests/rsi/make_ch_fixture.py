# SPDX-License-Identifier: Apache-2.0
"""Generate the Coordinates_hand (CH) RSI parity fixture used by ``tests/rsi/test_rsi_bins.py``.

It drives the real CH ``Curriculum`` (read-only; CH is put on ``sys.path``) exactly like CH
``dexmachina/tests/test_rsi.py:50-66``: one ``rsi_on_reset`` for the finishing lanes and one
``set_curriculum(t)`` per iteration, kp_init 80, zero_below 0.05, bin_frames 15, T = 300, and dumps the
per-iteration outcome counts (n_b, f_b) with the resulting tops, failure EMAs and start probabilities.

Run once, CPU only, with the DexMachina interpreter (torch + genesis):

    CUDA_VISIBLE_DEVICES="" /root/autodl-tmp/haoran.zhang/conda_envs/dexmachina/bin/python \
        tests/rsi/make_ch_fixture.py --ch /root/autodl-tmp/haoran.zhang/code/Coordinates_hand \
        --out tests/rsi/fixtures/ch_rsi_parity.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

T_MAX = 300
NUM_ENVS = 2000
FINISH_SHARE = 1 / 12
PROBS_EVERY = 5

# CH keyword -> name in RSIBinParams (flash_chord.lifecycle.rsi.bins)
PARAM_MAP = {
    "rsi_bin_frames": "bin_frames",
    "rsi_uniform_mix": "uniform_mix",
    "rsi_credit_gamma": "credit_gamma",
    "rsi_fail_alpha": "fail_alpha",
    "rsi_success_up": "success_up",
    "rsi_success_down": "success_down",
    "rsi_step": "step",
    "rsi_band": "band",
    "rsi_min_count": "min_count",
    "rsi_min_gap": "min_gap",
    "wait_epochs": "wait_iterations",
    "kp_init": "top0",
    "rsi_zero_below": "zero_below",
    "rsi_deadline": "deadline",
    "rsi_zero_ramp": "zero_ramp",
    "rsi_min_horizon": "min_horizon",
    "rsi_start_update": "start_update",
}


def fail_hard8(bins, kp, t):
    return np.where(bins == 8, 0.9, 0.0)


def fail_none(bins, kp, t):
    return np.zeros(len(bins))


def fail_p95(bins, kp, t):
    return np.full(len(bins), 0.95)


def fail_late(bins, kp, t):
    """Every bin succeeds at first and races below the floor; then bins 0, 4, 8, ... start failing at t = 300 (up moves
    from a positive top) and bins 1, 5, 9, ... at t = 900 after reaching zero (up moves from zero to zero_below)."""
    fail = np.full(len(bins), 0.02)
    fail = np.where((bins % 4 == 0) & (t >= 300), 0.9, fail)
    fail = np.where((bins % 4 == 1) & (t >= 900), 0.9, fail)
    return fail


SCENARIOS = (
    {"name": "always_succeed", "over": {"rsi_min_gap": 10}, "updates": 1000, "seed": 0, "fail": fail_none},
    {"name": "hard_bin", "over": {}, "updates": 1200, "seed": 1, "fail": fail_hard8},
    {
        "name": "deadline_forcing",
        "over": {"rsi_deadline": 400, "rsi_zero_ramp": 50, "rsi_start_update": 30},
        "updates": 650,
        "seed": 2,
        "fail": fail_p95,
    },
    {
        "name": "mixed_up_down",
        "over": {"rsi_deadline": 1500, "rsi_zero_ramp": 100, "wait_epochs": 20, "rsi_min_gap": 10},
        "updates": 1800,
        "seed": 5,
        "fail": fail_late,
    },
)


class FakeObject:
    """CH tests/test_rsi.py:15-31 (records the first per-DOF kp of every lane)."""

    device = "cpu"

    def __init__(self, num_envs):
        self.kp = np.full(num_envs, np.nan)

    def set_joint_gains(self, kp=None, kv=None, force_range=None, env_idxs=None, zero_velocity=True):
        self.kp[env_idxs.cpu().numpy()] = kp[:, 0].cpu().numpy()


def run_scenario(curriculum_module, torch, scenario):
    base = dict(
        kp_init=80.0, kv_init=5.0, force_range_init=50.0, schedule="rsi", fixed_mode="uniform",
        uniform_mode="slow", gain_mode="all", wait_epochs=100, deque_len=40,
        rew_thresholds=dict(task=0.6, con=0.0, imi=0.0, bc=0.0), rsi_step=0.9, rsi_band=2.0,
        rsi_bin_frames=15, rsi_zero_below=0.05,
    )
    base.update(scenario["over"])
    cfg = curriculum_module.get_curriculum_cfg(base)
    obj = FakeObject(NUM_ENVS)
    cur = curriculum_module.Curriculum(cfg, obj, ["task"], NUM_ENVS, 0, T_MAX)
    cur.post_scene_build_setup()
    params = {ours: (cfg[ch] if ch in cfg else curriculum_module.RSI_DEFAULTS[ch]) for ch, ours in PARAM_MAP.items()}
    params = {k: (int(v) if k in ("bin_frames", "min_count", "min_gap", "wait_iterations", "deadline", "zero_ramp",
                                    "min_horizon", "start_update") else float(v)) for k, v in params.items()}

    rng = np.random.default_rng(scenario["seed"])
    none = torch.zeros(NUM_ENVS, dtype=torch.bool)
    cur.rsi_on_reset(torch.arange(NUM_ENVS), none, none)
    record = {
        "name": scenario["name"],
        "ch_overrides": scenario["over"],
        "params": params,
        "num_envs": NUM_ENVS,
        "num_frames": T_MAX,
        "num_bins": int(cur.rsi_bins),
        "bin_lo": cur.rsi_bin_lo.tolist(),
        "bin_hi": cur.rsi_bin_hi.tolist(),
        "initial_top": cur.rsi_top.tolist(),
        "initial_probs": cur._rsi_start_probs().tolist(),
        "n": [],
        "f": [],
        "moved": [],
        "zero": [],
        "floor": [],
        "top_changes": [],
        "probs_every": PROBS_EVERY,
        "probs": [],
        "rate": [],
    }
    previous_top = cur.rsi_top.clone()
    for t in range(1, scenario["updates"] + 1):
        done = np.flatnonzero(rng.random(NUM_ENVS) < FINISH_SHARE)
        n_b = np.zeros(cur.rsi_bins, dtype=np.int64)
        f_b = np.zeros(cur.rsi_bins, dtype=np.int64)
        if len(done):
            bins = cur.rsi_env_bin[torch.as_tensor(done)].numpy()
            failed = rng.random(len(done)) < scenario["fail"](bins, obj.kp[done], t)
            n_b = np.bincount(bins, minlength=cur.rsi_bins)
            f_b = np.bincount(bins, weights=failed.astype(np.float64), minlength=cur.rsi_bins).astype(np.int64)
            n_before, f_before = cur.rsi_n.clone(), cur.rsi_f.clone()
            ended = torch.ones(len(done), dtype=torch.bool)
            cur.rsi_on_reset(torch.as_tensor(done), ended, torch.as_tensor(failed))
            # Cross-check the counts against CH's own accumulation (all finishing lanes are active here).
            assert np.array_equal((cur.rsi_n - n_before).numpy(), n_b.astype(np.float64))
            assert np.array_equal((cur.rsi_f - f_before).numpy(), f_b.astype(np.float64))
        zero, moved, _ = cur.set_curriculum(t)
        record["n"].append(n_b.tolist())
        record["f"].append(f_b.tolist())
        record["moved"].append(bool(moved))
        record["zero"].append(bool(zero))
        record["floor"].append(float(cur._rsi_floor(t)))
        if not bool(torch.equal(cur.rsi_top, previous_top)):
            record["top_changes"].append([t, cur.rsi_top.tolist()])
            previous_top = cur.rsi_top.clone()
        if t % PROBS_EVERY == 0 or t == scenario["updates"]:
            record["probs"].append([t, cur._rsi_start_probs().tolist()])
            record["rate"].append([t, cur.rsi_rate.tolist()])
    record["final"] = {
        "top": cur.rsi_top.tolist(),
        "rate": cur.rsi_rate.tolist(),
        "win_n": cur.rsi_n.tolist(),
        "win_f": cur.rsi_f.tolist(),
        "last_move": cur.rsi_last_move.tolist(),
        "counts": dict(cur.rsi_counts),
        "help_progress": float(cur.help_progress()),
        "num_epoch_since_zero": int(cur.num_epoch_since_zero),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ch", type=Path, required=True, help="Coordinates_hand checkout (read-only)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.ch.resolve()))
    import torch  # noqa: PLC0415 - the DexMachina interpreter provides torch

    from dexmachina.envs import curriculum as curriculum_module  # noqa: PLC0415

    try:
        commit = subprocess.run(
            ["git", "-C", str(args.ch), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        commit = ""
    payload = {
        "source": "Coordinates_hand dexmachina/envs/curriculum.py (Curriculum, schedule=rsi)",
        "ch_commit": commit,
        "ch_curriculum_file": str((args.ch / "dexmachina/envs/curriculum.py").resolve()),
        "torch": torch.__version__,
        "scenarios": [run_scenario(curriculum_module, torch, scenario) for scenario in SCENARIOS],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
    sizes = ", ".join(
        f"{s['name']}: {len(s['n'])} it, {len(s['top_changes'])} top changes" for s in payload["scenarios"]
    )
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes; {sizes})")


if __name__ == "__main__":
    main()
