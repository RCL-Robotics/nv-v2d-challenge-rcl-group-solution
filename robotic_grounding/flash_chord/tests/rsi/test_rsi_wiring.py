# SPDX-License-Identifier: Apache-2.0
"""Integration wiring (docs/coord_impl_spec.md section 5): WarpRLEnv.rsi detection and the local metric sinks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from flash_chord.lifecycle.rsi import RSIHooks
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.ppo.runner import CheckpointLogger, JsonlLogger, TeeLogger


class _StockLikeReset:
    def sample(self, reset_mask, curriculum_voc_scale):
        pass


class _RSILikeReset(_StockLikeReset):
    def bind_env(self, env):
        pass

    def on_transition(self, state):
        pass

    def restore(self, state, reset_mask):
        pass

    def host_update(self, iteration):
        return {}

    def save_state(self, path):
        pass

    def load_state(self, path):
        pass


def _fake_env(reset_policy):
    return SimpleNamespace(
        world_count=2,
        observation_dim=3,
        action=SimpleNamespace(action_dim=1),
        reset_policy=reset_policy,
    )


def test_warp_env_exposes_rsi_hooks_only_for_rsi_reset_policies():
    rsi_policy = _RSILikeReset()
    assert isinstance(rsi_policy, RSIHooks)
    assert not isinstance(_StockLikeReset(), RSIHooks)
    assert WarpRLEnv(_fake_env(rsi_policy)).rsi is rsi_policy
    assert WarpRLEnv(_fake_env(_StockLikeReset())).rsi is None
    assert WarpRLEnv(_fake_env(None)).rsi is None


class _RecordingCheckpointLogger:
    def __init__(self):
        self.records, self.checkpoints, self.closed = [], [], False

    def log(self, metrics, step):
        self.records.append((dict(metrics), step))

    def log_checkpoint(self, path, *, iteration, metadata):
        self.checkpoints.append((Path(path), iteration, dict(metadata)))

    def close(self):
        self.closed = True


def test_jsonl_and_tee_loggers(tmp_path):
    path = tmp_path / "run" / "metrics.jsonl"
    jsonl = JsonlLogger(path)
    recorder = _RecordingCheckpointLogger()
    tee = TeeLogger(recorder, None, jsonl)
    assert isinstance(tee, CheckpointLogger)
    tee.log({"a": 1, "rsi/top_mean": 0.5}, step=98304)
    tee.log_checkpoint(tmp_path / "model_1.safetensors", iteration=1, metadata={"iteration": 1})
    tee.close()
    assert recorder.closed and recorder.records == [({"a": 1, "rsi/top_mean": 0.5}, 98304)]
    assert recorder.checkpoints[0][1] == 1
    # Append mode: a resumed run continues the same file.
    again = JsonlLogger(path)
    again.log({"a": 2.0}, step=196608)
    again.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"step": 98304, "a": 1.0, "rsi/top_mean": 0.5}, {"step": 196608, "a": 2.0}]
