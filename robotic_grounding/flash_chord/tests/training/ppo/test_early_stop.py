# SPDX-License-Identifier: Apache-2.0
"""Early stop on zero RSI help (TrainingConfig.early_stop_help_zero_tail) and the final.json marker.

Coordinates_hand stops a run once the curriculum has kept zero gains for 1500 updates. Here: stop after the first
iteration N with every RSI start bin at help 0 continuously for N - help_zero_since >= tail iterations; save
model_N/rsi_N, write <output_dir>/final.json, exit normally. help_zero_since is recomputed from the RSI sidecar
(bins.top, bins.last_move), so a resumed run stops at the same iteration as an uninterrupted one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flash_chord.lifecycle.rsi.bins import RSIBinParams, RSIBins

# Shared runner fixtures of the non-finite guard tests (same directory; pytest's default prepend import mode).
from test_nonfinite_guard import _GuardEnv, _Logger, _dependencies, _metadata, _stock_setup  # noqa: E402

SAMPLES = 4 * 4  # _stock_setup(world_count=4, rollout_steps=4)

# 50 frames, 10-frame bins, min_horizon 10 -> 4 start bins. No success moves before iteration 1000, so only the
# deadline acts: floor > 0 until 4 + 2 = 6, every bin forced to 0 by host_update(6).
DEADLINE_PARAMS = RSIBinParams(bin_frames=10, min_horizon=10, deadline=4, zero_ramp=2, wait_iterations=1000)
ZERO_AT = 6


# -------------------------------------------------------------------------------------------------- config


def test_training_config_validates_the_tail():
    from flash_chord.training.ppo.config import TrainingConfig

    assert TrainingConfig().early_stop_help_zero_tail == 0
    assert TrainingConfig(early_stop_help_zero_tail=5860).early_stop_help_zero_tail == 5860
    for bad in (-1, True, 1.5, "5"):
        with pytest.raises(ValueError, match="early_stop_help_zero_tail"):
            TrainingConfig(early_stop_help_zero_tail=bad)


# ------------------------------------------------------------------------------------- RSIBins.zero_since


def _run_bins(bins, iterations, outcomes=None):
    since = []
    for t in iterations:
        n, f = outcomes(t) if outcomes else (np.zeros(bins.num_bins), np.zeros(bins.num_bins))
        bins.observe(n, f)
        bins.decide(t)
        since.append(bins.zero_since())
    return since


def test_zero_since_is_the_deadline_iteration_when_the_floor_forces_zero():
    bins = RSIBins(50, DEADLINE_PARAMS)
    assert bins.num_bins == 4 and bins.zero_since() is None
    since = _run_bins(bins, range(12))
    assert since == [None] * ZERO_AT + [ZERO_AT] * (12 - ZERO_AT)
    assert (bins.top == 0).all() and int(bins.last_move.max()) == ZERO_AT


def test_zero_since_restarts_after_an_up_move_breaks_the_streak():
    """'Continuously': a bin that regains help ends the streak; the next all-zero decision starts a new one."""
    params = RSIBinParams(bin_frames=10, min_horizon=10, top0=1.0, step=0.5, zero_below=0.6, min_count=1,
                          min_gap=0, wait_iterations=0, deadline=1000, zero_ramp=10)
    bins = RSIBins(30, params)
    assert bins.num_bins == 2
    script = {
        0: ([1, 1], [0, 0]),  # both succeed: 1.0 * 0.5 < 0.6 -> 0
        1: ([0, 0], [0, 0]),  # nothing scored
        2: ([1, 0], [1, 0]),  # bin 0 fails at 0 -> up-move to zero_below 0.6 (<= floor ~0.999)
        3: ([0, 0], [0, 0]),
        4: ([1, 0], [0, 0]),  # bin 0 succeeds: 0.6 * 0.5 < 0.6 -> 0 again
        5: ([0, 0], [0, 0]),
    }
    since = _run_bins(bins, range(6), lambda t: tuple(np.asarray(v, float) for v in script[t]))
    assert since == [0, 0, None, None, 4, 4]
    assert bins.counts["up"] == 1 and bins.counts["down"] == 3


@pytest.mark.parametrize("seed", range(12))
def test_zero_since_matches_an_explicit_streak_counter(seed):
    """Random outcomes, success moves, up-moves and the deadline: recomputing from (top, last_move) equals a counter
    kept alongside the decisions, at every iteration."""
    rng = np.random.default_rng(seed)
    params = RSIBinParams(bin_frames=10, min_horizon=10, top0=1.0, step=0.5, zero_below=0.3, success_up=0.8,
                          success_down=0.4, min_count=int(rng.integers(1, 4)), min_gap=int(rng.integers(0, 3)),
                          wait_iterations=int(rng.integers(0, 5)), deadline=int(rng.integers(40, 120)),
                          zero_ramp=int(rng.integers(1, 20)))
    bins = RSIBins(int(rng.integers(30, 60)), params)
    tracked, breaks, zero_steps = None, 0, 0
    p_fail = rng.uniform(0.1, 0.6)
    for t in range(200):
        n = rng.integers(0, 4, bins.num_bins).astype(float)
        f = np.minimum(n, rng.binomial(n.astype(int), p_fail)).astype(float)
        bins.observe(n, f)
        bins.decide(t)
        if (bins.top == 0).all():
            tracked = t if tracked is None else tracked
            zero_steps += 1
        else:
            breaks += tracked is not None
            tracked = None
        assert bins.zero_since() == tracked, t
    assert zero_steps > 0  # the deadline guarantees the all-zero state within 200 iterations


def test_zero_since_survives_the_sidecar_round_trip(tmp_path):
    from safetensors.numpy import load_file, save_file

    bins = RSIBins(50, DEADLINE_PARAMS)
    _run_bins(bins, range(ZERO_AT + 3))
    save_file({f"bins.{k}": np.ascontiguousarray(v) for k, v in bins.state_dict().items()}, str(tmp_path / "s.st"))
    restored = RSIBins(50, DEADLINE_PARAMS)
    restored.load_state_dict({k[5:]: v for k, v in load_file(str(tmp_path / "s.st")).items()})
    assert restored.zero_since() == bins.zero_since() == ZERO_AT


# ------------------------------------------------------------------------------------------------- runner


class _BinsRSI:
    """RSI hooks around a real RSIBins (no outcomes scored): the deadline drives every bin to zero help."""

    def __init__(self, params=DEADLINE_PARAMS, num_frames=50, with_zero_since=True):
        self.bins = RSIBins(num_frames, params)
        self.updates = []
        if not with_zero_since:
            self.help_zero_since = None

    def host_update(self, iteration):
        self.bins.observe(np.zeros(self.bins.num_bins), np.zeros(self.bins.num_bins))
        self.bins.decide(iteration)
        self.updates.append(iteration)
        return {"rsi/top_max": float(self.bins.top.max())}

    def help_zero_since(self):
        return self.bins.zero_since()

    def save_state(self, path):
        from safetensors.numpy import save_file

        save_file({f"bins.{k}": np.ascontiguousarray(v) for k, v in self.bins.state_dict().items()}, str(path))

    def load_state(self, path):
        from safetensors.numpy import load_file

        self.bins.load_state_dict({k[5:]: v for k, v in load_file(str(path)).items() if k.startswith("bins.")})


def _config(jax, jnp, **overrides):
    values = dict(rollout_steps=4, max_iterations=20, save_interval=5, early_stop_help_zero_tail=3)
    values.update(overrides)
    config, _, _ = _stock_setup(jax, jnp, **values)
    return config


def _env(jnp, **rsi):
    env = _GuardEnv(jnp)
    env.rsi = _BinsRSI(**rsi)
    return env


def _final(directory):
    return json.loads((Path(directory) / "final.json").read_text())


def test_early_stop_saves_the_final_checkpoint_and_writes_final_json(tmp_path, capsys):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    env, logger = _env(jnp), _Logger()
    learner = run_training(env, _config(jax, jnp), logger=logger, checkpoint_dir=tmp_path, log_interval=4,
                           checkpoint_metadata=_metadata())
    stop = ZERO_AT + 3  # host_update(6) zeroes every bin; iterations 7, 8, 9 (1-based) train without help
    assert int(learner.iteration) == stop
    assert env.rsi.updates == list(range(stop))
    saved = sorted(p.name for p in tmp_path.glob("*.safetensors"))
    assert saved == ["model_5.safetensors", "model_9.safetensors", "rsi_5.safetensors", "rsi_9.safetensors"]
    final = _final(tmp_path)
    assert {k: final[k] for k in ("iteration", "reason", "checkpoint", "rsi_state", "help_zero_since",
                                  "help_zero_iterations", "early_stop_help_zero_tail", "max_iterations",
                                  "environment_steps", "start_iteration", "resumed_from")} == {
        "iteration": 9, "reason": "help_zero_tail", "checkpoint": "model_9.safetensors",
        "rsi_state": "rsi_9.safetensors", "help_zero_since": ZERO_AT, "help_zero_iterations": 3,
        "early_stop_help_zero_tail": 3, "max_iterations": 20, "environment_steps": 9 * SAMPLES,
        "start_iteration": 0, "resumed_from": None,
    }  # fmt: skip
    assert final["wall_seconds"] >= 0 and final["finished_at"]
    # Records: iterations 4 and 8 (log_interval) plus the stop iteration 9.
    assert [step for step, _ in logger.records] == [4 * SAMPLES, 8 * SAMPLES, 9 * SAMPLES]
    assert [r["early_stop/help_zero_iterations"] for _, r in logger.records] == [0.0, 2.0, 3.0]
    assert [r.get("early_stop/stopped") for _, r in logger.records] == [None, None, 1.0]
    out = capsys.readouterr().out
    assert "[early-stop] iteration 9: every RSI start bin has had zero help for 3 iterations (since iteration 6" in out


def test_early_stop_on_a_save_interval_saves_once(tmp_path):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    run_training(_env(jnp), _config(jax, jnp, save_interval=3), checkpoint_dir=tmp_path,
                 checkpoint_metadata=_metadata())
    assert sorted(p.name for p in tmp_path.glob("model_*.safetensors")) == [
        "model_3.safetensors", "model_6.safetensors", "model_9.safetensors"]
    assert _final(tmp_path)["iteration"] == 9


@pytest.mark.parametrize("first_max", [5, 6, 7, 8])
def test_resume_across_the_boundary_stops_at_the_same_iteration(tmp_path, first_max):
    """Stop at max_iterations before (5), at (6), and after (7, 8) the zero-help streak began, resume to 20: the
    resumed run stops at iteration 9 like an uninterrupted one, with the same help_zero_since."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    reference = tmp_path / "uninterrupted"
    run_training(_env(jnp), _config(jax, jnp), checkpoint_dir=reference, checkpoint_metadata=_metadata())

    directory = tmp_path / "resumed"
    config = _config(jax, jnp, max_iterations=first_max)
    run_training(_env(jnp), config, checkpoint_dir=directory, checkpoint_metadata=_metadata())
    first = _final(directory)
    assert (first["iteration"], first["reason"]) == (first_max, "max_iterations")
    assert first["help_zero_since"] == (ZERO_AT if first_max > ZERO_AT else None)
    if first_max > ZERO_AT:
        assert first["help_zero_iterations"] == first_max - ZERO_AT

    env = _env(jnp)
    checkpoint = directory / f"model_{first_max}.safetensors"
    learner = run_training(env, replace(config, max_iterations=20, resume=True, checkpoint=str(checkpoint)),
                           checkpoint_dir=directory, checkpoint_metadata=_metadata())
    assert env.rsi.updates == list(range(first_max, 9))
    assert int(learner.iteration) == 9
    final, expected = _final(directory), _final(reference)
    for key in ("iteration", "reason", "checkpoint", "rsi_state", "help_zero_since", "help_zero_iterations",
                "environment_steps"):
        assert final[key] == expected[key], key
    assert final["start_iteration"] == first_max and final["resumed_from"] == str(checkpoint)
    # The first run's marker is kept as final.prev.json; training went on past it.
    assert json.loads((directory / "final.prev.json").read_text())["iteration"] == first_max


def test_resuming_a_run_that_already_qualifies_stops_after_one_iteration(tmp_path):
    """A checkpoint saved after the stop condition held (e.g. a crash before final.json) trains one iteration."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    config = _config(jax, jnp, early_stop_help_zero_tail=0, max_iterations=12)
    run_training(_env(jnp), config, checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    env = _env(jnp)
    resume = replace(config, early_stop_help_zero_tail=3, max_iterations=20, resume=True,
                     checkpoint=str(tmp_path / "model_12.safetensors"))
    learner = run_training(env, resume, checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    assert int(learner.iteration) == 13 and env.rsi.updates == [12]
    assert _final(tmp_path)["help_zero_iterations"] == 13 - ZERO_AT


def test_tail_zero_keeps_the_stock_loop_and_records(tmp_path):
    """Off (the default): no early_stop/* keys, max_iterations reached although help is zero, checkpoints only on
    the save interval and at the end; final.json still marks the end (reason max_iterations)."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    env, logger = _env(jnp), _Logger()
    learner = run_training(env, _config(jax, jnp, early_stop_help_zero_tail=0, max_iterations=12), logger=logger,
                           checkpoint_dir=tmp_path, log_interval=4, checkpoint_metadata=_metadata())
    assert int(learner.iteration) == 12
    assert [step for step, _ in logger.records] == [4 * SAMPLES, 8 * SAMPLES, 12 * SAMPLES]
    assert not any(key.startswith("early_stop/") for _, record in logger.records for key in record)
    assert sorted(p.name for p in tmp_path.glob("model_*.safetensors")) == [
        "model_10.safetensors", "model_12.safetensors", "model_5.safetensors"]
    final = _final(tmp_path)
    assert (final["iteration"], final["reason"], final["help_zero_since"]) == (12, "max_iterations", ZERO_AT)


def test_tail_zero_runs_are_bitwise_unchanged_by_the_option(tmp_path):
    """The same run with tail 0 and with a tail too long to fire: identical records and checkpoints."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    results = {}
    for tail in (0, 1000):
        directory, logger = tmp_path / f"tail{tail}", _Logger()
        run_training(_env(jnp), _config(jax, jnp, early_stop_help_zero_tail=tail, max_iterations=8), logger=logger,
                     checkpoint_dir=directory, log_interval=2, checkpoint_metadata=_metadata())
        results[tail] = (directory, logger.records)
    plain = [(step, record) for step, record in results[0][1]]
    with_option = [(step, {k: v for k, v in record.items() if not k.startswith("early_stop/")})
                   for step, record in results[1000][1]]
    assert plain == with_option
    from safetensors import safe_open

    for name in ("model_5.safetensors", "model_8.safetensors"):
        tensors = []
        for tail in (0, 1000):
            with safe_open(str(results[tail][0] / name), "numpy") as handle:
                tensors.append({key: handle.get_tensor(key) for key in handle.keys()})
        assert sorted(tensors[0]) == sorted(tensors[1])
        for key in tensors[0]:
            np.testing.assert_array_equal(tensors[0][key], tensors[1][key], err_msg=f"{name} {key}")


def test_early_stop_needs_the_rsi_reset(tmp_path):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    with pytest.raises(ValueError, match="early_stop_help_zero_tail > 0 needs the RSI reset"):
        run_training(_GuardEnv(jnp), _config(jax, jnp), checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    with pytest.raises(ValueError, match="early_stop_help_zero_tail > 0 needs the RSI reset"):
        run_training(_env(jnp, with_zero_since=False), _config(jax, jnp), checkpoint_dir=tmp_path,
                     checkpoint_metadata=_metadata())


def test_no_final_json_without_a_checkpoint_dir_or_without_iterations(tmp_path):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    learner = run_training(_env(jnp), _config(jax, jnp), checkpoint_metadata=_metadata())
    assert int(learner.iteration) == 9 and not any(tmp_path.iterdir())
    # Resuming a checkpoint already at max_iterations trains nothing and leaves final.json alone.
    config = _config(jax, jnp, early_stop_help_zero_tail=0, max_iterations=2)
    run_training(_env(jnp), config, checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    marker = (tmp_path / "final.json").read_text()
    run_training(_env(jnp), replace(config, resume=True, checkpoint=str(tmp_path / "model_2.safetensors")),
                 checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    assert (tmp_path / "final.json").read_text() == marker and not (tmp_path / "final.prev.json").exists()


def test_real_rsi_reset_exposes_help_zero_since():
    """RSIReconBodyReset delegates to its bins (no env needed for the host-side bins)."""
    pytest.importorskip("warp")
    from flash_chord.lifecycle.rsi_reset import RSIReconBodyReset

    assert callable(getattr(RSIReconBodyReset, "help_zero_since", None))
