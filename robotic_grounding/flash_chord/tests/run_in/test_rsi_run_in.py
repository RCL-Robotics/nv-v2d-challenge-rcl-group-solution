# SPDX-License-Identifier: Apache-2.0
"""RSI rolling start (dev ``run_in_steps``): config contract, the reference clock inside the reset hold, and the E2
export dropping the dev field with the rest of the RSI reset. CPU-device kernel tests."""

from __future__ import annotations

import numpy as np
import pytest

from flash_chord.configuration import compose_config, resolved_dict

R2 = "g1_recon_body_coord_rsi_r2"
RUN_IN = "g1_recon_body_coord_rsi_r2_runin"
NAMES = ("logging.experiment", "logging.run_name", "training.experiment_name", "training.run_name")
CHANGED = ("reset.run_in_steps", "env.reset.run_in_steps", "reset.min_horizon", "env.reset.min_horizon")


def _compose(experiment: str) -> dict:
    return resolved_dict(compose_config("train", [f"experiment={experiment}", "task.parquet=/tmp/reference.parquet"]))


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        if not value:
            return {prefix.rstrip("."): {}}
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}{index}."))
        return flat
    return {prefix.rstrip("."): value}


def test_run_in_arm_differs_from_c3r2_only_in_the_run_in_the_horizon_and_names():
    r2, arm = _flatten(_compose(R2)), _flatten(_compose(RUN_IN))
    changed = {key for key in r2.keys() | arm.keys() if r2.get(key, KeyError) != arm.get(key, KeyError)}
    unexpected = sorted(changed - {*CHANGED, *NAMES})
    assert not unexpected, f"{RUN_IN} drifted from {R2}: {unexpected}"
    assert arm["reset.run_in_steps"] == arm["env.reset.run_in_steps"] == 20
    assert arm["reset.min_horizon"] == arm["env.reset.min_horizon"] == 30
    assert r2["reset.min_horizon"] == 10 and "reset.run_in_steps" not in r2


def test_config_bounds():
    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    assert RSIReconBodyResetConfig().run_in_steps == 0
    assert RSIReconBodyResetConfig(run_in_steps=50).run_in_steps == 50
    with pytest.raises(ValueError):
        RSIReconBodyResetConfig(run_in_steps=51)
    with pytest.raises(ValueError):
        RSIReconBodyResetConfig(run_in_steps=-1)
    with pytest.raises(TypeError):
        RSIReconBodyResetConfig(run_in_steps=2.5)


def test_export_replaces_the_rsi_reset_with_the_stock_fields():
    from flash_chord.training.ppo.coordinate import export

    stock, _ = export.stock_config_with_notes(_compose(RUN_IN))
    r2_stock, _ = export.stock_config_with_notes(_compose(R2))
    assert "run_in_steps" not in stock["env"]["reset"] and "min_horizon" not in stock["env"]["reset"]
    assert stock["env"]["reset"] == r2_stock["env"]["reset"]


def _advance(clock_start: int, steps: int, freeze: int = 50) -> tuple[list[int], list[float]]:
    """Advance one mid-clip (age 0) and one immediate (age freeze + 1) world; return their clock per step."""
    import warp as wp

    from flash_chord.lifecycle.rsi import kernels

    with wp.ScopedDevice("cpu"):
        done = wp.zeros(2, dtype=wp.int32)
        help_ = wp.array([0.0, 0.0], dtype=wp.float32)
        raw = wp.zeros(2, dtype=wp.float32)
        age = wp.array([0, freeze + 1], dtype=wp.int32)
        voc = wp.zeros(2, dtype=wp.float32)
        offset = wp.zeros(2, dtype=wp.float32)
        timestep = wp.array([100, 0], dtype=wp.int32)
        episode_step = wp.zeros(2, dtype=wp.int32)
        clocks, vocs = [], []
        for _ in range(steps):
            wp.launch(
                kernels.advance_rsi_reset,
                dim=2,
                inputs=[done, done, freeze, clock_start, freeze - 10, 10, 1.0, 1, help_, raw],
                outputs=[age, voc, offset, timestep, episode_step],
            )
            clocks.append(timestep.numpy().tolist())
            vocs.append(float(voc.numpy()[0]))
    return clocks, vocs


def test_stock_release_keeps_the_clock_frozen_for_the_whole_hold():
    clocks, _ = _advance(clock_start=50, steps=51)
    assert [clock[0] for clock in clocks[:50]] == [100] * 50
    assert clocks[50][0] == 101
    assert clocks[50][1] == 51  # the immediate start runs from its first step


def test_run_in_starts_the_clock_twenty_steps_before_the_hold_ends_and_keeps_the_voc_schedule():
    clocks, vocs = _advance(clock_start=30, steps=51)
    mid = [clock[0] for clock in clocks]
    assert mid[:30] == [100] * 30
    assert mid[30:] == list(range(101, 122))
    assert clocks[50][1] == 51
    _, stock_vocs = _advance(clock_start=50, steps=51)
    np.testing.assert_array_equal(vocs, stock_vocs)  # VOC hold and decay are untouched
