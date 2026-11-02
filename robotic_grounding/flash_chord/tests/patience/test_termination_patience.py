# SPDX-License-Identifier: Apache-2.0
"""ReconBody termination patience (dev, training only): config contract, the consecutive-failure counter kernel, and
the E2 export stripping the dev field. CPU-device kernel tests in the style of tests/lifecycle/test_termination.py."""

from __future__ import annotations

import numpy as np
import pytest

from flash_chord.configuration import compose_config, resolved_dict

R2 = "g1_recon_body_coord_rsi_r2"
PATIENCE = "g1_recon_body_coord_rsi_r2_patience"
NAMES = ("logging.experiment", "logging.run_name", "training.experiment_name", "training.run_name")
PATIENCE_KEYS = ("termination.violation_patience", "env.termination.violation_patience")


# ====================================================================== config contract
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


def test_patience_arm_differs_from_c3r2_only_in_the_patience_and_names():
    r2, arm = _flatten(_compose(R2)), _flatten(_compose(PATIENCE))
    changed = {key for key in r2.keys() | arm.keys() if r2.get(key, KeyError) != arm.get(key, KeyError)}
    unexpected = sorted(changed - {*PATIENCE_KEYS, *NAMES})
    assert not unexpected, f"{PATIENCE} drifted from {R2}: {unexpected}"
    assert all(arm[key] == 5 for key in PATIENCE_KEYS)
    assert all(key not in r2 for key in PATIENCE_KEYS)


def test_config_rejects_negative_patience():
    from flash_chord.lifecycle.termination import ReconBodyTerminationConfig

    assert ReconBodyTerminationConfig().violation_patience == 0
    with pytest.raises(ValueError):
        ReconBodyTerminationConfig(violation_patience=-1)


def test_export_strips_the_patience_from_both_termination_blocks():
    from flash_chord.training.ppo.coordinate import export

    stock, notes = export.stock_config_with_notes(_compose(PATIENCE))
    assert set(PATIENCE_KEYS) <= set(notes["removed_keys"])
    assert "violation_patience" not in stock["termination"]
    assert "violation_patience" not in stock["env"]["termination"]
    r2_stock, r2_notes = export.stock_config_with_notes(_compose(R2))
    assert not set(PATIENCE_KEYS) & set(r2_notes["removed_keys"])
    assert stock["env"]["termination"] == r2_stock["env"]["termination"]


# ====================================================================== CPU kernel tests
class _World:
    """One world with two palms and one object; palm and object errors set per call, as in test_termination.py."""

    def __init__(self, patience: int):
        import warp as wp

        self.wp = wp
        self.patience = patience
        self.reference_joint_q = np.zeros((1, 7), dtype=np.float32)
        self.reference_joint_q[:, 6] = 1.0
        self.reference_wrist_quat = np.zeros((1, 2, 4), dtype=np.float32)
        self.reference_wrist_quat[..., 3] = 1.0
        self.timestep = wp.zeros(1, dtype=wp.int32, device="cpu")
        self.age = wp.array([51], dtype=wp.int32, device="cpu")
        self.terminated = wp.zeros(1, dtype=wp.int32, device="cpu")
        self.truncated = wp.zeros(1, dtype=wp.int32, device="cpu")
        self.causes = [wp.zeros(1, dtype=wp.int32, device="cpu") for _ in range(3)]
        self.errors = [wp.zeros(1, dtype=wp.float32, device="cpu") for _ in range(6)]
        self.diagnostics = wp.zeros(9, dtype=wp.float32, device="cpu")
        self.count = wp.zeros(1, dtype=wp.int32, device="cpu")

    def step(self, palm=0.0, obj=0.0, pelvis=0.0, age=51):
        """Evaluate one control step; return (terminated, [pelvis, palm, object] causes, count)."""
        wp = self.wp
        from flash_chord.lifecycle.termination import (
            apply_recon_body_violation_patience,
            evaluate_recon_body_termination,
        )

        self.age.assign([age])
        with wp.ScopedDevice("cpu"):
            body_q = wp.array(
                [
                    wp.transform(wp.vec3(pelvis, 0.0, 0.0), wp.quat_identity()),
                    wp.transform(wp.vec3(palm, 0.0, 0.0), wp.quat_identity()),
                    wp.transform_identity(),
                    wp.transform(wp.vec3(obj, 0.0, 0.0), wp.quat_identity()),
                ],
                dtype=wp.transform,
            )
            wp.launch(
                evaluate_recon_body_termination,
                dim=1,
                inputs=[
                    body_q,
                    self.timestep,
                    self.age,
                    wp.array(self.reference_joint_q, dtype=wp.float32),
                    wp.zeros(2, dtype=wp.vec3),
                    wp.array(self.reference_wrist_quat.reshape(-1, 4), dtype=wp.quat),
                    wp.array([1, 2], dtype=wp.int32),
                    wp.zeros(2, dtype=wp.vec3),
                    wp.array([wp.quat_identity(), wp.quat_identity()], dtype=wp.quat),
                    wp.array([3], dtype=wp.int32),
                    wp.zeros(1, dtype=wp.vec3),
                    wp.array([wp.quat_identity()], dtype=wp.quat),
                    0,
                    1,
                    4,
                    2,
                    1,
                    50,
                    0.7,
                    1.5,
                    0.15,
                    1.5,
                    0.1,
                    1.5,
                    1,
                ],
                outputs=[self.terminated, self.truncated, *self.causes, *self.errors, self.diagnostics],
            )
            wp.launch(
                apply_recon_body_violation_patience,
                dim=1,
                inputs=[self.patience, self.causes[0]],
                outputs=[self.causes[1], self.causes[2], self.count, self.terminated, self.diagnostics],
            )
        causes = [int(cause.numpy()[0]) for cause in self.causes]
        assert self.diagnostics.numpy()[:3].tolist() == causes
        return int(self.terminated.numpy()[0]), causes, int(self.count.numpy()[0])


@pytest.mark.parametrize("patience", [1, 5])
def test_object_failure_terminates_only_after_more_than_patience_consecutive_steps(patience):
    world = _World(patience)
    for step in range(1, patience + 1):
        assert world.step(obj=0.2) == (0, [0, 0, 0], step)
    assert world.step(obj=0.2) == (1, [0, 0, 1], patience + 1)


def test_palm_failure_is_held_like_the_object_and_reports_its_cause():
    world = _World(5)
    for step in range(1, 6):
        assert world.step(palm=0.2, obj=0.2)[:2] == (0, [0, 0, 0])
    assert world.step(palm=0.2, obj=0.2) == (1, [0, 1, 1], 6)


def test_a_clean_step_resets_the_count():
    world = _World(5)
    for _ in range(4):
        world.step(obj=0.2)
    assert world.step(obj=0.0) == (0, [0, 0, 0], 0)
    for step in range(1, 6):
        assert world.step(obj=0.2) == (0, [0, 0, 0], step)
    assert world.step(obj=0.2)[0] == 1


def test_pelvis_failure_still_terminates_at_once():
    world = _World(5)
    assert world.step(pelvis=0.8) == (1, [1, 0, 0], 0)
    world = _World(5)
    world.step(obj=0.2)
    assert world.step(obj=0.2, pelvis=0.8) == (1, [1, 0, 0], 2)


def test_the_reset_freeze_clears_the_count():
    world = _World(5)
    for _ in range(3):
        world.step(obj=0.2)
    assert world.step(obj=0.2, age=0) == (0, [0, 0, 0], 0)
    assert world.step(obj=0.2) == (0, [0, 0, 0], 1)


def test_zero_patience_matches_stock_immediate_termination():
    world = _World(0)
    assert world.step(obj=0.2) == (1, [0, 0, 1], 1)
