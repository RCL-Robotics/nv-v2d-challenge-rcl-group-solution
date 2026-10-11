# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-finite guard (TrainingConfig.nonfinite_guard): bit-identical no-op on finite data; one bad world can no
longer poison the observation normalizer or the PPO batch (C3 ep07 iron NaN, 2026-10-11)."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    return jax, jnp


def _assert_trees_equal(jax, left, right):
    # Leaves only: static TrainState fields (apply_fn, tx) differ by identity between independently built learners.
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for index, (a, b) in enumerate(zip(left_leaves, right_leaves, strict=True)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, index
        np.testing.assert_array_equal(a, b, err_msg=f"leaf {index}")


# ----------------------------------------------------------------------------------------- normalizer


def _random_batches(seed, count, rows, dim):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((rows, dim)) * rng.uniform(0.1, 30.0, dim) + rng.uniform(-5, 5, dim))
            .astype(np.float32) for _ in range(count)]


def test_guarded_normalization_is_bitwise_identity_on_finite_batches():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import (
        create_normalization_state,
        guard_normalization_update,
        update_normalization,
    )

    for rows, dim in ((4, 5), (64, 486), (4096, 486)):
        stock = guarded = create_normalization_state(dim)
        for batch in _random_batches(rows + dim, 4, rows, dim):
            observation = jnp.asarray(batch)
            stock = update_normalization(stock, observation)
            candidate = update_normalization(guarded, observation)
            guarded = guard_normalization_update(guarded, candidate, observation, jnp.ones((rows,), bool))
            _assert_trees_equal(jax, stock, guarded)


def test_guarded_normalization_ignores_nonfinite_and_invalid_rows():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import (
        create_normalization_state,
        guard_normalization_update,
        update_normalization,
    )

    first, second = _random_batches(3, 2, 8, 6)
    previous = update_normalization(create_normalization_state(6), jnp.asarray(first))
    poisoned = second.copy()
    poisoned[1, 2] = np.nan
    poisoned[4, :] = np.inf
    poisoned[6, 0] = -np.inf
    valid = np.ones(8, bool)
    valid[7] = False  # flagged by the environment although finite
    keep = np.asarray([True, False, True, True, False, True, False, False])
    candidate = update_normalization(previous, jnp.asarray(poisoned))
    assert not np.isfinite(np.asarray(candidate.mean)).all()  # the stock update is poisoned for good

    guarded = guard_normalization_update(previous, candidate, jnp.asarray(poisoned), jnp.asarray(valid))
    reference = update_normalization(previous, jnp.asarray(second[keep]))
    assert int(guarded.count) == int(previous.count) + int(keep.sum())
    np.testing.assert_allclose(np.asarray(guarded.mean), np.asarray(reference.mean), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(guarded.variance), np.asarray(reference.variance), rtol=2e-5, atol=1e-6)
    # Float64 population statistics of all accepted rows.
    full = np.concatenate([first, second[keep]]).astype(np.float64)
    np.testing.assert_allclose(np.asarray(guarded.mean), full.mean(0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(guarded.variance), full.var(0), rtol=1e-4, atol=1e-5)


def test_guarded_normalization_keeps_previous_without_usable_rows_and_skips_overflow_rows():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import (
        create_normalization_state,
        guard_normalization_update,
        update_normalization,
    )

    (first,) = _random_batches(5, 1, 4, 3)
    previous = update_normalization(create_normalization_state(3), jnp.asarray(first))
    all_bad = np.full((4, 3), np.nan, np.float32)
    candidate = update_normalization(previous, jnp.asarray(all_bad))
    kept = guard_normalization_update(previous, candidate, jnp.asarray(all_bad), jnp.ones((4,), bool))
    _assert_trees_equal(jax, previous, kept)
    kept = guard_normalization_update(previous, update_normalization(previous, jnp.asarray(first)),
                                      jnp.asarray(first), jnp.zeros((4,), bool))
    _assert_trees_equal(jax, previous, kept)

    # Finite values whose square overflows float32 make the stock variance infinite; those rows are dropped.
    overflow = first.copy()
    overflow[2, 1] = 3.0e30
    candidate = update_normalization(previous, jnp.asarray(overflow))
    assert not np.isfinite(np.asarray(candidate.variance)).all()
    guarded = guard_normalization_update(previous, candidate, jnp.asarray(overflow), jnp.ones((4,), bool))
    reference = update_normalization(previous, jnp.asarray(first[[0, 1, 3]]))
    assert int(guarded.count) == int(previous.count) + 3
    np.testing.assert_allclose(np.asarray(guarded.mean), np.asarray(reference.mean), rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(guarded.variance), np.asarray(reference.variance), rtol=2e-5, atol=1e-6)


# -------------------------------------------------------------------------------------------- rollout


class _GuardEnv:
    """Deterministic env; optionally one world goes non-finite at one global step.

    ``mode="flagged"`` mimics the guarded Warp env (flag=1, terminated=1, finite post-reset observation, zero
    reward, NaN diagnostics). ``mode="raw"`` mimics an unguarded env (no flag, NaN observation and reward).
    """

    world_count = 4
    observation_dim = 5
    action_dim = 3
    objective_term_names = ("term_a", "term_b")
    termination_cause_names = ("wrist", "object")
    tracking_error_names = ("position_m",)

    def __init__(self, jnp, event_step=None, event_world=1, mode="flagged", publish_flag=True):
        self.jnp = jnp
        self.step_count = 0
        self.global_step = 0
        self.event_step = event_step
        self.event_world = event_world
        self.mode = mode
        self.publish_flag = publish_flag

    def _observation(self):
        base = self.jnp.arange(self.world_count * self.observation_dim, dtype=self.jnp.float32)
        return self.jnp.sin(0.37 * (self.step_count + 1) + base.reshape(self.world_count, self.observation_dim))

    def reset(self):
        self.step_count = 0
        return self._observation()

    def step(self, action):
        from flash_chord.training.environment import VectorStep

        jnp = self.jnp
        self.step_count += 1
        self.global_step += 1
        s = self.step_count
        observation = self._observation() + 0.1 * action.sum(-1, keepdims=True)
        reward = 1.0 - jnp.square(action).mean(axis=-1) + 0.05 * s
        terminated = np.asarray([s % 3 == 0, False, s % 4 == 0, False])
        truncated = np.asarray([False, s % 3 == 0, s % 4 == 0, s % 5 == 0])
        causes = np.stack([terminated & (s % 2 == 0), terminated & (s % 2 == 1)], axis=-1)
        terms = jnp.stack([reward, 2.0 * reward], axis=-1)
        tracking = jnp.abs(observation[:, :1])
        flag = np.zeros(self.world_count, np.int32)
        if self.event_step == self.global_step:
            w = self.event_world
            tracking = tracking.at[w].set(jnp.nan)
            terms = terms.at[w, 1].set(jnp.nan)
            if self.mode == "flagged":
                terminated[w] = True
                truncated[w] = False
                causes[w] = False
                flag[w] = 1
                reward = reward.at[w].set(0.0)
            else:
                observation = observation.at[w, 1:3].set(jnp.nan)
                reward = reward.at[w].set(jnp.nan)
        completed = terminated | truncated
        return VectorStep(
            observation=observation,
            reward=reward,
            terminated=jnp.asarray(terminated, dtype=jnp.int32),
            truncated=jnp.asarray(truncated, dtype=jnp.int32),
            episode_return=jnp.where(jnp.asarray(completed), float(s), 0.0),
            episode_length=jnp.asarray(np.where(completed, s, 0), dtype=jnp.int32),
            episode_reference_progress=jnp.where(jnp.asarray(completed), 0.5, 0.0),
            objective_terms=terms,
            termination_causes=jnp.asarray(causes, dtype=jnp.int32),
            tracking_errors=tracking,
            nonfinite=jnp.asarray(flag) if self.publish_flag else None,
        )


def _stock_setup(jax, jnp, rollout_steps=6, **overrides):
    from flash_chord.training.ppo.algorithm import create_train_state
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.network import ActorCritic
    from flash_chord.training.ppo.normalization import create_normalization_state

    values = dict(
        world_count=4,
        rollout_steps=rollout_steps,
        max_iterations=3,
        save_interval=1,
        curriculum=None,
        network=NetworkConfig(actor_hidden_dims=(16, 8), critic_hidden_dims=(16, 8), initial_noise_std=0.5),
        ppo=PPOConfig(learning_epochs=2, mini_batches=2),
    )
    values.update(overrides)
    config = TrainingConfig(**values)
    model = ActorCritic(action_dim=3, config=config.network)
    parameters = model.init(jax.random.key(3), jnp.zeros((1, 5)))
    state = create_train_state(model, parameters, config.ppo)
    # A non-trivial running normalizer so normalized observations differ from raw ones.
    normalization = create_normalization_state(5).replace(
        mean=jnp.full((5,), 0.1, jnp.float32), variance=jnp.full((5,), 0.7, jnp.float32),
        count=jnp.asarray(40, jnp.int32))
    return config, state, normalization


@pytest.mark.parametrize("publish_flag", [True, False])
def test_rollout_guard_is_bitwise_noop_without_nonfinite_values(publish_flag):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.rollout import collect_rollout

    config, state, normalization = _stock_setup(jax, jnp)
    outputs = {}
    for guard in (False, True):
        env = _GuardEnv(jnp, publish_flag=publish_flag)
        outputs[guard] = collect_rollout(env, state, normalization, env.reset(), jax.random.PRNGKey(11),
                                         replace(config, nonfinite_guard=guard))
    plain, guarded = outputs[False], outputs[True]
    assert int(guarded[4].nonfinite_world_count) == 0
    assert plain[4].nonfinite_world_count is None
    if publish_flag:
        assert int(guarded[4].nonfinite_terminated_count) == 0
    stripped = guarded[4].replace(nonfinite_world_count=None, nonfinite_terminated_count=None)
    _assert_trees_equal(jax, plain[:4] + (plain[4],), guarded[:4] + (stripped,))


@pytest.mark.parametrize("mode", ["flagged", "raw"])
def test_rollout_guard_isolates_one_nonfinite_world(mode):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.algorithm import compute_gae
    from flash_chord.training.ppo.normalization import guard_normalization_update, update_normalization
    from flash_chord.training.ppo.rollout import collect_rollout

    event_step, world = 3, 1
    config, state, normalization = _stock_setup(jax, jnp)
    env = _GuardEnv(jnp, event_step=event_step, event_world=world, mode=mode, publish_flag=(mode == "flagged"))
    batch, observation, final_norm, _, metrics = collect_rollout(
        env, state, normalization, env.reset(), jax.random.PRNGKey(11), config)

    for leaf in jax.tree.leaves((batch, observation, final_norm, metrics)):
        assert np.isfinite(np.asarray(leaf)).all()
    k = event_step - 1
    done = np.asarray(batch.done)
    assert done[k, world] == 1.0
    np.testing.assert_array_equal(np.asarray(batch.reward)[k, world], np.asarray(batch.value)[k, world])
    assert int(metrics.nonfinite_world_count) == 1
    if mode == "flagged":
        assert int(metrics.nonfinite_terminated_count) == 1
        # Guard terminations are their own outcome class: every completed episode is classified exactly once.
        classified = int(metrics.reference_end_count) + int(np.asarray(metrics.termination_cause_count).sum()) + 1
        assert classified == int(metrics.episode_count)
    else:
        assert metrics.nonfinite_terminated_count is None

    # Zero TD error and zero advantage for the bad world-step (GAE stops there because done = 1).
    last_value = jnp.zeros((4,))
    advantage, _ = compute_gae(batch.reward, batch.value, batch.done, last_value,
                               config.ppo.discount, config.ppo.gae_lambda)
    assert np.asarray(advantage)[k, world] == 0.0

    # The normalizer equals the stock update with the bad row removed at the event step only.
    replay = _GuardEnv(jnp, event_step=event_step, event_world=world, mode=mode, publish_flag=(mode == "flagged"))
    replay.reset()
    expected = normalization
    for step in range(config.rollout_steps):
        transition = replay.step(jnp.asarray(np.asarray(batch.action)[step]))
        rows = jnp.asarray(transition.observation)
        valid = jnp.ones((4,), bool)
        if step == k:
            valid = valid.at[world].set(False)
            rows = jnp.where(jnp.all(jnp.isfinite(rows), axis=-1, keepdims=True), rows, 0.0)
        expected = guard_normalization_update(expected, update_normalization(expected, rows), rows, valid)
    _assert_trees_equal(jax, expected, final_norm)
    assert int(final_norm.count) == int(normalization.count) + 4 * config.rollout_steps - 1


def _coordinate_layout():
    from flash_chord.envs.recon_body import RECON_BODY_OBSERVATION_BLOCKS
    from flash_chord.runtime.actions.sonic import SONIC_DIRECT_JOINT_NAMES, SONIC_POLICY_JOINT_NAMES
    from flash_chord.training.ppo.config import CoordinateConfig
    from flash_chord.training.ppo.coordinate.layout import build_layout

    blocks = []
    cursor = 0
    for name, width in RECON_BODY_OBSERVATION_BLOCKS:
        blocks.append({"name": name, "start": cursor, "end": cursor + width})
        cursor += width
    body = f"sonic_body_joint_residual[{','.join(SONIC_POLICY_JOINT_NAMES)}]"
    fingers = f"dex3_direct_joint_residual[{','.join(SONIC_DIRECT_JOINT_NAMES)}]"
    schema = {
        "observation": {"dimension": cursor, "blocks": blocks},
        "action": {
            "dimension": 43,
            "blocks": [{"name": body, "start": 0, "end": 29}, {"name": fingers, "start": 29, "end": 43}],
            "input_mode": "raw",
            "input_mapping": "identity",
        },
    }
    return schema, build_layout(schema, CoordinateConfig())


class _CoordinatedGuardEnv(_GuardEnv):
    observation_dim = 486
    action_dim = 43


def _coordinated_config(**overrides):
    from flash_chord.training.ppo.config import CoordinateConfig, NetworkConfig, PPOConfig, TrainingConfig

    values = dict(
        world_count=4,
        rollout_steps=4,
        max_iterations=3,
        save_interval=1,
        curriculum=None,
        network=NetworkConfig(critic_hidden_dims=(32, 16), initial_noise_std=0.5),
        ppo=PPOConfig(learning_epochs=2, mini_batches=2),
        coordinate=CoordinateConfig(),
    )
    values.update(overrides)
    return TrainingConfig(**values)


def test_coordinated_rollout_guard_noop_and_event():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.coordinate.per_group import group_matrix
    from flash_chord.training.ppo.learner import create_coordinated_learner
    from flash_chord.training.ppo.rollout import collect_rollout_coordinated

    _, layout = _coordinate_layout()
    groups = group_matrix(layout.action_group, 3)
    outputs = {}
    for guard in (False, True):
        config = _coordinated_config(nonfinite_guard=guard)
        env = _CoordinatedGuardEnv(jnp)
        learner = create_coordinated_learner(env, config, layout)
        outputs[guard] = collect_rollout_coordinated(env, learner.train_state, learner.normalization,
                                                     learner.observation, learner.key, config, groups)
    stripped = outputs[True][4].replace(nonfinite_world_count=None, nonfinite_terminated_count=None)
    _assert_trees_equal(jax, outputs[False][:4] + (outputs[False][4],), outputs[True][:4] + (stripped,))

    config = _coordinated_config()
    env = _CoordinatedGuardEnv(jnp, event_step=2, event_world=2, mode="raw", publish_flag=False)
    learner = create_coordinated_learner(env, config, layout)
    batch, observation, normalization, _, metrics = collect_rollout_coordinated(
        env, learner.train_state, learner.normalization, learner.observation, learner.key, config, groups)
    for leaf in jax.tree.leaves((batch, observation, normalization, metrics)):
        assert np.isfinite(np.asarray(leaf)).all()
    assert np.asarray(batch.done)[1, 2] == 1.0
    np.testing.assert_array_equal(np.asarray(batch.reward)[1, 2], np.asarray(batch.value)[1, 2])  # all 3 columns
    assert int(normalization.count) == 4 * config.rollout_steps - 1


# --------------------------------------------------------------------------------------------- runner


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, metrics, step):
        self.records.append((step, {k: v for k, v in metrics.items() if not k.startswith("throughput/")}))

    def close(self):
        pass


def _checkpoint_tensors(path):
    from safetensors import safe_open

    with safe_open(str(path), "numpy") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}, handle.metadata()


def _metadata(**extra):
    metadata = {"resolved_config_json": json.dumps({"reference": {"object_articulations": {"version": 1,
                                                                                            "entries": []}}})}
    metadata.update(extra)
    return metadata


@pytest.mark.parametrize("coordinated", [False, True])
def test_runner_guard_is_bitwise_noop_across_iterations(tmp_path, coordinated):
    """Three PPO iterations (rollout, normalizer, update, checkpoint, records) are identical with the guard."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    results = {}
    for guard in (False, True):
        output = tmp_path / f"guard_{guard}"
        output.mkdir()
        logger = _Logger()
        if coordinated:
            schema, layout = _coordinate_layout()
            config = _coordinated_config(nonfinite_guard=guard)
            env = _CoordinatedGuardEnv(jnp)
            learner = run_training(env, config, logger=logger, checkpoint_dir=output,
                                   checkpoint_metadata=_metadata(policy_schema_json=json.dumps(schema)))
        else:
            config, _, _ = _stock_setup(jax, jnp, rollout_steps=4, nonfinite_guard=guard)
            env = _GuardEnv(jnp)
            learner = run_training(env, config, logger=logger, checkpoint_dir=output, checkpoint_metadata=_metadata())
        results[guard] = (output, logger.records, learner)
    assert results[False][1] == results[True][1]
    _assert_trees_equal(jax, results[False][2], results[True][2])
    for iteration in (1, 2, 3):
        name = f"model_{iteration}.safetensors"
        plain, plain_meta = _checkpoint_tensors(results[False][0] / name)
        guarded, guarded_meta = _checkpoint_tensors(results[True][0] / name)
        assert plain_meta == guarded_meta
        assert sorted(plain) == sorted(guarded)
        for key in plain:
            np.testing.assert_array_equal(plain[key], guarded[key], err_msg=key)


@pytest.mark.parametrize("guard", [True, False])
def test_runner_survives_one_nonfinite_world(tmp_path, guard, capsys):
    """With the guard, one NaN world-step costs nothing; without it the normalizer is poisoned for good."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    config, _, _ = _stock_setup(jax, jnp, rollout_steps=4, nonfinite_guard=guard, max_iterations=4)
    logger = _Logger()
    env = _GuardEnv(jnp, event_step=6, event_world=2, mode="raw", publish_flag=False)  # iteration 2, step 2
    learner = run_training(env, config, logger=logger, checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    skipped = [record["ppo/skipped_update_fraction_per_iteration"] for _, record in logger.records]
    normalizer_finite = bool(np.isfinite(np.asarray(learner.normalization.mean)).all()
                             and np.isfinite(np.asarray(learner.normalization.variance)).all())
    if guard:
        assert skipped == [0.0, 0.0, 0.0, 0.0]
        assert normalizer_finite
        for leaf in jax.tree.leaves(learner):
            assert np.isfinite(np.asarray(leaf)).all()
        guard_records = [record for _, record in logger.records if "guard/nonfinite_world_steps_per_iteration" in record]
        assert len(guard_records) == 1
        assert guard_records[0]["guard/nonfinite_world_steps_per_iteration"] == 1.0
        assert logger.records[-1][1]["guard/nonfinite_world_steps_total"] == 1.0
        assert "guard/nonfinite_world_steps_total" not in logger.records[0][1]
        assert "[nonfinite-guard] iteration 2: 1 non-finite world-steps" in capsys.readouterr().out
    else:
        # Stock behavior (the ep07 failure mode): poisoned normalizer, every later update skipped.
        assert not normalizer_finite
        assert skipped[0] == 0.0 and skipped[2:] == [1.0, 1.0]


def test_runner_classifies_guard_terminations(tmp_path):
    """A flagged world terminated without a tracking cause must not break the outcome-count invariant."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.runner import run_training

    config, _, _ = _stock_setup(jax, jnp, rollout_steps=4, max_iterations=2)
    logger = _Logger()
    env = _GuardEnv(jnp, event_step=5, event_world=1, mode="flagged")
    run_training(env, config, logger=logger, checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    record = logger.records[1][1]
    assert record["termination/nonfinite_fraction_per_completed_episode"] > 0.0
    assert record["guard/nonfinite_world_steps_per_iteration"] == 1.0
    assert "termination/nonfinite_fraction_per_completed_episode" not in logger.records[0][1]
    assert all(np.isfinite(value) for value in record.values())
