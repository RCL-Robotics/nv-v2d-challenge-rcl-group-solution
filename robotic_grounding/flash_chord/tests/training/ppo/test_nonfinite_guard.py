# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-finite guard (TrainingConfig.nonfinite_guard): bit-identical no-op on finite data; one bad world can no
longer poison the observation normalizer or the PPO batch (C3 ep07 iron NaN, 2026-10-11)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

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


# ------------------------------------------------------------------------------- fused per-step guard


def _assert_bitwise(name, left, right):
    left, right = np.asarray(left), np.asarray(right)
    assert left.dtype == right.dtype and left.shape == right.shape, name
    assert left.tobytes() == right.tobytes(), name


def _unfused_guard_262035e(jax, jnp, observation, reward, flag, terminated, value, bootstrapped, done,
                           episode_return, completed, terms, tracking):
    """The per-step guard ops exactly as committed in 262035e (one jitted mask, then ~12 eager dispatches)."""

    @jax.jit
    def nonfinite_rows(next_observation, step_reward, step_flag):
        observation_finite = jnp.all(jnp.isfinite(next_observation), axis=-1)
        bad = jnp.logical_not(observation_finite) | jnp.logical_not(jnp.isfinite(step_reward))
        if step_flag is not None:
            bad = bad | (step_flag != 0)
        sanitized = jnp.where(observation_finite[:, None], next_observation, jnp.zeros_like(next_observation))
        return bad, sanitized

    def rows_or_zero(bad, values):
        return jnp.where(bad.reshape(bad.shape + (1,) * (values.ndim - 1)), jnp.zeros_like(values), values)

    flag = None if flag is None else jnp.array(flag, copy=True)
    bad, sanitized = nonfinite_rows(jnp.array(observation, copy=True), jnp.array(reward, copy=True), flag)
    out = {"observation": sanitized, "valid": jnp.logical_not(bad), "nonfinite_count": bad.sum()}
    if flag is not None:
        out["nonfinite_terminated_count"] = jnp.logical_and(terminated != 0, flag != 0).sum()
    if bootstrapped.ndim == 2:
        out["reward"] = jnp.where(bad[:, None], value, bootstrapped)
    else:
        out["reward"] = jnp.where(bad, value, bootstrapped)
    out["done"] = jnp.where(bad, jnp.ones_like(done), done)
    out["raw_reward_mean"] = rows_or_zero(bad, reward).mean()
    finite_return = jnp.where(jnp.isfinite(episode_return), episode_return, 0.0)
    out["episode_return_sum"] = jnp.where(completed, finite_return, 0.0).sum()
    out["objective_terms"] = rows_or_zero(bad, jnp.array(terms, copy=True))
    out["tracking_errors"] = rows_or_zero(bad, jnp.array(tracking, copy=True))
    remaining = terminated.astype(bool)
    if flag is not None:
        remaining = jnp.logical_and(remaining, flag == 0)
    out["terminated_unflagged"] = remaining
    return out


@pytest.mark.parametrize("columns", [0, 3], ids=["stock", "coordinated"])
@pytest.mark.parametrize("case", ["finite", "events", "events_without_flag"])
def test_fused_guard_step_is_bitwise_equal_to_unfused_262035e_ops(columns, case):
    """The one-executable guard (rollout.guard_step) returns exactly the bits of the committed eager ops."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.rollout import guard_step

    worlds, dim = 4096, 486
    rng = np.random.default_rng(7 + columns)
    observation = (rng.standard_normal((worlds, dim)) * 3.0).astype(np.float32)
    reward = rng.standard_normal(worlds).astype(np.float32)
    terminated = (rng.uniform(size=worlds) < 0.02).astype(np.int32)
    truncated = (rng.uniform(size=worlds) < 0.02).astype(np.int32)
    flag = np.zeros(worlds, np.int32)
    value = rng.standard_normal((worlds, columns) if columns else worlds).astype(np.float32)
    episode_return = np.where(terminated | truncated, rng.standard_normal(worlds) * 50.0, 0.0).astype(np.float32)
    completed = jnp.asarray((terminated | truncated) > 0)
    terms = rng.standard_normal((worlds, 12)).astype(np.float32)
    tracking = np.abs(rng.standard_normal((worlds, 6))).astype(np.float32)
    if case != "finite":
        observation[5, 3] = np.nan
        observation[9, :] = np.inf
        observation[4000, 485] = -np.inf
        reward[17] = np.nan
        reward[18] = -np.inf
        flag[[30, 31, 4095]] = 1
        terminated[[31, 4095]] = 1
        episode_return[[40, 41]] = [np.nan, np.inf]
        terms[5, 2] = np.nan
        tracking[9, 0] = np.inf
    flag_input = None if case == "events_without_flag" else jnp.asarray(flag)
    observation, reward, value = jnp.asarray(observation), jnp.asarray(reward), jnp.asarray(value)
    terminated, truncated = jnp.asarray(terminated), jnp.asarray(truncated)
    episode_return, terms, tracking = jnp.asarray(episode_return), jnp.asarray(terms), jnp.asarray(tracking)
    # Stock arithmetic, unchanged by the guard and computed outside it.
    done = jnp.maximum(terminated, truncated).astype(jnp.float32)
    reference_end = jnp.logical_and(truncated.astype(bool), jnp.logical_not(terminated.astype(bool)))
    if columns:
        bootstrapped = reward[:, None] + 0.99 * value * reference_end.astype(jnp.float32)[:, None]
    else:
        bootstrapped = reward + 0.99 * value * reference_end.astype(jnp.float32)

    expected = _unfused_guard_262035e(jax, jnp, observation, reward, flag_input, terminated, value, bootstrapped,
                                      done, episode_return, completed, terms, tracking)
    fused = guard_step(jnp.array(observation, copy=True), reward, flag_input, terminated, value, bootstrapped, done,
                       episode_return, terms, tracking)
    actual = {
        "observation": fused.observation,
        "valid": fused.valid,
        "nonfinite_count": fused.nonfinite_count,
        "reward": fused.reward,
        "done": fused.done,
        "raw_reward_mean": fused.raw_reward.mean(),
        "episode_return_sum": jnp.where(completed, fused.episode_return, 0.0).sum(),
        "objective_terms": fused.objective_terms,
        "tracking_errors": fused.tracking_errors,
        "terminated_unflagged": fused.terminated_unflagged,
    }
    if flag_input is not None:
        actual["nonfinite_terminated_count"] = fused.nonfinite_terminated_count
    else:
        assert fused.nonfinite_terminated_count is None
    assert sorted(actual) == sorted(expected)
    for name in expected:
        _assert_bitwise(name, actual[name], expected[name])
    bad_rows = {"finite": 0, "events": 8, "events_without_flag": 5}[case]
    assert int(fused.nonfinite_count) == bad_rows
    if case == "finite":
        # All finite: every selection is the identity (the stock values, bit for bit).
        _assert_bitwise("observation", fused.observation, observation)
        _assert_bitwise("reward", fused.reward, bootstrapped)
        _assert_bitwise("done", fused.done, done)
        _assert_bitwise("raw_reward_mean", fused.raw_reward.mean(), reward.mean())
    for leaf in jax.tree.leaves(fused):
        assert np.isfinite(np.asarray(leaf)).all()


def test_rollout_calls_the_fused_guard_once_per_step():
    """Every guarded step goes through the single guard_step executable (plus the normalizer guard)."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo import rollout

    config, state, normalization = _stock_setup(jax, jnp)
    calls = []
    original = rollout.guard_step

    def counting(*args):
        calls.append(len(args))
        return original(*args)

    rollout.guard_step = counting
    try:
        env = _GuardEnv(jnp, event_step=2, event_world=1, mode="flagged")
        rollout.collect_rollout(env, state, normalization, env.reset(), jax.random.PRNGKey(11), config)
    finally:
        rollout.guard_step = original
    assert len(calls) == config.rollout_steps


# ------------------------------------------------------------------------------------ resume finite check


def _leaf_paths(jax, learner):
    return [(jax.tree_util.keystr(path), leaf) for path, leaf in jax.tree_util.tree_flatten_with_path(learner)[0]]


def _poison_checkpoint(source, target, leaf_index, value=np.nan):
    """Copy a checkpoint with one entry of one saved leaf replaced (format, manifest and metadata unchanged)."""
    from safetensors.numpy import save_file

    tensors, metadata = _checkpoint_tensors(source)
    key = f"state.{leaf_index:04d}"
    array = tensors[key].copy()
    array.reshape(-1)[-1] = value
    tensors[key] = array
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata=metadata)


def test_nonfinite_leaves_reports_only_nonfinite_floating_leaves():
    jax, jnp = _dependencies()
    from flash_chord.training.checkpoint import nonfinite_leaves

    tree = {
        "finite": jnp.ones((3,)),
        "ints": jnp.arange(4),
        "key": jax.random.key(0),
        "raw_key": jax.random.PRNGKey(0),
        "nested": {"bad": jnp.asarray([1.0, np.nan, np.inf, -np.inf]), "half": jnp.asarray([np.inf], jnp.bfloat16)},
        "host": np.asarray([[0.0, np.nan]], np.float64),
    }
    assert sorted(nonfinite_leaves(tree)) == [("['host']", 1, 2), ("['nested']['bad']", 3, 4),
                                              ("['nested']['half']", 1, 1)]
    assert nonfinite_leaves({"finite": jnp.ones((3,)), "ints": jnp.arange(4)}) == []


@pytest.mark.parametrize("leaf", ["params", "opt_state", "normalization", "observation"])
def test_guarded_resume_refuses_a_checkpoint_with_nonfinite_values(tmp_path, leaf):
    """Actor/critic parameters, optimizer moments, normalizer and cached observation are each checked."""
    jax, jnp = _dependencies()
    from flash_chord.training.checkpoint import NonfiniteCheckpointError
    from flash_chord.training.ppo.runner import run_training

    config, _, _ = _stock_setup(jax, jnp, rollout_steps=4, max_iterations=2)
    learner = run_training(_GuardEnv(jnp), config, checkpoint_dir=tmp_path / "run", checkpoint_metadata=_metadata())
    prefix = {"params": ".train_state.params", "opt_state": ".train_state.opt_state",
              "normalization": ".normalization.variance", "observation": ".observation"}[leaf]
    paths = _leaf_paths(jax, learner)
    index = next(i for i, (path, value) in enumerate(paths)
                 if path.startswith(prefix) and jnp.issubdtype(value.dtype, jnp.floating))
    healthy = tmp_path / "run" / "model_2.safetensors"
    poisoned = tmp_path / "poisoned" / "model_2.safetensors"
    _poison_checkpoint(healthy, poisoned, index, np.inf if leaf == "observation" else np.nan)

    resume = replace(config, max_iterations=3, resume=True, checkpoint=str(poisoned))
    with pytest.raises(NonfiniteCheckpointError) as error:
        run_training(_GuardEnv(jnp), resume, checkpoint_dir=tmp_path / "poisoned", checkpoint_metadata=_metadata())
    message = str(error.value)
    assert f"refusing to resume from {poisoned}: 1 saved tensor(s)" in message
    assert f"model_2.safetensors{paths[index][0]}: 1/{np.asarray(paths[index][1]).size} non-finite" in message
    assert "+training.nonfinite_guard=false" in message
    assert not (tmp_path / "poisoned" / "model_3.safetensors").exists()

    # The same resume from the finite checkpoint works, and the documented override restores stock behavior.
    resumed = run_training(_GuardEnv(jnp), replace(resume, checkpoint=str(healthy)), checkpoint_metadata=_metadata())
    assert int(resumed.iteration) == 3
    if leaf in ("normalization", "observation"):
        unguarded = run_training(_GuardEnv(jnp), replace(resume, nonfinite_guard=False),
                                 checkpoint_metadata=_metadata())
        assert int(unguarded.iteration) == 3


def test_guarded_resume_refuses_a_checkpoint_poisoned_by_a_stock_run(tmp_path):
    """The ep07 failure: an unguarded run poisons its normalizer; a guarded retry must not resume from it."""
    jax, jnp = _dependencies()
    from flash_chord.training.checkpoint import NonfiniteCheckpointError
    from flash_chord.training.ppo.runner import run_training

    config, _, _ = _stock_setup(jax, jnp, rollout_steps=4, nonfinite_guard=False, max_iterations=3)
    env = _GuardEnv(jnp, event_step=6, event_world=2, mode="raw", publish_flag=False)  # iteration 2, step 2
    run_training(env, config, checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())

    def retry(iteration):
        return replace(config, nonfinite_guard=True, max_iterations=4, resume=True,
                       checkpoint=str(tmp_path / f"model_{iteration}.safetensors"))

    # model_1 predates the event and resumes; model_2 and model_3 were saved after it.
    assert int(run_training(_GuardEnv(jnp), retry(1), checkpoint_metadata=_metadata()).iteration) == 4
    for iteration in (2, 3):
        with pytest.raises(NonfiniteCheckpointError, match=r"model_\d\.safetensors\.normalization\.mean: \d/5"):
            run_training(_GuardEnv(jnp), retry(iteration), checkpoint_metadata=_metadata())


class _SidecarRSI:
    """RSI hooks that write a real sidecar (bins, keyframe bank, pending counts) and record loads."""

    def __init__(self):
        self.loaded = []

    def host_update(self, iteration):
        return {}

    def save_state(self, path):
        from safetensors.numpy import save_file

        save_file({"bins.top": np.linspace(0.0, 1.0, 4), "bank.q": np.zeros((2, 3), np.float32),
                   "pending.n_iter": np.zeros(4, np.int32)}, str(path))

    def load_state(self, path):
        self.loaded.append(Path(path).name)


def test_guarded_resume_refuses_a_nonfinite_rsi_sidecar(tmp_path):
    jax, jnp = _dependencies()
    from safetensors.numpy import load_file, save_file

    from flash_chord.training.checkpoint import NonfiniteCheckpointError
    from flash_chord.training.ppo.runner import run_training

    config, _, _ = _stock_setup(jax, jnp, rollout_steps=4, max_iterations=2)
    env = _GuardEnv(jnp)
    env.rsi = _SidecarRSI()
    run_training(env, config, checkpoint_dir=tmp_path, checkpoint_metadata=_metadata())
    resume = replace(config, max_iterations=3, resume=True, checkpoint=str(tmp_path / "model_2.safetensors"))

    resumed_env = _GuardEnv(jnp)
    resumed_env.rsi = _SidecarRSI()
    run_training(resumed_env, resume, checkpoint_metadata=_metadata())
    assert resumed_env.rsi.loaded == ["rsi_2.safetensors"]

    sidecar = tmp_path / "rsi_2.safetensors"
    tensors = load_file(str(sidecar))
    tensors["bank.q"] = tensors["bank.q"].copy()
    tensors["bank.q"][1, 2] = np.inf
    save_file(tensors, str(sidecar))
    refused_env = _GuardEnv(jnp)
    refused_env.rsi = _SidecarRSI()
    with pytest.raises(NonfiniteCheckpointError, match=r"rsi_2\.safetensors\['bank\.q'\]: 1/6 non-finite"):
        run_training(refused_env, resume, checkpoint_metadata=_metadata())
    assert refused_env.rsi.loaded == []  # refused before the sidecar was loaded
