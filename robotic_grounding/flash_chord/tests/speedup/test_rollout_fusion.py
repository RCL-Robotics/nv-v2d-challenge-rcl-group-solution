# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speed-up of the rollout loop (fused per-step bookkeeping, fused termination diagnostics, asynchronous SONIC ONNX
runs) without a change of results: the PPO batch, final observation, normalizer and key of the fused rollout are
bit-identical to the eager rollout of bbde02f (tests/speedup/reference_rollout_bbde02f.py), with and without the
non-finite guard, with a non-finite event, and with an unnormalized policy; its logged metrics are equal (integer
counts) or equal to float rounding (sums and means)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    return jax, jnp


def _reference():
    spec = importlib.util.spec_from_file_location("reference_rollout_bbde02f", HERE / "reference_rollout_bbde02f.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Env:
    """Deterministic env with every optional diagnostic; optionally one world goes non-finite at one step
    (``mode="flagged"``: the guarded Warp env's flag; ``mode="raw"``: NaN observation and reward)."""

    world_count = 4
    objective_term_names = ("term_a", "term_b")
    termination_cause_names = ("wrist", "object", "pelvis")
    tracking_error_names = ("position_m", "orientation_rad")

    def __init__(self, jnp, observation_dim, action_dim, event_step=None, event_world=1, mode="flagged",
                 publish_flag=True):
        self.jnp = jnp
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.step_count = 0
        self.event_step, self.event_world, self.mode, self.publish_flag = event_step, event_world, mode, publish_flag

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
        s = self.step_count
        observation = self._observation() + 0.1 * action.sum(-1, keepdims=True)
        reward = 1.0 - jnp.square(action).mean(axis=-1) + 0.05 * s
        terminated = np.asarray([s % 3 == 0, s % 7 == 0, s % 4 == 0, False])
        truncated = np.asarray([False, s % 3 == 0, s % 4 == 0, s % 5 == 0])
        causes = np.stack([terminated & (s % 2 == 0), terminated, terminated & (s % 3 == 0)], axis=-1)  # overlapping
        terms = jnp.stack([reward, 2.0 * reward], axis=-1)
        tracking = jnp.concatenate([jnp.abs(observation[:, :1]), jnp.abs(observation[:, 1:2]) * 3.0], axis=-1)
        flag = np.zeros(self.world_count, np.int32)
        if self.event_step == s:
            w = self.event_world
            tracking = tracking.at[w].set(jnp.nan)
            terms = terms.at[w, 1].set(jnp.nan)
            if self.mode == "flagged":
                terminated[w], truncated[w], causes[w], flag[w] = True, False, False, 1
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
            episode_return=jnp.where(jnp.asarray(completed), 1.5 * s + 0.25, 0.0),
            episode_length=jnp.asarray(np.where(completed, s, 0), dtype=jnp.int32),
            episode_reference_progress=jnp.where(jnp.asarray(completed), 0.1 * s, 0.0),
            objective_terms=terms,
            termination_causes=jnp.asarray(causes, dtype=jnp.int32),
            tracking_errors=tracking,
            nonfinite=jnp.asarray(flag) if self.publish_flag else None,
        )


def _assert_identical(jax, left, right):
    left_leaves, right_leaves = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for index, (a, b) in enumerate(zip(left_leaves, right_leaves, strict=True)):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, index
        np.testing.assert_array_equal(a, b, err_msg=f"leaf {index}")


def _assert_metrics(left, right):
    """Integer counts exactly equal; float sums and means equal to rounding; the same None fields."""
    for name in left.__dataclass_fields__:
        a, b = getattr(left, name), getattr(right, name)
        assert (a is None) == (b is None), name
        if a is None:
            continue
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, name
        if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
            np.testing.assert_array_equal(a, b, err_msg=name)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6, err_msg=name)


def _stock(jax, jnp, **overrides):
    from flash_chord.training.ppo.algorithm import create_train_state
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.network import ActorCritic
    from flash_chord.training.ppo.normalization import create_normalization_state

    values = dict(world_count=4, rollout_steps=9, max_iterations=3, save_interval=1, curriculum=None,
                  network=NetworkConfig(actor_hidden_dims=(16, 8), critic_hidden_dims=(16, 8), initial_noise_std=0.5),
                  ppo=PPOConfig(learning_epochs=2, mini_batches=2))
    values.update(overrides)
    config = TrainingConfig(**values)
    model = ActorCritic(action_dim=3, config=config.network)
    state = create_train_state(model, model.init(jax.random.key(3), jnp.zeros((1, 5))), config.ppo)
    normalization = create_normalization_state(5).replace(
        mean=jnp.full((5,), 0.1, jnp.float32), variance=jnp.full((5,), 0.7, jnp.float32), count=jnp.asarray(40, jnp.int32))
    return config, state, normalization


CASES = [
    dict(guard=False),
    dict(guard=True),
    dict(guard=True, event_step=4, mode="flagged"),
    dict(guard=True, event_step=4, mode="raw", publish_flag=False),
    dict(guard=False, normalize=False),
    dict(guard=True, normalize=False, event_step=2, mode="flagged"),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_fused_rollout_matches_the_eager_rollout(case):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.rollout import collect_rollout

    reference = _reference()
    config, state, normalization = _stock(jax, jnp, nonfinite_guard=case["guard"],
                                          observation_normalization=case.get("normalize", True))
    env_kwargs = {k: case[k] for k in ("event_step", "mode", "publish_flag") if k in case}
    outputs = []
    for collect in (reference.collect_rollout, collect_rollout):
        env = _Env(jnp, 5, 3, **env_kwargs)
        outputs.append(collect(env, state, normalization, env.reset(), jax.random.PRNGKey(11), config))
    (ref_batch, ref_obs, ref_norm, ref_key, ref_metrics), (batch, obs, norm, key, metrics) = outputs
    _assert_identical(jax, (ref_batch, ref_obs, ref_norm, ref_key), (batch, obs, norm, key))
    _assert_metrics(ref_metrics, metrics)


def _coordinated(**overrides):
    from flash_chord.envs.recon_body import RECON_BODY_OBSERVATION_BLOCKS
    from flash_chord.runtime.actions.sonic import SONIC_DIRECT_JOINT_NAMES, SONIC_POLICY_JOINT_NAMES
    from flash_chord.training.ppo.config import CoordinateConfig, NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.coordinate.layout import build_layout

    blocks, cursor = [], 0
    for name, width in RECON_BODY_OBSERVATION_BLOCKS:
        blocks.append({"name": name, "start": cursor, "end": cursor + width})
        cursor += width
    body = f"sonic_body_joint_residual[{','.join(SONIC_POLICY_JOINT_NAMES)}]"
    fingers = f"dex3_direct_joint_residual[{','.join(SONIC_DIRECT_JOINT_NAMES)}]"
    schema = {"observation": {"dimension": cursor, "blocks": blocks},
              "action": {"dimension": 43, "blocks": [{"name": body, "start": 0, "end": 29},
                                                     {"name": fingers, "start": 29, "end": 43}],
                         "input_mode": "raw", "input_mapping": "identity"}}
    values = dict(world_count=4, rollout_steps=7, max_iterations=3, save_interval=1, curriculum=None,
                  network=NetworkConfig(critic_hidden_dims=(32, 16), initial_noise_std=0.5),
                  ppo=PPOConfig(learning_epochs=2, mini_batches=2), coordinate=CoordinateConfig())
    values.update(overrides)
    return TrainingConfig(**values), build_layout(schema, CoordinateConfig()), cursor


@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: "-".join(f"{k}={v}" for k, v in c.items()))
def test_fused_coordinated_rollout_matches_the_eager_rollout(case):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.coordinate.per_group import group_matrix
    from flash_chord.training.ppo.learner import create_coordinated_learner
    from flash_chord.training.ppo.rollout import collect_rollout_coordinated

    reference = _reference()
    config, layout, observation_dim = _coordinated(nonfinite_guard=case["guard"])
    groups = group_matrix(layout.action_group, 3)
    env_kwargs = {k: case[k] for k in ("event_step", "mode", "publish_flag") if k in case}
    outputs = []
    for collect in (reference.collect_rollout_coordinated, collect_rollout_coordinated):
        env = _Env(jnp, observation_dim, 43, **env_kwargs)
        learner = create_coordinated_learner(env, config, layout)
        outputs.append(collect(env, learner.train_state, learner.normalization, learner.observation, learner.key,
                               config, groups))
    (ref_batch, ref_obs, ref_norm, ref_key, ref_metrics), (batch, obs, norm, key, metrics) = outputs
    _assert_identical(jax, (ref_batch, ref_obs, ref_norm, ref_key), (batch, obs, norm, key))
    _assert_metrics(ref_metrics, metrics)


def test_copy_transition_returns_fresh_buffers_and_exact_flags():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.rollout import copy_transition

    observation = jnp.asarray(np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)).at[0, 0].set(-0.0)
    reward = jnp.asarray([1.0, -0.0, np.inf, 2.5], jnp.float32)
    terminated = jnp.asarray([1, 0, 1, 0], jnp.int32)
    truncated = jnp.asarray([0, 1, 1, 0], jnp.int32)
    out = copy_transition(observation, reward, terminated, truncated)
    for source, copy in ((observation, out.observation), (reward, out.reward), (terminated, out.terminated),
                         (truncated, out.truncated)):
        assert copy is not source and copy.unsafe_buffer_pointer() != source.unsafe_buffer_pointer()
        np.testing.assert_array_equal(np.asarray(copy), np.asarray(source))
    assert np.signbit(np.asarray(out.observation)[0, 0]) and np.signbit(np.asarray(out.reward)[1])  # -0.0 kept
    np.testing.assert_array_equal(np.asarray(out.done), [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(np.asarray(out.reference_end), [False, True, False, False])


def test_fused_termination_diagnostics_match_the_eager_reduction():
    jax, jnp = _dependencies()
    from flash_chord.training.environment import _row_maxima_stacked, _stack_last

    rng = np.random.default_rng(1)
    causes = tuple(jnp.asarray(rng.integers(0, 2, 8), jnp.int32) for _ in range(3))
    errors = tuple(jnp.asarray(rng.standard_normal((8, k)).astype(np.float32)) for k in (1, 4, 2))
    np.testing.assert_array_equal(np.asarray(_stack_last(causes)), np.asarray(jnp.stack(causes, axis=-1)))
    eager = jnp.stack([array.max(axis=-1) for array in errors], axis=-1)
    np.testing.assert_array_equal(np.asarray(_row_maxima_stacked(errors)), np.asarray(eager))


def test_sonic_runs_are_asynchronous_by_default():
    from flash_chord.runtime.actions.sonic import SonicJointResidualActionConfig, _OrtBoundSession

    assert SonicJointResidualActionConfig().synchronize_ort_runs is False
    assert _OrtBoundSession.__dataclass_fields__["run_options"].default is None  # None = ORT default (synchronous)
