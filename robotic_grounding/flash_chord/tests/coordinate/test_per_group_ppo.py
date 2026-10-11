# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-agent (W0-P) PPO: K=1 stock identity, K=2 CH parity, K=3 NumPy reference, learner and runner."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

STOCK_SRC = Path(
    "/root/autodl-tmp/haoran.zhang/code/video_to_data_8c778f9/robotic_grounding/flash_chord/src"
)
DEV_SRC = Path(__file__).resolve().parents[2] / "src"


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    pytest.importorskip("distrax")
    return jax, jnp


def _policy_schema():
    from flash_chord.envs.recon_body import RECON_BODY_OBSERVATION_BLOCKS
    from flash_chord.runtime.actions.sonic import SONIC_DIRECT_JOINT_NAMES, SONIC_POLICY_JOINT_NAMES

    blocks = []
    cursor = 0
    for name, width in RECON_BODY_OBSERVATION_BLOCKS:
        blocks.append({"name": name, "start": cursor, "end": cursor + width})
        cursor += width
    body = f"sonic_body_joint_residual[{','.join(SONIC_POLICY_JOINT_NAMES)}]"
    fingers = f"dex3_direct_joint_residual[{','.join(SONIC_DIRECT_JOINT_NAMES)}]"
    return {
        "observation": {"dimension": cursor, "blocks": blocks},
        "action": {
            "dimension": 43,
            "blocks": [{"name": body, "start": 0, "end": 29}, {"name": fingers, "start": 29, "end": 43}],
            "input_mode": "raw",
            "input_mapping": "identity",
        },
    }


def _layout(**overrides):
    from flash_chord.training.ppo.config import CoordinateConfig
    from flash_chord.training.ppo.coordinate.layout import build_layout

    return build_layout(_policy_schema(), CoordinateConfig(**overrides))


def _perturb(jax, tree, scale, seed):
    leaves, structure = jax.tree.flatten(tree)
    rng = np.random.default_rng(seed)
    return jax.tree.unflatten(
        structure,
        [np.asarray(leaf) + scale * rng.standard_normal(np.shape(leaf)).astype(np.float32) for leaf in leaves],
    )


# --------------------------------------------------------------------------- NumPy references (float64)


def _np_log_prob_per_group(action, mean, std, action_group, num_groups):
    per_dimension = -0.5 * ((action - mean) / std) ** 2 - np.log(std) - 0.5 * math.log(2.0 * math.pi)
    output = np.zeros(action.shape[:-1] + (num_groups,))
    for dimension, group in enumerate(action_group):
        output[..., group] += per_dimension[..., dimension]
    return output


def _np_gae(reward, value, done, last_value, discount, gae_lambda):
    steps = reward.shape[0]
    advantage = np.zeros_like(reward)
    running = np.zeros_like(last_value)
    next_value = last_value
    for step in reversed(range(steps)):
        not_done = (1.0 - done[step])[..., None]
        delta = reward[step] + discount * next_value * not_done - value[step]
        running = delta + discount * gae_lambda * not_done * running
        advantage[step] = running
        next_value = value[step]
    return advantage, advantage + value


def _np_normalize(advantage, ddof=0):
    return (advantage - advantage.mean(axis=0)) / (advantage.std(axis=0, ddof=ddof) + 1.0e-8)


# CH dexmachina/rl/wrench_ppo.py:46-57 and ablation_ppo.py:191-222; rl_games common_losses.py.
def _ch_per_hand_neglogp(actions, mu, sigma, num_hands=2):
    per_dimension = 0.5 * ((actions - mu) / sigma) ** 2 + np.log(sigma) + 0.5 * math.log(2.0 * math.pi)
    return per_dimension.reshape(actions.shape[0], num_hands, -1).sum(-1)


def _ch_actor_advantages(raw_advantages):
    advantages = raw_advantages
    return (advantages - advantages.mean(0, keepdims=True)) / (
        advantages.std(0, ddof=1 if advantages.shape[0] > 1 else 0, keepdims=True) + 1e-8
    )


def _ch_clipped_actor_loss(old_hand_nll, hand_nll, advantages, clip):
    ratio = np.exp(old_hand_nll - hand_nll)
    surr1 = advantages * ratio
    surr2 = advantages * np.clip(ratio, 1.0 - clip, 1.0 + clip)
    per_group = np.maximum(-surr1, -surr2)
    return per_group.sum(-1).mean(), per_group, ratio


def _ch_default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch):
    value_pred_clipped = value_preds_batch + np.clip(values - value_preds_batch, -curr_e_clip, curr_e_clip)
    value_losses = (values - return_batch) ** 2
    value_losses_clipped = (value_pred_clipped - return_batch) ** 2
    return np.maximum(value_losses, value_losses_clipped)


# ------------------------------------------------------------------------------------------ (1) K = 1


@pytest.mark.parametrize(
    "ppo_overrides",
    [
        {},
        {"normalize_advantage_per_mini_batch": True, "use_clipped_value_loss": False, "log_ratio_clip": 5.0},
        {"schedule": "fixed", "clip_parameter": 0.05},
    ],
)
def test_k1_coordinated_update_reproduces_stock_update(ppo_overrides):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.algorithm import RolloutBatch, create_train_state, update
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig
    from flash_chord.training.ppo.coordinate.per_group import CoordinatedRolloutBatch, update_coordinated
    from flash_chord.training.ppo.network import ActorCritic

    steps, worlds, observation_dim, action_dim = 6, 8, 7, 43
    network = NetworkConfig(actor_hidden_dims=(16, 16), critic_hidden_dims=(16,), initial_noise_std=0.5,
                            actor_output_gain=0.01)
    config = PPOConfig(learning_epochs=3, mini_batches=4, **ppo_overrides)
    model = ActorCritic(action_dim=action_dim, config=network)
    rng = np.random.default_rng(0)
    observation = jnp.asarray(rng.standard_normal((steps, worlds, observation_dim)), dtype=jnp.float32)
    parameters = model.init(jax.random.PRNGKey(1), observation[0])
    behavior = _perturb(jax, parameters, 0.05, seed=2)
    flat = observation.reshape(steps * worlds, observation_dim)
    behavior_policy, behavior_value = model.apply(behavior, flat)
    action = behavior_policy.sample(seed=jax.random.PRNGKey(3))
    reward = jnp.asarray(rng.standard_normal((steps, worlds)), dtype=jnp.float32)
    done = jnp.asarray(rng.random((steps, worlds)) < 0.15, dtype=jnp.float32)
    stock_batch = RolloutBatch(
        observation=observation,
        action=action.reshape(steps, worlds, action_dim),
        log_probability=behavior_policy.log_prob(action).reshape(steps, worlds),
        value=behavior_value.reshape(steps, worlds),
        reward=reward,
        done=done,
        action_mean=behavior_policy.mean().reshape(steps, worlds, action_dim),
        action_std=behavior_policy.stddev().reshape(steps, worlds, action_dim),
    )
    coordinated_batch = CoordinatedRolloutBatch(
        observation=stock_batch.observation,
        action=stock_batch.action,
        log_probability=stock_batch.log_probability[..., None],
        value=stock_batch.value[..., None],
        reward=stock_batch.reward[..., None],
        done=stock_batch.done,
        action_mean=stock_batch.action_mean,
        action_std=stock_batch.action_std,
    )
    last_value = jnp.asarray(rng.standard_normal(worlds), dtype=jnp.float32)
    state = create_train_state(model, parameters, config)
    key = jax.random.PRNGKey(7)
    learning_rate = jnp.asarray(config.learning_rate, dtype=jnp.float32)

    stock_state, stock_rate, stock_metrics = update(state, stock_batch, last_value, key, learning_rate, config)
    state_k1, rate_k1, metrics_k1 = update_coordinated(
        state, coordinated_batch, last_value[:, None], key, learning_rate, config, (0,) * action_dim, 1
    )

    differences = [
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(jax.tree.leaves(stock_state.params), jax.tree.leaves(state_k1.params), strict=True)
    ]
    moved = max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(jax.tree.leaves(parameters), jax.tree.leaves(stock_state.params), strict=True)
    )
    assert moved > 1.0e-4
    assert max(differences) <= 1.0e-6
    assert int(stock_state.step) == int(state_k1.step) == config.learning_epochs * config.mini_batches
    np.testing.assert_allclose(float(rate_k1), float(stock_rate), rtol=1.0e-6)
    for name in ("policy_loss", "value_loss", "entropy", "kl", "clip_fraction", "explained_variance",
                 "skipped_update_fraction", "learning_rate"):
        np.testing.assert_allclose(
            float(getattr(metrics_k1, name)), float(getattr(stock_metrics, name)), rtol=1.0e-5, atol=1.0e-6,
            err_msg=name,
        )
    assert float(stock_metrics.clip_fraction) > 0.0
    for name in ("policy_loss", "value_loss", "clip_fraction", "explained_variance"):
        np.testing.assert_allclose(
            np.asarray(getattr(metrics_k1, f"{name}_per_group")), [float(getattr(metrics_k1, name))],
            rtol=1.0e-5, atol=1.0e-7,
        )


def test_stock_runner_is_bitwise_identical_to_stock_runtime(tmp_path):
    """The DEV runner with ``coordinate=None`` reproduces the official runtime bit for bit (CPU)."""
    _dependencies()
    if not (STOCK_SRC / "flash_chord" / "training" / "ppo" / "runner.py").is_file():
        pytest.skip("stock FlashCHORD runtime is not available")
    script = tmp_path / "run_stock_ppo.py"
    script.write_text(textwrap.dedent('''
        import json, sys
        from pathlib import Path
        import jax.numpy as jnp
        import flash_chord
        from flash_chord.training.environment import VectorStep
        from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
        from flash_chord.training.ppo.runner import run_training

        class Env:
            world_count = 4
            observation_dim = 5
            action_dim = 3
            def __init__(self):
                self.step_count = 0
            def reset(self):
                self.step_count = 0
                return jnp.sin(jnp.arange(20.0).reshape(4, 5))
            def step(self, action):
                self.step_count += 1
                s = self.step_count
                observation = jnp.sin(0.3 * s + jnp.arange(20.0).reshape(4, 5)) + 0.1 * action.sum(-1, keepdims=True)
                completed = s % 3 == 0
                return VectorStep(
                    observation=observation,
                    reward=1.0 - jnp.square(action).mean(axis=-1) + 0.05 * s,
                    terminated=jnp.asarray([completed, False, s % 4 == 0, False], dtype=jnp.int32),
                    truncated=jnp.asarray([False, completed, s % 4 == 0, s % 5 == 0], dtype=jnp.int32),
                    episode_return=jnp.full((4,), float(s)),
                    episode_length=jnp.asarray([3 * completed, 3 * completed, 4 * (s % 4 == 0), 0], dtype=jnp.int32),
                    episode_reference_progress=jnp.full((4,), 0.5),
                )

        class Logger:
            def __init__(self):
                self.records = []
            def log(self, metrics, step):
                self.records.append([step, {k: v for k, v in metrics.items() if not k.startswith("throughput/")}])
            def close(self):
                pass

        output = Path(sys.argv[1])
        logger = Logger()
        config = TrainingConfig(
            world_count=4, rollout_steps=3, max_iterations=3, save_interval=1, curriculum=None,
            network=NetworkConfig(actor_hidden_dims=(16, 8), critic_hidden_dims=(16, 8), initial_noise_std=0.5),
            ppo=PPOConfig(learning_epochs=2, mini_batches=2),
        )
        run_training(Env(), config, logger=logger, checkpoint_dir=output,
                     checkpoint_metadata={"resolved_config_json": "{}"})
        (output / "records.json").write_text(json.dumps({"file": flash_chord.__file__, "records": logger.records}))
    '''))
    results = {}
    for label, source in (("stock", STOCK_SRC), ("dev", DEV_SRC)):
        output = tmp_path / label
        output.mkdir()
        environment = dict(os.environ)
        environment.update(PYTHONPATH=str(source), JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, str(script), str(output)], env=environment, capture_output=True, text=True,
            timeout=600, cwd=str(tmp_path),
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        records = json.loads((output / "records.json").read_text())
        assert Path(records["file"]).resolve().is_relative_to(source.resolve())
        results[label] = (output, records["records"])

    from safetensors import safe_open

    assert results["stock"][1] == results["dev"][1]
    for iteration in (1, 2, 3):
        name = f"model_{iteration}.safetensors"
        with safe_open(str(results["stock"][0] / name), "numpy") as stock, safe_open(
            str(results["dev"][0] / name), "numpy"
        ) as dev:
            assert stock.metadata() == dev.metadata()
            assert sorted(stock.keys()) == sorted(dev.keys())
            for key in stock.keys():
                np.testing.assert_array_equal(stock.get_tensor(key), dev.get_tensor(key), err_msg=key)


# ------------------------------------------------------------------------------------------ (2) K = 2


def test_k2_equal_blocks_match_ch_losses():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.coordinate.per_group import (
        clipped_surrogate_per_group,
        group_matrix,
        normalize_per_column,
        per_group_log_prob,
        value_loss_per_column,
    )

    rng = np.random.default_rng(10)
    samples, per_hand, clip = 256, 14, 0.2
    action = rng.standard_normal((samples, 2 * per_hand))
    old_mean = rng.standard_normal((samples, 2 * per_hand)) * 0.3
    new_mean = old_mean + rng.standard_normal((samples, 2 * per_hand)) * 0.05
    old_std = np.exp(rng.standard_normal(2 * per_hand) * 0.2 - 0.7) * np.ones((samples, 1))
    new_std = old_std * np.exp(rng.standard_normal(2 * per_hand) * 0.05)
    groups = group_matrix((0,) * per_hand + (1,) * per_hand, 2)
    f32 = lambda value: jnp.asarray(value, dtype=jnp.float32)  # noqa: E731

    old_log_prob = np.asarray(per_group_log_prob(f32(action), f32(old_mean), f32(old_std), groups))
    new_log_prob = np.asarray(per_group_log_prob(f32(action), f32(new_mean), f32(new_std), groups))
    old_nll = _ch_per_hand_neglogp(action, old_mean, old_std)
    nll = _ch_per_hand_neglogp(action, new_mean, new_std)
    np.testing.assert_allclose(old_log_prob, -old_nll, rtol=1.0e-5, atol=1.0e-4)
    np.testing.assert_allclose(new_log_prob, -nll, rtol=1.0e-5, atol=1.0e-4)

    raw = rng.standard_normal((samples, 2)) * np.asarray([2.0, 0.5]) + np.asarray([0.3, -1.0])
    ours = np.asarray(normalize_per_column(f32(raw)))
    theirs = _ch_actor_advantages(raw)
    np.testing.assert_allclose(ours, theirs, rtol=1.0 / samples, atol=1.0e-5)
    np.testing.assert_allclose(ours, _np_normalize(raw, ddof=0), rtol=1.0e-5, atol=1.0e-5)

    # Same advantages and log-probabilities in both: the summed per-group clipped surrogate.
    surrogate, ratio = clipped_surrogate_per_group(f32(new_log_prob), f32(old_log_prob), f32(theirs), clip)
    ch_loss, ch_per_group, ch_ratio = _ch_clipped_actor_loss(-old_log_prob, -new_log_prob, theirs, clip)
    np.testing.assert_allclose(np.asarray(ratio), ch_ratio, rtol=1.0e-5)
    np.testing.assert_allclose(np.asarray(surrogate), ch_per_group, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(float(np.asarray(surrogate).sum(-1).mean()), ch_loss, rtol=1.0e-5)
    assert np.any(np.abs(ch_ratio - 1.0) > clip)

    old_value = rng.standard_normal((samples, 2))
    value = old_value + rng.standard_normal((samples, 2)) * 0.4
    target = rng.standard_normal((samples, 2))
    ours = np.asarray(value_loss_per_column(f32(value), f32(old_value), f32(target), clip, True))
    np.testing.assert_allclose(ours, _ch_default_critic_loss(old_value, value, clip, target), rtol=1.0e-5,
                               atol=1.0e-6)
    unclipped = np.asarray(value_loss_per_column(f32(value), f32(old_value), f32(target), clip, False))
    np.testing.assert_allclose(unclipped, (target - value) ** 2, rtol=1.0e-5, atol=1.0e-6)


# ------------------------------------------------------------------------------------------ (3) K = 3


def test_k3_per_group_functions_match_numpy_loops():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.coordinate.per_group import (
        compute_gae_k,
        explained_variance_per_column,
        group_matrix,
        normalize_per_column,
        per_group_log_prob,
    )

    layout = _layout()
    assert [layout.action_group.count(k) for k in range(3)] == [14, 14, 15]
    rng = np.random.default_rng(20)
    action = rng.standard_normal((5, 7, 43))
    mean = rng.standard_normal((5, 7, 43)) * 0.4
    std = np.exp(rng.standard_normal(43) * 0.3 - 0.5)
    ours = per_group_log_prob(jnp.asarray(action, jnp.float32), jnp.asarray(mean, jnp.float32),
                              jnp.asarray(std, jnp.float32), group_matrix(layout.action_group, 3))
    np.testing.assert_allclose(np.asarray(ours), _np_log_prob_per_group(action, mean, std, layout.action_group, 3),
                               rtol=1.0e-5, atol=1.0e-4)

    steps, worlds = 9, 6
    reward = rng.standard_normal((steps, worlds, 3))
    value = rng.standard_normal((steps, worlds, 3))
    done = (rng.random((steps, worlds)) < 0.25).astype(np.float64)
    last_value = rng.standard_normal((worlds, 3))
    advantage, target = compute_gae_k(*(jnp.asarray(x, jnp.float32) for x in (reward, value, done, last_value)),
                                      0.97, 0.9)
    expected_advantage, expected_target = _np_gae(reward, value, done, last_value, 0.97, 0.9)
    np.testing.assert_allclose(np.asarray(advantage), expected_advantage, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(np.asarray(target), expected_target, rtol=1.0e-5, atol=1.0e-5)
    for column in range(3):
        from flash_chord.training.ppo.algorithm import compute_gae

        single, _ = compute_gae(*(jnp.asarray(x, jnp.float32) for x in (reward[..., column], value[..., column],
                                                                          done, last_value[:, column])), 0.97, 0.9)
        np.testing.assert_array_equal(np.asarray(single), np.asarray(advantage[..., column]))

    flat = expected_advantage.reshape(-1, 3) * np.asarray([1.0, 10.0, 0.1])
    np.testing.assert_allclose(np.asarray(normalize_per_column(jnp.asarray(flat, jnp.float32))), _np_normalize(flat),
                               rtol=1.0e-5, atol=1.0e-5)
    prediction = expected_target.reshape(-1, 3) + rng.standard_normal((steps * worlds, 3)) * [0.1, 1.0, 3.0]
    explained = explained_variance_per_column(jnp.asarray(expected_target.reshape(-1, 3), jnp.float32),
                                              jnp.asarray(prediction, jnp.float32))
    reference = 1.0 - np.var(expected_target.reshape(-1, 3) - prediction, axis=0) / np.var(
        expected_target.reshape(-1, 3), axis=0)
    np.testing.assert_allclose(np.asarray(explained), reference, rtol=1.0e-4, atol=1.0e-5)


def _coordinated_model(layout, gain=0.0):
    from flash_chord.training.ppo.config import CoordinateConfig, NetworkConfig
    from flash_chord.training.ppo.coordinate.network import CoordinatedActorCritic

    return CoordinatedActorCritic(
        action_dim=43,
        layout=layout,
        coordinate=CoordinateConfig(),
        network=NetworkConfig(critic_hidden_dims=(32, 16), initial_noise_std=0.5, actor_output_gain=gain),
    )


@pytest.mark.parametrize("use_clipped_value_loss", [True, False])
def test_k3_update_loss_matches_numpy_reference(use_clipped_value_loss):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.algorithm import create_train_state
    from flash_chord.training.ppo.config import PPOConfig
    from flash_chord.training.ppo.coordinate.per_group import CoordinatedRolloutBatch, update_coordinated

    layout = _layout()
    model = _coordinated_model(layout, gain=0.5)
    steps, worlds = 4, 6
    rng = np.random.default_rng(30)
    observation = rng.standard_normal((steps, worlds, 486)).astype(np.float32)
    parameters = _perturb(jax, model.init(jax.random.PRNGKey(0), jnp.zeros((1, 486))), 0.05, seed=31)
    flat_observation = observation.reshape(-1, 486)
    with jax.default_matmul_precision("highest"):
        current_policy, current_value = model.apply(parameters, jnp.asarray(flat_observation))
    # A nearby behavior policy (ratios around the clip boundary) and stale values.
    current_mean = np.asarray(current_policy.mean(), np.float64)
    current_std = np.asarray(current_policy.stddev(), np.float64)
    behavior_mean = (current_mean + 0.02 * rng.standard_normal(current_mean.shape)).astype(np.float32)
    behavior_std = (current_std * np.exp(0.02 * rng.standard_normal(43))).astype(np.float32).astype(np.float64)
    behavior_mean = behavior_mean.astype(np.float64)
    action = (behavior_mean + behavior_std * rng.standard_normal(behavior_mean.shape)).astype(np.float32)
    action = action.astype(np.float64)
    old_log_prob = _np_log_prob_per_group(action, behavior_mean, behavior_std, layout.action_group, 3)
    old_value = np.asarray(current_value, np.float64) + 0.3 * rng.standard_normal((steps * worlds, 3))
    old_value = old_value.astype(np.float32).astype(np.float64)
    reward = rng.standard_normal((steps, worlds, 3))
    done = (rng.random((steps, worlds)) < 0.2).astype(np.float64)
    last_value = rng.standard_normal((worlds, 3))
    batch = CoordinatedRolloutBatch(
        observation=jnp.asarray(observation),
        action=jnp.asarray(action.reshape(steps, worlds, 43), jnp.float32),
        log_probability=jnp.asarray(old_log_prob.reshape(steps, worlds, 3), jnp.float32),
        value=jnp.asarray(old_value.reshape(steps, worlds, 3), jnp.float32),
        reward=jnp.asarray(reward, jnp.float32),
        done=jnp.asarray(done, jnp.float32),
        action_mean=jnp.asarray(behavior_mean.reshape(steps, worlds, 43), jnp.float32),
        action_std=jnp.asarray(behavior_std.reshape(steps, worlds, 43), jnp.float32),
    )
    config = PPOConfig(learning_epochs=1, mini_batches=1, schedule="fixed", clip_parameter=0.1,
                       entropy_coefficient=0.01, use_clipped_value_loss=use_clipped_value_loss)
    state = create_train_state(model, parameters, config)
    with jax.default_matmul_precision("highest"):
        new_state, _, metrics = update_coordinated(
            state, batch, jnp.asarray(last_value, jnp.float32), jax.random.PRNGKey(4),
            jnp.asarray(config.learning_rate), config, layout.action_group, 3,
        )

    # NumPy reference of every reported loss term at the pre-update parameters.
    advantage, target = _np_gae(reward, old_value.reshape(steps, worlds, 3), done, last_value, config.discount,
                                config.gae_lambda)
    advantage = advantage.reshape(-1, 3)
    target = target.reshape(-1, 3)
    raw_std = advantage.std(axis=0)
    advantage = _np_normalize(advantage)
    mean = np.asarray(current_policy.mean(), np.float64)
    std = np.asarray(current_policy.stddev(), np.float64)
    value = np.asarray(current_value, np.float64)
    new_log_prob = _np_log_prob_per_group(action, mean, std, layout.action_group, 3)
    ratio = np.exp(new_log_prob - old_log_prob)
    per_group = np.maximum(-advantage * ratio, -advantage * np.clip(ratio, 0.9, 1.1))
    policy_loss = per_group.sum(-1).mean()
    if use_clipped_value_loss:
        clipped = old_value + np.clip(value - old_value, -0.1, 0.1)
        value_losses = np.maximum((value - target) ** 2, (clipped - target) ** 2)
    else:
        value_losses = (value - target) ** 2
    entropy = np.sum(np.log(std) + 0.5 * math.log(2.0 * math.pi * math.e), axis=-1).mean()
    kl = np.sum(np.log(std / behavior_std) + (behavior_std**2 + (behavior_mean - mean) ** 2) / (2 * std**2) - 0.5,
                axis=-1).mean()
    explained = 1.0 - np.var(target - old_value, axis=0) / np.var(target, axis=0)
    std_per_group = np.asarray([behavior_std[:, np.asarray(layout.action_group) == k].mean() for k in range(3)])
    clipped_fraction = np.abs(ratio - 1.0) > 0.1

    assert 0.0 < clipped_fraction.mean() < 1.0
    np.testing.assert_allclose(float(metrics.policy_loss), policy_loss, rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(np.asarray(metrics.policy_loss_per_group), per_group.mean(0), rtol=1.0e-4,
                               atol=1.0e-5)
    np.testing.assert_allclose(float(metrics.value_loss), value_losses.mean(), rtol=1.0e-4)
    np.testing.assert_allclose(np.asarray(metrics.value_loss_per_group), value_losses.mean(0), rtol=1.0e-4)
    np.testing.assert_allclose(float(metrics.entropy), entropy, rtol=1.0e-5)
    np.testing.assert_allclose(float(metrics.kl), kl, rtol=1.0e-3, atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(metrics.clip_fraction_per_group), clipped_fraction.mean(0), atol=1.0e-6)
    np.testing.assert_allclose(float(metrics.clip_fraction), clipped_fraction.mean(), atol=1.0e-6)
    np.testing.assert_allclose(np.asarray(metrics.explained_variance_per_group), explained, rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(float(metrics.explained_variance), explained.mean(), rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(np.asarray(metrics.advantage_raw_std_per_group), raw_std, rtol=1.0e-4)
    np.testing.assert_allclose(np.asarray(metrics.action_std_mean_per_group), std_per_group, rtol=1.0e-5)
    assert float(metrics.skipped_update_fraction) == 0.0
    assert int(new_state.step) == 1


# ------------------------------------------------------------------- (3b) rollout bootstrap, (4) learner


class _CoordinatedEnv:
    world_count = 4
    observation_dim = 486
    action_dim = 43

    def __init__(self, jnp):
        self.jnp = jnp
        self.step_count = 0
        self.reset_count = 0
        self.rewards = []
        self.stages = []

    def _observation(self):
        base = self.jnp.arange(self.observation_dim, dtype=self.jnp.float32)
        worlds = self.jnp.arange(self.world_count, dtype=self.jnp.float32)[:, None]
        return self.jnp.sin(0.013 * base[None] * (self.step_count + 1) + 0.7 * worlds)

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        return self._observation()

    def step(self, action):
        from flash_chord.training.environment import VectorStep

        jnp = self.jnp
        self.step_count += 1
        step = self.step_count
        reward = 1.0 - jnp.square(action).mean(axis=-1) + 0.1 * step
        self.rewards.append(np.asarray(reward))
        terminated = jnp.asarray([step % 3 == 0, False, step % 4 == 0, False], dtype=jnp.int32)
        truncated = jnp.asarray([False, step % 2 == 0, step % 4 == 0, step % 5 == 0], dtype=jnp.int32)
        completed = jnp.maximum(terminated, truncated) > 0
        return VectorStep(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_return=jnp.where(completed, 2.0, 0.0),
            episode_length=jnp.where(completed, 3, 0).astype(jnp.int32),
            episode_reference_progress=jnp.where(completed, 0.5, 0.0),
        )

    def set_curriculum(self, stage):
        self.stages.append(stage)


def _coordinated_training_config(**overrides):
    from flash_chord.training.ppo.config import CoordinateConfig, NetworkConfig, PPOConfig, TrainingConfig

    values = dict(
        world_count=4,
        rollout_steps=3,
        max_iterations=2,
        save_interval=1,
        network=NetworkConfig(critic_hidden_dims=(32, 16), initial_noise_std=0.5),
        ppo=PPOConfig(learning_epochs=2, mini_batches=2),
        coordinate=CoordinateConfig(),
    )
    values.update(overrides)
    return TrainingConfig(**values)


def test_coordinated_rollout_bootstraps_each_value_column():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.coordinate.per_group import group_matrix
    from flash_chord.training.ppo.learner import create_coordinated_learner
    from flash_chord.training.ppo.rollout import collect_rollout_coordinated

    layout = _layout()
    config = _coordinated_training_config(rollout_steps=8)
    env = _CoordinatedEnv(jnp)
    learner = create_coordinated_learner(env, config, layout)
    params = _perturb(jax, learner.train_state.params, 0.05, seed=40)
    state = learner.train_state.replace(params=params)
    batch, _, _, _, metrics = collect_rollout_coordinated(
        env, state, learner.normalization, learner.observation, learner.key, config,
        group_matrix(layout.action_group, 3),
    )
    assert batch.log_probability.shape == (8, 4, 3)
    assert batch.value.shape == batch.reward.shape == (8, 4, 3)
    assert batch.done.shape == (8, 4)
    rewards = np.stack(env.rewards)
    steps = np.arange(1, 9)
    terminated = np.stack([steps % 3 == 0, np.zeros(8, bool), steps % 4 == 0, np.zeros(8, bool)], axis=1)
    truncated = np.stack([np.zeros(8, bool), steps % 2 == 0, steps % 4 == 0, steps % 5 == 0], axis=1)
    reference_end = (truncated & ~terminated).astype(np.float64)
    value = np.asarray(batch.value, np.float64)
    expected = rewards[..., None] + config.ppo.discount * value * reference_end[..., None]
    np.testing.assert_allclose(np.asarray(batch.reward), expected, rtol=1.0e-6, atol=1.0e-6)
    assert np.any(np.abs(value[..., 0] - value[..., 2]) > 1.0e-4)
    np.testing.assert_array_equal(np.asarray(batch.done), (terminated | truncated).astype(np.float32))
    log_prob = _np_log_prob_per_group(
        np.asarray(batch.action, np.float64), np.asarray(batch.action_mean, np.float64),
        np.asarray(batch.action_std, np.float64), layout.action_group, 3,
    )
    np.testing.assert_allclose(np.asarray(batch.log_probability), log_prob, rtol=1.0e-5, atol=1.0e-4)
    assert int(metrics.reference_end_count) == int(reference_end.sum())


class _Logger:
    def __init__(self):
        self.records = []
        self.closed = False

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def close(self):
        self.closed = True


def _reference_metadata(**extra):
    metadata = {
        "resolved_config_json": json.dumps(
            {"reference": {"object_articulations": {"version": 1, "entries": []}}}
        ),
    }
    metadata.update(extra)
    return metadata


def test_coordinated_learner_trains_saves_and_strictly_reloads(tmp_path):
    jax, jnp = _dependencies()
    from flash_chord.training.checkpoint import load_checkpoint, read_checkpoint_metadata
    from flash_chord.training.ppo.coordinate.layout import layout_digest
    from flash_chord.training.ppo.coordinate.per_group import CoordinatedPPOMetrics
    from flash_chord.training.ppo.learner import create_learner_template, train_iteration_coordinated
    from flash_chord.training.ppo.runner import run_training

    layout = _layout()
    config = _coordinated_training_config()
    metadata = _reference_metadata(policy_schema_json=json.dumps(_policy_schema()))
    logger = _Logger()
    env = _CoordinatedEnv(jnp)
    learner = run_training(env, config, logger=logger, checkpoint_dir=tmp_path, checkpoint_metadata=metadata)

    assert int(learner.iteration) == 2
    assert int(learner.train_state.step) == 2 * config.ppo.learning_epochs * config.ppo.mini_batches
    parameter_leaves = len(jax.tree.leaves(learner.train_state.params))
    assert parameter_leaves == 31
    assert len(jax.tree.leaves(learner)) == 3 * parameter_leaves + 9
    for step, record in logger.records:
        for agent in ("left", "right", "body"):
            for prefix in ("ppo/policy_loss", "ppo/value_loss", "ppo/explained_variance", "ppo/clip_fraction",
                           "ppo/advantage_raw_std", "policy/action_std_mean"):
                assert np.isfinite(record[f"{prefix}_{agent}"]), (prefix, agent)
        assert record["ppo/skipped_update_fraction_per_iteration"] == 0.0
        assert record["policy/action_std_mean_left"] == pytest.approx(0.5, rel=0.05)
        assert np.isfinite(tuple(record.values())).all()
    assert [step for step, _ in logger.records] == [12, 24]

    checkpoint = tmp_path / "model_2.safetensors"
    saved = read_checkpoint_metadata(checkpoint)
    assert saved["coordinate_layout_sha256"] == layout_digest(layout)
    assert saved["policy_schema_json"] == metadata["policy_schema_json"]
    template = create_learner_template(config, 486, 43, layout, world_count=4)
    restored, _ = load_checkpoint(checkpoint, template)
    for before, after in zip(jax.tree.leaves(learner), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
    stock_template = create_learner_template(replace(config, coordinate=None), 486, 43, world_count=4)
    with pytest.raises(ValueError, match="leaves"):
        load_checkpoint(checkpoint, stock_template)

    # The restored learner continues training with the coordinated metrics.
    next_learner, metrics = train_iteration_coordinated(_CoordinatedEnv(jnp), restored.replace(
        observation=jnp.asarray(env.reset())), config, layout)
    assert isinstance(metrics.optimization, CoordinatedPPOMetrics)
    assert metrics.optimization.value_loss_per_group.shape == (3,)
    assert int(next_learner.iteration) == 3

    # Resume through the runner: strict restore plus the layout digest check.
    resumed_logger = _Logger()
    resumed_env = _CoordinatedEnv(jnp)
    resumed = run_training(
        resumed_env,
        replace(config, max_iterations=3, resume=True, checkpoint=str(checkpoint)),
        logger=resumed_logger,
        checkpoint_metadata=metadata,
    )
    assert int(resumed.iteration) == 3
    assert resumed_env.reset_count == 2
    assert [step for step, _ in resumed_logger.records] == [36]

    drop_config = replace(
        config, max_iterations=3, resume=True, checkpoint=str(checkpoint),
        coordinate=replace(config.coordinate, hand_world_terms="drop"),
    )
    with pytest.raises(ValueError, match="coordinated layout does not match"):
        run_training(_CoordinatedEnv(jnp), drop_config, checkpoint_metadata=metadata)
    with pytest.raises(ValueError, match="policy_schema_json"):
        run_training(_CoordinatedEnv(jnp), config, checkpoint_metadata=_reference_metadata())
    learner = run_training(_CoordinatedEnv(jnp), replace(config, max_iterations=1), coordinate_layout=layout)
    assert int(learner.iteration) == 1


def test_create_learner_template_matches_stock_create_learner():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.learner import create_coordinated_learner, create_learner, create_learner_template

    class _Env:
        world_count = 2
        observation_dim = 3
        action_dim = 2

        def reset(self):
            return jnp.zeros((2, 3))

    config = TrainingConfig(world_count=2, rollout_steps=2, network=NetworkConfig(actor_hidden_dims=(8,),
                            critic_hidden_dims=(8,)), ppo=PPOConfig(mini_batches=2))
    stock = create_learner(_Env(), config)
    template = create_learner_template(config, 3, 2, world_count=2)
    assert jax.tree.structure(stock.train_state.params) == jax.tree.structure(template.train_state.params)
    assert len(jax.tree.leaves(stock)) == len(jax.tree.leaves(template))
    for a, b in zip(jax.tree.leaves(stock), jax.tree.leaves(template), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert len(jax.tree.leaves(create_learner_template(_stock_g1_config(), 486, 43))) == 72

    layout = _layout()
    coordinated_config = _coordinated_training_config()
    env = _CoordinatedEnv(jnp)
    coordinated = create_coordinated_learner(env, coordinated_config, layout)
    coordinated_template = create_learner_template(coordinated_config, 486, 43, layout, world_count=4)
    for a, b in zip(jax.tree.leaves(coordinated.train_state), jax.tree.leaves(coordinated_template.train_state),
                    strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="requires training.coordinate"):
        create_learner_template(replace(coordinated_config, coordinate=None), 486, 43, layout)


def _stock_g1_config():
    from flash_chord.training.ppo.config import NetworkConfig, TrainingConfig

    return TrainingConfig(world_count=4096, network=NetworkConfig(initial_noise_std=0.5))


class _FakeRSI:
    def __init__(self):
        self.calls = []

    def host_update(self, iteration):
        self.calls.append(("host_update", iteration))
        return {"rsi/top_mean": 1.0 - 0.25 * iteration, "rsi/decisions": float(iteration)}

    def save_state(self, path):
        Path(path).write_bytes(b"rsi")
        self.calls.append(("save", Path(path).name))

    def load_state(self, path):
        assert Path(path).is_file()
        self.calls.append(("load", Path(path).name))


def test_runner_rsi_call_sites_for_stock_and_coordinated_learners(tmp_path):
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig
    from flash_chord.training.ppo.runner import rsi_state_path, run_training

    assert rsi_state_path("/a/model_500.safetensors") == Path("/a/rsi_500.safetensors")
    with pytest.raises(ValueError):
        rsi_state_path("/a/latest.safetensors")

    stock_config = TrainingConfig(
        world_count=4, rollout_steps=3, max_iterations=2, save_interval=1, curriculum=None,
        network=NetworkConfig(actor_hidden_dims=(8,), critic_hidden_dims=(8,)), ppo=PPOConfig(mini_batches=2),
    )
    for label, config, metadata in (
        ("stock", stock_config, _reference_metadata()),
        ("coordinated", _coordinated_training_config(),
         _reference_metadata(policy_schema_json=json.dumps(_policy_schema()))),
    ):
        directory = tmp_path / label
        env = _CoordinatedEnv(jnp) if label == "coordinated" else _SmallEnv(jnp)
        env.rsi = _FakeRSI()
        logger = _Logger()
        run_training(env, config, logger=logger, checkpoint_dir=directory, checkpoint_metadata=metadata)
        assert env.rsi.calls == [
            ("host_update", 0), ("save", "rsi_1.safetensors"), ("host_update", 1), ("save", "rsi_2.safetensors"),
        ]
        assert [record["rsi/top_mean"] for _, record in logger.records] == [1.0, 0.75]
        assert (directory / "rsi_2.safetensors").is_file() and (directory / "model_2.safetensors").is_file()

        resumed_env = _CoordinatedEnv(jnp) if label == "coordinated" else _SmallEnv(jnp)
        resumed_env.rsi = _FakeRSI()
        resume = replace(config, max_iterations=3, resume=True, checkpoint=str(directory / "model_2.safetensors"))
        run_training(resumed_env, resume, checkpoint_dir=directory, checkpoint_metadata=metadata)
        assert resumed_env.rsi.calls == [("load", "rsi_2.safetensors"), ("host_update", 2),
                                         ("save", "rsi_3.safetensors")]
        (directory / "rsi_3.safetensors").unlink()
        missing = replace(config, max_iterations=4, resume=True, checkpoint=str(directory / "model_3.safetensors"))
        missing_env = _CoordinatedEnv(jnp) if label == "coordinated" else _SmallEnv(jnp)
        missing_env.rsi = _FakeRSI()
        with pytest.raises(FileNotFoundError, match="rsi_3"):
            run_training(missing_env, missing, checkpoint_metadata=metadata)


class _SmallEnv(_CoordinatedEnv):
    observation_dim = 5
    action_dim = 3
