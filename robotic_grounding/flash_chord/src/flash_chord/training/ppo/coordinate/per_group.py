# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-agent (W0-P) PPO math: K-column values, per-group ratios and per-column advantages.

The update mirrors :func:`flash_chord.training.ppo.algorithm.update` (same epochs, mini-batch shuffle,
optimizer, adaptive learning rate and non-finite guard) with these differences:

* log-probabilities are summed per agent group, giving one PPO ratio per agent;
* GAE runs per value column with the shared ``done`` mask;
* advantages are normalized per column (population std, as the stock update);
* the policy loss is ``mean_S sum_k max(-A_k r_k, -A_k clip(r_k))`` (CH ``clipped_actor_loss``);
* the clipped value loss is averaged over samples and columns (coefficient as configured);
* entropy and KL use the joint diagonal Gaussian exactly as the stock update.

With one group covering every action and a single value column it reduces to the stock update.
"""

from __future__ import annotations

import math
from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from flash_chord.training.ppo.algorithm import (
    _tree_all_finite,
    adapt_learning_rate,
    compute_gae,
    gaussian_kl,
    probability_ratio,
)
from flash_chord.training.ppo.config import PPOConfig

_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


@flax.struct.dataclass
class CoordinatedRolloutBatch:
    """Time-major on-policy data with per-group log-probabilities and K value columns."""

    observation: jax.Array
    action: jax.Array
    log_probability: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    action_mean: jax.Array
    action_std: jax.Array


@flax.struct.dataclass
class CoordinatedPPOMetrics:
    """Stock :class:`PPOMetrics` fields plus per-agent diagnostics (each ``(K,)``)."""

    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    kl: jax.Array
    clip_fraction: jax.Array
    learning_rate: jax.Array
    explained_variance: jax.Array
    skipped_update_fraction: jax.Array
    policy_loss_per_group: jax.Array
    value_loss_per_group: jax.Array
    explained_variance_per_group: jax.Array
    clip_fraction_per_group: jax.Array
    advantage_raw_std_per_group: jax.Array
    action_std_mean_per_group: jax.Array


def group_matrix(action_group: tuple[int, ...], num_groups: int) -> jax.Array:
    """One-hot ``(action_dim, K)`` float32 membership matrix."""
    groups = jnp.asarray(action_group, dtype=jnp.int32)
    if len(action_group) == 0 or min(action_group) < 0 or max(action_group) >= num_groups:
        raise ValueError(f"action groups must lie in [0, {num_groups}), got {action_group}")
    return jax.nn.one_hot(groups, num_groups, dtype=jnp.float32)


def per_group_log_prob(action: jax.Array, mean: jax.Array, std: jax.Array, group_matrix: jax.Array) -> jax.Array:
    """Diagonal-Gaussian log-density summed within each action group, shape ``(..., K)``.

    The arithmetic follows ``distrax.MultivariateNormalDiag.log_prob`` operation by operation
    (standardize with the reciprocal scale, ``-0.5 z^2 - log(2 pi)/2`` summed, plus the summed inverse
    log-determinant ``-log|std|``), so one all-covering group reproduces the stock log-probability; the
    per-group reduction is an elementwise mask-and-sum rather than a matmul, which a GPU may run in TF32.
    """
    standardized = (1.0 / std) * (action - mean)
    log_unnormalized = -0.5 * jnp.square(standardized) - _LOG_SQRT_TWO_PI
    inverse_log_det = -jnp.log(jnp.abs(std))
    return jnp.sum(log_unnormalized[..., :, None] * group_matrix, axis=-2) + jnp.sum(
        inverse_log_det[..., :, None] * group_matrix, axis=-2
    )


def compute_gae_k(
    reward: jax.Array,
    value: jax.Array,
    done: jax.Array,
    last_value: jax.Array,
    discount: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Per-column GAE for ``(T, N, K)`` rewards/values with a shared ``(T, N)`` done mask."""
    return compute_gae(reward, value, done[..., None], last_value, discount, gae_lambda)


def normalize_per_column(advantage: jax.Array) -> jax.Array:
    """``(A - mean_k) / (std_k + 1e-8)`` per column of ``(S, K)`` with population std (ddof=0)."""
    return (advantage - advantage.mean(axis=0)) / (advantage.std(axis=0) + 1.0e-8)


def explained_variance_per_column(value_target: jax.Array, value_prediction: jax.Array) -> jax.Array:
    """Stock explained variance evaluated separately for each value column, shape ``(K,)``."""
    target = value_target.reshape(-1, value_target.shape[-1])
    prediction = value_prediction.reshape(-1, value_prediction.shape[-1])
    target_variance = jnp.var(target, axis=0)
    return jnp.where(
        target_variance > 1.0e-8,
        1.0 - jnp.var(target - prediction, axis=0) / target_variance,
        0.0,
    )


def clipped_surrogate_per_group(
    new_log_probability: jax.Array,
    old_log_probability: jax.Array,
    advantage: jax.Array,
    clip_parameter: float,
    log_ratio_clip: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Per-sample, per-group PPO loss ``max(-A r, -A clip(r))`` and the ratios, both ``(S, K)``."""
    ratio = probability_ratio(new_log_probability, old_log_probability, log_ratio_clip)
    unclipped = ratio * advantage
    clipped = jnp.clip(ratio, 1.0 - clip_parameter, 1.0 + clip_parameter) * advantage
    return -jnp.minimum(unclipped, clipped), ratio


def value_loss_per_column(
    value: jax.Array,
    old_value: jax.Array,
    target: jax.Array,
    clip_parameter: float,
    use_clipped_value_loss: bool,
) -> jax.Array:
    """Per-sample, per-column value loss (stock clipped form), shape ``(S, K)``."""
    if use_clipped_value_loss:
        clipped_value = old_value + jnp.clip(value - old_value, -clip_parameter, clip_parameter)
        return jnp.maximum(jnp.square(value - target), jnp.square(clipped_value - target))
    return jnp.square(value - target)


@partial(jax.jit, static_argnames=("config", "action_group", "num_groups"))
def update_coordinated(
    state: TrainState,
    batch: CoordinatedRolloutBatch,
    last_value: jax.Array,
    key: jax.Array,
    learning_rate: jax.Array,
    config: PPOConfig,
    action_group: tuple[int, ...],
    num_groups: int,
) -> tuple[TrainState, jax.Array, CoordinatedPPOMetrics]:
    """Run all configured PPO epochs and mini-batches with per-agent ratios and value columns."""
    groups = group_matrix(action_group, num_groups)
    advantage, value_target = compute_gae_k(
        batch.reward,
        batch.value,
        batch.done,
        last_value,
        config.discount,
        config.gae_lambda,
    )
    sample_count = batch.reward.shape[0] * batch.reward.shape[1]
    advantage = advantage.reshape(sample_count, num_groups)
    value_target = value_target.reshape(sample_count, num_groups)
    old_values = batch.value.reshape(sample_count, num_groups)
    explained_per_group = explained_variance_per_column(value_target, old_values)
    advantage_raw_std = advantage.std(axis=0)
    if not config.normalize_advantage_per_mini_batch:
        advantage = normalize_per_column(advantage)
    action_std_per_group = (
        jnp.sum(batch.action_std.reshape(sample_count, -1).mean(axis=0)[:, None] * groups, axis=0)
        / groups.sum(axis=0)
    )

    data = (
        batch.observation.reshape(sample_count, -1),
        batch.action.reshape(sample_count, -1),
        batch.log_probability.reshape(sample_count, num_groups),
        old_values,
        value_target,
        advantage,
        batch.action_mean.reshape(sample_count, -1),
        batch.action_std.reshape(sample_count, -1),
    )

    def mini_batch(carry, mini_batch_data):
        current_state, current_learning_rate = carry
        observation, action, old_log_probability, old_value, target, mini_advantage, old_mean, old_std = mini_batch_data
        if config.normalize_advantage_per_mini_batch:
            mini_advantage = normalize_per_column(mini_advantage)

        def loss(parameters):
            policy, value = current_state.apply_fn(parameters, observation)
            value = value.reshape(value.shape[0], -1)
            new_log_probability = per_group_log_prob(action, policy.mean(), policy.stddev(), groups)
            surrogate, ratio = clipped_surrogate_per_group(
                new_log_probability,
                old_log_probability,
                mini_advantage,
                config.clip_parameter,
                config.log_ratio_clip,
            )
            policy_loss_per_group = surrogate.mean(axis=0)
            policy_loss = surrogate.sum(axis=-1).mean()
            clipped = (jnp.abs(ratio - 1.0) > config.clip_parameter).astype(jnp.float32)
            clip_fraction_per_group = clipped.mean(axis=0)
            clip_fraction = clipped.mean()
            value_losses = value_loss_per_column(
                value,
                old_value,
                target,
                config.clip_parameter,
                config.use_clipped_value_loss,
            )
            value_loss_per_group = value_losses.mean(axis=0)
            value_loss = value_losses.mean()
            entropy = policy.entropy().mean()
            kl = gaussian_kl(old_mean, old_std, policy.mean(), policy.stddev())
            total = policy_loss + config.value_loss_coefficient * value_loss - config.entropy_coefficient * entropy
            return total, (
                policy_loss,
                value_loss,
                entropy,
                kl,
                clip_fraction,
                policy_loss_per_group,
                value_loss_per_group,
                clip_fraction_per_group,
            )

        (_, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(current_state.params)
        (
            policy_loss,
            value_loss,
            entropy,
            kl,
            clip_fraction,
            policy_loss_per_group,
            value_loss_per_group,
            clip_fraction_per_group,
        ) = metrics
        candidate_learning_rate = adapt_learning_rate(current_learning_rate, kl, config)
        updates, optimizer_state = current_state.tx.update(
            gradients,
            current_state.opt_state,
            current_state.params,
        )
        update_is_finite = _tree_all_finite(updates)
        updates = jax.tree.map(lambda value: -candidate_learning_rate * value, updates)
        parameters = optax.apply_updates(current_state.params, updates)
        candidate_state = current_state.replace(
            step=current_state.step + 1,
            params=parameters,
            opt_state=optimizer_state,
        )
        current_state = jax.lax.cond(
            update_is_finite,
            lambda: candidate_state,
            lambda: current_state,
        )
        current_learning_rate = jnp.where(update_is_finite, candidate_learning_rate, current_learning_rate)
        output = CoordinatedPPOMetrics(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            kl=kl,
            clip_fraction=clip_fraction,
            learning_rate=current_learning_rate,
            explained_variance=explained_per_group.mean(),
            skipped_update_fraction=1.0 - update_is_finite.astype(jnp.float32),
            policy_loss_per_group=policy_loss_per_group,
            value_loss_per_group=value_loss_per_group,
            explained_variance_per_group=explained_per_group,
            clip_fraction_per_group=clip_fraction_per_group,
            advantage_raw_std_per_group=advantage_raw_std,
            action_std_mean_per_group=action_std_per_group,
        )
        return (current_state, current_learning_rate), output

    def epoch(carry, _):
        current_state, current_learning_rate, current_key = carry
        current_key, permutation_key = jax.random.split(current_key)
        permutation = jax.random.permutation(permutation_key, sample_count)
        shuffled = jax.tree.map(lambda value: value[permutation], data)
        mini_batches = jax.tree.map(
            lambda value: value.reshape((config.mini_batches, -1) + value.shape[1:]),
            shuffled,
        )
        (current_state, current_learning_rate), metrics = jax.lax.scan(
            mini_batch,
            (current_state, current_learning_rate),
            mini_batches,
        )
        return (current_state, current_learning_rate, current_key), metrics

    (state, learning_rate, key), metrics = jax.lax.scan(
        epoch,
        (state, learning_rate, key),
        None,
        length=config.learning_epochs,
    )
    metrics = jax.tree.map(lambda value: value.mean(axis=(0, 1)), metrics)
    metrics = metrics.replace(learning_rate=learning_rate)
    return state, learning_rate, metrics
