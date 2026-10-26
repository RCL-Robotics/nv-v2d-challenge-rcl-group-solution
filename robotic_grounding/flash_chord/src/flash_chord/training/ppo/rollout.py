# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-device rollout collection across the Warp/JAX boundary."""

from __future__ import annotations

from typing import NamedTuple

import flax
import jax
import jax.numpy as jnp

from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.environment import JAXVectorEnv
from flash_chord.training.ppo.normalization import (
    NormalizationState,
    guard_normalization_update,
    normalize_observation,
    update_normalization,
)
from flash_chord.training.ppo.algorithm import RolloutBatch


@flax.struct.dataclass
class RolloutMetrics:
    """Device-resident aggregate rollout metrics."""

    mean_step_reward: jax.Array
    reference_end_count: jax.Array
    episode_return_sum: jax.Array
    episode_length_sum: jax.Array
    episode_reference_progress_sum: jax.Array
    episode_count: jax.Array
    action_mean_abs_mean: jax.Array
    action_std_mean: jax.Array
    objective_term_mean: jax.Array | None = None
    termination_cause_count: jax.Array | None = None
    tracking_error_mean: jax.Array | None = None
    # Non-finite guard (TrainingConfig.nonfinite_guard): world-steps sanitized in this rollout, and the subset
    # the environment terminated because its simulator state went non-finite (a separate outcome class).
    nonfinite_world_count: jax.Array | None = None
    nonfinite_terminated_count: jax.Array | None = None


class GuardedStep(NamedTuple):
    """Outputs of :func:`guard_step` for one environment step (``None`` where the matching input is ``None``)."""

    observation: jax.Array  # next observation with non-finite rows zeroed
    valid: jax.Array  # rows that may enter the observation normalizer (not bad)
    reward: jax.Array  # PPO reward after the stock bootstrap; bad rows replaced by their own value
    done: jax.Array  # bad rows end the episode
    raw_reward: jax.Array  # environment reward with bad rows zeroed (its mean is the logged step reward)
    episode_return: jax.Array | None  # completed-episode returns with non-finite entries zeroed
    objective_terms: jax.Array | None  # diagnostic rows of bad worlds zeroed
    tracking_errors: jax.Array | None  # diagnostic rows of bad worlds zeroed
    terminated_unflagged: jax.Array  # terminations left for the tracking-cause classes (guard ones excluded)
    nonfinite_count: jax.Array  # bad world-steps
    nonfinite_terminated_count: jax.Array | None  # environment-flagged terminations (None without a flag)


@jax.jit
def guard_step(
    next_observation: jax.Array,
    reward: jax.Array,
    flag: jax.Array | None,
    terminated: jax.Array,
    value: jax.Array,
    bootstrapped_reward: jax.Array,
    done: jax.Array,
    episode_return: jax.Array | None,
    objective_terms: jax.Array | None,
    tracking_errors: jax.Array | None,
) -> GuardedStep:
    """Every per-step selection of the non-finite guard (``TrainingConfig.nonfinite_guard``) in one executable.

    A world-step is bad when its next observation or environment ``reward`` is non-finite or the environment's
    guard ``flag`` is set. A bad step bootstraps from its own value (zero TD error) and ends the episode; its
    observation row is zeroed and kept out of the normalizer; its diagnostics are zeroed.

    Only elementwise selections and integer counts run here. The stock reward arithmetic (which yields
    ``bootstrapped_reward``), every floating-point reduction and the normalizer update stay separate
    executables, exactly as without the guard, so an all-finite step is bit-identical to the unguarded rollout
    and every output is bit-identical to the unfused guard of 262035e (which issued ~12 eager dispatches per
    step and cost 3-7% throughput). Every output is a fresh buffer, so zero-copy Warp views may be passed in.
    """
    observation_finite = jnp.all(jnp.isfinite(next_observation), axis=-1)
    bad = jnp.logical_not(observation_finite) | jnp.logical_not(jnp.isfinite(reward))
    terminated_unflagged = terminated.astype(bool)
    nonfinite_terminated_count = None
    if flag is not None:
        bad = bad | (flag != 0)
        nonfinite_terminated_count = jnp.logical_and(terminated != 0, flag != 0).sum()
        terminated_unflagged = jnp.logical_and(terminated_unflagged, flag == 0)

    def rows(values: jax.Array) -> jax.Array:
        return bad.reshape(bad.shape + (1,) * (values.ndim - 1))

    def zero_rows(values: jax.Array | None) -> jax.Array | None:
        return None if values is None else jnp.where(rows(values), jnp.zeros_like(values), values)

    return GuardedStep(
        observation=jnp.where(observation_finite[:, None], next_observation, jnp.zeros_like(next_observation)),
        valid=jnp.logical_not(bad),
        reward=jnp.where(rows(bootstrapped_reward), value, bootstrapped_reward),
        done=jnp.where(bad, jnp.ones_like(done), done),
        raw_reward=zero_rows(reward),
        episode_return=(
            None if episode_return is None else jnp.where(jnp.isfinite(episode_return), episode_return, 0.0)
        ),
        objective_terms=zero_rows(objective_terms),
        tracking_errors=zero_rows(tracking_errors),
        terminated_unflagged=terminated_unflagged,
        nonfinite_count=bad.sum(),
        nonfinite_terminated_count=nonfinite_terminated_count,
    )


class TransitionCopy(NamedTuple):
    """Outputs of :func:`copy_transition`: fresh copies of one step's zero-copy views and its exact episode flags."""

    observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array  # float32 max(terminated, truncated)
    reference_end: jax.Array  # bool: truncated and not terminated


@jax.jit
def copy_transition(observation, reward, terminated, truncated) -> TransitionCopy:
    """The per-step copies and episode flags of the rollout in one executable instead of ~10 eager dispatches.

    Only copies, an integer max, an exact integer-to-float cast and boolean logic run here, so every output is
    bit-identical to the eager sequence it replaces. Every output is a fresh buffer (``jnp.array(..., copy=True)``
    is a copy, never a forwarded input), so the zero-copy Warp views may be passed in and reused by the next step.
    """
    terminated = jnp.array(terminated, copy=True)
    truncated = jnp.array(truncated, copy=True)
    return TransitionCopy(
        observation=jnp.array(observation, copy=True),
        reward=jnp.array(reward, copy=True),
        terminated=terminated,
        truncated=truncated,
        done=jnp.maximum(terminated, truncated).astype(jnp.float32),
        reference_end=jnp.logical_and(truncated.astype(bool), jnp.logical_not(terminated.astype(bool))),
    )


class StepMetrics(NamedTuple):
    """Outputs of :func:`step_metrics`: one step's logged aggregates (``None`` where the input is ``None``)."""

    raw_reward_mean: jax.Array
    reference_end_count: jax.Array
    episode_return_sum: jax.Array
    episode_length_sum: jax.Array
    episode_reference_progress_sum: jax.Array
    episode_count: jax.Array
    objective_terms: jax.Array | None  # fresh copy
    termination_causes: jax.Array | None  # exclusive causes (first matching cause wins), bool [worlds, causes]
    tracking_errors: jax.Array | None  # fresh copy


@jax.jit
def step_metrics(
    raw_reward: jax.Array,
    reference_end: jax.Array,
    episode_return: jax.Array,
    episode_length: jax.Array,
    episode_reference_progress: jax.Array,
    objective_terms: jax.Array | None,
    termination_causes: jax.Array | None,
    terminated_remaining: jax.Array,
    tracking_errors: jax.Array | None,
) -> StepMetrics:
    """The rollout's per-step metric reductions and copies in one executable instead of ~20 eager dispatches.

    These values feed only the logged :class:`RolloutMetrics`, never the PPO batch. The boolean and integer
    results are exact; the float sums and the mean are the same reductions as the eager code (their last bit may
    differ if XLA fuses them differently). ``termination_causes`` are made exclusive in cause order, starting from
    the ``terminated_remaining`` worlds (cast to bool here), as in the eager loop.
    """
    completed = episode_length > 0
    exclusive = None
    if termination_causes is not None:
        remaining = terminated_remaining.astype(bool)
        selected_causes = []
        for cause in jnp.moveaxis(termination_causes, -1, 0):
            selected = jnp.logical_and(remaining, cause.astype(bool))
            selected_causes.append(selected)
            remaining = jnp.logical_and(remaining, jnp.logical_not(selected))
        exclusive = jnp.stack(selected_causes, axis=-1)
    return StepMetrics(
        raw_reward_mean=raw_reward.mean(),
        reference_end_count=reference_end.sum(),
        episode_return_sum=jnp.where(completed, episode_return, 0.0).sum(),
        episode_length_sum=jnp.where(completed, episode_length, 0).sum(),
        episode_reference_progress_sum=jnp.where(completed, episode_reference_progress, 0.0).sum(),
        episode_count=completed.sum(),
        objective_terms=None if objective_terms is None else jnp.array(objective_terms, copy=True),
        termination_causes=exclusive,
        tracking_errors=None if tracking_errors is None else jnp.array(tracking_errors, copy=True),
    )


def _append_step_metrics(
    transition,
    guarded: GuardedStep | None,
    copied: TransitionCopy,
    guard: bool,
    raw_rewards: list,
    reference_end_counts: list,
    episode_returns: list,
    episode_lengths: list,
    episode_reference_progress: list,
    episode_counts: list,
    objective_terms: list,
    termination_causes: list,
    tracking_errors: list,
) -> None:
    """Append one step's logged aggregates through :func:`step_metrics` (same inputs as the former eager code).

    Guard terminations form their own outcome class (``RolloutMetrics.nonfinite_terminated_count``), so with the
    guard the exclusive causes start from ``guarded.terminated_unflagged``.
    """
    metrics = step_metrics(
        guarded.raw_reward if guard else transition.reward,
        copied.reference_end,
        guarded.episode_return if guard else transition.episode_return,
        transition.episode_length,
        transition.episode_reference_progress,
        None if transition.objective_terms is None else (guarded.objective_terms if guard else transition.objective_terms),
        transition.termination_causes,
        guarded.terminated_unflagged if guard else copied.terminated,
        None if transition.tracking_errors is None else (guarded.tracking_errors if guard else transition.tracking_errors),
    )
    raw_rewards.append(metrics.raw_reward_mean)
    reference_end_counts.append(metrics.reference_end_count)
    episode_returns.append(metrics.episode_return_sum)
    episode_lengths.append(metrics.episode_length_sum)
    episode_reference_progress.append(metrics.episode_reference_progress_sum)
    episode_counts.append(metrics.episode_count)
    if metrics.objective_terms is not None:
        objective_terms.append(metrics.objective_terms)
    if metrics.termination_causes is not None:
        termination_causes.append(metrics.termination_causes)
    if metrics.tracking_errors is not None:
        tracking_errors.append(metrics.tracking_errors)


def _guard_transition(transition, next_observation, terminated, value, bootstrapped_reward, done) -> GuardedStep:
    """:func:`guard_step` on one :class:`VectorStep` (its zero-copy views are read before the next step)."""
    return guard_step(
        next_observation,
        transition.reward,
        transition.nonfinite,
        terminated,
        value,
        bootstrapped_reward,
        done,
        transition.episode_return,
        transition.objective_terms,
        transition.tracking_errors,
    )


@jax.jit
def sample_action(state, observation: jax.Array, key: jax.Array):
    """Sample one action batch and preserve its behavior-policy statistics."""
    policy, value = state.apply_fn(state.params, observation)
    action = policy.sample(seed=key)
    return action, policy.log_prob(action), value, policy.mean(), policy.stddev()


def collect_rollout(
    env: JAXVectorEnv,
    state,
    normalization: NormalizationState,
    observation: jax.Array,
    key: jax.Array,
    config: TrainingConfig,
) -> tuple[RolloutBatch, jax.Array, NormalizationState, jax.Array, RolloutMetrics]:
    """Collect one time-major on-policy rollout without host array transfers."""
    observations = []
    actions = []
    log_probabilities = []
    values = []
    rewards = []
    dones = []
    action_means = []
    action_stds = []
    raw_rewards = []
    reference_end_counts = []
    episode_returns = []
    episode_lengths = []
    episode_reference_progress = []
    episode_counts = []
    objective_terms = []
    termination_causes = []
    tracking_errors = []
    guard = config.nonfinite_guard
    nonfinite_world_counts = []
    nonfinite_terminated_counts = []

    for _ in range(config.rollout_steps):
        # ``observation`` is a fresh JAX array (the learner's copy or the previous step's copy); only the stored raw
        # observation of an unnormalized policy needs its own buffer.
        if config.observation_normalization:
            policy_observation = normalize_observation(
                normalization,
                observation,
                config.observation_normalization_epsilon,
            )
        else:
            policy_observation = jnp.array(observation, copy=True)
        key, action_key = jax.random.split(key)
        action, log_probability, value, action_mean, action_std = sample_action(
            state,
            policy_observation,
            action_key,
        )
        transition = env.step(action)
        copied = copy_transition(transition.observation, transition.reward, transition.terminated, transition.truncated)
        next_observation, reward, terminated, done = copied.observation, copied.reward, copied.terminated, copied.done
        reference_end = copied.reference_end
        reward = reward + config.ppo.discount * value * reference_end.astype(jnp.float32)
        guarded = None
        if guard:
            # A bad world-step bootstraps from its own value (zero TD error, zero advantage) and ends the episode.
            guarded = _guard_transition(transition, next_observation, terminated, value, reward, done)
            next_observation, reward, done = guarded.observation, guarded.reward, guarded.done
            nonfinite_world_counts.append(guarded.nonfinite_count)
            if guarded.nonfinite_terminated_count is not None:
                nonfinite_terminated_counts.append(guarded.nonfinite_terminated_count)

        observations.append(policy_observation)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        action_means.append(action_mean)
        action_stds.append(action_std)
        _append_step_metrics(
            transition, guarded, copied, guard,
            raw_rewards, reference_end_counts, episode_returns, episode_lengths, episode_reference_progress,
            episode_counts, objective_terms, termination_causes, tracking_errors,
        )

        if config.observation_normalization:
            if guard:
                candidate = update_normalization(normalization, next_observation)
                normalization = guard_normalization_update(normalization, candidate, next_observation, guarded.valid)
            else:
                normalization = update_normalization(normalization, next_observation)
        observation = next_observation

    batch = RolloutBatch(
        observation=jnp.stack(observations),
        action=jnp.stack(actions),
        log_probability=jnp.stack(log_probabilities),
        value=jnp.stack(values),
        reward=jnp.stack(rewards),
        done=jnp.stack(dones),
        action_mean=jnp.stack(action_means),
        action_std=jnp.stack(action_stds),
    )
    stacked_action_means = jnp.stack(action_means)
    stacked_action_stds = jnp.stack(action_stds)
    metrics = RolloutMetrics(
        mean_step_reward=jnp.stack(raw_rewards).mean(),
        reference_end_count=jnp.stack(reference_end_counts).sum(),
        episode_return_sum=jnp.stack(episode_returns).sum(),
        episode_length_sum=jnp.stack(episode_lengths).sum(),
        episode_reference_progress_sum=jnp.stack(episode_reference_progress).sum(),
        episode_count=jnp.stack(episode_counts).sum(),
        action_mean_abs_mean=jnp.abs(stacked_action_means).mean(),
        action_std_mean=stacked_action_stds.mean(),
        objective_term_mean=jnp.stack(objective_terms).mean(axis=(0, 1)) if objective_terms else None,
        termination_cause_count=(jnp.stack(termination_causes).sum(axis=(0, 1)) if termination_causes else None),
        tracking_error_mean=jnp.stack(tracking_errors).mean(axis=(0, 1)) if tracking_errors else None,
        nonfinite_world_count=jnp.stack(nonfinite_world_counts).sum() if nonfinite_world_counts else None,
        nonfinite_terminated_count=(
            jnp.stack(nonfinite_terminated_counts).sum() if nonfinite_terminated_counts else None
        ),
    )
    return batch, observation, normalization, key, metrics


@jax.jit
def sample_action_coordinated(state, observation: jax.Array, key: jax.Array, group_matrix: jax.Array):
    """Sample one action batch with per-group log-probabilities and K-column values."""
    from flash_chord.training.ppo.coordinate.per_group import per_group_log_prob

    policy, value = state.apply_fn(state.params, observation)
    action = policy.sample(seed=key)
    action_mean = policy.mean()
    action_std = policy.stddev()
    log_probability = per_group_log_prob(action, action_mean, action_std, group_matrix)
    return action, log_probability, value, action_mean, action_std


def collect_rollout_coordinated(
    env: JAXVectorEnv,
    state,
    normalization: NormalizationState,
    observation: jax.Array,
    key: jax.Array,
    config: TrainingConfig,
    group_matrix: jax.Array,
):
    """:func:`collect_rollout` with per-group log-probabilities and K value columns.

    The shared scalar reward is duplicated per value column and each column receives its own
    reference-end bootstrap ``gamma * V_k`` (stock rollout.py semantics applied per column).
    Returns ``(CoordinatedRolloutBatch, observation, normalization, key, RolloutMetrics)``.
    """
    from flash_chord.training.ppo.coordinate.per_group import CoordinatedRolloutBatch

    observations = []
    actions = []
    log_probabilities = []
    values = []
    rewards = []
    dones = []
    action_means = []
    action_stds = []
    raw_rewards = []
    reference_end_counts = []
    episode_returns = []
    episode_lengths = []
    episode_reference_progress = []
    episode_counts = []
    objective_terms = []
    termination_causes = []
    tracking_errors = []
    guard = config.nonfinite_guard
    nonfinite_world_counts = []
    nonfinite_terminated_counts = []

    for _ in range(config.rollout_steps):
        # ``observation`` is a fresh JAX array (the learner's copy or the previous step's copy); only the stored raw
        # observation of an unnormalized policy needs its own buffer.
        if config.observation_normalization:
            policy_observation = normalize_observation(
                normalization,
                observation,
                config.observation_normalization_epsilon,
            )
        else:
            policy_observation = jnp.array(observation, copy=True)
        key, action_key = jax.random.split(key)
        action, log_probability, value, action_mean, action_std = sample_action_coordinated(
            state,
            policy_observation,
            action_key,
            group_matrix,
        )
        transition = env.step(action)
        copied = copy_transition(transition.observation, transition.reward, transition.terminated, transition.truncated)
        next_observation, reward, terminated, done = copied.observation, copied.reward, copied.terminated, copied.done
        reference_end = copied.reference_end
        column_reward = reward[:, None] + config.ppo.discount * value * reference_end.astype(jnp.float32)[:, None]
        guarded = None
        if guard:
            # A bad world-step bootstraps from its own value (zero TD error, zero advantage) and ends the episode.
            guarded = _guard_transition(transition, next_observation, terminated, value, column_reward, done)
            next_observation, column_reward, done = guarded.observation, guarded.reward, guarded.done
            nonfinite_world_counts.append(guarded.nonfinite_count)
            if guarded.nonfinite_terminated_count is not None:
                nonfinite_terminated_counts.append(guarded.nonfinite_terminated_count)

        observations.append(policy_observation)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(column_reward)
        dones.append(done)
        action_means.append(action_mean)
        action_stds.append(action_std)
        _append_step_metrics(
            transition, guarded, copied, guard,
            raw_rewards, reference_end_counts, episode_returns, episode_lengths, episode_reference_progress,
            episode_counts, objective_terms, termination_causes, tracking_errors,
        )

        if config.observation_normalization:
            if guard:
                candidate = update_normalization(normalization, next_observation)
                normalization = guard_normalization_update(normalization, candidate, next_observation, guarded.valid)
            else:
                normalization = update_normalization(normalization, next_observation)
        observation = next_observation

    batch = CoordinatedRolloutBatch(
        observation=jnp.stack(observations),
        action=jnp.stack(actions),
        log_probability=jnp.stack(log_probabilities),
        value=jnp.stack(values),
        reward=jnp.stack(rewards),
        done=jnp.stack(dones),
        action_mean=jnp.stack(action_means),
        action_std=jnp.stack(action_stds),
    )
    stacked_action_means = jnp.stack(action_means)
    stacked_action_stds = jnp.stack(action_stds)
    metrics = RolloutMetrics(
        mean_step_reward=jnp.stack(raw_rewards).mean(),
        reference_end_count=jnp.stack(reference_end_counts).sum(),
        episode_return_sum=jnp.stack(episode_returns).sum(),
        episode_length_sum=jnp.stack(episode_lengths).sum(),
        episode_reference_progress_sum=jnp.stack(episode_reference_progress).sum(),
        episode_count=jnp.stack(episode_counts).sum(),
        action_mean_abs_mean=jnp.abs(stacked_action_means).mean(),
        action_std_mean=stacked_action_stds.mean(),
        objective_term_mean=jnp.stack(objective_terms).mean(axis=(0, 1)) if objective_terms else None,
        termination_cause_count=(jnp.stack(termination_causes).sum(axis=(0, 1)) if termination_causes else None),
        tracking_error_mean=jnp.stack(tracking_errors).mean(axis=(0, 1)) if tracking_errors else None,
        nonfinite_world_count=jnp.stack(nonfinite_world_counts).sum() if nonfinite_world_counts else None,
        nonfinite_terminated_count=(
            jnp.stack(nonfinite_terminated_counts).sum() if nonfinite_terminated_counts else None
        ),
    )
    return batch, observation, normalization, key, metrics
