# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composable learner state and one-iteration training operation."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flash_chord.training.environment import JAXVectorEnv
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.network import ActorCritic
from flash_chord.training.ppo.normalization import (
    NormalizationState,
    create_normalization_state,
    normalize_observation,
)
from flash_chord.training.ppo.algorithm import PPOMetrics, create_train_state, update
from flash_chord.training.ppo.rollout import RolloutMetrics, collect_rollout


@flax.struct.dataclass
class LearnerState:
    """Complete device-resident learner state between iterations."""

    train_state: TrainState
    normalization: NormalizationState
    observation: jax.Array
    key: jax.Array
    learning_rate: jax.Array
    iteration: jax.Array


@flax.struct.dataclass
class IterationMetrics:
    """Rollout and optimization metrics from one learner iteration."""

    rollout: RolloutMetrics
    optimization: PPOMetrics


def create_learner(env: JAXVectorEnv, config: TrainingConfig) -> LearnerState:
    """Initialize policy, optimizer, normalization, and the first environment observation."""
    if env.world_count != config.world_count:
        raise ValueError(f"environment has {env.world_count} worlds; config requests {config.world_count}")
    key, initialization_key = jax.random.split(jax.random.PRNGKey(config.seed))
    model = ActorCritic(action_dim=env.action_dim, config=config.network)
    parameters = model.init(
        initialization_key,
        jnp.zeros((1, env.observation_dim), dtype=jnp.float32),
    )
    train_state = create_train_state(model, parameters, config.ppo)
    return LearnerState(
        train_state=train_state,
        normalization=create_normalization_state(env.observation_dim),
        observation=jnp.array(env.reset(), copy=True),
        key=key,
        learning_rate=jnp.asarray(config.ppo.learning_rate, dtype=jnp.float32),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )


def train_iteration(
    env: JAXVectorEnv,
    learner: LearnerState,
    config: TrainingConfig,
) -> tuple[LearnerState, IterationMetrics]:
    """Collect one rollout and apply all configured PPO epochs and mini-batches."""
    batch, observation, normalization, key, rollout_metrics = collect_rollout(
        env,
        learner.train_state,
        learner.normalization,
        learner.observation,
        learner.key,
        config,
    )
    policy_observation = observation
    if config.observation_normalization:
        policy_observation = normalize_observation(
            normalization,
            observation,
            config.observation_normalization_epsilon,
        )
    _, last_value = learner.train_state.apply_fn(
        learner.train_state.params,
        policy_observation,
    )
    key, update_key = jax.random.split(key)
    train_state, learning_rate, optimization_metrics = update(
        learner.train_state,
        batch,
        last_value,
        update_key,
        learner.learning_rate,
        config.ppo,
    )
    learner = learner.replace(
        train_state=train_state,
        normalization=normalization,
        observation=observation,
        key=key,
        learning_rate=learning_rate,
        iteration=learner.iteration + 1,
    )
    return learner, IterationMetrics(
        rollout=rollout_metrics,
        optimization=optimization_metrics,
    )


def _model_for(config: TrainingConfig, action_dim: int, layout=None):
    if layout is None:
        return ActorCritic(action_dim=action_dim, config=config.network)
    if config.coordinate is None:
        raise ValueError("a coordinated layout requires training.coordinate")
    from flash_chord.training.ppo.coordinate.network import CoordinatedActorCritic

    if layout.observation_dim <= 0 or layout.action_dim != action_dim:
        raise ValueError(f"layout action_dim {layout.action_dim} differs from environment action_dim {action_dim}")
    return CoordinatedActorCritic(
        action_dim=action_dim,
        layout=layout,
        coordinate=config.coordinate,
        network=config.network,
    )


def create_learner_template(
    config: TrainingConfig,
    observation_dim: int,
    action_dim: int,
    layout=None,
    world_count: int = 1,
) -> LearnerState:
    """Environment-free learner with the same initialization as training (export, tests, restore).

    ``layout=None`` builds the stock :class:`ActorCritic`; a :class:`CoordinateLayout` builds the
    coordinated actor-critic. The observation leaf is ``zeros((world_count, observation_dim))``.
    """
    if layout is not None and layout.observation_dim != observation_dim:
        raise ValueError(f"layout observation_dim {layout.observation_dim} differs from {observation_dim}")
    key, initialization_key = jax.random.split(jax.random.PRNGKey(config.seed))
    model = _model_for(config, action_dim, layout)
    parameters = model.init(
        initialization_key,
        jnp.zeros((1, observation_dim), dtype=jnp.float32),
    )
    train_state = create_train_state(model, parameters, config.ppo)
    return LearnerState(
        train_state=train_state,
        normalization=create_normalization_state(observation_dim),
        observation=jnp.zeros((world_count, observation_dim), dtype=jnp.float32),
        key=key,
        learning_rate=jnp.asarray(config.ppo.learning_rate, dtype=jnp.float32),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )


def create_coordinated_learner(env: JAXVectorEnv, config: TrainingConfig, layout) -> LearnerState:
    """:func:`create_learner` with the coordinated actor-critic described by ``layout``."""
    if env.world_count != config.world_count:
        raise ValueError(f"environment has {env.world_count} worlds; config requests {config.world_count}")
    if layout.observation_dim != env.observation_dim:
        raise ValueError(
            f"environment observation_dim {env.observation_dim} differs from layout {layout.observation_dim}"
        )
    key, initialization_key = jax.random.split(jax.random.PRNGKey(config.seed))
    model = _model_for(config, env.action_dim, layout)
    parameters = model.init(
        initialization_key,
        jnp.zeros((1, env.observation_dim), dtype=jnp.float32),
    )
    train_state = create_train_state(model, parameters, config.ppo)
    return LearnerState(
        train_state=train_state,
        normalization=create_normalization_state(env.observation_dim),
        observation=jnp.array(env.reset(), copy=True),
        key=key,
        learning_rate=jnp.asarray(config.ppo.learning_rate, dtype=jnp.float32),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )


def train_iteration_coordinated(
    env: JAXVectorEnv,
    learner: LearnerState,
    config: TrainingConfig,
    layout,
) -> tuple[LearnerState, IterationMetrics]:
    """Collect one rollout and apply the per-agent (W0-P) PPO update.

    ``IterationMetrics.optimization`` is a :class:`CoordinatedPPOMetrics`.
    """
    from flash_chord.training.ppo.coordinate.per_group import group_matrix, update_coordinated
    from flash_chord.training.ppo.rollout import collect_rollout_coordinated

    num_groups = layout.num_agents
    batch, observation, normalization, key, rollout_metrics = collect_rollout_coordinated(
        env,
        learner.train_state,
        learner.normalization,
        learner.observation,
        learner.key,
        config,
        group_matrix(layout.action_group, num_groups),
    )
    policy_observation = observation
    if config.observation_normalization:
        policy_observation = normalize_observation(
            normalization,
            observation,
            config.observation_normalization_epsilon,
        )
    _, last_value = learner.train_state.apply_fn(
        learner.train_state.params,
        policy_observation,
    )
    key, update_key = jax.random.split(key)
    train_state, learning_rate, optimization_metrics = update_coordinated(
        learner.train_state,
        batch,
        last_value,
        update_key,
        learner.learning_rate,
        config.ppo,
        layout.action_group,
        num_groups,
    )
    learner = learner.replace(
        train_state=train_state,
        normalization=normalization,
        observation=observation,
        key=key,
        learning_rate=learning_rate,
        iteration=learner.iteration + 1,
    )
    return learner, IterationMetrics(
        rollout=rollout_metrics,
        optimization=optimization_metrics,
    )
