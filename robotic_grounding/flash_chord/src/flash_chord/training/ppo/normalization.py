# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Functional observation normalization matching RSL-RL semantics."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp


@flax.struct.dataclass
class NormalizationState:
    """Device-resident running population moments."""

    mean: jax.Array
    variance: jax.Array
    count: jax.Array


def create_normalization_state(observation_dim: int) -> NormalizationState:
    """Create the zero-mean, unit-variance state used by RSL-RL."""
    if observation_dim <= 0:
        raise ValueError(f"observation_dim must be positive, got {observation_dim}")
    return NormalizationState(
        mean=jnp.zeros((observation_dim,), dtype=jnp.float32),
        variance=jnp.ones((observation_dim,), dtype=jnp.float32),
        count=jnp.asarray(0, dtype=jnp.int32),
    )


@jax.jit
def update_normalization(state: NormalizationState, observation: jax.Array) -> NormalizationState:
    """Merge one observation batch into the running population moments."""
    batch_count = observation.shape[0]
    count = state.count + batch_count
    rate = jnp.asarray(batch_count, dtype=observation.dtype) / count.astype(observation.dtype)
    batch_mean = observation.mean(axis=0)
    batch_variance = observation.var(axis=0)
    delta_mean = batch_mean - state.mean
    mean = state.mean + rate * delta_mean
    variance = state.variance + rate * (batch_variance - state.variance + delta_mean * (batch_mean - mean))
    return NormalizationState(mean=mean, variance=variance, count=count)


@jax.jit
def normalize_observation(
    state: NormalizationState,
    observation: jax.Array,
    epsilon: float = 1.0e-2,
) -> jax.Array:
    """Normalize observations with RSL-RL's standard-deviation epsilon."""
    return (observation - state.mean) / (jnp.sqrt(state.variance) + epsilon)


# Rows whose magnitude could overflow the float32 variance update (x * x > 3.4e38) never enter the masked update.
_MASKED_UPDATE_ABS_LIMIT = 1.0e18


def _masked_update(state: NormalizationState, observation: jax.Array, rows: jax.Array) -> NormalizationState:
    """:func:`update_normalization` restricted to the selected rows (same parallel-moments formula)."""
    dtype = observation.dtype
    selected = rows[:, None]
    batch_count = jnp.sum(rows.astype(jnp.int32))
    count = state.count + batch_count
    rate = batch_count.astype(dtype) / jnp.maximum(count, 1).astype(dtype)
    divisor = jnp.maximum(batch_count, 1).astype(dtype)
    batch_mean = jnp.sum(jnp.where(selected, observation, 0.0), axis=0) / divisor
    centered = jnp.where(selected, observation - batch_mean, 0.0)
    batch_variance = jnp.sum(centered * centered, axis=0) / divisor
    delta_mean = batch_mean - state.mean
    mean = state.mean + rate * delta_mean
    variance = state.variance + rate * (batch_variance - state.variance + delta_mean * (batch_mean - mean))
    return NormalizationState(mean=mean, variance=variance, count=count)


def _all_finite(state: NormalizationState) -> jax.Array:
    return jnp.all(jnp.isfinite(state.mean)) & jnp.all(jnp.isfinite(state.variance))


@jax.jit
def guard_normalization_update(
    previous: NormalizationState,
    candidate: NormalizationState,
    observation: jax.Array,
    valid: jax.Array,
) -> NormalizationState:
    """Keep non-finite observation rows out of the running moments.

    ``candidate`` must be ``update_normalization(previous, observation)``. When every row is ``valid`` and
    finite and ``candidate`` is finite, ``candidate`` is returned unchanged, so a run without non-finite
    values is bit-identical to the unguarded update. Otherwise the moments are recomputed from the valid,
    finite, non-overflowing rows only; if none remain, or the result is still non-finite, ``previous`` is
    kept.
    """
    finite_rows = jnp.all(jnp.isfinite(observation), axis=-1)
    accept = jnp.all(valid & finite_rows) & _all_finite(candidate)

    def masked(_):
        rows = valid & finite_rows & jnp.all(jnp.abs(observation) <= _MASKED_UPDATE_ABS_LIMIT, axis=-1)
        updated = _masked_update(previous, observation, rows)
        usable = _all_finite(updated) & jnp.any(rows)
        return jax.tree.map(lambda new, old: jnp.where(usable, new, old), updated, previous)

    return jax.lax.cond(accept, lambda _: candidate, masked, None)
