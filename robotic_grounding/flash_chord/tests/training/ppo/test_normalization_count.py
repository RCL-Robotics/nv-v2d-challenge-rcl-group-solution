# SPDX-License-Identifier: Apache-2.0
"""Saturating int32 observation-normalizer count (normalization.saturating_count).

At 4096 worlds x 24 steps the count grows by 98,304 per PPO iteration and the stock int32 sum wraps during iteration
21,846. The saturating count must be bit-identical below 2**31 - 1, keep the int32 dtype, and keep the moments a
valid weighted merge once saturated.
"""

from __future__ import annotations

import numpy as np
import pytest

INT32_MAX = 2**31 - 1
ROWS = 4096  # worlds per env step
PER_ITERATION = ROWS * 24  # 98,304 samples per PPO iteration


def _dependencies():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    return jax, jnp


def _stock_update():
    """The pre-fix ``update_normalization`` (dde3c47..0eeb1b8), verbatim apart from the name: wraps at 2**31."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import NormalizationState

    @jax.jit
    def stock_update(state, observation):
        batch_count = observation.shape[0]
        count = state.count + batch_count
        rate = jnp.asarray(batch_count, dtype=observation.dtype) / count.astype(observation.dtype)
        batch_mean = observation.mean(axis=0)
        batch_variance = observation.var(axis=0)
        delta_mean = batch_mean - state.mean
        mean = state.mean + rate * delta_mean
        variance = state.variance + rate * (batch_variance - state.variance + delta_mean * (batch_mean - mean))
        return NormalizationState(mean=mean, variance=variance, count=count)

    return stock_update


def _state(jnp, count, dim=486, seed=0):
    from flash_chord.training.ppo.normalization import NormalizationState

    rng = np.random.default_rng(seed)
    return NormalizationState(
        mean=jnp.asarray(rng.uniform(-3, 3, dim).astype(np.float32)),
        variance=jnp.asarray(rng.uniform(0.1, 9.0, dim).astype(np.float32)),
        count=jnp.asarray(count, dtype=jnp.int32),
    )


def _batch(jnp, seed, rows=ROWS, dim=486):
    rng = np.random.default_rng(seed)
    scale, shift = rng.uniform(0.1, 30.0, dim), rng.uniform(-5, 5, dim)
    return jnp.asarray((rng.standard_normal((rows, dim)) * scale + shift).astype(np.float32))


def _assert_bitwise(left, right):
    for name in ("mean", "variance", "count"):
        a, b = np.asarray(getattr(left, name)), np.asarray(getattr(right, name))
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)


def test_saturating_count_is_the_plain_sum_below_the_limit_and_clamps_above():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import COUNT_MAX, saturating_count

    assert COUNT_MAX == INT32_MAX
    count_fn = jax.jit(saturating_count)
    for count in (0, 40, 21_845 * PER_ITERATION, INT32_MAX - ROWS):  # the last one fits exactly
        result = count_fn(jnp.asarray(count, jnp.int32), ROWS)
        assert result.dtype == jnp.int32 and int(result) == count + ROWS
    for count in (INT32_MAX - ROWS + 1, INT32_MAX - 1, INT32_MAX):
        result = count_fn(jnp.asarray(count, jnp.int32), ROWS)
        assert result.dtype == jnp.int32 and int(result) == INT32_MAX
    # Traced batch counts (the guarded, masked update) too.
    assert int(count_fn(jnp.asarray(INT32_MAX - 3, jnp.int32), jnp.asarray(7, jnp.int32))) == INT32_MAX
    assert int(count_fn(jnp.asarray(5, jnp.int32), jnp.asarray(7, jnp.int32))) == 12


@pytest.mark.parametrize("start", [0, 40, 1_000_000, 21_845 * PER_ITERATION, INT32_MAX - 3 * ROWS])
def test_update_is_bitwise_identical_to_the_stock_update_below_the_limit(start):
    """Up to and including a batch that lands exactly on 2**31 - 1, every output equals the stock update."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import update_normalization

    stock_update = _stock_update()
    stock = fixed = _state(jnp, start, seed=start % 97)
    for step in range(3):
        observation = _batch(jnp, seed=step + start % 13)
        stock, fixed = stock_update(stock, observation), update_normalization(fixed, observation)
        _assert_bitwise(stock, fixed)
    assert int(fixed.count) == start + 3 * ROWS


def test_count_saturates_where_the_stock_count_wraps_and_the_moments_stay_a_weighted_merge():
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import update_normalization

    stock_update = _stock_update()
    state = _state(jnp, INT32_MAX - 100, seed=3)
    observation = _batch(jnp, seed=4)
    wrapped = stock_update(state, observation)
    assert int(wrapped.count) < 0  # the bug: the stock count wraps and the merge rate turns negative

    fixed = update_normalization(state, observation)
    assert fixed.count.dtype == jnp.int32 and int(fixed.count) == INT32_MAX
    # Expected: the stock merge formula with the rate batch / COUNT_MAX (float32(2**31 - 1) == 2**31, so 2**-19).
    rate = np.float64(ROWS) / np.float64(np.float32(INT32_MAX))
    assert rate == 2.0**-19
    batch = np.asarray(observation, np.float64)
    mean0, var0 = np.asarray(state.mean, np.float64), np.asarray(state.variance, np.float64)
    delta = batch.mean(0) - mean0
    mean = (1 - rate) * mean0 + rate * batch.mean(0)
    variance = (1 - rate) * var0 + rate * batch.var(0) + rate * (1 - rate) * delta**2
    np.testing.assert_allclose(np.asarray(fixed.mean), mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fixed.variance), variance, rtol=1e-5, atol=1e-6)
    assert (np.asarray(fixed.variance) > 0).all()

    # Saturated: the count stays pinned and the moments move as an EMA with the fixed rate 2**-19.
    constant = jnp.full((ROWS, observation.shape[1]), 7.0, jnp.float32)
    pinned = fixed
    for _ in range(5):
        pinned = update_normalization(pinned, constant)
        assert int(pinned.count) == INT32_MAX
    expected_mean = 7.0 + (1 - rate) ** 5 * (np.asarray(fixed.mean, np.float64) - 7.0)
    np.testing.assert_allclose(np.asarray(pinned.mean), expected_mean, rtol=1e-5, atol=1e-5)
    assert np.isfinite(np.asarray(pinned.variance)).all() and (np.asarray(pinned.variance) > 0).all()


def test_training_scale_counts_saturate_during_iteration_21846():
    """24 env steps of 4096 rows per iteration: 21,845 iterations fit, iteration 21,846 saturates at step 8."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import update_normalization

    state = _state(jnp, 21_845 * PER_ITERATION, dim=4, seed=5)
    counts = []
    for step in range(24):
        state = update_normalization(state, _batch(jnp, seed=step, dim=4))
        counts.append(int(state.count))
        assert np.isfinite(np.asarray(state.mean)).all() and (np.asarray(state.variance) > 0).all()
    assert counts[:7] == [21_845 * PER_ITERATION + ROWS * (k + 1) for k in range(7)]
    assert counts[7:] == [INT32_MAX] * 17


def test_guarded_masked_update_saturates_too():
    """The guard's masked recompute (non-finite rows present) uses the same saturating count."""
    jax, jnp = _dependencies()
    from flash_chord.training.ppo.normalization import guard_normalization_update, update_normalization

    previous = _state(jnp, INT32_MAX - 10, dim=6, seed=6)
    observation = np.asarray(_batch(jnp, seed=7, rows=32, dim=6)).copy()
    observation[3, 2] = np.nan
    observation = jnp.asarray(observation)
    candidate = update_normalization(previous, observation)
    guarded = guard_normalization_update(previous, candidate, observation, jnp.ones((32,), bool))
    assert guarded.count.dtype == jnp.int32 and int(guarded.count) == INT32_MAX
    assert np.isfinite(np.asarray(guarded.mean)).all() and np.isfinite(np.asarray(guarded.variance)).all()
    # Far from the limit the masked path counts exactly the 31 finite rows, as before.
    small = _state(jnp, 40, dim=6, seed=8)
    small_guarded = guard_normalization_update(small, update_normalization(small, observation), observation,
                                               jnp.ones((32,), bool))
    assert int(small_guarded.count) == 40 + 31


def test_saturated_state_round_trips_through_a_checkpoint(tmp_path):
    """Same int32 leaf: a saturated count saves and restores like any other (old checkpoints load unchanged)."""
    jax, jnp = _dependencies()
    from flash_chord.training.checkpoint import load_checkpoint, save_checkpoint
    from flash_chord.training.ppo.normalization import create_normalization_state, update_normalization

    saturated = update_normalization(_state(jnp, INT32_MAX - 1, dim=3, seed=9), _batch(jnp, seed=10, rows=8, dim=3))
    save_checkpoint(tmp_path / "model_1.safetensors", saturated)
    restored, _ = load_checkpoint(tmp_path / "model_1.safetensors", create_normalization_state(3))
    _assert_bitwise(saturated, restored)
    assert int(restored.count) == INT32_MAX
