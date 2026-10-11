# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-device tests for the per-world non-finite simulator-state guard (runtime/nonfinite.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

WORLDS = 4
Q, DOF, BODIES, NV = 5, 4, 2, 6


def _state(wp, rng):
    joint_q = rng.standard_normal(WORLDS * Q).astype(np.float32)
    joint_qd = rng.standard_normal(WORLDS * DOF).astype(np.float32)
    body_q = rng.standard_normal((WORLDS * BODIES, 7)).astype(np.float32)
    body_qd = rng.standard_normal((WORLDS * BODIES, 6)).astype(np.float32)
    return joint_q, joint_qd, body_q, body_qd


def _guard(wp, warmstart):
    from flash_chord.runtime.nonfinite import NonfiniteWorldGuard

    model = SimpleNamespace(
        joint_coord_count=WORLDS * Q,
        joint_dof_count=WORLDS * DOF,
        body_count=WORLDS * BODIES,
        body_world=wp.array(np.repeat(np.arange(WORLDS), BODIES).astype(np.int32), dtype=wp.int32, device="cpu"),
    )
    solver = SimpleNamespace(mjw_data=SimpleNamespace(qacc_warmstart=warmstart))
    return NonfiniteWorldGuard.build(model, solver, WORLDS, device="cpu")


def _run(wp, arrays, terminated, warmstart_np, score):
    joint_q, joint_qd, body_q, body_qd = arrays
    state = SimpleNamespace(
        joint_q=wp.array(joint_q, dtype=wp.float32, device="cpu"),
        joint_qd=wp.array(joint_qd, dtype=wp.float32, device="cpu"),
        body_q=wp.array(body_q, dtype=wp.transform, device="cpu"),
        body_qd=wp.array(body_qd, dtype=wp.spatial_vector, device="cpu"),
    )
    warmstart = wp.array(warmstart_np, dtype=wp.float32, device="cpu")
    guard = _guard(wp, warmstart)
    terminated_wp = wp.array(terminated, dtype=wp.int32, device="cpu")
    score_wp = wp.array(score, dtype=wp.float32, device="cpu")
    guard.check(state, terminated_wp)
    guard.zero_scores(score_wp)
    wp.synchronize_device("cpu")
    return guard, terminated_wp.numpy(), warmstart.numpy(), score_wp.numpy(), state


def test_guard_is_a_noop_when_every_world_is_finite():
    wp = pytest.importorskip("warp")
    wp.init()
    rng = np.random.default_rng(0)
    arrays = _state(wp, rng)
    terminated = np.asarray([0, 1, 0, 0], np.int32)
    warmstart = rng.standard_normal((WORLDS, NV)).astype(np.float32)
    score = rng.standard_normal(WORLDS).astype(np.float32)
    guard, terminated_out, warmstart_out, score_out, state = _run(wp, arrays, terminated, warmstart, score)
    assert guard.flag.numpy().tolist() == [0, 0, 0, 0]
    assert int(guard.total.numpy()[0]) == 0
    np.testing.assert_array_equal(terminated_out, terminated)
    np.testing.assert_array_equal(warmstart_out, warmstart)
    np.testing.assert_array_equal(score_out, score)
    for name, original in zip(("joint_q", "joint_qd", "body_q", "body_qd"), arrays, strict=True):
        np.testing.assert_array_equal(getattr(state, name).numpy().reshape(original.shape), original)


@pytest.mark.parametrize(
    ("field", "index"),
    [("joint_q", 1 * Q + 3), ("joint_qd", 1 * DOF), ("body_q", (1 * BODIES + 1, 5)), ("body_qd", (1 * BODIES, 2)),
     ("warmstart", (1, NV - 1))],
)
def test_guard_terminates_one_nonfinite_world_and_zeroes_its_warm_start(field, index):
    wp = pytest.importorskip("warp")
    wp.init()
    rng = np.random.default_rng(1)
    arrays = list(_state(wp, rng))
    warmstart = rng.standard_normal((WORLDS, NV)).astype(np.float32)
    names = ("joint_q", "joint_qd", "body_q", "body_qd")
    if field == "warmstart":
        warmstart[index] = np.inf
    else:
        arrays[names.index(field)][index] = np.nan
    warmstart[1, 0] = np.nan  # a failed solve also leaves NaN in the warm start of its world
    terminated = np.asarray([0, 0, 1, 0], np.int32)
    score = np.asarray([0.5, np.nan, 0.25, 0.75], np.float32)
    guard, terminated_out, warmstart_out, score_out, _ = _run(wp, tuple(arrays), terminated, warmstart, score)

    assert guard.flag.numpy().tolist() == [0, 1, 0, 0]
    assert int(guard.total.numpy()[0]) == 1
    assert terminated_out.tolist() == [0, 1, 1, 0]
    np.testing.assert_array_equal(warmstart_out[1], np.zeros(NV, np.float32))
    np.testing.assert_array_equal(np.delete(warmstart_out, 1, axis=0), np.delete(warmstart, 1, axis=0))
    np.testing.assert_array_equal(score_out, np.asarray([0.5, 0.0, 0.25, 0.75], np.float32))


def test_guard_build_validates_layout_and_tolerates_missing_warm_start():
    wp = pytest.importorskip("warp")
    wp.init()
    from flash_chord.runtime.nonfinite import NonfiniteWorldGuard

    model = SimpleNamespace(joint_coord_count=WORLDS * Q, joint_dof_count=WORLDS * DOF + 1, body_count=WORLDS * BODIES,
                            body_world=None)
    with pytest.raises(ValueError, match="uniform per-world layout"):
        NonfiniteWorldGuard.build(model, SimpleNamespace(), WORLDS, device="cpu")
    interleaved = np.tile(np.arange(WORLDS), BODIES).astype(np.int32)
    model = SimpleNamespace(joint_coord_count=WORLDS * Q, joint_dof_count=WORLDS * DOF, body_count=WORLDS * BODIES,
                            body_world=wp.array(interleaved, dtype=wp.int32, device="cpu"))
    with pytest.raises(ValueError, match="contiguously"):
        NonfiniteWorldGuard.build(model, SimpleNamespace(), WORLDS, device="cpu")
    model.body_world = None
    guard = NonfiniteWorldGuard.build(model, SimpleNamespace(mjw_data=None), WORLDS, device="cpu")
    assert guard.warmstart_dim == 0
    rng = np.random.default_rng(2)
    joint_q, joint_qd, body_q, body_qd = _state(wp, rng)
    joint_qd[3 * DOF + 1] = np.nan
    state = SimpleNamespace(
        joint_q=wp.array(joint_q, dtype=wp.float32, device="cpu"),
        joint_qd=wp.array(joint_qd, dtype=wp.float32, device="cpu"),
        body_q=wp.array(body_q, dtype=wp.transform, device="cpu"),
        body_qd=wp.array(body_qd, dtype=wp.spatial_vector, device="cpu"),
    )
    terminated = wp.zeros(WORLDS, dtype=wp.int32, device="cpu")
    guard.check(state, terminated)
    assert guard.flag.numpy().tolist() == [0, 0, 0, 1]
    assert terminated.numpy().tolist() == [0, 0, 0, 1]
