# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-release options of the ReconBody ``object_task`` term (objectives/recon_body.py).

``object_task_position_only_after_release`` drops the rotation factor of ``object_task`` on the frames after the
final release (every frame after the last frame in which any reference contact label is on); ``no_drop_weight``
adds an ``object_no_drop`` term ``max(0, (z_ref - no_drop_margin_m) - z)`` on the same frames. Off (the defaults),
the objective must be the original ``object_task`` objective bit for bit: same term names, buffers, kernels and
outputs. ``_original_add_recon_body_object_task`` below is a verbatim copy of the kernel this option was added to
(``add_recon_body_object_task`` at b39b69b), so a later edit of the module cannot hide a drift.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

STOCK_TERMS = 12
_WORLDS = 10
_FRAMES = 12
_BODIES = 5  # 0 pelvis, 1-2 wrists, 3 object, 4 spare
_OBJECT = 3
_Q = 10
_SLOTS = 4  # two links per hand
_DT = 0.02
_TASK_WEIGHT = 5.0
_BETA_POS, _BETA_ROT = 10.0, 4.5

# Hand 0 on 2-4 and 6-7, hand 1 on 3-4 and 6-8: frames 0-1 precede the first contact, frame 5 is a mid-clip
# frame with both labels off, and the final release is after frame 8, so frames 9-11 are released.
_LABELS = np.zeros((_FRAMES, 2), dtype=np.float32)
_LABELS[2:5, 0] = 1.0
_LABELS[6:8, 0] = 1.0
_LABELS[3:5, 1] = 1.0
_LABELS[6:9, 1] = 1.0
_RELEASED_FROM = 9
# One world per case: pre-contact 0, both-off gap 5, contact 7, last contact 8, released 9/10/11, clamped 14 -> 11.
_TIMESTEPS = np.asarray([0, 5, 7, 8, 9, 10, 11, 14, 9, 11], dtype=np.int32)
_FRAME = np.clip(_TIMESTEPS, 0, _FRAMES - 1)
_RELEASED = _FRAME >= _RELEASED_FROM
# Object height relative to its target: below by more than the 5 cm margin, inside it, or above.
_DZ = np.asarray([-0.20, -0.20, -0.12, -0.12, -0.12, -0.03, 0.04, -0.30, 0.0, -0.06], dtype=np.float32)


def _original_kernel():
    """The b39b69b ``add_recon_body_object_task``, verbatim but for its constants (12 stock terms, stride 13)."""
    from flash_chord.objectives.recon_body import object_task_rotation_distance

    @wp.kernel
    def _original_add_recon_body_object_task(
        body_q: wp.array(dtype=wp.transform),
        stock_terms: wp.array(dtype=wp.float32),
        object_body_id: int,
        object_target_pos_w: wp.array(dtype=wp.vec3),
        object_target_quat_w: wp.array(dtype=wp.quat),
        bodies_per_world: int,
        position_beta: float,
        rotation_beta: float,
        weights: wp.array(dtype=wp.float32),
        frame_dt: float,
        terms: wp.array(dtype=wp.float32),
        score: wp.array(dtype=wp.float32),
    ) -> None:
        world = wp.tid()
        stock_base = world * 12
        term_base = world * 13
        for term in range(12):
            terms[term_base + term] = stock_terms[stock_base + term]

        object_xf = body_q[world * bodies_per_world + object_body_id]
        position_distance = wp.length(wp.transform_get_translation(object_xf) - object_target_pos_w[world])
        rotation_distance = object_task_rotation_distance(
            wp.transform_get_rotation(object_xf), object_target_quat_w[world]
        )
        task = wp.exp(-position_beta * position_distance) * wp.exp(-rotation_beta * rotation_distance)
        terms[term_base + 12] = task
        score[world] = score[world] + frame_dt * weights[12] * task

    return _original_add_recon_body_object_task


# --------------------------------------------------------------------------------------------------
# Synthetic objective (real buffers and kernels, stand-in runtime dependencies)
# --------------------------------------------------------------------------------------------------


def _random_unit_quats(rng, count):
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _axis_angle_xyzw(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.concatenate((axis * math.sin(angle / 2.0), [math.cos(angle / 2.0)]))


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _targets(seed: int = 0):
    rng = np.random.default_rng(seed)
    target_pos = rng.normal(scale=0.5, size=(_WORLDS, 3)).astype(np.float32)
    target_pos[:, 2] += 0.6
    target_quat = _random_unit_quats(rng, _WORLDS).astype(np.float32)
    return target_pos, target_quat


def _fake_objective(config, labels=_LABELS, seed: int = 0, release_mask="build"):
    from flash_chord.objectives.recon_body import (
        RECON_BODY_OBJECTIVE_TERM_NAMES,
        ReconBodyObjective,
        final_release_mask,
    )
    from flash_chord.objectives.wrench import friction_cone_angles, sample_wrench_basis

    rng = np.random.default_rng(seed)
    f32 = lambda values: wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32)  # noqa: E731
    i32 = lambda values: wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32)  # noqa: E731
    reference_joint_q = rng.normal(scale=0.3, size=(_FRAMES, _Q)).astype(np.float32)
    reference_joint_q[:, 3:7] = _random_unit_quats(rng, _FRAMES)
    contact_dir = rng.normal(size=(_WORLDS * _SLOTS, 3))
    contact_dir = (contact_dir / np.linalg.norm(contact_dir, axis=-1, keepdims=True)).astype(np.float32)
    num_basis, num_edges = 16, 8
    cone_cos, cone_sin = friction_cone_angles(num_edges)
    target_pos, target_quat = _targets(seed)
    if release_mask == "build":
        release_mask = (
            wp.array(final_release_mask(labels), dtype=wp.float32) if config.object_task_release_enabled else None
        )
    return ReconBodyObjective(
        config=config,
        world_count=_WORLDS,
        frame_dt=_DT,
        num_frames=_FRAMES,
        num_hands=2,
        num_bodies=1,
        num_basis=num_basis,
        num_edges=num_edges,
        num_joints=4,
        bodies_per_world=_BODIES,
        q_per_world=_Q,
        pelvis_body_id=0,
        object_body_id=_OBJECT,
        episode_start_frame=i32(rng.integers(0, 4, size=_WORLDS)),
        reference_joint_q_offset=f32(rng.normal(scale=0.01, size=_WORLDS * _Q)),
        reference=SimpleNamespace(
            joint_q=wp.array(reference_joint_q, dtype=wp.float32),
            wrist_pos_w=wp.array(rng.normal(scale=0.3, size=(_FRAMES * 2, 3)).astype(np.float32), dtype=wp.vec3),
            wrist_quat_w=wp.array(_random_unit_quats(rng, _FRAMES * 2).astype(np.float32), dtype=wp.quat),
        ),
        command=SimpleNamespace(
            object_radius=f32(np.full(_WORLDS, 0.1)),
            body_target_pos_w=wp.array(target_pos, dtype=wp.vec3),
            body_target_quat_w=wp.array(target_quat, dtype=wp.quat),
        ),
        contact=SimpleNamespace(
            contact_pos_o=wp.array(rng.normal(scale=0.05, size=(_WORLDS * _SLOTS, 3)).astype(np.float32), wp.vec3),
            contact_force_direction_o=wp.array(contact_dir, dtype=wp.vec3),
            hand_slot_starts=i32([0, 2]),
            link_counts=i32([2, 2]),
            layout=SimpleNamespace(slots_per_world=_SLOTS),
        ),
        action=SimpleNamespace(
            action_rate_l2=f32(rng.uniform(0.0, 2.0, size=_WORLDS)),
            action_l2=f32(rng.uniform(0.0, 2.0, size=_WORLDS)),
            sonic_q_ids=i32([7, 8, 9]),
        ),
        wrist_frames=SimpleNamespace(
            body_ids=i32([1, 2]),
            body_to_frame_pos=wp.array(rng.normal(scale=0.05, size=(2, 3)).astype(np.float32), dtype=wp.vec3),
            body_to_frame_quat=wp.array(_random_unit_quats(rng, 2).astype(np.float32), dtype=wp.quat),
        ),
        scalar_q_ids=i32([6, 7, 8, 9]),
        soft_lower=f32([-0.2, -0.3, -0.1, -0.4]),
        soft_upper=f32([0.2, 0.3, 0.1, 0.4]),
        contact_active=f32(np.asarray(labels, dtype=np.float32).reshape(-1)),
        wrench_basis=wp.array(sample_wrench_basis(num_basis, 1.0, 0), dtype=wp.spatial_vector),
        cone_cos=f32(cone_cos),
        cone_sin=f32(cone_sin),
        wrench_contact_enabled=wp.ones(_SLOTS, dtype=wp.int32),
        current_wrench_support=wp.zeros(_WORLDS * 2 * num_basis, dtype=wp.float32),
        force_closure=wp.zeros(_WORLDS, dtype=wp.float32),
        weights=wp.array(config.term_weights, dtype=wp.float32),
        terms=wp.zeros(_WORLDS * len(config.term_names), dtype=wp.float32),
        score=wp.zeros(_WORLDS, dtype=wp.float32),
        stock_terms=(
            wp.zeros(_WORLDS * len(RECON_BODY_OBJECTIVE_TERM_NAMES), dtype=wp.float32)
            if config.object_task_enabled
            else None
        ),
        release_mask=release_mask,
    )


def _state(seed: int):
    """Bodies at random poses; each world's object near its target: x/y within 4 cm, z offset _DZ, 0.2-2.5 rad off."""
    rng = np.random.default_rng(100 + seed)
    target_pos, target_quat = _targets(0)
    positions = rng.normal(scale=0.5, size=(_WORLDS * _BODIES, 3)).astype(np.float32)
    quats = _random_unit_quats(rng, _WORLDS * _BODIES).astype(np.float32)
    angles = rng.uniform(0.2, 2.5, size=_WORLDS)
    for world in range(_WORLDS):
        row = world * _BODIES + _OBJECT
        offset = np.asarray([*rng.uniform(-0.04, 0.04, size=2), _DZ[world]], dtype=np.float32)
        positions[row] = target_pos[world] + offset
        rotation = _axis_angle_xyzw(rng.normal(size=3), angles[world])
        quats[row] = _quat_mul_xyzw(target_quat[world].astype(np.float64), rotation).astype(np.float32)
    transforms = [wp.transform(wp.vec3(*positions[i]), wp.quat(*quats[i])) for i in range(_WORLDS * _BODIES)]
    state = SimpleNamespace(
        body_q=wp.array(transforms, dtype=wp.transform),
        joint_q=wp.array(rng.normal(scale=0.3, size=_WORLDS * _Q).astype(np.float32), dtype=wp.float32),
    )
    timestep = wp.array(_TIMESTEPS, dtype=wp.int32)
    terminated = wp.array((rng.uniform(size=_WORLDS) > 0.7).astype(np.int32), dtype=wp.int32)
    return state, timestep, terminated, positions, quats


def _evaluate(objective, seed: int):
    state, timestep, terminated, positions, quats = _state(seed)
    objective.evaluate(state, timestep, terminated)
    terms = objective.terms.numpy().copy().reshape(_WORLDS, len(objective.term_names))
    return terms, objective.score.numpy().copy(), positions, quats


def _bits(values):
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def _record_launches(monkeypatch):
    launches = []
    original = wp.launch

    def recording(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", recording)
    return launches


def _config(**kwargs):
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    return ReconBodyObjectiveConfig(
        object_task_weight=_TASK_WEIGHT,
        object_task_position_beta=_BETA_POS,
        object_task_rotation_beta=_BETA_ROT,
        **kwargs,
    )


def _object_rows(positions, quats):
    rows = np.arange(_WORLDS) * _BODIES + _OBJECT
    return positions[rows].astype(np.float64), quats[rows].astype(np.float64)


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------


def test_defaults_are_off_and_keep_the_object_task_objective():
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjectiveConfig

    stock = ReconBodyObjectiveConfig()
    assert stock.object_task_position_only_after_release is False
    assert (stock.no_drop_weight, stock.no_drop_margin_m) == (0.0, 0.05)
    assert not stock.object_no_drop_enabled and not stock.object_task_release_enabled
    assert stock.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES and stock.term_weights == stock.weights
    config = _config()
    assert not config.object_task_release_enabled
    assert config.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES + ("object_task",)
    assert config.term_weights == config.weights + (_TASK_WEIGHT,)


def test_position_only_keeps_the_term_layout_and_no_drop_appends_one_term():
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES

    position_only = _config(object_task_position_only_after_release=True)
    assert position_only.object_task_release_enabled and not position_only.object_no_drop_enabled
    assert position_only.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES + ("object_task",)
    assert position_only.term_weights == position_only.weights + (_TASK_WEIGHT,)
    both = _config(object_task_position_only_after_release=True, no_drop_weight=-50.0)
    assert both.object_no_drop_enabled and both.object_task_release_enabled
    assert both.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES + ("object_task", "object_no_drop")
    assert both.term_weights == both.weights + (_TASK_WEIGHT, -50.0)
    assert len(both.weights) == STOCK_TERMS
    no_drop_only = _config(no_drop_weight=-1.0, no_drop_margin_m=0.0)
    assert no_drop_only.object_task_release_enabled and not no_drop_only.object_task_position_only_after_release


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"object_task_weight": 0.0, "object_task_position_only_after_release": True}, "need object_task"),
        ({"object_task_weight": 0.0, "no_drop_weight": -10.0}, "need object_task"),
        ({"no_drop_weight": 10.0}, "<= 0"),
        ({"no_drop_weight": math.nan}, "finite"),
        ({"no_drop_weight": -math.inf}, "finite"),
        ({"no_drop_margin_m": math.inf}, "finite"),
        ({"no_drop_margin_m": -0.01}, "non-negative"),
        ({"object_task_position_only_after_release": 1}, "bool"),
        ({"object_task_position_only_after_release": "true"}, "bool"),
    ],
)
def test_invalid_release_options_are_rejected(kwargs, match):
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    values = {"object_task_weight": _TASK_WEIGHT, **kwargs}
    with pytest.raises(ValueError, match=match):
        ReconBodyObjectiveConfig(**values)


def test_final_release_mask_is_the_suffix_after_the_last_contact_label():
    from flash_chord.objectives.recon_body import final_release_mask

    mask = final_release_mask(_LABELS)
    assert mask.dtype == np.float32 and mask.shape == (_FRAMES,)
    np.testing.assert_array_equal(np.flatnonzero(mask), np.arange(_RELEASED_FROM, _FRAMES))
    assert mask[0] == 0.0 and mask[5] == 0.0  # pre-contact and the mid-clip both-off frame are not released
    np.testing.assert_array_equal(final_release_mask(np.zeros((_FRAMES, 2))), np.zeros(_FRAMES))
    held = _LABELS.copy()
    held[-1, 0] = 1.0
    np.testing.assert_array_equal(final_release_mask(held), np.zeros(_FRAMES))
    one_hand = np.zeros((_FRAMES, 1))
    one_hand[:4] = 1.0
    np.testing.assert_array_equal(np.flatnonzero(final_release_mask(one_hand)), np.arange(4, _FRAMES))
    # ep28 shape: left last on 876, right last on 887 -> released 888..987 (100 frames).
    ep28 = np.zeros((988, 2))
    ep28[102:877, 0] = 1.0
    ep28[108:888, 1] = 1.0
    np.testing.assert_array_equal(np.flatnonzero(final_release_mask(ep28)), np.arange(888, 988))
    with pytest.raises(ValueError, match="shape"):
        final_release_mask(np.zeros(_FRAMES))


# --------------------------------------------------------------------------------------------------
# Objective-level: evaluate() through a small CPU objective
# --------------------------------------------------------------------------------------------------


def test_options_off_are_bit_identical_to_the_original_object_task_kernel(monkeypatch):
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, compose_recon_body_objective

    original = _original_kernel()
    with wp.ScopedDevice("cpu"):
        objective = _fake_objective(_config())
        assert objective.release_mask is None and objective.stock_terms is not None
        assert objective.terms.shape == (_WORLDS * (STOCK_TERMS + 1),)
        launches = _record_launches(monkeypatch)
        state, timestep, terminated, _, _ = _state(1)
        objective.evaluate(state, timestep, terminated)
        assert launches == [
            "batched_wrench_support",
            "reduce_recon_body_force_closure",
            "compose_recon_body_objective",
            "add_recon_body_object_task",
        ]
        terms, score = objective.terms.numpy().copy(), objective.score.numpy().copy()

        # The stock kernel then the frozen original object_task kernel, launched by hand, give the same bits.
        o = objective
        manual_stock = wp.zeros(_WORLDS * STOCK_TERMS, dtype=wp.float32)
        manual_terms = wp.zeros(_WORLDS * (STOCK_TERMS + 1), dtype=wp.float32)
        manual_score = wp.zeros(_WORLDS, dtype=wp.float32)
        wp.launch(
            compose_recon_body_objective,
            dim=_WORLDS,
            inputs=[
                state.body_q, state.joint_q, timestep, o.episode_start_frame, terminated, o.action.action_rate_l2,
                o.action.action_l2, o.force_closure, o.reference_joint_q_offset, o.reference.joint_q,
                o.reference.wrist_pos_w, o.reference.wrist_quat_w, o.wrist_frames.body_ids,
                o.wrist_frames.body_to_frame_pos, o.wrist_frames.body_to_frame_quat, o.scalar_q_ids,
                o.action.sonic_q_ids, o.soft_lower, o.soft_upper, o.object_body_id, o.command.body_target_pos_w,
                o.pelvis_body_id, o.num_frames, o.bodies_per_world, o.q_per_world, o.num_joints,
                o.action.sonic_q_ids.shape[0], 0.3, 0.4, 1.0, 0.2, 0.2, 0.4, o.weights, o.frame_dt,
            ],
            outputs=[manual_stock, manual_score],
        )
        wp.launch(
            original,
            dim=_WORLDS,
            inputs=[
                state.body_q, manual_stock, o.object_body_id, o.command.body_target_pos_w,
                o.command.body_target_quat_w, o.bodies_per_world, _BETA_POS, _BETA_ROT, o.weights, o.frame_dt,
            ],
            outputs=[manual_terms, manual_score],
        )
    assert len(RECON_BODY_OBJECTIVE_TERM_NAMES) == STOCK_TERMS
    np.testing.assert_array_equal(_bits(terms), _bits(manual_terms.numpy()))
    np.testing.assert_array_equal(_bits(score), _bits(manual_score.numpy()))
    assert np.isfinite(terms).all() and np.abs(score).max() > 0.0


@pytest.mark.parametrize("seed", [2, 3])
def test_position_only_after_release(monkeypatch, seed):
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    with wp.ScopedDevice("cpu"):
        stock = _fake_objective(ReconBodyObjectiveConfig())
        baseline = _fake_objective(_config())
        objective = _fake_objective(_config(object_task_position_only_after_release=True))
        np.testing.assert_array_equal(objective.release_mask.numpy(), (np.arange(_FRAMES) >= _RELEASED_FROM))
        assert objective.terms.shape == (_WORLDS * (STOCK_TERMS + 1),)
        _, stock_score, _, _ = _evaluate(stock, seed)
        base_terms, base_score, _, _ = _evaluate(baseline, seed)
        launches = _record_launches(monkeypatch)
        terms, score, positions, quats = _evaluate(objective, seed)
        assert launches[-1] == "add_recon_body_object_task_after_release" and len(launches) == 4
    held = ~_RELEASED
    assert held.sum() == 4 and _RELEASED.sum() == 6
    # Not released (pre-contact, the both-off gap, contact, last contact): the original bits.
    np.testing.assert_array_equal(_bits(terms[held]), _bits(base_terms[held]))
    np.testing.assert_array_equal(_bits(score[held]), _bits(base_score[held]))
    # Released: the stock columns are unchanged and object_task is exp(-beta_pos |p - p_ref|) alone.
    np.testing.assert_array_equal(_bits(terms[:, :STOCK_TERMS]), _bits(base_terms[:, :STOCK_TERMS]))
    object_pos, _ = _object_rows(positions, quats)
    target_pos, _ = _targets(0)
    expected = np.exp(-_BETA_POS * np.linalg.norm(object_pos - target_pos.astype(np.float64), axis=-1))
    np.testing.assert_allclose(terms[_RELEASED, STOCK_TERMS], expected[_RELEASED], rtol=2.0e-6, atol=1.0e-7)
    assert np.all(terms[_RELEASED, STOCK_TERMS] > base_terms[_RELEASED, STOCK_TERMS])  # the rotation factor is gone
    # Score: the stock score plus frame_dt * weight * object_task, as in the original kernel.
    np.testing.assert_allclose(base_score, stock_score + _DT * _TASK_WEIGHT * base_terms[:, STOCK_TERMS], rtol=1.0e-6)
    np.testing.assert_allclose(score, stock_score + _DT * _TASK_WEIGHT * terms[:, STOCK_TERMS], rtol=1.0e-6)
    assert np.all(score[_RELEASED] > base_score[_RELEASED])


@pytest.mark.parametrize("position_only", [False, True])
def test_no_drop_term(monkeypatch, position_only):
    weight, margin = -50.0, 0.05
    config = _config(object_task_position_only_after_release=position_only, no_drop_weight=weight)
    with wp.ScopedDevice("cpu"):
        baseline = _fake_objective(_config(object_task_position_only_after_release=position_only))
        objective = _fake_objective(config)
        assert objective.term_names[-2:] == ("object_task", "object_no_drop")
        assert objective.terms.shape == (_WORLDS * (STOCK_TERMS + 2),)
        np.testing.assert_array_equal(objective.weights.numpy()[-2:], np.asarray([_TASK_WEIGHT, weight], np.float32))
        base_terms, base_score, _, _ = _evaluate(baseline, 4)
        launches = _record_launches(monkeypatch)
        terms, score, positions, _ = _evaluate(objective, 4)
        assert launches[-1] == "add_recon_body_object_task_after_release" and len(launches) == 4
    # Stock columns and object_task are those of the same objective without the no-drop term, bit for bit.
    np.testing.assert_array_equal(_bits(terms[:, : STOCK_TERMS + 1]), _bits(base_terms))
    target_pos, _ = _targets(0)
    rows = np.arange(_WORLDS) * _BODIES + _OBJECT
    z = positions[rows, 2].astype(np.float32)
    depth = np.maximum(np.float32(target_pos[:, 2]) - np.float32(margin) - z, np.float32(0.0))
    expected = np.where(_RELEASED, depth, 0.0)
    np.testing.assert_allclose(terms[:, STOCK_TERMS + 1], expected, rtol=1.0e-5, atol=2.0e-7)
    # Released worlds 4 (12 cm below its target), 7 (30 cm below; timestep 14 clamps to the last frame) and 9 (6 cm)
    # drop by depth - 5 cm; 5 (3 cm below), 6 (above) and 8 (level) do not; the not-released worlds never do.
    assert terms[4, STOCK_TERMS + 1] == pytest.approx(0.07, abs=1.0e-5)
    assert terms[7, STOCK_TERMS + 1] == pytest.approx(0.25, abs=1.0e-5)
    assert terms[9, STOCK_TERMS + 1] == pytest.approx(0.01, abs=1.0e-5)
    assert terms[[5, 6, 8], STOCK_TERMS + 1].tolist() == [0.0, 0.0, 0.0]
    assert np.all(terms[~_RELEASED, STOCK_TERMS + 1] == 0.0) and np.any(depth[~_RELEASED] > 0.0)
    # Score: the baseline score plus frame_dt * weight * drop; bit-identical where the drop is 0.
    np.testing.assert_allclose(score, base_score + _DT * weight * terms[:, STOCK_TERMS + 1], rtol=1.0e-5, atol=2.0e-6)
    no_drop = terms[:, STOCK_TERMS + 1] == 0.0
    np.testing.assert_array_equal(_bits(score[no_drop]), _bits(base_score[no_drop]))
    assert np.all(score[~no_drop] < base_score[~no_drop])


def test_set_weights_requires_the_object_no_drop_name():
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES

    stock = {name: float(index + 1) for index, name in enumerate(RECON_BODY_OBJECTIVE_TERM_NAMES)}
    with wp.ScopedDevice("cpu"):
        objective = _fake_objective(_config(no_drop_weight=-50.0))
        with pytest.raises(ValueError, match="exactly"):
            objective.set_weights({**stock, "object_task": 5.0})
        objective.set_weights({**stock, "object_task": 5.0, "object_no_drop": -20.0})
        np.testing.assert_array_equal(objective.weights.numpy()[-2:], np.asarray([5.0, -20.0], dtype=np.float32))
        position_only = _fake_objective(_config(object_task_position_only_after_release=True))
        with pytest.raises(ValueError, match="exactly"):
            position_only.set_weights({**stock, "object_task": 5.0, "object_no_drop": -20.0})


def test_release_options_refuse_to_run_without_the_mask():
    with wp.ScopedDevice("cpu"):
        objective = _fake_objective(_config(object_task_position_only_after_release=True), release_mask=None)
        state, timestep, terminated, _, _ = _state(5)
        with pytest.raises(ValueError, match="release mask"):
            objective.evaluate(state, timestep, terminated)
