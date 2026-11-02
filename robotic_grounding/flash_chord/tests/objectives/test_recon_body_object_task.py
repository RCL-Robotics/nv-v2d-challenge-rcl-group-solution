# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinates_hand object task term (``object_task``) of the ReconBody objective.

The reference below is a line-by-line numpy port of Coordinates_hand (CH) ``dexmachina/envs/reward_utils.py``
(``position_distance``, ``rotation_distance``), ``math_utils.py`` (``quat_mul``, ``quat_conjugate``; wxyz) and
``rewards.py`` ``compute_task_reward`` with ``multiply_task_rew=True``. With the term disabled (weight 0, the
default) the objective must be exactly the stock one: same term names, buffers, kernels and outputs.
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

STOCK_TERMS = 12


# --------------------------------------------------------------------------------------------------
# Coordinates_hand reference (float64 numpy, wxyz quaternions)
# --------------------------------------------------------------------------------------------------


def _ch_quat_conjugate(q):
    return np.concatenate((q[:, 0:1], -q[:, 1:]), axis=-1)


def _ch_quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1)


def _ch_rotation_distance(object_rot, target_rot):
    quat_diff = _ch_quat_mul(object_rot, _ch_quat_conjugate(target_rot))
    return 2.0 * np.arcsin(np.minimum(np.linalg.norm(quat_diff[:, 1:4], axis=-1), 1.0))


def _ch_position_distance(object_pos, target_pos):
    return np.linalg.norm(object_pos - target_pos, axis=-1)


def _ch_task_reward(obj_pos, obj_quat_wxyz, demo_pos, demo_quat_wxyz, pos_beta, rot_beta, arti_beta=20.0):
    obj_arti = np.zeros(obj_pos.shape[0])  # rigid objects: no articulation, demo articulation 0
    demo_arti = np.zeros(obj_pos.shape[0])
    pos_dist = _ch_position_distance(obj_pos, demo_pos)
    rot_dist = _ch_rotation_distance(obj_quat_wxyz, demo_quat_wxyz)
    arti_dist = (obj_arti - demo_arti) ** 2 / 2.0
    task_rew_weight = 1.0
    obj_pos_rew = np.exp(-pos_beta * pos_dist)
    obj_rot_rew = np.exp(-rot_beta * rot_dist)
    obj_arti_rew = np.exp(-arti_beta * arti_dist)
    return task_rew_weight * obj_pos_rew * obj_rot_rew * obj_arti_rew


def _xyzw_to_wxyz(q):
    return np.concatenate((q[:, 3:4], q[:, :3]), axis=-1)


def _random_unit_quats(rng, count):
    q = rng.normal(size=(count, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _axis_angle_xyzw(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    return np.concatenate((axis * math.sin(angle / 2.0), [math.cos(angle / 2.0)]))


def _quat_mul_xyzw(a, b):
    """Hamilton product of xyzw quaternions (float64)."""
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


def _poses(rng, count):
    """Random object/target poses (xyzw) plus edge cases: identical, tiny, ~pi and exactly-pi rotations."""
    object_pos = rng.uniform(-1.0, 1.0, size=(count, 3))
    target_pos = object_pos + rng.normal(scale=0.05, size=(count, 3))
    object_quat = _random_unit_quats(rng, count)
    target_quat = _random_unit_quats(rng, count)
    # identical pose
    target_pos[0], target_quat[0] = object_pos[0], object_quat[0]
    # small position and angle errors: 1 cm, 1e-3 rad
    for world, (distance, angle) in ((1, (0.01, 1.0e-3)), (2, (0.03, 0.1)), (3, (0.09, 0.5))):
        target_pos[world] = object_pos[world] + np.asarray([distance, 0.0, 0.0])
        target_quat[world] = _quat_mul_xyzw(object_quat[world], _axis_angle_xyzw([0.3, -0.5, 0.8], angle))
    # exactly and almost pi apart
    target_quat[4] = _quat_mul_xyzw(object_quat[4], _axis_angle_xyzw([1.0, 2.0, -1.0], math.pi))
    target_quat[5] = _quat_mul_xyzw(object_quat[5], _axis_angle_xyzw([0.0, 0.0, 1.0], math.pi - 1.0e-3))
    return object_pos, object_quat, target_pos, target_quat


# --------------------------------------------------------------------------------------------------
# Kernel-level checks
# --------------------------------------------------------------------------------------------------


def _launch_object_task(object_pos, object_quat, target_pos, target_quat, *, beta_pos, beta_rot, weight, frame_dt):
    import warp as wp

    from flash_chord.objectives.recon_body import add_recon_body_object_task

    world_count = object_pos.shape[0]
    bodies_per_world, object_body_id = 3, 2
    transforms = [wp.transform_identity()] * (world_count * bodies_per_world)
    for world in range(world_count):
        transforms[world * bodies_per_world + object_body_id] = wp.transform(
            wp.vec3(*object_pos[world].astype(np.float32)), wp.quat(*object_quat[world].astype(np.float32))
        )
    stock_terms = np.arange(world_count * STOCK_TERMS, dtype=np.float32) * 0.25 - 3.0
    score_in = np.linspace(-1.0, 1.0, world_count, dtype=np.float32)
    weights = np.zeros(STOCK_TERMS + 1, dtype=np.float32)
    weights[STOCK_TERMS] = weight
    with wp.ScopedDevice("cpu"):
        terms = wp.zeros(world_count * (STOCK_TERMS + 1), dtype=wp.float32)
        score = wp.array(score_in, dtype=wp.float32)
        wp.launch(
            add_recon_body_object_task,
            dim=world_count,
            inputs=[
                wp.array(transforms, dtype=wp.transform),
                wp.array(stock_terms, dtype=wp.float32),
                object_body_id,
                wp.array(target_pos.astype(np.float32), dtype=wp.vec3),
                wp.array(target_quat.astype(np.float32), dtype=wp.quat),
                bodies_per_world,
                beta_pos,
                beta_rot,
                wp.array(weights, dtype=wp.float32),
                frame_dt,
            ],
            outputs=[terms, score],
        )
        return stock_terms, score_in, terms.numpy().reshape(world_count, STOCK_TERMS + 1), score.numpy()


@pytest.mark.parametrize(("beta_pos", "beta_rot"), [(20.0, 5.0), (10.0, 1.0)])
def test_object_task_matches_coordinates_hand_task_reward(beta_pos, beta_rot):
    rng = np.random.default_rng(1234)
    object_pos, object_quat, target_pos, target_quat = _poses(rng, 64)
    stock_terms, score_in, terms, score = _launch_object_task(
        object_pos, object_quat, target_pos, target_quat, beta_pos=beta_pos, beta_rot=beta_rot, weight=5.0,
        frame_dt=0.02,
    )
    # CH sees the float32 poses the kernel sees.
    as32 = lambda value: value.astype(np.float32).astype(np.float64)  # noqa: E731
    expected = _ch_task_reward(
        as32(object_pos),
        _xyzw_to_wxyz(as32(object_quat)),
        as32(target_pos),
        _xyzw_to_wxyz(as32(target_quat)),
        beta_pos,
        beta_rot,
    )
    np.testing.assert_array_equal(terms[:, :STOCK_TERMS], stock_terms.reshape(-1, STOCK_TERMS))
    # Worlds 4-5 are (almost) pi apart: float32 asin next to 1 resolves the angle to ~1e-3 rad only (CH's float32
    # torch code has the same limit), so their tolerance is the angle tolerance times beta_rot.
    regular = np.ones(len(expected), dtype=bool)
    regular[[4, 5]] = False
    np.testing.assert_allclose(terms[regular, STOCK_TERMS], expected[regular], rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(terms[~regular, STOCK_TERMS], expected[~regular], rtol=2.0e-3 * beta_rot, atol=2.0e-6)
    np.testing.assert_allclose(score, score_in + 0.02 * 5.0 * terms[:, STOCK_TERMS], rtol=1.0e-6, atol=1.0e-7)
    assert terms[0, STOCK_TERMS] == pytest.approx(1.0, abs=1.0e-6)  # identical pose
    assert np.all(terms[:, STOCK_TERMS] > 0.0) and np.all(terms[:, STOCK_TERMS] <= 1.0)
    # The analytic edge cases: 1 cm / 1e-3 rad, 3 cm / 0.1 rad, 9 cm / 0.5 rad.
    for world, (distance, angle) in ((1, (0.01, 1.0e-3)), (2, (0.03, 0.1)), (3, (0.09, 0.5))):
        analytic = math.exp(-beta_pos * distance) * math.exp(-beta_rot * angle)
        assert terms[world, STOCK_TERMS] == pytest.approx(analytic, rel=1.0e-4)


def test_object_task_is_invariant_to_quaternion_sign():
    rng = np.random.default_rng(7)
    object_pos, object_quat, target_pos, target_quat = _poses(rng, 32)
    reference = _launch_object_task(
        object_pos, object_quat, target_pos, target_quat, beta_pos=20.0, beta_rot=5.0, weight=1.0, frame_dt=0.02
    )
    for object_sign, target_sign in ((-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)):
        flipped = _launch_object_task(
            object_pos,
            object_sign * object_quat,
            target_pos,
            target_sign * target_quat,
            beta_pos=20.0,
            beta_rot=5.0,
            weight=1.0,
            frame_dt=0.02,
        )
        np.testing.assert_array_equal(flipped[2], reference[2])
        np.testing.assert_array_equal(flipped[3], reference[3])
    # CH's own formula is sign invariant too (|vec(q1 q2*)| is).
    expected = _ch_rotation_distance(_xyzw_to_wxyz(object_quat), _xyzw_to_wxyz(target_quat))
    np.testing.assert_allclose(
        _ch_rotation_distance(_xyzw_to_wxyz(-object_quat), _xyzw_to_wxyz(target_quat)), expected, atol=1.0e-12
    )


def test_object_task_rotation_distance_is_the_geodesic_angle():
    import warp as wp

    from flash_chord.objectives.recon_body import object_task_rotation_distance

    @wp.kernel
    def _angles(a: wp.array(dtype=wp.quat), b: wp.array(dtype=wp.quat), out: wp.array(dtype=wp.float32)):
        i = wp.tid()
        out[i] = object_task_rotation_distance(a[i], b[i])

    rng = np.random.default_rng(3)
    edge_angles = [0.0, 1.0e-4, 1.0e-3, 0.01, 0.5, 1.0, 2.0, 3.0, math.pi - 1.0e-2, math.pi]
    angles = np.concatenate((edge_angles, rng.uniform(0.0, math.pi, 22)))
    a = _random_unit_quats(rng, angles.size)
    axes = rng.normal(size=(angles.size, 3))
    b = np.stack([_quat_mul_xyzw(a[i], _axis_angle_xyzw(axes[i], angles[i])) for i in range(angles.size)])
    with wp.ScopedDevice("cpu"):
        out = wp.zeros(angles.size, dtype=wp.float32)
        wp.launch(
            _angles,
            dim=angles.size,
            inputs=[wp.array(a.astype(np.float32), dtype=wp.quat), wp.array(b.astype(np.float32), dtype=wp.quat)],
            outputs=[out],
        )
        values = out.numpy()
    a64, b64 = (_xyzw_to_wxyz(value.astype(np.float32).astype(np.float64)) for value in (a, b))
    ch = _ch_rotation_distance(a64, b64)
    # float32 asin loses accuracy only next to pi, where exp(-beta * angle) is ~1e-7 or smaller.
    near_pi = angles > math.pi - 0.05
    np.testing.assert_allclose(values[~near_pi], angles[~near_pi], atol=2.0e-5)
    np.testing.assert_allclose(values[~near_pi], ch[~near_pi], atol=2.0e-5)
    np.testing.assert_allclose(values[near_pi], angles[near_pi], atol=2.0e-3)
    assert np.all(values >= 0.0) and np.all(values <= np.float32(math.pi))


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------


def test_default_config_is_the_stock_objective():
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjectiveConfig

    config = ReconBodyObjectiveConfig()
    assert config.object_task_weight == 0.0
    assert (config.object_task_position_beta, config.object_task_rotation_beta) == (20.0, 5.0)
    assert not config.object_task_enabled
    assert config.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES
    assert len(config.term_names) == STOCK_TERMS
    assert config.term_weights == config.weights
    assert len(config.weights) == STOCK_TERMS


def test_enabled_config_appends_object_task_last():
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjectiveConfig

    config = ReconBodyObjectiveConfig(
        object_task_weight=5.0, object_task_position_beta=10.0, object_task_rotation_beta=1.0
    )
    assert config.object_task_enabled
    assert config.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES + ("object_task",)
    assert config.term_weights == config.weights + (5.0,)
    assert len(config.weights) == STOCK_TERMS  # the stock weights tuple is unchanged


@pytest.mark.parametrize(
    "kwargs",
    [
        {"object_task_weight": math.nan},
        {"object_task_weight": math.inf},
        {"object_task_position_beta": math.inf},
        {"object_task_rotation_beta": math.nan},
        {"object_task_position_beta": -1.0},
        {"object_task_rotation_beta": -0.5},
    ],
)
def test_invalid_object_task_parameters_are_rejected(kwargs):
    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    with pytest.raises(ValueError, match="object task"):
        ReconBodyObjectiveConfig(**kwargs)


# --------------------------------------------------------------------------------------------------
# Objective-level: evaluate() through a small CPU objective
# --------------------------------------------------------------------------------------------------

_WORLDS = 6
_FRAMES = 11
_BODIES = 5  # 0 pelvis, 1-2 wrists, 3 object, 4 spare
_Q = 10
_SLOTS = 4  # two links per hand


def _fake_objective(config, seed: int = 0):
    """A ReconBodyObjective with real device buffers and kernels and stand-in runtime dependencies."""
    import warp as wp

    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjective
    from flash_chord.objectives.wrench import friction_cone_angles, sample_wrench_basis

    rng = np.random.default_rng(seed)
    f32 = lambda values: wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32)  # noqa: E731
    i32 = lambda values: wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32)  # noqa: E731
    reference_joint_q = rng.normal(scale=0.3, size=(_FRAMES, _Q)).astype(np.float32)
    reference_joint_q[:, 3:7] = _random_unit_quats(rng, _FRAMES)
    wrist_pos = rng.normal(scale=0.3, size=(_FRAMES * 2, 3)).astype(np.float32)
    wrist_quat = _random_unit_quats(rng, _FRAMES * 2).astype(np.float32)
    contact_pos = rng.normal(scale=0.05, size=(_WORLDS * _SLOTS, 3)).astype(np.float32)
    contact_dir = rng.normal(size=(_WORLDS * _SLOTS, 3))
    contact_dir = (contact_dir / np.linalg.norm(contact_dir, axis=-1, keepdims=True)).astype(np.float32)
    num_basis, num_edges = 16, 8
    basis = sample_wrench_basis(num_basis, 1.0, 0)
    cone_cos, cone_sin = friction_cone_angles(num_edges)
    labels = (rng.uniform(size=(_FRAMES, 2)) > 0.3).astype(np.float32)
    target_quat = _random_unit_quats(rng, _WORLDS).astype(np.float32)
    command = SimpleNamespace(
        object_radius=f32(np.full(_WORLDS, 0.1)),
        body_target_pos_w=wp.array(rng.normal(scale=0.5, size=(_WORLDS, 3)).astype(np.float32), dtype=wp.vec3),
        body_target_quat_w=wp.array(target_quat, dtype=wp.quat),
    )
    term_count = len(config.term_names)
    return ReconBodyObjective(
        config=config,
        world_count=_WORLDS,
        frame_dt=0.02,
        num_frames=_FRAMES,
        num_hands=2,
        num_bodies=1,
        num_basis=num_basis,
        num_edges=num_edges,
        num_joints=4,
        bodies_per_world=_BODIES,
        q_per_world=_Q,
        pelvis_body_id=0,
        object_body_id=3,
        episode_start_frame=i32(rng.integers(0, 4, size=_WORLDS)),
        reference_joint_q_offset=f32(rng.normal(scale=0.01, size=_WORLDS * _Q)),
        reference=SimpleNamespace(
            joint_q=wp.array(reference_joint_q, dtype=wp.float32),
            wrist_pos_w=wp.array(wrist_pos, dtype=wp.vec3),
            wrist_quat_w=wp.array(wrist_quat, dtype=wp.quat),
        ),
        command=command,
        contact=SimpleNamespace(
            contact_pos_o=wp.array(contact_pos, dtype=wp.vec3),
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
        contact_active=f32(labels.reshape(-1)),
        wrench_basis=wp.array(basis, dtype=wp.spatial_vector),
        cone_cos=f32(cone_cos),
        cone_sin=f32(cone_sin),
        wrench_contact_enabled=wp.ones(_SLOTS, dtype=wp.int32),
        current_wrench_support=wp.zeros(_WORLDS * 2 * num_basis, dtype=wp.float32),
        force_closure=wp.zeros(_WORLDS, dtype=wp.float32),
        weights=wp.array(config.term_weights, dtype=wp.float32),
        terms=wp.zeros(_WORLDS * term_count, dtype=wp.float32),
        score=wp.zeros(_WORLDS, dtype=wp.float32),
        stock_terms=(
            wp.zeros(_WORLDS * len(RECON_BODY_OBJECTIVE_TERM_NAMES), dtype=wp.float32)
            if config.object_task_enabled
            else None
        ),
    )


def _state(seed: int):
    import warp as wp

    rng = np.random.default_rng(seed)
    positions = rng.normal(scale=0.5, size=(_WORLDS * _BODIES, 3)).astype(np.float32)
    quats = _random_unit_quats(rng, _WORLDS * _BODIES).astype(np.float32)
    transforms = [wp.transform(wp.vec3(*positions[i]), wp.quat(*quats[i])) for i in range(_WORLDS * _BODIES)]
    state = SimpleNamespace(
        body_q=wp.array(transforms, dtype=wp.transform),
        joint_q=wp.array(rng.normal(scale=0.3, size=_WORLDS * _Q).astype(np.float32), dtype=wp.float32),
    )
    timestep = wp.array(rng.integers(0, _FRAMES + 2, size=_WORLDS).astype(np.int32), dtype=wp.int32)
    terminated = wp.array((rng.uniform(size=_WORLDS) > 0.7).astype(np.int32), dtype=wp.int32)
    return state, timestep, terminated, positions, quats


def _evaluate(objective, seed: int):
    state, timestep, terminated, positions, quats = _state(seed)
    objective.evaluate(state, timestep, terminated)
    return objective.terms.numpy().copy(), objective.score.numpy().copy(), positions, quats


def _record_launches(monkeypatch):
    import warp as wp

    launches = []
    original = wp.launch

    def recording(kernel, *args, **kwargs):
        launches.append(kernel.key)
        return original(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", recording)
    return launches


def test_disabled_objective_is_exactly_the_stock_kernel_path(monkeypatch):
    import warp as wp

    from flash_chord.objectives.recon_body import (
        RECON_BODY_OBJECTIVE_TERM_NAMES,
        ReconBodyObjectiveConfig,
        compose_recon_body_objective,
    )

    with wp.ScopedDevice("cpu"):
        objective = _fake_objective(ReconBodyObjectiveConfig())
        assert objective.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES
        assert objective.stock_terms is None
        assert objective.terms.shape == (_WORLDS * STOCK_TERMS,)
        assert objective.weights.shape == (STOCK_TERMS,)
        launches = _record_launches(monkeypatch)
        state, timestep, terminated, _, _ = _state(11)
        objective.evaluate(state, timestep, terminated)
        assert launches == ["batched_wrench_support", "reduce_recon_body_force_closure", "compose_recon_body_objective"]
        terms, score = objective.terms.numpy().copy(), objective.score.numpy().copy()

        # The same stock kernel launched by hand, with the objective's own force closure, gives identical bits.
        manual_terms = wp.zeros(_WORLDS * STOCK_TERMS, dtype=wp.float32)
        manual_score = wp.zeros(_WORLDS, dtype=wp.float32)
        o = objective
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
                o.action.sonic_q_ids.shape[0], 0.3, 0.4, 1.0, 0.2, 0.2, 0.4,
                wp.array(ReconBodyObjectiveConfig().weights, dtype=wp.float32), o.frame_dt,
            ],
            outputs=[manual_terms, manual_score],
        )
    np.testing.assert_array_equal(terms, manual_terms.numpy())
    np.testing.assert_array_equal(score, manual_score.numpy())
    assert np.isfinite(terms).all() and np.abs(score).max() > 0.0


def test_enabled_objective_adds_only_the_object_task_term(monkeypatch):
    import warp as wp

    from flash_chord.objectives.recon_body import ReconBodyObjectiveConfig

    enabled_config = ReconBodyObjectiveConfig(object_task_weight=5.0)
    with wp.ScopedDevice("cpu"):
        disabled = _fake_objective(ReconBodyObjectiveConfig())
        enabled = _fake_objective(enabled_config)
        assert enabled.term_names[-1] == "object_task" and len(enabled.term_names) == STOCK_TERMS + 1
        assert enabled.terms.shape == (_WORLDS * (STOCK_TERMS + 1),)
        launches = _record_launches(monkeypatch)
        for seed in (21, 22, 23):
            base_terms, base_score, _, _ = _evaluate(disabled, seed)
            del launches[:]
            terms, score, positions, quats = _evaluate(enabled, seed)
            assert launches[-1] == "add_recon_body_object_task" and len(launches) == 4
            terms = terms.reshape(_WORLDS, STOCK_TERMS + 1)
            np.testing.assert_array_equal(terms[:, :STOCK_TERMS], base_terms.reshape(_WORLDS, STOCK_TERMS))
            object_rows = np.arange(_WORLDS) * _BODIES + 3
            target_pos = enabled.command.body_target_pos_w.numpy()
            target_quat = enabled.command.body_target_quat_w.numpy()
            expected = _ch_task_reward(
                positions[object_rows].astype(np.float64),
                _xyzw_to_wxyz(quats[object_rows].astype(np.float64)),
                target_pos.astype(np.float64),
                _xyzw_to_wxyz(target_quat.astype(np.float64)),
                20.0,
                5.0,
            )
            np.testing.assert_allclose(terms[:, STOCK_TERMS], expected, rtol=2.0e-5, atol=2.0e-6)
            added = base_score + 0.02 * 5.0 * terms[:, STOCK_TERMS]
            np.testing.assert_allclose(score, added, rtol=1.0e-6, atol=1.0e-7)


def test_set_weights_requires_exactly_the_active_term_names():
    import warp as wp

    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjectiveConfig

    stock = {name: float(index + 1) for index, name in enumerate(RECON_BODY_OBJECTIVE_TERM_NAMES)}
    with wp.ScopedDevice("cpu"):
        disabled = _fake_objective(ReconBodyObjectiveConfig())
        disabled.set_weights(stock)
        np.testing.assert_array_equal(disabled.weights.numpy(), np.arange(1, STOCK_TERMS + 1, dtype=np.float32))
        with pytest.raises(ValueError, match="exactly"):
            disabled.set_weights({**stock, "object_task": 5.0})
        enabled = _fake_objective(ReconBodyObjectiveConfig(object_task_weight=1.0))
        np.testing.assert_array_equal(enabled.weights.numpy()[-1], 1.0)
        with pytest.raises(ValueError, match="exactly"):
            enabled.set_weights(stock)
        enabled.set_weights({**stock, "object_task": 7.5})
        np.testing.assert_array_equal(enabled.weights.numpy()[-1], 7.5)
        np.testing.assert_array_equal(enabled.weights.numpy()[:-1], np.arange(1, STOCK_TERMS + 1, dtype=np.float32))


def test_object_task_rejects_articulated_objects():
    from flash_chord.objectives.recon_body import ReconBodyObjective, ReconBodyObjectiveConfig
    from flash_chord.runtime.actions import RegularizedAction
    from flash_chord.runtime.actions.sonic import SonicJointResidualAction

    class _Action(SonicJointResidualAction, RegularizedAction):
        def __init__(self):
            pass

    command = SimpleNamespace(layout=SimpleNamespace(num_objects=1, num_bodies=1, num_articulations=1))
    contact = SimpleNamespace(layout=SimpleNamespace(sides=("left", "right")))
    with pytest.raises(ValueError, match="rigid objects only"):
        ReconBodyObjective.build(
            None, None, None, None, command, contact, _Action(), object(), object(), 1, 0.02,
            config=ReconBodyObjectiveConfig(object_task_weight=5.0),
        )


def test_dataclass_fields_append_only():
    """New fields come after every stock field, so positional and keyword construction are unchanged."""
    from flash_chord.objectives.recon_body import ReconBodyObjective, ReconBodyObjectiveConfig

    names = [field.name for field in dataclasses.fields(ReconBodyObjectiveConfig)]
    assert names[-6:-3] == ["object_task_weight", "object_task_position_beta", "object_task_rotation_beta"]
    # The post-release options (tests/objectives/test_recon_body_object_task_release.py) come after object_task.
    assert names[-3:] == ["object_task_position_only_after_release", "no_drop_weight", "no_drop_margin_m"]
    assert names[:-6][-1] == "sides" and len(names) == 31
    objective_names = [field.name for field in dataclasses.fields(ReconBodyObjective)]
    assert objective_names[-3:] == ["score", "stock_terms", "release_mask"]
