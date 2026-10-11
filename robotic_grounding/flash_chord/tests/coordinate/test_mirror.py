# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""URDF mirror signs, resolved agent layout, observation mirror and actor mirror equivariance."""

from __future__ import annotations

import numpy as np
import pytest

MIRROR = np.diag((1.0, -1.0, 1.0))


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


def _ranges(*pairs):
    indices = []
    for start, end in pairs:
        indices.extend(range(start, end + 1))
    return tuple(indices)


# ------------------------------------------------------------------------------------------------ (1)


def test_urdf_mirror_signs_match_tables_limits_and_defaults():
    from flash_chord.embodiments.base import resolve_joint_controls
    from flash_chord.embodiments.g1_dex3 import G1Dex3Config
    from flash_chord.training.ppo.coordinate.layout import (
        BODY_CANONICAL,
        BODY_MIRROR_PERM,
        BODY_MIRROR_SIGNS,
        HAND_CANONICAL,
        HAND_MIRROR_SIGNS,
        default_urdf_path,
        derive_mirror_signs,
        parse_urdf_joints,
    )

    signs = derive_mirror_signs(default_urdf_path())
    assert tuple(signs[f"left_{name}_joint"] for name in HAND_CANONICAL) == HAND_MIRROR_SIGNS
    body = []
    for index, name in enumerate(BODY_CANONICAL):
        key = BODY_CANONICAL[BODY_MIRROR_PERM[index]] if name.startswith("right_") else name
        body.append(signs[key])
    assert tuple(body) == BODY_MIRROR_SIGNS
    assert len(signs) == 23

    joints = parse_urdf_joints(default_urdf_path())
    for name, sign in signs.items():
        left = joints[name]
        right = joints[name.replace("left_", "right_", 1)]
        np.testing.assert_array_equal(left.axis, right.axis)
        if sign < 0:
            np.testing.assert_allclose((right.lower, right.upper), (-left.upper, -left.lower), atol=1.0e-6)
        else:
            np.testing.assert_allclose((right.lower, right.upper), (left.lower, left.upper), atol=1.0e-6)

    config = G1Dex3Config()
    names = tuple(name for name, joint in joints.items() if joint.joint_type == "revolute")
    defaults = {
        joint: spec.default_pos
        for joint, spec in zip(names, resolve_joint_controls(names, config.default_joint, config.joint_overrides))
    }
    assert defaults["left_shoulder_roll_joint"] == pytest.approx(0.2)
    for name, sign in signs.items():
        partner = name.replace("left_", "right_", 1)
        assert defaults[partner] == pytest.approx(sign * defaults[name]), name


def test_derive_mirror_signs_rejects_asymmetric_urdf(tmp_path):
    from flash_chord.training.ppo.coordinate.layout import default_urdf_path, derive_mirror_signs

    text = default_urdf_path().read_text()
    broken = text.replace('lower="-1.04719755" upper="0.72431163"', 'lower="-1.0" upper="0.72431163"', 1)
    assert broken != text
    path = tmp_path / "broken.urdf"
    path.write_text(broken)
    with pytest.raises(ValueError, match="limits"):
        derive_mirror_signs(path)


# ------------------------------------------------------------------------------------------------ (2)


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def _forward_kinematics(joints, configuration, link):
    by_child = {joint.child: joint for joint in joints.values()}
    rotation = np.eye(3)
    position = np.zeros(3)
    chain = []
    while link in by_child:
        joint = by_child[link]
        chain.append(joint)
        link = joint.parent
    assert link == "pelvis"
    for joint in reversed(chain):
        local = _rpy_matrix(joint.rpy)
        if joint.joint_type in ("revolute", "continuous"):
            local = local @ _axis_angle(joint.axis, configuration.get(joint.name, 0.0))
        position = position + rotation @ joint.xyz
        rotation = rotation @ local
    return position, rotation


def test_numpy_forward_kinematics_palms_are_sagittal_mirrors():
    from flash_chord.training.ppo.coordinate.layout import (
        HAND_CANONICAL,
        HAND_MIRROR_SIGNS,
        default_urdf_path,
        parse_urdf_joints,
    )

    joints = parse_urdf_joints(default_urdf_path())
    rng = np.random.default_rng(0)
    for _ in range(20):
        left = {}
        right = {}
        for name, sign in zip(HAND_CANONICAL, HAND_MIRROR_SIGNS, strict=True):
            joint = joints[f"left_{name}_joint"]
            value = rng.uniform(joint.lower, joint.upper)
            left[joint.name] = value
            right[f"right_{name}_joint"] = sign * value
        p_left, r_left = _forward_kinematics(joints, left, "left_hand_palm_link")
        p_right, r_right = _forward_kinematics(joints, right, "right_hand_palm_link")
        np.testing.assert_allclose(p_right, MIRROR @ p_left, atol=1.0e-4)
        np.testing.assert_allclose(r_right, MIRROR @ r_left @ MIRROR, atol=1.0e-4)


def test_numpy_forward_kinematics_full_body_mirror_matches_joint_mirror():
    from flash_chord.training.ppo.coordinate.layout import default_urdf_path, parse_urdf_joints

    layout = _layout()
    permutation, signs = layout.observation_mirror
    joints = parse_urdf_joints(default_urdf_path())
    from flash_chord.embodiments.g1_dex3 import G1_BODY_JOINT_NAMES, G1_DEX3_JOINT_NAMES

    scalar = G1_BODY_JOINT_NAMES[:22] + G1_DEX3_JOINT_NAMES[:7] + G1_BODY_JOINT_NAMES[22:] + G1_DEX3_JOINT_NAMES[7:]
    joint_offset = 33
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = np.array([rng.uniform(joints[name].lower, joints[name].upper) for name in scalar])
        observation = np.zeros(layout.observation_dim)
        observation[joint_offset : joint_offset + 43] = q
        mirrored = observation[np.asarray(permutation)] * np.asarray(signs)
        q_mirror = mirrored[joint_offset : joint_offset + 43]
        base = dict(zip(scalar, q, strict=True))
        mirror = dict(zip(scalar, q_mirror, strict=True))
        for link, partner in (
            ("left_hand_palm_link", "right_hand_palm_link"),
            ("right_hand_palm_link", "left_hand_palm_link"),
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            ("torso_link", "torso_link"),
        ):
            p, r = _forward_kinematics(joints, base, link)
            p_mirror, r_mirror = _forward_kinematics(joints, mirror, partner)
            np.testing.assert_allclose(p_mirror, MIRROR @ p, atol=1.0e-4)
            np.testing.assert_allclose(r_mirror, MIRROR @ r @ MIRROR, atol=1.0e-4)


# ------------------------------------------------------------------------------------------------ (3)


def test_layout_matches_appendix_a():
    layout = _layout()
    x_left, x_right, x_body = layout.actor_inputs
    assert layout.agent_names == ("left", "right", "body")
    assert layout.weight_group == ("hand", "hand", "body")
    assert (layout.observation_dim, layout.action_dim) == (486, 43)
    assert x_left.indices == _ranges(
        (3, 5), (12, 17), (21, 23), (48, 61), (91, 104), (161, 174), (204, 217), (247, 260), (278, 280),
        (284, 286), (290, 292), (299, 304), (311, 316), (323, 337), (372, 385), (415, 428), (458, 471),
        (24, 32), (347, 356),
    )
    assert set(x_left.signs) == {1.0}
    assert x_right.indices == _ranges(
        (0, 2), (6, 11), (18, 20), (62, 75), (105, 118), (175, 188), (218, 231), (261, 277), (281, 283),
        (287, 289), (293, 298), (305, 310), (317, 322), (338, 346), (386, 399), (429, 442), (472, 485),
        (24, 32), (347, 356),
    )
    assert sum(sign < 0 for sign in x_right.signs) == 109
    assert x_body.indices == _ranges(
        (33, 47), (76, 90), (146, 160), (189, 203), (232, 246), (357, 371), (400, 414), (443, 457),
        (119, 145), (0, 5), (24, 32), (347, 356),
    )
    assert set(x_body.signs) == {1.0}
    assert (x_left.dim, x_right.dim, x_body.dim) == (179, 179, 172)
    d_left, d_right = layout.hand_descriptors
    assert d_left.indices == _ranges((329, 337), (48, 61), (91, 104), (458, 471))
    assert d_right.indices == _ranges((338, 346), (62, 75), (105, 118), (472, 485))
    from flash_chord.training.ppo.coordinate.layout import HAND_MIRROR_SIGNS, ROT6D_SIGN, VEC_SIGN

    assert d_right.signs == VEC_SIGN + ROT6D_SIGN + HAND_MIRROR_SIGNS * 3
    assert layout.body_descriptor.indices == _ranges((33, 47), (76, 90), (443, 457))
    mirrored = layout.body_descriptor_mirrored
    assert mirrored.indices[:15] == (39, 40, 41, 42, 43, 44, 33, 34, 35, 36, 37, 38, 45, 46, 47)
    assert mirrored.signs[:15] == (1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1)
    assert mirrored.indices[15:30] == tuple(index + 43 for index in mirrored.indices[:15])
    assert mirrored.indices[30:] == tuple(index + 410 for index in mirrored.indices[:15])
    assert mirrored.signs[15:30] == mirrored.signs[:15] == mirrored.signs[30:]

    assert layout.action_indices == (
        (11, 15, 19, 21, 23, 25, 27, 31, 37, 41, 30, 36, 29, 35),
        (12, 16, 20, 22, 24, 26, 28, 34, 40, 42, 33, 39, 32, 38),
        (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8),
    )
    assert layout.action_signs[0] == (1.0,) * 14
    assert layout.action_signs[1] == HAND_MIRROR_SIGNS
    assert layout.action_signs[2] == (1.0,) * 15
    assert sorted(index for agent in layout.action_indices for index in agent) == list(range(43))
    for agent, indices in enumerate(layout.action_indices):
        assert all(layout.action_group[index] == agent for index in indices)

    used = set()
    for view in (*layout.actor_inputs, *layout.hand_descriptors, layout.body_descriptor):
        used |= set(view.indices)
    assert used == set(range(486))


def test_observation_mirror_is_a_signed_involution_consistent_with_views():
    from flash_chord.training.ppo.coordinate.layout import mirror_observation

    layout = _layout()
    permutation, signs = (np.asarray(value) for value in layout.observation_mirror)
    assert sorted(permutation.tolist()) == list(range(486))
    np.testing.assert_array_equal(permutation[permutation], np.arange(486))
    np.testing.assert_array_equal(signs * signs[permutation], np.ones(486))
    observation = np.random.default_rng(3).normal(size=(5, 486))
    np.testing.assert_array_equal(mirror_observation(layout, mirror_observation(layout, observation)), observation)
    mirrored = mirror_observation(layout, observation)
    x_left, x_right, _ = layout.actor_inputs

    def gather(view, value):
        return value[:, np.asarray(view.indices)] * np.asarray(view.signs)

    np.testing.assert_array_equal(gather(x_left, mirrored), gather(x_right, observation))
    np.testing.assert_array_equal(gather(x_right, mirrored), gather(x_left, observation))
    d_left, d_right = layout.hand_descriptors
    np.testing.assert_array_equal(gather(d_left, mirrored), gather(d_right, observation))
    np.testing.assert_array_equal(
        gather(layout.body_descriptor, mirrored), gather(layout.body_descriptor_mirrored, observation)
    )
    # Phase is kept, the object position is reflected and the wrist blocks swap sides.
    assert permutation[356] == 356 and signs[356] == 1.0
    assert (permutation[24:27].tolist(), signs[24:27].tolist()) == ([24, 25, 26], [1.0, -1.0, 1.0])
    assert permutation[0:6].tolist() == [3, 4, 5, 0, 1, 2]


def test_drop_world_terms_and_digest():
    from flash_chord.training.ppo.coordinate.layout import layout_digest

    reflect = _layout()
    drop = _layout(hand_world_terms="drop")
    assert [view.dim for view in drop.actor_inputs] == [170, 170, 172]
    removed_left = set(reflect.actor_inputs[0].indices) - set(drop.actor_inputs[0].indices)
    removed_right = set(reflect.actor_inputs[1].indices) - set(drop.actor_inputs[1].indices)
    assert removed_left == set(_ranges((278, 280), (284, 286), (290, 292)))
    assert removed_right == set(_ranges((275, 277), (281, 283), (287, 289)))
    assert layout_digest(reflect) == layout_digest(_layout())
    assert layout_digest(reflect) != layout_digest(drop)
    assert hash(reflect) == hash(_layout())
    assert len(layout_digest(reflect)) == 64


def test_build_layout_rejects_foreign_schema():
    from flash_chord.training.ppo.config import CoordinateConfig
    from flash_chord.training.ppo.coordinate.layout import build_layout

    schema = _policy_schema()
    schema["observation"]["blocks"][0]["name"] = "wrist_position_pelvis_m[left,right]"
    with pytest.raises(ValueError, match="exact ReconBody observation blocks"):
        build_layout(schema, CoordinateConfig())
    schema = _policy_schema()
    schema["action"]["blocks"][0]["name"] = schema["action"]["blocks"][0]["name"].replace("waist_yaw", "waist_yaw2")
    with pytest.raises(ValueError):
        build_layout(schema, CoordinateConfig())


def test_coordinate_config_validation_and_budget():
    from flash_chord.training.ppo.config import CoordinateConfig

    config = CoordinateConfig(agents=["left", "right", "body"], trunk_dims=[80, 40])
    assert config.agents == ("left", "right", "body") and config.trunk_dims == (80, 40)
    hash(config)
    with pytest.raises(ValueError, match="agents"):
        CoordinateConfig(agents=("left", "right"))
    with pytest.raises(ValueError, match="ratio_mode"):
        CoordinateConfig(ratio_mode="joint")
    with pytest.raises(ValueError, match="value_mode"):
        CoordinateConfig(value_mode="shared")
    with pytest.raises(ValueError, match="hand_world_terms"):
        CoordinateConfig(hand_world_terms="keep")
    with pytest.raises(ValueError, match="trunk_dims"):
        CoordinateConfig(trunk_dims=(80, 40, 20))
    with pytest.raises(ValueError, match="stock actor layer 0"):
        CoordinateConfig(partner_dim=200)
    with pytest.raises(ValueError, match="stock actor layer 1"):
        CoordinateConfig(message_dim=120)
    with pytest.raises(ValueError, match="stock actor layer 2"):
        CoordinateConfig(trunk_dims=(90, 40))
    with pytest.raises(ValueError, match="stock actor layer 3"):
        CoordinateConfig(trunk_dims=(80, 43))
    CoordinateConfig(mirror_body_message=False, partner_dim=245)
    with pytest.raises(ValueError, match="stock actor layer 0"):
        CoordinateConfig(mirror_body_message=True, partner_dim=245)


# ------------------------------------------------------------------------------------------------ (4)


def _network(layout, coordinate=None, network=None):
    from flash_chord.training.ppo.config import CoordinateConfig, NetworkConfig
    from flash_chord.training.ppo.coordinate.network import CoordinatedActorCritic

    return CoordinatedActorCritic(
        action_dim=43,
        layout=layout,
        coordinate=coordinate or CoordinateConfig(),
        network=network or NetworkConfig(initial_noise_std=0.5),
    )


def _random_parameters(jax, parameters, scale=0.3, seed=11):
    leaves, tree = jax.tree.flatten(parameters)
    rng = np.random.default_rng(seed)
    return jax.tree.unflatten(
        tree, [np.asarray(leaf) + scale * rng.standard_normal(np.shape(leaf)).astype(np.float32) for leaf in leaves]
    )


def test_network_parameter_names_shapes_and_initialization():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from flash_chord.training.ppo.config import CoordinateConfig, NetworkConfig

    layout = _layout()
    model = _network(layout)
    parameters = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 486)))["params"]
    shapes = {
        name: {key: tuple(value.shape) for key, value in leaf.items()} if hasattr(leaf, "items") else tuple(leaf.shape)
        for name, leaf in parameters.items()
    }
    expected = {
        "hand_encoder": {"kernel": (179, 96), "bias": (96,)},
        "body_encoder": {"kernel": (172, 96), "bias": (96,)},
        "hand_partner": {"kernel": (51, 64), "bias": (64,)},
        "body_partner": {"kernel": (45, 64), "bias": (64,)},
        "hand_message": {"kernel": (64, 64), "bias": (64,)},
        "body_message": {"kernel": (64, 64), "bias": (64,)},
        "hand_trunk_0": {"kernel": (160, 80), "bias": (80,)},
        "hand_trunk_1": {"kernel": (80, 40), "bias": (40,)},
        "body_trunk_0": {"kernel": (160, 80), "bias": (80,)},
        "body_trunk_1": {"kernel": (80, 40), "bias": (40,)},
        "hand_head": {"kernel": (40, 14), "bias": (14,)},
        "body_head": {"kernel": (40, 15), "bias": (15,)},
        "log_std": (43,),
        "critic_0": {"kernel": (486, 1024), "bias": (1024,)},
        "critic_1": {"kernel": (1024, 512), "bias": (512,)},
        "critic_2": {"kernel": (512, 256), "bias": (256,)},
        "critic_3": {"kernel": (256, 128), "bias": (128,)},
        "value": {"kernel": (128, 3), "bias": (3,)},
    }
    assert shapes == expected
    np.testing.assert_allclose(np.asarray(parameters["log_std"]), np.log(0.5), rtol=1.0e-6)
    for name in ("hand_head", "body_head"):
        assert not np.any(np.asarray(parameters[name]["kernel"])) and not np.any(np.asarray(parameters[name]["bias"]))
    assert all(not np.any(np.asarray(leaf["bias"])) for name, leaf in parameters.items() if name != "log_std")
    observation = jax.random.normal(jax.random.PRNGKey(1), (7, 486))
    policy, value = model.apply({"params": parameters}, observation)
    assert value.shape == (7, 3)
    np.testing.assert_array_equal(np.asarray(policy.mean()), np.zeros((7, 43)))
    assert policy.stddev().shape == (7, 43)

    drop = _layout(hand_world_terms="drop")
    drop_parameters = _network(drop, CoordinateConfig(hand_world_terms="drop")).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 486))
    )["params"]
    assert drop_parameters["hand_encoder"]["kernel"].shape == (170, 96)
    gain = _network(layout, network=NetworkConfig(actor_output_gain=0.01)).init(
        jax.random.PRNGKey(0), jnp.zeros((1, 486))
    )["params"]
    assert np.any(np.asarray(gain["hand_head"]["kernel"]))


def test_actor_mean_function_matches_module_and_scatter_signs():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from flash_chord.training.ppo.config import CoordinateConfig
    from flash_chord.training.ppo.coordinate.network import CoordinatedActorCritic, actor_mean

    layout = _layout()
    model = _network(layout)
    parameters = _random_parameters(jax, model.init(jax.random.PRNGKey(0), jnp.zeros((1, 486))))
    observation = jax.random.normal(jax.random.PRNGKey(2), (9, 486))
    with jax.default_matmul_precision("highest"):
        policy, _ = model.apply(parameters, observation)
        actor_only = {
            key: value for key, value in parameters["params"].items() if not key.startswith(("critic", "value"))
        }
        mean = actor_mean(actor_only, layout, CoordinateConfig(), observation)
        left, right, body = model.apply(parameters, observation, method=CoordinatedActorCritic.agent_means)
    np.testing.assert_allclose(np.asarray(mean), np.asarray(policy.mean()), atol=1.0e-6)
    mean = np.asarray(mean)
    for agent, canonical in enumerate((left, right, body)):
        indices = np.asarray(layout.action_indices[agent])
        signs = np.asarray(layout.action_signs[agent])
        np.testing.assert_array_equal(mean[:, indices], signs * np.asarray(canonical))


@pytest.mark.parametrize("partner_message", [True, False])
def test_hand_agents_are_mirror_equivariant(partner_message):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from flash_chord.training.ppo.config import CoordinateConfig
    from flash_chord.training.ppo.coordinate.layout import HAND_MIRROR_SIGNS, mirror_observation

    layout = _layout()
    coordinate = CoordinateConfig(partner_message=partner_message)
    model = _network(layout, coordinate)
    parameters = _random_parameters(jax, model.init(jax.random.PRNGKey(0), jnp.zeros((1, 486))), scale=0.5)
    observation = np.random.default_rng(5).standard_normal((64, 486)).astype(np.float32) * 1.5
    with jax.default_matmul_precision("highest"):
        action = np.asarray(model.apply(parameters, jnp.asarray(observation))[0].mean())
        mirrored = np.asarray(model.apply(parameters, jnp.asarray(mirror_observation(layout, observation)))[0].mean())
    left = np.asarray(layout.action_indices[0])
    right = np.asarray(layout.action_indices[1])
    signs = np.asarray(HAND_MIRROR_SIGNS, dtype=np.float32)
    assert np.abs(action[:, left]).max() > 1.0e-2
    np.testing.assert_allclose(mirrored[:, left], signs * action[:, right], atol=1.0e-5)
    np.testing.assert_allclose(mirrored[:, right], signs * action[:, left], atol=1.0e-5)


def test_mirror_equivariance_requires_mirrored_body_message():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from flash_chord.training.ppo.config import CoordinateConfig
    from flash_chord.training.ppo.coordinate.layout import HAND_MIRROR_SIGNS, mirror_observation

    layout = _layout()
    model = _network(layout, CoordinateConfig(mirror_body_message=False))
    parameters = _random_parameters(jax, model.init(jax.random.PRNGKey(0), jnp.zeros((1, 486))), scale=0.5)
    observation = np.random.default_rng(6).standard_normal((64, 486)).astype(np.float32)
    with jax.default_matmul_precision("highest"):
        action = np.asarray(model.apply(parameters, jnp.asarray(observation))[0].mean())
        mirrored = np.asarray(model.apply(parameters, jnp.asarray(mirror_observation(layout, observation)))[0].mean())
    left = np.asarray(layout.action_indices[0])
    right = np.asarray(layout.action_indices[1])
    signs = np.asarray(HAND_MIRROR_SIGNS, dtype=np.float32)
    assert np.abs(mirrored[:, left] - signs * action[:, right]).max() > 1.0e-3


def test_coordinated_module_is_hashable_inside_jit_train_state():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from flash_chord.training.ppo.algorithm import create_train_state
    from flash_chord.training.ppo.config import PPOConfig

    layout = _layout()
    model = _network(layout)
    state = create_train_state(model, model.init(jax.random.PRNGKey(0), jnp.zeros((1, 486))), PPOConfig())
    hash(model)

    @jax.jit
    def apply(state, observation):
        policy, value = state.apply_fn(state.params, observation)
        return policy.mean(), value

    mean, value = apply(state, jnp.ones((2, 486)))
    assert mean.shape == (2, 43) and value.shape == (2, 3)
