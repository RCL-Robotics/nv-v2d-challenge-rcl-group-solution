# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2 export equivalence (docs/coord_impl_spec.md sections 6.5 and 8, owner C).

Owner A's coordinated learner may land after this file. Every test therefore runs against A's real
interfaces when they import, and otherwise against ``_contract_*`` below: a transliteration of the
contract (sections 3.1-3.3 and Appendix A) that uses the exact parameter names of section 3.3.
``test_contract_stub_matches_owner_a`` cross-checks the two once A exists.

All equivalence arithmetic runs on the CPU. On the RTX PRO 6000 (Blackwell) GPUs, JAX's default
float32 matmul is TF32-like for large batches (measured ~3e-4 relative error on a 4096x486x1024
product), which would hide 1e-5 differences. For the evaluator's one-world batch the recorded
default-precision actions were full float32 (differences ~1e-7), and the GPU test reports both.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nn = pytest.importorskip("flax.linen")
distrax = pytest.importorskip("distrax")

from flash_chord.training.ppo.coordinate import export as e2  # noqa: E402

CPU = jax.devices("cpu")[0]
REPO = Path(__file__).resolve().parents[2]
STOCK_RUNTIME = Path(
    os.environ.get(
        "FC_STOCK_RUNTIME",
        "/root/autodl-tmp/haoran.zhang/code/video_to_data_8c778f9/robotic_grounding/flash_chord",
    )
)
KIT_ROOT = Path(
    os.environ.get("FC_KIT_ROOT", "/root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit")
)
STOCK_AVAILABLE = (STOCK_RUNTIME / "src" / "flash_chord" / "__init__.py").is_file()

# --------------------------------------------------------------------------------------------------
# Contract transliteration (spec 3.1-3.3, Appendix A). Used only when owner A's code is absent.
# --------------------------------------------------------------------------------------------------

VEC_SIGN = (1.0, -1.0, 1.0)
ROT6D_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
HAND_CANONICAL = (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
    "hand_thumb_0", "hand_thumb_1", "hand_thumb_2", "hand_middle_0", "hand_middle_1", "hand_index_0", "hand_index_1",
)  # fmt: skip
HAND_MIRROR_SIGNS = (1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1)
BODY_MIRROR_PERM = (6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14)
BODY_MIRROR_SIGNS = (1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1, 1)


@dataclass(frozen=True)
class _ContractCoordinateConfig:
    agents: tuple[str, ...] = ("left", "right", "body")
    encoder_dim: int = 96
    partner_dim: int = 64
    message_dim: int = 64
    trunk_dims: tuple[int, ...] = (80, 40)
    partner_message: bool = True
    mirror_body_message: bool = True
    hand_world_terms: str = "reflect"
    ratio_mode: str = "per_agent"
    value_mode: str = "per_agent"
    export_actor_hidden_dims: tuple[int, ...] = (1024, 512, 256, 128)

    def __post_init__(self):
        for name in ("agents", "trunk_dims", "export_actor_hidden_dims"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class _View:
    name: str
    indices: tuple[int, ...]
    signs: tuple[float, ...]


@dataclass(frozen=True)
class _ContractLayout:
    observation_dim: int
    action_dim: int
    agent_names: tuple[str, ...]
    weight_group: tuple[str, ...]
    actor_inputs: tuple[_View, ...]
    hand_descriptors: tuple[_View, _View]
    body_descriptor: _View
    body_descriptor_mirrored: _View
    action_indices: tuple[tuple[int, ...], ...]
    action_signs: tuple[tuple[float, ...], ...]
    action_group: tuple[int, ...] = field(default=())


def _policy_schema() -> dict:
    """Synthetic schema built from the real runtime constants (spec 6.5 test 1)."""
    from flash_chord.envs.recon_body import RECON_BODY_OBSERVATION_BLOCKS
    from flash_chord.runtime.actions.sonic import SONIC_DIRECT_JOINT_NAMES, SONIC_POLICY_JOINT_NAMES

    blocks, start = [], 0
    for name, width in RECON_BODY_OBSERVATION_BLOCKS:
        blocks.append({"name": name, "start": start, "end": start + width})
        start += width
    body = len(SONIC_POLICY_JOINT_NAMES)
    action_dim = body + len(SONIC_DIRECT_JOINT_NAMES)
    return {
        "action": {
            "dimension": action_dim,
            "blocks": [
                {"name": f"sonic_body_joint_residual[{','.join(SONIC_POLICY_JOINT_NAMES)}]", "start": 0, "end": body},
                {
                    "name": f"dex3_direct_joint_residual[{','.join(SONIC_DIRECT_JOINT_NAMES)}]",
                    "start": body,
                    "end": action_dim,
                },
            ],
            "input_mode": "raw",
            "input_mapping": "identity",
        },
        "observation": {"dimension": start, "blocks": blocks},
    }


def _contract_layout(schema: dict, coordinate) -> _ContractLayout:
    from flash_chord.embodiments.g1_dex3 import G1_BODY_JOINT_NAMES, G1_DEX3_JOINT_NAMES

    offsets = {block["name"].split("[")[0]: block["start"] for block in schema["observation"]["blocks"]}
    action_names = []
    for block in schema["action"]["blocks"]:
        action_names.extend(block["name"].split("[", 1)[1].rstrip("]").split(","))
    action_index = {name: index for index, name in enumerate(action_names)}
    scalar = G1_BODY_JOINT_NAMES[:22] + G1_DEX3_JOINT_NAMES[:7] + G1_BODY_JOINT_NAMES[22:] + G1_DEX3_JOINT_NAMES[7:]
    scalar_index = {name: index for index, name in enumerate(scalar)}
    joint_position = offsets["joint_position_relative_default_rad"]
    joint_velocity = offsets["joint_velocity_radps"]
    future_joint = offsets["future_joint_position_delta_rad"]
    history = offsets["processed_target_history"]

    def own(side):
        return [scalar_index[f"{side}_{name}_joint"] for name in HAND_CANONICAL]

    def build(name, parts):
        indices, signs = [], []
        for part_indices, part_signs in parts:
            part_indices = list(part_indices)
            part_signs = list(part_signs)
            assert len(part_indices) == len(part_signs)
            indices += part_indices
            signs += [float(sign) for sign in part_signs]
        return _View(name, tuple(int(i) for i in indices), tuple(signs))

    def hand_view(side):
        right = side == "right"
        h = 0 if right else 1  # wrist/palm blocks are [right, left]
        live = 1 if right else 0  # live_hand_pose_in_object is [left, right]

        def m(signs):
            return signs if right else (1.0,) * len(signs)

        joints = own(side)
        joint_signs = m(HAND_MIRROR_SIGNS)
        parts = [
            (range(offsets["wrist_position_pelvis_m"] + 3 * h, offsets["wrist_position_pelvis_m"] + 3 * h + 3), m(VEC_SIGN)),
            (range(offsets["wrist_orientation_pelvis_6d"] + 6 * h, offsets["wrist_orientation_pelvis_6d"] + 6 * h + 6), m(ROT6D_SIGN)),
            (range(offsets["wrist_linear_velocity_pelvis_mps"] + 3 * h, offsets["wrist_linear_velocity_pelvis_mps"] + 3 * h + 3), m(VEC_SIGN)),
            ([joint_position + j for j in joints], joint_signs),
            ([joint_velocity + j for j in joints], joint_signs),
        ]  # fmt: skip
        parts += [([future_joint + 43 * f + j for j in joints], joint_signs) for f in range(3)]
        if coordinate.hand_world_terms == "reflect":
            base = offsets["future_palm_position_delta_w_m"]
            parts += [(range(base + (2 * f + h) * 3, base + (2 * f + h) * 3 + 3), m(VEC_SIGN)) for f in range(3)]
        base = offsets["future_palm_orientation_delta_6d"]
        parts += [(range(base + (2 * f + h) * 6, base + (2 * f + h) * 6 + 6), m(ROT6D_SIGN)) for f in range(3)]
        base = offsets["live_hand_pose_in_object"] + 9 * live
        parts += [(range(base, base + 3), m(VEC_SIGN)), (range(base + 3, base + 9), m(ROT6D_SIGN))]
        parts += [([history + 43 * s + j for j in joints], joint_signs) for s in range(3)]
        target = offsets["target_object_pose_in_live_object"]
        parts += [
            (range(offsets["object_position_pelvis_m"], offsets["object_position_pelvis_m"] + 3), m(VEC_SIGN)),
            (range(offsets["object_orientation_pelvis_6d"], offsets["object_orientation_pelvis_6d"] + 6), m(ROT6D_SIGN)),
            (range(target, target + 3), m(VEC_SIGN)),
            (range(target + 3, target + 9), m(ROT6D_SIGN)),
            ([offsets["normalized_reference_phase"]], (1.0,)),
        ]  # fmt: skip
        return build(f"x_{side}", parts)

    def hand_descriptor(side):
        right = side == "right"
        live = 1 if right else 0
        joints = own(side)
        signs = HAND_MIRROR_SIGNS if right else (1.0,) * 14
        base = offsets["live_hand_pose_in_object"] + 9 * live
        return build(
            f"d_{side}",
            [
                (range(base, base + 3), VEC_SIGN if right else (1.0,) * 3),
                (range(base + 3, base + 9), ROT6D_SIGN if right else (1.0,) * 6),
                ([joint_position + j for j in joints], signs),
                ([joint_velocity + j for j in joints], signs),
                ([history + 43 * 2 + j for j in joints], signs),
            ],
        )

    body_joints = list(range(15))
    plus15 = (1.0,) * 15
    x_body = build(
        "x_body",
        [([joint_position + j for j in body_joints], plus15), ([joint_velocity + j for j in body_joints], plus15)]
        + [([future_joint + 43 * f + j for j in body_joints], plus15) for f in range(3)]
        + [([history + 43 * s + j for j in body_joints], plus15) for s in range(3)]
        + [
            (range(offsets["future_root_position_w_m"], offsets["future_root_position_w_m"] + 9), (1.0,) * 9),
            (range(offsets["future_root_orientation_delta_6d"], offsets["future_root_orientation_delta_6d"] + 18), (1.0,) * 18),
            (range(offsets["wrist_position_pelvis_m"], offsets["wrist_position_pelvis_m"] + 6), (1.0,) * 6),
            (range(offsets["object_position_pelvis_m"], offsets["object_position_pelvis_m"] + 9), (1.0,) * 9),
            (range(offsets["target_object_pose_in_live_object"], offsets["target_object_pose_in_live_object"] + 10), (1.0,) * 10),
        ],
    )  # fmt: skip
    body_blocks = [joint_position, joint_velocity, history + 43 * 2]
    d_body = build("d_body", [([base + j for j in body_joints], plus15) for base in body_blocks])
    d_body_mirrored = build(
        "d_body_mirrored", [([base + BODY_MIRROR_PERM[k] for k in range(15)], BODY_MIRROR_SIGNS) for base in body_blocks]
    )
    left_actions = tuple(action_index[f"left_{name}_joint"] for name in HAND_CANONICAL)
    right_actions = tuple(action_index[f"right_{name}_joint"] for name in HAND_CANONICAL)
    body_actions = tuple(action_index[name] for name in scalar[:15])
    action_group = [0] * len(action_names)
    for agent, indices in enumerate((left_actions, right_actions, body_actions)):
        for index in indices:
            action_group[index] = agent
    return _ContractLayout(
        observation_dim=int(schema["observation"]["dimension"]),
        action_dim=len(action_names),
        agent_names=("left", "right", "body"),
        weight_group=("hand", "hand", "body"),
        actor_inputs=(hand_view("left"), hand_view("right"), x_body),
        hand_descriptors=(hand_descriptor("left"), hand_descriptor("right")),
        body_descriptor=d_body,
        body_descriptor_mirrored=d_body_mirrored,
        action_indices=(left_actions, right_actions, body_actions),
        action_signs=((1.0,) * 14, tuple(float(s) for s in HAND_MIRROR_SIGNS), (1.0,) * 15),
        action_group=tuple(action_group),
    )


def _contract_digest(layout) -> str:
    payload = json.dumps(asdict(layout), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _gather(observation, view):
    return observation[..., jnp.asarray(view.indices)] * jnp.asarray(view.signs, dtype=jnp.float32)


class _ContractCoordinatedActorCritic(nn.Module):
    """Spec 3.3 forward pass with the exact parameter names."""

    action_dim: int
    layout: _ContractLayout
    coordinate: _ContractCoordinateConfig
    network: object

    @nn.compact
    def __call__(self, observation):
        c, layout = self.coordinate, self.layout

        def dense(width, name, gain=math.sqrt(2.0)):
            return nn.Dense(width, kernel_init=nn.initializers.orthogonal(gain), name=name)

        x_left, x_right, x_body = layout.actor_inputs
        d_left, d_right = layout.hand_descriptors
        hand_encoder = dense(c.encoder_dim, "hand_encoder")
        body_encoder = dense(c.encoder_dim, "body_encoder")
        e = [
            nn.elu(hand_encoder(_gather(observation, x_left))),
            nn.elu(hand_encoder(_gather(observation, x_right))),
            nn.elu(body_encoder(_gather(observation, x_body))),
        ]
        if c.partner_message:
            hand_partner = dense(c.partner_dim, "hand_partner")
            body_partner = dense(c.partner_dim, "body_partner")
            p_left = nn.elu(hand_partner(_gather(observation, d_left)))
            p_right = nn.elu(hand_partner(_gather(observation, d_right)))
            p_body = nn.elu(body_partner(_gather(observation, layout.body_descriptor)))
            p_body_mirrored = nn.elu(body_partner(_gather(observation, layout.body_descriptor_mirrored)))
            hand_message = dense(c.message_dim, "hand_message")
            body_message = dense(c.message_dim, "body_message")
            m = [
                nn.elu(hand_message(p_right + p_body)),
                nn.elu(hand_message(p_left + (p_body_mirrored if c.mirror_body_message else p_body))),
                nn.elu(body_message(p_left + p_right)),
            ]
        else:
            m = [jnp.zeros(observation.shape[:-1] + (c.message_dim,), observation.dtype)] * 3
        head_init = nn.initializers.constant(0.0)
        if self.network.actor_output_gain > 0.0:
            head_init = nn.initializers.orthogonal(self.network.actor_output_gain)
        layers = {
            group: (
                dense(c.trunk_dims[0], f"{group}_trunk_0"),
                dense(c.trunk_dims[1], f"{group}_trunk_1"),
                nn.Dense(14 if group == "hand" else 15, kernel_init=head_init, name=f"{group}_head"),
            )
            for group in ("hand", "body")
        }
        mean = jnp.zeros(observation.shape[:-1] + (self.action_dim,), observation.dtype)
        for agent, group in enumerate(layout.weight_group):
            trunk0, trunk1, head = layers[group]
            hidden = nn.elu(trunk1(nn.elu(trunk0(jnp.concatenate([e[agent], m[agent]], axis=-1)))))
            mu = head(hidden)
            indices = jnp.asarray(layout.action_indices[agent])
            mean = mean.at[..., indices].set(jnp.asarray(layout.action_signs[agent], dtype=mu.dtype) * mu)
        log_std = self.param(
            "log_std", nn.initializers.constant(math.log(self.network.initial_noise_std)), (self.action_dim,)
        )
        critic = observation
        for index, width in enumerate(self.network.critic_hidden_dims):
            critic = nn.elu(dense(width, f"critic_{index}")(critic))
        value = nn.Dense(len(layout.agent_names), kernel_init=nn.initializers.orthogonal(1.0), name="value")(critic)
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)), value


def _contract_learner_template(config, observation_dim, action_dim, layout=None, world_count=1):
    from flash_chord.training.ppo.algorithm import create_train_state
    from flash_chord.training.ppo.learner import LearnerState
    from flash_chord.training.ppo.normalization import create_normalization_state

    key, initialization_key = jax.random.split(jax.random.PRNGKey(config.seed))
    model = _ContractCoordinatedActorCritic(
        action_dim=action_dim, layout=layout, coordinate=config.coordinate, network=config.network
    )
    parameters = model.init(initialization_key, jnp.zeros((1, observation_dim), dtype=jnp.float32))
    return LearnerState(
        train_state=create_train_state(model, parameters, config.ppo),
        normalization=create_normalization_state(observation_dim),
        observation=jnp.zeros((world_count, observation_dim), dtype=jnp.float32),
        key=key,
        learning_rate=jnp.asarray(config.ppo.learning_rate, dtype=jnp.float32),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )


def _contract_training_config(training: dict):
    """Stock TrainingConfig fields plus a ``coordinate`` attribute (stand-in for A's field)."""
    import dataclasses

    stock = e2._stock_training_config({k: v for k, v in training.items() if k != "coordinate"})
    values = {item.name: getattr(stock, item.name) for item in dataclasses.fields(stock) if item.name != "coordinate"}
    coordinate = {k: v for k, v in training["coordinate"].items() if k != "_target_"}
    return SimpleNamespace(**values, coordinate=_ContractCoordinateConfig(**coordinate))


def _contract_api() -> SimpleNamespace:
    return SimpleNamespace(
        training_config=_contract_training_config,
        build_layout=_contract_layout,
        layout_digest=_contract_digest,
        create_learner_template=_contract_learner_template,
    )


def _owner_a_api():
    """Owner A's real interface, or None while it has not landed."""
    try:
        api = e2._coordinate_api()
        from flash_chord.training.ppo.config import CoordinateConfig
        from flash_chord.training.ppo.coordinate.network import CoordinatedActorCritic, actor_mean
    except ImportError:
        return None
    return SimpleNamespace(
        api=api, CoordinateConfig=CoordinateConfig, CoordinatedActorCritic=CoordinatedActorCritic, actor_mean=actor_mean
    )


OWNER_A = _owner_a_api()
IMPLEMENTATIONS = ["contract"] + (["owner_a"] if OWNER_A is not None else [])


def _coordinate_config(implementation: str, **overrides):
    if implementation == "owner_a":
        return OWNER_A.CoordinateConfig(**overrides)
    return _ContractCoordinateConfig(**overrides)


def _layout(implementation: str, coordinate):
    if implementation == "owner_a":
        return OWNER_A.api.build_layout(_policy_schema(), coordinate)
    return _contract_layout(_policy_schema(), coordinate)


def _network_config(**overrides):
    from flash_chord.training.ppo.config import NetworkConfig

    values = dict(actor_hidden_dims=(1024, 512, 256, 128), critic_hidden_dims=(1024, 512, 256, 128), initial_noise_std=0.5)
    values.update(overrides)
    return NetworkConfig(**values)


def _model(implementation: str, layout, coordinate, network):
    if implementation == "owner_a":
        return OWNER_A.CoordinatedActorCritic(action_dim=43, layout=layout, coordinate=coordinate, network=network)
    return _ContractCoordinatedActorCritic(action_dim=43, layout=layout, coordinate=coordinate, network=network)


def _randomize(variables, rng: np.random.Generator, head_scale: float = 1.0):
    """Non-zero heads, random biases and log_std so every block and the bias fold are exercised."""

    def visit(path, leaf):
        names = [getattr(entry, "key", str(entry)) for entry in path]
        array = np.asarray(leaf)
        if names[-1] == "log_std":
            return jnp.asarray(rng.normal(-1.0, 0.3, array.shape), dtype=jnp.float32)
        if names[-1] == "bias":
            scale = head_scale * 0.5 if any(name.endswith("_head") for name in names) else 0.5
            return jnp.asarray(rng.normal(0.0, scale, array.shape), dtype=jnp.float32)
        if any(name.endswith("_head") for name in names):
            return jnp.asarray(
                rng.normal(0.0, head_scale / math.sqrt(array.shape[0]), array.shape), dtype=jnp.float32
            )
        return jnp.asarray(array * rng.uniform(0.8, 1.2), dtype=jnp.float32)

    return jax.tree_util.tree_map_with_path(visit, variables)


def _random_normalization(rng: np.random.Generator, dim: int = 486):
    from flash_chord.training.ppo.normalization import NormalizationState

    return NormalizationState(
        mean=jnp.asarray(rng.normal(0.0, 2.0, dim), dtype=jnp.float32),
        variance=jnp.asarray(np.exp(rng.normal(0.0, 1.5, dim)), dtype=jnp.float32),
        count=jnp.asarray(1000, dtype=jnp.int32),
    )


def _stock_mean(actor: dict, normalized: np.ndarray, rng: np.random.Generator):
    from flash_chord.training.ppo.network import ActorCritic

    model = ActorCritic(action_dim=43, config=_network_config())
    variables = model.init(jax.random.PRNGKey(int(rng.integers(1 << 30))), jnp.zeros((1, 486), dtype=jnp.float32))
    inner = dict(variables["params"])
    inner.update(jax.tree.map(jnp.asarray, actor))
    policy, _ = model.apply({"params": inner}, jnp.asarray(normalized))
    return np.asarray(policy.mean()), np.asarray(jnp.exp(inner["log_std"]))


VARIANTS = {
    "default": {},
    "no_partner_message": {"partner_message": False},
    "no_mirror_body_message": {"mirror_body_message": False},
    "drop_world_terms": {"hand_world_terms": "drop"},
}


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_contract_layout_reproduces_appendix_a():
    """The stand-in layout equals spec Appendix A.4 (so tests built on it are tests of the contract)."""

    def ranges(*spans):
        out = []
        for span in spans:
            lo, hi = span if isinstance(span, tuple) else (span, span)
            out += list(range(lo, hi + 1))
        return out

    layout = _contract_layout(_policy_schema(), _ContractCoordinateConfig())
    x_left, x_right, x_body = layout.actor_inputs
    assert list(x_left.indices) == ranges(
        (3, 5), (12, 17), (21, 23), (48, 61), (91, 104), (161, 174), (204, 217), (247, 260), (278, 280),
        (284, 286), (290, 292), (299, 304), (311, 316), (323, 337), (372, 385), (415, 428), (458, 471),
        (24, 32), (347, 356),
    )  # fmt: skip
    assert set(x_left.signs) == {1.0}
    assert list(x_right.indices) == ranges(
        (0, 2), (6, 11), (18, 20), (62, 75), (105, 118), (175, 188), (218, 231), (261, 277), (281, 283),
        (287, 289), (293, 298), (305, 310), (317, 322), (338, 346), (386, 399), (429, 442), (472, 485),
        (24, 32), (347, 356),
    )  # fmt: skip
    assert sum(sign < 0 for sign in x_right.signs) == 109
    assert list(x_body.indices) == ranges(
        (33, 47), (76, 90), (146, 160), (189, 203), (232, 246), (357, 371), (400, 414), (443, 457), (119, 145),
        (0, 5), (24, 32), (347, 356),
    )  # fmt: skip
    assert list(layout.hand_descriptors[0].indices) == ranges((329, 337), (48, 61), (91, 104), (458, 471))
    assert list(layout.hand_descriptors[1].indices) == ranges((338, 346), (62, 75), (105, 118), (472, 485))
    assert list(layout.body_descriptor.indices) == ranges((33, 47), (76, 90), (443, 457))
    assert list(layout.body_descriptor_mirrored.indices[:15]) == [39, 40, 41, 42, 43, 44, 33, 34, 35, 36, 37, 38, 45, 46, 47]
    assert list(layout.body_descriptor_mirrored.signs[:15]) == list(map(float, BODY_MIRROR_SIGNS))
    assert layout.action_indices == (
        (11, 15, 19, 21, 23, 25, 27, 31, 37, 41, 30, 36, 29, 35),
        (12, 16, 20, 22, 24, 26, 28, 34, 40, 42, 33, 39, 32, 38),
        (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8),
    )
    assert sorted(sum(layout.action_indices, ())) == list(range(43))
    dropped = _contract_layout(_policy_schema(), _ContractCoordinateConfig(hand_world_terms="drop"))
    assert len(dropped.actor_inputs[0].indices) == len(dropped.actor_inputs[1].indices) == 170
    assert not set(dropped.actor_inputs[0].indices) & set(ranges((278, 280), (284, 286), (290, 292)))


@pytest.mark.skipif(OWNER_A is None, reason="owner A's coordinated learner has not landed")
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_contract_stub_matches_owner_a(variant):
    """A's layout and network equal the contract transliteration used by this file."""
    a_config = _coordinate_config("owner_a", **VARIANTS[variant])
    c_config = _coordinate_config("contract", **VARIANTS[variant])
    a_layout = _layout("owner_a", a_config)
    c_layout = _layout("contract", c_config)
    for name in ("observation_dim", "action_dim", "agent_names", "weight_group"):
        assert getattr(a_layout, name) == getattr(c_layout, name), name
    assert tuple(tuple(int(i) for i in indices) for indices in a_layout.action_indices) == c_layout.action_indices
    for a_view, c_view in zip(
        (*a_layout.actor_inputs, *a_layout.hand_descriptors, a_layout.body_descriptor, a_layout.body_descriptor_mirrored),
        (*c_layout.actor_inputs, *c_layout.hand_descriptors, c_layout.body_descriptor, c_layout.body_descriptor_mirrored),
        strict=True,
    ):
        assert tuple(a_view.indices) == c_view.indices, c_view.name
        assert tuple(float(s) for s in a_view.signs) == c_view.signs, c_view.name
    assert tuple(tuple(float(s) for s in signs) for signs in a_layout.action_signs) == c_layout.action_signs

    rng = np.random.default_rng(7)
    network = _network_config()
    with jax.default_device(CPU):
        a_model = _model("owner_a", a_layout, a_config, network)
        variables = _randomize(a_model.init(jax.random.PRNGKey(0), jnp.zeros((1, 486))), rng)
        x = jnp.asarray(np.clip(rng.standard_t(3.0, (256, 486)), -20, 20), dtype=jnp.float32)
        a_mean = np.asarray(a_model.apply(variables, x)[0].mean())
        c_model = _model("contract", c_layout, c_config, network)
        c_mean = np.asarray(c_model.apply(variables, x)[0].mean())
        a_actor_mean = np.asarray(OWNER_A.actor_mean(variables, a_layout, a_config, x))
    np.testing.assert_allclose(a_mean, c_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a_actor_mean, a_mean, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_embed_exact_random(implementation, variant):
    """Spec 6.5 test 1: stock MLP with the embedded actor reproduces the coordinated mean to 1e-5."""
    from flash_chord.training.ppo.normalization import normalize_observation

    rng = np.random.default_rng(zlib.crc32(f"{implementation}/{variant}".encode()))
    coordinate = _coordinate_config(implementation, **VARIANTS[variant])
    layout = _layout(implementation, coordinate)
    network = _network_config()
    with jax.default_device(CPU):
        model = _model(implementation, layout, coordinate, network)
        variables = _randomize(model.init(jax.random.PRNGKey(1), jnp.zeros((1, 486))), rng)
        normalization = _random_normalization(rng)
        raw = e2.make_probe_observations(normalization, 4096, int(rng.integers(1 << 30)), 0.01)
        normalized = np.asarray(normalize_observation(normalization, jnp.asarray(raw), 0.01))
        policy, value = model.apply(variables, jnp.asarray(normalized))
        coordinated = np.asarray(policy.mean())
        assert value.shape == (raw.shape[0], 3)
        actor = e2.embed_actor(variables, layout, coordinate, network.actor_hidden_dims)
        stock, stock_std = _stock_mean(actor, normalized, rng)

    # ELU sees both signs in every block.
    assert np.mean(normalized < 0) > 0.2 and np.mean(normalized > 0) > 0.2
    assert np.std(coordinated) > 0.1, "heads must be non-zero"
    comparison = e2.compare_actions(coordinated, stock)
    assert comparison["finite"]
    assert comparison["max_rel_diff"] <= 1e-5, comparison
    np.testing.assert_array_equal(np.asarray(actor["log_std"]), np.asarray(variables["params"]["log_std"]))
    np.testing.assert_allclose(stock_std, np.exp(np.asarray(variables["params"]["log_std"])), rtol=1e-6)

    # Units beyond the budget are inert: zero kernel columns, zero bias, zero outgoing rows.
    plan = e2.embedding_plan(coordinate, network.actor_hidden_dims)
    kernels = [actor[f"Dense_{i}"]["kernel"] for i in range(5)]
    biases = [actor[f"Dense_{i}"]["bias"] for i in range(5)]
    for layer, used in enumerate(plan.used_units):
        assert not np.any(kernels[layer][:, used:]) and not np.any(biases[layer][used:])
        assert not np.any(kernels[layer + 1][used:, :])


def test_embedding_budget_is_enforced():
    coordinate = _ContractCoordinateConfig()
    plan = e2.embedding_plan(coordinate, (1024, 512, 256, 128))
    assert plan.used_units == (544, 480, 240, 120)
    assert e2.embedding_plan(_ContractCoordinateConfig(mirror_body_message=False), (1024, 512, 256, 128)).used_units[0] == 480
    with pytest.raises(ValueError, match="budget"):
        e2.embedding_plan(coordinate, (512, 512, 256, 128))
    with pytest.raises(ValueError, match="budget"):
        e2.embedding_plan(_ContractCoordinateConfig(trunk_dims=(80, 50)), (1024, 512, 256, 128))
    with pytest.raises(ValueError, match="hidden layers"):
        e2.embedding_plan(coordinate, (1024, 512, 256))


def test_stock_field_tables_match_runtime_dataclasses():
    import dataclasses

    from flash_chord.lifecycle.reset import ReconBodyResetConfig
    from flash_chord.training.ppo.config import NetworkConfig, PPOConfig, TrainingConfig

    # Dev-only training keys; E2 export prunes them (every key outside STOCK_TRAINING_KEYS is removed).
    training_fields = {f.name for f in dataclasses.fields(TrainingConfig)} - {
        "coordinate", "nonfinite_guard", "early_stop_help_zero_tail"}
    assert set(e2.STOCK_TRAINING_KEYS) - {"_target_"} == training_fields
    assert set(e2.STOCK_NETWORK_KEYS) - {"_target_"} == {f.name for f in dataclasses.fields(NetworkConfig)}
    assert set(e2.STOCK_PPO_KEYS) - {"_target_"} == {f.name for f in dataclasses.fields(PPOConfig)}
    assert tuple(f.name for f in dataclasses.fields(ReconBodyResetConfig)) == e2.STOCK_RESET_FIELDS


def _rsi_reset_mapping(stock_reset: dict) -> dict:
    """A resolved RSI reset subtree as B's recon_body_rsi.yaml produces it (spec 4.1)."""
    return {
        **stock_reset,
        "_target_": "flash_chord.lifecycle.rsi_reset.RSIReconBodyResetConfig",
        "bin_frames": 25, "uniform_mix": 0.3, "credit_gamma": 0.75, "fail_alpha": 0.02, "success_up": 0.8,
        "success_down": 0.4, "step": 0.9, "band": 2.0, "min_count": 64, "min_gap": 40, "wait_iterations": 100,
        "top0": 1.0, "zero_below": 6.25e-4, "deadline": 11000, "zero_ramp": 1000, "min_horizon": 10,
        "frame0_share": 0.25, "start_update": 0, "keyframes": True, "kf_to_end": True, "kf_capacity": 256,
        "kf_prob": 0.002, "kf_min_fill": 32, "kf_min_task": 0.6, "kf_min_stepped": 15, "kf_step_score_min": 0.3,
        "task_pos_scale": 10.0,
    }  # fmt: skip


def _rsi_curriculum(stock_curriculum: dict) -> dict:
    stage = dict(stock_curriculum["stages"][0])
    stage["voc_scale"] = 0.0
    stage["reset_to_first_frame_probability"] = None
    stage["immediate_first_frame_probability"] = None
    return {"_target_": stock_curriculum["_target_"], "thresholds": [20000], "stages": [stage]}


def _c3_like_resolved_config(parquet: str = "/tmp/reference.parquet", num_frames: int = 440) -> dict:
    """Resolved C3 config. Composes the real C3 experiment when owner B's yamls exist, else C2 + an RSI-like swap."""
    from flash_chord.configuration import compose_config, resolved_dict

    configs = REPO / "src" / "flash_chord" / "configs"
    have_rsi = (configs / "reset" / "recon_body_rsi.yaml").is_file() and (
        configs / "curriculum" / "recon_body_rsi.yaml"
    ).is_file()
    experiment = "g1_recon_body_coord_rsi" if have_rsi else "g1_recon_body_coord"
    config = resolved_dict(
        compose_config("train", [f"experiment={experiment}", f"task.parquet={parquet}", "scene.support=false"])
    )
    if not have_rsi:
        rsi_reset = _rsi_reset_mapping(config["reset"])
        curriculum = _rsi_curriculum(config["curriculum"])
        config["reset"] = rsi_reset
        config["env"]["reset"] = dict(rsi_reset)
        config["curriculum"] = curriculum
        config["training"]["curriculum"] = json.loads(json.dumps(curriculum))
    assert config["env"]["reset"]["_target_"] == "flash_chord.lifecycle.rsi_reset.RSIReconBodyResetConfig"
    config["reference"] = {
        "parquet": parquet,
        "num_frames": num_frames,
        "playback_fps": 30.0,
        "object_articulations": {"version": 1, "entries": []},
    }
    return config


def test_stock_config_rewrite():
    """Spec 6.3 rules 1-5 on a C3-like resolved config."""
    saved = _c3_like_resolved_config()
    original = json.loads(json.dumps(saved))
    exported, notes = e2.stock_config_with_notes(saved)
    assert saved == original, "stock_config must not mutate its input"
    assert "coordinate" not in exported["training"]
    assert "training.coordinate" in notes["removed_keys"]
    for reset in (exported["env"]["reset"], exported["reset"]):
        assert reset["_target_"] == e2.STOCK_RESET_TARGET
        assert set(reset) == {"_target_", *e2.STOCK_RESET_FIELDS}
        assert reset["reset_to_first_frame_probability"] == reset["immediate_first_frame_probability"] == 0.0
        for name in set(e2.STOCK_RESET_FIELDS) - set(e2.STOCK_RESET_ZEROED):
            assert reset[name] == saved["reset"][name]
    assert exported["training"]["network"]["actor_hidden_dims"] == [1024, 512, 256, 128]
    assert exported["training"]["network"]["critic_hidden_dims"] == [1024, 512, 256, 128]
    assert exported["curriculum"] == saved["curriculum"]
    assert exported["training"]["curriculum"] == saved["training"]["curriculum"]
    untouched = set(saved) - {"training", "env", "reset"}
    for key in untouched:
        assert exported[key] == saved[key], key
    for key in set(saved["env"]) - {"reset"}:
        assert exported["env"][key] == saved["env"][key], key
    for key in set(saved["training"]) - {"coordinate", "network"}:
        assert exported["training"][key] == saved["training"][key], key
    # The DEV runtime instantiates the stock training config (no coordinate field set).
    training = e2._stock_training_config(exported["training"])
    assert getattr(training, "coordinate", None) is None
    with pytest.raises(ValueError, match="PPO"):
        e2.stock_config({**saved, "training": {**saved["training"], "_target_": "x.y.TrainingConfig"}})


def _write_coordinated_checkpoint(
    path: Path, config: dict, implementation: str, rng: np.random.Generator, *, iteration=300, head_scale=1.0, schema=None
):
    """Coordinated LearnerState with random (non-zero-head) params saved through the stock save path."""
    from flash_chord.training.checkpoint import save_checkpoint
    from flash_chord.training.checkpoint_metadata import build_checkpoint_metadata

    api = OWNER_A.api if implementation == "owner_a" else _contract_api()
    training = api.training_config(config["training"])
    schema = schema or _policy_schema()
    layout = api.build_layout(schema, training.coordinate)
    with jax.default_device(CPU):
        learner = api.create_learner_template(training, 486, 43, layout, world_count=8)
        params = _randomize(learner.train_state.params, rng, head_scale)
        learner = learner.replace(
            train_state=learner.train_state.replace(params=params, step=jnp.asarray(7, dtype=jnp.int32)),
            normalization=_random_normalization(rng),
            key=jax.random.PRNGKey(123),
            learning_rate=jnp.asarray(3e-4, dtype=jnp.float32),
            iteration=jnp.asarray(iteration, dtype=jnp.int32),
        )
    metadata = build_checkpoint_metadata(config, schema)
    metadata.update(
        {
            "iteration": iteration,
            "environment_steps": iteration * 24 * 4096,
            "voc_scale": 0.0,
            e2.LAYOUT_METADATA_KEY: api.layout_digest(layout),
        }
    )
    save_checkpoint(path, learner, metadata=metadata)
    return learner, training, layout


def _use_api(monkeypatch, implementation):
    if implementation == "contract":
        monkeypatch.setattr(e2, "_coordinate_api", _contract_api)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_export_roundtrip(tmp_path, monkeypatch, implementation):
    """Spec 6.5 test 2: coordinated C3-like checkpoint -> stock checkpoint the stock path restores."""
    from flash_chord.evaluation.harness import checkpoint_training_algorithm
    from flash_chord.training.checkpoint import load_checkpoint_for_inference, read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config, checkpoint_policy_schema
    from flash_chord.training.ppo.evaluation import EvaluationConfig, policy_action

    _use_api(monkeypatch, implementation)
    rng = np.random.default_rng(11)
    source = tmp_path / "coord" / "model_300.safetensors"
    config = _c3_like_resolved_config()
    learner, training, _ = _write_coordinated_checkpoint(source, config, implementation, rng)
    destination = tmp_path / "export" / "model_300.safetensors"
    report = e2.export_checkpoint(
        source, destination, stock_runtime=STOCK_RUNTIME if STOCK_AVAILABLE else None, probe_count=2048
    )
    assert report["status"] == "ok" and report["max_rel_diff"] <= 1e-5, report
    assert report["leaf_count"] == 72
    assert Path(report["report_path"]).is_file()

    metadata = read_checkpoint_metadata(destination)
    exported = checkpoint_config(metadata)
    assert checkpoint_training_algorithm(exported) == "ppo"
    assert exported["env"]["reset"]["_target_"] == e2.STOCK_RESET_TARGET == exported["reset"]["_target_"]
    assert "coordinate" not in exported["training"]
    assert checkpoint_policy_schema(metadata) == _policy_schema()
    assert metadata["policy_schema_json"] == read_checkpoint_metadata(source)["policy_schema_json"]
    assert metadata["iteration"] == "300" and metadata["export_format"] == e2.EXPORT_FORMAT
    assert metadata["export_source_sha256"] == e2.file_sha256(source)
    assert metadata[e2.LAYOUT_METADATA_KEY] == read_checkpoint_metadata(source)[e2.LAYOUT_METADATA_KEY]

    stock_training = e2._stock_training_config(exported["training"])
    with jax.default_device(CPU):
        restored, _ = load_checkpoint_for_inference(destination, e2.stock_learner_template(stock_training, 486, 43))
        assert len(jax.tree.leaves(restored)) == 72
        assert int(restored.iteration) == 300 and int(restored.train_state.step) == 0
        np.testing.assert_array_equal(np.asarray(restored.normalization.mean), np.asarray(learner.normalization.mean))
        np.testing.assert_array_equal(np.asarray(restored.key), np.asarray(learner.key))
        observations = e2.make_probe_observations(learner.normalization, 1024, 99, 0.01)
        evaluation = EvaluationConfig(checkpoint=str(destination))
        stock_actions = np.asarray(policy_action(restored, jnp.asarray(observations), stock_training, evaluation))
        coordinated = np.asarray(policy_action(learner, jnp.asarray(observations), training, evaluation))
    assert e2.compare_actions(coordinated, stock_actions)["max_rel_diff"] <= 1e-5
    if STOCK_AVAILABLE:
        assert report["stock_runtime"]["verified"], report["stock_runtime"]


def test_export_stock_source_copies_leaves(tmp_path):
    """C1 (stock learner + RSI config): leaves copied, only the config is rewritten."""
    from flash_chord.training.checkpoint import load_checkpoint_for_inference, read_checkpoint_metadata, save_checkpoint
    from flash_chord.training.checkpoint_metadata import build_checkpoint_metadata, checkpoint_config

    config = _c3_like_resolved_config()
    del config["training"]["coordinate"]  # C1: stock learner + RSI reset
    training = e2._stock_training_config(config["training"])
    rng = np.random.default_rng(3)
    with jax.default_device(CPU):
        learner = e2.stock_learner_template(training, 486, 43, world_count=4)
        learner = learner.replace(
            train_state=learner.train_state.replace(params=_randomize(learner.train_state.params, rng)),
            normalization=_random_normalization(rng),
            iteration=jnp.asarray(500, dtype=jnp.int32),
        )
    source = tmp_path / "c1" / "model_500.safetensors"
    metadata = build_checkpoint_metadata(config, _policy_schema())
    metadata.update({"iteration": 500, "environment_steps": 1, "voc_scale": 0.0})
    save_checkpoint(source, learner, metadata=metadata)
    destination = tmp_path / "export" / "model_500.safetensors"
    report = e2.export_checkpoint(source, destination, stock_runtime=None, probe_count=512)
    assert report["source_kind"] == "stock" and report["dev_restore"]["max_abs_diff"] == 0.0
    exported = checkpoint_config(read_checkpoint_metadata(destination))
    assert exported["env"]["reset"]["_target_"] == e2.STOCK_RESET_TARGET
    with jax.default_device(CPU):
        restored, _ = load_checkpoint_for_inference(destination, e2.stock_learner_template(training, 486, 43))
    for original, copied in zip(
        jax.tree.leaves(learner.replace(observation=restored.observation)), jax.tree.leaves(restored), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(copied))


def test_export_refuses_layout_digest_mismatch(tmp_path, monkeypatch):
    from safetensors.numpy import save_file

    from flash_chord.training.checkpoint import _read_checkpoint, read_checkpoint_metadata

    implementation = IMPLEMENTATIONS[-1]
    _use_api(monkeypatch, implementation)
    source = tmp_path / "model_1.safetensors"
    _write_coordinated_checkpoint(source, _c3_like_resolved_config(), implementation, np.random.default_rng(0))
    # Rewrite the stored digest.
    manifest, arrays, user = _read_checkpoint(source)
    assert user == read_checkpoint_metadata(source)
    raw = {entry["key"]: arrays[entry["key"]] for entry in manifest}
    header = {"format": "learner_state", "version": "1", "manifest": json.dumps(manifest, separators=(",", ":"))}
    header.update({f"user.{k}": v for k, v in user.items()})
    header[f"user.{e2.LAYOUT_METADATA_KEY}"] = "0" * 64
    save_file(raw, str(source), metadata=header)
    with pytest.raises(ValueError, match="digest mismatch"):
        e2.export_checkpoint(source, tmp_path / "out.safetensors", stock_runtime=None, probe_count=64)
    assert not (tmp_path / "out.safetensors").exists()


def test_stock_learner_template_matches_stock_create_learner():
    """The env-free template has exactly the leaves of stock create_learner (learner.py:44-62)."""
    from flash_chord.training.ppo.learner import create_learner

    training = e2._stock_training_config(e2.stock_config(_c3_like_resolved_config())["training"])
    training = replace(training, world_count=1)
    fake_env = SimpleNamespace(
        world_count=1, observation_dim=486, action_dim=43, reset=lambda: jnp.zeros((1, 486), dtype=jnp.float32)
    )
    with jax.default_device(CPU):
        stock = create_learner(fake_env, training)
        ours = e2.stock_learner_template(training, 486, 43)
    stock_paths = jax.tree_util.tree_flatten_with_path(stock)[0]
    our_paths = jax.tree_util.tree_flatten_with_path(ours)[0]
    assert len(stock_paths) == len(our_paths) == 72
    for (path_a, leaf_a), (path_b, leaf_b) in zip(stock_paths, our_paths, strict=True):
        assert jax.tree_util.keystr(path_a) == jax.tree_util.keystr(path_b)
        assert np.shape(leaf_a) == np.shape(leaf_b) and np.asarray(leaf_a).dtype == np.asarray(leaf_b).dtype
        np.testing.assert_array_equal(np.asarray(leaf_a), np.asarray(leaf_b))


@pytest.mark.skipif(not STOCK_AVAILABLE, reason=f"stock runtime not found at {STOCK_RUNTIME}")
def test_exported_config_instantiates_in_stock_runtime(tmp_path, monkeypatch):
    """Spec 6.5 test 3: the official package instantiates env/embodiment/collision/training and runs the export."""
    implementation = IMPLEMENTATIONS[-1]
    _use_api(monkeypatch, implementation)
    source = tmp_path / "model_20.safetensors"
    learner, training, _ = _write_coordinated_checkpoint(
        source, _c3_like_resolved_config(), implementation, np.random.default_rng(5), iteration=20
    )
    destination = tmp_path / "export.safetensors"
    e2.export_checkpoint(source, destination, stock_runtime=None, probe_count=256)
    observations = e2.make_probe_observations(learner.normalization, 512, 1, 0.01)
    stock = e2.verify_in_stock_runtime(destination, observations, STOCK_RUNTIME)
    assert stock["flash_chord_module"].startswith(str(STOCK_RUNTIME.resolve()))
    assert stock["algorithm"] == "ppo" and stock["leaf_count"] == 72
    assert stock["env_reset_type"] == "ReconBodyResetConfig"
    from flash_chord.training.ppo.evaluation import EvaluationConfig, policy_action

    with jax.default_device(CPU):
        coordinated = np.asarray(
            policy_action(learner, jnp.asarray(observations), training, EvaluationConfig(checkpoint=str(source)))
        )
    assert e2.compare_actions(coordinated, stock["actions"])["max_rel_diff"] <= 1e-5


# --------------------------------------------------------------------------------------------------
# GPU: the official evaluator on an exported checkpoint (spec 6.5 test 4)
# --------------------------------------------------------------------------------------------------


def _stock_env() -> dict:
    environment = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}
    environment["PYTHONPATH"] = str(STOCK_RUNTIME / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return environment


def _evaluator_equivalence(coordinated_checkpoint: Path, tmp_path: Path, episode: int, steps: int = 100) -> dict:
    """Export, record evaluator actions under the STOCK runtime, compare, then run the kit evaluator."""
    from flash_chord.training.checkpoint import load_checkpoint_for_inference, read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config, checkpoint_policy_schema
    from flash_chord.training.ppo.evaluation import EvaluationConfig, policy_action

    exported = tmp_path / "export" / coordinated_checkpoint.name
    report = e2.export_checkpoint(coordinated_checkpoint, exported, stock_runtime=STOCK_RUNTIME, probe_count=2048)
    assert report["status"] == "ok", report

    record = tmp_path / "eval_actions.npz"
    command = [
        sys.executable, str(REPO / "scripts" / "track2" / "record_eval_actions.py"),
        "--checkpoint", str(exported), "--steps", str(steps), "--output", str(record),
        "--runtime-root", str(STOCK_RUNTIME),
    ]  # fmt: skip
    subprocess.run(command, cwd=STOCK_RUNTIME, env=_stock_env(), check=True, timeout=1800)
    with np.load(record) as archive:
        observations = archive["observations"].reshape(-1, 486)
        evaluator_actions = archive["actions"].reshape(-1, 43)
        evaluator_highest = archive["actions_highest_precision"].reshape(-1, 43)

    metadata = read_checkpoint_metadata(coordinated_checkpoint)
    saved = checkpoint_config(metadata)
    api = e2._coordinate_api()
    training = api.training_config(saved["training"])
    layout = api.build_layout(checkpoint_policy_schema(metadata), training.coordinate)
    with jax.default_device(CPU):
        learner, _ = load_checkpoint_for_inference(
            coordinated_checkpoint, api.create_learner_template(training, 486, 43, layout, world_count=1)
        )
        coordinated = np.asarray(
            policy_action(
                learner, jnp.asarray(observations), training, EvaluationConfig(checkpoint=str(coordinated_checkpoint))
            )
        )
    exact = e2.compare_actions(coordinated, evaluator_highest)
    default_precision = e2.compare_actions(coordinated, evaluator_actions)
    tf32_noise = e2.compare_actions(evaluator_highest, evaluator_actions)
    print(json.dumps({"evaluator_highest_vs_coordinated": exact, "evaluator_default_vs_coordinated": default_precision,
                      "evaluator_default_vs_highest": tf32_noise}, indent=2))  # fmt: skip
    assert exact["finite"] and exact["max_rel_diff"] <= 1e-5, exact
    # The evaluator's own default GPU matmul precision (TF32-like) is the only remaining difference.
    assert default_precision["finite"] and default_precision["max_rel_diff"] <= 5e-2, default_precision

    results = tmp_path / "results"
    shard = tmp_path / "checkpoints.json"
    shard.write_text(json.dumps({str(episode): str(exported)}))
    kit_command = [
        sys.executable, str(KIT_ROOT / "examples" / "run_policy_evaluation.py"),
        "--eval-script", str(KIT_ROOT / "flash_chord" / "scripts" / "evaluate_policy.py"),
        "--runtime-root", str(STOCK_RUNTIME), "--world-count", "1",
        "--checkpoints", str(shard), "--results", str(results),
    ]  # fmt: skip
    subprocess.run(kit_command, cwd=KIT_ROOT, env=_stock_env(), check=True, timeout=3600)
    stem = f"episode_{episode:06d}"
    assert (results / f"{stem}.parquet").is_file() and (results / f"{stem}.json").is_file()
    evaluated = json.loads((results / f"{stem}.json").read_text())
    assert evaluated["checkpoint_sha256"] == e2.file_sha256(exported)
    return {"export": report, "exact": exact, "default_precision": default_precision, "evaluation": evaluated}


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not STOCK_AVAILABLE or not KIT_ROOT.is_dir(), reason="stock runtime or kit missing")
@pytest.mark.skipif("FC_COORD_SMOKE_CKPT" not in os.environ, reason="set FC_COORD_SMOKE_CKPT to a C3 smoke checkpoint")
def test_kit_evaluator_matches_coordinated_actor(tmp_path):
    checkpoint = Path(os.environ["FC_COORD_SMOKE_CKPT"]).resolve()
    _evaluator_equivalence(checkpoint, tmp_path, int(os.environ.get("FC_COORD_SMOKE_EPISODE", "17")))


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not STOCK_AVAILABLE or not KIT_ROOT.is_dir(), reason="stock runtime or kit missing")
@pytest.mark.skipif("FC_STOCK_SMOKE_CKPT" not in os.environ, reason="set FC_STOCK_SMOKE_CKPT to a stock G1 checkpoint")
def test_kit_evaluator_matches_synthetic_coordinated_actor(tmp_path, monkeypatch):
    """Same as above without a trained C3 run: random coordinated actor on a real G1 resolved config."""
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config, checkpoint_policy_schema

    implementation = IMPLEMENTATIONS[-1]
    _use_api(monkeypatch, implementation)
    stock_checkpoint = Path(os.environ["FC_STOCK_SMOKE_CKPT"]).resolve()
    stock_metadata = read_checkpoint_metadata(stock_checkpoint)
    config = checkpoint_config(stock_metadata)
    coordinate = {"_target_": "flash_chord.training.ppo.config.CoordinateConfig"}
    coordinate.update(asdict(_ContractCoordinateConfig()))
    coordinate = {k: list(v) if isinstance(v, tuple) else v for k, v in coordinate.items()}
    config["training"]["coordinate"] = coordinate
    for root in (config, config["env"]):
        root["reset"] = _rsi_reset_mapping(root["reset"])
    source = tmp_path / "coord" / "model_30.safetensors"
    _write_coordinated_checkpoint(
        source,
        config,
        implementation,
        np.random.default_rng(17),
        iteration=30,
        head_scale=0.05,
        schema=checkpoint_policy_schema(stock_metadata),
    )
    _evaluator_equivalence(source, tmp_path, int(os.environ.get("FC_STOCK_SMOKE_EPISODE", "17")))
