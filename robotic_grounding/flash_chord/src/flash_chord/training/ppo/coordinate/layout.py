# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name-resolved agent views, action scatter and left/right mirror for the coordinated G1+Dex3 actor.

Every index is resolved from the checkpoint policy schema (observation block names/widths and the two
action block names, which embed the joint names) plus the G1 scalar joint order. Mirror signs come from
the URDF: a left/right joint pair with the same axis ``a`` and mirrored parent frames satisfies
``q_R = s q_L`` with ``s = -(M a) . a`` under the sagittal reflection ``M = diag(1, -1, 1)``.

This module is pure Python/NumPy. The reference constants it checks against live in Warp/Newton
modules and are imported lazily inside :func:`build_layout`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VEC_SIGN = (1.0, -1.0, 1.0)
ROT6D_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)
HAND_CANONICAL = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
    "hand_thumb_0",
    "hand_thumb_1",
    "hand_thumb_2",
    "hand_middle_0",
    "hand_middle_1",
    "hand_index_0",
    "hand_index_1",
)
HAND_MIRROR_SIGNS = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
BODY_CANONICAL = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
BODY_MIRROR_PERM = (6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14)
BODY_MIRROR_SIGNS = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0)

AGENT_WEIGHT_GROUP = {"left": "hand", "right": "hand", "body": "body"}
SONIC_BODY_ACTION_BLOCK = "sonic_body_joint_residual"
DEX3_ACTION_BLOCK = "dex3_direct_joint_residual"
DEFAULT_URDF_RELATIVE_PATH = Path("urdfs") / "g1" / "main_with_hand.urdf"

_MIRROR = np.diag(VEC_SIGN)
_URDF_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class InputView:
    """A signed gather ``x[indices] * signs`` from the normalized policy observation."""

    name: str
    indices: tuple[int, ...]
    signs: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(int(index) for index in self.indices))
        object.__setattr__(self, "signs", tuple(float(sign) for sign in self.signs))
        if len(self.indices) != len(self.signs):
            raise ValueError(f"view {self.name!r} has {len(self.indices)} indices and {len(self.signs)} signs")
        if any(sign not in (1.0, -1.0) for sign in self.signs):
            raise ValueError(f"view {self.name!r} signs must be +/-1")

    @property
    def dim(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class CoordinateLayout:
    """Static, hashable index tables shared by the network, the export and the tests."""

    observation_dim: int
    action_dim: int
    agent_names: tuple[str, ...]
    weight_group: tuple[str, ...]
    actor_inputs: tuple[InputView, ...]
    hand_descriptors: tuple[InputView, InputView]
    body_descriptor: InputView
    body_descriptor_mirrored: InputView
    action_indices: tuple[tuple[int, ...], ...]
    action_signs: tuple[tuple[float, ...], ...]
    action_group: tuple[int, ...]
    observation_mirror: tuple[tuple[int, ...], tuple[float, ...]]

    @property
    def num_agents(self) -> int:
        return len(self.agent_names)


# --------------------------------------------------------------------------------------------- URDF


def default_urdf_path() -> Path:
    from flash_chord.assets import ASSETS_DIR

    return ASSETS_DIR / DEFAULT_URDF_RELATIVE_PATH


def _floats(text: str | None, count: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(count, default, dtype=np.float64)
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (count,):
        raise ValueError(f"expected {count} floats, got {text!r}")
    return values


@dataclass(frozen=True)
class URDFJoint:
    """The subset of a URDF joint needed for mirror checks and forward kinematics."""

    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def parse_urdf_joints(urdf_path: Path) -> dict[str, URDFJoint]:
    """Parse every URDF joint with ``xml.etree`` (no simulator)."""
    root = ET.parse(str(urdf_path)).getroot()
    joints: dict[str, URDFJoint] = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        limit = element.find("limit")
        joints[element.get("name")] = URDFJoint(
            name=element.get("name"),
            joint_type=element.get("type"),
            parent=element.find("parent").get("link"),
            child=element.find("child").get("link"),
            xyz=_floats(None if origin is None else origin.get("xyz"), 3),
            rpy=_floats(None if origin is None else origin.get("rpy"), 3),
            axis=np.asarray((1.0, 0.0, 0.0)) if axis is None else _floats(axis.get("xyz"), 3),
            lower=None if limit is None or limit.get("lower") is None else float(limit.get("lower")),
            upper=None if limit is None or limit.get("upper") is None else float(limit.get("upper")),
        )
    return joints


def _mirror_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name.removeprefix("left_")
    if name.startswith("right_"):
        return "left_" + name.removeprefix("right_")
    return name


def _axis_sign(joint: URDFJoint) -> float:
    axis = joint.axis
    nonzero = np.flatnonzero(np.abs(axis) > 1.0e-9)
    if nonzero.size != 1 or not math.isclose(abs(float(axis[nonzero[0]])), 1.0, abs_tol=1.0e-9):
        raise ValueError(f"joint {joint.name!r} axis {axis.tolist()} is not a coordinate axis")
    return float(-(_MIRROR @ axis) @ axis)


def derive_mirror_signs(urdf_path: Path | None = None) -> dict[str, float]:
    """Return ``{joint: s}`` with ``q_mirror = s q`` for every left and centre revolute joint.

    Left joints are keyed by their own (left) name and mirror to the ``right_`` joint; centre joints
    (no side prefix) mirror onto themselves. The check requires, for every pair, identical axes,
    origins mirrored as ``xyz -> M xyz`` and ``rpy -> (-r, p, -y)``, mirrored parent/child links, and
    limits mirrored for ``s = -1`` or equal for ``s = +1``. Centre joints need self-mirrored origins and
    symmetric limits when ``s = -1``.
    """
    joints = parse_urdf_joints(default_urdf_path() if urdf_path is None else Path(urdf_path))
    signs: dict[str, float] = {}
    for name, joint in joints.items():
        if joint.joint_type not in ("revolute", "continuous") or name.startswith("right_"):
            continue
        partner_name = _mirror_name(name)
        if partner_name not in joints:
            raise ValueError(f"URDF joint {name!r} has no mirror partner {partner_name!r}")
        partner = joints[partner_name]
        if partner.joint_type != joint.joint_type:
            raise ValueError(f"URDF joints {name!r}/{partner_name!r} have different types")
        if not np.allclose(partner.axis, joint.axis, atol=1.0e-9):
            raise ValueError(f"URDF joints {name!r}/{partner_name!r} have different axes")
        if partner.parent != _mirror_name(joint.parent) or partner.child != _mirror_name(joint.child):
            raise ValueError(f"URDF joints {name!r}/{partner_name!r} do not connect mirrored links")
        sign = _axis_sign(joint)
        mirrored_xyz = _MIRROR @ joint.xyz
        mirrored_rpy = joint.rpy * np.asarray((-1.0, 1.0, -1.0))
        if not np.allclose(partner.xyz, mirrored_xyz, atol=_URDF_TOLERANCE):
            raise ValueError(
                f"URDF joint origins {name!r}/{partner_name!r} are not mirrored: {joint.xyz}/{partner.xyz}"
            )
        if not np.allclose(partner.rpy, mirrored_rpy, atol=_URDF_TOLERANCE):
            raise ValueError(
                f"URDF joint rotations {name!r}/{partner_name!r} are not mirrored: {joint.rpy}/{partner.rpy}"
            )
        if joint.lower is not None and joint.upper is not None:
            if sign > 0.0:
                expected = (joint.lower, joint.upper)
            else:
                expected = (-joint.upper, -joint.lower)
            if not np.allclose((partner.lower, partner.upper), expected, atol=_URDF_TOLERANCE):
                raise ValueError(
                    f"URDF limits of {name!r}/{partner_name!r} do not mirror with sign {sign:+.0f}: "
                    f"{(joint.lower, joint.upper)} vs {(partner.lower, partner.upper)}"
                )
        signs[name] = sign
    return signs


# ------------------------------------------------------------------------------------- schema parse


def _bracket_items(name: str) -> tuple[str, list[str]]:
    match = re.fullmatch(r"([^\[\]]+)\[(.*)\]", name)
    if match is None:
        raise ValueError(f"schema block name {name!r} has no bracketed item list")
    return match.group(1), [item.strip() for item in match.group(2).split(",")]


def _schema_blocks(component: Mapping, label: str) -> list[tuple[str, int, int]]:
    blocks = component.get("blocks") if isinstance(component, Mapping) else None
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ValueError(f"policy schema {label} has no block list")
    parsed = []
    cursor = 0
    for block in blocks:
        name, start, end = block["name"], int(block["start"]), int(block["end"])
        if start != cursor or end <= start:
            raise ValueError(f"policy schema {label} block {name!r} is not contiguous at {cursor}")
        parsed.append((name, start, end))
        cursor = end
    if int(component.get("dimension", cursor)) != cursor:
        raise ValueError(f"policy schema {label} blocks cover {cursor} values, not {component.get('dimension')}")
    return parsed


def _action_joint_order(policy_schema: Mapping) -> tuple[str, ...]:
    blocks = _schema_blocks(policy_schema.get("action"), "action")
    if [_bracket_items(name)[0] for name, _, _ in blocks] != [SONIC_BODY_ACTION_BLOCK, DEX3_ACTION_BLOCK]:
        raise ValueError(
            f"coordinated actor requires action blocks [{SONIC_BODY_ACTION_BLOCK}, {DEX3_ACTION_BLOCK}], "
            f"got {[name for name, _, _ in blocks]}"
        )
    order: list[str] = []
    for name, start, end in blocks:
        _, joints = _bracket_items(name)
        if len(joints) != end - start:
            raise ValueError(f"action block {name!r} lists {len(joints)} joints for width {end - start}")
        order.extend(joints)
    if len(set(order)) != len(order):
        raise ValueError("action joint names are not unique")
    return tuple(order)


def _reference_constants() -> tuple[tuple[tuple[str, int], ...], tuple[str, ...]]:
    from flash_chord.embodiments.g1_dex3 import G1_BODY_JOINT_NAMES, G1_DEX3_JOINT_NAMES
    from flash_chord.envs.recon_body import RECON_BODY_OBSERVATION_BLOCKS

    scalar = G1_BODY_JOINT_NAMES[:22] + G1_DEX3_JOINT_NAMES[:7] + G1_BODY_JOINT_NAMES[22:] + G1_DEX3_JOINT_NAMES[7:]
    return tuple(RECON_BODY_OBSERVATION_BLOCKS), tuple(scalar)


# ----------------------------------------------------------------------------------- layout builder


class _Observation:
    """Offsets and per-side slots of the ReconBody observation, resolved from block names."""

    def __init__(self, blocks: Sequence[tuple[str, int, int]], scalar_names: Sequence[str]) -> None:
        self.start = {}
        self.width = {}
        self.items = {}
        for name, start, end in blocks:
            base, _, rest = name.partition("[")
            self.start[base] = start
            self.width[base] = end - start
            self.items[base] = rest.rstrip("]")
        self.dimension = blocks[-1][2]
        self.scalar_names = tuple(scalar_names)
        self.joint_count = len(self.scalar_names)
        self.scalar_index = {name: index for index, name in enumerate(self.scalar_names)}
        self.wrist_sides = self._sides("wrist_position_pelvis_m")
        for base in ("wrist_orientation_pelvis_6d", "wrist_linear_velocity_pelvis_mps"):
            if self._sides(base) != self.wrist_sides:
                raise ValueError(f"block {base} side order differs from the wrist position block")
        self.palm_sides = self._sides("future_palm_position_delta_w_m")
        if self._sides("future_palm_orientation_delta_6d") != self.palm_sides:
            raise ValueError("future palm orientation side order differs from the palm position block")
        self.object_sides = self._sides("live_hand_pose_in_object")
        self.future_count = len(self.items["future_root_position_w_m"].removeprefix("offsets=").split(","))
        self.history_count = self.width["processed_target_history"] // self.joint_count
        if self.width["future_joint_position_delta_rad"] != self.future_count * self.joint_count:
            raise ValueError("future joint block width does not match offsets x joints")
        if self.history_count * self.joint_count != self.width["processed_target_history"]:
            raise ValueError("history block width is not a multiple of the joint count")

    def _sides(self, base: str) -> tuple[str, ...]:
        sides = tuple(self.items[base].split(";")[-1].split(","))
        if sorted(sides) != ["left", "right"]:
            raise ValueError(f"block {base} does not list left/right sides: {self.items[base]!r}")
        return sides

    def vector(self, base: str, slot: int, width: int) -> list[int]:
        start = self.start[base] + slot * width
        return list(range(start, start + width))

    def joint(self, base: str, group: int, name: str) -> int:
        return self.start[base] + group * self.joint_count + self.scalar_index[name]


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_{joint}_joint" for joint in HAND_CANONICAL)


class _ViewBuilder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.indices: list[int] = []
        self.signs: list[float] = []

    def add(self, indices: Sequence[int], signs: Sequence[float]) -> None:
        if len(indices) != len(signs):
            raise ValueError("view segment indices and signs differ in length")
        self.indices.extend(int(index) for index in indices)
        self.signs.extend(float(sign) for sign in signs)

    def build(self) -> InputView:
        return InputView(self.name, tuple(self.indices), tuple(self.signs))


def _hand_view(obs: _Observation, side: str, world_terms: str) -> InputView:
    mirrored = side == "right"
    vec = VEC_SIGN if mirrored else (1.0,) * 3
    rot = ROT6D_SIGN if mirrored else (1.0,) * 6
    joint_signs = HAND_MIRROR_SIGNS if mirrored else (1.0,) * len(HAND_CANONICAL)
    joints = _hand_joint_names(side)
    wrist = obs.wrist_sides.index(side)
    palm = obs.palm_sides.index(side)
    view = _ViewBuilder(f"x_{side}")
    view.add(obs.vector("wrist_position_pelvis_m", wrist, 3), vec)
    view.add(obs.vector("wrist_orientation_pelvis_6d", wrist, 6), rot)
    view.add(obs.vector("wrist_linear_velocity_pelvis_mps", wrist, 3), vec)
    view.add([obs.joint("joint_position_relative_default_rad", 0, name) for name in joints], joint_signs)
    view.add([obs.joint("joint_velocity_radps", 0, name) for name in joints], joint_signs)
    for future in range(obs.future_count):
        view.add([obs.joint("future_joint_position_delta_rad", future, name) for name in joints], joint_signs)
    if world_terms == "reflect":
        for future in range(obs.future_count):
            view.add(obs.vector("future_palm_position_delta_w_m", 2 * future + palm, 3), vec)
    for future in range(obs.future_count):
        view.add(obs.vector("future_palm_orientation_delta_6d", 2 * future + palm, 6), rot)
    pose = obs.vector("live_hand_pose_in_object", obs.object_sides.index(side), 9)
    view.add(pose, (*vec, *rot))
    for slot in range(obs.history_count):
        view.add([obs.joint("processed_target_history", slot, name) for name in joints], joint_signs)
    _add_shared(view, obs, vec, rot)
    return view.build()


def _add_shared(view: _ViewBuilder, obs: _Observation, vec: Sequence[float], rot: Sequence[float]) -> None:
    view.add(obs.vector("object_position_pelvis_m", 0, 3), vec)
    view.add(obs.vector("object_orientation_pelvis_6d", 0, 6), rot)
    view.add(obs.vector("target_object_pose_in_live_object", 0, 9), (*vec, *rot))
    view.add(obs.vector("normalized_reference_phase", 0, 1), (1.0,))


def _body_view(obs: _Observation) -> InputView:
    ones = (1.0,) * len(BODY_CANONICAL)
    view = _ViewBuilder("x_body")
    view.add([obs.joint("joint_position_relative_default_rad", 0, name) for name in BODY_CANONICAL], ones)
    view.add([obs.joint("joint_velocity_radps", 0, name) for name in BODY_CANONICAL], ones)
    for future in range(obs.future_count):
        view.add([obs.joint("future_joint_position_delta_rad", future, name) for name in BODY_CANONICAL], ones)
    for slot in range(obs.history_count):
        view.add([obs.joint("processed_target_history", slot, name) for name in BODY_CANONICAL], ones)
    root = obs.vector("future_root_position_w_m", 0, obs.width["future_root_position_w_m"])
    view.add(root, (1.0,) * len(root))
    orientation = obs.vector("future_root_orientation_delta_6d", 0, obs.width["future_root_orientation_delta_6d"])
    view.add(orientation, (1.0,) * len(orientation))
    wrists = obs.vector("wrist_position_pelvis_m", 0, obs.width["wrist_position_pelvis_m"])
    view.add(wrists, (1.0,) * len(wrists))
    _add_shared(view, obs, (1.0,) * 3, (1.0,) * 6)
    return view.build()


def _hand_descriptor(obs: _Observation, side: str) -> InputView:
    mirrored = side == "right"
    vec = VEC_SIGN if mirrored else (1.0,) * 3
    rot = ROT6D_SIGN if mirrored else (1.0,) * 6
    joint_signs = HAND_MIRROR_SIGNS if mirrored else (1.0,) * len(HAND_CANONICAL)
    joints = _hand_joint_names(side)
    view = _ViewBuilder(f"d_{side}")
    view.add(obs.vector("live_hand_pose_in_object", obs.object_sides.index(side), 9), (*vec, *rot))
    view.add([obs.joint("joint_position_relative_default_rad", 0, name) for name in joints], joint_signs)
    view.add([obs.joint("joint_velocity_radps", 0, name) for name in joints], joint_signs)
    newest = obs.history_count - 1
    view.add([obs.joint("processed_target_history", newest, name) for name in joints], joint_signs)
    return view.build()


def _body_descriptor(obs: _Observation, mirrored: bool) -> InputView:
    if mirrored:
        names = tuple(BODY_CANONICAL[index] for index in BODY_MIRROR_PERM)
        signs = BODY_MIRROR_SIGNS
    else:
        names = BODY_CANONICAL
        signs = (1.0,) * len(BODY_CANONICAL)
    view = _ViewBuilder("d_body_mirrored" if mirrored else "d_body")
    view.add([obs.joint("joint_position_relative_default_rad", 0, name) for name in names], signs)
    view.add([obs.joint("joint_velocity_radps", 0, name) for name in names], signs)
    newest = obs.history_count - 1
    view.add([obs.joint("processed_target_history", newest, name) for name in names], signs)
    return view.build()


def _joint_mirror(obs: _Observation, joint_signs: Mapping[str, float]) -> tuple[list[int], list[float]]:
    permutation = []
    signs = []
    for name in obs.scalar_names:
        partner = _mirror_name(name)
        key = name if not name.startswith("right_") else partner
        if key not in joint_signs:
            raise ValueError(f"no URDF mirror sign for scalar joint {name!r}")
        permutation.append(obs.scalar_index[partner])
        signs.append(joint_signs[key])
    return permutation, signs


SignedPermutation = tuple[tuple[int, ...], tuple[float, ...]]


def _observation_mirror(obs: _Observation, joint_signs: Mapping[str, float]) -> SignedPermutation:
    permutation = np.full(obs.dimension, -1, dtype=np.int64)
    signs = np.zeros(obs.dimension, dtype=np.float64)

    def side_swap(base: str, sides: Sequence[str], width: int, pattern: Sequence[float], groups: int = 1) -> None:
        for group in range(groups):
            for side_index, side in enumerate(sides):
                other = sides.index("left" if side == "right" else "right")
                source = obs.vector(base, group * len(sides) + other, width)
                target = obs.vector(base, group * len(sides) + side_index, width)
                permutation[target] = source
                signs[target] = pattern

    def self_map(base: str, width: int, pattern: Sequence[float]) -> None:
        for group in range(obs.width[base] // width):
            target = obs.vector(base, group, width)
            permutation[target] = target
            signs[target] = pattern

    joint_permutation, joint_sign = _joint_mirror(obs, joint_signs)

    def joint_map(base: str) -> None:
        for group in range(obs.width[base] // obs.joint_count):
            offset = obs.start[base] + group * obs.joint_count
            permutation[offset : offset + obs.joint_count] = offset + np.asarray(joint_permutation)
            signs[offset : offset + obs.joint_count] = joint_sign

    side_swap("wrist_position_pelvis_m", obs.wrist_sides, 3, VEC_SIGN)
    side_swap("wrist_orientation_pelvis_6d", obs.wrist_sides, 6, ROT6D_SIGN)
    side_swap("wrist_linear_velocity_pelvis_mps", obs.wrist_sides, 3, VEC_SIGN)
    self_map("object_position_pelvis_m", 3, VEC_SIGN)
    self_map("object_orientation_pelvis_6d", 6, ROT6D_SIGN)
    joint_map("joint_position_relative_default_rad")
    joint_map("joint_velocity_radps")
    self_map("future_root_position_w_m", 3, VEC_SIGN)
    self_map("future_root_orientation_delta_6d", 6, ROT6D_SIGN)
    joint_map("future_joint_position_delta_rad")
    side_swap("future_palm_position_delta_w_m", obs.palm_sides, 3, VEC_SIGN, groups=obs.future_count)
    side_swap("future_palm_orientation_delta_6d", obs.palm_sides, 6, ROT6D_SIGN, groups=obs.future_count)
    side_swap("live_hand_pose_in_object", obs.object_sides, 9, (*VEC_SIGN, *ROT6D_SIGN))
    self_map("target_object_pose_in_live_object", 9, (*VEC_SIGN, *ROT6D_SIGN))
    self_map("normalized_reference_phase", 1, (1.0,))
    joint_map("processed_target_history")

    if np.any(permutation < 0) or np.any(signs == 0.0):
        raise ValueError("observation mirror does not cover every observation index")
    if not np.array_equal(permutation[permutation], np.arange(obs.dimension)):
        raise ValueError("observation mirror permutation is not an involution")
    if not np.all(signs * signs[permutation] == 1.0):
        raise ValueError("observation mirror signs are not an involution")
    return tuple(int(index) for index in permutation), tuple(float(sign) for sign in signs)


def _compose(view: InputView, mirror: SignedPermutation) -> SignedPermutation:
    """Indices/signs of ``view(M x)`` expressed directly on ``x``."""
    permutation, signs = mirror
    return (
        tuple(permutation[index] for index in view.indices),
        tuple(sign * signs[index] for index, sign in zip(view.indices, view.signs, strict=True)),
    )


def build_layout(
    policy_schema: Mapping,
    config,
    urdf_path: Path | None = None,
) -> CoordinateLayout:
    """Resolve every view, action scatter and the observation mirror from the checkpoint policy schema."""
    if tuple(config.agents) != ("left", "right", "body"):
        raise ValueError(f"coordinated layout supports agents ('left', 'right', 'body'), got {config.agents}")
    expected_blocks, scalar_names = _reference_constants()
    observation_blocks = _schema_blocks(policy_schema.get("observation"), "observation")
    actual = tuple((name, end - start) for name, start, end in observation_blocks)
    if actual != tuple((name, int(width)) for name, width in expected_blocks):
        raise ValueError(
            "coordinated actor requires the exact ReconBody observation blocks; got "
            f"{[name for name, _ in actual]}"
        )
    obs = _Observation(observation_blocks, scalar_names)
    if obs.scalar_names[: len(BODY_CANONICAL)] != BODY_CANONICAL:
        raise ValueError("scalar joints 0..14 are not the canonical body joints")

    joint_signs = derive_mirror_signs(urdf_path)
    hand_signs = tuple(joint_signs[name] for name in _hand_joint_names("left"))
    if hand_signs != HAND_MIRROR_SIGNS:
        raise ValueError(f"URDF hand mirror signs {hand_signs} differ from HAND_MIRROR_SIGNS {HAND_MIRROR_SIGNS}")
    for index, name in enumerate(BODY_CANONICAL):
        partner = BODY_CANONICAL[BODY_MIRROR_PERM[index]]
        if partner != _mirror_name(name):
            raise ValueError(f"BODY_MIRROR_PERM maps {name!r} to {partner!r}")
        key = partner if name.startswith("right_") else name
        if joint_signs[key] != BODY_MIRROR_SIGNS[index]:
            raise ValueError(
                f"URDF mirror sign of {name!r} is {joint_signs[key]}, table has {BODY_MIRROR_SIGNS[index]}"
            )

    action_order = _action_joint_order(policy_schema)
    if sorted(action_order) != sorted(obs.scalar_names):
        raise ValueError("action joints are not the 43 scalar joints")
    action_index = {name: index for index, name in enumerate(action_order)}
    agent_joints = (_hand_joint_names("left"), _hand_joint_names("right"), BODY_CANONICAL)
    action_indices = tuple(tuple(action_index[name] for name in names) for names in agent_joints)
    action_signs = (
        (1.0,) * len(HAND_CANONICAL),
        tuple(float(sign) for sign in HAND_MIRROR_SIGNS),
        (1.0,) * len(BODY_CANONICAL),
    )
    flat = [index for indices in action_indices for index in indices]
    if sorted(flat) != list(range(len(action_order))):
        raise ValueError("agent action indices do not partition the action vector")
    action_group = [0] * len(action_order)
    for agent, indices in enumerate(action_indices):
        for index in indices:
            action_group[index] = agent

    world_terms = config.hand_world_terms
    x_left = _hand_view(obs, "left", world_terms)
    x_right = _hand_view(obs, "right", world_terms)
    x_body = _body_view(obs)
    d_left = _hand_descriptor(obs, "left")
    d_right = _hand_descriptor(obs, "right")
    d_body = _body_descriptor(obs, mirrored=False)
    d_body_mirrored = _body_descriptor(obs, mirrored=True)
    mirror = _observation_mirror(obs, joint_signs)

    for left, right in ((x_left, x_right), (d_left, d_right), (d_body, d_body_mirrored)):
        if _compose(left, mirror) != (right.indices, right.signs):
            raise ValueError(f"view {right.name} is not {left.name} composed with the observation mirror")
    covered = set(x_left.indices) | set(x_right.indices) | set(x_body.indices)
    covered |= set(d_left.indices) | set(d_right.indices) | set(d_body.indices)
    if world_terms == "reflect" and covered != set(range(obs.dimension)):
        raise ValueError(f"agent views leave observation indices unused: {sorted(set(range(obs.dimension)) - covered)}")

    return CoordinateLayout(
        observation_dim=obs.dimension,
        action_dim=len(action_order),
        agent_names=("left", "right", "body"),
        weight_group=("hand", "hand", "body"),
        actor_inputs=(x_left, x_right, x_body),
        hand_descriptors=(d_left, d_right),
        body_descriptor=d_body,
        body_descriptor_mirrored=d_body_mirrored,
        action_indices=action_indices,
        action_signs=action_signs,
        action_group=tuple(action_group),
        observation_mirror=mirror,
    )


def layout_digest(layout: CoordinateLayout) -> str:
    """SHA-256 of the layout's canonical JSON (stored as ``coordinate_layout_sha256``)."""
    payload = json.dumps(dataclasses.asdict(layout), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mirror_observation(layout: CoordinateLayout, observation):
    """Apply the full-world mirror ``M`` to a NumPy or JAX observation batch (last axis)."""
    permutation, signs = layout.observation_mirror
    return observation[..., np.asarray(permutation)] * np.asarray(signs, dtype=np.float32)


def view_ranges(view: InputView) -> list[tuple[int, int]]:
    """Inclusive contiguous index ranges of a view, in view order (diagnostics and tests)."""
    ranges: list[tuple[int, int]] = []
    for index in view.indices:
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], index)
        else:
            ranges.append((index, index))
    return ranges
