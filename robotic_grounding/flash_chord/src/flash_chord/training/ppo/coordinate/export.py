# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2 export: embed a coordinated W0-P actor exactly into a stock FlashCHORD ``ActorCritic`` checkpoint.

The coordinated actor (docs/coord_impl_spec.md section 3.3) is block-structured. It embeds exactly
into the stock four-hidden-layer ELU MLP (``flash_chord/training/ppo/network.py``):

* ``Dense_0``: the three agent encoders ``e_L, e_R, e_B`` and the partner descriptors
  ``p_L, p_R, p_B`` (+ ``p_BM``) as signed gathers of the normalized observation.
* ``Dense_1``: a pass-through ``ELU(e_i + 1) = e_i + 1`` (exact because ``ELU >= -1``) and the
  three partner messages ``m_i``.
* ``Dense_2``: the first trunk layer per agent; the ``+1`` of the pass-through is folded into the bias.
* ``Dense_3``: the second trunk layer per agent.
* ``Dense_4``: the per-agent heads scattered into env action order with the mirror signs.

Unused units have zero kernels, biases, and outgoing weights, so ``ELU(0) = 0`` contributes nothing.
The exported checkpoint uses the stock learner-state leaves, the stock config targets, and the stock
policy schema, so the unmodified official evaluator restores and runs it.

CLI::

    python -m flash_chord.training.ppo.coordinate.export --input SRC --output DST \
        [--stock-runtime $STOCK] [--probe obs.npz] [--report report.json]

The export refuses to write ``DST`` unless the coordinated and the exported deterministic actions
agree to 1e-5 (relative to ``max(1, |a|)``) on a seeded probe batch, computed in float32 on the CPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

EXPORT_FORMAT = "coordinate_e2_v1"
STOCK_HIDDEN_DEPTH = 4
STOCK_ACTOR_HIDDEN_DIMS = (1024, 512, 256, 128)
STOCK_PPO_TRAINING_TARGET = "flash_chord.training.ppo.config.TrainingConfig"
STOCK_RESET_TARGET = "flash_chord.lifecycle.reset.ReconBodyResetConfig"
#: The ten fields of the stock ``ReconBodyResetConfig`` (lifecycle/reset.py:609-621).
STOCK_RESET_FIELDS = (
    "reset_freeze_steps",
    "voc_decay_steps",
    "reset_voc_scale",
    "shoulder_spread",
    "start_frame",
    "end_frame",
    "always_reset_to_first_frame",
    "reset_to_first_frame_probability",
    "immediate_first_frame_probability",
    "seed",
)
#: Stock reset fields that the export zeroes; the evaluator overrides them anyway (harness.py:228-240).
STOCK_RESET_ZEROED = ("reset_to_first_frame_probability", "immediate_first_frame_probability")
#: ``_target_`` plus the fields of the stock PPO ``TrainingConfig`` (training/ppo/config.py:205-226).
STOCK_TRAINING_KEYS = (
    "_target_",
    "seed",
    "world_count",
    "rollout_steps",
    "max_iterations",
    "save_interval",
    "observation_normalization",
    "observation_normalization_epsilon",
    "reward_normalization",
    "device",
    "resume",
    "checkpoint",
    "wandb_entity",
    "wandb_project",
    "experiment_name",
    "run_name",
    "curriculum",
    "network",
    "ppo",
)
#: ``_target_`` plus the stock ``NetworkConfig`` fields (config.py:123-132).
STOCK_NETWORK_KEYS = (
    "_target_",
    "actor_hidden_dims",
    "critic_hidden_dims",
    "activation",
    "initial_noise_std",
    "noise_std_type",
    "actor_output_gain",
)
#: ``_target_`` plus the stock ``PPOConfig`` fields (config.py:149-169).
STOCK_PPO_KEYS = (
    "_target_",
    "value_loss_coefficient",
    "use_clipped_value_loss",
    "clip_parameter",
    "entropy_coefficient",
    "learning_epochs",
    "mini_batches",
    "learning_rate",
    "schedule",
    "minimum_learning_rate",
    "maximum_learning_rate",
    "adaptive_schedule_factor",
    "discount",
    "gae_lambda",
    "desired_kl",
    "max_grad_norm",
    "normalize_advantage_per_mini_batch",
    "log_ratio_clip",
)
#: Dev-only ``ReconBodyObjectiveConfig`` fields and curriculum term names (objectives/recon_body.py object_task and
#: its post-release options: position-only object_task and the object_no_drop term).
DEV_OBJECTIVE_KEYS = (
    "object_task_weight",
    "object_task_position_beta",
    "object_task_rotation_beta",
    "object_task_position_only_after_release",
    "no_drop_weight",
    "no_drop_margin_m",
)
DEV_OBJECTIVE_TERMS = ("object_task", "object_no_drop")
#: Dev-only ``ReconBodyTerminationConfig`` fields (training only; evaluation disables every threshold anyway).
DEV_TERMINATION_KEYS = ("violation_patience",)
CONFIG_METADATA_KEY = "resolved_config_json"
POLICY_SCHEMA_METADATA_KEY = "policy_schema_json"
LAYOUT_METADATA_KEY = "coordinate_layout_sha256"
EXPORT_FORMAT_METADATA_KEY = "export_format"
EXPORT_SOURCE_METADATA_KEY = "export_source_sha256"
MVP_AGENTS = ("left", "right", "body")
MVP_WEIGHT_GROUPS = ("hand", "hand", "body")
DEFAULT_PROBE_COUNT = 8192
DEFAULT_TOLERANCE = 1.0e-5


class ExportVerificationError(RuntimeError):
    """The exported actor does not reproduce the coordinated actor; nothing was written."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


# --------------------------------------------------------------------------------------------------
# Embedding plan (unit budget)
# --------------------------------------------------------------------------------------------------


def _field(config: Any, name: str):
    """Read one field from a dataclass-like object or a mapping."""
    if isinstance(config, Mapping):
        return config[name]
    return getattr(config, name)


@dataclass(frozen=True)
class EmbeddingPlan:
    """Unit offsets of the coordinated blocks inside the four stock hidden layers (spec section 6.2)."""

    hidden_dims: tuple[int, int, int, int]
    encoder_dim: int
    partner_dim: int
    message_dim: int
    trunk_dims: tuple[int, int]
    partner_message: bool
    mirror_body_message: bool

    # Dense_0: [e_L | e_R | e_B | p_L | p_R | p_B | p_BM]
    def encoder(self, agent: int) -> slice:
        return slice(agent * self.encoder_dim, (agent + 1) * self.encoder_dim)

    def partner(self, name: str) -> slice:
        position = {"left": 0, "right": 1, "body": 2, "body_mirrored": 3}[name]
        start = 3 * self.encoder_dim + position * self.partner_dim
        return slice(start, start + self.partner_dim)

    # Dense_1: [pass-through e~_L | e~_R | e~_B | m_L | m_R | m_B]
    def message(self, agent: int) -> slice:
        start = 3 * self.encoder_dim + agent * self.message_dim
        return slice(start, start + self.message_dim)

    # Dense_2 and Dense_3: one block per agent.
    def trunk(self, layer: int, agent: int) -> slice:
        width = self.trunk_dims[layer]
        return slice(agent * width, (agent + 1) * width)

    @property
    def used_units(self) -> tuple[int, int, int, int]:
        partner_blocks = 4 if self.mirror_body_message else 3
        return (
            3 * self.encoder_dim + partner_blocks * self.partner_dim,
            3 * (self.encoder_dim + self.message_dim),
            3 * self.trunk_dims[0],
            3 * self.trunk_dims[1],
        )

    def budget(self) -> dict[str, list[int]]:
        return {"used": list(self.used_units), "available": list(self.hidden_dims)}


def embedding_plan(coordinate: Any, actor_hidden_dims: Sequence[int]) -> EmbeddingPlan:
    """Validate the E2 budget (spec section 6.2) and return the unit layout."""
    hidden = tuple(int(width) for width in actor_hidden_dims)
    if len(hidden) != STOCK_HIDDEN_DEPTH:
        raise ValueError(f"E2 export needs exactly {STOCK_HIDDEN_DEPTH} stock hidden layers, got {hidden}")
    trunk = tuple(int(width) for width in _field(coordinate, "trunk_dims"))
    if len(trunk) != 2:
        raise ValueError(f"E2 export needs exactly two coordinated trunk layers, got {trunk}")
    plan = EmbeddingPlan(
        hidden_dims=hidden,  # type: ignore[arg-type]
        encoder_dim=int(_field(coordinate, "encoder_dim")),
        partner_dim=int(_field(coordinate, "partner_dim")),
        message_dim=int(_field(coordinate, "message_dim")),
        trunk_dims=trunk,  # type: ignore[arg-type]
        partner_message=bool(_field(coordinate, "partner_message")),
        mirror_body_message=bool(_field(coordinate, "mirror_body_message")),
    )
    for layer, (used, available) in enumerate(zip(plan.used_units, hidden, strict=True)):
        if used > available:
            raise ValueError(
                f"E2 budget exceeded in stock hidden layer {layer}: {used} units needed, {available} available "
                f"(budget used={list(plan.used_units)}, available={list(hidden)})"
            )
    return plan


# --------------------------------------------------------------------------------------------------
# Actor embedding (pure numpy)
# --------------------------------------------------------------------------------------------------


def _actor_param_tree(coord_params: Mapping) -> Mapping:
    """Accept either Flax variables ``{"params": {...}}`` or the inner parameter mapping."""
    if "hand_encoder" not in coord_params and "params" in coord_params:
        return coord_params["params"]
    return coord_params


def _dense(params: Mapping, name: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if name not in params:
        raise KeyError(f"coordinated parameters lack {name!r}; found {sorted(params)}")
    kernel = np.asarray(params[name]["kernel"], dtype=np.float64)
    bias = np.asarray(params[name]["bias"], dtype=np.float64)
    if kernel.shape != shape or bias.shape != (shape[1],):
        raise ValueError(f"{name} has kernel {kernel.shape} and bias {bias.shape}; expected {shape} and ({shape[1]},)")
    return kernel, bias


def _view(view) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(view.indices, dtype=np.int64)
    signs = np.asarray(view.signs, dtype=np.float64)
    if indices.ndim != 1 or indices.shape != signs.shape:
        raise ValueError(f"view {getattr(view, 'name', '?')!r} has {indices.shape} indices and {signs.shape} signs")
    if not np.all(np.abs(signs) == 1.0):
        raise ValueError(f"view {getattr(view, 'name', '?')!r} signs must be +/-1")
    return indices, signs


def _scatter_view(kernel: np.ndarray, view, weights: np.ndarray, units: slice) -> None:
    """Add ``sign_j * weights[j, :]`` to ``kernel[view.indices[j], units]`` (repeated indices accumulate)."""
    indices, signs = _view(view)
    if weights.shape[0] != indices.size:
        raise ValueError(f"view {getattr(view, 'name', '?')!r} has {indices.size} inputs; weights have {weights.shape}")
    columns = np.arange(units.start, units.stop)
    np.add.at(kernel, (indices[:, None], columns[None, :]), signs[:, None] * weights)


def _check_layout(layout) -> None:
    if tuple(layout.agent_names) != MVP_AGENTS or tuple(layout.weight_group) != MVP_WEIGHT_GROUPS:
        raise ValueError(
            f"E2 export supports agents {MVP_AGENTS} with weight groups {MVP_WEIGHT_GROUPS}; "
            f"got {tuple(layout.agent_names)} / {tuple(layout.weight_group)}"
        )
    if len(layout.actor_inputs) != 3 or len(layout.hand_descriptors) != 2:
        raise ValueError("coordinate layout must provide three actor inputs and two hand descriptors")
    covered = np.zeros(int(layout.action_dim), dtype=np.int64)
    for indices, signs in zip(layout.action_indices, layout.action_signs, strict=True):
        if len(indices) != len(signs):
            raise ValueError("layout action indices and signs differ in length")
        covered[np.asarray(indices, dtype=np.int64)] += 1
    if not np.all(covered == 1):
        raise ValueError("layout action indices must partition the action vector exactly")


def embed_actor(
    coord_params: Mapping,
    layout,
    coordinate,
    actor_hidden_dims: Sequence[int],
) -> dict[str, Any]:
    """Return the stock actor subtree ``{Dense_0..Dense_4, log_std}`` computing the coordinated mean.

    ``coord_params`` follows the parameter names of spec section 3.3. ``layout`` is a
    ``CoordinateLayout`` and ``coordinate`` a ``CoordinateConfig`` (or any object or mapping with
    the same fields). Arithmetic is float64 and the result is rounded once to float32.
    """
    params = _actor_param_tree(coord_params)
    plan = embedding_plan(coordinate, actor_hidden_dims)
    _check_layout(layout)
    observation_dim = int(layout.observation_dim)
    action_dim = int(layout.action_dim)
    h1, h2, h3, h4 = plan.hidden_dims
    enc, partner, message = plan.encoder_dim, plan.partner_dim, plan.message_dim
    trunk0, trunk1 = plan.trunk_dims

    x_left, x_right, x_body = layout.actor_inputs
    hand_input = len(x_left.indices)
    if len(x_right.indices) != hand_input:
        raise ValueError("left and right hand views must have the same width")
    encoders = {
        "hand": _dense(params, "hand_encoder", (hand_input, enc)),
        "body": _dense(params, "body_encoder", (len(x_body.indices), enc)),
    }
    trunks = {
        group: (
            _dense(params, f"{group}_trunk_0", (enc + message, trunk0)),
            _dense(params, f"{group}_trunk_1", (trunk0, trunk1)),
        )
        for group in ("hand", "body")
    }
    head_widths = [len(indices) for indices in layout.action_indices]
    heads = {
        "hand": _dense(params, "hand_head", (trunk1, head_widths[0])),
        "body": _dense(params, "body_head", (trunk1, head_widths[2])),
    }
    if head_widths[1] != head_widths[0]:
        raise ValueError(f"left and right hand agents must have equal action widths, got {head_widths}")
    log_std = np.asarray(params["log_std"], dtype=np.float32)
    if log_std.shape != (action_dim,):
        raise ValueError(f"log_std has shape {log_std.shape}; expected ({action_dim},)")

    kernel0 = np.zeros((observation_dim, h1))
    bias0 = np.zeros(h1)
    kernel1 = np.zeros((h1, h2))
    bias1 = np.zeros(h2)
    kernel2 = np.zeros((h2, h3))
    bias2 = np.zeros(h3)
    kernel3 = np.zeros((h3, h4))
    bias3 = np.zeros(h4)
    kernel4 = np.zeros((h4, action_dim))
    bias4 = np.zeros(action_dim)

    # Dense_0: encoders.
    for agent, (view, group) in enumerate(zip(layout.actor_inputs, layout.weight_group, strict=True)):
        weight, bias = encoders[group]
        _scatter_view(kernel0, view, weight, plan.encoder(agent))
        bias0[plan.encoder(agent)] = bias

    # Dense_1: exact pass-through of every encoder unit as ELU(e + 1) = e + 1.
    passthrough = np.arange(3 * enc)
    kernel1[passthrough, passthrough] = 1.0
    bias1[passthrough] = 1.0

    if plan.partner_message:
        d_left, d_right = layout.hand_descriptors
        hand_partner = _dense(params, "hand_partner", (len(d_left.indices), partner))
        body_partner = _dense(params, "body_partner", (len(layout.body_descriptor.indices), partner))
        if len(d_right.indices) != len(d_left.indices):
            raise ValueError("left and right hand descriptors must have the same width")
        if len(layout.body_descriptor_mirrored.indices) != len(layout.body_descriptor.indices):
            raise ValueError("mirrored body descriptor must have the body descriptor's width")
        descriptors = [("left", d_left, hand_partner), ("right", d_right, hand_partner), ("body", layout.body_descriptor, body_partner)]
        if plan.mirror_body_message:
            descriptors.append(("body_mirrored", layout.body_descriptor_mirrored, body_partner))
        for name, view, (weight, bias) in descriptors:
            _scatter_view(kernel0, view, weight, plan.partner(name))
            bias0[plan.partner(name)] = bias

        hand_message = _dense(params, "hand_message", (partner, message))
        body_message = _dense(params, "body_message", (partner, message))
        right_body_partner = "body_mirrored" if plan.mirror_body_message else "body"
        routes = (
            (0, hand_message, ("right", "body")),  # m_L = ELU(Rho_h (p_R + p_B))
            (1, hand_message, ("left", right_body_partner)),  # m_R = ELU(Rho_h (p_L + p_BM))
            (2, body_message, ("left", "right")),  # m_B = ELU(Rho_b (p_L + p_R))
        )
        for agent, (weight, bias), sources in routes:
            for source in sources:
                kernel1[plan.partner(source), plan.message(agent)] += weight
            bias1[plan.message(agent)] = bias

    # Dense_2 / Dense_3: per-agent trunks. The pass-through carries e + 1, so subtract the column sums.
    for agent, group in enumerate(layout.weight_group):
        (weight0, bias_t0), (weight1, bias_t1) = trunks[group]
        units0 = plan.trunk(0, agent)
        kernel2[plan.encoder(agent), units0] = weight0[:enc]
        if plan.partner_message:
            kernel2[plan.message(agent), units0] = weight0[enc:]
        bias2[units0] = bias_t0 - weight0[:enc].sum(axis=0)
        units1 = plan.trunk(1, agent)
        kernel3[units0, units1] = weight1
        bias3[units1] = bias_t1

    # Dense_4: heads scattered into env action order with mirror signs.
    for agent, group in enumerate(layout.weight_group):
        weight, bias = heads[group]
        indices = np.asarray(layout.action_indices[agent], dtype=np.int64)
        signs = np.asarray(layout.action_signs[agent], dtype=np.float64)
        kernel4[plan.trunk(1, agent)][:, indices] = weight * signs[None, :]
        bias4[indices] = bias * signs

    def dense(kernel, bias):
        return {"kernel": kernel.astype(np.float32), "bias": bias.astype(np.float32)}

    return {
        "Dense_0": dense(kernel0, bias0),
        "Dense_1": dense(kernel1, bias1),
        "Dense_2": dense(kernel2, bias2),
        "Dense_3": dense(kernel3, bias3),
        "Dense_4": dense(kernel4, bias4),
        "log_std": log_std.copy(),
    }


# --------------------------------------------------------------------------------------------------
# Config rewrite (spec section 6.3, rules 1-5)
# --------------------------------------------------------------------------------------------------


def _prune(mapping: dict, allowed: Sequence[str], prefix: str, removed: list[str]) -> None:
    for key in sorted(set(mapping) - set(allowed)):
        del mapping[key]
        removed.append(f"{prefix}.{key}")


def _stock_reset(reset: Mapping[str, Any], where: str) -> dict[str, Any]:
    missing = [name for name in STOCK_RESET_FIELDS if name not in reset]
    if missing:
        raise ValueError(
            f"{where} ({reset.get('_target_')!r}) lacks stock ReconBody reset fields {missing}; "
            "E2 export supports ReconBody-family resets only"
        )
    output: dict[str, Any] = {"_target_": STOCK_RESET_TARGET}
    for name in STOCK_RESET_FIELDS:
        output[name] = copy.deepcopy(reset[name])
    for name in STOCK_RESET_ZEROED:
        output[name] = 0.0
    return output


def _strip_dev_objective_terms(config: dict, removed: list[str]) -> None:
    """Drop the dev-only ReconBody ``object_task`` / ``object_no_drop`` terms: objective fields and curriculum weights.

    The stock ``ReconBodyObjectiveConfig`` rejects unknown fields and the stock objective rejects unknown
    curriculum weight names; the term only shapes training reward, so the exported actor is unaffected.
    """
    objectives = (("objective", config.get("objective")), ("env.objective", config["env"].get("objective")))
    for prefix, objective in objectives:
        if isinstance(objective, dict):
            for key in DEV_OBJECTIVE_KEYS:
                if key in objective:
                    del objective[key]
                    removed.append(f"{prefix}.{key}")
    training_curriculum = config["training"].get("curriculum")
    for prefix, curriculum in (("curriculum", config.get("curriculum")), ("training.curriculum", training_curriculum)):
        stages = curriculum.get("stages") if isinstance(curriculum, dict) else None
        for index, stage in enumerate(stages if isinstance(stages, list) else ()):
            weights = stage.get("objective_weights") if isinstance(stage, dict) else None
            if not isinstance(weights, dict):
                continue
            for name in DEV_OBJECTIVE_TERMS:
                if name in weights:
                    del weights[name]
                    removed.append(f"{prefix}.stages.{index}.objective_weights.{name}")


def _strip_dev_termination_keys(config: dict, removed: list[str]) -> None:
    """Drop dev-only ReconBody termination fields, which the stock ``ReconBodyTerminationConfig`` rejects.

    They only shape training episodes; the evaluator disables every termination threshold before scoring.
    """
    terminations = (("termination", config.get("termination")), ("env.termination", config["env"].get("termination")))
    for prefix, termination in terminations:
        if isinstance(termination, dict):
            for key in DEV_TERMINATION_KEYS:
                if key in termination:
                    del termination[key]
                    removed.append(f"{prefix}.{key}")


def stock_config_with_notes(saved_config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``stock_config`` plus a record of every key it removed or replaced."""
    config = copy.deepcopy(dict(saved_config))
    removed: list[str] = []
    training = config.get("training")
    if not isinstance(training, dict):
        raise TypeError("checkpoint resolved config is missing its training mapping")
    if training.get("_target_") != STOCK_PPO_TRAINING_TARGET:
        raise ValueError(f"E2 export needs a PPO training config, got {training.get('_target_')!r}")
    coordinate = training.get("coordinate")
    _prune(training, STOCK_TRAINING_KEYS, "training", removed)
    network = training.get("network")
    ppo = training.get("ppo")
    if not isinstance(network, dict) or not isinstance(ppo, dict):
        raise TypeError("training.network and training.ppo must be mappings")
    _prune(network, STOCK_NETWORK_KEYS, "training.network", removed)
    _prune(ppo, STOCK_PPO_KEYS, "training.ppo", removed)
    if coordinate is not None:
        export_dims = coordinate.get("export_actor_hidden_dims", STOCK_ACTOR_HIDDEN_DIMS)
        network["actor_hidden_dims"] = [int(width) for width in export_dims]
    if len(network["actor_hidden_dims"]) != STOCK_HIDDEN_DEPTH:
        raise ValueError(f"exported actor_hidden_dims must have depth {STOCK_HIDDEN_DEPTH}")
    network["critic_hidden_dims"] = [int(width) for width in network["critic_hidden_dims"]]

    replaced = []
    env = config.get("env")
    if not isinstance(env, dict) or not isinstance(env.get("reset"), dict):
        raise TypeError("checkpoint resolved config is missing env.reset")
    original_reset_target = env["reset"].get("_target_")
    env["reset"] = _stock_reset(env["reset"], "env.reset")
    replaced.append("env.reset")
    if isinstance(config.get("reset"), dict):
        config["reset"] = _stock_reset(config["reset"], "reset")
        replaced.append("reset")
    _strip_dev_objective_terms(config, removed)
    _strip_dev_termination_keys(config, removed)
    notes = {
        "removed_keys": removed,
        "replaced_subtrees": replaced,
        "source_reset_target": original_reset_target,
        "actor_hidden_dims": list(network["actor_hidden_dims"]),
        "critic_hidden_dims": list(network["critic_hidden_dims"]),
    }
    return config, notes


def stock_config(saved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite a DEV resolved config so the official runtime can instantiate it (pure; spec 6.3 rules 1-5).

    1. Remove ``training.coordinate`` and every other non-stock key under ``training``,
       ``training.network`` and ``training.ppo``.
    2. Set ``training.network.actor_hidden_dims`` to ``coordinate.export_actor_hidden_dims`` for
       coordinated sources; keep the source dims otherwise.
    3. Replace ``env.reset`` and the top-level ``reset`` with the stock ``ReconBodyResetConfig``
       (ten stock fields copied; both first-frame probabilities set to 0).
    4./5. Keep everything else (curriculum, task, scene, sim, strategies, embodiment, ...).
    """
    return stock_config_with_notes(saved_config)[0]


# --------------------------------------------------------------------------------------------------
# Coordinated-learner interface (owner A), resolved lazily
# --------------------------------------------------------------------------------------------------


def _coordinate_api() -> SimpleNamespace:
    """Owner A's interfaces (spec sections 3.1, 3.2, 3.5). Tests may replace this function."""
    from omegaconf import OmegaConf

    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.ppo.config import TrainingConfig
    from flash_chord.training.ppo.coordinate.layout import build_layout, layout_digest
    from flash_chord.training.ppo.learner import create_learner_template

    def training_config(training: Mapping[str, Any]):
        config = instantiate_typed(OmegaConf.create(dict(training)), TrainingConfig)
        if getattr(config, "coordinate", None) is None:
            raise ValueError("coordinated source training config did not instantiate a CoordinateConfig")
        return config

    return SimpleNamespace(
        training_config=training_config,
        build_layout=build_layout,
        layout_digest=layout_digest,
        create_learner_template=create_learner_template,
    )


# --------------------------------------------------------------------------------------------------
# Stock learner state
# --------------------------------------------------------------------------------------------------


def _stock_training_config(training: Mapping[str, Any]):
    from omegaconf import OmegaConf

    from flash_chord.configuration import instantiate_typed
    from flash_chord.training.ppo.config import TrainingConfig

    return instantiate_typed(OmegaConf.create(dict(training)), TrainingConfig)


def stock_learner_template(training, observation_dim: int, action_dim: int, world_count: int = 1):
    """Env-free replica of stock ``create_learner`` (training/ppo/learner.py:44-62)."""
    import jax
    import jax.numpy as jnp

    from flash_chord.training.ppo.algorithm import create_train_state
    from flash_chord.training.ppo.learner import LearnerState
    from flash_chord.training.ppo.network import ActorCritic
    from flash_chord.training.ppo.normalization import create_normalization_state

    key, initialization_key = jax.random.split(jax.random.PRNGKey(training.seed))
    model = ActorCritic(action_dim=action_dim, config=training.network)
    parameters = model.init(initialization_key, jnp.zeros((1, observation_dim), dtype=jnp.float32))
    return LearnerState(
        train_state=create_train_state(model, parameters, training.ppo),
        normalization=create_normalization_state(observation_dim),
        observation=jnp.zeros((world_count, observation_dim), dtype=jnp.float32),
        key=key,
        learning_rate=jnp.asarray(training.ppo.learning_rate, dtype=jnp.float32),
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )


def _stock_learner_from_actor(actor: Mapping[str, Any], source, training, observation_dim: int, action_dim: int, seed: int):
    """Stock learner: embedded actor, fresh 1-column critic and Adam state, copied normalizer/key/lr/iteration."""
    import jax
    import jax.numpy as jnp

    from flash_chord.training.ppo.algorithm import create_train_state
    from flash_chord.training.ppo.learner import LearnerState
    from flash_chord.training.ppo.network import ActorCritic

    _, initialization_key = jax.random.split(jax.random.PRNGKey(seed))
    model = ActorCritic(action_dim=action_dim, config=training.network)
    variables = model.init(initialization_key, jnp.zeros((1, observation_dim), dtype=jnp.float32))
    inner = dict(variables["params"])
    for name, value in actor.items():
        if name not in inner:
            raise KeyError(f"stock ActorCritic has no parameter {name!r}")
        template = jax.tree.map(np.shape, inner[name])
        candidate = jax.tree.map(np.shape, value)
        if template != candidate:
            raise ValueError(f"embedded {name} has shapes {candidate}; stock expects {template}")
        inner[name] = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype=jnp.float32), value)
    train_state = create_train_state(model, {"params": inner}, training.ppo)
    return LearnerState(
        train_state=train_state,
        normalization=source.normalization,
        observation=jnp.zeros((1, observation_dim), dtype=jnp.float32),
        key=source.key,
        learning_rate=source.learning_rate,
        iteration=source.iteration,
    )


# --------------------------------------------------------------------------------------------------
# Verification helpers
# --------------------------------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_probe_observations(normalization, count: int, seed: int, epsilon: float, normalized: bool = True) -> np.ndarray:
    """Seeded raw observations whose normalized values are heavy-tailed (half Gaussian, half Student-t(3))."""
    mean = np.asarray(normalization.mean, dtype=np.float64)
    std = np.sqrt(np.asarray(normalization.variance, dtype=np.float64)) + epsilon
    if not normalized:
        mean = np.zeros_like(mean)
        std = np.ones_like(std)
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((count // 2, mean.size))
    heavy = np.clip(rng.standard_t(3.0, size=(count - count // 2, mean.size)), -30.0, 30.0)
    unit = np.concatenate([gaussian, heavy], axis=0)
    return (mean + std * unit).astype(np.float32)


def load_probe(path: str | Path, observation_dim: int) -> np.ndarray:
    """Load raw observations from ``.npy`` or ``.npz`` (key ``observations`` or ``observation``)."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as archive:
            key = "observations" if "observations" in archive else "observation"
            values = archive[key]
    else:
        values = np.load(path)
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != observation_dim:
        raise ValueError(f"probe {path} has last dimension {values.shape[-1]}; expected {observation_dim}")
    return values.reshape(-1, observation_dim)


def compare_actions(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Absolute and scaled differences; ``max_rel_diff`` divides by ``max(1, |reference|)``."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(f"action shapes differ: {reference.shape} vs {candidate.shape}")
    difference = np.abs(reference - candidate)
    finite = bool(np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate)))
    return {
        "max_abs_diff": float(np.max(difference)) if difference.size else 0.0,
        "max_rel_diff": float(np.max(difference / np.maximum(1.0, np.abs(reference)))) if difference.size else 0.0,
        "max_abs_action": float(np.max(np.abs(reference))) if reference.size else 0.0,
        "finite": finite,
    }


def _deterministic_actions(learner, observations: np.ndarray, training, checkpoint: Path) -> np.ndarray:
    import jax.numpy as jnp

    from flash_chord.training.ppo.evaluation import EvaluationConfig, policy_action

    evaluation = EvaluationConfig(checkpoint=str(checkpoint), deterministic=True)
    return np.asarray(policy_action(learner, jnp.asarray(observations), training, evaluation))


_STOCK_VERIFY_SCRIPT = r'''
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

checkpoint, probe, output, runtime = sys.argv[1:5]
import flash_chord
module = Path(flash_chord.__file__).resolve()
expected = (Path(runtime).resolve() / "src" / "flash_chord").resolve()
if module.parent != expected:
    raise SystemExit(f"stock verification imported {module}; expected the package in {expected}")

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.evaluation.harness import (
    _evaluation_curriculum_stage, apply_evaluation_composition, checkpoint_training_algorithm,
    evaluation_code_sha256, evaluation_thresholds, reference_tracking_criteria,
)
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.training.checkpoint import load_checkpoint_for_inference, read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.ppo.config import TrainingConfig
from flash_chord.training.ppo.evaluation import EvaluationConfig, policy_action
from flash_chord.training.ppo.learner import create_learner

metadata = read_checkpoint_metadata(checkpoint)
config = checkpoint_config(metadata)
algorithm = checkpoint_training_algorithm(config)
if algorithm != "ppo":
    raise SystemExit(f"stock runtime classifies the export as {algorithm!r}")
evaluation_thresholds(config)
reference_tracking_criteria(config)
apply_evaluation_composition(config, world_count=1, reset_mode="explicit")
cfg = OmegaConf.create(config)
env_config = instantiate_typed(cfg.env, RLEnvConfig)
embodiment = instantiate_typed(cfg.embodiment, Embodiment)
collision = instantiate_typed(cfg.collision, CollisionPolicy)
training = replace(instantiate_typed(cfg.training, TrainingConfig), world_count=1)
stage = None
if training.curriculum is not None:
    stage_index = training.curriculum.stage_index(max(int(metadata["iteration"]) - 1, 0))
    stage = _evaluation_curriculum_stage(training.curriculum.stages[stage_index], "explicit")

schema = json.loads(metadata["policy_schema_json"])
observation_dim = int(schema["observation"]["dimension"])
action_dim = int(schema["action"]["dimension"])


class _Env:
    world_count = 1

    def __init__(self):
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def reset(self):
        return jnp.zeros((1, observation_dim), dtype=jnp.float32)


learner, _ = load_checkpoint_for_inference(checkpoint, create_learner(_Env(), training))
observations = np.load(probe)["observations"]
evaluation = EvaluationConfig(checkpoint=checkpoint, deterministic=True)
actions = np.asarray(policy_action(learner, jnp.asarray(observations), training, evaluation))
np.savez(
    output,
    actions=actions,
    log_std=np.asarray(learner.train_state.params["params"]["log_std"]),
    leaf_count=np.asarray(len(jax.tree.leaves(learner))),
)
print(json.dumps({
    "flash_chord_module": str(module),
    "algorithm": algorithm,
    "env_reset_type": type(env_config.reset).__name__,
    "embodiment_type": type(embodiment).__name__,
    "collision_type": type(collision).__name__,
    "curriculum_stage_voc_scale": None if stage is None else stage.voc_scale,
    "eval_code_sha256": evaluation_code_sha256(),
    "jax_backend": jax.default_backend(),
}))
'''


def verify_in_stock_runtime(
    checkpoint: str | Path,
    observations: np.ndarray,
    stock_runtime: str | Path,
    *,
    python: str | None = None,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Instantiate and run an exported checkpoint with the STOCK package in a subprocess (spec 6.3.6).

    The subprocess imports only ``$STOCK/src``, instantiates ``env``, ``embodiment``, ``collision`` and
    ``training`` exactly as ``harness.evaluate_checkpoint`` does, restores the checkpoint with stock
    ``create_learner`` + ``load_checkpoint_for_inference``, and returns stock deterministic actions
    (float32 on the CPU) for ``observations``.
    """
    stock_runtime = Path(stock_runtime).resolve()
    if not (stock_runtime / "src" / "flash_chord" / "__init__.py").is_file():
        raise FileNotFoundError(f"{stock_runtime} is not a FlashCHORD runtime checkout (no src/flash_chord)")
    with tempfile.TemporaryDirectory(prefix="fc_e2_verify_") as scratch:
        scratch_path = Path(scratch)
        script = scratch_path / "verify_stock.py"
        script.write_text(_STOCK_VERIFY_SCRIPT)
        probe = scratch_path / "probe.npz"
        np.savez(probe, observations=np.asarray(observations, dtype=np.float32))
        output = scratch_path / "actions.npz"
        environment = {
            key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
        }
        environment.update(
            {
                "PYTHONPATH": str(stock_runtime / "src"),
                "JAX_PLATFORMS": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONDONTWRITEBYTECODE": "1",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        )
        completed = subprocess.run(
            [python or sys.executable, str(script), str(Path(checkpoint).resolve()), str(probe), str(output), str(stock_runtime)],
            cwd=stock_runtime,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "stock-runtime verification failed (exit "
                f"{completed.returncode}):\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
            )
        summary_line = [line for line in completed.stdout.splitlines() if line.startswith("{")]
        summary = json.loads(summary_line[-1]) if summary_line else {}
        with np.load(output) as archive:
            summary["actions"] = np.asarray(archive["actions"])
            summary["log_std"] = np.asarray(archive["log_std"])
            summary["leaf_count"] = int(archive["leaf_count"])
    return summary


# --------------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------------


def _default_report_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.stem}.export_report.json")


def export_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    stock_runtime: str | Path | None,
    probe_observations: str | Path | np.ndarray | None = None,
    seed: int = 0,
    probe_count: int = DEFAULT_PROBE_COUNT,
    tolerance: float = DEFAULT_TOLERANCE,
    report_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export ``source`` to a stock-format checkpoint at ``destination`` and return the report.

    Coordinated sources (``training.coordinate`` set) get the E2 embedding; stock-learner sources
    (C1) keep every policy leaf and only get the config rewrite. Either way the file is written only
    after its deterministic actions match the source's to ``tolerance`` (scaled by ``max(1, |a|)``)
    on the probe batch, re-read from disk with the stock template, and, when ``stock_runtime`` is
    given, also recomputed by the STOCK package in a subprocess.
    """
    import jax
    import jax.numpy as jnp

    from flash_chord.training.checkpoint import (
        load_checkpoint_for_inference,
        read_checkpoint_metadata,
        save_checkpoint,
    )
    from flash_chord.training.checkpoint_metadata import (
        CONFIG_METADATA_KEY as _CONFIG_KEY,
    )
    from flash_chord.training.checkpoint_metadata import (
        POLICY_SCHEMA_METADATA_KEY as _SCHEMA_KEY,
    )
    from flash_chord.training.checkpoint_metadata import checkpoint_config, checkpoint_policy_schema

    if (_CONFIG_KEY, _SCHEMA_KEY) != (CONFIG_METADATA_KEY, POLICY_SCHEMA_METADATA_KEY):
        raise RuntimeError("checkpoint metadata keys changed; update the E2 exporter")
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination == source:
        raise ValueError("export destination must differ from the source checkpoint")
    report_path = Path(report_path) if report_path is not None else _default_report_path(destination)

    metadata = read_checkpoint_metadata(source)
    saved = checkpoint_config(metadata)
    schema = checkpoint_policy_schema(metadata)
    observation_dim = int(schema["observation"]["dimension"])
    action_dim = int(schema["action"]["dimension"])
    exported_config, notes = stock_config_with_notes(saved)
    stock_training = _stock_training_config(exported_config["training"])
    coordinate_mapping = saved["training"].get("coordinate")

    jax_device = jax.devices(device)[0]
    report: dict[str, Any] = {
        "export_format": EXPORT_FORMAT,
        "source": str(source),
        "destination": str(destination),
        "source_sha256": file_sha256(source),
        "source_kind": "coordinated" if coordinate_mapping is not None else "stock",
        "iteration": metadata.get("iteration"),
        "config_rewrite": notes,
        "device": device,
        "tolerance": tolerance,
    }
    with jax.default_device(jax_device):
        if coordinate_mapping is not None:
            api = _coordinate_api()
            reference_training = api.training_config(saved["training"])
            coordinate = reference_training.coordinate
            layout = api.build_layout(schema, coordinate)
            digest = api.layout_digest(layout)
            stored_digest = metadata.get(LAYOUT_METADATA_KEY)
            if stored_digest is None:
                raise ValueError(f"coordinated checkpoint {source} lacks {LAYOUT_METADATA_KEY!r} metadata")
            if stored_digest != digest:
                raise ValueError(
                    f"coordinate layout digest mismatch: checkpoint {stored_digest}, rebuilt {digest}; "
                    "the layout code or schema changed since training"
                )
            template = api.create_learner_template(reference_training, observation_dim, action_dim, layout, world_count=1)
            reference_learner, _ = load_checkpoint_for_inference(source, template)
            plan = embedding_plan(coordinate, stock_training.network.actor_hidden_dims)
            actor = embed_actor(reference_learner.train_state.params, layout, coordinate, stock_training.network.actor_hidden_dims)
            exported_learner = _stock_learner_from_actor(
                actor, reference_learner, stock_training, observation_dim, action_dim, seed
            )
            report["coordinate_layout_sha256"] = digest
            report["budget"] = plan.budget()
        else:
            template = stock_learner_template(stock_training, observation_dim, action_dim)
            reference_learner, _ = load_checkpoint_for_inference(source, template)
            reference_training = stock_training
            exported_learner = reference_learner.replace(
                observation=jnp.zeros((1, observation_dim), dtype=jnp.float32)
            )
            report["budget"] = None

        exported_metadata = {key: value for key, value in metadata.items()}
        exported_metadata[CONFIG_METADATA_KEY] = json.dumps(exported_config, sort_keys=True, separators=(",", ":"))
        exported_metadata[POLICY_SCHEMA_METADATA_KEY] = metadata[POLICY_SCHEMA_METADATA_KEY]
        exported_metadata[EXPORT_FORMAT_METADATA_KEY] = EXPORT_FORMAT
        exported_metadata[EXPORT_SOURCE_METADATA_KEY] = report["source_sha256"]
        if "coordinate_layout_sha256" in report:
            exported_metadata[LAYOUT_METADATA_KEY] = report["coordinate_layout_sha256"]

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.export_staging")
        try:
            save_checkpoint(staging, exported_learner, metadata=exported_metadata)

            # Re-read the staged file exactly as an evaluator would: stock template from the exported config.
            staged_metadata = read_checkpoint_metadata(staging)
            staged_config = checkpoint_config(staged_metadata)
            staged_training = _stock_training_config(staged_config["training"])
            restored, _ = load_checkpoint_for_inference(
                staging, stock_learner_template(staged_training, observation_dim, action_dim)
            )
            if checkpoint_policy_schema(staged_metadata) != schema:
                raise RuntimeError("exported policy schema differs from the source schema")

            probes = make_probe_observations(
                reference_learner.normalization,
                probe_count,
                seed,
                reference_training.observation_normalization_epsilon,
                normalized=reference_training.observation_normalization,
            )
            if probe_observations is not None:
                extra = (
                    load_probe(probe_observations, observation_dim)
                    if not isinstance(probe_observations, np.ndarray)
                    else np.asarray(probe_observations, dtype=np.float32).reshape(-1, observation_dim)
                )
                probes = np.concatenate([probes, extra], axis=0)
            reference_actions = _deterministic_actions(reference_learner, probes, reference_training, source)
            exported_actions = _deterministic_actions(restored, probes, staged_training, staging)
            comparison = compare_actions(reference_actions, exported_actions)
            reference_log_std = np.asarray(reference_learner.train_state.params["params"]["log_std"])
            exported_log_std = np.asarray(restored.train_state.params["params"]["log_std"])
            report.update(
                {
                    "probe_count": int(probes.shape[0]),
                    "probe_seed": seed,
                    "dev_restore": comparison,
                    "log_std_equal": bool(np.array_equal(reference_log_std, exported_log_std)),
                    "leaf_count": len(jax.tree.leaves(restored)),
                }
            )
            failures = []
            if not comparison["finite"] or comparison["max_rel_diff"] > tolerance:
                failures.append(f"DEV restore: max_rel_diff {comparison['max_rel_diff']:.3e} > {tolerance:.1e}")
            if not report["log_std_equal"]:
                failures.append("log_std differs")

            if stock_runtime is not None:
                stock = verify_in_stock_runtime(staging, probes, stock_runtime)
                stock_comparison = compare_actions(reference_actions, stock.pop("actions"))
                stock_log_std = stock.pop("log_std")
                stock["comparison"] = stock_comparison
                stock["log_std_equal"] = bool(np.array_equal(reference_log_std, stock_log_std))
                stock["verified"] = (
                    stock_comparison["finite"]
                    and stock_comparison["max_rel_diff"] <= tolerance
                    and stock["log_std_equal"]
                    and stock.get("algorithm") == "ppo"
                )
                report["stock_runtime"] = {"path": str(Path(stock_runtime).resolve()), **stock}
                if not stock["verified"]:
                    failures.append(f"stock runtime: {stock_comparison} log_std_equal={stock['log_std_equal']}")
            else:
                report["stock_runtime"] = None

            report["max_abs_diff"] = max(
                comparison["max_abs_diff"],
                (report["stock_runtime"] or {}).get("comparison", {}).get("max_abs_diff", 0.0),
            )
            report["max_rel_diff"] = max(
                comparison["max_rel_diff"],
                (report["stock_runtime"] or {}).get("comparison", {}).get("max_rel_diff", 0.0),
            )
            if failures:
                report["status"] = "refused"
                report["failures"] = failures
                _write_report(report_path, report)
                raise ExportVerificationError("E2 export verification failed: " + "; ".join(failures), report)
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)

    report["status"] = "ok"
    report["destination_sha256"] = file_sha256(destination)
    report["report_path"] = str(report_path)
    _write_report(report_path, report)
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="DEV checkpoint (coordinated or stock learner)")
    parser.add_argument("--output", required=True, type=Path, help="stock-format checkpoint to write")
    parser.add_argument("--stock-runtime", type=Path, help="official FlashCHORD runtime root ($STOCK) for the check")
    parser.add_argument("--probe", type=Path, help="extra raw observations (.npy, or .npz with 'observations')")
    parser.add_argument("--probe-count", type=int, default=DEFAULT_PROBE_COUNT)
    parser.add_argument("--seed", type=int, default=0, help="seed for the probe batch and the fresh stock critic")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--report", type=Path, help="report JSON path (default: <output stem>.export_report.json)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "gpu", "cuda"),
                        help="JAX device for verification; cpu gives exact float32 matmuls (default)")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax

        jax.config.update("jax_platforms", "cpu")
    try:
        report = export_checkpoint(
            args.input,
            args.output,
            stock_runtime=args.stock_runtime,
            probe_observations=args.probe,
            seed=args.seed,
            probe_count=args.probe_count,
            tolerance=args.tolerance,
            report_path=args.report,
            device="cpu" if args.device == "cpu" else "gpu",
        )
    except ExportVerificationError as error:
        print(json.dumps(error.report, indent=2, sort_keys=True, default=_json_default), file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
