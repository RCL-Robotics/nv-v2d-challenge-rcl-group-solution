# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed RL training defaults matched to the reference W&B run."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from flash_chord.lifecycle.curriculum import FixedCurriculum, reference_curriculum


@dataclass(frozen=True)
class NetworkConfig:
    """Separate actor and critic network configuration."""

    actor_hidden_dims: tuple[int, ...] = (1024, 512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (1024, 512, 256, 128)
    activation: str = "elu"
    initial_noise_std: float = 0.1
    noise_std_type: str = "scalar"
    actor_output_gain: float = 0.0

    def __post_init__(self) -> None:
        if not self.actor_hidden_dims or min(self.actor_hidden_dims) <= 0:
            raise ValueError(f"actor_hidden_dims must be positive, got {self.actor_hidden_dims}")
        if not self.critic_hidden_dims or min(self.critic_hidden_dims) <= 0:
            raise ValueError(f"critic_hidden_dims must be positive, got {self.critic_hidden_dims}")
        if self.activation != "elu":
            raise ValueError(f"unsupported activation {self.activation!r}; expected 'elu'")
        if self.initial_noise_std <= 0.0:
            raise ValueError(f"initial_noise_std must be positive, got {self.initial_noise_std}")
        if self.noise_std_type != "scalar":
            raise ValueError(f"unsupported noise_std_type {self.noise_std_type!r}; expected 'scalar'")
        if self.actor_output_gain < 0.0:
            raise ValueError(f"actor_output_gain must be non-negative, got {self.actor_output_gain}")


@dataclass(frozen=True)
class PPOConfig:
    """PPO optimization settings."""

    value_loss_coefficient: float = 1.0
    use_clipped_value_loss: bool = True
    clip_parameter: float = 0.1
    entropy_coefficient: float = 0.001
    learning_epochs: int = 5
    mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    minimum_learning_rate: float = 1.0e-5
    maximum_learning_rate: float = 1.0e-2
    adaptive_schedule_factor: float = 1.5
    discount: float = 0.99
    gae_lambda: float = 0.95
    desired_kl: float = 0.005
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    log_ratio_clip: float | None = None

    def __post_init__(self) -> None:
        if self.value_loss_coefficient < 0.0:
            raise ValueError(f"value_loss_coefficient must be non-negative, got {self.value_loss_coefficient}")
        if self.clip_parameter <= 0.0:
            raise ValueError(f"clip_parameter must be positive, got {self.clip_parameter}")
        if self.entropy_coefficient < 0.0:
            raise ValueError(f"entropy_coefficient must be non-negative, got {self.entropy_coefficient}")
        if self.learning_epochs <= 0:
            raise ValueError(f"learning_epochs must be positive, got {self.learning_epochs}")
        if self.mini_batches <= 0:
            raise ValueError(f"mini_batches must be positive, got {self.mini_batches}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.schedule not in {"adaptive", "fixed"}:
            raise ValueError(f"schedule must be 'adaptive' or 'fixed', got {self.schedule!r}")
        if not 0.0 < self.minimum_learning_rate <= self.learning_rate <= self.maximum_learning_rate:
            raise ValueError(
                "learning rates must satisfy 0 < minimum <= initial <= maximum; got "
                f"{self.minimum_learning_rate}, {self.learning_rate}, {self.maximum_learning_rate}"
            )
        if self.adaptive_schedule_factor <= 1.0:
            raise ValueError(f"adaptive_schedule_factor must be greater than 1, got {self.adaptive_schedule_factor}")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {self.discount}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.desired_kl <= 0.0:
            raise ValueError(f"desired_kl must be positive, got {self.desired_kl}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.log_ratio_clip is not None and (not math.isfinite(self.log_ratio_clip) or self.log_ratio_clip <= 0.0):
            raise ValueError(f"log_ratio_clip must be finite and positive or None, got {self.log_ratio_clip}")


COORDINATE_AGENTS = ("left", "right", "body")
COORDINATE_HAND_WORLD_TERMS = ("reflect", "drop")


def coordinate_export_budget(
    encoder_dim: int,
    partner_dim: int,
    message_dim: int,
    trunk_dims: tuple[int, ...],
    mirror_body_message: bool,
) -> tuple[int, int, int, int]:
    """Hidden units the E2 export needs in each of the four stock actor layers (three agents)."""
    partner_blocks = 4 if mirror_body_message else 3
    return (
        3 * encoder_dim + partner_blocks * partner_dim,
        3 * (encoder_dim + message_dim),
        3 * trunk_dims[0],
        3 * trunk_dims[1],
    )


@dataclass(frozen=True)
class CoordinateConfig:
    """W0-P coordinated actor and per-agent PPO settings (three agents: left hand, right hand, body).

    ``None`` in ``TrainingConfig.coordinate`` selects the stock learner. The coordinated critic uses
    ``network.critic_hidden_dims``; the coordinated actor uses the widths below and ignores
    ``network.actor_hidden_dims``.
    """

    agents: tuple[str, ...] = COORDINATE_AGENTS
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

    def __post_init__(self) -> None:
        for name in ("agents", "trunk_dims", "export_actor_hidden_dims"):
            value = getattr(self, name)
            if isinstance(value, str) or not hasattr(value, "__iter__"):
                raise TypeError(f"coordinate.{name} must be a sequence, got {value!r}")
            object.__setattr__(self, name, tuple(value))
        if self.agents != COORDINATE_AGENTS:
            raise ValueError(f"coordinate.agents must be exactly {COORDINATE_AGENTS}, got {self.agents}")
        for name in ("encoder_dim", "partner_dim", "message_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"coordinate.{name} must be a positive integer, got {value!r}")
        if len(self.trunk_dims) != 2 or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0 for width in self.trunk_dims
        ):
            raise ValueError(f"coordinate.trunk_dims must be two positive integers, got {self.trunk_dims}")
        for name in ("partner_message", "mirror_body_message"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"coordinate.{name} must be a bool, got {getattr(self, name)!r}")
        if self.hand_world_terms not in COORDINATE_HAND_WORLD_TERMS:
            raise ValueError(
                f"coordinate.hand_world_terms must be one of {COORDINATE_HAND_WORLD_TERMS}, "
                f"got {self.hand_world_terms!r}"
            )
        if self.ratio_mode != "per_agent":
            raise ValueError(f"coordinate.ratio_mode supports only 'per_agent' (W0-P), got {self.ratio_mode!r}")
        if self.value_mode != "per_agent":
            raise ValueError(f"coordinate.value_mode supports only 'per_agent' (W0-P), got {self.value_mode!r}")
        if len(self.export_actor_hidden_dims) != 4 or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in self.export_actor_hidden_dims
        ):
            raise ValueError(
                "coordinate.export_actor_hidden_dims must be four positive integers (the stock actor depth), "
                f"got {self.export_actor_hidden_dims}"
            )
        required = coordinate_export_budget(
            self.encoder_dim,
            self.partner_dim,
            self.message_dim,
            self.trunk_dims,
            self.mirror_body_message,
        )
        for layer, (needed, available) in enumerate(zip(required, self.export_actor_hidden_dims, strict=True)):
            if needed > available:
                raise ValueError(
                    f"coordinated actor needs {needed} units in stock actor layer {layer}, but "
                    f"export_actor_hidden_dims[{layer}] is {available}; widths {required} must fit "
                    f"{self.export_actor_hidden_dims}"
                )


@dataclass(frozen=True)
class TrainingConfig:
    """Runner, normalization, checkpoint, and logging settings."""

    seed: int = 42
    world_count: int = 4096
    rollout_steps: int = 24
    max_iterations: int = 20_000
    save_interval: int = 200
    observation_normalization: bool = True
    observation_normalization_epsilon: float = 1.0e-2
    reward_normalization: bool = False
    device: str = "cuda:0"
    resume: bool = False
    checkpoint: str | None = None
    wandb_entity: str = "nvidia-isaac"
    wandb_project: str = "v2d-flash-chord"
    experiment_name: str = "sharpa"
    run_name: str = "ppo"
    curriculum: FixedCurriculum | None = field(default_factory=reference_curriculum)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    coordinate: CoordinateConfig | None = None
    # Reset worlds whose simulator state turns non-finite and keep non-finite rows out of the rollout batch and
    # the observation normalizer (runtime/nonfinite.py, rollout.py). A pure no-op while every value is finite.
    nonfinite_guard: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.nonfinite_guard, bool):
            raise TypeError(f"nonfinite_guard must be a bool, got {self.nonfinite_guard!r}")
        if self.world_count <= 0:
            raise ValueError(f"world_count must be positive, got {self.world_count}")
        if self.rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {self.rollout_steps}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {self.save_interval}")
        if self.observation_normalization_epsilon <= 0.0:
            raise ValueError(
                f"observation_normalization_epsilon must be positive, got {self.observation_normalization_epsilon}"
            )
        if self.world_count * self.rollout_steps % self.ppo.mini_batches != 0:
            raise ValueError(
                f"rollout batch {self.world_count * self.rollout_steps} must be divisible by "
                f"{self.ppo.mini_batches} mini-batches"
            )
        if self.resume and self.checkpoint is None:
            raise ValueError("resume requires a checkpoint path")
        if self.coordinate is not None and not isinstance(self.coordinate, CoordinateConfig):
            raise TypeError(
                f"coordinate must be a CoordinateConfig or None, got {type(self.coordinate).__name__}"
            )
