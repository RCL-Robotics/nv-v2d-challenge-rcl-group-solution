# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W0-P coordinated learner: agent layout and mirror, actor-critic network, per-agent PPO math.

The E2 export lives in :mod:`flash_chord.training.ppo.coordinate.export` and is deliberately not
imported here.
"""

from flash_chord.training.ppo.coordinate.layout import (
    BODY_CANONICAL,
    BODY_MIRROR_PERM,
    BODY_MIRROR_SIGNS,
    HAND_CANONICAL,
    HAND_MIRROR_SIGNS,
    ROT6D_SIGN,
    VEC_SIGN,
    CoordinateLayout,
    InputView,
    build_layout,
    derive_mirror_signs,
    layout_digest,
    mirror_observation,
)
from flash_chord.training.ppo.coordinate.network import CoordinatedActorCritic, actor_mean
from flash_chord.training.ppo.coordinate.per_group import (
    CoordinatedPPOMetrics,
    CoordinatedRolloutBatch,
    compute_gae_k,
    explained_variance_per_column,
    group_matrix,
    normalize_per_column,
    per_group_log_prob,
    update_coordinated,
)

__all__ = [
    "BODY_CANONICAL",
    "BODY_MIRROR_PERM",
    "BODY_MIRROR_SIGNS",
    "HAND_CANONICAL",
    "HAND_MIRROR_SIGNS",
    "ROT6D_SIGN",
    "VEC_SIGN",
    "CoordinateLayout",
    "CoordinatedActorCritic",
    "CoordinatedPPOMetrics",
    "CoordinatedRolloutBatch",
    "InputView",
    "actor_mean",
    "build_layout",
    "compute_gae_k",
    "derive_mirror_signs",
    "explained_variance_per_column",
    "group_matrix",
    "layout_digest",
    "mirror_observation",
    "normalize_per_column",
    "per_group_log_prob",
    "update_coordinated",
]
