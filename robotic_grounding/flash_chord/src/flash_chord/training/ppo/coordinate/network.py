# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W0-P coordinated actor with a K-column central critic (Flax).

Actor, per agent ``i`` in (left, right, body), on the already normalized 486-d observation::

    e_i  = ELU(Enc_g(x_i))                          g = hand for left/right (shared weights), body for body
    p_j  = ELU(Phi_g(d_j))                          partner descriptors; right/left hand use the mirror
    m_L  = ELU(Rho_h(p_R + p_B))
    m_R  = ELU(Rho_h(p_L + p_B^M))                  p_B^M = Phi_b(S_body d_body) when mirror_body_message
    m_B  = ELU(Rho_b(p_L + p_R))
    mu_i = Head_g(ELU(T1_g(ELU(T0_g([e_i, m_i])))))

The CH descriptor "active" flag and message count are constants (every agent active, two partners), so
they are folded into the biases. ``partner_message=False`` zeroes every message, as CH does. The means
are scattered into the env action order with the per-agent mirror signs. The critic is a central MLP on
the full observation with one value column per agent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from flash_chord.training.ppo.config import CoordinateConfig, NetworkConfig
from flash_chord.training.ppo.coordinate.layout import CoordinateLayout, InputView


def gather_view(observation, view: InputView):
    """Signed gather of one view from the last axis of ``observation``."""
    indices = np.asarray(view.indices, dtype=np.int32)
    signs = np.asarray(view.signs, dtype=np.float32)
    gathered = jnp.take(observation, indices, axis=-1)
    if np.all(signs == 1.0):
        return gathered
    return gathered * signs


def scatter_order(layout: CoordinateLayout) -> tuple[np.ndarray, np.ndarray]:
    """Gather indices/signs that map concatenated canonical agent outputs to the env action order."""
    concatenated = [index for indices in layout.action_indices for index in indices]
    signs = np.asarray([sign for agent in layout.action_signs for sign in agent], dtype=np.float32)
    inverse = np.empty(len(concatenated), dtype=np.int32)
    inverse[np.asarray(concatenated)] = np.arange(len(concatenated), dtype=np.int32)
    return inverse, signs


def _kernel_init():
    return nn.initializers.orthogonal(math.sqrt(2.0))


class CoordinatedActorCritic(nn.Module):
    """Coordinated actor plus central critic; ``__call__`` returns ``(policy, value[B, K])``."""

    action_dim: int
    layout: CoordinateLayout
    coordinate: CoordinateConfig
    network: NetworkConfig

    def setup(self) -> None:
        coordinate = self.coordinate
        if self.layout.action_dim != self.action_dim:
            raise ValueError(f"layout action_dim {self.layout.action_dim} differs from {self.action_dim}")
        head_init = nn.initializers.zeros
        if self.network.actor_output_gain > 0.0:
            head_init = nn.initializers.orthogonal(self.network.actor_output_gain)
        hand_dims = {view.dim for view in self.layout.actor_inputs[:2]}
        if len(hand_dims) != 1:
            raise ValueError(f"left/right hand views must have equal widths, got {hand_dims}")
        trunk_0, trunk_1 = coordinate.trunk_dims
        hand_actions = len(self.layout.action_indices[0])
        if len(self.layout.action_indices[1]) != hand_actions:
            raise ValueError("left/right hand agents must have equal action counts")
        body_actions = len(self.layout.action_indices[2])

        self.hand_encoder = nn.Dense(coordinate.encoder_dim, kernel_init=_kernel_init())
        self.body_encoder = nn.Dense(coordinate.encoder_dim, kernel_init=_kernel_init())
        self.hand_partner = nn.Dense(coordinate.partner_dim, kernel_init=_kernel_init())
        self.body_partner = nn.Dense(coordinate.partner_dim, kernel_init=_kernel_init())
        self.hand_message = nn.Dense(coordinate.message_dim, kernel_init=_kernel_init())
        self.body_message = nn.Dense(coordinate.message_dim, kernel_init=_kernel_init())
        self.hand_trunk_0 = nn.Dense(trunk_0, kernel_init=_kernel_init())
        self.hand_trunk_1 = nn.Dense(trunk_1, kernel_init=_kernel_init())
        self.body_trunk_0 = nn.Dense(trunk_0, kernel_init=_kernel_init())
        self.body_trunk_1 = nn.Dense(trunk_1, kernel_init=_kernel_init())
        self.hand_head = nn.Dense(hand_actions, kernel_init=head_init)
        self.body_head = nn.Dense(body_actions, kernel_init=head_init)
        self.log_std = self.param(
            "log_std",
            nn.initializers.constant(math.log(self.network.initial_noise_std)),
            (self.action_dim,),
        )
        self.critic = [nn.Dense(width, kernel_init=_kernel_init()) for width in self.network.critic_hidden_dims]
        self.value = nn.Dense(self.layout.num_agents, kernel_init=nn.initializers.orthogonal(1.0))

    def agent_means(self, observation):
        """Canonical (un-mirrored) per-agent means ``(mu_left, mu_right, mu_body)``."""
        layout = self.layout
        x_left, x_right, x_body = (gather_view(observation, view) for view in layout.actor_inputs)
        e_left = nn.elu(self.hand_encoder(x_left))
        e_right = nn.elu(self.hand_encoder(x_right))
        e_body = nn.elu(self.body_encoder(x_body))

        if self.coordinate.partner_message:
            d_left, d_right = (gather_view(observation, view) for view in layout.hand_descriptors)
            p_left = nn.elu(self.hand_partner(d_left))
            p_right = nn.elu(self.hand_partner(d_right))
            p_body = nn.elu(self.body_partner(gather_view(observation, layout.body_descriptor)))
            if self.coordinate.mirror_body_message:
                p_body_right = nn.elu(self.body_partner(gather_view(observation, layout.body_descriptor_mirrored)))
            else:
                p_body_right = p_body
            m_left = nn.elu(self.hand_message(p_right + p_body))
            m_right = nn.elu(self.hand_message(p_left + p_body_right))
            m_body = nn.elu(self.body_message(p_left + p_right))
        else:
            # CH --coord_no_partner_message: the message is exactly zero. The message layers still own
            # parameters so the checkpoint layout does not depend on this switch.
            zeros = jnp.zeros(observation.shape[:-1] + (self.coordinate.partner_dim,), dtype=observation.dtype)
            m_left = jnp.zeros_like(self.hand_message(zeros))
            m_right = m_left
            m_body = jnp.zeros_like(self.body_message(zeros))
            self.hand_partner(jnp.zeros(observation.shape[:-1] + (layout.hand_descriptors[0].dim,), observation.dtype))
            self.body_partner(jnp.zeros(observation.shape[:-1] + (layout.body_descriptor.dim,), observation.dtype))

        def hand(latent, message):
            hidden = nn.elu(self.hand_trunk_0(jnp.concatenate((latent, message), axis=-1)))
            return self.hand_head(nn.elu(self.hand_trunk_1(hidden)))

        def body(latent, message):
            hidden = nn.elu(self.body_trunk_0(jnp.concatenate((latent, message), axis=-1)))
            return self.body_head(nn.elu(self.body_trunk_1(hidden)))

        return hand(e_left, m_left), hand(e_right, m_right), body(e_body, m_body)

    def actor_mean(self, observation):
        """Deterministic action mean in the env action order, shape ``(..., action_dim)``."""
        inverse, signs = scatter_order(self.layout)
        canonical = jnp.concatenate(self.agent_means(observation), axis=-1) * signs
        return jnp.take(canonical, inverse, axis=-1)

    def critic_value(self, observation):
        critic = observation
        for layer in self.critic:
            critic = nn.elu(layer(critic))
        return self.value(critic)

    def __call__(self, observation):
        mean = self.actor_mean(observation)
        policy = distrax.MultivariateNormalDiag(mean, jnp.exp(self.log_std))
        return policy, self.critic_value(observation)


def _parameter_tree(params: Mapping) -> Mapping:
    if set(params) == {"params"}:
        return params["params"]
    return params


def actor_mean(params: Mapping, layout: CoordinateLayout, coordinate: CoordinateConfig, observation):
    """Coordinated deterministic action means from actor parameters only (export and tests).

    ``params`` is the coordinated parameter tree (with or without the outer ``"params"`` key);
    critic parameters may be absent. ``observation`` must already be normalized.
    """
    model = CoordinatedActorCritic(
        action_dim=layout.action_dim,
        layout=layout,
        coordinate=coordinate,
        network=NetworkConfig(),
    )
    return model.apply({"params": _parameter_tree(params)}, observation, method=CoordinatedActorCritic.actor_mean)
