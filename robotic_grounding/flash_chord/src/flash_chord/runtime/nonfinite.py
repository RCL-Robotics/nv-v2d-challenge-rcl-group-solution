# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-world non-finite simulator-state guard (training only; off unless ``nonfinite_guard`` is set).

A single world whose MuJoCo-Warp solve fails (float32 Newton solver on a degenerate contact) ends the
substep with NaN ``qacc`` and therefore NaN ``joint_q``/``joint_qd``/``body_q``/``body_qd``. Stock
FlashCHORD has no finite check anywhere in the step: the NaN world is not terminated (NaN comparisons
are false), its observation poisons the running observation normalizer, and MuJoCo-Warp copies the
NaN ``qacc`` into ``qacc_warmstart`` (forward.py) and starts the next solve from it unconditionally
(solver.py ``_solve``), so the world re-NaNs after every reset.

:class:`NonfiniteWorldGuard` runs inside the captured step graph, after physics and termination:

* ``flag[w] = 1`` when any ``joint_q``/``joint_qd``/``body_q``/``body_qd`` entry of world ``w`` or its
  MuJoCo-Warp ``qacc_warmstart`` row is non-finite, else ``0``;
* flagged worlds get ``terminated = 1`` (so the existing auto-reset restores them) and a zeroed warm
  start; after the objective, their ``score`` is zeroed so returns stay finite.

Every write is conditional on the flag, so when every value is finite the guard only writes zeros to its
own ``flag`` buffer and the simulation, rewards and resets are bit-identical to running without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@wp.kernel
def flag_nonfinite_worlds(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    q_per_world: int,
    dof_per_world: int,
    bodies_per_world: int,
    warmstart: wp.array2d(dtype=wp.float32),
    warmstart_dim: int,
    flag: wp.array(dtype=wp.int32),
    terminated: wp.array(dtype=wp.int32),
    total: wp.array(dtype=wp.int32),
) -> None:
    """Flag worlds with any non-finite state; terminate them and zero their solver warm start."""
    world = wp.tid()
    bad = int(0)
    q_base = world * q_per_world
    for index in range(q_per_world):
        if not wp.isfinite(joint_q[q_base + index]):
            bad = 1
    dof_base = world * dof_per_world
    for index in range(dof_per_world):
        if not wp.isfinite(joint_qd[dof_base + index]):
            bad = 1
    body_base = world * bodies_per_world
    for body in range(bodies_per_world):
        transform = body_q[body_base + body]
        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        velocity = body_qd[body_base + body]
        for axis in range(3):
            if not wp.isfinite(position[axis]):
                bad = 1
        for axis in range(4):
            if not wp.isfinite(rotation[axis]):
                bad = 1
        for axis in range(6):
            if not wp.isfinite(velocity[axis]):
                bad = 1
    for index in range(warmstart_dim):
        if not wp.isfinite(warmstart[world, index]):
            bad = 1
    flag[world] = bad
    if bad != 0:
        terminated[world] = 1
        for index in range(warmstart_dim):
            warmstart[world, index] = 0.0
        wp.atomic_add(total, 0, 1)


@wp.kernel
def zero_flagged_scores(
    flag: wp.array(dtype=wp.int32),
    score: wp.array(dtype=wp.float32),
) -> None:
    """Replace the (non-finite) objective score of flagged worlds with zero."""
    world = wp.tid()
    if flag[world] != 0:
        score[world] = 0.0


@dataclass
class NonfiniteWorldGuard:
    """Device buffers and launches for the per-world non-finite guard."""

    world_count: int
    q_per_world: int
    dof_per_world: int
    bodies_per_world: int
    flag: wp.array
    total: wp.array
    warmstart: wp.array
    warmstart_dim: int

    @classmethod
    def build(cls, model, solver, world_count: int, device=None) -> "NonfiniteWorldGuard":
        """Resolve the uniform per-world layout and the optional MuJoCo-Warp warm-start buffer."""
        if world_count <= 0:
            raise ValueError(f"world_count must be positive, got {world_count}")
        counts = {
            "joint_coord_count": model.joint_coord_count,
            "joint_dof_count": model.joint_dof_count,
            "body_count": model.body_count,
        }
        for name, count in counts.items():
            if count % world_count != 0:
                raise ValueError(f"nonfinite guard needs a uniform per-world layout; {name}={count} is not "
                                 f"divisible by world_count={world_count}")
        bodies_per_world = model.body_count // world_count
        body_world = getattr(model, "body_world", None)
        if body_world is not None and model.body_count:
            expected = np.repeat(np.arange(world_count, dtype=np.int64), bodies_per_world)
            if not np.array_equal(np.asarray(body_world.numpy(), dtype=np.int64), expected):
                raise ValueError("nonfinite guard needs bodies stored contiguously per world")
        warmstart = None
        data = getattr(solver, "mjw_data", None)
        if data is not None:
            warmstart = getattr(data, "qacc_warmstart", None)
        if warmstart is not None and (warmstart.ndim != 2 or warmstart.shape[0] != world_count):
            raise ValueError(f"unexpected MuJoCo-Warp qacc_warmstart shape {warmstart.shape}")
        warmstart_dim = 0 if warmstart is None else int(warmstart.shape[1])
        if warmstart is None:
            warmstart = wp.zeros((1, 1), dtype=wp.float32, device=device)
        return cls(
            world_count=world_count,
            q_per_world=model.joint_coord_count // world_count,
            dof_per_world=model.joint_dof_count // world_count,
            bodies_per_world=bodies_per_world,
            flag=wp.zeros(world_count, dtype=wp.int32, device=device),
            total=wp.zeros(1, dtype=wp.int32, device=device),
            warmstart=warmstart,
            warmstart_dim=warmstart_dim,
        )

    def check(self, state, terminated: wp.array) -> None:
        """Flag non-finite worlds of ``state``; terminate them and zero their warm start (graph-capturable)."""
        wp.launch(
            flag_nonfinite_worlds,
            dim=self.world_count,
            inputs=[
                state.joint_q,
                state.joint_qd,
                state.body_q,
                state.body_qd,
                self.q_per_world,
                self.dof_per_world,
                self.bodies_per_world,
                self.warmstart,
                self.warmstart_dim,
            ],
            outputs=[self.flag, terminated, self.total],
        )

    def zero_scores(self, score: wp.array) -> None:
        """Zero the objective score of the worlds flagged by the last :meth:`check`."""
        wp.launch(zero_flagged_scores, dim=self.world_count, inputs=[self.flag], outputs=[score])
