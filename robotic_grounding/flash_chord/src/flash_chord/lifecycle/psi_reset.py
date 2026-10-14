# SPDX-License-Identifier: Apache-2.0
"""PSI ReconBody reset: InterMimic physical state initialization on the FlashCHORD reset protocol.

A drop-in replacement of the RSI v3 reset (``rsi_reset.RSIReconBodyReset``) that changes only how a world's start
state is drawn; everything else in the training recipe stays as configured. It follows InterMimic
(Xu et al., CVPR 2025; ``isaacgym/src/intermimic/env/tasks/intermimic.py`` ``_reset_hybrid_state_init``, ``cal_cdf``
and the buffer update after resets; config ``omomo_train.yaml``):

* Buffer: every reference frame ``f`` has ``psi_slots`` simulation slots next to the reference state. The reference
  has the fixed weight 1; a simulation slot has a score in ``[0, 1]`` (0 while empty).
* Start frame: with probability ``frame0_share`` frame 0 (InterMimic ``hybridInitProb``), otherwise a frame in
  ``[0, max_start_frame]`` with probability proportional to ``1 / (1 + sum of the frame's slot scores)``: frames that
  already hold good simulated states are drawn less.
* Start state: at the drawn frame, the reference or one of the simulation slots with probability proportional to
  their weights. A simulation slot restores the stored simulator state and starts immediately; the reference keeps
  the stock reset of ReconBody (frame 0 immediate; later frames with the 50-step freeze, shoulder spread and finger
  offset).
* Insertion: a world snapshots its simulator state every ``psi_stride`` steps once it has settled; the snapshot at
  step ``j`` and frame ``f`` scores ``survived / horizon`` with ``horizon = min(psi_horizon, last frame - f)`` and
  ``survived`` the steps the world lasted after ``j`` (InterMimic's ``(end - j) / (end_max - j)`` over its 300-frame
  rollout window, taken here over a sliding window because FlashCHORD episodes run to the clip end). Snapshots above
  ``psi_threshold`` replace the lowest-scoring simulation slot of their frame; all simulation scores decay by
  ``psi_score_decay`` per iteration (InterMimic: ``1e-5`` per reset batch).
* No VOC help: lanes never get help (InterMimic has no assist); the reference starts after frame 0 keep the stock
  reset's freeze-time hold, which releases to zero.

The device side snapshots and scores inside the captured step graph and queues the accepted snapshots; the host
inserts them into the buffer once per iteration (``host_update``), so the insertion is sequential and deterministic.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp

from flash_chord.lifecycle.rsi import kernels as rsi_kernels
from flash_chord.lifecycle.rsi_reset import _SIDECAR_NAME, RSIReconBodyReset, RSIReconBodyResetConfig

PSI_SIDECAR_FORMAT = "psi_state"
PSI_SIDECAR_VERSION = 1

# start_stats slots (same layout as RSI: total, frame 0, simulation slot, anchor)
PSI_START_TOTAL = wp.constant(0)
PSI_START_FRAME0 = wp.constant(1)
PSI_START_SIM = wp.constant(2)
PSI_START_ANCHOR = wp.constant(3)
PSI_START_STATS_SIZE = 4

# step_stats slots
PSI_STEP_SNAPSHOTS = wp.constant(0)
PSI_STEP_RESOLVED = wp.constant(1)
PSI_STEP_ACCEPTED = wp.constant(2)
PSI_STEP_OVERFLOW = wp.constant(3)
PSI_STEP_RING_OVERWRITE = wp.constant(4)
PSI_STEP_DONE = wp.constant(5)
PSI_STEP_TERMINATED = wp.constant(6)
PSI_STEP_STATS_SIZE = 7

_EMPTY = wp.constant(-1)


@dataclass(frozen=True)
class PSIReconBodyResetConfig(RSIReconBodyResetConfig):
    """ReconBody reset whose start states follow InterMimic PSI (no VOC help, no RSI bins or keyframe bank).

    It inherits the stock and RSI fields so the stock reset protocol (freeze, shoulder spread, finger offset, VOC
    hold during the freeze) and ``min_horizon`` keep their meaning; the RSI bin and keyframe fields are unused.
    """

    frame0_share: float = 0.1  # InterMimic hybridInitProb (omomo_train.yaml)
    keyframes: bool = False  # the RSI keyframe bank is replaced by the PSI buffer
    psi_slots: int = 2  # InterMimic physicalBufferSize 3 = reference + 2 simulation slots
    psi_horizon: int = 300  # InterMimic rolloutLength (frames)
    psi_stride: int = 10  # snapshot every N control steps (InterMimic scores every frame of the window)
    psi_settle_steps: int = 10  # first snapshot N steps after the world settles (InterMimic: start + 10)
    psi_threshold: float = 0.5  # InterMimic: insert when the survival fraction is > 0.5
    psi_score_decay: float = 2.4e-4  # per iteration; InterMimic 1e-5 per reset batch x 24 steps
    psi_max_candidates: int = 16384  # accepted snapshots queued per iteration (overflow is counted and dropped)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.keyframes:
            raise ValueError("PSI reset replaces the RSI keyframe bank: keyframes must be false")
        for name in ("psi_slots", "psi_horizon", "psi_stride", "psi_max_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"PSI {name} must be a positive integer, got {value!r}")
        if isinstance(self.psi_settle_steps, bool) or int(self.psi_settle_steps) != self.psi_settle_steps:
            raise ValueError(f"PSI psi_settle_steps must be an integer, got {self.psi_settle_steps!r}")
        if self.psi_settle_steps < 0:
            raise ValueError(f"PSI psi_settle_steps must be >= 0, got {self.psi_settle_steps}")
        if not (math.isfinite(self.psi_threshold) and 0.0 <= self.psi_threshold < 1.0):
            raise ValueError(f"PSI psi_threshold must be in [0, 1), got {self.psi_threshold}")
        if not (math.isfinite(self.psi_score_decay) and 0.0 <= self.psi_score_decay < 1.0):
            raise ValueError(f"PSI psi_score_decay must be in [0, 1), got {self.psi_score_decay}")

    def build(self, world_count: int, num_frames: int, device=None) -> PSIReconBodyReset:
        return PSIReconBodyReset(self, world_count, num_frames, device=device)


# ====================================================================== kernels
@wp.kernel
def sample_psi_reset(
    reset_mask: wp.array(dtype=wp.int32),
    reference_joint_q: wp.array(dtype=wp.float32, ndim=2),
    q_per_world: int,
    freeze_steps: int,
    reset_voc_scale: float,
    shoulder_spread: float,
    left_shoulder_q_id: int,
    right_shoulder_q_id: int,
    finger_q_ids: wp.array(dtype=wp.int32),
    seed: int,
    settle_steps: int,
    frame0_share: wp.array(dtype=wp.float32),
    frame_cdf: wp.array(dtype=wp.float32),
    num_cdf_frames: int,
    slot_score: wp.array(dtype=wp.float32),
    num_slots: int,
    ring: int,
    reset_count: wp.array(dtype=wp.int32),
    reset_frame: wp.array(dtype=wp.int32),
    finger_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    raw_joint_q_offset: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    start_frame: wp.array(dtype=wp.int32),
    start_slot: wp.array(dtype=wp.int32),
    clean: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    lane_help: wp.array(dtype=wp.float32),
    capture_after: wp.array(dtype=wp.int32),
    snap_count: wp.array(dtype=wp.int32),
    ring_step: wp.array(dtype=wp.int32),
    start_stats: wp.array(dtype=wp.int32),
) -> None:
    """Draw a PSI start (frame from the inverse-score CDF or the frame-0 anchor, then reference or a stored slot)."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return

    rng = wp.rand_init(seed, world + reset_count[world] * 104729)
    u_anchor = wp.randf(rng)
    u_frame = wp.randf(rng)
    u_slot = wp.randf(rng)

    frame = int(0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    anchor = int(0)  # noqa: UP018
    if u_anchor < frame0_share[0]:
        anchor = 1
    else:
        frame = num_cdf_frames - 1
        for i in range(num_cdf_frames):
            if u_frame < frame_cdf[i]:
                frame = i
                break

    # Reference weight 1, simulation slots their scores (InterMimic: ref_reward[i, :, t] / sum).
    total = float(1.0)  # noqa: UP018
    for k in range(num_slots):
        total += slot_score[frame * num_slots + k]
    slot = int(-1)  # noqa: UP018
    x = u_slot * total
    if x >= 1.0:
        acc = float(1.0)  # noqa: UP018
        last_positive = int(-1)  # noqa: UP018
        for k in range(num_slots):
            score = slot_score[frame * num_slots + k]
            if score > 0.0:
                last_positive = k
                acc += score
                if slot < 0 and x < acc:
                    slot = k
        if slot < 0:
            slot = last_positive  # float round-off at the top of the CDF

    immediate = int(0)  # noqa: UP018
    if frame == 0 or slot >= 0:
        immediate = 1

    base = world * q_per_world
    for coordinate in range(q_per_world):
        raw_joint_q_offset[base + coordinate] = 0.0
        reference_joint_q_offset[base + coordinate] = 0.0
    if immediate == 0:
        # Stock / RSI mid-clip reference start: spread shoulders and open fingers during the freeze.
        raw_joint_q_offset[base + left_shoulder_q_id] = shoulder_spread
        raw_joint_q_offset[base + right_shoulder_q_id] = -shoulder_spread
        reference_joint_q_offset[base + left_shoulder_q_id] = shoulder_spread
        reference_joint_q_offset[base + right_shoulder_q_id] = -shoulder_spread
        for finger in range(finger_q_ids.shape[0]):
            q_id = finger_q_ids[finger]
            offset = -reference_joint_q[frame, q_id]
            raw_joint_q_offset[base + q_id] = offset
            reference_joint_q_offset[base + q_id] = offset

    reset_frame[world] = frame
    finger_scale[world] = 1.0
    start_frame[world] = frame
    start_slot[world] = slot
    lane_help[world] = 0.0
    if immediate != 0:
        steps_since_reset[world] = freeze_steps + 1
        applied_voc_scale[world] = 0.0
        capture_after[world] = settle_steps
    else:
        steps_since_reset[world] = 0
        applied_voc_scale[world] = reset_voc_scale
        capture_after[world] = freeze_steps + settle_steps
    clean[world] = immediate
    active[world] = 1
    snap_count[world] = 0
    for r in range(ring):
        ring_step[world * ring + r] = _EMPTY
    reset_count[world] += 1

    wp.atomic_add(start_stats, PSI_START_TOTAL, 1)
    if slot >= 0:
        wp.atomic_add(start_stats, PSI_START_SIM, 1)
    elif frame == 0:
        wp.atomic_add(start_stats, PSI_START_FRAME0, 1)
    if anchor != 0:
        wp.atomic_add(start_stats, PSI_START_ANCHOR, 1)


@wp.kernel
def psi_transition(
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    q_per_world: int,
    dof_per_world: int,
    object_root_q_ids: wp.array(dtype=wp.int32),
    object_root_position_offsets_w: wp.array(dtype=wp.vec3),
    num_objects: int,
    last_frame: int,
    max_start_frame: int,
    horizon_steps: int,
    stride: int,
    ring: int,
    threshold: float,
    max_candidates: int,
    capture_after: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    snap_count: wp.array(dtype=wp.int32),
    ring_step: wp.array(dtype=wp.int32),
    ring_frame: wp.array(dtype=wp.int32),
    ring_q: wp.array(dtype=wp.float32),
    ring_qd: wp.array(dtype=wp.float32),
    cand_count: wp.array(dtype=wp.int32),
    cand_frame: wp.array(dtype=wp.int32),
    cand_score: wp.array(dtype=wp.float32),
    cand_q: wp.array(dtype=wp.float32),
    cand_qd: wp.array(dtype=wp.float32),
    step_stats: wp.array(dtype=wp.int32),
) -> None:
    """Per step, after physics/termination/advance and before auto-reset: score due snapshots, then take a new one."""
    world = wp.tid()
    is_terminated = terminated[world] != 0
    is_truncated = truncated[world] != 0
    done = is_terminated or is_truncated
    steps = int(episode_step[world])
    if done:
        steps = steps + 1  # the advance skipped this world, so episode_step does not count this transition yet

    # Resolve snapshots whose window is over (the world ended, or it lasted the whole window).
    for r in range(ring):
        idx = world * ring + r
        taken = ring_step[idx]
        if taken < 0:
            continue
        horizon = wp.min(horizon_steps, last_frame - ring_frame[idx])
        if horizon < 1:
            horizon = 1
        survived = steps - taken
        if done or survived >= horizon:
            score = wp.min(1.0, float(survived) / float(horizon))
            wp.atomic_add(step_stats, PSI_STEP_RESOLVED, 1)
            if score > threshold:
                c = wp.atomic_add(cand_count, 0, 1)
                if c < max_candidates:
                    wp.atomic_add(step_stats, PSI_STEP_ACCEPTED, 1)
                    cand_frame[c] = ring_frame[idx]
                    cand_score[c] = score
                    for q in range(q_per_world):
                        cand_q[c * q_per_world + q] = ring_q[idx * q_per_world + q]
                    for d in range(dof_per_world):
                        cand_qd[c * dof_per_world + d] = ring_qd[idx * dof_per_world + d]
                else:
                    wp.atomic_add(step_stats, PSI_STEP_OVERFLOW, 1)
            ring_step[idx] = _EMPTY

    if done:
        if active[world] != 0:
            wp.atomic_add(step_stats, PSI_STEP_DONE, 1)
            if is_terminated and not is_truncated:
                wp.atomic_add(step_stats, PSI_STEP_TERMINATED, 1)
        active[world] = 0
        return
    if active[world] == 0:
        return

    # Snapshot every `stride` steps once settled, at frames a reset may start from (never frame 0, the exact start).
    frame = timestep[world]
    after = capture_after[world]
    if steps < after or (steps - after) % stride != 0:
        return
    if frame < 1 or frame > max_start_frame:
        return
    r = snap_count[world] % ring
    idx = world * ring + r
    if ring_step[idx] >= 0:
        wp.atomic_add(step_stats, PSI_STEP_RING_OVERWRITE, 1)
    base_q = world * q_per_world
    for c in range(q_per_world):
        ring_q[idx * q_per_world + c] = joint_q[base_q + c]
    for object_id in range(num_objects):
        root = idx * q_per_world + object_root_q_ids[object_id]
        offset = object_root_position_offsets_w[world * num_objects + object_id]
        ring_q[root] = ring_q[root] - offset[0]
        ring_q[root + 1] = ring_q[root + 1] - offset[1]
        ring_q[root + 2] = ring_q[root + 2] - offset[2]
    base_qd = world * dof_per_world
    for d in range(dof_per_world):
        ring_qd[idx * dof_per_world + d] = joint_qd[base_qd + d]
    ring_step[idx] = steps
    ring_frame[idx] = frame
    snap_count[world] += 1
    wp.atomic_add(step_stats, PSI_STEP_SNAPSHOTS, 1)


@wp.kernel
def clear_psi_lanes(
    reset_mask: wp.array(dtype=wp.int32),
    ring: int,
    start_slot: wp.array(dtype=wp.int32),
    snap_count: wp.array(dtype=wp.int32),
    ring_step: wp.array(dtype=wp.int32),
) -> None:
    """Explicit resets: no stored-state restore and no pending snapshots."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    start_slot[world] = _EMPTY
    snap_count[world] = 0
    for r in range(ring):
        ring_step[world * ring + r] = _EMPTY


# ====================================================================== reset policy
class PSIReconBodyReset(RSIReconBodyReset):
    """RSI-compatible reset hooks (``ResetPolicy``, ``ResetContext``, ``ReferenceOffsetReset``, ``RSIHooks``) with
    InterMimic PSI start states. The RSI bins and keyframe bank of the parent are allocated but never used."""

    def __init__(self, config: PSIReconBodyResetConfig, world_count: int, num_frames: int, device=None) -> None:
        super().__init__(config, world_count, num_frames, device=device)
        dev, world = self.device, self.world_count
        self.num_slots = int(config.psi_slots)
        self.ring = int(config.psi_horizon) // int(config.psi_stride) + 2
        self.num_cdf_frames = int(self.max_start_frame) + 1
        self.last_frame = self.num_frames - 1

        # Host buffer (float64 scores; float32 states) and its device mirror
        self.score = np.zeros((self.num_frames, self.num_slots), dtype=np.float64)
        self.slot_score = wp.zeros(self.num_frames * self.num_slots, dtype=wp.float32, device=dev)
        self.frame_cdf = wp.array(self._frame_cdf(), dtype=wp.float32, device=dev)
        self.psi_q_host = np.zeros(0, dtype=np.float32)
        self.psi_qd_host = np.zeros(0, dtype=np.float32)
        self.psi_q = wp.zeros(0, dtype=wp.float32, device=dev)
        self.psi_qd = wp.zeros(0, dtype=wp.float32, device=dev)

        # Per-world lineage
        self.start_frame = wp.zeros(world, dtype=wp.int32, device=dev)
        self.start_slot = wp.full(world, -1, dtype=wp.int32, device=dev)
        self.capture_after = wp.zeros(world, dtype=wp.int32, device=dev)
        self.snap_count = wp.zeros(world, dtype=wp.int32, device=dev)
        self.ring_step = wp.full(world * self.ring, -1, dtype=wp.int32, device=dev)
        self.ring_frame = wp.zeros(world * self.ring, dtype=wp.int32, device=dev)
        self.ring_q = wp.zeros(0, dtype=wp.float32, device=dev)
        self.ring_qd = wp.zeros(0, dtype=wp.float32, device=dev)

        # Accepted-snapshot queue, emptied by host_update once per iteration
        max_candidates = int(config.psi_max_candidates)
        self.cand_count = wp.zeros(1, dtype=wp.int32, device=dev)
        self.cand_frame = wp.zeros(max_candidates, dtype=wp.int32, device=dev)
        self.cand_score = wp.zeros(max_candidates, dtype=wp.float32, device=dev)
        self.cand_q = wp.zeros(0, dtype=wp.float32, device=dev)
        self.cand_qd = wp.zeros(0, dtype=wp.float32, device=dev)

        self.psi_start_stats = wp.zeros(PSI_START_STATS_SIZE, dtype=wp.int32, device=dev)
        self.step_stats = wp.zeros(PSI_STEP_STATS_SIZE, dtype=wp.int32, device=dev)
        self.inserted_total = 0

    # ------------------------------------------------------------------ binding
    def bind_env(self, env: Any) -> None:
        super().bind_env(env)
        q, d, dev = self.q_per_world, self.dof_per_world, self.device
        slots = self.num_frames * self.num_slots
        if self.psi_q_host.shape[0] != slots * q:
            self.psi_q_host = np.zeros(slots * q, dtype=np.float32)
            self.psi_qd_host = np.zeros(slots * d, dtype=np.float32)
            self.psi_q = wp.zeros(slots * q, dtype=wp.float32, device=dev)
            self.psi_qd = wp.zeros(slots * d, dtype=wp.float32, device=dev)
            self.ring_q = wp.zeros(self.world_count * self.ring * q, dtype=wp.float32, device=dev)
            self.ring_qd = wp.zeros(self.world_count * self.ring * d, dtype=wp.float32, device=dev)
            max_candidates = int(self.config.psi_max_candidates)
            self.cand_q = wp.zeros(max_candidates * q, dtype=wp.float32, device=dev)
            self.cand_qd = wp.zeros(max_candidates * d, dtype=wp.float32, device=dev)

    # ------------------------------------------------------------------ stock protocol
    def sample(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        """Draw PSI starts for the selected worlds; there is no VOC help (the global curriculum scale is unused)."""
        del curriculum_voc_scale
        self._require_bound()
        config = self.config
        wp.launch(
            sample_psi_reset,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.reference_joint_q,
                self.q_per_world,
                config.reset_freeze_steps,
                config.reset_voc_scale,
                config.shoulder_spread,
                self.left_shoulder_q_id,
                self.right_shoulder_q_id,
                self.finger_q_ids,
                config.seed,
                int(config.psi_settle_steps),
                self.frame0_share,
                self.frame_cdf,
                self.num_cdf_frames,
                self.slot_score,
                self.num_slots,
                self.ring,
            ],
            outputs=[
                self.reset_count,
                self.reset_frame,
                self.finger_scale,
                self.steps_since_reset,
                self.applied_voc_scale,
                self.raw_joint_q_offset,
                self.reference_joint_q_offset,
                self.start_frame,
                self.start_slot,
                self.clean,
                self.active,
                self.help,
                self.capture_after,
                self.snap_count,
                self.ring_step,
                self.psi_start_stats,
            ],
            device=self.device,
        )

    def prepare_explicit(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        """Stock explicit-reset semantics (parent), and no stored-state restore or pending snapshot for the lane."""
        super().prepare_explicit(reset_mask, curriculum_voc_scale)
        wp.launch(
            clear_psi_lanes,
            dim=self.world_count,
            inputs=[reset_mask, self.ring],
            outputs=[self.start_slot, self.snap_count, self.ring_step],
            device=self.device,
        )

    # ------------------------------------------------------------------ RSI hooks
    def on_transition(self, state: Any) -> None:
        """Score due snapshots and queue the accepted ones, then snapshot; inside the step graph, before auto-reset."""
        self._require_env()
        config = self.config
        wp.launch(
            psi_transition,
            dim=self.world_count,
            inputs=[
                self._terminated,
                self._truncated,
                self._timestep,
                self._episode_step,
                state.joint_q,
                state.joint_qd,
                self.q_per_world,
                self.dof_per_world,
                self._object_root_q_ids,
                self._object_root_position_offsets_w,
                self._num_objects,
                self.last_frame,
                self.max_start_frame,
                int(config.psi_horizon),
                int(config.psi_stride),
                self.ring,
                float(config.psi_threshold),
                int(config.psi_max_candidates),
                self.capture_after,
            ],
            outputs=[
                self.active,
                self.snap_count,
                self.ring_step,
                self.ring_frame,
                self.ring_q,
                self.ring_qd,
                self.cand_count,
                self.cand_frame,
                self.cand_score,
                self.cand_q,
                self.cand_qd,
                self.step_stats,
            ],
            device=self.device,
        )

    def restore(self, state: Any, reset_mask: wp.array) -> None:
        """Overwrite just-reset worlds that drew a simulation slot with its stored state, then refresh FK."""
        self._require_env()
        wp.launch(
            rsi_kernels.restore_keyframe,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.start_slot,
                self.start_frame,
                self.num_slots,
                self.q_per_world,
                self.dof_per_world,
                self.psi_q,
                self.psi_qd,
                self._object_root_q_ids,
                self._object_root_position_offsets_w,
                self._num_objects,
            ],
            outputs=[state.joint_q, state.joint_qd],
            device=self.device,
        )
        if hasattr(self._model, "joint_q"):  # a real Newton model; CPU kernel tests bind a stand-in without FK data
            newton.eval_fk(self._model, state.joint_q, state.joint_qd, state)

    # ------------------------------------------------------------------ host side
    def _frame_cdf(self) -> np.ndarray:
        """float32 CDF over start frames ``0..max_start_frame``: weight ``1 / (1 + sum of slot scores)``."""
        weights = 1.0 / (1.0 + self.score[: self.num_cdf_frames].sum(axis=1))
        cdf = np.cumsum(weights / weights.sum()).astype(np.float32)
        cdf[-1] = 1.0
        return cdf

    def _insert(self, frames: np.ndarray, scores: np.ndarray, q: np.ndarray, qd: np.ndarray) -> int:
        """InterMimic insertion: each accepted snapshot replaces the lowest-scoring simulation slot of its frame."""
        qn, dn, k = self.q_per_world, self.dof_per_world, self.num_slots
        bank_q = self.psi_q_host.reshape(self.num_frames, k, qn)
        bank_qd = self.psi_qd_host.reshape(self.num_frames, k, dn)
        inserted = 0
        for i in range(frames.shape[0]):
            frame = int(frames[i])
            if not 0 <= frame < self.num_frames:
                continue
            slot = int(np.argmin(self.score[frame]))
            self.score[frame, slot] = float(scores[i])
            bank_q[frame, slot] = q[i]
            bank_qd[frame, slot] = qd[i]
            inserted += 1
        return inserted

    def upload_buffer(self) -> None:
        self.slot_score.assign(self.score.reshape(-1).astype(np.float32))
        self.frame_cdf.assign(self._frame_cdf())
        if self.psi_q.shape[0]:
            self.psi_q.assign(self.psi_q_host)
            self.psi_qd.assign(self.psi_qd_host)

    def upload_bins(self) -> None:
        """The RSI bins are unused; keep the device start tables of PSI current instead."""
        self.upload_buffer()

    def host_update(self, iteration: int) -> dict[str, float]:
        """Once per iteration, outside the graph: insert the queued snapshots, decay scores, upload buffer and CDF."""
        iteration = int(iteration)
        config = self.config
        count = int(self.cand_count.numpy()[0])
        n = min(count, int(config.psi_max_candidates))
        inserted = 0
        if n > 0:
            qn, dn = self.q_per_world, self.dof_per_world
            frames = self.cand_frame.numpy()[:n]
            scores = self.cand_score.numpy()[:n]
            q = self.cand_q.numpy()[: n * qn].reshape(n, qn)
            qd = self.cand_qd.numpy()[: n * dn].reshape(n, dn)
            inserted = self._insert(frames, scores, q, qd)
        self.cand_count.zero_()
        self.score *= 1.0 - float(config.psi_score_decay)
        self.upload_buffer()
        self.inserted_total += inserted
        self._last_iteration = iteration

        starts = self.psi_start_stats.numpy().astype(np.int64)
        steps = self.step_stats.numpy().astype(np.int64)
        self.psi_start_stats.zero_()
        self.step_stats.zero_()
        startable = self.score[: self.num_cdf_frames]
        filled = startable > 0.0
        total_starts = max(int(starts[PSI_START_TOTAL]), 1)
        done = max(int(steps[PSI_STEP_DONE]), 1)
        return {
            "psi/inserted": float(inserted),
            "psi/inserted_total": float(self.inserted_total),
            "psi/queued": float(count),
            "psi/overflow": float(steps[PSI_STEP_OVERFLOW]),
            "psi/snapshots": float(steps[PSI_STEP_SNAPSHOTS]),
            "psi/resolved": float(steps[PSI_STEP_RESOLVED]),
            "psi/accept_frac": float(steps[PSI_STEP_ACCEPTED]) / max(int(steps[PSI_STEP_RESOLVED]), 1),
            "psi/ring_overwrite": float(steps[PSI_STEP_RING_OVERWRITE]),
            "psi/frames_with_state": float(filled.any(axis=1).mean()),
            "psi/slots_filled": float(filled.mean()),
            "psi/score_mean": float(startable[filled].mean()) if filled.any() else 0.0,
            "psi/sim_start_frac": float(starts[PSI_START_SIM]) / total_starts,
            "psi/frame0_start_frac": float(starts[PSI_START_FRAME0]) / total_starts,
            "psi/anchor_frac": float(starts[PSI_START_ANCHOR]) / total_starts,
            "psi/episodes_done": float(steps[PSI_STEP_DONE]),
            "psi/fail_rate": float(steps[PSI_STEP_TERMINATED]) / done,
        }

    def help_zero_since(self) -> int | None:
        """PSI has no VOC help schedule, so the RSI help-zero early stop does not apply."""
        return None

    # ------------------------------------------------------------------ save / resume
    def _psi_metadata(self, iteration: int) -> dict[str, str]:
        return {
            "format": PSI_SIDECAR_FORMAT,
            "version": str(PSI_SIDECAR_VERSION),
            "num_frames": str(self.num_frames),
            "num_slots": str(self.num_slots),
            "q_per_world": str(self.q_per_world),
            "dof_per_world": str(self.dof_per_world),
            "iteration": str(int(iteration)),
        }

    def save_state(self, path: str | os.PathLike) -> None:
        """Write the PSI buffer (scores and stored states) to the run's sidecar (``rsi_{iteration}.safetensors``)."""
        from safetensors.numpy import save_file  # noqa: PLC0415 - training-only dependency

        self._require_env()
        path = Path(path)
        match = _SIDECAR_NAME.match(path.name)
        iteration = int(match.group(1)) if match else self._last_iteration + 1
        k, qn, dn = self.num_slots, self.q_per_world, self.dof_per_world
        tensors = {
            "psi.score": np.ascontiguousarray(self.score),
            "psi.q": np.ascontiguousarray(self.psi_q_host.reshape(self.num_frames, k, qn)),
            "psi.qd": np.ascontiguousarray(self.psi_qd_host.reshape(self.num_frames, k, dn)),
            "psi.inserted_total": np.asarray([self.inserted_total], dtype=np.int64),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        save_file(tensors, str(temporary), metadata=self._psi_metadata(iteration))
        os.replace(temporary, path)

    def load_state(self, path: str | os.PathLike) -> None:
        """Restore the PSI buffer from a sidecar written by ``save_state`` for the same clip and scene."""
        from safetensors import safe_open  # noqa: PLC0415 - training-only dependency
        from safetensors.numpy import load_file  # noqa: PLC0415

        self._require_env()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"PSI sidecar {path} does not exist")
        with safe_open(str(path), framework="numpy") as handle:
            metadata = dict(handle.metadata() or {})
        match = _SIDECAR_NAME.match(path.name)
        expected = self._psi_metadata(int(match.group(1)) if match else int(metadata.get("iteration", -1)))
        mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatched:
            raise ValueError(f"PSI sidecar {path} does not match this run (stored, expected): {mismatched}")
        tensors = load_file(str(path))
        k, qn, dn = self.num_slots, self.q_per_world, self.dof_per_world
        score = np.asarray(tensors["psi.score"], dtype=np.float64)
        if score.shape != (self.num_frames, k):
            raise ValueError(f"PSI sidecar {path} has scores of shape {score.shape}; expected ({self.num_frames}, {k})")
        self.score = score.copy()
        self.psi_q_host = np.asarray(tensors["psi.q"], dtype=np.float32).reshape(-1).copy()
        self.psi_qd_host = np.asarray(tensors["psi.qd"], dtype=np.float32).reshape(-1).copy()
        if self.psi_q_host.shape[0] != self.num_frames * k * qn or self.psi_qd_host.shape[0] != self.num_frames * k * dn:
            raise ValueError(f"PSI sidecar {path} stores states of the wrong size")
        self.inserted_total = int(np.asarray(tensors["psi.inserted_total"]).reshape(-1)[0])
        for array in (self.psi_start_stats, self.step_stats, self.cand_count):
            array.zero_()
        self._last_iteration = int(metadata["iteration"]) - 1
        self.upload_buffer()

    # ------------------------------------------------------------------ test hooks
    def debug_force_keyframe(self, world: int, bin_index: int, slot: int) -> None:
        raise NotImplementedError("PSI reset has no RSI keyframe bank")
