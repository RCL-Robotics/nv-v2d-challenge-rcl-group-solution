# SPDX-License-Identifier: Apache-2.0
"""RSI v3 (fixed) ReconBody reset: Coordinates_hand reference-state initialization on the FlashCHORD reset protocol.

Per start bin (``bin_frames`` reference frames) the host keeps a VOC help top that drops by ``step`` when the bin's
episodes succeed and a failure EMA that raises the bin's start probability (``bins.RSIBins``). On the device every
reset world draws: with probability ``frame0_share`` the evaluator's exact frame-0 start, otherwise a start bin from
the failure-weighted CDF and a frame inside it, served from the keyframe bank of reached simulator states when the
bin holds ``kf_min_fill`` states whose help tag is <= the bin's current top (``kf_to_end`` semantics). Each lane gets
help ``U[top/band, top]`` of its start bin for the whole episode (scalar on the whole VOC wrench).

Frame-0 and keyframe starts are immediate (no freeze, no shoulder spread, no finger offsets); mid-clip reference
starts keep the stock 50-step freeze with ``reset_voc_scale`` decaying to the lane's help over the last 10 steps.

Dev ``help_mode="success"`` (help force fix): the bin tops follow each bin's EMA pass-through pass rate instead of
the iteration count (``bins.RSIBins`` docstring), and a lane's help is its band factor times the top at its current
frame (``kernels.update_frame_help``), so help stays where the clip is hard instead of following the start bin.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp

from flash_chord.data.reference import Reference
from flash_chord.lifecycle.reset import RESET_CONTEXT_NAMES, ReconBodyResetConfig
from flash_chord.lifecycle.rsi import kernels
from flash_chord.lifecycle.rsi.bank import KeyframeBank
from flash_chord.lifecycle.rsi.bins import RSIBinParams, RSIBins
from flash_chord.scene.builder import Scene

SIDECAR_FORMAT = "rsi_state"
SIDECAR_VERSION = 1
_SIDECAR_NAME = re.compile(r"^rsi_(\d+)\.safetensors$")


@dataclass(frozen=True)
class RSIReconBodyResetConfig(ReconBodyResetConfig):
    """ReconBody reset whose start distribution and per-lane VOC help follow RSI v3 (fixed).

    The stock fields keep their meaning; RSI owns the start distribution, so the stock frame window and
    first-frame probabilities must stay at their neutral values. Defaults are the user's D5 choices with the
    remaining values from CH ``RSI_DEFAULTS`` (curriculum.py:80-106) and ``KeyframeBuffer`` (keyframes.py:26, 72).
    """

    bin_frames: int = 25
    uniform_mix: float = 0.3
    credit_gamma: float = 0.75
    fail_alpha: float = 0.02
    success_up: float = 0.8
    success_down: float = 0.4
    step: float = 0.9
    band: float = 2.0
    min_count: int = 64
    min_gap: int = 40
    wait_iterations: int = 100
    top0: float = 1.0
    zero_below: float = 6.25e-4
    deadline: int = 11000
    zero_ramp: int = 1000
    min_horizon: int = 10
    frame0_share: float = 0.25
    start_update: int = 0
    keyframes: bool = True
    kf_to_end: bool = True
    kf_capacity: int = 256
    kf_prob: float = 0.002
    kf_min_fill: int = 32
    kf_min_task: float = 0.6
    kf_min_stepped: int = 15
    kf_step_score_min: float = 0.3
    task_pos_scale: float = 10.0
    # Dev rolling start: mid-clip reference starts run the reference clock for the last run_in_steps of the
    # reset_freeze_steps hold (VOC hold and termination grace unchanged), so robot and object already move with the
    # reference when the hold ends; 0 is the stock release (clock frozen for the whole hold).
    run_in_steps: int = 0
    # Dev help force fix (lifecycle/rsi/bins.py docstring): "success" drives each bin's help top by its EMA
    # pass-through pass rate (monotone ratchet, final cutoff) and applies it at the lane's current frame;
    # "deadline" is the CH schedule (start-bin windows, deadline floor, start-bin help for the whole episode).
    help_mode: str = "deadline"
    help_ema_tau: float = 200.0
    help_min_samples: float = 256.0
    help_wilson_z: float = 1.645
    help_lookahead_bins: int = 3
    help_lookahead_decay: float = 0.8
    help_success_lo: float = 0.6
    help_success_hi: float = 0.97
    help_update_interval: int = 100
    help_final_iteration: int = 0
    help_final_fraction: float = 0.85
    help_final_ramp_fraction: float = 0.05
    # Dev hard-start sampling (lifecycle/rsi/bins.py docstring): "hazard" weights the non-anchor starts by the
    # pass-through hazard (softmax at start_temperature, start_lookahead_* window, start_cap per bin); "credit" is CH.
    start_mode: str = "credit"
    start_temperature: float = 0.1
    start_lookahead_bins: int = 3
    start_lookahead_decay: float = 0.7
    start_cap: float = 1.0
    # Dev: reference starts begin start_frame_shift frames before the sampled frame (the rolling start's run_in_steps
    # makes the clock reach it as the hold ends); pass_after_grace counts no pass-through outcome while the palm and
    # object terminations are masked (episode step <= reset_freeze_steps).
    start_frame_shift: int = 0
    pass_after_grace: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.start_frame_shift, bool) or int(self.start_frame_shift) != self.start_frame_shift:
            raise TypeError(f"RSI start_frame_shift must be an integer, got {self.start_frame_shift!r}")
        if not 0 <= self.start_frame_shift <= self.reset_freeze_steps:
            raise ValueError(
                f"RSI start_frame_shift must be in [0, reset_freeze_steps={self.reset_freeze_steps}], "
                f"got {self.start_frame_shift}"
            )
        if not isinstance(self.pass_after_grace, bool):
            raise TypeError(f"RSI pass_after_grace must be a bool, got {self.pass_after_grace!r}")
        if isinstance(self.run_in_steps, bool) or int(self.run_in_steps) != self.run_in_steps:
            raise TypeError(f"RSI run_in_steps must be an integer, got {self.run_in_steps!r}")
        if not 0 <= self.run_in_steps <= self.reset_freeze_steps:
            raise ValueError(
                f"RSI run_in_steps must be in [0, reset_freeze_steps={self.reset_freeze_steps}], got {self.run_in_steps}"
            )
        if self.start_frame != 0 or self.end_frame != -1:
            raise ValueError("RSI reset owns the start distribution: start_frame must be 0 and end_frame -1")
        if self.always_reset_to_first_frame:
            raise ValueError("RSI reset owns the start distribution: always_reset_to_first_frame must be false")
        if self.reset_to_first_frame_probability != 0.0 or self.immediate_first_frame_probability != 0.0:
            raise ValueError(
                "RSI reset owns the start distribution: reset_to_first_frame_probability and "
                "immediate_first_frame_probability must be 0 (use frame0_share)"
            )
        RSIBinParams.from_object(self)  # CH _rsi_init range checks (curriculum.py:378-385)
        if not (math.isfinite(self.frame0_share) and 0.0 <= self.frame0_share < 1.0):
            raise ValueError(f"RSI frame0_share must be in [0, 1), got {self.frame0_share}")
        for name in ("keyframes", "kf_to_end"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"RSI {name} must be a bool, got {getattr(self, name)!r}")
        if self.keyframes and not self.kf_to_end:
            raise NotImplementedError("RSI keyframes without kf_to_end (confirmed-window capture) are not ported")
        for name in ("kf_capacity", "kf_min_fill", "kf_min_stepped"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"RSI {name} must be a positive integer, got {value!r}")
        if self.kf_min_fill > self.kf_capacity:
            raise ValueError(f"RSI kf_min_fill ({self.kf_min_fill}) cannot exceed kf_capacity ({self.kf_capacity})")
        if not (math.isfinite(self.kf_prob) and 0.0 <= self.kf_prob <= 1.0):
            raise ValueError(f"RSI kf_prob must be in [0, 1], got {self.kf_prob}")
        for name in ("kf_min_task", "kf_step_score_min"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"RSI {name} must be finite")
        if not (math.isfinite(self.task_pos_scale) and self.task_pos_scale >= 0.0):
            raise ValueError(f"RSI task_pos_scale must be finite and non-negative, got {self.task_pos_scale}")

    def build(self, world_count: int, num_frames: int, device=None) -> RSIReconBodyReset:
        return RSIReconBodyReset(self, world_count, num_frames, device=device)


class RSIReconBodyReset:
    """Graph-capturable RSI reset state (``ResetPolicy``, ``ResetContext``, ``ReferenceOffsetReset``, ``RSIHooks``).

    It deliberately does not implement ``set_reset_to_first_frame_probability`` /
    ``set_immediate_first_frame_probability``: curriculum stages must leave those overrides ``None``.
    """

    context_names = RESET_CONTEXT_NAMES

    def __init__(self, config: RSIReconBodyResetConfig, world_count: int, num_frames: int, device=None) -> None:
        if world_count <= 0:
            raise ValueError(f"RSI reset needs a positive world count, got {world_count}")
        self.config = config
        self.world_count = int(world_count)
        self.num_frames = int(num_frames)
        self.device = wp.get_device(device)
        self.bins = RSIBins(self.num_frames, RSIBinParams.from_object(config))
        self.num_bins = self.bins.num_bins
        self.max_start_frame = self.bins.max_start_frame
        world, bins, dev = self.world_count, self.num_bins, self.device

        # ResetPolicy protocol fields (reset.py:259-289)
        self.reset_frame = wp.zeros(world, dtype=wp.int32, device=dev)
        self.finger_scale = wp.ones(world, dtype=wp.float32, device=dev)
        self.applied_voc_scale = wp.ones(world, dtype=wp.float32, device=dev)
        self.steps_since_reset = wp.zeros(world, dtype=wp.int32, device=dev)
        self.reset_count = wp.zeros(world, dtype=wp.int32, device=dev)

        # Bin tables (host-updated once per iteration with .assign(); never kernel scalars)
        self.top = wp.array(self.bins.top.astype(np.float32), dtype=wp.float32, device=dev)
        self.cdf = wp.array(self.bins.start_cdf(), dtype=wp.float32, device=dev)
        self.bin_lo = wp.array(self.bins.bin_lo, dtype=wp.int32, device=dev)
        self.bin_hi = wp.array(self.bins.bin_hi, dtype=wp.int32, device=dev)
        self.frame0_share = wp.array([config.frame0_share], dtype=wp.float32, device=dev)
        self.n_iter = wp.zeros(bins, dtype=wp.int32, device=dev)
        self.f_iter = wp.zeros(bins, dtype=wp.int32, device=dev)
        self.requested = wp.zeros(bins, dtype=wp.int32, device=dev)
        self.served = wp.zeros(bins, dtype=wp.int32, device=dev)
        self.start_stats = wp.zeros(kernels.START_STATS_SIZE, dtype=wp.int32, device=dev)

        # Per-world lineage and bookkeeping
        self.start_bin = wp.zeros(world, dtype=wp.int32, device=dev)
        self.kf_slot = wp.full(world, -1, dtype=wp.int32, device=dev)
        self.clean = wp.zeros(world, dtype=wp.int32, device=dev)
        self.active = wp.zeros(world, dtype=wp.int32, device=dev)
        self.step_count = wp.zeros(world, dtype=wp.int32, device=dev)
        self.help = wp.zeros(world, dtype=wp.float32, device=dev)
        self.tag = wp.zeros(world, dtype=wp.float32, device=dev)
        self.cum_score = wp.zeros(world, dtype=wp.float32, device=dev)
        self.pend_on = wp.zeros(world, dtype=wp.int32, device=dev)
        self.pend_frame = wp.zeros(world, dtype=wp.int32, device=dev)
        self.pend_step = wp.zeros(world, dtype=wp.int32, device=dev)
        self.pend_cum = wp.zeros(world, dtype=wp.float32, device=dev)
        self.pend_tag = wp.zeros(world, dtype=wp.float32, device=dev)
        self.pend_q = wp.zeros(0, dtype=wp.float32, device=dev)
        self.pend_qd = wp.zeros(0, dtype=wp.float32, device=dev)
        self.force_bin = wp.full(world, -1, dtype=wp.int32, device=dev)
        self.force_slot = wp.zeros(world, dtype=wp.int32, device=dev)
        # Success-mode help: band factor and pass-through bookkeeping per lane, pass/fail counts per bin
        self.frame_help = int(config.help_mode == "success")
        self.help_factor = wp.ones(world, dtype=wp.float32, device=dev)
        self.visit_bin = wp.zeros(world, dtype=wp.int32, device=dev)
        self.pass_iter = wp.zeros(bins, dtype=wp.int32, device=dev)
        self.pass_fail_iter = wp.zeros(bins, dtype=wp.int32, device=dev)

        self.bank = KeyframeBank(bins, config.kf_capacity, device=dev)

        # Scene binding (bind_reference_joint_q_offset)
        self.q_per_world = 0
        self.left_shoulder_q_id = -1
        self.right_shoulder_q_id = -1
        self.raw_joint_q_offset = wp.empty(0, dtype=wp.float32, device=dev)
        self.reference_joint_q_offset = wp.empty(0, dtype=wp.float32, device=dev)
        self.reference_joint_q = wp.empty((0, 0), dtype=wp.float32, device=dev)
        self.finger_q_ids = wp.empty(0, dtype=wp.int32, device=dev)

        # Environment binding (bind_env)
        self.dof_per_world = 0
        self._model = None
        self._terminated = None
        self._truncated = None
        self._timestep = None
        self._episode_step = None
        self._object_position_error = None
        self._object_orientation_error = None
        self._object_root_q_ids = wp.zeros(0, dtype=wp.int32, device=dev)
        self._object_root_position_offsets_w = wp.zeros(0, dtype=wp.vec3, device=dev)
        self._num_objects = 0

        # Logging cadence of the bank statistics (B*C tag scan + W pending flags); the runner logs every 10.
        self.stats_interval = 10
        self._last_iteration = -1
        self._cached_eligible = np.zeros(bins, dtype=np.int64)
        self._cached_pending_frac = 0.0

    # ------------------------------------------------------------------ binding
    def bind_reference_joint_q_offset(self, scene: Scene, reference: Reference, output: wp.array) -> None:
        """Bind scene mappings and the env-owned offset buffer (same mappings as stock, reset.py:868-896)."""
        if scene.world_count != self.world_count:
            raise ValueError(f"RSI reset built for {self.world_count} worlds; scene has {scene.world_count}")
        if reference.num_frames != self.num_frames:
            raise ValueError(f"RSI reset built for {self.num_frames} frames; reference has {reference.num_frames}")
        scalar = scene.layout.scalar_joints
        if scalar is None:
            raise ValueError("RSI ReconBody reset requires named scalar robot joints")
        q_by_name = dict(zip(scalar.names, scalar.q_ids, strict=True))
        shoulder_names = ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint")
        missing = tuple(name for name in shoulder_names if name not in q_by_name)
        if missing:
            raise ValueError(f"RSI ReconBody reset is missing shoulder joints {missing}")
        finger_q_ids = tuple(q_id for hand in scene.layout.hands for q_id in hand.finger_q_ids)
        self.bind_arrays(
            q_per_world=scene.model.joint_coord_count // scene.world_count,
            left_shoulder_q_id=q_by_name[shoulder_names[0]],
            right_shoulder_q_id=q_by_name[shoulder_names[1]],
            reference_joint_q=scene.robot_reference.joint_q,
            finger_q_ids=finger_q_ids,
            output=output,
        )

    def bind_arrays(
        self,
        *,
        q_per_world: int,
        left_shoulder_q_id: int,
        right_shoulder_q_id: int,
        reference_joint_q: np.ndarray,
        finger_q_ids: Sequence[int],
        output: wp.array,
    ) -> None:
        """Scene-free binding used by ``bind_reference_joint_q_offset`` and by kernel-level tests."""
        expected = (self.world_count * q_per_world,)
        if output.shape != expected or output.dtype != wp.float32:
            raise ValueError(f"RSI reference offset must be float32 with shape {expected}")
        reference_joint_q = np.asarray(reference_joint_q, dtype=np.float32)
        if reference_joint_q.ndim != 2 or reference_joint_q.shape[0] != self.num_frames:
            raise ValueError(
                f"RSI reference joint table has shape {reference_joint_q.shape}; expected ({self.num_frames}, n)"
            )
        self.q_per_world = int(q_per_world)
        self.left_shoulder_q_id = int(left_shoulder_q_id)
        self.right_shoulder_q_id = int(right_shoulder_q_id)
        self.reference_joint_q_offset = output
        self.raw_joint_q_offset = wp.zeros(expected[0], dtype=wp.float32, device=output.device)
        self.reference_joint_q = wp.array(reference_joint_q, dtype=wp.float32, device=output.device)
        self.finger_q_ids = wp.array(list(finger_q_ids), dtype=wp.int32, device=output.device)

    def bind_env(self, env: Any) -> None:
        """Bind the environment buffers the hooks read; call once at the end of ``RLEnv.__init__``."""
        self._require_bound()
        if int(env.world_count) != self.world_count:
            raise ValueError(f"RSI reset built for {self.world_count} worlds; environment has {env.world_count}")
        termination = env.termination
        missing = [
            name
            for name in ("object_position_error", "object_orientation_error")
            if getattr(termination, name, None) is None
        ]
        if missing:
            raise TypeError(f"RSI reset needs termination arrays {missing} (use the ReconBody termination)")
        model = env.model
        q_per_world = model.joint_coord_count // self.world_count
        dof_per_world = model.joint_dof_count // self.world_count
        if q_per_world != self.q_per_world:
            raise ValueError(
                f"environment has {q_per_world} joint coordinates per world; reset bound {self.q_per_world}"
            )
        self._model = model
        self._terminated = env.terminated
        self._truncated = env.truncated
        self._timestep = env.timestep
        self._episode_step = env.episode_step
        self._object_position_error = termination.object_position_error
        self._object_orientation_error = termination.object_orientation_error
        table = env.reset_table
        self._object_root_q_ids = table.object_root_q_ids
        self._object_root_position_offsets_w = table.object_root_position_offsets_w
        self._num_objects = int(table.num_objects)
        self.dof_per_world = int(dof_per_world)
        if self.pend_q.shape[0] != self.world_count * self.q_per_world:
            self.pend_q = wp.zeros(self.world_count * self.q_per_world, dtype=wp.float32, device=self.device)
            self.pend_qd = wp.zeros(self.world_count * self.dof_per_world, dtype=wp.float32, device=self.device)
        self.bank.allocate_states(self.q_per_world, self.dof_per_world)

    @property
    def env_bound(self) -> bool:
        return self._model is not None

    def _require_bound(self) -> None:
        if self.q_per_world <= 0:
            raise RuntimeError("RSI reset has not been bound to a scene")

    def _require_env(self) -> None:
        if self._model is None:
            raise RuntimeError("RSI reset hooks need bind_env(env) first")

    # ------------------------------------------------------------------ stock protocol
    def sample(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        """Draw RSI starts for the selected worlds; the global curriculum scale is not used (help is per lane)."""
        del curriculum_voc_scale
        self._require_bound()
        config = self.config
        wp.launch(
            kernels.sample_rsi_reset,
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
                self.frame0_share,
                self.cdf,
                self.top,
                self.bin_lo,
                self.bin_hi,
                self.num_bins,
                self.max_start_frame,
                1.0 / config.band,
                int(config.keyframes),
                self.bank.capacity,
                config.kf_min_fill,
                self.bank.frame,
                self.bank.tag,
                self.bank.write_count,
                self.force_bin,
                self.force_slot,
                self.frame_help,
                config.bin_frames,
                int(config.start_frame_shift),
            ],
            outputs=[
                self.reset_count,
                self.reset_frame,
                self.finger_scale,
                self.steps_since_reset,
                self.applied_voc_scale,
                self.raw_joint_q_offset,
                self.reference_joint_q_offset,
                self.start_bin,
                self.kf_slot,
                self.clean,
                self.active,
                self.help,
                self.tag,
                self.cum_score,
                self.pend_on,
                self.requested,
                self.served,
                self.start_stats,
                self.bank.drop_total,
                self.help_factor,
                self.visit_bin,
            ],
            device=self.device,
        )

    def prepare_explicit(self, reset_mask: wp.array, curriculum_voc_scale: wp.array) -> None:
        """Stock explicit-reset semantics; the lane is not scored, not a capture source and not restored."""
        self._require_bound()
        wp.launch(
            kernels.prepare_explicit_rsi_reset,
            dim=self.world_count,
            inputs=[reset_mask, self.q_per_world, self.config.reset_freeze_steps + 1, curriculum_voc_scale],
            outputs=[
                self.finger_scale,
                self.steps_since_reset,
                self.applied_voc_scale,
                self.raw_joint_q_offset,
                self.reference_joint_q_offset,
                self.start_bin,
                self.kf_slot,
                self.clean,
                self.active,
                self.help,
                self.tag,
                self.cum_score,
                self.pend_on,
                self.force_bin,
                self.bank.drop_total,
            ],
            device=self.device,
        )

    def advance(
        self,
        terminated: wp.array,
        truncated: wp.array,
        timestep: wp.array,
        episode_step: wp.array,
        curriculum_voc_scale: wp.array,
    ) -> None:
        del curriculum_voc_scale
        self._require_bound()
        config = self.config
        if self.frame_help:
            wp.launch(
                kernels.update_frame_help,
                dim=self.world_count,
                inputs=[
                    terminated,
                    truncated,
                    self.active,
                    timestep,
                    self.help_factor,
                    self.top,
                    config.bin_frames,
                    self.num_bins,
                ],
                outputs=[self.help],
                device=self.device,
            )
        wp.launch(
            kernels.advance_rsi_reset,
            dim=self.world_count,
            inputs=[
                terminated,
                truncated,
                config.reset_freeze_steps,
                config.reset_freeze_steps - config.run_in_steps,
                config.reset_freeze_steps - config.voc_decay_steps,
                config.voc_decay_steps,
                config.reset_voc_scale,
                self.q_per_world,
                self.help,
                self.raw_joint_q_offset,
            ],
            outputs=[
                self.steps_since_reset,
                self.applied_voc_scale,
                self.reference_joint_q_offset,
                timestep,
                episode_step,
            ],
            device=self.device,
        )

    def sync_curriculum(self, curriculum_voc_scale: wp.array) -> None:
        """Settled worlds keep their lane help: the global curriculum scale does not drive RSI help."""
        del curriculum_voc_scale
        wp.launch(
            kernels.sync_rsi_voc,
            dim=self.world_count,
            inputs=[self.config.reset_freeze_steps, self.help, self.steps_since_reset],
            outputs=[self.applied_voc_scale],
            device=self.device,
        )

    def compute_context(self, curriculum_voc_scale: wp.array, output: wp.array) -> wp.array:
        del curriculum_voc_scale
        expected = (self.world_count * len(self.context_names),)
        if output.shape != expected or output.dtype != wp.float32:
            raise ValueError(f"RSI reset context must be float32 with shape {expected}")
        wp.launch(
            kernels.publish_rsi_reset_context,
            dim=self.world_count,
            inputs=[self.applied_voc_scale, self.help, self.steps_since_reset, self.config.reset_freeze_steps],
            outputs=[output],
            device=self.device,
        )
        return output

    # ------------------------------------------------------------------ RSI hooks
    def on_transition(self, state: Any) -> None:
        """Score, resolve pending keyframes and capture new ones; inside the step graph, before auto-reset."""
        self._require_env()
        config = self.config
        wp.launch(
            kernels.rsi_transition,
            dim=self.world_count,
            inputs=[
                self._terminated,
                self._truncated,
                self._timestep,
                self._episode_step,
                self._object_position_error,
                self._object_orientation_error,
                state.joint_q,
                state.joint_qd,
                self.q_per_world,
                self.dof_per_world,
                self._object_root_q_ids,
                self._object_root_position_offsets_w,
                self._num_objects,
                config.task_pos_scale,
                int(config.keyframes),
                config.kf_prob,
                config.kf_min_task,
                config.kf_min_stepped,
                config.kf_step_score_min,
                self.max_start_frame,
                config.bin_frames,
                self.num_bins,
                self.bank.capacity,
                config.seed,
                int(config.reset_freeze_steps) if config.pass_after_grace else 0,
            ],
            outputs=[
                self.active,
                self.clean,
                self.start_bin,
                self.tag,
                self.cum_score,
                self.step_count,
                self.pend_on,
                self.pend_frame,
                self.pend_step,
                self.pend_cum,
                self.pend_tag,
                self.pend_q,
                self.pend_qd,
                self.n_iter,
                self.f_iter,
                self.bank.q,
                self.bank.qd,
                self.bank.frame,
                self.bank.tag,
                self.bank.write_count,
                self.bank.commit_total,
                self.bank.drop_total,
                self.visit_bin,
                self.pass_iter,
                self.pass_fail_iter,
            ],
            device=self.device,
        )

    def restore(self, state: Any, reset_mask: wp.array) -> None:
        """Overwrite just-reset keyframe worlds with their bank state and refresh body transforms (FK, all worlds)."""
        if not self.config.keyframes:
            return  # no lane can hold a keyframe slot
        self._require_env()
        wp.launch(
            kernels.restore_keyframe,
            dim=self.world_count,
            inputs=[
                reset_mask,
                self.kf_slot,
                self.start_bin,
                self.bank.capacity,
                self.q_per_world,
                self.dof_per_world,
                self.bank.q,
                self.bank.qd,
                self._object_root_q_ids,
                self._object_root_position_offsets_w,
                self._num_objects,
            ],
            outputs=[state.joint_q, state.joint_qd],
            device=self.device,
        )
        newton.eval_fk(self._model, state.joint_q, state.joint_qd, state)

    def upload_bins(self) -> None:
        """Copy the host tops and start CDF to the device tables read by the captured kernels."""
        self.top.assign(self.bins.top.astype(np.float32))
        self.cdf.assign(self.bins.start_cdf())

    def set_frame0_share(self, share: float) -> None:
        if not (math.isfinite(share) and 0.0 <= share < 1.0):
            raise ValueError(f"RSI frame0_share must be in [0, 1), got {share}")
        self.frame0_share.assign(np.asarray([share], dtype=np.float32))

    def host_update(self, iteration: int) -> dict[str, float]:
        """Once per iteration, outside the graph: fold the device outcome counters into the bins, move bins and
        upload the new tops/CDF. ``iteration`` is the number of completed PPO iterations (the runner loop index)."""
        iteration = int(iteration)
        n = self.n_iter.numpy().astype(np.int64)
        f = self.f_iter.numpy().astype(np.int64)
        requested = self.requested.numpy().astype(np.int64)
        served = self.served.numpy().astype(np.int64)
        starts = self.start_stats.numpy().astype(np.int64)
        passes = self.pass_iter.numpy().astype(np.int64)
        pass_fails = self.pass_fail_iter.numpy().astype(np.int64)
        for array in (self.n_iter, self.f_iter, self.requested, self.served, self.start_stats, self.pass_iter,
                      self.pass_fail_iter):
            array.zero_()
        self.bins.observe(n, f)
        self.bins.observe_pass(passes, pass_fails)
        decision = self.bins.decide(iteration)
        self.upload_bins()
        self._last_iteration = iteration

        bins = self.bins
        write_count = self.bank.write_count.numpy().astype(np.int64)
        fill = np.minimum(write_count, self.bank.capacity)
        detailed = self.stats_interval <= 1 or (iteration + 1) % self.stats_interval == 0
        if detailed:
            self._cached_eligible = self.bank.eligible_counts(self.top)
            self._cached_pending_frac = float(self.pend_on.numpy().mean())
        min_fill = self.config.kf_min_fill
        total_starts = max(int(starts[kernels.START_TOTAL]), 1)
        total_requested = int(requested.sum())
        metrics: dict[str, float] = {
            "rsi/top_mean": float(bins.top.mean()),
            "rsi/top_min": float(bins.top.min()),
            "rsi/top_max": float(bins.top.max()),
            "rsi/bins_zero": float((bins.top == 0).sum()),
            "rsi/floor": float(decision["floor"]),
            "rsi/help_progress": bins.help_progress(),
            "rsi/fail_rate_mean": float(bins.rate.mean()),
            "rsi/fail_rate_max": float(bins.rate.max()),
            "rsi/worst_bin": float(int(bins.rate.argmax())),
            "rsi/episodes_scored": float(n.sum()),
            "rsi/moves_down": float(bins.counts["down"]),
            "rsi/moves_up": float(bins.counts["up"]),
            "rsi/forced": float(bins.counts["forced"]),
            "rsi/decisions": float(bins.counts["decisions"]),
            "rsi/kf_requested": float(total_requested),
            "rsi/kf_served": float(served.sum()),
            "rsi/kf_served_frac": float(served.sum()) / max(total_requested, 1),
            "rsi/kf_committed_total": float(self.bank.commit_total.numpy().sum()),
            "rsi/kf_dropped_total": float(self.bank.drop_total.numpy().sum()),
            "rsi/kf_bins_filled": float((fill >= min_fill).sum()),
            "rsi/kf_bins_eligible": float((self._cached_eligible >= min_fill).sum()),
            "rsi/kf_pending_frac": self._cached_pending_frac,
            "rsi/anchor_frac": float(starts[kernels.START_ANCHOR]) / total_starts,
            "rsi/frame0_start_frac": float(starts[kernels.START_FRAME0]) / total_starts,
            "rsi/keyframe_start_frac": float(starts[kernels.START_KEYFRAME]) / total_starts,
        }
        start = bins.start_probs()
        metrics["rsi/start_entropy"] = float(-(start * np.log(np.maximum(start, 1e-12))).sum() / math.log(self.num_bins))
        metrics["rsi/start_top_prob"] = float(start.max())
        metrics["rsi/start_top_bin"] = float(int(start.argmax()))
        if detailed:
            for b in range(self.num_bins):
                metrics[f"rsi/bin/start_{b:02d}"] = float(start[b])
        if bins.success_mode or bins.params.start_mode == "hazard":
            rate = bins.pass_rate()
            bound = bins.pass_lower_bound()
            metrics.update(
                {
                    "rsi/pass_rate_mean": float(rate.mean()),
                    "rsi/pass_rate_min": float(rate.min()),
                    "rsi/pass_lb_min": float(bound.min()),
                    "rsi/worst_pass_bin": float(int(rate.argmin())),
                    "rsi/pass_samples_min": float(bins.pass_samples().min()),
                    "rsi/pass_through_iter": float(passes.sum() + pass_fails.sum()),
                }
            )
        if detailed:
            for b in range(self.num_bins):
                metrics[f"rsi/bin/top_{b:02d}"] = float(bins.top[b])
                metrics[f"rsi/bin/fail_{b:02d}"] = float(bins.rate[b])
                metrics[f"rsi/bin/kf_eligible_{b:02d}"] = float(self._cached_eligible[b])
                if bins.success_mode or bins.params.start_mode == "hazard":
                    metrics[f"rsi/bin/pass_{b:02d}"] = float(rate[b])
                    metrics[f"rsi/bin/pass_lb_{b:02d}"] = float(bound[b])
        return metrics

    def help_zero_since(self) -> int | None:
        """Completed-iteration count ``t`` whose ``host_update(t)`` left every start bin at help 0, with no bin above
        0 since; None while any bin still has help. Iterations ``t + 1, t + 2, ...`` (1-based) trained without help.
        Recomputed from the bins (``RSIBins.zero_since``), so a resume from the sidecar restores it exactly."""
        return self.bins.zero_since()

    # ------------------------------------------------------------------ save / resume
    def _sidecar_metadata(self, iteration: int) -> dict[str, str]:
        return {
            "format": SIDECAR_FORMAT,
            "version": str(SIDECAR_VERSION),
            "num_frames": str(self.num_frames),
            "num_bins": str(self.num_bins),
            "bin_frames": str(self.config.bin_frames),
            "capacity": str(self.bank.capacity),
            "q_per_world": str(self.q_per_world),
            "dof_per_world": str(self.dof_per_world),
            "iteration": str(int(iteration)),
        }

    def save_state(self, path: str | os.PathLike) -> None:
        """Write ``rsi_{iteration}.safetensors``: bins, bank and the outcome counts not yet folded into the bins."""
        from safetensors.numpy import save_file  # noqa: PLC0415 - training-only dependency

        self._require_env()
        path = Path(path)
        match = _SIDECAR_NAME.match(path.name)
        iteration = int(match.group(1)) if match else self._last_iteration + 1
        tensors = {f"bins.{key}": np.ascontiguousarray(value) for key, value in self.bins.state_dict().items()}
        tensors.update({f"bank.{key}": np.ascontiguousarray(value) for key, value in self.bank.state_dict().items()})
        tensors["pending.n_iter"] = self.n_iter.numpy()
        tensors["pending.f_iter"] = self.f_iter.numpy()
        tensors["pending.pass_iter"] = self.pass_iter.numpy()
        tensors["pending.pass_fail_iter"] = self.pass_fail_iter.numpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        save_file(tensors, str(temporary), metadata=self._sidecar_metadata(iteration))
        os.replace(temporary, path)

    def load_state(self, path: str | os.PathLike) -> None:
        """Restore bins and bank from a sidecar written by ``save_state`` for the same clip and scene."""
        from safetensors import safe_open  # noqa: PLC0415 - training-only dependency
        from safetensors.numpy import load_file  # noqa: PLC0415

        self._require_env()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"RSI sidecar {path} does not exist")
        with safe_open(str(path), framework="numpy") as handle:
            metadata = dict(handle.metadata() or {})
        match = _SIDECAR_NAME.match(path.name)
        expected = self._sidecar_metadata(int(match.group(1)) if match else int(metadata.get("iteration", -1)))
        mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatched:
            raise ValueError(f"RSI sidecar {path} does not match this run (stored, expected): {mismatched}")
        tensors = load_file(str(path))
        self.bins.load_state_dict({key[5:]: value for key, value in tensors.items() if key.startswith("bins.")})
        self.bank.load_state_dict({key[5:]: value for key, value in tensors.items() if key.startswith("bank.")})
        for name, array in (("pending.n_iter", self.n_iter), ("pending.f_iter", self.f_iter)):
            value = tensors.get(name)
            if value is None or value.shape != (self.num_bins,):
                raise ValueError(f"RSI sidecar {path} lacks {name} with shape ({self.num_bins},)")
            array.assign(value.astype(np.int32))
        for name, array in (("pending.pass_iter", self.pass_iter), ("pending.pass_fail_iter", self.pass_fail_iter)):
            value = tensors.get(name)  # absent in sidecars written before the success mode
            if value is None:
                array.zero_()
            elif value.shape != (self.num_bins,):
                raise ValueError(f"RSI sidecar {path} has {name} with shape {value.shape}; expected ({self.num_bins},)")
            else:
                array.assign(value.astype(np.int32))
        for array in (self.requested, self.served, self.start_stats):
            array.zero_()
        self._last_iteration = int(metadata["iteration"]) - 1
        self.upload_bins()

    # ------------------------------------------------------------------ test hooks
    def debug_force_keyframe(self, world: int, bin_index: int, slot: int) -> None:
        """Make the next ``sample`` of ``world`` start from bank slot ``(bin_index, slot)`` (tests only)."""
        if not (0 <= world < self.world_count and 0 < bin_index < self.num_bins and 0 <= slot < self.bank.capacity):
            raise IndexError(f"cannot force world {world} onto bank slot ({bin_index}, {slot})")
        force_bin = self.force_bin.numpy()
        force_slot = self.force_slot.numpy()
        force_bin[world] = bin_index
        force_slot[world] = slot
        self.force_bin.assign(force_bin)
        self.force_slot.assign(force_slot)
