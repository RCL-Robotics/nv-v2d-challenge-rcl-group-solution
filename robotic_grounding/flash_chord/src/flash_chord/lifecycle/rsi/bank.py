# SPDX-License-Identifier: Apache-2.0
"""Device keyframe bank: per start bin, a ring buffer of reached simulator states (CH keyframes.py).

A slot holds one world's full ``joint_q`` / ``joint_qd`` (robot root and joints, object free joint and object
articulation coordinates), the reference frame it was reached at and its help tag (the top of the start bin
of the lane that reached it). Object root positions are stored without the source world's object offset.
The kernels that write (commit) and read (sample, restore) the bank live in :mod:`.kernels`.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from flash_chord.lifecycle.rsi.kernels import count_eligible_slots


class KeyframeBank:
    """Ring buffers ``[num_bins, capacity]`` of simulator states with their reference frame and help tag."""

    def __init__(self, num_bins: int, capacity: int, device=None) -> None:
        if num_bins <= 0 or capacity <= 0:
            raise ValueError(f"keyframe bank needs positive bins and capacity, got {num_bins} and {capacity}")
        self.num_bins = int(num_bins)
        self.capacity = int(capacity)
        self.device = wp.get_device(device)
        slots = self.num_bins * self.capacity
        self.frame = wp.zeros(slots, dtype=wp.int32, device=self.device)
        self.tag = wp.zeros(slots, dtype=wp.float32, device=self.device)
        self.write_count = wp.zeros(self.num_bins, dtype=wp.int32, device=self.device)
        self.commit_total = wp.zeros(self.num_bins, dtype=wp.int32, device=self.device)
        self.drop_total = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._eligible = wp.zeros(self.num_bins, dtype=wp.int32, device=self.device)
        # Per-slot simulator states; sized once the environment's per-world coordinate counts are known.
        self.q_per_world = 0
        self.dof_per_world = 0
        self.q = wp.zeros(0, dtype=wp.float32, device=self.device)
        self.qd = wp.zeros(0, dtype=wp.float32, device=self.device)

    @property
    def allocated(self) -> bool:
        return self.q_per_world > 0

    def allocate_states(self, q_per_world: int, dof_per_world: int) -> None:
        """Allocate the state slots (about 40 bins x 256 slots x 112 floats = 4.6 MB for a 1003-frame clip)."""
        if q_per_world <= 0 or dof_per_world <= 0:
            raise ValueError(f"keyframe states need positive sizes, got q={q_per_world} qd={dof_per_world}")
        if self.allocated:
            if (q_per_world, dof_per_world) != (self.q_per_world, self.dof_per_world):
                raise ValueError(
                    f"keyframe bank holds q={self.q_per_world}/qd={self.dof_per_world} states; "
                    f"cannot rebind to q={q_per_world}/qd={dof_per_world}"
                )
            return
        slots = self.num_bins * self.capacity
        self.q_per_world = int(q_per_world)
        self.dof_per_world = int(dof_per_world)
        self.q = wp.zeros(slots * self.q_per_world, dtype=wp.float32, device=self.device)
        self.qd = wp.zeros(slots * self.dof_per_world, dtype=wp.float32, device=self.device)

    def fill(self) -> np.ndarray:
        """Stored states per bin, ``min(write_count, capacity)`` (host sync)."""
        return np.minimum(self.write_count.numpy().astype(np.int64), self.capacity)

    def eligible_counts(self, top: wp.array) -> np.ndarray:
        """Per bin, stored states whose tag is <= the bin's current top (host sync)."""
        wp.launch(
            count_eligible_slots,
            dim=self.num_bins,
            inputs=[top, self.tag, self.write_count, self.capacity],
            outputs=[self._eligible],
            device=self.device,
        )
        return self._eligible.numpy().astype(np.int64)

    def debug_write(self, bin_index: int, slot: int, q: np.ndarray, qd: np.ndarray, frame: int, tag: float) -> None:
        """Test hook: write one state into ``(bin_index, slot)`` and count the slot as filled.

        ``q`` must already be in the bank convention (object root positions without the world's object offset).
        """
        if not self.allocated:
            raise RuntimeError("keyframe bank states are not allocated (bind the reset to an environment first)")
        if not (0 <= bin_index < self.num_bins and 0 <= slot < self.capacity):
            raise IndexError(f"bank slot ({bin_index}, {slot}) is outside {self.num_bins} x {self.capacity}")
        q = np.asarray(q, dtype=np.float32).reshape(self.q_per_world)
        qd = np.asarray(qd, dtype=np.float32).reshape(self.dof_per_world)
        index = bin_index * self.capacity + slot
        bank_q = self.q.numpy()
        bank_qd = self.qd.numpy()
        bank_q[index * self.q_per_world : (index + 1) * self.q_per_world] = q
        bank_qd[index * self.dof_per_world : (index + 1) * self.dof_per_world] = qd
        frames = self.frame.numpy()
        tags = self.tag.numpy()
        frames[index] = int(frame)
        tags[index] = float(tag)
        counts = self.write_count.numpy()
        counts[bin_index] = max(int(counts[bin_index]), slot + 1)
        self.q.assign(bank_q)
        self.qd.assign(bank_qd)
        self.frame.assign(frames)
        self.tag.assign(tags)
        self.write_count.assign(counts)

    def state_dict(self) -> dict[str, np.ndarray]:
        if not self.allocated:
            raise RuntimeError("keyframe bank states are not allocated (bind the reset to an environment first)")
        shape_q = (self.num_bins, self.capacity, self.q_per_world)
        shape_qd = (self.num_bins, self.capacity, self.dof_per_world)
        return {
            "q": self.q.numpy().reshape(shape_q),
            "qd": self.qd.numpy().reshape(shape_qd),
            "frame": self.frame.numpy().reshape(self.num_bins, self.capacity),
            "tag": self.tag.numpy().reshape(self.num_bins, self.capacity),
            "write_count": self.write_count.numpy(),
            "commit_total": self.commit_total.numpy(),
            "drop_total": self.drop_total.numpy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        if not self.allocated:
            raise RuntimeError("keyframe bank states are not allocated (bind the reset to an environment first)")
        expected = {
            "q": ((self.num_bins, self.capacity, self.q_per_world), np.float32),
            "qd": ((self.num_bins, self.capacity, self.dof_per_world), np.float32),
            "frame": ((self.num_bins, self.capacity), np.int32),
            "tag": ((self.num_bins, self.capacity), np.float32),
            "write_count": ((self.num_bins,), np.int32),
            "commit_total": ((self.num_bins,), np.int32),
            "drop_total": ((1,), np.int32),
        }
        if set(state) != set(expected):
            raise ValueError(f"keyframe bank state has keys {sorted(state)}; expected {sorted(expected)}")
        for name, (shape, dtype) in expected.items():
            value = np.asarray(state[name])
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"keyframe bank state {name} is {value.dtype}{value.shape}; expected {np.dtype(dtype)}{shape}"
                )
        self.q.assign(np.ascontiguousarray(state["q"]).reshape(-1))
        self.qd.assign(np.ascontiguousarray(state["qd"]).reshape(-1))
        self.frame.assign(np.ascontiguousarray(state["frame"]).reshape(-1))
        self.tag.assign(np.ascontiguousarray(state["tag"]).reshape(-1))
        self.write_count.assign(state["write_count"])
        self.commit_total.assign(state["commit_total"])
        self.drop_total.assign(state["drop_total"])
