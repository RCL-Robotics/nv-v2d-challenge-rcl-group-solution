# SPDX-License-Identifier: Apache-2.0
"""Host-side RSI v3 start-bin logic: a float64 NumPy port of Coordinates_hand ``curriculum.py:367-551``.

One instance owns the per-bin help tops, the failure-rate EMAs and the decision windows of one clip. It runs
on the host once per PPO iteration (``observe`` then ``decide``); the device side only reads the resulting
``top`` and ``start_cdf`` arrays (see :mod:`flash_chord.lifecycle.rsi_reset`).

Name map to CH: ``kp_init`` -> ``top0``, ``rsi_n``/``rsi_f`` -> ``win_n``/``win_f``, ``wait_epochs`` ->
``wait_iterations``, ``rsi_*`` -> the bare name. Only fixed-width bins (``rsi_bin_frames > 0``) are supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

COUNT_KEYS = ("down", "up", "forced", "decisions")


@dataclass(frozen=True)
class RSIBinParams:
    """The RSI fields the host logic needs (a subset of ``RSIReconBodyResetConfig``)."""

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
    start_update: int = 0

    def __post_init__(self) -> None:
        # Same conditions as CH _rsi_init (curriculum.py:378-385); the ones that need the clip length are in RSIBins.
        ints = ("bin_frames", "min_count", "min_gap", "wait_iterations", "deadline", "zero_ramp", "min_horizon",
                "start_update")
        for name in ints:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"RSI {name} must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        for field in fields(self):
            if field.name not in ints:
                value = float(getattr(self, field.name))
                if not math.isfinite(value):
                    raise ValueError(f"RSI {field.name} must be finite, got {value}")
                object.__setattr__(self, field.name, value)
        if self.bin_frames <= 0:
            raise ValueError("RSI bin_frames must be positive (legacy fixed-count bins are not supported)")
        if not (
            0.0 <= self.uniform_mix <= 1.0
            and 0.0 <= self.credit_gamma < 1.0
            and 0.0 < self.fail_alpha <= 1.0
            and 0.0 <= self.success_down < self.success_up <= 1.0
            and 0.0 < self.step < 1.0
            and self.band >= 1.0
            and self.min_count > 0
            and self.min_gap >= 0
            and self.wait_iterations >= 0
            and self.zero_below > 0.0
            and self.deadline > 0
            and self.zero_ramp > 0
            and self.min_horizon > 0
            and self.top0 > 0.0
            and self.start_update >= 0
        ):
            raise ValueError(f"invalid RSI assistance configuration: {self}")

    @classmethod
    def from_object(cls, source: Any) -> RSIBinParams:
        """Pick the host fields off any object carrying them (e.g. the reset config)."""
        return cls(**{field.name: getattr(source, field.name) for field in fields(cls)})


def bin_geometry(num_frames: int, bin_frames: int, min_horizon: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Start-bin bounds ``[lo, hi)`` and the last startable frame ``F_max = num_frames - 1 - min_horizon``.

    Equals CH ``startable = T - min_horizon`` with ``rsi_bins = ceil(startable / bin_frames)`` and
    ``hi = min(lo + bin_frames, startable)`` (curriculum.py:373-394).
    """
    if not 0 < min_horizon < num_frames:
        raise ValueError(f"RSI min_horizon must be in (0, {num_frames}), got {min_horizon}")
    startable = num_frames - min_horizon
    num_bins = math.ceil(startable / bin_frames)
    if num_bins <= 1:
        raise ValueError(
            f"RSI needs more than one start bin; {num_frames} frames, min_horizon {min_horizon} and "
            f"bin_frames {bin_frames} give {num_bins}"
        )
    lo = np.arange(num_bins, dtype=np.int64) * bin_frames
    hi = np.minimum(lo + bin_frames, startable)
    if not bool((hi > lo).all()):
        raise ValueError("every RSI start bin needs at least one start frame")
    return lo.astype(np.int32), hi.astype(np.int32), startable - 1


class RSIBins:
    """Per-bin help tops, failure EMAs and decision windows of one clip (float64 host state)."""

    def __init__(self, num_frames: int, params: RSIBinParams) -> None:
        self.params = params
        self.num_frames = int(num_frames)
        self.bin_lo, self.bin_hi, self.max_start_frame = bin_geometry(
            self.num_frames, params.bin_frames, params.min_horizon
        )
        self.num_bins = int(self.bin_lo.shape[0])
        self.top = np.full(self.num_bins, params.top0, dtype=np.float64)
        self.rate = np.zeros(self.num_bins, dtype=np.float64)
        self.win_n = np.zeros(self.num_bins, dtype=np.float64)
        self.win_f = np.zeros(self.num_bins, dtype=np.float64)
        self.last_move = np.full(self.num_bins, params.start_update, dtype=np.int64)
        self.counts = dict.fromkeys(COUNT_KEYS, 0)

    def bin_of(self, frame: np.ndarray | int) -> np.ndarray:
        """``min(frame // bin_frames, B - 1)``, the same bins as CH ``bucketize(bin_hi, right=True)``."""
        return np.minimum(np.asarray(frame) // self.params.bin_frames, self.num_bins - 1)

    def floor(self, iteration: int) -> float:
        """Help ceiling every bin must sit at or below at completed iteration ``iteration`` (CH _rsi_floor)."""
        p = self.params
        t = int(iteration) - p.start_update
        if t < p.deadline:
            return p.top0 * (p.zero_below / p.top0) ** (t / p.deadline)
        return p.zero_below * max(0.0, 1.0 - (t - p.deadline) / p.zero_ramp)

    def start_probs(self) -> np.ndarray:
        """Bin start probabilities: uniform mix plus backward-credited failure rates (CH _rsi_start_probs)."""
        p = self.params
        credit = np.zeros(self.num_bins, dtype=np.float64)
        acc = 0.0
        for b in range(self.num_bins - 1, -1, -1):
            acc = float(self.rate[b]) + p.credit_gamma * acc
            credit[b] = acc
        total = float(credit.sum())
        if total > 0:
            adaptive = credit / total
        else:
            adaptive = np.full(self.num_bins, 1.0 / self.num_bins, dtype=np.float64)
        return p.uniform_mix / self.num_bins + (1.0 - p.uniform_mix) * adaptive

    def start_cdf(self) -> np.ndarray:
        """float32 cumulative start distribution with ``cdf[-1] == 1`` exactly (device sampling table)."""
        cdf = np.cumsum(self.start_probs()).astype(np.float32)
        cdf[-1] = 1.0
        return cdf

    def observe(self, n: np.ndarray, f: np.ndarray) -> None:
        """Add one iteration's finished (n) and failed (f) episodes per start bin (CH rsi_on_reset scoring)."""
        n = np.asarray(n, dtype=np.float64).reshape(self.num_bins)
        f = np.asarray(f, dtype=np.float64).reshape(self.num_bins)
        if (n < 0).any() or (f < 0).any() or (f > n).any():
            raise ValueError(f"invalid RSI outcome counts n={n.tolist()} f={f.tolist()}")
        self.win_n += n
        self.win_f += f
        has = n > 0
        if has.any():
            w = 1.0 - (1.0 - self.params.fail_alpha) ** n[has]
            self.rate[has] = (1.0 - w) * self.rate[has] + w * (f[has] / n[has])

    def decide(self, iteration: int) -> dict[str, Any]:
        """Success-driven bin moves, then the deadline floor (CH _rsi_set_curriculum, curriculum.py:509-551)."""
        p = self.params
        t = int(iteration)
        before = self.top.copy()
        floor = self.floor(t)
        if t - p.start_update >= p.wait_iterations:
            for b in range(self.num_bins):
                if self.win_n[b] < p.min_count or t - int(self.last_move[b]) < p.min_gap:
                    continue
                success = 1.0 - float(self.win_f[b] / self.win_n[b])
                self.win_n[b] = 0.0
                self.win_f[b] = 0.0
                self.counts["decisions"] += 1
                top = float(self.top[b])
                if success >= p.success_up and top > 0:
                    new = top * p.step
                    self.top[b] = 0.0 if new < p.zero_below else new
                    self.counts["down"] += 1
                    self.last_move[b] = t
                elif success < p.success_down and top < p.top0:
                    new = p.zero_below if top == 0 else min(top / p.step, p.top0)
                    if new <= floor:
                        self.top[b] = new
                        self.counts["up"] += 1
                        self.last_move[b] = t
        # Deadline floor: acts only on bins a full step behind it (never pre-empts moves that keep pace).
        behind = self.top > (floor / p.step) if floor > 0 else self.top > 0
        forced = int(behind.sum())
        if forced:
            self.top[behind] = max(floor, 0.0)
            self.counts["forced"] += 1
            self.last_move[behind] = t
        return {
            "zero": bool((self.top == 0).all()),
            "moved": bool((self.top != before).any()),
            "floor": float(floor),
            "forced_bins": forced,
        }

    def zero_since(self) -> int | None:
        """Iteration ``t`` of the ``decide(t)`` call since which every bin's top has been 0 without a break, or None.

        Recomputed from ``top`` and ``last_move`` (both in the sidecar), so it survives a resume without new state.
        Exact because a bin's top changes only through a recorded move: it reaches 0 through a success step below
        ``zero_below`` or a forced step to a zero floor (both set ``last_move``), and it leaves 0 only through an
        up-move (which sets ``last_move`` too, and leaves the top > 0). So while ``top[b] == 0``, ``last_move[b]`` is
        the iteration bin ``b`` last became 0, and the all-zero streak began at ``max(last_move)``: the last bin to
        reach 0 was > 0 before that decision. ``top0 > 0`` means no bin starts at 0. Coordinates_hand's
        ``num_epoch_since_zero`` (curriculum.py:544-546) instead counts every all-zero update and never resets; the
        two agree after the deadline ramp, where up-moves are impossible (the floor is 0).
        """
        if not bool((self.top == 0).all()):
            return None
        return int(self.last_move.max())

    def help_progress(self) -> float:
        """Share of help left on the log scale its steps move along: 1 at the start, 0 with every bin at zero."""
        p = self.params
        low = float(p.zero_below)
        top = np.maximum(self.top, 0.0)
        if p.top0 <= low:
            return float((top > 0).mean())
        progress = np.log(np.maximum(top, low) / low) / math.log(p.top0 / low)
        return float(np.where(top > 0, np.clip(progress, 0.0, 1.0), 0.0).mean())

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "top": self.top.copy(),
            "rate": self.rate.copy(),
            "win_n": self.win_n.copy(),
            "win_f": self.win_f.copy(),
            "last_move": self.last_move.copy(),
            "counts": np.asarray([self.counts[key] for key in COUNT_KEYS], dtype=np.int64),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        expected = {"top", "rate", "win_n", "win_f", "last_move", "counts"}
        if set(state) != expected:
            raise ValueError(f"RSI bin state has keys {sorted(state)}; expected {sorted(expected)}")
        for name in ("top", "rate", "win_n", "win_f", "last_move"):
            value = np.asarray(state[name])
            if value.shape != (self.num_bins,):
                raise ValueError(f"RSI bin state {name} has shape {value.shape}; expected ({self.num_bins},)")
        counts = np.asarray(state["counts"])
        if counts.shape != (len(COUNT_KEYS),):
            raise ValueError(f"RSI bin state counts has shape {counts.shape}; expected ({len(COUNT_KEYS)},)")
        self.top = np.asarray(state["top"], dtype=np.float64).copy()
        self.rate = np.asarray(state["rate"], dtype=np.float64).copy()
        self.win_n = np.asarray(state["win_n"], dtype=np.float64).copy()
        self.win_f = np.asarray(state["win_f"], dtype=np.float64).copy()
        self.last_move = np.asarray(state["last_move"], dtype=np.int64).copy()
        self.counts = {key: int(value) for key, value in zip(COUNT_KEYS, counts.tolist(), strict=True)}
