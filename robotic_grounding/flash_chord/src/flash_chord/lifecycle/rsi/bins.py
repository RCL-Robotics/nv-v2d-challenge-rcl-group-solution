# SPDX-License-Identifier: Apache-2.0
"""Host-side RSI v3 start-bin logic: a float64 NumPy port of Coordinates_hand ``curriculum.py:367-551``.

One instance owns the per-bin help tops, the failure-rate EMAs and the decision windows of one clip. It runs
on the host once per PPO iteration (``observe`` then ``decide``); the device side only reads the resulting
``top`` and ``start_cdf`` arrays (see :mod:`flash_chord.lifecycle.rsi_reset`).

Name map to CH: ``kp_init`` -> ``top0``, ``rsi_n``/``rsi_f`` -> ``win_n``/``win_f``, ``wait_epochs`` ->
``wait_iterations``, ``rsi_*`` -> the bare name. Only fixed-width bins (``rsi_bin_frames > 0``) are supported.

Dev ``help_mode="success"`` (help force fix) replaces the iteration-driven help schedule (CH deadline floor and the
start-bin success windows) with a success-driven, monotone one:

* pass-through outcomes per bin (BeyondMimic attribution: a failure counts in the bin where the episode terminated,
  every bin an episode leaves or finishes in counts a pass; see ``kernels.rsi_transition``), folded per iteration
  into exponential moving averages of the pass and fail counts with time constant ``help_ema_tau`` iterations;
* the pass rate of a bin is ``P / (P + F)``, judged by its Wilson lower bound over the ``(P + F) / alpha`` samples in
  the EMA window, and only once that window holds ``help_min_samples``; a bin also looks ``help_lookahead_bins``
  ahead (the smaller of its own bound and the mean weighted ``help_lookahead_decay ** i``), so help stays in front
  of a later failure point;
* every ``help_update_interval`` iterations each bin's top moves towards the target ``top0 * (zero_below / top0) **
  p`` of its pass rate (``p`` runs from 0 at ``help_success_lo`` to 1 at ``help_success_hi``, zero help beyond) as a
  ratchet: ``top = min(top, max(top * step, target))`` - never up, at most one ``step`` factor per update;
* a final cutoff (``help_final_iteration > 0``): from ``help_final_fraction - help_final_ramp_fraction`` of it the
  tops are capped by a log-linear ceiling from ``top0`` to ``zero_below``, and zero from ``help_final_fraction``, so
  the policy trains its last iterations without help, as the evaluator runs it.

Dev ``start_mode="hazard"`` (hard-start sampling) replaces the CH start-bin weights (backward-credited failure EMA of
"start in bin b, fail before the clip end", which saturates at 1 for every bin before a failure point and so degrades
to uniform) with the pass-through hazard ``F / (P + F)`` of each bin (the success-mode EMAs above; a bin below
``help_min_samples`` takes the mean of the others), averaged over it and the next ``start_lookahead_bins`` - 1 bins
with weights
``start_lookahead_decay ** i`` (starts land in front of a failure point), turned into ``softmax(difficulty /
start_temperature)``, mixed with ``uniform_mix`` uniform, and capped at ``start_cap`` per bin (the excess goes to the
other bins in proportion). ``start_mode="credit"`` keeps the CH weights. In both modes the result is the distribution of
the non-anchor starts (``frame0_share`` of the resets start at frame 0 before it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

COUNT_KEYS = ("down", "up", "forced", "decisions")
HELP_MODES = ("deadline", "success")
START_MODES = ("credit", "hazard")


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
    # Dev help force fix (module docstring); "deadline" keeps the CH schedule above unchanged.
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
    # Dev hard-start sampling (module docstring); "credit" keeps the CH start weights unchanged.
    start_mode: str = "credit"
    start_temperature: float = 0.1
    start_lookahead_bins: int = 3
    start_lookahead_decay: float = 0.7
    start_cap: float = 1.0

    def __post_init__(self) -> None:
        # Same conditions as CH _rsi_init (curriculum.py:378-385); the ones that need the clip length are in RSIBins.
        ints = ("bin_frames", "min_count", "min_gap", "wait_iterations", "deadline", "zero_ramp", "min_horizon",
                "start_update", "help_lookahead_bins", "help_update_interval", "help_final_iteration",
                "start_lookahead_bins")
        strings = ("help_mode", "start_mode")
        for name in ints:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"RSI {name} must be an integer, got {value!r}")
            object.__setattr__(self, name, int(value))
        for field in fields(self):
            if field.name not in ints and field.name not in strings:
                value = float(getattr(self, field.name))
                if not math.isfinite(value):
                    raise ValueError(f"RSI {field.name} must be finite, got {value}")
                object.__setattr__(self, field.name, value)
        if self.help_mode not in HELP_MODES:
            raise ValueError(f"RSI help_mode must be one of {HELP_MODES}, got {self.help_mode!r}")
        if not (
            self.help_ema_tau >= 1.0
            and self.help_min_samples >= 0.0
            and self.help_wilson_z >= 0.0
            and self.help_lookahead_bins >= 1
            and 0.0 <= self.help_lookahead_decay <= 1.0
            and 0.0 <= self.help_success_lo < self.help_success_hi <= 1.0
            and self.help_update_interval >= 1
            and self.help_final_iteration >= 0
            and 0.0 < self.help_final_fraction <= 1.0
            and 0.0 <= self.help_final_ramp_fraction < self.help_final_fraction
        ):
            raise ValueError(f"invalid RSI success-driven help configuration: {self}")
        if self.start_mode not in START_MODES:
            raise ValueError(f"RSI start_mode must be one of {START_MODES}, got {self.start_mode!r}")
        if not (
            self.start_temperature > 0.0
            and self.start_lookahead_bins >= 1
            and 0.0 <= self.start_lookahead_decay <= 1.0
            and 0.0 < self.start_cap <= 1.0
        ):
            raise ValueError(f"invalid RSI hard-start sampling configuration: {self}")
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


def _cap(probs: np.ndarray, cap: float) -> np.ndarray:
    """Clip each entry at ``cap`` and hand the excess to the uncapped entries in proportion (sum stays 1); a cap below
    ``1 / len`` cannot hold and is raised to it."""
    probs = np.asarray(probs, dtype=np.float64).copy()
    cap = max(float(cap), 1.0 / probs.size)
    for _ in range(probs.size):
        over = probs > cap + 1e-15
        if not over.any():
            break
        excess = float((probs[over] - cap).sum())
        probs[over] = cap
        free = probs < cap - 1e-15
        if not free.any():
            break
        probs[free] += excess * probs[free] / probs[free].sum()
    return probs / probs.sum()


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
        # Success mode: per-iteration EMAs of the pass-through pass and fail counts (module docstring).
        self.pass_ema = np.zeros(self.num_bins, dtype=np.float64)
        self.fail_ema = np.zeros(self.num_bins, dtype=np.float64)

    @property
    def success_mode(self) -> bool:
        return self.params.help_mode == "success"

    @property
    def ema_alpha(self) -> float:
        """Per-iteration EMA weight of the success mode, ``1 - exp(-1 / help_ema_tau)``."""
        return 1.0 - math.exp(-1.0 / self.params.help_ema_tau)

    def bin_of(self, frame: np.ndarray | int) -> np.ndarray:
        """``min(frame // bin_frames, B - 1)``, the same bins as CH ``bucketize(bin_hi, right=True)``."""
        return np.minimum(np.asarray(frame) // self.params.bin_frames, self.num_bins - 1)

    def floor(self, iteration: int) -> float:
        """Help ceiling every bin must sit at or below at completed iteration ``iteration`` (CH _rsi_floor).

        In the success mode this is the final-cutoff ceiling (``top0`` before its ramp, 0 after it)."""
        p = self.params
        if self.success_mode:
            return self.final_ceiling(iteration)
        t = int(iteration) - p.start_update
        if t < p.deadline:
            return p.top0 * (p.zero_below / p.top0) ** (t / p.deadline)
        return p.zero_below * max(0.0, 1.0 - (t - p.deadline) / p.zero_ramp)

    def final_window(self) -> tuple[int, int] | None:
        """Success mode: ``(ramp start, cutoff)`` iterations of the final help cutoff, or None without one."""
        p = self.params
        if p.help_final_iteration <= 0:
            return None
        end = int(round(p.help_final_fraction * p.help_final_iteration))
        start = int(round((p.help_final_fraction - p.help_final_ramp_fraction) * p.help_final_iteration))
        return min(start, end), end

    def final_ceiling(self, iteration: int) -> float:
        """Success mode: ``top0`` before the final ramp, log-linear to ``zero_below`` across it, 0 from the cutoff."""
        p = self.params
        window = self.final_window()
        if window is None:
            return float(p.top0)
        start, end = window
        t = int(iteration)
        if t >= end:
            return 0.0
        if t < start:
            return float(p.top0)
        return float(p.top0 * (p.zero_below / p.top0) ** ((t - start) / max(end - start, 1)))

    def observe_pass(self, passes: np.ndarray, fails: np.ndarray) -> None:
        """Fold one iteration's pass-through passes and fails per bin into the success-mode EMAs."""
        passes = np.asarray(passes, dtype=np.float64).reshape(self.num_bins)
        fails = np.asarray(fails, dtype=np.float64).reshape(self.num_bins)
        if (passes < 0).any() or (fails < 0).any():
            raise ValueError(f"invalid RSI pass-through counts passes={passes.tolist()} fails={fails.tolist()}")
        alpha = self.ema_alpha
        self.pass_ema = (1.0 - alpha) * self.pass_ema + alpha * passes
        self.fail_ema = (1.0 - alpha) * self.fail_ema + alpha * fails

    def pass_samples(self) -> np.ndarray:
        """Effective number of pass-through outcomes in each bin's EMA window, ``(P + F) / alpha``."""
        return (self.pass_ema + self.fail_ema) / self.ema_alpha

    def pass_rate(self) -> np.ndarray:
        """Point estimate ``P / (P + F)`` per bin (0 for a bin without outcomes)."""
        total = self.pass_ema + self.fail_ema
        return np.divide(self.pass_ema, total, out=np.zeros(self.num_bins, dtype=np.float64), where=total > 0)

    def pass_lower_bound(self) -> np.ndarray:
        """Wilson lower bound of each bin's pass rate over its effective samples; 0 below ``help_min_samples``."""
        p = self.params
        n = self.pass_samples()
        s = self.pass_rate()
        z2 = p.help_wilson_z**2
        safe = np.maximum(n, 1e-12)
        centre = s + z2 / (2.0 * safe)
        spread = p.help_wilson_z * np.sqrt(np.maximum(s * (1.0 - s) / safe + z2 / (4.0 * safe * safe), 0.0))
        bound = np.clip((centre - spread) / (1.0 + z2 / safe), 0.0, 1.0)
        return np.where((n >= p.help_min_samples) & (n > 0), bound, 0.0)

    def lookahead_pass(self) -> np.ndarray:
        """Per bin: the smaller of its own pass lower bound and the decay-weighted mean of the bounds of it and the
        next bins (truncated at the end), so a hard bin keeps its own low rate and shields the bins before it."""
        p = self.params
        bound = self.pass_lower_bound()
        out = np.zeros(self.num_bins, dtype=np.float64)
        for b in range(self.num_bins):
            weights = np.asarray([p.help_lookahead_decay**i for i in range(min(p.help_lookahead_bins,
                                                                                  self.num_bins - b))])
            mean = float((weights * bound[b : b + weights.size]).sum() / weights.sum())
            out[b] = min(float(bound[b]), mean)
        return out

    def help_target(self) -> np.ndarray:
        """Success-mode target top per bin: ``top0 * (zero_below / top0) ** p``, 0 once ``p`` reaches 1."""
        p = self.params
        progress = np.clip((self.lookahead_pass() - p.help_success_lo) / (p.help_success_hi - p.help_success_lo),
                           0.0, 1.0)
        target = p.top0 * (p.zero_below / p.top0) ** progress
        return np.where(progress >= 1.0, 0.0, target)

    def start_difficulty(self) -> np.ndarray:
        """Hazard mode: pass-through hazard per bin, lookahead-averaged. A bin below ``help_min_samples`` takes the
        mean hazard of the bins above it (0 while none is): neutral, so a bin whose episodes end before they are
        counted is neither preferred nor starved of starts."""
        p = self.params
        total = self.pass_ema + self.fail_ema
        hazard = np.divide(self.fail_ema, total, out=np.zeros(self.num_bins, dtype=np.float64), where=total > 0)
        known = self.pass_samples() >= max(p.help_min_samples, 1e-9)
        hazard = np.where(known, hazard, float(hazard[known].mean()) if known.any() else 0.0)
        out = np.zeros(self.num_bins, dtype=np.float64)
        for b in range(self.num_bins):
            weights = p.start_lookahead_decay ** np.arange(min(p.start_lookahead_bins, self.num_bins - b))
            out[b] = float((weights * hazard[b : b + weights.size]).sum() / weights.sum())
        return out

    def start_probs(self) -> np.ndarray:
        """Bin start probabilities: uniform mix plus backward-credited failure rates (CH _rsi_start_probs), or in the
        hazard mode uniform mix plus ``softmax(difficulty / start_temperature)``, capped per bin."""
        p = self.params
        if p.start_mode == "hazard":
            difficulty = self.start_difficulty()
            z = np.exp((difficulty - difficulty.max()) / p.start_temperature)
            probs = p.uniform_mix / self.num_bins + (1.0 - p.uniform_mix) * z / z.sum()
            return _cap(probs, p.start_cap)
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
        if self.success_mode:
            return self._decide_success(iteration)
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

    def _decide_success(self, iteration: int) -> dict[str, Any]:
        """Success mode: the monotone, rate-limited ratchet towards each bin's pass-rate target, then the final
        cutoff ceiling. A top never rises."""
        p = self.params
        t = int(iteration)
        before = self.top.copy()
        elapsed = t - p.start_update
        if elapsed >= p.wait_iterations and elapsed % p.help_update_interval == 0:
            target = self.help_target()
            for b in range(self.num_bins):
                top = float(self.top[b])
                if top <= 0.0:
                    continue
                self.counts["decisions"] += 1
                new = min(top, max(top * p.step, float(target[b])))
                if new < p.zero_below:
                    new = 0.0
                if new < top:
                    self.top[b] = new
                    self.counts["down"] += 1
                    self.last_move[b] = t
        ceiling = self.final_ceiling(t)
        behind = self.top > ceiling
        forced = int(behind.sum())
        if forced:
            self.top[behind] = ceiling
            self.counts["forced"] += 1
            self.last_move[behind] = t
        return {
            "zero": bool((self.top == 0).all()),
            "moved": bool((self.top != before).any()),
            "floor": float(ceiling),
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
            "pass_ema": self.pass_ema.copy(),
            "fail_ema": self.fail_ema.copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Restore a state; sidecars written before the success mode (no ``pass_ema``/``fail_ema``) load with zero
        pass-through EMAs."""
        legacy = {"top", "rate", "win_n", "win_f", "last_move", "counts"}
        expected = legacy | {"pass_ema", "fail_ema"}
        if set(state) not in (legacy, expected):
            raise ValueError(f"RSI bin state has keys {sorted(state)}; expected {sorted(expected)}")
        names = ("top", "rate", "win_n", "win_f", "last_move") + (("pass_ema", "fail_ema") if "pass_ema" in state
                                                                     else ())
        for name in names:
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
        if "pass_ema" in state:
            self.pass_ema = np.asarray(state["pass_ema"], dtype=np.float64).copy()
            self.fail_ema = np.asarray(state["fail_ema"], dtype=np.float64).copy()
        else:
            self.pass_ema = np.zeros(self.num_bins, dtype=np.float64)
            self.fail_ema = np.zeros(self.num_bins, dtype=np.float64)
