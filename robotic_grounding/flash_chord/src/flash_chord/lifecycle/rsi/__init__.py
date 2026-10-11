# SPDX-License-Identifier: Apache-2.0
"""RSI v3 (fixed) reference-state initialization: adaptive start bins with per-bin VOC help and a keyframe bank.

Port of Coordinates_hand ``--curr_schedule rsi --rsi_keyframes --rsi_kf_to_end --rsi_frame0_share p`` to the
FlashCHORD reset protocol. The reset policy itself is :class:`flash_chord.lifecycle.rsi_reset.RSIReconBodyReset`
(exported here lazily, because that module imports this package).

Integration (``docs/coord_impl_spec.md`` section 5): an environment detects the extra hooks with
``isinstance(reset_policy, RSIHooks)`` and calls ``bind_env`` once after construction, ``on_transition`` inside the
step graph after the transition and before auto-reset, ``restore`` right after every ``BaseEnv.reset``; the
training loop calls ``host_update`` once per iteration and ``save_state`` / ``load_state`` next to checkpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from flash_chord.lifecycle.rsi.bins import RSIBinParams, RSIBins

if TYPE_CHECKING:
    from flash_chord.lifecycle.rsi.bank import KeyframeBank
    from flash_chord.lifecycle.rsi_reset import RSIReconBodyReset, RSIReconBodyResetConfig


@runtime_checkable
class RSIHooks(Protocol):
    """Extra hooks of an RSI reset policy beyond the stock ``ResetPolicy`` protocol."""

    def bind_env(self, env: Any) -> None:
        """Bind state buffers of a constructed RL environment; call once, before ``capture_step``."""
        ...

    def on_transition(self, state: Any) -> None:
        """Graph-capturable: score lanes, resolve/commit pending keyframes, capture new ones (before auto-reset)."""
        ...

    def restore(self, state: Any, reset_mask: Any) -> None:
        """Graph-capturable: overwrite just-reset keyframe worlds with their stored state, then refresh FK."""
        ...

    def host_update(self, iteration: int) -> dict[str, float]:
        """Host, once per iteration outside the graph: read outcomes, move bins, upload tops/CDF, return metrics."""
        ...

    def save_state(self, path: Any) -> None:
        """Write the bins and the keyframe bank to a sidecar file."""
        ...

    def load_state(self, path: Any) -> None:
        """Restore the bins and the keyframe bank from a sidecar file."""
        ...


_LAZY = {
    "KeyframeBank": ("flash_chord.lifecycle.rsi.bank", "KeyframeBank"),
    "RSIReconBodyReset": ("flash_chord.lifecycle.rsi_reset", "RSIReconBodyReset"),
    "RSIReconBodyResetConfig": ("flash_chord.lifecycle.rsi_reset", "RSIReconBodyResetConfig"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib  # noqa: PLC0415

        module_name, attribute = _LAZY[name]
        return getattr(importlib.import_module(module_name), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "KeyframeBank",
    "RSIBinParams",
    "RSIBins",
    "RSIHooks",
    "RSIReconBodyReset",
    "RSIReconBodyResetConfig",
]
