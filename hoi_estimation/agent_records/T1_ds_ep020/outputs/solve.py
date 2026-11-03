#!/usr/bin/env python3
"""Recover a temporally consistent world-space rice-cooker trajectory.

The released baseline is already a strong image-based estimate.  This solver
uses its dynamic rotation, but replaces estimator jitter in the two physical
rest phases, removes depth/tracker jitter from the manipulation trajectory, and
applies one small world-space correction supported by pooled silhouette
evidence.  It is deterministic and uses only the released inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# Boundaries verified from the baseline rotation-speed, the right-hand contact
# signal, and the object's screen-space motion:
#   * frames 0..203: object stationary (the hand arrives around frame 140 but
#     the object remains fixed until the clockwise turn begins around 209);
#   * frames 204..479: turn plus the final placement translation;
#   * frames 480..552: object stationary on the table.
STATIC_PHASES = ((0, 204), (480, 553))
MOVING_START = 204
MOVING_STOP = 480

# Pooled-fit correction in world metres.  The baseline projection is offset by
# approximately +2.1 px in x and -0.9 px in y.  Optimising a constant silhouette
# displacement over the two static clips gives the same correction in both:
# about -8 mm along world x and +7 mm along world y.  Depth is deliberately not
# altered because the baseline was already calibrated against the human/floor.
GLOBAL_XY_CORRECTION = np.array([-0.008, 0.007, 0.0], dtype=np.float64)

# The object translation follows the hand smoothly.  A 21-frame quadratic
# Savitzky-Golay filter removes per-frame depth/tracker jitter while preserving
# the eventual 10 cm placement motion.  Rotation is not filtered: the blue
# side-handle track agrees with the baseline rotation (area correlation 0.996),
# and filtering the fast turn introduced rotational overshoot.
TRANSLATION_WINDOW = 21
TRANSLATION_POLYORDER = 2
TRANSITION_FRAMES = 8


def _static_pose(T_world_mesh: np.ndarray, start: int, stop: int):
    """Robust constant pose for a known rest phase."""
    rotations = Rotation.from_matrix(T_world_mesh[start:stop, :3, :3])
    R_mean = rotations.mean().as_matrix()
    t_median = np.median(T_world_mesh[start:stop, :3, 3], axis=0)
    return R_mean, t_median


def _smoothstep(k: int, n: int) -> float:
    """0 -> 1 cosine ease used only at phase transitions."""
    return float(0.5 * (1.0 - np.cos(np.pi * (k + 1) / (n + 1))))


def solve(inputs: Path) -> np.ndarray:
    recon = np.load(inputs / "reconstruction.npz")
    T = recon["T_world_mesh"].astype(np.float64, copy=True)
    if T.shape[0] < 2 or T.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected baseline shape: {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError("Baseline contains non-finite transforms")

    frame_count = T.shape[0]
    if frame_count != 553:
        # The rest boundaries above are explicit for this episode.  Refuse to
        # silently apply them to a different clip.
        raise ValueError(f"Expected 553 frames, found {frame_count}")

    # Constant poses in the two rest phases.  This pools evidence over 204 and
    # 73 frames instead of accepting independent per-frame depth drift (the
    # baseline translation range is 17.5 cm and 10.2 cm respectively).
    static = []
    for start, stop in STATIC_PHASES:
        R_static, t_static = _static_pose(T, start, stop)
        static.append((R_static, t_static + GLOBAL_XY_CORRECTION))
        T[start:stop, :3, :3] = R_static
        T[start:stop, :3, 3] = t_static + GLOBAL_XY_CORRECTION

    # Low-pass the carried/placement translation.  Rotations in the moving
    # interval retain the calibrated baseline because the observed blue-handle
    # trajectory supports them and they contain no discontinuity.
    for axis in range(3):
        T[MOVING_START:MOVING_STOP, axis, 3] = (
            savgol_filter(
                T[MOVING_START:MOVING_STOP, axis, 3],
                window_length=TRANSLATION_WINDOW,
                polyorder=TRANSLATION_POLYORDER,
            )
            + GLOBAL_XY_CORRECTION[axis]
        )

    # Blend both pose components into the neighbouring constant rest poses.
    # The blend is short (8 frames) and is centred on the first/last detected
    # motion frames, avoiding even the sub-millimetre boundary jumps left by
    # independent filtering.
    (R_pre, t_pre), (R_post, t_post) = static
    for k in range(TRANSITION_FRAMES):
        s = _smoothstep(k, TRANSITION_FRAMES)

        i = MOVING_START + k
        rel = (Rotation.from_matrix(R_pre).inv() * Rotation.from_matrix(T[i, :3, :3])).as_rotvec()
        T[i, :3, :3] = Rotation.from_rotvec(rel * s).as_matrix() @ R_pre
        T[i, :3, 3] = (1.0 - s) * t_pre + s * T[i, :3, 3]

        i = MOVING_STOP - 1 - k
        rel = (Rotation.from_matrix(T[i, :3, :3]).inv() * Rotation.from_matrix(R_post)).as_rotvec()
        T[i, :3, :3] = Rotation.from_rotvec(rel * s).as_matrix() @ T[i, :3, :3]
        T[i, :3, 3] = (1.0 - s) * T[i, :3, 3] + s * t_post

    # Keep the file-level SE(3) contract exact and numerically well behaved.
    T[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    for i in range(frame_count):
        T[i, :3, :3] = Rotation.from_matrix(T[i, :3, :3]).as_matrix()
    return T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    T = solve(args.inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, T_world_mesh=T)
    print(f"wrote {args.out} with T_world_mesh shape {T.shape} and dtype {T.dtype}")


if __name__ == "__main__":
    main()
