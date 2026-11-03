# Bowl Trajectory Recovery Report

## Summary

I reconstructed a temporally consistent world-space trajectory for all 870 frames and wrote it to
`object_poses_world.npz` under `T_world_mesh` (float64, shape `(870,4,4)`).

The baseline translations are already depth-calibrated and were retained as the primary metric
translation evidence. The dominant failure is yaw about the bowl's near-symmetry axis: the baseline
changes yaw abruptly (up to 110.3 degrees between adjacent frames) while the cup remains upright and
the hands move smoothly. I replaced that yaw with a static-anchor + smooth manipulation trajectory,
kept a low-passed version of the baseline's measured upright-axis tilt, smoothed translation, and
pooled each static phase to one pose.

No ground-truth annotation or challenge trajectory data was used.

## Phases found

| Phase | Frames | Evidence and treatment |
|---|---:|---|
| Rest on chair | 0-199 | Bowl mask and projected pose are effectively stationary. Both-hand contact begins around frame 166, but the bowl is still resting and does not move until about frame 200. All poses are set to one pooled pose. |
| Lift and carry | 200-459 | Contact probabilities are 1 for both hands, object mask moves steadily, and the person carries the bowl away from the chair. Translation is smoothed with a short Gaussian filter; yaw follows a smooth anchored path. |
| Lower, slide, and release | 460-789 | Heavy hand/body occlusion; object mask is often only tens of pixels. The right-hand contact remains high through about frame 768. The end of the phase is blended to the pooled floor-rest pose. |
| Rest on floor | 790-869 | Bowl is stationary in the image from frame 790 onward. Rotation and translation are constant over this phase. |

## What changed

1. **Static pooling.** The baseline had small translation drift and pose changes during the two
   static phases. I replaced frames 0-194 and 790-869 with the component-wise median translation
   from the corresponding static evidence and a single rotation.
2. **Yaw correction.** The bowl is almost rotationally symmetric around its local Y axis. A coarse
   5-degree textured render search, pooled over six frames in each static phase, selected
   `R_y(+55 deg) @ R_frame0` for the initial pose and `R_y(-30 deg) @ R_frame869` for the final
   pose. These corrections are approximately 55 degrees and 30 degrees in opposite directions from
   the baseline yaw anchors.
3. **Manipulation trajectory.** Between the anchors I used a smoothstep yaw interpolation over
   frames 200-780. A weak cue from the vector between the reconstructed right/left hand centres
   modulates timing, but the final trajectory is mostly driven by the two pooled image anchors
   because the hand reconstruction is noisy.
4. **Timing residual.** Held-out textured render checks consistently favored a small negative
   smooth residual during the middle of the carry. I represented it by a Gaussian with amplitude
   -25 degrees, centre frame 400, and sigma 80 frames. It is zero at both static phases and never
   introduces more than about 4 degrees of frame-to-frame rotation by itself.
5. **Tilt and translation smoothing.** I removed yaw from the baseline rotation, smoothed the
   remaining tilt quaternion with sigma 2 frames, and re-applied the reconstructed yaw. Translation
   was low-pass filtered with sigma 2 frames and smoothly blended into the pooled static poses.
   No depth rescaling or floor-plane snapping was used.

## Validation without ground truth

Validation used every 15th frame plus all phase boundaries, rendered at 768x576 with the exact
textured mesh. Predicted visible pixels exclude a 9-pixel dilation of the SAM2 person mask. Mask IoU
is against the SAM2 object mask. Texture residual is normalized Lab-luminance L1 inside the visible
object region; **lower is better**.

| Phase | Frames sampled | Baseline mask IoU | Result mask IoU | Baseline texture | Result texture | Confidence |
|---|---:|---:|---:|---:|---:|---|
| Rest on chair | 16 | 0.7778 | 0.7660 | 0.9185 | **0.8634** | Medium-high (constant-pose evidence strong; absolute yaw is symmetry-ambiguous) |
| Lift and carry | 26 | 0.4996 | 0.4482 | 0.8714 | **0.8362** | Medium |
| Lower / slide / release | 34 | 0.2395 | 0.2378 | 0.8291 | **0.7575** | Medium-low (small visible area and occlusion) |
| Rest on floor | 10 | 0.8506 | **0.8532** | 0.8092 | **0.7920** | Medium-high |
| Overall | 86 | 0.4894 | 0.4712 | 0.8634 | **0.8192** | Medium |

The result gives up about 0.018 overall mask IoU, mostly because the baseline's per-frame yaw was
allowed to fit each silhouette independently and because translation is now temporally smoothed.
The textured appearance and temporal/physical checks improve substantially:

| Metric | Baseline | Result |
|---|---:|---:|
| Rotation step, median / p95 / max (deg) | 2.382 / 20.869 / 110.259 | **0.279 / 1.243 / 3.044** |
| Translation step, median / p95 / max (m) | 0.0091 / 0.0338 / 0.0865 | **0.0048 / 0.0165 / 0.0238** |
| Rotation spread inside static phases | nonzero/up to 6.9 deg | **exactly 0 deg** |
| Translation spread inside static phases | up to 5.3 cm | **exactly 0 m** |

Confidence is limited by the fact that the bowl is nearly rotationally symmetric. The two static
anchors are internally stable across frames, and the final floor pose improves over the baseline
texture and mask checks, but a residual yaw error of roughly 5-15 degrees remains plausible. I did
not force the bowl onto the ground plane: at the final frame the lowest mesh samples are about
3 cm below the fitted plane, comparable to the baseline and to the released plane-fit tolerance.

## Reproduction

Run offline:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The solver is deterministic (fixed seed, no randomness), requires only NumPy and SciPy at runtime,
and completed in **0.42 seconds** on the CPU. Re-running it produced the identical SHA-256:

`5dbb4b4de989d0d7fbae2b1f031a3924bdff3b4a00b6d8c5ef60467c134b4a02`

Output validation: finite values, shape `(870,4,4)`, maximum rotation orthogonality error
`8.9e-16`, and rotation determinants within `6e-16` of 1.
