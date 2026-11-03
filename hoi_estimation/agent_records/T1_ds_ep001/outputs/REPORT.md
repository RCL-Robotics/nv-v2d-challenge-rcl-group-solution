# Vacuum 6-DoF Trajectory Recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` with shape `(747, 4, 4)`,
dtype `float64`, for every frame. Rotations were projected back to SO(3), all
values are finite, and static phases use exactly one pose.

The released baseline (`T_world_mesh` in `reconstruction.npz`) was improved,
not replaced wholesale. Baseline poses remain the fallback and still provide
the metric depth calibration.

## Recovered phases

| Phase | Frames | Interpretation | Confidence |
|---|---:|---|---|
| Pre-contact rest | 0-101 | Vacuum leaning on/next to the stool, untouched | High |
| Pickup / lift | 102-159 | Hand makes sustained contact and raises the vacuum | High |
| Manipulation | 160-619 | Side-to-side vacuuming while carried | Medium |
| Placement | 620-636 | Vacuum returned toward the stool, contact ends | Medium-high |
| Post-contact rest | 637-746 | Vacuum left leaning on the stool | High |

The first/last sustained contact boundaries were detected from
`contact_prob_right_left > 0.05`; the first contact is frame 102 and the last
is frame 635, with a two-frame settling margin at the end.

## Method

1. **Static-pose pooling.** The baseline frames with the best silhouette
   agreement over the clean pre/post-contact intervals were selected
   automatically (frames 26 and 656). Each was refined with a small,
   regularized translation-only render-and-compare fit using the object mask.
   All frames in each rest phase then use that single pose.
2. **Long-axis ambiguity.** In the final rest phase the silhouette cannot
   distinguish rotation about the vacuum's local long axis. Pooled textured
   render comparison selected a constant `-20 deg` correction about that axis:
   mean red-chroma NCC was `0.818` for the result versus `0.523` for the
   baseline over frames 640/656/680/700/730. This is the only manually chosen
   per-phase rotation constant; it is documented in `solve.py`.
3. **Optical-flow PnP.** The object image is cropped (around the union of all
   object masks) and decoded once. For each dynamic frame, forward and backward
   pyramidal Lucas-Kanade tracks are initialized from visible mesh points
   rendered with the baseline pose. Backward-forward consistency rejects bad
   tracks. RANSAC PnP maps the corresponding mesh points to the tracked pixels,
   producing two independent pose hypotheses from the two temporal directions.
4. **Temporal decoding.** A 9-frame Savitzky-Golay baseline trajectory is the
   local prior. Flow hypotheses are accepted only when they are within
   `35 deg` and `0.15 m` of that prior. A three-pass robust outlier check
   replaces isolated rotation/translation jumps using neighbouring poses, and
   eight-frame blends join the dynamic segment to the pooled rest poses.
5. **No floor snapping.** The released floor plane was checked only as a
   plausibility diagnostic. Static poses put the lowest 1% of vertices between
   1.7 cm below and 0.5 cm below the fitted floor plane, inside the stated
   several-centimetre fit tolerance.

## Validation without ground truth

Mask metrics are computed against the released SAM2 object mask. Visible IoU
ignores predicted pixels covered by the person mask. Flow error is the median
pixel reprojection residual on a held-out 40% split of the forward LK tracks;
those points were not used by the PnP fit. Lower is better except for IoU/Dice.

| Phase | Frames | Dice baseline -> result | Visible IoU baseline -> result | Held-out flow px baseline -> result |
|---|---:|---:|---:|---:|
| Pre-contact rest | 0-101 | 0.648 -> **0.732** | 0.485 -> **0.579** | n/a (static) |
| Pickup / lift | 102-159 | 0.609 -> **0.616** | 0.444 -> **0.453** | 3.464 -> **0.419** |
| Manipulation | 160-619 | 0.490 -> **0.493** | 0.338 -> **0.341** | 2.602 -> **0.379** |
| Placement | 620-636 | 0.593 -> **0.600** | 0.427 -> **0.434** | 1.479 -> **0.125** |
| Post-contact rest | 637-746 | 0.557 -> **0.605** | 0.387 -> **0.434** | n/a (static) |
| **All frames** | 0-746 | 0.533 -> **0.554** | 0.376 -> **0.398** | n/a |

Temporal checks:

| Check | Baseline | Result |
|---|---:|---:|
| Mean / P95 angular step, all frames | 4.10 / 14.69 deg | **4.03 / 14.03 deg** |
| Max angular step, all frames | 61.53 deg | **58.36 deg** |
| Mean angular step, pre-contact rest | 1.188 deg | **0 deg** |
| Mean angular step, post-contact rest | 0.084 deg | **0 deg** |
| Static translation variation, both rest phases | up to 3.2 cm | **0** |
| Dynamic 2-D flow median residual | 2.60 px | **0.38 px** |

Texture checks on the dynamic sample (12 frames) were mixed for red chroma
(`0.267 -> 0.246`) but improved for image edges (`0.111 -> 0.145`) and gray
level (`0.066 -> 0.087`). The static texture checks improved strongly. This is
why confidence is only medium through the most occluded manipulation frames.

## Changes that were not adopted

- A large global smoothing filter reduced apparent jitter but degraded held-out
  optical-flow residuals; it was rejected.
- Additional per-anchor spin corrections from texture sometimes improved red
  NCC but raised held-out flow error from about `0.5 px` to `2.9-3.7 px`, so
  they were not applied to the dynamic trajectory.
- No floor snap or global depth-scale change was used: silhouette scale and
  MoGe depth were not consistently reliable enough on this thin, partly
  occluded object.

## Reproducibility and runtime

Final command:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Measured wall time in this container: **about 46 s** on the assigned GPU. The
script seeds Python/NumPy/Torch, uses single-threaded OpenCV, performs no
network access, and reads no scratch results. Peak GPU use is well below the
9 GB limit.
