# Tall bar stool trajectory recovery

## Summary

The released reconstruction is already a strong per-frame tracker: after excluding
the person, its rendered silhouette overlaps the SAM object mask at a full-frame
mean IoU of **0.8209**. Its main weakness in this episode is not a global symmetry
flip. It is frame-to-frame instability, especially in the two phases where the same
physical pose should explain many frames. The final static phase is the clearest
failure: the baseline pose keeps drifting by centimetres even after the person has
released the stool.

I replaced both static phases with one pooled 6-DoF pose per phase and made only a
small, conservative temporal adjustment during manipulation. The resulting
trajectory has mean mask IoU **0.8344**, improves both static phases overall (and
every final-static frame), keeps floor contact, and removes the large non-physical
jitter in the released pose.

Final output: `object_poses_world.npz`, key `T_world_mesh`, shape `(678, 4, 4)`,
dtype `float64`.

## Phases found

| Frames | Phase | Evidence | Confidence |
|---|---|---|---|
| 0-124 | Initial static stool under the desk | Object mask is fixed; baseline translation changes by only a few millimetres; person starts contacting it around frame 96 but does not move it | High |
| 125-419 | Grasped and dragged / turned | Mask and pose begin moving at 125; person and stool are both visible for most of this phase | Medium-high |
| 420-560 | Heavily occluded manipulation | Person mask covers most of the stool; only legs/back slivers remain visible around 430-460 and 520-550 | Medium-low |
| 561-591 | Final placement / settling | Stool becomes visible again and decelerates into its final location; contact probability drops to zero by 586 | Medium |
| 592-677 | Final static stool | Object mask and baseline pose are static after release | High |

The phase boundaries are fixed constants in `solve.py`: `INIT_END = 125` and
`FINAL_START = 592`. They were selected from the observed motion onset and the
contact/settling transition, not from ground truth.

## Method

1. **Diagnose the baseline.** I rendered the exact textured mesh at every frame with
   nvdiffrast and compared its silhouette with `masks.h5` after removing a dilated
   person mask. I also scored 180-degree flips about the mesh's three principal
   axes. On representative unoccluded frames the unflipped pose scored 0.862 mean
   IoU; the best flipped pose scored only 0.351. Thus the baseline does **not** have
   a global 180-degree symmetry failure in this episode.
2. **Pool static evidence.** For frames 0-124 and 592-677, I fit one shared pose to
   all frames in that phase. The objective is soft silhouette Dice against the
   visible object mask. A constant map is interpolated over nvdiffrast
   barycentrics and antialiased to obtain differentiable sub-pixel silhouette
   edges. Fitting is coarse-to-fine: 384x288 with 21/29 sampled frames, 768x576
   with 16/15 frames, then full 1536x1152 with every phase frame. Starting from the
   frame-0 (initial) or frame-592 (final) release pose, the six pose parameters are
   optimized while all frames share the same estimate.
3. **Use only weak manipulation smoothing.** Frames 125-591 otherwise retain the
   released trajectory. Rotation quaternions and translations are smoothed with a
   5-frame first-order Savitzky-Golay filter. The smoothed segment is blended back
   to the pooled static poses over 10 frames at each end. This reduces one-frame
   jumps while preserving the actual drag path.
4. **Keep depth evidence conservative.** The fit uses the exact metric mesh,
   calibrated camera, object masks, and the released baseline depth. I checked the
   result against MoGe-2 depth and the released floor plane, but did not force the
   stool onto the floor or tune each frame to the object-only depth map. The solver
   does not need MoGe depth.

## Validation table

All metrics are computed without ground truth. **Mask IoU** excludes the person
mask (dilated by 7 pixels) and uses the full-resolution rendered mesh silhouette.
**Floor min** is the minimum signed vertex-to-floor distance; the released floor
plane has several-cm fit tolerance, so small non-zero values are expected.

| Metric | Baseline | Result | Change |
|---|---:|---:|---:|
| Visible-mask IoU, initial 0-124 | 0.8712 | **0.8824** | +0.0112 |
| Visible-mask IoU, manipulation 125-591 | 0.8178 | **0.8197** | +0.0019 |
| Visible-mask IoU, final 592-677 | 0.7650 | **0.8440** | +0.0790 |
| Visible-mask IoU, all 678 frames | 0.8209 | **0.8344** | +0.0134 |
| Mask precision / recall, all frames | 0.9213 / 0.8803 | **0.9312 / 0.8869** | both improved |
| Initial static vertex-jitter RMS | 9.98 mm | **0.002 mm** | one shared pose |
| Final static vertex-jitter RMS | 4.01 mm | **0.001 mm** | one shared pose |
| Floor min absolute error, mean | 5.64 mm | 5.60 mm | effectively tied |
| Floor min absolute error, 95th percentile | 11.93 mm | **10.64 mm** | -1.29 mm |
| Floor min signed range | -31.2 to +18.2 mm | **-14.3 to +15.0 mm** | less penetration |
| Manipulation translation step, mean | 12.05 mm/frame | **9.58 mm/frame** | smoother |
| Manipulation rotation step, mean | 1.094 deg/frame | **0.943 deg/frame** | smoother |
| Manipulation rotation step, 95th percentile | 3.450 deg/frame | **3.053 deg/frame** | fewer jumps |
| MoGe-2 depth absolute residual, sampled frames | 82.2 mm | **81.0 mm** | slight improvement |

The static phase is the major gain. Every final-static frame improves in mask IoU,
not just a subset. The manipulation mask improvement is small because the tracker is
already good on visible frames; the temporal metrics indicate that the main benefit
there is removing jitter rather than changing the path.

### Symmetry-flip diagnostic

Mean visible-mask IoU on six representative unoccluded frames (0, 60, 120, 600,
650, 677):

| Hypothesis | Mean IoU |
|---|---:|
| Identity / selected pose | **0.8617** |
| 180 deg about mesh z | 0.2988 |
| 180 deg about mesh x | 0.3510 |
| 180 deg about mesh y | 0.2139 |

This strongly rules out a whole-episode 180-degree flip. The same identity orientation
is supported before manipulation, during the visible middle, and at the final rest.

## What changed and why

- Frames 0-124: replaced 125 noisy baseline poses with one pooled pose. This is
  consistent with the physical fact that the stool is stationary and removes
  centimetre-scale tracker jitter.
- Frames 125-591: retained the released trajectory, with only a 5-frame local-linear
  smoothing and short endpoint blends. Stronger smoothing was tested but reduced
  mask agreement, so it was rejected.
- Frames 592-677: replaced 86 drifting baseline poses with one pooled pose. This is
  the largest and most reliable improvement; the stool is visibly stationary.
- No symmetry flip was applied. No floor snapping was applied. No per-frame fitted
  tables or external residual corrections are embedded in the solver.

## Confidence

| Frames | Confidence | Reason |
|---|---|---|
| 0-124 | High | Long static interval, many pooled silhouette observations, floor contact |
| 125-419 | Medium-high | Object mostly visible; conservative changes only |
| 420-560 | Medium-low | Person occludes most of the object; trajectory is supported mainly by continuity and visible slivers |
| 561-591 | Medium | Object re-emerges; smooth settling motion |
| 592-677 | High | Completely static final interval with 86 pooled observations |

The least certain outputs are the depth and yaw during the heavily occluded interval
420-560. The released trajectory was therefore not strongly overridden there.

## Reproducibility and runtime

Command:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script is offline, uses fixed seeds, and requires no downloaded weights. The final
run took **63.3 seconds** on the available GPU (a second verification run on the shared GPU took
93.2 seconds; the two outputs agreed to `7.8e-6` maximum absolute error). Full-resolution fitting uses batches of
eight frames, keeping GPU memory comfortably below the 9 GB limit. Output validation:
678 finite float64 transforms, homogeneous last row exact, rotation orthogonality
error below `1.1e-5`, determinant in `[0.9999957, 1.0000002]`.
