# Monocular 6-DoF Basket Trajectory Recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` with shape `(980, 4, 4)`, dtype
`float64`. All matrices are finite rigid transforms and map the original GLB mesh
frame to the released world frame.

## Recovered phases

The frame boundaries below were selected from the object masks, the hand/object
contact signal, and visual inspection of the full-resolution video. They are hard
coded in `solve.py` because they are episode-level decisions, not per-frame pose
values.

| Phase | Frames | Description | Treatment |
|---|---:|---|---|
| A | 0-94 | Basket at rest on the round table | Constant pose, upright orientation |
| B | 95-258 | First right-hand pick-up, carry, and placement | Per-frame translation alignment, robust SO(3) smoothing |
| C | 259-414 | Basket at rest on the floor | Constant upright pose fitted to all static observations |
| D1 | 415-745 | Second pick-up and manipulation, basket still visible | Per-frame mask alignment, robust temporal rotation filtering |
| D2 | 746-845 | Basket carried/placed while fully hidden by the person | Hand-attached translation and SO(3) propagation between visible anchors |
| E | 846-979 | Basket released and stationary | One constant final pose for all 134 frames |

The two long mask gaps are exact and data-driven: frames **746-846** and
**868-979** have zero object-mask pixels. Frame 846 is included in phase E;
frames 847-867 provide the only final-phase direct observations.

## What was changed

1. **Static phases were made genuinely static.** The baseline has exact constant
   rotations in phase A but drifts in translation and contains invalid rotation
   excursions in phases C and E. The new trajectory uses one rigid pose per static
   phase. This lets all frames contribute to the estimate and removes per-frame
   tracking noise.

2. **Static pose estimation used independent cues.** For phases C and E the basket
   is resting upright on the released floor plane. `solve.py` searches upright yaw
   and translation using: (a) rendered silhouette agreement with the SAM object
   mask, excluding person pixels; (b) normalized-color/texture correlation with the
   textured mesh; and (c) a soft floor-contact residual computed from the mesh
   convex hull. The floor is a soft cue only; no vertex is force-snapped.

   Final floor residuals are **4.5 cm** for phase C and **0.28 cm** for phase E,
   both compatible with the released floor fit tolerance (`mean 4.3 cm`,
   `std 1.8 cm`). Phase A is on a table and was not tied to the floor.

3. **Visible carried frames were aligned in image space at fixed baseline depth.**
   The baseline often had the correct object scale/orientation but a 10-30 px
   image-space offset. `solve.py` renders the mesh, compares silhouette centroids
   with the visible object mask after removing person pixels, and iteratively
   corrects the two in-plane world directions. Depth was deliberately not fitted
   from the tiny silhouette alone.

4. **Rotations were filtered as a trajectory.** A robust rolling quaternion medoid
   removes isolated flips, followed by hemisphere-consistent quaternion smoothing
   and short anchor ramps at static/dynamic boundaries. During the final long
   occlusion, hand PCA frames propagate rotation from the last visible pose and
   ease into the final resting orientation; this prevents the baseline's 90-160
   degree single-frame jumps from entering the output.

5. **The fully occluded carry was anchored to the hand.** From 746-845 the baseline
   object center is up to 2.3 m from the reconstructed left hand. A local
   hand-to-object offset learned from the visible carry was applied to the hand
   trajectory, temporally smoothed, and blended to the final resting pose at
   release.

## Validation without ground truth

Validation used rendered-mask IoU on every second frame that has a usable object
mask (383 sampled visible frames). The predicted silhouette is intersected with
`~person` before comparison, matching the fact that SAM object masks represent the
visible object.

| Phase | Visible samples | Baseline IoU | Result IoU | Delta | Improved >0.01 | Worse >0.01 |
|---|---:|---:|---:|---:|---:|---:|
| A static | 48 | 0.874 | **0.933** | +0.059 | 100% | 0% |
| B carry | 82 | 0.757 | **0.835** | +0.077 | 89.0% | 1.2% |
| C static | 78 | 0.783 | **0.901** | +0.118 | 100% | 0% |
| D visible carry | 165 | 0.622 | **0.856** | +0.233 | 85.5% | 3.0% |
| E visible placement | 10 | 0.401 | **0.756** | +0.356 | 100% | 0% |
| All sampled visible frames | 383 | 0.710 | **0.868** | +0.158 | - | - |

Independent checks for the unobservable interval D2 (frames 746-845):

| Metric | Baseline | Result |
|---|---:|---:|
| Object-center to active left-hand distance, median | 110.1 cm | **9.8 cm** |
| Same distance, 90th percentile | 222.2 cm | **10.7 cm** |
| Same distance, maximum | 233.8 cm | **12.4 cm** |

Temporal plausibility also improved substantially:

| Segment | Metric | Baseline | Result |
|---|---|---:|---:|
| B | rotation step, median | 3.79 deg | **1.52 deg** |
| B | rotation step, maximum | 136.1 deg | **56.3 deg** |
| D1 | rotation step, median | 4.19 deg | **1.66 deg** |
| D1 | rotation step, maximum | 53.5 deg | **24.4 deg** |
| All frames | translation step, 99th percentile | 14.9 cm baseline* | **3.9 cm** |

\*The baseline 99th-percentile value is from the baseline transform sequence; the
result value is measured on the final output.

## Confidence

| Phase | Confidence | Reason |
|---|---|---|
| A | High | 95 frames, stable full mask, 93% silhouette IoU |
| B | Medium | Object visible but fast motion and occlusion; 84% IoU and much smoother rotation |
| C | High | 156 static frames, persistent mask, floor contact, 90% IoU |
| D1 visible | High | Large object, mask alignment improved 85% of sampled frames |
| D2 occluded | Medium-low | No object pixels for 101 frames; pose relies on hand attachment and smooth interpolation |
| E visible placement | Medium-high | Small but consistent visible strip; upright/floor constraint plus 76% IoU |
| E fully occluded | Medium | Pose is constant and floor-consistent, but frames 868-979 have no object mask |

The largest remaining uncertainty is the 180-degree/alias orientation branch of
the nearly rotationally symmetric basket during the long final occlusion. The
selected branch is supported by the last visible body silhouette, texture
correlation, upright-floor constraint, and continuity with the hand release; it
cannot be independently verified after frame 867.

## Reproducibility and runtime

Final command:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script uses deterministic seeds, reads only the released inputs, does not use
network access, and does not read scratch files. Final measured runtime was
**126.1 s** internal (**128.5 s** wall), comfortably below the 45 min GPU budget.
