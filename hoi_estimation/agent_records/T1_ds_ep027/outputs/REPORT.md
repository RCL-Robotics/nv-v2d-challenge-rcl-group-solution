# Woven-basket 6-DoF trajectory recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` as float64 with shape
`(565, 4, 4)`.  Each matrix maps the exact input mesh-file coordinates
to the released world frame.

## Recovered phases

| Phase | Frames | Motion model | Confidence |
|---|---:|---|---|
| Initial rest on floor | 0-152 | one constant pose | high |
| Pickup and carry | 153-437 | robustly smoothed rigid motion | medium |
| Placement transition | 438-449 | blend into the desk pose | medium-low |
| Final rest on desk | 450-564 | one constant pose | high |

The phase boundaries were selected from the pooled-mask test and verified in
the video.  The initial pooled pose retains mask IoU above 0.85 through frame
152 and drops at the lift; the final pooled pose reaches IoU above 0.90 as the
basket settles on the desk.

## What changed and why

1. **Pooled stationary poses.**  For each rest phase I formed a temporal median
   RGB crop and object mask.  The moving person is suppressed by the median.
   Starting from the baseline, I searched absolute vertical-axis yaw over the
   complete 360-degree range, fitted camera-ray depth from projected-area ratio,
   fitted image position from silhouette centroid, and selected yaw by pooled
   texture NCC plus silhouette IoU.  One pose is assigned to every frame in the
   phase, eliminating per-frame depth drift and jitter.
2. **Temporal decoding while carried.**  Yaw is unwrapped and passed through a
   median filter followed by a short Savitzky-Golay smoother.  The small tilt
   component is averaged in SO(3), and translation is Gaussian-smoothed in time.
   The 122.8-degree one-frame baseline rotation discontinuity is not retained as
   an isolated step.
3. **Endpoint blending.**  The carried trajectory is smoothly blended into the
   the two pooled static poses over short 12-frame transition windows.  This removes the large translation
   steps left by independent per-frame depth estimates at pickup and release.
4. **Symmetry branch.**  Both the baseline and its 180-degree vertical symmetry
   transform were scored with textured rendering.  Silhouette agreement favored
   the baseline branch throughout this episode, so no per-frame flip was applied;
   the temporal decoder still removes tracker jumps without changing branch.

The large moving-frame changes are therefore supported by temporal motion and
static anchors, not by a per-frame correction table.

## Validation without ground truth

Visible silhouette IoU excludes person-mask pixels and compares the rendered
mesh with the SAM2 object mask.  Values are mean / median; higher is better.

| Phase | Frames | Baseline IoU | Result IoU | Delta |
|---|---:|---:|---:|---:|
| Initial rest on floor | 0-152 | 0.760 / 0.764 | 0.892 / 0.898 | +0.133 |
| Pickup / carried | 153-437 | 0.607 / 0.640 | 0.606 / 0.617 | -0.001 |
| Placement transition | 438-449 | 0.851 / 0.850 | 0.878 / 0.871 | +0.026 |
| Final rest on desk | 450-564 | 0.856 / 0.856 | 0.937 / 0.938 | +0.081 |

Pooled-pose texture evidence: initial NCC `0.173` with mask
IoU `0.901` at yaw offset `-5.0` degrees;
final NCC `0.119` with mask IoU `0.938`
at yaw offset `-25.0` degrees.

Dynamic-interval temporal diagnostics (frames 153-449):

| Metric | Baseline | Result |
|---|---:|---:|
| Rotation step median | 3.60 deg | 1.56 deg |
| Rotation step p95 | 15.22 deg | 6.01 deg |
| Rotation step max | 122.77 deg | 29.06 deg |
| Translation step mean | 0.0190 m | 0.0130 m |
| Translation step max | 0.1389 m | 0.0361 m |

The result improves mean silhouette IoU over the full episode from
`0.704` to `0.757`.  The gain is concentrated
in the two stationary phases, where pooled evidence is strongest.  Carried-frame
IoU is essentially neutral, while temporal smoothness improves substantially;
this is the intended trade-off for noisy/occluded per-frame observations.
No ground truth, challenge annotations, or posed evaluation data were used.

## Confidence

* **Initial floor pose: high.**  The object is unobstructed, large enough for
  textured comparison, and all static observations support one pose.
* **Carried interval: medium.**  Translation and yaw are temporally constrained,
  but hand occlusion around frames 188-246 makes the absolute symmetry branch
  less certain than in the rest phases.
* **Final desk pose: high.**  The basket is visible from frame 450 onward and
  the pooled pose has the strongest silhouette support in the episode.

## Runtime

Deterministic solve runtime: **9.9 s** on the provided GPU.
The script uses no network access.  Besides the inputs, it reads the provided
`/pipeline/depth/moge2/aligned.h5` only to initialize depth for the two pooled
stationary poses; all final pose values are recomputed from rendered evidence.
