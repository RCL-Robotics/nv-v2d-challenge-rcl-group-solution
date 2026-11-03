# Tall bar stool trajectory recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` as a finite `(778, 4, 4)`
`float64` array, mapping the supplied mesh-file coordinates into the released
world frame.

The baseline was already a strong per-frame registration: its rendered
silhouette and image edges follow the stool closely, and its rotation sequence
contains no sudden 180 degree segment flip. I therefore made only changes that
had trajectory-level support rather than replacing the baseline wholesale.

## Phases found

| Frames | Phase | Evidence / treatment |
|---|---|---|
| 0-279 | stool lying at rest | The object mask is stationary. All 280 baseline poses are pooled into one rigid SE(3) pose. |
| 280-619 | two-handed lift and tip-up | The stool motion begins near frame 280. Rotation is smoothed in a local SO(3) tangent space and translation with a 17-frame quadratic Savitzky-Golay fit. |
| 620-777 | stool upright at rest | Rotation reaches its settled value by about frame 620; the human releases the stool at about frame 652. All frames 620-777 are pooled into one rigid pose. |

Eight-frame smoothstep pose transitions join the pooled rest poses to the
smoothed manipulation trajectory. A 20-frame smoothstep ramps the depth
correction across each phase boundary.

## What changed

1. **Static pose pooling.** Per-frame tracker noise in each rest phase was
   replaced by the chordal mean rotation and mean translation of that phase.
   This makes each rest pose exactly constant and lets hundreds of frames
   contribute to it.
2. **Temporal smoothing.** In the manipulation phase, the baseline rotation was
   written as tangent vectors relative to its mean, filtered with
   `savgol_filter(window=17, polyorder=2)`, then mapped back to SO(3).
   Translation received the same local quadratic filter. This retained the
   tipping motion while removing short-period tracking jitter.
3. **Camera-ray depth correction.** Baseline and final poses were rendered with
   nvdiffrast and compared with the aligned MoGe-2 object depth, excluding the
   SAM person mask. Pooled robust depth offsets were +3.75 cm along the viewing
   ray in the first rest phase, -4.2 cm during manipulation, and -12.0 cm in the
   final rest phase. The offset is ramped rather than stepped. Independent
   Canny silhouette-edge checks agreed with the direction of these corrections;
   the final correction also moves the stool's lowest mesh point from 2.5 cm
   below the estimated floor plane to 0.7 cm above it.
4. **Hypothesis checks.** FoundationPose and CoCoNet rotations from
   `reconstruction.npz`, plus interpolations toward them, were rendered as
   alternate orientation hypotheses. They did not improve the joint
   mask/edge/depth evidence consistently, so the baseline orientation was
   retained and only smoothed. No symmetry flip was introduced.

## Validation

All quantities are computed by the deterministic validation pass on the supplied
video, SAM masks, mesh, MoGe-2 aligned depth, and released ground plane. No ground
truth was used. Lower is better for chamfer, edge distance, depth magnitude,
and acceleration.

| Phase | Metric | Baseline | Result |
|---|---|---:|---:|
| All | SAM mask IoU (person excluded) | 0.7691 | **0.7859** |
| All | Outer-boundary chamfer (px) | 8.838 | **8.484** |
| All | Rendered contour to Canny-edge distance (px) | 1.302 | **1.231** |
| All | Rotation step (deg/frame) | 0.397 | **0.129** |
| All | Translation acceleration (m/frame^2) | 0.01505 | **0.00070** |
| All | Median floor-plane gap (m) | -0.0249 | **-0.0129** |
| Rest 0 | Mask IoU | **0.7285** | 0.7182 |
| Rest 0 | Boundary chamfer (px) | **10.446** | 10.448 |
| Rest 0 | Canny-edge distance (px) | 1.733 | **1.649** |
| Rest 0 | Median absolute depth residual (m) | 0.0890 | **0.0426** |
| Rest 0 | Rotation step (deg/frame) | 0.367 | **0.000** |
| Manipulation | Mask IoU | 0.7950 | **0.8111** |
| Manipulation | Boundary chamfer (px) | 8.751 | **8.445** |
| Manipulation | Canny-edge distance (px) | 1.058 | **1.041** |
| Manipulation | Median absolute depth residual (m) | 0.0991 | **0.0764** |
| Manipulation | Rotation step (deg/frame) | 0.495 | **0.294** |
| Final rest | Mask IoU | 0.7850 | **0.8518** |
| Final rest | Boundary chamfer (px) | 6.175 | **5.086** |
| Final rest | Canny-edge distance (px) | 1.063 | **0.902** |
| Final rest | Median absolute depth residual (m) | 0.0549 | **0.0500** |
| Final rest | Floor-plane gap (m) | -0.0252 | **+0.0067** |

The rest-0 SAM IoU and chamfer are essentially unchanged/slightly lower because
SAM fills some open stool-frame regions; the actual image-edge distance and the
independent depth residual both improve. I therefore retained the depth fix but
mark absolute translation confidence in that phase as medium-high rather than
high. The first rest phase still has a 4.7 cm signed floor gap, comparable to the
ground-plane estimate's own 3.9 cm mean foot-plane residual; no additional floor
snap was applied because it visibly degraded image-edge agreement.

## Confidence

- **Rest 0 (frames 0-279): high confidence in rotation and constancy; medium-high confidence in absolute depth.** Rotation is pooled over 280 static frames and the correction improves both image-edge and depth evidence. The SAM external-boundary metric is insensitive/conflicted because of mask fill in the stool's open frame.
- **Manipulation (frames 280-619): high confidence.** Every validation measure improves, the motion remains close to the baseline, and short-period jitter is reduced by roughly a factor of two in rotation step size.
- **Final rest (frames 620-777): high confidence.** Rotation and translation are pooled, mask IoU and edge alignment improve substantially, depth residual improves, and the final floor gap is near zero. The object is visibly upright and stationary over this interval.
- **No 180 degree flip:** the baseline rotation path is continuous, the rendered hand/seat/leg geometry matches the video throughout, and the result changes baseline rotations by only 0.43 degrees on average (maximum 1.88 degrees).

## Runtime and reproducibility

`solve.py` is deterministic, offline, and uses only NumPy, SciPy, the released
`reconstruction.npz`, and `camera.json`. It performs no rendering and no model
inference. Measured runtime was **0.64 s** on this container. The separate
nvdiffrast/depth validation pass took about one minute.

Run:

```bash
python /work/outputs/solve.py \
  --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```
