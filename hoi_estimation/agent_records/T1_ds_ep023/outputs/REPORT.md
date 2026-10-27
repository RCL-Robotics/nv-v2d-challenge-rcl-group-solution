# Toy-stroller 6-DoF trajectory recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` with shape `(946, 4, 4)` and dtype
`float64`. Each matrix maps the supplied mesh-file coordinates to the released world
frame. The run was deterministic and completed in **122.7 s** on the available GPU.

The supplied CARI4D/FoundationPose/CoCoNet trajectory was retained as the main
source of evidence. The change is a rigid-trajectory correction: pooled rest poses,
robust SE(3) smoothing, silhouette registration, and smooth phase transitions.
No ground-truth or challenge annotation was used.

## Phases found

| Frames | Phase | Evidence | Confidence |
|---|---|---|---|
| 0-131 | Initial rest | Object silhouette centroid and pooled textured-mesh scores stay fixed. Contact begins earlier, but the stroller does not visibly move until about frame 132. | High |
| 132-900 | Manipulation / push | Centroid moves and the object yaws through a long sweep while staying in contact with the floor. Includes short hand/leg occlusions. | Medium-high |
| 901-945 | Final rest | Motion settles at frame 901 and all subsequent frames support one pooled pose. | High for the image registration; medium for absolute depth |

The transitions use a 14-frame acceleration blend at 132-145 and a 11-frame
deceleration blend at 890-900, avoiding the abrupt 3-6 cm one-frame jumps produced
by concatenating the tracked and pooled segments.

## Method

1. **Temporal prior.** Translation is median-filtered over 5 frames and then fitted
   with an 11-frame cubic Savitzky-Golay filter. Rotations are smoothed as a
   continuous quaternion branch with a 5-frame component median and Gaussian
   smoothing. This preserves the physical yaw sweep past 180 degrees while removing
   isolated tracker jumps. Median translation step falls from 11.3 mm to 6.0 mm and
   median rotation step from 0.77 to 0.31 degrees.
2. **Constant rest poses.** Every frame in 0-131 and 901-945 is replaced by one
   chordal rotation mean and median translation. This is the largest reliable
   improvement: the original tracking jitter in these phases was about 1.2 cm and
   0.5 degrees around the constant truth.
3. **Silhouette registration.** The textured mesh is rasterized with nvdiffrast.
   Rendered visible pixels are compared with the SAM object mask after excluding the
   SAM person mask. The per-frame image-space centroid residual is outlier-clipped,
   interpolated at low-visibility frames, and temporally smoothed. The pose is shifted
   in the camera plane to register the projected object. This improved full-sequence
   mask IoU from 0.563 to 0.614 on the validation sample.
4. **Texture calibration.** Per-vertex GLB texture colours were rasterized and
   compared over predicted-visible object pixels using Sobel-gradient NCC and
   per-channel colour correlation. A coarse `[-2, 2] px` pooled-pose grid selected
   two-pixel rest-phase refinements. For the moving phase, gradient calibration
   favoured 1.5 px less downward silhouette correction while leaving mask IoU nearly
   unchanged.
5. **Symmetry check.** At every 20th frame I scored the result and 180-degree flips
   about all three oriented-bounding-box axes and world up. The unflipped trajectory
   had decisively higher mean mask and texture agreement. Flip-only wins occurred in
   low-visibility, low-texture frames with much lower silhouette IoU, and did not form
   a coherent segment. No flip was adopted.

## Validation without ground truth

Metrics were computed on every fifth frame (190 frames), using only released inputs.
`gradNCC` is the normalized gradient correlation between the input image and the
texture-rendered mesh; `colour` is the mean per-channel Pearson correlation.
`depth MAE` compares rendered camera depth with the supplied aligned MoGe-2 depth at
visible object-mask pixels. MoGe-2 is biased on the thin/open stroller geometry and is
therefore used only for relative comparison, not as a hard objective.

| Phase | Method | Mask IoU | gradNCC | colour | depth MAE (m) |
|---|---|---:|---:|---:|---:|
| Initial rest (0-131) | baseline | 0.5717 | -0.0563 | 0.2381 | 0.2545 |
| Initial rest (0-131) | result | **0.6566** | **0.3962** | **0.4154** | **0.2285** |
| Manipulation (132-900) | baseline | 0.5563 | 0.0716 | 0.1958 | **0.1740** |
| Manipulation (132-900) | result | **0.6046** | **0.1368** | **0.2769** | 0.1747 |
| Final rest (901-945) | baseline | **0.6569** | 0.1375 | 0.2873 | **0.0865** |
| Final rest (901-945) | result | 0.6472 | **0.2888** | **0.3622** | 0.0869 |
| All sampled frames | baseline | 0.5633 | 0.0565 | 0.2061 | 0.1813 |
| All sampled frames | result | **0.6140** | **0.1808** | **0.3006** | **0.1782** |

The final-rest pose has slightly lower silhouette IoU than the baseline, but roughly
twice the texture-gradient correlation and clearly higher colour correlation. I
preferred the texture evidence there because the rear-facing view has weak silhouette
information and strong texture asymmetry.

Additional physical checks:

| Metric | Baseline | Result |
|---|---:|---:|
| Translation step median / p95 / max (mm) | 11.3 / 45.8 / 175.9 | 5.95 / 17.7 / 34.9 |
| Rotation step median / p95 / max (deg) | 0.771 / 2.549 / 12.329 | 0.309 / 0.982 / 2.255 |
| Mean absolute floor-plane distance (cm) | 1.64 | 1.54 |
| Initial-rest pose variation | jitter present | exactly constant |
| Final-rest pose variation | jitter present | exactly constant |

## Failure modes and residual uncertainty

- Frames around 350-450 contain substantial person occlusion and motion blur; mask
  centroid residuals there are smoothed from neighbouring reliable frames.
- Absolute camera depth is the least certain component. The supplied MoGe-2 depth
  disagrees with the mesh surface by roughly 0.18 m on average because the thin open
  mesh and background mix at mask pixels. The result therefore preserves the
  baseline's calibrated depth and corrects mainly the image-plane and temporal terms.
- The final view is rear-facing and has less distinctive texture. Its image evidence
  is strong enough to reject 180-degree flips, but absolute depth confidence is lower
  than for the initial rest pose.

## Reproducibility

Run:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

`solve.py` reads only the task inputs, uses no network, fixes random seeds, and
recomputes every pose and phase correction from those inputs and the baseline
reconstruction.
