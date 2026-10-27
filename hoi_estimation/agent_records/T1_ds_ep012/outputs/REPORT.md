# Rigid object trajectory reconstruction

## Result

`object_poses_world.npz` contains `T_world_mesh` with shape `(700, 4, 4)` and
dtype `float64`, mapping the supplied GLB mesh coordinates directly into the
released world frame.

The final trajectory preserves the baseline evidence in ambiguous cases, but
uses the video-level constraints that the single-frame reconstruction ignores:
two genuinely rigid static phases, smooth object motion while carried, and
metric depth inferred from a pooled silhouette fit.

No ground-truth or challenge annotation was used.

## Phases

Frame differencing inside the visible object region (person pixels excluded)
gave a median absolute gray-level difference below 1 during both static
phases.  It rose to about 2-4 levels during manipulation.

| Phase | Frames | Description | Confidence |
|---|---:|---|---|
| Initial static | 0-195 | Platform resting on the floor, then touched while still stationary | High |
| Manipulation | 196-581 | Platform lifted, rotated about 180 degrees, and lowered | Medium-high |
| Final static | 582-699 | Overturned platform at rest on the floor | High |

The final relative orientation is a 179.91 degree rotation from the initial
pose about world-frame axis approximately `[-0.114, 0.004, -0.993]`, which
matches the instructed turn-over action.  In both static phases the output is
one exactly constant rigid transform.

## Method

1. **Phase segmentation.**  The boundaries were inspected in the video, object
   mask, person mask, and hand/object separation signals.  They are encoded as
   commented constants in `solve.py` as permitted by the task.  The initial
   pose pools frames 20-149 and the final pose pools frames 590-699; these
   windows avoid heavy hand/body occlusion while retaining clean static views.
2. **Temporal smoothing.**  Baseline translations and sign-aligned quaternions
   are Gaussian-smoothed (sigma 2 frames).  Static poses are then imposed and
   smoothly interpolated into the moving segment.
3. **Silhouette correction.**  The mesh is rendered with nvdiffrast at
   768x576.  Object masks are compared after removing person pixels.  Smooth
   corrections are applied to camera-frame x/y translation to remove the
   2-D centroid drift that remains in the tracker.
4. **Metric depth correction.**  The object mask is eroded by one pixel before
   depth scoring to compensate SAM2's approximately one-pixel boundary
   dilation.  For sparse anchor frames, the pose is moved along its camera ray
   (translation scaled without scaling the rigid mesh) and the depth scale that
   maximizes mask IoU is selected.  The scales are interpolated and smoothed
   over time, then tapered to the static phases.
5. **Rotation refinement.**  A sparse coordinate search tests 1-degree and
   2-degree local x/y/z corrections.  The sparse corrections are interpolated
   and strongly smoothed (sigma 10 frames) before being applied; this avoids
   independent per-frame mask overfitting.
6. **Final refit.**  Translation-centroid and depth passes are repeated after
   the rotation correction.  The pooled static transforms are re-imposed at
   the end.

## Validation

Mask IoU was measured at 768x576 over all 700 frames, with person pixels
excluded from both prediction and target.  These metrics are diagnostic only;
the optimization does not use held-out ground truth.

| Phase | Baseline mask IoU | Result mask IoU | Change |
|---|---:|---:|---:|
| Initial static (0-195) | 0.8780 | 0.8921 | +0.0140 |
| Manipulation (196-581) | 0.8460 | 0.9112 | +0.0652 |
| Final static (582-699) | 0.9057 | 0.9179 | +0.0122 |
| **All frames** | **0.8650** | **0.9070** | **+0.0420** |

An independent full-resolution check on every 14th frame (50 frames) gave
baseline/result IoU 0.8674/0.9024 overall, with the same positive change in
each phase.

Mean absolute silhouette-centroid error also decreased from 1.23 px x /
2.53 px y to 0.15 px x / 0.15 px y during manipulation.  Predicted/visible
mask area ratio increased from 0.879 to 0.930 after correcting the systematic
depth error.  The median camera-z correction during manipulation was
-8.9 cm (10th-90th percentile: -19.5 to -1.1 cm), with the correction tapering
to zero in both static phases.

Temporal checks over the moving segment:

| Metric | Baseline | Result |
|---|---:|---:|
| Median rotation step | 0.673 deg/frame | 0.485 deg/frame |
| 95th percentile rotation step | 1.391 deg/frame | 0.954 deg/frame |
| Maximum rotation step | 2.022 deg/frame | 1.879 deg/frame |
| Median translation step | 1.62 cm/frame | 0.44 cm/frame |
| 95th percentile translation step | 7.78 cm/frame | 1.06 cm/frame |
| Maximum translation step | 16.60 cm/frame | 1.98 cm/frame |

The output's minimum mesh-to-floor plane distances are -2.39 cm initially and
-2.00 cm finally.  These lie within the supplied floor-fit error scale
(`foot_plane_dist_stats` median 2.86 cm, std 1.92 cm), so the object was not
artificially snapped to the plane.

## Reproducibility and runtime

Command:

```bash
python /work/outputs/solve.py \
  --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The measured end-to-end runtime in this container was **70.2 seconds** on the
available CUDA GPU.  The script reads only `camera.json`,
`reconstruction.npz`, `masks.h5`, and `object_mesh.glb` from `--inputs`; it
does not use the network, scratch files, `/weights`, or `/pipeline`.  Random
seeds are fixed.  An independent rerun produced a bit-identical trajectory (0.0 mm translation
and 0.0 degree maximum difference).
