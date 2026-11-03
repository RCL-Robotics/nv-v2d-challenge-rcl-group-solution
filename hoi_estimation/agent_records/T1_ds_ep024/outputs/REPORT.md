# Watering-can 6-DoF trajectory — method and validation

## Result

`object_poses_world.npz` contains key `T_world_mesh` as a `(748, 4, 4)`
`float64` array. The same deterministic script reproduces it offline with:

```bash
python /work/outputs/solve.py --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The final end-to-end run took **630.4 s** on the available RTX 6000 Ada GPU.
Random seeds are fixed, and no network, challenge annotation, or ground-truth
data was used.

## Video interpretation and phases

| Phase | Frames | What happens | Confidence |
|---|---:|---|---|
| Rest before interaction | 0–128 | Can rests on the floor; the right hand approaches during the last ~20 frames but the can does not visibly move. | High |
| Grasp transition | 129–159 | Right hand closes on the handle and begins lifting/tilting. | Medium-high |
| Tip / pour | 160–509 | Can is tipped forward by about 75–80° relative to its initial pose and held there. | Medium |
| Return / release | 510–614 | Can returns toward upright and the hand releases. | Medium-high |
| Rest after interaction | 615–747 | Can is stationary again. Relative to frame 0 it has settled with about a 15.4° rotation and 4.5 cm translation. | High |

The right-hand keypoints are `21:41`; in this episode the second supplied
contact-probability column follows those keypoints. Both rest intervals were
visually checked to be truly static, not just unoccluded.

## Method

1. **Per-frame evidence.** The exact mesh is rendered in a fixed 210×180 ROI
   with nvdiffrast, including its texture. Starting from
   `reconstruction.npz:T_world_mesh`, a small left-multiplied SE(3) correction
   is optimized for every frame. The objective combines
   (a) silhouette agreement after removing the person mask, and
   (b) normalized high-pass texture correlation inside the predicted object.
   This corrects the systematic undersizing and much of the per-frame drift.
2. **Static-phase pooling.** Frames 0–128 and 615–747 are constrained to one
   world pose each. A pose is first initialized from rotation averaging and
   median translation, then fit to the median RGB crop and mean mask over the
   whole phase. A soft floor-plane term guards against implausible depth. This
   removes all frame-to-frame tracking jitter in the rest phases.
3. **Temporal decoding.** The dynamic per-frame poses are smoothed in
   quaternion + translation space with a quadratic Savitzky–Golay filter
   (window 9), then smoothly blended into the two pooled rest poses.
4. **Re-fit and final smoothing.** Frames 129–614 are re-optimized with a soft
   IoU silhouette objective plus texture correlation. A second, shorter
   Savitzky–Golay pass (window 7) removes residual observation noise. This
   lowers p90 frame-to-frame motion from 5.37 cm / 3.36° (baseline) to
   1.09 cm / 1.88°, while changing binary mask IoU by less than 0.002.
5. **Orientation sanity check.** 180° flips about the three mesh axes were
   scored against the masks. They were consistently worse (for example,
   frame 0 SAM IoU 0.662 for the baseline versus 0.514–0.577 for the three
   flips), so no symmetry flip was adopted. The recovered relative rotation
   follows the visible tip-and-return motion rather than a constant offset.

## Validation without ground truth

The supplied SAM masks are somewhat larger than the visible green object
boundary, so two mask scores are reported:

- **SAM IoU:** supplied object mask with dilated person pixels excluded.
- **Color IoU:** independent green-object segmentation in the ROI
  (`G - max(R,B) > 20`, restricted to a 4 px dilation of the SAM mask). This is
  used only for validation, not fitting.

### Appearance and centroid checks

| Phase | SAM IoU baseline → result | Color IoU baseline → result | Mask centroid error (px) baseline → result |
|---|---:|---:|---:|
| Rest before | 0.713 → **0.836** | 0.764 → **0.862** | 2.23 → **1.55** |
| Grasp transition | 0.672 → **0.794** | 0.727 → **0.799** | 4.87 → **2.61** |
| Tip / pour | 0.563 → **0.686** | 0.591 → **0.716** | 10.21 → **6.36** |
| Return / release | 0.624 → **0.779** | 0.654 → **0.789** | 6.99 → **2.41** |
| Rest after | 0.691 → **0.839** | 0.729 → **0.862** | 3.02 → **1.33** |
| **All frames** | **0.625 → 0.757** | **0.660 → 0.781** | **6.88 → 3.93** |

Both independent appearance measures improve in every phase. The improvement
is largest during the heavily occluded tip/pour phase.

### Motion and physical consistency checks

Rotation/translation step statistics are over the dynamic interval 129–614.
Floor distance is the minimum signed distance of the mesh convex hull to the
released floor plane; the plane fit itself has several-cm uncertainty.
Hand distance is the nearest sampled mesh vertex to the right-hand keypoints.

| Phase | Rotation step (deg/frame) baseline → result | Translation step (cm/frame) baseline → result | Median floor minimum (cm) baseline → result | Mean right-hand distance (cm; <3.5 cm coverage) baseline → result |
|---|---:|---:|---:|---:|
| Rest before | 1.38 → **0.00** | 1.73 → **0.00** | −0.84 → 5.55 | — |
| Grasp transition | 2.37 → **1.63** | 2.07 → **0.94** | −1.69 → 1.43 | 0.94 (100%) → 3.26 (61%) |
| Tip / pour | 1.77 → **0.64** | 2.13 → **0.47** | 5.63 → 4.84 | 1.03 (98%) → **0.45 (99%)** |
| Return / release | 2.37 → **1.28** | 1.74 → **0.58** | 3.96 → **3.85** | 2.27 (75%) → **1.23 (96%)** |
| Rest after | 1.43 → **0.02** | 1.96 → **0.00** | −2.00 → 4.43 | — |
| Dynamic 129–614 | 1.94 → **0.84** | 2.04 → **0.52** | — | — |

The result also reduces p90 dynamic jitter from 3.36° to 1.88° in rotation and
from 5.37 cm to 1.09 cm in translation. Mean right-hand surface distance falls
from 1.03 cm to 0.45 cm during pouring and from 2.27 cm to 1.23 cm during return/release. The earliest part
of the grasp transition (roughly frames 129–145) is less well constrained:
the object silhouette and the reconstructed hand disagree there, and the
mask/color-supported pooled pose was retained rather than snapping to the
hand. Static phase poses are exactly constant.

## Changes from the baseline

- Removed per-frame jitter in both rest phases by replacing 129 and 133 frames
  respectively with one pooled rigid pose.
- Corrected the baseline's persistently undersized object projection. The
  static poses move toward the camera and are about 4–5° different in
  orientation; the pooled initial and final poses differ by 15.4° and 4.5 cm,
  consistent with tipping and settling.
- Smoothed the complete manipulation trajectory and eliminated one-frame
  rotation/translation jumps while preserving the tip-and-return motion.
- Improved the right-hand contact during the carried interval.

## Remaining uncertainty

The largest remaining uncertainty is absolute depth along the camera ray. SAM
boundaries are a few pixels larger than the visible green object, so a purely
silhouette-driven fit can over-correct distance. Here depth is supported by both the SAM and independent green-color
silhouettes and by the floor-plane soft constraint; hand proximity gives
strong additional support during the main pour and return intervals. The
reconstructed hand is less consistent with the silhouette during the initial
reach, and MoGe object depth is noisy at this object size, so neither was used
as a hard constraint. Confidence is consequently **medium** for absolute depth
and **high** for temporal consistency and relative motion. Rotation is supported
by the asymmetric spout/handle silhouette, texture correlation, and the
rejected 180° alternatives; confidence is **medium-high** during manipulation
and **high** in the pooled rest phases.
