# Round-table 6-DoF trajectory recovery

## Summary

The final trajectory is produced by `/work/outputs/solve.py` and stored as
`/work/outputs/object_poses_world.npz` under `T_world_mesh` with shape
`(993, 4, 4)` and dtype `float64`.

The main failure mode in the supplied reconstruction is frame-wise angular
jitter, including a 28.8 degree one-frame rotation jump. The result robustly
smooths the released trajectory, searches locally for the pose that best
explains the object silhouette, refines it with rendered texture gradients
under a temporal acceleration prior, and pools the two untouched static
phases. No ground truth or challenge annotations were used.

## Phases and confidence

The static boundaries are derived from the supplied contact signal: first
touch is frame 96 and the last contact is frame 931. The object is upright and
untouched before/after these boundaries. Within the manipulation interval, the
phase boundaries below were selected from the object's tilt profile and checked
visually against the video.

| Frames | Phase | Confidence | Reason |
|---|---:|---|---|
| 0-95 | Static upright, start | High | 96-frame pooled pose; clear silhouette and stable appearance |
| 96-175 | Engage / initial roll | Medium | Mask improves clearly, but texture gradient and MoGe depth partially disagree |
| 176-399 | Lower onto side | High | Mask and texture improve; MoGe residual also improves |
| 400-519 | Flip wheels upward | Medium-high | Large mask gain; depth cue disagrees during the fast roll |
| 520-599 | Upside-down hold | High | Large object, consistent mask/texture, stable depth residual |
| 600-769 | Rotate back onto side | High | Large mask improvement, especially at the former jump near frame 604 |
| 770-829 | Side hold | High | Very large object silhouette and the largest mask gain |
| 830-931 | Return upright / release | High | Mask and texture both improve consistently |
| 932-992 | Static upright, end | Medium-high | 61-frame pooled pose; mask/texture improve, MoGe depth is less favorable |

## Method

1. **Contact and static-phase segmentation.** The supplied hand-contact signal
   gives untouched intervals `0:96` and `932:993`. A single pooled rotation and
   translation is used in each interval.
2. **Continuous-quaternion smoothing.** Rotations are represented as
   hemisphere-continuous quaternions and smoothed with a Savitzky-Golay filter;
   translations are smoothed in xyz.
3. **Mask-driven rigid search.** A textured mesh is rendered at 384x288 with
   nvdiffrast. Four coordinate-descent rounds search per-frame image-plane
   translation, translation along the camera ray, and small rotations about
   the camera axes and world vertical axis. Selection is by visible-object mask
   IoU.
4. **Appearance and temporal refinement.** The mesh texture is rendered and
   compared with the video using normalized gradient correlation plus a smaller
   raw-color NCC term. This data score is combined with an explicit penalty on
   rotational and translational acceleration, so ambiguous frames follow the
   trajectory instead of fitting mask noise. Three further coordinate-descent
   rounds are performed.
5. **Final regularization.** A second quaternion/xyz smoothing pass removes
   frame-switch noise while preserving the large corrections at the baseline's
   former jumps. The two static intervals are pooled over all frames.
6. **Symmetry/flip check.** All trajectory frames were compared against
   180-degree flips about the mesh x/y/z axes and the world vertical axis. No
   flipped pose improved mask IoU by more than 0.02 on any frame. Mean IoU for
   the flips was 0.29-0.68, versus 0.863 for the result.

## Validation without ground truth

The validation metrics are recomputed from the inputs, not from ground truth.

- **Mask IoU**: visible object mask, rendered at 768x576; supplied SAM2 object
  masks are the targets.
- **Gradient NCC**: normalized correlation of image gradients between the
  textured mesh render and video, averaged on visible object pixels at 384x288.
- **Depth residual**: median absolute camera-z difference to the supplied
  aligned MoGe-2 depth on visible object pixels. This is a diagnostic only; the
  baseline already contains a separately calibrated world-depth correction and
  MoGe has a systematic mesh-to-map offset of order 10 cm.

| Phase | Frames | Mask IoU base / result | Grad-NCC base / result | Depth residual cm base / result |
|---|---:|---:|---:|---:|
| Static upright, start | 0-95 | 0.804 / 0.808 | 0.575 / 0.568 | 15.9 / 15.9 |
| Engage / initial roll | 96-175 | 0.761 / 0.807 | 0.607 / 0.563 | 5.1 / 17.3 |
| Lower onto side | 176-399 | 0.879 / 0.917 | 0.457 / 0.556 | 12.5 / 12.0 |
| Flip wheels upward | 400-519 | 0.772 / 0.816 | 0.186 / 0.293 | 8.2 / 13.4 |
| Upside-down hold | 520-599 | 0.840 / 0.874 | 0.250 / 0.289 | 12.8 / 12.8 |
| Rotate back onto side | 600-769 | 0.800 / 0.888 | -0.047 / 0.017 | 7.4 / 7.7 |
| Side hold | 770-829 | 0.751 / 0.866 | -0.309 / 0.029 | 2.7 / 6.3 |
| Return upright / release | 830-931 | 0.734 / 0.811 | 0.024 / 0.301 | 15.1 / 15.5 |
| Static upright, end | 932-992 | 0.782 / 0.826 | 0.421 / 0.517 | 13.6 / 15.5 |
| **All frames** | **0-992** | **0.804 / 0.858** | **0.252 / 0.352** | **11.4 / 12.9** |

The mask improvement is not caused by a small number of extreme frames:
921 of 993 frames improve in IoU, and the median per-frame improvement is
0.043. The largest repairs are at the former tracker jumps: frame 283 improves
from IoU 0.652 to 0.956, frame 757 from 0.587 to 0.862, and frame 881 from
0.400 to 0.694.

### Trajectory and physical checks

| Check | Baseline | Result |
|---|---:|---:|
| Mean rotation step | 1.379 deg | 0.628 deg |
| 95th percentile rotation step | 4.483 deg | 1.684 deg |
| Maximum rotation step | 28.840 deg | 9.522 deg |
| 95th percentile rotation acceleration | 7.035 deg | 0.556 deg |
| 95th percentile translation step | 0.0278 m | 0.0122 m |
| Maximum translation step | 0.1026 m | 0.0218 m |

The result remains in contact with the reconstructed hands: on frames with
contact probability above 0.5, the median nearest mesh-to-hand-keypoint
distance is 5.0 mm (right) and 6.1 mm (left), versus 4.2 mm and 5.0 mm for the
baseline. Mean distances are 16.9 mm and 14.2 mm, versus 18.0 mm and 14.0 mm.

The static table has a minimum signed floor-plane distance of -1.71 cm at the
start and -1.49 cm at the end, within the released floor fit tolerance and
consistent with the floor being a fit rather than another pose constraint.

### Depth decision

No direct depth snap was applied. The baseline's world-depth calibration was
retained as the anchor. The local search did move poses radially when the
object's image scale changed, which is reliable here because the manipulated
object is large (median projected mask area 12k-53k pixels, roughly 123-259 px
equivalent diameter). Those radial changes were accepted only where the mask
and rendered appearance jointly improved. In the engage and flip phases MoGe
reports a several-cm disagreement, so depth confidence in those phases is lower
even though the image-space evidence is stronger; this is the main residual
risk of the submitted trajectory.

## Changes adopted

- Removed frame-wise tracker jumps by robust quaternion/xyz smoothing and a
  temporal acceleration term.
- Corrected local image-plane position and scale where the original pose did
  not explain the visible object.
- Recovered smoother and better appearance-consistent rotations through the
  large side-on and upside-down phases.
- Replaced jittering static poses with one pooled pose per static phase.
- Retained the baseline where the evidence margin was weak, particularly the
  start-phase depth.

## Reproducibility

Command run once at the end:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Measured runtime: 274.4 seconds on one RTX 6000 Ada GPU, with deterministic
settings and no network access. The output was checked for NaN/Inf, shape
`(993,4,4)`, dtype `float64`, and rigid-transform orthonormality (maximum
rotation orthogonality error `1.1e-15`).
