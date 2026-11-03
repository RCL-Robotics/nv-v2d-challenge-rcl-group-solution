# Spray-bottle 6-DoF trajectory recovery

## Summary

The output trajectory is a temporally constrained repair of the supplied
`T_world_mesh` baseline.  The main changes are:

1. **Phase segmentation from hand contact.**  The confident hand-contact run
   starts at frame 108 and ends at frame 366.  This gives a table-rest phase
   (0-106), a carried phase (107-368), and a floor-rest phase (379-476), plus a
   short release/settling interval (369-378).
2. **SO(3)-aware branch repair.**  The baseline contains exact-looking
   180-degree spin-flip failures.  The final segment's branch is selected
   automatically by continuity with the pre-release pose, then pooled.  This
   removes the 174-degree jump at frame 369.  Short discontinuous intervals at
   262-269 and 324-344 are replaced by shortest-path SO(3) interpolation.
3. **Temporal filtering.**  Rotations are smoothed as sign-consistent
   quaternions with a 5-frame Savitzky-Golay filter.  Translations use a
   5-frame median filter; the two 1.3-1.4 m tracking spikes at 343-344 are
   interpolated before filtering.
4. **Static-pose pooling.**  Every frame in each rest phase is forced to one
   rigid pose.  The pooled pose is then aligned to the phase's majority object
   mask using the textured mesh renderer.  A coarse search over +/-10% depth
   and +/-6 pixels image-plane translation improved the majority-mask IoU from
   0.812 to 0.879 for the initial rest pose and from 0.694 to 0.820 for the
   final rest pose.
5. **Release bridge.**  The directly observed frames 369-375 are retained;
   frames 376-378 smoothly bridge into the pooled final static pose.  This
   avoids a 10-cm one-frame boundary jump without discarding good image
   evidence during release.

The final floor-rest bottle bottom lies 1.9 cm above the supplied fitted floor
plane (negative signed plane distance), consistent with the released
`foot_plane_dist_stats` tolerance (mean 3.46 cm, std 1.74 cm).  I did not snap
the object to the floor.

## Phases and confidence

| Frames | Phase | Basis | Confidence |
|---|---|---|---|
| 0-106 | Resting on white desk | no contact; masks and translation stable | **High** for pose; high for static constancy |
| 107-323 | Carried, mostly visible | stable contact; visible bottle for almost all frames | **High** for large/visible frames; medium during hand/body overlap |
| 324-344 | Carried behind torso | no object mask in many frames; large baseline rotation jumps | **Low-medium**; rotation/translation paths are interpolated between reliable endpoints |
| 345-368 | Lowering / placement | contact remains high; object visible below hand | **Medium-high**; baseline image agreement is good after branch/outlier repair |
| 369-378 | Release and settling | contact ends; object visible around table leg | **Medium**; image fit retained, final four frames bridged |
| 379-476 | Resting on floor | no contact; image and translation static | **High** for translation and constancy; **medium-low** for residual spin about the long axis because a table/chair leg occludes much of the bottle |

The final 180-degree spin branch was chosen because continuing the
pre-release branch is 180 degrees closer than the raw tracker branch.  The
visible final sliver is too narrow to make a confident independent
texture-only decision, so this is the least certain rotational choice in the
solution.

## Validation against the video

These measures are diagnostics, not ground truth.  Mask IoU is averaged over
all frames in the phase.  Photometric NCC is the per-frame grayscale
correlation over visible object pixels (object mask, excluding person mask and
outside pixels), averaged over frames with at least 10 valid pixels.  "Valid"
counts are reported because some masks disappear under severe occlusion.

| Phase | Frames / valid | Baseline mask IoU | Result mask IoU | Baseline NCC | Result NCC |
|---|---:|---:|---:|---:|---:|
| Initial static 0-106 | 107 / 107 | 0.7918 | **0.8739** | 0.5180 | **0.6698** |
| Carry visible 107-323 | 217 / 215 | 0.7853 | 0.7839 | 0.3401 | **0.3430** |
| Occluded 324-344 | 21 / 4 | 0.2901 | **0.4954** | -0.0595 | **-0.0452** |
| Lower/place 345-368 | 24 / 24 | 0.5989 | **0.6034** | 0.1132 | **0.1232** |
| Release 369-378 | 10 / 10 | 0.7434 | **0.7591** | **0.2666** | 0.1928 |
| Final static 379-476 | 98 / 98 | 0.7130 | **0.8160** | 0.1874 | **0.2092** |
| All frames | 477 / 458 | 0.7564 | **0.7993** | 0.3320 | **0.3725** |

The release phase trades some photometric correlation for substantially better
temporal consistency and a smooth entry into the much stronger pooled final
pose.  The occluded-phase NCC remains low because only four frames have useful
independent mask evidence; its improvement should not be interpreted as strong
orientation validation.

Temporal consistency and physical plausibility:

| Phase | Baseline rotation step p90 / max (deg) | Result rotation step p90 / max (deg) | Baseline translation step p90 / max (m) | Result translation step p90 / max (m) |
|---|---:|---:|---:|---:|
| Initial static | 3.13 / 5.95 | **0 / 0** | 0.024 / 0.060 | **0 / 0** |
| Carry visible | 6.83 / 83.5 | **4.83 / 10.4** | 0.038 / 0.070 | **0.035 / 0.064** |
| Occluded | 170.74 / 175.78 | **1.62 / 2.75** | 0.297 / 1.434 | **0.049 / 0.168** |
| Lower/place | 19.93 / 35.96 | **13.87 / 15.76** | 0.076 / 0.140 | **0.053 / 0.070** |
| Release | 22.54 / 30.28 | **4.08 / 4.08** | 0.024 / 0.031 | 0.027 / 0.027 |
| Final static | 4.53 / 9.78 | **0 / 0** | 0.021 / 0.035 | **0 / 0** |
| All frames | 9.12 / 175.78 | **4.29 / 15.76** | 0.038 / 1.434 | **0.032 / 0.168** |

The largest remaining translation step is 0.168 m at frame 342, where the hand
moves rapidly during the severe-body-occlusion interval.  The baseline has a
1.43 m step in the same interval.

## Runtime and reproducibility

`solve.py` runs offline with the required command and completes in about
**8 seconds** on the allocated GPU.  Static-pose mask fitting is the only GPU
operation and uses about 1.4 GB of GPU memory.  Two independent runs produced
bit-identical arrays (max absolute difference 0.0).  Output SHA-256:
`52ac0d2b6819784546b7338ee4663c0c94119ec4e1dfcdf78f6683c2ca638711`.

## Limitations

- No ground truth was used.
- SAM2 object masks are missing or partial during the torso occlusion and, in
  the final phase, include only the side of the bottle not hidden by furniture.
- The final spin about the bottle's long axis is the weakest rotational degree
  of freedom.  The chosen continuation branch is physically and temporally
  preferred, but not independently strongly observable in the final frames.
- The method intentionally favors rigid rest phases and smooth carried motion;
  if the real object underwent a very fast (>10 deg/frame) wrist rotation that
  the baseline did not measure, that motion would be suppressed.
