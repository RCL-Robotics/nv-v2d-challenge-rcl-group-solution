# Step-ladder 6-DoF trajectory recovery

## Result

- Output: `object_poses_world.npz`, key `T_world_mesh`, shape `(719, 4, 4)`, `float64`.
- The result is a trajectory correction, not a per-frame re-registration. It retains the baseline's strong moving-object evidence, but replaces tracker noise/drift during the two rest phases with one pooled pose per phase and joins those anchors to a lightly smoothed drag trajectory.
- No ground-truth data or challenge annotations were used.
- Final offline run: **23.8 s wall time** (`solve.py` reports 21.7 s), GPU peak well below 1 GB. A repeat run reproduced the output byte-for-byte.

## Phases found

Boundaries were selected from the object-mask motion, the right-hand/object contact time, and a per-frame comparison of the baseline silhouette residual with the residual of the pooled static pose. The pooled pose is initially much better than the baseline, becomes worse immediately as motion starts, and becomes better again when motion stops.

| Phase | Frames | Interpretation | Confidence |
|---|---:|---|---|
| Initial rest | 0-200 | Ladder untouched on the floor; person approaches | High |
| Drag onset | 201-300 | Right hand starts pulling/rotating the ladder | Medium-high |
| Drag across | 301-450 | Ladder dragged across the bounding area | Medium |
| Reposition | 451-555 | Continuing drag/turn while heavily occluded by the person | Medium-high |
| Final rest | 556-718 | Ladder released; person walks away | High |

## What changed and why

1. **Static phases were pooled.** Frames 0-200 and 556-718 show a stationary object with a stationary camera. The baseline nevertheless changes pose slowly in both phases (final-phase standard deviations up to 6.1 cm and a 4.8 degree rotation span). A single pose was fitted to many clean frames using the exact mesh silhouette.
2. **Static fitting used a deterministic multi-frame objective.** At the initial phase, frames 88-150 were fitted at half resolution to avoid giving extra weight to the 1-3 px SAM boundary dilation. At the final phase, frames 660-718 were fitted at native resolution. The objective is symmetric outer-contour chamfer distance after excluding person pixels.
3. **The dynamic trajectory was only lightly modified.** Baseline rotations were converted to sign-consistent quaternions and translations to positions, then filtered with a 7-frame Savitzky-Golay filter. This reduces tracker jitter while keeping the actual drag motion. The added bias relative to the baseline is small: mean mesh-point displacement is 0.9 cm during drag onset, 0.5 cm during the central drag, and 1.8 cm during repositioning.
4. **Static anchors were joined locally.** Over 20 frames after the first rest phase and 22 frames before the final rest phase, a rigid correction fades between the smoothed baseline and the corresponding pooled pose. The middle of the drag is not globally warped to the anchors.
5. **Symmetry flips were tested and rejected.** At frame 0 the identity pose has 1.34 px symmetric outer-contour chamfer; 180-degree flips about the mesh z/x/y axes give 6.01/14.05/10.83 px. In the drag phase the identity pose is 0.2-4.6 cm from the right-hand keypoints, while the upright 180-degree flip is 1.2-2.2 m away. Texture renders also favoured the identity pose. No flip was introduced.
6. **Depth was not copied from MoGe nor snapped to the floor.** MoGe-2 depth showed systematic offsets and was unreliable on the thin ladder frame. The static image fit gives a floor-minimum median of 0.17 cm at the start and 2.06 cm at the end, both within the released plane-fit tolerance, so the image-supported pose was retained.

## Validation

Metrics below are computed automatically from the supplied masks/keypoints and the output trajectory. Chamfer is the mean symmetric outer-silhouette contour distance in pixels after removing a 7 px person dilation. Hand distance is the nearest right-hand MHR keypoint to the posed mesh surface. Floor values are the 50th/90th percentile of the mesh's minimum signed distance to the released floor plane (cm). Acceleration uses consecutive poses and is shown as translation p50/p90 in mm/frame² followed by rotation p50/p90 in deg/frame². Mask chamfer is elevated during heavy person occlusion because SAM object masks are incomplete; the baseline/result comparison is still paired on the same pixels.

| Phase | Baseline chamfer / IoU | Result chamfer / IoU | Baseline hand p50/p90 cm | Result hand p50/p90 cm | Baseline floor p50/p90 cm | Result floor p50/p90 cm | Baseline accel | Result accel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial rest | 1.182 / 0.637 | **0.768 / 0.720** | 22.53 / 46.45* | 24.35 / 46.37* | 0.78 / 1.60 | **0.17 / 0.17** | 9.89/25.34, 0.54/1.01 | **0/0, 0/0** |
| Drag onset | 1.068 / 0.742 | **1.047 / 0.745** | 0.37 / 3.02 | **0.29 / 2.93** | -1.52 / -0.62 | **-1.26 / 0.00** | 9.66/19.51, 1.17/2.39 | **2.57/5.02, 0.44/0.69** |
| Drag across | 2.572 / 0.452 | **2.563 / 0.453** | 0.57 / 3.88 | 0.59 / **3.79** | -2.38 / -0.61 | **-2.34 / -0.63** | 10.57/22.48, 1.18/2.64 | **3.09/7.73, 0.38/0.81** |
| Reposition | 2.045 / 0.465 | **2.019 / 0.471** | 1.15 / 2.28 | **1.03 / 2.02** | -1.88 / -0.97 | **-1.83 / 0.05** | 10.83/31.98, 1.19/2.48 | **3.36/7.81, 0.31/0.65** |
| Final rest | 0.876 / 0.651 | **0.522 / 0.701** | 61.04 / 125.99* | 55.24 / 125.68* | -0.17 / 2.12 | 2.06 / 2.06 | 16.89/41.85, 0.93/2.00 | **0/0, 0/0** |

\* In rest phases the person is far from the object; the hand distance is not a contact metric there.

Additional checks:

- Initial-phase pose drift: baseline translation std `(0.44, 0.88, 2.15)` cm and rotation span `2.24 deg`; result is exactly constant.
- Final-phase pose drift: baseline translation std `(1.16, 1.16, 6.09)` cm and rotation span `4.77 deg`; result is exactly constant.
- Mean mesh-point displacement from baseline: initial rest 2.66 cm, drag onset 0.88 cm, central drag 0.46 cm, reposition 1.82 cm, final rest 8.22 cm. These changes are concentrated where the static evidence contradicts per-frame tracker drift.
- All output matrices are finite, orthonormal (max orthogonality error `8.9e-16`), and have determinant 1.

## Remaining uncertainty

The central drag is the hardest region because the hand/body occlude large portions of the ladder and SAM masks are incomplete. There the result deliberately stays close to the baseline and mainly removes high-frequency pose jitter. The largest possible residual risk is a small depth bias during that interval; global endpoint warping made the silhouette substantially worse, so it was not used. Rest-phase poses have high confidence; dynamic-phase poses are medium confidence, with the largest uncertainty in the heavily occluded reposition interval.
