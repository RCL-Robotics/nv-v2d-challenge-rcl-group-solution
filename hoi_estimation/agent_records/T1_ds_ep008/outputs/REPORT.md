# 6-DoF Skinny Wood Chair Trajectory

## Result

`object_poses_world.npz` contains `T_world_mesh` as a `(708, 4, 4)` float64 array. The mapping is the mesh-file frame to the released world frame, using OpenCV camera/world conventions.

The final trajectory uses:

- **Exactly one constant world pose for frames 0--399** (untouched approach + grasp).
- **A smoothed, endpoint-corrected rigid trajectory for frames 400--595** (chair slides toward/under the round table).
- **Exactly one constant world pose for frames 596--707** (chair released and person walks away).

The deterministic entry point is:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Measured wall/GPU runtime for the final run: **195.1 s**.

## Phases found

| Frames | Object phase | Evidence | Confidence |
|---|---|---|---|
| 0--346 | Initial rest, person approaches | Chair mask/bbox and background pixels are static while person moves; contact probability is zero | High |
| 347--399 | Initial rest, person grasps | Contact rises to ~1.0, but frame-to-frame differences on visible chair pixels remain at background-noise level; tracker had a 3.0 cm depth jump at f374 while the image did not move | High |
| 400--595 | Chair slides/pushes | Sustained visible-chair image difference begins at f400; smooth world translation and about 29 deg of yaw accumulate | Medium-high |
| 596--707 | Final rest after release | Visible-chair image difference drops to background level by f596; later contact is zero and mask/bbox stay fixed | High |

These boundaries are fixed constants in `solve.py` with comments, because they were verified visually and against the mask/contact signals.

## Method

1. **Pooled static-pose fitting.** I used the exact textured mesh, given `K` and `T_world_cam`, and a full-resolution differentiable nvdiffrast silhouette. Frames covered by the dilated person mask were excluded from the silhouette score. For each static phase, one SE(3) pose was optimized over 49--100 frames using Powell's method.
2. **Mask-boundary bias control.** Raw SAM2 masks are consistently about 1--2 pixels outside the rendered mesh boundary. The first static fit pooled all frames; a second narrow refinement used one-pixel-eroded masks on unobstructed frames. A depth sweep confirmed that the eroded-mask optimum agrees with the raw pooled pose instead of the earlier, visibly too-close solution induced by mask halo.
3. **Temporal trajectory construction.** Baseline translations/quaternions were smoothed only inside frames 390--606 with a 17-frame Savitzky--Golay filter. The differences between pooled static poses and the baseline at f399/f596 were transferred through frames 400--595 with a C1-continuous (smoothstep) SE(3) correction. This removes the baseline's static depth drift without changing the measured relative motion.
4. **Hypothesis checks.** Identity orientation beats 180-degree flips about every mesh principal axis by a large mask-IoU margin. FoundationPose and CoCoNet rotations were also tested with the corrected translation; the corrected baseline rotation won on the moving segment.

### 180-degree flip check (mean visible-mask IoU)

| Segment | Identity | Local x 180 | Local y 180 | Local z 180 |
|---|---:|---:|---:|---:|
| Initial static | 0.8675 | 0.3345 | 0.3595 | 0.2831 |
| Moving | 0.8567 | 0.3897 | 0.4040 | 0.3857 |
| Final static | 0.8859 | 0.4599 | 0.4141 | 0.5998 |

The local-z flip is the symmetry-looking case and is still clearly rejected, so no segment uses a flipped orientation.

## Validation: baseline vs result

Metrics use all frames in each phase. IoU is between the rendered exact mesh and the SAM object mask after masking out person pixels. Boundary distance is a symmetric chamfer distance in pixels.

| Phase | Baseline IoU | Result IoU | Baseline boundary px | Result boundary px |
|---|---:|---:|---:|---:|
| 0--346 untouched static | 0.7648 | **0.8644** | 2.038 | **0.975** |
| 347--399 grasp static | 0.7937 | **0.8825** | 1.775 | **0.838** |
| 400--595 moving | 0.7965 | **0.8556** | 1.777 | **1.184** |
| 596--707 final static | 0.8197 | **0.8861** | 2.084 | **1.437** |

Additional checks:

| Check | Baseline | Result | Direction |
|---|---:|---:|---|
| Static mean translation step | 6.16 mm/frame | **0 mm/frame** | Correct |
| Static mean rotation step | 0.413 deg/frame | **0 deg/frame** | Correct |
| Moving mean translation second difference | 11.3 mm | **1.03 mm** | Smoother |
| Moving mean rotation step | 0.434 deg/frame | **0.191 deg/frame** | Smoother |
| Minimum hand-keypoint-to-chair distance (contact frames) | 2.67 mm | 3.74 mm | Comparable; within contact uncertainty |
| Mean absolute floor-plane distance of lowest mesh vertex | 1.05--1.76 cm | **0.45--1.17 cm** | More plausible |

The hand-distance metric worsened by about 1 mm but remains below 4 mm, while mask agreement, static consistency, and floor contact all improve substantially.

## Depth decision

I inspected the aligned MoGe-2 maps at frames 0, 100, 200, 300, 350, 375, 390, 400, 450, 500, 550, 600, 620, 650, and 700. Median `MoGe_depth - rendered_mesh_depth` over visible object pixels ranged from about +0.02 m to +0.23 m and frequently exceeded +0.15 m. The maps therefore did not agree with the metric silhouette/floor/human calibration well enough to justify a large depth change for this small object.

The final depth correction is consequently driven by pooled multi-frame silhouette scale, with hand contact and floor-plane checks as secondary evidence. No per-frame MoGe depth is used.

## Confidence and limitations

- **0--399 static:** high confidence. The image is static, hundreds of frames are pooled, and the pose beats per-frame estimates on both mask IoU and boundary distance.
- **400--595 moving:** medium-high confidence. Relative motion comes from the released tracker, but is smoothed and corrected at both ends. The main residual risk is a systematic object-scale/SAM-boundary depth bias.
- **596--707 static:** high confidence. The chair is unoccluded for much of this phase and the pooled pose improves mask and floor checks.
- The GLB texture atlas rendered with visible seams/patchwork under simple unlit sampling, so texture RGB was not used as a final cost. Silhouette, human contact, floor contact, and temporal consistency were more reliable here.
- No ground-truth annotations or challenge trajectory files were accessed.
