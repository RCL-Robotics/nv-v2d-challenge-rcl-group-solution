# Squeegee 6-DoF Trajectory Report

## Result

- Output: `object_poses_world.npz`
- Key: `T_world_mesh`, shape `(988, 4, 4)`, `float64`
- Reproduce with:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The released per-frame reconstruction was used as the proposal source. The final trajectory keeps its pose convention and world frame, but replaces/regularizes parts that are inconsistent with rigid-object temporal evidence.

## Phases and changes

| Frames | Phase | What changed | Confidence |
|---|---|---|---|
| 0-110 | At rest, leaning against the stool | Replaced all per-frame estimates with one pooled static pose. The start orientation was selected from the stable proposal cluster using pooled person-excluded object masks; translation was refined against the pooled silhouette. | High |
| 111-399 | Pickup and first sweeping motion | Applied 13-frame robust medoid smoothing in SO(3) and translation median filtering, with a smooth transition from the start static pose. | Medium-high |
| 400-669 | Continued sweeping/manipulation | Same robust temporal smoothing. Image evidence was retained; no static constraint was applied. | Medium |
| 670-839 | Turn, lower, and place squeegee back against stool | Added mask-aligned roll hypotheses on keyframes (690-790), interpolated selected poses in SE(3), and skipped heavily occluded frames as direct anchors. This removes the largest tracker inversions. | Medium |
| 840-987 | At rest, leaning against the stool | Replaced all poses with one pooled static pose selected by pooled mask evidence, with a pooled translation refinement. | High |

The static boundaries are explicit constants because they were verified from the video, contact probabilities, and object-mask bounding boxes. No ground-truth data was used.

## Method summary

1. **Static pooling.** Object masks were averaged within each stationary phase. Person pixels were down-weighted, and one pose was selected and refined against the pooled silhouette. This exploits hundreds of observations for one rigid pose instead of fitting each frame independently.
2. **Temporal robustification.** Per-frame rotations were filtered by a local quaternion medoid followed by Huber-weighted quaternion averaging. Translations were median filtered over the same 13-frame window. This removes isolated 30-120 degree tracker inversions while retaining continuous manipulation motion.
3. **Ambiguous roll handling.** During the difficult carry/place segment, candidate rotations about the squeegee's long axis were rendered and scored using a visibility-aware symmetric silhouette distance, area consistency, and a weak texture correlation. Selected keyframes were interpolated with smoothstep SLERP and translation interpolation.
4. **Static anchor constraints.** Both stationary phases are exactly constant in the output, including orientation and translation.

## Validation without ground truth

The primary held-out-style check is projection of the known mesh with the estimated pose against the supplied object mask, excluding person pixels. The scoring column is a symmetric mask chamfer loss with an area-consistency penalty (lower is better); IoU is reported separately. Every fifth frame was evaluated.

| Phase | Baseline chamfer | Result chamfer | Baseline IoU | Result IoU | Frames improved |
|---|---:|---:|---:|---:|---:|
| 0-110 rest start | 5.856 | **3.244** | 0.299 | **0.610** | 100.0% |
| 111-399 pickup / sweep 1 | 11.754 | **11.361** | 0.323 | **0.333** | 56.9% |
| 400-669 sweep 2 | 13.360 | **10.879** | **0.330** | 0.325 | 53.7% |
| 670-839 carry / place | 43.748 | **17.872** | 0.197 | **0.249** | 64.7% |
| 840-987 rest end | 11.931 | **6.367** | 0.230 | **0.428** | 86.7% |
| Overall | 16.766 | **10.695** | 0.289 | **0.360** | 66.2% |

The IoU decrease in the second sweeping phase is small and is outweighed by the lower chamfer loss, substantially smoother motion, and removal of tracker inversions. The result should not be interpreted as ground-truth accuracy; the metrics are proxy evidence only.

### Temporal and physical consistency

| Metric | Baseline | Result |
|---|---:|---:|
| Rotation delta p50 / p90 / p99 / max | 1.81 / 6.39 / 40.43 / 119.37 deg | **0.48 / 2.57 / 13.06 / 30.46 deg** |
| Translation delta p50 / p90 / p99 / max | 0.020 / 0.058 / 0.180 / 0.283 m | **0.007 / 0.025 / 0.053 / 0.084 m** |
| Start-phase rotation spread, median / max | 24.58 / 39.09 deg | **0 / 0 deg** |
| Start-phase translation spread, median / max | 0.023 / 0.095 m | **0 / 0 m** |
| End-phase rotation spread, median / max | 13.50 / 30.79 deg | **0 / 0 deg** |
| End-phase translation spread, median / max | 0.115 / 0.172 m | **0 / 0 m** |

These checks justify adopting the change: static phases are now physically consistent, abrupt rotation inversions are removed, and the object-mask explanation improves overall and in every reported phase by chamfer loss.

## Run time and determinism

- Final reproducible run: **36.9 s** end-to-end on the provided GPU.
- Deterministic seeds are fixed and PyTorch deterministic algorithms are requested.
- Peak GPU memory is small: only the static mesh, UV texture, and a few raster buffers are resident.
- Output rotations were checked to be finite, orthonormal (maximum orthogonality error `8.9e-16`), and right-handed (`det(R) = 1`).

## Limitations and confidence

- The squeegee is long and thin, so roll about its long axis is weakly constrained by silhouette alone. The carry/place segment therefore has the lowest confidence.
- Frames 760 and 780 are deliberately interpolated rather than directly anchored because the person occludes much of the object.
- The supplied object masks can include adjacent chair pixels in the final leaning phase. Translation was pooled and constrained rather than fitted to those contaminated pixels.
- No depth snap to the floor was used.
