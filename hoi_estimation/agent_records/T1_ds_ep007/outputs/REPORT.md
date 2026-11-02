# Monocular iron trajectory recovery

## Summary

The supplied reconstruction was not usable as a pose prior for this episode. Its
object mask at frame 0 was a box prompt covering both the iron and the tripod
behind it; SAM2 selected the tripod. The baseline pose consequently renders on
the tripod/background and has no overlap with the actual iron. I therefore
re-segmented the iron and recomputed the complete trajectory rather than
blending with the baseline.

Final output: `object_poses_world.npz`, key `T_world_mesh`, shape `(896,4,4)`,
dtype `float64`.

## Phases found (verified from the corrected masks and video)

| Frames | Phase | Treatment |
|---|---|---|
| 0–159 | Iron lies untouched on the round table | One constant world pose |
| 160–659 | Right-hand slide back and forth | Fixed lying rotation; per-frame translation/depth |
| 660–794 | Iron is rotated up to stand vertically | Monotone slerp from lying orientation to final upright orientation; Viterbi decoding over rotation progress and depth |
| 795–895 | Iron stands still after release | One constant world pose |

The object mask is unchanged to within SAM2 jitter through frame 159. The first
visible object displacement is at frame 160. The stand-up starts subtly near
frame 670 and is essentially complete by frame 795.

## Method

1. **Object re-segmentation.** SAM2.1-Hiera-L was prompted on frame 0 with a
   tight box `(665,610)-(780,675)`, which contains the iron but excludes the
   tripod. The resulting video mask tracks the iron through the complete
   manipulation. Tiny disconnected blobs were discarded; person pixels came
   from the supplied person mask and were treated as occluders, not object
   background.
2. **Render-and-compare evidence.** The exact textured GLB was rendered with
   `nvdiffrast` using the released camera intrinsics and `T_world_cam`.
   Hypotheses were scored by person-aware silhouette disagreement, visible-region
   RGB L1, and normalized colour correlation. The mask term excludes rendered
   object pixels under the person mask so hand occlusion does not force the
   object away.
3. **Static and sliding anchors.** The lying orientation and the final upright
   orientation were selected from multi-start render searches and checked by
   textured overlays and SIFT-affine inlier counts across several static frames.
   During sliding, the iron's rotation was held fixed while translation and
   camera depth were re-estimated per frame. This removes the baseline's
   frame-to-frame rotation flips and depth jumps.
4. **Stand-up decoding.** The two anchor rotations are connected by quaternion
   slerp. Every frame in 660–794 was scored over rotation progress and depth;
   a Viterbi pass enforced monotone stand-up progress and smooth depth. The
   decoded path follows the observed gradual lift rather than independent
   per-frame pose guesses.
5. **Temporal cleanup.** The two untouched intervals use one pose each. The
   dynamic translation was passed through a 7-frame median filter followed by a
   Savitzky–Golay filter; smoothstep blends were applied at phase boundaries.
   All rotations were projected back to SO(3).

No ground truth or challenge pose annotation was used.

## Validation without ground truth

Metrics below are computed on every frame against the corrected SAM2 iron mask.
Person-covered pixels are ignored when charging rendered false positives. RGB
values are measured only on the visible intersection of rendered and target
masks. Baseline and result use identical rendering, camera, mesh, masks, and
scoring code.

| Phase | Method | Mean mask IoU | Mean RGB L1 | Mean RGB NCC |
|---|---|---:|---:|---:|
| 0–159 static lying | Baseline | 0.000 | 0.000 | 0.000 |
| 0–159 static lying | **Result** | **0.799** | **0.232** | **0.824** |
| 160–659 sliding | Baseline | 0.000 | 0.000 | 0.000 |
| 160–659 sliding | **Result** | **0.840** | **0.239** | **0.811** |
| 660–794 stand-up | Baseline | 0.000 | 0.000 | 0.000 |
| 660–794 stand-up | **Result** | **0.828** | **0.250** | **0.769** |
| 795–895 static upright | Baseline | 0.000 | 0.000 | 0.000 |
| 795–895 static upright | **Result** | **0.856** | **0.248** | **0.750** |
| All 896 frames | Baseline | 0.000 | 0.000 | 0.000 |
| All 896 frames | **Result** | **0.833** | **0.241** | **0.800** |

The baseline's rendered mask has no pixels in common with the corrected iron
mask, hence the zero RGB terms are reported for the empty intersection. The
supplied baseline was fitted to the mask that included the tripod, so this is a
mask/data-association failure rather than a small appearance residual.

Temporal plausibility over all adjacent frame pairs:

| Method | Rotation step p50 / p90 / p99 / max (deg) | Translation step p50 / p90 / p99 / max (m) |
|---|---|---|
| Baseline | 0.000 / 21.36 / 96.11 / 153.18 | 0.0016 / 0.0666 / 0.1210 / 0.3232 |
| **Result** | **0.000 / 0.000 / 2.05 / 2.05** | **0.0050 / 0.0157 / 0.0283 / 0.0432** |

The finite translation steps include intentional motion and the short phase
blends. Static intervals are exactly constant.

## Confidence

- **0–159, lying at rest — high.** The mask is unchanged for 160 frames and the
  pose is pooled into one transform; silhouette IoU is 0.799 and RGB correlation is 0.824.
- **160–659, sliding — high.** The visual orientation is clearly constant and
  all per-frame fits have IoU around 0.80–0.90. The largest uncertainty is the
  metric depth, which is constrained by the known 0.305 m mesh and silhouette
  size rather than the unreliable object-only MoGe depth; the pooled phase IoU
  is 0.840.
- **660–794, stand-up — medium-high.** Hand occlusion is strongest here. The
  monotone trajectory and slerp path prevent isolated rotation flips; mean IoU
  is 0.828.
- **795–895, upright at rest — high.** The pose is pooled over 101 untouched
  frames. The final texture orientation was checked across four static frames
  with SIFT-affine inlier counts and the mean silhouette IoU is 0.856.

## Runtime

The final offline invocation of:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

completed in 422.6 seconds wall time (419.1 seconds reported internally). GPU
memory was kept below the 9 GB task limit by rendering in batches of at most 20
object poses.
