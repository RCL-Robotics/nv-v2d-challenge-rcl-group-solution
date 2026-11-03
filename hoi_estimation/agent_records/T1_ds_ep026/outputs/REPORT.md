# Skateboard 6-DoF trajectory recovery

## Result

- Output: `object_poses_world.npz`, key `T_world_mesh`, shape `(669, 4, 4)`, `float64`.
- Frames: all 669 frames of `inputs/video.mp4`.
- Convention: each matrix maps the released mesh-file coordinates to the released world frame.
- Reproducer: `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
- Final run time: **72.9 s** on the available RTX 6000 Ada GPU. The run is offline, fixed-seed, and writes only the requested NPZ.
- Output SHA-256 at the checked run: `79cd32ec178d0b48bddbbf05f5107c035607ed121e6e322bcf2ce7907db2f73a`.

## Phase segmentation

The camera is static. The object masks, baseline motion derivatives, contact signal, and visual inspection give five phases:

| Phase | Frames | Interpretation | Treatment |
|---|---:|---|---|
| Rest 1 | 0-285 | Board stationary while the person approaches and reaches it | One constant pose pooled from frames 0-230 |
| Spin 1 | 286-354 | First continuous rotation/push | Per-frame silhouette refinement plus temporal smoothing |
| Rest 2 | 355-404 | Short stationary interval while hands reposition | One constant pose pooled from frames 355-404 |
| Spin 2 | 405-504 | Faster second rotation/push | Per-frame silhouette refinement plus temporal smoothing |
| Rest 3 | 505-668 | Board stationary after release | One constant pose pooled from frames 530-668 |

The rest boundaries are explicit constants in `solve.py` because they were verified in the video and mask sequence. They also agree with the baseline angular-velocity plateaus.

## Method

1. **Static evidence pooling.** All static frames observe the same rigid pose. A differentiable nvdiffrast silhouette fit over the repeated masks suppresses per-frame noise. The fit is initialized from the rotation-mean/translation-median baseline pose and regularized to stay close to it.
2. **Independent checkpoint selection.** Several checkpoints of the static optimizations have nearly identical silhouette scores but visibly different texture alignment. Candidate checkpoints were scored with both mask IoU and textured-mesh grayscale correlation. The 10:1 score keeps silhouette dominant while resolving residual yaw/depth ambiguity.
3. **Bias anchoring.** For each moving interval, the correction between the baseline mean pose and the fitted static pose at each end is interpolated in SE(3). Applying this smooth correction to the baseline removes the slowly varying translation/depth bias without introducing jumps.
4. **Per-frame silhouette refinement.** During the two moving intervals, every frame is refined from the anchored baseline using the exact mesh silhouette and the object mask, with the person mask excluding occluded pixels. L2, BCE, and small pose regularizers constrain the update.
5. **Temporal smoothing.** Translation and unrolled relative rotation are filtered with a 7-frame Savitzky-Golay filter independently in each moving interval. The unrolled relative-rotation representation preserves the baseline's multiple continuous turns and avoids angle wrapping.
6. **No forced symmetry or floor snapping.** The baseline already had smooth adjacent-frame rotation continuity, so no 180-degree symmetry branch was imposed. The released floor plane was not used to snap the board because its several-centimeter foot-plane tolerance is not strong enough for the object height; small translation/depth corrections are instead supported by repeated masks and texture.

## Validation without ground truth

Mask IoU uses the rendered mesh silhouette versus SAM object pixels after excluding SAM person pixels. Texture NCC is grayscale normalized cross-correlation between the textured mesh render and the video inside the visible object region. It was not the optimization objective for dynamic frames and is therefore an independent check.

| Phase | Frames | Baseline mask IoU | Result mask IoU | Baseline texture NCC | Result texture NCC |
|---|---:|---:|---:|---:|---:|
| Rest 1 | 0-285 | 0.8450 | **0.8889** | 0.3043 | **0.3520** |
| Spin 1 | 286-354 | 0.6466 | **0.7393** | 0.0917 | **0.2953** |
| Rest 2 | 355-404 | 0.6982 | **0.7821** | 0.1840 | **0.2462** |
| Spin 2 | 405-504 | 0.6698 | **0.7348** | 0.2023 | **0.2647** |
| Rest 3 | 505-668 | 0.8159 | **0.8317** | 0.3212 | **0.3652** |

Held-out static checks (these frames were not used to fit their static pose):

| Held-out interval | Baseline mask IoU | Result mask IoU | Baseline texture NCC | Result texture NCC |
|---|---:|---:|---:|---:|
| Rest 1, frames 231-285 | 0.8011 | **0.8356** | 0.3310 | **0.3636** |
| Rest 3, frames 505-529 | 0.7234 | **0.7362** | 0.2784 | **0.3544** |

Temporal checks over the moving intervals:

| Interval | Rotation step, base/result | Translation step, base/result | Rotation acceleration, base/result | Translation acceleration, base/result |
|---|---:|---:|---:|---:|
| Spin 1 | 2.170 / **2.132 deg** | 7.895 / **4.486 mm** | 1.117 / **0.296 deg** | 12.13 / **3.38 mm** |
| Spin 2 | 2.850 / **2.803 deg** | 8.941 / **4.627 mm** | 1.070 / **0.286 deg** | 13.17 / **3.45 mm** |

The static treatment also removes baseline jitter completely within each rest phase. For example, baseline rest-1 translation standard deviation was `(4.36, 14.58, 35.61)` mm by world axis and its mean rotation deviation was `0.703 deg`; the result is exactly constant within each fitted rest phase.

## Confidence

| Phase | Confidence | Reason |
|---|---|---|
| Rest 1 | High | Hundreds of nearly unoccluded repeated observations; large, consistent gains in mask and texture scores; held-out frames improve. |
| Spin 1 | Medium-high | Good texture agreement and much lower acceleration, but hands occlude parts of the board and motion is fast. |
| Rest 2 | Medium-high | Silhouette and texture both improve; repeated frames support one pose, although hands frequently overlap the object. |
| Spin 2 | Medium-high | Largest visible motion and occlusion, but texture NCC and silhouette improve substantially and velocity remains continuous. |
| Rest 3 | High | Long static interval, clean post-release observations, and a held-out gain in both mask and texture scores. |

The retained trajectory is continuous in SE(3), contains all 669 frames, preserves the baseline's multi-turn yaw branch, and shows no unphysical pose teleport. The reported improvements are image-space and temporal checks only; no ground-truth trajectory or annotation was consulted.
