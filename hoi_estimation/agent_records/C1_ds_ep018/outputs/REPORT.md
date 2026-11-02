# Camera and world trajectory estimate

## Result

- Output: `object_poses_world.npz` with `T_world_mesh` for all 858 frames.
- Camera: `camera_estimate.json`; `K` is the supplied shared intrinsic matrix and was not changed.
- Object camera-frame trajectory is preserved. A single translation-depth scale **1.04256** is applied; multiplying the full camera-space point by this scale leaves its projection exactly unchanged.
- Final camera change from baseline: **3.34 deg**, centre **18.50 cm**.
- Final camera-plane disagreement with the independently fitted observed floor normal: **0.79 deg**; plane offset difference **-0.14 cm**.

## Cues and independent estimates

1. **Aligned MoGe-2 floor plane.** RANSAC fits on six frames (person/object masks excluded, lower-image region): mean normal `[-0.01836 -0.99367 -0.11085]`, median `d=1.47028` m. The floor-only camera implies **3.27 deg** and **18.55 cm** from baseline. Across-frame normal std is `[0.00143 0.00019 0.00162]`, and `d` std is `0.0406` m.
2. **Frame-0 exact human, dense vertices.** The supplied 18,056 world vertices are projected and density-splatted against the SAM2 person mask. A silhouette-only 6-DoF fit changes the camera by **1.18 deg / 3.24 cm** (mostly a small roll), while the joint human+floor fit changes the silhouette IoU `0.8356 -> 0.8495`. Thus the image evidence does not oppose the floor-driven height correction and slightly favors the final raster over baseline.
3. **Feet-on-floor tolerance.** Re-expressing the reconstructed human with the camera change gives closest-foot distances summarized below; the released data statistic is mean `0.042` m, median `0.039` m. This supports the final camera height.
4. **Resting object / ray-depth offset.** Frames whose baseline mesh bottom is within 8 cm of the plane are treated as contact frames. One camera-frame translation scale is fitted so their minimum mesh height is on the floor. Result: **1.04256**. This is a depth-only correction, not an image-space correction.

## Validation (baseline vs final)

| Metric | Baseline | Final |
|---|---:|---:|
| Frame-0 dense-human silhouette IoU | 0.8356 | 0.8495 |
| Frame-0 symmetric boundary distance (px) | 5.74 | 5.46 |
| Predicted / mask area (px) | 32468 / 31043 | 32290 / 31043 |
| Floor points mean abs residual (cm) | 6.19 | 4.01 |
| Floor points median abs residual (cm) | 5.33 | 3.62 |
| Feet closest distance, mean / median (cm) | 0.26 / 0.23 | 3.71 / 4.09 |
| Resting mesh bottom, median [p10,p90] (cm) | 1.57 [0.52,3.88] | 1.90 [0.49,4.69] |

Final feet closest-distance p10/p90: **-0.50 / 6.57 cm**. Resting-object sample count: 179.

## Floor cue fits

| Frame | RANSAC inliers | RMS (cm) | Normal vs mean (deg) |
|---:|---:|---:|---:|
| 0 | 643448 | 0.74 | 0.08 |
| 100 | 628869 | 0.66 | 0.15 |
| 300 | 634838 | 0.63 | 0.05 |
| 500 | 615644 | 0.68 | 0.16 |
| 700 | 642252 | 0.69 | 0.08 |
| 857 | 642274 | 0.74 | 0.17 |

## Decision and confidence

The floor-only estimate supplies the large correction (camera about 19 cm above / slightly behind the baseline), the dense-human fit accepts it and improves the frame-0 silhouette, and the feet/object contact checks agree on the resulting height/depth. The final camera retains a sub-degree floor-normal residual because the body surface and clothing do not have identical silhouettes. The object scale is robust but less certain than the camera rotation; the median resting-object height is the primary evidence for it.

Confidence is **medium-high for camera tilt/height**, **medium for yaw/backward translation**, and **medium for object depth scale**. Residual disagreements are dominated by monocular depth scale/bias and the person's loose clothing versus body surface.

Runtime for this deterministic run: **87.1 s**.
