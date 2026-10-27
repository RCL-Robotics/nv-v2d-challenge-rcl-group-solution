# Object trajectory refinement

## Result

`object_poses_world.npz` contains `T_world_mesh`, float64, shape **(980, 4, 4)**, in the released world frame and original mesh-file coordinates. All frames are present.

The result conservatively corrects image-plane position in **580 frames**, retaining the baseline **exactly in 400 frames**. All 980 rotation matrices are bit-for-bit unchanged. The per-frame Euclidean camera-to-object-centre distance is preserved; maximum numerical difference is **1.33e-15 m**. No floor snapping, depth fitting, global alignment, pose re-centring, or symmetry flip is applied.

## Evidence and decisions

I inspected video crops throughout the episode and rendered the exact textured mesh under the supplied poses. The clearest error is an upward/leftward projection drift during the later visible carry. The corrected outlines now follow the observed basket body. See [before_after.jpg](before_after.jpg); baseline outlines are red and corrected outlines are green.

The basket has a nearly circular body, repeating rabbit cutouts, and a flexible handle. Texture appearances also differ between the scan and the video because the background is visible through cutouts. These observations do **not** establish a reliable rotation or 180-degree flip correction. Rotations are therefore preserved, including during late occlusion. Range is preserved because the supplied whole-human calibration is stronger evidence than this small object's apparent scale.

On the 580 changed frames, mean person-occlusion-aware silhouette IoU improves from **0.725 to 0.886**. Every accepted frame is re-rendered from its actual final 3D pose and improves this score by at least 0.01. Examples (zero-based frames):

| Frame | Baseline IoU | Final IoU | Translation change |
|---|---:|---:|---:|
| 0 | 0.868 | 0.926 | 1.01 cm |
| 200 | 0.653 | 0.877 | 3.92 cm |
| 300 | 0.797 | 0.873 | 1.99 cm |
| 638 | 0.562 | 0.940 | 5.17 cm |
| 700 | 0.312 | 0.913 | 9.02 cm |
| 723 | 0.236 | 0.923 | 11.09 cm |

The median accepted translation change is **1.96 cm**; the maximum is **13.80 cm**, where the baseline projection visibly misses the basket. Frames 729–979 are unchanged because visibility/fit does not support correction. Other weakly supported intervals are also retained.

## Deterministic method

1. Load the exact GLB using `trimesh.load(..., force="scene", process=False).to_geometry()`. Transform baseline poses into the camera frame using the released camera transform.
2. Rasterize silhouettes in a 224-pixel crop. Apply a 3-pixel morphological opening to both observed and rendered masks to reduce sensitivity to flexible handles and narrow SAM fragments. Retain the largest observed component; require at least 150 pixels.
3. Search image translations within ±48 pixels. Score intersection-over-union, excluding rendered pixels hidden by the released person mask, while treating observed object pixels as visible. Do not optimize rotation, scale, or range.
4. Require strong shape/visibility evidence: IoU ≥0.84, occlusion ≤18%, and observed/rendered area ratio 0.80–1.25. Nearby supported frames may use IoU ≥0.75, occlusion ≤35%, and area ratio 0.65–1.35. They must be within four frames of strong evidence, in a supported run of at least seven frames containing at least three strong frames. All constants and their purposes are in `solve.py`; there are no manually specified frame corrections.
5. Apply a mild three-frame median filter to the registered image position, blend small/low-benefit changes toward the baseline, and taper corrections at visibility boundaries. Form the new camera ray through the corrected image centre and rescale it to the **original per-frame centre distance**. Convert only the translation change back to world coordinates.
6. Re-render each candidate and reject it unless the actual fixed-range result improves IoU by ≥0.01. Leave unsupported frames exactly unchanged.

## Validation, confidence, and limitations

- Validated shape, dtype, finite values, unchanged homogeneous rows, identical baseline rotations, exact baseline fallback, and preservation of camera-centre distance.
- Ran the required command twice in separate processes and confirmed **bitwise-identical pose arrays**.
- High confidence in the large visible position corrections, moderate confidence in small corrections, and insufficient evidence to modify rotations or occluded poses.
- Silhouette IoU is a fitting diagnostic, **not a ground-truth ADD measurement**. No ground truth or annotations were accessed, and actual ADD improvement cannot be quantified here. Baseline orientation/depth errors and uncertainties during occlusion remain.
- The method does not propagate inferred motion across invisible intervals. Easing at confidence boundaries is intentionally conservative and may leave residual position drift there.

`alignment_audit.npz` records per-frame measurements, confidence gates, accepted corrections, and before/after IoU. `metrics.json` includes aggregate results and all accepted frame intervals (end-exclusive).

## Reproduction and runtime

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Final measured runtime: **8.72 seconds**, excluding Python startup, on the selected NVIDIA RTX 6000 Ada GPU. Random seeds are fixed, TF32 is disabled, and deterministic PyTorch/CUBLAS behavior is enabled. A repeat run reproduced the result exactly. Total GPU-using command wall time during exploration, rendering, and validation was well under five minutes, far below the 45-minute budget; kernel-active GPU time was not separately profiled.

The solve is offline and uses no scratch files, downloaded models, or prior diagnostic outputs. See [TOOLS.md](TOOLS.md) for installed versions and the renderer source revision.
