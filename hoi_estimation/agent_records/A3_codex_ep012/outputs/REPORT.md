# Platform trajectory refinement

Produced `object_poses_world.npz` with `T_world_mesh`, shape `(700, 4, 4)`, dtype `float64`, in the released world frame and the exact mesh-file coordinate convention. All frames are present and finite.

## Changes and evidence

Applied small camera-space rotation and image-position corrections to 566 frames. The other 134 frames retain the baseline values exactly. The platform's complete turnover and its orientation branch are preserved: no symmetry swaps or 180-degree flips were introduced. The sampled textured renders agreed with the observed faces before refinement; they did not justify a discrete orientation change.

The baseline's per-frame distance between the camera and the mesh origin is preserved. Maximum numerical distance change is 1.8e-15 m. No floor snapping, object-only depth fitting, world-frame alignment, body refitting, or depth-scale changes were performed.

The largest accepted correction is 3.181 degrees, 7.255 image pixels, and 3.256 cm of translation. Mean sampled mesh-point displacement from the baseline is 1.059 cm; this describes the size of the edits, **not** ground-truth ADD.

| Image agreement over all 700 frames | Baseline | Result |
|---|---:|---:|
| Visible silhouette L1 discrepancy / visible target area | 0.15254 | 0.12177 |
| Rendered outer contour to video Canny edge, clipped mean distance (pixels) | 1.4229 | 1.1343 |

Silhouette discrepancy falls by 20.2%; contour distance falls by 20.3%. Video-edge agreement corroborates the segmentation evidence and rejects some otherwise attractive mask fits. These measurements are image-based consistency checks, not a measurement of 3D accuracy. `final_comparison.jpg` shows video, baseline contours, and final contours; green is the supplied object mask and red is the rendered mesh outline.

A separate raster-contour check used additional Canny thresholds that were not used for candidate selection:

| Canny thresholds | Baseline distance (pixels) | Result distance (pixels) |
|---|---:|---:|
| (15, 40) | 1.2348 | 1.0307 |
| (25, 65) | 1.4158 | 1.1297 |
| (40, 100) | 1.6030 | 1.2420 |

## Fitting procedure and conservative decisions

Loaded the exact GLB with `trimesh.load(..., force="scene", process=False).to_geometry()`. Rasterized it with nvdiffrast in 256-pixel crops centred on the baseline projection, using the released intrinsics and camera transform. Fitted five small corrections per frame: a camera-space left rotation vector and two image offsets. The translation is obtained by normalizing the new image ray to the **original camera distance**.

Adam ran 140 steps per frame, in batches of 20, with fixed seeds, no TF32, and explicit penalties toward the baseline. Pixels belonging to the supplied person mask, dilated by two pixels, were ignored. Median filtering over five frames and Gaussian smoothing with sigma three frames were applied to the corrections only; the baseline motion itself was retained. Rotation proposals were limited to four degrees.

Acceptance required a 21-frame median silhouette improvement greater than 7%, median video-edge improvement greater than 0.08 pixels, and visible target fraction greater than 55%. Confidence transitions were smoothed over five frames. Corrections with contrary per-frame evidence were rejected and tapered toward zero over six neighbouring frames, avoiding isolated jumps. All choices are encoded numerically in `fit_refinement.py`; there are no manually chosen frame intervals or face labels. The rendered mesh's outer contours were used for the video-edge test, excluding the human and limiting edge distances to five pixels.

## Confidence and limitations

Confidence is moderate: multiple consecutive silhouettes and actual image edges support the small adjustments, especially during the lift and near-vertical part of the turnover. Depth and the baseline's orientation branch have stronger prior support and were preserved. Fine rotations remain partially ambiguous because the object is dark, its scan is imperfect, and the image width is only around 150 pixels. Some mask boundaries include blur or shadow. No ground truth was accessed; improvement in the eventual ADD score cannot be established from these inputs alone.

The released baseline has small numerical rotation errors: orthogonality error reaches about 2.03e-5. Left multiplication by a rotation preserves that existing precision. Validation allows 3e-5 rather than silently changing the original rotations or unchanged poses. The final maximum orthogonality error is 2.03e-05.

## Reproduction and runtime

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

`solve.py` embeds the accepted per-frame correction table as explicit, commented constants, and computes the final world poses from the supplied baseline and camera. The table was generated exclusively by the fitting procedure above. Freezing the accepted numerical fit avoids possible nondeterminism in CUDA rasterizer backward reductions. The solver verifies SHA256 fingerprints of the five inputs used for fitting, requires only NumPy and SciPy, and does not read previous outputs, scratch files, downloaded tools, or any network resource. `fit_refinement.py` retains the complete fitting source for inspection and optional refitting; it is not a runtime dependency of `solve.py`.

The final fitting pass took 206.33 seconds on the one selected RTX 6000 Ada. Two full fitting passes and brief exploratory renders/fits stayed below 15 minutes of total GPU process wall time, within the 45-minute budget. The deterministic offline reproduction took 0.377 seconds. Two invocations produced exactly equal arrays and byte-identical archives; the packaged solver also reproduces the fitted array exactly. All input files were left unmodified. No tools, weights, or data were downloaded.

Additional audit files: `refinement_diagnostics.npz` contains proposals, accepted corrections, visibility, and per-frame scores; `validation.json`, `independent_validation.json`, and `reproduction_check.json` contain numerical checks. `validate_result.py` records the file, geometry, distance-preservation, and edge-sensitivity checks. Package versions and the existing nvdiffrast git revision are in `TOOLS.md`.
