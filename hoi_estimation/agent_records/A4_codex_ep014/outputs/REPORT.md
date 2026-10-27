# Round-table trajectory refinement

`object_poses_world.npz` contains `T_world_mesh`, float64, shape `(993, 4, 4)`, covering every input frame in the released world frame and original GLB coordinates. No ground truth, external video data, or downloaded models were used.

## Changes and evidence

The baseline follows the complete roll/flip sequence. Two refinements were supported by the supplied images:

1. **Small image-position and rotation corrections.** Differentiable rendering of the exact mesh was fitted to the SAM2 object masks, excluding person occlusion. Optimization preserved the per-frame camera distance of the oriented-bounding-box centre, regularized rotation and image displacement toward the baseline, and limited the search to 12 degrees and 10 pixels. Corrections were fitted every second frame, interpolated, and smoothed. An independent check rejected local proposals that worsened either visible IoU or mask loss. Actual accepted local corrections average 0.84 degrees and 1.70 pixels, with maxima of 3.64 degrees and 7.60 pixels.

2. **Material orientation correction in frames 500–830.** The round top and five-leg base produce misleadingly similar silhouettes at different rotations. The exact mesh has a distinctive light label and adjustment lever on the brown underside. Textured rendering showed the baseline putting both on the wrong side while the label was visible. Automatic neutral-color component detection and reprojection among all five 72-degree phase candidates independently selected approximately −144 degrees before the baseline's abrupt yaw change around frame 605, and −72 degrees afterward. The label's mesh coordinate `[-0.16595, -0.15357, -0.08807]` was measured from the supplied mesh texture, not from an annotation. The script requires an error below 8 pixels and a margin above 5 pixels over competing phases, rejects isolated detections, and bridges gaps of at most 60 frames bounded by reliable observations. It smooths the observed material twist while preserving the baseline table normal. The readable-label search interval and feature coordinate are explicitly commented constants in the script.

| Measurement | Baseline | Local corrections | Final result |
|---|---:|---:|---:|
| Mean visible silhouette IoU, all 993 frames | 0.8189 | 0.8474 | 0.8341 |
| Mean silhouette objective, all frames | 0.1990 | — | 0.1792 |
| Median label reprojection error, 121 reliable observations | 85.11 px | — | 4.28 px |

The final material correction deliberately accepts a lower silhouette score than the local-only result: the label and lever distinguish physical mesh coordinates that the repeated legs cannot. ADD has no symmetry handling, so that correspondence matters. Final silhouette agreement still exceeds the released baseline.

See [orientation_evidence.jpg](orientation_evidence.jpg) for video / textured baseline / textured final comparisons, and [evidence.jpg](evidence.jpg) for mesh overlays across the sequence. Numerical measurements and corrections are in `refinement_summary.json` and `refinement_diagnostics.npz`.

## Preservation, confidence, and limits

The per-frame distance of the mesh's oriented-bounding-box centre is preserved to a maximum absolute error of **1.34e-15 metres**. There is no object-scale/depth fit, floor snapping, or world-frame alignment. Exactly **134 frames** remain bitwise identical to the input baseline.

Confidence is high that the label-supported interval has the correct material phase; small mask corrections and interpolation across label occlusion have moderate confidence. Material phase outside frames 500–830 remains uncertain and is retained from the baseline. This conservative restriction leaves large phase transitions at the interval boundaries rather than assuming the same correction is valid in unobserved views. Mesh texture distortion, freely oriented casters, SAM boundaries, and estimated intrinsics limit further precision. ADD improvement cannot be measured without ground truth; the reported evidence is image agreement, not a ground-truth accuracy claim.

## Reproduction, timing, and checks

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The default export takes **1.29 seconds** on this container and uses CPU float64. It reconstructs poses from the original baseline and fitted rotation/ray/phase coefficients embedded in the script. The coefficients were obtained only from the supplied masks, mesh, and video. They are frozen because CUDA rasterizer gradient atomics can vary slightly between runs despite fixed seeds. Repeated default exports were **bitwise identical**, including to the original fitted trajectory. No other output file or scratch asset is required.

The same script retains the full fitting implementation, available with `--refit`. The final fit took **368.85 seconds** wall time on the single selected GPU; cumulative GPU-enabled experiments stayed below 20 minutes of process wall time, conservatively below the 45-minute GPU budget. Seeds are fixed to zero. Optional refitting can differ slightly in its final numerical coefficients due to CUDA atomics.

Validation checked frame count, float64 dtype, finiteness, homogeneous rows, rotation determinants, baseline-level orthogonality precision, unchanged frames, centre distances, and exact export reproduction. The released float32 rotations already have orthogonality residuals up to 2.16e-5; those rounding errors were retained to preserve baseline frames exactly. Tool versions and dependencies are listed in [TOOLS.md](TOOLS.md).
