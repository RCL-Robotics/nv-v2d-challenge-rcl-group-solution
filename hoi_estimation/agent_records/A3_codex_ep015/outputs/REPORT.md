# Object trajectory refinement

The output contains a float64 `T_world_mesh` array of shape `(715, 4, 4)` in the released world frame, using the exact supplied mesh coordinates. **449 frames receive small, image-supported corrections; 266 frames retain the supplied baseline exactly after conversion to float64.**

## Changes and evidence

I rendered the supplied textured mesh at each baseline pose and matched its printed warning symbol and lettering to the video using SIFT. Each rendered feature has a mesh-coordinate position recovered from the rasterizer. Robust least squares fits three local rotation parameters and two camera-plane center parameters. The camera range of the oriented-bounding-box center remains fixed separately in every frame. No ground-plane snapping, object-only depth estimation, symmetry flips, learned model, external annotation, or ground truth is used.

The mesh-coordinate center used for this constraint is `[0.0041584492, -0.0039267586, 0.0196669399]` metres. The largest camera-range change in the saved trajectory is **8.9e-16 m**, numerical roundoff.

Updates require at least ten distinct spatially distributed matches, robust image-space agreement, and three-fold held-out feature checks. Fits must improve at least two held-out folds, improve mean held-out error by more than 0.3 pixels, and remain consistent across folds. Rotation and center displacement are capped at 8 degrees and 2.5 cm. Small corrections are smoothed over neighboring supported frames, then checked again against the observed features; unsupported frames are left unchanged. All thresholds are explicit constants or predicates in `solve.py`; there are no manually selected frame intervals or hand-labeled image correspondences.

On the 449 modified frames:

| Measurement | Result |
|---|---:|
| Mean of per-frame median printed-feature error, baseline | 3.361 pixels |
| Same measurement, final output | 0.921 pixels |
| Mean held-out improvement of the unsmoothed fitted candidates | 2.552 pixels |
| Rotation correction, median / 95th percentile / maximum | 1.875 / 4.395 / 6.081 degrees |
| Center displacement, median / 95th percentile / maximum | 1.188 / 1.898 / 2.213 cm |

Example frame errors, measured in original 1536×1152 video pixels:

| Frame | Distinct feature matches | Baseline median error | Final median error |
|---|---:|---:|---:|
| 0 | 34 | 2.965 | 0.478 |
| 180 | 36 | 1.570 | 1.232 |
| 300 | 20 | 5.446 | 0.843 |
| 340 | 53 | 2.327 | 1.507 |
| 420 | 23 | 1.884 | 0.921 |
| 540 | 21 | 4.741 | 1.146 |

`pose_comparison.jpg` shows observed crops and before/after textured renders. `refinement_diagnostics.json` records the automatic decisions, match counts, held-out results, and final errors for every frame. Renderings do not composite the person over the object, so otherwise hidden geometry remains visible in those comparisons.

## Rejected alternative and uncertainty

An exploratory differentiable RGB/silhouette fit reduced its own objective but sometimes worsened independent printed-feature alignment. The SAM masks fill some real holes, and the person hides much of the sign around frames 450–505. That optimizer was rejected. Alternative FoundationPose rotations were inspected but not adopted without independent evidence.

The final corrections have strong image-plane evidence and moderate confidence in their small orientation adjustments. The method does not resolve large orientation ambiguities behind the person. It preserves the baseline when features are too sparse, matches are inconsistent, or the sign is nearly edge-on. For example, frames 460 and 500 retain the baseline, and **all frames 551–714 remain unchanged**. The original calibrated depth and any unresolved baseline errors therefore remain. Printed-feature reprojection is an observable proxy; no ground-truth ADD score was available or measured, so an ADD improvement cannot be certified here.

## Reproduction and validation

Run offline in the supplied environment:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The solver reads only the supplied input files and installed packages. It needs no scratch files, downloaded resources, pipeline intermediates, weights, or network. NumPy, PyTorch, and OpenCV seeds are fixed; OpenCV and CPU numerical work use one thread. The first complete run took **152.1 seconds** on the selected RTX 6000 Ada GPU. Exploratory work and two full solver runs together remained below ten minutes of GPU-enabled process wall time, an upper bound on active GPU time, within the 45-minute budget.

Validation confirms all 715 finite float64 poses, correct homogeneous rows, unchanged baseline entries on all 266 retained frames, and preservation of camera range. Rotation determinants range from 0.99999625 to 1.00000066; the small deviations are inherited from the supplied float32 baseline rotations. The required output is saved with the exact key `T_world_mesh`.

A second execution of the exact offline command took **152.4 seconds** and reproduced the trajectory **bit-for-bit** (`np.array_equal` passed; maximum pose-entry difference 0). The NPZ files also had identical SHA256 hashes: `831d53abf0983ac61951e866fe812901492bacd731104cbafd1029fc4fad661b`.
