# Squeegee trajectory correction

## Deliverable and offline reproduction

The required object_poses_world.npz contains T_world_mesh, shape **(988, 4, 4)**, **float64**, in the released world frame and original mesh-file coordinates. Every frame is present.

Run:

    python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

The solver is self-contained and reads only supplied inputs. It requires no exploratory output, scratch file, network, model weights, or pipeline intermediates. Installed dependencies are listed in TOOLS.md. Two clean executions produced **bit-for-bit identical pose arrays** (maximum matrix difference 0.0).

## Changes and evidence

The baseline fails during the turn behind the actor, around frames 700–790. Later it places the wide blade near the visible handle loop while the actual blade is at the upper end near the stool. The visible orange triangular mount, end loop, blade direction, and motion continuity provide clear evidence of this endpoint/orientation error.

1. Extract orange components and shaft geometry from supplied RGB and SAM masks. A saturated-orange threshold recovers the head where SAM omits it behind the arm.
2. Initialize the failed interval using identified blade and handle endpoints. Solve their camera-ray positions subject to the baseline center-to-camera distance. Preserve the continuous depth branch anchored before the failure.
3. Recover roll from visible blade directions. During partial occlusion, use explicit image-plane directions checked against RGB with OpenCV line detection. Constants and reasons are commented in solve.py; these are supplied-image measurements, not external 3-D annotations. The blade remains nearly vertical through frame 840, then becomes horizontal around frame 870.
4. Refine orientation and image-plane center with deterministic single-threaded CPU least squares. Fixed area-stratified samples from the exact supplied mesh are matched to object masks; person masks permit occlusion. Color landmarks, a baseline orientation prior, and temporal filtering of residual corrections constrain the fit. No random mesh sampling or GPU backward optimization is used in the final solver.
5. Render the full 95,575-face mesh for an every-frame acceptance check. Before the failed interval, require an overlap gain over 0.055, a supporting nine-frame median, noncontradictory landmark evidence, and bounded rotation/translation changes. Rejected frames retain the original world matrices exactly. The clearly failed late interval uses the endpoint-consistent reconstruction.

The final output changes **887 frames** and retains **101 baseline frames exactly**. Earlier changes are usually small; the large late rotation corrections are justified by the visible endpoint reversal.

An earlier differentiable GPU optimizer was tried and rejected after repeat testing exposed nondeterministic backward reductions. The final CPU optimizer was selected specifically to satisfy deterministic reproduction. OpenCV precise distance-transform results also varied slightly across processes; replacing that operation with SciPy Euclidean distance_transform_edt removed the remaining instability. Experimental outputs are not dependencies of solve.py.

## Distance and coordinate safeguards

The mesh-file origin equals its surface-area centroid to within about 4 nanometers. Its per-frame Euclidean camera distance is fixed to the baseline. The maximum measured difference is **8.88e-16 m**. Camera z can change geometrically when the image ray changes; no independent depth or scale correction is estimated.

No ground-plane snapping, world alignment, rescaling, symmetry equivalence, or mesh-coordinate recentering is applied. All transforms use the original mesh-file frame and released camera-to-world matrix. Unchanged poses retain their small original float32 rotation errors; maximum rotation orthogonality error is **2.58e-06**. All transforms are finite, have valid homogeneous bottom rows, and positive determinants near one.

## Image-consistency measurements

These diagnostics are **not ground-truth ADD scores**. Masks and orange cues also inform fitting, so they are not a held-out accuracy test. No ground truth was accessed; actual ADD improvement cannot be certified locally.

The diagnostic renderer uses the full mesh at 768×576, discounts person-occluded predicted pixels, and compares with supplied masks. This summary covers 335 frames: every third frame plus 980–987. Landmark errors use only detected corresponding orange components.

| Frames | Baseline IoU | Final IoU | Baseline head error, px | Final head error, px |
|---|---:|---:|---:|---:|
| 0–129 | 0.369 | 0.484 | 11.03 | 8.74 |
| 130–499 | 0.345 | 0.639 | 11.66 | 3.22 |
| 500–659 | 0.278 | 0.632 | 20.04 | 4.44 |
| 660–689 | 0.337 | 0.658 | 15.80 | 1.68 |
| 690–729 | 0.081 | 0.646 | 96.92 | 1.21 |
| 730–789 | 0.137 | 0.652 | 99.83 | 4.05 |
| 790–919 | 0.278 | 0.564 | 295.03 | 8.05 |
| 920–987 | 0.192 | 0.715 | 288.48 | 1.70 |
| Overall | 0.292 | 0.616 | 80.92 | 4.58 |

Mean handle-loop error falls from **156.47 to 3.01 pixels**. Maximum adjacent rotation step falls from **119.36° to 28.33°**. Steps above 25° change from **23 to 2**. Median rotation step changes from 1.81° to 1.53°.

Additional non-optimized evidence: At frame 750, nearest reconstructed hand-to-shaft distances change from 46.8/51.6 cm to 14.3/14.2 cm. At frame 800, nearest reconstructed hand-to-shaft distances change from 25.2/33.5 cm to 2.3/4.2 cm. Reconstructed hands are approximate and were not treated as ground truth.

Evidence files include baseline_contact_sheet.jpg, rgb_details.jpg, transition.jpg, head_details.jpg, final_render.jpg, and metrics_final.json. Green contours are the supplied object masks.

## Confidence and limitations

Confidence is high that the late endpoint reversal is corrected and image alignment improves. Confidence is moderate in fine roll and shaft tilt through heavy occlusion, and lower in absolute metric accuracy where the fixed baseline distance is imperfect. The final stationary-looking segment deliberately retains baseline per-frame camera-distance drift. No ground-plane constraint was used to override it.

The 180° roll branch is propagated continuously from the correctly tracked blade before the turn, rather than treating the blade as symmetric. Short occluded intervals retain some roll uncertainty. Early frames with weak support for an edit preserve the baseline.

## Runtime and verification

The final raw-input run took **354.25 seconds (5.90 minutes)**. Deterministic CPU fitting took **188.82 seconds**. Only forward full-mesh validation uses the selected logical CUDA device 0 (NVIDIA RTX 6000 Ada Generation); no other GPU is selected. Total GPU work across experiments and checks remained below the 45-minute limit (conservative cumulative process-time upper bound below 40 minutes).

Both final clean runs used only the inputs and installed dependencies. Pose arrays are exactly equal, not merely approximately equal. Exact baseline equality was verified for all rejected frames. All 988 transforms passed finite-value, shape, dtype, determinant, orthogonality, homogeneous-row, and fixed-distance checks.

object_poses_world_run.json records runtime and structural checks. object_poses_world_diagnostics.npz records acceptance weights, candidate image scores, detected endpoints, and fitting losses. REPRODUCIBILITY.json records the repeat comparison and artifact hashes.
