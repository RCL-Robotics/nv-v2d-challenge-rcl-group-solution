The output is `object_poses_world.npz`, containing `T_world_mesh` with shape **(870, 4, 4)** and dtype **float64**, in the released world frame and original mesh-file coordinates. It contains every video frame.

The result conservatively refines the supplied reconstruction: **246 frames receive a small image-plane position correction**, and **frames 265, 266, and 383 receive a heading correction**. **622 complete poses are copied exactly from the baseline.** The other 867 rotations are copied exactly. No ground-truth poses, challenge data, annotations, alignment to an evaluation trajectory, or external observations were obtained or used.

The calibrated camera distance of the mesh surface centroid is preserved separately in every frame. The largest numerical change in that distance is **3.88 × 10⁻¹¹ m**. The released ground plane is not used to adjust the object. No depth estimates, silhouette scale estimates, or PnP estimates replace the baseline distance.

Position corrections are supported by the supplied video and masks. The exact mesh is rendered at the supplied camera intrinsics. Translation is fitted against a symmetric contour-distance objective in a 160-pixel crop. The bowl's convex outer silhouette permits filling the narrow chair-leg stripe and small mask cutouts. Hand occlusion is excluded using skin-coloured pixels within the person mask; the entire person mask cannot be excluded because it also contains the visible bowl and torso behind it.

A proposal requires a visible-area ratio of at least 0.75, contour coverage of at least 0.70, a fitted contour distance below 2.50 pixels, improvement of at least 0.35 pixels, agreement with a five-frame median, and support in three of five frames. The script applies 75% of the fitted shift, capped at five pixels, to limit sensitivity to mesh/mask/calibration errors. It converts the shifted centre ray to a 3D translation at the original radial distance. Every candidate is then re-rendered at its actual new 3D pose and retained only if contour distance improves by at least 0.20 pixels and silhouette IoU by at least 0.005. Initial stationary frames and heavily occluded frames remain unchanged when this evidence is insufficient.

The accepted position changes average **1.18 cm** and reach at most **1.89 cm**. The measured image evidence is:

| Metric over the 246 changed-position frames | Baseline | Output |
| --- | ---: | ---: |
| Mean symmetric contour distance, pixels | 2.858 | 1.935 |
| Mean silhouette IoU, occlusion-aware convex target | 0.8345 | 0.8821 |
| Mean IoU against the original, unfilled object masks, without hand exclusion | 0.7829 | 0.8332 |

The last row is an additional check rather than the fitting objective. Original-mask IoU improves in **all 246** changed-position frames. Detailed measurements appear in `validation.json`, `frame_diagnostics.csv`, and `diagnostics/independent_checks.json`.

Heading corrections address two isolated temporal anomalies while both hands have contact probability above 0.95. Four surrounding orientations agree within 6.23° around frames 265–266 and within 3.51° around frame 383, whereas the intervening baseline headings deviate by approximately 25–27°. The script detects these intervals automatically, interpolates the surrounding heading, and rotates only about the mesh's local y axis, preserving its opening-axis direction. Final heading changes are 26.15°, 27.29°, and 28.64°, respectively. The approximately 25–29° entry/exit jumps become approximately 1–2° steps. All broader heading ambiguities remain as supplied.

Confidence is high that these brief heading spikes are tracking artifacts, and moderate that the bounded position changes improve the true trajectory. The bowl is nearly rotationally symmetric in the available images. Thirty-degree texture/heading sweeps did not provide dependable absolute heading evidence: lighting reflections and scan-texture differences dominate the small visible details. Broad heading changes, inferred rigid hand attachment over long intervals, and floor snapping were therefore not adopted. The image measurements support these changes, but **ADD and evaluation-threshold accuracy cannot be measured without the withheld poses**.

Reproduce the final result offline with:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

`solve.py` is self-contained and needs no other output, diagnostic cache, scratch file, pipeline output, model weight, or network connection. It uses the existing installed packages listed in `TOOLS.md` and the permitted input files. Seeds and CPU thread settings are fixed; insignificant raster/optimiser noise is quantized below the image measurement resolution. Two complete final runs produced **bitwise-identical pose arrays**. Array SHA-256: `57fed02f853297c9d0a1c5f8947601cd2ed6ab830a6dc7ec5519cda3d0372ed1`.

A final solve took approximately **13 seconds**, excluding Python import/startup time, on the provided RTX 6000 Ada. Its timed GPU rendering section took approximately **0.6 seconds**. There was no training or learned-model inference; total GPU work across exploration and final runs stayed well below the 45-minute limit (under five minutes even using a conservative wall-time bound for GPU-bearing commands).

`diagnostics/final_overlay.jpg` shows examples: cyan is the baseline silhouette and magenta is the output; unchanged silhouettes overlap. The exploratory scripts, yaw sheets, and numerical logs are retained under `outputs` for inspection. They are not required by `solve.py`.
