# Fixed-rotation translation refinement

Produced `object_poses_world.npz`: `T_world_mesh`, float64, shape **(553, 4, 4)**, in the released world frame and original mesh-file coordinates. Every baseline rotation and homogeneous bottom row is preserved exactly. Only translation changes. No ground-truth poses, annotations, additional video data, network access, or model downloads were used.

**Decision and evidence.** Correct temporal depth drift and image-plane displacement conservatively; retain the baseline's episode-wide metric-depth anchor. The video shows the cooker remaining supported by the round table while turning. There is strong evidence for static positions before and after contact, but insufficient evidence for a uniform depth offset or scale correction.

- The exact mesh dimensions are 22.04 × 21.72 × 23.26 cm. Its projected silhouette and the provided object masks constrain translation at fixed baseline rotations. The baseline has a repeatable lateral displacement of about 2 pixels. The corrected camera translation moves left by 8.9 mm on average.
- Static periods are detected automatically, without embedded frame indices: at least 30 consecutive frames with maximum hand contact probability below 0.1, mask-center span at most 2 pixels, mask-size span at most 3 pixels, and rotation span below 5°. These criteria select frames **0–130** and **459–552**. Their measured center spans are respectively [1.5, 0.5] and [1, 0.5] pixels; size spans [3, 2] and [2, 1] pixels; rotation spans 2.44° and 2.28°. Baseline depth varies substantially despite these stable observations.
- The released floor normal supplies an approximate tabletop normal. The tabletop height is a fitted nuisance variable; the cooker is never forced onto the floor. An 8 mm support tolerance accommodates mesh-bottom, rotation, and normal errors. Visual inspection supports continued tabletop contact; this episode-specific physical assumption is explicitly commented in `solve.py`.
- The exact frame-0 reference human projects to bounding box [446.1, 385.3, 616.6, 895.1], versus person-mask bounds [442, 374, 615, 910]. This discrepancy makes a precise focal-length/depth correction from silhouette alone unreliable. The solver allows a common focal nuisance with a 2.5% prior, rather than treating the estimated K as exact. Its fitted value is +1.155%; it is not a separately validated camera calibration or a change to output rotations.
- Reconstructed hand positions are pipeline estimates coupled to the object. They are not treated as independent metric-depth measurements. Contact probabilities only help identify untouched periods, with independent mask and rotation checks.

**Independent depth check.** `evidence.py` samples 24 frames (every 25 frames plus frame 552), erodes the object mask by four pixels, excludes person pixels, and compares aligned MoGe-2 depth with ray intersections against the posed exact-mesh convex hull. It uses up to 400 deterministic rays per sampled frame. The hull is an approximation to visible surface depth, adequate for checking decimetre disagreements; these depth maps are not optimization inputs.

Raw MoGe surface medians span 2.707–2.870 m; aligned surface medians span 2.958–3.297 m. They do not support a consistent global correction. Across sampled frames, median signed aligned-depth discrepancy improves from −2.42 cm to −0.16 cm, median absolute discrepancy from 8.95 to 6.24 cm, and discrepancy standard deviation from 12.93 to 9.81 cm. This supports the drift correction while demonstrating why direct per-frame replacement by MoGe would be unreliable. Alignment itself used the pipeline human, so this is a separate cue, not an independent ground-truth sensor.

**Method.** The deterministic CPU solver jointly refines all camera translations using robust projected mesh supports in 32 directions, conservative baseline priors (2 cm laterally, 6 cm in depth), temporal acceleration, tabletop support, and constant positions for verified static periods. Cardinal silhouette directions receive greater weight; upper directions are suppressed when the person mask approaches the object. Correlated supports are normalized to four effective image observations. Focal length and tabletop level are nuisance variables. Ordinary translation changes are limited to 8 cm; verified static periods allow 12 cm to remove unsupported tracker spikes. The resulting camera translation differences are rotated into the released world frame. No rotations are optimized.

**Measured results.** Image residuals below are computed after all correction limits, using the original released K unless specified. These are consistency measurements, not ADD scores.

| Measurement | Baseline | Refined |
|---|---:|---:|
| Weighted silhouette-support RMS, original K | 2.294 px | 0.916 px |
| All-direction silhouette-support RMS, original K | 2.268 px | 1.186 px |
| Camera depth minimum–maximum | 3.040–3.250 m | 3.116–3.190 m |
| Frame-to-frame depth-step RMS | 24.80 mm | 1.19 mm |
| Static depth standard deviation, frames 0–130 | 40.70 mm | 0.176 mm |
| Static depth standard deviation, frames 459–552 | 32.34 mm | 0.065 mm |
| Mesh-bottom support-level standard deviation | 8.67 mm | 2.18 mm |

Mean translation change is 3.24 cm, maximum 11.17 cm, with large changes confined to baseline spikes or supported static corrections. Mean camera-depth change is **−0.066 mm**, so the baseline global depth anchor is effectively preserved. Static-position depths are 3.116 m initially and 3.154 m finally; the object is allowed to move during the interaction. Weighted image RMS with the fitted focal nuisance is 0.777 pixels.

**Confidence and limitations.** Confidence is high that the untouched-period drift and fast depth spikes are unsupported, and moderate that the translation refinement improves overall ADD. Absolute depth remains uncertain by several centimetres because K, the floor fit, segmentation, and reconstructed human motion are imperfect. Holding rotations fixed also limits achievable silhouette agreement. No ADD improvement can be measured without ground truth, which was neither accessed nor sought. Support-plane and static-period consistency are partly enforced properties; the unused MoGe comparison supplies the additional check.

**Reproduction and validation.** Run offline:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

This needs only the released inputs and installed CPU packages, not `/pipeline`, `/weights`, `/code`, scratch files, or network access. Runtime was **38.0 seconds**, CPU only; GPU time was zero. Repeated runs produced bit-identical pose arrays. Checked all 553 finite float64 matrices, exact rotation and bottom-row preservation, and Python syntax. `diagnostics.json` is regenerated alongside the poses.

Optional audit: `python /work/outputs/evidence.py` uses `/pipeline/depth/moge2/aligned.h5`, writes `independent_evidence.json`, and regenerates the comparison figures. `depth_comparison.png` shades the detected static periods. In `projection_comparison.jpg`, red is baseline and green is refined, both projected with the original K.
