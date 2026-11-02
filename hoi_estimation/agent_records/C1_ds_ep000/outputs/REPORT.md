# Camera and world-trajectory estimate

## Adopted estimate

- Intrinsics `K`: unchanged from `camera.json`.
- Camera rotation change from baseline: **0.526 deg**.
- Camera centre change from baseline: **3.31 cm**.
- Object camera-ray depth scale: **0.9896** (`T_cam_mesh` translation scaled; rotation and metric mesh size unchanged).
- Frames written: **778**.

The baseline camera-frame object trajectory is preserved except for the small global ray-depth calibration. The output is `T_world_mesh = T_world_cam_new @ T_cam_mesh_scaled`.

## Evidence and per-cue agreement

| Cue | Suggested camera change | Agreement with adopted estimate |
|---|---|---|
| Visible dense frame-0 human (2959 correspondences) | 0.526 deg, 3.03 cm | primary 6-DoF estimate; reprojection mean 1.608 px |
| Released floor normal from 39 MoGe frames | 0.440 deg tilt | adopted normal error 0.013 deg at dense solution; floor cue and human agree |
| Resting object floor contact | 0.34 cm along upward normal before depth scaling | same sign; weak scalar height constraint; final contact explicitly enforced |
| Exact object mesh vs object masks | depth scale 0.9896; IoU 0.7505 -> 0.7672 | independent metric-depth calibration; adopted |

The dense human and floor-normal cues agree to about **0.01 deg**. The object contact and dense translation both imply a small upward correction. This agreement is why the baseline camera was changed rather than left unchanged.

For reference, trusting the full aligned MoGe plane distance would suggest a 10.3 cm normal-direction offset. That quantity is scale-sensitive (per-frame MoGe/SAM3D scales vary), so it was not used directly; only the scale-free normal was combined with the human cue. The object-image scale scan independently finds a ~1.0% metric depth correction.

## Validation (baseline vs adopted)

| Metric | Baseline | Adopted |
|---|---:|---:|
| Frame-0 reference-human projected-vertex silhouette IoU | 0.8171 | 0.8176 |
| Human boundary distance, render -> mask, median / p95 (px) | 5.00 / 12.12 | 5.00 / 12.00 |
| Human boundary distance, mask -> render, median / p95 (px) | 5.00 / 12.00 | 5.00 / 12.00 |
| MoGe floor points to released plane, median abs / RMS (m) | 0.0786 / 0.0841 | 0.0709 / 0.0763 |
| Human lowest-vertex floor distance, median (m) | -0.0194 | -0.0116 |
| Human 0.5th-percentile floor distance, median (m) | -0.0161 | -0.0088 |
| Resting object lowest-vertex floor distance, median (m) | -0.0256 | -0.0084 |
| Resting object lowest-vertex mean abs distance (m) | 0.0282 | 0.0119 |
| Exact-object mask IoU over episode | 0.7505 | 0.7672 |

The reference NPZ contains vertices but not its own face connectivity and its vertex order differs from the MHR checkpoint. Silhouette validation therefore uses a dense projected-vertex support with a fixed 5-pixel disk radius (the projected nearest-neighbour spacing is ~0.7-3.7 px and this gives baseline support area close to the SAM2 mask area). This avoids applying the checkpoint's incompatible face indices.

## Details

- Dense correspondences use nearest-surface matches under 3 cm, then retain points inside the person mask whose reconstructed depth agrees with aligned MoGe within 8 cm. This removes hidden/back-surface matches. Reprojection: mean 1.608 px, median 1.455 px, p90 2.903 px.
- Static object plateaus contain 93 sampled frames. Contact smooth-min before/after height adjustment: -0.0034 / 0.0000 m.
- Final floor-normal error relative to the median MoGe floor normal: 0.013 deg before contact; contact adjustment is purely along that normal.
- Runtime: 47.0 s on one GPU/CPU process.

## Confidence

**Medium-high.** The camera tilt is corroborated by two independent static cues (human dense reprojection and floor normal), and the object depth scale improves exact-mesh mask IoU without changing mesh size. The largest remaining uncertainty is the released floor plane's absolute scale versus MoGe/SAM3D metric depth; camera orientation is much better constrained than absolute camera height. Residual human-floor penetration of ~1-2 cm is within the released foot-plane tolerance (mean 3.9 cm). World ADD should improve mainly through the corrected camera rotation/translation and the ~1% object depth scale, but ambiguity along the camera ray remains the dominant risk.
