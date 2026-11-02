# Object trajectory refinement report

## Result

Produced object_poses_world.npz with key T_world_mesh, shape **(859, 4, 4)**, dtype **float64**, in the released world frame. Every frame is populated. **267 frames are refined and 592 retain the supplied baseline exactly**, after lossless float32-to-float64 conversion. Input files are unchanged.

The refinements address projected-axis errors, cap/body alignment, and wrong axial roll indicated by the blue latch. They use the original mesh-file coordinates, without symmetry equivalence or alignment to another coordinate frame.

**Camera-to-object-centre distance is preserved in every frame.** The maximum numerical change is 1.33e-15 metres. No floor constraint or snapping is applied. Camera calibration, intrinsics, world frame, and human reconstruction are not re-estimated.

## Evidence

These are image-consistency measurements against the supplied video and masks, **not ground-truth ADD measurements**. They also contribute to optimization and are not independent proof of 3D accuracy.

| Measurement | Baseline | Final |
| --- | ---: | ---: |
| Mean occlusion-aware silhouette IoU, all 859 frames | 0.7521 | 0.8373 |
| Mean occlusion-aware silhouette IoU, 267 edited frames | 0.6358 | 0.9097 |
| Mean blue-latch soft IoU, frames with over 80 detected blue pixels | 0.3355 | 0.5335 |

Examples include frames 240–250, where the baseline bottle axis deviates from the image during the handoff; 543–561, where foreshortening is poorly captured; 594–618, where the projected axis lags the rotation; and 648–722, where both large axis errors and wrong latch-facing orientation occur. The cap is at positive mesh z. The distinctive blue latch occupies one side of the upper bottle and distinguishes rolls that appear equivalent on the cylindrical body.

The initial resting sequence and final settled sequence remain baseline. Changes elsewhere require the same automatic evidence gates, including clearly visible latch mismatches. No frame-specific pose, angle, translation, or interval override is used. Exact edited intervals and per-frame diagnostics are in metrics.json and frame_diagnostics.csv.

Among edited frames, median rotation change is 32.78 degrees, median projected-centre shift is 5.25 pixels, and median 3D translation change is 1.78 cm. Maximum translation change is 6.11 cm, while camera distance remains fixed. Large axial-roll corrections are intentional where the visible latch contradicts the baseline.

## Method

1. Load the supplied GLB with trimesh.load(path, force="scene", process=False).to_geometry(). Merge duplicate vertices while preserving UV seams, retaining the same mesh surface and texture. Transform the baseline into the released camera frame.
2. Decode the supplied video and read the object/person SAM2 masks. Use 160-pixel crops around the baseline projection. Person pixels outside the object mask may occlude the object and are excluded from the rendered-background penalty.
3. Render the exact textured mesh using installed nvdiffrast. Score silhouette IoU, exterior distance, cyan-body versus cap agreement, and blue-latch agreement. Color ratios reduce sensitivity to overall brightness.
4. Initialize from the baseline, the released FoundationPose rotation, nearby baseline rotations, axial-roll alternatives, and an explicit end-flip alternative. Never adopt FoundationPose depth. Search three rotation coordinates and two image-plane coordinates. Construct each translation from a unit camera ray times the baseline camera distance.
5. Use deterministic forward-only coordinate search with fixed order, steps, and tie-breaking. Initial differentiable Adam experiments were rejected because GPU gradient reductions were not exactly repeatable. The final script performs no backward pass.
6. For visible but mismatched latches, add eight axial-roll starts at 45-degree spacing and refine them. This resolves the round bottle body's roll ambiguity using the supplied texture.
7. Require independent projected-axis or latch evidence, substantial image-score gain, improved silhouette IoU, and adequate absolute IoU before admitting a replacement. A three-frame eligibility dilation covers transition edges. Large cap-end reversals need stronger evidence. A confidently visible latch excludes alternatives that hide it. Temporal dynamic programming penalizes abrupt rotation and image-plane changes.
8. Convert accepted poses back into the released world frame. Normalize edited rotations to proper SO(3), restore baseline camera distance in float64, and retain unedited matrices exactly as supplied.

All thresholds and weights, including color criteria, search steps, and acceptance gates, are explicit in solve.py. Visual inspection motivated the cap/latch cues and conservative rules; all final decisions are computed automatically from the inputs, without a manual trajectory table.

## Reproduction and validation

Run offline:

    python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

The single script is self-contained. It reads no exploratory cache, authored helper module, scratch file, pretrained weights, pipeline intermediates, or network resources. It regenerates the NPZ, metrics.json, and frame_diagnostics.csv from supplied inputs and installed packages. Seeds are fixed. The existing CUDA_VISIBLE_DEVICES=0 restricts it to one GPU.

Independent validation checks the exact NPZ key, frame count, float64 dtype, finite values, homogeneous bottom rows, rotation determinants, exact baseline retention, preserved camera distance, and SHA256 equality of every original input file. Edited-rotation orthogonality error is 4.55e-15. Unedited matrices deliberately retain the baseline's small numerical nonorthogonality (maximum 5.36e-6).

Two complete offline executions produced **bitwise-identical pose arrays and identical NPZ SHA256 hashes**. The maximum repeated-run pose difference is 0.0. The second run took 221.56 seconds. Results are recorded in validation.json.

The first full standalone run took **221.01 seconds** wall time, including decoding, GPU rendering/search, selection, and writing. Only one visible GPU was used. Recorded development/search phases and two final executions remain below a conservative 25-minute GPU-process budget, under the 45-minute limit. This is a process-time accounting bound, not a hardware-utilization trace.

## Confidence and limitations

Confidence is high for gross projected-axis corrections and latch-supported roll changes, and moderate for exact out-of-plane tilt or cap orientation under hand occlusion. The small monocular object, imperfect masks, estimated intrinsics, and lighting differences leave residual ambiguity. Distance is intentionally not inferred from object scale. Uncertain frames retain baseline.

No ground-truth poses or challenge annotations were used or sought. Actual ADD improvement cannot be measured here and is not claimed as a verified result. The refinement is supported by substantially better image agreement and asymmetric texture evidence, with conservative baseline retention.

## Files

- object_poses_world.npz: final world poses for every frame.
- solve.py: single deterministic offline producer.
- TOOLS.md: package versions, inspected revisions, and approaches tried; no downloads or external weights.
- metrics.json and frame_diagnostics.csv: aggregate and per-frame evidence.
- validation.json and input_sha256.json: structural, integrity, and repeatability checks.
- final_evidence.jpg: original / baseline / final rendered comparison from the saved result.
- solve_run.log and repeat_run.log: final execution logs.

Final NPZ SHA256: 98820a3cc3195748dd56fdec6a1653643fad07468456c481b6ca97369f1dbd0b.
