"""Assemble the human-readable report from final validation artifacts."""
from pathlib import Path
import json
import numpy as np

out=Path('/work/outputs')
v=json.loads((out/'validation.json').read_text())
f=json.loads((out/'full_mesh_validation.json').read_text())
r=json.loads((out/'reproducibility.json').read_text())
s=np.load(out/'selection_diagnostics.npz')
changed=s['changed']
examples=[]
for frame in (123,203,246,329,339,370,411,470,472,555,596,606,658):
 if changed[frame]:
  examples.append(f'| {frame} | {s["base_loss"][frame]:.3f} | {s["selected_loss"][frame]:.3f} | {np.rad2deg(s["axis_change"][frame]):.1f} |')
text=f'''The result is `object_poses_world.npz`, containing `T_world_mesh` as a float64 array of shape `(803, 4, 4)`, in the released world frame and the original mesh-file coordinate convention. {v['changed_frames']} frames were corrected; {v['unchanged_frames']} retain the supplied baseline exactly. No alignment or change to the released camera was applied.

The changes correct the bat's projected long-axis direction and its position across the image. During handling the baseline sometimes points the handle towards the wide barrel, or places the projected axis outside the visible bat. Frames around 329 are a clear direction reversal; frames 123, 246, 339, 411, 555, 596, and 606 show other large projection errors. `evidence.jpg` shows selected baseline/output axes over the actual video and masks. Magenta is the baseline, yellow the output, red the handle end, and blue the barrel end.

The baseline's per-frame Euclidean camera distance of the mesh centre is a hard constraint. Its maximum numerical change is {v['max_camera_distance_change_metres']:.3g} m. Image position changes move the centre along a sphere around the camera, rather than estimating a new object depth. The object was not snapped to the floor. The supplied human-calibrated distances, world frame, and camera calibration were retained.

The offline algorithm is:

1. Read the object/person masks and build 128-square crops containing the baseline projection and observed bat. Initialize candidates from both signs of the mask's principal axis, several inclinations, the baseline, and the supplied FoundationPose axis. Use both baseline-centred and mask-centred starts.
2. Rasterize a 6,000-triangle proxy derived from the exact supplied mesh. Optimize two long-axis angles and two image coordinates, with the camera distance fixed. Apply the shortest rotation from the original axis to the fitted axis; there is no independent axial-roll fit. The proxy's sampled surface error is 0.104 mm mean, 0.388 mm at the 99th percentile, and 0.758 mm maximum.
3. Minimize missed object area plus rendered excess area outside the person mask, normalized by visible object area. Dilating the person mask by 5 pixels avoids treating hand occlusion as a bat boundary. A tilt prior discourages changing monocular inclination without evidence. Finite differences of hard rasterized silhouettes give a deterministic numerical gradient; random seeds, batch size, iteration count, and computation settings are fixed.
4. Check barrel/handle brightness in the video: the handle is narrow and dark, while the barrel is wider and light. Compare eroded-mask pixel medians in separated axial regions, and apply this check only when both regions have enough pixels. It is a weak polarity check, not a texture or roll estimator.
5. Accept only when baseline visible error exceeds 0.34, absolute improvement exceeds 0.14, remaining error is below 0.38, relative remaining error is below 0.72 of baseline, and centre movement is below 11 cm. Contradictory brightness rejects the fit. Tilt changes above 25 degrees additionally require over 0.4 error improvement, remaining error below 0.30, and agreement with the supplied FoundationPose axis (within 25 degrees of inclination and 30 degrees of projected direction). All other frames retain the baseline.

The only manually chosen frame intervals are explicitly commented in `solve.py`: frames 0–112 and 690–802 retain the baseline. In the video these are resting portions where the baseline's handle/barrel ends and axis already align well. Small silhouette-thickness discrepancies there are insufficient evidence to change a strong calibrated reconstruction. Other decisions use the automatic criteria above.

Independent verification uses the full unmodified 98,866-triangle input mesh, not the optimization proxy, at 256-square crop resolution. Across the {f['changed_frames']} corrected frames, normalized visible silhouette error decreases from {f['baseline_visible_error']:.3f} to {f['output_visible_error']:.3f}; {100*f['fraction_changed_frames_improved']:.1f}% of corrected frames improve. Across all 803 frames, the fitting-resolution error decreases from {v['baseline_mean_visible_silhouette_error']:.3f} to {v['output_mean_visible_silhouette_error']:.3f}. These are image-consistency measurements, not ground-truth ADD scores.

| Frame | Baseline visible error | Output visible error | Long-axis correction (degrees) |
|---|---:|---:|---:|
{chr(10).join(examples)}

Verification also checks finite float64 transforms for all frames, exact homogeneous bottom rows, exact retention of every rejected baseline pose, and camera-distance preservation. Corrected rotations are projected onto SO(3); their maximum orthogonality residual is {f['max_corrected_rotation_orthogonality_error']:.3g}. The untouched baseline's original small floating-point rotation residuals are deliberately retained. Two complete runs produce identical pose arrays: `{r['poses_bitwise_equal']}`. See `validation.json`, `full_mesh_validation.json`, `reproducibility.json`, and the per-frame NPZ diagnostics for the checks.

Confidence is high in the accepted image-plane and projected-axis corrections, and moderate in the resulting full 3D improvement. Depth-sign and axial-roll ambiguities cannot be resolved confidently from all monocular views of this nearly round bat. The solver inherits the baseline's roll by the minimum required axis correction, regularizes inclination, and keeps the baseline in uncertain or heavily occluded cases. Some visible baseline errors therefore remain, including frame 462. No true poses, hidden annotations, external challenge data, or ground-truth alignment were accessed. True ADD improvement is not measured here.

Reproduce offline with:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The measured complete run takes {v['total_seconds']:.1f} seconds ({v['total_seconds']/60:.2f} minutes), including {v['mask_preparation_seconds']:.1f} seconds of mask preparation and {v['gpu_fit_wall_seconds']:.1f} seconds of GPU fitting wall time. The independent full-mesh check takes {f['elapsed_seconds']:.1f} seconds. Only the selected GPU was used; total GPU-fitting/validation time across development and the two final runs remained under 15 minutes, below the 45-minute budget. No downloads or network are needed. `TOOLS.md` lists the installed package versions and the input-derived rendering proxy required by the solver. Optional scripts `build_render_mesh.py`, `make_preview.py`, and `validate.py` document proxy construction, visual evidence, and independent checks; they are not prerequisites for the reproduction command.
'''
(out/'REPORT.md').write_text(text)
