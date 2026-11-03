import json
from pathlib import Path
p=Path('/work/outputs');v=json.loads((p/'validation.json').read_text());r=json.loads((p/'runtime.json').read_text());report=p/'REPORT.md';s=report.read_text().split('## Final validation')[0].rstrip()
s+='\n\n## Final validation\n\n'
s+='The exact required offline command completed successfully. Checks use the final saved poses and the released inputs, without alignment. Mask IoU uses the original K and unscaled mesh projections, with person-covered background excluded equally for baseline and final. It is a fitting diagnostic, not a held-out accuracy metric.\n\n'
s+='| Check | Baseline | Final |\n|---|---:|---:|\n'
s+=f"| Mean visible-mask IoU, all frames | {v['baseline_visible_mask_IoU_mean']:.4f} | {v['final_visible_mask_IoU_mean']:.4f} |\n"
s+=f"| Rotation steps above 45 degrees | {v['baseline_steps_over_45']} | {v['final_steps_over_45']} |\n"
s+=f"| Median rotation step | {v['baseline_rotation_step_deg_percentiles'][0]:.2f} deg | {v['final_rotation_step_deg_percentiles'][0]:.2f} deg |\n"
s+=f"| Maximum rotation step | {v['baseline_rotation_step_deg_percentiles'][-1]:.2f} deg | {v['final_rotation_step_deg_percentiles'][-1]:.2f} deg |\n"
s+=f"\nMask overlap improves in {v['mask_iou_improved_frames']}/{v['frames']} frames. Median image-centre adjustment: {v['pixel_centre_change_percentiles'][2]:.2f} px; maximum: {v['pixel_centre_change_percentiles'][-1]:.2f} px. Median rotation change from baseline: {v['rotation_change_deg_percentiles'][2]:.2f} degrees, reflecting the handle-side repairs rather than an assumed axial symmetry.\n"
s+=f"\nThe NPZ has key `T_world_mesh`, shape `(781, 4, 4)`, dtype `float64`, with finite entries throughout. Maximum change in per-frame Euclidean camera distance: {v['camera_distance_max_change_m']:.3g} m. Maximum rotation orthogonality error: {v['rotation_orthogonality_max']:.3g}; determinant range: [{v['rotation_determinant_range'][0]:.16f}, {v['rotation_determinant_range'][1]:.16f}]. Homogeneous rows are valid.\n"
s+=f"\nFinal complete offline run: {r['elapsed_seconds']:.1f} s ({r['elapsed_seconds']/60:.2f} min), including video/mask I/O, fitting and writing outputs, on one RTX 6000 Ada GPU. Summed wall-clock bounds of all CUDA-using exploration, discarded fitting trials, final generation and verification are conservatively below 35 minutes; concurrent experiments are counted separately in that bound. This is below the 45-minute GPU budget.\n"
s+='\nReview `comparison.jpg` for video/baseline/final overlays, `validation.json` for numerical checks, `runtime.json` for final-run timing, and `fit_diagnostics.npz` / `solver_candidates.npz` for fitting evidence. Exploratory scripts and images are archived in `experiments/`; `solve.py` does not read them. All confidence claims concern visible evidence and continuity; ground-truth ADD has not been evaluated.\n'
report.write_text(s)
print('Report finalized')
