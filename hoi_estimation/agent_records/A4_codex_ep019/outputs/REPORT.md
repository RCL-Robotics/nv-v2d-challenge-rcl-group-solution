# Cane trajectory result

Wrote `object_poses_world.npz`: `T_world_mesh` is a finite float64 array of shape (886, 4, 4) in the released world frame and original mesh-file coordinates. Corrected 739 frames; copied the original baseline exactly in 147 uncertain frames, including empty-mask frames 741–742.

## Changes and evidence

The baseline has visible swing-direction errors, incorrect handle-facing directions, and a late handle/base reversal. Examples at frames 210, 770, and 805 have clear silhouette evidence. The cane's curved handle, four-foot base, and orange/black pattern distinguish many visually similar rotations. `final_evidence_0.jpg` through `final_evidence_2.jpg` show raw input, baseline projection, and the final projection. Occluded cases such as frames 241, 560, and 630 retain the baseline under the conservative acceptance criteria.

On the 739 changed frames, median visible-silhouette IoU increases from 0.248 to 0.613; mean robust render-distance cost decreases from 13.190 to 1.175. Across all frames, including unchanged fallbacks, mean cost decreases from 21.014 to 10.993. These are image-fit diagnostics using the supplied masks and video, not independent pose accuracy measurements or ADD.

The camera-to-mesh-origin distance is constrained to the baseline in every frame. Maximum measured change is 1.33e-15 m. No depth-from-object scaling, ground-plane snapping, human-depth recalibration, or world alignment was applied. Median image-plane translation change among corrected poses is 3.86 cm. Large rotation changes are intentional where image evidence resolves a wrong handle direction or end-for-end flip.

## Method and conservative decisions

1. Load the exact textured mesh with `trimesh.load(..., force="scene", process=False).to_geometry()`. Sample its surface with seed 0 and classify its orange texture.
2. For each nonempty mask, fit 17 starting rotations using the baseline, FoundationPose, both projected long-axis directions, and four rolls. Optimize rotation and image-plane centre direction, with baseline camera distance fixed. Robust bidirectional distances explain object-mask pixels and orange pixels; the person mask allows occlusion. Missing orange evidence disables the color terms.
3. Render every candidate and the exact original baseline with the full textured mesh. Score silhouette and orange-pattern distances in source-image pixel units with a robust noise allowance. These render checks use mesh visibility rather than the surface samples used for optimization.
4. Select a continuous trajectory using dynamic programming, penalizing rotational discontinuities and weakly preferring the original pose when image scores tie. A projected-hand cue is active during frames 110–788: a commented constant based on observing the supplied grasp/swing interval, not an external annotation. At poor fits and discontinuities, refit rotations initialized from neighboring frames and validate these new proposals against the current image.
5. Accept a correction only if the object mask has at least 600 pixels, at least 55% of the rendered object is visible, render cost is at most 3, and cost improves by at least 0.75 and 20% over the baseline. These explicit thresholds reject small visible fragments and poorly explained poses. All other frames copy the baseline exactly.
6. Project accepted rotations onto SO(3), restore baseline camera distance at float64 precision, and transform into the released world frame. Exact baseline copies retain their original float32 rotation roundoff (maximum orthogonality error across the whole output: 2.2e-06).

## Reproduction and validation

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

This command was run successfully twice with the final script; the output arrays were bit-for-bit identical. Both runs validated every cached asset and input SHA-256. All 886 matrices are finite; homogeneous rows, baseline fallbacks, and camera distances were checked. `validation.json` contains detailed measurements and `acceptance.npz` records the per-frame acceptance decision.

The default run uses two locally generated fitting/validation caches under `/work/tools`; they contain intermediate hypotheses and their image evidence, not ground truth. The script computes temporal selection, conservative acceptance, exact distance restoration, and final world poses. The cache manifests verify the video, mesh, masks, intrinsics, and reconstruction. `--recompute` runs the same included fitting and rendering code without either cache. No scratch files, exploratory scripts, network, pipeline outputs, downloaded models, or annotations are needed. See `TOOLS.md` for dependencies and cache hashes.

Default offline run time was 3.4 seconds after import; initial fitting took about 458 seconds and the final validation/refinement pass about 447 seconds using the initial-fit cache. GPU-backed runs during development, including abandoned attempts, consumed approximately 40 minutes of cumulative process wall time; actual GPU-active time is lower. Only the already-selected GPU 0 was used, within the 45-minute GPU budget.

## Confidence and limitations

High confidence: corrections to the visible handle/base reversal and gross projected swing errors. Moderate confidence: fine roll and out-of-plane tilt in accepted frames. The monocular video, estimated intrinsics, person occlusion, and motion blur leave ambiguity; 147 frames therefore remain exactly at baseline. Acceptance boundaries can retain baseline temporal jumps. The unchanged camera distance can also leave residual silhouette-length disagreement. No ground truth was accessed, and an ADD improvement cannot be claimed from the available evidence alone.
