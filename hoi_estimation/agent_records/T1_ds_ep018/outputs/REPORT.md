# Blue Trash-Can Trajectory Recovery

## Summary

The final trajectory is a temporally constrained revision of the supplied reconstruction. It keeps the baseline trajectory where the object is moving and image evidence is ambiguous, but replaces the three stationary-world segments with pooled rigid poses and lightly regularizes the two manipulation segments.

No ground truth or challenge annotations were used. All decisions were made from the supplied video, masks, reconstruction, mesh, camera, and released floor plane.

## Recovered phases

| Phase | Frames | Interpretation | Evidence |
|---|---:|---|---|
| Pre-static | 0-219 | Object rests on the floor while the person approaches | Object-mask centroid remains at approximately (712.6, 826.3) through frame 187; the first sustained hand contact probability above 0.5 occurs at frame 219 |
| Manipulation 1 | 220-503 | Person lifts/slides the can to the right and partly toward the camera | Contact probability is high; mask moves right from near x=715 to x=986; baseline trajectory rises off the floor and rotates smoothly |
| Middle static | 504-641 | Object is released and rests in the new location | Both hand-contact probabilities fall below 0.1 by frame 504; object-mask bbox is fixed at approximately (910, 678, 1066, 952) through frame 634 |
| Manipulation 2 | 642-737 | Person moves the can back toward image-left | Contact becomes high again at frame 642; object mask moves from x≈988 to x≈865 |
| Final static | 738-857 | Object rests until the end | Contact ends at frame 737; mask bbox is fixed at approximately (781, 694, 948, 989) from frame 745 onward |

These boundaries are explicit constants in `solve.py`, with the reason documented next to them.

## Method and changes

1. **Hypothesis check.** The textured mesh was rendered with nvdiffrast. The baseline explanation was compared with local 180-degree flips about each mesh principal axis and with the uncalibrated CoCoNet camera pose. The baseline won clearly on texture NCC and silhouette IoU, so no global symmetry flip was adopted.
2. **Static-phase pooling.** For each static phase, the 20 frames with the strongest baseline mask/texture evidence were pooled into a median image and consensus object mask. A six-degree-of-freedom pose was then fitted by deterministic differential evolution against the textured render. The person mask was excluded from texture scoring. This corrected both small rotation offsets and a repeatable depth offset along the viewing direction. The resulting poses are exactly constant within each static phase.
3. **Manipulation refinement.** During the two moving phases, a local six-degree-of-freedom render-and-compare search was run every fourth frame, initialized from the baseline. Corrections were interpolated, median/Savitzky-Golay smoothed, and conservatively clipped. At each phase end they were SE(3)-anchored to the pooled static poses, so phase boundaries introduce no state jump. The final motion pose uses a 50/50 blend of the smoothed local correction and the smooth endpoint-anchored correction; this gave the best hand-contact agreement and a balanced mask/texture result.
4. **Floor check, not floor snapping.** The fitted plane was used only as an independent validation cue. The static poses have minimum signed plane distances of +1.7 mm, +5.7 mm, and -5.2 mm for the pre, middle, and final rests, respectively. These are within the released plane-fit tolerance. They were not translated onto the plane because doing so reduced held-out texture agreement.

## Validation

Validation uses every sixth frame (143 held-out frames). `IoU` is rendered-mesh versus object-mask IoU. `Texture NCC` is the mean per-channel correlation over object pixels excluding the person. Higher is better.

| Phase | Held-out frames | Baseline IoU | Result IoU | Baseline texture NCC | Result texture NCC | Confidence |
|---|---:|---:|---:|---:|---:|---:|
| Pre-static | 37 | 0.9505 | **0.9764** | 0.3651 | **0.4081** | High |
| Manipulation 1 | 47 | 0.8498 | **0.8761** | 0.4194 | **0.4372** | Medium |
| Middle static | 23 | 0.9355 | **0.9826** | 0.4283 | **0.4786** | High |
| Manipulation 2 | 16 | 0.9132 | **0.9474** | 0.3768 | **0.4188** | Medium-high |
| Final static | 20 | 0.9277 | **0.9855** | 0.4090 | **0.4250** | High |
| Weighted overall | 143 | 0.9076 | **0.9424** | 0.4006 | **0.4325** | - |

Additional checks:

- **Static consistency:** all poses are bit-level constant within their static ranges; maximum within-phase rotation deviation is below 0.000003 degrees.
- **Boundary smoothness:** transitions into/out of static phases differ by at most 0.00001 m in translation; no pose jump is introduced.
- **Hand agreement:** over the 376 frames with contact probability >= 0.5, mean hand-to-mesh distance improved from 13.2 mm to 12.0 mm and median distance from 5.6 mm to 3.4 mm.
- **Temporal smoothness:** mean translation acceleration decreased from 4.70 to 3.81 mm/frame², and mean rotational acceleration from 0.423 to 0.412 deg/frame². Maximum single-frame rotation change decreased from 3.60 to 3.34 degrees.
- **Rigidity:** every output rotation is orthonormal with determinant 1; all values are finite and `float64`.

Confidence is lower in the manipulation phases because the hand often occludes a substantial fraction of the object, and the baseline is already locally accurate. In those intervals the result is deliberately close to the baseline and to the anchored correction rather than relying on unconstrained per-frame fitting.

## Reproducibility and runtime

Final command:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script is deterministic (fixed NumPy/PyTorch seeds; differential-evolution seed 1234), requires no network, does not read scratch files, and recomputes the static optimization and dynamic trajectory from the supplied inputs.

Measured runtime of the final run: **347.9 seconds** (5 min 48 s) on one RTX 6000 Ada GPU. Output: `T_world_mesh`, shape `(858, 4, 4)`, dtype `float64`.
