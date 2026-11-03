# White-desk trajectory recovery

## Result

The final trajectory is in `object_poses_world.npz` under `T_world_mesh`
(794 x 4 x 4, float64). It was produced by the required offline command:

```bash
python /work/outputs/solve.py --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The solve took **about 8 s wall time** on the allocated RTX 6000 Ada GPU
(5.6 s measured inside the final script run). The GPU footprint is one mesh and
raster buffers no larger than 1536 x 1152, comfortably below the 9 GB limit.

## Method

The released per-frame trajectory is a useful moving-object proposal, but its
static phases contain obvious tracker drift. I kept its frame-specific view
evidence during manipulation and replaced the static portions with one pooled
pose per phase.

1. **Static-pose pooling.** For a candidate world pose, the exact textured
   mesh was rasterized with nvdiffrast at 768 x 576. The objective pooled four
   fully visible frames for each static pose and combined:
   - object-mask IoU;
   - rendered pixels outside `object | person` (false positives);
   - visible object pixels missed by the render (false negatives);
   - symmetric boundary chamfer distance.
   A deterministic 6-DoF coordinate search started from baseline frame 0 for
   the initial pose and baseline frame 650 for the final pose. The recovered
   perturbations were about `(-1, 0, -3) cm` and
   `(+0.4, -2.25, +0.75) deg` for the initial pose, and
   `(0, 0, -2) cm` and `(+3, 0, 0) deg` for the final pose.
2. **Phase decoding.** The object contour is stationary through frame 260.
   Grasp contact starts earlier (around frame 141), but contour motion starts
   after 260. The desk reaches its floor pose around frame 620 and is settled by
   frame 650. The final pose is held for frames 650-793.
3. **Transition anchoring.** A rigid world correction is applied to the
   moving proposal throughout frames 261-290, ramping from the pooled initial
   pose back to the per-frame proposal. This removes accumulated static-phase
   tracker bias without introducing a one-frame pose jump. Frames 620-649
   similarly blend from the moving proposal into the pooled final pose.
4. **Moving-phase temporal filtering.** For frames 261-641, rotation
   quaternions (sign-continuous) and translations were filtered with an
   11-frame, second-order Savitzky-Golay filter. This preserves the lowering
   motion while reducing high-frequency pose jitter.
5. **Symmetry test.** I rendered the baseline and post/pre-multiplied 180 deg
   flips about all three mesh and world axes on representative initial and
   final frames. No flip won both silhouette and texture evidence. For example,
   at frame 650 the best mesh-x flip had a comparable silhouette IoU
   (0.877 vs 0.868 for the baseline), but its masked grayscale correlation was
   -0.052 versus +0.152 for the baseline. The unflipped stream was retained.
6. **Depth policy.** MoGe-2 depth was used as a diagnostic, not as an absolute
   pose target. Moving the final desk about 0.3 m toward the camera would
   reduce the raw depth residual, but silhouette IoU dropped from about 0.87
   to 0.63 at frame 650. The silhouette, hand-contact, and monocular depth cues
   therefore did not justify a depth snap. No floor-plane snapping was used.

## Phases and confidence

| Frames | Phase | Decision | Confidence |
|---|---|---|---|
| 0-260 | rest / grasp preparation | one constant world pose | High |
| 261-290 | first motion | ramp out of static correction | Medium |
| 291-619 | lowering while carried | smoothed moving proposal | Medium-high |
| 620-649 | settle onto floor | blend to final pooled pose | Medium-high |
| 650-793 | rest / person leaves | one constant world pose | High |

The initial pose has especially strong support: the object is fully visible in
the fitting frames and remains visually fixed for hundreds of frames. The
final pose is supported by 144 frames, including a long period after the person
releases the desk. The lowering phase is harder because the hand occludes the
object and several surfaces become nearly edge-on; there the baseline's
frame-specific registration was retained.

## Validation (no ground truth)

All numbers below are diagnostics, not ground-truth error. Masks were compared
at half resolution (768 x 576). `FP` is rendered pixels outside
`object | person`; `FN` is visible object pixels missed by the render. Chamfer
is in half-resolution pixels, so multiply by two for approximate original
pixels. Depth MAD is the per-frame median absolute residual around the frame's
median MoGe-2 offset; it measures depth-shape agreement, not absolute scale.

### Silhouette agreement

| Phase | Method | IoU | FP | FN | Chamfer |
|---|---|---:|---:|---:|---:|
| 0-260 rest | baseline | 0.629 | 0.104 | 0.140 | 3.112 |
| 0-260 rest | **result** | **0.683** | **0.094** | **0.083** | **2.734** |
| 261-290 onset | baseline | 0.503 | **0.127** | 0.153 | 3.713 |
| 261-290 onset | **result** | **0.523** | 0.139 | **0.118** | **3.591** |
| 291-619 lowering | baseline | 0.7528 | 0.1128 | 0.1141 | 1.640 |
| 291-619 lowering | **result** | **0.7538** | **0.1122** | **0.1134** | **1.614** |
| 620-649 settle | baseline | 0.8853 | 0.0394 | 0.0600 | 1.451 |
| 620-649 settle | **result** | **0.8931** | **0.0376** | **0.0544** | **1.398** |
| 650-793 rest | baseline | 0.7442 | 0.0473 | 0.0823 | 2.993 |
| 650-793 rest | **result** | **0.7585** | **0.0398** | **0.0706** | **2.864** |
| Overall | baseline | 0.7062 | 0.0956 | 0.1163 | 2.440 |
| Overall | **result** | **0.7279** | **0.0912** | **0.0937** | **2.276** |

### Physical and temporal checks

| Phase | Method | Depth MAD (m) | Rotation step (deg/frame) | Center accel. (m/frame^2) | Right-hand min dist. (m) |
|---|---|---:|---:|---:|---:|
| 0-260 rest | baseline | 0.0718 | 0.258 | 0.00734 | 0.0086 |
| 0-260 rest | **result** | **0.0672** | **0.0011** | **0.00000** | 0.0119 |
| 261-290 onset | baseline | 0.1164 | 0.422 | 0.01092 | 0.00422 |
| 261-290 onset | **result** | 0.1188 | **0.338** | **0.00128** | **0.00407** |
| 291-619 lowering | baseline | 0.1034 | 0.498 | 0.01713 | 0.00677 |
| 291-619 lowering | **result** | 0.1036 | **0.319** | **0.00237** | **0.00633** |
| 620-649 settle | baseline | 0.0526 | 0.316 | 0.01506 | 0.0884 |
| 620-649 settle | **result** | 0.0530 | **0.088** | **0.00183** | 0.0898 |
| 650-793 rest | baseline | 0.0485 | 0.322 | 0.01286 | n/a |
| 650-793 rest | **result** | 0.0498 | **0.000** | **0.00000** | n/a |
| Overall | baseline | 0.0816 | 0.377 | 0.01283 | |
| Overall | **result** | **0.0805** | **0.149** | **0.00110** | |

Static-pose consistency also improved directly:

| Phase | Method | Rotation spread, p95 (deg) | Center spread, std (m) |
|---|---|---:|---:|
| 0-260 | baseline | 1.286 | 0.00788 |
| 0-260 | **result** | **~0** | **~0** |
| 650-793 | baseline | 1.081 | 0.01731 |
| 650-793 | **result** | **~0** | **~0** |

The result improves mask overlap and boundary agreement in every phase, and
removes nearly all static-phase drift. The moving proposal is preserved where
occlusion makes the silhouette weak, rather than being overfit to an
incomplete object mask. Right-hand distance remains at approximately 4-7 mm
during the carrying interval, consistent with the reconstructed human contact.
Small increases in a few metrics (onset FP, final depth MAD, and release-frame
hand distance) were accepted because the object is occluded or no longer in
contact at those frames and the corresponding silhouette/temporal evidence is
strongly better.

## Reproduction and data use

- No ground truth, challenge annotations, or external episode data were used.
- No network access was used by the solver.
- `solve.py` recomputes the pose from `/work/inputs` only.
- The input files used by the solver are `camera.json`, `episode.json`,
  `object_mesh.glb`, `masks.h5`, and `reconstruction.npz`.
- MoGe-2 aligned depth in `/pipeline` was read only for offline validation.
- The solve is deterministic: fixed seeds and no stochastic optimization.
