# Monocular 6-DoF Bottle Trajectory

## Result

- Output: `object_poses_world.npz`, key `T_world_mesh`, shape `(859, 4, 4)`, `float64`.
- SHA-256: `d58955d35116f07063fece76f6ce4471808c2ca36b72d548f760e32dd38d0270`.
- Object: `cyan_water_bottle`; exact supplied textured mesh.
- Released world frame and OpenCV camera convention are used throughout.
- No ground-truth or challenge annotation was accessed.

## Phases found

The hand-contact probability brackets the active interval at frames `96..796`. Mask evidence against the pooled rest poses showed that the object remains stationary after first touch and before final release, so the physical rest phases were extended using a sustained mask-IoU criterion:

| Phase | Frames | Time | Interpretation | Confidence |
|---|---:|---:|---|---|
| Initial rest | 0-115 | 0.00-3.87 s | Bottle standing on the round table before lift-off | High |
| Manipulation | 116-774 | 3.87-25.83 s | Pickup, alternating-hand rotation, and handovers | Medium-high |
| Final rest | 775-858 | 25.83-28.63 s | Bottle replaced on the round table | High |

Eight-frame SE(3) blends are used at the two rest/manipulation boundaries to remove residual depth jumps while keeping the rest poses exactly constant.

## Method

1. **Rest-pose evidence pooling.** Candidate rest frames were scored by rendering the exact mesh and comparing its visible silhouette to the SAM2 object mask on all other rest frames. The best anchor was rotation-refined with a multi-frame mask-IoU search (±12° around camera x/y) and translation-refined with the median visible-mask centroid. Initial anchor: frame 27; final anchor: frame 818. The resulting pose is used unchanged for every frame in each rest phase.
2. **Robust trajectory initialisation.** The released per-frame pose trajectory was converted to continuous quaternions, median-filtered, then fitted with a local robust quadratic model (radius 9 frames, 20° rotation outlier scale, 12 cm translation scale). This removes one-frame network flips without imposing constant velocity.
3. **Image-space translation correction.** The exact mesh mask is rendered for every manipulation frame. The camera-space lateral translation is adjusted to the visible object-mask centroid. Five smoothed correction iterations reduced the centroid error from about 6.1 px to about 2.8 px. Depth was not changed: monocular object scale alone was not considered reliable enough.
4. **Multi-stage Viterbi orientation decoding.** Per frame, candidate rotations about camera x/y were rendered around the current estimate. The data cost is visible-mask disagreement; transition and small-deviation costs penalise nonphysical jumps and favour the temporally filtered path. Coarse ±36° (6° step), medium ±6° (1° step), and fine ±3° (0.5° step) stages were decoded with dynamic programming. Rest poses were fixed at both endpoints.
5. **Spin resolution.** The bottle is nearly rotationally symmetric in silhouette, so the remaining ambiguity is spin about its local z (long) axis. We render the supplied texture in a crop around the object for 24 spin hypotheses (15° spacing), score RGB chromaticity and high-pass texture correlation, and decode a smooth spin path. The transition model forbids one-frame flips larger than 15°, begins/ends at the rest spin, and the selected path is circularly smoothed.
6. **Final cleanup.** Rest phases are reassigned to their exact pooled poses; boundary frames are slerped; all rotations are re-orthonormalised; output is checked finite with shape `(859,4,4)`.

## Validation (no ground truth)

Exact-mesh masks were rendered with the supplied `K` and `T_world_cam`. “Visible IoU” removes pixels covered by a dilated person mask. Centroid error is in pixels. Rotation/translation steps are frame-to-frame changes inside each phase.

| Phase | Method | Mask IoU | Visible IoU | Centroid err (px) | Rot step median / p90 | Trans step median / p90 |
|---|---|---:|---:|---:|---:|---:|
| Initial rest (0-115) | Baseline | 0.778 | 0.843 | 3.94 | 0.00° / 2.61° | 0.05 / 1.00 cm |
| Initial rest (0-115) | Result | **0.931** | **0.970** | **1.20** | **0° / 0°** | **0 / 0 cm** |
| Manipulation (116-774) | Baseline | 0.662 | 0.749 | 6.14 | 6.73° / 23.81° | 1.56 / 3.33 cm |
| Manipulation (116-774) | Result | **0.828** | **0.915** | **2.79** | **6.02° / 15.17°** | **1.07 / 1.86 cm** |
| Final rest (775-858) | Baseline | 0.786 | 0.858 | 6.74 | 0.00° / 14.39° | 0.02 / 1.79 cm |
| Final rest (775-858) | Result | **0.901** | **0.972** | **1.50** | **0° / 0°** | **0 / 0 cm** |

Additional diagnostics:

- **Photometric check:** on 33 manipulation frames spanning 120-760, textured-render chromaticity correlation rose from `0.641` (baseline) to `0.732` (result); 25/33 frames improved. Absolute rendered colour was not trusted because of unknown lighting, so this is used only as a relative check.
- **MoGe depth check:** on 35 frames sampled every 25 frames, median absolute camera-depth residual at rendered object pixels fell from `0.188 m` to `0.155 m`. The result bias versus MoGe is about `0.11 m`, so this weak cue was not used to alter depth.
- **Static consistency:** both rest phases have exactly zero rotation and translation variance in the output.
- **Visual audit:** rendered contours were checked at 30 frames spanning the whole episode. The large baseline failures around frames ~240, ~400, ~600, and ~690 are corrected; no long-segment 180° flip remains visible in silhouette or texture checks.

## Confidence and limitations

- Initial and final rest: **high**. Hundreds of mask observations support one constant world pose, and multi-frame pooling resolves most silhouette ambiguity.
- Manipulation: **medium-high for translation and long-axis orientation**, **medium for spin** during heavy occlusion/motion blur. The mask and texture scores improve strongly, but monocular depth and texture albedo remain ambiguous under unknown lighting.
- The result preserves the baseline depth along each viewing ray except for pooling at rest and the small lateral corrections. No floor snapping was performed.

## Reproducibility and runtime

`solve.py` is deterministic, fixes all random seeds, uses only local inputs and installed packages, and does not read `/work/scratch` or the network. Final command:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Measured end-to-end runtime: `582 s` on the available GPU. Full-resolution raster batches are bounded (at most 169 candidate poses in the coarse stage); no OOM occurred.
