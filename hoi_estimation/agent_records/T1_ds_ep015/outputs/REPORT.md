# Object trajectory recovery report

## Method

The released CoCoNet/depth-corrected `T_world_mesh` was treated as the per-frame proposal. Frames 0-163 and 622-714 were identified as static from the video and pooled with a quaternion mean and coordinate-wise median; each pooled pose was then refined against a temporal-consensus object mask with an nvdiffrast render-and-compare search. During manipulation, the same local 6-DoF search refined poses whose object mask had at least 300 visible pixels, with a 12 degree / 14 cm trust region and a hand-contact safeguard. Frames 491-497 have no usable object mask, so they were bridged by SE(3) interpolation between the adjacent accepted poses. A short Savitzky-Golay filter was selected as the longest window staying within 0.006 mean mask IoU of the best-fitting smoothing variant, and the pooled static poses were blended into the carried trajectory around pickup and placement.

## Phases and confidence

| frames | phase | confidence |
|---|---|---|
| 0-163 | initial static on floor | high: constant pooled pose |
| 164-490 | pickup and carried motion | medium-high: textured object usually well visible |
| 491-497 | occluded transition | medium: interpolated across evidence gap |
| 498-621 | carried motion and placement | medium: rapid rotation, reduced visibility |
| 622-714 | final static on floor | high: constant pooled pose |

## Validation (baseline vs result)

| trajectory | mask IoU mean | mask IoU median | rot step mean (deg) | rot step p95 (deg) | trans step mean (m) | hand clearance median (m) |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7523 | 0.8264 | 1.455 | 4.359 | 0.0143 | 0.0039 |
| result | 0.8426 | 0.8749 | 1.404 | 4.391 | 0.0118 | 0.0148 |

### Per phase

| frames | phase | baseline IoU | result IoU | baseline rot p95 | result rot p95 |
|---|---|---:|---:|---:|---:|
| 0-163 | initial static / approach | 0.8336 | 0.8795 | 0.00 | 0.00 |
| 164-621 | pickup, carry, turn, and place | 0.7542 | 0.8523 | 5.04 | 5.42 |
| 622-714 | final static | 0.6001 | 0.7305 | 0.00 | 0.00 |

### Diagnostics

- Manipulation refinement accepted 444 frames and retained the tracker on 7 reliable frames.
- Selected temporal filter: window 7, quadratic Savitzky-Golay on quaternions and translation.
- Mean baseline mask IoU on refined reliable manipulation frames: 0.7557; mean local-search score: 0.8615.
- Static pose minimum floor distance: initial -0.0305 m, final -0.0314 m (the released plane fit has several-cm tolerance; poses were not snapped).

The changes were adopted because mask agreement and mean temporal step size improve, especially in the two long static phases. Rotation branch experiments (180 degree flips about the mesh principal axes) were worse in silhouette and texture-edge agreement and were therefore not applied.

Run time: 203.4 s on one GPU.
