# Episode 11 (light_blue_book): trajectory-level object pose recovery

Output: `object_poses_world.npz` (`T_world_mesh`, 901×4×4 float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(181 s wall time on one RTX 6000 Ada, peak GPU memory 5.4 GB; about 20 s of that is reading the video and masks).

## 1. What happens in the video (phases)
`task.txt` says "pick up the book from the table and place it on the floor". The video actually shows a longer
sequence: table → floor → table → floor (with a slide) → held behind the table. The book is small in the image
(about 70×30 px at 3.1–3.6 m), so 1 px ≈ 4 mm.

| phase | frames | state | how found |
|---|---|---|---|
| A | 0–79 | lies untouched on the table (hand reaches it ~50–80) | frame-difference signal in the object region, mask bbox, crops checked by eye |
| M0 | 80–211 | lifted, carried, lowered under the table to the floor (passes behind/under the table edge ~130–150) | |
| B | 212–303 | lies on the floor under the table, hand resting on it | bbox bottom settles at 212; lift at ~305 |
| M1 | 304–404 | lifted, carried back up (hidden behind the table edge ~355–365), put on the table | |
| C | 405–466 | lies on the table | |
| M2 | 467–655 | lifted, turned upright in front of the torso (500–550), carried down under the table (570–590, partly hidden by the table), put on the floor | |
| D1 | 656–669 | on the floor | |
| M3 | 670–694 | hands slide and rotate the book on the floor | visible yaw change in fixed crops |
| D2 | 695–747 | on the floor | |
| M4 | 748–900 | lifted and carried up behind the table; 830–900 held with both hands behind the far table edge, partly hidden by the table top, slowly lowered | *not* static: the visible strip shrinks and drops |

The boundaries are explicit constants in `solve.py` (`STATIC`), with comments. I chose them from the automatic
motion signal (mean |I_t − I_{t−3}| over unoccluded object pixels; static is below ~3 grey levels) plus the
stability of the mask centroid and bbox, then checked them by eye on fixed-position crops.

## 2. Diagnosis of the baseline
- The baseline equals CoCoNet's camera-frame estimate up to a constant 1.3° rotation (the change from `K_pipeline`
  to `K`) plus a per-frame depth correction.
- **No symmetry flips.** Between consecutive frames the baseline's rotation is always closest to the identity
  rather than to one of the three 180° box flips. The largest jumps are 20–35°/frame, all during occlusion.
- **The flip identity looks correct.** Rendered textured-mesh overlays agree with the image on the spine side and
  on the end where the spine's dark publisher logo sits. This holds in A, C and D1 (logo visible) and in B and D2
  (pages side vs spine side). My automatic texture score separates "spine vs pages towards the camera" reliably.
  It does **not** reliably separate the front cover from the back cover at this resolution: different
  optimisation runs gave different winners. So I kept the baseline's flip identity everywhere and searched no
  flips.
- **Large position errors where the book is hidden.** Examples:
  - ~575–590: the baseline puts the book about 20 px (~7 cm) above the table edge, hidden behind the table top. The
    video shows it *below* the table top.
  - ~135–150 and ~320–335: similar offsets.
- **Rotation jitter and jumps.** p95 rotation speed is 11.8°/frame overall and up to 18.7°/frame in carried
  segments. In static phases the baseline rotation jitters by about 1–2° and depth by up to 3 cm.

## 3. Method (`solve.py`, `hoi_lib.py`)
1. **Evidence per hypothesis** (nvdiffrast, textured mesh, rendered in a 192×144 crop with 2× supersampling):
   - **Silhouette:** L2 between the antialiased render and the SAM2 object mask, plus a chamfer term (distance
     transform of the mask) that still pulls the render towards the mask when they don't overlap.
   - **Occlusion handling:**
     - Person pixels within 14 px of the projected MHR hand keypoints are treated as occluders and ignored.
     - Other person pixels count half as evidence that the object is absent there. Without this, the render
       could slide over the torso for free.
     - Rendered pixels behind the static background depth (+5 cm) are treated as hidden by the table. The
       background depth is the per-pixel median of the MoGe-2 aligned depth over the episode.
   - **Shading-invariant texture score:** R² of regressing the blurred grey image on albedo·[1, n_x, n_y, n_z]
     and a constant, i.e. first-order lighting with unknown gain and offset. A plain NCC was negative on the
     floor because of shading.
2. **Depth target:** the baseline camera-frame z, median-filtered and smoothed (2nd-difference least squares),
   constant within each static phase. All candidates are moved along their viewing ray to this depth.
3. **Static phases:** one constant camera-frame pose per phase (static camera, so constant in the world too).
   - Starting point: the pooled baseline pose (chordal-mean rotation, median position), flattened onto the support
     plane (floor normal; the table is assumed parallel to the floor, and the baseline agrees to <1°).
   - Refinement: joint fit over the 12 least-occluded frames, three yaw starts (−4°, 0°, +4°), a prior towards the
     pooled rotation and a soft lie-flat term.
   - **Accept/reject rule:** keep the refined pose only if, over all frames of the phase, it raises IoU by >0.01
     and does not lower texture R² by >0.01 compared with the pooled baseline pose. Otherwise use the pooled
     baseline pose.
   - Outcome: refined for A, C, D2; pooled baseline for B and D1. In D1 the refined fit shortens the book
     under the hand. I confirmed by eye that the baseline is right there: the spine logo is visible at the end.
4. **Carried / manipulated segments:**
   - (a) **Shift search:** for every frame, find the best image-plane shift of the baseline silhouette by
     correlation (±40 px).
   - (b) **Viterbi** over {baseline, shifted baseline}, with a translation-change transition cost (σ = 5 cm/frame).
   - (c) **Joint trajectory optimisation** of all poses in the segment: image terms (silhouette + texture,
     weight 1.0) + second-difference smoothness on rotation and translation + a weak prior towards the decoded
     initial rotation. The neighbouring static poses are fixed boundary conditions, and z is fixed to the depth
     target.
5. **Depth re-anchoring:** in each phase and segment, camera-frame positions are scaled along the viewing rays so
   that the median z equals the baseline's median z over the same frames, then mapped back to world with
   `T_world_cam`.

## 4. Validation (no ground truth)

### 4.1 Held-out frames (carried segments)
Separate run (`--holdout`): odd frames of the carried segments get no image term, and their shift candidate comes
from the neighbours. They are then scored with a plain criterion (all person pixels ignored; table occlusion as
above). This run used the final carried-segment settings (texture weight 1.0, rotation prior 1.0). Its static
phases used an earlier single-start version, which affects the carried segments only through their boundary poses.

| moving segment | held-out frames | IoU base | IoU ours | tex R² base | tex R² ours | chamfer base | chamfer ours |
|---|---|---|---|---|---|---|---|
| M0 80–211 | 65 | 0.700 | 0.794 | 0.467 | 0.475 | 0.0426 | 0.0037 |
| M1 304–404 | 49 | 0.698 | 0.825 | 0.490 | 0.583 | 0.0076 | 0.0033 |
| M2 467–655 | 90 | 0.737 | 0.817 | 0.610 | 0.655 | 0.0590 | 0.0253 |
| M3 670–694 | 12 | 0.905 | 0.901 | 0.610 | 0.620 | 0.0213 | 0.0155 |
| M4 748–900 | 72 | 0.823 | 0.830 | 0.369 | 0.557 | 0.0091 | 0.0080 |
| ALL | 288 | 0.751 | **0.820** | 0.497 | **0.576** | 0.0325 | **0.0119** |

Choosing the weights:
- Texture weight 0.3 gave held-out IoU 0.859 / R² 0.539, and in M0 its texture R² fell below the baseline.
- Texture weight 1.0 beats the baseline on both measures in every segment except a −0.004 IoU in the 12-frame M3.
- Texture is the only evidence that pins rotation for a box-like silhouette, so I chose 1.0.
- Rotation prior 0.1 vs 1.0 gave the same held-out scores, so I kept the more conservative 1.0.

### 4.2 Final result vs baseline, per phase (`validate.py`)
IoU is with person pixels ignored and table occlusion handled as above. dz is ours minus baseline in camera-frame z.

| phase | frames | type | IoU base | IoU ours | tex R² base | tex R² ours | rot speed base (med/p95 °/fr) | rot speed ours | trans. accel. p95 base (cm/fr²) | ours | dz mean (cm) | dz median (cm) | mean rot diff to base (°) | bottom-to-floor base/ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0-79 | static | 0.786 | 0.837 | 0.569 | 0.815 | 0.06/1.48 | 0.01/0.01 | 1.91 | 0.00 | +0.27 | -0.00 | 2.1 | 89.3/88.6 |
| M0 | 80-211 | moving | 0.694 | 0.806 | 0.466 | 0.506 | 1.92/18.73 | 1.65/5.82 | 8.90 | 1.07 | -0.70 | -0.69 | 6.8 | 71.7/65.9 |
| B | 212-303 | static | 0.891 | 0.884 | 0.337 | 0.353 | 0.10/1.32 | 0.00/0.00 | 7.14 | 0.00 | -0.38 | +0.00 | 1.6 | -0.5/-0.4 |
| M1 | 304-404 | moving | 0.703 | 0.839 | 0.487 | 0.561 | 2.53/11.01 | 1.86/4.30 | 9.46 | 0.88 | +0.58 | +0.49 | 7.4 | 62.8/66.4 |
| C | 405-466 | static | 0.785 | 0.858 | 0.707 | 0.825 | 0.70/1.85 | 0.00/0.00 | 1.82 | 0.00 | -0.70 | +0.00 | 4.3 | 88.8/88.0 |
| M2 | 467-655 | moving | 0.737 | 0.807 | 0.620 | 0.651 | 2.51/13.74 | 2.22/4.86 | 8.56 | 0.93 | -0.61 | -0.59 | 6.3 | 81.2/80.0 |
| D1 | 656-669 | static | 0.786 | 0.784 | 0.638 | 0.619 | 0.67/1.06 | 0.00/0.00 | 2.68 | 0.00 | +0.06 | +0.00 | 0.5 | 1.7/1.8 |
| M3 | 670-694 | moving | 0.906 | 0.896 | 0.605 | 0.636 | 0.62/1.58 | 0.98/2.36 | 6.93 | 0.38 | +0.53 | +0.30 | 4.0 | 0.2/-0.1 |
| D2 | 695-747 | static | 0.942 | 0.953 | 0.559 | 0.621 | 0.08/0.77 | 0.00/0.00 | 3.49 | 0.00 | -0.05 | +0.00 | 0.8 | 1.6/1.5 |
| M4 | 748-900 | moving | 0.820 | 0.845 | 0.370 | 0.548 | 2.08/16.78 | 1.33/3.85 | 11.98 | 1.28 | +0.05 | -0.07 | 5.9 | 70.2/71.3 |
| ALL | 0-900 |  | 0.783 | **0.842** | 0.509 | **0.596** | 1.25/11.76 | 1.05/4.27 | 8.73 | 0.99 | -0.21 | -0.14 | 5.0 | 67.6/66.7 |

The static-phase numbers are in-sample: their frames were used in the fit. Frames in a static phase are nearly
identical images, so holding some out would not be an independent test. B and D1 are the pooled baseline pose,
so they differ from the per-frame baseline only by the pooling.

### 4.3 Depth (re-anchoring and independent cues)
Per phase, the median camera depth equals the baseline's by construction: dz median is 0.0 cm in static phases
and within ±0.7 cm in the carried segments; mean dz is within ±0.7 cm everywhere (table above). The cues
(`depth_cues.py`) suggest these shifts along the viewing ray:

| phase | floor contact (lowest mesh point on the released floor plane) | MoGe-2 depth vs rendered depth (median over object pixels) |
|---|---|---|
| A (table) | n/a | +7.5 cm |
| B (floor) | −1.1 cm | +10.8 cm |
| C (table) | n/a | +1.5 cm |
| D1 (floor) | +4.8 cm | +20.2 cm |
| D2 (floor) | +4.0 cm | +19.5 cm |

Other cues:
- **Hand keypoints:** over carried frames, the nearest MHR hand is a median +3.1 cm deeper than the object
  centre (IQR −0.2 to +7.1). This is not independent: the human was optimised jointly with the object.
- **Table height:** the book's underside is at 88.6 cm (A) and 88.0 cm (C) above the floor plane. That is
  consistent, but says nothing about absolute depth.

No two independent cues agree within 2 cm on a correction larger than their disagreement:
- MoGe on this small object is biased towards the background.
- The floor-contact cue is within the floor-plane tolerance (`foot_plane_dist_stats` median 4 cm).

**Applied depth shift: 0 cm in every phase.**

## 5. Confidence per phase
- **A, C (table), D2 (floor): fairly confident it is an improvement.** One constant pose, flat on the support,
  and better IoU and texture R² than every baseline frame. The rotation changes are small (1–4°).
- **B, D1: same as the baseline, pooled (high confidence of no harm).** The refined fits did not pass the
  accept/reject rule.
- **M0, M1, M2, M4 (carried): moderately confident it is an improvement.**
  - Held-out IoU and texture both improve.
  - The large misplacements behind the table edge (~140, ~330, ~580) are fixed; I checked overlays by eye.
  - Rotation jumps are removed (p95 11–19 → 4–6°/frame).
  - Rotations differ from the baseline by 6–7° on average. At a few frames (e.g. ~585–620) texture R² is below
    the baseline even with weight 1.0, so some frames may have rotation slightly worse than the baseline.
- **M4 830–900 (book behind the far table edge): low confidence for both.** Only a thin strip of the cover is
  visible. The vertical position is pinned mainly by the strip's top edge and the hand evidence, and our result
  stays close to the baseline and the hand keypoints there.
- **M3 (slide on the floor): small changes, neutral.**
- **Symmetry: the baseline's flip identity is kept everywhere.** Confident for the spine side; moderately
  confident for front vs back cover. The evidence is the spine logo, judged by eye, and the baseline never flips.

## 6. Reproducibility notes
- Seeds are fixed, the solver uses only `/work/inputs` and `/pipeline/depth/moge2/aligned.h5`, and no network.
- nvdiffrast's backward pass accumulates gradients with GPU atomics, so repeated runs are not bit-identical. The
  differences are typically below 1–2° in the static fits and small in the carried segments.
- Two safeguards limit this: the multi-start static fit and the accept/reject rule against the pooled baseline.
  An earlier single-start version once landed in a poor local minimum in A; the rule caught it.
- Run time: 181 s wall (≈150 s GPU work) at 5.4 GB peak GPU memory. Total GPU time across development including
  that run: roughly 38 min (within the 45 min budget).
