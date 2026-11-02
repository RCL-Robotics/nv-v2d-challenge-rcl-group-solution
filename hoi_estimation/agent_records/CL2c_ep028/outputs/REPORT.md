# Squeegee episode 28: trajectory-level object pose recovery

Result: `object_poses_world.npz`, key `T_world_mesh`, shape (988, 4, 4), float64, produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(wall time 5 min 08 s, about 3 min of it on the GPU, peak GPU memory about 2.6 GB). Two runs gave bit-identical output: a development run with a cached stage 1 and the final clean run.

## 1. What the video shows (phases)
The object is a 1.54 m squeegee: a long handle (mesh +z) with an orange cap at the end and a 44 cm head (mesh z ≈ -0.5) that carries an orange bracket.
The bracket's two faces differ. The +y face shows a large oval hole next to the handle socket. The -y face shows a smaller bracket, and the grey rubber face of the blade is visible from that side.

| frames | phase | how found |
|---|---|---|
| 0-117 | **static**: head resting on the bar-stool seat, handle end on the floor | automatic: object mask IoU with frame 0 > 0.75 |
| 118-~150 | pick-up: lifted with both hands and turned head-down | contact, mask motion |
| ~150-~700 | carrying and sweeping, head on the floor | |
| ~700-~790 | re-orientation behind the body: the squeegee is swung so the head goes up. Heavy occlusion; SAM2 misses the raised head | by eye; handled automatically |
| ~790-907 | head up and placed on the stool seat, blade turned horizontal at ~850; hand holds the handle | by eye; handled automatically |
| 908-987 | **static**: head on the stool seat, handle end on the floor | automatic: mask IoU with the last frame > 0.75 |

Only the static-phase boundaries are used as phases inside `solve.py`, and they are detected automatically. The other boundaries are for reporting only.

## 2. Diagnosis of the baseline
I rendered the textured mesh over the frames and looked at close-ups of the head.
1. **Frames ~750-987: end-to-end flip.** The baseline (and FoundationPose / CoCoNet) put the head at the bottom. In the video the head is up and then rests on the stool seat. This is about 240 frames with errors of order 1 m.
2. **Frames 0-117: roll flipped 180° about the handle axis.** The image shows the bracket's large oval hole and the blade seen edge-on (thin dark line). The baseline shows the opposite face, with the hole hidden and a wide grey blade face. The silhouettes are almost identical.
3. **Frames ~690-790: lost.** Orientation jumps by up to 120°/frame. Mask IoU is 0.11 and the hand-to-handle distance is 21 cm.
4. Everywhere, image alignment is a few pixels off, which matters for a 1-2 cm thick handle (mask IoU 0.25-0.33). Rotation jitter is 1.3-1.9°/frame even in the static phase.

## 3. Method (`solve.py`)
1. **Evidence per frame.** From the SAM2 object and person masks:
   - truncated distance maps of the object mask and of "squeegee-orange" pixels (HSV hue 3-11, sat ≥ 145; this excludes skin, hue 13-16). Orange is searched within 250 px of the mask because SAM2 often loses the raised head; the orange blobs are added to the object mask.
   - a person-occlusion map, and samples of object and orange pixels.
2. **Hypotheses.** For every frame there are 24 hypotheses: the baseline pose × {same end, end-to-end flip} × 8 roll angles (45° steps) about the handle axis, plus depth-tilt reflections. Each is refined by batched point-based render-and-compare (80 Adam steps, annealed robust kernel). The terms are:
   - two-way silhouette chamfer, with model points behind the person ignored;
   - two-way orange-part chamfer with back-face culling;
   - floor penetration (> 3 cm below the released plane);
   - hand-handle proximity weighted by `contact_prob`;
   - a weak prior to the baseline depth.
3. **Appearance score.** nvdiffrast renders the z-buffered *visible orange coverage* of the textured mesh in a window around the bracket. Its NCC with the image's orange score distinguishes the bracket faces (hole and shape), which the silhouette cannot. I checked it by eye on about 20 frames (static phase, sweeping, 515, 600, the end phase) and it agreed with my visual reading every time. The full-RGB NCC did not agree: it is biased by the texture's bright grey blade face, so it was dropped.
4. **Static phases.** One pose per phase is fitted jointly to 24 frames spread over the phase, from the 6 best per-frame hypothesis classes. It is chosen by pooled geometric cost × 20 + (1 − NCC) × 8.
5. **Decoding.**
   - First Viterbi pass: unary = 20 × geometric + 8 × (1 − NCC); transition = Δθ/8° + Δc/6 cm. The static frames are forced to the static pose.
   - Propagation: every moving frame is re-registered from the decoded poses at f±1 and f±3, then a second Viterbi pass is run.
   - The Viterbi path is stable for appearance weights 8-16. With weight 4 it changes only in the 790-850 roll (see confidence).
6. **Smoothing and refinement.** Gaussian SE(3) smoothing (σ = 2 frames), then per-frame refinement with a prior to the smoothed path, then light smoothing (σ = 1). This is done in the moving phase only.
7. **Depth re-anchoring** (section 5).

## 4. Validation (no ground truth): baseline vs. result per phase
`validate.py` produced this table on the final output. Caveats:
- "geo cost", "orange NCC" and "hand dist" are terms of my own objective, so they show consistency, not independent proof.
- "mask IoU" (rendered silhouette vs. the original SAM2 mask, person pixels ignored) is not optimised directly. However, it is sensitive to depth: anchoring to the baseline depth lowers it.

| phase | frames | traj | mask IoU | orange NCC | geo cost | hand dist cm (med) | min height above floor cm (med) | handle-tip height cm (med) | ang. vel deg/f (med / p95) | rot. to baseline deg (med) | dz vs baseline cm (mean / med) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-117 | 117 | baseline | 0.311 | 0.307 | 0.503 | 5.0 | -4.4 | -3.4 | 1.34 / 4.5 | 0.0 | 0.0 / 0.0 |
| 0-117 | 117 | result | 0.667 | 0.710 | 0.260 | 7.5 | 3.4 | 4.5 | 0.00 / 0.0 | 167.2 | 1.5 / -0.0 |
| 117-180 | 63 | baseline | 0.254 | 0.347 | 0.854 | 5.1 | 49.0 | 52.0 | 3.70 / 14.6 | 0.0 | 0.0 / 0.0 |
| 117-180 | 63 | result | 0.639 | 0.775 | 0.296 | 4.4 | 45.6 | 61.2 | 2.02 / 3.2 | 59.3 | 0.2 / -0.1 |
| 180-690 | 510 | baseline | 0.332 | 0.269 | 0.837 | 4.0 | 8.7 | 116.1 | 1.67 / 5.5 | 0.0 | 0.0 / 0.0 |
| 180-690 | 510 | result | 0.637 | 0.763 | 0.286 | 3.5 | 6.4 | 120.9 | 0.74 / 2.0 | 5.8 | -2.1 / -3.6 |
| 690-790 | 100 | baseline | 0.112 | 0.018 | 2.290 | 21.1 | 35.2 | 62.3 | 5.66 / 54.4 | 0.0 | 0.0 / 0.0 |
| 690-790 | 100 | result | 0.519 | 0.584 | 0.844 | 2.8 | 29.4 | 72.2 | 2.35 / 26.1 | 84.1 | -26.4 / -34.0 |
| 790-905 | 115 | baseline | 0.301 | -0.025 | 1.586 | 28.4 | -2.1 | 56.3 | 2.13 / 7.2 | 0.0 | 0.0 / 0.0 |
| 790-905 | 115 | result | 0.593 | 0.753 | 0.297 | 4.7 | 6.5 | 7.9 | 0.63 / 7.8 | 174.2 | -25.5 / -26.6 |
| 905-988 | 83 | baseline | 0.201 | -0.013 | 1.424 | n/a | 4.0 | 95.7 | 1.88 / 8.0 | 0.1 | 0.0 / 0.0 |
| 905-988 | 83 | result | 0.703 | 0.829 | 0.119 | n/a | 1.5 | 3.0 | 0.03 / 0.0 | 176.3 | -1.9 / -0.6 |

Notes:
- **Depth.** "dz" is the camera-frame depth of the object centre (bounding-box centre), result minus baseline.
  - In 0-180 and 180-690 it is ≈ 0, as intended.
  - In 690-905 the result is 25-34 cm closer than the baseline. There the baseline orientation is wrong (flipped or lost), so its depth is not a reference; see section 5.
- **Physical plausibility.**
  - In both static phases the handle tip is 3-4.5 cm above the released floor plane. Feet are 5.2 ± 1.5 cm above it (`foot_plane_dist_stats`), so the tip is on the floor. In the baseline's flipped end phase the tip floats 96 cm high, and in 0-117 it is 3.4 cm below the plane.
  - In the end phase the head's lowest point is 75 cm above the floor, which fits a bar-stool seat.
  - While sweeping, the lowest point of the head is 5.7 cm above the plane (foot level).
- **Smoothness.** Median angular velocity drops from 1.3-2.1 to 0.6-0.7°/frame, and the static phases are exactly constant.
- **Remaining jumps** (> 20°/frame after smoothing) are in frames 733-737, 745-746, 753-755 and 787-790, with translation jumps of 10-15 cm/frame at 734-737. These all fall inside the fast, occluded re-orientation.

Difference between result and baseline (mean point distance over mesh points, i.e. ADD between the two):

| frames | 0-117 | 118-150 | 151-703 | 704-789 | 790-907 | 908-987 | all |
|---|---|---|---|---|---|---|---|
| mean / median (cm) | 15.6 / 15.0 | 28.2 / 28.2 | 9.8 / 8.3 | 72.9 / 78.2 | 100.4 / 100.4 | 102.4 / 102.9 | 34.9 / 13.8 |
| frames > 90° from baseline | 118 | 18 | 0 | 48 | 118 | 80 | 382 |

## 5. Depth (distance from the camera)
The default rule is applied. Camera-frame object-centre positions are scaled along their viewing rays so that the median camera depth equals the baseline's median over the same frames. The anchor units are chosen automatically:
- the two static phases;
- runs of at least 30 frames of the moving phase where my orientation agrees with the baseline's (< 45°). This gives one run, 151-703.

Scales applied: 0-117: ×1.0226; 151-703: ×0.9785; 908-987: ×0.9954.

Where the baseline orientation is wrong (118-150 and 704-907), its per-frame depth correction refers to a different configuration (e.g. hand contact on the wrong end), so it is not used. Instead the scale is interpolated in time between the neighbouring anchor units and lightly smoothed, which keeps the trajectory continuous. Checks of this choice for 790-907:
- the handle tip stays on the floor (7.9 cm, close to foot level);
- the hand-handle distance is 4.7 cm (baseline 28 cm);
- the trajectory joins the end static pose, which is itself anchored to the baseline and agrees with my independent fit within 0.5 %.

Physical cues for a further depth shift (positive = farther from the camera than applied):

| unit | silhouette/size (my un-anchored fit) | hand contact | floor | applied |
|---|---|---|---|---|
| 0-117 static | -7.1 cm | -10.1 cm (only 15 grasp frames at the end of the phase) | -2.4 cm (handle tip → foot-level height 5.2 cm) | 0 |
| 151-703 sweeping | +5.8 cm | +4.0 cm (553 frames; median distance 3.6 → 2.1 cm) | +1.7 cm (lowest head point → foot level; ±7 cm for ±1.5 cm plane uncertainty) | 0 |
| 908-987 static | +1.2 cm | n/a | -6.0 cm (tip) | 0 |

Why no shift was applied:
- **0-117 and 908-987:** the cues disagree by more than 2 cm.
- **151-703:** the size and hand cues agree (+4 to +6 cm). They are not independent, though: the hand term is part of the objective that produced the un-anchored fit. The floor cue (+1.7 cm, large uncertainty) does not confirm them.

## 6. Confidence per phase
- **0-117 static: orientation high, roll medium-high.** End-to-end: certain. The 180° roll relative to the baseline is clearly supported by close-ups: the hole is visible and the blade is edge-on. The automatic margin is small, though (pooled score 5.25 vs. 5.39 for the baseline roll). The silhouette slightly prefers the baseline roll and the bracket appearance clearly prefers the flip.
- **118-150 pick-up: medium.** The roll changes during the lift and turn, and the path between the two well-determined phases is decoded but not verified frame by frame.
- **151-703 sweeping: high.** Same orientation as the baseline (median 6° difference), much better image alignment, less jitter, depth anchored. Two isolated segments (~520-540, ~630) have heavy occlusion by the leg.
- **704-789 re-orientation: low-medium.** The end-flip is right from ~745 on (head visible up at the shoulder), but there are fast swings, occlusion and missing masks. Expect remaining errors of tens of degrees on some frames.
- **790-907 placing: end orientation high, roll medium-low in 790-850, depth medium.**
  - In 790-850 the head is seen edge-on and there is a remaining 180° roll ambiguity: the decision changes if the appearance weight is lowered to 4.
  - From ~850 the blade is horizontal and the bracket is visible, so the roll is well determined.
- **908-987 static: high.** Head on the seat, blade horizontal along the seat edge, bracket face matched (NCC 0.83, geometric cost 0.12 vs. 1.42 for the baseline), tip on the floor.

## 7. Development notes
- Development GPU time was about 28 min, including the final run and the validation.
- Things tried and dropped:
  - FoundationPose camera-frame poses as extra hypotheses: they added no new solutions.
  - A first orange threshold that included skin (hue up to 16): it made the static-phase roll decision flip-flop.
  - Full-RGB NCC: biased, see section 3.
- The model was not re-run (no SAM2, FoundationPose or MoGe), and no external tools were downloaded (see `TOOLS.md`).
- Determinism: seeds are fixed. A cached development run and the final clean run gave bit-identical output.
