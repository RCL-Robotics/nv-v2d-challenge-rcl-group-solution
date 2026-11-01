# Episode 25 – cyan water bottle: trajectory-level 6-DoF recovery

Output: `object_poses_world.npz` (`T_world_mesh`, 859×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(helpers in `hoi_lib.py`; GT-free checks in `validate.py`).
Final run: 4 min 48 s wall time (≈81 s CPU video/mask loading, ≈3.4 min GPU), peak GPU memory 5.4 GB.
Total GPU use during development, including the final run, was about 30 min.

## 1. Diagnosis of the baseline
* The baseline rotation is the CoCoNet output (within about 1.2°). Its depth is CoCoNet's ×≈1.02.
* **Static phases.** The baseline is offset by a few pixels and tilted about 6–7° away from the table normal. Mask IoU is about 0.81.
  * Between frames 0–40 it drifts.
  * The final rest pose jumps between z = 2.54 m and 2.35 m (frames 780–820) while the bottle does not move.
* **Carried phase.** There are 88 frame-to-frame rotation jumps over 20°, some of 60–145°. Several segments are frozen at a constant pose while the bottle moves (680–710, 780–790). Wrong axis directions appear around 240–250 and 660–710. Mask IoU is 0.73, with p10 = 0.54.
* **Why spin matters.** The bottle is nearly a cylinder (r = 4.6 cm, length 28 cm), so its spin about the long axis matters for ADD: a 180° spin moves surface points by about 9 cm. The only strong spin cue is a blue strap on the upper part of the bottle near the cap. It is visible over about 150° of the 360° spin range. The end-over-end direction is given by the dark cap.

## 2. Phases (automatic, from the median-filtered object-mask centroid; ±2-frame margins)
| phase | frames | rule | checked by eye |
|---|---|---|---|
| static_start (on the round table, untouched) | 0–114 | centroid within 2.5 px of the rest position | the bottle starts moving at 116–117 |
| carried / manipulated (hand-overs, rotations) | 115–776 | everything else | – |
| static_end (put back on the table; the hand touches it until about 816 but does not move it) | 777–858 | centroid within 2.5 px of the final rest | stable from about 776 |

## 3. Method
1. **Per-frame evidence (render-and-compare, nvdiffrast, 192 px crop around the SAM2 mask).** Each pose is scored on five terms:
   * multi-scale silhouette L2 against the object mask, with pixels covered by the person (and not by the object) ignored;
   * colour L1 with a per-channel gain;
   * blueness (B−G) L1 and cap-darkness L1, after a 1.5 px blur;
   * a normalised **strap-overlap** term on soft-thresholded blue maps (≈0 if neither the render nor the image shows the strap, ≈1 if only one does).

   The plain colour loss turned out to prefer "strap hidden" spins. In the static frames the strap is clearly visible on the left (spin 0–15°), while the colour loss picked 300°. The blue and strap terms fix this, and with them every sampled static frame chooses the same spin.
2. **Hypotheses.** For every frame, four starting poses are refined for 40 Adam steps on rotation and translation: the baseline, FoundationPose, and the end-over-end flip of each. Distinct axis directions are kept (merged within 12°), and the toward/away **tilt mirror** of each is added. Each axis is combined with **24 spins** (15° steps) about the long axis and scored by rendering. This gives about 100 states per frame.
3. **Viterbi decoding over all 859 frames.** The cost per transition is 0.004 per degree of rotation change plus 1.0 per metre of translation change, multiplied by 20 inside static phases.
4. **Joint continuous refinement** of all poses at once (60 iterations), combining:
   * the image unary on every frame;
   * second-order smoothness on rotation matrices and translation, with z smoothed more;
   * one shared pose per static phase, so the evidence of 115 or 82 frames is pooled.
5. **Depth re-anchoring to the baseline.** Camera-frame positions are scaled along the viewing ray, so the image position is unchanged.
   * In each static phase, the median z is set to the baseline's median z over the same frames.
   * In the carried phase, a 61-frame sliding window does the same (low-frequency depth from the baseline, high-frequency from my fit).
   * No extra depth shift is applied (see §5).

## 4. Validation without ground truth (`validate.py`, baseline vs result)
The appearance metrics are evaluated on crops around the object mask. The terms the result was optimised on are marked †; they are reported for completeness, not as independent evidence.

| phase | traj | mask IoU | IoU p10 | colour L1 | blue L1† | cap-dark L1† | strap mismatch† | median rot step (deg) | #steps >20° | transl. accel (mm/frame²) | Δz vs base mean / median (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| static_start | baseline | 0.814 | 0.786 | 0.267 | 0.050 | 0.020 | 0.193 | 0.08 | 2 | 4.5 | 0 / 0 |
| static_start | **result** | **0.951** | 0.936 | 0.229 | 0.044 | 0.014 | 0.046 | 0.03 | 0 | 0.04 | +0.08 / 0.00 |
| carried | baseline | 0.727 | 0.535 | 0.323 | 0.059 | 0.029 | 0.528 | 6.7 | 88 | 23.4 | 0 / 0 |
| carried | **result** | **0.901** | 0.865 | 0.267 | 0.052 | 0.018 | 0.308 | 3.9 | 12 | 6.9 | +0.15 / 0.00 |
| static_end | baseline | 0.807 | 0.720 | 0.219 | 0.066 | 0.020 | 0.297 | 0.00 | 4 | 11.9 | 0 / 0 |
| static_end | **result** | **0.945** | 0.930 | 0.142 | 0.043 | 0.011 | 0.053 | 0.06 | 0 | 1.4 | −6.45 / 0.00 |
| all | baseline | 0.747 | 0.579 | 0.305 | 0.058 | 0.027 | 0.461 | 5.1 | 94 | 19.8 | 0 / 0 |
| all | **result** | **0.912** | 0.871 | 0.250 | 0.050 | 0.017 | 0.248 | 3.0 | 12 | 5.5 | −0.49 / 0.00 |

* **static_end Δz.** The mean is −6.45 cm because the baseline itself jumps between 2.47–2.54 m and 2.35 m inside this static phase. My single shared pose sits at the baseline's median depth (2.355 m).
* **Mask size cue.** That depth agrees with the mask-area ratio between the two rest phases: √(2975/2660) = 1.058 × 2.48 m ≈ 2.35 m.

**Physical checks.**
* The bottom of the upright bottle is 0.814 m above the released floor plane at the start and 0.810 m at the end. That is the same table height within 4 mm (baseline: 0.803 / 0.804).
* The bottle's tilt from the floor normal is 4.7° / 2.0° (baseline 6.6° / 3.0°), so it stands more upright on the level table.

**Visual audit.** I inspected over 100 frames, side by side with the image crops, at each iteration.
* Earlier pipeline variants were adopted only where these checks favoured them. In particular, version 2 with the strap term beat version 1 in all 10 inspected frames where the two differed by more than 30°.
* The final run was checked against version 2 on the 28 frames that differ: the final version was better or equal.

## 5. Depth (distance along the viewing ray)
Default kept: the baseline's median depth per phase, and per 61-frame window in the carried phase. **Applied extra shift: 0 cm.**

| cue | suggested shift vs baseline |
|---|---|
| silhouette size, render-and-compare (object only) | −1.6 cm (static_start), −7.9 cm (carried; biased by motion blur and the hand in the mask), −0.3 cm (static_end) |
| hand contact (MHR hand keypoints vs object centre) | inconclusive: object–hand distances of 0.06–0.77 m even at contact probability >0.9, no consistent sign |
| object resting on the table, consistent with the floor plane | 0 cm (the baseline is already consistent: equal table height at start and end) |

No two independent cues agree on a non-zero correction, so none was applied.

## 6. What changed relative to the baseline, and confidence
| phase | change | confidence |
|---|---|---|
| static_start (0–114) | One constant pose. Tilt corrected (more upright). Spin set by the strap, visible on the left, with the same argmin in every sampled frame. Image position corrected by a few px (about 1 cm). | high for rotation; depth = baseline |
| carried (115–776) | Jumps, frozen segments and axis errors removed. Spin follows the strap where it is visible, and continuity where it is hidden. Smooth trajectory. | medium-high for the axis direction (IoU 0.90, cap/dark consistent). Medium for spin: when the strap is hidden for long stretches, spin comes only from temporal continuity. Known residual error: frames ≈636–641 (bottle pointing roughly along the viewing ray, cap end occluded by the hand) still have the cap tilted toward the camera instead of away. |
| static_end (777–858) | One constant pose instead of the baseline's 2.54 → 2.35 m jump. Spin from the strap (consistent argmin). | high for rotation; depth = baseline median (2.355 m), supported by the mask-area and table-height cues |

## 7. Things tried and rejected
* A welded (watertight) mesh for speed: with it the `dr.antialias` silhouette gradients made the optimisation diverge (tested on several frames), so the raw triangle soup is used.
* The plain colour L1 as the spin cue: inconsistent with the visible strap (see §3).
* Forward/backward multi-hypothesis gradient tracking without spin sweeps: it got stuck in wrong spins because spin is a weak, non-convex direction for gradients. It was replaced by the discrete spin sweep plus Viterbi.
