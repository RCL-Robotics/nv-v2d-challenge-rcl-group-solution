# Episode 3: contigo coffee mug, 781 frames. Trajectory re-estimation report

Output: `object_poses_world.npz` (`T_world_mesh`, 781×4×4, float64).
Produced by `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`.
- Runtime: 339 s wall (one RTX 6000 Ada, shared). The GPU is used for roughly half of that.
- GPU memory: peak torch allocation 5.1 GB.
- Run-to-run differences come only from GPU atomics in nvdiffrast. Two full runs agree to 0.09 cm mean ADD (max 0.25 cm).

Code:
- `solve.py`: the pipeline.
- `lib.py`: rendering, fitting, cues.
- `validate.py`: ground-truth-free checks; produces the table below.
- `object_poses_world_log.json`: every per-phase decision and cue value from the final run.

## 1. Diagnosis of the baseline
- **Wrong handle direction for almost the whole episode.** The mug is rotationally symmetric apart from the
  handle, the "contigo" logo, the cream strip at the handle foot, and the lid slider. In every resting phase
  the true handle points image-left and slightly away from the camera. It is grasped there by the hand, the
  logo is visible at the lower-left of the body, and in phase 0 the handle loop is visible.
  - The baseline puts the handle at az ≈ 1–38° (back-right) in all static phases. That is about 80–135°
    off in yaw; az = atan2(z, x) of the mesh +x (handle side) axis in the camera.
  - The silhouette is nearly identical for these hypotheses. The handle is black and usually seen against
    the person's black shirt, so SAM2 also misses it.
- **Jitter and jumps.** Frame-to-frame rotation changes average 10.8°/frame, with jumps of 50–150° during
  the carries (frames ~300, 420, 540, 655). Depth jitters ±3 cm per frame.
- **Depth.** The baseline puts the resting mug 11 cm further from the camera after the first carry
  (z 2.595 m → 2.70–2.74 m). Yet the mug comes back to the same image spot on the same rigid table (mask bottom row
  671–674 px, lower-body column within 8 px). This is not physically possible (see §4).

## 2. Phases (automatic, from the SAM2 object mask)
A frame is *static* when the mask's bottom row stays at the table level and the lower-body column (bottom
25 rows; the handle is higher up) is stable to 1.5 px and 0.8 px over ±3 frames. Runs of 15 frames or more are kept.
I checked the boundaries by eye; the detector is conservative by 2–7 frames at each end.

| phase | frames | content |
|---|---|---|
| S0 | 0–129 | resting on the table; approached; grasped from ~108 |
| C1 | 130–213 | lift with the right hand (image-left), tilt toward the body, set down |
| S1 | 214–268 | resting (released 236–262, re-grasped from 264) |
| C2 | 269–348 | carry (strongest tilt ~60° around frame 300) |
| S2 | 349–398 | resting |
| C3 | 399–479 | carry |
| S3 | 480–526 | resting |
| C4 | 527–604 | carry |
| S4 | 605–640 | resting |
| C5 | 641–688 | short carry |
| S5 | 689–780 | resting, untouched until the end |

## 3. Method
**Static phases: one pose per phase.**
1. **Pooled observation.** A temporal-median crop is built over the phase, with pixels covered by the person
   excluded, along with a median object mask.
2. **Alignment.** The baseline pose is made upright (its axis set to the floor normal) and spun about the fitted mug body
   axis into 18 yaw initialisations. Each is refined by differentiable render-and-compare (nvdiffrast): soft
   silhouette IoU plus blurred-gray NCC, with depth free *for alignment only*. Of the near-upright results, the
   one with the best IoU fixes the tilt and the image position.
3. **Yaw from the logo.** The yaw about the axis comes from the **logo position**. The white logo texels on the
   mesh are found geometrically: lower body, azimuth 75–140° in the mesh frame. The long white streaks elsewhere
   are baked specular highlights.
   - Image side: bright, vertically short detail (gray minus a vertical morphological opening, which
     suppresses specular streaks) inside the observed mask on the logo rows. Its centre is measured as a
     fraction of the body width.
   - Render side: the same fraction is computed from a rendered logo-label texture for yaws in 2° steps.
     The fraction is monotonic in yaw, so the match is unique.
   - Measured fractions are 0.18–0.31, giving az = 134°, 122°, 120°, 118°, 116°, 114° (S0–S5), with residual
     mismatch < 0.008 of the body width (≈ ±2–5°).
   - Independent agreement: the hand grips the left of the mug at every grasp; the handle loop and cream
     strip are visible in S0 (judged by eye at ~130°); the projected handle lands on the projected hand
     (4–6 px, vs 35–39 px for the baseline).
4. **Depth.** A final translation-only alignment is done, then depth is set as in §4.

The generic photometric scores I tried first did not decide the yaw: per-frame IoU, local-normalised NCC,
variance-weighted NCC and edge chamfer each peaked at different angles between 60° and 200°. They were dropped.

**Carried phases.**
1. **Hand-rigid prediction.** The reconstructed grasping hand (21 MHR keypoints) is tracked by a rigid fit
   (Kabsch). Forward from the lift-off pose and backward from the set-down pose, the static poses are
   propagated rigidly with the hand. The two are slerp-blended by time.
   - Check: propagating the start pose through the whole carry predicts the next resting pose to within
     4–21°, while the hand rotates up to 66° mid-carry.
2. **Candidates per frame.** The hand-rigid pose; and the baseline, FoundationPose and CoCoNet rotations, each
   spun about its own mug axis to the yaw nearest the hand-rigid prediction (their axis tilt fits the
   silhouette better), with the baseline centre position.
3. **Viterbi.** Data cost −20·IoU + (angle to the hand-rigid pose)/15°. Transition cost: rotation change / 5°.
   The ends are pinned to the static poses.
4. **Smoothing.** A ±3-frame quaternion and position mean.
5. **Joint temporal refinement.** Silhouette + NCC, second-difference smoothness, depth fixed, end poses fixed.
   On held-out odd frames (fitted on even frames only) it raised IoU by 0.014–0.024 and NCC by
   0.06–0.08 in every carry, so it was adopted.
6. Depth set as in §4.

## 4. Depth along the viewing ray
**Default.** Each phase is scaled along its viewing rays so that its median camera z equals the baseline
median z over the same frames. Image positions are kept.

**Correction rule** (encoded in `solve.py`). A static phase is corrected only if two independent physical cues agree
within 2.5 cm and the correction is larger than their disagreement. The cues:
- **table**: support surface. A RANSAC plane is fitted to the MoGe-2 metric-aligned depth of the table top
  around the mug, kept only if its normal is within 12° of the floor normal; the measured median table
  normal is 1.9° from the floor normal. It is intersected with the viewing ray of the mug's bottom centre.
  Reliable only if ≥10 frames and IQR ≤ 4 cm.
- **hand**: hand contact. The depth shift that minimises the distance between the handle centre and the
  grasping-hand centroid over the contact frames at the phase ends (≥15 frames, residual ≤ 4 cm).
- **rel**: same table spot as an already-corrected phase. The relative depth from the free-depth image fits.
  Absolute object-size depth is biased (SAM masks look dilated: it says 15–17 cm closer than the baseline),
  but the difference between phases is consistent.

Carried frames get the static corrections linearly interpolated between the adjacent phases.

| phase | baseline z (m) | table cue | hand cue | rel cue | applied | final z (m) |
|---|---|---|---|---|---|---|
| S0 | 2.595 | unreliable (IQR 8.8 cm) | 0.0 cm | +4.1 cm | **0** (hand and rel disagree) | 2.595 |
| S1 | 2.716 | −8.3 cm | −8.5 cm | – | **−8.4 cm** | 2.631 |
| S2 | 2.725 | −9.9 cm | −10.0 cm | – | **−9.9 cm** | 2.626 |
| S3 | 2.736 | −8.1 cm | −9.0 cm | – | **−8.6 cm** | 2.650 |
| S4 | 2.731 | −11.3 cm | −10.5 cm | – | **−10.9 cm** | 2.622 |
| S5 | 2.700 | −9.4 cm | −9.5 cm | – | **−9.5 cm** | 2.605 |

Hand cue inside the carries after correction (an independent check; ideally 0):
−1.0, −1.0, +3.0, +0.5, −2.5 cm (C1–C5), residuals 1.4–3.3 cm. Before correction it was −4.5 to −13 cm.

**Caveat.** The table and hand cues both inherit the human's metric scale: the MoGe depth is aligned to the
MHR human depth. They measure different things (table geometry vs. hand position), and the free-depth
object-size fit moves the same way. Even so, this is the riskiest decision in the result: if it is wrong,
S1–S5 and C1–C5 are about 9 cm too close. I applied it because:
1. all three cues point the same way in every phase from S1 on;
2. it removes the physically impossible 11 cm jump of a mug resting at the same image spot;
3. after the correction the final resting depths agree to within ±2.5 cm across all six phases, as same-spot
   physics requires.

## 5. Validation (no ground truth)
Columns:
- **IoU**: silhouette vs SAM2 mask on unoccluded pixels, no shift search.
- **NCC**: blurred-gray appearance correlation over the rendered region.
- **dR, dt**: mean frame-to-frame change, in °/frame and cm/frame.
- **hand2d, hand3d**: handle centre to grasping-hand centroid on contact frames, in px and cm.
- **dz**: camera-frame depth of the body centre minus baseline, in cm.

| phase | frames | traj | IoU | NCC | dR | dt | hand2d | hand3d | dz mean | dz median |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 0–129 | baseline | 0.722 | 0.515 | 3.98 | 1.48 | 35.6 | 13.2 | | |
| | | result | 0.797 | 0.625 | 0.00 | 0.00 | 6.3 | 2.3 | +0.03 | 0.00 |
| C1 | 130–213 | baseline | 0.758 | 0.321 | 8.42 | 1.49 | 19.0 | 10.5 | | |
| | | result | 0.853 | 0.573 | 1.12 | 0.92 | 3.7 | 2.0 | −5.16 | −4.02 |
| S1 | 214–268 | baseline | 0.660 | 0.254 | 5.89 | 1.18 | 37.4 | 15.4 | | |
| | | result | 0.848 | 0.600 | 0.09 | 0.00 | 4.8 | 1.8 | −8.53 | −8.41 |
| C2 | 269–348 | baseline | 0.561 | 0.113 | 19.25 | 2.00 | 24.5 | 12.8 | | |
| | | result | 0.822 | 0.438 | 1.55 | 1.14 | 5.5 | 2.8 | −9.21 | −8.81 |
| S2 | 349–398 | baseline | 0.650 | 0.261 | 6.60 | 0.89 | 38.7 | 17.1 | | |
| | | result | 0.866 | 0.591 | 0.00 | 0.00 | 4.0 | 1.5 | −10.40 | −10.04 |
| C3 | 399–479 | baseline | 0.633 | 0.152 | 16.25 | 2.02 | 34.1 | 13.8 | | |
| | | result | 0.837 | 0.484 | 1.74 | 1.16 | 4.7 | 4.1 | −10.04 | −10.05 |
| S3 | 480–526 | baseline | 0.644 | 0.178 | 5.60 | 1.07 | 36.3 | 15.2 | | |
| | | result | 0.841 | 0.580 | 0.00 | 0.00 | 3.6 | 1.6 | −8.79 | −8.60 |
| C4 | 527–604 | baseline | 0.656 | 0.150 | 18.79 | 1.89 | 33.0 | 15.2 | | |
| | | result | 0.832 | 0.445 | 1.69 | 1.23 | 3.7 | 2.3 | −10.01 | −9.06 |
| S4 | 605–640 | baseline | 0.620 | 0.160 | 6.43 | 1.13 | 38.0 | 17.1 | | |
| | | result | 0.848 | 0.590 | 0.09 | 0.00 | 4.4 | 2.7 | −10.90 | −10.91 |
| C5 | 641–688 | baseline | 0.516 | 0.037 | 25.96 | 2.06 | 34.6 | 17.4 | | |
| | | result | 0.825 | 0.525 | 1.55 | 0.95 | 3.5 | 3.0 | −10.59 | −10.06 |
| S5 | 689–780 | baseline | 0.694 | 0.322 | 5.45 | 1.03 | 36.0 | 16.8 | | |
| | | result | 0.802 | 0.538 | 0.00 | 0.00 | 3.1 | 3.9 | −10.02 | −9.46 |
| **all** | 0–780 | baseline | 0.659 | 0.255 | 10.76 | 1.51 | 30.5 | 14.1 | | |
| | | **result** | **0.828** | **0.543** | 0.75 | 0.54 | 4.4 | 2.7 | −7.66 | −8.67 |

The dR of 0.09 in S1 and S4 comes from the boundary frame between two static runs. Intermediate version v1
(new rotations, baseline depth per phase) scored IoU 0.775, NCC 0.435 and hand3d 8.5 cm overall. The depth
correction raised IoU and NCC further and brought the 3-D hand-handle distance from 8.5 to 2.7 cm.

## 6. Confidence
- **Static rotations (S0–S5, 451 frames): high.**
  - Yaw ±5–10°, from the logo measurement, consistent with the hand side and the visible handle loop/strip.
  - Tilt within 0.6–3.5° of the floor/table normal; the lateral component is ≤ 1.4° and observable.
  - A 180° flip vs. the baseline is excluded by three independent cues.
- **Carried rotations (C1–C5, 330 frames): medium.** The handle is in the hand in every checked frame and the
  tilt matches the silhouette (IoU 0.82–0.85). Yaw relies on the hand-rigid model, probably within ~15–20°.
  The strongly tilted frames (around 300, 440, 560) are the least certain.
- **Positions:**
  - Image-plane position: high.
  - Depth: S0 high (cues agree with the baseline). S1–S5 and C1–C5: medium-high. Two cues agree to ≤ 1.5 cm
    and the result is physically consistent, but both cues share the human's metric scale (see caveat in §4).

## 7. What was not done / limitations
- No new network inference (FoundationPose re-registration, point trackers) was run. The texture cues plus the
  hand model were enough to resolve the main failure, and the GPU budget was kept small.
- SAM2 masks were not recomputed.
- The static phases assume no in-place rotation while the mug stands on the table and is held (e.g.
  108–129). The mask's lower-body column is stable to < 1 px there, but a small yaw while the hand holds it
  cannot be ruled out.
- The baseline-depth version (v1) was intermediate only. To revert the depth decision, set `corr_static`
  to zeros in `solve.py` §4. This gives the "rotation-only" variant, whose overall scores are listed in §5.
- "SAM masks look dilated" is an inference, not a measurement. Edge-based width measurement was tried and
  was too noisy (±2 px on 26 px) to confirm it.
