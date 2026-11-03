# Episode 16 – spray bottle: trajectory-level re-estimation of the object pose

Result: `object_poses_world.npz` (`T_world_mesh`, 477×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(run time 76 s wall, of which ≈ 60 s GPU, peak GPU memory < 3 GB; log in `solve_log.txt`).
Helper modules: `hoi_lib.py` (rendering with nvdiffrast, scoring), `validate.py` (validation table),
`depth_cues.py` (depth-cue report). No ground truth was used or sought.

## 1. What the video shows and the phases found

Task: "pick up the spray bottle with your left hand and place it on the ground". The bottle (29.7 cm tall,
square 8.7 cm body, asymmetric trigger head) is 2–3.2 m from the static camera, 80–130 px tall.

The phases are detected automatically from the SAM2 object mask. The criterion is that the bottom-centre
of the mask stays within 2 px of the median rest position of the first/last 60 frames.

| phase | frames | description |
|---|---|---|
| S1 static | 0–122 | standing on the white desk (hand reaches it from ~108, object still until 122) |
| carried | 123–375 | lifted by the left hand, carried, the person turns round; **322–340: object completely hidden by the body (empty mask)**; 341–375: lowered and released next to the table leg |
| S2 static | 376–476 | standing on the floor behind the table foot bar |

## 2. Diagnosis of the baseline (evidence collected, no GT)

* **S1:** the rotation is right. A texture-based yaw sweep peaks at the baseline yaw in every frame, with a
  correlation of 0.45–0.6 against ≤ 0.1 for the 180° flip; the logo faces the camera. But the baseline
  **drifts by 19 cm in camera depth (3.04 → 2.85 m)** and jitters 1.6°/frame while the object is at rest.
* **Mask principal axis vs projected bottle axis:** the in-plane orientation of the baseline is within ±3° of
  the mask almost everywhere. Its errors are therefore yaw (about the bottle axis) and out-of-plane tilt.
* **Carried 160–270:** the bottle slowly turns in the hand, from the text side to the logo side. The baseline
  lags by 15–45° in yaw (texture sweep and visual check: the blue logo appears on the right half of the
  label 30–45° earlier than in the baseline). Then it jumps 40° at frame 264.
* **Occlusion 322–340:** the baseline poses are garbage (upside-down poses, 60–175° frame-to-frame jumps).
* **Placement 341–368 vs S2:** the baseline shows the *logo* side during placement. Then, at frames
  369–371, exactly when the hand releases the bottle, it rotates by ~130–150° to the S2 orientation, which
  shows the *text* side. A released bottle cannot spin like that, so one of the two is a tracker flip.
  * Floor (S2) orientation: checked on the 100-frame temporal-median image. The trigger head is seen
    side-on, about 7.5 cm wide; an end-on head (yaw 90/270°) would be about 4.5 cm. The lever is on the
    left. Head-silhouette IoU on the median image is 0.72 at the baseline yaw, against 0.62 at +180° and
    0.66 at +120°. The dark-blue logo is not visible.
  * Placement frames: likewise, no dark logo patch is visible.
  * So the S2 baseline yaw is kept, and the placement segment is the flipped part. The person turns
    their back to the camera during the occlusion, which is where the in-hand orientation change
    happens.
* **S2 depth:** the baseline places the bottle **4.6 cm below the floor plane**. The per-frame jitter is
  2.3°/frame.

## 3. Method (solve.py)

1. **Static phases → one pose each.**
   * Rotation: the chordal mean of the baseline rotations over the phase, minimally rotated so that the
     bottle axis is parallel to the floor normal (a bottle standing on its flat bottom). This changes the
     axis by 0.5° (S1) and 2.9° (S2).
   * S1 yaw: a deterministic 2°-step grid search on the phase-median image. The score is texture
     correlation plus person-aware silhouette IoU. Best yaw is +0° (score 0.97, against ≤ 0.62 for any yaw
     more than 45° away).
   * S2 yaw: kept (see §2; texture there is not discriminative).
   * Translation: the median baseline camera-frame position.
2. **Carried phase → hypotheses + Viterbi.**
   * Hypotheses: for every frame with ≥ 400 visible mask pixels and a best hypothesis IoU ≥ 0.45
     (228 of 253 frames), 180 hypotheses = 36 yaws (10° steps about the bottle axis) × 5 out-of-plane
     tilts (0, ±5° about camera x/y), all built around the baseline rotation.
   * Data term: correlation between the textured nvdiffrast render (blurred to the video sharpness) and the
     frame inside the visible mask, plus 0.5 × person-aware silhouette IoU.
   * Transition cost: 5e-4·angle²/frame-gap, a random-walk prior. Large rotations are cheap only across
     long gaps such as the occlusion, and never in a single visible frame.
   * Boundary costs tie the chain to the S1 and S2 poses.
   * Frames without evidence (the occlusion) are bridged by SLERP between decoded neighbours. Rotations
     are then Gaussian-smoothed (σ = 1.5 frames); static frames stay fixed.
   * Translations: the baseline's, median-filtered (5) and Gaussian-smoothed (σ = 2). The baseline
     positions during the occlusion stay within 7–17 cm of the left wrist, i.e. plausible.
3. **Depth anchoring.** Per phase, camera-frame positions are scaled along the viewing rays so that the
   median camera depth equals the baseline median of the phase. Then the independent physical cues are
   evaluated automatically for the static phases (table below). A correction is applied only if two cues
   agree within 2 cm and the correction exceeds their disagreement; it is the reliability-weighted mean
   of all cues (σ: floor 3 cm, hand 4 cm, size 5 cm). Finally, the carried translation is blended linearly
   into the static positions over the first 20 and last 30 carried frames, so lift-off and release are
   continuous: the object leaves and reaches its rest position.

### Depth cues (camera-depth shift suggested by each cue, cm; + = farther)

| phase | metric size vs image (median-image IoU) | hand contact (finger tips vs trigger head) | floor contact | applied |
|---|---|---|---|---|
| S1 table | −9 | +9.4 (lift-off, frames 123–136) | n/a (no table plane) | **0** (cues disagree) |
| carried | – | −1.0 (222 frames, spread 5.4) | – | **0** |
| S2 floor | −16 | −14.2 (release, frames 352–366) | −11.2 | **−13.0** |

For S2, size and hand agree within 1.8 cm, and floor and hand within 3.0 cm; all three cues point the same
way. After the correction, the lowest point of the bottle is 0.7 cm above the released floor plane, instead of
4.6 cm below it. The silhouette IoU on S2 also rises from 0.718 to 0.746. The cues re-evaluated on the final
result are in `depth_cues.txt` (floor +1.8, size −5, hand at release −6.8 cm with 7.1 spread).

## 4. Validation (no ground truth): baseline vs result (`validation_table.txt`)

* **IoU** is person-aware: rendered pixels on the person are ignored.
* **tex** is the correlation of the blurred textured render with the image inside the visible mask.
* **rot/trans step** is the frame-to-frame change.
* **dz** is the camera-frame depth difference between result and baseline.

| phase | IoU base → ours | tex base → ours | rot step mean / p95 / max (deg) base → ours | trans step p95 (cm) | dz vs base mean / median (cm) | lowest point above floor |
|---|---|---|---|---|---|---|
| S1 table (123 fr) | 0.802 → **0.826** | 0.503 → **0.548** | 1.6 / 3.3 / 6.0 → 0 / 0 / 0 | 3.0 → 0 | −2.2 / 0.0 | – |
| carried (253 fr) | 0.782 → **0.797** | 0.311 → **0.390** | 12.4 / 57 / 176 → 2.3 / 5.3 / 9.2 | 7.0 → 3.3 | −1.3 / −1.1 | – |
| S2 floor (101 fr) | 0.724 → **0.746** | 0.184 → **0.243** | 2.3 / 5.6 / 9.8 → 0.05 / 0 / 4.8 | 2.3 → 0 | −13.2 / −13.0 | −4.6 → **+0.7 cm** |

**Held-out test of the Viterbi decoding.** I decoded with the evidence of all odd frames hidden, then scored
those odd frames. Settings were chosen on this test (λ ∈ {1e-4, 5e-4, 2e-3, 8e-3} and tilt ∈ {0, 5, 8}°,
yaw step ∈ {15, 10}°). Held-out tex / IoU for the carried phase:

| | tex | IoU |
|---|---|---|
| baseline | 0.309 | 0.772 |
| ours (λ = 5e-4, 15° yaw, no tilt) | 0.363 | 0.792 |
| ours final (10° yaw, ±5° tilt) | 0.389 | 0.794 |

For the final setting, by segment: 123–250 0.528 vs 0.415, 250–320 0.249 vs 0.189, 320–376 0.166 vs 0.168
(no gain in the dim placement segment).

**Differences from the baseline per sub-phase** (rotation angle; mean displacement of mesh points =
"ADD to baseline"):

| segment | rot diff mean / median | dz mean / median (cm) | mean point displacement vs baseline |
|---|---|---|---|
| S1 0–122 | 3° / 2° | −2.2 / 0.0 | 4.6 cm (removal of the 19 cm depth drift) |
| carried visible 123–321 | 17° / 11° | −0.9 / −0.9 | 1.9 cm |
| occluded 322–340 | 122° / 164° | −1.9 / −2.0 | 12.7 cm |
| placement 341–375 | 110° / 133° | −3.6 / −6.6 | 13.9 cm |
| S2 376–476 | 4° / 4° | −13.2 / −13.0 | 14.1 cm |

## 5. Confidence per phase

* **S1 (table):** high for rotation (texture decisive, upright). Translation: constant pose at the baseline
  median depth; the baseline drift is removed. The absolute depth is the baseline's (the two cues disagree:
  size −9, hand +9 cm).
* **Carried 123–321:** medium-high. Yaw corrections of 15–45° in 160–270 are supported by texture (held-out
  gain) and a visual check. The out-of-plane tilt corrections (≤ 5°) are smaller and less certain.
* **Occlusion 322–340:** low. No image evidence; this is a SLERP between the visible ends, about a 150°
  in-hand/body turn. The exact path is unknown, but it is no worse than the baseline's random poses.
* **Placement 341–375:** medium. The 180° flip relative to the baseline rests on physical consistency with
  the rest pose, and on the S2 head-silhouette/logo evidence; per-frame texture there is ambiguous (0 vs 180).
* **S2 (floor):** rotation medium-high (head silhouette side-on with lever left, no logo visible; the
  alternative yaws are worse). Depth: the −13 cm correction rests on agreeing physical cues (floor contact,
  hand at release, metric size). Medium confidence; if those cues are wrong, this phase loses a few cm.

## 6. Things tried and not adopted

* Gradient-based render-and-compare refinement per frame (silhouette + colour). It wandered into wrong
  local minima on small, blurred crops, and is not deterministic (nvdiffrast atomics), so it was replaced by
  discrete hypotheses + Viterbi.
* A linear-in-angle transition cost: it put the 150° in-hand turn at the release frame, not during the
  occlusion; replaced by the quadratic/gap cost.
* Colour-texture yaw for S2: not discriminative there (dim, low contrast; the luminance correlation curve is
  flat); the trigger-head silhouette was used to verify the baseline yaw instead.
* Transferring the left-hand rotation (MHR keypoints) across the occlusion. Even in visible frames the
  hand-relative object rotation varied by 30–60°, and its prediction for S2 (yaw +70–120°) contradicted the
  head silhouette, so it was not used.
* The S1 yaw re-estimate by gradient refinement suggested +6°. The deterministic grid gives +0°; the grid is
  used.

## 7. Decisions from looking at images (encoded as constants/criteria)

* The episode starts and ends at rest; the phase boundaries are found by the 2 px mask-stationarity rule.
* The S2 yaw is kept from the baseline (verified via the trigger-head silhouette, see §2); `solve.py` does
  not re-estimate it (`refine_yaw=False`, commented).
* Left hand = manipulating hand (task text): `LTIPS` keypoints for the hand cue.
