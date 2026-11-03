# Episode 27 (woven_basket): trajectory-level object pose

Output: `object_poses_world.npz` (`T_world_mesh`, 565×4×4, float64). Command:
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(155 s wall, about 2 GPU-min, <2 GB GPU memory). The code is `solve.py` plus `hoilib.py` (rendering, silhouette fitting, colour model, yaw parametrisation). `validate.py` produces the table below; its output is in `validation_output.txt`.

## What the video shows / phases
The task: walk to a woven basket on the floor, pick it up with both hands, carry it, and place it on the white desk. The basket is an oval about 35 × 25 × 7 cm. Its shape is symmetric under a 180° turn about its vertical axis (mesh +y points to the bottom) to within 2 mm. It is small in the image: 55–100 px wide at 2.8–3.7 m.

| phase | frames | how found |
|---|---|---|
| A floor, at rest (touched by the left hand from ~138) | 0–148 | automatic: object-mask IoU with the median mask of frames 0–29 ≥ 0.93 (person-covered pixels ignored) |
| B carried (lift; hidden behind the desk top 186–197; walking in profile 200–300; re-grip and turn to face the camera 300–340; carried towards the desk; placement) | 149–449 | in between |
| C desk, at rest (hands release at ~468) | 450–564 | automatic: mask IoU with the median mask of the last 30 frames ≥ 0.93 |

## Diagnosis of the baseline
* **Static phases.** The rotation is constant, but the position jitters: camera depth wanders by 3.61–3.72 m while the mask doesn't move.
* **Carried phase.** There are 16 frame-to-frame rotation jumps over 15°. The largest is 123° between frames 197 and 198, where the basket reappears from behind the desk top. That is where the tracker lost the basket and re-acquired it rotated 180° about its vertical axis.
* **Evidence for the flip.** I built a per-point colour model of the basket from the video itself (independent of the scan texture) and asked whether pairs of segments agree better as-is or with one of them yawed 180°.
  * Under the baseline, segments 198–240 and 240–290 are consistent with the flip and inconsistent with everything else. For example, 240–290 vs floor scores 0.17 same / 0.62 flipped (luminance correlation); vs 380–445 it is 0.18 / 0.56.
  * All other segments agree with each other as-is: floor vs 330–380 is 0.72 / 0.12, floor vs desk 0.34 / −0.23.
  * In a 6-frame sliding window, the baseline is flipped from 198 to ~276 and wrong in other ways up to ~310.
* **Absolute yaw.** The 180° ambiguity of the static poses needs the scan texture. I re-ran FoundationPose registration (252 hypotheses, RGB-D scorer) on 11 floor frames and 12 desk frames. The best-scoring candidate was in the baseline's yaw class on 11/11 floor frames and 10/12 desk frames. This agrees with the video-only consistency between floor and desk. So both static yaws are kept, and the error is the 198–310 segment. (A physical argument based on the hand grip pointed the other way. The video shows a re-grip at 300–330: the right hand moves from the near rim to the far end. So that argument does not hold, and I did not use it.)

## Method (`solve.py`)
1. **Phases** from mask stability (above).
2. **Static phases: one pose per phase.**
   * Upright: the mesh bottom axis is aligned with the released floor normal. The baseline tilt was 1–4°.
   * Yaw: circular mean of the baseline yaw. A pooled silhouette yaw scan peaks at the baseline yaw on the floor. On the desk it is flat from −109° to −83°, so I keep the baseline.
   * Depth: the baseline's median camera depth over the phase.
   * Image position: median of per-frame silhouette fits with rotation and depth held fixed. SAM2 masks are used, and pixels covered only by the person are ignored.
3. **Carried phase: decoding over time.**
   * **Hypotheses.** For every frame, 36 yaw hypotheses about the gravity axis. Tilt comes from the smoothed baseline, position from the smoothed baseline, then a 2-D silhouette alignment per hypothesis (40 Adam steps, nvdiffrast).
   * **Score:** silhouette IoU, and luminance correlation with a colour model of the object built from the video.
     * The crops are large (2.4× the mask bbox), so a hypothesis that hides behind the person stays in view.
     * Rendered pixels on person-occluded areas count as half a mismatch.
     * The colour model starts from the two static phases. It is rebuilt after a first decode from the static phases plus confidently decoded carried frames, then the decode is repeated. The second pass changed 0 frames by more than 45°.
   * **Viterbi.** Data term `−10·(IoU − best IoU) − colour`. Transition `0.05·(Δψ/10°)² + 0.15·|Δψ|°`. The path is anchored to the floor yaw at the start and the desk yaw at the end. The L1 (total-variation) term removes temporary yaw swings in frames where the basket is half hidden by the hands. I checked frames 226–256 by eye and the basket does not turn there.
   * **Robustness.** All of these give the same path within 20° on every frame: colour weight 0, 1 or 2; IoU weight 10 or 20; quadratic weight 0.05 or 0.2; L1 weight 0.1, 0.15 or 0.25. Using silhouettes only (no colour) and no L1 term gives the same path within 45° on every frame.
   * **Refinement.** Fine yaw search ±12° in 3° steps around the decoded yaw. Frames with no image evidence (hidden behind the desk top) are interpolated. Then a median-5 filter and a Gaussian (σ = 2 frames) on the unwrapped yaw.
   * **Position.** Image position from a final silhouette fit (smoothed baseline if the fit IoU < 0.5). Depth is the smoothed baseline depth. Median-5 plus Gaussian smoothing.
4. **Depth re-anchoring.** Per phase, positions are scaled along the viewing rays so that the median camera depth equals the baseline's. The scale factors were 1.0000, 0.9987 and 1.0000.

## What changed vs the baseline

| segment | rotation difference to baseline (median) | comment |
|---|---|---|
| 0–148 floor | 2.3° | constant upright pose; position jitter removed (median position change 2.5 cm, depth median unchanged) |
| 149–197 lift | 7° | smoothed |
| **198–320 carried, walking** | **160°** | baseline is 180°-flipped (or worse); corrected to the continuous trajectory |
| 321–449 turn, carry, placement | 6° | smoothed; baseline jumps removed |
| 450–564 desk | 0.9° | constant upright pose |

Corrected yaw path (deg, about gravity): floor −174 → lift ≈ −162 → walking −150…−180 → turn at 318–345 → −100 … −108 → desk −112.

## Validation (no ground truth; `validate.py`)
Definitions:
* **Mask IoU:** rendered silhouette vs SAM2 mask, person-occluded pixels ignored.
* **Held-out colour:** each trajectory builds its own colour model from its even frames and is scored on its odd frames (luminance correlation).
* **dz:** camera-frame depth of the result minus the baseline's.

| phase | traj | mask IoU mean / p10 | held-out colour | median dR/frame | jumps >15° | median dT/frame | dz mean / median (cm) |
|---|---|---|---|---|---|---|---|
| floor 0–148 | baseline | 0.785 / 0.765 | 0.937 | 0.10° | 0 | 0.11 cm | 0 / 0 |
| | **result** | **0.857 / 0.847** | **0.960** | 0 | 0 | 0 | +0.5 / +0.0 |
| carried 149–449 | baseline | 0.645 / 0.394 | 0.546 | 3.59° | 16 | 1.54 cm | 0 / 0 |
| | **result** | **0.790 / 0.713** | **0.651** | 0.75° | 0 | 1.00 cm | −0.4 / −0.4 |
| desk 450–564 | baseline | 0.853 / 0.840 | 0.842 | 0 | 0 | 0.04 cm | 0 / 0 |
| | **result** | **0.863 / 0.851** | 0.827 | 0 | 0 | 0 | +0.3 / +0.0 |
| all | baseline | 0.724 / 0.482 | 0.712 | 1.19° | 16 | 0.66 cm | 0 / 0 |
| | **result** | **0.822 / 0.766** | **0.770** | 0.25° | 0 | 0.25 cm | −0.0 / −0.3 |

* **Segment-pair colour consistency (same / flipped).** Under the result, every pair of segments prefers "same", with two near-ties, both involving 198–240: 0.38 / 0.43 against the floor and 0.59 / 0.61 against 290–330. Under the baseline, 198–290 preferred "flipped" against almost every other segment.
* **The desk.** Held-out colour is slightly lower than the baseline's (0.827 vs 0.842). Both are a single static pose with the same yaw, so this reflects a 1–2 px position difference, not a rotation issue.

### Depth along the viewing ray: cues and decision (applied shift: **0 cm**)

| cue | suggested shift (+ = farther) |
|---|---|
| untouched basket resting on the released floor plane (lowest mesh point is 0.8 cm above it; the plane fit has a 3.7 cm median foot offset) | +2.1 cm |
| metric mesh size vs SAM2 mask size (free-depth silhouette fit), floor / desk | −30 cm / −13 cm (masks are 1–2 px fatter than the object, so this is unreliable at 60 px) |
| hand keypoints vs object surface while carried | hands 10 cm in front of the surface at the hand pixel (hands hold the near rim, so not a usable depth cue) |

The cues disagree by far more than 2 cm, so the baseline depth is kept. The basket's bottom is 0.8 cm above the floor plane and the desk-phase bottom is 88.7 cm above the floor plane, which is physically plausible.

## Confidence
* **Floor (0–148): high.** The pose is constant, and image fit and colour improved. The absolute yaw relies on FoundationPose's scorer, 11/11 frames but with small margins. If it were wrong, every phase would be wrong by the same 180°; the video-only consistency cannot detect a global flip.
* **Lift (149–197): high.** Frames 186–197 are hidden behind the desk top and are interpolated.
* **Walking (198–320): medium–high.** This is the main change (122 frames turned by about 160–180°). Four independent arguments support it:
  * physical continuity (no 123° jump in one frame);
  * the colour-consistency matrix;
  * the silhouette-only decode, which gives the same path;
  * by eye, the basket stays end-on to the camera.

  The weak spots are 235–250 (half occluded; yaw ±30°) and 286–310 (low colour evidence).
* **Turn and carry (321–449): high.** Agrees with the baseline up to jitter.
* **Desk (450–564): medium–high.** The yaw is within about ±10°: silhouettes are flat over −109…−83°, and FoundationPose's own optimum lies 3–6° from the baseline.
* **Translation:** the baseline depth is kept on purpose; image positions come from silhouettes.

## Reproducibility notes
* Seeds are fixed (numpy, torch, surface sampling). nvdiffrast's backward pass uses atomic adds, so repeated runs are not bit-identical. Two full runs differed by 0.38° mean and 7.5° max in rotation, and 0.09 cm mean and 1.6 cm max in translation. The largest differences are in the occluded frames 190–197 and in 299–306, 383–386 and 414–416. The phases are identical, the final yaw path agrees to within 7.5° on every frame, and the validation metrics change by at most 0.003.
* No per-frame results or fitted tables are embedded. The only constants are thresholds and weights, explained in the code.

## Run time / GPU
* `solve.py`: 155 s, about 2 GPU-min, peak below 2 GB.
* Whole investigation (FoundationPose re-registration experiments, yaw scans, 5 solve runs, validations): about 37 GPU-min.
