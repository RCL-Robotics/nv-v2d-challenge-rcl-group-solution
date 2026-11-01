# Episode 9 – big red bowl: trajectory-level 6-DoF recovery

Result: `object_poses_world.npz` (`T_world_mesh`, 870×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(123 s wall time, about 75 s of it on the GPU; ≈1 GB GPU memory; bit-identical output across two runs).

## 1. What the video shows and what is (not) observable

* The object is a 30 cm × 13 cm red plastic bowl. The mesh is **rotationally symmetric to within ~3 mm**, and its
  texture is uniform red (the only "features" are baked-in specular highlights). **Spin about the bowl axis (yaw) is
  therefore not observable in the image.** Tilt (the axis direction), image position and depth are observable.
* The baseline tracker has no handle on yaw either. During the carry its yaw wanders randomly: 128 frame-to-frame
  rotation steps are larger than 10°, the largest is 110°, and the baseline yaw relative to the hands changes by
  over 200° across the carry. Its axis (tilt), image position and per-frame depth are noisy (median acceleration
  1.7 cm/frame²).

## 2. Phases

These are found automatically from the SAM2 object-mask bounding box: a frame is static if the median of the four
bbox edges moves ≤ 1 px within ±4 frames. Runs of 10+ frames are merged across gaps of ≤ 12 frames at the same
place, and runs of 25+ frames are kept. I checked the result on contact sheets of the video.

| phase | frames | what happens (verified by eye) |
|---|---|---|
| A static | 0–189 | bowl on the chair seat; hands touch it from ~160, but it does not move until 190 |
| carry | 190–671 | lifted with both hands, carried away from and back toward the camera, back to the camera 500–610 (heavily occluded), put on the floor ~650–670 |
| B static | 672–708 | resting on the floor in front of the chair, hand on the rim |
| slide | 709–761 | pushed along the floor under the chair |
| C static | 762–869 | resting under the chair, untouched from ~786 |

## 3. Method (`solve.py`)

1. **Per-frame render-and-compare** (nvdiffrast silhouettes of the exact mesh, 192 px crops). Each frame is
   initialised from the baseline and fitted for image position, log-depth and two tilt angles; spin is left fixed.
   The fit is coarse-to-fine on blurred masks and uses Adam on **central finite-difference** gradients. Only
   forward rasterisation is used, so the result is bit-reproducible; nvdiffrast's backward pass uses atomics, and
   Adam turned that float noise into mm-level run-to-run differences.
   *Occlusion handling (an important fix):* if the bowl is in front of the torso (from the reconstructed keypoint
   depths), the person's dark clothes are treated as background. Only bright (skin) person pixels are ignored as
   possible occluders. Otherwise the whole person mask is ignored. My first version ignored all person pixels,
   which let the bowl grow over the black shirt and pulled the depth up to 25 cm too close.
2. **Static phases:** one pose per phase, fitted to the pooled masks of all its frames, with depth fixed by the
   anchoring rule in §4. On the floor (B, C) the axis is fixed to the floor normal, since the bowl's flat foot rests
   on the floor.
3. **Moving frames:** robust (Huber) local-linear temporal fusion of several sources:
   * image position from the silhouette fits, weighted by visibility × IoU;
   * a rigid hand-attachment prediction: the projected midpoint of the reconstructed hands plus a smooth offset;
   * the baseline, with low weight.

   Depth comes from the silhouette only where nearly the whole outline is visible (vis ≥ 0.75); elsewhere it is the
   baseline depth times a smooth silhouette/baseline ratio. Static poses enter as hard constraints, so the
   trajectory is continuous at the phase boundaries.
   A **second pass** refits image position and tilt with depth fixed, then smooths again. This is what makes the
   occluded segment work, since partial silhouettes constrain position once scale is fixed.
4. **Floor frames** (B + slide + C): the bowl's foot lies on one plane parallel to the released floor plane.
   Each centre is the viewing ray through the fitted image position intersected with that plane. The plane is not
   snapped to the floor: its offset is set by the depth-anchoring rule (median depth over the floor frames =
   baseline median × (1+s)). It ended up 0.6 cm above the released plane, compared with 5 cm *below* it for the
   baseline.
5. **Yaw (unobservable):** propagated by rigid attachment, anchored to the baseline's yaw in phase A (the
   frame-0 FoundationPose registration, the only yaw evidence there is):
   * constant in A;
   * while carried, it follows the person's yaw (mean of shoulder-line and wrist-line yaw from the reconstructed
     human, smoothed);
   * frozen once the bowl is put down (B, slide, C).

   The resulting yaw change is −81° when the person faces the camera (frame 385) and +43° at the final rest. The
   rotation is built as yaw-about-world-up applied to the phase-A rotation, then minimally tilted onto the
   estimated axis.
6. **Depth anchoring:** the median camera depth of every phase is tied to the baseline median times (1+s), moving
   along the viewing rays only. Moving segments are linearly blended (20 frames) into their static neighbours.
   *s* is a global relative correction, applied only because independent cues agree (§4).

## 4. Depth along the viewing ray: cues and decision

All cues are computed inside `solve.py` (`depth_cues()`). Shifts are relative to the baseline, converted to cm at
the median baseline depth of 3.03 m.

| cue | suggested shift |
|---|---|
| (a) reconstructed human vs. the true frame-0 reference human: the reconstructed human sits 2.6% too far, with the same 2-D outline. The baseline was calibrated against this human. | −7.9 cm |
| (b) metric mesh size vs. image size: silhouette depth in the unoccluded static frames A and C (~270 frames) | −10.0 cm |
| (c) untouched bowl resting on the floor (B, C): scale so the foot touches the released plane | −8.7 cm |
| (d) hand contact at the grasp (frames 170–200): midpoint of the reconstructed hands vs. bowl depth | −8.9 cm |
| **applied** (median of the cues that agree within 2 cm; the rule needs ≥ 2 independent cues agreeing within 2 cm and a correction larger than their spread) | **−2.9% ≈ −8.8 cm** |

Caveats:
* The released floor plane is ~1.8 cm above the reference human's soles. If the true floor is that much lower,
  cue (c) shrinks to about −5 cm, which is still the same sign.
* Cue (a) is indirect: it only shows that the baseline's calibration source is biased.
* Before the correction the baseline bowl penetrated the floor plane by 5 cm in B, the slide and C. After it, the
  bowl sits 0.6 cm above the plane.
* The phase-B depth (−15.5 cm vs. the baseline) comes from requiring B, the slide and C to lie on one floor plane.
  The silhouette depths at B (2.70–2.72 m) agree with it, and the baseline's B depth (2.90 m) was inconsistent
  with C on the same floor.

Mean / median camera-depth differences (cm) between my result and the baseline, per phase: A −8.6/−8.9,
carry −8.0/−6.1, B −15.3/−15.5, slide −15.2/−15.3, C −10.8/−9.7, all −9.2/−8.6.

## 5. Validation (no ground truth)

IoU is mask IoU in 192 px crops over the pixels not hidden by the person, using the same hand-aware occluder model.
Other columns:
* **acc** – median |second difference| of position, cm/frame².
* **angv** – median frame-to-frame rotation, degrees.
* **floor** – 5th percentile of the bowl's lowest point above the floor plane, cm.
* **dz** – camera-depth difference to the baseline, cm.
* **hand sd** – std of the bowl-to-hand-midpoint distance, cm.

| phase | run | IoU | #IoU<0.8 | acc | angv | floor | dz mean | dz med | hand sd |
|---|---|---|---|---|---|---|---|---|---|
| A chair static | baseline | 0.907 | 0 | 0.047 | 0.00 | 45.0 | 0 | 0 | 22.2 |
| | **result** | **0.969** | 0 | 0.000 | 0.00 | 48.9 | −8.6 | −8.9 | 25.5 |
| carry | baseline | 0.813 | 150 | 1.710 | 4.05 | 20.0 | 0 | 0 | 2.9 |
| | **result** | **0.913** | 50 | 0.056 | 0.51 | 24.6 | −8.0 | −6.1 | 7.0 |
| B floor + hand | baseline | 0.778 | 34 | 0.717 | 4.50 | −5.6 | 0 | 0 | 3.1 |
| | **result** | **0.945** | 0 | 0.000 | 0.00 | 0.6 | −15.3 | −15.5 | 1.3 |
| slide | baseline | 0.779 | 42 | 1.341 | 5.26 | −5.7 | 0 | 0 | 2.4 |
| | **result** | **0.937** | 0 | 0.045 | 0.00 | 0.6 | −15.2 | −15.3 | 1.3 |
| C floor static | baseline | 0.837 | 7 | 0.132 | 0.00 | −5.2 | 0 | 0 | 28.2 |
| | **result** | **0.945** | 0 | 0.000 | 0.00 | 0.6 | −10.8 | −9.7 | 29.1 |
| all | baseline | 0.833 | 233 | 1.067 | 2.38 | −5.1 | 0 | 0 | 29.5 |
| | **result** | **0.932** | 50 | 0.033 | 0.14 | 0.6 | −9.2 | −8.6 | 28.3 |

**Held-out frames.** I reran with the mask evidence of every 5th frame (i mod 5 = 2, 174 frames) removed from all
fits and pooling, then scored IoU on those unseen frames:

| | A | carry | B | slide | C | all |
|---|---|---|---|---|---|---|
| baseline | 0.908 | 0.820 | 0.777 | 0.777 | 0.837 | 0.837 |
| result, held-out run | 0.968 | 0.910 | 0.942 | 0.936 | 0.946 | **0.930** |

The temporal model predicts unseen frames almost as well as seen ones.

Other checks:
* Largest frame-to-frame step: 3.7 cm / 2.4° (during the lowering), against 8.7 cm / 110° for the baseline.
* Static phases are exactly constant.
* The worst IoU windows are 490–530 (0.54–0.77, essentially the same as the baseline). There the bowl is almost
  entirely hidden behind the arm and back.
* **hand sd** gets worse in the carry (7.0 vs 2.9). This is expected and not evidence against the result: the
  baseline was optimised for hand contact, and the reconstructed human's depth is itself unreliable. While the
  person walks towards the camera (frames 320–430), the visible bowl width (≈104 px at frame 385) rules out the
  baseline depth (it implies ≤ 2.42 m, while the baseline says 2.54 m). The hands sit even further back.
* I visually checked overlays of the result against the masks every 30 frames.

How the result differs from the baseline per phase:

| phase | mean ADD(result, baseline) | median rotation diff | median axis diff |
|---|---|---|---|
| A | 9.8 cm | 2° | 1.7° |
| carry | 15.3 cm | 70° | 3.2° |
| B | 23.7 cm | 165° | 1.8° |
| slide | 22.8 cm | 128° | 2.4° |
| C | 21.6 cm | 135° | 0.3° |

Most of the change in the carry and later phases is yaw.

## 6. Confidence per phase

* **A (0–189), high** for axis, image position and the shape of the motion. Depth: medium-high (four cues agree on
  −8.8 ± 1 cm). Yaw = baseline registration: unknowable for a symmetric, untextured bowl.
* **Carry (190–671), medium.** Tilt and image position are good wherever the bowl is visible. Depth is good where
  the silhouette is unoccluded (300–440), and interpolated baseline shape elsewhere. 500–600 (bowl behind the
  person) is low confidence. Yaw follows body rotation: physically motivated, but MHR body yaw is noisy (±10–20°)
  and depends on the phase-A anchor.
* **B / slide (672–761), medium.** Upright and on the floor (physics); depth set by the common floor plane. The
  hand occludes the bowl, and yaw during the push is assumed constant.
* **C (762–869), medium-high** for position and axis (unoccluded, upright, constant). Yaw = propagated.
* The dominant remaining risk for ADD is yaw: about 8–9 cm per 45° of yaw error. The baseline's own yaw is equally
  unconstrained.

## 7. What I tried and dropped

* Ignoring all person pixels in the silhouette loss: bowl too large and too close while carried in front of the
  torso.
* Hand-keypoint disks as the occluder model: too coarse.
* Floor-normal axis via free tilt: the free fit leaves a 4° pitch with the plane normal. I kept the physical
  constraint.
* Autodiff (nvdiffrast backward) fits: not reproducible.
* Width-visibility weighting of silhouette depth: too strict, because the hands cover the rim tips in almost every
  carried frame.

## 8. Run time and resources

* Final `solve.py`: 123 s wall time.
* GPU time over the whole investigation: ≈ 25 min.
* Peak torch GPU memory 0.97 GB (process total < 1.5 GB).
* No network access and no scratch files are needed.
