# Episode 23 – toy stroller: trajectory recovery report

**Result:** `object_poses_world.npz` (`T_world_mesh`, 946×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(71 s wall time on the final run, of which roughly 50 s is GPU; peak torch GPU memory 0.92 GiB).
`validate.py` reproduces the validation tables below. It needs a `--diag` file written by `solve.py --diag`.

## 1. Diagnosis of the baseline

* The task (`task.txt`): the person walks to a toy stroller standing on the floor, then pushes it away, turns it and
  pushes it back and forth. The stroller is ~53 cm tall and always rolls on the floor (checked by eye on contact
  sheets of the video). It is never lifted or carried.
* Baseline rotations equal CoCoNet's (a constant 1.1° offset comes from the change of camera frame). No 180° flip
  was found: overlays of the baseline and of its yaw+180° hypothesis at frames 300–390 show the seat-back panel and
  the hand on the handle, consistent with the baseline.
* Main defects found:
  1. **Jitter.** The translation changes 1–3 cm per frame and the rotation ~1° per frame, even while the stroller
     is untouched (frames 0–134).
  2. **Inconsistent height above the floor.** The lowest point of the stroller wanders between −7 cm and +6 cm
     relative to the released floor plane, and the stroller tilts by up to 12°. Given the camera geometry, this is
     a per-frame depth error along the viewing ray of up to ~20 cm. It is worst during the occluded turn,
     frames ~340–470: the baseline depth jumps by 17–24 cm there.
  3. **Lost hand contact during the occluded turn.** The baseline object is 4–5 cm away from the reconstructed
     hands in frames 340–460, the only phase where the human was not attached to it.
  4. **Moving vs resting height.** The baseline places the object ~3 cm higher above the floor while it is pushed
     than while it rests. That is physically impossible for a rolling stroller.

## 2. Phases (automatic, from the registered trajectory)

| phase | frames | how detected | model |
|---|---|---|---|
| static, untouched → grasped but not yet moving | 0–134 | registered pose within 1 cm / 3° of the median of the first 30 frames, until 5 consecutive frames exceed it | one constant pose (median) |
| pushed away (moving) | 135–339 | – | per-frame registration + smoothing |
| turn, heavily occluded by the legs | 340–459 | (by eye; reporting only) | same |
| pushed back, back-and-forth | 460–879 | – | same |
| released, static | 880–945 (879 on the final run, see §6) | same rule as the start phase, run backwards from the last frame | one constant pose (median) |

The contact probability in `reconstruction.npz` and the mask IoU against the first and last frames agree with
these boundaries. The hand touches the stroller at ~96 but it only starts moving at ~135; contact ends at ~891,
and motion stops at ~880 within 1 cm.

## 3. Method (what `solve.py` does)

1. **Floor-contact model.** The support face is the largest convex-hull face of the mesh whose normal is within 5°
   of the baseline's average "down" direction. It is the wheel contact polygon, 2.0° from the baseline's mean
   down direction. The world pose is then constrained to yaw about gravity plus a 2D position on a plane parallel
   to the released floor. Roll/pitch are fixed (the support face lies flat), and the height is
   h0 = median baseline support height = 1.5 cm. This removes all tilt and height noise and links depth to the
   image position of the object.
2. **Initialisation.** Every baseline pose is projected onto the model with the minimal tilt correction. The
   bbox centre is kept on its viewing ray.
3. **Per-frame render-and-compare.** Using nvdiffrast, 3 parameters per frame (yaw, a, b) are optimised with 120
   Adam steps on 256 px crops. The loss is the soft IoU between the rendered silhouette and the SAM2 object mask.
   Two details matter:
   * The SAM2 masks are hole-filled and about 3 px "fat". With an undilated render, the fit biased the object
     3–13 cm towards the camera. Dilating the render by 3 px (the best of {0,2,3,4,6} px on the static frames)
     models this.
   * Pixels covered by the person (and not by the object) are ignored, which handles occlusion by the legs and hands.
4. **Flip test.** On every 6th frame, the 180° yaw hypothesis (rotated about the vertical axis through the bbox
   centre) was re-registered. It fits better on only 1% of the tested frames, isolated and never in a run, so the
   adoption rule (a run of ≥30 frames with a mean loss gain >0.05) never fires. Hypotheses that flip about other
   axes are physically excluded: they would put the stroller upside down or on its side.
5. **Temporal model.** Each static phase gets one pose: the median of its per-frame registrations, so 135 and 66
   frames are pooled respectively. For moving frames: a Hampel outlier test (window ±7 frames), then Gaussian
   smoothing with σ = 2 frames. The smoothing is weighted by visibility (visible mask area relative to its 1 s
   running mean) and by registration loss.
6. **Depth anchoring.** One global factor (1.0138) scales the camera-frame translations along their viewing rays.
   It makes the median per-frame camera depth ratio to the baseline equal to 1, so the median per-frame depth
   difference is 0.0 cm and the mean is −0.3 cm.
   * Why not match median(z) directly: that gives a factor of 1.032. On this trajectory, median(z) is decided by
     the frames near 3.1 m (mainly the final static phase), and it would shift the whole trajectory 5.3 cm
     farther than the baseline on average.
   * Why one factor for the whole episode rather than one per phase: the floor model links the phases physically.
     Per-phase factors would reintroduce the 3 cm "moving vs resting" height inconsistency.

### Depth cues (reported, not applied; applied shift = 0 cm)

| cue | suggested shift along the ray (at the median depth 3.09 m) |
|---|---|
| untouched stroller resting on the released floor plane (lowest point at h = 0) | −0.1 cm (the result already has its lowest point at −0.1 cm) |
| floor height from the true frame-0 human soles (+1.7 cm above the released plane, right next to the stroller) | −3.9 cm |
| hand contact in the occluded turn phase (only phase where the human was not attached to the baseline object) | hands touch the handle (<1 cm) for any shift between −9.3 and 0.0 cm; contact is lost beyond +1 cm |
| metric mesh size vs image size (6-DoF silhouette fit on the static phases) | inconsistent (−7 cm in the end phase; it trades off against tilt), so not usable |

The floor-plane and soles cues disagree by 3.8 cm, which is more than 2 cm and about as large as the correction
itself. The rule therefore says keep the baseline depth, and the applied shift is 0. All cues agree that the result
should not be farther from the camera.

## 4. Validation (no ground truth)

`validate.py` uses crops around the baseline. "IoU dil3" is the silhouette IoU against SAM2 masks with person
pixels excluded and a 3 px dilated render; "raw" uses no dilation. "accel" is the median second difference.
"lowest pt h" is the lowest mesh point above the released plane. "hand-obj" is the median minimum distance between
hand keypoints and the mesh in frames with contact probability >0.5. "dz" is the camera-frame depth of our result
minus the baseline's.

| phase (frames) | trajectory | IoU dil3 | IoU p10 | IoU raw | accel t cm/f² | accel R °/f² | lowest pt h cm (med / std) | tilt ° | hand-obj cm | dz vs base mean / med cm |
|---|---|---|---|---|---|---|---|---|---|---|
| static_start (0-134) | baseline | 0.735 | 0.657 | 0.571 | 1.34 | 0.82 | -0.7 / 1.2 | 2.6 | 0.3 | 0.0 / 0.0 |
| static_start (0-134) | **result** | **0.862** | 0.843 | 0.666 | 0.00 | 0.00 | -0.0 / 0.0 | 0.0 | 0.4 | -2.4 / -3.5 |
| push_away (135-339) | baseline | 0.727 | 0.678 | 0.616 | 1.39 | 0.96 | 1.5 / 1.1 | 2.6 | 0.3 | 0.0 / 0.0 |
| push_away (135-339) | **result** | **0.874** | 0.844 | 0.707 | 0.02 | 0.03 | -0.0 / 0.0 | 0.0 | 0.3 | 2.3 / 2.8 |
| turn_occluded (340-459) | baseline | 0.762 | 0.679 | 0.628 | 2.41 | 2.88 | -1.5 / 1.5 | 2.7 | 4.3 | 0.0 / 0.0 |
| turn_occluded (340-459) | **result** | **0.858** | 0.803 | 0.643 | 0.04 | 0.12 | -0.0 / 0.0 | 0.0 | 0.9 | -4.8 / -3.4 |
| return_back_forth (460-879) | baseline | 0.803 | 0.759 | 0.605 | 1.47 | 1.27 | 1.6 / 1.4 | 2.8 | 0.3 | 0.0 / 0.0 |
| return_back_forth (460-879) | **result** | **0.881** | 0.854 | 0.630 | 0.06 | 0.06 | -0.0 / 0.0 | 0.0 | 0.3 | 0.9 / 0.5 |
| static_end (880-945) | baseline | 0.889 | 0.864 | 0.665 | 1.01 | 0.88 | -1.2 / 1.1 | 1.1 | 1.3 | 0.0 / 0.0 |
| static_end (880-945) | **result** | **0.917** | 0.908 | 0.682 | 0.00 | 0.00 | -0.0 / 0.0 | 0.0 | 2.2* | -3.6 / -4.7 |
| ALL (0-945) | baseline | 0.778 | 0.693 | 0.610 | 1.44 | 1.16 | 1.0 / 1.8 | 2.6 | 0.3 | 0.0 / 0.0 |
| ALL (0-945) | **result** | **0.877** | 0.845 | 0.657 | 0.03 | 0.04 | -0.0 / 0.0 | 0.0 | 0.3 | -0.3 / -0.0 |

\* Contact frames 880–891 are while the hand lets go; the stroller is already treated as static there.

**Held-out frames.** The odd frames were predicted only from the per-frame registrations of the even frames
(smoothing with the odd frames' weights set to 0), then scored against the odd frames' masks (IoU dil3):

| phase | held-out prediction | baseline | frames where held-out is better |
|---|---|---|---|
| static_start | 0.850 | 0.735 | 100% |
| push_away | 0.864 | 0.727 | 100% |
| turn_occluded | 0.853 | 0.760 | 98% |
| return_back_forth | 0.875 | 0.803 | 100% |
| static_end | 0.911 | 0.889 | 100% |

**Other checks done during development:**

* On the static frames, two independent image terms gave the same yaw correction relative to the baseline
  (−4 to −5°) and the same lateral correction (−1.2 to −1.6 cm): the 3 px-dilated silhouette, and a photometric
  render-over-background-plate term.
* Overlays of the result at 12 frames across the episode (0 … 930), including the turn, align with the stroller.
* Hand keypoints were never used in the fit. Even so, the result keeps the reconstructed hands on the handle in
  every pushing window (median 0.3–0.9 cm). The baseline fails this in the turn (4.3 cm), and so does the plain
  floor projection of the baseline in frames 500–560 (3–4 cm).

**Difference from the baseline.** Median rotation change: 5.1° (static start), 2.9° (push away), 4.3° (turn), 3.8°
(return), 1.3° (static end). Median ADD between our poses and the baseline's: 3.3–4.8 cm per phase.

## 5. Confidence per phase

* **Static start (0–134), high.** A single pose pooled over 135 frames, with an unoccluded large view. The yaw
  correction was confirmed by two image terms.
* **Push away (135–339), high.** Good visibility, smooth motion, silhouette IoU 0.87.
* **Turn, occluded (340–459), medium.** The stroller is largely hidden behind the legs for ~60 frames. The result
  is smooth, keeps hand contact and fixes the baseline's depth jumps, but its yaw during frames ~380–420 depends on
  few visible pixels plus smoothing.
* **Return / back-and-forth (460–879), medium-high.** The object is small (≈1–2k px), so yaw is less certain than
  near the camera. The position is well constrained by the floor model.
* **Static end (880–945), high** for rotation and image position.
* **Absolute depth (all phases): the main remaining uncertainty.** It is inherited from the baseline by
  construction. The physical cues bracket the true value between about 4 cm closer and the current value.

## 6. Known limitations and things not done

* Tilt is fixed at 0 relative to the floor. A free 6-DoF silhouette fit preferred 1–4° of tilt on the static
  frames, but this is confounded with depth and was not adopted. If the swivel wheels make the real stroller sit
  slightly tilted, the cost is ~1 cm ADD.
* Run-to-run differences of up to ~4 mm in translation come from non-deterministic atomics in nvdiffrast's
  backward pass. Because of this, the detected end of the final static phase is frame 879 or 880 depending on the
  run (the validation table used 880; the submitted file used 879). Seeds are fixed, and every decision (phase boundaries, flip rule) is automatic.
* No new networks were run (FoundationPose re-registration, point trackers): the floor-constrained silhouette
  registration was already consistent on held-out frames.
* GPU use for the whole development, including experiments, was roughly 15 min. The final `solve.py` uses about 1 min.
