# Episode 15 — wet floor sign: trajectory-level object pose recovery

Final output: `object_poses_world.npz` (`T_world_mesh`, 715×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(run time 3 min 44 s wall, about 2.5 min of it on the GPU; peak GPU memory about 3 GB allocated / 4 GB reserved, measured on one 60-frame batch). Validation: `validate.py` (numbers below).

## 1. What the video shows (phases)

The object is an A-frame "CAUTION WET FLOOR" sign, 65 cm tall. Mesh z is vertical, mesh x is the direction the
two panels spread apart, and mesh y runs along the panel width and handle. The person walks to the sign, lifts it
by the top handle with the right hand (MHR keypoints 21–41), walks a full loop, and puts it down somewhere else.

| phase | frames | how detected | content |
|---|---|---|---|
| static start | 0–168 | object-mask IoU (outside the person mask) with the first 10 frames ≥ 0.9 | untouched on the floor; the hand touches it from about frame 130 |
| carried | 169–591 | between the two static phases | lifted (169–225, rotating through a face-on view), carried by the handle; 3/4 and face-on views; edge-on V views at 375–405 and 545–590; behind the legs at 430–470; **fully hidden 488–500** |
| static end | 592–714 | mask IoU with the last 10 frames ≥ 0.9 | back on the floor; the hand lets go at about 617 |

The phase boundaries are found automatically from the masks (`trajectory.detect_static_phases`). I checked them
by eye: the mask IoU with frame 0 falls from 0.965 (frame 166) to 0.916 (168) and 0.849 (170); on the way down
the sign reaches its final position at frame 591/592.

## 2. Diagnosis of the baseline

* **Depth jitter in the static start phase.** While the sign is untouched, the baseline's camera-frame depth
  wanders between 3.554 and 3.658 m. The mean per-frame displacement from one fixed pose is 2.7 cm, and up to
  about 6 cm on some frames. The rotation there is already constant.
* **Small drift in the end phase** while the hand is still on the sign (592–625): about 2 cm in y.
* **A jump during full occlusion (488–504).** The baseline yaw jumps 67° → 111° within 8 frames. Rotation speed
  reaches 23°/frame and angular jitter is 6°/frame² in 430–510, against about 1.5 elsewhere.
* **Systematic yaw errors of a few degrees**, seen by per-frame render-and-compare (section 3):
  * −3.5° over the whole start phase (very consistent, std 0.7°);
  * +5° at 219–246;
  * +8 to +16° at 405–480 (visible-mask IoU rises from 0.76–0.85 to 0.90–0.96);
  * −4 to −11° in the edge-on segment 384–402.

  Silhouette-only and texture-weighted variants of the fit agree on these corrections (per-segment correlation
  0.69–0.91; `analysis/yaw_robustness.py`).
* **180° symmetry flip: none found** (section 4).

## 3. Method (`solve.py`)

1. **Evidence.** For every frame, a 320×320 crop around the SAM2 object mask holds the image, the object mask and
   a valid mask (pixels not covered by the person). It also stores the fraction of the baseline render that is
   hidden by the person ("visibility").
2. **Per-frame refinement** with nvdiffrast. The textured mesh is rendered with soft silhouettes (antialias), and
   the cost is the silhouette L2 against the object mask on valid pixels, plus 0.3 × a locally-normalised colour
   residual. Free parameters per frame: **yaw about the mesh's vertical axis and the image-plane position**.
   Depth and tilt stay at the baseline values (tilt prior σ = 2° with weight 1.0, so it is effectively fixed).
   There are 100 Adam steps per frame, starting from the baseline. A second pass started from the smoothed
   trajectory moves yaw by only about 1°, so one pass has converged.
   *Why tilt is not refined:* with tilt free, the end-phase fit tilts the sign 7–13° about its panel-width axis.
   That would lift two feet 3–6 cm off the floor, which is physically impossible for an untouched sign. The
   cause is mask dilation on the thin legs and the near edge-on view. So silhouettes are biased for tilt here;
   yaw is the well-observed degree of freedom (V width, panel visibility).
3. **Static phases**: one pose each:
   * rotation = chordal mean of the refined per-frame rotations (visibility-weighted);
   * image position = median ray;
   * depth = **baseline median depth**.
4. **Carried phase**: robust SE(3) smoothing (`trajectory.smooth_se3`):
   * data term: Huber on the rotation and translation residuals to the refined observations, weighted by
     visibility² (minimum 0.02, so occluded frames are interpolated);
   * prior: a second-difference (constant-velocity) prior on rotation increments and positions;
   * boundaries: hard ties to the two static poses.

   The smoothing strength is chosen automatically by hold-out cross-validation over {1, 3, 10, 30}: every 5th
   well-visible frame is removed and the error predicting it is measured. λ = 1 wins (λ = 3 is nearly tied).
5. **Depth re-anchoring.** In each phase, the camera-frame positions are scaled along their viewing rays so that
   the median camera z equals the baseline's median z over the same frames. The scales applied were 1.00000,
   1.00013 and 1.00000, because depth was never changed in the first place.
6. Map back to the world frame with `T_world_cam`.

## 4. The 180° question (yaw about the vertical axis)

The two panels carry identical prints, so a 180° rotation about mesh z gives nearly the same silhouette and
texture. My photometric "flipped vs not" test gave contradictory answers: the baseline wins at 0–104 and
208–400, the flip wins at 112–200 and 408–714. A metric that flips its preference inside a single static phase
is not trustworthy at this margin. The decision therefore rests on three independent, more reliable arguments:

* **Body lock.** The sign's yaw relative to the person's body (shoulder/hip axis) stays at 161–180° throughout
  222–620, while the person turns a full 360° loop (`analysis/body_relative_yaw.py`). A tracker flip would show
  up as a 180° jump in this relative angle; there is none.
* **Continuity through the pick-up.** At 160–236 the sign turns from "second panel showing on the left" through
  face-on to "second panel showing on the right", and is never seen edge-on. A continuous rotation that ends
  180° away would have to pass through an edge-on view, so the start phase and the carried phase share the same
  flip state. The same holds at the put-down.
* **Colour of the asymmetric tape.** The two hinge sides carry different tape (a pink disc vs. an orange roll).
  Among reddish pixels, the median B/R ratio is:

  | frames | image | baseline | flipped |
  |---|---|---|---|
  | start (0–100) | 0.03 | 0.05 | 0.09 |
  | end (600–700) | 0.09–0.10 | 0.11 | 0.06–0.07 |

  The baseline matches better in both phases (`analysis/flip_hue_check.py`).

**Decision: keep the baseline's flip state everywhere.** Confidence is moderate to high. The continuity
arguments are strong, and the remaining risk is a flip of the whole episode, which the colour cue argues
against.

## 5. Validation (no ground truth)

Notes for the tables:
* IoU is the visible-mask IoU (render minus the person mask, against the SAM2 object mask, in 320 px crops),
  over frames with visibility > 0.3.
* "held-out IoU": the trajectory is refitted with all odd frames removed and scored on the odd frames. This
  tests whether the corrections generalise to frames that were not fitted.
* photo is the locally-normalised colour residual (lower is better).
* Jitter is the mean |Δ angular velocity| in °/frame².
* Depth Δ is the result's camera-frame z minus the baseline's, in cm.

| phase | IoU base → result | held-out IoU base → result | photo base → result | jitter base → result | mean rot change | mean / median depth Δ (cm) | lowest point above floor (cm) base → result |
|---|---|---|---|---|---|---|---|
| static start (0–168) | 0.832 → **0.852** | 0.832 → 0.852 | 1.156 → **0.978** | 0.03 → 0 | 3.3° | +0.06 / 0.00 | 1.57 → 1.20 |
| carried (169–591) | 0.774 → **0.830** | 0.771 → **0.830** | 1.242 → **1.068** | 2.65 → 0.43 | 3.9° | +0.04 / +0.07 | – |
| static end (592–714) | 0.588 → **0.658** | 0.589 → 0.658 | 1.398 → **1.333** | 0.21 → 0 | 0.9° | −0.26 / 0.00 | 0.62 → −0.32 |
| ↳ pick-up 169–225 | 0.861 → 0.893 | 0.857 → 0.890 | 1.076 → 0.805 | 1.56 → 0.30 | | | |
| ↳ edge-on 375–405 | 0.552 → 0.611 | 0.529 → 0.613 | 1.496 → 1.464 | 5.13 → 1.68 | | | |
| ↳ occlusion 430–510 | 0.812 → 0.882 | 0.814 → 0.884 | 1.408 → 1.238 | 6.13 → 0.47 | | | |

Other checks:
* Cross-validation of the smoother: held-out prediction error is 1.24° / 1.02 cm (λ = 1).
* Run-to-run reproducibility: two runs of `solve.py` differ by 0.13° and 1.0 mm on average, and by at most
  1.4° and 11 mm (on fully occluded frames). The cause is float atomics in nvdiffrast's backward pass; every
  seed is fixed.
* The end-phase IoU stays low (about 0.66) because the legs are thin and SAM2 dilates them. The mask area is
  about 1.25× the render area in that view.

### Depth (distance along the viewing ray): cues and decision

| cue | suggested depth shift |
|---|---|
| start phase: lowest point exactly on the released floor plane | +3.9 cm |
| start phase: lowest point 1.8 cm above the plane (offset of the frame-0 reference human's soles) | −2.1 cm |
| end phase: on the plane / 1.8 cm above it | −0.8 cm / −5.6 cm |
| carried: handle vs. right-hand finger bases | −5.9 cm (IQR −8.0 to −2.9) |
| silhouette size (free-depth fit) | +4 to +19 cm, inconsistent between frames of the same static pose; rejected |

The cues disagree by much more than 2 cm. The floor-plane fit itself has several cm of tolerance (median foot
distance 4.1 cm). For comparison, the reconstructed human at frame 0 sits 2.2 cm farther than the reference
human. **Applied depth shift: 0 in every phase.** The result's median camera depth equals the baseline's.

## 6. What changed, and confidence per phase

* **Static start (0–168)**, confidence high: one constant pose. This removes 2.7 cm mean / up to about 6 cm of
  depth and position jitter, and corrects yaw by −3.5°, which every cost variant agrees on.
* **Carried (169–591)**, confidence medium-high: yaw corrections of a few degrees that are coherent in time and
  confirmed on held-out frames; the full-occlusion jump at 488–504 is replaced by a smooth interpolation; frame
  noise is smoothed out. Lowest confidence: 488–500 (fully hidden, interpolated) and the edge-on segment
  375–405 (thin silhouette).
* **Static end (592–714)**, confidence high: one constant pose, with less than 1° of change from the baseline
  rotation.
* Kept unchanged on purpose: the flip state, tilt, and depth along the viewing ray.

## 7. Files

* `solve.py` is the full pipeline, with helpers in `common.py` (rendering, I/O), `evidence.py` (crops,
  render-and-compare, yaw refinement) and `trajectory.py` (phases, static poses, robust SE(3) smoother).
* `validate.py` produces the tables above (`diag.npz` and `diag_val.npy` are its inputs/outputs).
* `analysis/` holds the scripts behind sections 2, 4 and 5 (the yaw scan, robustness check, body-relative yaw,
  hue flip check and depth cues). `solve.py` does not need them. `handframe.py` is used by `depth_cues.py`.
