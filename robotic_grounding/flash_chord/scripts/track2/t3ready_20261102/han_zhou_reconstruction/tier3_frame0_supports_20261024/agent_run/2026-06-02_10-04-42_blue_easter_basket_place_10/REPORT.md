# Episode 2 – blue easter basket: trajectory-level object pose re-estimation

Result: `object_poses_world.npz` (`T_world_mesh`, 980×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(helpers in `hoilib.py`). Run time is about 80 s wall-clock and about 1 GPU-minute, with peak GPU memory about 4.1 GB (nvidia-smi).
Two consecutive runs give bit-identical output. Side products: `solve_log.json` and `solve_diag.npz` (diagnostics),
`validate.py` / `validation.md` (the validation table below).

## 1. What the video shows (phases)
The object is small: about 2000 px (about 50×45 px) at 3.3–3.9 m, and at most about 4400 px when carried close to the camera.

| phase | frames | content | how found |
|---|---|---|---|
| S1 | 0–124 | basket rests on the round table (the hand touches it from ~95, it starts moving at 125) | mask bbox constant ±1 px, by eye |
| C1 | 125–213 | carried with one hand and put on the floor | |
| S2 | 214–436 | rests on the floor (the person walks around it, with partial occlusions) | mask bbox constant 214–436 |
| C2 | 437–846 | picked up, handed over to the other hand, carried. Visible until 742; **hidden in front of the body 743–846**; put down at ~846 | mask area → 0 at 743 |
| S3 | 847–979 | rests on the floor behind/between the person's legs. Only a 20 px wide sliver is visible (847–867); fully occluded afterwards | by eye + mask |

Boundaries are explicit constants in `solve.py` (`PHASES`, `OCCLUDED_CARRY`).

## 2. Diagnosis of the baseline
* **Yaw about the bucket axis is inconsistent.** The bucket is a near-rotationally-symmetric pail. Its texture is a
  ring of 8 identical bunnies (45° period), and the wire bail hangs from two lugs that are 180° apart. The baseline
  yaw is 65° in S1. In S2 it takes the values −69/−150/178/−132/−157 within one static phase. In C2 it jumps by 45° at
  ~600 and by 120° at 645–650 while the hand does not rotate. In S3 its yaw spread (circular std) is 63°.
* **Image misalignment while carried:** at 640–735 the baseline silhouette is 15–30 px off the mask (6–9 cm
  sideways at 2.5 m). Here the IoU drops to 0.25–0.35.
* **Lost tracking:** during 743–846 the baseline places the object 0.5–1.9 m too far (z 4.5–5.0 m while the person is
  at ~2.7 m) and upside down (tilt 140–177°).
* **S3:** tilt fluctuates 2–80°, and the position jitters by several cm while the object is at rest.
* The baseline is otherwise well placed: S2 depth puts the bucket bottom 0.1 cm from the released floor plane.

## 3. Method (all in `solve.py`)
1. **Deterministic silhouette fitting.** A Levenberg–Marquardt fit with finite differences on forward-only
   nvdiffrast renders: 4× supersampled coverage, 128² crops, person-mask pixels ignored. Pass 1 fits per frame:
   image position of an axis point plus a small rotation increment, with **depth fixed to the baseline**. Static
   phases start upright. I first used Adam with nvdiffrast's antialias gradients, but its atomics made runs differ by up to 6° / 2.7 cm. The LM version is
   bit-reproducible and fits better (IoU 0.906 vs 0.87).
2. **Yaw evidence from texture unwrapping.** For every visible frame, the video crop is unwrapped onto the bucket's
   cylindrical wall (azimuth × height), using the per-pixel mesh azimuth rendered from the pass-1 pose and only the
   outer wall facing the camera. The panorama is high-passed and circularly cross-correlated with the panorama of the
   mesh texture. The mesh texture's bunny outlines are dark, while in the video they are bright perforations, so the
   correlation uses the inverted texture; a synthetic test with known yaw offsets recovered them exactly. This gives
   a per-frame score over absolute world yaw. In C2 (545–735) it shows clear ridges that move smoothly with the
   object's rotation.
3. **Physical lug cue.** In S1 (wire loop at the left edge, frames 0–100), S2 (lug at the left edge, frame 350) and
   C2 (555–700, wire loops on both sides), the bail lugs are seen at the silhouette sides (verified by eye; constant
   `LUG_SIDE_RANGES`). So the lug axis must be perpendicular to the viewing ray: cost 3·cos²(angle).
4. **Weak baseline vote** (0.3·cos(Δyaw)), used only on frames where the baseline IoU > 0.6 and the frame is reliable.
5. **Global decoding over time.** Forward–backward over 120 yaw bins for all 980 frames. Static phases are forced to
   one constant yaw, and their total evidence weight is capped (near-identical frames). While carried, yaw may change
   ≤ 6°/frame with a quadratic cost. Temperature 3 accounts for correlated evidence. Each frame takes the circular
   mean of its posterior's dominant mode (±60° around the MAP bin). A hard Viterbi argmax flipped S2 by 20–40°
   between near-identical inputs; the posterior mean is stable.
6. **Rotations.** Static phases are upright on the released floor normal (the table top is assumed parallel to the
   floor; the baseline tilt in S1 was 0.8°). Carried frames use the pass-1 bucket axis from reliable frames only
   (visible mask ≥ 60 % of the expected silhouette, IoU > 0.6), forced upright at both ends of each carry
   (picked from / put on a horizontal support), interpolated and smoothed (σ = 3 frames). This is combined with the
   decoded yaw.
7. **Translations.**
   * Static phases: one shared image position fitted on all visible frames, at the baseline median depth.
   * Carried: per-frame LM image position, then visibility-weighted temporal smoothing (σ = 2 frames). Depth is the
     smoothed baseline (median filter 9, Gaussian σ = 2).
   * **Occluded carry 743–846:** the object is attached to the holding hand. This is the MHR hand keypoint set that
     was ~10 cm from the object before the occlusion. The object–hand offset is interpolated linearly from its value
     before the occlusion to its value at S3. Rotation follows the smoothed axis/yaw (upright at the put-down).
8. **Depth re-anchoring.** Per phase, camera-frame positions are scaled along the viewing rays so that the median
   camera z of the mesh origin equals the baseline's median over the same frames. Scale factors are 0.998, 0.998,
   0.999, 1.003, 0.998. The occluded carry is excluded from re-anchoring, because the baseline has lost the object
   there.

## 4. Depth along the viewing ray (applied shift: 0, except the occluded carry)

| phase | cue 1: floor contact | cue 2: hand | cue 3: image size | decision |
|---|---|---|---|---|
| S2 | lowest point 0.1 cm above the floor plane → shift +0.1 cm | hand at put-down 10.8 cm from the bucket centre (a grasp on the rim is plausible) → 0 | – | baseline depth confirmed, shift 0 |
| S3 | lowest point **7.7 cm below** the floor plane → shift **−18.8 cm** (towards the camera) | the releasing hand is 40 cm away; reaching ~12 cm needs **−37.7 cm** | visible mask height 55–57 px < rendered 62 px → farther (positive shift) | the cues disagree by ~19 cm and in sign → **shift 0 (baseline)**. Low confidence in S3 depth |
| C2 occluded (743–846) | – | object follows the hand: z is ~3.1 m vs the baseline's 4.5–5 m | – | the baseline is clearly lost (object 0.5–1.9 m behind the person). Here the hand chain, anchored at both ends by the re-anchored C2 visible part and S3, replaces it. Mean / median dz vs baseline −81.5 / −54.4 cm |

## 5. Validation (no ground truth): baseline vs result (`validate.py`)
IoU and centroid error are measured against the SAM2 object mask, with person pixels ignored. Texture z is the
normalised unwrapped-texture correlation at the trajectory's yaw (higher is better; not independent of the
optimisation). Lug cos² is measured only in lug-verified ranges (0 = lugs at the silhouette sides). dz = ours − baseline
camera z of the mesh origin.

| phase | frames | vis | IoU base | IoU ours | centroid err px base/ours | texture z base/ours | lug cos^2 base/ours | median dR deg/frame base/ours | median acc cm/f^2 base/ours | floor height cm base/ours | tilt vs floor deg base/ours | hand dist cm base/ours | dz mean/median cm (ours-base) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_table_static | 0-124 | 125 | 0.85 | 0.92 | 3.27/0.39 | 0.23/2.08 | 0.51/0.00 | 0.13/0.00 | 0.04/0.00 | 81.82/81.91 | 0.84/0.00 | 78.97/78.10 | -0.41/0.00 |
| C1_carry | 125-213 | 86 | 0.75 | 0.85 | 2.64/0.77 | -0.06/0.39 | -/- | 6.67/1.14 | 2.64/0.19 | - | 8.14/14.76 | 15.88/14.63 | -0.92/-0.55 |
| S2_floor_static | 214-436 | 223 | 0.80 | 0.85 | 3.75/0.59 | 0.13/-0.29 | 0.77/0.23 | 0.00/0.00 | 0.10/0.00 | 1.08/0.08 | 2.59/0.00 | 61.44/61.95 | 0.24/0.00 |
| C2_carry_handover | 437-846 | 304 | 0.64 | 0.89 | 6.61/0.60 | 0.02/0.52 | 0.34/0.02 | 3.98/0.40 | 2.47/0.12 | - | 6.35/4.34 | 12.21/14.74 | -19.91/0.75 |
|  ↳ C2 visible part | 437-742 | | | | | | | | | | | | 1.03/0.98 |
|  ↳ C2 occluded carry | 743-846 | 0 | | | | | | | | | | | -81.54/-54.39 |
| S3_floor_static_occluded | 847-979 | 20 | 0.45 | 0.89 | 5.97/0.52 | -/- | -/- | 5.62/0.00 | 5.17/0.00 | 5.27/-7.67 | 12.05/0.00 | 73.04/83.85 | 0.33/0.00 |
| ALL | 0-979 | 758 | 0.73 | 0.88 | 3.64/0.54 | 0.08/0.53 | 0.55/0.10 | 1.48/0.05 | 1.82/0.03 | - | 3.95/0.44 | 19.98/19.74 | -8.37/0.16 |

Yaw per phase (circular mean, world): baseline 65 / 273 / 219 / 293 / 93 vs ours 294 / 271 / 267 / 289 / 291.
The maximum rotation jump per frame drops from 136 / 35 / 58 / 159 / 79° (baseline) to 0 / 6 / 5 / 10 / 0°.

## 6. Confidence per phase
* **S1 (table):** position high (IoU 0.92, centroid 0.4 px; same depth). Upright tilt high.
  **Yaw medium-low:** the texture and the lug cue both choose 294°, which is 131° from the baseline's 65°. The
  strongest alternative is the 180° twin (~105°). This is what FoundationPose's own first registration (frames 0–50)
  returned before it drifted to 65°. If the truth is the twin, this phase becomes worse than the baseline. Evidence
  in favour of 294°: texture z 2.08 vs about 0.8 at ~105°, and consistency with C2, where the large view supports the
  290° family.
* **C1 (carry):** position high (IoU 0.75 → 0.85), yaw medium-low (smooth, ~270°).
* **S2 (floor):** position high, and the depth is confirmed by two cues. **Yaw low-medium:** the evidence is spread
  over 258–315°, and the decoder picks 267°. The phase's own data alone prefers about 290°, while continuity with
  C1 pulls it lower.
* **C2 visible (437–742):** position high. This is the biggest clear gain: the 6–9 cm lateral errors at 640–735 are
  fixed (IoU 0.64 → 0.89). Yaw medium: lugs at the sides verified by eye, texture ridges tracked, and the baseline
  agrees (~280°) on 650–735.
* **C2 occluded (743–846):** position medium-low (hand-attached; the baseline was off by 0.5–1.9 m), rotation low
  (smooth interpolation).
* **S3 (floor, occluded):** rotation upright (high for tilt; baseline tilt was up to 80°), yaw low (continuity only).
  Position medium-low: the image position comes from a 20 px sliver, and the depth stays at the baseline even though
  floor contact suggests the basket is 19 cm closer (see §4).

## 7. What did not work / was not used
* Simple render-and-compare scores (IoU, colour, gradient NCC) over yaw disagreed with each other and were not used.
* The FoundationPose ScorePredictor (run on yaw hypotheses for frames 30, 350, 580–700) returned essentially flat
  scores (spread 0.03 logits) on these small crops. It was not used.
* Re-estimating depth from silhouettes would move the object ~5 % closer (masks slightly larger than the render).
  It was rejected, following the task guidance and the floor-plane check in S2.
* `/pipeline/foundationpose/*.pkl` contains `gt_*` / `oracle` fields. These were never read or used.

## 8. Run time
`solve.py` takes about 80 s wall-clock and about 1 GPU-minute (pass-1 LM fits about 20 s, texture unwrapping about
50 s mostly CPU, final fits about 10 s). Total GPU time during development, including about 12 full runs and the
experiments, was roughly 25 min.
