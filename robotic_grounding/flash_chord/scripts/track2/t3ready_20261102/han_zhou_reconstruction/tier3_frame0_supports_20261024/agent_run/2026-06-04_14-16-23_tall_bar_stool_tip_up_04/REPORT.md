# Episode 0 – tall bar stool: object trajectory re-estimation

Task in the video: the person walks to a bar stool lying on the floor, grips it with both hands, tips it
back to upright and sets it down, then walks around behind it.

Result: `object_poses_world.npz` (`T_world_mesh`, 778×4×4, float64). It was produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(2 min 50 s wall time on one GPU, of which about 2 min is GPU fitting; peak torch GPU memory about 3.2 GB, measured; per-frame fits run in chunks of 32 frames).

## 1. Diagnosis of the baseline

* **No 180° flips, jumps or lost tracking.** I rendered the textured mesh over the video at 15+ frames across
  the episode. The backrest makes the stool clearly asymmetric, and the baseline orientation is right
  everywhere. The largest frame-to-frame baseline rotation change is < 5°.
* **The rotation is consistently about 2–3° off** (up to about 5–6° around frames 540–600). Render-and-compare
  fits from the baseline always move the rotation the same way: about 2.5–3° in both static phases. Two mask
  models (raw render vs. a 1-px-dilated render) agree to within 0.5–1.5° with each other, and both move
  2.5–3° away from the baseline. In the overlays the thin legs are visibly misplaced in the baseline
  (e.g. frames 380, 460, 560) and aligned after the fit.
* **The baseline has per-frame noise and drift during static phases.** The stool is untouched for
  frames 0–274, yet the baseline's camera depth wanders between 2.70 and 2.81 m (std 3.0 cm). Its median
  ADD to its own mean pose is 2.4 cm in phase A and 1.4 cm in phase B. In the moving phase the
  mesh-centre acceleration jitter is 12 mm/frame².
* **Depth (placement along the viewing ray).** Render-and-compare with free depth puts the object 6–9 cm
  closer than the baseline. That value depends entirely on the assumed fattening of the SAM2 masks: a
  1-px dilation of the render changes it from −10 cm to about −1 cm, and 2 px gives +3 cm. It also depends on the
  focal length. So I did not use it (see §4).

## 2. Phases (detected automatically in `solve.py::detect_phases`, verified by eye)

Criterion: d[i] is the mean |I_i − I_{i+6}| over the object masks of both frames, with person pixels
excluded. A boundary is where d > 2.5 grey levels for ≥ 8 consecutive frames. The noise floor is 1.0–2.3,
and the manipulation gives 3–15.

| phase | frames | content |
|---|---|---|
| A static | 0–274 | stool lying on the floor (backrest/seat down, legs up). Untouched until ~185, then held by both hands without moving. |
| moving | 275–621 | stool lifted at one end, tipped up and rotated by ~90°, walked to the right and set down upright |
| B static | 622–777 | stool standing upright. Hands release at ~650, and the person walks around and behind it (occluded by the stool). |

## 3. Method (`solve.py`)

1. **Occlusion-aware evidence.** The SAM2 object/person masks are used at half resolution. A person pixel only
   counts as an occluder if the reconstructed human is in front of the object there: human vertices are
   splatted into a z-buffer and compared with the rendered object depth. Without this, the late-B frames
   (person standing behind the stool) let the fit drift by 10°. A truncated distance transform of the
   Canny edges of each frame is also computed.
2. **Per-frame render-and-compare (nvdiffrast).** Starting from the baseline pose, a 6-DoF fit for each frame
   (200 Adam steps, coarse-to-fine mask blur, cosine LR decay). The loss is the silhouette L2 against the
   SAM2 mask on valid pixels plus 0.3 × the mean edge distance along the rendered antialiased silhouette
   boundary. The edge weight was chosen on 16 frames using full-resolution edge chamfer (w = 0, 0.3, 1, 3 →
   0.3 best).
3. **Static phases: one pose per phase.** One shared pose is fitted jointly to 48 frames spread over the
   phase, starting from the robust mean of the per-frame fits (400 steps; run-to-run spread < 0.1°). This
   pools the evidence from hundreds of frames and makes the object exactly static.
4. **Depth re-anchoring (before smoothing, so static and moving poses share one depth scale).** Every pose
   is moved along its own viewing ray, so the rotation and image position are unchanged:
   * static phases: z is set to the baseline's median z over the phase;
   * moving phase: z = z_fit · lowpass_σ=30(z_baseline / z_fit). The slow depth level comes from the
     baseline, and only the fast relative changes come from the image fit. Then a smooth sine-shaped
     correction (zero at both phase ends, 0.04 %) makes the moving phase's median z equal the baseline's.
5. **Moving phase: robust temporal smoothing on SE(3).** A Gaussian window (σ = 4 frames) with a weighted
   chordal rotation mean, and a weighted mean of (x/z, y/z, log z). Weights come from the fit IoU, and
   outliers (> 4° or > 8 px from the smooth track) are down-weighted in two passes. No outliers were found.
   The static poses are included as heavy anchors, so the trajectory joins the static phases continuously.
6. The camera-frame poses are mapped back to the world with `T_world_cam`.

No baseline per-frame results or correction tables are embedded. The only constants are the hyperparameters
documented at the top of `solve.py` and the phase-detection thresholds.

## 4. Depth: cues and decision (applied shift = 0)

Each cue is reported as the shift along the viewing ray (cm, + = away from the camera) that the cue
suggests relative to the delivered result (`depth_cues.py`). The result has the baseline's median depth in
every phase.

| phase | metric size (render-and-compare, free depth) | floor: lowest point to true-human sole level (−2.9 cm w.r.t. plane) | floor: plane taken literally | hand keypoints ↔ mesh surface |
|---|---|---|---|---|
| A static | −5.9 | −1.0 | −8.0 | +2.0 (flat: 0.66 → 0.40 cm) |
| moving | −6.1 | – | – | +2.0 (flat: 0.32 → 0.31 cm) |
| B static | −8.8 | +1.0 | −10.0 | +6.0 (10 frames only) |

* The size cue depends on the assumed mask dilation and on the focal length. As shown in §1, it ranges
  from −10 to +3 cm across dilation assumptions, so it is not reliable.
* The floor cue depends on where the floor really is. The released plane is a fit, and the true frame-0
  human's lowest vertex lies 2.9 cm below it. Measured against that level, the floor says −1/+1 cm, i.e. the
  baseline depth is right. Taken literally, the plane agrees with the size cue.
  The task statement says the plane has several cm of tolerance, and along these shallow viewing rays
  1 cm of height corresponds to 3–6 cm of depth.
* The hand cue (the human is calibrated to the frame-0 true human: identical mean depth) favours 0 to +2 cm.
  However, it is not fully independent, because the pipeline's post-optimisation pulled the hands onto the
  baseline object.

Two independent cues do not agree on a correction within 2 cm that is larger than their disagreement:
size and the literal plane agree, but the literal plane contradicts the true human's feet, and the hand
cue contradicts both. So no depth correction is applied (shift 0). If the true object is closer, this is
the largest remaining error source.

## 5. Validation (no ground truth; `validate.py`)

IoU is computed against the SAM2 mask with occluded pixels ignored; this is the cue that was fitted. "chamf" is
the full-resolution Canny-edge chamfer (px) of the rendered outer and depth-discontinuity edges on every 6th
frame. It is only partly fitted: the fit used half-resolution outer silhouettes only. jitR and jitT are the
median second differences of rotation and mesh-centre position. spread is the median ADD of each frame to
the phase's mean pose. floor is the lowest mesh point above the released plane. hand is the median closest
hand-keypoint ↔ mesh distance on contact frames. dz is the camera-frame depth difference result − baseline.

| phase | traj | IoU ↑ | chamf px ↓ | jitR deg | jitT mm | spread cm | floor cm | hand cm | dz mean cm | dz med cm | ADD to base cm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A static | baseline | 0.728 | 3.13 | 0.142 | 8.51 | 2.39 | -2.8 | 2.47 | +0.00 | +0.00 | 0.00 |
| A static | result | 0.792 | 2.26 | 0.000 | 0.00 | 0.00 | -3.4 | 0.66 | -0.35 | +0.00 | 3.91 |
| moving | baseline | 0.797 | 2.30 | 0.201 | 11.77 | – | – | 0.34 | +0.00 | +0.00 | 0.00 |
| moving | result | 0.838 | 1.56 | 0.010 | 0.24 | – | – | 0.32 | -0.08 | -0.02 | 2.20 |
| B static | baseline | 0.784 | 2.32 | 0.096 | 19.08 | 1.44 | -2.6 | 0.41 | +0.00 | +0.00 | 0.00 |
| B static | result | 0.816 | 2.06 | 0.000 | 0.00 | 0.00 | -2.7 | 2.32 | +0.39 | +0.00 | 2.19 |
| all | baseline | 0.770 | 2.60 | 0.153 | 11.04 | – | – | 0.41 | +0.00 | +0.00 | 0.00 |
| all | result | 0.817 | 1.91 | 0.000 | 0.00 | – | – | 0.39 | -0.08 | -0.01 | 2.80 |

Note: the result's IoU and chamfer are measured after re-anchoring to the baseline depth, which makes the
render 4–6 % smaller than the image. The image-only fit before re-anchoring reaches IoU 0.85–0.87.

Additional checks:
* **Clean static images.** For each static phase I built a temporal-median image with person pixels
  masked out, which gives a person-free view of the stool. Edge chamfer render→image: A 3.23 (baseline mean
  pose) → 2.29 (result); B 2.33 → 2.01 (image→render: A 3.99 → 3.39, B 3.87 → 3.88).
* **Physical plausibility.** In both static phases the lowest point stays at the same height relative to the
  floor plane as in the baseline (−3.4/−2.7 cm vs −2.8/−2.6 cm). In B the four leg-tip heights span 1.6 cm in the baseline
  and 1.3 cm in the result, so the rotation change did not tilt the stool off the floor.
  Hand contact is unchanged overall (0.39 vs 0.41 cm). It improves in A (0.66 vs 2.47 cm) and gets worse in
  the 10 B frames with contact (2.3 vs 0.4 cm), where the baseline hands had been optimised onto the
  baseline object.
* **Reproducibility.** Two independent full runs differ by 0.02/0.06/0.23 cm mean ADD (A/moving/B) and a
  median of 0.03° rotation. Optimiser settings (200 vs 400 vs 800 steps) change the depth-normalised poses
  by about 0.3 cm ADD. Both are an order of magnitude below the change w.r.t. the baseline (2–4 cm ADD).
  The residual run-to-run variation comes from nvdiffrast's atomic gradient accumulation; seeds are fixed.
* **Phase boundaries.** I inspected frames 268–292 and 606–630 visually. The trajectory leaves the static
  pose and joins the next one smoothly.

## 6. Confidence per phase

| phase | change vs baseline | confidence |
|---|---|---|
| A static (0–274) | one constant pose (removes 2–3 cm baseline jitter/drift); rotation corrected by 2.9°; depth = baseline median | high for "static" and for the rotation direction (all fits and edge checks agree); medium on the exact 2.9° (±0.5–1°) |
| moving (275–621) | per-frame refit + smoothing: rotation corrected 1–6° (largest at 540–600), legs aligned; jitter removed; depth follows the baseline level | medium-high: consistent gains on edge chamfer (2.30 → 1.56 px) and IoU on nearly all frames |
| B static (622–777) | one constant pose; rotation corrected by 2.7°; depth = baseline median | medium: the improvement on the clean median image is smaller (2.33 → 2.01 px one way, equal the other), and the D0/D1 mask models differed by 1–3° here |
| all | depth (placement) unchanged on purpose | the main remaining uncertainty is a possible common depth offset (§4), which the baseline also has |

## 7. Time / resources

Development used about 30 GPU-minutes in total (including all experiments). The final `solve.py` run takes
2 min 50 s wall, about 2 min of it GPU, with peak torch GPU memory about 3.2 GB. Nothing was downloaded. See
TOOLS.md.
