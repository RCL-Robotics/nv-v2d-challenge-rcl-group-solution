# Bat trajectory, episode 4: report

Final output: `object_poses_world.npz` (`T_world_mesh`, 803×4×4 float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(deterministic, offline; log in `solve_log.txt`, validation in `validation_table.md`, `validation_details.npz`,
report-only depth cues in `depth_cues.json` from `depth_cues.py`).

## 1. What I found in the data

- **Object.** The object is a 54 cm toy bat with a light-blue barrel (radius 4.5 cm, mesh −z) and a dark-blue handle
  (radius 1.5 cm, mesh +z) ending in a knob. The bat is almost a solid of revolution, so the silhouette determines
  the axis. Which end is the barrel is visible from width and colour. The roll about the long axis is only weakly
  visible.
- **Phases.** These are detected automatically: the object mask, ignoring pixels covered by the person, keeps IoU
  ≥ 0.9 with the first or last frame's mask.
  - static-1, **0–110**: the bat lies on the floor.
  - moving, **111–705**: pickup (≈111–160), then carried and spun with both hands while the person turns around
    (≈160–650, with long stretches where the person's back faces the camera and the barrel is hidden behind the
    head or arms, ≈330–410), then put down (≈650–705).
  - static-2, **706–802**: the bat lies on the floor at a new place.
- **Baseline failures.** In the moving phase, 40% of frames have mask IoU < 0.5. I looked at frames such as 270,
  280, 310, 340–410, 440, 480 and 600. The errors include end-to-end flips (for example 310, 340–400), lateral
  offsets of several centimetres, lag at pickup (frame 120), and jumps of tens of degrees per frame (p95 rotation
  step 36°/frame).
- **The SAM2 object masks often miss the dark handle.** Examples are 455–460, 653–658 and 262–264: only the barrel
  is masked, even though the handle is clearly visible. A naive silhouette fit then explains the truncated mask
  with a falsely foreshortened bat.

## 2. Method (what `solve.py` does)

1. **Mask repair.** Saturated-blue pixels (HSV hue 95–125, S ≥ 60) that are not on the person and are connected to
   the SAM2 mask are added. Connections through person pixels within 50 px of the mask are allowed, because the
   gripping hand separates barrel and handle. I checked this on about 15 frames: it recovers the handle in 456,
   264, 345 and 410, and partially in 655. A few small spurious blobs (signs, tripod parts) remain; they penalise
   all hypotheses equally.
2. **12 hypotheses per frame, each refined.** The starting poses are the baseline, FoundationPose and CoCoNet. The
   latter two are converted from `K_pipeline` to `K` and depth-scaled to the baseline's median. Each comes in four
   variants: as is, flipped end-to-end (180° about mesh x), mirrored out of plane (axis reflected about the plane
   perpendicular to the viewing ray: same 2-D image, opposite tilt), and both.
   - Each hypothesis is refined by render-and-compare. I used nvdiffrast on a 6000-face decimated copy of the mesh,
     with a 192×192 crop, and optimised a soft IoU plus a distance-transform pull. Person pixels count as "don't
     care". The fit runs 80 Adam steps, all hypotheses batched on the GPU.
   - Colour is not used inside the fit: in a test it dragged the poses off the silhouette.
3. **Evidence per hypothesis:**
   - silhouette IoU loss plus distance term;
   - colour agreement with a per-channel gain (separates the light barrel from the dark handle, i.e. end flips);
   - a **MoGe-2 relative depth profile**: the median absolute residual of (MoGe depth − rendered depth) after
     removing the median offset, on eroded object pixels. This resolves the out-of-plane mirror ambiguity, which the
     silhouette cannot see.
   - a **floor term**: a penalty only if the bat would penetrate the released floor plane by more than 10 cm at the
     baseline depth. This rejects impossible mirror hypotheses, e.g. one at static-2 would stick 28 cm into the
     floor.
4. **Static phases.**
   - All candidates from 12 sampled frames are scored on all sampled frames (pooled evidence). The best one is
     refined per frame and averaged into one constant pose.
   - Automatic fallback: the baseline's phase pose, carrying our decoded roll, is also scored. It is adopted if it is
     better on both pooled checks (silhouette and depth profile).
     - static-1: ours is better on silhouette (0.192 vs 0.223), so ours is kept.
     - static-2: the baseline wins on both (silhouette 0.254 vs 0.273, depth profile 0.9 vs 1.6 cm), so the
       baseline's axis and position are used with our roll.
5. **Viterbi decoding** of the moving phase over the candidates.
   - Transition cost: Huber on the axis-direction change, the translation change (with depth down-weighted) and the
     roll change.
   - Boundary conditions: continuity with the two static poses.
   - A propagation pass then re-fits each frame from its decoded neighbours (i±1), adds these as candidates, and
     decodes again.
6. **Roll decoding.** For 36 roll angles per frame, relative to a minimal-twist (parallel-transported) reference
   frame, the full textured mesh is rendered and scored on two things:
   - texture-luminance correlation with the image;
   - silhouette IoU (the scanned bat is slightly bent, so roll changes its outline by up to 3% IoU).

   A DP with a quadratic roll-change cost and constant roll inside static phases picks the sequence.
7. **Joint refinement.** All moving-phase poses are optimised together. The objective is the per-frame silhouette
   terms plus SE(3) second-difference smoothness (translation σ = 4 mm/frame², rotation σ = 0.03) plus a penalty on
   the roll rate. Frames with small visible masks get less image weight. Static poses are held fixed.
8. **Depth re-anchoring (keeps the baseline's placement).**
   - The bat centre (mesh point (0,0,0.0616)) is moved along its viewing ray, so the image position and rotation
     stay unchanged.
   - Its camera depth is set to the baseline's bat-centre depth: median-9 plus Gaussian-σ2 smoothed in the moving
     phase, and the phase median in the static phases.
   - The camera-frame poses are then mapped to the world with `T_world_cam`.

Why per-frame smoothed baseline depth rather than one scale per phase: my silhouette depth is 4–10% closer than the
baseline (consistent with slightly bloated masks) and noisy (p90 19 cm). The independent hand-contact check favoured
the baseline's depth profile. Median distance from the bat axis to the closest reconstructed hand keypoint while
carried:

| depth source | median | p90 |
|---|---|---|
| phase-constant scale | 4.4 cm | 21 cm |
| windowed scale | 3.1–3.7 cm | — |
| smoothed baseline depth (chosen) | 2.4 cm | 6.6 cm |

## 3. Validation (no ground truth)

Columns:
- `iou_*`: full-resolution mask IoU (full textured mesh, original SAM2 masks, person pixels not on the object
  ignored). `iou_lt05` is the fraction of frames with IoU < 0.5.
- `iou_ext`: IoU against the handle-extended masks.
- `depthprof_cm`: MoGe-2 relative depth-profile residual.
- `hand_cm`: median distance from the closest hand keypoint to the bat axis (meaningful while carried).
- `floor_min_cm`: lowest point relative to the floor plane (negative = below).
- `rot_step`: rotation change per frame (°).
- `acc_mm`: median second difference of the bat-centre position.
- `dz_*`: camera-frame depth difference of the bat centre to the baseline (cm).

| phase | method | frames | iou_mean | iou_med | iou_lt05 | iou_ext | depthprof_cm | hand_cm | floor_min_cm | rot_step_med | rot_step_p95 | acc_mm | dz_mean_cm | dz_med_cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static-1 | baseline | 0-110 | 0.785 | 0.783 | 0.000 | 0.777 | 0.442 | 104.6 | -4.710 | 0.000 | 11.5 | 0.589 | 0.000 | 0.000 |
| static-1 | final | 0-110 | 0.814 | 0.820 | 0.000 | 0.809 | 0.522 | 103.8 | -3.200 | 0.000 | 0.000 | 0.000 | 0.175 | 0.000 |
| moving | baseline | 111-705 | 0.533 | 0.600 | 0.403 | 0.507 | 2.535 | 2.423 | -14.2 | 8.718 | 36.3 | 26.0 | 0.000 | 0.000 |
| moving | decoded (before joint) | 111-705 | 0.744 | 0.774 | 0.040 | 0.707 | 2.583 | 1.894 | -6.364 | 5.759 | 13.9 | 7.961 | -0.057 | 0.018 |
| moving | final | 111-705 | 0.732 | 0.761 | 0.040 | 0.693 | 2.561 | 1.895 | -6.650 | 5.047 | 10.8 | 2.669 | -0.057 | 0.018 |
| static-2 | baseline | 706-802 | 0.767 | 0.769 | 0.000 | 0.753 | 0.866 | 69.1 | -7.556 | 0.000 | 2.820 | 0.517 | 0.000 | 0.000 |
| static-2 | final | 706-802 | 0.763 | 0.761 | 0.000 | 0.749 | 0.834 | 69.1 | -6.487 | 0.000 | 0.000 | 0.000 | 0.119 | 0.000 |
| all | baseline | 0-802 | 0.596 | 0.705 | 0.299 | 0.574 | 1.702 | 3.952 | -14.2 | 6.279 | 30.0 | 19.2 | 0.000 | 0.000 |
| all | final | 0-802 | 0.747 | 0.770 | 0.030 | 0.716 | 1.772 | 2.970 | -6.650 | 3.950 | 10.3 | 1.940 | -0.004 | 0.008 |

**Camera-frame depth difference to the baseline (bat centre)**, mean / median:

| phase | mean | median |
|---|---|---|
| static-1 | +0.18 cm | 0.00 cm |
| moving | −0.06 cm | +0.02 cm |
| static-2 | +0.12 cm | 0.00 cm |

The baseline is not constant within its "static" phases, which is why the means are not exactly 0.

**How much the result departs from the baseline** (mean point distance under the two poses, same measure as ADD):

| phase | mean point distance | axis direction | notes |
|---|---|---|---|
| static-1 | 2.7 cm | 4° | |
| moving | 8.6 cm (median 6.0) | 9° median | 65 frames have the opposite end assignment to the baseline; 174 frames differ by > 30° in axis direction |
| static-2 | 3.9 cm | 0° | almost entirely roll |

**Visual checks.** I rendered overlays (green = our render contour, magenta = SAM2 mask; dots mark the barrel and
knob ends) for about 60 frames across all phases. End assignment matches the light and dark parts of the bat in
every checked frame. Frames where the barrel is hidden behind arms or head (270–280, 350–395) look plausible, but
cannot be confirmed from the image.

Smoothing trade-off: the joint refinement lowers moving-phase IoU slightly (0.744 → 0.732). In exchange it removes
jitter: p95 rotation step 13.9° → 10.8°, translation acceleration 8.0 → 2.7 mm. I kept it because per-frame IoU
rewards fitting mask noise, and a rigid object held by hands moves smoothly.

## 4. Depth (along-ray) correction: applied shift = 0

Each cue below gives the depth shift it would suggest (negative = closer to the camera), computed by
`depth_cues.py`:

| phase | silhouette size (my fit) | floor contact (lowest point on plane) | hand contact (axis to hand keypoints) |
|---|---|---|---|
| static-1 | −14.6 cm | −6.0 cm | – |
| moving | −17.1 cm | – | −5.0 cm |
| static-2 | −25.1 cm | −13.5 cm | – |

Within every phase the two available cues disagree by 9–12 cm, far more than 2 cm. The silhouette cue is also known
to be biased: a 1-px mask bloat on a 9 cm-wide object is several percent of its size. The floor plane has
several-cm tolerance (feet sit about 4 cm from it). So I kept the baseline depth: **the applied shift is 0 in all
phases.** The floor and hand cues both hint that the object might be ~5 cm closer in static-1 and the moving phase.
That is not strong enough to act on under the rule.

## 5. Confidence per phase

- **static-1 (0–110): high** for position and axis. The result is within 4° and 2–3 cm of the baseline, and image
  IoU went 0.785 → 0.814. Roll is medium: texture prefers our roll, which is also within ~10° of the baseline's.
- **moving (111–705):**
  - The **end assignment and axis direction are clearly better than the baseline**: IoU < 0.5 frames went 40% → 4%,
    the hand–bat distance 2.4 → 1.9 cm, the worst floor penetration −14 → −6.6 cm, and there are no 180° jumps.
    Confidence is high where the bat is fully visible (≈160–330, 410–650).
  - Confidence is medium during the back-facing segment (330–410) and end-on views (≈480, 640–670). There the mask
    is small, and the out-of-plane tilt sign rests on the MoGe depth-profile and temporal continuity.
  - **Roll is low confidence** throughout. It comes from weak texture and silhouette evidence plus continuity,
    though it is now smooth instead of random.
  - Depth follows the smoothed baseline, so its accuracy is the baseline's.
- **static-2 (706–802):** axis and position are high confidence: they are the baseline's, chosen automatically
  because they explain the images better than my pooled fit. **Roll is low-to-medium:** we differ from the baseline
  by about 140° in roll.
  - In favour of our roll: both pooled cues (texture correlation and full-resolution silhouette IoU) prefer it over
    the baseline's roll, and it is continuous with strong texture evidence in end-on views just before the put-down.
  - Against it: the individual cues have different optima, and the texture likelihood has a near-180° secondary
    minimum.

  If the baseline's roll is right, static-2 frames carry about 4–5 cm of ADD error from roll alone.

## 6. Run time and resources

- `solve.py`: 337 s wall on a GPU shared with other jobs. That breaks down as about 85 s video decoding and crops,
  151 s for the per-frame fits of 803×12 hypotheses, about 50 s for decoding, roll and static checks, and 40 s for
  joint refinement.
- Built-in validation adds 265 s.
- This process used ≤ 2 GB of GPU memory.
- Total GPU time over the whole project, including experiments and three full runs, was about 37–40 min.

## 7. Limitations and what I did not do

- I did not re-segment the bat with SAM2 or track texture with a point tracker or optical flow. The colour-based
  mask repair is heuristic and only partially recovers the handle when it is far from the mask or blurred (e.g.
  655).
- I did not model rigid hand–object attachment during carrying: hand keypoints are used only for validation and the
  depth-cue analysis.
- Depth stays at the baseline's (smoothed) value by design. Any constant depth bias in the baseline is inherited.
