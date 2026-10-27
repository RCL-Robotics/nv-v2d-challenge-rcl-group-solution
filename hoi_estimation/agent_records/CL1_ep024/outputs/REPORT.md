# Episode 24 – watering can: trajectory-level object pose recovery

Output: `object_poses_world.npz` (`T_world_mesh`, 748×4×4 float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(log in `solve_log.txt`, 105 s wall-clock including about 70 s on the GPU, peak GPU memory under 2 GB).
Validation: `python validate.py object_poses_world.npz solve_diagnostics.npz` writes the table below
(`validation_table.md`).

## What I found when diagnosing the baseline
1. **The baseline is CoCoNet plus a per-frame depth correction.** Its rotation equals CoCoNet's (a constant 1.3° offset)
   and differs from FoundationPose by 5–10°. In the two static phases its rotation wanders by up to 5° and its depth
   jitters with σ = 2–3 cm, even though the can does not move.
2. **The SAM2 masks are blobby** (dilated by 1–2 px; they fill in the hole in the handle). The can is saturated green, so a
   colour map (Lab a*) gives a sharper, sub-pixel silhouette that shows the handle hole. I use this map, not the SAM
   masks, as image evidence.
3. **Silhouette size vs. metric depth.** At the baseline's depth the mesh renders about 9% smaller than the can in the
   image. A free 6-DoF fit pooled over a whole static phase moves the can to z ≈ 2.72 m instead of 2.96 m. At that depth
   it would float 10 cm above the floor, which is impossible for a can standing on the ground. The factor is the same in
   both static phases (1.089–1.092) and in the held phase (1.06–1.10). The reference human projects onto the person mask
   well, and an error in K would scale the human and the object alike. So this is a size/calibration mismatch of the
   object, not a depth signal. Consequence: **absolute depth is not taken from the silhouette.** Because
   (R, t) with the mesh scaled by κ projects exactly like (R, κt), the image fit is done "at silhouette scale" and only
   rotation and the object's viewing ray are used. Depth comes from the world-frame cues.
4. **Near mirror-symmetric silhouette.** The can has a mirror plane (normal ≈ (0.01, 0.98, −0.21) in mesh
   coordinates, residual 1.2 mm), so every pose has a mirror twin with almost the same silhouette. The silhouette alone
   slightly prefers the twin. The twin is physically wrong:
   - in the static frames it shows the can from below the floor;
   - in the held frames it would show the white barcode on the base, which never appears in the video.

   The baseline is on the correct branch, so I initialise from it and enforce temporal continuity from the static anchors.
5. **Occluders.** In the held phase the person's leg is *behind* the can. Treating all person pixels as possible
   occluders, as I did at first, let the fit bulge into the leg; this caused 10–20° errors. Only person pixels inside
   the dilated convex hull of the projected hand keypoints are treated as occluders. The hand keypoints project
   accurately onto the gripping hand.

## Phases (detected automatically from the video, in `solve.py`)
The criterion is the IoU of each frame's green map with the first or last frame (5-frame mean ≥ 0.93).

| phase | frames | content |
|---|---|---|
| A static | 0–132 | can stands on the floor; hand touches the handle from about 115 but the can does not move |
| moving | 133–597 | grasp and lift (133–200); held tipped forward "pouring", slowly moving (200–510); put down (510–597) |
| B static | 598–747 | can back on the floor (hand releases at about 630) |

## Method (`solve.py`)
1. **Evidence.**
   - For every frame, a 160×160 crop around the SAM2 object mask, holding the green-probability map plus an occluder
     mask (hand region only).
   - Differentiable silhouette rendering with nvdiffrast, on an 8k-face quadric-decimated copy of the exact mesh, at 2×
     crop resolution with antialiasing.
   - Loss: squared difference to the green map over non-occluded pixels.
2. **Static phases.** One shared 6-DoF pose per phase, fitted over every 3rd frame of the phase (500 Adam steps,
   rotation plus image position and log-depth parametrisation). Fits from two different initialisations converged to
   the same pose. Depth along the fitted ray is the mean of two cues:
   - (a) floor contact: lowest mesh point 1 cm below the released plane, because the true frame-0 human's soles reach
     −1 to −2.5 cm;
   - (b) the baseline's median depth over the phase.

   They agree to 0.7 cm in phase A (2.952 vs. 2.959 m) and 2.9 cm in phase B (2.973 vs. 3.002 m).
3. **Moving phase.**
   - Joint optimisation of all 465 poses: silhouette term plus second-order smoothness on rotation (camera-frame
     increments) and translation.
   - Anchored with zero velocity to the static poses at both ends; initialised from the baseline, with its depth
     divided by κ.
   - The rotation and the object's image position are kept from this fit.
4. **Depth in the moving phase.** Robust (Huber) second-order smoothing of the baseline camera depth (calibrated by the
   pipeline against the human body and floor), with the static values held fixed so the depth is continuous at the
   phase boundaries. The silhouette depth, rescaled by the κ calibrated on the static phases, agrees with the baseline
   depth to about 3% in the held phase, so I kept the better-calibrated baseline value and only removed its jitter.
5. Output: `T_world = T_world_cam · T_cam`.

## Validation (no ground truth; `validation_table.md`)
"Scale-free IoU" renders each pose with translation / κ, which is equivalent to scaling the mesh by κ. It measures
whether rotation and image position explain the video, independent of the depth convention. "As-is" uses the metric
pose unchanged. Held-out: the static fits only saw frames with i % 3 = 0.

| metric | A base | A ours | moving base | moving ours | B base | B ours | all base | all ours |
|---|---|---|---|---|---|---|---|---|
| silhouette IoU, scale-free (mean) | 0.770 | **0.934** | 0.790 | **0.914** | 0.745 | **0.926** | 0.777 | **0.920** |
| silhouette IoU, scale-free (10th pct) | 0.734 | 0.928 | 0.705 | 0.897 | 0.680 | 0.916 | 0.706 | 0.901 |
| silhouette IoU, as-is metric depth (mean) | 0.760 | 0.785 | 0.701 | 0.811 | 0.728 | 0.798 | 0.717 | 0.804 |
| scale-free IoU, held-out frames | 0.770 | 0.934 | 0.789 | 0.914 | 0.745 | 0.926 | 0.777 | 0.920 |
| frame-to-frame rotation, median / 99th pct (deg) | 1.23 / 3.56 | 0 / 0 | 1.72 / 6.82 | 0.52 / 3.00 | 1.16 / 4.71 | 0 / 0 | 1.47 / 5.96 | 0.24 / 2.85 |
| frame-to-frame translation, median (cm) | 1.12 | 0 | 1.28 | 0.15 | 1.02 | 0 | 1.20 | 0.07 |
| tilt of the can's base vs. floor normal (deg; should be ≈0 when resting) | 1.9 | **0.5** | – | – | 1.8 | **0.5** | | |
| lowest mesh point above plane, min (cm; penetration) | −2.7 | −1.2 | −2.9 | −1.5 | −4.1 | −1.6 | −4.1 | −1.6 |
| holding-hand keypoints to mesh, median (cm; contact) | – | – | 3.0 | 2.8 | – | – | | |

On frames with contact probability > 0.5, the hand-to-mesh distance is 3.3 cm for the baseline and 3.1 cm for ours.
Mean point distance between our result and the baseline is 2.35 cm (A), 2.38 cm (moving) and 3.40 cm (B).
Rotation changes are 4.6° (A) and 1.7° (B) against the baseline's phase-mean pose, and 2–13° in the held phase.

Run-to-run reproducibility: nvdiffrast's backward pass uses atomic adds, so repeated runs differ slightly. Two full runs
differed by ≤ 0.32 cm in mean mesh-point distance (≤ 0.15 cm in the static phases). Seeds are fixed.

## Confidence per phase
- **A static (0–132): high.**
  - The rotation fits the silhouette, including the handle hole, on held-out frames (IoU 0.93 vs. 0.77).
  - It stands flat on the floor (0.5° tilt), which is an independent physical check.
  - Depth comes from two agreeing cues (0.7 cm apart).
  - The remaining risk is a global depth/scale error shared with the baseline (see "Open issue" below).
- **B static (598–747): high.** Same evidence; the floor and baseline depth cues are 2.9 cm apart, and I used their
  mean.
- **Moving (133–597): medium-high.**
  - Rotation is clearly better supported by the image (IoU 0.91 vs. 0.79, worst frames ≥ 0.875), it is smooth, and it
    is consistent with hand contact.
  - Rotation about the flat direction of the silhouette loss is uncertain by a few degrees; run-to-run differences
    reach 2.8° there.
  - Depth is the smoothed baseline depth, so any slow bias in the baseline's depth during the hold remains.
- **Open issue: the 9% silhouette-size mismatch.** I treated it as a size/calibration discrepancy and kept the
  floor- and human-consistent depth, which the baseline also uses. If the evaluator's ground truth instead followed the
  silhouette, the can would be about 25 cm closer to the camera. I rejected that because the can would then float
  10 cm above the floor in both static phases.

## What was not done
- No new FoundationPose/CoCoNet runs.
- No optical-flow or point-tracking term: the can is nearly textureless uniform green, about 50 px across.
- No photometric term: it was used only for the manual barcode check in the mirror analysis.
- Hand–object rigid attachment was not enforced explicitly; it is checked through the hand-to-mesh distance only.
