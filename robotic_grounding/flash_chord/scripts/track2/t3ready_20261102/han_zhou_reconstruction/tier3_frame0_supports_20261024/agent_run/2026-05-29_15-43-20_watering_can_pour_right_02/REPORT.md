# Episode 24 – watering can: object trajectory report

Task: "walk over to the watering can on the ground, kneel, tip it forward as if pouring using right hand".
Output: `object_poses_world.npz` (`T_world_mesh`, 748×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`.

## 1. Phases found
Phases are found automatically in `solve.py`. Each frame's SAM2 object mask is compared (IoU, ignoring pixels covered by the person) with the median mask of the first and of the last 30 frames. A frame counts as unmoved while that IoU is at least 0.94. I checked the boundaries by eye on zoomed crops.

| phase | frames | what happens |
|---|---|---|
| S1 static | 0–132 | can stands on the floor. The hand reaches it at about 100 and grips the handle from about 110, but the can does not move until about 133 |
| lift (carried) | 133–229 | lifted and turned into pouring position |
| pour (carried) | 230–499 | held in the right hand, spout down, small motions |
| put-down (carried) | 500–595 | turned back upright and set down |
| S2 static | 596–747 | on the floor in a new pose. The hand lets go at about 635 and the person walks away at about 670 |

(The lift / pour / put-down split is used only for reporting; the solver treats 133–595 as one carried phase.)

## 2. Diagnosis of the baseline
* **No flips.** The can's mirror symmetry would only cause a confusion that is a rotation of more than 90°. The baseline is smooth in rotation, agrees with FoundationPose to within about 7°, and stands upright on the floor in both static phases. I found no 180° or depth-flip errors and no tracking loss.
* **Jitter.** The baseline's camera depth jitters frame to frame:
  * standard deviation 2.2 cm in S1 and 3.1 cm in S2, even though the can does not move;
  * median frame-to-frame translation step 1.2 cm and rotation step 1.2–2.4°;
  * slow drift of about 5 cm inside S2, when the hand releases the can.
* **Image-position offset.** Every frame of the baseline is shifted about 3–4 px in the image (left and down) relative to the can, which is about 1.5 cm sideways. This is confirmed by two independent masks: SAM2, and a colour (HSV green) threshold. The SAM2 masks are trustworthy for position because the person mask at frame 0 is centred on the projected *true* frame-0 human to within 0.5 px. The offset seems to come from the CoCoNet refinement: raw FoundationPose sits closer to the mask.
* **Rotation errors of a few degrees.** These are visible in the lift and put-down; for example, at frames 502–516 and 540–560 the baseline spout clearly misses the image spout.
* **Masks are too large.** SAM2 masks here are about 2–3 px too large on every side. The person mask at frame 0 is 4.5% larger than the projected true human, but centred on it. On a 60–70 px object this makes a naïve silhouette fit pull the can 15–40 cm toward the camera. The amount depends on the mask threshold, so it is clearly an artefact. I model it by dilating the rendered silhouette by 2 px. With that, a pooled static fit lands on the baseline depth to within 0–2 cm.

## 3. Method (`solve.py`)
1. **Silhouette term.** The mesh (decimated to 20k faces) is rendered with nvdiffrast, differentiable and anti-aliased, then softly dilated by 2 px. Loss is 1 − soft IoU with the SAM2 object mask; pixels covered by the person (hand or leg occluding the can) are ignored.
2. **Static phases.** One shared 6-DoF pose per phase, fitted to all of its frames at once (133 and 152 frames). Initialised from the chordal mean of the baseline.
3. **Carried phase.** All 463 poses are optimised jointly:
   * silhouette term;
   * second-order smoothness on rotation and translation, with the two static poses fixed as boundary conditions so the trajectory joins them without jumps;
   * a weak prior toward the baseline, which keeps the solution unique along directions the silhouette cannot see.

   Initialised from the baseline. Smoothness weights were chosen by IoU on held-out (odd) frames.
4. **Depth along the viewing ray is not taken from the image.** It is decoded from the baseline:
   * in static phases, constant = the baseline's median camera z over the phase;
   * in the carried phase, a robust (Huber-IRLS) second-difference-smoothed baseline z, continuous with the static constants.

   Every fitted pose is then moved along its own viewing ray to that depth, so its image position and rotation are kept. This also does the requested re-anchoring: median camera z equals the baseline's in every phase.
5. Map back to the world with `T_world_cam`.

What changed compared with the baseline:
* rotations: median change 1.6°, up to about 10° in the transitions;
* image position: median sideways change 1.4 cm;
* depth: only the jitter is removed (static phases constant, carried phase smoothed); its level is unchanged.

Things I tried and rejected:
* **Silhouette-derived depth.** Biased by mask bloat, and drifts about 6 cm inside S2 just like the baseline.
* **Texture/edge correlation score.** Too noisy at this object size; its preferred scale disagrees with the silhouette.
* **Fixing depth while fitting without the bloat model.** It tilts the can to fake the extra size, making the floor tilt worse (6°).

## 4. Depth: physical cues (applied shift = **0 cm**)
Each cue is given as the suggested shift along the viewing ray (− means closer to the camera), computed on the final poses.

| cue | S1 | S2 | comment |
|---|---|---|---|
| released floor plane (lowest point of the can on the plane) | −3.0 cm | −4.2 cm | the plane is a fit with several cm of tolerance; the true frame-0 human penetrates it by 1.5–2.5 cm |
| local floor from the true frame-0 human's lowest vertices | +0.5 cm | −0.7 cm | uses the reference human only as a floor height |
| right-hand keypoints touching the handle (contact frames) | −7.7 cm (whole episode) | | not independent: CARI4D optimised hand and object together; at the baseline depth the fingers are already within 1.0 cm of the handle |
| metric mesh size vs. image size (silhouette with the 2 px bloat model) | −0.3 cm | −1.8 cm | without the bloat model this cue says −24 cm, which is an artefact (see §2) |

The cues do not agree within 2 cm on a correction larger than their disagreement, so I kept the baseline depth.
Camera-frame depth difference, result − baseline, per phase (cm, mean / median):
* S1: +0.31 / 0.00
* lift: −0.21 / 0.00
* pour: −0.19 / +0.12
* put-down: +0.25 / −0.02
* S2: −0.07 / 0.00
* whole episode: −0.03 / +0.05

## 5. Validation (no ground truth)
* **IoU:** the rendered silhouette (2 px soft dilation) against the SAM2 mask, ignoring person pixels.
* **Held-out IoU:** from a separate run of `solve.py --holdout 1`, where odd frames get no image term; the score is on the odd frames.
* **Step:** median frame-to-frame change.
* **Hand–handle:** median distance from the 5 nearest right-hand keypoints to the mesh in contact frames.

| phase | frames | IoU base | IoU ours | held-out IoU ours | rot step base/ours (°) | trans step base/ours (cm) | rot change vs base (°, median) | lateral shift (cm, median) | dz mean / median (cm) | mean point distance ours vs base (cm) | hand–handle base/ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 static | 0-132 | 0.837 | 0.943 | 0.945 | 1.23 / 0.00 | 1.12 / 0.00 | 1.7 | 0.95 | +0.31 / +0.00 | 2.20 | 3.0 / 2.2 |
| lift | 133-229 | 0.795 | 0.938 | 0.934 | 2.36 / 1.33 | 1.28 / 0.35 | 1.6 | 1.35 | -0.21 / +0.00 | 2.49 | 0.5 / 0.6 |
| pour | 230-499 | 0.745 | 0.904 | 0.899 | 1.40 / 0.42 | 1.33 / 0.13 | 1.4 | 1.44 | -0.19 / +0.12 | 2.29 | 1.3 / 1.3 |
| put-down | 500-595 | 0.730 | 0.920 | 0.916 | 2.30 / 1.44 | 1.22 / 0.36 | 1.5 | 1.96 | +0.25 / -0.02 | 2.75 | 2.5 / 3.1 |
| S2 static | 596-747 | 0.790 | 0.931 | 0.928 | 1.16 / 0.00 | 1.01 / 0.00 | 2.1 | 1.48 | -0.07 / +0.00 | 3.35 | 3.7 / 6.2 |
| ALL | 0-747 | 0.775 | 0.923 | 0.920 | 1.47 / 0.32 | 1.20 / 0.10 | 1.6 | 1.42 | -0.03 / +0.05 | 2.58 | 1.5 / 1.5 |

Other checks:
* **Held-out frames:** IoU on odd frames is 0.920 for my result versus 0.776 for the baseline. Before the depth re-anchoring the held-out run scores 0.925, so the re-anchoring costs almost nothing in the image.
* **Floor contact in the static phases:**
  * tilt of the can's base against the floor normal: S1 0.68° (baseline mean pose 0.56°), S2 1.64° (baseline 1.49°);
  * lowest point relative to the plane: S1 −1.4 cm, S2 −2.3 cm (baseline per-frame medians −0.8 and −1.6 cm, jittering with a standard deviation of 1.2–1.3 cm). Both are within the plane's tolerance.
* **Hand–handle distance** is only a weak check, because the baseline hand was optimised together with the baseline object. It is unchanged overall. It gets worse in S2 (3.7 → 6.2 cm): while the hand still holds the resting can (596–635), the baseline depth was pulled toward the hand. Once the hand leaves, the baseline depth drifts back by about 5 cm, but the can does not move.
* **Visual checks** of contour overlays every 7 frames through the lift and put-down: the result follows the spout and body where the baseline is visibly off.
* **Reproducibility:** nvdiffrast's backward pass accumulates gradients with GPU atomic adds, so runs are not bit-identical. Three runs of the final script differ by 0.35° rotation mean (1.3° max) and 0.09 cm translation mean (0.4 cm max). That is well below the size of the corrections. Seeds are fixed, and there is no randomness otherwise.

## 6. Confidence
* **Static phases (S1, S2): high** for the main gain, which is replacing 2–3 cm of per-frame jitter and drift with one pose (the image evidence of more than 130 frames is pooled). The about 1.5 cm sideways correction is confirmed by two independent masks. **Medium** for the 1.7–2.1° rotation change, which is within the uncertainty of the floor-tilt check.
* **Carried phase: medium-high.** Image alignment improves a lot (held-out IoU 0.74–0.80 → 0.90–0.93), the trajectory is smooth and physically plausible, and the larger rotation fixes in the lift and put-down are visible by eye. The pour phase has the most hand occlusion and the lowest IoU (0.90).
* **Absolute depth: unchanged from the baseline (by design).** Any constant error the baseline has along the viewing ray is still in my result. The cues in §4 do not justify a correction.

## 7. Run time / resources
* `solve.py`: about 85 s wall-clock on one RTX 6000 Ada, GPU memory under 3 GB. Static fits about 30–45 s, carried joint fit about 30 s, the rest under 10 s.
* Total GPU time over the whole investigation, all experiments included: roughly 30 minutes.
* `solve.py` reads only `/work/inputs`: the masks, mesh, camera and baseline (frame0-human, video and ground plane are used only in analysis). It uses no `/pipeline` or `/weights` files and no network.

Hard-coded constants are listed and explained at the top of `solve.py`: the static-IoU threshold 0.94, the 2 px mask bloat, the smoothness and prior weights, and depth shift 0.
