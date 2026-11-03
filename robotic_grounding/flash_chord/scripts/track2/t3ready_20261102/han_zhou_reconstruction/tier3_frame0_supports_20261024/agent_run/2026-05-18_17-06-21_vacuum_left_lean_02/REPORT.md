# Vacuum episode: object trajectory report

**Result:** `object_poses_world.npz` (`T_world_mesh`, 747×4×4, float64). It is produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(deterministic, offline, 196 s wall-clock for the final run; log in `solve_log.txt`).

I changed rotations and image-plane position only. Depth is kept on the baseline: per-phase median camera-frame depth equals the baseline's, and the applied depth correction is 0.

## Phases
These are found automatically by comparing each frame's object mask with the first and last frame (person pixels excluded), and checked by eye.

| phase | frames | what happens |
|---|---|---|
| static_start | 0–124 | Vacuum leans on the bar stool, untouched until about frame 100. The hand grasps it around 105–124, with no visible motion (mask IoU to frame 0 ≥ 0.95). |
| carried | 125–656 | Picked up, carried, side-to-side vacuuming, put back. Frames 612–656: already back against the stool but still held. Around frames 240–300 the person turns the vacuum so its back (grey body, black cord loops) faces the camera. |
| static_end | 657–746 | Leaning on the stool again, untouched. This pose differs from the starting one. |

## Diagnosis of the baseline
- **Static phases:** the per-frame baseline rotation wanders by up to 11° (static_start) although the object does not move. Its depth z varies by about 10 cm.
- **Carried phase:** rotation jumps (median 3.5°/frame, 95th percentile 17.6°/frame).
- **Wrong side shown at frames ~244–268:** the baseline shows the red front while the video shows the back, i.e. it is 180° wrong about the long axis. FoundationPose is about 150–180° away from the baseline around frames 270–310.
- **Image position:** mask IoU of the baseline is low throughout (0.30–0.45).

## Method (`solve.py`)
1. **Occluders.** These are excluded from all image terms: the dilated person mask, plus the bar stool. The stool is segmented once with SAM2 using a constant box chosen by eye; it is static, with IoU 0.94 between frame 0 and frame 746.
2. **Evidence per frame.** Render the textured mesh with nvdiffrast into a crop around the object, then compute:
   - occlusion-aware silhouette IoU plus a distance-transform term;
   - an appearance distance on blurred "redness" and "brightness" maps. This separates the red front from the grey back. An earlier per-channel normalised colour score could not, and was dropped.
3. **Mask-size bias.** A free-depth fit of each static phase wants the object about 10% closer (scale 0.904 and 0.902). The SAM2 person mask is 3–5% larger than the projection of the true frame-0 human, so the masks are bloated. I therefore fit silhouettes with one global factor of 0.903 along the viewing rays and discard it afterwards. Without it, fitted rotations differed by about 4° on average.
4. **Static phases: one pose each.** The pose is fitted jointly over every 3rd frame of the phase, with depth fixed at the baseline median. Starting points: the mean baseline rotation, rolls about the long axis (0, ±30, 90, 180, 270°) and 180° flips about the other axes. The best pooled score wins:
   - static_start: roll −30° start, result 13° from the baseline;
   - static_end: 0° start, result 8.7° from the baseline;
   - the flipped alternatives score clearly worse (pooled score 0.6–1.3 vs 0.45–0.49).
5. **Carried phase: hypotheses.** Eight per frame: the baseline rotation and the FoundationPose rotation, each rolled by 0/90/180/270° about the long axis. Each is refined on the silhouette with depth fixed at the temporally smoothed baseline z. Score = (1 − IoU) + 2·dt + appearance.
6. **Viterbi** over the hypotheses, with costs for rotation change beyond 3°/frame and for translation change. Result: 446 of 532 frames keep the baseline or FoundationPose orientation; the rest take rolled hypotheses, concentrated at 241–268 and 300–320.
7. **Joint trajectory refinement** of all carried frames (150 Adam iterations):
   - silhouette terms, weighted down for tiny masks;
   - second-difference smoothness on rotation (chordal) and position;
   - anchored at both ends to the static poses.
8. **Depth re-anchoring.** Per phase, camera-frame positions are scaled along the rays so that median z equals the baseline median z. The factors were 1.000, 1.003 and 1.000.

## Validation (no ground truth)
Run with `validate.py`; table also in `validation_table.md`. This table excludes only the person, not the stool, for both trajectories, so IoUs are lower than inside `solve.py`. "IoU (scale 0.903)" accounts for the mask bloat.

| phase | IoU base / ours (scale 1) | IoU base / ours (scale 0.903) | appearance dist base / ours | rot step med / p95 °/frame base | ours | centre accel median cm base / ours | lowest point vs floor plane cm base / ours | depth ours−base mean / median cm | rot change vs base median / p90 ° |
|---|---|---|---|---|---|---|---|---|---|
| static_start 0–124 | 0.451 / 0.447 | 0.395 / 0.550 | 0.247 / 0.177 | 0.07 / 5.6 | 0 / 0 | 0.14 / 0 | −3.1 / −5.6 | −0.82 / 0.00 | 13.2 / 16.4 |
| carried 125–656 | 0.311 / 0.395 | 0.299 / 0.470 | 0.235 / 0.195 | 3.47 / 17.6 | 1.86 / 7.6 | 3.43 / 0.21 | −5.0 / −7.4 | 0.84 / 0.84 | 21.7 / 68.5 |
| static_end 657–746 | 0.354 / 0.324 | 0.323 / 0.429 | 0.206 / 0.150 | 0.12 / 0.1 | 0 / 0 | 0.02 / 0 | −2.0 / −4.6 | 0.93 / 0.00 | 8.7 / 8.7 |

- **Held-out test of the carried trajectory.** I ran solve.py with `--holdout 5`, which removes the image term on every 5th carried frame (107 frames), so their poses come only from the neighbours and the smoothness prior. On those frames IoU is 0.39 for the baseline and 0.65 for ours; the score U is 0.87 for the baseline and 0.55 for ours. This run used an earlier static-pose variant; the carried part was identical. The result is insensitive to the smoothness weight (λ = 2, 5, 15 give 0.582, 0.582, 0.581 in an earlier ablation).
- **Visual checks of overlays.** In my result:
  - frames 250–290 now show the back of the vacuum, as in the video;
  - frame 300 has the correct strong tilt (the baseline is upright there);
  - static poses fit the foot and cord silhouette better.
- **Remaining visual doubt:** around frame 316, during the turn back to the front, my result still shows the back side.
- **Scale-1 IoU in static_end is lower for ours (0.354 vs 0.324).** The fit is made at the bloat-corrected scale, so at scale 1 the rendered object is slightly too small for the bloated mask.

## Depth (kept on the baseline; applied shift = 0 cm)
Suggested shifts along the viewing ray (negative = closer to the camera), from `depth_cues.py`:

| cue | suggested shift |
|---|---|
| Floor contact: lowest point onto the true-human sole level (≈ 0 cm from the plane) | baseline poses −4 / −3 cm; our static poses −10 / −9 cm |
| Mesh size vs mask size | −30 cm uncorrected; −16 to −21 cm after the measured 3–5% mask bloat |
| Hand contact: left hand vs handle top, carried frames where they are closest | −3 cm, but the 3D distance is 6 cm and the right hand is inconsistent |

The cues disagree by far more than 2 cm, so I applied none of them. Mean and median depth differences to the baseline are in the table (≤ 1 cm).

One caveat: at the baseline depth, our static poses put the foot 2.5 cm (static_start) and 2.6 cm (static_end) lower than the baseline does, i.e. further into the floor. I tried enforcing floor contact during the fit at fixed depth. It reduced the image fit substantially (static_end IoU 0.43 → 0.37) and was dropped (`USE_FLOOR = False` in `solve.py`). The likely explanation is that the baseline depth is a few cm too far, not that our rotations are wrong — but I can't confirm this.

## Confidence
| phase | confidence | reason |
|---|---|---|
| static_start | Medium | Pooling over 125 frames removes the wandering. Nearby roll optima (±15°) score within 0.01–0.03 of each other, so the roll is uncertain by about ±10–15°; the long-axis direction (≈ 2°) is reliable. |
| static_end | Medium-high | Changes from the baseline are small (8.7°) and consistent across starting points. |
| carried | Medium | Clear gains in smoothness, held-out IoU and the frames 244–268 flip fix. Transitions (around 300–320) and frames behind the stool remain uncertain. |

## Run time / resources
- Final `solve.py`: 196 s, GPU memory under about 5 GB.
- Total GPU time over all experiments: about 35 minutes, within the 45-minute budget.
