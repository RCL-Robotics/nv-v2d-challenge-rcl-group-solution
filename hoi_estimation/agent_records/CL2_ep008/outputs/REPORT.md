# Episode 8: skinny_wood_chair ("walk to the chair and push it into the round table")

Result: `object_poses_world.npz` (`T_world_mesh`, 708×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(83 s wall time on the shared RTX 6000 Ada, peak GPU memory about 6.5 GB).

## 1. What the video shows (phases)

Phases are detected automatically in `segment_phases()`. For each frame I compute the object-mask IoU against frame 0 and against the last frame, counting only pixels the person does not cover. The person can only lower these IoUs, so the detector looks for runs of high IoU: at least 10 consecutive frames above (median IoU of the first or last 60 frames) − 0.04. I checked the boundaries by eye in crop montages.

| Phase | Frames | Content |
|---|---|---|
| A, static | 0–399 | The chair stands untouched next to the table. The person walks in front of it (heavy occlusion in 90–230) and rests a hand on the backrest from about 290. The mask does not move: leg-region IoU with frame 0 stays at the static-noise level until frame 400. |
| Moving | 400–591 | The chair is pushed on the floor. It first slides (400–500, about 0.5 m, with little rotation), then turns about 30° while being pushed under the table (500–591). |
| B, static | 592–707 | The chair rests at the table. The hand leaves around 625. |

## 2. Diagnosis of the baseline

- **Rotation:** no 180° flip. The rendered baseline matches the chair's seat, back and legs in every phase. In static phases the rotation jitters by about 0.4°/frame, with jumps up to 2.9°/frame during the occlusion around 140–190 (up to 7° away from the static pose).
- **Translation is not constant in static phases.** Depth z drifts from 2.60 to 2.69 m in phase A (std 2.6 cm), x by 2 cm, and frame-to-frame jumps reach 3.8 cm.
- **Moving phase:** the overall motion is right, but it is jittery (up to 2°/frame and 3 cm/frame).
- **Uprightness:** the mesh's leg-tip plane is tilted 4.1° from mesh z. Relative to that plane, the baseline is within about 1–2° of the floor normal.

## 3. Method (`solve.py`)

1. **Render-and-compare evidence.** nvdiffrast renders the exact mesh's soft silhouette in 512×512 full-resolution crops. The loss is a multi-scale (8/4/2/1) L2 against the SAM2 object mask, ignoring pixels covered by the person mask (dilated by 7 px). Rotation is optimised about the object centre.
2. **Static phases A and B.** One pose per phase is fitted jointly to every 3rd frame of the phase (134 and 39 frames), starting from the chordal mean and median translation of the baseline. Pooling removes the occlusion-induced jumps and the drift.
3. **Moving phase.** Each frame gets its own silhouette fit, starting from the baseline. Then a Gaussian kernel regression on SE(3) over the whole sequence, with the static poses included at both ends so the motion joins them continuously:
   - rotation and image position: σ = 2.5 frames;
   - depth: σ = 6 frames, because silhouettes barely constrain depth;
   - frames with low fit IoU get less weight.

   The σ values come from cross-validation: fit the even frames, then score silhouette IoU on the odd frames. σ ≤ 3 matches the per-frame fits (0.891) and σ ≥ 5 starts to lose (0.888 at 5, 0.882 at 8). A joint optimisation with strong smoothness terms converged poorly and was dropped.
4. **Weak upright prior.** `LAM_UP = 1` pulls the chair's leg-plane normal towards the released floor normal. On its own, the silhouette prefers about 1.6° of extra pitch about the camera x-axis. That pitch has the same sign in both static phases even though the chair's yaw differs by 30°, so it points to camera/floor calibration or the weak pitch observability of silhouettes rather than a real lean. Forcing exact uprightness costs 1.3% IoU. The prior ends up near upright (0.1–0.2° residual tilt), which physically fits a chair standing on the floor; the impact on ADD is under 1 cm either way.
5. **Depth re-anchoring** (as required). Silhouette fits place the chair about 4% (11–13 cm) closer than the baseline, consistently in every phase. This is a scale bias that image metrics cannot detect. Camera-frame positions are therefore scaled along the viewing rays:
   - in each static phase, so that the median z equals the baseline's median z over the same frames (factors A 1.0416, B 1.0389);
   - in the moving phase, with a factor interpolated linearly between those two plus a bump that is zero at the ends, so the moving-phase median z also matches the baseline (target 1.0469).

   The image position is kept: the shift is along the ray, not sideways.

## 4. Depth cues (suggested shift along the viewing ray; negative = towards the camera)

| Phase | Silhouette size (exact mesh) | Hand contact (MHR hand keypoints → chair surface) | Legs on released floor plane | Legs on floor at frame-0 true-human sole level (−2.1 cm vs. plane) |
|---|---|---|---|---|
| A | −10.6 cm | −9.3 cm (50 contact frames) | −2.0 cm | +4.5 cm |
| Moving | −12.9 cm | −5.0 cm (192 frames) | −1.7 cm | +5.3 cm |
| B | −11.4 cm | −7.4 cm (14 frames) | −1.8 cm | +5.5 cm |

Silhouette and hand agree within 2 cm only in phase A. They are 4–8 cm apart in the other phases. The hand cue uses joint centres, not the hand surface, which biases it towards the camera. The floor cue disagrees with both in magnitude (released plane) or in sign (true human soles). No pair of independent cues agrees consistently. **Applied depth shift: 0 cm** (baseline median depth kept per phase).

## 5. Validation (no ground truth; all computed by `/work/scratch/validate.py` and `klt_check.py`)

IoU is the silhouette IoU of the rendered pose vs. the SAM2 mask, person pixels excluded, every 2nd frame. Our IoU is measured **after** re-anchoring to the baseline depth, which shrinks our silhouettes by about 4%. Before re-anchoring the fits reach 0.876 / 0.886 / 0.894 (A / moving / B).

| Metric | A static (0–399): base | A static: ours | Moving (400–591): base | Moving: ours | B static (592–707): base | B static: ours |
|---|---|---|---|---|---|---|
| Mask IoU (mean) | 0.746 | **0.817** | 0.779 | **0.838** | 0.809 | **0.852** |
| Held-out IoU (fit even frames, test odd frames) | – | – | 0.781 | **0.891** (σ = 2.5, before re-anchoring) | – | – |
| KLT texture-track reprojection error, t→t+3 (median px) | – | – | 0.93 | **0.39** | – | – |
| Tracked features landing on the rendered object | – | – | 0.72 | **0.83** | – | – |
| Rotation step, mean / max (°/frame) | 0.43 / 2.92 | 0 / 0 | 0.48 / 1.96 | 0.17 / 0.50 | 0.33 / 1.38 | 0 / 0 |
| Translation step, mean / max (cm/frame) | 0.58 / 3.80 | 0 / 0 | 0.78 / 3.09 | 0.33 / 0.55 | 0.27 / 2.01 | 0 / 0 |
| Position std in static phase, x / y / z (cm) | 0.56 / 0.83 / 2.64 | 0 | – | – | 0.35 / 0.43 / 0.72 | 0 |
| Tilt of the leg plane vs. floor normal (°) | 1.15 | 0.14 | 1.17 | 0.18 | 1.79 | 0.04 |
| Leg-tip height spread, all 4 legs (cm) | 1.20 | 0.33 | 1.16 | 0.39 | 1.65 | 0.28 |
| Mean leg height above released plane (cm) | −0.39 | −0.58 | −0.41 | −0.53 | −0.57 | −0.60 |
| **Camera depth, ours − baseline: mean / median (cm)** | | **+0.67 / 0.00** | | **+0.74 / +0.14** | | **−0.11 / 0.00** |
| Rotation difference to baseline: mean / max (°) | | 1.45 / 6.16 | | 1.31 / 4.55 | | 2.53 / 4.22 |
| Image-position change of the object centre (mean px) | | 3.3 | | 2.4 | | 1.5 |

The moving phase's z std grows from 8.5 to 10.3 cm. That is the true motion: the chair moves away from the camera by about 0.3 m. It is not noise; ours is smooth.

## 6. Changes vs. baseline and confidence

- **A static (0–399): one pose instead of a drifting and jumping per-frame estimate.**
  - Rotation changes by 1.45° on average from the baseline, up to 6° at the occluded frames.
  - Depth is the baseline's median; per frame, it differs from the baseline by up to ±4 cm because the baseline drifted.
  - Confidence: high that the chair is static (mask evidence) and that rotation and lateral position improved. The absolute depth is only as good as the baseline's median.
- **Moving (400–591): smooth silhouette-fitted trajectory.**
  - It explains the image better on both the held-out silhouettes and the independent KLT texture tracks (0.39 vs. 0.93 px).
  - Confidence: high for rotation and relative motion, medium for depth (baseline-anchored).
- **B static (592–707): one pose**, rotated 2.5° from the baseline's mean (consistent with the silhouette, and upright). Confidence: high, depth as above.
- **No 180° symmetry flip** was present or introduced: the chair has no near-symmetry and the overlays match.

## 7. Caveats and reproducibility

- **Determinism.** Seeds are fixed and the optimisation is plain Adam from fixed starting poses. However, the nvdiffrast rasteriser/antialias backward pass uses GPU atomics, so two runs differ slightly: up to 0.29° and 2.2 mm (mean 1.1 mm) between my development run and the final run.
- **Run time.** `solve.py` takes 83 s (≈1.5 min GPU). The development experiments used about 15–20 min of GPU time in total, so the total is under the 45 min budget. Peak GPU memory is about 6.5 GB.
- **Constants.** All hand-set constants are commented at the top of `solve.py`. The phase boundaries are not hard-coded; they come from the mask criterion and were verified by eye.
- **Data used.** `solve.py` uses only `/work/inputs` (video frames are not needed; masks, mesh, camera, plane, baseline). It does not use `/pipeline` or any network access.
