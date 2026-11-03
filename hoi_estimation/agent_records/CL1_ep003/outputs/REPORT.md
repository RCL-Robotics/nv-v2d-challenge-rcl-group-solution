# Mug trajectory, episode 3 (`contigo_coffee_mug`, 781 frames): report

Result: `object_poses_world.npz` (`T_world_mesh`, 781×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(log: `solve_log.txt`). Run time 302 s (≈40 s of it is video/mask decoding). Peak GPU memory 1.5 GB.

## 1. What the baseline gets wrong (diagnosis)
* **Azimuth about the mug axis.** The mug is a near-surface of revolution (height 20 cm, Ø 8.8 cm). Only the handle
  breaks the symmetry, and the SAM2 masks largely miss the handle (it is dark against the dark shirt or chairs). In
  frame 0 the baseline puts the handle at φ = +29° (pointing away and to the right). Image evidence and the grasping
  hand put it at φ ≈ −105° (image left), about 135° off. In the later rest phases the baseline handle sits 15–20 cm
  from the hand that is holding it.
* **Tilt and jumps.** The baseline is never exactly upright on the table (≈7° tilt) and jitters 3–6° per frame while
  the mug is at rest. During lifts there are jumps of 50–150° (e.g. frames 299–301, 416–422, 540–545, 653–660).
* **Depth.** The mug is put back on the same table spot every time: the mask bottom row is 671–674 px in all rest
  phases, and the mug's image size is identical in phase 0 and phase 5 to within 1 px (lid edge to bottom edge, and
  the widths). Yet the baseline puts phase 0 at 2.595 m and the other rest phases at 2.70–2.74 m. A 12 cm difference
  would be a 4.5 % (≈3 px) size change, which is not observed.

## 2. Phases (automatic, from the SAM2 object mask)
"On table" means the mask bottom row is within 2.5 px of the rest level (mode = row 672). Runs of ≥12 frames are rest
phases, shrunk by 2 frames at each lift boundary. A rest boundary frame whose static-pose body IoU falls more than
0.05 below the phase median is handed to the lift solver (this moved 403–408 into lift 3).

| phase | frames | content (verified by eye) |
|---|---|---|
| R0 | 0–132 | mug on table; person walks in; hand grasps the handle from ~105 |
| L1 | 133–209 | lift with the image-left hand, carry, set down |
| R1 | 210–272 | on table, hand on/near the handle |
| L2 | 273–343 | lift (person looks into the mug, bottom tilted towards the camera) |
| R2 | 344–402 | on table |
| L3 | 403–474 | lift |
| R3 | 475–532 | on table, hand releases and re-grasps |
| L4 | 533–598 | lift |
| R4 | 599–643 | on table |
| L5 | 644–684 | short lift |
| R5 | 685–780 | on table, hand leaves at ~703 |

(The task text says the hands alternate. In the video the same image-left hand grasps every time, and both the
keypoints and the images agree on that.)

## 3. Method
**Renderer.** nvdiffrast renders the textured mesh in 128-px windows around the object, with a separate handle-face
mask. Silhouette terms use the *body only*: pixels near the rendered handle are ignored, because the SAM2 mask misses
the handle. Person-mask pixels count as possible occluders only if they lie within 14 px of a hand keypoint that is in
front of the object. The rest of the person (the torso behind the mug) is treated as background.

**Rest phases (one pose per phase).**
1. The mug is upright: its bottom-face normal is aligned with the gravity of the released floor plane.
2. The handle azimuth φ is searched on a 5° grid. For each φ the hypothesis is rendered once at 2× supersampling,
   aligned by half-pixel shift search to the body silhouette over ~24 frames of the phase, and scored by grey-level
   NCC against every frame.
3. The pooled NCC curve (smoothed ±10°) is combined with a prior from the grasping hand: von Mises-like, σ = 35°,
   centred on the horizontal direction from the mug axis to the hand centroid, over frames where a hand is within
   12 cm of the axis.
4. Sub-pixel image position comes from continuous refinement of the body silhouette.
5. Depth: all rest phases lie on one table plane parallel to the floor. Its height is set so the rest depth matches
   the baseline's median rest depth (2.708 m). The baseline depth is calibrated against the human and floor, so the
   overall level is kept, but the per-phase scatter is removed. The fitted table height is 0.752 m, and each phase on
   its own gives 0.749–0.756 m, so the phases are consistent to ±3 mm.

**Lift phases.**
1. *Initialisation by hand propagation.* The 21 keypoints of the grasping hand move almost rigidly during a lift
   (Kabsch residual 1–6 mm), and their rotation follows the visible mug tilt (50–64°). The lift-off rest pose is
   propagated forward, the set-down rest pose backward, and the two are slerp-blended over time.
2. *Four candidate sequences per lift:*
   * hand-propagated;
   * baseline with its azimuth snapped so the handle points to the hand, at the hand-propagated depth;
   * the silhouette-ambiguous mirror-tilt twin of each of the above (body axis reflected about the plane orthogonal
     to the viewing ray).
3. Each candidate is refined jointly over the whole lift. The terms are body silhouette (coarse-to-fine blur),
   per-frame centroid pre-alignment, a weak rotation prior towards the hand propagation, a depth prior (the image term
   barely constrains depth), and SE(3) second-difference smoothness with the neighbouring rest poses as fixed
   endpoints.
4. Per-frame score = body IoU + 0.5·NCC, plus a small penalty for deviating from the hand propagation (σ = 30°).
   Viterbi picks one candidate per frame, with a transition cost of (Δrot/10°)² + (Δt/2 cm)². A final joint
   refinement runs from the decoded sequence.

## 4. Validation (no ground truth)
Metrics are computed on all frames. For rest phases most frames are held out, since only ~24 per phase are used in
the fit. *IoU_body*: IoU with the SAM2 mask, with handle and hand-occluder pixels ignored. *NCC*: textured render vs
image inside the rendered mask. *hand–handle*: median distance between the handle centre and the nearest hand
centroid. It is independent evidence for the rest-phase position and depth; for lifts it is partly circular, because
the hand initialises them. In R0 and R5 the hand is mostly away, so large distances are expected there.

| phase | IoU_body base → ours | IoU_full base → ours | NCC base → ours | hand–handle cm base → ours | median Δrot/frame base → ours |
|---|---|---|---|---|---|
| R0 0–132 | 0.708 → 0.822 | 0.702 → 0.760 | 0.427 → 0.480 | (65 → 59, no grasp) | 3.2° → 0 |
| L1 133–209 | 0.769 → 0.859 | 0.745 → 0.839 | 0.243 → 0.380 | 7.6 → 3.6 | 6.6° → 1.8° |
| R1 210–272 | 0.721 → 0.851 | 0.651 → 0.819 | 0.226 → 0.401 | 16.6 → 3.5 | 5.5° → 0 |
| L2 273–343 | 0.544 → 0.862 | 0.547 → 0.837 | 0.091 → 0.374 | 11.9 → 2.8 | 10.7° → 2.2° |
| R2 344–402 | 0.690 → 0.866 | 0.627 → 0.817 | 0.219 → 0.427 | 19.0 → 5.3 | 5.5° → 0 |
| L3 403–474 | 0.658 → 0.837 | 0.614 → 0.817 | 0.130 → 0.345 | 13.3 → 3.2 | 7.7° → 2.5° |
| R3 475–532 | 0.704 → 0.831 | 0.636 → 0.795 | 0.155 → 0.472 | 16.3 → 4.9 | 5.6° → 0 |
| L4 533–598 | 0.688 → 0.861 | 0.659 → 0.847 | 0.140 → 0.341 | 14.9 → 2.7 | 8.2° → 3.3° |
| R4 599–643 | 0.667 → 0.844 | 0.604 → 0.830 | 0.146 → 0.494 | 18.2 → 2.6 | 6.4° → 0 |
| L5 644–684 | 0.531 → 0.857 | 0.497 → 0.846 | 0.041 → 0.408 | 18.1 → 1.9 | 14.0° → 1.5° |
| R5 685–780 | 0.748 → 0.850 | 0.677 → 0.768 | 0.283 → 0.527 | (39 → 45, hand leaves) | 4.6° → 0 |
| **all** | **0.686 → 0.847** | **0.646 → 0.809** | **0.220 → 0.428** | | |

* Per frame: ours beats the baseline by >0.05 body IoU on 614 of 781 frames and is worse by >0.05 on 3 (129, 133,
  173). Minimum IoU is 0.67 (frames 434, 452).
* **Cross-phase consistency check.** The rest azimuths are estimated independently per phase: −105, −120, −125, −110,
  −100 and −115°. Propagating each lift-off pose through the lift with the hand motion lands within 8–10° (20° for
  the short L5) and 0.4–8 cm of the independently estimated set-down pose.
* **Physical checks.** Rest poses sit exactly on the table plane, 0.754 m above the floor, which is consistent with
  the floor-plane tolerance. At lift-off in L3 the lowest mesh point dips about 7 mm into the table top; nothing comes
  close to the floor.
* Our trajectory differs from the baseline by 9.3 cm mean ADD. Most of that comes from the corrected azimuth (handle
  side) and the phase-0 depth.
* **Determinism.** Seeds are fixed. Two runs differ by 0.03 cm mean ADD (max 0.5 cm), caused by float atomic adds in
  nvdiffrast's antialias backward.

## 5. Depth: cues and their disagreement
| cue | rest-phase centre depth |
|---|---|
| baseline per phase | 2.595 / 2.719 / 2.727 / 2.735 / 2.732 / 2.700 m |
| same table pixel and same image size in all phases (geometry) | all rest phases equal to within ~3 cm per px of bottom row |
| hand contact: z(hand centroid) − z(handle) at our depth | −0.4 to +1.3 cm in R1–R5 and L2–L5; −5.4 cm in R0 grasp, −3.3 cm in L1 |
| MoGe-2 aligned depth on the mask (+4 cm surface→axis) | 2.61–2.69 m (noisy, ±5 cm per frame) |
| silhouette scale vs SAM2 mask | monotone, prefers ≤2.55 m (mask is dilated relative to the mug; unusable) |
| image-edge size (mask-free) | flat/noisy over 2.55–2.70 m |

Because the cues do not agree, I did not move the global depth level. It stays at the baseline's median rest depth
(2.708 m). I only made it consistent across phases (one table plane), which corrects R0 by +11 cm, and smooth in the
lifts. MoGe and image size lean 2–5 cm closer, while the hand-contact cue supports the current level. A residual
global depth error of up to ~±4 cm is the main remaining risk, and it would affect all frames equally.

## 6. Confidence per phase
* **R0: high.** 16 of 17 sampled frames have their per-frame NCC maximum at −100° (10° grid; the pooled 5° grid gives −105°), the handle is clearly visible against a light background,
  and the grasp at 120–134 agrees (−91 to −100°). The +11 cm depth correction rests on the geometric argument above:
  medium-high.
* **R1–R4: medium-high.** The handle is mostly hidden by the hand, so pooled NCC is flat. The hand prior decides, and
  agrees with NCC in R2–R4. In R1, NCC alone prefers a secondary mode (+35°) and the hand decides (−120°); the lift
  propagation from R0 independently arrives within 10° of −120°.
* **R5: medium-high.** NCC peak −115° agrees with the hand at release (−113°).
* **Lifts: medium.** Silhouettes and handle-in-hand are well explained. The out-of-plane tilt sign is weakly
  observable at 40–60 px object size. The Viterbi choice between mirror twins is decided by small score differences
  in L1 and L4.

## 7. What did not work / not used
* A free 6-DoF silhouette fit against the SAM2 masks tilts the mug to explain the missing handle; this is why the
  silhouette is body-only and rest poses are kept upright.
* Re-segmenting with SAM2 on 4× crops recovered the handle only sometimes.
* MoGe-2 depth on the object is too noisy to use.
* Raw texture NCC is weakened by specular stripes baked into the texture. It is only used pooled over many frames or
  combined with other cues.

GPU usage over the whole session was roughly 30 min of GPU-process wall time, below the 45 min limit; the final run
took 5 min with a 1.5 GB peak.
