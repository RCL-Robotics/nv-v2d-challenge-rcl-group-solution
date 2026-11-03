# Episode 12 – black_platform (a plastic pallet): object trajectory

The task: "Walk over to the platform on the ground and turn the platform over. Place on the ground when done."
700 frames, static exo camera; the pallet is about 3.5 m from the camera and at most ~190 px across.

Output: `object_poses_world.npz` (`T_world_mesh`, 700×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(log: `solve_log.txt`; every number below is also in `object_poses_world_diagnostics.json`).

## 1. Diagnosis of the baseline
* The baseline silhouettes match the video well. Its rotation is smooth, with no jumps (per-frame change is always under 1.7°),
  and it is essentially the CoCoNet rotation: a constant 1.3° away, which comes from the K difference.
* **Its translation jitters.** In the two phases where the object is provably still, the baseline's camera depth
  z has a standard deviation of 3.3 cm (A) and 2.1 cm (B), with outliers up to 8 cm. Rotation varies by 0.8° and 0.6°.
* During the flip, the baseline's depth swings far back by 10–18 cm around frames 270–370. There the pallet sinks
  up to 7 cm into the floor, while in the static phases the floor contact is within 1–2 cm.
* Symmetry: the mesh is nearly symmetric under a 180° turn about its thin (deck-normal) axis. After ICP the
  mean surface distance is 4 mm and the 99th percentile is 2.4 cm, so a silhouette cannot tell the two apart. The texture is uniformly black.
  The other two 180° flips (legs up vs. down) are clearly rejected by the silhouette: loss ×2.

## 2. Phases (detected automatically in `solve.py`)
Criterion: "motion energy" = mean |I(t+2) − I(t−2)| inside the dilated object masks, with person pixels excluded.
The threshold is 1.6× the level of the first or last 60 frames. I checked the boundaries by eye: the lift starts at about 196,
and the pallet is put down at about 582.

| phase | frames | content |
|---|---|---|
| A, static | 0–196 | pallet standing on its legs. The person walks up and grasps it from about frame 168, but it does not move yet |
| M, moving | 197–584 | pallet tipped up over one long edge, stood on edge (~350–450), and lowered onto its other face. It stays in contact with the floor throughout (see §4) |
| B, static | 585–699 | pallet lying legs-up; the person lets go at about 605 and walks off |

## 3. Method (what `solve.py` does)
1. **Per-frame render-and-compare.** Each frame gets a full 6-DoF optimisation (Adam, 150 iterations), starting from the baseline.
   It compares the nvdiffrast antialiased silhouette of the exact mesh, rendered in a 320 px full-resolution crop, with
   the SAM2 object mask. Pixels covered by the (dilated) person mask are ignored.
   * These fits are very self-consistent: in static phase A their depth std is 0.46 cm (baseline 3.3) and their rotation spread is 0.36°
     (baseline 0.77).
   * Their absolute depth is biased: the silhouette puts the object 3–4% (10–14 cm) closer than the baseline.
     That bias is a size/scale effect, so only the *relative* motion is taken from the fits (see step 4).
2. **Symmetry hypotheses.** I fitted the flipped branch (fit ∘ S, where S is the ICP-refined 180° self-symmetry about the thin axis)
   the same way. I scored both branches on a **held-out cue** that was not used for fitting: a dark-pixel probability map of the image,
   which, unlike the SAM mask, sees the floor through the pallet's openings.

   | phase | frames where the baseline branch is better (dark map / SAM) |
   |---|---|
   | A | 75% / 43% |
   | M | 52% / 34% |
   | B | 75% / 100% |

   The flip rule requires both static phases to prefer the flipped branch in more than 70% of frames. Because the rotation is smooth and no 180°
   spin is visible, the branch choice must be the same for the whole episode. **Result: keep the baseline branch.**
3. **Temporal model.**
   * Static phases: one pose per phase, fitted jointly to every second frame of the phase.
     The initialisation is the chordal rotation mean and the median centre of the per-frame fits.
   * Moving phase: per-frame fits smoothed with a Gaussian (σ = 2 frames) on quaternions and on object-centre positions.
     The static poses act as boundary context. The residual high-frequency noise of the fits is small: 0.34° and 0.5 cm
     against σ = 2 smoothing, compared with 0.41° and 1.4 cm for the baseline.
4. **Depth re-anchoring.** This is done in the camera frame, by moving the object centre along its viewing ray (image position unchanged).
   * A and B: scaled so that the phase median of camera z equals the baseline's median z. The scale factors are 1.0387 and 1.0296.
   * M: the scale ramps linearly from s_A to s_B, plus an offset (tapered to 0 over 30 frames at both phase ends)
     chosen so that the median z of M equals the baseline's median z of M. The trajectory stays continuous across the phase boundaries.
5. Map back to the world with `T_world_cam`.

## 4. Depth cues and the depth decision (moving phase)
Each cue below is the shift of the median camera depth of phase M relative to the baseline anchoring.

| cue | suggested shift |
|---|---|
| floor contact: lowest hull point at the same floor height as in the static phases, where it agrees with the baseline to 0.1–0.2% | −2.0 cm |
| silhouette size constancy: silhouette-to-metric scale calibrated by the baseline in A and B, interpolated across M | −4.7 cm |
| contact with the reconstructed hands (not independent of the baseline; reported only) | −4.4 cm |
| **applied** | **0 cm** |

The two independent cues agree on the direction (closer), but they differ by 2.5–2.7 cm, which is more than the 2 cm rule.
So the baseline's median depth is kept for M. This decision is borderline: if the rule had passed, I would have
applied about −3.4 cm. Even with that median kept, the *shape* of the depth profile inside M changes: it now follows the
smooth silhouette motion, and frames 270–370 move up to ~16 cm closer than the baseline, where the baseline swings back and sinks into the floor.
Around frames 505–510 they move up to ~18 cm farther, where the baseline briefly jumps toward the camera.

## 5. Validation (no ground truth; baseline vs. result)
* IoU and SAM losses are measured against the person-excluded SAM mask.
* "Dark loss" is the held-out dark-pixel map, which was not used for fitting.
* "Floor pen." is the 5th percentile of the signed height of the lowest mesh point above the released floor plane.
* "Accel." is the mean ‖second difference‖ of the world centre, per frame.
* Δz is our camera z minus the baseline's, per frame.

| phase | IoU base → ours | dark loss base → ours | floor pen. (cm) base → ours | accel. (cm/frame²) base → ours | rot. jerk (°) base → ours | Δz mean / median (cm) | mean change vs. baseline: rot° / pos cm |
|---|---|---|---|---|---|---|---|
| A 0–196 | 0.868 → 0.892 | 0.144 → 0.118 | −4.4 → −1.8 | 2.52 → 0.00 | 0.136 → 0.002 | −0.44 / 0.00 | 1.2 / 3.4 |
| M 197–584 | 0.854 → 0.883 | 0.121 → 0.098 | −6.9 → −4.1 | 4.23 → 0.12 | 0.293 → 0.033 | −0.91 / −1.22 | 1.5 / 6.6 |
| B 585–699 | 0.895 → 0.899 | 0.090 → 0.090 | −3.9 → −1.2 | 2.09 → 0.02 | 0.106 → 0.003 | −0.48 / 0.00 | 1.0 / 2.0 |
| all | 0.865 → 0.889 | 0.122 → 0.102 | −5.6 → −3.8 | 3.40 → 0.07 | 0.219 → 0.019 | −0.71 / −0.08 | 1.3 / 4.9 |

* The median camera depth per phase equals the baseline's: the difference of medians is 0 by construction. The median of the per-frame
  differences in M is −1.2 cm, because the baseline's per-frame depth there is skewed by its 270–370 excursion.
* The IoU gain is measured at the baseline's depth, after re-anchoring. The silhouette-optimal depth would give a higher IoU, but it was
  deliberately not used.
* Mean ADD between our trajectory and the baseline: A 3.4 cm, M 6.6 cm, B 2.0 cm.
* `solve.py` is seeded, but nvdiffrast's antialias gradients use atomics, so repeated runs differ slightly:
  1.3 mm mean, 6.6 mm maximum per-frame ADD between two runs.

## 6. Confidence
* **Static phases A and B (45% of frames): high** that the result beats the baseline. The object is provably still, a single pose
  explains all frames, and the baseline's 2–3 cm per-frame depth noise and 0.6–0.8° rotation noise are gone.
  The remaining error is mostly the shared anchor depth, which is the baseline's own.
* **Moving phase M: medium.** The rotations change by 1.5° on average and agree better with the held-out dark map.
  The trajectory is smooth and penetrates the floor less. The median depth is the baseline's.
  The main risk is the depth profile: it now follows the silhouette scale, not the baseline's hand-driven depth.
  Both the floor cue and the size cue support this.
* **Symmetry (yaw 180°): medium-high** that the baseline branch is right. The held-out cue prefers it in both static phases
  (75% of frames), and so do the baseline's own FoundationPose and CoCoNet. The scan is only weakly asymmetric, though, so if
  this were wrong every frame would be wrong. Nothing could be done about that with the available evidence.

## 7. Runtime
`solve.py`: about 125–190 s wall time on the shared RTX 6000 Ada, of which roughly 2 min is GPU. Peak GPU memory was not measured; I estimate it is well under the 9 GB limit
(batches of 50 crops at 320×320 and a 95k-face mesh). Total GPU time over the whole session, including exploration, was about 25 min.
