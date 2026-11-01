# Episode 22 — step ladder, dragged on the floor: trajectory-level object pose

Result: `object_poses_world.npz` (`T_world_mesh`, 719×4×4 float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(run log: `solve_log.txt`; validation table: `validation_table.md`, produced by `validate.py`).

## 1. Diagnosis of the baseline
* **No 180° flip.** In each static phase I tested the baseline orientation against its 180° flips about the 3
  principal (OBB) axes. Each was locally refitted to the masks. The baseline orientation wins by a wide margin
  (silhouette loss 0.15 / 0.17 vs ≥ 0.36 / ≥ 0.25 for the best flip). Visual overlays agree.
* **Static phases are not static in the baseline.** In the last static phase (560–718) the baseline camera depth
  drifts from 3.94 m to 3.77 m (std 6 cm), with up to 3° frame-to-frame rotation jumps. In 0–201 the depth std is 2 cm.
* **Drag phase:** the rotation is off by 3–5° on many frames (worst around 360–480, where the silhouette is visibly
  shifted from the mask) and there is jitter of 0.8° and 0.6 cm.
* **Floor:** the baseline is already floor-consistent to within ±3 cm, and the hand is already attached to the top
  rail (contact prob 1.0). So the hand-contact cue is not independent of the baseline.
* **The SAM2 masks are "fat".** They are 2–3 px wider than the true silhouette of the thin tubes (the best IoU
  comes from dilating the render by r = 2–3 px) and include shadows. A naive silhouette fit pulls the object
  **46 cm toward the camera** to fill the fat mask. Even with a fat-mask model, the silhouette scale still suggests
  −5 cm (near phase) and −22 cm (far phase), and its per-frame noise in the far phase is 10 cm. Silhouette scale
  is therefore not used for depth.

## 2. Phases (automatic, from the video; verified by eye)
Motion score = fraction of object-mask pixels (person pixels excluded) whose grey level changes by more than 15
between consecutive frames. A frame is static if the score (5-frame median) is below 0.01; static runs must last at least 30 frames.

| phase | frames | content |
|---|---|---|
| S1 static | 0–201 | ladder standing on the floor; the person walks past (79–81 gave a weak 3-frame blip, score ≤ 0.012, merged automatically; separate fits before and after it agree within 0.7° and 0.7 cm laterally); the hand grasps the rail from about 190 |
| M moving | 202–559 | ladder tipped and dragged on the floor with one hand (fast tipping 215–235) |
| S2 static | 560–718 | ladder standing again; the hand releases it around 590; small in the image (z ≈ 3.9 m) |

## 3. Method (what `solve.py` does)
1. **Evidence model.** I render the mesh with nvdiffrast in batched square crops (192²) per frame. The mesh is
   inflated along its vertex normals by 2.5 px·z/f to model the fat SAM masks. The data term is soft IoU with
   the object mask, ignoring pixels covered by the person, on a coarse-to-fine Gaussian blur pyramid. An optional
   edge term measures the distance from the rendered (non-inflated) occluding contour to Canny edges in the image
   (truncated distance transform).
   **The camera depth of the object centre is always held fixed**: only rotation (about the object centre) and
   lateral translation (in the plane z = const) are optimised.
2. **Static phases (one pose each).** I initialise at the robust mean of the baseline (quaternion mean rotation,
   median centre). Then I fit one shared pose to every 2nd frame of the phase, using silhouette plus edge term
   (weight 0.05). I run 3 optimisation schedules and average them; they agree within 0.4° (S1) and 0.9° (S2).
   The edge term was needed in S2: without it the schedules spread over 1.5°.
3. **Drag phase (per frame, several hypotheses, temporal selection).** I initialise with the SE(3)-Gaussian-smoothed
   baseline (σ = 3 frames).
   * Pass 1: per-frame fit.
   * Pass 2: re-fit each frame starting from an IRLS-robust smoothing of pass 1.
   * Per frame, three hypotheses (pass 1, pass 2, smoothed trajectory = propagation from the neighbours) are scored by
     silhouette loss + 0.004 per degree of deviation from the smooth trajectory, and the cheapest is kept. This removed
     the local-minimum spikes at frames 311–317 (leg occluding the ladder; up to 9–11° in single-pass runs).
   * Finally a light smoothing (σ = 1 frame).
   I tested an in-fit smoothness prior (2nd differences), and the edge term in this phase. Both were rejected by
   the held-out test (§5).
4. **Depth re-anchoring (task default).** For each phase, camera-frame positions are scaled along their viewing rays
   so that the median camera z equals the baseline's median z over the same frames.
5. **Continuity with the static anchors (the only deviation from the baseline depth).** After step 4, the drag
   phase ends 2.8 cm (at 201/202) and 5.9 cm (at 559/560) away in depth from the static poses next to it. These
   are non-physical jumps (the object barely moves there) caused by the slow drift of the baseline depth. I apply
   a linear ramp of the along-ray scale (0.9908 → 0.9859) across the drag phase so that its ends meet the static
   poses. The ramp is applied only if an independent cue agrees, and this check is automatic in `solve.py`: per
   third of the phase, the floor cue (depth shift that puts the median lowest point at the height of the static
   phases' lowest point) must agree within 2 cm, and the correction must be larger than the disagreement.

### Depth cues and the shift applied (cm, negative = toward the camera)
| segment | floor-contact cue | anchor-continuity cue | silhouette-scale cue | hand-contact cue | applied |
|---|---|---|---|---|---|
| S1 0–201 | lowest point −0.5 cm vs plane, consistent with the plane (frame-0 reference human sole: −1.6 cm), no shift needed | — | −5 (depends on the assumed mask fatness) | not independent (baseline post-opt) | **0** |
| M 202–321 | −2.1 | −2.4 | +2…+6 (biased) | not independent | **−2.4** (ramp) |
| M 322–440 | −4.7 | −2.8 | ~0 | not independent | **−2.8** (ramp) |
| M 441–559 | −6.1 | −4.6 | −5…−13 (biased) | not independent | **−4.6** (ramp) |
| S2 560–718 | lowest point −0.5 cm, no shift needed | — | −22 (unreliable, 10 cm noise) | — | **0** |

Over the whole episode the median camera depth is 1.75 cm closer than the baseline (mean 0.9 cm).

## 4. Validation (no ground truth) — baseline vs result, per phase
IoU is computed over frames (every 2nd frame) with person-covered pixels ignored. Edge chamfer is the mean
distance (px, truncated at 10) from rendered occluding contours to image Canny edges. It does not use the masks.

| phase | metric | baseline | result |
|---|---|---|---|
| S1 0–201 | mask IoU (fat-mask model) / plain render | 0.749 / 0.642 | **0.846 / 0.717** |
| S1 0–201 | edge chamfer px | 1.572 | **1.091** |
| S1 0–201 | rot / centre jitter (median dev. from smoothed) | 0.25° / 0.49 cm | 0 / 0 |
| S1 0–201 | lowest point above floor plane: median (std) | 0.77 (1.11) cm | −0.53 (0) cm |
| S1 0–201 | camera depth result − baseline: mean / median | — | 0.12 / 0.00 cm |
| M 202–559 | mask IoU (fat-mask model) / plain render | 0.653 / 0.542 | **0.782 / 0.651** |
| M 202–559 | edge chamfer px | 1.764 | **1.465** |
| M 202–559 | rot / centre jitter | 0.78° / 0.56 cm | 0.42° / 0.18 cm |
| M 202–559 | lowest point above floor plane: median (std) | −1.89 (1.00) cm | −1.09 (0.84) cm |
| M 202–559 | camera depth result − baseline: mean / median | — | −2.64 / −2.35 cm |
| S2 560–718 | mask IoU (fat-mask model) / plain render | 0.766 / 0.631 | **0.827 / 0.640** |
| S2 560–718 | edge chamfer px | 1.380 | **1.082** |
| S2 560–718 | rot / centre jitter | 0.39° / 0.71 cm | 0 / 0 |
| S2 560–718 | lowest point above floor plane: median (std) | −0.13 (1.57) cm | −0.50 (0) cm |
| S2 560–718 | camera depth result − baseline: mean / median | — | 1.83 / 0.00 cm |
| ALL | mask IoU (fat) / edge chamfer | 0.705 / 1.628 | **0.810 / 1.278** |
| ALL | depth jumps at phase boundaries 201→202 / 559→560 | 1.0 / 0.7 cm (baseline not static) | 0.97 / 0.23 cm |

Change vs the baseline (median / p90): rotation 0.9 / 1.4° (S1), 2.8 / 4.7° (M), 2.1 / 3.2° (S2). Translation
2.1 / 3.9 cm (S1), 3.3 / 5.1 cm (M), 5.1 / 10.4 cm (S2; mostly removing the baseline's depth drift around the
phase median).

**Held-out test (drag phase).** I fitted on even frames only, interpolated to the odd frames, and scored the odd
frames:

| variant | held-out silhouette loss | held-out edge chamfer (px) |
|---|---|---|
| baseline | 0.346 | 1.829 |
| smoothed baseline | 0.345 | 1.813 |
| per-frame fit | **0.236** | **1.537** |
| fit + edge term | 0.237 | 1.542 |
| fit + in-fit smoothness prior | 0.286–0.296 | 1.70–1.76 |

On a first full run the edge chamfer improved on 92.5% of the drag-phase frames. The per-frame corrections are
temporally coherent (lag-1 autocorrelation 0.95), not mask noise. The static-phase fits were checked on held-out
frames (odd frames, every 10th) with the edge score: 1.59 → 1.10 (S1) and 1.40 → 1.08 (S2).

**Reproducibility.** nvdiffrast's backward pass uses float atomics, so repeated runs are not bit-identical. Between
two runs, static poses differ by ≤ 0.12° and ≤ 0.03 cm, and drag-phase frames by a median of 0.2°. The 311–317
instability, which reached 11° before the 3-hypothesis selection, is gone: after the final run no frame deviates
from the robust-smoothed trajectory by more than 2.5° (one frame > 4° remains in the log, accepted on image evidence).

## 5. Confidence per phase
* **S1 (0–201): high** for rotation and lateral position: one pose, 3 schedules within 0.4°, a large gain in
  mask and edge agreement, and per-frame free fits agree to 0.1–0.5 cm. The depth is the baseline median
  (uncertainty is the baseline's own calibration).
* **M (202–559): medium-high** for rotation and lateral position (held-out gains, coherent corrections, outliers
  handled). **Medium** for the depth ramp (−2.4 to −4.6 cm): two cues agree within 2 cm, but both are indirect.
* **S2 (560–718): medium.** The object is small (~100 px), so the rotation optimum is flatter (≈1° spread). The
  depth is the baseline median, but the baseline drifts ±8 cm inside this phase, so the absolute depth is the
  least certain part of the episode.

## 6. Run time / resources
`solve.py`: about 4 min wall on one RTX 6000 Ada (static fits about 1.5 min, drag phase 2 passes about
1.5 min). Peak GPU memory is well under 9 GB (crop rendering in chunks of ≤ 128 frames). Total GPU time over the
whole session, including experiments, is about 30 min. No network, no ground truth, and no files outside
`/work/inputs` and the code in `/work/outputs` are used by `solve.py`.

## 7. Files
* `solve.py`: the full pipeline.
* `render_utils.py`: nvdiffrast rendering, crops, and normal inflation.
* `silfit.py`: silhouette and edge render-and-compare optimiser.
* `traj_utils.py`: SO(3)/SE(3) smoothing and re-anchoring.
* `edgescore.py` and `validate.py`: validation only.
* `solve_log.txt` and `validation_table.md`.
