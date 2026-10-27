# Episode 11 — light_blue_book: trajectory-level object pose recovery

Result: `object_poses_world.npz` (`T_world_mesh`, 901×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`.
That run took 3 min 52 s wall-clock, of which about 2.5 min was on the GPU (<3 GB). Its log is `solve_log.txt`.
All analysis and experiments together used roughly 24 min of GPU time, including the final run.

## 1. What is in the video (phases)

Task text: "pick up the book from the table and place it on the floor". In fact the person moves the book four
times. The camera is ~1.33 m above the floor. The book (19 × 5.9 × 26 cm) is ~3.1–3.6 m away and only ~70×40 px
in the image.

| phase | frames | state | how found |
|---|---|---|---|
| S1 | 0–77 | lying on the table, back cover up, spine toward the camera | automatic (image motion) |
| C1 | 78–214 | lifted, carried behind the table's front-left corner, put on the floor | between rest phases |
| S2 | 215–294 | lying on the floor (hand resting on it, no motion) | automatic |
| C2 | 295–404 | picked up, carried back, put on the table | |
| S3 | 405–465 | lying on the table (hand on it) | automatic |
| C3 | 466–657 | picked up, turned over in the hands (≈500–550), carried down to the floor and adjusted | |
| S4 | 658–747 | lying on the floor, front cover up | automatic |
| C4 | 748–829 | picked up, lifted behind the table | |
| E  | 830–900 | held flat by both hands just behind the far edge of the table, slowly lowered, so the tabletop hides most of it | low motion, but rejected as a rest phase (see below) |

The rest phases are detected automatically. The criterion is the median-filtered mean absolute intensity change
over 5 frames, inside the object-mask region with person pixels excluded, below 6.5 grey levels. Morphological
clean-up follows, and a phase must last at least 30 frames. A rest phase must also be supported: its baseline
lowest point must be on the floor (< 8 cm) or at the height of the first table phase (±5 cm). Segment E has low
motion, but its baseline lowest point is 11 cm below the table top, and the video shows it held by both hands. It
is therefore handled as a (slow) moving segment.

Orientation bookkeeping, verified by eye on zoomed frames against renders of the textured mesh: the KAP logo on the
spine is a dark dot at one end, and front and back covers are distinguishable. Back cover up in S1, S2 and S3. The
book is turned over in the hands during C3, so front cover up in S4 and in E.

## 2. Diagnosis of the baseline

- **Baseline = CoCoNet.** The baseline equals the CoCoNet camera-frame estimate, rotated by a constant 1.27° and
  rescaled per frame in depth (same pixel position). FoundationPose differs from it mainly where both are poor; it
  flips by ~170° at frames 575, 825 and 900.
- **Static phases: good.** In every rest phase the baseline orientation wins a pooled texture test against all
  three 180° flips about the mesh principal axes (§3.2). So there is no global symmetry flip in this episode. The
  per-frame jitter is small (median 0.4–2° rotation, 1–2 cm, mostly depth).
- **Carried phases: gross errors.** The per-frame tracker lags the motion and jumps. At 135–150, 325–330 and
  585–590 the rendered book is 10–25 px (5–15 cm) away from the visible book; silhouette IoU is 0–0.4. Rotation
  jumps >8°/frame occur at 81 frames, all of them in carried phases.
- **IoU in segment E was a modelling artefact.** At 830–900 the baseline's apparently low IoU came from my
  evaluation ignoring the tabletop as an occluder. With the table modelled, the baseline is consistent there
  (IoU 0.92).

## 3. Method (what `solve.py` does)

1. **Evidence model.** nvdiffrast renders the textured mesh into a fixed 160×160 crop per frame (2× supersampled,
   antialiased, differentiable). The data terms are:
   - silhouette IoU plus an L2 between blurred rendered and observed masks;
   - a high-pass texture NCC between rendered texture and image, inside the eroded object region.

   Pixels are ignored ("unknown") where something may cover the book:
   - **person mask** (SAM2);
   - **table legs**: two constant rectangles read off the background image by eye; the legs have too little
     contrast with the carpet to segment by threshold;
   - **tabletop**: the tabletop pixels lie on a plane parallel to the floor at the table height, and a pixel is
     ignored when the rendered book surface is behind that plane. The tabletop is segmented automatically: the
     bright, unsaturated connected component of the median background directly below the book in frame 0. The
     table height (0.891 m) is the median lowest-point height of the baseline in the two table rest phases.

   The tabletop occluder is essential: the book is hidden behind it in C1, C2, C4 and E.
2. **Rest phases → one pose each.** The pose is the chordal mean rotation plus the median translation of the
   baseline frames of the phase.
3. **Carried segments.**
   - (a) *Shift search.* For every frame, an FFT search over ±48 px finds the 2-D shift of the baseline silhouette
     that best explains the observed mask (tabletop pixels don't care). The shift is applied as a lateral
     translation only where it raises IoU by >0.1, i.e. gross misplacement (165 frames).
   - (b) *Joint optimisation.* All poses of the segment are optimised together with Adam (150 iterations). Terms:
     the data terms above; constant-velocity smoothness (rotation and translation acceleration, σ = 0.03 rad and
     6 mm per frame²); a robust Geman-McClure prior toward the (shift-corrected) baseline (rotation c = 15°,
     lateral c = 5 cm); and a Gaussian prior on depth along the viewing ray (σ = 3 cm). The neighbouring rest-phase
     poses are fixed anchors at both ends, so the trajectory has to connect them smoothly.

## 4. Experiments and decisions (all without ground truth)

| experiment | finding | decision |
|---|---|---|
| per-frame silhouette refinement of baseline / FoundationPose / 3 flips | IoU 0.8 → 0.95, but rotation scatter inside rest phases grew (S4 median 1.1° → 5.1°); silhouettes cannot separate the flips | not used per frame |
| texture term inside per-frame gradient refinement | diverges (IoU 0.79 → 0.65) | texture used only pooled or inside the joint smooth optimisation |
| pooled flip test per rest phase, 4 hypotheses × all frames | baseline orientation wins all four phases (coarse texture score, e.g. S1 0.27 vs 0.23/−0.02/0.07; S2 wins 100 % of frames) | keep the baseline orientation, no flips |
| free pooled silhouette fit per rest phase | tilts the book ~6° (top toward the camera) on the table AND on the floor, and lowers the texture score (S1 0.68 → 0.27) | rejected: the mask is biased by 1–2 px on a thin silhouette; rest poses stay the averaged baseline |
| flat-on-plane constrained fit (yaw + position) | still lower texture score than the baseline in S1 and S4 | rejected |
| joint segment optimisation, silhouette only | held-out IoU up a lot, but texture score down in C1/C2 (C1 0.087 → −0.04) | rotation drift from mask bias → add texture term |
| joint optimisation + texture term + stronger prior (w_prior = 1) | held-out IoU and held-out texture score both better than the baseline in every carried phase | **adopted** |

## 5. Validation table (baseline vs. result)

- **IoU:** occlusion-aware silhouette IoU against SAM2 masks.
- **texture:** coarse texture score = mean of band-pass (σ 1.5–5 px) NCC and raw NCC of rendered texture vs.
  image. It is not the term being optimised.
- **held-out:** a separate run in which every 4th frame (f % 4 == 2) contributed no data term. Its metrics are
  measured on those frames only.
- **rot-acc / trans-acc:** median frame-to-frame acceleration (smoothness).
- **lowest pt:** median height of the lowest mesh point above the released floor plane.

| phase | IoU base | IoU result | held-out IoU base → ho-run | texture base | texture result | held-out texture base → ho-run | rot-acc base / result (°) | trans-acc base / result (cm) | lowest pt base / result (cm) |
|---|---|---|---|---|---|---|---|---|---|
| S1 table 0–77 | 0.793 | 0.790 | 0.792 → 0.790 | 0.684 | 0.676 | – | 0.00 / 0.00 | 0.03 / 0.00 | 89.4 / 89.4 |
| C1 carry 78–214 | 0.710 | **0.800** | 0.718 → **0.810** | 0.087 | **0.140** | 0.107 → **0.126** | 2.66 / 0.33 | 1.83 / 0.20 | 71.4 / 63.0 |
| S2 floor 215–294 | 0.904 | 0.906 | 0.903 → 0.905 | −0.085 | −0.081 | – | 0.00 / 0.00 | 0.05 / 0.00 | −0.5 / −0.5 |
| C2 carry 295–404 | 0.717 | **0.810** | 0.710 → **0.813** | 0.171 | 0.176 | 0.184 → 0.196 | 3.87 / 0.41 | 1.92 / 0.24 | 57.8 / 59.3 |
| S3 table 405–465 | 0.791 | 0.799 | 0.791 → 0.801 | 0.607 | 0.620 | – | 1.15 / 0.00 | 0.36 / 0.00 | 88.8 / 88.9 |
| C3 carry 466–657 | 0.740 | **0.822** | 0.740 → **0.813** | 0.243 | **0.366** | 0.245 → **0.349** | 2.58 / 0.24 | 1.58 / 0.16 | 80.5 / 80.3 |
| S4 floor 658–747 | 0.912 | 0.911 | 0.910 → 0.910 | 0.156 | 0.152 | – | 0.71 / 0.00 | 0.94 / 0.00 | 1.5 / 1.5 |
| C4 carry 748–829 | 0.800 | **0.816** | 0.815 → **0.842** | 0.017 | **0.081** | 0.003 → **0.067** | 2.98 / 0.14 | 3.40 / 0.16 | 32.2 / 33.2 |
| E held 830–900 | 0.916 | **0.939** | 0.934 → **0.964** | 0.102 | **0.242** | 0.118 → **0.177** | 3.09 / 0.36 | 2.06 / 0.19 | 77.7 / 79.2 |
| all frames with a mask | 0.793 | **0.838** | | 0.202 | **0.252** | | | | |

In the rest phases the result is constant by construction. Its IoU and texture score match the per-frame baseline
(±0.01), so averaging removed jitter without losing agreement with the image. Physical plausibility: the rest poses
sit on the floor (−0.5 and +1.5 cm, within the floor-fit tolerance; feet are 1.8–21 cm above it) or on the table
(88.9–89.4 cm, the same in both table phases). How far the result moved from the baseline: median 1.9° and 1.4 cm;
90th percentile 7° and 4.2 cm; maximum 23° and 14 cm (C1 and C3, where the baseline was visibly off).

## 6. Depth

I kept the baseline's depth calibration, except that it is averaged within rest phases and smoothed (σ 3 cm prior
plus smoothness) in carried phases. The independent cues do not agree on a correction:
- **MoGe-2 aligned depth** at object pixels versus baseline rendered depth: ratio 0.97 (S1), 0.97 (S2), 0.99 (S3),
  0.945 (S4). In S4 MoGe puts the book ~20 cm farther away, which would push it ~7 cm below the floor.
- **Floor contact** (S2, S4): the baseline bottom is −0.5 / +1.5 cm from the floor plane, which is consistent.
- **Table contact:** both table phases give the same bottom height (0.89 m), but they share the baseline's
  calibration. A geometric estimate from the table legs (column foot on the floor plane, column top in the image)
  gives an underside of 0.80–0.82 m, so a tabletop surface of ≈0.83–0.85 m. That is 4–6 cm lower than 0.89 m, but
  it rests on hand-picked pixels and a ~1-px-sensitive geometry. Lowering the book that much would mean moving it
  ~35 cm along the ray, which the silhouette scale rejects.
- **Silhouette scale** alone: ±5 cm per pixel of mask bias.

The cues disagree by up to 20 cm, and in different directions, so no depth correction was applied.

## 7. Confidence per phase

- **S1, S3 (table), S2, S4 (floor): high** for the orientation: the baseline orientation won the pooled texture test
  against every flip, and I confirmed it by eye from the logo and cover. Position is high laterally; depth carries
  the baseline's calibration uncertainty, a few cm. The pose is constant and only jitter is removed, so the gain is
  small but very unlikely to be negative.
- **C1, C2, C3, C4: medium.**
  - Gross translation errors of the baseline (5–15 cm at 135–150, 325–330, 570–600) are clearly fixed: render vs.
    image checked by eye, held-out IoU +0.03 to +0.10.
  - Rotation is a compromise between baseline, silhouette, texture and smoothness. Where the book is mostly hidden
    (frames ~140, 355–362, 575–585, 815–828: no mask or < 100 px) the poses are interpolated by the smooth model
    between anchors.
  - Fast in-hand rotation (C3, ~500–550) is the least certain. Any error there is within the baseline's own range,
    since the same baseline hypothesis is refined.
- **E (830–900): medium-low.** Only a thin strip of the book is visible above the table edge. The result is the
  baseline, smoothed and slightly re-fitted (IoU 0.92 → 0.94, texture 0.10 → 0.24). The depth and height of the
  book behind the table cannot be verified independently.

## 8. Limitations / not done

- No new registrations: FoundationPose re-runs, other pose estimators, optical flow and point tracks were not
  used. The GPU budget was kept for the render-and-compare loop.
- Hand–object rigid attachment was not used. The hand keypoints were not needed to fix the observed failures.
- The table-leg rectangles are constants read off the background image by eye (they are commented in `solve.py`).
- nvdiffrast's antialias gradient uses atomics, so reruns can differ in the last digits (sub-millimetre). Seeds are
  fixed.
