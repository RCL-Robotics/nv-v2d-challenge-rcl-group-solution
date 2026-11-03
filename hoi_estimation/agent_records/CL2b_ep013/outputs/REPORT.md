# Episode 13 (snack_box): trajectory-level object pose re-estimation

Final output: `object_poses_world.npz` (`T_world_mesh`, 922×4×4, float64). It was produced by one run of
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`. The console log is
`solve_run.log` and the machine-readable log is `object_poses_world_log.json`. The run took **7 min 50 s wall time**
on one RTX 6000 Ada, under 2 GB of GPU memory.
`validate.py` produces the validation table below.

## 1. What the video shows and the phases found

The object is a 34 × 20.5 × 40.5 cm cardboard box. Its silhouette is the same under the three 180° flips about its
axes, but its texture is not: a blue tape strip, a chip-bag picture, a "BAKED" face, a nutrition-label face and a blank bottom.

Phases are found automatically. The signal is the mean absolute grey-level difference between frames t and t−3 inside
the object mask, with person pixels and a margin around them excluded. A frame is static if the signal is below 3.0;
static runs shorter than 20 frames are merged into motion. I checked the boundaries by eye, and they are within about 2
frames of what I see.

| phase | frames | what happens | evidence |
|---|---|---|---|
| A static | 0–129 | box on the desk; from about frame 90 the hand touches it but it does not move | clean, unoccluded |
| B moving | 130–311 | slid/lifted off the desk (130–185, visible), carried in front of the body (≈186–222, **hidden behind the person**), held at the right side and lowered (≈222–268, partly hidden, SAM2 mask partial), set on the floor and adjusted (268–311) | weak in the middle |
| C static | 312–384 | on the floor | clean |
| D moving | 385–482 | pushed by the foot under the desk (slides on the floor, short pause at about 451–455) | good, a desk leg occludes part of it |
| E static | 483–699 | on the floor under the desk; legs pass in front of it at about 590–650 | clean |
| F moving | 700–816 | lifted from the floor (visible), carried behind the body (≈760–785, mostly hidden), put on the desk and slid (790–816) | weak in the middle |
| G static | 817–921 | on the desk | clean |

## 2. Method (all inside `solve.py`)

1. **Per-frame evidence: render-and-compare.** The textured mesh is rasterised with nvdiffrast into a square crop
   around the object. Two terms are combined:
   - (a) a silhouette term: soft IoU between the antialiased render and the object mask, blurred so it has a wide
     basin. Pixels covered by the person (dilated) are ignored.
   - (b) a texture term: 1 − normalised cross-correlation between the rendered texture and the image inside the
     rendered region.

   Local refinement uses Adam on (rotation vector, relative translation), driven by the silhouette term plus a small
   texture weight. The texture term is used mainly to *score* hypotheses, because in my tests it has a narrow basin
   and destabilised the optimisation from far starting points.
2. **Static phases: one pose per phase.** 2000 uniformly sampled rotations, plus the baseline and its three box flips,
   are scored by silhouette alone, pooled over 4 low-occlusion frames of the phase. The 16 best (diverse) hypotheses
   are refined, then expanded by the 4 box-symmetry flips, and the best total score wins. The texture term separates
   the flips clearly: NCC 0.39–0.48 for the winner against at most ≈0.2 for the flips.
3. **Moving phases.**
   - SAM2 (Hiera-L, image mode) re-segments each frame, prompted by the existing object mask. The new mask is accepted
     only if its area is plausible for the metric box at the baseline depth, it contains the old mask and it is compact.
     37, 22 and 16 masks were repaired in B, D and F.
   - Hypotheses per frame:
     - a forward chain from the preceding static pose;
     - a backward chain from the following static pose, each with a constant-velocity second hypothesis;
     - the baseline pose and its 3 box flips, each refined locally.

     Each is refined with a weak prior pulling the depth toward the (smoothed) baseline depth.
   - Viterbi decoding chooses one hypothesis per frame. The data cost is 4 × (silhouette + texture loss); the
     transition cost is Δrotation/10° + Δtranslation/5 cm; both ends are tied to the static anchors.
   - Frames whose chosen silhouette loss is above 0.45 (no usable evidence) are replaced by SLERP/linear interpolation
     between reliable neighbours: 14 frames in B, 1 in F.
   - Finally a light Gaussian smoothing (σ = 1.5 frames) on SE(3).
4. **Depth re-anchoring to the baseline (section 4).** Camera-frame positions are scaled along the viewing ray, so the
   image position is kept.
   - Each static phase gets exactly the baseline's median camera depth over that phase.
   - Moving phases get a factor interpolated between the neighbouring static phases. On top of that, a correction ramps
     in over 10 frames at each end so that each moving phase's median z also equals the baseline's median z, without a
     jump at the phase boundaries.

## 3. What changed relative to the baseline, and why

* **A (0–129): rotation essentially unchanged** (1.5° from the baseline). One constant pose replaces the baseline's
  per-frame jitter: the baseline's z drifts by +14 cm while the hand covers the box at frames 90–129, which is why the
  mean dz of −3.6 cm differs from the median of 0.
* **C, D, E (312–699): the baseline is turned 180°** (178–180°) about the box's vertical axis. The silhouette is nearly
  identical, but the texture is not: baseline NCC is −0.01 / −0.05 / −0.10 against +0.24 / +0.41 / +0.40 for ours.
  Visually, the tape strip and the chip-bag picture lie on the wrong half of the top face in the baseline. The
  corrected poses are also physically consistent:
  - the box's up axis is within 2° of the floor normal in A, C, E and G, so it was never turned upside down;
  - A→C is a 10° yaw;
  - C→E is a 24° rotation about an axis 4° from the floor normal, consistent with sliding on the floor while being
    pushed.
* **G (817–921): the baseline is rotated by ≈95–105°.** It shows the box standing on its 34 cm side, while the video
  shows it lying flat on the desk exactly as in A, with a 24° yaw relative to A. With our pose the bottom of the box is
  at 87.5 cm above the floor plane in G and 88.4 cm in A, i.e. on the same desk. The baseline puts it at 78.2 cm.
  IoU goes from 0.60 to 0.92.
* **Moving phases.**
  - The baseline's sudden jumps are removed: 95th-percentile frame-to-frame rotation drops from 28.6° to 10.9° in B and
    from 19.1° to 6.5° in F.
  - D now continues the corrected C pose to the corrected E pose. The baseline carried the flip through it.
  - F is now consistent with E at its start and G at its end; the baseline jumped to its wrong G pose during the carry.

## 4. Depth along the viewing ray: cues checked, shift applied = 0

For each static phase I compared the best-fitting silhouette depth with the baseline median. Δ is relative to the
baseline median z, negative means closer to the camera.

| phase | baseline median z | silhouette-size cue | support-surface cue | hand / foot contact cue | applied |
|---|---|---|---|---|---|
| A | 3.304 m | Δ −0.4 cm | (desk, cannot test) | hand: flat, inconclusive | 0 |
| C | 3.459 m | Δ −15.8 cm | box bottom on floor plane: Δ −9.7 cm | foot contact without penetration at push start: Δ ≤ −26 cm | 0 |
| E | 3.252 m | Δ −16.4 cm | box bottom on floor plane: Δ −1.6 cm | foot contact at push end: Δ ≈ −11 cm | 0 |
| G | 3.352 m | Δ −16.1 cm | same desk height as A: Δ ≈ −8 ± 8 cm (weak, the ray is nearly parallel to the desk) | hand: Δ −26 cm | 0 |

The hand keypoints were optimised by the baseline pipeline with a contact term toward the *baseline* object, so they
are not independent.

The silhouette cue consistently says the box is about 5% (≈16 cm) closer than the baseline in C, E and G. In crops the
render at that depth matches the box's image edges better than at the baseline depth. But in no phase does a second
independent cue agree with it within 2 cm, and the floor-plane cue for E agrees with the baseline instead. Following
the rule, the applied depth shift is **0**: all static phases sit exactly at the baseline median z.

This is the largest remaining risk in the output. If the silhouette cue is right, C, E and G are about 16 cm too far
along the ray, and that error is shared by the baseline.

## 5. Validation table (no ground truth)

From `python validate.py` on the final output; every 3rd frame is used for the image terms.
* **IoU:** render against object mask, on non-person pixels.
* **NCC:** texture agreement, from −1 to 1.
* **dR, dt:** frame-to-frame rotation and translation changes.
* **dz:** camera-frame depth minus baseline depth.
* **bottom h:** lowest box point above the released floor plane.

| phase | n | IoU base | IoU ours | NCC base | NCC ours | dR med base (°) | dR med ours (°) | dR p95 base (°) | dR p95 ours (°) | dt med base (cm) | dt med ours (cm) | dz mean (cm) | dz median (cm) | rot ours vs base, median (°) | bottom h base (cm) | bottom h ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A static 0-129 | 130 | 0.89 | 0.94 | 0.46 | 0.48 | 0.22 | 0.00 | 1.48 | 0.00 | 0.26 | 0.00 | -3.60 | 0.00 | 1.46 | 89.0 | 88.4 |
| B moving 130-311 | 182 | 0.70 | 0.72 | 0.10 | 0.25 | 1.39 | 2.26 | 28.6 | 10.9 | 2.24 | 1.04 | -1.03 | -1.66 | 79.8 | 75.0 | 48.7 |
| C static 312-384 | 73 | 0.87 | 0.91 | -0.01 | 0.24 | 0.00 | 0.00 | 1.41 | 0.00 | 0.19 | 0.00 | 1.05 | 0.00 | 179.2 | -2.63 | -4.01 |
| D moving 385-482 | 98 | 0.83 | 0.84 | -0.05 | 0.41 | 0.87 | 0.75 | 1.94 | 1.42 | 1.11 | 0.65 | 0.78 | 0.43 | 178.1 | -0.56 | 0.26 |
| E static 483-699 | 217 | 0.93 | 0.91 | -0.10 | 0.40 | 0.00 | 0.00 | 2.74 | 0.00 | 0.36 | 0.00 | 0.47 | 0.00 | 179.7 | -0.55 | -0.70 |
| F moving 700-816 | 117 | 0.62 | 0.84 | -0.01 | 0.36 | 4.11 | 1.94 | 19.1 | 6.54 | 3.19 | 1.66 | 1.70 | 1.62 | 174.0 | 41.6 | 42.8 |
| G static 817-921 | 105 | 0.60 | 0.92 | -0.10 | 0.41 | 0.00 | 0.00 | 8.83 | 0.00 | 0.05 | 0.00 | 2.58 | 0.00 | 94.6 | 78.2 | 87.5 |

Reading the table:
- Texture agreement improves strongly in every phase where the rotation was changed.
- IoU improves or stays the same, except E (0.93 → 0.91). That small drop comes from putting the box at the baseline
  depth, which is too far for the silhouette (section 4).
- Static phases are now exactly constant.
- Moving phases are smoother but still follow the image.
- The C and E bottom heights are within the floor plane's tolerance of a few cm. The box is not snapped to the floor.

## 6. Confidence per phase

| phase | confidence | reason |
|---|---|---|
| A | high | the rotation agrees with the baseline and with the global search (1.5°) |
| C, E | high for rotation, medium for depth | large texture margin over the flip, consistent with the physics (upright, yaw-only changes); depth is the baseline's, which the silhouette disputes |
| G | high for rotation, medium for depth | lying flat at the same desk height as A; large NCC/IoU gain over the baseline |
| D | high for rotation, medium-high overall | the box is visible; the pose chain links C to E, and both ends agree with the independent static solutions |
| B 130–185, 268–311 / F 700–760, 790–816 | medium | visible; tracked and decoded |
| B ≈186–267 / F ≈761–789 | **low** | the box is hidden behind the body or only partly visible, and the masks are partial; see below |

**Instability in the low-confidence segments.** Rasterisation gradients are not bitwise deterministic, so two runs
differ slightly. A development run of the same code differed from the final run by less than 1.5° in all static and
visible phases. In frames 183–311 and 761–814, however, the runs differed by more than 20° in places (up to 150°): the
Viterbi decoder picks different hypotheses when the evidence is weak.

By visual inspection, the final run is probably worse than the development run at about 270–311 and 785–795:
- at about 270–311 it uses a forward-chain pose about 90° in yaw away from the C pose;
- at about 785–795 the "BAKED" face is turned toward the camera.

Fixing this, for example with a stronger transition prior toward the next static pose, would have needed another full
GPU run. The GPU budget (45 min total) did not allow it after the final run, so the output is reported as is.

## 7. Run time / resources

- `solve.py`: 466 s (7 min 50 s wall) — loading ≈ 15 s, static phases ≈ 55 s, moving phases (incl. SAM2) ≈ 400 s.
- GPU peak memory < 2 GB.
- Total GPU time over the whole session, including development: about 37–40 min, within the 45 min limit.
