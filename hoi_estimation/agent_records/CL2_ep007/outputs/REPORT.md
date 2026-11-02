# Iron trajectory, episode 7: report

Task: "Walk over to the iron on the round table and using your right hand in the handle slide it back and forth
horizontally on the desk. Stand it up vertically when finished." 896 frames, static camera.

Final output: `object_poses_world.npz` (`T_world_mesh`, 896×4×4, float64). It is produced by
`python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz` in **5 min 56 s wall time**
(SAM2 about 1.4 min, fitting about 4.5 min). `validate.py` produces the tables below.

## 1. Diagnosis: the baseline tracks the wrong object
- The released `masks.h5` `object` masks and the pipeline's SAM2/Grounding-DINO prompt
  (`/pipeline/work/prompts.json`, object box x 675–769, y 476–631) segment the **tripod head behind the table**,
  not the iron. The iron sits on the round table at about x 685–757, y 612–657 in frame 0.
- So FoundationPose, CoCoNet and the baseline `T_world_mesh` place the iron on the tripod head. Rendered baseline
  poses have **0.00 mask IoU with the iron in every frame**. The baseline is ~100 px away from the iron in the
  image and about 20–90 cm away along the viewing ray (table below).
- There is therefore no usable baseline placement to keep. The instruction "re-anchor to the baseline's median
  depth" would put the iron 25–90 cm behind the table, so I did not follow it. I estimated the depth from
  physical cues instead (section 4). The per-phase depth difference vs. the baseline is still reported, as
  required.

## 2. Phases (automatic, checked by eye)
| phase | frames | how it was detected | model |
|---|---|---|---|
| P1 rest on the soleplate, static | 0–155 | the person's mask first touches the iron at frame 105; the fitted pose does not move until 156 (within 1°/4 mm) | one pose pooled over all frames |
| P2 slid back and forth on the table | 156–659 | the sole-down model fits; the flat phase ends when it stays bad for 4 keyframes (670), minus 10 frames of margin | yaw + 2-D position on the table plane per frame |
| P3 stand-up transition (pivot on the heel) | 660–779 | between the sole-down and heel-down phases | free 6-DoF with a table-contact penalty |
| P4 standing on the heel, still held | 780–793 | the heel-down model fits as well as at the end | yaw + 2-D position on the table plane |
| P5 standing, released, static | 794–895 | no hand contact from frame ~800; the pose is constant | one pose pooled |

## 3. Method (`solve.py` → `stages.py`)
1. **Segmentation.** SAM2.1-L video predictor on a 480×400 crop around the table, with a box prompt on the iron
   at frame 0 (a constant found by eye). The crop is upsampled for SAM, so the masks are clean even when the iron
   is only ~70 px wide. The person mask from `masks.h5` marks occluded pixels; they are ignored.
2. **Evidence per hypothesis.** nvdiffrast renders the textured mesh into a 160×160 crop. The score is soft IoU
   with the iron mask on non-occluded pixels, plus a mean/variance-normalised colour residual inside the
   object. Colour is what separates the near-symmetric flips: in the standing phase, yaw 218° vs. 137° have
   almost the same silhouette (0.09 vs. 0.10) but colour residual 0.49 vs. 1.00.
3. **Physical model.** The rest facets come from the mesh's convex hull: the soleplate facet (normal
   (−0.98, 0.08, 0.17)) and the heel-rest facet. Flat and standing poses keep that facet on the table plane, so
   each frame has 3 DoF.
4. **Table normal.** Estimated from the data: free 6-DoF refinement of the keyframe solutions, then the median
   facet normal. It is 1.8° (sole) and 1.4° (heel) from the released floor normal. The table-rim ellipse also
   indicates ~1° of tilt toward the camera.
5. **Hypotheses.** Keyframes every 10 frames: a 36-value yaw grid → silhouette fit → colour refinement of the
   best 3. Viterbi over keyframes (unary = fit score, transition = yaw and position change). Dense per-frame
   refinement from the interpolated Viterbi path.
6. **Static pooling and smoothing.** Untouched or non-moving segments get one score-weighted pose. Moving frames
   get Hampel + Gaussian smoothing (σ = 1.5 frames).
7. **Transition.** Keyframes every 5 frames: 216 orientation hypotheses (yaw × 6 tilts from sole-down to
   heel-down) → 6 candidates refined with contact. Viterbi with both ends pinned to the neighbouring phases.
   Slerp initialisation, dense 6-DoF refinement with the contact penalty, then smoothing.
8. **Depth anchoring (section 4).** Steps 3–7 are done in a "fit space" where the iron's image size matches the
   mesh. The trajectory is then scaled along the viewing rays by one global factor k, so rotations and image
   positions are unchanged.

## 4. Depth (distance along the viewing ray)
The image size of the iron with the given mesh puts it at **2.88 m** (median). All physical cues put it further
away. The SAM mask is slightly fat, so the silhouette alone gives 2.61 m; colour pulls this to 2.88 m. Following
the task's guidance, object-only depth was **not** used. Each cue's implied median iron depth:

| cue | how it was measured | median iron depth | shift vs. applied |
|---|---|---|---|
| table geometry | 5 caster contacts (clicked on frame 0) on the released floor plane, lowered by 1.9 cm so the true frame-0 human's soles touch it → circle centre (residual 2 cm) → vertical column → table-top height (0.81 m) and radius (0.40 m) fitted to the rim ellipse (residual ≤ 3.7 px) | 3.043 m | −2.7 cm |
| MoGe-2 table plane | `/pipeline` MoGe depth (aligned to the reconstructed human) on table-top pixels, frames 0–96, fitted with the table normal | 3.070 m | 0 |
| right hand on the handle | reconstructed right-hand keypoints (mean of 21) vs. the handle top, 702 frames in contact; ratio IQR 1.073–1.086 | 3.110 m | +4.0 cm |
| object size (excluded) | given mesh, silhouette + colour | 2.880 m | −19 cm |

- **Applied:** the median of the three physical cues, k = 1.0659, which gives a median iron depth of 3.070 m.
- **Agreement:** the three physical cues span 6.7 cm. The two closest independent ones (floor/table geometry
  and MoGe) agree within 2.7 cm.
- **Expected error:** I expect ±3–4 cm of global depth error. This is the dominant uncertainty of the result.
- **No baseline anchoring:** the baseline offers no depth for the iron, since it is on the tripod.

## 5. Validation (no ground truth)

Column definitions:
- **IoU:** hard mask IoU of the rendered mesh vs. the SAM2 iron mask, person-occluded pixels excluded.
- **colour:** normalised colour residual; it is undefined for the baseline because there is no overlap.
- **rot p95:** 95th-percentile rotation change per frame.
- **lowest pt:** median height of the lowest mesh point above the table plane.
- **hand-handle:** median 3-D distance between the right-hand keypoint mean and the handle top, in frames where
  the hand touches the iron.
- **Δz:** camera-frame depth of the mesh centre, result − baseline.

| phase | method | IoU mean (p10) | colour | rot p95 (°/frame) | lowest pt (cm) | hand-handle (cm) | Δz mean / median (cm) |
|---|---|---|---|---|---|---|---|
| P1 static 0–155 | baseline | 0.00 (0.00) | n/a | 18.6 | +23.9 | 47.1 | |
| | **result** | **0.81 (0.79)** | 0.58 | 0.0 | 0.0 | 3.3 | −19.6 / −20.1 |
| P2 sliding 156–659 | baseline | 0.00 (0.00) | n/a | 47.3 | +40.0 | 51.4 | |
| | **result** | **0.85 (0.84)** | 0.50 | 0.29 | 0.0 | 3.9 | −25.0 / −24.9 |
| P3 stand-up 660–779 | baseline | 0.00 (0.00) | n/a | 39.1 | +38.3 | 46.4 | |
| | **result** | **0.86 (0.83)** | 0.58 | 2.1 | +1.2 | 6.9 | −23.0 / −22.7 |
| P4 standing, held 780–793 | baseline | 0.00 | n/a | 4.7 | +30.5 | 69.8 | |
| | **result** | **0.83 (0.82)** | 0.56 | 0.87 | +0.7 | 4.6 | −48.1 / −52.7 |
| P5 standing, static 794–895 | baseline | 0.00 | n/a | 0.02 | +30.1 | (hand not on iron) | |
| | **result** | **0.83 (0.82)** | 0.61 | 0.03 | +0.7 | (hand not on iron) | −71.3 / −86.9 |
| all | baseline | 0.00 | n/a | 31.8 | +33.0 | 50.1 | |
| | **result** | **0.84 (0.81)** | 0.56 | 1.2 | 0.0 | 4.5 | −29.4 / −23.9 |

- **IoU ceiling:** IoU is capped at ~0.85 because the SAM mask is a few pixels wider than the blurred
  70–100 px iron. Free per-frame fits reached ~0.92 only by shrinking the depth or tilting the iron.
- **Physical consistency:** the iron never floats or penetrates in the constrained phases. During the stand-up,
  the contact penalty keeps the lowest point within ~1 cm of the table.
- **Hand consistency:** the independently reconstructed right hand stays 3–5 cm from the handle top during
  sliding. Because of this, the hand is also used as a depth cue.
- **Visual check:** overlays of the final world-frame poses at frames 0, 200, 375, 560, 690, 730, 760 and 850
  (projected with `camera.json` K) match the iron. The heel (white disc) is correctly on the camera-facing end
  throughout P1–P3.

## 6. Confidence per phase
- **P1, P2 (rotations): high.** Yaw is unambiguous: flip margins in fit score are 0.3, colour separates heel and
  tip strongly, and the sole is on the table. I expect yaw errors of a few degrees.
- **P4, P5 (rotations): high.** The standing yaw is decided by colour with a large margin; contact is consistent.
- **P3 (stand-up, 120 frames): medium.** The free 6-DoF fit with the contact prior is smooth (median 1°/frame)
  and matches the masks (IoU 0.86), but the pivot dynamics are only weakly constrained between keyframes.
- **Global depth: medium.** ±3–4 cm from the cue disagreement. This affects all phases equally, and the
  in-plane (image) position is accurate to ~1 px (~4 mm).

## 7. What I did not do or could not verify
- FoundationPose and CoCoNet were not re-run with correct masks. Render-and-compare with physical constraints was
  enough, and FoundationPose would also have needed a depth map at the correct scale.
- The caster and table-rim pixel positions are constants measured by eye on frame 0 (commented in `solve.py`).
  So is the SAM2 box prompt.
- The hand cue assumes the mean of the 21 right-hand keypoints sits at the handle top. A systematic offset of a
  few cm along the viewing ray would bias that cue; this is one reason for taking the median of three cues.

GPU use: the final run takes about 6 min of GPU time; the whole session including development took about 30 min.
Peak GPU memory measured for SAM2 was 1.9 GB (frames and state offloaded to CPU). Fitting uses 160×160 crops in batches of ≤64, well below the 9 GB limit.
