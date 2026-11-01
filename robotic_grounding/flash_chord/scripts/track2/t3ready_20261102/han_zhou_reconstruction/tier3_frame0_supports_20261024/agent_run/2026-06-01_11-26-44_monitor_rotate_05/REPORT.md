# Episode 6: monitor on a round table, 6-DoF object trajectory

Output: `object_poses_world.npz` (`T_world_mesh`, 546×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`.
The final run took 7 min 36 s wall time on a shared GPU, essentially all of it GPU work.
Total GPU time over the whole session, experiments included, was about 30 min. Peak GPU memory was well under the 9 GB limit (half-resolution crops, batches of at most 120 renders; not logged exactly).

## What happens in the video (checked by eye, then encoded as automatic criteria)
A person walks to a monitor standing on a round pedestal table. They hold the monitor by its base and turn it by about 180° on the table top while sliding it about 35 cm to the right. Then they let go. The base never leaves the table. I checked zoomed crops of every 4th frame through the fastest part (frames 330–390).

| phase | frames | how it was detected (in `solve.py`) | content |
|---|---|---|---|
| static 1 | 0–163 | image-motion score (mean grey change over un-occluded object pixels, lag 5) stays < 1.3 until frame 164 | monitor untouched until about 114, then hands rest on the base without moving it; screen (white paper) faces the camera |
| moving, front | 164–347 | motion > 1.3 | slides and turns; screen near-frontal around 230–320 |
| moving, side view | 348–379 | (this is where the baseline is physically implausible: tilt > 10°) | monitor seen edge-on, base under the hand, heavily occluded |
| moving, back | 380–468 | | back of the monitor (Dell logo) visible, small adjustments |
| static 2 | 469–545 | first frame after the last hand contact (`contact_prob` > 0.5) where motion < 1.3 | released, back facing the camera |

## Diagnosis of the baseline
- **Frames 348–379 failed completely.** The baseline tilts the monitor by 30–90° (sideways, flipped and floating). Its yaw jumps by up to 160°, and its texture-flow error is 10 px or more.
- **Per-frame tilt noise everywhere.** Even where yaw is fine, the monitor's resting face is 4–6° off the floor normal, in directions that are inconsistent between phases in both the world and the object frame. A 5° tilt on a 0.54 m object is about 2–3 cm of ADD error.
- **Jitter in static phases.** Depth wanders by ±5 cm, and yaw by ±1–2°, while the object is perfectly still. In static 1 the depth drifts from 2.57 m to 2.67 m.
- **No front/back flip.** The baseline shows the screen at the start and the back at the end, which is correct. My front/back term confirms it.

## Method (`solve.py`, `hoi_fit.py`)
1. **Physical pose model.** The monitor stands upright on a table parallel to the released floor plane. Its pose is yaw about the floor normal plus a 2-D position on the table plane, i.e. 3 DoF per frame.
   - **Resting face:** the area-weighted bottom faces of the mesh's convex hull. They are 3° off the mesh x axis.
   - **Table height:** the median base-contact height of the plausible baseline frames.
2. **Per-frame evidence on a 3° yaw grid (120 hypotheses).** For each hypothesis the table position is optimised with Adam. Silhouettes are rendered with nvdiffrast at half resolution, from an 8k-face decimated mesh, with part labels for screen panel and screen front. The cost has three parts:
   - **Two-sided silhouette error against the SAM2 mask.** It takes the better of "mask = whole monitor" and "mask = screen panel only", because SAM2 often drops the stand and base.
   - **Occlusion handling.** Person pixels are ignored only where the reconstructed human (splatted vertices) is in front of the object. Person pixels behind the monitor stay as evidence; ignoring them all made the screen swell into the person.
   - **Front/back term.** The real screen is covered by white paper, so it is bright, while the back is dark grey. The mesh's screen texture (a dark wallpaper) is therefore not used.

   This term is decisive: without it, the screen silhouette fits the 180°-flipped pose equally well.
3. **Static phases pool evidence.** Each static phase uses the mean cost curve over all its frames, which gives one pose for the whole phase.
4. **Global decoding.** Viterbi over yaw for the whole episode:
   - Transition cost uses σ = 4°/frame while moving and 0.3° in static phases.
   - The data cost is capped, so occluded frames cannot dominate.
   - A weak, capped prior pulls toward the baseline yaw on plausible baseline frames. It only matters where the image cost is flat, i.e. near-frontal views where ±10° of yaw changes the silhouette by about 1%.

   After Viterbi: parabolic sub-grid refinement, then quadratic second-difference smoothing with the static phases held fixed. Table positions get median filtering and the same smoothing.
5. **Depth re-anchoring (as required).** Each pose is moved along its viewing ray with rotation unchanged. In static phases the camera-frame depth is set to the baseline's median depth. In the moving phase the ratio to the baseline depth is heavily smoothed (σ = 30 frames, computed only over plausible baseline frames). Rotations, image positions and the shape of the motion are mine; overall depth is the baseline's.

### Parameters chosen with checks
- **Baseline-prior weight 0.3 and translation smoothing λ = 100** were chosen with the optical-flow check described below. That check is never used in fitting.
  - Flow error, prior weight 0 / 0.1 / 0.3 / 1 / 3: 0.77 / 0.73 / 0.73 / 0.82 / 0.87 px.
  - Flow error, λ = 8 / 50 / 200 / 1000: 0.73 / 0.68 / 0.66 / 0.73 px.
- **Gradient joint refinement was rejected.** I tried a joint gradient refinement of all frames (silhouette + smoothness + shared static poses). It got stuck in static 2 at a yaw 8° away from what a pooled 1-D yaw scan and a texture-gradient correlation both preferred, so I dropped it.

## Depth: cues and decision (applied shift: 0, i.e. the baseline's depth is kept)
| cue | suggested depth change vs baseline |
|---|---|
| metric mesh size vs image size: free 6-DoF fits to the clean full-object mask of frames 0, 30, 60 | −14 to −18 cm (closer); pooled fits over other frames scatter between −10 and −35 cm |
| object resting on the table: table located independently from the 5 caster contacts (back-projected onto the released floor plane; regular pentagon, radius 0.324–0.343 m, so floor and K are consistent) plus the table-top ellipse, giving a table-top height of 0.82–0.84 m vs the baseline's base height of 0.849 m | +5 to +15 cm (farther), ±8 cm |
| hand contact | not informative: the hands hold the base during the whole moving phase, so contact only constrains the base, which both depths satisfy |

The two cues disagree in sign, so per the rules the depth stays at the baseline's. The result's median camera-frame depth equals the baseline's in every phase (table below). The exception is the baseline-failure segment, where the baseline depth itself is unreliable and the smoothed ratio from the neighbouring frames is used.

## Validation (no ground truth; `python /work/outputs/validate.py`)
- **flow err:** points tracked with LK (forward-backward checked) on un-occluded object texture between frames t and t+4. They are lifted to 3-D with the rendered depth of the pose at t, moved with the trajectory's relative motion, and reprojected. The column is the mean of min(error, 10 px). This measures relative motion and is not used in fitting.
- **mask IoU:** against SAM2, best of whole monitor / screen only, person-occluded pixels excluded.
- **tilt:** angle between the base's resting face and the floor normal.

| trajectory | phase | flow err px (lower better) | mask IoU | tilt deg (median) | 95% \|dyaw\|/frame deg | 95% accel cm/f² | dz vs baseline mean / median cm | base height above floor m |
|---|---|---|---|---|---|---|---|---|
| baseline | static-1 [0-163] | 1.04 | 0.861 | 4.1 | 0.68 | 3.41 | 0.0 / 0.0 | 0.849 |
| **result** | static-1 [0-163] | **0.07** | 0.841 | 0.0 | 0.00 | 0.00 | 0.8 / 0.0 | 0.849 |
| baseline | moving (front) [164-347] | 1.53 | 0.857 | 3.9 | 2.67 | 4.29 | 0.0 / 0.0 | 0.857 |
| **result** | moving (front) [164-347] | **1.08** | 0.858 | 0.0 | 2.31 | 0.11 | 0.5 / -0.2 | 0.854 |
| baseline | moving (baseline failure) [348-379] | 10.00 | 0.378 | 54.8 | 106.36 | 45.64 | 0.0 / 0.0 | 1.042 |
| **result** | moving (baseline failure) [348-379] | **3.40** | **0.657** | 0.0 | 1.39 | 0.16 | -5.9 / -5.9 | 0.847 |
| baseline | moving (back) [380-468] | 1.70 | 0.702 | 3.3 | 1.46 | 6.20 | 0.0 / 0.0 | 0.851 |
| **result** | moving (back) [380-468] | **1.19** | 0.710 | 0.0 | 0.30 | 0.12 | -0.9 / -1.0 | 0.843 |
| baseline | static-2 [469-545] | 1.62 | 0.744 | 5.6 | 1.31 | 6.61 | 0.0 / 0.0 | 0.836 |
| **result** | static-2 [469-545] | **0.08** | 0.694 | 0.0 | 0.00 | 0.00 | 1.6 / 0.0 | 0.841 |
| baseline | all | 1.80 | 0.789 | 4.3 | 4.92 | 9.69 | 0.0 / 0.0 | 0.852 |
| **result** | all | **0.70** | 0.794 | 0.0 | 1.78 | 0.11 | 0.1 / -0.4 | 0.849 |

Notes:
- **Slightly lower IoU in the static phases is expected.** It comes from re-anchoring to the baseline depth: before re-anchoring, the result's IoU in static 2 was 0.728. This is the size/depth tension described above. The task says image-space checks cannot judge depth, so it was accepted deliberately.
- **Pooled yaw scans agree with the result.** In static 1 the silhouette cost has a clean minimum at the result's yaw (+0.9° relative to the baseline frame-0 orientation; the baseline has about 0°). In static 2 the silhouette minimum is at −175° to −177° and the texture-gradient correlation peaks at −179°. The result has −177.7° and the baseline −180° to −182°.
- **Overall difference from the baseline:** mean ADD 4.8 cm. Per phase: 3.5 (static 1), 3.5 (moving front), 25 (failure segment), 3.1 (moving back), 4.2 (static 2) cm. Outside the failure segment this is mostly the removed tilt and jitter plus a 1–1.5 cm lateral image-position correction.

## Confidence per phase
- **Static 1, high.** Yaw is ±2°. Image position is ±1 cm. Depth is the baseline median, so it is as good as the baseline's calibration; the image size suggests about 14 cm closer and the table cue suggests farther.
- **Moving, front (164–347), medium.** Near-frontal views (about 228–320) leave yaw ambiguous by about ±10°. There the result stays close to a smoothed baseline: mean |Δyaw| is 2.5°. Flow consistency is better than the baseline's.
- **Baseline failure (348–379), medium.** The result is upright on the table and rotates smoothly from −150° to −174°. It matches the edge-on silhouette visually, but the side view under the hand is hard; yaw is probably within ±10°. It is far better than the baseline there.
- **Moving, back (380–468), medium-high. Static 2, high** for rotation (±3°); depth as in static 1.

## Reproducibility notes
- **Run-to-run variation:** nvdiffrast's backward pass uses atomics, so two runs of `solve.py` differ slightly. Measured between two full runs: mean 0.11 cm ADD, max 0.87 cm. Seeds are fixed and all inits are deterministic.
- **What `solve.py` derives at run time:** everything (table plane, phases, costs, decoding, re-anchoring) is recomputed from `--inputs`.
- **Constants:** the only hand-set constants are the cost/smoothing weights, the motion threshold 1.3 (static object noise is about 0.8; moving is above 3) and the fixed image crop that contains the monitor in every frame. All are commented in the code.
- **Diagnostic only:** the caster and table-top measurements were made by eye. They were used only for the depth diagnosis above and are not part of `solve.py`.
- **Files:** `hoi_fit.py` (model, evidence, decoding), `hoi_render.py` (renderer), `hoi_validate.py` and `validate.py` (checks), `solve_diagnostics.npz` (grid costs, phases, motion score, depth ratios).
