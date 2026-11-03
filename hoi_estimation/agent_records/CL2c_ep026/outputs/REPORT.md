# Episode 26: skateboard rotated on the ground. Trajectory-level pose re-estimation

**Result:** `object_poses_world.npz` (`T_world_mesh`, 669×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(one deterministic run: 5 min 42 s wall time, about 3.5 GB of GPU memory).

## 1. Main finding: the baseline is turned 180° nose/tail for the whole episode

The skateboard looks almost the same under a 180° rotation about its vertical axis. But the mesh has one truck
with **green** wheels and one with **grey** wheels, plus a directional "MINECRAFT" logo on the grip tape. The
baseline is upright and smooth, but over the whole episode it puts the green wheels at the wrong end. The
corrected pose is `T_base @ diag(-1, 1, -1, 1)`, i.e. a 180° rotation about the mesh y axis, which is the floor
normal. For a 78 cm board this error alone puts the ends about 75 cm off, so the baseline fails ADD at every
threshold in every frame.

The evidence comes from four independent lines:

1. **Wheel-colour test** (not part of the optimisation objective, which is greyscale only). I compute the image
   green-excess `G - (R+B)/2` at the pixels where the rendered model shows the green wheels, minus the same
   quantity at the rendered grey wheels. With the baseline pose this is negative in all 30-frame windows
   (≈ −0.02 to −0.05). With the flipped pose it is positive in all of them (≈ +0.02 to +0.05). In the final
   validation table (section 5) the mean is −0.013 for the baseline and +0.041 for the result.
2. **Texture NCC with a ±4 px shift search.** The flip wins in every phase, both on the logo region alone and
   on the whole object.
   - A first test *without* the shift search favoured the baseline in phase S1. The cause was misregistration:
     the flip rotates about the mesh origin, so the small logo landed a few pixels off. Fitting a local pose
     removes this.
3. **Pooled render-and-compare score per phase in `solve.py`.** Each hypothesis is locally refined, then
   scored as IoU + NCC (the log of the final run is quoted here):

   | phase | baseline (`id`) | flip about mesh y | flip about mesh x (wheels up) | flip about mesh z (wheels up) |
   |---|---|---|---|---|
   | S1 0–280 | 1.557 | **1.644** | 1.097 | 1.052 |
   | M1 281–356 | 1.620 | **1.660** | 1.131 | 1.125 |
   | S2 357–398 | 1.492 | **1.557** | 1.081 | 1.111 |
   | M2 399–509 | 1.568 | **1.605** | 1.040 | 1.053 |
   | S3 510–668 | 1.570 | **1.598** | 1.272 | 1.217 |

   In a per-frame run the flip won in 100% / 73% / 100% / 100% / 96% of the frames of S1 / M1 / S2 / M2 / S3.
4. **Visual check.** I compared zoomed crops at frames 0, 100, 380 and 600 against both renders. In the video
   the near-end wheel is green (very clear at 330–600, faint at 0–250). The baseline renders grey there.

## 2. Phases

Phases are found automatically from the object-mask change over 3 frames, ignoring pixels covered by the person.
The threshold is 0.06; static noise is 0.00–0.05, while motion gives 0.1–0.37. Moving segments are padded by
4 frames.

| phase | frames | content |
|---|---|---|
| S1 static | 0–280 | board on the floor, untouched; the person walks up and from about 260 puts the hands on it |
| M1 moving | 281–356 | the board is turned on the floor, about −146° of yaw, and slides about 12 cm |
| S2 static | 357–398 | hands on the board, board not moving |
| M2 moving | 399–509 | turned again, about −275° of yaw; between about 455 and 482 one side is lifted (≈3° roll) |
| S3 static | 510–668 | board released, at rest |

These agree with the contact probabilities and with the mask motion signal.

## 3. Method (`solve.py`)

- **Renderer and scorer.** nvdiffrast renders the textured mesh into a crop around the object at 2× and pools
  it down (anti-aliasing). The score is:
  - silhouette IoU with the SAM2 object mask, counted only on pixels *not* covered by the person mask, plus
  - normalised cross-correlation of greyscale texture inside the rendered silhouette, again outside the person.
- **Board-on-floor model.** The baseline keeps the mesh y axis along the floor normal; this is checked. Each pose
  is: a yaw about the floor normal, a 2-D position on a plane parallel to the released floor at a fixed height
  `h0` (the median baseline height), and, in moving phases only, a roll about the board's long axis. The roll
  pivots on the wheels that stay on the floor, so the lowest point keeps its height.
  - With the height fixed, the image position determines the depth along the ray. The relative depth
    between phases therefore comes from the floor geometry instead of the per-frame depth corrections.
- **Hypotheses.** For every phase: the baseline and its 180° flips about the mesh x, y and z axes, each refined
  by a local search and pooled over the frames of the phase. The best yaw-type hypothesis is kept; the
  wheels-up flips score far lower.
- **Static phases.** One pose per phase, from a coarse-to-fine grid search over yaw, a and b with scores pooled
  over every 3rd frame. Grids: ±12° by 3°, ±4.5 cm by 1.5 cm; then ±2° by 1°, ±1 cm by 0.5 cm; then ±0.5°, ±2.5 mm.
- **Moving phases.** Per frame, the same coarse-to-fine search starts from the flip-corrected baseline. Then:
  - a roll search (−6° to +6°) that re-fits the planar position for each candidate, with a 0.003-per-degree
    penalty;
  - a final fine (yaw, a, b) search.
- **Temporal decoding.** A weighted sparse least-squares solve over all frames (yaw unwrapped, a, b, pitch, roll):
  - data terms in moving frames, weighted by the frame's score;
  - static frames tied to their phase pose;
  - a second-difference smoothness penalty λ = 3.

  λ was chosen by **held-out validation**. Every 4th moving frame's data term is dropped, and the smoothed pose
  at that frame is scored against the image:

  | λ | 0.3 | 1 | 3 | 10 | 30 | 100 | 300 | 1000 |
  |---|---|---|---|---|---|---|---|---|
  | held-out score | 1.6381 | 1.6383 | 1.6382 | 1.6368 | 1.6316 | 1.6136 | 1.5729 | 1.4962 |

  For reference, the per-frame fit scores 1.6361 on those same frames, so the smoothed prediction beats the
  direct fit (grid quantisation). Residuals of the per-frame fits around the smooth trajectory are 0.3° in yaw
  and 0.15 cm in position. I found no jumps.
- **Depth re-anchoring.** Camera-frame positions are scaled along the viewing rays so that the episode median
  camera z equals the baseline's. The scale is 1.0013. This keeps the baseline's overall placement.
- **Decisions encoded as constants** (each commented in `solve.py`):
  - motion threshold 0.06;
  - tilt penalty 0.003 per degree;
  - pitch is not searched: in a development run it was selected (≤1°) in only 5 of 187 moving frames;
  - flip-test frame stride 20.

  No per-frame results are embedded; everything is recomputed from the inputs.

## 4. Depth: cues, decision and per-phase changes

Each cue below is the camera-z shift it suggests for the refined pose, in cm (negative = closer to the camera).

| cue | S1 | S2 | S3 |
|---|---|---|---|
| wheels touching the released floor plane | −6.8 | −2.8 | −10.1 |
| metric mesh size vs SAM mask (best IoU scale) | −11.4 | −2.9 | −8.8 |
| metric mesh size vs image edges (gradient on rendered contour; no SAM mask) | −2.9 | +5.7 | −5.9 |
| hand contact (MHR hand keypoints vs board surface) | – | flat from −8 to +1 (no constraint) | – |
| texture NCC | ≈0 | ≈+1 | ≈−1 |

- The mask-based size cue is biased: the masks are slightly inflated, which makes the board look bigger and
  therefore closer.
- The hand cue is not independent anyway, because the pipeline's post-optimisation pulled hands and object
  together.
- Floor and edge-size cues disagree by 4–8.5 cm in every phase, more than the 2 cm required.

**Absolute depth shift applied: 0** (episode median z equals the baseline's).

The *relative* depth between phases does change. The board rests on a flat floor in every phase, but the
baseline's per-phase depths put its wheels at different heights relative to the floor plane: −2.6 / −0.7 / −1.1 /
−1.6 / −4.8 cm in S1 / M1 / S2 / M2 / S3. That is physically inconsistent and corresponds to ±4 cm of depth
noise. In my result the wheels are at a constant −2.9 cm, and the episode-level anchor keeps the overall depth.
The resulting per-phase camera-depth differences from the baseline are the `dz` columns below.

## 5. Validation table (baseline vs result; every 3rd frame)

Columns:
- IoU and NCC are the optimisation terms.
- Wheel-colour and edge strength are independent checks (higher is better; wheel-colour > 0 means the green
  wheels are rendered where the image is green).
- dz is result minus baseline camera-frame depth.

| phase | traj | IoU | NCC | wheel-colour | edge grad | dz mean cm | dz median cm | median rot. diff to baseline (°) | mean frame-to-frame rot. (°) | mean frame-to-frame transl. (cm) | lowest wheel height vs floor plane (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 static | baseline | 0.861 | 0.515 | −0.016 | 0.3 | 0 | 0 | – | 0.40 | 0.93 | −2.6 |
| S1 static | **result** | **0.886** | **0.785** | **+0.031** | 0.4 | −0.22 | −1.41 | 179.5 | 0.03 | 0.00 | −2.9 |
| M1 rotate | baseline | 0.764 | 0.112 | −0.003 | 0.3 | 0 | 0 | – | 2.01 | 0.77 | −0.7 |
| M1 rotate | **result** | **0.845** | **0.803** | **+0.040** | 0.4 | +2.39 | +1.88 | 178.8 | 1.97 | 0.24 | −2.9 |
| S2 static | baseline | 0.766 | 0.190 | +0.005 | 0.3 | 0 | 0 | – | 0.42 | 0.78 | −1.1 |
| S2 static | **result** | **0.809** | **0.749** | **+0.031** | 0.4 | +2.67 | +2.82 | 176.2 | 0.03 | 0.00 | −2.9 |
| M2 rotate | baseline | 0.790 | 0.284 | −0.009 | 0.3 | 0 | 0 | – | 2.63 | 0.97 | −1.6 |
| M2 rotate | **result** | **0.850** | **0.805** | **+0.041** | 0.4 | +2.03 | +2.41 | 178.5 | 2.51 | 0.14 | −2.9 |
| S3 static | baseline | 0.831 | 0.599 | −0.017 | 0.4 | 0 | 0 | – | 0.46 | 0.73 | −4.8 |
| S3 static | **result** | **0.849** | **0.797** | **+0.061** | 0.4 | −4.44 | −4.55 | 178.6 | 0.03 | 0.00 | −2.9 |
| ALL | baseline | 0.825 | 0.431 | −0.013 | 0.3 | 0 | 0 | – | 0.96 | 0.86 | −2.7 |
| ALL | **result** | **0.862** | **0.791** | **+0.041** | 0.4 | −0.37 | −0.57 | 179.1 | 0.66 | 0.05 | −2.9 |

- Edge gradient is mean |∇I| on the rendered contour, with intensities in [0, 1], rounded.
- The "rot. diff to baseline" column is ≈180° by construction: it is the flip. The remaining differences
  (0.5–4°) are yaw refinements.
- In the static phases the result has no frame-to-frame jitter; the baseline jitters by 0.4° and 0.8–0.9 cm per
  frame.

## 6. Confidence per phase

| phase | flip correction | rotation, after the flip | relative position | absolute depth |
|---|---|---|---|---|
| S1 | very high | high, about 1° | high | as the baseline (episode anchor) |
| M1 | high | good, about 1–2° | good | as the baseline ±2–3 cm |
| S2 | high | good | good | as the baseline ±3 cm |
| M2 | high | good, about 1–2° | good | as the baseline ±2–3 cm |
| S3 | very high | high | high | as the baseline −4.5 cm |

- **Flip evidence by phase.** It is strongest in S1 and S3: the board is unoccluded, the margin is large, and
  the wheel-colour signal is clear (in S3 the green wheel is directly visible). In M1/M2 the hands occlude the
  board, but the trajectory is continuous with the static phases.
- **Roll in M2.** The ≈3° roll at 455–482 is supported by the image (+0.1 score) and moves points by less than
  1 cm.
- **Remaining risk: absolute depth.** It is kept at the baseline's episode median. Both the floor cue and the
  size cues suggest the board may really be about 3–10 cm closer, but the cues do not agree well enough to apply
  a change.
- **Relative per-phase depths** now follow the flat-floor model instead of the baseline's per-phase depths. If
  the baseline's per-phase depths were the better ones, S3 would be about 4.5 cm off. I consider this unlikely,
  because the board rests on the same flat floor throughout.

## 7. Run time and resources

- **`solve.py`:** 5 min 42 s wall time. That is about 1 min of video decoding and crops, then GPU work: flip test
  ≈ 1 min, static phases ≈ 1 min, moving phases ≈ 3 min. Peak GPU memory is about 3.5 GB, on a GPU shared with
  other jobs.
- **Total GPU time for the whole project, including development runs:** about 40 min (estimated, under the
  45 min budget).
- **External tools:** none (see TOOLS.md).
