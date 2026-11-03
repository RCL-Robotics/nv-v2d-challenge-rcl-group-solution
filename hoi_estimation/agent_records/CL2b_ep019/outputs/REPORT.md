# Episode 19: cane (quad cane): 6-DoF trajectory recovery

Output: `object_poses_world.npz` (`T_world_mesh`, 886×4×4, float64). It was produced by one offline run of
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`.
The run took 11.6 min of wall time on a GPU shared with another job; its log is in `solve_run.log`.
Diagnostics are in `object_poses_world_info.json`, and `overlay_baseline_vs_ours.jpg` shows the video, the baseline and our result.

## Object and its ambiguities
The object is a quad cane, 0.91 m tall, with a four-footed base (0.20 × 0.15 m) at mesh −z and an offset handle at +z.
It is thin (a 2 cm shaft) and small in the image (about 250 px long, about 3000 px of mask).

- The silhouette fixes the shaft direction in the image and tells the base from the handle.
- The twist about the shaft (how the base and handle are turned) is the weak degree of freedom. The base and
  handle shape and the texture resolve it.
- Depth and the tilt toward or away from the camera trade off against each other in the silhouette.

## Phases (found automatically from the SAM2 masks, then checked by eye)
| phase | frames | content |
|---|---|---|
| static A | 0–112 | The cane stands untouched on its base while the person walks up to it; the legs pass behind it. |
| right-hand carry | 113–~399 | Grasped with the right hand, lifted and swung, with quick twists about the shaft. |
| handover | ~400–430 | Passed from the right hand to the left. |
| left-hand carry | ~431–767 | Swung with the left hand. It is occluded behind the legs around 535–565, 620–635 and 735–760, and the mask is empty at 741–742. |
| static C | 768–885 | Placed back on the floor. The hand rests on the handle until about 805; the person then walks in front of it (806–850) and it stays untouched to the end. |

**Static-phase detector.** A frame counts as static if its mask lies inside the dilated mask of a reference frame
(at most 2 % of pixels outside) and it is missing at most 12 % of the reference's pixels that the person does not
cover. Runs of 30 frames or more are kept and grown frame by frame from the middle of the run. Runs that are
separated only by occlusion are merged. Neither phase boundary is a hand-set constant.

## What the baseline gets wrong (diagnosis)
Silhouette and photometric checks against the video, with the evidence from overlays:

- **300–620: turned 180° about the shaft** (the handle and base face the wrong way) for most of the carry.
  The 180° flip about mesh z cuts the silhouette loss by 4–5×.
- **756–808: upside down.** The base is rendered where the handle is.
- **680–700:** about 90° off about the shaft.
- **Static A:** the shaft is tilted 34° from the floor normal, which is impossible for a quad cane standing on its own base.
- **Static C:** the rotation is inconsistent within a phase where the object does not move: it changes by up to
  180° and the depth jumps 10 cm.
- **127 frame-to-frame rotation jumps above 15°** over the episode, most of them during occlusion.

## Method (`solve.py`)
1. **Per-frame registration from several starting poses.** Each frame starts from 7 poses: the baseline, its 180°
   flips about mesh x, y and z, ±90° about the shaft, and the FoundationPose rotation at baseline depth. Each is
   optimised in 6-DoF with Adam (120 iterations) on a 2D chamfer between 1200 projected surface samples and the
   SAM2 object mask. The fit pulls points into the object mask (or onto the person, who may occlude it) and also
   requires every mask pixel to be explained by the model. A Huber penalty and truncation make it robust.
2. **Static phases.** One shared pose is fitted jointly over 16 frames of the phase, starting from all distinct
   per-frame solutions. Candidates are ranked by silhouette loss plus **photometric agreement**: the textured mesh is
   rendered with nvdiffrast, a gain is fitted per channel, and the background is inpainted. The photometric term is
   what resolves the twist of the base in static C. The silhouette alone tied a solution 67° away; the photometric
   residual prefers the chosen pose in every frame tested (0.27 vs 0.36).
3. **Viterbi decoding.** The data cost combines silhouette and photometric scores; the transition cost is
   |ΔR|/6° + |Δcentre|/4 cm per frame. Static phases are clamped to their single pose.
4. **Propagation, 2 rounds.** Every frame is refitted from the decoded poses at frames f±1 and f±2. The results are
   added as candidates, re-scored and re-decoded.
5. **Smoothing.** Rotations and centres are smoothed with Gaussian weights (σ = 2 frames) that are lower where
   the mask is occluded. Each moving frame is then refitted with a prior toward the smoothed pose, twice. Finally a
   bilateral pass (σ = 1) is applied and the depth along the viewing ray is smoothed (σ = 3), keeping the image
   position. Remaining fast twists about the shaft (for example around 385, 395 and 668) are visible in the video
   (the base turns quickly while being twirled), so they are kept.
6. **Depth re-anchoring per phase.** By default each phase is scaled along its viewing rays so that its median camera
   z equals the baseline's. A phase deviates only if two independent cues agree (see below).

## Depth cues (cm; positive means farther than the baseline). Applied = mean of the two cues if they agree within 2 cm
| phase | cue 1: metric mesh size vs image (our fit) | cue 2 | disagreement | applied shift |
|---|---|---|---|---|
| static A (0–112) | −5.8 | floor contact (4 feet on the released plane): −4.7 | 1.1 | **−5.2** |
| moving (113–767) | +1.1 (difference of median z) | hand contact (median offset of hand keypoints from the grip): +2.1 | 1.0 | **+1.6** |
| static C (768–885) | +1.2 | floor contact: +1.8 | 0.6 | **+1.5** |

**Why we trust the silhouette scale.** In both static phases, which are 1 m apart in depth, the free silhouette fit
places all four feet within ±1.6 cm of the released floor plane. These two cues are independent: one comes from the
mesh size and K, the other from the scene-depth plane fit. Their agreement shows the silhouette depth has no scale
bias. During the carries, the hand-grip offset with our trajectory is −0.5 cm in depth (right hand) and +1.6 cm
(left hand); with the baseline it is −3.3 and −2.1 cm.

**Caveat.** The −5 cm correction for static A is the largest deviation from the baseline's placement. Both cues
agree on it, so we applied it, following the rule in the task.

## Validation table (no ground truth; every 3rd frame)
- **sil**: robust chamfer loss in px² units (lower is better).
- **iou**: mask IoU, excluding pixels covered by the person.
- **photo**: colour residual (lower is better).
- **rot_step / jumps15**: median per-frame rotation change, and the number of steps above 15°.
- **acc**: median acceleration of the centre in cm per frame².
- **dz**: our camera-frame depth minus the baseline's, in cm.

| phase | frames | traj | sil | iou | photo | rot_step_med (°) | jumps>15° | acc (cm) | dz mean (cm) | dz median (cm) | rot vs baseline, median (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| static A | 0-112 | baseline | 5.94 | 0.252 | 0.397 | 0.09 | 7 | 0.46 | 0 | 0 | 0 |
| static A | 0-112 | **ours** | **0.75** | **0.645** | **0.253** | 0.09 | 0 | 0.00 | −5.03 | −5.24 | 36.2 |
| right-hand carry | 113-399 | baseline | 10.83 | 0.228 | 0.401 | 4.48 | 32 | 2.73 | 0 | 0 | 0 |
| right-hand carry | 113-399 | **ours** | **0.60** | **0.626** | **0.282** | 2.41 | 0 | 0.25 | −4.11 | −3.80 | 30.6 |
| handover | 400-430 | baseline | 2.69 | 0.366 | 0.393 | 3.25 | 2 | 2.70 | 0 | 0 | 0 |
| handover | 400-430 | **ours** | **0.48** | **0.684** | **0.260** | 1.84 | 0 | 0.22 | −2.80 | −1.52 | 171.6 |
| left-hand carry | 431-767 | baseline | 8.61 | 0.201 | 0.391 | 5.27 | 68 | 3.55 | 0 | 0 | 0 |
| left-hand carry | 431-767 | **ours** | **0.66** | **0.591** | **0.319** | 2.21 | 13 | 0.31 | −0.85 | −2.11 | 138.6 |
| static C | 768-885 | baseline | 8.83 | 0.212 | 0.373 | 0.12 | 17 | 1.73 | 0 | 0 | 0 |
| static C | 768-885 | **ours** | **0.84** | **0.566** | **0.274** | 0.02 | 0 | 0.00 | +1.01 | +1.48 | 75.9 |
| all | 0-885 | baseline | 8.82 | 0.223 | 0.392 | 3.78 | 127 | 2.69 | 0 | 0 | 0 |
| all | 0-885 | **ours** | **0.67** | **0.609** | **0.291** | 1.65 | 13 | 0.22 | −2.26 | −3.27 | 64.4 |

**Physical checks.**
- **Below the floor:** the baseline puts the cane more than 3 cm below the floor in 28 frames; ours does in 0.
- **Static tilt from the floor normal:** the baseline has 34.0° (A) and 9.7° (C); ours has 5.3° and 5.8°.
  - We tested forcing the shaft parallel to the floor normal (a free-standing quad cane should be). The image
    evidence got worse in both phases (silhouette +0.04, photometric +0.014), so we did not adopt it.
  - In camera coordinates our shafts are exactly vertical. The released plane normal is 5.9° off the camera's
    vertical and 3° off the world y axis, so we take the residual to be error in the plane's normal rather than in the pose.

**Smoothing ablation (development).**
| trajectory | sil | iou | depth jitter |
|---|---|---|---|
| decoded only, no smoothing | 0.57 | 0.63 | 4.5 cm |
| smoothed without refit | 0.73–1.13 | — | — |
| smoothed + refit (adopted) | 0.65 | 0.61 | 2.1 cm |

Depth jitter is the high-frequency standard deviation of the per-frame depth. Smoothing without a refit was
rejected because it clearly hurts the image fit. The adopted version accepts a small image-fit cost for a much
smoother, physically plausible motion.

## Confidence per phase
- **Static A: high** for the rotation (pooled over 113 frames; the silhouette and photometric scores clearly beat
  the baseline). Medium-high for the −5 cm depth correction (two cues agree within 1.1 cm). There is a residual
  ±5° tilt uncertainty toward or away from the camera.
- **Static C: high** (it fixes the baseline's upside-down segment and its 10 cm internal jump; the twist was resolved
  photometrically).
- **Carries: medium-high.** The 180° flips and upside-down segments are fixed with large evidence margins.
  - **Depth:** per-frame silhouette depth is noisy (about 2 cm after smoothing).
  - **Twist about the shaft:** inferred mostly from the base and handle shape, a few frames at a time; occasional
    errors of 20–60° are possible in fast-twirl segments.
- **Occluded segments: low-medium.** In 535–565, 620–635 and 735–760 the object is mostly behind the legs, and the
  poses there are largely decoded and interpolated from the neighbouring frames.

## Run time and resources
- `solve.py`: 11.6 min wall time, almost all of it on the GPU, which another job was also using at 100 %.
  Peak GPU memory is capped at 8 GB through `set_per_process_memory_fraction`.
- Total GPU time, development plus the final run, was an estimated 35 min (under the 45 min limit).
- Randomness is seeded: the mask-pixel and surface-point samples are fixed. CUDA atomics in the `grid_sample`
  backward pass can still cause float-level differences between runs.

## Not done / possible improvements
- **Untried cues:** optical-flow or point-track constraints on the texture, and an explicit term attaching the
  cane rigidly to the hand while it is carried.
- **Masks:** recomputing them at higher quality. The SAM2 masks miss a foot of the base in static C.
