# Episode 21 – tall bar stool: trajectory refinement

Command: `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz [--validate]`
(code: `solve.py`, `fit.py`, `stages.py`, `common.py`; log of the final run: `solve_log.txt`).
Final run: 278 s wall-clock including the `--validate` held-out refit. Without `--validate` it takes about 2–2.5 min, of which the joint fit is 97–155 s depending on GPU sharing. The GPU stays under about 4.5 GB.
Total GPU time over the whole project, including experiments: about 30 min.

## Diagnosis of the baseline
- **No symmetry flips.** I rotated the whole baseline by 180° about each mesh axis and by ±90° about the vertical axis. Mean mask IoU over every 4th frame was 0.81 for the baseline against at most 0.56 for any flip (z90). Visual overlays (backrest direction, legs) agree. The baseline's yaw curve is continuous; the largest frame-to-frame jump is 6.6° (frames 530–534, real fast rotation).
- **Jitter and drift in static phases.** While the stool is untouched, the baseline still moves: up to about 5 cm in position over frames 0–119 and about 1.5 cm over 582–677, with 0.2–0.4° of rotation jitter per frame.
- **Systematic misfit.** The silhouette shows a consistent misfit of a few px and 1–2° (legs offset, backrest too wide). It is worst when the person stands next to or in front of the stool: frames 317–374 have IoU 0.74, and frames 430–470 have IoU 0.57–0.72.
- **Tilt in static phases.** I took the plane through the 4 mesh feet. Against the released floor normal, the baseline's tilt is 2.4° in phase 0 and 0.5–0.9° elsewhere. A stool standing on its four feet on a flat floor should have the same, near-zero tilt in every static phase.
- **Depth.** Fitting the silhouette with depth left free agrees with the baseline depth to within 1 cm in 6 of 7 phases (details below). I found no evidence of a depth bias I could correct.

## Phases (automatic: photometric stability of the object region)
For each frame pair (i, i+3) I take the mean absolute grey-level difference inside the dilated object mask, excluding dilated person pixels, at half resolution. The noise floor is 0.3–0.9 grey levels; motion gives 2–20. The threshold is 1.2. The boundaries were checked by eye on difference images. For example, frames 582–600 look static even though SAM-mask IoU flickers there, which is why I used photometry rather than mask overlap.

| # | frames | phase |
|---|---|---|
| 0 | 0–119 | static: at the desk, untouched (person walks over) |
| 1 | 120–316 | moving: pulled out and turned (yaw −49° → −134° → −51°) |
| 2 | 317–374 | static: hands touch it at 317–330 and 360–374, but it does not move |
| 3 | 375–482 | moving: pushed back to the desk, heavily occluded by the person at 425–470 |
| 4 | 483–509 | static: at the desk, hand on the backrest |
| 5 | 510–581 | moving: pulled away to the left, turned by about 130° |
| 6 | 582–677 | static: released |

## Method
1. **Hypotheses.** I scored global flip hypotheses (see the diagnosis above). A flip would only be adopted if it beat the baseline by 0.05 IoU; none did.
2. **Joint render-and-compare fit of all 678 frames.** This runs at half resolution with nvdiffrast.
   - **Data term:** soft IoU between the antialiased silhouette and the SAM2 object mask. Pixels that are person but not object are ignored (occlusion).
   - **Pose parameters:** each static phase has one shared pose, so evidence from all its frames is pooled. Each moving frame has its own pose.
   - **Smoothness:** second differences of rotation matrices (weight 100) and centres (weight 1000) over consecutive frames. These terms also tie static phases to their neighbours.
   - **Translation parametrisation:** image-plane offset plus log depth. The poorly observed depth direction gets a smaller step size and a prior. An earlier version with xyz translation and no prior drifted 15 cm toward the camera in occluded phase 4, so I changed it.
   - **Priors:** weak priors toward the baseline on rotation and depth.
   - **Physical prior:** the feet plane should be parallel to the floor, weighted 2 per frame in static phases and 0.5 per frame while moving.
   - **Optimiser:** Adam, 200 iterations, cosine step-size schedule.
   - **Ambiguity check:** in phase 4 a separate static fit could not tell the true yaw from yaw + 90° (both silhouettes look similar when half-occluded). The joint fit settles this through temporal continuity with the visible moving phases.
3. **Depth re-anchoring.** For each phase, I scale the camera-frame object centres along their viewing rays so that their median depth equals the baseline's median over the same frames, then map back to world. This keeps the baseline's calibrated placement. The scale factors are in the table below.

## Validation (no ground truth; from the final run, `object_poses_world_validation.md`)
For the held-out columns, the same fit was rerun with every odd frame given zero data weight. Odd frames are then predicted only by the shared static pose or by the smoothness terms, and their IoU is measured against the SAM masks.

| phase | frames | type | IoU base | IoU ours | held-out IoU base | held-out IoU ours | rot change med/max (°) | depth-anchor factor | dz ours−base mean/median (cm) | frame-to-frame rot jitter base/ours (°, median) | tilt vs floor base/ours (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0-119 | static | 0.873 | 0.883 | 0.874 | 0.877 | 2.30/3.22 | 0.9974 | 0.46/0.29 | 0.31/0.00 | 2.42/0.23 |
| 1 | 120-316 | moving | 0.835 | 0.902 | 0.836 | 0.901 | 0.43/2.75 | 1.0001 | -0.01/0.01 | 0.70/0.62 | 0.49/0.45 |
| 2 | 317-374 | static | 0.740 | 0.898 | 0.740 | 0.897 | 0.68/1.24 | 1.0012 | 0.06/0.09 | 0.35/0.00 | 0.50/0.15 |
| 3 | 375-482 | moving | 0.754 | 0.867 | 0.753 | 0.865 | 0.69/2.50 | 1.0015 | 0.38/0.36 | 1.23/0.87 | 0.97/0.96 |
| 4 | 483-509 | static | 0.838 | 0.861 | 0.839 | 0.864 | 0.83/1.29 | 0.9974 | 0.12/-0.01 | 0.44/0.01 | 0.81/0.37 |
| 5 | 510-581 | moving | 0.816 | 0.837 | 0.816 | 0.834 | 0.64/3.62 | 0.9969 | -1.02/-1.01 | 1.64/1.97 | 0.70/0.41 |
| 6 | 582-677 | static | 0.789 | 0.836 | 0.789 | 0.836 | 1.16/1.47 | 0.9919 | -0.41/-0.15 | 0.29/0.03 | 0.92/0.26 |
| all | 0-677 | | 0.812 | 0.875 | 0.813 | 0.873 | 0.78/3.62 | | -0.02/0.01 | | |

How to read the table:
- **dz** is measured at the mesh origin (`T[:,2,3]` in the camera frame). Re-anchoring is done on the mesh bounding-box centre, so dz is not exactly zero. The ±1 cm in phase 5 comes from the 9 cm lever between the origin and the centre combined with the rotation change.
- **Frame-to-frame rotation in phase 5** is mostly real fast rotation (about 1.9°/frame), so this column is not purely jitter.
- **Other checks:**
  - The lowest mesh point stays within −8 to +6 mm of the floor plane in every phase (baseline: −7 to +9 mm).
  - The mean ADD between our trajectory and the baseline is 1.1 cm (median), 2.1 cm (90th percentile) and 4.6 cm (max).
  - Two full runs differ by 0.01 cm median and 0.37 cm max ADD. Seeds are fixed, but nvdiffrast's antialias backward pass uses atomic adds, so runs are not bit-identical.

## Depth along the viewing ray: cues and decision
| cue | suggested shift (+ = farther), per phase 0..6 |
|---|---|
| metric mesh size vs. silhouette size (free-depth fit) | +0.8, 0.0, −0.3, −0.4, +0.8, +0.9, +2.3 cm |
| untouched / resting stool on the released floor plane | about 0 (the lowest point is already within ±1 cm of the plane; a 1 cm height error corresponds to about 2 cm along the ray, inside the plane's several-cm tolerance) |
| hand contact | not evaluated |

No two independent cues agree on a correction of more than 2 cm, so the **applied depth shift is 0** and the trajectory is re-anchored to the baseline's median depth in every phase.

## Confidence per phase
- **Phases 1, 2 and 6 – high.** Large, consistent held-out IoU gains (+0.05 to +0.16). Static poses are pooled over 58–120 frames. Changes are about 1° and a few mm.
- **Phase 0 – medium.** The IoU gain is small (+0.003 held out). The 2.3° rotation change is driven mostly by the feet-on-floor prior: tilt goes from 2.4° to 0.2°, which matches the other static phases.
- **Phases 3 and 5 – medium.** These are occluded moving phases. IoU gains are clear, but in frames 425–470 only one or two legs are visible, so the poses there come mainly from smoothness and the baseline prior.
- **Phase 4 – medium.** This phase is half-occluded. A standalone fit was ambiguous (yaw + 90°); the joint fit stays within 0.8° of the baseline yaw and improves held-out IoU by 0.025.

## What I did not do
- I did not add texture/edge or optical-flow terms; the stool is mostly uniform brown leather.
- I did not re-run FoundationPose.
- I did not use hand-object attachment.
- I did not recompute the masks. The SAM masks draw the footrest rungs thicker than the mesh, which slightly biases the IoU.
