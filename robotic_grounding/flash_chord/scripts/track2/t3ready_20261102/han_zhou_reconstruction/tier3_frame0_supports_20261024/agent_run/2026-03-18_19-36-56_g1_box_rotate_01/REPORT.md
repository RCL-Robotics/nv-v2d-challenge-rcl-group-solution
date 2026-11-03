# Episode 5 (big_black_box): object trajectory recovery

**Result:** `object_poses_world.npz` (`T_world_mesh`, 1003×4×4, float64). It was produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`, which takes
185 s wall time, about 140 s of it GPU optimisation, with peak GPU memory under 2.5 GB. All my GPU work in this
session, including experiments, came to about 20 minutes.

## 1. What the video shows and how good the baseline is

A person walks up to a black flight case (52×40×86 cm) that is standing upright on its castors. She tips it over,
lays it on its side, turns it on the floor, and finally stands it upright again on its castors. The case touches
the floor throughout and is never carried.

What I found in the baseline:
* **No 180° flips and no jumps.**
  * I checked visually: the trolley handle is on a large face and the castors on one end, and both match the video
    on the frames I inspected.
  * I also checked automatically (`validation` §4): each 180° flip about an OBB axis loses on every phase, both in
    mask IoU (by at least 0.015) and in edge alignment (by at least 0.4 px).
  * The largest per-frame rotation step is 5°, with a median of 0.57°/frame.
* **Rotation:** this is the CoCoNet rotation up to a constant 1.4° camera offset. It is about 1–3° away from what
  the image supports, mostly yaw: 1.5° median correction, 4.6° maximum.
* **Image position:** this is CoCoNet's, accurate to about 1 px.
* **Depth along the viewing ray:** this is the main weakness. Baseline depth is CoCoNet depth times a slowly
  varying correction (0.90–0.985). It has **2–4 cm frame-to-frame jitter**, even while the object is static: in
  static phases the depth std is 2.0 cm (frames 0–158), 2.6 cm (583–713) and 3.7 cm (340–450). That jitter makes
  the object breathe back and forth along the ray.

## 2. Phases

The phases are found automatically. A frame counts as static when the SAM2 object mask, with person pixels
excluded, has IoU ≥ 0.98 with the masks 10 frames before and after. Static runs must be at least 20 frames long.
I checked the boundaries by eye on crop sheets.

| phase | frames | content |
|---|---|---|
| static | 0–158 | upright on castors; the person walks in and touches it from about frame 120 |
| moving | 159–345 | tipped over and laid on its side |
| static | 346–365 | lying, briefly at rest |
| moving | 366–421 | small pushes while lying |
| static | 422–448 | lying, at rest |
| moving | 449–582 | turned on the floor (fast rotation, heavy hand occlusion) |
| static | 583–713 | lying, untouched or lightly touched |
| moving | 714–937 | slow nudges, then stood upright (frames 790–905), then settles |
| static | 938–1002 | upright, at rest |

## 3. Method (`solve.py`, with `traj_opt.py` and `hoi_lib.py`)

1. **Hypotheses and initialisation.** The starting point is the baseline pose for each frame. Each static phase
   starts from a single pose: the rotation chordal mean, the median image position and the median depth over the
   phase. I tested 180° flip hypotheses about the three OBB axes and rejected them on every phase (§4.3), so they
   are not part of the decoding.
2. **One joint render-and-compare optimisation of all 1003 frames.** All static frames share one 6-DoF node, and
   every moving frame has its own node. The loss has four parts:
   * **Silhouette term:** nvdiffrast antialiased silhouette of the mesh (decimated to 8k faces) compared with the
     SAM2 mask. It uses 192² crops and three blur levels, and ignores pixels where the person covers the object.
   * **Box crease-edge chamfer term:** the box's internal face-boundary edges are rendered from the current pose,
     lifted to 3D mesh points, re-extracted every 50 iterations, projected, and scored against the truncated
     distance transform of Canny edges of the contrast-equalised frame. Person pixels are excluded. This term
     mainly fixes yaw, which the silhouette alone pulls about 3° too far.
   * **Smoothness on SE(3):** an acceleration penalty on R and t, plus a motion-adaptive velocity penalty. Its
     weight comes from the same mask-stability signal, so it is strong when the masks say the object is not
     moving. This removed the ±2.5 cm depth wander that silhouettes show while the person occludes the lying box.
   * **Static frames:** each static phase's data term is weighted like √n frames.

   The optimiser is Adam: 150 iterations on silhouettes only, then 100 iterations with silhouettes and edges.
3. **Depth re-anchoring to the baseline.** In the camera frame, translations are scaled along the viewing rays,
   so the image position is unchanged. The scale is a smooth ratio ρ(t) fitted to log(z_base/z_ours): least squares
   with a second-difference smoothness penalty (about a 40-frame scale), and one value per static phase. The result
   keeps our relative motion and removes the baseline jitter, while following the baseline's low-frequency
   (calibrated) depth.
4. **Physical depth cues.** These are computed and reported, and a shift is applied only if at least two
   independent cues agree within 2 cm and the shift is larger than their disagreement.

Constants were chosen with held-out checks:
* The acceleration weight λ=10 was picked using silhouette IoU on held-out frames. The held-out IoU was 0.967,
  0.957 and 0.950 for λ = 10, 50 and 250.
* The edge weight was checked on held-out frames: edges were used only on even frames and scored on odd frames.
  The error went from 2.15 px with silhouettes only to 1.91 px with edges.

## 4. Validation (no ground truth)

All numbers come from `validate.py` (`validation_table.md`).
* IoU is computed with the full textured mesh at half resolution, ignoring person pixels.
* "Edge px" is the mean truncated distance from the rendered internal box creases to Canny edges, at full
  resolution, every 5th frame.
* "Held-out" columns come from a separate run (`solve.py --holdout`) that hides all image evidence of every 5th
  moving frame (120 frames). Those frames are then scored.
* Static spread is the mean distance of mesh points from their phase mean.
* dz is our depth minus the baseline depth, per frame, in the camera frame.

| phase | frames | IoU base | IoU ours | held-out IoU | edge px base | edge px ours | held-out edge px | rot step deg base/ours | z-accel cm base/ours | static spread cm base/ours | floor h cm base/ours (std) | dz mean / median cm | diff of median z cm | ADD(base,ours) cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static 0-158 | 159 | 0.943 | 0.956 | - | 3.19 | 2.13 | - | 0.44 / 0.02 | 0.94 / 0.00 | 1.80 / 0.00 | -1.4 (0.6) / -2.3 (0.0) | 0.07 / 0.04 | 0.04 | 2.47 |
| moving 159-345 | 187 | 0.926 | 0.963 | 0.962 | 2.91 | 2.18 | 2.22 | 0.81 / 0.42 | 1.30 / 0.04 | - | -0.4 (1.9) / 0.4 (1.7) | 0.39 / 0.30 | 0.11 | 3.19 |
| static 346-365 | 20 | 0.931 | 0.970 | - | 2.19 | 1.73 | - | 0.74 / 0.00 | 3.30 / 0.00 | 2.77 / 0.00 | 2.1 (1.3) / 1.4 (0.0) | -0.37 / -0.94 | -0.94 | 4.26 |
| moving 366-421 | 56 | 0.944 | 0.966 | 0.965 | 2.33 | 1.85 | 1.85 | 0.45 / 0.24 | 1.65 / 0.01 | - | 1.9 (1.1) / 1.8 (0.3) | -0.57 / -0.56 | -1.09 | 3.44 |
| static 422-448 | 27 | 0.950 | 0.964 | - | 2.55 | 2.06 | - | 0.50 / 0.03 | 2.59 / 0.00 | 3.29 / 0.00 | 1.7 (1.2) / 1.8 (0.0) | -0.99 / -1.61 | -1.61 | 3.98 |
| moving 449-582 | 134 | 0.921 | 0.952 | 0.950 | 2.43 | 2.10 | 2.14 | 0.80 / 0.57 | 3.10 / 0.04 | - | 1.8 (2.0) / 1.4 (0.7) | 1.56 / 1.75 | -2.10 | 5.18 |
| static 583-713 | 131 | 0.941 | 0.958 | - | 1.91 | 1.63 | - | 0.45 / 0.01 | 1.95 / 0.00 | 2.39 / 0.00 | -0.2 (1.0) / 0.5 (0.0) | -1.02 / -0.78 | -0.78 | 3.05 |
| moving 714-937 | 224 | 0.927 | 0.945 | 0.946 | 2.24 | 1.89 | 1.90 | 0.62 / 0.44 | 1.22 / 0.03 | - | -1.0 (0.9) / -0.6 (1.2) | -0.61 / -1.00 | 4.62 | 2.87 |
| static 938-1002 | 65 | 0.920 | 0.927 | - | 2.52 | 2.18 | - | 0.52 / 0.00 | 0.75 / 0.00 | 0.65 / 0.00 | -1.9 (0.2) / -1.9 (0.0) | 0.32 / 0.15 | 0.15 | 1.35 |
| ALL | 1003 | 0.932 | 0.954 | 0.953 | 2.53 | 2.00 | 2.04 | 0.57 / 0.25 | 1.37 / 0.01 | - | -0.7 (1.8) / 0.3 (1.6) | -0.02 / -0.16 | 2.94 | 3.19 |

On the 120 held-out frames, baseline IoU 0.926 rose to 0.953 and edge error fell from 2.50 to 2.04 px; the
optimiser never saw those frames. Ours vs baseline rotation: median change 1.5°, maximum 4.6°.

**4.1 Image evidence.**
* Every phase improves on both image metrics, including the edge metric, which the silhouette term does not see.
* The held-out frames improve as much as the fitted frames, so the gain is not overfitting to masks.
* The smallest IoU gain is in the last static phase (0.920 → 0.927). There the depth re-anchoring moves the box
  4–5% closer than its silhouette scale suggests (see 4.2).

**4.2 Depth (kept at the baseline).**
* For every phase, the median per-frame depth difference from the baseline is ≤ 1.75 cm, and the episode mean is
  −0.02 cm.
* The "difference of medians" column is large only for moving 714–937 (+4.6 cm). That is a statistics artefact,
  not a shift: depth falls steeply from 2.93 m to 2.25 m inside this phase, so the median of z falls on the steep
  part, and a slightly different timing of the stand-up moves it a lot. The per-frame median difference there is
  −1.0 cm.

The depth cues give conflicting suggestions (cm along the ray, + = farther):

| cue | all frames | per phase (static0, mov, static, mov, static, mov, static, mov, static) |
|---|---|---|
| floor contact: lowest mesh point to the floor level (median foot–plane distance, 3.4 cm) | −10.8 | −19.9, −11.7, −6.9, −5.4, −5.7, −6.6, −9.2, −13.3, −16.8 |
| hand contact: reconstructed hand keypoints onto the object surface (contact frames) | 0.0 | −4.5, −3.0, −8.5, −3.0, +5.0, +2.0, +3.0, +3.0, +9.0 |
| metric mesh size vs silhouette size (our unanchored optimum) | +5.4 | +4.1, +3.7, +1.2, +1.5, +1.1, +6.9, +5.0, +5.0, +7.3 |

The cues disagree in sign and by much more than 2 cm, so **the applied shift is 0 cm**, as the rule requires.
* The floor cue is weak: the height changes only 0.3 cm per cm along the ray, and the plane is uncertain by
  several cm.
* The hand cue is not independent of the baseline, because the human was calibrated together with it.

**4.3 Symmetry flips.** I scored every 20th frame (51 frames) with the pose flipped 180° about each OBB axis:

| phase | ours IoU / edge | flip A | flip B | flip C |
|---|---|---|---|---|
| static 0-158 | 0.956 / 2.10 | 0.855 / 2.82 | 0.861 / 4.13 | 0.923 / 4.31 |
| moving 449-582 | 0.952 / 2.13 | 0.880 / 2.80 | 0.867 / 3.78 | 0.919 / 2.92 |
| static 583-713 | 0.958 / 1.58 | 0.890 / 2.47 | 0.918 / 2.17 | 0.943 / 2.57 |
| static 938-1002 | 0.929 / 2.15 | 0.826 / 2.88 | 0.816 / 3.28 | 0.905 / 3.45 |

The other phases show the same pattern. The orientation is unambiguous everywhere.

## 5. What changed compared with the baseline, and confidence

* **Static phases (0–158, 346–365, 422–448, 583–713, 938–1002):** each is one constant pose. That removes
  0.65–3.3 cm of mean point jitter, and the rotation is refined by about 1–2° using silhouettes and edges pooled
  over all frames. **Confidence: high** for rotation and the image-plane part. Depth stays at the baseline's
  per-phase level, so any constant baseline depth bias remains.
* **Moving phases:** smooth trajectory, about 1.5° rotation corrections, and no depth jitter (z-acceleration
  dropped from 1.3 cm to 0.03 cm). **Confidence: high** for 159–421 and 714–937. **Medium** for 449–582, which has
  fast rotation and the hands covering much of the box; there our depth sits 1.6 cm (mean) farther than the
  baseline because of how the anchoring ratio was smoothed.
* **Not changed:** the overall depth placement, which stays the baseline's per phase. No 180° flip was needed.
* **Remaining risks:**
  * A global baseline depth bias, if there is one, is untouched. My silhouette scale says about 5 cm farther and
    the floor says about 11 cm closer; neither is trustworthy alone.
  * The SAM2 mask and scanned-mesh artefacts (white blobs around the castors) limit silhouette accuracy to about
    1°.
  * GPU atomics in nvdiffrast backward make reruns bitwise non-identical, at the 1e-5 level. Seeds are fixed.

## 6. Files
* `solve.py`: the full pipeline. Its only inputs are `/work/inputs`.
* `traj_opt.py`, `hoi_lib.py`: the optimiser and rendering code.
* `validate.py`: produces `validation_table.md`.
* `object_poses_world_diag.json` / `.npz`: phases, cues, the unanchored optimum, and the ratios.
* `solve_run.log`: log of the final run.
