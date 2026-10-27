# Episode 29: rolling luggage, pushed with the left foot. Trajectory re-estimation

Output: `object_poses_world.npz` (`T_world_mesh`, 467×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(log of the final run: `solve_run.log`).

## 1. What the video shows (phases)
The camera is static. A carry-on spinner suitcase (handle extended) stands on the carpet. The person pushes it
three times with the left foot, each time toward the camera; it rolls on its four wheels and turns slightly. It is
never lifted or tipped (checked by eye on frame strips of all three pushes).

Phases are detected automatically from the SAM2 object mask: bbox and centroid change over ±3 frames after a
5-frame median filter, threshold 2.5 px, then morphological cleanup. The result matched my visual check.

| phase | frames | content |
|---|---|---|
| S1 static | 0–102 | luggage untouched, side view |
| M1 push | 103–134 | rolls ~15 cm toward the camera, yaw −30° |
| S2 static | 135–219 | at rest |
| M2 push | 220–251 | rolls ~25 cm toward the camera |
| S3 static | 252–339 | at rest |
| M3 push | 340–375 | rolls ~25 cm to the right and toward the camera, yaw −5° |
| S4 static | 376–466 | at rest |

## 2. Diagnosis of the baseline
* **Coarse orientation is correct.** Rendering the textured mesh shows the right face, the handle on the
  correct side and the label in the right place. There is no 180° flip, and the flip test in §4 confirms this.
* **Static phases are not constant.** Rotation jitters 0.5–1°/frame, with up to ~6° excursions inside S1.
  Camera-frame depth jitters ±3 cm.
* **Tilt is wrong.** The baseline tilts the case 0–3° relative to the floor normal, varying per phase and
  mostly toward the camera. A four-wheel suitcase on a flat floor cannot do that: its wheel plane is parallel to
  the floor. Because the handle reaches 0.66 m above the mesh origin, a 3° tilt moves its top by ~3 cm.
* **The case rises during push M2.** In M2 the baseline lifts the case ~2 cm relative to the neighbouring static
  phases (wheel height −2.1 → 0.0 → −1.0 cm). That is the typical along-the-ray depth error of a per-frame
  estimate.

## 3. Method (`solve.py`)
1. **Physical model.** The pose is "upright on the floor": the mesh's wheel-contact plane (fitted through the
   four lowest wheel points) lies on the released floor plane. The free parameters are yaw about the floor
   normal and a 2-D position on the floor. Each static phase gets one shared pose; moving frames (and 3-frame
   margins around them) get one pose per frame.
2. **Evidence.** Differentiable silhouettes (nvdiffrast) of the luggage *body* (z < 0.235 m in the mesh), compared
   with the SAM2 object mask by soft IoU, ignoring person-mask pixels. The telescopic handle is excluded from this
   term because the SAM2 mask does not contain the rods.
3. **Image-scale nuisance.** At the floor-consistent (and baseline) distance, the real suitcase appears ~4.3%
   larger than the metric mesh rendered with `K`. This holds in every phase (1.037–1.056 when fitted per phase).
   Without a nuisance term, the fit absorbs this by rotating or tilting the case: a free 6-DoF fit tilted it 20°
   toward the camera. A single global scale factor absorbs it instead; it is used only during fitting and never
   applied to the output poses.
4. **Hypotheses.** Each static phase is re-fitted from the baseline yaw and from yaw+180°. The two are scored by
   body IoU and by a chamfer distance between the rendered handle outline and Canny edges of the video. The
   handle is the main asymmetric part. A flip is adopted only if both scores clearly favour it; none did.
5. **Global decoding.** All 467 frames are fitted jointly, with an acceleration penalty on yaw and floor position
   (weight chosen on held-out frames, see §5). Adam runs 600 iterations with a cosine-decayed learning rate.
6. **Depth re-anchoring.** Positions are scaled along the viewing rays in the camera frame so that each static
   phase's median camera depth equals the baseline's. Moving frames use a scale factor interpolated linearly
   between the neighbouring static phases, so the static anchoring stays continuous and the rolling motion stays
   on a constant-height path. Factors are 1.017–1.025. The image position is unchanged.

## 4. Depth: cues and decision (applied shift = 0)

| cue (independent) | suggested shift of the object along the ray (cm), S1 / S2 / S3 / S4 |
|---|---|
| floor contact: wheels at the sole height of the true frame-0 human (−0.75 cm vs. plane) | −6.2 / −5.1 / −2.7 / −3.1 |
| metric mesh size vs. image size (silhouette scale 1.043) | ≈ −13.7 / −12.9 / −11.6 / −10.8 |
| foot contact in pushes: closest reconstructed-human vertex is 4–7 cm from the case at shift 0 | +3 to +6 (farther) |

The three cues disagree by much more than 2 cm and do not even agree in sign. **No depth shift beyond the baseline
anchoring was applied.**

The size cue is also inconsistent with the floor: moving the case 12 cm closer would leave the wheels ~4 cm
above the floor, while the frame-0 human's soles touch the plane to within 1 cm. I therefore treat the 4% size
excess as an unexplained intrinsics or mesh-scale anomaly, not as depth. The re-anchored result leaves the wheels
2–3 cm below the plane, as the baseline does.

Camera-frame depth difference, final result minus baseline (cm):

| phase | S1 | M1 | S2 | M2 | S3 | M3 | S4 |
|---|---|---|---|---|---|---|---|
| mean | −0.2 | 0.9 | 1.8 | 5.6 | 0.7 | 0.9 | 0.2 |
| median | −0.1 | 0.4 | −0.1 | 6.0 | 0.1 | 1.0 | −0.0 |

S2's mean differs from its median because the baseline drifts ~10 cm in depth within that static phase. M2
differs by 6 cm because the baseline brings the case closer (and higher) during that push; I keep the
floor-consistent rolling path between the anchored static phases. Over the episode, the mean |Δz| is 2.3 cm.

## 5. Validation (no ground truth)
IoU is body-silhouette IoU against the SAM2 mask, with person pixels excluded.

**Held-out frames.** Fit on even frames only, evaluate on odd frames (W_ACC = 1e2, run before the final
cosine-LR change):

| phase | IoU baseline | IoU ours (+image-scale nuisance) |
|---|---|---|
| S1 0–102 | 0.860 | 0.946 |
| M1 103–134 | 0.853 | 0.929 |
| S2 135–219 | 0.871 | 0.941 |
| M2 220–251 | 0.897 | 0.948 |
| S3 252–339 | 0.892 | 0.949 |
| M3 340–375 | 0.898 | 0.959 |
| S4 376–466 | 0.884 | 0.960 |

Smoothness weight on held-out moving frames (M1 / M2 / M3): 1e4 gives 0.883 / 0.936 / 0.947; 1e3 gives
0.895 / 0.946 / 0.954; 1e2 gives 0.929 / 0.948 / 0.959; 2e1 gives 0.932 / 0.948 / 0.959. I chose 1e2.

**Same-scale comparison.** Both pose sets rendered with the same 1.045 image-scale factor, all frames:

| phase | S1 | M1 | S2 | M2 | S3 | M3 | S4 |
|---|---|---|---|---|---|---|---|
| baseline | 0.923 | 0.908 | 0.918 | 0.941 | 0.928 | 0.921 | 0.916 |
| ours | 0.931 | 0.912 | 0.924 | 0.935 | 0.939 | 0.950 | 0.948 |

At scale 1.0 (metric mesh, `K`, baseline depth), the final poses score 0.85–0.88 against the baseline's 0.85–0.90
(table in `solve_run.log`). The baseline's tilt partly compensates for the size anomaly, so this unscaled metric
cannot separate the two.

**Final run diagnostics** (`solve_run.log`):

| phase | rot diff ours−base mean/max (°) | rotation jitter base / ours (°/frame) | wheel height above plane base / ours (cm) |
|---|---|---|---|
| S1 | 1.7 / 5.7 | 0.95 / 0.01 | −2.8 / −2.8 |
| M1 | 1.8 / 3.1 | 1.22 / 0.94 (real turning, ~1°/frame) | −2.2 / −2.7 |
| S2 | 2.4 / 4.0 | 0.57 / 0.01 | −2.1 / −2.7 |
| M2 | 2.1 / 2.8 | 0.73 / 0.19 | 0.0 / −2.2 |
| S3 | 2.9 / 4.6 | 0.55 / 0.00 | −1.0 / −1.8 |
| M3 | 3.7 / 4.6 | 0.65 / 0.28 | −1.1 / −1.9 |
| S4 | 3.3 / 3.8 | 0.54 / 0.02 | −1.1 / −2.1 |

**Other checks.**
* **Flip test** (body IoU / handle chamfer in px, keep vs. yaw+180°):
  * S1: 0.945 / 2.4 vs 0.925 / 7.4
  * S2: 0.941 / 2.7 vs 0.947 / 4.8
  * S3: 0.950 / 2.0 vs 0.950 / 3.2
  * S4: 0.963 / 1.6 vs 0.926 / 5.5

  The handle rejects the flip everywhere. The body silhouette alone is ambiguous in S2 and S3: an earlier version
  that used only IoU wrongly flipped those phases, which is why the handle term and the margin exist.
* **Internal-edge cue** (development check, not used by `solve.py`): chamfer between Canny edges of the rendered
  texture and of the image inside the mask, sweeping yaw ±8°. The minimum lies within 0 to −4° of our yaw in
  every static phase. Our poses score better than the baseline's in S3/S4 (2.64 vs 2.74 px, 2.74 vs 2.86 px) and
  equal in S1/S2. This independently supports the yaw to within ~3°.
* **Free tilt.** When tilt is a free global parameter, the silhouette fit picks −2.8° and the baseline has
  +0–3°, opposite signs. That axis is barely observable from a single view, so I use the physical prior of 0 tilt
  (left and right wheels are at equal height in the mesh). The fit with tilt fixed reaches a slightly lower data
  loss (0.0563 vs 0.0574).

## 6. Confidence
* **Static phases S1–S4: medium-high.** A constant pose is certain, the orientation (no flip) is certain, and the
  upright prior is physically sound. Yaw is accurate to ~1–2°, slightly less in S2, where the yaw objective is
  flatter (two early runs differed by 1.6° before the LR schedule was added). Lateral position is
  image-constrained; depth equals the baseline median.
* **Pushes M1, M3: medium.** They are smooth, keep the wheels on the floor and match held-out silhouettes well.
  The foot occludes part of the case.
* **Push M2: medium.** It is 6 cm farther than the baseline on purpose (baseline lifts the case, see §2). If the
  baseline's depth during M2 happens to be right, this costs accuracy on those 32 frames.
* **Main residual risk.** The unexplained ~4% image-scale anomaly. Absolute depth is left at the baseline; if the
  intrinsics are what is off, both the baseline and this result carry a lateral error of ~1–4 cm.

## 7. Run time and resources
* `solve.py`: ~65 s wall time, peak GPU memory 1.8 GB.
* Total GPU time for the whole project: ≈ 15 min, including development runs.
* **Determinism:** seeds are fixed, but nvdiffrast's gradient atomics are not bit-deterministic. Two independent
  runs of `solve.py` differ by at most 0.9 cm and 0.4° per frame.
