# White desk (episode 10): trajectory-level object pose recovery

Output: `object_poses_world.npz` (`T_world_mesh`, 794×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(run log: `solve_log.txt`; overlays of the final result: `validation_overlays.jpg`, green = my render, red = SAM2 object mask).

## 1. What happens in the video, and the phases

The person walks up to a standing desk, grabs the edge nearest to them, gives the desk a small pull, turns it a little, and
then tips it over: it pivots on the ends of its two feet until the tabletop edge reaches the floor. The desk
then rests on its side (tabletop vertical, legs horizontal) until the end. The desk is never carried: at least its
feet stay close to the floor the whole time.

`solve.py` finds the phases automatically. It measures the mean absolute grey-level change between frames i and i-3
inside the dilated object mask, ignoring the dilated person mask. When nothing moves this value is 0.6–1.5; motion
gives 2–25. Frames above 2.0 count as moving, and a static run must last at least 20 frames. Result:

| phase | frames | content |
|---|---|---|
| S0 | 0–166 | desk standing, untouched (person walks up) |
| T1 | 167–197 | grasp: the desk is pulled ~2 cm and turned ~1.5° (visible as edge displacement in difference images) |
| S1 | 198–260 | desk standing, held but not moving |
| M  | 261–596 | manipulation: turned about the vertical (~270–330), then tipped about the foot ends (~330–590) |
| S2 | 597–793 | desk lying on its side on the floor (hands leave it at ~640) |

I checked these boundaries by eye on difference images and frames. They are not hard-coded.

## 2. Diagnosis of the baseline

* **No flips or jumps.** The largest frame-to-frame rotation step is 1.6°, and the total rotation is ~90°, which matches the tip-over.
  Symmetry check: I fitted the static phases from the baseline and from 180° flips about each principal axis.
  * S2: the baseline orientation wins clearly (held-out IoU 0.946 vs 0.923 / 0.69 / 0.42), because the underside cross-bar and the leg geometry are visible.
  * S0: the flip about the near-vertical axis is a silhouette tie (0.843 vs 0.846). The upright desk really is ambiguous there.
  * S0's orientation is still settled by continuity. The baseline (and my trajectory) goes smoothly from S0 to the unambiguous S2 through a single ~90° tip. A flipped S0 would need an extra 180° turn that the video does not show.
* **Per-frame jitter.** Even in the static phases the centre moves 7–13 mm/frame² (2nd difference) and the rotation 0.3°/frame.
* **Systematic misfit.** Silhouette IoU is 0.72–0.77 in S0/S1/M. Overlays show the legs and feet visibly offset, i.e. rotation errors of ~1.5–2°.
* **Depth errors along the viewing ray during the tip-over.**
  * The two foot ends barely move in the image from frame 380 to the end (they are the pivot). Even so, the baseline puts them 2–4.5 cm *below* the floor plane around frames 460–510 and moves their camera depth by ±7 cm.
  * The pivot points' world positions spread by 3.5 cm (frames 400–793). Physically they should hardly move.

## 3. Method (`solve.py`)

1. **Phase segmentation** (above).
2. **Joint render-and-compare over the whole episode** (nvdiffrast, half resolution 768×576, mesh as given).
   * **Unknowns:** one pose per static phase (S0, S1, S2), so evidence is pooled over every 2nd frame of the phase, and one pose per moving frame.
   * **Pose parameters:** a rotation about the object centre, the centre's image-plane position, and its log-depth.
   * **Silhouette term:** L2 between the anti-aliased render and the SAM2 object mask, with the dilated person mask excluded.
   * **Smoothness term:** 2nd differences of the centre (σ = 2 mm/frame²) and of the rotation matrix, weight λ = 0.01.
   * **Floor term:** the desk's lowest point must not go below the released floor plane.
     * It is a one-sided hinge, so lifting is allowed. Around frames 300–360 the silhouettes suggest the desk was lifted ~2 cm while being turned.
     * Its gradient reaches **only the depth along the viewing ray**; rotation and image position come from the images alone.
     * It is ramped in after the silhouettes have converged.
   * **Depth anchoring:** per phase group, the median camera depth of the object centre is held at the baseline's median. Groups are A = S0+T1+S1 (0–260), B = M (261–596), C = S2 (597–793).
   * **Optimiser:** initialised from the baseline (rotation mean / median translation for the static phases); Adam, 300 iterations.
3. **Exact re-anchoring:** after optimisation, each group's camera-frame centre positions are scaled along their viewing rays so the median depth equals the baseline's exactly. This keeps the image position. The result is then mapped back to world with `T_world_cam`.

Why S0, T1 and S1 share one anchoring group: two independent cues say S1 is ~3.5 cm closer to the camera than S0.
* Silhouette scale: same viewpoint and same object, so mask bias cancels between the two phases.
* Floor height: with the shift, the lowest-point height is 1.7 cm in S0 and 1.5 cm in S1, consistent with a desk sliding on the same floor. Without it, S1's feet would drop ~1.5 cm relative to S0.

The baseline instead puts S1 1.3 cm *further* than S0. So only the group's overall placement is taken from the baseline.

## 4. Experiments (25 % of frames held out: 5 of every 20 get no silhouette term)

Held-out IoU is half resolution with person pixels excluded. "Pivot spread" is the mean distance of the two foot-end points from their median world position over frames 400–793; physically it should be close to 0.

| variant | M held-out IoU | all held-out IoU | lowest point in M (cm) | pivot spread (cm) |
|---|---|---|---|---|
| baseline | 0.774 | 0.789 | −4.5 | 3.54 |
| silhouette + smoothness, no floor | 0.845 | 0.868 | −3.9 | 2.36 |
| + floor non-penetration, full gradient | 0.816 | 0.855 | −0.4 | – |
| + floor contact-equality (lifting forbidden) | 0.829 | 0.860 | −0.1 | 2.06 |
| **+ floor non-penetration, ray-only gradient (final)** | **0.852** | **0.871** | −1.3 | **1.29** |
| same, λ = 0.003 | 0.860 | 0.873 | −1.3 | 1.75 |

Key test: I took the no-floor solution and moved only the penetrating frames closer along their rays until they stopped penetrating. IoU *rose* (frames 460–500: 0.87 → 0.90). So image evidence and floor evidence agree on that depth correction; the optimiser had simply not been able to find it in metric-translation coordinates. That is why I switched to (image position, log-depth) parameters and a ray-only floor gradient.

Forcing contact everywhere hurt IoU at 300–360, consistent with a small lift, so I kept the floor term one-sided.

I picked λ = 0.01 over 0.003. The IoU difference is small (0.008), while λ = 0.003 doubles the depth deviation from the baseline in M and makes the pivot less stationary. Rotation flips (§2) were tested and rejected.

## 5. Validation table: final result vs. baseline (all frames)

IoU is at half resolution with person pixels excluded. The held-out IoU column comes from the identical configuration run with 25 % of frames held out. Acceleration is the centre's 2nd difference. dz is my camera-frame centre depth minus the baseline's. "Rot. change" and "centre change" are the mean rotation and centre difference to the baseline.

| phase | IoU base | IoU final | held-out IoU (base → final) | accel. base / final (mm/f²) | lowest point base / final (cm above plane) | dz mean (cm) | dz median (cm) | rot. change (°) | centre change (cm) |
|---|---|---|---|---|---|---|---|---|---|
| S0 static | 0.755 | 0.845 | 0.754 → 0.844 | 7.1 / 0.0 | 0.9 / 1.7 | −0.12 | −0.04 | 1.47 | 1.74 |
| T1 grasp | 0.768 | 0.876 | 0.763 → 0.872 | 7.2 / 0.3 | 2.3 / 0.9 | −0.39 | −0.43 | 1.18 | 1.65 |
| S1 static | 0.724 | 0.876 | 0.720 → 0.874 | 7.8 / 0.1 | 1.4 / 1.5 | −3.12 | −3.70 | 1.26 | 3.59 |
| M manipulation | 0.774 | 0.858 | 0.774 → 0.852 | 16.6 / 0.6 | −4.5 / −1.1 | −3.10 | −1.75 | 1.84 | 3.96 |
| S2 on floor | 0.868 | 0.925 | 0.868 → 0.922 | 13.3 / 0.0 | −3.7 / 0.3 | +0.26 | 0.00 | 1.75 | 3.18 |
| all | 0.789 | 0.874 | 0.789 → 0.871 | 12.7 / 0.3 | −4.5 / −1.1 | −1.53 | −0.76 | 1.67 | 3.18 |

* The median depth of each anchoring group equals the baseline's: residual scales are 1.00002, 1.00109 and 1.00002 before the exact re-anchoring.
* M's dz mean of −3.1 cm (median of differences −1.75 cm) comes from lifting the baseline's floor-penetrating frames (460–510, 555–575) along their rays. Silhouettes and the floor both support that change (§4).
* Pivot spread is 3.54 cm for the baseline and 1.41 cm for the final result.

## 6. Depth cues (shift along the viewing ray that each cue suggests for my trajectory; + = further away)

| phase | floor (lowest point at true-human-sole level, +1.4 cm above plane) | object-only silhouette scale (free fit) | hand contact (MHR hand keypoints) |
|---|---|---|---|
| S0 | +1.0 cm | −1.5 cm | uninformative (few contact frames, flat cost) |
| S1 | +0.2 cm | (relative to S0: −3.5 cm, used inside group A) | uninformative |
| M  | −2.3 cm | −7.5 cm | flat (−1 cm, cost changes by <1 cm over ±10 cm) |
| S2 | −3.1 cm | not measured | no contact after frame 640 |

No two independent cues agree within 2 cm on the same correction, so **the applied global depth shift is 0 in every group**: each group's median depth equals the baseline's.

## 7. Confidence per phase

* **S0 (0–166): high** for rotation (~1.5° correction; overlays fit legs and feet clearly better) and image position. Depth is the baseline's. Orientation rests on continuity with S2 because the upright desk's vertical-axis flip is a silhouette tie.
* **T1/S1 (167–260): medium-high.** The pull towards the person (−1.8 cm x, −2.1 cm z relative to S0) comes from silhouettes and floor consistency. It is small but disagrees with the baseline's +1.3 cm.
* **M (261–596): medium-high** for rotation and image position. Overlays fit throughout, and the trajectory is smooth and physically plausible (no floor penetration beyond ~1 cm, stationary pivot). Medium for per-frame depth: depth comes from silhouettes plus floor, with only the median taken from the baseline.
* **S2 (597–793): high** for rotation (the cross-bar and feet are distinctive; flip hypotheses clearly rejected). Depth is the baseline's median.

## 8. Run time and resources

* `solve.py`: 195 s wall-clock. About 75 s of that is CPU loading (video decode and masks); about 110 s uses the GPU. Peak GPU memory is 3.1 GiB.
* Total GPU time of the whole investigation, including all experiments: about 27 minutes.
* `solve.py` fixes its seeds and uses fixed iteration counts and data order. nvdiffrast's anti-aliasing backward pass uses atomic adds, so reruns may differ at float32 rounding level.
* I ran it once for the final output and did not rerun it to check bit-identity, to stay within the GPU budget.
