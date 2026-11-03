# Object trajectory result

`object_poses_world.npz` contains `T_world_mesh`, shape **(922, 4, 4)**, dtype **float64**, for every video frame. Poses map the original GLB mesh-file coordinates into the released world frame. No global alignment or coordinate redefinition was applied.

The solver modifies **709 frames** and preserves **213 frames exactly** after conversion of the original float32 values to float64. The largest corrections resolve the box's incorrect texture orientation on the floor and its incorrect standing orientation after return to the desk.

## Changes and evidence

The original reconstruction often has a plausible box outline while assigning the wrong physical faces. On the floor, it renders the blue tape underneath and printed text upside down. A half-turn about the mesh z axis resolves this ambiguity. At the final desk position, the baseline renders a tall face instead of the observed short face; independent texture correspondences recover the correct orientation and improve depth.

The method renders the exact supplied textured mesh, including a per-pixel map back to mesh-file coordinates. SIFT matches provide 2D–3D correspondences for RANSAC PnP. Four half-turn hypotheses are tested at every third frame, and two additional earlier-view orientation proposals are available after the second lift. Every proposal is validated against the current video; the earlier poses do not directly dictate the solution.

Safeguards reject repeated-text votes for the same image feature, poorly distributed matches, implausible translations, low silhouette overlap, and weak texture evidence. Large orientation changes require at least 18 inliers and a 12% reduction in normalized color error. Accepted keyframes number **169 of 308**. Successful nearby fits are interpolated only across gaps of at most nine frames. A discrete half-turn is propagated through one 44-frame gap only because independently confirmed corrections agree at both ends and the baseline rotates continuously throughout. That gap retains baseline translation before the floor-contact adjustment below.

Four periods are visibly stationary. Their explicit, commented frame intervals are encoded in `solve.py` (zero-based, end exclusive):

- **[0, 140):** the initial box corners stay fixed, while baseline depth drifts by about 17 cm. Five existing baseline poses with the best joint silhouette/texture fit to the unobstructed frame-0 image are robustly pooled. This uses existing poses without imposing a new scale. Frame-0 outline IoU improves from 0.921 to 0.942.
- **[332, 388):** the box rests on the floor before the push. Nineteen accepted fits agree within a median translation scatter of 9.3 mm.
- **[486, 692):** the box rests beyond the desk, with the person sometimes occluding it. Sixty-three accepted fits agree within 6.5 mm; robust pooling removes drift during occlusion.
- **[834, 922):** the box rests on the desk after release. Thirty accepted fits agree within 4.1 mm.

These intervals were chosen by inspecting the permitted video and persistent mask corners. Pooling also requires automatic agreement of the fitted poses; it is skipped if their median scatter exceeds 3.5 cm or 5 degrees.

During **[306, 700)** the box visibly stays in contact with the carpet, including the foot push. After texture fitting, its rotated mesh's lowest point is constrained to the released floor plane by moving the center along its camera viewing ray. This preserves its projected center and anchors monocular depth to released world geometry. Adjustments above 35 cm would be rejected. The actual median adjustment is 0.123 m. This final contact step can reduce silhouette IoU slightly, as expected when correcting estimated camera/depth inconsistencies.

## Image-space validation

The following measurements use supplied object/person masks and mesh renders. IoU compares visible silhouettes, excluding person occlusion. Color error compares brightness-normalized, blurred RGB over the observed object. These are evidence of image agreement, **not ground-truth ADD measurements**.

| Frame | Baseline IoU | Result IoU | Baseline color error | Result color error |
|---|---:|---:|---:|---:|
| 0 | 0.921 | 0.942 | 0.0159 | 0.0175 |
| 360 | 0.875 | 0.902 | 0.0794 | 0.0541 |
| 420 | 0.735 | 0.741 | 0.0740 | 0.0621 |
| 480 | 0.947 | 0.882 | 0.0986 | 0.0707 |
| 660 | 0.910 | 0.907 | 0.0919 | 0.0765 |
| 720 | 0.774 | 0.904 | 0.0646 | 0.0156 |
| 840 | 0.598 | 0.960 | 0.0794 | 0.0291 |
| 921 | 0.597 | 0.955 | 0.0792 | 0.0288 |

`result_comparison.jpg` shows video / baseline render / final render for representative frames. `validation.json` contains the measurements and array checks. `diagnostics.json` records acceptance decisions, fit residuals, confidence evidence, and elapsed time. `fit_diagnostics.npz` records the correction source for each frame: 0 = baseline; 1 = accepted fit; 2 = short interpolation; 3 = pooled stationary fit; 4 = initial stationary pose selection; 5 = confirmed half-turn propagation.

## Confidence and limitations

Confidence is high in the floor half-turn correction and the final desk orientation: both the unique blue tape and the printed faces agree with the video. Confidence in metric translation is moderate because the supplied camera intrinsics and world transform are estimated. Depth uses the exact mesh size, accepted correspondences, and floor support where contact is directly visible. Camera calibration is unchanged.

Before the contact step, the pooled floor pose's lowest point lies about 4.7 cm above the supplied plane. The frame-0 reference human reaches that plane within 9 mm, supporting its use as a world anchor. The final floor pose has effectively zero lowest-point distance to the plane. The plane itself is estimated, so its error and camera-ray calibration remain sources of depth uncertainty. Contact is imposed only on the explicitly observed floor interval, never during carrying or placement on the desk.

Heavily occluded carrying frames and the second lift retain the baseline where no reliable alternative is supported: the unchanged intervals are [140,306) and [748,795). Thus some known visual errors remain, including near frame 780. No ground truth object poses, challenge annotations, external video data, downloaded models, or network access were used. The permitted frame-0 human reference was used only to check the released plane. ADD improvement cannot be quantified from the permitted inputs alone.

## Reproduction, checks, and runtime

Run offline:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script is self-contained and requires no other output, scratch file, model weight, or `/pipeline` intermediate. Dependencies and versions are in `TOOLS.md`. Seeds are fixed; SIFT/PnP use deterministic CPU processing and rendering uses only the selected CUDA device 0.

The final solve took **204.0 seconds wall time** (about 3.40 minutes), including input decoding, fitting, rendering, and saving. One GPU was used; all GPU-using exploratory, solve, validation, and repetition runs together remained below 25 minutes of process wall time, below the 45-minute budget. No training was performed.

Validation confirms all 922 finite float64 transforms, exact homogeneous bottom rows, and exact preservation of untouched baseline frames. Maximum rotation orthogonality error is 1.09e-05; small errors of this order are inherited from the unchanged float32 baseline (whose original maximum is 1.23e-5). Corrected fitted rotations are constructed as proper rotations.

A second complete invocation produced **exactly equal pose arrays** (`numpy.array_equal`), maximum absolute difference 0.0.
