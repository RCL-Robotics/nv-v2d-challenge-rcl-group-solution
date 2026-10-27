# Traffic-cone trajectory (episode 17): report

**Bottom line.** I changed the baseline where the evidence is clear:
- the 90° yaw jump that starts while the cone is still on the floor (frame 118);
- the jittery and partly flipped rotations during the carry;
- the drifting per-frame poses in the two static phases.

Mask IoU over the video goes from **0.821 to 0.895**. The per-frame rotation jitter (95th percentile) goes from **25° to 3.5°**. The median camera depth equals the baseline's in every phase.

The biggest remaining risk is the yaw "branch" (a 90° step) of the cone **after** it reappears from behind the person and on the table. For the table phase, the evidence is a coin flip, so I kept the baseline's branch.

Final output: `object_poses_world.npz` (`T_world_mesh`, 440×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`.
That run took 3 min 24 s (about 3 min of GPU time).

## 1. Diagnosis of the baseline

* The object is a 38 cm traffic cone with a 20 cm square base and a white reflective band.
  * Geometry and texture are almost exactly **4-fold symmetric about the cone axis** (I call rotation about this axis the *twist*).
  * The silhouette therefore fixes the twist only modulo 90°.
  * ADD has no symmetry handling. A twist error of 20° / 45° / 90° / 180° costs 2.3 / 5.0 / 9.3 / 13.1 cm, so the twist matters most.
* FoundationPose (`T_cam_mesh_foundationpose`) jumps by **+91° of twist at frame 104**, while the cone is untouched on the floor.
  * The baseline (`T_world_mesh`, from CoCoNet) carries the same +89° jump from **frame 118**.
  * The cone has only just been grasped at that point; the mask doesn't move until frame 125, and the base is visibly unchanged.
  * So the baseline's carried phase starts on the wrong 90° branch relative to its own floor phase.
* While the cone is hidden behind the person (frames 197–247), the baseline rotations are noise: 20–80° jumps every few frames, and swings up to 23°.
* During the carry the baseline jitters frame to frame (median rotation step 4.7°/frame, 95th percentile 27°).
* In the floor phase the baseline depth drifts by 3 cm, although the cone is static.

## 2. Phases (detected automatically in `solve.py`)

Phases are found from the IoU between each frame's object mask and a reference mask from the first and last 15 frames. Person pixels are ignored, the threshold is 0.85, and the table phase also needs contact probability < 0.5.

| phase | frames | content |
|---|---|---|
| static floor | 0–125 | untouched until ~117, then grasped; the mask stays fixed until 125 |
| carried | 126–334 | lifted with one hand, swung and turned (126–196); **hidden behind the person 197–247**; carried back with two hands (248–305); placed on the table (306–334, hands still on it) |
| static table | 335–439 | untouched on the round table |

I checked these boundaries by eye on contact sheets of crops; they agree.

## 3. Method

1. **Static phases: one pose each.**
   * Every 3rd frame of the phase is used: 42 frames on the floor, 35 on the table.
   * One shared 6-DoF pose is fitted by differentiable silhouette rendering (nvdiffrast) to the SAM2 object mask. Person pixels that are not object are ignored.
   * Depth is held at the baseline's median depth, and the fit is started from 9 twists between −40° and +40°.
   * Floor: every start converges to the same pose, at twist −1.3° from the baseline. That confirms the baseline's floor pose up to a ~1 cm sideways shift and removes its 3 cm depth drift.
   * Table: every start converges to twist ≡ 22.5° (mod 90), ±1°.
2. **Carried phase: per-frame twist profiles.**
   * For each frame with a visible mask (≥150 px; 159 frames), the pose is fitted on a 5° grid of absolute twists (0–85°).
   * At each grid twist, the tilt (axis direction) and the sideways position are optimised.
   * Depth stays fixed to the temporally smoothed baseline depth.
   * The result is the silhouette IoU as a function of twist (mod 90) for every frame.
3. **Global decoding of the twist.** I use an HMM over the absolute twist (−360…360°, 2.5° states) from the last floor frame to the first table frame.
   * **Data term:** the per-frame IoU profile, with an IoU difference of 0.01 treated as one standard deviation.
   * **Transition:** the expected twist change per frame is the change of the person's torso yaw (computed from the MHR shoulder and hip keypoints), with a standard deviation of 6°/frame. The idea is that an object held in the hands turns with the body. No data term is used in the hidden segment, so this transition alone carries the twist across it.
   * **Anchors:** the floor pose (absolute) and the table pose (mod 90 only).
   * The forward pass gives the posterior of the table phase's 90° branch.
   * **Branch rule:** keep the baseline's table branch unless another branch has posterior > 0.7. The final run gave −67.5° (the baseline's branch): 0.49, −157.5°: 0.49, 112.5°: 0.02. So the baseline's branch is kept.
   * Viterbi then gives the maximum-likelihood twist path, which is lightly smoothed.
4. **Carried poses.**
   * For each visible frame, I take the profile fit closest to the decoded twist and set the twist exactly.
   * The cone-axis direction and the cone-centre position are smoothed (Gaussian, σ = 2 frames), with the static poses as fixed endpoints.
   * Frames with no mask, including the 51-frame hidden segment, use a **hand-attached model**: cone centre = wrist midpoint + an offset interpolated between the nearest visible frames. The axis direction is interpolated.
5. **Depth re-anchoring.** In each phase, camera-frame positions are scaled along their viewing rays so that the median camera z equals the baseline's median z. The scale factors were 1.0005 / 1.0002 / 0.9984.

### Evidence I tried but did not use for decisions
* **Texture.** The scanned texture has a pale smear on the lower body. If it were real, it would break the symmetry.
  * It is not visible in any frame, including median-filtered images of the static phases.
  * In the atlas it sits in a tiny 34×39 px chart next to charts with beige background bleed, so it is almost certainly a scan artifact.
  * A saturation-based photometric score was not consistent between the phases.
  * So texture is not used.
* **Pure body-yaw prior** (twist change = −torso yaw change over the whole carry). It predicts a table twist of −65°, the baseline's branch.
  * It is not reliable while the cone is swung in one hand: over frames 130–196 the torso turned 139° but the silhouette-tracked cone turned only about 80°.
  * So the prior is used only as a soft transition mean inside the HMM.

## 4. Validation (no ground truth)

IoU is the mask IoU of the hard-rendered silhouette against the SAM2 object mask, with person pixels ignored, on frames whose mask has ≥150 px. The static fits used every 3rd frame, so 2/3 of the static frames are held out for this check. "ADD(base,ours)" is the mean distance between mesh points under the baseline pose and under my pose, i.e. how much I changed things. "dz" is my camera-frame depth minus the baseline's.

| segment | frames | IoU base | IoU ours | twist base (start→end) | twist ours (start→end) | mean ADD(base,ours) cm | dz mean / median cm |
|---|---|---|---|---|---|---|---|
| floor (static) | 0–125 | 0.852 | **0.940** | 0 → 87 | −1 → −1 | 2.4 | −0.21 / 0.00 |
| pick-up / carry, visible | 126–196 | 0.797 | **0.901** | 87 → −159 | 0 → 79 | 10.1 | 1.07 / 1.33 |
| carry, hidden | 197–247 | 0.446 (few px) | 0.733 (few px) | −161 → −12 | 82 → 155 | 12.6 | −0.12 / 0.11 |
| carry, visible + placement | 248–334 | 0.788 | **0.849** | −15 → −63 | 159 → −73 | 8.7 | 1.43 / 1.36 |
| table (static) | 335–439 | 0.839 | **0.880** | −66 → −71 | −67 → −67 | 1.5 | 0.14 / −0.00 |

| phase | IoU base | IoU ours | median / p95 rotation step (°/frame), base → ours | median position step (cm/frame), base → ours | dz mean / median (cm) |
|---|---|---|---|---|---|
| floor | 0.852 | 0.940 | 0.00 / 0.8 → 0.00 / 0.0 | 0.01 → 0.00 | −0.21 / 0.00 |
| carried | 0.786 | 0.870 | 4.65 / 27.0 → 1.80 / 4.5 | 2.62 → 1.95 | 0.93 / 1.31 |
| table | 0.839 | 0.880 | 0 / 0 → 0 / 0 | 0 → 0 | 0.14 / 0.00 |
| all | 0.821 | 0.895 | p95: 25.1 → 3.5 | | 0.41 / 0.45 |

Other checks:
* **Silhouette overlays:** checked by eye on contact sheets for frames 120–330.
* **Floor contact:** the lowest point of the cone base is 1.4 cm below the released floor plane (baseline: 0.9–3.0 cm, drifting). The cone axis is 3.6° from the floor normal (baseline 3.8°).
* **Robustness of the table-branch decision:** the table yaw fit gives the same value (±1°) from all starting twists. The HMM posterior was recomputed for 54 settings: IoU noise 0.01–0.03, step noise 4–10°/frame, body-prior weight 0, 0.5 or 1, and table anchor 15° or 20°.
  * The −158° branch is the top branch in many settings, with probability up to 0.79.
  * The baseline branch gets 0.0–0.59; without the body prior, the 22° or 112° branches win instead.
  * No branch is clearly favoured across settings, so the 0.7 rule keeps the baseline's branch.

### Depth cues (rule: deviate only if two independent cues agree within 2 cm)

| phase | cue | suggested change of camera depth |
|---|---|---|
| floor | base resting on the released floor plane | −2.0 cm (shift along the ray) |
| floor | silhouette scale (pooled fit with depth free) | −4.3 cm |
| table | silhouette scale | −19 cm (mask about 3% taller and 14% larger than the model) |
| table / placement | hand contact: wrist midpoint vs cone centre during 316–334 | about +3 cm (hands slightly behind the centre) |
| carried | wrist midpoint vs cone centre, frames with contact | +1.3 cm (cone vs hands); rough, since the wrists are ~10 cm from the surface |

* Floor: the two cues differ by 2.3 cm, which is not smaller than the ~3 cm correction.
* Table: the silhouette and hand cues contradict each other.

**Applied depth shift: 0 in every phase** (only re-anchoring to the baseline median, with scale factors within 0.16% of 1).

## 5. Confidence per phase

| phase | confidence | why |
|---|---|---|
| floor 0–125 | **high** that it is at least as good as the baseline; it gains on frames 118–125 | Same 90° branch as the baseline. One pose for the whole phase, which removes the 3 cm drift and the 89° jump at 118–125. The silhouette agrees with the baseline's twist to within 1.3°. |
| pick-up / carry 126–196 | **medium-high** | The twist is tracked continuously from the floor pose: the base corners turn from 0° to about +80° (monotonic, clearly visible in IoU profiles with ~0.05 range). The turn agrees with the torso turning −80° in 130–165. Here the baseline is +90…+130° off from a twist that would be continuous with its own floor pose. |
| hidden 197–247 | **low** | No image evidence; twist comes from the HMM with the body-yaw prior, position from the hand model. The baseline is noise here too. |
| carry + placement 248–334 | **medium-low** | The twist is reliable mod 90°, but the absolute branch depends on the hidden segment and on the table branch. My twist path differs from the baseline's by ~180° at 258, shrinking to ~0° at the table, because it follows the person's torso turn, which the baseline doesn't. |
| table 335–439 | **medium** (rotation within the branch, translation); **≈50%** (90° branch) | The pose is very stable, but whether the baseline's branch is right is a coin flip. It is kept by rule. |

## 6. Files, run time, reproducibility

* `solve.py` is self-contained. It reads only `/work/inputs`, uses no network, and doesn't touch `/pipeline` or `/weights`.
  * Phase boundaries, the branch rule and all thresholds are explicit constants at the top of the file.
  * The run takes ~3–4 min with a peak GPU memory of a few GB.
  * Seeds are fixed. The only nondeterminism is GPU float atomics in nvdiffrast's antialias gradients: two runs differ by at most 6 mm in position, and the branch posterior moved by ≈0.02. The 0.7 branch rule makes the decision insensitive to this.
* Total GPU time over all experiments plus the final run: about 25 minutes.
* Scratch analysis scripts are under `/work/scratch` and are not needed by `solve.py`.
