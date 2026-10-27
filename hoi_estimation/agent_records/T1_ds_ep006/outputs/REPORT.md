# Monitor 6-DoF trajectory recovery

## Summary

- Input episode: `episode_000006`, object `monitor`, 546 frames.
- Output: `object_poses_world.npz`, key `T_world_mesh`, shape `(546, 4, 4)`, `float64`.
- The released `T_world_mesh` is used as the depth/world-calibration fallback. Its camera-frame CoCoNet track supplies the object proposal in every frame.
- I did not use ground truth or challenge annotations. Validation uses only the supplied video, masks, reconstructed human, mesh, and camera/floor metadata.

The main correction is temporal. The baseline's yaw reverses direction for about 30 frames during the fast edge-on turn and then jumps back near the final 180-degree orientation. The corrected track keeps one rigid object, turns monotonically about the table-normal axis, pools the two static phases into single poses, removes tracker translation detours, preserves the baseline's human/floor-calibrated support height, and refines static rotation under the released camera `K` without sacrificing table contact.

## Phases

| Phase | Frames | Interpretation | Confidence |
|---|---:|---|---|
| P0 | 0-190 | Monitor untouched on the table; static world pose. | High |
| P1 | 191-339 | Person approaches and begins carrying/turning the monitor; baseline pose is mostly reliable. | High |
| P2 | 340-379 | Fast front-to-back turn. The display crosses edge-on near frame 345 and is heavily occluded by the person. Baseline yaw oscillates/reverses here. | Medium |
| P3 | 380-399 | Monitor settles toward its final 180-degree orientation. | Medium-high |
| P4 | 400-545 | Monitor resting on the table; static world pose. | High |

The boundaries are encoded as commented constants in `solve.py`. They were selected from the contact interval, the first sustained baseline yaw reversal, the dark/back-facing display appearance after the edge-on crossing, and the point where the mask footprint stops changing.

## Diagnosis and changes

### Baseline failure modes observed

1. **Yaw reversal during the turn.** Baseline world rotation has 118.4 degrees of one-frame change and 179.7 degrees of five-frame change. From approximately frames 345-375 the estimated monitor faces the camera again, while the video shows the dark back side after the edge-on crossing. The baseline then jumps back to the final near-180-degree orientation.
2. **Non-constant static poses.** In the untouched phase, baseline world translation spans 0.153 m and rotation varies by 2.22 degrees. In the final resting phase, translation spans 0.185 m and rotation varies by 7.29 degrees.
3. **Translation detours.** CoCoNet translation moves laterally away from the hand/table during the occluded turn and returns several frames later. This is not physically consistent with the visible base moving smoothly on the table.

### Corrections

1. **Static-phase pose pooling.** I optimized one 6-DoF camera pose for each static phase against every fifth SAM2 silhouette. The score removes person-mask pixels before comparison and uses a deterministic bounded Powell search initialized from the robust mean/minimum-median pose. Pooled silhouette IoU under `K_pipeline` was 0.877 for P0 and 0.713 for P4.
2. **Monotone yaw propagation.** For frames 344-380 I propagated the last reliable orientation about the table-normal axis. The turn uses square-root easing because the display crosses edge-on almost immediately after frame 344 and then slowly settles. A conventional smoothstep turned too slowly and reproduced the visible front side too long, losing about 0.03 mean mask IoU in this phase.
3. **Translation smoothing and repair.** The reliable 190-344 portion of the CoCoNet translation is Savitzky-Golay smoothed. The tracker detour in 345-379 is replaced by a smooth interpolation between reliable endpoint poses. The final static phase uses one pooled translation.
4. **World-depth and support preservation.** Camera rotations are transferred to the released world frame through the nearly constant rotation `R_world_mesh @ R_cam_mesh^-1`. Camera translations in the moving segment retain the baseline's human/floor depth calibration rather than replacing depth with the less reliable object-silhouette scale. A free 6-DoF fit under `K` improved masks but lifted the base roughly 10 cm above the table, so depth was not allowed to float freely.
5. **K-consistent static rotation with table contact.** For each static phase I refined only the 3-DoF rotation under the released `K`, holding the pooled baseline translation fixed during silhouette scoring. The resulting world pose was then translated along the released ground-plane normal so its lowest support percentile matched the same table height as the untouched initial pose. This removed the baseline's 14.6 mm initial/final support disagreement and raised static visible-mask IoU substantially without snapping the object to the floor.

## Validation

No ground truth is available. The table below uses the same metrics for baseline and result. `K` is the released camera intrinsic matrix; `K_pipeline` is the MoGe/FoundationPose intrinsic used by the proposals. The two differ mainly by principal point (about 10 px in x and 15 px in y), so I report both. Mask IoU excludes pixels covered by the person mask.

### Visible-mask IoU, baseline -> result

| Phase | Frames | IoU with `K` | IoU with `K_pipeline` |
|---|---:|---:|---:|
| P0 static | 0-190 | 0.5001 -> **0.5704** | 0.5895 -> **0.6291** |
| P1 approach | 191-339 | 0.5082 -> **0.5132** | 0.6046 -> **0.6136** |
| P2 edge-on turn | 340-379 | 0.3584 -> **0.4163** | 0.3509 -> **0.4113** |
| P3 settle | 380-399 | 0.3864 -> **0.4900** | 0.4468 -> **0.5663** |
| P4 static | 400-545 | 0.3718 -> **0.5232** | 0.4238 -> **0.5773** |
| Overall | 0-545 | 0.4535 -> **0.5279** | 0.5266 -> **0.5928** |

All phases improve in the pooled checks. The largest gains are in P0 and P4, where a single support-constrained static pose removes baseline drift, and in P3-P4 after the turn, where the baseline had lost the correct orientation/support state. P2 remains the least certain phase because the monitor is edge-on and occluded.

### Physical and temporal checks

| Check | Baseline | Result |
|---|---:|---:|
| Hand-to-object p90 minimum distance, frames 100-450 | 40.5 mm | **5.4 mm** |
| Frames with hand within 5 cm | 94.4% | **100.0%** |
| Median one-frame rotation change | 0.772 deg | **0.000 deg** |
| P95 one-frame rotation change | 12.438 deg | **2.072 deg** |
| Maximum one-frame rotation change | 118.407 deg | **5.247 deg** |
| P95 one-frame translation change | 38.9 mm | **5.3 mm** |
| P0 translation span | 0.153 m | **0 m** |
| P4 translation span | 0.185 m | **0 m** |
| P4 maximum rotation deviation | 7.29 deg | **0 deg** |
| Initial-to-final table support difference | -14.6 mm | **0.0 mm** |

The contact check uses reconstructed MHR hand keypoints and the nearest sampled object-mesh vertices. It is a consistency check, not an independent measurement, but it favours the result strongly.

## Adopted decisions and confidence

- **P0:** Adopt one pooled, support-constrained pose. High confidence; the object is untouched and the hand contact probability is low for most of the phase.
- **P1:** Retain the CoCoNet proposal with smoothing and a 20-frame blend from the pooled initial pose. High confidence through frame 339 because the silhouette is large and baseline yaw remains monotone.
- **P2:** Replace the baseline yaw reversal with monotone turn propagation. Medium confidence because the screen is nearly edge-on and the person occupies much of it. Continuity, the observed front/back appearance transition, depth ordering, and the hand grasp all favour the replacement.
- **P3:** Blend into the pooled final pose. Medium-high confidence.
- **P4:** Use one pooled resting pose, refine rotation under released `K`, and align its table support to the initial pose. High confidence; the object footprint and hand contact are stable.

The main residual uncertainty is rotation during the most occluded part of P2. A wrong ±180-degree front/back branch there would dominate ADD, but the rendered normal direction follows the observed bright-front to dark-back transition, whereas the baseline reverses through the front side and then jumps.

## Reproducibility and runtime

Final command:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script is deterministic, seeds NumPy/PyTorch, performs no network access, and reads only the supplied inputs. Final measured runtime in this container was 216 seconds on one GPU. It uses nvdiffrast only for silhouette scoring and does not reload any external model weights. Output file SHA-256 at submission time:

```text
a6d3ca81d75051ca484dfdceb58627907c915398e55e0ca408ac860a94c86efc  object_poses_world.npz
```
