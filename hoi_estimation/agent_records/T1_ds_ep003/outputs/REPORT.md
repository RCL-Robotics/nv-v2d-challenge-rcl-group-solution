# Monocular rigid-object trajectory recovery

## Approach

I treated the supplied CARI4D trajectory as a calibrated proposal, not as the final answer. The FoundationPose branch supplies a smoother per-frame rotation proposal than the CoCoNet refinement. Its 180-degree long-axis branch was corrected using the visible handle side, then every phase was refined by rendering the exact textured mesh against the SAM2 object mask while ignoring pixels covered by the person. Static intervals were decoded to one constant SE(3) pose. Motion intervals were fitted per frame, then passed through a second-order quaternion/translation trajectory smoother with the neighbouring static poses as hard anchors. The first and last eight frames of each motion interval are smoothly ramped to those anchors.

No ground-truth or challenge annotation was used. Evidence came only from video, SAM2 masks, the released camera/world calibration, the supplied reconstructions, and the exact mesh.

## Phases

| Phase | Frames | Interpretation | Confidence |
|---|---:|---|---|
| rest0 | 0-130 | mug resting on table; constant pose | high |
| motion0 | 131-214 | lift/drink/place with alternating hand | medium-high |
| rest1 | 215-270 | mug resting on table; constant pose | high |
| motion1 | 271-349 | lift/drink/place with alternating hand | medium-high |
| rest2 | 350-405 | mug resting on table; constant pose | high |
| motion2 | 406-479 | lift/drink/place with alternating hand | medium-high |
| rest3 | 480-530 | mug resting on table; constant pose | high |
| motion3 | 531-601 | lift/drink/place with alternating hand | medium-high |
| rest4 | 602-640 | mug resting on table; constant pose | high |
| motion4 | 641-691 | lift/drink/place with alternating hand | medium-high |
| rest5 | 692-780 | mug resting on table; constant pose | high |

## Validation

Silhouette IoU uses the SAM2 object mask after excluding a 3-pixel dilation of the person mask. Colour is mean RGB normalized cross-correlation on a deterministic 10-frame subsample. P95 rotation step is the 95th percentile of consecutive-frame geodesic rotation inside the phase.

| Phase | IoU baseline | IoU result | Colour corr baseline | Colour corr result | Rot-step p95 baseline (deg) | Rot-step p95 result (deg) |
|---|---:|---:|---:|---:|---:|---:|
| rest0 (0-130) | 0.716 | 0.848 | 0.254 | 0.174 | 8.43 | 0.00 |
| rest1 (215-270) | 0.624 | 0.804 | 0.097 | 0.338 | 13.74 | 0.00 |
| rest2 (350-405) | 0.628 | 0.815 | 0.087 | 0.228 | 16.20 | 0.00 |
| rest3 (480-530) | 0.620 | 0.814 | 0.088 | 0.183 | 12.38 | 0.00 |
| rest4 (602-640) | 0.601 | 0.829 | 0.018 | 0.198 | 14.98 | 0.00 |
| rest5 (692-780) | 0.670 | 0.823 | 0.132 | 0.219 | 12.22 | 0.00 |
| motion0 (131-214) | 0.587 | 0.666 | 0.180 | 0.215 | 26.17 | 12.61 |
| motion1 (271-349) | 0.501 | 0.705 | 0.144 | 0.174 | 61.56 | 21.73 |
| motion2 (406-479) | 0.575 | 0.725 | 0.101 | 0.125 | 77.82 | 9.40 |
| motion3 (531-601) | 0.624 | 0.766 | -0.022 | 0.041 | 79.33 | 11.03 |
| motion4 (641-691) | 0.505 | 0.767 | 0.044 | 0.192 | 91.38 | 12.74 |

Overall silhouette IoU: baseline 0.615, result 0.778. Overall texture NCC: baseline 0.122, result 0.185. The result is a large improvement in mask explanation, texture agreement and temporal consistency. Residual colour mismatch is expected because the scan texture contains baked lighting from a different capture; mask and temporal evidence were therefore given higher weight.

## Changes and physical checks

- Fixed the whole-episode 180-degree long-axis ambiguity. The original CoCoNet rotation places the handle on the wrong image side.
- Replaced per-frame rotations during table contact with one pose per static interval, eliminating drift and flicker.
- Re-estimated camera-ray depth and lateral translation by silhouette registration against the exact mesh. The corrections are typically a few centimetres and vary smoothly.
- Replaced the CoCoNet rotation jumps (many 40-160 degree single-frame jumps) with image-registered rotations and a smooth SE(3) trajectory.
- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.
- During the 282 frames with confident hand contact and the mug lifted, median minimum hand-keypoint-to-mesh distance is 0.33 cm for the result versus 2.03 cm for the baseline.

## Confidence

- Static/table phases: high. Hundreds of nearly unoccluded frames support the constant poses, and the handle-side convention is unambiguous.
- Motion phases away from the mouth: medium-high. The object is large and visible; silhouette registration is stable.
- Frames nearest the mouth/hand occlusion: medium. The mask is partial, so orientation is carried by the smoother and FoundationPose prior.
- Depth: medium. The baseline depth is already calibrated to the human/floor; changes were admitted only when silhouette scale and lateral centroid improved. No floor snapping was applied.

## Reproducibility

Deterministic `solve.py` runtime: 387.4 s. Output key: `T_world_mesh` (F,4,4) float64 in the released world frame.
Random seed: 0. No network access is used by `solve.py`.
