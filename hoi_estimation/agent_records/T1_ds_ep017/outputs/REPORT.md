# Traffic-cone 6-DoF trajectory recovery

## Result

- Output: `object_poses_world.npz`, key `T_world_mesh`, shape `(440, 4, 4)`, `float64`.
- Final solve command:
  `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
- Final run time: **17.4 s** on the available RTX 6000 Ada GPU. Peak GPU memory remained well below the 9 GB limit.
- No ground truth, challenge annotations, or challenge trajectory files were used.

## Diagnosed phases

The object masks, object optical flow, hand contacts, released tracker rotations, and rendered overlays support these phases:

| Frames | Phase | Treatment |
|---|---|---|
| 0-112 | Cone at rest on floor | One globally fitted pose for every frame |
| 113-116 | Grasp and lift transition | Smooth SE(3)/quaternion bridge into the tracked carry pose |
| 117-189 | Carried, mostly visible | Smoothed tracker trajectory |
| 190-254 | Carried behind the torso | Translation from interpolated two-hand midpoint; wide-window rotation smoothing |
| 255-339 | Transport and placement approach | Smoothed tracker trajectory |
| 340-439 | Cone at rest on table | One pooled pose for every frame |

The longest fully empty object-mask run is 200-246. SAM briefly reports a few object pixels around 247-251 and then empty again at 252-253, so the hand-derived translation gap is extended to 193-256.

## What was changed and why

### 1. Initial rest-phase symmetry branch

Rendering the released baseline and comparing against the floor-rest silhouettes exposed a long symmetry-branch error in frames 0-112. A pose search initialized at a 180-degree twist about the cone's local symmetry axis was optimized jointly over six rest frames. It converged to a **174.72-degree** twist and a small camera-frame translation correction of **(-1.70, -0.32, -0.03) cm**. The nonzero departure from exactly 180 degrees comes from the scanned mesh's small asymmetries and residual camera/pose error.

On the six fitting frames, pooled mask IoU increased from **0.739 to 0.945**. Over all 113 initial-rest frames, the independent checks below improve even more strongly. The one pose is reused exactly through frame 112; this is the single largest change in the submission.

### 2. Remove the tracker branch jump at pickup

The released trajectory rotates by about 99 degrees at frame 117 while the object image and hand positions change only slightly. A smoothstep quaternion/translation bridge spans frames 113-116 from the corrected rest pose to the first stable post-grasp pose. No static-pose constraint is applied once the cone is lifted.

### 3. Hand-anchored translation during occlusion

For frames 193-256, where the cone is completely or almost completely hidden, translation is reconstructed from the midpoint of the 21 right- and 21 left-hand MHR keypoints. The hand-to-object offset is interpolated between the visible anchors on either side of the occlusion; it is not assumed constant over the whole episode. Hand *orientation* was tested as a rotation source but was too noisy, so rotations use temporal continuity instead.

### 4. Trajectory smoothing

The carry segment uses deterministic Savitzky-Golay smoothing:

- rotation quaternions: window 9, cubic;
- translation: window 7, quadratic;
- the empty-mask interval uses a wider rotation window of 25, because no image evidence exists there and a rigid held object cannot physically jitter.

Quaternion signs are made hemisphere-consistent before filtering and normalized afterwards. This removes single-frame tracker jumps while retaining the gross lift, turn, and place motion. It reduces 90th-percentile rotation acceleration from **0.594 to 0.102 rad/frame²** and translation acceleration from **0.0933 to 0.0121 m/frame²**.

### 5. Static table phase and depth

Frames 340-439 are emitted with one exact rotation and one exact translation, pooled from the released table-rest poses. I did not snap the object onto the floor or table plane. I also did not apply a silhouette-only depth translation: the released camera-space depth is consistent with the aligned MoGe-2 object depth (about 2.75-2.85 m at the table), whereas an unconstrained silhouette fit would move the object several centimetres closer. The only depth-direction change in the initial pose fit is 0.3 mm.

## Validation (no ground truth)

These are internal checks only. Mask IoU is computed against SAM object masks; photometric NCC is the correlation of high-pass-filtered grayscale render and image inside the rendered/object mask. They are useful for relative comparison, not calibrated ADD.

| Phase | Frames | Scored | Mask IoU baseline | Mask IoU result | Photo NCC baseline | Photo NCC result |
|---|---:|---:|---:|---:|---:|---:|
| Initial rest | 0-112 | 113 | 0.819 | **0.945** | 0.213 | **0.623** |
| Pick transition | 113-116 | 4 | 0.817 | **0.831** | 0.271 | **0.497** |
| Carry visible | 117-189 | 73 | 0.716 | **0.722** | 0.275 | **0.288** |
| Carry occluded | 190-254 | 16 | 0.295 | **0.316** | 0.038 | **0.182** |
| Place approach | 255-339 | 85 | 0.727 | **0.727** | 0.358 | **0.364** |
| Table rest | 340-439 | 100 | 0.837 | **0.837** | 0.477 | **0.491** |
| All scored frames | 0-439 | 391 | 0.763 | **0.802** | 0.318 | **0.452** |

Temporal checks over the carry segment:

| Metric | Baseline | Result |
|---|---:|---:|
| Rotation acceleration, p90 (rad/frame²) | 0.594 | **0.102** |
| Translation acceleration, p90 (m/frame²) | 0.0933 | **0.0121** |
| Maximum rotation step (deg) | 61.5 | **19.4** |
| Maximum translation step (m) | 0.163 | **0.043** |

Structural checks on the output: all 440 matrices are finite `float64`; rotation determinants are 1.0 to 1e-6; the initial-rest and table-rest phases each have exactly zero pose spread.

## Confidence by phase

| Phase | Confidence | Reason |
|---|---|---|
| Initial rest, 0-112 | High | Pooled silhouette and texture evidence both strongly favour the symmetry correction; 113 frames are explained by one rigid pose |
| Pick transition, 113-116 | Medium | Only four frames, and hand occlusion limits direct evidence; the bridge avoids the baseline's 99-degree jump |
| Carry visible, 117-189 | High for smoothness, medium for absolute pose | Both image proxies improve, and temporal noise falls sharply; remaining uncertainty comes from fast rotation and partial hand occlusion |
| Carry occluded, 190-254 | Medium | No useful object pixels; translation is hand-anchored and rotation is continuity-constrained |
| Place approach, 255-339 | High | Object is visible again; checks improve slightly and acceleration falls by an order of magnitude |
| Table rest, 340-439 | High | A single pose explains 100 frames; photometric agreement improves and translation is pooled |

Overall confidence is **medium-high**. The largest intended correction is the initial rest-phase symmetry flip; it is supported by a large, phase-wide mask and photometric improvement rather than a single-frame decision.
