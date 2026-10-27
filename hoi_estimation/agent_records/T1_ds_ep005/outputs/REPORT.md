# Big Black Box: World-Frame Trajectory Recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` as `(1003, 4, 4)` `float64`.
Every transform is finite, has determinant 1 to numerical precision, and maps the
exact `object_mesh.glb` mesh coordinates into the released world frame.

The baseline rotation trajectory was retained as the rotation prior. I evaluated
textured renders of the baseline and of 180-degree alternatives about each local
principal axis. The alternatives reduced silhouette agreement and broke visible
handle/latch correspondence, so no global symmetry flip was applied. The main
gains came from pooling the two rest phases, removing isolated depth/tracker
noise, and re-registering translation against the actual image silhouette.

## Phases

| Index range | Phase | Evidence and treatment |
|---|---|---|
| 0-159 | Initial rest | Box upright and untouched; image silhouette and baseline pose are constant. One pooled rigid pose is used for all 160 frames. |
| 160-519 | Tip and roll | The person tips the box onto its side and rolls/moves it toward the camera. Rotation is kept close to the tracked motion; translation is re-registered and interpolated smoothly. |
| 520-849 | Manipulate on side | The box is reoriented while mostly on its broad side. Several rapid depth changes in the baseline are removed by keyframe registration and temporal interpolation. |
| 850-899 | Lift upright / set down | The box is brought upright and lowered. This transition receives a separate validation row because silhouette occlusion and motion are strongest here. |
| 900-1002 | Final rest | Box upright on the floor; the pose is exactly constant over the phase. |

These boundaries were checked visually in the video and against the baseline
velocity profile. Since the object is rigid, a pose change inside either rest
phase is physically impossible; the result enforces that constraint exactly.

## Method

1. **Rest-pose pooling.** For each rest phase, rotations are averaged on SO(3)
   with the SVD projection of the chordal mean. Translation registrations from
   several unobstructed frames are pooled with a component-wise median. This
   converts noisy per-frame estimates into one rigid pose per rest phase.
2. **Rotation trajectory.** Manipulation rotations are sign-continuous
   quaternions and are smoothed with a deterministic 7-frame Savitzky-Golay
   filter. They blend back to the exact rest anchors at the transitions.
3. **Translation re-registration.** At every 10th manipulation frame, the
   projected mesh silhouette is optimized against two independent image cues:
   (a) distance from its outer contour to Canny image edges near the object and
   (b) visible-mask IoU with the SAM object mask after excluding the person.
   A weak prior keeps ambiguous/occluded frames near the calibrated baseline.
4. **Global trajectory decoding.** Robustified absolute translation keyframes
   are interpolated with a C1 PCHIP spline. Rest intervals use their exact pooled
   pose rather than a filtered per-frame pose. This removes the baseline's
   10.9 cm worst one-frame translation jump without inventing pose flips.
5. **Depth decision.** The baseline projection was systematically too large,
   especially in the final phase. MoGe-2 object depth indicated moving farther
   from the camera, while image edges/masks indicated the same direction but a
   smaller magnitude during the initial phase. I followed the image-silhouette
   evidence, which is strongly supported by the exact mesh, and did not snap the
   box to the floor plane. This reduced the object-depth disagreement with MoGe
   in every phase (table below).

The final script is deterministic, CPU-only for the final solve, and reads only
the supplied inputs. It does not use `/pipeline` or any scratch artifact.

## Validation Without Ground Truth

Metrics are computed on all 1003 frames for visible-mask IoU; contour-to-edge
distance is sampled every fifth frame against Canny edges near the object.
Temporal steps are adjacent-frame rotation/translation changes. Larger IoU and
smaller edge distance/steps are better.

| Phase | Visible-mask IoU baseline -> result | Contour-to-edge px baseline -> result | Mean rotation step deg baseline -> result | Mean translation step cm baseline -> result |
|---|---:|---:|---:|---:|
| 0-159 rest | 0.908 -> **0.936** | 6.48 -> **5.62** | 0.445 -> **0.000** | 0.892 -> **0.000** |
| 160-519 tip/roll | 0.870 -> **0.933** | 5.19 -> **3.46** | 0.815 -> **0.544** | 1.668 -> **0.416** |
| 520-849 manipulate | 0.884 -> **0.938** | 3.01 -> **1.86** | 0.654 -> **0.420** | 1.702 -> **0.251** |
| 850-899 upright/set down | 0.856 -> **0.936** | 6.39 -> **3.53** | 1.001 -> **0.735** | 0.961 -> **0.596** |
| 900-1002 rest | 0.862 -> **0.952** | 5.87 -> **2.40** | 0.515 -> **0.003** | 0.632 -> **0.008** |
| All frames | 0.879 -> **0.937** | 4.81 -> **3.17** | 0.682 -> **0.371** | 1.414 -> **0.262** |

Additional checks:

* Worst adjacent translation jump: **10.85 cm -> 1.40 cm**.
* Worst adjacent rotation step: **4.94 deg -> 2.96 deg**.
* Baseline rest-phase spread: 0.68 deg / 1.45 cm median (initial) and
  0.67 deg / 0.62 cm (final). The result is exactly 0 in both phases.
* Median `MoGe-2 depth - rendered object depth` disagreement (cm):
  phase 0 `21.98 -> 15.30`, phase 1 `26.89 -> 10.66`, phase 2 `26.28 -> 12.80`,
  phase 3 `26.18 -> 8.72`, phase 4 `21.74 -> 5.83`. MoGe has its own bias, so
  this is treated as supporting evidence, not truth.
* Relative to the released floor plane, the median minimum signed mesh distance
  changes from -0.76 cm to -4.57 cm. I did not force floor contact: the released
  plane itself has several-centimetre foot-plane scatter, and enforcing it would
  visibly pull the final silhouette inside the real box edges. The result does
  not float; it slightly penetrates the released plane. This is the main
  unresolved physical/geometric ambiguity.

## Confidence

| Phase | Confidence | Reason |
|---|---|---|
| 0-159 | High | Long static interval, stable mask, exact pooled pose, 0.936 IoU. |
| 160-519 | High for translation; medium-high for rotation | Large object, strong outer edges, 0.933 IoU. Rotation follows the smooth baseline track. |
| 520-849 | High | Large visible silhouette and stable textured faces; 0.938 IoU and 1.86 px edge residual. |
| 850-899 | Medium-high | Fast tip/set-down and person occlusion, but the re-registered contour remains close to the real boundary. |
| 900-1002 | High | Long final rest; exact constant pose and 0.952 IoU. |

No ground truth or challenge annotations were accessed.

## Runtime and Reproduction

Final run:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Measured wall time in this container: **92.6 s** (about 1 min 34 s including
process startup), CPU-only, no GPU use. The script fixes RNGs, processes video
sequentially, and contains no fitted per-frame tables.
