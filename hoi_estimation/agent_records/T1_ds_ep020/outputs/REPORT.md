# Rice-cooker trajectory recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` as a `(553, 4, 4)` float64 array.
It maps the released `object_mesh.glb` coordinates to the released world frame
for every frame of `video.mp4`.

The released baseline is strong in this episode: its dynamic rotation follows
the real clockwise turn, and it does **not** contain a 180-degree segment flip.
The improvement therefore comes from enforcing the physical trajectory phases,
pooling evidence inside each rest phase, removing per-frame depth jitter, and
correcting a small systematic image-plane offset.

## Phases found

The phase boundaries were selected from the baseline unwrapped rotation speed,
the right-hand contact probability, the object screen-space motion, and visual
checks of the blue side handle. They are encoded as explicit constants in
`solve.py` with the same reasons.

| Frames | Phase | Evidence |
|---|---|---|
| 0-139 | Rest before contact | Constant image silhouette; no right-hand contact; baseline yaw varies by only 2.36 deg and its apparent depth wanders by 17.5 cm. |
| 140-203 | Held, still at rest | Right-hand contact becomes stable, but the object does not turn until about frame 209. Kept in the same constant-pose rest phase. |
| 204-419 | Clockwise rotation | Baseline yaw changes continuously by about 308 deg. The blue handle enters at the right image edge around frame 280, sweeps across the front, and reaches the left edge around frame 420. |
| 420-479 | Final placement translation | Rotation stops near frame 420, while object depth/translation continues changing by up to about 9 cm before settling. Included in the moving phase. |
| 480-552 | Final rest | Right-hand contact has ended; object silhouette and yaw are constant. Baseline depth still has 10.2 cm peak-to-peak jitter. |

## What changed and why

1. **Constant poses in rest phases.** For frames 0-203 and 480-552, all
   rotations were replaced by their chordal mean and all translations by the
   per-axis median. This pools 204 and 73 frames respectively instead of
   trusting each independently estimated depth.
2. **Translation smoothing while moved.** The moving-phase translation was
   filtered with a deterministic 21-frame, second-order Savitzky-Golay filter.
   This preserves the 10 cm placement motion but removes per-frame depth
   tracking noise. The mean second difference of translation fell from
   2.69 cm to 0.15 cm.
3. **Small pose correction.** A constant world offset of
   `[-0.008, +0.007, 0.0]` m was applied. The baseline projected-mask centroid
   was consistently `+2.08 px` in x and `-0.94 px` in y. A pooled silhouette
   fit over both static clips independently found the same approximately
   `-8 mm x, +7 mm y` correction. Depth was not altered: the baseline depth was
   calibrated against the body/floor and the per-frame depth was the main source
   of the static jitter being removed.
4. **Rotation kept from the baseline during motion.** A blue-handle texture
   track was used as an independent orientation check. Mesh vertices whose UV
   texture is blue rendered the same visible marker interval as the observed
   blue pixels; over the 279 overlapping frames the predicted/observed visible
   blue areas had correlation 0.996 and centroid errors of 1.60 px x / 0.73 px y.
   This rules out a 180-degree flip and supports the baseline turn curve.
   Savitzky-Golay filtering of the rotation was tested but introduced 7-8 deg
   overshoots during the fastest part of the turn, so it was rejected.
5. **Short smooth transitions.** Eight-frame cosine blends join the moving
   trajectory to the two constant rest poses. Boundary translation jumps are
   below 1 mm and boundary rotation jumps are below 0.12 deg.

## Validation table

Validation renders used the exact released mesh and `camera.json`, nvdiffrast
rasterization at full image resolution, and the released object masks. "Blue"
is the IoU between observed blue-handle pixels and mesh vertices classified as
blue from the released UV texture; `n/a` means the marker was on the hidden side.

| Frames | Phase | Mask IoU baseline | Mask IoU result | Blue IoU baseline | Blue IoU result | Confidence |
|---|---|---:|---:|---:|---:|---|
| 0-139 | Rest before contact | 0.8851 | **0.9261** | n/a | n/a | High |
| 140-203 | Held, still at rest | 0.9025 | **0.9291** | n/a | n/a | High |
| 204-419 | Clockwise rotation | 0.8899 | **0.9111** | 0.5754 | **0.6064** | High |
| 420-479 | Placement translation | 0.8788 | **0.9221** | 0.6821 | **0.7379** | Medium-high |
| 480-552 | Final rest | 0.8846 | **0.9584** | 0.6918 | **0.7237** | High |
| **All** | | **0.8882** | **0.9244** | **0.6286** | **0.6652** | |

Additional checks:

| Check | Baseline | Result |
|---|---:|---:|
| Mean object-mask centroid error, x | 2.08 px | **0.01 px** |
| Mean absolute object-mask centroid error, y | 0.98 px | 1.17 px |
| Mean second difference of translation | 2.69 cm | **0.15 cm** |
| 99th percentile translation step | 8.55 cm | **1.02 cm** |
| Peak-to-peak depth in frames 0-203 | 17.50 cm | **0 cm** |
| Peak-to-peak depth in frames 480-552 | 10.22 cm | **0 cm** |

The y centroid is slightly less centered than the baseline, but the larger and
more informative silhouette IoU improves in every phase. The correction was
chosen by silhouette overlap rather than centroid alone because the cooker's
asymmetric handle and lid make the mask centroid an imperfect proxy.

No ground truth, challenge annotation, or externally downloaded model/data was
used. No floor snapping was performed. The solve path does not modify the
baseline depth scale; it only removes temporal jitter and the image-alignment
offset.

## Confidence

- **0-203 high:** the object visibly rests and its pose is highly overdetermined
  after pooling. The hand contacts it late in this interval but does not move it.
- **204-419 high:** the turn is smooth and the independent blue-handle track
  follows the baseline orientation throughout the distinctive asymmetric part.
- **420-479 medium-high:** the object is being placed and the hand occludes part
  of it, but the translation end point is anchored by the following rest phase.
- **480-552 high:** the object is static, unobstructed, and pooled over 73 frames.

## Reproducibility and run time

Final command:

```bash
python /work/outputs/solve.py \
  --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

Wall-clock run time in this container: **0.9 s** on CPU. The script is
deterministic, uses no network, and reads only `inputs/reconstruction.npz`.
The heavier rendered diagnostics were used during development and are not
required by the final solver.
