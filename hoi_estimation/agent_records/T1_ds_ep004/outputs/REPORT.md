# Baseball-bat 6-DoF trajectory recovery

## Result

- Video: `inputs/video.mp4`, 803 frames at 30 fps, 1536 x 1152.
- Object: textured baseball bat, mesh-file coordinates.
- Output: `outputs/object_poses_world.npz`, key `T_world_mesh`, shape `(803, 4, 4)`, dtype `float64`.
- Released-frame convention and `camera.json` were used throughout.
- No ground-truth trajectory or challenge annotation was used.

The released CARI4D rotations were retained as the main per-frame hypothesis. The
largest gains came from (1) pooling each floor-contact phase into one rigid pose,
(2) re-fitting translation against person-excluded object masks, and (3) replacing
symmetry-branch errors selected by textured render-and-compare. All branch/model
choices are deterministic and encoded in `solve.py`.

## Phases found

| Frames | Phase | Evidence | Confidence |
|---:|---|---|---|
| 0-110 | Initial rest on floor; person approaches | Object mask and midpoint are static; zero contact probability; pooled texture/mask is sharp | High, 0.95 |
| 111-199 | Pickup and first lift | Contact begins around 109-112; mask motion increases | Medium-high, 0.80 |
| 200-314 | Slow carry and turning | Object moves with hand; several tracker branch changes | Medium, 0.65 |
| 315-411 | End-for-end rotation and body turn | Large object-mask orientation changes; textured rendering resolves reversed barrel/handle | Medium-low, 0.58 |
| 412-599 | Continued carry/turn | Mostly strong mask and texture evidence, with short hand occlusions | Medium-high, 0.78 |
| 600-745 | Final turning and placement | Human lowers and releases bat; object approaches floor pose | Medium-high, 0.80 |
| 746-802 | Final rest on floor | Person-excluded mask union and median appearance are static; no contact after 709 | High, 0.95 |

The final static pose is applied from frame 746; frames 746-749 are a short
geodesic blend from the placement trajectory, after which the pose is constant.

## Method

1. **Static-phase evidence pooling.** For frames 0-110 and 746-802, object masks
   are unioned and RGB is median-pooled only at pixels unoccluded by the person.
   This yields a near-complete object silhouette and texture observation for each
   rest phase.
2. **Rigid static-pose fitting.** The bat is rendered with its glTF texture and
   optimized against pooled silhouette IoU plus normalized colour correlation.
   Coordinate descent is used for translation then rotation (world axes and the
   local long axis).
3. **Carried-phase translation fitting.** For every frame, the released rotation
   is rendered at half resolution through nvdiffrast. Translation is optimized
   against mask IoU after explicitly excluding a dilated person mask. High-quality
   raw fits are blended with a Savitzky-Golay prior; low-visibility frames inherit
   more of the smooth prediction.
4. **Rotation branch hypotheses.** The released rotation, and its 180-degree flips
   about the mesh's x/y/z axes, were rendered with the actual textured glTF. A
   full-resolution texture NCC and person-excluded silhouette score were used to
   decode branch runs with a switch penalty. The resulting intervals are listed
   and commented in `solve.py`. Endpoint colour (light barrel vs. dark handle)
   was an independent check for the x/y choices.
5. **Temporal decoding.** Translation is low-pass filtered in the carried phase.
   Branch changes are converted into five-frame geodesic transitions. Remaining
   one-frame jumps above 50 degrees are replaced by geodesic interpolation between
   anchors two frames on either side. This removes tracker teleports without
   imposing a global constant-velocity assumption on real hand motion.
6. **Output normalization.** Every rotation is projected back to SO(3), every
   matrix is written as float64, and all finite/shape checks are enforced.

## Validation without ground truth

Metrics are computed on the same observations used for fitting; they are not
ground-truth ADD. They are useful because the released baseline and result are
scored by exactly the same renderer and masks.

- `IoU`: visible object-mask IoU after the renderer's silhouette is restricted to
  pixels outside the dilated person mask.
- `NCC`: per-channel normalized colour correlation between the textured render and
  the video inside the SAM object mask, excluding the person and eroded boundaries.
- `Score`: `IoU + 0.3 * max(NCC, -0.5)`.

### Baseline vs result

| Frames | Baseline IoU | Result IoU | Baseline NCC | Result NCC | Result score change |
|---:|---:|---:|---:|---:|---:|
| 0-110 | 0.748 | 0.797 | 0.674 | 0.749 | +0.072 |
| 111-199 | 0.687 | 0.770 | 0.484 | 0.476 | +0.081 |
| 200-299 | 0.486 | 0.626 | 0.255 | 0.363 | +0.164 |
| 300-399 | 0.306 | 0.522 | -0.252 | -0.043 | +0.249 |
| 400-499 | 0.483 | 0.599 | 0.219 | 0.318 | +0.146 |
| 500-599 | 0.588 | 0.700 | 0.531 | 0.478 | +0.096 |
| 600-745 | 0.675 | 0.748 | 0.510 | 0.570 | +0.091 |
| 746-802 | 0.774 | 0.767 | 0.736 | 0.843 | +0.025 |
| **All** | **0.589** | **0.691** | **0.385** | **0.459** | **+0.118** |

Aggregate render score improves from **0.713** to **0.831**. The result wins
silhouette evidence in seven of eight phases and substantially improves both
silhouette and texture in the difficult 300-399 segment. In 500-599 the mask IoU
improves while the colour score decreases slightly; the aggregate score still
improves. In the final rest phase silhouette is essentially tied and colour improves.

### Temporal and physical checks

| Check | Baseline | Result |
|---|---:|---:|
| Rotation steps > 60 deg/frame | 14 | 0 |
| Rotation steps > 100 deg/frame | 4 | 0 |
| Largest rotation step | 163.3 deg | 49.9 deg |
| Rotation-step p99 | 71.3 deg | 45.0 deg |
| Translation-step p95 | 0.078 m | 0.065 m |
| Translation-step p99 | 0.113 m | 0.089 m |
| Initial rest: position range over phase | up to 6.8 cm | 0 cm |
| Initial rest: 90th percentile rotation deviation | 10.3 deg | 0 deg |
| Final rest after blend: position range | up to 0.9 cm | 0.8 cm |

The rotation p95 is slightly higher than baseline because branch corrections are
distributed over a few transition frames rather than emitted as one-frame
teleports. P99, maximum, and the number of implausible jumps all improve.

## What changed and why

- The released rotations are preserved where their person-excluded silhouette and
  texture evidence remain strong.
- Translation is corrected in every carried frame using the object mask, which
  fixes several 10-20 cm baseline offsets during hand occlusions.
- Short/long x/y/z symmetry branches are corrected where full-resolution textured
  rendering consistently favoured a branch. These include the strongly visible
  barrel/handle reversal around frames 315-344.
- Static phases are no longer left as noisy per-frame estimates: each is one
  pooled rigid pose over the whole contact phase.
- No floor-penetration term was hard-enforced because the released floor plane is
  global and its local offset at the bat is not exact; enforcing it reduced mask
  evidence in validation.

## Reproducibility and runtime

Run offline with:

```bash
python /work/outputs/solve.py \
  --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The final run completed in **1m18s** wall-clock on one RTX 6000 Ada GPU while
remaining below the 9 GB memory limit. Random seeds are fixed; no network access,
scratch result, or precomputed per-frame table is used.
