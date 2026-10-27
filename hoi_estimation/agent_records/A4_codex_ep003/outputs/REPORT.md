# Monocular mug trajectory repair

The output contains a mesh-file-to-released-world transform for every one of the 781 video frames. The camera-to-object-centre distance from the released reconstruction is preserved separately in every frame. No floor projection, global alignment, mesh recentering, object rescaling, or human-based recalibration is applied.

## Evidence and method

The released baseline contains 30 successive rotation changes above 45 degrees between successive frames, with a maximum of approximately 159 degrees. In frames 300, 301 and 422 its mesh +Z axis points upward in the image even though the visible mug remains lid-up. The baseline textured renders also put the handle on the camera-right side over substantial intervals, whereas the video shows it on the left. The released mesh is used with its texture and original coordinates; its +X direction points into the handle and +Z points from lid toward base.

`solve.py` loads the supplied GLB using the specified trimesh scene-to-geometry conversion, then renders it using nvdiffrast and the released camera. It fits visible silhouettes and the silver/black texture pattern, excluding person-covered pixels from penalties for otherwise invisible mesh. Exposure gain and offset are nuisance parameters in the texture comparison.

For every fourth frame, twelve handle azimuths at 30-degree spacing receive continuous local refinement. Initialization uses the silhouette's principal image direction and the baseline's camera-axis tilt, with a nine-frame median to reject isolated axis flips. A small bounded image-scale nuisance parameter prevents the observed render/video scale mismatch from forcing a depth change or an artificial tilt. This scale affects fitting only: it is never applied to the mesh, camera calibration, translation distance, or final transforms.

Dynamic programming selects a temporally consistent candidate sequence. Two explicit observations from the video are encoded and commented in the script: the mug is never inverted, and its handle remains on the camera-left side. Candidates with the lid-to-base axis pointing upward or a mesh +X camera-x component above -0.35 are rejected. The latter is a deliberately loose bound on the clearly leftward handle projection; it does not fix its angle toward/away from the camera. Occluded handle orientations inherit evidence from nearby visible views rather than relying on body-only texture scores. All frames then receive local refinement, interpolation and short symmetric temporal filtering. Camera distance is restored exactly after filtering. Framewise range fluctuations already present in the baseline are therefore inherited. The largest image-plane corrections occur around tracking failures; changing rotation also shifts the mesh body relative to its mesh-coordinate origin, so a handle-side repair needs a corresponding centre adjustment.

All fitting is offline and uses only released data. No challenge annotations, outside images, ground-truth poses, downloaded model or external pose-estimation system were used.

## Confidence and limits

Confidence is high in correcting the isolated inversion errors and in correcting clearly wrong handle sides. Confidence is moderate in the precise axial angle and local image-plane adjustments. Tilt toward/away from the camera and exact handle angle remain less constrained during hand occlusion and blur. Silhouette alignment and continuity are observable checks, not a measurement of ground-truth ADD. The method deliberately retains the baseline distance even where apparent object scale could suggest a different distance.

The reproducible command is:

```sh
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The solver recomputes its candidates and requires no exploratory outputs. `TOOLS.md` gives dependencies. `input_sha256.json` identifies the released inputs used. Final numerical checks, runtime and diagnostic links are recorded below after validation.

## Final validation

The exact required offline command completed successfully. Checks use the final saved poses and the released inputs, without alignment. Mask IoU uses the original K and unscaled mesh projections, with person-covered background excluded equally for baseline and final. It is a fitting diagnostic, not a held-out accuracy metric.

| Check | Baseline | Final |
|---|---:|---:|
| Mean visible-mask IoU, all frames | 0.6570 | 0.8327 |
| Rotation steps above 45 degrees | 30 | 0 |
| Median rotation step | 5.84 deg | 1.51 deg |
| Maximum rotation step | 158.85 deg | 19.64 deg |

Mask overlap improves in 753/781 frames. Median image-centre adjustment: 5.20 px; maximum: 22.94 px. Median rotation change from baseline: 162.82 degrees, reflecting the handle-side repairs rather than an assumed axial symmetry.

The NPZ has key `T_world_mesh`, shape `(781, 4, 4)`, dtype `float64`, with finite entries throughout. Maximum change in per-frame Euclidean camera distance: 1.33e-15 m. Maximum rotation orthogonality error: 1.33e-15; determinant range: [0.9999999999999992, 1.0000000000000013]. Homogeneous rows are valid.

Final complete offline run: 439.2 s (7.32 min), including video/mask I/O, fitting and writing outputs, on one RTX 6000 Ada GPU. Summed wall-clock bounds of all CUDA-using exploration, discarded fitting trials, final generation and verification are conservatively below 35 minutes; concurrent experiments are counted separately in that bound. This is below the 45-minute GPU budget.

Review `comparison.jpg` for video/baseline/final overlays, `validation.json` for numerical checks, `runtime.json` for final-run timing, and `fit_diagnostics.npz` / `solver_candidates.npz` for fitting evidence. Exploratory scripts and images are archived in `experiments/`; `solve.py` does not read them. All confidence claims concern visible evidence and continuity; ground-truth ADD has not been evaluated.
