# Basket trajectory refinement

Produced `object_poses_world.npz`: `T_world_mesh` is float64, shape `(565, 4, 4)`, in the released world frame and original mesh-file coordinates.

## Changes and evidence

The result conservatively adjusts image-plane position in **151 frames**. The other **414 frames are exactly the supplied baseline**. Every rotation is exactly unchanged. The maximum translation correction is **1.741 cm**.

Sampled video/mesh overlays showed a persistent vertical offset before pickup. An exact perspective rasterization of the provided mesh confirmed that its baseline silhouette was approximately four pixels above the SAM object mask. Across accepted frames the median applied shift is **(-0.832, +3.412) pixels**, after conservative shrinkage and temporal filtering.

On the 151 edited frames, mean full-mask silhouette IoU improves from **0.74524 to 0.84182**. Every accepted correction improves actual rendered IoU by at least 0.025 and also improves the robust distance objective. These are image agreement measurements, not ground-truth pose or ADD measurements. See [alignment_examples.jpg](alignment_examples.jpg) for video, baseline outline (magenta), and refined outline (cyan); [object_poses_world.diagnostics.json](object_poses_world.diagnostics.json) records all edited indices. The automatic criteria also accept a few short, well-observed intervals during carrying.

## Method

`solve.py` loads only the supplied reconstruction, calibration, mesh and masks. It renders each baseline pose in a 256-pixel crop, then fits a bounded two-dimensional translation using a signed distance transform of the object mask. Distances are capped at five pixels. Dilated person-mask regions are ignored as negative evidence, while observed object pixels remain valid.

A candidate needs at least 85% predicted visibility, baseline IoU at least 0.65, objective improvement at least 0.12, and displacement below six pixels per axis. A nine-frame median, agreement within 1.5 pixels, and support in at least five of nine neighboring frames reject unstable fits. Gaussian smoothing, fading at reliability boundaries and 15% shrinkage limit segmentation/rasterization sensitivity. Each correction is subsequently checked by rendering the actual adjusted 3D pose.

The mesh-bounds centre moves along the viewing sphere: its **baseline Euclidean camera distance is preserved**, with numerical residual below `2e-15 m` in independent verification. The released ground plane is not used. Mesh scale, camera calibration and all baseline rotations are preserved. There are no manually edited frame indices or external fitting artifacts.

## Orientation investigation and limits

Inspected textured baseline renders and mesh-local 180-degree yaw alternatives throughout pickup, carrying and placement. The baseline has a 122.8-degree orientation step between frames 197 and 198, under substantial hand occlusion. The woven basket has visually similar ends, and these images did not establish which absolute yaw branch is correct. Changing an entire branch would risk a large ADD error. Local nine-frame rotation averaging was also tried on plausible orientation outliers; its silhouette agreement was mixed and often worse. Neither experiment is applied to the submitted poses.

Confidence is high that the accepted changes improve visible image alignment, and moderate that these small corrections improve 3D ADD. Absolute orientation ambiguity and baseline depth error remain unresolved. No ground truth, annotation, challenge trajectory, dataset download, or network access was used.

## Reproduction and validation

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script fixes seeds, runs offline and depends on no scratch files, `/pipeline`, `/weights`, or network services. It additionally writes diagnostic JSON and an alignment NPZ next to the requested pose output. The evidence image is supplementary; `make_evidence.py` can regenerate it from the supplied video and the final result.

A production run took **42.53 seconds**. A second independent process took **42.39 seconds** and produced bit-identical pose arrays and alignment measurements. Independent validation checked all 565 finite rigid transforms, output dtype/shape, exact preservation of rotations and unedited poses, and preservation of camera distance. The inherited baseline rotations have a maximum orthogonality residual of `5.95e-6`; these entries are intentionally preserved rather than re-orthogonalized. See `validation.json`. GPU rendering across all exploratory checks and both solver runs consumed less than five minutes of elapsed rendering-process time, within the 45-minute limit. No neural model inference was run.
