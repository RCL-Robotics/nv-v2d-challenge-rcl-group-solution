# Tools used

No external tools, repositories, model weights or pip packages were downloaded or installed. Everything comes from the
container as provided.

| Tool | Version | Used for | Needed by `solve.py` |
|---|---|---|---|
| Python | 3.11 (conda, container) | everything | yes |
| torch | 2.5.1+cu124 (container) | Adam optimisation of per-frame poses on the GPU | yes |
| nvdiffrast | 0.4.0 (container) | differentiable rasterisation of the object mesh (silhouettes; textured renders only in experiments) | yes |
| trimesh | 5.1.1 (container) | mesh loading, convex hull (support face) | yes |
| numpy | 1.26.3 (container) | — | yes |
| scipy | 1.16.3 (container) | rotations (`scipy.spatial.transform`), KD-tree in `validate.py` | yes |
| opencv-python | 4.11.0.86 (container) | mask crop resizing, video decoding in experiments | yes |
| h5py | 3.16.0 (container) | reading `masks.h5` | yes |
| matplotlib | (container) | diagnostic plots in experiments only | no |

Data used by `solve.py`: only `/work/inputs` (`video.mp4` is not needed by the final method; it was used in
experiments for the photometric term and the background plate). `/pipeline`, `/weights` and `/code` were not used
by the final method. No network access is needed.

Tried and not adopted (no extra software involved): a photometric analysis-by-synthesis term (textured
nvdiffrast render composited over a median background plate built from the video). It agreed with the silhouette
term on well-visible frames, but it was unreliable when the object was small or occluded (frames ~290–560), so it
was dropped. See REPORT.md.
