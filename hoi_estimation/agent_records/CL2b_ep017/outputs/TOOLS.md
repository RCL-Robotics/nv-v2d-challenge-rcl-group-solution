# Tools used

Nothing was downloaded: no pip packages, git repositories or model weights were fetched, and `/work/tools` is empty.
Everything below was already installed in the container.

| Tool | Version | Used for | Needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | everything | yes |
| torch | 2.5.1+cu124 | Adam optimisation of poses, GPU tensors | yes |
| nvdiffrast | 0.4.0 | differentiable silhouette rendering, textured renders for inspection | yes |
| numpy | 1.26.3 | | yes |
| scipy | 1.16.3 | rotations, Gaussian smoothing, KD-tree (depth-cue analysis only) | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| opencv-python / opencv-contrib-python | 4.11.0.86 | crop resizing; video decoding and figures (analysis only) | yes (resize) |

Data used by `solve.py`: only `/work/inputs` (video frames are not even decoded by the final script; it uses
the SAM2 masks, the mesh, the camera, the baseline reconstruction and the human keypoints).
The read-only `/pipeline`, `/weights` and `/code` trees were inspected but are not used: I read the MHR-70
keypoint names in `/code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/metadata/mhr70.py`, and the
MoGe depth report.
No neural network was re-run (no FoundationPose, SAM2, MoGe or CoCoNet inference).

GPU: a single RTX 6000 Ada, with peak memory well below 9 GB (batched 128x128 silhouette renders). Total GPU time over
all experiments plus the final run was about 25 minutes.
