# T1_ds_ep025

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T15:01:50Z

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/AGENT_TASK.md && printf '\\n---FILES---\\n' && find /work -maxdepth 2 -type f -printf '%p\\n' | head -200"
```
<details><summary>output</summary>

```
# Task: recover the full 6-DoF object trajectory of one monocular human-object-interaction video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose; no symmetry handling, no alignment), averaged over thresholds 1-9 cm. Rotation
and translation both matter; a  …[+11045 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,220p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
# Task: recover the full 6-DoF object trajectory of one monocular human-object-interaction video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose; no symmetry handling, no alignment), averaged over thresholds 1-9 cm. Rotation
and translation both matter; a  …[+9786 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '60,115p'"
```
<details><summary>output</summary>

```
    60	object alone (silhouette scale on an object a few dozen pixels wide, with an estimated K) is far less reliable. You
    61	may correct depth, but only when several independent cues agree (hand contact with the reconstructed human, static
    62	object on the released floor plane, MoGe-2 depth, metric mesh size vs. image size), and quantify their disagreement.
    63	Do not snap the object onto the floor plane: it is a fit with several-cm tolerance (see `foot_plane_dist_stats` in
    64	`ground_plane.json`), and the object may rest on other surfaces.
    65	
    66	## Inputs (`/work/inputs`)
    67	- `video.mp4` - the only camera (static exo camera), 1536x1152.
    68	- `task.txt` - what the person does. `episode.json` - object name, frame count.
    69	- `object_mesh.glb` - the exac …[+4583 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '40,68p'; nl -ba /work/AGENT_TASK.md | sed -n '115,140p'; printf '\\n---inputs---\\n'; cat /work/inputs/task.txt; cat /work/inputs/episode.json; cat /work/inputs/camera.json; cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
    40	   registrations (e.g. render-and-compare optimisation from several initialisations, or FoundationPose
    41	   re-registration on chosen frames).
    42	4. **Global decoding over time**: choose one hypothesis per frame (or per segment) jointly, e.g. dynamic programming /
    43	   Viterbi with a data cost from (2) and a transition cost that penalises rotation and translation changes (much more
    44	   in static phases), then refine continuously, e.g. a joint optimisation of all poses with image terms plus
    45	   smoothness / constant-velocity terms on SE(3), and rigid hand-object attachment while carried.
    46	5. **Anchors and propagation**: find frames where the pose is unambiguous (object large, little occluded, distinctive
    47	   view, high evidence margin between hyp …[+3944 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
p='/work/inputs/reconstruction.npz'
z=np.load(p,allow_pickle=True)
print(z.files)
for k in z.files:
 a=z[k]
 print(k, a.shape,a.dtype, 'finite',np.isfinite(a).all() if a.dtype.kind in 'fc' else '-')
 if k in ('T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet'):
  print(' first t',a[0,:3,3], 'last t',a[-1,:3,3])
PY"
```
<details><summary>output</summary>

```
['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (859, 4, 4) float32 finite True
 first t [-0.19504395  0.26613533  3.369296  ] last t [-0.18014984  0.25561875  3.5264919 ]
T_cam_mesh_foundationpose (859, 4, 4) float32 finite True
 first t [0.2450437  0.23881237 2.5760324 ] last t [0.23907165 0.24942364 2.3697045 ]
T_cam_mesh_coconet (859, 4, 4) float32 finite True
 first t [0.24498121 0.23846297 2.4859161 ] last t [0.2374365  0.25153273 2.3113613 ]
human_vertices_world (859, 4610, 3) float32 finite True
human_keypoints_world (859, 70, 3) float32 finite True
contact_prob_right_left (859, 2) float32 finite True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 4 -type f -printf '%p %s\\n' | sort | head -300"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1981
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/05_object_pose.json 1772
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 932131004
/pipeline/depth/moge2/depth_report.json 1808
/pipeline/depth/moge2/intrinsics.pkl 91461
/pipeline/depth/moge2/raw.npy 3039953024
/pipeline/export/episode_000025/edex 563
/pipeline/export/episode_000025/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000025/human_masks/front_stereo_camera_left.h5 20860635
/pipeline/export/episode_000025/images/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000025/images/front_ste …[+528 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=nw=1 /work/inputs/video.mp4; python - <<'PY'
import h5py
p='/work/inputs/masks.h5'
with h5py.File(p,'r') as f:
 print(list(f.keys()))
 for k in f:
  print(k,f[k].shape,f[k].dtype)
PY"
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=28.633333
nb_frames=859
['object', 'person']
object (859, 1152, 1536) bool
person (859, 1152, 1536) bool

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m),m.vertices.shape,m.faces.shape)
print('bounds',m.bounds,'extents',m.extents,'centroid',m.centroid)
print('visual',type(m.visual), getattr(m.visual,'uv',None).shape if hasattr(m.visual,'uv') else None)
print('components',len(m.split(only_watertight=False)))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> (259484, 3) (86534, 3)
bounds [[-0.04554274 -0.04614785 -0.13553183]
 [ 0.04659176  0.04503646  0.14767875]] extents [0.09213449 0.09118432 0.28321058] centroid [-2.06236196e-11  6.26810388e-11  7.37185415e-11]
visual <class 'trimesh.visual.texture.TextureVisuals'> (259484, 2)
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/trimesh/base.py", line 1454, in split
    return graph.split(self, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trimesh/graph.py", line 391, in split
    return mesh.submesh(
           ^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trimesh/base.py", line 2957, in submesh
    return util.submesh(
      …[+1038 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'mkdir -p /work/scratch/frames /work/outputs /work/tools; ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf "fps=2,scale=384:288,tile=6x8:padding=4:margin=4" -frames:v 2 /work/scratch/contact_%02d.jpg; ls -lh /work/scratch/contact_*.jpg'
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 284K Oct 20 18:56 /work/scratch/contact_01.jpg
-rw-rw-r-- 1 hanzhou hanzhou  90K Oct 20 18:56 /work/scratch/contact_02.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; F=len(T)
t=T[:,:3,3]; Rm=T[:,:3,:3]
ang=R.from_matrix(Rm).as_rotvec()
# local delta in world rotvec magnitude, relative delta rotvec
rel=np.stack([ (Rm[i-1].T@Rm[i]) for i in range(1,F)])
dang=R.from_matrix(rel).magnitude()*180/np.pi
dt=np.linalg.norm(np.diff(t,axis=0),axis=1)
# align signs? print summary by every 30
print('idx time trans xyz rotdeg_from_ref deltadeg delta_cm contact')
for i in list(range(0,F,30))+[F-1]:
 ref=R.from_matrix(Rm[0].T@Rm[i]).magnitude()*180/np.pi
 dd=dang[i-1] if i else 0
 print(f'{i:3d} {i/30:6.2f} {t[i,0]: .3f} {t[i,1]: .3f} {t[i,2]: .3f} ref={ref:6.1f} d={dd:6.2f} cm={100*(dt[i-1] if i else 0):6.2f} cp={z[\"contact_prob_right_left\"][i]}')
print('max delta deg',dang.max(),np.argmax(dang)+1,'pct>10',np.mean(dang>10),'pct>20',np.mean(dang>20))
print('max trans cm',dt.max()*100,np.argmax(dt)+1)
print('contact >0.5 frames',np.where(z['contact_prob_right_left'].max(1)>.5)[0][[0,-1]] if np.any(z['contact_prob_right_left'].max(1)>.5) else None)
# ranges
print('wrist/keypoint distances to object center or nearest transformed vertex later')
PY"
```
<details><summary>output</summary>

```
idx time trans xyz rotdeg_from_ref deltadeg delta_cm contact
  0   0.00 -0.195  0.266  3.369 ref=   0.0 d=  0.00 cm=  0.00 cp=[1.2155471e-15 2.7162167e-13]
 30   1.00 -0.180  0.249  3.434 ref=   5.6 d=  5.11 cm=  0.63 cp=[3.1674276e-15 3.6060996e-12]
 60   2.00 -0.193  0.263  3.399 ref=   9.6 d=  0.00 cm=  0.00 cp=[2.9697980e-10 2.0772418e-07]
 90   3.00 -0.192  0.262  3.402 ref=   9.6 d=  0.00 cm=  0.00 cp=[1.5589691e-08 3.7932098e-02]
120   4.00 -0.179  0.252  3.361 ref=  13.0 d=  3.92 cm=  1.76 cp=[3.2802614e-08 9.9996960e-01]
150   5.00 -0.036 -0.005  3.084 ref=  79.2 d=  2.30 cm=  1.80 cp=[5.3136564e-08 9.9999726e-01]
180   6.00  0.020 -0.189  3.057 ref=  91.8 d= 25.05 cm=  0.94 cp=[9.1198437e-08 9.9999994e-01]
210   7.00 -0.189 -0.263  3.091 ref= 139.7 d=  7.13 cm=  0.59 cp=[0.471052 …[+2270 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/.stages/01_depth.json; cat /pipeline/.stages/02_export.json; cat /pipeline/.stages/03_human_initialization.json; cat /pipeline/.stages/04_depth_alignment.json; cat /pipeline/.stages/05_object_pose.json; cat /pipeline/depth/moge2/depth_report.json'
```
<details><summary>output</summary>

```
{
  "name": "01_depth",
  "outputs": [
    {
      "mtime_ns": 1791266930638565005,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000025/depth/moge2/raw.npy",
      "size": 3039953024
    },
    {
      "mtime_ns": 1791266930799565005,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000025/depth/moge2/intrinsics.pkl",
      "size": 91461
    },
    {
      "mtime_ns": 1791266933223564995,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000025/depth/moge2/depth_report.json",
      "size": 1808
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/mhr_wild_depth.py",
      "--video",
      "/home/hanzhou/Projects/v2d/run/work/episode_000025/episode_000025.0.co …[+10499 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 3 -type f | sort | sed -n '1,240p'"
```
<details><summary>output</summary>

```
/code/reconstruction/.cursorignore
/code/reconstruction/.dockerignore
/code/reconstruction/.gitattributes
/code/reconstruction/.gitignore
/code/reconstruction/.venv/.gitignore
/code/reconstruction/.venv/.lock
/code/reconstruction/.venv/CACHEDIR.TAG
/code/reconstruction/.venv/bin/activate
/code/reconstruction/.venv/bin/activate.bat
/code/reconstruction/.venv/bin/activate.csh
/code/reconstruction/.venv/bin/activate.fish
/code/reconstruction/.venv/bin/activate.nu
/code/reconstruction/.venv/bin/activate.ps1
/code/reconstruction/.venv/bin/activate.xsh
/code/reconstruction/.venv/bin/activate_this.py
/code/reconstruction/.venv/bin/deactivate.bat
/code/reconstruction/.venv/bin/f2py
/code/reconstruction/.venv/bin/flask
/code/reconstruction/.venv/bin/jp.py
/code/reconstruction/.venv/bin/numpy-config …[+7037 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; k=z['human_keypoints_world']; c=z['contact_prob_right_left']
print('keypoint names maybe no')
# print hand point ranges and center means
for name,ids in [('rwrist',[21]),('rhand',list(range(21,42))),('lwrist',[42]),('lhand',list(range(42,63)))]:
 p=k[:,ids].mean(1)
 print(name, p[0],p[400],p[-1])
# nearest hand keypoint to object origin
for i in [0,50,90,100,120,200,300,400,500,600,700,790,830,858]:
 p=k[i,21:63]
 d=np.linalg.norm(p-T[i,:3,3],axis=1)
 print(i,'minhand',d.min(), 'obj',T[i,:3,3], 'cp',c[i])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, cv2, h5py
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
cam=__import__('json').load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
with h5py.File('/work/inputs/masks.h5','r') as f:
 om=f['object'][:]; pm=f['person'][:]
for i in list(range(0,859,30))+[858]:
 X=T[i,:3,3]; xc=Tcw[:3,:3]@X+Tcw[:3,3]; uv=K@xc; uv=uv[:2]/uv[2]
 ys,xs=np.where(om[i]); 
 if len(xs): print(i, 'proj',uv.astype(int),'mask cen',int(xs.mean()),int(ys.mean()),'bbox',xs.min(),ys.min(),xs.max(),ys.max(),'area',len(xs),'person area',pm[i].sum(), 'maskoverlap person',np.logical_and(om[i],pm[i]).sum())
 else: print(i,'empty object mask')
PY"
```
<details><summary>output</summary>

```
0 proj [847 653] mask cen 844 652 bbox 830 607 860 697 area 2590 person area 19260 maskoverlap person 0
30 proj [846 653] mask cen 844 651 bbox 829 604 860 698 area 2725 person area 25081 maskoverlap person 133
60 proj [848 655] mask cen 844 651 bbox 829 603 859 697 area 2660 person area 29156 maskoverlap person 113
90 proj [848 655] mask cen 844 652 bbox 829 605 860 697 area 2668 person area 34249 maskoverlap person 124
120 proj [842 648] mask cen 838 647 bbox 820 600 857 694 area 2745 person area 35411 maskoverlap person 61
150 proj [790 552] mask cen 787 552 bbox 772 512 802 591 area 2074 person area 39645 maskoverlap person 0
180 proj [775 496] mask cen 771 493 bbox 740 455 802 530 area 2295 person area 35017 maskoverlap person 28
210 proj [839 476] mask cen 834 477 bbox 801 441 869 51 …[+2408 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np, os
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
im=m.visual.material.image
print(im.size,im.mode)
im.save('/work/scratch/mesh_texture.png')
# material props
print(m.visual.material)
# save screenshots? use trimesh scene export? Mesh with vertex colors maybe
print('uv range',m.visual.uv.min(0),m.visual.uv.max(0))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace/v2d_cari4d -maxdepth 3 -type f | sort | head -100; grep -R \"T_world_mesh\\|coconet\\|refined.pth\\|pose_abs\" -n /workspace/v2d_cari4d /code/reconstruction/modules/v2d_cari4d 2>/dev/null | head -200"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/__init__.py
/workspace/v2d_cari4d/lib/cari4d/Utils.py
/workspace/v2d_cari4d/lib/cari4d/__init__.py
/workspace/v2d_cari4d/lib/cari4d/estimater.py
/workspace/v2d_cari4d/lib/cari4d/render_h5_codec.py
/workspace/v2d_cari4d/lib/download_weights.py
/workspace/v2d_cari4d/lib/pack_masks.py
/workspace/v2d_cari4d/lib/pip-constraints.txt
/workspace/v2d_cari4d/lib/pyproject.toml
/workspace/v2d_cari4d/lib/run_inference.py
/workspace/v2d_cari4d/lib/run_training.py
/workspace/v2d_cari4d/lib/v2d_cari4d_lib.egg-info/PKG-INFO
/workspace/v2d_cari4d/lib/v2d_cari4d_lib.egg-info/SOURCES.txt
/workspace/v2d_cari4d/lib/v2d_cari4d_lib.egg-info/dependency_links.txt
/workspace/v2d_cari4d/lib/v2d_cari4d_lib.egg-info/requires.txt
/workspace/v2d_cari4d/lib/v2d_cari4d_lib.egg-info/top_level.txt
 …[+22895 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /code/reconstruction && git rev-parse HEAD && git status --short | head; python - <<'PY'
import cv2
print('cv2',cv2.__version__)
try:
 import nvdiffrast; print('nvdiffrast',nvdiffrast.__file__)
except Exception as e: print('nvdiffrast err',e)
import torch; print('torch',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY"
```
<details><summary>output</summary>

```
cv2 4.11.0
nvdiffrast /opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py
torch 2.5.1+cu124 True NVIDIA RTX 6000 Ada Generation

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -type f -printf '%p %s\\n' | sort"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1981
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/05_object_pose.json 1772
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 932131004
/pipeline/depth/moge2/depth_report.json 1808
/pipeline/depth/moge2/intrinsics.pkl 91461
/pipeline/depth/moge2/raw.npy 3039953024
/pipeline/export/episode_000025/edex 563
/pipeline/export/episode_000025/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000025/human_masks/front_stereo_camera_left.h5 20860635
/pipeline/export/episode_000025/images/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000025/images/front_ste …[+528 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,240p' /workspace/v2d_cari4d/lib/cari4d/estimater.py; sed -n '1,220p' /workspace/v2d_cari4d/lib/cari4d/Utils.py"
```
<details><summary>output</summary>

```
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import tempfile
import uuid

import torch

import Utils
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
from scipy.spatial.transform import Rotation as R


def geodesic_distance(R1: R, R2: R) -> float:
    """
    Parameters
    ----------
    R1, R2 : scipy.spatial.transform.Rotation
        Rotation …[+16444 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mat=m.visual.material
print(type(mat))
print(mat.__dict__)
print(type(m.visual))
print(m.visual.__dict__.keys())
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.material.PBRMaterial'>
{'_data': {'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=1464x1440 at 0x7B150B5071D0>, 'doubleSided': True, 'name': '_texture'}}
<class 'trimesh.visual.texture.TextureVisuals'>
dict_keys(['vertex_attributes', '_cache', 'material', 'face_materials', 'mesh'])

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
im=m.visual.material.baseColorTexture
print(im.size,im.mode,np.asarray(im).mean((0,1)),np.asarray(im).min(),np.asarray(im).max())
im.save('/work/scratch/mesh_texture.png')
PY"
```
<details><summary>output</summary>

```
(1464, 1440) RGB [ 53.93678326 101.61663299  97.15387115] 0 240

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"def make_mesh_tensors\\|def nvdiffrast_render\\|def transform_pts\\|def glcam\\|def projection_matrix\" /workspace/v2d_cari4d/lib/cari4d/Utils.py; sed -n '1,180p' /workspace/v2d_cari4d/lib/cari4d/Utils.py; sed -n '180,430p' /workspace/v2d_cari4d/lib/cari4d/Utils.py"
```
<details><summary>output</summary>

```
86:def make_mesh_tensors(mesh, device='cuda', max_tex_size=None):
120:def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
545:def transform_pts(pts,tf):
778:def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosur …[+16383 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '120,260p' /workspace/v2d_cari4d/lib/cari4d/Utils.py; sed -n '770,815p' /workspace/v2d_cari4d/lib/cari4d/Utils.py"
```
<details><summary>output</summary>

```
def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
  '''Just plain rendering, not support any gradient
  @K: (3,3) np array
  @ob_in_cams: (N,4,4) torch tensor, openCV camera
  @projection_mat: np array (4,4)
  @output_size: (height, width)
  @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
  @light_dir: in cam space
  @light_pos: in cam space
  '''
  if glctx is None:
    if context == 'gl':
      glctx = dr.RasterizeGLContext()
    elif context=='cuda':
      glctx = dr.RasterizeCudaContext()
    else:
 …[+7297 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '815,850p' /workspace/v2d_cari4d/lib/cari4d/Utils.py; sed -n '535,560p' /workspace/v2d_cari4d/lib/cari4d/Utils.py"
```
<details><summary>output</summary>

```

  # Draw the images upright and modify the projection matrix so that OpenGL
  # will generate window coords that compensate for the flipped image coords.
  elif window_coords == 'y_down':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])
  else:
    raise NotImplementedError

  return proj



def symmetry_spec_from_info(info, rot_angle_discrete=5):
  from lib_mhr.object_symmetry import bop_object_symmetry

  return bop_object_symmetry(info, continuous_angle_degrees=rot_angle_discrete)


def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
  '''Used for Pose Refine …[+1103 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_crop_sheet.py <<'PY'
import cv2, h5py, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object'][:]
inds=list(range(0,859,15))
thumbs=[]
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); assert ok
 ys,xs=np.where(om[i]);
 if len(xs):
  cx=int((xs.min()+xs.max())/2); cy=int((ys.min()+ys.max())/2)
 else: cx,cy=768,576
 x0=max(0,cx-100); x1=min(fr.shape[1],cx+100); y0=max(0,cy-130); y1=min(fr.shape[0],cy+130)
 crop=fr[y0:y1,x0:x1]
 crop=cv2.resize(crop,(180,234),interpolation=cv2.INTER_AREA)
 cv2.putText(crop,f'{i} {i/30:.1f}s',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
thumbs.append(crop)
cols=6; rows=(len(thumbs)+cols-1)//cols
canvas=np.zeros((rows*238,cols*184,3),np.uint8)
for j,im in enumerate(thumbs):
 canvas[(j//cols)*238:(j//cols)*238+234,(j%cols)*184:(j%cols)*184+180]=im
cv2.imwrite('/work/scratch/object_crops.jpg',canvas)
PY
python /work/scratch/make_crop_sheet.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nl -ba /work/scratch/make_crop_sheet.py'
```
<details><summary>output</summary>

```
     1	import cv2, h5py, numpy as np
     2	cap=cv2.VideoCapture('/work/inputs/video.mp4')
     3	with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object'][:]
     4	inds=list(range(0,859,15))
     5	thumbs=[]
     6	for i in inds:
     7	 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); assert ok
     8	 ys,xs=np.where(om[i]);
     9	 if len(xs):
    10	  cx=int((xs.min()+xs.max())/2); cy=int((ys.min()+ys.max())/2)
    11	 else: cx,cy=768,576
    12	 x0=max(0,cx-100); x1=min(fr.shape[1],cx+100); y0=max(0,cy-130); y1=min(fr.shape[0],cy+130)
    13	 crop=fr[y0:y1,x0:x1]
    14	 crop=cv2.resize(crop,(180,234),interpolation=cv2.INTER_AREA)
    15	 cv2.putText(crop,f'{i} {i/30:.1f}s',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
    16	thumbs.append(crop)
    17	co …[+272 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '16s/"'^/ /'"' /work/scratch/make_crop_sheet.py; python /work/scratch/make_crop_sheet.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_baseline.py <<'PY'
import sys,os,numpy as np,torch,trimesh,cv2,json
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d import Utils
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mesh.visual.material.image=mesh.visual.material.baseColorTexture
mt=make_mesh_tensors(mesh)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam']); K=np.array(cam['K']); Tcw=np.linalg.inv(Twc)
inds=[0,75,90,105,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,800,820,858]
poses=np.stack([Tcw@T[i] for i in inds]).astype(np.float32)
glctx=dr.RasterizeCudaContext()
cap=cv2.VideoCapture('/work/inputs/video.mp4')
cols=5; thumbs=[]
for b0 in range(0,len(inds),5):
 ids=inds[b0:b0+5]; ps=torch.tensor(poses[b0:b0+5],device='cuda')
 col,dep,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=ps,glctx=glctx,mesh_tensors=mt,use_light=False)
 col=(col.cpu().numpy()*255).astype(np.uint8)
 for j,i in enumerate(ids):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); assert ok
  mask=(col[j].max(2)>0)
  ys,xs=np.where(mask)
  ov=fr.copy(); ov[mask]=(0.45*fr[mask]+0.55*col[j][mask]).astype(np.uint8)
  # contour
  cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIM …[+609 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_flips.py <<'PY'
import sys,os,numpy as np,torch,trimesh,cv2,json
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mesh.visual.material.image=mesh.visual.material.baseColorTexture
mt=make_mesh_tensors(mesh)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam']); K=np.array(cam['K']); Tcw=np.linalg.inv(Twc)
inds=[0,105,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780]
rots=[np.eye(4),np.diag([-1,-1,1,1]),np.diag([1,-1,-1,1]),np.diag([-1,1,-1,1])]
glctx=dr.RasterizeCudaContext(); cap=cv2.VideoCapture('/work/inputs/video.mp4')
rows=[]
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); assert ok
 Ps=torch.tensor(np.stack([ (Tcw@T[i]@q).astype(np.float32) for q in rots]),device='cuda')
 cols,_,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=Ps,glctx=glctx,mesh_tensors=mt,use_light=False)
 cols=(cols.cpu().numpy()*255).astype(np.uint8)
 # bbox from baseline render
 mask=(cols[0].max(2)>0); ys,xs=np.where(mask); cx=int(xs.mean()); cy=int(ys.mean())
 x0=max(0,cx-100);x1=min(1536,cx+100);y0=max(0,cy-140);y1=min(1152,cy+140)
 rr=[]
 for j,c in enumerate(cols):
  m=(c.max(2)>0); ov=fr.copy(); ov[m]=(0.45*fr[m]+0.55* …[+356 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_zoom.py <<'PY'
exec(open('/work/scratch/render_flips.py').read().split('inds=')[0])
inds=[0,150,300,450,600,720,780]
rots=[np.eye(4),np.diag([-1,-1,1,1]),np.diag([1,-1,-1,1]),np.diag([-1,1,-1,1])]
glctx=dr.RasterizeCudaContext(); cap=cv2.VideoCapture('/work/inputs/video.mp4'); rows=[]
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); assert ok
 Ps=torch.tensor(np.stack([ (Tcw@T[i]@q).astype(np.float32) for q in rots]),device='cuda')
 cols,_,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=Ps,glctx=glctx,mesh_tensors=mt,use_light=False); cols=(cols.cpu().numpy()*255).astype(np.uint8)
 mask=(cols[0].max(2)>0);ys,xs=np.where(mask);cx=int(xs.mean());cy=int(ys.mean());x0=cx-55;x1=cx+55;y0=cy-80;y1=cy+80
 rr=[]
 raw=fr[y0:y1,x0:x1]; raw=cv2.resize(raw,(220,320),interpolation=cv2.INTER_NEAREST); cv2.putText(raw,f'obs {i}',(4,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1); rr.append(raw)
 for j,c in enumerate(cols):
  m=(c.max(2)>0); ov=fr.copy(); ov[m]=(0.35*fr[m]+0.65*c[m]).astype(np.uint8)
  cr=cv2.resize(ov[y0:y1,x0:x1],(220,320),interpolation=cv2.INTER_NEAREST); cv2.putText(cr,f'{[\"base\",\"rz\",\"rx\",\"ry\"][j]} {i}',(4,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1);rr.append(cr)
 rows.append(np.concatenate(rr,1))
cv2.imwrite('/work/scratch/flip_zoom.jpg',np.concatenate(rows,0))
PY
python /work/scratch/render_zoom.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
for i in [0,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,858]:
 for s in [1,-1]:
  x=T[i,:3,:3]@np.array([0,0,.1477*s])+T[i,:3,3]
  xc=Tcw[:3,:3]@x+Tcw[:3,3]; uv=K@xc; uv=uv[:2]/uv[2]
  print(i,s,uv.round(1),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
0 1 [847.1 605.9]; 0 -1 [848.3 700.8]; 
150 1 [787.5 509.1]; 150 -1 [794.4 596.4]; 
180 1 [798.2 458.7]; 180 -1 [752.2 534.3]; 
210 1 [869.2 509.5]; 210 -1 [809.8 442.9]; 
240 1 [883.6 562.3]; 240 -1 [892.7 471.3]; 
270 1 [915.4 599.3]; 270 -1 [1001.8  568.8]; 
300 1 [947.1 502.6]; 300 -1 [857.9 476.6]; 
330 1 [822.  447.8]; 330 -1 [823.8 538.1]; 
360 1 [751.4 518.3]; 360 -1 [785.5 595. ]; 
390 1 [757.6 515.9]; 390 -1 [780.8 596.4]; 
420 1 [843.2 474.7]; 420 -1 [757.  448.3]; 
450 1 [872.6 504. ]; 450 -1 [881.3 413.1]; 
480 1 [889.6 509.4]; 480 -1 [952.5 539. ]; 
510 1 [939.4 564.8]; 510 -1 [862.4 612.9]; 
540 1 [761.4 455.1]; 540 -1 [760.8 530.8]; 
570 1 [719.8 515.9]; 570 -1 [786.5 492.2]; 
600 1 [729.3 515.2]; 600 -1 [789.2 482.5]; 
630 1 [759.1 538.8]; 630 -1 [696.9 516. ]; 
660 1 [745 …[+254 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import struct,json
p='/work/inputs/object_mesh.glb'
b=open(p,'rb').read(); magic,ver,n=struct.unpack_from('<III',b,0); off=12; chunks=[]
while off<n:
 l,t=struct.unpack_from('<II',b,off); dat=b[off+8:off+8+l]; chunks.append((t,dat)); off+=8+l
j=json.loads(chunks[0][1].decode('utf-8').rstrip('\\x00 '))
print(json.dumps(j.get('materials'),indent=2)[:4000])
print('textures',j.get('textures')); print('samplers',j.get('samplers')); print('images',j.get('images'))
print('extensionsUsed',j.get('extensionsUsed'))
# print minmax uv accessor
prim=j['meshes'][0]['primitives'][0]
print(prim)
for i,a in enumerate(j['accessors']): print(i,a.get('type'),a.get('componentType'),a.get('count'),a.get('min'),a.get('max'),a.get('bufferView'))
PY"
```
<details><summary>output</summary>

```
[
  {
    "pbrMetallicRoughness": {
      "baseColorTexture": {
        "index": 0
      },
      "metallicFactor": 0.0
    },
    "name": "_texture",
    "doubleSided": true
  }
]
textures [{'source': 0}]
samplers None
images [{'bufferView': 2, 'mimeType': 'image/png'}]
extensionsUsed None
{'attributes': {'POSITION': 1, 'TEXCOORD_0': 2}, 'indices': 0, 'mode': 4, 'material': 0}
0 SCALAR 5125 259602 [0] [259483] 0
1 VEC3 5126 259484 [-0.04554273560643196, -0.04614785313606262, -0.13553182780742645] [0.046591755002737045, 0.04503646492958069, 0.14767874777317047] 1
2 VEC2 5126 259484 [0.003372000064700842, 0.003586292266845703] [0.9969244599342346, 0.996720016002655] 3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/score_spin.py <<'PY'
import sys,os,numpy as np,torch,trimesh,cv2,json
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mesh.visual.material.image=mesh.visual.material.baseColorTexture
mt=make_mesh_tensors(mesh)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam']); K=np.array(cam['K']); Tcw=np.linalg.inv(Twc)
import h5py
with h5py.File('/work/inputs/masks.h5','r') as f: om=f['object'][:]; pm=f['person'][:]
inds=[0,105,150,210,240,300,360,420,450,510,600,630,660,690,720,750,780,840]
angles=np.arange(0,360,15)
glctx=dr.RasterizeCudaContext(); cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); assert ok; obs=fr[:,:,::-1].astype(np.float32)/255
 Ps=[]
 for a in angles:
  Rz=np.eye(4); Rz[:3,:3]=SR.from_euler('z',a,degrees=True).as_matrix()
  Ps.append(Tcw@T[i]@Rz)
 Ps=torch.tensor(np.array(Ps,dtype=np.float32),device='cuda')
 cols,_,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=Ps,glctx=glctx,mesh_tensors=mt,use_light=False)
 cols=cols.cpu().numpy()
 target=om[i] & ~pm[i]
 if target.sum()<50: target=om[i]
 res=[]
 for a,c in zip(angles,cols):
  rm=c.max( …[+1131 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

FRAME 0 target 2590 iou0 0.8072413790319858
by MAE: [(285, 0.0576, 0.76, 0.911, 0.818), (300, 0.0581, 0.767, 0.914, 0.815), (105, 0.0603, 0.736, 0.847, 0.818), (270, 0.0605, 0.747, 0.911, 0.842), (315, 0.0611, 0.773, 0.921, 0.81), (90, 0.0616, 0.729, 0.845, 0.82)]
by corr: [(0, 0.0681, 0.798, 0.953), (15, 0.0679, 0.793, 0.944), (345, 0.0673, 0.789, 0.952), (30, 0.0653, 0.786, 0.926), (330, 0.0646, 0.781, 0.938), (315, 0.0611, 0.773, 0.921)]

FRAME 105 target 2565 iou0 0.7274214657305559
by MAE: [(300, 0.0537, 0.798, 0.921, 0.732), (285, 0.0542, 0.784, 0.92 …[+7464 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; F=len(T)
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
for name in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 A=z[name]; Tc=np.stack([Tcw@T[i] for i in range(F)])
 dr=R.from_matrix(np.einsum('fji,fjk->fik',Tc[:,:3,:3],A[:,:3,:3])).magnitude()*180/np.pi
 dt=np.linalg.norm(Tc[:,:3,3]-A[:,:3,3],axis=1)
 print(name,'rot median/mean/p90/max',np.median(dr),dr.mean(),np.percentile(dr,90),dr.max(),'trans median,p90,max',np.median(dt),np.percentile(dt,90),dt.max())
 print('corr rot first',dr[:10], 'tr first',dt[:10])
# fp to coconet
A=z['T_cam_mesh_foundationpose']; B=z['T_cam_mesh_coconet']
dr=R.from_matrix(np.einsum('fji,fjk->fik',A[:,:3,:3],B[:,:3,:3])).magnitude()*180/np.pi
dt=np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1)
print('fp-coco rot',np.median(dr),np.percentile(dr,90),dr.max(),'trans',np.median(dt),np.percentile(dt,90),dt.max())
# static baseline T stats first 90 and last 50
for sl in [slice(0,90),slice(800,859)]:
 t=T[sl,:3,3]; rr=R.from_matrix(T[sl,:3,:3])
 print(sl,'t mean/std',t.mean(0),t.std(0),'range',t.max(0)-t.min(0))
 # median rotation geodesic
 rmed=R.from_matrix(np.median(T[sl,:3,:3],axis=0).reshape(1,3,3))
PY"
```
<details><summary>output</summary>

```
T_cam_mesh_foundationpose rot median/mean/p90/max 8.892946792453781 21.466091906125037 48.26281425698872 172.66762360049825 trans median,p90,max 0.16523784432332886 0.33774516117466974 2.0738780542833792
corr rot first [4.52120183 3.96391383 3.92667726 3.58942315 3.85699771 3.64580189
 4.37823925 5.32741749 5.34365508 5.4136325 ] tr first [0.08376511 0.08631893 0.10023327 0.09987015 0.10858097 0.09768583
 0.095313   0.10448028 0.09130633 0.07857566]
T_cam_mesh_coconet rot median/mean/p90/max 1.2931183310882963 1.2618250855557553 1.3428651581729114 1.354126521743993 trans median,p90,max 0.07646401650473296 0.09962374291622042 0.1039419919016153
corr rot first [1.15821842 1.1581333  1.1581913  1.15879183 1.15868221 1.15937748
 1.15933707 1.16036523 1.160356   1.15974219] tr first [0.0617473  …[+508 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
for sl in [(0,90),(0,50),(800,859),(812,859)]:
 Rs=T[sl,:3,:3]; ref=R.from_matrix(Rs).mean().as_matrix()
 d=R.from_matrix(np.einsum('fji,jk->fik',Rs,ref)).magnitude()*180/np.pi
 t=T[sl,:3,3]
 print(sl,'rot med mean p90 max',np.median(d),d.mean(),np.percentile(d,90),d.max(),'t med',np.median(t,0),'std',t.std(0))
PY"
```
<details><summary>output</summary>

```
(0, 90) rot med mean p90 max 4.807580377443938 4.807580377443938 4.807580377443939 4.807580377443939 t med [-0.1935541   0.26406443  3.3858023 ] std [0.00148985 0.00207089 0.01650631]
(0, 50) rot med mean p90 max 4.807580377443938 4.807580377443938 4.807580377443939 4.807580377443939 t med [-0.19304535  0.26325536  3.3888688 ] std [0.00199861 0.00287998 0.01957273]
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
IndexError: index 859 is out of bounds for axis 0 with size 859

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
for sl in [(0,90),(0,50),(800,859),(812,859)]:
 A=T[sl[0]:sl[1]]; Rs=A[:,:3,:3]; ref=R.from_matrix(Rs).mean().as_matrix()
 d=R.from_matrix(np.transpose(Rs,(0,2,1))@ref).magnitude()*180/np.pi
 t=A[:,:3,3]
 print(sl,'rot med mean p90 max',np.median(d),d.mean(),np.percentile(d,90),d.max(),'t med',np.median(t,0),'std',t.std(0))
PY"
```
<details><summary>output</summary>

```
(0, 90) rot med mean p90 max 2.4254034472779478 3.318043676384537 5.099790192241054 18.809816708323897 t med [-0.19104673  0.26283538  3.3993359 ] std [0.00448763 0.00562659 0.02033216]
(0, 50) rot med mean p90 max 3.434025417742267 3.7295368333127676 4.458794014353519 17.6651874734812 t med [-0.1890792   0.26001894  3.4100041 ] std [0.00512712 0.00695866 0.02666453]
(800, 859) rot med mean p90 max 6.650818095827045 11.379203461023849 31.212732869447894 51.37947642936557 t med [-0.17991929  0.2551244   3.5278502 ] std [0.0124434  0.00380216 0.0534433 ]
(812, 859) rot med mean p90 max 0.1157912351573447 0.2235572770743007 0.1157912351573447 2.299863976704996 t med [-0.17965712  0.2548191   3.5293946 ] std [0.001421   0.00175408 0.00992547]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/plot_traj.py <<'PY'
import numpy as np, matplotlib.pyplot as plt, json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; F=len(T); f=np.arange(F)
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
Tc=np.stack([Tcw@T[i] for i in range(F)])
# quaternion continuity
q=R.from_matrix(T[:,:3,:3]).as_quat(); q=np.where(np.sum(q[1:]*q[:-1],1,keepdims=True)<0,-q[1:],q[1:]) # wrong starts
# careful
out=[q[0]]
for qi in q[1:]: out.append(-qi if np.dot(out[-1],qi)<0 else qi)
q=np.array(out); e=R.from_quat(q).as_euler('xyz',degrees=True)
dang=R.from_matrix(np.transpose(T[:-1,:3,:3],(0,2,1))@T[1:,:3,:3]).magnitude()*180/np.pi
dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
cp=z['contact_prob_right_left']; k=z['human_keypoints_world']
hand=k[:,21:63]; td=np.min(np.linalg.norm(hand-T[:,None,:3,3],axis=2),axis=1)
fig,axs=plt.subplots(5,1,figsize=(18,13),sharex=True)
axs[0].plot(f,T[:,:3,3]); axs[0].set_ylabel('world xyz m'); axs[0].legend(['x','y','z'])
axs[1].plot(f,Tc[:,:3,3]); axs[1].set_ylabel('cam xyz m'); axs[1].legend(['x','y','z'])
axs[2].plot(f,e); axs[2].set_ylabel('Euler deg'); axs[2].legend(['x','y','z']); axs[2].set_ylim(-200,200)
axs[3].plot(np.arange(1,F),dang,label='rot delta deg'); axs[3].plot(np.arange(1,F),dt*100,label='trans delta cm'); axs[3].legend();axs[3].set_ylim(0,80)
axs[4].plot(f,cp); axs[4].plot(f,td,'k',label='h …[+317 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/plot_traj.py", line 19, in <module>
    axs[2].plot(f,e); axs[2].set_ylabel('Euler deg'); axs[2].legend(['x','y','z']); axs[2].set_ylim(-200,200)
    ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/matplotlib/axes/_axes.py", line 1792, in plot
    lines = [*self._get_lines(self, *args, data=data, **kwargs)]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/matplotlib/axes/_base.py", line 331, in __call__
    yield from self._plot_args(
               ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/matplotlib/axes/_base.py", line 509, in _plot_args
    raise ValueError(f"x and y must have same first dimension, but "
ValueError: x and y mus …[+65 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/plot_traj.py'
s=open(p).read();s=s.replace(\"q=np.where(np.sum(q[1:]*q[:-1],1,keepdims=True)<0,-q[1:],q[1:]) # wrong starts\\n# careful\\nout=[q[0]]\\nfor qi in q[1:]: out.append(-qi if np.dot(out[-1],qi)<0 else qi)\",\"out=[q[0]]\\nfor qi in q[1:]: out.append(-qi if np.dot(out[-1],qi)<0 else qi)\")
open(p,'w').write(s)
PY
python /work/scratch/plot_traj.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 print(list(f.keys()))
 def vis(name,obj):
  if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype)
 f.visititems(vis)
PY"
```
<details><summary>output</summary>

```
['aligned', 'alignment', 'frame_names', 'raw']
aligned/front_stereo_camera_left (859,) object
alignment/front_stereo_camera_left/scale (859,) float32
alignment/front_stereo_camera_left/shift (859,) float32
alignment/front_stereo_camera_left/valid_count (859,) int32
frame_names/front_stereo_camera_left (859,) object
raw/front_stereo_camera_left (859,) object

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for g in ['aligned','raw']:
  o=f[g]['front_stereo_camera_left']; print(g,type(o),list(o.keys()) if hasattr(o,'keys') else '')
  if hasattr(o,'keys'):
   for k in o: print(' ',k,o[k].shape,o[k].dtype)
PY"
```
<details><summary>output</summary>

```
aligned <class 'h5py._hl.dataset.Dataset'> 
raw <class 'h5py._hl.dataset.Dataset'> 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for g in ['aligned','raw']:
  d=f[g]['front_stereo_camera_left']; print(g,d.shape,d.dtype)
PY"
```
<details><summary>output</summary>

```
aligned (859,) object
raw (859,) object

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for g in ['aligned','raw']:
  d=f[g]['front_stereo_camera_left'][0]; print(g,type(d),getattr(d,'shape',None),getattr(d,'dtype',None)); print(d if np.size(d)<10 else np.asarray(d).reshape(-1)[:5])
PY"
```
<details><summary>output</summary>

```
aligned <class 'numpy.ndarray'> (581647,) uint8
[137  80  78  71  13]
raw <class 'numpy.ndarray'> (505296,) uint8
[137  80  78  71  13]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for g in ['aligned','raw']:
  b=f[g]['front_stereo_camera_left'][0]; a=cv2.imdecode(b,cv2.IMREAD_UNCHANGED); print(g,a.shape,a.dtype,a.min(),a.max(),np.percentile(a,[1,50,99]))
 print('scale',f['alignment/front_stereo_camera_left/scale'][0],f['alignment/front_stereo_camera_left/shift'][0])
PY"
```
<details><summary>output</summary>

```
aligned (1152, 1536) uint16 1852 8796 [1951. 4169. 8238.]
raw (1152, 1536) uint16 1498 7113 [1578. 3372. 6662.]
scale 1.2366531 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/filter_candidates.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); F=len(T)
cp=z['contact_prob_right_left'].max(1)
first=int(np.argmax(cp>.5)); last=int(F-1-np.argmax(cp[::-1]>.5))
print('contact',first,last)

def qcont(q):
 out=[q[0]]
 for x in q[1:]: out.append(-x if np.dot(out[-1],x)<0 else x)
 return np.array(out)

def qmed(q,rad,iters=2):
 q=q.copy(); n=len(q)
 for it in range(iters):
  out=[]
  for i in range(n):
   ids=np.arange(max(0,i-rad),min(n,i+rad+1)); cand=q[ids]
   # median candidate minimize robust sum angles
   dots=np.abs(cand@cand.T); ang=np.arccos(np.clip(dots,-1,1))
   j=ids[np.argmin(ang.sum(1))]
   out.append(q[j])
  q=qcont(np.array(out))
 return q

def robust_quad_series(Y,rad,thr,robust=True):
 # Y NxD euclidean
 N=len(Y); out=np.empty_like(Y)
 for i in range(N):
  ids=np.arange(max(0,i-rad),min(N,i+rad+1)); x=(ids-i).astype(float); yy=Y[ids]
  med=np.median(yy,0); w=np.ones(len(ids))
  for it in range(2):
   A=np.stack([np.ones_like(x),x,x*x],1)*np.sqrt(w[:,None])
   yw=yy*np.sqrt(w[:,None]); coef=np.linalg.lstsq(A,yw,rcond=None)[0]
   pred=np.stack([np.ones_like(x),x,x*x],1)@coef
   if robust:
    err=np.linalg.norm(yy-pred,axis=1)/thr
    w=1/np.maximum(1,err**2) # cauchy
  A=np.stack([np.ones_like(x),x,x*x],1)
  out[i]=coef[0]
 return out

def filter_pose(T,rad=4,rthr=12,tthr=0.08,static=True, rot_me …[+3321 chars]
```
<details><summary>output</summary>

```
contact 96 796
base {'rot_med': 5.111361741972479, 'rot_p90': 20.43434646390443, 'rot_mean': 8.773550020963048, 'rot_max': 145.1903515331527, 'trans_med': 0.013256261602599996, 'trans_p90': 0.02994324360894867, 'trans_max': 0.13427998936921537, 'accmed': 0.013958907145751806}
f 3 {'rot_med': 3.3581497927324615, 'rot_p90': 11.092461308601457, 'rot_mean': 5.084532679809201, 'rot_max': 126.62794746423846, 'trans_med': 0.009773419903117718, 'trans_p90': 0.019793384089559704, 'trans_max': 0.14937521793087088, 'accmed': 0.004076096775934125}
f 5 {'rot_med': 3.2954164668404435, 'rot_p90': 10.299030365293774, 'rot_mean': 4.588665610127216, 'rot_max': 85.15931940258906, 'trans_med': 0.008990401261880655, 'trans_p90': 0.017910109232393393, 'trans_max': 0.14960193054136856, 'accmed': 0.00255951480941 …[+529 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/mask_metrics.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2,time
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda')
G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
# dilate person modestly due SAM edge
ker=np.ones((5,5),np.uint8)
perd=np.empty_like(per)
for i in range(len(per)): perd[i]=cv2.dilate(per[i].astype(np.uint8),ker)>0
z=np.load('/work/scratch/candidates.npz'); names=[k for k in z.files if k not in ('first','last')]
print('names',names)
inds=np.arange(0,len(obj),3)
ctx=dr.RasterizeCudaContext()
allres={}
for name in names:
 T=z[name].astype(np.float32); vals=[]; t0=time.time()
 for b0 in range(0,len(inds),12):
  ids=inds[b0:b0+12]
  poses=torch.tensor(np.stack([Tcw@T[i] for i in ids]),device='cuda')
  homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
  mtx=P[None]@G[None] …[+1238 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'f3', 'f5', 'f7', 'f9']
Traceback (most recent call last):
  File "/work/scratch/mask_metrics.py", line 26, in <module>
    mtx=P[None]@G[None]@poses
        ~~~~~~~~~~~~~~~^~~~~~
RuntimeError: expected scalar type Double but found Float

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/dtype=torch.float32,device='cuda')/dtype=torch.float32,device='cuda')/\" /work/scratch/mask_metrics.py; python - <<'PY'
p='/work/scratch/mask_metrics.py';s=open(p).read();s=s.replace(\"P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda')\",\"P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda')\")
open(p,'w').write(s)
PY
python /work/scratch/mask_metrics.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'f3', 'f5', 'f7', 'f9']
Traceback (most recent call last):
  File "/work/scratch/mask_metrics.py", line 26, in <module>
    mtx=P[None]@G[None]@poses
        ~~~~~~~~~~~~~~~^~~~~~
RuntimeError: expected scalar type Double but found Float

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')/G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')/\" /work/scratch/mask_metrics.py; grep -n '"'^P='"\\|"'^G='"' /work/scratch/mask_metrics.py; python - <<'PY'
p='/work/scratch/mask_metrics.py';s=open(p).read();s=s.replace(\"G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')\",\"G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')\")
# actual g line inspect
print([l for l in s.splitlines() if l.startswith('P=') or l.startswith('G=')])
PY"
```
<details><summary>output</summary>

```
8:P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda')
9:G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
["P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda')", "G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')"]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/poses=torch.tensor(np.stack(\\[Tcw@T\\[i\\] for i in ids\\]),device='cuda')/poses=torch.tensor(np.stack([Tcw@T[i] for i in ids]),dtype=torch.float32,device='cuda')/\" /work/scratch/mask_metrics.py; python /work/scratch/mask_metrics.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'f3', 'f5', 'f7', 'f9']
base IoU 0.0012334237053773062 0.0013139512803819438 visIoU 0.0011051561961108681 0.0011809165269095455 CE 99.24930898126526 time 8.768928289413452
f3 IoU 0.0012334237053773062 0.0013139512803819438 visIoU 0.0011051561961108681 0.0011809165269095455 CE 99.24930898126526 time 8.813495397567749
Traceback (most recent call last):
  File "/work/scratch/mask_metrics.py", line 35, in <module>
    ys,xs=np.where(r); yt,xt=np.where(tgt)
          ^^^^^^^^^^^
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/rm=rast\\[...,3\\].flip(1).cpu().numpy()>=0/rm=rast[...,3].flip(1).cpu().numpy()>0/' /work/scratch/mask_metrics.py; python /work/scratch/mask_metrics.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'f3', 'f5', 'f7', 'f9']
base IoU 0.6894899307535464 0.731187472813949 visIoU 0.7723512453781898 0.8095238095234744 CE 5.898781214838638 time 5.738786458969116
f3 IoU 0.6971317572870941 0.738735327527172 visIoU 0.7804541007496593 0.8113975576658474 CE 5.796737958850854 time 5.656301498413086
f5 IoU 0.700384946326444 0.736142625607541 visIoU 0.7836212317852144 0.8120770432062404 CE 5.756005525364371 time 5.692124843597412
f7 IoU 0.702027343438985 0.7379488838561185 visIoU 0.7861993129466132 0.8152507676556728 CE 5.708930280419227 time 5.9414978 …[+545 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/mask_metrics.npz'); inds=z['inds']
for name in ['base','f5','f9']:
 a=z[name]
 for sl in [(0,96),(797,859)]:
  m=(inds>=sl[0])&(inds<sl[1]); ii=np.where(m)[0]; o=np.argsort(-a[ii,1])[:8]
  print(name,sl,[(int(inds[ii[j]]),round(float(a[ii[j],1]),3),round(float(a[ii[j],2]),1)) for j in o])
PY"
```
<details><summary>output</summary>

```
base (0, 96) [(9, 0.917, 1.0), (21, 0.915, 2.3), (15, 0.906, 3.9), (27, 0.906, 2.5), (30, 0.903, 2.4), (12, 0.902, 2.5), (24, 0.897, 3.1), (6, 0.883, 1.9)]
base (797, 859) [(816, 0.929, 3.5), (843, 0.922, 3.4), (837, 0.921, 3.5), (858, 0.921, 3.3), (819, 0.917, 3.2), (834, 0.917, 3.3), (825, 0.909, 3.1), (831, 0.909, 3.3)]
f5 (0, 96) [(33, 0.873, 4.4), (39, 0.872, 4.9), (36, 0.87, 5.0), (42, 0.859, 5.1), (15, 0.849, 4.3), (48, 0.847, 5.3), (21, 0.846, 4.6), (51, 0.845, 5.0)]
f5 (797, 859) [(816, 0.948, 3.4), (813, 0.947, 2.5), (843, 0.944, 3.3), (804, 0.944, 2.8), (837, 0.944, 3.4), (858, 0.942, 3.2), (819, 0.94, 3.0), (834, 0.937, 3.2)]
f9 (0, 96) [(33, 0.873, 4.4), (39, 0.872, 4.9), (36, 0.869, 5.0), (42, 0.858, 5.1), (15, 0.849, 4.4), (48, 0.847, 5.3), (21, 0.845, 4.6), (51, 0.845, 5.0) …[+168 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_anchor_search.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2,time
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda'); G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
ker=np.ones((5,5),np.uint8); perd=np.empty_like(per)
for i in range(len(per)): perd[i]=cv2.dilate(per[i].astype(np.uint8),ker)>0
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(np.float32)
ctx=dr.RasterizeCudaContext(); homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
def render_masks(poses):
 poses=torch.tensor(poses,dtype=torch.float32,device='cuda'); mtx=P[None]@G[None]@poses
 clip=(mtx[:,None]@homo[None,:,:,None])[...,0]; rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536))
 return rast[...,3].flip(1).cpu().numpy()>0
for seg in [(0,96),(797,859)]:
 ids=np.arange(seg[0],seg[1],3) …[+717 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
SEG (0, 96) best
27 0.9116 0.8791
21 0.9112 0.8821
9 0.9072 0.8606
6 0.9056 0.8606
30 0.9053 0.8628
24 0.8984 0.8646
12 0.8921 0.8463
15 0.8911 0.8383
3 0.8811 0.8245
0 0.8681 0.7946
18 0.8601 0.7911
33 0.8332 0.7528
36 0.8262 0.7439
63 0.8239 0.7503
69 0.8239 0.7503
SEG (797, 859) best
818 0.9088 0.8556
854 0.9086 0.8555
857 0.9086 0.8555
821 0.9085 0.8553
797 0.9083 0.8316
848 0.9082 0.8552
845 0.9082 0.8552
842 0.9082 0.8556
851 0.9082 0.8552
836 0.9082 0.8558
833 0.9082 0.8558
827 0.908 0.8555
824 0.908 0.8555
839 0.908 0.8555
830 0.908 0.8555

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/spin_metrics2.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
from scipy.spatial.transform import Rotation as SR
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mesh.visual.material.image=mesh.visual.material.baseColorTexture; mt=make_mesh_tensors(mesh)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
with h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
ang=np.arange(0,360,30); ctx=dr.RasterizeCudaContext(); cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,9,27,838,840,858]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read(); obs=fr[:,:,::-1].astype(np.float32)/255; tgt=obj[i]&~per[i]
 Ps=[]
 for a in ang:
  Rz=np.eye(4);Rz[:3,:3]=SR.from_euler('z',a,degrees=True).as_matrix();Ps.append(Tcw@T[i]@Rz)
 cols,_,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=torch.tensor(np.array(Ps,dtype=np.float32),device='cuda'),glctx=ctx,mesh_tensors=mt,use_light=False);cols=cols.cpu().numpy()
 print('\\nF',i)
 for a,c in zip(ang,cols):
  rm=c.max(2)>.01; inter=rm&tgt
  x=c[inter];y=obs[inter]
  pred=np.empty_like(x)
  for ch in range(3):
   A=np.stack([x[:,ch],np.ones(len(x))],1); …[+683 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

F 0
  0 mae 0.0681 corr 0.798 cc 0.953 lab 4.56 n2341
 30 mae 0.0653 corr 0.786 cc 0.926 lab 5.33 n2334
 60 mae 0.0643 corr 0.722 cc 0.870 lab 5.38 n2328
 90 mae 0.0616 corr 0.729 cc 0.845 lab 5.06 n2332
120 mae 0.0626 corr 0.727 cc 0.856 lab 4.91 n2349
150 mae 0.0693 corr 0.710 cc 0.893 lab 5.16 n2366
180 mae 0.0709 corr 0.716 cc 0.915 lab 5.23 n2363
210 mae 0.0652 corr 0.743 cc 0.918 lab 5.04 n2348
240 mae 0.0636 corr 0.742 cc 0.916 lab 4.97 n2353
270 mae 0.0605 corr 0.747 cc 0.911 lab 4.87 n2369
300 mae 0.0581 corr 0.767 cc 0.914 lab 4.67 n2343
330 mae  …[+3074 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/filter_overlay_sheet.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh.visual.material.image=mesh.visual.material.baseColorTexture;mt=make_mesh_tensors(mesh)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
z=np.load('/work/scratch/candidates.npz'); T0=z['base'];T1=z['f9']
m=np.load('/work/scratch/mask_metrics.npz');inds=m['inds'];gain=m['f9'][:,1]-m['base'][:,1]
# dynamic top gains
cand=inds[(inds>96)&(inds<797)];g=gain[(inds>96)&(inds<797)]; order=np.argsort(-g)
sel=[]
for j in order:
 if len(sel)>=24:break
 if all(abs(int(cand[j])-x)>4 for x in sel):sel.append(int(cand[j]))
print(sel)
ctx=dr.RasterizeCudaContext();cap=cv2.VideoCapture('/work/inputs/video.mp4');thumbs=[]
for i in sel:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();assert ok
 Ps=torch.tensor(np.stack([Tcw@T0[i],Tcw@T1[i]]).astype(np.float32),device='cuda')
 cols,_,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=Ps,glctx=ctx,mesh_tensors=mt,use_light=False);cols=(cols.cpu().numpy()*255).astype(np.uint8)
 masks=[c.max(2)>0 for c in cols]
 ys,xs=np.where(masks[0]|masks[1]);cx=int(xs.mean());cy=int(ys.mean());x0=max(0,cx-100);x1=min(1536,cx+100);y0=max(0,cy-130);y1 …[+917 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
[741, 693, 660, 408, 603, 402, 291, 471, 684, 708, 612, 282, 651, 249, 642, 672, 750, 315, 717, 378, 417, 240, 627, 360]
Traceback (most recent call last):
  File "/work/scratch/filter_overlay_sheet.py", line 24, in <module>
    cv2.drawContours(ov,[cv2.findContours(cv2.UMat(m.astype(np.uint8)),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]],-1,(0,255,0),2)
                                                   ^^^^^^^^
AttributeError: 'NpzFile' object has no attribute 'astype'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '/cv2.UMat/d' /work/scratch/filter_overlay_sheet.py; python /work/scratch/filter_overlay_sheet.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
[741, 693, 660, 408, 603, 402, 291, 471, 684, 708, 612, 282, 651, 249, 642, 672, 750, 315, 717, 378, 417, 240, 627, 360]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,time
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
t=time.time();h=m.convex_hull;print('time',time.time()-t,'verts',len(h.vertices),'faces',len(h.faces))
PY"
```
<details><summary>output</summary>

```
time 0.1322612762451172 verts 3352 faces 6700

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/hull_refine.py <<'PY'
import numpy as np,trimesh,cv2,h5py,json,time
from scipy.spatial.transform import Rotation as R
# init: f9 smooth, static replaced by better anchors
cand=np.load('/work/scratch/candidates.npz'); base=cand['base']; start=cand['f9'].copy(); F=len(start)
Tanchor=np.load('/work/scratch/static_anchor_27.npz')['cand'][0]
start[:96]=Tanchor
# final static use robust mean f9 pose (all pooled) at 818
start[797:]=start[818]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));Twc=np.linalg.inv(Tcw)
with h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
ker=np.ones((5,5),np.uint8); perd=np.empty_like(per)
for i in range(F): perd[i]=cv2.dilate(per[i].astype(np.uint8),ker)>0
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); H=mesh.convex_hull.vertices
Xh=np.concatenate([H,np.ones((len(H),1))],1)
# cam rotations about origin
fxs=K[0,0]; fys=K[1,1]
angs=[-10,-8,-6,-4,-2,0,2,4,6,8,10]
deltas=[]
for ax in angs:
 for ay in angs:
  deltas.append(R.from_euler('xy',[ax,ay],degrees=True).as_matrix())
deltas=np.array(deltas)
print('candidates',len(deltas))
def project_poly(T):
 M=Tcw@T
 Xc=(M[:3,:3]@H.T).T+M[:3,3]
 uv=Xc[:,:2]/Xc[:,2:3]
 uv=uv*K[[0,1],[0,1]][None]+K[:2,2]
 return uv,Xc[:,2]
def score_one(i,Tcand):
 uv,z=project_poly(Tcand)
 # integer bbox/roi
 xmin=max(0,int(uv[:,0].min())-5);xmax=min …[+2480 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/hull_refine.py", line 5, in <module>
    Tanchor=np.load('/work/scratch/static_anchor_27.npz')['cand'][0]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/static_anchor_27.npz'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/static_anchor_27/static_anchor_0/" /work/scratch/hull_refine.py; python /work/scratch/hull_refine.py'
```
<details><summary>output</summary>

```
candidates 121
0 0.0 0.0 -1 (0, 0) elapsed 0.05824685096740723
100 0.822 0.925 120 (-2.3778097175293205, -2.8047410918912306) elapsed 5.483132362365723
200 0.823 0.946 50 (-2.2890107316242236, 2.566389342707371) elapsed 12.427064418792725
300 0.866 0.947 109 (-2.546047282381906, -1.5893264468058064) elapsed 19.35525393486023
400 0.826 0.91 115 (-1.7722394567501176, 2.3201571525486244) elapsed 26.441121101379395
500 0.927 0.965 42 (-1.7627858927729108, 1.0976621797045851) elapsed 33.4994695186615
600 0.567 0.723 97 (-4.975864313996439, -1.4714209472686548) elapsed 40.037105083465576
700 0.498 0.642 2 (-2.874905665853941, 4.410493827160494) elapsed 46.52200889587402
800 0.895 0.962 12 (-0.5008837517516014, -1.994659508545098) elapsed 53.35395002365112
mean hull sample 0.0 0.7979727141447199  …[+51 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/start_refined.npz');T=z['T'];c=np.load('/work/scratch/candidates.npz')
for name,A in [('base',c['base']),('f9',c['f9']),('ref',T)]:
 d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi
 dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
 db=R.from_matrix(np.transpose(c['f9'][:,:3,:3],(0,2,1))@A[:,:3,:3]).magnitude()*180/np.pi
 print(name,'rot med/p90/max',np.median(d),np.percentile(d,90),d.max(),'tr',np.median(dt),np.percentile(dt,90),dt.max(),'difff9',np.median(db),np.percentile(db,90),db.max())
print('select best angle distribution',np.bincount(z['best'],minlength=121).max(),np.unique(z['best'],return_counts=True))
PY"
```
<details><summary>output</summary>

```
base rot med/p90/max 5.111361741972479 20.43434646390443 145.1903515331527 tr 1.3256261602599997 2.994324360894867 13.427998936921536 difff9 6.4420518230661274 20.39997617570873 97.84659456460555
f9 rot med/p90/max 3.272565261274924 9.035615024452385 36.71585606872173 tr 0.8233912742728837 1.6389481436686364 15.318287006776979 difff9 0.0 0.0 0.0
ref rot med/p90/max 6.421882493078341 18.40524614959029 159.1616236840534 tr 0.8925450063278545 1.7441277019502432 100.85711025126412 difff9 10.768442803572803 171.0484310794765 171.0484310794765
select best angle distribution 858 (array([  0, 120]), array([858,   1]))

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/candidates.npz');b=z['base'];f=z['f9'];a=np.load('/work/scratch/static_anchor_0.npz')['cand'][0]
for i in list(range(80,121,2))+list(range(785,820,2)):
 db=R.from_matrix(np.transpose(a[:3,:3])@b[i,:3,:3]).magnitude()*180/np.pi
 df=R.from_matrix(np.transpose(a[:3,:3])@f[i,:3,:3]).magnitude()*180/np.pi
 print(i,round(db,1),round(df,1), 'base-f9',round(R.from_matrix(np.transpose(b[i,:3,:3])@f[i,:3,:3]).magnitude()*180/np.pi,1))
PY"
```
<details><summary>output</summary>

```
80 169.2 171.0 base-f9 2.2
82 169.2 171.0 base-f9 2.2
84 169.2 171.0 base-f9 2.2
86 169.2 171.0 base-f9 2.2
88 169.2 171.0 base-f9 2.2
90 169.2 171.0 base-f9 2.2
92 169.2 171.0 base-f9 2.2
94 169.2 171.0 base-f9 2.2
96 169.2 169.2 base-f9 0.0
98 169.2 169.2 base-f9 0.0
100 169.2 169.2 base-f9 0.0
102 169.2 169.2 base-f9 0.0
104 169.2 169.2 base-f9 0.0
106 169.2 169.1 base-f9 0.1
108 169.2 169.1 base-f9 0.1
110 169.2 169.3 base-f9 0.4
112 170.9 170.2 base-f9 1.0
114 167.2 170.6 base-f9 3.8
116 172.0 170.9 base-f9 1.6
118 174.7 169.9 base-f9 4.9
120 168.4 168.0 base-f9 0.4
785 118.2 117.6 base-f9 0.6
787 118.2 117.3 base-f9 1.0
789 118.2 119.3 base-f9 1.3
791 118.2 118.8 base-f9 0.8
793 118.2 116.9 base-f9 3.0
795 118.2 113.2 base-f9 6.8
797 109.4 131.6 base-f9 22.7
799 102.2 131.6 base-f9 2 …[+286 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh']
ref=T[27,:3,:3]
for i in range(0,96,3):
 rel=ref.T@T[i,:3,:3]; rv=R.from_matrix(rel).as_rotvec(); ang=np.linalg.norm(rv)*180/np.pi
 # local euler z y x
 e=R.from_matrix(rel).as_euler('zyx',degrees=True)
 print(i,round(ang,1),e.round(1))
PY"
```
<details><summary>output</summary>

```
0 6.6 [ 3.3 -2.3  5.3]
3 5.0 [ 1.7 -2.5  4.1]
6 4.4 [ 0.8 -2.5  3.5]
9 5.8 [ 0.9 -4.2  4. ]
12 6.3 [-0.7 -3.6  5.1]
15 3.1 [ 0.5 -1.4  2.7]
18 4.6 [ 0.8 -2.   4. ]
21 3.3 [-0.7 -1.3  3. ]
24 3.3 [-2.1  0.9 -2.3]
27 0.0 [-0.  0.  0.]
30 3.1 [ 3.1 -0.2  0.2]
33 6.4 [-4.5 -2.4  3.8]
36 6.7 [-2.7 -3.1  5.2]
39 5.1 [-4.8 -1.5 -0. ]
42 5.1 [-4.8 -1.5 -0. ]
45 5.1 [-4.8 -1.5 -0. ]
48 5.1 [-4.8 -1.5 -0. ]
51 5.1 [-4.8 -1.5 -0. ]
54 5.1 [-4.8 -1.5 -0. ]
57 5.1 [-4.8 -1.5 -0. ]
60 5.1 [-4.8 -1.5 -0. ]
63 5.1 [-4.8 -1.5 -0. ]
66 5.1 [-4.8 -1.5 -0. ]
69 5.1 [-4.8 -1.5 -0. ]
72 5.1 [-4.8 -1.5 -0. ]
75 5.1 [-4.8 -1.5 -0. ]
78 5.1 [-4.8 -1.5 -0. ]
81 5.1 [-4.8 -1.5 -0. ]
84 5.1 [-4.8 -1.5 -0. ]
87 5.1 [-4.8 -1.5 -0. ]
90 5.1 [-4.8 -1.5 -0. ]
93 5.1 [-4.8 -1.5 -0. ]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/hull_refine.py';s=open(p).read()
s=s.replace(\"start[:96]=Tanchor\",\"start[:96]=Twc@Tanchor\")
s=s.replace(\"bestv=iou0;bestj=-1;bestshift=(0,0)\",\"bestv=iou0;bestj=-1;bestshift=(0,0);ious0[i]=iou0\")
s=s.replace(\"start[i]=Tcand;best[j]=j;shifts[i]=[dx,dy];ious[i]=bestv\",\"start[i]=Tcand;best[i]=bestj;shifts[i]=[dx,dy];ious[i]=bestv\")
open(p,'w').write(s)
PY
python /work/scratch/hull_refine.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/hull_refine.py", line 6, in <module>
    start[:96]=Twc@Tanchor
               ^^^
NameError: name 'Twc' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nl -ba /work/scratch/hull_refine.py | head -20'
```
<details><summary>output</summary>

```
     1	import numpy as np,trimesh,cv2,h5py,json,time
     2	from scipy.spatial.transform import Rotation as R
     3	# init: f9 smooth, static replaced by better anchors
     4	cand=np.load('/work/scratch/candidates.npz'); base=cand['base']; start=cand['f9'].copy(); F=len(start)
     5	Tanchor=np.load('/work/scratch/static_anchor_0.npz')['cand'][0]
     6	start[:96]=Twc@Tanchor
     7	# final static use robust mean f9 pose (all pooled) at 818
     8	start[797:]=start[818]
     9	cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));Twc=np.linalg.inv(Tcw)
    10	with h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
    11	ker=np.ones((5,5),np.uint8); perd=np.empty_like(per)
    12	for i in rang …[+409 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/hull_refine.py';s=open(p).read()
s=s.replace(\"start[:96]=Twc@Tanchor\\n# final static use robust mean f9 pose (all pooled) at 818\\nstart[797:]=start[818]\\ncam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));Twc=np.linalg.inv(Tcw)\",\"cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));Twc=np.linalg.inv(Tcw)\\nstart[:96]=Twc@Tanchor\\n# final static use robust mean f9 pose (all pooled) at 818\\nstart[797:]=start[818]\")
open(p,'w').write(s)
PY
python /work/scratch/hull_refine.py"
```
<details><summary>output</summary>

```
candidates 121
0 0.817 0.942 18 (-2.562468345972004, 0.20108043836993517) elapsed 0.08051443099975586
100 0.822 0.925 120 (-2.3778097175293205, -2.8047410918912306) elapsed 7.405438184738159
200 0.823 0.946 50 (-2.2890107316242236, 2.566389342707371) elapsed 14.650304555892944
300 0.866 0.947 109 (-2.546047282381906, -1.5893264468058064) elapsed 22.08606743812561
400 0.826 0.91 115 (-1.7722394567501176, 2.3201571525486244) elapsed 29.06779909133911
500 0.927 0.965 42 (-1.7627858927729108, 1.0976621797045851) elapsed 36.136287689208984
600 0.567 0.723 97 (-4.975864313996439, -1.4714209472686548) elapsed 42.54716897010803
700 0.498 0.642 2 (-2.874905665853941, 4.410493827160494) elapsed 48.750179290771484
800 0.895 0.962 12 (-0.5008837517516014, -1.994659508545098) elapsed 56.96177816390991
 …[+106 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/start_refined.npz');T=z['T'];c=np.load('/work/scratch/candidates.npz')
ang=np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10]);b=z['best'];ax=ang[b//11];ay=ang[b%11]
inds=np.arange(859)
print('best=0count',np.sum(b==60), 'nonzero',np.sum(b"'!=60))
for s,e in [(0,96),(96,797),(797,859)]:
 q=b[s:e];aa=ax[s:e];yy=ay[s:e]
 print((s,e),'"'zero',np.mean(q==60),'ax mean/std',aa.mean(),aa.std(),'ay',yy.mean(),yy.std(),'ranges',aa.min(),aa.max(),yy.min(),yy.max())
for name,A in [('base',c['base']),('f9',c['f9']),('ref',T)]:
 d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi
 dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
 print(name,'rot med/p90/max',np.median(d),np.percentile(d,90),d.max(),'tr',np.median(dt),np.percentile(dt,90),dt.max())
PY"
```
<details><summary>output</summary>

```
best=0count 4 nonzero 855
(0, 96) zero 0.0 ax mean/std -6.5625 6.087577001566387 ay -4.270833333333333 6.816767266983833 ranges -10 10 -10 10
(96, 797) zero 0.005706134094151213 ax mean/std 1.586305278174037 7.2639288727639775 ay 0.7617689015691869 7.3940654970231545 ranges -10 10 -10 10
(797, 859) zero 0.0 ax mean/std -2.032258064516129 8.044167877203241 ay -1.8387096774193548 8.074638808235 ranges -10 10 -10 10
base rot med/p90/max 5.111361741972479 20.43434646390443 145.1903515331527 tr 1.3256261602599997 2.994324360894867 13.427998936921536
f9 rot med/p90/max 3.272565261274924 9.035615024452385 36.71585606872173 tr 0.8233912742728837 1.6389481436686364 15.318287006776979
ref rot med/p90/max 6.768362844957423 19.354981994533855 63.51857908509211 tr 0.8925450063278545 1.7441277019502432  …[+19 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/smooth_ref.py <<'PY'
exec(open('/work/scratch/filter_candidates.py').read().split(\"print('base'\")[0])
z=np.load('/work/scratch/start_refined.npz'); ref=z['T'].astype(float)
f9=np.load('/work/scratch/candidates.npz')['f9'].astype(float)
first,last=96,796
Cs={'base':np.load('/work/scratch/candidates.npz')['base'].astype(float),'f9':f9,'ref':ref}
for rad,thr in [(3,6),(4,8),(5,10),(6,12)]:
 T2=filter_pose(ref,rad,thr,.04,False)
 # reapply static anchors exactly from ref (which are same throughout)
 T2[:first]=ref[:first];T2[last+1:]=ref[last+1:]
 Cs[f'refs{rad}']=T2
 print('refs',rad,smooth_metrics(T2))
np.savez('/work/scratch/candidates2.npz',**Cs)
PY
python /work/scratch/smooth_ref.py
sed 's#candidates.npz#candidates2.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics2.py
python /work/scratch/mask_metrics2.py"
```
<details><summary>output</summary>

```
contact 96 796
refs 3 {'rot_med': 4.49221569943123, 'rot_p90': 11.376509300913053, 'rot_mean': 5.558626802517425, 'rot_max': 31.020155584006496, 'trans_med': 0.008699943051456462, 'trans_p90': 0.016653356431740637, 'trans_max': 0.11392366107130972, 'accmed': 0.001763401697525845}
refs 4 {'rot_med': 4.469881481653152, 'rot_p90': 10.914060398224644, 'rot_mean': 5.424064737613041, 'rot_max': 29.586110782347156, 'trans_med': 0.008589172227045899, 'trans_p90': 0.016547962973732986, 'trans_max': 0.10634967881660735, 'accmed': 0.0015353379467792066}
refs 5 {'rot_med': 4.336111140091242, 'rot_p90': 10.542225627718196, 'rot_mean': 5.29953414686046, 'rot_max': 28.212088521132138, 'trans_med': 0.008615620519041724, 'trans_p90': 0.016570627713893796, 'trans_max': 0.10523577919011491, 'accmed': 0.00146 …[+2113 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/exact_search_test.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
from scipy.spatial.transform import Rotation as R
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda');G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ker=np.ones((5,5),np.uint8);perd=np.array([cv2.dilate(x.astype(np.uint8),ker)>0 for x in per])
cand=np.load('/work/scratch/candidates.npz');T=cand['f9']
angs=np.arange(-6,6.1,2);ds=[]
for ax in angs:
 for ay in angs:ds.append(R.from_euler('xy',[ax,ay],degrees=True).as_matrix())
ds=np.array(ds);ctx=dr.RasterizeCudaContext()
def render(Ts):
 poses=torch.tensor(np.stack([Tcw@x for x in Ts]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,re …[+1185 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/exact_search_test.py", line 33, in <module>
    print('\n',i,'f9 iou',round(iou0,3),'best',[(round(float(v[0]),3),round(float(v[1]),3),tuple(round(float(z),1) for z in v[2]),R.from_matrix(v[3]).as_euler('xy',degrees=True).round(1)) for v in vals[:8]])
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/exact_search_test.py", li …[+678 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/as_euler('xy',degrees=True)/as_euler('xyz',degrees=True)/\" /work/scratch/exact_search_test.py; python /work/scratch/exact_search_test.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

 402 f9 iou 0.794 best [(0.859, 0.812, (-2.8, 0.6), array([ 6., -6.,  0.])), (0.858, 0.811, (-2.9, 0.8), array([ 4., -6.,  0.])), (0.854, 0.809, (-2.9, 0.9), array([ 2., -6.,  0.])), (0.854, 0.809, (-3.0, 1.0), array([ 0., -6.,  0.])), (0.849, 0.804, (-3.0, 1.1), array([-2., -6.,  0.])), (0.846, 0.801, (-3.1, 1.3), array([-4., -6., -0.])), (0.846, 0.809, (-2.8, 0.6), array([ 6., -4.,  0.])), (0.845, 0.809, (-2.9, 0.7), array([ 4., -4.,  0.]))]

 408 f9 iou 0.604 best [(0.664, 0.617, (-1.9, -3.2), array([-6., -6.,  0.])), (0.663, 0.617, (-1.8, -3.2), array( …[+3880 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/centroid_correct.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda');G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ker=np.ones((5,5),np.uint8);perd=np.array([cv2.dilate(x.astype(np.uint8),ker)>0 for x in per])
from scipy.spatial.transform import Rotation as R
C=np.load('/work/scratch/candidates2.npz'); T=C['f9'].astype(float).copy();F=len(T)
ctx=dr.RasterizeCudaContext()
def render(Ts):
 poses=torch.tensor(np.stack([Tcw@x for x in Ts]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536));return rast[...,3].flip(1).cpu().numpy()>0
shifts=[]
for b0 in range(0,F,12):
 ids=np.arange(b0 …[+1495 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
raw shift median/p90/max [1.9355966  1.82519145] [4.20822251 5.73998171] [18.42702703 17.86559543]
sm shift max [13.93926746 17.39156892]
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'f9', 'ref', 'refs3', 'refs4', 'refs5', 'refs6', 'cent']
base IoU 0.6894899307535464 0.731187472813949 visIoU 0.7723512453781898 0.8095238095234744 CE 5.898781214838638 time …[+1624 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nl -ba /work/scratch/centroid_variants.py'
```
<details><summary>output</summary>

```
     1	import numpy as np,json
     2	from scipy.signal import medfilt,savgol_filter
     3	from scipy.ndimage import gaussian_filter1d
     4	C=np.load('/work/scratch/candidates2.npz');T=C['f9'].astype(float);F=len(T)
     5	z=np.load('/work/scratch/centroid_corrected.npz');s=z['shifts'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));Twc=np.linalg.inv(Tcw)
     6	vars={'raw':s,'med3':np.stack([medfilt(s[:,k],3) for k in range(2)],1),'med5':np.stack([medfilt(s[:,k],5) for k in range(2)],1),'med9':np.stack([medfilt(s[:,k],9) for k in range(2)],1),'g1':np.stack([gaussian_filter1d(s[:,k],1) for k in range(2)],1),'sg7':savgol_filter(s,7,2,axis=0),'med5g1':np.stack([gaussian_filter1d(medfilt(s[:,k],5),1) for k in range(2)],1) …[+724 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/centroid_variants.py <<'PY'
import numpy as np,json
from scipy.signal import medfilt,savgol_filter
from scipy.ndimage import gaussian_filter1d
C=np.load('/work/scratch/candidates2.npz');T=C['f9'].astype(float);F=len(T)
z=np.load('/work/scratch/centroid_corrected.npz');s=z['shifts'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));Twc=np.linalg.inv(Tcw)
vars={'raw':s,'med3':np.stack([medfilt(s[:,k],3) for k in range(2)],1),'med5':np.stack([medfilt(s[:,k],5) for k in range(2)],1),'med9':np.stack([medfilt(s[:,k],9) for k in range(2)],1),'g1':np.stack([gaussian_filter1d(s[:,k],1) for k in range(2)],1),'sg7':savgol_filter(s,7,2,axis=0),'med5g1':np.stack([gaussian_filter1d(medfilt(s[:,k],5),1) for k in range(2)],1)}
out={}
for name,ss in vars.items():
 Ts=T.copy()
 for i in range(96,797):
  zc=(Tcw@T[i])[2,3]; dc=np.array([ss[i,0]*zc/K[0,0],ss[i,1]*zc/K[1,1],0.]);Ts[i,:3,3]+=Tcw[:3,:3]@dc
 out['cent_'+name]=Ts
 c=np.load('/work/scratch/candidates3.npz');break
base=dict(np.load('/work/scratch/candidates2.npz'));base.update(out);np.savez('/work/scratch/candidates4.npz',**base)
# print trans step dynamic
from scipy.spatial.transform import Rotation as R
for n,A in out.items():
 dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
 print(n,'trans step med/p90/max dyn',np.median(dt[96:796]),np.percentile(dt[96:796],90),dt[96:796].max())
PY
python /work/scratch/centroid_variants.p …[+143 chars]
```
<details><summary>output</summary>

```
cent_raw trans step med/p90/max dyn 1.0040141102757436 1.8613670744169069 10.060263368690482
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'f9', 'ref', 'refs3', 'refs4', 'refs5', 'refs6', 'cent_raw']
base IoU 0.6894899307535464 0.731187472813949 visIoU 0.7723512453781898 0.8095238095234744 CE 5.898781214838638 time 6.974013328552246
f9 IoU 0.7029279973105397 0.7389597644747338 visIoU 0.7871204832441732 0.8157024793385059 CE 5.66862595858748 time 7.353713750839233
Traceback (most recent call last):
  File "/work/scratch/mask_metrics4.py", line 33, in <module>
    inter=(r&tgt).sum(); unio …[+242 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '13d' /work/scratch/centroid_variants.py; python /work/scratch/centroid_variants.py"
```
<details><summary>output</summary>

```
cent_raw trans step med/p90/max dyn 1.0040141102757436 1.8613670744169069 10.060263368690482
cent_med3 trans step med/p90/max dyn 0.9999895816782606 1.8261022425798465 7.6435692846488195
cent_med5 trans step med/p90/max dyn 0.9989555662377989 1.8193465308715993 5.313418892538946
cent_med9 trans step med/p90/max dyn 0.9898877279968603 1.8117274736785516 4.762418468129196
cent_g1 trans step med/p90/max dyn 1.0022669837337086 1.8286988175672552 3.401622039369588
cent_sg7 trans step med/p90/max dyn 1.003913768153981 1.819719057616013 3.141993226394721
cent_med5g1 trans step med/p90/max dyn 0.9952945544562497 1.8000644380120945 3.1435679398192353

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/candidates4.npz'))
for k in list(z):
 if not (k.startswith('cent_') or k=='f9'): del z[k]
np.savez('/work/scratch/cent_metrics_in.npz',**z)
PY
sed 's#candidates.npz#cent_metrics_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_cent.py
python /work/scratch/mask_metrics_cent.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['f9', 'cent_raw', 'cent_med3', 'cent_med5', 'cent_med9', 'cent_g1', 'cent_sg7', 'cent_med5g1']
f9 IoU 0.7029279973105397 0.7389597644747338 visIoU 0.7871204832441732 0.8157024793385059 CE 5.66862595858748 time 7.466169118881226
cent_raw IoU 0.773222791888159 0.8177121771214695 visIoU 0.8540902742714522 0.8963051251486308 CE 3.1988278329869444 time 7.345335006713867
cent_med3 IoU 0.7730461607588932 0.8187414500681195 visIoU 0.8537556697699262 0.8944932901430381 CE 3.2315678101131287 time 7.417585134506226
cent_med5 IoU 0.773505613448217 0.817712177121 …[+1342 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/centroid_iter.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
from scipy.ndimage import gaussian_filter1d
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda');G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ker=np.ones((5,5),np.uint8);perd=np.array([cv2.dilate(x.astype(np.uint8),ker)>0 for x in per])
T0=np.load('/work/scratch/candidates2.npz')['f9'].astype(float).copy();F=len(T0);ctx=dr.RasterizeCudaContext()
def render(Ts):
 poses=torch.tensor(np.stack([Tcw@x for x in Ts]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536));return rast[...,3].flip(1).cpu().numpy()>0
def one(T,smooth):
 allsh=[]
 for b0 in range(0,F,12):
  ids=np. …[+1295 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
g1_1 0 median shift [1.94984844 1.77241212] p90 [4.31260611 5.52722847]
g1_2 0 median shift [1.94984844 1.77241212] p90 [4.31260611 5.52722847]
g1_2 1 median shift [0.50032661 0.61565419] p90 [1.89232078 2.16101686]
g1_3 0 median shift [1.94984844 1.77241212] p90 [4.31260611 5.52722847]
g1_3 1 median shift [0.50032661 0.61565419] p90 [1.89232078 2.16101686]
g1_3 2 median shift [0.20005761 0.19976067] p90 [1.83425765 1.91587831]
raw_2 0 median shift [1.9355966  1.82519145] p90 [4.20822251 5.73998171]
raw_2 1 median shift [0.53114884 0.63107105] p90 [1.946758 …[+3356 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/centroid_iter2.py <<'PY'
exec(open('/work/scratch/centroid_iter.py').read().split('out={}')[0])
T=T0.copy(); out={}
for it in range(8):
 T,s=one(T,True);print('it',it+1,'median',np.median(np.abs(s),0),'p90',np.percentile(np.abs(s),90,axis=0))
 if it+1 in (3,5,8): out[f'cent_g1_{it+1}']=T.copy()
np.savez('/work/scratch/cent_iter2_out.npz',**out)
PY
python /work/scratch/centroid_iter2.py
python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/candidates5.npz'));z.update(dict(np.load('/work/scratch/cent_iter2_out.npz')))
z={k:v for k,v in z.items() if k in ['cent_g1_3','cent_g1_5','cent_g1_8','cent_raw_3']};np.savez('/work/scratch/cent_iter2_in.npz',**z)
PY
sed 's#candidates.npz#cent_iter2_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_iter2.py
python /work/scratch/mask_metrics_iter2.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
it 1 median [1.94984844 1.77241212] p90 [4.31260611 5.52722847]
it 2 median [0.50032661 0.61565419] p90 [1.89232078 2.16101686]
it 3 median [0.20005761 0.19976067] p90 [1.83425765 1.91587831]
it 4 median [0.09547451 0.09952472] p90 [1.77940491 1.78265368]
it 5 median [0.05234448 0.05677187] p90 [1.83132757 1.78265368]
it 6 median [0.03226839 0.03660847] p90 [1.82189421 1.78265368]
it 7 median [0.02322274 0.0235544 ] p90 [1.82189421 1.78265368]
it 8 median [0.01796933 0.02123559] p90 [1.82189421 1.78265368]
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driv …[+1174 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'head -15 /work/scratch/centroid_iter.py; nl -ba /work/scratch/static_centroid_correct.py | head'
```
<details><summary>output</summary>

```
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
from scipy.ndimage import gaussian_filter1d
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda');G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device …[+1354 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_centroid_correct.py <<'PY'
exec(open('/work/scratch/centroid_iter.py').read().split('T0=')[0])
T=np.load('/work/scratch/cent_iter2_out.npz')['cent_g1_5'].astype(float).copy()
# static reference poses: initial from anchor, final from pooled f9 all identical 797+
Tanchor=np.load('/work/scratch/static_anchor_0.npz')['cand'][0]
T[:96]=Twc@Tanchor
T[797:]=T[818]
ctx=dr.RasterizeCudaContext()
def render(Ts):
 poses=torch.tensor(np.stack([Tcw@x for x in Ts]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536));return rast[...,3].flip(1).cpu().numpy()>0
for it in range(4):
 for s,e in [(0,96),(797,859)]:
  cur=T[s:e].copy();rms=render(cur);sh=[]
  for i,r in zip(range(s,e),rms):
   rv=r&~perd[i];tv=obj[i]&~perd[i];ys,xs=np.where(rv);yt,xt=np.where(tv)
   if len(xs)>50 and len(xt)>50:sh.append((xt.mean()-xs.mean(),yt.mean()-ys.mean()))
  sh=np.array(sh); med=np.median(sh,0); print('it',it,'seg',s,e,'median shift',med,'mean',sh.mean(0),'p90abs',np.percentile(np.abs(sh-med),90,axis=0))
  for i in range(s,e):
   zc=(Tcw@T[i])[2,3];dc=np.array([med[0]*zc/K[0,0],med[1]*zc/K[1,1],0.]);T[i,:3,3]+=Tcw[:3,:3]@dc
np.savez('/work/scratch/static_corrected.npz',T=T)
PY
python /work/scratch/static_centroid_correct.py
python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/candidates5.npz'));z.update(dict(np.load('/work/scratch/cent_iter2 …[+362 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/static_centroid_correct.py", line 5, in <module>
    T[:96]=Twc@Tanchor
           ^^^
NameError: name 'Twc' is not defined
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/static_corrected.npz'
Warp 1.17.0  …[+794 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '2i Twc=np.linalg.inv(Tcw)' /work/scratch/static_centroid_correct.py; python /work/scratch/static_centroid_correct.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
it 0 seg 0 96 median shift [-1.77175909 -0.69285411] mean [-1.72947616 -0.64428876] p90abs [0.69183884 0.38961014]
it 0 seg 797 859 median shift [-0.5906278  -1.48506094] mean [-0.54775873 -1.52031533] p90abs [0.35161845 0.36926064]
it 1 seg 0 96 median shift [-0.74211281 -0.4122183 ] mean [-0.72668655 -0.39037369] p90abs [0.37249068 0.32315109]
it 1 seg 797 859 median shift [-0.28974879 -0.84178758] mean [-0.25603547 -0.79121556] p90abs [0.36426251 0.40005421]
it 2 seg 0 96 median shift [-0.30709944 -0.33791961] mean [-0.30688999 -0.2882823 ] p90abs [0.313 …[+369 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/candidates5.npz'));z.update(dict(np.load('/work/scratch/cent_iter2_out.npz')));z['final']=np.load('/work/scratch/static_corrected.npz')['T'];z={k:v for k,v in z.items() if k in ['base','f9','cent_g1_5','final']};np.savez('/work/scratch/final_candidates_in.npz',**z)
PY
python /work/scratch/mask_metrics_final.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'f9', 'cent_g1_5', 'final']
base IoU 0.6894899307535464 0.731187472813949 visIoU 0.7723512453781898 0.8095238095234744 CE 5.898781214838638 time 6.432915210723877
f9 IoU 0.7029279973105397 0.7389597644747338 visIoU 0.7871204832441732 0.8157024793385059 CE 5.66862595858748 time 5.394958257675171
cent_g1_5 IoU 0.7890433433411924 0.8549679487176062 visIoU 0.8689345110878133 0.9126348764354637 CE 2.772269884413289 time 5.543914794921875
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.1693946357290 …[+368 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/depth_metrics_test.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda');ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ker=np.ones((5,5),np.uint8);perd=np.array([cv2.dilate(x.astype(np.uint8),ker)>0 for x in per])
z=np.load('/work/scratch/final_candidates_in.npz');Ts={'base':z['base'],'final':z['final']}
for kname in ['K','K_pipeline']:
 K=np.array(cam[kname]);P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda')
 print('INTR',kname)
 for i in [0,27,90,150,300,450,600,720,840,858]:
  with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
   dep=cv2.imdecode(f['aligned']['front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
  for name,T in Ts.items():
   pose=torch.tensor((Tcw@T[i]).astype(np.float32 …[+864 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
INTR K
Traceback (most recent call last):
  File "/work/scratch/depth_metrics_test.py", line 20, in <module>
    xyz,_=dr.interpolate(vc[None],rast,Fc);zmap=xyz[0,:, :,2].flip(0).cpu().numpy();rm=rast[0,:,:,3].flip(0).cpu().numpy()>0
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 5 …[+430 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/vc=torch.tensor(((Tcw@T\\[i\\])\\[:3,:3\\]@mesh.vertices.T).T+(Tcw@T\\[i\\])\\[:3,3\\],dtype=torch.float32,device='\"'\"'cuda'\"'\"')/vc=torch.tensor(((Tcw@T[i])[:3,:3]@mesh.vertices.T).T+(Tcw@T[i])[:3,3],dtype=torch.float32,device='\"'\"'cuda'\"'\"').contiguous()/' /work/scratch/depth_metrics_test.py; sed -i \"s/xyz,_=dr.interpolate(vc\\[None\\],rast,Fc)/xyz,_=dr.interpolate(vc[None].contiguous(),rast.contiguous(),Fc.contiguous())/\" /work/scratch/depth_metrics_test.py; python /work/scratch/depth_metrics_test.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
INTR K
0 base n 2604 mederr 0.07 bias -0.07 p90 0.226
0 final n 2486 mederr 0.072 bias -0.072 p90 0.098
27 base n 2428 mederr 0.01 bias 0.007 p90 0.026
27 final n 2373 mederr 0.062 bias 0.062 p90 0.071
90 base n 2203 mederr 0.098 bias 0.098 p90 0.123
90 final n 2358 mederr 0.127 bias 0.126 p90 0.146
150 base n 2031 mederr 0.017 bias -0.009 p90 2.159
150 final n 2078 mederr 0.021 bias -0.003 p90 0.177
300 base n 1915 mederr 0.173 bias 0.172 p90 0.198
300 final n 2040 mederr 0.162 bias 0.162 p90 0.181
450 base n 1514 mederr 0.188 bias 0.188 p90 0.208
450 fina …[+1458 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s/T=cand\\['f9'\\]/T=np.load('\\/work\\/scratch\\/final_candidates_in.npz')['final']/\" /work/scratch/exact_search_test.py | sed \"s/np.arange(-6,6.1,2)/np.arange(-8,8.1,2)/\" > /work/scratch/exact_search_test2.py; python /work/scratch/exact_search_test2.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

 402 f9 iou 0.818 best [(0.898, 0.892, (0.1, -0.8), array([ 8., -8.,  0.])), (0.893, 0.888, (0.1, -0.7), array([ 6., -8.,  0.])), (0.883, 0.883, (0.0, -0.5), array([ 4., -8.,  0.])), (0.881, 0.881, (-0.0, -0.3), array([ 2., -8.,  0.])), (0.876, 0.87, (0.1, -0.7), array([ 8., -6., -0.])), (0.873, 0.867, (0.1, -0.6), array([ 6., -6.,  0.])), (0.873, 0.873, (-0.1, -0.2), array([ 0., -8.,  0.])), (0.87, 0.87, (-0.1, -0.1), array([-2., -8.,  0.]))]

 408 f9 iou 0.62 best [(0.639, 0.639, (-0.1, -0.1), array([-8., -8., -0.])), (0.638, 0.638, (-0.0, -0.1), array([ …[+3831 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/exact_viterbi.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2,time
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
from scipy.spatial.transform import Rotation as R
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda');G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ker=np.ones((5,5),np.uint8);perd=np.array([cv2.dilate(x.astype(np.uint8),ker)>0 for x in per])
T0=np.load('/work/scratch/final_candidates_in.npz')['final'].astype(np.float32);F=len(T0)
angles=np.arange(-10,10.1,2.0); ds=[]
for ax in angles:
 for ay in angles:ds.append(R.from_euler('xy',[ax,ay],degrees=True).as_matrix())
ds=np.array(ds,dtype=np.float32);nA=len(ds);ctx=dr.RasterizeCudaContext()
scores=np.full((F,nA),-1,dtype=np.float32); shifts=np.zeros((F,nA,2),dtype=np.float32)
start,end=96,797; t0=time.time()
for i in range( …[+3027 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/exact_viterbi.py", line 25, in <module>
    rmc=rm[:,y0:y1,x0:x1]; pvc=torch.tensor(perd[i,y0:y1,x0:x1],device='cuda');tvc=torch.tensor(obj[i,y0:y1,x0:x1]&~perd[i],device='cuda')
                                                                                                ~~~~~~~~~~~~~~~~~~^~~~~~~~~
ValueError: operands could not be broadcast together with shapes (180,120) (1152,1536) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/exact_viterbi.py';s=open(p).read();s=s.replace(\"obj[i,y0:y1,x0:x1]&~perd[i]\",\"obj[i,y0:y1,x0:x1]&~perd[i,y0:y1,x0:x1]\");open(p,'w').write(s)
PY
python /work/scratch/exact_viterbi.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
100 time 2.1831257343292236 best 10.0 10.0 0.9488012790679932 base 0.9054000973701477
150 time 5.099937915802002 best 0.0 0.0 0.9291744828224182 base 0.9291744828224182
200 time 8.139437437057495 best 2.0 -8.0 0.9528558254241943 base 0.9436175227165222
250 time 11.215813636779785 best -10.0 8.0 0.7612673044204712 base 0.7015053629875183
300 time 14.325818061828613 best -6.0 10.0 0.963350772857666 base 0.955986738204956
350 time 17.48532724380493 best 8.0 2.0 0.9249875545501709 base 0.9127255082130432
400 time 20.572373151779175 best 10.0 0.0 0.9023383855819 …[+727 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/final_candidates_in.npz');v=np.load('/work/scratch/viterbi_rot.npz')['T'];np.savez('/work/scratch/vit_in.npz',base=z['base'],final=z['final'],vit=v)
PY
sed 's#candidates.npz#vit_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_vit.py
python /work/scratch/mask_metrics_vit.py
python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for name in ['final','vit']:
 z=np.load('/work/scratch/vit_in.npz');A=z[name]
 d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi
 dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
 print(name,'rot',np.median(d[96:796]),np.percentile(d[96:796],90),d[96:796].max(),'trans',np.median(dt[96:796]),np.percentile(dt[96:796],90),dt[96:796].max())
PY"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'final', 'vit']
base IoU 0.6894899307535464 0.731187472813949 visIoU 0.7723512453781898 0.8095238095234744 CE 5.898781214838638 time 7.152855634689331
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 7.369892597198486
vit IoU 0.8311272190483432 0.8921404682270517 visIoU 0.9078251390375011 0.9442700156982166 CE 2.1686783005511603 time 6.615321636199951
phase summary means
base iouall 0.7831 0.6649 0.8213 iouv 0.8494 0.7518 0.884 ce 3.961 6.284 4.557
final iouall 0.9259 0.785 …[+364 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/vit_centroid.py <<'PY'
exec(open('/work/scratch/centroid_iter.py').read().split('T0=')[0])
Twc=np.linalg.inv(Tcw)
T0=np.load('/work/scratch/viterbi_rot.npz')['T'].astype(float).copy()
F=len(T0);ctx=dr.RasterizeCudaContext()
def render(Ts):
 poses=torch.tensor(np.stack([Tcw@x for x in Ts]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536));return rast[...,3].flip(1).cpu().numpy()>0
def one(T,smooth):
 allsh=[]
 for b0 in range(0,F,12):
  ids=np.arange(b0,min(F,b0+12));rms=render(T[ids])
  for i,r in zip(ids,rms):
   rv=r&~perd[i];tv=obj[i]&~perd[i];ys,xs=np.where(rv);yt,xt=np.where(tv)
   allsh.append((xt.mean()-xs.mean() if len(xs) and len(yt) else 0,yt.mean()-ys.mean() if len(ys) and len(yt) else 0))
 s=np.array(allsh)
 if smooth:s=np.stack([gaussian_filter1d(s[:,k],1) for k in range(2)],1)
 Tn=T.copy()
 for i in range(96,797):
  zc=(Tcw@T[i])[2,3];dc=np.array([s[i,0]*zc/K[0,0],s[i,1]*zc/K[1,1],0.]);Tn[i,:3,3]+=Tcw[:3,:3]@dc
 return Tn,s
T=T0.copy();out={}
for it in range(5):
 T,s=one(T,True);print('it',it+1,'median',np.median(np.abs(s),0),'p90',np.percentile(np.abs(s),90,axis=0))
 if it+1 in (3,5):out[f'vit_cent_g1_{it+1}']=T.copy()
# static mean centroid correction
for it in range(3):
 for s0,e0 in [(0,96),(797,859)]:
  cur=T[s0:e0].copy();rms=render(cur);sh=[]
  for i,r in zip(range(s0,e0),rms):
   rv=r&~perd[i];tv=obj[i]&~p …[+808 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
it 1 median [0.10791809 0.15258112] p90 [0.39602965 0.60009726]
it 2 median [0.04882742 0.06192295] p90 [0.22915344 0.30780812]
it 3 median [0.027147   0.03161046] p90 [0.19167473 0.28942148]
it 4 median [0.01883961 0.02346551] p90 [0.18410168 0.26536964]
it 5 median [0.01491734 0.01807107] p90 [0.17467529 0.25102405]
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names  …[+1914 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/vit_overlay.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda');G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda');ctx=dr.RasterizeCudaContext()
z=np.load('/work/scratch/vit2_in.npz');A=z['final'];B=z['vit'];m=np.load('/work/scratch/mask_metrics_vit2.npz');gain=m['vit'][:,1]-m['final'][:,1];inds=m['inds']
cand=inds[(inds>=96)&(inds<797)];g=gain[(inds>=96)&(inds<797)];order=np.argsort(-g);sel=[]
for j in order:
 if len(sel)>=24:break
 if all(abs(int(cand[j])-x)>4 for x in sel):sel.append(int(cand[j]))
print(sel,[(i,round(float(gain[np.where(inds==i)[0][0]]),2)) for i in sel])
cap=cv2.VideoCapture('/work/inputs/video.mp4');thumbs=[]
def render(ps):
 poses=torch.tensor(np.stack([Tcw@x for x in ps]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[N …[+1008 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/vit_overlay.py", line 7, in <module>
    z=np.load('/work/scratch/vit2_in.npz');A=z['final'];B=z['vit'];m=np.load('/work/scratch/mask_metrics_vit2.npz');gain=m['vit'][:,1]-m['final'][:,1];inds=m['inds']
                                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^ …[+111 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/mask_metrics_vit2.npz/mask_metrics.npz/' /work/scratch/vit_overlay.py; python /work/scratch/vit_overlay.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
[558, 654, 744, 708, 669, 687, 648, 639, 603, 564, 402, 714, 246, 633, 675, 702, 681, 282, 594, 627, 252, 693, 543, 99] [(558, 0.42), (654, 0.19), (744, 0.17), (708, 0.15), (669, 0.14), (687, 0.13), (648, 0.12), (639, 0.11), (603, 0.1), (564, 0.1), (402, 0.09), (714, 0.09), (246, 0.09), (633, 0.08), (675, 0.08), (702, 0.08), (681, 0.07), (282, 0.06), (594, 0.06), (627, 0.06), (252, 0.05), (693, 0.05), (543, 0.05), (99, 0.05)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/viterbi_rot.npz');a=z['angles'];p=z['path'];ax=a[p//len(a)];ay=a[p%len(a)]
print('border frac',np.mean((np.abs(ax)==10)|(np.abs(ay)==10)))
print('ax hist',dict(zip(*np.unique(ax,return_counts=True))))
print('ay hist',dict(zip(*np.unique(ay,return_counts=True))))
print('first 30',list(zip(ax[:30],ay[:30])))
PY"
```
<details><summary>output</summary>

```
border frac 0.24536376604850213
ax hist {-10.0: 27, -8.0: 28, -6.0: 19, -4.0: 49, -2.0: 94, 0.0: 141, 2.0: 140, 4.0: 80, 6.0: 46, 8.0: 31, 10.0: 46}
ay hist {-10.0: 37, -8.0: 13, -6.0: 33, -4.0: 49, -2.0: 80, 0.0: 152, 2.0: 127, 4.0: 73, 6.0: 46, 8.0: 16, 10.0: 75}
first 30 [(4.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (6.0, 10.0), (8.0, 10.0), (8.0, 10.0), (8.0, 10.0), (8.0, 10.0), (8.0, 10.0), (8.0, 10.0), (8.0, 10.0), (8.0, 10.0), (8.0, 10.0), (6.0, 8.0), (2.0, 6.0), (0.0, 4.0), (-2.0, 4.0), (-4.0, 2.0), (-4.0, 0.0), (-2.0, 0.0), (0.0, 0.0), (-2.0, 0.0), (-2.0, 0.0)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed "s/np.arange(-10,10.1,2.0)/np.arange(-24,24.1,4.0)/" /work/scratch/exact_viterbi.py | sed "s#viterbi_rot.npz#viterbi_rot_wide.npz#" > /work/scratch/exact_viterbi_wide.py; python /work/scratch/exact_viterbi_wide.py'
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
100 time 0.5991685390472412 best 8.0 16.0 0.9593595862388611 base 0.9054000973701477
150 time 4.8258795738220215 best -24.0 4.0 0.932578980922699 base 0.9291744828224182
200 time 8.999537229537964 best 0.0 -16.0 0.9514298439025879 base 0.9436175227165222
250 time 12.954176664352417 best 24.0 24.0 0.8466172218322754 base 0.7015053629875183
300 time 17.002569437026978 best -4.0 16.0 0.9665871262550354 base 0.955986738204956
350 time 21.320854663848877 best 16.0 4.0 0.9270516633987427 base 0.9127255082130432
400 time 25.506075620651245 best 20.0 0.0 0.90673577 …[+736 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/viterbi_rot_wide.npz');a=z['angles'];p=z['path'];ax=a[p//len(a)];ay=a[p%len(a)];print('border',np.mean((np.abs(ax)==24)|(np.abs(ay)==24)));print(dict(zip(*np.unique(ax,return_counts=True))));print(dict(zip(*np.unique(ay,return_counts=True))))
q=np.load('/work/scratch/vit2_in.npz');np.savez('/work/scratch/wide_in.npz',final=q['final'],vit=q['vit'],wide=z['T'])
PY
sed 's#candidates.npz#wide_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_wide.py
python /work/scratch/mask_metrics_wide.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'vit', 'wide']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 7.876674652099609
vit IoU 0.8311272190483432 0.8921404682270517 visIoU 0.9078251390375011 0.9442700156982166 CE 2.1686783005511603 time 7.984593152999878
wide IoU 0.843049648680181 0.8905213270137959 visIoU 0.9194707296193315 0.944995491433298 CE 2.1659512901275773 time 8.288679361343384
phase summary means
final iouall 0.9259 0.7858 0.943 iouv 0.9655 0.8686 0.9751 ce 0.613 2.512 0.722
vit iouall 0.9259 0.8081 …[+133 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed "s/np.arange(-10,10.1,2.0)/np.arange(-36,36.1,6.0)/" /work/scratch/exact_viterbi.py | sed "s#viterbi_rot.npz#viterbi_rot_xwide.npz#" > /work/scratch/exact_viterbi_xwide.py; python /work/scratch/exact_viterbi_xwide.py'
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
100 time 0.5390400886535645 best 6.0 18.0 0.9538210034370422 base 0.9054000973701477
150 time 4.177769660949707 best -30.0 6.0 0.9469112157821655 base 0.9291744828224182
200 time 7.80300235748291 best 0.0 -18.0 0.9501133561134338 base 0.9436175227165222
250 time 11.410861730575562 best 36.0 30.0 0.9318944811820984 base 0.7015053629875183
300 time 15.030723094940186 best 6.0 18.0 0.9656816124916077 base 0.955986738204956
350 time 18.68355703353882 best 6.0 0.0 0.9189723134040833 base 0.9127255082130432
400 time 22.653056144714355 best 18.0 0.0 0.906030833721 …[+731 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
q=np.load('/work/scratch/vit2_in.npz');z=np.load('/work/scratch/viterbi_rot_xwide.npz');np.savez('/work/scratch/xwide_in.npz',final=q['final'],vit=q['vit'],wide=np.load('/work/scratch/viterbi_rot_wide.npz')['T'],xwide=z['T'])
a=z['angles'];p=z['path'];ax=a[p//len(a)];ay=a[p%len(a)];print('border',np.mean((np.abs(ax)==36)|(np.abs(ay)==36)));print(dict(zip(*np.unique(ax,return_counts=True))));print(dict(zip(*np.unique(ay,return_counts=True))))
PY
sed 's#candidates.npz#xwide_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_xwide.py
python /work/scratch/mask_metrics_xwide.py"
```
<details><summary>output</summary>

```
border 0.024251069900142655
{-36.0: 2, -30.0: 3, -24.0: 3, -18.0: 13, -12.0: 20, -6.0: 68, 0.0: 407, 6.0: 119, 12.0: 48, 18.0: 15, 24.0: 3}
{-36.0: 15, -30.0: 11, -24.0: 8, -18.0: 6, -12.0: 6, -6.0: 99, 0.0: 357, 6.0: 127, 12.0: 40, 18.0: 16, 24.0: 12, 30.0: 4}
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'vit', 'wide', 'xwide']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 6.411454677581787
vit IoU 0.8311272190483432 0.8921404682270517 visIoU 0.9078251390375011 0.9442700156982166 CE 2.1686783005511603 time 5.3464 …[+620 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/xwide_in.npz')
for name in ['final','vit','wide','xwide']:
 A=z[name];d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi;dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
 print(name,'rot',np.median(d[96:796]),np.percentile(d[96:796],90),d[96:796].max(),'trans',np.median(dt[96:796]),np.percentile(dt[96:796],90),dt[96:796].max())
PY"
```
<details><summary>output</summary>

```
final rot 4.110657704547486 9.604422481060423 36.71585606872173 trans 1.0169656287474336 1.8132402067600017 5.324062954135539
vit rot 4.299233469252376 9.57928382641878 43.015994800072484 trans 1.016962 1.8132427334785468 5.324063
wide rot 4.575255111999594 9.943861381594107 49.76847227696837 trans 1.016962 1.8132427334785468 5.324063
xwide rot 4.630549858809124 9.730380662549644 48.24547292944447 trans 1.016962 1.8132427334785468 5.324063

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_rot_search.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2,time
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import projection_matrix_from_intrinsics,glcam_in_cvcam
from scipy.spatial.transform import Rotation as R
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Fc=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');homo=torch.cat([V,torch.ones((len(V),1),device='cuda')],1)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda');G=torch.tensor(glcam_in_cvcam,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ker=np.ones((5,5),np.uint8);perd=np.array([cv2.dilate(x.astype(np.uint8),ker)>0 for x in per])
T=np.load('/work/scratch/xwide_in.npz')['xwide'].astype(np.float32);ctx=dr.RasterizeCudaContext()
angs=np.arange(-5,5.1,1.0);ds=[]
for ax in angs:
 for ay in angs:ds.append(R.from_euler('xy',[ax,ay],degrees=True).as_matrix())
ds=np.array(ds,dtype=np.float32)
def render(ps):
 poses=torch.tensor(np.stack([Tcw@x for x in ps]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0]; …[+1259 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
SEG 0 96 n 12 base 0.9655722181002299 best
-5.0 -5.0 0.97641
-5.0 -4.0 0.97633
-5.0 -3.0 0.97546
-5.0 -2.0 0.97543
-5.0 -1.0 0.97517
-4.0 -4.0 0.9742
-4.0 -5.0 0.97418
-5.0 0.0 0.97406
-5.0 1.0 0.97391
-4.0 -3.0 0.97385
SEG 797 859 n 13 base 0.9751301682912387 best
-1.0 -4.0 0.9775
-2.0 -5.0 0.97735
-3.0 5.0 0.97718
-2.0 5.0 0.97696
-4.0 5.0 0.97695
-1.0 -2.0 0.97695
-4.0 3.0 0.97692
-5.0 1.0 0.97692
-3.0 -5.0 0.97691
-3.0 0.0 0.97685
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA  …[+830 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed "s/np.arange(-5,5.1,1.0)/np.arange(-12,12.1,2.0)/" /work/scratch/static_rot_search.py | sed "s#static_rot_ref.npz#static_rot_ref2.npz#" > /work/scratch/static_rot_search2.py; python /work/scratch/static_rot_search2.py'
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
SEG 0 96 n 12 base 0.9655722181002299 best
-8.0 -4.0 0.97857
-6.0 -12.0 0.97835
-6.0 -10.0 0.97815
-8.0 -6.0 0.97806
-8.0 -2.0 0.97782
8.0 12.0 0.9778
-8.0 0.0 0.97777
-6.0 -8.0 0.97768
-6.0 -4.0 0.97728
-8.0 -8.0 0.97718
SEG 797 859 n 13 base 0.9751301682912387 best
-2.0 -6.0 0.97765
-2.0 -8.0 0.97754
-2.0 6.0 0.97719
-4.0 -12.0 0.97707
-4.0 -10.0 0.97707
-4.0 -8.0 0.97688
-4.0 6.0 0.97684
-4.0 -4.0 0.97682
-4.0 -2.0 0.97682
-4.0 -6.0 0.97677

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/xwide_in.npz');q=np.load('/work/scratch/static_rot_ref2.npz')['T'];np.savez('/work/scratch/staticrot2_in.npz',final=z['final'],xwide=z['xwide'],staticrot2=q)
PY
sed 's#candidates.npz#staticrot2_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_staticrot2.py
python /work/scratch/mask_metrics_staticrot2.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'xwide', 'staticrot2']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 6.8922975063323975
xwide IoU 0.8452313801980819 0.887987640015107 visIoU 0.9224071493996997 0.9446401610463289 CE 2.172403682329639 time 6.365023374557495
staticrot2 IoU 0.8470544608805054 0.887987640015107 visIoU 0.9238510678349889 0.9446401610463289 CE 2.182360127065739 time 5.59197735786438
phase summary means
final iouall 0.9259 0.7858 0.943 iouv 0.9655 0.8686 0.9751 ce 0.613 2.512 0.722
xwide ioua …[+155 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_shift_final.py <<'PY'
exec(open('/work/scratch/centroid_iter.py').read().split('T0=')[0])
Twc=np.linalg.inv(Tcw);T=np.load('/work/scratch/static_rot_ref2.npz')['T'].astype(float);ctx=dr.RasterizeCudaContext()
def render(Ts):
 poses=torch.tensor(np.stack([Tcw@x for x in Ts]),dtype=torch.float32,device='cuda');mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536));return rast[...,3].flip(1).cpu().numpy()>0
for it in range(2):
 for s,e in [(0,96),(797,859)]:
  rms=render(T[s:e]);sh=[]
  for i,r in zip(range(s,e),rms):
   rv=r&~perd[i];tv=obj[i]&~perd[i];ys,xs=np.where(rv);yt,xt=np.where(tv)
   if len(xs)>50 and len(xt)>50:sh.append((xt.mean()-xs.mean(),yt.mean()-ys.mean()))
  med=np.median(np.array(sh),0);print(it,s,med)
  for i in range(s,e):
   zc=(Tcw@T[i])[2,3];dc=np.array([med[0]*zc/K[0,0],med[1]*zc/K[1,1],0.]);T[i,:3,3]+=Tcw[:3,:3]@dc
np.savez('/work/scratch/staticrot2shift.npz',T=T)
PY
python /work/scratch/static_shift_final.py
python - <<'PY'
import numpy as np
z=np.load('/work/scratch/staticrot2_in.npz');z['staticrot2shift']=np.load('/work/scratch/staticrot2shift.npz')['T'];np.savez('/work/scratch/staticshift_in.npz',**z)
PY
sed 's#candidates.npz#staticshift_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_staticshift.py
python /work/scratch/mask_metrics_staticshift.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
0 0 [0.09552329 0.04537173]
0 797 [-0.03385468 -0.09181998]
1 0 [0.03146803 0.09384965]
1 797 [-0.01540339 -0.0397867 ]
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
TypeError: 'NpzFile' object does not support item assignment
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/mask_metrics_st …[+488 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/staticrot2_in.npz'));z['staticrot2shift']=np.load('/work/scratch/staticrot2shift.npz')['T'];np.savez('/work/scratch/staticshift_in.npz',**z)
PY
python /work/scratch/mask_metrics_staticshift.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'xwide', 'staticrot2', 'staticrot2shift']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 7.558822154998779
xwide IoU 0.8452313801980819 0.887987640015107 visIoU 0.9224071493996997 0.9446401610463289 CE 2.172403682329639 time 6.46507453918457
staticrot2 IoU 0.8470544608805054 0.887987640015107 visIoU 0.9238510678349889 0.9446401610463289 CE 2.182360127065739 time 6.758646726608276
staticrot2shift IoU 0.8471093377673836 0.887987640015107 visIoU 0.9237215372716155 0.9447576 …[+411 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/exact_viterbi.py').read()
s=s.replace(\"T0=np.load('/work/scratch/final_candidates_in.npz')['final'].astype(np.float32)\",\"T0=np.load('/work/scratch/xwide_in.npz')['xwide'].astype(np.float32)\")
s=s.replace('np.arange(-10,10.1,2.0)','np.arange(-3,3.1,1.0)')
s=s.replace(\"viterbi_rot.npz\",\"viterbi_rot_fine.npz\")
open('/work/scratch/exact_viterbi_fine.py','w').write(s)
PY
python /work/scratch/exact_viterbi_fine.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
100 time 0.3269004821777344 best 3.0 3.0 0.9557014107704163 base 0.9448946714401245
150 time 1.537745714187622 best -1.0 -1.0 0.9301453232765198 base 0.9291744828224182
200 time 2.779400110244751 best 3.0 -2.0 0.9524025321006775 base 0.9497054815292358
250 time 3.955442428588867 best -3.0 3.0 0.7838447690010071 base 0.7615658640861511
300 time 5.137437105178833 best -3.0 3.0 0.9605137705802917 base 0.9591254591941833
350 time 6.3156373500823975 best 2.0 1.0 0.9181773662567139 base 0.9127255082130432
400 time 7.500324010848999 best 3.0 -1.0 0.900641024112701 …[+709 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/staticrot2_in.npz');f=np.load('/work/scratch/viterbi_rot_fine.npz')['T'];z=dict(z);z['fine']=f;np.savez('/work/scratch/fine_in.npz',**z)
PY
sed 's#candidates.npz#fine_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_fine.py
python /work/scratch/mask_metrics_fine.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'xwide', 'staticrot2', 'fine']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 5.0649755001068115
xwide IoU 0.8452313801980819 0.887987640015107 visIoU 0.9224071493996997 0.9446401610463289 CE 2.172403682329639 time 5.869180917739868
staticrot2 IoU 0.8470544608805054 0.887987640015107 visIoU 0.9238510678349889 0.9446401610463289 CE 2.182360127065739 time 4.89604115486145
fine IoU 0.8513388156442395 0.8941398865781118 visIoU 0.928777141997906 0.9467312348656819 CE 2.171622 …[+377 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/exact_viterbi.py').read()
s=s.replace(\"T0=np.load('/work/scratch/final_candidates_in.npz')['final'].astype(np.float32)\",\"T0=np.load('/work/scratch/viterbi_rot_fine.npz')['T'].astype(np.float32)\")
s=s.replace('np.arange(-10,10.1,2.0)','np.arange(-6,6.1,1.0)')
s=s.replace(\"viterbi_rot.npz\",\"viterbi_rot_fine2.npz\")
open('/work/scratch/exact_viterbi_fine2.py','w').write(s)
PY
python /work/scratch/exact_viterbi_fine2.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
100 time 0.5470151901245117 best 0.0 2.0 0.9601151347160339 base 0.9557014107704163
150 time 4.550786733627319 best 0.0 0.0 0.9301453232765198 base 0.9301453232765198
200 time 8.569502353668213 best 3.0 0.0 0.9519492387771606 base 0.9510425925254822
250 time 13.809831619262695 best -6.0 0.0 0.8154761791229248 base 0.7838447690010071
300 time 17.81420660018921 best 0.0 6.0 0.9624345898628235 base 0.9596007466316223
350 time 21.822559118270874 best 6.0 1.0 0.9249875545501709 base 0.9181773662567139
400 time 25.84000849723816 best 5.0 0.0 0.9079925417900085 ba …[+701 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/fine_in.npz'));z['fine2']=np.load('/work/scratch/viterbi_rot_fine2.npz')['T'];np.savez('/work/scratch/fine2_in.npz',**z)
PY
sed 's#candidates.npz#fine2_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_fine2.py
python /work/scratch/mask_metrics_fine2.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'xwide', 'staticrot2', 'fine', 'fine2']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 8.036951065063477
xwide IoU 0.8452313801980819 0.887987640015107 visIoU 0.9224071493996997 0.9446401610463289 CE 2.172403682329639 time 7.586837530136108
staticrot2 IoU 0.8470544608805054 0.887987640015107 visIoU 0.9238510678349889 0.9446401610463289 CE 2.182360127065739 time 7.55113410949707
fine IoU 0.8513388156442395 0.8941398865781118 visIoU 0.928777141997906 0.9467312348656819 CE  …[+604 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/exact_viterbi.py').read()
s=s.replace(\"T0=np.load('/work/scratch/final_candidates_in.npz')['final'].astype(np.float32)\",\"T0=np.load('/work/scratch/viterbi_rot_fine2.npz')['T'].astype(np.float32)\")
s=s.replace('np.arange(-10,10.1,2.0)','np.arange(-3,3.01,0.5)')
s=s.replace(\"viterbi_rot.npz\",\"viterbi_rot_fine3.npz\")
open('/work/scratch/exact_viterbi_fine3.py','w').write(s)
PY
python /work/scratch/exact_viterbi_fine3.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
100 time 0.7061715126037598 best 0.0 0.5 0.9609375 base 0.9601151347160339
150 time 4.780539274215698 best 0.5 1.0 0.9309859275817871 base 0.9296435117721558
200 time 8.9712815284729 best 2.5 -0.5 0.9519492387771606 base 0.9510425925254822
250 time 13.1519193649292 best -3.0 -2.0 0.8318787217140198 base 0.8132420182228088
300 time 17.349010229110718 best 0.5 3.0 0.9615201950073242 base 0.9605513215065002
350 time 21.321744203567505 best 3.0 1.0 0.92555832862854 base 0.9216362833976746
400 time 25.335472345352173 best 0.0 0.0 0.9079925417900085 base 0.907992 …[+701 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/fine2_in.npz'));z['fine3']=np.load('/work/scratch/viterbi_rot_fine3.npz')['T'];np.savez('/work/scratch/fine3_in.npz',**z)
PY
sed 's#candidates.npz#fine3_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_fine3.py
python /work/scratch/mask_metrics_fine3.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'xwide', 'staticrot2', 'fine', 'fine2', 'fine3']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 5.746220111846924
xwide IoU 0.8452313801980819 0.887987640015107 visIoU 0.9224071493996997 0.9446401610463289 CE 2.172403682329639 time 5.948578119277954
staticrot2 IoU 0.8470544608805054 0.887987640015107 visIoU 0.9238510678349889 0.9446401610463289 CE 2.182360127065739 time 5.370534658432007
fine IoU 0.8513388156442395 0.8941398865781118 visIoU 0.928777141997906 0.9467312348 …[+830 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
f=np.load('/work/scratch/viterbi_rot_fine3.npz')['T'];s=np.load('/work/scratch/static_rot_ref2.npz')['T']
best=f.copy();best[:96]=s[:96];best[797:]=s[797:]
for i in range(90,110):
 d=R.from_matrix(f[i,:3,:3].T@s[95,:3,:3]).magnitude()*180/np.pi
 print(i,round(d,2))
print('steps around')
d=R.from_matrix(np.transpose(best[:-1,:3,:3],(0,2,1))@best[1:,:3,:3]).magnitude()*180/np.pi
print([(i,round(d[i-1],2)) for i in range(92,106)])
print([(i,round(d[i-1],2)) for i in range(790,803)])
PY"
```
<details><summary>output</summary>

```
90 8.94
91 8.94
92 8.94
93 8.94
94 8.94
95 8.94
96 31.5
97 36.2
98 37.01
99 37.3
100 36.9
101 37.3
102 37.3
103 36.51
104 37.31
105 37.12
106 36.62
107 36.7
108 36.61
109 36.35
steps around
[(92, 0.0), (93, 0.0), (94, 0.0), (95, 0.0), (96, 31.5), (97, 5.12), (98, 1.0), (99, 0.5), (100, 0.5), (101, 0.5), (102, 0.0), (103, 1.0), (104, 1.0), (105, 0.45)]
[(790, 10.1), (791, 1.19), (792, 1.11), (793, 3.3), (794, 2.96), (795, 1.87), (796, 6.92), (797, 19.87), (798, 0.0), (799, 0.0), (800, 0.0), (801, 0.0), (802, 0.0)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
zs={n:np.load(p,allow_pickle=True)[k] for n,p,k in [('base','/work/scratch/candidates5.npz','base'),('xwide','/work/scratch/xwide_in.npz','xwide'),('fine3','/work/scratch/viterbi_rot_fine3.npz','T'),('static','/work/scratch/static_rot_ref2.npz','T')]}
ref=zs['static'][95,:3,:3]
for i in range(93,103):
 print(i,[(n,round(R.from_matrix(ref.T@A[i,:3,:3]).magnitude()*180/np.pi,2)) for n,A in zs.items()])
PY"
```
<details><summary>output</summary>

```
93 [('base', 17.83), ('xwide', 8.94), ('fine3', 8.94), ('static', 0.0)]
94 [('base', 17.83), ('xwide', 8.94), ('fine3', 8.94), ('static', 0.0)]
95 [('base', 17.83), ('xwide', 8.94), ('fine3', 8.94), ('static', 0.0)]
96 [('base', 17.83), ('xwide', 27.45), ('fine3', 31.5), ('static', 27.45)]
97 [('base', 17.83), ('xwide', 30.72), ('fine3', 36.2), ('static', 30.72)]
98 [('base', 17.83), ('xwide', 30.72), ('fine3', 37.01), ('static', 30.72)]
99 [('base', 17.83), ('xwide', 30.72), ('fine3', 37.3), ('static', 30.72)]
100 [('base', 17.83), ('xwide', 30.72), ('fine3', 36.9), ('static', 30.72)]
101 [('base', 17.83), ('xwide', 30.72), ('fine3', 37.3), ('static', 30.72)]
102 [('base', 17.83), ('xwide', 30.72), ('fine3', 37.3), ('static', 30.72)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/onset_test.py <<'PY'
exec(open('/work/scratch/exact_search_test.py').read().split('for i in')[0])
# load static/fine
q=np.load('/work/scratch/static_rot_ref2.npz')['T'];f=np.load('/work/scratch/viterbi_rot_fine3.npz')['T'];b=np.load('/work/scratch/candidates5.npz')['base']
for i in range(92,116):
 rms=render([q[95],b[i],f[i]])
 tgt=obj[i];pv=perd[i];tv=tgt&~pv
 out=[]
 for r in rms:
  rv=r&~pv;out.append(round(float((rv&tv).sum()/((rv|tv).sum()+1e-9)),3))
 print(i,out,'projcent',[tuple(np.round(np.where(r)[1].mean(axis=None) if r.any() else [0],1) for _ in [0]) for r in []])
PY
python /work/scratch/onset_test.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
92 [0.968, 0.815, 0.949] projcent []
93 [0.968, 0.8, 0.938] projcent []
94 [0.969, 0.806, 0.943] projcent []
95 [0.969, 0.819, 0.953] projcent []
96 [0.971, 0.805, 0.95] projcent []
97 [0.954, 0.81, 0.959] projcent []
98 [0.966, 0.815, 0.967] projcent []
99 [0.961, 0.81, 0.962] projcent []
100 [0.961, 0.809, 0.961] projcent []
101 [0.963, 0.811, 0.964] projcent []
102 [0.968, 0.814, 0.965] projcent []
103 [0.962, 0.802, 0.962] projcent []
104 [0.963, 0.8, 0.964] projcent []
105 [0.964, 0.801, 0.965] projcent []
106 [0.966, 0.802, 0.966] projcent []
107 [0.9 …[+331 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/range(92,116)/range(116,166,2)/' /work/scratch/onset_test.py; python /work/scratch/onset_test.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
116 [0.947, 0.825, 0.977] projcent []
118 [0.863, 0.836, 0.976] projcent []
120 [0.72, 0.847, 0.972] projcent []
122 [0.527, 0.827, 0.963] projcent []
124 [0.346, 0.938, 0.968] projcent []
126 [0.208, 0.894, 0.943] projcent []
128 [0.079, 0.926, 0.951] projcent []
130 [0.007, 0.9, 0.945] projcent []
132 [0.0, 0.912, 0.938] projcent []
134 [0.0, 0.874, 0.93] projcent []
136 [0.0, 0.861, 0.94] projcent []
138 [0.0, 0.719, 0.943] projcent []
140 [0.0, 0.682, 0.939] projcent []
142 [0.0, 0.734, 0.929] projcent []
144 [0.0, 0.771, 0.941] projcent []
146 [0.0, 0. …[+347 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/range(116,166,2)/range(770,811,2)/' /work/scratch/onset_test.py; python /work/scratch/onset_test.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
770 [0.901, 0.845, 0.99] projcent []
772 [0.872, 0.816, 0.979] projcent []
774 [0.79, 0.712, 0.976] projcent []
776 [0.742, 0.752, 0.974] projcent []
778 [0.738, 0.764, 0.964] projcent []
780 [0.73, 0.782, 0.954] projcent []
782 [0.728, 0.781, 0.953] projcent []
784 [0.725, 0.779, 0.952] projcent []
786 [0.729, 0.783, 0.953] projcent []
788 [0.73, 0.791, 0.962] projcent []
790 [0.74, 0.782, 0.958] projcent []
792 [0.729, 0.779, 0.954] projcent []
794 [0.706, 0.784, 0.956] projcent []
796 [0.702, 0.904, 0.949] projcent []
798 [0.704, 0.891, 0.978] projcent [ …[+228 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/q\\[95\\]/q[858]/g' /work/scratch/onset_test.py; sed -i 's/range(770,811,2)/range(760,859,2)/' /work/scratch/onset_test.py; python /work/scratch/onset_test.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
760 [0.166, 0.967, 0.997] projcent []
762 [0.287, 0.96, 0.994] projcent []
764 [0.388, 0.924, 0.995] projcent []
766 [0.508, 0.922, 1.0] projcent []
768 [0.601, 0.86, 0.986] projcent []
770 [0.724, 0.845, 0.99] projcent []
772 [0.871, 0.816, 0.979] projcent []
774 [0.944, 0.712, 0.976] projcent []
776 [0.984, 0.752, 0.974] projcent []
778 [0.976, 0.764, 0.964] projcent []
780 [0.975, 0.782, 0.954] projcent []
782 [0.977, 0.781, 0.953] projcent []
784 [0.978, 0.779, 0.952] projcent []
786 [0.977, 0.783, 0.953] projcent []
788 [0.977, 0.791, 0.962] projcent [ …[+1328 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
f=np.load('/work/scratch/viterbi_rot_fine3.npz')['T'].copy();s=np.load('/work/scratch/static_rot_ref2.npz')['T']
best=f.copy();best[:116]=s[95];best[776:]=s[858]
np.savez('/work/scratch/best_phase.npz',T=best)
z=dict(np.load('/work/scratch/fine3_in.npz'));z['bestphase']=best;np.savez('/work/scratch/bestphase_in.npz',**z)
PY
sed 's#candidates.npz#bestphase_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_bestphase.py
python /work/scratch/mask_metrics_bestphase.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'xwide', 'staticrot2', 'fine', 'fine2', 'fine3', 'bestphase']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 5.489861965179443
xwide IoU 0.8452313801980819 0.887987640015107 visIoU 0.9224071493996997 0.9446401610463289 CE 2.172403682329639 time 5.2861902713775635
staticrot2 IoU 0.8470544608805054 0.887987640015107 visIoU 0.9238510678349889 0.9446401610463289 CE 2.182360127065739 time 5.3305745124816895
fine IoU 0.8513388156442395 0.8941398865781118 visIoU 0.9287771419979 …[+1070 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
A=np.load('/work/scratch/best_phase.npz')['T']
d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi
dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
print([(i,round(d[i-1],2),round(dt[i-1],2)) for i in list(range(110,122))+list(range(770,781))])
print('dyn rot med/p90/max',np.median(d[116:775]),np.percentile(d[116:775],90),d[116:775].max(),'trans',np.median(dt[116:775]),np.percentile(dt[116:775],90),dt[116:775].max())
PY"
```
<details><summary>output</summary>

```
[(110, 0.0, 0.0), (111, 0.0, 0.0), (112, 0.0, 0.0), (113, 0.0, 0.0), (114, 0.0, 0.0), (115, 0.0, 0.0), (116, 28.91, 2.11), (117, 9.09, 0.51), (118, 1.45, 0.66), (119, 3.39, 0.69), (120, 1.05, 0.77), (121, 2.91, 0.9), (770, 2.99, 1.16), (771, 2.52, 1.38), (772, 3.61, 1.45), (773, 1.23, 1.11), (774, 8.74, 0.91), (775, 1.06, 1.11), (776, 20.45, 14.71), (777, 0.0, 0.0), (778, 0.0, 0.0), (779, 0.0, 0.0), (780, 0.0, 0.0)]
dyn rot med/p90/max 5.15355010932594 10.652628161118335 61.61602194283508 trans 1.0586909 1.8651695013046266 5.324063

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/blend_boundaries.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
A=np.load('/work/scratch/best_phase.npz')['T'].astype(float);F=len(A)
out={}
def slerp(q0,q1,t):
 r0=R.from_quat(q0);r1=R.from_quat(q1)
 # relative rotvec shortest
 rv=(r0.inv()*r1).as_rotvec();return (r0*R.from_rotvec(rv*t)).as_quat()
for w in [5,8,11,15,21]:
 B=A.copy()
 # initial static pose s=A[115], transition to dynamic endpoint A[115+w]
 s=A[115]; e=A[115+w];q0=R.from_matrix(s[:3,:3]).as_quat();q1=R.from_matrix(e[:3,:3]).as_quat()
 for i in range(116,116+w):
  t=(i-115)/w
  B[i,:3,:3]=R.from_quat(slerp(q0,q1,t)).as_matrix();B[i,:3,3]=(1-t)*s[:3,3]+t*e[:3,3]
 # final static pose s=A[776], from dynamic A[776-w]
 s=A[776];e=A[776-w];q0=R.from_matrix(e[:3,:3]).as_quat();q1=R.from_matrix(s[:3,:3]).as_quat()
 for i in range(776-w+1,776):
  t=(i-(776-w))/(w-1)
  B[i,:3,:3]=R.from_quat(slerp(q0,q1,t)).as_matrix();B[i,:3,3]=(1-t)*e[:3,3]+t*s[:3,3]
 out[f'blend{w}']=B
# base candidates
z=dict(np.load('/work/scratch/bestphase_in.npz'));z.update(out);np.savez('/work/scratch/blend_in.npz',**z)
PY
python /work/scratch/blend_boundaries.py
python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/blend_in.npz'));z={k:v for k,v in z.items() if k in ['base','final','fine3','bestphase','blend5','blend8','blend11','blend15','blend21']};np.savez('/work/scratch/blend_metrics_in.npz',**z)
PY
sed 's#candidates.npz#blend_metrics_in.npz#' /work/scratch/mas …[+94 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['final', 'fine3', 'bestphase', 'blend5', 'blend8', 'blend11', 'blend15', 'blend21']
final IoU 0.8128926393361203 0.8765243902435684 visIoU 0.8871691638684842 0.9384042992968423 CE 2.169394635729065 time 6.7342209815979
fine3 IoU 0.8569086317379567 0.8981481481477701 visIoU 0.934592537642507 0.9515624999995044 CE 2.1789470569229445 time 7.080324172973633
bestphase IoU 0.8595250936720115 0.8996699669964028 visIoU 0.9365252367480791 0.9523355576735213 CE 2.192727183956263 time 6.720968723297119
blend5 IoU 0.8594642618836901 0.8996699669964028 visIoU 0.9 …[+1299 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/blend_in.npz')
for name in ['bestphase','blend5','blend8','blend11']:
 A=z[name];d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi;dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
 print(name,'dyn rot',np.median(d[116:775]),np.percentile(d[116:775],90),d[116:775].max(),'all max trans',dt.max(),'trans p90',np.percentile(dt[116:775],90),'near boundaries',[(i,round(d[i-1],1),round(dt[i-1],1)) for i in list(range(114,126))+list(range(768,778))])
PY"
```
<details><summary>output</summary>

```
bestphase dyn rot 5.15355010932594 10.652628161118335 61.61602194283508 all max trans 14.712377 trans p90 1.8651695013046266 near boundaries [(114, 0.0, 0.0), (115, 0.0, 0.0), (116, 28.9, 2.1), (117, 9.1, 0.5), (118, 1.4, 0.7), (119, 3.4, 0.7), (120, 1.0, 0.8), (121, 2.9, 0.9), (122, 4.0, 1.2), (123, 3.0, 1.4), (124, 1.3, 1.2), (125, 3.2, 1.2), (768, 2.0, 1.1), (769, 2.7, 1.2), (770, 3.0, 1.2), (771, 2.5, 1.4), (772, 3.6, 1.4), (773, 1.2, 1.1), (774, 8.7, 0.9), (775, 1.1, 1.1), (776, 20.4, 14.7), (777, 0.0, 0.0)]
blend5 dyn rot 5.129075391199842 10.652628042807697 61.616022910965505 all max trans 5.3240625978170115 trans p90 1.9129347436288504 near boundaries [(114, 0.0, 0.0), (115, 0.0, 0.0), (116, 3.8, 0.5), (117, 3.8, 0.5), (118, 3.8, 0.5), (119, 3.8, 0.5), (120, 3.8, 0.5), (121, 2.9, 0 …[+1292 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
A=np.load('/work/scratch/blend_in.npz')['blend8'];d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi
for i in np.where(d>25)[0]: print(i+1,round(d[i],1))
PY"
```
<details><summary>output</summary>

```
249 30.4
250 29.2
251 30.3
695 61.6
696 45.1

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/spike_sheet.py <<'PY'
exec(open('/work/scratch/vit_overlay.py').read().split(\"cand=inds\")[0])
A=np.load('/work/scratch/blend_in.npz')['blend8']
sel=list(range(244,257,2))+list(range(688,703,2))
thumbs=[]
for i in sel:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();assert ok
 rm=render([A[i]])[0];ov=fr.copy();cs,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,(0,255,0),2)
 ys,xs=np.where(rm);cx=int(xs.mean());cy=int(ys.mean());x0=max(0,cx-110);x1=min(1536,cx+110);y0=max(0,cy-140);y1=min(1152,cy+140)
 crop=cv2.resize(ov[y0:y1,x0:x1],(190,250));cv2.putText(crop,str(i),(4,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);thumbs.append(crop)
canvas=np.zeros((((len(thumbs)+5)//6)*254,6*194,3),np.uint8)
for j,im in enumerate(thumbs):canvas[(j//6)*254:(j//6)*254+250,(j%6)*194:(j%6)*194+190]=im
cv2.imwrite('/work/scratch/spikes.jpg',canvas)
PY
python /work/scratch/spike_sheet.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/spike_sheet.py", line 1, in <module>
    exec(open('/work/scratch/vit_overlay.py').read().split("cand=inds")[0])
  File "<string>", line 7, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 263, in __getitem__
    raise KeyError(f"{key} is not a file in the archive")
KeyError: 'vit is not a file in the archive'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"1s#.*#exec(open('/work/scratch/vit_overlay.py').read().split(\\\"z=np.load('/work/scratch/vit2_in.npz')\\\")[0])#\" /work/scratch/spike_sheet.py; python /work/scratch/spike_sheet.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/spike_sheet.py", line 6, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();assert ok
    ^^^
NameError: name 'cap' is not defined. Did you mean: 'cam'?

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i '2i cap=cv2.VideoCapture(\"/work/inputs/video.mp4\")' /work/scratch/spike_sheet.py; python /work/scratch/spike_sheet.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/spike_sheet.py", line 8, in <module>
    rm=render([A[i]])[0];ov=fr.copy();cs,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cs,-1,(0,255,0),2)
       ^^^^^^
NameError: name 'render' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '2i ctx=dr.RasterizeCudaContext()' /work/scratch/spike_sheet.py; sed -i '3i def render(ps):\\n poses=torch.tensor(np.stack([Tcw@x for x in ps]),dtype=torch.float32,device=\"cuda\");mtx=P[None]@G[None]@poses;clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536));return rast[...,3].flip(1).cpu().numpy()>0' /work/scratch/spike_sheet.py; python /work/scratch/spike_sheet.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/dynamic_spin_test.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
from scipy.spatial.transform import Rotation as SR
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh.visual.material.image=mesh.visual.material.baseColorTexture;mt=make_mesh_tensors(mesh)
z=np.load('/work/scratch/blend_in.npz');T=z['blend8'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ang=np.arange(0,360,30);ctx=dr.RasterizeCudaContext();cap=cv2.VideoCapture('/work/inputs/video.mp4')
inds=np.arange(120,776,25)
allm=[]
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();obs=fr[:,:,::-1].astype(np.float32)/255;tgt=obj[i]&~per[i]
 Ps=[]
 for a in ang:
  Rz=np.eye(4);Rz[:3,:3]=SR.from_euler('z',a,degrees=True).as_matrix();Ps.append(Tcw@T[i]@Rz)
 cols,_,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=torch.tensor(np.array(Ps,dtype=np.float32),device='cuda'),glctx=ctx,mesh_tensors=mt,use_light=False);cols=cols.cpu().numpy()
 metrics=[]
 for a,c in zip(ang,cols):
  rm=c.max(2)>.01;inter=rm&tgt;x=c[inter];y=obs[inter]
  if len(x)<50:metrics.append((-1,9));continue
  pred=np.empty_like(x)
  for ch in range(3):
   A …[+716 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
120 best corr 30 0.938 base0 0.928 180 0.893 mae best 300 top [(30, 0.938), (0, 0.928), (240, 0.916), (270, 0.915)]
145 best corr 300 0.933 base0 0.914 180 0.864 mae best 0 top [(300, 0.933), (270, 0.923), (330, 0.921), (60, 0.916)]
170 best corr 0 0.931 base0 0.931 180 0.889 mae best 60 top [(0, 0.931), (120, 0.921), (330, 0.918), (30, 0.915)]
195 best corr 30 0.9 base0 0.893 180 0.867 mae best 270 top [(30, 0.9), (240, 0.894), (0, 0.893), (210, 0.891)]
220 best corr 0 0.922 base0 0.922 180 0.891 mae best 0 top [(0, 0.922), (30, 0.922), (240, 0.921), (270, …[+2551 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/crop_render_test.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh.visual.material.image=mesh.visual.material.baseColorTexture;mt=make_mesh_tensors(mesh)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
T=np.load('/work/scratch/blend_in.npz')['blend8'];i=670
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read()
# roi 256
cx,cy=780,500; side=256;x0=cx-side//2;y0=cy-side//2
Kp=K.copy();Kp[0,2]-=x0;Kp[1,2]-=y0
col,dep,_=nvdiffrast_render(K=Kp,H=side,W=side,ob_in_cams=torch.tensor((Tcw@T[i])[None].astype(np.float32),device='cuda'),glctx=dr.RasterizeCudaContext(),mesh_tensors=mt,use_light=False)
col=(col[0].cpu().numpy()*255).astype(np.uint8);m=col.max(2)>0;crop=fr[y0:y0+side,x0:x0+side].copy();ov=crop.copy();ov[m]=(0.45*crop[m]+0.55*col[m]).astype(np.uint8);cv2.imwrite('/work/scratch/crop_test.png',np.concatenate([crop,ov],1))
PY
python /work/scratch/crop_render_test.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/spin_viterbi.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py,time
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
from scipy.spatial.transform import Rotation as SR
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh.visual.material.image=mesh.visual.material.baseColorTexture;mt=make_mesh_tensors(mesh)
cam=json.load(open('/work/inputs/camera.json'));K0=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
Tf=np.load('/work/scratch/blend_in.npz')['blend8'].astype(np.float32);F=len(Tf)
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
angles=np.arange(0,360,15);C=len(angles);Rz=[]
for a in angles:
 r=np.eye(4,dtype=np.float32);r[:3,:3]=SR.from_euler('z',a,degrees=True).as_matrix();Rz.append(r)
Rz=np.array(Rz);ctx=dr.RasterizeCudaContext();cap=cv2.VideoCapture('/work/inputs/video.mp4')
cc=np.full((F,C),-1,dtype=np.float32);hc=np.full((F,C),-1,dtype=np.float32);areas=np.zeros(F,dtype=np.int32);t0=time.time()
for i in range(116,776):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();assert ok
 yy,xx=np.where(obj[i]); side=int(max(xx.max()-xx.min(),yy.max()-yy.min())+36);side=max(128,min(320,side));cx=int((xx.min()+xx.max())//2);cy=int((yy.min()+yy.max())//2)
 x0=max(0,min(1536-side,cx-side//2));y0=max(0,min(1152-side,cy-side//2))
 K=K0.copy();K[0 …[+2337 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/spin_viterbi.py", line 22, in <module>
    obs=fr[y0:y0+side,x0:x0+side,:,::-1].astype(np.float32)/255;tgt=obj[i,y0:y0+side,x0:x0+side]&~per[i,y0:y0+side,x0:x0+side];areas[i]=tgt.sum()
        ~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/fr\\[y0:y0+side,x0:x0+side,:\\::-1\\]/fr[y0:y0+side,x0:x0+side,::-1]/' /work/scratch/spin_viterbi.py; python /work/scratch/spin_viterbi.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/spin_viterbi.py", line 22, in <module>
    obs=fr[y0:y0+side,x0:x0+side,:,::-1].astype(np.float32)/255;tgt=obj[i,y0:y0+side,x0:x0+side]&~per[i,y0:y0+side,x0:x0+side];areas[i]=tgt.sum()
        ~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'obs=' /work/scratch/spin_viterbi.py"
```
<details><summary>output</summary>

```
22: obs=fr[y0:y0+side,x0:x0+side,:,::-1].astype(np.float32)/255;tgt=obj[i,y0:y0+side,x0:x0+side]&~per[i,y0:y0+side,x0:x0+side];areas[i]=tgt.sum()
23: hpobs=(obs-cv2.GaussianBlur(obs,(0,0),2)).reshape(-1)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/spin_viterbi.py';s=open(p).read();s=s.replace('fr[y0:y0+side,x0:x0+side,:,::-1]','fr[y0:y0+side,x0:x0+side,::-1]');open(p,'w').write(s)
PY
python /work/scratch/spin_viterbi.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
150 time 9.9 best 15 0.836 base 0.833 besthp 45
200 time 31.5 best 270 0.75 base 0.564 besthp 345
250 time 52.1 best 45 0.741 base 0.614 besthp 60
300 time 64.3 best 330 0.805 base 0.753 besthp 330
350 time 71.3 best 15 0.799 base 0.794 besthp 30
400 time 80.2 best 300 0.788 base 0.704 besthp 300
450 time 90.4 best 150 0.822 base 0.81 besthp 120
500 time 102.4 best 75 0.759 base 0.587 besthp 315
550 time 109.9 best 90 0.475 base 0.436 besthp 345
600 time 116.5 best 60 0.703 base 0.528 besthp 60
650 time 125.3 best 165 0.563 base 0.514 besthp 90
700 time 143 …[+345 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/spin_viterbi_out.npz');a=z['angles'];p=z['path'];sel=z['selang'];sd=np.load('/work/scratch/spin_scores.npz');cc=sd['cc'];hc=sd['hc'];idx=np.arange(116,776)
raw0=cc[idx,0];rawb=cc[idx,p];h0=hc[idx,0];hb=hc[idx,p]
print('ccorr base/best/improve means',raw0.mean(),rawb.mean(),(rawb-raw0).mean(),'median',np.median(rawb-raw0),'frac improve',np.mean(rawb>raw0))
print('hp base/best',h0.mean(),hb.mean(),(hb-h0).mean(),np.mean(hb>h0))
print('first',sel[:30]); print('last',sel[-30:]);print('max jumps',np.percentile(np.abs((np.diff(sel)+180)%360-180),[50,90,99,100]))
# combine data score
score=14*cc+6*hc
print('comb base/best',score[idx,0].mean(),score[idx,p].mean())
PY"
```
<details><summary>output</summary>

```
ccorr base/best/improve means 0.6635526 0.7371503 0.073597714 median 0.033168197 frac improve 0.8166666666666667
hp base/best 0.11553936 0.155413 0.039873656 0.7181818181818181
first [ 15   0 300 300 315 315   0   0   0  30  30  15  15  15  15   0   0   0
   0 345 330   0   0   0   0 345 345   0  15   0]
last [ 45  45  30 300 315 315 315 330 330 345 345 345 345 345 345 345 345 345
 345   0   0   0   0   0   0 345 345 345 345 330]
max jumps [  0.   15.  126.3 165. ]
comb base/best 9.982972 11.252582

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/spin_dp_var.py <<'PY'
import numpy as np
s=np.load('/work/scratch/spin_scores.npz');cc=s['cc'];hc=s['hc'];areas=s['areas'];ang=s['angles'];C=len(ang);F=len(cc)
ccf=np.maximum(cc,0);hcf=np.maximum(hc,0);weight=np.clip(np.sqrt(np.maximum(areas,1)/1800.),.35,1.);data=((1-ccf)*14+(1-hcf)*6)*weight[:,None];bad=cc<0;data[bad]=np.median(data[~bad])
rad=np.deg2rad(ang);sharp=np.abs((rad+np.pi)%(2*np.pi)-np.pi)*180/np.pi;prior=.0008*(sharp/15)**2
D=np.abs((ang[:,None]-ang[None,:]+180)%360-180)
sidx,eidx=116,776;idx=np.arange(sidx,eidx)
base=cc[idx,0].mean();hbase=hc[idx,0].mean()
for maxd,lam in [(30,.05),(30,.1),(45,.08),(45,.15),(60,.2),(180,.25)]:
 trans=lam*(D/15)**2;trans[D>maxd]=1e6
 dp=data[sidx]+prior*20;dp[1:]+=1000
 back=np.zeros((eidx-sidx,C),dtype=np.int16)
 for i in range(sidx+1,eidx):
  vals=dp[None,:]+trans;back[i-sidx]=np.argmin(vals,1);dp=data[i]+np.min(vals,1)+prior
 path=np.zeros(eidx-sidx,dtype=np.int16);path[-1]=0
 for k in range(len(path)-2,-1,-1):path[k]=back[k+1,path[k+1]]
 sel=ang[path];steps=np.abs((np.diff(sel)+180)%360-180); comb=14*cc[idx,path]+6*hc[idx,path]
 print('p',maxd,lam,'sel meanabs',sharp[path].mean(),'score',comb.mean(),'base',14*cc[idx,0].mean()+6*hc[idx,0].mean(),'steps',np.percentile(steps,[50,90,99,100]),'hist0',np.mean(sel==0))
 np.save(f'/work/scratch/spin_path_{maxd}_{str(lam).replace(\".\",\"\")}.npy',path)
PY
python /work/scratch/spin_dp_var.py"
```
<details><summary>output</summary>

```
p 30 0.1 sel meanabs 50.75 score 11.1829605 base 9.982972294092178 steps [ 0. 15. 30. 30.] hist0 0.10909090909090909
p 45 0.08 sel meanabs 51.47727272727273 score 11.18916 base 9.982972294092178 steps [ 0. 15. 45. 45.] hist0 0.10909090909090909
p 45 0.15 sel meanabs 50.88636363636363 score 11.172023 base 9.982972294092178 steps [ 0.  15.  36.3 45. ] hist0 0.11060606060606061
p 60 0.2 sel meanabs 50.65909090909091 score 11.158998 base 9.982972294092178 steps [ 0. 15. 30. 60.] hist0 0.10757575757575757
p 180 0.25 sel meanabs 50.84090909090909 score 11.166793 base 9.982972294092178 steps [ 0. 15. 30. 45.] hist0 0.09696969696969697

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/spin_visual.py <<'PY'
exec(open('/work/scratch/crop_render_test.py').read().split('T=np.load')[0])
cap=cv2.VideoCapture('/work/inputs/video.mp4');T=np.load('/work/scratch/blend_in.npz')['blend8'];paths=np.load('/work/scratch/spin_path_30_01.npy');ang=np.arange(0,360,15);with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:]
inds=[150,200,245,300,400,470,500,550,595,620,650,670,695,720,750]
rows=[]
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();yy,xx=np.where(obj[i]);side=min(240,int(max(xx.max()-xx.min(),yy.max()-yy.min())+50));cx=int((xx.min()+xx.max())//2);cy=int((yy.min()+yy.max())//2);x0=cx-side//2;y0=cy-side//2
 Kp=K.copy();Kp[0,2]-=x0;Kp[1,2]-=y0
 path=paths[i-116];aa=ang[path]
 Ps=[]
 for a in [0,aa]:
  Rz=np.eye(4);Rz[:3,:3]=__import__('scipy').spatial.transform.Rotation.from_euler('z',a,degrees=True).as_matrix();Ps.append(Tcw@T[i]@Rz)
 cols,_,_=nvdiffrast_render(K=Kp,H=side,W=side,ob_in_cams=torch.tensor(np.array(Ps,dtype=np.float32),device='cuda'),glctx=dr.RasterizeCudaContext(),mesh_tensors=mt,use_light=False);cols=(cols.cpu().numpy()*255).astype(np.uint8)
 crop=fr[y0:y0+side,x0:x0+side].copy();rr=[cv2.resize(crop,(200,230))]
 for j,c in enumerate(cols):
  m=c.max(2)>0;ov=crop.copy();ov[m]=(0.25*crop[m]+0.75*c[m]).astype(np.uint8);z=cv2.resize(ov,(200,230));cv2.putText(z,f'{i} spin{int([0,aa][j])}',(3,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1);rr.append(z)
 rows.append(np.concatenate(rr,1)) …[+108 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/spin_visual.py", line 2
    cap=cv2.VideoCapture('/work/inputs/video.mp4');T=np.load('/work/scratch/blend_in.npz')['blend8'];paths=np.load('/work/scratch/spin_path_30_01.npy');ang=np.arange(0,360,15);with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:]
                                                                                                                                                                                ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/spin_visual.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('/work/inputs/video.mp4');T=np.load('/work/scratch/blend_in.npz')['blend8'];paths=np.load('/work/scratch/spin_path_30_01.npy');ang=np.arange(0,360,15);with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:]\",\"cap=cv2.VideoCapture('/work/inputs/video.mp4');T=np.load('/work/scratch/blend_in.npz')['blend8'];paths=np.load('/work/scratch/spin_path_30_01.npy');ang=np.arange(0,360,15)\\nwith h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]\")
open(p,'w').write(s)
PY
python /work/scratch/spin_visual.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/spin_visual.py", line 3, in <module>
    with h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]
         ^^^^
NameError: name 'h5py' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '1s/"'$/;import h5py/'"' /work/scratch/spin_visual.py; python /work/scratch/spin_visual.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/spin_viterbi_out.npz');ang=z['angles'];p=np.load('/work/scratch/spin_path_30_01.npy') if False else np.load('/work/scratch/spin_path_30_01.npy')
# path filenames? list
import glob
print(glob.glob('/work/scratch/spin_path*'))
PY"
```
<details><summary>output</summary>

```
['/work/scratch/spin_path_30_005.npy', '/work/scratch/spin_path_30_01.npy', '/work/scratch/spin_path_45_008.npy', '/work/scratch/spin_path_45_015.npy', '/work/scratch/spin_path_60_02.npy', '/work/scratch/spin_path_180_025.npy']

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/apply_spin.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/blend_in.npz')['blend8'].astype(float).copy();ang=np.arange(0,360,15);p=np.load('/work/scratch/spin_path_30_01.npy')
for idx,i in enumerate(range(116,776)):
 Rz=np.eye(4);Rz[:3,:3]=R.from_euler('z',ang[p[idx]],degrees=True).as_matrix();T[i,:3,:3]=T[i,:3,:3]@Rz[:3,:3]
np.savez('/work/scratch/best_spin.npz',T=T)
print('first',ang[p[:20]]);print('last',ang[p[-20:]])
PY
python /work/scratch/apply_spin.py
python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/blend_in.npz');spin=np.load('/work/scratch/best_spin.npz')['T']
for name,A in [('blend8',z['blend8']),('bestspin',spin)]:
 d=R.from_matrix(np.transpose(A[:-1,:3,:3],(0,2,1))@A[1:,:3,:3]).magnitude()*180/np.pi;dt=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)*100
 print(name,'rot dyn',np.median(d[116:775]),np.percentile(d[116:775],90),d[116:775].max(),'all',d.max(),'trans',np.percentile(dt[116:775],90))
PY
python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/blend_in.npz'));z['bestspin']=np.load('/work/scratch/best_spin.npz')['T'];np.savez('/work/scratch/spin_metrics_in.npz',base=z['base'],blend8=z['blend8'],bestspin=z['bestspin'])
PY
sed 's#candidates.npz#spin_metrics_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_spin.py
python /work/scratch/mask_metrics_spin.py"
```
<details><summary>output</summary>

```
first [  0   0   0  15  15  15   0   0   0  15  15  15  15  15  15   0   0   0
   0 345]
last [345 345 345 345 345 345 345 345 345   0   0   0   0   0   0 345 345 345
 345   0]
blend8 rot dyn 5.129075391199842 10.652628042807697 61.616022910965505 all 61.616022910965505 trans 1.8651694858306693
bestspin rot dyn 6.149613226596499 16.310326894733027 82.41333171943484 all 82.41333171943484 trans 1.8651694858306693
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
KeyError: 'base'
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/m …[+498 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/spin_dp_var.py';s=open(p).read().replace(\"[(30,.05),(30,.1),(45,.08),(45,.15),(60,.2),(180,.25)]\",\"[(15,.05),(15,.1),(15,.2),(30,.2),(30,.4)]\");open('/work/scratch/spin_dp_var2.py','w').write(s)
PY
python /work/scratch/spin_dp_var2.py"
```
<details><summary>output</summary>

```
p 15 0.05 sel meanabs 48.52272727272727 score 11.137955 base 9.982972294092178 steps [ 0. 15. 15. 15.] hist0 0.11212121212121212
p 15 0.1 sel meanabs 48.59090909090909 score 11.1328535 base 9.982972294092178 steps [ 0. 15. 15. 15.] hist0 0.11363636363636363
p 15 0.2 sel meanabs 48.20454545454545 score 11.127175 base 9.982972294092178 steps [ 0. 15. 15. 15.] hist0 0.10606060606060606
p 30 0.2 sel meanabs 51.02272727272727 score 11.163183 base 9.982972294092178 steps [ 0. 15. 30. 30.] hist0 0.10757575757575757
p 30 0.4 sel meanabs 47.88636363636363 score 11.136215 base 9.982972294092178 steps [ 0. 15. 15. 30.] hist0 0.09696969696969697

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import glob,numpy as np
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.signal import savgol_filter
print(glob.glob('/work/scratch/spin_path_15*'))
s=np.load('/work/scratch/spin_scores.npz');cc=s['cc'];hc=s['hc'];areas=s['areas'];ang=np.arange(0,360,15);idx=np.arange(116,776)
base_score=(14*cc[idx,0]+6*hc[idx,0]).mean()
for fn in glob.glob('/work/scratch/spin_path_15*'):
 p=np.load(fn);sel=ang[p];un=np.rad2deg(np.unwrap(np.deg2rad(sel)))
 print('\\n',fn,'raw score',(14*cc[idx,p]+6*hc[idx,p]).mean(),'steps',np.max(np.abs(np.diff(sel))))
 for name,y in [('med3',median_filter(un,3,mode='nearest')),('med5',median_filter(un,5,mode='nearest')),('g1',gaussian_filter1d(un,1)),('g2',gaussian_filter1d(un,2)),('sg7',savgol_filter(un,7,2)),('sg11',savgol_filter(un,11,2))]:
  q=np.rint((y%360)/15).astype(int)%len(ang);score=(14*cc[idx,q]+6*hc[idx,q]).mean();st=np.abs((np.diff(ang[q])+180)%360-180)
  print(name,'score',round(float(score),4),'steps',np.percentile(st,[50,90,100]))
 print('base',base_score)
PY"
```
<details><summary>output</summary>

```
['/work/scratch/spin_path_15_005.npy', '/work/scratch/spin_path_15_01.npy', '/work/scratch/spin_path_15_02.npy']

 /work/scratch/spin_path_15_005.npy raw score 11.137955 steps 345
med3 score 11.1271 steps [ 0. 15. 15.]
med5 score 11.1213 steps [ 0. 15. 15.]
g1 score 11.1292 steps [ 0. 15. 15.]
g2 score 11.0874 steps [ 0. 15. 15.]
sg7 score 11.126 steps [ 0. 15. 15.]
sg11 score 11.1026 steps [ 0. 15. 15.]
base 9.982972

 /work/scratch/spin_path_15_01.npy raw score 11.1328535 steps 345
med3 score 11.1275 steps [ 0. 15. 15.]
med5 score 11.1223 steps [ 0. 15. 15.]
g1 score 11.1296 steps [ 0. 15. 15.]
g2 score 11.0915 steps [ 0. 15. 15.]
sg7 score 11.127 steps [ 0. 15. 15.]
sg11 score 11.1091 steps [ 0. 15. 15.]
base 9.982972

 /work/scratch/spin_path_15_02.npy raw score 11.127175 steps 345
med …[+239 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/finalize_traj.py <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
base=np.load('/work/scratch/blend_in.npz')['blend8'].astype(float).copy()
ang=np.arange(0,360,15);p=np.load('/work/scratch/spin_path_15_02.npy');sel=ang[p]
un=np.rad2deg(np.unwrap(np.deg2rad(sel)));sm=gaussian_filter1d(un,1,mode='nearest')
# force exact endpoint corrections and gently blend first/last 5 frames
sm[0]=0;sm[-1]=0
sm[:5]*=np.linspace(0,1,5);sm[-5:]*=np.linspace(1,0,5)
q=np.rint((sm%360)/15).astype(int)%len(ang);corr=ang[q]
T=base.copy()
for idx,i in enumerate(range(116,776)):
 Rz=np.eye(4);Rz[:3,:3]=R.from_euler('z',corr[idx],degrees=True).as_matrix();T[i,:3,:3]=base[i,:3,:3]@Rz[:3,:3]
np.savez('/work/scratch/final_traj.npz',T=T,corr=corr)
print('corr first',corr[:20]);print('corr last',corr[-20:]);print('corr abs mean',np.mean(np.minimum(corr,360-corr)))
PY
python /work/scratch/finalize_traj.py
python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/blend_in.npz'));z['finaltraj']=np.load('/work/scratch/final_traj.npz')['T'];z['base']=np.load('/work/scratch/candidates5.npz')['base'];np.savez('/work/scratch/finaltraj_in.npz',base=z['base'],blend8=z['blend8'],finaltraj=z['finaltraj'])
PY
sed 's#candidates.npz#finaltraj_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_finaltraj.py
python /work/scratch/mask_metrics_finaltraj.py"
```
<details><summary>output</summary>

```
corr first [  0   0   0  15  15  15   0   0   0  15  15  15  15  15  15   0   0   0
   0 345]
corr last [345 345 345 345 345 345 345 345 345   0   0   0   0   0   0 345 255 180
  90   0]
corr abs mean 48.63636363636363
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'blend8', 'finaltraj']
base IoU 0.6894899307535464 0.731187472813949 visIoU 0.7723512453781898 0.8095238095234744 CE 5.898781214838638 time 6.735148668289185
blend8 IoU 0.8588263102561765 0.8991111111107115 visIoU 0.936091560614887 0.9518633540367744 CE 2.220876781264475 time 6.711014986038208
finaltraj IoU 0.8553894005599338 0 …[+372 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/finalize_traj.py';s=open(p).read();s=s.replace(\"sm[0]=0;sm[-1]=0\\n# force exact endpoint corrections and gently blend first/last 5 frames\\nsm[:5]*=np.linspace(0,1,5);sm[-5:]*=np.linspace(1,0,5)\",\"# force endpoints to exact 0 modulo 360 using the nearest representative\\nsm[0]=0\\nsm[-1]=360*np.round(sm[-2]/360.)\\n# gently fade the first correction in from the static pose\\nsm[:5]*=np.linspace(0,1,5)\")
open(p,'w').write(s)
PY
python /work/scratch/finalize_traj.py"
```
<details><summary>output</summary>

```
corr first [  0   0   0  15  15  15   0   0   0  15  15  15  15  15  15   0   0   0
   0 345]
corr last [345 345 345 345 345 345 345 345 345   0   0   0   0   0   0 345 255 180
  90   0]
corr abs mean 48.63636363636363

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
ang=np.arange(0,360,15);p=np.load('/work/scratch/spin_path_15_02.npy');sel=ang[p];un=np.rad2deg(np.unwrap(np.deg2rad(sel)));sm=gaussian_filter1d(un,1,mode='nearest');print(sel[-20:]);print(un[-20:]);print(sm[-20:])
PY"
```
<details><summary>output</summary>

```
[345 345 345 345 345 345 345 345 345   0   0   0   0   0   0 345 345 345
 345   0]
[345. 345. 345. 345. 345. 345. 345. 345. 345. 360. 360. 360. 360. 360.
 360. 345. 345. 345. 345. 360.]
[344.12164771 344.93151462 344.99799254 345.         345.
 345.00200746 345.06848538 345.87835229 349.50792398 355.49207602
 359.12164771 359.92950716 359.92950716 359.12164771 355.49207602
 349.50993144 345.94683768 345.94683768 349.50993144 355.49207602]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'sm\\[' /work/scratch/finalize_traj.py"
```
<details><summary>output</summary>

```
8:sm[0]=0;sm[-1]=0
9:sm[:5]*=np.linspace(0,1,5);sm[-5:]*=np.linspace(1,0,5)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/finalize_traj.py';s=open(p).read();s=s.replace(\"sm[0]=0;sm[-1]=0\\nsm[:5]*=np.linspace(0,1,5);sm[-5:]*=np.linspace(1,0,5)\",\"sm[0]=0\\nsm[-1]=360*np.round(sm[-2]/360.)\\nsm[:5]*=np.linspace(0,1,5)\")
open(p,'w').write(s)
PY
python /work/scratch/finalize_traj.py"
```
<details><summary>output</summary>

```
corr first [  0   0   0  15  15  15   0   0   0  15  15  15  15  15  15   0   0   0
   0 345]
corr last [345 345 345 345 345 345 345 345 345   0   0   0   0   0   0 345 345 345
 345   0]
corr abs mean 48.13636363636363

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/blend_in.npz');A=z['blend8'];B=np.load('/work/scratch/final_traj.npz')['T']
for name,T in [('blend8',A),('final',B)]:
 d=R.from_matrix(np.transpose(T[:-1,:3,:3],(0,2,1))@T[1:,:3,:3]).magnitude()*180/np.pi;dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)*100
 print(name,'rot dyn',np.median(d[116:775]),np.percentile(d[116:775],90),d[116:775].max(),'all max',d.max(),'trans p90',np.percentile(dt[116:775],90),dt[116:775].max())
PY
python - <<'PY'
import numpy as np
z=dict(np.load('/work/scratch/blend_in.npz'));z['finaltraj']=np.load('/work/scratch/final_traj.npz')['T'];z['base']=np.load('/work/scratch/candidates5.npz')['base'];np.savez('/work/scratch/finaltraj_in.npz',base=z['base'],blend8=z['blend8'],finaltraj=z['finaltraj'])
PY
sed 's#candidates.npz#finaltraj_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_finaltraj.py
python /work/scratch/mask_metrics_finaltraj.py"
```
<details><summary>output</summary>

```
blend8 rot dyn 5.129075391199842 10.652628042807697 61.616022910965505 all max 61.616022910965505 trans p90 1.8651694858306693 5.3240625978170115
final rot dyn 6.0812501666139545 15.106595585728481 61.616022910965505 all max 61.616022910965505 trans p90 1.8651694858306693 5.3240625978170115
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'blend8', 'finaltraj']
base IoU 0.6894899307535464 0.731187472813949 visIoU 0.7723512453781898 0.8095238095234744 CE 5.898781214838638 time 5.709715127944946
blend8 IoU 0.8588263102561765 0.8991111111107115 visIoU 0.936091560614887 0.9518633540367744 CE 2. …[+444 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/summary_metrics.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
m=np.load('/work/scratch/mask_metrics.npz');inds=m['inds']
Tbase=np.load('/work/scratch/finaltraj_in.npz')['base'];Tres=np.load('/work/scratch/finaltraj_in.npz')['finaltraj']
# masks phases
for name,T in [('baseline',Tbase),('result',Tres)]:
 print(name)
 for s,e in [(0,116),(116,776),(776,859)]:
  mm=(inds>=s)&(inds<e); print(' phase',s,e,'n',mm.sum(),'iou',m[name][mm,0].mean(),'vis',m[name][mm,1].mean(),'ce',m[name][mm,2].mean())
  d=R.from_matrix(np.transpose(T[s:e-1,:3,:3],(0,2,1))@T[s+1:e,:3,:3]).magnitude()*180/np.pi
  dt=np.linalg.norm(np.diff(T[s:e,:3,3],axis=0),axis=1)*100
  print('  rotstep',np.median(d),np.percentile(d,90),'transstep',np.median(dt),np.percentile(dt,90),'max',dt.max())
# spin scores on dynamic
s=np.load('/work/scratch/spin_scores.npz');cc=s['cc'];hc=s['hc'];idx=np.arange(116,776);corr=np.load('/work/scratch/final_traj.npz')['corr'];q=np.rint((corr%360)/15).astype(int)%24
print('photo ccorr base/result',cc[idx,0].mean(),cc[idx,q].mean(),'hcorr',hc[idx,0].mean(),hc[idx,q].mean(),'improved frac',np.mean(cc[idx,q]>cc[idx,0]))
PY
python /work/scratch/summary_metrics.py"
```
<details><summary>output</summary>

```
baseline
Traceback (most recent call last):
  File "/work/scratch/summary_metrics.py", line 9, in <module>
    mm=(inds>=s)&(inds<e); print(' phase',s,e,'n',mm.sum(),'iou',m[name][mm,0].mean(),'vis',m[name][mm,1].mean(),'ce',m[name][mm,2].mean())
                                                                 ~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 263, in __getitem__
    raise KeyError(f"{key} is not a file in the archive")
KeyError: 'baseline is not a file in the archive'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/m\\[name\\]/m[\"base\" if name==\"baseline\" else name]/' /work/scratch/summary_metrics.py; python /work/scratch/summary_metrics.py"
```
<details><summary>output</summary>

```
baseline
Traceback (most recent call last):
  File "/work/scratch/summary_metrics.py", line 9, in <module>
    mm=(inds>=s)&(inds<e); print(' phase',s,e,'n',mm.sum(),'iou',m["base" if name=="baseline" else name][mm,0].mean(),'vis',m[name][mm,1].mean(),'ce',m[name][mm,2].mean())
                                                                                                                            ~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 263, in __getitem__
    raise KeyError(f"{key} is not a file in the archive")
KeyError: 'baseline is not a file in the archive'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/summary_metrics.py';s=open(p).read();s=s.replace(\"for name,T in [('baseline',Tbase),('result',Tres)]:\\n print(name)\",\"for name,T in [('baseline',Tbase),('result',Tres)]:\\n key='base' if name=='baseline' else name;print(name)\")
s=s.replace('m[\"base\" if name==\"baseline\" else name]','m[key]').replace('m[name]','m[key]')
open(p,'w').write(s)
PY
python /work/scratch/summary_metrics.py"
```
<details><summary>output</summary>

```
baseline
 phase 0 116 n 39 iou 0.777658211935773 vis 0.8431514360713837 ce 3.9367764126774247
  rotstep 3.661898210282912e-15 2.60641094037941 transstep 0.047623618073232156 0.9966560887301832 max 2.378426084621125
 phase 116 776 n 220 iou 0.661542218944438 vis 0.7488973184061221 ce 6.139485412807384
  rotstep 6.728453981247139 23.811589513803863 transstep 1.5636452933515348 3.3276835500619897 max 9.875283329216922
 phase 776 859 n 28 iou 0.7862732747498683 vis 0.8580175488360604 ce 6.740326348094465
  rotstep 4.64228745351725e-15 14.386293980534653 transstep 0.018205857220712753 1.7872031269164572 max 13.427998936921536
result
Traceback (most recent call last):
  File "/work/scratch/summary_metrics.py", line 9, in <module>
    mm=(inds>=s)&(inds<e); print(' phase',s,e,'n',mm.sum(),'iou',m …[+343 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/key='base' if name=='baseline' else name/key='base' if name=='baseline' else 'finaltraj'/\" /work/scratch/summary_metrics.py; python /work/scratch/summary_metrics.py"
```
<details><summary>output</summary>

```
baseline
 phase 0 116 n 39 iou 0.777658211935773 vis 0.8431514360713837 ce 3.9367764126774247
  rotstep 3.661898210282912e-15 2.60641094037941 transstep 0.047623618073232156 0.9966560887301832 max 2.378426084621125
 phase 116 776 n 220 iou 0.661542218944438 vis 0.7488973184061221 ce 6.139485412807384
  rotstep 6.728453981247139 23.811589513803863 transstep 1.5636452933515348 3.3276835500619897 max 9.875283329216922
 phase 776 859 n 28 iou 0.7862732747498683 vis 0.8580175488360604 ce 6.740326348094465
  rotstep 4.64228745351725e-15 14.386293980534653 transstep 0.018205857220712753 1.7872031269164572 max 13.427998936921536
result
 phase 0 116 n 39 iou 0.9381520479860556 vis 0.9753644907536215 ce 0.8145916024291174
  rotstep 6.204432138754706e-14 6.204432138754706e-14 transstep 0.0 0.0 max 0. …[+483 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_validation.py <<'PY'
exec(open('/work/scratch/depth_metrics_test.py').read().split('for kname')[0])
Ts={'base':np.load('/work/scratch/finaltraj_in.npz')['base'],'result':np.load('/work/scratch/finaltraj_in.npz')['finaltraj']}
K=np.array(cam['K']);P=torch.tensor(projection_matrix_from_intrinsics(K,1152,1536,.001,100),dtype=torch.float32,device='cuda')
allerr={k:[] for k in Ts};allbias={k:[] for k in Ts}
for i in range(0,859,25):
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: dep=cv2.imdecode(f['aligned']['front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
 for name,T in Ts.items():
  pose=T[i];M=Tcw@pose;mtx=P[None]@G[None]@torch.tensor(M[None],dtype=torch.float32,device='cuda');clip=(mtx[:,None]@homo[None,:,:,None])[...,0];rast,_=dr.rasterize(ctx,clip,Fc,resolution=(1152,1536));vc=torch.tensor((M[:3,:3]@mesh.vertices.T).T+M[:3,3],dtype=torch.float32,device='cuda').contiguous();xyz,_=dr.interpolate(vc[None].contiguous(),rast.contiguous(),Fc.contiguous());zmap=xyz[0,:,:,2].flip(0).cpu().numpy();rm=rast[0,:,:,3].flip(0).cpu().numpy()>0
  valid=rm&~perd[i];d1=zmap[valid];d2=dep[valid];ok=np.isfinite(d1)&(d2>.1);err=np.abs(d1[ok]-d2[ok]);allerr[name].append((i,np.median(err)));allbias[name].append((i,np.median(d1[ok]-d2[ok])))
for name in Ts:
 print(name,'median abs depth err',np.median([x[1] for x in allerr[name]]),'p90 frames',np.percentile([x[1] for x in allerr[name]],90),'median bias',np.median …[+100 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
base median abs depth err 0.1878854 p90 frames 0.27342863082885743 median bias 0.121815205
result median abs depth err 0.1543932 p90 frames 0.2726490020751953 median bias 0.1300447
errors {'base': [(0, 0.06958175), (25, 0.020178199), (50, 0.09070635), (75, 0.07910955), (100, 0.10809183), (125, 2.020141), (150, 0.016918182), (175, 0.26373613), (200, 0.13311934), (225, 0.22531772), (250, 0.098463535), (275, 0.21226025), (300, 0.17262673), (325, 0.2076602), (350, 0.097162485), (375, 0.21060991), (400, 0.23092437), (425, 0.03329587), (450, 0.1878854), (475, 0.2 …[+972 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Deterministic trajectory recovery for a monocular rigid-object video.

The method starts from the released frame-wise reconstruction, enforces the two
visually verified rest phases, robustly filters the manipulation interval,
re-centres each silhouette in the image, decodes smooth multi-frame rotation
corrections with Viterbi, and finally resolves spin about the bottle'"'s long axis
using the textured mesh.  It does not read any hidden/ground-truth annotation.
\"\"\"
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

SEED = 20261020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = 'cuda'
H, W = 1152, 1536


def projection_matrix(K: np.ndarray, h: int, w: int, near=0.001, far=100.0) -> np.ndarray:
    \"\"\"OpenCV intrinsic -> OpenGL clip projection used by nvdiffrast (y down).\"\"\"
    depth = float(far - near)
    q = -(far + near) / depth
    qn = -2.0 * far * near / depth
    return np.array([
        [2.0 * K[0, 0] / w, -2.0 * K[0, 1] / w, (-2.0 * K[0, 2] + w) / w, 0.0],
       …[+22293 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"dp = data[0] + prior\\n    dp[1:] += 1000.0  # the first manipulated frame still matches the rest pose\",\"dp = data[0] + prior\\n    zero_idx = int(np.argmin(np.abs(angle_values)))\\n    dp[zero_idx] = data[0, zero_idx] + prior[zero_idx]\\n    dp[np.arange(na) "'!= zero_idx] += 1000.0  # first frame still matches rest pose")
s=s.replace("dp += 1000.0'"\\n    dp[0] = 0.0  # and the last manipulated frame returns to the final rest pose\\n    path = np.zeros(e - s, dtype=np.int16)\\n    path[-1] = int(np.argmin(dp))\",\"dp += 1000.0\\n    dp[zero_idx] = data[-1, zero_idx] + prior[zero_idx]  # returns to final rest pose\\n    path = np.zeros(e - s, dtype=np.int16)\\n    path[-1] = zero_idx\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
contact brackets: 96..796
Traceback (most recent call last):
  File "/work/outputs/solve.py", line 569, in <module>
    main()
  File "/work/outputs/solve.py", line 481, in main
    S0 = shift_static_pose(S0, np.arange(0, first_contact), renderer, obj, perd,
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 223, in shift_static_pose
    r = renderer.masks(out[segment_ids[0]][None])[0]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 141, in masks
    poses_world = np.asarray(poses_world, dtype=np.float32).reshape(-1, 4, 4)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: cannot reshape array of size 4 into shape (4,4)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"r = renderer.masks(out[segment_ids[0]][None])[0]\",\"r = renderer.masks(out[None])[0]\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
contact brackets: 96..796
initial rest anchor/refinement: frame 27
final rest anchor/refinement: frame 818
rest phases extended to [0,116) and [775,859)
  rotation search frame 116/775
  rotation search frame 196/775
^CTraceback (most recent call last):
  File "/work/outputs/solve.py", line 569, in <module>
    main()
  File "/work/outputs/solve.py", line 528, in main
    T = rotation_viterbi(T, s, e, np.arange(-36, 36.1, 6.0),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 289, in rotation_viterbi
    rms = renderer.masks(np.asarray(poses))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 149, in masks
    return rast[..., 3].flip(1).cpu().numpy() > 0
           ^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+21 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
old=\"\"\"    def masks(self, poses_world: np.ndarray) -> np.ndarray:
        poses_world = np.asarray(poses_world, dtype=np.float32).reshape(-1, 4, 4)
        cam = torch.tensor(np.stack([self.Tcw @ p for p in poses_world]),
                           dtype=torch.float32, device=DEVICE)
        mtx = self.P[None] @ self.G[None] @ cam
        clip = (mtx[:, None] @ self.homo[None, :, :, None])[..., 0]
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.faces,
                               resolution=(H, W))
        # nvdiffrast triangle IDs are 1-based; background is 0.
        return rast[..., 3].flip(1).cpu().numpy() > 0
\"\"\"
new=\"\"\"    def raster_ids(self, poses_world: np.ndarray):
        poses_world = np.asarray(poses_world, dtype=np.float32).reshape(-1, 4, 4)
        cam = torch.tensor(np.stack([self.Tcw @ p for p in poses_world]),
                           dtype=torch.float32, device=DEVICE)
        mtx = self.P[None] @ self.G[None] @ cam
        clip = (mtx[:, None] @ self.homo[None, :, :, None])[..., 0]
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.faces,
                               resolution=(H, W))
        # nvdiffrast triangle IDs are 1-based; background is 0.
        return rast[..., 3].flip(1)

    def masks(self, poses_world: np.ndarray) -> np.ndarray:
        return self.raster_ids(poses_world).cpu().numpy() > 0
\"\"\"
assert old in s;s= …[+1300 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
contact brackets: 96..796
initial rest anchor/refinement: frame 27
final rest anchor/refinement: frame 818
rest phases extended to [0,116) and [775,859)
  rotation search frame 116/775
  rotation search frame 196/775
  rotation search frame 276/775
  rotation search frame 356/775
  rotation search frame 436/775
  rotation search frame 516/775
  rotation search frame 596/775
  rotation search frame 676/775
  rotation search frame 756/775
  rotation search frame 116/775
  rotation search frame 196/775
  rotation search frame 276/775
  rotation search frame 356/775
  rotation search frame 436/775
  rotation search frame 516/775
  rotation search frame 596/775
  rotation search frame 676/775
  rotation search frame 756/775
  rotation search frame 116/775
  rotation searc …[+627 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
s=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];print(s.shape,s.dtype,np.isfinite(s).all(),s[0],s[-1]);print('frames given',s.shape[0])
manual=np.load('/work/scratch/finaltraj_in.npz');base=manual['base'];fin=manual['finaltraj'];np.savez('/work/scratch/solve_eval_in.npz',base=base,manual=fin,solve=s.astype(float));print('diff solve/manual rot median',np.median(__import__('scipy').spatial.transform.Rotation.from_matrix(np.transpose(fin[:,:3,:3],(0,2,1))@s[:,:3,:3]).magnitude()*180/np.pi),'trans',np.median(np.linalg.norm(fin[:,:3,3]-s[:,:3,3],axis=1)))
PY
sed 's#candidates.npz#solve_eval_in.npz#' /work/scratch/mask_metrics.py > /work/scratch/mask_metrics_solve.py
python /work/scratch/mask_metrics_solve.py"
```
<details><summary>output</summary>

```
(859, 4, 4) float64 True [[-0.99910835  0.02349426 -0.03507883 -0.17872817]
 [ 0.04035032  0.28686326 -0.95712137  0.24624525]
 [-0.01242403 -0.9576834  -0.28755548  3.42116863]
 [ 0.          0.          0.          1.        ]] [[-0.42213113  0.90641216  0.01491007 -0.17821814]
 [-0.12682501 -0.04276295 -0.9910029   0.25175814]
 [-0.89761948 -0.42022414  0.13300731  3.52533405]
 [ 0.          0.          0.          1.        ]]
frames given 859
diff solve/manual rot median 2.549920178436242 trans 0.0011834547633929743
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
names ['base', 'manual', 'solve']
ba …[+665 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/photo_validate.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/workspace/v2d_cari4d/lib')
from cari4d.Utils import make_mesh_tensors,nvdiffrast_render
import nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh.visual.material.image=mesh.visual.material.baseColorTexture;mt=make_mesh_tensors(mesh)
cam=json.load(open('/work/inputs/camera.json'));K0=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
z=np.load('/work/scratch/solve_eval_in.npz');Ts={'base':z['base'],'solve':z['solve']}
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');out={k:[] for k in Ts}
for i in range(120,776,20):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,fr=cap.read();assert ok
 yy,xx=np.where(obj[i]);side=int(np.clip(max(xx.max()-xx.min(),yy.max()-yy.min())+36,128,320));cx=int((xx.min()+xx.max())//2);cy=int((yy.min()+yy.max())//2);x0=int(np.clip(cx-side//2,0,1536-side));y0=int(np.clip(cy-side//2,0,1152-side));K=K0.copy();K[0,2]-=x0;K[1,2]-=y0
 obs=fr[y0:y0+side,x0:x0+side,::-1].astype(np.float32)/255;tgt=obj[i,y0:y0+side,x0:x0+side]&~per[i,y0:y0+side,x0:x0+side]
 for name,T in Ts.items():
  col,_,_=nvdiffrast_render(K=K,H=side,W=side,ob_in_cams=torch.tensor((Tcw@T[i])[None].astype(np.float32),device='cuda'),glctx=dr.RasterizeCtx if False else None,mesh_tensors=mt,use_light=Fals …[+1296 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
base 0.6408227039397659 0.7063961270438186 33
solve 0.7315567720169572 0.7539329766873585 33
improved 0.7575757575757576 delta 0.09073406807719143

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s#z=np.load('/work/inputs/reconstruction.npz'); T=z\\['T_world_mesh'\\]#z=np.load('/work/outputs/object_poses_world.npz'); T=z['T_world_mesh']#\" /work/scratch/render_baseline.py | sed 's#baseline_renders.jpg#solve_renders.jpg#' > /work/scratch/render_solve.py
python /work/scratch/render_solve.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for name,s,e in [('initial-rest',0,116),('manipulation',116,775),('final-rest',775,859)]:
 d=R.from_matrix(np.transpose(T[s:e-1,:3,:3],(0,2,1))@T[s+1:e,:3,:3]).magnitude()*180/np.pi
 dt=np.linalg.norm(np.diff(T[s:e,:3,3],axis=0),axis=1)*100
 print(name,'rot median/p90/max',np.median(d),np.percentile(d,90),d.max(),'trans median/p90/max cm',np.median(dt),np.percentile(dt,90),dt.max())
# static variance
for s,e in [(0,116),(775,859)]:
 print('static',s,e,'trans range cm',(T[s:e,:3,3].max(0)-T[s:e,:3,3].min(0))*100,'rot max deg',R.from_matrix(np.transpose(T[s:s+1,:3,:3],(0,2,1))@T[s:e,:3,:3]).magnitude().max()*180/np.pi)
PY"
```
<details><summary>output</summary>

```
initial-rest rot median/p90/max 0.0 0.0 0.0 trans median/p90/max cm 0.0 0.0 0.0
manipulation rot median/p90/max 6.015508542842483 15.169556179878454 57.78175333859994 trans median/p90/max cm 1.069598923179116 1.862441918956486 5.26122060370471
final-rest rot median/p90/max 0.0 0.0 0.0 trans median/p90/max cm 0.0 0.0 0.0
static 0 116 trans range cm [0. 0. 0.] rot max deg 0.0
static 775 859 trans range cm [0. 0. 0.] rot max deg 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,torch,cv2,trimesh,h5py
mods=[('numpy',numpy),('scipy',scipy),('torch',torch),('opencv-python',cv2),('trimesh',trimesh),('h5py',h5py)]
for n,m in mods: print(n,getattr(m,'__version__','unknown'))
import nvdiffrast
print('nvdiffrast',getattr(nvdiffrast,'__version__','unknown'),nvdiffrast.__file__)
PY
cd /code/reconstruction && git rev-parse HEAD && git log -1 --format='%H %ad %s' --date=iso-strict
sha256sum /work/outputs/object_poses_world.npz /work/inputs/object_mesh.glb /work/inputs/reconstruction.npz"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
torch 2.5.1+cu124
opencv-python 4.11.0
trimesh 5.1.1
h5py 3.16.0
nvdiffrast 0.4.0 /opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80 2026-10-01T23:10:28-07:00 Add challenge FAQ links to contact and submission sections (#191)
d58955d35116f07063fece76f6ce4471808c2ca36b72d548f760e32dd38d0270  /work/outputs/object_poses_world.npz
ebc656d5de2c6fb0075acd136503ac936b6c821fbfe05b21286efbc9af7225d6  /work/inputs/object_mesh.glb
eedb974ea351822f94897e36ae01db88893d8236370828e4ab6acf191b54284a  /work/inputs/reconstruction.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
m=np.load('/work/scratch/mask_metrics.npz');inds=m['inds']
for key in ['base','solve']:
 print(key)
 for s,e in [(0,116),(116,775),(775,859)]:
  mm=(inds>=s)&(inds<e)
  print(s,e,'iou',round(float(m[key][mm,0].mean()),4),'vis',round(float(m[key][mm,1].mean()),4),'ce',round(float(m[key][mm,2].mean()),3))
PY
sed \"s/Ts={'base':np.load('\\/work\\/scratch\\/finaltraj_in.npz')\\['base'\\],'result':np.load('\\/work\\/scratch\\/finaltraj_in.npz')\\['finaltraj'\\]}/Ts={'base':np.load('\\/work\\/scratch\\/solve_eval_in.npz')['base'],'result':np.load('\\/work\\/scratch\\/solve_eval_in.npz')['solve']}/\" /work/scratch/depth_validation.py > /work/scratch/depth_validation_solve.py
python /work/scratch/depth_validation_solve.py | tail -3"
```
<details><summary>output</summary>

```
base median abs depth err 0.1878854 p90 frames 0.27342863082885743 median bias 0.121815205
result median abs depth err 0.15544295 p90 frames 0.2736844539642334 median bias 0.112027764
errors {'base': [(0, 0.06958175), (25, 0.020178199), (50, 0.09070635), (75, 0.07910955), (100, 0.10809183), (125, 2.020141), (150, 0.016918182), (175, 0.26373613), (200, 0.13311934), (225, 0.22531772), (250, 0.098463535), (275, 0.21226025), (300, 0.17262673), (325, 0.2076602), (350, 0.097162485), (375, 0.21060991), (400, 0.23092437), (425, 0.03329587), (450, 0.1878854), (475, 0.20326257), (500, 0.28203273), (525, 0.2798903), (550, 0.21457171), (575, 0.09885335), (600, 0.26187444), (625, 0.20803332), (650, 0.14561903), (675, 1.6892877), (700, 0.22323537), (725, 0.048567295), (750, 0.027462482), (775, 0.2313826 …[+737 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Monocular 6-DoF Bottle Trajectory

## Result

- Output: "'`object_poses_world.npz`, key `T_world_mesh`, shape `(859, 4, 4)`, `float64`.
- SHA-256: `d58955d35116f07063fece76f6ce4471808c2ca36b72d548f760e32dd38d0270`.
- Object: `cyan_water_bottle`; exact supplied textured mesh.
- Released world frame and OpenCV camera convention are used throughout.
- No ground-truth or challenge annotation was accessed.

## Phases found

The hand-contact probability brackets the active interval at frames `96..796`. Mask evidence against the pooled rest poses showed that the object remains stationary after first touch and before final release, so the physical rest phases were extended using a sustained mask-IoU criterion:

| Phase | Frames | Time | Interpretation | Confidence |
|---|---:|---:|---|---|
| Initial rest | 0-115 | 0.00-3.87 s | Bottle standing on the round table before lift-off | High |
| Manipulation | 116-774 | 3.87-25.83 s | Pickup, alternating-hand rotation, and handovers | Medium-high |
| Final rest | 775-858 | 25.83-28.63 s | Bottle replaced on the round table | High |

Eight-frame SE(3) blends are used at the two rest/manipulation boundaries to remove residual depth jumps while keeping the rest poses exactly constant.

## Method

1. **Rest-pose evidence pooling.** Candidate rest frames were scored by rendering the exact mesh and comparing its visible silhouette to the SAM2 object mask on all other rest frames. The best a …[+7268 chars]
```
<details><summary>output</summary>

```
['T_world_mesh'] (859, 4, 4) float64

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
d58955d35116f07063fece76f6ce4471808c2ca36b72d548f760e32dd38d0270  /work/outputs/object_poses_world.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '/work/scratch\\|/pipeline\\|/workspace\\|/code/reconstruction\\|http' /work/outputs/solve.py || true"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /work && python - <<'PY'
import numpy as np, hashlib
p='/work/outputs/object_poses_world.npz'
z=np.load(p)
T=z['T_world_mesh']
assert z.files==['T_world_mesh']
assert T.shape==(859,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
assert np.allclose(T[:,3,:],[0,0,0,1])
print('OK',T.shape,T.dtype,hashlib.sha256(open(p,'rb').read()).hexdigest())
PY
ls -la /work/outputs"
```
<details><summary>output</summary>

```
total 156
drwxrwxr-x 3 hanzhou hanzhou   4096 Oct 20 20:20 .
drwxrwxr-x 9 hanzhou hanzhou   4096 Oct 20 18:56 ..
-rw-rw-r-- 1 hanzhou hanzhou   6599 Oct 20 20:20 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   1920 Oct 20 20:20 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 20 20:08 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 110230 Oct 20 20:18 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  24389 Oct 20 20:08 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import os,glob
for p in glob.glob('/work/outputs/__pycache__/*'):
 os.remove(p)
os.rmdir('/work/outputs/__pycache__')
PY
ls -la /work/outputs"
```
<details><summary>output</summary>

```
total 152
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 20 20:21 .
drwxrwxr-x 9 hanzhou hanzhou   4096 Oct 20 18:56 ..
-rw-rw-r-- 1 hanzhou hanzhou   6599 Oct 20 20:20 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   1920 Oct 20 20:20 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 110230 Oct 20 20:18 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  24389 Oct 20 20:08 solve.py

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/object_poses_world.npz` — `(859, 4, 4)` `float64`, key `T_world_mesh`
- `/work/outputs/solve.py` — deterministic offline reproduction script
- `/work/outputs/REPORT.md` — phases, method, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — tools, versions, revisions, hashes

Validation highlights:

| Phase | Baseline visible IoU | Result visible IoU | Centroid error |
|---|---:|---:|---:|
| Initial rest | 0.843 | **0.970** | 3.94 → **1.20 px** |
| Manipulation | 0.749 | **0.915** | 6.14 → **2.79 px** |
| Final rest | 0.858 | **0.972** | 6.74 → **1.50 px** |

Additional checks showed improved textured-render correlation (`0.641 → 0.732`) and reduced MoGe depth residual (`0.188 m → 0.155 m`). Both static phases are exactly constant in the output. Final `solve.py` runtime was 582 seconds.

Output SHA-256: `d58955d35116f07063fece76f6ce4471808c2ca36b72d548f760e32dd38d0270`.
