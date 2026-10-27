# T1_ds_ep015

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T17:14:29Z

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/AGENT_TASK.md'
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
and translation both matter; a  …[+9792 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ls -lah /work/inputs && echo '---' && cat /work/inputs/task.txt && echo '---' && cat /work/inputs/episode.json && echo '---' && cat /work/inputs/camera.json && echo '---' && cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
total 75M
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 17:50 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 21:08 ..
-rw-rw-r-- 1 hanzhou hanzhou  829 Oct 20 17:49 camera.json
-rw-rw-r-- 1 hanzhou hanzhou   69 Oct 20 17:49 episode.json
-rw-rw-r-- 1 hanzhou hanzhou 190K Oct 20 17:50 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou  398 Oct 20 17:49 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou 6.4M Oct 20 17:50 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou  13M Oct 20 17:49 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou  35M Oct 20 17:50 reconstruction.npz
-rw-rw-r-- 1 hanzhou hanzhou  149 Oct 20 17:49 task.txt
-rw-rw-r-- 1 hanzhou hanzhou  21M Oct 20 17:49 video.mp4
---
Walk to the wet floor sign, pick it up with one hand and hold onto it while walking around for at least 10 seconds, then place it b …[+1326 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work -maxdepth 2 -type f | sed 's#"'^#/##'"' | head -200 && echo '--- dirs' && find /work -maxdepth 2 -type d | sort | head -100"
```
<details><summary>output</summary>

```
//work/inputs/video.mp4
//work/inputs/object_mesh.glb
//work/inputs/ground_plane.json
//work/inputs/task.txt
//work/inputs/episode.json
//work/inputs/camera.json
//work/inputs/reconstruction.npz
//work/inputs/frame0_reference_human.npz
//work/inputs/masks.h5
//work/logs/run_manifest.txt
//work/logs/proxy.log
//work/logs/codex_events.jsonl
//work/logs/codex_stderr.txt
//work/.codex_home/model_catalog.json
//work/.codex_home/config.toml
//work/.codex_home/state_5.sqlite
//work/.codex_home/state_5.sqlite-wal
//work/.codex_home/state_5.sqlite-shm
//work/.codex_home/logs_2.sqlite
//work/.codex_home/logs_2.sqlite-wal
//work/.codex_home/logs_2.sqlite-shm
//work/.codex_home/goals_1.sqlite
//work/.codex_home/goals_1.sqlite-wal
//work/.codex_home/goals_1.sqlite-shm
//work/.codex_home/memories_1.sqli …[+722 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader && python - <<'PY'
import numpy as np
p='/work/inputs/reconstruction.npz'
z=np.load(p)
for k in z.files:
 print(k,z[k].shape,z[k].dtype)
PY"
```
<details><summary>output</summary>

```
T_world_mesh (715, 4, 4) float32
T_cam_mesh_foundationpose (715, 4, 4) float32
T_cam_mesh_coconet (715, 4, 4) float32
human_vertices_world (715, 4610, 3) float32
human_keypoints_world (715, 70, 3) float32
contact_prob_right_left (715, 2) float32

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '35,90p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
   compare with the image: silhouette agreement with the object mask (excluding pixels covered by the person),
   edges / texture / colour agreement, consistency with the MoGe-2 depth map; optical flow or feature tracks between
   neighbouring frames (a correct trajectory must explain the image motion of the object's texture).
3. **Several hypotheses per frame, not one**: the baseline, the FoundationPose and CoCoNet estimates, their 180 deg
   symmetry flips about each principal axis of the mesh, poses propagated from neighbouring frames, and fresh
   registrations (e.g. render-and-compare optimisation from several initialisations, or FoundationPose
   re-registration on chosen frames).
4. **Global decoding over time**: choose one hypothesis per frame (or per segment) jointly, e.g. dynamic …[+4101 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 print(f.keys())
 for k in f:
  x=f[k]; print(k,x.shape,x.dtype)
  if len(x.shape)==3:
   for i in [0,1,50,100,150,200,300,400,500,600,700]:
    if i<x.shape[0]: print(i,int(x[i].sum()))
PY"
```
<details><summary>output</summary>

```
<KeysViewHDF5 ['object', 'person']>
object (715, 1152, 1536) bool
0 8610
1 8512
50 8427
100 8375
150 8191
200 9413
300 9061
400 6857
500 1596
600 5865
700 6266
person (715, 1152, 1536) bool
0 70082
1 69004
50 45898
100 26193
150 24477
200 19579
300 48482
400 71831
500 34116
600 52453
700 88622

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json, numpy as np, trimesh
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
print(type(m), len(m) if hasattr(m,'__len__') else None)
try:
 print('bounds',m.bounds,'extents',m.extents,'verts',len(m.vertices),'faces',len(m.faces),'visual',type(m.visual))
except Exception as e: print(e)
z=np.load('/work/inputs/reconstruction.npz')
for k in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 T=z[k].astype(float)
 print(k)
 print('det',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
 tr=T[:,:3,3]
 print('trans min max',tr.min(0),tr.max(0),'med',np.median(tr,0))
 # rot relative consecutive angles
 R=T[:,:3,:3]
 ang=np.degrees(np.arccos(np.clip((np.trace(R[1:]@R[:-1].transpose(0,2,1),axis1=1,axis2=2)-1)/2,-1,1)))
 d=np.linalg.norm(tr[1:]-tr[:-1],axis=1)
 print('ang stats',np.quantile(ang,[0,.5,.9,.95,.99,1]),'trans',np.quantile(d,[0,.5,.9,.95,.99,1]))
 print('large ang idx',np.where(ang>20)[0][:80]+1)
 print('large trans idx',np.where(d>0.05)[0][:80]+1)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> None
bounds [[-0.13955034 -0.14493057 -0.30348039]
 [ 0.14996175  0.13710172  0.34234235]] extents [0.2895121  0.2820323  0.64582273] verts 199398 faces 66486 visual <class 'trimesh.visual.texture.TextureVisuals'>
T_world_mesh
det 0.9999962302822418 1.0000006922814
trans min max [-0.74246287  0.54082471  2.15948009] [1.02394772 0.95285714 3.82427692] med [-0.02743914  0.82309926  3.36980033]
ang stats [ 0.          0.98793899  3.1762809   4.36057918 11.74709744 22.79049976] trans [0.         0.01142963 0.03295545 0.04122009 0.0702959  0.10525141]
large ang idx [487 495]
large trans idx [171 174 178 179 180 182 184 185 254 255 279 315 343 440 477 480 487 489
 491 498 499 503 525]
T_cam_mesh_foundationpose
det 0.9999962380084939 1.0000006324806259
trans min max …[+1158 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /work/inputs/video.mp4'
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
avg_frame_rate=30/1
duration=23.833333
nb_frames=715

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); R=T[:,:3,:3]; t=T[:,:3,3]
# root relative rotation Euler
from scipy.spatial.transform import Rotation
rpy=Rotation.from_matrix(R).as_euler('xyz',degrees=True)
print('i t xyz rpy contact')
for i in list(range(0,715,25)):
 print(i,*np.round(t[i],3),*np.round(rpy[i],1),*np.round(z['contact_prob_right_left'][i],2))
# find constant phases translation/rot speed smoothed
for i in range(1,715):
 ang=np.degrees(np.arccos(np.clip((np.trace(R[i]@R[i-1].T)-1)/2,-1,1)))
 d=np.linalg.norm(t[i]-t[i-1])
 if ang>8 or d>.07: print('JUMP',i,'ang',round(ang,1),'d',round(d,3),'t',np.round(t[i],3),'c',np.round(z['contact_prob_right_left'][i],2))
PY"
```
<details><summary>output</summary>

```
i t xyz rpy contact
0 0.07 0.86 3.616 90.9 34.3 -0.6 0.0 0.0
25 0.069 0.854 3.593 90.9 34.3 -0.6 0.0 0.0
50 0.07 0.859 3.613 90.9 34.3 -0.6 0.0 0.0
75 0.07 0.87 3.658 90.9 34.3 -0.6 0.0 0.0
100 0.07 0.879 3.695 90.9 34.3 -0.6 0.0 0.0
125 0.07 0.869 3.652 90.9 34.3 -0.6 0.0 0.0
150 0.07 0.865 3.638 90.9 34.3 -0.6 0.0 1.0
175 0.04 0.828 3.666 90.0 42.8 1.1 0.0 1.0
200 -0.066 0.605 3.607 98.1 81.8 4.9 0.0 1.0
225 -0.258 0.571 3.527 -94.8 46.7 171.9 0.0 1.0
250 -0.388 0.67 3.425 -103.5 40.4 177.2 0.0 1.0
275 -0.186 0.649 3.058 -86.2 36.8 -179.1 0.0 1.0
300 -0.226 0.761 2.95 -108.1 25.2 177.9 0.0 1.0
325 -0.044 0.811 2.63 -97.4 45.4 -178.5 0.0 1.0
350 0.49 0.796 2.188 -92.8 81.0 179.8 0.0 1.0
375 0.981 0.845 2.266 85.1 30.2 -8.8 0.0 1.0
400 0.943 0.768 2.702 89.7 15.8 -6.8 0.0 1.0
425 0.806 0.7 …[+1770 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float)
# contacts stats
c=z['contact_prob_right_left']
print('contacts sample')
for i in range(0,715,25): print(i, np.round(c[i],3))
print('max contact range',c.max(0), 'first>0.5',np.where(c.max(1)>.5)[0][:10], np.where(c.max(1)>.5)[0][-10:])
# bounding boxes of object masks per 10
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(0,715,10):
  ys,xs=np.where(f['object'][i])
  if len(xs):
   print(i, len(xs), (xs.min(),ys.min(),xs.max(),ys.max()), 'cent',(xs.mean(),ys.mean()))
PY"
```
<details><summary>output</summary>

```
contacts sample
0 [0. 0.]
25 [0. 0.]
50 [0. 0.]
75 [0. 0.]
100 [0. 0.]
125 [0. 0.]
150 [0. 1.]
175 [0. 1.]
200 [0. 1.]
225 [0. 1.]
250 [0. 1.]
275 [0. 1.]
300 [0. 1.]
325 [0. 1.]
350 [0. 1.]
375 [0. 1.]
400 [0. 1.]
425 [0.    0.995]
450 [0.    0.682]
475 [0. 0.]
500 [0.    0.993]
525 [0.    0.818]
550 [0. 1.]
575 [0.    0.999]
600 [0. 1.]
625 [0. 0.]
650 [0. 0.]
675 [0. 0.]
700 [0. 0.]
max contact range [1.566994e-04 1.000000e+00] first>0.5 [128 129 130 131 132 133 134 135 136 137] [606 607 608 609 610 611 612 613 614 619]
0 8610 (699, 652, 779, 821) cent (738.8387921022068, 740.4908246225319)
10 8702 (699, 652, 779, 821) cent (739.0371179039302, 741.0055159733395)
20 8433 (699, 652, 779, 821) cent (738.9233961816673, 740.4578441835646)
30 8448 (699, 652, 779, 821) cent (738.9753787878788, …[+5046 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/analysis && python - <<'PY'
import cv2, numpy as np, os
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[]
idx=[0,50,100,125,140,150,160,170,175,180,185,190,195,200,205,210,215,220,225,230,240,250,275,300,325,350,360,370,375,380,390,400,420,440,460,475,480,485,490,495,500,505,510,520,540,560,580,600,620,650,700]
for i in idx:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read();
 if not ok: continue
 # crop around object bbox but at least 360x360
 h,w=im.shape[:2]
 # just resize entire image half
 thumb=cv2.resize(im,(384,288))
 cv2.putText(thumb,str(i),(8,26),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2,cv2.LINE_AA)
 frames.append(thumb)
rows=[]
for j in range(0,len(frames),6):
 row=frames[j:j+6]
 while len(row)<6: row.append(np.zeros_like(frames[0]))
 rows.append(np.hstack(row))
out=np.vstack(rows)
cv2.imwrite('/work/scratch/analysis/contact_raw.jpg',out)
print(out.shape)
PY"
```
<details><summary>output</summary>

```
(2592, 2304, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'find /pipeline -maxdepth 3 -type f | head -100'
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json
/pipeline/.stages/01_depth.json
/pipeline/.stages/02_export.json
/pipeline/.stages/03_human_initialization.json
/pipeline/.stages/05_object_pose.json
/pipeline/.stages/06_coconet.json
/pipeline/.stages/07_refinement.json
/pipeline/.stages/08_render.json
/pipeline/depth/moge2/depth_report.json
/pipeline/depth/moge2/aligned.h5
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock
/pipeline/depth/moge2/intrinsics.pkl
/pipeline/depth/moge2/raw.npy
/pipeline/export/episode_000015/edex
/pipeline/export/episode_000015/wild_export.json
/pipeline/foundationpose/episode_000015.pkl
/pipeline/inference/.materialized_inputs.h5.lock
/pipeline/inference/coconet.pth
/pipeline/inference/materialized_inputs.h5
/pipeline/inference/refined.pth
/pipeline/inference/epis …[+361 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m.visual.material)
print('uv',m.visual.uv.shape,m.visual.uv.min(0),m.visual.uv.max(0))
im=m.visual.material.image
print('image',im.size,im.mode)
print('centroid',m.centroid,'volume',m.volume,'is_watertight',m.is_watertight)
print('principal',trimesh.principal_axes(m))
print('obb transform\\n',trimesh.bounds.oriented_bounds(m)[0])
PY"
```
<details><summary>output</summary>

```
<trimesh.visual.material.PBRMaterial object at 0x787709d0ad50>
uv (199398, 2) [0.00082324 0.00116748] [0.99924588 0.99916834]
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['nvdiffrast','pyrender','pytorch3d','open3d','trimesh','cv2','torch','scipy','skimage','h5py']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'NO',e)
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
pytorch3d OK 0.7.9
open3d OK 0.20.0
trimesh OK 5.1.1
cv2 OK 4.11.0
torch OK 2.5.1+cu124
scipy OK 1.16.3
skimage OK 0.26.0
h5py OK 3.16.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/.stages/05_object_pose.json && echo --- && cat /pipeline/.stages/06_coconet.json && echo --- && cat /pipeline/.stages/07_refinement.json && echo --- && cat /pipeline/.stages/08_render.json'
```
<details><summary>output</summary>

```
{
  "name": "05_object_pose",
  "outputs": [
    {
      "mtime_ns": 1791255952137606885,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000015/foundationpose/episode_000015.pkl",
      "size": 205089
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000015/export/episode_000015",
      "--depth-root",
      "/home/hanzhou/Projects/v2d/run/out/episode_000015/depth/moge2/aligned.h5",
      "--out-file",
      "/home/hanzhou/Projects/v2d/run/out/episode_000015/foundationpose/episode_000015.pkl",
      "--camera-id",
      "0",
      "--iteration",
      "5",
      "--max-attempts",
      "5",
 …[+8469 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
p='/pipeline/foundationpose/episode_000015.pkl'
o=pickle.load(open(p,'rb'))
print(type(o))
if isinstance(o,dict):
 print(o.keys())
 for k,v in o.items():
  import numpy as np
  print(k,type(v),getattr(v,'shape',None))
  if isinstance(v,(list,tuple)): print(' len',len(v),type(v[0]) if v else None)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mat=m.visual.material
print(mat.__dict__.keys())
for k,v in mat.__dict__.items():
 print(k, type(v), getattr(v,'shape',None), getattr(v,'size',None), getattr(v,'mode',None))
PY"
```
<details><summary>output</summary>

```
dict_keys(['_data'])
_data <class 'dict'> None None None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
o=pickle.load(open('/pipeline/foundationpose/episode_000015.pkl','rb'))
print('metadata',o['metadata'])
for k in ['fp_candidate_count','fp_selected_candidate_index','fp_selected_iou','fp_selected_visibility','fp_reliable','fp_translation_threshold_m','fp_selected_translation_distance_m','fp_tracking_state','fp_confirmed_pose_source','fp_emitted_pose_source','fp_gap_length','fp_observed_visible_pixels','fp_rendered_visible_pixels','fp_rendered_visibility','fp_full_image_iou','fp_tolerant_precision','fp_tolerant_recall','fp_centroid_distance','fp_confirmation_count','fp_rejection_mask','fp_reacquisition_latency']:
 a=np.asarray(o[k]); print('\\n',k,a.shape,a.dtype)
 for i in range(0,715,50): print(i,a[i])
PY"
```
<details><summary>output</summary>

```
metadata {'source': 'foundationpose_register_first_then_track_metric_depth', 'depth_root': '/home/hanzhou/Projects/v2d/run/out/episode_000015/depth/moge2/aligned.h5', 'depth_source': 'metric_depth', 'depth_input_identity': {'path': '/home/hanzhou/Projects/v2d/run/out/episode_000015/depth/moge2/aligned.h5', 'size': 880128268, 'mtime_ns': 1791255880227607159}, 'mesh': '/home/hanzhou/Projects/v2d/run/out/episode_000015/export/episode_000015/object_mesh/output_aligned.glb', 'mask_identities': {'human': {'raw': {'storage': 'h5', 'source': {'path': '/home/hanzhou/Projects/v2d/run/out/episode_000015/export/episode_000015/human_masks/front_stereo_camera_left.h5', 'size': 17587621, 'mtime_ns': 1791255448075608808}}, 'effective': None}, 'object': {'raw': {'storage': 'h5', 'source': {'path': '/home/h …[+10590 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mat=m.visual.material
print(repr(mat._data))
for k,v in mat._data.items():
 print(k,type(v),getattr(v,'shape',None),getattr(v,'size',None),getattr(v,'mode',None),repr(v)[:300])
PY"
```
<details><summary>output</summary>

```
{'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=4755x4626 at 0x7D756190B4D0>, 'doubleSided': True, 'name': '_texture'}
metallicFactor <class 'float'> None None None 0.0
baseColorTexture <class 'PIL.Image.Image'> None (4755, 4626) RGB <PIL.Image.Image image mode=RGB size=4755x4626 at 0x7D756190B4D0>
doubleSided <class 'bool'> None None None True
name <class 'str'> None None None '_texture'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /pipeline/inference/episode_000015_step200000_before_after.mp4 && mkdir -p /work/scratch/vid && ffmpeg -hide_banner -loglevel error -i /pipeline/inference/episode_000015_step200000_before_after.mp4 -vf \"select='eq(n,0)+eq(n,175)+eq(n,200)+eq(n,225)+eq(n,250)+eq(n,350)+eq(n,375)+eq(n,400)+eq(n,475)+eq(n,500)+eq(n,525)+eq(n,600)'\" -vsync 0 /work/scratch/vid/ba_%02d.jpg && ls -lh /work/scratch/vid"
```
<details><summary>output</summary>

```
width=4608
height=1236
r_frame_rate=30/1
duration=23.833333
nb_frames=715
total 2.3M
-rw-rw-r-- 1 hanzhou hanzhou 242K Oct 20 21:09 ba_01.jpg
-rw-rw-r-- 1 hanzhou hanzhou 276K Oct 20 21:09 ba_02.jpg
-rw-rw-r-- 1 hanzhou hanzhou 258K Oct 20 21:09 ba_03.jpg
-rw-rw-r-- 1 hanzhou hanzhou 240K Oct 20 21:09 ba_04.jpg
-rw-rw-r-- 1 hanzhou hanzhou 206K Oct 20 21:09 ba_05.jpg
-rw-rw-r-- 1 hanzhou hanzhou 190K Oct 20 21:09 ba_06.jpg
-rw-rw-r-- 1 hanzhou hanzhou 168K Oct 20 21:09 ba_07.jpg
-rw-rw-r-- 1 hanzhou hanzhou 154K Oct 20 21:09 ba_08.jpg
-rw-rw-r-- 1 hanzhou hanzhou 141K Oct 20 21:09 ba_09.jpg
-rw-rw-r-- 1 hanzhou hanzhou 141K Oct 20 21:09 ba_10.jpg
-rw-rw-r-- 1 hanzhou hanzhou 141K Oct 20 21:09 ba_11.jpg
-rw-rw-r-- 1 hanzhou hanzhou 144K Oct 20 21:09 ba_12.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,glob
fs=glob.glob('/work/scratch/vid/ba_*.jpg')
ims=[]
for f in fs:
 im=cv2.imread(f); im=cv2.resize(im,(1152,309)); ims.append(im)
out=np.vstack(ims)
cv2.imwrite('/work/scratch/vid/montage.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,90])
print(out.shape)
PY"
```
<details><summary>output</summary>

```
(3708, 1152, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
im=m.visual.material.baseColorTexture
im.thumbnail((1200,1200))
im.save('/work/scratch/analysis/texture.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz');c=json.load(open('/work/inputs/camera.json')); W=np.array(c['T_world_cam']); Winv=np.linalg.inv(W)
T=z['T_world_mesh'].astype(float); Tc=Winv@T
for key in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 C=z[key].astype(float)
 # rotation err
 Rerr=C[:,:3,:3]@np.swapaxes(Tc[:,:3,:3],1,2)
 ang=np.degrees(np.arccos(np.clip((np.trace(Rerr,axis1=1,axis2=2)-1)/2,-1,1)))
 dt=np.linalg.norm(C[:,:3,3]-Tc[:,:3,3],axis=1)
 print(key,'ang',np.quantile(ang,[0,.5,.9,.95,.99,1]),'dt',np.quantile(dt,[0,.5,.9,.95,.99,1]))
for i in range(0,715,25):
 print(i, np.round(Tc[i,:3,3],3),ang[i] if False else '')
PY"
```
<details><summary>output</summary>

```
T_cam_mesh_foundationpose ang [  0.39933746   6.49178583  16.06612774  19.36448118  78.71524304
 167.34312591] dt [0.03586485 0.10990136 0.18175556 0.20357129 0.3703342  1.32164555]
T_cam_mesh_coconet ang [1.05111313 1.2462511  1.26378031 1.26955623 1.27373076 1.27457905] dt [0.05563574 0.11256953 0.15769728 0.16298644 0.17510656 0.17952406]
0 [-0.081  0.758  3.579] 
25 [-0.08   0.753  3.556] 
50 [-0.081  0.757  3.576] 
75 [-0.082  0.767  3.621] 
100 [-0.082  0.775  3.658] 
125 [-0.081  0.766  3.615] 
150 [-0.081  0.763  3.601] 
175 [-0.111  0.724  3.627] 
200 [-0.215  0.503  3.557] 
225 [-0.405  0.472  3.471] 
250 [-0.533  0.574  3.37 ] 
275 [-0.322  0.568  3.007] 
300 [-0.359  0.684  2.902] 
325 [-0.169  0.747  2.589] 
350 [0.375 0.749 2.16 ] 
375 [0.864 0.797 2.253] 
400 [0.815 0.703 2. …[+333 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/test_render.py <<'PY'
import torch, numpy as np, trimesh, json, h5py, cv2
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation
# load
M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(M.vertices,dtype=torch.float32,device='cuda')
F=torch.tensor(M.faces,dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device='cuda')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); W=np.array(cam['T_world_cam'])
Winv=torch.tensor(np.linalg.inv(W),dtype=torch.float32,device='cuda')
i=0
Tcm=Winv@T[i]
vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
vc=(Tcm@vh.T).T[:,:3]
fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
for ys in [1,-1]:
 clip=torch.stack([fx*vc[:,0]+cx*vc[:,2], ys*(fy*vc[:,1]+cy*vc[:,2]), vc[:,2], vc[:,2]],1)[None]
 glctx=dr.RasterizeCudaContext()
 rast,_=dr.rasterize(glctx,clip,F,resolution=(1152,1536))
 m=(rast[0,:,:,3]>0).cpu().numpy()
 with h5py.File('/work/inputs/masks.h5','r') as h: gt=h['object'][i]
 for fl in [False,True]:
  mm=m[::-1] if fl else m
  inter=(mm&gt).sum(); uni=(mm|gt).sum()
  print('ys',ys,'flip',fl,'sum',mm.sum(),'iou',inter/uni,'prec',inter/mm.sum(),'rec',inter/gt.sum())
 cv2.imwrite(f'/work/scratch/analysis/render_y{ys}.png',(m*255).astype(np.uint8))
PY
python /work/scratch/analysis/test_render.py"
```
<details><summary>output</summary>

```
/work/scratch/analysis/test_render.py:25: RuntimeWarning: invalid value encountered in scalar divide
  print('ys',ys,'flip',fl,'sum',mm.sum(),'iou',inter/uni,'prec',inter/mm.sum(),'rec',inter/gt.sum())
ys 1 flip False sum 0 iou 0.0 prec nan rec 0.0
ys 1 flip True sum 0 iou 0.0 prec nan rec 0.0
ys -1 flip False sum 0 iou 0.0 prec nan rec 0.0
ys -1 flip True sum 0 iou 0.0 prec nan rec 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/vc\\[:,2\\], vc\\[:,2\\]/torch.zeros_like(vc[:,2]), vc[:,2]/' /work/scratch/analysis/test_render.py && python /work/scratch/analysis/test_render.py"
```
<details><summary>output</summary>

```
/work/scratch/analysis/test_render.py:25: RuntimeWarning: invalid value encountered in scalar divide
  print('ys',ys,'flip',fl,'sum',mm.sum(),'iou',inter/uni,'prec',inter/mm.sum(),'rec',inter/gt.sum())
ys 1 flip False sum 0 iou 0.0 prec nan rec 0.0
ys 1 flip True sum 0 iou 0.0 prec nan rec 0.0
ys -1 flip False sum 0 iou 0.0 prec nan rec 0.0
ys -1 flip True sum 0 iou 0.0 prec nan rec 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep clip /work/scratch/analysis/test_render.py && python - <<'PY'
import torch,nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext()
v=torch.tensor([[[-0.5,-0.5,0,1],[0.5,-0.5,0,1],[0,0.5,0,1]]],device='cuda')
f=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
r,_=dr.rasterize(ctx,v,f,resolution=(100,100))
print(r.shape,r[...,3].max(),(r[...,3]>0).sum())
PY"
```
<details><summary>output</summary>

```
torch.Size([1, 100, 100, 4]) tensor(1., device='cuda:0') tensor(1250, device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/analysis/test_render.py').read().split('for ys')[0])
print(vc.min(0),vc.max(0), W, Tcm)
print(fx,fy,cx,cy)
PY"
```
<details><summary>output</summary>

```
torch.return_types.min(
values=tensor([-0.2713,  0.4152,  3.3925], device='cuda:0'),
indices=tensor([   457,   7209, 121249], device='cuda:0')) torch.return_types.max(
values=tensor([0.1199, 1.0685, 3.7795], device='cuda:0'),
indices=tensor([ 78638, 138360,  13142], device='cuda:0')) [[ 0.99969078  0.0026545   0.02472446  0.05957703]
 [-0.00358835  0.99927877  0.03780304 -0.03325376]
 [-0.02460628 -0.03788007  0.99897929  0.06746381]
 [ 0.          0.          0.          1.        ]] tensor([[ 0.8394,  0.5433, -0.0154, -0.0806],
        [ 0.0149, -0.0513, -0.9986,  0.7578],
        [-0.5433,  0.8380, -0.0511,  3.5788],
        [ 0.0000,  0.0000,  0.0000,  1.0000]], device='cuda:0')
820.0010375976562 820.0010375976562 758.1000366210938 560.5

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/test_render.py'
s=open(p).read()
s=s.replace(\"clip=torch.stack([fx*vc[:,0]+cx*vc[:,2], ys*(fy*vc[:,1]+cy*vc[:,2]), torch.zeros_like(vc[:,2]), vc[:,2]],1)[None]\",\"clip=torch.stack([(2*fx/1536)*vc[:,0]+(2*cx/1536-1)*vc[:,2], ys*((2*fy/1152)*vc[:,1]+(2*cy/1152-1)*vc[:,2]), torch.zeros_like(vc[:,2]), vc[:,2]],1)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/test_render.py"
```
<details><summary>output</summary>

```
ys 1 flip False sum 8172 iou 0.8275073505390396 prec 0.9298825256975036 rec 0.8825783972125436
ys 1 flip True sum 8172 iou 0.0 prec 0.0 rec 0.0
ys -1 flip False sum 8172 iou 0.0 prec 0.0 rec 0.0
ys -1 flip True sum 8172 iou 0.8275073505390396 prec 0.9298825256975036 rec 0.8825783972125436

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.vertices
c=v.mean(0); C=np.cov((v-c).T); w,U=np.linalg.eigh(C)
print('bounds',m.bounds,'center',c,'mean',m.centroid)
print('eigvals',w,'axes cols\\n',U)
# SVD oriented from trimesh
S,T=trimesh.bounds.oriented_bounds(m)
print('S',S,'T',T,'extents',m.extents)
PY"
```
<details><summary>output</summary>

```
bounds [[-0.13955034 -0.14493057 -0.30348039]
 [ 0.14996175  0.13710172  0.34234235]] center [ 0.00232044 -0.00188225  0.01819139] mean [ 6.20172116e-11 -1.20533349e-10  4.34089368e-10]
eigvals [0.00699448 0.00717254 0.03937269] axes cols
 [[-0.13356738 -0.99024254 -0.03974251]
 [ 0.99101301 -0.13375085  0.00198191]
 [-0.00727816 -0.03912063  0.99920799]]
S [[-6.03289009e-08  9.99999999e-01  4.02355273e-05  3.92596758e-03]
 [ 9.99994608e-01 -7.17965332e-08  3.28379215e-03 -4.22300917e-03]
 [ 3.28379215e-03  4.02355084e-05 -9.99994608e-01  1.96533364e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] T [0.28203242 0.28952054 0.64536667] extents [0.2895121  0.2820323  0.64582273]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
for i in [0,50,100,150,175,200,225,250,300,350,375,400,450,475,500,525,550,575,600,700]:
 R=T[i,:3,:3]
 print(i,'xaxis',np.round(R[:,0],2),'yaxis',np.round(R[:,1],2),'zaxis',np.round(R[:,2],2),'t',np.round(T[i,:3,3],2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
sel=[0,150,175,190,200,210,220,225,250,300,350,375,400,450,475,485,490,495,500,505,510,525,550,575,600,620,700]
with h5py.File('/work/inputs/masks.h5','r') as f:
 tiles=[]
 for i in sel:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
  ys,xs=np.where(f['object'][i]); x0,x1=max(0,xs.min()-80),min(1536,xs.max()+80); y0,y1=max(0,ys.min()-80),min(1152,ys.max()+80)
  # force size 300x300 preserving by padding maybe
  crop=im[y0:y1,x0:x1].copy()
  cv2.putText(crop,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
  # resize to 320x320
  crop=cv2.resize(crop,(320,320))
  tiles.append(crop)
 rows=[]
 for j in range(0,len(tiles),6):
  row=tiles[j:j+6]
  while len(row)<6: row.append(np.zeros_like(tiles[0]))
  rows.append(np.hstack(row))
 cv2.imwrite('/work/scratch/analysis/contact_obj.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py", line 45, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation minimum which has no identity

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
sel=[0,150,175,190,200,210,220,225,250,300,350,375,400,450,475,485,490,495,500,505,510,525,550,575,600,620,700]
with h5py.File('/work/inputs/masks.h5','r') as f:
 tiles=[]
 for i in sel:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
  ys,xs=np.where(f['object'][i])
  if len(xs)==0:
   print('empty',i); x0,y0,x1,y1=500,500,900,950
  else: x0,x1=max(0,xs.min()-80),min(1536,xs.max()+80); y0,y1=max(0,ys.min()-80),min(1152,ys.max()+80)
  crop=im[y0:y1,x0:x1].copy(); cv2.putText(crop,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
  crop=cv2.resize(crop,(320,320)); tiles.append(crop)
 rows=[]
 for j in range(0,len(tiles),6):
  row=tiles[j:j+6]
  while len(row)<6: row.append(np.zeros_like(tiles[0]))
  rows.append(np.hstack(row))
 cv2.imwrite('/work/scratch/analysis/contact_obj.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```
empty 495

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/score_candidates.py <<'PY'
import torch, numpy as np, trimesh, json, h5py, cv2, time
import nvdiffrast.torch as dr
M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(M.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(M.faces,dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz'); Tall=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device='cuda')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); W=np.array(cam['T_world_cam']); Winv=torch.tensor(np.linalg.inv(W),dtype=torch.float32,device='cuda')
fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
glctx=dr.RasterizeCudaContext()
center=torch.tensor(M.centroid,dtype=torch.float32,device='cuda')
def cand(T,name):
 R=T[:3,:3]; Rc=torch.eye(3,device='cuda'); c=torch.tensor(M.centroid,dtype=torch.float32,device='cuda')
 if name=='base': return T
 ax={'x':0,'y':1,'z':2}[name[-1]]
 Rc[ax,ax]=-1
 # post rotate around c
 out=T.clone(); out[:3,:3]=R@Rc; out[:3,3]=T[:3,3]+R@(c-Rc@c)
 return out
def render(T):
 Tcm=Winv@T
 vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 vc=(Tcm@vh.T).T[:,:3]
 clip=torch.stack([(2*fx/1536)*vc[:,0]+(2*cx/1536-1)*vc[:,2], (2*fy/1152)*vc[:,1]+(2*cy/1152-1)*vc[:,2], torch.zeros_like(vc[:,2]), vc[:,2]],1)[None]
 rast,_=dr.rasterize(glctx,clip,F,resolution=(1152,1536))
 return (rast[0,:,:,3]>0)
sel=[0,50,100,150,175,190,200,210,220,225,250,275,300,325,350,375,400,425,450 …[+652 chars]
```
<details><summary>output</summary>

```
0 gt 8610 base:8172/0.828/0.930/0.883 flipx:7944/0.842/0.953/0.879 flipy:7923/0.798/0.926/0.852 flipz:7690/0.598/0.793/0.708
50 gt 8427 base:8186/0.828/0.919/0.893 flipx:7953/0.844/0.943/0.890 flipy:7929/0.805/0.920/0.866 flipz:8131/0.570/0.739/0.713
100 gt 8375 base:7794/0.832/0.942/0.877 flipx:7568/0.845/0.965/0.872 flipy:7581/0.801/0.936/0.847 flipz:7776/0.567/0.751/0.697
150 gt 8191 base:7845/0.848/0.938/0.899 flipx:7616/0.864/0.962/0.894 flipy:7771/0.813/0.921/0.874 flipz:7453/0.609/0.794/0.723
175 gt 8924 base:8435/0.871/0.958/0.905 flipx:8242/0.884/0.977/0.902 flipy:8359/0.817/0.930/0.871 flipz:7690/0.660/0.859/0.740
190 gt 9652 base:8863/0.861/0.967/0.888 flipx:8739/0.866/0.977/0.884 flipy:8757/0.778/0.920/0.835 flipz:8366/0.703/0.889/0.771
200 gt 9413 base:8351/0.874/0.992/0.880 f …[+3521 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/overlay_masks.py <<'PY'
exec(open('/work/scratch/analysis/score_candidates.py').read().split('sel=')[0])
import h5py,cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4')
sel=[0,175,200,225,300,350,375,400,450,475,485,490,500,510,525,550,575,600,620,650,700]
tiles=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in sel:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
  gt=torch.tensor(f['object'][i],device='cuda'); per=torch.tensor(f['person'][i],device='cuda')
  m=render(Tall[i])&~per
  mn=m.cpu().numpy(); gn=gt.cpu().numpy()
  # bbox union expand
  ys,xs=np.where(mn|gn); x0,x1=max(0,xs.min()-30),min(1536,xs.max()+30);y0,y1=max(0,ys.min()-30),min(1152,ys.max()+30)
  crop=im[y0:y1,x0:x1].copy()
  # color contours not fill
  cnts,_=cv2.findContours(gn[y0:y1,x0:x1].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop,cnts,-1,(0,255,0),2)
  cnts,_=cv2.findContours(mn[y0:y1,x0:x1].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop,cnts,-1,(0,0,255),2)
  cv2.putText(crop,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,255),2)
  crop=cv2.resize(crop,(320,320));tiles.append(crop)
rows=[]
for j in range(0,len(tiles),6):
 row=tiles[j:j+6]
 while len(row)<6:row.append(np.zeros_like(tiles[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/analysis/overlay_masks.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95])
PY
python /work/scratch/analysis/overlay_m …[+8 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/hands.py <<'PY'
import numpy as np,trimesh
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);K=z['human_keypoints_world'].astype(float)
M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V0=M.vertices
# subsample verts and world
for i in [0,100,130,150,175,190,200,225,250,300,350,375,400,425,450,475,500,525,550,575,600,620,700]:
 V=(T[i,:3,:3]@V0.T).T+T[i,:3,3]
 hr=K[i,21:42]; hl=K[i,42:63]
 for name,h in [('R',hr),('L',hl)]:
  d=np.linalg.norm(h[:,None,:]-V[None,::5,:],axis=2)
  print(i,name,'handmean',np.round(h.mean(0),3),'mind',round(d.min(),3),'p10',round(np.percentile(d.min(1),10),3),'centdist',round(np.linalg.norm(h.mean(0)-T[i,:3,3]),3))
PY
python /work/scratch/analysis/hands.py"
```
<details><summary>output</summary>

```
0 R handmean [-0.074  0.501  2.613] mind 0.866 p10 0.868 centdist 1.075
0 L handmean [-0.557  0.474  2.794] mind 0.889 p10 0.89 centdist 1.103
100 R handmean [-0.263  0.455  3.564] mind 0.243 p10 0.248 centdist 0.555
100 L handmean [-0.223  0.407  4.066] mind 0.414 p10 0.429 centdist 0.669
130 R handmean [0.067 0.461 3.532] mind 0.052 p10 0.069 centdist 0.427
130 L handmean [-0.141  0.389  4.069] mind 0.39 p10 0.413 centdist 0.667
150 R handmean [0.062 0.503 3.591] mind 0.001 p10 0.003 centdist 0.365
150 L handmean [-0.158  0.365  4.141] mind 0.483 p10 0.5 centdist 0.745
175 R handmean [0.03  0.459 3.596] mind 0.002 p10 0.011 centdist 0.376
175 L handmean [-0.182  0.361  4.152] mind 0.471 p10 0.486 centdist 0.709
190 R handmean [-0.037  0.343  3.572] mind 0.013 p10 0.022 centdist 0.35
190  …[+2523 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/test_texture.py <<'PY'
import torch,numpy as np,trimesh,json,h5py,cv2
import torch.nn.functional as Fnn
import nvdiffrast.torch as dr
M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(M.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(M.faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(M.visual.uv,dtype=torch.float32,device='cuda')
# texture down 2048
from PIL import Image
im=M.visual.material.baseColorTexture
im2=im.resize((2048,round(im.height*2048/im.width)),Image.Resampling.LANCZOS)
tex=torch.tensor(np.array(im2),dtype=torch.float32,device='cuda').permute(2,0,1)[None]/255
z=np.load('/work/inputs/reconstruction.npz');Tall=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Winv=torch.tensor(np.linalg.inv(np.array(cam['T_world_cam'])),dtype=torch.float32,device='cuda')
ctx=dr.RasterizeCudaContext()
def render(i,crop):
 x0,y0,size=crop
 T=Winv@Tall[i]; vh=torch.cat([V,torch.ones_like(V[:,:1])],1); vc=(T@vh.T).T[:,:3]
 fx,fy,cx,cy=K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0
 clip=torch.stack([(2*fx/size)*vc[:,0]+(2*cx/size-1)*vc[:,2],(2*fy/size)*vc[:,1]+(2*cy/size-1)*vc[:,2],torch.zeros_like(vc[:,2]),vc[:,2]],1)[None]
 rast,_=dr.rasterize(ctx,clip,F,resolution=(size,size))
 uv,db=dr.interpolate(torch.cat([UV,torch.ones_like(UV[:,:1])],1)[None],rast,F,rast_db=None)
 uv=uv[0,:,:,:2]
 grid=t …[+836 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/1-uv\\[...,1\\]\\*2/uv[...,1]*2-1/' /work/scratch/analysis/test_texture.py > /work/scratch/analysis/test_texture2.py && python /work/scratch/analysis/test_texture2.py && mv /work/scratch/analysis/tex_000.jpg /work/scratch/analysis/tex2_000.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); uv=M.visual.uv; im=np.array(M.visual.material.baseColorTexture); h,w=im.shape[:2]
for conv in ['top','bottom']:
 x=np.clip((uv[:,0]*w).astype(int),0,w-1); y=np.clip(((uv[:,1] if conv=='top' else 1-uv[:,1])*h).astype(int),0,h-1)
 c=im[y,x]
 print(conv,'med',np.median(c,0),'mean',c.mean(0),'yellow%',((c[:,0]>120)&(c[:,1]>100)&(c[:,2]<120)).mean(),'black%',(c.mean(1)<30).mean(),'cyan%',((c[:,0]<100)&(c[:,1]>100)&(c[:,2]>100)).mean())
PY"
```
<details><summary>output</summary>

```
top med [0. 0. 0.] mean [64.49175518 47.6453475   3.47019529] yellow% 0.3222650177032869 black% 0.5948404698141405 cyan% 0.0
bottom med [158. 120.   2.] mean [153.61264907 113.04727229  10.3667439 ] yellow% 0.7785083100131396 black% 0.016374286602674048 cyan% 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/analysis/test_texture.py').read().split(\"cap=cv2\")[0])
rgb,mask=render(0,(578,580,320))
print('mask',mask.sum())
ys,xs=np.where(mask)
# reload uv from modified function? recreate quickly
PY"
```
<details><summary>output</summary>

```
mask 8172

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
for f in ['tex_000.jpg','tex2_000.jpg']:
 im=cv2.imread('/work/scratch/analysis/'+f)[:,640:960]
 # BGR pixels mask
 print(f,im.reshape(-1,3).mean(0),np.percentile(im.reshape(-1,3),[10,50,90],axis=0))
PY"
```
<details><summary>output</summary>

```
[ WARN:0@0.017] global loadsave.cpp:268 findDecoder imread_('/work/scratch/analysis/tex_000.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
TypeError: 'NoneType' object is not subscriptable

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ls /work/scratch/analysis/tex* | head && python - <<'PY'
import cv2,numpy as np,glob
for f in glob.glob('/work/scratch/analysis/tex*'):
 im=cv2.imread(f)
 if im is None: continue
 right=im[:,640:960]; p=right.reshape(-1,3); nz=p[p.sum(1)>10]
 print(f.split('/')[-1],'BGR med',np.median(nz,0),'mean',nz.mean(0))
PY"
```
<details><summary>output</summary>

```
texture.jpg BGR med [  1. 122. 158.] mean [  6.8876299  110.86196598 140.50703564]
tex_175.jpg BGR med [50. 29.  0.] mean [76.13116709 58.08093606 15.97391564]
tex_225.jpg BGR med [50. 33.  0.] mean [75.56709168 57.61961141 16.30085003]
tex_350.jpg BGR med [57. 39.  2.] mean [78.31838679 59.54755029 16.84774788]
tex_400.jpg BGR med [35. 15.  0.] mean [63.58441911 47.46416938 16.5092291 ]
tex_550.jpg BGR med [35. 16.  0.] mean [66.24048583 50.24331984 16.86356275]
tex_620.jpg BGR med [32. 14.  0.] mean [61.79369703 46.41260593 16.69623941]
tex_700.jpg BGR med [32. 14.  0.] mean [63.00501313 47.44258773 16.56409644]
tex2_000.jpg BGR med [49. 27.  0.] mean [75.41378215 57.59415688 15.89139409]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/test_texture.py';s=open(p).read();s=s.replace(\"cv2.imwrite(f'/work/scratch/analysis/tex_{i:03d}.jpg',np.hstack([img,black,rgb]))\",\"cv2.imwrite(f'/work/scratch/analysis/tex_{i:03d}.jpg',np.hstack([img,black,rgb[:,:,::-1]]))\")
open('/work/scratch/analysis/test_texture3.py','w').write(s)
PY
python /work/scratch/analysis/test_texture3.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,time
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
for fn in ['simplify_quadric_decimation','simplify_quadric_decimation']:
 try:
  t=time.time(); s=m.simplify_quadric_decimation(face_count=3000); print('ok',len(s.vertices),len(s.faces),time.time()-t); break
 except Exception as e: print('err',e)
PY"
```
<details><summary>output</summary>

```
ok 6962 3000 0.11080002784729004

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/opt_mask.py <<'PY'
import torch, numpy as np, trimesh, json, h5py, cv2, time, math
import nvdiffrast.torch as dr
# decimated silhouette mesh
Mf=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
M=Mf.simplify_quadric_decimation(face_count=5000)
V=torch.tensor(M.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(M.faces,dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz'); Tall=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Winv=torch.tensor(np.linalg.inv(np.array(cam['T_world_cam'])),dtype=torch.float32,device='cuda')
ctx=dr.RasterizeCudaContext()
def rodrigues(w):
 th=torch.linalg.norm(w)+1e-8; k=w/th
 Kx=torch.zeros(3,3,device=w.device); Kx[0,1]=-k[2];Kx[0,2]=k[1];Kx[1,0]=k[2];Kx[1,2]=-k[0];Kx[2,0]=-k[1];Kx[2,1]=k[0]
 return torch.eye(3,device=w.device)+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)

def make_crop(i,gmask):
 ys,xs=np.where(gmask)
 if len(xs)>20:
  cx,cy=xs.mean(),ys.mean(); sx=xs.max()-xs.min();sy=ys.max()-ys.min()
 else:
  # project baseline center
  T=Winv@Tall[i]; c=(T@torch.tensor([0,0,0,1.],device='cuda'))[:3]; u=K[0,0]*c[0]/c[2]+K[0,2];v=K[1,1]*c[1]/c[2]+K[1,2];cx,cy=u.item(),v.item();sx=sy=100
 size=int(max(160,min(512,max(sx,sy)*2.2+50)))
 size=(size+31)//32*32
 x0=int(np.clip(cx-size/2,0,1536-size));y0=int(np.clip(cy-size/2,0,1152-size));
 return x0,y …[+2066 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/opt_mask.py", line 69, in <module>
    t=time.time();b,T,c,w,dt=optimize(i,True,True);print('FRAME',i,'best',b,'crop',c,'wdeg',np.degrees(w),'dt',dt,'time',time.time()-t)
                             ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/opt_mask.py", line 42, in optimize
    crop=make_crop(i,h['object'][i]); x0,y0,size=crop
                     ~^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 407, in __getitem__
    return self._get(name)
           ^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py",  …[+406 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/crop=make_crop(i,h\\['object'\\]\\[i\\])/crop=make_crop(i,h['object'][i].astype(bool))/\" /work/scratch/analysis/opt_mask.py && python /work/scratch/analysis/opt_mask.py 0 175 225 350 400 450 510 550 620 700"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/opt_mask.py", line 69, in <module>
    t=time.time();b,T,c,w,dt=optimize(i,True,True);print('FRAME',i,'best',b,'crop',c,'wdeg',np.degrees(w),'dt',dt,'time',time.time()-t)
                             ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/opt_mask.py", line 42, in optimize
    crop=make_crop(i,h['object'][i].astype(bool)); x0,y0,size=crop
                     ~^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 407, in __getitem__
    return self._get(name)
           ^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_h …[+419 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n crop= /work/scratch/analysis/opt_mask.py && python - <<'PY'
import h5py
h=h5py.File('/work/inputs/masks.h5','r');print(h.keys(),h['object'][0].shape)
PY"
```
<details><summary>output</summary>

```
<KeysViewHDF5 ['object', 'person']> (1152, 1536)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/analysis/opt_mask.py | sed -n '35,48p'"
```
<details><summary>output</summary>

```
    35	def score(m,gt,person):
    36	 m=m&~person
    37	 inter=(m&gt).sum().item(); uni=(m|gt).sum().item()
    38	 return inter/max(1,uni)
    39	def optimize(i,steps=True,verbose=False):
    40	 with h5py.File('/work/inputs/masks.h5','r') as h:
    41	  gt=torch.tensor(h['object'][i],device='cuda');person=torch.tensor(h['person'][i],device='cuda')
    42	 crop=make_crop(i,h['object'][i].astype(bool)); x0,y0,size=crop
    43	 gt=gt[y0:y0+size,x0:x0+size];person=person[y0:y0+size,x0:x0+size]
    44	 T0=Tall[i];R0=T0[:3,:3].clone();t0=T0[:3,3].clone(); w=torch.zeros(3,device='cuda');dt=torch.zeros(3,device='cuda')
    45	 def mkT(w,dt): 
    46	  T=T0.clone();T[:3,:3]=rodrigues(w)@R0;T[:3,3]=t0+dt;return T
    47	 best=score(render(T0,crop),gt,person)
    48	 if verbose: print('init',best …[+2 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/opt_mask.py';s=open(p).read();s=s.replace(\"  gt=torch.tensor(h['object'][i],device='cuda');person=torch.tensor(h['person'][i],device='cuda')\\n crop=make_crop(i,h['object'][i].astype(bool)); x0,y0,size=crop\", \"  arr=h['object'][i].astype(bool)\\n  gt=torch.tensor(arr,device='cuda');person=torch.tensor(h['person'][i],device='cuda')\\n  crop=make_crop(i,arr); x0,y0,size=crop\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/opt_mask.py 0 175 225 350 400 450 510 550 620 700"
```
<details><summary>output</summary>

```
init 0.07962664208342936
8 0.04 0.09163117301715207 [0.13962634 0.13962634 0.        ] [0.   0.04 0.04]
4 0.02 0.09163117301715207 [0.13962634 0.13962634 0.        ] [0.   0.04 0.04]
2 0.01 0.09163117301715207 [0.13962634 0.13962634 0.        ] [0.   0.04 0.04]
1 0.005 0.09207190596911731 [0.13962634 0.13962634 0.        ] [-0.005  0.045  0.04 ]
0.5 0.0025 0.09207190596911731 [0.13962634 0.13962634 0.        ] [-0.005  0.045  0.04 ]
FRAME 0 best 0.09207190596911731 crop (514, 516, 448) wdeg [7.9999995 7.9999995 0.       ] dt [-0.005  0.045  0.04 ] time 0.49692201614379883
init 0.08096939915121733
8 0.04 0.09251640164572446 [0.13962634 0.13962634 0.        ] [0.   0.04 0.  ]
4 0.02 0.09251640164572446 [0.13962634 0.13962634 0.        ] [0.   0.04 0.  ]
2 0.01 0.09251640164572446 [0.13962634 …[+5008 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
for fc in [3000,5000,10000,20000,40000]:
 s=m.simplify_quadric_decimation(face_count=fc)
 print(fc,len(s.vertices),len(s.faces),s.extents,s.is_watertight,s.volume)
PY"
```
<details><summary>output</summary>

```
3000 6962 3000 [0.2895121  0.2820323  0.64582273] False 0.00020998724660688768
5000 10750 5000 [0.2895121  0.2820323  0.64582273] False 0.00043435390024862554
10000 18480 10000 [0.2895121  0.2820323  0.64582273] False 0.001083011780700369
20000 28159 20000 [0.2895121  0.2820323  0.64582273] False 0.0026559092037894815
40000 34259 40000 [0.2895121  0.2820323  0.64582273] False 0.005515144404183309

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/analysis/opt_mask.py').read().split(\"if __name__\")[0];exec(s)
with h5py.File('/work/inputs/masks.h5','r') as h:
 arr=h['object'][0].astype(bool);gt=torch.tensor(arr,device='cuda');per=torch.tensor(h['person'][0],device='cuda');crop=make_crop(0,arr);x0,y0,size=crop
 m=render(Tall[0],crop);print('mask sum',m.sum().item(),'gt',gt[y0:y0+size,x0:x0+size].sum().item(),'score',score(m,gt[y0:y0+size,x0:x0+size],per[y0:y0+size,x0:x0+size]))
 # full mesh render current earlier function absent; compare raw full via script? 
PY"
```
<details><summary>output</summary>

```
mask sum 759 gt 8610 score 0.07962664208342936

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed "s/M=Mf.simplify_quadric_decimation(face_count=5000)/M=Mf/" /work/scratch/analysis/opt_mask.py > /work/scratch/analysis/opt_mask_full.py && python /work/scratch/analysis/opt_mask_full.py 0 175 225 350 400 450 510 550 620 700'
```
<details><summary>output</summary>

```
init 0.8275073505390396
8 0.04 0.8641691313396868 [0.         0.13962634 0.        ] [0.   0.04 0.08]
4 0.02 0.8703601108033241 [0.         0.06981317 0.        ] [0.   0.04 0.06]
2 0.01 0.875 [-0.03490658  0.10471976  0.        ] [0.   0.04 0.06]
1 0.005 0.875 [-0.03490658  0.10471976  0.        ] [0.   0.04 0.06]
0.5 0.0025 0.875 [-0.03490658  0.10471976  0.        ] [0.   0.04 0.06]
FRAME 0 best 0.875 crop (514, 516, 448) wdeg [-1.9999999  6.         0.       ] dt [0.   0.04 0.06] time 1.0705838203430176
init 0.8705818965517241
8 0.04 0.8924231216316179 [-0.13962634  0.          0.        ] [ 0.    0.   -0.08]
4 0.02 0.8997763340078816 [-0.06981317  0.06981317  0.        ] [ 0.    0.   -0.06]
2 0.01 0.9042137271937446 [-0.06981317  0.03490658  0.        ] [-0.01  0.   -0.06]
1 0.005 0.9 …[+5063 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/tex_score.py <<'PY'
import torch,numpy as np,trimesh,json,h5py,cv2
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(M.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(M.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(M.visual.uv,dtype=torch.float32,device='cuda')
from PIL import Image
im=M.visual.material.baseColorTexture;im2=im.resize((2048,round(im.height*2048/im.width)),Image.Resampling.LANCZOS)
tex=torch.tensor(np.array(im2),dtype=torch.float32,device='cuda').permute(2,0,1)[None]/255
z=np.load('/work/inputs/reconstruction.npz');Tall=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Winv=torch.tensor(np.linalg.inv(np.array(cam['T_world_cam'])),dtype=torch.float32,device='cuda')
ctx=dr.RasterizeCudaContext()
def render(i,T,crop):
 x0,y0,size=crop;Tcm=Winv@T;vh=torch.cat([V,torch.ones_like(V[:,:1])],1);vc=(Tcm@vh.T).T[:,:3]
 fx,fy,cx,cy=K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0
 clip=torch.stack([(2*fx/size)*vc[:,0]+(2*cx/size-1)*vc[:,2],(2*fy/size)*vc[:,1]+(2*cy/size-1)*vc[:,2],torch.zeros_like(vc[:,2]),vc[:,2]],1)[None]
 rast,_=dr.rasterize(ctx,clip,F,resolution=(size,size))
 uv,_=dr.interpolate(torch.cat([UV,torch.ones_like(UV[:,:1])],1)[None],rast,F)
 uv=uv[0,:,:,:2];grid=torch.stack([uv[...,0]*2-1,1-uv[...,1]*2],-1) …[+2104 chars]
```
<details><summary>output</summary>

```
0 base grayNCC 0.1835 gradNCC -0.0851 valid 7497
0 flipx grayNCC 0.3153 gradNCC 0.0834 valid 7497
0 flipy grayNCC 0.1376 gradNCC -0.0482 valid 7497
0 flipz grayNCC -0.0980 gradNCC 0.0029 valid 7497
175 base grayNCC 0.1870 gradNCC -0.0132 valid 7800
175 flipx grayNCC 0.2424 gradNCC 0.0748 valid 7800
175 flipy grayNCC 0.1343 gradNCC -0.0414 valid 7800
175 flipz grayNCC -0.0551 gradNCC -0.0017 valid 7800
225 base grayNCC 0.1685 gradNCC 0.0010 valid 8163
225 flipx grayNCC 0.1631 gradNCC 0.0224 valid 8163
225 flipy grayNCC 0.0633 gradNCC -0.0478 valid 8163
225 flipz grayNCC -0.0502 gradNCC -0.0085 valid 8163
300 base grayNCC 0.0832 gradNCC -0.0039 valid 7646
300 flipx grayNCC 0.0983 gradNCC 0.0037 valid 7646
300 flipy grayNCC 0.0961 gradNCC 0.0570 valid 7646
300 flipz grayNCC -0.2574 gradNCC -0 …[+1465 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/compare_flips.py <<'PY'
exec(open('/work/scratch/analysis/tex_score.py').read().split(\"if __name__\")[0])
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,175,225,300,350,400,510,550,620,700]:
 with h5py.File('/work/inputs/masks.h5','r') as h:
  ys,xs=np.where(h['object'][i])
 if len(xs)==0:continue
 size=int(max(180,min(512,max(xs.max()-xs.min(),ys.max()-ys.min())*2.2+50)));size=(size+31)//32*32;x0=int(np.clip(xs.mean()-size/2,0,1536-size));y0=int(np.clip(ys.mean()-size/2,0,1152-size));crop=(x0,y0,size)
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();img=img[y0:y0+size,x0:x0+size]
 outs=[img]
 for n in ['base','flipx','flipy','flipz']:
  T=Tall[i].clone();c=torch.tensor(M.centroid,device='cuda')
  if n"'!='"'base':
   Rc=torch.eye(3,device='cuda');ax={'x':0,'y':1,'z':2}[n[-1]];Rc[ax,ax]=-1;R=T[:3,:3]@Rc;T[:3,:3]=R;T[:3,3]=T[:3,3]+Tall[i][:3,:3]@(c-Rc@c)
  rgb,mask=render(i,T,crop);rgb=rgb.detach().cpu().numpy()[:,:,::-1];rgb=(rgb*255).astype(np.uint8);rgb[~mask.cpu().numpy()]=0
  cv2.putText(rgb,n,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,255),2);outs.append(rgb)
 out=np.hstack(outs);cv2.imwrite(f'/work/scratch/analysis/flips_{i:03d}.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])
PY
python /work/scratch/analysis/compare_flips.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/compare_flips.py", line 16, in <module>
    cv2.putText(rgb,n,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,255),2);outs.append(rgb)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'putText'
> Overload resolution failed:
>  - Layout of the output array img is incompatible with cv::Mat
>  - Expected Ptr<cv::UMat> for argument 'img'


```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/rgb=(rgb\\*255).astype(np.uint8);/rgb=np.ascontiguousarray((rgb*255).astype(np.uint8));/\" /work/scratch/analysis/compare_flips.py && python /work/scratch/analysis/compare_flips.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/compare_flips.py", line 14, in <module>
    Rc=torch.eye(3,device='cuda');ax={'x':0,'y':1,'z':2}[n[-1]];Rc[ax,ax]=-1;R=T[:3,:3]@Rc;T[:3,:3]=R;T[:3,3]=T[:3,3]+Tall[i][:3,:3]@(c-Rc@c)
                                                                                                                                        ~~^~
RuntimeError: expected scalar type Float but found Double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/torch.tensor(M.centroid,device='cuda')/torch.tensor(M.centroid,dtype=torch.float32,device='cuda')/\" /work/scratch/analysis/compare_flips.py && python /work/scratch/analysis/compare_flips.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# make larger flip montage frame0
exec(open('/work/scratch/analysis/tex_score.py').read().split(\"if __name__\")[0])
import cv2,h5py,numpy as np
i=0
with h5py.File('/work/inputs/masks.h5','r') as h:
 ys,xs=np.where(h['object'][i])
x0,y0,x1,y1=xs.min()-40,ys.min()-40,xs.max()+40,ys.max()+40
size=640;crop=(x0,y0,x1-x0,y1-y0) # renderer square assumption so can't
# use square crop
w=max(x1-x0,y1-y0);x0=int(xs.mean()-w/2);y0=int(ys.mean()-w/2);crop=(x0,y0,w)
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();img=img[y0:y0+w,x0:x0+w]
outs=[cv2.resize(img,(500,500))]
for n in ['base','flipx','flipy']:
 T=Tall[i].clone();c=torch.tensor(M.centroid,dtype=torch.float32,device='cuda')
 if n"'!='"'base':
  Rc=torch.eye(3,device='cuda');ax={'x':0,'y':1,'z':2}[n[-1]];Rc[ax,ax]=-1;T[:3,:3]=T[:3,:3]@Rc;T[:3,3]=T[:3,3]+Tall[i][:3,:3]@(c-Rc@c)
 rgb,mask=render(i,T,crop);rgb=np.ascontiguousarray((rgb.detach().cpu().numpy()[:,:,::-1]*255).astype(np.uint8));rgb[~mask.cpu().numpy()]=0;rgb=cv2.resize(rgb,(500,500));cv2.putText(rgb,n,(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2);outs.append(rgb)
cv2.imwrite('/work/scratch/analysis/flips_big0.jpg',np.hstack(outs),[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/eval_opt_tex.py <<'PY'
exec(open('/work/scratch/analysis/opt_mask_full.py').read().split(\"if __name__\")[0])
# Import texscore without its main; avoid variable overwrite? names okay
s=open('/work/scratch/analysis/tex_score.py').read().split(\"if __name__\")[0]
# Execute only function defs, but duplicate T etc acceptable
ns={};exec(s,ns);texscore=ns['texscore']
for i in [0,175,225,350,400,450,510,550,620,700]:
 b,T,crop,w,dt=optimize(i,True,False)
 s1=texscore(i,Tall[i],crop);s2=texscore(i,T,crop)
 print(i,'mask %.3f->%.3f'%(b,score(render(T,crop),torch.tensor(__import__('h5py').File('/work/inputs/masks.h5','r')['object'][i,crop[1]:crop[1]+crop[2],crop[0]:crop[0]+crop[2]],device='cuda'),torch.tensor(__import__('h5py').File('/work/inputs/masks.h5','r')['person'][i,crop[1]:crop[1]+crop[2],crop[0]:crop[0]+crop[2]],device='cuda'))),'tex gray %.3f->%.3f grad %.3f->%.3f'%(s1[0],s2[0],s1[1],s2[1]),'w',np.round(np.degrees(w),1),'dt',np.round(dt,3),flush=True)
PY
python /work/scratch/analysis/eval_opt_tex.py"
```
<details><summary>output</summary>

```
0 mask 0.875->0.875 tex gray 0.183->-0.025 grad -0.085->0.014 w [-2.  6.  0.] dt [0.   0.04 0.06]
175 mask 0.918->0.918 tex gray 0.187->0.104 grad -0.013->0.010 w [-4.5  1.5  1. ] dt [-0.005  0.005 -0.055]
225 mask 0.925->0.925 tex gray 0.169->0.400 grad 0.001->0.176 w [-3.5 -5.   0. ] dt [ 0.     0.005 -0.1  ]
350 mask 0.943->0.943 tex gray 0.269->0.262 grad 0.054->0.036 w [-9.  7.  0.] dt [-0.002  0.    -0.02 ]
400 mask 0.753->0.753 tex gray 0.017->0.022 grad 0.019->0.017 w [ -9. -10.   2.] dt [0.    0.042 0.012]
450 mask 0.862->0.862 tex gray 0.184->0.163 grad 0.025->-0.059 w [-8.5  0.   1. ] dt [0.    0.027 0.037]
510 mask 0.932->0.932 tex gray 0.190->0.201 grad -0.002->0.038 w [ 1.5  0.  -1. ] dt [ 0.    -0.005 -0.065]
550 mask 0.824->0.824 tex gray -0.085->0.120 grad 0.020->0.041 w [ …[+245 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);C=W[:3,3];Rc=W[:3,:3]
n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
for uv in [(739,821),(514,904),(515,800),(740,800)]:
 u,v=uv;ray=Rc@np.array([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1]);lam=-(n@C+d)/(n@ray);p=C+lam*ray;print(uv,'ray',ray,'lam',lam,'p',p,'depthcam',((p-C)@Rc[:,2]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);C=W[:3,3];Rc=W[:3,:3]
n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
for uv in [(739,821),(514,904),(515,800),(740,800)]:
 u,v=uv;ray=Rc@np.array([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1]);lam=-(n@C+d)/(n@ray);p=C+lam*ray;print(uv,'p',np.round(p,3),'depthcam',round((p-C)@Rc[:,2],3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json,numpy as np
x=json.load(open('/work/inputs/ground_plane.json'));print(x, np.array(x['plane']).shape)
PY"
```
<details><summary>output</summary>

```
{'plane': [-0.017496757389992512, -0.9984323698095767, -0.053166402899466475, 1.3840047702699394], 'n_inliers': 484707, 'n_candidates': 925088, 'n_total_points': 3132263, 'foot_plane_dist_stats': {'mean': 0.0457162582778017, 'std': 0.021601731224446783, 'min': 0.015949199565913785, 'max': 0.17807463747479102, 'median': 0.04070534346535981}} (4,)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);C=W[:3,3];Rc=W[:3,:3]
a=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=a[:3];d=a[3]
for uv in [(739,821),(514,904),(515,800),(740,800)]:
 u,v=uv;ray=Rc@np.array([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1]);lam=-(n@C+d)/(n@ray);p=C+lam*ray;print(uv,'p',np.round(p,3),'depthcam',round((p-C)@Rc[:,2],3))
PY"
```
<details><summary>output</summary>

```
(739, 821) p [0.067 1.199 3.492] depthcam 3.468
(514, 904) p [-0.701  1.247  2.84 ] depthcam 2.799
(515, 800) p [-0.952  1.201  3.783] depthcam 3.733
(740, 800) p [0.072 1.187 3.724] depthcam 3.699

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);W=np.array(c['T_world_cam']);C=W[:3,3];Rc=W[:3,:3]
a=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=a[:3];d=a[3]
u,v=739,821;q=np.array([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1]);ray=Rc@q
print('C',C,'ray',ray,'nray',n@ray,'nc',n@C,'d',d,'nC+d',n@C+d)
lam=-(n@C+d)/(n@ray);p=C+lam*ray;print(lam,p,n@p+d)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=M.vertices
a=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=a[:3];d=a[3]
for i in [0,25,50,75,100,125,140,150,620,640,660,680,700,714]:
 P=(T[i,:3,:3]@V.T).T+T[i,:3,3]; dist=P@n+d
 print(i,'floor min/med/max',dist.min(),np.percentile(dist,[5,50]),dist.max(),'t',T[i,:3,3])
# static averages
for lo,hi in [(0,125),(0,140),(630,715),(620,715)]:
 TT=T[lo:hi];print('phase',lo,hi,'trans std',TT[:,:3,3].std(0),'mean',TT[:,:3,3].mean(0),'med',np.median(TT[:,:3,3],0))
PY"
```
<details><summary>output</summary>

```
0 floor min/med/max 0.022103640215273224 [0.06514921 0.3450613 ] 0.6751512205805987 t [0.06953237 0.85958636 3.6158514 ]
25 floor min/med/max 0.028933353651558358 [0.07197892 0.35189101] 0.6819809340168843 t [0.06946942 0.85394162 3.59341764]
50 floor min/med/max 0.022869167664159473 [0.06591473 0.34582683] 0.6759167480294852 t [0.0695253  0.85895365 3.6133368 ]
75 floor min/med/max 0.009396953296773702 [0.05244252 0.33235461] 0.6624445336620995 t [0.06964947 0.8700884  3.65758944]
100 floor min/med/max -0.0018676712771357185 [0.0411779  0.32108999] 0.6511799090881899 t [0.06975328 0.87939858 3.69459057]
125 floor min/med/max 0.011047140433059655 [0.05409271 0.3340048 ] 0.6640947207983853 t [0.06963426 0.86872452 3.65216899]
140 floor min/med/max 0.014899561801267502 [0.05794513 0.33785722 …[+1519 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);R=T[:,:3,:3];t=T[:,:3,3]
a=np.degrees(np.arccos(np.clip((np.trace(R[1:]@R[:-1].transpose(0,2,1),axis1=1,axis2=2)-1)/2,-1,1)));d=np.linalg.norm(t[1:]-t[:-1],axis=1)
for i in range(120,220):print(i+1,round(a[i],2),round(d[i],4),np.round(t[i+1],3))
print('end')
for i in range(605,650):print(i+1,round(a[i],2),round(d[i],4))
PY"
```
<details><summary>output</summary>

```
121 0.09 0.0013 [0.07  0.869 3.652]
122 0.09 0.0011 [0.07  0.868 3.651]
123 0.09 0.0 [0.07  0.868 3.651]
124 0.09 0.0011 [0.07  0.869 3.652]
125 0.09 0.0 [0.07  0.869 3.652]
126 0.09 0.0 [0.07  0.869 3.652]
127 0.09 0.0014 [0.07  0.869 3.654]
128 0.09 0.0029 [0.07  0.87  3.656]
129 0.09 0.0 [0.07  0.87  3.656]
130 0.09 0.0 [0.07  0.87  3.656]
131 0.09 0.0 [0.07  0.87  3.656]
132 0.09 0.0029 [0.07  0.869 3.654]
133 0.09 0.0083 [0.07  0.867 3.646]
134 0.09 0.0 [0.07  0.867 3.646]
135 0.09 0.0023 [0.07  0.866 3.643]
136 0.09 0.003 [0.07  0.866 3.64 ]
137 0.09 0.0009 [0.07  0.866 3.64 ]
138 0.09 0.0 [0.07  0.866 3.64 ]
139 0.09 0.0 [0.07  0.866 3.64 ]
140 0.09 0.0 [0.07  0.866 3.64 ]
141 0.09 0.0 [0.07  0.866 3.64 ]
142 0.09 0.0008 [0.07  0.865 3.639]
143 0.09 0.0 [0.07  0.865 3.639]
144 0.09  …[+3534 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# evaluate hand min distance for opt params known computed quickly by imported funcs
exec(open('/work/scratch/analysis/opt_mask_full.py').read().split(\"if __name__\")[0])
for i in [175,225,350,400,450,510,550,620]:
 b,T,c,w,dt=optimize(i,True,False)
 Kp=torch.tensor(z['human_keypoints_world'][i],dtype=torch.float32,device='cuda')
 def mind(T):
  P=(T[:3,:3]@V.T).T+T[:3,3]
  return torch.cdist(Kp[21:42],P[::5]).min().item()
 print(i,'mind base/opt',mind(Tall[i]),mind(T),'wdeg',np.round(np.degrees(w),1),'dt',np.round(dt,3))
PY"
```
<details><summary>output</summary>

```
175 mind base/opt 0.001953125 0.0009765625 wdeg [-4.5  1.5  1. ] dt [-0.005  0.005 -0.055]
225 mind base/opt 0.014385663904249668 0.0009765625 wdeg [-3.5 -5.   0. ] dt [ 0.     0.005 -0.1  ]
350 mind base/opt 0.0033829116728156805 0.0026744266506284475 wdeg [-9.  7.  0.] dt [-0.002  0.    -0.02 ]
400 mind base/opt 0.002183660166338086 0.015223102644085884 wdeg [ -9. -10.   2.] dt [0.    0.042 0.012]
450 mind base/opt 0.013601941987872124 0.05623507872223854 wdeg [-8.5  0.   1. ] dt [0.    0.027 0.037]
510 mind base/opt 0.09684723615646362 0.0358944833278656 wdeg [ 1.5  0.  -1. ] dt [ 0.    -0.005 -0.065]
550 mind base/opt 0.001953125 0.017277155071496964 wdeg [ 0.5  4.5 -2. ] dt [ 0.    0.03 -0.04]
620 mind base/opt 0.06644214689731598 0.109047532081604 wdeg [ 1.  6. -2.] dt [0.    0.038 0 …[+6 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);R=Rot.from_matrix(T[:,:3,:3]);t=T[:,:3,3]
for i in range(470,521):
 print(i,'quat',np.round(R[i].as_quat(),3),'t',np.round(t[i],3),'area',end=' ')
 import h5py
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(470,521): print(i,int(f['object'][i].sum()),'stepdeg',np.degrees((R[i-1].inv()*R[i]).magnitude()) if i>0 else 0)
PY"
```
<details><summary>output</summary>

```
470 quat [ 0.574 -0.347  0.234  0.704] t [0.384 0.752 3.51 ] area 471 quat [ 0.59  -0.336  0.221  0.7  ] t [0.364 0.743 3.509] area 472 quat [ 0.596 -0.327  0.214  0.702] t [0.35  0.749 3.517] area 473 quat [ 0.604 -0.326  0.216  0.695] t [0.334 0.747 3.524] area 474 quat [ 0.607 -0.319  0.224  0.693] t [0.314 0.747 3.517] area 475 quat [ 0.609 -0.319  0.24   0.685] t [0.292 0.745 3.521] area 476 quat [ 0.612 -0.311  0.249  0.684] t [0.267 0.747 3.55 ] area 477 quat [ 0.612 -0.316  0.265  0.675] t [0.249 0.751 3.615] area 478 quat [ 0.607 -0.33   0.28   0.667] t [0.228 0.753 3.623] area 479 quat [ 0.61  -0.326  0.281  0.665] t [0.213 0.757 3.604] area 480 quat [ 0.605 -0.343  0.297  0.654] t [0.183 0.776 3.672] area 481 quat [ 0.606 -0.341  0.295  0.656] t [0.16  0.775 3.676] area 482 quat …[+4471 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);K=z['human_keypoints_world'].astype(float);H=K[:,21:42]
def kaba(A,B):
 ca=A.mean(0);cb=B.mean(0);U,S,Vt=np.linalg.svd((A-ca).T@(B-cb));R=Vt.T@U.T
 if np.linalg.det(R)<0: Vt[-1]*=-1;R=Vt.T@U.T
 return R
for i in [0,100,130,150,164,175,200,225,250,300,350,375,400,425,450,470,480,485,490,495,500,505,510,525,550,575,600,620]:
 R=kaba(H[470],H[i]);ang=np.degrees(Rot.from_matrix(R).magnitude());Ro=T[i,:3,:3]@T[470,:3,:3].T;ango=np.degrees(Rot.from_matrix(Ro).magnitude())
 print(i,'hand vs470',round(ang,1),'obj',round(ango,1),'diff rel?',np.round(Ro@R.T,2).flatten()[:3])
PY"
```
<details><summary>output</summary>

```
0 hand vs470 58.0 obj 84.2 diff rel? [0.87 0.16 0.47]
100 hand vs470 132.6 obj 84.2 diff rel? [ 0.65  0.28 -0.7 ]
130 hand vs470 132.3 obj 84.2 diff rel? [ 0.67 -0.06 -0.74]
150 hand vs470 133.9 obj 84.2 diff rel? [ 0.66  0.01 -0.75]
164 hand vs470 133.4 obj 83.9 diff rel? [ 0.65  0.   -0.76]
175 hand vs470 136.9 obj 92.5 diff rel? [ 0.72 -0.03 -0.69]
200 hand vs470 161.1 obj 131.2 diff rel? [ 0.79 -0.39 -0.47]
225 hand vs470 158.6 obj 177.5 diff rel? [ 0.82 -0.51 -0.26]
250 hand vs470 152.7 obj 171.2 diff rel? [ 0.9  -0.33 -0.29]
300 hand vs470 147.5 obj 155.9 diff rel? [ 0.99  0.04 -0.15]
350 hand vs470 156.8 obj 148.1 diff rel? [ 0.99 -0.03 -0.15]
375 hand vs470 99.5 obj 79.5 diff rel? [ 0.94  0.12 -0.31]
400 hand vs470 79.6 obj 65.5 diff rel? [ 0.97  0.   -0.24]
425 hand vs470 22.5 obj …[+859 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);M=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=M.vertices
a=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=a[:3];d=a[3]
def avg(TT):
 q=Rot.from_matrix(TT[:,:3,:3]).as_quat(); ref=q[0];
 for i in range(len(q)):
  if q[i]@ref<0:q[i]*=-1
 qm=q.mean(0);qm/=np.linalg.norm(qm);R=Rot.from_quat(qm).as_matrix();t=np.median(TT[:,:3,3],0)
 X=np.eye(4);X[:3,:3]=R;X[:3,3]=t;return X
for lo,hi in [(0,164),(0,130),(630,715),(622,715)]:
 X=avg(T[lo:hi]);P=(X[:3,:3]@V.T).T+X[:3,3];dist=P@n+d
 print(lo,hi,'T',X,'floor min',dist.min())
PY"
```
<details><summary>output</summary>

```
0 164 T [[ 0.82576641  0.56368065 -0.01933826  0.06959722]
 [-0.00870765 -0.02154155 -0.99973003  0.86540285]
 [-0.56394505  0.82571187 -0.01287995  3.63896775]
 [ 0.          0.          0.          1.        ]] floor min 0.015065889709444225
0 130 T [[ 0.82576641  0.56368065 -0.01933826  0.06961952]
 [-0.00870765 -0.02154155 -0.99973003  0.86740288]
 [-0.56394505  0.82571187 -0.01287995  3.64691627]
 [ 0.          0.          0.          1.        ]] floor min 0.012646006897207851
630 715 T [[-0.98823643  0.15271408 -0.00819512 -0.73326737]
 [ 0.01407672  0.03747252 -0.99919851  0.92285573]
 [-0.15228459 -0.98755973 -0.03918142  2.97645283]
 [ 0.          0.          0.          1.        ]] floor min 0.01055302042597095
622 715 T [[-0.98823643  0.15271408 -0.00819512 -0.73333782]
 [ 0.0 …[+178 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 a=np.array([f['object'][i].sum() for i in range(715)])
low=np.where(a<300)[0]
print('low<300',low)
print('low<1000',np.where(a<1000)[0])
# contiguous runs
for th in [100,300,1000,2000]:
 idx=np.where(a<th)[0];runs=[]
 if len(idx):
  st=pr=idx[0]
  for x in idx[1:]:
   if x>pr+1:runs.append((st,pr));st=x
   pr=x
  runs.append((st,pr))
 print(th,runs)
PY"
```
<details><summary>output</summary>

```
low<300 [491 492 493 494 495 496 497]
low<1000 [490 491 492 493 494 495 496 497 498]
100 [(491, 496)]
300 [(491, 497)]
1000 [(490, 498)]
2000 [(488, 500)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/static_pool_test.py <<'PY'
exec(open('/work/scratch/analysis/opt_mask_full.py').read().split(\"if __name__\")[0])
from scipy.spatial.transform import Rotation as Rot

def avg(TT):
 q=Rot.from_matrix(TT[:,:3,:3]).as_quat();ref=q[0].copy()
 for j in range(len(q)):
  if q[j]@ref<0:q[j]*=-1
 qm=q.mean(0);qm/=np.linalg.norm(qm);X=np.eye(4);X[:3,:3]=Rot.from_quat(qm).as_matrix();X[:3,3]=np.median(TT[:,:3,3],0);return torch.tensor(X,dtype=torch.float32,device='cuda')

def optimize_g(T0,gmask,person=None,verbose=False):
 if person is None:person=np.zeros_like(gmask)
 ys,xs=np.where(gmask); size=int(max(160,min(512,max(xs.max()-xs.min(),ys.max()-ys.min())*2.2+50)));size=(size+31)//32*32;x0=int(np.clip(xs.mean()-size/2,0,1536-size));y0=int(np.clip(ys.mean()-size/2,0,1152-size));crop=(x0,y0,size)
 gt=torch.tensor(gmask[y0:y0+size,x0:x0+size],device='cuda');per=torch.tensor(person[y0:y0+size,x0:x0+size],device='cuda')
 R0=T0[:3,:3].clone();t0=T0[:3,3].clone();w=torch.zeros(3,device='cuda');dt=torch.zeros(3,device='cuda')
 def mkT(w,dt):
  T=T0.clone();T[:3,:3]=rodrigues(w)@R0;T[:3,3]=t0+dt;return T
 best=score(render(T0,crop),gt,per)
 for rotstep,trstep in [(6,.04),(3,.02),(1.5,.01),(.75,.005)]:
  for rounds in range(2):
   improved=False
   for k in range(6):
    bv=best;bp=None
    for s in [-1,1]:
     ww,dd=w.clone(),dt.clone()
     if k<3:ww[k]+=s*math.radians(rotstep)
     else:dd[k-3]+=s*trstep
     sc=score(render(mkT(ww,dd),crop),gt …[+1039 chars]
```
<details><summary>output</summary>

```
opt 0.8740015749803127 [0.   5.25 0.  ] [0.   0.04 0.06]
phase 0 128 mean [0.83152608 0.87497026] med [0.83091792 0.87464951]
opt 0.7394375583100093 [ 3.   4.5 -1.5] [0.   0.04 0.01]
phase 650 715 mean [0.60117663 0.73230803] med [0.60644082 0.73819055]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/edge_score.py <<'PY'
exec(open('/work/scratch/analysis/tex_score.py').read().split(\"if __name__\")[0])
import cv2,numpy as np,h5py,torch.nn.functional as Fn
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def sob(x):
 kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device='cuda',dtype=torch.float32)[None,None]/8
 ky=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],device='cuda',dtype=torch.float32)[None,None]/8
 return Fn.conv2d(x[None,None],kx,padding=1)[0,0],Fn.conv2d(x[None,None],ky,padding=1)[0,0]
def edgescore(i,T,crop):
 rgb,mask=render(i,T,crop);x0,y0,s=crop;cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();img=img[y0:y0+s,x0:x0+s]
 with h5py.File('/work/inputs/masks.h5','r') as h:
  gt=torch.tensor(h['object'][i,y0:y0+s,x0:x0+s],device='cuda');per=torch.tensor(h['person'][i,y0:y0+s,x0:x0+s],device='cuda')
 tgt=torch.tensor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),device='cuda')/255.
 valid=gt&~per;valid=~Fn.max_pool2d((~valid)[None,None].float(),7,1,3)[0,0].bool()
 a=tgt@torch.tensor([.299,.587,.114],device='cuda');b=rgb@torch.tensor([.299,.587,.114],device='cuda')
 ax,ay=sob(a);bx,by=sob(b)
 m=valid&(valid.roll(1,0))&(valid.roll(-1,0))&(valid.roll(1,1))&(valid.roll(-1,1))
 # corr concatenated gradients
 A=torch.cat([ax[m],ay[m]]);B=torch.cat([bx[m],by[m]]);A=A-A.mean();B=B-B.mean();gc=((A*B).sum()/(A.square().sum()*B.square().sum()).sqrt()).item()
 # magnitude maps nonnegative minus mean
 am=(ax.square()+ay.square()).sqrt();bm=(bx.s …[+892 chars]
```
<details><summary>output</summary>

```
0 base (-0.010193941183388233, 0.17304639518260956)
0 flipx (0.20219725370407104, 0.1764095425605774)
0 flipy (0.000363547180313617, 0.14388523995876312)
0 flipz (0.006673799362033606, 0.06474445760250092)
175 base (0.031764399260282516, 0.22989866137504578)
175 flipx (0.11799129843711853, 0.1990213692188263)
175 flipy (-0.018615571781992912, 0.1745544970035553)
175 flipz (-0.019564323127269745, 0.06311088055372238)
225 base (0.04559430107474327, 0.2609165906906128)
225 flipx (0.06502646207809448, 0.16443650424480438)
225 flipy (-0.05435127019882202, 0.1338372677564621)
225 flipz (-0.004580445121973753, 0.14996238052845)
300 base (0.014564587734639645, 0.04519568011164665)
300 flipx (-0.0052622295916080475, 0.055133406072854996)
300 flipy (0.07322516292333603, 0.1256355345249176)
300 flipz …[+1559 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
im=cv2.imread('/work/inputs/video.mp4')
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,img=cap.read()
crop=img[620:850,670:810];crop=cv2.resize(crop,(560,920),interpolation=cv2.INTER_CUBIC);cv2.imwrite('/work/scratch/analysis/actual0_zoom.jpg',crop)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/flip_opt_texture.py <<'PY'
exec(open('/work/scratch/analysis/edge_score.py').read().split('for i in [')[0])
# generic optimize from T0 on target frame
def optimize(i,T0):
 with h5py.File('/work/inputs/masks.h5','r') as h: arr=h['object'][i].astype(bool);person=h['person'][i].astype(bool)
 ys,xs=np.where(arr);size=int(max(180,min(512,max(xs.max()-xs.min(),ys.max()-ys.min())*2.2+50)));size=(size+31)//32*32;x0=int(np.clip(xs.mean()-size/2,0,1536-size));y0=int(np.clip(ys.mean()-size/2,0,1152-size));crop=(x0,y0,size)
 gt=torch.tensor(arr[y0:y0+size,x0:x0+size],device='cuda');per=torch.tensor(person[y0:y0+size,x0:x0+size],device='cuda')
 # score local helper with full render but score_candidates variables conflict? render here returns rgb,mask
 def scr(T):
  rgb,m=render(i,T,crop);m=m&~per;inter=(m&gt).sum().item();return inter/max(1,(m|gt).sum().item())
 R0=T0[:3,:3].clone();t0=T0[:3,3].clone();w=torch.zeros(3,device='cuda');dt=torch.zeros(3,device='cuda')
 def mkT(w,dt):
  T=T0.clone();T[:3,:3]=rodrigues(w)@R0;T[:3,3]=t0+dt;return T
 R=torch.zeros(3,3);R[0,1]=1;R[1,0]=-1 # placeholder
 best=scr(T0)
 for rs,ts in [(6,.04),(3,.02),(1.5,.01),(.75,.005)]:
  for rr in range(2):
   imp=False
   for k in range(6):
    bv=best;bp=None
    for s in [-1,1]:
     ww,dd=w.clone(),dt.clone()
     if k<3:ww[k]+=s*math.radians(rs)
     else:dd[k-3]+=s*ts
     sc=scr(mkT(ww,dd))
     if sc>bv+1e-5:bv=sc;bp=(ww,dd)
    if bp:w,dt=bp;best=bv;imp=True
 …[+670 chars]
```
<details><summary>output</summary>

```
FRAME 0
Traceback (most recent call last):
  File "/work/scratch/analysis/flip_opt_texture.py", line 35, in <module>
    b,T,crop,w,dt=optimize(i,T)
                  ^^^^^^^^^^^^^
  File "/work/scratch/analysis/flip_opt_texture.py", line 22, in optimize
    if k<3:ww[k]+=s*math.radians(rs)
                    ^^^^
NameError: name 'math' is not defined

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i '1i import math' /work/scratch/analysis/flip_opt_texture.py && python /work/scratch/analysis/flip_opt_texture.py"
```
<details><summary>output</summary>

```
FRAME 0
Traceback (most recent call last):
  File "/work/scratch/analysis/flip_opt_texture.py", line 36, in <module>
    b,T,crop,w,dt=optimize(i,T)
                  ^^^^^^^^^^^^^
  File "/work/scratch/analysis/flip_opt_texture.py", line 25, in optimize
    sc=scr(mkT(ww,dd))
           ^^^^^^^^^^
  File "/work/scratch/analysis/flip_opt_texture.py", line 13, in mkT
    T=T0.clone();T[:3,:3]=rodrigues(w)@R0;T[:3,3]=t0+dt;return T
                          ^^^^^^^^^
NameError: name 'rodrigues' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '2i from opt_mask_full import rodrigues' /work/scratch/analysis/flip_opt_texture.py && PYTHONPATH=/work/scratch/analysis python /work/scratch/analysis/flip_opt_texture.py"
```
<details><summary>output</summary>

```
FRAME 0
base iou 0.875 edge [-0.027  0.282] gray -0.035 w [-2.2  6.   0. ] dt [0.    0.04  0.055]
flipx iou 0.876 edge [0.073 0.178] gray 0.158 w [-1.5  4.5  0. ] dt [ 0.     0.005 -0.05 ]
flipy iou 0.887 edge [-0.016  0.278] gray -0.03 w [-3.  6.  0.] dt [-0.01  0.04  0.05]
flipz iou 0.712 edge [0.034 0.19 ] gray 0.069 w [-21.7   7.5 -19.5] dt [-0.005  0.01  -0.02 ]
FRAME 175
base iou 0.915 edge [0.052 0.221] gray 0.085 w [-4.5  2.2  1.5] dt [-0.005  0.005 -0.055]
flipx iou 0.917 edge [0.053 0.159] gray 0.108 w [-4.5  2.2  0. ] dt [ 0.     0.    -0.075]
flipy iou 0.917 edge [0.015 0.245] gray 0.108 w [-5.2  2.2  0.8] dt [-0.015  0.    -0.075]
flipz iou 0.797 edge [-0.004  0.206] gray 0.127 w [-19.5   1.5 -20.2] dt [-0.01 -0.01 -0.09]
FRAME 300
base iou 0.851 edge [0.103 0.188] gray 0.086  …[+699 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/save_flip_opts.py <<'PY'
exec(open('/work/scratch/analysis/flip_opt_texture.py').read().split('for i in [')[0])
import cv2
i=0
with h5py.File('/work/inputs/masks.h5','r') as h:
 ys,xs=np.where(h['object'][i])
w=520;x0=int(xs.mean()-w/2);y0=int(ys.mean()-w/2);crop=(x0,y0,w)
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();img=img[y0:y0+w,x0:x0+w];img=cv2.resize(img,(600,600));outs=[img]
for n in ['base','flipx','flipy']:
 T=Tall[i].clone();c=torch.tensor(M.centroid,dtype=torch.float32,device='cuda')
 if n"'!='"'base':
  Rc=torch.eye(3,device='cuda');ax={'x':0,'y':1,'z':2}[n[-1]];Rc[ax,ax]=-1;T[:3,:3]=T[:3,:3]@Rc;T[:3,3]=T[:3,3]+Tall[i][:3,:3]@(c-Rc@c)
 b,T,crop,w,dt=optimize(i,T)
 rgb,mask=render(i,T,crop);rgb=np.ascontiguousarray((rgb.detach().cpu().numpy()[:,:,::-1]*255).astype(np.uint8));rgb[~mask.cpu().numpy()]=0;rgb=cv2.resize(rgb,(600,600));cv2.putText(rgb,n,(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2);outs.append(rgb)
cv2.imwrite('/work/scratch/analysis/flip_opts0.jpg',np.hstack(outs),[cv2.IMWRITE_JPEG_QUALITY,95])
PY
PYTHONPATH=/work/scratch/analysis python /work/scratch/analysis/save_flip_opts.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/save_flip_opts.py", line 1, in <module>
    exec(open('/work/scratch/analysis/flip_opt_texture.py').read().split('for i in [')[0])
  File "<string>", line 3
    exec(open('/work/scratch/analysis/edge_score.py').read().split('
                                                                   ^
SyntaxError: unterminated string literal (detected at line 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/save_flip_opts.py <<'PY'
from flip_opt_texture import *
import cv2
i=0
with h5py.File('/work/inputs/masks.h5','r') as h:
 ys,xs=np.where(h['object'][i])
w=520;x0=int(xs.mean()-w/2);y0=int(ys.mean()-w/2);crop=(x0,y0,w)
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();img=img[y0:y0+w,x0:x0+w];img=cv2.resize(img,(600,600));outs=[img]
for n in ['base','flipx','flipy']:
 T=Tall[i].clone();c=torch.tensor(M.centroid,dtype=torch.float32,device='cuda')
 if n"'!='"'base':
  Rc=torch.eye(3,device='cuda');ax={'x':0,'y':1,'z':2}[n[-1]];Rc[ax,ax]=-1;T[:3,:3]=T[:3,:3]@Rc;T[:3,3]=T[:3,3]+Tall[i][:3,:3]@(c-Rc@c)
 b,T,crop,w,dt=optimize(i,T)
 rgb,mask=render(i,T,crop);rgb=np.ascontiguousarray((rgb.detach().cpu().numpy()[:,:,::-1]*255).astype(np.uint8));rgb[~mask.cpu().numpy()]=0;rgb=cv2.resize(rgb,(600,600));cv2.putText(rgb,n,(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2);outs.append(rgb)
cv2.imwrite('/work/scratch/analysis/flip_opts0.jpg',np.hstack(outs),[cv2.IMWRITE_JPEG_QUALITY,95])
PY
PYTHONPATH=/work/scratch/analysis python /work/scratch/analysis/save_flip_opts.py"
```
<details><summary>output</summary>

```
FRAME 0
base iou 0.875 edge [-0.027  0.282] gray -0.035 w [-2.2  6.   0. ] dt [0.    0.04  0.055]
flipx iou 0.876 edge [0.073 0.178] gray 0.158 w [-1.5  4.5  0. ] dt [ 0.     0.005 -0.05 ]
flipy iou 0.887 edge [-0.016  0.278] gray -0.03 w [-3.  6.  0.] dt [-0.01  0.04  0.05]
flipz iou 0.712 edge [0.034 0.19 ] gray 0.069 w [-21.7   7.5 -19.5] dt [-0.005  0.01  -0.02 ]
FRAME 175
base iou 0.915 edge [0.052 0.221] gray 0.085 w [-4.5  2.2  1.5] dt [-0.005  0.005 -0.055]
flipx iou 0.917 edge [0.053 0.159] gray 0.108 w [-4.5  2.2  0. ] dt [ 0.     0.    -0.075]
flipy iou 0.917 edge [0.015 0.245] gray 0.108 w [-5.2  2.2  0.8] dt [-0.015  0.    -0.075]
flipz iou 0.797 edge [-0.004  0.206] gray 0.127 w [-19.5   1.5 -20.2] dt [-0.01 -0.01 -0.09]
FRAME 300
base iou 0.851 edge [0.103 0.188] gray 0.086  …[+699 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs && cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally coherent world-space object trajectory.

The released reconstruction is used as the per-frame data source.  The changes
made here are deliberately limited to evidence that can be checked from the
inputs without ground truth:

* two long, visually static phases are represented by one pooled pose each;
* during manipulation, a local render-and-compare search improves each pose
  against the SAM object mask while staying within a small trust region of the
  released tracker;
* the one contiguous interval with no object evidence is bridged by SE(3)
  interpolation;
* a small Savitzky-Golay temporal filter is selected only when it preserves the
  mask fit.

No network access, fitted per-frame correction table, or embedded trajectory is
used.  All decisions below are recomputed from the command-line inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

# Keep the CUDA work deterministic and modest.  nvdiffrast rasterization is the
# only GPU operation used by this script.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.b …[+21859 chars]
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
frame 000: mask IoU 0.830 -> 0.874, rot [0.   5.25 0.  ], dt [0.   0.04 0.06]
frame 714: mask IoU 0.608 -> 0.739, rot [ 3.   4.5 -1.5], dt [0.   0.04 0.01]
manipulation frame 164/621: accepted=0 rejected=1
manipulation frame 214/621: accepted=44 rejected=7
manipulation frame 264/621: accepted=94 rejected=7
manipulation frame 314/621: accepted=144 rejected=7
manipulation frame 364/621: accepted=194 rejected=7
manipulation frame 414/621: accepted=244 rejected=7
manipulation frame 464/621: accepted=294 rejected=7
manipulation frame 514/621: accepted=337 rejected=7
manipulation frame 564/621: accepted=387 rejected=7
manipulation frame 614/621: accepted=437 rejected=7
AttributeError: 'scipy.spatial.transform._rotation.Rotation' object has no attribute 'degrees'

The above exceptio …[+678 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from scipy.spatial.transform import Rotation
import numpy as np
r=Rotation.from_euler('x',57.2958,degrees=True);print(r.magnitude(),r.as_rotvec(),r.magnitude()*180/np.pi)
PY"
```
<details><summary>output</summary>

```
1.0000003575641672 [1.00000036 0.         0.        ] 57.295800000000014

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'degrees(q\\|\\.magnitude' /work/outputs/solve.py"
```
<details><summary>output</summary>

```
324:    steps_deg = np.degrees(q[:-1].inv() * q[1:]).magnitude()
449:        acc = np.degrees((q[1:-1].inv() * q[:-2]).magnitude()
450:                         + (q[2:].inv() * q[1:-1]).magnitude())

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace('steps_deg = np.degrees(q[:-1].inv() * q[1:]).magnitude()','steps_deg = (q[:-1].inv() * q[1:]).magnitude() * (180.0 / math.pi)')
s=s.replace('acc = np.degrees((q[1:-1].inv() * q[:-2]).magnitude()\\n                         + (q[2:].inv() * q[1:-1]).magnitude())','acc = ((q[1:-1].inv() * q[:-2]).magnitude()\\n               + (q[2:].inv() * q[1:-1]).magnitude()) * (180.0 / math.pi)')
old='''        f\"- Static pose floor clearance: initial {float(((T_init[:3,:3] @ mesh.vertices.T).T + T_init[:3,3]).dot(np.r_[np.asarray(json.load(open(args.inputs / 'ground_plane.json'))['plane'])[:3], 1]) if False else 0):.0f}\"
        \" (ground-plane distances are reported in the console run because the released plane fit has several-cm tolerance).\\\\n\"
'''
new='''        f\"- Static pose minimum floor distance: initial {static_floor_init:.4f} m, final {static_floor_final:.4f} m \"
        \"(the released plane fit has several-cm tolerance; poses were not snapped).\\\\n\"
'''
assert old in s
s=s.replace(old,new)
# add computed vars before report append static text
needle='''    report.append(\"### Diagnostics\\\\n\")
    report.append(
'''
insert='''    report.append(\"### Diagnostics\\\\n\")
    plane = np.asarray(json.load(open(args.inputs / \"ground_plane.json\"))[\"plane\"])
    def min_floor_distance(Tx):
        P = (Tx[:3, :3] @ mesh.vertices.T).T + Tx[:3, 3]
        return float((P @ plane[:3] …[+322 chars]
```
<details><summary>output</summary>

```
324:    steps_deg = (q[:-1].inv() * q[1:]).magnitude() * (180.0 / math.pi)
340:        "rot_step_mean_deg": float(steps_deg.mean()),
341:        "rot_step_p95_deg": float(np.percentile(steps_deg, 95)),
526:    static_floor_init = min_floor_distance(T_init)
527:    static_floor_final = min_floor_distance(T_final)
533:        f"- Static pose minimum floor distance: initial {static_floor_init:.4f} m, final {static_floor_final:.4f} m "

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
frame 000: mask IoU 0.830 -> 0.874, rot [0.   5.25 0.  ], dt [0.   0.04 0.06]
frame 714: mask IoU 0.608 -> 0.739, rot [ 3.   4.5 -1.5], dt [0.   0.04 0.01]
manipulation frame 164/621: accepted=0 rejected=1
manipulation frame 214/621: accepted=44 rejected=7
manipulation frame 264/621: accepted=94 rejected=7
manipulation frame 314/621: accepted=144 rejected=7
manipulation frame 364/621: accepted=194 rejected=7
manipulation frame 414/621: accepted=244 rejected=7
manipulation frame 464/621: accepted=294 rejected=7
manipulation frame 514/621: accepted=337 rejected=7
manipulation frame 564/621: accepted=387 rejected=7
manipulation frame 614/621: accepted=437 rejected=7
Traceback (most recent call last):
  File "/work/outputs/solve.py", line 553, in <module>
    main()
  File "/wor …[+507 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace('return trajectory_metrics(Tx, object_masks[lo:hi], person_masks[lo:hi],\\n                                  opt, keypoints[lo:hi], contact[lo:hi], 1)','return trajectory_metrics(Tx[lo:hi], object_masks[lo:hi], person_masks[lo:hi],\\n                                  opt, keypoints[lo:hi], contact[lo:hi], 1)');open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py"
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
frame 000: mask IoU 0.830 -> 0.874, rot [0.   5.25 0.  ], dt [0.   0.04 0.06]
frame 714: mask IoU 0.608 -> 0.739, rot [ 3.   4.5 -1.5], dt [0.   0.04 0.01]
manipulation frame 164/621: accepted=0 rejected=1
manipulation frame 214/621: accepted=44 rejected=7
manipulation frame 264/621: accepted=94 rejected=7
manipulation frame 314/621: accepted=144 rejected=7
manipulation frame 364/621: accepted=194 rejected=7
manipulation frame 414/621: accepted=244 rejected=7
manipulation frame 464/621: accepted=294 rejected=7
manipulation frame 514/621: accepted=337 rejected=7
manipulation frame 564/621: accepted=387 rejected=7
manipulation frame 614/621: accepted=437 rejected=7

=== validation ===
baseline {'mask_iou_mean': 0.7523249259632286, 'mask_iou_median': 0.8263994772380745, 'rot_ste …[+580 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
z=np.load('/work/outputs/object_poses_world.npz');T=z['T_world_mesh'].astype(float);B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for name,X in [('base',B),('res',T)]:
 R=Rotation.from_matrix(X[:,:3,:3]); a=(R[:-1].inv()*R[1:]).magnitude()*180/np.pi;d=np.linalg.norm(X[1:,:3,3]-X[:-1,:3,3],axis=1)
 print(name,'max rot frames',[(int(i),round(a[i],1)) for i in np.argsort(a)[-20:][::-1]])
 print(name,'max trans frames',[(int(i),round(d[i],3)) for i in np.argsort(d)[-20:][::-1]])
PY"
```
<details><summary>output</summary>

```
base max rot frames [(486, 22.8), (494, 22.4), (504, 18.8), (490, 17.7), (484, 13.9), (392, 12.9), (493, 12.4), (491, 12.0), (495, 10.0), (503, 10.0), (402, 10.0), (488, 9.5), (391, 7.9), (465, 7.7), (394, 7.5), (439, 7.0), (487, 7.0), (462, 6.9), (443, 5.6), (398, 5.4)]
base max trans frames [(184, 0.105), (183, 0.102), (178, 0.087), (181, 0.086), (179, 0.085), (173, 0.08), (479, 0.077), (254, 0.071), (476, 0.068), (486, 0.067), (170, 0.066), (177, 0.065), (490, 0.063), (439, 0.06), (278, 0.057), (524, 0.055), (497, 0.053), (502, 0.053), (253, 0.052), (498, 0.052)]
res max rot frames [(484, 33.1), (486, 31.9), (392, 24.4), (391, 24.1), (402, 23.0), (396, 21.8), (398, 21.7), (452, 21.6), (564, 21.0), (457, 19.4), (585, 19.2), (401, 19.1), (453, 17.7), (587, 17.5), (351, 16.7), (586, 14.9), …[+355 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/outputs/REPORT.md && echo '--- files' && ls -lh /work/outputs"
```
<details><summary>output</summary>

```
# Object trajectory recovery report

## Method

The released CoCoNet/depth-corrected `T_world_mesh` was treated as the per-frame proposal. Frames 0-163 and 622-714 were identified as static from the video and pooled with a quaternion mean and coordinate-wise median; each pooled pose was then refined against a temporal-consensus object mask with an nvdiffrast render-and-compare search. During manipulation, the same local 6-DoF search refined poses whose object mask had at least 300 visible pixels, with a 12 degree / 14 cm trust region and a hand-contact safeguard. Frames 491-497 have no usable object mask, so they were bridged by SE(3) interpolation between the adjacent accepted poses. A short Savitzky-Golay filter was selected only where it did not reduce mask agreement.

## Phases and con …[+2160 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/test_smoothing.py <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs')
import solve
from pathlib import Path
inputs=Path('/work/inputs')
cam,recon,T_base,kp,contact,mesh,om,pm=solve.load_inputs(inputs)
opt=solve.MaskPoseOptimizer(cam,mesh,T_base)
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float)
for w in [1,3,5,7,9,11,15,21]:
 Tc=solve.smooth_trajectory(T,164,622,w) if w>1 else T.copy()
 Tc[:164]=T[:164];Tc[622:]=T[622:]
 m=solve.trajectory_metrics(Tc,om,pm,opt,kp,contact,1)
 print(w,m,flush=True)
PY
python /work/scratch/analysis/test_smoothing.py"
```
<details><summary>output</summary>

```
1 {'mask_iou_mean': 0.847346626832665, 'mask_iou_median': 0.8749296408870877, 'rot_step_mean_deg': 2.5797416529655184, 'rot_step_p95_deg': 8.758553753529135, 'trans_step_mean_m': 0.01907126485157295, 'trans_step_p95_m': 0.06647421680103867, 'hand_clearance_median_m': 0.015491663839963058}
3 {'mask_iou_mean': 0.847346626832665, 'mask_iou_median': 0.8749296408870877, 'rot_step_mean_deg': 2.579741652965518, 'rot_step_p95_deg': 8.75855375352912, 'trans_step_mean_m': 0.01907126485157294, 'trans_step_p95_m': 0.06647421680103867, 'hand_clearance_median_m': 0.015491663839963372}
5 {'mask_iou_mean': 0.8430471763418229, 'mask_iou_median': 0.8749296408870877, 'rot_step_mean_deg': 1.7056896774384476, 'rot_step_p95_deg': 5.6577086949242705, 'trans_step_mean_m': 0.013594696851963748, 'trans_step_p95_m …[+1540 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float);B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for name,X in [('base',B),('res',T)]:
 print(name)
 for i in list(range(158,172))+list(range(615,627)):
  R=Rotation.from_matrix(X[:,:3,:3]);a=(R[i-1].inv()*R[i]).magnitude()*180/np.pi;d=np.linalg.norm(X[i,:3,3]-X[i-1,:3,3])
  print(i,round(a,2),round(d,3),np.round(X[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
base
158 0.0 0.0 [0.07  0.864 3.634]
159 0.0 0.005 [0.07  0.863 3.63 ]
160 0.0 0.005 [0.07  0.864 3.634]
161 0.0 0.0 [0.07  0.864 3.634]
162 0.0 0.0 [0.07  0.864 3.634]
163 0.0 0.0 [0.07  0.864 3.634]
164 1.33 0.027 [0.066 0.861 3.608]
165 0.25 0.004 [0.066 0.858 3.604]
166 1.31 0.024 [0.064 0.864 3.627]
167 0.89 0.034 [0.064 0.871 3.661]
168 1.33 0.013 [0.062 0.87  3.673]
169 0.96 0.018 [0.061 0.86  3.658]
170 1.12 0.022 [0.057 0.862 3.68 ]
171 0.53 0.066 [0.054 0.838 3.618]
615 0.0 0.001 [-0.725  0.93   2.957]
616 0.0 0.001 [-0.725  0.929  2.956]
617 0.83 0.022 [-0.732  0.933  2.976]
618 1.63 0.008 [-0.736  0.936  2.982]
619 1.48 0.024 [-0.727  0.927  2.961]
620 0.89 0.012 [-0.732  0.927  2.972]
621 2.55 0.04 [-0.742  0.934  3.01 ]
622 0.0 0.002 [-0.742  0.933  3.008]
623 0.0 0.004 [-0.7 …[+1044 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/test_boundary.py <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
from pathlib import Path
inputs=Path('/work/inputs');cam,recon,T_base,kp,contact,mesh,om,pm=solve.load_inputs(inputs);opt=solve.MaskPoseOptimizer(cam,mesh,T_base)
T0=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float);Tinit=T0[0].copy();Tfinal=T0[-1].copy()
def smoothstep(x):return x*x*(3-2*x)
def interp2(A,B,a):
 qa=solve.Rotation.from_matrix(A[:3,:3]).as_quat();qb=solve.Rotation.from_matrix(B[:3,:3]).as_quat()
 if qa@qb<0:qb=-qb
 q=qa*(1-a)+qb*a;q/=np.linalg.norm(q);X=np.eye(4);X[:3,:3]=solve.Rotation.from_quat(q).as_matrix();X[:3,3]=(1-a)*A[:3,3]+a*B[:3,3];return X
for w in [9,11,15]:
 T=solve.smooth_trajectory(T0,164,622,w)
 # blend from static at 164 into smoothed carried over 12 frames
 raw=T.copy()
 for i in range(164,177):
  a=smoothstep((i-164+1)/13)
  T[i]=interp2(Tinit,raw[i],a)
 # blend into final static over 10 frames 613..621
 for i in range(613,622):
  a=smoothstep((i-613+1)/10)
  T[i]=interp2(raw[i],Tfinal,a)
 T[:164]=Tinit;T[622:]=Tfinal
 m=solve.trajectory_metrics(T,om,pm,opt,kp,contact,1)
 print(w,m,flush=True)
PY
python /work/scratch/analysis/test_boundary.py"
```
<details><summary>output</summary>

```
9 {'mask_iou_mean': 0.8417954345400992, 'mask_iou_median': 0.8749718658564033, 'rot_step_mean_deg': 1.24924916623864, 'rot_step_p95_deg': 3.8553144362675504, 'trans_step_mean_m': 0.011249546306812443, 'trans_step_p95_m': 0.03280221045245237, 'hand_clearance_median_m': 0.014668438757951813}
11 {'mask_iou_mean': 0.8414719048838574, 'mask_iou_median': 0.874901475059115, 'rot_step_mean_deg': 1.1950345598092684, 'rot_step_p95_deg': 3.54453349852183, 'trans_step_mean_m': 0.010817573496406656, 'trans_step_p95_m': 0.031103590310847782, 'hand_clearance_median_m': 0.014980281134232451}
15 {'mask_iou_mean': 0.8407914478771379, 'mask_iou_median': 0.8748036796051155, 'rot_step_mean_deg': 1.1081975370833637, 'rot_step_p95_deg': 3.2013452177338184, 'trans_step_mean_m': 0.01027485337704187, 'trans_step_ …[+80 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
# add blend function after interpolate_poses
needle='''def smooth_trajectory(T: np.ndarray, lo: int, hi: int, window: int) -> np.ndarray:
'''
insert='''def blend_pose(A: np.ndarray, B: np.ndarray, a: float) -> np.ndarray:
    \"\"\"Interpolate two poses; "'`a=0` is A and `a=1` is B."""
    qa = Rotation.from_matrix(A[:3, :3]).as_quat()
    qb = Rotation.from_matrix(B[:3, :3]).as_quat()
    if np.dot(qa, qb) < 0:
        qb = -qb
    q = qa * (1.0 - a) + qb * a
    q /= np.linalg.norm(q)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Rotation.from_quat(q).as_matrix()
    out[:3, 3] = (1.0 - a) * A[:3, 3] + a * B[:3, 3]
    return out


def smooth_trajectory(T: np.ndarray, lo: int, hi: int, window: int) -> np.ndarray:
'"'''
assert needle in s;s=s.replace(needle,insert)
# filter candidates and threshold
s=s.replace('''    candidates = {w: smooth_trajectory(T, STATIC_INITIAL_END, STATIC_FINAL_START, w)
                  for w in (1, 3, 5, 7)}''','''    candidates = {w: smooth_trajectory(T, STATIC_INITIAL_END, STATIC_FINAL_START, w)
                  for w in (1, 5, 7, 9, 11, 15, 21)}''')
s=s.replace('''    eligible = [w for w, v in scored.items() if v[0][\"mask_iou_mean\"] >= best_iou - 0.001]
    chosen_window = max(eligible)
    T = candidates[chosen_window]

    # Re-pin the pooled static phases after filtering.
    T[:STATIC_INITIAL_END] = T_init
    T[STATIC_FINAL_START:] = T_final''','''  …[+1136 chars]
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
frame 000: mask IoU 0.830 -> 0.874, rot [0.   5.25 0.  ], dt [0.   0.04 0.06]
frame 714: mask IoU 0.608 -> 0.739, rot [ 3.   4.5 -1.5], dt [0.   0.04 0.01]
manipulation frame 164/621: accepted=0 rejected=1
manipulation frame 214/621: accepted=44 rejected=7
manipulation frame 264/621: accepted=94 rejected=7
manipulation frame 314/621: accepted=144 rejected=7
manipulation frame 364/621: accepted=194 rejected=7
manipulation frame 414/621: accepted=244 rejected=7
manipulation frame 464/621: accepted=294 rejected=7
manipulation frame 514/621: accepted=337 rejected=7
manipulation frame 564/621: accepted=387 rejected=7
manipulation frame 614/621: accepted=437 rejected=7

=== validation ===
baseline {'mask_iou_mean': 0.7523249259632286, 'mask_iou_median': 0.8263994772380745, 'rot_ste …[+580 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/outputs/REPORT.md'
```
<details><summary>output</summary>

```
# Object trajectory recovery report

## Method

The released CoCoNet/depth-corrected `T_world_mesh` was treated as the per-frame proposal. Frames 0-163 and 622-714 were identified as static from the video and pooled with a quaternion mean and coordinate-wise median; each pooled pose was then refined against a temporal-consensus object mask with an nvdiffrast render-and-compare search. During manipulation, the same local 6-DoF search refined poses whose object mask had at least 300 visible pixels, with a 12 degree / 14 cm trust region and a hand-contact safeguard. Frames 491-497 have no usable object mask, so they were bridged by SE(3) interpolation between the adjacent accepted poses. A short Savitzky-Golay filter was selected only where it did not reduce mask agreement.

## Phases and con …[+1898 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz';z=np.load(p);print(z.files);T=z['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all());print('det',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max());print('orth err',np.abs(T[:,:3,:3]@T[:,:3,:3].transpose(0,2,1)-np.eye(3)).max())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float);B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
# difference stats and jumps
R=Rot.from_matrix(T[:,:3,:3]);Rb=Rot.from_matrix(B[:,:3,:3])
ang=(Rb.inv()*R).magnitude()*180/np.pi
d=np.linalg.norm(T[:,:3,3]-B[:,:3,3],axis=1)
print('change rot deg quant',np.quantile(ang,[0,.5,.9,.95,.99,1]));print('change trans quant',np.quantile(d,[0,.5,.9,.95,.99,1]))
for i in [0,163,164,165,170,175,350,400,490,495,500,550,600,621,622,700]:
 print(i,'diffrot',round(ang[i],2),'difft',np.round(T[i,:3,3]-B[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
change rot deg quant [ 0.66750161  5.37968676  8.58071954  9.83152041 12.96803429 30.58142254]
change trans quant [0.00493735 0.05040146 0.11147507 0.12260673 0.13542808 0.17088374]
0 diffrot 5.25 difft [0.    0.047 0.089]
163 diffrot 5.25 difft [0.    0.043 0.071]
164 diffrot 5.76 difft [0.003 0.045 0.096]
165 diffrot 5.4 difft [0.004 0.046 0.095]
170 diffrot 3.24 difft [ 0.003  0.02  -0.027]
175 diffrot 4.24 difft [-0.003 -0.001 -0.049]
350 diffrot 8.08 difft [ 0.002  0.005 -0.007]
400 diffrot 8.64 difft [0.002 0.04  0.018]
490 diffrot 10.53 difft [-0.009  0.032  0.058]
495 diffrot 8.21 difft [-0.03  -0.007  0.146]
500 diffrot 11.82 difft [ 0.014 -0.038  0.024]
550 diffrot 7.17 difft [ 0.001  0.028 -0.038]
600 diffrot 6.75 difft [-0.002  0.021 -0.012]
621 diffrot 5.61 difft [ 0.01   0.02 …[+102 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/overlay_result.py <<'PY'
import sys,numpy as np,cv2,h5py,torch
sys.path.insert(0,'/work/outputs');import solve
from pathlib import Path
cam,recon,Tb,kp,contact,mesh,om,pm=solve.load_inputs(Path('/work/inputs'));opt=solve.MaskPoseOptimizer(cam,mesh,Tb);T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float)
cap=cv2.VideoCapture('/work/inputs/video.mp4');sel=[0,150,175,200,225,300,350,375,400,450,475,490,495,500,510,525,550,575,600,620,650,700]
tiles=[]
for i in sel:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();gt=om[i];per=pm[i]
 crop=opt.crop_from_mask(gt) if gt.sum()>0 else (550,500,320);x0,y0,s=crop
 Ti=torch.tensor(T[i],dtype=torch.float32,device='cuda');m=(opt.render_mask(Ti,crop)&~torch.tensor(per[y0:y0+s,x0:x0+s],device='cuda')).cpu().numpy();g=gt[y0:y0+s,x0:x0+s]
 c=im[y0:y0+s,x0:x0+s].copy();cs,_=cv2.findContours(g.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(c,cs,-1,(0,255,0),2);cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(c,cs,-1,(0,0,255),2);cv2.putText(c,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,255),2);tiles.append(cv2.resize(c,(320,320)))
rows=[]
for j in range(0,len(tiles),6):
 row=tiles[j:j+6]
 while len(row)<6:row.append(np.zeros_like(tiles[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/analysis/overlay_result.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95])
PY
python /wo …[+38 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,torch,trimesh,h5py,PIL,cv2,sys
for m in [numpy,scipy,torch,trimesh,h5py,PIL,cv2]: print(m.__name__,m.__version__)
print('python',sys.version.split()[0])
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
torch 2.5.1+cu124
trimesh 5.1.1
h5py 3.16.0
PIL 10.2.0
cv2 4.11.0
python 3.11.10

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/TOOLS.md <<'MD'
# Tools used

No external code repository, model weights, or data were downloaded.  All
downloads/weight fields below are intentionally recorded as none.  The solution
uses only the files supplied in "'`/work/inputs` and software already installed in
the container.

## Required by `solve.py`

| tool | version | role |
|---|---:|---|
| Python | 3.11.10 | runtime |
| NumPy | 1.26.3 | pose/mask arrays and interpolation |
| SciPy | 1.16.3 | rotation quaternions and Savitzky-Golay filtering |
| PyTorch | 2.5.1+cu124 | GPU tensor operations |
| nvdiffrast | 0.4.0 | differentiable-free mesh rasterization for mask scoring |
| trimesh | 5.1.1 | loading the supplied textured GLB mesh |
| h5py | 3.16.0 | reading the supplied SAM2 masks |

## Inspected / tried but not required by the final script

| tool | version | status |
|---|---:|---|
| OpenCV (`cv2`) | 4.11.0 | used only for diagnostic overlay montages; not used by `solve.py` |
| Pillow | 10.2.0 | used only for diagnostic texture inspection; not used by `solve.py` |
| pyrender | 0.1.45 | availability checked; not used |
| pytorch3d | 0.7.9 | availability checked; not used |
| Open3D | 0.20.0 | availability checked; not used |

## Git repositories

None cloned or used.

## Downloaded model weights and URLs

None downloaded or used by this solution.  The supplied CARI4D intermediate
reconstruction under `/pipeline` was inspected, but the deterministic solver
uses only `/work/input …[+389 chars]
```
<details><summary>output</summary>

```
total 80K
-rw-rw-r-- 1 hanzhou hanzhou 2.7K Oct 20 21:32 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.8K Oct 20 21:33 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:29 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  41K Oct 20 21:31 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  24K Oct 20 21:29 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py'
s=open(p).read()
s=s.replace('they were bridged by SE(3) interpolation between the adjacent accepted poses. A short \"\\n        \"Savitzky-Golay filter was selected only where it did not reduce mask agreement.\\\\n', 'they were bridged by SE(3) interpolation between the adjacent accepted poses. A short \"\\n        \"Savitzky-Golay filter was selected as the longest window staying within 0.006 mean mask IoU \"\\n        \"of the best-fitting smoothing variant, and the pooled static poses were blended into the \"\\n        \"carried trajectory around pickup and placement.\\\\n')
s=s.replace('The changes were adopted because mask agreement and temporal consistency improve, especially in the two \"\\n        \"long static phases.', 'The changes were adopted because mask agreement and mean temporal step size improve, especially in the two \"\\n        \"long static phases.')
open(p,'w').write(s)
p2='/work/outputs/REPORT.md'
r=open(p2).read()
r=r.replace('they were bridged by SE(3) interpolation between the adjacent accepted poses. A short Savitzky-Golay filter was selected only where it did not reduce mask agreement.', 'they were bridged by SE(3) interpolation between the adjacent accepted poses. A short Savitzky-Golay filter was selected as the longest window staying within 0.006 mean mask IoU of the best-fitting smoothing variant, and the pooled static poses were blended into the carried trajectory around pickup and  …[+372 chars]
```
<details><summary>output</summary>

```
total 84K
-rw-rw-r-- 1 hanzhou hanzhou 2.8K Oct 20 21:34 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.8K Oct 20 21:33 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:34 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  41K Oct 20 21:31 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  25K Oct 20 21:34 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'longest window\\|mean temporal' /work/outputs/solve.py /work/outputs/REPORT.md && python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
assert T.shape==(715,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
print('final output verified',T.shape,T.dtype)
PY"
```
<details><summary>output</summary>

```

```
</details>

**Agent:** Completed the trajectory recovery task.

Outputs created:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(715, 4, 4)`, float64
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, methodology, confidence, and validation
- `/work/outputs/TOOLS.md` — software versions and usage

Key validation result:

| Metric | Baseline | Result |
|---|---:|---:|
| Mean mask IoU | 0.7523 | **0.8426** |
| Median mask IoU | 0.8264 | **0.8749** |
| Mean translation step | 0.0143 m | **0.0118 m** |

The final solver run completed in 203.4 seconds and passed output dtype, shape, and rigidity checks.
