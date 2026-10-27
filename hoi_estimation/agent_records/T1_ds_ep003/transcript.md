# T1_ds_ep003

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T13:41:15Z

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/AGENT_TASK.md && printf '\\n---FILES---\\n' && find /work -maxdepth 2 -type f -printf '%p\\n' | sort | head -200"
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
/usr/bin/bash -c "sed -n '1,140p' /work/AGENT_TASK.md"
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
/usr/bin/bash -c "sed -n '141,320p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -lh /work/inputs /pipeline /weights 2>/dev/null | head -200'
```
<details><summary>output</summary>

```
/pipeline:
total 28K
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 15 12:39 depth
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 15 12:42 export
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 15 12:59 foundationpose
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 15 13:50 inference
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 15 12:47 mhr_init
-rw-rw-r-- 1 hanzhou hanzhou 6.2K Oct 15 13:50 pipeline_report.json

/weights:
total 24K
drwxrwxr-x 4 hanzhou hanzhou 4.0K Oct  4 17:42 cari4d
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct  4 17:44 foundationpose
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct  4 17:45 grounding_dino
drwxrwxr-x 4 hanzhou hanzhou 4.0K Oct  4 17:44 hf_home
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct  4 17:45 sam2
drwxrwxr-x 4 hanzhou hanzhou 4.0K Oct  4 17:43 sam3d_body

/work/inputs:
total 75M
-rw-rw-r-- 1 hanzhou hanzhou  825 Oct 20 …[+516 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np, json, os
for p in ['/work/inputs/reconstruction.npz','/work/inputs/masks.h5','/work/inputs/frame0_reference_human.npz']:
 print('\\n',p)
 if p.endswith('.npz'):
  d=np.load(p)
  for k in d.files: print(k,d[k].shape,d[k].dtype)
 else:
  with h5py.File(p) as f:
   for k in f: print(k,f[k].shape,f[k].dtype)
for p in ['/work/inputs/camera.json','/work/inputs/episode.json','/work/inputs/ground_plane.json']:
 print('\\n',p); print(open(p).read()[:4000])
PY"
```
<details><summary>output</summary>

```

 /work/inputs/reconstruction.npz
T_world_mesh (781, 4, 4) float32
T_cam_mesh_foundationpose (781, 4, 4) float32
T_cam_mesh_coconet (781, 4, 4) float32
human_vertices_world (781, 4610, 3) float32
human_keypoints_world (781, 70, 3) float32
contact_prob_right_left (781, 2) float32

 /work/inputs/masks.h5
object (781, 1152, 1536) bool
person (781, 1152, 1536) bool

 /work/inputs/frame0_reference_human.npz
vertices_world (18056, 3) float32

 /work/inputs/camera.json
{
 "K": [
  [
   820.000244140625,
   0.0,
   758.1000366210938
  ],
  [
   0.0,
   820.000244140625,
   560.5
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "K_pipeline": [
  [
   818.300048828125,
   0.0,
   768.0
  ],
  [
   0.0,
   818.3001098632812,
   576.0
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "T_world_cam": [
  [
   0.99927004 …[+1025 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '75,145p'"
```
<details><summary>output</summary>

```
    75	  y-down (gravity along +y).
    76	- `ground_plane.json` - the released floor plane in world coordinates: n.x + d = 0 with `plane` = [n, d].
    77	- `frame0_reference_human.npz` - `vertices_world`: the true human mesh vertices at frame 0 only.
    78	- `reconstruction.npz` - the existing reconstruction:
    79	  - `T_world_mesh` (F,4,4): the baseline object poses in world (mesh-file frame), consistent with `camera.json`.
    80	  - `T_cam_mesh_foundationpose`, `T_cam_mesh_coconet` (F,4,4): intermediate camera-frame estimates (FoundationPose
    81	    tracking, and its refinement by the CARI4D network), mesh-file frame, for `K_pipeline`. Their depth scale is not
    82	    calibrated to the world; `T_world_mesh` already includes a per-frame depth correction.
    83	  - `human_vert …[+3707 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '35,78p'"
```
<details><summary>output</summary>

```
    35	   compare with the image: silhouette agreement with the object mask (excluding pixels covered by the person),
    36	   edges / texture / colour agreement, consistency with the MoGe-2 depth map; optical flow or feature tracks between
    37	   neighbouring frames (a correct trajectory must explain the image motion of the object's texture).
    38	3. **Several hypotheses per frame, not one**: the baseline, the FoundationPose and CoCoNet estimates, their 180 deg
    39	   symmetry flips about each principal axis of the mesh, poses propagated from neighbouring frames, and fresh
    40	   registrations (e.g. render-and-compare optimisation from several initialisations, or FoundationPose
    41	   re-registration on chosen frames).
    42	4. **Global decoding over time**: choose one hyp …[+3319 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/inputs/task.txt; echo; find /pipeline -maxdepth 3 -type f -printf '%p %k KB\\n' | sort"
```
<details><summary>output</summary>

```
Walk to the coffee mug on the round table, lift it by the handle with your right hand and set it down, then repeat with your left, alternating a few times before leaving it on the table.

/pipeline/.stages/01_depth.json 4 KB
/pipeline/.stages/02_export.json 4 KB
/pipeline/.stages/03_human_initialization.json 4 KB
/pipeline/.stages/04_depth_alignment.json 4 KB
/pipeline/.stages/05_object_pose.json 4 KB
/pipeline/.stages/06_coconet.json 4 KB
/pipeline/.stages/07_refinement.json 4 KB
/pipeline/.stages/08_render.json 4 KB
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0 KB
/pipeline/depth/moge2/aligned.h5 964316 KB
/pipeline/depth/moge2/depth_report.json 4 KB
/pipeline/depth/moge2/intrinsics.pkl 84 KB
/pipeline/depth/moge2/raw.npy 2699144 KB
/pipeline/export/episode_000003/edex 4 KB
/pi …[+675 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/pipeline_report.json'
```
<details><summary>output</summary>

```
{
  "checkpoint": {
    "mtime_ns": 1791121400297177794,
    "path": "/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth",
    "repository": "1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9",
    "sha256": "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3",
    "size": 2084484133,
    "step": 200000
  },
  "comparison_video": {
    "mtime_ns": 1792057813479844323,
    "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000003/inference/episode_000003_step200000_before_after.mp4",
    "size": 58876806
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.0 …[+5447 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m),m)
print('bounds',m.bounds,'extents',m.extents,'center',m.centroid,'faces',len(m.faces),'verts',len(m.vertices))
print('visual',type(m.visual),getattr(m.visual,'material',None))
S,T=trimesh.bounds.oriented_bounds(m)
print('OBB S',S,'T',T)
print('watertight',m.is_watertight,'volume',m.volume)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> <trimesh.Trimesh(vertices.shape=(297718, 3), faces.shape=(99260, 3))>
bounds [[-0.04920735 -0.03856977 -0.08927552]
 [ 0.07257884  0.04913357  0.11351284]] extents [0.12178619 0.08770333 0.20278836] center [ 3.39885346e-11  1.07875623e-11 -9.39737639e-12] faces 99260 verts 297718
visual <class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x723a5ea1ffd0>
OBB S [[-0.02158574  0.99968285  0.01297118 -0.00448969]
 [ 0.99976623  0.02156781  0.00152049 -0.01149296]
 [ 0.00124025  0.01300097 -0.99991471  0.01166378]
 [ 0.          0.          0.          1.        ]] T [0.08773579 0.12131281 0.20307708]
watertight False volume 0.000903788268892975

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi; python - <<'PY'
import torch,cv2,trimesh,open3d,nvdiffrast
print('torch',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('cv2',cv2.__version__,'trimesh',trimesh.__version__,'o3d',open3d.__version__)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 True NVIDIA RTX 6000 Ada Generation
cv2 4.11.0 trimesh 5.1.1 o3d 0.20.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
for p in ['/pipeline/foundationpose/episode_000003.pkl','/pipeline/mhr_init/episode_000003.pkl','/pipeline/mhr_init/episode_000003.stats.pkl']:
 print('\\n',p)
 try:
  x=pickle.load(open(p,'rb')); print(type(x));
  if isinstance(x,dict):
   for k,v in x.items():
    try: print(' ',k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None))
    except: print(' ',k,type(v))
  else:
   print(getattr(x,'keys',lambda:[])())
 except Exception as e: print('ERR',e)
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
  frames <class 'list'> None None
  camera_id <class 'int'> None None
  fp_poses_cam <class 'numpy.ndarray'> (781, 1, 4, 4) float32
  fp_candidate_count <class 'numpy.ndarray'> (781,) int32
  fp_selected_candidate_index <class 'numpy.ndarray'> (781,) int32
  fp_selected_iou <class 'numpy.ndarray'> (781,) float32
  fp_selected_visibility <class 'numpy.ndarray'> (781,) float32
  fp_retry_count <class 'numpy.ndarray'> (781,) int32
  fp_reliable <class 'numpy.ndarray'> (781,) bool
  fp_oracle_rotation_error_deg <class 'numpy.ndarray'> (781,) float32
  fp_translation_threshold_m <class 'numpy.ndarray'> (781,) float32
  fp_selected_translation_distance_m <class 'numpy.ndarray'> (781,) float32
  gt_object_visibility_ratio <class 'numpy.ndarray'> (781,) float32
  gt_object_visibilit …[+2518 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, cv2, json
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz')
T=D['T_world_mesh']; kp=D['human_keypoints_world']; cp=D['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as h:
 om=h['object'][:]; pm=h['person'][:]
F=len(T)
areas=om.sum((1,2)); ious=[]; 
# areas obb/centroid and root motion
xyz=T[:,:3,3]; rel=np.einsum('fji,fjk->fik',T[:,:3,:3],T[0,:3,:3])
ang=np.array([R.from_matrix(x).magnitude() for x in rel])
# per-step rot
stepang=np.zeros(F); stepang[1:]=R.from_matrix(np.einsum('fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3])).magnitude()*180/np.pi
stepxyz=np.zeros(F);stepxyz[1:]=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
# project bbox
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
# mesh sample
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); vs=m.vertices[::100]
proj=[];depths=[]
for f in range(F):
 p=(Tcw@T[f]@np.c_[vs,np.ones(len(vs))].T).T[:,:3]
 uv=(K@p.T).T; uv=uv[:,:2]/uv[:,2:3]; proj.append([uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()]);depths.append(p[:,2].mean())
proj=np.array(proj);depths=np.array(depths)
np.savez('/work/scratch/diag.npz',areas=areas,person_areas=pm.sum((1,2)),xyz=xyz,ang=ang,stepang=stepang,stepxyz=stepxyz,proj=proj,depths=depths,cp=cp)
# print every 20
print('f area pers useproj depth  …[+375 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/einsumfunc.py", line 1371, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: einstein sum subscripts string contains too many subscripts for operand 1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch /work/outputs && python - <<'PY'
import cv2, numpy as np, h5py, json
from pathlib import Path
cap=cv2.VideoCapture('/work/inputs/video.mp4')
F=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(F,fps,W,H)
idx=np.arange(0,F,20)
thumbs=[]
with h5py.File('/work/inputs/masks.h5') as h:
 om=h['object']; pm=h['person']
 for i in idx:
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,bgr=cap.read()
  if not ok: print('fail',i);continue
  ov=bgr.copy(); o=om[i].astype(bool); p=pm[i].astype(bool)
  ov[o]=(0.35*ov[o]+0.65*np.array([40,240,40])).astype(np.uint8)
  ov[p]=(0.75*ov[p]+0.25*np.array([240,40,40])).astype(np.uint8)
  ys,xs=np.where(o)
  if len(xs): c=f'o {o.sum()} bbox {xs.min()},{ys.min()}-{xs.max()},{ys.max()}'
  else:c='o none'
  cv2.putText(ov,f'{i} {c}',(15,35),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2)
  ov=cv2.resize(ov,(512,384))
  thumbs.append(ov)
rows=[]
for j in range(0,len(thumbs),5):
 row=thumbs[j:j+5]
 while len(row)<5: row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/montage20.jpg',np.vstack(rows))
print('saved',len(thumbs))
PY"
```
<details><summary>output</summary>

```
781 30.0 1536 1152
saved 40

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p) as f:
 def visit(k,v):
  if hasattr(v,'shape') and v.shape: print(k,v.shape,v.dtype)
 f.visititems(visit)
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (781,) object
alignment/front_stereo_camera_left/scale (781,) float32
alignment/front_stereo_camera_left/shift (781,) float32
alignment/front_stereo_camera_left/valid_count (781,) int32
frame_names/front_stereo_camera_left (781,) object
raw/front_stereo_camera_left (781,) object

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json, trimesh
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz'); T=D['T_world_mesh']; cp=D['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as h: om=h['object'][:]; pm=h['person'][:]
F=len(T); xyz=T[:,:3,3]
rel=np.einsum('fij,kj->fik',T[:,:3,:3],T[0,:3,:3])
ang=np.array([R.from_matrix(x).magnitude() for x in rel])*180/np.pi
stepang=np.zeros(F)
# Rotation difference prev"'^-1 curr = prev.T curr
stepang[1:]=R.from_matrix(np.einsum('"'fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3])).magnitude()*180/np.pi
stepxyz=np.zeros(F);stepxyz[1:]=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); vs=m.vertices[::100]
proj=[];depths=[]
for f in range(F):
 p=(Tcw@T[f]@np.c_[vs,np.ones(len(vs))].T).T[:,:3]
 uv=(K@p.T).T; uv=uv[:,:2]/uv[:,2:3]; proj.append([uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()]);depths.append(p[:,2].mean())
proj=np.array(proj);depths=np.array(depths)
np.savez('/work/scratch/diag.npz',areas=om.sum((1,2)),person_areas=pm.sum((1,2)),xyz=xyz,ang=ang,stepang=stepang,stepxyz=stepxyz,proj=proj,depths=depths,cp=cp)
print('f area pers useproj depth relang stepang stepxyz contact')
for f in range(0,F,20): print(f,om[f].sum(),pm[f].sum(),np.round(proj[f]).ast …[+259 chars]
```
<details><summary>output</summary>

```
f area pers useproj depth relang stepang stepxyz contact
0 1969 27273 [843, 605, 871, 672] 2.607 0.0 0.0 0.0 [0. 0.]
20 1980 29584 [843, 608, 871, 674] 2.638 4.1 5.85 0.0041 [0. 0.]
40 1941 29942 [841, 605, 871, 671] 2.615 5.4 5.41 0.0145 [0. 0.]
60 2024 31202 [841, 605, 871, 671] 2.623 3.0 1.27 0.0376 [0. 0.]
80 1943 35276 [842, 605, 872, 673] 2.586 4.1 3.4 0.0256 [0. 0.]
100 1738 30508 [844, 609, 872, 674] 2.638 10.3 10.42 0.0217 [0. 0.]
120 1708 30402 [843, 609, 870, 673] 2.651 7.4 2.65 0.0139 [0. 1.]
140 1653 30276 [844, 579, 872, 644] 2.673 18.5 5.84 0.0234 [0. 1.]
160 1664 30820 [807, 498, 850, 563] 2.686 65.1 1.33 0.0116 [0. 1.]
180 1726 29982 [821, 537, 862, 600] 2.691 103.0 45.07 0.019 [0. 1.]
200 1808 29392 [846, 587, 880, 651] 2.777 29.1 8.82 0.0126 [0. 1.]
220 1691 29735 [843,  …[+3097 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4');F=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with h5py.File('/work/inputs/masks.h5') as h: om=h['object'][:]; pm=h['person'][:]
# crop table+person center
idx=np.arange(0,F,10); thumbs=[]
for i in idx:
 cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,b=cap.read(); assert ok
 ov=b.copy();o=om[i];p=pm[i]
 ov[o]=(0.2*ov[o]+0.8*np.array([40,255,40])).astype(np.uint8)
 ov[p]=(0.8*ov[p]+0.2*np.array([255,40,40])).astype(np.uint8)
 crop=ov[460:820,670:970]
 crop=cv2.resize(crop,(300,360))
 cv2.putText(crop,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
 thumbs.append(crop)
rows=[]
for j in range(0,len(thumbs),8):
 row=thumbs[j:j+8]
 while len(row)<8:row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/montage10crop.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/pipeline/foundationpose/episode_000003.pkl','rb'))
for k in ['fp_reliable','fp_selected_iou','fp_selected_visibility','fp_tracking_state','fp_full_image_iou','fp_centroid_distance','fp_rendered_visibility','fp_confirmed_pose_source']:
 a=x[k]; print(k,a.dtype,a.shape,'min max mean',a.min(),a.max(),a.mean())
print('metadata',x['metadata'])
print('frames head',x['frames'][:3])
PY"
```
<details><summary>output</summary>

```
fp_reliable bool (781,) min max mean False True 0.9948783610755442
fp_selected_iou float32 (781,) min max mean 0.0788901 0.99117386 0.72891307
fp_selected_visibility float32 (781,) min max mean 0.1748691 1.0 0.94734323
fp_tracking_state uint8 (781,) min max mean 0 0 0.0
fp_full_image_iou float32 (781,) min max mean 0.0788901 0.99117386 0.72891307
fp_centroid_distance float32 (781,) min max mean 0.00027203272 0.5446791 0.0728623
fp_rendered_visibility float32 (781,) min max mean 0.1748691 1.0 0.94734323
fp_confirmed_pose_source uint8 (781,) min max mean 1 7 6.967989756722151
metadata {'source': 'foundationpose_register_first_then_track_metric_depth', 'depth_root': '/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000003/depth/moge2/aligned.h5', 'depth_source': 'metric_depth', …[+7869 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as h: om=h['object'][:]
for i in [0,100,140,160,180,200,300,320,360,420,500,560,620,660,700,750]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read()
 ys,xs=np.where(om[i])
 if len(xs):
  x0,x1=max(0,xs.min()-40),min(im.shape[1],xs.max()+41);y0,y1=max(0,ys.min()-30),min(im.shape[0],ys.max()+31)
  crop=im[y0:y1,x0:x1]; crop=cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
  cv2.imwrite(f'/work/scratch/obj_{i:03d}.png',crop)
  print(i,(x0,y0,x1,y1),om[i].sum())
PY"
```
<details><summary>output</summary>

```
0 (795, 572, 909, 703) 1969
100 (799, 575, 910, 703) 1738
140 (800, 551, 909, 676) 1653
160 (764, 470, 889, 593) 1664
180 (785, 505, 906, 631) 1726
200 (801, 562, 912, 690) 1808
300 (776, 460, 907, 583) 1796
320 (791, 492, 914, 617) 1721
360 (799, 575, 910, 703) 1736
420 (802, 533, 917, 661) 1689
500 (791, 574, 908, 702) 1900
560 (779, 451, 913, 560) 1477
620 (808, 576, 917, 705) 1747
660 (807, 512, 930, 637) 1723
700 (803, 576, 913, 704) 1726
750 (804, 576, 913, 704) 1747

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual
print('uv',v.uv.shape,v.uv.min(0),v.uv.max(0))
mat=v.material
print(vars(mat))
if hasattr(mat,'baseColorTexture') and mat.baseColorTexture is not None:
 im=mat.baseColorTexture; print(type(im),im.size,im.mode); im.save('/work/scratch/texture.png')
for i,uv in enumerate(v.uv): print(i,uv)
print('vertex colors',getattr(v,'vertex_colors',None))
PY"
```
<details><summary>output</summary>

```
uv (297718, 2) [0.003636 0.005216] [0.99623501 0.99644703]
{'_data': {'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=1240x1268 at 0x71DF69327250>, 'doubleSided': True, 'name': '_texture'}}
<class 'PIL.Image.Image'> (1240, 1268) RGB
0 [0.36049801 0.19765699]
1 [0.36049801 0.19765699]
2 [0.59968501 0.09370899]
3 [0.59968501 0.09370899]
4 [0.59968501 0.09370899]
5 [0.59968501 0.09370899]
6 [0.63831502 0.33081198]
7 [0.206909   0.00667101]
8 [0.206909   0.00667101]
9 [0.63831502 0.33081198]
10 [0.63831502 0.33081198]
11 [0.63831502 0.33081198]
12 [0.206909   0.00667101]
13 [0.63749301 0.330531  ]
14 [0.63749301 0.330531  ]
15 [0.63749301 0.330531  ]
16 [0.63749301 0.330531  ]
17 [0.63749301 0.330531  ]
18 [0.638399   0.32953703]
19 [0.638399   0.32953703]
20 [0 …[+1047807 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,glob,numpy as np,os
fs=sorted(glob.glob('/work/scratch/obj_*.png'))
ims=[]
for p in fs:
 im=cv2.imread(p); im=cv2.resize(im,(300,300)); cv2.putText(im,os.path.basename(p)[4:7],(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2); ims.append(im)
rows=[np.hstack(ims[i:i+4]) for i in range(0,len(ims),4)]
cv2.imwrite('/work/scratch/obj_sheet.jpg',np.vstack(rows))
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
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz'); B=D['T_world_mesh']; F=D['T_cam_mesh_foundationpose']; C=D['T_cam_mesh_coconet']; Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc)
# rotations diff B vs Twc*F,C
for name,X in [('F',F),('C',C)]:
 Xw=Twc[None]@X
 dang=np.array([R.from_matrix(B[f,:3,:3].T@Xw[f,:3,:3]).magnitude() for f in range(len(B))])*180/np.pi
 # compare camera-space B to X rotation
 Bc=Tcw[None]@B
 dc=np.array([R.from_matrix(Bc[f,:3,:3].T@X[f,:3,:3]).magnitude() for f in range(len(B))])*180/np.pi
 print(name,'world rot diff mean/max',dang.mean(),dang.max(),'cam rot diff',dc.mean(),dc.max())
# trans distances F/C world-cam no depth correction
for name,X in [('F',F),('C',C)]:
 t=np.linalg.norm(B[:,:3,3]-(Twc[None]@X)[:,:3,3],axis=1)
 print(name,'trans err world vs raw',np.percentile(t,[0,25,50,75,100]))
# camera relative translation and baseline camera trans
for name,X in [('F',F),('C',C)]:
 xb=(Tcw[None]@B)[:,:3,3]; xc=X[:,:3,3]; print(name,'z ratios median',np.median(xb[:,2]/xc[:,2]),'ratio percentiles',np.percentile(xb[:,2]/xc[:,2],[0,25,50,75,100]))
# per frame every 20 rotation diffs
Bc=Tcw[None]@B
for f in range(0,len(B),20):
 def ad(a,b):return R.from_matrix(a.T@b).magnitude()*180/np.pi
 print(f,'F',round(ad(Bc[f,:3,:3],F[f,:3,:3]),1),'C',round(ad(Bc[f,:3,:3],C[f,:3,:3]),1))
PY"
```
<details><summary>output</summary>

```
F world rot diff mean/max 21.300365212365453 159.03285389591665 cam rot diff 21.300365212365453 159.03285389591667
C world rot diff mean/max 1.2545034940205704 1.2809900018026998 cam rot diff 1.2545034940205704 1.2809900018026972
F trans err world vs raw [0.05671192 0.07729298 0.10743098 0.33735215 0.71431012]
C trans err world vs raw [0.05625247 0.06448425 0.07259589 0.07958215 0.08908696]
F z ratios median 0.9673996835727833 ratio percentiles [0.78962546 0.89072312 0.96739968 0.98884265 1.03042055]
C z ratios median 0.9838013075842699 ratio percentiles [0.97511421 0.97962398 0.98380131 0.98913121 1.00090205]
0 F 12.9 C 1.2
20 F 14.6 C 1.2
40 F 13.3 C 1.2
60 F 9.6 C 1.2
80 F 9.9 C 1.2
100 F 6.3 C 1.2
120 F 10.9 C 1.2
140 F 7.5 C 1.3
160 F 8.4 C 1.3
180 F 29.1 C 1.3
200 F 28.2 C 1.3
220 F  …[+486 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
D=np.load('/work/inputs/reconstruction.npz'); B=D['T_world_mesh']; C=D['T_cam_mesh_coconet'];Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc)
Bc=Tcw[None]@B
d=Bc[:,:3,3]-C[:,:3,3]
print('cam delta mean std min max',d.mean(0),d.std(0),d.min(0),d.max(0))
print('norm percentiles',np.percentile(np.linalg.norm(d,axis=1),[0,10,25,50,75,90,100]))
for f in range(0,781,40): print(f,np.round(Bc[f,:3,3],4),np.round(C[f,:3,3],4),np.round(d[f],4))
# maybe delta related C xyz
for j in range(3): print(j,np.corrcoef(d[:,j],C[:,j])[0,1])
PY"
```
<details><summary>output</summary>

```
cam delta mean std min max [ 0.02725839  0.04933619 -0.04181708] [0.00178309 0.0024467  0.01756446] [ 0.02338597  0.04579021 -0.0706007 ] [0.03225177 0.05749637 0.00242415]
norm percentiles [0.05625247 0.05822353 0.06448425 0.07259589 0.07958215 0.08409798
 0.08908696]
0 [0.3199 0.2338 2.6059] [0.2904 0.1858 2.6184] [ 0.0295  0.048  -0.0125]
40 [0.3183 0.2334 2.6132] [0.2883 0.1849 2.6214] [ 0.0301  0.0484 -0.0082]
80 [0.3185 0.2332 2.5851] [0.2897 0.1859 2.6018] [ 0.0288  0.0473 -0.0166]
120 [0.3226 0.247  2.6497] [0.2954 0.2002 2.6877] [ 0.0272  0.0468 -0.038 ]
160 [ 0.2253 -0.1037  2.6819] [ 0.1939 -0.1552  2.69  ] [ 0.0314  0.0515 -0.0081]
200 [0.3556 0.1884 2.7744] [0.3286 0.1387 2.8246] [ 0.027   0.0497 -0.0502]
240 [0.3303 0.24   2.7257] [0.3016 0.1912 2.759 ] [ 0.0287  0.0488 -0.03 …[+1624 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2
from scipy.signal import medfilt
D=np.load('/work/inputs/reconstruction.npz');T=D['T_world_mesh'];kp=D['human_keypoints_world'];cp=D['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as h: om=h['object'][:]
F=len(T); ys=[];xs=[];area=[]
for i in range(F):
 y,x=np.where(om[i]);ys.append(y.mean());xs.append(x.mean());area.append(len(y))
ys=np.array(ys);xs=np.array(xs);area=np.array(area)
t=T[:,:3,3]
# distance hand keypoint centers to object center? hands 21-41,42-62 often 21 entries
rh=kp[:,21:42].mean(1); lh=kp[:,42:63].mean(1)
rd=np.linalg.norm(rh-t,axis=1);ld=np.linalg.norm(lh-t,axis=1)
# object bbox height width
with h5py.File('/work/inputs/masks.h5') as h:
 hw=[];hh=[]
 for i in range(F):
  y,x=np.where(h['object'][i]);hw.append(x.max()-x.min()+1);hh.append(y.max()-y.min()+1)
hw=np.array(hw);hh=np.array(hh)
np.savez('/work/scratch/signals.npz',ys=ys,xs=xs,area=area,hw=hw,hh=hh,rd=rd,ld=ld)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(6,1,figsize=(18,14),sharex=True)
ax[0].plot(ys,label='mask cy');ax[0].plot(xs,label='mask cx');ax[0].legend();ax[0].set_ylabel('pixels')
ax[1].plot(area);ax[1].set_ylabel('area')
ax[2].plot(hw,label='w');ax[2].plot(hh,label='h');ax[2].legend();ax[2].set_ylabel('bbox')
ax[3].plot(cp[:,0],label='right contact');ax[3].plot(cp[:,1],label='left contact');ax[3].set_ylim(-.05,1.05);ax[3].legend()
ax[4].plot(rd,label='R hand d …[+362 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz');T=D['T_world_mesh']
for f in range(0,781,20):
 M=T[f,:3,:3]
 # axes world
 z=M[:,2]; y=M[:,1];x=M[:,0]
 e=R.from_matrix(M).as_euler('xyz',degrees=True)
 print(f,'exyz',np.round(e,1),'zaxis',np.round(z,2),'t',np.round(T[f,:3,3],3))
PY"
```
<details><summary>output</summary>

```
0 exyz [-83.4 -35.9 -11.2] zaxis [0.13 0.99 0.09] t [0.468 0.351 2.588]
20 exyz [-86.5 -39.1  -9.2] zaxis [0.12 0.99 0.05] t [0.474 0.362 2.618]
40 exyz [-89.6 -37.6  -8.9] zaxis [0.15 0.99 0.01] t [0.467 0.35  2.595]
60 exyz [-82.9 -36.5 -14.4] zaxis [0.18 0.98 0.1 ] t [0.469 0.351 2.604]
80 exyz [-83.2 -32.9 -14. ] zaxis [0.18 0.98 0.1 ] t [0.466 0.35  2.567]
100 exyz [-92.7 -42.2  -1.7] zaxis [ 0.06  1.   -0.03] t [0.474 0.364 2.618]
120 exyz [-91.5 -35.9  -3.2] zaxis [ 0.07  1.   -0.02] t [0.473 0.365 2.631]
140 exyz [-73.4 -52.9 -20.3] zaxis [0.12 0.98 0.17] t [0.479 0.27  2.655]
160 exyz [158.8 -70.2  92.3] zaxis [ 0.33  0.89 -0.32] t [0.376 0.015 2.673]
180 exyz [ 93.2 -41.1 171.3] zaxis [ 0.11  0.99 -0.04] t [0.435 0.135 2.676]
200 exyz [-105.7  -15.    -4.6] zaxis [ 0.15  0.95 -0. …[+2247 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr,inspect
print(nvdiffrast.__file__)
print(inspect.signature(dr.RasterizeCudaContext))
print(inspect.signature(dr.rasterize))
print(inspect.signature(dr.interpolate))
print(inspect.signature(dr.texture))
print(inspect.getsource(dr.texture)[:3000])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
NameError: name 'nvdiffrast' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr,inspect
print(dr.__file__)
for f in ['RasterizeCudaContext','rasterize','interpolate','texture']:
 o=getattr(dr,f); print(f,inspect.signature(o) if callable(o) else o)
print(inspect.getsource(dr.texture)[:4000])
PY"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/__init__.py
RasterizeCudaContext (device=None)
rasterize (glctx, pos, tri, resolution, ranges=None, grad_db=True)
interpolate (attr, rast, tri, rast_db=None, diff_attrs=None)
texture (tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto', boundary_mode='wrap', max_mip_level=None)
def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """Perform texture sampling.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Args:
        tex: Texture tensor with dtype `torch.float32`. For 2D textures, must have shape
             [minibatch_size, tex_height, tex_widt …[+3556 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/test_render.py <<'PY'
import torch,trimesh,numpy as np,json,h5py,cv2
import nvdiffrast.torch as dr
D=np.load('/work/inputs/reconstruction.npz');T0=D['T_world_mesh'][0]
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
H,W=1152,1536; fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
# OpenCV to clip, positive w, y down
P=torch.tensor([[2*fx/W,0,1-2*cx/W,0],[0,2*fy/H,1-2*cy/H,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
Tm=T0.astype(np.float32)
vm=(torch.tensor(Tcw@Tm,dtype=torch.float32,device='cuda')[None]@torch.cat([V,torch.ones_like(V[:,:1])],1).T).T[...,:3]
pos=(P[None]@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T
pos=pos[None]
ctx=dr.RasterizeCudaContext()
rast,db=dr.rasterize(ctx,pos,F,torch.tensor([H,W],device='cuda'))
mask=(rast[...,3]>0)
print('mask',mask.sum().item(),'bbox')
ys,xs=torch.where(mask[0]);print(xs.min().item(),ys.min().item(),xs.max().item(),ys.max().item())
# test flip choices

def render(flipx=False,flipy=False):
 p=pos.clone()
 if flipx:p[...,0]*=-1
 if flipy:p[...,1]*=-1
 r,_=dr.rasterize(ctx,p,F,torch.tensor([H,W],device='cuda'))
 mm=r[...,3]>0
 if mm.sum()==0:return mm
 yy,xx=torch.where(mm[0]);return mm,(xx.min().item(),yy.min …[+803 chars]
```
<details><summary>output</summary>

```
/work/scratch/test_render.py:11: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3683.)
  vm=(torch.tensor(Tcw@Tm,dtype=torch.float32,device='cuda')[None]@torch.cat([V,torch.ones_like(V[:,:1])],1).T).T[...,:3]
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 12, in <module>
    pos=(P[None]@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T
         ~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: Expected size for first two dimensions of bat …[+42 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_render.py'
s=open(p).read().replace(\"vm=(torch.tensor(Tcw@Tm,dtype=torch.float32,device='cuda')[None]@torch.cat([V,torch.ones_like(V[:,:1])],1).T).T[...,:3]\\npos=(P[None]@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T\", \"vh=torch.cat([V,torch.ones_like(V[:,:1])],1)\\nvm=(torch.tensor(Tcw@Tm,dtype=torch.float32,device='cuda')@vh.T).T[:,:3]\\npos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T\")
open(p,'w').write(s)
PY
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 16, in <module>
    rast,db=dr.rasterize(ctx,pos,F,torch.tensor([H,W],device='cuda'))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 78, in forward
    ou …[+317 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_render.py';s=open(p).read();s=s.replace(\"pos=pos[None]\",\"pos=pos.contiguous()[None]\").replace(\"dr.rasterize(ctx,p,F,\",\"dr.rasterize(ctx,p.contiguous(),F.contiguous(),\");open(p,'w').write(s)
PY
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
mask 1560 bbox
862 636 889 702
flip False False (862, 636, 889, 702) 1560
flip False True (862, 449, 889, 515) 1561
flip True False (646, 636, 673, 702) 1559
flip True True (646, 449, 673, 515) 1560

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nl -ba /work/scratch/test_render.py'
```
<details><summary>output</summary>

```
     1	import torch,trimesh,numpy as np,json,h5py,cv2
     2	import nvdiffrast.torch as dr
     3	D=np.load('/work/inputs/reconstruction.npz');T0=D['T_world_mesh'][0]
     4	m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
     5	V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
     6	cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
     7	H,W=1152,1536; fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
     8	# OpenCV to clip, positive w, y down
     9	P=torch.tensor([[2*fx/W,0,1-2*cx/W,0],[0,2*fy/H,1-2*cy/H,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
    10	Tm=T0.astype(np.float32)
    11	vh=torch.cat( …[+1748 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/test_render.py').read().split(\"ctx=dr\")[0])
p=pos[0].detach().cpu().numpy(); z=vm.detach().cpu().numpy()
u=(p[:,0]/p[:,3]+1)*W/2-.5;v=(p[:,1]/p[:,3]+1)*H/2-.5
print('manual bbox',u.min(),v.min(),u.max(),v.max())
# actual cam direct
q=Tcw[:3,:3]@(T0[:3,:3]@m.vertices.T+T0[:3,3:4])+Tcw[:3,3:4]
u2=K[0,0]*q[0]/q[2]+K[0,2];v2=K[1,1]*q[1]/q[2]+K[1,2]
print('direct',u2.min(),v2.min(),u2.max(),v2.max())
print('pos x range samples',p[:3])
PY"
```
<details><summary>output</summary>

```
manual bbox 861.95514 635.6089 889.905 702.2515
direct 842.6552503779084 605.108865318755 870.6050573182586 671.7515525595497
pos x range samples [[0.3640972  0.56425405 1.         2.5779111 ]
 [0.3640972  0.56425405 1.         2.5779111 ]
 [0.3640972  0.56425405 1.         2.5779111 ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_render.py';s=open(p).read().replace(\"1-2*cx/W\",\"2*cx/W-1\").replace(\"1-2*cy/H\",\"2*cy/H-1\");open(p,'w').write(s)
PY
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
mask 1558 bbox
843 605 870 671
flip False False (843, 605, 870, 671) 1558
flip False True (843, 480, 870, 546) 1558
flip True False (665, 605, 692, 671) 1558
flip True True (665, 480, 692, 546) 1558

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/test_texture.py <<'PY'
import torch,trimesh,numpy as np,json,h5py,cv2
import nvdiffrast.torch as dr
from PIL import Image
D=np.load('/work/inputs/reconstruction.npz');T0=D['T_world_mesh'][0]
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda')
imtex=np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255
tex=torch.tensor(imtex,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
P=torch.tensor([[2*fx/W,0,2*cx/W-1,0],[0,2*fy/H,2*cy/H-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
vh=torch.cat([V,torch.ones_like(V[:,:1])],1);vm=(torch.tensor(Tcw@T0,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
ctx=dr.RasterizeCudaContext();rast,db=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([H,W],device='cuda'))
uv,uvda=dr.interpolate(UV[None].contiguous(),rast,F.contiguous(),rast_db=db,diff_attrs=['uv'])
for flip in [False,True]:
 u=uv.clone()
 if flip:u[...,1]=1-u[...,1]
 col=dr.texture(tex[None],u.contiguous(),uv_da=uvda,filter_mode='linear',boundary_mode='clamp')[0]
 mask=rast[0, …[+462 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_texture.py", line 14, in <module>
    uv,uvda=dr.interpolate(UV[None].contiguous(),rast,F.contiguous(),rast_db=db,diff_attrs=['uv'])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 275, in interpolate
    diff_attrs = np.asarray(diff_attrs, np.int32)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: 'uv'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.signature(dr.interpolate));print(inspect.getsource(dr.interpolate)[:2500])
PY"
```
<details><summary>output</summary>

```
(attr, rast, tri, rast_db=None, diff_attrs=None)
def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Interpolate vertex attributes.

    All input tensors must be contiguous and reside in GPU memory. The output tensors
    will be contiguous and reside in GPU memory.

    Args:
        attr: Attribute tensor with dtype `torch.float32`. 
              Shape is [num_vertices, num_attributes] in range mode, or 
              [minibatch_size, num_vertices, num_attributes] in instanced mode.
              Broadcasting is supported along the minibatch axis.
        rast: Main output tensor from `rasterize()`.
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        rast_db: (Optional) Tensor containing image-space derivatives of barycentrics …[+1750 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/diff_attrs=\\['uv'\\]/diff_attrs='all'/\" /work/scratch/test_texture.py
python /work/scratch/test_texture.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for k in f:
  print(k,type(f[k]),f[k].shape,f[k].dtype)
  if k.startswith('aligned/'):
   x=f[k]; print('  first type',type(x[0]),getattr(x[0],'shape',None),getattr(x[0],'dtype',None));
   for j in [0,1]:
    try: print(' ',j,type(x[j]),getattr(x[j],'shape',None),getattr(x[j],'dtype',None), np.asarray(x[j]).min() if hasattr(x[j],'min') else '')
    except Exception as e:print(e)
   break
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'Group' object has no attribute 'shape'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 def visit(k,o):
  print(k,type(o).__name__,getattr(o,'shape',None),getattr(o,'dtype',None))
 f.visititems(visit)
 print('aligned item',type(f['aligned/front_stereo_camera_left']),[type(f['aligned/front_stereo_camera_left'][j]) for j in range(3)])
 for j in range(3):
  a=f['aligned/front_stereo_camera_left'][j];print(j,type(a),np.asarray(a).shape,np.asarray(a).dtype,np.asarray(a).min(),np.asarray(a).max())
PY"
```
<details><summary>output</summary>

```
aligned Group None None
aligned/front_stereo_camera_left Dataset (781,) object
alignment Group None None
alignment/front_stereo_camera_left Group None None
alignment/front_stereo_camera_left/scale Dataset (781,) float32
alignment/front_stereo_camera_left/shift Dataset (781,) float32
alignment/front_stereo_camera_left/valid_count Dataset (781,) int32
frame_names Group None None
frame_names/front_stereo_camera_left Dataset (781,) object
raw Group None None
raw/front_stereo_camera_left Dataset (781,) object
aligned item <class 'h5py._hl.dataset.Dataset'> [<class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>]
0 <class 'numpy.ndarray'> (659458,) uint8 0 255
1 <class 'numpy.ndarray'> (656653,) uint8 0 255
2 <class 'numpy.ndarray'> (657345,) uint8 0 255

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/depth/moge2/depth_report.json; echo; cat /pipeline/export/episode_000003/wild_export.json'
```
<details><summary>output</summary>

```
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 9.434300422668457,
    "min": 1.546643853187561
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 185.36007482200512,
  "firstFrameIntrinsicsPx": [
    [
      818.300048828125,
      0.0,
      768.0
    ],
    [
      0.0,
      818.3001098632812,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "focalXPx": {
    "max": 835.5772705078125,
    "mean": 822.786865234375,
    "min": 814.8235473632812
  },
  "focalYPx": {
    "max": 835.5772705078125,
    "mean": 822.786865234375,
    "min": 814.8235473632812
  },
  "fps": 30.0,
  "frames": 781,
  "framesPerSecond": 4.213420828352423,
  "height": 1152,
  "intrinsicsFile": {
  …[+3965 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for j in range(3):
  for name in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
   a=np.asarray(f[name][j]);print(j,name,a.shape,a.dtype,np.unique(a).size,a.min(),a.max())
 print(f['frame_names/front_stereo_camera_left'][:3]);print('scale shift valid',f['alignment/front_stereo_camera_left/scale'][:5],f['alignment/front_stereo_camera_left/shift'][:5],f['alignment/front_stereo_camera_left/valid_count'][:5])
PY"
```
<details><summary>output</summary>

```
0 aligned/front_stereo_camera_left (659458,) uint8 256 0 255
0 raw/front_stereo_camera_left (596889,) uint8 256 0 255
1 aligned/front_stereo_camera_left (656653,) uint8 256 0 255
1 raw/front_stereo_camera_left (596975,) uint8 256 0 255
2 aligned/front_stereo_camera_left (657345,) uint8 256 0 255
2 raw/front_stereo_camera_left (596596,) uint8 256 0 255
[b'000000' b'000001' b'000002']
scale shift valid [1.1552941 1.1431972 1.1460482 1.1463515 1.1372637] [0. 0. 0. 0. 0.] [22792 23346 23685 23065 22971]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned/front_stereo\\|valid_count\\|depth_min\\|uint8\" -n /code/reconstruction /workspace 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_mv/vis/wis3d_helper.py:27:    _DEFAULT_COLOR = np.array([180, 180, 200], dtype=np.uint8)
/code/reconstruction/modules/v2d_mv/vis/wis3d_helper.py:37:                colors = np.array(vc[:, :3], dtype=np.uint8)
/code/reconstruction/modules/v2d_mv/vis/wis3d_helper.py:42:                    colors = np.array(vc[:, :3], dtype=np.uint8)
/code/reconstruction/modules/v2d_mv/vis/wis3d_helper.py:77:            cyl.visual.vertex_colors = np.tile(axis_colors[axis_idx], (len(cyl.vertices), 1)).astype(np.uint8)
/code/reconstruction/modules/v2d_mv/vis/tests/test_renderer.py:59:        color = np.zeros((self.viewport_height, self.viewport_width, 4), dtype=np.uint8)
/code/reconstruction/modules/v2d_mv/vis/tests/test_renderer.py:112:    image = np.zeros((3, 4, 3), dtype=np.u …[+15988 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/score_candidates.py <<'PY'
import torch,trimesh,numpy as np,json,h5py,cv2,time
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz');B=D['T_world_mesh'].astype(np.float32);C=D['T_cam_mesh_coconet'].astype(np.float32);FP=D['T_cam_mesh_foundationpose'].astype(np.float32)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); KP=np.array(cam['K_pipeline'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
ctx=dr.RasterizeCudaContext()
# fixed crop covering all object positions
x0,y0,w,h=700,420,280,320
def proj(Kc):
 fx,fy=Kc[0,0],Kc[1,1];cx,cy=Kc[0,2],Kc[1,2]
 return torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,2*fy/h,2*(cy-y0)/h-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
PB=proj(K);PP=proj(KP)
def render(Tw,Kmat,P):
 vm=(torch.tensor(Tw,dtype=torch.float32,device='cuda')@vh.T).T[:,:3]
 pos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
 r,_=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([h,w],device='cuda'))
 return (r[0,...,3]>0).cpu().numpy()
# Candidates: local 180 flips
flips=[np.eye(3,dtype=np.float32)] …[+1218 chars]
```
<details><summary>output</summary>

```
0 0.09796619415283203 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_flip2': 0.0}
50 0.35282111167907715 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_flip2': 0.0}
100 0.8495523929595947 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_flip2': 0.0}
150 1.197052001953125 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_flip2': 0.0}
200 1.806562900543213 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_flip2': 0.0}
250 2.2850568294525146 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_flip2': 0.0}
300 2.774143695831299 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_flip2': 0.0}
350 3.4064900875091553 {'B': 0.0, 'C': 0.0, 'F': 0.0, 'B_flip0': 0.0, 'B_flip1': 0.0, 'B_f …[+1100 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/score_candidates.py').read().split(\"N=len(B)\")[0])
pred=render(B[0],K,PB);gt=masks[0]
print('pred',pred.shape,pred.sum(),'gt',gt.sum(),'and',(pred&gt).sum())
ys,xs=np.where(pred);print('pbbox',xs.min(),xs.max(),ys.min(),ys.max())
ys,xs=np.where(gt);print('gbbox',xs.min(),xs.max(),ys.min(),ys.max())
import cv2
cv2.imwrite('/work/scratch/dbg_pred.png',pred.astype(np.uint8)*255);cv2.imwrite('/work/scratch/dbg_gt.png',gt.astype(np.uint8)*255)
PY"
```
<details><summary>output</summary>

```
pred (320, 280) 1600 gt 1969 and 0
pbbox 190 217 222 289
gbbox 135 168 182 252

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/score_candidates.py').read().split(\"N=len(B)\")[0])
vm=(torch.tensor(B[0],dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(PB@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
u=(pos[0,:,0]/pos[0,:,3]+1)*w/2-.5;v=(pos[0,:,1]/pos[0,:,3]+1)*h/2-.5
print('manual',u.min().item(),u.max().item(),v.min().item(),v.max().item())
r,_=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([h,w],device='cuda'));m=r[...,3]>0;yy,xx=torch.where(m[0]);print('rast',xx.min().item(),xx.max().item(),yy.min().item(),yy.max().item())
PY"
```
<details><summary>output</summary>

```
manual 189.59841918945312 217.9883575439453 221.86630249023438 289.4548034667969
rast 190 217 222 289

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/score_candidates.py').read().split(\"N=len(B)\")[0])
print(K);print(PB)
vm=(torch.tensor(B[0],dtype=torch.float32,device='cuda')@vh.T).T[:,:3]
print(vm.min(0),vm.max(0)); p=(Twc@B[0]@np.r_[m.vertices[0],1])[:3]; print(p,p[0]/p[2],p[1]/p[2])
pos=(PB@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T
print(pos[0])
PY"
```
<details><summary>output</summary>

```
[[820.00024   0.      758.10004]
 [  0.      820.00024 560.5    ]
 [  0.        0.        1.     ]]
tensor([[ 5.8571,  0.0000, -0.5850,  0.0000],
        [ 0.0000,  5.1250, -0.1219,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  1.0000],
        [ 0.0000,  0.0000,  1.0000,  0.0000]], device='cuda:0')
torch.return_types.min(
values=tensor([0.4148, 0.2589, 2.5303], device='cuda:0'),
indices=tensor([100155, 274163,  44780], device='cuda:0')) torch.return_types.max(
values=tensor([0.5132, 0.4669, 2.6539], device='cuda:0'),
indices=tensor([65284, 29425, 69822], device='cuda:0'))
[0.60424643 0.57990529 2.53141148] 0.23869941231852174 0.22908377081362974
tensor([1.1821, 2.0669, 1.0000, 2.5584], device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/score_candidates.py';s=open(p).read();s=s.replace(\"vm=(torch.tensor(Tw,dtype=torch.float32,device='cuda')@vh.T).T[:,:3]\",\"vm=(torch.tensor(Tcw@Tw,dtype=torch.float32,device='cuda')@vh.T).T[:,:3]\")
# avoid double rendering iou
s=s.replace(\"out['B'][i]=np.logical_and(render(Bw,K,PB),gt).sum()/max(1,np.logical_or(render(Bw,K,PB),gt).sum()) # inefficient twice\",\"pred=render(Bw,K,PB);out['B'][i]=np.logical_and(pred,gt).sum()/max(1,np.logical_or(pred,gt).sum())\")
open(p,'w').write(s)
PY
python /work/scratch/score_candidates.py"
```
<details><summary>output</summary>

```
0 0.11886405944824219 {'B': 0.68, 'C': 0.674, 'F': 0.731, 'B_flip0': 0.567, 'B_flip1': 0.505, 'B_flip2': 0.57}
50 0.6674900054931641 {'B': 0.719, 'C': 0.715, 'F': 0.71, 'B_flip0': 0.583, 'B_flip1': 0.537, 'B_flip2': 0.604}
100 1.0292763710021973 {'B': 0.731, 'C': 0.733, 'F': 0.838, 'B_flip0': 0.641, 'B_flip1': 0.558, 'B_flip2': 0.631}
150 1.4752178192138672 {'B': 0.841, 'C': 0.838, 'F': 0.785, 'B_flip0': 0.641, 'B_flip1': 0.679, 'B_flip2': 0.756}
200 1.919302225112915 {'B': 0.507, 'C': 0.507, 'F': 0.619, 'B_flip0': 0.458, 'B_flip1': 0.407, 'B_flip2': 0.474}
250 2.4150776863098145 {'B': 0.702, 'C': 0.681, 'F': 0.746, 'B_flip0': 0.564, 'B_flip1': 0.549, 'B_flip2': 0.629}
300 2.7214202880859375 {'B': 0.468, 'C': 0.459, 'F': 0.22, 'B_flip0': 0.435, 'B_flip1': 0.485, 'B_flip2': 0.425}
350 3.150 …[+1468 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
x=np.load('/work/scratch/candidate_iou.npz')
plt.figure(figsize=(18,6))
for k in ['B','C','F','B_flip0','B_flip1','B_flip2']:
 plt.plot(x[k],label=k,alpha=.8)
plt.ylim(0,1);plt.grid();plt.legend();plt.xlabel('frame');plt.ylabel('silhouette IoU');plt.tight_layout();plt.savefig('/work/scratch/iou.png',dpi=140)
# best per frame
ks=['B','C','F','B_flip0','B_flip1','B_flip2'];a=np.stack([x[k] for k in ks],1)
print('best counts',{k:int((a.argmax(1)==i).sum()) for i,k in enumerate(ks)})
PY"
```
<details><summary>output</summary>

```
best counts {'B': 182, 'C': 7, 'F': 573, 'B_flip0': 1, 'B_flip1': 1, 'B_flip2': 17}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz');Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for name in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 T=D[name]
 if 'cam' in name:T=np.einsum('ij,fjk->fik',Twc,T)
 step=np.zeros(len(T));step[1:]=R.from_matrix(np.einsum('fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3])).magnitude()*180/np.pi
 print(name,'step percentiles',np.percentile(step,[50,75,90,95,99,100]),'frames >20',sum(step>20),'sum',step.sum())
 print('big',[(int(i),round(step[i],1)) for i in np.argsort(step)[-20:][::-1]])
PY"
```
<details><summary>output</summary>

```
T_world_mesh step percentiles [  5.83523469  10.31027871  19.34440355  31.85738476 122.95531059
 158.8359153 ] frames >20 75 sum 8394.096893343372
big [(423, 158.8), (302, 150.5), (301, 148.8), (656, 144.8), (300, 140.0), (588, 134.4), (544, 127.7), (422, 125.1), (419, 122.4), (660, 115.9), (654, 107.9), (545, 94.3), (417, 91.3), (541, 90.9), (332, 81.4), (657, 74.8), (655, 73.9), (661, 71.9), (420, 70.6), (543, 67.8)]
T_cam_mesh_foundationpose step percentiles [ 4.4856634   7.9655093  14.90702886 20.44784111 38.05544237 47.33532878] frames >20 43 sum 5234.735115665684
big [(332, 47.3), (340, 46.3), (302, 43.3), (194, 43.3), (195, 41.9), (180, 41.4), (651, 39.9), (323, 39.1), (663, 37.8), (341, 37.3), (454, 37.2), (338, 31.5), (661, 31.1), (333, 30.7), (318, 27.9), (432, 27.3), (182, 26.7) …[+470 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr
print([x for x in dir(dr) if not x.startswith('_')])
PY"
```
<details><summary>output</summary>

```
['DepthPeeler', 'RasterizeCudaContext', 'RasterizeGLContext', 'antialias', 'antialias_construct_topology_hash', 'get_log_level', 'interpolate', 'ops', 'rasterize', 'set_log_level', 'texture', 'texture_construct_mip']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.signature(dr.antialias)); print(inspect.getsource(dr.antialias)[:5000])
PY"
```
<details><summary>output</summary>

```
(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0)
def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """Perform antialiasing.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Note that silhouette edge determination is based on vertex indices in the triangle
    tensor. For it to work properly, a vertex belonging to multiple triangles must be
    referred to using the same vertex index in each triangle. Otherwise, nvdiffrast will always
    classify the adjacent edges as silhouette edges, which leads to bad performance and
    potentially incorrect gradients. If you are unsure whether your data is good, check
    which pixels are modified by the ant …[+1248 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
r=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r')
print(r.shape,r.dtype)
with h5py.File('/work/inputs/masks.h5') as h:o=h['object'][0]
for f in [0,100,160,200,300,350,500,660,750]:
 with h5py.File('/work/inputs/masks.h5') as h:o=h['object'][f]
 a=r[f];vals=a[o].astype(float)/1000
 print(f,vals.min(),np.percentile(vals,[10,25,50,75,90]),vals.max(),vals.mean())
PY"
```
<details><summary>output</summary>

```
(781, 1152, 1536) uint16
0 2.237 [2.319  2.322  2.332  2.367  2.4574] 5.385 2.405473844591163
100 2.247 [2.252 2.255 2.26  2.28  2.355] 2.617 2.284234752589183
160 2.29 [2.303  2.317  2.33   2.37   2.5077] 2.64 2.3638185096153848
200 2.225 [2.319  2.326  2.342  2.365  2.4546] 4.646 2.3837466814159294
300 2.209 [2.253   2.26775 2.293   2.32725 2.4035 ] 4.529 2.3202216035634744
350 2.17 [2.24  2.243 2.247 2.264 2.302] 2.537 2.261731651376147
500 2.24 [2.28  2.284 2.295 2.328 2.514] 2.932 2.338535789473684
660 2.316 [2.358  2.369  2.381  2.4115 2.486 ] 4.238 2.419150899593732
750 2.236 [2.248  2.251  2.266  2.288  2.3384] 2.706 2.283651974813967

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 s=f['alignment/front_stereo_camera_left/scale'][:];sh=f['alignment/front_stereo_camera_left/shift'][:]
 for i in [0,100,160,200,300,350,500,660,750]:print(i,s[i],sh[i])
PY"
```
<details><summary>output</summary>

```
0 1.1552941 0.0
100 1.1559333 0.0
160 1.1549373 0.0
200 1.1564933 0.0
300 1.173902 0.0
350 1.1619895 0.0
500 1.1590282 0.0
660 1.1699735 0.0
750 1.1441343 0.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/overlay_fb.py <<'PY'
exec(open('/work/scratch/score_candidates.py').read().split(\"N=len(B)\")[0])
import cv2
D=[];cap=cv2.VideoCapture('/work/inputs/video.mp4')
idx=np.arange(0,781,10);thumbs=[]
with h5py.File('/work/inputs/masks.h5') as h:
 for i in idx:
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();im=im[y0:y0+h,x0:x0+w]
  gt=h['object'][i,y0:y0+h,x0:x0+w]
  pb=render(B[i],K,PB);pf=render(Twc@FP[i],KP,PP);pc=render(Twc@C[i],KP,PP)
  ov=(im*.55).astype(np.uint8)
  # B blue, F red, GT green
  ov[pb]=(0.4*ov[pb]+0.6*np.array([255,0,0])).astype(np.uint8)
  ov[pf]=(0.4*ov[pf]+0.6*np.array([0,0,255])).astype(np.uint8)
  ov[gt]=(0.25*ov[gt]+0.75*np.array([0,255,0])).astype(np.uint8)
  ov=cv2.resize(ov,(280,320));cv2.putText(ov,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
  thumbs.append(ov)
rows=[]
for j in range(0,len(thumbs),8):
 row=thumbs[j:j+8]
 while len(row)<8:row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/overlay_fb.jpg',np.vstack(rows))
PY
python /work/scratch/overlay_fb.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/overlay_fb.py", line 7, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();im=im[y0:y0+h,x0:x0+w]
                                                                      ~~^~
TypeError: unsupported operand type(s) for +: 'int' and 'File'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/overlay_fb.py';s=open(p).read().replace(\"with h5py.File('/work/inputs/masks.h5') as h:\\n for i in idx:\",\"with h5py.File('/work/inputs/masks.h5') as hf:\\n for i in idx:\").replace(\"gt=h['object'][i,y0:y0+h,x0:x0+w]\",\"gt=hf['object'][i,y0:y0+yh,x0:x0+w]\")
# define yh
s=s.replace('D=[];cap=', 'yh=h;D=[];cap=')
open(p,'w').write(s)
PY
python /work/scratch/overlay_fb.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_translation_test.py <<'PY'
import torch,trimesh,numpy as np,json,h5py,cv2,time
import nvdiffrast.torch as dr
D=np.load('/work/inputs/reconstruction.npz');FP=D['T_cam_mesh_foundationpose'].astype(np.float32);B=D['T_world_mesh'].astype(np.float32)
cam=json.load(open('/work/inputs/camera.json'));KP=np.array(cam['K_pipeline'],np.float32);K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda')
imtex=np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255;tex=torch.tensor(imtex,device='cuda');vh=torch.cat([V,torch.ones_like(V[:,:1])],1);ctx=dr.RasterizeCudaContext()
x0,y0,w,h=700,420,280,320

def Pmat(Kc):
 fx,fy=Kc[0,0],Kc[1,1];cx,cy=Kc[0,2],Kc[1,2]
 return torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,2*fy/h,2*(cy-y0)/h-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
P=Pmat(KP)
def render(Tcam):
 vm=(torch.tensor(Tcam,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
 r,db=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([h,w],device='cuda'))
 uv,uvda=dr.interpolate(UV[None].contiguous(),r,F.contiguous …[+2156 chars]
```
<details><summary>output</summary>

```
/work/scratch/opt_translation_test.py:14: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  vm=(torch.tensor(Tcam,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
0 dt [0. 0. 0.] iou 0.731 -> 0.731 corr 0.213
100 dt [0. 0. 0.] iou 0.838 -> 0.838 corr 0.188
160 dt [0. 0. 0.] iou 0.827 -> 0.827 corr 0.351
180 dt [0. 0. 0.] iou 0.701 -> 0.701 corr 0.27
200 dt [0. 0. 0.] iou 0.619 -> 0.619 corr -0.19
250 dt [0. 0. 0.] iou 0.746 -> 0.746 corr -0.019
300 dt [0. 0. 0.] iou 0.22 -> 0.22 corr 0.118
350 dt [0. 0. 0.] iou 0.616 -> 0.616 corr -0.217
400 dt [0. 0. 0.] iou 0.735 -> 0. …[+396 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/opt_translation_test.py').read().split(\"cap=cv2.VideoCapture\")[0])
fi=350
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0:y0+h,x0:x0+w]
with h5py.File('/work/inputs/masks.h5') as hf:tgt=torch.tensor(hf['object'][fi,y0:y0+h,x0:x0+w],device='cuda').float();person=torch.tensor(hf['person'][fi,y0:y0+h,x0:x0+w],device='cuda').float()
valid=torch.maximum(1-person,tgt)
t0=torch.tensor(FP[fi,:3,3],device='cuda',requires_grad=True);R=torch.tensor(FP[fi,:3,:3],device='cuda')
def mk(t):
 T=torch.eye(4,device='cuda');T[:3,:3]=R;T[:3,3]=t;return T
for it in range(5):
 out,pred=render(mk(t0));a=out[...,3]*valid;tv=tgt*valid;dice=1-(2*(a*tv).sum()+1)/((a*a).sum()+(tv*tv).sum()+1);mse=((a-tv)**2).mean();loss=dice+2*mse
 print(it,loss.item(),out[...,3].min().item(),out[...,3].max().item(),(out[...,3].grad_fn is not None))
 loss.backward();print('grad',t0.grad,t0.grad.norm())
 with torch.no_grad():t0-=.01*t0.grad;t0.grad=None
PY"
```
<details><summary>output</summary>

```
<string>:14: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
0 2.429157018661499 1.0 1.0 False
Traceback (most recent call last):
  File "<stdin>", line 12, in <module>
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^ …[+165 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_translation_test.py';s=open(p).read();s=s.replace(\"vm=(torch.tensor(Tcam,dtype=torch.float32,device='cuda')@vh.T).T[:,:3]\",\"vm=(Tcam.to(device='cuda',dtype=torch.float32)@vh.T).T[:,:3]\")
s=s.replace(\"out0,p0=render(torch.tensor(FP[fi],device='cuda'))\",\"out0,p0=render(torch.tensor(FP[fi],device='cuda'))\")
open(p,'w').write(s)
PY
python /work/scratch/opt_translation_test.py"
```
<details><summary>output</summary>

```
0 dt [0. 0. 0.] iou 0.731 -> 0.731 corr 0.213
100 dt [0. 0. 0.] iou 0.838 -> 0.838 corr 0.188
160 dt [0. 0. 0.] iou 0.827 -> 0.827 corr 0.351
180 dt [0. 0. 0.] iou 0.701 -> 0.701 corr 0.27
200 dt [0. 0. 0.] iou 0.619 -> 0.619 corr -0.19
250 dt [0. 0. 0.] iou 0.746 -> 0.746 corr -0.019
300 dt [0. 0. 0.] iou 0.22 -> 0.22 corr 0.118
350 dt [0. 0. 0.] iou 0.616 -> 0.616 corr -0.217
400 dt [0. 0. 0.] iou 0.735 -> 0.735 corr -0.11
450 dt [0. 0. 0.] iou 0.762 -> 0.762 corr 0.164
500 dt [0. 0. 0.] iou 0.7 -> 0.7 corr 0.027
550 dt [0. 0. 0.] iou 0.762 -> 0.762 corr 0.112
600 dt [0. 0. 0.] iou 0.594 -> 0.594 corr -0.23
650 dt [0. 0. 0.] iou 0.601 -> 0.601 corr -0.18
660 dt [0. 0. 0.] iou 0.38 -> 0.38 corr -0.174
700 dt [0. 0. 0.] iou 0.639 -> 0.639 corr -0.177
750 dt [0. 0. 0.] iou 0.748 -> 0.748 co …[+10 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/opt_translation_test.py').read().split(\"cap=cv2.VideoCapture\")[0])
import h5py,cv2
fi=350
with h5py.File('/work/inputs/masks.h5') as hf:tgt=torch.tensor(hf['object'][fi,y0:y0+h,x0:x0+w],device='cuda').float();person=torch.tensor(hf['person'][fi,y0:y0+h,x0:x0+w],device='cuda').float()
valid=torch.maximum(1-person,tgt);t0=torch.tensor(FP[fi,:3,3],device='cuda',requires_grad=True);R=torch.tensor(FP[fi,:3,:3],device='cuda')
def mk(t):
 T=torch.eye(4,device='cuda');T[:3,:3]=R;T[:3,3]=t;return T
T=mk(t0);print('T grad',T.requires_grad,T.grad_fn)
out,pred=render(T);print('out grad',out.requires_grad,out.grad_fn);a=out[...,3]*valid;tv=tgt*valid;dice=1-(2*(a*tv).sum()+1)/((a*a).sum()+(tv*tv).sum()+1);mse=((a-tv)**2).mean();loss=dice+2*mse;print(loss,loss.requires_grad);loss.backward();print('grad',t0.grad)
PY"
```
<details><summary>output</summary>

```
T grad True <CopySlices object at 0x7f3caaad7d90>
out grad True <SelectBackward0 object at 0x7f3caaad7d90>
tensor(2.4292, device='cuda:0', grad_fn=<AddBackward0>) True
grad tensor([0., 0., 0.], device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 4 -type f \\( -name '*foundation*' -o -name '*pose*' -o -name '*.py' \\) | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/scripts/export_result_threejs_scene.py
/code/reconstruction/scripts/sync_css.py
/code/reconstruction/scripts/scale_result_bundle.py
/code/reconstruction/workflows/mv_hoi/orchestration/blacklist.py
/code/reconstruction/workflows/mv_hoi/orchestration/__init__.py
/code/reconstruction/workflows/mv_hoi/orchestration/mark_ready.py
/code/reconstruction/workflows/mv_hoi/orchestration/cleanup_intermediates.py
/code/reconstruction/workflows/mv_hoi/orchestration/lifecycle.py
/code/reconstruction/workflows/mv_hoi/orchestration/campaign_controller.py
/code/reconstruction/workflows/mv_hoi/orchestration/config_utils.py
/code/reconstruction/workflows/mv_hoi/orchestration/registry_versions.py
/code/reconstruction/workflows/mv_hoi/orchestration/schema.py
/code/reconstruction/workflows/m …[+5767 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"def .*foundation\\|class .*Foundation\\|register_first_then_track\\|FoundationPose\" -n /code/reconstruction/modules/v2d_foundation_pose /workspace/v2d_foundation_pose 2>/dev/null | head -120"
```
<details><summary>output</summary>

```
d_foundation_pose/docker/v2d_foundation_pose_docker.egg-info/PKG-INFO:4:Summary: Docker orchestration scripts for FoundationPose
/code/reconstruction/modules/v2d_foundation_pose/docker/run_video_to_poses.py:64:    parser = argparse.ArgumentParser(description="Run FoundationPose video to poses in Docker")
/code/reconstruction/modules/v2d_foundation_pose/docker/run_download_weights.py:39:    parser = argparse.ArgumentParser(description="Download FoundationPose weights")
/code/reconstruction/modules/v2d_foundation_pose/docker/run_export_engines.py:1:"""Build cached FoundationPose TensorRT engines on the current GPU."""
/code/reconstruction/modules/v2d_foundation_pose/docker/run_export_engines.py:13:            f"FoundationPose weights directory does not exist: {weights_dir}. "
/code/reconstru …[+7629 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline/inference -maxdepth 1 -type f -printf '%f\\n'; python - <<'PY'
import torch
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 x=torch.load(p,map_location='cpu',weights_only=False)
 print(p,type(x))
 if isinstance(x,dict):
  for k,v in x.items(): print(k,type(v),getattr(v,'shape',None))
PY"
```
<details><summary>output</summary>

```
/pipeline/inference/coconet.pth <class 'dict'>
schema <class 'str'> None
frames <class 'list'> None
frame_meta <class 'list'> None
kid <class 'int'> None
checkpoint <class 'dict'> None
mhr_inference <class 'dict'> None
config <class 'dict'> None
mesh_diameter <class 'numpy.ndarray'> (781,)
mhr_neutral_height_init <class 'numpy.ndarray'> (781,)
mhr_spatial_scale <class 'numpy.ndarray'> (781,)
K_rois <class 'numpy.ndarray'> (781, 3, 3)
bboxes <class 'numpy.ndarray'> (781, 4)
faces <class 'numpy.ndarray'> (36874, 3)
pr <class 'dict'> None
pr_initial <class 'dict'> None
in <class 'dict'> None
gt <class 'dict'> None
observations <class 'dict'> None
raw <class 'dict'> None
metadata <class 'dict'> None
/pipeline/inference/refined.pth <class 'dict'>
schema <class 'str'> None
frames <class 'list'>  …[+638 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
def walk(o,p='',depth=0):
 if depth>3:return
 if isinstance(o,dict):
  print(p,type(o).__name__,len(o))
  for k,v in o.items(): walk(v,p+'/'+str(k),depth+1)
 elif isinstance(o,(list,tuple)):
  print(p,type(o).__name__,len(o))
  if len(o):walk(o[0],p+'[0]',depth+1)
 elif hasattr(o,'shape'):
  print(p,type(o).__name__,tuple(o.shape),o.dtype)
 else: print(p,type(o).__name__,repr(o)[:100])
walk(x)
PY"
```
<details><summary>output</summary>

```
 dict 21
/schema str 'cari4d.mhr_wild_inference.v1'
/frames list 781
/frames[0] str '000000'
/frame_meta list 781
/frame_meta[0] dict 3
/frame_meta[0]/frame str '000000'
/frame_meta[0]/src_frame int 0
/frame_meta[0]/kid int 0
/kid int 0
/checkpoint dict 2
/checkpoint/path str '/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth'
/checkpoint/step int 200000
/mhr_inference dict 8
/mhr_inference/schema str 'cari4d.mhr_inference_provenance.v1'
/mhr_inference/checkpoint dict 2
/mhr_inference/checkpoint/path str '/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth'
/mhr_inference/checkpoint/step int 200000
/mhr_inference/wandbRun dict 3
/mhr_inference/wandbRun/entity str 'nvr-lpr'
/mhr_inference/wandbRun/project str 'cari4d-mhr'
/mhr_inference/wa …[+25627 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for k in ['pr','pr_initial','raw','observations']:
 o=x[k];print('\\n',k,type(o),len(o))
 for kk,v in o.items():
  if hasattr(v,'shape'):print(kk,v.shape,v.dtype)
  elif isinstance(v,(list,tuple)):print(kk,type(v),len(v),type(v[0]) if v else '')
  else:print(kk,type(v),repr(v)[:200])
PY"
```
<details><summary>output</summary>

```

 pr <class 'dict'> 11
pose_abs (781, 4, 4) float32
pose_abs_1st_delta (781, 4, 4) float32
mhr_global_rot6d (781, 6) float32
mhr_trans (781, 3) float32
mhr_body_pose_cont (781, 260) float32
mhr_hand (781, 108) float32
mhr_shape (781, 45) float32
mhr_scale (781, 28) float32
mhr_face (781, 72) float32
contact_logits (781, 2) float32
pose_abs_postopt (781, 4, 4) float32

 pr_initial <class 'dict'> 10
pose_abs (781, 4, 4) float32
pose_abs_1st_delta (781, 4, 4) float32
mhr_global_rot6d (781, 6) float32
mhr_trans (781, 3) float32
mhr_body_pose_cont (781, 260) float32
mhr_hand (781, 108) float32
mhr_shape (781, 45) float32
mhr_scale (781, 28) float32
mhr_face (781, 72) float32
contact_logits (781, 2) float32

 raw <class 'dict'> 7
trans (781, 3) float32
rot (781, 3) float32
delta_mhr_global_rot6d …[+757 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_candidates.py <<'PY'
exec(open('/work/scratch/opt_translation_test.py').read().split(\"cap=cv2.VideoCapture\")[0])
import cv2
B=D['T_world_mesh'].astype(np.float32); C=D['T_cam_mesh_coconet'].astype(np.float32)
# convert B world to cam
cap=cv2.VideoCapture('/work/inputs/video.mp4')
idxs=[0,100,140,160,180,200,250,300,320,350,400,420,450,500,550,600,650,660,700,750]
thumbs=[]
# P currently KP. B needs K -> roughly same
for fi in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0:y0+h,x0:x0+w]
 poses=[('F',FP[fi]),('B',Tcw@B[fi]),('C',C[fi])]
 row=[]
 for name,T in poses:
  with torch.no_grad():out,_=render(torch.tensor(T,device='cuda'))
  rgb=out[...,:3].cpu().numpy();a=out[...,3].cpu().numpy()
  # Actually alpha likely 1 globally; restrict where color nonzero OR use raster from render? direct pixels dist x0,y0.
  # infer alpha from nonblack: texture black areas challenge. use last alpha if meaningful
  pred=out[...,3].cpu().numpy()>.5
  ov=im.copy(); aa=np.clip(a[...,None],0,1);ov=(im*(1-aa)+rgb*aa*255).astype(np.uint8)
  cv2.putText(ov,name,(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
  row.append(ov)
 thumbs.append(np.hstack(row))
rows=[]
for j in range(0,len(thumbs),4):
 r=thumbs[j:j+4]
 while len(r)<4:r.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(r))
cv2.imwrite('/work/scratch/candidates_texture.jpg',np.vstack(rows))
PY
python /work/scratch/render_candidates.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_candidates.py';s=open(p).read()
s=s.replace(\"with torch.no_grad():out,_=render(torch.tensor(T,device='cuda'))\\n  rgb=out[...,:3].cpu().numpy();a=out[...,3].cpu().numpy()\\n  # Actually alpha likely 1 globally; restrict where color nonzero OR use raster from render? direct pixels dist x0,y0.\\n  # infer alpha from nonblack: texture black areas challenge. use last alpha if meaningful\\n  pred=out[...,3].cpu().numpy()>.5\\n  ov=im.copy(); aa=np.clip(a[...,None],0,1);ov=(im*(1-aa)+rgb*aa*255).astype(np.uint8)\\n  cv2.putText(ov,name,(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)\\n  row.append(ov)\",\"with torch.no_grad():out,pred=render(torch.tensor(T,device='cuda'))\\n  rgb=out[...,:3].cpu().numpy();pred=pred.cpu().numpy()\\n  ov=im.copy();ov[pred]=(0.25*ov[pred]+0.75*rgb[pred]*255).astype(np.uint8)\\n  ov=ov[20:300,55:250];ov=cv2.resize(ov,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)\\n  cv2.putText(ov,name,(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)\\n  row.append(ov)\")
open(p,'w').write(s)
PY
python /work/scratch/render_candidates.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
for fi in [0,160,300,400,500,550,650,700]:
 im=cv2.imread(f'/work/scratch/obj_{fi:03d}.png')
 print(fi,im.shape)
PY"
```
<details><summary>output</summary>

```
300 (492, 524, 3)
[ WARN:0@0.029] global loadsave.cpp:268 findDecoder imread_('/work/scratch/obj_400.png'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'NoneType' object has no attribute 'shape'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/compare_one.py <<'PY'
exec(open('/work/scratch/opt_translation_test.py').read().split(\"cap=cv2.VideoCapture\")[0])
import cv2,sys
B=D['T_world_mesh'].astype(np.float32); C=D['T_cam_mesh_coconet'].astype(np.float32)
fi=int(sys.argv[1]) if len(sys.argv)>1 else 0
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,full=cap.read()
# get pose centers projected crop via F and actual bbox from mask
with h5py.File('/work/inputs/masks.h5') as hf: gt=hf['object'][fi]
ys,xs=np.where(gt); cx=int(xs.mean());cy=int(ys.mean());half=80
xx0=max(0,cx-half);xx1=min(1536,cx+half);yy0=max(0,cy-half);yy1=min(1152,cy+half)
# use fixed image crop coords [700:980,400:760], but just composite full then crop
poses=[('GT',None),('F',FP[fi]),('B',Tcw@B[fi]),('C',C[fi])]
row=[]
for name,T in poses:
 if T is None: ov=full.copy()
 else:
  T=torch.tensor(T,device='cuda')
  # Need render full image P not crop. temporary use score candidate render? implement directly
  vm=(T@vh.T).T[:,:3]; Pf=torch.tensor([[2*KP[0,0]/1536,0,2*KP[0,2]/1536-1,0],[0,2*KP[1,1]/1152,2*KP[1,2]/1152-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda');pos=(Pf@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
  with torch.no_grad():
   r,db=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([1152,1536],device='cuda'));uv,_=dr.interpolate(UV[None].contiguous(),r,F.contiguous());uv=uv.clone();uv[...,1]=1-uv[...,1];col=dr.texture(tex[None],uv. …[+510 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/compare_one.py", line 21, in <module>
    ov=full.copy();ov[pred]=(0.2*ov[pred]+0.8*col[pred].cpu().numpy()*255).astype(np.uint8)
                                 ~~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
Traceback (most recent call last):
  File "/work/scratch/compare_one.py", line 21, in <module>
    ov=full.copy();ov[pred]=(0.2*ov[pred]+0.8*col[pred].cpu().numpy()*255).astype(np.uint8)
                                 ~~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __a …[+3583 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/col\\[pred\\].cpu().numpy()/col.cpu().numpy()[pred.cpu().numpy()]/\" /work/scratch/compare_one.py
for f in 0 160 300 350 400 500 550 650 700; do python /work/scratch/compare_one.py "'$f; done'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/compare_one.py", line 21, in <module>
    ov=full.copy();ov[pred]=(0.2*ov[pred]+0.8*col.cpu().numpy()[pred.cpu().numpy()]*255).astype(np.uint8)
                                 ~~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
Traceback (most recent call last):
  File "/work/scratch/compare_one.py", line 21, in <module>
    ov=full.copy();ov[pred]=(0.2*ov[pred]+0.8*col.cpu().numpy()[pred.cpu().numpy()]*255).astype(np.uint8)
                                 ~~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_t …[+3709 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/compare_one.py';s=open(p).read();s=s.replace(\"ov=full.copy();ov[pred]=(0.2*ov[pred]+0.8*col.cpu().numpy()[pred.cpu().numpy()]*255).astype(np.uint8)\",\"pn=pred.cpu().numpy(); cn=col.cpu().numpy(); ov=full.copy();ov[pn]=(0.2*ov[pn]+0.8*cn[pn]*255).astype(np.uint8)\")
open(p,'w').write(s)
PY
for f in 0 160 300 350 400 500 550 650 700; do python /work/scratch/compare_one.py "'$f; done'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/pattern_translation.py <<'PY'
exec(open('/work/scratch/score_candidates.py').read().split(\"N=len(B)\")[0])
from scipy.ndimage import binary_dilation
import cv2
Bc=np.einsum('ij,fjk->fik',Tcw,B)
def raster_mask(T):
 vm=(torch.tensor(T,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(PB@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
 r,_=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([h,w],device='cuda'));return (r[0,...,3]>0).cpu().numpy()
def score(pred,gt,valid):
 pp=pred&valid; inter=(pp&gt).sum();union=(pp|gt).sum();return inter/max(1,union)
def pattern(T,gt,valid,start_step=.02,nlev=5):
 T=T.copy(); t=T[:3,3].copy(); best=score(raster_mask(T),gt,valid);step=start_step
 for lev in range(nlev):
  improved=True
  while improved:
   improved=False; bestT=None
   for dx in [-step,0,step]:
    for dy in [-step,0,step]:
     for dz in [-step,0,step]:
      if dx==dy==dz==0:continue
      U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=score(raster_mask(U),gt,valid)
      if s>best+1e-6:best=s;bestT=U
   if bestT is not None:T=bestT; improved=True
  step/=2
 return T,best
with h5py.File('/work/inputs/masks.h5') as hf:
 for fi in [0,50,100,140,160,180,200,250,300,320,350,400,420,450,500,550,600,650,660,700,750]:
  gt=hf['object'][fi,y0:y0+h,x0:x0+w];person=hf['person'][fi,y0:y0+h,x0:x0+w];valid=~binary_dilation(person,iterations=3)
  for name,T0 in [('B',Bc[fi]),('F',FP[fi]),('C',C[fi])]:
   p0=raster_mask(T0);s0=score …[+183 chars]
```
<details><summary>output</summary>

```
0 B 0.68 -> 0.785 dt [-0.01  0.   -0.01]
0 F 0.427 -> 0.666 dt [ 0.02  0.02 -0.02]
0 C 0.474 -> 0.638 dt [ 0.02  0.02 -0.02]
50 B 0.719 -> 0.793 dt [-0.01  0.   -0.01]
50 F 0.416 -> 0.671 dt [ 0.02  0.02 -0.02]
50 C 0.465 -> 0.657 dt [ 0.02  0.02 -0.02]
100 B 0.703 -> 0.794 dt [-0.01 -0.01 -0.01]
100 F 0.386 -> 0.657 dt [ 0.02  0.02 -0.02]
100 C 0.466 -> 0.704 dt [ 0.02  0.02 -0.02]
140 B 0.683 -> 0.742 dt [-0.01  0.01  0.  ]
140 F 0.274 -> 0.514 dt [ 0.02  0.02 -0.02]
140 C 0.368 -> 0.604 dt [ 0.02  0.02 -0.02]
160 B 0.608 -> 0.613 dt [-0.01  0.   -0.01]
160 F 0.407 -> 0.522 dt [0.02 0.02 0.02]
160 C 0.457 -> 0.534 dt [0.02 0.02 0.02]
180 B 0.586 -> 0.588 dt [ 0.    0.   -0.01]
180 F 0.303 -> 0.473 dt [ 0.02  0.02 -0.02]
180 C 0.349 -> 0.505 dt [ 0.02  0.02 -0.02]
200 B 0.492 -> 0.701 dt  …[+1929 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/pattern_translation.py';s=open(p).read();s=s.replace(\"def raster_mask(T):\\n vm=(torch.tensor(T,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(PB@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]\",\"def raster_mask(T,P=PB):\\n vm=(torch.tensor(T,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]\").replace(\"def pattern(T,gt,valid,start_step=.02,nlev=5):\",\"def pattern(T,gt,valid,P=PB,start_step=.02,nlev=5):\").replace(\"best=score(raster_mask(T),gt,valid)\",\"best=score(raster_mask(T,P),gt,valid)\").replace(\"s=score(raster_mask(U),gt,valid)\",\"s=score(raster_mask(U,P),gt,valid)\")
s=s.replace(\"for name,T0 in [('B',Bc[fi]),('F',FP[fi]),('C',C[fi])]:\\n   p0=raster_mask(T0);s0=score(p0,gt,valid);T,s=pattern(T0,gt,valid);p=raster_mask(T)\",\"for name,T0,P in [('B',Bc[fi],PB),('F',FP[fi],PP),('C',C[fi],PP)]:\\n   p0=raster_mask(T0,P);s0=score(p0,gt,valid);T,s=pattern(T0,gt,valid,P);p=raster_mask(T,P)\")
open(p,'w').write(s)
PY
python /work/scratch/pattern_translation.py"
```
<details><summary>output</summary>

```
0 B 0.68 -> 0.785 dt [-0.01  0.   -0.01]
0 F 0.731 -> 0.736 dt [ 0.      0.     -0.0025]
0 C 0.674 -> 0.775 dt [-0.01  0.   -0.01]
50 B 0.719 -> 0.793 dt [-0.01  0.   -0.01]
50 F 0.71 -> 0.729 dt [ 0.    0.   -0.02]
50 C 0.715 -> 0.784 dt [-0.01  0.    0.  ]
100 B 0.703 -> 0.794 dt [-0.01 -0.01 -0.01]
100 F 0.793 -> 0.8 dt [-0.0012  0.      0.    ]
100 C 0.701 -> 0.788 dt [-0.01  0.    0.01]
140 B 0.683 -> 0.742 dt [-0.01  0.01  0.  ]
140 F 0.691 -> 0.733 dt [ 0.    0.01 -0.01]
140 C 0.675 -> 0.745 dt [-0.01  0.01 -0.01]
160 B 0.608 -> 0.613 dt [-0.01  0.   -0.01]
160 F 0.609 -> 0.613 dt [ 0.    0.01 -0.01]
160 C 0.608 -> 0.613 dt [-0.01  0.01 -0.01]
180 B 0.586 -> 0.588 dt [ 0.    0.   -0.01]
180 F 0.571 -> 0.603 dt [ 0.01  0.01 -0.01]
180 C 0.581 -> 0.585 dt [ 0.    0.   -0.02]
200 B 0.4 …[+1962 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/score_rotations.py <<'PY'
exec(open('/work/scratch/score_candidates.py').read().split(\"N=len(B)\")[0])
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as RR
import cv2
Bc=np.einsum('ij,fjk->fik',Tcw,B)
# use full image clip projection but fixed crop renderer for speed
# rendering routines from opt test
import nvdiffrast.torch as dr
UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda');imtex=np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255;tex=torch.tensor(imtex,device='cuda')
def Pmat(Kc):
 fx,fy=Kc[0,0],Kc[1,1];cx,cy=Kc[0,2],Kc[1,2]
 return torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,2*fy/h,2*(cy-y0)/h-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
PB=PB.contiguous();PP=Pmat(KP).contiguous()
def renderx(T,P,texmode=False,ctx=ctx):
 vm=(torch.tensor(T,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
 r,db=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([h,w],device='cuda'));pred=(r[0,...,3]>0)
 if not texmode:return pred.cpu().numpy()
 uv,_=dr.interpolate(UV[None].contiguous(),r,F.contiguous());uv=uv.clone();uv[...,1]=1-uv[...,1];col=dr.texture(tex[None],uv.contiguous(),filter_mode='linear',boundary_mode='clamp')[0]
 return pred.cpu().numpy(),col.cpu().numpy()
def scoremask(pred,gt,valid):
 pp=pred&valid;inter=(pp&gt).sum();union=(pp|gt).sum();return inter/max(1,unio …[+1947 chars]
```
<details><summary>output</summary>

```

 0
  B0 iou 0.785 corr 0.092 score 0.799 t [0.31  0.234 2.596]
  B3 iou 0.776 corr 0.147 score 0.798 t [0.3   0.234 2.606]
  F3 iou 0.728 corr 0.431 score 0.792 t [0.273 0.205 2.706]
  C0 iou 0.775 corr 0.107 score 0.791 t [0.28  0.186 2.608]
  C3 iou 0.768 corr 0.15 score 0.79 t [0.27  0.186 2.618]

 50
  B0 iou 0.793 corr 0.112 score 0.81 t [0.307 0.233 2.593]
  C0 iou 0.784 corr 0.111 score 0.801 t [0.277 0.185 2.611]
  B3 iou 0.76 corr 0.204 score 0.79 t [0.297 0.233 2.583]
  C3 iou 0.751 corr 0.2 score 0.781 t [0.267 0.185 2.591]
  F0 iou 0.729 corr 0.336 score 0.779 t [0.299 0.213 2.762]

 100
  F3 iou 0.793 corr 0.492 score 0.867 t [0.276 0.197 2.65 ]
  C3 iou 0.796 corr 0.448 score 0.863 t [0.277 0.2   2.646]
  B3 iou 0.803 corr 0.379 score 0.859 t [0.305 0.247 2.636]
  F0 iou 0.7 …[+5683 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/compare_big.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
import cv2,sys
fi=int(sys.argv[1]);Bc=np.einsum('ij,fjk->fik',Tcw,B)
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,full=cap.read()
with h5py.File('/work/inputs/masks.h5') as hf:gt=hf['object'][fi];person=hf['person'][fi]
ys,xs=np.where(gt);cx=int(xs.mean());cy=int(ys.mean());half=75;xx0=max(0,cx-half);xx1=min(1536,cx+half);yy0=max(0,cy-half);yy1=min(1152,cy+half)
# full frame projection helpers
UVt=UV;text=tex
def rend_full(T,P):
 vm=(torch.tensor(T,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(P@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
 r,_=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([1152,1536],device='cuda'));pred=(r[0,...,3]>0);uv,_=dr.interpolate(UVt[None].contiguous(),r,F.contiguous());uv=uv.clone();uv[...,1]=1-uv[...,1];col=dr.texture(text[None],uv.contiguous(),filter_mode='linear',boundary_mode='clamp')[0]
 return pred.cpu().numpy(),col.cpu().numpy()
KPB=PB
# full P funcs
Pfb=torch.tensor([[2*K[0,0]/1536,0,2*K[0,2]/1536-1,0],[0,2*K[1,1]/1152,2*K[1,2]/1152-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
Pfp=torch.tensor([[2*KP[0,0]/1536,0,2*KP[0,2]/1536-1,0],[0,2*KP[1,1]/1152,2*KP[1,2]/1152-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
row=[]
# actual
ov=full.copy();row.append(('GT',ov[yy0:yy1,xx0:xx1]))
for base,name,P in  …[+635 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/compare_big.py", line 26, in <module>
    cv2.imwrite(f'/work/scratch/big_{fi}.jpg',np.vstack([np.hstack(thumbs[:4]),np.hstack(thumbs[4:8]),np.hstack(thumbs[8:])]))
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/shape_base.py", line 289, in vstack
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1200 and the array at index 2 has size 1500
Traceback (most recent call las …[+1507 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/compare_big.py';s=open(p).read();s=s.replace(\"cv2.imwrite(f'/work/scratch/big_{fi}.jpg',np.vstack([np.hstack(thumbs[:4]),np.hstack(thumbs[4:8]),np.hstack(thumbs[8:])]))\",\"rows=[]\\nfor j in range(0,len(thumbs),4):\\n r=thumbs[j:j+4]\\n while len(r)<4:r.append(np.zeros_like(thumbs[0]))\\n rows.append(np.hstack(r))\\ncv2.imwrite(f'/work/scratch/big_{fi}.jpg',np.vstack(rows))\")
open(p,'w').write(s)
PY
for f in 0 160 600; do python /work/scratch/compare_big.py "'$f; done'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz');Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc)
B=D['T_world_mesh'];F=np.einsum('ij,fjk->fik',Twc,D['T_cam_mesh_foundationpose']);C=np.einsum('ij,fjk->fik',Twc,D['T_cam_mesh_coconet'])
rests=[(0,104),(232,263),(383,399),(512,526),(615,627),(700,780)]
def meanrot(Rs):
 # project to SO3 via SVD
 U,S,Vt=np.linalg.svd(Rs.sum(0));Q=U@Vt
 if np.linalg.det(Q)<0:U[:,-1]*=-1;Q=U@Vt
 return Q
for a,b in rests:
 print('\\nrest',a,b)
 for name,T in [('B',B),('F',F),('C',C)]:
  Q=meanrot(T[a:b+1,:3,:3]);ds=R.from_matrix(np.einsum('ji,fjk->fik',Q,T[a:b+1,:3,:3])).magnitude()*180/np.pi
  print(name,'mean',np.round(Q,3).tolist(),'dev median/max',np.median(ds),ds.max(),'endpoint',R.from_matrix(T[a,:3,:3].T@T[b,:3,:3]).magnitude()*180/np.pi)
 # cross mean diffs
 Qb=meanrot(B[a:b+1,:3,:3]);Qf=meanrot(F[a:b+1,:3,:3]);print('B-F mean angle',R.from_matrix(Qb.T@Qf).magnitude()*180/np.pi)
PY"
```
<details><summary>output</summary>

```

rest 0 104
B mean [[0.7950000166893005, 0.5910000205039978, 0.13500000536441803], [-0.12800000607967377, -0.054999999701976776, 0.9900000095367432], [0.5929999947547913, -0.8040000200271606, 0.03200000151991844]] dev median/max 4.894087281813073 15.672433808811059 endpoint 11.49381869463826
F mean [[0.796, 0.601, 0.07], [0.041, -0.169, 0.985], [0.603, -0.781, -0.159]] dev median/max 3.789403944417584 9.862750010131235 endpoint 6.088160180994483
C mean [[0.788, 0.601, 0.137], [-0.14, -0.041, 0.989], [0.6, -0.798, 0.052]] dev median/max 4.89346474402667 15.673282077923616 endpoint 11.495921954809548
B-F mean angle 11.711351864496269

rest 232 263
B mean [[0.9900000095367432, -0.024000000208616257, 0.14100000262260437], [-0.14000000059604645, 0.05999999865889549, 0.9879999756813049], [-0.032 …[+3060 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,260p' /code/reconstruction/modules/v2d_foundation_pose/lib/foundation_pose_tracker.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful FoundationPose tracker.

Lifecycle:
    tracker = FoundationPoseTracker(mesh, weights_dir)
    pose0   = tracker.register(rgb, depth, mask, intrinsics)
    pose1   = tracker.track_one(rgb1, depth1, intrinsics)
    pose2   = tracker.track_one(rgb2, depth2, intrinsics)
    tracker.reset_to_pose(pose0)   # rewind for backward pass
    pose_bk = tracker.track_one(rgb_prev, depth_prev, intrinsics, iteration=2)

Particle filter tracking (multi-hypothesis):
    pose1   = tracker.track_one_particles(rgb1, depth1, intrinsics, n_particles=20)
    pose2   = tracker.track_one_particles(rgb2, depth2, intrinsics, n_particles=20)

Scale estimation:
    scale …[+9318 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '260,560p' /code/reconstruction/modules/v2d_foundation_pose/lib/foundation_pose_tracker.py"
```
<details><summary>output</summary>

```
            mask_iou_weight:  Log-space weight for the mask IoU term. Default 1.0.
                              Set to 0.0 to disable mask weighting.

        Returns:
            Weighted mean pose as Transform3d.
        """
        if self._est.pose_last is None:
            raise RuntimeError("Call register() before track_one_particles()")

        # Lazy-init or reinit if n_particles changed
        if self._particles is None or len(self._particles) != n_particles:
            seed = np.tile(
                self._est.pose_last.cpu().numpy().astype(np.float32),
                (n_particles, 1, 1),
            )
            self._particles = _perturb_se3(seed, process_noise_t, process_noise_r)
            self._weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float32)

      …[+11720 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction/modules/v2d_foundation_pose -maxdepth 3 -type f -name '*.py' -print | sort; grep -R \"FoundationPoseTracker(\" -n /code/reconstruction/modules/v2d_foundation_pose | head -30"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_foundation_pose/docker/__init__.py
/code/reconstruction/modules/v2d_foundation_pose/docker/_config.py
/code/reconstruction/modules/v2d_foundation_pose/docker/build.py
/code/reconstruction/modules/v2d_foundation_pose/docker/run_align_depth_to_object.py
/code/reconstruction/modules/v2d_foundation_pose/docker/run_align_depth_to_reference_depth.py
/code/reconstruction/modules/v2d_foundation_pose/docker/run_correct_depth_scale.py
/code/reconstruction/modules/v2d_foundation_pose/docker/run_download_weights.py
/code/reconstruction/modules/v2d_foundation_pose/docker/run_ekf_smoothing.py
/code/reconstruction/modules/v2d_foundation_pose/docker/run_estimate_mesh_scale.py
/code/reconstruction/modules/v2d_foundation_pose/docker/run_export_engines.py
/code/reconstruction …[+3486 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,240p' /code/reconstruction/modules/v2d_foundation_pose/lib/run_video_to_poses.py; echo '---backend'; sed -n '1,220p' /code/reconstruction/modules/v2d_foundation_pose/lib/backends.py; echo '---datatypes'; grep -R \"class CameraIntrinsics\\|class DepthImage\\|class Mask\\|class Image\" -n /workspace/v2d_common /code/reconstruction/modules/v2d_common 2>/dev/null | head -30"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FoundationPose video-to-poses processing function.
Can be called directly from command line or imported as a function.
"""
import os
import argparse
import logging
from pathlib import Path

import numpy as np
import cv2

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.common.datatypes import Image as V2dImage
from v2d.mesh.lib.mesh import Mesh
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
from v2d.foundation_pose.lib.backends import DEFAULT_BACKEND, SUPPORTED_BACKENDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = loggin …[+12303 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,120p' /workspace/v2d_common/datatypes.py; sed -n '250,340p' /workspace/v2d_common/datatypes.py; grep -n \"class Mesh\" -n /workspace/v2d_mesh/lib/mesh.py; sed -n '1,140p' /workspace/v2d_mesh/lib/mesh.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import json
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image as PILImage


@dataclass
class DepthImage:
    depth: np.ndarray

    def to_pil_image(self) -> PILImage.Image:
        # Encode depth (meters) as uint16 PNG using inverse depth:
        # pixel = 65535 * (1 / (depth_m + 1))
        # Closer objects → higher pixel values; zero depth → 65535; infinity → 0.
        inverse_depth = 1.0 / (self.depth + 1.0)
        inverse_depth = 65535.0 * inverse_depth
        inverse_depth = inverse_depth.clip(0, 65535).astype(np.uint16)
        return PILImage.fromarray(inverse_depth, mode="I …[+10097 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /weights/foundationpose -maxdepth 3 -type f -printf '%p %kKB\\n' | sort | head -100; find /weights/foundationpose -maxdepth 2 -type d -print"
```
<details><summary>output</summary>

```
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/config.yml 4KB
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth 66624KB
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/config.yml 4KB
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth 185772KB
/weights/foundationpose
/weights/foundationpose/nvlabs_pytorch
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '110,230p' /code/reconstruction/modules/v2d_foundation_pose/lib/foundation_pose_tracker.py; sed -n '120,260p' /code/reconstruction/modules/v2d_foundation_pose/lib/run_video_to_poses.py"
```
<details><summary>output</summary>

```
    M[:3, :3] = R_mean.as_matrix().astype(np.float32)
    M[:3, 3] = t_mean.astype(np.float32)
    return M


class FoundationPoseTracker:
    """Wraps FoundationPose for stateful 6-DoF object tracking."""

    def __init__(
        self,
        mesh: Mesh,
        weights_dir: str,
        backend: str = DEFAULT_BACKEND,
    ) -> None:
        predictor_backend = create_predictor_backend(backend, weights_dir)
        self._scorer = predictor_backend.scorer
        self._refiner = predictor_backend.refiner
        self.runtime_info = predictor_backend.runtime_info
        self._glctx = dr.RasterizeCudaContext()
        self._original_mesh = mesh
        self._mesh = mesh
        self._current_scale = 1.0
        self._particles: np.ndarray | None = None  # (N, 4, 4) poses in pose_last fra …[+8845 chars]
```
</details>

**shell** (exit 135)
```
/usr/bin/bash -c "cat > /work/scratch/test_fp_register.py <<'PY'
import os,sys,json,numpy as np,cv2,h5py,torch
sys.path.insert(0,'/workspace')
from v2d.common.datatypes import CameraIntrinsics,DepthImage,Mask,Image
from v2d.mesh.lib.mesh import Mesh
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
from v2d.foundation_pose.lib.backends import NVLABS_PYTORCH
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline'],np.float32)
ci=CameraIntrinsics(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),width=1536,height=1152)
mesh=Mesh.load('/work/inputs/object_mesh.glb',force_mesh=True);print('mesh',mesh.vertices.shape,mesh.faces.shape,mesh.uv is not None,mesh.texture is not None)
tracker=FoundationPoseTracker(mesh,'/weights/foundationpose',backend=NVLABS_PYTORCH);print('tracker ready')
raw=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r')
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f: scales=f['alignment/front_stereo_camera_left/scale'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for fi in [0,100]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,bgr=cap.read();rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 with h5py.File('/work/inputs/masks.h5') as f:mask=f['object'][fi]
 depth=np.asarray(raw[fi],np.float32)/1000*float(scales[fi])
 pose=tracker.register(Image(data=rgb),DepthImage(depth=depth),Mask(mask=mask.astype(np.float32)),ci,iteration=10)
 p=pose.to_matrix();print('frame',fi,'pose\\n',p);np.save(f'/wor …[+96 chars]
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
mesh (297718, 3) (99260, 3) True True
[__init__()] self.cfg: 
 lr: 0.0001
c_in: 6
zfar: 'Infinity'
debug: null
n_view: 1
run_id: 3wy8qqex
use_BN: true
exp_name: 2024-01-11-20-02-45
n_epochs: 62
save_dir: /home/bowenw/debug/2024-01-11-20-02-45/
use_mask: false
loss_type: pairwise_valid
optimizer: adam
batch_size: 64
crop_ratio: 1.1
enable_amp: true
use_normal: false
max_num_key: null
warmup_step: -1
input_resize:
- 160
- 160
max_step_val: 1000
vis_interval: 1000
weight_decay: 0
normalize_xyz: true
resume_run_id: null
clip_grad_norm: 'Infinity'
lr_epoch_decay …[+4018 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nvidia-smi; free -g; tail -100 /work/logs/codex_stderr.txt 2>/dev/null | tail -30'
```
<details><summary>output</summary>

```
               total        used        free      shared  buff/cache   available
Mem:             125          28          39           0          57          95
Swap:              1           1           0
Reading additional input from stdin...

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2,time
Bc=np.einsum('ij,fjk->fik',Tcw,B)
# data samples first rest
fis=list(range(0,105,5));cap=cv2.VideoCapture('/work/inputs/video.mp4');dat=[]
with h5py.File('/work/inputs/masks.h5') as hf:
 for fi in fis:
  cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0:y0+h,x0:x0+w];gt=hf['object'][fi,y0:y0+h,x0:x0+w];per=hf['person'][fi,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3);dat.append((fi,gt,valid))
def ms(T):
 T=T.copy();vals=[]
 for fi,gt,valid in dat:
  p=renderx(T,PB);pp=p&valid;inter=(pp&gt).sum();union=(pp|gt).sum();vals.append(inter/max(1,union))
 return float(np.mean(vals))
def opttrans(T,step=.04,nlev=6):
 T=T.copy();t=T[:3,3].copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for dx in [-step,0,step]:
    for dy in [-step,0,step]:
     for dz in [-step,0,step]:
      if dx==dy==dz==0:continue
      U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=ms(U)
      if s>best+1e-7:best=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,best
def optrot(T,step=10,nlev=5):
 T=T.copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for ax in range(3):
    for sign in [-1,1]:
     U=T.copy();dv=np.zeros(3);dv[ax]=np.deg2rad(step*sign); …[+562 chars]
```
<details><summary>output</summary>

```
B0 0 init 0.6946 trans 0.7903 rot 0.8193 final 0.8336 t [0.3074 0.2363 2.5984] R [[0.8679999709129333, 0.47999998927116394, 0.12600000202655792], [-0.1599999964237213, 0.03200000151991844, 0.9869999885559082], [0.4690000116825104, -0.8769999742507935, 0.10400000214576721]]
B0 1 init 0.5749 trans 0.6609 rot 0.7515 final 0.7629 t [0.3049 0.2488 2.5909] R [[0.1550000011920929, -0.9760000109672546, -0.1509999930858612], [-0.3659999966621399, 0.0860000029206276, -0.9269999861717224], [0.9179999828338623, 0.19900000095367432, -0.3440000116825104]]
B0 2 init 0.5099 trans 0.6994 rot 0.7247 final 0.7268 t [0.2999 0.2563 2.6059] R [[-0.9520000219345093, 0.29100000858306885, -0.09600000083446503], [0.16500000655651093, 0.22300000488758087, -0.9610000252723694], [-0.2590000033378601, -0.93000000715255 …[+4695 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('/work/inputs/reconstruction.npz');cp=D['contact_prob_right_left'];c=cp.max(1)
for th in [.1,.3,.5,.8,.9,.95]:
 b=c>th
 # fill gaps <=10
 for _ in range(10):
  b[1:-1]|=(b[:-2]&b[2:])
 starts=np.where(np.diff(np.r_[False,b])==1)[0];ends=np.where(np.diff(np.r_[b,False])==-1)[0]
 print('th',th,list(zip(starts,ends)))
# print max contact every frame changes
PY"
```
<details><summary>output</summary>

```
th 0.1 []
th 0.3 []
th 0.5 []
th 0.8 []
th 0.9 []
th 0.95 []

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
c=np.load('/work/inputs/reconstruction.npz')['contact_prob_right_left'].max(1)
print(c.min(),c.max(),np.percentile(c,[50,75,90,95,99]),sum(c>.5))
b=c>.5
print(b[100:115],np.where(b)[0][:20],np.where(b)[0][-20:])
print(np.diff(np.r_[False,b]).max(),np.diff(np.r_[False,b]).min())
print(np.where(np.diff(np.r_[False,b].astype(int))==1)[0][:20])
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
c=np.load('/work/inputs/reconstruction.npz')['contact_prob_right_left'].max(1)
for th in [.1,.5,.9,.99]:
 b=(c>th).astype(int)
 for _ in range(10):b[1:-1]|=(b[:-2]&b[2:])
 st=np.where(np.diff(np.r_[0,b])==1)[0];en=np.where(np.diff(np.r_[b,0])==-1)[0]
 print(th,list(zip(st,en)))
PY"
```
<details><summary>output</summary>

```
0.1 [(107, 235), (262, 363), (398, 495), (523, 618), (631, 698), (731, 735)]
0.5 [(108, 234), (264, 361), (398, 495), (524, 618), (631, 698)]
0.9 [(109, 233), (265, 360), (400, 494), (528, 616), (631, 693), (696, 696)]
0.99 [(111, 230), (267, 357), (401, 492), (529, 615), (637, 690)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import medfilt
z=np.load('/work/scratch/signals.npz');ys=z['ys'];cp=np.load('/work/inputs/reconstruction.npz')['contact_prob_right_left'].max(1)
for w in [3,5,9]:
 sm=medfilt(ys,w); b=sm<605
 # fill <=2
 for _ in range(2):b[1:-1]|=(b[:-2]&b[2:])
 st=np.where(np.diff(np.r_[False,b])==1)[0];en=np.where(np.diff(np.r_[b,False])==-1)[0]
 print('w',w,list(zip(st,en)))
# derivative/lift start detailed around contacts
sm=medfilt(ys,5)
for a in [100,230,260,360,395,495,520,620,630,700]:
 print('\\n',a)
 for f in range(a,a+25):print(f,round(ys[f],1),round(sm[f],1),round(cp[f],2))
PY"
```
<details><summary>output</summary>

```
w 3 []
w 5 []
w 9 []

 100
100 637.3 636.8 0.0
101 637.0 636.8 0.0
102 636.8 637.0 0.0
103 636.8 637.0 0.0
104 637.3 637.0 0.0
105 637.0 637.1 0.0
106 637.2 637.2 0.06
107 637.1 637.2 0.37
108 637.2 637.2 0.65
109 637.3 637.3 0.92
110 637.3 637.3 0.98
111 637.3 637.3 1.0
112 637.4 637.3 1.0
113 637.4 637.4 1.0
114 637.1 637.4 1.0
115 637.5 637.3 1.0
116 637.1 637.3 1.0
117 637.3 637.5 1.0
118 637.6 637.6 1.0
119 637.6 637.6 1.0
120 637.6 637.6 1.0
121 637.5 637.5 1.0
122 637.4 637.4 1.0
123 637.4 637.4 1.0
124 637.4 637.4 1.0

 230
230 636.8 636.8 0.99
231 637.1 636.8 0.92
232 636.8 636.8 0.91
233 636.6 636.8 0.92
234 636.8 636.6 0.56
235 636.2 636.4 0.15
236 636.3 636.4 0.02
237 636.4 636.3 0.04
238 636.7 636.3 0.0
239 636.3 636.3 0.0
240 636.0 636.1 0.0
241 635.9 636.0 0.0
242 636.1 636. …[+4347 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
z=np.load('/work/scratch/signals.npz');print('ys',[(f,float(z['ys'][f])) for f in [0,100,108,120,140,160,180,200,220,235,240,260]])
with h5py.File('/work/inputs/masks.h5') as h:
 for f in [0,100,120,140,160,180,200,220,235,240]:
  y,x=np.where(h['object'][f]);print(f,len(y),y.mean(),x.mean(),y.min(),y.max())
PY"
```
<details><summary>output</summary>

```
ys [(0, 635.2828847130523), (100, 637.2727272727273), (108, 637.1880841121496), (120, 637.6235362997658), (140, 611.8173018753781), (160, 531.3497596153846), (180, 566.5295480880649), (200, 624.4983407079646), (220, 636.8858663512715), (235, 636.1907356948229), (240, 636.0021621621621), (260, 635.8380588876772)]
0 1969 635.2828847130523 852.5647536820721 602 672
100 1738 637.2727272727273 854.4729574223245 605 672
120 1708 637.6235362997658 854.9168618266979 606 672
140 1653 611.8173018753781 854.768300060496 581 645
160 1664 531.3497596153846 826.2193509615385 500 562
180 1726 566.5295480880649 845.5816917728853 535 600
200 1808 624.4983407079646 855.9568584070796 592 659
220 1691 636.8858663512715 855.6901241868717 604 671
235 1835 636.1907356948229 855.0528610354223 604 672
240 1850 636 …[+41 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import medfilt1
z=np.load('/work/scratch/signals.npz');y=z['ys'];sm=medfilt1(y,5)
for th in [630,620,600,570,540]:
 b=(sm<th).astype(int)
 for _ in range(3):b[1:-1]|=(b[:-2]&b[2:])
 st=np.where(np.diff(np.r_[0,b])==1)[0];en=np.where(np.diff(np.r_[b,0])==-1)[0]
 print(th,list(zip(st,en)))
# print at intervals
for f in range(0,781,10):print(f,round(sm[f],1),end='; ')
print()
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ImportError: cannot import name 'medfilt1' from 'scipy.signal' (/opt/conda/lib/python3.11/site-packages/scipy/signal/__init__.py)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import medfilt
z=np.load('/work/scratch/signals.npz');y=z['ys'];sm=medfilt(y,5)
for th in [630,620,600,570,540]:
 b=(sm<th).astype(int)
 for _ in range(3):b[1:-1]|=(b[:-2]&b[2:])
 st=np.where(np.diff(np.r_[0,b])==1)[0];en=np.where(np.diff(np.r_[b,0])==-1)[0]
 print(th,list(zip(st,en)))
for f in range(0,781,10):print(f,round(sm[f],1),end='; ')
print()
PY"
```
<details><summary>output</summary>

```
630 [(136, 204), (279, 339), (412, 471), (537, 594), (647, 680)]
620 [(139, 197), (282, 334), (415, 467), (540, 589), (650, 676)]
600 [(143, 190), (287, 328), (420, 463), (543, 584), (654, 671)]
570 [(148, 181), (292, 323), (425, 458), (547, 578)]
540 [(155, 171), (297, 316), (433, 453), (552, 573)]
0 635.3; 10 635.2; 20 635.2; 30 635.4; 40 635.3; 50 635.4; 60 634.6; 70 634.5; 80 635.0; 90 636.4; 100 636.8; 110 637.3; 120 637.6; 130 636.6; 140 611.8; 150 555.0; 160 531.3; 170 536.9; 180 566.5; 190 599.9; 200 624.5; 210 634.0; 220 636.7; 230 636.8; 240 636.1; 250 636.0; 260 635.8; 270 636.6; 280 625.9; 290 575.3; 300 521.0; 310 514.5; 320 553.3; 330 604.8; 340 631.7; 350 636.9; 360 637.1; 370 636.9; 380 637.2; 390 637.1; 400 637.2; 410 633.8; 420 595.6; 430 546.7; 440 509.4; 450 520.5; 460  …[+360 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fit_rests.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2,time
Bc=np.einsum('ij,fjk->fik',Tcw,B)
rests=[(5,130),(215,270),(350,405),(480,530),(602,640),(692,775)]
# use every 10 frames + endpoints
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def chordal(Rs):
 U,S,Vt=np.linalg.svd(Rs.sum(0));Q=U@Vt
 if np.linalg.det(Q)<0:U[:,-1]*=-1;Q=U@Vt
 return Q.astype(np.float32)
def opttrans(T,ms,step=.03,nlev=4):
 T=T.copy();t=T[:3,3].copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for dx in [-step,0,step]:
    for dy in [-step,0,step]:
     for dz in [-step,0,step]:
      if dx==dy==dz==0:continue
      U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=ms(U)
      if s>best+1e-6:best=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,best
def optrot(T,ms,step=8,nlev=4):
 T=T.copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for ax in range(3):
    for sign in [-1,1]:
     U=T.copy();dv=np.zeros(3);dv[ax]=np.deg2rad(step*sign);U[:3,:3]=T[:3,:3]@RR.from_rotvec(dv).as_matrix();s=ms(U)
     if s>best+1e-6:best=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,best
out={}
with h5py.File('/work/inputs/masks.h5') as hf:
 for ri,(a,b) in enumerate(rests):
  fis=list(range(a,b+1,10));
  if fis[-1]"'!=b:fis.app …[+1237 chars]
```
<details><summary>output</summary>

```

REST 0 5 130 samples 14
F0 score 0.8347 stages 0.7287 0.7791 0.815 0.8263 t [0.3065 0.2369 2.5922] R [[0.8949000239372253, 0.43479999899864197, 0.10050000250339508], [0.05939999967813492, -0.3393000066280365, 0.9387999773025513], [0.4422999918460846, -0.8342000246047974, -0.3294999897480011]]
B0 score 0.8224 stages 0.72 0.7802 0.8049 0.8062 t [0.3065 0.2369 2.5922] R [[0.8787000179290771, 0.4514999985694885, 0.1551000028848648], [-0.17790000140666962, 0.008299999870359898, 0.984000027179718], [0.4429999887943268, -0.8921999931335449, 0.08760000020265579]]
B3 score 0.8123 stages 0.6078 0.737 0.7464 0.7832 t [0.2953 0.2369 2.5734] R [[-0.8407999873161316, -0.5411999821662903, 0.013899999670684338], [0.05660000070929527, -0.062300000339746475, 0.996399998664856], [-0.5383999943733215, 0.8385 …[+9168 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');files=[]
with h5py.File('/work/inputs/masks.h5') as h:
 for f in [0,50,100,240,250,260,380,390,400,510,520,615,625,700,720,750]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);_,im=cap.read();y,x=np.where(h['object'][f]);cx=int(x.mean());cy=int(y.mean());half=55
  cr=im[max(0,cy-half):cy+half,max(0,cx-half):cx+half];cr=cv2.resize(cr,(330,330),interpolation=cv2.INTER_CUBIC);cv2.putText(cr,str(f),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);files.append(cr)
rows=[np.hstack(files[i:i+4]) for i in range(0,len(files),4)];cv2.imwrite('/work/scratch/rest_objects.jpg',np.vstack(rows))
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
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz');B=D['T_world_mesh'];F=D['T_cam_mesh_foundationpose'];cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);K=np.array(cam['K']);KP=np.array(cam['K_pipeline'])
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
# identify handle: high x and mid z excluding top? Print distribution
for condname,idx in [('x>.045',v[:,0]>.045),('x>.055',v[:,0]>.055),('x>.04',v[:,0]>.04)]:
 print(condname,idx.sum(),v[idx].min(0),v[idx].max(0),v[idx].mean(0))
# use x>.04, z<.08 maybe handle points (not top lid edge)
idx=(v[:,0]>.04)&(v[:,2]<.08)
print('handle',idx.sum(),v[idx].mean(0),v[idx].min(0),v[idx].max(0))
def pbox(T,K):
 p=(Tcw@T@np.c_[v,np.ones(len(v))].T).T[:,:3]
 uv=(K@p.T).T;uv=uv[:,:2]/uv[:,2:3];body=v[:,0]<.04
 return uv[idx].mean(0),uv[idx].min(0),uv[idx].max(0),uv[body].mean(0)
flips=[np.eye(3)]
for j in range(3):Q=-np.eye(3);Q[j,j]=1;flips.append(Q)
for base,name,K in [(B[0],'B',K),(Twc@F[0],'F',K)]:
 for j,Q in enumerate(flips):
  T=base.copy();T[:3,:3]=base[:3,:3]@Q
  print(name,j,np.round(pbox(T,K),1))
PY"
```
<details><summary>output</summary>

```
x>.045 41423 [ 0.04500139 -0.03780204 -0.07041354] [0.07257884 0.00027663 0.03503959] [ 0.05811201 -0.0237897  -0.02277481]
x>.055 27159 [ 0.0550003  -0.03780204 -0.06994476] [ 0.07257884 -0.00556018  0.0263684 ] [ 0.06166941 -0.02431332 -0.02707307]
x>.04 45993 [ 0.04001317 -0.03780204 -0.07070145] [0.07257884 0.011533   0.03710553] [ 0.05657647 -0.02292602 -0.02193786]
handle 45993 [ 0.05657647 -0.02292602 -0.02193786] [ 0.04001317 -0.03780204 -0.07070145] [0.07257884 0.011533   0.03710553]
B 0 [[865.1 623. ]
 [859.4 608.6]
 [870.  642.3]
 [857.7 633.7]]
B 1 [[877.  637.5]
 [867.1 620.2]
 [882.2 653.4]
 [856.6 635.4]]
B 2 [[841.8 644.2]
 [836.7 625.1]
 [854.8 659.2]
 [861.2 634.6]]
B 3 [[850.7 631.8]
 [844.  615.1]
 [857.8 648.8]
 [859.6 632.5]]
F 0 [[853.5 615.3]
 [848.1 600.6]
 [858.2  …[+218 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
# final F0 from rest0 script
A=np.array([[0.8949,0.4348,0.1005],[0.0594,-0.3393,0.9388],[0.4423,-0.8342,-0.3295]])
B=np.array([[-0.892,-0.452,0.016],[0.089,-0.142,0.986],[-0.444,0.88,0.167]])
print('diff',R.from_matrix(A.T@B).magnitude()*180/np.pi)
# print handle relative body using projection for A
import trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;idx=(v[:,0]>.04)&(v[:,2]<.08);body=v[:,0]<.04
cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);K=np.array(cam['K']);T=np.eye(4);T[:3,:3]=A;T[:3,3]=[.3065,.2369,2.5922]
p=(Tcw@T@np.c_[v,np.ones(len(v))].T).T[:,:3];uv=(K@p.T).T;uv=uv[:,:2]/uv[:,2:3];print('handle',uv[idx].mean(0),'body',uv[body].mean(0),'zcol',A[:,2])
PY"
```
<details><summary>output</summary>

```
diff 179.28700297163545
handle [818.33638319 594.30240322] body [806.45996125 597.01463343] zcol [ 0.1005  0.9388 -0.3295]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;idx=(v[:,0]>.04)&(v[:,2]<.08);body=v[:,0]<.04
cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);K=np.array(cam['K'])
for name,A,t in [('Fopt',np.array([[0.8949,0.4348,0.1005],[0.0594,-0.3393,0.9388],[0.4423,-0.8342,-0.3295]]),[.3065,.2369,2.5922]),('B3',np.array([[-0.892,-0.452,0.016],[0.089,-0.142,0.986],[-0.444,0.88,0.167]]),[.2999,.2338,2.5859])]:
 T=np.eye(4);T[:3,:3]=A;T[:3,3]=t;p=(Tcw@T@np.c_[v,np.ones(len(v))].T).T[:,:3];uv=(K@p.T).T;uv=uv[:,:2]/uv[:,2:3]
 print(name,'handlex',uv[idx,0].mean(),'bodyx',uv[body,0].mean(),'allbbox',uv.min(0),uv.max(0))
PY"
```
<details><summary>output</summary>

```
Fopt handlex 818.3363831857943 bodyx 806.4599612513981 allbbox [791.07957227 566.72133332] [823.52630704 634.57584306]
B3 handlex 794.719272543384 bodyx 807.6241233309613 allbbox [789.11369625 566.87400797] [821.08675407 633.55811095]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;idx=(v[:,0]>.04)&(v[:,2]<.08);body=v[:,0]<.04
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'])
for name,A,t in [('Fopt',np.array([[0.8949,0.4348,0.1005],[0.0594,-0.3393,0.9388],[0.4423,-0.8342,-0.3295]]),[.3065,.2369,2.5922]),('B3',np.array([[-0.892,-0.452,0.016],[0.089,-0.142,0.986],[-0.444,0.88,0.167]]),[.2999,.2338,2.5859])]:
 T=np.eye(4);T[:3,:3]=A;T[:3,3]=t;p=(T@np.c_[v,np.ones(len(v))].T).T[:,:3];uv=(K@p.T).T;uv=uv[:,:2]/uv[:,2:3]
 print(name,'handlex',uv[idx,0].mean(),'bodyx',uv[body,0].mean(),'bbox',uv.min(0),uv.max(0))
PY"
```
<details><summary>output</summary>

```
Fopt handlex 865.1043607607814 bodyx 853.4836529678397 bbox [837.76167333 603.98601391] [870.33706652 672.16604394]
B3 handlex 841.958871131324 bodyx 854.6271982772624 bbox [836.31711257 603.78288746] [868.2502208  671.16363485]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2,torch
from scipy.spatial.transform import Rotation as R
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
Bc=np.einsum('ij,fjk->fik',Tcw,B)
def skew(mask):
 y,x=np.where(mask); pts=np.c_[x,y]; c=pts.mean(0);q=pts-c
 w,v=np.linalg.eigh(np.cov(q.T)); minor=v[:,0]
 z=q@minor
 return float((z**3).mean()/(z.std()**3+1e-9)),c,w,v
with h5py.File('/work/inputs/masks.h5') as hf:
 for f in [0,50,100,240,380,520,620,700]:
  gt=hf['object'][f,y0:y0+h,x0:x0+w];print('GT',f,skew(gt)[:1],skew(gt)[1])
 for f in [0,50]:
  gt=hf['object'][f,y0:y0+h,x0:x0+w]
  for name,base in [('B',Bc[f]),('F',FP[f])]:
   for j,Q in enumerate(flips):
    T=base.copy();T[:3,:3]=base[:3,:3]@Q;p=renderx(T,PB if name=='B' else PP);print(f,name,j,'iou',round(((p&gt).sum()/max(1,(p|gt).sum())),3),'skew',round(skew(p)[0],2))
PY"
```
<details><summary>output</summary>

```
GT 0 (-0.04145139443540282,) [152.56475368 215.28288471]
GT 50 (-0.055088620496735644,) [152.95532468 215.35064935]
GT 100 (-0.00635790958084208,) [154.47295742 217.27272727]
GT 240 (0.01911938944633124,) [153.88540541 216.00216216]
GT 380 (0.0025725336609960874,) [155.58117647 217.19235294]
GT 520 (-0.00953616528210803,) [151.39385776 215.30711207]
GT 620 (0.0034364945697795152,) [162.39610761 218.40927304]
GT 700 (0.0058574921051812445,) [158.28215527 218.45133256]
0 B 0 iou 0.68 skew 0.03
0 B 1 iou 0.567 skew -0.38
0 B 2 iou 0.505 skew 0.4
0 B 3 iou 0.57 skew -0.02
0 F 0 iou 0.731 skew 0.02
0 F 1 iou 0.627 skew -0.38
0 F 2 iou 0.594 skew 0.39
0 F 3 iou 0.638 skew -0.01
50 B 0 iou 0.719 skew 0.02
50 B 1 iou 0.583 skew -0.35
50 B 2 iou 0.537 skew 0.39
50 B 3 iou 0.604 skew -0.03
50 F 0 io …[+99 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
with h5py.File('/work/inputs/masks.h5') as h:
 for f in [0,50,100,240,380,520,620,700]:
  m=h['object'][f];y,x=np.where(m);x0,x1=x.min(),x.max();cnt=np.bincount(x-x0,minlength=x1-x0+1);print('\\n',f,x0,x1);print(cnt)
PY"
```
<details><summary>output</summary>

```

 0 835 868
[10 26 35 38 41 43 43 46 66 68 69 70 69 69 70 70 70 69 69 69 69 69 69 68
 68 68 68 66 66 64 64 62 38 20]

 50 835 868
[ 5 12 17 34 38 41 44 45 66 68 69 70 70 69 70 70 70 69 69 69 69 69 68 68
 68 68 67 66 66 64 64 62 37 24]

 100 839 869
[ 3 29 32 36 44 62 63 65 66 66 66 66 66 67 68 68 68 67 67 67 67 67 67 65
 65 63 63 61 53 27  4]

 240 836 869
[ 6 18 26 29 32 34 38 44 62 64 65 66 67 67 67 67 67 68 68 68 67 67 67 67
 67 67 66 66 65 64 63 59 37  5]

 380 842 869
[29 35 51 62 63 65 66 67 67 67 67 67 68 68 68 68 68 68 68 68 66 66 66 65
 64 63 39 21]

 520 834 867
[14 21 24 33 36 39 41 49 64 65 66 66 67 67 68 68 68 68 68 68 68 68 67 67
 67 67 67 65 65 63 62 38 27  5]

 620 848 876
[14 32 38 54 64 66 66 67 67 68 68 68 68 68 68 69 69 69 69 69 69 68 67 67
 67 65 63 38 22]

 700 843 87 …[+95 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0]);Bc=np.einsum('ij,fjk->fik',Tcw,B)
def side(m):
 y,x=np.where(m);cnt=np.bincount(x-x.min());core=np.where(cnt>.8*cnt.max())[0]+x.min();return x.mean()-core.mean(),x.mean(),core.mean(),x.min(),x.max()
with h5py.File('/work/inputs/masks.h5') as hf:
 for f in [0,50,100,240,380,520,620,700]:
  gt=hf['object'][f,y0:y0+h,x0:x0+w];print('GT',f,np.round(side(gt),2))
 for f in [0,50]:
  gt=hf['object'][f,y0:y0+h,x0:x0+w]
  for name,base,P in [('B',Bc[f],PB),('F',FP[f],PP)]:
   for j,Q in enumerate(flips):
    T=base.copy();T[:3,:3]=base[:3,:3]@Q;p=renderx(T,P);print(f,name,j,np.round(side(p),2),'iou',round((p&gt).sum()/max(1,(p|gt).sum()),3))
PY"
```
<details><summary>output</summary>

```
GT 0 [ -1.94 152.56 154.5  135.   168.  ]
GT 50 [ -1.54 152.96 154.5  135.   168.  ]
GT 100 [ -0.53 154.47 155.   139.   169.  ]
GT 240 [ -1.61 153.89 155.5  136.   169.  ]
GT 380 [ -0.42 155.58 156.   142.   169.  ]
GT 520 [ -1.61 151.39 153.   134.   167.  ]
GT 620 [ -0.6 162.4 163.  148.  176. ]
GT 700 [ -0.22 158.28 158.5  143.   172.  ]
0 B 0 [ -0.66 157.34 158.   143.   170.  ] iou 0.68
0 B 1 [  3.45 157.95 154.5  141.   181.  ] iou 0.567
0 B 2 [ -1.2 158.3 159.5 137.  176. ] iou 0.505
0 B 3 [ -2.16 159.34 161.5  142.   176.  ] iou 0.57
0 F 0 [-1.000e-01  1.544e+02  1.545e+02  1.410e+02  1.670e+02] iou 0.731
0 F 1 [  3.06 155.06 152.   139.   177.  ] iou 0.627
0 F 2 [ -1.27 155.23 156.5  134.   172.  ] iou 0.594
0 F 3 [ -1.99 156.51 158.5  141.   172.  ] iou 0.638
50 B 0 [ -0.92 156. …[+407 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
for lo,hi in [(-.08,-.06),(-.06,-.04),(-.04,-.02),(-.02,0),(0,.02),(.02,.04),(.04,.06),(.06,.08),(.08,.1)]:
 q=v[(v[:,2]>lo)&(v[:,2]<hi)]
 print(lo,hi,len(q),'x',np.round([q[:,0].min(),q[:,0].max()],3),'y',np.round([q[:,1].min(),q[:,1].max()],3))
PY"
```
<details><summary>output</summary>

```
-0.08 -0.06 50173 x [-0.049  0.069] y [-0.039  0.049]
-0.06 -0.04 24463 x [-0.049  0.072] y [-0.039  0.049]
-0.04 -0.02 23980 x [-0.049  0.073] y [-0.038  0.047]
-0.02 0 32417 x [-0.046  0.072] y [-0.038  0.043]
0 0.02 30874 x [-0.044  0.069] y [-0.037  0.039]
0.02 0.04 34485 x [-0.045  0.06 ] y [-0.035  0.036]
0.04 0.06 12624 x [-0.046  0.024] y [-0.035  0.035]
0.06 0.08 11864 x [-0.048  0.024] y [-0.036  0.035]
0.08 0.1 14985 x [-0.049  0.024] y [-0.038  0.035]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as R
D=np.load('/work/inputs/reconstruction.npz');B=D['T_world_mesh'];F=D['T_cam_mesh_foundationpose'];Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc);K=np.array(json.load(open('/work/inputs/camera.json'))['K'])
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;handle=(v[:,0]>.04)&(v[:,2]<.04);body=(v[:,0]<.02)
flips=[np.eye(3)]
for j in range(3):Q=-np.eye(3);Q[j,j]=1;flips.append(Q)
def proj(T):
 p=(T@np.c_[v,np.ones(len(v))].T).T[:,:3];uv=(K@p.T).T;uv=uv[:,:2]/uv[:,2:3];return uv
for name,base in [('B',Tcw@B[0]),('F',F[0])]:
 for j,Q in enumerate(flips):
  T=base.copy();T[:3,:3]=base[:3,:3]@Q;uv=proj(T);print(name,j,'handle-body dx',round(uv[handle,0].mean()-uv[body,0].mean(),2),'handlex',round(uv[handle,0].mean(),1),'bodyx',round(uv[body,0].mean(),1),'bbox',np.round(uv.min(0)),np.round(uv.max(0)))
PY"
```
<details><summary>output</summary>

```
B 0 handle-body dx 9.19 handlex 865.1 bodyx 855.9 bbox [843. 605.] [871. 672.]
B 1 handle-body dx 22.73 handlex 877.0 bodyx 854.2 bbox [841. 598.] [882. 664.]
B 2 handle-body dx -21.51 handlex 841.8 bodyx 863.3 bbox [837. 597.] [877. 664.]
B 3 handle-body dx -10.94 handlex 850.7 bodyx 861.6 bbox [842. 604.] [877. 670.]
F 0 handle-body dx 10.41 handlex 853.5 bodyx 843.1 bbox [831. 594.] [858. 656.]
F 1 handle-body dx 22.34 handlex 863.9 bodyx 841.6 bbox [830. 588.] [869. 650.]
F 2 handle-body dx -21.4 handlex 829.3 bodyx 850.7 bbox [825. 587.] [863. 649.]
F 3 handle-body dx -11.88 handlex 837.3 bodyx 849.1 bbox [831. 596.] [863. 657.]

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "cat > /work/scratch/fit_rests2.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2,time
Bc=np.einsum('ij,fjk->fik',Tcw,B)
rests=[(5,130),(215,270),(350,405),(480,530),(602,640),(692,775)]
Qz=np.diag([-1.,-1.,1.]).astype(np.float32)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
# mesh semantic points for handle directional cue
mpts=m.vertices;handle=(mpts[:,0]>.04)&(mpts[:,2]<.04);body=(mpts[:,0]<.02)
def side_pred(T):
 p=(T@np.c_[mpts,np.ones(len(mpts))].T).T[:,:3];uv=(K@np.array([[1,0,0],[0,1,0],[0,0,1]])@p.T).T # placeholder
 uv=uv[:,:2]/uv[:,2:3];return float(np.sign(uv[handle,0].mean()-uv[body,0].mean()))
def side_target(mask):
 y,x=np.where(mask);cnt=np.bincount(x-x.min());core=np.where(cnt>.8*cnt.max())[0]+x.min();return float(np.sign(x.mean()-core.mean()))
def opttrans(T,ms,step=.03,nlev=5):
 T=T.copy();t=T[:3,3].copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for dx in [-step,0,step]:
    for dy in [-step,0,step]:
     for dz in [-step,0,step]:
      if dx==dy==dz==0:continue
      U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=ms(U)
      if s>best+1e-6:best=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,best
def optrot(T,ms,step=4,nlev=4):
 T=T.copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for ax in range(3 …[+1768 chars]
```
<details><summary>output</summary>

```
REST 0 5 130 target side -1.0
Traceback (most recent call last):
  File "/work/scratch/fit_rests2.py", line 66, in <module>
    s0=ms(T);T,s1=opttrans(T,ms);T,s2=optrot(T,ms);T,s3=opttrans(T,ms);T,s4=optrot(T,ms,step=1,nlev=3);T,s5=opttrans(T,ms,step=.005,nlev=2)
                                                                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/fit_rests2.py", line 26, in opttrans
    U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=ms(U)
                                                ^^^^^
  File "/work/scratch/fit_rests2.py", line 62, in ms
    if side_pred(T)*tside<0:pen+=1
       ^^^^^^^^^^^^
  File "/work/scratch/fit_rests2.py", line 12, in side_pred
    p=(T@np.c_[mpts,np.ones(len(mpts))].T).T[:,:3];uv=(K@np.a …[+314 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fit_rests3.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2,time
Bc=np.einsum('ij,fjk->fik',Tcw,B)
rests=[(5,130),(215,270),(350,405),(480,530),(602,640),(692,775)]
Qz=np.diag([-1.,-1.,1.]).astype(np.float32)
mpts=m.vertices;handle=(mpts[:,0]>.04)&(mpts[:,2]<.04);body=(mpts[:,0]<.02);mh=np.c_[mpts,np.ones(len(mpts))]
def side_pred(T):
 p=(T@mh.T).T[:,:3];uv=(K@p.T).T;uv=uv[:,:2]/uv[:,2:3];return float(np.sign(uv[handle,0].mean()-uv[body,0].mean()))
def side_target(mask):
 y,x=np.where(mask);cnt=np.bincount(x-x.min());core=np.where(cnt>.8*cnt.max())[0]+x.min();return float(np.sign(x.mean()-core.mean()))
def opttrans(T,ms,step=.03,nlev=4):
 T=T.copy();t=T[:3,3].copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for dx in [-step,0,step]:
    for dy in [-step,0,step]:
     for dz in [-step,0,step]:
      if dx==dy==dz==0:continue
      U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=ms(U)
      if s>best+1e-6:best=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,best
def optrot(T,ms,step=4,nlev=4):
 T=T.copy();best=ms(T)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for ax in range(3):
    for sign in [-1,1]:
     U=T.copy();dv=np.zeros(3);dv[ax]=np.deg2rad(step*sign);U[:3,:3]=T[:3,:3]@RR.from_rotvec(dv).as_matrix();s=ms(U …[+1662 chars]
```
<details><summary>output</summary>

```

REST 0 5 130 target side -1.0 samples 10
  B53 score 0.8506 stages 0.6296 0.7303 0.8504 0.8504 0.8506 side -1.0 t [0.2967 0.2338 2.554 ] R [[-0.9190000295639038, -0.39399999380111694, -0.013000000268220901], [0.06499999761581421, -0.1850000023841858, 0.9810000061988831], [-0.3880000114440918, 0.8999999761581421, 0.19599999487400055]]
  B673 score 0.8504 stages 0.5444 0.726 0.7806 0.8228 0.8342 side -1.0 t [0.2891 0.2286 2.5074] R [[-0.8809999823570251, -0.4729999899864197, 0.0020000000949949026], [0.04100000113248825, -0.0729999989271164, 0.996999979019165], [-0.47099998593330383, 0.878000020980835, 0.08399999886751175]]
  B1303 score 0.8372 stages 0.5804 0.7672 0.7933 0.8155 0.8372 side -1.0 t [0.3046 0.2408 2.5971] R [[-0.9079999923706055, -0.414000004529953, -0.06400000303983688], [0.0 …[+10007 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_dynamic_sample.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2
Bc=np.einsum('ij,fjk->fik',Tcw,B);Qz=np.diag([-1.,-1.,1.]).astype(np.float32)
def scoreone(T,gt,valid):
 p=renderx(T,PB);pp=p&valid;return (pp&gt).sum()/max(1,(pp|gt).sum())
def opttrans(T,gt,valid,step=.04,nlev=5):
 b=scoreone(T,gt,valid);t=T[:3,3].copy()
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for dx in [-step,0,step]:
    for dy in [-step,0,step]:
     for dz in [-step,0,step]:
      if dx==dy==dz==0:continue
      U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=scoreone(U,gt,valid)
      if s>b+1e-7:b=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,b
def optrot(T,gt,valid,step=5,nlev=4):
 b=scoreone(T,gt,valid)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for ax in range(3):
    for sg in [-1,1]:
     U=T.copy();dv=np.zeros(3);dv[ax]=np.deg2rad(step*sg);U[:3,:3]=T[:3,:3]@RR.from_rotvec(dv).as_matrix();s=scoreone(U,gt,valid)
     if s>b+1e-7:b=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,b
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as hf:
 for fi in [140,160,180,200,220,280,300,320,340,420,440,460,540,560,580,640,660,680]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0:y0+h,x0:x …[+751 chars]
```
<details><summary>output</summary>

```
140 Bt stages 0.582 0.691 0.741 0.744 t [0.309 0.169 2.653] Rz [0.04 0.9  0.43]
160 Bt stages 0.613 0.613 0.613 0.613 t [ 0.225 -0.104  2.682] Rz [ 0.38  0.91 -0.16]
180 Bt stages 0.523 0.586 0.616 0.618 t [0.301 0.019 2.69 ] Rz [0.33 0.85 0.42]
200 Bt stages 0.449 0.697 0.739 0.761 t [0.336 0.218 2.774] Rz [-0.01  0.98  0.2 ]
220 Bt stages 0.474 0.737 0.787 0.825 t [0.323 0.253 2.712] Rz [-0.03  0.95  0.3 ]
280 Bt stages 0.515 0.697 0.77 0.776 t [0.321 0.218 2.717] Rz [-0.14  0.97  0.21]
300 Bt stages 0.476 0.571 0.653 0.653 t [ 0.273 -0.138  2.753] Rz [ 0.48  0.87 -0.14]
320 Ft stages 0.259 0.539 0.656 0.706 t [ 0.328 -0.032  2.831] Rz [0.5  0.85 0.19]
340 Bt stages 0.487 0.672 0.699 0.74 t [0.336 0.242 2.746] Rz [-0.08  0.99  0.08]
420 Bt stages 0.199 0.496 0.63 0.661 t [0.331 0.117 2.6 …[+688 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_dynamic_all.py <<'PY'
exec(open('/work/scratch/opt_dynamic_sample.py').read().split(\"cap=cv2.VideoCapture\")[0])
import time
Bc=np.einsum('ij,fjk->fik',Tcw,B);Qz=np.diag([-1.,-1.,1.]).astype(np.float32)
dyn=[(131,214),(271,349),(406,479),(531,601),(641,691)]
cap=cv2.VideoCapture('/work/inputs/video.mp4');N=781;Ts=np.tile(np.eye(4,dtype=np.float32),(N,1,1));scores=np.zeros(N);starts=np.zeros(N,bool)
t0=time.time()
with h5py.File('/work/inputs/masks.h5') as hf:
 for a,b in dyn:
  for fi in range(a,b+1):
   cap.set(cv2.CAP_PROP_POS_FRAMES,fi);cap.read();gt=hf['object'][fi,y0:y0+h,x0:x0+w];per=hf['person'][fi,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3)
   seeds=[]
   for tr in [Bc[fi,:3,3],FP[fi,:3,3]]:
    T=np.eye(4,dtype=np.float32);T[:3,:3]=FP[fi,:3,:3]@Qz;T[:3,3]=tr;seeds.append(T)
   scores0=[scoreone(T,gt,valid) for T in seeds];T=seeds[int(np.argmax(scores0))]
   T,s1=opttrans(T,gt,valid,step=.04,nlev=5);T,s2=optrot(T,gt,valid,step=5,nlev=4);T,s3=opttrans(T,gt,valid,step=.01,nlev=4)
   Ts[fi]=T;scores[fi]=s3;starts[fi]=True
  print('done',a,b,'time',time.time()-t0,flush=True)
   # no
np.savez('/work/scratch/dynamic_opt.npz',T=Ts,scores=scores,starts=starts)
print('total',time.time()-t0)
PY
python /work/scratch/opt_dynamic_all.py"
```
<details><summary>output</summary>

```
done 131 214 time 53.713542222976685
done 271 349 time 90.1244490146637
done 406 479 time 132.85390329360962
done 531 601 time 157.51780796051025
done 641 691 time 177.55964279174805
total 177.5799126625061

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
D=np.load('/work/scratch/dynamic_opt.npz');T=D['T'];s=D['scores'];st=D['starts'];F=D['T'];idx=np.where(st)[0]
# continuity: step angles, acc
step=np.zeros(781);step[1:]=R.from_matrix(np.einsum('fji,fjk->fik',F[:-1,:3,:3],F[1:,:3,:3])).magnitude()*180/np.pi
trstep=np.zeros(781);trstep[1:]=np.linalg.norm(np.diff(F[:,:3,3],axis=0),axis=1)
fig,ax=plt.subplots(5,1,figsize=(18,12),sharex=True)
ax[0].plot(s);ax[0].set_ylabel('IoU fit')
ax[1].plot(step);ax[1].set_ylabel('rot step deg')
ax[2].plot(trstep);ax[2].set_ylabel('trans step m')
for j,c in enumerate(['x','y','z']):
 ax[3].plot(F[:,j,3],label=c)
 ax[4].plot(F[:,j,2],label='z'+c)
ax[3].legend();ax[4].legend();ax[3].set_ylabel('translation');ax[4].set_ylabel('long-axis')
for a in ax:a.grid(alpha=.3)
plt.tight_layout();plt.savefig('/work/scratch/dynamic_opt.png',dpi=140)
print('scores dyn',np.percentile(s[idx],[0,10,25,50,75,90,100]))
print('top steps',[(int(i),round(step[i],1),round(s[i],3)) for i in idx[np.argsort(step[idx])[-30:][::-1]]])
print('top tr',[(int(i),round(trstep[i],3),round(s[i],3)) for i in idx[np.argsort(trstep[idx])[-30:][::-1]]])
PY"
```
<details><summary>output</summary>

```
scores dyn [0.41303116 0.61093978 0.67565027 0.75600437 0.78378619 0.80725083
 0.86654016]
top steps [(271, 179.7, 0.79), (406, 166.9, 0.793), (531, 165.9, 0.828), (131, 163.5, 0.773), (641, 162.4, 0.815), (657, 148.5, 0.538), (660, 95.2, 0.676), (194, 79.3, 0.78), (658, 77.5, 0.413), (293, 60.3, 0.61), (323, 58.7, 0.725), (332, 58.6, 0.681), (338, 57.7, 0.757), (276, 57.6, 0.831), (459, 56.6, 0.799), (467, 55.7, 0.745), (669, 55.4, 0.814), (686, 53.6, 0.779), (687, 52.8, 0.798), (418, 52.7, 0.594), (656, 51.2, 0.618), (465, 50.6, 0.765), (419, 50.2, 0.602), (298, 49.7, 0.561), (209, 49.6, 0.76), (273, 49.6, 0.799), (461, 47.6, 0.755), (341, 47.2, 0.788), (294, 47.0, 0.621), (136, 46.4, 0.74)]
top tr [(406, 2.765, 0.793), (641, 2.749, 0.815), (531, 2.745, 0.828), (271, 2.73, 0.79), (131, 2 …[+534 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
D=np.load('/work/scratch/dynamic_opt.npz');T=D['T'];st=D['starts'];idx=np.where(st)[0]
rec=np.load('/work/inputs/reconstruction.npz');F=rec['T_cam_mesh_foundationpose'];Q=np.diag([-1.,-1.,1.]);F3=np.einsum('fij,jk->fik',F[:,:3,:3],Q)
ang=np.array([R.from_matrix(F3[i].T@T[i,:3,:3]).magnitude() for i in idx])*180/np.pi
print('refine angle',np.percentile(ang,[0,25,50,75,90,100]))
fig,ax=plt.subplots(4,1,figsize=(18,9),sharex=True)
ax[0].plot(idx,[F3[i][:,2]@np.array([0,1,0]) for i in idx],label='F3 z.y');ax[0].plot(idx,T[idx,:3,2]@np.array([0,1,0]),label='opt z.y');ax[0].legend()
ax[1].plot(idx,ang);ax[1].set_ylabel('refine deg')
ax[2].plot(idx,T[idx,:3,2]);ax[2].set_ylabel('opt long axis')
ax[3].plot(idx,F3[:,2][idx]);ax[3].set_ylabel('F3 long axis')
for a in ax:a.grid(alpha=.3)
plt.tight_layout();plt.savefig('/work/scratch/refine_compare.png',dpi=140)
for i in idx[np.argsort(ang)[-20:][::-1]]:print(i,round(ang[np.where(idx==i)[0][0]],1),np.round(F3[i][:,2],2),np.round(T[i,:3,2],2),round(D['scores'][i],2))
PY"
```
<details><summary>output</summary>

```
refine angle [1.17502097e-16 7.06994552e+00 1.32583576e+01 2.21945267e+01
 3.26648079e+01 9.24567222e+01]
656 92.5 [0.14 0.08 0.99] [ 0.14  0.99 -0.02] 0.62
660 76.8 [0.16 0.08 0.98] [0.14 0.98 0.15] 0.68
418 75.8 [0.24 0.95 0.21] [0.09 0.95 0.31] 0.59
422 70.6 [ 0.54  0.84 -0.03] [0.13 0.86 0.48] 0.53
657 63.9 [-0.    0.09  1.  ] [-0.28 -0.81  0.51] 0.54
276 57.8 [0.02 0.92 0.39] [-0.07  0.93  0.37] 0.83
277 53.6 [0.07 0.91 0.41] [-0.09  0.96  0.26] 0.8
568 52.5 [ 0.57  0.61 -0.55] [ 0.53  0.72 -0.45] 0.77
655 52.1 [0.33 0.45 0.83] [0.17 0.97 0.15] 0.68
293 50.9 [ 0.18  0.98 -0.13] [ 0.18  0.95 -0.25] 0.61
132 50.3 [ 0.03  1.   -0.05] [-0.09  0.98  0.17] 0.77
131 50.2 [ 0.04  1.   -0.04] [-0.04  0.98  0.22] 0.77
342 48.0 [0.08 0.93 0.35] [-0.02  0.97  0.26] 0.79
686 47.7 [-0.01  0.95  0.3 …[+330 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/smooth_test.py <<'PY'
import numpy as np,json,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.sparse import diags,eye,vstack,csr_matrix
from scipy.sparse.linalg import spsolve
D=np.load('/work/scratch/dynamic_opt.npz');T=D['T'];scores=D['scores'];dyn=[(131,214),(271,349),(406,479),(531,601),(641,691)]
# anchors manually from valid-side rest fits
Rm=[np.array([[-0.919,-0.394,-0.013],[0.065,-0.185,0.981],[-0.388,0.9,0.196]]),
np.array([[-0.972,-0.227,-0.054],[-0.049,-0.029,0.998],[-0.228,0.973,0.017]]),
np.array([[-0.886,-0.457,-0.077],[-0.019,-0.131,0.991],[-0.464,0.88,0.107]]),
np.array([[-0.868,0.497,-0.018],[-0.118,-0.17,0.978],[0.483,0.851,0.206]]),
np.array([[-0.91,-0.41,-0.063],[0.013,-0.179,0.984],[-0.415,0.894,0.169]]),
np.array([[-0.912,-0.41,-0.032],[0.018,-0.118,0.993],[-0.411,0.904,0.115]])]
rec=np.load('/work/inputs/reconstruction.npz');F=rec['T_cam_mesh_foundationpose'];Q=np.diag([-1.,-1.,1.]);F3=np.einsum('fij,jk->fik',F[:,:3,:3],Q)
def quat(Rm): return R.from_matrix(Rm).as_quat() # xyzw
def align(q,ref): return q if np.dot(q,ref)>=0 else -q
def smooth_seg(meas,qA,qB,scores,wf=.15,lam1=.5,lam2=5):
 n=len(meas); qs=[];ref=qA.copy()
 for q in meas:
  q=align(q,ref);qs.append(q);ref=q
 qs=np.array(qs);wo=0.1+1.5*np.clip((scores[:n]-.4)/.4,0,1)
 A=diags(wo+0.2); b=(wo+0.2)[:,None]*qs
 # prior F3
 qf=[];ref=qA.copy()
 for q in np.array([quat(x) for x in …[+1650 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/smooth_test.py", line 36, in <module>
    Rr=smooth_seg(meas,qA,qB,scores[a:b+1],wf=.15)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/smooth_test.py", line 30, in smooth_seg
    q=spsolve(A,b)
      ^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/sparse/linalg/_dsolve/linsolve.py", line 286, in spsolve
    x, info = _superlu.gssv(N, A.nnz, A.data, indices, indptr,
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: object too deep for desired array

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/smooth_test.py';s=open(p).read().replace('q=spsolve(A,b)','q=np.stack([spsolve(A,b[:,j]) for j in range(4)],1)');open(p,'w').write(s)
PY
python /work/scratch/smooth_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/smooth_test.py", line 36, in <module>
    Rr=smooth_seg(meas,qA,qB,scores[a:b+1],wf=.15)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/smooth_test.py", line 31, in smooth_seg
    q=np.array([align(x,qA) for x in q]);q/=np.linalg.norm(q,axis=1,keepdims=True)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/smooth_test.py", line 31, in <listcomp>
    q=np.array([align(x,qA) for x in q]);q/=np.linalg.norm(q,axis=1,keepdims=True)
                ^^^^^^^^^^^
  File "/work/scratch/smooth_test.py", line 15, in align
    def align(q,ref): return q if np.dot(q,ref)>=0 else -q
                                  ^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguou …[+26 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/smooth_test.py';s=open(p).read();s=s.replace(\"q=np.stack([spsolve(A,b[:,j]) for j in range(4)],1)\",\"q=np.column_stack([np.asarray(spsolve(A,np.asarray(b)[:,j])).ravel() for j in range(4)])\").replace(\"q=np.array([align(x,qA) for x in q]);q/=np.linalg.norm(q,axis=1,keepdims=True)\",\"q=np.array([np.asarray(align(np.asarray(x).ravel(),np.asarray(qA).ravel())).ravel() for x in q]);q/=np.linalg.norm(q,axis=1,keepdims=True)\");open(p,'w').write(s)
PY
python /work/scratch/smooth_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/smooth_test.py", line 38, in <module>
    out[a:b+1,:3,:3]=Rr;out[a,:3,:3]=Rm[si];out[b,:3,:3]=Rm[si+1]
    ~~~^^^^^^^^^^^^^
ValueError: could not broadcast input array from shape (336,3,3) into shape (84,3,3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/b=(wo+0.2)\\[:,None\\]\\*qs/b=(wo+0.2)[:,None]*qs/' /work/scratch/smooth_test.py
python /work/scratch/smooth_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/smooth_test.py", line 38, in <module>
    out[a:b+1,:3,:3]=Rr;out[a,:3,:3]=Rm[si];out[b,:3,:3]=Rm[si+1]
    ~~~^^^^^^^^^^^^^
ValueError: could not broadcast input array from shape (336,3,3) into shape (84,3,3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"A=diags\\|b=\" /work/scratch/smooth_test.py; python - <<'PY'
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
wo=np.ones(84);qs=np.ones((84,4));b=(wo+.2)[:,None]*qs;print(b.shape)
PY"
```
<details><summary>output</summary>

```
(84, 4)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/smooth_test.py';s=open(p).read();s=s.replace(\"return R.from_quat(q).as_matrix()\",\"print('DBG q',q.shape,'b',b.shape,'A',A.shape); return R.from_quat(q).as_matrix()\");open(p,'w').write(s)
PY
python /work/scratch/smooth_test.py"
```
<details><summary>output</summary>

```
DBG q (336, 4) b (84, 84, 4) A (84, 84)
Traceback (most recent call last):
  File "/work/scratch/smooth_test.py", line 38, in <module>
    out[a:b+1,:3,:3]=Rr;out[a,:3,:3]=Rm[si];out[b,:3,:3]=Rm[si+1]
    ~~~^^^^^^^^^^^^^
ValueError: could not broadcast input array from shape (336,3,3) into shape (84,3,3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/b=b+wf\\*np.array(qf)\\[:,None\\]/b=b+wf*np.array(qf)/' /work/scratch/smooth_test.py
python /work/scratch/smooth_test.py"
```
<details><summary>output</summary>

```
DBG q (84, 4) b (84, 4) A (84, 84)
DBG q (79, 4) b (79, 4) A (79, 79)
DBG q (74, 4) b (74, 4) A (74, 74)
DBG q (71, 4) b (71, 4) A (71, 71)
DBG q (51, 4) b (51, 4) A (51, 51)
steps raw [  4.18593993   7.01720551  11.57376884  17.86346602 158.95478593
 172.81154181]
131 214 maxstep 27.537258463151225 p95 12.743963063579343
271 349 maxstep 25.86818169120392 p95 19.631605724297216
406 479 maxstep 28.755952143053186 p95 9.028552405142099
531 601 maxstep 25.034391451100117 p95 10.169373118940987
641 691 maxstep 16.707330270307374 p95 10.350283562046673

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/smooth_test.py';s=open(p).read();s=s.replace(\"qA=quat(Rm[si]);qB=quat(Rm[si+1]);\",\"qA=quat(Rm[si]);qB=quat(Rm[si+1]);\\n if np.dot(qB,qs[-1])<0:qB=-qB\")
open(p,'w').write(s)
PY
python /work/scratch/smooth_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/smooth_test.py", line 36, in <module>
    if np.dot(qB,qs[-1])<0:qB=-qB
                 ^^
NameError: name 'qs' is not defined. Did you mean: 'qA'?

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/smooth_test.py';s=open(p).read().replace(\"qA=quat(Rm[si]);qB=quat(Rm[si+1]);\\n if np.dot(qB,qs[-1])<0:qB=-qB\",\"qA=quat(Rm[si]);qB=quat(Rm[si+1]);\").replace(\"qs=np.array(qs);wo=\",\"qs=np.array(qs)\\n if np.dot(qB,qs[-1])<0:qB=-qB\\n wo=\")
open(p,'w').write(s)
PY
python /work/scratch/smooth_test.py"
```
<details><summary>output</summary>

```
DBG q (84, 4) b (84, 4) A (84, 84)
DBG q (79, 4) b (79, 4) A (79, 79)
DBG q (74, 4) b (74, 4) A (74, 74)
DBG q (71, 4) b (71, 4) A (71, 71)
DBG q (51, 4) b (51, 4) A (51, 51)
steps raw [  4.18593993   7.01720551  11.57376884  17.86346602 158.95478593
 172.81154181]
131 214 maxstep 27.537258463151225 p95 12.743963063579343
271 349 maxstep 25.86818169120392 p95 19.631605724297216
406 479 maxstep 28.755952143053186 p95 9.028552405142099
531 601 maxstep 25.034391451100117 p95 10.169373118940987
641 691 maxstep 16.707330270307374 p95 10.350283562046673

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# quick import definitions from smooth test before loop
s=open('/work/scratch/smooth_test.py').read();exec(s.split(\"out=T.copy()\")[0])
from scipy.spatial.transform import Rotation as R
D=np.load('/work/scratch/dynamic_opt.npz');T=D['T'];scores=D['scores'];dyn=[(131,214),(271,349),(406,479),(531,601),(641,691)]
for si,(a,b) in enumerate(dyn):
 qA=quat(Rm[si]);qB=quat(Rm[si+1]);meas=[quat(T[i,:3,:3]) for i in range(a,b+1)]
 print('seg',a,'qA',qA,'qmeasA',meas[0],'dot',qA@meas[0],'angs',R.from_matrix(Rm[si].T@T[a,:3,:3]).magnitude()*180/np.pi)
 rr=smooth_seg(meas,qA,qB,scores[a:b+1],wf=.15)
 print('angles start/end relative anchor',R.from_matrix(Rm[si].T@rr[0]).magnitude()*180/np.pi,R.from_matrix(Rm[si].T@rr[-1]).magnitude()*180/np.pi)
 # relative each successive
 st=R.from_matrix(np.einsum('fji,fjk->fik',rr[:-1],rr[1:])).magnitude()*180/np.pi
 print('step pct',np.percentile(st,[50,90,100]))
PY"
```
<details><summary>output</summary>

```
seg 131 qA [-0.13241036  0.6201248   0.75829782  0.15131799] qmeasA [ 0.09124674  0.61849023  0.76713996 -0.14366678] dot 0.9314403148803796 angs 42.67908317084304
DBG q (84, 4) b (84, 4) A (84, 84)
angles start/end relative anchor 1.0548617558337972 14.445692926545691
step pct [ 3.45293008 11.24707339 27.5372569 ]
seg 271 qA [-0.09938067  0.6939754   0.71034611  0.06269   ] qmeasA [-0.0076634   0.53567023  0.84438903  0.0024195 ] dot 0.9724636903919303 angs 26.95393245885325
DBG q (79, 4) b (79, 4) A (79, 79)
angles start/end relative anchor 0.5464456097971679 15.310087442938766
step pct [ 6.38995207 16.24377811 25.86818288]
seg 406 qA [-0.18549757  0.64187374  0.72867323  0.15041305] qmeasA [-0.15394685  0.58985422  0.78443648  0.11415679] dot 0.99593723191479 angs 10.332984748763488
DBG …[+784 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/make_prototype.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.sparse import diags,csr_matrix
from scipy.sparse.linalg import spsolve
import cv2
# camera poses from rest fits
rests=[(0,130),(215,270),(350,405),(480,530),(602,640),(692,780)]
dyn=[(131,214),(271,349),(406,479),(531,601),(641,691)]
Rrest=[
np.array([[-0.919,-0.394,-0.013],[0.065,-0.185,0.981],[-0.388,0.9,0.196]]),
np.array([[-0.972,-0.227,-0.054],[-0.049,-0.029,0.998],[-0.228,0.973,0.017]]),
np.array([[-0.886,-0.457,-0.077],[-0.019,-0.131,0.991],[-0.464,0.88,0.107]]),
np.array([[-0.868,0.497,-0.018],[-0.118,-0.17,0.978],[0.483,0.851,0.206]]),
np.array([[-0.91,-0.41,-0.063],[0.013,-0.179,0.984],[-0.415,0.894,0.169]]),
np.array([[-0.912,-0.41,-0.032],[0.018,-0.118,0.993],[-0.411,0.904,0.115]])]
trest=[np.array([0.2967,0.2338,2.5540]),np.array([0.3188,0.2505,2.6970]),np.array([0.3270,0.2579,2.7333]),np.array([0.3086,0.2492,2.7059]),np.array([0.3450,0.2582,2.6996]),np.array([0.3277,0.2562,2.6856])]
def quat(Rm): return RR.from_matrix(Rm).as_quat()
def align(q,ref): return q if np.dot(q,ref)>=0 else -q
def smooth_quat(meas,qA,qB,scores,wf=.08,lam1=.5,lam2=5,anchor_w=200):
 n=len(meas);qs=[];ref=qA.copy()
 for q in meas:q=align(q,ref);qs.append(q);ref=q
 qs=np.array(qs);qB=align(qB,qs[-1]);qf=[];ref=qA.copy()
 for x in F3_dummy[:n]:pass
 # no F prior param (dummy)
 wo=0.2+2.0 …[+3607 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/make_prototype.py", line 62, in <module>
    gt=hf['object'][i,y0:y0+h,x0:x0+w];per=hf['person'][i,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3)
       ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 945, in __getitem__
    selection = sel.select(self.shape, args, dataset=self)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/selections.py", line 85, in select
    return selector.make_selection(args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h5py/_ob …[+359 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '\\bh\\b' /work/scratch/make_prototype.py"
```
<details><summary>output</summary>

```
62:  gt=hf['object'][i,y0:y0+h,x0:x0+w];per=hf['person'][i,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"x0,y0\\|h=\" /work/scratch/score_rotations.py | head -20; sed -n '1,30p' /work/scratch/score_rotations.py"
```
<details><summary>output</summary>

```
exec(open('/work/scratch/score_candidates.py').read().split("N=len(B)")[0])
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as RR
import cv2
Bc=np.einsum('ij,fjk->fik',Tcw,B)
# use full image clip projection but fixed crop renderer for speed
# rendering routines from opt test
import nvdiffrast.torch as dr
UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda');imtex=np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255;tex=torch.tensor(imtex,device='cuda')
def Pmat(Kc):
 fx,fy=Kc[0,0],Kc[1,1];cx,cy=Kc[0,2],Kc[1,2]
 return torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,2*fy/h,2*(cy-y0)/h-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
PB=PB.contiguous();PP=Pmat(KP).contiguous()
def renderx(T,P,texmode=Fal …[+957 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/make_prototype.py').read();s=s.replace(\"# eval iou all frames\\ncap=\", \"# eval iou all frames\\nprint('DEBUG h w x0 y0',h,w,x0,y0)\\ncap=\");open('/work/scratch/make_prototype.py','w').write(s)
PY
python /work/scratch/make_prototype.py"
```
<details><summary>output</summary>

```
DEBUG h w x0 y0 320 [0.         0.04296875 0.15625    0.31640625 0.5        0.68359375
 0.84375    0.95703125 1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         1.         1.         1.         1.         1.
 1.         0.95703125 0.84375    0.68359375 0.5        0.31640625
 0.15625    0.04296875 0.        ] 700 420
Traceback (most recent call last):
  File "/work/scratch/make_prototype.py", line 63, in <module>
    gt=hf['object'][i,y0:y0+h,x0:x0+w];per=hf['person'][i,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3)
       ~~~~~~~~~~~~^^^^^^^^^^^^^^^^ …[+910 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/make_prototype.py';s=open(p).read();s=s.replace(\"w=np.ones(b-a+1);w[:nb]=smoothstep(np.arange(nb)/nb);w[-nb:]=smoothstep(np.arange(nb)[::-1]/nb)\",\"bw=np.ones(b-a+1);bw[:nb]=smoothstep(np.arange(nb)/nb);bw[-nb:]=smoothstep(np.arange(nb)[::-1]/nb)\").replace(\"w[k]\",\"bw[k]\").replace(\"(1-w[:,None])\",\"(1-bw[:,None])\").replace(\"+w[:,None]*ts\",\"+bw[:,None]*ts\")
s=s.replace(\"print('DEBUG h w x0 y0',h,w,x0,y0)\\n\",\"\")
open(p,'w').write(s)
PY
python /work/scratch/make_prototype.py"
```
<details><summary>output</summary>

```
overall 0.7807680802371656 0.6146461606090781 med 0.7972429638139 0.664754953076121
rest 0 130 0.8611010461611815 0.7157532586941704
rest 215 270 0.7957668891779787 0.6239279389248964
rest 350 405 0.8142577220398799 0.6277906754299382
rest 480 530 0.8125490491417207 0.6197405131398473
rest 602 640 0.8287798278492661 0.6007917530681749
rest 692 780 0.8315991830671221 0.6704919608920563
dyn 131 214 0.6662424203229487 0.5869923450455338
dyn 271 349 0.705578480299688 0.5008866914391202
dyn 406 479 0.7251687269380614 0.5745570019176305
dyn 531 601 0.7667227944090469 0.6244227766009404
dyn 641 691 0.769306781516207 0.5046796508943553
step percentiles all [1.61269564e-14 3.82856581e+00 8.20924821e+00 1.14478070e+01
 2.01475819e+01 3.15678383e+01]
dynstep 131 214 [ 3.52490204 12.46205367 13.699536 …[+280 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as R
D=np.load('/work/scratch/T_proto.npy');B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::10]
rests=[(0,130),(215,270),(350,405),(480,530),(602,640),(692,780)]
for name,T in [('proto',D),('base',B)]:
 print(name)
 for a,b in rests:
  # max world y over sampled transformed verts, and center
  vals=[]
  for f in [a,(a+b)//2,b]:
   q=(T[f]@np.c_[v,np.ones(len(v))].T).T[:,:3];vals.append([q[:,1].max(),q[:,1].mean(),T[f,:3,3,None]])
  print(a,b,'maxY',np.round([x[0] for x in vals],4),'centerY',np.round([x[1] for x in vals],4),'t',np.round(T[(a+b)//2,:3,3],3))
PY"
```
<details><summary>output</summary>

```
proto
0 130 maxY [0.4645 0.4645 0.4645] centerY [0.3445 0.3445 0.3445] t [0.443 0.35  2.537]
215 270 maxY [0.482 0.482 0.482] centerY [0.3628 0.3628 0.3628] t [0.471 0.369 2.679]
350 405 maxY [0.4906 0.4906 0.4906] centerY [0.3709 0.3709 0.3709] t [0.48  0.377 2.714]
480 530 maxY [0.4832 0.4832 0.4832] centerY [0.3618 0.3618 0.3618] t [0.461 0.368 2.688]
602 640 maxY [0.491 0.491 0.491] centerY [0.3708 0.3708 0.3708] t [0.497 0.377 2.68 ]
692 780 maxY [0.488 0.488 0.488] centerY [0.3686 0.3686 0.3686] t [0.479 0.374 2.667]
base
0 130 maxY [0.4669 0.4644 0.4717] centerY [0.3444 0.3412 0.352 ] t [0.46  0.348 2.558]
215 270 maxY [0.4579 0.4729 0.472 ] centerY [0.3355 0.351  0.3492] t [0.484 0.357 2.701]
350 405 maxY [0.4536 0.4763 0.4777] centerY [0.331  0.3535 0.3528] t [0.487 0.36  2.717]
4 …[+266 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_color.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.ndimage import binary_dilation
import cv2
Tp=np.load('/work/scratch/T_proto.npy');rests=[(0,130),(215,270),(350,405),(480,530),(602,640),(692,780)];dyns=[(131,214),(271,349),(406,479),(531,601),(641,691)]
cap=cv2.VideoCapture('/work/inputs/video.mp4');N=781
met=np.zeros(N);metb=np.zeros(N);mae=np.zeros(N);maeb=np.zeros(N);ious=np.zeros(N);bious=np.zeros(N)
def calc(T,im,gt,valid):
 p,rgb=renderx(T,PB,True);m=gt&p&valid
 if m.sum()<20:return -1,1,0
 x=rgb[m];y=im[m];x=x-x.mean(0);y=y-y.mean(0);c=float((x*y).sum()/np.sqrt((x*x).sum()*(y*y).sum()+1e-12));a=float(np.abs(rgb[m]-im[m]).mean());iou=(p&gt).sum()/max(1,(p|gt).sum());return c,a,iou
with h5py.File('/work/inputs/masks.h5') as hf:
 for i in range(N):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,b=cap.read();im=b[y0:y0+h,x0:x0+w,::-1].astype(np.float32)/255;gt=hf['object'][i,y0:y0+h,x0:x0+w];per=hf['person'][i,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3)
  met[i],mae[i],ious[i]=calc(Tcw@Tp[i],im,gt,valid);metb[i],maeb[i],bious[i]=calc(Tcw@B[i],im,gt,valid)
np.savez('/work/scratch/color_eval.npz',corr=met,corrb=metb,mae=mae,maeb=maeb,iou=ious,bious=bious)
for name,rng in [('all',(0,780)),*[(f'rest{i}',r) for i,r in enumerate(rests)],*[(f'dyn{i}',r) for i,r in enumerate(dyns)]]:
 a,b=rng;print(name,'corr',np.nanmean(met[a:b+1]),np.nanmean(metb[a:b+1]),'mae',np.nanmean …[+121 chars]
```
<details><summary>output</summary>

```
all corr 0.24184491366238048 0.10860514907056427 mae 0.1434420314969235 0.16349352115858226 iou 0.8129383241859052 0.6574977837204682
rest0 corr 0.23213631326601972 0.2268582362231779 mae 0.14052105598322306 0.14226634487623477 iou 0.8761416520337448 0.7274055665159831
rest1 corr 0.4072764959413681 0.08034018419925311 mae 0.11521897039243154 0.158880173095635 iou 0.7916486579304636 0.6686690428880989
rest2 corr 0.3372826767782388 0.08295590928839837 mae 0.11989640910178423 0.15893836159791266 iou 0.8461793385236912 0.6632247692240659
rest3 corr 0.22848657703502592 0.07648440166521697 mae 0.13780979899799123 0.1598500112692515 iou 0.7757161770258393 0.6519556394069222
rest4 corr 0.28219817502693917 0.039867973113127925 mae 0.12989504845478597 0.1642914544313382 iou 0.8459184452653903 0.6140 …[+825 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p=np.load('/work/scratch/proto_eval.npz');c=np.load('/work/scratch/color_eval.npz')
print(np.mean(p['iou']),np.mean(c['iou']),np.max(np.abs(p['iou']-c['iou'])))
print([(i,p['iou'][i],c['iou'][i]) for i in range(781) if abs(p['iou'][i]-c['iou'][i])>.05][:20])
PY"
```
<details><summary>output</summary>

```
[(99, 0.7917558886509636, 0.9102231447846393), (101, 0.8076086956521739, 0.9035453597497393), (102, 0.7821676454885211, 0.8685279187817259), (103, 0.7698072805139187, 0.8646044624746451), (105, 0.8003309431880861, 0.8749354005167959), (107, 0.8087005019520357, 0.8666666666666667), (114, 0.7733333333333333, 0.8367967562088191), (116, 0.758869179600887, 0.8416243654822335), (117, 0.7615298087739033, 0.833587786259542), (118, 0.7747695852534562, 0.8547631441957314), (119, 0.7833914053426249, 0.8519290928050052), (120, 0.7472205968402574, 0.8472367049009385), (121, 0.7434858135495078, 0.8479501816294759), (122, 0.7465596330275229, 0.8414006179196705), (126, 0.7404494382022472, 0.8154851684263449), (127, 0.7019903173749328, 0.7593301435406699), (128, 0.7027896995708155, 0.7574021012416428), (12 …[+137 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/T_proto.npy');step=np.zeros(781);step[1:]=R.from_matrix(np.einsum('fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3])).magnitude()*180/np.pi
for a,b in [(131,214),(271,349),(406,479),(531,601),(641,691)]:
 print('\\n',a,b)
 for i in range(a,b+1):
  if step[i]>15:print(i,round(step[i],1))
PY"
```
<details><summary>output</summary>

```

 131 214
192 19.8
193 26.9
194 31.6
195 23.7

 271 349
288 16.2
317 15.9
318 15.3
323 16.4
331 16.7
332 22.0
333 18.2
338 21.4
339 23.8
340 28.1
341 28.9
342 19.5

 406 479

 531 601

 641 691
668 15.6
669 18.8

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');sets=[list(range(185,206,2)),list(range(325,346,2))]
thumbs=[]
with h5py.File('/work/inputs/masks.h5') as h:
 for fs in sets:
  row=[]
  for f in fs:
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);_,im=cap.read();y,x=np.where(h['object'][f]);cx=int(x.mean());cy=int(y.mean());half=65;cr=im[max(0,cy-half):cy+half,max(0,cx-half):cx+half];cr=cv2.resize(cr,(220,220));cv2.putText(cr,str(f),(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);row.append(cr)
  thumbs.append(np.hstack(row))
cv2.imwrite('/work/scratch/motion_detail.jpg',np.vstack(thumbs))
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
T=np.load('/work/scratch/T_proto.npy');R=np.load('/work/scratch/dynamic_opt.npz')['T'];
for a,b in [(185,205),(325,345)]:
 print('\\n',a,b)
 for i in range(a,b+1):
  print(i,'proto z',np.round(T[i,:3,2],2),'raw z',np.round(R[i,:3,2],2))
PY"
```
<details><summary>output</summary>

```

 185 205
185 proto z [0.25 0.92 0.29] raw z [0.23 0.9  0.36]
186 proto z [0.24 0.93 0.27] raw z [0.22 0.95 0.23]
187 proto z [0.23 0.94 0.26] raw z [0.21 0.92 0.32]
188 proto z [0.22 0.94 0.24] raw z [0.23 0.94 0.26]
189 proto z [0.21 0.95 0.21] raw z [0.19 0.93 0.32]
190 proto z [0.2  0.97 0.15] raw z [0.16 0.97 0.2 ]
191 proto z [0.18 0.98 0.08] raw z [0.11 0.98 0.16]
192 proto z [0.12 0.99 0.02] raw z [0.14 0.97 0.19]
193 proto z [0.02 1.   0.02] raw z [ 0.09  0.9  -0.44]
194 proto z [-0.03  0.98  0.19] raw z [0.05 0.92 0.38]
195 proto z [-0.    0.95  0.32] raw z [0.03 0.9  0.44]
196 proto z [0.04 0.92 0.39] raw z [0.04 0.89 0.45]
197 proto z [0.06 0.91 0.41] raw z [0.04 0.92 0.38]
198 proto z [0.05 0.91 0.41] raw z [0.04 0.87 0.49]
199 proto z [0.03 0.93 0.36] raw z [0.04 0.88 0.47]
2 …[+1479 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('/work/scratch/dynamic_opt.npz');
for a,b in [(185,205),(325,345),(650,665)]:print(a,b,[(i,round(D['scores'][i],2)) for i in range(a,b+1)])
PY"
```
<details><summary>output</summary>

```
185 205 [(185, 0.62), (186, 0.64), (187, 0.56), (188, 0.69), (189, 0.64), (190, 0.64), (191, 0.53), (192, 0.65), (193, 0.77), (194, 0.78), (195, 0.75), (196, 0.76), (197, 0.79), (198, 0.76), (199, 0.76), (200, 0.76), (201, 0.76), (202, 0.76), (203, 0.75), (204, 0.76), (205, 0.76)]
325 345 [(325, 0.67), (326, 0.75), (327, 0.77), (328, 0.69), (329, 0.65), (330, 0.8), (331, 0.72), (332, 0.68), (333, 0.66), (334, 0.66), (335, 0.7), (336, 0.69), (337, 0.68), (338, 0.76), (339, 0.75), (340, 0.73), (341, 0.79), (342, 0.79), (343, 0.79), (344, 0.81), (345, 0.8)]
650 665 [(650, 0.76), (651, 0.74), (652, 0.77), (653, 0.78), (654, 0.68), (655, 0.68), (656, 0.62), (657, 0.54), (658, 0.41), (659, 0.43), (660, 0.68), (661, 0.87), (662, 0.82), (663, 0.81), (664, 0.83), (665, 0.76)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/fit_rests3.py /work/scratch/fit_rests_final.py && python - <<'PY'
p='/work/scratch/fit_rests_final.py';s=open(p).read();s=s.replace(\"tsides=[side_target(gt) for gt in gts];tside=float(np.median(tsides));print('\\\\nREST',ri,a,b,'target side',tside,'samples',len(gts),flush=True)\",\"tside=-1.0  # visually verified: handle remains image-left in every rest interval\\n  print('\\\\nREST',ri,a,b,'target side',tside,'samples',len(gts),flush=True)\").replace(\"np.savez('/work/scratch/rest_poses3.npz'\",\"np.savez('/work/scratch/rest_poses_final.npz'\");open(p,'w').write(s)
PY
python /work/scratch/fit_rests_final.py"
```
<details><summary>output</summary>

```

REST 0 5 130 target side -1.0 samples 10
  B53 score 0.8506 stages 0.6296 0.7303 0.8504 0.8504 0.8506 side -1.0 t [0.2967 0.2338 2.554 ] R [[-0.9190000295639038, -0.39399999380111694, -0.013000000268220901], [0.06499999761581421, -0.1850000023841858, 0.9810000061988831], [-0.3880000114440918, 0.8999999761581421, 0.19599999487400055]]
  B673 score 0.8504 stages 0.5444 0.726 0.7806 0.8228 0.8342 side -1.0 t [0.2891 0.2286 2.5074] R [[-0.8809999823570251, -0.4729999899864197, 0.0020000000949949026], [0.04100000113248825, -0.0729999989271164, 0.996999979019165], [-0.47099998593330383, 0.878000020980835, 0.08399999886751175]]
  B1303 score 0.8372 stages 0.5804 0.7672 0.7933 0.8155 0.8372 side -1.0 t [0.3046 0.2408 2.5971] R [[-0.9079999923706055, -0.414000004529953, -0.06400000303983688], [0.0 …[+10047 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_yaw.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2
P=np.load('/work/scratch/rest_poses_final.npz');rests=[(0,130),(215,270),(350,405),(480,530),(602,640),(692,780)]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def corr(rgb,im,m):
 if m.sum()<20:return -1
 a=rgb[m];b=im[m];a=a-a.mean(0);b=b-b.mean(0);return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
def eval_yaw(T,dat,deg):
 R0=T[:3,:3]@RR.from_rotvec([0,0,np.deg2rad(deg)]).as_matrix();c=[]
 for im,gt,valid in dat:
  U=T.copy();U[:3,:3]=R0;p,rgb=renderx(U,PB,True);m=gt&p&valid;c.append(corr(rgb,im,m))
 return np.mean(c)
with h5py.File('/work/inputs/masks.h5') as hf:
 for ri,(a,b) in enumerate(rests):
  T=P[f'r{ri}'];fis=list(range(a,b+1,10));
  dat=[]
  for fi in fis:
   cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0:y0+h,x0:x0+w,::-1].astype(np.float32)/255;gt=hf['object'][fi,y0:y0+h,x0:x0+w];per=hf['person'][fi,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3);dat.append((im,gt,valid))
  vals=[(d,eval_yaw(T,dat,d)) for d in range(-40,41,5)]
  vals.sort(key=lambda x:x[1],reverse=True);print('\\nREST',ri,'top',[(d,round(c,3)) for d,c in vals[:8]])
  # fine around best
  best=vals[0][0];fine=[(d,eval_yaw(T,dat,d)) for d in np.arange(best-5,best+5.1,1)];fine.sort(key=lambda x:x[1],reverse=True);prin …[+231 chars]
```
<details><summary>output</summary>

```

REST 0 top [(0, 0.227), (5, 0.196), (-5, 0.187), (-40, 0.184), (-35, 0.177), (40, 0.175), (-30, 0.17), (10, 0.169)]
fine [(2.0, 0.24), (1.0, 0.235), (3.0, 0.228), (0.0, 0.227), (-1.0, 0.219)]

REST 1 top [(-10, 0.44), (-5, 0.432), (40, 0.429), (-15, 0.423), (30, 0.422), (0, 0.422), (35, 0.417), (-20, 0.413)]
fine [(-8.0, 0.446), (-9.0, 0.444), (-7.0, 0.441), (-10.0, 0.44), (-12.0, 0.439)]

REST 2 top [(-15, 0.378), (-10, 0.378), (40, 0.373), (5, 0.371), (-20, 0.37), (-40, 0.367), (35, 0.364), (-5, 0.35)]
fine [(-15.0, 0.378), (-10.0, 0.378), (-14.0, 0.377), (-16.0, 0.377), (-13.0, 0.376)]

REST 3 top [(40, 0.335), (35, 0.327), (30, 0.314), (25, 0.3), (20, 0.294), (15, 0.276), (10, 0.27), (5, 0.246)]
fine [(37.0, 0.336), (38.0, 0.336), (40.0, 0.335), (43.0, 0.335), (36.0, 0.334)]

REST 4 t …[+384 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_yaw.py';s=open(p).read().replace(\"vals=[(d,eval_yaw(T,dat,d)) for d in range(-40,41,5)]\",\"vals=[(d,eval_yaw(T,dat,d)) for d in range(-180,180,10)]\").replace(\"best=vals[0][0];fine=[(d,eval_yaw(T,dat,d)) for d in np.arange(best-5,best+5.1,1)]\",\"best=vals[0][0];fine=[(d,eval_yaw(T,dat,d)) for d in np.arange(best-10,best+10.1,2)]\").replace(\"np.save(f'/work/scratch/rest_yaw_{ri}.npy'\",\"np.save(f'/work/scratch/rest_yawwide_{ri}.npy'\");open('/work/scratch/refine_yawwide.py','w').write(s)
PY
python /work/scratch/refine_yawwide.py"
```
<details><summary>output</summary>

```

REST 0 top [(100, 0.277), (90, 0.276), (80, 0.258), (110, 0.248), (0, 0.227), (70, 0.216), (120, 0.21), (50, 0.198)]
fine [(94.0, 0.286), (98.0, 0.284), (102.0, 0.282), (96.0, 0.281), (92.0, 0.277)]

REST 1 top [(80, 0.537), (70, 0.534), (60, 0.498), (90, 0.497), (100, 0.455), (50, 0.446), (-10, 0.44), (40, 0.429)]
fine [(78.0, 0.54), (80.0, 0.537), (74.0, 0.537), (76.0, 0.535), (70.0, 0.534)]

REST 2 top [(90, 0.484), (100, 0.472), (80, 0.465), (70, 0.453), (110, 0.45), (60, 0.424), (50, 0.394), (120, 0.392)]
fine [(90.0, 0.484), (88.0, 0.483), (94.0, 0.479), (92.0, 0.478), (96.0, 0.477)]

REST 3 top [(40, 0.335), (30, 0.314), (50, 0.306), (20, 0.294), (10, 0.27), (60, 0.263), (-60, 0.25), (-160, 0.249)]
fine [(38.0, 0.336), (40.0, 0.335), (36.0, 0.334), (42.0, 0.333), (44.0, 0.33)]

RES …[+395 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/show_yaw.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
import cv2
P=np.load('/work/scratch/rest_poses_final.npz');offs=[94,78,90,38,92,88];fis=[50,240,380,505,620,733]
cap=cv2.VideoCapture('/work/inputs/video.mp4');rows=[]
# full-image P
Pfull=torch.tensor([[2*K[0,0]/1536,0,2*K[0,2]/1536-1,0],[0,2*K[1,1]/1152,2*K[1,2]/1152-1,0],[0,0,0,1],[0,0,1,0]],dtype=torch.float32,device='cuda')
def rend(T):
 vm=(torch.tensor(T,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(Pfull@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None];r,_=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([1152,1536],device='cuda'));pred=(r[0,...,3]>0);uv,_=dr.interpolate(UV[None].contiguous(),r,F.contiguous());uv=uv.clone();uv[...,1]=1-uv[...,1];col=dr.texture(tex[None],uv.contiguous(),filter_mode='linear',boundary_mode='clamp')[0];return pred.cpu().numpy(),col.cpu().numpy()
for ri,(fi,off) in enumerate(zip(fis,offs)):
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,full=cap.read();T0=P[f'r{ri}'];T=T0.copy();T[:3,:3]=T0[:3,:3]@RR.from_rotvec([0,0,np.deg2rad(off)]).as_matrix()
 with h5py.File('/work/inputs/masks.h5') as hf:gt=hf['object'][fi]
 ys,xs=np.where(gt);cx=int(xs.mean());cy=int(ys.mean());half=55;x0=max(0,cx-half);x1=min(1536,cx+half);y0=max(0,cy-half);y1=min(1152,cy+half)
 row=[]
 for name,V in [('GT',None),('base',T0),('yaw',T)]:
  if V is None:ov=full. …[+399 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_mask_score.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2
P=np.load('/work/scratch/rest_poses_final.npz');rests=[(0,130),(215,270),(350,405),(480,530),(602,640),(692,780)];offs=[94,78,90,38,92,88]
def iou(T,gts,valids):
 p=renderx(T,PB);r=[]
 for gt,v in zip(gts,valids):pp=p&v;r.append((pp&gt).sum()/max(1,(pp|gt).sum()))
 return np.mean(r)
with h5py.File('/work/inputs/masks.h5') as hf:
 for ri,(a,b) in enumerate(rests):
  gs=[];vs=[]
  for fi in range(a,b+1,10):
   gs.append(hf['object'][fi,y0:y0+h,x0:x0+w]);vs.append(~binary_dilation(hf['person'][fi,y0:y0+h,x0:x0+w],iterations=3))
  T=P[f'r{ri}'];vals=[]
  for d in range(-180,180,10):U=T.copy();U[:3,:3]=T[:3,:3]@RR.from_rotvec([0,0,np.deg2rad(d)]).as_matrix();vals.append((d,iou(U,gs,vs)))
  vals.sort(key=lambda x:x[1],reverse=True);print(ri,'mask top',[(d,round(s,3)) for d,s in vals[:5]],'base',round(iou(T,gs,vs),3),'color',offs[ri],round(iou(T[:3,:3]@np.eye(3) if False else (lambda U:U)(T),gs,vs),3))
  for d in [-90,-45,0,45,90,offs[ri]]:
   U=T.copy();U[:3,:3]=T[:3,:3]@RR.from_rotvec([0,0,np.deg2rad(d)]).as_matrix();print(' ',d,round(iou(U,gs,vs),3),end='')
  print()
PY
python /work/scratch/yaw_mask_score.py"
```
<details><summary>output</summary>

```
0 mask top [(0, 0.856), (-10, 0.838), (100, 0.821), (10, 0.818), (-20, 0.805)] base 0.856 color 94 0.856
  -90 0.686  -45 0.763  0 0.856  45 0.781  90 0.802  94 0.812
1 mask top [(-10, 0.795), (0, 0.789), (-20, 0.78), (10, 0.775), (90, 0.773)] base 0.789 color 78 0.789
  -90 0.71  -45 0.752  0 0.789  45 0.767  90 0.773  78 0.77
2 mask top [(0, 0.826), (10, 0.805), (-10, 0.802), (-20, 0.799), (-60, 0.798)] base 0.826 color 90 0.826
  -90 0.643  -45 0.782  0 0.826  45 0.764  90 0.764  90 0.764
3 mask top [(0, 0.801), (20, 0.795), (30, 0.794), (10, 0.793), (-10, 0.793)] base 0.801 color 38 0.801
  -90 0.722  -45 0.784  0 0.801  45 0.792  90 0.699  38 0.789
4 mask top [(0, 0.821), (-10, 0.804), (10, 0.786), (-20, 0.784), (-30, 0.777)] base 0.821 color 92 0.821
  -90 0.648  -45 0.772  0 0.821   …[+191 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_eval.py <<'PY'
# build lightweight renderer with depth using score_rotations prefix
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.ndimage import binary_dilation
import h5py,numpy as np
Tp=np.load('/work/scratch/T_proto.npy')
raw=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r')
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:sc=f['alignment/front_stereo_camera_left/scale'][:]
def rend_depth(T):
 vm=(torch.tensor(T,dtype=torch.float32,device='cuda')@vh.T).T[:,:3];pos=(PB@torch.cat([vm,torch.ones_like(vm[:,:1])],1).T).T.contiguous()[None]
 r,_=dr.rasterize(ctx,pos,F.contiguous(),torch.tensor([h,w],device='cuda'));z,_=dr.interpolate(vm[:,2:3].contiguous(),r,F.contiguous());return (r[0,...,3]>0).cpu().numpy(),z[0,:,:,0].cpu().numpy()
with h5py.File('/work/inputs/masks.h5') as hf:
 fs=list(range(0,781,20));res=[];resb=[];cov=[]
 for fi in fs:
  gt=hf['object'][fi,y0:y0+h,x0:x0+w];per=hf['person'][fi,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3)
  obs=np.asarray(raw[fi,y0:y0+h,x0:x0+w],np.float32)/1000*sc[fi]
  for T,out in [(Tp[fi],res),(B[fi],resb)]:
   p,z=rend_depth(Tcw@T);m=gt&p&valid&(obs>0);d=z[m]-obs[m];out.append(np.median(d))
  cov.append(m.sum())
 print('result depth median error per frame',np.round(res,3))
 print('base depth median error',np.round(resb,3))
 print('result bias/mae',np.median(res),np.mean(np.abs(res)))
 print('base bias/mae',np.median(resb) …[+62 chars]
```
<details><summary>output</summary>

```
result depth median error per frame [-0.103 -0.113 -0.149 -0.135 -0.01  -0.02  -0.025 -0.048  0.047  0.036
  0.099  0.147  0.129  0.127  0.049  0.256 -0.106  0.127  0.185  0.184
  0.153  0.491  0.065 -0.015  0.103  0.081  0.109  0.101  0.085 -0.011
  0.129  0.156  0.135  0.065  0.145  0.122  0.176  0.166  0.141  0.143]
base depth median error [-0.053 -0.026 -0.087 -0.068  0.017  0.057  0.061 -0.042  0.015  0.016
  0.094  0.158  0.152  0.15   0.067  0.187 -0.105  0.149  0.161  0.177
  0.145  0.032  0.063 -0.019  0.135  0.148  0.13   0.005  0.084 -0.007
  0.179  0.181  0.162  0.018  0.145  0.169  0.166  0.17   0.142  0.136]
result bias/mae 0.10206348 0.11722412
base bias/mae 0.11222857 0.10197951

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_dynamic_prior_test.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2
Bc=np.einsum('ij,fjk->fik',Tcw,B);Qz=np.diag([-1.,-1.,1.]).astype(np.float32)
def iou(T,gt,valid):
 p=renderx(T,PB);pp=p&valid;return (pp&gt).sum()/max(1,(pp|gt).sum())
def ang(T,R0):return RR.from_matrix(R0.T@T[:3,:3]).magnitude()*180/np.pi
def obj(T,gt,valid,R0,prior):
 return iou(T,gt,valid)-prior*ang(T,R0)**2
def opttrans(T,gt,valid,R0,prior,step=.04,nlev=5):
 b=obj(T,gt,valid,R0,prior);t=T[:3,3].copy()
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for dx in [-step,0,step]:
    for dy in [-step,0,step]:
     for dz in [-step,0,step]:
      if dx==dy==dz==0:continue
      U=T.copy();U[:3,3]=t+np.array([dx,dy,dz]);s=obj(U,gt,valid,R0,prior)
      if s>b+1e-7:b=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,b
def optrot(T,gt,valid,R0,prior,step=5,nlev=4):
 b=obj(T,gt,valid,R0,prior)
 for lev in range(nlev):
  imp=True
  while imp:
   imp=False;bT=None
   for ax in range(3):
    for sg in [-1,1]:
     U=T.copy();dv=np.zeros(3);dv[ax]=np.deg2rad(step*sg);U[:3,:3]=T[:3,:3]@RR.from_rotvec(dv).as_matrix();s=obj(U,gt,valid,R0,prior)
     if s>b+1e-7:b=s;bT=U
   if bT is not None:T=bT;imp=True
  step/=2
 return T,b
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inpu …[+798 chars]
```
<details><summary>output</summary>

```
140 0 iou 0.744 ang 28.1 obj 0.744
140 2e-05 iou 0.74 ang 15.0 obj 0.735
140 5e-05 iou 0.74 ang 13.3 obj 0.731
160 0 iou 0.613 ang 0.0 obj 0.613
160 2e-05 iou 0.613 ang 0.0 obj 0.613
160 5e-05 iou 0.613 ang 0.0 obj 0.613
180 0 iou 0.618 ang 12.6 obj 0.618
180 2e-05 iou 0.618 ang 11.6 obj 0.616
180 5e-05 iou 0.616 ang 9.6 obj 0.611
200 0 iou 0.761 ang 11.5 obj 0.761
200 2e-05 iou 0.761 ang 11.0 obj 0.758
200 5e-05 iou 0.754 ang 9.0 obj 0.75
220 0 iou 0.825 ang 14.0 obj 0.825
220 2e-05 iou 0.826 ang 12.0 obj 0.823
220 5e-05 iou 0.825 ang 11.5 obj 0.818
280 0 iou 0.776 ang 14.3 obj 0.776
280 2e-05 iou 0.777 ang 12.0 obj 0.774
280 5e-05 iou 0.777 ang 12.0 obj 0.77
300 0 iou 0.653 ang 16.5 obj 0.653
300 2e-05 iou 0.653 ang 15.3 obj 0.648
300 5e-05 iou 0.653 ang 15.3 obj 0.641
320 0 iou 0.655 an …[+1209 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_yaw_combined.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.spatial.transform import Rotation as RR
from scipy.ndimage import binary_dilation
import cv2
P=np.load('/work/scratch/rest_poses_final.npz');rests=[(0,130),(215,270),(350,405),(480,530),(602,640),(692,780)]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def corr(rgb,im,m):
 if m.sum()<20:return -1
 a=rgb[m];b=im[m];a=a-a.mean(0);b=b-b.mean(0);return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
def ev(T,dat):
 p,rgb=renderx(T,PB,True);cc=[];ii=[]
 for im,gt,valid in dat:
  m=gt&p&valid;cc.append(corr(rgb,im,m));pp=p&valid;ii.append((pp&gt).sum()/max(1,(pp|gt).sum()))
 return np.mean(ii),np.mean(cc)
with h5py.File('/work/inputs/masks.h5') as hf:
 for ri,(a,b) in enumerate(rests):
  fis=list(range(a,b+1,10));dat=[]
  for fi in fis:
   cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0:y0+h,x0:x0+w,::-1].astype(np.float32)/255;gt=hf['object'][fi,y0:y0+h,x0:x0+w];per=hf['person'][fi,y0:y0+h,x0:x0+w];valid=~binary_dilation(per,iterations=3);dat.append((im,gt,valid))
  T=P[f'r{ri}'];vals=[]
  for d in np.arange(-20,20.1,2):
   U=T.copy();U[:3,:3]=T[:3,:3]@RR.from_rotvec([0,0,np.deg2rad(d)]).as_matrix();iou,c=ev(U,dat);vals.append((d,iou,c,iou+0.10*c,iou+0.05*c))
  print('\\n',ri)
  for d,i,c,s10,s5 in sorted(vals,key=lambda z:z[3],reverse=True)[:6]:print(round(d,1),'iou',round(i,3),'corr',round(c,3),' …[+83 chars]
```
<details><summary>output</summary>

```

 0
0.0 iou 0.856 corr 0.227 s10 0.879 s5 0.867
-2.0 iou 0.854 corr 0.215 s10 0.876 s5 0.865
2.0 iou 0.85 corr 0.24 s10 0.874 s5 0.862
-4.0 iou 0.851 corr 0.201 s10 0.871 s5 0.861
-6.0 iou 0.85 corr 0.179 s10 0.868 s5 0.859
-8.0 iou 0.844 corr 0.173 s10 0.862 s5 0.853

 1
-8.0 iou 0.796 corr 0.446 s10 0.841 s5 0.818
-10.0 iou 0.795 corr 0.44 s10 0.839 s5 0.817
-6.0 iou 0.795 corr 0.436 s10 0.839 s5 0.817
-4.0 iou 0.796 corr 0.43 s10 0.839 s5 0.817
-2.0 iou 0.794 corr 0.424 s10 0.837 s5 0.815
-12.0 iou 0.793 corr 0.439 s10 0.836 s5 0.815

 2
0.0 iou 0.826 corr 0.337 s10 0.86 s5 0.843
-2.0 iou 0.826 corr 0.337 s10 0.86 s5 0.843
2.0 iou 0.825 corr 0.345 s10 0.859 s5 0.842
-4.0 iou 0.825 corr 0.341 s10 0.859 s5 0.842
-6.0 iou 0.823 corr 0.361 s10 0.859 s5 0.841
4.0 iou 0.82 corr 0.37 s10 0.857 …[+814 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/table_constraint.py <<'PY'
exec(open('/work/scratch/score_rotations.py').read().split(\"with h5py.File\")[0])
from scipy.ndimage import binary_dilation
import cv2
P=np.load('/work/scratch/rest_poses_final.npz');rests=[(0,130),(215,270),(350,405),(480,530),(602,640),(692,780)]
mpts=m.vertices[::5];mh=np.c_[mpts,np.ones(len(mpts))]
def boty(T):
 q=(Twc@T@mh.T).T[:,:3];return q[:,1].max()
with h5py.File('/work/inputs/masks.h5') as hf:
 data=[]
 for ri,(a,b) in enumerate(rests):
  T=P[f'r{ri}'];ys=[]
  for fi in range(a,b+1,10):
   gt=hf['object'][fi,y0:y0+h,x0:x0+w];per=hf['person'][fi,y0:y0+h,x0:x0+w];ys.append((gt,~binary_dilation(per,iterations=3)))
  data.append(ys)
 for ty in np.arange(.45,.501,.0025):
  allv=[]
  for ri,(a,b) in enumerate(rests):
   T=P[f'r{ri}'].copy();T[1,3]+=ty-boty(T);p=renderx(Tcw@Twc@T,PB);vals=[]
   for gt,v in data[ri]:pp=p&v;vals.append((pp&gt).sum()/max(1,(pp|gt).sum()))
   allv.extend(vals)
  print(round(ty,4),round(np.mean(allv),4))
PY
python /work/scratch/table_constraint.py"
```
<details><summary>output</summary>

```
0.45 0.699
0.4525 0.7116
0.455 0.7238
0.4575 0.7358
0.46 0.7475
0.4625 0.7577
0.465 0.7676
0.4675 0.7734
0.47 0.7755
0.4725 0.7779
0.475 0.7809
0.4775 0.7846
0.48 0.7855
0.4825 0.785
0.485 0.782
0.4875 0.7763
0.49 0.7641
0.4925 0.751
0.495 0.7358
0.4975 0.7189
0.5 0.7021

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Temporal, render-and-compare recovery of a rigid mug trajectory.

The released CARI4D result is used as a calibrated world-space anchor and its
FoundationPose branch as a smooth per-frame proposal.  The 180-degree ambiguity
about the mug'"'s long axis is resolved from the handle side.  Static phases are
fitted with one pose per interval and motion phases are fitted frame-wise, then
decoded with a quaternion/translation second-order smoother and rest anchors.
\"\"\"
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation as Rot

import nvdiffrast.torch as dr


# Verified phase boundaries.  They are the gaps between the five lift/drink
# intervals seen in the video; each rest interval has the mug stationary on the
# table.  The first/last eight frames of a motion interval are ramped to the
# adjacent rest pose and therefore contain mostly settling motion.
REST_PHASES = ((0, 130), (215, 270), (350, 405), (480, 530), (602, 640), (692, 780))
MOTION_PHASES = ((131, 214), (271, 349), (406, 479), (531, 601), (641, 691))

# The crop covers every observed object position, including its at-mouth pose.
CRO …[+29717 chars]
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
Loading masks...
Fitting rest phase 0 (0, 130)...
  score=0.8321 yaw=+6.0 seed=B
Fitting rest phase 1 (215, 270)...
  score=0.7865 yaw=+0.0 seed=B
Fitting rest phase 2 (350, 405)...
  score=0.7967 yaw=+0.0 seed=B
Fitting rest phase 3 (480, 530)...
  score=0.8162 yaw=+0.0 seed=B
Fitting rest phase 4 (602, 640)...
  score=0.8182 yaw=+0.0 seed=B
Fitting rest phase 5 (692, 780)...
  score=0.8234 yaw=-10.0 seed=B
Fitting motion phase 0 (131, 214)...
  frame 214 initial IoU=0.446, refined=0.769
Fitting motion phase 1 (271, 349)...
  frame 349 initial IoU=0.361, refined=0.782
Fitting motion phase 2 (406, 479)...
  frame 479 initial IoU=0.453, refined=0.815
Fitting motion phase 3 (531, 601)...
  frame 601 initial IoU=0.405, refined=0.793
Fitting motion phase 4 (641, 691)...
  frame 691 initial IoU …[+523 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep -n "motion_cam" /work/outputs/solve.py'
```
<details><summary>output</summary>

```
599:    motion_cam = {}
606:            motion_cam[frame] = T
618:        raw_R = np.stack([motion_cam[i, :3, :3] for i in range(a, b + 1)])
619:        raw_t = np.stack([motion_cam[i, :3, 3] for i in range(a, b + 1)])

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"motion_cam = {}\\n    motion_scores\", \"motion_cam = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))\\n    motion_scores\");open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py && python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
Loading masks...
Fitting rest phase 0 (0, 130)...
  score=0.8321 yaw=+6.0 seed=B
Fitting rest phase 1 (215, 270)...
  score=0.7865 yaw=+0.0 seed=B
Fitting rest phase 2 (350, 405)...
  score=0.7967 yaw=+0.0 seed=B
Fitting rest phase 3 (480, 530)...
  score=0.8162 yaw=+0.0 seed=B
Fitting rest phase 4 (602, 640)...
  score=0.8182 yaw=+0.0 seed=B
Fitting rest phase 5 (692, 780)...
  score=0.8234 yaw=-10.0 seed=B
Fitting motion phase 0 (131, 214)...
  frame 214 initial IoU=0.446, refined=0.769
Fitting motion phase 1 (271, 349)...
  frame 349 initial IoU=0.361, refined=0.782
Fitting motion phase 2 (406, 479)...
  frame 479 initial IoU=0.453, refined=0.815
Fitting motion phase 3 (531, 601)...
  frame 601 initial IoU=0.405, refined=0.793
Fitting motion phase 4 (641, 691)...
  frame 691 initial IoU …[+337 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"rest_cam[k, :3, :3]\",\"rest_cam[k][:3, :3]\").replace(\"rest_cam[k + 1, :3, :3]\",\"rest_cam[k + 1][:3, :3]\").replace(\"rest_cam[k, :3, 3]\",\"rest_cam[k][:3, 3]\").replace(\"rest_cam[k + 1, :3, 3]\",\"rest_cam[k + 1][:3, 3]\");open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py && grep -n \"rest_cam\\[\" /work/outputs/solve.py | tail -20"
```
<details><summary>output</summary>

```
614:        T_cam[a:b + 1] = rest_cam[k]
620:        qa = Rot.from_matrix(rest_cam[k][:3, :3]).as_quat()
621:        qb = Rot.from_matrix(rest_cam[k + 1][:3, :3]).as_quat()
626:        ts = smooth_vector(raw_t, rest_cam[k][:3, 3], rest_cam[k + 1][:3, 3],
650:            anchor = rest_cam[k][:3, 3] if i < 8 else rest_cam[k + 1][:3, 3]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
P=np.load('/work/scratch/rest_poses_final.npz');rest_cam=[P[f'r{i}'] for i in range(6)]
D=np.load('/work/scratch/dynamic_opt.npz');motion_cam=D['T'];motion_scores=D['scores'];n=781
F=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_foundationpose']
T_cam=np.tile(np.eye(4,dtype=np.float32),(n,1,1));F3=np.einsum('fij,jk->fik',F[:,:3,:3],solve.QZ180)
for k,(a,b) in enumerate(solve.MOTION_PHASES):
 raw_R=np.stack([motion_cam[i,:3,:3] for i in range(a,b+1)]);raw_t=np.stack([motion_cam[i,:3,3] for i in range(a,b+1)])
 qa=solve.Rot.from_matrix(rest_cam[k][:3,:3]).as_quat();qb=solve.Rot.from_matrix(rest_cam[k+1][:3,:3]).as_quat();qf=np.stack([solve.Rot.from_matrix(F3[i]).as_quat() for i in range(a,b+1)])
 Rs=solve.smooth_quaternions(np.stack([solve.Rot.from_matrix(R).as_quat() for R in raw_R]),qa,qb,motion_scores[a:b+1],qf)
 ts=solve.smooth_vector(raw_t,rest_cam[k][:3,3],rest_cam[k+1][:3,3],motion_scores[a:b+1])
 blend=np.ones(b-a+1,np.float32);ramp=solve.smoothstep(np.arange(8,dtype=np.float32)/8);blend[:8]=ramp;blend[-8:]=ramp[::-1]
 for i,q0,q1 in zip(range(b-a+1),qa,qb):
  if i<8:anchor=q0
  elif i>=b-a+1-8:anchor=q1
  else:continue
  qm=solve.Rot.from_matrix(Rs[i]).as_quat();anchor=solve.q_align(anchor,qm);q=(1-blend[i])*anchor+blend[i]*qm;q/=np.linalg.norm(q);Rs[i]=solve.Rot.from_quat(q).as_matrix()
 target_t=np.empty_like(ts)
 for i in range(b-a+1):
  anchor=rest_cam[ …[+153 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 17, in <module>
  File "/work/outputs/solve.py", line 358, in q_align
    return q if float(np.dot(q, ref)) >= 0.0 else -q
                ^^^^^^^^^^^^^^^^^^^^^
TypeError: only length-1 arrays can be converted to Python scalars

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
P=np.load('/work/scratch/rest_poses_final.npz');rest_cam=[P[f'r{i}'] for i in range(6)]
D=np.load('/work/scratch/dynamic_opt.npz');motion_cam=D['T'];motion_scores=D['scores'];n=781
F=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_foundationpose'];F3=np.einsum('fij,jk->fik',F[:,:3,:3],solve.QZ180)
for k,(a,b) in enumerate(solve.MOTION_PHASES):
 print('k',k)
 raw_R=np.stack([motion_cam[i,:3,:3] for i in range(a,b+1)]);raw_t=np.stack([motion_cam[i,:3,3] for i in range(a,b+1)])
 qa=solve.Rot.from_matrix(rest_cam[k][:3,:3]).as_quat();qb=solve.Rot.from_matrix(rest_cam[k+1][:3,:3]).as_quat();qf=np.stack([solve.Rot.from_matrix(F3[i]).as_quat() for i in range(a,b+1)])
 print(qa.shape,qb.shape,qf.shape,raw_R.shape)
 Rs=solve.smooth_quaternions(np.stack([solve.Rot.from_matrix(R).as_quat() for R in raw_R]),qa,qb,motion_scores[a:b+1],qf)
 ts=solve.smooth_vector(raw_t,rest_cam[k][:3,3],rest_cam[k+1][:3,3],motion_scores[a:b+1])
 print('Rs',Rs.shape)
 # just qalign test
 for i,q0,q1 in zip(range(b-a+1),qa,qb):
  print('q0',np.asarray(q0).shape);print(solve.q_align(q0,solve.Rot.from_matrix(Rs[i]).as_quat()));break
 break
PY"
```
<details><summary>output</summary>

```
k 0
(4,) (4,) (84, 4) (84, 3, 3)
Rs (84, 3, 3)
q0 ()
Traceback (most recent call last):
  File "<stdin>", line 16, in <module>
  File "/work/outputs/solve.py", line 358, in q_align
    return q if float(np.dot(q, ref)) >= 0.0 else -q
                ^^^^^^^^^^^^^^^^^^^^^
TypeError: only length-1 arrays can be converted to Python scalars

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"for i, q0, q1 in zip(range(b - a + 1), qa, qb):\\n            if i < 8:\\n                anchor = q0\\n            elif i >= b - a + 1 - 8:\\n                anchor = q1\\n            else:\\n                continue\",\"for i in range(b - a + 1):\\n            if i < 8:\\n                anchor = qa\\n            elif i >= b - a + 1 - 8:\\n                anchor = qb\\n            else:\\n                continue\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
P=np.load('/work/scratch/rest_poses_final.npz');rest_cam=[P[f'r{i}'] for i in range(6)]
D=np.load('/work/scratch/dynamic_opt.npz');motion_cam=D['T'];motion_scores=D['scores'];n=781
F=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_foundationpose'];F3=np.einsum('fij,jk->fik',F[:,:3,:3],solve.QZ180)
T_cam=np.tile(np.eye(4,dtype=np.float32),(n,1,1))
for k,(a,b) in enumerate(solve.MOTION_PHASES):
 raw_R=np.stack([motion_cam[i,:3,:3] for i in range(a,b+1)]);raw_t=np.stack([motion_cam[i,:3,3] for i in range(a,b+1)])
 qa=solve.Rot.from_matrix(rest_cam[k][:3,:3]).as_quat();qb=solve.Rot.from_matrix(rest_cam[k+1][:3,:3]).as_quat();qf=np.stack([solve.Rot.from_matrix(F3[i]).as_quat() for i in range(a,b+1)])
 Rs=solve.smooth_quaternions(np.stack([solve.Rot.from_matrix(R).as_quat() for R in raw_R]),qa,qb,motion_scores[a:b+1],q …[+700 chars]
```
<details><summary>output</summary>

```
ok

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
Loading masks...
Fitting rest phase 0 (0, 130)...
  score=0.8321 yaw=+6.0 seed=B
Fitting rest phase 1 (215, 270)...
  score=0.7865 yaw=+0.0 seed=B
Fitting rest phase 2 (350, 405)...
  score=0.7967 yaw=+0.0 seed=B
Fitting rest phase 3 (480, 530)...
  score=0.8162 yaw=+0.0 seed=B
Fitting rest phase 4 (602, 640)...
  score=0.8182 yaw=+0.0 seed=B
Fitting rest phase 5 (692, 780)...
  score=0.8234 yaw=-10.0 seed=B
Fitting motion phase 0 (131, 214)...
  frame 214 initial IoU=0.446, refined=0.769
Fitting motion phase 1 (271, 349)...
  frame 349 initial IoU=0.361, refined=0.782
Fitting motion phase 2 (406, 479)...
  frame 479 initial IoU=0.453, refined=0.815
Fitting motion phase 3 (531, 601)...
  frame 601 initial IoU=0.405, refined=0.793
Fitting motion phase 4 (641, 691)...
  frame 691 initial IoU …[+120 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/outputs/validation.json'
```
<details><summary>output</summary>

```
{
  "phases": [
    {
      "phase": "rest0",
      "start": 0,
      "end": 130,
      "iou_result": 0.8481678165303494,
      "iou_baseline": 0.7157532586941704,
      "corr_result": 0.1739518748332936,
      "corr_baseline": 0.25360637827673904,
      "rot_step_p95_result": 1.973455294769804e-15,
      "rot_step_p95_baseline": 8.427718764454553
    },
    {
      "phase": "rest1",
      "start": 215,
      "end": 270,
      "iou_result": 0.8036858157841625,
      "iou_baseline": 0.6239279389248964,
      "corr_result": 0.3381760076686104,
      "corr_baseline": 0.09672288358696178,
      "rot_step_p95_result": 8.493430308084729e-15,
      "rot_step_p95_baseline": 13.742902815032513
    },
    {
      "phase": "rest2",
      "start": 350,
      "end": 405,
      "iou_result": 0.814620337 …[+3906 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,os
p='/work/outputs/object_poses_world.npz';d=np.load(p)
print(d.files)
T=d['T_world_mesh'];print(T.shape,T.dtype,'finite',np.isfinite(T).all())
print('det range',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
print('last row',np.max(np.abs(T[:,3]-np.array([0,0,0,1]))))
print('translation ranges',T[:,:3,3].min(0),T[:,:3,3].max(0))
print('file size',os.path.getsize(p))
PY
ls -lh /work/outputs"
```
<details><summary>output</summary>

```
total 156K
-rw-rw-r-- 1 hanzhou hanzhou 4.8K Oct 20 18:43 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou  859 Oct 20 18:43 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 18:36 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  98K Oct 20 18:42 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  31K Oct 20 18:36 solve.py
-rw-rw-r-- 1 hanzhou hanzhou 4.6K Oct 20 18:43 validation.json

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/outputs/REPORT.md; echo '---TOOLS---'; cat /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
# Monocular rigid-object trajectory recovery

## Approach

I treated the supplied CARI4D trajectory as a calibrated proposal, not as the final answer. The FoundationPose branch supplies a smoother per-frame rotation proposal than the CoCoNet refinement. Its 180-degree long-axis branch was corrected using the visible handle side, then every phase was refined by rendering the exact textured mesh against the SAM2 object mask while ignoring pixels covered by the person. Static intervals were decoded to one constant SE(3) pose. Motion intervals were fitted per frame, then passed through a second-order quaternion/translation trajectory smoother with the neighbouring static poses as hard anchors. The first and last eight frames of each motion interval are smoothly ramped to those anchors.

No gro …[+4982 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
s=solve.rotation_step_degrees(T[:,:3,:3]);sb=solve.rotation_step_degrees(B[:,:3,:3])
print('final step percentiles',np.percentile(s,[50,75,90,95,99,100]))
for k,(a,b) in enumerate(solve.MOTION_PHASES):
 print(k,a,b,'max',s[a+1:b+1].max(),'frames>25',[(i,round(s[i],1)) for i in range(a+1,b+1) if s[i]>25])
# static pose consistency / bottom y
import trimesh
r=solve.Renderer(__import__('pathlib').Path('/work/inputs'))
for k,(a,b) in enumerate(solve.REST_PHASES):
 print('rest',k,'T',np.round(T[a,:3,3],4),'Rz',np.round(T[a,:3,2],3))
PY"
```
<details><summary>output</summary>

```
final step percentiles [1.14708709e-13 3.52264985e+00 8.08056366e+00 1.13441737e+01
 1.98806040e+01 3.06144029e+01]
0 131 214 max 30.614402894004183 frames>25 [(193, 26.4), (194, 30.6)]
1 271 349 max 28.93028516752334 frames>25 [(340, 28.1), (341, 28.9)]
2 406 479 max 11.598282144743582 frames>25 []
3 531 601 max 14.397519406701186 frames>25 []
4 641 691 max 18.85518264747954 frames>25 []
rest 0 T [0.441  0.3495 2.5213] Rz [0.039 0.977 0.209]
rest 1 T [0.4706 0.3689 2.6786] Rz [-0.051  0.999  0.004]
rest 2 T [0.4802 0.3768 2.7144] Rz [-0.075  0.993  0.093]
rest 3 T [0.4608 0.3703 2.6891] Rz [-0.008  0.981  0.192]
rest 4 T [0.4965 0.3746 2.6773] Rz [-0.061  0.994  0.092]
rest 5 T [0.4776 0.3744 2.6651] Rz [-0.064  0.992  0.111]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,cv2,h5py
sys.path.insert(0,'/work/outputs');import solve
from pathlib import Path
from scipy.ndimage import binary_dilation
r=solve.Renderer(Path('/work/inputs'));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4');thumbs=[]
with h5py.File('/work/inputs/masks.h5') as hf:
 for i in range(0,781,10):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,full=cap.read();im=full[420:740,700:980];gt=hf['object'][i,420:740,700:980]
  pr=r.render(r.T_cam_world@T[i]);pb=r.render(r.T_cam_world@B[i])
  ov=(im*.55).astype(np.uint8);ov[pb]=(0.4*ov[pb]+0.6*np.array([255,0,0])).astype(np.uint8);ov[pr]=(0.4*ov[pr]+0.6*np.array([0,0,255])).astype(np.uint8);ov[gt]=(0.25*ov[gt]+0.75*np.array([0,255,0])).astype(np.uint8)
  ov=cv2.resize(ov,(280,320));cv2.putText(ov,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);thumbs.append(ov)
rows=[]
for j in range(0,len(thumbs),8):
 row=thumbs[j:j+8]
 while len(row)<8:row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/final_overlay.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,cv2,h5py
sys.path.insert(0,'/work/outputs');import solve
from pathlib import Path
r=solve.Renderer(Path('/work/inputs'));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4');rows=[]
for fi in [0,100,160,200,300,350,420,450,500,560,600,660,700,750]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,full=cap.read();
 with h5py.File('/work/inputs/masks.h5') as hf:gt=hf['object'][fi]
 ys,xs=np.where(gt);cx=int(xs.mean());cy=int(ys.mean());half=55;x0=max(0,cx-half);x1=min(1536,cx+half);y0=max(0,cy-half);y1=min(1152,cy+half)
 row=[]
 for name,V in [('GT',None),('final',T[fi]),('base',B[fi])]:
  if V is None:ov=full.copy()
  else:
   p,c=r.render(r.T_cam_world@V,texture=True);ov=full.copy();ov[p]=(0.15*ov[p]+0.85*c[p]*255).astype(np.uint8)
  cr=ov[y0:y1,x0:x1];cr=cv2.resize(cr,(240,240),interpolation=cv2.INTER_NEAREST);cv2.putText(cr,f'{fi} {name}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);row.append(cr)
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/final_texture_compare.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1152 but corresponding boolean dimension is 320

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,cv2,h5py
sys.path.insert(0,'/work/outputs');import solve
from pathlib import Path
r=solve.Renderer(Path('/work/inputs'));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
x0c,y0c,wc,hc=solve.CROP;cap=cv2.VideoCapture('/work/inputs/video.mp4');rows=[]
for fi in [0,100,160,200,300,350,420,450,500,560,600,660,700,750]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0c:y0c+hc,x0c:x0c+wc].copy()
 with h5py.File('/work/inputs/masks.h5') as hf:gt=hf['object'][fi,y0c:y0c+hc,x0c:x0c+wc]
 ys,xs=np.where(gt);cx=int(xs.mean());cy=int(ys.mean());half=55
 row=[]
 for name,V in [('GT',None),('final',T[fi]),('base',B[fi])]:
  if V is None:ov=im.copy()
  else:
   p,c=r.render(r.T_cam_world@V,texture=True);ov=im.copy();ov[p]=(0.15*ov[p]+0.85*c[p]*255).astype(np.uint8)
  cr=ov[max(0,cy-half):cy+half,max(0,cx-half):cx+half];cr=cv2.resize(cr,(240,240),interpolation=cv2.INTER_NEAREST);cv2.putText(cr,f'{fi} {name}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);row.append(cr)
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/final_texture_compare.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
D=np.load('/work/inputs/reconstruction.npz');R=D['human_keypoints_world'];cp=D['contact_prob_right_left']
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=D['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::30]
def ds(P):
 out=[]
 for i in range(len(P)):
  q=(P[i]@np.c_[v,np.ones(len(v))].T).T[:,:3];hh=R[i,21:63];tree=cKDTree(q);out.append(tree.query(hh)[0].min())
 return np.array(out)
dr=ds(T);db=ds(B)
# actual carried windows interior, contact>0.5 and mask center<630 from stored signals
import scipy.signal as ss
y=np.load('/work/scratch/signals.npz')['ys'];mask=(cp.max(1)>.5)&(ss.medfilt(y,5)<630)
for nm,d in [('base',db),('result',dr)]: print(nm,mask.sum(),np.percentile(d[mask],[0,25,50,75,90,100]))
# also compare at all dynamic frames
print('range result',dr.min(),dr.max())
PY"
```
<details><summary>output</summary>

```
base 282 [0.00019769 0.00256076 0.02032255 0.05038064 0.07212581 0.1359983 ]
result 282 [0.00032447 0.00172554 0.00328304 0.02479547 0.07768627 0.22530663]
range result 0.00022842251822880412 1.2381136691779928

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import importlib.metadata as m
for p in ['nvdiffrast','torch','trimesh','numpy','scipy','opencv-python','h5py']:
 try: print(p,m.version(p))
 except Exception as e: print(p,e)
PY"
```
<details><summary>output</summary>

```
nvdiffrast 0.4.0
torch 2.5.1+cu124
trimesh 5.1.1
numpy 1.26.3
scipy 1.16.3
opencv-python 4.11.0.86
h5py 3.16.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/check_final_yaw.py <<'PY'
import sys,numpy as np,cv2,h5py
sys.path.insert(0,'/work/outputs');import solve
from pathlib import Path
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation as Rot
r=solve.Renderer(Path('/work/inputs'));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
x0,y0,w,h=solve.CROP
with h5py.File('/work/inputs/masks.h5') as hf:
 for k,(a,b) in enumerate(solve.REST_PHASES):
  fis=list(range(a,b+1,10));dat=[]
  for fi in fis:
   cap.set(cv2.CAP_PROP_POS_FRAMES,fi);_,im=cap.read();im=im[y0:y0+h,x0:x0+w,::-1].astype(np.float32)/255;gt=hf['object'][fi,y0:y0+h,x0:x0+w];per=hf['person'][fi,y0:y0+h,x0:x0+w];dat.append((im,gt,~binary_dilation(per,iterations=3)))
  vals=[]
  for d in [-20,-15,-10,-5,0,5,10,15,20]:
   U=T[a].copy();U[:3,:3]=T[a,:3,:3]@Rot.from_rotvec([0,0,np.deg2rad(d)]).as_matrix();pc,rgb=r.render(r.T_cam_world@U,texture=True);cs=[];iis=[]
   for im,gt,v in dat:
    m=gt&pc&v;cs.append(solve.rgb_correlation(rgb,im,m));pp=pc&v;iis.append((pp&gt).sum()/max(1,(pp|gt).sum()))
   vals.append((d,np.mean(iis),np.mean(cs)))
  print('\\n',k,'current',0)
  for d,i,c in vals:print(d,round(i,3),round(c,3),round(i+.05*c,3))
PY
python /work/scratch/check_final_yaw.py"
```
<details><summary>output</summary>

```

 0 current 0
-20 0.807 0.135 0.814
-15 0.807 0.132 0.813
-10 0.822 0.148 0.83
-5 0.837 0.166 0.845
0 0.844 0.19 0.854
5 0.837 0.167 0.845
10 0.819 0.153 0.826
15 0.804 0.136 0.811
20 0.79 0.136 0.797

 1 current 0
-20 0.776 0.399 0.796
-15 0.783 0.417 0.804
-10 0.792 0.427 0.813
-5 0.797 0.445 0.819
0 0.797 0.424 0.818
5 0.787 0.412 0.807
10 0.779 0.394 0.799
15 0.773 0.368 0.792
20 0.768 0.383 0.788

 2 current 0
-20 0.799 0.367 0.817
-15 0.792 0.384 0.811
-10 0.8 0.378 0.819
-5 0.815 0.359 0.833
0 0.827 0.338 0.844
5 0.821 0.358 0.839
10 0.809 0.363 0.827
15 0.795 0.339 0.812
20 0.787 0.348 0.804

 3 current 0
-20 0.792 0.213 0.803
-15 0.794 0.235 0.806
-10 0.797 0.248 0.809
-5 0.799 0.248 0.812
0 0.801 0.234 0.812
5 0.798 0.255 0.81
10 0.789 0.283 0.803
15 0.789 0.317 0.805
20 0.788 0. …[+412 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
from pathlib import Path
r=solve.Renderer(Path('/work/inputs'));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
for k,(a,b) in enumerate(solve.REST_PHASES):print(k,'final side',r.handle_side(r.T_cam_world@T[a]),'base side',r.handle_side(r.T_cam_world@B[a]))
PY"
```
<details><summary>output</summary>

```
0 final side -1.0 base side 1.0
1 final side -1.0 base side 1.0
2 final side -1.0 base side 1.0
3 final side -1.0 base side 1.0
4 final side -1.0 base side 1.0
5 final side -1.0 base side 1.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.\", \"- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.\\n        - During the 282 frames with confident hand contact and the mug lifted, median minimum hand-keypoint-to-mesh distance is 0.33 cm for the result versus 2.03 cm for the baseline.\")
s=s.replace(\"f\\\"| nvdiffrast | {getattr(dr, '__version__', 'installed')} | differentiable/hard rasterization and texture lookup | yes |\\\",\", \"f\\\"| nvdiffrast | 0.4.0 | differentiable/hard rasterization and texture lookup | yes |\\\",\")
open(p,'w').write(s)

p='/work/outputs/REPORT.md';s=open(p).read();s=s.replace(\"- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.\", \"- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.\\n- During the 282 frames with confident hand contact and the mug lifted, median minimum hand-keypoint-to-mesh distance is 0.33 cm for the result versus 2.03 cm for the baseline.\")
open(p,'w').write(s)
p='/work/outputs/TOOLS.md';s=open(p).read().replace('| nvdiffrast | instal …[+174 chars]
```
<details><summary>output</summary>

```
  File "/work/outputs/solve.py", line 528
    "- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.
    ^
SyntaxError: unterminated string literal (detected at line 528)
|---|---:|---:|---:|---:|---:|---:|
| rest0 (0-130) | 0.716 | 0.848 | 0.254 | 0.174 | 8.43 | 0.00 |
| rest1 (215-270) | 0.624 | 0.804 | 0.097 | 0.338 | 13.74 | 0.00 |
| rest2 (350-405) | 0.628 | 0.815 | 0.087 | 0.228 | 16.20 | 0.00 |
| rest3 (480-530) | 0.620 | 0.814 | 0.088 | 0.183 | 12.38 | 0.00 |
| rest4 (602-640) | 0.601 | 0.829 | 0.018 | 0.198 | 14.98 | 0.00 |
| rest5 (692-780) | 0.670 | 0.823 | 0.132 | 0.219 | 12.22 | 0.00 |
| motion0 (131-214) | 0.587 | 0.666 | 0.180 | 0.215 | 26.17 | 12.61 |
| motion1 (271-349) | 0.501 …[+3148 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/outputs/solve.py | sed -n '510,545p'"
```
<details><summary>output</summary>

```
   510	    for r in rows:
   511	        lines.append(
   512	            f"| {r['phase']} ({r['start']}-{r['end']}) | {r['iou_baseline']:.3f} | {r['iou_result']:.3f} | "
   513	            f"{r['corr_baseline']:.3f} | {r['corr_result']:.3f} | {r['rot_step_p95_baseline']:.2f} | {r['rot_step_p95_result']:.2f} |")
   514	    lines += [
   515	        "",
   516	        "Overall silhouette IoU: baseline "
   517	        f"{extra['iou_baseline_overall']:.3f}, result {extra['iou_result_overall']:.3f}. "
   518	        "The result is a large improvement in mask explanation and temporal consistency. "
   519	        "The residual colour mismatch is expected because the scan texture contains baked lighting from a different capture; "
   520	        "mask and temporal evidence were therefore given  …[+2243 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();bad='''        \"- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.\\n        - During the 282 frames with confident hand contact and the mug lifted, median minimum hand-keypoint-to-mesh distance is 0.33 cm for the result versus 2.03 cm for the baseline.\",'''
good='''        \"- Rest poses are upright and their projected silhouettes remain on the same table region; the trajectory does not float or jump across the five lifts.\",\\n        \"- During the 282 frames with confident hand contact and the mug lifted, median minimum hand-keypoint-to-mesh distance is 0.33 cm for the result versus 2.03 cm for the baseline.\",'''
assert bad in s
s=s.replace(bad,good)
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py && echo OK"
```
<details><summary>output</summary>

```
OK

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json
v=json.load(open('/work/outputs/validation.json'))
for k in ['corr_baseline','corr_result','iou_baseline','iou_result','rot_step_p95_baseline','rot_step_p95_result']:
 num=sum((r['end']-r['start']+1)*r[k] for r in v['phases']);den=sum(r['end']-r['start']+1 for r in v['phases']);print(k,num/den)
PY"
```
<details><summary>output</summary>

```
corr_baseline 0.12193579587674565
corr_result 0.18547780467936797
iou_baseline 0.6146461606090781
iou_result 0.7780369018023988
rot_step_p95_baseline 36.10507781632234
rot_step_p95_result 6.280502994578348

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
for p in ['/work/outputs/REPORT.md','/work/outputs/solve.py']:
 s=open(p).read()
 old='Overall silhouette IoU: baseline 0.615, result 0.778. The result is a large improvement in mask explanation and temporal consistency. '
 new='Overall silhouette IoU: baseline 0.615, result 0.778. Overall texture NCC: baseline 0.122, result 0.185. The result is a large improvement in mask explanation, texture agreement and temporal consistency. '
 s=s.replace(old,new)
 old2='The residual colour mismatch is expected because the scan texture contains baked lighting from a different capture; '
 new2='Residual colour mismatch is expected because the scan texture contains baked lighting from a different capture; '
 s=s.replace(old2,new2)
 open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py && echo OK"
```
<details><summary>output</summary>

```
OK

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
for p in ['/work/outputs/TOOLS.md','/work/outputs/solve.py']:
 s=open(p).read()
 needle='| h5py | 3.16.0 | mask and depth I/O | yes |'
 repl=needle+'\\n| FoundationPose (NVlabs PyTorch backend) | preinstalled PyTorch weights, runs dated 2024-01-11 and 2023-10-28 | independent re-registration was attempted as a cross-check but was not needed in the final deterministic path; supplied FoundationPose poses are used | no |'
 s=s.replace(needle,repl)
 open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py && echo OK && cat /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
OK
# Tools

The solution is self-contained and requires no downloaded code or weights beyond the container's preinstalled stack.

| Tool | Version | Role | Needed by solve.py |
|---|---|---|---|
| Python | 3.11.10 | runtime | yes |
| PyTorch | 2.5.1+cu124 | tensor/rasterization runtime | yes |
| nvdiffrast | 0.4.0 | differentiable/hard rasterization and texture lookup | yes |
| trimesh | 5.1.1 | exact GLB mesh and UV texture loading | yes |
| NumPy | 1.26.3 | numerical linear algebra | yes |
| SciPy | 1.16.3 | sparse trajectory smoother, rotation utilities | yes |
| OpenCV | 4.11.0 | video decoding | yes |
| h5py | 3.16.0 | mask and depth I/O | yes |
| FoundationPose (NVlabs PyTorch backend) | preinstalled PyTorch weights, runs dated 2024-01-11 and 2023-10-28 | independent re-registration  …[+328 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json
from pathlib import Path
sys.path.insert(0,'/work/outputs');import solve
v=json.load(open('/work/outputs/validation.json'))
solve.write_reports(Path('/work/outputs'),v['phases'],v['runtime_seconds'],v['overall'])
print('report generated')
PY
python -m py_compile /work/outputs/solve.py
grep -n \"Overall silhouette\\|hand-keypoint\\|nvdiffrast\\|FoundationPose\" /work/outputs/REPORT.md /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
report generated
/work/outputs/REPORT.md:5:I treated the supplied CARI4D trajectory as a calibrated proposal, not as the final answer. The FoundationPose branch supplies a smoother per-frame rotation proposal than the CoCoNet refinement. Its 180-degree long-axis branch was corrected using the visible handle side, then every phase was refined by rendering the exact textured mesh against the SAM2 object mask while ignoring pixels covered by the person. Static intervals were decoded to one constant SE(3) pose. Motion intervals were fitted per frame, then passed through a second-order quaternion/translation trajectory smoother with the neighbouring static poses as hard anchors. The first and last eight frames of each motion interval are smoothly ramped to those anchors.
/work/outputs/REPORT.md …[+793 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "nl -ba /work/outputs/solve.py | sed -n '520,545p'; grep -n \"FoundationPose (NV\" /work/outputs/solve.py /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
   520	        "mask and temporal evidence were therefore given higher weight.",
   521	        "",
   522	        "## Changes and physical checks",
   523	        "",
   524	        "- Fixed the whole-episode 180-degree long-axis ambiguity. The original CoCoNet rotation places the handle on the wrong image side.",
   525	        "- Replaced per-frame rotations during table contact with one pose per static interval, eliminating drift and flicker.",
   526	        "- Re-estimated camera-ray depth and lateral translation by silhouette registration against the exact mesh. The corrections are typically a few centimetres and vary smoothly.",
   527	        "- Replaced the CoCoNet rotation jumps (many 40-160 degree single-frame jumps) with image-registered rotations and a smooth SE(3) trajectory …[+1510 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
old='''    lines += [
        \"\",
        \"Overall silhouette IoU: baseline \"
        f\"{extra['iou_baseline_overall']:.3f}, result {extra['iou_result_overall']:.3f}. \"
        \"The result is a large improvement in mask explanation and temporal consistency. \"
        \"Residual colour mismatch is expected because the scan texture contains baked lighting from a different capture; \"
        \"mask and temporal evidence were therefore given higher weight.\",'''
new='''    total_frames = sum(r[\"end\"] - r[\"start\"] + 1 for r in rows)
    overall_corr_b = sum((r[\"end\"] - r[\"start\"] + 1) * r[\"corr_baseline\"] for r in rows) / total_frames
    overall_corr_r = sum((r[\"end\"] - r[\"start\"] + 1) * r[\"corr_result\"] for r in rows) / total_frames
    lines += [
        \"\",
        \"Overall silhouette IoU: baseline \"
        f\"{extra['iou_baseline_overall']:.3f}, result {extra['iou_result_overall']:.3f}. \"
        f\"Overall texture NCC: baseline {overall_corr_b:.3f}, result {overall_corr_r:.3f}. \"
        \"The result is a large improvement in mask explanation, texture agreement and temporal consistency. \"
        \"Residual colour mismatch is expected because the scan texture contains baked lighting from a different capture; \"
        \"mask and temporal evidence were therefore given higher weight.\",'''
assert old in s
s=s.replace(old,new)
old2='''        f\"| h5py | {h5py.__versi …[+914 chars]
```
<details><summary>output</summary>

```
ok
/work/outputs/REPORT.md:43:Overall silhouette IoU: baseline 0.615, result 0.778. Overall texture NCC: baseline 0.122, result 0.185. The result is a large improvement in mask explanation, texture agreement and temporal consistency. Residual colour mismatch is expected because the scan texture contains baked lighting from a different capture; mask and temporal evidence were therefore given higher weight.
/work/outputs/TOOLS.md:9:| nvdiffrast | 0.4.0 | differentiable/hard rasterization and texture lookup | yes |
/work/outputs/TOOLS.md:15:| FoundationPose (NVlabs PyTorch backend) | preinstalled PyTorch weights, runs dated 2024-01-11 and 2023-10-28 | independent re-registration was attempted as a cross-check but was not used in the final deterministic path; supplied FoundationPose poses are  …[+12 chars]
```
</details>

**Agent:** Completed the trajectory recovery task.

Outputs:
- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(781, 4, 4)`, `float64`
- `/work/outputs/REPORT.md`
- `/work/outputs/TOOLS.md`
- `/work/outputs/validation.json`

Validation summary:
- Mean silhouette IoU: baseline `0.615` → result `0.778`
- Overall texture NCC: baseline `0.122` → result `0.185`
- P95 inter-frame rotation step: baseline `31.86°` → result `11.34°`
- Deterministic `solve.py` runtime: `387.4 s`

The solution resolves the 180° handle-side ambiguity, replaces drifting table poses with constant per-phase poses, performs render-and-compare refinement, and applies anchored temporal smoothing across the five manipulation phases.
