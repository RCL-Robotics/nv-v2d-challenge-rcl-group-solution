# T1_ds_ep001

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T15:04:39Z

**shell** (exit 0)
```
/usr/bin/bash -lc 'cat /work/AGENT_TASK.md'
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
/usr/bin/bash -lc "sed -n '1,90p' /work/AGENT_TASK.md"
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
and translation both matter; a  …[+6977 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '91,180p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
- Model weights (read-only) in `/weights`: sam2, foundationpose, sam3d_body, cari4d (CoCoNet), grounding_dino,
  hf_home (MoGe-2). HuggingFace is not reachable; set paths to these local weights.
- `/pipeline` (read-only): this episode's full CARI4D intermediate outputs (depth/moge2 with aligned depth maps and
  intrinsics, foundationpose, mhr_init, export, inference/coconet.pth and refined.pth, work/masks). Note: inside
  these CARI4D files, object poses `pose_abs` are for the mesh re-centred to its oriented bounding box
  (`S, _ = trimesh.bounds.oriented_bounds(mesh)`; mesh-file pose = pose_abs @ S) and use `K_pipeline`.
- The internet (except HuggingFace): you may install packages with pip (`pip install --user ...`), clone code from
  GitHub, and download other public model weights or to …[+2015 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "ls -lah /work/inputs && find /work/inputs -maxdepth 2 -type f -printf '%p %s\\n' | sort"
```
<details><summary>output</summary>

```
total 78M
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 17:36 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 18:58 ..
-rw-rw-r-- 1 hanzhou hanzhou  826 Oct 20 17:36 camera.json
-rw-rw-r-- 1 hanzhou hanzhou   60 Oct 20 17:35 episode.json
-rw-rw-r-- 1 hanzhou hanzhou 187K Oct 20 17:36 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou  398 Oct 20 17:35 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou 6.4M Oct 20 17:36 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou  15M Oct 20 17:35 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou  37M Oct 20 17:36 reconstruction.npz
-rw-rw-r-- 1 hanzhou hanzhou  224 Oct 20 17:35 task.txt
-rw-rw-r-- 1 hanzhou hanzhou  21M Oct 20 17:35 video.mp4
/work/inputs/camera.json 826
/work/inputs/episode.json 60
/work/inputs/frame0_reference_human.npz 190750
/work/inputs/ground_plane.json …[+172 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader && python - <<'PY'
import torch, sys
print(torch.__version__, torch.cuda.is_available())
PY"
```
<details><summary>output</summary>

```
2.5.1+cu124 True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'cat /work/inputs/task.txt; echo; cat /work/inputs/episode.json; echo; cat /work/inputs/camera.json; echo; cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk over to the vacuum leaning on the tall bar stool and pick up the vacuum using your left hand and hold the vacuum infront of you and perform side to side vacuum motions. Place it back leaning on the stool when complete.

{
 "episode_index": 1,
 "object": "vacuum",
 "frames": 747
}
{
 "K": [
  [
   820.0008544921875,
   0.0,
   758.1000366210938
  ],
  [
   0.0,
   820.0008544921875,
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
   831.2747802734375,
   0.0,
   768.0
  ],
  [
   0.0,
   831.2747802734375,
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
   0.9995759647192438,
   -0.011646362686561017,
   0.026688068340846565,
   0.06078298602767497
  ],
  [
   0.010997478792960593,
   0.9996434225332343,
   0.0243327607607422,
   0.049692131 …[+711 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np, h5py, json
p='/work/inputs/reconstruction.npz'
z=np.load(p)
for k in z.files:
 a=z[k]
 print(k,a.shape,a.dtype, (a.min(),a.max()) if np.issubdtype(a.dtype,np.number) else '')
with h5py.File('/work/inputs/masks.h5') as f:
 print('masks',list(f.keys()))
 for k in f: print(k,f[k].shape,f[k].dtype)
PY"
```
<details><summary>output</summary>

```
T_world_mesh (747, 4, 4) float32 (-0.99730104, 3.8002799)
T_cam_mesh_foundationpose (747, 4, 4) float32 (-0.98797655, 4.2716255)
T_cam_mesh_coconet (747, 4, 4) float32 (-0.99661934, 3.6997101)
human_vertices_world (747, 4610, 3) float32 (-0.7559207, 4.0606475)
human_keypoints_world (747, 70, 3) float32 (-0.74876344, 4.054879)
contact_prob_right_left (747, 2) float32 (5.369665e-18, 1.0)
masks ['object', 'person']
object (747, 1152, 1536) bool
person (747, 1152, 1536) bool

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "ffprobe -v error -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /work/inputs/video.mp4; find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100"
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
avg_frame_rate=30/1
duration=24.900000
nb_frames=747
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1939
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1772
/pipeline/.stages/06_coconet.json 2828
/pipeline/.stages/07_refinement.json 2131
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/depth_report.json 1810
/pipeline/depth/moge2/aligned.h5 955977668
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 79589
/pipeline/depth/moge2/raw.npy 2643591296
/pipeline/export/episode_000001/edex 563
/pipeline/export/episode_000001/wild_export.json 2787
/pipeline/foundationpose/episode_000001.pkl 213 …[+642 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m),len(m))
for k,v in m.items():
 print(k,type(v),v.vertices.shape,v.faces.shape,v.visual.kind, v.bounds.tolist(),v.extents.tolist())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
TypeError: object of type 'Trimesh' has no len()

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(type(s),s.geometry.keys())
for k,v in s.geometry.items():
 print(k,type(v),v.vertices.shape,v.faces.shape,v.visual.kind,v.bounds.tolist(),v.extents.tolist(), 'uv',getattr(v.visual,'uv',None).shape if getattr(v.visual,'uv',None) is not None else None)
g=s.to_geometry()
print('to_geometry',type(g))
if hasattr(g,'vertices'):
 print(g.vertices.shape,g.faces.shape,g.visual.kind,g.bounds.tolist(),g.extents.tolist())
else:
 print(len(g))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.scene.scene.Scene'> odict_keys(['vacuum'])
vacuum <class 'trimesh.base.Trimesh'> (283294, 3) (94444, 3) texture [[-0.12395333498716354, -0.07117505371570587, -0.3832121193408966], [0.12843188643455505, 0.07261943072080612, 0.6658868193626404]] [0.2523852214217186, 0.143794484436512, 1.049098938703537] uv (283294, 2)
to_geometry <class 'trimesh.base.Trimesh'>
(283294, 3) (94444, 3) texture [[-0.12395333498716354, -0.07117505371570587, -0.3832121193408966], [0.12843188643455505, 0.07261943072080612, 0.6658868193626404]] [0.2523852214217186, 0.143794484436512, 1.049098938703537]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat /pipeline/pipeline_report.json; echo '--- pose stage'; cat /pipeline/.stages/05_object_pose.json; echo '--- refinement'; cat /pipeline/.stages/07_refinement.json"
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
    "mtime_ns": 1791205198248800496,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000001/inference/episode_000001_step200000_before_after.mp4",
    "size": 54751287
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+9041 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "mkdir -p /work/scratch/frames && python - <<'PY'
import cv2, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
idxs=np.arange(0,747,25)
ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); assert ok
 im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
 ims.append(im)
rows=[]
for j in range(0,len(ims),4):
 row=ims[j:j+4]
 while len(row)<4: row.append(np.zeros_like(ims[0]))
 rows.append(np.concatenate(row,1))
cv2.imwrite('/work/scratch/contact.jpg',np.concatenate(rows,0))
print('saved',len(ims))
PY"
```
<details><summary>output</summary>

```
saved 30

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; F=len(T)
t=T[:,:3,3]; q=R.from_matrix(T[:,:3,:3]).as_quat()
# angular velocity
ang=np.degrees(R.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude())
dt=np.linalg.norm(np.diff(t,axis=0),axis=1)
print('t xyz quantiles'); print(np.quantile(t,[0,.01,.1,.5,.9,.99,1],axis=0))
print('ang quantiles',np.quantile(ang,[0,.1,.5,.9,.95,.99,1]))
print('dt quantiles',np.quantile(dt,[0,.1,.5,.9,.95,.99,1]))
for i in range(0,F,25): print(i, 't',np.round(t[i],3),'angprev',round(ang[max(0,i-1)],2),'dprev',round(dt[max(0,i-1)],4),'contact',np.round(z['contact_prob_right_left'][i],3))
# largest jumps
for i in np.argsort(ang)[-30:][::-1]: print('JUMP',i,round(ang[i],1),round(dt[i],3),np.round(t[i],2),np.round(t[i+1],2))
PY"
```
<details><summary>output</summary>

```
t xyz quantiles
[[-0.00658408  0.73171026  2.67671537]
 [ 0.01131369  0.75487978  2.68387218]
 [ 0.08618762  0.85938416  2.69947791]
 [ 0.24235138  0.88489729  3.14101911]
 [ 0.4190578   0.92507325  3.63970418]
 [ 0.51643416  0.96423761  3.76767218]
 [ 0.53377104  0.97763419  3.80027986]]
ang quantiles [5.65099272e-16 1.06803104e-14 2.43325791e+00 1.00399197e+01
 1.46871513e+01 2.73480845e+01 6.15288905e+01]
dt quantiles [0.         0.00013336 0.0153977  0.04758891 0.05991602 0.0927202
 0.11949538]
0 t [0.296 0.876 2.677] angprev 0.0 dprev 0.0024 contact [0. 0.]
25 t [0.293 0.89  2.686] angprev 0.95 dprev 0.0065 contact [0. 0.]
50 t [0.299 0.884 2.695] angprev 0.0 dprev 0.0039 contact [0. 0.]
75 t [0.3   0.888 2.707] angprev 0.0 dprev 0.0 contact [0. 0.]
100 t [0.299 0.882 2.689] angprev 0 …[+3411 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np, h5py
with h5py.File('/work/inputs/masks.h5') as f:
 o=f['object'][:]; p=f['person'][:]
ao=o.sum((1,2)); ap=p.sum((1,2)); inter=(o&p).sum((1,2))
print('obj area quantiles',np.quantile(ao,[0,.01,.1,.5,.9,.99,1]))
for i in range(0,747,20): print(i,ao[i],ap[i],inter[i],round(inter[i]/max(1,ao[i]),2))
PY"
```
<details><summary>output</summary>

```
obj area quantiles [2294.   2428.36 3207.4  5122.   8024.8  8616.42 9082.  ]
0 7958 37853 0 0.0
20 8045 31765 230 0.03
40 7650 25055 124 0.02
60 8011 35510 0 0.0
80 7997 55449 0 0.0
100 7975 69663 0 0.0
120 7521 70474 2 0.0
140 7286 56691 0 0.0
160 3855 39102 0 0.0
180 3649 31831 2 0.0
200 3313 25701 1 0.0
220 3588 25962 2 0.0
240 3539 30729 6 0.0
260 6124 33795 0 0.0
280 5161 34218 0 0.0
300 4270 32266 5 0.0
320 4345 25618 0 0.0
340 2294 24534 9 0.0
360 3204 25693 1 0.0
380 4715 27276 0 0.0
400 4861 28090 1 0.0
420 3747 31488 309 0.08
440 3405 32656 47 0.01
460 4521 34577 17 0.0
480 4820 33391 0 0.0
500 5017 26510 1 0.0
520 3263 24636 0 0.0
540 3148 26724 7 0.0
560 7246 28462 1 0.0
580 8170 37528 2 0.0
600 7417 38966 90 0.01
620 5035 39727 402 0.08
640 5373 41502 513 0.1
660 5546 35176 0  …[+88 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2, numpy as np, h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:
 for i in [0,80,100,120,140,160,180,200,220,240,260,280,300,320,350,380,400,420,450,480,500,520,540,560,580,600,620,650,680,720,746]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); assert ok
  om=f['object'][i]; pm=f['person'][i]
  vis=im.copy(); vis[om]=(0.5*vis[om]+0.5*np.array([0,0,255])).astype(np.uint8); vis[pm & ~om]=(0.6*vis[pm & ~om]+0.4*np.array([255,0,0])).astype(np.uint8)
  cv2.putText(vis,f'{i} obj={om.sum()} per={pm.sum()}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,255,255),3)
  cv2.imwrite(f'/work/scratch/frames/mask_{i:03d}.jpg',vis)
print('done')
PY"
```
<details><summary>output</summary>

```
done

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,glob,numpy as np,os
files=sorted(glob.glob('/work/scratch/frames/mask_*.jpg'))
ims=[]
for fn in files:
 im=cv2.imread(fn); im=cv2.resize(im,(384,288)); ims.append(im)
rows=[]
for i in range(0,len(ims),4):
 r=ims[i:i+4]
 while len(r)<4:r.append(np.zeros_like(ims[0]))
 rows.append(np.concatenate(r,1))
cv2.imwrite('/work/scratch/masks_contact.jpg',np.concatenate(rows,0))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "find /code/reconstruction -maxdepth 3 -type f | head -100; find /workspace -maxdepth 3 -type f -iname '*render*' | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/scripts/install_ego_reconstruction_packages.sh
/code/reconstruction/scripts/build_ego_e2e_containers.sh
/code/reconstruction/scripts/build_containers.sh
/code/reconstruction/scripts/install_packages.sh
/code/reconstruction/scripts/export_result_threejs_scene.py
/code/reconstruction/scripts/setup_css_env.sh
/code/reconstruction/scripts/download_ego_reconstruction_weights.sh
/code/reconstruction/scripts/build_ego_reconstruction_packages.sh
/code/reconstruction/scripts/sync_css.py
/code/reconstruction/scripts/setup_hitl_aws_env.sh
/code/reconstruction/scripts/install_ego_e2e_packages.sh
/code/reconstruction/scripts/scale_result_bundle.py
/code/reconstruction/scripts/setup_kratos_drs_env.sh
/code/reconstruction/scripts/setup_mv_hoi_status_publish_env.sh
/code/reconstructio …[+5109 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np, cv2, h5py
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; import json
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
# mesh bbox corners
mn=np.array([-.123953,-.071175,-.383212,1]); mx=np.array([.128432,.072619,.665887,1])
cs=np.array([[x,y,zz,1] for x in [mn[0],mx[0]] for y in [mn[1],mx[1]] for zz in [mn[2],mx[2]]]).T
with h5py.File('/work/inputs/masks.h5') as f:
 o=f['object'][:]
 for i in [0,80,100,120,130,150,180,220,260,280,300,320,350,400,450,500,550,600,620,640,680,740]:
  pc=(Tcw@T[i]@cs)[:3]; uv=(K@pc); uv=uv[:2]/uv[2]
  ys,xs=np.where(o[i]); print(i,'pred uv',np.round(uv.min(1),1),np.round(uv.max(1),1),'mask',xs.min(),ys.min(),xs.max(),ys.max(),'ctr',np.round(uv.mean(1),1),np.round([xs.mean(),ys.mean()],1),'z',np.round(pc[2].mean(),3))
PY"
```
<details><summary>output</summary>

```
0 pred uv [727.1 591.7] [873.3 946.2] mask 745 603 873 942 ctr [801.2 761.1] [809.1 788.6] z 2.576
80 pred uv [730.1 594.6] [878.1 944.7] mask 744 603 873 943 ctr [801.1 761.6] [809.6 790. ] z 2.594
100 pred uv [729.7 593.7] [878.3 945.3] mask 744 604 873 943 ctr [801.  761.4] [809.7 790. ] z 2.584
120 pred uv [732.3 598.6] [881.1 950.3] mask 762 629 873 943 ctr [804.3 766.5] [813.  801.9] z 2.587
130 pred uv [714.9 600.2] [887.8 937.7] mask 755 614 885 928 ctr [800.1 760.5] [808.2 780.5] z 2.612
150 pred uv [646.1 556.8] [826.4 830.9] mask 690 574 822 821 ctr [734.2 689.5] [736.4 677.5] z 3.057
180 pred uv [660.9 578.9] [799.4 844.1] mask 695 597 791 839 ctr [731.4 708.3] [728.3 697.7] z 3.399
220 pred uv [655.8 610.9] [853.  867.5] mask 687 627 850 870 ctr [756.  735.3] [754.4 735.8] z 3 …[+1417 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
mods=['nvdiffrast','pyrender','pytorch3d','open3d','kornia','skimage','scipy','h5py','trimesh']
for m in mods:
 try:
  mo=__import__(m); print(m,'OK',getattr(mo,'__version__',''))
 except Exception as e: print(m,'NO',e)
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
pytorch3d OK 0.7.9
open3d OK 0.20.0
kornia OK 0.8.3
skimage OK 0.26.0
scipy OK 1.16.3
h5py OK 3.16.0
trimesh OK 5.1.1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh'][:,:3,:3]; F=z['T_cam_mesh_foundationpose'][:,:3,:3]; C=z['T_cam_mesh_coconet'][:,:3,:3]
def a(A,B): return np.degrees(R.from_matrix(A.transpose(0,2,1)@B).magnitude())
for name,X in [('FP',F),('Co',C)]:
 d=a(T,X)
 print(name,'quant',np.quantile(d,[0,.1,.5,.9,.99,1]))
 print(np.round(d[::20],1))
# compare FP and C
print('FP Co',np.quantile(a(F,C),[0,.1,.5,.9,.99,1]))
# list frames where either rotation differs baseline >60
for name,X in [('FP',F),('Co',C)]:
 d=a(T,X); print(name,'large',[(i,round(d[i],1)) for i in np.where(d>60)[0]])
PY"
```
<details><summary>output</summary>

```
FP quant [  0.65970939   3.56182916  11.31331664  51.32425534 176.61871888
 179.84864821]
[  7.7  11.4  11.4   4.7   7.3   6.6   5.8  23.5  25.1  28.9  32.9  35.4
  32.9  53.9 164.3 147.3  51.3  35.2  10.7  50.7   5.4   4.1  11.3   4.8
  20.    4.3  12.6  33.8  32.    4.8   5.7   5.7   2.5   3.6   2.1   4.5
   3.4   3.8]
Co quant [3.38774225 3.40527495 3.43487268 3.46350853 3.47389351 3.47537079]
[3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.4 3.5 3.5 3.5 3.4 3.4
 3.4 3.4 3.4 3.4 3.4 3.5 3.5 3.4 3.4 3.4 3.5 3.5 3.5 3.4 3.4 3.4 3.4 3.4
 3.4 3.4]
FP Co [  0.67733379   3.43251907  10.39875795  50.42658514 175.15302728
 179.94106393]
FP large [(244, 64.3), (263, 102.1), (264, 111.5), (265, 113.7), (266, 121.3), (267, 141.7), (268, 149.1), (269, 147.2), (270, 149.5), (271, 160.1), (272, 15 …[+765 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import pickle
for fn in ['/pipeline/foundationpose/episode_000001.pkl','/pipeline/mhr_init/episode_000001.pkl','/pipeline/export/episode_000001/wild_export.json']:
 print('\\n',fn)
 try:
  x=pickle.load(open(fn,'rb'))
  print(type(x))
  if isinstance(x,dict):
   for k,v in x.items(): print(k,type(v),getattr(v,'shape',None), (str(v)[:200] if not hasattr(v,'shape') else ''))
  else:
   print(x)
 except Exception as e: print(e)
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
frames <class 'list'> None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019',
camera_id <class 'int'> None 0
fp_poses_cam <class 'numpy.ndarray'> (747, 1, 4, 4) 
fp_candidate_count <class 'numpy.ndarray'> (747,) 
fp_selected_candidate_index <class 'numpy.ndarray'> (747,) 
fp_selected_iou <class 'numpy.ndarray'> (747,) 
fp_selected_visibility <class 'numpy.ndarray'> (747,) 
fp_retry_count <class 'numpy.ndarray'> (747,) 
fp_reliable <class 'numpy.ndarray'> (747,) 
fp_oracle_rotation_error_deg <class 'numpy.ndarray'> (747,) 
fp_translation_threshold_m <class 'numpy.ndarray'> (747,) 
fp_selected_translation_distance_m <class 'numpy …[+2515 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'cat /pipeline/export/episode_000001/wild_export.json; cat /pipeline/.stages/06_coconet.json; cat /pipeline/.stages/08_render.json'
```
<details><summary>output</summary>

```
{
  "aligned_object_extents_m": [
    0.14402921497821808,
    0.252399206161499,
    1.0507014989852905
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
  "frame_count": 747,
  "height": 1152,
  "intrinsics": [
    [
      831.2747802734375,
      0.0,
      768.0
    ],
    [
      0.0,
      831.2747802734375,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "object_mesh_file": "/home/hanzhou/Projects/v2d/run/out/episode_000001/export/episode_000001/object_mesh/output_aligned.glb",
  "object_mesh_to_training_transfo …[+7341 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import torch
for fn in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth','/pipeline/inference/refinement_checkpoint.pth']:
 print('\\n',fn)
 x=torch.load(fn,map_location='cpu',weights_only=False)
 print(type(x))
 if isinstance(x,dict):
  for k,v in x.items():
   try: print(k,type(v),getattr(v,'shape',None))
   except: pass
 else: print(x.keys() if hasattr(x,'keys') else len(x))
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
<class 'dict'>
schema <class 'str'> None
frames <class 'list'> None
frame_meta <class 'list'> None
kid <class 'int'> None
checkpoint <class 'dict'> None
mhr_inference <class 'dict'> None
config <class 'dict'> None
mesh_diameter <class 'numpy.ndarray'> (747,)
mhr_neutral_height_init <class 'numpy.ndarray'> (747,)
mhr_spatial_scale <class 'numpy.ndarray'> (747,)
K_rois <class 'numpy.ndarray'> (747, 3, 3)
bboxes <class 'numpy.ndarray'> (747, 4)
faces <class 'numpy.ndarray'> (36874, 3)
pr <class 'dict'> None
pr_initial <class 'dict'> None
in <class 'dict'> None
gt <class 'dict'> None
observations <class 'dict'> None
raw <class 'dict'> None
metadata <class 'dict'> None

 /pipeline/inference/refined.pth
<class 'dict'>
schema <class 'str'> None
frames <class 'lis …[+1157 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import torch,numpy as np
for fn in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 x=torch.load(fn,map_location='cpu',weights_only=False)
 print('\\n====',fn)
 for top in ['pr','pr_initial','in','gt','observations','raw','metadata','postopt','config','frame_meta']:
  if top not in x: continue
  v=x[top]; print('--',top,type(v))
  if isinstance(v,dict):
   for k,vv in v.items():
    print(' ',k,type(vv),getattr(vv,'shape',None),getattr(vv,'dtype',''))
  else: print(str(v)[:1000])
 print('frames firstlast',x['frames'][:2],x['frames'][-2:])
 print('frame_meta0',x['frame_meta'][0])
 print('mesh_diam',x['mesh_diameter'][:5])
PY"
```
<details><summary>output</summary>

```

==== /pipeline/inference/coconet.pth
-- pr <class 'dict'>
  pose_abs <class 'numpy.ndarray'> (747, 4, 4) float32
  pose_abs_1st_delta <class 'numpy.ndarray'> (747, 4, 4) float32
  mhr_global_rot6d <class 'numpy.ndarray'> (747, 6) float32
  mhr_trans <class 'numpy.ndarray'> (747, 3) float32
  mhr_body_pose_cont <class 'numpy.ndarray'> (747, 260) float32
  mhr_hand <class 'numpy.ndarray'> (747, 108) float32
  mhr_shape <class 'numpy.ndarray'> (747, 45) float32
  mhr_scale <class 'numpy.ndarray'> (747, 28) float32
  mhr_face <class 'numpy.ndarray'> (747, 72) float32
  contact_logits <class 'numpy.ndarray'> (747, 2) float32
-- pr_initial <class 'dict'>
  pose_abs <class 'numpy.ndarray'> (747, 4, 4) float32
  pose_abs_1st_delta <class 'numpy.ndarray'> (747, 4, 4) float32
  mhr_global_rot6d <cl …[+25139 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "mkdir -p /work/scratch/vid && python - <<'PY'
import cv2,numpy as np
for name,path in [('recon','/pipeline/inference/episode_000001_step200000_reconstruction.mp4'),('beforeafter','/pipeline/inference/episode_000001_step200000_before_after.mp4')]:
 cap=cv2.VideoCapture(path); ims=[]; idxs=np.arange(0,747,50)
 for i in idxs:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read();
  if not ok: print('fail',name,i);break
  im=cv2.resize(im,(384,288)); cv2.putText(im,f'{name} {i}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2); ims.append(im)
 rows=[]
 for j in range(0,len(ims),4):
  r=ims[j:j+4]
  while len(r)<4:r.append(np.zeros_like(ims[0]))
  rows.append(np.concatenate(r,1))
 cv2.imwrite(f'/work/scratch/vid/{name}.jpg',np.concatenate(rows,0))
 print(name,cap.get(cv2.CAP_PROP_FRAME_COUNT))
PY"
```
<details><summary>output</summary>

```
recon 747.0
beforeafter 747.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m.visual))
print(m.visual.material)
for a in ['baseColorTexture','image','baseColorFactor','metallicFactor','roughnessFactor']:
 print(a,getattr(m.visual.material,a,None))
# export screenshot scene
s=trimesh.Scene([m]); s.show(viewer='glb') if False else None
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'>
<trimesh.visual.material.PBRMaterial object at 0x78557df91710>
baseColorTexture <PIL.Image.Image image mode=RGB size=4171x4189 at 0x78557DF91790>
image None
baseColorFactor None
metallicFactor 0.0
roughnessFactor None

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh, numpy as np
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
im=m.visual.material.image
print(im.size,im.mode)
im.save('/work/scratch/texture.png')
# uv samples and dominant colors
uv=m.visual.uv; tex=np.array(im)
xy=np.clip((uv*[tex.shape[1]-1,tex.shape[0]-1]).astype(int),0,[tex.shape[1]-1,tex.shape[0]-1])
cols=tex[xy[:,1],xy[:,0],:3]
print('mean',cols.mean(0),'med',np.median(cols,0),'qt',np.quantile(cols,[.1,.5,.9],axis=0))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,120,140,160,180,260,280,300,320,600,620,640]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read()
 cv2.imwrite(f'/work/scratch/frames/orig_{i:03d}.png',im)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
im=m.visual.material.baseColorTexture
im.resize((1200,1200)).save('/work/scratch/texture_small.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/render_test.py <<'PY'
import os, json, numpy as np, torch, trimesh, cv2, h5py
import nvdiffrast.torch as dr
assert torch.cuda.is_available()
dev='cuda'
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev)
F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=dev)
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255
tex=torch.tensor(tex,device=dev).permute(2,0,1)[None]
H,W=1152,1536
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc).astype(np.float32)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
ctx=dr.RasterizeCudaContext()
def render(Tm):
    vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
    vw=(torch.tensor(Tm,dtype=torch.float32,device=dev)@vh.T).T[:,:3]
    vc=(torch.tensor(Tcw,device=dev)@torch.cat([vw,torch.ones_like(vw[:,:1])],1).T).T[:,:3]
    u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]; v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]
    # test direct y down ndc
    clip=torch.stack([2*u/W-1,2*v/H-1,(vc[:,2]-2.5)/2.0,torch.ones_like(u)],1)[None]
    rast,_=dr.rasterize(ctx,clip,F[None],(H,W))
    mask=rast[0,:,:,3]>0
    uv,_=dr.interpolate(UV[None],rast,F[None])
    uvn=uv.clone(); uvn[...,0]=uvn[...,0]*2-1; uvn[...,1]=uvn[...,1]*2-1
    col=torch.nn.functiona …[+952 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 31, in <module>
    rast,mask,col=render(T[i])
                  ^^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 22, in render
    rast,_=dr.rasterize(ctx,clip,F[None],(H,W))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/p …[+348 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.getsource(dr.rasterize))
PY"
```
<details><summary>output</summary>

```
def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    '''Rasterize triangles.

    All input tensors must be contiguous and reside in GPU memory except for
    the `ranges` tensor that, if specified, has to reside in CPU memory. The
    output tensors will be contiguous and reside in GPU memory.

    Args:
        glctx: Rasterizer context of type `RasterizeCudaContext`.
        pos: Vertex position tensor with dtype `torch.float32`. To enable range
             mode, this tensor should have a 2D shape [num_vertices, 4]. To enable
             instanced mode, use a 3D shape [minibatch_size, num_vertices, 4].
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        resolution: Output resolution as integer tuple (height, width).
        …[+1402 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/dr.rasterize(ctx,clip,F\\[None\\],(H,W))/dr.rasterize(ctx,clip,F,(H,W))/\" /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 31, in <module>
    rast,mask,col=render(T[i])
                  ^^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 24, in render
    uv,_=dr.interpolate(UV[None],rast,F[None])
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in …[+181 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.getsource(dr.interpolate)[:1800])
PY"
```
<details><summary>output</summary>

```
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
        rast_db: (Optional) Tensor containing image-space derivatives of barycentrics, 
                 i.e., the second output tenso …[+1001 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/dr.interpolate(UV\\[None\\],rast,F\\[None\\])/dr.interpolate(UV[None],rast,F)/\" /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 bbox 756 599 872 935 iou 0.46719231466376654 targ 7958 pred 8231
150 bbox 672 565 824 821 iou 0.3603684277855955 targ 4913 pred 5721
180 bbox 691 586 797 837 iou 0.3560281517747858 targ 3649 pred 5214
300 bbox 805 609 876 831 iou 0.24397244546498278 targ 4270 pred 4398
400 bbox 687 645 881 862 iou 0.36267274922849857 targ 4861 pred 5295
600 bbox 772 605 882 897 iou 0.505062902730899 targ 7417 pred 7298
680 bbox 749 596 855 910 iou 0.39123743074081674 targ 5474 pred 8085

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,im=cap.read()
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][0];p=f['person'][0]
# crop around x 700:930 y550:1000
x0,y0,x1,y1=650,500,950,1000
crop=im[y0:y1,x0:x1].copy(); mask=o[y0:y1,x0:x1]
cont,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,cont,-1,(0,255,255),3)
cv2.imwrite('/work/scratch/crop0.jpg',cv2.resize(crop,None,fx=2,fy=2))
# mask itself
vis=np.zeros_like(im);vis[o]=(255,255,255);vis[p]=(0,0,255)
cv2.imwrite('/work/scratch/mask0.jpg',cv2.resize(vis[y0:y1,x0:x1],None,fx=2,fy=2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2
im=cv2.imread('/work/scratch/frames/render_000.jpg'); cv2.imwrite('/work/scratch/render0crop.jpg',cv2.resize(im[500:1000,650:950],None,fx=2,fy=2))
im=cv2.imread('/work/scratch/frames/render_180.jpg'); cv2.imwrite('/work/scratch/render180crop.jpg',cv2.resize(im[500:900,630:850],None,fx=3,fy=3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh']
for a,b,name in [(0,130,'start'),(640,747,'end'),(0,747,'all')]:
 X=T[a:b]
 rm=R.from_matrix(X[:,:3,:3]).mean().as_matrix()
 tm=np.median(X[:,:3,3],0)
 da=np.degrees(R.from_matrix(X[:,:3,:3].transpose(0,2,1)@rm).magnitude())
 dt=np.linalg.norm(X[:,:3,3]-tm,axis=1)
 print(name,'rot med/max',np.median(da),da.max(),'trans med/max',np.median(dt),dt.max())
 print('rm',rm,'tm',tm)
# detect actual T constant ranges based derivative
from scipy.spatial.transform import Rotation as RR
ang=np.degrees(RR.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()); dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
print('static candidates')
for i in range(747):
 pass
for th in [0.05,.1,.2,.5,1]:
 good=(ang<th)&(dt<.001)
 # contiguous ranges
 runs=[];s=None
 for i,g in enumerate(good):
  if g and s is None:s=i
  if s is not None and (not g or i==len(good)-1):
   e=i if not g else i+1
   if e-s>=5:runs.append((s,e))
   s=None
 print(th,runs[:20])
PY"
```
<details><summary>output</summary>

```
start rot med/max 2.6948797490998824 23.133662691578767 trans med/max 0.011756672 0.08172267
rm [[ 0.04719886 -0.95762978 -0.28410822]
 [ 0.01993605  0.28527166 -0.95823934]
 [ 0.99868655  0.03956381  0.03255586]] tm [0.299667  0.8870867 2.6965978]
end rot med/max 0.3791499412619202 3.6364355210670247 trans med/max 0.017147887 0.041393664
rm [[-0.56558977 -0.79807034 -0.20782671]
 [-0.01696549  0.26321334 -0.96458847]
 [ 0.82451221 -0.54203548 -0.16241043]] tm [0.23784043 0.86833274 2.75986   ]
all rot med/max 27.614680019951376 179.67130046917876 trans med/max 0.37385792 0.7052666
rm [[-0.13467322 -0.90595016 -0.40139436]
 [ 0.02124457  0.40235168 -0.91523866]
 [ 0.9906623  -0.13178559 -0.03493947]] tm [0.24235138 0.8848973  3.141019  ]
static candidates
0.05 [(11, 17), (62, 71), (78, 86) …[+545 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import torch,numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); B=z['T_world_mesh']
for fn in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 x=torch.load(fn,map_location='cpu',weights_only=False)
 for grp in ['pose_abs']:
  for k in ['pr_initial','pr','in']:
   P=x[k][grp]
   d=R.from_matrix(B[:,:3,:3].transpose(0,2,1)@P[:,:3,:3]).magnitude()*180/np.pi
   print(fn.split('/')[-2],k,'rot',np.quantile(d,[0,.1,.5,.9,1]))
   # direct t difference
   print('t diff',np.quantile(np.linalg.norm(B[:,:3,3]-P[:,:3,3],axis=1),[0,.1,.5,.9,1]))
 if 'postopt' in x:
  print('postopt keys',x['postopt'].keys())
  for k,v in x['postopt'].items(): print(k,type(v),getattr(v,'shape',None))
PY"
```
<details><summary>output</summary>

```
inference pr_initial rot [116.45114417 151.53593297 176.70372331 179.58358666 179.99913591]
t diff [0.24768932 0.3505197  0.4001652  0.54354994 0.65516502]
inference pr rot [116.45114417 151.53593297 176.70372331 179.58358666 179.99913591]
t diff [0.24768932 0.3505197  0.4001652  0.54354994 0.65516502]
inference in rot [ 92.39305791 120.40199333 164.06775953 176.99473873 179.88338959]
t diff [0.26791954 0.31755087 0.43021515 1.76364856 2.56939912]
inference pr_initial rot [116.45114417 151.53593297 176.70372331 179.58358666 179.99913591]
t diff [0.24768932 0.3505197  0.4001652  0.54354994 0.65516502]
inference pr rot [116.45114417 151.53593297 176.70372331 179.58358666 179.99913591]
t diff [0.24703871 0.349434   0.39909282 0.54215612 0.66659391]
inference in rot [ 92.39305791 120.40199333  …[+1347 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:];p=f['person'][:]
F=len(o)
iou=[];cnt=[];cent=[]
for i in range(F):
 y,x=np.where(o[i]); cent.append([x.mean(),y.mean()]);cnt.append(len(x))
 if i:
  a=o[i-1];b=o[i]; iou.append((a&b).sum()/(a|b).sum())
cent=np.array(cent);iou=np.r_[1,iou]
for i in range(0,F,10):
 print(i,'iouprev',round(iou[i],3),'ctr',np.round(cent[i],1),'d',np.round(np.linalg.norm(cent[i]-cent[i-1]) if i else 0,1),'area',cnt[i],'contact',np.round(np.load('/work/inputs/reconstruction.npz')['contact_prob_right_left'][i],2))
PY"
```
<details><summary>output</summary>

```
0 iouprev 1.0 ctr [809.1 788.6] d 0 area 7958 contact [0. 0.]
10 iouprev 0.991 ctr [809.2 788.9] d 0.4 area 8072 contact [0. 0.]
20 iouprev 0.982 ctr [809.5 790. ] d 0.2 area 8045 contact [0. 0.]
30 iouprev 0.957 ctr [809.6 792.1] d 0.8 area 7824 contact [0. 0.]
40 iouprev 0.957 ctr [810.2 793.5] d 0.6 area 7650 contact [0. 0.]
50 iouprev 0.996 ctr [809.6 790. ] d 0.1 area 8010 contact [0. 0.]
60 iouprev 0.997 ctr [809.5 789.9] d 0.1 area 8011 contact [0. 0.]
70 iouprev 0.996 ctr [809.5 789.8] d 0.1 area 8014 contact [0. 0.]
80 iouprev 0.996 ctr [809.6 790. ] d 0.2 area 7997 contact [0. 0.]
90 iouprev 0.995 ctr [809.7 790.2] d 0.2 area 8007 contact [0. 0.]
100 iouprev 0.991 ctr [809.7 790. ] d 0.0 area 7975 contact [0. 0.]
110 iouprev 0.986 ctr [809.4 789.6] d 0.2 area 8022 contact [1. 0.] …[+4364 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/opt_mask.py <<'PY'
import os,json,time,cv2,h5py,numpy as np,torch,trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
dev='cuda'; H,W=1152,1536
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev); F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh']
ctx=dr.RasterizeCudaContext()
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:
 obj=f['object'][:]; per=f['person'][:]
# crop around object
x0,y0,x1,y1=650,500,950,1000; cw=x1-x0;ch=y1-y0

def render(Tm,crop=(x0,y0,x1,y1),res=(None,None)):
 x0,y0,x1,y1=crop; cw=x1-x0;ch=y1-y0
 if res[0] is not None:
  rw,rh=res
 else: rw,rh=cw,ch
 vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 vw=(Tm@vh.T).T[:,:3]
 vc=(torch.tensor(Tcw,device=dev)@torch.cat([vw,torch.ones_like(vw[:,:1])],1).T).T[:,:3]
 u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]-x0; v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]-y0
 # scale to res
 u=u*rw/cw; v=v*rh/ch
 clip=torch.stack([2*u/rw-1,2*v/rh-1,(vc[:,2]-3)/2,torch.ones_like(u)],1)[None]
 rast,_=dr.rasterize(ctx,c …[+2485 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_mask.py", line 77, in <module>
    t=time.time(); Tout=optimize(0,iters=200);print('time',time.time()-t);print(Tout)
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask.py", line 62, in optimize
    opt.zero_grad(); loss,alpha=lossfn(); loss.backward();opt.step()
                                ^^^^^^^^
  File "/work/scratch/opt_mask.py", line 52, in lossfn
    alpha,rast=render(Tm,res=res)
               ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask.py", line 31, in render
    alpha=dr.antialias(clip,rast,F,(rh,rw))[0,:,:,3]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 517, in antialias
    assert all(isinstance(x, torch.Te …[+132 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.getsource(dr.antialias)[:2500])
PY"
```
<details><summary>output</summary>

```
def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """Perform antialiasing.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Note that silhouette edge determination is based on vertex indices in the triangle
    tensor. For it to work properly, a vertex belonging to multiple triangles must be
    referred to using the same vertex index in each triangle. Otherwise, nvdiffrast will always
    classify the adjacent edges as silhouette edges, which leads to bad performance and
    potentially incorrect gradients. If you are unsure whether your data is good, check
    which pixels are modified by the antialias operation and compare to the example in the
    documentation …[+1180 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py'
s=open(p).read()
s=s.replace(\"alpha=dr.antialias(clip,rast,F,(rh,rw))[0,:,:,3]\", \"cov,_=dr.interpolate(torch.ones((1,V.shape[0],1),device=dev),rast,F)\\n alpha=dr.antialias(cov,rast,clip,F)[0,:,:,0]\")
open(p,'w').write(s)
PY
python /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_mask.py", line 78, in <module>
    t=time.time(); Tout=optimize(0,iters=200);print('time',time.time()-t);print(Tout)
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask.py", line 63, in optimize
    opt.zero_grad(); loss,alpha=lossfn(); loss.backward();opt.step()
                                ^^^^^^^^
  File "/work/scratch/opt_mask.py", line 56, in lossfn
    loss=(alpha*torch.clamp(outside,0,b)).sum()/(alpha.sum()+1e-6) + ((1-alpha)*torch.clamp(inside,0,b)).sum()/((1-alpha).sum()+1e-6)
          ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (300) must match the size of tensor b (500) at non-singleton dimension 2

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
a,r=render(torch.tensor(B[0]),res=(300,500));print(a.shape,r.shape)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "<string>", line 24, in render
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat2 in method wrapper_CUDA_mm)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read();s=s.replace(\"alpha,rast=render(Tm,res=res)\\n  inside\", \"alpha,rast=render(Tm,res=res)\\n  if not hasattr(optimize,'x'): print('shapes',alpha.shape,sdf.shape);optimize.x=1\\n  inside\")
open('/work/scratch/opt_mask_dbg.py','w').write(s)
PY
python /work/scratch/opt_mask_dbg.py"
```
<details><summary>output</summary>

```
shapes torch.Size([500, 300]) torch.Size([1, 300, 500])
Traceback (most recent call last):
  File "/work/scratch/opt_mask_dbg.py", line 79, in <module>
    t=time.time(); Tout=optimize(0,iters=200);print('time',time.time()-t);print(Tout)
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask_dbg.py", line 64, in optimize
    opt.zero_grad(); loss,alpha=lossfn(); loss.backward();opt.step()
                                ^^^^^^^^
  File "/work/scratch/opt_mask_dbg.py", line 57, in lossfn
    loss=(alpha*torch.clamp(outside,0,b)).sum()/(alpha.sum()+1e-6) + ((1-alpha)*torch.clamp(inside,0,b)).sum()/((1-alpha).sum()+1e-6)
          ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (300) must match the size of tensor b (500) at non-singleton dimension 2

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read()
s=s.replace(\"clip=torch.stack([2*u/rw-1,2*v/rh-1,(vc[:,2]-3)/2,torch.ones_like(u)],1)[None]\\n rast,_=dr.rasterize(ctx,clip,F,(rh,rw))\\n cov,_=dr.interpolate(torch.ones((1,V.shape[0],1),device=dev),rast,F)\\n alpha=dr.antialias(cov,rast,clip,F)[0,:,:,0]\", \"clip=torch.stack([2*u/rw-1,2*v/rh-1,(vc[:,2]-3)/2,torch.ones_like(u)],1)\\n rast,_=dr.rasterize(ctx,clip,F,(rh,rw))\\n cov,_=dr.interpolate(torch.ones((V.shape[0],1),device=dev),rast,F)\\n alpha=dr.antialias(cov,rast,clip,F)[:,:,0]\")
open(p,'w').write(s)
PY
python /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_mask.py", line 78, in <module>
    t=time.time(); Tout=optimize(0,iters=200);print('time',time.time()-t);print(Tout)
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask.py", line 63, in optimize
    opt.zero_grad(); loss,alpha=lossfn(); loss.backward();opt.step()
                                ^^^^^^^^
  File "/work/scratch/opt_mask.py", line 53, in lossfn
    alpha,rast=render(Tm,res=res)
               ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask.py", line 30, in render
    rast,_=dr.rasterize(ctx,clip,F,(rh,rw))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, t …[+710 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read()
s=s.replace(\"clip=torch.stack([2*u/rw-1,2*v/rh-1,(vc[:,2]-3)/2,torch.ones_like(u)],1)\\n rast,_=dr.rasterize(ctx,clip,F,(rh,rw))\\n cov,_=dr.interpolate(torch.ones((V.shape[0],1),device=dev),rast,F)\\n alpha=dr.antialias(cov,rast,clip,F)[:,:,0]\", \"clip=torch.stack([2*u/rw-1,2*v/rh-1,(vc[:,2]-3)/2,torch.ones_like(u)],1)[None]\\n rast,_=dr.rasterize(ctx,clip,F,(rh,rw))\\n cov,_=dr.interpolate(torch.ones((1,V.shape[0],1),device=dev),rast,F)\\n alpha=dr.antialias(cov,rast,clip,F)[0,:,:,0]\\n if alpha.shape[0]"'!=rh: alpha=alpha.transpose(0,1)")
open(p,'"'w').write(s)
PY
python /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_mask.py", line 79, in <module>
    t=time.time(); Tout=optimize(0,iters=200);print('time',time.time()-t);print(Tout)
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask.py", line 64, in optimize
    opt.zero_grad(); loss,alpha=lossfn(); loss.backward();opt.step()
                                ^^^^^^^^
  File "/work/scratch/opt_mask.py", line 57, in lossfn
    loss=(alpha*torch.clamp(outside,0,b)).sum()/(alpha.sum()+1e-6) + ((1-alpha)*torch.clamp(inside,0,b)).sum()/((1-alpha).sum()+1e-6)
          ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (300) must match the size of tensor b (500) at non-singleton dimension 2

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "grep -n \"alpha\" /work/scratch/opt_mask.py | head; python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read();s=s.replace(\"alpha,rast=render(Tm,res=res)\\n  inside\", \"alpha,rast=render(Tm,res=res)\\n  print('SH',alpha.shape,sdf.shape)\\n  inside\")
open('/work/scratch/opt_mask_dbg.py','w').write(s)
PY
python /work/scratch/opt_mask_dbg.py"
```
<details><summary>output</summary>

```
SH torch.Size([500, 300]) torch.Size([1, 300, 500])
Traceback (most recent call last):
  File "/work/scratch/opt_mask_dbg.py", line 80, in <module>
    t=time.time(); Tout=optimize(0,iters=200);print('time',time.time()-t);print(Tout)
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask_dbg.py", line 65, in optimize
    opt.zero_grad(); loss,alpha=lossfn(); loss.backward();opt.step()
                                ^^^^^^^^
  File "/work/scratch/opt_mask_dbg.py", line 58, in lossfn
    loss=(alpha*torch.clamp(outside,0,b)).sum()/(alpha.sum()+1e-6) + ((1-alpha)*torch.clamp(inside,0,b)).sum()/((1-alpha).sum()+1e-6)
          ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (300) must match the size of tensor b (500) at non-singleton dimension 2

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '22,36p' /work/scratch/opt_mask.py; sed -n '48,60p' /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
 else: rw,rh=cw,ch
 vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 vw=(Tm@vh.T).T[:,:3]
 vc=(torch.tensor(Tcw,device=dev)@torch.cat([vw,torch.ones_like(vw[:,:1])],1).T).T[:,:3]
 u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]-x0; v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]-y0
 # scale to res
 u=u*rw/cw; v=v*rh/ch
 clip=torch.stack([2*u/rw-1,2*v/rh-1,(vc[:,2]-3)/2,torch.ones_like(u)],1)[None]
 rast,_=dr.rasterize(ctx,clip,F,(rh,rw))
 cov,_=dr.interpolate(torch.ones((1,V.shape[0],1),device=dev),rast,F)
 alpha=dr.antialias(cov,rast,clip,F)[0,:,:,0]
 if alpha.shape[0]!=rh: alpha=alpha.transpose(0,1)
 return alpha,rast

def crop_resize(a,res,interp=cv2.INTER_NEAREST):
 r0=torch.tensor(init[:3,:3],dtype=torch.float32,device=dev); t0=torch.tensor(init[:3,3],dtype=torch.float32,device=dev)
 aa=torch.zeros(3,device=dev,require …[+626 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '17,23p' /work/scratch/opt_mask.py; grep -n 'def optimize' /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```

def render(Tm,crop=(x0,y0,x1,y1),res=(None,None)):
 x0,y0,x1,y1=crop; cw=x1-x0;ch=y1-y0
 if res[0] is not None:
  rw,rh=res
 else: rw,rh=cw,ch
 vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
41:def optimize(i,init=None,iters=200,lr=.01,res=(300,500)):

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read();s=s.replace(\"rh,rw=res\\n if a.dtype==bool: return cv2.resize(a.astype(np.uint8),(rw,rh),interpolation=interp)>0\\n return cv2.resize(a,(rw,rh),interpolation=interp)\",\"rw,rh=res\\n if a.dtype==bool: return cv2.resize(a.astype(np.uint8),(rw,rh),interpolation=interp)>0\\n return cv2.resize(a,(rw,rh),interpolation=interp)\")
open(p,'w').write(s)
PY
python /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
0 0 20.269760131835938
0 20 19.831634521484375
0 40 19.831674575805664
0 60 19.83125114440918
0 80 19.83096694946289
0 100 19.830703735351562
0 120 19.830461502075195
0 140 19.82988929748535
0 160 19.82798957824707
0 180 19.82636260986328
Traceback (most recent call last):
  File "/work/scratch/opt_mask.py", line 79, in <module>
    t=time.time(); Tout=optimize(0,iters=200);print('time',time.time()-t);print(Tout)
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_mask.py", line 74, in optimize
    tp=(pm&tm).sum(); fp=(pm&~(tm|pp)).sum(); union=tm.sum()+fp
        ~~^~~
ValueError: operands could not be broadcast together with shapes (300,500) (500,300) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/res=(ch,cw)/res=(cw,ch)/' /work/scratch/opt_mask.py
python /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
0 0 20.269760131835938
0 20 19.831634521484375
0 40 19.831674575805664
0 60 19.83125114440918
0 80 19.83096694946289
0 100 19.830703735351562
0 120 19.830461502075195
0 140 19.82988929748535
0 160 19.82798957824707
0 180 19.82636260986328
FINAL iou_visible 0.13187070580886945 tp 1689 fp 4850 target 7958 alpha baseline?
time 2.768340826034546
[[ 0.10855938 -0.95601004 -0.27250648  0.34976035]
 [ 0.01741241  0.27591324 -0.9610247   0.8241795 ]
 [ 0.99393743  0.09958328  0.04659941  2.9225209 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read()
old=\"loss=(alpha*torch.clamp(outside,0,b)).sum()/(alpha.sum()+1e-6) + ((1-alpha)*torch.clamp(inside,0,b)).sum()/((1-alpha).sum()+1e-6)\\n  # overlap weighting / balanced bce ignore person\\n  valid=1-pers\\n  loss=loss + .5*(((alpha-img)**2*valid).sum()/(valid.sum()))\"
new=\"dice=2*(alpha*img).sum()/(alpha.sum()+img.sum()+1e-6)\\n  boundary=(alpha*torch.clamp(outside,0,b)).sum()/(alpha.sum()+1e-6)\\n  loss=(1-dice)+0.02*boundary\"
s=s.replace(old,new);open(p,'w').write(s)
PY
python /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
0 0 0.9069181084632874
0 20 0.8953112363815308
0 40 0.8853849768638611
0 60 0.8858221769332886
0 80 0.8616991639137268
0 100 0.8573293089866638
0 120 0.854995608329773
0 140 0.8527641296386719
0 160 0.8407203555107117
0 180 0.8314817547798157
FINAL iou_visible 0.2200392927308448 tp 2464 fp 3240 target 7958 alpha baseline?
time 2.2513182163238525
[[ 0.10855938 -0.95601004 -0.27250648  0.37180573]
 [ 0.01741241  0.27591324 -0.9610247   1.2863432 ]
 [ 0.99393743  0.09958328  0.04659941  3.1878304 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read();s=s.replace(\"with torch.no_grad():\\n  loss,alpha=lossfn()\", \"with torch.no_grad():\\n  a0,_=render(torch.tensor(init,dtype=torch.float32,device=dev),res=(cw,ch)); p0=a0.cpu().numpy()>.5; tm0=obj[i][y0:y1,x0:x1];pp0=per[i][y0:y1,x0:x1];print('BASE visible', (p0&tm0).sum()/(tm0.sum()+(p0&~(tm0|pp0)).sum()), 'shapes',p0.shape,tm0.shape)\\n  loss,alpha=lossfn()\")
open('/work/scratch/opt_mask_dbg.py','w').write(s)
PY
python /work/scratch/opt_mask_dbg.py"
```
<details><summary>output</summary>

```
0 0 0.9069181084632874
0 20 0.8948609232902527
0 40 0.8859224319458008
0 60 0.8850244283676147
0 80 0.8763200044631958
0 100 0.8595430850982666
0 120 0.854267418384552
0 140 0.8439281582832336
0 160 0.8402459025382996
0 180 0.827448308467865
BASE visible 0.46188913202042303 shapes (500, 300) (500, 300)
FINAL iou_visible 0.21815056098443722 tp 2411 fp 3094 target 7958 alpha baseline?
time 2.2616913318634033
[[ 0.10855938 -0.95601004 -0.27250648  0.37557596]
 [ 0.01741241  0.27591324 -0.9610247   1.3037164 ]
 [ 0.99393743  0.09958328  0.04659941  3.2336743 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat >> /work/scratch/opt_mask_dbg.py <<'PY'
PY
python - <<'PY'
# import script without main
ns={};exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0],ns)
import torch,numpy as np
torch.manual_seed(0)
Tm=torch.tensor(ns['B'][0],device=ns['dev'])
a,r=ns['render'](Tm,res=(300,500))
m=r[0,:,:,3]>0
x=ns['obj'][0,500:1000,650:950]
for name,z in [('raster',m.cpu().numpy()),('alpha',a.cpu().numpy()>.5),('alphaT',a.T.cpu().numpy()>.5)]:
 print(name,z.shape,(z&x).sum()/(z|x).sum(),(z&x).sum(),z.sum(),x.sum())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
ValueError: operands could not be broadcast together with shapes (300,500) (500,300) 
raster (500, 300) 0.46719231466376654 5155 8231 7958
alpha (500, 300) 0.46188913202042303 5066 8076 7958

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
ns={};exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0],ns)
import torch,numpy as np
Tm=torch.tensor(ns['B'][0],device=ns['dev']);a,r=ns['render'](Tm,res=(300,500));x=torch.tensor(ns['obj'][0,500:1000,650:950].astype(np.float32),device=ns['dev'])
print(a.min().item(),a.max().item(),a.sum().item(),x.sum().item(),(a*x).sum().item(),2*(a*x).sum()/(a.sum()+x.sum()))
# align orientation maybe alpha.T?
aT=a.T
print('T',2*(aT*x).sum()/(aT.sum()+x.sum()))
PY"
```
<details><summary>output</summary>

```
-0.23070889711380005 1.0 7745.65185546875 7958.0 4931.7578125 Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
RuntimeError: The size of tensor a (500) must match the size of tensor b (300) at non-singleton dimension 1
tensor(0.6281, device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/opt_mask.py';s=open(p).read().replace(\"loss=(1-dice)+0.02*boundary\",\"loss=(1-dice)+0.002*boundary\").replace(\"lr=.01\",\"lr=.001\")
open(p,'w').write(s)
PY
python /work/scratch/opt_mask.py"
```
<details><summary>output</summary>

```
0 0 0.9010419845581055
0 20 0.889426589012146
0 40 0.8816035985946655
0 60 0.8805364966392517
0 80 0.8804139494895935
0 100 0.8819167017936707
0 120 0.8807548880577087
0 140 0.8802409768104553
0 160 0.8821351528167725
0 180 0.8813208937644958
FINAL iou_visible 0.44848913235553983 tp 5076 fp 3360 target 7958 alpha baseline?
time 2.2618329524993896
[[ 0.10855938 -0.95601004 -0.27250648  0.2926776 ]
 [ 0.01741241  0.27591324 -0.9610247   0.9255865 ]
 [ 0.99393743  0.09958328  0.04659941  2.6353574 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_static.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
import time
for phase,idxs,ref in [('start',range(0,128),0),('end',range(640,747),700)]:
 scores=[];t=time.time()
 with torch.no_grad():
  for i in idxs:
   a,r=render(torch.tensor(B[i],dtype=torch.float32,device=dev),res=(cw,ch))
   p=(r[0,:,:,3]>0).cpu().numpy(); tm=obj[ref,y0:y1,x0:x1]; pp=per[i,y0:y1,x0:x1]
   vis=(p&tm).sum()/(tm.sum()+(p&~(tm|pp)).sum())
   full=(p&obj[i,y0:y1,x0:x1]).sum()/(p|obj[i,y0:y1,x0:x1]).sum()
   scores.append((vis,full,i))
 print(phase,'time',time.time()-t,'mean',np.mean(scores,0)[:2])
 print('top vis',sorted(scores,reverse=True)[:15])
 print('top full',sorted(scores,key=lambda x:x[1],reverse=True)[:15])
PY
python /work/scratch/eval_static.py"
```
<details><summary>output</summary>

```
start time 0.3695945739746094 mean [0.47601787 0.47532502]
top vis [(0.5543828789211375, 0.5318963006117099, 26), (0.5526623756582797, 0.5105248118868464, 25), (0.5391169635941131, 0.49174125680538894, 23), (0.5386206896551724, 0.48952953328951077, 29), (0.5383119408790354, 0.5158693006272821, 22), (0.5380440050402249, 0.49453777655375014, 28), (0.5376831148804935, 0.5000939143501127, 24), (0.5260502571068206, 0.4821002386634845, 27), (0.5117795950155763, 0.4763288647229774, 31), (0.508526537650891, 0.48804526373425805, 21), (0.5074034646278912, 0.45594594594594595, 30), (0.5070291406759148, 0.5073495423869835, 43), (0.5044719314938154, 0.4973836408702837, 19), (0.5033633349123638, 0.486298900306472, 20), (0.49981035463682916, 0.48333968012185835, 41)]
top full [(0.49106406378883694, 0.543 …[+2182 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
ns={};exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0],ns)
import torch,cv2,numpy as np
for i in [0,22,26,100,127,640,656,659,700]:
 a,r=ns['render'](torch.tensor(ns['B'][i],device=ns['dev']),res=(ns['cw'],ns['ch']))
 p=(r[0,:,:,3]>0).cpu().numpy().astype(np.uint8)
 cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read(); crop=im[ns['y0']:ns['y1'],ns['x0']:ns['x1']]
 # contour green, target yellow
 pc,_=cv2.findContours(p,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,pc,-1,(0,255,0),1)
 o=ns['obj'][i,ns['y0']:ns['y1'],ns['x0']:ns['x1']].astype(np.uint8);oc,_=cv2.findContours(o,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,oc,-1,(0,255,255),1)
 cv2.putText(crop,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
 cv2.imwrite(f'/work/scratch/frames/cont_{i:03d}.png',crop)
PY
python - <<'PY'
import cv2,numpy as np
ims=[]
for i in [0,22,26,100,127,640,656,659,700]:
 im=cv2.imread(f'/work/scratch/frames/cont_{i:03d}.png'); im=cv2.resize(im,(300,500));ims.append(im)
cv2.imwrite('/work/scratch/cont_contact.jpg',np.concatenate([np.concatenate(ims[:5],1),np.concatenate(ims[5:]+[np.zeros_like(ims[0])]*(5-len(ims[5:])),1)],0))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_static2.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
from scipy.ndimage import distance_transform_edt
import cv2,time

def contour_dist(p,t):
 # distances from p pixels to nearest t pixel and vice versa
 dt=distance_transform_edt(~t); dp=distance_transform_edt(~p)
 return dt[p].mean()+dp[t].mean()
for phase,idxs,ref in [('start',range(0,128),0),('end',range(640,747),700)]:
 targ=obj[ref,y0:y1,x0:x1]
 scores=[]
 with torch.no_grad():
  for i in idxs:
   a,r=render(torch.tensor(B[i],dtype=torch.float32,device=dev),res=(cw,ch))
   p=(r[0,:,:,3]>0).cpu().numpy();tgt=obj[i,y0:y1,x0:x1];per_i=per[i,y0:y1,x0:x1]
   # visible masks: target and set person don't care, use full mask dice too
   dice=2*(p.astype(float)*tgt).sum()/(p.sum()+tgt.sum())
   visdice=2*(p.astype(float)*tgt).sum()/(p.sum()+tgt.sum())
   cd=contour_dist(p,tgt)
   scores.append((dice,cd,visdice,i))
 print('\\n',phase)
 print('dice top',sorted(scores,reverse=True)[:15])
 print('chamfer top',sorted(scores,key=lambda x:x[1])[:15])
PY
python /work/scratch/eval_static2.py"
```
<details><summary>output</summary>

```

 start
dice top [(0.7040135806960107, 3.437582076538079, 0.7040135806960107, 127), (0.6944285985928884, 3.6185034858771195, 0.6944285985928884, 26), (0.6806250386016923, 3.6913615453533666, 0.6806250386016923, 22), (0.6777025332330072, 3.5948979345529293, 0.6777025332330072, 125), (0.6759568699161359, 3.8310614924467457, 0.6759568699161359, 25), (0.6731677399570684, 3.873616028468957, 0.6731677399570684, 43), (0.6667501408627058, 3.808776875641672, 0.6667501408627058, 24), (0.6643369505241862, 3.6835926613496306, 0.6643369505241862, 19), (0.6618552852609967, 3.818026030213513, 0.6618552852609967, 126), (0.6617936117936118, 4.0714841543045175, 0.6617936117936118, 28), (0.6592849189657305, 3.939289579852884, 0.6592849189657305, 23), (0.6572941621485311, 4.05809136604325, 0.6572941621485311, …[+3194 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/all_cont.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in range(120,650,10):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();crop=im[y0:y1,x0:x1].copy()
 with torch.no_grad():
  a,r=render(torch.tensor(B[i],device=dev),res=(cw,ch));p=(r[0,:,:,3]>0).cpu().numpy().astype(np.uint8)
 pc,_=cv2.findContours(p,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,pc,-1,(0,255,0),2)
 o=obj[i,y0:y1,x0:x1].astype(np.uint8);oc,_=cv2.findContours(o,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,oc,-1,(0,255,255),2)
 cv2.putText(crop,str(i),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),3)
 cv2.imwrite(f'/work/scratch/frames/all_{i:03d}.jpg',cv2.resize(crop,(180,300)))
ims=[]
for i in range(120,650,10):ims.append(cv2.imread(f'/work/scratch/frames/all_{i:03d}.jpg'))
rows=[]
for j in range(0,len(ims),7):
 r=ims[j:j+7]
 while len(r)<7:r.append(np.zeros_like(ims[0]))
 rows.append(np.concatenate(r,1))
cv2.imwrite('/work/scratch/all_cont.jpg',np.concatenate(rows,0))
PY
python /work/scratch/all_cont.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/photo_eval.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
import torch,cv2,numpy as np
from scipy.spatial.transform import Rotation as Rot
# texture
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255
tex=torch.tensor(tex,device=dev).permute(2,0,1)[None]
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=dev)
def render_col(Tm):
 rw,rh=cw,ch
 vh=torch.cat([V,torch.ones_like(V[:,:1])],1);vw=(Tm@vh.T).T[:,:3]
 vc=(torch.tensor(Tcw,device=dev)@torch.cat([vw,torch.ones_like(vw[:,:1])],1).T).T[:,:3]
 u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]-x0;v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]-y0
 clip=torch.stack([2*u/rw-1,2*v/rh-1,(vc[:,2]-3)/2,torch.ones_like(u)],1)[None]
 rast,_=dr.rasterize(ctx,clip,F,(rh,rw))
 mask=rast[0,:,:,3]>0
 uv,_=dr.interpolate(UV[None],rast,F)
 uvn=torch.stack([uv[...,0]*2-1,uv[...,1]*2-1],-1)
 col=torch.nn.functional.grid_sample(tex,uvn,align_corners=False,mode='bilinear',padding_mode='border')[0].permute(1,2,0)
 return mask,col
# centroid center for flips in mesh coordinates
cen=torch.tensor(mesh.bounds.mean(0),dtype=torch.float32,device=dev)
def flip(Tm,axis):
 Rd=torch.tensor(Rot.from_rotvec(np.eye(3)[axis]*np.pi).as_matrix(),dtype=torch.float32,device=dev)
 M=torch.eye(4,device=dev);M[:3,:3]=Rd;M[:3,3]=cen-Rd@cen
 return Tm@M
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def target(i):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();return cv2.cvtColor(im,cv2.COLOR …[+1139 chars]
```
<details><summary>output</summary>

```

FRAME 0
base (0.19071580732665389, 0.11849083784736576, 5155)
fx (-0.07233060547154567, -0.11651471962668579, 4408)
fy (0.15651218005393291, -0.05749512551222726, 3392)
fz (0.07434904388890833, 0.17301495280271112, 4511)

FRAME 10
base (0.19653955233686896, 0.12720997868700037, 5232)
fx (-0.08978598005053053, -0.11221979442828159, 4484)
fy (0.16485052329402408, -0.07972413237114698, 3356)
fz (0.07204174822578888, 0.14948583684950714, 4487)

FRAME 20
base (0.21174400996681697, 0.05102574900851157, 5259)
fx (-0.06307829708706177, -0.08447178490391934, 4597)
fy (0.16127948168082648, -0.07368324121201132, 3107)
fz (0.0943001384997751, 0.07610925864342767, 4096)

FRAME 26
base (0.21974903174764068, 0.06777840457974978, 5434)
fx (-0.05212316798325668, -0.02758954157688248, 4029)
fy (0.200964553 …[+3457 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/zoom_cont.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
idxs=list(range(240,331,3))
ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();crop=im[y0:y1,x0:x1].copy()
 with torch.no_grad():
  a,r=render(torch.tensor(B[i],device=dev),res=(cw,ch));p=(r[0,:,:,3]>0).cpu().numpy().astype(np.uint8)
 pc,_=cv2.findContours(p,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,pc,-1,(0,255,0),2)
 o=obj[i,y0:y1,x0:x1].astype(np.uint8);oc,_=cv2.findContours(o,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,oc,-1,(0,255,255),2)
 cv2.putText(crop,str(i),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),3)
 ims.append(cv2.resize(crop,(180,300)))
rows=[]
for j in range(0,len(ims),8):
 r=ims[j:j+8]
 while len(r)<8:r.append(np.zeros_like(ims[0]))
 rows.append(np.concatenate(r,1))
cv2.imwrite('/work/scratch/zoom_cont.jpg',np.concatenate(rows,0))
PY
python /work/scratch/zoom_cont.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];Rs=T[:,:3,:3]
for j in [243,247,254,262,272,273,276,287,297,304,308,312,321,360,433,442,609]:
 for i in range(max(0,j-3),min(746,j+3)+1):
  D=Rs[i].T@Rs[i+1]; rv=R.from_matrix(D).as_rotvec();ang=np.linalg.norm(rv)*180/np.pi
  if ang>5:
   ax=rv/(np.linalg.norm(rv)+1e-9)
   print(i,'->',i+1,'ang',round(ang,1),'local axis',np.round(ax,2),'world long',np.round(Rs[i][:,2],2))
 print()
PY"
```
<details><summary>output</summary>

```
241 -> 242 ang 16.2 local axis [ 0.06  0.17 -0.98] world long [-0.44 -0.9  -0.  ]
242 -> 243 ang 17.8 local axis [ 0.06  0.15 -0.99] world long [-0.4  -0.92  0.03]
243 -> 244 ang 17.4 local axis [-0.29 -0.17  0.94] world long [-0.36 -0.93  0.05]
244 -> 245 ang 20.1 local axis [ 0.16  0.3  -0.94] world long [-0.45 -0.89  0.08]
245 -> 246 ang 14.3 local axis [-0.    0.12  0.99] world long [-0.34 -0.93  0.11]
246 -> 247 ang 14.3 local axis [ 0.01 -0.12 -0.99] world long [-0.32 -0.94  0.13]

244 -> 245 ang 20.1 local axis [ 0.16  0.3  -0.94] world long [-0.45 -0.89  0.08]
245 -> 246 ang 14.3 local axis [-0.    0.12  0.99] world long [-0.34 -0.93  0.11]
246 -> 247 ang 14.3 local axis [ 0.01 -0.12 -0.99] world long [-0.32 -0.94  0.13]
247 -> 248 ang 25.5 local axis [ 0.18  0.27 -0.95] world long …[+5821 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_smooth.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import distance_transform_edt
import numpy as np,torch,time
F=len(B)
def ortho(M):
 U,s,Vt=np.linalg.svd(M); D=np.eye(3);D[2,2]=np.sign(np.linalg.det(U@Vt));return U@D@Vt
def smooth_T(win, poly=2, static=True):
 T=B.copy()
 if static:
  T[:128]=B[127];T[639:]=B[659]
 for a,b in [(128,639)]:
  # savgol rotation matrices then ortho
  for i in range(3):
   for j in range(3):T[a:b,i,j]=savgol_filter(B[a:b,i,j],win,poly,mode='interp')
  for i in range(a,b):T[i,:3,:3]=ortho(T[i,:3,:3])
  for j in range(3):T[a:b,j,3]=savgol_filter(B[a:b,j,3],win,poly,mode='interp')
 return T
# metrics target ref per frame
def metric(Tm):
 with torch.no_grad():a,r=render(torch.tensor(Tm,dtype=torch.float32,device=dev),res=(cw,ch))
 p=(r[0,:,:,3]>0).cpu().numpy();t=obj[i,y0:y1,x0:x1];pp=per[i,y0:y1,x0:x1]
 dice=2*(p&t).sum()/(p.sum()+t.sum())
 visdice=2*(p&t).sum()/(p.sum()+t.sum())
 iou=(p&t).sum()/(p|t).sum()
 # visible precision with person ignore
 vig=(p&t).sum()/(t.sum()+(p&~(t|pp)).sum())
 return dice,iou,vig
for name,T in [('base',B)]+[(f'sg{w}',smooth_T(w)) for w in [3,5,7,9,11,15]]:
 st=time.time();vals=[]; ang=[];dt=[]
 for i in range(F):
  vals.append(metric(T[i]))
  if i: ang.append(np.degrees(Rot.from_matrix(T[i-1,:3,:3].T@T[i,:3,:3]).magnitude())); …[+379 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_smooth.py", line 33, in <module>
    vals.append(metric(T[i]))
                ^^^^^^^^^^^^
  File "/work/scratch/eval_smooth.py", line 22, in metric
    with torch.no_grad():a,r=render(torch.tensor(Tm,dtype=torch.float32,device=dev),res=(cw,ch))
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 30, in render
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 123, in rasterize
    assert isinstance(pos, torch.Tensor) and isinstance(tri, torch.Tensor)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/"'^F=len(B)/NF=len(B)/; s/for i in range(F):/for i in range(NF):/'"' /work/scratch/eval_smooth.py
python /work/scratch/eval_smooth.py"
```
<details><summary>output</summary>

```
base dice [0.5330923  0.37114567 0.37561756] q10 0.36468958460038425 smooth 4.0990069329932295 14.687151301118227 0.020550922 time 2.4951586723327637
phase [(0, 128, 0.644, 0.48), (128, 639, 0.5, 0.347), (639, 747, 0.557, 0.387)]
sg3 dice [0.54033928 0.37860945 0.38285247] q10 0.36468958460038425 smooth 3.8733718563732773 14.687150537283554 0.01939337 time 3.9298877716064453
phase [(0, 128, 0.664, 0.501), (128, 639, 0.5, 0.347), (639, 747, 0.584, 0.413)]
sg5 dice [0.54182687 0.37969719 0.38394579] q10 0.3725175910238504 smooth 2.5755346965534662 9.291349952401264 0.012814279 time 2.21172833442688
phase [(0, 128, 0.664, 0.501), (128, 639, 0.502, 0.349), (639, 747, 0.584, 0.413)]
sg7 dice [0.54224784 0.3800045  0.38423405] q10 0.36877892812184393 smooth 2.1512697289490337 7.145152008023105 0 …[+805 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/opt_trans.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
import torch,time,numpy as np

def opt(i,init=None,iters=40,lr=.002,lam=0.02):
 if init is None:init=B[i]
 tgt=torch.tensor((cv2.resize(obj[i,y0:y1,x0:x1].astype(np.uint8),(cw,ch),interpolation=cv2.INTER_NEAREST)>0).astype(np.float32),device=dev)
 r0=torch.tensor(init[:3,:3],dtype=torch.float32,device=dev);t0=torch.tensor(init[:3,3],dtype=torch.float32,device=dev)
 dt=torch.zeros(3,device=dev,requires_grad=True);optm=torch.optim.Adam([dt],lr=lr)
 best=(1e9,None)
 for it in range(iters):
  Tm=torch.eye(4,device=dev);Tm[:3,:3]=r0;Tm[:3,3]=t0+dt
  a,r=render(Tm,res=(cw,ch))
  dice=2*(a*tgt).sum()/(a.sum()+tgt.sum()+1e-6)
  loss=1-dice+lam*(dt*dt).sum()
  optm.zero_grad();loss.backward();optm.step()
  if loss.item()<best[0]:best=(loss.item(),dt.detach().clone())
 dt=best[1];Tm=torch.eye(4,device=dev);Tm[:3,:3]=r0;Tm[:3,3]=t0+dt
 with torch.no_grad():a,r=render(Tm,res=(cw,ch));p=(r[0,:,:,3]>0).cpu().numpy();t=tgt.cpu().numpy()>0;dice=2*(p&t).sum()/(p.sum()+t.sum());iou=(p&t).sum()/(p|t).sum()
 return Tm.cpu().numpy(),dt.cpu().numpy(),dice,iou
for i in [0,26,100,127,150,180,220,260,300,330,400,500,600,620,640,656,700]:
 t=time.time();Tm,d,dice,iou=opt(i);print(i,'d',np.round(d,3),'dice',round(dice,3),'iou',round(iou,3),'time',round(time.time()-t,2))
PY
python /work/scratch/opt_trans.py"
```
<details><summary>output</summary>

```
0 d [-0.009  0.013 -0.008] dice 0.701 iou 0.54 time 1.65
26 d [-0.006  0.006  0.002] dice 0.737 iou 0.583 time 0.19
100 d [-0.01   0.009 -0.024] dice 0.742 iou 0.59 time 0.24
127 d [-0.008 -0.    -0.019] dice 0.744 iou 0.593 time 0.22
150 d [ 0.012 -0.021 -0.018] dice 0.585 iou 0.413 time 0.26
180 d [-0.013 -0.008  0.008] dice 0.57 iou 0.399 time 0.27
220 d [-0.     0.025 -0.054] dice 0.479 iou 0.315 time 0.23
260 d [-0.021 -0.034  0.032] dice 0.689 iou 0.525 time 0.21
300 d [-0.035  0.041  0.025] dice 0.483 iou 0.318 time 0.25
330 d [-0.004  0.008 -0.009] dice 0.456 iou 0.295 time 0.14
400 d [ 0.012  0.028 -0.046] dice 0.625 iou 0.454 time 0.11
500 d [-0.012  0.028 -0.041] dice 0.624 iou 0.453 time 0.11
600 d [-0.011  0.014 -0.017] dice 0.771 iou 0.627 time 0.11
620 d [ 0.    -0.002 -0.02 …[+213 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh,numpy as np,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices; P=json.load(open('/work/inputs/ground_plane.json'))['plane'];n=np.array(P[:3]);d=P[3]
for i in [0,26,100,127,150,180,220,260,300,400,500,600,620,640,656,700,746]:
 vw=(T[i,:3,:3]@V.T).T+T[i,:3,3];dist=vw@n+d
 print(i,'min/max/quant',np.round([dist.min(),dist.max(),np.quantile(dist,.01),np.quantile(dist,.1)],3),'mesh ymin/max',np.round([vw[:,1].min(),vw[:,1].max()],3))
PY"
```
<details><summary>output</summary>

```
0 min/max/quant [-0.021  1.005 -0.008  0.016] mesh ymin/max [0.233 1.254]
26 min/max/quant [-0.039  0.982 -0.026  0.   ] mesh ymin/max [0.254 1.271]
100 min/max/quant [-0.027  0.998 -0.015  0.009] mesh ymin/max [0.24  1.259]
127 min/max/quant [-0.048  0.974 -0.037 -0.012] mesh ymin/max [0.266 1.28 ]
150 min/max/quant [0.106 1.083 0.12  0.157] mesh ymin/max [0.138 1.101]
180 min/max/quant [-0.048  0.959 -0.04  -0.011] mesh ymin/max [0.238 1.248]
220 min/max/quant [-0.061  0.819 -0.052 -0.   ] mesh ymin/max [0.381 1.263]
260 min/max/quant [-0.059  0.993 -0.054 -0.031] mesh ymin/max [0.215 1.263]
300 min/max/quant [-0.129  0.881 -0.115 -0.086] mesh ymin/max [0.319 1.312]
400 min/max/quant [-0.086  0.758 -0.074 -0.021] mesh ymin/max [0.452 1.285]
500 min/max/quant [-0.039  0.839 -0.03   0.022] …[+483 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];Rs=T[:,:3,:3].astype(float);F=len(Rs)
def frame_from_z(a):
 a=a/np.linalg.norm(a);ref=np.array([0.,0.,1.]);v=np.cross(ref,a);c=ref@a
 if c<-0.999: return np.diag([1,-1,-1])
 if np.linalg.norm(v)<1e-9:return np.eye(3)
 K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
 return np.eye(3)+K+K@K/(1+c)
ths=[]; d=[]
for i in range(F):
 a=Rs[i][:,2];Q=frame_from_z(a);S=Q.T@Rs[i];rv=R.from_matrix(S).as_rotvec(); ths.append(rv[2]);
 # other components
 d.append(np.linalg.norm(rv[:2]))
ths=np.unwrap(np.array(ths));print('theta stats',np.quantile(ths,[0,.1,.5,.9,1]),'other',np.quantile(d,[0,.5,.9,1]))
for i in range(0,F,10):print(i,round(ths[i],2),round(d[i],2))
print('diff deg'); print(np.round(np.degrees(np.diff(ths))[::10],1))
PY"
```
<details><summary>output</summary>

```
theta stats [-6.20477544 -5.52348794 -4.34765476  1.28515291  1.80979842] other [9.08778705e-09 3.17442753e-07 5.35191293e-07 6.62307836e-07]
0 1.19 0.0
10 1.19 0.0
20 1.03 0.0
30 1.07 0.0
40 1.19 0.0
50 1.29 0.0
60 1.29 0.0
70 1.29 0.0
80 1.29 0.0
90 1.29 0.0
100 1.29 0.0
110 1.27 0.0
120 1.26 0.0
130 1.69 0.0
140 1.34 0.0
150 1.4 0.0
160 1.49 0.0
170 1.16 0.0
180 0.81 0.0
190 0.6 0.0
200 0.35 0.0
210 0.31 0.0
220 0.61 0.0
230 0.5 0.0
240 0.76 0.0
250 0.13 0.0
260 -0.71 0.0
270 -2.23 0.0
280 -3.08 0.0
290 -3.75 0.0
300 -3.65 0.0
310 -5.75 0.0
320 -5.3 0.0
330 -5.09 0.0
340 -4.62 0.0
350 -5.0 0.0
360 -4.76 0.0
370 -4.89 0.0
380 -5.56 0.0
390 -5.56 0.0
400 -5.42 0.0
410 -4.77 0.0
420 -4.68 0.0
430 -4.87 0.0
440 -4.64 0.0
450 -4.63 0.0
460 -4.31 0.0
470 -4.18 0.0
480 -4.42 0.0
490 -5.13 0.0
 …[+813 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_axis.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
import numpy as np,torch,time
NF=len(B)
def Qfromz(a):
 a=a/np.linalg.norm(a,axis=-1,keepdims=True);ref=np.array([0.,0.,1.]);v=np.cross(ref,a);c=a@ref
 out=[]
 for ai,vi,ci in zip(a,v,c):
  if ci<-0.999:out.append(np.diag([1.,-1.,-1.]))
  elif np.linalg.norm(vi)<1e-9:out.append(np.eye(3))
  else:
   K=np.array([[0,-vi[2],vi[1]],[vi[2],0,-vi[0]],[-vi[1],vi[0],0]])
   out.append(np.eye(3)+K+K@K/(1+ci))
 return np.array(out)
def decompose(Rs):
 a=Rs[:,:,2];Q=Qfromz(a);ths=[]
 for i in range(NF):ths.append(Rot.from_matrix(Q[i].T@Rs[i]).as_rotvec()[2])
 return a,Q,np.unwrap(np.array(ths))
def smooth_T(wa,wt,static=True):
 T=B.copy();a,Q,th=decompose(B[:,:3,:3].astype(float))
 if static:
  T[:128]=B[127];T[639:]=B[659]
 for A,Bb in [(128,639)]:
  for j in range(3):a[A:Bb,j]=savgol_filter(a[A:Bb,j],wa,2,mode='interp');T[A:Bb,j,3]=savgol_filter(B[A:Bb,j,3],wa,2,mode='interp')
  th[A:Bb]=savgol_filter(th[A:Bb],wt,2,mode='interp')
  Qs=Qfromz(a[A:Bb])
  for i,ii in enumerate(range(A,Bb)):
   c,s=np.cos(th[ii]),np.sin(th[ii]);Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
   T[ii,:3,:3]=Qs[i]@Rz
 return T
def metric(Tm,i):
 with torch.no_grad():a,r=render(torch.tensor(Tm,dtype=torch.float32,device=dev),res=(cw,ch))
 p=(r[0,:,:,3]>0).cpu().numpy();t=obj[i,y0:y1,x0:x1];pp=per …[+609 chars]
```
<details><summary>output</summary>

```
(3, 3) dice [0.5403 0.3786 0.3829] dyn 0.5002 smoothmean/p95 3.87 14.69
(5, 5) dice [0.5417 0.3796 0.3839] dyn 0.5022 smoothmean/p95 2.59 9.33
(9, 9) dice [0.5425 0.3802 0.3844] dyn 0.5033 smoothmean/p95 1.94 6.14
(15, 15) dice [0.543  0.3806 0.3848] dyn 0.5041 smoothmean/p95 1.59 5.0
(21, 21) dice [0.5432 0.3808 0.3849] dyn 0.5044 smoothmean/p95 1.45 4.48
(31, 31) dice [0.5414 0.3795 0.3836] dyn 0.5018 smoothmean/p95 1.34 4.44
(15, 31) dice [0.5427 0.3803 0.3845] dyn 0.5037 smoothmean/p95 1.44 4.72
(31, 15) dice [0.5416 0.3797 0.3838] dyn 0.5021 smoothmean/p95 1.61 5.32
(21, 51) dice [0.5426 0.3802 0.3843] dyn 0.5035 smoothmean/p95 1.33 4.44

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,trimesh
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];K=z['human_keypoints_world'];V=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices[::20]
for i in [120,130,140,150,160,180,200,220,260,300,350,400,450,500,550,600,620,630,640,650,680,700]:
 vw=(T[i,:3,:3]@V.T).T+T[i,:3,3]
 d=np.linalg.norm(vw[:,None]-K[i,42:63][None],axis=2).min()
 print(i,'mindist left',round(float(d),3),'contact',np.round(z['contact_prob_right_left'][i],2),'left centroid',np.round(K[i,42:63].mean(0),2))
PY"
```
<details><summary>output</summary>

```
120 mindist left 0.058 contact [1. 0.] left centroid [0.14 0.3  2.58]
130 mindist left 0.001 contact [1. 0.] left centroid [0.11 0.27 2.65]
140 mindist left 0.002 contact [1. 0.] left centroid [-0.02  0.19  2.86]
150 mindist left 0.029 contact [1.   0.51] left centroid [-0.1   0.16  3.06]
160 mindist left 0.067 contact [1.   0.88] left centroid [-0.13  0.17  3.21]
180 mindist left 0.003 contact [1.   0.99] left centroid [-0.1   0.25  3.52]
200 mindist left 0.005 contact [1.   0.03] left centroid [-0.02  0.32  3.74]
220 mindist left 0.004 contact [0.99 0.  ] left centroid [-0.13  0.39  3.47]
260 mindist left 0.13 contact [1.   0.99] left centroid [0.35 0.29 2.97]
300 mindist left 0.148 contact [0.97 0.01] left centroid [0.25 0.46 3.21]
350 mindist left 0.149 contact [1. 0.] left centroid [- …[+807 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,trimesh
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];K=z['human_keypoints_world'];V=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices[::20]
for i in [130,150,180,220,260,280,300,320,340,350,400,450,500,550,600,620,630]:
 vw=(T[i,:3,:3]@V.T).T+T[i,:3,3]
 dl=np.linalg.norm(vw[:,None]-K[i,42:63][None],axis=2).min();dr=np.linalg.norm(vw[:,None]-K[i,21:42][None],axis=2).min()
 # min human verts all (sample every 4 already)
 dh=np.linalg.norm(vw[:,None]-z['human_vertices_world'][i][None,::20],axis=2).min()
 print(i,'L',round(float(dl),3),'R',round(float(dr),3),'H',round(float(dh),3),'prob',np.round(z['contact_prob_right_left'][i],2))
PY"
```
<details><summary>output</summary>

```
130 L 0.001 R 0.752 H 0.001 prob [1. 0.]
150 L 0.029 R 0.719 H 0.028 prob [1.   0.51]
180 L 0.003 R 0.558 H 0.003 prob [1.   0.99]
220 L 0.004 R 0.412 H 0.002 prob [0.99 0.  ]
260 L 0.13 R 0.647 H 0.12 prob [1.   0.99]
280 L 0.197 R 0.631 H 0.156 prob [1. 1.]
300 L 0.148 R 0.657 H 0.136 prob [0.97 0.01]
320 L 0.018 R 0.5 H 0.014 prob [1. 0.]
340 L 0.002 R 0.488 H 0.002 prob [1. 0.]
350 L 0.149 R 0.557 H 0.086 prob [1. 0.]
400 L 0.005 R 0.208 H 0.002 prob [1. 0.]
450 L 0.031 R 0.254 H 0.023 prob [1.   0.22]
500 L 0.094 R 0.6 H 0.086 prob [1. 0.]
550 L 0.002 R 0.487 H 0.002 prob [1. 0.]
600 L 0.002 R 0.046 H 0.001 prob [1. 1.]
620 L 0.012 R 0.051 H 0.014 prob [1. 1.]
630 L 0.002 R 0.042 H 0.004 prob [1. 1.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,trimesh,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];K=z['human_keypoints_world'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices
cam=json.load(open('/work/inputs/camera.json'));Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
for i in [130,150,180,220,240,250,260,270,280,290,300,310,320,340,350,360,400,450,500,550,600,620]:
 vw=(T[i,:3,:3]@V.T).T+T[i,:3,3]
 dist=np.linalg.norm(vw[:,None]-K[i,42:63][None],axis=2); vi,ki=np.unravel_index(dist.argmin(),dist.shape)
 vec=K[i,42+ki]-vw[vi]
 vc=(Tcw[:3,:3]@vw[vi]+Tcw[:3,3]); hc=(Tcw[:3,:3]@K[i,42+ki]+Tcw[:3,3])
 print(i,'d',round(dist[vi,ki],3),'vec',np.round(vec,3),'cam z obj/hand',np.round([vc[2],hc[2]],3),'xy pix?', 'prob',np.round(z['contact_prob_right_left'][i],2))
PY"
```
<details><summary>output</summary>

```
130 d 0.001 vec [-0.     0.001 -0.   ] cam z obj/hand [2.547 2.547] xy pix? prob [1. 0.]
150 d 0.029 vec [ 0.013 -0.006 -0.025] cam z obj/hand [2.991 2.967] xy pix? prob [1.   0.51]
180 d 0.001 vec [ 0.001  0.    -0.001] cam z obj/hand [3.408 3.407] xy pix? prob [1.   0.99]
220 d 0.002 vec [ 0.     0.002 -0.001] cam z obj/hand [3.356 3.355] xy pix? prob [0.99 0.  ]
240 d 0.099 vec [ 0.004 -0.028 -0.095] cam z obj/hand [3.049 2.953] xy pix? prob [1.   0.86]
250 d 0.22 vec [-0.121 -0.019 -0.183] cam z obj/hand [3.079 2.893] xy pix? prob [0.97 0.13]
260 d 0.13 vec [ 0.004 -0.033 -0.125] cam z obj/hand [3.017 2.891] xy pix? prob [1.   0.99]
270 d 0.163 vec [-0.075  0.009 -0.144] cam z obj/hand [3.082 2.936] xy pix? prob [1.   0.97]
280 d 0.197 vec [-0.112  0.057 -0.151] cam z obj/hand [2.975 2 …[+1186 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [240,250,260,270,280,290,300,310,350,360,500]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();cv2.putText(im,str(i),(20,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,255),3);cv2.imwrite(f'/work/scratch/frames/origx_{i:03d}.jpg',im)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p) as f:
 print(f.keys())
 for k in f.keys():
  v=f[k];print(k,v.shape,v.dtype, dict(v.attrs))
  x=v[0];print(' vals',np.nanmin(x),np.nanmax(x),np.nanmedian(x))
PY"
```
<details><summary>output</summary>

```
<KeysViewHDF5 ['aligned', 'alignment', 'frame_names', 'raw']>
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
AttributeError: 'Group' object has no attribute 'shape'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p) as f:
 for k in f.keys():
  v=f[k];print('\\n',k,type(v))
  if isinstance(v,h5py.Dataset):
   print(v.shape,v.dtype,dict(v.attrs));x=v[0];print(np.nanmin(x),np.nanmax(x),np.nanmedian(x))
  else:
   print(list(v.keys()))
   for kk in v.keys():
    w=v[kk];print(' ',kk,getattr(w,'shape',None),getattr(w,'dtype',None),dict(w.attrs))
    if isinstance(w,h5py.Dataset):
     x=w[0];
     if np.issubdtype(x.dtype,np.number):print('  vals',np.nanmin(x),np.nanmax(x),np.nanmedian(x))
PY"
```
<details><summary>output</summary>

```
  vals 0 255 127.0

 alignment <class 'h5py._hl.group.Group'>
['front_stereo_camera_left']
  front_stereo_camera_left None None {}

 frame_names <class 'h5py._hl.group.Group'>
['front_stereo_camera_left']
  front_stereo_camera_left (747,) object {}
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
AttributeError: 'bytes' object has no attribute 'dtype'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 ds=f['aligned/front_stereo_camera_left']; print(ds.shape,ds.dtype);v=ds[0];print(type(v),getattr(v,'shape',None),getattr(v,'dtype',None));print(v)
 ds=f['raw/front_stereo_camera_left']; print(ds.shape,ds.dtype);v=ds[0];print(type(v),getattr(v,'shape',None),getattr(v,'dtype',None)); print(v if not hasattr(v,'shape') else (v.shape,v.dtype,np.nanmin(v),np.nanmax(v),np.nanmedian(v)))
PY"
```
<details><summary>output</summary>

```
(747,) object
<class 'numpy.ndarray'> (676571,) uint8
[137  80  78 ...  66  96 130]
(747,) object
<class 'numpy.ndarray'> (622313,) uint8
((622313,), dtype('uint8'), 0, 255, 127.0)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for grp in ['aligned','raw']:
  a=f[grp]['front_stereo_camera_left'][0];im=cv2.imdecode(a,cv2.IMREAD_UNCHANGED);print(grp,a.shape,im.shape,im.dtype,np.nanmin(im),np.nanmax(im),np.nanmedian(im))
  cv2.imwrite(f'/work/scratch/{grp}0.png',im)
PY"
```
<details><summary>output</summary>

```
aligned (676571,) (1152, 1536) uint16 1636 10016 4068.0
raw (622313,) (1152, 1536) uint16 1467 8979 3647.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,cv2,numpy as np,json,trimesh
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices[::20]
cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f, h5py.File('/work/inputs/masks.h5') as fm:
 ds=f['aligned/front_stereo_camera_left'];ob=fm['object'];per=fm['person']
 for i in [0,50,100,127,130,150,180,220,260,280,300,350,400,450,500,550,600,620,640,656,700]:
  d=cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
  # object mask excluding person
  mm=ob[i]&~per[i]; vals=d[mm]
  vw=(T[i,:3,:3]@V.T).T+T[i,:3,3];vc=(Tcw[:3,:3]@vw.T).T+Tcw[:3,3];pre=vc[:,2]
  print(i,'MoGe med/q10q90',np.round([np.median(vals),np.quantile(vals,.1),np.quantile(vals,.9)],3),'model med/q',np.round([np.median(pre),np.quantile(pre,.1),np.quantile(pre,.9)],3),'diff med',round(float(np.median(pre)-np.median(vals)),3))
PY"
```
<details><summary>output</summary>

```
0 MoGe med/q10q90 [2.716 2.586 2.94 ] model med/q [2.569 2.529 2.614] diff med -0.147
50 MoGe med/q10q90 [2.744 2.671 3.06 ] model med/q [2.583 2.545 2.63 ] diff med -0.161
100 MoGe med/q10q90 [2.638 2.501 2.845] model med/q [2.578 2.539 2.625] diff med -0.06
127 MoGe med/q10q90 [2.617 2.5   2.805] model med/q [2.575 2.528 2.638] diff med -0.042
130 MoGe med/q10q90 [2.612 2.498 2.741] model med/q [2.612 2.564 2.676] diff med -0.0
150 MoGe med/q10q90 [3.08  2.948 3.79 ] model med/q [3.059 3.007 3.126] diff med -0.021
180 MoGe med/q10q90 [3.294 3.092 3.652] model med/q [3.39  3.35  3.435] diff med 0.096
220 MoGe med/q10q90 [3.318 3.196 3.548] model med/q [3.261 3.177 3.338] diff med -0.057
260 MoGe med/q10q90 [3.27  3.023 3.635] model med/q [3.145 3.062 3.217] diff med -0.125
280 MoGe med/q1 …[+1039 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/photo_compare.py <<'PY'
ns={};exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0],ns)
exec(open('/work/scratch/photo_eval.py').read().split(\"for i in [\")[0],ns)
# ns now has render_col, score, smooth_T from separate execs? score globals point photo namespace and B etc same
import numpy as np,torch
T0=ns['B'];T1=ns['smooth_T'](21,21)
for i in [0,26,100,127,130,140,150,180,220,240,260,280,300,320,350,400,450,500,550,600,620,639,650,656,700]:
 vals=[]
 for T in [T0,T1]:
  vals.append(ns['score'](i,torch.tensor(T[i],dtype=torch.float32,device=ns['dev'])))
 print(i,'base',tuple(round(x,3) if isinstance(x,float) else x for x in vals[0]),'smooth',tuple(round(x,3) if isinstance(x,float) else x for x in vals[1]))
PY
python /work/scratch/photo_compare.py"
```
<details><summary>output</summary>

```
0 base (0.191, 0.118, 5155) smooth (0.198, 0.187, 5358)
26 base (0.22, 0.068, 5434) smooth (0.207, 0.116, 5207)
100 base (0.157, 0.13, 5229) smooth (0.192, 0.182, 5427)
127 base (0.202, 0.18, 5802) smooth (0.202, 0.18, 5802)
130 base (0.2, 0.175, 5846) smooth (0.207, 0.165, 5677)
140 base (0.102, 0.123, 4479) smooth (0.134, 0.103, 4566)
150 base (0.1, 0.046, 2817) smooth (0.098, 0.063, 2889)
180 base (0.087, -0.064, 2325) smooth (0.092, -0.062, 2332)
220 base (0.007, 0.002, 1636) smooth (0.039, 0.014, 1721)
240 base (0.061, 0.031, 1989) smooth (-0.011, -0.044, 1825)
260 base (-0.063, 0.019, 3688) smooth (-0.034, -0.004, 3348)
280 base (0.056, 0.226, 2537) smooth (-0.016, 0.213, 2294)
300 base (0.06, 0.139, 1700) smooth (-0.015, 0.031, 1821)
320 base (0.291, 0.024, 1751) smooth (0.183, 0.14 …[+651 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');K=z['human_keypoints_world']
for i in [130,180,280,400,600]:
 h=K[i,42:63];w=h[0];print('\\nframe',i,'wrist',np.round(w,2));
 for j in range(21):print(j,np.round(h[j]-w,3),round(np.linalg.norm(h[j]-w),3))
PY"
```
<details><summary>output</summary>

```

frame 130 wrist [0.13 0.27 2.61]
0 [0. 0. 0.] 0.0
1 [-0.013 -0.018  0.011] 0.025
2 [-0.034 -0.04   0.024] 0.058
3 [-0.052 -0.058  0.043] 0.088
4 [0.004 0.031 0.009] 0.032
5 [0.013 0.014 0.017] 0.026
6 [ 0.019 -0.006  0.028] 0.034
7 [ 0.001 -0.036  0.04 ] 0.054
8 [-0.02   0.025  0.021] 0.038
9 [-0.001  0.021  0.031] 0.037
10 [0.016 0.009 0.045] 0.049
11 [ 0.001 -0.025  0.063] 0.067
12 [-0.031  0.025  0.031] 0.05
13 [-0.014  0.025  0.043] 0.052
14 [0.002 0.015 0.061] 0.062
15 [-0.01  -0.017  0.077] 0.08
16 [-0.034  0.023  0.049] 0.064
17 [-0.021  0.024  0.062] 0.07
18 [-0.013  0.015  0.077] 0.08
19 [-0.026 -0.008  0.089] 0.093
20 [-0.064 -0.067  0.078] 0.121

frame 180 wrist [-0.09  0.28  3.48]
0 [0. 0. 0.] 0.0
1 [-0.004 -0.025  0.001] 0.025
2 [-0.015 -0.056  0.004] 0.058
3 [-0.025 -0.083   …[+2551 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];K=z['human_keypoints_world'];Rs=T[:,:3,:3]
def rmats(i):
 h=K[i,42:63]; o=h[0]
 x=h[9]-h[0];y=h[17]-h[5];z=np.cross(x,y);x=np.cross(y,z)
 x/=np.linalg.norm(x);y/=np.linalg.norm(y);z/=np.linalg.norm(z)
 return np.stack([x,y,z],1)
for st,en in [(130,230),(130,630),(340,630),(400,630),(550,630)]:
 X=[]
 for i in range(st,en):
  H=rmats(i);X.append(H.T@Rs[i])
 X=np.array(X);mean=R.from_matrix(X).mean().as_matrix();d=np.degrees(R.from_matrix(X.transpose(0,2,1)@mean).magnitude())
 print(st,en,'rel mean\\n',np.round(mean,3),'angle med/max',np.median(d),d.max())
PY"
```
<details><summary>output</summary>

```
130 230 rel mean
 [[ 0.643 -0.022 -0.766]
 [ 0.569  0.684  0.458]
 [ 0.514 -0.73   0.452]] angle med/max 40.67331068833582 57.683438140829125
130 630 rel mean
 [[ 0.462  0.12  -0.879]
 [ 0.715  0.536  0.449]
 [ 0.525 -0.836  0.162]] angle med/max 47.57702328745296 170.02600330509833
340 630 rel mean
 [[ 0.431  0.147 -0.89 ]
 [ 0.768  0.458  0.447]
 [ 0.474 -0.877  0.084]] angle med/max 46.735205690871865 113.46808797919279
400 630 rel mean
 [[ 0.225  0.161 -0.961]
 [ 0.871  0.409  0.273]
 [ 0.437 -0.898 -0.048]] angle med/max 39.50634216695971 119.72790717577892
550 630 rel mean
 [[-0.198 -0.287 -0.937]
 [ 0.976 -0.149 -0.161]
 [-0.094 -0.946  0.309]] angle med/max 50.32753414038936 116.944013854232

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/pnp_test.py <<'PY'
import os,json,cv2,h5py,numpy as np,trimesh,torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
dev='cuda';H,W=1152,1536
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev);F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'];ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:];per=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');gray=[]
def imread(i):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();return im

def visible_points(i,Tm,n=3000):
 # choose deterministic target-mask pixels maximizing coverage
 ys,xs=np.where(obj[i] & ~per[i])
 # grid-ish random
 rng=np.random.default_rng(123+i);sel=rng.choice(len(xs),min(n,len(xs)),replace=False);u=xs[sel].astype(np.float32);v=ys[sel].astype(np.float32)
 vw=(Tm[:3,:3]@V.T).T+Tm[:3,3]
 vc=(Tcw[:3,:3]@vw.T).T+Tcw[:3,3]
 uu=K[0,0]*vc[:,0]/vc[:,2]+K[0,2];vv=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]
 clip=torch.stack([2*uu/W-1,2*vv/H-1,(vc[:,2]-2)/4,torch.ones_like(uu)],1)[None]
 rast,_=dr.rasterize(ctx,clip,F,(H,W))
 tri=rast[0,:,:,3].long(); uv=rast[0,:,:,:2]
 # bilin …[+2543 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pnp_test.py", line 74, in <module>
    eval_pair(i,i+1)
  File "/work/scratch/pnp_test.py", line 54, in eval_pair
    pts,xyz,tri=visible_points(i,B[i],2000)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pnp_test.py", line 19, in visible_points
    vw=(Tm[:3,:3]@V.T).T+Tm[:3,3]
        ~~~~~~~~~^~~~
TypeError: unsupported operand type(s) for @: 'numpy.ndarray' and 'Tensor'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/pnp_test.py';s=open(p).read();s=s.replace(\"V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev);F=\", \"Vn=mesh.vertices.astype(np.float32);V=torch.tensor(Vn,dtype=torch.float32,device=dev);F=\").replace(\"vw=(Tm[:3,:3]@V.T).T+Tm[:3,3]\", \"vw=(Tm[:3,:3]@Vn.T).T+Tm[:3,3]\")
open(p,'w').write(s)
PY
python /work/scratch/pnp_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pnp_test.py", line 74, in <module>
    eval_pair(i,i+1)
  File "/work/scratch/pnp_test.py", line 54, in eval_pair
    pts,xyz,tri=visible_points(i,B[i],2000)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pnp_test.py", line 22, in visible_points
    clip=torch.stack([2*uu/W-1,2*vv/H-1,(vc[:,2]-2)/4,torch.ones_like(uu)],1)[None]
                                                      ^^^^^^^^^^^^^^^^^^^
TypeError: ones_like(): argument 'input' (position 1) must be Tensor, not TrackedArray

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/pnp_test.py';s=open(p).read();s=s.replace(\"clip=torch.stack([2*uu/W-1,2*vv/H-1,(vc[:,2]-2)/4,torch.ones_like(uu)],1)[None]\", \"clip=torch.tensor(np.stack([2*uu/W-1,2*vv/H-1,(vc[:,2]-2)/4,np.ones_like(uu)],1),dtype=torch.float32,device=dev)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/pnp_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pnp_test.py", line 74, in <module>
    eval_pair(i,i+1)
  File "/work/scratch/pnp_test.py", line 54, in eval_pair
    pts,xyz,tri=visible_points(i,B[i],2000)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pnp_test.py", line 28, in visible_points
    xyz,_=dr.interpolate(vw[None],rast,F)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 283, in interpolate
    assert all(isinstance(x, torch.Tensor) for x in (attr, rast, tri))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i 's/dr.interpolate(vw\\[None\\],rast,F)/dr.interpolate(torch.tensor(vw[None],dtype=torch.float32,device=dev),rast,F)/' /work/scratch/pnp_test.py
python /work/scratch/pnp_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pnp_test.py", line 74, in <module>
    eval_pair(i,i+1)
  File "/work/scratch/pnp_test.py", line 54, in eval_pair
    pts,xyz,tri=visible_points(i,B[i],2000)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pnp_test.py", line 28, in visible_points
    xyz,_=dr.interpolate(torch.tensor(vw[None],dtype=torch.float32,device=dev),rast,F)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    r …[+412 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/torch.tensor(vw\\[None\\],dtype=torch.float32,device=dev)/torch.tensor(vw[None],dtype=torch.float32,device=dev).contiguous()/' /work/scratch/pnp_test.py
python /work/scratch/pnp_test.py"
```
<details><summary>output</summary>

```

PAIR 0 1 pts 1289 tracked 1279 fb 0.0016580905
base err med/mean 0.6831006104965384 0.694797674622766 pnp 0.025075163568736757 0.042077276461326536
rot diff base-pnp 0.02981318768197546 t diff 0.0038412253
base t [0.29623052 0.87714106 2.6790168 ] pnp t [0.29445058 0.87476784 2.6765766 ]

PAIR 50 51 pts 1321 tracked 1317 fb 0.0014196415
base err med/mean 0.6643621424856186 0.679174756398097 pnp 0.036262719414892196 0.04726088664544404
rot diff base-pnp 0.10682614882288567 t diff 0.003828857
base t [0.2992627  0.88484627 2.6963818 ] pnp t [0.29745322 0.8825517  2.6939077 ]

PAIR 100 101 pts 1319 tracked 1317 fb 0.00080279214
base err med/mean 0.7094386821156874 0.7072465429674574 pnp 0.02882268735668711 0.04456707674506363
rot diff base-pnp 0.029404844612501847 t diff 0.00215695
base t [0. …[+4910 chars]
```
</details>

**shell** (exit 130)
```
/usr/bin/bash -lc "cat > /work/scratch/propagate.py <<'PY'
exec(open('/work/scratch/pnp_test.py').read().split(\"if __name__\")[0])
# override visible_points with crop
x0,y0,x1,y1=650,500,950,1000;cw=x1-x0;ch=y1-y0
def visible_points_crop(i,Tm,n=1800):
 ys,xs=np.where(obj[i,y0:y1,x0:x1] & ~per[i,y0:y1,x0:x1])
 rng=np.random.default_rng(123+i);sel=rng.choice(len(xs),min(n,len(xs)),replace=False);uc=xs[sel].astype(np.float32);vc_=ys[sel].astype(np.float32)
 vw=(Tm[:3,:3]@Vn.T).T+Tm[:3,3]
 camer=(Tcw[:3,:3]@vw.T).T+Tcw[:3,3]
 u=K[0,0]*camer[:,0]/camer[:,2]+K[0,2]-x0;v=K[1,1]*camer[:,1]/camer[:,2]+K[1,2]-y0
 clip=torch.tensor(np.stack([2*u/cw-1,2*v/ch-1,(camer[:,2]-2)/4,np.ones_like(u)],1),dtype=torch.float32,device=dev)[None]
 rast,_=dr.rasterize(ctx,clip,F,(ch,cw))
 tri=rast[0,:,:,3].long()
 ui=np.clip(np.round(uc).astype(int),0,cw-1);vi=np.clip(np.round(vc_).astype(int),0,ch-1)
 triid=tri[vi,ui].cpu().numpy();ok=triid>0
 xyz,_=dr.interpolate(torch.tensor(vw[None],dtype=torch.float32,device=dev).contiguous(),rast,F);xyz=xyz[0].cpu().numpy()
 pts=np.stack([uc[ok]+x0,vc_[ok]+y0],1);return pts,xyz[vi[ok],ui[ok]],triid[ok]

def estimate(i,j,Ti,vis=1800):
 pts,xyz,tri=visible_points_crop(i,Ti,vis)
 p1,ok,fb=track(i,j,pts)
 o=(np.linalg.inv(Ti)@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
 if ok.sum()<20:return None,dict(n=int(ok.sum()),err=99,fb=99)
 Tp=pnp(o[ok],p1[ok],B[j])
 # LM refine
 Tc=Tcw@Tp;rv=Rot.from_matrix(Tc[:3,:3]).as_rotvec().astype(np.float32);tv=Tc[:3,3].astype(np.float3 …[+1473 chars]
```
<details><summary>output</summary>

```
FW 140 {'n': 1067, 'err': 1.318266975653222, 'mean': 1.4722442633920092, 'fb': 0.05901768058538437} smooth
FW 160 {'n': 1181, 'err': 0.3188640159308365, 'mean': 0.331364546085197, 'fb': 0.006737684365361929} smooth
Traceback (most recent call last):
  File "/work/scratch/propagate.py", line 36, in <module>
    Tp,q=estimate(i,i+1,fw[i]);
         ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/propagate.py", line 20, in estimate
    p1,ok,fb=track(i,j,pts)
             ^^^^^^^^^^^^^^
  File "<string>", line 33, in track
  File "<string>", line 12, in imread
KeyboardInterrupt

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/propagate.py';s=open(p).read()
s=s.replace(\"x0,y0,x1,y1=650,500,950,1000;cw=x1-x0;ch=y1-y0\", \"x0,y0,x1,y1=650,500,950,1000;cw=x1-x0;ch=y1-y0\\nprint('preload video crops')\\n_cap=cv2.VideoCapture('/work/inputs/video.mp4');G=[]\\nwhile True:\\n _ok,_im=_cap.read()\\n if not _ok:break\\n G.append(cv2.cvtColor(_im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY))\\nprint('loaded',len(G))\\n_cap.release()\")
s=s.replace(\"def track(i,j,pts):\\n g0=cv2.cvtColor(imread(i),cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(imread(j),cv2.COLOR_BGR2GRAY)\\n p0=pts.reshape(-1,1,2).astype(np.float32)\", \"def track(i,j,pts):\\n g0=G[i];g1=G[j]\\n p0=(pts-np.array([x0,y0])).reshape(-1,1,2).astype(np.float32)\")
s=s.replace(\"p1=p1.reshape(-1,2)\\n # keep inside obj/person situation next frame, or adjacent to object\\n x=p1[:,0];y=p1[:,1];xi=np.clip(np.round(x).astype(int),0,W-1);yi=np.clip(np.round(y).astype(int),0,H-1)\\n ok &= obj[j][yi,xi] # texture point may fall in small gaps; strict\\n return p1,ok,fb\", \"p1=p1.reshape(-1,2)\\n pfull=p1+np.array([x0,y0],dtype=np.float32)\\n x=pfull[:,0];y=pfull[:,1];xi=np.clip(np.round(x).astype(int),0,W-1);yi=np.clip(np.round(y).astype(int),0,H-1)\\n ok &= obj[j][yi,xi] # texture point may fall in small gaps; strict\\n return pfull,ok,fb\")
s=s.replace(\"def estimate(i,j,Ti,vis=1800):\",\"def estimate(i,j,Ti,vis=1000):\")
open(p,'w').write(s)
PY
python /work/scratch/propagate.py"
```
<details><summary>output</summary>

```
preload video crops
loaded 747
FW 140 {'n': 607, 'err': 1.3564492803398696, 'mean': 1.5170084168318494, 'fb': 0.066127248108387} smooth
FW 160 {'n': 657, 'err': 0.3065611803952909, 'mean': 0.3250715022335209, 'fb': 0.006591796875} smooth
Traceback (most recent call last):
  File "/work/scratch/propagate.py", line 44, in <module>
    Tp,q=estimate(i,i+1,fw[i]);
         ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/propagate.py", line 28, in estimate
    p1,ok,fb=track(i,j,pts)
             ^^^^^^^^^^^^^^
  File "<string>", line 33, in track
  File "<string>", line 12, in imread
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n \"def track\" -A15 /work/scratch/propagate.py; sed -n '1,20p' /work/scratch/propagate.py"
```
<details><summary>output</summary>

```
exec(open('/work/scratch/pnp_test.py').read().split("if __name__")[0])
# override visible_points with crop
x0,y0,x1,y1=650,500,950,1000;cw=x1-x0;ch=y1-y0
print('preload video crops')
_cap=cv2.VideoCapture('/work/inputs/video.mp4');G=[]
while True:
 _ok,_im=_cap.read()
 if not _ok:break
 G.append(cv2.cvtColor(_im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY))
print('loaded',len(G))
_cap.release()
def visible_points_crop(i,Tm,n=1800):
 ys,xs=np.where(obj[i,y0:y1,x0:x1] & ~per[i,y0:y1,x0:x1])
 rng=np.random.default_rng(123+i);sel=rng.choice(len(xs),min(n,len(xs)),replace=False);uc=xs[sel].astype(np.float32);vc_=ys[sel].astype(np.float32)
 vw=(Tm[:3,:3]@Vn.T).T+Tm[:3,3]
 camer=(Tcw[:3,:3]@vw.T).T+Tcw[:3,3]
 u=K[0,0]*camer[:,0]/camer[:,2]+K[0,2]-x0;v=K[1,1]*camer[:,1]/camer[:,2]+K[1,2]-y0
 clip=torch.tensor …[+171 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/propagate2.py <<'PY'
import os,json,cv2,h5py,numpy as np,trimesh,torch,time
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
dev='cuda';H,W=1152,1536
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();Vn=mesh.vertices.astype(np.float32)
V=torch.tensor(Vn,dtype=torch.float32,device=dev);F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'];ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:];per=f['person'][:]
x0,y0,x1,y1=650,500,950,1000;cw=x1-x0;ch=y1-y0
print('preload')
cap=cv2.VideoCapture('/work/inputs/video.mp4');G=[]
while True:
 ok,im=cap.read()
 if not ok:break
 G.append(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY))
cap.release();print('loaded',len(G))

def visible_points_crop(i,Tm,n=700):
 ys,xs=np.where(obj[i,y0:y1,x0:x1] & ~per[i,y0:y1,x0:x1])
 rng=np.random.default_rng(123+i);sel=rng.choice(len(xs),min(n,len(xs)),replace=False);uc=xs[sel].astype(np.float32);vc_=ys[sel].astype(np.float32)
 vw=(Tm[:3,:3]@Vn.T).T+Tm[:3,3]
 camer=(Tcw[:3,:3]@vw.T).T+Tcw[:3,3]
 u=K[0,0]*camer[:,0]/camer[:,2]+K[0,2]-x0;v=K[1,1]*camer[:,1]/camer[:,2]+K[1,2]-y0
 clip=torch.tensor(np.stack([2*u/cw-1,2*v/ch …[+3595 chars]
```
<details><summary>output</summary>

```
preload
loaded 747
127 {'n': 438, 'nin': 308, 'err': 1.1454849588639484, 'mean': 1.5160353889228213, 'fb': 0.12956741452217102} time 0.06381988525390625 rotdiff 10.82860866445116
150 {'n': 329, 'nin': 304, 'err': 0.28177542464738364, 'mean': 0.6252570564385829, 'fb': 0.03060799278318882} time 0.018603086471557617 rotdiff 2.2911859585176533
220 {'n': 276, 'nin': 261, 'err': 0.4076255548053249, 'mean': 0.6121531932747406, 'fb': 0.05857217311859131} time 0.017871379852294922 rotdiff 5.050838434204723
273 {'n': 380, 'nin': 380, 'err': 0.47380086333018123, 'mean': 0.5121551753998563, 'fb': 0.011518243700265884} time 0.01793646812438965 rotdiff 54.03973679830336
300 {'n': 281, 'nin': 267, 'err': 0.45814877458540215, 'mean': 0.5584349251124355, 'fb': 0.057112570852041245} time 0.01680326461791992 …[+516 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc 'python /work/scratch/propagate2.py'
```
<details><summary>output</summary>

```
preload
loaded 747
FW 140 {'n': 359, 'nin': 283, 'err': 0.3812802774476602, 'mean': 1.1081159821924826, 'fb': 0.05458837375044823}
FW 160 {'n': 356, 'nin': 356, 'err': 0.3521455795448583, 'mean': 0.34329597009764606, 'fb': 0.011988690122961998}
FW 180 {'n': 220, 'nin': 207, 'err': 0.3201027061551326, 'mean': 0.48313901737856496, 'fb': 0.03646162152290344}
FW 200 {'n': 156, 'nin': 156, 'err': 0.34121458825190654, 'mean': 0.4384188063110073, 'fb': 0.029357191175222397}
FW 220 {'n': 92, 'nin': 86, 'err': 0.524145697115211, 'mean': 0.690794233833355, 'fb': 0.11704562604427338}
Traceback (most recent call last):
  File "/work/scratch/propagate2.py", line 66, in <module>
    Tp,q=estimate(i,i+1,fw[i]);
         ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/propagate2.py", line 43, in estimate
     …[+697 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/independent.py <<'PY'
exec(open('/work/scratch/propagate2.py').read().split(\"if __name__\")[0])
import time
Fw={};Bw={};Qf={};Qb={};t=time.time()
for j in range(128,640):
 try:Tf,q=estimate(j-1,j,B[j-1],500)
 except Exception as e:Tf=None;q={'e':str(e)}
 Fw[j]=Tf;Qf[j]=q
 if j%20==0:print('F',j,q)
print('forward',time.time()-t);t=time.time()
for j in range(128,640):
 try:Tb,q=estimate(j+1,j,B[j+1],500)
 except Exception as e:Tb=None;q={'e':str(e)}
 Bw[j]=Tb;Qb[j]=q
 if j%20==0:print('B',j,q)
print('backward',time.time()-t)
np.savez('/work/scratch/independent.npz',Fw=np.array([Fw.get(i,np.eye(4)) for i in range(747)]),Bw=np.array([Bw.get(i,np.eye(4)) for i in range(747)]),qf=np.array([Qf.get(i,{}) for i in range(747)],dtype=object),qb=np.array([Qb.get(i,{}) for i in range(747)],dtype=object))
PY
python /work/scratch/independent.py"
```
<details><summary>output</summary>

```
preload
loaded 747
F 140 {'n': 287, 'nin': 235, 'err': 0.38963293286958356, 'mean': 1.04649517482779, 'fb': 0.06652137637138367}
F 160 {'n': 268, 'nin': 267, 'err': 0.27952111146222425, 'mean': 0.34323088621334374, 'fb': 0.01178003940731287}
F 180 {'n': 278, 'nin': 257, 'err': 0.3683976138139532, 'mean': 0.4999247047757526, 'fb': 0.02567277103662491}
F 200 {'n': 273, 'nin': 267, 'err': 0.3218129618808063, 'mean': 0.4582282551292182, 'fb': 0.026348376646637917}
F 220 {'n': 178, 'nin': 168, 'err': 0.4546981193389233, 'mean': 0.658911779470681, 'fb': 0.07041161507368088}
F 240 {'n': 190, 'nin': 165, 'err': 0.7265363854365148, 'mean': 1.1032989302477392, 'fb': 0.08045965433120728}
F 260 {'n': 225, 'nin': 214, 'err': 0.31073872893941445, 'mean': 0.4344724397137251, 'fb': 0.020058246329426765}
F …[+4864 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/analyze_ind.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
from scipy.spatial.transform import Rotation as Rot
D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw'];qf=D['qf'];qb=D['qb']
def ad(a,b):return np.degrees(Rot.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())
def td(a,b):return np.linalg.norm(a[:3,3]-b[:3,3])
for i in range(128,640,10):
 print(i,'Fbase',round(ad(FW[i],B[i]),1),'Bbase',round(ad(BW[i],B[i]),1),'FB',round(ad(FW[i],BW[i]),1),'tdFB',round(td(FW[i],BW[i]),3),'errs',round(float(qf[i].get('err',99)),2),round(float(qb[i].get('err',99)),2))
# metrics selected poses
for name,Ts in [('base',B),('fw',FW),('bw',BW)]:
 vals=[]
 for i in range(128,640):vals.append(metric(Ts[i],i))
 print(name,np.round(np.mean(vals,0),4))
PY
python /work/scratch/analyze_ind.py"
```
<details><summary>output</summary>

```
128 Fbase 7.1 Bbase 2.4 FB 9.4 tdFB 0.022 errs 1.1 1.05
138 Fbase 6.8 Bbase 2.3 FB 4.7 tdFB 0.055 errs 0.5 0.41
148 Fbase 3.8 Bbase 1.5 FB 3.8 tdFB 0.042 errs 0.56 0.42
158 Fbase 5.8 Bbase 10.4 FB 16.0 tdFB 0.079 errs 0.3 0.35
168 Fbase 6.7 Bbase 6.6 FB 13.1 tdFB 0.036 errs 0.28 0.51
178 Fbase 2.6 Bbase 3.0 FB 5.5 tdFB 0.02 errs 0.38 0.3
188 Fbase 2.5 Bbase 5.7 FB 3.7 tdFB 0.048 errs 0.37 0.27
198 Fbase 3.8 Bbase 1.3 FB 5.0 tdFB 0.008 errs 0.25 0.3
208 Fbase 7.8 Bbase 1.0 FB 8.3 tdFB 0.048 errs 0.69 0.83
218 Fbase 2.2 Bbase 4.8 FB 3.1 tdFB 0.032 errs 0.68 0.45
228 Fbase 2.2 Bbase 3.9 FB 5.6 tdFB 0.029 errs 0.41 0.55
238 Fbase 14.4 Bbase 5.3 FB 11.0 tdFB 0.107 errs 0.49 0.79
248 Fbase 26.1 Bbase 8.7 FB 20.2 tdFB 0.091 errs 0.67 0.57
258 Fbase 19.0 Bbase 9.1 FB 24.5 tdFB 0.048 errs 0.38 0.47 …[+2263 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
def dec(Rs):
 a=Rs[:,:,2];Q=Qfromz(a);th=[]
 for i in range(len(Rs)):th.append(Rot.from_matrix(Q[i].T@Rs[i]).as_rotvec()[2])
 return np.unwrap(np.array(th))
tb=dec(B[:747,:3,:3].astype(float));tf=dec(FW[:747,:3,:3].astype(float));tw=dec(BW[:747,:3,:3].astype(float))
print('theta degrees region')
for i in range(200,341,2):print(i,round(np.degrees(tb[i]),1),round(np.degrees(tf[i]),1),round(np.degrees(tw[i]),1), 'd',round(np.degrees(tf[i]-tb[i]),1),round(np.degrees(tw[i]-tb[i]),1))
PY"
```
<details><summary>output</summary>

```
theta degrees region
200 19.9 19.8 17.2 d -0.0 -2.6
202 12.3 15.8 15.4 d 3.5 3.1
204 12.1 15.2 13.7 d 3.1 1.6
206 16.5 12.5 27.7 d -4.1 11.1
208 19.3 26.7 18.3 d 7.4 -1.0
210 17.9 15.0 27.8 d -2.9 9.8
212 23.1 23.3 17.3 d 0.2 -5.8
214 18.9 16.5 25.8 d -2.4 6.9
216 25.4 23.2 23.9 d -2.2 -1.5
218 28.9 27.5 25.1 d -1.4 -3.8
220 35.0 26.9 33.5 d -8.1 -1.5
222 39.3 35.5 29.9 d -3.8 -9.4
224 38.8 32.0 42.1 d -6.8 3.3
226 31.0 43.9 28.4 d 12.8 -2.6
228 31.1 31.4 27.3 d 0.3 -3.8
230 28.4 29.1 22.6 d 0.7 -5.8
232 25.0 25.6 27.2 d 0.6 2.2
234 35.1 28.1 23.9 d -7.0 -11.2
236 31.5 26.0 35.6 d -5.6 4.1
238 25.6 39.1 30.5 d 13.5 4.9
240 43.7 34.0 42.4 d -9.7 -1.4
242 29.6 44.7 12.7 d 15.1 -16.8
244 25.7 16.8 11.0 d -8.8 -14.6
246 28.1 14.5 10.5 d -13.6 -17.5
248 -4.1 15.2 6.8 d 19.3 10.8
250 7.2 8.3 -2. …[+1625 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
def dec(Rs):
 a=Rs[:,:,2];Q=Qfromz(a);return np.unwrap(np.array([Rot.from_matrix(Q[i].T@Rs[i]).as_rotvec()[2] for i in range(len(Rs))]))
tb=dec(B[:,:3,:3].astype(float));tf=dec(FW[:,:3,:3].astype(float));tw=dec(BW[:,:3,:3].astype(float))
for i in range(255,321):print(i,round(np.degrees(tb[i]),1),round(np.degrees(tf[i]),1),round(np.degrees(tw[i]),1), 'fb',round(np.degrees(tf[i]-tw[i]),1))
PY"
```
<details><summary>output</summary>

```
255 -0.4 24.2 -1.3 fb 25.5
256 -0.4 0.6 -9.0 fb 9.6
257 -8.8 0.2 -26.5 fb 26.7
258 -26.9 -8.8 -32.6 fb 23.8
259 -32.6 -27.1 -40.7 fb 13.6
260 -40.4 -32.6 -40.3 fb 7.7
261 -40.4 -40.4 -40.5 fb 0.1
262 -40.4 -40.2 -82.4 fb 42.2
263 -82.5 -40.3 -86.2 fb 45.9
264 -86.4 -82.6 -86.2 fb 3.6
265 -86.4 -86.5 -94.0 fb 7.5
266 -94.2 -86.6 -108.1 fb 21.5
267 -108.5 -94.4 -117.4 fb 23.0
268 -117.8 -108.7 -127.1 fb 18.3
269 -127.7 -118.5 -126.8 fb 8.4
270 -127.7 -128.4 -138.6 fb 10.2
271 -139.0 -128.9 -138.6 fb 9.8
272 -139.0 -139.6 -108.2 fb -31.4
273 -108.5 -139.6 -161.4 fb 21.8
274 -161.9 -108.7 -151.3 fb 42.6
275 -151.8 -162.2 -139.4 fb -22.9
276 -139.5 -152.2 -174.6 fb 22.4
277 -175.0 -139.6 -175.1 fb 35.5
278 -175.0 -175.1 -156.4 fb -18.8
279 -156.1 -174.4 -176.4 fb 1.9
280 -176.2 -155.9 -186.2 fb …[+1332 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/flow_validate.py <<'PY'
exec(open('/work/scratch/propagate2.py').read().split(\"if __name__\")[0])
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle,time
pairs={}
t=time.time()
for i in range(127,639):
 pts,xyz,tri=visible_points_crop(i,B[i],400);p1,ok,fb=track_local(i,i+1,pts)
 o=(np.linalg.inv(B[i])@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
 pairs[i+1]=(o[ok],p1[ok],ok.sum())
 if i%100==0:print(i,ok.sum(),time.time()-t)
print('track time',time.time()-t)
def err(Ts):
 es=[];ems=[];es2=[]
 for j,(o,p,n) in pairs.items():
  pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];e=np.linalg.norm(uv-p,axis=1)
  es.append(np.median(e));ems.append(np.mean(e));es2.extend(e)
 return np.mean(es),np.median(es),np.mean(ems),np.quantile(es2,[.5,.8,.9,.95])
print('BASE',err(B))
for w in [3,5,9,15,21,31,41,51]:print('SG',w,err(smooth_T(w,w)))
# independent merged: choose candidate closer to smooth per frame or average rotations/trans
for w in [9,15,21,31]:
 S=smooth_T(w,w);T=S.copy();D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
 for i in range(128,640):
  # average FW/BW and blend 50% with smooth
  Rm=Rot.from_matrix(np.stack([FW[i][:3,:3],BW[i][:3,:3]])).mean().as_matrix();tm=(FW[i][:3,3]+BW[i][:3,3])/2
  R=Rot.from_matrix(np.stack([S[i][:3,:3],Rm])).mean().as_matrix();T[i,:3,:3]=R;T[i,:3,3]=(S[i][:3,3]+tm)/2
 prin …[+61 chars]
```
<details><summary>output</summary>

```
preload
loaded 747
200 217 1.1839289665222168
300 146 3.0646021366119385
400 188 5.349461793899536
500 116 7.255330324172974
600 243 9.559114694595337
track time 10.293525695800781
BASE (3.4452854856324406, 2.6007384864441914, 3.6104054885538175, array([2.55261837, 4.84321924, 6.87708461, 9.3075652 ]))
SG 3 (3.4460038394731534, 2.600739556156859, 3.6111209130483415, array([2.55245656, 4.84323095, 6.87707099, 9.30756306]))
SG 5 (2.6311521705182743, 2.099901581615436, 2.7785588236568435, array([2.05910882, 3.86637231, 5.31558153, 6.83927399]))
SG 9 (2.5680830215965376, 2.0960798362761097, 2.6980692721954562, array([2.0217833 , 3.7813618 , 5.13339118, 6.61066092]))
SG 15 (2.748576673758376, 2.2799146493102724, 2.8970157112161985, array([2.21400556, 4.0632728 , 5.40703929, 6.80237898]))
SG 21  …[+989 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/flow_validate.py';s=open(p).read();s=s.replace(\"print('track time',time.time()-t)\",\"print('track time',time.time()-t)\\nwith open('/work/scratch/pairs.pkl','wb') as ff: pickle.dump(pairs,ff)\")
open(p,'w').write(s)
PY
python /work/scratch/flow_validate.py"
```
<details><summary>output</summary>

```
preload
loaded 747
200 217 1.3696985244750977
300 146 3.0297155380249023
400 188 5.155730485916138
500 116 7.133701801300049
600 243 9.220705032348633
track time 9.927741765975952
BASE (3.4452854856324406, 2.6007384864441914, 3.6104054885538175, array([2.55261837, 4.84321924, 6.87708461, 9.3075652 ]))
SG 3 (3.4460038394731534, 2.600739556156859, 3.6111209130483415, array([2.55245656, 4.84323095, 6.87707099, 9.30756306]))
SG 5 (2.6311521705182743, 2.099901581615436, 2.7785588236568435, array([2.05910882, 3.86637231, 5.31558153, 6.83927399]))
SG 9 (2.5680830215965376, 2.0960798362761097, 2.6980692721954562, array([2.0217833 , 3.7813618 , 5.13339118, 6.61066092]))
SG 15 (2.748576673758376, 2.2799146493102724, 2.8970157112161985, array([2.21400556, 4.0632728 , 5.40703929, 6.80237898]))
SG 21 ( …[+988 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/grid.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle
from scipy.signal import savgol_filter
pairs=pickle.load(open('/work/scratch/pairs.pkl','rb'));D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
def flowerr(Ts):
 es=[];ems=[]
 for j,(o,p,n) in pairs.items():
  pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];e=np.linalg.norm(uv-p,axis=1);es.append(np.median(e));ems.append(np.mean(e))
 return np.mean(es),np.median(es),np.mean(ems)
def ortho(M):
 U,s,Vt=np.linalg.svd(M);D=np.eye(3);D[2,2]=np.sign(np.linalg.det(U@Vt));return U@D@Vt
def blend(R1,R2,a):
 rv=Rot.from_matrix(R1.T@R2).as_rotvec();return R1@Rot.from_rotvec(a*rv).as_matrix()
def mergesmooth(alpha,ws):
 S=smooth_T(9,9);T=S.copy()
 for i in range(128,639):
  Rm=Rot.from_matrix(np.stack([FW[i][:3,:3],BW[i][:3,:3]])).mean().as_matrix();tm=(FW[i][:3,3]+BW[i][:3,3])/2
  T[i,:3,:3]=blend(S[i,:3,:3],Rm,alpha);T[i,:3,3]=(1-alpha)*S[i,:3,3]+alpha*tm
 if ws:
  for a,b in [(128,639)]:
   for ii in range(3):
    for jj in range(3):T[a:b,ii,jj]=savgol_filter(T[a:b,ii,jj],ws,2,mode='interp')
    T[a:b,ii,3]=savgol_filter(T[a:b,ii,3],ws,2,mode='interp')
   for i in range(a,b):T[i,:3,:3]=ortho(T[i,:3,:3])
 return T
res=[]
for alpha in [0,.2,.4,.5,.6,.8,1.0]:
 for ws in [0,3,5,7,9]:
  T=mergesmooth(alpha,ws);e=flowerr(T);res.append((e[0],e[1],e[2 …[+187 chars]
```
<details><summary>output</summary>

```
0 0 (2.5680830215965376, 2.0960798362761097, 2.6980692721954562)
0 3 (2.5680829757407797, 2.0960803871378855, 2.698069319656881)
0 5 (2.5455604298468226, 2.094548054343635, 2.677763542339583)
0 7 (2.526288493623875, 2.074064083912264, 2.6637340180857194)
0 9 (2.562030309656712, 2.1205988296531717, 2.699603849009493)
0.2 0 (2.442645859309101, 2.010967576360933, 2.579143864647751)
0.2 3 (2.442645848692475, 2.0109670241809323, 2.5791438406773626)
0.2 5 (2.5120662802555387, 2.07353493913441, 2.6487176153663343)
0.2 7 (2.495499448968581, 2.0389375241139183, 2.6326769739527704)
0.2 9 (2.5266384869467675, 2.060275508229413, 2.6662109761657478)
0.4 0 (2.340403650869953, 1.9185438132718546, 2.4826789135883365)
0.4 3 (2.3404036319422232, 1.9185440963769298, 2.4826788388966916)
0.4 5 (2.4878330233350 …[+2531 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_candidates.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
from scipy.spatial.transform import Rotation as Rot
D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
def merge(alpha=1):
 S=smooth_T(9,9);T=S.copy()
 for i in range(128,639):
  Rm=Rot.from_matrix(np.stack([FW[i][:3,:3],BW[i][:3,:3]])).mean().as_matrix();tm=(FW[i][:3,3]+BW[i][:3,3])/2
  rv=Rot.from_matrix(S[i,:3,:3].T@Rm).as_rotvec();T[i,:3,:3]=S[i,:3,:3]@Rot.from_rotvec(alpha*rv).as_matrix();T[i,:3,3]=(1-alpha)*S[i,:3,3]+alpha*tm
 return T
T=merge(1)
# save static optimization? later
for name,Ts in [('base',B),('sg9',smooth_T(9,9)),('merge',T)]:
 vals=[];angs=[];dts=[]
 for i in range(747):
  vals.append(metric(Ts[i],i))
  if i:angs.append(np.degrees(Rot.from_matrix(Ts[i-1,:3,:3].T@Ts[i,:3,:3]).magnitude()));dts.append(np.linalg.norm(Ts[i,:3,3]-Ts[i-1,:3,3]))
 vals=np.array(vals)
 print(name,'mask mean',np.round(np.mean(vals,0),4),'dyn',np.round(np.mean(vals[128:639],0),4),'ang',np.mean(angs),np.quantile(angs,.95),ang.max(),'dt',np.mean(dts),np.quantile(dts,.95),np.max(dts))
np.save('/work/scratch/T_merge.npy',T)
PY
python /work/scratch/eval_candidates.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_candidates.py", line 18, in <module>
    print(name,'mask mean',np.round(np.mean(vals,0),4),'dyn',np.round(np.mean(vals[128:639],0),4),'ang',np.mean(angs),np.quantile(angs,.95),ang.max(),'dt',np.mean(dts),np.quantile(dts,.95),np.max(dts))
                                                                                                                                            ^^^
NameError: name 'ang' is not defined. Did you mean: 'angs'?

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/ang.max()/np.max(angs)/' /work/scratch/eval_candidates.py
python /work/scratch/eval_candidates.py"
```
<details><summary>output</summary>

```
base mask mean [0.5331 0.3711 0.3756] dyn [0.5002 0.3419 0.347 ] ang 4.0990069329932295 14.687151301118227 61.52889048311231 dt 0.020550922 0.05991602037101984 0.119495384
sg9 mask mean [0.5425 0.3802 0.3844] dyn [0.5033 0.3442 0.3492] ang 1.940398901068455 6.144151714169297 20.426708293836402 dt 0.010132192 0.029231357853859663 0.043512505
merge mask mean [0.5448 0.3824 0.3868] dyn [0.5067 0.3474 0.3527] ang 2.9276003386405804 9.997286639281047 31.21510575330021 dt 0.018427463 0.05578584782779217 0.12801296

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import numpy as np
D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
from scipy.spatial.transform import Rotation as Rot
S=smooth_T(9,9);T=S.copy()
for i in range(128,639):
 Rm=Rot.from_matrix(np.stack([FW[i][:3,:3],BW[i][:3,:3]])).mean().as_matrix();tm=(FW[i][:3,3]+BW[i][:3,3])/2
 rv=Rot.from_matrix(S[i,:3,:3].T@Rm).as_rotvec();T[i,:3,:3]=S[i,:3,:3]@Rot.from_rotvec(rv).as_matrix();T[i,:3,3]=tm
a,Q,th=decompose(T[:,:3,:3].astype(float))
dt=np.degrees(np.diff(th));
print('theta vel q',np.quantile(dt,[0,.01,.1,.5,.9,.95,.99,1]))
for i in range(128,639):
 if abs(dt[i])>12: print(i,'vel',round(dt[i],1),'theta',round(np.degrees(th[i]),1))
PY"
```
<details><summary>output</summary>

```
theta vel q [-43.97593743 -18.64162384  -5.08334807   0.           4.05780912
   5.81120568  11.40676032  22.96322041]
243 vel -12.7 theta 26.6
244 vel 13.3 theta 13.9
245 vel -14.6 theta 27.1
255 vel -15.3 theta 11.3
258 vel -12.2 theta -21.4
261 vel -20.8 theta -40.4
263 vel -21.5 theta -62.9
272 vel -27.1 theta -123.7
273 vel 20.7 theta -150.7
274 vel -20.9 theta -130.0
275 vel -12.6 theta -150.9
280 vel -12.6 theta -171.1
288 vel -22.2 theta -188.4
291 vel 23.0 theta -211.8
292 vel -21.4 theta -188.9
293 vel 13.6 theta -210.3
294 vel -12.4 theta -196.7
296 vel -14.2 theta -204.7
303 vel -44.0 theta -219.6
305 vel -37.0 theta -265.9
307 vel -15.6 theta -310.0
311 vel 17.7 theta -326.5
350 vel 14.1 theta -295.3
436 vel -16.1 theta -263.5
610 vel 14.9 theta -262.6

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/grid2.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle
from scipy.signal import savgol_filter,medfilt
pairs=pickle.load(open('/work/scratch/pairs.pkl','rb'));D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
def flowerr(Ts):
 es=[];ems=[]
 for j,(o,p,n) in pairs.items():
  pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];e=np.linalg.norm(uv-p,axis=1);es.append(np.median(e));ems.append(np.mean(e))
 return np.mean(es),np.median(es),np.mean(ems)
def build(wt=0,ts=0,robust=0):
 S=smooth_T(9,9);T=S.copy()
 for i in range(128,639):
  Rm=Rot.from_matrix(np.stack([FW[i][:3,:3],BW[i][:3,:3]])).mean().as_matrix();tm=(FW[i][:3,3]+BW[i][:3,3])/2
  rv=Rot.from_matrix(S[i,:3,:3].T@Rm).as_rotvec();T[i,:3,:3]=S[i,:3,:3]@Rot.from_rotvec(rv).as_matrix();T[i,:3,3]=tm
 a,Q,th=decompose(T[:,:3,:3].astype(float)); orig=th.copy()
 if robust:
  for i in range(128,639):
   lo=max(128,i-robust);hi=min(639,i+robust+1);med=np.median(orig[lo:hi]);
   if abs(orig[i]-med)>np.deg2rad(12):th[i]=med
 if wt:
  th[128:639]=savgol_filter(th[128:639],wt,2,mode='interp')
 Qs=Qfromz(a[128:639])
 for ii,i in enumerate(range(128,639)):
  c,s=np.cos(th[i]),np.sin(th[i]);T[i,:3,:3]=Qs[ii]@np.array([[c,-s,0],[s,c,0],[0,0,1.]])
 if ts:
  for j in range(3):T[128:639,j,3]=savgol_filter(T[128:639,j,3],ts,2,mode='interp')
 return T
res=[] …[+426 chars]
```
<details><summary>output</summary>

```
theta 0 ts 0 flow [2.199 1.791 2.353] dice 0.5067
theta 0 ts 5 flow [2.463 1.982 2.601] dice 0.5072
theta 0 ts 9 flow [2.433 1.979 2.574] dice 0.5075
theta 3 ts 0 flow [2.199 1.791 2.353] dice 0.5067
theta 3 ts 5 flow [2.463 1.982 2.601] dice 0.5072
theta 3 ts 9 flow [2.433 1.979 2.574] dice 0.5075
theta 5 ts 0 flow [2.229 1.828 2.383] dice 0.5067
theta 5 ts 5 flow [2.473 1.964 2.62 ] dice 0.5072
theta 5 ts 9 flow [2.449 1.956 2.593] dice 0.5073
theta 7 ts 0 flow [2.227 1.84  2.381] dice 0.5067
theta 7 ts 5 flow [2.482 1.986 2.622] dice 0.5072
theta 7 ts 9 flow [2.451 1.963 2.593] dice 0.5074
theta 9 ts 0 flow [2.235 1.838 2.387] dice 0.5067
theta 9 ts 5 flow [2.486 1.976 2.626] dice 0.5072
theta 9 ts 9 flow [2.456 1.988 2.597] dice 0.5074
theta 11 ts 0 flow [2.244 1.847 2.396] dice 0.5068 …[+908 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/multistep_test.py <<'PY'
exec(open('/work/scratch/propagate2.py').read().split(\"if __name__\")[0])
# redefine track with higher pyramid
def track_local(i,j,pts):
 p0=(pts-np.array([x0,y0])).reshape(-1,1,2).astype(np.float32)
 p1,st,err=cv2.calcOpticalFlowPyrLK(G[i],G[j],p0,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
 p0b,stb,errb=cv2.calcOpticalFlowPyrLK(G[j],G[i],p1,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
 fb=np.linalg.norm(p0-p0b,axis=2).ravel();ok=(st.ravel()>0)&(stb.ravel()>0)&(fb<1.0)
 pfull=p1.reshape(-1,2)+np.array([x0,y0],dtype=np.float32);xi=np.clip(np.round(pfull[:,0]).astype(int),0,W-1);yi=np.clip(np.round(pfull[:,1]).astype(int),0,H-1);ok&=obj[j][yi,xi]
 return pfull,ok,fb
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
for tgt in [262,263,272,273,274,288,304,305,306,308,309]:
 vals=[]
 for d in [-5,-4,-3,-2,-1,1,2,3,4,5]:
  src=tgt+d
  T,q=estimate(src,tgt,B[src],400)
  if T is None:print('fail',src,tgt,q);continue
  a=T[:3,:3];Q=Qfromz(a[:,2][None])[0];th=np.degrees(Rot.from_matrix(Q.T@a).as_rotvec()[2]);vals.append((d,round(th,1),q['n'],round(q['err'],2)))
 print('TGT',tgt,vals)
PY
python /work/scratch/multistep_test.py"
```
<details><summary>output</summary>

```
preload
loaded 747
TGT 262 [(-5, -9.6, 243, 1.19), (-4, -25.4, 170, 1.02), (-3, -32.4, 193, 0.49), (-2, -40.3, 230, 0.22), (-1, -40.3, 225, 0.12), (1, -82.4, 205, 0.08), (2, -86.1, 210, 0.18), (3, -86.0, 206, 0.16), (4, -93.5, 185, 0.17), (5, -107.5, 221, 0.19)]
TGT 263 [(-5, -25.3, 172, 0.94), (-4, -32.4, 192, 0.6), (-3, -40.3, 229, 0.31), (-2, -40.2, 225, 0.19), (-1, -40.3, 211, 0.08), (1, -86.2, 212, 0.11), (2, -86.1, 205, 0.11), (3, -93.6, 187, 0.13), (4, -107.7, 218, 0.19), (5, -116.8, 236, 0.42)]
TGT 272 [(-5, -109.9, 193, 1.34), (-4, -119.4, 202, 0.97), (-3, -129.0, 192, 1.04), (-2, -128.7, 192, 1.0), (-1, -139.5, 209, 0.41), (1, -108.3, 224, 0.42), (2, -161.0, 184, 0.53), (3, -150.3, 166, 0.64), (4, -138.2, 149, 0.92), (5, -174.4, 167, 1.04)]
TGT 273 [(-5, -121.0, 177, 1.29), (-4,  …[+1875 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '1,260p' /workspace/v2d_foundation_pose/lib/render_overlay.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Render mesh overlay on video frames using per-frame object-to-camera poses.

Inputs:
  --video_path              path to input video
  --poses_dir               directory of per-frame 4×4 matrix JSONs (T_cam_from_obj)
  --mesh_path               path to mesh .obj file
  --camera_intrinsics_path  JSON with {fx, fy, cx, cy, width, height}
  --output_dir              output directory for overlay frames (%06d.jpg)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh

# CV camera → OpenGL camera: flip Y and Z axes
_T_CV_TO_GL = np. …[+2430 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '1,260p' /workspace/v2d_mesh/lib/mesh_render_image.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import pyglet
pyglet.options['headless'] = True

import numpy as np

from v2d.common.datatypes import CameraIntrinsics, Image
from v2d.mesh.lib.mesh import Mesh
from v2d.mesh.lib.mesh_render_depth import _CV_TO_GL


def mesh_render_image(
    mesh: Mesh,
    intrinsics: CameraIntrinsics,
    background: Image | None = None,
    renderer=None,
) -> Image:
    """
    Render an RGB image of the mesh from a camera at the origin.

    The camera convention matches mesh_render_depth (OpenCV: looking along +Z, Y down).
    A directional light is placed at the camera position so front-facing surfaces a …[+1658 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'grep -R "def .*render" -n /workspace/v2d_cari4d/lib | head -50'
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/Utils.py:120:def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
/workspace/v2d_cari4d/lib/cari4d/Utils.py:228:def nvdiff_rasterize(glctx, mesh_tensors, pos_clip, render_size):
/workspace/v2d_cari4d/lib/cari4d/Utils.py:245:def nvdiff_color_depth_render(Krois, glctx, mesh_tensors, render_size, pts_cam, depth_only=False):
/workspace/v2d_cari4d/lib/cari4d/behave_data/utils.py:195:def get_render_template_path_from_seq(seq_name):
/workspace/v2d_cari4d/lib/cari4d/learning/datasets/mhr_rank_local …[+11513 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/pyrender_test.py <<'PY'
import os;os.environ['PYOPENGL_PLATFORM']='egl'
import pyrender,trimesh,numpy as np,cv2,json,h5py
from scipy.spatial.transform import Rotation as Rot
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh']
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
# pyrender material perhaps
pm=pyrender.Mesh.from_trimesh(mesh,smooth=False)
x0,y0,x1,y1=650,500,950,1000;W=x1-x0;H=y1-y0
renderer=pyrender.OffscreenRenderer(W,H)
def rend(T):
 Tcm=Tcw@T;scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.5,.5,.5]);scene.add(pm,pose=Tcm)
 light=pyrender.DirectionalLight(color=np.ones(3),intensity=3.0);scene.add(light,pose=np.diag([1.,-1.,-1.,1.]))
 intr=pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2]-x0,cy=K[1,2]-y0,znear=.01,zfar=100)
 scene.add(intr,pose=np.diag([1.,-1.,-1.,1.]))
 c,d=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);return c,d
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);_,im=cap.read();tgt=im[y0:y1,x0:x1]
c,d=rend(B[0]);a=c[:,:,3:4].astype(float)/255
ov=(c[:,:,:3]*a+tgt*(1-a)).astype(np.uint8)
cv2.imwrite('/work/scratch/pyrender0.jpg',np.concatenate([tgt,ov,c[:,:,:3]],1))
print('mask',((c[:,:,3]>0)).sum(),'tex mean',c[c[:,:,3]>0,:3].mean(0),'tgt',tgt[obj].mean(0) if False else '')
renderer.d …[+49 chars]
```
<details><summary>output</summary>

```
mask 8742 tex mean [78.40082361 57.28746282 55.10318005] tgt 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
# modify photo_eval copy vflip
s=open('/work/scratch/photo_eval.py').read().split(\"for i in [\")[0]
s=s.replace(\"uvn=torch.stack([uv[...,0]*2-1,uv[...,1]*2-1],-1)\",\"uvn=torch.stack([uv[...,0]*2-1,1-uv[...,1]*2],-1)\")
ns={};exec(s,ns)
import torch,cv2,numpy as np
B=ns['B'];i=0
with torch.no_grad():m,c=ns['render_col'](torch.tensor(B[i],device=ns['dev']))
c=(c.clamp(0,1).cpu().numpy()*255).astype(np.uint8);cap=cv2.VideoCapture('/work/inputs/video.mp4');_,im=cap.read();crop=im[ns['y0']:ns['y1'],ns['x0']:ns['x1']]
vis=crop.copy();vis[m.cpu().numpy()]=(.4*crop[m.cpu().numpy()]+.6*c[m.cpu().numpy()]).astype(np.uint8)
cv2.imwrite('/work/scratch/vflip.jpg',np.concatenate([crop,vis,c],1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np,trimesh,h5py,json
# use pyrender result recompute save raw via script import
ns={};exec(open('/work/scratch/pyrender_test.py').read().split(\"cap=cv2\")[0],ns)
c,d=ns['rend'](ns['B'][0]);mask=c[:,:,3]>0
cap=cv2.VideoCapture('/work/inputs/video.mp4');_,im=cap.read();tgt=im[ns['y0']:ns['y1'],ns['x0']:ns['x1']][:,:,::-1]
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][0,ns['y0']:ns['y1'],ns['x0']:ns['x1']]
ov=mask&o
print('render mean rgb',c[ov,:3].mean(0),'target',tgt[ov].mean(0))
print('render median',np.median(c[ov,:3],0),'target',np.median(tgt[ov],0))
print('render bgr',c[ov,:3][:,::-1].mean(0))
PY"
```
<details><summary>output</summary>

```
render mean rgb [79.23642764 55.57216972 52.16175653] target [57.42708912 37.49601631 36.42171577]
render median [72. 54. 51.] target [48. 32. 32.]
render bgr [52.16175653 55.57216972 79.23642764]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/spin_eval.py <<'PY'
os='none'
import os,json,cv2,h5py,numpy as np,trimesh
os.environ['PYOPENGL_PLATFORM']='egl'
import pyrender
from scipy.spatial.transform import Rotation as Rot
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh']
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();pm=pyrender.Mesh.from_trimesh(mesh,smooth=False)
x0,y0,x1,y1=650,500,950,1000;W=x1-x0;H=y1-y0;renderer=pyrender.OffscreenRenderer(W,H)
cen=np.array(mesh.bounds.mean(0));noise=np.eye(4)
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:];per=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
def getim(i):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();return im[y0:y1,x0:x1]
def rend(T):
 scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[.5,.5,.5]);scene.add(pm,pose=Tcw@T);scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=3.0),pose=np.diag([1.,-1.,-1.,1.]));scene.add(pyrender.IntrinsicsCamera(fx=K[0,0],fy=K[1,1],cx=K[0,2]-x0,cy=K[1,2]-y0,znear=.01,zfar=100),pose=np.diag([1.,-1.,-1.,1.]));c,d=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);return c
def corr(a,b,m):
 a=a[m].astype(float);b=b[m].astype(float);a-=a.mean();b-=b.mean();return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-9))
def score(i,T):
 tgt=getim(i)[:,:,::-1].a …[+1262 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/spin_eval.py", line 42, in <module>
    vals.append((ang,*score(i,T)))
                      ^^^^^^^^^^
  File "/work/scratch/spin_eval.py", line 32, in score
    edge=corr(grad(tg),grad(cr),m)
              ^^^^^^^^
  File "/work/scratch/spin_eval.py", line 31, in grad
    def grad(x):gx=cv2.Sobel(x.astype(np.float32),cv2.CV_32F,1,0);gy=cv2.Sobel(x,cv2.CV_32F,0,1);return np.sqrt(gx*gx+gy*gy)
                                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgproc/src/filter.simd.hpp:3096: error: (-213:The function/feature is not implemented) Unsupported combination of buffer format (=6), and destination format (=5) in function 'getLinearColumnFilter'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'sed -i "s/gy=cv2.Sobel(x,cv2.CV_32F/gy=cv2.Sobel(x.astype(np.float32),cv2.CV_32F/" /work/scratch/spin_eval.py
python /work/scratch/spin_eval.py'
```
<details><summary>output</summary>

```
FRAME 0 best (-40, 0.48450830442639653, 0.2543837248753157, 0.3160347510865537) base (0, 0.7062055075166672, 0.1502847039119548, 0.05560069462668586)
  [(-180, 0.1, 0.24, -0.02), (-160, 0.14, 0.2, -0.11), (-140, 0.14, 0.14, -0.11), (-120, 0.21, 0.11, -0.15), (-100, 0.23, 0.03, -0.28), (-80, 0.3, 0.11, 0.01), (-60, 0.33, 0.24, 0.04), (-40, 0.48, 0.25, 0.32), (-20, 0.71, 0.14, 0.14), (0, 0.71, 0.15, 0.06), (20, 0.49, 0.12, 0.05), (40, 0.31, 0.13, -0.01), (60, 0.17, 0.16, -0.14), (80, 0.27, 0.15, -0.05), (100, 0.43, 0.09, -0.1), (120, 0.44, 0.05, -0.01), (140, 0.26, 0.14, 0.02), (160, 0.07, 0.21, -0.0), (180, 0.1, 0.24, -0.02)]
FRAME 26 best (-20, 0.6788965762267646, 0.19504904873250703, 0.3803967016520091) base (0, 0.7792641471228933, 0.14197240346322107, 0.2993320621055244)
  [(-180, 0.05,  …[+7530 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/nv_spin_test.py <<'PY'
ns={};exec(open('/work/scratch/photo_eval.py').read().split(\"for i in [\")[0],ns)
import torch,cv2,numpy as np
from scipy.spatial.transform import Rotation as Rot
B=ns['B'];mesh=ns['mesh'];cen=torch.tensor(mesh.bounds.mean(0),dtype=torch.float32,device=ns['dev'])
def corr(a,b,m):
 a=a[m].astype(float);b=b[m].astype(float);a-=a.mean();b-=b.mean();return float((a*b).sum()/(np.sqrt((a*a).sum()*(b*b).sum())+1e-9))
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,26,100,180,240,280,300,400,500,600,656,700]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();tgt=im[ns['y0']:ns['y1'],ns['x0']:ns['x1']][:,:,::-1].astype(float);tr=tgt[:,:,0]-(tgt[:,:,1]+tgt[:,:,2])/2
 vals=[]
 for ang in range(-180,181,20):
  M=torch.eye(4,device=ns['dev']);M[:3,:3]=torch.tensor(Rot.from_rotvec([0,0,np.deg2rad(ang)]).as_matrix(),dtype=torch.float32,device=ns['dev']);M[:3,3]=cen-M[:3,:3]@cen
  with torch.no_grad():mask,c=ns['render_col'](torch.tensor(B[i],dtype=torch.float32,device=ns['dev'])@M)
  m=(mask.cpu().numpy()&ns['obj'][i,ns['y0']:ns['y1'],ns['x0']:ns['x1']]&~ns['per'][i,ns['y0']:ns['y1'],ns['x0']:ns['x1']]);cc=(c.cpu().numpy()[:,:,0]-(c.cpu().numpy()[:,:,1]+c.cpu().numpy()[:,:,2])/2)
  vals.append((ang,corr(tr,cc,m)))
 print(i,'best',max(vals,key=lambda x:x[1]),'csv',[(a,round(v,2)) for a,v in vals])
PY
python /work/scratch/nv_spin_test.py"
```
<details><summary>output</summary>

```
0 best (-20, 0.23168566649974462) csv [(-180, -0.13), (-160, -0.13), (-140, 0.13), (-120, 0.11), (-100, -0.05), (-80, -0.16), (-60, -0.04), (-40, 0.18), (-20, 0.23), (0, 0.14), (20, 0.03), (40, -0.03), (60, -0.09), (80, -0.15), (100, -0.08), (120, -0.02), (140, -0.1), (160, -0.12), (180, -0.13)]
26 best (-20, 0.22407610475181974) csv [(-180, -0.16), (-160, -0.17), (-140, 0.11), (-120, 0.12), (-100, 0.01), (-80, -0.08), (-60, -0.06), (-40, 0.09), (-20, 0.22), (0, 0.19), (20, 0.07), (40, 0.0), (60, -0.09), (80, -0.14), (100, -0.06), (120, -0.02), (140, -0.1), (160, -0.15), (180, -0.16)]
100 best (-140, 0.21441505118525628) csv [(-180, -0.15), (-160, -0.04), (-140, 0.21), (-120, 0.08), (-100, -0.07), (-80, -0.12), (-60, 0.02), (-40, 0.19), (-20, 0.19), (0, 0.11), (20, 0.02), (40, -0.03), (60, …[+2765 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
from scipy.spatial.transform import Rotation as Rot
D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw'];S=smooth_T(9,9);T=S.copy()
for i in range(128,639):
 Rm=Rot.from_matrix(np.stack([FW[i][:3,:3],BW[i][:3,:3]])).mean().as_matrix();tm=(FW[i][:3,3]+BW[i][:3,3])/2;rv=Rot.from_matrix(S[i,:3,:3].T@Rm).as_rotvec();T[i,:3,:3]=S[i,:3,:3]@Rot.from_rotvec(rv).as_matrix();T[i,:3,3]=tm
def dec(Rs):
 a=Rs[:,:,2];Q=Qfromz(a);return np.unwrap(np.array([Rot.from_matrix(Q[i].T@Rs[i]).as_rotvec()[2] for i in range(len(Rs))]))
tb=dec(B[:,:3,:3].astype(float));tm=dec(T[:,:3,:3].astype(float));
for i in range(0,747,20):print(i,round(np.degrees(tb[i]),1),round(np.degrees(tm[i]),1),round(np.degrees((tm[i]-tb[i]+np.pi)%(2*np.pi)-np.pi),1))
PY"
```
<details><summary>output</summary>

```
0 68.4 83.9 15.5
20 58.9 83.9 25.0
40 68.1 83.9 15.8
60 73.6 83.9 10.3
80 73.6 83.9 10.3
100 73.6 83.9 10.3
120 72.4 83.9 11.5
140 77.0 83.4 6.4
160 85.5 87.6 2.2
180 46.3 47.7 1.4
200 19.9 18.5 -1.3
220 35.0 30.2 -4.8
240 43.7 38.2 -5.5
260 -40.4 -36.3 4.1
280 -176.2 -171.1 5.1
300 -209.2 -215.8 -6.6
320 -303.5 -298.1 5.4
340 -264.7 -270.4 -5.8
360 -272.9 -284.5 -11.5
380 -318.4 -324.2 -5.8
400 -310.5 -310.4 0.1
420 -268.3 -266.3 2.0
440 -265.8 -279.3 -13.5
460 -247.0 -244.6 2.4
480 -253.0 -256.1 -3.1
500 -308.8 -307.0 1.8
520 -313.5 -316.2 -2.7
540 -343.5 -343.1 0.4
560 -338.1 -338.3 -0.2
580 -299.4 -295.7 3.7
600 -246.5 -245.7 0.8
620 -251.4 -249.7 1.7
640 -247.8 -249.1 -1.3
660 -249.1 -249.1 0.0
680 -249.1 -249.1 0.0
700 -249.1 -249.1 0.0
720 -249.1 -249.1 0.0
740 -249.1 -249.1 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/perframe_flow.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle
from scipy.spatial.transform import Rotation as Rot
pairs=pickle.load(open('/work/scratch/pairs.pkl','rb'));D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw'];S=smooth_T(9,9);M=S.copy()
for i in range(128,639):
 Rm=Rot.from_matrix(np.stack([FW[i][:3,:3],BW[i][:3,:3]])).mean().as_matrix();tm=(FW[i][:3,3]+BW[i][:3,3])/2;rv=Rot.from_matrix(S[i,:3,:3].T@Rm).as_rotvec();M[i,:3,:3]=S[i,:3,:3]@Rot.from_rotvec(rv).as_matrix();M[i,:3,3]=tm
def pe(Ts,j):
 o,p,n=pairs[j];pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];e=np.linalg.norm(uv-p,axis=1);return np.median(e),np.mean(e)
for j in range(128,640):
 eb=pe(B,j);ef=pe(FW,j);ew=pe(BW,j);em=pe(M,j);ess=pe(S,j)
 best=np.argmin([eb[0],ef[0],ew[0],em[0],ess[0]])
 if j%5==0:print(j,'B',round(eb[0],1),'F',round(ef[0],1),'W',round(ew[0],1),'M',round(em[0],1),'S',round(ess[0],1),'best','BFWMS'[best])
PY
python /work/scratch/perframe_flow.py"
```
<details><summary>output</summary>

```
130 B 2.1 F 0.6 W 1.4 M 1.0 S 1.3 best F
135 B 5.6 F 0.4 W 1.6 M 0.9 S 1.1 best F
140 B 6.6 F 0.4 W 3.2 M 1.8 S 2.4 best F
145 B 2.9 F 0.3 W 5.4 M 2.8 S 2.3 best F
150 B 3.9 F 0.4 W 5.8 M 3.0 S 2.4 best F
155 B 3.8 F 0.2 W 3.0 M 1.5 S 1.6 best F
160 B 1.6 F 0.3 W 7.6 M 3.8 S 3.3 best F
165 B 1.2 F 0.3 W 4.3 M 2.2 S 0.6 best F
170 B 2.5 F 0.8 W 3.9 M 1.9 S 3.0 best F
175 B 2.2 F 0.8 W 4.7 M 2.6 S 2.6 best F
180 B 1.0 F 0.4 W 1.9 M 1.1 S 0.9 best F
185 B 1.0 F 0.3 W 0.9 M 0.5 S 0.6 best F
190 B 2.4 F 0.3 W 1.5 M 0.8 S 1.2 best F
195 B 2.4 F 0.2 W 1.0 M 0.5 S 1.7 best F
200 B 2.9 F 0.3 W 2.8 M 1.4 S 2.5 best F
205 B 2.5 F 0.5 W 1.9 M 1.0 S 1.9 best F
210 B 2.2 F 0.9 W 3.5 M 2.0 S 2.5 best F
215 B 2.4 F 1.3 W 2.1 M 1.3 S 1.5 best M
220 B 3.5 F 0.4 W 3.4 M 1.7 S 2.6 best F
225 B 4.4 F 0.4 W 3.5 …[+3391 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/val_split.py <<'PY'
exec(open('/work/scratch/propagate2.py').read().split(\"if __name__\")[0])
import pickle
from scipy.spatial.transform import Rotation as Rot
FWtr={};tests={}
for i in range(127,639):
 pts,xyz,tri=visible_points_crop(i,B[i],500);p1,ok,fb=track_local(i,i+1,pts);o=(np.linalg.inv(B[i])@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3];o=o[ok];p1=p1[ok]
 n=len(o);rng=np.random.default_rng(9000+i);idx=rng.permutation(n);ntr=int(.6*n);tr=idx[:ntr];te=idx[ntr:]
 if ntr<12:FWtr[i+1]=B[i+1];tests[i+1]=(o,p1,te);continue
 try:
  okp,rv,tv,inx=cv2.solvePnPRansac(o[tr].astype(np.float32),p1[tr].astype(np.float32),K,None,reprojectionError=2.0,iterationsCount=100,confidence=.99,flags=cv2.SOLVEPNP_EPNP)
  if not okp:raise RuntimeError()
  rv,tv=cv2.solvePnPRefineLM(o[tr].astype(np.float32),p1[tr].astype(np.float32),K,None,rv,tv)
  R,_=cv2.Rodrigues(rv);Tc=np.eye(4);Tc[:3,:3]=R;Tc[:3,3]=tv.ravel();FWtr[i+1]=Twc@Tc
 except Exception:FWtr[i+1]=B[i+1]
 tests[i+1]=(o,p1,te)
 if i%100==0:print(i,n)
with open('/work/scratch/val_split.pkl','wb') as f:pickle.dump((FWtr,tests),f)
# compare test errors
D=np.load('/work/scratch/independent.npz',allow_pickle=True);BW=D['Bw'];S=smooth_T(9,9)
M=S.copy()
for j in range(129,639):
 Rm=Rot.from_matrix(np.stack([FWtr[j][:3,:3],BW[j][:3,:3]])).mean().as_matrix();tm=(FWtr[j][:3,3]+BW[j][:3,3])/2;rv=Rot.from_matrix(S[j,:3,:3].T@Rm).as_rotvec();M[j,:3,:3]=S[j,:3,:3]@Rot.from_rotvec(rv).as_matrix();M[j,:3,3]=tm
def er …[+424 chars]
```
<details><summary>output</summary>

```
preload
loaded 747
200 271
300 199
400 228
500 149
600 256
Traceback (most recent call last):
  File "/work/scratch/val_split.py", line 19, in <module>
    D=np.load('/work/scratch/independent.npz',allow_pickle=True);BW=D['Bw'];S=smooth_T(9,9)
                                                                              ^^^^^^^^
NameError: name 'smooth_T' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/val_split_eval.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle
from scipy.spatial.transform import Rotation as Rot
FWtr,tests=pickle.load(open('/work/scratch/val_split.pkl','rb'));D=np.load('/work/scratch/independent.npz',allow_pickle=True);BW=D['Bw'];S=smooth_T(9,9);M=S.copy()
for j in range(129,639):
 Rm=Rot.from_matrix(np.stack([FWtr[j][:3,:3],BW[j][:3,:3]])).mean().as_matrix();tm=(FWtr[j][:3,3]+BW[j][:3,3])/2;rv=Rot.from_matrix(S[j,:3,:3].T@Rm).as_rotvec();M[j,:3,:3]=S[j,:3,:3]@Rot.from_rotvec(rv).as_matrix();M[j,:3,3]=tm
def er(Ts,j):
 o,p,te=tests[j];pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];return np.median(np.linalg.norm(uv-p,axis=1)[te])
for name,Ts in [('B',B),('S',S),('FWtr',FWtr),('BW',BW),('M',M)]:
 es=[er(Ts,j) for j in range(129,639)];print(name,'test med mean',np.mean(es),np.median(es),'q',np.quantile(es,[.75,.9,.95]))
PY
python /work/scratch/val_split_eval.py"
```
<details><summary>output</summary>

```
B test med mean 3.4428433219812855 2.5944144303720544 q [4.2703786  6.52879565 8.55193922]
S test med mean 2.5586780137205523 2.106184454733374 q [3.2464848  4.71578006 6.06859059]
FWtr test med mean 0.49078739925169856 0.43576890813442615 q [0.61522194 0.90475091 1.13996699]
BW test med mean 4.260249120744096 3.481863030023339 q [5.57610445 8.23952574 9.86058627]
M test med mean 2.2052239877816087 1.8095724631276424 q [2.77945569 4.14693561 4.98619102]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/grid_val.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle
from scipy.spatial.transform import Rotation as Rot
FW,tests=pickle.load(open('/work/scratch/val_split.pkl','rb'));D=np.load('/work/scratch/independent.npz',allow_pickle=True);BW=D['Bw'];S=smooth_T(9,9)
def blend(R1,R2,a):return R1@Rot.from_rotvec(a*Rot.from_matrix(R1.T@R2).as_rotvec()).as_matrix()
def make(alpha,beta):
 T=S.copy()
 for j in range(129,639):
  Rm=blend(FW[j,:3,:3],BW[j,:3,:3],beta);tm=(1-beta)*FW[j,:3,3]+beta*BW[j,:3,3]
  T[j,:3,:3]=blend(S[j,:3,:3],Rm,alpha);T[j,:3,3]=(1-alpha)*S[j,:3,3]+alpha*tm
 return T
def er(Ts,j):
 o,p,te=tests[j];pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];return np.median(np.linalg.norm(uv-p,axis=1)[te])
res=[]
for alpha in [0,.25,.5,.75,1]:
 for beta in [0,.25,.5,.75,1]:
  T=make(alpha,beta);es=[er(T,j) for j in range(129,639)];print(alpha,beta,round(np.mean(es),3),round(np.median(es),3),np.round(np.quantile(es,[.9,.95]),2));res.append((np.mean(es),alpha,beta))
print('BEST',sorted(res)[:10])
np.save('/work/scratch/T_valbest.npy',make(*sorted(res)[0][1:]))
PY
python /work/scratch/grid_val.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/grid_val.py", line 17, in <module>
    T=make(alpha,beta);es=[er(T,j) for j in range(129,639)];print(alpha,beta,round(np.mean(es),3),round(np.median(es),3),np.round(np.quantile(es,[.9,.95]),2));res.append((np.mean(es),alpha,beta))
      ^^^^^^^^^^^^^^^^
  File "/work/scratch/grid_val.py", line 9, in make
    Rm=blend(FW[j,:3,:3],BW[j,:3,:3],beta);tm=(1-beta)*FW[j,:3,3]+beta*BW[j,:3,3]
             ~~^^^^^^^^^
TypeError: unhashable type: 'slice'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/FW\\[j,:3,:3\\]/FW[j][:3,:3]/g; s/FW\\[j,:3,3\\]/FW[j][:3,3]/g' /work/scratch/grid_val.py
python /work/scratch/grid_val.py"
```
<details><summary>output</summary>

```
0 0 2.559 2.106 [4.72 6.07]
0 0.25 2.559 2.106 [4.72 6.07]
0 0.5 2.559 2.106 [4.72 6.07]
0 0.75 2.559 2.106 [4.72 6.07]
0 1 2.559 2.106 [4.72 6.07]
0.25 0 1.957 1.61 [3.59 4.64]
0.25 0.25 2.18 1.807 [4.02 5.03]
0.25 0.5 2.411 1.99 [4.62 5.63]
0.25 0.75 2.645 2.175 [5.12 6.11]
0.25 1 2.886 2.371 [5.59 6.53]
0.5 0 1.366 1.15 [2.4  3.08]
0.5 0.25 1.816 1.52 [3.44 4.04]
0.5 0.5 2.296 1.877 [4.44 5.26]
0.5 0.75 2.794 2.307 [5.36 6.51]
0.5 1 3.299 2.701 [6.4  7.88]
0.75 0 0.825 0.734 [1.41 1.67]
0.75 0.25 1.483 1.224 [2.78 3.38]
0.75 0.5 2.23 1.81 [4.31 5.33]
0.75 0.75 2.994 2.401 [5.75 7.08]
0.75 1 3.762 3.013 [7.2  8.82]
1 0 0.491 0.436 [0.9  1.14]
1 0.25 1.205 1.023 [2.11 2.69]
1 0.5 2.205 1.81 [4.15 4.99]
1 0.75 3.227 2.634 [6.23 7.53]
1 1 4.26 3.482 [8.24 9.86]
BEST [(0.4907865723924082, 1, …[+289 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('/work/scratch/independent.npz',allow_pickle=True)
for name,T in [('base',B),('fw',D['Fw']),('bw',D['Bw']),('merge',np.load('/work/scratch/T_merge.npy'))]:
 ang=[];dt=[]
 for i in range(128,639):ang.append(np.degrees(Rot.from_matrix(T[i-1,:3,:3].T@T[i,:3,:3]).magnitude()));dt.append(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]))
 print(name,np.round([np.mean(ang),np.median(ang),np.quantile(ang,.95),np.max(ang),np.mean(dt),np.median(dt),np.quantile(dt,.95),np.max(dt)],2))
PY"
```
<details><summary>output</summary>

```
base [5.650e+00 3.600e+00 1.766e+01 6.153e+01 3.000e-02 2.000e-02 7.000e-02
 1.200e-01]
fw [6.1200e+00 3.9400e+00 1.7720e+01 1.1866e+02 4.0000e-02 3.0000e-02
 9.0000e-02 2.8600e+00]
bw [6.0400e+00 3.7400e+00 1.8140e+01 1.2432e+02 4.0000e-02 3.0000e-02
 9.0000e-02 2.8400e+00]
merge [4.270e+00 3.060e+00 1.102e+01 3.122e+01 3.000e-02 2.000e-02 6.000e-02
 1.300e-01]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/extend_ind.py <<'PY'
exec(open('/work/scratch/propagate2.py').read().split(\"if __name__\")[0])
D=np.load('/work/scratch/independent.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw'];qf=D['qf'];qb=D['qb'];FW=FW.copy();BW=BW.copy();qf=qf.copy();qb=qb.copy()
for j in range(100,128):
 T,q=estimate(j-1,j,B[j-1],500);FW[j]=T if T is not None else B[j];qf[j]=q
 T,q=estimate(j+1,j,B[j+1],500);BW[j]=T if T is not None else B[j];qb[j]=q
 print(j,q)
np.savez('/work/scratch/independent2.npz',Fw=FW,Bw=BW,qf=qf,qb=qb)
PY
python /work/scratch/extend_ind.py"
```
<details><summary>output</summary>

```
preload
loaded 747
100 {'n': 333, 'nin': 333, 'err': 0.031034098070890942, 'mean': 0.05297147189465239, 'fb': 0.0015689864521846175}
101 {'n': 333, 'nin': 332, 'err': 0.044828328172172395, 'mean': 0.08588836456311151, 'fb': 0.0018631350249052048}
102 {'n': 326, 'nin': 317, 'err': 0.10239129299232166, 'mean': 0.15481036728617073, 'fb': 0.0027335931081324816}
103 {'n': 322, 'nin': 322, 'err': 0.045035299967914534, 'mean': 0.07280760674001915, 'fb': 0.0016065260861068964}
104 {'n': 331, 'nin': 331, 'err': 0.052294709145068646, 'mean': 0.08005683658516814, 'fb': 0.001599846058525145}
105 {'n': 334, 'nin': 334, 'err': 0.04218101955660833, 'mean': 0.06039486126747416, 'fb': 0.0016293707303702831}
106 {'n': 308, 'nin': 308, 'err': 0.0582357869113205, 'mean': 0.09049723861527073, 'fb': 0.002052085 …[+2353 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh']
for a,b in [(26,656),(26,659),(0,700)]:print(a,b,np.degrees(R.from_matrix(B[a,:3,:3].T@B[b,:3,:3]).magnitude()),np.linalg.norm(B[a,:3,3]-B[b,:3,3]))
PY"
```
<details><summary>output</summary>

```
26 656 43.61831937392877 0.078572534
26 659 40.49195841373853 0.09098532
0 700 40.42264003205335 0.12052088

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/fuse_grid.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle
from scipy.spatial.transform import Rotation as Rot
FW,tests=pickle.load(open('/work/scratch/val_split.pkl','rb'));D=np.load('/work/scratch/independent.npz',allow_pickle=True);BW=D['Bw'];S=smooth_T(9,9)
def blend(R1,R2,a):return R1@Rot.from_rotvec(a*Rot.from_matrix(R1.T@R2).as_rotvec()).as_matrix()
def make(thr,w):
 T=S.copy()
 for j in range(129,639):
  af=np.degrees(Rot.from_matrix(S[j,:3,:3].T@FW[j][:3,:3]).magnitude());ab=np.degrees(Rot.from_matrix(S[j,:3,:3].T@BW[j][:3,:3]).magnitude());tf=np.linalg.norm(FW[j][:3,3]-S[j,:3,3]);tb=np.linalg.norm(BW[j][:3,3]-S[j,:3,3])
  C=None
  if af<thr and tf<.15:C=FW[j]
  elif ab<thr and tb<.15:C=BW[j]
  if C is not None:
   T[j,:3,:3]=blend(S[j,:3,:3],C[:3,:3],w);T[j,:3,3]=(1-w)*S[j,:3,3]+w*C[:3,3]
 return T
def er(Ts,j):
 o,p,te=tests[j];pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];return np.median(np.linalg.norm(uv-p,axis=1)[te])
for thr in [15,20,25,30,35,40,50,70,180]:
 for w in [.5,.75,1]:
  T=make(thr,w);es=[er(T,j) for j in range(129,639)];vals=[metric(T[i],i)[0] for i in range(129,639)];ang=[]
  for j in range(130,639):ang.append(np.degrees(Rot.from_matrix(T[j-1,:3,:3].T@T[j,:3,:3]).magnitude()))
  print(thr,w,'flow',round(np.mean(es),3),'mask',round(np.mean(vals),4),'ang95',round(np.quantile(ang,.95),1),'max' …[+57 chars]
```
<details><summary>output</summary>

```
15 0.5 flow 1.484 mask 0.5037 ang95 10.1 max 24.2
15 0.75 flow 1.0 mask 0.5034 ang95 12.9 max 28.5
15 1 flow 0.714 mask 0.5026 ang95 16.0 max 34.2
20 0.5 flow 1.436 mask 0.504 ang95 10.6 max 27.3
20 0.75 flow 0.93 mask 0.5039 ang95 13.9 max 36.1
20 1 flow 0.623 mask 0.5031 ang95 17.3 max 44.9
25 0.5 flow 1.405 mask 0.5038 ang95 10.3 max 36.9
25 0.75 flow 0.884 mask 0.5036 ang95 13.2 max 47.8
25 1 flow 0.564 mask 0.5028 ang95 17.0 max 58.7
30 0.5 flow 1.392 mask 0.5038 ang95 10.3 max 36.9
30 0.75 flow 0.864 mask 0.5037 ang95 13.2 max 47.8
30 1 flow 0.538 mask 0.5029 ang95 17.0 max 58.7
35 0.5 flow 1.38 mask 0.5039 ang95 10.3 max 40.6
35 0.75 flow 0.845 mask 0.5036 ang95 13.5 max 50.9
35 1 flow 0.514 mask 0.5027 ang95 17.5 max 61.2
40 0.5 flow 1.38 mask 0.5039 ang95 10.3 max 40.6
40 0.75 flo …[+535 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh']
for a,b in [(26,100),(26,110),(26,120),(127,128),(639,640),(639,656),(640,656)]:print(a,b,'rot',np.degrees(R.from_matrix(B[a,:3,:3].T@B[b,:3,:3]).magnitude()),'t',np.linalg.norm(B[a,:3,3]-B[b,:3,3]))
PY"
```
<details><summary>output</summary>

```
26 100 rot 6.031016371779109 t 0.017555932
26 110 rot 5.762299441737902 t 0.019840706
26 120 rot 4.996622801720571 t 0.01743706
127 128 rot 6.934178037407195 t 0.015402851
639 640 rot 5.650992715431081e-16 t 0.001293671
639 656 rot 2.29295640754635 t 0.008777417
640 656 rot 2.29295640754635 t 0.008153867

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/compose_final.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
D=np.load('/work/scratch/independent2.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']
# make dynamic smooth base from B over dynamic only, no static override
S=B.copy()
for j in range(3):
 S[100:640,j,3]=savgol_filter(B[100:640,j,3],9,2,mode='interp')
 for k in range(3):S[100:640,j,k]=savgol_filter(B[100:640,j,k],9,2,mode='interp')
for i in range(100,640):S[i,:3,:3]=ortho(S[i,:3,:3])
# static anchors
Tstart=B[26].copy();Tstart[:3,3]+=np.array([-.006,.006,.002])
cen=np.array(mesh.bounds.mean(0));Mspin=np.eye(4);Mspin[:3,:3]=Rot.from_rotvec([0,0,np.deg2rad(-20)]).as_matrix();Mspin[:3,3]=cen-Mspin[:3,:3]@cen
Tend=B[656].copy();Tend=Tend@Mspin;Tend[:3,3]+=np.array([-.004,-.003,-.039])
# fuse
T=np.tile(np.eye(4),(747,1,1));T[:100]=Tstart;T[640:]=Tend;T[100:640]=S[100:640]
def blend(R1,R2,a):return R1@Rot.from_rotvec(a*Rot.from_matrix(R1.T@R2).as_rotvec()).as_matrix()
accepted=0
for j in range(100,640):
 af=np.degrees(Rot.from_matrix(S[j,:3,:3].T@FW[j,:3,:3]).magnitude());ab=np.degrees(Rot.from_matrix(S[j,:3,:3].T@BW[j,:3,:3]).magnitude());tf=np.linalg.norm(FW[j][:3,3]-S[j,:3,3]);tb=np.linalg.norm(BW[j][:3,3]-S[j,:3,3])
 C=None
 if af<35 and tf<.15:C=FW[j]
 elif ab<35 and tb<.15:C=BW[j]
 if C is not None:T[j]=C;accepted+=1
print('accepted',accepted)
# robust tem …[+1383 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/compose_final.py", line 10, in <module>
    for i in range(100,640):S[i,:3,:3]=ortho(S[i,:3,:3])
                                       ^^^^^
NameError: name 'ortho' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/compose_final.py';s=open(p).read();s=s.replace(\"D=np.load('/work/scratch/independent2.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']\", \"D=np.load('/work/scratch/independent2.npz',allow_pickle=True);FW=D['Fw'];BW=D['Bw']\\ndef ortho(M):\\n U,s,Vt=np.linalg.svd(M);D=np.eye(3);D[2,2]=np.sign(np.linalg.det(U@Vt));return U@D@Vt\")
open(p,'w').write(s)
PY
python /work/scratch/compose_final.py"
```
<details><summary>output</summary>

```
accepted 537
robust 0 5
robust 1 0
robust 2 0
overall [0.55274566 0.39252511 0.39641158] phases [(0, 100, array([0.72907586, 0.57367134, 0.57404831])), (100, 640, array([0.51077561, 0.35188231, 0.35715838])), (640, 747, array([0.59976253, 0.42834277, 0.42849618]))]
smooth ang 4.031967248651872 2.462003725137392 14.034679518134551 58.358536873515945 dt 0.023581991274512104 0.07515355408377963 0.14002454818640245

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle,numpy as np
FW,tests=pickle.load(open('/work/scratch/val_split.pkl','rb'));D=np.load('/work/scratch/independent.npz',allow_pickle=True);S=smooth_T(9,9);M=np.load('/work/scratch/T_merge.npy');F=np.load('/work/scratch/T_final_candidate.npy')
def er(Ts,j):
 o,p,te=tests[j];pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];return np.median(np.linalg.norm(uv-p,axis=1)[te])
for name,T in [('base',B),('sg',S),('merge',M),('final',F)]:
 es=[er(T,j) for j in range(129,639)];print(name,np.mean(es),np.median(es),np.quantile(es,[.75,.9,.95]))
PY"
```
<details><summary>output</summary>

```
base 3.4428433219812855 2.5944144303720544 [4.2703786  6.52879565 8.55193922]
sg 2.5586780137205523 2.106184454733374 [3.2464848  4.71578006 6.06859059]
merge 2.2043391831188197 1.8112186661458254 [2.8227164  4.13915343 5.08429157]
final 0.548527377247926 0.38442778634399544 [0.5467081  0.81255088 1.00785972]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/photo_final.py <<'PY'
exec(open('/work/scratch/spin_eval.py').read().split(\"if __name__\")[0])
import numpy as np
from scipy.spatial.transform import Rotation as Rot
Ts={'base':B,'sg':smooth_T(9,9) if False else B,'merge':np.load('/work/scratch/T_merge.npy'),'final':np.load('/work/scratch/T_final_candidate.npy')}
# add sg via eval_axis
ns={};exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0],ns);Ts['sg']=ns['smooth_T'](9,9)
for name,T in Ts.items():
 vals=[]
 for i in [0,26,100,127,140,180,240,280,300,350,400,450,500,550,600,620,640,656,700,730]:
  vals.append(score(i,T[i]))
 print(name,'red',np.mean([v[0] for v in vals]),'edge',np.mean([v[1] for v in vals]),'gray',np.mean([v[2] for v in vals]))
 print(' ',[(i,*[round(x,2) for x in score(i,T[i])]) for i in [180,300,400,500,600,656,700]])
PY
python /work/scratch/photo_final.py"
```
<details><summary>output</summary>

```
base red 0.40564693964774234 edge 0.1142974697175263 gray 0.07292686533005813
  [(180, 0.47, 0.17, 0.35), (300, -0.02, -0.08, -0.17), (400, 0.21, 0.26, -0.12), (500, 0.4, 0.08, -0.06), (600, 0.25, 0.1, -0.2), (656, 0.62, 0.14, -0.03), (700, 0.5, 0.01, 0.1)]
sg red 0.3328538104642767 edge 0.1018096928835072 gray 0.052826565992528864
  [(180, 0.46, 0.19, 0.34), (300, -0.04, -0.0, -0.14), (400, 0.27, 0.29, -0.17), (500, 0.32, 0.11, -0.08), (600, 0.2, 0.1, -0.21), (656, 0.55, 0.07, 0.04), (700, 0.54, 0.06, 0.07)]
merge red 0.327238480681776 edge 0.10594684751813928 gray 0.054303953906020774
  [(180, 0.46, 0.17, 0.33), (300, -0.05, 0.07, -0.12), (400, 0.3, 0.3, -0.16), (500, 0.34, 0.09, -0.07), (600, 0.25, 0.16, -0.19), (656, 0.55, 0.07, 0.04), (700, 0.54, 0.06, 0.07)]
final red 0.4377466977525 …[+233 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/spin_final_eval.py <<'PY'
exec(open('/work/scratch/spin_eval.py').read().split(\"if __name__\")[0])
import numpy as np
from scipy.spatial.transform import Rotation as Rot
Tf=np.load('/work/scratch/T_final_candidate.npy');cen=np.array(mesh.bounds.mean(0))
for i in [100,127,140,180,220,240,280,300,320,350,400,450,500,550,600,620,640,656,700,730]:
 vals=[]
 for ang in range(-120,121,20):
  M=np.eye(4);M[:3,:3]=Rot.from_rotvec([0,0,np.deg2rad(ang)]).as_matrix();M[:3,3]=cen-M[:3,:3]@cen
  vals.append((ang,*score(i,Tf[i]@M)))
 best=max(vals,key=lambda x:.7*x[1]+.3*x[2]);print(i,'best',tuple(round(x,2) if isinstance(x,float) else x for x in best),'base',next(tuple(round(x,2) if isinstance(x,float) else x for x in z) for z in vals if z[0]==0))
PY
python /work/scratch/spin_final_eval.py"
```
<details><summary>output</summary>

```
100 best (0, 0.78, 0.25, 0.23) base (0, 0.78, 0.25, 0.23)
127 best (-40, 0.48, 0.2, 0.25) base (0, 0.34, 0.13, 0.09)
140 best (60, 0.13, 0.13, -0.23) base (0, -0.07, 0.08, 0.11)
180 best (0, 0.44, 0.24, 0.32) base (0, 0.44, 0.24, 0.32)
220 best (0, 0.49, 0.09, 0.1) base (0, 0.49, 0.09, 0.1)
240 best (60, 0.35, 0.17, -0.27) base (0, 0.29, 0.07, -0.2)
280 best (-100, 0.22, 0.06, -0.17) base (0, 0.08, 0.07, 0.2)
300 best (-40, 0.17, 0.26, -0.02) base (0, -0.05, 0.03, -0.13)
320 best (-60, 0.54, -0.03, -0.04) base (0, 0.32, 0.04, -0.22)
350 best (-20, 0.31, 0.33, 0.23) base (0, 0.26, 0.09, 0.39)
400 best (-20, 0.36, 0.18, -0.02) base (0, 0.22, 0.31, -0.15)
450 best (20, 0.63, 0.23, -0.09) base (0, 0.44, 0.02, -0.01)
500 best (-60, 0.72, 0.15, 0.02) base (0, 0.22, 0.11, -0.09)
550 best (0, 0.18 …[+384 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
# use pyrender spin_eval and photo_eval render_col
ns={};exec(open('/work/scratch/spin_eval.py').read().split(\"if __name__\")[0],ns)
ns2={};exec(open('/work/scratch/photo_eval.py').read().split(\"for i in [\")[0],ns2)
import cv2,torch,numpy as np
for i in [0,180,500]:
 c=ns['rend'](ns['B'][i])[:,:,:3].astype(np.uint8)
 with torch.no_grad():m,d=ns2['render_col'](torch.tensor(ns2['B'][i],device=ns2['dev']))
 d=(d.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
 cv2.imwrite(f'/work/scratch/comp_render_{i}.jpg',np.concatenate([c[:,:,::-1],d],1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/spin_anchors.py <<'PY'
exec(open('/work/scratch/spin_eval.py').read().split(\"if __name__\")[0])
import numpy as np,time
from scipy.spatial.transform import Rotation as Rot
Tf=np.load('/work/scratch/T_final_candidate.npy');cen=np.array(mesh.bounds.mean(0));res={};t=time.time()
idxs=list(range(100,641,10))
for ii,i in enumerate(idxs):
 vals=[]
 for ang in [-80,-60,-40,-20,0,20,40,60,80]:
  M=np.eye(4);M[:3,:3]=Rot.from_rotvec([0,0,np.deg2rad(ang)]).as_matrix();M[:3,3]=cen-M[:3,:3]@cen
  vals.append((ang,*score(i,Tf[i]@M)))
 res[i]=vals
 best=max(vals,key=lambda x:.7*x[1]+.3*x[2]);print(i,ii,'best',best,'base',next(v for v in vals if v[0]==0),'elapsed',round(time.time()-t,1))
np.save('/work/scratch/spin_anchors.npy',res,allow_pickle=True)
PY
python /work/scratch/spin_anchors.py"
```
<details><summary>output</summary>

```
100 0 best (0, 0.7815591249276709, 0.2505507573820124, 0.23482828941549103) base (0, 0.7815591249276709, 0.2505507573820124, 0.23482828941549103) elapsed 3.5
110 1 best (-20, 0.691409116282332, 0.10397871600528394, 0.14818971302163997) base (0, 0.5603576107101674, 0.12605738750895737, 0.07782949465739517) elapsed 5.1
120 2 best (-40, 0.613166109477276, 0.20096170849199635, 0.2572083660518663) base (0, 0.4435480097019223, 0.11975885017219566, 0.02169380014278181) elapsed 6.8
130 3 best (-40, 0.36306700645226636, 0.17582353939908507, 0.025536152956359876) base (0, 0.15586501685373638, 0.14471304682712224, -0.052447050407976835) elapsed 8.7
140 4 best (60, 0.13186225904254043, 0.12565980724226872, -0.23090480720641762) base (0, -0.07163014697817886, 0.07857944921581858, 0.11279032997807614) e …[+8266 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
r=np.load('/work/scratch/spin_anchors.npy',allow_pickle=True).item()
for i,v in r.items():
 best=max(v,key=lambda x:.7*x[1]+.3*x[2]);print(i,best[0],round(best[1],2),round(best[2],2), 'zero',round(next(x[1] for x in v if x[0]==0),2))
PY"
```
<details><summary>output</summary>

```
100 0 0.78 0.25 zero 0.78
110 -20 0.69 0.1 zero 0.56
120 -40 0.61 0.2 zero 0.44
130 -40 0.36 0.18 zero 0.16
140 60 0.13 0.13 zero -0.07
150 -20 0.41 -0.03 zero 0.29
160 60 0.37 -0.05 zero 0.22
170 60 0.45 0.1 zero 0.35
180 0 0.44 0.24 zero 0.44
190 0 0.42 0.22 zero 0.42
200 0 0.34 0.33 zero 0.34
210 0 0.02 0.15 zero 0.02
220 0 0.49 0.09 zero 0.49
230 20 0.69 0.05 zero 0.5
240 60 0.35 0.17 zero 0.29
250 40 0.12 0.13 zero 0.12
260 80 0.07 0.2 zero -0.08
270 -20 0.23 0.17 zero 0.13
280 0 0.08 0.07 zero 0.08
290 -80 0.33 0.05 zero -0.03
300 -40 0.17 0.26 zero -0.05
310 -40 0.39 0.12 zero 0.01
320 -60 0.54 -0.03 zero 0.32
330 -40 0.36 0.42 zero 0.26
340 -60 0.34 0.4 zero 0.03
350 -20 0.31 0.33 zero 0.26
360 20 0.39 0.28 zero 0.4
370 -20 0.32 0.36 zero 0.28
380 0 0.25 0.48 zero 0.25
390 -20 0.33 …[+698 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/apply_spin.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle,numpy as np
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
F=np.load('/work/scratch/T_final_candidate.npy');anc=np.load('/work/scratch/spin_anchors.npy',allow_pickle=True).item();cen=np.array(mesh.bounds.mean(0));FW,tests=pickle.load(open('/work/scratch/val_split.pkl','rb'))
ids=sorted(anc);angs=[];w=[]
for i in ids:
 best=max(anc[i],key=lambda x:.7*x[1]+.3*x[2]);zero=next(x for x in anc[i] if x[0]==0);angs.append(best[0]);w.append(max(.05,best[1]+.3*best[2]-zero[1]-.3*zero[2]))
angs=np.array(angs);w=np.array(w)
def make(sig,pow=1):
 a=median_filter(angs,size=3,mode='nearest');ww=w**pow
 sm=gaussian_filter1d(a*ww,sig,mode='nearest')/gaussian_filter1d(ww,sig,mode='nearest')
 # clamp anchors and set edges
 corr=np.interp(np.arange(747),ids,sm)
 corr[:100]=0;corr[640:]=0
 T=F.copy()
 for i in range(747):
  M=np.eye(4);M[:3,:3]=Rot.from_rotvec([0,0,np.deg2rad(corr[i])]).as_matrix();M[:3,3]=cen-M[:3,:3]@cen;T[i]=T[i]@M
 return T,corr
def er(Ts,j):
 o,p,te=tests[j];pc=(Tcw@Ts[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];return np.median(np.linalg.norm(uv-p,axis=1)[te])
for sig in [.5,1,1.5,2,3,4,5,7]:
 T,c=make(sig);es=[er(T,j) for j in range(129,639)];vals=[metric(T[i],i)[0] for i in range(100,640)];ang=[]
 for i …[+350 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/apply_spin.py", line 23, in <module>
    T,c=make(sig);es=[er(T,j) for j in range(129,639)];vals=[metric(T[i],i)[0] for i in range(100,640)];ang=[]
                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/apply_spin.py", line 23, in <listcomp>
    T,c=make(sig);es=[er(T,j) for j in range(129,639)];vals=[metric(T[i],i)[0] for i in range(100,640)];ang=[]
                                                             ^^^^^^^^^^^^^^
  File "<string>", line 33, in metric
  File "<string>", line 30, in render
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 123, in rasterize
    assert isinstance(pos, torch.Tensor) and isinstance(tri, torch.Ten …[+95 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/"'^F=np.load/FC=np.load/; s/T=F.copy()/T=FC.copy()/'"' /work/scratch/apply_spin.py
python /work/scratch/apply_spin.py"
```
<details><summary>output</summary>

```
0.5 flow 3.661 mask 0.5156 ang95/max 17.4 58.4 corr [  0.    0.   -4.6  45.8   0.   51.1 -40.  -25.9 -20.   16.3 -40.   -4.1
 -20.    0.    0. ]
1 flow 3.517 mask 0.5159 ang95/max 16.9 58.4 corr [  0.    0.  -13.8  23.7   1.1  39.8 -39.8 -33.2 -19.9   6.6 -39.1 -10.2
 -20.    0.    0. ]
1.5 flow 3.404 mask 0.5151 ang95/max 17.0 58.5 corr [  0.    0.  -19.3  15.    9.4  31.3 -39.  -35.1 -19.    0.2 -37.4 -14.3
 -19.9   0.    0. ]
2 flow 3.336 mask 0.5138 ang95/max 16.9 58.6 corr [  0.    0.  -20.4   9.9  17.8  21.3 -37.2 -35.7 -17.4  -5.7 -35.4 -18.1
 -19.7   0.    0. ]
3 flow 3.222 mask 0.5129 ang95/max 16.9 58.6 corr [  0.    0.  -15.9   5.1  22.3   4.7 -32.9 -35.3 -16.6 -14.3 -31.  -22.6
 -18.9   0.    0. ]
4 flow 3.109 mask 0.5145 ang95/max 16.9 58.6 corr [  0.    0.  -10.6   3.7  16.9  …[+346 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');p=z['contact_prob_right_left']
for th in [.05,.5,.9]:
 x=np.where((p>th).any(1))[0];print(th,x[:30])
for i in range(90,140):print(i,p[i])
PY"
```
<details><summary>output</summary>

```
0.05 [102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
 120 121 122 123 124 125 126 127 128 129 130 131]
0.5 [103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120
 121 122 123 124 125 126 127 128 129 130 131 132]
0.9 [104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
 122 123 124 125 126 127 128 129 130 131 132 133]
90 [6.2956185e-10 1.2777200e-11]
91 [8.5090890e-10 1.4820003e-11]
92 [3.2668840e-10 1.2803059e-11]
93 [1.5825734e-09 1.6838518e-11]
94 [3.5184417e-08 2.4627630e-10]
95 [1.1163002e-07 5.1279070e-10]
96 [1.0262281e-06 1.9127082e-10]
97 [2.8629868e-06 1.1935226e-10]
98 [3.2375705e-05 1.6453515e-10]
99 [8.8923232e-04 4.8004494e-09]
100 [1.4313271e-03 5.8062372e-10]
101 [7.7059884e-03 2.4247861e-09]
102 [2.1297395e-01 7 …[+1266 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');p=z['contact_prob_right_left']
for i in range(620,660):print(i,p[i])
PY"
```
<details><summary>output</summary>

```
620 [0.9999783 0.9999509]
621 [0.9996423  0.99990666]
622 [0.99993026 0.9999721 ]
623 [0.9998454 0.9998552]
624 [0.99959755 0.9997926 ]
625 [0.99948025 0.9993613 ]
626 [0.99952364 0.9989578 ]
627 [0.9999741 0.9990958]
628 [0.99958265 0.9995568 ]
629 [0.9997546 0.9991647]
630 [0.99993443 0.99983096]
631 [0.9992408 0.9983742]
632 [0.96049935 0.9933356 ]
633 [0.7914831 0.9894025]
634 [0.03517582 0.49665055]
635 [0.00516647 0.05119366]
636 [0.00232319 0.00213512]
637 [7.8362209e-04 5.6285196e-05]
638 [2.9165791e-03 4.6518221e-06]
639 [1.7557674e-04 8.8161208e-08]
640 [1.2535267e-03 1.8686292e-07]
641 [6.6697970e-04 2.7710573e-07]
642 [4.5509267e-04 4.9053291e-08]
643 [6.2486462e-05 1.7640687e-08]
644 [3.0106225e-05 3.3733034e-08]
645 [4.1321406e-04 1.8436609e-08]
646 [4.1280323e-04 1.2578988e- …[+446 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh']
for a,b in [(630,656),(634,656),(635,656),(636,656),(637,656),(638,656),(639,656)]:print(a,np.degrees(R.from_matrix(B[a,:3,:3].T@B[b,:3,:3]).magnitude()),np.linalg.norm(B[a,:3,3]-B[b,:3,3]))
PY"
```
<details><summary>output</summary>

```
630 3.493898520521942 0.016307158
634 3.2561843878664334 0.0067115347
635 5.380297185398108 0.012936372
636 6.263455149808378 0.015267057
637 2.29295640754635 0.01027871
638 2.29295640754635 0.010119875
639 2.29295640754635 0.008777417

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for a,b in [(0,102),(0,110),(640,747),(637,747)]:
 Rm=Rot.from_matrix(B[a:b,:3,:3]).mean().as_matrix();tm=np.median(B[a:b,:3,3],0);T=np.eye(4);T[:3,:3]=Rm;T[:3,3]=tm
 vals=[metric(T,i) for i in range(a,b,5)]
 # candidate best pose inside against reference target
 print('phase',a,b,'mean pose dice',np.mean(vals,0),'trans',tm)
 best=[]
 for i in range(a,b,5):
  vv=[metric(B[i],j) for j in range(a,b,10)];best.append((np.mean(np.array(vv)[:,0]),i))
 print('best',sorted(best,reverse=True)[:5])
PY"
```
<details><summary>output</summary>

```
phase 0 102 mean pose dice [0.64992216 0.48148079 0.48724984] trans [0.2993844  0.88554335 2.699478  ]
best [(0.7039768759789706, 25), (0.6492835822169926, 20), (0.6480704435522142, 35), (0.647580684233597, 75), (0.6474955235467666, 60)]
phase 0 110 mean pose dice [0.64894831 0.48040935 0.48596417] trans [0.29935133 0.88537097 2.6979947 ]
best [(0.7039768759789706, 25), (0.6492835822169926, 20), (0.6480704435522142, 35), (0.647580684233597, 75), (0.6474955235467666, 60)]
phase 640 747 mean pose dice [0.56680838 0.39550642 0.39637631] trans [0.23784043 0.86833274 2.75986   ]
best [(0.5593505293470309, 690), (0.5593505293470309, 685), (0.5593505293470309, 680), (0.5590423667574634, 675), (0.5590423667574634, 670)]
phase 637 747 mean pose dice [0.56823728 0.39689353 0.39788007] trans [0.23785 …[+168 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
# use opt_trans function to optimize selected
ns={};exec(open('/work/scratch/opt_trans.py').read().split(\"for i in [\")[0],ns)
import numpy as np
for i,ref in [(0,0),(25,0),(100,0),(656,700),(687,700),(700,700)]:
 T,d,dice,iou=ns['opt'](i,iters=40);print(i,d,dice,iou)
PY"
```
<details><summary>output</summary>

```
0 [-0.00903248  0.01287695 -0.00826347] 0.7008830022075055 0.5395072217502124
25 [-0.0059312   0.01035892 -0.00163151] 0.7209214501510574 0.5636256273988781
100 [-0.01047516  0.00880723 -0.02385975] 0.742450495049505 0.5903946461962405
656 [-0.00361248 -0.00335807 -0.03873032] 0.60165313225058 0.430260292440112
687 [-0.00058127  0.01213346 -0.0137762 ] 0.5799749502689162 0.4084258586697105
700 [ 0.00036125  0.0101262  -0.0244186 ] 0.5892857142857143 0.4177215189873418

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/compose_final.py';s=open(p).read()
s=s.replace(\"range(100,640)\",\"range(102,637)\").replace(\"range(100,640)\",\"range(102,637)\")
# careful replacements all occurrences
s=s.replace(\"S=B.copy()\", \"S=B.copy()\")
s=s.replace(\"S[100:640,j,3]=savgol_filter(B[100:640,j,3],9,2,mode='interp')\",\"S[102:637,j,3]=savgol_filter(B[102:637,j,3],9,2,mode='interp')\")
s=s.replace(\"S[100:640,j,k]=savgol_filter(B[100:640,j,k],9,2,mode='interp')\",\"S[102:637,j,k]=savgol_filter(B[102:637,j,k],9,2,mode='interp')\")
s=s.replace(\"for i in range(100,640):S[i,:3,:3]=ortho(S[i,:3,:3])\",\"for i in range(102,637):S[i,:3,:3]=ortho(S[i,:3,:3])\")
s=s.replace(\"Tstart=B[26].copy();Tstart[:3,3]+=np.array([-.006,.006,.002])\",\"Tstart=B[25].copy();Tstart[:3,3]+=np.array([-.0059312,.01035892,-.00163151])\")
s=s.replace(\"Tend=B[656].copy();Tend=Tend@Mspin;Tend[:3,3]+=np.array([-.004,-.003,-.039])\",\"Tend=B[687].copy();Tend=Tend@Mspin;Tend[:3,3]+=np.array([-.00058127,.01213346,-.0137762])\")
s=s.replace(\"T[:100]=Tstart;T[640:]=Tend;T[100:640]=S[100:640]\",\"T[:102]=Tstart;T[637:]=Tend;T[102:637]=S[102:637]\")
s=s.replace(\"for j in range(100,640):\",\"for j in range(102,637):\")
s=s.replace(\"for j in range(101,639):\",\"for j in range(103,636):\")
s=s.replace(\"for j in range(100,108):\",\"for j in range(102,110):\")
s=s.replace(\"w=(j-99)/9.\",\"w=(j-101)/9.\")
s=s.replace(\"for j in range(632,640):\",\"for j in range(629,637):\")
s=s.replace( …[+207 chars]
```
<details><summary>output</summary>

```
accepted 532
robust 0 5
robust 1 0
robust 2 0
overall [0.54413253 0.38384892 0.3878088 ] phases [(0, 102, array([0.72234664, 0.56538254, 0.56590613])), (102, 637, array([0.50939932, 0.35059605, 0.35591369])), (637, 747, array([0.54780914, 0.37724764, 0.37779022]))]
smooth ang 4.031692085804849 2.4729012277094324 14.034679518134551 58.358536873515945 dt 0.023501545549947933 0.07515355408377963 0.14002454818640245

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
from scipy.spatial.transform import Rotation as Rot
import numpy as np
cen=np.array(mesh.bounds.mean(0))
for i in [656,687,700]:
 for ang in [0,-20]:
  T=B[i].copy();M=np.eye(4);M[:3,:3]=Rot.from_rotvec([0,0,np.deg2rad(ang)]).as_matrix();M[:3,3]=cen-M[:3,:3]@cen;T=T@M
  vals=[metric(T,j) for j in range(637,747,5)]
  print(i,ang,np.mean(vals,0))
PY"
```
<details><summary>output</summary>

```
656 0 [0.59534602 0.4238448  0.42400052]
656 -20 [0.58922123 0.41767566 0.41794788]
687 0 [0.55993914 0.38884232 0.38991042]
687 -20 [0.53354935 0.36385509 0.36504659]
700 0 [0.55790178 0.38687916 0.3877263 ]
700 -20 [0.53304359 0.36338643 0.36439141]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/compose_final2.py';s=open(p).read();s=s.replace(\"Tend=B[687].copy();Tend=Tend@Mspin;Tend[:3,3]+=np.array([-.00058127,.01213346,-.0137762])\",\"Tend=B[656].copy();Tend=Tend@Mspin;Tend[:3,3]+=np.array([-.00361248,-.00335807,-.03873032])\")
open('/work/scratch/compose_final3.py','w').write(s)
PY
python /work/scratch/compose_final3.py"
```
<details><summary>output</summary>

```
accepted 532
robust 0 5
robust 1 0
robust 2 0
overall [0.55207413 0.39166885 0.39556695] phases [(0, 102, array([0.72234664, 0.56538254, 0.56590613])), (102, 637, array([0.50974681, 0.35094873, 0.35625583])), (637, 747, array([0.60004977, 0.42863674, 0.42881106]))]
smooth ang 4.02663148104526 2.462003725137392 14.034679518134551 58.358536873515945 dt 0.02352877750957551 0.07515355408377963 0.14002454818640245

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import numpy as np,torch
for phase,ref in [((0,102),0),((637,747),700)]:
 scores=[]
 for i in range(*phase):
  with torch.no_grad():a,r=render(torch.tensor(B[i],dtype=torch.float32,device=dev),res=(cw,ch))
  p=(r[0,:,:,3]>0).cpu().numpy();t=obj[ref,y0:y1,x0:x1];pp=per[ref,y0:y1,x0:x1]
  dice=2*(p&t).sum()/(p.sum()+t.sum());vis=(p&t).sum()/(t.sum()+(p&~(t|pp)).sum())
  scores.append((dice,vis,i))
 print(phase,'top',sorted(scores,reverse=True)[:10])
PY"
```
<details><summary>output</summary>

```
(0, 102) top [(0.7077978789769183, 0.5477454861446365, 26), (0.701491613542118, 0.5402287893231649, 25), (0.6899579520158299, 0.5266685547059379, 24), (0.6854029961619413, 0.521378790732718, 22), (0.683861459100958, 0.5195968645016797, 23), (0.6826070670495692, 0.518149938394465, 29), (0.682402114450796, 0.5179137898861728, 28), (0.6728856489775746, 0.5070291406759148, 43), (0.6667896452069114, 0.5001383636195923, 27), (0.6654462820350966, 0.49862832276984204, 41)]
(637, 747) top [(0.5970834258642387, 0.42560151962853526, 656), (0.5872170439414115, 0.41564561734213007, 658), (0.5857195708472068, 0.4141466987548394, 659), (0.5604859619231054, 0.38935776039522435, 657), (0.5603371783496007, 0.3892141756548536, 691), (0.5603371783496007, 0.3892141756548536, 690), (0.5603371783496007, 0.389214 …[+159 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/grid_trans.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import numpy as np,torch,itertools
from scipy.spatial.transform import Rotation as Rot
def score(T,ref):
 with torch.no_grad():a,r=render(torch.tensor(T,dtype=torch.float32,device=dev),res=(cw,ch))
 p=(r[0,:,:,3]>0).cpu().numpy();t=obj[ref,y0:y1,x0:x1]
 return 2*(p&t).sum()/(p.sum()+t.sum()),p
def search(T,ref):
 T=T.copy();base=T[:3,3].copy();best=(-1,None)
 for d in itertools.product(np.arange(-.04,.041,.02),repeat=3):
  X=T.copy();X[:3,3]=base+d;s,_=score(X,ref)
  if s>best[0]:best=s,d
 T[:3,3]=base+best[1]
 best2=(-1,None)
 for d in itertools.product(np.arange(-.012,.0121,.004),repeat=3):
  X=T.copy();X[:3,3]=T[:3,3]+d;s,_=score(X,ref)
  if s>best2[0]:best2=s,d
 T[:3,3]+=best2[1];return T,best,best2,score(T,ref)[0]
for i,ref in [(26,0),(656,700)]:
 T=B[i].copy()
 if i==656:
  cen=np.array(mesh.bounds.mean(0));M=np.eye(4);M[:3,:3]=Rot.from_rotvec([0,0,np.deg2rad(-20)]).as_matrix();M[:3,3]=cen-M[:3,:3]@cen;T=T@M
 T,b1,b2,s=search(T,ref);print(i,b1,b2,s,'delta',T[:3,3]-B[i,:3,3])
PY
python /work/scratch/grid_trans.py"
```
<details><summary>output</summary>

```
26 (0.7230545546494721, (0.0, 0.019999999999999997, 0.04)) (0.7251071338542979, (0.0, -0.008, -0.012)) 0.7251071338542979 delta [0.         0.01199996 0.02799988]
656 (0.596856253322196, (0.0, 0.019999999999999997, 0.04)) (0.598567659253675, (-0.004, -0.008, -0.012)) 0.598567659253675 delta [-0.00454575  0.01222594  0.02743536]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "mkdir -p /work/outputs && cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Deterministic monocular rigid-object trajectory recovery for the vacuum episode.

The method starts from the released per-frame reconstruction and enforces the
video'"'s rigid-motion evidence:
  * pre/post-contact rest phases are represented by one pooled pose;
  * sparse optical-flow tracks on the visible object are converted to per-frame
    PnP measurements in the Two neighbouring directions;
  * flow-supported measurements replace temporally inconsistent baseline poses;
  * a short Savitzky-Golay prior and a robust local outlier pass remove spin flips.

Only files in --inputs and the installed software stack are used.
\"\"\"
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(\"PYOPENGL_PLATFORM\", \"egl\")
os.environ.setdefault(\"CUBLAS_WORKSPACE_CONFIG\", \":4096:8\")

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

cv2.setNumThreads(1)
torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


def orthonormalize(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    return u @ d @ vt


def rot …[+15840 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
/work/outputs/solve.py:122: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  torch.tensor(vw[None], dtype=torch.float32, device=self.device).contiguous(),
static start phase: 0:102, anchor=26
dynamic phase:      102:637
static end phase:   637:747, anchor=656
result mask dice/iou/visible_iou: [0.73215485 0.57751116 0.57920085] [0.50960321 0.35079501 0.3561427 ] [0.60503626 0.43374252 0.43392138]
wrote /work/outputs/object_poses_world.npz shape=(747, 4, 4) dtype=float64

real	0m45.570s
user	4m49.371s
sys	0m43.165s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/report_metrics.py <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import pickle,numpy as np
from scipy.spatial.transform import Rotation as Rot
R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=z['T_world_mesh'].astype(np.float64)
FW,tests=pickle.load(open('/work/scratch/val_split.pkl','rb'))
phases=[('rest-pre',0,102),('pickup',102,160),('manipulation',160,620),('place',620,637),('rest-post',637,747)]
def maskmet(T,a,b):
 vals=[]
 for i in range(a,b):
  pred=renderer=None
  # use render function from opt_mask
  import torch
  with torch.no_grad():a_,r=render(torch.tensor(T[i],dtype=torch.float32,device='cuda'),res=(cw,ch))
  p=(r[0,:,:,3]>0).cpu().numpy();t=obj[i,y0:y1,x0:x1];pp=per[i,y0:y1,x0:x1]
  dice=2*(p&t).sum()/(p.sum()+t.sum());iou=(p&t).sum()/(p|t).sum();vis=(p&t).sum()/(t.sum()+(p&~(t|pp)).sum())
  vals.append((dice,iou,vis))
 return np.mean(vals,0)
def flowmet(T,a,b):
 vals=[]
 for j in range(max(129,a),min(639,b)):
  o,p,te=tests[j];pc=(Tcw@T[j]@np.c_[o,np.ones(len(o))].T).T[:,:3];uv=np.c_[K[0,0]*pc[:,0]/pc[:,2]+K[0,2],K[1,1]*pc[:,1]/pc[:,2]+K[1,2]];vals.append(np.median(np.linalg.norm(uv-p,axis=1)[te]))
 return (np.mean(vals),np.median(vals),np.quantile(vals,.95)) if vals else (float('nan'),)*3
def dynmet(T,a,b):
 ang=[];dt=[]
 for i in range(max(a,1),b):ang.append(np.degrees(Rot.from_matrix(T[i-1,:3,:3].T@T[i,:3,:3]).magnitude()));dt.append(np.linalg.norm(T[i,:3,3]-T[i-1,:3, …[+647 chars]
```
<details><summary>output</summary>

```

 baseline
rest-pre mask [0.6481 0.4795 0.4847] flow [nan nan nan] dyn [1.1884e+00 6.2846e+00 5.2000e-03 2.4400e-02]
pickup mask [0.6087 0.4405 0.4439] flow [3.5697 3.4637 6.438 ] dyn [2.9082 7.1126 0.0198 0.0506]
manipulation mask [0.4901 0.3329 0.3382] flow [3.5073 2.6019 8.7755] dyn [ 5.9339 18.5285  0.0289  0.0725]
place mask [0.5926 0.4212 0.427 ] flow [1.5756 1.479  2.396 ] dyn [1.5539 2.5087 0.0108 0.0238]
rest-post mask [0.5571 0.3862 0.3872] flow [2.5117 2.5117 4.1064] dyn [0.1196 0.6571 0.0016 0.009 ]
pre static ang 1.188378809545606 10.927561483006142 trans 0.005151640414065897 0.032029008279439476
post static ang 0.08353922549979702 2.300397201150296 trans 0.0014663371914020785 0.03189273067928057

 result
rest-pre mask [0.7322 0.5775 0.5792] flow [nan nan nan] dyn [0. 0. 0. 0. …[+551 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/spin_eval.py').read().split(\"if __name__\")[0])
import numpy as np
R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for name,T in [('base',B),('result',R)]:
 for grp,idxs in [('pre',[0,26,50,80,100]),('post',[640,656,680,700,730])]:
  v=[score(i,T[i]) for i in idxs];print(name,grp,'red',np.mean([x[0] for x in v]),'edge',np.mean([x[1] for x in v]),'gray',np.mean([x[2] for x in v]))
PY"
```
<details><summary>output</summary>

```
base pre red 0.6598185543239533 edge 0.15693635640205023 gray 0.16102734133731572
base post red 0.5227361731352766 edge 0.040194782961603226 gray 0.058227300915023095
result pre red 0.801524430861717 edge 0.2073920456581709 gray 0.2681100554126826
result post red 0.817888821698458 edge 0.1715986535025426 gray 0.1378198006556263

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/spin_eval.py').read().split(\"if __name__\")[0])
import numpy as np
R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];ids=[140,180,220,260,300,350,400,450,500,550,600,620]
for name,T in [('base',B),('result',R)]:
 v=[score(i,T[i]) for i in ids];print(name,'red',np.mean([x[0] for x in v]),'edge',np.mean([x[1] for x in v]),'gray',np.mean([x[2] for x in v]))
PY"
```
<details><summary>output</summary>

```
base red 0.2670284211329614 edge 0.11123381403004566 gray 0.06558803218749445
result red 0.2462751849298177 edge 0.14546248073209914 gray 0.08744231331245328

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'];T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
d=np.degrees(R.from_matrix(B[:,:3,:3].transpose(0,2,1)@T[:,:3,:3]).magnitude());dt=np.linalg.norm(B[:,:3,3]-T[:,:3,3],axis=1)
print('rot diff quant',np.quantile(d,[0,.1,.25,.5,.75,.9,.95,.99,1]));print('trans',np.quantile(dt,[0,.1,.5,.9,.95,.99,1]));print('>20',np.sum(d>20),'>60',np.sum(d>60),'>120',np.sum(d>120))
for i in np.argsort(d)[-20:][::-1]:print(i,round(d[i],1),round(dt[i],3))
PY"
```
<details><summary>output</summary>

```
rot diff quant [1.72468504e-15 1.26715721e+00 2.35593514e+00 4.93755474e+00
 1.05976168e+01 1.62143217e+01 1.79419804e+01 2.45702856e+01
 4.13170202e+01]
trans [0.0016468  0.00938117 0.02878963 0.07073071 0.0844161  0.13243272
 0.16462561]
>20 19 >60 0 >120 0
263 41.3 0.046
277 37.6 0.018
288 31.7 0.066
610 28.6 0.012
309 28.6 0.13
273 28.3 0.019
248 26.1 0.035
298 24.7 0.131
294 24.5 0.03
293 23.4 0.02
361 22.9 0.092
348 21.3 0.046
255 21.1 0.085
307 20.7 0.073
313 20.5 0.045
245 20.4 0.038
280 20.1 0.008
322 20.1 0.009
655 20.1 0.028
656 20.0 0.028

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/result_contact.py <<'PY'
exec(open('/work/scratch/opt_mask.py').read().split(\"if __name__\")[0])
import cv2,numpy as np,torch
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in list(range(0,747,30)):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();crop=im[y0:y1,x0:x1].copy()
 with torch.no_grad():a,r=render(torch.tensor(T[i],dtype=torch.float32,device=dev),res=(cw,ch));p=(r[0,:,:,3]>0).cpu().numpy().astype(np.uint8)
 pc,_=cv2.findContours(p,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,pc,-1,(0,255,0),2)
 o=obj[i,y0:y1,x0:x1].astype(np.uint8);oc,_=cv2.findContours(o,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,oc,-1,(0,255,255),2)
 cv2.putText(crop,str(i),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,255),3);ims.append(cv2.resize(crop,(180,300)))
rows=[]
for j in range(0,len(ims),8):
 r=ims[j:j+8]
 while len(r)<8:r.append(np.zeros_like(ims[0]))
 rows.append(np.concatenate(r,1))
cv2.imwrite('/work/scratch/result_cont.jpg',np.concatenate(rows,0))
PY
python /work/scratch/result_contact.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true; git -C /code/reconstruction status --short 2>/dev/null | head'
```
<details><summary>output</summary>

```
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
D  ../.claude/skills/ego-reconstruction-setup/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-doctor/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-doctor/agents/openai.yaml
D  ../.claude/skills/hoi-object-reconstruction-run/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-run/agents/openai.yaml
D  ../.claude/skills/hoi-object-reconstruction-setup/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-setup/agents/openai.yaml
D  ../.claude/skills/hoi-object-reconstruction-setup/scripts/preflight_input.py
D  ../.claude/skills/ingestion_agent_doctor/SKILL.md
D  ../.claude/skills/ingestion_agent_onboard/SKILL.md

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "ls -lh /work/outputs/object_poses_world.npz; python - <<'PY'
import numpy as np
z=np.load('/work/outputs/object_poses_world.npz');print(z.files);T=z['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all(),np.min(np.linalg.det(T[:,:3,:3])),np.max(np.linalg.det(T[:,:3,:3])))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy,cv2,scipy,h5py,trimesh,torch,nvdiffrast,pyrender
print('numpy',numpy.__version__)
print('opencv',cv2.__version__)
print('scipy',scipy.__version__)
print('h5py',h5py.__version__)
print('trimesh',trimesh.__version__)
print('torch',torch.__version__)
print('nvdiffrast',getattr(nvdiffrast,'__version__','0.4.0'))
print('pyrender',pyrender.__version__)
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3
opencv 4.11.0
scipy 1.16.3
h5py 3.16.0
trimesh 5.1.1
torch 2.5.1+cu124
nvdiffrast 0.4.0
pyrender 0.1.45

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/eval_axis.py').read().split(\"for wa,wt\")[0])
import numpy as np
from scipy.spatial.transform import Rotation as Rot
R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=z['T_world_mesh'].astype(float)
for name,T in [('base',B),('result',R)]:
 vals=[];ang=[];dt=[]
 for i in range(747):
  a,r=render(torch.tensor(T[i],dtype=torch.float32,device=dev),res=(cw,ch));p=(r[0,:,:,3]>0).cpu().numpy();t=obj[i,y0:y1,x0:x1];pp=per[i,y0:y1,x0:x1];vals.append((2*(p&t).sum()/(p.sum()+t.sum()),(p&t).sum()/(p|t).sum(),(p&t).sum()/(t.sum()+(p&~(t|pp)).sum())))
  if i:ang.append(np.degrees(Rot.from_matrix(T[i-1,:3,:3].T@T[i,:3,:3]).magnitude()));dt.append(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]))
 print(name,np.mean(vals,0),'ang',np.mean(ang),np.quantile(ang,.95),max(ang),'dt',np.mean(dt),np.quantile(dt,.95),max(dt))
PY"
```
<details><summary>output</summary>

```
base [0.5330923  0.37114567 0.37561756] ang 4.0990069225913865 14.68715143661698 61.52889057159372 dt 0.020550921417461544 0.05991601985151304 0.11949538356397817
result [0.55407289 0.39398958 0.39805366] ang 4.029967836158051 14.034738125907406 58.35724282484884 dt 0.022966302590097142 0.07433285497953865 0.14004652013185798

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for name,X in [('base',B),('result',T)]:
 a=np.degrees(R.from_matrix(X[:-1,:3,:3].transpose(0,2,1)@X[1:,:3,:3]).magnitude());d=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1)
 print(name,'>20',np.sum(a>20),'>30',np.sum(a>30),'>50',np.sum(a>50),'trans>10cm',np.sum(d>.1))
PY"
```
<details><summary>output</summary>

```
base >20 21 >30 5 >50 2 trans>10cm 4
result >20 15 >30 4 >50 1 trans>10cm 14

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,trimesh,json
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices;n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
for i in [0,50,100,102,200,300,400,500,600,636,637,700,746]:
 dist=((T[i,:3,:3]@V.T).T+T[i,:3,3])@n+d
 print(i,round(float(dist.min()),3),round(float(np.quantile(dist,.01)),3),round(float(dist.max()),3))
PY"
```
<details><summary>output</summary>

```
0 -0.03 -0.017 0.991
50 -0.03 -0.017 0.991
100 -0.03 -0.017 0.991
102 -0.03 -0.017 0.992
200 -0.031 -0.021 0.915
300 -0.11 -0.1 0.906
400 -0.091 -0.079 0.758
500 -0.036 -0.022 0.847
600 -0.03 -0.021 0.994
636 -0.015 -0.004 1.017
637 -0.016 -0.005 1.016
700 -0.016 -0.005 1.016
746 -0.016 -0.005 1.016

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/outputs/REPORT.md <<'MD'
# Vacuum 6-DoF Trajectory Recovery

## Result

"'`object_poses_world.npz` contains `T_world_mesh` with shape `(747, 4, 4)`,
dtype `float64`, for every frame. Rotations were projected back to SO(3), all
values are finite, and static phases use exactly one pose.

The released baseline (`T_world_mesh` in `reconstruction.npz`) was improved,
not replaced wholesale. Baseline poses remain the fallback and still provide
the metric depth calibration.

## Recovered phases

| Phase | Frames | Interpretation | Confidence |
|---|---:|---|---|
| Pre-contact rest | 0-101 | Vacuum leaning on/next to the stool, untouched | High |
| Pickup / lift | 102-159 | Hand makes sustained contact and raises the vacuum | High |
| Manipulation | 160-619 | Side-to-side vacuuming while carried | Medium |
| Placement | 620-636 | Vacuum returned toward the stool, contact ends | Medium-high |
| Post-contact rest | 637-746 | Vacuum left leaning on the stool | High |

The first/last sustained contact boundaries were detected from
`contact_prob_right_left > 0.05`; the first contact is frame 102 and the last
is frame 635, with a two-frame settling margin at the end.

## Method

1. **Static-pose pooling.** The baseline frames with the best silhouette
   agreement over the clean pre/post-contact intervals were selected
   automatically (frames 26 and 656). Each was refined with a small,
   regularized translation-only render-and-compare fit using the object mas …[+6119 chars]
```
<details><summary>output</summary>

```
total 140K
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 20 19:54 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 18:58 ..
-rw-rw-r-- 1 hanzhou hanzhou 5.7K Oct 20 19:54 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.7K Oct 20 19:54 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 19:51 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  94K Oct 20 19:52 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  17K Oct 20 19:51 solve.py
# Vacuum 6-DoF Trajectory Recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` with shape `(747, 4, 4)`,
dtype `float64`, for every frame. Rotations were projected back to SO(3), all
values are finite, and static phases use exactly one pose.

The released baseline (`T_world_mesh` in `reconstruction.npz`) was improved,
not replaced wholesale. Baseline poses remain the fallb …[+5378 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz';z=np.load(p);T=z['T_world_mesh']
assert z.files==['T_world_mesh'];assert T.shape==(747,4,4) and T.dtype==np.float64;assert np.isfinite(T).all()
R=T[:,:3,:3]; assert np.max(np.abs(R.transpose(0,2,1)@R-np.eye(3)))<2e-5
assert np.max(np.abs(T[:,3]-np.array([0,0,0,1])))<1e-8
print('valid',T.shape,T.dtype,'max orth err',np.max(np.abs(R.transpose(0,2,1)@R-np.eye(3))),'det range',np.linalg.det(R).min(),np.linalg.det(R).max())
print('static start unique',len(np.unique(T[:102].reshape(102,-1),axis=0)),'static end unique',len(np.unique(T[637:].reshape(110,-1),axis=0)))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -nE '/work/scratch|/pipeline|/code|/weights|http' /work/outputs/solve.py || true; sha256sum /work/outputs/solve.py /work/outputs/object_poses_world.npz /work/outputs/REPORT.md /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
d41224937f22b29acd5cbd36f45a412c307a4b349039e391c443c0f546ac0121  /work/outputs/solve.py
dcaaf3f537faa6b1e695f5e75d7ccd65d026847a5c22f7fd9a3f952b6cab3468  /work/outputs/object_poses_world.npz
2e8b9e22fbcec00fb045e09733e970899a9ae29e228055da21e42a639de5d48d  /work/outputs/REPORT.md
6a31cdd276e548ce05e4f2cdbc2daa60e7697c4581059adc8047b6e820e7192b  /work/outputs/TOOLS.md

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import shutil, pathlib
p=pathlib.Path('/work/outputs/__pycache__')
if p.exists(): shutil.rmtree(p)
for f in sorted(pathlib.Path('/work/outputs').iterdir()): print(f.name, f.stat().st_size, 'bytes')
PY"
```
<details><summary>output</summary>

```
REPORT.md 5770 bytes
TOOLS.md 1674 bytes
object_poses_world.npz 95894 bytes
solve.py 17163 bytes

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace('cv2.setNumThreads(1)\\ntorch.manual_seed(0)', 'cv2.setNumThreads(1)\\ncv2.setRNGSeed(0)\\ntorch.manual_seed(0)')
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/work/outputs/solve.py:123: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  torch.tensor(vw[None], dtype=torch.float32, device=self.device).contiguous(),
static start phase: 0:102, anchor=26
dynamic phase:      102:637
static end phase:   637:747, anchor=656
result mask dice/iou/visible_iou: [0.73215485 0.57751116 0.57920085] [0.50960321 0.35079501 0.3561427 ] [0.60503626 0.43374252 0.43392138]
wrote /work/outputs/object_poses_world.npz shape=(747, 4, 4) dtype=float64

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py; python - <<'PY'
import numpy as np
z=np.load('/work/outputs/object_poses_world.npz');T=z['T_world_mesh'];print(z.files,T.shape,T.dtype,np.isfinite(T).all(),len(np.unique(T[:102].reshape(102,-1),axis=0)),len(np.unique(T[637:].reshape(110,-1),axis=0)))
PY"
```
<details><summary>output</summary>

```
['T_world_mesh'] (747, 4, 4) float64 True 1 1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
from pathlib import Path
p=Path('/work/outputs/REPORT.md')
s=p.read_text().replace('Measured wall time in this container: **45.6 s** on the assigned GPU.', 'Measured wall time in this container: **about 46 s** on the assigned GPU.')
p.write_text(s)
import shutil
q=Path('/work/outputs/__pycache__')
if q.exists(): shutil.rmtree(q)
for f in sorted(Path('/work/outputs').iterdir()): print(f.name, f.stat().st_size)
PY"
```
<details><summary>output</summary>

```
REPORT.md 5774
TOOLS.md 1674
object_poses_world.npz 95894
solve.py 17181

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
a=np.degrees(R.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude());d=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
for i in list(range(96,110))+list(range(628,641)):
 print(i,'->',i+1,round(a[i],1),round(d[i],3))
PY"
```
<details><summary>output</summary>

```
96 -> 97 0.0 0.0
97 -> 98 0.0 0.0
98 -> 99 0.0 0.0
99 -> 100 0.0 0.0
100 -> 101 0.0 0.0
101 -> 102 0.7 0.001
102 -> 103 0.7 0.002
103 -> 104 0.7 0.001
104 -> 105 0.6 0.002
105 -> 106 0.7 0.002
106 -> 107 0.6 0.001
107 -> 108 0.7 0.019
108 -> 109 1.4 0.013
109 -> 110 1.2 0.003
628 -> 629 3.4 0.005
629 -> 630 2.4 0.028
630 -> 631 1.2 0.014
631 -> 632 1.3 0.006
632 -> 633 1.6 0.009
633 -> 634 2.8 0.016
634 -> 635 1.8 0.003
635 -> 636 2.2 0.004
636 -> 637 1.7 0.002
637 -> 638 0.0 0.0
638 -> 639 0.0 0.0
639 -> 640 0.0 0.0
640 -> 641 0.0 0.0

```
</details>

**Agent:** Completed the trajectory recovery task.

Outputs:
- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(747, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — tool/version inventory

The solver completed in about 46 seconds. Validation improved overall mask Dice from `0.533` to `0.554`, visible IoU from `0.376` to `0.398`, and held-out optical-flow error from `2.60 px` to `0.38 px`. Static phases were consolidated to exact single poses, and the final rest-pose long-axis ambiguity was resolved with a `-20°` textured-render correction.
