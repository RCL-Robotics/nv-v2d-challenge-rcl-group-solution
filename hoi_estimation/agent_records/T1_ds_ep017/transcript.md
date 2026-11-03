# T1_ds_ep017

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T17:31:41Z

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
and translation both matter; a  …[+9792 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '38,62p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
3. **Several hypotheses per frame, not one**: the baseline, the FoundationPose and CoCoNet estimates, their 180 deg
   symmetry flips about each principal axis of the mesh, poses propagated from neighbouring frames, and fresh
   registrations (e.g. render-and-compare optimisation from several initialisations, or FoundationPose
   re-registration on chosen frames).
4. **Global decoding over time**: choose one hypothesis per frame (or per segment) jointly, e.g. dynamic programming /
   Viterbi with a data cost from (2) and a transition cost that penalises rotation and translation changes (much more
   in static phases), then refine continuously, e.g. a joint optimisation of all poses with image terms plus
   smoothness / constant-velocity terms on SE(3), and rigid hand-object attachment whil …[+1474 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '63,88p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
Do not snap the object onto the floor plane: it is a fit with several-cm tolerance (see `foot_plane_dist_stats` in
`ground_plane.json`), and the object may rest on other surfaces.

## Inputs (`/work/inputs`)
- `video.mp4` - the only camera (static exo camera), 1536x1152.
- `task.txt` - what the person does. `episode.json` - object name, frame count.
- `object_mesh.glb` - the exact object mesh (metres), textured. Load it with
  `trimesh.load(path, force="scene", process=False).to_geometry()`; all poses below map THESE mesh-file
  coordinates to the target frame.
- `camera.json` - `K` (pinhole intrinsics, shared by all episodes of this camera), `T_world_cam` (4x4 camera-to-world,
  estimated from the frame-0 reference human and the floor). `K_pipeline`: the per-episode MoGe-2 K the pipeline' …[+1305 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ls -lah /work/inputs && find /work -maxdepth 2 -type f -printf '%p %s\\n' | sort | head -100"
```
<details><summary>output</summary>

```
total 51M
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 17:52 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 21:25 ..
-rw-rw-r-- 1 hanzhou hanzhou  824 Oct 20 17:52 camera.json
-rw-rw-r-- 1 hanzhou hanzhou   67 Oct 20 17:51 episode.json
-rw-rw-r-- 1 hanzhou hanzhou 191K Oct 20 17:52 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou  395 Oct 20 17:51 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou 4.4M Oct 20 17:52 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou  11M Oct 20 17:51 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou  22M Oct 20 17:52 reconstruction.npz
-rw-rw-r-- 1 hanzhou hanzhou  111 Oct 20 17:51 task.txt
-rw-rw-r-- 1 hanzhou hanzhou  14M Oct 20 17:51 video.mp4
/work/.codex_home/.sandbox_migration 3
/work/.codex_home/config.toml 361
/work/.codex_home/goals_1.sqlite 4096
/work/.codex_home/goals_ …[+1292 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat inputs/task.txt; cat inputs/episode.json; cat inputs/camera.json; cat inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk over to the traffic cone on the ground, pick up the cone with both hands and place it on the round table.
{
 "episode_index": 17,
 "object": "traffic_cone",
 "frames": 440
}{
 "K": [
  [
   820.000732421875,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   820.000732421875,
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
   841.2771606445312,
   0.0,
   768.0
  ],
  [
   0.0,
   841.277099609375,
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
   0.9907481061382687,
   0.0008407743906240135,
   0.13571102859332357,
   -0.2730447794847568
  ],
  [
   -0.006012771348203462,
   0.9992708249928022,
   0.037704971806359684,
   0.009256233953355127
  ],
  [
   -0.1355803701283782,
   -0.03817212879350883,
   0.9900307327650104,
   0.1795939773642 …[+597 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -100"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1963
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/05_object_pose.json 1772
/pipeline/.stages/06_coconet.json 2828
/pipeline/.stages/07_refinement.json 2131
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 555351256
/pipeline/depth/moge2/depth_report.json 1814
/pipeline/depth/moge2/intrinsics.pkl 47038
/pipeline/depth/moge2/raw.npy 1557135488
/pipeline/export/episode_000017/edex 562
/pipeline/export/episode_000017/wild_export.json 2648
/pipeline/foundationpose/episode_000017.pkl 128905
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 3367404 …[+548 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json
r=np.load('/work/inputs/reconstruction.npz')
for k in r.files: print(k,r[k].shape,r[k].dtype)
with h5py.File('/work/inputs/masks.h5') as f:
 print('masks',list(f.keys()))
 for k in f: print(k,f[k].shape,f[k].dtype)
PY"
```
<details><summary>output</summary>

```
T_world_mesh (440, 4, 4) float32
T_cam_mesh_foundationpose (440, 4, 4) float32
T_cam_mesh_coconet (440, 4, 4) float32
human_vertices_world (440, 4610, 3) float32
human_keypoints_world (440, 70, 3) float32
contact_prob_right_left (440, 2) float32
masks ['object', 'person']
object (440, 1152, 1536) bool
person (440, 1152, 1536) bool

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi; python - <<'PY'
import torch,cv2,trimesh,open3d
print('torch',torch.__version__,torch.cuda.is_available())
print('cv2',cv2.__version__,'trimesh',trimesh.__version__,'o3d',open3d.__version__)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 True
cv2 4.11.0 trimesh 5.1.1 o3d 0.20.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch /work/outputs && python - <<'PY'
import numpy as np, trimesh, json, h5py
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); F=len(T)
with h5py.File('/work/inputs/masks.h5') as f:
 m=np.asarray(f['object']); p=np.asarray(f['person'])
# mask stats
for name,a in [('obj',m),('person',p)]:
 area=a.sum((1,2)); ys,xs=np.where(a.reshape(F,-1)); 
 print(name,'area min med max',np.min(area),np.median(area),np.max(area))
# object bbox row extrema
for i in list(range(0,F,20))+[F-1]:
 ys,xs=np.where(m[i]);
 print(i,'area',m[i].sum(),'bbox',xs.min() if len(xs) else None,xs.max() if len(xs) else None,ys.min() if len(ys) else None,ys.max() if len(ys) else None,'cont',r['contact_prob_right_left'][i])
print('mesh')
sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(type(sc),len(sc.geometry));
for n,g in sc.geometry.items():
 print(n,g.vertices.shape,g.faces.shape,g.bounds,g.visual.kind if hasattr(g.visual,'kind') else '')
mesh=sc.to_geometry()
print('go',type(mesh),mesh.bounds, 'ext',mesh.extents,'centroid',mesh.centroid)
# pose sums
trans=T[:,:3,3]; quat=R.from_matrix(T[:,:3,:3]).as_quat()
print('trans first last\\n',trans[[0,1,2,3,4,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,439]])
# vel radians / meters between seq
reld=[]; transd=[]
for i in range(1,F):
 rr=R.from_matrix(T[i,:3,:3].T@T[i-1,:3,:3]); r …[+489 chars]
```
<details><summary>output</summary>

```
obj area min med max 0 4250.5 9892
person area min med max 28152 50342.0 83777
0 area 9735 bbox 722 814 908 1101 cont [2.7040407e-15 3.9392857e-13]
20 area 9852 bbox 721 815 908 1101 cont [3.5931951e-14 2.3432176e-10]
40 area 9680 bbox 722 814 908 1101 cont [1.0267210e-09 1.0842396e-08]
60 area 9683 bbox 722 814 908 1101 cont [2.3131844e-11 1.8275260e-10]
80 area 9687 bbox 722 814 908 1101 cont [3.0131370e-11 2.1446826e-10]
100 area 9740 bbox 721 814 909 1102 cont [4.6634168e-09 5.0658794e-08]
120 area 9487 bbox 722 814 908 1101 cont [0.8160752 0.141988 ]
140 area 8370 bbox 744 842 724 908 cont [0.9999902  0.99999964]
160 area 5892 bbox 790 897 566 719 cont [0.99874276 0.99980843]
180 area 3654 bbox 797 865 492 624 cont [0.99999666 0.60264856]
200 area 0 bbox None None None None cont [0.99 …[+5461 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, h5py, numpy as np, os
cap=cv2.VideoCapture('/work/inputs/video.mp4'); F=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); print(F)
with h5py.File('/work/inputs/masks.h5') as f:
 obj=np.asarray(f['object']); per=np.asarray(f['person'])
idxs=list(range(0,440,20))+[439]
thumbs=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); assert ok
 # crop broad region x450-900,y400-1152 maybe resize
 om=obj[i]; pm=per[i]
 ov=im.copy(); ov[om]=(0.3*ov[om]+0.7*np.array([255,0,255])).astype(np.uint8)
 ov[pm & ~om]=(0.75*ov[pm & ~om]+0.25*np.array([0,255,0])).astype(np.uint8)
 # put frame label
 cv2.putText(ov,str(i),(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
 thumbs.append(cv2.resize(ov,(384,288)))
rows=[]
for j in range(0,len(thumbs),4):
 row=thumbs[j:j+4]
 while len(row)<4: row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/contact.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,90])
PY"
```
<details><summary>output</summary>

```
440

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
for i in [0,100,111,112,113,114,115,116,117,118,120,125,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,400,439]:
 M=T[i]; q=R.from_matrix(M[:3,:3]).as_quat(); e=R.from_matrix(M[:3,:3]).as_euler('xyz',degrees=True)
 print(i,'t',np.round(M[:3,3],5),'euler',np.round(e,2),'mesh_z',np.round(M[:3,2],3),'mesh_x',np.round(M[:3,0],3))
PY"
```
<details><summary>output</summary>

```
0 t [0.01935 1.18453 2.03552] euler [89.82 88.79 -3.19] mesh_z [-0.052 -0.999  0.   ] mesh_x [ 0.021 -0.001 -1.   ]
100 t [0.02025 1.18814 2.04123] euler [89.82 88.79 -3.19] mesh_z [-0.052 -0.999  0.   ] mesh_x [ 0.021 -0.001 -1.   ]
111 t [0.02025 1.18814 2.04123] euler [89.82 88.79 -3.19] mesh_z [-0.052 -0.999  0.   ] mesh_x [ 0.021 -0.001 -1.   ]
112 t [0.0201  1.18751 2.04023] euler [89.82 88.79 -3.19] mesh_z [-0.052 -0.999  0.   ] mesh_x [ 0.021 -0.001 -1.   ]
113 t [0.02004 1.18729 2.03988] euler [89.82 88.79 -3.19] mesh_z [-0.052 -0.999  0.   ] mesh_x [ 0.021 -0.001 -1.   ]
114 t [0.02004 1.18729 2.03988] euler [89.82 88.79 -3.19] mesh_z [-0.052 -0.999  0.   ] mesh_x [ 0.021 -0.001 -1.   ]
115 t [0.02004 1.18729 2.03988] euler [89.82 88.79 -3.19] mesh_z [-0.052 -0.999  0.   ] mesh_x …[+3578 chars]
```
</details>

**shell** (exit 2)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, os
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
for n,g in s.geometry.items():
 print(g.visual)
 v=g.visual
 print('uv',g.visual.uv.shape if hasattr(g.visual,'uv') else None)
 if hasattr(v,'material'):
  m=v.material
  print(type(m),vars(m).keys())
  print('base',getattr(m,'baseColorTexture',None),'img',getattr(m,'image',None))
  img=getattr(m,'image',None)
  if img is not None:
   print(img.size,img.mode); img.save('/work/scratch/texture.png')
PY
ls -lh /work/scratch/texture.png"
```
<details><summary>output</summary>

```
<trimesh.visual.texture.TextureVisuals object at 0x7407b8b83750>
uv (282928, 2)
<class 'trimesh.visual.material.PBRMaterial'> dict_keys(['_data'])
base <PIL.PngImagePlugin.PngImageFile image mode=RGB size=2232x2280 at 0x74077F10BB50> img None
ls: cannot access '/work/scratch/texture.png': No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=next(iter(s.geometry.values())); im=g.visual.material.baseColorTexture; im.save('/work/scratch/texture.png')
print(im.size)
PY"
```
<details><summary>output</summary>

```
(2232, 2280)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/.stages/05_object_pose.json; cat /pipeline/.stages/06_coconet.json; cat /pipeline/.stages/07_refinement.json; cat /pipeline/.stages/08_render.json'
```
<details><summary>output</summary>

```
{
  "name": "05_object_pose",
  "outputs": [
    {
      "mtime_ns": 1791258848947595834,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000017/foundationpose/episode_000017.pkl",
      "size": 128905
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000017/export/episode_000017",
      "--depth-root",
      "/home/hanzhou/Projects/v2d/run/out/episode_000017/depth/moge2/aligned.h5",
      "--out-file",
      "/home/hanzhou/Projects/v2d/run/out/episode_000017/foundationpose/episode_000017.pkl",
      "--camera-id",
      "0",
      "--iteration",
      "5",
      "--max-attempts",
      "5",
 …[+8457 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'find /code/reconstruction -maxdepth 3 -type f | sort | head -100'
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
/code/reconstruction/.venv/bin/numpy-config …[+4097 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for path in ['/pipeline/inference/materialized_inputs.h5','/pipeline/depth/moge2/aligned.h5','/pipeline/work/masks_k0.h5']:
 print('\\n',path)
 with h5py.File(path) as f:
  def walk(name,obj):
   if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype)
  f.visititems(walk)
PY"
```
<details><summary>output</summary>

```
K_rois (440, 3, 3) float32
bboxes (440, 4) float32
identity_json () object
input_rgbs (440, 3, 224, 224) float32
input_xyz (440, 6, 224, 224) float32
observed_human_masks (440, 224, 224) bool
observed_object_masks (440, 224, 224) bool
render_rgbs (440, 3, 224, 224) float32
render_xyz (440, 6, 224, 224) float32

 /pipeline/depth/moge2/aligned.h5
aligned/front_stereo_camera_left (440,) object
alignment/front_stereo_camera_left/scale (440,) float32
alignment/front_stereo_camera_left/shift (440,) float32
alignment/front_stereo_camera_left/valid_count (440,) int32
frame_names/front_stereo_camera_left (440,) object
raw/front_stereo_camera_left (440,) object

 /pipeline/work/masks_k0.h5
episode_000017/000000-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000017/000000-k0.person_mask.png (1152, 1 …[+50347 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffprobe -v error -show_entries stream=width,height,nb_frames,duration -of default=nw=1 /pipeline/inference/episode_000017_step200000_before_after.mp4; ffprobe -v error -show_entries stream=width,height,nb_frames,duration -of default=nw=1 /pipeline/inference/episode_000017_step200000_reconstruction.mp4'
```
<details><summary>output</summary>

```
width=4608
height=1236
duration=14.666667
nb_frames=440
width=3072
height=1152
duration=14.666667
nb_frames=440

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
for p in ['/pipeline/foundationpose/episode_000017.pkl','/pipeline/mhr_init/episode_000017.pkl']:
 d=pickle.load(open(p,'rb')); print('\\n',p,type(d))
 if isinstance(d,dict):
  for k,v in d.items():
   import numpy as np
   if hasattr(v,'shape'): print(k,v.shape,v.dtype)
   else: print(k,type(v),str(v)[:300])
 else: print(str(d)[:1000])
PY"
```
<details><summary>output</summary>

```

 /pipeline/foundationpose/episode_000017.pkl <class 'dict'>
frames <class 'list'> ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029',
camera_id <class 'int'> 0
fp_poses_cam (440, 1, 4, 4) float32
fp_candidate_count (440,) int32
fp_selected_candidate_index (440,) int32
fp_selected_iou (440,) float32
fp_selected_visibility (440,) float32
fp_retry_count (440,) int32
fp_reliable (440,) bool
fp_oracle_rotation_error_deg (440,) float32
fp_translation_threshold_m (440,) float32
fp_selected_translation_distance_m (440,) float32
gt_object_visibility_rat …[+2122 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth','/pipeline/inference/refinement_checkpoint.pth']:
 d=torch.load(p,map_location='cpu',weights_only=False)
 print('\\n',p,type(d))
 if isinstance(d,dict):
  print(d.keys())
  for k,v in d.items():
   if hasattr(v,'shape'): print(k,v.shape,v.dtype)
   else: print(k,type(v),str(v)[:300])
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth <class 'dict'>
dict_keys(['schema', 'frames', 'frame_meta', 'kid', 'checkpoint', 'mhr_inference', 'config', 'mesh_diameter', 'mhr_neutral_height_init', 'mhr_spatial_scale', 'K_rois', 'bboxes', 'faces', 'pr', 'pr_initial', 'in', 'gt', 'observations', 'raw', 'metadata'])
schema <class 'str'> cari4d.mhr_wild_inference.v1
frames <class 'list'> ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029',
frame_meta <class 'list'> [{'frame': '000000', 'src_frame': 0, 'kid': 0}, {'frame': '000001', 'src_frame': 1, 'kid': 0}, {' …[+9680 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch, numpy as np
r=np.load('/work/inputs/reconstruction.npz'); Tb=r['T_world_mesh']
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 d=torch.load(p,map_location='cpu',weights_only=False)
 print('\\n',p)
 for k in ['pr','pr_initial','in','raw','postopt']:
  if k not in d: continue
  print(k,'keys',d[k].keys())
  for kk,v in d[k].items():
   if hasattr(v,'shape'): print(' ',kk,v.shape,v.dtype, 'range', float(v.min()),float(v.max()))
   else: print(' ',kk,type(v),str(v)[:200])
 # compare pose_abs mapping against baseline world after possible S
 if 'pr' in d:
  pa=np.asarray(d['pr']['pose_abs'])
  print('pr pose first',pa[0])
 print('in first',np.asarray(d['in']['pose_abs'])[0])
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
pr keys dict_keys(['pose_abs', 'pose_abs_1st_delta', 'mhr_global_rot6d', 'mhr_trans', 'mhr_body_pose_cont', 'mhr_hand', 'mhr_shape', 'mhr_scale', 'mhr_face', 'contact_logits'])
  pose_abs (440, 4, 4) float32 range -0.9998784065246582 3.6640491485595703
  pose_abs_1st_delta (440, 4, 4) float32 range -0.9998784065246582 3.6640491485595703
  mhr_global_rot6d (440, 6) float32 range -0.996924877166748 0.9999037981033325
  mhr_trans (440, 3) float32 range -0.2516537606716156 3.4845592975616455
  mhr_body_pose_cont (440, 260) float32 range -0.6161454916000366 1.288469910621643
  mhr_hand (440, 108) float32 range -0.5791059136390686 0.49284887313842773
  mhr_shape (440, 45) float32 range -0.9261484146118164 0.8568246364593506
  mhr_scale (440, 28) float32 range -3 …[+9717 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, torch, json
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
for i in [0,100,120,200,300,400]:
 print('\\ni',i)
 for name in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
  A=r[name][i].astype(float)
  print(name);print(np.round(A,4))
 print('derived');print(np.round(Tc[i],4))
PY"
```
<details><summary>output</summary>

```

i 0
T_cam_mesh_foundationpose
[[ 0.2651  0.9642 -0.004  -0.0062]
 [ 0.1301 -0.0399 -0.9907  1.0992]
 [-0.9554  0.2621 -0.136   2.0685]
 [ 0.      0.      0.      1.    ]]
T_cam_mesh_coconet
[[ 0.1653  0.9854 -0.0406  0.0078]
 [ 0.0626 -0.0515 -0.9967  1.0654]
 [-0.9843  0.1622 -0.0702  1.9674]
 [ 0.      0.      0.      1.    ]]
derived
[[ 0.1564  0.9866 -0.046   0.031 ]
 [ 0.037  -0.0524 -0.9979  1.1038]
 [-0.987   0.1544 -0.0447  1.9214]
 [ 0.      0.      0.      1.    ]]

i 100
T_cam_mesh_foundationpose
[[ 0.3982  0.9173 -0.0062 -0.0069]
 [ 0.1295 -0.0629 -0.9896  1.0853]
 [-0.9081  0.3932 -0.1438  2.0506]
 [ 0.      0.      0.      1.    ]]
T_cam_mesh_coconet
[[ 0.1653  0.9854 -0.0406  0.0078]
 [ 0.0626 -0.0515 -0.9967  1.0654]
 [-0.9843  0.1622 -0.0702  1.9674]
 [ 0.      0.      0. …[+2224 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f: obj=np.asarray(f['object'])
ims=[]
for i in [0,112,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340,439]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 ys,xs=np.where(obj[i])
 if len(xs)==0:
  # hand approximate crop
  crop=im[350:750,500:900]
 else:
  x0,x1=max(0,xs.min()-50),min(1536,xs.max()+50); y0,y1=max(0,ys.min()-50),min(1152,ys.max()+50)
  crop=im[y0:y1,x0:x1]
 crop=cv2.resize(crop,(240,300))
 cv2.putText(crop,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
 ims.append(crop)
rows=[np.hstack(ims[i:i+5]) for i in range(0,len(ims),5)]
cv2.imwrite('/work/scratch/crops.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 18, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/shape_base.py", line 289, in vstack
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1200 and the array at index 3 has size 960

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f: obj=np.asarray(f['object'])
ims=[]
for i in [0,112,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340,439]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 ys,xs=np.where(obj[i])
 if len(xs)==0: crop=im[350:750,500:900]
 else:
  x0,x1=max(0,xs.min()-50),min(1536,xs.max()+50); y0,y1=max(0,ys.min()-50),min(1152,ys.max()+50)
  crop=im[y0:y1,x0:x1]
 crop=cv2.resize(crop,(240,300))
 cv2.putText(crop,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
 ims.append(crop)
rows=[]
for j in range(0,len(ims),5):
 row=ims[j:j+5]
 while len(row)<5: row.append(np.zeros((300,240,3),np.uint8))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/crops.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
try:
 import nvdiffrast.torch as dr
 print('nvdiffrast',dr)
 print(dir(dr))
except Exception as e: print(type(e),e)
PY"
```
<details><summary>output</summary>

```
nvdiffrast <module 'nvdiffrast.torch' from '/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/__init__.py'>
['DepthPeeler', 'RasterizeCudaContext', 'RasterizeGLContext', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'antialias', 'antialias_construct_topology_hash', 'get_log_level', 'interpolate', 'ops', 'rasterize', 'set_log_level', 'texture', 'texture_construct_mip']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_render.py <<'PY'
import torch, numpy as np, trimesh, json, cv2, h5py
import nvdiffrast.torch as dr
from PIL import Image
D='cuda'
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
with h5py.File('/work/inputs/masks.h5') as f: mobj=f['object'][0]; mper=f['person'][0]
sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); g=sc.to_geometry()
v=torch.tensor(g.vertices,dtype=torch.float32,device=D); f=torch.tensor(g.faces,dtype=torch.int32,device=D)
# image crop coords x620:880,y850:1130
x0,x1,y0,y1=650,880,880,1130; CW=x1-x0; CH=y1-y0
gl=dr.RasterizeCudaContext(device=D)
def render(Tw):
 M=torch.tensor(Tw,dtype=torch.float32,device=D)
 vw=v@M[:3,:3].T+M[:3,3]
 vc=vw@torch.tensor(np.linalg.inv(Twc)[:3,:3].T,dtype=torch.float32,device=D)+torch.tensor(np.linalg.inv(Twc)[:3,3],dtype=torch.float32,device=D)
 z=vc[:,2]
 # local crop pixels projections
 x=K[0,0]*vc[:,0]/z+K[0,2]; y=K[1,1]*vc[:,1]/z+K[1,2]
 sx=(x-x0)/CW*2-1; sy=1-(y-y0)/CH*2
 clip=torch.stack([sx,sy,vc[:,2],vc[:,2]],1)[None]
 rast, _=dr.rasterize(gl,clip,f,resolution=(CH,CW))
 mask=rast[0,...,3]>0
 return mask.cpu().numpy()
for i in [0,100,120,140,180,260,320,400,439]:
 o=render(T[i])
 gt=mobj[y0:y1,x0:x1]
 inter=(o&gt).sum(); union=(o|gt).sum(); print(i,o.sum(),gt.sum(),inter/max(union,1),(o&~gt).sum(),(gt …[+315 chars]
```
<details><summary>output</summary>

```
0 2760 9735 0.24056791104050834 337 7312
100 2730 9735 0.2378351539225422 335 7340
120 2576 9735 0.2268061783756851 300 7459
140 1823 9735 0.12115627121932292 574 8486
180 0 9735 0.0 0 9735
260 89 9735 0.0 89 9735
320 0 9735 0.0 0 9735
400 73 9735 0.0 73 9735
439 65 9735 0.0 65 9735

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=sc.to_geometry();print(g.vertices.min(0),g.vertices.max(0));print(sc.graph)
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'][0];cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);K=np.array(cam['K'])
X=g.vertices@T[:3,:3].T+T[:3,3]; Xc=X@np.linalg.inv(Twc)[:3,:3].T+np.linalg.inv(Twc)[:3,3]
uv=K@ (Xc/Xc[:,2:3]).T; uv=uv[:2].T
print(uv.min(0),uv.max(0),Xc[:,2].min(),Xc[:,2].max())
PY"
```
<details><summary>output</summary>

```
[-0.09974834 -0.1046155  -0.09771387] [0.10502636 0.09852374 0.28576073]
<trimesh.scene.transforms.SceneGraph object at 0x7aa2591992d0>
[724.76509173 911.90944163] [ 823.03466905 1107.91699987] 1.8084167811758287 2.0366619704176245

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/test_render.py').read().split('for i in')[0])
o=render(T[0]); cv2.imwrite('/work/scratch/rawmask.png',(o*255).astype(np.uint8))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json,cv2,nvdiffrast.torch as dr
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0]
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=torch.tensor(g.vertices,dtype=torch.float32,device='cuda');f=torch.tensor(g.faces,dtype=torch.int32,device='cuda')
M=torch.tensor(T,dtype=torch.float32,device='cuda');vw=v@M[:3,:3].T+M[:3,3]
Tc=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');vc=vw@Tc[:3,:3].T+Tc[:3,3];z=vc[:,2]
x=K[0,0]*vc[:,0]/z+K[0,2]; y=K[1,1]*vc[:,1]/z+K[1,2]
clip=torch.stack([x/W*2-1,1-y/H*2,vc[:,2],vc[:,2]],1)[None]
gl=dr.RasterizeCudaContext(device='cuda');rast,_=dr.rasterize(gl,clip,f,resolution=(H,W));m=rast[0,...,3]>0
print(m.sum().item(),rast[0,...,3].min().item(),rast[0,...,3].max().item())
cv2.imwrite('/work/scratch/fullmask.png',(m.cpu().numpy()*255).astype(np.uint8))
PY"
```
<details><summary>output</summary>

```
3453 0.0 94147.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
m=cv2.imread('/work/scratch/fullmask.png',0);cv2.imwrite('/work/scratch/fullmaskcrop.png',m[880:1130,650:880]);print((m[880:1130,650:880]>0).sum())
PY"
```
<details><summary>output</summary>

```
0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
m=cv2.imread('/work/scratch/fullmask.png',0)>0
ys,xs=np.where(m);print(len(xs),xs.min(),xs.max(),ys.min(),ys.max())
PY"
```
<details><summary>output</summary>

```
3453 746 797 282 400

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/1-y\\/H\\*2/y\\/H*2-1/' /dev/stdin <<'EOF' >/dev/null
EOF
python - <<'PY'
# quick rerender both y signs full
import torch,numpy as np,trimesh,json,cv2,nvdiffrast.torch as dr
W,H=1536,1152;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0]
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(g.vertices,dtype=torch.float32,device='cuda');f=torch.tensor(g.faces,dtype=torch.int32,device='cuda');M=torch.tensor(T,dtype=torch.float32,device='cuda');vw=v@M[:3,:3].T+M[:3,3];Tc=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');vc=vw@Tc[:3,:3].T+Tc[:3,3];z=vc[:,2];x=K[0,0]*vc[:,0]/z+K[0,2];y=K[1,1]*vc[:,1]/z+K[1,2]
gl=dr.RasterizeCudaContext(device='cuda')
for s in [1,-1]:
 clip=torch.stack([x/W*2-1,s*(y/H*2-1),z,z],1)[None];rast,_=dr.rasterize(gl,clip,f,resolution=(H,W));m=rast[0,...,3]>0;ys,xs=np.where(m.cpu().numpy());print(s,m.sum().item(),xs.min(),xs.max(),ys.min(),ys.max())
PY"
```
<details><summary>output</summary>

```
1 3453 746 797 751 869
-1 3453 746 797 282 400

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json,cv2,nvdiffrast.torch as dr
W,H=1536,1152;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0]
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(g.vertices,dtype=torch.float32,device='cuda');f=torch.tensor(g.faces,dtype=torch.int32,device='cuda');M=torch.tensor(T,dtype=torch.float32,device='cuda');vw=v@M[:3,:3].T+M[:3,3];Tc=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');vc=vw@Tc[:3,:3].T+Tc[:3,3];z=vc[:,2];x=K[0,0]*vc[:,0]/z+K[0,2];y=K[1,1]*vc[:,1]/z+K[1,2]
ndx=x/W*2-1; ndy=1-y/H*2
clip=torch.stack([ndx*z,ndy*z,z,z],1)[None];gl=dr.RasterizeCudaContext(device='cuda');rast,_=dr.rasterize(gl,clip,f,resolution=(H,W));m=rast[0,...,3]>0;ys,xs=np.where(m.cpu().numpy());print(m.sum().item(),xs.min(),xs.max(),ys.min(),ys.max());cv2.imwrite('/work/scratch/fullmask.png',(m.cpu().numpy()*255).astype(np.uint8))
PY"
```
<details><summary>output</summary>

```
9624 725 822 44 239

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/ndy=1-y\\/H\\*2/ndy=y\\/H*2-1/' /work/scratch/test_render.py 2>/dev/null || true
python - <<'PY'
import torch,numpy as np,trimesh,json,cv2,nvdiffrast.torch as dr
W,H=1536,1152;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0]
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(g.vertices,dtype=torch.float32,device='cuda');f=torch.tensor(g.faces,dtype=torch.int32,device='cuda');M=torch.tensor(T,dtype=torch.float32,device='cuda');vw=v@M[:3,:3].T+M[:3,3];Tc=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');vc=vw@Tc[:3,:3].T+Tc[:3,3];z=vc[:,2];x=K[0,0]*vc[:,0]/z+K[0,2];y=K[1,1]*vc[:,1]/z+K[1,2]
ndx=x/W*2-1; ndy=y/H*2-1
clip=torch.stack([ndx*z,ndy*z,z,z],1)[None];gl=dr.RasterizeCudaContext(device='cuda');rast,_=dr.rasterize(gl,clip,f,resolution=(H,W));m=rast[0,...,3]>0;ys,xs=np.where(m.cpu().numpy());print(m.sum().item(),xs.min(),xs.max(),ys.min(),ys.max());cv2.imwrite('/work/scratch/fullmask.png',(m.cpu().numpy()*255).astype(np.uint8))
PY"
```
<details><summary>output</summary>

```
9624 725 822 912 1107

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json,cv2,h5py,nvdiffrast.torch as dr
W,H=1536,1152;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tc=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz');Ts=r['T_world_mesh'].astype(np.float32);perc=r['contact_prob_right_left']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(g.vertices,dtype=torch.float32,device='cuda');f=torch.tensor(g.faces,dtype=torch.int32,device='cuda')
gl=dr.RasterizeCudaContext(device='cuda')
with h5py.File('/work/inputs/masks.h5') as h: gtobj=np.asarray(h['object']); gtper=np.asarray(h['person'])
ious=[]
for i,T in enumerate(Ts):
 M=torch.tensor(T,device='cuda');vw=v@M[:3,:3].T+M[:3,3];vc=vw@torch.tensor(Tc[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(Tc[:3,3],dtype=torch.float32,device='cuda');z=vc[:,2];x=K[0,0]*vc[:,0]/z+K[0,2];y=K[1,1]*vc[:,1]/z+K[1,2]
 clip=torch.stack([(x/W*2-1)*z,(y/H*2-1)*z,z,z],1)[None];rast,_=dr.rasterize(gl,clip,f,resolution=(H,W));pred=(rast[0,...,3]>0).cpu().numpy();gt=gtobj[i]; union=(pred|gt).sum(); iou=(pred&gt).sum()/union if union else 0
 ious.append(iou)
print('iou quant',np.quantile(ious,[0,.1,.25,.5,.75,.9,1]))
for i in range(0,440,10): print(i,round(ious[i],3),int(gtobj[i].sum()),round(float(perc[i,0]),2),round(float(perc[i,1]),2))
np.save('/work/scratch/baseline_iou.npy',ious)
PY"
```
<details><summary>output</summary>

```
iou quant [0.         0.         0.68245743 0.81769527 0.82451419 0.83974067
 0.85894436]
0 0.814 9735 0.0 0.0
10 0.822 9847 0.0 0.0
20 0.822 9852 0.0 0.0
30 0.82 9644 0.0 0.0
40 0.82 9680 0.0 0.0
50 0.819 9679 0.0 0.0
60 0.82 9683 0.0 0.0
70 0.819 9685 0.0 0.0
80 0.819 9687 0.0 0.0
90 0.822 9818 0.0 0.0
100 0.805 9740 0.0 0.0
110 0.82 9813 0.0 0.0
120 0.821 9487 0.82 0.14
130 0.717 9050 1.0 1.0
140 0.671 8370 1.0 1.0
150 0.818 6801 1.0 1.0
160 0.784 5892 1.0 1.0
170 0.617 4274 1.0 0.99
180 0.678 3654 1.0 0.6
190 0.23 837 1.0 0.77
200 0.0 0 1.0 1.0
210 0.0 0 1.0 1.0
220 0.0 0 1.0 1.0
230 0.0 0 1.0 1.0
240 0.0 0 1.0 1.0
250 0.569 1948 1.0 1.0
260 0.53 2214 1.0 1.0
270 0.562 2213 1.0 1.0
280 0.771 3178 1.0 1.0
290 0.804 3592 1.0 1.0
300 0.788 3820 1.0 1.0
310 0.723 3651 1.0 1.0
320 0.741 409 …[+263 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);_,im=cap.read()
m=cv2.imread('/work/scratch/fullmask.png',0)>0
with h5py.File('/work/inputs/masks.h5') as f:g=f['object'][0]
out=im.copy();out[m]=(out[m]*.2+np.array([0,255,0])*.8).astype(np.uint8);out[g]=(out[g]*.5+np.array([255,0,255])*.5).astype(np.uint8)
cv2.imwrite('/work/scratch/overlay.jpg',out[870:1130,690:850],[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(dr.texture.__doc__)
print(dr.interpolate.__doc__)
PY"
```
<details><summary>output</summary>

```
Perform texture sampling.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Args:
        tex: Texture tensor with dtype `torch.float32`. For 2D textures, must have shape
             [minibatch_size, tex_height, tex_width, tex_channels]. For cube map textures,
             must have shape [minibatch_size, 6, tex_height, tex_width, tex_channels] where
             tex_width and tex_height are equal. Note that `boundary_mode` must also be set
             to 'cube' to enable cube map mode. Broadcasting is supported along the minibatch axis.
        uv: Tensor containing per-pixel texture coordinates. When sampling a 2D texture,
            must have shape [minibatch_size, height, width, 2]. When sampli …[+4527 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"dr.texture\\|texture(\" -n /code/reconstruction/modules/v2d_cari4d /workspace 2>/dev/null | head -50"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py:193:    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py:236:    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py:271:    color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
/workspace/third_party/kaolin/build/lib.linux-x86_64-cpython-311/kaolin/experimental/dash3d/static/thirdparty.js:8:!function(t,e){"object"==typeof exports&&"undefined"!=typeof module?e(exports):"function"==typeof define&&define.amd?define(["exports"],e):e((t="undefined"!=typeof globalThis?globalThis:t||self).THREE={})}(this,(function(t){"use strict";const e="137",n=100,i=300,r=301,s=302,a= …[+1047806 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch, numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float)
# yaw decomposition about world -y approx: matrix cone axis mesh z mapped to world -y. calculate angle of projection mesh x onto x-z plane
for i in [0,100,112,117,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,400,439]:
 # plane angle atan2(z,x)
 ax=T[i,:3,0]; az=T[i,:3,2]
 print(i, round(np.degrees(np.arctan2(ax[2],ax[0])),1),round(np.degrees(np.arctan2(az[2],az[0])),1),np.round(T[i,:,3],3))
PY"
```
<details><summary>output</summary>

```
0 -88.8 179.9 [0.019 1.185 2.036 1.   ]
100 -88.8 179.9 [0.02  1.188 2.041 1.   ]
112 -88.8 179.9 [0.02  1.188 2.04  1.   ]
117 0.1 169.6 [0.015 1.195 2.056 1.   ]
120 0.7 166.0 [0.017 1.202 2.063 1.   ]
130 12.2 131.7 [0.025 1.067 2.067 1.   ]
140 59.5 164.1 [0.079 0.772 2.092 1.   ]
150 59.9 153.2 [0.151 0.553 2.135 1.   ]
160 69.4 174.2 [0.256 0.372 2.291 1.   ]
170 82.3 141.7 [0.239 0.237 2.47  1.   ]
180 91.6 -161.8 [0.28  0.191 2.656 1.   ]
190 124.7 111.3 [0.356 0.194 2.906 1.   ]
200 135.0 -122.1 [0.424 0.144 3.009 1.   ]
210 113.6 -123.9 [0.418 0.155 3.124 1.   ]
220 -58.3 -108.0 [0.367 0.23  3.316 1.   ]
230 -42.2 -158.5 [0.374 0.102 3.371 1.   ]
240 -86.0 -68.0 [0.287 0.158 3.618 1.   ]
250 -114.7 -13.4 [0.053 0.153 3.662 1.   ]
260 -100.1 -91.6 [-0.138  0.172  3.627  1.   ]
270 …[+466 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for i in range(108,136):
 ax=T[i,:3,0]; az=T[i,:3,2]
 print(i, f'yaw={np.degrees(np.arctan2(ax[2],ax[0])):7.2f}', 't',np.round(T[i,:3,3],4),'relprev', np.degrees(np.linalg.norm(R.from_matrix(T[i,:3,:3].T@T[i-1,:3,:3]).as_rotvec())) if i else 0)
PY"
```
<details><summary>output</summary>

```
108 yaw= -88.80 t [0.0203 1.1881 2.0412] relprev 1.2898261792893e-14
109 yaw= -88.80 t [0.0203 1.1881 2.0412] relprev 1.2898261792893e-14
110 yaw= -88.80 t [0.0203 1.1881 2.0412] relprev 1.2898261792893e-14
111 yaw= -88.80 t [0.0203 1.1881 2.0412] relprev 1.2898261792893e-14
112 yaw= -88.80 t [0.0201 1.1875 2.0402] relprev 1.2898261792893e-14
113 yaw= -88.80 t [0.02   1.1873 2.0399] relprev 1.2898261792893e-14
114 yaw= -88.80 t [0.02   1.1873 2.0399] relprev 1.2898261792893e-14
115 yaw= -88.80 t [0.02   1.1873 2.0399] relprev 1.2898261792893e-14
116 yaw= -88.80 t [0.02   1.1873 2.0399] relprev 1.2898261792893e-14
117 yaw=   0.07 t [0.0148 1.1949 2.0558] relprev 88.86023710554727
118 yaw=   0.14 t [0.0144 1.1957 2.0574] relprev 0.7989984042639297
119 yaw=  -0.09 t [0.0157 1.1982 2.058 ] rel …[+1108 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in range(108,137,2):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();crop=im[840:1130,680:860];crop=cv2.resize(crop,(270,435));cv2.putText(crop,str(i),(5,35),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3);ims.append(crop)
while len(ims)%7:ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/pick.jpg',np.vstack([np.hstack(ims[j:j+7]) for j in range(0,len(ims),7)]),[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '130,290p' /code/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py"
```
<details><summary>output</summary>

```
  if glctx is None:
    if context == 'gl':
      glctx = dr.RasterizeGLContext()
    elif context=='cuda':
      glctx = dr.RasterizeCudaContext()
    else:
      raise NotImplementedError
    logging.info("created context")

  if mesh_tensors is None:
    mesh_tensors = make_mesh_tensors(mesh)
  pos = mesh_tensors['pos']
  vnormals = mesh_tensors['vnormals']
  pos_idx = mesh_tensors['faces']
  has_tex = 'tex' in mesh_tensors
  batched_positions = pos.ndim == 3
  if batched_positions and pos.shape[0] != ob_in_cams.shape[0]:
    raise ValueError(f'Batched mesh positions and camera transforms differ: {pos.shape[0]} != {ob_in_cams.shape[0]}')
  if pos.ndim not in (2, 3):
    raise ValueError(f'Mesh positions must have shape [V,3] or [B,V,3], got {tuple(pos.shape)}')

  ob_in_glcams = torch.t …[+6184 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,130p' /code/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py; grep -R \"def uv_idx\\|uv_idx\" -n /code/reconstruction/modules/v2d_cari4d/lib/cari4d | head -20"
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


import os, sys, torch, trimesh, logging, importlib, scipy
import nvdiffrast.torch as dr
import torch.nn.functional as F
import torchvision
from functools import lru_cache
import open3d as o3d
import cv2
import numpy as np
from collections import defaultdict
from transformations import *
from collections import OrderedDict
import ruamel.yaml
yaml = ruamel.yaml.YAML()
cod …[+5780 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();m=g.visual.material
print('image',getattr(m,'image','no'),'base',getattr(m,'baseColorTexture','no'))
print(hasattr(m,'image'))
PY"
```
<details><summary>output</summary>

```
image no base <PIL.Image.Image image mode=RGB size=2232x2280 at 0x7D7EF326FC90>
False

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_tex.py <<'PY'
import torch,numpy as np,trimesh,json,cv2,nvdiffrast.torch as dr, h5py
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tc=np.linalg.inv(np.array(cam['T_world_cam'])).astype(np.float32)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
img=np.array(g.visual.material.baseColorTexture.convert('RGB'))
tex=torch.tensor(img,dtype=torch.float32,device='cuda')[None]/255
v=torch.tensor(g.vertices,dtype=torch.float32,device='cuda');uv=torch.tensor(g.visual.uv,dtype=torch.float32,device='cuda');uv[:,1]=1-uv[:,1]
faces=torch.tensor(g.faces,dtype=torch.int32,device='cuda')
gl=dr.RasterizeCudaContext(device='cuda')
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def render(i,x0=650,x1=880,y0=880,y1=1130):
 cw,ch=x1-x0,y1-y0;M=torch.tensor(T[i],dtype=torch.float32,device='cuda');vw=v@M[:3,:3].T+M[:3,3];vc=vw@torch.tensor(Tc[:3,:3],device='cuda').T+torch.tensor(Tc[:3,3],device='cuda');z=vc[:,2];xp=K[0,0]*vc[:,0]/z+K[0,2];yp=K[1,1]*vc[:,1]/z+K[1,2]
 sx=(xp-x0)/cw*2-1;sy=(yp-y0)/ch*2-1
 clip=torch.stack([sx*z,sy*z,z,z],1)[None]
 rast,_=dr.rasterize(gl,clip,faces,resolution=(ch,cw));uvm,_=dr.interpolate(uv,rast,faces);col=dr.texture(tex,uvm,filter_mode='linear');mask=(rast[0,...,3]>0)
 col=col[0].cpu().numpy(); mask=mask.cpu().numpy() # this is y down? likely raster image display direct
 return col,mask
if __name__=='__main__':
 for i in [0 …[+428 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
for i in [0,112,118,140,200,260,320,400,439]:
 col=cv2.imread(f'/work/scratch/tex{i}.png') # this BGR file actually RGB values stored -> values swapped but imread returns raw BGR = original RGB
 # render function col is RGB. img imread BGR from raw saved RGB, so col_bgr = col[..., ::-1]
 out=col[...,::-1]
 cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();crop=im[880:1130,650:880].copy()
 # mask black
 m=(out.sum(2)>0);crop[m]=(.4*crop[m]+.6*out[m]).astype('uint8');cv2.imwrite(f'/work/scratch/texov{i}.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_score.py <<'PY'
import torch,numpy as np,trimesh,json,cv2,nvdiffrast.torch as dr,h5py
from scipy.spatial.transform import Rotation as R
W,H=1536,1152
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tc=np.linalg.inv(np.array(cam['T_world_cam'])).astype(np.float32);Tc_t=torch.tensor(Tc,device='cuda')
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();tex=np.array(g.visual.material.baseColorTexture.convert('RGB'));tex_t=torch.tensor(tex,dtype=torch.float32,device='cuda')[None]/255
v=torch.tensor(g.vertices,dtype=torch.float32,device='cuda');uv=torch.tensor(g.visual.uv,dtype=torch.float32,device='cuda');uv[:,1]=1-uv[:,1];faces=torch.tensor(g.faces,dtype=torch.int32,device='cuda');gl=dr.RasterizeCudaContext(device='cuda')
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
with h5py.File('/work/inputs/masks.h5') as h:objmask=np.asarray(h['object']);permask=np.asarray(h['person'])
cap=cv2.VideoCapture('/work/inputs/video.mp4');frames={}
def getframe(i):
 if i not in frames: cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,frames[i]=cap.read()
 return frames[i]
def render_pose(M,x0,x1,y0,y1):
 cw,ch=x1-x0,y1-y0;M=torch.tensor(M,dtype=torch.float32,device='cuda');vw=v@M[:3,:3].T+M[:3,3];vc=vw@Tc_t[:3,:3].T+Tc_t[:3,3];z=vc[:,2];xp=K[0,0]*vc[:,0]/z+K[0,2];yp=K[1,1]*vc[:,1]/z+K[1,2]
 sx=(xp-x0)/cw*2-1;sy=(yp-y0)/ch*2-1;clip=torch.stack([sx*z,sy*z,z,z],1)[None];rast,_=dr.rasterize …[+1310 chars]
```
<details><summary>output</summary>

```
0 -180 (0.3997186898087049, 0.10378828270169511, 8198)
0 -170 (0.3768583094883285, 0.1051903414844204, 8203)
0 -160 (0.3707365923501534, 0.10517522753747038, 8188)
0 -150 (0.36404046306749877, 0.10562045998198732, 8197)
0 -140 (0.36059437262553234, 0.10533933982966764, 8228)
0 -130 (0.349189689349381, 0.10532620661473756, 8315)
0 -120 (0.33769675824313583, 0.10496692645239565, 8444)
0 -110 (0.33301002416372266, 0.10430747905659611, 8714)
0 -100 (0.3116198104868644, 0.1039133582651433, 8945)
0 -90 (0.334224117186437, 0.102932513738364, 8959)
0 -80 (0.3717052561284095, 0.10206288745314945, 8886)
0 -70 (0.4014470937283766, 0.10096062154564861, 8761)
0 -60 (0.40916969321013735, 0.10084764621267611, 8655)
0 -50 (0.43301789744546937, 0.09955837243394687, 8591)
0 -40 (0.4301930185667863, 0.098984 …[+1143 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/yaw_score.py').read().split(\"if __name__\")[0])
for i in [0,50,100,112]:
 print('frame',i)
 best=[]
 for d in range(-180,180,5):
  ang=np.deg2rad(d);c,s=np.cos(ang),np.sin(ang);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float);M=T[i].astype(float)@Rz
  ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26)
  col,m=render_pose(M,x0,x1,y0,y1);gt=objmask[i,y0:y1,x0:x1];iou=(m&gt).sum()/(m|gt).sum()
  best.append((iou,d))
 print(sorted(best,reverse=True)[:10])
PY"
```
<details><summary>output</summary>

```
frame 0
[(0.8628659476117103, -95), (0.8623544133217826, -90), (0.8611057599691774, -85), (0.8573756350043132, -100), (0.8533563814462691, -80), (0.8400458102691354, -75), (0.8399237732253454, -105), (0.830662747700768, -70), (0.8279113625648279, -15), (0.8263410955029697, -10)]
frame 50
[(0.8622313806299676, -95), (0.8615203299941072, -90), (0.8579400822400627, -100), (0.8559106057635758, -85), (0.8477921062915201, -80), (0.8410634581797012, -105), (0.836360104884918, -75), (0.8342333654773385, -10), (0.8274536191483226, -15), (0.827161681254208, -5)]
frame 100
[(0.8508021904121433, -90), (0.8497985804718972, -95), (0.8475715108466116, -85), (0.8458301453710788, -100), (0.8406393568147014, -80), (0.8301868892894412, -105), (0.8300809138505474, -75), (0.8215396212192594, -10), (0.819093575 …[+325 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/yaw_score.py').read().split(\"if __name__\")[0])
import cv2
for i,d in [(0,-90),(0,0),(112,-90),(120,-90),(120,0)]:
 ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26)
 a=np.deg2rad(d);c,s=np.cos(a),np.sin(a);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float);col,m=render_pose(T[i]@Rz,x0,x1,y0,y1)
 im=getframe(i)[y0:y1,x0:x1].copy(); rgb=(col*255).astype(np.uint8)[...,::-1];im[m]=(.45*im[m]+.55*rgb[m]).astype(np.uint8);cv2.imwrite(f'/work/scratch/yaw{i}_{d}.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95])
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
from scipy.spatial.transform import Rotation as R
f=pickle.load(open('/pipeline/foundationpose/episode_000017.pkl','rb'))
r=np.load('/work/inputs/reconstruction.npz')
for i in [0,50,100,110,112,116,117,118,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340,400]:
 print(i,'fp iou',round(float(f['fp_full_image_iou'][i]),3),'reliable',bool(f['fp_reliable'][i]),'cand',int(f['fp_candidate_count'][i]),'sel',int(f['fp_selected_candidate_index'][i]),'vis',round(float(f['fp_selected_visibility'][i]),2),'state',int(f['fp_tracking_state'][i]),'src',int(f['fp_emitted_pose_source'][i]),'conf',int(f['fp_confirmed_pose_source'][i]))
PY"
```
<details><summary>output</summary>

```
0 fp iou 0.848 reliable True cand 252 sel 155 vis 1.0 state 0 src 6 conf 6
50 fp iou 0.859 reliable True cand 1 sel -1 vis 1.0 state 0 src 7 conf 7
100 fp iou 0.827 reliable True cand 1 sel -1 vis 1.0 state 0 src 7 conf 7
110 fp iou 0.862 reliable True cand 1 sel -1 vis 0.99 state 0 src 7 conf 7
112 fp iou 0.846 reliable True cand 1 sel -1 vis 1.0 state 0 src 7 conf 7
116 fp iou 0.871 reliable True cand 1 sel -1 vis 0.99 state 0 src 7 conf 7
117 fp iou 0.848 reliable True cand 1 sel -1 vis 1.0 state 0 src 7 conf 7
118 fp iou 0.869 reliable True cand 1 sel -1 vis 1.0 state 0 src 7 conf 7
120 fp iou 0.863 reliable True cand 1 sel -1 vis 0.99 state 0 src 7 conf 7
130 fp iou 0.788 reliable True cand 1 sel -1 vis 0.9 state 0 src 7 conf 7
140 fp iou 0.87 reliable True cand 1 sel -1 vis 0.98 stat …[+1046 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
r=np.load('/work/inputs/reconstruction.npz');Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for name in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 print('\\n',name)
 for i in [0,50,100,112,117,120,130,140,160,180,200,220,240,260,280,300,320,340,400]:
  M=r[name][i] if name=='T_world_mesh' else Twc@r[name][i]
  ax=M[:3,0];az=M[:3,2]
  print(i,round(float(np.degrees(np.arctan2(ax[2],ax[0]))),1),round(float(np.degrees(np.arctan2(az[2],az[0]))),1),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```

 T_world_mesh
0 -88.8 179.9; 50 -88.8 179.9; 100 -88.8 179.9; 112 -88.8 179.9; 117 0.1 169.6; 120 0.7 166.0; 130 12.2 131.7; 140 59.5 164.1; 160 69.4 174.2; 180 91.6 -161.8; 200 135.0 -122.1; 220 -58.3 -108.0; 240 -86.0 -68.0; 260 -100.1 -91.6; 280 -69.8 48.5; 300 -130.4 144.5; 320 -154.0 -78.5; 340 -160.0 -101.7; 400 -160.0 -101.7; 

 T_cam_mesh_foundationpose
0 -82.3 -103.6; 50 -80.9 -107.3; 100 -74.2 -104.3; 112 8.9 -119.2; 117 6.7 -119.0; 120 5.0 -125.0; 130 40.9 -154.4; 140 80.7 -155.9; 160 70.7 -178.0; 180 82.6 132.0; 200 150.5 -82.7; 220 -100.8 169.3; 240 -117.6 103.4; 260 -87.9 -0.5; 280 -72.4 47.8; 300 -125.9 147.9; 320 -146.0 -84.0; 340 -161.3 80.7; 400 -160.8 90.2; 

 T_cam_mesh_coconet
0 -88.3 -152.8; 50 -88.3 -152.8; 100 -88.3 -152.8; 112 -88.3 -152.8; 117 0.5 -170.2; 120 1.1 …[+233 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for a,b in [(0,117),(117,320),(320,340),(340,440),(340,400),(400,440)]:
 X=T[a:b]; tt=X[:,:3,3]; rr=R.from_matrix(X[:,:3,:3]).as_rotvec()
 print('\\n',a,b,'n',len(X))
 print('t mean',tt.mean(0),'std',tt.std(0),'range',np.ptp(tt,0))
 print('r mean',rr.mean(0),'std',rr.std(0),'range',np.ptp(rr,0))
 # center distances to mean pose
 tm=np.median(tt,0); rm=R.from_matrix(T[a:b,:3,:3]).as_rotvec(); rmed=np.median(rm,0)
 print('dist to median t',np.quantile(np.linalg.norm(tt-tm,axis=1),[0,.5,.9,1]))
 print('rot to median deg',np.quantile(np.degrees(np.linalg.norm(rm-rmed,axis=1)),[0,.5,.9,1]))
# per-frame yaw around local z accurately via projected x
for i in range(320,440):
 if i%5==0:
  M=T[i]; print(i,round(float(np.degrees(np.arctan2(M[2,0],M[0,0]))),2),np.round(M[:3,3],4))
PY"
```
<details><summary>output</summary>

```

 0 117 n 117
t mean [0.0216874  1.19390628 2.05032844] std [0.00189077 0.00759978 0.01200115] range [0.00516735 0.02076983 0.03279829]
r mean [ 1.25714581  1.16791034 -1.23236865] std [6.66133815e-16 2.44249065e-15 6.66133815e-16] range [0. 0. 0.]
dist to median t [0.         0.01456478 0.0210714  0.02287415]
rot to median deg [0. 0. 0. 0.]

 117 320 n 203
t mean [0.09473488 0.33686216 2.91123063] std [0.24392052 0.29505205 0.53424803] range [0.80923867 1.09952753 1.67474818]
r mean [ 1.14130883  0.17353042 -0.30630667] std [0.27810026 1.26836107 1.09258289] range [1.32180007 4.05829082 3.63717097]
dist to median t [0.20434803 0.53131171 1.08963401 1.34734039]
rot to median deg [ 25.13050821  79.42579274 167.89504324 188.72463116]

 320 340 n 20
t mean [-0.35448879  0.36685503  2.85153942 …[+2327 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,torch
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=m.to_geometry()
S,to_orig=trimesh.bounds.oriented_bounds(g)
print('S',S,'det',np.linalg.det(S[:3,:3]))
r=np.load('/work/inputs/reconstruction.npz')
d=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for key in ['pr','pr_initial','in']:
 pa=np.asarray(d[key]['pose_abs'])
 P=pa@S
 print('\\n',key)
 for i in [0,100,117,140,200,300,400]:
  print(i,'Pcam\\n',np.round(P[i],4))
PY"
```
<details><summary>output</summary>

```
S [[ 0.         -1.          0.         -0.00304588]
 [-1.          0.          0.          0.00263901]
 [ 0.          0.         -1.          0.09402343]
 [ 0.          0.          0.          1.        ]] det 1.0

 pr
0 Pcam
 [[ 0.188   0.9817 -0.03    0.0071]
 [ 0.0561 -0.0412 -0.9976  1.0684]
 [-0.9806  0.1859 -0.0628  1.9617]
 [ 0.      0.      0.      1.    ]]
100 Pcam
 [[ 0.2065  0.9775 -0.0433  0.0065]
 [ 0.0643 -0.0577 -0.9963  1.077 ]
 [-0.9763  0.2029 -0.0748  1.9841]
 [ 0.      0.      0.      1.    ]]
117 Pcam
 [[ 9.8740e-01 -1.4470e-01 -6.4100e-02  5.0000e-04]
 [-7.2000e-02 -5.0100e-02 -9.9610e-01  1.0697e+00]
 [ 1.4090e-01  9.8820e-01 -5.9900e-02  1.9782e+00]
 [ 0.0000e+00  0.0000e+00  0.0000e+00  1.0000e+00]]
140 Pcam
 [[-0.2432 -0.9531 -0.1801  0.0474]
 [-0.0496  0.1977 -0 …[+2795 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/yaw_score.py').read().split(\"if __name__\")[0])
import cv2,numpy as np
i=0
ys,xs=np.where(objmask[i]);x0=xs.min()-10;x1=xs.max()+11;y0=ys.min()-10;y1=ys.max()+11
im=getframe(i)[y0:y1,x0:x1]
out=[cv2.resize(im,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)]
for d in [0,-90,-45,45,90,180]:
 a=np.deg2rad(d);c,s=np.cos(a),np.sin(a);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float);col,m=render_pose(T[i]@Rz,x0,x1,y0,y1)
 z=np.zeros_like(im);z[m]=(col[m]*255).astype(np.uint8)[...,::-1];out.append(cv2.resize(z,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC))
cv2.imwrite('/work/scratch/yawcompare.jpg',np.hstack(out),[cv2.IMWRITE_JPEG_QUALITY,98])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"right hand.*21\\|42.*62\\|keypoint.*hand\" -n /code/reconstruction/modules/v2d_cari4d /workspace/v2d_cari4d 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_supervision.py:146:                    geometry_name = "decoded MHR COCO17 keypoints" if mode == MHR_JOINT_SUPERVISION_LEGACY_COCO17_ALL_BLOCKS else ("decoded MHR body-12 keypoints with hand/face frozen" if mode == MHR_JOINT_SUPERVISION_BODY12_FREEZE_HAND_FACE_ALL_LOSSES else "decoded MHR body-12 keypoints with pose-only gradients")
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/behave_data/const.py:120:                        [958.4615478515625, 550.42987060546875], [956.14801025390625, 555.01593017578125]])
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/body_pose.py:9:BODY_1DOF_ROT_IDXS = np.array([1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62,  …[+1060 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');kp=r['human_keypoints_world'];T=r['T_world_mesh']
for i in [0,112,117,120,130,140,150,160,170,180,190,200,220,240,260,280,300,320,340]:
 print('\\ni',i,'contact',r['contact_prob_right_left'][i])
 print('right',np.round(kp[i,21:42:5],3).tolist())
 print('left',np.round(kp[i,42:63:5],3).tolist())
 print('obj',np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```

i 0 contact [2.7040407e-15 3.9392857e-13]
right [[0.03099999949336052, 0.4830000102519989, 3.1410000324249268], [0.013000000268220901, 0.5099999904632568, 3.11899995803833], [-0.01600000075995922, 0.503000020980835, 3.111999988555908], [-0.041999999433755875, 0.4650000035762787, 3.1040000915527344], [-0.0430000014603138, 0.3930000066757202, 3.117000102996826]]
left [[0.25600001215934753, 0.4569999873638153, 3.428999900817871], [0.28299999237060547, 0.4779999852180481, 3.4660000801086426], [0.2849999964237213, 0.45899999141693115, 3.4860000610351562], [0.28700000047683716, 0.41100001335144043, 3.496999979019165], [0.2709999978542328, 0.3449999988079071, 3.4719998836517334]]
obj [0.019 1.185 2.036]

i 112 contact [8.853545e-07 3.554177e-07]
right [[-0.04100000113248825, 0.8619999885559082,  …[+12622 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
r=np.load('/work/inputs/reconstruction.npz');kp=r['human_keypoints_world'];T=r['T_world_mesh'];c=r['contact_prob_right_left']
# variants: all keypoints mean, wrist first, palm first 5 etc
for i in range(112,341,8):
 hand=np.vstack([kp[i,21:42],kp[i,42:63]])
 for lab,p in [('all',hand.mean(0)),('wrist',np.r_[kp[i,21],kp[i,42]].mean(0)),('first5',np.vstack([kp[i,21:26],kp[i,42:47]]).mean(0))]:
  if lab=='all': print('\\n',i,'obj',np.round(T[i,:3,3],3),'cont',np.round(c[i],2))
  print(lab,np.round(T[i,:3,3]-p,3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
 obj [0.256 0.372 2.291] cont [1. 1.]
all [ 0.112  0.127 -0.004]; wrist [-0.632 -0.516  1.403]; first5 [ 0.121  0.155 -0.004]; 

 168 obj [0.233 0.263 2.433] cont [1.   0.99]
all [0.02  0.086 0.02 ]; wrist [-0.695 -0.665  1.505]; first5 [0.038 0.106 0.022]; 

 176 obj [0.257 0.199 2.591] cont [1.   0.94]
all [0.022 0.137 0.042]; wrist [-0.68  -0.739  1.654]; first5 [0.043 0.159 0.045]; 

 184 obj [0.293 0.157 2.713] cont [1.   0.62]
all [0.023 0.089 0.054]; wrist [-0.693 -0.829  1.726]; first5 [0.045 0.109 0.062]; 

 192 obj [0.363 0.202 2.975] cont [0.99 0.03]
all [0.079 0.136 0.176]; wrist [-0.674 -0.834  1.939]; first5 [0.093 0.158 0.193]; 

 200 obj [0.424 0.144 3.009] cont [1. 1.]
all [0.067 0.134 0.085]; wrist [-0.658 -0.937  1.928]; first5 [0.078 0.154 0.108]; 

 208 obj [0.39  0.14 …[+2265 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');kp=r['human_keypoints_world'];T=r['T_world_mesh'];c=r['contact_prob_right_left']
# hand centroid all
R=kp[:,21:42].mean(1);L=kp[:,42:63].mean(1);V=L-R
# obj yaw from mesh x projection
yaw=np.degrees(np.arctan2(T[:,2,0],T[:,0,0]));yang=np.degrees(np.arctan2(V[:,2],V[:,0]))
# unwrap baseline yaw
for i in range(112,341,4):
 print(i,'cont',f'{c[i,0]:.2f}',f'{c[i,1]:.2f}','yaw',f'{yaw[i]:.1f}','vangle',f'{yang[i]:.1f}','diff',f'{((yaw[i]-yang[i]+180)%360)-180:.1f}','vnorm',f'{np.linalg.norm(V[i]):.3f}')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/yaw_score.py').read().split(\"if __name__\")[0])
# score subsets frame0 yaw, use actual bbox; inspect gradient NCC and base region
i=0;ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26)
im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(float)/255
for region in ['all','base','bands','upper']:
 print('\\nREGION',region)
 vals=[]
 for d in range(-180,180,5):
  a=np.deg2rad(d);c,s=np.cos(a),np.sin(a);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float);col,m=render_pose(T[i]@Rz,x0,x1,y0,y1)
  yy,xx=np.mgrid[y0:y1,x0:x1]; rel=(yy-ys.min())/(ys.max()-ys.min());
  if region=='base': reg=rel>.82
  elif region=='bands': reg=(rel>.25)&(rel<.65)
  elif region=='upper': reg=rel<.4
  else: reg=np.ones_like(m)
  pm=cv2.dilate(permask[i,y0:y1,x0:x1].astype(np.uint8),np.ones((7,7),np.uint8))>0
  val=m&objmask[i,y0:y1,x0:x1]&~pm&reg
  if val.sum()<20: continue
  A=col[val].mean(1);B=im[val].mean(1)
  ncc=np.corrcoef(A,B)[0,1]
  # gradients grayscale
  ga=cv2.Sobel(col.mean(2),cv2.CV_32F,1,0,ksize=3);gb=cv2.Sobel(im.mean(2),cv2.CV_32F,1,0,ksize=3)
  # per valid flatten gradient correlation
  gcc=np.corrcoef(ga[val],gb[val])[0,1]
  vals.append((ncc,gcc,d,val.sum()))
 print('ncc best',sorted(vals,reverse=True)[:5])
 print('gcc best',sorted(vals,key=lambda x:x[1],reverse=True)[:5])
PY"
```
<details><summary>output</summary>

```

REGION all
Traceback (most recent call last):
  File "<stdin>", line 21, in <module>
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgproc/src/filter.simd.hpp:3096: error: (-213:The function/feature is not implemented) Unsupported combination of buffer format (=6), and destination format (=5) in function 'getLinearColumnFilter'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/yaw_score.py').read().split(\"if __name__\")[0])
i=0;ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26)
im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255
yy,xx=np.mgrid[y0:y1,x0:x1]; rel=(yy-ys.min())/(ys.max()-ys.min())
for region in ['all','base','bands','upper']:
 print('\\nREGION',region)
 vals=[]
 for d in range(-180,180,5):
  a=np.deg2rad(d);c,s=np.cos(a),np.sin(a);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float);col,m=render_pose(T[i]@Rz,x0,x1,y0,y1);col=col.astype(np.float32)
  if region=='base': reg=rel>.82
  elif region=='bands': reg=(rel>.25)&(rel<.65)
  elif region=='upper': reg=rel<.4
  else: reg=np.ones_like(m)
  pm=cv2.dilate(permask[i,y0:y1,x0:x1].astype(np.uint8),np.ones((7,7),np.uint8))>0
  val=m&objmask[i,y0:y1,x0:x1]&~pm&reg
  if val.sum()<20: continue
  A=col[val].mean(1);B=im[val].mean(1);ncc=np.corrcoef(A,B)[0,1]
  ga=cv2.Sobel(col.mean(2),cv2.CV_32F,1,0,ksize=3);gb=cv2.Sobel(im.mean(2),cv2.CV_32F,1,0,ksize=3);gcc=np.corrcoef(ga[val],gb[val])[0,1]
  vals.append((float(ncc),float(gcc),d,int(val.sum())))
 print('ncc best',sorted(vals,reverse=True)[:5])
 print('gcc best',sorted(vals,key=lambda x:x[1],reverse=True)[:5])
PY"
```
<details><summary>output</summary>

```

REGION all
ncc best [(0.43859999326257204, 0.015711556595031352, -45, 8561), (0.4330178967528427, 0.034883203827366975, -50, 8591), (0.43019301887965034, 0.0237318720419112, -40, 8570), (0.4249778780440448, 0.030061177377869588, -55, 8618), (0.4236073087930371, 0.018150309134094204, -35, 8569)]
gcc best [(0.40948601039756, 0.05207376000147453, 175, 8180), (0.40653227373399975, 0.03928150715857242, 50, 7957), (0.36404046077216085, 0.03796420426573031, -150, 8197), (0.39815186679446996, 0.0360858189918749, 150, 7833), (0.4330178967528427, 0.034883203827366975, -50, 8591)]

REGION base
ncc best [(0.44494712149916515, 0.019834983851628313, 115, 2344), (0.4399238642734734, 0.007825396004857087, 125, 2343), (0.43165199090679807, 0.04954449614466424, 110, 2370), (0.4277656635588791, 0.0299407731 …[+1505 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,torch
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();S,_=trimesh.bounds.oriented_bounds(m)
d=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
r=np.load('/work/inputs/reconstruction.npz')
for i in [0,112,117,140,200,300,400]:
 print('\\ni',i)
 for key,a in [('pr',d['pr']['pose_abs'][i]@S),('in',d['in']['pose_abs'][i]@S),('r_coco',r['T_cam_mesh_coconet'][i]),('r_fp',r['T_cam_mesh_foundationpose'][i])]:
  print(key,'R',np.round(a[:3,:3],3),'t',np.round(a[:3,3],3))
PY"
```
<details><summary>output</summary>

```

i 0
pr R [[ 0.188  0.982 -0.03 ]
 [ 0.056 -0.041 -0.998]
 [-0.981  0.186 -0.063]] t [0.007 1.068 1.962]
in R [[ 0.265  0.964 -0.004]
 [ 0.13  -0.04  -0.991]
 [-0.955  0.262 -0.136]] t [-0.006  1.099  2.068]
r_coco R [[ 0.165  0.985 -0.041]
 [ 0.063 -0.052 -0.997]
 [-0.984  0.162 -0.07 ]] t [0.008 1.065 1.967]
r_fp R [[ 0.265  0.964 -0.004]
 [ 0.13  -0.04  -0.991]
 [-0.955  0.262 -0.136]] t [-0.006  1.099  2.068]

i 112
pr R [[ 0.984 -0.162 -0.078]
 [-0.087 -0.049 -0.995]
 [ 0.157  0.986 -0.062]] t [-1.000e-03  1.066e+00  1.965e+00]
in R [[ 0.956 -0.291 -0.03 ]
 [-0.066 -0.115 -0.991]
 [ 0.285  0.95  -0.129]] t [-0.015  1.097  2.064]
r_coco R [[ 0.165  0.985 -0.041]
 [ 0.063 -0.052 -0.997]
 [-0.984  0.162 -0.07 ]] t [0.008 1.065 1.967]
r_fp R [[ 0.956 -0.291 -0.03 ]
 [-0.066 -0.115 -0.991] …[+2179 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in range(108,137):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();crop=im[1010:1130,690:850];crop=cv2.resize(crop,(400,300),interpolation=cv2.INTER_CUBIC);cv2.putText(crop,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);ims.append(crop)
while len(ims)%5:ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/base_seq.jpg',np.vstack([np.hstack(ims[j:j+5]) for j in range(0,len(ims),5)]),[cv2.IMWRITE_JPEG_QUALITY,98])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_sil.py <<'PY'
exec(open('/work/scratch/yaw_score.py').read().split(\"if __name__\")[0])
from scipy.optimize import minimize
import sys

def render_mask_world(M,x0,x1,y0,y1):
 col,m=render_pose(M,x0,x1,y0,y1);return m
# pose correction in camera frame: T_world' = Twc @ (delta_cam @ inv(Twc) @ T_world) basically add world translation
# easier param as translation in world transformed by camera rot: t world = t base + R_wc[:, :3]@dc
def makeM(base,yaw,dc):
 a=np.deg2rad(yaw);c,s=np.cos(a),np.sin(a);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float);M=base.astype(float)@Rz
 M[:3,3]+=np.array(cam['T_world_cam'])[:3,:3]@np.asarray(dc)
 return M
def run(i,yaw):
 ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-30);x1=min(W,xs.max()+31);y0=max(0,ys.min()-30);y1=min(H,ys.max()+31);gt=objmask[i,y0:y1,x0:x1]
 count=[0]
 def obj(dc):
  count[0]+=1;m=render_mask_world(makeM(T[i],yaw,dc),x0,x1,y0,y1);return 1-(m&gt).sum()/(m|gt).sum()
 res=minimize(obj,[0,0,0],method='Powell',bounds=[(-.05,.05),(-.05,.05),(-.1,.1)],options={'xtol':1e-4,'ftol':1e-5,'maxiter':20})
 m=render_mask_world(makeM(T[i],yaw,res.x),x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum()
 print('frame',i,'yaw',yaw,'dc',res.x,'iou',iou,'eval',count[0])
 return iou,res.x
if __name__=='__main__':
 i=int(sys.argv[1]);yaw=float(sys.argv[2]);run(i,yaw)
PY
for y in 0 -90 -45 45 90 180; do python /work/scratch/opt_sil.py 0 "'$y; done'
```
<details><summary>output</summary>

```
frame 0 yaw 0.0 dc [-0.01044517 -0.00725806  0.00025705] iou 0.916815210932858 eval 199
frame 0 yaw -90.0 dc [-0.00496557 -0.00182222  0.00202924] iou 0.9095581694178116 eval 118
frame 0 yaw -45.0 dc [-0.00507703 -0.01037343 -0.00841711] iou 0.8154912428584808 eval 75
frame 0 yaw 45.0 dc [-0.01601736 -0.01528942 -0.00524594] iou 0.8229511295963702 eval 160
frame 0 yaw 90.0 dc [-0.01904494 -0.00704163  0.00038734] iou 0.9335474804256173 eval 75
frame 0 yaw 180.0 dc [-0.0167078  -0.00412031  0.00167527] iou 0.9427104309303262 eval 73

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/opt_sil4.py <<'PY'
exec(open('/work/scratch/opt_sil.py').read().split(\"if __name__\")[0])
from scipy.optimize import minimize
import sys
def run(i,yaw):
 ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-30);x1=min(W,xs.max()+31);y0=max(0,ys.min()-30);y1=min(H,ys.max()+31);gt=objmask[i,y0:y1,x0:x1];count=[0]
 def obj(p):
  count[0]+=1;m=render_mask_world(makeM(T[i],p[0],p[1:]),x0,x1,y0,y1);return 1-(m&gt).sum()/(m|gt).sum()
 res=minimize(obj,[yaw,0,0,0],method='Powell',bounds=[(yaw-20,yaw+20),(-.04,.04),(-.04,.04),(-.05,.05)],options={'xtol':1e-4,'ftol':1e-5,'maxiter':40})
 m=render_mask_world(makeM(T[i],res.x[0],res.x[1:]),x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum();print('frame',i,'start',yaw,'best',res.x,'iou',iou,'eval',count[0])
for y in [0,90,180,-90]:run(0,y)
PY
python /work/scratch/opt_sil4.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_sil4.py", line 1, in <module>
    exec(open('/work/scratch/opt_sil.py').read().split("if __name__")[0])
  File "<string>", line 1
    exec(open('/work/scratch/yaw_score.py').read().split("
                                                         ^
SyntaxError: unterminated string literal (detected at line 1)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_sil4.py <<'PY'
import sys
sys.path.insert(0,'/work/scratch')
from opt_sil import *
from scipy.optimize import minimize
def run(i,yaw):
 ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-30);x1=min(W,xs.max()+31);y0=max(0,ys.min()-30);y1=min(H,ys.max()+31);gt=objmask[i,y0:y1,x0:x1];count=[0]
 def obj(p):
  count[0]+=1;m=render_mask_world(makeM(T[i],p[0],p[1:]),x0,x1,y0,y1);return 1-(m&gt).sum()/(m|gt).sum()
 res=minimize(obj,[yaw,0,0,0],method='Powell',bounds=[(yaw-20,yaw+20),(-.04,.04),(-.04,.04),(-.05,.05)],options={'xtol':1e-4,'ftol':1e-5,'maxiter':40})
 m=render_mask_world(makeM(T[i],res.x[0],res.x[1:]),x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum();print('frame',i,'start',yaw,'best',res.x,'iou',iou,'eval',count[0])
for y in [0,90,180,-90]:run(0,y)
PY
python /work/scratch/opt_sil4.py"
```
<details><summary>output</summary>

```
frame 0 start 0 best [-3.00169822e+00 -9.47605245e-03 -8.24365194e-03 -9.94650048e-04] iou 0.9176645225136071 eval 241
frame 0 start 90 best [ 8.85496345e+01 -2.03773435e-02 -9.45449231e-03 -3.15638591e-03] iou 0.9354191736863217 eval 380
frame 0 start 180 best [ 1.75636210e+02 -1.64979554e-02 -3.18168196e-03  1.64913420e-03] iou 0.9462018940157163 eval 234
frame 0 start -90 best [-9.30538808e+01 -6.37052973e-03 -3.21724774e-03 -2.00106399e-04] iou 0.9137913989125062 eval 184

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
import sys
sys.path.insert(0,'/work/scratch')
from opt_sil import *
from scipy.optimize import minimize
import numpy as np

def run(frames,yaw,base_idx=0):
 # use same base pose, crop union around masks
 x0=650;x1=880;y0=850;y1=1130
 gts=[objmask[i,y0:y1,x0:x1] for i in frames]; count=[0]
 def obj(p):
  count[0]+=1;m=render_mask_world(makeM(T[base_idx],p[0],p[1:]),x0,x1,y0,y1)
  ious=[(m&gt).sum()/(m|gt).sum() for gt in gts]
  return 1-np.mean(ious)
 res=minimize(obj,[yaw,0,0,0],method='Powell',bounds=[(yaw-25,yaw+25),(-.04,.04),(-.04,.04),(-.06,.06)],options={'xtol':1e-3,'ftol':1e-6,'maxiter':30})
 m=render_mask_world(makeM(T[base_idx],res.x[0],res.x[1:]),x0,x1,y0,y1);ious=[(m&gt).sum()/(m|gt).sum() for gt in gts]
 print('frames',frames,'start',yaw,'best',res.x,'mean',np.mean(ious),'each',np.round(ious,3),'eval',count[0])
 return res.x
if __name__=='__main__':
 frames=[0,20,40,60,80,100]
 for y in [0,90,180,-90]:run(frames,y)
PY
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
frames [0, 20, 40, 60, 80, 100] start 0 best [-1.43034515 -0.01007738 -0.01181335 -0.00639274] mean 0.9166854566467671 each [0.917 0.917 0.917 0.918 0.917 0.915] eval 137
frames [0, 20, 40, 60, 80, 100] start 90 best [ 8.87416405e+01 -2.03518888e-02 -7.89371171e-03  9.82100400e-04] mean 0.9342328098038174 each [0.935 0.932 0.936 0.936 0.936 0.93 ] eval 203
frames [0, 20, 40, 60, 80, 100] start 180 best [ 1.75810635e+02 -1.72860849e-02 -4.78911623e-03 -6.57464850e-04] mean 0.9440842410615314 each [0.945 0.941 0.947 0.947 0.947 0.938] eval 179
frames [0, 20, 40, 60, 80, 100] start -90 best [-9.29416384e+01 -6.57044063e-03 -4.44659531e-03 -1.80660958e-03] mean 0.9120976622762956 each [0.912 0.913 0.91  0.912 0.912 0.914] eval 140

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys
sys.path.insert(0,'/work/scratch')
from opt_sil import *
# use optimized values
fits={0:[-1.43034515,-0.01007738,-0.01181335,-0.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26);im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);val=m&objmask[i,y0:y1,x0:x1];a=col[val];b=im[val]
 ncc=np.corrcoef(a.mean(1),b.mean(1))[0,1]
 # chroma normalized ncc
 an=a/(a.sum(1,keepdims=True)+1e-6);bn=b/(b.sum(1,keepdims=True)+1e-6)
 print(d,'ncc',ncc,'chroma',np.corrcoef(an.ravel(),bn.ravel())[0,1],'mae',np.mean(np.abs(a-b)))
PY"
```
<details><summary>output</summary>

```
0 ncc 0.3448093387098085 chroma 0.9357040759881153 mae 0.1091485
90 ncc 0.4281605652833172 chroma 0.9318837364110052 mae 0.10597238
180 ncc 0.37004977318319804 chroma 0.9267028952010045 mae 0.10971312
-90 ncc 0.3361216448428649 chroma 0.9335166966187415 mae 0.10892945

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys
sys.path.insert(0,'/work/scratch')
from opt_static import run
# base start pose at 340; crop currently y850:1130 x650:880, end object x584-655 y552-679 -> fails. override function local issue; just call custom variant copied
from opt_sil import *
from scipy.optimize import minimize
import numpy as np
frames=[340,360,380,400,420,439]
# union crop
ys,xs=np.where(objmask[frames].any(0));x0=max(0,xs.min()-30);x1=min(W,xs.max()+31);y0=max(0,ys.min()-30);y1=min(H,ys.max()+31);gts=[objmask[i,y0:y1,x0:x1] for i in frames]
for y in [0,90,180,-90]:
 def obj(p):
  m=render_mask_world(makeM(T[340],p[0],p[1:]),x0,x1,y0,y1);ious=[(m&gt).sum()/(m|gt).sum() for gt in gts];return 1-np.mean(ious)
 res=minimize(obj,[y,0,0,0],method='Powell',bounds=[(y-25,y+25),(-.04,.04),(-.04,.04),(-.06,.06)],options={'xtol':1e-3,'ftol':1e-6,'maxiter':30})
 m=render_mask_world(makeM(T[340],res.x[0],res.x[1:]),x0,x1,y0,y1);ious=[(m&gt).sum()/(m|gt).sum() for gt in gts]
 print('start',y,'best',res.x,'mean',np.mean(ious),'each',np.round(ious,3))
PY"
```
<details><summary>output</summary>

```
start 0 best [ 5.74675501e+00 -3.11042898e-03  5.28204526e-03 -2.02699379e-02] mean 0.8946883397972741 each [0.895 0.899 0.908 0.872 0.894 0.9  ]
start 90 best [ 8.90326252e+01  4.43717393e-04 -8.85107002e-05 -5.81707114e-02] mean 0.9090763294657656 each [0.904 0.911 0.92  0.893 0.91  0.916]
start 180 best [ 1.75654391e+02 -1.18343820e-02  1.75403697e-03 -3.98024690e-02] mean 0.8950697049335803 each [0.896 0.898 0.908 0.872 0.895 0.901]
start -90 best [-8.77610109e+01 -7.60886838e-03  5.53487977e-03 -3.39574372e-02] mean 0.89714902533819 each [0.901 0.899 0.91  0.874 0.897 0.901]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *;from scipy.optimize import minimize;import numpy as np
frames=[340,360,380,400,420,439];ys,xs=np.where(objmask[frames].any(0));x0=max(0,xs.min()-30);x1=min(W,xs.max()+31);y0=max(0,ys.min()-30);y1=min(H,ys.max()+31);gts=[objmask[i,y0:y1,x0:x1] for i in frames]
for y in [0,90,180,-90]:
 def obj(p):
  M=makeM(T[340],y,[p[0],p[1],0]);m=render_mask_world(M,x0,x1,y0,y1);ious=[(m&gt).sum()/(m|gt).sum() for gt in gts];return 1-np.mean(ious)
 res=minimize(obj,[0,0],method='Powell',bounds=[(-.04,.04),(-.04,.04)],options={'xtol':1e-3,'ftol':1e-6})
 M=makeM(T[340],y,[res.x[0],res.x[1],0]);m=render_mask_world(M,x0,x1,y0,y1);ious=[(m&gt).sum()/(m|gt).sum() for gt in gts]
 print(y,res.x,np.mean(ious),np.round(ious,3))
PY"
```
<details><summary>output</summary>

```
0 [-0.00508525  0.00844617] 0.8849826834986044 [0.883 0.886 0.898 0.866 0.885 0.891]
90 [-0.01005489  0.00888703] 0.8841841731933232 [0.88  0.887 0.896 0.865 0.885 0.892]
180 [-0.01776106  0.00823457] 0.878021680884352 [0.879 0.88  0.892 0.855 0.878 0.884]
-90 [-0.01320098  0.01008918] 0.8797623214577799 [0.884 0.882 0.893 0.857 0.878 0.884]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26);im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);val=m&objmask[i,y0:y1,x0:x1]
 print('\\nyaw',d)
 for sig in [0,.7,1.2,2,3,4,6]:
  a=col.mean(2);b=im.mean(2)
  if sig:a=cv2.GaussianBlur(a,(0,0),sig);b=cv2.GaussianBlur(b,(0,0),sig)
  ncc=np.corrcoef(a[val],b[val])[0,1]
  # highpass normalized
  ah=a-cv2.GaussianBlur(a,(0,0),8);bh=b-cv2.GaussianBlur(b,(0,0),8);h=np.corrcoef(ah[val],bh[val])[0,1]
  print(sig,round(ncc,3),round(h,3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```

yaw 0
0 0.345 0.256; 0.7 0.405 0.36; 1.2 0.424 0.425; 2 0.43 0.483; 3 0.422 0.513; 4 0.41 0.516; 6 0.386 0.493; 

yaw 90
0 0.428 0.375; 0.7 0.495 0.496; 1.2 0.505 0.554; 2 0.493 0.596; 3 0.467 0.615; 4 0.44 0.614; 6 0.393 0.586; 

yaw 180
0 0.37 0.31; 0.7 0.429 0.438; 1.2 0.443 0.51; 2 0.44 0.571; 3 0.425 0.6; 4 0.409 0.598; 6 0.381 0.57; 

yaw -90
0 0.336 0.309; 0.7 0.377 0.398; 1.2 0.383 0.444; 2 0.376 0.482; 3 0.359 0.501; 4 0.341 0.501; 6 0.31 0.478; 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import cv2,numpy as np
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-5;x1=xs.max()+6;y0=ys.min()-5;y1=ys.max()+6;im=getframe(i)[y0:y1,x0:x1]
out=[cv2.resize(im,None,fx=5,fy=5,interpolation=cv2.INTER_CUBIC)]
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);rgb=(col*255).astype(np.uint8)[...,::-1];z=im.copy();z[m]=(.35*z[m]+.65*rgb[m]).astype(np.uint8);out.append(cv2.resize(z,None,fx=5,fy=5,interpolation=cv2.INTER_CUBIC))
cv2.imwrite('/work/scratch/texcompare.jpg',np.hstack(out),[cv2.IMWRITE_JPEG_QUALITY,98])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_photo.py <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *;from scipy.optimize import minimize
import numpy as np, cv2
i=0;ys,xs=np.where(objmask[i]);x0=max(0,xs.min()-20);x1=min(W,xs.max()+21);y0=max(0,ys.min()-20);y1=min(H,ys.max()+21);im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255
gtm=objmask[i,y0:y1,x0:x1]
def calc(p):
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);val=m&gtm
 if val.sum()<100:return 0,0,m
 a=cv2.GaussianBlur(col.mean(2),(0,0),2);b=cv2.GaussianBlur(im.mean(2),(0,0),2)
 ah=a-cv2.GaussianBlur(a,(0,0),8);bh=b-cv2.GaussianBlur(b,(0,0),8)
 return np.corrcoef(a[val],b[val])[0,1],np.corrcoef(ah[val],bh[val])[0,1],m
for typ in ['gray','high']:
 print('TYPE',typ)
 for yaw in [0,90,180,-90]:
  def obj(p):
   ncc,h,m=calc(p);return 1-(h if typ=='high' else ncc)
  res=minimize(obj,[yaw,0,0,0.0],method='Powell',bounds=[(yaw-25,yaw+25),(-.04,.04),(-.04,.04),(-.06,.06)],options={'xtol':1e-3,'ftol':1e-6,'maxiter':30})
  ncc,h,m=calc(res.x);print('start',yaw,'best',np.round(res.x,4),'gray',round(ncc,4),'high',round(h,4))
PY
python /work/scratch/opt_photo.py"
```
<details><summary>output</summary>

```
TYPE gray
start 0 best [3.5554 0.0231 0.0204 0.0142] gray 0.5609 high 0.0228
start 90 best [8.13168e+01 4.80000e-03 2.11000e-02 6.50000e-03] gray 0.5614 high 0.0877
start 180 best [ 1.729129e+02 -3.600000e-03  9.400000e-03 -1.420000e-02] gray 0.5692 high 0.2575
start -90 best [-7.22949e+01  2.13000e-02  1.39000e-02  5.80000e-03] gray 0.5173 high 0.0285
TYPE high
start 0 best [ 0.4705 -0.0191 -0.0064  0.0142] gray 0.3153 high 0.5182
start 90 best [ 8.72678e+01 -2.11000e-02 -9.80000e-03  3.80000e-03] gray 0.4703 high 0.6265
start 180 best [ 1.729166e+02 -1.790000e-02 -3.500000e-03  6.700000e-03] gray 0.41 high 0.5906
start -90 best [-9.17418e+01 -1.14000e-02 -9.80000e-03 -1.60000e-03] gray 0.3451 high 0.5658

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-5;x1=xs.max()+6;y0=ys.min()-5;y1=ys.max()+6;im=getframe(i)[y0:y1,x0:x1]
def whitemask(rgb):
 hsv=cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_RGB2HSV);return (hsv[...,1]<100)&(hsv[...,2]>110)
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1); rgb=im[...,::-1].astype(np.float32)/255
 wm=whitemask(rgb)&objmask[i,y0:y1,x0:x1];wr=whitemask(col)&m
 # image coords mask bbox rel y
 yy,xx=np.mgrid[y0:y1,x0:x1]; rel=(yy-y0)/(y1-y0)
 for lo,hi,name in [(0,.3,'upperorange'),(.55,.85,'lowerorange'),(.85,1,'base')]:
  valid=objmask[i,y0:y1,x0:x1]&(rel>=lo)&(rel<hi)
  a=wm&valid;b=wr&valid
  # tolerate 2px
  bd=cv2.dilate(b.astype(np.uint8),np.ones((5,5),np.uint8))>0;ad=cv2.dilate(a.astype(np.uint8),np.ones((5,5),np.uint8))>0
  precision=(a&bd).sum()/max(a.sum(),1);recall=(b&ad).sum()/max(b.sum(),1)
  print(d,name,'a',a.sum(),'b',b.sum(),'p',round(precision,2),'r',round(recall,2),end=' | ')
 print()
PY"
```
<details><summary>output</summary>

```
0 upperorange a 18 b 143 p 0.22 r 0.06 | 0 lowerorange a 7 b 4 p 0.0 r 0.0 | 0 base a 0 b 0 p 0.0 r 0.0 | 
90 upperorange a 18 b 129 p 0.22 r 0.09 | 90 lowerorange a 7 b 1 p 0.0 r 0.0 | 90 base a 0 b 5 p 0.0 r 0.0 | 
180 upperorange a 18 b 184 p 0.22 r 0.09 | 180 lowerorange a 7 b 4 p 0.0 r 0.0 | 180 base a 0 b 6 p 0.0 r 0.0 | 
-90 upperorange a 18 b 188 p 0.17 r 0.06 | -90 lowerorange a 7 b 0 p 0.0 r 0.0 | -90 base a 0 b 1 p 0.0 r 0.0 | 

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=g.vertices
for d in [45,90,180]:
 a=np.deg2rad(d);R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
 print(d,'point mean',np.linalg.norm(v@R.T-v,axis=1).mean(),'umeyama',np.sqrt(((v@R.T-v)**2).sum(1).mean()))
 # nearest surface distances sampled
 print('surface nearest',trimesh.proximity.closest_point(g,v@R.T)[1].mean())
PY"
```
<details><summary>output</summary>

```
45 point mean 0.05367728529814739 umeyama 0.05758800920708647
surface nearest 0.00483190078157351
90 point mean 0.09918269049545476 umeyama 0.10640876604899745
surface nearest 0.00495542421997209
180 point mean 0.14026550605132523 umeyama 0.15048472010187794
Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/site-packages/trimesh/proximity.py", line 72, in nearby_faces
    hit_ids, hit_counts = rtree.intersection_v(bounds[:, :3], bounds[:, 3:])
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/rtree/index.py", line 1101, in intersection_v
    ids.resize(2 * len(ids) + counts[offn], refcheck=False)
TypeError: expected a sequence of integers or a single integer, got '1132057.0'

During handling of th …[+985 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import cv2,numpy as np
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
frames=[0,20,40,60,80,100]
for d,p in fits.items():
 vals=[]
 for i in frames:
  ys,xs=np.where(objmask[i]);x0=xs.min()-20;x1=xs.max()+21;y0=ys.min()-20;y1=ys.max()+21
  im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255;M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);val=m&objmask[i,y0:y1,x0:x1]
  a=cv2.GaussianBlur(col.mean(2),(0,0),2);b=cv2.GaussianBlur(im.mean(2),(0,0),2);ah=a-cv2.GaussianBlur(a,(0,0),8);bh=b-cv2.GaussianBlur(b,(0,0),8)
  vals.append((np.corrcoef(a[val],b[val])[0,1],np.corrcoef(ah[val],bh[val])[0,1]))
 print(d,'gray',np.mean(vals,0)[0],'high',np.mean(vals,0)[1],np.round(vals,3)[:,1])
PY"
```
<details><summary>output</summary>

```
0 gray 0.44376858162612737 high 0.514956998640756 [0.483 0.483 0.517 0.511 0.524 0.571]
90 gray 0.4973639205556631 high 0.6109481414615289 [0.596 0.58  0.585 0.609 0.61  0.686]
180 gray 0.4778376081820322 high 0.6093469246926951 [0.571 0.585 0.587 0.59  0.627 0.696]
-90 gray 0.4380833101610921 high 0.5563515494728667 [0.482 0.516 0.555 0.56  0.572 0.651]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import cv2,numpy as np
frames=[0,20,40,60,80,100]
# use average translation maybe optimize translation varying yaw rough; take 180 fit
p0=np.array([-1.72860849,-.00478911623,-.00065746485])
vals=[]
for d in range(-180,180,5):
 ss=[]
 for i in frames:
  ys,xs=np.where(objmask[i]);x0=xs.min()-20;x1=xs.max()+21;y0=ys.min()-20;y1=ys.max()+21
  im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255;M=makeM(T[i],d,p0);col,m=render_pose(M,x0,x1,y0,y1);val=m&objmask[i,y0:y1,x0:x1]
  a=cv2.GaussianBlur(col.mean(2),(0,0),2);b=cv2.GaussianBlur(im.mean(2),(0,0),2);ah=a-cv2.GaussianBlur(a,(0,0),8);bh=b-cv2.GaussianBlur(b,(0,0),8)
  ss.append(np.corrcoef(ah[val],bh[val])[0,1])
 vals.append((np.mean(ss),d))
print(sorted(vals,reverse=True)[:20])
PY"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:520: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis, **keepdims_kw)
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2889: RuntimeWarning: Degrees of freedom <= 0 for slice
  c = cov(x, y, rowvar, dtype=dtype)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2748: RuntimeWarning: divide by zero encountered in divide
  c *= np.true_divide(1, fact)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2748: RuntimeWarning: invalid value encountered in multiply
  c *= np.true_divide(1, fact)
[(nan, -180), (nan, -175), (nan, -17 …[+222 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-10;x1=xs.max()+11;y0=ys.min()-10;y1=ys.max()+11;gt=objmask[i,y0:y1,x0:x1]
ims=[]
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1)
 out=np.zeros((*m.shape,3),np.uint8);out[m&gt]=(0,255,0);out[gt&~m]=(255,0,255);out[m&gt]=np.array((90,90,90),np.uint8)[None,:] # this overwrites no; bad
 # make correctly
 out=np.zeros((*m.shape,3),np.uint8);out[m&gt]=(80,220,80);out[gt&~m]=(240,60,240);out[m&~gt]=(60,180,240)
 cv2.putText(out,f'{d} {(m&gt).sum()/(m|gt).sum():.3f}',(2,12),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
 ims.append(cv2.resize(out,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
cv2.imwrite('/work/scratch/diffs.png',np.hstack(ims))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import numpy as np
res={d:[] for d in [0,90,180,-90]}
for i in range(440):
 ys,xs=np.where(objmask[i]);
 if len(xs)<100:continue
 x0=max(0,xs.min()-20);x1=min(W,xs.max()+21);y0=max(0,ys.min()-20);y1=min(H,ys.max()+21);gt=objmask[i,y0:y1,x0:x1]
 for d in res:
  M=makeM(T[i],d,[0,0,0]);m=render_mask_world(M,x0,x1,y0,y1);res[d].append((m&gt).sum()/(m|gt).sum())
for d,a in res.items():print(d,len(a),np.mean(a),np.quantile(a,[.1,.5,.9]))
# per phase ranges
for a,b in [(0,117),(117,190),(190,255),(255,340),(340,440)]:
 print('\\nphase',a,b)
 for d in res:
  arr=[res[d][i] for i in range(a,min(b,len(res[d])))] if False else None
 # recompute quick loop aggregate fixed
 for d in res:
  vals=[]
  for i in range(a,b):
   ys,xs=np.where(objmask[i]);
   if len(xs)<100:continue
   x0=max(0,xs.min()-20);x1=min(W,xs.max()+21);y0=max(0,ys.min()-20);y1=min(H,ys.max()+21);gt=objmask[i,y0:y1,x0:x1];m=render_mask_world(makeM(T[i],d,[0,0,0]),x0,x1,y0,y1);vals.append((m&gt).sum()/(m|gt).sum())
  print(d,len(vals),round(float(np.mean(vals)),3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
0 390 0.7646957801691344 [0.60224115 0.81903935 0.8410127 ]
90 390 0.7015598183067128 [0.53954176 0.7075766  0.79368343]
180 390 0.6653383392195251 [0.53515975 0.67339466 0.74872566]
-90 390 0.7369216742193432 [0.60376075 0.72348298 0.86535764]

phase 0 117
0 117 0.819; 90 117 0.705; 180 117 0.742; -90 117 0.863; 

phase 117 190
0 73 0.716; 90 73 0.726; 180 73 0.716; -90 73 0.723; 

phase 190 255
0 15 0.313; 90 15 0.28; 180 15 0.276; -90 15 0.303; 

phase 255 340
0 85 0.727; 90 85 0.654; 180 85 0.605; -90 85 0.682; 

phase 340 440
0 100 0.837; 90 100 0.783; 180 100 0.649; -90 100 0.711; 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
try:import skimage;print(skimage.__version__)
except Exception as e:print(e)
PY"
```
<details><summary>output</summary>

```
0.26.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import cv2,numpy as np
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
# average static image, use frame 0 perspective
i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-10;x1=xs.max()+11;y0=ys.min()-10;y1=ys.max()+11;im=getframe(i)[y0:y1,x0:x1]
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);mask=objmask[i,y0:y1,x0:x1].astype(np.uint8)*255
sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.005,edgeThreshold=20)
kpq,dq=sift.detectAndCompute(gray,mask);print('query kp',len(kpq))
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);rgb=(col*255).astype(np.uint8)[...,::-1];g=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);mm=m.astype(np.uint8)*255;kpt,dt=sift.detectAndCompute(g,mm);print('\\nyaw',d,'kp',len(kpt))
 if len(kpt) and dq is not None:
  bf=cv2.BFMatcher(cv2.NORM_L2)
  ms=bf.knnMatch(dq,dt,k=2);good=[a for a,b in ms if a.distance<.75*b.distance]
  print('good',len(good),'dist median',np.median([x.distance for x in good]) if good else None)
  # count geometrically consistent translations under same pose (should near zero)
  disp=np.array([kpt[x.trainIdx].pt for x in good])-np.array([kpq[x.queryIdx].pt for x in good]) if good else []
  if len(disp): print('disp mean',disp.mean(0),'med',n …[+70 chars]
```
<details><summary>output</summary>

```
query kp 124

yaw 0 kp 124
good 6 dist median 245.73365020751953
disp mean [20.17109203  6.44038518] med [27.11074829 23.94013977] within2 0.0

yaw 90 kp 131
good 3 dist median 235.81349182128906
disp mean [ -6.92528152 -80.92410787] med [-17.89670944 -76.55628204] within2 0.0

yaw 180 kp 101
good 1 dist median 338.4035949707031
disp mean [37.29114151 25.39838409] med [37.29114151 25.39838409] within2 0.0

yaw -90 kp 107
good 1 dist median 247.2023468017578
disp mean [ 5.38222122 93.03047943] med [ 5.38222122 93.03047943] within2 0.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-5;x1=xs.max()+6;y0=ys.min()+105;y1=ys.max()+6;im=getframe(i)[y0:y1,x0:x1]
out=[cv2.resize(im,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)]
for d,p in fits.items():
 M=makeM(T[i],d,p);col,m=render_pose(M,x0,x1,y0,y1);rgb=(cv2.GaussianBlur(col,(0,0),.8)*255).astype(np.uint8)[...,::-1];z=np.zeros_like(im);z[m]=rgb[m];out.append(cv2.resize(z,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST))
cv2.imwrite('/work/scratch/basecompare.png',np.hstack(out))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/work/scratch/opt_sil.py", line 11, in makeM
    M[:3,3]+=np.array(cam['T_world_cam'])[:3,:3]@np.asarray(dc)
             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 4 is different from 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-5;x1=xs.max()+6;y0=ys.min()+105;y1=ys.max()+6;im=getframe(i)[y0:y1,x0:x1]
out=[cv2.resize(im,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)]
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);rgb=(cv2.GaussianBlur(col,(0,0),.8)*255).astype(np.uint8)[...,::-1];z=np.zeros_like(im);z[m]=rgb[m];out.append(cv2.resize(z,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST))
cv2.imwrite('/work/scratch/basecompare.png',np.hstack(out))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import numpy as np
for i in list(range(117,201,4))+list(range(252,341,4)):
 ys,xs=np.where(objmask[i]);
 if len(xs)<100:continue
 x0=max(0,xs.min()-20);x1=min(W,xs.max()+21);y0=max(0,ys.min()-20);y1=min(H,ys.max()+21);gt=objmask[i,y0:y1,x0:x1];scores=[]
 for d in [0,90,180,-90]:
  m=render_mask_world(makeM(T[i],d,[0,0,0]),x0,x1,y0,y1);scores.append((m&gt).sum()/(m|gt).sum())
 print(i,'area',len(xs),'best',[0,90,180,-90][int(np.argmax(scores))],'scores',np.round(scores,3))
PY"
```
<details><summary>output</summary>

```
117 area 9591 best -90 scores [0.825 0.861 0.863 0.872]
121 area 9590 best -90 scores [0.828 0.856 0.855 0.859]
125 area 9304 best -90 scores [0.824 0.841 0.855 0.857]
129 area 9047 best 90 scores [0.752 0.79  0.78  0.786]
133 area 8781 best 90 scores [0.723 0.763 0.744 0.734]
137 area 8585 best 0 scores [0.785 0.765 0.723 0.767]
141 area 8287 best 0 scores [0.804 0.798 0.788 0.769]
145 area 7468 best -90 scores [0.802 0.817 0.762 0.818]
149 area 6954 best -90 scores [0.824 0.799 0.742 0.843]
153 area 6560 best 180 scores [0.7   0.789 0.815 0.742]
157 area 6294 best -90 scores [0.826 0.79  0.762 0.838]
161 area 5833 best -90 scores [0.771 0.735 0.718 0.778]
165 area 4852 best -90 scores [0.705 0.705 0.692 0.708]
169 area 4293 best -90 scores [0.686 0.689 0.676 0.691]
173 area 3999 best 90  …[+1558 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_frames.py <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
from scipy.optimize import minimize
import numpy as np
for i in [112,117,121,125,129,133,140,150,160,170,180,185,260,280,300,320,340,400]:
 ys,xs=np.where(objmask[i]);
 if len(xs)<100:continue
 x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26);gt=objmask[i,y0:y1,x0:x1]
 out=[]
 for y in [0,90,180,-90]:
  def obj(p):
   m=render_mask_world(makeM(T[i],y,p),x0,x1,y0,y1);return 1-(m&gt).sum()/(m|gt).sum()
  res=minimize(obj,[0,0,0],method='Powell',bounds=[(-.035,.035),(-.035,.035),(-.06,.06)],options={'xtol':2e-3,'ftol':1e-3,'maxiter':20})
  M=makeM(T[i],y,res.x);m=render_mask_world(M,x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum();out.append((iou,y,res.x))
 print(i,[(round(float(q[0]),3),q[1],np.round(q[2],3).tolist()) for q in out],flush=True)
PY
python /work/scratch/opt_frames.py"
```
<details><summary>output</summary>

```
112 [(0.916, 0, [-0.01, -0.006, 0.003]), (0.933, 90, [-0.018, -0.004, 0.002]), (0.938, 180, [-0.016, -0.004, -0.001]), (0.906, -90, [-0.006, -0.0, 0.005])]
117 [(0.918, 0, [-0.011, -0.005, -0.003]), (0.912, 90, [-0.008, -0.0, -0.001]), (0.872, 180, [0.002, -0.002, -0.002]), (0.881, -90, [-0.003, -0.005, -0.002])]
121 [(0.908, 0, [-0.01, -0.003, 0.001]), (0.902, 90, [-0.007, -0.001, -0.002]), (0.861, 180, [0.002, -0.002, -0.002]), (0.869, -90, [-0.004, -0.009, -0.007])]
125 [(0.895, 0, [-0.011, -0.002, -0.001]), (0.895, 90, [-0.008, -0.013, -0.025]), (0.857, 180, [0.002, 0.002, -0.0]), (0.862, -90, [-0.001, 0.0, 0.006])]
129 [(0.835, 0, [-0.008, 0.012, -0.01]), (0.836, 90, [-0.005, 0.008, -0.01]), (0.805, 180, [0.005, 0.012, -0.002]), (0.807, -90, [-0.0, 0.009, -0.003])]
133 [(0.79, 0, [-0. …[+2017 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
# use opt from printed approximate rerun quick? use previous optimization values unavailable; use local optimize
from scipy.optimize import minimize
import numpy as np,cv2
for i in [112,117,125,140]:
 ys,xs=np.where(objmask[i]);x0=xs.min()-10;x1=xs.max()+11;y0=ys.min()-10;y1=ys.max()+11;gt=objmask[i,y0:y1,x0:x1];outs=[]
 for y in [0,90,180,-90]:
  def obj(p):
   m=render_mask_world(makeM(T[i],y,p),x0,x1,y0,y1);return 1-(m&gt).sum()/(m|gt).sum()
  res=minimize(obj,[0,0,0],method='Powell',bounds=[(-.035,.035),(-.035,.035),(-.06,.06)],options={'xtol':2e-3,'ftol':1e-3,'maxiter':20})
  m=render_mask_world(makeM(T[i],y,res.x),x0,x1,y0,y1);out=np.zeros((*m.shape,3),np.uint8);out[m&gt]=(80,220,80);out[gt&~m]=(240,60,240);out[m&~gt]=(60,180,240);cv2.putText(out,str(y),(2,12),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1);outs.append(cv2.resize(out,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
 cv2.imwrite(f'/work/scratch/diff{i}.png',np.hstack(outs))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import cv2,numpy as np
fits={0:[-1.43034515,-.01007738,-.01181335,-.00639274],90:[88.7416405,-.0203518888,-.00789371171,.0009821],180:[175.810635,-.0172860849,-.00478911623,-.00065746485],-90:[-92.9416304,-.00657044063,-.00444659531,-.00180660958]}
i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-3;x1=xs.max()+4;y0=ys.min()+100;y1=ys.max()+4;im=getframe(i)[y0:y1,x0:x1]
def enh(x):
 x=cv2.resize(x,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC);g=cv2.cvtColor(x,cv2.COLOR_BGR2LAB);g[...,0]=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(g[...,0]);return cv2.cvtColor(g,cv2.COLOR_LAB2BGR)
out=[enh(im)]
for d,p in fits.items():
 M=makeM(T[i],p[0],p[1:]);col,m=render_pose(M,x0,x1,y0,y1);rgb=(col*255).astype(np.uint8)[...,::-1];z=np.zeros_like(im);z[m]=rgb[m];out.append(enh(z))
cv2.imwrite('/work/scratch/baseenh.png',np.hstack(out))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [108,109,110,111,112,113,114,115,116,117,118,119]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();cr=im[1020:1110,705:835];cr=cv2.resize(cr,None,fx=5,fy=5,interpolation=cv2.INTER_LANCZOS4);cv2.putText(cr,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2);ims.append(cr)
cv2.imwrite('/work/scratch/base1212.jpg',np.hstack(ims),[cv2.IMWRITE_JPEG_QUALITY,100])
# difference images consecutive in alignment with phase correlation
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *;from scipy.optimize import minimize;import numpy as np
frames=[340,360,380,400,420,439];ys,xs=np.where(objmask[frames].any(0));x0=max(0,xs.min()-30);x1=min(W,xs.max()+31);y0=max(0,ys.min()-30);y1=min(H,ys.max()+31);gts=[objmask[i,y0:y1,x0:x1] for i in frames]
for z in [0,-.02,-.04,-.05,-.06]:
 print('z',z)
 for y in [0,90,180,-90]:
  def obj(p):
   m=render_mask_world(makeM(T[340],y,[p[0],p[1],z]),x0,x1,y0,y1);ious=[(m&gt).sum()/(m|gt).sum() for gt in gts];return 1-np.mean(ious)
  res=minimize(obj,[0,0],method='Powell',bounds=[(-.03,.03),(-.03,.03)],options={'xtol':1e-3,'ftol':1e-5})
  m=render_mask_world(makeM(T[340],y,[res.x[0],res.x[1],z]),x0,x1,y0,y1);ious=[(m&gt).sum()/(m|gt).sum() for gt in gts];print(y,round(float(np.mean(ious)),3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
z 0
0 0.885; 90 0.884; 180 0.88; -90 0.88; 
z -0.02
0 0.897; 90 0.892; 180 0.885; -90 0.89; 
z -0.04
0 0.906; 90 0.902; 180 0.895; -90 0.897; 
z -0.05
0 0.907; 90 0.905; 180 0.898; -90 0.904; 
z -0.06
0 0.914; 90 0.911; 180 0.902; -90 0.907; 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def fr(i):cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,a=cap.read();return a
with h5py.File('/work/inputs/masks.h5') as f:om=np.asarray(f['object'])
for i in range(110,126):
 a=cv2.cvtColor(fr(i),cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(fr(i+1),cv2.COLOR_BGR2GRAY)
 flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,4,21,4,7,1.5,0)
 both=om[i]&om[i+1]
 # erode by 3 to avoid boundaries
 er=cv2.erode(both.astype(np.uint8),np.ones((9,9),np.uint8))>0
 f=flow[er];mag=np.linalg.norm(f,axis=1);med=np.median(f,axis=0);res=np.linalg.norm(f-med,axis=1)
 print(i,'pix',len(f),'mean',np.round(f.mean(0),2),'med',np.round(med,2),'mag med/p90',np.round(np.median(mag),2),np.round(np.percentile(mag,90),2),'res med/p90',np.round(np.median(res),2),np.round(np.percentile(res,90),2))
PY"
```
<details><summary>output</summary>

```
110 pix 7569 mean [-0.02  0.02] med [-0. -0.] mag med/p90 0.05 0.28 res med/p90 0.05 0.28
111 pix 7566 mean [-0.02  0.05] med [-0.    0.01] mag med/p90 0.05 0.27 res med/p90 0.05 0.26
112 pix 7556 mean [-0.01  0.02] med [-0.  0.] mag med/p90 0.04 0.2 res med/p90 0.04 0.2
113 pix 7544 mean [0.03 0.04] med [0.01 0.01] mag med/p90 0.05 0.25 res med/p90 0.05 0.24
114 pix 7405 mean [-0.01  0.14] med [-0. -0.] mag med/p90 0.05 0.25 res med/p90 0.05 0.25
115 pix 7303 mean [0.03 0.22] med [ 0. -0.] mag med/p90 0.04 0.29 res med/p90 0.04 0.29
116 pix 7351 mean [0.02 0.3 ] med [0. 0.] mag med/p90 0.05 0.29 res med/p90 0.05 0.29
117 pix 7320 mean [0.02 0.29] med [-0.  0.] mag med/p90 0.05 0.27 res med/p90 0.05 0.27
118 pix 7261 mean [0.04 0.21] med [-0.  0.] mag med/p90 0.07 0.3 res med/p90 0.06 0.3
 …[+645 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');kp=r['human_keypoints_world'];T=r['T_world_mesh'];c=r['contact_prob_right_left']
R=kp[:,21:42].mean(1);L=kp[:,42:63].mean(1);V=L-R
yaw=np.unwrap(np.arctan2(T[:,2,0],T[:,0,0]));hv=np.unwrap(np.arctan2(V[:,2],V[:,0]))
for a,b in [(112,140),(117,200),(117,330),(140,200),(250,330)]:
 # regress derivative yaw vs hand angle derivative while contact
 ok=(c[:,0]>.5)&(c[:,1]>.5);ok[:a]=False;ok[b:]=False
 yy=savgol_filter(yaw,11,2);hh=savgol_filter(hv,11,2);dy=np.diff(yy);dh=np.diff(hh);m=ok[1:]&ok[:-1]
 print(a,b,'n',m.sum(),'corr dyaw,dhand',np.corrcoef(dy[m],dh[m])[0,1],'std',np.std(dy[m]),np.std(dh[m]))
 # residual with best constant offset
 d=(yaw[ok]-hv[ok]+np.pi)%(2*np.pi)-np.pi
 print('resid median',np.median(d),'mad',np.median(np.abs(d-np.median(d))),'std',np.std(d))
PY"
```
<details><summary>output</summary>

```
112 140 n 18 corr dyaw,dhand 0.588400987970971 std 0.07014562 0.03640313
resid median 0.13941693 mad 0.14249015 std 0.2575704
117 200 n 68 corr dyaw,dhand 0.14549868264561636 std 0.062613554 0.03935678
resid median -0.16130805 mad 0.25048018 std 0.39246303
117 330 n 198 corr dyaw,dhand 0.025885675048284592 std 0.09890673 0.061313268
resid median -0.19063258 mad 0.67108655 std 1.1222795
140 200 n 49 corr dyaw,dhand -0.02389386448954616 std 0.055748813 0.040585004
resid median -0.27250803 mad 0.21620142 std 0.34602875
250 330 n 79 corr dyaw,dhand 0.0011905873508424492 std 0.061220154 0.08191281
resid median -1.134773 mad 0.63969123 std 0.8409905

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/dp_yaw.py <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import median_filter
# candidate rotations world R_i @ Rz(k90)
F=len(T); Ks=[0,1,2,3]
Rc=np.zeros((F,4,3,3));Rbase=T[:,:3,:3].astype(float)
for i in range(F):
 for k in Ks:
  a=k*np.pi/2;Rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
  Rc[i,k]=Rbase[i]@Rz
# data mask
D=np.full((F,4),.5)
for i in range(F):
 ys,xs=np.where(objmask[i]);
 if len(xs)<80:continue
 x0=max(0,xs.min()-20);x1=min(W,xs.max()+21);y0=max(0,ys.min()-20);y1=min(H,ys.max()+21);gt=objmask[i,y0:y1,x0:x1]
 for k in Ks:
  m=render_mask_world(makeM(T[i],k*90,[0,0,0]),x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum();D[i,k]=0.5*(1-iou) # scale
# second-order DP with states 16, initial velocity zero or prev same
for wa in [0.005,.01,.02,.05,.1]:
 n=16; dp=np.full((F,n),1e9);back=np.zeros((F,n),np.int8)
 # state prev k and curr k; start assumes prev k
 for a in Ks:
  for b in Ks:
   dp[0,a*4+b]=D[0,a]+D[0,b]+10*int(a"'!=b)
 for i in range(1,F):
  for a in Ks: # prevprev
   for b in Ks: # prev
    prevVal=dp[i-1,a*4+b]
    if prevVal>=1e8:continue
    vprev=Rot.from_matrix(Rc[i-2,a].T@Rc[i-2,b]).as_rotvec() if i>=2 else np.zeros(3) # hmm prior velocity uses bases 2 back
    for c in Ks:
     vcur=Rot.from_matrix(Rc[i-1,b].T@Rc[i,c]).as_rotvec()
     cost=wa*np.linalg.norm(vcur-vprev)**2+D[ …[+629 chars]
```
<details><summary>output</summary>

```
wa 0.005 cost 67.5776114966482 counts [233  35   3 169] switches [128 129 146 152 153 154 170 171 188 266 276 289 304] initial [3 3 3 3 3 3 3 3 3 3] end [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
segments [(0, 127, 3), (128, 128, 2), (129, 145, 1), (146, 151, 0), (152, 152, 1), (153, 153, 2), (154, 169, 3), (170, 170, 2), (171, 187, 1), (188, 265, 0), (266, 275, 3), (276, 288, 0), (289, 303, 3), (304, 439, 0)]
wa 0.01 cost 67.80614672871992 counts [203  38   9 190] switches [128 129 141 152 153 154 170 171 190 191 192 193 194 195 196 197 198 199
 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 266 306] initial [3 3 3 3 3 3 3 3 3 3] end [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
segments [(0, 127, 3), (128, 128, 2), (129, 140, 1), (141, 151, 0), (152, 152, 1), (153, 153, 2), (154, …[+3588 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/mask_align.py <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import numpy as np

def align_score(i,k,iters=4):
 ys,xs=np.where(objmask[i])
 if len(xs)<80:return None,None
 x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26);gt=objmask[i,y0:y1,x0:x1]
 a=k*np.pi/2;c,s=np.cos(a),np.sin(a);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float)
 M=T[i].astype(float)@Rz
 ygt,xgt=np.where(gt); yc0,yc1,xc0,xc1=ygt.min()+y0,ygt.max()+y0,xgt.min()+x0,xgt.max()+x0; area0=len(xgt);cent0=np.array([xgt.mean()+x0,ygt.mean()+y0])
 for it in range(iters):
  m=render_mask_world(M,x0,x1,y0,y1);ysp,xsp=np.where(m)
  if len(xsp)==0:break
  cent=np.array([xsp.mean()+x0,ysp.mean()+y0]);area=len(xsp)
  # camera transform and current z
  Tc=np.linalg.inv(np.array(cam['T_world_cam'])).astype(float);Mc=Tc@M;z=Mc[2,3]
  dx,dy=cent0-cent; scale=np.sqrt(area0/area)
  Mc[0,3]+=dx*z/K[0,0];Mc[1,3]+=dy*z/K[1,1];Mc[2,3]*=scale
  M=np.array(cam['T_world_cam'])@Mc
 m=render_mask_world(M,x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum();return iou,M
if __name__=='__main__':
 for i in [0,50,100,112,117,121,125,129,140,160,180,260,300,320,340,400]:
  print(i,[(k,round(float(align_score(i,k)[0]),3)) for k in range(4)],flush=True)
PY
python /work/scratch/mask_align.py"
```
<details><summary>output</summary>

```
0 [(0, 0.57), (1, 0.43), (2, 0.466), (3, 0.522)]
50 [(0, 0.218), (1, 0.132), (2, 0.171), (3, 0.184)]
100 [(0, 0.46), (1, 0.314), (2, 0.39), (3, 0.418)]
112 [(0, 0.393), (1, 0.285), (2, 0.322), (3, 0.365)]
117 [(0, 0.317), (1, 0.399), (2, 0.402), (3, 0.474)]
121 [(0, 0.46), (1, 0.587), (2, 0.63), (3, 0.699)]
125 [(0, 0.857), (1, 0.738), (2, 0.614), (3, 0.595)]
129 [(0, 0.322), (1, 0.262), (2, 0.378), (3, 0.418)]
140 [(0, 0.287), (1, 0.258), (2, 0.308), (3, 0.268)]
160 [(0, 0.514), (1, 0.588), (2, 0.514), (3, 0.581)]
180 [(0, 0.141), (1, 0.147), (2, 0.141), (3, 0.137)]
260 [(0, 0.248), (1, 0.48), (2, 0.324), (3, 0.394)]
300 [(0, 0.036), (1, 0.026), (2, 0.028), (3, 0.005)]
320 [(0, 0.83), (1, 0.68), (2, 0.707), (3, 0.551)]
340 [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
400 [(0, 0.0), (1, 0.0),  …[+20 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
def rot(k):
 a=k*np.pi/2;return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
Ts=T.copy()
for i in range(117,440):Ts[i,:3,:3]=T[i,:3,:3]@rot(1)
for i in range(117,201):
 if i%2==0:
  ax=Ts[i,:3,0]
  v=R.from_matrix(Ts[i,:3,:3].T@Ts[i-1,:3,:3]).as_rotvec()
  print(i,'yaw',round(np.degrees(np.arctan2(ax[2],ax[0])),1),'stepdeg',round(np.degrees(np.linalg.norm(v)),1),'axis',np.round(v/np.linalg.norm(v),2))
PY"
```
<details><summary>output</summary>

```
118 yaw 90.0 stepdeg 0.8 axis [ 0.11 -0.99 -0.09]
120 yaw 90.6 stepdeg 0.9 axis [ 0.09  0.48 -0.87]
122 yaw 88.2 stepdeg 2.8 axis [ 0.07 -0.27  0.96]
124 yaw 90.2 stepdeg 2.3 axis [-0.2  -0.29 -0.93]
126 yaw 87.4 stepdeg 1.8 axis [ 0.25 -0.91  0.33]
128 yaw 93.1 stepdeg 5.2 axis [ 0.19 -0.71 -0.68]
130 yaw 100.8 stepdeg 6.0 axis [0.2  0.19 0.96]
132 yaw 135.3 stepdeg 25.3 axis [ 0.21  0.16 -0.96]
134 yaw 135.3 stepdeg 0.0 axis [-0.32  0.53  0.79]
136 yaw 138.3 stepdeg 0.0 axis [-0.21  0.63  0.75]
138 yaw 147.8 stepdeg 12.1 axis [ 0.41 -0.66 -0.63]
140 yaw 147.8 stepdeg 0.0 axis [-0.09  0.67  0.73]
142 yaw 150.3 stepdeg 25.4 axis [-0.11 -0.33  0.94]
144 yaw 140.5 stepdeg 33.3 axis [-0.05 -0.01  1.  ]
146 yaw 138.2 stepdeg 4.5 axis [ 0.37 -0.59 -0.72]
148 yaw 134.3 stepdeg 3.3 axis [ 0.59 -0 …[+1351 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];kp=r['human_keypoints_world'];c=r['contact_prob_right_left']
R=kp[:,21:42].mean(1);L=kp[:,42:63].mean(1);h=(R+L)/2;t=T[:,:3,3]
ok=(c[:,0]>.5)&(c[:,1]>.5)
print('ok ranges',np.where(ok)[0].min(),np.where(ok)[0].max(),ok.sum())
off=t-h
# visible/contact stats by ranges
for a,b in [(117,190),(190,255),(255,337)]:
 m=ok.copy();m[:a]=False;m[b:]=False
 print(a,b,'n',m.sum(),'off mean',off[m].mean(0),'std',off[m].std(0),'res p90',np.percentile(np.linalg.norm(off[m]-np.median(off[m],0),axis=1),90))
# smooth offset with SG and hand path predicted
off_s=np.zeros_like(off)
for j in range(3):off_s[117:337,j]=savgol_filter(off[117:337,j],31,3)
pred=h+off_s
for a,b in [(117,190),(190,255),(255,337)]:
 m=ok.copy();m[:a]=False;m[b:]=False
 print('range',a,b,'base diff from pred',np.mean(np.linalg.norm(t[m]-pred[m],axis=1)),np.percentile(np.linalg.norm(t[m]-pred[m],axis=1),90))
# translation step smoothness in manipulate
for name,x in [('base',t),('handpred',pred)]:
 v=np.diff(x,axis=0)
 print(name,'step median/p90/max',np.quantile(np.linalg.norm(v[122:336],axis=1),[.5,.9,1]),'acc p90',np.percentile(np.linalg.norm(np.diff(v[122:336],axis=0),axis=1),90))
PY"
```
<details><summary>output</summary>

```
ok ranges 121 332 206
117 190 n 65 off mean [ 0.03636879  0.14302683 -0.00936601] std [0.0286226  0.03113686 0.04781992] res p90 0.094302536547184
190 255 n 63 off mean [0.0378957  0.13950871 0.04942036] std [0.03436566 0.03741549 0.05328704] res p90 0.10719597041606903
255 337 n 78 off mean [-0.02341508  0.1216151   0.02871108] std [0.01485667 0.00892444 0.02974627] res p90 0.051862296089529976
range 117 190 base diff from pred 0.018578999 0.034352369606494904
range 190 255 base diff from pred 0.050510574 0.0861914932727814
range 255 337 base diff from pred 0.013353832 0.02180844210088253
base step median/p90/max [0.02581156 0.05718413 0.16349374] acc p90 0.09420364052057267
handpred step median/p90/max [0.02283715 0.03678138 0.06075149] acc p90 0.04336176738142968

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');kp=r['human_keypoints_world'].astype(float);T=r['T_world_mesh'].astype(float);c=r['contact_prob_right_left']
def frame(p):
 w=p[0]; idx=p[5]; mid=p[9]; pky=p[17]
 x=idx-pky;x/=np.linalg.norm(x)
 y=mid-w;y-=x*np.dot(x,y);y/=np.linalg.norm(y)
 z=np.cross(x,y);z/=np.linalg.norm(z)
 return np.stack([x,y,z],1)
Rh=np.array([frame(kp[i,21:42]) for i in range(440)]);Lh=np.array([frame(kp[i,42:63]) for i in range(440)])
ok=(c[:,0]>.5)&(c[:,1]>.5)
for name,H in [('R',Rh),('L',Lh)]:
 rel=np.array([H[i].T@T[i,:3,:3] for i in range(440)])
 # mean rotation via quaternion
 m=ok.copy();m[:117]=False;m[337:]=False
 rr=Rot.from_matrix(rel[m]);mean=rr.mean()
 err=rr*np.linalg.inv(mean);angs=np.degrees(np.linalg.norm(err.as_rotvec(),axis=1))
 print(name,'n',m.sum(),'res deg quant',np.quantile(angs,[0,.1,.5,.9,1]))
 # piecewise ranges
 for a,b in [(117,190),(190,255),(255,337)]:
  mm=m.copy();mm[:a]=False;mm[b:]=False;rr=Rot.from_matrix(rel[mm]);mean=rr.mean();angs=np.degrees(np.linalg.norm((rr*np.linalg.inv(mean)).as_rotvec(),axis=1));print(a,b,np.median(angs),np.percentile(angs,90))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 17, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/linalg/linalg.py", line 555, in inv
    _assert_stacked_2d(a)
  File "/opt/conda/lib/python3.11/site-packages/numpy/linalg/linalg.py", line 206, in _assert_stacked_2d
    raise LinAlgError('%d-dimensional array given. Array must be '
numpy.linalg.LinAlgError: 0-dimensional array given. Array must be at least two-dimensional

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');kp=r['human_keypoints_world'].astype(float);T=r['T_world_mesh'].astype(float);c=r['contact_prob_right_left']
def frame(p):
 w=p[0]; idx=p[5]; mid=p[9]; pky=p[17]
 x=idx-pky;x/=np.linalg.norm(x);y=mid-w;y-=x*np.dot(x,y);y/=np.linalg.norm(y);z=np.cross(x,y);z/=np.linalg.norm(z);return np.stack([x,y,z],1)
Rh=np.array([frame(kp[i,21:42]) for i in range(440)]);Lh=np.array([frame(kp[i,42:63]) for i in range(440)])
ok=(c[:,0]>.5)&(c[:,1]>.5)
for name,H in [('R',Rh),('L',Lh)]:
 rel=np.array([H[i].T@T[i,:3,:3] for i in range(440)]);m=ok.copy();m[:117]=False;m[337:]=False
 for a,b in [(117,190),(190,255),(255,337),(117,337)]:
  mm=m.copy();mm[:a]=False;mm[b:]=False;rr=Rot.from_matrix(rel[mm]);mean=rr.mean();angs=np.degrees(np.linalg.norm((rr*mean.inv()).as_rotvec(),axis=1));print(name,a,b,'n',mm.sum(),'res med/p90',np.median(angs),np.percentile(angs,90))
PY"
```
<details><summary>output</summary>

```
R 117 190 n 65 res med/p90 58.189387023111934 77.48769567303532
R 190 255 n 63 res med/p90 68.9910390453744 107.61059093268382
R 255 337 n 78 res med/p90 28.073216791330587 48.12735250846619
R 117 337 n 206 res med/p90 52.97305706096232 127.33953803241133
L 117 190 n 65 res med/p90 42.83880375162032 67.12901908695204
L 190 255 n 63 res med/p90 65.57271629175 109.74600980740756
L 255 337 n 78 res med/p90 30.66518642935305 64.80531805080494
L 117 337 n 206 res med/p90 62.75171557999852 128.52958759252155

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/align_xy.py <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import numpy as np
def align_xy(i,k,iters=3):
 ys,xs=np.where(objmask[i])
 if len(xs)<80:return None,None,None
 x0=max(0,xs.min()-25);x1=min(W,xs.max()+26);y0=max(0,ys.min()-25);y1=min(H,ys.max()+26);gt=objmask[i,y0:y1,x0:x1]
 a=k*np.pi/2;c,s=np.cos(a),np.sin(a);Rz=np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]],float);M=T[i].astype(float)@Rz
 ygt,xgt=np.where(gt);cent0=np.array([xgt.mean()+x0,ygt.mean()+y0])
 for it in range(iters):
  m=render_mask_world(M,x0,x1,y0,y1);ysp,xsp=np.where(m)
  if len(xsp)==0:break
  cent=np.array([xsp.mean()+x0,ysp.mean()+y0]);Tc=np.linalg.inv(np.array(cam['T_world_cam'])).astype(float);Mc=Tc@M;z=Mc[2,3]
  dx,dy=cent0-cent;Mc[0,3]+=dx*z/K[0,0];Mc[1,3]+=dy*z/K[1,1];M=np.array(cam['T_world_cam'])@Mc
 m=render_mask_world(M,x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum()
 # photo highpass
 im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255;col,m2=render_pose(M,x0,x1,y0,y1);val=m2&gt
 a1=cv2.GaussianBlur(col.mean(2),(0,0),2);b1=cv2.GaussianBlur(im.mean(2),(0,0),2);ah=a1-cv2.GaussianBlur(a1,(0,0),8);bh=b1-cv2.GaussianBlur(b1,(0,0),8);photo=float(np.corrcoef(ah[val],bh[val])[0,1]) if val.sum()>100 else 0
 return iou,photo,M
if __name__=='__main__':
 for i in [0,50,100,112,117,121,125,129,140,160,180,260,280,300,320,340,400]:
  z=[align_xy(i,k)[:2] for k in range(4)];print(i,[(round(a,3),round(b,3)) for a,b in z],flush=T …[+41 chars]
```
<details><summary>output</summary>

```
0 [(0.908, 0.533), (0.928, 0.578), (0.94, 0.573), (0.897, 0.543)]
50 [(0.903, 0.556), (0.921, 0.596), (0.937, 0.61), (0.893, 0.575)]
100 [(0.913, 0.64), (0.923, 0.7), (0.939, 0.679), (0.901, 0.638)]
112 [(0.912, 0.639), (0.929, 0.706), (0.938, 0.713), (0.896, 0.679)]
117 [(0.914, 0.709), (0.912, 0.654), (0.86, 0.634), (0.879, 0.609)]
121 [(0.904, 0.588), (0.9, 0.583), (0.852, 0.539), (0.868, 0.525)]
125 [(0.892, 0.47), (0.885, 0.466), (0.848, 0.411), (0.86, 0.395)]
129 [(0.842, 0.595), (0.828, 0.555), (0.798, 0.513), (0.804, 0.53)]
140 [(0.702, 0.241), (0.662, 0.152), (0.658, 0.218), (0.677, 0.212)]
160 [(0.756, 0.343), (0.725, 0.236), (0.728, 0.16), (0.743, 0.318)]
180 [(0.634, -0.115), (0.635, -0.239), (0.645, -0.177), (0.632, -0.217)]
260 [(0.702, 0.19), (0.702, 0.062), (0.7, 0.013), (0 …[+352 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def fr(i):cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,a=cap.read();return a
with h5py.File('/work/inputs/masks.h5') as f:om=np.asarray(f['object'])
# use crop around object, align 112 to each with ECC translation, compare intersect eroded
base=fr(112);mb=om[112];ys,xs=np.where(mb);x0=xs.min()-20;x1=xs.max()+21;y0=ys.min()-20;y1=ys.max()+21
g0=cv2.cvtColor(base[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
for j in range(108,126):
 im=fr(j);g=cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
 if j==112:warp=g
 else:
  warp=cv2.warpAffine(g,cv2.invertAffineTransform(cv2.estimateAffinePartial2D(g0,g,method=cv2.RANSAC)[0]),(g.shape[1],g.shape[0]),flags=cv2.INTER_LINEAR)
 mask=mb[y0:y1,x0:x1]&om[j,y0:y1,x0:x1];mask=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
 # normalized intensity residual after linear fit, highpass
 a=g0-cv2.GaussianBlur(g0,(0,0),4);b=warp-cv2.GaussianBlur(warp,(0,0),4)
 print(j,'mae',np.mean(np.abs(g0[mask]-warp[mask])),'hpmae',np.mean(np.abs(a[mask]-b[mask])),'corr',np.corrcoef(a[mask],b[mask])[0,1])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 12, in <module>
cv2.error: OpenCV(4.11.0) /io/opencv/modules/calib3d/src/ptsetreg.cpp:1108: error: (-215:Assertion failed) count >= 0 && to.checkVector(2) == count in function 'estimateAffinePartial2D'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def fr(i):cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,a=cap.read();return a
with h5py.File('/work/inputs/masks.h5') as f:om=np.asarray(f['object'])
base=fr(112);mb=om[112];ys,xs=np.where(mb);x0=xs.min()-20;x1=xs.max()+21;y0=ys.min()-20;y1=ys.max()+21
g0=cv2.cvtColor(base[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
for j in range(108,126):
 im=fr(j);g=cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
 warp_mat=np.eye(2,3,dtype=np.float32)
 try:
  cc,warp_mat=cv2.findTransformECC(g0,g,warp_mat,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-6))
  warp=cv2.warpAffine(g,warp_mat,(g.shape[1],g.shape[0]),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
 except Exception as e:print(e);warp=g;cc=0
 mask=mb[y0:y1,x0:x1]&om[j,y0:y1,x0:x1];mask=cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0
 a=g0-cv2.GaussianBlur(g0,(0,0),4);b=warp-cv2.GaussianBlur(warp,(0,0),4)
 print(j,'cc',round(cc,4),'shift',np.round(warp_mat[:,2],2),'hpmae',round(float(np.mean(np.abs(a[mask]-b[mask]))),2),'corr',round(float(np.corrcoef(a[mask],b[mask])[0,1]),3))
PY"
```
<details><summary>output</summary>

```
108 cc 0.9504 shift [ 0.31 -0.16] hpmae 1.5 corr 0.955
109 cc 0.9646 shift [ 0.35 -0.15] hpmae 1.4 corr 0.961
110 cc 0.9777 shift [ 0.24 -0.1 ] hpmae 1.22 corr 0.97
111 cc 0.9895 shift [ 0.1  -0.09] hpmae 0.95 corr 0.983
112 cc 1.0 shift [0. 0.] hpmae 0.0 corr 1.0
113 cc 0.9861 shift [-0.09  0.11] hpmae 0.93 corr 0.984
114 cc 0.9659 shift [-0.18  0.17] hpmae 1.28 corr 0.971
115 cc 0.939 shift [-0.22  0.27] hpmae 1.59 corr 0.943
116 cc 0.9053 shift [-0.27  0.37] hpmae 1.78 corr 0.925
117 cc 0.8769 shift [-0.37  0.35] hpmae 1.93 corr 0.913
118 cc 0.8591 shift [-0.42  0.4 ] hpmae 2.1 corr 0.892
119 cc 0.8375 shift [-0.47  0.41] hpmae 2.23 corr 0.871
120 cc 0.8226 shift [-0.49  0.46] hpmae 2.48 corr 0.833
121 cc 0.8318 shift [-0.52  0.52] hpmae 2.68 corr 0.796
122 cc 0.8288 shift [-0.45  0.56] …[+188 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p) as f:
 for i in [0,200,340,439]:
  d=f['aligned/front_stereo_camera_left'][i];print(i,type(d),d.shape,d.dtype,d.min(),d.max(),np.median(d),f['alignment/front_stereo_camera_left/scale'][i],f['alignment/front_stereo_camera_left/shift'][i])
PY"
```
<details><summary>output</summary>

```
0 <class 'numpy.ndarray'> (658995,) uint8 0 255 128.0 1.1378613 0.0
200 <class 'numpy.ndarray'> (660561,) uint8 0 255 128.0 1.1253703 0.0
340 <class 'numpy.ndarray'> (663206,) uint8 0 255 128.0 1.1451199 0.0
439 <class 'numpy.ndarray'> (650026,) uint8 0 255 127.0 1.0983771 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 d=f['aligned/front_stereo_camera_left'];print(d.attrs);print(d[0].attrs if hasattr(d[0],'attrs') else '');print(list(f['aligned/front_stereo_camera_left'].keys()) if hasattr(f['aligned/front_stereo_camera_left'],'keys') else '')
PY"
```
<details><summary>output</summary>

```
<Attributes of HDF5 object at 136136074516880>



```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned/front_stereo\\|front_stereo_camera_left\\|imdecode\\|cv2.decode\" -n /code/reconstruction/modules/v2d_cari4d /workspace/v2d_cari4d /code/reconstruction/modules/v2d_moge 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_export_utils.py:23:    "front_stereo_camera_left",
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_export_utils.py:23:    "front_stereo_camera_left",

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned.h5\\|aligned/.*front_stereo\\|decode.*depth\\|depth.*decode\" -n /code/reconstruction/modules/v2d_cari4d /code/reconstruction/modules/v2d_moge /workspace/v2d_cari4d 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py:107:def _input_cache_identity(export_seq: Path, depth_h5: Path, depth_provenance: Mapping[str, Any], mhr_init: Path, foundationpose_file: Path, cfg: Any, object_mesh_path: Path, object_pose_frame: Any, decoder_identity: Mapping[str, str], names: Sequence[str]) -> dict[str, Any]:
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py:150:    timings["depth_decode"] = time.perf_counter() - depth_started
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py:573:    cache_identity = _input_cache_identity(export_seq, depth_h5, depth_provenance, mhr_init, foundationpose_file, cfg, resolve_object_mesh_path(export_seq), object_pose_frame, layer.decoder_identit …[+4978 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -n '330,410p' /code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py; python - <<'PY'
import sys,h5py,numpy as np
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib')
from cari4d.prep.mhr_depth_h5 import read_metric_depth
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 d=read_metric_depth(f,'aligned','front_stereo_camera_left',340);print(d.shape,d.dtype,np.nanmin(d),np.nanmax(d),np.nanmedian(d))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
  File "/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py", line 377, in read_metric_depth
    path_key = str(Path(path).resolve())
                   ^^^^^^^^^^
  File "/opt/conda/lib/python3.11/pathlib.py", line 871, in __new__
    self = cls._from_parts(args)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/pathlib.py", line 509, in _from_parts
    drv, root, parts = self._parse_args(args)
                       ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/pathlib.py", line 493, in _parse_args
    a = os.fspath(a)
        ^^^^^^^^^^^^
TypeError: expected str, bytes or os.PathLike object, not File

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib')
from cari4d.prep.mhr_depth_h5 import read_metric_depth,depth_frame_names
p='/pipeline/depth/moge2/aligned.h5';names=depth_frame_names(p,'front_stereo_camera_left');print(names[:3],names[200],names[340],len(names))
for i in [0,200,340,439]:
 d=read_metric_depth(p,'aligned','front_stereo_camera_left',names[i]);print(i,d.shape,d.dtype,np.nanmin(d),np.nanmax(d),np.nanmedian(d))
PY"
```
<details><summary>output</summary>

```
['000000', '000001', '000002'] 000200 000340 440
0 (1152, 1536) float32 1.804 10.356 4.106
200 (1152, 1536) float32 1.773 10.067 3.957
340 (1152, 1536) float32 1.78 10.369 4.028
439 (1152, 1536) float32 1.736 9.886 3.913

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,h5py
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib');from cari4d.prep.mhr_depth_h5 import read_metric_depth,depth_frame_names
names=depth_frame_names('/pipeline/depth/moge2/aligned.h5','front_stereo_camera_left');p='/pipeline/depth/moge2/aligned.h5'
with h5py.File('/work/inputs/masks.h5') as f:om=np.asarray(f['object']);pm=np.asarray(f['person'])
for i in [0,50,100,117,140,160,180,260,280,300,320,340,400,439]:
 d=read_metric_depth(p,'aligned','front_stereo_camera_left',names[i]);v=d[om[i]];print(i,'n',len(v),'depth q',np.round(np.percentile(v,[10,25,50,75,90]),3),'mean',round(float(v.mean()),3))
PY"
```
<details><summary>output</summary>

```
0 n 9735 depth q [1.995 2.007 2.014 2.035 2.101] mean 2.038
50 n 9679 depth q [1.991 2.    2.007 2.028 2.092] mean 2.03
100 n 9740 depth q [1.975 1.99  2.007 2.027 2.088] mean 2.022
117 n 9591 depth q [2.012 2.025 2.039 2.059 2.104] mean 2.05
140 n 8370 depth q [1.994 2.001 2.012 2.028 2.07 ] mean 2.035
160 n 5892 depth q [2.085 2.096 2.111 2.152 2.453] mean 2.288
180 n 3654 depth q [2.419 2.443 2.454 2.504 2.917] mean 2.667
260 n 2214 depth q [3.221 3.233 3.246 3.532 4.263] mean 3.583
280 n 3178 depth q [3.121 3.13  3.154 3.237 3.286] mean 3.199
300 n 3820 depth q [2.912 2.926 2.942 2.991 3.047] mean 2.99
320 n 4094 depth q [2.683 2.723 2.749 2.775 2.861] mean 2.799
340 n 4184 depth q [2.702 2.727 2.753 2.795 2.839] mean 2.82
400 n 4404 depth q [2.7   2.743 2.77  2.797 2.913] mean 2.856
4 …[+61 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
r=np.load('/work/inputs/reconstruction.npz');Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tc=np.linalg.inv(Twc)
for i in [0,100,117,140,160,180,260,280,300,320,325,330,335,340,350,400,439]:
 A=Tc@r['T_world_mesh'][i].astype(float);print(i,np.round(A[:3,3],3))
PY"
```
<details><summary>output</summary>

```
0 [0.031 1.104 1.921]
100 [0.031 1.107 1.927]
117 [0.024 1.113 1.941]
140 [0.085 0.689 1.97 ]
160 [0.235 0.282 2.176]
180 [0.211 0.087 2.533]
260 [-0.335  0.031  3.437]
280 [-0.397  0.072  3.133]
300 [-0.234  0.088  2.743]
320 [-0.438  0.244  2.627]
325 [-0.447  0.254  2.634]
330 [-0.443  0.258  2.627]
335 [-0.444  0.257  2.668]
340 [-0.453  0.259  2.712]
350 [-0.454  0.26   2.715]
400 [-0.454  0.26   2.719]
439 [-0.453  0.259  2.709]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
# compare rendered depth distribution to moge within mask for poses
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import nvdiffrast.torch as dr
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib');from cari4d.prep.mhr_depth_h5 import read_metric_depth,depth_frame_names
names=depth_frame_names('/pipeline/depth/moge2/aligned.h5','front_stereo_camera_left')
# render depth function
def rdepth(M):
 i=0;ys,xs=np.where(objmask[i]);x0=xs.min()-20;x1=xs.max()+21;y0=ys.min()-20;y1=ys.max()+21;cw,ch=x1-x0,y1-y0
 vv=v;fc=faces;M=torch.tensor(M,dtype=torch.float32,device='cuda');vw=vv@M[:3,:3].T+M[:3,3];vc=vw@Tc_t[:3,:3].T+Tc_t[:3,3];z=vc[:,2];xp=K[0,0]*vc[:,0]/z+K[0,2];yp=K[1,1]*vc[:,1]/z+K[1,2];sx=(xp-x0)/cw*2-1;sy=(yp-y0)/ch*2-1;clip=torch.stack([sx*z,sy*z,z,z],1)[None];rast,_=dr.rasterize(gl,clip,fc,resolution=(ch,cw));dm,_=dr.interpolate(z[:,None],rast,fc);m=rast[0,...,3]>0;return dm[0,:,:,0].cpu().numpy(),m.cpu().numpy(),(x0,x1,y0,y1)
fits={0:T[340].copy(),1:T[340].copy()}
# use approximate xyz translation corrections from above in camera space via makeM and add z
p0=makeM(T[340],0,[.002,.002,0]);p1=makeM(T[340],0,[.002,.002,-.05])
for name,M in [('base',p0),('near',p1)]:
 dm,m,(x0,x1,y0,y1)=rdepth(M);depth=read_metric_depth('/pipeline/depth/moge2/aligned.h5','aligned','front_stereo_camera_left',names[340]);a=depth[y0:y1,x0:x1];gt=objmask[340,y0:y1,x0:x1];v=dm[m&gt];mg=a[m&gt]
 print(name,'render mesh pixels',len(v) …[+112 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
  File "<stdin>", line 9, in rdepth
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _nvdiffrast_c.interpolate_fwd(attr, rast, tri)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: interpolate_fwd_da(): Inputs attr, rast, tri m …[+26 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import nvdiffrast.torch as dr
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib');from cari4d.prep.mhr_depth_h5 import read_metric_depth,depth_frame_names
names=depth_frame_names('/pipeline/depth/moge2/aligned.h5','front_stereo_camera_left')
def rdepth(M):
 i=0;ys,xs=np.where(objmask[i]);x0=int(xs.min()-20);x1=int(xs.max()+21);y0=int(ys.min()-20);y1=int(ys.max()+21);cw,ch=x1-x0,y1-y0
 M=torch.tensor(M,dtype=torch.float32,device='cuda');vw=v@M[:3,:3].T+M[:3,3];vc=vw@Tc_t[:3,:3].T+Tc_t[:3,3];z=vc[:,2];xp=K[0,0]*vc[:,0]/z+K[0,2];yp=K[1,1]*vc[:,1]/z+K[1,2];sx=(xp-x0)/cw*2-1;sy=(yp-y0)/ch*2-1;clip=torch.stack([sx*z,sy*z,z,z],1).contiguous();rast,_=dr.rasterize(gl,clip,faces,resolution=(ch,cw));dm,_=dr.interpolate(z[:,None].contiguous(),rast,faces);m=rast[0,...,3]>0;return dm[0,:,:,0].detach().cpu().numpy(),m.cpu().numpy(),(x0,x1,y0,y1)
p0=makeM(T[340],0,[.002,.002,0]);p1=makeM(T[340],0,[.002,.002,-.05])
for name,M in [('base',p0),('near',p1)]:
 dm,m,(x0,x1,y0,y1)=rdepth(M);depth=read_metric_depth('/pipeline/depth/moge2/aligned.h5','aligned','front_stereo_camera_left',names[340]);a=depth[y0:y1,x0:x1];gt=objmask[340,y0:y1,x0:x1];v=dm[m&gt];mg=a[m&gt]
 print(name,'n',len(v),'render q',np.round(np.percentile(v,[10,50,90]),3),'moge q',np.round(np.percentile(mg,[10,50,90]),3),'mean diff',round(float(np.mean(v-mg)),3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "<stdin>", line 7, in rdepth
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 78, in forward
    out, out_db = _nvdiffrast_c.rasterize_fwd_cuda(raster_ctx.cpp_wrapper, pos, tri, resolution, ranges, peeling_idx)
              …[+163 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/all_scores.py <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from align_xy import align_xy
import numpy as np,time
F=440;D=np.zeros((F,4,2),np.float32);valid=np.zeros(F,bool)
t0=time.time()
for i in range(F):
 ys,xs=np.where(objmask[i])
 if len(xs)>=80:
  valid[i]=True
  for k in range(4):
   try:
    iou,photo,M=align_xy(i,k);D[i,k]=[iou,photo]
   except Exception as e:
    print('ERR',i,k,e);D[i,k]=[0,0]
 if i%50==0:print(i,time.time()-t0,flush=True)
np.savez('/work/scratch/all_scores.npz',D=D,valid=valid)
print('done',time.time()-t0)
PY
python /work/scratch/all_scores.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/all_scores.py", line 6, in <module>
    ys,xs=np.where(objmask[i])
                   ^^^^^^^
NameError: name 'objmask' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '1a sys.path.insert(0,\"/work/scratch\")\\nfrom opt_sil import *' /work/scratch/all_scores.py
python /work/scratch/all_scores.py"
```
<details><summary>output</summary>

```
0 0.21216630935668945
50 5.465678930282593
100 14.11534333229065
150 22.93735098838806
200 38.52012014389038
250 42.65284967422485
300 51.38416528701782
350 58.31199407577515
400 67.12331485748291
done 75.53697276115417

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/all_scores.npz');D=z['D'];valid=z['valid']
print('valid',valid.sum())
for a,b in [(0,113),(113,117),(117,130),(130,160),(160,190),(190,255),(255,300),(300,340),(340,440)]:
 print('\\n',a,b)
 for k in range(4):
  m=valid[a:b];print(k,'mask',round(float(D[a:b,k,0][m].mean()),3),'photo',round(float(D[a:b,k,1][m].mean()),3),end='; ')
 print()
# count per-frame photo winner, mask winner by ranges
for typ,j in [('mask',0),('photo',1)]:
 print('\\n',typ)
 for a,b in [(0,113),(117,130),(130,160),(160,190),(190,255),(255,300),(300,340),(340,440)]:
  m=valid[a:b];w=np.argmax(D[a:b,:,j],axis=1);vals=np.bincount(w[m],minlength=4);print(a,b,vals)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/proto.py <<'PY'
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');T0=r['T_world_mesh'].astype(np.float64);F=len(T0);kp=r['human_keypoints_world'].astype(float);c=r['contact_prob_right_left']
# static initial optimized from all_scores/previous: optimize automatic later, here hard values just prototype
a=np.deg2rad(175.810635);Rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
T=T0.copy();T[:113,:3,:3]=T0[:113,:3,:3]@Rz
# translation correction in camera coordinates
cam=json.load(open('/work/inputs/camera.json'));Rwc=np.array(cam['T_world_cam'])[:3,:3]
dc=np.array([-.0172860849,-.00478911623,-.00065746485]);T[:113,:3,3]+=Rwc@dc
# transition 113:117 slerp static optimized -> baseline 117
R0=Rot.from_matrix(T[112,:3,:3]);R1=Rot.from_matrix(T0[117,:3,:3]);t0=T[112,:3,3];t1=T0[117,:3,3]
for i in range(113,117):
 u=(i-112)/5;u=u*u*(3-2*u);Rsl=Slerp([0,1],Rot.concatenate([R0,R1]))(u);T[i,:3,:3]=Rsl.as_matrix();T[i,:3,3]=(1-u)*t0+u*t1
# dynamic 117:340 baseline, with hand translation gap 195:255
Rk=kp[:,21:42].mean(1);Lk=kp[:,42:63].mean(1);hand=(Rk+Lk)/2
# compute residual baseline-hand at gap boundaries, interpolate
for i in range(117,340):T[i]=T0[i]
gap0,gap1=192,256
off0=T0[gap0,:3,3]-hand[gap0];off1=T0[gap1,:3,3]-hand[gap1]
for i in range(gap0,gap1+1):
 u=(i-gap0)/(gap1-gap0);off=(1-u)*off0+u*off1;T[i,:3, …[+1045 chars]
```
<details><summary>output</summary>

```
0 [2.000e-03 1.180e+00 2.037e+00] [ 89.9 -87.   -2.9]
50 [0.007 1.199 2.068] [ 89.9 -87.   -2.9]
100 [0.003 1.183 2.043] [ 89.9 -87.   -2.9]
112 [0.003 1.183 2.042] [ 89.9 -87.   -2.9]
113 [0.004 1.184 2.044] [ 90.1 -78.   -3.3]
114 [0.007 1.187 2.047] [ 89.9 -56.4  -3.5]
115 [0.011 1.191 2.051] [ 89.7 -30.7  -3.8]
116 [0.014 1.194 2.054] [89.4 -9.1 -4.1]
117 [0.014 1.194 2.055] [89.2  0.1 -4.3]
120 [0.016 1.197 2.058] [88.9 -0.2 -4.4]
130 [0.023 1.069 2.067] [ 80.3 -19.3  -5. ]
140 [0.076 0.769 2.088] [ 79.1 -68.3  -7.4]
160 [0.256 0.37  2.287] [ 80.2 -64.8 -20.5]
180 [0.276 0.192 2.652] [111.3 -83.9 -27. ]
190 [0.35  0.194 2.899] [ -91.6  -58.  -179.3]
192 [0.363 0.202 2.963] [-107.5  -55.8 -169.2]
200 [0.43  0.156 3.082] [-119.3  -56.5 -167. ]
220 [0.401 0.162 3.388] [104.6  65.4  11.1] …[+562 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');from opt_sil import *
import numpy as np,cv2
P=np.load('/work/scratch/proto_poses.npy');B=T
# metric function render mask iou/photo at pose
def score(M,i):
 ys,xs=np.where(objmask[i]);
 if len(xs)<80:return None,None
 x0=max(0,xs.min()-20);x1=min(W,xs.max()+21);y0=max(0,ys.min()-20);y1=min(H,ys.max()+21);gt=objmask[i,y0:y1,x0:x1]
 m=render_mask_world(M,x0,x1,y0,y1);iou=(m&gt).sum()/(m|gt).sum();im=getframe(i)[y0:y1,x0:x1][...,::-1].astype(np.float32)/255;col,m2=render_pose(M,x0,x1,y0,y1);val=m2&gt
 a=cv2.GaussianBlur(col.mean(2),(0,0),2);b=cv2.GaussianBlur(im.mean(2),(0,0),2);ah=a-cv2.GaussianBlur(a,(0,0),8);bh=b-cv2.GaussianBlur(b,(0,0),8);photo=np.corrcoef(ah[val],bh[val])[0,1] if val.sum()>100 else np.nan
 return iou,photo
for a,b in [(0,113),(113,117),(117,190),(190,255),(255,340),(340,440),(0,440)]:
 vals=[]
 for i in range(a,b):
  sb=score(B[i],i);sp=score(P[i],i)
  if sb[0] is not None:vals.append((sb[0],sp[0],sb[1],sp[1]))
 v=np.array(vals);print(a,b,'n',len(v),'mask B/P',np.nanmean(v[:,0]),np.nanmean(v[:,1]),'photo B/P',np.nanmean(v[:,2]),np.nanmean(v[:,3]))
# temporal angular acceleration and translation
from scipy.spatial.transform import Rotation as R
def smooth_metrics(X,a,b):
 rv=R.from_matrix(X[a:b,:3,:3]).as_rotvec();v=np.diff(rv,axis=0);acc=np.diff(v,axis=0);tv=np.diff(X[a:b,:3,3],axis=0);tacc=np.diff(tv,axis=0)
 return np.percentile(np.linalg.norm(acc,axis=1),[50,90,99]),np.perce …[+188 chars]
```
<details><summary>output</summary>

```
0 113 n 113 mask B/P 0.8190494305965417 0.9388931094567953 photo B/P 0.21319478940804762 0.6159801949652555
113 117 n 4 mask B/P 0.8169790174815609 0.8304644859711967 photo B/P 0.2714780513449911 0.511759012100206
117 190 n 73 mask B/P 0.7162746360797898 0.7221837410899271 photo B/P 0.27487658789103164 0.2881935550345265
190 255 n 16 mask B/P 0.29510131622639046 0.313087024038007 photo B/P 0.03815619991977946 0.1893041707797394
255 340 n 85 mask B/P 0.7265906802299021 0.7279920004701772 photo B/P 0.35843492960166173 0.3638255102214559
340 440 n 100 mask B/P 0.836740878084198 0.8368263600884519 photo B/P 0.4765880425873267 0.49079395716398944
0 440 n 391 mask B/P 0.7628247233983219 0.7997635315741322 photo B/P 0.3177975143543725 0.45008988267670236
117 340 B (array([0.17890916, 0.5939197 ,  …[+661 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:a=f['object'][:].sum((1,2))
for i in range(110,341):
 if i%5==0:print(i,a[i],end='; ')
print('\\nlt200',[(i,int(a[i])) for i in range(110,341) if a[i]<200][:5], 'range', [i for i in range(110,341) if a[i]<200][:1],[i for i in range(110,341) if a[i]<200][-1:])
PY"
```
<details><summary>output</summary>

```
110 9813; 115 9704; 120 9487; 125 9304; 130 9050; 135 8775; 140 8370; 145 7468; 150 6801; 155 6430; 160 5892; 165 4852; 170 4274; 175 3944; 180 3654; 185 2629; 190 837; 195 361; 200 0; 205 0; 210 0; 215 0; 220 0; 225 0; 230 0; 235 0; 240 0; 245 0; 250 1948; 255 2022; 260 2214; 265 2261; 270 2213; 275 2334; 280 3178; 285 3257; 290 3592; 295 3659; 300 3820; 305 3701; 310 3651; 315 3921; 320 4094; 325 3715; 330 3641; 335 4239; 340 4184; 
lt200 [(198, 177), (199, 92), (200, 0), (201, 0), (202, 0)] range [198] [253]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent world-space traffic-cone trajectory.

The released baseline is per-frame and contains a symmetry-branch error in the
long initial rest phase.  This solver (1) resolves that branch and jointly fits
the one static pose using all sampled static silhouettes, (2) bridges from that
pose into the tracker'"'s post-grasp branch, (3) replaces the object translation
by the reconstructed two-hand midpoint only while the object is fully obscured,
and (4) applies deterministic Savitzky-Golay smoothing to the manipulated-segment
rotation and translation.  Since a stationary rigid object has one pose, both
observed rest phases are emitted with constant rotations (and constant poses).
\"\"\"
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

# Explicit phase decisions.  Optical flow over the object is below 0.1 px/frame
# through frame 112 and begins to rise around 113.  Frame 117 is the first
# stable post-grasp tracker pose.  The hands leave the cone just before frame
# 340, after which SAM reports the same table-top silhouette for all frames.
STATIC_END = 112
DYNAMIC_START = 117
TABLE_START = 34 …[+15860 chars]
```
<details><summary>output</summary>

```
{
  "static_pose_fit": {
    "twist_deg": 174.72135930514494,
    "camera_translation_correction_m": [
      -0.017009169781303336,
      -0.003219451520014642,
      -0.0002961882956643413
    ],
    "sample_mask_iou_before": 0.7385682118082917,
    "sample_mask_iou_after": 0.9450735158855755,
    "optimizer_success": true
  },
  "trajectory": {
    "occlusion_mask_run": [
      199,
      246
    ],
    "hand_translation_gap": [
      193,
      249
    ]
  },
  "phase_validation": [
    {
      "phase": "initial_rest",
      "frames": [
        0,
        112
      ],
      "n_scored": 113,
      "mask_iou_baseline": 0.8190494305965417,
      "mask_iou_result": 0.9454556155697442,
      "photo_ncc_baseline": 0.21319480313338932,
      "photo_ncc_result": 0.622817336168776
    },
    {
  …[+2315 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/work/inputs/masks.h5') as f:a=f['object'][:].sum((1,2))
for i in range(185,265):print(i,a[i])
PY"
```
<details><summary>output</summary>

```
185 2629
186 2177
187 1721
188 1377
189 1077
190 837
191 672
192 526
193 413
194 380
195 361
196 316
197 313
198 177
199 92
200 0
201 0
202 0
203 0
204 0
205 0
206 0
207 0
208 0
209 0
210 0
211 0
212 0
213 0
214 0
215 0
216 0
217 0
218 0
219 0
220 0
221 0
222 0
223 0
224 0
225 0
226 0
227 0
228 0
229 0
230 0
231 0
232 0
233 0
234 0
235 0
236 0
237 0
238 0
239 0
240 0
241 0
242 0
243 0
244 0
245 0
246 0
247 1444
248 1796
249 1792
250 1948
251 1861
252 0
253 0
254 1990
255 2022
256 2019
257 2045
258 2097
259 2180
260 2214
261 2152
262 2162
263 2130
264 2171

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/outputs/object_poses_world.npz /work/scratch/first_result.npz; sed -i 's/OCCLUSION_AREA_THRESHOLD = 100/OCCLUSION_AREA_THRESHOLD = 2000/' /work/outputs/solve.py
cd /work && python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > /work/scratch/run2.log
python - <<'PY'
import json
j=json.load(open('/work/outputs/validation.json'))
print(j['trajectory'])
for r in j['phase_validation']:print(r)
print(j['temporal_result'])
PY"
```
<details><summary>output</summary>

```
{'occlusion_mask_run': [187, 254], 'hand_translation_gap': [181, 257]}
{'phase': 'initial_rest', 'frames': [0, 112], 'n_scored': 113, 'mask_iou_baseline': 0.8190494305965417, 'mask_iou_result': 0.9454556155697442, 'photo_ncc_baseline': 0.21319480313338932, 'photo_ncc_result': 0.622817336168776}
{'phase': 'pick_transition', 'frames': [113, 116], 'n_scored': 4, 'mask_iou_baseline': 0.8169790174815609, 'mask_iou_result': 0.8309050912318106, 'photo_ncc_baseline': 0.2714780718457421, 'photo_ncc_result': 0.4969817614694599}
{'phase': 'carry_visible', 'frames': [117, 189], 'n_scored': 73, 'mask_iou_baseline': 0.7162746360797898, 'mask_iou_result': 0.7184935606055187, 'photo_ncc_baseline': 0.2748765939844633, 'photo_ncc_result': 0.2604469541011356}
{'phase': 'carry_occluded', 'frames': [190, 254], …[+1064 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py'
s=open(p).read()
s=s.replace('OCCLUSION_AREA_THRESHOLD = 2000','OCCLUSION_AREA_THRESHOLD = 0  # exact empty masks delimit the unusable core')
old='''    occluded = areas < OCCLUSION_AREA_THRESHOLD
    # Longest low-visibility run inside the carry phase.
    runs = []
    start = None
    for i in range(DYNAMIC_START, TABLE_START):
        if occluded[i] and start is None:
            start = i
        if not occluded[i] and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, TABLE_START - 1))
    if not runs:
        raise RuntimeError(\"No object-occlusion run found while carried\")
    occ0, occ1 = max(runs, key=lambda ab: ab[1] - ab[0])
    hand_center = 0.5 * (keypoints[:, 21:42].mean(axis=1) + keypoints[:, 42:63].mean(axis=1))
    gap0 = max(DYNAMIC_START, occ0 - 6)
    gap1 = min(TABLE_START - 1, occ1 + 3)
'''
new='''    occluded = areas <= OCCLUSION_AREA_THRESHOLD
    # Longest fully empty run inside the carry phase.  SAM briefly flickers
    # back on for a few pixels behind the body, so extend to the last empty
    # frame in the carry interval and then add three frames of hand-only use.
    runs = []
    start = None
    for i in range(DYNAMIC_START, TABLE_START):
        if occluded[i] and start is None:
            start = i
        if not occluded[i] and start is not None:
            runs.append((start, i - 1))
       …[+985 chars]
```
<details><summary>output</summary>

```
{'occlusion_mask_run': [200, 246], 'hand_translation_gap': [193, 256]}
initial_rest 0.8190494305965417 0.9454556155697442 0.21319480313338932 0.622817336168776
pick_transition 0.8169790174815609 0.8309050912318106 0.2714780718457421 0.4969817614694599
carry_visible 0.7162746360797898 0.7221837410899271 0.2748765939844633 0.28819356237546556
carry_occluded 0.29510131622639046 0.3114301711318821 0.038156200751533045 0.14908031249322154
place_approach 0.7265906802299021 0.7280326130500444 0.3584349365389661 0.3635884999308211
table_rest 0.836740878084198 0.8368263600884519 0.47658806007942217 0.49079397513451944
all_frames 0.7628247233983219 0.8016056493096727 0.31779752571115155 0.4503206173880219
{'rotation_accel_p90_rad_frame2': 0.12285314460928211, 'translation_accel_p90_m_frame2': 0.0121 …[+108 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,cv2,torch,trimesh,h5py
print('numpy',numpy.__version__)
print('scipy',scipy.__version__)
print('opencv',cv2.__version__)
print('torch',torch.__version__)
print('trimesh',trimesh.__version__)
print('h5py',h5py.__version__)
try:
 import nvdiffrast
 print('nvdiffrast',getattr(nvdiffrast,'__version__','unknown'),nvdiffrast.__file__)
except Exception as e:print(e)
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
opencv 4.11.0
torch 2.5.1+cu124
trimesh 5.1.1
h5py 3.16.0
nvdiffrast 0.4.0 /opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
print('shape dtype finite',T.shape,T.dtype,np.isfinite(T).all())
# orthonormal error
err=np.abs(T[:,:3,:3].transpose(0,2,1)@T[:,:3,:3]-np.eye(3)[None]).max()
print('orth err',err,'det range',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
print('static start max t spread',np.ptp(T[:113,:3,3],axis=0),'R diff',np.abs(T[:113,:3,:3]-T[0,:3,:3]).max())
print('static table spread',np.ptp(T[340:,:3,3],axis=0),'R diff',np.abs(T[340:,:3,:3]-T[340,:3,:3]).max())
# floor signed distances for initial mesh vertices
import trimesh
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=g.vertices
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);d=v@T[0,:3,:3].T+T[0,:3,3];sd=d@pl[:3]+pl[3]
print('plane signed dist initial q',np.quantile(sd,[0,.01,.05,.5,.95,1]))
# step maxima locations/result
for name,X in [('out',T),('base',np.load('/work/inputs/reconstruction.npz')['T_world_mesh'])]:
 rv=R.from_matrix(X[:,:3,:3]).as_rotvec();ang=np.degrees(np.linalg.norm(np.diff(rv,axis=0),axis=1));tr=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);print(name,'max ang',ang.argmax()+1,ang.max(),'max tr',tr.argmax()+1,tr.max())
PY"
```
<details><summary>output</summary>

```
shape dtype finite (440, 4, 4) float64 True
orth err 7.963663792853737e-07 det range 0.9999992871987587 1.000000000000001
static start max t spread [0. 0. 0.] R diff 0.0
static table spread [0. 0. 0.] R diff 0.0
plane signed dist initial q [-0.01091556 -0.00670714 -0.00283537  0.04456262  0.30340692  0.37998539]
out max ang 213 34.97511286603741 max tr 246 0.043399153515963275
base max ang 117 98.8066359914294 max tr 247 0.16349374

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/outputs');from solve import Renderer
import numpy as np,json,cv2
inp='/work/inputs';cam=json.load(open(inp+'/camera.json'));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];rend=Renderer(__import__('pathlib').Path(inp+'/object_mesh.glb'),cam)
cap=cv2.VideoCapture(inp+'/video.mp4');ims=[]
for i in [0,112,117,140,160,180,200,220,240,260,300,340,400,439]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();
 # crop around projected object full render mask, determine bbox
 m=rend.mask(T[i],(0,rend.W,0,rend.H));ys,xs=np.where(m)
 if len(xs):
  x0=max(0,xs.min()-45);x1=min(rend.W,xs.max()+46);y0=max(0,ys.min()-45);y1=min(rend.H,ys.max()+46)
 else:x0,x1,y0,y1=500,900,350,750
 crop=im[y0:y1,x0:x1].copy();mc=m[y0:y1,x0:x1];crop[mc]=(crop[mc]*.3+np.array([0,255,0])*.7).astype(np.uint8)
 crop=cv2.resize(crop,(240,300));cv2.putText(crop,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);ims.append(crop)
while len(ims)%5:ims.append(np.zeros((300,240,3),np.uint8))
cv2.imwrite('/work/scratch/final_contact.jpg',np.vstack([np.hstack(ims[j:j+5]) for j in range(0,len(ims),5)]),[cv2.IMWRITE_JPEG_QUALITY,94])
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
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
for i in range(188,260):
 v=R.from_matrix(T[i,:3,:3].T@T[i-1,:3,:3]).as_rotvec();vb=R.from_matrix(B[i,:3,:3].T@B[i-1,:3,:3]).as_rotvec()
 if i%2==0: print(i,'res',round(np.degrees(np.linalg.norm(v)),1),'base',round(np.degrees(np.linalg.norm(vb)),1),'t',np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
188 res 5.0 base 21.2 t [0.336 0.183 2.839]
190 res 4.5 base 4.1 t [0.351 0.195 2.901]
192 res 5.8 base 7.6 t [0.361 0.197 2.956]
194 res 4.1 base 4.8 t [0.373 0.189 2.976]
196 res 2.7 base 3.9 t [0.392 0.181 2.996]
198 res 2.1 base 9.5 t [0.409 0.165 3.028]
200 res 1.1 base 9.6 t [0.425 0.143 3.062]
202 res 4.6 base 0.0 t [0.442 0.138 3.095]
204 res 1.4 base 0.0 t [0.456 0.152 3.118]
206 res 0.3 base 3.9 t [0.46  0.154 3.13 ]
208 res 1.4 base 3.9 t [0.459 0.149 3.153]
210 res 15.1 base 0.0 t [0.454 0.159 3.193]
212 res 26.8 base 40.7 t [0.444 0.164 3.223]
214 res 27.2 base 7.9 t [0.436 0.157 3.253]
216 res 12.6 base 0.0 t [0.429 0.16  3.295]
218 res 8.5 base 0.0 t [0.413 0.165 3.339]
220 res 9.8 base 0.1 t [0.398 0.154 3.375]
222 res 5.3 base 7.3 t [0.387 0.135 3.41 ]
224 res 5.6 base 0.0 …[+766 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_occ_smooth.py <<'PY'
import sys,json,numpy as np,cv2,h5py
from pathlib import Path
sys.path.insert(0,'/work/outputs');from solve import Renderer,validation_metrics,temporal_metrics,smooth_quaternions,DYNAMIC_START,TABLE_START
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
inp=Path('/work/inputs');cam=json.load(open(inp/'camera.json'));R=Renderer(inp/'object_mesh.glb',cam);T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load(inp/'reconstruction.npz')['T_world_mesh']
with h5py.File(inp/'masks.h5') as f:om=np.asarray(f['object']);pm=np.asarray(f['person'])
cap=cv2.VideoCapture(str(inp/'video.mp4'));frames={}
for i in range(440):_,frames[i]=cap.read()
for w in [9,13,17,21,25,31]:
 X=T.copy();sl=slice(190,257);q=Rot.from_matrix(X[sl,:3,:3]).as_quat()
 for i in range(1,len(q)):
  if np.dot(q[i],q[i-1])<0:q[i]*=-1
 for j in range(4):q[:,j]=savgol_filter(q[:,j],w,3,mode='interp')
 q/=np.linalg.norm(q,axis=1,keepdims=True);X[sl,:3,:3]=Rot.from_quat(q).as_matrix()
 # metrics occ only
 rows=validation_metrics(R,frames,B,X,om,pm);occ=[r for r in rows if r['phase']=='carry_occluded'][0];tm=temporal_metrics(X)
 print(w,occ,tm)
PY
python /work/scratch/test_occ_smooth.py"
```
<details><summary>output</summary>

```
9 {'phase': 'carry_occluded', 'frames': [190, 254], 'n_scored': 16, 'mask_iou_baseline': 0.29510131622639046, 'mask_iou_result': 0.31238296278609246, 'photo_ncc_baseline': 0.038156200751533045, 'photo_ncc_result': 0.1505090002574209} {'rotation_accel_p90_rad_frame2': 0.10876601971245863, 'translation_accel_p90_m_frame2': 0.012108943038437526, 'max_step_rotation_deg': 30.44853718611545, 'max_step_translation_m': 0.043399153515963275}
13 {'phase': 'carry_occluded', 'frames': [190, 254], 'n_scored': 16, 'mask_iou_baseline': 0.29510131622639046, 'mask_iou_result': 0.3126282271355413, 'photo_ncc_baseline': 0.038156200751533045, 'photo_ncc_result': 0.15750600117248695} {'rotation_accel_p90_rad_frame2': 0.1016321882726517, 'translation_accel_p90_m_frame2': 0.012108943038437526, 'max_step_rotation …[+1832 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');from solve import temporal_metrics,DYNAMIC_START,TABLE_START
from scipy.signal import savgol_filter
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for w in [7,9,11,13,15]:
 X=T.copy()
 for j in range(3):X[DYNAMIC_START:TABLE_START,j,3]=savgol_filter(X[DYNAMIC_START:TABLE_START,j,3],w,2,mode='interp')
 print(w,temporal_metrics(X))
PY"
```
<details><summary>output</summary>

```
7 {'rotation_accel_p90_rad_frame2': 0.12285314460928211, 'translation_accel_p90_m_frame2': 0.0069932702884769925, 'max_step_rotation_deg': 34.97511286603741, 'max_step_translation_m': 0.03816959672166883}
9 {'rotation_accel_p90_rad_frame2': 0.12285314460928211, 'translation_accel_p90_m_frame2': 0.005911871596869144, 'max_step_rotation_deg': 34.97511286603741, 'max_step_translation_m': 0.03665351609324383}
11 {'rotation_accel_p90_rad_frame2': 0.12285314460928211, 'translation_accel_p90_m_frame2': 0.00464425159011614, 'max_step_rotation_deg': 34.97511286603741, 'max_step_translation_m': 0.03626958254389357}
13 {'rotation_accel_p90_rad_frame2': 0.12285314460928211, 'translation_accel_p90_m_frame2': 0.004232089222791094, 'max_step_rotation_deg': 34.97511286603741, 'max_step_translation_m': 0.0 …[+223 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace('ROT_SMOOTH_WINDOW = 9\\nTRANS_SMOOTH_WINDOW = 7','ROT_SMOOTH_WINDOW = 9\\nTRANS_SMOOTH_WINDOW = 7\\n# No image evidence exists in the empty-mask run; a wider rotation window is\\n# justified there because a hand-held rigid object cannot physically jitter.\\nOCCLUSION_ROT_SMOOTH_WINDOW = 25')
old='''    for j in range(3):
        out[DYNAMIC_START:TABLE_START, j, 3] = savgol_filter(
            out[DYNAMIC_START:TABLE_START, j, 3],
            TRANS_SMOOTH_WINDOW,
            2,
            mode=\"interp\",
        )

    # Smoothly bridge the corrected rest pose into the first tracked carry pose.
'''
new='''    for j in range(3):
        out[DYNAMIC_START:TABLE_START, j, 3] = savgol_filter(
            out[DYNAMIC_START:TABLE_START, j, 3],
            TRANS_SMOOTH_WINDOW,
            2,
            mode=\"interp\",
        )

    # There are no object pixels in the middle of the carry.  Re-smooth that
    # interval with a wider temporal window; the hand midpoint still supplies
    # the gross translation, while rotation must obey rigid-body continuity.
    occ_a = max(DYNAMIC_START, occ0 - 10)
    occ_b = min(TABLE_START, last_empty + 4)
    out[occ_a:occ_b, :3, :3] = smooth_quaternions(
        Rot.from_matrix(out[occ_a:occ_b, :3, :3]), OCCLUSION_ROT_SMOOTH_WINDOW
    ).as_matrix()

    # Smoothly bridge the corrected rest pose into the first tracked carry pose.
'''
assert old in s;s= …[+540 chars]
```
<details><summary>output</summary>

```
/usr/bin/bash: line 36: /usr/bin/time: No such file or directory
{'twist_deg': 174.72135930514494, 'camera_translation_correction_m': [-0.017009169781303336, -0.003219451520014642, -0.0002961882956643413], 'sample_mask_iou_before': 0.7385682118082917, 'sample_mask_iou_after': 0.9450735158855755, 'optimizer_success': True}
{'occlusion_mask_run': [200, 246], 'hand_translation_gap': [193, 256]}
{'phase': 'initial_rest', 'frames': [0, 112], 'n_scored': 113, 'mask_iou_baseline': 0.8190494305965417, 'mask_iou_result': 0.9454556155697442, 'photo_ncc_baseline': 0.21319480313338932, 'photo_ncc_result': 0.622817336168776}
{'phase': 'pick_transition', 'frames': [113, 116], 'n_scored': 4, 'mask_iou_baseline': 0.8169790174815609, 'mask_iou_result': 0.8309050912318106, 'photo_ncc_baseline': 0.2714780718 …[+1619 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /work && TIMEFORMAT='ELAPSED=%3R'; { time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > /work/scratch/final_run.log; } 2> /work/scratch/final_time.log
cat /work/scratch/final_time.log
python - <<'PY'
import json
j=json.load(open('/work/outputs/validation.json'))
print(j['trajectory'])
for r in j['phase_validation']:print(r['phase'],r['mask_iou_baseline'],r['mask_iou_result'],r['photo_ncc_baseline'],r['photo_ncc_result'])
print(j['temporal_result'])
PY"
```
<details><summary>output</summary>

```
ELAPSED=15.036
{'occlusion_mask_run': [200, 246], 'hand_translation_gap': [193, 256]}
initial_rest 0.8190494305965417 0.9454556155697442 0.21319480313338932 0.622817336168776
pick_transition 0.8169790174815609 0.8309050912318106 0.2714780718457421 0.4969817614694599
carry_visible 0.7162746360797898 0.7221837410899271 0.2748765939844633 0.28819356237546556
carry_occluded 0.29510131622639046 0.31616893388981926 0.038156200751533045 0.18229232967501882
place_approach 0.7265906802299021 0.7271649560332282 0.3584349365389661 0.3639879141389745
table_rest 0.836740878084198 0.8368263600884519 0.47658806007942217 0.49079397513451944
all_frames 0.7628247233983219 0.8016109417846028 0.31779752571115155 0.45168505447884255
{'rotation_accel_p90_rad_frame2': 0.10157494330194834, 'translation_accel_p90_ …[+125 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py /work/outputs/validation.json'
```
<details><summary>output</summary>

```
8a27ac300581d8a3ca9e92e4e53a0eb1c368963f9cad74c8f7ab22e71841aa46  /work/outputs/object_poses_world.npz
347e683d744858df1f7e25d2d3bc6246178ef3f99877ea9eab759973286f5945  /work/outputs/solve.py
467fe06766473bce8a2c14839c9349569d1541759e32c37ff85a5667590c7e8b  /work/outputs/validation.json

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json,numpy as np
j=json.load(open('/work/outputs/validation.json'));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
print(np.ptp(T[:113,:3,3],axis=0),np.abs(T[:113,:3,:3]-T[0,:3,:3]).max())
print(np.ptp(T[340:,:3,3],axis=0),np.abs(T[340:,:3,:3]-T[340,:3,:3]).max())
print(T.shape,T.dtype,np.isfinite(T).all())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Traffic-cone 6-DoF trajectory recovery

## Result

- Output: "'`object_poses_world.npz`, key `T_world_mesh`, shape `(440, 4, 4)`, `float64`.
- Final solve command:
  `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
- Final run time: **15.0 s** on the available RTX 6000 Ada GPU. Peak GPU memory remained well below the 9 GB limit.
- No ground truth, challenge annotations, or challenge trajectory files were used.

## Diagnosed phases

The object masks, object optical flow, hand contacts, released tracker rotations, and rendered overlays support these phases:

| Frames | Phase | Treatment |
|---|---|---|
| 0-112 | Cone at rest on floor | One globally fitted pose for every frame |
| 113-116 | Grasp and lift transition | Smooth SE(3)/quaternion bridge into the tracked carry pose |
| 117-189 | Carried, mostly visible | Smoothed tracker trajectory |
| 190-254 | Carried behind the torso | Translation from interpolated two-hand midpoint; wide-window rotation smoothing |
| 255-339 | Transport and placement approach | Smoothed tracker trajectory |
| 340-439 | Cone at rest on table | One pooled pose for every frame |

The longest fully empty object-mask run is 200-246. SAM briefly reports a few object pixels around 247-251 and then empty again at 252-253, so the hand-derived translation gap is extended to 193-256.

## What was changed and why

### 1. Initial rest-phase symmetry branch
 …[+7789 chars]
```
<details><summary>output</summary>

```
total 96K
-rw-rw-r-- 1 hanzhou hanzhou 6.9K Oct 20 21:57 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 2.2K Oct 20 21:57 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:55 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  56K Oct 20 21:57 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  18K Oct 20 21:57 solve.py
-rw-rw-r-- 1 hanzhou hanzhou 3.1K Oct 20 21:57 validation.json

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /work
sha256sum /work/outputs/object_poses_world.npz > /work/scratch/hash_before.txt
TIMEFORMAT='ELAPSED=%3R'; { time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > /work/scratch/final_rerun.log; } 2> /work/scratch/final_rerun_time.txt
sha256sum /work/outputs/object_poses_world.npz > /work/scratch/hash_after.txt
cat /work/scratch/final_rerun_time.txt
cat /work/scratch/hash_before.txt /work/scratch/hash_after.txt
python - <<'PY'
import numpy as np,json,hashlib
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
assert T.shape==(440,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
assert np.allclose(T[:,3,:],np.array([0,0,0,1.]))
assert np.array_equal(T[:113],np.repeat(T[:1],113,axis=0))
assert np.array_equal(T[340:],np.repeat(T[340:341],100,axis=0))
v=json.load(open('/work/outputs/validation.json'))
assert len(v['phase_validation'])==7
print('FINAL_CHECK_OK',T.shape,T.dtype,'sha',hashlib.sha256(open('/work/outputs/object_poses_world.npz','rb').read()).hexdigest())
PY
find /work/outputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort"
```
<details><summary>output</summary>

```
ELAPSED=17.437
8a27ac300581d8a3ca9e92e4e53a0eb1c368963f9cad74c8f7ab22e71841aa46  /work/outputs/object_poses_world.npz
8a27ac300581d8a3ca9e92e4e53a0eb1c368963f9cad74c8f7ab22e71841aa46  /work/outputs/object_poses_world.npz
FINAL_CHECK_OK (440, 4, 4) float64 sha 8a27ac300581d8a3ca9e92e4e53a0eb1c368963f9cad74c8f7ab22e71841aa46
REPORT.md 6974 bytes
TOOLS.md 2184 bytes
object_poses_world.npz 56598 bytes
solve.py 18010 bytes
validation.json 3116 bytes

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/Final run time: \\*\\*15.0 s\\*\\*/Final run time: **17.4 s**/' /work/outputs/REPORT.md
grep -n \"Final run time\\|All scored\\|Rotation acceleration\" /work/outputs/REPORT.md
ls -la /work/outputs"
```
<details><summary>output</summary>

```
8:- Final run time: **17.4 s** on the available RTX 6000 Ada GPU. Peak GPU memory remained well below the 9 GB limit.
68:| All scored frames | 0-439 | 391 | 0.763 | **0.802** | 0.318 | **0.452** |
74:| Rotation acceleration, p90 (rad/frame²) | 0.594 | **0.102** |
total 104
drwxrwxr-x 3 hanzhou hanzhou  4096 Oct 20 21:58 .
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:25 ..
-rw-rw-r-- 1 hanzhou hanzhou  6974 Oct 20 21:58 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou  2184 Oct 20 21:57 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:55 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 56598 Oct 20 21:58 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 18010 Oct 20 21:57 solve.py
-rw-rw-r-- 1 hanzhou hanzhou  3116 Oct 20 21:58 validation.json

```
</details>

**Agent:** Completed the trajectory recovery task.

### Outputs
- `/work/outputs/object_poses_world.npz` — `(440, 4, 4) float64`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, validation, confidence
- `/work/outputs/TOOLS.md` — tools and dependencies
- `/work/outputs/validation.json` — detailed validation metrics

### Key results
- Corrected a long initial rest-phase symmetry-branch error, fitting a pooled pose with a 174.72° twist.
- Replaced the tracker’s ~99° pickup jump with a smooth pose bridge.
- Used hand-anchored translation through the 193–256 occlusion interval.
- Enforced constant poses for both static phases and smoothed the carried trajectory.

Validation across 391 scored frames:
- Mask IoU: **0.763 → 0.802**
- Photometric NCC: **0.318 → 0.452**
- Rotation acceleration p90: **0.594 → 0.102 rad/frame²**
- Translation acceleration p90: **0.0933 → 0.0121 m/frame²**

Final deterministic run completed in **17.4 s**. Re-running produced the identical output SHA-256:

`8a27ac300581d8a3ca9e92e4e53a0eb1c368963f9cad74c8f7ab22e71841aa46`
