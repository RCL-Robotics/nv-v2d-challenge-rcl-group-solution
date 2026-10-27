# T1_ds_ep010

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T16:53:57Z

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
/usr/bin/bash -c "sed -n '1,80p' /work/AGENT_TASK.md"
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
and translation both matter; a  …[+6115 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '81,160p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
    tracking, and its refinement by the CARI4D network), mesh-file frame, for `K_pipeline`. Their depth scale is not
    calibrated to the world; `T_world_mesh` already includes a per-frame depth correction.
  - `human_vertices_world` (F,V,3, every 4th vertex), `human_keypoints_world` (F,70,3; MHR keypoints, right hand
    21-41, left hand 42-62) of the reconstructed human in world; `contact_prob_right_left` (F,2).
- `masks.h5` - `object` and `person` (F,1152,1536) boolean SAM2 masks (may be imperfect; you may recompute them).

## Tools you may use
- This container's software (Python, torch + CUDA, nvdiffrast, opencv, trimesh, open3d, ...). The CARI4D pipeline
  modules are installed under `/workspace` (v2d_sam2, v2d_moge, v2d_foundation_pose, v2d_sam3d_body, v2d_cari4d);
  their source an …[+2877 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work -maxdepth 2 -type f -printf '%p %s\\n' | sort | head -100; echo '---dirs'; find /work -maxdepth 2 -type d -printf '%p\\n' | sort"
```
<details><summary>output</summary>

```
/work/.codex_home/.sandbox_migration 3
/work/.codex_home/config.toml 361
/work/.codex_home/goals_1.sqlite 4096
/work/.codex_home/goals_1.sqlite-shm 32768
/work/.codex_home/goals_1.sqlite-wal 78312
/work/.codex_home/installation_id 36
/work/.codex_home/logs_2.sqlite 49152
/work/.codex_home/logs_2.sqlite-shm 32768
/work/.codex_home/logs_2.sqlite-wal 119512
/work/.codex_home/memories_1.sqlite 4096
/work/.codex_home/memories_1.sqlite-shm 32768
/work/.codex_home/memories_1.sqlite-wal 61832
/work/.codex_home/model_catalog.json 3042
/work/.codex_home/queue_1.sqlite 4096
/work/.codex_home/queue_1.sqlite-shm 32768
/work/.codex_home/queue_1.sqlite-wal 82432
/work/.codex_home/state_5.sqlite 4096
/work/.codex_home/state_5.sqlite-shm 32768
/work/.codex_home/state_5.sqlite-wal 2484392
/work/.codex_home/ …[+891 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/inputs/task.txt; echo; cat /work/inputs/episode.json; echo; cat /work/inputs/camera.json; echo; cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk to the white desk, use both your hands to grasp on the edge of the table closest to you, and lower the desk to the ground.

{
 "episode_index": 10,
 "object": "white_desk",
 "frames": 794
}
{
 "K": [
  [
   820.0005493164062,
   0.0,
   758.1000366210938
  ],
  [
   0.0,
   820.0005493164062,
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
   797.921630859375,
   0.0,
   768.0
  ],
  [
   0.0,
   797.921630859375,
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
   0.9998582665216487,
   0.0013641948371882612,
   -0.016780519681155497,
   0.16532575245244926
  ],
  [
   -0.0009022997539102226,
   0.9996213272428819,
   0.027502508546186605,
   0.005527359122372545
  ],
  [
   0.016811684135670392,
   -0.027483469461208176,
   0.9994808783478 …[+623 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1955
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/05_object_pose.json 1772
/pipeline/.stages/06_coconet.json 2828
/pipeline/.stages/07_refinement.json 2131
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 979101056
/pipeline/depth/moge2/depth_report.json 1810
/pipeline/depth/moge2/intrinsics.pkl 84571
/pipeline/depth/moge2/raw.npy 2809921664
/pipeline/export/episode_000010/edex 561
/pipeline/export/episode_000010/wild_export.json 2779
/pipeline/foundationpose/episode_000010.pkl 226981
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 6072935 …[+548 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /work/inputs/video.mp4; python - <<'PY'
import cv2
v=cv2.VideoCapture('/work/inputs/video.mp4'); print('frames',int(v.get(7)),'fps',v.get(5),'size',v.get(3),v.get(4))
PY"
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=26.466667
nb_frames=794
frames 794 fps 30.0 size 1536.0 1152.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json, trimesh
p='/work/inputs/reconstruction.npz'
z=np.load(p)
print('NPZ keys:')
for k in z.files:
 a=z[k]
 print(k, a.shape, a.dtype, 'min/max', (float(np.nanmin(a)),float(np.nanmax(a))) if a.size and np.issubdtype(a.dtype,np.number) else '')
with h5py.File('/work/inputs/masks.h5') as f:
 print('masks keys',list(f.keys()))
 for k in f:
  print(k, f[k].shape, f[k].dtype)
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=s.to_geometry()
print('scene geoms',len(g),type(g))
for i,(name,geo) in enumerate(g.items()):
 print(name, 'v',len(geo.vertices),'f',len(geo.faces),'bounds',geo.bounds,'extents',geo.extents,'visual',type(geo.visual).__name__)
PY"
```
<details><summary>output</summary>

```
NPZ keys:
T_world_mesh (794, 4, 4) float32 min/max (-0.7167591452598572, 3.700427293777466)
T_cam_mesh_foundationpose (794, 4, 4) float32 min/max (-0.7240909934043884, 3.552736759185791)
T_cam_mesh_coconet (794, 4, 4) float32 min/max (-0.7108051180839539, 3.4738004207611084)
human_vertices_world (794, 4610, 3) float32 min/max (-1.2177702188491821, 4.065323352813721)
human_keypoints_world (794, 70, 3) float32 min/max (-1.2106273174285889, 4.060754299163818)
contact_prob_right_left (794, 2) float32 min/max (4.55703313625988e-12, 1.0)
masks keys ['object', 'person']
object (794, 1152, 1536) bool
person (794, 1152, 1536) bool
Traceback (most recent call last):
  File "<stdin>", line 14, in <module>
TypeError: object of type 'Trimesh' has no len()

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for f in /pipeline/.stages/*.json /pipeline/pipeline_report.json /pipeline/export/episode_000010/wild_export.json /pipeline/depth/moge2/depth_report.json; do echo ==== $f; cat $f; done'
```
<details><summary>output</summary>

```
==== /pipeline/.stages/01_depth.json
{
  "name": "01_depth",
  "outputs": [
    {
      "mtime_ns": 1791211321259777138,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000010/depth/moge2/raw.npy",
      "size": 2809921664
    },
    {
      "mtime_ns": 1791211321584777137,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000010/depth/moge2/intrinsics.pkl",
      "size": 84571
    },
    {
      "mtime_ns": 1791211323899777128,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000010/depth/moge2/depth_report.json",
      "size": 1810
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/mhr_wild_depth.py",
      "--video",
      "/home/hanzhou/Projects/v2d/run/wo …[+27125 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=s.to_geometry()
print(type(s),type(g))
if isinstance(g,trimesh.Trimesh): g={'mesh':g}
for name,geo in g.items():
 print(name, type(geo), 'v',len(geo.vertices),'f',len(geo.faces),'bounds\\n',geo.bounds,'\\nextents',geo.extents,'visual',type(geo.visual).__name__)
 print('uv',getattr(geo.visual,'uv',None).shape if hasattr(getattr(geo.visual,'uv',None),'shape') else None)
 print('material',geo.visual.material)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.scene.scene.Scene'> <class 'trimesh.base.Trimesh'>
mesh <class 'trimesh.base.Trimesh'> v 296588 f 98906 bounds
 [[-0.18534426 -0.31789014 -0.51160693]
 [ 0.7292825   0.30986634  0.5207454 ]] 
extents [0.91462676 0.62775648 1.03235233] visual TextureVisuals
uv (296588, 2)
material <trimesh.visual.material.PBRMaterial object at 0x7a15748fee10>

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz')
for k in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 T=z[k].astype(float)
 R=T[:,:3,:3]; t=T[:,:3,3]
 # rot angle neighbor and translation
 rel=np.einsum('fji,fjk->fik',R[:-1],R[1:])
 ang=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
 dt=np.linalg.norm(t[1:]-t[:-1],axis=1)
 print('\\n',k)
 print('t range',t.min(0),t.max(0),'pathlen',dt.sum())
 for q in [0,10,25,50,75,90,95,99,100]: print('angp',q,np.percentile(ang,q))
 print('angles >10',np.where(ang>10)[0]+1)
 print('angles >30',np.where(ang>30)[0]+1)
 print('dtp',np.percentile(dt,[0,50,90,95,99,100]))
 print('dt>0.03',np.where(dt>.03)[0]+1)
c=z['contact_prob_right_left']
print('\\ncontact max/mean',c.max(0),c.mean(0))
for th in [.1,.5,.9]: print(th,np.where(np.max(c,1)>th)[0].min(initial=9999),np.where(np.max(c,1)>th)[0].max(initial=-1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 print(list(f.keys()))
 for k in f:
  print(k,f[k].shape,f[k].dtype)
with h5py.File('/pipeline/work/masks_k0.h5','r') as f:
 print('masks',list(f.keys()),f[list(f.keys())[0]].shape)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/vis && python - <<'PY'
import cv2, numpy as np
p='/work/inputs/video.mp4'; cap=cv2.VideoCapture(p)
idxs=np.arange(0,794,40)
ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 im=cv2.resize(im,(384,288),interpolation=cv2.INTER_AREA)
 cv2.putText(im,str(i),(8,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2,cv2.LINE_AA)
 ims.append(im)
while len(ims)%5: ims.append(np.zeros_like(ims[0]))
rows=[np.hstack(ims[i:i+5]) for i in range(0,len(ims),5)]
out=np.vstack(rows); cv2.imwrite('/work/scratch/vis/contact.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,90])
print(out.shape)
PY"
```
<details><summary>output</summary>

```
(1152, 1920, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, cv2, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); t=T[:,:3,3]; R=T[:,:3,:3]
c=z['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as f:
 obj=f['object'][:].astype(np.uint8); per=f['person'][:].astype(np.uint8)
areas=obj.sum((1,2)); ious=[]
for i in range(len(obj)-1): ious.append((obj[i]&obj[i+1]).sum()/max(1,(obj[i]|obj[i+1]).sum()))
ious=np.array(ious)
fig,axs=plt.subplots(5,1,figsize=(16,14),sharex=True)
for j,n in enumerate('xyz'): axs[0].plot(t[:,j],label=n)
axs[0].legend();axs[0].set_ylabel('translation')
axs[1].plot(c[:,0],label='right');axs[1].plot(c[:,1],label='left');axs[1].legend();axs[1].set_ylabel('contact')
axs[2].plot(areas);axs[2].set_ylabel('object area')
axs[3].plot(ious);axs[3].set_ylabel('mask IoU t,t+1')
# rotation via quat
ang=np.degrees(np.arccos(np.clip((np.trace(np.einsum('fji,fjk->fik',R[:-1],R[1:]),axis1=1,axis2=2)-1)/2,-1,1)))
axs[4].plot(ang);axs[4].set_ylabel('rot deg/frame')
for ax in axs: ax.grid(True)
plt.tight_layout();plt.savefig('/work/scratch/vis/traj.png',dpi=120)
print('areas percentiles',np.percentile(areas,[0,5,10,25,50,75,90,95,100]))
for i in range(0,794,50): print(i, 't',t[i].round(3),'contact',c[i].round(2),'area',areas[i],'iou',ious[i-1] if i else 1)
PY"
```
<details><summary>output</summary>

```
areas percentiles [14511.   17358.65 17987.7  19512.   26324.5  31632.75 33897.7  34024.05
 34362.  ]
0 t [0.492 0.475 2.872] contact [0. 0.] area 29554 iou 1
50 t [0.494 0.475 2.873] contact [0. 0.] area 29496 iou 0.9974604679511055
100 t [0.495 0.481 2.879] contact [0. 0.] area 21490 iou 0.9621465353690487
150 t [0.492 0.478 2.878] contact [0.69 0.95] area 23387 iou 0.9894337693323676
200 t [0.474 0.473 2.851] contact [1. 1.] area 21340 iou 0.9923321488685244
250 t [0.48  0.481 2.885] contact [1. 1.] area 18499 iou 0.9853210022583073
300 t [0.365 0.443 2.944] contact [1. 1.] area 18062 iou 0.8984070049583289
350 t [0.309 0.419 3.049] contact [1. 1.] area 19044 iou 0.9774694899342861
400 t [0.256 0.431 3.176] contact [1. 1.] area 21448 iou 0.9551997121266642
450 t [0.065 0.486 3.354] cont …[+517 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); idxs=list(range(240,501,10)); ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); im=cv2.resize(im,(512,384))
 cv2.putText(im,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA);ims.append(im)
rows=[]
for j in range(0,len(ims),5):
 row=ims[j:j+5]
 while len(row)<5: row.append(np.zeros_like(ims[0]))
 rows.append(np.hstack(row))
out=np.vstack(rows);cv2.imwrite('/work/scratch/vis/contact_240_500.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92]);print(out.shape)
PY"
```
<details><summary>output</summary>

```
(2304, 2560, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'find /code/reconstruction -maxdepth 3 -type f | head -100; echo ---; find /workspace -maxdepth 2 -type d | head -100'
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
/code/reconstructio …[+5419 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 def rec(name,obj):
  if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype, {k:obj.attrs[k] for k in obj.attrs})
  else: print(name,'GROUP',dict(obj.attrs))
 f.visititems(rec)
print('--- masks')
with h5py.File('/work/inputs/masks.h5','r') as f:
 for k in f: print(k,f[k].shape,f[k].dtype,dict(f[k].attrs))
PY"
```
<details><summary>output</summary>

```
aligned GROUP {}
aligned/front_stereo_camera_left (794,) object {}
alignment GROUP {}
alignment/front_stereo_camera_left GROUP {}
alignment/front_stereo_camera_left/scale (794,) float32 {}
alignment/front_stereo_camera_left/shift (794,) float32 {}
alignment/front_stereo_camera_left/valid_count (794,) int32 {}
frame_names GROUP {}
frame_names/front_stereo_camera_left (794,) object {}
raw GROUP {}
raw/front_stereo_camera_left (794,) object {}
--- masks
object (794, 1152, 1536) bool {}
person (794, 1152, 1536) bool {}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr, torch
print('nvdiffrast ok',dr,torch.cuda.is_available(),torch.cuda.get_device_name())
PY"
```
<details><summary>output</summary>

```
nvdiffrast ok <module 'nvdiffrast.torch' from '/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/__init__.py'> True NVIDIA RTX 6000 Ada Generation

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for path in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left','frame_names/front_stereo_camera_left']:
  d=f[path]; print(path,type(d),d.shape,d.dtype)
  print('first',d[0])
  if path.startswith('aligned'):
   a=d[:];print('a range',np.nanmin(a),np.nanmax(a),np.nanpercentile(a,[0,1,50,99,100]))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (794,) object
first [137  80  78 ...  66  96 130]
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py", line 349, in nanmin
    a, mask = _replace_nan(a, +np.inf)
              ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py", line 100, in _replace_nan
    mask = np.not_equal(a, a, dtype=bool)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for path in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left','frame_names/front_stereo_camera_left']:
  d=f[path]; print('\\n',path,d.shape,d.dtype)
  for i in [0,1,793]:
   a=d[i]; print(i,type(a),getattr(a,'shape',None),getattr(a,'dtype',None),repr(a)[:200])
PY"
```
<details><summary>output</summary>

```

 aligned/front_stereo_camera_left (794,) object
0 <class 'numpy.ndarray'> (637984,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
1 <class 'numpy.ndarray'> (637286,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
793 <class 'numpy.ndarray'> (623193,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)

 raw/front_stereo_camera_left (794,) object
0 <class 'numpy.ndarray'> (605064,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
1 <class 'numpy.ndarray'> (605825,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
793 <class 'numpy.ndarray'> (591516,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)

 frame_names/front_stereo_camera_left (794,) object
0 <class 'bytes'> None None b'000000'
1 <class 'bytes'> None  …[+55 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for path in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  a=f[path][0]; print(path,a[:20].tolist(),a.shape)
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left [137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 6, 0] (637984,)
raw/front_stereo_camera_left [137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 6, 0] (605064,)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for path in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  a=f[path][0]; im=cv2.imdecode(a,cv2.IMREAD_UNCHANGED)
  print(path,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[0,1,50,99,100]))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (1152, 1536) uint16 1672 9155 [1672. 1750. 3660. 8658. 9155.]
raw/front_stereo_camera_left (1152, 1536) uint16 1581 8656 [1581. 1655. 3461. 8186. 8656.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=s.to_geometry(); m=g.visual.material
print(type(m),m)
for k in ['baseColorTexture','image','baseColorFactor','metallicFactor','roughnessFactor','normalTexture','emissiveTexture']:
 v=getattr(m,k,None);print(k,type(v),getattr(v,'shape',None),repr(v)[:300])
print('uv range',g.visual.uv.min(0),g.visual.uv.max(0))
print('vertex normals?',g.vertex_normals.shape if g.vertex_normals is not None else None)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.material.PBRMaterial'> <trimesh.visual.material.PBRMaterial object at 0x707131c067d0>
baseColorTexture <class 'PIL.Image.Image'> None <PIL.Image.Image image mode=RGB size=3004x3017 at 0x707131C06950>
image <class 'NoneType'> None None
baseColorFactor <class 'NoneType'> None None
metallicFactor <class 'float'> None 0.0
roughnessFactor <class 'NoneType'> None None
normalTexture <class 'NoneType'> None None
emissiveTexture <class 'NoneType'> None None
uv range [0.001165   0.00116003] [0.99883503 0.99883997]
vertex normals? (296588, 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, cv2, h5py, json
import nvdiffrast.torch as dr
from PIL import Image
W,H=1536,1152
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); m=s.to_geometry()
V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda')
F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda')
tex=torch.tensor(np.array(m.visual.material.baseColorTexture.resize((1024,1024))),dtype=torch.float32,device='cuda')/255
tex=tex.permute(2,0,1)[None]
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],float); Twc=np.array(cam['T_world_cam'],float)
gl=dr.RasterizeCudaContext()
def render(T):
    Tc=np.linalg.inv(Twc)@T
    Tc=torch.tensor(Tc,dtype=torch.float32,device='cuda')
    v=V@Tc[:3,:3].T+Tc[:3,3]
    z=v[:,2].clamp(min=1e-4)
    x=2*K[0,0]/W*v[:,0]+(2*K[0,2]/W-1)*z
    y=(1-2*K[1,2]/H)*z-2*K[1,1]/H*v[:,1]
    pos=torch.stack([x,y,z,z],-1)[None]
    rast,_=dr.rasterize(gl,pos,F,resolution=[H,W])
    zmap=dr.interpolate(z[:,None],rast,F)[0,:,:,0]
    uv=dr.interpolate(UV[None],rast,F)[0]
    mask=rast[0,:,:,3]>0
    # sample texture (UV image origin top-left)
    grid=uv*2-1; grid=grid[None]
    rgb=torch.nn.functional.grid_sample(tex,grid,align_corners=False,mode='bilinear')[0]
    rgb=rgb.permute(1,2,0).clamp(0,1)
    return mask.detach().cpu().numpy(),zmap.detach().cpu().numpy(),(rgb.detach().cpu().numpy()* …[+859 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 34, in <module>
    mask,d,rgb=render(T)
               ^^^^^^^^^
  File "/work/scratch/render_test.py", line 22, in render
    zmap=dr.interpolate(z[:,None],rast,F)[0,:,:,0]
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
TypeError: tuple indices must be integers or slices, not tuple

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/zmap=dr.interpolate(z\\[:,None\\],rast,F)\\[0,:,:,0\\]/zmap=dr.interpolate(z[:,None],rast,F)[0][0,:,:,0]/; s/uv=dr.interpolate(UV\\[None\\],rast,F)\\[0\\]/uv=dr.interpolate(UV[None],rast,F)[0][0]/' /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 flip False objIoU 0.0 visIoU 0.0 prec 0.0
0 flip True objIoU 0.7552873773389407 visIoU 0.27565956310895584 prec 0.8837541006989017
50 flip False objIoU 0.0 visIoU 0.0 prec 0.0
50 flip True objIoU 0.7539578440045145 visIoU 0.26740449839343094 prec 0.8826239108698757
150 flip False objIoU 0.0 visIoU 0.009787076883550646 prec 0.0
150 flip True objIoU 0.6466167817856229 visIoU 0.24729508603163583 prec 0.7194549206575108
250 flip False objIoU 0.0 visIoU 0.028998663123669312 prec 0.0
250 flip True objIoU 0.5051101521689757 visIoU 0.24444355732093134 prec 0.5581929006812477
350 flip False objIoU 0.0 visIoU 0.02566968550575108 prec 0.0
350 flip True objIoU 0.6134327234395001 visIoU 0.21786131560738228 prec 0.6842768352091383
450 flip False objIoU 0.013641724313953284 visIoU 0.06859585201793722 p …[+559 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/overlay_baseline.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import matplotlib;matplotlib.use('Agg')
z=np.load('/work/inputs/reconstruction.npz');cap=cv2.VideoCapture('/work/inputs/video.mp4')
idxs=[0,100,250,300,350,400,450,500,550,600,650,700,750,793]
ims=[]
for i in idxs:
 mask,d,rgb=render(z['T_world_mesh'][i].astype(float)); mask=mask[::-1];rgb=rgb[::-1]
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 ov=im.copy(); ov[mask]=(0.35*im[mask]+0.65*rgb[mask]).astype(np.uint8)
 # contour red
 cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cnts,-1,(255,0,0),2)
 ov=cv2.resize(cv2.cvtColor(ov,cv2.COLOR_RGB2BGR),(640,480))
 cv2.putText(ov,str(i),(12,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2,cv2.LINE_AA);ims.append(ov)
rows=[]
for j in range(0,len(ims),2): rows.append(np.hstack(ims[j:j+2]))
out=np.vstack(rows);cv2.imwrite('/work/scratch/vis/overlay_baseline.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92]);print(out.shape)
PY
python /work/scratch/overlay_baseline.py"
```
<details><summary>output</summary>

```
(3360, 1280, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,cv2,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
im=np.array(m.visual.material.baseColorTexture)
print(im.shape,im.min(),im.max(),im.mean((0,1)))
cv2.imwrite('/work/scratch/vis/texture.jpg',cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
# sample pixels
PY"
```
<details><summary>output</summary>

```
(3017, 3004, 3) 0 255 [62.829121   74.86997924 70.27419291]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/texture_debug.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'][0].astype(float)
# monkey copy rendering with flips
Tc=np.linalg.inv(Twc)@T;Tc=torch.tensor(Tc,dtype=torch.float32,device='cuda');v=V@Tc[:3,:3].T+Tc[:3,3];zc=v[:,2].clamp(min=1e-4)
x=2*K[0,0]/W*v[:,0]+(2*K[0,2]/W-1)*zc;y=(1-2*K[1,2]/H)*zc-2*K[1,1]/H*v[:,1];pos=torch.stack([x,y,zc,zc],-1)[None]
rast,_=dr.rasterize(gl,pos,F,resolution=[H,W]);uv=dr.interpolate(UV[None],rast,F)[0][0]
mask=(rast[0,:,:,3]>0).detach().cpu().numpy()[::-1]
cap=cv2.VideoCapture('/work/inputs/video.mp4');ok,im=cap.read()
ims=[]
for uvf in [False,True]:
 uu=uv.clone();uu[...,1]=1-uu[...,1] if uvf else uu[...,1]
 rgb=torch.nn.functional.grid_sample(tex,uu[None]*2-1,align_corners=False,mode='bilinear')[0].permute(1,2,0).clamp(0,1)
 rgb=(rgb.detach().cpu().numpy()*255).astype(np.uint8)[::-1]
 ov=im.copy();ov[mask]=(im[mask]*.25+rgb[mask]*.75).astype(np.uint8);cv2.putText(ov,'flipv='+str(uvf),(30,60),0,1.5,(0,0,255),3);ims.append(ov)
out=np.hstack([cv2.resize(x,(768,576)) for x in ims]);cv2.imwrite('/work/scratch/vis/texture_flip.jpg',out)
PY
python /work/scratch/texture_debug.py"
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
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
for base in [0,100,250,300,350,400,450,500,550,600,650,700]:
 rv=(Rot.from_matrix(T[base,:3,:3]).inv()*Rot.from_matrix(T[:,:3,:3])).as_rotvec();ang=np.degrees(np.linalg.norm(rv,axis=1))
 print('\\nbase',base)
 for i in [0,100,250,300,350,400,450,500,550,600,650,700,793]:
  print(i,round(ang[i],2),np.round(T[i,:3,3],3))
print('euler degrees per 25')
for i in range(0,794,25): print(i,np.round(Rot.from_matrix(T[i,:3,:3]).as_euler('xyz',degrees=True),2),np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```

base 0
0 0.0 [0.492 0.475 2.872]
100 1.01 [0.495 0.481 2.879]
250 1.03 [0.48  0.481 2.885]
300 11.32 [0.365 0.443 2.944]
350 18.82 [0.309 0.419 3.049]
400 23.99 [0.256 0.431 3.176]
450 43.87 [0.065 0.486 3.354]
500 79.33 [-0.182  0.785  3.591]
550 86.05 [-0.207  0.838  3.612]
600 89.2 [-0.228  0.883  3.598]
650 89.94 [-0.234  0.903  3.659]
700 89.69 [-0.234  0.91   3.681]
793 88.32 [-0.216  0.872  3.575]

base 100
0 1.01 [0.492 0.475 2.872]
100 0.0 [0.495 0.481 2.879]
250 1.99 [0.48  0.481 2.885]
300 11.91 [0.365 0.443 2.944]
350 19.22 [0.309 0.419 3.049]
400 24.39 [0.256 0.431 3.176]
450 44.19 [0.065 0.486 3.354]
500 79.63 [-0.182  0.785  3.591]
550 86.37 [-0.207  0.838  3.612]
600 89.52 [-0.228  0.883  3.598]
650 90.24 [-0.234  0.903  3.659]
700 90.0 [-0.234  0.91   3.681]
793 88.63 [-0 …[+5588 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz'); k=z['human_keypoints_world'].astype(float); c=z['contact_prob_right_left'];T=z['T_world_mesh'].astype(float)
for i in range(0,794,25):
 print(i,'R',np.round(k[i,21:42].mean(0),3),'L',np.round(k[i,42:63].mean(0),3),'cp',np.round(c[i],2),'objt',np.round(T[i,:3,3],3))
print('kp ranges',k.min((0,1)),k.max((0,1)))
PY"
```
<details><summary>output</summary>

```
0 R [-0.385  0.501  2.568] L [-0.685  0.451  2.903] cp [0. 0.] objt [0.492 0.475 2.872]
25 R [-0.475  0.524  2.431] L [-0.66   0.463  2.905] cp [0. 0.] objt [0.489 0.465 2.859]
50 R [-0.345  0.556  2.184] L [-0.375  0.501  2.69 ] cp [0. 0.] objt [0.494 0.475 2.873]
75 R [-0.117  0.563  2.136] L [-0.213  0.517  2.626] cp [0. 0.] objt [0.496 0.473 2.877]
100 R [0.242 0.523 2.163] L [-0.066  0.494  2.56 ] cp [0. 0.] objt [0.495 0.481 2.879]
125 R [0.383 0.512 2.247] L [-0.142  0.512  2.451] cp [0. 0.] objt [0.494 0.48  2.874]
150 R [0.358 0.387 2.338] L [-0.107  0.339  2.823] cp [0.69 0.95] objt [0.492 0.478 2.878]
175 R [0.341 0.384 2.358] L [-0.043  0.357  2.779] cp [1. 1.] objt [0.477 0.476 2.862]
200 R [0.33  0.383 2.364] L [-0.051  0.367  2.77 ] cp [1. 1.] objt [0.474 0.473 2.851]
225 R  …[+2154 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5') as f:
 o=f['object'][:];p=f['person'][:]
for i in range(0,794,25):
 y,x=np.where(o[i]);yp,xp=np.where(p[i]);
 print(i,'obj cent',round(x.mean(),1),round(y.mean(),1),'bbox',x.min(),y.min(),x.max(),y.max(),'area',len(x),'person area',len(xp))
PY"
```
<details><summary>output</summary>

```
0 obj cent 852.5 735.5 bbox 693 605 1017 963 area 29554 person area 57094
25 obj cent 853.3 734.2 bbox 693 605 1018 963 area 28867 person area 55286
50 obj cent 853.6 734.6 bbox 693 605 1018 963 area 29496 person area 59650
75 obj cent 853.7 734.7 bbox 698 605 1018 964 area 29428 person area 73532
100 obj cent 882.4 735.8 bbox 783 605 1018 963 area 21490 person area 80736
125 obj cent 914.3 729.7 bbox 805 605 1018 963 area 14878 person area 90944
150 obj cent 872.9 746.7 bbox 774 605 1018 963 area 23387 person area 74328
175 obj cent 877.5 744.3 bbox 772 605 1017 966 area 22150 person area 74070
200 obj cent 879.7 741.5 bbox 772 605 1016 965 area 21340 person area 73184
225 obj cent 888.3 749.3 bbox 777 605 1015 965 area 17876 person area 77339
250 obj cent 888.1 748.5 bbox 778 605 1016 96 …[+1607 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_depth.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import h5py
z=np.load('/work/inputs/reconstruction.npz'); Tworld=np.array(cam['T_world_cam'],float)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as fd, h5py.File('/work/inputs/masks.h5','r') as fm:
 for i in [0,25,50,75,100,150,200,250,300,350,400,450,500,550,600,650,700,750,793]:
  mask,dep,rgb=render(z['T_world_mesh'][i].astype(float)); mask=mask[::-1];dep=dep[::-1]
  D=cv2.imdecode(fd['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
  obj=fm['object'][i];per=fm['person'][i]
  valid=mask & ~per & obj
  r=(dep[valid]-D[valid])
  # also all render on nonperson rough 90 percentile etc
  print(i,'n',valid.sum(),'res median',np.median(r),'mean',r.mean(),'mae',np.mean(abs(r)),'p10/90',np.percentile(r,[10,90]) if r.size else None,'objD',np.median(D[obj]),'renderD',np.median(dep[mask&obj]))
PY
python /work/scratch/eval_depth.py"
```
<details><summary>output</summary>

```
0 n 24784 res median 0.001757677078247033 mean 0.0006322721200577298 mae 0.09706216522045025 p10/90 [-0.11227534  0.13964007] objD 2.628 renderD 2.584331
25 n 23401 res median -0.12067851066589341 mean -0.12160234320637996 mae 0.1552025100594224 p10/90 [-0.24862644  0.0243741 ] objD 2.706 renderD 2.5527601
50 n 24717 res median -0.052132656097412244 mean -0.05895508706871107 mae 0.11239226590382771 p10/90 [-0.16121785  0.0811426 ] objD 2.671 renderD 2.5813541
75 n 24782 res median -0.0036510076522826562 mean -0.010816709897023855 mae 0.10416772410363746 p10/90 [-0.10661099  0.12811977] objD 2.624 renderD 2.5942285
100 n 18650 res median -0.04566380214691157 mean -0.03393147005280604 mae 0.10899432544084402 p10/90 [-0.14497164  0.13176252] objD 2.844 renderD 2.78088
150 n 20171 res median 0 …[+2114 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,cv2,h5py,json
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);Twc=np.array(cam['T_world_cam'],float); Tcw=np.linalg.inv(Twc)
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);print(pl)
W,H=1536,1152
y,x=np.mgrid[0:H,0:W]; rays=np.stack([(x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],np.ones_like(x)],-1)
rw=rays@Twc[:3,:3].T
# world p = C + z rw; plane n.p+d=0 => z=-(n.C+d)/(n.rw)
C=Twc[:3,3]
zf=-(pl[:3]@C+pl[3])/(rw@pl[:3]+1e-9)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 D=cv2.imdecode(f['aligned/front_stereo_camera_left'][0],cv2.IMREAD_UNCHANGED)/1000
 for y0,y1,x0,x1 in [(900,1100,100,1400),(1000,1140,100,1400),(700,850,50,500),(600,700,900,1100)]:
  sl=(slice(y0,y1),slice(x0,x1)); z=zf[sl];d=D[sl];ok=(z>0)&(z<10)
  r=d[ok]-z[ok];print((y0,y1,x0,x1),np.median(r),r.mean(),np.percentile(r,[10,50,90]),'D',np.median(d[ok]),'z',np.median(z[ok]))
PY"
```
<details><summary>output</summary>

```
[-0.01889224 -0.99823916 -0.05622866  1.39009273]
(900, 1100, 100, 1400) 0.0015948196183881436 -0.0023069562374556005 [-0.01036861  0.00159482  0.00872837] D 2.229 z 2.228063417579927
(1000, 1140, 100, 1400) 0.004435447220963562 0.005629656981695478 [0.00077954 0.00443545 0.01232395] D 1.964 z 1.9579593379298195
(700, 850, 50, 500) -0.10200600227183765 -0.37956219007645947 [-1.2657871  -0.102006   -0.03925982] D 3.772 z 4.135278618749532
(600, 700, 900, 1100) -0.9889122786995763 -1.9225635888247807 [-5.21704451 -0.98891228 -0.15283064] D 5.708 z 6.949499436229612

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();c=m.bounds.mean(0);print('mesh center',c)
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
for i in range(0,794,25):
 p=T[i,:3,:3]@c+T[i,:3,3];print(i,np.round(p,3))
PY"
```
<details><summary>output</summary>

```
mesh center [ 0.27196912 -0.0040119   0.00456923]
0 [0.496 0.747 2.886]
25 [0.496 0.737 2.872]
50 [0.498 0.746 2.886]
75 [0.497 0.745 2.893]
100 [0.499 0.752 2.894]
125 [0.499 0.751 2.888]
150 [0.497 0.75  2.893]
175 [0.482 0.748 2.877]
200 [0.477 0.745 2.867]
225 [0.483 0.752 2.903]
250 [0.485 0.753 2.9  ]
275 [0.476 0.764 2.937]
300 [0.41  0.711 2.926]
325 [0.378 0.679 2.977]
350 [0.372 0.678 2.996]
375 [0.356 0.688 3.065]
400 [0.335 0.682 3.107]
425 [0.288 0.68  3.161]
450 [0.204 0.687 3.233]
475 [0.099 0.751 3.413]
500 [0.015 0.841 3.412]
525 [-0.009  0.864  3.431]
550 [-0.009  0.863  3.428]
575 [-0.012  0.886  3.502]
600 [-0.024  0.893  3.418]
625 [-0.02   0.884  3.384]
650 [-0.03   0.909  3.479]
675 [-0.029  0.908  3.477]
700 [-0.03   0.918  3.501]
725 [-0.023  0.904  3.46 ]
750 [-0. …[+46 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/scan_depth.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import h5py
z=np.load('/work/inputs/reconstruction.npz'); Rcw=Twc[:3,:3].T
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as fd, h5py.File('/work/inputs/masks.h5','r') as fm:
 for i in [0,50,650,675,700]:
  T0=z['T_world_mesh'][i].astype(float);D=cv2.imdecode(fd['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
  obj=fm['object'][i];per=fm['person'][i]; vis=obj|per
  print('\\nFRAME',i)
  for dz in np.arange(-.35,.351,.05):
   T=T0.copy();T[:3,3]+=Rcw@np.array([0,0,dz])
   mask,dep,rgb=render(T);mask=mask[::-1];dep=dep[::-1]
   valid=mask&obj&~per
   r=dep[valid]-D[valid]
   fp=(mask&~vis).sum(); fn=(obj&~mask).sum(); inter=(mask&obj).sum(); union=(mask|obj).sum()
   print(f'dz {dz:+.2f} med {np.median(r):+.3f} mae {np.mean(abs(r)):.3f} objIoU {inter/union:.3f} fp {fp} fn {fn} rendered {mask.sum()}')
PY
python /work/scratch/scan_depth.py"
```
<details><summary>output</summary>

```

FRAME 0
dz -0.35 med -0.253 mae 0.264 objIoU 0.348 fp 21180 fn 11893 rendered 38841
dz -0.30 med -0.202 mae 0.227 objIoU 0.388 fp 18263 fn 10992 rendered 36825
dz -0.25 med -0.154 mae 0.194 objIoU 0.430 fp 15718 fn 10070 rendered 35202
dz -0.20 med -0.121 mae 0.163 objIoU 0.485 fp 12966 fn 8928 rendered 33592
dz -0.15 med -0.084 mae 0.131 objIoU 0.557 fp 10031 fn 7498 rendered 32087
dz -0.10 med -0.067 mae 0.106 objIoU 0.662 fp 6639 fn 5597 rendered 30596
dz -0.05 med -0.032 mae 0.088 objIoU 0.755 fp 3963 fn 4233 rendered 29284
dz -0.00 med +0.002 mae 0.097 objIoU 0.755 fp 3260 fn 4770 rendered 28044
dz +0.05 med +0.045 mae 0.122 objIoU 0.684 fp 3948 fn 6646 rendered 26856
dz +0.10 med +0.100 mae 0.155 objIoU 0.606 fp 4919 fn 8666 rendered 25807
dz +0.15 med +0.152 mae 0.183 objIoU 0.554  …[+4842 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/scan_pose.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import h5py
z=np.load('/work/inputs/reconstruction.npz'); Rcw=Twc[:3,:3].T
with h5py.File('/work/inputs/masks.h5','r') as fm:
 for i in [0,650]:
  T0=z['T_world_mesh'][i].astype(float);obj=fm['object'][i];per=fm['person'][i];vis=obj|per
  print('\\nFRAME',i)
  best=[]
  for dx in np.arange(-.12,.121,.04):
   for dy in np.arange(-.12,.121,.04):
    for dz in [-.08,-.04,0,.04]:
     T=T0.copy();T[:3,3]+=Rcw@np.array([dx,dy,dz])
     mask,dep,rgb=render(T);mask=mask[::-1]
     inter=(mask&obj).sum();union=(mask|obj).sum();fp=(mask&~vis).sum();fn=(obj&~mask).sum()
     score=inter/union - .3*fp/obj.sum() - .1*fn/obj.sum()
     best.append((score,inter/union,fp,fn,dx,dy,dz,mask.sum()))
  best.sort(reverse=True)
  for b in best[:15]:print('score %.3f iou %.3f fp %d fn %d dx %.2f dy %.2f dz %.2f area %d'%b)
PY
python /work/scratch/scan_pose.py"
```
<details><summary>output</summary>

```

FRAME 0
score 0.716 iou 0.767 fp 3587 fn 4138 dx 0.00 dy 0.00 dz -0.04 area 29003
score 0.711 iou 0.758 fp 3073 fn 4809 dx 0.00 dy 0.04 dz 0.04 area 27818
score 0.706 iou 0.755 fp 3260 fn 4770 dx 0.00 dy 0.00 dz 0.00 area 28044
score 0.636 iou 0.696 fp 3826 fn 6308 dx 0.00 dy 0.00 dz 0.04 area 27072
score 0.631 iou 0.703 fp 5420 fn 4962 dx 0.00 dy 0.00 dz -0.08 area 30012
score 0.573 iou 0.653 fp 5715 fn 6518 dx 0.00 dy 0.04 dz 0.00 area 28751
score 0.572 iou 0.650 fp 5314 fn 6901 dx -0.04 dy 0.00 dz 0.00 area 27967
score 0.572 iou 0.653 fp 5832 fn 6456 dx -0.04 dy 0.00 dz -0.04 area 28930
score 0.512 iou 0.611 fp 7347 fn 7016 dx -0.04 dy 0.00 dz -0.08 area 29885
score 0.510 iou 0.607 fp 7122 fn 7299 dx 0.00 dy -0.04 dz -0.08 area 29377
score 0.493 iou 0.584 fp 6116 fn 8707 dx -0.04 dy 0. …[+1445 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/scan_rot.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import h5py
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz'); mcenter=m.bounds.mean(0); Rcw=Twc[:3,:3].T

def perturb(T0,rv,trans=np.zeros(3)):
 T=T0.copy();c=T0[:3,:3]@mcenter+T0[:3,3]
 R=Rot.from_rotvec(rv).as_matrix()@T0[:3,:3]
 T[:3,:3]=R;T[:3,3]=c-R@mcenter
 T[:3,3]+=Rcw@trans
 return T
with h5py.File('/work/inputs/masks.h5','r') as fm:
 for i in [0,650]:
  T0=z['T_world_mesh'][i].astype(float);obj=fm['object'][i];per=fm['person'][i];vis=obj|per
  res=[]
  for ax in np.deg2rad([-4,-2,0,2,4]):
   for ay in np.deg2rad([-4,-2,0,2,4]):
    for az in np.deg2rad([-4,-2,0,2,4]):
     T=perturb(T0,np.array([ax,ay,az]),np.array([0,-.04 if i==650 else 0,-.06 if i==650 else -.03]))
     mask,dep,rgb=render(T);mask=mask[::-1];inter=(mask&obj).sum();union=(mask|obj).sum();fp=(mask&~vis).sum();fn=(obj&~mask).sum();score=inter/union-.3*fp/obj.sum()-.1*fn/obj.sum();res.append((score,inter/union,fp,fn,*np.rad2deg([ax,ay,az]),mask.sum()))
  res.sort(reverse=True);print('\\nFRAME',i)
  for r in res[:20]:print('score %.3f iou %.3f fp %d fn %d rot %.1f %.1f %.1f area %d'%r)
PY
python /work/scratch/scan_rot.py"
```
<details><summary>output</summary>

```

FRAME 0
score 0.735 iou 0.781 fp 3226 fn 3938 rot 0.0 -2.0 0.0 area 28842
score 0.734 iou 0.780 fp 3175 fn 4040 rot 0.0 -2.0 2.0 area 28689
score 0.732 iou 0.783 fp 3963 fn 3304 rot 2.0 -2.0 2.0 area 30213
score 0.726 iou 0.774 fp 3344 fn 4101 rot 0.0 0.0 0.0 area 28797
score 0.715 iou 0.770 fp 4261 fn 3512 rot 2.0 0.0 2.0 area 30303
score 0.710 iou 0.761 fp 3479 fn 4426 rot 0.0 -4.0 2.0 area 28607
score 0.708 iou 0.760 fp 3638 fn 4338 rot 0.0 -4.0 0.0 area 28854
score 0.708 iou 0.765 fp 4475 fn 3513 rot 2.0 -2.0 0.0 area 30516
score 0.707 iou 0.758 fp 3567 fn 4455 rot 0.0 0.0 2.0 area 28666
score 0.702 iou 0.760 fp 4519 fn 3651 rot 2.0 0.0 0.0 area 30422
score 0.700 iou 0.758 fp 4440 fn 3791 rot 2.0 -4.0 2.0 area 30203
score 0.684 iou 0.735 fp 3183 fn 5496 rot -2.0 -2.0 0.0 area 27241
sc …[+1850 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();uv=m.visual.uv;im=np.array(m.visual.material.baseColorTexture)
# bilinear nearest vertex colors
x=np.clip((uv[:,0]*im.shape[1]).astype(int),0,im.shape[1]-1);y=np.clip((uv[:,1]*im.shape[0]).astype(int),0,im.shape[0]-1)
col=im[y,x]
print('vertex sampled brightness',np.percentile(col.mean(1),[0,5,10,25,50,75,90,95,100]))
print('fraction dark<20',(col.mean(1)<20).mean(),'white>180',(col.mean(1)>180).mean())
# flip y
x=np.clip((uv[:,0]*im.shape[1]).astype(int),0,im.shape[1]-1);y=np.clip(((1-uv[:,1])*im.shape[0]).astype(int),0,im.shape[0]-1);col2=im[y,x]
print('flip brightness',np.percentile(col2.mean(1),[0,5,10,25,50,75,90,95,100]),'dark', (col2.mean(1)<20).mean())
# face centroid uv
fc=uv[m.faces].mean(1);print('centroid u/v',fc.min(0),fc.max(0))
print('face centroid black fraction', (im[np.clip((fc[:,1]*im.shape[0]).astype(int),0,im.shape[0]-1),np.clip((fc[:,0]*im.shape[1]).astype(int),0,im.shape[1]-1)].mean(1)<20).mean())
PY"
```
<details><summary>output</summary>

```
vertex sampled brightness [  0.           0.           0.           0.         106.66666667
 140.         163.         174.66666667 247.66666667]
fraction dark<20 0.35022657693500747 white>180 0.042186467422822234
flip brightness [  0.          35.          49.66666667  84.66666667 113.66666667
 139.33333333 166.66666667 182.         237.        ] dark 0.011615439599714082
centroid u/v [0.00148767 0.00136701] [0.99852234 0.99860499]
face centroid black fraction 0.35035286029159

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/grid=uv\\*2-1; grid=grid\\[None\\]/uv=uv.clone(); uv[...,1]=1-uv[...,1]; grid=uv*2-1; grid=grid[None]/' /work/scratch/render_test.py
python /work/scratch/overlay_baseline.py"
```
<details><summary>output</summary>

```
(3360, 1280, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/flip_eval.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import h5py
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz'); mcenter=m.bounds.mean(0)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as fm:
 for i in [0,50,650,700]:
  T0=z['T_world_mesh'][i].astype(float);obj=fm['object'][i];per=fm['person'][i];vis=obj|per
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(float)
  print('\\nFRAME',i)
  for name,axis,ang,pre in [('base',None,0,False),('mx',0,np.pi,False),('my',1,np.pi,False),('mz',2,np.pi,False),('wx',0,np.pi,True),('wy',1,np.pi,True),('wz',2,np.pi,True)]:
   T=T0.copy();c=T0[:3,:3]@mcenter+T0[:3,3]
   if axis is not None:
    P=Rot.from_rotvec(np.eye(3)[axis]*ang).as_matrix()
    Rnew=(P@T0[:3,:3]) if pre else (T0[:3,:3]@P)
    T[:3,:3]=Rnew;T[:3,3]=c-Rnew@mcenter
   mask,dep,rgb=render(T);mask=mask[::-1];rgb=rgb[::-1]
   inter=(mask&obj).sum();union=(mask|obj).sum();iou=inter/union;fp=(mask&~vis).sum();fn=(obj&~mask).sum()
   m=(mask&obj&~per);m=cv2.erode(m.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
   pred=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
   a=gray[m];b=pred[m]
   corr=np.corrcoef(a,b)[0,1] if a.std()>0 and b.std()>0 else 0
   # gradients on masked overlap
   gx1=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy1=cv2.Sobel(gray …[+375 chars]
```
<details><summary>output</summary>

```

FRAME 0
base  iou 0.755 fp 3260 fn 4770 corr -0.207 gcorr 0.023 mae 79.8
mx    iou 0.716 fp 4918 fn 4856 corr -0.312 gcorr 0.000 mae 84.1
my    iou 0.362 fp 23890 fn 10194 corr 0.014 gcorr -0.015 mae 49.6
mz    iou 0.359 fp 22120 fn 11020 corr -0.207 gcorr -0.004 mae 52.9
wx    iou 0.138 fp 32109 fn 21048 corr 0.116 gcorr 0.038 mae 45.8
wy    iou 0.594 fp 4464 fn 9335 corr -0.388 gcorr -0.016 mae 76.5
wz    iou 0.174 fp 29115 fn 19334 corr -0.135 gcorr -0.025 mae 52.1

FRAME 50
base  iou 0.754 fp 3287 fn 4779 corr -0.205 gcorr 0.023 mae 79.7
mx    iou 0.711 fp 5031 fn 4932 corr -0.340 gcorr 0.007 mae 83.6
my    iou 0.357 fp 24025 fn 10414 corr 0.041 gcorr -0.011 mae 49.4
mz    iou 0.354 fp 22264 fn 11178 corr -0.188 gcorr -0.009 mae 52.6
wx    iou 0.139 fp 31815 fn 20967 corr 0.167 gcorr  …[+1102 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
# crop existing overlay? regenerate simple side sides at 650
import cv2,numpy as np
im=cv2.imread('/work/scratch/vis/overlay_baseline.jpg')
# not easy; use crop from overlay program rerun maybe files don't crop saved
PY
cat > /work/scratch/crop_render.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2
z=np.load('/work/inputs/reconstruction.npz');cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,650]:
 mask,d,rgb=render(z['T_world_mesh'][i].astype(float));mask=mask[::-1];rgb=rgb[::-1]
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=im[:,:,::-1]
 ys,xs=np.where(mask);x0,x1=max(0,xs.min()-30),min(W,xs.max()+30);y0,y1=max(0,ys.min()-30),min(H,ys.max()+30)
 crop=im[y0:y1,x0:x1].copy();crop[mask[y0:y1,x0:x1]]=(crop[mask[y0:y1,x0:x1]]*.2+rgb[y0:y1,x0:x1]*.8).astype(np.uint8)
 cv2.imwrite(f'/work/scratch/vis/crop_{i}.jpg',cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,95])
 print(i,crop.shape)
PY
python /work/scratch/crop_render.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/crop_render.py", line 8, in <module>
    crop=im[y0:y1,x0:x1].copy();crop[mask[y0:y1,x0:x1]]=(crop[mask[y0:y1,x0:x1]]*.2+rgb[y0:y1,x0:x1]*.8).astype(np.uint8)
                                                         ~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (28044,3) (416,393,3) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/crop_render.py'
s=open(p).read();s=s.replace(\"crop=im[y0:y1,x0:x1].copy();crop[mask[y0:y1,x0:x1]]=(crop[mask[y0:y1,x0:x1]]*.2+rgb[y0:y1,x0:x1]*.8).astype(np.uint8)\",\"crop=im[y0:y1,x0:x1].copy(); mm=mask[y0:y1,x0:x1]; rr=rgb[y0:y1,x0:x1]; crop[mm]=(crop[mm]*.2+rr[mm]*.8).astype(np.uint8)\")
open(p,'w').write(s)
PY
python /work/scratch/crop_render.py"
```
<details><summary>output</summary>

```
0 (416, 393, 3)
650 (298, 374, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/uvmodes.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'][0].astype(float)
Tc=np.linalg.inv(Twc)@T;Tc=torch.tensor(Tc,dtype=torch.float32,device='cuda');v=V@Tc[:3,:3].T+Tc[:3,3];zc=v[:,2];x=2*K[0,0]/W*v[:,0]+(2*K[0,2]/W-1)*zc;y=(1-2*K[1,2]/H)*zc-2*K[1,1]/H*v[:,1];pos=torch.stack([x,y,zc,zc],-1)[None]
rast,_=dr.rasterize(gl,pos,F,resolution=[H,W]);uv=dr.interpolate(UV[None],rast,F)[0][0];mask=(rast[0,:,:,3]>0)
for fu,fv in [(0,0),(0,1),(1,0),(1,1)]:
 u=uv.clone();u[...,0]=1-u[...,0] if fu else u[...,0];u[...,1]=1-u[...,1] if fv else u[...,1]
 rgb=torch.nn.functional.grid_sample(tex,u[None]*2-1,align_corners=False,mode='bilinear')[0].permute(1,2,0)
 gray=rgb.mean(2);print(fu,fv,'inside dark',((gray<.08)&mask).sum().item()/mask.sum().item(),'verydark',((gray<.02)&mask).sum().item()/mask.sum().item(),'mean',gray[mask].mean().item(),'p10',torch.quantile(gray[mask],.1).item())
PY
python /work/scratch/uvmodes.py"
```
<details><summary>output</summary>

```
0 0 inside dark 0.37031093995150477 verydark 0.3597560975609756 mean 0.32593590021133423 p10 0.0
0 1 inside dark 0.00652545999144202 verydark 7.13165026387106e-05 mean 0.5331203937530518 p10 0.34109947085380554
1 0 inside dark 0.37894023677078875 verydark 0.3721295107687919 mean 0.324314147233963 p10 0.0
1 1 inside dark 0.3675652545999144 verydark 0.3532306375695336 mean 0.3288837671279907 p10 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/crop_render.py';s=open(p).read();s=s.replace(\"crop[mask[y0:y1,x0:x1]]\", \"crop[mask[y0:y1,x0:x1]]\")
# create render only version
s=s.replace(\"crop=im[y0:y1,x0:x1].copy(); mm=mask[y0:y1,x0:x1]; rr=rgb[y0:y1,x0:x1]; crop[mm]=(crop[mm]*.2+rr[mm]*.8).astype(np.uint8)\",\"crop=im[y0:y1,x0:x1].copy(); mm=mask[y0:y1,x0:x1]; rr=rgb[y0:y1,x0:x1]; crop[mm]=rr[mm]\")
s=s.replace(\"crop_{i}.jpg\",\"renderonly_{i}.jpg\")
open('/work/scratch/crop_render_only.py','w').write(s)
PY
python /work/scratch/crop_render_only.py"
```
<details><summary>output</summary>

```
0 (416, 393, 3)
650 (298, 374, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'uv\\|grid' /work/scratch/render_test.py; python - <<'PY'
# compute dark fraction with render_test exact
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');mask,d,rgb=render(z['T_world_mesh'][0].astype(float));g=rgb.mean(2);print('frac dark',((g<20)&mask).sum()/mask.sum(),'frac <50',((g<50)&mask).sum()/mask.sum(),'mean',g[mask].mean())
PY"
```
<details><summary>output</summary>

```
frac dark 0.0064898017401226644 frac <50 0.03558693481671659 mean 135.47090286692338

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');idxs=list(range(480,601,5));ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.resize(im,(512,384));cv2.putText(im,str(i),(10,35),0,1,(0,0,255),2);ims.append(im)
rows=[]
for j in range(0,len(ims),5):
 row=ims[j:j+5]
 while len(row)<5:row.append(np.zeros_like(ims[0]))
 rows.append(np.hstack(row))
out=np.vstack(rows);cv2.imwrite('/work/scratch/vis/contact_480_600.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92]);print(out.shape)
PY"
```
<details><summary>output</summary>

```
(1920, 2560, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');idxs=list(range(250,331,5));ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.resize(im,(640,480));cv2.putText(im,str(i),(10,40),0,1.2,(0,0,255),2);cv2.putText(im,'same desk left upright',(10,470),0,.6,(0,0,255),1);ims.append(im)
rows=[]
for j in range(0,len(ims),4):
 row=ims[j:j+4]
 while len(row)<4:row.append(np.zeros_like(ims[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/vis/contact_250_330.jpg',np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,93])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_candidates.py <<'PY'
import numpy as np, torch, trimesh, cv2, h5py, json, time
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
S=0.5; W=int(1536*S);H=int(1152*S)
sm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(sm.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(sm.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float).copy();K[:2]*=S;Twc=np.array(cam['T_world_cam'],float);gl=dr.RasterizeCudaContext()
def render_mask(T):
 Tc=np.linalg.inv(Twc)@T;Tc=torch.tensor(Tc,dtype=torch.float32,device='cuda');v=V@Tc[:3,:3].T+Tc[:3,3];z=v[:,2].clamp(min=1e-4)
 x=2*K[0,0]/W*v[:,0]+(2*K[0,2]/W-1)*z;y=(1-2*K[1,2]/H)*z-2*K[1,1]/H*v[:,1];pos=torch.stack([x,y,z,z],-1)[None]
 rast,_=dr.rasterize(gl,pos,F,resolution=[H,W]);return (rast[0,:,:,3]>0).cpu().numpy()[::-1]
z=np.load('/work/inputs/reconstruction.npz');Tb=z['T_world_mesh'].astype(float);mc=sm.bounds.mean(0)
with h5py.File('/work/inputs/masks.h5','r') as f:
 obj0=f['object'][:];per0=f['person'][:]
# downsample masks nearest
obj=np.asarray([cv2.resize(x.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool) for x in obj0])
per=np.asarray([cv2.resize(x.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool) for x in per0])

def pose_candidate(p):return Tb[p]
def avg_candidate(ps):
 # average centers, quaternion m …[+1511 chars]
```
<details><summary>output</summary>

```

PHASE init frames [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
single 175 score 0.5106 iou/fp/fn [0.6181 0.1055 0.1344]
single 200 score 0.5025 iou/fp/fn [0.607  0.0999 0.1382]
single 225 score 0.4996 iou/fp/fn [0.6079 0.0996 0.1545]
single 250 score 0.4991 iou/fp/fn [0.6087 0.1014 0.1544]
single 275 score 0.4707 iou/fp/fn [0.5782 0.085  0.1919]
single 100 score 0.4665 iou/fp/fn [0.6076 0.1467 0.1536]
single 150 score 0.4606 iou/fp/fn [0.603  0.1466 0.1591]
single 125 score 0.4595 iou/fp/fn [0.6029 0.1486 0.1573]
single 75 score 0.4392 iou/fp/fn [0.5904 0.1564 0.167 ]
single 0 score 0.436 iou/fp/fn [0.5857 0.1515 0.1743]
single 50 score 0.4308 iou/fp/fn [0.5827 0.153  0.1792]
single 10 score 0.43 iou/fp/fn [0.5792 0.1477 0.1832]
avg 200 [150, 175, 200, 225, 250] s …[+1999 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/mask_errors.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2,h5py
z=np.load('/work/inputs/reconstruction.npz');cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,50,175,650,700]:
  m,d,rgb=render(z['T_world_mesh'][i].astype(float));m=m[::-1];o=f['object'][i];p=f['person'][i]
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();out=im.copy()
  # red render only, green mask only, yellow overlap, cyan person
  out[m&~o]=(.2*out[m&~o]+.8*np.array([0,0,255])).astype(np.uint8)
  out[o&~m]=(.2*out[o&~m]+.8*np.array([0,255,0])).astype(np.uint8)
  out[m&o]=(.2*out[m&o]+.8*np.array([0,255,255])).astype(np.uint8)
  # crop around union
  ys,xs=np.where(m|o);x0,x1=xs.min()-40,xs.max()+40;y0,y1=ys.min()-40,ys.max()+40
  cv2.imwrite(f'/work/scratch/vis/err_{i}.jpg',out[max(0,y0):min(H,y1),max(0,x0):min(W,x1)],[cv2.IMWRITE_JPEG_QUALITY,95])
PY
python /work/scratch/mask_errors.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fine_trans.py <<'PY'
import numpy as np, torch, trimesh, cv2, h5py, json, time
import nvdiffrast.torch as dr
S=.5;W=int(1536*S);H=int(1152*S)
sm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(sm.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(sm.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float).copy();K[:2]*=S;Twc=np.array(cam['T_world_cam'],float);Rcw=Twc[:3,:3].T;gl=dr.RasterizeCudaContext()
def render_mask(T):
 Tc=np.linalg.inv(Twc)@T;Tc=torch.tensor(Tc,dtype=torch.float32,device='cuda');v=V@Tc[:3,:3].T+Tc[:3,3];z=v[:,2].clamp(min=1e-4);x=2*K[0,0]/W*v[:,0]+(2*K[0,2]/W-1)*z;y=(1-2*K[1,2]/H)*z-2*K[1,1]/H*v[:,1];pos=torch.stack([x,y,z,z],-1)[None];rast,_=dr.rasterize(gl,pos,F,resolution=[H,W]);return (rast[0,:,:,3]>0).cpu().numpy()[::-1]
z=np.load('/work/inputs/reconstruction.npz');Tb=z['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as f:O=f['object'][:];P=f['person'][:]
for i in [0,650]:
 o=cv2.resize(O[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(P[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);vis=o|p
 # contours and dt
 oc=o.astype(np.uint8);oc=cv2.morphologyEx(oc,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));dt_o=cv2.distanceTransform(1-oc,cv2.DIST_L2,5)
 res=[]
 for dx in np.arange(-.04,.0401,.01):
  for dy in np.ara …[+707 chars]
```
<details><summary>output</summary>

```

frame 0
score 0.6184 cham 0.972 iou 0.8020 fp 0.080 fn 0.138 d -0.01 0.02 0.03
score 0.6101 cham 1.012 iou 0.8023 fp 0.093 fn 0.126 d -0.01 0.01 -0.01
score 0.6093 cham 1.014 iou 0.8004 fp 0.085 fn 0.136 d -0.01 0.01 0.01
score 0.6086 cham 1.046 iou 0.8044 fp 0.087 fn 0.129 d -0.01 0.01 0.00
score 0.6074 cham 1.035 iou 0.8014 fp 0.084 fn 0.135 d 0.00 0.02 0.02
score 0.6041 cham 1.036 iou 0.7984 fp 0.083 fn 0.139 d -0.01 0.02 0.02
score 0.6039 cham 1.024 iou 0.7978 fp 0.090 fn 0.134 d -0.01 0.02 0.01
score 0.6014 cham 1.043 iou 0.7962 fp 0.080 fn 0.145 d -0.01 0.02 0.04
score 0.6006 cham 1.061 iou 0.7983 fp 0.082 fn 0.141 d 0.00 0.03 0.04
score 0.5973 cham 1.055 iou 0.7967 fp 0.093 fn 0.132 d 0.00 0.02 0.01
score 0.5939 cham 1.047 iou 0.7909 fp 0.084 fn 0.147 d 0.00 0.02 0.04
score 0.5893  …[+2077 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
for a,b in [(0,299),(540,793)]:
 vals=[]
 for i in range(a,b+1):
  ov=(o[i]&p[i]).sum();vals.append((ov,i))
 vals.sort()
 print(a,b,'best',vals[:20])
 print('worst',vals[-10:])
for i in range(0,794,10):
 ov=(o[i]&p[i]).sum();
 if i%50==0: print(i,ov)
PY"
```
<details><summary>output</summary>

```
0 299 best [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19)]
worst [(100, 276), (115, 275), (117, 291), (137, 279), (138, 129), (142, 280), (143, 160), (179, 130), (184, 131), (186, 132)]
540 793 best [(0, 676), (0, 677), (0, 678), (0, 679), (0, 681), (0, 683), (0, 684), (0, 685), (0, 687), (0, 688), (0, 691), (0, 692), (0, 693), (0, 694), (0, 695), (0, 696), (0, 697), (0, 698), (0, 699), (0, 700)]
worst [(238, 554), (244, 575), (260, 547), (267, 543), (269, 546), (271, 545), (273, 548), (274, 541), (294, 540), (302, 544)]
0 0
50 0
100 10
150 50
200 18
250 53
300 11
350 8
400 0
450 20
500 0
550 72
600 50
650 25
700 0
750 2

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:]
for i in range(600,761,5): print(i,o[i].sum())
PY"
```
<details><summary>output</summary>

```
600 31646
605 31553
610 31647
615 31712
620 31859
625 32134
630 32727
635 33222
640 33674
645 33998
650 34019
655 34010
660 34041
665 34018
670 33993
675 34200
680 34031
685 33249
690 33919
695 33972
700 34043
705 33910
710 33976
715 33354
720 32140
725 30249
730 28219
735 27763
740 26788
745 25993
750 24047
755 22507
760 21435

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
import numpy as np, torch, trimesh, cv2, h5py, json, time
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
S=.5;W=int(1536*S);H=int(1152*S)
sm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(sm.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(sm.faces,dtype=torch.int32,device='cuda');MC=sm.bounds.mean(0)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float).copy();K[:2]*=S;Twc=np.array(cam['T_world_cam'],float);Rcw=Twc[:3,:3].T;gl=dr.RasterizeCudaContext()
z=np.load('/work/inputs/reconstruction.npz');Tb=z['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as f:Oall=f['object'][:]
O={}
def render_mask(T):
 Tc=np.linalg.inv(Twc)@T;Tc=torch.tensor(Tc,dtype=torch.float32,device='cuda');v=V@Tc[:3,:3].T+Tc[:3,3];zz=v[:,2].clamp(min=1e-4);x=2*K[0,0]/W*v[:,0]+(2*K[0,2]/W-1)*zz;y=(1-2*K[1,2]/H)*zz-2*K[1,1]/H*v[:,1];pos=torch.stack([x,y,zz,zz],-1)[None];rast,_=dr.rasterize(gl,pos,F,resolution=[H,W]);return (rast[0,:,:,3]>0).cpu().numpy()[::-1]
def target(i):
 if i not in O:
  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);O[i]=o
 return O[i]
DT={}
def getdt(i):
 if i not in DT:
  o=target(i).astype(np.uint8);bc=cv2.morphologyEx(o,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8));DT[i]=cv2.distanceTransform(1-bc,cv2.DIST_L2,5)
 return DT[i]
def apply(T0, …[+1669 chars]
```
<details><summary>output</summary>

```

 init start 0.5847924929175234
lev 0 best 0.60841 p [ 0.    0.   -0.04  0.    0.    0.  ]
lev 0 best 0.62551 p [ 0.    0.   -0.04  0.   -3.    0.  ]
lev 1 best 0.64452 p [-0.02  0.   -0.04  0.   -3.    0.  ]
lev 2 best 0.69278 p [-0.01  0.   -0.04  0.   -3.    0.  ]
lev 2 best 0.69636 p [-0.01  0.   -0.03  0.   -3.    0.  ]
lev 2 best 0.70351 p [-0.01  0.   -0.03  0.75 -3.    0.  ]
lev 2 best 0.71608 p [-0.01  0.   -0.03  0.75 -2.25  0.  ]
lev 2 best 0.72880 p [-0.01  0.   -0.03  0.75 -2.25  0.75]
lev 3 best 0.73287 p [-0.01   0.    -0.03   0.375 -2.25   0.75 ]
FINAL init 0.7328735122534951 p [-0.01   0.    -0.03   0.375 -2.25   0.75 ] T= array([[-0.01987263, -0.72410347,  0.68940494,  0.48518968],
       [ 0.99821037, -0.0532696 , -0.02717643,  0.47578762],
       [ 0.05640284,  0.687631 …[+641 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_all.py <<'PY'
import numpy as np, torch, trimesh, cv2, h5py, json, time
import nvdiffrast.torch as dr
S=.5;W=int(1536*S);H=int(1152*S)
sm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(sm.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(sm.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float).copy();K[:2]*=S;Twc=np.array(cam['T_world_cam'],float);Tcw=np.linalg.inv(Twc);gl=dr.RasterizeCudaContext()
z=np.load('/work/inputs/reconstruction.npz');Ts=z['T_world_mesh'].astype(np.float32)
with h5py.File('/work/inputs/masks.h5','r') as f:Oall=f['object'][:];Pall=f['person'][:]
t0=time.time();ious=[];fps=[];fns=[];areas=[];r_med=[];r_mad=[]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as fd:
 for i in range(794):
  T=torch.tensor(Tcw@Ts[i],dtype=torch.float32,device='cuda');v=V@T[:3,:3].T+T[:3,3];zz=v[:,2].clamp(min=1e-4);x=2*K[0,0]/W*v[:,0]+(2*K[0,2]/W-1)*zz;y=(1-2*K[1,2]/H)*zz-2*K[1,1]/H*v[:,1];pos=torch.stack([x,y,zz,zz],-1)[None];rast,_=dr.rasterize(gl,pos,F,resolution=[H,W]);m=(rast[0,:,:,3]>0).cpu().numpy()[::-1];dep=dr.interpolate(zz[:,None],rast,F)[0][0,:,:,0].cpu().numpy()[::-1]
  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  inter=(m&o).sum();uni=(m|o).sum();ious.appe …[+925 chars]
```
<details><summary>output</summary>

```
0 iou 0.7562104237700925 t 0.2
50 iou 0.7530443253774963 t 1.5
100 iou 0.6016511867905057 t 2.8
150 iou 0.6514879425346332 t 4.1
200 iou 0.6097749196141479 t 5.4
250 iou 0.5069525666016894 t 6.8
300 iou 0.5963578780680918 t 8.1
350 iou 0.6185878092939047 t 9.5
400 iou 0.6438515081206496 t 10.8
450 iou 0.8053252951519718 t 12.1
500 iou 0.8609315068493151 t 13.5
550 iou 0.8361735969097932 t 14.8
600 iou 0.8531890660592255 t 16.1
650 iou 0.8752138701950496 t 17.4
700 iou 0.8545577923546709 t 18.7
750 iou 0.6319252077562327 t 20.1
phase 0 298 iou 0.6157027647260532 fp 0.10667110731859734 fn 0.14165190598772587 rm 0.013130391 rmad 0.07782463
phase 299 539 iou 0.7276669297538519 fp 0.13745105365764573 fn 0.12884713504645118 rm 0.19639796 rmad 0.11308222
phase 540 793 iou 0.7925178413057005 fp 0. …[+906 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/view_static_opt.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2
z=np.load('/work/inputs/reconstruction.npz');cap=cv2.VideoCapture('/work/inputs/video.mp4')
sets=[('init','/work/scratch/static_init_T.npy',[0,50,100,150,200,250,298]),('final','/work/scratch/static_final_T.npy',[540,600,650,700,750,793])]
ims=[]
for name,p,idxs in sets:
 T=np.load(p)
 for i in idxs:
  mask,d,rgb=render(T);mask=mask[::-1];rgb=rgb[::-1];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=im[:,:,::-1]
  ov=im.copy();ov[mask]=(.35*im[mask]+.65*rgb[mask]).astype(np.uint8);cnts,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cnts,-1,(255,0,0),2);ov=cv2.resize(cv2.cvtColor(ov,cv2.COLOR_RGB2BGR),(640,480));cv2.putText(ov,name+' '+str(i),(12,38),0,1,(0,0,255),2);ims.append(ov)
while len(ims)<14:ims.append(np.zeros_like(ims[0]))
out=np.vstack([np.hstack(ims[i:i+2]) for i in range(0,14,2)]);cv2.imwrite('/work/scratch/vis/static_opt.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92])
PY
python /work/scratch/view_static_opt.py"
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
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
for name,p,rng in [('init','/work/scratch/static_init_T.npy',[295,296,297,298,299,300,301,302,305,310]),('final','/work/scratch/static_final_T.npy',[530,535,538,539,540,541,542,545,550])]:
 S=np.load(p);print('\\n',name)
 for i in rng:
  a=T[i];ang=np.degrees(np.linalg.norm((R.from_matrix(S[:3,:3]).inv()*R.from_matrix(a[:3,:3])).as_rotvec()));dt=np.linalg.norm(S[:3,3]-a[:3,3]);c1=S[:3,:3]@np.array([.27196912,-.0040119,.00456923])+S[:3,3];c2=a[:3,:3]@np.array([.27196912,-.0040119,.00456923])+a[:3,3];dc=np.linalg.norm(c1-c2)
  print(i,'rot',round(ang,2),'dt',round(dt,3),'dc',round(dc,3))
PY"
```
<details><summary>output</summary>

```

 init
295 rot 10.8 dt 0.142 dc 0.101
296 rot 11.23 dt 0.145 dc 0.102
297 rot 11.72 dt 0.143 dc 0.098
298 rot 12.01 dt 0.157 dc 0.11
299 rot 12.66 dt 0.152 dc 0.102
300 rot 12.92 dt 0.162 dc 0.109
301 rot 13.13 dt 0.164 dc 0.111
302 rot 13.09 dt 0.166 dc 0.112
305 rot 13.76 dt 0.182 dc 0.125
310 rot 15.38 dt 0.207 dc 0.143

 final
530 rot 2.67 dt 0.067 dc 0.064
535 rot 3.01 dt 0.06 dc 0.053
538 rot 3.0 dt 0.07 dc 0.066
539 rot 2.86 dt 0.066 dc 0.06
540 rot 2.79 dt 0.06 dc 0.052
541 rot 2.99 dt 0.063 dc 0.057
542 rot 3.04 dt 0.06 dc 0.054
545 rot 2.96 dt 0.073 dc 0.07
550 rot 2.91 dt 0.068 dc 0.06

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for i in range(0,311,10):
 ang=np.degrees(np.linalg.norm((R.from_matrix(T[0,:3,:3]).inv()*R.from_matrix(T[i,:3,:3])).as_rotvec()))
 print(i,round(ang,2),np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
0 0.0 [0.492 0.475 2.872]
10 0.37 [0.488 0.475 2.879]
20 0.35 [0.488 0.473 2.874]
30 0.6 [0.492 0.467 2.864]
40 0.53 [0.493 0.472 2.859]
50 0.29 [0.494 0.475 2.873]
60 0.36 [0.492 0.477 2.883]
70 0.9 [0.494 0.474 2.869]
80 0.97 [0.498 0.477 2.892]
90 0.88 [0.495 0.481 2.884]
100 1.01 [0.495 0.481 2.879]
110 0.71 [0.495 0.484 2.88 ]
120 1.36 [0.499 0.476 2.865]
130 0.98 [0.493 0.482 2.87 ]
140 0.94 [0.494 0.48  2.879]
150 1.11 [0.492 0.478 2.878]
160 0.76 [0.492 0.478 2.876]
170 1.12 [0.49  0.475 2.866]
180 0.67 [0.477 0.476 2.863]
190 0.95 [0.477 0.473 2.856]
200 1.05 [0.474 0.473 2.851]
210 0.83 [0.469 0.475 2.86 ]
220 0.64 [0.477 0.477 2.873]
230 0.93 [0.477 0.484 2.897]
240 1.08 [0.476 0.482 2.89 ]
250 1.03 [0.48  0.481 2.885]
260 0.98 [0.481 0.484 2.891]
270 2.19 [0.473 0.491 2.906]
28 …[+115 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
for i in range(240,331,2):
 ys,xs=np.where(o[i]); print(i,len(xs),xs.min(),ys.min(),xs.max(),ys.max(),round(xs.mean(),1),round(ys.mean(),1))
PY"
```
<details><summary>output</summary>

```
240 17773 779 605 1015 965 889.9 747.4
242 17794 778 605 1015 965 889.7 747.4
244 17722 779 605 1015 965 890.2 747.2
246 17693 779 605 1015 965 890.0 747.2
248 18368 779 605 1016 965 888.7 747.8
250 18499 778 605 1016 965 888.1 748.5
252 18581 777 605 1016 965 887.3 749.3
254 18555 776 605 1016 965 887.3 749.8
256 18382 776 605 1016 965 887.9 749.9
258 18188 776 605 1016 965 888.9 749.9
260 17879 777 605 1016 965 890.3 749.4
262 17208 777 605 1015 965 891.5 749.0
264 17271 779 605 1015 965 892.7 748.9
266 16904 779 605 1015 965 893.5 749.0
268 16343 779 605 1014 964 894.2 748.6
270 16216 779 605 1015 964 893.8 749.4
272 16567 777 605 1014 963 891.7 750.2
274 16836 776 605 1014 963 889.3 751.6
276 17576 775 605 1013 963 886.6 752.0
278 17509 773 605 1012 961 885.3 752.5
280 17419 772 605 10 …[+972 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,270);ok,im=cap.read()
cols=[(0,0,255),(0,128,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,255)]
for i,c in zip([250,260,270,280,290,300,310],cols):
 cnts,_=cv2.findContours(o[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnts,-1,c,1);cv2.putText(im,str(i),(1020,650+cols.index(c)*18),0,.5,c,1)
crop=im[540:1000,650:1060];crop=cv2.resize(crop,(820,920));cv2.imwrite('/work/scratch/vis/contours_motion.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,530);ok,im=cap.read()
cols=[(0,0,255),(0,128,255),(0,255,255),(0,255,0),(255,255,0),(255,0,0),(255,0,255)]
for i,c in zip([500,510,520,530,540,550,560],cols):
 cnts,_=cv2.findContours(o[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnts,-1,c,1);cv2.putText(im,str(i),(700,680+cols.index(c)*18),0,.5,c,1)
crop=im[600:930,500:930];crop=cv2.resize(crop,(860,660));cv2.imwrite('/work/scratch/vis/contours_end.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_corrections.py <<'PY'
import numpy as np, torch, trimesh, cv2, h5py, json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
S=.5;W=int(1536*S);H=int(1152*S)
sm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(sm.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(sm.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float).copy();K[:2]*=S;Twc=np.array(cam['T_world_cam'],float);Tcw=np.linalg.inv(Twc);gl=dr.RasterizeCudaContext()
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'].astype(float);Si=np.load('/work/scratch/static_init_T.npy');Sf=np.load('/work/scratch/static_final_T.npy')
with h5py.File('/work/inputs/masks.h5','r') as f:Oall=f['object'][:];Pall=f['person'][:]
def inv(T):
 U=np.eye(4);U[:3,:3]=T[:3,:3].T;U[:3,3]=-T[:3,:3].T@T[:3,3];return U
def corr(S,B): return S@inv(B)
def interp_corr(C0,C1,p):
 C=np.eye(4);r0=Rot.from_matrix(C0[:3,:3]);r1=Rot.from_matrix(C1[:3,:3]);C[:3,:3]=(r0.proj(r1.proj(r0.inv())).as_matrix() if False else (r1*r0.inv()).as_matrix())
 # above wrong; use slerp of relative
 C[:3,:3]=r0.as_matrix() if p==0 else (r1*r0.inv()).as_matrix() # placeholder
 rel=Rot.from_matrix(C1[:3,:3])*Rot.from_matrix(C0[:3,:3]).inv()
 C[:3,:3]=(Rot.from_rotvec(rel.as_rotvec()*p)*Rot.from_matrix(C0[:3,:3])).as_matrix()
 C[:3,3]=(1-p)*C0[:3,3]+p*C1[:3,3];return C
def …[+2101 chars]
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]

 baseline mean [0.65762195 0.72503041 0.1372453  0.12915072] visible frac 0.9283974447155059 iou when vis>.8 0.7334648746947862
294 iou 0.583 vis 0.77 fp 0.121 fn 0.155
300 iou 0.596 vis 0.81 fp 0.145 fn 0.164
320 iou 0.622 vis 0.8 fp 0.173 fn 0.137
350 iou 0.619 vis 0.8 fp 0.187 fn 0.141
400 iou 0.644 vis 0.9 fp 0.228 fn 0.171
450 iou 0.805 vis 0.98 fp 0.101 fn 0.097
500 iou 0.861 vis 1.02 fp 0.051 fn 0.083
530 iou 0.86 vis 0.96 fp 0.039 fn 0.054
539 iou 0.841 vis 0.96 fp 0.035 fn 0.067

 freeze mean [0.65762195 0.72503041 0.1372453  0.12915072] visible frac 0.9283974447155059 iou when vis>.8 0.7334648746947862
294 iou 0.583 vis 0.77 fp 0.121 fn 0.155
300 iou 0.5 …[+1293 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for i in [293,294,295,300,320,350,400,450,500,530,539]:
 p=(i-294)/(539-294);p=max(0,p);C=interp_corr(C0,C1,p);T=C@B[i]
 print(i,'corrT center',np.round(T[:3,:3]@np.array([.27196912,-.0040119,.00456923])+T[:3,3],3),'angleS',np.degrees(np.linalg.norm((Rot.from_matrix(Si[:3,:3]).inv()*Rot.from_matrix(T[:3,:3])).as_rotvec())))
print('C0 Si compare',np.max(abs(C0@B[293]-Si)),'C0 identity?',np.max(abs(C0@B[294]-Si)))
PY"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
293 corrT center [0.486 0.747 2.856] angleS 9.236618425881686e-12
294 corrT center [0.48  0.744 2.85 ] angleS 0.7458080021113872
295 corrT center [0.476 0.741 2.837] angleS 1.065487083963609
300 corrT center [0.459 0.72  2.824] angleS 4.176338534283867
320 corrT center [0.421 0.682 2.875] angleS 11.130270477335786
350 corrT center [0.402 0.689 2.922] angleS 14.14822954848845
400 corrT center [0.343 0.695 3.064] angleS 21.083405980280506
450 corrT center [0.195 0.703 3.215] angleS 42.97887975720894
500 corrT center [-0.007  0.871  3.421] angleS 80.35385830553209
530 corrT center [-0.029  0.908  3.446] angleS 88.92883902804758
539 corrT center [-0.031  0.91   3.459]  …[+95 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/debug_interp.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
for i in [293,294,300]:
 for nm,T in [('Si',Si),('B',B[i]),('corr',interp_corr(C0,C1,0 if i==294 else (i-294)/245)@B[i])]:
  m=render_mask(T);o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);inter=(m&o).sum();uni=(m|o).sum();print(i,nm,'iou',inter/uni,'fp',(m&~o&~p).sum()/m.sum(),'fn',(o&~m).sum()/o.sum(),'area',m.sum(),o.sum())
PY
python /work/scratch/debug_interp.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
293 Si iou 0.26694134229643285 fp 0.5713304013194063 fn 0.44159673395327737 area 7276 4409
293 B iou 0.5745375408052231 fp 0.13933566433566433 fn 0.16171467452937174 area 5720 4409
293 corr iou 0.2652884511310748 fp 0.5738222771597308 fn 0.4440916307552733 area 7281 4409
294 Si iou 0.24538371224250186 fp 0.6001924134139637 fn 0.47654826958105645 area 7276 4392
294 B iou 0.5827970491288652 fp 0.12122276879831342 fn 0.1545992714025501 area 5692 4392
294 corr iou 0.2698325856220593 fp 0.5674476639401081 fn 0.4385245901639344 area 7213 4392
300 Si iou 0.1496193636541089 fp 0.7395547003848268 fn 0.6595602931379081 area 7276 4503
300 B iou 0.5963578780680918 fp 0.1446755 …[+141 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/err_static293.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2,h5py
Si=np.load('/work/scratch/static_init_T.npy');cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [250,280,290,293,294,300]:
  m,d,rgb=render(Si);m=m[::-1];o=f['object'][i];p=f['person'][i];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();out=im.copy();out[m&~o&~p]=(.2*out[m&~o&~p]+.8*np.array([0,0,255])).astype(np.uint8);out[o&~m]=(.2*out[o&~m]+.8*np.array([0,255,0])).astype(np.uint8);out[m&o]=(.2*out[m&o]+.8*np.array([0,255,255])).astype(np.uint8);cv2.putText(out,str(i),(700,550),0,1,(0,0,255),2);crop=out[520:1000,550:1080];cv2.imwrite(f'/work/scratch/vis/staticky_{i}.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,92])
PY
python /work/scratch/err_static293.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/err_static293.py';s=open(p).read().replace('[250,280,290,293,294,300]','[260,270,275,280,285,290]').replace('staticky_','staticky2_');open('/work/scratch/err_static2.py','w').write(s)
PY
python /work/scratch/err_static2.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/for i in range(294,540):/for i in range(275,540):/; s/Tc\\[:294\\]/Tc[:275]/; s/Tk\\[:294\\]/Tk[:275]/; s/for i in range(294,540):/for i in range(275,540):/' /work/scratch/eval_corrections.py > /work/scratch/eval_corr275.py
# Need fix boundary definitions manually via python replacements
python - <<'PY'
p='/work/scratch/eval_corr275.py';s=open(p).read();s=s.replace('C0=corr(Si,B[293])','C0=corr(Si,B[274])').replace('(i-294)/(539-294)','(i-275)/(539-275)').replace('if i==293','if i==274').replace('range(294,540)','range(275,540)').replace('Tc[:294]','Tc[:275]').replace('Tk[:294]','Tk[:275]').replace('Tc[540:]','Tc[540:]').replace('Tk[540:]','Tk[540:]').replace('for i in range(275,540):\\n  m=render_mask','for i in range(275,540):\\n  m=render_mask')
s=s.replace(\"print('\\\\n',name,'mean',np.mean(vals,0)\",\"print('\\\\n',name,'mean',np.mean(vals,0)\")
open(p,'w').write(s)
PY
grep -n 'C0\\|range(275\\|Tc\\[\\|Tk\\[' /work/scratch/eval_corr275.py | head -30
python /work/scratch/eval_corr275.py"
```
<details><summary>output</summary>

```
C0 ang 4.472430902871871 t [ 0.24440189  0.01073595 -0.11594896]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]

 baseline mean [0.64329844 0.71099025 0.13700886 0.13159138] visible frac 0.9124396513748665 iou when vis>.8 0.7334648746947862
294 iou 0.583 vis 0.77 fp 0.121 fn 0.155
300 iou 0.596 vis 0.81 fp 0.145 fn 0.164
320 iou 0.622 vis 0.8 fp 0.173 fn 0.137
350 iou 0.619 vis 0.8 fp 0.187 fn 0.141
400 iou 0.644 vis 0.9 fp 0.228 fn 0.171
450 iou 0.805 vis 0.98 fp 0.101 fn 0.097
500 iou 0.861 vis 1.02 fp 0.051 fn 0.083
530 iou 0.86 vis 0.96 fp 0.039 fn 0.054
539 iou 0.841 vis 0.96 fp 0.035 fn 0.067

 freeze mean [0.63088558 0.70445126 0.15273362 0.13405945] visible frac 0.9051302668408441 iou when vis>.8 0.7334648746947862
294 iou 0.583 vis 0.77 fp 0.121 fn 0.155
300 iou 0.5 …[+1292 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'].astype(float);Si=np.load('/work/scratch/static_init_T.npy');Sf=np.load('/work/scratch/static_final_T.npy');mc=np.array([.27196912,-.0040119,.00456923])
for i in range(255,286):
 a=B[i];c1=Si[:3,:3]@mc+Si[:3,3];c2=a[:3,:3]@mc+a[:3,3];ang=np.degrees(np.linalg.norm((R.from_matrix(Si[:3,:3]).inv()*R.from_matrix(a[:3,:3])).as_rotvec()));print(i,'rot',round(ang,2),'dc',round(np.linalg.norm(c1-c2),3),'centerdiff',np.round(c1-c2,3))
PY"
```
<details><summary>output</summary>

```
255 rot 3.38 dc 0.049 centerdiff [ 0.003 -0.008 -0.048]
256 rot 3.15 dc 0.054 centerdiff [ 0.003 -0.01  -0.053]
257 rot 3.33 dc 0.049 centerdiff [ 0.004 -0.007 -0.049]
258 rot 3.21 dc 0.048 centerdiff [ 0.003 -0.007 -0.047]
259 rot 3.11 dc 0.051 centerdiff [ 0.001 -0.008 -0.05 ]
260 rot 3.24 dc 0.051 centerdiff [ 0.    -0.008 -0.05 ]
261 rot 3.48 dc 0.054 centerdiff [ 0.002 -0.01  -0.053]
262 rot 3.44 dc 0.05 centerdiff [ 0.002 -0.009 -0.05 ]
263 rot 3.54 dc 0.052 centerdiff [ 0.005 -0.01  -0.051]
264 rot 3.61 dc 0.051 centerdiff [ 0.003 -0.011 -0.049]
265 rot 3.89 dc 0.051 centerdiff [ 0.004 -0.01  -0.05 ]
266 rot 3.88 dc 0.052 centerdiff [ 0.003 -0.011 -0.05 ]
267 rot 3.91 dc 0.062 centerdiff [ 0.006 -0.013 -0.06 ]
268 rot 3.96 dc 0.059 centerdiff [ 0.007 -0.012 -0.057]
269 rot 4.19 dc 0 …[+932 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/compare_static.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
for rng in [(0,75),(76,150),(151,225),(226,270),(271,290),(294,539),(540,600),(601,715)]:
 vals={k:[] for k in ['B','Si','Sf']}
 for i in range(rng[0],rng[1]+1):
  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  for k,T in [('B',B[i]),('Si',Si),('Sf',Sf)]:
   m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();vals[k].append((inter/uni,(m&~o&~p).sum()/m.sum(),(o&~m).sum()/o.sum()))
 print(rng,{k:np.round(np.mean(v,0),4) for k,v in vals.items()})
PY
python /work/scratch/compare_static.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
(0, 75) {'B': array([0.7549, 0.1182, 0.1604]), 'Si': array([0.8381, 0.0819, 0.0942]), 'Sf': array([0.016 , 0.7409, 0.9666])}
(76, 150) {'B': array([0.5906, 0.0943, 0.1307]), 'Si': array([0.6328, 0.0628, 0.0775]), 'Sf': array([0.0053, 0.5179, 0.9879])}
(151, 225) {'B': array([0.5996, 0.0924, 0.1251]), 'Si': array([0.6441, 0.1091, 0.0744]), 'Sf': array([0.0057, 0.5435, 0.9858])}
(226, 270) {'B': array([0.4986, 0.1169, 0.147 ]), 'Si': array([0.5316, 0.1597, 0.0847]), 'Sf': array([0.0015, 0.5385, 0.9958])}
(271, 290) {'B': array([0.5144, 0.1303, 0.1594]), 'Si': array([0.4736, 0.2973, 0.1433]), 'Sf': array([0.0034, 0.4764, 0.9902])}
(294, 539) {'B': array([0.725 , 0.137 …[+347 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# reuse definitions without metrics
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
for start in range(500,721,10):
 vals=[]
 for i in range(start,min(start+20,794)):
  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  row=[]
  for T in [B[i],Sf]:
   m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();row.append((inter/uni,(m&~o&~p).sum()/m.sum(),(o&~m).sum()/o.sum()))
  vals.append(row)
 print(start,'B',np.round(np.mean(vals,0)[0],4),'Sf',np.round(np.mean(vals,0)[1],4))
PY"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
500 B [0.8846 0.0317 0.0793] Sf [0.6419 0.1559 0.2339]
510 B [0.8929 0.0285 0.0649] Sf [0.7266 0.1163 0.166 ]
520 B [0.8723 0.0333 0.0568] Sf [0.7433 0.103  0.1406]
530 B [0.8505 0.0347 0.0585] Sf [0.7267 0.1027 0.1417]
540 B [0.8354 0.036  0.0718] Sf [0.7326 0.0998 0.1349]
550 B [0.8192 0.0357 0.0928] Sf [0.7355 0.1019 0.1296]
560 B [0.8246 0.033  0.0913] Sf [0.7502 0.0944 0.1198]
570 B [0.8491 0.0315 0.0672] Sf [0.8046 0.0632 0.0897]
580 B [0.8623 0.0347 0.0505] Sf [0.8429 0.0437 0.0673]
590 B [0.8589 0.0415 0.0484] Sf [0.8503 0.0423 0.0602]
600 B [0.8563 0.043  0.0522] Sf [0.8519 0.0432 0.0586]
610 B [0.8625 0.0438 0.0517] Sf [0.8598 0.0427 0.058 ]
620 B [0.8816 …[+592 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
for i in [260,265,268,269,270,271,272,275,280]:
 o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 out=[]
 for T in [B[i],Si]:
  m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();out.append((inter/uni,(m&~o&~p).sum()/m.sum(),(o&~m).sum()/o.sum()))
 print(i,'B',np.round(out[0],3),'S',np.round(out[1],3))
PY"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
260 B [0.496 0.118 0.151] S [0.533 0.149 0.086]
265 B [0.483 0.117 0.139] S [0.51  0.171 0.082]
268 B [0.478 0.117 0.133] S [0.492 0.198 0.083]
269 B [0.482 0.112 0.127] S [0.492 0.204 0.083]
270 B [0.476 0.12  0.147] S [0.492 0.213 0.077]
271 B [0.49  0.109 0.132] S [0.498 0.212 0.085]
272 B [0.489 0.121 0.137] S [0.502 0.216 0.079]
275 B [0.499 0.121 0.145] S [0.51  0.221 0.089]
280 B [0.507 0.127 0.159] S [0.506 0.259 0.102]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_ramps.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
def slerp_corr(C,p):
 # p 0->C, 1->I, left-interpolate
 R=Rot.from_matrix(C[:3,:3]);r=(R.inv()*Rot.from_matrix(np.eye(3))).as_rotvec() # wrong orientation?
 # want R(p)=exp(p log(R)); then translation interpolate
 out=np.eye(4);out[:3,:3]=Rot.from_rotvec(R.as_rotvec()*p).as_matrix();out[:3,3]=C[:3,3]*(1-p);return out
def ramp(b0,lam):return slerp_corr(b0,lam)
C0=corr(Si,B[274]);C1=corr(Sf,B[649])
for L1,L2 in [(20,20),(30,30),(50,50),(30,50),(50,30)]:
 T=B.copy();T[:275]=Si
 for i in range(275,min(275+L1,794)):
  p=(i-274)/L1;T[i]=ramp(C0,p)@B[i]
 for i in range(650-L2,650):
  p=(i-(650-L2)+1)/L2;T[i]=ramp(C1,1-p)@B[i]
 T[650:]=Sf
 vals=[]
 for i in list(range(250,330))+list(range(590,680)):
  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);m=render_mask(T[i]);inter=(m&o).sum();uni=(m|o).sum();iou=inter/max(1,uni);fp=(m&~o&~p).sum()/m.sum();fn=(o&~m).sum()/o.sum();vals.append((iou,fp,fn))
 print('L',L1,L2,'mean',np.mean(vals,0),'early',np.mean(vals[:80],0),'late',np.mean(vals[80:],0))
np.savez('/work/scratch/ramp_traj.npz',**{f'L{a}_{b}':T for a,b in [(0,0)]})
PY
python /work/scratch/eval_ramps.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
L 20 20 mean [0.6691709  0.14345958 0.16123422] early [0.49409709 0.21990552 0.23521998] late [0.82479207 0.07550763 0.0954691 ]
L 30 30 mean [0.63076553 0.17229293 0.2020098 ] early [0.44154705 0.26392908 0.30172975] late [0.79895973 0.09083858 0.11336984]
L 50 50 mean [0.54756816 0.23353601 0.28489871] early [0.33111641 0.35452373 0.43209521] late [0.73996972 0.12599137 0.15405738]
L 30 50 mean [0.59953552 0.19090323 0.22355026] early [0.44154705 0.26392908 0.30172975] late [0.73996972 0.12599137 0.15405738]
L 50 30 mean [0.57879817 0.21492571 0.26335825] early [0.33111641 0.35452373 0.43209521] late [0.79895973 0.09083858 0.11336984]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_ramps.py';s=open(p).read()
start=s.index('for L1,L2 in');end=s.index(\"np.savez\",start)
new='''for L1,L2 in [(1,1),(2,2),(4,4),(6,6),(8,8),(10,10),(1,8),(4,8)]:\\n T=B.copy();T[:275]=Si\\n for i in range(275,min(275+L1,794)):\\n  p=(i-274)/L1;T[i]=ramp(C0,p)@B[i]\\n for i in range(650-L2,650):\\n  p=(i-(650-L2)+1)/L2;T[i]=ramp(C1,1-p)@B[i]\\n T[650:]=Sf\\n vals=[];allm=[]\\n for i in range(250,680):\\n  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);m=render_mask(T[i]);inter=(m&o).sum();uni=(m|o).sum();iou=inter/max(1,uni);fp=(m&~o&~p).sum()/m.sum();fn=(o&~m).sum()/o.sum();vals.append((iou,fp,fn));allm.append(iou)\\n print('L',L1,L2,'early',np.mean(vals[:80],0),'trans',np.mean(vals[40:60],0),'late',np.mean(vals[400:],0),'overall',np.mean(allm))\\n'''
s=s[:start]+new+s[end:];open('/work/scratch/eval_ramps2.py','w').write(s)
PY
python /work/scratch/eval_ramps2.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
L 1 1 early [0.56767405 0.15291669 0.1391891 ] trans [0.61213887 0.13636015 0.15046424] late [0.9025086  0.03811258 0.064021  ] overall 0.7484420451136218
L 2 2 early [0.56764434 0.15372134 0.13891851] trans [0.61213887 0.13636015 0.15046424] late [0.9025086  0.03811258 0.064021  ] overall 0.7484800573113952
L 4 4 early [0.55973481 0.16189568 0.14998546] trans [0.61213887 0.13636015 0.15046424] late [0.9025086  0.03811258 0.064021  ] overall 0.7457419834071232
L 6 6 early [0.55187323 0.16935974 0.16056434] trans [0.61213887 0.13636015 0.15046424] late [0.9025086  0.03811258 0.064021  ] overall 0.7430776573975048
L 8 8 early [0.54465551 0.17596355 0.16989966] trans  …[+568 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
for i in range(615,646):
 o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);row=[]
 for T in [B[i],Sf]:
  m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();row.append((inter/uni,(m&~o&~p).sum()/m.sum(),(o&~m).sum()/o.sum()))
 print(i,'B',np.round(row[0],3),'S',np.round(row[1],3))
PY"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
615 B [0.864 0.04  0.047] S [0.853 0.043 0.058]
616 B [0.863 0.037 0.053] S [0.855 0.042 0.059]
617 B [0.867 0.038 0.048] S [0.856 0.042 0.059]
618 B [0.868 0.034 0.052] S [0.856 0.042 0.059]
619 B [0.86  0.035 0.059] S [0.856 0.042 0.059]
620 B [0.869 0.036 0.049] S [0.857 0.041 0.058]
621 B [0.859 0.046 0.05 ] S [0.858 0.043 0.058]
622 B [0.854 0.051 0.055] S [0.858 0.044 0.058]
623 B [0.856 0.052 0.053] S [0.86  0.043 0.057]
624 B [0.862 0.051 0.047] S [0.864 0.042 0.058]
625 B [0.849 0.061 0.052] S [0.866 0.043 0.057]
626 B [0.866 0.051 0.049] S [0.869 0.043 0.057]
627 B [0.871 0.053 0.048] S [0.873 0.043 0.057]
628 B [0.887 0.039 0.045] S [0.876 0.042 0.057]
6 …[+815 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);mc=np.array([.27196912,-.0040119,.00456923]);c=np.einsum('fij,j->fi',T[:,:3,:3],mc)+T[:,:3,3]
q=R.from_matrix(T[:,:3,:3]).as_quat();rv=R.from_matrix(T[:,:3,:3]).as_rotvec();rel=(R.from_matrix(T[:-1,:3,:3]).inv()*R.from_matrix(T[1:,:3,:3])).as_rotvec();av=np.degrees(np.linalg.norm(rel,axis=1));vel=np.linalg.norm(np.diff(c,axis=0),axis=1)
fig,axs=plt.subplots(3,1,figsize=(18,10),sharex=True)
for j,n in enumerate('xyz'):axs[0].plot(c[:,j],label=n)
axs[0].legend();axs[0].set_ylabel('center')
axs[1].plot(av);axs[1].set_ylabel('angular speed deg/frame')
axs[2].plot(vel);axs[2].set_ylabel('center speed m/frame')
for ax in axs:ax.grid();ax.axvline(275,color='r');ax.axvline(650,color='r')
plt.tight_layout();plt.savefig('/work/scratch/vis/motion.png',dpi=110)
# print peaks
for i in np.argsort(vel)[-30:]:print('v',i+1,round(vel[i],4),'center',c[i+1].round(3))
PY"
```
<details><summary>output</summary>

```
v 356 0.0257 center [0.365 0.68  3.005]
v 589 0.026 center [-0.019  0.887  3.416]
v 495 0.0261 center [0.016 0.847 3.495]
v 506 0.0262 center [-1.00e-03  8.62e-01  3.46e+00]
v 513 0.0266 center [-0.012  0.869  3.469]
v 355 0.0266 center [0.366 0.688 3.03 ]
v 474 0.0273 center [0.103 0.749 3.424]
v 299 0.0274 center [0.413 0.713 2.919]
v 354 0.0278 center [0.369 0.678 3.005]
v 369 0.0279 center [0.36  0.687 3.054]
v 512 0.0282 center [-0.012  0.874  3.495]
v 560 0.0286 center [-0.006  0.877  3.521]
v 739 0.0287 center [-0.012  0.888  3.434]
v 487 0.0305 center [0.04  0.813 3.495]
v 346 0.0313 center [0.368 0.679 2.989]
v 546 0.0327 center [-0.01   0.867  3.447]
v 556 0.0328 center [-0.016  0.88   3.492]
v 363 0.0336 center [0.357 0.681 3.027]
v 384 0.0339 center [0.344 0.676 3.059]
v 494 0. …[+438 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_smooth.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
from scipy.signal import savgol_filter
# smooth quaternion with sign alignment
q=Rot.from_matrix(B[:,:3,:3]).as_quat()
for i in range(1,len(q)):
 if np.dot(q[i],q[i-1])<0:q[i]*=-1
def make(w):
 qs=q.copy();ts=B[:,:3,3].copy()
 qs[275:642]=savgol_filter(qs[275:642],w,2,axis=0,mode='interp');qs[275:642]/=np.linalg.norm(qs[275:642],axis=1,keepdims=True)
 ts[275:642]=savgol_filter(ts[275:642],w,2,axis=0,mode='interp')
 T=np.eye(4)[None].repeat(len(B),0);T[:3,:3]=Rot.from_quat(qs).as_matrix();T[:3,3]=ts;return T
for w in [3,5,7,9,11,15]:
 T=make(w);vals=[];acc0=[];acc1=[]
 for i in range(275,642):
  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);m=render_mask(T[i]);inter=(m&o).sum();uni=(m|o).sum();iou=inter/max(1,uni);fp=(m&~o&~p).sum()/m.sum();fn=(o&~m).sum()/o.sum();vals.append((iou,fp,fn))
  if 300<i<630:
   acc0.append(np.linalg.norm(B[i+1,:3,3]-2*B[i,:3,3]+B[i-1,:3,3]));acc1.append(np.linalg.norm(T[i+1,:3,3]-2*T[i,:3,3]+T[i-1,:3,3]))
 print(w,'metrics',np.round(np.mean(vals,0),4),'acc',np.mean(acc0),np.mean(acc1))
# baseline metrics once on same
vals=[]
for i in range(275,642):
 o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.u …[+290 chars]
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
Traceback (most recent call last):
  File "/work/scratch/eval_smooth.py", line 14, in <module>
    T=make(w);vals=[];acc0=[];acc1=[]
      ^^^^^^^
  File "/work/scratch/eval_smooth.py", line 12, in make
    T=np.eye(4)[None].repeat(len(B),0);T[:3,:3]=Rot.from_quat(qs).as_matrix();T[:3,3]=ts;return T
                                       ~^^^^^^^
ValueError: could not broadcast input array from shape (794,3,3) into shape (3,3,4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/T\\[:3,:3\\]=Rot/T[:,:3,:3]=Rot/; s/T\\[:3,3\\]=ts/T[:,:3,3]=ts/' /work/scratch/eval_smooth.py
python /work/scratch/eval_smooth.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
3 metrics [0.7505 0.1094 0.1127] acc 0.017401305293728613 0.017401305293728613
5 metrics [0.751  0.1091 0.1123] acc 0.017401305293728613 0.007478237025989129
7 metrics [0.7512 0.109  0.1121] acc 0.017401305293728613 0.004334549991848203
9 metrics [0.7513 0.1089 0.1121] acc 0.017401305293728613 0.003183517483636071
11 metrics [0.7516 0.1087 0.1119] acc 0.017401305293728613 0.0024399312489658013
15 metrics [0.7519 0.1086 0.1116] acc 0.017401305293728613 0.0017904174845847765
BASE [0.7505 0.1093 0.1127]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# modify eval_smooth to save w11 and compare per frame quickly
p='/work/scratch/eval_smooth.py';s=open(p).read();
s=s.replace(\"for w in [3,5,7,9,11,15]:\",\"for w in [11]:\").replace(\"print(w,'metrics',np.round(np.mean(vals,0),4),'acc',np.mean(acc0),np.mean(acc1))\",\"print(w,'metrics',np.round(np.mean(vals,0),4),'acc',np.mean(acc0),np.mean(acc1));np.save('/work/scratch/smooth11_T.npy',T);smv=vals\")
s += \"\\nfor i in range(275,642):\\n print(i,round(vals[i-275][0],4),round(smv[i-275][0],4),round(smv[i-275][0]-vals[i-275][0],4))\\n\" 
# issue vals overwritten by baseline at end; not useful. skip
open('/work/scratch/save_smooth.py','w').write(s)
PY
python /work/scratch/save_smooth.py >/work/scratch/vis/smoothlog.txt
head -5 /work/scratch/vis/smoothlog.txt; tail -5 /work/scratch/vis/smoothlog.txt; ls -l /work/scratch/smooth11_T.npy"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
11 metrics [0.7516 0.1087 0.1119] acc 0.017401305293728613 0.0024399312489658013
BASE [0.7505 0.1093 0.1127]
275 0.4986 0.4992 0.0006
637 0.8989 0.9046 0.0057
638 0.8992 0.9045 0.0053
639 0.9079 0.9067 -0.0011
640 0.9126 0.9125 -0.0001
641 0.908 0.9125 0.0045
-rw-rw-r-- 1 hanzhou hanzhou 101760 Oct 20 21:00 /work/scratch/smooth11_T.npy

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);K=z['human_keypoints_world'].astype(float);C=z['contact_prob_right_left'];mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=mesh.vertices[::20]
for i in range(0,794,25):
 W=np.einsum('ij,kj->ki',T[i,:3,:3],V)+T[i,:3,3];tree=cKDTree(W)
 print(i,'cp',np.round(C[i],2),'Rd',np.round(np.percentile(tree.query(K[i,21:42])[0],[0,25,50]),3),'Ld',np.round(np.percentile(tree.query(K[i,42:63])[0],[0,25,50]),3))
PY"
```
<details><summary>output</summary>

```
0 cp [0. 0.] Rd [0.357 0.376 0.393] Ld [0.622 0.64  0.656]
25 cp [0. 0.] Rd [0.492 0.512 0.53 ] Ld [0.6   0.615 0.637]
50 cp [0. 0.] Rd [0.57  0.583 0.606] Ld [0.32  0.338 0.36 ]
75 cp [0. 0.] Rd [0.456 0.48  0.503] Ld [0.208 0.244 0.275]
100 cp [0. 0.] Rd [0.208 0.237 0.261] Ld [0.151 0.178 0.198]
125 cp [0. 0.] Rd [0.107 0.159 0.192] Ld [0.253 0.279 0.294]
150 cp [0.69 0.95] Rd [0.006 0.035 0.054] Ld [0.08  0.086 0.101]
175 cp [1. 1.] Rd [0.015 0.041 0.054] Ld [0.004 0.021 0.036]
200 cp [1. 1.] Rd [0.011 0.038 0.051] Ld [0.006 0.03  0.047]
225 cp [1. 1.] Rd [0.008 0.021 0.032] Ld [0.006 0.016 0.029]
250 cp [1. 1.] Rd [0.008 0.016 0.024] Ld [0.006 0.018 0.026]
275 cp [1. 1.] Rd [0.005 0.017 0.03 ] Ld [0.003 0.008 0.024]
300 cp [1. 1.] Rd [0.008 0.015 0.022] Ld [0.045 0.057 0.071]
325 cp [ …[+1155 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'].astype(float);K=z['human_keypoints_world'].astype(float);Si=np.load('/work/scratch/static_init_T.npy');Sf=np.load('/work/scratch/static_final_T.npy');V=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices[::20]
for name,T in [('B',B),('Si',Si),('Sf',Sf)]:
 for a,b in [(140,200),(200,270),(600,640),(640,715)]:
  ds=[]
  for i in range(a,b,5):
   W=np.einsum('ij,kj->ki',T[i,:3,:3] if name=='B' else T[:3,:3],V)+ (T[i,:3,3] if name=='B' else T[:3,3]);tree=cKDTree(W);ds.append(tree.query(K[i,21:42])[0].min())
  print(name,a,b,'median',np.median(ds),'mean',np.mean(ds))
PY"
```
<details><summary>output</summary>

```
B 140 200 median 0.010334533883523647 mean 0.010290176711708942
B 200 270 median 0.0077260468684918785 mean 0.006917640552400044
B 600 640 median 0.0134015196158008 mean 0.03756759942830423
B 640 715 median 0.24415235875747762 mean 0.27730979147159684
Si 140 200 median 0.01362200219533792 mean 0.013075542102258192
Si 200 270 median 0.009675602588957177 mean 0.009901678810895178
Si 600 640 median 0.8291948781700038 mean 0.8364409318992985
Si 640 715 median 1.1103985428987149 mean 1.0392616501156249
Sf 140 200 median 0.6311704667725362 mean 0.6350015520962696
Sf 200 270 median 0.6276593870708989 mean 0.6288099302952398
Sf 600 640 median 0.029616610522893647 mean 0.04722704931650904
Sf 640 715 median 0.25522457972708573 mean 0.27279259858296284

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/score_static_photo.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2,h5py
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'].astype(float);Si=np.load('/work/scratch/static_init_T.npy');Sf=np.load('/work/scratch/static_final_T.npy');cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f:O=f['object'][:];P=f['person'][:]
for i,poses in [(0,[('B',B[i]),('Si',Si)]),(50,[('B',B[i]),('Si',Si)]),(650,[('B',B[i]),('Sf',Sf)]),(700,[('B',B[i]),('Sf',Sf)])]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
 print('\\n',i)
 for nm,T in poses:
  m,d,rgb=render(T);m=m[::-1];rgb=rgb[::-1];pred=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float);q=m&O[i]&~P[i];q=cv2.erode(q.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
  a=gray[q];b=pred[q];corr=np.corrcoef(a,b)[0,1];mae=np.mean(abs(a-b));gx1=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);gy1=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3);gx2=cv2.Sobel(pred,cv2.CV_64F,1,0,ksize=3);gy2=cv2.Sobel(pred,cv2.CV_64F,0,1,ksize=3);A=np.stack([gx1[q],gy1[q]],1);D=np.stack([gx2[q],gy2[q]],1);gc=np.corrcoef(A.ravel(),D.ravel())[0,1]
  inter=(m&O[i]).sum();uni=(m|O[i]).sum();print(nm,'iou',inter/uni,'corr',corr,'gcorr',gc,'mae',mae)
PY
python /work/scratch/score_static_photo.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/score_static_photo.py", line 5, in <module>
    for i,poses in [(0,[('B',B[i]),('Si',Si)]),(50,[('B',B[i]),('Si',Si)]),(650,[('B',B[i]),('Sf',Sf)]),(700,[('B',B[i]),('Sf',Sf)])]:
                               ^
NameError: name 'i' is not defined. Did you mean: 'Si'?

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/('B',B\\[i\\])/('B',B[0])/g\" /work/scratch/score_static_photo.py
# Oops all become B0; need individual proper. rewrite line
python - <<'PY'
p='/work/scratch/score_static_photo.py';s=open(p).read();s=s.replace(\"[(0,[('B',B[0]),('Si',Si)]),(50,[('B',B[0]),('Si',Si)]),(650,[('B',B[0]),('Sf',Sf)]),(700,[('B',B[0]),('Sf',Sf)])]\",\"[(0,[('B',B[0]),('Si',Si)]),(50,[('B',B[50]),('Si',Si)]),(650,[('B',B[650]),('Sf',Sf)]),(700,[('B',B[700]),('Sf',Sf)])]\")
open(p,'w').write(s)
PY
python /work/scratch/score_static_photo.py"
```
<details><summary>output</summary>

```

 0
B iou 0.7552873773389407 corr -0.2247660738881592 gcorr 0.006062938489414519 mae 82.1045796781101
Si iou 0.8423849806597692 corr -0.21033363622095355 gcorr 0.04032187970831276 mae 78.45190952001471

 50
B iou 0.7539578440045145 corr -0.23380926108187283 gcorr 0.021643527344778292 mae 81.7685270548188
Si iou 0.8361149454134772 corr -0.21428364319937826 gcorr 0.039683331395880636 mae 79.57610039564787

 650
B iou 0.8681135225375626 corr 0.15207403023320318 gcorr 0.028498968482391977 mae 29.130859464147154
Sf iou 0.8957413070562917 corr 0.08702600125238373 gcorr 0.023459064875649027 mae 29.567505917796428

 700
B iou 0.8481650343352471 corr 0.13143680093683602 gcorr -0.009932665337938006 mae 29.950408768304733
Sf iou 0.8936326622935987 corr 0.09046559643913499 gcorr 0.024258342604880957 m …[+21 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);S=np.load('/work/scratch/smooth11_T.npy');Si=np.load('/work/scratch/static_init_T.npy');Sf=np.load('/work/scratch/static_final_T.npy');mc=np.array([.27196912,-.0040119,.00456923])
def diff(A,C):
 ang=np.degrees(np.linalg.norm((R.from_matrix(A[:3,:3]).inv()*R.from_matrix(C[:3,:3])).as_rotvec()));c1=A[:3,:3]@mc+A[:3,3];c2=C[:3,:3]@mc+C[:3,3];return ang,np.linalg.norm(c1-c2)
for a,b in [(Si,S[275]),(S[274],S[275]),(S[640],S[641]),(S[641],Sf),(B[274],B[275]),(B[641],B[642])]:print(diff(a,b))
# compare smooth last to sf
for i in [635,640,641,642,650,660]:
 T=Sf if i>=642 else S[i]
 print(i,'B-S',diff(B[i],T),'B',B[i,:3,3])
PY"
```
<details><summary>output</summary>

```
(4.760291152597994, 0.07685017516336848)
(0.35360711836346864, 0.006179685876897957)
(0.1678558830352252, 0.0010941187417737835)
(2.5204172852875257, 0.01915957287960692)
(0.4042543318459156, 0.0013768646162446317)
(0.4802395066126728, 0.01181511845914769)
635 B-S (0.23303007768765424, 0.006741766305415618) B [-0.2324647   0.8855567   3.61423731]
640 B-S (0.2933783929725196, 0.008722874002853686) B [-0.23010655  0.88646698  3.61966419]
641 B-S (0.2901669681099436, 0.0040177860015765164) B [-0.23187907  0.88878965  3.63057876]
642 B-S (2.235343733903189, 0.012544529129556434) B [-0.23138775  0.88844031  3.64174771]
650 B-S (3.0000000000000058, 0.020000000029288777) B [-0.23436038  0.90253532  3.65930676]
660 B-S (3.1276233605281534, 0.02348684259165529) B [-0.23888135  0.90701658  3.6630373 …[+3 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_blend.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
def corr_interp(C,p): # p=1 C, p=0 I
 out=np.eye(4);r=Rot.from_matrix(C[:3,:3]);out[:3,:3]=Rot.from_rotvec(r.as_rotvec()*p).as_matrix();out[:3,3]=C[:3,3]*p;return out
def score(T,i):
 o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();iou=inter/max(1,uni);fp=(m&~o&~p).sum()/m.sum();fn=(o&~m).sum()/o.sum();return iou-.35*fp-.15*fn,(iou,fp,fn)
C0=corr(Si,B[270]);C1=corr(Sf,B[650])
print('C0',np.degrees(np.linalg.norm(Rot.from_matrix(C0[:3,:3]).as_rotvec())),C0[:3,3])
for i in range(260,306):
 out=[]
 for p in np.linspace(1,0,11):out.append((*score(corr_interp(C0,p)@B[i],i),p))
 best=max(out)
 print(i,'best',round(best[0],3),'p',best[2],'m',np.round(best[1],3),'static',round(out[0][0],3),'base',round(out[-1][0],3))
print('FINAL')
for i in range(620,671):
 out=[]
 for p in np.linspace(0,1,11):out.append((*score(corr_interp(C1,p)@B[i],i),p))
 best=max(out)
 print(i,'best',round(best[0],3),'p',best[2],'m',np.round(best[1],3),'base',round(out[0][0],3),'static',round(out[-1][0],3))
PY
python /work/scratch/opt_blend.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
C0 4.526199333416427 [ 0.24068207  0.03163273 -0.10101357]
260 best 0.432 p 0.0 m [0.496 0.118 0.151] static 0.381 base 0.432
261 best 0.422 p 0.09999999999999998 m [0.489 0.131 0.143] static 0.377 base 0.419
262 best 0.414 p 0.09999999999999998 m [0.484 0.137 0.144] static 0.368 base 0.413
263 best 0.425 p 0.0 m [0.486 0.117 0.133] static 0.376 base 0.425
264 best 0.432 p 0.19999999999999996 m [0.496 0.129 0.124] static 0.386 base 0.432
265 best 0.421 p 0.0 m [0.483 0.117 0.139] static 0.377 base 0.421
266 best 0.408 p 0.0 m [0.475 0.126 0.15 ] static 0.373 base 0.408
267 best 0.412 p 0.09999999999999998 m [0.477 0.13  0.131] static 0.382 base 0.409
268 best 0.42  …[+6430 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'].astype(float);c=z['contact_prob_right_left'];Si=np.load('/work/scratch/static_init_T.npy');Sf=np.load('/work/scratch/static_final_T.npy');mc=np.array([.27196912,-.0040119,.00456923])
print('first contact',np.where(c.max(1)>.1)[0][:20])
for i in [75,100,120,135,140,144,145,146,150,160]:
 c1=Si[:3,:3]@mc+Si[:3,3];c2=B[i,:3,:3]@mc+B[i,:3,3];ang=np.degrees(np.linalg.norm((R.from_matrix(Si[:3,:3]).inv()*R.from_matrix(B[i,:3,:3])).as_rotvec()));print(i,'rot',round(ang,2),'dc',round(np.linalg.norm(c1-c2),3),'cp',c[i])
PY"
```
<details><summary>output</summary>

```
first contact [141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158
 159 160]
75 rot 1.59 dc 0.039 cp [5.6630192e-06 2.0549367e-06]
100 rot 1.42 dc 0.04 cp [7.8187140e-06 1.1656044e-06]
120 rot 1.08 dc 0.028 cp [4.762884e-05 2.535121e-05]
135 rot 0.84 dc 0.042 cp [0.00165948 0.00031683]
140 rot 1.46 dc 0.039 cp [0.00062317 0.06068571]
144 rot 1.5 dc 0.046 cp [0.29139468 0.33963692]
145 rot 1.6 dc 0.045 cp [0.44831952 0.6370584 ]
146 rot 1.72 dc 0.041 cp [0.7713627 0.7950075]
150 rot 1.45 dc 0.038 cp [0.68715644 0.94567156]
160 rot 1.72 dc 0.036 cp [0.9762734 0.9834326]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
for i in range(100,301,10):
 o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);row=[]
 for T in [B[i],Si]:
  m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();iou=inter/uni;fp=(m&~o&~p).sum()/m.sum();fn=(o&~m).sum()/o.sum();sc=iou-.35*fp-.15*fn;row.append((sc,iou,fp,fn))
 print(i,'B',np.round(row[0],3),'S',np.round(row[1],3))
PY"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
100 B [0.549 0.602 0.094 0.131] S [0.608 0.641 0.058 0.08 ]
110 B [0.392 0.447 0.088 0.158] S [0.463 0.491 0.044 0.083]
120 B [0.353 0.404 0.079 0.161] S [0.429 0.453 0.037 0.072]
130 B [0.454 0.501 0.082 0.121] S [0.481 0.517 0.065 0.088]
140 B [0.519 0.567 0.085 0.125] S [0.564 0.596 0.058 0.079]
150 B [0.594 0.651 0.108 0.131] S [0.681 0.715 0.07  0.064]
160 B [0.588 0.641 0.09  0.139] S [0.673 0.704 0.057 0.071]
170 B [0.595 0.649 0.105 0.118] S [0.668 0.703 0.071 0.066]
180 B [0.568 0.615 0.082 0.123] S [0.617 0.663 0.098 0.073]
190 B [0.568 0.613 0.083 0.107] S [0.59  0.643 0.12  0.073]
200 B [0.565 0.61  0.081 0.11 ] S [0.577 0.635 0.131 0.082]
210 B [0.533  …[+591 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);S=np.load('/work/scratch/smooth11_T.npy');Si=np.load('/work/scratch/static_init_T.npy');Sf=np.load('/work/scratch/static_final_T.npy');mc=np.array([.27196912,-.0040119,.00456923])
def d(A,C):return np.degrees(np.linalg.norm((R.from_matrix(A[:3,:3]).inv()*R.from_matrix(C[:3,:3])).as_rotvec())),np.linalg.norm((A[:3,:3]@mc+A[:3,3])-(C[:3,:3]@mc+C[:3,3]))
for a,b in [(Si,S[261]),(Si,B[261]),(S[641],Sf),(B[641],Sf)]:print(d(a,b))
PY"
```
<details><summary>output</summary>

```
(3.4803171641121886, 0.054317849538978474)
(3.4803171641121886, 0.05431744160074104)
(2.5204172852875257, 0.01915957287960692)
(2.689730362843726, 0.015768798943315305)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_init_transition.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
def interpC(C,p):
 out=np.eye(4);R=Rot.from_matrix(C[:3,:3]);out[:3,:3]=Rot.from_rotvec(R.as_rotvec()*p).as_matrix();out[:3,3]=C[:3,3]*p;return out
def score(T,i):
 o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();iou=inter/uni;fp=(m&~o&~p).sum()/m.sum();fn=(o&~m).sum()/o.sum();return iou-.35*fp-.15*fn
for end in [240,250,260,270]:
 C0=corr(Si,B[end])
 for L in [1,2,4,6,8,10,15,20,30]:
  T=B.copy();T[:end+1]=Si
  for i in range(end+1,min(end+1+L,794)):
   lam=1-(i-end)/L;T[i]=interpC(C0,lam)@B[i]
  s=[score(T[i],i) for i in range(220,321)]
  print('end',end,'L',L,'avg',round(np.mean(s),4),'220-260',round(np.mean(s[:41]),4),'261-280',round(np.mean(s[41:61]),4),'281-320',round(np.mean(s[61:]),4))
PY
python /work/scratch/eval_init_transition.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
end 240 L 1 avg 0.4804 220-260 0.4573 261-280 0.4264 281-320 0.5311
end 240 L 2 avg 0.4806 220-260 0.4578 261-280 0.4264 281-320 0.5311
end 240 L 4 avg 0.4809 220-260 0.4586 261-280 0.4264 281-320 0.5311
end 240 L 6 avg 0.4813 220-260 0.4595 261-280 0.4264 281-320 0.5311
end 240 L 8 avg 0.4816 220-260 0.4603 261-280 0.4264 281-320 0.5311
end 240 L 10 avg 0.4821 220-260 0.4614 261-280 0.4264 281-320 0.5311
end 240 L 15 avg 0.4826 220-260 0.4628 261-280 0.4264 281-320 0.5311
end 240 L 20 avg 0.4835 220-260 0.4649 261-280 0.4264 281-320 0.5311
end 240 L 30 avg 0.4847 220-260 0.4668 261-280 0.4285 281-320 0.5311
end 250 L 1 avg 0.4841 220-260 0.4664 261-280 0.4264 281- …[+1783 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_final_transition.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
def blend(A,C,lam):
 out=np.eye(4);ra=Rot.from_matrix(A[:3,:3]);rc=Rot.from_matrix(C[:3,:3]);out[:3,:3]=(ra*Rot.from_rotvec((ra.inv()*rc).as_rotvec()*lam)).as_matrix();out[:3,3]=(1-lam)*A[:3,3]+lam*C[:3,3];return out
def score(T,i):
 o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);m=render_mask(T);inter=(m&o).sum();uni=(m|o).sum();iou=inter/uni;fp=(m&~o&~p).sum()/m.sum();fn=(o&~m).sum()/o.sum();return iou-.35*fp-.15*fn
for start in [620,630,640,642,650]:
 for L in [1,5,10,15,20,30]:
  T=B.copy();T[start+L:]=Sf
  for i in range(start,min(start+L,794)):
   lam=(i-start+1)/L;T[i]=blend(B[i],Sf,lam)
  s=[score(T[i],i) for i in range(610,701)]
  print('start',start,'L',L,'avg',round(np.mean(s),4),'610-640',round(np.mean(s[:31]),4),'641-670',round(np.mean(s[31:61]),4),'671-700',round(np.mean(s[61:]),4))
PY
python /work/scratch/eval_final_transition.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
start 620 L 1 avg 0.8699 610-640 0.8501 641-670 0.8809 671-700 0.8794
start 620 L 5 avg 0.8703 610-640 0.8513 641-670 0.8809 671-700 0.8794
start 620 L 10 avg 0.8704 610-640 0.8516 641-670 0.8809 671-700 0.8794
start 620 L 15 avg 0.871 610-640 0.8532 641-670 0.8809 671-700 0.8794
start 620 L 20 avg 0.8714 610-640 0.8545 641-670 0.8809 671-700 0.8794
start 620 L 30 avg 0.8725 610-640 0.8575 641-670 0.8812 671-700 0.8794
start 630 L 1 avg 0.8699 610-640 0.8499 641-670 0.8809 671-700 0.8794
start 630 L 5 avg 0.8705 610-640 0.852 641-670 0.8809 671-700 0.8794
start 630 L 10 avg 0.8711 610-640 0.8537 641-670 0.8809 671-700 0.8794
start 630 L 15 avg 0.8718 610-640 0.8555 …[+1438 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_candidate.py <<'PY'
exec(open('/work/scratch/eval_corrections.py').read().split('# metrics')[0])
import cv2
from scipy.signal import savgol_filter
# smoothed baseline on moving interval
S=B.copy();q=Rot.from_matrix(B[:,:3,:3]).as_quat()
for i in range(1,len(q)):
 if np.dot(q[i],q[i-1])<0:q[i]*=-1
qs=q.copy();ts=B[:,:3,3].copy();a,b=261,642;qs[a:b]=savgol_filter(qs[a:b],11,2,axis=0,mode='interp');qs[a:b]/=np.linalg.norm(qs[a:b],axis=1,keepdims=True);ts[a:b]=savgol_filter(ts[a:b],11,2,axis=0,mode='interp');S[:,:3,:3]=Rot.from_quat(qs).as_matrix();S[:,:3,3]=ts
def corr(A,C): return A@inv(C)
def interpC(C,p):
 out=np.eye(4);R=Rot.from_matrix(C[:3,:3]);out[:3,:3]=Rot.from_rotvec(R.as_rotvec()*p).as_matrix();out[:3,3]=C[:3,3]*p;return out
def blend(A,C,lam):
 out=np.eye(4);ra=Rot.from_matrix(A[:3,:3]);rc=Rot.from_matrix(C[:3,:3]);out[:3,:3]=(ra*Rot.from_rotvec((ra.inv()*rc).as_rotvec()*lam)).as_matrix();out[:3,3]=(1-lam)*A[:3,3]+lam*C[:3,3];return out
T=S.copy();T[:261]=Si
C0=corr(Si,S[260])
for i in range(261,291):
 lam=(290-i)/30;T[i]=interpC(C0,lam)@S[i]
for i in range(620,650):
 lam=(i-619)/30;T[i]=blend(S[i],Sf,lam)
T[650:]=Sf
np.save('/work/scratch/candidate_T.npy',T)
# metrics by ranges compared baseline
ranges=[(0,260),(261,290),(291,619),(620,649),(650,793),(0,793)]
for a,b in ranges:
 vals=[]
 for i in range(a,b+1):
  o=cv2.resize(Oall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(Pall[i].a …[+777 chars]
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
0 260 BASE [0.6294 0.1036 0.1402 0.5721] RES [0.6833 0.0938 0.0825 0.6381]
261 290 BASE [0.5033 0.1268 0.153  0.436 ] RES [0.5229 0.1381 0.1192 0.4567]
291 619 BASE [0.7528 0.1128 0.1141 0.6962] RES [0.7538 0.1122 0.1134 0.6975]
620 649 BASE [0.8853 0.0394 0.06   0.8626] RES [0.8931 0.0376 0.0544 0.8718]
650 793 BASE [0.7442 0.0473 0.0823 0.7153] RES [0.7638 0.0397 0.0638 0.7403]
0 793 BASE [0.7062 0.0956 0.1163 0.6553] RES [0.729  0.0912 0.0922 0.6832]
jump 260 base 0.1664417542789499 0.00037191739535869146 res 0.0 0.0
jump 261 base 0.3545311536434523 0.005344760173644146 res 0.37118035745585026 0.00493754614714274
jump 290 base 0.8262406878329466 0.00751026088682 …[+535 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/view_candidate.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split(\"z=np.load\")[0])
import cv2
T=np.load('/work/scratch/candidate_T.npy');cap=cv2.VideoCapture('/work/inputs/video.mp4');idxs=[0,100,200,260,280,300,350,400,450,500,550,600,620,650,700,750,793];ims=[]
for i in idxs:
 m,d,rgb=render(T[i]);m=m[::-1];rgb=rgb[::-1];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=im[:,:,::-1];ov=im.copy();ov[m]=(.30*im[m]+.70*rgb[m]).astype(np.uint8);cnts,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cnts,-1,(255,0,0),2);ov=cv2.resize(cv2.cvtColor(ov,cv2.COLOR_RGB2BGR),(512,384));cv2.putText(ov,str(i),(10,35),0,1,(0,0,255),2);ims.append(ov)
while len(ims)%3:ims.append(np.zeros_like(ims[0]))
out=np.vstack([np.hstack(ims[i:i+3]) for i in range(0,len(ims),3)]);cv2.imwrite('/work/scratch/vis/candidate_overlay.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92]);print(out.shape)
PY
python /work/scratch/view_candidate.py"
```
<details><summary>output</summary>

```
(2304, 1536, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/lam=(290-i)\\/30/lam=(290-i)\\/29/' /work/scratch/build_candidate.py > /work/scratch/build_candidate29.py
python /work/scratch/build_candidate29.py"
```
<details><summary>output</summary>

```
C0 ang 10.05776404597425 t [ 0.44918345  0.22010702 -0.17812045]
C1 ang 2.861112688830598 t [0.0393893  0.04501044 0.03420413]
0 260 BASE [0.6294 0.1036 0.1402 0.5721] RES [0.6833 0.0938 0.0825 0.6381]
261 290 BASE [0.5033 0.1268 0.153  0.436 ] RES [0.5232 0.1389 0.1182 0.4568]
291 619 BASE [0.7528 0.1128 0.1141 0.6962] RES [0.7538 0.1122 0.1134 0.6975]
620 649 BASE [0.8853 0.0394 0.06   0.8626] RES [0.8931 0.0376 0.0544 0.8718]
650 793 BASE [0.7442 0.0473 0.0823 0.7153] RES [0.7638 0.0397 0.0638 0.7403]
0 793 BASE [0.7062 0.0956 0.1163 0.6553] RES [0.729  0.0912 0.0922 0.6832]
jump 260 base 0.1664417542789499 0.00037191739535869146 res 0.0 0.0
jump 261 base 0.3545311536434523 0.005344760173644146 res 0.2951899457529771 0.0036669334076381566
jump 290 base 0.8262406878329466 0.0075102608868 …[+538 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a phase-aware 6-DoF world trajectory for the desk episode.

The method uses the provided FoundationPose/CoCoNet reconstruction as the
moving-frame proposal, but replaces its per-frame static-phase drift with a
pooled render-and-compare estimate and smooths the carried phase.  It uses no
network at solve time; the exact mesh, camera, masks and baseline trajectory
are the only episode data read.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Phase constants.
#
# These boundaries were verified from the video and object-mask contours:
# the desk is visibly stationary until the first controlled tilt just after
# frame 260.  It reaches the floor around frame 620 and is fully settled by
# frame 650.  A per-frame mask score against the two pooled static poses also
# changes preference at these boundaries.  Keeping explicit constants makes
# the decode deterministic and reproducible.
# ---------------------------------------------------------------------------
INITIAL_STATIC_END = 260       # frames 0..260 use one pooled world pose
INITIAL_RAMP_START = 261       # blend out the static-pose correct …[+11759 chars]
```
<details><summary>output</summary>

```
#!/usr/bin/env python3
"""Recover a phase-aware 6-DoF world trajectory for the desk episode.

The method uses the provided FoundationPose/CoCoNet reconstruction as the
moving-frame proposal, but replaces its per-frame static-phase drift with a
pooled render-and-compare estimate and smooths the carried phase.  It uses no
network at solve time; the exact mesh, camera, masks and baseline trajectory
are the only episode data read.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "/usr/bin/time -f 'WALL %e sec, MAXRSS %M KB' python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
wrote /work/outputs/object_poses_world.npz with shape (794, 4, 4), frames=794, elapsed=6.0s

real	0m8.749s
user	0m10.787s
sys	0m2.953s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
A=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/scratch/candidate_T.npy')
print(A.shape,A.dtype,np.isfinite(A).all(),np.max(abs(A-B)),np.mean(abs(A-B)))
print('det range',np.linalg.det(A[:,:3,:3]).min(),np.linalg.det(A[:,:3,:3]).max())
print('bottom',np.max(abs(A[:,3,:]-np.array([0,0,0,1]))))
print('static initial std',A[:261,:3,3].std(0),'final std',A[650:,:3,3].std(0))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/validate_final.py <<'PY'
import sys,json,cv2,h5py,numpy as np,torch,trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
sys.path.insert(0,'/work/outputs');import solve
# load
inputs=Path('/work/inputs'); cam=json.load(open(inputs/'camera.json'));K=np.array(cam['K'],float);Twc=np.array(cam['T_world_cam'],float);F=794
z=np.load(inputs/'reconstruction.npz');B=z['T_world_mesh'].astype(float);T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float);KP=z['human_keypoints_world'].astype(float);CP=z['contact_prob_right_left'].astype(float)
rend=solve.MaskRenderer(inputs/'object_mesh.glb',K,Twc,cam['width'],cam['height'],.5);W,H=rend.width,rend.height
mesh=trimesh.load(str(inputs/'object_mesh.glb'),force='scene',process=False).to_geometry();V=mesh.vertices[::20]
phases=[('rest_initial',0,260),('grasp_onset',261,290),('lowering',291,619),('settle',620,649),('rest_final',650,793),('overall',0,793)]
with h5py.File(inputs/'masks.h5','r') as fm, h5py.File('/pipeline/depth/moge2/aligned.h5','r') as fd:
 O=[None]*F;P=[None]*F
 for i in range(F):
  O[i]=solve.resize_mask(fm['object'][i],W,H);P[i]=solve.resize_mask(fm['person'][i],W,H)
 def boundary(m):return cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
 rows={}
 for name,a,b in phases[:-1]:
  metrics={}
  for lab,X in [('base',B),('result',T)]:
   vals=[];deps=[]
   for i in range(a …[+2990 chars]
```
<details><summary>output</summary>

```
rest_final {'base': {'iou': 0.7442111246444294, 'fp': 0.047274597017137404, 'fn': 0.08231966755239888, 'chamfer_halfpx': 2.9927812268336615, 'depth_shape_mad_m': 0.04850830137729645, 'right_hand_min_dist_m': None, 'rot_step_deg': 0.3218090835022188, 'center_accel_m': 0.012863999673251143, 'rot_spread_p95_deg': 1.0811984706404374, 'center_spread_std_m': 0.017309070876332}, 'result': {'iou': 0.75852006340376, 'fp': 0.039755460588793916, 'fn': 0.07061203446563244, 'chamfer_halfpx': 2.8641162754760847, 'depth_shape_mad_m': 0.04975132271647453, 'right_hand_min_dist_m': None, 'rot_step_deg': 0.0, 'center_accel_m': 0.0, 'rot_spread_p95_deg': 4.089873765583611e-14, 'center_spread_std_m': 4.336808689942018e-15}}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/scratch/validation.json; echo; wc -l /work/scratch/vis/validation_log.txt; cat /work/scratch/vis/validation_log.txt'
```
<details><summary>output</summary>

```
{
  "rest_initial": {
    "base": {
      "iou": 0.6293866006095113,
      "fp": 0.10362733322602044,
      "fn": 0.1401575533895579,
      "chamfer_halfpx": 3.1121365051159913,
      "depth_shape_mad_m": 0.07177107036113739,
      "right_hand_min_dist_m": 0.008648421385972153,
      "rot_step_deg": 0.2584683912358834,
      "center_accel_m": 0.007342674782835902,
      "rot_spread_p95_deg": 1.2861594856865,
      "center_spread_std_m": 0.007882769469040984
    },
    "result": {
      "iou": 0.6828506269618674,
      "fp": 0.09371401953174606,
      "fn": 0.08329542375486643,
      "chamfer_halfpx": 2.7340748362614278,
      "depth_shape_mad_m": 0.06715133786201477,
      "right_hand_min_dist_m": 0.011916312961481344,
      "rot_step_deg": 0.0011353459452036155,
      "center_accel_m": 0. …[+3914 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json,numpy as np
r=json.load(open('/work/scratch/validation.json'));lens={'rest_initial':261,'grasp_onset':30,'lowering':329,'settle':30,'rest_final':144}
for lab in ['base','result']:
 print(lab)
 for k in ['iou','fp','fn','chamfer_halfpx','depth_shape_mad_m','rot_step_deg','center_accel_m']:
  vals=[r[p][lab][k] for p in lens if r[p][lab][k] is not None];ns=[lens[p] for p in lens if r[p][lab][k] is not None]
  print(k,np.average(vals,weights=ns))
PY"
```
<details><summary>output</summary>

```
base
iou 0.7062498594652596
fp 0.09564327786683956
fn 0.11634395816988238
chamfer_halfpx 2.4404051649930794
depth_shape_mad_m 0.08160090972892284
rot_step_deg 0.3773586239131621
center_accel_m 0.012825141292998972
result
iou 0.7278654224116464
fp 0.09118890306769502
fn 0.09369829909745093
chamfer_halfpx 2.27554777332157
depth_shape_mad_m 0.08050432728992901
rot_step_deg 0.14861427163939092
center_accel_m 0.001101080976452804

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffprobe -version | head -2; git --version | head -1'
```
<details><summary>output</summary>

```
ffprobe version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2007-2021 the FFmpeg developers
built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)
git version 2.34.1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sha256sum /work/inputs/object_mesh.glb /work/inputs/video.mp4 /work/inputs/reconstruction.npz /work/inputs/masks.h5 | sed 's#/work/inputs/##'"
```
<details><summary>output</summary>

```
20bc239794adf00df9089347eef7bb72c9540ab03379d1fb3762c9d156a4fefd  object_mesh.glb
9d52402823157bb7f071cf0f9f8c36177308a7cde677be8b3ea779ad676493c9  video.mp4
757b7709beb699fc0da8a9325d6ae95e9d41d0ddc322ca42ca3230463495ec5b  reconstruction.npz
dec2a1c255f30d8db7e81c13861a9271a4c1b2ddf2504336f4507f41c1f9a2ce  masks.h5

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy,scipy,cv2,trimesh,h5py,torch,nvdiffrast
mods=[('python',sys.version.split()[0]),('numpy',numpy.__version__),('scipy',scipy.__version__),('opencv-python',cv2.__version__),('trimesh',trimesh.__version__),('h5py',h5py.__version__),('torch',torch.__version__)]
for x in mods:print(*x)
import importlib.metadata as md
for p in ['nvdiffrast','opencv-python','opencv-python-headless']:
 try:print(p,md.version(p))
 except Exception as e:print(p,'not installed metadata')
PY"
```
<details><summary>output</summary>

```
python 3.11.10
numpy 1.26.3
scipy 1.16.3
opencv-python 4.11.0
trimesh 5.1.1
h5py 3.16.0
torch 2.5.1+cu124
nvdiffrast 0.4.0
opencv-python 4.11.0.86
opencv-python-headless not installed metadata

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import matplotlib,PIL
print('matplotlib',matplotlib.__version__)
print('Pillow',PIL.__version__)
PY"
```
<details><summary>output</summary>

```
matplotlib 3.11.2
Pillow 10.2.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# White-desk trajectory recovery

## Result

The final trajectory is in "'`object_poses_world.npz` under `T_world_mesh`
(794 x 4 x 4, float64). It was produced by the required offline command:

```bash
python /work/outputs/solve.py --inputs /work/inputs '"\\
  --out /work/outputs/object_poses_world.npz
"'```

The solve took **8.7 s wall time** on the allocated RTX 6000 Ada GPU
(6.0 s measured inside the script). Peak observed GPU use was far below the
9 GB limit.

## Method

The released per-frame trajectory is a useful moving-object proposal, but its
static phases contain obvious tracker drift. I kept its frame-specific view
evidence during manipulation and replaced the static portions with one pooled
pose per phase.

1. **Static-pose pooling.** For a candidate world pose, the exact textured
   mesh was rasterized with nvdiffrast at 768 x 576. The objective pooled four
   fully visible frames for each static pose and combined:
   - object-mask IoU;
   - rendered pixels outside `object | person` (false positives);
   - visible object pixels missed by the render (false negatives);
   - symmetric boundary chamfer distance.
   A deterministic 6-DoF coordinate search started from baseline frame 0 for
   the initial pose and baseline frame 650 for the final pose. The recovered
   perturbations were about `(-1, 0, -3) cm` and
   `(+0.4, -2.25, +0.75) deg` for the initial pose, and
   `(0, 0, -2) cm` and `(+3, 0, 0) deg` for the …[+7982 chars]
```
<details><summary>output</summary>

```
total 132K
-rw-rw-r-- 1 hanzhou hanzhou 7.8K Oct 20 21:07 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.3K Oct 20 21:07 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:04 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 100K Oct 20 21:04 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  13K Oct 20 21:04 solve.py
---
# White-desk trajectory recovery

## Result

The final trajectory is in `object_poses_world.npz` under `T_world_mesh`
(794 x 4 x 4, float64). It was produced by the required offline command:

```bash
python /work/outputs/solve.py --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The solve took **8.7 s wall time** on the allocated RTX 6000 Ada GPU
(6.0 s measured inside the script). Peak observed GPU use was far below the
9 GB limit.

## Method

The released p …[+1327 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
p=Path('/work/outputs/REPORT.md')
s=p.read_text()
s=s.replace('(6.0 s measured inside the script). Peak observed GPU use was far below the\\n9 GB limit.', '(6.0 s measured inside the script). The GPU footprint is one mesh and\\nraster buffers no larger than 1536 x 1152, comfortably below the 9 GB limit.')
p.write_text(s)
cache=Path('/work/outputs/__pycache__')
if cache.exists():
    for f in cache.iterdir():
        f.unlink()
    cache.rmdir()
print('\\n'.join(x.name for x in Path('/work/outputs').iterdir()))
PY"
```
<details><summary>output</summary>

```
solve.py
object_poses_world.npz
REPORT.md
TOOLS.md

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sha256sum /work/outputs/object_poses_world.npz > /work/scratch/final_hash_before.txt
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
sha256sum /work/outputs/object_poses_world.npz > /work/scratch/final_hash_after.txt
cat /work/scratch/final_hash_before.txt /work/scratch/final_hash_after.txt
python - <<'PY'
import numpy as np, json
p='/work/outputs/object_poses_world.npz';z=np.load(p)
print('keys',z.files)
T=z['T_world_mesh']
print('shape',T.shape,'dtype',T.dtype,'finite',np.isfinite(T).all())
print('det',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
print('bottom_maxerr',np.max(np.abs(T[:,3,:]-[0,0,0,1])))
print('initial_constant',np.max(np.abs(T[:261]-T[0])),'final_constant',np.max(np.abs(T[650:]-T[650])))
PY
wc -l /work/outputs/solve.py /work/outputs/REPORT.md /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
wrote /work/outputs/object_poses_world.npz with shape (794, 4, 4), frames=794, elapsed=5.6s
5631568f0ff696101b4362620ad2a2115429cb98717d78cc267fdafa14003ce4  /work/outputs/object_poses_world.npz
5631568f0ff696101b4362620ad2a2115429cb98717d78cc267fdafa14003ce4  /work/outputs/object_poses_world.npz
keys ['T_world_mesh']
shape (794, 4, 4) dtype float64 finite True
det 0.9999999999999992 1.0000000000000009
bottom_maxerr 0.0
initial_constant 0.0 final_constant 0.0
  326 /work/outputs/solve.py
  148 /work/outputs/REPORT.md
   23 /work/outputs/TOOLS.md
  497 total

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
p=Path('/work/outputs/REPORT.md');s=p.read_text();s=s.replace('The solve took **8.7 s wall time** on the allocated RTX 6000 Ada GPU\\n(6.0 s measured inside the script).','The solve took **about 8 s wall time** on the allocated RTX 6000 Ada GPU\\n(5.6 s measured inside the final script run).')
p.write_text(s)
PY
grep -n 'solve took\\|GPU footprint' /work/outputs/REPORT.md; tail -20 /work/outputs/REPORT.md; echo '---TOOLS'; cat /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
13:The solve took **about 8 s wall time** on the allocated RTX 6000 Ada GPU
14:(5.6 s measured inside the final script run). The GPU footprint is one mesh and

The result improves mask overlap and boundary agreement in every phase, and
removes nearly all static-phase drift. The moving proposal is preserved where
occlusion makes the silhouette weak, rather than being overfit to an
incomplete object mask. Right-hand distance remains at approximately 4-7 mm
during the carrying interval, consistent with the reconstructed human contact.
Small increases in a few metrics (onset FP, final depth MAD, and release-frame
hand distance) were accepted because the object is occluded or no longer in
contact at those frames and the corresponding silhouette/temporal evidence is
strongly better.

## Reproduc …[+1794 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/outputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort; sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py /work/outputs/REPORT.md /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
REPORT.md 8059 bytes
TOOLS.md 1304 bytes
object_poses_world.npz 101910 bytes
solve.py 13006 bytes
5631568f0ff696101b4362620ad2a2115429cb98717d78cc267fdafa14003ce4  /work/outputs/object_poses_world.npz
2c0d353c7c84d96521745e83e9f2b02336be0e0014d8dec8bdcfeb0a3e22b41d  /work/outputs/solve.py
0bb6537541594a09053bb0673c0230bf6265e11f70aa2d75ca4965016d0cef6f  /work/outputs/REPORT.md
7122780900fa042b37d7bd13ba55c1b72bda01af9da95cbfa2cfbca3f1188835  /work/outputs/TOOLS.md

```
</details>

**Agent:** Completed the trajectory recovery task.

Outputs created:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(794, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — method, phase segmentation, validation table, confidence
- `/work/outputs/TOOLS.md` — tool and package inventory

Summary of improvements over baseline:

- Pooled constant poses for the initial and final static phases
- Smooth temporal correction through grasp/onset and settling transitions
- Savitzky-Golay filtering during manipulation
- Mask IoU improved overall from `0.7062` to `0.7279`
- Boundary chamfer improved from `2.440` to `2.276` half-resolution pixels
- Static-phase drift reduced to effectively zero
- Mean center acceleration reduced from `0.01283` to `0.00110 m/frame²`

The required command was rerun successfully and produced an identical output hash:

```text
5631568f0ff696101b4362620ad2a2115429cb98717d78cc267fdafa14003ce4
```
