# T1_ds_ep011

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T17:12:49Z

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
/usr/bin/bash -c 'cat inputs/task.txt; cat inputs/episode.json; cat inputs/camera.json; cat inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Pick up the book from the table and place it on the floor.
{
 "episode_index": 11,
 "object": "light_blue_book",
 "frames": 901
}{
 "K": [
  [
   819.999755859375,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.999755859375,
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
   825.64306640625,
   0.0,
   768.0
  ],
  [
   0.0,
   825.6430053710938,
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
   0.998745270406387,
   0.017199279151546447,
   0.047032644381747896,
   0.013660205963114007
  ],
  [
   -0.017874693614864247,
   0.9997425244606121,
   0.013977843656909903,
   0.05586822317571786
  ],
  [
   -0.046780125791275,
   -0.01480099935103928,
   0.9987955497744084,
   -0.01974717119499238
  ],
  [
   0.0,
   0.0,
   0.0,
   1.0
  ]
 ] …[+546 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find inputs -maxdepth 1 -type f -printf '%f %s bytes\\n'; find pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100; find code/reconstruction -maxdepth 2 -type f -printf '%p\\n' | head -100"
```
<details><summary>output</summary>

```
video.mp4 33226981 bytes
object_mesh.glb 10541188 bytes
ground_plane.json 395 bytes
task.txt 59 bytes
episode.json 70 bytes
camera.json 822 bytes
reconstruction.npz 45897895 bytes
frame0_reference_human.npz 193877 bytes
masks.h5 6116444 bytes
find: ‘pipeline’: No such file or directory
find: ‘code/reconstruction’: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json
r=np.load('inputs/reconstruction.npz')
for k in r.files:
 a=r[k]; print(k,a.shape,a.dtype, np.nanmin(a) if np.issubdtype(a.dtype,np.number) else '')
with h5py.File('inputs/masks.h5') as f:
 print(list(f.keys()), {k:(f[k].shape,f[k].dtype) for k in f.keys()})
PY"
```
<details><summary>output</summary>

```
T_world_mesh (901, 4, 4) float32 -0.99994445
T_cam_mesh_foundationpose (901, 4, 4) float32 -1.2280359
T_cam_mesh_coconet (901, 4, 4) float32 -0.99990886
human_vertices_world (901, 4610, 3) float32 -0.47908786
human_keypoints_world (901, 70, 3) float32 -0.46023417
contact_prob_right_left (901, 2) float32 3.7125962e-11
['object', 'person'] {'object': ((901, 1152, 1536), dtype('bool')), 'person': ((901, 1152, 1536), dtype('bool'))}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -la /work; ls -la /workspace 2>/dev/null | head; ls -la /pipeline 2>/dev/null | head; ls -la /code 2>/dev/null | head'
```
<details><summary>output</summary>

```
total 40
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:06 .
drwxr-xr-x 1 hanzhou hanzhou   200 Oct 20 21:12 ..
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:06 .codex_home
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:06 .net
-rw-rw-r-- 1 hanzhou hanzhou 10592 Oct 20 21:06 AGENT_TASK.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 17:48 inputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:06 logs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:06 outputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:06 scratch
total 0
drwxrwxr-x 4 hanzhou hanzhou 0 Oct  4 20:18 third_party
drwxr-xr-x 3 hanzhou hanzhou 0 Oct  4 20:03 v2d_cari4d
drwxrwxr-x 4 hanzhou hanzhou 0 Oct  4 20:03 v2d_common
drwxr-xr-x 3 hanzhou hanzhou 0 Oct  4 20:03 v2d_foundation_pose
drwxr-xr-x 3 hanzhou hanzhou 0 Oct  4 20:03 v2d_mesh
drwx …[+1243 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, numpy as np
cap=cv2.VideoCapture('inputs/video.mp4')
print(cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
PY"
```
<details><summary>output</summary>

```
901.0 30.0 1536.0 1152.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; python - <<'PY'
import trimesh
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m), len(m.geometry))
for k,g in m.geometry.items():
 print(k,g.vertices.shape,g.faces.shape, g.visual)
 print('bounds',g.bounds,'extents',g.extents,'center_mass',g.center_mass)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'Trimesh' object has no attribute 'geometry'. Did you mean: 'symmetry'?

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 4 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1975
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2133
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 1115675348
/pipeline/depth/moge2/depth_report.json 1810
/pipeline/depth/moge2/intrinsics.pkl 95913
/pipeline/depth/moge2/raw.npy 3188588672
/pipeline/export/episode_000011/edex 561
/pipeline/export/episode_000011/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000011/human_masks/front_stereo_camera_left.h5 20964756
/pipeline/export/e …[+7548 chars]
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
    "mtime_ns": 1791294297279460610,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000011/inference/episode_000011_step200000_before_after.mp4",
    "size": 72637244
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+5024 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace/v2d_cari4d -maxdepth 3 -type f -printf '%p\\n' | head -100; find /code/reconstruction -maxdepth 2 -type f -printf '%p\\n' | head -100"
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
 …[+1543 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
x=trimesh.load('inputs/object_mesh.glb',force='scene',process=False)
print(type(x),x)
y=x.to_geometry()
print(type(y))
if hasattr(y,'geometry'):
 print(y.geometry)
else:
 print(y.vertices.shape,y.faces.shape,y.bounds,y.extents,y.center_mass,type(y.visual))
 print(y.visual)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.scene.scene.Scene'> <trimesh.Scene(len(geometry)=1)>
<class 'trimesh.base.Trimesh'>
(271275, 3) (90566, 3) [[-0.09479323 -0.02947056 -0.13077636]
 [ 0.09639004  0.02936917  0.1300355 ]] [0.19118327 0.05883973 0.26081187] [ 3.47009689e-03  2.67739245e-04 -2.28867719e-05] <class 'trimesh.visual.texture.TextureVisuals'>
<trimesh.visual.texture.TextureVisuals object at 0x7d6196d900d0>

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
      "mtime_ns": 1791218833264748482,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000011/foundationpose/episode_000011.pkl",
      "size": 256620
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000011/export/episode_000011",
      "--depth-root",
      "/home/hanzhou/Projects/v2d/run/out/episode_000011/depth/moge2/aligned.h5",
      "--out-file",
      "/home/hanzhou/Projects/v2d/run/out/episode_000011/foundationpose/episode_000011.pkl",
      "--camera-id",
      "0",
      "--iteration",
      "5",
      "--max-attempts",
      "5",
 …[+8461 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/depth/moge2/depth_report.json; cat /pipeline/export/episode_000011/wild_export.json'
```
<details><summary>output</summary>

```
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 9.283381462097168,
    "min": 1.5831542015075684
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 140.15602144016884,
  "firstFrameIntrinsicsPx": [
    [
      825.64306640625,
      0.0,
      768.0
    ],
    [
      0.0,
      825.6430053710938,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "focalXPx": {
    "max": 835.900390625,
    "mean": 823.6602783203125,
    "min": 814.3628540039062
  },
  "focalYPx": {
    "max": 835.9003295898438,
    "mean": 823.6602783203125,
    "min": 814.3628540039062
  },
  "fps": 30.0,
  "frames": 901,
  "framesPerSecond": 6.428550059724888,
  "height": 1152,
  "intrinsicsFile": {
    …[+3705 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,pickle
for p in ['/pipeline/foundationpose/episode_000011.pkl','/pipeline/inference/refined.pth','/pipeline/inference/coconet.pth','/pipeline/inference/refinement_checkpoint.pth']:
 print('\\n',p)
 try:
  if p.endswith('.pkl'):
   x=pickle.load(open(p,'rb')); print(type(x));
   if isinstance(x,dict): print(x.keys())
  else:
   x=torch.load(p,map_location='cpu',weights_only=False); print(type(x));
   if isinstance(x,dict): print(x.keys())
 except Exception as e: print(type(e),e)
PY"
```
<details><summary>output</summary>

```

 /pipeline/foundationpose/episode_000011.pkl
<class 'dict'>
dict_keys(['frames', 'camera_id', 'fp_poses_cam', 'fp_candidate_count', 'fp_selected_candidate_index', 'fp_selected_iou', 'fp_selected_visibility', 'fp_retry_count', 'fp_reliable', 'fp_oracle_rotation_error_deg', 'fp_translation_threshold_m', 'fp_selected_translation_distance_m', 'gt_object_visibility_ratio', 'gt_object_visibility_valid', 'gt_object_rendered_pixels', 'gt_object_visible_pixels', 'obj_pose_world', 'obj_rot_init', 'obj_t_init', 'metadata', 'fp_tracking_state', 'fp_confirmed_pose_source', 'fp_pending_pose_source', 'fp_emitted_pose_source', 'fp_gap_length', 'fp_observed_visible_pixels', 'fp_rendered_visible_pixels', 'fp_rendered_visibility', 'fp_full_image_iou', 'fp_tolerant_precision', 'fp_tolerant_recall', 'fp_centr …[+976 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch, numpy as np
for p in ['/pipeline/inference/refined.pth','/pipeline/inference/coconet.pth']:
 x=torch.load(p,map_location='cpu',weights_only=False)
 print('\\n###',p)
 def rec(v,pre='',d=0):
  if d>3:return
  if isinstance(v,dict):
   for k,z in v.items():
    if torch.is_tensor(z): print(pre+k,tuple(z.shape),z.dtype)
    elif isinstance(z,np.ndarray): print(pre+k,z.shape,z.dtype)
    elif isinstance(z,(str,int,float,bool,type(None))): print(pre+k,repr(z)[:200])
    else:
     print(pre+k,type(z))
     rec(z,pre+k+'.',d+1)
  elif isinstance(v,(list,tuple)):
   print(pre,type(v),len(v))
   if v: rec(v[0],pre+'[0].',d+1)
 rec(x)
PY"
```
<details><summary>output</summary>

```

### /pipeline/inference/refined.pth
schema 'cari4d.mhr_wild_inference.v1'
frames <class 'list'>
frames. <class 'list'> 901
frame_meta <class 'list'>
frame_meta. <class 'list'> 901
frame_meta.[0].frame '000000'
frame_meta.[0].src_frame 0
frame_meta.[0].kid 0
kid 0
checkpoint <class 'dict'>
checkpoint.path '/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth'
checkpoint.step 200000
mhr_inference <class 'dict'>
mhr_inference.schema 'cari4d.mhr_inference_provenance.v1'
mhr_inference.checkpoint <class 'dict'>
mhr_inference.checkpoint.path '/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth'
mhr_inference.checkpoint.step 200000
mhr_inference.wandbRun <class 'dict'>
mhr_inference.wandbRun.entity 'nvr-lpr'
mhr_inference.wandbRun.project 'cari4d-m …[+46794 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,pprint
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
print('frames0:',type(x['frames'][0])); pprint.pp(x['frames'][0],depth=3)
print('postopt:',type(x['postopt'])); pprint.pp(x['postopt'],depth=3)
print('in.pose abs at starts:',x['in']['pose_abs'][[0,100,300,600,900]])
print('pr abs ends:',x['pr']['pose_abs'][[0,100,300,600,900]])
PY"
```
<details><summary>output</summary>

```
frames0: <class 'str'>
'000000'
postopt: <class 'dict'>
{'mode': 'smplh_parity',
 'checkpoint_schema': 'cari4d.mhr_postopt_smplh_parity.v10',
 'batch_sampling': 'shuffled_contiguous_window_epoch_v1',
 'resolved_batch_size': 384,
 'object_mesh_loader_revision': 'trimesh-scene-node-transforms-v1',
 'frame_indices': [0,
                   1,
                   2,
                   3,
                   4,
                   5,
                   6,
                   7,
                   8,
                   9,
                   10,
                   11,
                   12,
                   13,
                   14,
                   15,
                   16,
                   17,
                   18,
                   19,
                   20,
                   21,
        …[+56102 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, torch, json
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh']; F=len(T)
cam=json.load(open('inputs/camera.json')); W=np.array(cam['T_world_cam'])
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for nm,P in [('in',x['in']['pose_abs'].numpy()),('pr',x['pr']['pose_abs'].numpy()),('init',x['pr_initial']['pose_abs'].numpy())]:
 Q=np.einsum('ij,fjk->fik',W,P)
 print(nm,'trans',np.linalg.norm(Q[:,:3,3]-T[:,:3,3],axis=1).mean(),np.linalg.norm(Q[:,:3,3]-T[:,:3,3],axis=1).max())
 # angle
 d=np.einsum('fij,fij->fi',Q[:,:3,:3],T[:,:3,:3]); ang=np.degrees(np.arccos(np.clip((d.sum(1)-1)/2,-1,1)))
 print('ang mean max med',ang.mean(),ang.max(),np.median(ang))
print('baseline trans steps',np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1).mean(),np.max(np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)))
# rotation steps
R=T[:,:3,:3]; rel=np.einsum('fji,fjk->fik',R[:-1],R[1:]); ang=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
print('rotsteps pct',np.percentile(ang,[0,25,50,75,90,95,99,100]),'frames>10',np.where(ang>10)[0][:50], 'count', (ang>10).sum())
print('trans pct',np.percentile(np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1),[0,25,50,75,90,95,99,100]))
# positions every 30
for i in range(0,901,30): print(i, np.round(T[i,:3,3],3), np.round(T[i,:3,:3],2).tolist())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'numpy'. Did you mean: 'dump'?

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, torch, json
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh']; F=len(T)
cam=json.load(open('inputs/camera.json')); W=np.array(cam['T_world_cam'])
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
def arr(z):return z.numpy() if torch.is_tensor(z) else np.asarray(z)
for nm,P in [('in',arr(x['in']['pose_abs'])),('pr',arr(x['pr']['pose_abs'])),('init',arr(x['pr_initial']['pose_abs']))]:
 Q=np.einsum('ij,fjk->fik',W,P)
 print(nm,'trans mean/max',np.linalg.norm(Q[:,:3,3]-T[:,:3,3],axis=1).mean(),np.linalg.norm(Q[:,:3,3]-T[:,:3,3],axis=1).max())
 d=np.einsum('fij,fij->fi',Q[:,:3,:3],T[:,:3,:3]); ang=np.degrees(np.arccos(np.clip((d.sum(1)-1)/2,-1,1)))
 print('ang mean max med',ang.mean(),ang.max(),np.median(ang))
R=T[:,:3,:3]; rel=np.einsum('fji,fjk->fik',R[:-1],R[1:]); ang=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
ts=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
print('rotsteps pct',np.percentile(ang,[0,25,50,75,90,95,99,100]),'frames>10',np.where(ang>10)[0][:50], 'count', (ang>10).sum())
print('trans pct',np.percentile(ts,[0,25,50,75,90,95,99,100]))
for i in range(0,901,30): print(i, np.round(T[i,:3,3],3), np.round(T[i,:3,:3],2).tolist())
PY"
```
<details><summary>output</summary>

```
in trans mean/max 0.7425755002104552 1.5626107242164462
ang mean max med 147.66823118719856 179.95114631857865 155.85498913040377
pr trans mean/max 0.2997478357909736 0.6832886470679265
ang mean max med 144.02564438133612 180.0 152.14969640071635
init trans mean/max 0.3107145320489043 0.7058849613500012
ang mean max med 144.02564438133612 180.0 152.14969640071635
rotsteps pct [ 0.          0.3863695   1.25460595  2.96292931  7.22867351 11.76320066
 25.72012737 34.98636627] frames>10 [108 116 128 135 136 137 141 145 146 147 149 150 155 156 158 159 323 327
 328 330 351 361 362 372 489 491 492 569 572 574 579 580 581 584 585 586
 587 591 592 606 607 811 812 827 828 869 871 872 875 876] count 58
trans pct [0.         0.00321277 0.011431   0.02262923 0.0377274  0.05582531
 0.09457603 0.1675868  …[+6498 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"T_world_mesh\\|depth correction\\|depth_correction\\|obj_pose_world\" -n /workspace/v2d_cari4d /code/reconstruction/modules 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py:360:    expected_shapes = {"fp_poses_cam": (frame_count, 1, 4, 4), "obj_pose_world": (frame_count, 4, 4), "obj_rot_init": (frame_count, 3, 3), "obj_t_init": (frame_count, 3)}
/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py:391:    validate_rigid_transforms(np.asarray(result["obj_pose_world"]), f"cached FoundationPose world poses {path}")
/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py:882:        "obj_pose_world": poses_world_arr,
/workspace/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py:553:    object_poses = np.asarray(fp["obj_pose_world"], dtype=np.float32)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py:553:    object_poses = np. …[+1532 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep -R "pose_abs" -n /workspace/v2d_cari4d/lib/cari4d | head -100'
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:520:        total = len(bundle["pr"]["pose_abs"])
/workspace/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:555:        self.pose_init = _as_tensor(self.pr["pose_abs"], device=self.device, dtype=self.dtype)
/workspace/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:556:        self.pose_ref = _as_tensor(self.ref.get("pose_abs", self.pr["pose_abs"]), device=self.device, dtype=self.dtype)
/workspace/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:803:        result["pr"]["pose_abs"] = _to_numpy(pose).astype(np.float32)
/workspace/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:804:        result["pr"]["pose_abs_postopt"] = _to_numpy(pose).astype(np.float32)
/workspace/ …[+1549 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
po=x['postopt']
for k,v in po.items():
 if isinstance(v,(str,int,float,bool,list,tuple,dict)):
  if isinstance(v,dict): print(k,'dict keys',list(v.keys())[:30])
  elif isinstance(v,list): print(k,'list',len(v),str(v[:3])[:200])
  else: print(k,repr(v)[:300])
 elif torch.is_tensor(v): print(k,tuple(v.shape),v.dtype)
 elif isinstance(v,np.ndarray): print(k,v.shape,v.dtype)
PY"
```
<details><summary>output</summary>

```
mode 'smplh_parity'
checkpoint_schema 'cari4d.mhr_postopt_smplh_parity.v10'
batch_sampling 'shuffled_contiguous_window_epoch_v1'
resolved_batch_size 384
object_mesh_loader_revision 'trimesh-scene-node-transforms-v1'
frame_indices list 901 [0, 1, 2]
config dict keys ['num_steps', 'batch_size', 'device', 'frame_start', 'frame_limit', 'lr', 'warmup_steps', 'schedule_steps', 'object_surface_samples', 'penetration_surface_samples', 'penetration_point_reduction', 'penetration_human_mesh_mode', 'penetration_collision_proxy_path', 'penetration_bbox_rejection', 'penetration_bbox_margin_m', 'hand_surface_mode', 'hand_surface_spec_path', 'contact_activation_distance_m', 'w_contact', 'w_silhouette', 'w_penetration', 'w_human_pose_prior', 'human_pose_prior_beta', 'w_temporal', 'w_object_translation_pri …[+1929 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '500,710p' /workspace/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py; sed -n '300,420p' /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py; sed -n '760,910p' /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py"
```
<details><summary>output</summary>

```
        if self.profiler is not None:
            self.profiler.record("persistent_input_cache_write", started)
        return path


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def write_wild_inference_bundle(bundle: Mapping[str, Any], output: str | Path) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(bundle, temporary, pickle_protocol=4)
    os.replace( …[+36841 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"reconstruction.npz\\|T_world_mesh\" -n /workspace /code 2>/dev/null | head -100"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('inputs/reconstruction.npz')['T_world_mesh']
print(T[0]); print(T[1]); print(T.dtype)
for i in range(20): print(i,np.round(T[i,:3,3],4),np.linalg.norm(np.diff(T[max(0,i-1):i+2,:3,3],axis=0),axis=1) if i else '')
print('unique translation count each dim', [len(np.unique(T[:,j,3])) for j in range(3)])
PY"
```
<details><summary>output</summary>

```
[[-0.846754   -0.02910021 -0.5311883   0.37519768]
 [ 0.06859153 -0.9961404  -0.05476816  0.28386205]
 [-0.5275445  -0.0828102   0.8454808   3.108251  ]
 [ 0.          0.          0.          1.        ]]
[[-0.846754   -0.02910021 -0.5311883   0.37526658]
 [ 0.06859153 -0.9961404  -0.05476816  0.28390554]
 [-0.5275445  -0.0828102   0.8454808   3.1088474 ]
 [ 0.          0.          0.          1.        ]]
float32
0 [0.3752 0.2839 3.1083] 
1 [0.3753 0.2839 3.1088] [6.0182548e-04 4.0898735e-06]
2 [0.3753 0.2839 3.1089] [4.0898735e-06 4.0919358e-06]
3 [0.3753 0.2839 3.1089] [4.091936e-06 4.091936e-06]
4 [0.3753 0.2839 3.1089] [4.0919358e-06 4.0898735e-06]
5 [0.3753 0.2839 3.1088] [4.0898735e-06 6.3293126e-05]
6 [0.3753 0.2839 3.1088] [6.3293126e-05 6.4003048e-05]
7 [0.3753 0.2839 3.1087] [6. …[+703 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p scratch outputs tools && python - <<'PY'
import cv2,numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
cap=cv2.VideoCapture('inputs/video.mp4'); ims=[]
with h5py.File('inputs/masks.h5') as f:
 obj=f['object']; per=f['person']
 for i in range(0,901,30):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); assert ok
  o=obj[i].astype(bool); p=per[i].astype(bool)
  # overlay object cyan, person magenta transparent, contours
  ov=im.copy(); ov[o]=(0.4*ov[o]+0.6*np.array([255,255,0])).astype(np.uint8); ov[p]=(0.65*ov[p]+0.35*np.array([255,0,255])).astype(np.uint8)
  cnts,_=cv2.findContours(o.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov,cnts,-1,(0,255,255),3)
  cnts,_=cv2.findContours(p.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov,cnts,-1,(255,0,255),2)
  cv2.putText(ov,f'frame {i}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),4); cv2.putText(ov,f'frame {i}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
  ov=cv2.resize(ov,(384,288)); ims.append(ov)
# 6 cols x 6 rows =31
canvas=np.zeros((6*288,5*384,3),np.uint8)
for j,im in enumerate(ims): canvas[(j//5)*288:(j//5+1)*288,(j%5)*384:(j%5+1)*384]=im
cv2.imwrite('scratch/contact_30.jpg',canvas)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 17, in <module>
ValueError: could not broadcast input array from shape (288,384,3) into shape (0,384,3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('inputs/video.mp4'); ims=[]
with h5py.File('inputs/masks.h5') as f:
 obj=f['object']; per=f['person']
 for i in range(0,901,30):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); assert ok
  o=obj[i].astype(bool); p=per[i].astype(bool)
  ov=im.copy(); ov[o]=(0.4*ov[o]+0.6*np.array([255,255,0])).astype(np.uint8); ov[p]=(0.65*ov[p]+0.35*np.array([255,0,255])).astype(np.uint8)
  cnts,_=cv2.findContours(o.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov,cnts,-1,(0,255,255),3)
  cnts,_=cv2.findContours(p.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(ov,cnts,-1,(255,0,255),2)
  cv2.putText(ov,f'frame {i}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),4); cv2.putText(ov,f'frame {i}',(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
  ov=cv2.resize(ov,(384,288)); ims.append(ov)
canvas=np.zeros((7*288,5*384,3),np.uint8)
for j,im in enumerate(ims): canvas[(j//5)*288:(j//5+1)*288,(j%5)*384:(j%5+1)*384]=im
cv2.imwrite('scratch/contact_30.jpg',canvas)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
from scipy.ndimage import label
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh']; kp=r['human_keypoints_world']; cp=r['contact_prob_right_left']
with h5py.File('inputs/masks.h5') as f:
 obj=f['object'][:]; per=f['person'][:]
areas=obj.sum((1,2)); print('areas percentiles',np.percentile(areas,[0,1,5,10,25,50,75,90,95,99,100]))
# bbox center/width height
ys,xs=np.where(obj.reshape(len(obj),-1)); # won't per
boxes=[]
for o in obj:
 y,x=np.where(o); boxes.append((x.min(),y.min(),x.max(),y.max()) if len(x) else (0,0,0,0))
boxes=np.array(boxes)
# static based T deltas
R=T[:,:3,:3]; rel=np.einsum('fji,fjk->fik',R[:-1],R[1:]); ra=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
ta=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
for i in range(0,901,10):
 # object nearest hand surface/keypoint dist crude keypoints
 d=np.min(np.linalg.norm(kp[i,21:63,None,:] if False else np.zeros((1,1)),axis=-1)) if False else 0
 print(f'{i:3d} area {areas[i]:6d} bbox {boxes[i]} tr {T[i,:3,3].round(3)} cp {cp[i].round(3)} stepR {(ra[i-1] if i else 0):5.1f} stepT {(ta[i-1] if i else 0):.3f}')
PY"
```
<details><summary>output</summary>

```
areas percentiles [   0.    0.  318.  554. 1230. 1673. 1971. 2229. 2269. 2358. 2422.]
  0 area   1929 bbox [768 598 851 626] tr [0.375 0.284 3.108] cp [0. 0.] stepR   0.0 stepT 0.000
 10 area   1944 bbox [768 598 852 626] tr [0.375 0.284 3.109] cp [0. 0.] stepR   0.1 stepT 0.000
 20 area   1972 bbox [768 598 853 626] tr [0.375 0.284 3.108] cp [0. 0.] stepR   0.1 stepT 0.000
 30 area   1966 bbox [768 598 853 626] tr [0.374 0.283 3.102] cp [0. 0.] stepR   0.1 stepT 0.000
 40 area   1964 bbox [768 598 853 626] tr [0.373 0.282 3.085] cp [0. 0.] stepR   0.1 stepT 0.000
 50 area   1871 bbox [769 598 852 626] tr [0.372 0.282 3.084] cp [0. 0.] stepR   0.1 stepT 0.000
 60 area   1957 bbox [768 598 852 626] tr [0.372 0.282 3.081] cp [0.002 0.003] stepR   0.1 stepT 0.000
 70 area   1811 bbox [776 598 …[+8378 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
idx=[0,70,90,100,110,120,130,140,150,160,180,200,220,250,280,310,330,350,360,370,390,400,420,450,480,500,520,550,570,590,610,630,650,680,700,720,750,770,790,810,830,850,870,900]
cap=cv2.VideoCapture('inputs/video.mp4'); ims=[]
with h5py.File('inputs/masks.h5') as f: obj=f['object']; per=f['person']
for i in idx:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); assert ok
 o=obj[i]; p=per[i]
 y,x=np.where(o)
 if len(x): cx,cy=int(np.median(x)),int(np.median(y))
 else:
  y,x=np.where(p); cx,cy=int(np.median(x)),int(np.median(y))
 # crop 240x180 around center
 x0=max(0,cx-120); y0=max(0,cy-90); crop=im[y0:y0+180,x0:x0+240].copy()
 if crop.shape[:2]"'!=(180,240): crop=cv2.resize(crop,(240,180))
 # contours
 oo=o[y0:y0+180,x0:x0+240].astype(np.uint8); pp=p[y0:y0+180,x0:x0+240].astype(np.uint8)
 cnts,_=cv2.findContours(oo,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop,cnts,-1,(0,255,255),1)
 cv2.putText(crop,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(crop,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
 ims.append(crop)
# sheet 5 cols
rows=(len(ims)+4)//5; canvas=np.zeros((rows*180,5*240,3),np.uint8)
for j,im in enumerate(ims): canvas[(j//5)*180:(j//5+1)*180,(j%5)*240:(j%5+1)*240]=im
cv2.imwrite('"'scratch/object_crops.jpg',canvas)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 870, in __getitem__
    if self._fast_read_ok and (new_dtype is None):
       ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/base.py", line 534, in __get__
    value = obj.__dict__[self.func.__name__] = self.func(obj)
                                               ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 849, in _fast_read_ok
    self._extent_type == h5s.SIMPLE
    ^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packa …[+794 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
f=h5py.File('inputs/masks.h5','r'); print(f['object'][0].shape); print(f['object'][0].sum()); print(f['object'][1].sum())
PY"
```
<details><summary>output</summary>

```
(1152, 1536)
1929
1974

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
idx=[0,70,90,100,110,120,130,140,150,160,180,200,220,250,280,310,330,350,360,370,390,400,420,450,480,500,520,550,570,590,610,630,650,680,700,720,750,770,790,810,830,850,870,900]
cap=cv2.VideoCapture('inputs/video.mp4'); ims=[]
f=h5py.File('inputs/masks.h5','r')
for i in idx:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); assert ok
 o=f['object'][i]; p=f['person'][i]
 y,x=np.where(o)
 if len(x): cx,cy=int(np.median(x)),int(np.median(y))
 else:
  y,x=np.where(p); cx,cy=int(np.median(x)),int(np.median(y))
 x0=max(0,cx-120); y0=max(0,cy-90); crop=im[y0:y0+180,x0:x0+240].copy()
 if crop.shape[:2]"'!=(180,240):
  tmp=np.zeros((180,240,3),np.uint8); tmp[:crop.shape[0],:crop.shape[1]]=crop; crop=tmp
 oo=o[y0:y0+180,x0:x0+240].astype(np.uint8)
 cnts,_=cv2.findContours(oo,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop,cnts,-1,(0,255,255),1)
 cv2.putText(crop,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(crop,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
 ims.append(crop)
f.close()
rows=(len(ims)+4)//5; canvas=np.zeros((rows*180,5*240,3),np.uint8)
for j,im in enumerate(ims): canvas[(j//5)*180:(j//5+1)*180,(j%5)*240:(j%5+1)*240]=im
cv2.imwrite('"'scratch/object_crops.jpg',canvas)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,os
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual
print(type(v),v.material)
print(type(v.material),v.material.__dict__.keys())
print(v.uv.shape,v.uv.min(0),v.uv.max(0))
for k,a in v.material.__dict__.items():
 print(k,type(a),getattr(a,'shape',None),a if isinstance(a,(str,int,float,bool,type(None))) else '')
if v.material.image is not None:
 from PIL import Image
 im=Image.fromarray(v.material.image); im.save('scratch/mesh_texture.png');print('saved',im.size,im.mode)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7f47ce10a290>
<class 'trimesh.visual.material.PBRMaterial'> dict_keys(['_data'])
(271275, 2) [0.002088   0.00335097] [0.99762601 0.99795997]
_data <class 'dict'> NoneTraceback (most recent call last):
  File "<stdin>", line 9, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'
 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mat=m.visual.material
for k in dir(mat):
 if not k.startswith('_'):
  try:
   v=getattr(mat,k)
   if isinstance(v,(str,int,float,bool,type(None))): print(k,repr(v)[:200])
   else: print(k,type(v),getattr(v,'shape',None))
  except Exception as e: pass
PY"
```
<details><summary>output</summary>

```
alphaCutoff None
alphaMode None
baseColorFactor None
baseColorTexture <class 'PIL.Image.Image'> None
copy <class 'method'> None
doubleSided True
emissiveFactor None
emissiveTexture None
main_color <class 'numpy.ndarray'> (4,)
metallicFactor 0.0
metallicRoughnessTexture None
name '_texture'
normalTexture None
occlusionTexture None
roughnessFactor None
to_color <class 'method'> None
to_simple <class 'method'> None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); im=m.visual.material.baseColorTexture
print(im.size,im.mode); im.save('scratch/mesh_texture.png')
PY"
```
<details><summary>output</summary>

```
(2168, 2332) RGB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['nvdiffrast','pytorch3d','pyrender','open3d','torchvision','kornia','romatch','lightglue','cotracker']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'NO',type(e).__name__,str(e)[:100])
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
open3d OK 0.20.0
torchvision OK 0.20.1+cu124
kornia OK 0.8.3
romatch NO ModuleNotFoundError No module named 'romatch'
lightglue NO ModuleNotFoundError No module named 'lightglue'
cotracker NO ModuleNotFoundError No module named 'cotracker'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, nvdiffrast.torch as dr
from PIL import Image
W,H=1536,1152
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=torch.tensor(m.vertices,dtype=torch.float32,device='cuda'); f=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
tex=torch.tensor(np.array(m.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')/255
uv=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda')
T=np.load('inputs/reconstruction.npz')['T_world_mesh']; cam=json.load(open('inputs/camera.json')); K=torch.tensor(cam['K'],dtype=torch.float32,device='cuda'); Wc=torch.tensor(cam['T_world_cam'],dtype=torch.float32,device='cuda')
Im=torch.eye(4,device='cuda'); C=torch.linalg.inv(Wc)
gl=dr.RasterizeCudaContext()
def render(i):
 Tw=torch.tensor(T[i],device='cuda'); Tm=C@Tw; vc=v@Tm[:3,:3].T+Tm[:3,3]
 x=vc[:,0]/vc[:,2]*K[0,0]+K[0,2]; y=vc[:,1]/vc[:,2]*K[1,1]+K[1,2]
 pos=torch.stack([2*x/W-1,1-2*y/H,vc[:,2]/10,torch.ones_like(x)],-1)[None]
 r,interp=dr.rasterize(gl,pos,f,(H,W))
 uvq=dr.interpolate(uv[None],r,f)[0]
 rgb=dr.texture(tex[None],uvq[None],filter_mode='linear')[0]
 mask=r[0,:,:,3]>0
 return rgb,mask,vc
rgb,mask,vc=render(0)
cap=cv2.VideoCapture('inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);_,im=cap.read()
rgbn=(rgb.cpu().numpy()*255).astype(np.uint8); mk=mask.cpu().numpy()
ov=im.copy(); ov[mk]=(0.25*im[mk]+0.75*rgbn[mk]).astype(np.uint8)
cv2.imwri …[+217 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 20, in <module>
    rgb,mask,vc=render(0)
                ^^^^^^^^^
  File "/work/scratch/render_test.py", line 17, in render
    rgb=dr.texture(tex[None],uvq[None],filter_mode='linear')[0]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 439, in texture
    return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^ …[+319 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(dr.texture.__doc__)
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
            must have shape [minibatch_size, height, width, 2]. When sampli …[+2838 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/uvq=dr.interpolate(uv\\[None\\],r,f)\\[0\\]/uvq=dr.interpolate(uv[None],r,f)[0]; print('uvq',uvq.shape,'r',r.shape)/\" scratch/render_test.py
python scratch/render_test.py"
```
<details><summary>output</summary>

```
uvq torch.Size([1, 1152, 1536, 2]) r torch.Size([1, 1152, 1536, 4])
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 20, in <module>
    rgb,mask,vc=render(0)
                ^^^^^^^^^
  File "/work/scratch/render_test.py", line 17, in render
    rgb=dr.texture(tex[None],uvq[None],filter_mode='linear')[0]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 439, in texture
    return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*arg …[+387 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/uvq\\[None\\],filter_mode/uvq,filter_mode/\" scratch/render_test.py
python scratch/render_test.py"
```
<details><summary>output</summary>

```
uvq torch.Size([1, 1152, 1536, 2]) r torch.Size([1, 1152, 1536, 4])
mask 1646 bbox 773 853 528 555

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
T=np.load('inputs/reconstruction.npz')['T_world_mesh'][0]; c=json.load(open('inputs/camera.json')); W=np.array(c['T_world_cam']); C=np.linalg.inv(W); K=np.array(c['K'])
for p in [T[:3,3], np.array([0.375,.284,3.108]), np.array([.375,.2,3.1])]:
 pc=C[:3,:3]@p+C[:3,3]; print(p,pc, (K@pc)[:2]/pc[2])
PY"
```
<details><summary>output</summary>

```
[0.37519768 0.28386205 3.108251  ] [0.21068037 0.18785581 3.14442167] [813.04103278 609.48888761]
[0.375 0.284 3.108] [0.21049222 0.18799404 3.14416351] [812.99647419 609.52895959]
[0.375 0.2   3.1  ] [0.21236793 0.10413407 3.13499901] [813.64757016 587.73762128]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat >> scratch/render_test.py <<'PY'
print('direct projection trans',((T[0]@np.r_[0,0,0,1])[:3]))
PY
python - <<'PY'
exec(open('scratch/render_test.py').read().split(\"rgb,mask,vc=render(0)\")[0])
rgb,mask,vc=render(0)
import numpy as np
print('vc trans',vc.mean(0),vc.min(0),vc.max(0))
Tm=C@torch.tensor(T[0],device='cuda'); vt=torch.tensor(m.vertices,device='cuda')@Tm[:3,:3].T+Tm[:3,3]
x=vt[:,0]/vt[:,2]*K[0,0]+K[0,2];y=vt[:,1]/vt[:,2]*K[1,1]+K[1,2]
print('proj bounds',x.min().item(),x.max().item(),y.min().item(),y.max().item())
print('r mask bbox',torch.where(mask)[0].min().item(),torch.where(mask)[0].max().item())
PY"
```
<details><summary>output</summary>

```
uvq torch.Size([1, 1152, 1536, 2]) r torch.Size([1, 1152, 1536, 4])
vc trans tensor([0.2114, 0.1892, 3.1434], device='cuda:0') torch.return_types.min(
values=tensor([0.0586, 0.1443, 2.9822], device='cuda:0'),
indices=tensor([235379,  65445,  51694], device='cuda:0')) torch.return_types.max(
values=tensor([0.3618, 0.2317, 3.3066], device='cuda:0'),
indices=tensor([89702, 58938,    43], device='cuda:0'))
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: double != float

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr
print(dr.rasterize.__doc__[:3000])
PY"
```
<details><summary>output</summary>

```
Rasterize triangles.

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
        ranges: In range mode, tensor with shape [minibatch_size, 2] and dtype
       …[+577 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/1-2\\*y\\/H/2*y\\/H-1/' scratch/render_test.py
python scratch/render_test.py"
```
<details><summary>output</summary>

```
uvq torch.Size([1, 1152, 1536, 2]) r torch.Size([1, 1152, 1536, 4])
mask 1647 bbox 773 853 596 623
direct projection trans [0.37519768 0.28386205 3.10825109]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
im=cv2.imread('scratch/base_render0.png'); cv2.imwrite('scratch/base_render0_crop.png',im[560:650,730:890])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp scratch/render_test.py scratch/render_flipy.py; sed -i 's/uvq=dr.interpolate(uv\\[None\\],r,f)/uvq=dr.interpolate(uv[:, [1,0]][None],r,f)/' scratch/render_flipy.py; sed -i \"s#base_render0.png#base_render0_flip.png#\" scratch/render_flipy.py; python scratch/render_flipy.py; python - <<'PY'
import cv2
im=cv2.imread('scratch/base_render0_flip.png'); cv2.imwrite('scratch/base_render0_flip_crop.png',im[560:650,730:890])
PY"
```
<details><summary>output</summary>

```
uvq torch.Size([1, 1152, 1536, 2]) r torch.Size([1, 1152, 1536, 4])
mask 1647 bbox 773 853 596 623
direct projection trans [0.37519768 0.28386205 3.10825109]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/render_zoom.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, nvdiffrast.torch as dr
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=torch.tensor(m.vertices,dtype=torch.float32,device='cuda'); f=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
tex=torch.tensor(np.array(m.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')/255; uv=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda')
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];c=json.load(open('inputs/camera.json'));K=np.array(c['K']);Wc=torch.tensor(c['T_world_cam'],dtype=torch.float32,device='cuda');C=torch.linalg.inv(Wc)
gl=dr.RasterizeCudaContext()
def render(i,cx,cy,S=5):
 W,H=200*S,140*S
 vc=v@ (C@torch.tensor(T[i],device='cuda'))[:3,:3].T+(C@torch.tensor(T[i],device='cuda'))[:3,3]
 x=(vc[:,0]/vc[:,2]*K[0,0]+K[0,2]-cx)*S; y=(vc[:,1]/vc[:,2]*K[1,1]+K[1,2]-cy)*S
 pos=torch.stack([2*x/W-1,2*y/H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None]
 r,interp=dr.rasterize(gl,pos,f,(H,W)); uvq=dr.interpolate(uv[None],r,f)[0]
 rgb=dr.texture(tex[None],uvq,filter_mode='linear')[0]; mask=r[0,:,:,3]>0
 return rgb.cpu().numpy(),mask.cpu().numpy()
cap=cv2.VideoCapture('inputs/video.mp4')
for i,cx,cy in [(0,815,610),(120,720,585),(200,645,815),(500,745,550),(650,605,805),(850,690,618)]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read(); rgb,mk=render(i,cx,cy,5)
 x0=int((cx-100)*5);y0=int((cy-70)*5) # wrong global crop for image
  …[+342 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
ims=[cv2.imread(f'scratch/zoom_{i}.jpg') for i in [0,120,200,500,650,850]]
c=np.zeros((1400,3000,3),np.uint8)
for j,im in enumerate(ims): c[(j//3)*700:(j//3+1)*700,(j%3)*1000:(j%3+1)*1000]=im
cv2.imwrite('scratch/zooms.jpg',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp scratch/render_test.py scratch/render_flipv.py; sed -i 's/uvq=dr.interpolate(uv\\[None\\],r,f)/uvq=dr.interpolate((uv*torch.tensor([1.,-1.],device=\"cuda\")+torch.tensor([0.,1.],device=\"cuda\"))[None],r,f)/' scratch/render_flipv.py; sed -i 's/base_render0.png/base_render0_flipv.png/' scratch/render_flipv.py; python scratch/render_flipv.py; python - <<'PY'
import cv2
im=cv2.imread('scratch/base_render0_flipv.png'); cv2.imwrite('scratch/base_render0_flipv_crop.png',im[560:650,730:890])
PY"
```
<details><summary>output</summary>

```
uvq torch.Size([1, 1152, 1536, 2]) r torch.Size([1, 1152, 1536, 4])
mask 1647 bbox 773 853 596 623
direct projection trans [0.37519768 0.28386205 3.10825109]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5']:
 print('\\n',p)
 with h5py.File(p,'r') as f:
  def visit(n,o):
   if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype,o.attrs)
  f.visititems(visit)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
aligned/front_stereo_camera_left (901,) object <Attributes of HDF5 object at 127907302396832>
alignment/front_stereo_camera_left/scale (901,) float32 <Attributes of HDF5 object at 127907346833920>
alignment/front_stereo_camera_left/shift (901,) float32 <Attributes of HDF5 object at 127907346833920>
alignment/front_stereo_camera_left/valid_count (901,) int32 <Attributes of HDF5 object at 127907346833920>
frame_names/front_stereo_camera_left (901,) object <Attributes of HDF5 object at 127907346833920>
raw/front_stereo_camera_left (901,) object <Attributes of HDF5 object at 127907302396832>

 /pipeline/inference/materialized_inputs.h5
K_rois (901, 3, 3) float32 <Attributes of HDF5 object at 127907556625200>
bboxes (901, 4) float32 <Attributes of HDF5 object  …[+610 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
p='/pipeline/depth/moge2/intrinsics.pkl';x=pickle.load(open(p,'rb'));print(type(x));
if hasattr(x,'keys'):print(x.keys())
if isinstance(x,dict):
 for k,v in x.items():
  import numpy as np
  print(k,type(v),getattr(v,'shape',None),v if isinstance(v,(str,int,float,bool)) else '')
PY"
```
<details><summary>output</summary>

```
fx <class 'float'> None 825.64306640625
fy <class 'float'> None 825.6430053710938
cx <class 'float'> None 768.0
cy <class 'float'> None 576.0
focals <class 'list'> None 
per_frame_intrinsics <class 'list'> None 
H <class 'int'> None 1152
W <class 'int'> None 1536
depth_backend <class 'str'> None moge2
camera_policy <class 'str'> None first_frame_export_per_frame_depth
model_id <class 'str'> None Ruicheng/moge-2-vitl-normal
model_revision <class 'str'> None b135031bae30b5ac2ae141a0e68717795ce38340
source_commit <class 'str'> None 925b8ed835a7a9cdb7578ba15c658a0afc969030
batch_size <class 'int'> None 8
devices <class 'list'> None 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import joblib,numpy as np
x=joblib.load('/pipeline/foundationpose/episode_000011.pkl')
for k,v in x.items():
 a=np.asarray(v)
 print(k,a.shape,a.dtype, (a.min(),a.max()) if np.issubdtype(a.dtype,np.number) and a.size else '')
PY"
```
<details><summary>output</summary>

```
frames (901,) <U6 
camera_id () int64 (0, 0)
fp_poses_cam (901, 1, 4, 4) float32 (-0.99995863, 4.906853)
fp_candidate_count (901,) int32 (1, 252)
fp_selected_candidate_index (901,) int32 (-1, 98)
fp_selected_iou (901,) float32 (0.0, 0.95415473)
fp_selected_visibility (901,) float32 (0.0, 1.0)
fp_retry_count (901,) int32 (0, 0)
fp_reliable (901,) bool 
fp_oracle_rotation_error_deg (901,) float32 (nan, nan)
fp_translation_threshold_m (901,) float32 (nan, nan)
fp_selected_translation_distance_m (901,) float32 (nan, nan)
gt_object_visibility_ratio (901,) float32 (nan, nan)
gt_object_visibility_valid (901,) bool 
gt_object_rendered_pixels (901,) int32 (0, 0)
gt_object_visible_pixels (901,) int32 (0, 0)
obj_pose_world (901, 4, 4) float32 (-0.99995863, 4.906853)
obj_rot_init (901, 3, 3) float32 ( …[+766 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
 d=f[k]; print(k,d.shape,d.dtype)
 z=d[0]; print(type(z),getattr(z,'shape',None),getattr(z,'dtype',None));
 if isinstance(z,np.ndarray): print(z.shape,z.dtype,z.min(),z.max())
 print('attrs',dict(d.attrs))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (901,) object
<class 'numpy.ndarray'> (648767,) uint8
(648767,) uint8 0 255
attrs {}
raw/front_stereo_camera_left (901,) object
<class 'numpy.ndarray'> (595225,) uint8
(595225,) uint8 0 255
attrs {}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned/front_stereo_camera_left\\|raw/front_stereo_camera_left\\|decode.*depth\\|aligned.h5\" -n /workspace/v2d_cari4d/lib/cari4d | head -100"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:143:            raise ValueError(f"Preserved raw depth payload is not canonical or differs from decoded shape: payload={raw_shape}, decoded={np.asarray(record.raw_depth_m).shape}")
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:147:def decode_depth_png_uint16(payload: np.ndarray) -> np.ndarray:
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:168:def _decode_metric_depth(payload: np.ndarray) -> np.ndarray:
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:169:    return decode_depth_png_uint16(payload).astype(np.float32) / 1000.0
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:174:        decoded = decode_depth_png_uint16(payload)
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:384:        return _decode_metr …[+184 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '130,190p' /workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py; sed -n '360,395p' /workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py"
```
<details><summary>output</summary>

```

def _encode_metric_depth(depth_m: np.ndarray) -> np.ndarray:
    depth_mm = np.clip(np.asarray(depth_m, dtype=np.float32) * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
    return _encode_depth_png_uint16(depth_mm)


def _encode_depth_record(record: DepthFrameRecord) -> tuple[np.ndarray, np.ndarray]:
    if record.raw_payload is None:
        raw_payload = _encode_metric_depth(record.raw_depth_m)
    else:
        raw_payload = np.asarray(record.raw_payload, dtype=np.uint8)
        raw_shape = _canonical_depth_shape(raw_payload)
        if raw_shape is None or raw_shape != tuple(np.asarray(record.raw_depth_m).shape):
            raise ValueError(f"Preserved raw depth payload is not canonical or differs from decoded shape: payload={raw_shape}, decoded={np.asarray(record.raw_dept …[+4339 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,torch,joblib
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh']; cam=json.load(open('inputs/camera.json'));W=np.array(cam['T_world_cam'])
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
P=x['pr']['pose_abs'].numpy() if torch.is_tensor(x['pr']['pose_abs']) else np.asarray(x['pr']['pose_abs'])
fp=joblib.load('/pipeline/foundationpose/episode_000011.pkl'); F=fp['obj_pose_world']; C=fp['fp_poses_cam'][:,0]
print('FP world shapes')
for i in [0,50,100,120,150,180,250,350,390,450,500,550,600,650,700,780,850,900]:
 def desc(M):
  R=M[:3,:3];return 't'+np.round(M[:3,3],2).astype(str).tolist()+' yaxis'+np.round(R[:,1],2).astype(str).tolist()
 print(i,'Tbase',desc(T[i]),'pr',desc(W@P[i]),'FP',desc(F[i] if not np.allclose(F[i],0) else C[i]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "<stdin>", line 9, in desc
TypeError: can only concatenate str (not "list") to str
FP world shapes

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,torch,joblib
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh']; cam=json.load(open('inputs/camera.json'));W=np.array(cam['T_world_cam'])
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False);P=np.asarray(x['pr']['pose_abs'])
fp=joblib.load('/pipeline/foundationpose/episode_000011.pkl'); F=np.asarray(fp['obj_pose_world']); C=np.asarray(fp['fp_poses_cam'][:,0]); WW=np.array(cam['T_world_cam'])
for i in [0,50,100,120,150,180,250,350,390,450,500,550,600,650,700,780,850,900]:
 def desc(M): return f\"t{np.round(M[:3,3],2)} yaxis{np.round(M[:3,1],2)} xaxis{np.round(M[:3,0],2)}\"
 print(i,'Tbase',desc(T[i]),'pr',desc(WW@P[i]),'FP',desc(F[i] if np.isfinite(F[i]).all() and np.linalg.norm(F[i,:3,3])<10 else WW@C[i]))
PY"
```
<details><summary>output</summary>

```
0 Tbase t[0.38 0.28 3.11] yaxis[-0.03 -1.   -0.08] xaxis[-0.85  0.07 -0.53] pr t[0.33 0.23 3.04] yaxis[-0.84  0.09 -0.54] xaxis[-0.03 -0.99 -0.11] FP t[0.16 0.14 3.21] yaxis[-0.8   0.06 -0.59] xaxis[ 0.01 -0.99 -0.12]
50 Tbase t[0.37 0.28 3.08] yaxis[-0.03 -1.   -0.08] xaxis[-0.85  0.07 -0.53] pr t[0.33 0.23 3.08] yaxis[-0.83  0.07 -0.55] xaxis[-0.02 -0.99 -0.1 ] FP t[0.16 0.13 3.19] yaxis[-0.81  0.05 -0.59] xaxis[ 0.02 -0.99 -0.11]
100 Tbase t[0.36 0.24 3.14] yaxis[ 0.07 -0.97 -0.22] xaxis[-0.94  0.01 -0.33] pr t[0.31 0.18 3.1 ] yaxis[-0.94  0.02 -0.34] xaxis[ 0.07 -0.97 -0.23] FP t[0.14 0.09 3.21] yaxis[-0.91 -0.   -0.42] xaxis[ 0.12 -0.96 -0.25]
120 Tbase t[0.04 0.21 3.28] yaxis[-0.01 -1.    0.08] xaxis[-0.96  0.04  0.28] pr t[0.   0.14 3.21] yaxis[-0.96  0.03  0.27] xaxis[-0.01 -1.     …[+3238 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,joblib
r=np.load('inputs/reconstruction.npz'); fp=joblib.load('/pipeline/foundationpose/episode_000011.pkl')
for name,A in [('fp_cam',r['T_cam_mesh_foundationpose']),('coco_cam',r['T_cam_mesh_coconet'])]:
 print(name,A[0])
for name,B in [('fp_pkl_world',fp['obj_pose_world']),('fp_cam',fp['fp_poses_cam'][:,0])]: print(name,B[0])
PY"
```
<details><summary>output</summary>

```
fp_cam [[-0.80413544  0.01387627 -0.5942842   0.16265811]
 [ 0.05886712 -0.9929547  -0.10283908  0.13619441]
 [-0.59152424 -0.11768031  0.797653    3.2134035 ]
 [ 0.          0.          0.          1.        ]]
coco_cam [[-0.81514835 -0.00658886 -0.5792149   0.16878326]
 [ 0.07312884 -0.9931053  -0.09161945  0.12549365]
 [-0.57461786 -0.11704081  0.8100089   3.0939493 ]
 [ 0.          0.          0.          1.        ]]
fp_pkl_world [[ 0.01387627 -0.80413544  0.5942842   0.16223553]
 [-0.9929547   0.05886712  0.10283908  0.13632984]
 [-0.11768031 -0.59152424 -0.797653    3.2126417 ]
 [ 0.          0.          0.          1.        ]]
fp_cam [[ 0.01387627 -0.80413544  0.5942842   0.16223553]
 [-0.9929547   0.05886712  0.10283908  0.13632984]
 [-0.11768031 -0.59152424 -0.797653    3.212641 …[+56 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh']; W=np.array(json.load(open('inputs/camera.json'))['T_world_cam'])
for nm in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 P=np.einsum('ij,fjk->fik',W,r[nm])
 d=np.einsum('fij,fij->fi',P[:,:3,:3],T[:,:3,:3]); ang=np.degrees(np.arccos(np.clip((d.sum(1)-1)/2,-1,1)))
 tr=np.linalg.norm(P[:,:3,3]-T[:,:3,3],axis=1)
 print('\\n',nm,'angle pct',np.percentile(ang,[0,25,50,75,90,95,99,100]),'trans pct',np.percentile(tr,[0,25,50,75,90,95,99,100]))
 for i in [0,50,100,120,150,180,200,250,300,350,390,420,450,500,550,600,650,700,750,800,850,900]: print(i,round(ang[i],1),round(tr[i],3))
PY"
```
<details><summary>output</summary>

```

 T_cam_mesh_foundationpose angle pct [  0.34285776   2.04025066   2.94576315   7.29457788  18.1942964
  31.39602564 156.80987357 176.36520297] trans pct [0.0564411  0.09270508 0.13429666 0.20668995 0.41818142 0.55252022
 3.75940026 3.95879309]
0 2.4 0.099
50 2.2 0.099
100 3.6 0.083
120 2.5 0.119
150 26.0 3.728
180 9.2 0.379
200 0.4 0.144
250 1.7 0.1
300 1.8 0.109
350 8.3 0.374
390 4.2 0.093
420 2.1 0.084
450 2.0 0.083
500 5.1 0.077
550 2.3 0.076
600 2.9 0.185
650 1.5 0.234
700 2.5 0.195
750 3.5 0.196
800 7.0 0.142
850 3.1 0.131
900 175.2 0.242

 T_cam_mesh_coconet angle pct [1.25093679 1.26506748 1.27517974 1.2893361  1.30349255 1.3039063
 1.30547864 1.30659583] trans pct [0.07455799 0.08099637 0.08929001 0.0977491  0.11291247 0.12017452
 0.12744356 0.13089603]
0 1.3 0.091
50 1.3 0.077
10 …[+277 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
cap=cv2.VideoCapture('inputs/video.mp4')
for i,box in [(0,(740,570,880,640)),(120,(660,550,780,625)),(500,(690,500,800,600)),(650,(560,770,650,840)),(850,(640,590,740,640))]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();x0,y0,x1,y1=box; cr=im[y0:y1,x0:x1]; cr=cv2.resize(cr,None,fx=6,fy=6,interpolation=cv2.INTER_LANCZOS4);cv2.imwrite(f'scratch/orig_{i}.png',cr)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/pose_opt.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py, nvdiffrast.torch as dr
from scipy.ndimage import distance_transform_edt
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
DEV='cuda'; W,H=1536,1152
# mesh
mesh=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV)
F=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=DEV); UV=torch.stack([UV[:,0],1-UV[:,1]],-1)
TEX=torch.tensor(np.array(mesh.visual.material.baseColorTexture),dtype=torch.float32,device=DEV)/255
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K'],np.float32); TW=np.array(cam['T_world_cam'],np.float32); CW=np.linalg.inv(TW)
Kt=torch.tensor(K,device=DEV); CWt=torch.tensor(CW,device=DEV)
gl=dr.RasterizeCudaContext()
rec=np.load('inputs/reconstruction.npz'); BASEw=rec['T_world_mesh']; FPCw=rec['T_cam_mesh_foundationpose']; COCw=rec['T_cam_mesh_coconet']
BASEc=np.einsum('ij,fjk->fik',CW,BASEw); FPC=cam['T_world_cam']
# video
cap=cv2.VideoCapture('inputs/video.mp4')
maskf=h5py.File('inputs/masks.h5','r')

def get_frame(i):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok
 return im, maskf['object'][i],maskf['person'][i]

def render_crop(Tc, crop, S=2):
 x0,y0,x1,y1=crop; w=(x1-x0)*S; h=(y1-y0)*S
 vc=V@Tc[:3,:3].T+Tc[:3,3]
 px=vc[:,0]/vc[:,2]*Kt[0,0]+Kt[0,2]; py=vc[:,1]/vc[:,2]*Kt[1,1 …[+5044 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pose_opt.py", line 117, in <module>
    print(i,nm,'init',losses(Tc0,crop,2,im,obj,per,True))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_opt.py", line 66, in losses
    ml=torch.nn.functional.binary_cross_entropy(soft.clamp(1e-5,1-1e-5),dto)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/functional.py", line 3545, in binary_cross_entropy
    raise ValueError(
ValueError: Using a target size (torch.Size([60, 177])) that is different to the input size (torch.Size([120, 354])) is deprecated. Please ensure they have the same size.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pose_opt.py'
s=open(p).read()
s=s.replace(\" soft=cv2.resize(cov.cpu().numpy().astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)\\n soft=torch.tensor(soft,device=DEV)\\n # Coverage is already high; average via pool\\n soft=torch.nn.functional.avg_pool2d(cov.float()[None,None],S,stride=S,ceil_mode=False)[0,0]\\n soft=torch.nn.functional.interpolate(soft[None,None],(h,w),mode='bilinear',align_corners=False)[0,0]\\n # downsample target to native better\\n tom=torch.nn.functional.interpolate(to.float()[None,None],(h//S,w//S),mode='area')[0,0]\\n\", \" # anti-aliased coverage by high-res pooling\\n soft=torch.nn.functional.avg_pool2d(cov.float()[None,None],S,stride=S,ceil_mode=False)[0,0]\\n\")
s=s.replace(\"return (max(0,int(cx-ww/2)),max(0,int(cy-hh/2)),min(W,int(cx+ww/2)),min(H,int(cy+hh/2))))\",\"x0=max(0,int(cx-ww/2)); y0=max(0,int(cy-hh/2)); x1=min(W,int(cx+ww/2)); y1=min(H,int(cy+hh/2));\\n return (x0,y0,x0+((x1-x0)//2)*2,y0+((y1-y0)//2)*2)\")
open(p,'w').write(s)
PY
python scratch/pose_opt.py 0"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pose_opt.py", line 112, in <module>
    print(i,nm,'init',losses(Tc0,crop,2,im,obj,per,True))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_opt.py", line 70, in losses
    ma=(blur*wgt).sum()/n; mb=(bblur*wgt).sum()/n
                               ~~~~~^~~~
RuntimeError: The size of tensor a (354) must match the size of tensor b (177) at non-singleton dimension 3

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pose_opt.py';s=open(p).read();s=s.replace(\"wgt=vis[None,None]\",\"wgt=torch.nn.functional.interpolate(vis[None,None],(h,w),mode='nearest')\")
open(p,'w').write(s)
PY
python scratch/pose_opt.py 0"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pose_opt.py", line 112, in <module>
    print(i,nm,'init',losses(Tc0,crop,2,im,obj,per,True))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_opt.py", line 70, in losses
    ma=(blur*wgt).sum()/n; mb=(bblur*wgt).sum()/n
        ~~~~^~~~
RuntimeError: The size of tensor a (177) must match the size of tensor b (354) at non-singleton dimension 3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba scratch/pose_opt.py | sed -n '50,80p'"
```
<details><summary>output</summary>

```
    50	 tp=torch.tensor(cv2.resize(target_per[y0:y1,x0:x1].astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST),device=DEV)>0.5
    51	 # mask coverage from SxS blocks
    52	 # anti-aliased coverage by high-res pooling
    53	 soft=torch.nn.functional.avg_pool2d(cov.float()[None,None],S,stride=S,ceil_mode=False)[0,0]
    54	 # exact loss on native coverage? soft avg shape h//S
    55	 dimg=torch.nn.functional.interpolate(ti.permute(2,0,1)[None],(h//S,w//S),mode='area')[0]
    56	 dto=torch.nn.functional.interpolate(to.float()[None,None],(h//S,w//S),mode='area')[0,0]
    57	 dtp=torch.nn.functional.interpolate(tp.float()[None,None],(h//S,w//S),mode='area')[0,0]
    58	 objw=dto.clamp(0,1)
    59	 vis=(dto*(1-dtp)).clamp(0,1)
    60	 # mask loss weighted visible + full object, ignore p …[+1400 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pose_opt.py';s=open(p).read();s=s.replace(\"ag=torch.sum(dimg*torch.tensor([.299,.587,.114],device=DEV)[None,:,None,None],1,keepdim=True)\",\"ag=torch.sum(ti.permute(2,0,1)[None]*torch.tensor([.299,.587,.114],device=DEV)[None,:,None,None],1,keepdim=True)\")
open(p,'w').write(s)
PY
python scratch/pose_opt.py 0"
```
<details><summary>output</summary>

```
0 base init {'loss': -2912.9580078125, 'mask': 0.42275285720825195, 'ncc': 2914.955322265625, 'l1': 0.7590437531471252, 'iou': 0.7776659959758552}
0 base opt {'loss': -1135.2078857421875, 'mask': 0.32184967398643494, 'ncc': 1137.0244140625, 'l1': 0.8642575740814209, 'iou': 0.8155339805825242} deltaR 0.0625572578997782 deltaT 0.0324696
0 coco init {'loss': 1355.9993896484375, 'mask': 2.9685146808624268, 'ncc': -1348.8123779296875, 'l1': 1.25003981590271, 'iou': 0.12515762925598992}
0 coco opt {'loss': 468.5760192871094, 'mask': 3.1739752292633057, 'ncc': -461.05670166015625, 'l1': 0.8567970991134644, 'iou': 0.0} deltaR 0.0625572578997782 deltaT 0.04332477
0 fp init {'loss': 510.91461181640625, 'mask': 2.694894552230835, 'ncc': -504.2825927734375, 'l1': 1.2110283374786377, 'iou': 0.156385461 …[+186 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pose_opt.py';s=open(p).read();s=s.replace(\"va=((blur-ma)**2*wgt).sum()/n; vb=((bblur-mb)**2*wgt).sum()/n\",\"va=((blur-ma)**2*wgt).sum()/n+1e-4; vb=((bblur-mb)**2*wgt).sum()/n+1e-4\")
open(p,'w').write(s)
PY
cat >> scratch/pose_opt.py <<'PY'
PY
python scratch/pose_opt.py 0"
```
<details><summary>output</summary>

```
0 base init {'loss': -2887.60400390625, 'mask': 0.42275285720825195, 'ncc': 2889.6005859375, 'l1': 0.7555720210075378, 'iou': 0.7776659959758552}
0 base opt {'loss': -2852.4775390625, 'mask': 0.26628729701042175, 'ncc': 2854.150146484375, 'l1': 0.7008946537971497, 'iou': 0.8380755302638386} deltaR 0.0625572578997782 deltaT 0.021937722
0 coco init {'loss': 1344.5450439453125, 'mask': 2.9685146808624268, 'ncc': -1337.359130859375, 'l1': 1.2443954944610596, 'iou': 0.12515762925598992}
0 coco opt {'loss': 650.2584838867188, 'mask': 2.7230939865112305, 'ncc': -643.5728759765625, 'l1': 1.1971696615219116, 'iou': 0.17050990581357583} deltaR 0.0625572578997782 deltaT 0.012670693
0 fp init {'loss': 506.7017517089844, 'mask': 2.694894552230835, 'ncc': -500.0707702636719, 'l1': 1.2059047222137451, 'i …[+203 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/ncc=((blur-ma)\\*(bblur-mb)\\*wgt).sum()\\/torch.sqrt(va\\*vb+1e-8)/ncc=(((blur-ma)*(bblur-mb)*wgt).sum()\\/n)\\/torch.sqrt(va*vb+1e-8)/' scratch/pose_opt.py
python scratch/pose_opt.py 0"
```
<details><summary>output</summary>

```
0 base init {'loss': 1.6221253871917725, 'mask': 0.42275285720825195, 'ncc': 0.37449464201927185, 'l1': 0.7555720210075378, 'iou': 0.7776659959758552}
0 base opt {'loss': 1.3240238428115845, 'mask': 0.2629203796386719, 'ncc': 0.34449857473373413, 'l1': 0.7134086489677429, 'iou': 0.8403100775193798} deltaR 0.0625572578997782 deltaT 0.027654225
0 coco init {'loss': 7.359231472015381, 'mask': 2.9685146808624268, 'ncc': -0.173322856426239, 'l1': 1.2443954944610596, 'iou': 0.12515762925598992}
0 coco opt {'loss': 7.503828525543213, 'mask': 3.024451494216919, 'ncc': -0.20356149971485138, 'l1': 1.2568219900131226, 'iou': 0.12075941487706193} deltaR 0.0625572578997782 deltaT 0.03564406
0 fp init {'loss': 6.695779323577881, 'mask': 2.694894552230835, 'ncc': -0.06480959057807922, 'l1': 1.20590472221 …[+212 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat >> scratch/pose_opt.py <<'PY'

def save_overlay(i,Tc,crop,fn,S=4):
 im,obj,per=get_frame(i);rgb,cov,(w,h)=render_crop(torch.tensor(Tc,device=DEV),crop,S)
 x0,y0,x1,y1=crop; cropim=cv2.resize(im[y0:y1,x0:x1],(w,h),interpolation=cv2.INTER_CUBIC)
 r=(rgb.cpu().numpy()*255).astype(np.uint8); m=cov.cpu().numpy(); ov=cropim.copy();ov[m]=(.25*cropim[m]+.75*r[m]).astype(np.uint8)
 cv2.imwrite(fn,cv2.cvtColor(ov,cv2.COLOR_RGB2BGR))
PY
# modify main via separate call
python - <<'PY'
import sys;sys.path.insert(0,'scratch');import pose_opt as p
im,obj,per=p.get_frame(0); crop=p.crop_bounds(obj)
b=p.BASEc[0]
r=p.optimize(0,b,steps=200);p.save_overlay(0,b,crop,'scratch/ov_base0.png');p.save_overlay(0,r[0],crop,'scratch/ov_opt0.png');print(r[1])
PY"
```
<details><summary>output</summary>

```
{'loss': 1.875783920288086, 'mask': 0.5370568633079529, 'ncc': 0.3547363579273224, 'l1': 0.7820326685905457, 'iou': 0.7344054580896686}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.spatial.transform import Rotation
im,obj,per=p.get_frame(0);crop=p.crop_bounds(obj);b=p.BASEc[0]
flips={'I':np.eye(3),'Rx':np.diag([1,-1,-1]),'Ry':np.diag([-1,1,-1]),'Rz':np.diag([-1,-1,1]),'RxRy':np.diag([-1,-1,1]),'RxRz':np.diag([-1,1,-1]),'RyRz':np.diag([1,-1,-1])}
for nm,S in flips.items():
 Q=b.copy();Q[:3,:3]=b[:3,:3]@S
 T=torch.tensor(Q,device=p.DEV,dtype=torch.float32)
 print(nm,p.losses(T,crop,2,im,obj,per,True))
 p.save_overlay(0,Q,crop,f'scratch/ov0_{nm}.png',3)
PY"
```
<details><summary>output</summary>

```
I {'loss': 1.6221253871917725, 'mask': 0.42275285720825195, 'ncc': 0.37449464201927185, 'l1': 0.7555720210075378, 'iou': 0.7776659959758552}
Rx {'loss': 1.6929361820220947, 'mask': 0.42540332674980164, 'ncc': 0.3171815574169159, 'l1': 0.7965551614761353, 'iou': 0.7772227772227772}
Ry {'loss': 1.776781439781189, 'mask': 0.4464145302772522, 'ncc': 0.28078386187553406, 'l1': 0.8236812353134155, 'iou': 0.771485345255837}
Rz {'loss': 1.7238439321517944, 'mask': 0.4499022960662842, 'ncc': 0.33527711033821106, 'l1': 0.7965822219848633, 'iou': 0.7678482276585122}
RxRy {'loss': 1.7238439321517944, 'mask': 0.4499022960662842, 'ncc': 0.33527711033821106, 'l1': 0.7965822219848633, 'iou': 0.7678482276585122}
RxRz {'loss': 1.776781439781189, 'mask': 0.4464145302772522, 'ncc': 0.28078386187553406, 'l1':  …[+189 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
names=['I','Rx','Ry','Rz']
ims=[cv2.imread(f'scratch/ov0_{n}.png') for n in names]; h,w=ims[0].shape[:2];c=np.zeros((2*h,2*w,3),np.uint8)
for j,im in enumerate(ims):c[(j//2)*h:(j//2+1)*h,(j%2)*w:(j%2+1)*w]=im
cv2.imwrite('scratch/flips0.jpg',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp scratch/pose_opt.py scratch/pose_opt_uvtest.py; python - <<'PY'
p='scratch/pose_opt_uvtest.py';s=open(p).read();
s=s.replace(\"UV=torch.stack([UV[:,0],1-UV[:,1]],-1)\",\"UV=torch.stack([1-UV[:,0],1-UV[:,1]],-1)\")
open(p,'w').write(s)
PY
python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt_uvtest as p
im,obj,per=p.get_frame(0);crop=p.crop_bounds(obj);b=p.BASEc[0];p.save_overlay(0,b,crop,'scratch/ov0_uvflipboth.png',3)
print(p.losses(torch.tensor(b,device=p.DEV),crop,2,im,obj,per,True))
PY"
```
<details><summary>output</summary>

```
{'loss': 1.7965360879898071, 'mask': 0.42275285720825195, 'ncc': 0.23296765983104706, 'l1': 0.9199901223182678, 'iou': 0.7776659959758552}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,cv2,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p
ids=[0,60,100,120,150,180,220,280,330,360,390,420,460,500,530,560,590,620,650,700,740,780,810,830,850,875,900]
ims=[]
for i in ids:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj)
 if crop is None:
  # around person
  y,x=np.where(per);cx=int(np.median(x));cy=int(np.median(y));crop=(max(0,cx-100),max(0,cy-70),min(p.W,cx+100),min(p.H,cy+70))
 p.save_overlay(i,p.BASEc[i],crop,f'scratch/baseov_{i}.png',4)
 j=cv2.imread(f'scratch/baseov_{i}.png');
 # normalize to 400x280
 j=cv2.resize(j,(400,280));cv2.putText(j,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3);cv2.putText(j,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1);ims.append(j)
rows=(len(ims)+4)//5;c=np.zeros((rows*280,2000,3),np.uint8)
for j,im in enumerate(ims): c[(j//5)*280:(j//5+1)*280,(j%5)*400:(j%5+1)*400]=im
cv2.imwrite('scratch/base_overlays.jpg',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
cap=cv2.VideoCapture('inputs/video.mp4')
for i,box in [(180,(580,740,720,840)),(220,(580,760,720,860)),(280,(570,760,710,860)),(420,(720,570,850,640)),(650,(550,760,670,850)),(900,(630,590,730,640))]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();x0,y0,x1,y1=box;cr=im[y0:y1,x0:x1];cr=cv2.resize(cr,None,fx=5,fy=5,interpolation=cv2.INTER_LANCZOS4);cv2.imwrite(f'scratch/raw_{i}.png',cr)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,torch,numpy as np,cv2;sys.path.insert(0,'scratch');import pose_opt as p
im,obj,per=p.get_frame(420);crop=p.crop_bounds(obj);b=p.BASEc[420]
flips={'I':np.eye(3),'Rx':np.diag([1,-1,-1]),'Ry':np.diag([-1,1,-1]),'Rz':np.diag([-1,-1,1])}
for nm,S in flips.items():
 Q=b.copy();Q[:3,:3]=b[:3,:3]@S
 rgb,cov,(w,h)=p.render_crop(torch.tensor(Q,device=p.DEV),crop,4)
 r=(rgb.cpu().numpy()*255).astype(np.uint8)[:,:,::-1];m=cov.cpu().numpy();out=np.zeros_like(r);out[m]=r[m];cv2.putText(out,nm,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);cv2.imwrite(f'scratch/render420_{nm}.png',out)
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
ims=[cv2.imread(f'scratch/render420_{n}.png') for n in ['I','Rx','Ry','Rz']];h,w=ims[0].shape[:2];c=np.zeros((2*h,2*w,3),np.uint8)
for j,im in enumerate(ims):c[(j//2)*h:(j//2+1)*h,(j%2)*w:(j%2+1)*w]=im
cv2.imwrite('scratch/render420_flips.jpg',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;uv=m.visual.uv;fc=v[m.faces].mean(1)
for nm,mask in [('yp',fc[:,1]>v[:,1].max()-.002),('yn',fc[:,1]<v[:,1].min()+.002),('xp',fc[:,0]>v[:,0].max()-.002),('xn',fc[:,0]<v[:,0].min()+.002),('zp',fc[:,2]>v[:,2].max()-.002),('zn',fc[:,2]<v[:,2].min()+.002)]:
 ids=np.where(mask)[0]; u=uv[m.faces[ids]].reshape(-1,2)
 print(nm,len(ids),'area',m.area_faces[ids].sum(),'uv mean',u.mean(0),'min',u.min(0),'max',u.max(0))
PY"
```
<details><summary>output</summary>

```
yp 15892 area 0.04775186753584852 uv mean [0.56448826 0.51262656] min [0.03420477 0.01137555] max [0.99762601 0.99795997]
yn 18882 area 0.04770936615354858 uv mean [0.38882311 0.50693441] min [0.002107   0.02176541] max [0.96456963 0.99773902]
xp 6106 area 0.0069330062161005335 uv mean [0.33233404 0.38154617] min [0.09919716 0.003483  ] max [0.9791404  0.98137498]
xn 5738 area 0.001674367919170776 uv mean [0.44801957 0.27025223] min [0.00919859 0.00933015] max [0.96456963 0.99773902]
zp 8046 area 0.0025055304672121454 uv mean [0.47349786 0.39622927] min [0.002516   0.00454199] max [0.98282397 0.99710798]
zn 7376 area 0.0018685648098205332 uv mean [0.55317855 0.3516021 ] min [0.003498   0.00335097] max [0.99762601 0.68573499]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
s=trimesh.load('inputs/object_mesh.glb',force='scene',process=False)
# set camera looking at object from a direction
try:
 img=s.save_image(resolution=(800,600),visible=True)
 print(img.shape,img.dtype);from PIL import Image;Image.fromarray(img).save('scratch/trimesh_scene.png')
except Exception as e:print(type(e),e)
PY"
```
<details><summary>output</summary>

```
<class 'ImportError'> `trimesh.viewer.windowed` requires `pip install "pyglet<2"`

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/pyrender_canon.py <<'PY'
import trimesh,pyrender,numpy as np
from PIL import Image
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mesh=pyrender.Mesh.from_trimesh(m,smooth=False)
for nm,eye in [('yp',[0,1,0]),('yn',[0,-1,0]),('zp',[0,0,1]),('zn',[0,0,-1]),('xp',[1,0,0]),('xn',[-1,0,0])]:
 scene=pyrender.Scene(bg_color=[0,0,0,255]);scene.add(mesh)
 cam=pyrender.PerspectiveCamera(yfov=np.pi/4,aspectRatio=1.0)
 pose=pyrender.lookAt_matrix(np.array(eye)*.45,np.zeros(3),np.array([0,0,1] if abs(eye[1])>.5 else [0,1,0]))
 scene.add(cam,pose=pose)
 r=pyrender.OffscreenRenderer(600,600);col,depth=r.render(scene);r.delete()
 Image.fromarray(col).save(f'scratch/canon_{nm}.png')
PY
python scratch/pyrender_canon.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pyrender_canon.py", line 8, in <module>
    pose=pyrender.lookAt_matrix(np.array(eye)*.45,np.zeros(3),np.array([0,0,1] if abs(eye[1])>.5 else [0,1,0]))
         ^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'pyrender' has no attribute 'lookAt_matrix'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pyrender
print([x for x in dir(pyrender) if 'look' in x.lower()])
import pyrender.camera as c
print(dir(c))
PY"
```
<details><summary>output</summary>

```
[]
['Camera', 'DEFAULT_Z_FAR', 'DEFAULT_Z_NEAR', 'IntrinsicsCamera', 'OrthographicCamera', 'PerspectiveCamera', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'abc', 'np', 'six', 'sys']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pyrender_canon.py';s=open(p).read();s=s.replace(\"pose=pyrender.lookAt_matrix(np.array(eye)*.45,np.zeros(3),np.array([0,0,1] if abs(eye[1])>.5 else [0,1,0]))\", \"e=np.array(eye,dtype=float);e=e/np.linalg.norm(e);up=np.array([0.,0.,1.]) if abs(e[1])>.5 else np.array([0.,1.,0.]);z=e.copy();x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x);pose=np.eye(4);pose[:3,0]=x;pose[:3,1]=y;pose[:3,2]=z;pose[:3,3]=e*.45\")
open(p,'w').write(s)
PY
python scratch/pyrender_canon.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
ims=[cv2.imread(f'scratch/canon_{n}.png') for n in ['yp','yn','zp','zn','xp','xn']];c=np.zeros((1200,1800,3),np.uint8)
for j,im in enumerate(ims):c[(j//3)*600:(j//3+1)*600,(j%3)*600:(j%3+1)*600]=im
cv2.imwrite('scratch/canon_sheet.jpg',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,cv2
from PIL import Image
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;uv=m.visual.uv;fc=v[m.faces].mean(1);tex=np.array(m.visual.material.baseColorTexture);T=tex.copy()
# map uv to pixels, note v bottom origin assumed image y=(1-v)*H
for nm,mask,col in [('yp',fc[:,1]>v[:,1].max()-.002,(0,0,255)),('yn',fc[:,1]<v[:,1].min()+.002,(0,255,0)),('xp',fc[:,0]>v[:,0].max()-.002,(255,0,0)),('xn',fc[:,0]<v[:,0].min()+.002,(0,255,255))]:
 ids=np.where(mask)[0];u=uv[m.faces[ids]].mean(1);x=(u[:,0]*(T.shape[1]-1)).astype(int);y=((1-u[:,1])*(T.shape[0]-1)).astype(int)
 for xx,yy in zip(x,y):cv2.circle(T,(xx,yy),3,col,-1)
Image.fromarray(T).save('scratch/uv_faces.png')
print('saved')
PY"
```
<details><summary>output</summary>

```
saved

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;uv=m.visual.uv;F=m.faces
for nm,sel in [('yp',v[F,1].max(1)>v[:,1].max()-.0005),('yn',v[F,1].min(1)<v[:,1].min()+.0005)]:
 ids=np.where(sel)[0];u=uv[F[ids]].reshape(-1,2)
 print(nm,len(ids),m.area_faces[ids].sum(),np.percentile(u,[0,1,5,25,50,75,95,99,100],axis=0))
PY"
```
<details><summary>output</summary>

```
yp 6868 0.037872636002406423 [[0.03420477 0.01955903]
 [0.04776597 0.03197891]
 [0.23758441 0.10049695]
 [0.61438048 0.30931121]
 [0.77707157 0.49679857]
 [0.86555225 0.61950034]
 [0.94909465 0.88593817]
 [0.98032349 0.92769319]
 [0.99736023 0.96427101]]
yn 2602 0.0036880101719239536 [[0.00255938 0.10333073]
 [0.01929873 0.12754111]
 [0.04824512 0.1646052 ]
 [0.30490936 0.33021767]
 [0.33325934 0.53061074]
 [0.36424667 0.75997105]
 [0.4647508  0.94535518]
 [0.62595409 0.98670244]
 [0.64123785 0.99773902]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2
from scipy.ndimage import distance_transform_edt
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'];kp=r['human_keypoints_world']; cp=r['contact_prob_right_left']
cap=cv2.VideoCapture('inputs/video.mp4'); frames=[]
for i in range(901):
 ok,f=cap.read();assert ok;frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
frames=np.stack(frames)
f=h5py.File('inputs/masks.h5','r');obj=f['object'][:];per=f['person'][:];f.close()
# centroids/bboxes
cent=[]; bb=[]
for o in obj:
 y,x=np.where(o)
 if len(x):cent.append([x.mean(),y.mean()]);bb.append([x.min(),y.min(),x.max(),y.max()])
 else:cent.append([np.nan,np.nan]);bb.append([np.nan]*4)
cent=np.array(cent);bb=np.array(bb)
# person-object bbox approx and keypoint center dist
for i in range(0,901,10):
 # min person pixel dist from object bbox expanded small via distance transform around bbox center
 y,x=np.where(per[i]);
 if np.isfinite(bb[i,0]):
  bx=bb[i];
  # dist hand keypoints to baseline object center
  d=np.linalg.norm(kp[i,21:63]-T[i,:3,3],axis=1); dmin=d.min(); hand=np.argmin(d)+21
 else:dmin=np.nan;hand=-1
 # frame diff within union obj masks 10-apart aligned around bbox center, compare bbox crop from frames
 if i>=10 and np.isfinite(bb[i,0]) and np.isfinite(bb[i-10,0]):
  cx,cy=np.nanmean(cent[i-5:i+1],axis=0); x0=int(cx-60);y0=int(cy-40);x1=int(cx+60);y1=int(cy+40)
  a=frames[i-10,y0:y1,x0:x1].astype(float);b=frames[i,y0:y1,x0:x1].astype(float)
  imdiff=np.mean …[+175 chars]
```
<details><summary>output</summary>

```
  0 c  809.5, 611.3 bb [768 598 851 626] dhand 0.813 kp 28 cp [1.5974570e-10 9.2131275e-10] imgdiff nan
 10 c  809.8, 611.3 bb [768 598 852 626] dhand 0.791 kp 27 cp [1.1577156e-10 1.0545067e-09] imgdiff 1.2
 20 c  810.4, 611.4 bb [768 598 853 626] dhand 0.784 kp 27 cp [7.6607404e-10 1.8111628e-08] imgdiff 0.9
 30 c  810.4, 611.3 bb [768 598 853 626] dhand 0.762 kp 27 cp [2.0441697e-09 2.4835634e-08] imgdiff 0.9
 40 c  810.4, 611.3 bb [768 598 853 626] dhand 0.630 kp 48 cp [4.852033e-10 9.586964e-08] imgdiff 0.8
 50 c  810.4, 611.2 bb [769 598 852 626] dhand 0.317 kp 46 cp [3.0893423e-07 2.1484086e-05] imgdiff 3.6
 60 c  810.2, 611.3 bb [768 598 852 626] dhand 0.167 kp 25 cp [0.00224262 0.00258621] imgdiff 9.5
 70 c  813.1, 611.4 bb [776 598 852 626] dhand 0.071 kp 46 cp [0.9990666 0.99991 …[+8109 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
ids=list(range(800,901,5));cap=cv2.VideoCapture('inputs/video.mp4');f=h5py.File('inputs/masks.h5','r');ims=[]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();o=f['object'][i];y,x=np.where(o)
 if len(x):cx,cy=int(np.median(x)),int(np.median(y))
 else:y,x=np.where(f['person'][i]);cx,cy=int(np.median(x)),int(np.median(y))
 cr=im[max(0,cy-60):cy+60,max(0,cx-100):cx+100].copy();cr=cv2.resize(cr,(300,180),interpolation=cv2.INTER_CUBIC)
 cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);ims.append(cr)
f.close();rows=(len(ims)+4)//5;c=np.zeros((rows*180,1500,3),np.uint8)
for j,im in enumerate(ims):c[(j//5)*180:(j//5+1)*180,(j%5)*300:(j%5+1)*300]=im
cv2.imwrite('scratch/end_sheet.jpg',c)
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
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'];W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W)
for nm in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 P=np.einsum('ij,fjk->fik',W,r[nm]);d=P[:,:3,3]-T[:,:3,3];dc=d@C[:3,:3].T
 # in camera coords diff, projection along ray vs image
 ray=P[:,:3,3]@C[:3,:3].T+C[:3,3]
 dep=(dc*ray).sum(1)/np.linalg.norm(ray,axis=1)
 lat=np.linalg.norm(dc-dep[:,None]*ray/np.linalg.norm(ray,axis=1)[:,None],axis=1)
 print(nm,'depth diff pct',np.percentile(dep,[0,25,50,75,90,99,100]),'lat',np.percentile(lat,[0,25,50,75,90,99,100]))
 for i in [0,100,200,300,400,500,600,700,800,900]:print(i,round(dep[i],3),round(lat[i],3),np.round(dc[i],3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/texture_scores.py <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.ndimage import gaussian_filter
for i in [0,60,220,280,420,650,700,860,900]:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj)
 if crop is None:continue
 print('\\nframe',i,'crop',crop)
 b=p.BASEc[i]; flips={'I':np.eye(3),'Rx':np.diag([1,-1,-1]),'Ry':np.diag([-1,1,-1]),'Rz':np.diag([-1,-1,1])}
 x0,y0,x1,y1=crop
 for nm,S in flips.items():
  Q=b.copy();Q[:3,:3]=b[:3,:3]@S
  rgb,cov,(w,h)=p.render_crop(torch.tensor(Q,device=p.DEV),crop,2)
  rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)
  cc=cv2.resize(cov.cpu().numpy().astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)>.5
  tt=im[y0:y1,x0:x1].astype(np.float32)/255;oo=obj[y0:y1,x0:x1];pp=per[y0:y1,x0:x1]
  m=oo&~pp
  # erase person from rendered via template weight only
  # blur both
  a=gaussian_filter(tt,(1.2,1.2,0));bb=gaussian_filter(rr,(1.2,1.2,0))
  vals=[]
  for c in range(3):
   aa=a[:,:,c][m];bbb=bb[:,:,c][m]
   if aa.std()<1e-4 or bbb.std()<1e-4: vals.append(0)
   else:vals.append(np.corrcoef(aa,bbb)[0,1])
  # mean rgb
  md=np.mean(np.abs(a[m]-bb[m])); ncc=np.mean(vals)
  soft=cc; iou=(soft&oo).sum()/max(1,(soft|oo).sum())
  print(nm,'ncc',round(ncc,3),'rgbcorrs',np.round(vals,2),'mae',round(md,3),'iou',round(iou,3),'mean',np.round(bb[m].mean(0),2))
PY
python scratch/texture_scores.py"
```
<details><summary>output</summary>

```

frame 0 crop (721, 582, 898, 642)
I ncc 0.355 rgbcorrs [0.45 0.34 0.27] mae 0.322 iou 0.778 mean [0.4  0.42 0.39]
Rx ncc 0.306 rgbcorrs [0.35 0.28 0.28] mae 0.33 iou 0.777 mean [0.39 0.41 0.39]
Ry ncc 0.275 rgbcorrs [0.3  0.27 0.26] mae 0.304 iou 0.771 mean [0.42 0.44 0.41]
Rz ncc 0.318 rgbcorrs [0.37 0.32 0.26] mae 0.312 iou 0.768 mean [0.42 0.43 0.39]

frame 60 crop (721, 582, 899, 642)
I ncc 0.304 rgbcorrs [0.4  0.27 0.24] mae 0.304 iou 0.79 mean [0.4  0.43 0.39]
Rx ncc 0.266 rgbcorrs [0.31 0.23 0.25] mae 0.311 iou 0.787 mean [0.39 0.42 0.39]
Ry ncc 0.263 rgbcorrs [0.29 0.24 0.26] mae 0.285 iou 0.784 mean [0.42 0.44 0.42]
Rz ncc 0.305 rgbcorrs [0.37 0.29 0.26] mae 0.294 iou 0.778 mean [0.42 0.43 0.4 ]

frame 220 crop (570, 774, 714, 868)
I ncc 0.295 rgbcorrs [0.12 0.33 0.44] mae 0.1 io …[+2457 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/classify_flips.py <<'PY'
import sys,numpy as np,torch,cv2,time;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.ndimage import gaussian_filter
flips={'I':np.eye(3),'Rx':np.diag([1,-1,-1]),'Ry':np.diag([-1,1,-1]),'Rz':np.diag([-1,-1,1])}
rows=[]
for i in range(0,901,5):
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj)
 if crop is None:
  rows.append((i,*[np.nan]*8));continue
 x0,y0,x1,y1=crop;tt=im[y0:y1,x0:x1].astype(np.float32)/255;oo=obj[y0:y1,x0:x1];pp=per[y0:y1,x0:x1];m=oo&~pp
 sc=[];ious=[]
 for nm,S in flips.items():
  Q=p.BASEc[i].copy();Q[:3,:3]=Q[:3,:3]@S
  rgb,cov,(w,h)=p.render_crop(torch.tensor(Q,device=p.DEV),crop,2)
  rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);cc=cv2.resize(cov.cpu().numpy().astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)>.5
  a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));vals=[]
  for c in range(3):
   aa=a[:,:,c][m];bbb=b[:,:,c][m]
   vals.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else -1)
  sc.append(np.mean(vals));ious.append((cc&oo).sum()/max(1,(cc|oo).sum()))
 b=np.nanargmax(sc)
 rows.append((i,*sc,*ious,b))
 if i%50==0:print(i,np.round(sc,3),np.round(ious,3),list(flips)[b],flush=True)
np.save('scratch/flip_scores.npy',np.array(rows))
PY
python scratch/classify_flips.py"
```
<details><summary>output</summary>

```
0 [0.355 0.306 0.275 0.318] [0.778 0.777 0.771 0.768] I
50 [0.341 0.289 0.258 0.308] [0.801 0.799 0.792 0.787] I
100 [0.638 0.575 0.547 0.554] [0.768 0.769 0.78  0.773] I
150 [-0.131 -0.14  -0.129 -0.122] [0.119 0.123 0.117 0.118] Rz
200 [0.458 0.371 0.51  0.481] [0.865 0.848 0.836 0.849] Ry
250 [0.438 0.314 0.4   0.406] [0.866 0.847 0.833 0.848] I
300 [0.236 0.145 0.284 0.24 ] [0.849 0.833 0.819 0.835] Ry
350 [-0.287 -0.279 -0.284 -0.291] [0.546 0.549 0.54  0.543] Rx
400 [0.347 0.325 0.208 0.236] [0.804 0.808 0.799 0.791] I
450 [0.476 0.432 0.431 0.435] [0.677 0.679 0.667 0.662] I
500 [0.432 0.418 0.575 0.453] [0.787 0.775 0.805 0.826] Ry
550 [0.494 0.529 0.34  0.406] [0.835 0.82  0.823 0.847] Rx
600 [ 0.093  0.113 -0.053  0.006] [0.429 0.431 0.432 0.432] Rx
650 [0.625 0.725 0.584 0.598]  …[+705 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/rows.append((i,\\*\\[np.nan\\]\\*8));continue/rows.append((i,*[np.nan]*8,np.nan));continue/\" scratch/classify_flips.py
python scratch/classify_flips.py > scratch/flip_run.log 2>&1; tail -30 scratch/flip_run.log"
```
<details><summary>output</summary>

```
0 [0.355 0.306 0.275 0.318] [0.778 0.777 0.771 0.768] I
50 [0.341 0.289 0.258 0.308] [0.801 0.799 0.792 0.787] I
100 [0.638 0.575 0.547 0.554] [0.768 0.769 0.78  0.773] I
150 [-0.131 -0.14  -0.129 -0.122] [0.119 0.123 0.117 0.118] Rz
200 [0.458 0.371 0.51  0.481] [0.865 0.848 0.836 0.849] Ry
250 [0.438 0.314 0.4   0.406] [0.866 0.847 0.833 0.848] I
300 [0.236 0.145 0.284 0.24 ] [0.849 0.833 0.819 0.835] Ry
350 [-0.287 -0.279 -0.284 -0.291] [0.546 0.549 0.54  0.543] Rx
400 [0.347 0.325 0.208 0.236] [0.804 0.808 0.799 0.791] I
450 [0.476 0.432 0.431 0.435] [0.677 0.679 0.667 0.662] I
500 [0.432 0.418 0.575 0.453] [0.787 0.775 0.805 0.826] Ry
550 [0.494 0.529 0.34  0.406] [0.835 0.82  0.823 0.847] Rx
600 [ 0.093  0.113 -0.053  0.006] [0.429 0.431 0.432 0.432] Rx
650 [0.625 0.725 0.584 0.598]  …[+327 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('scratch/flip_scores.npy');names=['I','Rx','Ry','Rz'];idx=a[:,0].astype(int);s=a[:,1:5];iou=a[:,5:9];best=np.nanargmax(s,1)
# print runs raw
runs=[];st=0
for j in range(1,len(best)+1):
 if j==len(best) or best[j]"'!=best[st]:runs.append((idx[st],idx[j-1],names[best[st]],np.nanmean(s[st:j,0]),np.nanmean(s[st:j,best[st]],np.nanmean(iou[st:j,0]))));st=j
for r in runs:print(r)
print('"'\\nevery 10:')
for j in range(0,len(a),2): print(idx[j],np.round(s[j],2),names[best[j]],round(iou[j,0],2))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py", line 613, in nanargmax
    raise ValueError("All-NaN slice encountered")
ValueError: All-NaN slice encountered

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('scratch/flip_scores.npy');names=['I','Rx','Ry','Rz'];idx=a[:,0].astype(int);s=a[:,1:5];iou=a[:,5:9];best=np.full(len(a),-1)
for j in range(len(a)):
 if np.isfinite(s[j]).any(): best[j]=np.nanargmax(s[j])
# print runs ignoring -1 separately
runs=[];st=0
for j in range(1,len(best)+1):
 if j==len(best) or best[j]"'!=best[st]:runs.append((idx[st],idx[j-1],'"'NA' if best[st]<0 else names[best[st]],np.nanmean(s[st:j,0]) if np.isfinite(s[st:j,0]).any() else np.nan,np.nanmean(s[st:j,best[st]]) if best[st]>=0 else np.nan,np.nanmean(iou[st:j,0])));st=j
for r in runs:print(r)
print('\\nevery 10:')
for j in range(0,len(a),2): print(idx[j],np.round(s[j],2),'NA' if best[j]<0 else names[best[j]],round(iou[j,0],2))
PY"
```
<details><summary>output</summary>

```
<stdin>:8: RuntimeWarning: Mean of empty slice
(0, 55, 'I', 0.30571840739337, 0.30571840739337, 0.7832702784362203)
(60, 60, 'Rz', 0.3040336916967222, 0.3054816029244987, 0.7902341803687095)
(65, 105, 'I', 0.5310488712187776, 0.5310488712187776, 0.8128103122780284)
(110, 110, 'Ry', 0.4139437079839685, 0.4959677956972497, 0.647945205479452)
(115, 115, 'I', 0.814650848356127, 0.814650848356127, 0.6526508226691042)
(120, 120, 'Rx', 0.7542732583022032, 0.7733317743029692, 0.7329974811083123)
(125, 130, 'I', 0.7019063543675339, 0.7019063543675339, 0.6441666666666667)
(135, 135, 'Rz', 0.80158972128239, 0.8034570031539561, 0.13618677042801555)
(140, 145, 'I', -1.0, -1.0, 0.0)
(150, 150, 'Rz', -0.1309250930367757, -0.12249714946385722, 0.1185617103984451)
(155, 155, 'Rx', 0.396893056223433, 0.4293 …[+7035 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3]
def relang(a,b):return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))
for a,b in [(0,60),(200,220),(200,300),(210,300),(270,305),(400,430),(400,470),(630,650),(650,700),(700,750),(850,880),(850,900),(860,900)]:print(a,b,'R',relang(R[a],R[b]),'t',np.linalg.norm(T[a,:3,3]-T[b,:3,3]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
ids=list(range(190,321,10));cap=cv2.VideoCapture('inputs/video.mp4');f=h5py.File('inputs/masks.h5','r');ims=[]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();o=f['object'][i];y,x=np.where(o);cx=int(np.median(x));cy=int(np.median(y));cr=im[cy-50:cy+50,cx-90:cx+90];cr=cv2.resize(cr,(360,200),interpolation=cv2.INTER_CUBIC);cv2.putText(cr,str(i),(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(cr,str(i),(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);ims.append(cr)
f.close();c=np.zeros((400,7*360,3),np.uint8)
for j,im in enumerate(ims):c[(j//7)*200:(j//7+1)*200,(j%7)*360:(j%7+1)*360]=im
cv2.imwrite('scratch/b_sheet.jpg',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
sets={'C':range(390,481,10),'D':range(620,751,10),'E':range(830,901,10),'A':range(0,91,10)}
cap=cv2.VideoCapture('inputs/video.mp4');f=h5py.File('inputs/masks.h5','r')
for nm,ids in sets.items():
 ims=[]
 for i in ids:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();o=f['object'][i];y,x=np.where(o)
  if len(x):cx=int(np.median(x));cy=int(np.median(y))
  else:y,x=np.where(f['person'][i]);cx=int(np.median(x));cy=int(np.median(y))
  cr=im[max(0,cy-45):cy+45,max(0,cx-80):cx+80];cr=cv2.resize(cr,(320,180),interpolation=cv2.INTER_CUBIC);cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);ims.append(cr)
 rows=(len(ims)+4)//5;c=np.zeros((rows*180,5*320,3),np.uint8)
 for j,im in enumerate(ims):c[(j//5)*180:(j//5+1)*180,(j%5)*320:(j%5+1)*320]=im
 cv2.imwrite(f'scratch/sheet_{nm}.jpg',c)
f.close()
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
a=np.load('scratch/flip_scores.npy');idx=a[:,0].astype(int);names=['I','Rx','Ry','Rz'];s=a[:,1:5];iou=a[:,5:9]
ph={'A':(0,60),'B':(195,305),'C':(390,470),'D':(620,748),'E':(840,900)}
for nm,(lo,hi) in ph.items():
 m=(idx>=lo)&(idx<=hi)&np.isfinite(s).all(1)&(iou[:,0]>.5)
 print('\\n',nm,'n',m.sum(),'mean',np.round(np.nanmean(s[m],0),3),'median',np.round(np.nanmedian(s[m],0),3),'wins',[int((np.argmax(s[m],1)==j).sum()) for j in range(4)])
 best=np.argmax(s[m],1)
 for j,n in enumerate(names):print(n,'wins',np.sum(best==j))
 # compare means
 for j,n in enumerate(names): print(n, np.mean(s[m,j]))
PY"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
I nan
Rx nan
Ry nan
Rz nan

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('inputs/video.mp4');f=h5py.File('inputs/masks.h5','r');ims=[]
for i in range(806,851,2):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();o=f['object'][i];y,x=np.where(o)
 if len(x):cx=int(np.median(x));cy=int(np.median(y))
 else:y,x=np.where(f['person'][i]);cx=int(np.median(x));cy=int(np.median(y))
 cr=im[cy-40:cy+45,cx-75:cx+75];cr=cv2.resize(cr,(300,170),interpolation=cv2.INTER_CUBIC);cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3);cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1);ims.append(cr)
f.close();rows=(len(ims)+4)//5;c=np.zeros((rows*170,1500,3),np.uint8)
for j,im in enumerate(ims):c[(j//5)*170:(j//5+1)*170,(j%5)*300:(j%5+1)*300]=im
cv2.imwrite('scratch/e_trans.jpg',c)
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
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3]
def rel(a,b): return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))
Sx=np.diag([1,-1,-1]); Sry=np.diag([-1,1,-1])
for i in [620,650,700,740,750,800,810,820,830,840,850,900]:
 print(i,'base Dcorr vs E base',rel(R[650]@Sx,R[i]),'E Ry',rel(R[650]@Sx,R[i]@Sry))
PY"
```
<details><summary>output</summary>

```
620 base Dcorr vs E base 176.38520359104098 E Ry 175.75095374903495
650 base Dcorr vs E base 179.908663034473 E Ry 180.0
700 base Dcorr vs E base 179.27372383211542 E Ry 179.90605700806705
740 base Dcorr vs E base 179.27384775440552 E Ry 179.2755086405866
750 base Dcorr vs E base 176.35641298462514 E Ry 179.72193976356738
800 base Dcorr vs E base 169.61095087848844 E Ry 179.0834939122783
810 base Dcorr vs E base 178.65583870998856 E Ry 178.49293544335393
820 base Dcorr vs E base 178.6732974771655 E Ry 178.5076273598562
830 base Dcorr vs E base 179.32843009005916 E Ry 176.44849250423678
840 base Dcorr vs E base 177.50794556013685 E Ry 179.82088864409857
850 base Dcorr vs E base 175.14361218849382 E Ry 179.81963498223323
900 base Dcorr vs E base 174.62202008023172 E Ry 172.88790424570246

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3]
def rel(a,b): return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))
Sx=np.diag([1,-1,-1]);Sry=np.diag([-1,1,-1]);Srz=np.diag([-1,-1,1])
print('base D-E',rel(R[650],R[850]),rel(R[650],R[900]))
# compare Dcorr to E flips
for n,S in [('I',np.eye(3)),('Rx',Sx),('Ry',Sry),('Rz',Srz)]:
 print(n,rel(R[650]@Sx,R[850]@S),rel(R[650]@Sx,R[900]@S))
PY"
```
<details><summary>output</summary>

```
base D-E 56.3050275667754 21.425186840255385
I 175.14361218849382 174.62202008023172
Rx 56.305022351451605 21.425180305957866
Ry 179.81963498223323 172.88790424570246
Rz 123.94290054253649 160.56016407186223

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3]
def rel(a,b): return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))
pts=[470,480,490,500,510,520,530,540,550,560,570,580,590,600,610,620,630,650]
for a,b in zip(pts[:-1],pts[1:]): print(a,b,round(rel(R[a],R[b]),1))
print('cumulative');print([(i,round(rel(R[470],R[i]),1),round(rel(R[650],R[i]),1)) for i in pts])
PY"
```
<details><summary>output</summary>

```
470 480 18.9
480 490 25.6
490 500 12.0
500 510 21.3
510 520 17.9
520 530 24.4
530 540 31.7
540 550 21.3
550 560 19.1
560 570 12.9
570 580 18.3
580 590 14.0
590 600 2.2
600 610 9.0
610 620 11.5
620 630 8.5
630 650 0.5
cumulative
[(470, 0.1, 178.9), (480, 18.9, 168.2), (490, 38.2, 150.1), (500, 46.0, 138.7), (510, 66.4, 117.8), (520, 83.9, 100.4), (530, 107.0, 76.8), (540, 135.9, 45.2), (550, 155.8, 30.8), (560, 172.3, 15.6), (570, 179.0, 9.7), (580, 169.6, 10.9), (590, 169.0, 21.1), (600, 168.5, 19.6), (610, 175.7, 18.0), (620, 173.5, 8.8), (630, 178.5, 0.5), (650, 178.9, 0.0)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3]
for a,b in [(470,570),(390,650),(420,650),(0,220),(0,420),(650,900),(650,850)]:
 M=R[a].T@R[b];ang=np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)));w,v=np.linalg.eig(M);axis=np.real(v[:,np.argmin(np.abs(w-1))]);axis=axis/np.linalg.norm(axis)
 print(a,b,'ang',ang,'axis',np.round(axis,3),'M',np.round(M,2))
PY"
```
<details><summary>output</summary>

```
470 570 ang 178.959201407187 axis [-0.638  0.016  0.77 ] M [[-0.19 -0.03 -0.98]
 [-0.01 -1.    0.04]
 [-0.98  0.01  0.19]]
390 650 ang 177.16344414714206 axis [-0.711  0.004  0.703] M [[ 0.01  0.03 -1.  ]
 [-0.04 -1.   -0.03]
 [-1.    0.04 -0.01]]
420 650 ang 179.41633460138786 axis [ 0.733 -0.017 -0.68 ] M [[ 0.07 -0.02 -1.  ]
 [-0.03 -1.    0.02]
 [-1.    0.03 -0.08]]
0 220 ang 77.50745119561962 axis [-0.035 -0.999 -0.009] M [[ 0.22  0.04 -0.98]
 [ 0.02  1.    0.04]
 [ 0.98 -0.03  0.22]]
0 420 ang 2.4610042524552482 axis [0.374 0.587 0.718] M [[ 1.   -0.03  0.03]
 [ 0.03  1.   -0.02]
 [-0.02  0.02  1.  ]]
650 900 ang 21.425186840255385 axis [ 0.252 -0.908  0.334] M [[ 0.94 -0.14 -0.33]
 [ 0.11  0.99 -0.11]
 [ 0.34  0.07  0.94]]
650 850 ang 56.3050275667754 axis [-0.09   0.996 -0.003] M [ …[+63 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/sweep_flip_axis.py <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.ndimage import gaussian_filter
# static phase score sets
sets={'B':list(range(205,306,10)),'D':list(range(620,751,10)),'E':list(range(840,901,10))}
for phase,ids in sets.items():
 print('\\nPHASE',phase)
 for deg in range(0,181,15):
  ph=np.radians(deg);c=np.cos(2*ph);s=np.sin(2*ph)
  S=np.array([[c,0,s],[0,-1,0],[s,0,-c]])
  vals=[];ious=[]
  for i in ids:
   im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj)
   if crop is None:continue
   x0,y0,x1,y1=crop;o=obj[y0:y1,x0:x1];pp=per[y0:y1,x0:x1];m=o&~pp
   if m.sum()<100:continue
   Q=p.BASEc[i].copy();Q[:3,:3]=Q[:3,:3]@S
   rgb,cov,(w,h)=p.render_crop(torch.tensor(Q,device=p.DEV),crop,2)
   rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);cc=cv2.resize(cov.cpu().numpy().astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)>.5
   tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));ccs=[]
   for ch in range(3):
    aa=a[:,:,ch][m];bbb=b[:,:,ch][m];ccs.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else -1)
   vals.append(np.mean(ccs));ious.append((cc&obj[y0:y1,x0:x1]).sum()/max(1,(cc|obj[y0:y1,x0:x1]).sum()))
  print(deg,round(np.mean(vals),4),round(np.median(vals),4),round(np.mean(ious),4),len(vals))
PY
python scratch/sweep_flip_axis.py"
```
<details><summary>output</summary>

```

PHASE B
0 0.2393 0.2325 0.8308 11
15 0.2084 0.214 0.746 11
30 0.2183 0.2261 0.7081 11
45 0.2146 0.2026 0.6691 11
60 0.0315 0.0451 0.6765 11
75 0.0393 0.0169 0.768 11
90 0.3381 0.3293 0.8279 11
105 0.1796 0.1705 0.7485 11
120 0.1725 0.1834 0.7271 11
135 0.1189 0.109 0.6952 11
150 -0.0011 0.0051 0.7112 11
165 -0.0407 -0.0699 0.7718 11
180 0.2393 0.2325 0.8308 11

PHASE D
0 0.4975 0.4597 0.7108 14
15 0.0166 0.046 0.6633 14
30 -0.1312 -0.1239 0.6199 14
45 0.0559 0.0401 0.599 14
60 0.2215 0.1213 0.6234 14
75 0.2394 0.1196 0.6536 14
90 0.2276 0.1052 0.7284 14
105 -0.0356 -0.0351 0.6652 14
120 -0.0869 -0.1193 0.6424 14
135 0.1249 0.1128 0.597 14
150 0.3493 0.2782 0.614 14
165 0.4371 0.3838 0.6485 14
180 0.4975 0.4597 0.7108 14

PHASE E
0 0.2991 0.4061 0.3223 6
15 0.2593 0.2334 0.3238 6
30 0.3026 …[+280 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/static_opt.py <<'PY'
import sys,numpy as np,torch,cv2,time;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.spatial.transform import Rotation
def meanR(Rs):
 q=Rotation.from_matrix(Rs).as_quat();ref=q[0];q=q*np.where(q@ref<0,-1,1)[:,None];q=q.mean(0);q/=np.linalg.norm(q);return Rotation.from_quat(q).as_matrix()
def makebase(ids,S=np.eye(3)):
 Rs=p.BASEw[ids,:3,:3];ts=p.BASEw[ids,:3,3]
 Rw=meanR(Rs)@S; tw=np.median(ts,0)
 B=np.eye(4);B[:3,:3]=Rw;B[:3,3]=tw;return p.CW@B

def opt_static(ids,S=np.eye(3),steps=120,batch=6,lrR=.006,lrT=.002,seed=0):
 rng=np.random.default_rng(seed);base=torch.tensor(makebase(ids,S),device=p.DEV,dtype=torch.float32)
 w=torch.zeros(3,device=p.DEV,requires_grad=True);t=torch.zeros(3,device=p.DEV,requires_grad=True);opt=torch.optim.Adam([w,t],lr=lrT)
 cache={}
 def data(i):
  if i not in cache:
   im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);cache[i]=(im,obj,per,crop)
  return cache[i]
 valid=[i for i in ids if data(i)[3] is not None and (data(i)[1]&~data(i)[2]).sum()>100]
 # eval all with simple score
 def ev(Tc,sel,dbg=False):
  vals=[]
  for i in sel:
   im,obj,per,crop=data(i);vals.append(p.losses(Tc,crop,2,im,obj,per,dbg))
  return vals
 best=(1e9,None,None)
 for k in range(steps):
  opt.param_groups[0]['lr']=lrT*(1-k/steps)+lrT*.03
  sel=list(rng.choice(valid,min(batch,len(valid)),replace=False))
  Tc=p.makeT(w,t,base)
  ls=ev(Tc,sel);loss=sum(x['loss'] if isinstance(x,dict) else x for x in ls)/le …[+983 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_opt.py", line 42, in <module>
    b=opt_static(ids,S,steps=int(sys.argv[2]) if len(sys.argv)>2 else 120)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/static_opt.py", line 34, in opt_static
    Tc=p.makeT(w,t,base);evs=ev(Tc,valid[:]);score=np.mean([x['loss'] if isinstance(x,dict) else x for x in evs])
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/fromnumeric.py", line 3504, in mean
    return _methods._mean(a, axis=axis, dtype=dtype,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/_method …[+332 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/static_opt.py';s=open(p).read();s=s.replace(\"def ev(Tc,sel,dbg=False):\\n  vals=[]\\n  for i in sel:\\n   im,obj,per,crop=data(i);vals.append(p.losses(Tc,crop,2,im,obj,per,dbg))\\n  return vals\", \"def ev(Tc,sel,dbg=False):\\n  vals=[]\\n  for i in sel:\\n   im,obj,per,crop=data(i);v=p.losses(Tc,crop,2,im,obj,per,dbg);vals.append(v)\\n  return vals\")
s=s.replace(\"ls=ev(Tc,sel);loss=sum(x['loss'] if isinstance(x,dict) else x for x in ls)/len(ls)\",\"ls=ev(Tc,sel);loss=sum((x['loss'] if isinstance(x,dict) else float(x)) for x in ls)/len(ls)\")
s=s.replace(\"evs=ev(Tc,valid[:]);score=np.mean([x['loss'] if isinstance(x,dict) else x for x in evs])\",\"evs=ev(Tc,valid[:],True);score=np.mean([x['loss'] for x in evs])\")
open(p,'w').write(s)
PY
python scratch/static_opt.py B 120"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_opt.py", line 42, in <module>
    b=opt_static(ids,S,steps=int(sys.argv[2]) if len(sys.argv)>2 else 120)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/static_opt.py", line 31, in opt_static
    opt.zero_grad();loss.backward();opt.step()
                    ^^^^^^^^^^^^^
AttributeError: 'float' object has no attribute 'backward'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/static_opt.py';s=open(p).read();s=s.replace(\"ls=ev(Tc,sel);loss=sum((x['loss'] if isinstance(x,dict) else float(x)) for x in ls)/len(ls)\",\"ls=ev(Tc,sel);loss=torch.stack([x['loss'] if isinstance(x,dict) else x for x in ls]).mean()\")
open(p,'w').write(s)
PY
python scratch/static_opt.py B 120"
```
<details><summary>output</summary>

```
0 sample 1.519592046737671 all 1.5621215956551688 best 1.5621215956551688 mask 0.3500704155081794 delta 0.0 0.0035680179
10 sample 2.1022303104400635 all 2.1063212269828435 best 1.5621215956551688 mask 0.6124206270490374 delta 0.0 0.018725818
20 sample 2.3242528438568115 all 2.4320146242777505 best 1.5621215956551688 mask 0.7762677442459833 delta 0.0 0.027342046
30 sample 2.4746646881103516 all 2.785772363344828 best 1.5621215956551688 mask 0.9674472070875622 delta 0.0 0.03510773
40 sample 3.216049909591675 all 3.045195363816761 best 1.5621215956551688 mask 1.1095702477863856 delta 0.0 0.0415943
50 sample 2.9418582916259766 all 3.1600355761391774 best 1.5621215956551688 mask 1.1724783380826314 delta 0.0 0.044521775
60 sample 3.3594436645507812 all 3.2695034799121676 best 1.5621215956551688 …[+796 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pose_opt.py';s=open(p).read();s=s.replace(\"ml=torch.nn.functional.binary_cross_entropy(soft.clamp(1e-5,1-1e-5),dto)\",\"ml=1.0-(2.0*(soft*dto).sum()+1e-4)/(soft.sum()+dto.sum()+1e-4)\")
open(p,'w').write(s)
PY
python scratch/static_opt.py B 120"
```
<details><summary>output</summary>

```
0 sample 1.1084046363830566 all 1.070409459727151 best 1.070409459727151 mask 0.1042143532208034 delta 0.0 0.0035680179
10 sample 1.219801425933838 all 1.2305967978068761 best 1.070409459727151 mask 0.17455842949095227 delta 0.0 0.018725818
20 sample 1.2984986305236816 all 1.316033851532709 best 1.070409459727151 mask 0.2182773578734625 delta 0.0 0.027342046
30 sample 1.3088629245758057 all 1.3899473973682948 best 1.070409459727151 mask 0.26953472126097905 delta 0.0 0.03510773
40 sample 1.4858168363571167 all 1.442566746757144 best 1.070409459727151 mask 0.30825595060984295 delta 0.0 0.0415943
50 sample 1.3937029838562012 all 1.4653400296256656 best 1.070409459727151 mask 0.3251305477959769 delta 0.0 0.044521775
60 sample 1.5501121282577515 all 1.493669072786967 best 1.070409459727151 mask …[+788 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
ids=list(range(205,306,5));base=so.makebase(ids,np.eye(3));opt=np.load('scratch/static_B.npy')
for i in [210,250,280,300]:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);p.save_overlay(i,base,crop,f'scratch/Bbase_{i}.png',4);p.save_overlay(i,opt,crop,f'scratch/Bopt_{i}.png',4)
 # bboxes via direct render
 for nm,T in [('b',base),('o',opt)]:
  rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV),crop,2); xy=np.where(cov.cpu().numpy()); print(i,nm,'bbox',xy[1].min()/2+crop[0],xy[1].max()/2+crop[0],xy[0].min()/2+crop[1],xy[0].max()/2+crop[1],'target',crop,obj[i].sum())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "/work/scratch/pose_opt.py", line 117, in save_overlay
    im,obj,per=get_frame(i);rgb,cov,(w,h)=render_crop(torch.tensor(Tc,device=DEV),crop,S)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_opt.py", line 27, in render_crop
    vc=V@Tc[:3,:3].T+Tc[:3,3]
       ~^~~~~~~~~~~~
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
ids=list(range(205,306,5));base=so.makebase(ids,np.eye(3)).astype(np.float32);opt=np.load('scratch/static_B.npy').astype(np.float32)
for i in [210,250,280,300]:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);p.save_overlay(i,base,crop,f'scratch/Bbase_{i}.png',4);p.save_overlay(i,opt,crop,f'scratch/Bopt_{i}.png',4)
 for nm,T in [('b',base),('o',opt)]:
  rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV),crop,2); xy=np.where(cov.cpu().numpy()); print(i,nm,'bbox',xy[1].min()/2+crop[0],xy[1].max()/2+crop[0],xy[0].min()/2+crop[1],xy[0].max()/2+crop[1],'target',crop,obj[i].sum())
PY"
```
<details><summary>output</summary>

```
210 b bbox 609.0 687.5 797.5 844.0 target (570, 772, 714, 869) 0
210 o bbox 609.5 688.0 797.0 843.0 target (570, 772, 714, 869) 0
250 b bbox 609.0 687.5 797.5 844.0 target (568, 775, 719, 868) 0
250 o bbox 609.5 688.0 797.0 843.0 target (568, 775, 719, 868) 0
280 b bbox 609.0 687.5 797.5 844.0 target (572, 777, 712, 867) 0
280 o bbox 609.5 688.0 797.0 843.0 target (572, 777, 712, 867) 0
300 b bbox 609.0 687.5 797.5 844.0 target (571, 768, 714, 870) 0
300 o bbox 609.5 688.0 797.0 843.0 target (571, 768, 714, 870) 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,io,numpy as np
from PIL import Image
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
for i in [0,120,220,420,650,850]:
 raw=f['aligned/front_stereo_camera_left'][i]; im=Image.open(io.BytesIO(raw.tobytes())); a=np.asarray(im).astype(np.float32)/1000
 print(i,a.shape,a.min(),a.max(),np.percentile(a,[1,10,50,90,99]),'center',a[576,768])
PY"
```
<details><summary>output</summary>

```
0 (1152, 1536) 1.812 10.078 [1.88499999 2.24000001 4.16900015 7.03000021 9.33899975] center 9.56
120 (1152, 1536) 1.806 10.043 [1.88100004 2.23699999 4.15700006 7.02099991 9.22999954] center 3.424
220 (1152, 1536) 1.748 9.595 [1.82200003 2.16700006 4.03700018 6.85099983 8.9289999 ] center 9.146
420 (1152, 1536) 1.819 10.331 [1.89699996 2.25399995 4.17500019 7.07499981 9.32699966] center 3.576
650 (1152, 1536) 1.784 9.703 [1.85800004 2.20600009 4.0999999  6.94799995 9.02999973] center 9.245
850 (1152, 1536) 1.827 10.171 [1.90600002 2.26699996 4.22650003 7.13899994 9.4659996 ] center 9.726

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/depth_eval.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,io,json;sys.path.insert(0,'scratch');import pose_opt as p
from PIL import Image
hf=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
for i in [0,60,100,120,150,180,220,280,350,390,420,470,500,550,600,650,700,750,800,850,900]:
 im,obj,per=p.get_frame(i);T=torch.tensor(p.BASEc[i],device=p.DEV)
 # render full
 vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
 pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None]
 r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));mask=r[0,:,:,3]>0;z=r[0,:,:,2]*10
 raw=hf['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000
 valid=obj&mask&~per;valid2=obj&mask
 if valid.sum()>20:
  diff=(z.cpu().numpy()[valid]-dep[valid])
  print(i,'n',valid.sum(),'median',np.median(diff),'mean',np.mean(diff),'abs',np.median(np.abs(diff)),'p10/90',np.percentile(diff,[10,90]),'depthobj',np.median(dep[valid]))
PY
python scratch/depth_eval.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_eval.py", line 11, in <module>
    valid=obj&mask&~per;valid2=obj&mask
          ~~~^~~~~
TypeError: unsupported operand type(s) for &: 'numpy.ndarray' and 'Tensor'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/mask=r\\[0,:,:\\:,3\\]>0;z=r\\[0,:,:\\:,2\\]\\*10/mask=(r[0,:,:,3]>0).cpu().numpy();z=(r[0,:,:,2]*10).cpu().numpy()/\" scratch/depth_eval.py
sed -i 's/z.cpu().numpy()\\[valid\\]/z[valid]/' scratch/depth_eval.py
python scratch/depth_eval.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_eval.py", line 11, in <module>
    valid=obj&mask&~per;valid2=obj&mask
          ~~~^~~~~
TypeError: unsupported operand type(s) for &: 'numpy.ndarray' and 'Tensor'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'mask=' scratch/depth_eval.py"
```
<details><summary>output</summary>

```
9: r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));mask=r[0,:,:,3]>0;z=r[0,:,:,2]*10

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/depth_eval.py';s=open(p).read();s=s.replace(\"mask=r[0,:,:,3]>0;z=r[0,:,:,2]*10\",\"mask=(r[0,:,:,3]>0).cpu().numpy();z=(r[0,:,:,2]*10).cpu().numpy()\");s=s.replace(\"z.cpu().numpy()[valid]\",\"z[valid]\");open(p,'w').write(s)
PY
python scratch/depth_eval.py"
```
<details><summary>output</summary>

```
0 n 1573 median -0.07591987 mean -0.17051606 abs 0.07591987 p10/90 [-0.15382128 -0.04433627] depthobj 3.151
60 n 1602 median -0.06961334 mean -0.12395084 abs 0.06961334 p10/90 [-0.13931355 -0.05562975] depthobj 3.1115
100 n 1736 median -0.04740846 mean -0.05382911 abs 0.04740846 p10/90 [-0.07931948 -0.02251673] depthobj 3.1415
120 n 832 median 0.10431564 mean 0.11003192 abs 0.10431564 p10/90 [0.0700768  0.15560155] depthobj 3.081
150 n 249 median -0.17472863 mean -0.5478364 abs 0.3814788 p10/90 [-1.87449942  0.13044567] depthobj 3.458
180 n 1779 median 0.34024525 mean 0.31744176 abs 0.34056616 p10/90 [0.25921822 0.38509412] depthobj 2.995
220 n 2095 median -0.11427331 mean -0.12098668 abs 0.11427331 p10/90 [-0.14356141 -0.10199776] depthobj 3.428
280 n 1949 median -0.07209158 mean -0.07619 …[+1476 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/depth_correct_eval.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,io;sys.path.insert(0,'scratch');import pose_opt as p
from PIL import Image
from scipy.ndimage import gaussian_filter
hf=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def rend(T):
 vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
 pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None]
 r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy(),(r[0,:,:,2]*10).cpu().numpy()
def iou(a,b):return (a&b).sum()/max(1,(a|b).sum())
for i in [0,30,60,80,100,120,150,180,200,220,250,280,300,350,370,390,420,450,470,480,500,530,550,570,590,610,630,650,700,740,760,790,810,830,850,880,900]:
 im,obj,per=p.get_frame(i);T=p.BASEc[i].copy();raw=hf['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000
 m,z=rend(torch.tensor(T,device=p.DEV));valid=obj&m&~per
 if valid.sum()<50:
  print(i,'skip',valid.sum());continue
 dz=np.median(dep[valid]-z[valid]);T2=T.copy();T2[2,3]+=dz
 m2,z2=rend(torch.tensor(T2,device=p.DEV));v2=obj&m2&~per
 # texture corr
 def tc(mm):
  crop=p.crop_bounds(obj);x0,y0,x1,y1=crop
  # render crop at native using full m/z only (approx crop from full)
  a=gaussian_filter(im[y0:y1,x0:x1].astype(np.float32)/255,(1.2,1.2,0));b=gaussian_filter(im[y0:y1,x0:x1].astype(np.float32)/255,(1.2,1.2,0))
  return 0
 print(i,'dz',round(d …[+164 chars]
```
<details><summary>output</summary>

```
0 dz 0.076 iou 0.785 -> 0.706 valid 1573 n 3.0689793 3.151
30 dz 0.142 iou 0.789 -> 0.636 valid 1596 n 3.0626163 3.212
60 dz 0.07 iou 0.792 -> 0.715 valid 1602 n 3.041483 3.1115
80 dz 0.014 iou 0.844 -> 0.841 valid 1570 n 3.0987363 3.124
100 dz 0.047 iou 0.776 -> 0.753 valid 1736 n 3.0892463 3.1415
120 dz -0.104 iou 0.757 -> 0.768 valid 832 n 3.1844585 3.081
150 dz 0.175 iou 0.118 -> 0.012 valid 249 n 3.2900186 3.458
180 dz -0.34 iou 0.756 -> 0.163 valid 1779 n 3.3385837 2.995
200 dz 0.129 iou 0.868 -> 0.476 valid 2094 n 3.2900586 3.423
220 dz 0.114 iou 0.864 -> 0.506 valid 2095 n 3.3197608 3.428
250 dz 0.084 iou 0.871 -> 0.578 valid 2094 n 3.3592863 3.438
280 dz 0.072 iou 0.827 -> 0.597 valid 1949 n 3.331046 3.402
300 dz 0.087 iou 0.85 -> 0.611 valid 2135 n 3.3472445 3.431
350 dz -0.477 i …[+1441 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'pos=torch.stack' -A2 scratch/pose_opt.py scratch/depth_eval.py"
```
<details><summary>output</summary>

```
scratch/pose_opt.py:30: pos=torch.stack([2*x/w-1,2*y/h-1,vc[:,2]/10,torch.ones_like(x)],-1)[None]
scratch/pose_opt.py-31- r,interp=dr.rasterize(gl,pos,F,(h,w))
scratch/pose_opt.py-32- uvq=dr.interpolate(UV[None],r,F)[0]
--
scratch/depth_eval.py:8: pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None]
scratch/depth_eval.py-9- r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));mask=(r[0,:,:,3]>0).cpu().numpy();z=(r[0,:,:,2]*10).cpu().numpy()
scratch/depth_eval.py-10- raw=hf['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2,h5py,io;sys.path.insert(0,'scratch');import pose_opt as p
from PIL import Image
im,obj,per=p.get_frame(250);T=p.BASEc[i] if False else p.BASEc[250]
# brute z shifts and bbox/iou
for dz in [-.2,-.1,0,.05,.1,.15,.2]:
 Q=T.copy();Q[2,3]+=dz
 vc=p.V@torch.tensor(Q,device=p.DEV,dtype=torch.float32)[:3,:3].T+torch.tensor(Q,device=p.DEV,dtype=torch.float32)[:3,3]
 x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
 pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None]
 r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));m=(r[0,:,:,3]>0).cpu().numpy();yy,xx=np.where(m)
 print(dz,xx.min(),xx.max(),yy.min(),yy.max(),(m&obj).sum()/(m|obj).sum())
PY"
```
<details><summary>output</summary>

```
-0.2 599 681 812 861 0.37117780294450736
-0.1 604 684 805 852 0.6416754663850757
0 609 686 798 843 0.8706860706860707
0.05 611 687 795 839 0.6897210546427207
0.1 613 688 791 834 0.5331452750352609
0.15 615 689 788 830 0.4113965744400527
0.2 617 690 785 826 0.3112562189054726

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py,io;sys.path.insert(0,'scratch');import pose_opt as p
from PIL import Image
def rend(T):
 vc=p.V@torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,:3].T+torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
 pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy(),(r[0,:,:,2]*10).cpu().numpy()
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
for i in [0,30,60,80,100,120,180,200,220,250,280,300,350,390,420,450,470,500,530,550,570,630,650,700,740,760,790,810,850,900]:
 im,obj,per=p.get_frame(i);T=p.BASEc[i];m,z=rend(T);valid=obj&m&~per
 if valid.sum()<50: print(i,'skip');continue
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000
 # robust depth shift and scale translation around origin
 dz=np.median(dep[valid]-z[valid]); scale=(T[2,3]+dz)/T[2,3]
 Q=T.copy();Q[:3,3]*=scale;m2,z2=rend(Q)
 print(i,'dz',round(dz,3),'scale',round(scale,3),'iou',round((m&obj).sum()/(m|obj).sum(),3),'->',round((m2&obj).sum()/(m2|obj).sum(),3),'ctr',np.mean(np.where(m)[1]),np.mean(np.where(obj)[1]))
PY"
```
<details><summary>output</summary>

```
0 dz 0.076 scale 1.024 iou 0.785 -> 0.77 ctr 812.7140255009108 809.5272161741835
30 dz 0.142 scale 1.045 iou 0.789 -> 0.75 ctr 812.7144585601936 810.4465920651069
60 dz 0.07 scale 1.022 iou 0.792 -> 0.776 ctr 812.795589988081 810.2381195707716
80 dz 0.014 scale 1.005 iou 0.844 -> 0.839 ctr 815.8777915632754 813.7081733406472
100 dz 0.047 scale 1.015 iou 0.776 -> 0.766 ctr 807.1065934065934 807.0151724137932
120 dz -0.104 scale 0.968 iou 0.757 -> 0.782 ctr 724.4496439471008 720.5475352112676
180 dz -0.34 scale 0.9 iou 0.756 -> 0.699 ctr 654.8420065635255 647.8529264632316
200 dz 0.129 scale 1.039 iou 0.868 -> 0.843 ctr 646.8938793483047 644.8108831400535
220 dz 0.114 scale 1.034 iou 0.864 -> 0.838 ctr 647.3112467306016 643.965038099507
250 dz 0.084 scale 1.025 iou 0.871 -> 0.851 ctr 646.979 …[+1683 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/depth_center_eval.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,io;sys.path.insert(0,'scratch');import pose_opt as p
from PIL import Image
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def rend(T):
 vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
 pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy(),(r[0,:,:,2]*10).cpu().numpy()
def align(T,obj,dep,do_depth=True,iters=3):
 for it in range(iters):
  m,z=rend(T);valid=obj&m
  if valid.sum()<30:break
  if do_depth:
   dz=np.median(dep[valid]-z[valid]);scale=(T[2,3]+dz)/T[2,3];T[:3,3]*=scale
  m,z=rend(T); yy,xx=np.where(m); ty,tx=np.where(obj)
  dx=np.median(tx)-np.median(xx);dy=np.median(ty)-np.median(yy);zc=T[2,3]
  T[0,3]+=dx*zc/p.K[0,0];T[1,3]+=dy*zc/p.K[1,1]
 return T
for i in [0,30,60,80,100,120,180,200,220,250,280,300,350,390,420,450,470,500,530,550,570,630,650,700,740,760,790,810,850,900]:
 im,obj,per=p.get_frame(i);raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000
 T=p.BASEc[i];m,z=rend(T);i0=(m&obj).sum()/max(1,(m|obj).sum())
 T1=align(T.copy(),obj,dep,True);m1,z1=rend(T1);i1=(m1&obj).sum()/max(1,(m1|obj).sum())
 T2=align(T.copy(),obj,dep,False);m2,z2=rend(T2);i2=(m2&obj).sum()/max(1,(m2|obj).sum())
 print(i,'iou',round(i0,3),'depth+',round(i1,3),'cente …[+118 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_center_eval.py", line 19, in <module>
    T=p.BASEc[i];m,z=rend(T);i0=(m&obj).sum()/max(1,(m|obj).sum())
                     ^^^^^^^
  File "/work/scratch/depth_center_eval.py", line 5, in rend
    vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
       ~~~^~~~~~~~~~~
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/depth_center_eval.py';s=open(p).read();s=s.replace(\"def rend(T):\\n vc=p.V@T[:3,:3].T+T[:3,3]\",\"def rend(T):\\n T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3]\");open(p,'w').write(s)
PY
python scratch/depth_center_eval.py"
```
<details><summary>output</summary>

```
0 iou 0.785 depth+ 0.823 center 0.855 dz 0.066 dt [-0.007  0.012  0.066]
30 iou 0.789 depth+ 0.776 center 0.842 dz 0.129 dt [0.001 0.016 0.129]
60 iou 0.792 depth+ 0.821 center 0.856 dz 0.068 dt [-0.007  0.012  0.068]
80 iou 0.844 depth+ 0.832 center 0.839 dz 0.021 dt [-0.01   0.005  0.021]
100 iou 0.776 depth+ 0.799 center 0.815 dz 0.039 dt [-0.002  0.014  0.039]
120 iou 0.757 depth+ 0.818 center 0.811 dz -0.104 dt [-0.014 -0.007 -0.104]
180 iou 0.756 depth+ 0.739 center 0.822 dz -0.331 dt [ 0.015 -0.092 -0.331]
200 iou 0.868 depth+ 0.868 center 0.882 dz 0.117 dt [-0.024  0.04   0.117]
220 iou 0.864 depth+ 0.877 center 0.881 dz 0.108 dt [-0.027  0.038  0.108]
250 iou 0.871 depth+ 0.884 center 0.894 dz 0.079 dt [-0.019  0.029  0.079]
280 iou 0.827 depth+ 0.858 center 0.854 dz 0.064 dt [-0. …[+1438 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
ids=list(range(205,306,5));base=torch.tensor(so.makebase(ids,np.eye(3)).astype(np.float32),device=p.DEV)
w=torch.zeros(3,device=p.DEV,requires_grad=True);t=torch.zeros(3,device=p.DEV,requires_grad=True)
Tc=p.makeT(w,t,base)
for i in [220,250,280,300]:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);l=p.losses(Tc,crop,2,im,obj,per);print(i,l.item());l.backward(retain_graph=True);print('wgrad',w.grad,'tgrad',t.grad);w.grad.zero_();t.grad.zero_()
PY"
```
<details><summary>output</summary>

```
220 1.0382840633392334
wgrad None tgrad tensor([-3.8868, 11.3608, -4.3201], device='cuda:0')Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
AttributeError: 'NoneType' object has no attribute 'zero_'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pose_opt.py';s=open(p).read();s=s.replace(\"Wm=torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],device=w.device)\",\"z=torch.zeros((),device=w.device,dtype=w.dtype)\\n Wm=torch.stack([torch.stack([z,-w[2],w[1]]),torch.stack([w[2],z,-w[0]]),torch.stack([-w[1],w[0],z])])\")
open(p,'w').write(s)
PY
python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
ids=list(range(205,306,5));base=torch.tensor(so.makebase(ids,np.eye(3)).astype(np.float32),device=p.DEV)
w=torch.zeros(3,device=p.DEV,requires_grad=True);t=torch.zeros(3,device=p.DEV,requires_grad=True)
Tc=p.makeT(w,t,base);im,obj,per=p.get_frame(250);crop=p.crop_bounds(obj);l=p.losses(Tc,crop,2,im,obj,per);l.backward();print(l.item(),'wgrad',w.grad,'tgrad',t.grad)
PY"
```
<details><summary>output</summary>

```
0.900044322013855 wgrad tensor([-44.9237,  -8.8679,  -4.4221], device='cuda:0') tgrad tensor([-2.0052, 12.6163, -4.3808], device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python scratch/static_opt.py B 120'
```
<details><summary>output</summary>

```
0 sample 1.1084046363830566 all 1.187195931162153 best 1.187195931162153 mask 0.15754002332687378 delta 0.20558422779686011 0.013115256
10 sample 2.195207118988037 all 2.227099447023301 best 1.187195931162153 mask 0.5252686256454104 delta 1.030975688094453 0.08437742
20 sample 2.4587199687957764 all 2.502074423290434 best 1.187195931162153 mask 0.6030396081152416 delta 1.249604951985495 0.10239027
30 sample 2.0881195068359375 all 2.1353652534030734 best 1.187195931162153 mask 0.487080142611549 delta 0.9719670057632895 0.07952616
40 sample 2.038975954055786 all 1.9793727681750344 best 1.187195931162153 mask 0.4754371444384257 delta 0.8499580917804045 0.06777537
50 sample 1.8685290813446045 all 1.9366192987986974 best 1.187195931162153 mask 0.522361193384443 delta 0.8847019703970608 0.064667 …[+962 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/static_rot_opt.py <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
def align_center(T,obj):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32)
 vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
 pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));m=(r[0,:,:,3]>0).cpu().numpy()
 if m.sum()==0:return T.cpu().numpy()
 yy,xx=np.where(m);ty,tx=np.where(obj);dx=np.median(tx)-np.median(xx);dy=np.median(ty)-np.median(yy);Q=T.cpu().numpy().copy();Q[0,3]+=dx*T[2,3].item()/p.K[0,0];Q[1,3]+=dy*T[2,3].item()/p.K[1,1];return Q
def run(ids,S=np.eye(3),steps=150):
 base=so.makebase(ids,S);cache={}
 def data(i):
  if i not in cache:
   im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);cache[i]=(im,obj,per,crop)
  return cache[i]
 valid=[i for i in ids if data(i)[3] is not None and (data(i)[1]&~data(i)[2]).sum()>100]
 # center align base on representative middle frame
 mid=valid[len(valid)//2];base=align_center(base,data(mid)[1])
 b=torch.tensor(base,device=p.DEV);w=torch.zeros(3,device=p.DEV,requires_grad=True);t=torch.zeros(3,device=p.DEV,requires_grad=True)
 opt=torch.optim.Adam([w],lr=.001)
 def ev(dbg=False):
  vals=[]
  with torch.no_grad():
   Tc=p.makeT(w,t,b)
   for i in valid:
    im,obj,per,crop=data(i);vals.append(p.losses(Tc,crop,2,im,obj,per,dbg))
  return vals
 best=(1e9, …[+1204 chars]
```
<details><summary>output</summary>

```
0 1.2047 best 1.2047 delta 0.1 ncc 0.184 mask 0.089 iou 0.84
10 1.2154 best 1.2047 delta 0.31 ncc 0.314 mask 0.169 iou 0.712
20 1.2364 best 1.2047 delta 0.46 ncc 0.309 mask 0.178 iou 0.696
30 1.3291 best 1.2047 delta 0.65 ncc 0.279 mask 0.209 iou 0.654
40 1.416 best 1.2047 delta 0.79 ncc 0.286 mask 0.256 iou 0.593
50 1.4308 best 1.2047 delta 0.81 ncc 0.291 mask 0.266 iou 0.581
60 1.4902 best 1.2047 delta 0.88 ncc 0.34 mask 0.323 iou 0.512
70 1.6079 best 1.2047 delta 1.0 ncc 0.341 mask 0.382 iou 0.448
80 1.8386 best 1.2047 delta 1.15 ncc 0.271 mask 0.458 iou 0.371
90 1.9514 best 1.2047 delta 1.24 ncc 0.229 mask 0.491 iou 0.341
100 2.0245 best 1.2047 delta 1.3 ncc 0.204 mask 0.514 iou 0.322
110 2.0308 best 1.2047 delta 1.31 ncc 0.2 mask 0.515 iou 0.322
final 1.2046915179207212

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr
gl=dr.RasterizeCudaContext();f=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
for yn in [-.5,0,.5]:
 pos=torch.tensor([[[-.2,yn,0,1],[.2,yn,0,1],[0,yn-.2,0,1]]],device='cuda',dtype=torch.float32)
 r,_=dr.rasterize(gl,pos,f,(100,100));m=r[0,:,:,3]>0;y,x=torch.where(m);print(yn,y.float().mean().item(),y.min().item(),y.max().item())
PY"
```
<details><summary>output</summary>

```
-0.5 21.149999618530273 15 24
0 46.14999771118164 40 49
0.5 71.1500015258789 65 74

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
ids=list(range(205,306,5));base=torch.tensor(so.makebase(ids,np.eye(3)).astype(np.float32),device=p.DEV)
im,obj,per=p.get_frame(250);crop=p.crop_bounds(obj)
w=torch.zeros(3,device=p.DEV,requires_grad=True);t=torch.zeros(3,device=p.DEV,requires_grad=True)
for dy in [-.02,-.01,0,.01,.02]:
 t2=torch.tensor([0,dy,0],device=p.DEV);Tc=p.makeT(w,t2,base);print('dy',dy,p.losses(Tc,crop,2,im,obj,per,True))
t.grad=None;Tc=p.makeT(w,t,base);l=p.losses(Tc,crop,2,im,obj,per);l.backward();print('loss',l.item(),'grad',t.grad)
PY"
```
<details><summary>output</summary>

```
dy -0.02 {'loss': 1.3285460472106934, 'mask': 0.21167320013046265, 'ncc': 0.28883349895477295, 'l1': 0.9701656103134155, 'iou': 0.6501650165016502}
dy -0.01 {'loss': 1.127051591873169, 'mask': 0.1356501579284668, 'ncc': 0.3334546387195587, 'l1': 0.9460294246673584, 'iou': 0.7594689574385006}
dy 0 {'loss': 0.900044322013855, 'mask': 0.07456284761428833, 'ncc': 0.43096980452537537, 'l1': 0.9094423651695251, 'iou': 0.857201646090535}
dy 0.01 {'loss': 1.2825202941894531, 'mask': 0.07536917924880981, 'ncc': 0.08310316503047943, 'l1': 1.0744253396987915, 'iou': 0.860608552631579}
dy 0.02 {'loss': 1.5913602113723755, 'mask': 0.14468854665756226, 'ncc': -0.0794789120554924, 'l1': 1.1125210523605347, 'iou': 0.7482652274479569}
loss 0.900044322013855 grad tensor([-2.0052, 12.6163, -4.3808], device=' …[+9 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
ids=list(range(205,306,5));base=torch.tensor(so.makebase(ids,np.eye(3)).astype(np.float32),device=p.DEV)
im,obj,per=p.get_frame(250);crop=p.crop_bounds(obj)
for ax in range(3):
 print('axis',ax)
 for deg in [-2,-1,-.5,0,.5,1,2]:
  w=torch.zeros(3,device=p.DEV);w[ax]=np.radians(deg);Tc=p.makeT(w,torch.zeros(3,device=p.DEV),base);print(deg,round(p.losses(Tc,crop,2,im,obj,per,True)['loss'],3),end=' ')
 print()
PY"
```
<details><summary>output</summary>

```
axis 0
-2 3.133 -1 2.451 -0.5 1.806 0 0.9 0.5 1.455 1 1.878 2 3.122 
axis 1
-2 1.751 -1 1.392 -0.5 1.252 0 0.9 0.5 1.388 1 1.811 2 2.499 
axis 2
-2 1.796 -1 1.411 -0.5 1.111 0 0.9 0.5 1.035 1 1.175 2 1.33 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/static_rot_opt.py';s=open(p).read();s=s.replace(\"opt=torch.optim.Adam([w],lr=.001)\",\"opt=torch.optim.Adam([w],lr=.0002)\");s=s.replace(\"for k in range(steps):\\n  sel=\", \"for k in range(steps):\\n  opt.param_groups[0]['lr']=.0002*(1-k/steps)+.00001\\n  sel=\");open(p,'w').write(s)
PY
python scratch/static_rot_opt.py B 100"
```
<details><summary>output</summary>

```
0 1.2199 best 1.2199 delta 0.04 ncc 0.157 mask 0.08 iou 0.854
10 1.1571 best 1.1571 delta 0.07 ncc 0.213 mask 0.08 iou 0.853
20 1.0786 best 1.0786 delta 0.14 ncc 0.296 mask 0.089 iou 0.839
30 1.0619 best 1.0619 delta 0.19 ncc 0.319 mask 0.093 iou 0.83
40 1.0774 best 1.0619 delta 0.24 ncc 0.319 mask 0.101 iou 0.815
50 1.1076 best 1.0619 delta 0.29 ncc 0.304 mask 0.108 iou 0.801
60 1.141 best 1.0619 delta 0.33 ncc 0.285 mask 0.114 iou 0.793
70 1.164 best 1.0619 delta 0.35 ncc 0.273 mask 0.119 iou 0.782
80 1.1705 best 1.0619 delta 0.36 ncc 0.273 mask 0.123 iou 0.779
90 1.1662 best 1.0619 delta 0.36 ncc 0.277 mask 0.123 iou 0.779
final 1.0619217526345026

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python scratch/static_rot_opt.py D 130'
```
<details><summary>output</summary>

```
0 1.5036 best 1.5036 delta 0.03 ncc 0.118 mask 0.202 iou 0.683
10 1.5646 best 1.5036 delta 0.04 ncc 0.072 mask 0.206 iou 0.677
20 1.6332 best 1.5036 delta 0.06 ncc 0.018 mask 0.21 iou 0.672
30 1.5455 best 1.5036 delta 0.07 ncc 0.09 mask 0.207 iou 0.676
40 1.5765 best 1.5036 delta 0.07 ncc 0.066 mask 0.209 iou 0.672
50 1.5776 best 1.5036 delta 0.08 ncc 0.065 mask 0.21 iou 0.672
60 1.578 best 1.5036 delta 0.08 ncc 0.065 mask 0.209 iou 0.672
70 1.5795 best 1.5036 delta 0.07 ncc 0.064 mask 0.21 iou 0.672
80 1.5807 best 1.5036 delta 0.07 ncc 0.062 mask 0.209 iou 0.672
90 1.5791 best 1.5036 delta 0.07 ncc 0.064 mask 0.209 iou 0.672
100 1.5764 best 1.5036 delta 0.08 ncc 0.067 mask 0.21 iou 0.671
110 1.5777 best 1.5036 delta 0.08 ncc 0.065 mask 0.21 iou 0.671
120 1.5768 best 1.5036 delta 0.08 ncc  …[+50 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3]
def rel(a,b):return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))
for i in range(740,901,10):print(i,round(rel(R[740],R[i]),1),round(rel(R[i-10],R[i]),1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
for i in [630,650,700,740]:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);outs=[]
 for nm,S in [('I',np.eye(3)),('Rx',np.diag([1,-1,-1])),('Rz',np.diag([-1,-1,1]))]:
  Q=p.BASEc[i].copy();Q[:3,:3]=Q[:3,:3]@S;p.save_overlay(i,Q,crop,f'scratch/D_{i}_{nm}.png',5);o=cv2.imread(f'scratch/D_{i}_{nm}.png');o=cv2.resize(o,(400,250));cv2.putText(o,nm,(10,28),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3);cv2.putText(o,nm,(10,28),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1);outs.append(o)
 cv2.imwrite(f'scratch/Dcmp_{i}.jpg',np.concatenate(outs,1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
for i in [650,700]:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;S=5
 raw=cv2.resize(im[y0:y1,x0:x1],((x1-x0)*S,(y1-y0)*S),interpolation=cv2.INTER_CUBIC);outs=[raw]
 for nm,SS in [('I',np.eye(3)),('Rx',np.diag([1,-1,-1])),('Rz',np.diag([-1,-1,1]))]:
  Q=p.BASEc[i].copy();Q[:3,:3]=Q[:3,:3]@SS;rgb,cov,_=p.render_crop(torch.tensor(Q,device=p.DEV,dtype=torch.float32),crop,S);r=(rgb.cpu().numpy()*255).astype(np.uint8)[:,:,::-1];m=cov.cpu().numpy();o=np.zeros_like(r);o[m]=r[m];cv2.putText(o,nm,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);outs.append(o)
 cv2.imwrite(f'scratch/Draw_{i}.jpg',np.concatenate(outs,1))
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
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))]
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];pl=np.array(json.load(open('inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
for ph,ids,S in [('A',range(0,61,10),np.eye(3)),('B',range(200,301,10),np.eye(3)),('C',range(400,451,10),np.eye(3)),('D',range(650,741,10),np.diag([1,-1,-1])),('E',range(850,901,10),np.eye(3))]:
 print('\\n',ph)
 for i in ids:
  R=T[i].copy();R[:3,:3]=R[:3,:3]@S;p=(V@R.T)[:,:3];sd=p@n+d
  print(i,'min',round(sd.min(),4),'p1',round(np.percentile(sd,1),4),'max',round(sd.max(),4))
PY"
```
<details><summary>output</summary>

```

 A
0 min 0.8933 p1 0.8945 max 0.9596
10 min 0.8932 p1 0.8944 max 0.9595
20 min 0.8934 p1 0.8946 max 0.9597
30 min 0.8941 p1 0.8954 max 0.9604
40 min 0.8963 p1 0.8975 max 0.9626
50 min 0.8965 p1 0.8977 max 0.9628
60 min 0.8968 p1 0.898 max 0.9631

 B
200 min 0.037 p1 0.0385 max 0.1126
210 min 0.0006 p1 0.0025 max 0.0812
220 min -0.0042 p1 -0.0021 max 0.0759
230 min -0.0042 p1 -0.0021 max 0.0759
240 min -0.0046 p1 -0.0027 max 0.0709
250 min -0.0168 p1 -0.0149 max 0.0587
260 min -0.0056 p1 -0.0042 max 0.0629
270 min -0.0107 p1 -0.0094 max 0.0569
280 min -0.0074 p1 -0.0061 max 0.0603
290 min -0.0025 p1 -0.0012 max 0.0651
300 min -0.0173 p1 -0.0147 max 0.081

 C
400 min 0.8871 p1 0.8878 max 0.9522
410 min 0.8875 p1 0.8879 max 0.9486
420 min 0.887 p1 0.8874 max 0.9467
430 min 0.8904 p1 0.8911 m …[+661 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))];T=np.load('inputs/reconstruction.npz')['T_world_mesh'];pl=np.array(json.load(open('inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
for ph,ids in [('B',range(210,301,20)),('D',range(650,741,20))]:
 print(ph)
 for S in [('I',np.eye(3)),('Rx',np.diag([1,-1,-1])),('Ry',np.diag([-1,1,-1])),('Rz',np.diag([-1,-1,1]))]:
  vals=[]
  for i in ids:
   Q=T[i].copy();Q[:3,:3]=Q[:3,:3]@S[1];p=(V@Q.T)[:,:3];vals.append((p@n+d).min())
  print(S[0],np.round(vals,3))
PY"
```
<details><summary>output</summary>

```
B
I [ 0.001 -0.004 -0.017 -0.011 -0.002]
Rx [ 0.001 -0.004 -0.016 -0.01  -0.002]
Ry [ 0.001 -0.004 -0.016 -0.011 -0.002]
Rz [ 0.001 -0.004 -0.016 -0.011 -0.002]
D
I [0.021 0.004 0.017 0.013 0.022]
Rx [0.021 0.004 0.018 0.013 0.022]
Ry [0.021 0.004 0.018 0.013 0.022]
Rz [0.02  0.004 0.017 0.012 0.022]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))];T=np.load('inputs/reconstruction.npz')['T_world_mesh'];n,d=np.array(json.load(open('inputs/ground_plane.json'))['plane'])
for ph,ids,S in [('A',range(0,61,10),np.eye(3)),('C',range(400,451,10),np.eye(3)),('E',range(850,901,10),np.eye(3))]:
 print('\\n',ph)
 for i in ids:
  Q=T[i].copy();Q[:3,:3]=Q[:3,:3]@S;p=(V@Q.T)[:,:3];print(i,'miny',p[:,1].min(),'minplane', (p@n+d).min(),'center',Q[:3,3])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))];T=np.load('inputs/reconstruction.npz')['T_world_mesh'];pl=np.array(json.load(open('inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
for ph,ids,S in [('A',range(0,61,10),np.eye(3)),('C',range(400,451,10),np.eye(3)),('E',range(850,901,10),np.eye(3))]:
 print('\\n',ph)
 for i in ids:
  Q=T[i].copy();Q[:3,:3]=Q[:3,:3]@S;p=(V@Q.T)[:,:3];print(i,'miny',round(p[:,1].min(),4),'minplane',round((p@n+d).min(),4),'center',np.round(Q[:3,3],3))
PY"
```
<details><summary>output</summary>

```

 A
0 miny 0.2425 minplane 0.8933 center [0.375 0.284 3.108]
10 miny 0.2426 minplane 0.8932 center [0.375 0.284 3.109]
20 miny 0.2425 minplane 0.8934 center [0.375 0.284 3.108]
30 miny 0.242 minplane 0.8941 center [0.374 0.283 3.102]
40 miny 0.2408 minplane 0.8963 center [0.373 0.282 3.085]
50 miny 0.2407 minplane 0.8965 center [0.372 0.282 3.084]
60 miny 0.2405 minplane 0.8968 center [0.372 0.282 3.081]

 C
400 miny 0.2501 minplane 0.8871 center [0.275 0.284 3.263]
410 miny 0.2497 minplane 0.8875 center [0.282 0.286 3.257]
420 miny 0.2514 minplane 0.887 center [0.275 0.288 3.243]
430 miny 0.2467 minplane 0.8904 center [0.275 0.284 3.245]
440 miny 0.2424 minplane 0.8911 center [0.274 0.282 3.243]
450 miny 0.248 minplane 0.8866 center [0.279 0.286 3.276]

 E
850 miny 0.3248 minplane 0.7776  …[+340 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
cap=cv2.VideoCapture('inputs/video.mp4')
for i in [830,850,900]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();cr=im[500:760,500:900];cv2.imwrite(f'scratch/wide_{i}.jpg',cr)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/table_plane.py <<'PY'
import numpy as np,h5py,io,json,h5py,cv2
from PIL import Image
from sklearn.linear_model import RANSACRegressor
H,W=1152,1536;c=json.load(open('inputs/camera.json'));Wc=np.array(c['T_world_cam']);K=np.array(c['K_pipeline'])
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r');mf=h5py.File('inputs/masks.h5','r')
for i in [0,20,40,200,250,300,400,420,450,650,700,740,850,870,900]:
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000
 obj=mf['object'][i];per=mf['person'][i];y,x=np.where(obj)
 if len(x)==0:continue
 cx,cy=int(np.median(x)),int(np.median(y));x0,x1=max(0,cx-180),min(W,cx+180);y0,y1=max(0,cy-100),min(H,cy+100)
 yy,xx=np.mgrid[y0:y1,x0:x1];d=dep[y0:y1,x0:x1];m=(~obj[y0:y1,x0:x1])&(~per[y0:y1,x0:x1])&(d>.5)&(d<8)
 xc=(xx[m]-K[0,2])/K[0,0]*d[m];yc=(yy[m]-K[1,2])/K[1,1]*d[m];pc=np.stack([xc,yc,d[m]],1);pw=pc@Wc[:3,:3].T+Wc[:3,3]
 # RANSAC plane via z as function x,y after select near horizontal normal y
 # fit nl
 import open3d as o3d
 pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pw)); plane,inds=pcd.segment_plane(.01,3,1000)
 n=np.array(plane[:3]);dd=plane[3];n=n/np.linalg.norm(n);dd/=np.linalg.norm(plane[:3])
 if n[1]<0:n=-n;dd=-dd
 # distance object center
 print(i,'npts',m.sum(),'in',len(inds),'plane',np.round(plane,4),'normalized n',np.round(n,3),'d',round(dd,3),'objctr dist',round(n@np.array(c['T_world_cam'])[:3,:3]@np.array([0,0,0]) …[+40 chars]
```
<details><summary>output</summary>

```
0 npts 40076 in 12123 plane [ 0.0254  0.995   0.0963 -1.5434] normalized n [0.025 0.995 0.096] d -1.543 objctr dist 0.0
20 npts 39964 in 11802 plane [ 0.0287  0.9946  0.1    -1.5656] normalized n [0.029 0.995 0.1  ] d -1.566 objctr dist 0.0
40 npts 38575 in 12162 plane [ 0.0296  0.9951  0.0945 -1.5443] normalized n [0.03  0.995 0.094] d -1.544 objctr dist 0.0
200 npts 62197 in 55224 plane [ 0.0319  0.9948  0.0968 -1.494 ] normalized n [0.032 0.995 0.097] d -1.494 objctr dist 0.0
250 npts 62299 in 55276 plane [ 0.0332  0.9947  0.0974 -1.495 ] normalized n [0.033 0.995 0.097] d -1.495 objctr dist 0.0
300 npts 62313 in 55104 plane [ 0.0327  0.9947  0.0977 -1.4941] normalized n [0.033 0.995 0.098] d -1.494 objctr dist 0.0
400 npts 43673 in 10216 plane [ 0.0268  0.995   0.0963 -1.5759] normaliz …[+1025 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/table_plane2.py <<'PY'
import numpy as np,h5py,io,json,cv2,open3d as o3d
from PIL import Image
H,W=1152,1536;c=json.load(open('inputs/camera.json'));Wc=np.array(c['T_world_cam']);K=np.array(c['K_pipeline'])
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r');mf=h5py.File('inputs/masks.h5','r')
for i in [0,20,40,400,420,450,830,850,870,900]:
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000
 obj=mf['object'][i];per=mf['person'][i];y,x=np.where(obj)
 if len(x)==0:continue
 cx,cy=int(np.median(x)),int(np.median(y));x0,x1=max(0,cx-250),min(W,cx+250);y0,y1=max(0,cy-80),min(H,cy+140)
 yy,xx=np.mgrid[y0:y1,x0:x1];d=dep[y0:y1,x0:x1];m=(~obj[y0:y1,x0:x1])&(~per[y0:y1,x0:x1])&(d>.5)&(d<8)
 xc=(xx[m]-K[0,2])/K[0,0]*d[m];yc=(yy[m]-K[1,2])/K[1,1]*d[m];pc=np.stack([xc,yc,d[m]],1);pw=pc@Wc[:3,:3].T+Wc[:3,3]
 # only above floor and near expected table top; use y<0.8
 sel=pw[:,1]<0.8;pw=pw[sel]
 if len(pw)<100:print(i,'few',len(pw));continue
 pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pw));plane,inds=pcd.segment_plane(.015,3,1000)
 n=np.array(plane[:3]);dd=plane[3];nn=np.linalg.norm(plane[:3]);n/=nn;dd/=nn
 if n[1]<0:n=-n;dd=-dd
 # Table plane must have n largely vertical and near object center y
 print(i,'pts',len(pw),'in',len(inds),'n',np.round(n,3),'d',round(dd,3),'y at objxy',round(-(n[0]*np.median(pw[:,0])+n[2]*np.median(pw[:,2])+dd)/n[1],3))
PY
python scratch/table_plane …[+5 chars]
```
<details><summary>output</summary>

```
0 pts 40996 in 10277 n [0.055 0.996 0.067] d -0.507 y at objxy 0.02
20 pts 41108 in 10453 n [0.046 0.996 0.071] d -0.517 y at objxy 0.005
40 pts 40196 in 10270 n [0.029 0.997 0.075] d -0.529 y at objxy -0.003
400 pts 43287 in 11107 n [0.037 0.997 0.072] d -0.517 y at objxy 0.019
420 pts 44644 in 10887 n [0.041 0.997 0.067] d -0.5 y at objxy 0.033
450 pts 42990 in 10870 n [0.06  0.996 0.065] d -0.502 y at objxy 0.055
830 pts 46313 in 11358 n [0.025 0.997 0.069] d -0.51 y at objxy 0.089
850 pts 45360 in 11581 n [0.041 0.997 0.072] d -0.523 y at objxy 0.094
870 pts 45330 in 11519 n [0.03  0.997 0.074] d -0.526 y at objxy 0.076
900 pts 45165 in 11372 n [0.031 0.995 0.093] d -0.574 y at objxy 0.02

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/table_plane2.py';s=open(p).read();s=s.replace(\"sel=pw[:,1]<0.8;pw=pw[sel]\",\"# restrict near object world z and plausible tabletop depth\\n zc=Wc[:3,:3]@np.array([0,0,0]) if False else None\\n sel=(pw[:,1]<0.8)&(pw[:,1]>-0.2);pw=pw[sel]\")
open(p,'w').write(s)
PY
# Instead use known object bbox world via baseline, simpler rewrite quick
python - <<'PY'
import numpy as np,h5py,io,json,cv2,open3d as o3d
from PIL import Image
c=json.load(open('inputs/camera.json'));Wc=np.array(c['T_world_cam']);K=np.array(c['K_pipeline']);h=h5py.File('/pipeline/depth/moge2/aligned.h5','r');mf=h5py.File('inputs/masks.h5','r');T=np.load('inputs/reconstruction.npz')['T_world_mesh']
for i in [0,20,400,420,850,870,900]:
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000;obj=mf['object'][i];per=mf['person'][i]
 yy,xx=np.mgrid[550:720,450:1000];d=dep[550:720,450:1000];m=(~obj[550:720,450:1000])&(~per[550:720,450:1000])&(d>.5)&(d<8);xc=(xx[m]-K[0,2])/K[0,0]*d[m];yc=(yy[m]-K[1,2])/K[1,1]*d[m];pw=np.stack([xc,yc,d[m]],1)@Wc[:3,:3].T+Wc[:3,3]
 ctr=T[i,:3,3];sel=(np.abs(pw[:,0]-ctr[0])<.6)&(np.abs(pw[:,2]-ctr[2])<.6)&(pw[:,1]<.8)&(pw[:,1]>-.1);pw=pw[sel]
 pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pw));plane,inds=pcd.segment_plane(.008,3,2000);n=np.array(plane[:3]);dd=plane[3];nn=np.linalg.norm(n);n/=nn;dd/=nn
 if n[1]<0:n=-n;dd=-dd
 print(i,'ctr',np.round(ctr,2),'pts',len(pw),' …[+100 chars]
```
<details><summary>output</summary>

```
0 ctr [0.38 0.28 3.11] pts 12492 in 7629 n [0.031 0.995 0.094] d -0.579 y 0.278
20 ctr [0.38 0.28 3.11] pts 12428 in 7597 n [0.032 0.995 0.092] d -0.574 y 0.278
400 ctr [0.28 0.28 3.26] pts 11888 in 7017 n [0.033 0.996 0.088] d -0.562 y 0.265
420 ctr [0.28 0.29 3.24] pts 11874 in 6955 n [0.036 0.995 0.091] d -0.564 y 0.261
850 ctr [-0.09  0.38  3.46] pts 6147 in 4697 n [0.032 0.996 0.085] d -0.557 y 0.267
870 ctr [-0.09  0.37  3.52] pts 5197 in 3916 n [0.033 0.996 0.079] d -0.54 y 0.265
900 ctr [-0.11  0.38  3.54] pts 3833 in 3086 n [0.033 0.996 0.084] d -0.546 y 0.255

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# test vertical shift E to table plane estimates
import sys,numpy as np,torch,cv2,trimesh;sys.path.insert(0,'scratch');import pose_opt as p
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))]
# plane normal and d fitted local from previous: y=.265 approx horizontal
for i in [830,840,850,860,870,880,890,900]:
 im,obj,per=p.get_frame(i);T=p.BASEc[i].copy()
 # world pose
 Tw=p.TW@T;pw=(V@Tw.T)[:,:3]
 # shift world y so min=0.265
 dy=.265-pw[:,1].min(); Tw2=Tw.copy();Tw2[1,3]+=dy;T2=p.CW@Tw2
 def rend(Q):
  vc=p.V@torch.tensor(Q,device=p.DEV,dtype=torch.float32)[:3,:3].T+torch.tensor(Q,device=p.DEV,dtype=torch.float32)[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
 a=rend(T);b=rend(T2);print(i,'dy',round(dy,3),'iou',round((a&obj).sum()/(a|obj).sum(),3),'->',round((b&obj).sum()/(b|obj).sum(),3),'bbox',end=' ')
 for mm in [a,b]:
  y,x=np.where(mm);print((x.min(),x.max(),y.min(),y.max()),end=' ')
 print()
PY"
```
<details><summary>output</summary>

```
830 dy -0.087 iou 0.186 -> 0.24 bbox (654, 713, 615, 645) (655, 714, 595, 624) 
840 dy -0.063 iou 0.416 -> 0.349 bbox (664, 720, 610, 636) (664, 720, 596, 621) 
850 dy -0.06 iou 0.369 -> 0.369 bbox (668, 719, 610, 638) (669, 719, 596, 623) 
860 dy -0.06 iou 0.337 -> 0.338 bbox (671, 719, 609, 638) (671, 719, 596, 624) 
870 dy -0.069 iou 0.342 -> 0.018 bbox (663, 722, 611, 630) (664, 722, 595, 613) 
880 dy -0.07 iou 0.252 -> 0.139 bbox (663, 722, 609, 636) (663, 723, 594, 620) 
890 dy -0.056 iou 0.182 -> 0.171 bbox (654, 725, 606, 640) (654, 725, 594, 627) 
900 dy -0.047 iou 0.139 -> 0.14 bbox (652, 724, 605, 640) (653, 725, 594, 629) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,io,json,cv2,open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation
c=json.load(open('inputs/camera.json'));Wc=np.array(c['T_world_cam']);K=np.array(c['K_pipeline']);h=h5py.File('/pipeline/depth/moge2/aligned.h5','r');mf=h5py.File('inputs/masks.h5','r');T=np.load('inputs/reconstruction.npz')['T_world_mesh']
for i in [0,20,400,420,840,850,870,900]:
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000;obj=mf['object'][i];per=mf['person'][i]
 ring=binary_dilation(obj,iterations=18)&~binary_dilation(obj,iterations=2)&~per
 yy,xx=np.where(ring);d=dep[yy,xx];m=(d>.5)&(d<8)
 xc=(xx[m]-K[0,2])/K[0,0]*d[m];yc=(yy[m]-K[1,2])/K[1,1]*d[m];pw=np.stack([xc,yc,d[m]],1)@Wc[:3,:3].T+Wc[:3,3]
 # points with depth near object near median; table should be a plane with similar y
 med=np.median(pw[:,1]);sel=np.abs(pw[:,1]-med)<.15;pw=pw[sel]
 pcd=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pw));plane,inds=pcd.segment_plane(.01,3,1000);n=np.array(plane[:3]);dd=plane[3];nn=np.linalg.norm(n);n/=nn;dd/=nn
 if n[1]<0:n=-n;dd=-dd
 ctr=T[i,:3,3]; print(i,'pts',len(pw),'in',len(inds),'n',np.round(n,3),'d',round(dd,3),'yctr',round(-(n[0]*ctr[0]+n[2]*ctr[2]+dd)/n[1],3),'medy',round(med,3))
PY"
```
<details><summary>output</summary>

```
0 pts 2000 in 1747 n [0.025 0.996 0.091] d -0.569 yctr 0.277 medy 0.293
20 pts 1978 in 1744 n [0.027 0.996 0.091] d -0.568 yctr 0.277 medy 0.293
400 pts 1630 in 1466 n [0.03  0.995 0.092] d -0.57 yctr 0.264 medy 0.285
420 pts 1743 in 1577 n [0.043 0.995 0.09 ] d -0.562 yctr 0.26 medy 0.284
840 pts 1236 in 1177 n [0.005 0.996 0.085] d -0.561 yctr 0.272 medy 0.311
850 pts 1170 in 1093 n [0.001 0.996 0.085] d -0.562 yctr 0.269 medy 0.309
870 pts 1009 in 972 n [0.007 0.997 0.078] d -0.54 yctr 0.267 medy 0.306
900 pts 876 in 840 n [0.019 0.996 0.086] d -0.554 yctr 0.255 medy 0.305

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,io,json
from PIL import Image
from scipy.ndimage import binary_dilation
c=json.load(open('inputs/camera.json'));Wc=np.array(c['T_world_cam']);K=np.array(c['K_pipeline']);h=h5py.File('/pipeline/depth/moge2/aligned.h5','r');mf=h5py.File('inputs/masks.h5','r')
for i in [0,20,400,420,840,850,870,900]:
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000;obj=mf['object'][i];per=mf['person'][i]
 ring=binary_dilation(obj,iterations=18)&~binary_dilation(obj,iterations=2)&~per
 for typ,m in [('obj',obj&~per),('ring',ring)]:
  yy,xx=np.where(m);d=dep[yy,xx];d=d[(d>.5)&(d<8)];xc=(xx[d>.5]-K[0,2])/K[0,0]*d;yc=(yy[d>.5]-K[1,2])/K[1,1]*d;pw=np.stack([xc,yc,d],1)@Wc[:3,:3].T+Wc[:3,3];
  print(i,typ,'dmed',round(np.median(d),3),'world median',np.round(np.median(pw,0),3),'y pct',np.round(np.percentile(pw[:,1],[10,50,90]),3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1929 but corresponding boolean dimension is 1928

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,io,json
from PIL import Image
from scipy.ndimage import binary_dilation
c=json.load(open('inputs/camera.json'));Wc=np.array(c['T_world_cam']);K=np.array(c['K_pipeline']);h=h5py.File('/pipeline/depth/moge2/aligned.h5','r');mf=h5py.File('inputs/masks.h5','r')
for i in [0,20,400,420,840,850,870,900]:
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000;obj=mf['object'][i];per=mf['person'][i]
 ring=binary_dilation(obj,iterations=18)&~binary_dilation(obj,iterations=2)&~per
 for typ,m in [('obj',obj&~per),('ring',ring)]:
  yy,xx=np.where(m);d=dep[yy,xx];good=(d>.5)&(d<8);d=d[good];xx=xx[good];yy=yy[good];xc=(xx-K[0,2])/K[0,0]*d;yc=(yy-K[1,2])/K[1,1]*d;pw=np.stack([xc,yc,d],1)@Wc[:3,:3].T+Wc[:3,3]
  print(i,typ,'dmed',round(np.median(d),3),'world median',np.round(np.median(pw,0),3),'y pct',np.round(np.percentile(pw[:,1],[10,50,90]),3))
PY"
```
<details><summary>output</summary>

```
0 obj dmed 3.152 world median [0.326 0.234 3.118] y pct [0.203 0.234 0.269]
0 ring dmed 3.026 world median [0.33  0.293 2.992] y pct [0.264 0.293 0.312]
20 obj dmed 3.155 world median [0.328 0.235 3.121] y pct [0.204 0.235 0.27 ]
20 ring dmed 3.015 world median [0.333 0.293 2.981] y pct [0.264 0.293 0.312]
400 obj dmed 3.235 world median [0.223 0.221 3.205] y pct [0.191 0.221 0.256]
400 ring dmed 3.118 world median [0.246 0.285 3.09 ] y pct [0.254 0.285 0.304]
420 obj dmed 3.186 world median [0.216 0.22  3.157] y pct [0.188 0.22  0.255]
420 ring dmed 3.067 world median [0.242 0.284 3.04 ] y pct [0.258 0.284 0.304]
840 obj dmed 3.332 world median [-0.148  0.277  3.319] y pct [0.259 0.277 0.294]
840 ring dmed 3.061 world median [-0.147  0.311  3.048] y pct [0.292 0.311 0.523]
850 obj dmed 3. …[+473 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,io,json
from PIL import Image
from scipy.ndimage import binary_dilation
c=json.load(open('inputs/camera.json'));Wc=np.array(c['T_world_cam']);K=np.array(c['K_pipeline']);h=h5py.File('/pipeline/depth/moge2/aligned.h5','r');mf=h5py.File('inputs/masks.h5','r');T=np.load('inputs/reconstruction.npz')['T_world_mesh']
for i in [200,220,250,280,300,650,700,740,760,780,800,810]:
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000;obj=mf['object'][i];per=mf['person'][i]
 yy,xx=np.where(obj&~per);d=dep[yy,xx];good=(d>.5)&(d<8);d=d[good];xx=xx[good];yy=yy[good];xc=(xx-K[0,2])/K[0,0]*d;yc=(yy-K[1,2])/K[1,1]*d;pw=np.stack([xc,yc,d],1)@Wc[:3,:3].T+Wc[:3,3]
 print(i,'MoGe center',np.round(np.median(pw,0),3),'y pct',np.round(np.percentile(pw[:,1],[10,50,90]),3),'baseline',np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
200 MoGe center [-0.317  1.095  3.409] y pct [1.067 1.095 1.156] baseline [-0.259  1.137  3.323]
220 MoGe center [-0.32   1.127  3.413] y pct [1.105 1.127 1.176] baseline [-0.259  1.174  3.351]
250 MoGe center [-0.317  1.13   3.418] y pct [1.117 1.13  1.176] baseline [-0.264  1.187  3.388]
280 MoGe center [-0.317  1.122  3.386] y pct [1.097 1.122 1.171] baseline [-0.257  1.183  3.364]
300 MoGe center [-0.316  1.122  3.418] y pct [1.096 1.122 1.17 ] baseline [-0.256  1.177  3.376]
650 MoGe center [-0.526  1.142  3.632] y pct [1.125 1.142 1.183] baseline [-0.405  1.153  3.481]
700 MoGe center [-0.469  1.132  3.605] y pct [1.114 1.132 1.174] baseline [-0.4    1.164  3.467]
740 MoGe center [-0.479  1.133  3.593] y pct [1.121 1.133 1.172] baseline [-0.396  1.158  3.452]
760 MoGe center [-0.469  …[+364 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/depth_xyz_eval.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,io;sys.path.insert(0,'scratch');import pose_opt as p
from PIL import Image
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));m=(r[0,:,:,3]>0).cpu().numpy();z=(r[0,:,:,2]*10).cpu().numpy();return m,z
for i in [0,30,60,80,100,120,180,200,220,250,280,300,350,390,420,450,470,500,530,550,570,630,650,700,740,760,790,810,830,850,870,900]:
 im,obj,per=p.get_frame(i);T=p.BASEc[i];m,z=rend(T);valid=obj&m&~per
 if valid.sum()<30:print(i,'skip');continue
 raw=h['aligned/front_stereo_camera_left'][i];dep=np.asarray(Image.open(io.BytesIO(raw.tobytes()))).astype(np.float32)/1000
 yy,xx=np.where(valid);zc=z[valid];xd=dep[valid]
 # pipeline K for observed depth backprojection, actual K for rendered camera coords
 pc_obs=np.stack([(xx-p.KP[0,2])/p.KP[0,0]*xd,(yy-p.KP[1,2])/p.KP[1,1]*xd,xd],1)
 pc_rend=np.stack([(xx-p.K[0,2])/p.K[0,0]*zc,(yy-p.K[1,2])/p.K[1,1]*zc,zc],1)
 pw_obs=pc_obs@p.TW[:3,:3].T+p.TW[:3,3];pw_ren=pc_rend@p.TW[:3,:3].T+p.TW[:3,3]
 diff=pw_obs-pw_ren;med=np.median(diff,0);mad=np.median(np.abs(diff-med),0)
 print(i,'n',valid.sum(),'diffmed',np.round(med,3),'mad',np.round(mad,3),'mean',np.round(np. …[+210 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_xyz_eval.py", line 13, in <module>
    pc_obs=np.stack([(xx-p.KP[0,2])/p.KP[0,0]*xd,(yy-p.KP[1,2])/p.KP[1,1]*xd,xd],1)
                         ~~~~^^^^^
TypeError: list indices must be integers or slices, not tuple

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/\\['K_pipeline'\\]/['K_pipeline'])/\" scratch/depth_xyz_eval.py; sed -i \"s/p.KP=__import__('json').load(open('inputs\\/camera.json'))\\['K_pipeline'\\])/p.KP=np.array(__import__('json').load(open('inputs\\/camera.json'))['K_pipeline'])/\" scratch/depth_xyz_eval.py; grep KP scratch/depth_xyz_eval.py | head; python scratch/depth_xyz_eval.py"
```
<details><summary>output</summary>

```
0 n 1573 diffmed [-0.03  -0.053  0.078] mad [0.003 0.001 0.026] mean [-0.023 -0.05   0.173]
30 n 1596 diffmed [-0.023 -0.049  0.144] mad [0.005 0.001 0.028] mean [-0.014 -0.046  0.258]
60 n 1602 diffmed [-0.031 -0.053  0.072] mad [0.002 0.001 0.012] mean [-0.027 -0.052  0.126]
80 n 1570 diffmed [-0.038 -0.058  0.017] mad [0.002 0.002 0.012] mean [-0.038 -0.058  0.02 ]
100 n 1736 diffmed [-0.034 -0.056  0.05 ] mad [0.002 0.002 0.012] mean [-0.034 -0.057  0.056]
120 n 832 diffmed [-0.037 -0.063 -0.102] mad [0.002 0.001 0.017] mean [-0.038 -0.063 -0.108]
180 n 1779 diffmed [-0.008 -0.157 -0.338] mad [0.008 0.006 0.019] mean [-0.008 -0.152 -0.315]
200 n 2094 diffmed [-0.05  -0.029  0.132] mad [0.002 0.005 0.013] mean [-0.052 -0.025  0.145]
220 n 2095 diffmed [-0.049 -0.033  0.117] mad [0.002 0 …[+2194 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/(xx-p.KP\\[0,2\\])\\/p.KP\\[0,0\\]/(xx-p.K[0,2])\\/p.K[0,0]/; s/(yy-p.KP\\[1,2\\])\\/p.KP\\[1,1\\]/(yy-p.K[1,2])\\/p.K[1,1]/' scratch/depth_xyz_eval.py
python scratch/depth_xyz_eval.py | head -40"
```
<details><summary>output</summary>

```
0 n 1573 diffmed [0.009 0.006 0.075] mad [0.004 0.002 0.026] mean [0.018 0.011 0.17 ]
30 n 1596 diffmed [0.017 0.011 0.142] mad [0.006 0.002 0.028] mean [0.028 0.017 0.255]
60 n 1602 diffmed [0.008 0.005 0.069] mad [0.002 0.001 0.012] mean [0.013 0.008 0.123]
80 n 1570 diffmed [0.001 0.001 0.014] mad [0.002 0.001 0.012] mean [0.002 0.001 0.017]
100 n 1736 diffmed [0.006 0.003 0.047] mad [0.001 0.001 0.012] mean [0.006 0.003 0.054]
120 n 832 diffmed [-0.    -0.005 -0.104] mad [0.002 0.001 0.017] mean [-0.001 -0.005 -0.11 ]
180 n 1779 diffmed [ 0.026 -0.096 -0.34 ] mad [0.007 0.006 0.019] mean [ 0.026 -0.09  -0.318]
200 n 2094 diffmed [-0.011  0.042  0.129] mad [0.003 0.005 0.013] mean [-0.012  0.046  0.142]
220 n 2095 diffmed [-0.01   0.038  0.114] mad [0.002 0.002 0.008] mean [-0.01   0.04 …[+2146 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys;sys.path.insert(0,'scratch');import static_opt as so
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3]
def meanR(ids):return so.meanR(R[list(ids)])
RC=meanR(range(395,471,5));RD=meanR(range(625,746,5));S=np.diag([1,-1,-1]);M=RC.T@(RD@S);ang=np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)));w,v=np.linalg.eig(M);ax=np.real(v[:,np.argmin(abs(w-1))]);ax/=np.linalg.norm(ax);print('ang',ang,'axis local C',ax);print(np.round(M,3))
# axes between baseline C to D
for label,B in [('baseD',RD),('trueD',RD@S)]:
 M=RC.T@B;w,v=np.linalg.eig(M);ax=np.real(v[:,np.argmin(abs(w-1))]);ax/=np.linalg.norm(ax);print(label,np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1))),ax,M)
PY"
```
<details><summary>output</summary>

```
ang 86.68127587469 axis local C [ 0.01338016 -0.99981753  0.01363387]
[[ 0.058  0.001  0.998]
 [-0.026  1.     0.001]
 [-0.998 -0.026  0.058]]
baseD 178.94766072146768 [-0.72731682  0.00935779  0.68623806] [[ 5.80589443e-02 -1.00772614e-03 -9.98312648e-01]
 [-2.62142832e-02 -9.99656214e-01 -5.15463676e-04]
 [-9.97968923e-01  2.61999778e-02 -5.80654012e-02]]
trueD 86.68127587469 [ 0.01338016 -0.99981753  0.01363387] [[ 5.80589443e-02  1.00772614e-03  9.98312648e-01]
 [-2.62142832e-02  9.99656214e-01  5.15463676e-04]
 [-9.97968923e-01 -2.61999778e-02  5.80654012e-02]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys;sys.path.insert(0,'scratch');import static_opt as so
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3];RC=so.meanR(R[list(range(395,471,5))]);RD=so.meanR(R[list(range(625,746,5))]);RE=so.meanR(R[list(range(840,901,5))]);Sx=np.diag([1,-1,-1]);Sy=np.diag([-1,1,-1]);Sz=np.diag([-1,-1,1])
for nm,S in [('I',np.eye(3)),('Rx',Sx),('Ry',Sy),('Rz',Sz)]:
 M=(RD@Sx).T@(RE@S);ang=np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)));w,v=np.linalg.eig(M);ax=np.real(v[:,np.argmin(abs(w-1))]);ax/=np.linalg.norm(ax)
 print(nm,'ang',round(ang,1),'axis',np.round(ax,3),'offnormal',round(np.degrees(np.arccos(min(1,abs(ax[1])))),1),'M',np.round(M,2))
PY"
```
<details><summary>output</summary>

```
I ang 177.0 axis [-0.922  0.009  0.388] offnormal 89.5 M [[ 0.7  -0.04 -0.71]
 [ 0.   -1.    0.06]
 [-0.71 -0.04 -0.7 ]]
Rx ang 45.7 axis [ 0.067  0.997 -0.023] offnormal 4.1 M [[ 0.7   0.04  0.71]
 [ 0.    1.   -0.06]
 [-0.71  0.04  0.7 ]]
Ry ang 179.0 axis [-0.388  0.026 -0.921] offnormal 88.5 M [[-0.7  -0.04  0.71]
 [-0.   -1.   -0.06]
 [ 0.71 -0.04  0.7 ]]
Rz ang 134.4 axis [0.01  1.    0.028] offnormal 1.7 M [[-0.7   0.04 -0.71]
 [-0.    1.    0.06]
 [ 0.71  0.04 -0.7 ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
for i in [830,840,850,870,900]:
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);outs=[]
 for nm,S in [('I',np.eye(3)),('Rx',np.diag([1,-1,-1])),('Rz',np.diag([-1,-1,1]))]:
  Q=p.BASEc[i].copy();Q[:3,:3]=Q[:3,:3]@S;p.save_overlay(i,Q,crop,f'scratch/E_{i}_{nm}.png',5);o=cv2.imread(f'scratch/E_{i}_{nm}.png');o=cv2.resize(o,(400,250));cv2.putText(o,nm,(10,28),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3);cv2.putText(o,nm,(10,28),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1);outs.append(o)
 cv2.imwrite(f'scratch/Ecmp_{i}.jpg',np.concatenate(outs,1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/sweep_x_dynamic.py <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
for i in list(range(470,651,10))+list(range(750,851,10)):
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj)
 if crop is None: print(i,'NA');continue
 x0,y0,x1,y1=crop;o=obj[y0:y1,x0:x1];pp=per[y0:y1,x0:x1];m=o&~pp
 if m.sum()<50: print(i,'small',m.sum());continue
 vals=[]
 for deg in range(0,181,15):
  S=Rotation.from_rotvec(np.array([np.radians(deg),0,0])).as_matrix();Q=p.BASEc[i].copy();Q[:3,:3]=Q[:3,:3]@S
  rgb,cov,(w,h)=p.render_crop(torch.tensor(Q,device=p.DEV,dtype=torch.float32),crop,2)
  rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);cc=cv2.resize(cov.cpu().numpy().astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)>.5
  tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));cs=[]
  for ch in range(3):
   aa=a[:,:,ch][m];bbb=b[:,:,ch][m];cs.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else -1)
  vals.append(np.mean(cs))
 print(i,'best',np.argmax(vals)*15,'scores',np.round(vals,2),flush=True)
PY
python scratch/sweep_x_dynamic.py > scratch/sweep_x.log 2>&1; cat scratch/sweep_x.log"
```
<details><summary>output</summary>

```
470 best 165 scores [ 0.55 -0.14  0.02  0.21  0.29  0.33  0.38  0.41  0.45  0.51  0.62  0.74
  0.53]
480 best 165 scores [ 0.65  0.02 -0.1   0.2   0.31  0.37  0.45  0.51  0.58  0.63  0.61  0.71
  0.63]
490 best 165 scores [ 0.53 -0.06 -0.09  0.12  0.11  0.06  0.11  0.17  0.21  0.24  0.06  0.56
  0.53]
500 best 0 scores [ 0.43  0.23 -0.01 -0.06 -0.07 -0.03 -0.06 -0.11 -0.14 -0.16 -0.46 -0.2
  0.42]
510 best 0 scores [ 0.63  0.47  0.11 -0.05 -0.09 -0.14 -0.18 -0.23 -0.29 -0.4  -0.63 -0.28
  0.56]
520 best 0 scores [ 0.8   0.62  0.43  0.29  0.22  0.13  0.03 -0.1  -0.26 -0.49 -0.32  0.07
  0.73]
530 best 180 scores [ 0.33 -0.02 -0.21 -0.27 -0.28 -0.28 -0.27 -0.22 -0.33 -0.24 -0.06  0.12
  0.36]
540 best 180 scores [ 0.36 -0.05 -0.1  -0.08 -0.06 -0.04 -0.03 -0.03 -0.04  0.01  0.14  0.36
  0.39] …[+2203 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('inputs/video.mp4');f=h5py.File('inputs/masks.h5','r');ims=[]
for i in range(470,581,5):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();o=f['object'][i];y,x=np.where(o)
 if len(x):cx=int(np.median(x));cy=int(np.median(y))
 else:y,x=np.where(f['person'][i]);cx=int(np.median(x));cy=int(np.median(y))
 cr=im[max(0,cy-55):cy+55,max(0,cx-80):cx+80];cr=cv2.resize(cr,(320,220),interpolation=cv2.INTER_CUBIC);cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),3);cv2.putText(cr,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1);ims.append(cr)
f.close();rows=(len(ims)+4)//5;c=np.zeros((rows*220,1600,3),np.uint8)
for j,im in enumerate(ims):c[(j//5)*220:(j//5+1)*220,(j%5)*320:(j%5+1)*320]=im
cv2.imwrite('scratch/cd_trans.jpg',c)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/refine_trans.py <<'PY'
import sys,numpy as np,torch,h5py,io;sys.path.insert(0,'scratch');import pose_opt as p
from PIL import Image
hf=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def render_full(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy(),(r[0,:,:,2]*10).cpu().numpy()
def refine(T,obj,per,dep,iters=3,maxz=.35,maxcenter=.08):
 T=T.copy();orig=T.copy()
 for it in range(iters):
  m,z=render_full(T);valid=obj&m&~per
  if valid.sum()>=25:
   dz=np.median(dep[valid]-z[valid]);dz=np.clip(dz,-maxz,maxz);T[2,3]+=dz
  m,z=render_full(T)
  if m.sum()>20 and obj.sum()>20:
   yy,xx=np.where(m);ty,tx=np.where(obj)
   dx=np.median(tx)-np.median(xx);dy=np.median(ty)-np.median(yy);zc=T[2,3]
   T[0,3]+=dx*zc/p.K[0,0];T[1,3]+=dy*zc/p.K[1,1]
  # limit total lateral shift relative orig
  dt=T[:3,3]-orig[:3,3]
  # no hard limit for now
 return T
def evaluate(i,T):
 im,obj,per=p.get_frame(i);m,z=render_full(T);iou=(m&obj).sum()/max(1,(m|obj).sum());valid=obj&m&~per
 dz=np.median(dep-(z)[valid]) if valid.sum() else np.nan
 return iou,dz,valid.sum()
if __name__=='__main__':
 for i in [0,30,60,80,100,120,180,200,220,250,280,300,350,390,420,450,470,500,530,550,570,630,650,700,740,760,790,810,830, …[+494 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/refine_trans.py", line 30, in <module>
    a=evaluate(i,T);b=evaluate(i,T2)
      ^^^^^^^^^^^^^
  File "/work/scratch/refine_trans.py", line 23, in evaluate
    dz=np.median(dep-(z)[valid]) if valid.sum() else np.nan
                 ~~~^~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (1152,1536) (1573,) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dz=np.median(dep-(z)\\[valid\\])/dz=np.median(dep[valid]-z[valid])/' scratch/refine_trans.py
python scratch/refine_trans.py"
```
<details><summary>output</summary>

```
0 iou 0.785 -> 0.822 dz 0.076 -> 0.0 dt [-0.008  0.012  0.066]
30 iou 0.789 -> 0.777 dz 0.142 -> -0.0 dt [0.    0.016 0.129]
60 iou 0.792 -> 0.822 dz 0.07 -> -0.0 dt [-0.008  0.012  0.069]
80 iou 0.844 -> 0.829 dz 0.014 -> -0.0 dt [-0.012  0.004  0.021]
100 iou 0.776 -> 0.811 dz 0.047 -> 0.0 dt [0.    0.016 0.038]
120 iou 0.757 -> 0.828 dz -0.104 -> -0.0 dt [-0.014 -0.008 -0.104]
180 iou 0.756 -> 0.756 dz -0.34 -> -0.004 dt [ 0.015 -0.089 -0.331]
200 iou 0.868 -> 0.872 dz 0.129 -> -0.004 dt [-0.025  0.042  0.118]
220 iou 0.864 -> 0.876 dz 0.114 -> 0.001 dt [-0.03   0.038  0.105]
250 iou 0.871 -> 0.884 dz 0.084 -> -0.001 dt [-0.019  0.03   0.08 ]
280 iou 0.827 -> 0.85 dz 0.072 -> 0.0 dt [-0.025  0.025  0.06 ]
300 iou 0.85 -> 0.868 dz 0.087 -> -0.001 dt [-0.025  0.03   0.083]
350 iou 0.549 - …[+1311 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys;sys.path.insert(0,'scratch');import static_opt as so
T=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=T[:,:3,:3];RA=so.meanR(R[list(range(5,56,5))]);RB=so.meanR(R[list(range(205,306,5))]);RC=so.meanR(R[list(range(395,471,5))]);RD=so.meanR(R[list(range(625,746,5))]);RE=so.meanR(R[list(range(840,901,5))]);Sx=np.diag([1,-1,-1])
for a,b,n in [(RA,RB,'A-B'),(RB,RC,'B-C'),(RC,RD@Sx,'C-Dtrue'),(RD@Sx,RE,'Dtrue-E')]:
 M=a.T@b;ang=np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)));w,v=np.linalg.eig(M);ax=np.real(v[:,np.argmin(abs(w-1))]);ax/=np.linalg.norm(ax);print(n,round(ang,1),np.round(ax,3))
PY"
```
<details><summary>output</summary>

```
A-B 78.5 [-0.028 -1.    -0.011]
B-C 77.9 [0.043 0.999 0.014]
C-Dtrue 86.7 [ 0.013 -1.     0.014]
Dtrue-E 177.0 [-0.922  0.009  0.388]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/build_poses.py <<'PY'
import sys,numpy as np,torch,h5py,io,cv2;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.spatial.transform import Rotation
from PIL import Image
r=np.load('inputs/reconstruction.npz'); Tw_base=r['T_world_mesh']; Tc_base=p.BASEc.copy(); N=len(Tc_base)
hf=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def corr_angle(i):
    # Rotation about object's local x, selected from pooled texture evidence.
    if i<=520: return 0.0
    if i<540: return 180.0*(i-520)/20.0
    if i<=830: return 180.0
    if i<840: return 180.0*(840-i)/10.0
    return 0.0
def Rx(deg): return Rotation.from_rotvec(np.array([np.deg2rad(deg),0,0])).as_matrix()
def render_full(T):
    T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3]
    x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2]
    pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None]
    rr,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (rr[0,:,:,3]>0).cpu().numpy(),(rr[0,:,:,2]*10).cpu().numpy()
def refine_trans(T,obj,per,dep):
    T=T.copy();m,z=render_full(T);valid=obj&m&~per
    if valid.sum()>=25:
        dz=np.median(dep[valid]-z[valid]);dz=float(np.clip(dz,-.35,.35));T[2,3]+=dz
    for it in range(2):
        m,z=render_full(T)
        if m.sum()>20 and obj.sum()>20:
            yy,xx=np.where(m);ty,tx=np.where(obj)
            dx=np.median(tx)-np.median(xx);dy=np.median(ty)-np.median(yy);zc=T[2,3 …[+1082 chars]
```
<details><summary>output</summary>

```
done (901, 4, 4)
0 0.0 [7.85322017e-01 8.14411612e-01 1.57300000e+03 7.59198666e-02]
50 0.0 [8.00813008e-01 8.50427350e-01 1.57600000e+03 7.77155161e-02]
100 0.0 [7.76345042e-01 8.06289881e-01 1.73600000e+03 4.74084616e-02]
150 0.0 [1.17619273e-01 5.87861680e-01 2.49000000e+02 1.74728632e-01]
200 0.0 [8.67963576e-01 8.71883061e-01 2.09400000e+03 1.28794670e-01]
250 0.0 [8.70686071e-01 8.81950385e-01 2.09400000e+03 8.39829445e-02]
300 0.0 [8.50497018e-01 8.59016393e-01 2.13500000e+03 8.73641968e-02]
350 0.0 [ 5.48956661e-01  6.24438903e-01  1.02600000e+03 -4.76524591e-01]
400 0.0 [8.16025641e-01 8.05787781e-01 1.25200000e+03 3.64021063e-02]
450 0.0 [6.84426230e-01 7.68089054e-01 1.14300000e+03 2.84719467e-02]
500 0.0 [ 7.88732394e-01  8.52918878e-01  8.55000000e+02 -2.92983055e-02]
550 180. …[+564 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/pool_static.py <<'PY'
import numpy as np,sys,json,trimesh;sys.path.insert(0,'scratch');import static_opt as so
T=np.load('scratch/refined_cam.npy');W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);Tw=np.einsum('ij,fjk->fik',W,T)
R=Tw[:,:3,:3]
phases={'A':(0,58),'B':(210,305),'C':(392,470),'D':(625,745),'E':(840,900)}
for nm,(a,b) in phases.items():
 ids=list(range(a,b+1));Rm=so.meanR(R[ids]);tm=np.median(Tw[ids,:3,3],axis=0);Q=np.eye(4);Q[:3,:3]=Rm;Q[:3,3]=tm
 # variation
 ang=[]
 for i in ids: ang.append(np.degrees(np.arccos(np.clip((np.trace(Rm.T@R[i])-1)/2,-1,1))))
 print(nm,a,b,'t med',np.round(tm,3),'R spread',np.round(np.percentile(ang,[0,50,90,100]),2),'t pct',np.round(np.percentile(Tw[ids,:3,3],[10,50,90],axis=0),3).tolist())
 meshm=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[meshm.vertices,np.ones(len(meshm.vertices))];p=(V@Q.T)[:,:3]
 print('  y range',round(p[:,1].min(),3),round(p[:,1].max(),3),'center y',round(tm[1],3))
PY
python scratch/pool_static.py"
```
<details><summary>output</summary>

```
A 0 58 t med [0.374 0.297 3.186] R spread [0.04 0.04 0.04 0.04] t pct [[0.369, 0.295, 3.168], [0.374, 0.297, 3.186], [0.378, 0.3, 3.225]]
  y range 0.255 0.338 center y 0.297
B 210 305 t med [-0.277  1.21   3.445] R spread [0.71 1.46 3.03 7.42] t pct [[-0.282, 1.177, 3.356], [-0.277, 1.21, 3.445], [-0.261, 1.229, 3.5]]
  y range 1.164 1.257 center y 1.21
C 392 470 t med [0.266 0.289 3.275] R spread [0.33 2.83 3.73 5.79] t pct [[0.26, 0.284, 3.238], [0.266, 0.289, 3.275], [0.277, 0.302, 3.341]]
  y range 0.251 0.327 center y 0.289
D 625 745 t med [-0.442  1.229  3.676] R spread [0.28 1.5  2.97 4.23] t pct [[-0.486, 1.164, 3.587], [-0.442, 1.229, 3.676], [-0.417, 1.247, 3.717]]
  y range 1.187 1.269 center y 1.229
E 840 900 t med [-0.11   0.356  3.421] R spread [ 3.38 11.38 23.6  48.32] t pc …[+111 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/scale_search.py <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
def center(T,obj,it=2):
 T=T.copy()
 for _ in range(it):
  m=rend(T);yy,xx=np.where(m);ty,tx=np.where(obj);dx=np.median(tx)-np.median(xx);dy=np.median(ty)-np.median(yy);T[0,3]+=dx*T[2,3]/p.K[0,0];T[1,3]+=dy*T[2,3]/p.K[1,1]
 return T
for i in [0,30,60,80,100,120,180,200,220,250,280,300,350,390,420,450,470,500,530,550,570,600,630,650,700,740,760,790,810,830,850,870,900]:
 im,obj,per=p.get_frame(i);T=p.BASEc[i];vals=[]
 for dz in np.arange(-.2,.201,.025):
  Q=T.copy();Q[2,3]+=dz;Q=center(Q,obj);m=rend(Q);vals.append((m&obj).sum()/max(1,(m|obj).sum()))
 j=int(np.argmax(vals));dz=round(-.2+j*.025,3);base=center(T,obj);m=rend(base);i0=(m&obj).sum()/max(1,(m|obj).sum())
 print(i,'dz',dz,'iou',round(i0,3),'->',round(vals[j],3),'profile',np.round(vals,2).tolist())
PY
python scratch/scale_search.py"
```
<details><summary>output</summary>

```
0 dz -0.175 iou 0.855 -> 0.941 profile [0.93, 0.94, 0.93, 0.92, 0.91, 0.89, 0.88, 0.87, 0.86, 0.84, 0.83, 0.81, 0.8, 0.79, 0.78, 0.77, 0.76]
30 dz -0.2 iou 0.842 -> 0.927 profile [0.93, 0.93, 0.92, 0.91, 0.89, 0.88, 0.87, 0.85, 0.84, 0.83, 0.82, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75]
60 dz -0.175 iou 0.856 -> 0.933 profile [0.92, 0.93, 0.92, 0.92, 0.9, 0.9, 0.88, 0.87, 0.86, 0.84, 0.83, 0.82, 0.8, 0.79, 0.78, 0.77, 0.76]
80 dz -0.2 iou 0.839 -> 0.882 profile [0.88, 0.88, 0.88, 0.88, 0.87, 0.86, 0.85, 0.85, 0.84, 0.83, 0.82, 0.81, 0.81, 0.79, 0.79, 0.79, 0.77]
100 dz -0.2 iou 0.815 -> 0.875 profile [0.88, 0.86, 0.85, 0.86, 0.86, 0.84, 0.83, 0.82, 0.82, 0.8, 0.8, 0.79, 0.78, 0.76, 0.76, 0.76, 0.74]
120 dz -0.1 iou 0.811 -> 0.829 profile [0.83, 0.82, 0.81, 0.82, 0.83, 0.81, 0.82, 0.82, 0.81, 0.78 …[+3826 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/static_trans_search.py <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
def center(T,obj,it=2):
 T=T.copy()
 for _ in range(it):
  m=rend(T);yy,xx=np.where(m);ty,tx=np.where(obj);dx=np.median(tx)-np.median(xx);dy=np.median(ty)-np.median(yy);T[0,3]+=dx*T[2,3]/p.K[0,0];T[1,3]+=dy*T[2,3]/p.K[1,1]
 return T
def search(i,T):
 im,obj,per=p.get_frame(i);vals=[];poses=[]
 for dz in np.arange(-.35,.351,.05):
  Q=T.copy();Q[2,3]+=dz;Q=center(Q,obj);m=rend(Q);vals.append((m&obj).sum()/max(1,(m|obj).sum()));poses.append(Q)
 j=int(np.argmax(vals));return np.array(vals),poses[j]
ph={'A':list(range(0,59,5)),'B':list(range(210,306,5)),'C':list(range(392,471,5)),'D':list(range(625,746,5)),'E':list(range(840,901,5))}
for nm,ids in ph.items():
 arr=[]
 for i in ids:
  v,Q=search(i,p.BASEc[i]);arr.append((i,v,Q))
 d=np.array([a[2] for a in arr]);
 print('\\n',nm,'n',len(ids),'median pose',np.round(np.median(d,0),4).tolist())
 dz=np.array([(a[2][2,3]-p.BASEc[a[0]][2,3]) for a in arr]);print('dz pct',np.round(np.percentile(dz,[0,25,50,75,100]),3))
 print('best dz idx per frame',[(a[0],float(np.argmax(a[1] …[+165 chars]
```
<details><summary>output</summary>

```

 A n 12 median pose [[-0.8222000002861023, -0.007400000002235174, -0.569100022315979, 0.18359999358654022], [0.061799999326467514, -0.995199978351593, -0.07639999687671661, 0.17839999496936798], [-0.5658000111579895, -0.09799999743700027, 0.8187000155448914, 2.866300106048584], [0.0, 0.0, 0.0, 1.0]]
dz pct [-0.3   -0.3   -0.275 -0.238 -0.15 ]
best dz idx per frame [(0, -0.19999999999999996, 0.935), (5, -0.3, 0.936), (10, -0.3, 0.935), (15, -0.3, 0.935), (20, -0.24999999999999997, 0.944), (25, -0.24999999999999997, 0.946), (30, -0.24999999999999997, 0.945), (35, -0.3, 0.936), (40, -0.19999999999999996, 0.92), (45, -0.3, 0.936), (50, -0.14999999999999997, 0.945), (55, -0.3, 0.938)]

 B n 20 median pose [[-0.714900016784668, -0.014100000262260437, 0.6991000175476074, -0.4645000100135803], [- …[+3844 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp scratch/static_trans_search.py scratch/static_trans_search2.py; python - <<'PY'
p='scratch/static_trans_search2.py';s=open(p).read();s=s.replace(\"def center(T,obj,it=2):\",\"def center(T,obj,per,it=2):\").replace(\"m=rend(T);yy,xx=np.where(m);ty,tx=np.where(obj)\",\"m=rend(T)&~per;yy,xx=np.where(m);ty,tx=np.where(obj)\").replace(\"Q=center(Q,obj);m=rend(Q);vals.append((m&obj).sum()/max(1,(m|obj).sum()))\",\"Q=center(Q,obj,per);m=rend(Q)&~per;vals.append((m&obj).sum()/max(1,(m|obj).sum()))\").replace(\"sttrans_{nm}\",\"sttrans2_{nm}\")
open(p,'w').write(s)
PY
python scratch/static_trans_search2.py"
```
<details><summary>output</summary>

```

 A n 12 median pose [[-0.8222000002861023, -0.007400000002235174, -0.569100022315979, 0.18410000205039978], [0.061799999326467514, -0.995199978351593, -0.07639999687671661, 0.179299995303154], [-0.5658000111579895, -0.09799999743700027, 0.8187000155448914, 2.890000104904175], [0.0, 0.0, 0.0, 1.0]]
dz pct [-0.3  -0.3  -0.25 -0.2  -0.15]
best dz idx per frame [(0, -0.19999999999999996, 0.935), (5, -0.3, 0.936), (10, -0.3, 0.935), (15, -0.3, 0.935), (20, -0.24999999999999997, 0.944), (25, -0.24999999999999997, 0.946), (30, -0.24999999999999997, 0.945), (35, -0.3, 0.936), (40, -0.19999999999999996, 0.92), (45, -0.3, 0.936), (50, -0.14999999999999997, 0.945), (55, -0.19999999999999996, 0.933)]

 B n 20 median pose [[-0.714900016784668, -0.014100000262260437, 0.6991000175476074, -0.462199985980 …[+3751 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p,static_trans_search2 as st
for i in [420,422,700,850]:
 im,obj,per=p.get_frame(i);T=p.BASEc[i]
 for dz in [0,-.35,.2]:
  Q=T.copy();Q[2,3]+=dz;Q=st.center(Q,obj,per);m=st.rend(Q)&~per;mm=st.rend(Q)
  y,x=np.where(m);yy,xx=np.where(mm);ty,tx=np.where(obj)
  print(i,dz,'vis bbox',x.min(),x.max(),y.min(),y.max(),'full',xx.min(),xx.max(),yy.min(),yy.max(),'target',tx.min(),tx.max(),ty.min(),ty.max(),'iou',(m&obj).sum()/(m|obj).sum())
PY"
```
<details><summary>output</summary>

```

 A n 12 median pose [[-0.8222000002861023, -0.007400000002235174, -0.569100022315979, 0.18410000205039978], [0.061799999326467514, -0.995199978351593, -0.07639999687671661, 0.179299995303154], [-0.5658000111579895, -0.09799999743700027, 0.8187000155448914, 2.890000104904175], [0.0, 0.0, 0.0, 1.0]]
dz pct [-0.3  -0.3  -0.25 -0.2  -0.15]
best dz idx per frame [(0, -0.19999999999999996, 0.935), (5, -0.3, 0.936), (10, -0.3, 0.935), (15, -0.3, 0.935), (20, -0.24999999999999997, 0.944), (25, -0.24999999999999997, 0.946), (30, -0.24999999999999997, 0.945), (35, -0.3, 0.936), (40, -0.19999999999999996, 0.92), (45, -0.3, 0.936), (50, -0.14999999999999997, 0.945), (55, -0.19999999999999996, 0.933)]

 B n 20 median pose [[-0.714900016784668, -0.014100000262260437, 0.6991000175476074, -0.462199985980 …[+4935 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p
# local funcs
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
def cent(T,obj,per):
 for _ in range(2):
  m=rend(T)&~per;yy,xx=np.where(m);ty,tx=np.where(obj);T[0,3]+=(np.median(tx)-np.median(xx))*T[2,3]/p.K[0,0];T[1,3]+=(np.median(ty)-np.median(yy))*T[2,3]/p.K[1,1]
 return T
for i in [0,30,420,700,850]:
 im,obj,per=p.get_frame(i);print('\\n',i)
 for dz in np.arange(-.8,.801,.1):
  Q=p.BASEc[i].copy();Q[2,3]+=dz;Q=cent(Q,obj,per);m=rend(Q)&~per;iou=(m&obj).sum()/max(1,(m|obj).sum());print(round(dz,1),round(iou,3),end=' | ')
 print()
PY"
```
<details><summary>output</summary>

```

 0
-0.8 0.65 | -0.7 0.706 | -0.6 0.765 | -0.5 0.828 | -0.4 0.891 | -0.3 0.933 | -0.2 0.935 | -0.1 0.906 | -0.0 0.855 | 0.1 0.802 | 0.2 0.759 | 0.3 0.711 | 0.4 0.674 | 0.5 0.639 | 0.6 0.603 | 0.7 0.572 | 0.8 0.545 | 

 30
-0.8 0.658 | -0.7 0.716 | -0.6 0.776 | -0.5 0.839 | -0.4 0.9 | -0.3 0.936 | -0.2 0.927 | -0.1 0.893 | -0.0 0.842 | 0.1 0.793 | 0.2 0.747 | 0.3 0.702 | 0.4 0.665 | 0.5 0.631 | 0.6 0.593 | 0.7 0.563 | 0.8 0.536 | 

 420
-0.8 0.853 | -0.7 0.885 | -0.6 0.908 | -0.5 0.94 | -0.4 0.947 | -0.3 0.929 | -0.2 0.893 | -0.1 0.863 | -0.0 0.822 | 0.1 0.774 | 0.2 0.732 | 0.3 0.697 | 0.4 0.655 | 0.5 0.62 | 0.6 0.589 | 0.7 0.562 | 0.8 0.532 | 

 700
-0.8 0.65 | -0.7 0.693 | -0.6 0.74 | -0.5 0.783 | -0.4 0.818 | -0.3 0.859 | -0.2 0.906 | -0.1 0.925 | -0.0 0.906 | 0.1 0.865 | 0.2 0.8 | 0.3 0 …[+284 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/all_scale_search.py <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
def center(T,obj,per):
 for _ in range(2):
  m=rend(T)&~per;yy,xx=np.where(m);ty,tx=np.where(obj);T[0,3]+=(np.median(tx)-np.median(xx))*T[2,3]/p.K[0,0];T[1,3]+=(np.median(ty)-np.median(yy))*T[2,3]/p.K[1,1]
 return T
res=[]
for i in range(0,901,5):
 im,obj,per=p.get_frame(i);mbase=rend(p.BASEc[i]);occ=(mbase&per).sum()/max(1,mbase.sum())
 vals=[];poses=[]
 for dz in np.arange(-.8,.801,.05):
  Q=p.BASEc[i].copy();Q[2,3]+=dz;Q=center(Q,obj,per);m=rend(Q)&~per;vals.append((m&obj).sum()/max(1,(m|obj).sum()));poses.append(Q)
 j=int(np.argmax(vals));res.append((i,round(-.8+j*.05,3),float(vals[j]),float(occ),obj.sum(),*poses[j][:3,3]))
 if i%50==0:print(res[-1],flush=True)
np.save('scratch/all_scale.npy',np.array(res))
PY
python scratch/all_scale_search.py"
```
<details><summary>output</summary>

```
(0, -0.2, 0.9349923819197562, 0.0, 1929, 0.18554507, 0.1806743, 2.9444215)
(50, -0.15, 0.9445316588173731, 0.0, 1871, 0.19091633, 0.18275759, 2.9697049)
(100, -0.35, 0.934714092751013, 0.027472527472527472, 2175, 0.15890115, 0.13753967, 2.8238835)
(150, -0.1, 0.5942028985507246, 0.0, 1284, -0.44711033, 0.42988238, 3.2941103)
(200, -0.05, 0.8857731958762887, 0.005724350506384853, 2242, -0.44775075, 1.0147014, 3.2912486)
(250, -0.05, 0.8961794019933554, 0.0, 2281, -0.45651639, 1.0637825, 3.3562589)
(300, 0.05, 0.8661224489795918, 0.001697792869269949, 2298, -0.46896607, 1.0826846, 3.444974)
(350, 0.3, 0.6799242424242424, 0.022742040285899934, 1356, -0.4853509, 0.49871466, 3.7218733)
/opt/conda/lib/python3.11/site-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  …[+1223 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('scratch/all_scale.npy')
for row in a: print(f'{int(row[0]):3d} dz {row[1]:+.2f} iou {row[2]:.2f} occ {row[3]:.2f} area {int(row[4]):4d} t {row[5]:.2f},{row[6]:.2f},{row[7]:.2f}')
PY"
```
<details><summary>output</summary>

```
  0 dz -0.20 iou 0.93 occ 0.00 area 1929 t 0.19,0.18,2.94
  5 dz -0.30 iou 0.94 occ 0.00 area 1975 t 0.18,0.18,2.85
 10 dz -0.30 iou 0.94 occ 0.00 area 1944 t 0.18,0.18,2.84
 15 dz -0.30 iou 0.94 occ 0.00 area 1970 t 0.18,0.18,2.84
 20 dz -0.25 iou 0.94 occ 0.00 area 1972 t 0.19,0.18,2.89
 25 dz -0.25 iou 0.95 occ 0.00 area 1965 t 0.19,0.18,2.89
 30 dz -0.25 iou 0.95 occ 0.00 area 1966 t 0.19,0.18,2.89
 35 dz -0.30 iou 0.94 occ 0.00 area 1964 t 0.18,0.17,2.82
 40 dz -0.20 iou 0.92 occ 0.00 area 1964 t 0.18,0.18,2.92
 45 dz -0.30 iou 0.94 occ 0.00 area 1961 t 0.18,0.17,2.82
 50 dz -0.15 iou 0.94 occ 0.00 area 1871 t 0.19,0.18,2.97
 55 dz -0.20 iou 0.93 occ 0.00 area 1979 t 0.18,0.18,2.92
 60 dz -0.30 iou 0.94 occ 0.02 area 1957 t 0.18,0.17,2.82
 65 dz -0.45 iou 0.94 occ 0.01 area 1881 t 0.1 …[+9830 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# fine sweep no import side effects
import sys,numpy as np,torch;sys.path.insert(0,'scratch');import pose_opt as p
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
def cent(T,obj,per):
 for _ in range(2):
  m=rend(T)&~per;yy,xx=np.where(m);ty,tx=np.where(obj);T[0,3]+=(np.median(tx)-np.median(xx))*T[2,3]/p.K[0,0];T[1,3]+=(np.median(ty)-np.median(yy))*T[2,3]/p.K[1,1]
 return T
for i in [0,420,700]:
 im,obj,per=p.get_frame(i);print(i)
 for dz in np.arange(-.6,.051,.025):
  Q=p.BASEc[i].copy();Q[2,3]+=dz;Q=cent(Q,obj,per);m=rend(Q)&~per;iou=(m&obj).sum()/(m|obj).sum()
  if iou>.8: print(round(dz,3),round(iou,3),end=';')
 print()
PY"
```
<details><summary>output</summary>

```
0
-0.525 0.81;-0.5 0.828;-0.475 0.842;-0.45 0.851;-0.425 0.875;-0.4 0.891;-0.375 0.902;-0.35 0.916;-0.325 0.927;-0.3 0.933;-0.275 0.944;-0.25 0.926;-0.225 0.95;-0.2 0.935;-0.175 0.941;-0.15 0.93;-0.125 0.921;-0.1 0.906;-0.075 0.893;-0.05 0.882;-0.025 0.866;0.0 0.855;0.025 0.843;0.05 0.829;
420
-0.6 0.908;-0.575 0.927;-0.55 0.923;-0.525 0.927;-0.5 0.94;-0.475 0.933;-0.45 0.94;-0.425 0.935;-0.4 0.947;-0.375 0.936;-0.35 0.939;-0.325 0.934;-0.3 0.929;-0.275 0.924;-0.25 0.913;-0.225 0.911;-0.2 0.893;-0.175 0.892;-0.15 0.88;-0.125 0.866;-0.1 0.863;-0.075 0.853;-0.05 0.843;-0.025 0.835;0.0 0.822;0.025 0.815;
700
-0.45 0.802;-0.425 0.82;-0.4 0.818;-0.375 0.835;-0.35 0.849;-0.325 0.848;-0.3 0.859;-0.275 0.872;-0.25 0.883;-0.225 0.896;-0.2 0.906;-0.175 0.913;-0.15 0.921;-0.125 0.925;-0.1 0.925;-0.07 …[+67 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/static_pose_search.py <<'PY'
import sys,numpy as np,torch,cv2,time;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
from scipy.ndimage import gaussian_filter
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
def center_const(T, objs, pers):
 for _ in range(2):
  mc=rend(T); yy,xx=np.where(mc)
  cs=[]
  for o,q in zip(objs,pers):
   m=o&~q
   if m.sum()>20:
    y,x=np.where(m);cs.append((np.median(x),np.median(y)))
  if not cs:break
  tx=np.mean([z[0] for z in cs]);ty=np.mean([z[1] for z in cs]);zc=T[2,3]
  T[0,3]+=(tx-np.median(xx))*zc/p.K[0,0];T[1,3]+=(ty-np.median(yy))*zc/p.K[1,1]
 return T
ph={'A':list(range(0,61,5)),'B':list(range(210,306,5)),'C':list(range(395,471,5)),'D':list(range(625,746,5)),'E':list(range(840,901,5))}
rotopts={'A':[('I',np.eye(3))],'B':[('I',np.eye(3)),('Ry',np.diag([-1,1,-1]))],'C':[('I',np.eye(3))],'D':[('Rx',np.diag([1,-1,-1]))],'E':[('I',np.eye(3)),('Rx',np.diag([1,-1,-1])),('Rz',np.diag([-1,-1,1]))]}
hf=None
for nm,ids in ph.items():
 frames=[p.get_frame(i) for i in ids]
 base=so.makebase(ids,np.eye(3)); baseR=base[:3,:3].copy();
 best=(1e9,None)
 print('\\nPHASE',nm,flush=True)
 for rn,S in rotopts[nm]:
  Tc=base.copy();Tc[ …[+660 chars]
```
<details><summary>output</summary>

```
I -0.4 0.2854
I -0.35 0.2764
I -0.3 0.2864
I -0.25 0.2931
I -0.2 0.3008
I -0.15 0.3061
I -0.1 0.3123
I -0.05 0.3185
I -0.0 0.3215
I 0.05 0.3478
I 0.1 0.3518
I 0.15 0.3533
I 0.2 0.3532
I 0.25 0.3545
I 0.3 0.3798
I 0.35 0.3808
I 0.4 0.3781
I 0.45 0.3751
I 0.5 0.398
I 0.55 0.3934
I 0.6 0.3892
I 0.65 0.4124
I 0.7 0.4056
I 0.75 0.3981
I 0.8 0.4185
Rx -0.5 0.2697
Rx -0.45 0.2808
Rx -0.4 0.2908
Rx -0.35 0.281
Rx -0.3 0.2904
Rx -0.25 0.2979
Rx -0.2 0.3045
Rx -0.15 0.3106
Rx -0.1 0.3163
Rx -0.05 0.3223
Rx -0.0 0.3263
Rx 0.05 0.3522
Rx 0.1 0.3554
Rx 0.15 0.3582
Rx 0.2 0.3598
Rx 0.25 0.358
Rx 0.3 0.3828
Rx 0.35 0.3825
Rx 0.4 0.3828
Rx 0.45 0.3773
Rx 0.5 0.4006
Rx 0.55 0.3968
Rx 0.6 0.3895
Rx 0.65 0.414
Rx 0.7 0.4099
Rx 0.75 0.4293
Rx 0.8 0.4211
Rz -0.5 0.2569
Rz -0.45 0.2659
Rz -0.4 0.2772
Rz -0.35 0 …[+518 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep 'BEST' scratch/static_pose.log; grep -E '"'^PHASE|''^I |''^Rx |''^Ry |''^Rz '"' scratch/static_pose.log | awk 'BEGIN{p=\"\"} /"'^PHASE/{p=$2; print} !/''^PHASE/ && ($2=="-0.50"||$2=="-0.40"||$2=="-0.30"||$2=="-0.20"||$2=="-0.10"||$2=="-0.00"||$2=="0.00"||$2=="0.10"||$2=="0.20"||$2=="0.30"||$2=="0.40"||$2=="0.50"||$2=="0.60"||$2=="0.70"||$2=="0.80") {print}'"'"
```
<details><summary>output</summary>

```
BEST A 0.06603153189683164 I -0.30000000000000004 [[-0.8222, -0.0074, -0.5691, 0.1815], [0.0618, -0.9952, -0.0764, 0.1771], [-0.5658, -0.098, 0.8187, 2.8377], [0.0, 0.0, 0.0, 1.0]]
BEST B 0.13668671354870499 I -1.1102230246251565e-16 [[-0.7214, -0.024, 0.6921, -0.4603], [-0.0734, -0.9911, -0.1109, 1.0715], [0.6886, -0.1308, 0.7133, 3.3747], [0.0, 0.0, 0.0, 1.0]]
BEST C 0.1041699732110446 I -0.35000000000000003 [[-0.8293, -0.0021, -0.5588, 0.0834], [0.0458, -0.9969, -0.0643, 0.1698], [-0.5569, -0.0789, 0.8268, 2.9385], [0.0, 0.0, 0.0, 1.0]]
BEST D 0.2651106695504505 Rx 0.2999999999999998 [[0.5096, 0.0117, -0.8604, -0.6763], [0.0929, -0.9948, 0.0415, 1.1388], [-0.8554, -0.1011, -0.508, 3.7759], [0.0, 0.0, 0.0, 1.0]]
BEST E 0.5706591572612131 Rx 0.7499999999999998 [[0.9714, -0.005, -0.2374, - …[+141 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '"'^PHASE'"\\|"'^BEST'"' scratch/static_pose.log; sed -n '1,80p' scratch/static_pose.log; sed -n '80,170p' scratch/static_pose.log"
```
<details><summary>output</summary>

```
2:PHASE A
30:BEST A 0.06603153189683164 I -0.30000000000000004 [[-0.8222, -0.0074, -0.5691, 0.1815], [0.0618, -0.9952, -0.0764, 0.1771], [-0.5658, -0.098, 0.8187, 2.8377], [0.0, 0.0, 0.0, 1.0]]
32:PHASE B
87:BEST B 0.13668671354870499 I -1.1102230246251565e-16 [[-0.7214, -0.024, 0.6921, -0.4603], [-0.0734, -0.9911, -0.1109, 1.0715], [0.6886, -0.1308, 0.7133, 3.3747], [0.0, 0.0, 0.0, 1.0]]
89:PHASE C
117:BEST C 0.1041699732110446 I -0.35000000000000003 [[-0.8293, -0.0021, -0.5588, 0.0834], [0.0458, -0.9969, -0.0643, 0.1698], [-0.5569, -0.0789, 0.8268, 2.9385], [0.0, 0.0, 0.0, 1.0]]
119:PHASE D
147:BEST D 0.2651106695504505 Rx 0.2999999999999998 [[0.5096, 0.0117, -0.8604, -0.6763], [0.0929, -0.9948, 0.0415, 1.1388], [-0.8554, -0.1011, -0.508, 3.7759], [0.0, 0.0, 0.0, 1.0]]
149:PHASE E
231:BE …[+3150 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# load module without executing main? static_pose_search has top-level; copy functions via exec pieces messy. Minimal script similar.
import sys,numpy as np,torch,json;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
def rend(T):
 T=torch.as_tensor(T,device=p.DEV,dtype=torch.float32);vc=p.V@T[:3,:3].T+T[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy()
def center_const(T,fs):
 for _ in range(2):
  mc=rend(T);yy,xx=np.where(mc);cs=[]
  for im,o,q in fs:
   m=o&~q
   if m.sum()>20:
    y,x=np.where(m);cs.append((np.median(x),np.median(y)))
  tx=np.mean([z[0] for z in cs]);ty=np.mean([z[1] for z in cs]);T[0,3]+=(tx-np.median(xx))*T[2,3]/p.K[0,0];T[1,3]+=(ty-np.median(yy))*T[2,3]/p.K[1,1]
 return T
ids=list(range(840,871,5));fs=[p.get_frame(i) for i in ids];base=so.makebase(ids,np.eye(3));baseR=base[:3,:3].copy();best=(9,None)
for rn,S in [('I',np.eye(3)),('Rx',np.diag([1,-1,-1])),('Ry',np.diag([-1,1,-1])),('Rz',np.diag([-1,-1,1]))]:
 for dz in np.arange(-.2,1.01,.05):
  Q=base.copy();Q[:3,:3]=baseR@S;Q[2,3]+=dz;Q=center_const(Q,fs);ious=[]
  for im,o,q in fs:
   m=rend(Q)&~q;ious.append((m&o).sum()/max(1,(m|o).sum()))
  if 1-np.mean(ious)<best[0]:best=(1-np.mean(ious),(rn,dz,Q.copy(),np.mean(ious)))
  print(rn,round(dz,2),round(np.mean(ious), …[+122 chars]
```
<details><summary>output</summary>

```
I -0.2 0.42
I -0.15 0.426
I -0.1 0.454
I -0.05 0.454
I -0.0 0.436
I 0.05 0.438
I 0.1 0.485
I 0.15 0.487
I 0.2 0.489
I 0.25 0.49
I 0.3 0.477
I 0.35 0.47
I 0.4 0.493
I 0.45 0.495
I 0.5 0.519
I 0.55 0.518
I 0.6 0.518
I 0.65 0.508
I 0.7 0.519
I 0.75 0.518
I 0.8 0.54
I 0.85 0.543
I 0.9 0.539
I 0.95 0.536
I 1.0 0.538
Rx -0.2 0.424
Rx -0.15 0.429
Rx -0.1 0.455
Rx -0.05 0.456
Rx -0.0 0.46
Rx 0.05 0.476
Rx 0.1 0.46
Rx 0.15 0.484
Rx 0.2 0.487
Rx 0.25 0.488
Rx 0.3 0.487
Rx 0.35 0.5
Rx 0.4 0.494
Rx 0.45 0.494
Rx 0.5 0.516
Rx 0.55 0.517
Rx 0.6 0.515
Rx 0.65 0.512
Rx 0.7 0.514
Rx 0.75 0.513
Rx 0.8 0.534
Rx 0.85 0.53
Rx 0.9 0.535
Rx 0.95 0.533
Rx 1.0 0.528
Ry -0.2 0.443
Ry -0.15 0.445
Ry -0.1 0.451
Ry -0.05 0.451
Ry -0.0 0.46
Ry 0.05 0.481
Ry 0.1 0.481
Ry 0.15 0.482
Ry 0.2 0.483
Ry 0.25 0.513
Ry 0.3 0.51 …[+781 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
m=np.load('scratch/refine_meta.npy');Tref=np.load('scratch/refined_cam.npy');base=np.load('inputs/reconstruction.npz')['T_world_mesh'];import json
W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W);B=np.einsum('ij,fjk->fik',C,base)
ph={'A':(0,60),'B':(210,305),'C':(392,470),'D':(625,745),'E':(840,900)}
for nm,(a,b) in ph.items():
 ids=np.arange(a,b+1);print(nm,'raw dz median',np.nanmedian(m[ids,3]),'pct',np.percentile(m[ids,3],[10,25,50,75,90]),'accepted',np.mean(np.abs(Tref[ids,2,3]-B[ids,2,3])>1e-6),'applied dz med',np.median(Tref[ids,2,3]-B[ids,2,3]),'range',np.percentile(Tref[ids,2,3]-B[ids,2,3],[10,90]))
PY"
```
<details><summary>output</summary>

```
A raw dz median 0.0853111743927002 pct [0.06477094 0.07433259 0.08531117 0.11906075 0.13003123] accepted 1.0 applied dz med 0.08531118193921827 range [0.064771   0.13003115]
B raw dz median 0.09298276901245117 pct [0.02787852 0.06560177 0.09298277 0.14250207 0.1942935 ] accepted 0.875 applied dz med 0.08337720108616531 range [2.49971381e-08 1.50317151e-01]
C raw dz median 0.028471946716308594 pct [-0.0050416   0.00899005  0.02847195  0.03880286  0.04890718] accepted 0.9113924050632911 applied dz med 0.02519924095506676 range [-0.0050417   0.04641108]
D raw dz median 0.2086324691772461 pct [0.17244005 0.19212437 0.20863247 0.22041965 0.23038459] accepted 0.9008264462809917 applied dz med 0.20863245368827465 range [0.09488222 0.2303845 ]
E raw dz median -0.06903624534606934 pct [-0.13098276  …[+130 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
T=np.load('scratch/refined_cam.npy');W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);Tw=np.einsum('ij,fjk->fik',W,T)
for nm,(a,b) in {'A':(0,60),'B':(210,305),'C':(392,470),'D':(625,745),'E':(840,900)}.items():
 print(nm,np.round(np.median(Tw[a:b+1,:3,3],axis=0),3), 'pct',np.round(np.percentile(Tw[a:b+1,:3,3],[10,90],axis=0),3).tolist())
PY"
```
<details><summary>output</summary>

```
A [0.374 0.297 3.185] pct [[0.368, 0.295, 3.161], [0.378, 0.3, 3.225]]
B [-0.277  1.21   3.445] pct [[-0.282, 1.177, 3.356], [-0.261, 1.229, 3.5]]
C [0.266 0.289 3.275] pct [[0.26, 0.284, 3.238], [0.277, 0.302, 3.341]]
D [-0.442  1.229  3.676] pct [[-0.486, 1.164, 3.587], [-0.417, 1.247, 3.717]]
E [-0.11   0.356  3.421] pct [[-0.141, 0.343, 3.365], [-0.097, 0.365, 3.564]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_static_candidates.py <<'PY'
import sys,numpy as np,torch,cv2,json,trimesh;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
from scipy.ndimage import gaussian_filter
W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W)
baseW=np.load('inputs/reconstruction.npz')['T_world_mesh'];TrefW=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'))
# candidate static poses from earlier
phases={'A':list(range(0,61,5)),'B':list(range(210,306,5)),'C':list(range(395,471,5)),'D':list(range(625,746,5)),'E':list(range(840,901,5))}
def meanR(ids,R):return so.meanR(R[ids])
def evaluate(ids,Q):
 vals=[];ious=[]
 for i in ids:
  im,obj,per=p.get_frame(i);T=C@Q
  # render full
  vc=p.V@torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,:3].T+torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));m=(r[0,:,:,3]>0).cpu().numpy()&~per
  ious.append((m&obj).sum()/max(1,(m|obj).sum()))
  # texture on tight crop
  crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2)
  rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255
  a=gaussian_filter(tt,(1.2,1.2,0 …[+1054 chars]
```
<details><summary>output</summary>

```

PHASE A
baseline (0.7885178414892253, 0.3062144581028115, 0.7879387048937222, 0.2968589651467242)
depthmed (0.8010044924676976, 0.3959778925878211, 0.7973523421588594, 0.3876473820237756)
masksearch (0.0, -1.0, 0.0, -1.0)

PHASE B
baseline (0.8450237974783574, 0.3070158473365184, 0.84782326867754, 0.3290215973692533)
depthmed (0.8554336066833571, 0.30168185193665625, 0.8599904356183903, 0.3134992132092783)
masksearch (0.0, -1.0, 0.0, -1.0)
I (0.8554336066833571, 0.30168185193665625, 0.8599904356183903, 0.3134992132092783)
Rx (0.8414366146708572, 0.21876192368664907, 0.8473360694508668, 0.22363613163029405)
Ry (0.8342511137979318, 0.30838485265878635, 0.8386822120600392, 0.3239540691948146)
Rz (0.8440003588337831, 0.3138194354748346, 0.84714770254973, 0.3242052894671131)

PHASE C
baseline  …[+1313 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2,json;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
from scipy.ndimage import gaussian_filter
W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W);Tw=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'));ids=list(range(840,901,5));Rd=so.meanR(Tw[ids,:3,:3]);td=np.median(Tw[ids,:3,3],0)
for i in ids:
 im,obj,per=p.get_frame(i);vals=[]
 for nm,S in [('I',np.eye(3)),('Rx',np.diag([1,-1,-1])),('Ry',np.diag([-1,1,-1])),('Rz',np.diag([-1,-1,1]))]:
  Q=np.eye(4);Q[:3,:3]=Rd@S;Q[:3,3]=td;T=C@Q;crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));cs=[]
  for ch in range(3):
   aa=a[:,:,ch][mm];bbb=b[:,:,ch][mm];cs.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else -1)
  vals.append(np.mean(cs))
 print(i,np.round(vals,2),['I','Rx','Ry','Rz'][np.argmax(vals)])
PY"
```
<details><summary>output</summary>

```
840 [0.22 0.03 0.42 0.43] Rz
845 [0.37 0.22 0.5  0.6 ] Rz
850 [0.23 0.01 0.42 0.52] Rz
855 [0.32 0.09 0.5  0.57] Rz
860 [0.37 0.16 0.55 0.59] Rz
865 [0.31 0.07 0.5  0.54] Rz
870 [0.37 0.28 0.48 0.48] Ry
875 [0.59 0.57 0.63 0.63] Rz
880 [0.56 0.42 0.52 0.54] I
885 [0.55 0.37 0.46 0.53] I
890 [0.49 0.24 0.35 0.47] I
895 [ 0.41 -0.14  0.07 -0.  ] I
900 [ 0.26 -0.11  0.05  0.04] I

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,sys;sys.path.insert(0,'scratch');import static_opt as so
bw=np.load('inputs/reconstruction.npz')['T_world_mesh'];W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);rw=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'));ids=list(range(840,901,5))
print('tbase',np.median(bw[ids,:3,3],0),'tref',np.median(rw[ids,:3,3],0))
Rb=so.meanR(bw[ids,:3,:3]);Rr=so.meanR(rw[ids,:3,:3]);print('Rdiff',np.degrees(np.arccos(np.clip((np.trace(Rb.T@Rr)-1)/2,-1,1))))
PY"
```
<details><summary>output</summary>

```
tbase [-0.09231727  0.3783865   3.5209434 ] tref [-0.10993803  0.35271642  3.41152033]
Rdiff 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_static_zonly.py <<'PY'
import sys,numpy as np,torch,cv2,json,json;sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
from scipy.ndimage import gaussian_filter
W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W);bw=np.load('inputs/reconstruction.npz')['T_world_mesh'];meta=np.load('scratch/refine_meta.npy')
ph={'A':(0,60,np.eye(3)),'B':(210,305,np.eye(3)),'C':(392,470,np.eye(3)),'D':(625,745,np.diag([1,-1,-1])),'E':(840,900,np.eye(3))}
def evalp(ids,Q):
 ious=[];csall=[]
 for i in ids:
  im,obj,per=p.get_frame(i);T=C@Q;crop=p.crop_bounds(obj);x0,y0,x1,y1=crop
  vc=p.V@torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,:3].T+torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));m=(r[0,:,:,3]>0).cpu().numpy()&~per;ious.append((m&obj).sum()/max(1,(m|obj).sum()))
  rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));cs=[]
  for ch in range(3):
   aa=a[:,:,ch][mm];bbb=b[:,:,ch][mm];cs.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else …[+466 chars]
```
<details><summary>output</summary>

```
A base dz 0.085 (0.6951984356115244, 0.13040274732583185, 0.6936026936026936, 0.12358454926470236)
A corr dz 0.085 (0.6951984356115244, 0.13040274732583185, 0.6936026936026936, 0.12358454926470236)
B base dz 0.089 (0.5852828012206628, 0.36136564910356284, 0.5731737332405987, 0.33347911152630894)
B corr dz 0.089 (0.5852828012206628, 0.36136564910356284, 0.5731737332405987, 0.33347911152630894)
C base dz 0.028 (0.7794248370184043, 0.35699273885588323, 0.7891812459690917, 0.34535171137942466)
C corr dz 0.028 (0.7794248370184043, 0.35699273885588323, 0.7891812459690917, 0.34535171137942466)
D base dz 0.207 (0.30406873298715564, 0.3765048986799314, 0.2864056313242411, 0.3944480448475492)
D corr dz 0.207 (0.30034778984502564, 0.35822789845631325, 0.28137817883511074, 0.3674996465360542)
E base d …[+190 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('inputs/video.mp4');frames=[]
for i in range(901):
 ok,f=cap.read();frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
h=h5py.File('inputs/masks.h5','r')
for i in list(range(180,321,10))+list(range(390,481,10))+list(range(620,751,10))+list(range(830,901,10)):
 a=frames[i];b=frames[i+10];o=h['object'][i]
 flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
 yy,xx=np.where(o)
 if len(xx):
  mag=np.linalg.norm(flow[yy,xx],axis=1);dx=np.median(flow[yy,xx,0]);dy=np.median(flow[yy,xx,1]);print(i,'n',len(xx),'dxdy',round(dx,2),round(dy,2),'mag med',round(np.median(mag),2))
h.close()
PY"
```
<details><summary>output</summary>

```
180 n 1999 dxdy 0.78 22.11 mag med 22.18
190 n 2179 dxdy -2.71 9.23 mag med 9.7
200 n 2242 dxdy -1.03 6.66 mag med 6.8
210 n 2247 dxdy 0.7 1.12 mag med 1.47
220 n 2231 dxdy 0.27 0.42 mag med 0.83
230 n 2229 dxdy -0.0 -0.02 mag med 0.21
240 n 2292 dxdy 0.02 -0.01 mag med 0.1
250 n 2281 dxdy -0.04 -0.01 mag med 0.22
260 n 2157 dxdy -0.02 -0.02 mag med 0.16
270 n 2057 dxdy 0.05 -0.02 mag med 0.26
280 n 2118 dxdy 0.06 -0.1 mag med 0.33
290 n 2183 dxdy -0.24 -0.83 mag med 1.51
300 n 2298 dxdy -0.28 -5.08 mag med 5.29
310 n 2422 dxdy -1.63 -13.17 mag med 13.69
320 n 1875 dxdy 6.5 -2.03 mag med 7.14
390 n 1581 dxdy 4.0 2.02 mag med 5.0
400 n 1515 dxdy 2.12 0.49 mag med 2.22
410 n 1566 dxdy -0.03 0.0 mag med 0.31
420 n 1721 dxdy -0.02 -0.02 mag med 0.14
430 n 1715 dxdy -0.09 0.01 mag med 0.29
440  …[+1131 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,sys;sys.path.insert(0,'scratch');import static_opt as so
base=np.load('inputs/reconstruction.npz')['T_world_mesh'];W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);ref=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'));Sx=np.diag([1,-1,-1]);Rz=np.diag([-1,-1,1]);Ry=np.diag([-1,1,-1]);
for nm,ids,S,src in [('A',range(5,56,5),np.eye(3),ref),('B',range(225,291,5),np.eye(3),ref),('C',range(410,456,5),np.eye(3),base),('D',range(675,746,5),Sx,ref),('E',range(840,871,5),Rz,ref)]:
 ids=list(ids);R=so.meanR(src[ids,:3,:3])@S;t=np.median(src[ids,:3,3],0);print(nm,'R\\n',np.round(R,4),'t',np.round(t,4))
PY"
```
<details><summary>output</summary>

```
A R
 [[-0.8468 -0.0291 -0.5312]
 [ 0.0686 -0.9961 -0.0548]
 [-0.5275 -0.0828  0.8455]] t [0.3748 0.2968 3.1925]
B R
 [[-0.6834 -0.0358  0.7292]
 [-0.0485 -0.9944 -0.0943]
 [ 0.7284 -0.0998  0.6778]] t [-0.2767  1.2068  3.4274]
C R
 [[-0.8443 -0.0208 -0.5354]
 [ 0.0502 -0.9979 -0.0406]
 [-0.5335 -0.0611  0.8436]] t [0.2758 0.2845 3.2487]
D R
 [[ 0.4561  0.021   0.8897]
 [ 0.0696  0.9958 -0.0592]
 [-0.8872  0.089   0.4528]] t [-0.4362  1.2262  3.6619]
E R
 [[-0.9925 -0.0423  0.1147]
 [ 0.0269 -0.9908 -0.133 ]
 [ 0.1192 -0.1289  0.9845]] t [-0.101   0.3473  3.3902]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,sys;sys.path.insert(0,'scratch');import static_opt as so
base=np.load('inputs/reconstruction.npz')['T_world_mesh'];W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);ref=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'));Sx=np.diag([1,-1,-1]);Rz=np.diag([-1,-1,1]);vals={}
for nm,ids,S,src in [('A',range(5,56,5),np.eye(3),ref),('B',range(225,291,5),np.eye(3),ref),('C',range(410,456,5),np.eye(3),base),('D',range(675,746,5),Sx,ref),('E',range(840,871,5),Rz,ref)]:vals[nm]=so.meanR(src[list(ids),:3,:3])@S
for a,b in zip('ABCD', 'BCDE'):
 M=vals[a].T@vals[b];ang=np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)));w,v=np.linalg.eig(M);ax=np.real(v[:,np.argmin(abs(w-1))]);ax/=np.linalg.norm(ax);print(a,b,round(ang,2),np.round(ax,3))
PY"
```
<details><summary>output</summary>

```
A B 78.99 [-0.014 -1.    -0.003]
B C 79.33 [0.032 0.999 0.008]
C D 179.59 [ 0.739 -0.013 -0.674]
D E 179.69 [ 0.471  0.023 -0.882]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys;sys.path.insert(0,'scratch');import static_opt as so
b=np.load('inputs/reconstruction.npz')['T_world_mesh'];Sx=np.diag([1,-1,-1]);Rz=np.diag([-1,-1,1]);v={}
for nm,ids,S in [('A',range(5,56,5),np.eye(3)),('B',range(225,291,5),np.eye(3)),('C',range(410,456,5),np.eye(3)),('D',range(675,746,5),Sx),('E',range(840,871,5),Rz)]:v[nm]=so.meanR(b[list(ids),:3,:3])@S
for a,c in zip('ABCD','BCDE'):
 M=v[a].T@v[c];ang=np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1)));w,z=np.linalg.eig(M);ax=np.real(z[:,np.argmin(abs(w-1))]);ax/=np.linalg.norm(ax);print(a,c,round(ang,2),np.round(ax,3))
PY"
```
<details><summary>output</summary>

```
A B 78.99 [-0.014 -1.    -0.003]
B C 79.33 [0.032 0.999 0.008]
C D 84.74 [ 0.005 -1.     0.02 ]
D E 123.82 [0.003 1.    0.026]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json,sys;sys.path.insert(0,'scratch');import static_opt as so
b=np.load('inputs/reconstruction.npz')['T_world_mesh'];W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);ref=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'));pl=np.array(json.load(open('inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3];n=n/np.linalg.norm(n);d=d/np.linalg.norm(pl[:3]);m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))]
Sx=np.diag([1,-1,-1]);Rz=np.diag([-1,-1,1]);
for nm,ids,S,src in [('A',range(5,56,5),np.eye(3),ref),('B',range(225,291,5),np.eye(3),ref),('C',range(410,456,5),np.eye(3),b),('D',range(675,746,5),Sx,ref),('E',range(840,871,5),Rz,ref)]:
 ids=list(ids);R=so.meanR(src[ids,:3,:3])@S;t=np.median(src[ids,:3,3],0);Q=np.eye(4);Q[:3,:3]=R;Q[:3,3]=t;p=(V@Q.T)[:,:3];print(nm,'t',t,'yminmax',p[:,1].min(),p[:,1].max(),'floor min', (p@n+d).min(),'floormax',(p@n+d).max())
PY"
```
<details><summary>output</summary>

```
A t [0.37476482 0.29683814 3.19251106] yminmax 0.2554979460567496 0.33844353922871906 floor min 0.875648665168351 floormax 0.9419345129305499
B t [-0.2766961   1.20676962  3.42737857] yminmax 1.1623580435962044 1.2520429755265876 floor min -0.036206559642930625 floormax 0.035679346514103205
C t [0.27583674 0.28447595 3.2487345 ] yminmax 0.24649168223129403 0.32290176500512807 floor min 0.8899259101873214 floormax 0.9499654119853296
D t [-0.43618714  1.22621646  3.66189296] yminmax 1.1828433610014346 1.2682297167788856 floor min -0.06288044694705985 floormax 0.004879556955789344
E t [-0.10095941  0.34733972  3.39018231] yminmax 0.2996766700065214 0.39570199161155734 floor min 0.8164274264372127 floormax 0.8962015723397794

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/final_build.py <<'PY'
import numpy as np,json,sys,trimesh
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0,'scratch');import static_opt as so
# data
baseW=np.load('inputs/reconstruction.npz')['T_world_mesh'];W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W)
refW=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'))
N=len(baseW);Sx=np.diag([1.,-1.,-1.]);Rz=np.diag([-1.,-1.,1.]);Srel=Sx.T@Rz
# phase ranges (inclusive static plateaus verified by optical flow/mask stability)
A=(0,60);B=(225,290);C=(410,455);D=(675,745);E=(840,870)
def stat(ids,S,src):
 return so.meanR(src[list(ids),:3,:3])@S,np.median(src[list(ids),:3,3],axis=0)
RA,tA=stat(range(5,56,5),np.eye(3),refW)
RB,tB=stat(range(225,291,5),np.eye(3),refW)
RC,tC=stat(range(410,456,5),np.eye(3),baseW)
RD,tD=stat(range(675,746,5),Sx,refW)
RE,tE=stat(range(840,871,5),Rz,refW)
# physical support: floor phases must not penetrate released floor plane
pl=np.array(json.load(open('inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3];nn=np.linalg.norm(n);n=n/nn;d=d/nn
mesh=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[mesh.vertices,np.ones(len(mesh.vertices))]
def support_shift(R,t):
 P=(V@np.array([[R[0,0],R[0,1],R[0,2],t[0]],[R[1,0],R[1,1],R[1,2],t[1]],[R[2,0],R[2,1],R[2,2],t[2]],[0,0,0,1]]).T)[:,:3]
 return -(P@n+d).min()*n
tB=tB+support_shift(RB,tB);tD=tD+ …[+2716 chars]
```
<details><summary>output</summary>

```
anchors [('A', array([0.37476482, 0.29683814, 3.19251106])), ('B', array([-0.27737723,  1.17062525,  3.42536959])), ('C', array([0.27583674, 0.28447595, 3.2487345 ], dtype=float32)), ('D', array([-0.43737007,  1.16344403,  3.65840394])), ('E', array([-0.10095941,  0.34733972,  3.39018231]))]
saved (901, 4, 4)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/eval_final.py <<'PY'
import numpy as np,torch,cv2,json,h5py,io,sys
from scipy.ndimage import gaussian_filter
from PIL import Image
sys.path.insert(0,'scratch');import pose_opt as p
W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W)
B=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=np.load('scratch/final_world.npy')
hf=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def rend(T):
 vc=p.V@torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,:3].T+torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));return (r[0,:,:,3]>0).cpu().numpy(),(r[0,:,:,2]*10).cpu().numpy()
def met(i,tw):
 im,obj,per=p.get_frame(i);T=C@tw;m,z=rend(T);mvis=m&~per;iou=(mvis&obj).sum()/max(1,(mvis|obj).sum())
 crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));cs=[]
 for ch in range(3):
  aa=a[:,:,ch][mm];bbb=b[:,:,ch][mm];cs.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else -1)
 # depth residual
 raw=hf['aligned/front_stereo_came …[+1148 chars]
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.586 0.243 0.055 0.427]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.57  0.363 0.086 0.627]
Traceback (most recent call last):
  File "/work/scratch/eval_final.py", line 23, in <module>
    bm.append(met(i,B[i]));rm.append(met(i,R[i]))
              ^^^^^^^^^^^
  File "/work/scratch/eval_final.py", line 12, in met
    crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter( …[+116 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/final_build.py';s=open(p).read();s=s.replace(\"Tc=refW[:,:3,3].copy()\",\"Tc=baseW[:,:3,3].copy()\")
s=s.replace(\"sl=slice(lo+1,hi);Tc[sl]=np.stack([gaussian_filter1d(Tc[lo+1:hi,j],1.0,mode='nearest') for j in range(3)],1)\",\"sl=slice(lo+1,hi);Tc[sl]=np.stack([gaussian_filter1d(Tc[lo+1:hi,j],0.6,mode='nearest') for j in range(3)],1)\")
s=s.replace(\"sl=slice(E[1]+1,N);Tc[sl]=np.stack([gaussian_filter1d(Tc[E[1]+1:N,j],1.0,mode='nearest') for j in range(3)],1)\",\"sl=slice(E[1]+1,N);Tc[sl]=np.stack([gaussian_filter1d(Tc[E[1]+1:N,j],0.6,mode='nearest') for j in range(3)],1)\")
open(p,'w').write(s)
PY
python scratch/final_build.py >/dev/null
python scratch/eval_final.py"
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.606 0.305 0.191 0.895]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.571 0.362 0.089 0.655]
Traceback (most recent call last):
  File "/work/scratch/eval_final.py", line 23, in <module>
    bm.append(met(i,B[i]));rm.append(met(i,R[i]))
              ^^^^^^^^^^^
  File "/work/scratch/eval_final.py", line 12, in met
    crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter( …[+116 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/final_build.py';s=open(p).read()
old=\"\"\"intervals=[(A[1],B[0],tA,tB),(B[1],C[0],tB,tC),(C[1],D[0],tC,tD),(D[1],E[0],tD,tE)]
for lo,hi,s,e in intervals:
 if hi-lo<=1:continue
 ds=Tc[lo+1]-s;de=Tc[hi-1]-e
 for k,i in enumerate(range(lo+1,hi)):
  u=k/(hi-lo-1);Tc[i]=Tc[i]-(1-u)*ds-u*de
# post-E motion keeps refined, blends from E anchor
for i in range(E[1]+1,N):
 u=min(1,(i-E[1])/6);Tc[i]=Tc[i]+(1-u)*(tE-Tc[E[1]+1])
# smooth transitions only
for lo,hi,_,_ in intervals:
 sl=slice(lo+1,hi);Tc[sl]=np.stack([gaussian_filter1d(Tc[lo+1:hi,j],0.6,mode='nearest') for j in range(3)],1)
sl=slice(E[1]+1,N);Tc[sl]=np.stack([gaussian_filter1d(Tc[E[1]+1:N,j],0.6,mode='nearest') for j in range(3)],1)\"\"\"
new=\"\"\"intervals=[(A[1],B[0],tA,tB),(B[1],C[0],tB,tC),(C[1],D[0],tC,tD),(D[1],E[0],tD,tE)]
# Preserve the baseline's accurate per-frame motion in the middle of each
# transition, and blend only a short endpoint neighbourhood onto static anchors.
for lo,hi,s,e in intervals:
 if hi-lo<=1:continue
 for i in range(lo+1,hi):
  if i<=lo+8:
   u=(i-lo)/9.0;Tc[i]=Tc[i]+(1-u)*(s-Tc[lo+1])
  elif i>=hi-8:
   u=(hi-i)/9.0;Tc[i]=Tc[i]+(1-u)*(e-Tc[hi-1])
# post-E motion: baseline trajectory with a short blend away from E anchor
for i in range(E[1]+1,N):
 if i<=E[1]+8:
  u=(i-E[1])/9.0;Tc[i]=Tc[i]+(1-u)*(tE-Tc[E[1]+1])\"\"\"
assert old in s;s=s.replace(old,new);open(p,'w').write(s)
PY
python scratch/final_build.py >/dev/null
python scratch/eval_final.py"
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.662 0.289 0.158 0.676]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.623 0.342 0.095 0.709]
Traceback (most recent call last):
  File "/work/scratch/eval_final.py", line 23, in <module>
    bm.append(met(i,B[i]));rm.append(met(i,R[i]))
              ^^^^^^^^^^^
  File "/work/scratch/eval_final.py", line 12, in met
    crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter( …[+116 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/final_build.py';s=open(p).read();s=s.replace(\"B=(225,290);C=(410,455);D=(675,745);E=(840,870)\",\"B=(210,295);C=(400,455);D=(650,745);E=(840,870)\")
s=s.replace(\"RB,tB=stat(range(225,291,5),np.eye(3),refW)\",\"RB,tB=stat(range(210,296,5),np.eye(3),refW)\")
s=s.replace(\"RC,tC=stat(range(410,456,5),np.eye(3),baseW)\",\"RC,tC=stat(range(400,456,5),np.eye(3),baseW)\")
s=s.replace(\"RD,tD=stat(range(675,746,5),Sx,refW)\",\"RD,tD=stat(range(650,746,5),Sx,refW)\")
open(p,'w').write(s)
PY
python scratch/final_build.py >/dev/null
python scratch/eval_final.py"
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.647 0.293 0.155 0.671]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.517 0.414 0.086 0.642]
Traceback (most recent call last):
  File "/work/scratch/eval_final.py", line 23, in <module>
    bm.append(met(i,B[i]));rm.append(met(i,R[i]))
              ^^^^^^^^^^^
  File "/work/scratch/eval_final.py", line 12, in met
    crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter( …[+116 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_trans_options.py <<'PY'
import numpy as np,torch,cv2,json,sys
from scipy.ndimage import gaussian_filter
sys.path.insert(0,'scratch');import pose_opt as p,static_opt as so
W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W);base=np.load('inputs/reconstruction.npz')['T_world_mesh'];ref=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'));Sx=np.diag([1,-1,-1]);Rz=np.diag([-1,-1,1])
def ev(ids,R,t):
 ious=[];cs=[]
 for i in ids:
  im,obj,per=p.get_frame(i);Q=np.eye(4);Q[:3,:3]=R;Q[:3,3]=t;T=C@Q;crop=p.crop_bounds(obj)
  vc=p.V@torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,:3].T+torch.tensor(T,device=p.DEV,dtype=torch.float32)[:3,3];x=vc[:,0]/vc[:,2]*p.Kt[0,0]+p.Kt[0,2];y=vc[:,1]/vc[:,2]*p.Kt[1,1]+p.Kt[1,2];pos=torch.stack([2*x/p.W-1,2*y/p.H-1,vc[:,2]/10,torch.ones_like(x)],-1)[None];r,_=p.dr.rasterize(p.gl,pos,p.F,(p.H,p.W));m=(r[0,:,:,3]>0).cpu().numpy()&~per;ious.append((m&obj).sum()/max(1,(m|obj).sum()))
  x0,y0,x1,y1=crop;rgb,cov,_=p.render_crop(torch.tensor(T,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);mm=obj[y0:y1,x0:x1]&~per[y0:y1,x0:x1];tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));cc=[]
  for ch in range(3):
   aa=a[:,:,ch][mm];bbb=b[:,:,ch][mm];cc.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else -1)
  cs.append(np.mean(cc))
 ret …[+637 chars]
```
<details><summary>output</summary>

```
A Rbase tbase [0.374 0.283 3.102] (0.7885178414892253, 0.3062144581028115, 0.7879387048937222, 0.2968589651467242)
A Rref tbase [0.374 0.283 3.102] (0.7885178414892253, 0.3062144598836003, 0.7879387048937222, 0.2968589660942778)
A Rbase tref [0.374 0.297 3.186] (0.8010044924676976, 0.3959778996525859, 0.7973523421588594, 0.3876473897885493)
A Rref tref [0.374 0.297 3.186] (0.8010044924676976, 0.3959778925878211, 0.7973523421588594, 0.3876473820237756)
B Rbase tbase [-0.258  1.177  3.356] (0.8510243576012821, 0.3551056606596951, 0.8493061396131203, 0.3575496705390261)
B Rref tbase [-0.258  1.177  3.356] (0.8510243576012821, 0.3551056742243924, 0.8493061396131203, 0.35754968577574875)
B Rbase tref [-0.278  1.212  3.443] (0.8711662162323798, 0.2260537625340736, 0.8731497228521038, 0.222971828 …[+1526 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/final_build.py';s=open(p).read();s=s.replace(\"RB,tB=stat(range(210,296,5),np.eye(3),refW)\",\"RB,tB=stat(range(210,296,5),np.eye(3),baseW)\")
s=s.replace(\"RC,tC=stat(range(400,456,5),np.eye(3),baseW)\",\"RC,tC=stat(range(400,456,5),np.eye(3),baseW)\")
s=s.replace(\"RD,tD=stat(range(650,746,5),Sx,refW)\",\"RD,tD=stat(range(650,746,5),Sx,baseW)\")
# manually remove support shift lines
s=s.replace(\"tB=tB+support_shift(RB,tB);tD=tD+support_shift(RD,tD)\",\"# Baseline translations are already supported by the floor in these phases;\\n# the depth-map medians were biased downward and broke the visible support.\")
open(p,'w').write(s)
PY
python scratch/final_build.py
python scratch/eval_final.py"
```
<details><summary>output</summary>

```
anchors [('A', array([0.37476482, 0.29683814, 3.19251106])), ('B', array([-0.25824237,  1.1769005 ,  3.3561451 ], dtype=float32)), ('C', array([0.27583674, 0.28447595, 3.2529483 ], dtype=float32)), ('D', array([-0.39812368,  1.1561992 ,  3.4603314 ], dtype=float32)), ('E', array([-0.10095941,  0.34733972,  3.39018231]))]
saved (901, 4, 4)
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.661 0.278 0.157 0.676]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.851 0.355 0.105 0.842]
Traceback (most recent call last):
  File "/work/scratch/eval_final.py", line 23, in <module>
    bm.append(met(i,B[i]));rm.append(met(i,R[i]))
              ^^^^^^^^^^^
  File "/work/scratch/eval_final.py", line 12,  …[+457 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/eval_final.py';s=open(p).read();s=s.replace(\"def met(i,tw):\\n im,obj,per=p.get_frame(i);T=C@tw;m,z=rend(T)\",\"def met(i,tw):\\n im,obj,per=p.get_frame(i);\\n if obj.sum()<20:return np.nan,np.nan,np.nan,np.nan\\n T=C@tw;m,z=rend(T)\")
open(p,'w').write(s)
PY
python scratch/eval_final.py"
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.661 0.278 0.157 0.676]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.851 0.355 0.105 0.842]
B>C base iou/corr/depth [0.642 0.284 0.242 0.772] final [0.644 0.288 0.242 0.771]
C base iou/corr/depth [0.801 0.416 0.033 0.219] final [0.806 0.417 0.037 0.29 ]
C>D base iou/corr/depth [0.649 0.329 0.121 0.676] final [0.446 0.021 0.14  0.754]
D base iou/corr/depth [0.859 0.356 0.206 1.   ] final [0.802 0.551 0.214 1.   ]
D>E base iou/corr/depth [0.676 0.258 0.619 0.912] final [0.63  0.162 0.603 0.86 ]
E base iou/corr/depth [0.267 0.216 0.088 0.704] final [0.321 0.435 0.101 0.676]
base rotstep pct [ 1.25  7.23 11.76 25.72 34.99] n>10 58 trans pct [0.01 …[+137 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/final_build.py';s=open(p).read();old=\"\"\"for i in range(0,N):
 if A[1]<i<B[0]: c=CA
 elif B[1]<i<C[0]: c=CB
 elif C[1]<i<D[0]: c=blend_corr(i,C[1],D[0],CC,CD)
 elif D[1]<i<E[0]: c=blend_corr(i,D[1],E[0],CD,CE)
 elif E[1]<i: c=CE
 else: continue
 Rout[i]=baseW[i,:3,:3]@c\"\"\"
new=\"\"\"for i in range(0,N):
 if A[1]<i<B[0]: c=CA
 elif B[1]<i<C[0]: c=CB
 elif C[1]<i<=515: c=CC
 elif 515<i<540: c=blend_corr(i,515,540,CC,CD)
 elif 540<=i<830: c=CD
 elif 830<=i<840: c=blend_corr(i,830,840,CD,CE)
 elif i>=840: c=CE
 else: continue
 Rout[i]=baseW[i,:3,:3]@c\"\"\"
assert old in s;s=s.replace(old,new);open(p,'w').write(s)
PY
python scratch/final_build.py >/dev/null
python scratch/eval_final.py"
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.661 0.278 0.157 0.676]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.851 0.355 0.105 0.842]
B>C base iou/corr/depth [0.642 0.284 0.242 0.772] final [0.644 0.288 0.242 0.771]
C base iou/corr/depth [0.801 0.416 0.033 0.219] final [0.806 0.417 0.037 0.29 ]
C>D base iou/corr/depth [0.649 0.329 0.121 0.676] final [0.622 0.28  0.122 0.669]
D base iou/corr/depth [0.859 0.356 0.206 1.   ] final [0.824 0.542 0.212 1.   ]
D>E base iou/corr/depth [0.676 0.258 0.619 0.912] final [0.68  0.286 0.615 0.855]
E base iou/corr/depth [0.267 0.216 0.088 0.704] final [0.321 0.435 0.101 0.676]
base rotstep pct [ 1.25  7.23 11.76 25.72 34.99] n>10 58 trans pct [0.01 …[+137 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys;sys.path.insert(0,'scratch');import eval_final as e
B=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=np.load('scratch/final_world.npy')
for i in range(460,626,5):
 a=e.met(i,B[i]);b=e.met(i,R[i]);print(i,'iou',round(a[0],2),round(b[0],2),'corr',round(a[1],2),round(b[1],2))
PY"
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.798 0.378 0.023 0.131]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.661 0.278 0.157 0.676]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.851 0.355 0.105 0.842]
B>C base iou/corr/depth [0.642 0.284 0.242 0.772] final [0.644 0.288 0.242 0.771]
C base iou/corr/depth [0.801 0.416 0.033 0.219] final [0.806 0.417 0.037 0.29 ]
C>D base iou/corr/depth [0.649 0.329 0.121 0.676] final [0.622 0.28  0.122 0.669]
D base iou/corr/depth [0.859 0.356 0.206 1.   ] final [0.824 0.542 0.212 1.   ]
D>E base iou/corr/depth [0.676 0.258 0.619 0.912] final [0.68  0.286 0.615 0.855]
E base iou/corr/depth [0.267 0.216 0.088 0.704] final [0.321 0.435 0.101 0.676]
base rotstep pct [ 1.25  7.23 11.76 25.72 34.99] n>10 58 trans pct [0.01 …[+1246 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/final_build.py';s=open(p).read();s=s.replace(\"elif C[1]<i<=515: c=CC\\n elif 515<i<540: c=blend_corr(i,515,540,CC,CD)\\n elif 540<=i<830: c=CD\",\"elif C[1]<i<=534: c=CC\\n elif 534<i<540: c=blend_corr(i,534,540,CC,CD)\\n elif 540<=i<830: c=CD\");open(p,'w').write(s)
PY
python scratch/final_build.py >/dev/null"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed '/"'^ph=/,/''^for nm,T in/d'"' scratch/eval_final.py > /tmp/eval_prefix.py
cat /tmp/eval_prefix.py > scratch/quick_compare.py
cat >> scratch/quick_compare.py <<'PY'
B=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=np.load('scratch/final_world.npy')
for i in range(515,566,5):
 a=met(i,B[i]);b=met(i,R[i]);print(i,'iou',round(a[0],2),round(b[0],2),'corr',round(a[1],2),round(b[1],2))
PY
python scratch/quick_compare.py"
```
<details><summary>output</summary>

```
515 iou 0.86 0.86 corr 0.69 0.7
520 iou 0.85 0.85 corr 0.8 0.8
525 iou 0.87 0.87 corr 0.74 0.76
530 iou 0.93 0.93 corr 0.33 0.34
535 iou 0.88 0.76 corr 0.45 -0.11
540 iou 0.84 0.78 corr 0.36 0.45
545 iou 0.85 0.84 corr 0.23 0.22
550 iou 0.91 0.89 corr 0.49 0.54
555 iou 0.88 0.86 corr 0.1 0.1
560 iou 0.86 0.84 corr -0.05 -0.03
565 iou 0.61 0.61 corr 0.22 0.18

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/final_build.py';s=open(p).read();s=s.replace(\"elif C[1]<i<=534: c=CC\\n elif 534<i<540: c=blend_corr(i,534,540,CC,CD)\",\"elif C[1]<i<=536: c=CC\\n elif 536<i<540: c=blend_corr(i,536,540,CC,CD)\");open(p,'w').write(s)
PY
python scratch/final_build.py >/dev/null
python scratch/quick_compare.py"
```
<details><summary>output</summary>

```
515 iou 0.86 0.86 corr 0.69 0.7
520 iou 0.85 0.85 corr 0.8 0.8
525 iou 0.87 0.87 corr 0.74 0.76
530 iou 0.93 0.93 corr 0.33 0.34
535 iou 0.88 0.88 corr 0.45 0.46
540 iou 0.84 0.77 corr 0.36 0.45
545 iou 0.85 0.84 corr 0.23 0.22
550 iou 0.91 0.89 corr 0.49 0.54
555 iou 0.88 0.86 corr 0.1 0.1
560 iou 0.86 0.84 corr -0.05 -0.03
565 iou 0.61 0.61 corr 0.22 0.18

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -80 scratch/final_build.py'
```
<details><summary>output</summary>

```
# data
baseW=np.load('inputs/reconstruction.npz')['T_world_mesh'];W=np.array(json.load(open('inputs/camera.json'))['T_world_cam']);C=np.linalg.inv(W)
refW=np.einsum('ij,fjk->fik',W,np.load('scratch/refined_cam.npy'))
N=len(baseW);Sx=np.diag([1.,-1.,-1.]);Rz=np.diag([-1.,-1.,1.]);Srel=Sx.T@Rz
# phase ranges (inclusive static plateaus verified by optical flow/mask stability)
A=(0,60);B=(210,295);C=(400,455);D=(650,745);E=(840,870)
def stat(ids,S,src):
 return so.meanR(src[list(ids),:3,:3])@S,np.median(src[list(ids),:3,3],axis=0)
RA,tA=stat(range(5,56,5),np.eye(3),refW)
RB,tB=stat(range(210,296,5),np.eye(3),baseW)
RC,tC=stat(range(400,456,5),np.eye(3),baseW)
RD,tD=stat(range(650,746,5),Sx,baseW)
RE,tE=stat(range(840,871,5),Rz,refW)
# physical support: floor phases must not penetrate released  …[+3246 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally coherent world-frame trajectory for the light-blue book.

The method uses only the released inputs, the deterministic CARI4D intermediate
poses, SAM2 masks, and the aligned MoGe-2 depth maps.  It has four components:
  1. static-phase detection by image-mask/optical-flow plateaus (constants below);
  2. render-and-depth anchor refinement for two unusually ambiguous plateaus;
  3. symmetry-branch selection from pooled textured-mesh evidence;
  4. endpoint-anchored trajectory construction that preserves the baseline'"'s
     correct per-frame motion and removes static drift / rotational jumps.

No ground-truth data or challenge annotations are used.
\"\"\"
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from PIL import Image
from scipy.spatial.transform import Rotation


def mean_rotation(mats: np.ndarray) -> np.ndarray:
    \"\"\"Robust-ish quaternion mean with hemisphere alignment.\"\"\"
    q = Rotation.from_matrix(np.asarray(mats)).as_quat()
    ref = q[0]
    q = q * np.where(q @ ref < 0.0, -1.0, 1.0)[:, None]
    q = q.mean(axis=0)
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--inputs\", default=\"/work/inputs\")
    ap.add …[+11264 chars]
```
<details><summary>output</summary>

```
Static anchors:
  A: t=[0.3744 0.2968 3.186 ]
  B: t=[-0.2582  1.1769  3.3561]
  C: t=[0.2758 0.2845 3.2529]
  D: t=[-0.3981  1.1562  3.4603]
  E: t=[-0.1056  0.3238  3.3905]
Wrote /work/outputs/object_poses_world.npz (901 poses)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c 'sed "s#scratch/final_world.npy#outputs/object_poses_world.npz#" scratch/eval_final.py > scratch/eval_solution.py
python scratch/eval_solution.py'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_solution.py", line 25, in <module>
    bm.append(met(i,B[i]));rm.append(met(i,R[i]))
                                           ~^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 263, in __getitem__
    raise KeyError(f"{key} is not a file in the archive")
KeyError: '0 is not a file in the archive'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/eval_solution.py';s=open(p).read();s=s.replace(\"R=np.load('outputs/object_poses_world.npz')\",\"R=np.load('outputs/object_poses_world.npz')['T_world_mesh']\");open(p,'w').write(s)
PY
python scratch/eval_solution.py"
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.801 0.396 0.021 0.124]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.661 0.278 0.157 0.676]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.851 0.355 0.105 0.842]
B>C base iou/corr/depth [0.642 0.284 0.242 0.772] final [0.644 0.288 0.242 0.771]
C base iou/corr/depth [0.801 0.416 0.033 0.219] final [0.806 0.417 0.037 0.29 ]
C>D base iou/corr/depth [0.649 0.329 0.121 0.676] final [0.645 0.345 0.122 0.68 ]
D base iou/corr/depth [0.859 0.356 0.206 1.   ] final [0.824 0.542 0.212 1.   ]
D>E base iou/corr/depth [0.676 0.258 0.619 0.912] final [0.695 0.267 0.616 0.871]
E base iou/corr/depth [0.267 0.216 0.088 0.704] final [0.435 0.022 0.104 0.688]
base rotstep pct [ 1.25  7.23 11.76 25.72 34.99] n>10 58 trans pct [0.01 …[+137 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
A=np.load('scratch/final_world.npy');B=np.load('outputs/object_poses_world.npz')['T_world_mesh']
for i in [840,850,860,870,880,890,900]:
 dA=A[i,:3,:3];dB=B[i,:3,:3];ang=np.degrees(np.arccos(np.clip((np.trace(dA.T@dB)-1)/2,-1,1)))
 print(i,'ang',round(ang,2),'tA',np.round(A[i,:3,3],3),'tB',np.round(B[i,:3,3],3),'dt',np.round(B[i,:3,3]-A[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='outputs/solve.py';s=open(p).read();s=s.replace(\"iou0 = mask_iou(m0 & ~person, obj)\",\"iou0 = mask_iou(m0, obj)\")
s=s.replace(\"mvis = m & ~person\\n            if mvis.sum() > 20 and obj.sum() > 20:\\n                yy, xx = np.where(mvis)\",\"if m.sum() > 20 and obj.sum() > 20:\\n                yy, xx = np.where(m)\")
s=s.replace(\"iou2 = mask_iou(m2 & ~person, obj)\",\"iou2 = mask_iou(m2, obj)\")
open(p,'w').write(s)
PY
python outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
Static anchors:
  A: t=[0.3744 0.2968 3.186 ]
  B: t=[-0.2582  1.1769  3.3561]
  C: t=[0.2758 0.2845 3.2529]
  D: t=[-0.3981  1.1562  3.4603]
  E: t=[-0.101   0.3473  3.3902]
Wrote /work/outputs/object_poses_world.npz (901 poses)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python scratch/eval_solution.py > scratch/final_validation.log 2>&1; cat scratch/final_validation.log'
```
<details><summary>output</summary>

```
A base iou/corr/depth [0.789 0.306 0.096 0.911] final [0.801 0.396 0.021 0.124]
A>B base iou/corr/depth [0.661 0.287 0.159 0.68 ] final [0.661 0.278 0.157 0.676]
B base iou/corr/depth [0.858 0.334 0.103 0.813] final [0.851 0.355 0.105 0.842]
B>C base iou/corr/depth [0.642 0.284 0.242 0.772] final [0.644 0.288 0.242 0.771]
C base iou/corr/depth [0.801 0.416 0.033 0.219] final [0.806 0.417 0.037 0.29 ]
C>D base iou/corr/depth [0.649 0.329 0.121 0.676] final [0.645 0.345 0.122 0.68 ]
D base iou/corr/depth [0.859 0.356 0.206 1.   ] final [0.824 0.542 0.212 1.   ]
D>E base iou/corr/depth [0.676 0.258 0.619 0.912] final [0.68  0.286 0.615 0.855]
E base iou/corr/depth [0.267 0.216 0.088 0.704] final [0.321 0.435 0.101 0.676]
base rotstep pct [ 1.25  7.23 11.76 25.72 34.99] n>10 58 trans pct [0.01 …[+137 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
B=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=np.load('outputs/object_poses_world.npz')['T_world_mesh']
def steps(T,a,b):
 out=[]
 for i in range(a,b):
  M=T[i,:3,:3].T@T[i+1,:3,:3];out.append(np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1))))
 return out
for nm,T in [('base',B),('final',R)]:
 print(nm)
 for i in range(525,546): print(i,round(steps(T,i,i+1)[0],1),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
base
525 4.2; 526 5.3; 527 2.2; 528 4.0; 529 1.8; 530 2.7; 531 2.2; 532 3.4; 533 3.4; 534 3.6; 535 4.9; 536 3.5; 537 3.2; 538 4.0; 539 2.7; 540 3.7; 541 1.9; 542 1.3; 543 1.5; 544 1.2; 545 2.2; 
final
525 3.7; 526 4.5; 527 2.7; 528 3.4; 529 1.8; 530 2.2; 531 2.2; 532 3.2; 533 3.4; 534 3.7; 535 9.0; 536 41.2; 537 46.9; 538 46.6; 539 40.8; 540 8.5; 541 2.0; 542 1.3; 543 1.2; 544 0.9; 545 1.8; 

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
B=np.load('inputs/reconstruction.npz')['T_world_mesh'];R=np.load('outputs/object_poses_world.npz')['T_world_mesh']
outs=[]
for i in range(532,545,2):
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj)
 for nm,T in [('B',B),('F',R)]:
  p.save_overlay(i,T[i],crop,f'scratch/cmp_{i}_{nm}.png',4);im2=cv2.imread(f'scratch/cmp_{i}_{nm}.png');im2=cv2.resize(im2,(300,180));cv2.putText(im2,f'{i} {nm}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(im2,f'{i} {nm}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);outs.append(im2)
c=np.zeros((2*180,7*300,3),np.uint8)
for j,im in enumerate(outs):c[(j//7)*180:(j//7+1)*180,(j%7)*300:(j%7+1)*300]=im
cv2.imwrite('scratch/cmp_532_544.jpg',c)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/work/scratch/pose_opt.py", line 118, in save_overlay
    im,obj,per=get_frame(i);rgb,cov,(w,h)=render_crop(torch.tensor(Tc,device=DEV),crop,S)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_opt.py", line 27, in render_crop
    vc=V@Tc[:3,:3].T+Tc[:3,3]
       ~^~~~~~~~~~~~
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
B=np.load('inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32);R=np.load('outputs/object_poses_world.npz')['T_world_mesh'].astype(np.float32)
outs=[]
for i in range(532,545,2):
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj)
 for nm,T in [('B',B),('F',R)]:
  p.save_overlay(i,T[i],crop,f'scratch/cmp_{i}_{nm}.png',4);im2=cv2.imread(f'scratch/cmp_{i}_{nm}.png');im2=cv2.resize(im2,(300,180));cv2.putText(im2,f'{i} {nm}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(im2,f'{i} {nm}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);outs.append(im2)
c=np.zeros((2*180,7*300,3),np.uint8)
for j,im in enumerate(outs):c[(j//7)*180:(j//7+1)*180,(j%7)*300:(j%7+1)*300]=im
cv2.imwrite('scratch/cmp_532_544.jpg',c)
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
R=np.load('outputs/object_poses_world.npz')['T_world_mesh']
a=[]
for i in range(900):
 M=R[i,:3,:3].T@R[i+1,:3,:3];a.append(np.degrees(np.arccos(np.clip((np.trace(M)-1)/2,-1,1))))
a=np.array(a)
for i in np.argsort(a)[-20:][::-1]:print(i,round(a[i],2),np.round(R[i,:3,3],2),np.round(R[i+1,:3,3],2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2;sys.path.insert(0,'scratch');import pose_opt as p
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
for i in range(534,542):
 im,obj,per=p.get_frame(i);crop=p.crop_bounds(obj);x0,y0,x1,y1=crop;o=obj[y0:y1,x0:x1];pp=per[y0:y1,x0:x1];m=o&~pp
 vals=[]
 for deg in range(0,181,20):
  S=Rotation.from_rotvec(np.array([np.radians(deg),0,0])).as_matrix();Q=p.BASEc[i].copy();Q[:3,:3]=Q[:3,:3]@S
  rgb,cov,(w,h)=p.render_crop(torch.tensor(Q,device=p.DEV,dtype=torch.float32),crop,2);rr=cv2.resize(rgb.cpu().numpy(),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA);tt=im[y0:y1,x0:x1].astype(np.float32)/255;a=gaussian_filter(tt,(1.2,1.2,0));b=gaussian_filter(rr,(1.2,1.2,0));cs=[]
  for ch in range(3):
   aa=a[:,:,ch][m];bbb=b[:,:,ch][m];cs.append(np.corrcoef(aa,bbb)[0,1] if aa.std()>1e-4 and bbb.std()>1e-4 else -1)
  vals.append(np.mean(cs))
 print(i,'best',np.argmax(vals)*20,np.round(vals,2))
PY"
```
<details><summary>output</summary>

```
534 best 180 [ 0.5  -0.03 -0.16 -0.2  -0.19 -0.14 -0.22 -0.11  0.19  0.52]
535 best 180 [ 0.45 -0.01 -0.15 -0.16 -0.18 -0.16 -0.25 -0.13  0.17  0.47]
536 best 180 [ 0.54  0.11 -0.07 -0.12 -0.17 -0.15 -0.23 -0.13  0.22  0.57]
537 best 180 [ 0.3  -0.08 -0.15 -0.16 -0.18 -0.16 -0.24 -0.12  0.19  0.32]
538 best 180 [ 0.37 -0.07 -0.09 -0.11 -0.1  -0.09 -0.13 -0.07  0.25  0.4 ]
539 best 160 [ 0.24 -0.17 -0.16 -0.12 -0.1  -0.07 -0.1  -0.04  0.26  0.25]
540 best 180 [ 0.36 -0.08 -0.08 -0.06 -0.04 -0.01 -0.04 -0.    0.32  0.39]
541 best 180 [ 0.36  0.   -0.02 -0.02 -0.04 -0.05 -0.06 -0.04  0.33  0.38]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys;sys.path.insert(0,'scratch');import static_opt as so
r=np.load('inputs/reconstruction.npz');kp=r['human_keypoints_world'];T=np.load('inputs/reconstruction.npz')['T_world_mesh'];Sx=np.diag([1,-1,-1]);Rz=np.diag([-1,-1,1]);
def handframe(i,side):
 off=21 if side=='R' else 42; h=kp[i,off:off+21]
 wrist=h[0];ind=h[5];mid=h[9];ring=h[13];pinky=h[17]
 x=ind-pinky;x/=np.linalg.norm(x); y=(mid+ring)/2-wrist;y/=np.linalg.norm(y);z=np.cross(x,y);z/=np.linalg.norm(z);y=np.cross(z,x);return np.stack([x,y,z],1)
def ang(A,B):return np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1)))
for side in ['R','L']:
 print(side)
 for nm,ids,S in [('C',range(400,456,5),np.eye(3)),('D',range(650,746,5),Sx),('E',range(840,871,5),Rz)]:
  rels=[]
  for i in ids:
   H=handframe(i,side);O=T[i,:3,:3]@S;rels.append(H.T@O)
  m=so.meanR(np.array(rels));spread=[ang(m,x) for x in rels]
  print(nm,'rel spread',round(np.median(spread),1),round(np.max(spread),1))
 # compare relative at means
 Hc=handframe(425,side);Hd=handframe(700,side);He=handframe(855,side);Rc=so.meanR(T[list(range(400,456,5)),:3,:3]);Rd=so.meanR(T[list(range(650,746,5)),:3,:3])@Sx;Re=so.meanR(T[list(range(840,871,5)),:3,:3])@Rz
 print('C-D relative rot',ang(Hc.T@Rc,Hd.T@Rd),'D-E',ang(Hd.T@Rd,He.T@Re))
PY"
```
<details><summary>output</summary>

```
R
C rel spread 14.2 28.8
D rel spread 6.2 9.8
E rel spread 4.2 14.2
C-D relative rot 172.85641094228666 D-E 150.03139455228708
L
C rel spread 11.7 24.6
D rel spread 8.5 10.8
E rel spread 3.1 14.9
C-D relative rot 152.19072229033648 D-E 149.39408906566112

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('inputs/reconstruction.npz');kp=r['human_keypoints_world']
def hf(i,off):
 h=kp[i,off:off+21];wrist=h[0];ind=h[5];mid=h[9];ring=h[13];pinky=h[17];x=ind-pinky;x/=np.linalg.norm(x);y=(mid+ring)/2-wrist;y/=np.linalg.norm(y);z=np.cross(x,y);z/=np.linalg.norm(z);y=np.cross(z,x);return np.stack([x,y,z],1)
def ang(a,b):return np.degrees(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1)))
for side,off in [('R',21),('L',42)]:
 print(side)
 for i in list(range(525,546))+list(range(830,841)):
  print(i,round(ang(hf(i,off),hf(i+1,off)),1),end=';')
 print()
PY"
```
<details><summary>output</summary>

```

```
</details>
