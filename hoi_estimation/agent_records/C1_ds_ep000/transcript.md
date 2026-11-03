# C1_ds_ep000

Codex DeepSeek 估计相机  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T15:28:14Z

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/AGENT_TASK.md'
```
<details><summary>output</summary>

```
# Task: estimate the camera (and with it the world trajectory of the object) of one monocular HOI video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released WORLD frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose, in the world frame; no symmetry handling, no alignment), averaged over thresholds
1-9 cm.

This task is …[+9174 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%p %s bytes\\n' | sort && echo '--- camera.json ---' && cat /work/inputs/camera.json && echo '--- ground_plane.json ---' && cat /work/inputs/ground_plane.json && echo '--- episode.json ---' && cat /work/inputs/episode.json && echo '--- task.txt ---' && cat /work/inputs/task.txt"
```
<details><summary>output</summary>

```
/work/inputs/camera.json 837 bytes
/work/inputs/episode.json 68 bytes
/work/inputs/frame0_reference_human.npz 192762 bytes
/work/inputs/ground_plane.json 396 bytes
/work/inputs/masks.h5 7634672 bytes
/work/inputs/object_mesh.glb 14269716 bytes
/work/inputs/reconstruction.npz 39369502 bytes
/work/inputs/task.txt 122 bytes
/work/inputs/video.mp4 24069401 bytes
--- camera.json ---
{
 "K": [
  [
   819.9987182617188,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9987182617188,
   560.5000610351562
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "K_pipeline": [
  [
   818.9199829101562,
   0.0,
   768.0
  ],
  [
   0.0,
   818.919921875,
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
   0.9994056724264703,
   -0.00630215050390631,
   0.03389077781375614,
   0.05715221830 …[+1068 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 974843416
/pipeline/depth/moge2/depth_report.json 1804
/pipeline/depth/moge2/intrinsics.pkl 82875
/pipeline/depth/moge2/raw.npy 2753298560
/pipeline/export/episode_000000/edex 559
/pipeline/export/episode_000000/wild_export.json 2778
/pipeline/foundationpose/episode_000000.pkl 222559
/pipeline/inference/coconet.pth 595065104
/pipeline/inference/episode_000000_step200000_before_after.mp4 57953578
/pipeline/inference/episode_000000_step200000_reconstruction.mp4 21223419
/pipeline/inference/refined.pth 595128736
/pipeline/mhr_init/episode_000000.pkl 3472357
/pipeline/mhr_init/episode_000000.sam3d.h5 782407
/pipeline/mhr_init/episode_000000.stats.pkl 7507
/pipeline/pipeline_report.json 5837
/pipeline/work/ …[+104 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
p='/work/inputs/reconstruction.npz'
z=np.load(p)
print('reconstruction keys/shapes/dtypes:')
for k in z.files:
 print(k,z[k].shape,z[k].dtype, float(np.nanmin(z[k])) if np.issubdtype(z[k].dtype,np.number) else '')
p='/work/inputs/frame0_reference_human.npz'
z=np.load(p)
print('human keys/shapes:')
for k in z.files:
 print(k,z[k].shape,z[k].dtype, float(np.nanmin(z[k])) if np.issubdtype(z[k].dtype,np.number) else '')
PY"
```
<details><summary>output</summary>

```
reconstruction keys/shapes/dtypes:
T_world_mesh (778, 4, 4) float32 -0.7801939249038696
T_cam_mesh_foundationpose (778, 4, 4) float32 -0.7555374503135681
T_cam_mesh_coconet (778, 4, 4) float32 -0.7507079839706421
human_vertices_world (778, 4610, 3) float32 -1.0448040962219238
human_keypoints_world (778, 70, 3) float32 -1.0350888967514038
contact_prob_right_left (778, 2) float32 1.1246928735553219e-11
human keys/shapes:
vertices_world (18056, 3) float32 -0.8610458374023438

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work -maxdepth 2 -type f -printf '%p %s\\n' | sort | head -100; echo '--- GPU ---'; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true; echo '--- Python ---'; python - <<'PY'
import cv2,trimesh,numpy,scipy,torch
print('cv2',cv2.__version__,'trimesh',trimesh.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__,'torch',torch.__version__)
PY"
```
<details><summary>output</summary>

```
cv2 4.11.0 trimesh 5.1.1 numpy 1.26.3 scipy 1.16.3 torch 2.5.1+cu124

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/depth/moge2/depth_report.json; echo '--- report'; cat /pipeline/pipeline_report.json; echo '--- export'; cat /pipeline/export/episode_000000/wild_export.json; echo '--- prompts'; cat /pipeline/work/prompts.json"
```
<details><summary>output</summary>

```
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 9.494523048400879,
    "min": 1.501441240310669
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 71.26411462604301,
  "firstFrameIntrinsicsPx": [
    [
      818.9199829101562,
      0.0,
      768.0
    ],
    [
      0.0,
      818.919921875,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "focalXPx": {
    "max": 829.91943359375,
    "mean": 816.3920288085938,
    "min": 790.4642944335938
  },
  "focalYPx": {
    "max": 829.9193725585938,
    "mean": 816.3920288085938,
    "min": 790.4642333984375
  },
  "fps": 30.0,
  "frames": 778,
  "framesPerSecond": 10.917135560899608,
  "height": 1152,
  "intrinsicsFile": {
     …[+10249 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, numpy as np, os
for p in ['/pipeline/depth/moge2/intrinsics.pkl','/pipeline/mhr_init/episode_000000.pkl','/pipeline/mhr_init/episode_000000.stats.pkl','/pipeline/foundationpose/episode_000000.pkl']:
 print('\\n===',p)
 with open(p,'rb') as f: z=pickle.load(f)
 print(type(z))
 if isinstance(z,dict):
  for k,v in z.items():
   print(k, type(v), getattr(v,'shape',None), getattr(v,'dtype',None), repr(v)[:200])
 else: print(repr(z)[:1000])
PY"
```
<details><summary>output</summary>

```
 <class 'list'> None None [[[818.9199829101562, 0.0, 768.0], [0.0, 818.919921875, 576.0], [0.0, 0.0, 1.0]], [[812.6511840820312, 0.0, 768.0], [0.0, 812.6511840820312, 576.0], [0.0, 0.0, 1.0]], [[815.6727294921875, 0.0, 768.0],
H <class 'int'> None None 1152
W <class 'int'> None None 1536
depth_backend <class 'str'> None None 'moge2'
camera_policy <class 'str'> None None 'first_frame_export_per_frame_depth'
model_id <class 'str'> None None 'Ruicheng/moge-2-vitl-normal'
model_revision <class 'str'> None None 'b135031bae30b5ac2ae141a0e68717795ce38340'
source_commit <class 'str'> None None '925b8ed835a7a9cdb7578ba15c658a0afc969030'
batch_size <class 'int'> None None 8
devices <class 'list'> None None ['cuda']

=== /pipeline/mhr_init/episode_000000.pkl
<class 'dict'>
body_model <class 'str'> No …[+12606 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction /workspace -type f \\( -iname '*mhr*' -o -iname '*.obj' -o -iname '*.npy' -o -iname '*.npz' \\) 2>/dev/null | head -200"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/scripts/build_mhr_dataset_index.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/models/mhr_heads.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_object_pose_loss.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_input_grid.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_supervision.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_losses.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/datasets/mhr_loader_config.py
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/datasets/mhr_sample_metadata.py
/code/recon …[+13636 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/work/inputs/masks.h5','/pipeline/mhr_init/episode_000000.sam3d.h5','/pipeline/work/masks_k0.h5','/pipeline/depth/moge2/aligned.h5']:
 print('\\n===',p)
 with h5py.File(p,'r') as f:
  def show(name,obj):
   if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype,obj.chunks)
  f.visititems(show)
PY"
```
<details><summary>output</summary>

```

=== /work/inputs/masks.h5
object (778, 1152, 1536) bool (25, 72, 96)
person (778, 1152, 1536) bool (25, 72, 96)

=== /pipeline/mhr_init/episode_000000.sam3d.h5
frames (778,) object None
predictions/body_pose_params (778, 133) float32 (98, 34)
predictions/expr_params (778, 72) float32 (195, 18)
predictions/global_rot (778, 3) float32 (778, 3)
predictions/hand_pose_params (778, 108) float32 (98, 27)
predictions/pred_cam_t (778, 3) float32 (778, 3)
predictions/scale_params (778, 28) float32 (195, 14)
predictions/shape_params (778, 45) float32 (195, 12)

=== /pipeline/work/masks_k0.h5
episode_000000/000000-k0.obj_rend_mask.png (1152, 1536) uint8 (144, 192)
episode_000000/000000-k0.person_mask.png (1152, 1536) uint8 (144, 192)
episode_000000/000001-k0.obj_rend_mask.png (1152, 1536) uint8 (144, …[+112202 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction/modules/v2d_sam3d_body -maxdepth 4 -type f | sed -n '1,240p'"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_sam3d_body/lib/__init__.py
/code/reconstruction/modules/v2d_sam3d_body/lib/download_weights.py
/code/reconstruction/modules/v2d_sam3d_body/lib/export_soma.py
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.yaml
/code/reconstruction/modules/v2d_sam3d_body/lib/tests/test_mv_optimize_mhr_params_init.py
/code/reconstruction/modules/v2d_sam3d_body/lib/tests/test_download_weights.py
/code/reconstruction/modules/v2d_sam3d_body/lib/tests/test_export_soma_options.py
/code/reconstruction/modules/v2d_sam3d_body/lib/tests/test_estimate_mhr_mask_prompts.py
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py
/code/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py
/code/reconstruction/modules/v2d_sam3d_body/lib/t …[+3379 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"18056\\|faces\\|triangles\\|mhr.*npz\\|mhr.*pkl\" -n /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body /code/reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr 2>/dev/null | head -240"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py:87:        self.faces = nn.Parameter(torch.zeros(36874, 3).long(), requires_grad=False)
/code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py:383:            "faces": self.faces.cpu().numpy(),
/code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/visualization/renderer.py:136:    def __init__(self, focal_length, faces=None):
/code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/visualization/renderer.py:141:            faces (np.array): Array of shape (F, 3) containing the mesh faces.
/code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/visualization/renderer.py:145:        self.faces = faces
/code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/visua …[+25956 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /weights -maxdepth 4 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/weights/cari4d/.cache/huggingface/.gitignore 1
/weights/cari4d/.cache/huggingface/CACHEDIR.TAG 191
/weights/cari4d/2026-08-25-09-35-57/manifest.json 449
/weights/cari4d/2026-08-25-09-35-57/resolved_config.yaml 9842
/weights/cari4d/2026-08-25-09-35-57/step200000.pth 2084484133
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/config.yml 708
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth 68220109
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/config.yml 778
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth 190229389
/weights/grounding_dino/groundingdino_swint_ogc.pth 693997677
/weights/hf_home/.agent_harnesses.json 5778
/weights/hf_home/.check_for_skill_update_done 0
/weights/hf_home/.check_for_update_done 0
/weights/hf …[+930 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, numpy as np
for p in ['/pipeline/mhr_init/episode_000000.pkl','/pipeline/foundationpose/episode_000000.pkl']:
 print('\\n===',p)
 z=pickle.load(open(p,'rb'))
 for k,v in z.items():
  if isinstance(v,np.ndarray): print(k,v.shape,v.dtype)
  elif isinstance(v,list): print(k,'list',len(v))
  elif isinstance(v,dict): print(k,'dict keys',list(v)[:20])
  else: print(k,type(v).__name__,str(v)[:100])
PY"
```
<details><summary>output</summary>

```

=== /pipeline/mhr_init/episode_000000.pkl
body_model str mhr
frames list 778
kids list 1
metadata dict keys ['source', 'camera_id', 'depth_root', 'depth_source', 'alignment', 'alignment_source_sampling', 'mean_depth_align_translation_m', 'mean_depth_align_scale', 'fallback_count', 'sam3d_direct_cache', 'sam3d_human_prompt', 'mhr_vertex_storage', 'mhr_init_root_revision', 'mhr_init_root_convention', 'mhr_init_translation_revision', 'mhr_init_translation_convention']
mhr_global_rot6d (778, 6) float32
mhr_trans (778, 3) float32
mhr_body_pose_cont (778, 260) float32
mhr_hand (778, 108) float32
mhr_shape (778, 45) float32
mhr_scale (778, 28) float32
mhr_face (778, 72) float32
mhr_joints (778, 127, 3) float32
mhr_keypoints (778, 70, 3) float32

=== /pipeline/foundationpose/episode_000000.pkl
fr …[+1868 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,220p' /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py; sed -n '1,120p' /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/sam_3d_body_estimator.py"
```
<details><summary>output</summary>

```
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import warnings
from typing import Optional

import roma
import torch
import torch.nn as nn

from ..modules import rot6d_to_rotmat
from ..modules.mhr_utils import (
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    compact_model_params_to_cont_body,
    mhr_param_hand_mask,
)

from ..modules.transformer import FFN

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR

        MOMENTUM_ENABLED = True
        warnings.warn("Momentum is enabled")
    else:
        warnings.warn("Momentum is not enabled")
        raise ImportError
except:
    MOMENTUM_ENABLED = False
    warnings.warn("Momentum is not enabled")


class MHRHead(nn …[+10895 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz'); h=np.load('/work/inputs/frame0_reference_human.npz')
for name,arr in [('true',h['vertices_world']),('recon',r['human_vertices_world'][0]),('kps',r['human_keypoints_world'][0])]:
 print(name,'min',arr.min(0),'max',arr.max(0),'mean',arr.mean(0),'centroid',arr.mean(0))
T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
print('cam center',T[:3,3]); print('cam R',T[:3,:3])
# compare recon reconstruct every fourth to true world?
for off in range(4):
 d=np.linalg.norm(r['human_vertices_world'][0]-h['vertices_world'][off::4],axis=1)
 print('offset',off,'mean',d.mean(),'median',np.median(d),'max',d.max())
print('recon verts to true nearest via cKD not yet')
PY"
```
<details><summary>output</summary>

```
true min [-0.86104584 -0.39850104  3.0353856 ] max [-0.31906325  1.2420924   3.416338  ] mean [-0.60657394  0.35195863  3.1561983 ] centroid [-0.60657394  0.35195863  3.1561983 ]
recon min [-0.8599326  -0.40705746  3.0181105 ] max [-0.3237478  1.246326   3.4133868] mean [-0.6031205   0.32215515  3.157242  ] centroid [-0.6031205   0.32215515  3.157242  ]
kps min [-0.85496557 -0.28676492  3.0252788 ] max [-0.33796734  1.2350252   3.3561544 ] mean [-0.60691524  0.47688854  3.152224  ] centroid [-0.60691524  0.47688854  3.152224  ]
cam center [0.05715222 0.00344673 0.0349674 ]
cam R [[ 0.99940567 -0.00630215  0.03389078]
 [ 0.00506315  0.99932006  0.03652095]
 [-0.03409789 -0.03632765  0.99875805]]
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
ValueError: operands co …[+61 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
p='/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt'
z=torch.load(p,map_location='cpu',weights_only=False)
print(type(z))
if isinstance(z,dict):
 print(z.keys())
 for k,v in z.items():
  if isinstance(v,dict):
   print(k,len(v),list(v)[:30])
   for kk,vv in v.items():
    if 'face' in kk or 'mesh' in kk or 'topolog' in kk:
     try: print(' ',kk, getattr(vv,'shape',None),getattr(vv,'dtype',None))
     except: pass
  elif hasattr(v,'shape'): print(k,v.shape,v.dtype)
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
dict_keys(['backbone.encoder.cls_token', 'backbone.encoder.storage_tokens', 'backbone.encoder.patch_embed.proj.weight', 'backbone.encoder.patch_embed.proj.bias', 'backbone.encoder.rope_embed.periods', 'backbone.encoder.blocks.0.norm1.weight', 'backbone.encoder.blocks.0.norm1.bias', 'backbone.encoder.blocks.0.attn.qkv.weight', 'backbone.encoder.blocks.0.attn.qkv.bias', 'backbone.encoder.blocks.0.attn.qkv.bias_mask', 'backbone.encoder.blocks.0.attn.proj.weight', 'backbone.encoder.blocks.0.attn.proj.bias', 'backbone.encoder.blocks.0.ls1.gamma', 'backbone.encoder.blocks.0.norm2.weight', 'backbone.encoder.blocks.0.norm2.bias', 'backbone.encoder.blocks.0.mlp.w1.weight', 'backbone.encoder.blocks.0.mlp.w1.bias', 'backbone.encoder.blocks.0.mlp.w2.weight', 'backbone.encoder.blocks.0.m …[+129330 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, cv2, h5py
from scipy.spatial import cKDTree
r=np.load('/work/inputs/reconstruction.npz'); h=np.load('/work/inputs/frame0_reference_human.npz')
a=h['vertices_world']; b=r['human_vertices_world'][0]
# nearest distances from recon to true and true to recon
for x,y,label in [(a,b,'true->recon'),(b,a,'recon->true')]:
 d,i=cKDTree(y).query(x)
 print(label,'mean',d.mean(),'median',np.median(d),'p90',np.percentile(d,90),'max',d.max())
# find exact correspondences by nearest (true likely identical topology)
d,i=cKDTree(a).query(b)
print('recon indices in true first20',i[:20], 'uniques',len(np.unique(i)))
print('nearest d quantiles',np.quantile(d,[0,.1,.5,.9,.99,1]))
PY"
```
<details><summary>output</summary>

```
true->recon mean 0.011095499082482098 median 0.009615990793210887 p90 0.020185137212416323 max 0.06371879328536605
recon->true mean 0.008981635558432203 median 0.008474256206424162 p90 0.016003741086851347 max 0.0296854126221724
recon indices in true first20 [13940 13894  5657  5340  5347 13980  5823  5421    97  5340   489  5434
   671 14056    97  5340 13941 14191  5421  5423] uniques 3795
nearest d quantiles [0.00020318 0.00264204 0.00847426 0.01600374 0.02127358 0.02968541]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 for k in f:
  m=f[k][0]
  ys,xs=np.nonzero(m)
  print(k,'area',len(xs),'bbox',xs.min(),ys.min(),xs.max(),ys.max())
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 print('aligned keys',list(f.keys()),list(f['aligned'].keys()))
 for k in f['aligned']:
  ds=f['aligned'][k]
  print(k,ds.shape,ds.dtype,ds[0].shape if hasattr(ds[0],'shape') else ds[0], ds[0].dtype if hasattr(ds[0],'dtype') else '')
  if k.startswith('front') and ds[0].ndim==2:
   d=ds[0]; print(' depth stats',d.min(),d.max(),np.median(d))
PY"
```
<details><summary>output</summary>

```
object area 22185 bbox 559 728 835 931
person area 43398 bbox 484 410 635 865
aligned keys ['aligned', 'alignment', 'frame_names', 'raw'] ['front_stereo_camera_left']
front_stereo_camera_left (778,) object (659465,) uint8

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, pprint, torch
z=pickle.load(open('/pipeline/mhr_init/episode_000000.pkl','rb'))
pprint.pp(z['metadata'])
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False)
for k in sorted(w):
 if 'faces' in k or 'keypoint_mapping' in k or 'root_coords' in k or 'hand_joint' in k:
  print(k, w[k].shape, w[k].dtype, w[k].flatten()[:10].tolist())
PY"
```
<details><summary>output</summary>

```
{'source': 'sam3d_body_rgb',
 'camera_id': 0,
 'depth_root': '/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/export/episode_000000',
 'depth_source': 'metric_depth',
 'alignment': 'none',
 'alignment_source_sampling': 'none',
 'mean_depth_align_translation_m': 0.0,
 'mean_depth_align_scale': 1.0,
 'fallback_count': 0,
 'sam3d_direct_cache': '/l/users/han.zhou/v2d/outputs/cari4d/episode_000000/mhr_init/episode_000000.sam3d.h5',
 'sam3d_human_prompt': {'mode': 'sam2_bbox_mask',
                        'bbox_source': 'effective_human_mask_nonzero_support',
                        'mask_source': 'effective_human_mask',
                        'mask_score': 1.0},
 'mhr_vertex_storage': 'parameters_only',
 'mhr_init_root_revision': 'sam3d_camera_axis_world_root_v1',
 'mhr_init_root_conventi …[+1288 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 def walk(name,obj):
  if isinstance(obj,h5py.Dataset):
   print(name,obj.shape,obj.dtype,'attrs',dict(obj.attrs))
  else: print(name,'GROUP attrs',dict(obj.attrs))
 f.visititems(walk)
 print('aligned sample bytes',f['aligned/front_stereo_camera_left'][0][:128])
 print('raw sample bytes',f['raw/front_stereo_camera_left'][0][:128])
PY"
```
<details><summary>output</summary>

```
aligned GROUP attrs {}
aligned/front_stereo_camera_left (778,) object attrs {}
alignment GROUP attrs {}
alignment/front_stereo_camera_left GROUP attrs {}
alignment/front_stereo_camera_left/scale (778,) float32 attrs {}
alignment/front_stereo_camera_left/shift (778,) float32 attrs {}
alignment/front_stereo_camera_left/valid_count (778,) int32 attrs {}
frame_names GROUP attrs {}
frame_names/front_stereo_camera_left (778,) object attrs {}
raw GROUP attrs {}
raw/front_stereo_camera_left (778,) object attrs {}
aligned sample bytes [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0   0   0   4 128  16   0   0   0   0  76  58   5   5   0   1   0
   0  73  68  65  84 120 218 172 253 217 150  37  59 178  36 136 137 194
 108 251 236  49 159 204 188  83  87 247 170  98  …[+791 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned.h5\\|front_stereo_camera_left\\|raw/front\" -n /code/reconstruction/modules/v2d_cari4d/lib /workspace/v2d_cari4d/lib | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr_export_utils.py:23:    "front_stereo_camera_left",
/code/reconstruction/modules/v2d_cari4d/lib/run_inference.py:116:    aligned_depth = output_root / "depth/moge2/aligned.h5"
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_export_utils.py:23:    "front_stereo_camera_left",
/workspace/v2d_cari4d/lib/run_inference.py:116:    aligned_depth = output_root / "depth/moge2/aligned.h5"

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False)
f=w['head_pose.faces'].numpy(); print(f.shape, f.min(),f.max(),len(np.unique(f)))
# subset faces for 18056 and 4610 indexing maybe determine which vertex scheme
for n in [18056,18439,18566,4610]:
 fs=f[(f<n).all(1)]
 print(n,'valid faces',len(fs),'verts used',len(np.unique(fs)))
PY"
```
<details><summary>output</summary>

```
(36874, 3) 0 18438 18439
18056 valid faces 36067 verts used 18056
18439 valid faces 36874 verts used 18439
18566 valid faces 36874 verts used 18439
4610 valid faces 7896 verts used 4450

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for grp in ['aligned','raw']:
  b=bytes(f[f'{grp}/front_stereo_camera_left'][0]); im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
  print(grp,im.shape,im.dtype,im.min(),im.max(),np.median(im),np.quantile(im,[.01,.1,.5,.9,.99]))
PY"
```
<details><summary>output</summary>

```
aligned (1152, 1536) uint16 1765 9847 4023.0 [1846. 2193. 4023. 6892. 9173.]
raw (1152, 1536) uint16 1579 8808 3599.0 [1652. 1962. 3599. 6165. 8205.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json,numpy as np
z=np.load('/work/inputs/reconstruction.npz'); T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);
Tw=z['T_world_mesh']; Tc=z['T_cam_mesh_coconet']; Tf=z['T_cam_mesh_foundationpose']
# verify baseline T relation
err=np.linalg.inv(T)@Tw
print('world relation coconet vs T"'^-1Tw max/mean'"',np.abs(err-Tc).max(),np.abs(err-Tc).mean())
print('found vs coco translation diff',np.linalg.norm(Tf[:,:3,3]-Tc[:,:3,3],axis=1).mean())
# object positions stats world/cam
for name,A in [('world',Tw),('coco',Tc),('found',Tf)]: print(name,'t mean',A[:,:3,3].mean(0),'range',A[:,:3,3].min(0),A[:,:3,3].max(0))
PY"
```
<details><summary>output</summary>

```
world relation coconet vs T^-1Tw max/mean 0.12765180540045584 0.015441435701046436
found vs coco translation diff 0.093222074
world t mean [0.09166646 0.81774855 2.9116828 ] range [-0.14188658  0.6256984   2.6934566 ] [0.46059287 1.0129839  3.243944  ]
coco t mean [-0.09287319  0.6407676   2.840834  ] range [-0.31004956  0.4605027   2.624422  ] [0.25168097 0.83997804 3.1650047 ]
found t mean [-0.10120757  0.66173714  2.912372  ] range [-0.33340168  0.47015873  2.5340974 ] [0.26056296 0.89091355 3.3316243 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/vis /work/outputs && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"select='eq(n,0)+eq(n,100)+eq(n,300)+eq(n,500)+eq(n,700)+eq(n,777)'\" -vsync 0 -q:v 2 /work/scratch/vis/f%06d.jpg && python - <<'PY'
import cv2,glob,numpy as np, os
fs=sorted(glob.glob('/work/scratch/vis/f*.jpg'))
ims=[]
for i,p in enumerate(fs):
 im=cv2.imread(p); im=cv2.resize(im,(384,288)); cv2.putText(im,os.path.basename(p), (10,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2); ims.append(im)
cv2.imwrite('/work/scratch/vis/montage.jpg',np.vstack(ims))
print(fs)
PY"
```
<details><summary>output</summary>

```
['/work/scratch/vis/f000001.jpg', '/work/scratch/vis/f000002.jpg', '/work/scratch/vis/f000003.jpg', '/work/scratch/vis/f000004.jpg', '/work/scratch/vis/f000005.jpg', '/work/scratch/vis/f000006.jpg']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
c=json.load(open('/work/inputs/camera.json')); T=np.array(c['T_world_cam']); n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); nw=n[:3]; dw=n[3]
nc=T[:3,:3].T@nw; dc=dw+nw@T[:3,3]
print('baseline n_cam',nc,'d_cam',dc,'norm',np.linalg.norm(nc))
# camera height above plane
print('cam signed distance',nw@T[:3,3]+dw)
# points expected: lower image rays intersect plane
K=np.array(c['K']); H=c['height']; W=c['width']
for y in [500,600,700,800,900,1000,1100]:
 ray=np.array([(W/2-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],1])
 z=-dc/(nc@ray); print(y,z)
PY"
```
<details><summary>output</summary>

```
baseline n_cam [-0.02204671 -0.99546902 -0.09249523] d_cam 1.3791749153771582 norm 1.0
cam signed distance 1.3791749153771582
500 71.4042402679343
600 9.80127136535444
700 5.261762425266648
800 3.5961754027075297
900 2.73152390407757
1000 2.202066857336732
1100 1.8445364600129521

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analyze_floor.py <<'PY'
import h5py,cv2,numpy as np,json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); T=np.array(c['T_world_cam'])
nwp=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); nw=nwp[:3]; dw=nwp[3]
nc=T[:3,:3].T@nw; dc=nw@T[:3,3]+dw
print('baseline ncam,d',nc,dc)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f, h5py.File('/work/inputs/masks.h5','r') as fm:
  for fi in [0,100,300,500,700,777]:
    b=bytes(f['aligned/front_stereo_camera_left'][fi]); D=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
    pm=fm['person'][fi]; om=fm['object'][fi]
    vv,uu=np.mgrid[:1152,:1536]
    # lower region, exclude masks and perhaps very near ceiling? 
    sel=(vv>=620)&(vv<1140)&~pm&~om
    # sample
    sel &= ((uu%3)==0)&((vv%3)==0)
    x=(uu[sel]-K[0,2])/K[0,0]*D[sel]; y=(vv[sel]-K[1,2])/K[1,1]*D[sel]; z=D[sel]
    P=np.stack([x,y,z],1)
    dist=P@nc+dc
    print('\\nframe',fi,'npts',len(P),'dist q',np.quantile(dist,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
    # expected inliers abs<.08, iterative plane TLS via SVD
    inl=np.abs(dist)<.10
    for it in range(5):
      Q=P[inl]
      cen=Q.mean(0)
      _,_,vh=np.linalg.svd(Q-cen,full_matrices=False)
      n=vh[-1]; n*=np.sign(n@nc)
      d=-n@cen
      dist=P@n+d
      new=np.abs(dist)<max(.015,.04/(it+1))
      # onl …[+392 chars]
```
<details><summary>output</summary>

```
baseline ncam,d [-0.02204671 -0.99546902 -0.09249523] 1.3791749153771582

frame 0 npts 83696 dist q [-0.1200695  -0.09567302 -0.09459255 -0.09376966 -0.09124337 -0.08612907
 -0.07020547  0.09763386  0.36356482  0.6183112   0.90829162]
 it 0 inl 75351 n [-0.01818033 -0.99452769 -0.10287933] d 1.4900168559015317 rms 0.021363763291435783 new 72119
 it 1 inl 72119 n [-0.01751178 -0.99490315 -0.0993029 ] d 1.482036679025183 rms 0.007665755496122163 new 70572
 it 2 inl 70572 n [-0.01730954 -0.99496163 -0.09875088] d 1.4807269480818461 rms 0.006077755500392778 new 68602
 it 3 inl 68602 n [-0.01731471 -0.99495779 -0.09878868] d 1.4810730561242949 rms 0.005437514631146968 new 68514
 it 4 inl 68514 n [-0.01732225 -0.99495957 -0.09876943] d 1.4810471199776831 rms 0.005413745429297293 new 68501
 angle …[+3950 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for k in ['scale','shift','valid_count']:
  a=f['alignment/front_stereo_camera_left/'+k][:]
  print(k,a.shape,a.dtype,'quantiles',np.quantile(a,[0,.01,.1,.5,.9,.99,1]),a[:10])
PY"
```
<details><summary>output</summary>

```
scale (778,) float32 quantiles [1.03259504 1.047042   1.07848488 1.10514152 1.14877378 1.18097675
 1.18968606] [1.118034  1.107011  1.1068578 1.0982063 1.1228015 1.1290153 1.1250422
 1.1209191 1.1296592 1.1310667]
shift (778,) float32 quantiles [0. 0. 0. 0. 0. 0. 0.] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
valid_count (778,) int32 quantiles [17050.   17247.85 20764.4  38176.   49409.4  50435.23 50566.  ] [35699 36769 36311 37366 36455 36700 37081 37346 37201 37376]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_human.py <<'PY'
import numpy as np, cv2, json, h5py, torch, time
from scipy.ndimage import distance_transform_edt
W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
V=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False)
F=w['head_pose.faces'].numpy(); F=F[(F<V.shape[0]).all(1)]
print('V,F',V.shape,F.shape)
with h5py.File('/work/inputs/masks.h5','r') as f: mask=f['person'][0].astype(np.uint8)

def render(T,scale=0.5):
 W2,H2=int(W*scale),int(H*scale)
 Xc=(np.linalg.inv(T)@np.c_[V,np.ones(len(V))].T).T[:,:3]
 uv=(K@Xc.T).T; uv=uv[:,:2]/uv[:,2:3]*scale
 valid=Xc[:,2]>0
 q=np.clip(uv, -1e5,1e5).astype(np.int32)
 # triangles all valid; fillpoly expects list maybe
 out=np.zeros((H2,W2),np.uint8)
 # split valid triangle groups where all verts front
 fv=valid[F].all(1); ff=F[fv]
 print('faces visible',len(ff))
 cv2.fillPoly(out,[q[ff].reshape(-1,2)],1)
 return out.astype(bool),uv*0+1
ms=cv2.resize(mask,(int(W*scale),int(H*scale)),interpolation=cv2.INTER_NEAREST).astype(bool)
for name,T in [('base',Twc)]:
 t=time.time(); r,_=render(T); print('render time',time.time()-t)
 inter=(r&ms).sum(); union=(r|ms).sum(); iou=inter/union
 # boundary distances
 e1=cv2.Canny(r.astype(np.uint8)*255,0,1)>0; e2=cv2.Canny(ms.astype(np.uint8)*255,0,1)>0
 d1=distance_tra …[+449 chars]
```
<details><summary>output</summary>

```
V,F (18056, 3) (36067, 3)
Traceback (most recent call last):
  File "/work/scratch/render_human.py", line 24, in <module>
    ms=cv2.resize(mask,(int(W*scale),int(H*scale)),interpolation=cv2.INTER_NEAREST).astype(bool)
                              ^^^^^
NameError: name 'scale' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_human.py'
s=open(p).read().replace(\"ms=cv2.resize(mask,(int(W*scale),int(H*scale))\", \"scale=0.5\\nms=cv2.resize(mask,(int(W*scale),int(H*scale))\")
open(p,'w').write(s)
PY
python /work/scratch/render_human.py"
```
<details><summary>output</summary>

```
V,F (18056, 3) (36067, 3)
faces visible 36067
render time 40.07903242111206
base iou 0.7961529056972858 areas 11188 10849 bd render->mask [ 6.         13.41640786 14.14213562 18.43908891] bd mask->render [ 7.21110255 14.         15.23154621 20.        ]
[ WARN:0@43.625] global loadsave.cpp:848 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/test_nvdiff.py <<'PY'
import numpy as np, torch, time, json, h5py
import nvdiffrast.torch as dr
from scipy.ndimage import distance_transform_edt
W,H=1536,1152; scale=.5; W2,H2=int(W*scale),int(H*scale)
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K'],np.float32); Twc=np.array(c['T_world_cam'],np.float32)
V=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False)
F=w['head_pose.faces'].numpy(); F=F[(F<V.shape[0]).all(1)]
# print dr contexts
for typ in ['cuda','gl']:
 try:
  ctx=dr.RasterizeCudaContext() if typ=='cuda' else dr.RasterizeGLContext()
  print('ctx ok',typ,ctx)
  break
 except Exception as e: print('ctx fail',typ,repr(e))
Vt=torch.tensor(V,device='cuda',dtype=torch.float32)
Ft=torch.tensor(F,device='cuda',dtype=torch.int32)
T=torch.tensor(Twc,device='cuda',dtype=torch.float32)
Xc=(Vt@T[:3,:3]+T[:3,3]) # wait Xcam=R"'^T(Xw-t); this is wrong
R=T[:3,:3].T; tt=-R@T[:3,3]
Xc=Vt@R.T+tt
px=torch.cat([(K[0,0]*Xc[:,:1]+K[0,2]*Xc[:,2:3])*scale, (K[1,1]*Xc[:,1:2]+K[1,2]*Xc[:,2:3])*scale],1)/Xc[:,2:3]
pos=torch.cat([(px[:,0:1]/(W2/2)-1), (1-px[:,1:2]/(H2/2)), torch.zeros_like(px[:,:1]), torch.ones_like(px[:,:1])],1)[None]
for i in range(3):
 t=time.time();
 with torch.no_grad():
  rast,_=dr.rasterize(ctx,pos,Ft,resolution=[H2,W2])
 print('"'raster',i,time.time()-t,rast.shape)
r=(rast[0,:,:,3]>0).cpu …[+594 chars]
```
<details><summary>output</summary>

```
ctx ok cuda <nvdiffrast.torch.ops.RasterizeCudaContext object at 0x7d5c7a04a710>
raster 0 0.005370140075683594 torch.Size([1, 576, 768, 4])
raster 1 0.0029380321502685547 torch.Size([1, 576, 768, 4])
raster 2 0.0014083385467529297 torch.Size([1, 576, 768, 4])
iouTraceback (most recent call last):
  File "/work/scratch/test_nvdiff.py", line 33, in <module>
 0.4788703501094092 areas 10778 10849
    e1=cv2.Canny(r.astype(np.uint8)*255,0,1)>0; e2=cv2.Canny(ms.astype(np.uint8)*255,0,1)>0
       ^^^
NameError: name 'cv2' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_nvdiff.py'; s=open(p).read().replace('import nvdiffrast.torch as dr','import nvdiffrast.torch as dr\\nimport cv2'); s=s.replace(\"inter=(r&ms).sum(); union=(r|ms).sum(); print('iou',inter/union,'areas',r.sum(),ms.sum())\", \"inter=(r&ms).sum(); union=(r|ms).sum(); print('iou',inter/union,'areas',r.sum(),ms.sum());\\nfor nm,a in [('r',r),('m',ms)]:\\n y,x=np.nonzero(a); print(nm,'bbox',x.min(),y.min(),x.max(),y.max(),'cent',x.mean(),y.mean())\\ncv2.imwrite('/work/scratch/vis/nv_r.png',(r*255).astype('uint8')); cv2.imwrite('/work/scratch/vis/nv_m.png',(ms*255).astype('uint8')); cv2.imwrite('/work/scratch/vis/nv_ov.png',np.stack([ms*255,r*255,np.zeros_like(r)],2))\")
open(p,'w').write(s)
PY
python /work/scratch/test_nvdiff.py"
```
<details><summary>output</summary>

```
ctx ok cuda <nvdiffrast.torch.ops.RasterizeCudaContext object at 0x7660817949d0>
raster 0 0.0037784576416015625 torch.Size([1, 576, 768, 4])
raster 1 0.0017032623291015625 torch.Size([1, 576, 768, 4])
raster 2 0.0013704299926757812 torch.Size([1, 576, 768, 4])
iou 0.4788703501094092 areas 10778 10849
r bbox 245 149 315 363 cent 278.75988123956205 254.3316014102802
m bbox 242 205 317 432 cent 277.61471103327494 314.096875288045
[ WARN:0@2.414] global loadsave.cpp:848 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
bd [ 22.49404839  98.03264584 110.00272705 123.36936411] [ 30.         119.93329626 134.41352231 147.58048651]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_nvdiff.py'; s=open(p).read(); s=s.replace(\"R=T[:3,:3].T; tt=-R@T[:3,3]\\nXc=Vt@R.T+tt\", \"Rc=T[:3,:3].T; tc=-Rc@T[:3,3]\\nXc=Vt@Rc.T+tc\")
open(p,'w').write(s)
PY
python /work/scratch/test_nvdiff.py"
```
<details><summary>output</summary>

```
ctx ok cuda <nvdiffrast.torch.ops.RasterizeCudaContext object at 0x7d09a0977250>
raster 0 0.003376007080078125 torch.Size([1, 576, 768, 4])
raster 1 0.001481771469116211 torch.Size([1, 576, 768, 4])
raster 2 0.001317739486694336 torch.Size([1, 576, 768, 4])
iou 0.4788703501094092 areas 10778 10849
r bbox 245 149 315 363 cent 278.75988123956205 254.3316014102802
m bbox 242 205 317 432 cent 277.61471103327494 314.096875288045
[ WARN:0@3.237] global loadsave.cpp:848 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
bd [ 22.49404839  98.03264584 110.00272705 123.36936411] [ 30.         119.93329626 134.41352231 147.58048651]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'Xc\\|Rc\\|R=' /work/scratch/test_nvdiff.py"
```
<details><summary>output</summary>

```
20:Xc=(Vt@T[:3,:3]+T[:3,3]) # wait Xcam=R^T(Xw-t); this is wrong
21:Rc=T[:3,:3].T; tc=-Rc@T[:3,3]
22:Xc=Vt@Rc.T+tc
23:px=torch.cat([(K[0,0]*Xc[:,:1]+K[0,2]*Xc[:,2:3])*scale, (K[1,1]*Xc[:,1:2]+K[1,2]*Xc[:,2:3])*scale],1)/Xc[:,2:3]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_nvdiff.py'; s=open(p).read(); s=s.replace(\"Xc=(Vt@T[:3,:3]+T[:3,3]) # wait Xcam=R"'^T(Xw-t); this is wrong'"\\nRc=T[:3,:3].T; tc=-Rc@T[:3,3]\\nXc=Vt@Rc.T+tc\", \"tc=-T[:3,:3].T@T[:3,3]\\nXc=Vt@T[:3,:3]+tc\")
open(p,'w').write(s)
PY
python /work/scratch/test_nvdiff.py"
```
<details><summary>output</summary>

```
ctx ok cuda <nvdiffrast.torch.ops.RasterizeCudaContext object at 0x76be88f89d50>
raster 0 0.004393577575683594 torch.Size([1, 576, 768, 4])
raster 1 0.0030944347381591797 torch.Size([1, 576, 768, 4])
raster 2 0.0018012523651123047 torch.Size([1, 576, 768, 4])
iou 0.4788703501094092 areas 10778 10849
r bbox 245 149 315 363 cent 278.75988123956205 254.3316014102802
m bbox 242 205 317 432 cent 277.61471103327494 314.096875288045
[ WARN:0@2.421] global loadsave.cpp:848 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
bd [ 22.49404839  98.03264584 110.00272705 123.36936411] [ 30.         119.93329626 134.41352231 147.58048651]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '17,24p' /work/scratch/test_nvdiff.py; python - <<'PY'
import numpy as np,json
V=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; c=json.load(open('/work/inputs/camera.json')); T=np.array(c['T_world_cam']); K=np.array(c['K']);
# compare points manually
for label,Xc in [('right',V@T[:3,:3].T+T[:3,3]),('wrong',V@T[:3,:3]+(-T[:3,:3].T@T[:3,3]))]:
 uv=(K@Xc.T).T; uv=uv[:,:2]/uv[:,2:3]; print(label,uv.min(0),uv.max(0),uv.mean(0),Xc[:,2].min(),Xc[:,2].max())
PY"
```
<details><summary>output</summary>

```
right [579.52146474 487.94411308] [719.23289225 911.32483301] [644.37139881 680.03090404] 3.0612744144442967 3.4559483478641635
wrong [490.50003354 423.73712439] [632.3193181  854.24767356] [555.66001886 621.69441361] 2.96551549059779 3.3673351857188276

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_nvdiff.py'; s=open(p).read(); s=s.replace(\"(1-px[:,1:2]/(H2/2))\", \"(px[:,1:2]/(H2/2)-1)\")
open(p,'w').write(s)
PY
python /work/scratch/test_nvdiff.py"
```
<details><summary>output</summary>

```
ctx ok cuda <nvdiffrast.torch.ops.RasterizeCudaContext object at 0x750389bab1d0>
raster 0 0.004355669021606445 torch.Size([1, 576, 768, 4])
raster 1 0.0019974708557128906 torch.Size([1, 576, 768, 4])
raster 2 0.003251314163208008 torch.Size([1, 576, 768, 4])
iou 0.7941762070681931 areas 10778 10849
r bbox 245 212 315 426 cent 278.75988123956205 320.6683985897198
m bbox 242 205 317 432 cent 277.61471103327494 314.096875288045
[ WARN:0@3.134] global loadsave.cpp:848 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
bd [ 6.         12.68053821 14.14213562 16.97056275] [ 8.         15.23154621 16.1245155  34.525353  ]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/obj_floor.py <<'PY'
import trimesh,numpy as np,json,time
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(mesh),mesh.vertices.shape,getattr(mesh,'faces',None).shape, mesh.bounds,mesh.extents)
V=np.c_[mesh.vertices,np.ones(len(mesh.vertices))]
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
# transform vertices each frame, min/percentiles of signed plane distance
mins=[]; q10=[]
for i in range(len(P)):
 W=(P[i]@V.T).T[:,:3]; sd=W@n+d; mins.append(sd.min()); q10.append(np.quantile(sd,.01))
mins=np.array(mins); q10=np.array(q10)
print('min distance quantiles',np.quantile(mins,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
print('per 20',[(i,round(float(mins[i]),4)) for i in range(0,778,20)])
print('min center of contact xyz frame0', (P[0]@V.T).T[np.argmin((P[0]@V.T).T[:,:3]@n+d),:3])
# object lowest signed > maybe geometry includes detail.
PY
python /work/scratch/obj_floor.py"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> (51247, 3) (95759, 3) [[-0.20217043 -0.23005383 -0.35287058]
 [ 0.26516873  0.22619514  0.53656369]] [0.46733916 0.45624897 0.88943428]
Traceback (most recent call last):
  File "/work/scratch/obj_floor.py", line 6, in <module>
    n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
    ^^^
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/obj_floor.py'; s=open(p).read().replace(\"n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])\", \"n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])\\nn=n[:3]\")
open(p,'w').write(s)
PY
python /work/scratch/obj_floor.py"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> (51247, 3) (95759, 3) [[-0.20217043 -0.23005383 -0.35287058]
 [ 0.26516873  0.22619514  0.53656369]] [0.46733916 0.45624897 0.88943428]
Traceback (most recent call last):
  File "/work/scratch/obj_floor.py", line 6, in <module>
    n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
    ^^^
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,12p' /work/scratch/obj_floor.py"
```
<details><summary>output</summary>

```
import trimesh,numpy as np,json,time
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(mesh),mesh.vertices.shape,getattr(mesh,'faces',None).shape, mesh.bounds,mesh.extents)
V=np.c_[mesh.vertices,np.ones(len(mesh.vertices))]
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
n,d=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
n=n[:3]
# transform vertices each frame, min/percentiles of signed plane distance
mins=[]; q10=[]
for i in range(len(P)):
 W=(P[i]@V.T).T[:,:3]; sd=W@n+d; mins.append(sd.min()); q10.append(np.quantile(sd,.01))
mins=np.array(mins); q10=np.array(q10)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/n,d=np.array(json.load(open('\\/work\\/inputs\\/ground_plane.json'))\\['plane'\\])/plane=np.array(json.load(open('\\/work\\/inputs\\/ground_plane.json'))['plane']); n,d=plane[:3],plane[3]/\" /work/scratch/obj_floor.py
python /work/scratch/obj_floor.py"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> (51247, 3) (95759, 3) [[-0.20217043 -0.23005383 -0.35287058]
 [ 0.26516873  0.22619514  0.53656369]] [0.46733916 0.45624897 0.88943428]
min distance quantiles [-0.06034286 -0.05428126 -0.0479238  -0.03964337 -0.03167432 -0.02491543
 -0.01741308 -0.00960595 -0.00469239 -0.000378    0.00329463]
per 20 [(0, -0.0269), (20, -0.0218), (40, -0.0267), (60, -0.0331), (80, -0.0148), (100, -0.0396), (120, -0.0432), (140, -0.0571), (160, -0.0521), (180, -0.045), (200, -0.0221), (220, -0.023), (240, -0.0225), (260, -0.0206), (280, -0.0214), (300, -0.0349), (320, -0.0048), (340, -0.0134), (360, -0.0289), (380, -0.0333), (400, -0.0173), (420, -0.0285), (440, -0.0211), (460, -0.0285), (480, -0.0203), (500, -0.0113), (520, -0.0421), (540, -0.0341), (560, -0.0073), (580, -0.00 …[+216 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); V=z['human_vertices_world']; P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n,d=P[:3],P[3]
sd=V@n+d
mins=sd.min(1); qs=np.quantile(sd,[.001,.005,.01,.02],axis=1)
print('min q',np.quantile(mins,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
print('q1% mean',qs.mean(1), 'median',np.median(qs,1))
print('frame0 min q',mins[0],qs[:,0], 'frame last',mins[-1],qs[:,-1])
print('min quantiles first100',np.quantile(mins[:100],[0,.25,.5,.75,1]))
PY"
```
<details><summary>output</summary>

```
min q [-0.0655393  -0.0579901  -0.0482469  -0.03986846 -0.02888088 -0.01941598
 -0.0125275  -0.00547059 -0.00273251  0.00327063  0.01287927]
q1% mean [-0.02032548 -0.01788646 -0.01563937 -0.01103076] median [-0.01809451 -0.01607271 -0.01394212 -0.00972919]
frame0 min q -0.026168326205187276 [-0.02560851 -0.02395156 -0.02167214 -0.01546897] frame last -0.012784254389781147 [-0.01251106 -0.01196879 -0.01033826 -0.00616281]
min quantiles first100 [-0.05975848 -0.03645365 -0.03178713 -0.02751175 -0.01914938]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
for stride in [4]:
 a=r[:len(h)//stride]; b=h[::stride][:len(a)]
 d=np.linalg.norm(a-b,axis=1)
 print(len(a),len(b),d.mean(),np.median(d),np.quantile(d,[.9,.99,1]))
 print('first',d[:20])
PY"
```
<details><summary>output</summary>

```
4514 4514 0.5743668 0.6015191 [0.89221399 1.51621644 1.62391484]
first [0.13760136 0.14495431 0.13055782 0.12740903 0.14218275 0.12915443
 0.14765744 0.11169463 0.13274263 0.11323095 0.11873046 0.13388734
 0.14780262 0.14365664 0.18189335 0.22791676 0.17897584 0.5355339
 0.43300465 0.4180836 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/floor_aggregate.py <<'PY'
import h5py,cv2,numpy as np,json
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); T=np.array(c['T_world_cam']); nwp=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); nw=nwp[:3]; dw=nwp[3]
nc0=T[:3,:3].T@nw; dc0=nw@T[:3,3]+dw
frames=np.arange(0,778,20)
allP=[]; per=[]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f, h5py.File('/work/inputs/masks.h5','r') as fm:
 for fi in frames:
  b=bytes(f['aligned/front_stereo_camera_left'][fi]); D=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
  pm=fm['person'][fi]; om=fm['object'][fi]
  # Use lower image and coarse stride. Exclude person/object.
  vv,uu=np.mgrid[600:1152:5,0:1536:5]; Ds=D[600:1152:5,0:1536:5]; pms=pm[600:1152:5,0:1536:5]; oms=om[600:1152:5,0:1536:5]
  sel=(~pms)&(~oms)&np.isfinite(Ds)&(Ds>0)
  u=uu[sel];v=vv[sel];z=Ds[sel]
  P=np.stack([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z],1)
  dist=P@nc0+dc0
  inl=np.abs(dist)<.14
  for it in range(6):
   Q=P[inl]; cen=np.median(Q,0)
   _,_,vh=np.linalg.svd(Q-cen,full_matrices=False); n=vh[-1]; n*=np.sign(n@nc0); d=-n@cen
   dist=P@n+d; new=np.abs(dist)<max(.012,.03/(it+1))
   inl=new
  # Keep points near np median plane
  resid=P[inl]@n+d
  mad=1.4826*np.median(np.abs(resid-np.median(resid)))
  per.append((fi,len(Q),n,d,np.sqrt(np.mean(resid**2)),mad))
  # subsample and append central inliers
  allP.append(P[inl][::2])
per=np.array …[+910 chars]
```
<details><summary>output</summary>

```
per frame
  0 n=23869 n=[-0.01711 -0.99487 -0.09967] d=1.48658 rms=0.0053 mad=0.0051
 20 n=23483 n=[-0.0183  -0.99496 -0.09856] d=1.49127 rms=0.0051 mad=0.0052
 40 n=22466 n=[-0.01998 -0.99483 -0.09953] d=1.49675 rms=0.0052 mad=0.0049
 60 n=23380 n=[-0.01956 -0.99475 -0.10044] d=1.50278 rms=0.0052 mad=0.0047
 80 n=22270 n=[-0.01913 -0.99471 -0.10091] d=1.52890 rms=0.0054 mad=0.0049
100 n=22898 n=[-0.01849 -0.99473 -0.10081] d=1.48740 rms=0.0051 mad=0.0051
120 n=23207 n=[-0.01879 -0.99469 -0.10118] d=1.44735 rms=0.0049 mad=0.0047
140 n=22360 n=[-0.01981 -0.99455 -0.10238] d=1.50391 rms=0.0055 mad=0.0050
160 n=22851 n=[-0.01813 -0.99457 -0.10248] d=1.47520 rms=0.0054 mad=0.0050
180 n=20888 n=[-0.02067 -0.99451 -0.10257] d=1.47086 rms=0.0056 mad=0.0046
200 n=21692 n=[-0.01999 -0.9943  -0.1046 …[+3201 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/opt_camera.py <<'PY'
import os,json,time,h5py,cv2,numpy as np,torch
import nvdiffrast.torch as dr
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
os.environ['PYTHONHASHSEED']='0'; np.random.seed(0); torch.manual_seed(0)
W,H=1536,1152
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K'],np.float32); T0=np.array(c['T_world_cam'],np.float32)
V=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'].astype(np.float32)
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False)
F=w['head_pose.faces'].numpy(); F=F[(F<len(V)).all(1)]
with h5py.File('/work/inputs/masks.h5','r') as f: mask0=f['person'][0].astype(np.uint8)
# plane observations
plane=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); nw=plane[:3]; dw=plane[3]
# from floor aggregate robust per-frame median (computed from depth); bake exploratory only
nobs=np.array([-0.02020254,-0.99477472,-0.09991089]); nobs/=np.linalg.norm(nobs)
dobs=1.4822659279354825
# Note d depends on depth scale; center around baseline distance in joint weighting.
print('T0',T0)

def pose_from_x(x):
    drv=np.asarray(x[:3]); dt=np.asarray(x[3:6]); Rr=Rotation.from_rotvec(drv).as_matrix().astype(np.float32)
    R=Rr@T0[:3,:3]; t=(T0[:3,3]+dt).astype(np.float32)
    return R,t

# renderer class at scale
class  …[+3037 chars]
```
<details><summary>output</summary>

```
T0 [[ 0.9994057  -0.00630215  0.03389078  0.05715222]
 [ 0.00506315  0.9993201   0.03652095  0.00344673]
 [-0.0340979  -0.03632765  0.998758    0.0349674 ]
 [ 0.          0.          0.          1.        ]]
Traceback (most recent call last):
  File "/work/scratch/opt_camera.py", line 54, in <module>
    t=time.time(); print('base batch iou',R.iou(np.zeros(6)),'time',time.time()-t)
                                          ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 49, in iou
    return float(1-self.render(x)[0][0].cpu())
                   ^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 40, in render
    Xc=torch.einsum('bij,vj->bvi',Rt.transpose(1,2),self.V[None]-tt[:,None,:])
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Fi …[+353 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_camera.py'; s=open(p).read().replace(\"Xc=torch.einsum('bij,vj->bvi',Rt.transpose(1,2),self.V[None]-tt[:,None,:])\", \"Xc=torch.bmm((self.V[None]-tt[:,None,:]),Rt)\")
open(p,'w').write(s)
PY
python /work/scratch/opt_camera.py"
```
<details><summary>output</summary>

```
T0 [[ 0.9994057  -0.00630215  0.03389078  0.05715222]
 [ 0.00506315  0.9993201   0.03652095  0.00344673]
 [-0.0340979  -0.03632765  0.998758    0.0349674 ]
 [ 0.          0.          0.          1.        ]]
Traceback (most recent call last):
  File "/work/scratch/opt_camera.py", line 54, in <module>
    t=time.time(); print('base batch iou',R.iou(np.zeros(6)),'time',time.time()-t)
                                          ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 49, in iou
    return float(1-self.render(x)[0][0].cpu())
                   ^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 40, in render
    Xc=torch.bmm((self.V[None]-tt[:,None,:]),Rt)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: expected scalar type Float but found Double

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_camera.py'; s=open(p).read().replace(\"Rt=torch.tensor(Rs,device='cuda'); tt=torch.tensor(ts,device='cuda')\", \"Rt=torch.tensor(Rs,device='cuda',dtype=torch.float32); tt=torch.tensor(ts,device='cuda',dtype=torch.float32)\")
open(p,'w').write(s)
PY
python /work/scratch/opt_camera.py"
```
<details><summary>output</summary>

```
T0 [[ 0.9994057  -0.00630215  0.03389078  0.05715222]
 [ 0.00506315  0.9993201   0.03652095  0.00344673]
 [-0.0340979  -0.03632765  0.998758    0.0349674 ]
 [ 0.          0.          0.          1.        ]]
Traceback (most recent call last):
  File "/work/scratch/opt_camera.py", line 54, in <module>
    t=time.time(); print('base batch iou',R.iou(np.zeros(6)),'time',time.time()-t)
                                          ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 49, in iou
    return float(1-self.render(x)[0][0].cpu())
                   ^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 44, in render
    rast,_=dr.rasterize(self.ctx,pos,self.F[None].expand(B,-1,-1),resolution=[self.H2,self.W2])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+867 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_camera.py'; s=open(p).read().replace(\"self.F[None].expand(B,-1,-1)\", \"self.F[None].repeat(B,1,1)\")
open(p,'w').write(s)
PY
python /work/scratch/opt_camera.py"
```
<details><summary>output</summary>

```
T0 [[ 0.9994057  -0.00630215  0.03389078  0.05715222]
 [ 0.00506315  0.9993201   0.03652095  0.00344673]
 [-0.0340979  -0.03632765  0.998758    0.0349674 ]
 [ 0.          0.          0.          1.        ]]
Traceback (most recent call last):
  File "/work/scratch/opt_camera.py", line 54, in <module>
    t=time.time(); print('base batch iou',R.iou(np.zeros(6)),'time',time.time()-t)
                                          ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 49, in iou
    return float(1-self.render(x)[0][0].cpu())
                   ^^^^^^^^^^^^^^
  File "/work/scratch/opt_camera.py", line 44, in render
    rast,_=dr.rasterize(self.ctx,pos,self.F[None].repeat(B,1,1),resolution=[self.H2,self.W2])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+863 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_camera.py'; s=open(p).read().replace(\"self.F[None].repeat(B,1,1)\", \"self.F\")
open(p,'w').write(s)
PY
python /work/scratch/opt_camera.py"
```
<details><summary>output</summary>

```
T0 [[ 0.9994057  -0.00630215  0.03389078  0.05715222]
 [ 0.00506315  0.9993201   0.03652095  0.00344673]
 [-0.0340979  -0.03632765  0.998758    0.0349674 ]
 [ 0.          0.          0.          1.        ]]
base batch iou 0.7941762208938599 time 0.3953583240509033
batch72 0.014472723007202148 torch.Size([72]) 1.39536384
start DE half
Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_differentialevolution.py", line 1339, in _calculate_population_energies
    self._mapwrapper(self.func, parameters_pop[0:S])
  File "/opt/conda/lib/python3.11/site-packages/scipy/_lib/_util.py", line 671, in __call__
    return self._mapfunc(func, iterable)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_d …[+1820 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_camera.py'; s=open(p).read().replace(\"def obj_de(x):\\n base=R.render(x)[0].cpu().numpy().astype(np.float64)\", \"def obj_de(x):\\n x=np.asarray(x).T\\n base=R.render(x)[0].cpu().numpy().astype(np.float64)\")
open(p,'w').write(s)
PY
python /work/scratch/opt_camera.py"
```
<details><summary>output</summary>

```
T0 [[ 0.9994057  -0.00630215  0.03389078  0.05715222]
 [ 0.00506315  0.9993201   0.03652095  0.00344673]
 [-0.0340979  -0.03632765  0.998758    0.0349674 ]
 [ 0.          0.          0.          1.        ]]
base batch iou 0.7941762208938599 time 0.178358793258667
batch72 0.012906312942504883 torch.Size([72]) 1.39536384
start DE half
differential_evolution step 1: f(x)= 0.23800966993597777
differential_evolution step 2: f(x)= 0.23800966993597777
differential_evolution step 3: f(x)= 0.23800966993597777
differential_evolution step 4: f(x)= 0.22149892330990473
differential_evolution step 5: f(x)= 0.22149892330990473
differential_evolution step 6: f(x)= 0.21961895458213407
differential_evolution step 7: f(x)= 0.2070130516176296
differential_evolution step 8: f(x)= 0.17900359270016997
different …[+3955 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/check_mask_opt.py <<'PY'
exec(open('/work/scratch/opt_camera.py').read().split('ctx=dr.RasterizeCudaContext()')[0])
from scipy.ndimage import distance_transform_edt
ctx=dr.RasterizeCudaContext(); R=HumanRenderer(1.,ctx)
for name,x in [('base',np.zeros(6)),('mask',np.load('/work/scratch/opt_x_mask.npy'))]:
 loss,r=R.render(x); r=r[0].cpu().numpy(); m=mask0.astype(bool)
 iou=(r&m).sum()/(r|m).sum(); e1=cv2.Canny(r.astype(np.uint8)*255,0,1)>0;e2=cv2.Canny(m.astype(np.uint8)*255,0,1)>0
 d1=distance_transform_edt(~e1);d2=distance_transform_edt(~e2)
 print(name,'iou',iou,'areas',r.sum(),m.sum(),'cent',np.array(np.nonzero(r)).mean(1)[::-1],np.array(np.nonzero(m)).mean(1)[::-1])
 print(' bd r2m',np.quantile(d2[e1],[.5,.9,.95,1]),'m2r',np.quantile(d1[e2],[.5,.9,.95,1]))
 cv2.imwrite('/work/scratch/vis/overlay_'+name+'_full.png',np.stack([m*255,r*255,np.zeros_like(m)],2))
PY
python /work/scratch/check_mask_opt.py"
```
<details><summary>output</summary>

```
T0 [[ 0.9994057  -0.00630215  0.03389078  0.05715222]
 [ 0.00506315  0.9993201   0.03652095  0.00344673]
 [-0.0340979  -0.03632765  0.998758    0.0349674 ]
 [ 0.          0.          0.          1.        ]]
base iou 0.7912472557060602 areas 43087 43398 cent [558.00926033 641.83038968] [555.24480391 628.22017144]
 bd r2m [ 6.         13.41640786 14.76482306 18.02775638] m2r [ 7.61577311 15.23154621 16.97056275 33.83784863]
[ WARN:0@3.204] global loadsave.cpp:848 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
mask iou 0.8278846153846153 areas 47850 43398 cent [554.63262278 640.3400418 ] [555.24480391 628.22017144]
 bd r2m [ 3.60555128 13.03840481 16.         24.04163056] m2r [ 5.         17.48999504 21.         36.49657518]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"keypoints_2d\\|keypoint.*2d\\|pred_keypoints\" -n /code/reconstruction/modules/v2d_sam3d_body/lib | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py:103:                "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py:122:            "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py:350:        "pred_keypoints_3d": processed["pred_keypoints_3d"],
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py:413:    gt_keypoints_2d: torch.Tensor,
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py:415:    pred_keypoints_3d: torch.Tensor,
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py:423:    Projects pred_keypoints_3d into each camera and computes the  …[+13423 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,220p' /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/metadata/mhr70.py"
```
<details><summary>output</summary>

```
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The first 70 of 308 MHR keypoints, ignoring the rest for face keypoints"""

mhr_names = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-big-toe-tip",
    "left-small-toe-tip",
    "left-heel",
    "right-big-toe-tip",
    "right-small-toe-tip",
    "right-heel",
    "right-thumb-tip",
    "right-thumb-first-joint",
    "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip",
    "right-index-first-joint",
    "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-t …[+5432 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /weights -type f \\( -iname '*.onnx' -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' \\) -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/weights/cari4d/2026-08-25-09-35-57/step200000.pth 2084484133
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth 68220109
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth 190229389
/weights/grounding_dino/groundingdino_swint_ogc.pth 693997677
/weights/sam2/sam2.1_hiera_large.pt 898083611
/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt 696110248
/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt 2109129346
/weights/sam3d_body/torch_home/hub/checkpoints/dinov2_vitb14_pretrain.pth 346378731
/weights/sam3d_body/torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth 88283115

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['mediapipe','mmpose','ultralytics','timm','torchvision','onnxruntime','rtmlib','kornia','open3d']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__','?'))
 except Exception as e: print(m,'NO',type(e).__name__)
PY"
```
<details><summary>output</summary>

```
timm OK 1.0.30
torchvision OK 0.20.1+cu124
onnxruntime NO ModuleNotFoundError
rtmlib NO ModuleNotFoundError
kornia OK 0.8.3
open3d OK 0.20.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '300,410p' /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py; grep -n \"def _full_to_crop\\|pred_keypoints_2d\" -n /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/meta_arch/sam3d_body.py | head -30; sed -n '430,510p' /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/meta_arch/sam3d_body.py"
```
<details><summary>output</summary>

```
            init_estimate: [B, self.npose]
        """
        batch_size = x.shape[0]
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        # From pred, we want to pull out individual predictions.

        ## First, get globals
        ### Global rotation is first 6.
        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)  # B x 3 x 3
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)  # B x 3
        global_trans = torch.zeros_like(global_rot_euler)

        ## Next, get body pose.
        ### Hold onto raw, continuous version for iterative correction.
        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.bod …[+7854 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,240p' /code/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import imageio.v3 as iio
import numpy as np
import pyglet
pyglet.options['headless'] = True
import torch
from tqdm import tqdm

from v2d.common.datatypes import CameraIntrinsics
from v2d.common.video import FrameSource, get_video_writer

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
import trimesh
from v2d.mv.vis.renderer import Renderer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_SOURCE_MASK = "mask"
PROMPT_SOURCE_BBOX_FALLBACK = "bbox_fallback"
PROMPT_SOURCE_FULL_IMAGE_FALLBACK = "full_image_f …[+7187 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,260p' /code/reconstruction/modules/v2d_sam3d_body/docker/run_estimate_mhr_params.py; cat /weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model_config.yaml"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_estimate_mhr_params(
    rgb_path: str,
    weights_dir: str,
    output_params_path: str,
    cam_intrinsics_path: str | None = None,
    bbox_path: str | None = None,
    mask_path: str | None = None,
    output_mesh_path: str | None = None,
    batch_size: int = 1,
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
        "rgb_path": rgb_path,
        "weights_dir": weights_dir,
    }
    if cam_intrinsics_path is not None:
        inputs["cam_intrinsics_path"] = cam_intrinsics_path
 …[+4082 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction/modules/v2d_sam3d_body -maxdepth 2 -type f -printf '%p\\n'; find /workspace/v2d_sam3d_body -maxdepth 2 -type f -printf '%p\\n' | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_sam3d_body/lib/__init__.py
/code/reconstruction/modules/v2d_sam3d_body/lib/download_weights.py
/code/reconstruction/modules/v2d_sam3d_body/lib/export_soma.py
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.yaml
/code/reconstruction/modules/v2d_sam3d_body/lib/mv_optimize_mhr_params.py
/code/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py
/code/reconstruction/modules/v2d_sam3d_body/lib/renderer_gpu.py
/code/reconstruction/modules/v2d_sam3d_body/lib/visibility.py
/code/reconstruction/modules/v2d_sam3d_body/lib/pyproject.toml
/code/reconstruction/modules/v2d_sam3d_body/docker/__init__.py
/code/reconstruction/modules/v2d_sam3d_body/docker/run_mv_optimize_mhr_params.py
/code/reconstruction/modules/v2d_sam3d_body/docker/run_ …[+1725 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -type f \\( -iname '*keypoint*' -o -iname '*.json' -o -iname '*.pt' -o -iname '*.pth' \\) -printf '%p %s\\n' | sort | head -100; grep -n \"output_params_path\\|torch.save\\|pred_keypoints_2d\" /code/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py | tail -30"
```
<details><summary>output</summary>

```
/pipeline/depth/moge2/depth_report.json 1804
/pipeline/export/episode_000000/wild_export.json 2778
/pipeline/inference/coconet.pth 595065104
/pipeline/inference/refined.pth 595128736
/pipeline/pipeline_report.json 5837
/pipeline/work/prompts.json 596
208:        "pred_keypoints_2d": _stack("pred_keypoints_2d"),
247:        "pred_keypoints_2d": mhr_outputs["pred_keypoints_2d"],
263:    output_params_path: Path,
284:    Returns the coalesced mhr_params dict and saves it to *output_params_path*.
286:    output_params_path = Path(output_params_path).resolve()
287:    output_params_path.parent.mkdir(parents=True, exist_ok=True)
332:        (output_params_path.parent / "mhr_overlay").mkdir(parents=True, exist_ok=True)
334:            writer = get_video_writer(output_params_path.parent / "mhr_ove …[+721 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '250,520p' /code/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py; sed -n '1,260p' /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/build_models.py"
```
<details><summary>output</summary>

```
        "pred_global_rots": mhr_outputs["pred_global_rots"],
        "mhr_model_params": mhr_model_params,
    }
    mhr_mesh = {
        "faces": faces,
        "pred_vertices": pred_vertices,
        "pred_cam_t": mhr_outputs["pred_cam_t"],
    }
    return mhr_params, mhr_mesh


def estimate_mhr_params(
    rgb_path: Path,
    output_params_path: Path,
    cam_intrinsics: np.ndarray | None = None,
    bbox_path: Path | None = None,
    mask_path: Path | None = None,
    output_mesh_path: Path | None = None,
    estimator: SAM3DBodyEstimator | None = None,
    weights_dir: Path | None = None,
    batch_size: int = 1,
    debug: int = 0,
) -> dict:
    """Run SAM3D-Body inference on a single camera's frames.

    If *cam_intrinsics* is None, the model uses a default FOV.
    Either *bbox_ …[+12275 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep -R "class FrameSource" -n /workspace /code/reconstruction 2>/dev/null | head; grep -R "def from_path" -n /workspace/v2d* /code/reconstruction/modules 2>/dev/null | head -20'
```
<details><summary>output</summary>

```
/workspace/v2d_common/video.py:214:class FrameSource:
/code/reconstruction/modules/v2d_common/video.py:214:class FrameSource:
/workspace/v2d_common/video.py:236:    def from_path(cls, path: str | Path, frames_slice: slice | None = None) -> "FrameSource":
/workspace/v2d_common/video.py:611:    def from_path(
/workspace/v2d_sam2/lib/tests/test_mv_videos_to_masks.py:45:    def from_path(cls, _path):
/code/reconstruction/modules/v2d_cari4d/tests/test_runtime_contract.py:97:        def from_path(cls, path):
/code/reconstruction/modules/v2d_common/video.py:236:    def from_path(cls, path: str | Path, frames_slice: slice | None = None) -> "FrameSource":
/code/reconstruction/modules/v2d_common/video.py:611:    def from_path(
/code/reconstruction/modules/v2d_sam2/lib/tests/test_mv_videos_to_masks.p …[+519 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '214,330p' /workspace/v2d_common/video.py"
```
<details><summary>output</summary>

```
class FrameSource:
    """Read-only frame sequence backed by an image directory, video, or HDF5.

    Construct via the :meth:`from_path` factory, which auto-detects the
    backend from the path.  All backends expose the same public interface:

    * ``n_frames``, ``image_size``, ``stems``, ``path``
    * ``__getitem__(i)`` -- random access (not available for video)
    * ``iter_frames()`` / ``iter_batches(batch_size)`` -- sequential
    * ``close()`` / context-manager protocol
    """

    n_frames: int
    image_size: tuple[int, int]  # (W, H)
    _path: Path
    _stems: list[str]

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: s …[+3303 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /workspace/v2d_sam3d_body/assets/cam_intrinsics.json; grep -n "cam_intrinsics_path" -A30 /code/reconstruction/modules/v2d_sam3d_body/lib/estimate_mhr_params.py | tail -30; grep -n "def load_cam" -R /workspace/v2d* | head'
```
<details><summary>output</summary>

```
/workspace/v2d_mv/rig/params.py:78:def load_camera_params(source_path: Path) -> list[CameraParam]:
/workspace/v2d_mv/rig/rig.py:108:    def load_camera_params(self, camera_params_path: str | Path) -> None:

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/scratch/frame0rgb /work/scratch/frame0mask && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -frames:v 1 /work/scratch/frame0rgb/000000.png -y && python - <<'PY'
import h5py,cv2,numpy as np,json
with h5py.File('/work/inputs/masks.h5','r') as f: m=(f['person'][0].astype(np.uint8)*255)
cv2.imwrite('/work/scratch/frame0mask/000000.png',m)
json.dump({'fx':819.9987182617188,'fy':819.9987182617188,'cx':758.0999755859375,'cy':560.5000610351562,'width':1536,'height':1152},open('/work/scratch/cam_frame0.json','w'))
PY
cd /workspace && PYTHONPATH=/workspace/v2d_sam3d_body/lib:/workspace CUDA_VISIBLE_DEVICES=0 python -m v2d_sam3d_body.lib.estimate_mhr_params --rgb_path /work/scratch/frame0rgb --mask_path /work/scratch/frame0mask --cam_intrinsics_path /work/scratch/cam_frame0.json --weights_dir /weights/sam3d_body --output_params_path /work/scratch/frame0_sam3d_params.pt --batch_size 1 --debug 0"
```
<details><summary>output</summary>

```
/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py:33: UserWarning: Momentum is not enabled
  warnings.warn("Momentum is not enabled")
Loading SAM 3D Body model...
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/v2d_sam3d_body/lib/estimate_mhr_params.py", line 529, in <module>
    estimate_mhr_params(
  File "/workspace/v2d_sam3d_body/lib/estimate_mhr_params.py", line 320, in estimate_mhr_params
    model, model_cfg = load_sam_3d_body(
                       ^^^^^^^^^^^^^^^^^
  File "/workspace/v2d_sam3d_body/lib/sam_3d_body/build_models.py", line 21, in load_sam_3d_body
    model_cfg = get_config(model_cfg)
                ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/ …[+471 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cd /workspace && PYTHONPATH=/workspace/v2d_sam3d_body/lib:/workspace CUDA_VISIBLE_DEVICES=0 python -m v2d_sam3d_body.lib.estimate_mhr_params --rgb_path /work/scratch/frame0rgb --mask_path /work/scratch/frame0mask --cam_intrinsics_path /work/scratch/cam_frame0.json --weights_dir /weights/sam3d_body/checkpoints --output_params_path /work/scratch/frame0_sam3d_params.pt --batch_size 1 --debug 0'
```
<details><summary>output</summary>

```
/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py:33: UserWarning: Momentum is not enabled
  warnings.warn("Momentum is not enabled")
Loading SAM 3D Body model...
Using cache found in /weights/sam3d_body/torch_home/hub/facebookresearch_dinov3_main
Ignored kwargs: {'drop_path': 0.1}
The model and loaded state dict do not match exactly

missing keys in source state_dict: backbone.encoder.mask_token, head_pose.hand_pose_comps_ori, head_pose.mhr.face_expressions_model.shape_vectors, head_pose.mhr.pose_correctives_model.pose_dirs_predictor.0.sparse_indices, head_pose.mhr.pose_correctives_model.pose_dirs_predictor.0.sparse_weight, head_pose.mhr.pose_correctives_model.pose_dirs_predictor.2.weight, head_pose.mhr.character_torch.skeleton.joint_translation_offsets, head_pose.mhr.cha …[+7764 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"keypoint_mapping\\|pred_keypoints_3d\\|return_keypoints\" /workspace/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py /workspace/v2d_sam3d_body/lib/sam_3d_body/models/modules/mhr_utils.py /code/reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body/models/modules/mhr_utils.py | head -100; sed -n '40,150p' /workspace/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py:23:    "keypoint_mapping",
/workspace/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py:387:        vertices = context.head.mhr_forward(global_trans=trans.new_zeros(trans.shape), global_rot=context.global_rot, body_pose_params=body_pose_params, hand_pose_params=context.hand, scale_params=scale, shape_params=shape, expr_params=context.face, return_keypoints=False, return_joint_coords=False, return_model_params=False, return_joint_rotations=False)
/workspace/v2d_cari4d/lib/cari4d/lib_mhr/mhr_layer.py:413:            vertices = head.mhr_forward(global_trans=current.new_zeros((count, 3)), global_rot=current.new_zeros((count, 3)), body_pose_params=current.new_zeros((count, 133)), hand_pose_params=current.new_zeros((count, 108)), scale_params=curren …[+3719 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False)
M=w['head_pose.keypoint_mapping'].numpy()
print(M.shape,np.unique(M).size,M.min(),M.max(),np.count_nonzero(M,axis=1)[:100],np.quantile(np.count_nonzero(M,axis=1),[0,.5,.9,1]))
for i in range(20):
 idx=np.nonzero(M[i])[0]; print(i,len(idx),idx[:20],M[i,idx[:20]],'vsum',M[i,:18439].sum(),'jsum',M[i,18439:].sum())
PY"
```
<details><summary>output</summary>

```
(308, 18566) 2578 -6.8081676e-14 1.0 [ 8 15 15 19 24  1  1 21 18  1  1  1  1 40 43 17 24 30 18 28 30  1  1  1
  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1 16 17 39 31 20 22  2  9 10
  9  9  8  8  7  5  7  7  9  8  8  7  7  8  8  7  7  8  8  6  6  8  9  8
  9  8  5  9] [ 1.   8.  12.3 43. ]
0 8 [ 419  500 1799 2766 2767 3305 3386 4685] [1.8876980e-01 3.3581744e-06 3.0774336e-06 2.7440894e-05 5.9114116e-01
 2.2004767e-01 3.9145334e-06 3.5872813e-06] vsum 1.0 jsum 0.0
1 15 [1383 1495 1497 1498 1756 1757 1912 1913 1916 1926 1932 2615 2616 2621
 2622] [ 1.9830442e-03  7.4310170e-05 -2.4545559e-16  1.4419011e-05
  1.6022161e-02  7.7285267e-02  3.8866478e-06  3.7238455e-01
  6.0379542e-02  1.5150152e-01  7.1751639e-02  1.20 …[+5239 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
z=torch.load('/work/scratch/frame0_sam3d_params.pt',map_location='cpu',weights_only=False)
for k,v in z.items():
 if hasattr(v,'shape'): print(k,tuple(v.shape),v.dtype)
 else: print(k,type(v))
k=z['pred_keypoints_2d']; print('2d',k.shape,k[:3])
print('3d',z['pred_keypoints_3d'].shape,z['pred_keypoints_3d'][0,:5])
print('cam_t',z['pred_cam_t'])
print('focal',z['focal_length'])
PY"
```
<details><summary>output</summary>

```
global_rot (1, 3) torch.float32
body_pose_params (1, 133) torch.float32
hand_pose_params (1, 108) torch.float32
scale_params (1, 28) torch.float32
shape_params (1, 45) torch.float32
expr_params (1, 72) torch.float32
pred_cam_t (1, 3) torch.float32
focal_length (1,) torch.float32
pred_keypoints_3d (1, 70, 3) torch.float32
pred_keypoints_2d (1, 70, 2) torch.float32
pred_joint_coords (1, 127, 3) torch.float32
pred_pose_raw (1, 266) torch.float32
pred_global_rots (1, 127, 3, 3) torch.float32
mhr_model_params (1, 204) torch.float32
2d torch.Size([1, 70, 2]) tensor([[[547.2609, 466.4711],
         [558.9039, 455.1150],
         [541.9235, 453.5519],
         [581.7285, 455.9144],
         [541.6138, 452.5995],
         [609.6974, 509.6192],
         [523.1788, 503.9528],
         [616.8578, 586. …[+2212 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
z=torch.load('/work/scratch/frame0_sam3d_params.pt',map_location='cpu',weights_only=False)
K=np.array([[819.9987,0,758.1],[0,819.9987,560.5],[0,0,1]])
p=z['pred_keypoints_3d'][0].numpy(); uv=z['pred_keypoints_2d'][0].numpy(); X=p.copy(); X[:,1]*=-1; X[:,2]*=-1
proj=(K@X.T).T; proj=proj[:,:2]/proj[:,2:]
print('error if flip yz',np.linalg.norm(proj-uv,axis=1).mean(),np.quantile(np.linalg.norm(proj-uv,axis=1),[.5,.9,1]))
proj=(K@p.T).T; proj=proj[:,:2]/proj[:,2:]
print('error raw',np.linalg.norm(proj-uv,axis=1).mean(),np.quantile(np.linalg.norm(proj-uv,axis=1),[.5,.9,1]))
print(p[:3],uv[:3])
PY"
```
<details><summary>output</summary>

```
error if flip yz 414.5807436438497 [410.0352231  531.44895573 538.17902797]
error raw 7.336427257413667e-05 [7.26267651e-05 1.03071275e-04 1.10297653e-04]
[[-0.74778134 -0.33349192  2.908283  ]
 [-0.7090609  -0.37513006  2.9188776 ]
 [-0.7719369  -0.38189733  2.9281049 ]] [[547.26086 466.47107]
 [558.9039  455.115  ]
 [541.9235  453.55194]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
M=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False)['head_pose.keypoint_mapping'].numpy()[:70]
for i in range(70):
 idx=np.nonzero(M[i])[0]; v=idx[idx<18439]; j=idx[idx>=18439]
 print(i,'verts',len(v),'max',v.max() if len(v) else '-','joints',j-18439 if len(j) else '')
PY"
```
<details><summary>output</summary>

```
0 verts 8 max 4685 joints 
1 verts 15 max 2622 joints 
2 verts 15 max 5502 joints 
3 verts 19 max 324 joints 
4 verts 24 max 3207 joints 
5 verts 0 max - joints [75]
6 verts 0 max - joints [39]
7 verts 21 max 8459 joints 
8 verts 18 max 13601 joints 
9 verts 0 max - joints [2]
10 verts 0 max - joints [18]
11 verts 0 max - joints [3]
12 verts 0 max - joints [19]
13 verts 40 max 12517 joints 
14 verts 43 max 17603 joints 
15 verts 17 max 13262 joints 
16 verts 24 max 13263 joints 
17 verts 30 max 12513 joints 
18 verts 18 max 18426 joints 
19 verts 28 max 18436 joints 
20 verts 30 max 17655 joints 
21 verts 0 max - joints [64]
22 verts 0 max - joints [63]
23 verts 0 max - joints [62]
24 verts 0 max - joints [61]
25 verts 0 max - joints [59]
26 verts 0 max - joints [58]
27 verts 0 max - joint …[+1233 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
z=torch.load('/work/scratch/frame0_sam3d_params.pt',map_location='cpu',weights_only=False)
print('params keys',z.keys())
p=z['pred_vertices'][0].numpy(); print(p.shape)
r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
# fresh pred is camera coords. reconstruct world via baseline
import json
T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
R=T[:3,:3];t=T[:3,3]; pw=p@R.T+t
print('fresh vs reconstruction0 nearest')
from scipy.spatial import cKDTree
d,i=cKDTree(r).query(pw); print(d.mean(),np.median(d),np.quantile(d,[.9,.99,1]),'unique',len(np.unique(i)))
# exact subset hypothesis fresh indices i? are they i=0..step?
print(i[:50]); print('unique sorted first',np.unique(i)[:50])
PY"
```
<details><summary>output</summary>

```
params keys dict_keys(['global_rot', 'body_pose_params', 'hand_pose_params', 'scale_params', 'shape_params', 'expr_params', 'pred_cam_t', 'focal_length', 'pred_keypoints_3d', 'pred_keypoints_2d', 'pred_joint_coords', 'pred_pose_raw', 'pred_global_rots', 'mhr_model_params'])
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
KeyError: 'pred_vertices'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cd /workspace && PYTHONPATH=/workspace/v2d_sam3d_body/lib:/workspace CUDA_VISIBLE_DEVICES=0 python -m v2d_sam3d_body.lib.estimate_mhr_params --rgb_path /work/scratch/frame0rgb --mask_path /work/scratch/frame0mask --cam_intrinsics_path /work/scratch/cam_frame0.json --weights_dir /weights/sam3d_body/checkpoints --output_params_path /work/scratch/frame0_sam3d_params.pt --output_mesh_path /work/scratch/frame0_sam3d_mesh.pt --batch_size 1 --debug 0'
```
<details><summary>output</summary>

```
/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py:33: UserWarning: Momentum is not enabled
  warnings.warn("Momentum is not enabled")
Loading SAM 3D Body model...
Using cache found in /weights/sam3d_body/torch_home/hub/facebookresearch_dinov3_main
Ignored kwargs: {'drop_path': 0.1}
The model and loaded state dict do not match exactly

missing keys in source state_dict: backbone.encoder.mask_token, head_pose.hand_pose_comps_ori, head_pose.mhr.face_expressions_model.shape_vectors, head_pose.mhr.pose_correctives_model.pose_dirs_predictor.0.sparse_indices, head_pose.mhr.pose_correctives_model.pose_dirs_predictor.0.sparse_weight, head_pose.mhr.pose_correctives_model.pose_dirs_predictor.2.weight, head_pose.mhr.character_torch.skeleton.joint_translation_offsets, head_pose.mhr.cha …[+7830 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,json
from scipy.spatial import cKDTree
z=torch.load('/work/scratch/frame0_sam3d_mesh.pt',map_location='cpu',weights_only=False)
print(z.keys(),z['pred_vertices'].shape,z['faces'].shape)
p=z['pred_vertices'][0].numpy(); T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
pw=p@T[:3,:3].T+T[:3,3]
r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
print('fresh world range',pw.min(0),pw.max(0),'recon',r.min(0),r.max(0))
d,i=cKDTree(r).query(pw); print('nearest d',d.mean(),np.median(d),np.quantile(d,[.9,.99,1]),'unique',len(np.unique(i)))
# likely recon is fresh pred[::4]
a=pw[::4]; print(len(a),a.shape,r.shape)
d=np.linalg.norm(a-r,axis=1); print('pw[::4] vs recon',d.mean(),np.median(d),np.quantile(d,[.9,.99,1]))
# exact?
print('max first',d[:20])
PY"
```
<details><summary>output</summary>

```
dict_keys(['faces', 'pred_vertices', 'pred_cam_t']) torch.Size([1, 18439, 3]) torch.Size([36874, 3])
fresh world range [-0.88194017 -0.40880677  2.96882453] [-0.33425196  1.24721219  3.39497097] recon [-0.8599326  -0.40705746  3.0181105 ] [-0.3237478  1.246326   3.4133868]
nearest d 0.0273293942076161 0.0281588075551172 [0.04495412 0.05922142 0.06860141] unique 2606
4610 (4610, 3) (4610, 3)
pw[::4] vs recon 0.04477564198846081 0.04710559527689989 [0.0656914  0.06947446 0.06994174]
max first [0.04646968 0.04655934 0.04683809 0.04664494 0.04652859 0.04640915
 0.04666013 0.04699129 0.04683266 0.04664973 0.04652601 0.04645279
 0.04646934 0.04690213 0.0468485  0.04670824 0.04656233 0.04649128
 0.04697449 0.04683929]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,json
z=torch.load('/work/scratch/frame0_sam3d_mesh.pt',map_location='cpu',weights_only=False);p=z['pred_vertices'][0].numpy()
T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);pw=p@T[:3,:3].T+T[:3,3]
r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]; a=pw[::4]
d=a-r; print('diff mean',d.mean(0),'std',d.std(0),'norms',np.linalg.norm(d-d.mean(0),axis=1).mean(),np.quantile(np.linalg.norm(d-d.mean(0),axis=1),[.5,.9,1]))
# Procrustes similarity
A=a-a.mean(0); B=r-r.mean(0); U,S,Vt=np.linalg.svd(A.T@B); R=U@Vt; s=(S.sum())/np.sum(A*A); print('rot',R,'s',s,'resid',np.linalg.norm(s*A@R-B)/len(A))
PY"
```
<details><summary>output</summary>

```
diff mean [-0.00786716 -0.00344702 -0.03977511] std [0.00678241 0.00485257 0.02147415] norms 0.01824360118305189 [0.01164829 0.04541502 0.05603535]
rot [[ 0.99953447 -0.01310829  0.02755009]
 [ 0.0137563   0.99963001 -0.0234649 ]
 [-0.02723231  0.02383296  0.99934498]] s 0.9980183559599014 resid 0.00025817120243828377

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/dense_opt.py <<'PY'
import numpy as np,json
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); T0=np.array(c['T_world_cam'])
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
d,idx=cKDTree(h).query(r)
# likely filters
for thresh in [.01,.015,.02,.03,.05]:
 sel=d<thresh; print('thresh',thresh,'n',sel.sum(),'uniq',len(np.unique(idx[sel])))
# target uv based on recon under baseline
def project(T,X,K=K):
 R=T[:3,:3];t=T[:3,3]; Xc=(X-t)@R # row: R"'^T(X-t)
 q=Xc@K.T; return q[:,:2]/q[:,2:3],Xc[:,2]
uv,z=project(T0,r)
# baseline reproj matched
p0,_=project(T0,h[idx]); e=np.linalg.norm(p0-uv,axis=1)
print('"'baseline dense matched pixel err',np.mean(e),np.median(e),np.quantile(e,[.9,.95,.99,1]))
# weighted correspondences using nearest distances and mask target inside? Reject far match and non-unique not needed
for thresh in [.012,.02,.03]:
 sel=d<thresh
 X=h[idx[sel]]; target=uv[sel]; w=1/(.003+d[sel])**2
 def fun(x):
  Rr=Rotation.from_rotvec(x[:3]).as_matrix(); R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:]
  pp,_=project(np.block([[R,t[:,None]],[0,0,0,1]]),X)
  return ((pp-target)*np.sqrt(w[:,None])).ravel()
 res=least_squares(fun,np.zeros(6),loss='soft_l1',f_scale=3.0,max_nfev=2000,xtol=1e-12,ftol=1e-12,gtol=1e-12,verbose=0)
 x=r …[+583 chars]
```
<details><summary>output</summary>

```
thresh 0.01 n 2775 uniq 2343
thresh 0.015 n 3986 uniq 3288
thresh 0.02 n 4511 uniq 3715
thresh 0.03 n 4610 uniq 3795
thresh 0.05 n 4610 uniq 3795
baseline dense matched pixel err 1.7300034690890984 1.6059611242073881 [3.15830445 3.69083029 4.64765825 7.15574838]

thresh 0.012 n 3342 x [-0.00211575  0.00144642 -0.00027864 -0.00484379 -0.00733803  0.00027717] obj 1631208.7408109664 change rot 0.14770949661615573 center cm 0.8796920122472808
e2 all 1.7138228799790132 1.5692832010144322 [3.16107977 3.6488305  4.66451343 7.21262135] e2 sel 1.3394956709082957
R [[ 0.99935666 -0.00607778  0.03534578]
 [ 0.004711    0.9992427   0.03862428]
 [-0.03555376 -0.03843292  0.99862848]] t [ 0.05230843 -0.0038913   0.03524457]

thresh 0.02 n 4511 x [-0.00306131  0.00360123 -0.00046831 -0.01163367 -0.010372 …[+828 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation
T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];dw=P[3]
x=np.load('/work/scratch/dense_x_0.03.npy');Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T[:3,:3];t=T[:3,3]+x[3:];nc=R.T@nw;d=nw@t+dw
nobs=np.array([-0.02020254,-0.99477472,-0.09991089]);nobs/=np.linalg.norm(nobs)
print('dense nc',nc,'dc',d,'angle',np.degrees(np.arccos(np.clip(nc@nobs,-1,1))))
print('base nc',T[:3,:3].T@nw,'dc',nw@T[:3,3]+dw,'angle',np.degrees(np.arccos(np.clip(T[:3,:3].T@nw@nobs,-1,1))))
PY"
```
<details><summary>output</summary>

```
dense nc [-0.02126402 -0.99518353 -0.0956953 ] dc 1.3900780449628647 angle 0.25017397043330913
base nc [-0.02204671 -0.99546902 -0.09249523] dc 1.3791749153771582 angle 0.43963441201081566

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/kp_opt.py <<'PY'
import numpy as np,torch,json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam'])
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; rec=np.load('/work/inputs/reconstruction.npz')['human_keypoints_world'][0]
z=torch.load('/work/scratch/frame0_sam3d_params.pt',map_location='cpu',weights_only=False); uv=z['pred_keypoints_2d'][0].numpy(); k3cam=z['pred_keypoints_3d'][0].numpy()
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False);M=w['head_pose.keypoint_mapping'].numpy()[:70]
X=[]
for i,row in enumerate(M):
 v=np.nonzero(row[:18439])[0]; j=np.nonzero(row[18439:])[0]
 if len(v) and v.max()<len(h): X.append(row[v]@h[v])
 else: X.append(rec[i])
X=np.array(X)
print('X finite',X.min(0),X.max(0))
def project(T,X):
 R=T[:3,:3];t=T[:3,3];Xc=(X-t)@R;q=Xc@K.T;return q[:,:2]/q[:,2:3]
def makeT(x):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T
p0=project(T0,X);e=np.linalg.norm(p0-uv,axis=1)
print('baseline kp err',np.mean(e),np.median(e),np.quantile(e,[.9,.95,.99,1]));print('per',np.round(e,2));print('body 0-20,63-69',e[np.r_[0:21,63:70]].mean())
sets={'body':np.r_[0:21,63:70],'body_no_feet':np.r_[0:18,63:70],'all':np.arange(70)}
for name,sel in sets.items( …[+640 chars]
```
<details><summary>output</summary>

```
X finite [-0.85496557 -0.25605345  3.0420303 ] [-0.35770863  1.2166988   3.282249  ]
baseline kp err 36.235546475384396 6.978994186794999 [138.79221096 159.54420058 212.63147392 296.94902831]
per [ 54.32 296.95  21.04  12.65 170.12   3.5    2.67 142.21  34.74   4.51
   3.33   5.02   7.44 146.62 139.45 109.6  103.   173.9    7.06   6.07
 122.01   5.24   5.22   5.18   5.12   5.27   5.37   5.46   5.46   5.03
   5.18   5.33   5.32   4.97   5.08   5.19   5.14   5.03   5.06   5.05
   4.96   5.     7.27   7.14   6.94   6.7    7.46   7.39   7.3    7.06
   7.4    7.32   7.2    6.93   7.3    7.2    7.05   6.78   7.13   7.02
   6.86   6.65   6.39  89.37 113.77 138.72  66.78 126.54 174.75   3.17]
body 0-20,63-69 81.40444363421157

 body n 28 x [ 0.94421069 -0.6309473   0.63262564  0.14650237  2.243199 …[+1192 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,json
r=np.load('/work/inputs/reconstruction.npz')['human_keypoints_world'][0]
z=torch.load('/work/scratch/frame0_sam3d_params.pt',map_location='cpu',weights_only=False);p=z['pred_keypoints_3d'][0].numpy();T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);pw=p@T[:3,:3].T+T[:3,3]
d=np.linalg.norm(r-pw,axis=1);print('same index',d.mean(),np.median(d),np.quantile(d,[.9,1]))
from scipy.spatial import cKDTree
dd,ii=cKDTree(pw).query(r);print('nearest',dd.mean(),np.median(dd),np.quantile(dd,[.9,1]),'mapping',ii)
print('r first5',r[:5]);print('pw first5',pw[:5])
PY"
```
<details><summary>output</summary>

```
same index 0.050740659456100574 0.05157969394323541 [0.06869887 0.06972756]
nearest 0.039126837467265106 0.03956226860878573 [0.06152294 0.06868423] mapping [ 0  1  2  3  4  5  6 63 64  9 10 11 12 13 14 15 16 17 18 19 20 33 22 23
 24 33 33 37 36 33 37 37 40 37 37 38 40 37 37 38 40 41 58 58 44 62 54 58
 58 61 58 58 58 61 58 58 59 61 58 58 59 61 62 63 64  7  8  5  6 69]
r first5 [[-0.58812815 -0.22758718  3.0252788 ]
 [-0.5490067  -0.26894587  3.0356016 ]
 [-0.6110062  -0.2752627   3.0481641 ]
 [-0.48338148 -0.27263263  3.122699  ]
 [-0.6338624  -0.28676492  3.154066  ]]
pw first5 [[-0.58951911 -0.22739131  2.97725143]
 [-0.55020024 -0.26841817  2.98802498]
 [-0.61268341 -0.27516219  2.99963104]
 [-0.48523018 -0.27343855  3.07604128]
 [-0.63763091 -0.28859751  3.10524581]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/dense_normals.py <<'PY'
import numpy as np,trimesh,torch,json
from scipy.spatial import cKDTree
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
w=torch.load('/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt',map_location='cpu',weights_only=False);F=w['head_pose.faces'].numpy(); Ftrue=F[(F<len(h)).all(1)]
th=trimesh.Trimesh(h,Ftrue,process=False,validate=False); nt=th.vertex_normals
# fresh mesh normals
z=torch.load('/work/scratch/frame0_sam3d_mesh.pt',map_location='cpu',weights_only=False);V=z['pred_vertices'][0].numpy(); T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); pw=V@T[:3,:3].T+T[:3,3]
tm=trimesh.Trimesh(V,F,process=False,validate=False); nm=tm.vertex_normals[::4]
# similarity orientation from fresh world pw[::4] to reconstruction r
A=pw[::4]-pw[::4].mean(0);B=r-r.mean(0);U,S,Vt=np.linalg.svd(A.T@B);Rsim=U@Vt;nm=nm@Rsim.T
d,idx=cKDTree(h).query(r); ndot=np.sum(nm*nt[idx],axis=1)
print('dist q',np.quantile(d,[0,.5,.9,.99,1]));print('dot q',np.quantile(ndot,[0,.05,.1,.25,.5,.75,.9,.95,1]))
for dt in [.01,.015,.02,.03]:
 for ang in [90,60,45,30]:
  sel=(d<dt)&(ndot>np.cos(np.radians(ang)))
  print('d',dt,'ang',ang,'n',sel.sum(),'uniq',len(np.unique(idx[sel])))
PY
python /work/scratch/dense_normals.py"
```
<details><summary>output</summary>

```
dist q [0.00020318 0.00847426 0.01600374 0.02127358 0.02968541]
dot q [-0.99986895 -0.95211358 -0.89575729 -0.59810444 -0.05464721  0.53797427
  0.86269125  0.94248397  0.99994962]
d 0.01 ang 90 n 1362 uniq 1194
d 0.01 ang 60 n 776 uniq 691
d 0.01 ang 45 n 486 uniq 447
d 0.01 ang 30 n 268 uniq 246
d 0.015 ang 90 n 1922 uniq 1654
d 0.015 ang 60 n 1064 uniq 945
d 0.015 ang 45 n 688 uniq 624
d 0.015 ang 30 n 384 uniq 354
d 0.02 ang 90 n 2146 uniq 1834
d 0.02 ang 60 n 1199 uniq 1054
d 0.02 ang 45 n 782 uniq 702
d 0.02 ang 30 n 434 uniq 398
d 0.03 ang 90 n 2195 uniq 1879
d 0.03 ang 60 n 1230 uniq 1082
d 0.03 ang 45 n 804 uniq 721
d 0.03 ang 30 n 448 uniq 411

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace/v2d_sam3d_body /code/reconstruction/modules/v2d_sam3d_body /workspace/v2d_cari4d -type f | grep -Ei '\\.(npz|obj|ply|glb|gltf|pt|pth)"'$'"' | sort"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_sam3d_body/assets/bbox_track.pt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"18056\\|18439\\|18566\\|36874\" -n /workspace/v2d_sam3d_body /workspace/v2d_cari4d /code/reconstruction/modules/v2d_sam3d_body /code/reconstruction/modules/v2d_cari4d 2>/dev/null | head -200"
```
<details><summary>output</summary>

```
/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py:87:        self.faces = nn.Parameter(torch.zeros(36874, 3).long(), requires_grad=False)
/workspace/v2d_sam3d_body/lib/sam_3d_body/models/heads/mhr_head.py:97:            torch.zeros(308, 18439 + 127), requires_grad=False
/workspace/v2d_sam3d_body/lib/tools/vis_utils.py:121:    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
/workspace/v2d_cari4d/lib/cari4d/lib_mhr/hand_surface_contact.py:122:    if vertices.shape != (1, 18439, 3) or not np.isfinite(vertices).all():
/workspace/v2d_cari4d/lib/cari4d/lib_mhr/hand_surface_contact.py:123:        raise ValueError(f"neutral MHR vertices must be finite [1,18439,3], got {vertices.shape}")
/workspace/v2d_cari4d/lib/c …[+1402 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
p='/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
m=torch.jit.load(p,map_location='cpu')
print(m)
print('attrs/params')
for name,p in m.named_parameters(): print(name,tuple(p.shape))
for name,b in m.named_buffers(): print('B',name,tuple(b.shape))
PY"
```
<details><summary>output</summary>

```
RecursiveScriptModule(
  original_name=MHRDemo
  (face_expressions_model): RecursiveScriptModule(original_name=BlendShapeBase)
  (pose_correctives_model): RecursiveScriptModule(
    original_name=MHRPoseCorrectivesModel
    (pose_dirs_predictor): RecursiveScriptModule(
      original_name=Sequential
      (0): RecursiveScriptModule(original_name=SparseLinear)
      (1): RecursiveScriptModule(original_name=ReLU)
      (2): RecursiveScriptModule(original_name=Linear)
    )
  )
  (character_torch): RecursiveScriptModule(
    original_name=Character
    (skeleton): RecursiveScriptModule(original_name=Skeleton)
    (mesh): RecursiveScriptModule(original_name=Mesh)
    (parameter_transform): RecursiveScriptModule(original_name=ParameterTransform)
    (linear_blend_skinning): RecursiveScriptModul …[+3426 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
p='/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
m=torch.jit.load(p,map_location='cpu')
for meth in ['mesh_faces','mesh_vertices','rest_vertices','forward']:
 print(meth,hasattr(m,meth))
 if hasattr(m,meth):
  try: print(' call',getattr(m,meth)())
  except Exception as e: print(' err',repr(e)[:200])
PY"
```
<details><summary>output</summary>

```
mesh_faces False
mesh_vertices False
rest_vertices False
forward True
 err RuntimeError("forward() is missing value for argument 'identity_coeffs'. Declaration: forward(__torch__.MHRDemo self, Tensor identity_coeffs, Tensor model_parameters, Tensor face_expr_coeffs, bool app

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /weights -maxdepth 5 -type f -printf '%p %s\\n' | sort | grep -v 'hf_home\\|cari4d\\|sam2\\|foundationpose\\|grounding' | head -300; find /workspace /code -type f \\( -iname '*soma*' -o -iname '*smpl*' -o -iname '*neutral*' -o -iname '*body_model*' \\) 2>/dev/null | head -200"
```
<details><summary>output</summary>

```
/workspace/v2d_sam3d_body/docker/run_export_soma.py
/workspace/v2d_sam3d_body/lib/export_soma.py
/workspace/v2d_sam3d_body/lib/tests/test_export_soma_options.py
/code/robotic_grounding/scripts/setup_soma_assets.py
/code/robotic_grounding/scripts/retarget/create_soma_task.py
/code/robotic_grounding/scripts/retarget/soma_to_g1.py
/code/robotic_grounding/scripts/retarget/soma_loader_probe.py
/code/robotic_grounding/scripts/retarget/process_soma_sequence.sh
/code/robotic_grounding/tests/test_read_soma_assets.py
/code/robotic_grounding/source/robotic_grounding/robotic_grounding/retarget/read_soma.py
/code/reconstruction/modules/v2d_sam3d_body/lib/export_soma.py
/code/reconstruction/modules/v2d_sam3d_body/lib/tests/test_export_soma_options.py
/code/reconstruction/modules/v2d_sam3d_body/docker/ru …[+17 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,240p' /code/robotic_grounding/scripts/setup_soma_assets.py; sed -n '1,240p' /workspace/v2d_sam3d_body/lib/export_soma.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-time downloader for SOMA-X body-model assets.

Calls ``soma.assets.get_assets_dir(revision=...)`` so SOMA-X downloads a
pinned legacy asset bundle into its built-in HuggingFace cache, then copies the
specific files that ``robotic_grounding.retarget.read_soma`` requires into the
canonical repo path (``assets/body_models/soma/`` by default).

The HF cache path is ephemeral inside the retarget Docker container
(``HOME=/tmp`` is tmpfs), so the bind-mounted repo path is the only
location that survives container restarts.

The required asset list is the source of truth in
``robotic_grounding.retarget.read_soma._SOMA_REQUIRED_ASSETS`` /
``_SOMA_REQUIRED_B …[+15741 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['soma','soma_x','robotic_grounding']
for m in mods:
 try:
  x=__import__(m);print(m,x.__file__)
 except Exception as e: print(m,repr(e))
PY
find / -path '*soma*' -type f \\( -iname '*.npz' -o -iname '*.npy' -o -iname '*.obj' -o -iname '*.pt' \\) 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
soma /opt/conda/lib/python3.11/site-packages/soma/__init__.py
soma_x ModuleNotFoundError("No module named 'soma_x'")
robotic_grounding ModuleNotFoundError("No module named 'robotic_grounding'")

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /opt/conda/lib/python3.11/site-packages/soma -maxdepth 3 -type f -printf '%p %s\\n' | head -100; grep -R \"class MHR\\|lod\\|SOMA_neutral\\|faces\" -n /opt/conda/lib/python3.11/site-packages/soma | head -100"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/soma/__init__.py 3115
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/__init__.cpython-311.pyc 3673
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/_smpl_family_loader.cpython-311.pyc 13953
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/_warp_utils.cpython-311.pyc 5442
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/assets.cpython-311.pyc 1370
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/correctives_model.cpython-311.pyc 21889
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/identity_model.cpython-311.pyc 36615
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/io.cpython-311.pyc 89215
/opt/conda/lib/python3.11/site-packages/soma/__pycache__/pose_inversion.cpython-311.pyc 98414
/opt/conda …[+17405 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /opt/conda/lib/python3.11/site-packages/soma/assets.py; python - <<'PY'
import soma
print(soma.__dict__.keys())
PY"
```
<details><summary>output</summary>

```
dict_keys(['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__path__', '__file__', '__cached__', '__builtins__', '__version__', 'Path', 'assets', 'get_assets_dir', 'geometry', 'remove_joint_orient_local', '_smpl_family_loader', 'units', 'identity_model', 'BaseIdentityModel', 'create_identity_model', 'io', 'SOMA_TEMPLATE_RIG_FILENAME', 'SOMA_XLO_TEMPLATE_RIG_FILENAME', 'add_npz_args', 'fan_triangulate', 'find_lod_skin_mesh_name', 'list_usd_meshes', 'load_lod_rig_from_usd', 'load_lod_rigs_from_usd', 'load_rig_from_usd', 'load_usd_animation', 'load_usd_mesh', 'load_usd_skeleton', 'load_usd_skinning', 'save_soma_npz', 'save_soma_usd', 'write_usd_mesh', '_warp_utils', 'pose_inversion', 'smpl', 'SMPLFamilyPoseTransferResult', 'SMPLFamilyTopologyBridge', 'SMPLLayer', 'SMPLXLayer', …[+227 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /weights/hf_home/hub -maxdepth 3 -printf '%y %p -> %l\\n' | head -200; du -sh /weights/hf_home/hub/* 2>/dev/null"
```
<details><summary>output</summary>

```
d /weights/hf_home/hub -> 
d /weights/hf_home/hub/models--Ruicheng--moge-2-vitl-normal -> 
d /weights/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/trees -> 
f /weights/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/trees/b135031bae30b5ac2ae141a0e68717795ce38340.json -> 
d /weights/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/blobs -> 
f /weights/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/blobs/a6344aac8c09253b3b630fb776ae94478aa0275b -> 
l /weights/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/blobs/280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01 -> ../../blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c
f /weights/hf_home/hub/models--Ruicheng--moge-2-vitl-normal/blobs/7be5fc7f47d5db027d120b8024982df93db95b74 -> 
d /weights/hf_hom …[+1266 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
try:
 import mhr
 print('mhr',mhr,mhr.__file__)
 from mhr.mhr import MHR
 print(MHR)
except Exception as e: print(type(e),e)
PY
find /opt/conda/lib/python3.11/site-packages -maxdepth 2 -iname '*mhr*' -printf '%p\\n' | head -50"
```
<details><summary>output</summary>

```
<class 'ModuleNotFoundError'> No module named 'mhr'
/opt/conda/lib/python3.11/site-packages/soma/pose_inversion_mhr.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,json
from scipy.spatial import cKDTree
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; z=torch.load('/work/scratch/frame0_sam3d_mesh.pt',map_location='cpu',weights_only=False);V=z['pred_vertices'][0].numpy();T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);pw=V@T[:3,:3].T+T[:3,3]
d,i=cKDTree(pw).query(h);print(h.shape,V.shape,'d',np.quantile(d,[0,.5,.9,.99,1]),'unique',len(np.unique(i)), 'mean',d.mean())
# reverse
d2,i2=cKDTree(h).query(pw);print('rev',np.quantile(d2,[0,.5,.9,.99,1]),'unique',len(np.unique(i2)))
print('first mapping',i[:100]);print('is monotonic?',np.mean(i[1:]>i[:-1]),'corr',np.corrcoef(np.arange(len(i)),i)[0,1])
PY"
```
<details><summary>output</summary>

```
(18056, 3) (18439, 3) d [0.00039822 0.02500573 0.05144376 0.06454397 0.07124366] unique 5335 mean 0.02693729096455587
rev [0.00039822 0.02975945 0.05383514 0.06586628 0.0701964 ] unique 5530
first mapping [6932 4902 4881 4882 2817 2819 2818 2822 2824 4894 2820 5579 3595 3596
 3495 5332 3336 4216 4215 5381 4217 4626 3705 3596 3814 3498 5341 5335
 3426 3709 3425 3421 3910 5227 4734 3497 4913 4914 3496 4768 5123 4156
 3410 5462 5227 3778 3334 4988 4216 4069 5638 3338 4930 5638 3507 3505
 4526 2816 4883 2855 5683 6934 6937 6939 6935 5715 5718 5746 6953 5709
 5699 5701 5702 5707 6950 6948 6946 5732 6969 5722 5726 6965 6960 5687
 5693 5695 6941 5711 5837 5690 5835 5697 2854 6944 2853 6943 5705  542
  539  540]
is monotonic? 0.41129880919412903 corr 0.26935544160116637

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/contact_opt.py <<'PY'
import numpy as np,json,trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));T0=np.array(c['T_world_cam']);pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=pl[:3];dw=pl[3]
# human lowest 100 vertices per frame under baseline
rec=np.load('/work/inputs/reconstruction.npz');H=rec['human_vertices_world'];nlow=120
inds=np.argsort(H@nw+dw,axis=1)[:,:nlow]
Hc=np.take_along_axis(H,inds[:,:,None],axis=1) # F,n,3
# object lowest 300 sampled mesh vertices per frame
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=mesh.vertices[::10].astype(np.float64);vm=np.c_[V,np.ones(len(V))]
Tw=rec['T_world_mesh'].astype(np.float64);O=np.einsum('fij,vj->fvi',Tw,vm)[:,:,:3]
nlow2=300; oi=np.argsort(O@nw+dw,axis=1)[:,:nlow2];Oc=np.take_along_axis(O,oi[:,:,None],axis=1)
# baseline softmins
def stats(P):
 sd=P@nw+dw; sm=-.004*np.log(np.exp(-sd/.004).mean(1));return sm
print('human sm q',np.quantile(stats(Hc),[0,.1,.5,.9,1]));print('obj sm q',np.quantile(stats(Oc),[0,.1,.5,.9,1]))
def pose(x):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];return R,t
def trans(P,x):
 R,t=pose(x);Q=R@T0[:3,:3].T;return (P-T0[:3,3])@Q.T+t
def sm(P,tau=.003):
 sd=P@nw+dw;return -tau*np.log(np.exp(-sd/tau).mean(1))
def fit(P,label,indsx=slice(None)):
 def fun(x): return sm(trans(P,x …[+1190 chars]
```
<details><summary>output</summary>

```
human sm q [-0.06221304 -0.03438319 -0.01416129 -0.00079731  0.01756287]
obj sm q [-0.04546    -0.02554267 -0.01089134  0.00476834  0.01710656]
human x [ 0.00872934 -0.31763751  0.0140808   0.08268628  0.04807075  0.00068792] rot deg 18.224026765163465 t cm 9.56466974378089 res [-4.53400074e-02 -1.59122395e-02  6.06015064e-05  1.21125606e-02
  3.26040215e-02] cost 0.045143246839438975
R [[ 0.95988417 -0.00985202 -0.2802237 ]
 [ 0.01789317  0.99949784  0.02615161]
 [ 0.27982534 -0.0301166   0.95957843]] t [0.1398385  0.05151748 0.03565532]
nc [-0.05151986 -0.99592392 -0.07403548] dc 1.3295862684910646
object x [-0.01307818  0.0069056  -0.01838629 -0.00119309 -0.04786126 -0.00247053] rot deg 1.3519639206611351 t cm 4.793982807555213 res [-0.03223762 -0.01444998 -0.0003019   0.01508706  0.025 …[+807 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
ns=[]
text='''[-0.01711,-0.99487,-0.09967]
[-0.0183,-0.99496,-0.09856]
[-0.01998,-0.99483,-0.09953]
[-0.01956,-0.99475,-0.10044]
[-0.01913,-0.99471,-0.10091]
[-0.01849,-0.99473,-0.10081]
[-0.01879,-0.99469,-0.10118]
[-0.01981,-0.99455,-0.10238]
[-0.01813,-0.99457,-0.10248]
[-0.02067,-0.99451,-0.10257]
[-0.01999,-0.9943,-0.10469]
[-0.01858,-0.99419,-0.10602]
[-0.0187,-0.99421,-0.10586]
[-0.02013,-0.9943,-0.10467]
[-0.02015,-0.99433,-0.10438]
[-0.01985,-0.99443,-0.1035]
[-0.01922,-0.99464,-0.10157]
[-0.0196,-0.99467,-0.10125]
[-0.02106,-0.99477,-0.0999]
[-0.01829,-0.99489,-0.09929]
[-0.02035,-0.99479,-0.09991]
[-0.02018,-0.9949,-0.09883]
[-0.02076,-0.99491,-0.09857]
[-0.0202,-0.99488,-0.09903]
[-0.02232,-0.99502,-0.09719]
[-0.02355,-0.99508,-0.09625]
[-0.02325,-0.99523,-0.0947]
[-0.02323,-0.9951,-0.09615]
[-0.02456,-0.99486,-0.09825]
[-0.02481,-0.9945,-0.10179]
[-0.0229,-0.99473,-0.09996]
[-0.02124,-0.99491,-0.09852]
[-0.0209,-0.99487,-0.09894]
[-0.02182,-0.99484,-0.09906]
[-0.02235,-0.99479,-0.09947]
[-0.02218,-0.99489,-0.0985]
[-0.02058,-0.99501,-0.09764]
[-0.02107,-0.99473,-0.10033]
[-0.02168,-0.9947,-0.10048]'''
for ln in text.splitlines(): ns.append(np.fromstring(ln.strip('[]'),sep=','))
ns=np.array(ns);ns/=np.linalg.norm(ns,axis=1,keepdims=True);med=np.median(ns,0);med/=np.linalg.norm(med)
angs=np.degrees(np.arccos(np.clip(ns@med,-1,1)));print(med,'angle q',np.quantile(angs,[0,.1,.5,.9,1]),'std',angs.std())
PY"
```
<details><summary>output</summary>

```
[-0.02020033 -0.99479125 -0.09991152] angle q [0.00857583 0.05046484 0.13254758 0.28498986 0.36372983] std 0.09593443136288789

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/dense_boot.py <<'PY'
import numpy as np,json
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam'])
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
d,idx=cKDTree(h).query(r);sel=d<.02
X=h[idx[sel]];R0=T0[:3,:3];t0=T0[:3,3]
def proj(x,X):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@R0;t=t0+x[3:];Xc=(X-t)@R;q=Xc@K.T;return q[:,:2]/q[:,2:3]
# target
Xr=r[sel]; Xc=(Xr-t0)@R0;q=Xc@K.T;target=q[:,:2]/q[:,2:3]
def fit(ii):
 def fun(x):return (proj(x,X[ii])-target[ii]).ravel()
 return least_squares(fun,np.zeros(6),loss='soft_l1',f_scale=2.,max_nfev=100).x
rng=np.random.default_rng(0);N=len(X)
xs=[]
for it in range(100):
 ii=rng.choice(N,size=1000,replace=True);xs.append(fit(ii))
xs=np.array(xs)
print('mean',xs.mean(0),'std',xs.std(0))
print('rot deg norm stats',np.quantile(np.degrees(np.linalg.norm(xs[:,:3],axis=1)),[.05,.5,.95]),'t cm norm',np.quantile(100*np.linalg.norm(xs[:,3:],axis=1),[.05,.5,.95]))
print('q rot components deg',np.degrees(np.quantile(xs[:,:3],[.05,.5,.95],axis=0)))
print('q trans cm',100*np.quantile(xs[:,3:],[.05,.5,.95],axis=0))
PY
python /work/scratch/dense_boot.py"
```
<details><summary>output</summary>

```
mean [-0.00896152  0.00911951 -0.00196009 -0.03039779 -0.02977484  0.00372017] std [0.0012944  0.00152679 0.00037336 0.00469801 0.0041317  0.00174336]
rot deg norm stats [0.61684795 0.74819417 0.90982729] t cm norm [3.55921318 4.31775236 5.17423842]
q rot components deg [[-0.63465014  0.39081171 -0.14603452]
 [-0.51329788  0.51071931 -0.11427321]
 [-0.38760215  0.69771741 -0.07582755]]
q trans cm [[-3.96191317 -3.69017829  0.07673018]
 [-2.98389206 -2.97762078  0.35768845]
 [-2.37965275 -2.29774485  0.64850997]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,torch
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];z=torch.load('/work/scratch/frame0_sam3d_mesh.pt',map_location='cpu',weights_only=False);V=z['pred_vertices'][0].numpy();F=z['faces'].numpy();F=F[(F<len(h)).all(1)]
nt=trimesh.Trimesh(h,F,process=False,validate=False).vertex_normals
nf=trimesh.Trimesh(V,F,process=False,validate=False).vertex_normals[:len(h)]
dot=(nt*nf).sum(1); print(np.quantile(dot,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]),dot.mean())
# distances same index
d=np.linalg.norm(h-V[:len(h)],axis=1);print('dist same index',d.mean(),np.median(d),np.quantile(d,[.9,.99,1]))
PY"
```
<details><summary>output</summary>

```
[-0.99993585 -0.98071075 -0.90361853 -0.80776381 -0.51145013  0.00608419
  0.5045626   0.80713045  0.90509518  0.98087769  0.99993282] 0.002064061402997537
dist same index 0.62219805 0.5876243 [0.99426728 1.62987311 1.76580369]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/pc_render.py <<'PY'
import numpy as np,cv2,json,h5py,torch,trimesh
from scipy.spatial import cKDTree
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T=np.array(c['T_world_cam']);h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
with h5py.File('/work/inputs/masks.h5','r') as f:m=f['person'][0].astype(bool)
def proj(T,X):
 Xc=(X-T[:3,3])@T[:3,:3];q=Xc@K.T;return q[:,:2]/q[:,2:3]
uv=proj(T,h)
print('uv bbox',uv.min(0),uv.max(0),uv.mean(0))
for r in [0,1,2,3,4,5]:
 pts=np.zeros(m.shape,np.uint8);ii=np.clip(np.round(uv).astype(int),0,[1535,1151]);pts[ii[:,1],ii[:,0]]=1
 if r: pts=cv2.dilate(pts,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
 iou=(pts.astype(bool)&m).sum()/(pts.astype(bool)|m).sum(); print('r',r,'iou',iou,'area',pts.sum(),m.sum())
# mapping full fresh->recon->true
z=torch.load('/work/scratch/frame0_sam3d_mesh.pt',map_location='cpu',weights_only=False);Vf=z['pred_vertices'][0].numpy();F=z['faces'].numpy();pf=Vf@T[:3,:3].T+T[:3,3]
r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
A=pf[::4]-pf[::4].mean(0);B=r-r.mean(0);U,S,Vt=np.linalg.svd(A.T@B);Rsim=U@Vt;s=S.sum()/np.sum(A*A);pal=(pf-pf[::4].mean(0))@Rsim*s+r.mean(0)
d,ii=cKDTree(h).query(pal); Xmap=h[ii]
# render mapped true with proper MHR faces
for name,V in [('mapped',Xmap),('fresh',pal)]:
 uv2=proj(T,V);mask=np.zeros(m.shape,np.uint8);cv2.fillPoly(mask,[np.round(uv2[F]).astype(np.int32).reshape(-1,2)] …[+262 chars]
```
<details><summary>output</summary>

```
uv bbox [490.50003354 423.73712439] [632.3193181  854.24767356] [555.66001886 621.69441361]
r 0 iou 0.17945911433392622 area 8326 43398
r 1 iou 0.4570131699572927 area 21764 43398
r 2 iou 0.7157085941946499 area 34979 43398
r 3 iou 0.7904090633918534 area 39412 43398
r 4 iou 0.8102080845983285 area 41508 43398
r 5 iou 0.8170634420791871 area 43156 43398
mapped mapping d 0.009169577990458254 iou 0.7741176470588236 area 36526
[ WARN:0@2.408] global loadsave.cpp:848 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
fresh mapping d 0.009169577990458254 iou 0.8073738478362755 area 37585

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/point_opt.py <<'PY'
import numpy as np,cv2,h5py,json,time
from scipy.optimize import differential_evolution,minimize
from scipy.spatial.transform import Rotation
W,H=1536,1152;scale=.5;W2,H2=int(W*scale),int(H*scale)
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam']);V=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
with h5py.File('/work/inputs/masks.h5','r') as f:m=f['person'][0]
m=cv2.resize(m.astype(np.uint8),(W2,H2),interpolation=cv2.INTER_NEAREST).astype(bool)
# choose radius full 4 => half 2 => kernel 5
ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
def pose(x):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];return R,t
def render(x):
 R,t=pose(x);Xc=(V-t)@R;q=Xc@K.T;uv=q[:,:2]/q[:,2:3]*scale
 im=np.zeros((H2,W2),np.uint8);ij=np.clip(np.round(uv).astype(np.int32),[0,0],[W2-1,H2-1]);im[ij[:,1],ij[:,0]]=1;im=cv2.dilate(im,ker)
 return im.astype(bool)
def iou(x):
 r=render(x);return (r&m).sum()/(r|m).sum()
print('base iou',iou(np.zeros(6)))
t=time.time()
for i in range(20):iou(np.random.normal(0,[.01,.01,.01,.03,.03,.03]))
print('20 eval',time.time()-t)
bounds=np.array([[-.03,.03],[-.03,.03],[-.03,.03],[-.15,.15],[-.15,.15],[-.2,.2]])
def obj(x):return 1-iou(x)
t=time.time();res=differential_evolution(obj,bounds,seed=0,popsize=8,maxiter=45,polish=False,tol=.001,atol=.0005,disp=True,workers=1);print('de',time.time()-t,res.x,res.fun,iou(r …[+455 chars]
```
<details><summary>output</summary>

```
base iou 0.8139475908706678
20 eval 0.03301501274108887
differential_evolution step 1: f(x)= 0.12222222222222223
differential_evolution step 2: f(x)= 0.12222222222222223
differential_evolution step 3: f(x)= 0.11272914168237103
differential_evolution step 4: f(x)= 0.11272914168237103
differential_evolution step 5: f(x)= 0.11272914168237103
differential_evolution step 6: f(x)= 0.11272914168237103
differential_evolution step 7: f(x)= 0.11272914168237103
differential_evolution step 8: f(x)= 0.11272914168237103
differential_evolution step 9: f(x)= 0.11272914168237103
differential_evolution step 10: f(x)= 0.10982311523269794
differential_evolution step 11: f(x)= 0.10982311523269794
differential_evolution step 12: f(x)= 0.10982311523269794
differential_evolution step 13: f(x)= 0.10982311523269794 …[+2467 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation
T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];dw=P[3];nobs=np.array([-.02020033,-.99479125,-.09991152]);
for f in ['point_x.npy','dense_x_0.03.npy']:
 x=np.load('/work/scratch/'+f);Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T[:3,:3];t=T[:3,3]+x[3:];nc=R.T@nw;print(f,'nc',nc,'dc',nw@t+dw,'floor angle',np.degrees(np.arccos(np.clip(nc@nobs,-1,1))))
PY"
```
<details><summary>output</summary>

```
point_x.npy nc [-0.0036935  -0.99593545 -0.08999413] dc 1.3638017478316544 floor angle 1.1053107576577679
dense_x_0.03.npy nc [-0.02126402 -0.99518353 -0.0956953 ] dc 1.3900780449628647 floor angle 0.2501771627334237

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/cue_compare.py <<'PY'
import numpy as np,json,cv2,h5py
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam']);P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];dw=P[3];nobs=np.array([-.02020033,-.99479125,-.09991152])
# dense points
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];rec=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
d,idx=cKDTree(h).query(rec);sel=d<.03;X=h[idx[sel]];Xr=rec[sel]
def projT(T,X):
 Xc=(X-T[:3,3])@T[:3,:3];q=Xc@K.T;return q[:,:2]/q[:,2:3]
uv=projT(T0,Xr)
xd=np.load('/work/scratch/dense_x_0.03.npy')
# floor rotation correction x: Rnew=R0 Rcorr where Rcorr maps nc0 to nobs
nc0=T0[:3,:3].T@nw; Rcorr=Rotation.align_vectors(nobs[None],nc0[None])[0].as_matrix();Rf=T0[:3,:3]@Rcorr;xfrot=Rotation.from_matrix(Rf@T0[:3,:3].T).as_rotvec()
print('floor xfrot deg',np.degrees(xfrot),np.degrees(np.linalg.norm(xfrot)))
# mask render pointcloud
with h5py.File('/work/inputs/masks.h5','r') as f: mm=f['person'][0]
sc=.5; ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5));mres=cv2.resize(mm.astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool)
def render(T):
 Xc=(h-T[:3,3])@T[:3,:3];q=Xc@K.T;uv=q[:,:2]/q[:,2:3]*sc;im=np.zeros((576,768),np.uint8);ii=np.clip(np.round(uv).astype(int),[0,0],[767,575]);im[ii[:,1],ii[:,0]]=1 …[+689 chars]
```
<details><summary>output</summary>

```
floor xfrot deg [ 0.43       -0.01316563  0.09047532] 0.43961246413689803
lam -0.25 rot 0.110 tcm 1.62 dense mean 3.742 med 3.600 p90 5.690 iou 0.7562 floor 0.330
lam -0.12 rot 0.055 tcm 1.62 dense mean 4.070 med 3.920 p90 6.040 iou 0.7538 floor 0.385
lam 0.00 rot 0.000 tcm 1.62 dense mean 4.529 med 4.393 p90 6.468 iou 0.7505 floor 0.440
lam 0.12 rot 0.055 tcm 1.62 dense mean 5.084 med 4.963 p90 7.027 iou 0.7474 floor 0.495
lam 0.25 rot 0.110 tcm 1.62 dense mean 5.708 med 5.603 p90 7.643 iou 0.7450 floor 0.550
lam 0.38 rot 0.165 tcm 1.62 dense mean 6.380 med 6.280 p90 8.316 iou 0.7411 floor 0.604
lam 0.50 rot 0.220 tcm 1.62 dense mean 7.086 med 6.992 p90 9.045 iou 0.7375 floor 0.659
lam 0.62 rot 0.275 tcm 1.62 dense mean 7.816 med 7.728 p90 9.765 iou 0.7333 floor 0.714
lam 0.75 rot 0.330 t …[+613 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/cue_compare.py';s=open(p).read();s=s.replace(\"x=np.zeros(6);x[:3]=lam*xfrot;x[3:]=xd[3:] # keep dense translation for now\", \"x=np.zeros(6);x[:3]=xd[:3]+lam*(xfrot-xd[:3]);x[3:]=xd[3:] # interpolate dense->floor rotation, keep dense translation\")
s=s.replace(\"for lam in np.linspace(-.25,1.5,15):\", \"for lam in np.linspace(-.25,1.5,15):\")
open(p,'w').write(s)
PY
python /work/scratch/cue_compare.py"
```
<details><summary>output</summary>

```
floor xfrot deg [ 0.43       -0.01316563  0.09047532] 0.43961246413689803
lam -0.25 rot 0.436 tcm 1.62 dense mean 2.977 med 2.870 p90 4.525 iou 0.8335 floor 0.093
lam -0.12 rot 0.359 tcm 1.62 dense mean 2.119 med 1.966 p90 3.561 iou 0.8249 floor 0.172
lam 0.00 rot 0.284 tcm 1.62 dense mean 1.707 med 1.565 p90 3.121 iou 0.8153 floor 0.250
lam 0.12 rot 0.216 tcm 1.62 dense mean 2.072 med 1.811 p90 3.740 iou 0.8044 floor 0.329
lam 0.25 rot 0.162 tcm 1.62 dense mean 2.937 med 2.750 p90 4.763 iou 0.7926 floor 0.407
lam 0.38 rot 0.140 tcm 1.62 dense mean 4.030 med 3.897 p90 5.904 iou 0.7788 floor 0.486
lam 0.50 rot 0.164 tcm 1.62 dense mean 5.208 med 5.096 p90 7.118 iou 0.7672 floor 0.565
lam 0.62 rot 0.219 tcm 1.62 dense mean 6.416 med 6.311 p90 8.337 iou 0.7560 floor 0.643
lam 0.75 rot 0.287 t …[+612 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation
z=np.load('/work/inputs/reconstruction.npz');T0=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);B=np.linalg.inv(T0)@z['T_world_mesh'].astype(np.float64)
for key in ['T_cam_mesh_coconet','T_cam_mesh_foundationpose']:
 A=z[key].astype(np.float64);D=np.linalg.inv(B)@A
 rot=np.degrees(np.linalg.norm(Rotation.from_matrix(D[:,:3,:3]).as_rotvec(),axis=1));tr=np.linalg.norm(D[:,:3,3],axis=1)
 print(key,'rot q',np.quantile(rot,[0,.5,.9,.99,1]),'t q',np.quantile(tr,[0,.5,.9,.99,1]),'mean t vectors',D[:,:3,3].mean(0))
 print('T positions mean',A[:,:3,3].mean(0),'B',B[:,:3,3].mean(0))
PY"
```
<details><summary>output</summary>

```
T_cam_mesh_coconet rot q [1.18100081 1.22638195 1.23986809 1.24257238 1.24274613] t q [0.05877738 0.10661391 0.12936445 0.1580501  0.15870024] mean t vectors [-0.01296916 -0.02100028 -0.07116966]
T positions mean [-0.09287316  0.64076766  2.84083455] B [-0.05947328  0.70902585  2.90405128]
T_cam_mesh_foundationpose rot q [0.13540683 1.72131547 3.73191463 4.02377237 4.27613007] t q [0.06081904 0.10640743 0.18512614 0.21192559 0.24158955] mean t vectors [-0.0080489   0.03695062 -0.04014439]
T positions mean [-0.10120751  0.66173752  2.91237259] B [-0.05947328  0.70902585  2.90405128]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/object_depth_opt.py <<'PY'
import numpy as np,json,h5py,cv2,torch,trimesh,time
import nvdiffrast.torch as dr
W,H=1536,1152;scale=.25;W2,H2=int(W*scale),int(H*scale)
z=np.load('/work/inputs/reconstruction.npz');T0=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'],np.float32);K=np.array(json.load(open('/work/inputs/camera.json'))['K'],np.float32)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.asarray(mesh.vertices,np.float32);F=np.asarray(mesh.faces,np.int32)
Tb=np.linalg.inv(T0)@z['T_world_mesh'].astype(np.float64)
print('mesh',V.shape,F.shape,'Tb',Tb[0])
with h5py.File('/work/inputs/masks.h5','r') as f: masks=np.stack([f['object'][i] for i in range(len(Tb))])
# target resize batch
targets=torch.tensor(np.stack([cv2.resize(m.astype(np.uint8),(W2,H2),interpolation=cv2.INTER_NEAREST) for m in masks]),device='cuda',dtype=torch.bool)
print('target area',targets.sum().item())
Vt=torch.tensor(V,device='cuda');Ft=torch.tensor(F,device='cuda',dtype=torch.int32)
ctx=dr.RasterizeCudaContext()
def render_batch(Ts,s):
 B=len(Ts);Rt=torch.tensor(Ts[:,:3,:3],device='cuda',dtype=torch.float32).transpose(1,2); tt=torch.tensor(Ts[:,:3,3],device='cuda',dtype=torch.float32);tt[:,2]*=s
 # Xcam=R*V+t ; rows V@R"'^T + t
 Xc=torch.bmm(Vt[None].expand(B,-1,-1),Rt)+tt[:,None,:]
 z=Xc[:,:,2];px=torch.stack([(K[0,0]*Xc[:,:,0]+K[0,2]*z)*scale,(K[1,1]*Xc[:,:,1]+K[1,2]*z)*scale],2)/z[:,:,None]
 pos=tor …[+1105 chars]
```
<details><summary>output</summary>

```
mesh (51247, 3) (95759, 3) Tb [[-0.03301754 -0.64875645  0.76027955 -0.26525589]
 [ 0.99542273 -0.08960503 -0.03323169  0.88206138]
 [ 0.08968419  0.75570242  0.6487455   2.71499803]
 [ 0.          0.          0.          1.        ]]
target area 971303
base 0.7527426 0.1695868968963623
s 0.94 mean 0.5697387 med 0.6171621 q10 0.419101282954216 time 0.12764286994934082
s 0.95 mean 0.6189752 med 0.66656756 q10 0.47259512841701506 time 0.12310576438903809
s 0.96 mean 0.67620605 med 0.71850985 q10 0.5469640851020813 time 0.1295607089996338
s 0.97 mean 0.73229057 med 0.7607887 q10 0.6278144359588623 time 0.11900138854980469
s 0.98 mean 0.7730468 med 0.77773285 q10 0.688002073764801 time 0.12360143661499023
s 0.99 mean 0.77983254 med 0.7879925 q10 0.6876907110214233 time 0.12350177764892578
s 1. …[+1115 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/dense_visible.py <<'PY'
import numpy as np,json,h5py,cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam'])
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
d,idx=cKDTree(h).query(r)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:b=bytes(f['aligned/front_stereo_camera_left'][0]);D=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
with h5py.File('/work/inputs/masks.h5','r') as f:mask=f['person'][0]
Rc=T0[:3,:3].T;tc=-Rc@T0[:3,3];Xc=r@Rc.T+tc;q=Xc@K.T;uv=q[:,:2]/q[:,2:3];z=Xc[:,2]
u=np.clip(np.round(uv[:,0]).astype(int),0,1535);v=np.clip(np.round(uv[:,1]).astype(int),0,1151);dep=D[v,u];inside=mask[v,u]
diff=np.abs(z-dep)
for th in [.01,.02,.03,.05,.08,.1]:
 sel=(d<.03)&inside&(diff<th)
 print('th',th,'n',sel.sum(),'dep diff q',np.quantile(diff[sel],[.5,.9,1]) if sel.any() else '')
# optimize for selected thresholds
for th in [.02,.03,.05,.08,.1]:
 sel=(d<.03)&inside&(diff<th);X=h[idx[sel]];Xr=r[sel];targ=uv[sel]
 def proj(x):
  Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];P=(X-t)@R;qq=P@K.T;return qq[:,:2]/qq[:,2:3]
 res=least_squares(lambda x:(proj(x)-targ).ravel(),np.zeros(6),loss='soft_l1',f_scale=2.,max_nfe …[+415 chars]
```
<details><summary>output</summary>

```
th 0.01 n 427 dep diff q [0.00506474 0.00909793 0.00996334]
th 0.02 n 893 dep diff q [0.0106378  0.01811545 0.01999685]
th 0.03 n 1257 dep diff q [0.014782   0.02642361 0.0299647 ]
th 0.05 n 2081 dep diff q [0.02365278 0.04531664 0.04999291]
th 0.08 n 2959 dep diff q [0.0361173  0.06702929 0.07998738]
th 0.1 n 3280 dep diff q [0.03944431 0.07969983 0.0998793 ]
opt th 0.02 n 893 x [-0.00900852  0.00570601 -0.00187203 -0.01994411 -0.03010712  0.00867187] rot 0.6203218057052649 tcm 3.7140374506549882 e 1.5966336096091998 1.4739308453692126 2.8741854193102885
opt th 0.03 n 1257 x [-0.00908912  0.00866166 -0.00165934 -0.02912857 -0.02993688  0.00594531] rot 0.7256232788977494 tcm 4.219048139447367 e 1.5815805179031355 1.4351774866382678 2.8416494449692395
opt th 0.05 n 2081 x [-0.00932187  0.00 …[+554 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation
T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];dw=P[3];nobs=np.array([-.02020033,-.99479125,-.09991152]);
for th in [.02,.03,.05,.08,.1]:
 x=np.load(f'/work/scratch/dense_vis_x_{th}.npy');R=Rotation.from_rotvec(x[:3]).as_matrix()@T[:3,:3];t=T[:3,3]+x[3:];nc=R.T@nw;print(th,'floor',np.degrees(np.arccos(np.clip(nc@nobs,-1,1))),'dc',nw@t+dw)
PY"
```
<details><summary>output</summary>

```
0.02 floor 0.09841803975579991 dc 1.4091261543800515
0.03 floor 0.10493278447920955 dc 1.4092809712136327
0.05 floor 0.12528425805214938 dc 1.410450364031026
0.08 floor 0.03350450619255584 dc 1.4019889621222474
0.1 floor 0.011365563805644335 dc 1.4040826088092884

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_candidates.py <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));T0=np.array(c['T_world_cam']);P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];dw=P[3]
r=np.load('/work/inputs/reconstruction.npz');H=r['human_vertices_world'];Tw=r['T_world_mesh'];mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[mesh.vertices[::10],np.ones(len(mesh.vertices[::10]))];O=np.einsum('fij,vj->fvi',Tw.astype(np.float64),V)[:,:,:3]
def make(x):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];return R,t
def transform(A,R,t):
 Q=R@T0[:3,:3].T;return (A-T0[:3,3])@Q.T+t
cands={'base':np.zeros(6),'dense_all':np.load('/work/scratch/dense_x_0.03.npy')}
for th in ['.02','.03','.05','.08','.1']:cands['vis'+th]=np.load('/work/scratch/dense_vis_x_'+th+'.npy')
cands['point']=np.load('/work/scratch/point_x.npy');cands['objcontact']=np.load('/work/scratch/contact_x_object.npy')
nobs=np.array([-.02020033,-.99479125,-.09991152]);
for name,x in cands.items():
 R,t=make(x);nc=R.T@nw
 Hw=transform(H,R,t);sd=Hw@nw+dw;hs=np.quantile(sd,[.001,.005,.01],axis=1)
 Ow=transform(O,R,t);os=Ow@nw+dw
 print('%-12s rot=%5.2f tcm=%5.2f floorang=%5.2f dc=%5.3f Hmin med=%7.4f q10=%7.4f Omin med=%7.4f q10=%7.4f'%(name,np.degrees(np.linalg.norm(x[:3])),100*np.linalg.norm(x[3:]),np.degrees(np.arccos(np.clip(nc@ …[+152 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_candidates.py", line 10, in <module>
    for th in ['.02','.03','.05','.08','.1']:cands['vis'+th]=np.load('/work/scratch/dense_vis_x_'+th+'.npy')
                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/dense_vis_x_.02.npy'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/\\['.02','.03','.05','.08','.1'\\]/['0.02','0.03','0.05','0.08','0.1']/\" /work/scratch/eval_candidates.py; python /work/scratch/eval_candidates.py"
```
<details><summary>output</summary>

```
base         rot= 0.00 tcm= 0.00 floorang= 0.44 dc=1.379 Hmin med=-0.0181 q10=-0.0384 Omin med=-0.0239 q10=-0.0385
dense_all    rot= 0.28 tcm= 1.62 floorang= 0.25 dc=1.390 Hmin med=-0.0158 q10=-0.0384 Omin med=-0.0224 q10=-0.0373
vis0.02      rot= 0.62 tcm= 3.71 floorang= 0.10 dc=1.409 Hmin med=-0.0132 q10=-0.0389 Omin med=-0.0205 q10=-0.0359
vis0.03      rot= 0.73 tcm= 4.22 floorang= 0.10 dc=1.409 Hmin med=-0.0134 q10=-0.0392 Omin med=-0.0208 q10=-0.0361
vis0.05      rot= 0.68 tcm= 3.97 floorang= 0.13 dc=1.410 Hmin med=-0.0130 q10=-0.0387 Omin med=-0.0200 q10=-0.0354
vis0.08      rot= 0.51 tcm= 2.91 floorang= 0.03 dc=1.402 Hmin med=-0.0141 q10=-0.0386 Omin med=-0.0213 q10=-0.0363
vis0.1       rot= 0.59 tcm= 3.35 floorang= 0.01 dc=1.404 Hmin med=-0.0140 q10=-0.0391 Omin med=-0.0214 q10=-0. …[+235 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=P[:3],P[3]
w=z['human_vertices_world']
# rot vel and trans vel
vel=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1); rv=np.degrees(np.linalg.norm(Rotation.from_matrix(T[1:,:3,:3]@T[:-1,:3,:3].transpose(0,2,1)).as_rotvec(),axis=1))
for st in range(0,778,25):print(st,'vmax',vel[max(0,st-25):st].max() if st else 0,'mean',vel[max(0,st-25):st].mean() if st else 0,'rv',rv[max(0,st-25):st].max() if st else 0)
# static segments low vel
low=(vel<0.002)&(rv<0.3)
print('low count',low.sum())
# runs
idx=np.nonzero(low)[0];runs=[]
for i in idx:
 if not runs or i>runs[-1][-1]+1:runs.append([i])
 else:runs[-1].append(i)
print('runs',[(r[0],r[-1]+1,len(r)) for r in runs if len(r)>5][:100])
PY"
```
<details><summary>output</summary>

```
0 vmax 0 mean 0 rv 0
25 vmax 0.04248773 mean 0.010075694 rv 0.643197943180661
50 vmax 0.020825429 mean 0.005502732 rv 0.7647790240351197
75 vmax 0.031065343 mean 0.01020553 rv 0.709607326177328
100 vmax 0.051990096 mean 0.0076624495 rv 0.92049541265125
125 vmax 0.00929842 mean 0.0044004456 rv 0.45714057789256984
150 vmax 0.015958913 mean 0.006191969 rv 0.7890186495469581
175 vmax 0.019026147 mean 0.006731056 rv 0.5099944543844823
200 vmax 0.016173813 mean 0.006649284 rv 0.8418795726862216
225 vmax 0.013320224 mean 0.0058699804 rv 1.1381754876600043
250 vmax 0.0153129315 mean 0.0045727207 rv 0.7807412276143838
275 vmax 0.010943506 mean 0.004500729 rv 0.9972670079772004
300 vmax 0.01585851 mean 0.007369509 rv 1.025668732874548
325 vmax 0.017901165 mean 0.008492628 rv 0.9555516182544302
350 v …[+1083 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];P=T[:,:3,3]; print('start',P[0]); print('end',P[-1]);
for i in range(0,778,20):
 dt=np.linalg.norm(P[i]-P[0]); rel=np.degrees(np.linalg.norm(Rotation.from_matrix(T[i,:3,:3]@T[0,:3,:3].T).as_rotvec())); print(i,dt,rel,P[i])
PY"
```
<details><summary>output</summary>

```
start [-0.12149151  0.98271966  2.723595  ]
end [0.45725334 0.69993377 3.2301097 ]
0 0.0 1.9485116348881757e-15 [-0.12149151  0.98271966  2.723595  ]
20 0.00318134 0.9820119138273685 [-0.11938911  0.98128974  2.7216828 ]
40 0.0056814975 1.4295118528167245 [-0.12039088  0.9847207   2.7183926 ]
60 0.01252014 1.3375138704721241 [-0.11962337  0.989029    2.7342465 ]
80 0.027629267 1.5349562001998633 [-0.11167256  0.97472584  2.6990376 ]
100 0.04052157 1.2124078894894417 [-0.11252268  0.99662554  2.7605839 ]
120 0.05459528 1.1755886801333646 [-0.11784859  0.9995164   2.7754142 ]
140 0.091753155 1.6130913220930183 [-0.10446329  1.0110406   2.8091905 ]
160 0.08210661 1.4495687054748319 [-0.11121479  1.00669     2.8014493 ]
180 0.06810528 1.6327164786738226 [-0.10958954  1.0040684   2.787163  ]
20 …[+1994 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/object_static_contact.py <<'PY'
import numpy as np,json,trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));T0=np.array(c['T_world_cam']);P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];dw=P[3]
z=np.load('/work/inputs/reconstruction.npz');Tw=z['T_world_mesh'].astype(np.float64);mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); selv=np.arange(0,len(mesh.vertices),5);V=np.c_[mesh.vertices[selv],np.ones(len(selv))];O=np.einsum('fij,vj->fvi',Tw,V)[:,:,:3]
frames=np.r_[np.arange(0,281,5),np.arange(620,778,5)]
low=80;ii=np.argsort(O[frames]@nw+dw,axis=1)[:,:low];Os=np.take_along_axis(O[frames],ii[:,:,None],axis=1)
def sm(P,tau=.003):sd=P@nw+dw;return -tau*np.log(np.exp(-sd/tau).mean(1))
def trans(A,x):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];Q=R@T0[:3,:3].T;return (A-T0[:3,3])@Q.T+t
print('base sm q',np.quantile(sm(Os),[0,.1,.5,.9,1]))
for loss in ['linear','huber','soft_l1']:
 res=least_squares(lambda x:sm(trans(Os,x)),np.zeros(6),loss=loss,f_scale=.01,max_nfev=1000)
 x=res.x;Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];Rch=R@T0[:3,:3].T
 print(loss,x,'rot',np.degrees(np.linalg.norm(Rotation.from_matrix(Rch).as_rotvec())),'tcm',100*np.linalg.norm(x[3:]),'after',np.quantile(sm(trans(Os,x)),[0,.1,.5,.9,1]),'nc',R.T@nw,'d …[+112 chars]
```
<details><summary>output</summary>

```
base sm q [-0.05044743 -0.03751972 -0.02105425 -0.01040914 -0.00759861]
linear [-0.00577705  0.43937043  0.00654695 -0.06283278 -0.04419064 -0.01851096] rot 25.179041345548473 tcm 7.9015355692986535 after [-0.02555951 -0.01427153  0.00168019  0.01296416  0.01779322] nc [-0.00141531 -0.99483008 -0.10154367] dc 1.4255035599070136
huber [-0.00252751  0.29514935  0.00071236  0.00389383 -0.03387677 -0.00702333] rot 16.91148152504856 tcm 3.481557720025047 after [-0.02724036 -0.01445999  0.00046672  0.01149307  0.015665  ] nc [-0.00524199 -0.99521117 -0.09760764] dc 1.4133094544581581
soft_l1 [-0.00767511 -0.11431837 -0.02459729 -0.70948774 -0.0357939   0.22976715] rot 6.714279287375757 tcm 74.66237304380195 after [-0.02700183 -0.01451407  0.00035847  0.01168885  0.01599679] nc [-0.00397465 -0.99 …[+42 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/add_contact_height.py <<'PY'
import numpy as np,json,trimesh
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam']);P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];dw=P[3]
z=np.load('/work/inputs/reconstruction.npz');Tw=z['T_world_mesh'].astype(np.float64);mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();selv=np.arange(0,len(mesh.vertices),5);V=np.c_[mesh.vertices[selv],np.ones(len(selv))];O=np.einsum('fij,vj->fvi',Tw,V)[:,:,:3]
frames=np.r_[np.arange(0,281,5),np.arange(620,778,5)];ii=np.argsort(O[frames]@nw+dw,axis=1)[:,:80];Os=np.take_along_axis(O[frames],ii[:,:,None],axis=1)
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];rec=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0];d,hi=cKDTree(h).query(rec);seld=d<.03;X=h[hi[seld]];Xr=rec[seld];uv=( (Xr-T0[:3,3])@T0[:3,:3])@K.T;uv=uv[:,:2]/uv[:,2:3]
def make(x):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];return R,t
def XW(A,x,s=1):
 R,t=make(x);B=np.linalg.inv(T0)@Tw[frames] # not used
 # object points: first take baseline world to baseline cam, scale t, new cam
 Rcw=T0[:3,:3].T;Tcw=-Rcw@T0[:3,3];Ac=(A-T0[:3,3])@T0[:3,:3] # row Rcw? Xcam=(A-t0)@R0? check: (A-t)@R0 = R0"'^T(A-t)
 # baseline Xca …[+1077 chars]
```
<details><summary>output</summary>

```
scale 1
Traceback (most recent call last):
  File "/work/scratch/add_contact_height.py", line 29, in <module>
    alpha=brentq(f,-.05,.08);x=x0.copy();x[3:]+=alpha*nw
          ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_zeros_py.py", line 846, in brentq
    r = _zeros._brentq(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_zeros_py.py", line 94, in f_raise
    fx = f(x, *args)
         ^^^^^^^^^^^
  File "/work/scratch/add_contact_height.py", line 28, in <lambda>
    f=lambda a: objmed(x0+a*nw,s)
                       ~~^~~~~
ValueError: operands could not be broadcast together with shapes (6,) (3,) 

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/add_contact_height.py';s=open(p).read().replace(\"f=lambda a: objmed(x0+a*nw,s)\", \"f=lambda a: objmed(np.r_[x0[:3],x0[3:]+a*nw],s)\")
open(p,'w').write(s)
PY
python /work/scratch/add_contact_height.py"
```
<details><summary>output</summary>

```
scale 1
vis0.08 alpha 0.018386474329138452 rot 0.505362632014268 tcm 4.497149507930884 objmed -4.463096558993096e-17 dense 5.028938727767883 5.019598406820402 6.769423164208915 R [[ 0.99921268 -0.00514919  0.03933845]
 [ 0.00344375  0.99905625  0.04329837]
 [-0.03952428 -0.04312881  0.9982874 ]] t [ 0.03896251 -0.03761285  0.03735086] dc 1.4203754364513859
Traceback (most recent call last):
  File "/work/scratch/add_contact_height.py", line 27, in <module>
    x0=np.load('/work/scratch/dense_vis_x_'+nm[-4:]+'.npy')
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundErr …[+78 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/add_contact_height.py';s=open(p).read().replace(\"for nm in ['vis0.08','vis0.1']:\\n  x0=np.load('/work/scratch/dense_vis_x_'+nm[-4:]+'.npy')\", \"for nm,th in [('vis0.08','.08'),('vis0.1','.1')]:\\n  x0=np.load('/work/scratch/dense_vis_x_'+th+'.npy')\")
open(p,'w').write(s)
PY
python /work/scratch/add_contact_height.py"
```
<details><summary>output</summary>

```
scale 1
Traceback (most recent call last):
  File "/work/scratch/add_contact_height.py", line 27, in <module>
    x0=np.load('/work/scratch/dense_vis_x_'+th+'.npy')
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/dense_vis_x_.08.npy'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/('vis0.08','.08'),('vis0.1','.1')/('vis0.08','0.08'),('vis0.1','0.1')/\" /work/scratch/add_contact_height.py; python /work/scratch/add_contact_height.py"
```
<details><summary>output</summary>

```
scale 1
vis0.08 alpha 0.018386474329138452 rot 0.505362632014268 tcm 4.497149507930884 objmed -4.463096558993096e-17 dense 5.028938727767883 5.019598406820402 6.769423164208915 R [[ 0.99921268 -0.00514919  0.03933845]
 [ 0.00344375  0.99905625  0.04329837]
 [-0.03952428 -0.04312881  0.9982874 ]] t [ 0.03896251 -0.03761285  0.03735086] dc 1.4203754364513859
vis0.1 alpha 0.01848388965311451 rot 0.5872759250635737 tcm 4.88307048136957 objmed 2.6645352591003876e-17 dense 5.019004539132093 5.016336000403623 6.754018243657734 R [[ 0.99915533 -0.00528289  0.04075196]
 [ 0.00348746  0.9990253   0.04400331]
 [-0.04094471 -0.04382402  0.99819987]] t [ 0.03442855 -0.03971518  0.03723057] dc 1.422566498462403
scale 0.9875
vis0.08 alpha 0.0006183236798018785 rot 0.505362632014268 tcm 2.9563971597363756 …[+631 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh']
r0=np.degrees(np.linalg.norm(Rotation.from_matrix(T[:,:3,:3]@T[0,:3,:3].T).as_rotvec(),axis=1));r1=np.degrees(np.linalg.norm(Rotation.from_matrix(T[:,:3,:3]@T[-1,:3,:3].T).as_rotvec(),axis=1))
for th in [3,5,8,10]:
 m=(r0<th)|(r1<th);idx=np.nonzero(m)[0]; print(th,m.sum(),idx.min(),idx.max())
 print('prefix max',np.max(np.nonzero(r0<th)[0]),'suffix min',np.min(np.nonzero(r1<th)[0]))
PY"
```
<details><summary>output</summary>

```
3 460 0 777
prefix max 290 suffix min 609
5 473 0 777
prefix max 294 suffix min 600
8 488 0 777
prefix max 301 suffix min 592
10 506 0 777
prefix max 311 suffix min 584

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/scale=.25/scale=.5/' /work/scratch/object_depth_opt.py | sed \"s/np.linspace(.94,1.06,13)/np.array([0.980,0.9825,0.985,0.9875,0.99,0.9925,0.995])/\" | sed \"s/np.linspace(best\\[0\\]-.01,best\\[0\\]+.01,9)/np.array([best[0]-.0025,best[0],best[0]+.0025])/\" | sed 's#object_scale_scan.npy#object_scale_scan_half.npy#' > /work/scratch/object_depth_half.py
python /work/scratch/object_depth_half.py"
```
<details><summary>output</summary>

```
mesh (51247, 3) (95759, 3) Tb [[-0.03301754 -0.64875645  0.76027955 -0.26525589]
 [ 0.99542273 -0.08960503 -0.03323169  0.88206138]
 [ 0.08968419  0.75570242  0.6487455   2.71499803]
 [ 0.          0.          0.          1.        ]]
target area 3874290
base 0.75053394 0.21645164489746094
s 0.98 mean 0.748316 med 0.7530595 q10 0.6596210956573486 time 0.15804290771484375
s 0.9825 mean 0.7566215 med 0.75741696 q10 0.6676050424575806 time 0.1563858985900879
s 0.985 mean 0.76286393 med 0.76519185 q10 0.673650997877121 time 0.1560981273651123
s 0.9875 mean 0.7663708 med 0.77080405 q10 0.6740582883358002 time 0.15636444091796875
s 0.99 mean 0.7671609 med 0.77451956 q10 0.6740931570529938 time 0.2705669403076172
s 0.9925 mean 0.7655848 med 0.77544206 q10 0.673341166973114 time 0.3411941528320312 …[+616 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('/work/scratch/object_scale_scan_half.npy');a=a[np.unique(a[:,0],return_index=True)[1]];i=np.argsort(a[:,0]);a=a[i];print(a)
# quadratic fits local
for c in [[.9875,.99,.9925],[.985,.9875,.99,.9925,.995]]:
 m=np.isin(a[:,0],c);coef=np.polyfit(a[m,0],a[m,1],2);x=-coef[1]/(2*coef[0]);print(c,'peak',x,np.polyval(coef,x),coef)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/joint_opt_test.py <<'PY'
import numpy as np,json,h5py,cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam']);P=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nw=P[:3];nobs=np.array([-.02020033,-.99479125,-.09991152]);nobs/=np.linalg.norm(nobs)
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
d,idx=cKDTree(h).query(r);sel=d<.03;X=h[idx[sel]];Xr=r[sel]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: b=bytes(f['aligned/front_stereo_camera_left'][0]);D=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
with h5py.File('/work/inputs/masks.h5','r') as f:mask=f['person'][0]
Xc=(Xr-T0[:3,3])@T0[:3,:3];q=Xc@K.T;uv=Xc.newbyteorder() if False else None;uv=q[:,:2]/q[:,2:3];z=Xc[:,2];u=np.clip(np.round(uv[:,0]).astype(int),0,1535);v=np.clip(np.round(uv[:,1]).astype(int),0,1151);sel2=mask[v,u]&(np.abs(z-D[v,u])<.08)&(d[sel]<.03);X=X[sel2];uv=uv[sel2];print('used',len(X))
def pose(x):
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];return R,t
def proj(x):
 R,t=pose(x);Pp=(X-t)@R;qq=Pp@K.T;return qq[:,:2]/qq[:,2:3]
for w in [0,10,30,100,300,1000,3000]:
 def fun(x):
  nc=pose(x)[0].T@nw
  return np.r_[(proj( …[+526 chars]
```
<details><summary>output</summary>

```
used 2959
w 0 x [-0.00683759  0.005401   -0.00136863 -0.0178423  -0.02270469  0.00340327] rot 0.5053624520347584 tcm 2.9076322604301095 floorang 0.0333198506030173 e 1.6076873087377639 1.4531171734522692 2.901413903102323 cost 1043.085653367783
w 10 x [-0.00683775  0.00540101 -0.00136866 -0.01784238 -0.02270518  0.00340343] rot 0.5053700265062478 tcm 2.907677726719692 floorang 0.03331143251029842 e 1.6076874874552383 1.453115487756052 2.901413148729477 cost 1043.0856702727344
w 30 x [-0.00683895  0.00540114 -0.00136886 -0.01784304 -0.02270906  0.00340471] rot 0.50542971003057 tcm 2.9080358406026887 floorang 0.033245219340929166 e 1.6076888919653762 1.453110982744514 2.90140717453639 cost 1043.0858052001013
w 100 x [-0.00685228  0.0054025  -0.00137109 -0.01785027 -0.02275192  0.00341889] ro …[+876 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# adapt add_contact_height to joint x
p='/work/scratch/add_contact_height.py';s=open(p).read()
s=s.replace(\"for nm,th in [('vis0.08','0.08'),('vis0.1','0.1')]:\\n  x0=np.load('/work/scratch/dense_vis_x_'+th+'.npy')\", \"for nm,th in [('joint1000','1000')]:\\n  x0=np.load('/work/scratch/joint_x_'+th+'.npy')\")
open('/work/scratch/add_contact_joint.py','w').write(s)
PY
python /work/scratch/add_contact_joint.py"
```
<details><summary>output</summary>

```
scale 1
joint1000 alpha 0.018228810953841594 rot 0.5261771604801778 tcm 4.613263021258371 objmed -4.9960036108131626e-17 dense 4.989229381504606 4.983477513533426 6.725293329801739 R [[ 0.99920986 -0.00510261  0.03941599]
 [ 0.0033782   0.99903933  0.04369227]
 [-0.03960107 -0.04352459  0.99826718]] t [ 0.03864778 -0.03872023  0.03775178] dc 1.421464624676199
scale 0.9875
joint1000 alpha 0.0004468792452596085 rot 0.5261771604801778 tcm 3.066645105252661 objmed 1.1923795284474419e-16 dense 1.7055457813738455 1.5569080848873313 3.0762557185466712 R [[ 0.99920986 -0.00510261  0.03941599]
 [ 0.0033782   0.99903933  0.04369227]
 [-0.03960107 -0.04352459  0.99826718]] t [ 0.03898377 -0.02096885  0.03873807] dc 1.403682692967617

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/alpha_validate.py <<'PY'
import numpy as np,cv2,h5py,json
from scipy.spatial import Delaunay,cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam']);V=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
with h5py.File('/work/inputs/masks.h5','r') as f:m=f['person'][0].astype(bool)
for name,x in [('base',np.zeros(6)),('ours',np.load('/work/scratch/joint_x_1000.npy'))]:
 Rr=Rotation.from_rotvec(x[:3]).as_matrix();R=Rr@T0[:3,:3];t=T0[:3,3]+x[3:];Xc=(V-t)@R;q=Xc@K.T;uv=q[:,:2]/q[:,2:3]
 # remove extreme clipped? all inside
 tree=cKDTree(uv);dd,ii=tree.query(uv,k=7);med=np.median(dd[:,1:],axis=1);print(name,'nn q',np.quantile(med,[.5,.9,.95,.99]))
 tri=Delaunay(uv);P=uv[tri.simplices];edges=np.linalg.norm(P-np.roll(P,1,axis=1),axis=2).max(1);areas=np.abs(np.cross(P[:,1]-P[:,0],P[:,2]-P[:,0]))/2
 for th in [8,10,12,15,20]:
  good=(edges<th)&(areas>0.1);mask=np.zeros(m.shape,np.uint8);polys=np.round(P[good]).astype(np.int32); 
  # fill all triangles one call
  cv2.fillPoly(mask,list(polys),1);iou=(mask.astype(bool)&m).sum()/(mask.astype(bool)|m).sum()
  e1=cv2.Canny(mask,0,1)>0;e2=cv2.Canny(m.astype(np.uint8),0,1)>0;d1=distance_transform_edt(~e1);d2=distance_transform_edt(~e2)
  print(' th',th,'tri',good.sum(),'area',mask.sum(),'iou',iou,'bd',np.median(d2[e1]),np.median(d1[e2]))
PY
python / …[+31 chars]
```
<details><summary>output</summary>

```
base nn q [0.65308459 2.84886918 3.67672558 5.27311697]
 th 8 tri 20554 area 35038 iou 0.7455824097565318 bd 9.055385138137417 6.082762530298219
 th 10 tri 20660 area 36280 iou 0.7732229492144034 bd 6.324555320336759 6.324555320336759
 th 12 tri 20676 area 36403 iou 0.775683674150553 bd 6.082762530298219 6.324555320336759
 th 15 tri 20704 area 36707 iou 0.7818533677373432 bd 6.0 6.324555320336759
 th 20 tri 20749 area 37366 iou 0.7899029298346705 bd 5.830951894845301 6.082762530298219
ours nn q [0.65187297 2.86321765 3.6999133  5.28174109]
 th 8 tri 20596 area 35108 iou 0.7497102612106624 bd 8.54400374531753 6.0
 th 10 tri 20705 area 36349 iou 0.7769731271447035 bd 6.0 6.082762530298219
 th 12 tri 20722 area 36474 iou 0.7797584562591916 bd 6.0 6.082762530298219
 th 15 tri 20749 area 36755  …[+110 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/add_contact_joint.py';s=open(p).read().replace('for s in [1,.9875]:','for s in [0.9897]:').replace(\"for nm,th in [('joint1000','1000')]:\\n  x0=np.load('/work/scratch/joint_x_'+th+'.npy')\", \"for nm,th in [('joint1000','1000')]:\\n  x0=np.load('/work/scratch/joint_x_'+th+'.npy')\")
open('/work/scratch/add_contact_final.py','w').write(s)
PY
python /work/scratch/add_contact_final.py"
```
<details><summary>output</summary>

```
scale 0.9897
joint1000 alpha 0.0035764734645511165 rot 0.5261771604801778 tcm 3.3220842528208054 objmed -5.662137425588245e-17 dense 1.9205070624616878 1.7245270311298655 3.419148013085772 R [[ 0.99920986 -0.00510261  0.03941599]
 [ 0.0033782   0.99903933  0.04369227]
 [-0.03960107 -0.04352459  0.99826718]] t [ 0.03892463 -0.02409307  0.03856449] dc 1.4068122871869084

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# load final x from script via recompute? adapt load not saved maybe check file
import os
print(os.path.exists('/work/scratch/cand_joint1000_s0.9897.npy'))
PY"
```
<details><summary>output</summary>

```
True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs && cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Deterministic camera/world-pose solver for the single monocular HOI episode.

The camera is refined from three independent, static-scene cues:
  * dense photo-aligned frame-0 human correspondences (world mesh + person mask),
  * the released world floor plane fitted in the MoGe camera depth maps,
  * resting-object floor contact and metric object image size.

Only the allowed inputs, the released CARI4D intermediates in /pipeline, and the
preinstalled scientific stack are used.  No network access is required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.optimize import brentq, least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# Determinism.  These data are CPU/GPU rasterization driven and do not train.
SEED = 20261020
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
np.random.seed(SEED)
torch.manual_seed(SEED)

# Image/rasterization choices.  Half resolution is enough to estimate a global
# depth scale (the object spans >1000 mask pixels over the episode) and keeps
# the whole solve comfortably below the 8 GB GPU budget.
OBJ_RENDER_SCALE = 0.5
OBJ_BATCH = 16
# The  …[+33221 chars]
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
[1/5] fitting released floor plane in camera depth
[2/5] optimizing camera on visible dense frame-0 human correspondences
[3/5] scanning global object depth scale against exact mesh/masks
[4/5] enforcing resting-object floor contact (height direction only)
Traceback (most recent call last):
  File "/work/outputs/solve.py", line 755, in <module>
    main()
  File "/work/outputs/solve.py", line 578, in main
    x_final, contact_info = object_contact_height_adjustment(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 375, in object_contact_height_adjustment
    Rotation.from_matrix(T[:, :3, :3] @ T[0, :3, :3].T).as_rotvec(axis=1),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "scipy/spatial/transform/_rotati …[+141 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep -n "as_rotvec(axis" /work/outputs/solve.py'
```
<details><summary>output</summary>

```
375:            Rotation.from_matrix(T[:, :3, :3] @ T[0, :3, :3].T).as_rotvec(axis=1),
381:            Rotation.from_matrix(T[:, :3, :3] @ T[-1, :3, :3].T).as_rotvec(axis=1),
603:            Rotation.from_matrix(T_world_mesh[:, :3, :3] @ T_world_mesh[0, :3, :3].T).as_rotvec(axis=1),
609:            Rotation.from_matrix(T_world_mesh[:, :3, :3] @ T_world_mesh[-1, :3, :3].T).as_rotvec(axis=1),

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace('.as_rotvec(axis=1),','.as_rotvec(),');s=s.replace(\"u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, K.shape[1] * 0 + 1535)\",\"u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, person_mask0.shape[1] - 1)\");s=s.replace(\"v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, 1151)\",\"v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, person_mask0.shape[0] - 1)\");open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
[1/5] fitting released floor plane in camera depth
[2/5] optimizing camera on visible dense frame-0 human correspondences
[3/5] scanning global object depth scale against exact mesh/masks
[4/5] enforcing resting-object floor contact (height direction only)
[5/5] validation and report
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/camera_estimate.json
Wrote /work/outputs/REPORT.md
Wrote /work/outputs/TOOLS.md
camera delta: 0.526 deg, 3.31 cm; object depth scale 0.9896; runtime 49.8s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/outputs/camera_estimate.json; echo '--- REPORT ---'; cat /work/outputs/REPORT.md; echo '--- TOOLS ---'; cat /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
{
  "K": [
    [
      819.9987182617188,
      0.0,
      758.0999755859375
    ],
    [
      0.0,
      819.9987182617188,
      560.5000610351562
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "T_world_cam": [
    [
      0.9992098310536693,
      -0.0051028667435672305,
      0.039416675109597975,
      0.03892642847269065
    ],
    [
      0.0033784161880157535,
      0.9990393091777499,
      0.04369261976244162,
      -0.023887467236247576
    ],
    [
      -0.03960176548790165,
      -0.04352492927785347,
      0.9982671389470854,
      0.03857575484148066
    ],
    [
      0.0,
      0.0,
      0.0,
      1.0
    ]
  ],
  "width": 1536,
  "height": 1152,
  "convention": "OpenCV camera x-right, y-down, z-forward; world y-down",
  "camera_change_from_baseline": {
     …[+5668 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
p='/work/outputs/object_poses_world.npz';z=np.load(p)
print(z.files)
A=z['T_world_mesh'];print(A.shape,A.dtype,'finite',np.isfinite(A).all())
print('first\\n',A[0]);print('last\\n',A[-1])
print('translation min/max/mean',A[:,:3,3].min(0),A[:,:3,3].max(0),A[:,:3,3].mean(0))
print('rotation orth err',np.abs(np.einsum('fij,fkj->fik',A[:,:3,:3],A[:,:3,:3])-np.eye(3)).max(),'det',np.linalg.det(A[:,:3,:3]).min(),np.linalg.det(A[:,:3,:3]).max())
# compare baseline metrics / difference
b=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];
d=np.linalg.norm(A-b,axis=2); print('translation change cm q',np.quantile(100*np.linalg.norm(A[:,:3,3]-b[:,:3,3],axis=1),[0,.5,1]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /work/outputs/solve.py /work/outputs/object_poses_world.npz /work/outputs/camera_estimate.json /work/outputs/REPORT.md /work/outputs/TOOLS.md; ls -lh /work/outputs'
```
<details><summary>output</summary>

```
7293265ad10295a4996156403ed37d20b428b5025c8932fc9f0af725e35fdd1c  /work/outputs/solve.py
1831d3b378f7bd430f56180f27b8457b286e4b0a19dc3d71c1bab6b21299967a  /work/outputs/object_poses_world.npz
a925286388489cfa397f1416910ef08c2c672c491969bf1b17b8fd0fa8cb77d7  /work/outputs/camera_estimate.json
fa524c8438d4a30911dfc539e3b40f7aea8a1743bcdeac71aaf6eaa623e625d0  /work/outputs/REPORT.md
29a6274cb669a99a5021c447f90d4e3d94903f3c6cd6e59b411bea42660ab1fb  /work/outputs/TOOLS.md
total 128K
-rw-rw-r-- 1 hanzhou hanzhou 4.3K Oct 20 19:45 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1001 Oct 20 19:45 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 19:44 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 1.2K Oct 20 19:45 camera_estimate.json
-rw-rw-r-- 1 hanzhou hanzhou  70K Oct 20 19:45 object_poses_world.npz
-rwxrwx …[+49 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,cv2,h5py
z=np.load('/work/inputs/reconstruction.npz');V=z['human_vertices_world'][0];T=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Ks={'cam':np.array(json.load(open('/work/inputs/camera.json'))['K']),'pipe':np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])}
with h5py.File('/work/inputs/masks.h5','r') as f:m=f['person'][0].astype(bool)
for name,K in Ks.items():
 Xc=(V-T[:3,3])@T[:3,:3];q=Xc@K.T;uv=q[:,:2]/q[:,2:3];im=np.zeros(m.shape,np.uint8);ij=np.clip(np.round(uv).astype(int),[0,0],[1535,1151]);im[ij[:,1],ij[:,0]]=1
 for r in [3,4,5]:
  rr=cv2.dilate(im,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))).astype(bool);iou=(rr&m).sum()/(rr|m).sum();print(name,r,iou,rr.sum())
PY"
```
<details><summary>output</summary>

```
cam 3 0.6691203774491601 33016
cam 4 0.7721289933435299 38866
cam 5 0.8205541305407754 42483
pipe 3 0.48463347782983107 32976
pipe 4 0.5690608788526091 38871
pipe 5 0.6133360897592512 42446

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffmpeg -version | head -1; python - <<'PY'
import torchvision,kornia
print(torchvision.__version__,kornia.__version__)
PY"
```
<details><summary>output</summary>

```
0.20.1+cu124 0.8.3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"- ffmpeg: used only to extract a visual frame montage during development.\\n- SAM3D-Body checkpoint at "'`/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt`: tried for diagnostic 2-D keypoints. The final solver does not run it.'"\\n- Open3D {metadata.version('open3d') if _has_dist('open3d') else '0.20.0'}: imported while checking available tools; not used.\\n- CARI4D source under "'`/workspace` and `/code/reconstruction`: inspected for conventions; no code was copied into the result.'"\\n\", \"- ffmpeg 4.4.2-0ubuntu0.22.04.1: used only to extract a visual frame montage during development.\\n- SAM3D-Body checkpoint at "'`/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt`: tried for diagnostic 2-D keypoints. Its supporting torchvision 0.20.1+cu124 was used by that diagnostic only. The final solver does not run the model.'"\\n- Open3D {metadata.version('open3d') if _has_dist('open3d') else '0.20.0'}: imported while checking available tools; not used.\\n- kornia {metadata.version('kornia') if _has_dist('kornia') else '0.8.3'}: availability probe only; not used.\\n- Availability probes that were not installed and not used: mediapipe, mmpose, ultralytics, onnxruntime, rtmlib.\\n- CARI4D source under "'`/workspace` and `/code/reconstruction`: inspected for conventions; no code was copied into the result.'"\\n- Git repositories cloned or updated: none.\\n\")
open(p,'w') …[+216 chars]
```
<details><summary>output</summary>

```
b943fbe2cd9dd14d27eda5e5661d665314f1428d916b38e842e474dd87ca9a0d

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
[1/5] fitting released floor plane in camera depth
[2/5] optimizing camera on visible dense frame-0 human correspondences
[3/5] scanning global object depth scale against exact mesh/masks
[4/5] enforcing resting-object floor contact (height direction only)
[5/5] validation and report
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/camera_estimate.json
Wrote /work/outputs/REPORT.md
Wrote /work/outputs/TOOLS.md
camera delta: 0.526 deg, 3.31 cm; object depth scale 0.9896; runtime 51.9s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import hashlib,numpy as np,json
z=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
print('array sha256',hashlib.sha256(z.tobytes()).hexdigest())
print('shape/dtype/finite',z.shape,z.dtype,np.isfinite(z).all())
print(json.load(open('/work/outputs/camera_estimate.json'))['camera_change_from_baseline'])
# Remove generated __pycache__ safely if present.
from pathlib import Path
import shutil
p=Path('/work/outputs/__pycache__')
if p.exists(): shutil.rmtree(p)
print('outputs',sorted(x.name for x in Path('/work/outputs').iterdir()))
PY
cat /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
array sha256 b943fbe2cd9dd14d27eda5e5661d665314f1428d916b38e842e474dd87ca9a0d
shape/dtype/finite (778, 4, 4) float64 True
{'rotation_deg': 0.5262140783997293, 'center_cm': 3.3050839583233094}
outputs ['REPORT.md', 'TOOLS.md', 'camera_estimate.json', 'object_poses_world.npz', 'solve.py']
# Tools and provenance

No weights, repositories, datasets, or annotations were downloaded for the final solve. All inputs are the released files in `/work/inputs`; camera-frame depth comes from `/pipeline/depth/moge2/aligned.h5`.

## Runtime dependencies used by `solve.py`

- Python 3.11
- numpy 1.26.3
- scipy 1.16.3
- opencv-python 4.11.0
- h5py 3.16.0
- trimesh 5.1.1
- PyTorch 2.5.1+cu124
- nvdiffrast 0.4.0

## Tools inspected or tried during development (not required by `solve.py`)

- ffmpeg 4.4.2-0ubun …[+801 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py,cv2
from scipy.spatial import cKDTree
h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0];c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T=np.array(c['T_world_cam']);d,idx=cKDTree(h).query(r);s=d<.03;Xr=r[s];Xc=(Xr-T[:3,3])@T[:3,:3];q=Xc@K.T;uv=q[:,:2]/q[:,2:3];u=np.clip(np.round(uv[:,0]).astype(int),0,1535);v=np.clip(np.round(uv[:,1]).astype(int),0,1151)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:b=bytes(f['aligned/front_stereo_camera_left'][0]);D=cv2.imdecode(np.frombuffer(b,np.uint8),1).astype(np.float32)/1000
with h5py.File('/work/inputs/masks.h5','r') as f:m=f['person'][0]
vis=m[v,u]&(np.abs(Xc[:,2]-D[v,u])<.08);ids=idx[s][vis];print('sel',vis.sum(),'unique',len(np.unique(ids)),'counts',np.bincount(np.unique(ids,return_counts=True)[1])[:10], 'maxcount',np.bincount(ids).max())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
ValueError: operands could not be broadcast together with shapes (4610,) (4610,3) 

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/dense_unique_test.py <<'PY'
import numpy as np,json,h5py,cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);T0=np.array(c['T_world_cam']);h=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world'];r=np.load('/work/inputs/reconstruction.npz')['human_vertices_world'][0]
d,idx=cKDTree(h).query(r);s=d<.03
Xr=r[s];Xctrue=(Xr-T0[:3,3])@T0[:3,:3];q=Xctrue@K.T;uv=q[:,:2]/q[:,2:3];u=np.clip(np.round(uv[:,0]).astype(int),0,1535);v=np.clip(np.round(uv[:,1]).astype(int),0,1151)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:b=bytes(f['aligned/front_stereo_camera_left'][0]);D=cv2.imdecode(np.frombuffer(b,np.uint8),1).astype(np.float32)/1000
with h5py.File('/work/inputs/masks.h5','r') as f:m=f['person'][0]
vis=m[v,u]&(np.abs(Xctrue[:,2]-D[v,u])<.08); rr=np.nonzero(s)[0][vis];ii=idx[s][vis]
print('all visible',len(rr),'unique',len(np.unique(ii)))
# unique min reconstruction match distance per true index
order=np.argsort(d[rr]);seen=set();keep=[]
for k in order:
 if ii[k] not in seen: seen.add(ii[k]);keep.append(k)
keep=np.array(keep);print('unique keep',len(keep))
P=np.array([-.02020033,-.99479125,-.09991152]);nw=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3])
for nm,kk in [('all',np.arange(len(rr))),('unique',keep)]:
 X=h[ii[kk]];tar=uv[kk]
 def pose(x):R=Rotation.fro …[+526 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/dense_unique_test.py", line 10, in <module>
    vis=m[v,u]&(np.abs(Xctrue[:,2]-D[v,u])<.08); rr=np.nonzero(s)[0][vis];ii=idx[s][vis]
                       ~~~~~~~~~~~^~~~~~~
ValueError: operands could not be broadcast together with shapes (4610,) (4610,3) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"| same sign and magnitude as dense translation; final contact explicitly enforced |\",\"| same sign; weak scalar height constraint; final contact explicitly enforced |\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
[1/5] fitting released floor plane in camera depth
[2/5] optimizing camera on visible dense frame-0 human correspondences
[3/5] scanning global object depth scale against exact mesh/masks
[4/5] enforcing resting-object floor contact (height direction only)
[5/5] validation and report
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/camera_estimate.json
Wrote /work/outputs/REPORT.md
Wrote /work/outputs/TOOLS.md
camera delta: 0.526 deg, 3.31 cm; object depth scale 0.9896; runtime 47.0s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
import shutil,hashlib,numpy as np,json
p=Path('/work/outputs/__pycache__')
if p.exists(): shutil.rmtree(p)
z=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
print('keys',np.load('/work/outputs/object_poses_world.npz').files)
print('shape/dtype/finite',z.shape,z.dtype,bool(np.isfinite(z).all()))
print('array_sha256',hashlib.sha256(z.tobytes()).hexdigest())
print('deterministic_expected',hashlib.sha256(z.tobytes()).hexdigest()=='b943fbe2cd9dd14d27eda5e5661d665314f1428d916b38e842e474dd87ca9a0d')
c=json.load(open('/work/outputs/camera_estimate.json'));print('K',c['K']);print('cam_center',[c['T_world_cam'][i][3] for i in range(3)]);print('delta',c['camera_change_from_baseline'],'scale',c['object_depth_scale'])
print('files',sorted(x.name for x in Path('/work/outputs').iterdir()))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"Resting object floor contact\\|Validation\\|Runtime\\|same sign\" /work/outputs/REPORT.md; tail -20 /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
19:| Resting object floor contact | 0.34 cm along upward normal before depth scaling | same sign; weak scalar height constraint; final contact explicitly enforced |
26:## Validation (baseline vs adopted)
47:- Runtime: 47.0 s on one GPU/CPU process.
| Human boundary distance, mask -> render, median / p95 (px) | 5.00 / 12.00 | 5.00 / 12.00 |
| MoGe floor points to released plane, median abs / RMS (m) | 0.0786 / 0.0841 | 0.0709 / 0.0763 |
| Human lowest-vertex floor distance, median (m) | -0.0194 | -0.0116 |
| Human 0.5th-percentile floor distance, median (m) | -0.0161 | -0.0088 |
| Resting object lowest-vertex floor distance, median (m) | -0.0256 | -0.0084 |
| Resting object lowest-vertex mean abs distance (m) | 0.0282 | 0.0119 |
| Exact-object mask IoU over episode | 0.7505 | 0.7672 |

The  …[+1694 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -3; sha256sum /work/outputs/solve.py /work/outputs/object_poses_world.npz /work/outputs/camera_estimate.json /work/outputs/REPORT.md /work/outputs/TOOLS.md'
```
<details><summary>output</summary>

```
NVIDIA RTX 6000 Ada Generation, 691 MiB
NVIDIA RTX 6000 Ada Generation, 14325 MiB
168993a5549ef3b682b1c5f2cba8bf4c83f963039f7097c8b67489f3709d288e  /work/outputs/solve.py
1831d3b378f7bd430f56180f27b8457b286e4b0a19dc3d71c1bab6b21299967a  /work/outputs/object_poses_world.npz
a925286388489cfa397f1416910ef08c2c672c491969bf1b17b8fd0fa8cb77d7  /work/outputs/camera_estimate.json
a655bb1730a3dfa9a0b4d79fa1cc4eb689a9a834fb5a030b333f66ddb1ae2554  /work/outputs/REPORT.md
42c9291de02ee1e636ccfb005dbeb33eb56993146c691cfaee54395f8b5c00f5  /work/outputs/TOOLS.md

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(778, 4, 4)`, `float64`
- `/work/outputs/camera_estimate.json`
- `/work/outputs/REPORT.md`
- `/work/outputs/TOOLS.md`

Key results:
- Camera rotation change: **0.526°**
- Camera center change: **3.31 cm**
- Object camera-ray depth scale: **0.9896**
- Frame-0 human silhouette IoU: **0.8171 → 0.8176**
- MoGe floor residual RMS: **0.0841 → 0.0763 m**
- Exact-object mask IoU: **0.7505 → 0.7672**
- Final solve runtime: **47 s**
- Numeric output verified finite, correctly shaped, and deterministic across repeated runs.
