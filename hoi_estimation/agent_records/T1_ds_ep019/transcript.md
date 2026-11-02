# T1_ds_ep019

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T14:52:21Z

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
and translation both matter; a  …[+9786 chars]
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
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort"
```
<details><summary>output</summary>

```
.preexported 0 bytes
camera.json 838 bytes
episode.json 59 bytes
frame0_reference_human.npz 196171 bytes
ground_plane.json 394 bytes
masks.h5 7673836 bytes
object_mesh.glb 12389176 bytes
reconstruction.npz 44729457 bytes
task.txt 233 bytes
video.mp4 25478983 bytes

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/inputs/task.txt; printf '\\n--- episode ---\\n'; cat /work/inputs/episode.json; printf '\\n--- camera ---\\n'; cat /work/inputs/camera.json; printf '\\n--- ground ---\\n'; cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
Walk over to the cane on the ground and use your right hand to grasp the cane then swing the cane on your side and then proceed to handover the cane to your left hand and perform a swinging motion. Place on the ground when complete.

--- episode ---
{
 "episode_index": 19,
 "object": "cane",
 "frames": 886
}
--- camera ---
{
 "K": [
  [
   819.9994506835938,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9994506835938,
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
   820.802978515625,
   0.0,
   768.0
  ],
  [
   0.0,
   820.8029174804688,
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
   0.9990210356746521,
   0.0028383411854098228,
   0.04414650721020355,
   0.010305840814246198
  ],
  [
   -0.0049460542995448335,
   0.998 …[+773 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, cv2
p='/work/inputs/reconstruction.npz'
z=np.load(p)
print('npz keys:')
for k in z.files: print(k,z[k].shape,z[k].dtype)
with h5py.File('/work/inputs/masks.h5') as f:
 print('h5 keys')
 for k in f: print(k,f[k].shape,f[k].dtype)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
print('video',cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
PY"
```
<details><summary>output</summary>

```
npz keys:
T_world_mesh (886, 4, 4) float32
T_cam_mesh_foundationpose (886, 4, 4) float32
T_cam_mesh_coconet (886, 4, 4) float32
human_vertices_world (886, 4610, 3) float32
human_keypoints_world (886, 70, 3) float32
contact_prob_right_left (886, 2) float32
h5 keys
object (886, 1152, 1536) bool
person (886, 1152, 1536) bool
video 886.0 30.0 1536.0 1152.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch /work/outputs /work/tools && python - <<'PY'
import numpy as np, trimesh, json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh'].astype(float)
print('T shapes',T.shape)
t=T[:,:3,3]
print('trans min max mean',t.min(0),t.max(0),t.mean(0))
print('trans first/last',t[0],t[-1])
print('translational per-frame speed quantiles',np.quantile(np.linalg.norm(np.diff(t,axis=0),axis=1),[0,.1,.25,.5,.75,.9,.95,.99,1]))
# rot deltas
rel=T[1:,:3,:3]@np.swapaxes(T[:-1,:3,:3],1,2)
ang=R.from_matrix(rel).magnitude()*180/np.pi
print('rot delta quantiles',np.quantile(ang,[0,.1,.25,.5,.75,.9,.95,.99,1]))
print('big rot frame indices')
for i in np.where(ang>10)[0]: print(i,ang[i])
scene=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(scene)
for i,g in scene.geometry.items():
 print(i,g, g.vertices.shape,g.faces.shape,'bounds',g.bounds,'extents',g.extents,'watertight',g.is_watertight)
 if hasattr(g,'visual'): print('visual',type(g.visual), getattr(g.visual,'material',None), getattr(g.visual,'uv',None).shape if getattr(g.visual,'uv',None) is not None else None)
# contact and masks
c=z['contact_prob_right_left']
print('contact right/left at percentiles', np.quantile(c[:,0],[0,.1,.25,.5,.75,.9,1]),np.quantile(c[:,1],[0,.1,.25,.5,.75,.9,1]))
print('contact > .5 ranges right',np.where(c[:,0]>.5)[0][[0,-1]] if np.any(c[:,0]>.5) else None,'left',np.where(c[:,1]>.5)[0][[0,- …[+39 chars]
```
<details><summary>output</summary>

```
T shapes (886, 4, 4)
trans min max mean [-0.4806577   0.02801887  2.50286651] [0.71224314 0.91251862 4.08410072] [0.09327446 0.63845057 3.21239135]
trans first/last [0.03929602 0.85473436 2.95688128] [0.20713741 0.78276062 3.82325506]
translational per-frame speed quantiles [0.         0.00073323 0.01287305 0.02762245 0.04622976 0.06964918
 0.08709582 0.12867509 0.23456497]
rot delta quantiles [3.09924958e-15 8.30847962e-15 2.74010885e-02 3.78183484e+00
 9.13829372e+00 1.96081998e+01 2.72390336e+01 5.52063514e+01
 1.21244688e+02]
big rot frame indices
9 12.030993825280792
10 14.019844330922846
12 19.170074165477505
13 22.06016894573351
32 17.20063731472468
33 20.159570595889637
38 24.148184544427796
100 31.826576512077256
101 19.70627309739223
117 12.631486192343157
120 16.314794781510415
 …[+4971 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/diagnose.py <<'PY'
import numpy as np, h5py, cv2, json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; c=z['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as f:
    obj=f['object'][:]; per=f['person'][:]
print('object area',np.quantile(obj.sum((1,2)),[0,.01,.1,.25,.5,.75,.9,.95,.99,1]))
print('person area',np.quantile(per.sum((1,2)),[0,.1,.5,.9,1]))
# object components and bbox/centroid
n=[]; boxes=[]; cents=[]; comps=[]
for i in range(len(obj)):
    m=obj[i].astype(np.uint8)
    k,lab,stats,cen=cv2.connectedComponentsWithStats(m,8)
    if k>1:
      ids=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]); a=stats[ids,cv2.CC_STAT_AREA]; x,y,w,h=stats[ids,:4]; cc=cen[ids]
    else: a=0;x=y=w=h=0;cc=[np.nan,np.nan]
    boxes.append([x,y,w,h]);cents.append(cc);comps.append(k-1)
boxes=np.array(boxes);cents=np.array(cents)
# print each 10
print('i area cx cy wh ncomp contactR contactL trans(xyz) speed rotdeg')
t=T[:,:3,3]
rel=T[1:,:3,:3]@np.swapaxes(T[:-1,:3,:3],1,2); ang=R.from_matrix(rel).magnitude()*180/np.pi
for i in range(0,len(obj),10):
 sp=np.linalg.norm(t[i]-t[i-1]) if i else 0
 rd=ang[i-1] if i else 0
 print(f'{i:3d} {obj[i].sum():7d} {cents[i,0]:7.1f} {cents[i,1]:7.1f} {boxes[i,2]:4d} {boxes[i,3]:4d} {comps[i]:2d} {c[i,0]:.2f} {c[i,1]:.2f} {t[i,0]: .3f} {t[i,1]: .3f} {t[i,2]: .3f} {sp:.3f} {rd:6.1f}')
np.savez('/work/scratch/basic.npz',obj_area=obj.sum((1,2)), …[+177 chars]
```
<details><summary>output</summary>

```
object area [   0.    335.35  954.5  2230.25 3068.5  3704.   4100.   4240.75 4489.15
 5036.  ]
person area [20880.  30170.5 40401.5 54563.5 59159. ]
i area cx cy wh ncomp contactR contactL trans(xyz) speed rotdeg
  0    3996   726.4   803.1   67  294  1 0.00 1.00  0.039  0.855  2.957 0.000    0.0
 10    4012   725.9   801.5   68  295  1 0.00 1.00  0.042  0.845  2.915 0.013   12.0
 20    3852   726.1   804.9   68  295  1 0.00 0.03  0.052  0.860  2.972 0.018    2.8
 30    3484   726.7   805.7   67  296  1 0.00 0.01  0.047  0.842  2.998 0.013    2.0
 40    3827   726.0   805.6   68  297  1 0.00 0.00  0.039  0.812  2.925 0.000    0.0
 50    3714   726.2   806.6   67  296  1 0.00 0.00  0.039  0.813  2.929 0.000    0.0
 60    3708   726.3   806.5   67  296  1 0.00 0.00  0.039  0.816  2.937 0.000 …[+6978 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, h5py, numpy as np, os
cap=cv2.VideoCapture('/work/inputs/video.mp4')
ids=list(range(0,886,25))
with h5py.File('/work/inputs/masks.h5') as f:
 obj=f['object'][:]; per=f['person'][:]
thumbs=[]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 # mask outlines
 m=(obj[i].astype(np.uint8)*255); cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(rgb,cnt,-1,(0,255,0),3)
 p=(per[i].astype(np.uint8)*255); cnt,_=cv2.findContours(p,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(rgb,cnt,-1,(255,0,255),2)
 rgb=cv2.resize(rgb,(384,288))
 cv2.putText(rgb,f'{i}',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2,cv2.LINE_AA)
 thumbs.append(rgb)
rows=[]
for j in range(0,len(thumbs),6): rows.append(np.concatenate(thumbs[j:j+6],axis=1))
while len(rows[-1].shape)==3 and rows[-1].shape[1]<2304:
 rows[-1]=np.concatenate([rows[-1],np.zeros((288,2304-rows[-1].shape[1],3),np.uint8)],axis=1)
out=np.concatenate(rows,axis=0)
cv2.imwrite('/work/scratch/contact.jpg',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
print(out.shape)
PY"
```
<details><summary>output</summary>

```
(1728, 2304, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1931
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2133
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 1092899680
/pipeline/depth/moge2/depth_report.json 1812
/pipeline/depth/moge2/intrinsics.pkl 94323
/pipeline/depth/moge2/raw.npy 3135504512
/pipeline/export/episode_000019/edex 562
/pipeline/export/episode_000019/wild_export.json 2780
/pipeline/foundationpose/episode_000019.pkl 252465
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 677606 …[+549 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace -maxdepth 2 -type d -printf '%p\\n' | sort | head -100; find /code/reconstruction -maxdepth 2 -type f -printf '%p\\n' | sort | head -100"
```
<details><summary>output</summary>

```
/workspace
/workspace/third_party
/workspace/third_party/kaolin
/workspace/third_party/nvdiffrast
/workspace/v2d_cari4d
/workspace/v2d_cari4d/lib
/workspace/v2d_common
/workspace/v2d_common/tests
/workspace/v2d_common/v2d_common.egg-info
/workspace/v2d_foundation_pose
/workspace/v2d_foundation_pose/lib
/workspace/v2d_mesh
/workspace/v2d_mesh/lib
/workspace/v2d_moge
/workspace/v2d_moge/lib
/workspace/v2d_mv
/workspace/v2d_mv/cli
/workspace/v2d_mv/io
/workspace/v2d_mv/math
/workspace/v2d_mv/rig
/workspace/v2d_mv/v2d_mv.egg-info
/workspace/v2d_mv/vis
/workspace/v2d_sam2
/workspace/v2d_sam2/assets
/workspace/v2d_sam2/docker
/workspace/v2d_sam2/lib
/workspace/v2d_sam3d_body
/workspace/v2d_sam3d_body/assets
/workspace/v2d_sam3d_body/docker
/workspace/v2d_sam3d_body/lib
/code/reconstruction/.curs …[+1517 chars]
```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader; /usr/bin/time -f '%e sec %M KB' python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
try:
 import nvdiffrast; print('nvdiffrast ok')
except Exception as e: print('nvdiffrast err',e)
PY"
```
<details><summary>output</summary>

```
NVIDIA RTX 6000 Ada Generation, 49140 MiB, 2601 MiB, 45901 MiB, 1 %
NVIDIA RTX 6000 Ada Generation, 49140 MiB, 14325 MiB, 34185 MiB, 21 %
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "env | sort | grep -E 'CUDA|NVIDIA|PYTHON|HF|TORCH'"
```
<details><summary>output</summary>

```
CUDA_VERSION=12.4.1
CUDA_VISIBLE_DEVICES=0
HF_HOME=/weights/hf_home
HF_HUB_OFFLINE=1
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_PRODUCT_NAME=CUDA
NVIDIA_REQUIRE_CUDA=cuda>=12.4 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver> …[+1020 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/.stages/01_depth.json; echo; cat /pipeline/.stages/05_object_pose.json; echo; cat /pipeline/.stages/06_coconet.json; echo; cat /pipeline/.stages/07_refinement.json; echo; cat /pipeline/.stages/08_render.json'
```
<details><summary>output</summary>

```
{
  "name": "01_depth",
  "outputs": [
    {
      "mtime_ns": 1791261021136587548,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000019/depth/moge2/raw.npy",
      "size": 3135504512
    },
    {
      "mtime_ns": 1791261021353587547,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000019/depth/moge2/intrinsics.pkl",
      "size": 94323
    },
    {
      "mtime_ns": 1791261023830587538,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000019/depth/moge2/depth_report.json",
      "size": 1812
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/mhr_wild_depth.py",
      "--video",
      "/home/hanzhou/Projects/v2d/run/work/episode_000019/episode_000019.0.co …[+10208 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch, os
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())
try:
 import nvdiffrast; print('nvdiffrast ok')
except Exception as e: print('nvdiffrast err',repr(e))
PY"
```
<details><summary>output</summary>

```
2.5.1+cu124 12.4 True 1
nvdiffrast ok

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np, os, cv2
scene=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=scene.geometry['goodcane']
m=g.visual.material
print(type(m),m)
for k,v in vars(m).items():
 print(k,type(v),getattr(v,'shape',None),repr(v)[:200])
if getattr(m,'baseColorTexture',None) is not None:
 im=np.array(m.baseColorTexture); print('tex',im.shape,im.dtype,im.min(),im.max()); cv2.imwrite('/work/scratch/texture.png',cv2.cvtColor(im,cv2.COLOR_RGB2BGR)); print(im.mean((0,1)))
# vertex colors / normals / components
print('uv bounds',g.visual.uv.min(0),g.visual.uv.max(0))
print('components',len(g.split(only_watertight=False)))
for j,c in enumerate(g.split(only_watertight=False)):
 print(j,c.vertices.shape,c.faces.shape,c.bounds,c.extents)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.material.PBRMaterial'> <trimesh.visual.material.PBRMaterial object at 0x7545f0f54bd0>
_data <class 'dict'> None {'metallicFactor': 0.0, 'baseColorTexture': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=2216x2192 at 0x7545F0F80E90>, 'doubleSided': True, 'name': '_texture'}
tex (2192, 2216, 3) uint8 0 255
[97.04808612 75.85726608 45.66820828]
uv bounds [0.002058 0.002069] [0.99794602 0.99793398]
Traceback (most recent call last):
  File "<stdin>", line 12, in <module>
  File "/opt/conda/lib/python3.11/site-packages/trimesh/base.py", line 1454, in split
    return graph.split(self, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trimesh/graph.py", line 391, in split
    return mesh.submesh(
           ^^^^^^^^^^^^^ …[+988 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
for path,prefix in [('/pipeline/inference/episode_000019_step200000_before_after.mp4','ba'),('/pipeline/inference/episode_000019_step200000_reconstruction.mp4','recon')]:
 cap=cv2.VideoCapture(path); print(path,cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FPS))
 ids=[0,100,150,200,250,300,400,500,550,600,650,700,750,800,850,885]
 ims=[]
 for i in ids:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
  if not ok: print('bad',i); continue
  im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(8,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
  ims.append(im)
 rows=[]
 for j in range(0,len(ims),4): rows.append(cv2.hconcat(ims[j:j+4]))
 cv2.imwrite(f'/work/scratch/{prefix}_contact.jpg',cv2.vconcat(rows))
PY"
```
<details><summary>output</summary>

```
/pipeline/inference/episode_000019_step200000_before_after.mp4 886.0 4608.0 1236.0 30.0
/pipeline/inference/episode_000019_step200000_reconstruction.mp4 886.0 3072.0 1152.0 30.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
Tw=z['T_world_mesh'].astype(float); Tf=z['T_cam_mesh_foundationpose'].astype(float); Tc=z['T_cam_mesh_coconet'].astype(float)
cam=json.load(open('/work/inputs/camera.json')); W=np.array(cam['T_world_cam'])
Wf=W@Tf; Wc=W@Tc
for name,A,B in [('base-fp',Tw,Wf),('base-coco',Tw,Wc),('fp-coco',Wf,Wc)]:
 rel=A[:,:3,:3]@np.swapaxes(B[:,:3,:3],1,2)
 a=R.from_matrix(rel).magnitude()*180/np.pi
 print(name,'rot diff',np.quantile(a,[0,.1,.25,.5,.75,.9,.95,.99,1]))
 dt=np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1)
 print(' translation diff',np.quantile(dt,[0,.1,.5,.9,1]))
# compare interframe rot deltas
for name,A in [('base',Tw),('fp',Wf),('coco',Wc)]:
 rel=A[1:,:3,:3]@np.swapaxes(A[:-1,:3,:3],1,2); a=R.from_matrix(rel).magnitude()*180/np.pi
 print(name,'delta rot',np.quantile(a,[.5,.75,.9,.95,.99,1]))
# translation variations
for name,A in [('base',Tw),('fp',Wf),('coco',Wc)]:
 t=A[:,:3,3]; sp=np.linalg.norm(np.diff(t,axis=0),axis=1)
 print(name,'speed',np.quantile(sp,[.5,.75,.9,.95,.99,1]))
# print rows every 10 for first 120
for i in range(0,130,5):
 print(i,'base',Tw[i,:3,3], 'coco',Wc[i,:3,3], 'fp',Wf[i,:3,3])
PY"
```
<details><summary>output</summary>

```
base-fp rot diff [  1.03867602   5.33009597  11.47750328  23.4509156   53.58811004
  96.8627796  144.17392203 169.02012273 179.06977867]
 translation diff [0.03908344 0.10981585 0.41180806 1.12776862 3.76307941]
base-coco rot diff [1.205092   1.22945642 1.23453609 1.25751411 1.26330656 1.27040494
 1.27464821 1.28125297 1.28295482]
 translation diff [0.05326097 0.06106493 0.07205866 0.08586507 0.11496471]
fp-coco rot diff [  0.61291601   4.83087059  10.80913348  22.86058549  53.92119864
  96.79508204 144.62718065 169.55303872 179.24658498]
 translation diff [0.01603052 0.12425193 0.44540596 1.14205013 3.77585706]
base delta rot [  3.78183484   9.13829372  19.6081998   27.23903361  55.20635144
 121.24468817]
fp delta rot [  5.20428215  11.19817867  22.66071665  34.23126309 165.9216335
 179.4 …[+3742 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py, time
import nvdiffrast.torch as dr
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
dev='cuda'
# load
scene=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=scene.geometry['goodcane']
V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev)
F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev)
uv=torch.tensor(np.asarray(g.visual.uv),dtype=torch.float32,device=dev)
tex=np.array(g.visual.material.baseColorTexture)
tex=torch.tensor(tex,dtype=torch.float32,device=dev)/255
tex=tex.permute(2,0,1)[None]
print('V,F,uv,tex',V.shape,F.shape,uv.shape,tex.shape)
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'].astype(np.float32),device=dev)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); W=np.array(cam['T_world_cam'],np.float32); Wc_inv=np.linalg.inv(W).astype(np.float32)
Kt=torch.tensor(K,device=dev)
gl=dr.RasterizeCudaContext()
H,Wd=1152,1536
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f: objs=f['object'][:]; pers=f['person'][:]

def render(i):
    Tt=T[i]
    vw=V@Tt[:3,:3].T+Tt[:3,3]
    vc=vw@torch.tensor(Wc_inv[:3,:3],device=dev).T+torch.tensor(Wc_inv[:3,3],device=dev)
    x,y,z=vc.unbind(1)
    clip=torch.stack([Kt[0,0]*x+Kt[0,2]*z,Kt[1,1]*y+Kt[1,2]*z,z,z],1)[None]
    rast, _ = dr.rasterize(gl,clip, …[+2100 chars]
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 43, in <module>
    t=time.time(); rgb,mask,zz=render(i); torch.cuda.synchronize(); dt=time.time()-t
                               ^^^^^^^^^
  File "/work/scratch/render_test.py", line 37, in render
    zz,_=dr.interpolate(z[None,None],rast,F) # N,H,W,1
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    retur …[+408 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dr.interpolate(z\\[None,None\\],rast,F)/dr.interpolate(z[None,:,None].contiguous(),rast,F)/' /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
0 iou 0.0 iou_vis 0.0 area render/mask 0 3996 time 0.11772274971008301
100 iou 0.0 iou_vis 0.0 area render/mask 0 3610 time 0.0018355846405029297
120 iou 0.0 iou_vis 0.0 area render/mask 0 3769 time 0.0013167858123779297
150 iou 0.0 iou_vis 0.0 area render/mask 0 2609 time 0.0012319087982177734
200 iou 0.0 iou_vis 0.0 area render/mask 0 3987 time 0.0011622905731201172
300 iou 0.0 iou_vis 0.0 area render/mask 0 3987 time 0.0012464523315429688
400 iou 0.0 iou_vis 0.0 area render/mask 0 4313 time 0.0013318061828613281
500 iou 0.0 iou_vis 0.0 area render/mask 0 3336 time 0.0012927055358886719
600 iou 0.0 iou_vis 0.0 area render/mask 0 3419 time 0.0012505054473876953
700 iou 0.0 iou_ …[+207 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py'
s=open(p).read()
s=s.replace(\"clip=torch.stack([Kt[0,0]*x+Kt[0,2]*z,Kt[1,1]*y+Kt[1,2]*z,z,z],1)[None]\", \"u=Kt[0,0]*x+Kt[0,2]*z; v=Kt[1,1]*y+Kt[1,2]*z\\n    clip=torch.stack([(2.0*u/Wd-1.0)*z,(1.0-2.0*v/H)*z,0.5*z,z],1)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
0 iou 0.0 iou_vis 0.0 area render/mask 0 3996 time 0.11303949356079102
100 iou 0.0 iou_vis 0.0 area render/mask 0 3610 time 0.0020682811737060547
120 iou 0.0 iou_vis 0.0 area render/mask 0 3769 time 0.0013203620910644531
150 iou 0.0 iou_vis 0.0 area render/mask 0 2609 time 0.0012590885162353516
200 iou 0.0 iou_vis 0.0 area render/mask 0 3987 time 0.0012018680572509766
300 iou 0.0 iou_vis 0.0 area render/mask 0 3987 time 0.0014154911041259766
400 iou 0.0 iou_vis 0.0 area render/mask 0 4313 time 0.0015654563903808594
500 iou 0.0 iou_vis 0.0 area render/mask 0 3336 time 0.0013456344604492188
600 iou 0.0 iou_vis 0.0 area render/mask 0 3419 time 0.005608081817626953
700 iou 0.0 iou_v …[+208 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr, inspect
print(inspect.signature(dr.rasterize)); print(dr.rasterize.__doc__)
PY"
```
<details><summary>output</summary>

```
(glctx, pos, tri, resolution, ranges=None, grad_db=True)
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
        ranges: In range mod …[+634 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'][0].astype(float)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane']
V=np.array(g.vertices); vw=V@T[:3,:3].T+T[:3,3]
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); W=np.array(c['T_world_cam']); Wi=np.linalg.inv(W)
vc=vw@Wi[:3,:3].T+Wi[:3,3]
uv=vc@K.T; uv=uv[:,:2]/uv[:,2:3]
print('depth',np.quantile(vc[:,2],[0,.5,1]))
print('uv',np.quantile(uv,[0,.5,1],axis=0))
print('centroid',uv.mean(0))
print('T',T)
print('cam',W)
PY"
```
<details><summary>output</summary>

```
depth [2.44839434 2.66186354 3.13025798]
uv [[699.11712391 648.08408941]
 [730.1190086  844.82410754]
 [764.74354324 915.07058667]]
centroid [730.2275626  807.64077859]
T [[-0.32779336 -0.94410121 -0.03499364  0.03929602]
 [ 0.57666391 -0.17060429 -0.79896981  0.85473436]
 [ 0.74833822 -0.28207663  0.60035193  2.95688128]
 [ 0.          0.          0.          1.        ]]
cam [[ 0.99902104  0.00283834  0.04414651  0.01030584]
 [-0.00494605  0.99884908  0.04770791 -0.04685502]
 [-0.04396029 -0.04787956  0.99788528  0.23275473]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr
gl=dr.RasterizeCudaContext(); dev='cuda'
for zz in [0,0.5,0.9,-0.5,1.0]:
 pos=torch.tensor([[-.5,-.5,zz,1],[.5,-.5,zz,1],[0,.5,zz,1]],device=dev,dtype=torch.float32)[None]
 tri=torch.tensor([[0,1,2]],device=dev,dtype=torch.int32)
 r,_=dr.rasterize(gl,pos,tri,[100,100]); print(zz,(r[0,:,:,3]>0).sum().item())
PY"
```
<details><summary>output</summary>

```
0 1250
0.5 1250
0.9 1250
-0.5 1250
1.0 1250

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/render_test.py | sed -n '20,55p'"
```
<details><summary>output</summary>

```
    20	H,Wd=1152,1536
    21	cap=cv2.VideoCapture('/work/inputs/video.mp4')
    22	with h5py.File('/work/inputs/masks.h5') as f: objs=f['object'][:]; pers=f['person'][:]
    23	
    24	def render(i):
    25	    Tt=T[i]
    26	    vw=V@Tt[:3,:3].T+Tt[:3,3]
    27	    vc=vw@torch.tensor(Wc_inv[:3,:3],device=dev).T+torch.tensor(Wc_inv[:3,3],device=dev)
    28	    x,y,z=vc.unbind(1)
    29	    u=Kt[0,0]*x+Kt[0,2]*z; v=Kt[1,1]*y+Kt[1,2]*z
    30	    clip=torch.stack([(2.0*u/Wd-1.0)*z,(1.0-2.0*v/H)*z,0.5*z,z],1)[None]
    31	    rast, _ = dr.rasterize(gl,clip,F,resolution=[H,Wd])
    32	    vul, _ = dr.interpolate(uv[None],rast,F)
    33	    # uv from trimesh image convention? likely origin upper-left, so normalize direct
    34	    uv_grid=vul*2-1
    35	    rgb=torch.nn.functional.grid_sample( …[+1375 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split('cap=cv2.VideoCapture')[0])
Tt=T[0]; vw=V@Tt[:3,:3].T+Tt[:3,3]; vc=vw@torch.tensor(Wc_inv[:3,:3],device=dev).T+torch.tensor(Wc_inv[:3,3],device=dev)
x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z
clip=torch.stack([(2*u/Wd-1)*z,(1-2*v/H)*z,.5*z,z],1)[None].contiguous()
print('nan',torch.isnan(clip).any(),'min',clip[...,:3].min((0,1)).values,'max',clip[...,:3].max((0,1)).values,'w',clip[...,3].min(),clip[...,3].max())
for nf in [1,10,100,1000,10000,99266]:
 r,_=dr.rasterize(gl,clip,F[:nf],[H,Wd]); print(nf,(r[0,:,:,3]>0).sum().item(),r[0,:,:,3].max().item())
# triangle bbox projected first face inspect
print('face 0',clip[0,F[0]],F[0])
PY"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
TypeError: min() received an invalid combination of arguments - got (tuple), but expected one of:
 * ()
      didn't match because some of the arguments have invalid types: (!tuple of (int, int)!)
 * (Tensor other)
      didn't match because some of the arguments have invalid types: (!tuple of (int, int)!)
 * (int dim, bool keepdim = False)
 * (name dim, bool keepdim = False)


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split('cap=cv2.VideoCapture')[0])
Tt=T[0]; vw=V@Tt[:3,:3].T+Tt[:3,3]; vc=vw@torch.tensor(Wc_inv[:3,:3],device=dev).T+torch.tensor(Wc_inv[:3,3],device=dev)
x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z
clip=torch.stack([(2*u/Wd-1)*z,(1-2*v/H)*z,.5*z,z],1)[None].contiguous()
print('nan',torch.isnan(clip).any(),'min',clip[...,:3].amin((0,1)),'max',clip[...,:3].amax((0,1)),'w',clip[...,3].min(),clip[...,3].max())
for nf in [1,10,100,1000,10000,99266]:
 r,_=dr.rasterize(gl,clip,F[:nf],[H,Wd]); print(nf,(r[0,:,:,3]>0).sum().item(),r[0,:,:,3].max().item())
print('face 0',clip[0,F[0]],F[0])
PY"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
nan tensor(False, device='cuda:0') min tensor([ 3.1710, -8.4857,  1.2242], device='cuda:0') max tensor([ 6.0461, -6.9149,  1.5651], device='cuda:0') w tensor(2.4484, device='cuda:0') tensor(3.1303, device='cuda:0')
1 0 0.0
10 0 0.0
100 0 0.0
1000 0 0.0
10000 0 0.0
99266 0 0.0
face 0 tensor([[ 3.6664, -8.1383,  1.3063,  2.6126],
        [ 3.6635, -8.1395,  1.3059,  2.6117],
        [ 3.6636, -8.1520,  1.3062,  2.6125]], device='cuda:0') tensor([69499, 20987, 69488], device='cuda:0', dtype=torch.int32)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split('cap=cv2.VideoCapture')[0])
Tt=T[0]; vw=V@Tt[:3,:3].T+Tt[:3,3]; vc=vw@torch.tensor(Wc_inv[:3,:3],device=dev).T+torch.tensor(Wc_inv[:3,3],device=dev)
x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z
print('vc',vc[:3],'z',z[:3],'u',u[:3],'v',v[:3]);print('uv', (u/z)[:3],(v/z)[:3]);print('clip',torch.stack([(2*u/Wd-1)*z,(1-2*v/H)*z,.5*z,z],1)[:3])
print('V first',V[:3]); print('T',Tt)
# numpy
import numpy as np
Vn=np.array(g.vertices); Tn=z['T_world_mesh'][0].astype(float); Wn=np.array(cam['T_world_cam']); Ki=np.array(cam['K']); wi=np.linalg.inv(Wn)
vwn=Vn@Tn[:3,:3].T+Tn[:3,3]; vcn=vwn@wi[:3,:3].T+wi[:3,3]; uvn=vcn@Ki.T; uvn=uvn[:,:2]/uvn[:,2:3]
print('np uv',uvn[:3], 'vcn',vcn[:3])
PY"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
vc tensor([[-0.1706,  1.1194,  2.6099],
        [-0.1706,  1.1194,  2.6099],
        [-0.1706,  1.1194,  2.6099]], device='cuda:0') z tensor([2.6099, 2.6099, 2.6099], device='cuda:0') u tensor([1838.6631, 1838.6631, 1838.6631], device='cuda:0') v tensor([2380.7131, 2380.7131, 2380.7131], device='cuda:0')
uv tensor([704.4995, 704.4995, 704.4995], device='cuda:0') tensor([912.1907, 912.1907, 912.1907], device='cuda:0')
clip tensor([[ 3.6384, -8.1772,  1.3049,  2.6099],
        [ 3.6384, -8.1772,  1.3049,  2.6099],
        [ 3.6384, -8.1772,  1.3049,  2.6099]], device='cuda:0')
V first tensor([[ 0.0996,  0.0650, -0.3699],
        [ 0.0996,  0.0650, -0.3699],
        [ 0.0996,  0.06 …[+364 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read()
s=s.replace(\"clip=torch.stack([(2.0*u/Wd-1.0)*z,(1.0-2.0*v/H)*z,0.5*z,z],1)[None]\", \"clip=torch.stack([2.0*u/Wd-z,z-2.0*v/H,0.5*z,z],1)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
0 iou 0.0 iou_vis 0.0 area render/mask 2512 3996 time 0.09204769134521484
100 iou 0.0 iou_vis 0.0 area render/mask 2346 3610 time 0.0016481876373291016
120 iou 0.0 iou_vis 0.0 area render/mask 2984 3769 time 0.001262664794921875
150 iou 0.0 iou_vis 0.0 area render/mask 1999 2609 time 0.0015954971313476562
200 iou 0.01087102402362099 iou_vis 0.01087102402362099 area render/mask 3545 3987 time 0.001367330551147461
300 iou 0.011179758752574287 iou_vis 0.011179758752574287 area render/mask 2887 3987 time 0.0014433860778808594
400 iou 0.0013924050632911392 iou_vis 0.001392757660167131 area render/mask 3598 4313 time 0.0014412403106689453
500 iou 0.012642669007901668 iou_vis 0.0129147 …[+410 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# Run minimal inspect through existing script definitions
exec(open('/work/scratch/render_test.py').read().split('for i in [0')[0])
rgb,mask,zz=render(0); import numpy as np
y,x=torch.where(mask); print('render bbox',x.min().item(),x.max().item(),y.min().item(),y.max().item(),'centroid',x.float().mean().item(),y.float().mean().item(),'area',len(x))
with __import__('h5py').File('/work/inputs/masks.h5') as f: m=f['object'][0]
y,x=np.where(m); print('mask bbox',x.min(),x.max(),y.min(),y.max(),'centroid',x.mean(),y.mean(),'area',len(x))
# overlap metrics
print('intersection',(mask.cpu().numpy()&m).sum())
PY"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
render bbox 699 764 237 503 centroid 730.2368774414062 350.0378112792969 area 2512
mask bbox 692 758 636 929 centroid 726.3613613613613 803.1323823823824 area 3996
intersection 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/z-2.0\\*v\\/H/2.0*v\\/H-z/' /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])
0 iou 0.37743437764606264 iou_vis 0.370801317233809 area render/mask 2511 3996 time 0.10662412643432617
100 iou 0.267177196341204 iou_vis 0.267177196341204 area render/mask 2347 3610 time 0.005063772201538086
120 iou 0.4394714407502131 iou_vis 0.4394714407502131 area render/mask 2985 3769 time 0.0028977394104003906
150 iou 0.3630987581312833 iou_vis 0.3630987581312833 area render/mask 2001 2609 time 0.0017523765563964844
200 iou 0.05402379286214136 iou_vis 0.05402379286214136 area render/mask 3544 3987 time 0.0016903877258300781
300 iou 0.291862431873708 iou_vis 0.291862431873708 area render/mask 2887 3987 time 0.0013659000396728516
400 iou 0.19010377500375997 iou_vis 0.19016097 …[+609 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5','/pipeline/work/masks_k0.h5']:
 print('\\n',p)
 with h5py.File(p,'r') as f:
  def vis(name,obj):
   if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype)
  f.visititems(vis)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
aligned/front_stereo_camera_left (886,) object
alignment/front_stereo_camera_left/scale (886,) float32
alignment/front_stereo_camera_left/shift (886,) float32
alignment/front_stereo_camera_left/valid_count (886,) int32
frame_names/front_stereo_camera_left (886,) object
raw/front_stereo_camera_left (886,) object

 /pipeline/inference/materialized_inputs.h5
K_rois (886, 3, 3) float32
bboxes (886, 4) float32
identity_json () object
input_rgbs (886, 3, 224, 224) float32
input_xyz (886, 6, 224, 224) float32
observed_human_masks (886, 224, 224) bool
observed_object_masks (886, 224, 224) bool
render_rgbs (886, 3, 224, 224) float32
render_xyz (886, 6, 224, 224) float32

 /pipeline/work/masks_k0.h5
episode_000019/000000-k0.obj_rend_mask.png (1152, 1536) uint8
epis …[+108026 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 for g in f:
  print('group',g, list(f[g].keys()))
  for k in f[g]:
   o=f[g][k]; print(' ',k,o.shape,o.dtype, list(o.keys()) if isinstance(o,h5py.Group) else '')
 for i in [0,100]:
  ds=f['aligned/front_stereo_camera_left'][i]
  print(type(ds),ds.shape,ds.dtype,ds[:2,:2])
PY"
```
<details><summary>output</summary>

```
group aligned ['front_stereo_camera_left']
  front_stereo_camera_left (886,) object 
group alignment ['front_stereo_camera_left']
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
AttributeError: 'Group' object has no attribute 'shape'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 def rec(g,prefix=''):
  for k in g:
   o=g[k]
   if isinstance(o,h5py.Dataset): print(prefix+k,o.shape,o.dtype)
   else: rec(o,prefix+k+'/')
 rec(f)
 # inspect frame 0 group
 g=f['aligned/front_stereo_camera_left/000000']
 print('group keys',list(g.keys()))
 for k in g: print(k,g[k].shape,g[k].dtype, g[k][:2,:2] if g[k].ndim==2 else '')
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 407, in __getitem__
aligned/front_stereo_camera_left (886,) object
alignment/front_stereo_camera_left/scale (886,) float32
alignment/front_stereo_camera_left/shift (886,) float32
alignment/front_stereo_camera_left/valid_count (886,) int32
frame_names/front_stereo_camera_left (886,) object
raw/front_stereo_camera_left (886,) object
    return self._get(name)
           ^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 421, in _get
    oid = h5o.open(self.id, self._e( …[+339 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left','frame_names/front_stereo_camera_left']:
  ds=f[k]; print(k,ds.shape,ds.dtype)
  a=ds[0]; print(type(a),repr(a)[:500]);
  if hasattr(a,'shape'): print(a.shape,a.dtype,a[:2,:2] if a.ndim==2 else a[:20])
PY"
```
<details><summary>output</summary>

```
<class 'numpy.ndarray'> array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
(635727,) uint8 [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0]
raw/front_stereo_camera_left (886,) object
<class 'numpy.ndarray'> array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
(589749,) uint8 [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0]
frame_names/front_stereo_camera_left (886,) object
<class 'bytes'> b'000000'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/depth/moge2/depth_report.json; python - <<'PY'
import h5py,numpy as np,cv2,io,pickle
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for i in [0,100,500,800]:
  b=f['aligned/front_stereo_camera_left'][i]; im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED)
  print(i,im.shape,im.dtype,im.min(),im.max(),np.quantile(im,[.1,.5,.9]))
print(pickle.load(open('/pipeline/depth/moge2/intrinsics.pkl','rb')))
PY"
```
<details><summary>output</summary>

```
0 (1152, 1536) uint16 1721 9734 [2153. 4032. 6761.]
100 (1152, 1536) uint16 1758 10183 [2198. 4076. 6819.]
500 (1152, 1536) uint16 1719 10015 [2142. 3994. 6652.]
800 (1152, 1536) uint16 1716 9978 [2145. 4030. 6782.]
{'fx': 820.802978515625, 'fy': 820.8029174804688, 'cx': 768.0, 'cy': 576.0, 'focals': [820.802978515625, 820.9348754882812, 817.3068237304688, 819.7271118164062, 818.556640625, 813.68505859375, 814.0493774414062, 818.6054077148438, 818.9371948242188, 817.7799072265625, 818.5794067382812, 816.9885864257812, 816.0452880859375, 819.163330078125, 817.6122436523438, 818.046630859375, 820.7678833007812, 823.52734375, 824.3172607421875, 827.604736328125, 830.0552978515625, 828.00048828125, 826.387939453125, 823.078857421875, 827.3681640625, 831.32763671875, 830.898193359375, 827.59295 …[+90250 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane']
S,T=trimesh.bounds.oriented_bounds(m)
print('S\\n',S); print('T\\n',T)
# pca of vertices centered
v=np.array(m.vertices); c=v.mean(0); C=(v-c).T@(v-c)/len(v); w,V=np.linalg.eigh(C); print('cent',c,'eig',w,'V\\n',V)
print('bounds',m.bounds,'centroid mass',m.center_mass)
PY"
```
<details><summary>output</summary>

```
S
 [[ 4.08917444e-05  9.99235038e-01 -3.91067525e-02 -2.57315733e-02]
 [ 9.99999487e-01 -1.28195290e-06  1.01288777e-03  1.86593206e-03]
 [ 1.01206281e-03 -3.91067738e-02 -9.99234525e-01  7.50481524e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
T
 [0.15134054 0.20627418 0.90845043]
cent [ 0.00042009  0.00087216 -0.04103041] eig [0.00109258 0.00265329 0.10963667] V
 [[ 0.03016879  0.9995366  -0.00405368]
 [ 0.99948007 -0.03012043  0.01150437]
 [ 0.01137694 -0.00439865 -0.99992561]]
bounds [[-0.10462306 -0.06460006 -0.37794164]
 [ 0.10165078  0.08681585  0.52977979]] centroid mass [ 0.0018385  -0.00030722  0.04695566]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split('for i in [0')[0])
import cv2,h5py,numpy as np
for i in [0,100,120,200,300,400,800]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 rgb,mask,zz=render(i); a=rgb.permute(1,2,0).cpu().numpy(); a=np.clip(a*255,0,255).astype(np.uint8)
 # crop around render/mask bbox
 with h5py.File('/work/inputs/masks.h5') as f: m=f['object'][i]
 mm=mask.cpu().numpy()|m
 yy,xx=np.where(mm); pad=30;x0=max(0,xx.min()-pad);x1=min(Wd,xx.max()+pad);y0=max(0,yy.min()-pad);y1=min(H,yy.max()+pad)
 # background white; concat actual/render
 ar=im[y0:y1,x0:x1]; rr=a[y0:y1,x0:x1]
 out=np.concatenate([ar,rr],1); out=cv2.resize(out,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
 cv2.imwrite(f'/work/scratch/rgbcmp_{i}.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
PY"
```
<details><summary>output</summary>

```
V,F,uv,tex torch.Size([297734, 3]) torch.Size([99266, 3]) torch.Size([297734, 2]) torch.Size([1, 3, 2192, 2216])

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/gen_flips.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py, time, itertools
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda'; torch.manual_seed(0); np.random.seed(0)
scene=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); g=scene.geometry['goodcane']
V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev); F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev)
cent=np.array(g.vertices).mean(0); print('centroid',cent)
T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); W=np.array(cam['T_world_cam'],np.float32); Wi=np.linalg.inv(W).astype(np.float32)
Kt=torch.tensor(K,device=dev); Wri=torch.tensor(Wi[:3,:3],device=dev); Wti=torch.tensor(Wi[:3,3],device=dev)
gl=dr.RasterizeCudaContext(); H,Wd=1152,1536
# target frames
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:
  obj=f['object']; per=f['person']
frames=list(range(0,886,5)); frames=sorted(set(frames+[885]))
# transforms: all proper signed flips (det +1); index bits for sign flips x,y,z with even count
cands=[]; names=[]
for bits in itertools.product([1,-1],repeat=3):
 if np.prod(bits)==1:
  H=np.eye(4,np.float32); H[:3,:3]=np.diag(bits).astype(np.float32)
  # keep centroid fixed: T' = T @ Tc @ H @ Tc"'^-1
  shift=cent-H[:3,:3]@cent
   …[+2498 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/gen_flips.py", line 21, in <module>
    H=np.eye(4,np.float32); H[:3,:3]=np.diag(bits).astype(np.float32)
      ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/twodim_base.py", line 211, in eye
    m = zeros((N, M), dtype=dtype, order=order)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'type' object cannot be interpreted as an integer
centroid [ 0.00042009  0.00087216 -0.04103041]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/np.eye(4,np.float32)/np.eye(4,dtype=np.float32)/" /work/scratch/gen_flips.py
python /work/scratch/gen_flips.py > /work/scratch/flips.log
tail -140 /work/scratch/flips.log'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/gen_flips.py", line 59, in <module>
    m=torch.tensor(obj[fi],device=dev); p=torch.tensor(per[fi],device=dev)
                   ~~~^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 870, in __getitem__
    if self._fast_read_ok and (new_dtype is None):
       ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/base.py", line 534, in __get__
    value = obj.__dict__[self.func.__name__] = self.func(obj)
                                               ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 84 …[+994 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/gen_flips.py';s=open(p).read()
s=s.replace(\"with h5py.File('/work/inputs/masks.h5') as f:\\n  obj=f['object']; per=f['person']\", \"with h5py.File('/work/inputs/masks.h5') as f:\\n  obj=f['object'][:]; per=f['person'][:]\")
s=s.replace(\"names.append(''.join('x' if b==-1 else '' for b in bits) or 'id')\", \"names.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id')\")
open(p,'w').write(s)
PY
python /work/scratch/gen_flips.py > /work/scratch/flips.log
tail -160 /work/scratch/flips.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/gen_flips.py", line 64, in <module>
    rm=render_mask(Tc)
       ^^^^^^^^^^^^^^^
  File "/work/scratch/gen_flips.py", line 32, in render_mask
    clip=torch.stack([2*u/Wd-z,2*v/H-z,.5*z,z],1)[None]
                               ~~~^~
RuntimeError: The size of tensor a (297734) must match the size of tensor b (4) at non-singleton dimension 1
centroid [ 0.00042009  0.00087216 -0.04103041]
cands ['id', 'yz', 'xz', 'xy']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/gen_flips.py';s=open(p).read().replace('for (H,sh),name in zip(cands_t,names):','for (hmat,sh),name in zip(cands_t,names):').replace('Tc[:3,:3]=T[:3,:3]@H[:3,:3]','Tc[:3,:3]=T[:3,:3]@hmat[:3,:3]')
open(p,'w').write(s)
PY
python /work/scratch/gen_flips.py > /work/scratch/flips.log
tail -160 /work/scratch/flips.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/gen_flips.py", line 64, in <module>
    rm=render_mask(Tc)
       ^^^^^^^^^^^^^^^
  File "/work/scratch/gen_flips.py", line 32, in render_mask
    clip=torch.stack([2*u/Wd-z,2*v/H-z,.5*z,z],1)[None]
                               ~~~^~
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
centroid [ 0.00042009  0.00087216 -0.04103041]
cands ['id', 'yz', 'xz', 'xy']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '\\bH\\b' /work/scratch/gen_flips.py"
```
<details><summary>output</summary>

```
11:gl=dr.RasterizeCudaContext(); H,Wd=1152,1536
21:  H=np.eye(4,dtype=np.float32); H[:3,:3]=np.diag(bits).astype(np.float32)
22:  # keep centroid fixed: T' = T @ Tc @ H @ Tc^-1
23:  shift=cent-H[:3,:3]@cent
24:  cands.append((H,shift)); names.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id')
26:cands_t=[(torch.tensor(H,device=dev),torch.tensor(sh,device=dev,dtype=torch.float32)) for H,sh in cands]
32:  clip=torch.stack([2*u/Wd-z,2*v/H-z,.5*z,z],1)[None]
33:  rast,_=dr.rasterize(gl,clip,F,[H,Wd])

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/gen_flips.py';s=open(p).read()
s=s.replace(\"  H=np.eye(4,dtype=np.float32); H[:3,:3]=np.diag(bits).astype(np.float32)\",\"  Rm=np.eye(4,dtype=np.float32); Rm[:3,:3]=np.diag(bits).astype(np.float32)\")
s=s.replace(\"shift=cent-H[:3,:3]@cent\",\"shift=cent-Rm[:3,:3]@cent\")
s=s.replace(\"cands.append((H,shift))\",\"cands.append((Rm,shift))\")
s=s.replace(\"cands_t=[(torch.tensor(H,device=dev),torch.tensor(sh,device=dev,dtype=torch.float32)) for H,sh in cands]\",\"cands_t=[(torch.tensor(Rm,device=dev),torch.tensor(sh,device=dev,dtype=torch.float32)) for Rm,sh in cands]\")
open(p,'w').write(s)
PY
python /work/scratch/gen_flips.py > /work/scratch/flips.log
tail -180 /work/scratch/flips.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/gen_flips.py", line 65, in <module>
    row.append(stats(rm,m,p))
               ^^^^^^^^^^^^^
  File "/work/scratch/gen_flips.py", line 44, in stats
    a=cv2.resize(tgt.astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST)
                 ^^^^^^^^^^
AttributeError: 'Tensor' object has no attribute 'astype'. Did you mean: 'dtype'?
centroid [ 0.00042009  0.00087216 -0.04103041]
cands ['id', 'yz', 'xz', 'xy']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/gen_flips.py';s=open(p).read()
s=s.replace(\"  a=cv2.resize(tgt.astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST)\",\"  a=cv2.resize(tgt.cpu().numpy().astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST)\")
open(p,'w').write(s)
PY
python /work/scratch/gen_flips.py > /work/scratch/flips.log
tail -180 /work/scratch/flips.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/gen_flips.py", line 65, in <module>
    row.append(stats(rm,m,p))
               ^^^^^^^^^^^^^
  File "/work/scratch/gen_flips.py", line 54, in stats
    return iou,rec,prec,cham
           ^^^
NameError: name 'iou' is not defined
centroid [ 0.00042009  0.00087216 -0.04103041]
cands ['id', 'yz', 'xz', 'xy']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/return iou,rec,prec,cham/return inter\\/max(union,1),rec,prec,cham/' /work/scratch/gen_flips.py
python /work/scratch/gen_flips.py > /work/scratch/flips.log
tail -180 /work/scratch/flips.log"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:118: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
frame best
0 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
5 yz id:0.374 yz:0.263 xz:0.202 xy:0.259
10 id id:0.374 yz:0.263 xz:0.202 xy:0.259
15 yz id:0.374 yz:0.263 xz:0.202 xy:0.259
20 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
25 yz id:0.374 yz:0.263 xz:0.202 xy:0.259
30 yz id:0.374 yz:0.263 xz:0.202 xy:0.259
35 yz id:0.374 yz:0.263 xz:0.202 xy:0.259
40 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
45 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
50 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
55 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
60 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
65 xy id:0.374 yz:0.263 xz:0.202 xy:0.259
70 xy id:0.374 yz:0.263 xz …[+7050 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/flips.npz'); print(z['scores'].shape); s=z['scores']; print(np.unique(s[:,0,0]).size,np.unique(s[:,0,0])[:20]);print(z['frames'][:10]); print(s[:10,:,0]);print(np.nanmin(s),np.nanmax(s))
PY"
```
<details><summary>output</summary>

```
(178, 4, 4)
171 [0.         0.00177778 0.00497512 0.00857143 0.00925926 0.01782178
 0.02107376 0.02160288 0.02255639 0.02551382 0.02660054 0.03250271
 0.03544061 0.03554724 0.03798556 0.04421149 0.04751442 0.04841402
 0.05129651 0.05185418]
[ 0  5 10 15 20 25 30 35 40 45]
[[0.37441809 0.30073108 0.23644462 0.42581504]
 [0.148215   0.26309773 0.17824548 0.25720123]
 [0.22478325 0.19821461 0.20242046 0.19935627]
 [0.1710199  0.28184391 0.21630156 0.25900335]
 [0.17341166 0.27340206 0.20354339 0.28029623]
 [0.19300129 0.30177268 0.24802001 0.28491015]
 [0.19030406 0.3372093  0.25375774 0.32460372]
 [0.1972135  0.32829238 0.24692826 0.31129653]
 [0.25878268 0.27424749 0.25861729 0.28788501]
 [0.2625     0.27686447 0.26427997 0.29109159]]
0.0 inf

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/flips.npz'); fr=z['frames']; names=z['names']; S=z['scores']; iou=S[:,:,0]; best=iou.argmax(1); srt=np.sort(iou,1); margin=srt[:,-1]-srt[:,-2]
for j,(f,b) in enumerate(zip(fr,best)):
 if j%2==0 or f>780:
  print(f'{f:3d} {names[b]:2s} {iou[j,b]:.3f} sec {iou[j].max():.2f} marg {margin[j]:.3f} all',' '.join(f'{n}:{v:.2f}' for n,v in zip(names,iou[j])))
# distribution in phase ranges
for a,b in [(0,100),(105,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,800),(800,886)]:
 mask=(fr>=a)&(fr<=b); c=np.bincount(best[mask],minlength=4)
 print(a,b,'n',mask.sum(),dict(zip(names,c)),'mean iou',iou[mask].mean(0))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/view_flips.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py, itertools
import nvdiffrast.torch as dr
dev='cuda'
scene=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); g=scene.geometry['goodcane']
V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev); F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev)
cent=np.array(g.vertices).mean(0)
T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); W=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
Kt=torch.tensor(K,device=dev); Wri=torch.tensor(W[:3,:3],device=dev); Wti=torch.tensor(W[:3,3],device=dev)
gl=dr.RasterizeCudaContext(); HH,WW=1152,1536
cands=[];names=[]
for bits in itertools.product([1,-1],repeat=3):
 if np.prod(bits)==1:
  Rm=np.diag(bits).astype(np.float32); names.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id')
  shift=cent-Rm@cent; cands.append((Rm,shift))
def rend(T):
 vw=V@T[:3,:3].T+T[:3,3]; vc=vw@Wri.T+Wti;x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z
 clip=torch.stack([2*u/WW-z,2*v/HH-z,.5*z,z],1)[None]; r,_=dr.rasterize(gl,clip,F,[HH,WW]); return r[0,:,:,3].cpu().numpy()>0
with h5py.File('/work/inputs/masks.h5') as f: obj=f['object'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for fi in [0,50,100,120,150,180,200,220,270,300,350,400,430,470,500 …[+1745 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/view_flips.py", line 25, in <module>
    Tc=torch.eye(4,device=dev);Tc[:3,:3]=T[:3,:3]@torch.tensor(Rm,device=dev);Tc[:3,3]=T[:3,3]+T[:3,:3]@torch.tensor(sh,device=dev)
                                                                                               ~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: expected scalar type Float but found Double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/shift=cent-Rm@cent/shift=(cent-Rm@cent).astype(np.float32)/' /work/scratch/view_flips.py
python /work/scratch/view_flips.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/hand_analysis.py <<'PY'
import numpy as np, trimesh
from scipy.spatial import cKDTree
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh'].astype(float); K=z['human_keypoints_world'].astype(float); c=z['contact_prob_right_left']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane']; V=np.array(g.vertices)
# sample mesh vertices every 4 maybe
Vs=V[::5]
print('contact schedule high')
for hand,name,inds in [(None,'right',slice(21,42)),(None,'left',slice(42,63))]:
 p=c[:,0 if name=='right' else 1]
 on=p>.5; runs=[]; s=None
 for i,v in enumerate(on):
  if v and s is None:s=i
  if not v and s is not None:runs.append((s,i-1));s=None
 if s is not None:runs.append((s,len(on)-1))
 print(name,runs)
for i in range(0,886,20):
 Tt=T[i]; vw=Vs@Tt[:3,:3].T+Tt[:3,3]
 hr=K[i,21:42]; hl=K[i,42:63]
 # min point-to-vertex distances
 dr=np.linalg.norm(hr[:,None]-vw[None],axis=2); dl=np.linalg.norm(hl[:,None]-vw[None],axis=2)
 ir=np.unravel_index(np.argmin(dr),dr.shape);il=np.unravel_index(np.argmin(dl),dl.shape)
 print(f'{i:3d} R {dr[ir]:.3f} kp{ir[0]} obj{ir[1]*5} L {dl[il]:.3f} kp{il[0]} obj{il[1]*5} cp {c[i,0]:.2f}/{c[i,1]:.2f}')
PY
python /work/scratch/hand_analysis.py"
```
<details><summary>output</summary>

```
contact schedule high
right [(113, 117), (139, 139), (141, 147), (191, 191), (390, 426), (429, 432), (439, 479), (483, 557), (560, 560), (562, 599), (603, 604), (606, 791)]
left [(0, 12), (86, 159), (163, 424), (453, 453), (505, 505), (528, 528), (534, 534), (536, 536), (539, 557), (591, 591), (601, 605), (607, 609), (613, 617), (628, 640), (728, 728), (737, 743), (745, 746), (748, 791)]
  0 R 0.106 kp10 obj136890 L 0.557 kp0 obj162205 cp 0.00/1.00
 20 R 0.051 kp9 obj37000 L 0.456 kp0 obj270305 cp 0.00/0.03
 40 R 0.191 kp18 obj221245 L 0.583 kp0 obj116220 cp 0.00/0.00
 60 R 0.247 kp11 obj60695 L 0.564 kp0 obj46630 cp 0.00/0.00
 80 R 0.057 kp15 obj297565 L 0.428 kp0 obj136110 cp 0.00/0.00
100 R 0.209 kp19 obj260055 L 0.364 kp0 obj51495 cp 0.00/1.00
120 R 0.011 kp19 obj277650 L 0.429 kp2 obj …[+2331 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,trimesh,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));W=np.array(cam['T_world_cam']);Wi=np.linalg.inv(W);K=np.array(cam['K'])
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=np.array(g.vertices)[::10]
with h5py.File('/work/inputs/masks.h5') as f: obj=f['object'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for i in [0,50,100,120,200,300,400,500,600,700,800,850,880]:
  b=f['aligned/front_stereo_camera_left'][i]; d=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED).astype(float)/1000
  m=obj[i]; ys,xs=np.where(m)
  dm=d[ys,xs]
  Tt=T[i]; vc=(V@Tt[:3,:3].T+Tt[:3,3])@Wi[:3,:3].T+Wi[:3,3]
  uv=vc@K.T;uv=uv[:,:2]/uv[:,2:3]; ok=(uv[:,0]>=0)&(uv[:,0]<1536)&(uv[:,1]>=0)&(uv[:,1]<1152); uu=uv[ok]
  dep=vc[ok,2]
  print(i,'mask depth q',np.quantile(dm,[.1,.25,.5,.75,.9]),'base obj depth q',np.quantile(dep,[.1,.5,.9]))
PY"
```
<details><summary>output</summary>

```
0 mask depth q [2.575 2.596 2.666 2.787 2.94 ] base obj depth q [2.51419733 2.66422637 3.03449022]
50 mask depth q [2.558   2.63    2.757   2.90375 3.489  ] base obj depth q [2.49943341 2.65285531 2.95854508]
100 mask depth q [2.609 2.664 2.769 2.898 3.311] base obj depth q [2.4982447  2.6516666  2.95735637]
120 mask depth q [2.642  2.692  2.79   2.962  3.5706] base obj depth q [2.47443239 2.58444517 2.7413987 ]
200 mask depth q [4.8676 5.057  5.301  5.638  6.3622] base obj depth q [2.27187933 2.37033548 2.59856902]
300 mask depth q [3.225  3.3815 3.547  4.178  5.0608] base obj depth q [2.47265673 2.58421015 2.86567307]
400 mask depth q [2.726 2.746 2.833 2.928 3.04 ] base obj depth q [2.22943432 2.32976763 2.43201619]
500 mask depth q [3.968 4.173 4.617 5.314 6.326] base obj depth q [2.75 …[+565 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/frames=list(range(0,886,5)); frames=sorted(set(frames+\\[885\\]))/frames=list(range(886))/; s#flips.npz#flips_all.npz#' /work/scratch/gen_flips.py > /work/scratch/gen_flips_all.py
python /work/scratch/gen_flips_all.py > /work/scratch/flips_all.log
tail -20 /work/scratch/flips_all.log"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:118: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
867 id id:0.374 yz:0.312 xz:0.193 xy:0.307
868 id id:0.374 yz:0.312 xz:0.193 xy:0.307
869 id id:0.374 yz:0.312 xz:0.193 xy:0.307
870 id id:0.374 yz:0.312 xz:0.193 xy:0.307
871 id id:0.374 yz:0.312 xz:0.193 xy:0.307
872 id id:0.374 yz:0.312 xz:0.193 xy:0.307
873 id id:0.374 yz:0.312 xz:0.193 xy:0.307
874 id id:0.374 yz:0.312 xz:0.193 xy:0.307
875 id id:0.374 yz:0.312 xz:0.193 xy:0.307
876 id id:0.374 yz:0.312 xz:0.193 xy:0.307
877 id id:0.374 yz:0.312 xz:0.193 xy:0.307
878 id id:0.374 yz:0.312 xz:0.193 xy:0.307
879 id id:0.374 yz:0.312 xz:0.193 xy:0.307
880 id id:0.374 yz:0.312 xz:0.193 xy:0.307
881 id id:0.374 yz:0. …[+224 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/viterbi.py <<'PY'
import numpy as np, itertools
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/flips_all.npz'); S=z['scores']; names=z['names']; iou=S[:,:,0]
raw=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
# same candidate mesh transforms, centroid
# recompute local transforms and center correction
import trimesh
cent=np.array(trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'].vertices).mean(0)
Hs=[];names2=[]
for bits in itertools.product([1,-1],repeat=3):
 if np.prod(bits)==1:
  Rm=np.diag(bits).astype(np.float32); names2.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id');Hs.append(Rm)
assert names2==list(names)
def cand_pose(i,k):
 Rm=Hs[k]; T=raw[i].copy(); sh=cent-Rm@cent; T[:3,:3]=T[:3,:3]@Rm; T[:3,3]=T[:3,3]+raw[i,:3,:3]@sh; return T
poses=np.stack([[cand_pose(i,k) for k in range(4)] for i in range(len(raw))]) # i,k,4,4
# transition cost per pair
rot=np.zeros((len(raw),4,4)); trans=np.zeros_like(rot)
for i in range(1,len(raw)):
 for a in range(4):
  for b in range(4):
   rel=poses[i,a,:3,:3].T@poses[i-1,b,:3,:3]
   rot[i,a,b]=R.from_matrix(rel).magnitude()*180/np.pi
   trans[i,a,b]=np.linalg.norm(poses[i,a,:3,3]-poses[i-1,b,:3,3])
for lam in [0,.001,.003,.005,.01,.02,.05,.1]:
 for lt in [.5]:
  dp=np.full((len(raw),4),np.inf); back=np.zeros((len(raw),4),np.int32); dp[0]=-100*iou[0]
  for i in range(1,len(raw)):
   cost=-1 …[+811 chars]
```
<details><summary>output</summary>

```
lam 0 cost -22048.248263560443 mean iou 0.24904006103708304 med rot 5.143593740387417 p90 rot 178.06906437298935 med tr 0.02983753144358703 p90 tr 0.08559808104569552 counts [435  61  41 349]
 runs [(0, 3, 'xy'), (4, 5, 'yz'), (6, 6, 'xy'), (7, 8, 'yz'), (9, 10, 'id'), (11, 11, 'xy'), (12, 18, 'yz'), (19, 22, 'xy'), (23, 25, 'yz'), (26, 26, 'xy'), (27, 33, 'yz'), (34, 34, 'xy'), (35, 35, 'yz'), (36, 36, 'xy'), (37, 37, 'yz'), (38, 100, 'xy'), (101, 123, 'id'), (124, 126, 'xy'), (127, 134, 'id'), (135, 135, 'xy'), (136, 156, 'id'), (157, 164, 'xy'), (165, 165, 'xz'), (166, 167, 'id'), (168, 170, 'yz'), (171, 172, 'xy'), (173, 190, 'id'), (191, 191, 'xy'), (192, 192, 'id'), (193, 194, 'yz'), (195, 195, 'id'), (196, 196, 'xz'), (197, 199, 'id'), (200, 200, 'xy'), (201, 201, 'xz'), (202, 203,  …[+10181 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
q=R.from_matrix(T[:,:3,:3]).as_quat() # x,y,z,w
# unwrap each quaternion component sign
for i in range(1,len(q)):
 if np.dot(q[i],q[i-1])<0:q[i]*=-1
rv=R.from_matrix(T[:,:3,:3]).as_rotvec()*180/np.pi
t=T[:,:3,3]
fig,axs=plt.subplots(5,1,figsize=(18,12),sharex=True)
for k in range(3):axs[0].plot(rv[:,k],label='xyz'[k]);axs[0].set_ylabel('rotvec deg');axs[0].legend()
for k in range(4):axs[1].plot(q[:,k],label='xyzw'[k]);axs[1].set_ylabel('quat');axs[1].legend()
for k in range(3):axs[2].plot(t[:,k],label='xyz'[k]);axs[2].set_ylabel('trans');axs[2].legend()
# relative delta log in local frame
rel=np.swapaxes(T[:-1,:3,:3],1,2)@T[1:,:3,:3]
dd=R.from_matrix(rel).as_rotvec()*180/np.pi
for k in range(3):axs[3].plot(dd[:,k],label='xyz'[k]);axs[3].set_ylabel('local delta');axs[3].legend();axs[3].set_ylim(-120,120)
axs[4].plot(np.linalg.norm(np.diff(t,axis=0),axis=1));axs[4].set_ylabel('speed m/f');axs[4].set_xlabel('frame')
for ax in axs:ax.grid(True,alpha=.3);ax.axvspan(0,105,color='g',alpha=.1);ax.axvspan(800,885,color='g',alpha=.1)
plt.tight_layout();plt.savefig('/work/scratch/trajplot.png',dpi=100)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_smooth.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py, time
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import median_filter, gaussian_filter1d, uniform_filter1d
from scipy.signal import savgol_filter
dev='cuda'; torch.manual_seed(0); np.random.seed(0)
scene=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); g=scene.geometry['goodcane']
V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev); F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev)
raw=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); W=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
Kt=torch.tensor(K,device=dev); Wri=torch.tensor(W[:3,:3],device=dev); Wti=torch.tensor(W[:3,3],device=dev)
gl=dr.RasterizeCudaContext(); HH,WW=1152,1536
with h5py.File('/work/inputs/masks.h5') as f: obj=f['object'][:]; per=f['person'][:]
# no need load whole? above lo
q=R.from_matrix(raw[:,:3,:3]).as_quat().astype(np.float64)
for i in range(1,len(q)):
 if q[i]@q[i-1]<0:q[i]*=-1
t=raw[:,:3,3].astype(np.float64)
# static pooled averages
q_static0=q[40:101].copy();q_static0[ np.linalg.norm(q_static0,axis=1)<.5 ]=q[50]
# align flips? average naive
q_static0 /= np.linalg.norm(q_static0,axis=1,keepdims=True); qs0=q_static0.mean(0);qs0/=np.linalg.n …[+2462 chars]
```
<details><summary>output</summary>

```
static q [ 0.36374821 -0.33319146  0.6007031   0.62914742] [-0.4862619  -0.46746552  0.5065401  -0.53706841] t [0.03939909 0.81315041 2.92832804] [0.20645963 0.77906078 3.80895281]
Traceback (most recent call last):
  File "/work/scratch/eval_smooth.py", line 54, in <module>
    rm=render_mask(T[i]); m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev)
       ^^^^^^^^^^^^^^^^^
  File "/work/scratch/eval_smooth.py", line 44, in render_mask
    vw=V@T[:3,:3].T+T[:3,3];vc=vw@Wri.T+Wti;x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z
       ~^~~~~~~~~~~
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu( …[+43 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_smooth.py';s=open(p).read();s=s.replace(\"def render_mask(T):\\n vw=V@T[:3,:3].T+T[:3,3]\",\"def render_mask(T):\\n T=torch.as_tensor(T,device=dev,dtype=torch.float32)\\n vw=V@T[:3,:3].T+T[:3,3]\")
open(p,'w').write(s)
PY
python /work/scratch/eval_smooth.py"
```
<details><summary>output</summary>

```
static q [ 0.36374821 -0.33319146  0.6007031   0.62914742] [-0.4862619  -0.46746552  0.5065401  -0.53706841] t [0.03939909 0.81315041 2.92832804] [0.20645963 0.77906078 3.80895281]
raw all 0.22334812421665737 ph0 0.23998361303809101 dyn 0.21937363368229992 ph1 0.2359266919881858 q10/50/90 [0.03190895 0.22899206 0.39757019]
sg5 all 0.24101383316801164 ph0 0.268257198739973 dyn 0.2202139783420719 ph1 0.38343511534210845 q10/50/90 [0.04167667 0.26089272 0.41128887]
sg7 all 0.24173198714343963 ph0 0.268257198739973 dyn 0.2211242564854613 ph1 0.38343511534210845 q10/50/90 [0.04089578 0.26140214 0.41698988]
sg9 all 0.2438451091991172 ph0 0.268257198739973 dyn 0.22380269159466062 ph1 0.38343511534210845 q10/50/90 [0.04063109 0.26226445 0.42106533]
sg11 all 0.24564146785683086 ph0 0.26825719873997 …[+2513 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# create smooth contact overlay using saved sg9
import numpy as np, torch, trimesh,cv2,json,h5py
import nvdiffrast.torch as dr
dev='cuda';z=np.load('/work/scratch/smooth_iou.npz');T=z['Ts']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev)
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);W=np.linalg.inv(np.array(c['T_world_cam'],np.float32)).astype(np.float32);Kt=torch.tensor(K,device=dev);Wr=torch.tensor(W[:3,:3],device=dev);Wt=torch.tensor(W[:3,3],device=dev);gl=dr.RasterizeCudaContext();HH,WW=1152,1536
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');tiles=[]
for fi in range(0,886,25):
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 Tt=torch.tensor(T[fi],device=dev);vw=V@Tt[:3,:3].T+Tt[:3,3];vc=vw@Wr.T+Wt;x,y,zc=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*zc;v=Kt[1,1]*y+Kt[1,2]*zc;clip=torch.stack([2*u/WW-zc,2*v/HH-zc,.5*zc,zc],1)[None];r,_=dr.rasterize(gl,clip,F,[HH,WW]);rm=r[0,:,:,3].cpu().numpy()>0
 o=im.copy();o[rm]=(0.5*o[rm]+0.5*np.array([0,255,0])).astype(np.uint8);cnt,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(o,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(objs[fi].astype(np.uint …[+377 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/pipeline/foundationpose/episode_000019.pkl','rb'))
print(type(p))
if isinstance(p,dict):
 for k,v in p.items(): print(k,type(v),getattr(v,'shape',None),repr(v)[:300])
else: print(repr(p)[:2000])
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
frames <class 'list'> None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029',
camera_id <class 'int'> None 0
fp_poses_cam <class 'numpy.ndarray'> (886, 1, 4, 4) array([[[[-0.8401607 , -0.53544855,  0.08616766, -0.16491516],
         [-0.43414253,  0.7592303 ,  0.48486024,  0.8333944 ],
         [-0.32503888,  0.36995146, -0.8703364 ,  3.0154147 ],
         [ 0.        ,  0.        ,  0.        ,  1.        ]]],


       [[[-0.80986005, -0.5822089 ,  0.07182
fp_candidate_count <class 'numpy.ndarray'> (886,) array([252,   1,   1,   …[+11234 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_fp.py <<'PY'
import numpy as np,torch,trimesh,cv2,json,h5py
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda';z=np.load('/work/inputs/reconstruction.npz');base=z['T_world_mesh'].astype(np.float32);fpc=z['T_cam_mesh_foundationpose'].astype(np.float32);coc=z['T_cam_mesh_coconet'].astype(np.float32)
c=json.load(open('/work/inputs/camera.json'));W=np.array(c['T_world_cam'],np.float32);Wi=np.linalg.inv(W).astype(np.float32);K=np.array(c['K'],np.float32)
fpw=W@fpc; cocw=W@coc
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev)
Kt=torch.tensor(K,device=dev);Wri=torch.tensor(Wi[:3,:3],device=dev);Wti=torch.tensor(Wi[:3,3],device=dev);gl=dr.RasterizeCudaContext();HH,WW=1152,1536
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:];per=f['person'][:]
def rend(T):
 vw=V@T[:3,:3].T+T[:3,3];vc=vw@Wri.T+Wti;x,y,zc=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*zc;v=Kt[1,1]*y+Kt[1,2]*zc;clip=torch.stack([2*u/WW-zc,2*v/HH-zc,.5*zc,zc],1)[None];r,_=dr.rasterize(gl,clip,F,[HH,WW]);return r[0,:,:,3]>0
for name,A in [('base',base),('fpw',fpw),('cocw',cocw)]:
 io=[]
 for i in range(len(A)):
  rm=rend(torch.tensor(A[i],device=dev));m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev);rv=rm&~p;inter=(rv&m).sum().item();uni=(rv|m …[+1353 chars]
```
<details><summary>output</summary>

```
base 0.22334812421665737 [0.03190895 0.22899206 0.39757019] ph0 0.23998361303809101 dyn 0.21937363368229992 ph1 0.2359266919881858
fpw 0.09476440079571415 [0.02602873 0.07022391 0.2104819 ] ph0 0.05357657203654555 dyn 0.10189362966145679 ph1 0.08673258424155066
cocw 0.10598750783346307 [0.02324759 0.08226193 0.2276394 ] ph0 0.17772373907310138 dyn 0.1038873532097588 ph1 0.03203267614818593
Traceback (most recent call last):
  File "/work/scratch/eval_fp.py", line 21, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);rm=rend(torch.tensor(fpw[fi],device=dev)).cpu().numpy();m=obj[fi];mm=rm|m;yy,xx=np.where(mm);pad=30;x0=max(0,xx.min()-pad);x1=min(WW,xx.max()+pad);y0=max(0,yy.min()-pad);y1=min(HH,yy.max()+pad);o=im.copy();o[rm]=(.5*o[rm] …[+1451 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_rgb.py <<'PY'
import numpy as np,torch,trimesh,cv2,json,h5py,itertools
import nvdiffrast.torch as dr
dev='cuda';z=np.load('/work/inputs/reconstruction.npz');raw=z['T_world_mesh'].astype(np.float32)
c=json.load(open('/work/inputs/camera.json'));W=np.linalg.inv(np.array(c['T_world_cam'],np.float32)).astype(np.float32);K=np.array(c['K'],np.float32)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev);uv=torch.tensor(np.asarray(g.visual.uv),dtype=torch.float32,device=dev)
tex=torch.tensor(np.array(g.visual.material.baseColorTexture),dtype=torch.float32,device=dev).permute(2,0,1)[None]/255
cent=np.array(g.vertices).mean(0)
Kt=torch.tensor(K,device=dev);Wri=torch.tensor(W[:3,:3],device=dev);Wti=torch.tensor(W[:3,3],device=dev);gl=dr.RasterizeCudaContext();HH,WW=1152,1536
cands=[];names=[]
for bits in itertools.product([1,-1],repeat=3):
 if np.prod(bits)==1:
  Rm=np.diag(bits).astype(np.float32);names.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id');cands.append((Rm,(cent-Rm@cent).astype(np.float32)))
def rend(T):
 vw=V@T[:3,:3].T+T[:3,3];vc=vw@Wri.T+Wti;x,y,zc=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*zc;v=Kt[1,1]*y+Kt[1,2]*zc;clip=torch.stack([2*u/WW-zc,2*v/HH-zc,.5*zc,zc],1)[None];r,_=dr.rasterize(gl,clip,F,[HH,WW]);vul,_=dr.interpolate(uv[None],r,F);rgb=tor …[+1899 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_rgb.py", line 24, in <module>
    Tc=torch.eye(4,device=dev);Tc[:3,:3]=T[:3,:3]@torch.tensor(Rm,device=dev);Tc[:3,3]=T[:3,3]+T[:3,:3]@torch.tensor(sh,device=dev);name=names[cands.index((Rm,sh))]
                                                                                                                                               ^^^^^^^^^^^^^^^^^^^^
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

frame 0
 raw iou 0.377 n  1305 l1 0.211 lum 0.187 corr -0.070
 id iou 0.377 n  1305 l1 0.211 lum 0.187 corr -0.070

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_rgb.py';s=open(p).read().replace(\" for Rm,sh in [(None,None)]+cands:\",\" for ci,(Rm,sh) in enumerate([(None,None)]+cands):\").replace(\"name=names[cands.index((Rm,sh))]\",\"name=names[ci-1]\")
open(p,'w').write(s)
PY
python /work/scratch/eval_rgb.py > /work/scratch/rgb.log
sed -n '1,340p' /work/scratch/rgb.log"
```
<details><summary>output</summary>

```

frame 0
 raw iou 0.377 n  1305 l1 0.211 lum 0.187 corr -0.070
 id iou 0.377 n  1305 l1 0.211 lum 0.187 corr -0.070
 yz iou 0.302 n  1166 l1 0.201 lum 0.175 corr -0.068
 xz iou 0.247 n   918 l1 0.194 lum 0.160 corr 0.004
 xy iou 0.425 n  1590 l1 0.208 lum 0.184 corr -0.105

frame 20
 raw iou 0.145 n   522 l1 0.188 lum 0.156 corr 0.033
 id iou 0.145 n   522 l1 0.188 lum 0.156 corr 0.033
 yz iou 0.251 n  1058 l1 0.203 lum 0.174 corr 0.019
 xz iou 0.181 n   753 l1 0.209 lum 0.181 corr -0.094
 xy iou 0.249 n   925 l1 0.192 lum 0.165 corr 0.056

frame 50
 raw iou 0.262 n  1031 l1 0.195 lum 0.167 corr -0.037
 id iou 0.262 n  1031 l1 0.195 lum 0.167 corr -0.037
 yz iou 0.278 n  1205 l1 0.210 lum 0.188 corr -0.086
 xz iou 0.264 n  1138 l1 0.212 lum 0.183 corr -0.003
 xy iou 0.292 n  1140 l1 0.196  …[+7191 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/canonical.py <<'PY'
import numpy as np,torch,trimesh,cv2,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda';g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev);uv=torch.tensor(np.asarray(g.visual.uv),dtype=torch.float32,device=dev);tex=torch.tensor(np.array(g.visual.material.baseColorTexture),dtype=torch.float32,device=dev).permute(2,0,1)[None]/255
# normalize to centered at centroid
V=V-torch.tensor(np.array(g.vertices).mean(0),dtype=torch.float32,device=dev)
gl=dr.RasterizeCudaContext();H=W=800;K=torch.tensor([[900,0,400],[0,900,400],[0,0,1]],dtype=torch.float32,device=dev)
out=[]
for name,Rm in [('front',np.eye(3)),('x90',R.from_euler('x',90,degrees=True).as_matrix()),('y90',R.from_euler('y',90,degrees=True).as_matrix()),('x180',R.from_euler('x',180,degrees=True).as_matrix()),('y180',R.from_euler('y',180,degrees=True).as_matrix()),('z90',R.from_euler('z',90,degrees=True).as_matrix())]:
 Rm=torch.tensor(Rm,dtype=torch.float32,device=dev);vv=V@Rm.T;vv=vv+torch.tensor([0,0,2.5],device=dev);x,y,z=vv.unbind(1);u=K[0,0]*x+K[0,2]*z;v=K[1,1]*y+K[1,2]*z;clip=torch.stack([2*u/W-z,2*v/H-z,.5*z,z],1)[None];r,_=dr.rasterize(gl,clip,F,[H,W]);vul,_=dr.interpolate(uv[None],r,F);rgb=torch.nn.functional.grid_sample(tex,vul*2-1,align_ …[+416 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,trimesh.visual.texture as t
print(inspect.getsource(t.TextureVisuals.to_color))
PY"
```
<details><summary>output</summary>

```
    def to_color(self):
        """
        Convert textured visuals to a ColorVisuals with vertex
        color calculated from texture.

        Returns
        -----------
        vis : trimesh.visuals.ColorVisuals
          Contains vertex color from texture
        """
        # find the color at each UV coordinate
        colors = self.material.to_color(self.uv)
        # create ColorVisuals from result
        vis = color.ColorVisuals(vertex_colors=colors)
        return vis


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,trimesh.visual.material as m
print(inspect.getsource(m.PBRMaterial.to_color))
PY"
```
<details><summary>output</summary>

```
    def to_color(self, uv):
        """
        Get the rough color at a list of specified UV
        coordinates.

        Parameters
        -------------
        uv : (n, 2) float
          UV coordinates on the material

        Returns
        -------------
        colors
        """
        colors = color.uv_to_color(uv=uv, image=self.baseColorTexture)
        if colors is None and self.baseColorFactor is not None:
            colors = self.baseColorFactor.copy()
        return colors


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,trimesh.visual.color as c
print(inspect.getsource(c.uv_to_color))
PY"
```
<details><summary>output</summary>

```
def uv_to_color(uv, image) -> NDArray[np.uint8]:
    """
    Get the color in a texture image.

    Parameters
    -------------
    uv : (n, 2) float
      UV coordinates on texture image
    image : PIL.Image
      Texture image

    Returns
    ----------
    colors : (n, 4) uint4
      RGBA color at each of the UV coordinates
    """
    if image is None or uv is None:
        return None

    # UV coordinates should be (n, 2) float
    uv = np.asanyarray(uv, dtype=np.float64)

    # get texture image pixel positions of UV coordinates
    x = (uv[:, 0] * (image.width - 1)) % image.width
    y = ((1 - uv[:, 1]) * (image.height - 1)) % image.height

    # access colors from pixel locations
    # make sure image is RGBA before getting values
    colors = np.asanyarray(image.convert("RGBA" …[+274 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_rgb.py';s=open(p).read()
s=s.replace(\"rgb=torch.nn.functional.grid_sample(tex,vul*2-1,align_corners=False,mode='bilinear',padding_mode='border')[0]\",\"uvg=torch.stack([vul[...,0]*2-1,1-vul[...,1]*2],-1); rgb=torch.nn.functional.grid_sample(tex,uvg,align_corners=True,mode='bilinear',padding_mode='border')[0]\")
open(p,'w').write(s)
p='/work/scratch/render_test.py';s=open(p).read();s=s.replace(\"uv_grid=vul*2-1\\n    rgb=torch.nn.functional.grid_sample(tex,uv_grid,align_corners=False,mode='bilinear',padding_mode='border')[0]\",\"uv_grid=torch.stack([vul[...,0]*2-1,1-vul[...,1]*2],-1)\\n    rgb=torch.nn.functional.grid_sample(tex,uv_grid,align_corners=True,mode='bilinear',padding_mode='border')[0]\")
open(p,'w').write(s)
PY
python /work/scratch/eval_rgb.py > /work/scratch/rgb2.log
sed -n '1,340p' /work/scratch/rgb2.log"
```
<details><summary>output</summary>

```

frame 0
 raw iou 0.377 n  1305 l1 0.196 lum 0.165 corr -0.042
 id iou 0.377 n  1305 l1 0.196 lum 0.165 corr -0.042
 yz iou 0.302 n  1166 l1 0.227 lum 0.204 corr 0.042
 xz iou 0.247 n   918 l1 0.224 lum 0.199 corr 0.127
 xy iou 0.425 n  1590 l1 0.206 lum 0.181 corr -0.053

frame 20
 raw iou 0.145 n   522 l1 0.178 lum 0.142 corr -0.076
 id iou 0.145 n   522 l1 0.178 lum 0.142 corr -0.076
 yz iou 0.251 n  1058 l1 0.225 lum 0.196 corr 0.081
 xz iou 0.181 n   753 l1 0.222 lum 0.195 corr 0.119
 xy iou 0.249 n   925 l1 0.215 lum 0.185 corr -0.132

frame 50
 raw iou 0.262 n  1031 l1 0.223 lum 0.196 corr -0.213
 id iou 0.262 n  1031 l1 0.223 lum 0.196 corr -0.213
 yz iou 0.278 n  1205 l1 0.222 lum 0.192 corr 0.041
 xz iou 0.264 n  1138 l1 0.214 lum 0.178 corr 0.042
 xy iou 0.292 n  1140 l1 0.215 l …[+7176 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_pose.py <<'PY'
import numpy as np,torch,trimesh,cv2,json,h5py,itertools,time,sys
import nvdiffrast.torch as dr
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
dev='cuda';np.random.seed(0);torch.manual_seed(0)
z=np.load('/work/inputs/reconstruction.npz');raw=z['T_world_mesh'].astype(np.float32)
c=json.load(open('/work/inputs/camera.json'));W=np.array(c['T_world_cam'],np.float32);Wi=np.linalg.inv(W).astype(np.float32);K=np.array(c['K'],np.float32)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];Vt=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);Ft=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev);cent=np.array(g.vertices).mean(0)
Kt=torch.tensor(K,device=dev);Wri=torch.tensor(Wi[:3,:3],device=dev);Wti=torch.tensor(Wi[:3,3],device=dev);gl=dr.RasterizeCudaContext()
SS=2;HH,WW=1152//SS,1536//SS
with h5py.File('/work/inputs/masks.h5') as f: objs=f['object'][:];pers=f['person'][:]
Hs=[];names=[]
for bits in itertools.product([1,-1],repeat=3):
 if np.prod(bits)==1:
  Rm=np.diag(bits).astype(np.float32);names.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id');Hs.append(Rm)

def rend(T):
 vw=Vt@T[:3,:3].T+T[:3,3];vc=vw@Wri.T+Wti;x,y,zc=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*zc;v=Kt[1,1]*y+Kt[1,2]*zc;clip=torch.stack([2*u/WW-zc,2*v/HH-zc,.5*zc,zc],1)[None];r,_=dr.rasterize(gl,clip,Ft,[HH,WW]);return (r[0,:,:,3]>0).cpu().nu …[+2663 chars]
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:118: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2357: RuntimeWarning: invalid value encountered in scalar multiply
  q = (xf - fulc) * (fx - fnfc)
frame 300
id fun 2.8038109741210935 iou 0.0 x [ -0.12   -0.117  -0.12   24.399 -24.904 -24.979] nfev 213 True sec 1.6460375785827637
yz fun 2.8463663330078126 iou 0.0 x [ -0.12   -0.119  -0.111  22.501 -23.972 -24.898] nfev 425 True sec 2.985236883163452
xz fun 2.862056640625 iou 0.0 x [ -0.119  -0.119  -0.116  23.121 -24.443 -24.747] nfev 551 True sec 3.821125030517578
xy fun 2.8426661376953124 iou 0.0 x [ -0.12   -0.118  -0.12   24.63  -24.932 -24.99 …[+40 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# load definitions without running
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
fi=300
tm=objs[fi].astype(np.uint8);tm=cv2.resize(tm,(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pers[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool)
for ci,(Rm,sh) in enumerate(zip(Hs,[cent-Rm@cent for Rm in Hs])):
 Ti=np.eye(4,dtype=np.float32);Ti[:3,:3]=raw[fi,:3,:3]@Rm;Ti[:3,3]=raw[fi,:3,3]+raw[fi,:3,:3]@sh;Ti=torch.tensor(Ti,device=dev)
 def make(x):
  Tt=Ti.clone();dw=torch.tensor(W[:3,:3],device=dev)@torch.tensor(x[:3],device=dev,dtype=torch.float32);Tt[:3,3]+=dw
  Rd=R.from_rotvec(np.deg2rad(x[3:])).as_matrix().astype(np.float32);Rd=torch.tensor(Rd,device=dev);Rcam=torch.tensor(Wi[:3,:3],device=dev)@Tt[:3,:3];Rcam=Rd@Rcam;Tt[:3,:3]=torch.tensor(W[:3,:3],device=dev)@Rcam;return Tt
 for x in [np.zeros(6),np.array([0.,0.,0.,1,0,0]),np.array([.05,0,0,0,0,0])]:
  rm=rend(make(x));l,iou=loss_mask(rm,tm,pm);print(names[ci],x,'area',rm.sum(),'iou',iou,'loss',l)
PY"
```
<details><summary>output</summary>

```
id [0. 0. 0. 0. 0. 0.] area 552 iou 0.0 loss 3.6537495727539064
id [0. 0. 0. 1. 0. 0.] area 569 iou 0.0 loss 3.6512635498046877
id [0.05 0.   0.   0.   0.   0.  ] area 538 iou 0.0 loss 3.736890625
yz [0. 0. 0. 0. 0. 0.] area 24 iou 0.0 loss 3.507127685546875
yz [0. 0. 0. 1. 0. 0.] area 31 iou 0.0 loss 3.5006468505859374
yz [0.05 0.   0.   0.   0.   0.  ] area 24 iou 0.0 loss 3.592670166015625
xz [0. 0. 0. 0. 0. 0.] area 40 iou 0.0 loss 3.523743408203125
xz [0. 0. 0. 1. 0. 0.] area 47 iou 0.0 loss 3.5173006591796874
xz [0.05 0.   0.   0.   0.   0.  ] area 42 iou 0.0 loss 3.6062265625
xy [0. 0. 0. 0. 0. 0.] area 621 iou 0.0 loss 3.66939794921875
xy [0. 0. 0. 1. 0. 0.] area 622 iou 0.0 loss 3.6662609863281252
xy [0.05 0.   0.   0.   0.   0.  ] area 618 iou 0.0 loss 3.7520059814453126

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
fi=300;tm=cv2.resize(objs[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool)
T=torch.tensor(raw[fi],device=dev);rm=rend(T)
y,x=np.where(rm);print('render',x.min(),x.max(),y.min(),y.max(),x.mean(),y.mean(),len(x));y,x=np.where(tm);print('target',x.min(),x.max(),y.min(),y.max(),x.mean(),y.mean(),len(x));print('inter',(rm&tm).sum())
PY"
```
<details><summary>output</summary>

```
render 696 724 521 575 711.856884057971 546.5289855072464 552
target 274 357 266 397 316.7056451612903 343.21774193548384 992
inter 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'][300];g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=np.array(g.vertices);c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);W=np.linalg.inv(np.array(c['T_world_cam']));vc=(V@T[:3,:3].T+T[:3,3])@W[:3,:3].T+W[:3,3];uv=vc@K.T;uv=uv[:,:2]/uv[:,2:3];print(uv.min(0),uv.max(0),uv.mean(0))
PY"
```
<details><summary>output</summary>

```
[552.6372846  521.00370099] [725.25801994 790.34609167] [637.01333326 680.57666358]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
print('HWW',HH,WW)
T=torch.tensor(raw[300],device=dev);vw=Vt@T[:3,:3].T+T[:3,3];vc=vw@Wri.T+Wti;x,y,zc=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*zc;v=Kt[1,1]*y+Kt[1,2]*zc;print('uv', (u/zc).min(),(u/zc).max(),(v/zc).min(),(v/zc).max());clip=torch.stack([2*u/WW-zc,2*v/HH-zc,.5*zc,zc],1)[None];print('ndc', (clip[...,0]/clip[...,3]).min(),(clip[...,0]/clip[...,3]).max(),(clip[...,1]/clip[...,3]).min(),(clip[...,1]/clip[...,3]).max())
# save r
r,_=dr.rasterize(gl,clip,Ft,[HH,WW]);print(r.shape,(r[0,:,:,3]>0).sum())
PY"
```
<details><summary>output</summary>

```
HWW 576 768
uv tensor(552.6373, device='cuda:0') tensor(725.2581, device='cuda:0') tensor(521.0037, device='cuda:0') tensor(790.3461, device='cuda:0')
ndc tensor(0.4392, device='cuda:0') tensor(0.8887, device='cuda:0') tensor(0.8090, device='cuda:0') tensor(1.7443, device='cuda:0')
torch.Size([1, 576, 768, 4]) tensor(552, device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_pose.py';s=open(p).read()
s=s.replace(\"SS=2;HH,WW=1152//SS,1536//SS\",\"SS=2;H0,W0=1152,1536;HH,WW=H0//SS,W0//SS\")
s=s.replace(\"2*u/WW-zc,2*v/HH-zc\", \"2*u/W0-zc,2*v/H0-zc\")
open(p,'w').write(s)
PY
python /work/scratch/opt_pose.py 300"
```
<details><summary>output</summary>

```
frame 300
id fun 0.5560843559106191 iou 0.45166666666666666 x [ 0.012  0.01  -0.066 -4.567  1.979 -0.879] nfev 266 True sec 3.2847900390625
yz fun 0.7037799223864352 iou 0.3222486144101346 x [ 1.7000e-02 -2.8000e-02  8.5000e-02  2.4147e+01 -2.1000e-01  3.7390e+00] nfev 636 True sec 5.3538782596588135
xz fun 0.7660047392003677 iou 0.26691176470588235 x [ 1.600e-02 -4.000e-03  9.800e-02  6.764e+00  6.532e+00  1.291e+00] nfev 330 True sec 2.829299211502075
xy fun 0.3843693942340274 iou 0.6196660482374768 x [ 0.007  0.005 -0.054  0.624  2.331 -0.186] nfev 330 True sec 2.789534330368042

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -60 /work/scratch/opt_selected.log; ps -o pid,etime,pcpu,pmem,cmd -C python | tail -5'
```
<details><summary>output</summary>

```
frame 0
id fun 0.4706279188132812 iou 0.5340364333652924 x [-0.008  0.015 -0.01  -5.901 -1.054  1.38 ] nfev 345 True sec 3.076202630996704
yz fun 0.6452133582344246 iou 0.37447873227689743 x [ 3.0000e-03 -9.8000e-02 -4.5000e-02  1.3387e+01  5.4500e+00 -4.9800e-01] nfev 447 True sec 3.891766309738159
xz fun 0.6109869115693228 iou 0.40267857142857144 x [-0.016 -0.117  0.083 -4.727  1.117 -1.519] nfev 311 True sec 2.644704818725586
xy fun 0.493711498695776 iou 0.5119266055045871 x [ 6.0000e-03 -2.0000e-03 -4.6000e-02  2.4229e+01  1.1280e+00 -2.7600e-01] nfev 408 True sec 3.5290539264678955
frame 100
id fun 0.48177869201841805 iou 0.5238095238095238 x [-0.006  0.019 -0.022 -0.3   -5.909 -0.883] nfev 340 True sec 2.983785629272461
yz fun 0.5663661315613234 iou 0.4465832531280077 x [ 5.000e-03 - …[+2518 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -45 /work/scratch/opt_selected.log'
```
<details><summary>output</summary>

```
frame 0
id fun 0.4706279188132812 iou 0.5340364333652924 x [-0.008  0.015 -0.01  -5.901 -1.054  1.38 ] nfev 345 True sec 3.076202630996704
yz fun 0.6452133582344246 iou 0.37447873227689743 x [ 3.0000e-03 -9.8000e-02 -4.5000e-02  1.3387e+01  5.4500e+00 -4.9800e-01] nfev 447 True sec 3.891766309738159
xz fun 0.6109869115693228 iou 0.40267857142857144 x [-0.016 -0.117  0.083 -4.727  1.117 -1.519] nfev 311 True sec 2.644704818725586
xy fun 0.493711498695776 iou 0.5119266055045871 x [ 6.0000e-03 -2.0000e-03 -4.6000e-02  2.4229e+01  1.1280e+00 -2.7600e-01] nfev 408 True sec 3.5290539264678955
frame 100
id fun 0.48177869201841805 iou 0.5238095238095238 x [-0.006  0.019 -0.022 -0.3   -5.909 -0.883] nfev 340 True sec 2.983785629272461
yz fun 0.5663661315613234 iou 0.4465832531280077 x [ 5.000e-03 - …[+3494 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -30 /work/scratch/opt_selected.log'
```
<details><summary>output</summary>

```
frame 200
id fun 0.35295259759198655 iou 0.6501766784452296 x [ 1.0000e-02  4.0000e-02 -4.2000e-02 -2.2154e+01 -4.0160e+00  1.9080e+01] nfev 601 True sec 5.352222919464111
yz fun 0.8011585226089943 iou 0.24041585445094216 x [-1.4000e-02  0.0000e+00  1.0000e-01 -2.3121e+01  2.0952e+01  1.3370e+01] nfev 712 True sec 6.474788188934326
xz fun 0.6776142337772236 iou 0.34317089910775567 x [ 8.7000e-02 -5.6000e-02  1.6000e-02 -2.1097e+01  6.1400e-01  1.7657e+01] nfev 1167 True sec 11.071348667144775
xy fun 0.5353490886540376 iou 0.475968992248062 x [  0.041   0.028  -0.087 -23.624  10.735  13.197] nfev 457 True sec 4.312493324279785
frame 400
id fun 0.43767257201017035 iou 0.5682196339434277 x [ 2.9000e-02  6.0000e-03 -1.1000e-02 -1.9353e+01 -1.3098e+01 -1.3550e+00] nfev 315 True sec 3.2419593334 …[+2878 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/opt_pose.py 0 100 120 200 400 500 600 700 800 850 880 > /work/scratch/opt_selected.log
cat /work/scratch/opt_selected.log'
```
<details><summary>output</summary>

```
frame 0
id fun 0.4706279188132812 iou 0.5340364333652924 x [-0.008  0.015 -0.01  -5.901 -1.054  1.38 ] nfev 345 True sec 3.076202630996704
yz fun 0.6452133582344246 iou 0.37447873227689743 x [ 3.0000e-03 -9.8000e-02 -4.5000e-02  1.3387e+01  5.4500e+00 -4.9800e-01] nfev 447 True sec 3.891766309738159
xz fun 0.6109869115693228 iou 0.40267857142857144 x [-0.016 -0.117  0.083 -4.727  1.117 -1.519] nfev 311 True sec 2.644704818725586
xy fun 0.493711498695776 iou 0.5119266055045871 x [ 6.0000e-03 -2.0000e-03 -4.6000e-02  2.4229e+01  1.1280e+00 -2.7600e-01] nfev 408 True sec 3.5290539264678955
frame 100
id fun 0.48177869201841805 iou 0.5238095238095238 x [-0.006  0.019 -0.022 -0.3   -5.909 -0.883] nfev 340 True sec 2.983785629272461
yz fun 0.5663661315613234 iou 0.4465832531280077 x [ 5.000e-03 - …[+5847 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -30 /work/scratch/static.log'
```
<details><summary>output</summary>

```
weights [0.09420429 0.09392322 0.09212321 0.08766257 0.09183578 0.09043362
 0.0907255  0.0900919  0.08999403 0.08965063 0.08935524]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -30 /work/scratch/static.log; ps -o pid,etime,pcpu,pmem -C python | tail -3'
```
<details><summary>output</summary>

```
weights [0.09420429 0.09392322 0.09212321 0.08766257 0.09183578 0.09043362
 0.0907255  0.0900919  0.08999403 0.08965063 0.08935524]
start id fun 0.49863770712089966 miou 0.5074209359363543 per [0.471 0.5   0.502 0.516 0.506 0.511 0.511 0.513 0.514 0.513 0.524] x [-2.000e-03 -4.000e-03 -8.400e-02 -1.490e-01 -6.666e+00 -8.240e-01] nfev 377 sec 32.674999713897705
    PID     ELAPSED %CPU %MEM
     15       19:34  0.2  0.0
   8385       01:09  108  2.9

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/static.log'
```
<details><summary>output</summary>

```
weights [0.09420429 0.09392322 0.09212321 0.08766257 0.09183578 0.09043362
 0.0907255  0.0900919  0.08999403 0.08965063 0.08935524]
start id fun 0.49863770712089966 miou 0.5074209359363543 per [0.471 0.5   0.502 0.516 0.506 0.511 0.511 0.513 0.514 0.513 0.524] x [-2.000e-03 -4.000e-03 -8.400e-02 -1.490e-01 -6.666e+00 -8.240e-01] nfev 377 sec 32.674999713897705
start yz fun 0.563412814251679 miou 0.44980122821820995 per [0.418 0.46  0.474 0.469 0.438 0.444 0.442 0.447 0.448 0.455 0.454] x [ 6.0000e-03 -1.1600e-01 -6.4000e-02  1.7705e+01 -3.8040e+00  6.7700e-01] nfev 436 sec 37.34023642539978
start xz fun 0.5921289271737341 miou 0.42153975224857737 per [0.427 0.424 0.437 0.424 0.412 0.418 0.417 0.422 0.421 0.419 0.417] x [-0.013 -0.104  0.085 -8.31  -5.113  0.198] nfev 370 sec 34.91115474700 …[+4 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -25 /work/scratch/static.log'
```
<details><summary>output</summary>

```
weights [0.09420429 0.09392322 0.09212321 0.08766257 0.09183578 0.09043362
 0.0907255  0.0900919  0.08999403 0.08965063 0.08935524]
start id fun 0.49863770712089966 miou 0.5074209359363543 per [0.471 0.5   0.502 0.516 0.506 0.511 0.511 0.513 0.514 0.513 0.524] x [-2.000e-03 -4.000e-03 -8.400e-02 -1.490e-01 -6.666e+00 -8.240e-01] nfev 377 sec 32.674999713897705
start yz fun 0.563412814251679 miou 0.44980122821820995 per [0.418 0.46  0.474 0.469 0.438 0.444 0.442 0.447 0.448 0.455 0.454] x [ 6.0000e-03 -1.1600e-01 -6.4000e-02  1.7705e+01 -3.8040e+00  6.7700e-01] nfev 436 sec 37.34023642539978
start xz fun 0.5921289271737341 miou 0.42153975224857737 per [0.427 0.424 0.437 0.424 0.412 0.418 0.417 0.422 0.421 0.419 0.417] x [-0.013 -0.104  0.085 -8.31  -5.113  0.198] nfev 370 sec 34.91115474700 …[+1198 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
import sys
def run_static(frames,init_frame,tag):
 tm_all=[];pm_all=[];weights=[]
 for fi in frames:
  tm=objs[fi].astype(np.uint8);tm=cv2.resize(tm,(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pers[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);tm_all.append(tm);pm_all.append(pm);weights.append(min(1.,max(.2,np.sqrt(tm.sum()/3000.))))
 weights=np.array(weights);weights/=weights.sum();print('weights',weights)
 results=[]
 for ci,(Rm,sh) in enumerate(zip(Hs,[cent-Rm@cent for Rm in Hs])):
  Ti=np.eye(4,dtype=np.float32);Ti[:3,:3]=raw[init_frame,:3,:3]@Rm;Ti[:3,3]=raw[init_frame,:3,3]+raw[init_frame,:3,:3]@sh;Ti=torch.tensor(Ti,device=dev)
  def make(x):
   Tt=Ti.clone();dw=torch.tensor(W[:3,:3],device=dev)@torch.tensor(x[:3],device=dev,dtype=torch.float32);Tt[:3,3]+=dw
   Rd=R.from_rotvec(np.deg2rad(x[3:])).as_matrix().astype(np.float32);Rd=torch.tensor(Rd,device=dev);Rcam=torch.tensor(Wi[:3,:3],device=dev)@Tt[:3,:3];Rcam=Rd@Rcam;Tt[:3,:3]=torch.tensor(W[:3,:3],device=dev)@Rcam;return Tt
  def obj(x):
   Tt=make(x);ls=[]
   for tm,pm,wt in zip(tm_all,pm_all,weights):ls.append(wt*loss_mask(rend(Tt),tm,pm)[0])
   return sum(ls)
  st=time.time();r=minimize(obj,np.zeros(6),method='Powell',bounds=[(-.12,.12)]*3+[(-25,25)]*3,options={'maxiter':60,'xtol':1e-3,'ftol':1e-4,'disp':False})
  Tt=make(r.x);ios= …[+663 chars]
```
<details><summary>output</summary>

```
weights [0.09420429 0.09392322 0.09212321 0.08766257 0.09183578 0.09043362
 0.0907255  0.0900919  0.08999403 0.08965063 0.08935524]
start id fun 0.49863770712089966 miou 0.5074209359363543 per [0.471 0.5   0.502 0.516 0.506 0.511 0.511 0.513 0.514 0.513 0.524] x [-2.000e-03 -4.000e-03 -8.400e-02 -1.490e-01 -6.666e+00 -8.240e-01] nfev 377 sec 32.674999713897705
start yz fun 0.563412814251679 miou 0.44980122821820995 per [0.418 0.46  0.474 0.469 0.438 0.444 0.442 0.447 0.448 0.455 0.454] x [ 6.0000e-03 -1.1600e-01 -6.4000e-02  1.7705e+01 -3.8040e+00  6.7700e-01] nfev 436 sec 37.34023642539978
start xz fun 0.5921289271737341 miou 0.42153975224857737 per [0.427 0.424 0.437 0.424 0.412 0.418 0.417 0.422 0.421 0.419 0.417] x [-0.013 -0.104  0.085 -8.31  -5.113  0.198] nfev 370 sec 34.91115474700 …[+1589 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];v=np.array(g.vertices)
for z0,z1 in [(-.4,-.3),(-.3,-.2),(-.2,-.1),(-.1,0),(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6)]:
 p=v[(v[:,2]>=z0)&(v[:,2]<z1)]
 if len(p): print(f'z {z0:.1f} {z1:.1f} n{len(p)} x {p[:,0].min():.3f} {p[:,0].max():.3f} y {p[:,1].min():.3f} {p[:,1].max():.3f} mean {p.mean(0)}')
# top/bottom extreme
for idx in np.argsort(v[:,2])[:5]:print('bottom',idx,v[idx])
for idx in np.argsort(v[:,2])[-5:]:print('top',idx,v[idx])
PY"
```
<details><summary>output</summary>

```
z -0.4 -0.3 n130911 x -0.105 0.102 y -0.065 0.087 mean [-3.33030120e-05  7.56277810e-03 -3.39156317e-01]
z -0.3 -0.2 n15942 x -0.014 0.013 y -0.022 0.004 mean [-0.00105707 -0.00915494 -0.26025332]
z -0.2 -0.1 n17487 x -0.012 0.013 y -0.020 0.006 mean [ 0.00029482 -0.0068684  -0.14109742]
z -0.1 0.0 n15442 x -0.010 0.013 y -0.018 0.005 mean [ 0.00081521 -0.00596794 -0.04780086]
z 0.0 0.1 n14702 x -0.011 0.012 y -0.017 0.007 mean [ 0.00133707 -0.00541329  0.05387502]
z 0.1 0.2 n17512 x -0.011 0.013 y -0.020 0.008 mean [ 0.00155444 -0.00593838  0.15699433]
z 0.2 0.3 n15834 x -0.012 0.013 y -0.015 0.009 mean [ 0.00353358 -0.00413414  0.25148578]
z 0.3 0.4 n17005 x -0.047 0.013 y -0.014 0.009 mean [-0.01323396 -0.00261018  0.34871458]
z 0.4 0.5 n25506 x -0.069 0.095 y -0.014 0.015 mean [-0.0174 …[+620 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
# view pooled static best
import numpy as np,torch,trimesh,cv2,json,h5py
import nvdiffrast.torch as dr
dev='cuda';T=np.load('/work/scratch/static_start.npz')['T'];g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev);c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Wi=np.linalg.inv(np.array(c['T_world_cam'],np.float32)).astype(np.float32);Kt=torch.tensor(K,device=dev);Wr=torch.tensor(Wi[:3,:3],device=dev);Wt=torch.tensor(Wi[:3,3],device=dev);gl=dr.RasterizeCudaContext();HH,WW=1152,1536
def rend(T):
 T=torch.tensor(T,device=dev);vw=V@T[:3,:3].T+T[:3,3];vc=vw@Wr.T+Wt;x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z;clip=torch.stack([2*u/WW-z,2*v/HH-z,.5*z,z],1)[None];r,_=dr.rasterize(gl,clip,F,[HH,WW]);return r[0,:,:,3].cpu().numpy()>0
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for fi in range(0,101,10):
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);rm=rend(T);m=objs[fi];mm=rm|m;y,x=np.where(mm);pad=40;x0=max(0,x.min()-pad);x1=min(WW,x.max()+pad);y0=max(0,y.min()-pad);y1=min(HH,y.max()+pad);o=im.copy();o[rm]=(.5*o[rm]+.5*np.array([0,255,0])).astype(np.uint8);cnt,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_E …[+552 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
ValueError: could not broadcast input array from shape (732,300,3) into shape (216,300,3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_view.py'
s='''import numpy as np,torch,trimesh,cv2,json,h5py
import nvdiffrast.torch as dr
dev=\"cuda\";T=np.load(\"/work/scratch/static_start.npz\")[\"T\"];g=trimesh.load(\"/work/inputs/object_mesh.glb\",force=\"scene\",process=False).geometry[\"goodcane\"];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev);c=json.load(open(\"/work/inputs/camera.json\"));K=np.array(c[\"K\"],np.float32);Wi=np.linalg.inv(np.array(c[\"T_world_cam\"],np.float32)).astype(np.float32);Kt=torch.tensor(K,device=dev);Wr=torch.tensor(Wi[:3,:3],device=dev);Wt=torch.tensor(Wi[:3,3],device=dev);gl=dr.RasterizeCudaContext();HH,WW=1152,1536
def rend(T):
 T=torch.tensor(T,device=dev);vw=V@T[:3,:3].T+T[:3,3];vc=vw@Wr.T+Wt;x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z;clip=torch.stack([2*u/WW-z,2*v/HH-z,.5*z,z],1)[None];r,_=dr.rasterize(gl,clip,F,[HH,WW]);return r[0,:,:,3].cpu().numpy()>0
with h5py.File(\"/work/inputs/masks.h5\") as f:objs=f[\"object\"][:]
cap=cv2.VideoCapture(\"/work/inputs/video.mp4\");ims=[]
for fi in range(0,101,10):
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);rm=rend(T);m=objs[fi];mm=rm|m;y,x=np.where(mm);pad=40;x0=max(0,x.min()-pad);x1=min(WW,x.max()+pad);y0=max(0,y.min()-pad);y1=min(HH,y.max()+pad);o=im.copy();o[rm]=(.5*o[rm]+.5*np.array([0,255,0])).astype(np.uint8);cnt,_=cv2.findCon …[+732 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
z=np.load('/work/inputs/reconstruction.npz');raw=z['T_world_mesh'];Ts=np.load('/work/scratch/static_start.npz')['T'];c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K']);Wi=np.linalg.inv(np.array(c['T_world_cam']))
base=np.array([0,0,-.34,1.]);handle=np.array([0,0,.52,1.])
for name,T in [('raw0',raw[0]),('raw50',raw[50]),('static',Ts)]:
 for lab,p in [('base',base),('handle',handle)]:
  pc=(T@p)[:3];cc=Wi[:3,:3]@pc+Wi[:3,3];uv=K@cc;uv=uv[:2]/uv[2];print(name,lab,uv,cc[2])
 print('R',T[:3,:3])
PY"
```
<details><summary>output</summary>

```
raw0 base [733.95853882 895.62918864] 2.5724556539889742
raw0 handle [724.50613321 651.86217736] 3.0535571342922556
R [[-0.32779336 -0.9441012  -0.03499364]
 [ 0.5766639  -0.17060429 -0.7989698 ]
 [ 0.7483382  -0.28207663  0.6003519 ]]
raw50 base [728.29085099 888.81606873] 2.572418756123223
raw50 handle [732.21607719 635.33928749] 2.978430475654662
R [[ 0.05627816 -0.9982568   0.01775631]
 [ 0.5134652   0.01368618 -0.85800105]
 [ 0.85626155  0.0574044   0.5133418 ]]
static base [736.94854576 897.97623777] 2.660590305652145
static handle [722.31046649 632.67219565] 2.774923954169722
R [[ 0.14525321  0.9876981  -0.05790832]
 [-0.18935435 -0.02969563 -0.98146   ]
 [-0.97110456  0.15352495  0.18271235]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');raw=z['T_world_mesh'];Ts=np.load('/work/scratch/static_start.npz')['T']
rel=Ts[:3,:3].T@raw[50,:3,:3];print('rel\\n',rel);print('ang',R.from_matrix(rel).magnitude()*180/np.pi)
print('raw50',raw[50,:3,:3]);print('Ts',Ts[:3,:3]);print('diag diff',raw[50,:3,:3].T@Ts[:3,:3])
PY"
```
<details><summary>output</summary>

```
rel
 [[-0.9205718  -0.20333722 -0.33346316]
 [ 0.17179567 -0.97756976  0.12182752]
 [-0.35075498  0.05486343  0.93485934]]
ang 169.00474134936772
raw50 [[ 0.05627816 -0.9982568   0.01775631]
 [ 0.5134652   0.01368618 -0.85800105]
 [ 0.85626155  0.0574044   0.5133418 ]]
Ts [[ 0.14525321  0.9876981  -0.05790832]
 [-0.18935435 -0.02969563 -0.98146   ]
 [-0.97110456  0.15352495  0.18271235]]
diag diff [[-0.9205718   0.17179567 -0.35075498]
 [-0.20333722 -0.97756976  0.05486343]
 [-0.33346316  0.12182752  0.93485934]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
raw=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Ts=np.load('/work/scratch/static_start.npz')['T']
rel=raw[50,:3,:3].T@Ts[:3,:3];print(rel, np.round(R.from_matrix(rel).as_rotvec()*180/np.pi,2));print(raw[50,:3,:3]@rel)
PY"
```
<details><summary>output</summary>

```
[[-0.9205718   0.17179567 -0.35075498]
 [-0.20333722 -0.97756976  0.05486343]
 [-0.33346316  0.12182752  0.93485934]] [  29.67   -7.66 -166.2 ]
[[ 0.1452536   0.98769724 -0.05790798]
 [-0.18935278 -0.02969624 -0.9814599 ]
 [-0.97110325  0.15352438  0.18271375]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
for n,H in zip(names,Hs):print(n,H)
PY"
```
<details><summary>output</summary>

```
id [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
yz [[ 1.  0.  0.]
 [ 0. -1.  0.]
 [ 0.  0. -1.]]
xz [[-1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0. -1.]]
xy [[-1.  0.  0.]
 [ 0. -1.  0.]
 [ 0.  0.  1.]]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_global_flips.py <<'PY'
# reuse eval smoothing setup
exec(open('/work/scratch/eval_smooth.py').read().split(\"names=['raw'\")[0])
# note raw loaded, q,t, etc. Build global flips and smooth
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
cent=np.array(g.vertices).mean(0).astype(np.float32)
def apply_flip(raw,H,cent):
 Rm=H[:3,:3]; sh=cent-Rm@cent; out=raw.copy();out[:,:3,:3]=raw[:,:3,:3]@Rm;out[:,:3,3]=raw[:,:3,3]+raw[:,:3,:3]@sh;return out
def smooth_pose(rawx,sigma=2.0):
 q=R.from_matrix(rawx[:,:3,:3]).as_quat().astype(np.float64)
 for i in range(1,len(q)):
  if q[i]@q[i-1]<0:q[i]*=-1
 t=rawx[:,:3,3].astype(np.float64)
 q=gaussian_filter1d(q,sigma,axis=0,mode='nearest');t=gaussian_filter1d(t,sigma,axis=0,mode='nearest');q/=np.linalg.norm(q,axis=1,keepdims=True)
 T=np.tile(np.eye(4),(len(rawx),1,1));T[:,:3,:3]=R.from_quat(q).as_matrix();T[:,:3,3]=t
 return T.astype(np.float32)
def score(T):
 io=[]
 for i in range(len(T)):
  rm=render_mask(T[i]);m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev);rv=rm&~p;inter=(rv&m).sum().item();uni=(rv|m).sum().item();io.append(inter/max(uni,1))
 return np.array(io)
Hs=[];names=[]
for bits in itertools.product([1,-1],repeat=3):
 if np.prod(bits)==1:
  Rm=np.diag(bits).astype(np.float32);names.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id');Hs.append(np.block([[Rm,np.zeros((3,1),np.float32)],[np.zeros((1,3),np.float32),np.ones((1,1),np …[+395 chars]
```
<details><summary>output</summary>

```
static q [ 0.36374821 -0.33319146  0.6007031   0.62914742] [-0.4862619  -0.46746552  0.5065401  -0.53706841] t [0.03939909 0.81315041 2.92832804] [0.20645963 0.77906078 3.80895281]
Traceback (most recent call last):
  File "/work/scratch/eval_global_flips.py", line 23, in <module>
    for bits in itertools.product([1,-1],repeat=3):
                ^^^^^^^^^
NameError: name 'itertools' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '1i import itertools' /work/scratch/eval_global_flips.py
python /work/scratch/eval_global_flips.py"
```
<details><summary>output</summary>

```
static q [ 0.36374821 -0.33319146  0.6007031   0.62914742] [-0.4862619  -0.46746552  0.5065401  -0.53706841] t [0.03939909 0.81315041 2.92832804] [0.20645963 0.77906078 3.80895281]
id sig 0 all 0.22334812421665737 ph0 0.23998361303809101 dyn 0.21937363368229992 ph1 0.2359266919881858 q10/50/90 [0.03190895 0.22899206 0.39757019]
id sig 2 all 0.23145386480974756 ph0 0.23819543092098466 dyn 0.2294208699765384 ph1 0.24015141294064102 q10/50/90 [0.03672139 0.24401431 0.41477325]
yz sig 0 all 0.14412912079102447 ph0 0.27647133292956194 dyn 0.13277898449613515 ph1 0.07141952317616106 q10/50/90 [0.01909971 0.13143684 0.27831791]
yz sig 2 all 0.15043786717727461 ph0 0.282292088483794 dyn 0.13978019738921313 ph1 0.07245028113667043 q10/50/90 [0.01846382 0.14675963 0.27946468]
xz sig 0 all 0.15517294 …[+576 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/flips_all.npz');S=z['scores'][:,:,0];n=z['names'];F=np.arange(len(S))
for a,b in [(0,105),(105,200),(200,400),(400,600),(600,804),(804,886),(105,804)]:
 m=(F>=a)&(F<b);print(a,b,np.round(S[m].mean(0),3),'wins',np.bincount(S[m].argmax(1),minlength=4),'maxmean',S[m].max(1).mean())
# count frames per state where best margin > .03
best=S.argmax(1);marg=np.sort(S,1)[:,-1]-np.sort(S,1)[:,-2]
for k,name in enumerate(n):
 m=(best==k)&(marg>.03);print(name,m.sum(),'mean score',S[m,k].mean() if m.any() else 0)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_texture_all.py <<'PY'
import numpy as np,torch,trimesh,cv2,json,h5py,itertools,time
import nvdiffrast.torch as dr
dev='cuda';np.random.seed(0);torch.manual_seed(0)
raw=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32)
c=json.load(open('/work/inputs/camera.json'));W=np.linalg.inv(np.array(c['T_world_cam'],np.float32)).astype(np.float32);K=np.array(c['K'],np.float32)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev);uv=torch.tensor(np.asarray(g.visual.uv),dtype=torch.float32,device=dev);tex=torch.tensor(np.array(g.visual.material.baseColorTexture),dtype=torch.float32,device=dev).permute(2,0,1)[None]/255
cent=np.array(g.vertices).mean(0)
Kt=torch.tensor(K,device=dev);Wri=torch.tensor(W[:3,:3],device=dev);Wti=torch.tensor(W[:3,3],device=dev);gl=dr.RasterizeCudaContext();HH,WW=1152,1536
Hs=[];names=[]
for bits in itertools.product([1,-1],repeat=3):
 if np.prod(bits)==1:
  Rm=np.diag(bits).astype(np.float32);names.append(''.join(a for a,b in zip('xyz',bits) if b==-1) or 'id');Hs.append(Rm)
def rend(T):
 vw=V@T[:3,:3].T+T[:3,3];vc=vw@Wri.T+Wti;x,y,zc=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*zc;v=Kt[1,1]*y+Kt[1,2]*zc;clip=torch.stack([2*u/WW-zc,2*v/HH-zc,.5*zc,zc],1)[None];r,_=dr.rasterize(gl,clip,F,[HH,WW]);vul,_=dr.interpolate(uv[None],r,F);u …[+2491 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_texture_all.py", line 23, in <module>
    sh=cent-Rm@cent;T=torch.tensor(raw[i],device=dev);Tc=torch.eye(4,device=dev);Tc[:3,:3]=T[:3,:3]@torch.tensor(Rm,device=dev);Tc[:3,3]=T[:3,3]+T[:3,:3]@torch.tensor(sh,device=dev)
                                                                                                                                                 ~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: expected scalar type Float but found Double

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/sh=cent-Rm@cent;T=/sh=(cent-Rm@cent).astype(np.float32);T=/' /work/scratch/eval_texture_all.py
python /work/scratch/eval_texture_all.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_texture_all.py", line 33, in <module>
    row.append([iou,n,corr.item(),gc.item(),l1.item()])
                      ^^^^^^^^^
AttributeError: 'int' object has no attribute 'item'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; ps -o pid,etime,pcpu,pmem,cmd -C python | tail -4'
```
<details><summary>output</summary>

```
10183, 856 MiB
    PID     ELAPSED %CPU %MEM CMD
     15       31:37  0.2  0.0 python /work/.net/allowlist_proxy.py inner --sock /work/.net/proxy.sock
  10183       04:45  431  3.2 python /work/scratch/eval_texture_all.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/corr.item(),gc.item(),l1.item()/float(corr),float(gc),float(l1)/' /work/scratch/eval_texture_all.py
python /work/scratch/eval_texture_all.py"
```
<details><summary>output</summary>

```
id n 799 iou 0.23323442803083042 corr 0.01634663152101931 gcorr -0.004443532812979669 l1 0.15815166586331642
 phases iou [0.23398319414372756, 0.2065003328785096, 0.22675934444525664] corr [-0.1290282767266035, 0.03925183363482897, 0.02816683537196041]
yz n 767 iou 0.15869339444073 corr 0.03184038559692782 gcorr 0.017734152887199685 l1 0.158600258605732
 phases iou [0.2715816017271003, 0.1272564932948455, 0.06598884025530675] corr [0.029021740140437725, 0.018035353153270373, 0.1960521314670947]
xz n 776 iou 0.1693923207463805 corr 0.01962179734843077 gcorr 0.00833093620251264 l1 0.1590334075819893
 phases iou [0.24278465567830831, 0.13833943481398178, 0.12520886602349918] corr [0.042317235021341944, 0.01452091177885609, 0.03377412032568827]
xy n 800 iou 0.2145757682502407 corr -0.003576175 …[+207 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);Rr=T[:,:3,:3]
# local axes directions
x=Rr[:,:,0];y=Rr[:,:,1];zz=Rr[:,:,2]
dot=np.sum(x[1:]*x[:-1],axis=1)
print('dot x quant',np.quantile(dot,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]),'neg',np.sum(dot<0))
idx=np.where(dot<0)[0]+1;print('negative indices',idx)
# axis z continuity
dotz=np.sum(zz[1:]*zz[:-1],axis=1);print('z dot neg',np.sum(dotz<0),np.where(dotz<0)[0]+1)
# build toggled signs based on x continuity; note if true rotation >90 per frame this fails
sign=np.ones(len(T),int)
for i in range(1,len(T)):
 sign[i]=sign[i-1]*(-1 if dot[i-1]<0 else 1)
print('state counts',np.bincount(sign))
# print runs
runs=[];s=0
for i in range(1,len(sign)+1):
 if i==len(sign) or sign[i]"'!=sign[s]:runs.append((s,i-1,int(sign[s]),float(dot[max(0,s-1)] if s else 1)));s=i
print('"'runs',runs)
# plot axes and state
fig,ax=plt.subplots(4,1,figsize=(18,12),sharex=True)
for k in range(3):ax[0].plot(Rr[:,k,0],label='xyz'[k]);ax[0].plot(Rr[:,k,1],'--',alpha=.6)
ax[0].set_ylabel('local x/y axes');ax[0].legend()
for k in range(3):ax[1].plot(Rr[:,k,2],label='xyz'[k]);ax[1].set_ylabel('local z axis');ax[1].legend()
ax[2].plot(dot);ax[2].axhline(0,color='r');ax[2].set_ylabel('dot local x');ax[2].set_ylim(-1.05,1.05)
ax[3].plot(sign);ax[3].set_ylabel('toggle sta …[+208 chars]
```
<details><summary>output</summary>

```
dot x quant [-0.24663562  0.74399333  0.9069086   0.95601265  0.98958756  0.99816214
  0.99999761  0.9999998   1.          1.0000006   1.00000082] neg 2
negative indices [295 636]
z dot neg 1 [756]
Traceback (most recent call last):
  File "<stdin>", line 15, in <module>
ValueError: 'list' argument must have no negative elements

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
# reconstruct identity and xy poses from reported x, init raw50
outs={}
for name,ci,x in [('id',0,[-.002,-.004,-.084,-.149,-6.666,-.824]),('xy',3,[.0060649,.0279948,-.0279077,20.0733,-12.3678,.34996])]:
 Rm=Hs[ci];sh=(cent-Rm@cent).astype(np.float32);Ti=torch.eye(4,device=dev);Ti[:3,:3]=torch.tensor(raw[50,:3,:3],device=dev)@torch.tensor(Rm,device=dev);Ti[:3,3]=torch.tensor(raw[50,:3,3],device=dev)+torch.tensor(raw[50,:3,:3],device=dev)@torch.tensor(sh,device=dev)
 x=np.array(x);Tt=Ti.clone();Tt[:3,3]+=torch.tensor(W[:3,:3],device=dev)@torch.tensor(x[:3],device=dev,dtype=torch.float32);Rd=R.from_rotvec(np.deg2rad(x[3:])).as_matrix().astype(np.float32);Rcam=torch.tensor(Wi[:3,:3],device=dev)@Tt[:3,:3];Rcam=torch.tensor(Rd,device=dev)@Rcam;Tt[:3,:3]=torch.tensor(W[:3,:3],device=dev)@Rcam;outs[name]=Tt.cpu().numpy();print(name,Tt)
np.savez('/work/scratch/static_cmp.npz',**outs)
PY
# modify static view to compare two at frame 0 and 50
sed 's/T=np.load(\"\\/work\\/scratch\\/static_start.npz\")\\[\"T\"\\]/T1=np.load(\"\\/work\\/scratch\\/static_cmp.npz\")[\"id\"];T2=np.load(\"\\/work\\/scratch\\/static_cmp.npz\")[\"xy\"]/' /work/scratch/static_view.py | sed 's/rm=rend(T);m=objs\\[fi\\];mm=rm|m/rm=rend(T1);rm2=rend(T2);m=objs[fi];mm=rm|rm2|m/' > /work/scratch/static_cmp_view.py
# easier manually patch to grid later? just execute to see errors
python - <<'PY'
# create two-panel  …[+1494 chars]
```
<details><summary>output</summary>

```
id tensor([[-0.0394, -0.9980, -0.0498,  0.0337],
        [ 0.5164,  0.0223, -0.8560,  0.8054],
        [ 0.8554, -0.0595,  0.5145,  2.8455],
        [ 0.0000,  0.0000,  0.0000,  1.0000]], device='cuda:0')
xy tensor([[ 0.1453,  0.9877, -0.0579,  0.0426],
        [-0.1894, -0.0297, -0.9815,  0.8404],
        [-0.9711,  0.1535,  0.1827,  2.9004],
        [ 0.0000,  0.0000,  0.0000,  1.0000]], device='cuda:0')
  File "<stdin>", line 4
    Ts=np.load('/work/scratch/static_cmp.npz');cap=cv2.VideoCapture('/work/inputs/video.mp4');with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:]
                                                                                              ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_cmp_view2.py <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
import cv2,h5py
Ts=np.load('/work/scratch/static_cmp.npz');cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:]
alltiles=[]
for fi in [0,25,50,75,100]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=objs[fi];rms=[]
 for n in ['id','xy']:
  rm=rend(torch.tensor(Ts[n],device=dev));rms.append(rm)
 mm=m|rms[0]|rms[1];yy,xx=np.where(mm);pad=40;x0=max(0,xx.min()-pad);x1=min(WW,xx.max()+pad);y0=max(0,yy.min()-pad);y1=min(HH,yy.max()+pad)
 row=[]
 for n,rm in zip(['id','xy'],rms):
  o=im.copy();o[rm]=(.5*o[rm]+.5*np.array([0,255,0])).astype(np.uint8);cnt,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(o,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(o,cnt,-1,(255,0,255),2);o=o[y0:y1,x0:x1];sc=min(250/o.shape[1],250/o.shape[0]);o=cv2.resize(o,(int(o.shape[1]*sc),int(o.shape[0]*sc)));can=np.zeros((280,250,3),np.uint8);oy=(250-o.shape[0])//2;ox=(250-o.shape[1])//2;can[oy:oy+o.shape[0],ox:ox+o.shape[1]]=o;cv2.putText(can,f'{fi} {n}',(5,272),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,0),2);row.append(can)
 alltiles.append(np.concatenate(row,1))
cv2.imwrite('/work/scratch/static_cmp.jpg',cv2.cvtColor(np.concatenate(alltiles,1),cv2.C …[+59 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_cmp_view2.py", line 10, in <module>
    mm=m|rms[0]|rms[1];yy,xx=np.where(mm);pad=40;x0=max(0,xx.min()-pad);x1=min(WW,xx.max()+pad);y0=max(0,yy.min()-pad);y1=min(HH,yy.max()+pad)
       ~^~~~~~~
ValueError: operands could not be broadcast together with shapes (1152,1536) (576,768) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_cmp_view2.py';s=open(p).read()
s=s.replace(\"cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=objs[fi];rms=[]\", \"cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(cv2.resize(im,(WW,HH)),cv2.COLOR_BGR2RGB);m=cv2.resize(objs[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);rms=[]\")
open(p,'w').write(s)
PY
python /work/scratch/static_cmp_view2.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# create full-res id vs xy overlays at frame 50
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
import cv2,h5py
Ts=np.load('/work/scratch/static_cmp.npz');cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,50);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
with h5py.File('/work/inputs/masks.h5') as f:m=f['object'][50]
rms=[]
# use full res setup by changing dimensions? current rend half. upscale image and draw
im2=cv2.resize(im,(WW,HH));m2=cv2.resize(m.astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool)
for n in ['id','xy']:rms.append(rend(torch.tensor(Ts[n],device=dev)))
mm=m2|rms[0]|rms[1];y,x=np.where(mm);pad=20;x0=max(0,x.min()-pad);x1=min(WW,x.max()+pad);y0=max(0,y.min()-pad);y1=min(HH,y.max()+pad)
out=[]
for n,rm in zip(['id','xy'],rms):
 o=im2.copy();o[rm]=(.5*o[rm]+.5*np.array([0,255,0])).astype(np.uint8);cnt,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(o,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(m2.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(o,cnt,-1,(255,0,255),2);o=o[y0:y1,x0:x1];o=cv2.resize(o,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST);cv2.putText(o,n,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,0),3);out.append(o)
cv2.imwrite('/work/scratch/static_zoom.jpg',cv2.cvtColor(np.concatenate(out,1),cv2.COLOR_RGB2BGR))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/optcmp.log'
```
<details><summary>output</summary>

```
id 10 frame 195 iou 0.19706632653061223 base 0.04517221908526256

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/optcmp.log'
```
<details><summary>output</summary>

```
id 10 frame 195 iou 0.19706632653061223 base 0.04517221908526256
id 20 frame 295 iou 0.42764753435731606 base 0.2519741564967696
id 30 frame 395 iou 0.479933110367893 base 0.2677916360968452
id 40 frame 495 iou 0.4853273137697517 base 0.11390284757118928

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/optcmp.log'
```
<details><summary>output</summary>

```
id 10 frame 195 iou 0.19706632653061223 base 0.04517221908526256
id 20 frame 295 iou 0.42764753435731606 base 0.2519741564967696
id 30 frame 395 iou 0.479933110367893 base 0.2677916360968452
id 40 frame 495 iou 0.4853273137697517 base 0.11390284757118928
id 50 frame 595 iou 0.45303326810176126 base 0.14917127071823205
id 60 frame 695 iou 0.48375768217734855 base 0.18464880521361332
id 70 frame 795 iou 0.3446866485013624 base 0.243142144638404
xy 10 frame 195 iou 0.2664406779661017 base 0.0423968343697004

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/optcmp.log'
```
<details><summary>output</summary>

```
id 10 frame 195 iou 0.19706632653061223 base 0.04517221908526256
id 20 frame 295 iou 0.42764753435731606 base 0.2519741564967696
id 30 frame 395 iou 0.479933110367893 base 0.2677916360968452
id 40 frame 495 iou 0.4853273137697517 base 0.11390284757118928
id 50 frame 595 iou 0.45303326810176126 base 0.14917127071823205
id 60 frame 695 iou 0.48375768217734855 base 0.18464880521361332
id 70 frame 795 iou 0.3446866485013624 base 0.243142144638404
xy 10 frame 195 iou 0.2664406779661017 base 0.0423968343697004
xy 20 frame 295 iou 0.5046968403074296 base 0.3113278319579895
xy 30 frame 395 iou 0.5726569217540842 base 0.3546686746987952
xy 40 frame 495 iou 0.4977220956719818 base 0.23485554520037277
xy 50 frame 595 iou 0.45862412761714855 base 0.22424242424242424

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_compare.py <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
from scipy.ndimage import gaussian_filter1d
# global flips: id and xy=Rz(long axis)
def flip_raw(rawx,Rm):
 sh=(cent-Rm@cent).astype(np.float32);o=rawx.copy();o[:,:3,:3]=rawx[:,:3,:3]@Rm;o[:,:3,3]=rawx[:,:3,3]+rawx[:,:3,:3]@sh;return o
def smooth_raw(rawx,sigma=2.0):
 q=R.from_matrix(rawx[:,:3,:3]).as_quat().astype(np.float64)
 for i in range(1,len(q)):
  if q[i]@q[i-1]<0:q[i]*=-1
 t=rawx[:,:3,3].astype(np.float64);q=gaussian_filter1d(q,sigma,axis=0,mode='nearest');t=gaussian_filter1d(t,sigma,axis=0,mode='nearest');q/=np.linalg.norm(q,axis=1,keepdims=True)
 T=np.tile(np.eye(4),(len(rawx),1,1));T[:,:3,:3]=R.from_quat(q).as_matrix();T[:,:3,3]=t;return T.astype(np.float32)
flips={'id':np.eye(3,dtype=np.float32),'xy':np.diag([-1,-1,1]).astype(np.float32)}
starts={k:smooth_raw(flip_raw(raw,Rm),2.0) for k,Rm in flips.items()}
# optimize all selected anchors, both starts
frames=list(range(105,805,10))
allres={k:[] for k in flips}
tmcache={};pmcache={}
for fi in frames:
 tm=objs[fi].astype(np.uint8);tm=cv2.resize(tm,(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pers[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);tmcache[fi]=tm;pmcache[fi]=pm
for k in flips:
 for fi in frames:
  Ti=torch.tensor(starts[k][fi],device=dev);tm=tmcache[fi];pm=pmcache[fi]
  def make(x):
   Tt=Ti.clone();dw=torch.tensor(W[:3,:3] …[+1185 chars]
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:118: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2357: RuntimeWarning: invalid value encountered in scalar multiply
  q = (xf - fulc) * (fx - fnfc)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2358: RuntimeWarning: invalid value encountered in scalar subtract
  p = (xf - fulc) * q - (xf - nfc) * r
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2359: RuntimeWarning: invalid value encountered in scalar subtract
  q = 2.0 * (q - r)
id 10 frame 195 iou 0.19706632653061223 base 0.04517221908526256
id 20 frame 295 iou 0.42764753435731606 base 0.251974156496769 …[+1076 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
A=np.load('/work/scratch/optcmp_id.npy');B=np.load('/work/scratch/optcmp_xy.npy')
d=B[:,1]-A[:,1];print('Rz better frames',np.sum(d>0),'id better',np.sum(d<0),'median diff',np.median(d),'mean',d.mean())
for i,(a,b,dd) in enumerate(zip(A,B,d)):
 if abs(dd)>.04:print(int(a[0]),'id',a[1],'xy',b[1],'d',dd)
PY"
```
<details><summary>output</summary>

```
Rz better frames 36 id better 32 median diff 0.003812930775194623 mean 0.003828738491353414
105 id 0.6447638603696099 xy 0.4963898916967509 d -0.14837396867285896
115 id 0.6005830903790087 xy 0.518450184501845 d -0.08213290587716371
125 id 0.5767045454545454 xy 0.46469049694856146 d -0.11201404850598395
135 id 0.6997690531177829 xy 0.43529411764705883 d -0.2644749354707241
145 id 0.6445012787723785 xy 0.42935982339955847 d -0.21514145537282003
155 id 0.5986159169550173 xy 0.4221556886227545 d -0.17646022833226277
165 id 0.5284810126582279 xy 0.4166666666666667 d -0.1118143459915612
175 id 0.6478076379066479 xy 0.4525810324129652 d -0.19522660549368265
185 id 0.6513859275053305 xy 0.41795104261106075 d -0.23343488489426978
195 id 0.19706632653061223 xy 0.2664406779661017 d 0.069374351435489 …[+3183 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/roll_candidates.py <<'PY'
import numpy as np,torch,trimesh,cv2,json,h5py,itertools,time
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
dev='cuda';raw=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32)
c=json.load(open('/work/inputs/camera.json'));K=np.array(c['K'],np.float32);Wi=np.linalg.inv(np.array(c['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=torch.tensor(np.asarray(g.vertices),dtype=torch.float32,device=dev);F=torch.tensor(np.asarray(g.faces),dtype=torch.int32,device=dev);cent=np.array(g.vertices).mean(0)
Kt=torch.tensor(K,device=dev);Wr=torch.tensor(Wi[:3,:3],device=dev);Wt=torch.tensor(Wi[:3,3],device=dev);gl=dr.RasterizeCudaContext();HH,WW=1152,1536
angles=np.arange(0,360,45);Rms=[R.from_rotvec([0,0,np.deg2rad(a)]).as_matrix().astype(np.float32) for a in angles]
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:];per=f['person'][:]
def rend(T):
 vw=V@T[:3,:3].T+T[:3,3];vc=vw@Wr.T+Wt;x,y,z=vc.unbind(1);u=Kt[0,0]*x+Kt[0,2]*z;v=Kt[1,1]*y+Kt[1,2]*z;clip=torch.stack([2*u/WW-z,2*v/HH-z,.5*z,z],1)[None];r,_=dr.rasterize(gl,clip,F,[HH,WW]);return r[0,:,:,3]>0
out=[]
for i in range(len(raw)):
 m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev);row=[]
 for Rm in Rms:
  sh=(cent-Rm@cent).astype(np.float32);T=torch.tensor(raw[i],device=dev);Tc=torch.eye(4, …[+801 chars]
```
<details><summary>output</summary>

```
0 0.22334812421665737 [0.03190895 0.22899206 0.39757019]
45 0.2066986806850208 [0.0315184  0.21914068 0.3566571 ]
90 0.20073311579149766 [0.02687772 0.21667765 0.35239403]
135 0.20059104398862507 [0.03308832 0.19100392 0.36844863]
180 0.20666470662586325 [0.02998286 0.22147552 0.37802127]
225 0.2015660979691082 [0.02838516 0.198207   0.35626558]
270 0.2103818261474037 [0.0291345  0.22101883 0.36609809]
315 0.21636982287352532 [0.02906272 0.23360144 0.36898186]
best counts [225 113  33  86 118  97 128  86]
0 105 {0: 2, 45: 4, 90: 0, 135: 16, 180: 17, 225: 3, 270: 63, 315: 0} mean best 0.3291116030657419
105 200 {0: 36, 45: 26, 90: 0, 135: 1, 180: 5, 225: 16, 270: 7, 315: 4} mean best 0.3809298284950827
200 300 {0: 29, 45: 23, 90: 7, 135: 0, 180: 5, 225: 18, 270: 12, 315: 6} mean best 0.1692 …[+630 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/view_roll_base.py <<'PY'
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
from scipy.spatial.transform import Rotation as R
import h5py,cv2
Ts=np.load('/work/scratch/static_start.npz')['T']
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for fi in [0,50,100]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=objs[fi]
 tiles=[]
 for ang in range(0,360,45):
  Rz=R.from_rotvec([0,0,np.deg2rad(ang)]).as_matrix().astype(np.float32);H=np.eye(4,dtype=np.float32);H[:3,:3]=Rz;sh=(cent-Rz@cent).astype(np.float32);Tc=Ts.copy();Tc[:3,:3]=Ts[:3,:3]@Rz;Tc[:3,3]=Ts[:3,3]+Ts[:3,:3]@sh
  # full-res render because opt_pose rend half; use full resolution separately? current rend half. scale masks/img half
  imh=cv2.resize(im,(WW,HH));mh=cv2.resize(m.astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);rm=rend(torch.tensor(Tc,device=dev))
  # focus around bottom 30% of combined bbox
  mm=rm|mh;y,x=np.where(mm);x0=max(0,x.min()-20);x1=min(WW,x.max()+20);y0=max(0,int((y.min()+.65*(y.max()-y.min())))-30);y1=min(HH,y.max()+20)
  o=imh.copy();o[rm]=(.5*o[rm]+.5*np.array([0,255,0])).astype(np.uint8);cnt,_=cv2.findContours(rm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(o,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(mh.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawCo …[+505 chars]
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
z=np.load('/work/inputs/reconstruction.npz');b=z['T_world_mesh'].astype(float);fp=z['T_cam_mesh_foundationpose'].astype(float);co=z['T_cam_mesh_coconet'].astype(float);W=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);fpw=W@fp;cow=W@co
for i in [0,50,100,120,200,300,400,500,600,700,800,850,880]:
 for n,A in [('fp',fpw),('co',cow)]:
  rel=b[i,:3,:3].T@A[i,:3,:3];a=R.from_matrix(rel).magnitude()*180/np.pi
  print(i,n,'rot',a,'t',A[i,:3,3])
 print()
PY"
```
<details><summary>output</summary>

```
0 fp rot 30.693940111119563 t [0.00575257 0.97635623 3.14946022]
0 co rot 1.2253897108791476 t [0.00594608 0.80283537 2.96344533]

50 fp rot 53.50987152767228 t [-2.07157963e-03  1.02454527e+00  3.19969994e+00]
50 co rot 1.230586418184522 t [0.00644007 0.75702559 2.9187831 ]

100 fp rot 57.56968331869825 t [0.00782163 1.07682057 3.23871447]
100 co rot 1.230586418184522 t [0.00644007 0.75702559 2.9187831 ]

120 fp rot 86.5272011493548 t [0.00849922 1.02202452 3.19891676]
120 co rot 1.2297283274358561 t [0.00757064 0.72135763 2.78298401]

200 fp rot 148.99882475577013 t [-0.99682598  0.33973524  5.99594339]
200 co rot 1.248123354914118 t [-0.36341616  0.03948999  2.64761982]

300 fp rot 21.61684044115742 t [-0.33366825  0.52456132  3.12731372]
300 co rot 1.2638577695378994 t [-0.2643207   0. …[+971 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_fp_rot.py <<'PY'
# setup rendering mask from eval_fp prefix
exec(open('/work/scratch/eval_fp.py').read().split(\"for name,A\")[0])
cent=np.array(g.vertices).mean(0)
# base centroid
cb=base[:,:3,:3]@cent+base[:,:3,3]
# fp rotation with base centroid
Tfp=np.tile(np.eye(4),(len(base),1,1));Tfp[:,:3,:3]=fpw[:,:3,:3];Tfp[:,:3,3]=cb-fpw[:,:3,:3]@cent
# hybrid: slerp rotations maybe
for name,A in [('base',base),('fprot',Tfp)]:
 io=[]
 for i in range(len(A)):
  rm=rend(torch.tensor(A[i],device=dev));m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev);rv=rm&~p;inter=(rv&m).sum().item();uni=(rv|m).sum().item();io.append(inter/max(uni,1))
 io=np.array(io);print(name,io.mean(),'ph0',io[:105].mean(),'dyn',io[105:804].mean(),'ph1',io[804:].mean(),np.quantile(io,[.1,.5,.9]));np.save('/work/scratch/iou_'+name+'.npy',io)
PY
python /work/scratch/eval_fp_rot.py"
```
<details><summary>output</summary>

```
base 0.22334812421665737 ph0 0.23998361303809101 dyn 0.21937363368229992 ph1 0.2359266919881858 [0.03190895 0.22899206 0.39757019]
Traceback (most recent call last):
  File "/work/scratch/eval_fp_rot.py", line 12, in <module>
    rm=rend(torch.tensor(A[i],device=dev));m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev);rv=rm&~p;inter=(rv&m).sum().item();uni=(rv|m).sum().item();io.append(inter/max(uni,1))
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 11, in rend
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/cent=np.array(g.vertices).mean(0)/cent=np.array(g.vertices).mean(0).astype(np.float32)/' /work/scratch/eval_fp_rot.py
python /work/scratch/eval_fp_rot.py"
```
<details><summary>output</summary>

```
base 0.22334812421665737 ph0 0.23998361303809101 dyn 0.21937363368229992 ph1 0.2359266919881858 [0.03190895 0.22899206 0.39757019]
Traceback (most recent call last):
  File "/work/scratch/eval_fp_rot.py", line 12, in <module>
    rm=rend(torch.tensor(A[i],device=dev));m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev);rv=rm&~p;inter=(rv&m).sum().item();uni=(rv|m).sum().item();io.append(inter/max(uni,1))
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 11, in rend
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/rm=rend(torch.tensor(A\\[i\\],device=dev))/rm=rend(torch.tensor(A[i],device=dev,dtype=torch.float32))/' /work/scratch/eval_fp_rot.py
python /work/scratch/eval_fp_rot.py"
```
<details><summary>output</summary>

```
base 0.22334812421665737 ph0 0.23998361303809101 dyn 0.21937363368229992 ph1 0.2359266919881858 [0.03190895 0.22899206 0.39757019]
fprot 0.21148380065500713 ph0 0.24013519017014723 dyn 0.2019424891107772 ph1 0.25612990882972636 [0.0262146  0.22410528 0.36521341]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/optcmp_fp.log'
```
<details><summary>output</summary>

```
fp 10 frame 195 iou 0.38055130168453294 base 0.14267515923566879
fp 20 frame 295 iou 0.45364238410596025 base 0.36298649722001586
fp 30 frame 395 iou 0.4399363564041368 base 0.22582872928176795
fp 40 frame 495 iou 0.4607948442534909 base 0.09866220735785954
fp 50 frame 595 iou 0.5549792531120332 base 0.1678714859437751

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat >> /work/scratch/opt_compare_fp.py <<'PY'
exec(open('/work/scratch/opt_compare.py').read().split(\"frames=list\")[0])
# Build smoothed fp orientation, baseline smoothed centroid
fpw=W@np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_foundationpose'].astype(np.float32)
q=R.from_matrix(fpw[:,:3,:3]).as_quat().astype(np.float64)
for i in range(1,len(q)):
 if q[i]@q[i-1]<0:q[i]*=-1
q=gaussian_filter1d(q,2.0,axis=0,mode='nearest');q/=np.linalg.norm(q,axis=1,keepdims=True)
Tb=smooth_raw(raw,2.0);cb=Tb[:,:3,:3]@cent+Tb[:,:3,3]
Tfp=np.tile(np.eye(4),(len(raw),1,1));Tfp[:,:3,:3]=R.from_quat(q).as_matrix();Tfp[:,:3,3]=cb-Tfp[:,:3,:3]@cent;starts={'fp':Tfp.astype(np.float32)}
frames=list(range(105,805,10));tmcache={};pmcache={}
for fi in frames:
 tm=objs[fi].astype(np.uint8);tm=cv2.resize(tm,(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pers[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);tmcache[fi]=tm;pmcache[fi]=pm
k='fp';allres=[]
for fi in frames:
 Ti=torch.tensor(starts[k][fi],device=dev);tm=tmcache[fi];pm=pmcache[fi]
 def make(x):
  Tt=Ti.clone();dw=torch.tensor(W[:3,:3],device=dev)@torch.tensor(x[:3],device=dev,dtype=torch.float32);Tt[:3,3]+=dw
  Rd=R.from_rotvec(np.deg2rad(x[3:])).as_matrix().astype(np.float32);Rd=torch.tensor(Rd,device=dev);Rcam=torch.tensor(Wi[:3,:3],device=dev)@Tt[:3,:3];Rcam=Rd@Rcam;Tt[:3,:3]=torch.tensor(W[:3,:3],device=dev)@Rcam;return Tt
 def obj(x):
  ml=loss_mask(rend(make(x)) …[+707 chars]
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:118: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2356: RuntimeWarning: invalid value encountered in scalar subtract
  r = (xf - nfc) * (fx - ffulc)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2357: RuntimeWarning: invalid value encountered in scalar subtract
  q = (xf - fulc) * (fx - fnfc)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2358: RuntimeWarning: invalid value encountered in scalar subtract
  p = (xf - fulc) * q - (xf - nfc) * r
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2359: RuntimeWarning: invalid value encountered …[+760 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
A=np.load('/work/scratch/optcmp_id.npy');B=np.load('/work/scratch/optcmp_xy.npy');C=np.load('/work/scratch/optcmp_fp.npy')
S=np.stack([A[:,1],B[:,1],C[:,1]],1);names=['id','xy','fp'];best=S.argmax(1)
print('mean best',S.max(1).mean(),'counts',np.bincount(best,minlength=3))
for a,b in [(105,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,805)]:
 m=(A[:,0]>=a)&(A[:,0]<b);print(a,b,np.round(S[m].mean(0),3),np.bincount(best[m],minlength=3),S[m].max(1).mean())
for i in range(len(A)):
 if best[i]"'!=0 and S[i,best[i]]-S[i,0]>.02:print(int(A[i,0]),names[best[i]],np.round(S[i],3))
PY'
```
<details><summary>output</summary>

```
mean best 0.5158997066605813 counts [17 32 21]
105 200 [0.579 0.432 0.574] [5 0 5] 0.6028833094677284
200 300 [0.385 0.338 0.389] [4 2 4] 0.4184257679700745
300 400 [0.443 0.59  0.442] [ 0 10  0] 0.5895963565450806
400 500 [0.474 0.545 0.464] [1 9 0] 0.5455796425652355
500 600 [0.386 0.409 0.386] [2 4 4] 0.4561425933357456
600 700 [0.532 0.53  0.536] [2 5 3] 0.589724311296999
700 805 [0.356 0.338 0.322] [3 2 5] 0.40894596544320566
165 fp [0.528 0.417 0.55 ]
195 fp [0.197 0.266 0.381]
215 fp [0.168 0.215 0.338]
225 fp [0.571 0.45  0.612]
275 fp [0.588 0.426 0.612]
295 xy [0.428 0.505 0.454]
305 xy [0.447 0.581 0.464]
315 xy [0.487 0.634 0.465]
325 xy [0.423 0.632 0.469]
345 xy [0.459 0.611 0.489]
355 xy [0.463 0.662 0.468]
365 xy [0.46  0.672 0.446]
375 xy [0.41  0.592 0.414]
385 xy [0.433  …[+769 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/viterbi_opt.py <<'PY'
exec(open('/work/scratch/opt_compare.py').read().split(\"frames=list\")[0])
# Rebuild starts id,xy and fp exactly
fpw=W@np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_foundationpose'].astype(np.float32)
def smoothq(A,sigma=2.0):
 q=R.from_matrix(A[:,:3,:3]).as_quat().astype(np.float64)
 for i in range(1,len(q)):
  if q[i]@q[i-1]<0:q[i]*=-1
 return gaussian_filter1d(q,sigma,axis=0,mode='nearest')
qfp=smoothq(fpw);qfp/=np.linalg.norm(qfp,axis=1,keepdims=True)
Tb=smooth_raw(raw,2.0);cb=Tb[:,:3,:3]@cent+Tb[:,:3,3]
Tfp=np.tile(np.eye(4),(len(raw),1,1));Tfp[:,:3,:3]=R.from_quat(qfp).as_matrix();Tfp[:,:3,3]=cb-Tfp[:,:3,:3]@cent;Tfp=Tfp.astype(np.float32)
starts={'id':starts_id if False else smooth_raw(flip_raw(raw,np.eye(3,dtype=np.float32)),2.0),'xy':smooth_raw(flip_raw(raw,np.diag([-1,-1,1]).astype(np.float32)),2.0),'fp':Tfp}
# reconstruct candidate poses
arrs={'id':np.load('/work/scratch/optcmp_id.npy'),'xy':np.load('/work/scratch/optcmp_xy.npy'),'fp':np.load('/work/scratch/optcmp_fp.npy')}
names=list(arrs);frames=arrs['id'][:,0].astype(int);Kf=len(frames);poses=np.zeros((Kf,3,4,4),np.float32);scores=np.zeros((Kf,3))
def make_pose(Ti,x):
 Tt=Ti.copy();Tt[:3,3]+=W[:3,:3]@x[:3].astype(np.float32);Rd=R.from_rotvec(np.deg2rad(x[3:])).as_matrix().astype(np.float32);Rcam=Wi[:3,:3]@Tt[:3,:3];Rcam=Rd@Rcam;Tt[:3,:3]=W[:3,:3]@Rcam;return Tt
for k,n in enumerate(names):
 A=arrs[n]
 for j,row in enumerate(A):
  fi=int(row[0] …[+1419 chars]
```
<details><summary>output</summary>

```
lam 0 miou 0.5158997066605813 best 0.5158997066605813 counts [17 32 21] rot med/p90/max 30.88746050899083 173.25343356772711 179.81567551382986 tr 0.23596595227718353 0.4450457513332367 0.5337192416191101
 runs [(105, 135, 'id'), (145, 165, 'fp'), (175, 175, 'id'), (185, 195, 'fp'), (205, 205, 'id'), (215, 225, 'fp'), (235, 235, 'xy'), (245, 255, 'id'), (265, 275, 'fp'), (285, 285, 'id'), (295, 445, 'xy'), (455, 455, 'id'), (465, 495, 'xy'), (505, 505, 'fp'), (515, 535, 'xy'), (545, 545, 'id'), (555, 565, 'fp'), (575, 575, 'xy'), (585, 585, 'id'), (595, 595, 'fp'), (605, 615, 'xy'), (625, 625, 'fp'), (635, 635, 'id'), (645, 655, 'fp'), (665, 665, 'id'), (675, 695, 'xy'), (705, 725, 'id'), (735, 735, 'fp'), (745, 745, 'xy'), (755, 785, 'fp'), (795, 795, 'xy')]
lam 0.002 miou 0.5157988670977 …[+4677 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
z=np.load('/work/scratch/opt_candidates.npz');poses=z['poses'];frames=z['frames'];names=z['names'];rec=np.load('/work/inputs/reconstruction.npz');K=rec['human_keypoints_world'];c=rec['contact_prob_right_left']
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];V=np.array(g.vertices);V=V[V[:,2]>.25][::10];print('handle pts',V.shape)
for j,fi in enumerate(frames):
 if fi in [105,145,195,225,295,355,415,505,555,595,645,705,755,795]:
  row=[]
  for k,n in enumerate(names):
   T=poses[j,k];vw=V@T[:3,:3].T+T[:3,3];dr=np.linalg.norm(K[fi,21:42,None]-vw[None],axis=2).min();dl=np.linalg.norm(K[fi,42:63,None]-vw[None],axis=2).min();row.append(f'{n} R{dr:.3f} L{dl:.3f}')
  print(fi,'cp',c[fi],' | '.join(row))
PY"
```
<details><summary>output</summary>

```
handle pts (7822, 3)
105 cp [2.2901336e-04 9.9997157e-01] id R0.007 L0.368 | xy R0.003 L0.370 | fp R0.372 L0.421
145 cp [0.9209579 0.9999987] id R0.003 L0.424 | xy R0.002 L0.426 | fp R0.001 L0.403
195 cp [2.1814657e-04 9.9961585e-01] id R0.038 L0.634 | xy R0.031 L0.654 | fp R0.080 L0.620
225 cp [1.03999455e-05 9.99997914e-01] id R0.028 L0.521 | xy R0.059 L0.549 | fp R0.002 L0.491
295 cp [1.9468134e-05 9.9880081e-01] id R0.010 L0.577 | xy R0.055 L0.624 | fp R0.000 L0.543
355 cp [8.5337146e-05 9.9958807e-01] id R0.001 L0.472 | xy R0.003 L0.487 | fp R0.048 L0.416
415 cp [0.99997514 0.994761  ] id R0.101 L0.000 | xy R0.094 L0.023 | fp R0.086 L0.001
505 cp [0.9999091  0.69747376] id R0.684 L0.028 | xy R0.686 L0.002 | fp R0.761 L0.227
555 cp [0.98659635 0.7164477 ] id R1.314 L0.798 | xy R1.298 L …[+463 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_hand_test.py <<'PY'
exec(open('/work/scratch/opt_compare.py').read().split(\"frames=list\")[0])
import sys
# starts id,xy,fp
fpw=W@np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_foundationpose'].astype(np.float32)
def smoothq(A,sigma=2.0):
 q=R.from_matrix(A[:,:3,:3]).as_quat().astype(np.float64)
 for i in range(1,len(q)):
  if q[i]@q[i-1]<0:q[i]*=-1
 return gaussian_filter1d(q,sigma,axis=0,mode='nearest')
qfp=smoothq(fpw);qfp/=np.linalg.norm(qfp,axis=1,keepdims=True)
Tb=smooth_raw(raw,2.0);cb=Tb[:,:3,:3]@cent+Tb[:,:3,3]
Tfp=np.tile(np.eye(4),(len(raw),1,1));Tfp[:,:3,:3]=R.from_quat(qfp).as_matrix();Tfp[:,:3,3]=cb-Tfp[:,:3,:3]@cent;Tfp=Tfp.astype(np.float32)
starts={'id':smooth_raw(flip_raw(raw,np.eye(3,dtype=np.float32)),2.0),'xy':smooth_raw(flip_raw(raw,np.diag([-1,-1,1]).astype(np.float32)),2.0),'fp':Tfp}
# handle points
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];vv=np.array(g.vertices);Hp=torch.tensor(vv[vv[:,2]>.20][::10],dtype=torch.float32,device=dev)
hk=torch.tensor(np.load('/work/inputs/reconstruction.npz')['human_keypoints_world'].astype(np.float32),device=dev);cp=torch.tensor(np.load('/work/inputs/reconstruction.npz')['contact_prob_right_left'].astype(np.float32),device=dev)
def hand_loss(T,fi):
 vw=Hp@T[:3,:3].T+T[:3,3];kp=torch.cat([hk[fi,21:42],hk[fi,42:63]],0);d=torch.cdist(vw,kp).min();return (torch.clamp(d/.05,max=4)**2)*0.3,d
for fi in [int(x) for x in sys. …[+1308 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py", line 566, in function_wrapper
    fx = np.asarray(fx).item()
         ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/work/scratch/opt_hand_test.py", line 30, in <module>
    r=minimize(obj,np.zeros(6),method='Powell',bounds=[(-.08,.08)]*3+[(-25,25)]*3,options={'maxiter':30,'maxfev':300,'xtol':1e-3,'ftol':1e-4})
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+796 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_hand_test.py';s=open(p).read();s=s.replace(\"return ml+0.005*np.sum((x[:3]/.05)**2)+0.0005*np.sum((x[3:]/10)**2)\",\"return float(ml+0.005*np.sum((x[:3]/.05)**2)+0.0005*np.sum((x[3:]/10)**2))\")
open(p,'w').write(s)
PY
python /work/scratch/opt_hand_test.py 505 555 595 645 705 755 > /work/scratch/handtest.log
cat /work/scratch/handtest.log"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:118: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2357: RuntimeWarning: invalid value encountered in scalar multiply
  q = (xf - fulc) * (fx - fnfc)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2358: RuntimeWarning: invalid value encountered in scalar subtract
  p = (xf - fulc) * q - (xf - nfc) * r
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2359: RuntimeWarning: invalid value encountered in scalar subtract
  q = 2.0 * (q - r)
FRAME 505 cp [0.9999091  0.69747376] contact? True
id iou 0.396 hd 0.002 x [ 1.1000e-02 -2.6000e-02 -5.0000e-02 -2.0526e+01 -1. …[+2045 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_interp.py <<'PY'
# imports/render from eval_smooth
exec(open('/work/scratch/eval_smooth.py').read().split(\"names=['raw'\")[0])
from scipy.spatial.transform import Rotation as R
cand=np.load('/work/scratch/opt_candidates.npz');ca=cand['frames'].astype(int);poses=cand['poses'];names=list(cand['names']);scores=cand['scores']
N=len(raw);Tint=np.zeros((3,N,4,4),np.float32)
for k in range(3):
 qs=R.from_matrix(poses[:,k,:3,:3]).as_quat()
 for i in range(1,len(qs)):
  if qs[i]@qs[i-1]<0:qs[i]*=-1
 for fi in range(105,805):
  # bracketing anchors
  j=np.searchsorted(ca,fi);j0=max(0,min(j-1,len(ca)-1));j1=min(len(ca)-1,j0+1)
  if ca[j0]==fi:a=b=1.0; j1=j0
  else:a=(fi-ca[j0])/(ca[j1]-ca[j0]);b=1-a
  Tint[k,fi]=np.eye(4)
  Tint[k,fi,:3,:3]=R.from_quat((1-a)*qs[j0]+a*qs[j1]).as_matrix() if a<1 else R.from_quat(qs[j0]).as_matrix()
  Tint[k,fi,:3,3]=(1-a)*poses[j0,k,:3,3]+a*poses[j1,k,:3,3]
 # static phases from pooled poses
 Ts0=np.load('/work/scratch/static_start.npz')['T'];Ts1=np.load('/work/scratch/static_end.npz')['T'];Tint[k,:105]=Ts0;Tint[k,804:]=Ts1
# score all frames
io=np.zeros((3,N))
for k in range(3):
 for i in range(N):
  rm=render_mask(Tint[k,i]);m=torch.tensor(obj[i],device=dev);p=torch.tensor(per[i],device=dev);rv=rm&~p;inter=(rv&m).sum().item();uni=(rv|m).sum().item();io[k,i]=inter/max(uni,1)
 print(names[k],io[k].mean(),io[k,:105].mean(),io[k,105:804].mean(),io[k,804:].mean(),np.quantile(io[k],[.1,.5,.9]))
np.savez('/work/scrat …[+1233 chars]
```
<details><summary>output</summary>

```
/work/scratch/eval_interp.py:14: RuntimeWarning: divide by zero encountered in scalar divide
  else:a=(fi-ca[j0])/(ca[j1]-ca[j0]);b=1-a
/work/scratch/eval_interp.py:17: RuntimeWarning: invalid value encountered in add
  Tint[k,fi,:3,3]=(1-a)*poses[j0,k,:3,3]+a*poses[j1,k,:3,3]
static q [ 0.36374821 -0.33319146  0.6007031   0.62914742] [-0.4862619  -0.46746552  0.5065401  -0.53706841] t [0.03939909 0.81315041 2.92832804] [0.20645963 0.77906078 3.80895281]
id 0.3370944443067662 0.5515911472783432 0.29743374585775095 0.40051730289025556 [0.08934307 0.35617177 0.55482782]
xy 0.34032760267947526 0.5515911472783432 0.3015318550397541 0.40051730289025556 [0.10298805 0.36964466 0.55590135]
fp 0.3211903250152024 0.5515911472783432 0.2772748764838946 0.40051730289025556 [0.04917465 0.34829569 0.5538 …[+14304 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/viterbi_opt.py').read().split(\"for lam in\")[0])
for lam in [.2,.5,1,2,5]:
 lt=.5;dp=np.full((Kf,3),np.inf);bk=np.zeros((Kf,3),np.int32);dp[0]=-100*scores[0]
 for j in range(1,Kf):
  cost=-100*scores[j][:,None].T+lam*rot[j]+lt*tr[j];q=dp[j-1,:,None]+cost;bk[j]=q.argmin(0);dp[j]=q[bk[j],range(3)]
 p=[int(dp[-1].argmin())]
 for j in range(Kf-1,0,-1):p.append(int(bk[j,p[-1]]))
 p=np.array(p[::-1]);sr=rot[np.arange(Kf),p,np.r_[p[0],p[:-1]]]
 print('\\nlam',lam,'iou',scores[np.arange(Kf),p].mean(),'counts',np.bincount(p,minlength=3),'r med/p90/max',np.median(sr),np.quantile(sr,.9),sr.max())
 runs=[];s=0
 for j in range(1,Kf+1):
  if j==Kf or p[j]"'!=p[s]:runs.append((int(frames[s]),int(frames[j-1]),names[p[s]]));s=j
 print(runs)
PY'
```
<details><summary>output</summary>

```

lam 0.2 iou 0.49845685106413934 counts [26 32 12] r med/p90/max 23.266916390904605 61.54199425879826 164.11156827160926
[(105, 145, 'id'), (155, 175, 'fp'), (185, 185, 'id'), (195, 215, 'fp'), (225, 265, 'id'), (275, 275, 'fp'), (285, 285, 'id'), (295, 295, 'fp'), (305, 615, 'xy'), (625, 625, 'fp'), (635, 685, 'id'), (695, 695, 'fp'), (705, 745, 'id'), (755, 755, 'fp'), (765, 765, 'id'), (775, 775, 'fp'), (785, 795, 'id')]

lam 0.5 iou 0.49282914012279 counts [29 32  9] r med/p90/max 23.13227682740186 56.64080019274388 141.00227889523026
[(105, 145, 'id'), (155, 175, 'fp'), (185, 275, 'id'), (285, 295, 'fp'), (305, 615, 'xy'), (625, 625, 'fp'), (635, 685, 'id'), (695, 695, 'fp'), (705, 745, 'id'), (755, 755, 'fp'), (765, 765, 'id'), (775, 775, 'fp'), (785, 795, 'id')]

lam 1 iou 0.4924577 …[+1114 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/optcmp_rolls.log'
```
<details><summary>output</summary>

```
90 10 frame 195 iou 0.4259699129057799 base 0.04195804195804196
90 20 frame 295 iou 0.5366695427092321 base 0.25828729281767954
90 30 frame 395 iou 0.5046570702794242 base 0.3029850746268657
90 40 frame 495 iou 0.35968379446640314 base 0.14344262295081966
90 50 frame 595 iou 0.4330949948927477 base 0.13495934959349593

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/optcmp_rolls.log'
```
<details><summary>output</summary>

```
90 10 frame 195 iou 0.4259699129057799 base 0.04195804195804196
90 20 frame 295 iou 0.5366695427092321 base 0.25828729281767954
90 30 frame 395 iou 0.5046570702794242 base 0.3029850746268657
90 40 frame 495 iou 0.35968379446640314 base 0.14344262295081966
90 50 frame 595 iou 0.4330949948927477 base 0.13495934959349593
90 60 frame 695 iou 0.592737978410206 base 0.21305326331582897
90 70 frame 795 iou 0.37777777777777777 base 0.1549636803874092
RESULT 90 0.40983698941579905 0.21037877063289864 0.4413100637390417 [0.17638536 0.44131006 0.52951746]
270 10 frame 195 iou 0.2277432712215321 base 0.042590431738623105
270 20 frame 295 iou 0.442596991290578 base 0.2686671318911375
270 30 frame 395 iou 0.5207972270363952 base 0.37664473684210525

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -15 /work/scratch/optcmp_rolls.log'
```
<details><summary>output</summary>

```
90 10 frame 195 iou 0.4259699129057799 base 0.04195804195804196
90 20 frame 295 iou 0.5366695427092321 base 0.25828729281767954
90 30 frame 395 iou 0.5046570702794242 base 0.3029850746268657
90 40 frame 495 iou 0.35968379446640314 base 0.14344262295081966
90 50 frame 595 iou 0.4330949948927477 base 0.13495934959349593
90 60 frame 695 iou 0.592737978410206 base 0.21305326331582897
90 70 frame 795 iou 0.37777777777777777 base 0.1549636803874092
RESULT 90 0.40983698941579905 0.21037877063289864 0.4413100637390417 [0.17638536 0.44131006 0.52951746]
270 10 frame 195 iou 0.2277432712215321 base 0.042590431738623105
270 20 frame 295 iou 0.442596991290578 base 0.2686671318911375
270 30 frame 395 iou 0.5207972270363952 base 0.37664473684210525
270 40 frame 495 iou 0.423692636072572 base 0.174280879 …[+74 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_compare_rolls.py <<'PY'
exec(open('/work/scratch/opt_compare.py').read().split(\"frames=list\")[0])
angles=[90,270]
starts={str(a):smooth_raw(flip_raw(raw,R.from_rotvec([0,0,np.deg2rad(a)]).as_matrix().astype(np.float32)),2.0) for a in angles}
frames=list(range(105,805,10));tmcache={};pmcache={}
for fi in frames:
 tm=objs[fi].astype(np.uint8);tm=cv2.resize(tm,(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pers[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);tmcache[fi]=tm;pmcache[fi]=pm
for k in starts:
 allres=[]
 for fi in frames:
  Ti=torch.tensor(starts[k][fi],device=dev);tm=tmcache[fi];pm=pmcache[fi]
  def make(x):
   Tt=Ti.clone();Tt[:3,3]+=torch.tensor(W[:3,:3],device=dev)@torch.tensor(x[:3],device=dev,dtype=torch.float32);Rd=R.from_rotvec(np.deg2rad(x[3:])).as_matrix().astype(np.float32);Rd=torch.tensor(Rd,device=dev);Rcam=torch.tensor(Wi[:3,:3],device=dev)@Tt[:3,:3];Rcam=Rd@Rcam;Tt[:3,:3]=torch.tensor(W[:3,:3],device=dev)@Rcam;return Tt
  def obj(x):
   ml=loss_mask(rend(make(x)),tm,pm)[0];return float(ml+0.005*np.sum((x[:3]/.05)**2)+0.0005*np.sum((x[3:]/10)**2))
  r=minimize(obj,np.zeros(6),method='Powell',bounds=[(-.06,.06)]*3+[(-15,15)]*3,options={'maxiter':15,'maxfev':180,'xtol':2e-3,'ftol':2e-4})
  Tt=make(r.x);lq,iou=loss_mask(rend(Tt),tm,pm);baseiou=loss_mask(rend(Ti),tm,pm)[1];allres.append([fi,iou,baseiou,r.fun,r.nfev,*r.x])
  if len(allres)%10==0:print(k,len(allres),'f …[+325 chars]
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:118: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2357: RuntimeWarning: invalid value encountered in scalar multiply
  q = (xf - fulc) * (fx - fnfc)
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2358: RuntimeWarning: invalid value encountered in scalar subtract
  p = (xf - fulc) * q - (xf - nfc) * r
/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py:2359: RuntimeWarning: invalid value encountered in scalar subtract
  q = 2.0 * (q - r)
90 10 frame 195 iou 0.4259699129057799 base 0.04195804195804196
90 20 frame 295 iou 0.5366695427092321 base 0.25828729281767954 …[+983 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
files=['id','xy','fp','roll90','roll270'];A={n:np.load('/work/scratch/optcmp_'+n+'.npy') for n in files};S=np.stack([A[n][:,1] for n in files],1);frames=A['id'][:,0]
best=S.argmax(1);print('best mean',S.max(1).mean(),'counts',dict(zip(files,np.bincount(best,minlength=5))))
for a,b in [(105,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,805)]:
 m=(frames>=a)&(frames<b);print(a,b,np.round(S[m].mean(0),3),dict(zip(files,np.bincount(best[m],minlength=5))),S[m].max(1).mean())
# Compare best roll per frame
for i in range(len(frames)):
 if S[i].max()-S[i,0]>.1:print(int(frames[i]),files[best[i]],np.round(S[i],3))
PY"
```
<details><summary>output</summary>

```
best mean 0.5209905430115463 counts {'id': 14, 'xy': 30, 'fp': 15, 'roll90': 6, 'roll270': 5}
105 200 [0.579 0.432 0.574 0.453 0.488] {'id': 5, 'xy': 0, 'fp': 4, 'roll90': 1, 'roll270': 0} 0.6074251705898531
200 300 [0.385 0.338 0.389 0.35  0.348] {'id': 3, 'xy': 1, 'fp': 4, 'roll90': 1, 'roll270': 1} 0.4218927958498753
300 400 [0.443 0.59  0.442 0.472 0.477] {'id': 0, 'xy': 10, 'fp': 0, 'roll90': 0, 'roll270': 0} 0.5895963565450806
400 500 [0.474 0.545 0.464 0.415 0.435] {'id': 0, 'xy': 9, 'fp': 0, 'roll90': 0, 'roll270': 1} 0.5530215030303518
500 600 [0.386 0.409 0.386 0.389 0.372] {'id': 1, 'xy': 4, 'fp': 1, 'roll90': 1, 'roll270': 3} 0.4688535352187423
600 700 [0.532 0.53  0.536 0.494 0.448] {'id': 2, 'xy': 4, 'fp': 3, 'roll90': 1, 'roll270': 0} 0.5920088618261916
700 805 [0.356 0.338  …[+973 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/viterbi_opt.py').read().split(\"# transition matrices\")[0])
# add roll candidates
files=['id','xy','fp','roll90','roll270'];arrs={n:np.load('/work/scratch/optcmp_'+n+'.npy') for n in files}
# starts
R90=R.from_rotvec([0,0,np.pi/2]).as_matrix().astype(np.float32);R270=R.from_rotvec([0,0,3*np.pi/2]).as_matrix().astype(np.float32)
starts['roll90']=smooth_raw(flip_raw(raw,R90),2.0);starts['roll270']=smooth_raw(flip_raw(raw,R270),2.0)
names=files;Kf=len(frames);poses=np.zeros((Kf,5,4,4),np.float32);scores=np.zeros((Kf,5))
for k,n in enumerate(names):
 for j,row in enumerate(arrs[n]):
  poses[j,k]=make_pose(starts[n][frames[j]],row[5:11]);scores[j,k]=row[1]
rot=np.zeros((Kf,5,5));tr=np.zeros_like(rot)
for j in range(1,Kf):
 for a in range(5):
  for b in range(5):
   rel=poses[j,a,:3,:3].T@poses[j-1,b,:3,:3];rot[j,a,b]=R.from_matrix(rel).magnitude()*180/np.pi;tr[j,a,b]=np.linalg.norm(poses[j,a,:3,3]-poses[j-1,b,:3,3])
for lam in [.1,.2,.5,1]:
 dp=np.full((Kf,5),np.inf);bk=np.zeros((Kf,5),np.int32);dp[0]=-100*scores[0]
 for j in range(1,Kf):
  cost=-100*scores[j][:,None].T+lam*rot[j]+.5*tr[j];q=dp[j-1,:,None]+cost;bk[j]=q.argmin(0);dp[j]=q[bk[j],range(5)]
 p=[int(dp[-1].argmin())]
 for j in range(Kf-1,0,-1):p.append(int(bk[j,p[-1]]))
 p=np.array(p[::-1]);sr=rot[np.arange(Kf),p,np.r_[p[0],p[:-1]]]
 print('\\nlam',lam,'iou',scores[np.arange(Kf),p].mean(),'best',scores.max(1).mean(),'counts',dict(zip(names,np.bincount(p,minleng …[+341 chars]
```
<details><summary>output</summary>

```

lam 0.1 iou 0.5116866476576135 best 0.5209905430115463 counts {'id': 19, 'xy': 32, 'fp': 10, 'roll90': 5, 'roll270': 4} rmed/p90/max 24.04521656466705 128.01320114824534 175.00784573147135
[(105, 135, 'id'), (145, 165, 'fp'), (175, 185, 'id'), (195, 215, 'fp'), (225, 265, 'id'), (275, 275, 'fp'), (285, 285, 'id'), (295, 295, 'roll270'), (305, 495, 'xy'), (505, 505, 'roll90'), (515, 545, 'xy'), (555, 565, 'roll270'), (575, 615, 'xy'), (625, 625, 'fp'), (635, 645, 'id'), (655, 655, 'fp'), (665, 665, 'id'), (675, 695, 'xy'), (705, 735, 'id'), (745, 745, 'roll270'), (755, 755, 'fp'), (765, 795, 'roll90')]

lam 0.2 iou 0.49779232301294524 best 0.5209905430115463 counts {'id': 22, 'xy': 32, 'fp': 10, 'roll90': 4, 'roll270': 2} rmed/p90/max 23.13227682740186 65.24978409552845 175.00784573147135
 …[+1203 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_all.py <<'PY'
exec(open('/work/scratch/opt_compare.py').read().split(\"frames=list\")[0])
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
# Load candidate path
C=np.load('/work/scratch/cand5.npz');ca=C['frames'].astype(int);poses=C['poses'];names=list(C['names']);path=C['path'];scores=C['scores'];N=len(raw)
# interpolation along selected path
qs=R.from_matrix(poses[np.arange(len(ca)),path,:3,:3]).as_quat();qs=R.from_quat(qs).as_quat()
for i in range(1,len(qs)):
 if qs[i]@qs[i-1]<0:qs[i]*=-1
Traw=np.zeros((N,4,4),np.float32)
Ts0=np.load('/work/scratch/static_start.npz')['T'];Ts1=np.load('/work/scratch/static_end.npz')['T']
for fi in range(N):
 if fi<=105:Traw[fi]=Ts0
 elif fi>=804:Traw[fi]=Ts1
 else:
  j=np.searchsorted(ca,fi);j0=max(0,min(j-1,len(ca)-1));j1=min(len(ca)-1,j0+1)
  if ca[j0]==fi: a=0.;q=qs[j0];t=poses[j0,path[j0],:3,3]
  else:
   a=(fi-ca[j0])/(ca[j1]-ca[j0]);q=R.from_quat((1-a)*qs[j0]+a*qs[j1]).as_quat();t=(1-a)*poses[j0,path[j0],:3,3]+a*poses[j1,path[j1],:3,3]
  Traw[fi]=np.eye(4);Traw[fi,:3,:3]=R.from_quat(q).as_matrix();Traw[fi,:3,3]=t
# smooth interpolation mildly
q=R.from_matrix(Traw[:,:3,:3]).as_quat()
for i in range(1,N):
 if q[i]@q[i-1]<0:q[i]*=-1
q=gaussian_filter1d(q,0.7,axis=0,mode='nearest');q/=np.linalg.norm(q,axis=1,keepdims=True)
Tbase=Traw.copy();Tbase[:,:3,:3]=R.from_quat(q).as_matrix();Tbase[:,:3,3]=gaussian_filter1d(Traw[:,:3,3],0.7,axis=0,mode='nearest …[+3573 chars]
```
<details><summary>output</summary>

```
/work/scratch/refine_all.py:19: RuntimeWarning: divide by zero encountered in scalar divide
  a=(fi-ca[j0])/(ca[j1]-ca[j0]);q=R.from_quat((1-a)*qs[j0]+a*qs[j1]).as_quat();t=(1-a)*poses[j0,path[j0],:3,3]+a*poses[j1,path[j1],:3,3]
/work/scratch/refine_all.py:19: RuntimeWarning: invalid value encountered in add
  a=(fi-ca[j0])/(ca[j1]-ca[j0]);q=R.from_quat((1-a)*qs[j0]+a*qs[j1]).as_quat();t=(1-a)*poses[j0,path[j0],:3,3]+a*poses[j1,path[j1],:3,3]
Traceback (most recent call last):
  File "/work/scratch/refine_all.py", line 19, in <module>
    a=(fi-ca[j0])/(ca[j1]-ca[j0]);q=R.from_quat((1-a)*qs[j0]+a*qs[j1]).as_quat();t=(1-a)*poses[j0,path[j0],:3,3]+a*poses[j1,path[j1],:3,3]
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "scipy/spatial/transform/_rotation.pyx", l …[+230 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/refine.log'
```
<details><summary>output</summary>

```
125 area 3735 iou 0.576 -> 0.593 hd0 0.002 hand True
150 area 2609 iou 0.575 -> 0.622 hd0 0.002 hand True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -30 /work/scratch/refine.log; ps -o pid,etime,pcpu,pmem -C python | tail -2'
```
<details><summary>output</summary>

```
125 area 3735 iou 0.576 -> 0.593 hd0 0.002 hand True
150 area 2609 iou 0.575 -> 0.622 hd0 0.002 hand True
175 area 2725 iou 0.65 -> 0.652 hd0 0.002 hand True
200 area 3987 iou 0.074 -> 0.101 hd0 0.497 hand False
225 area 3754 iou 0.4 -> 0.556 hd0 0.042 hand True
250 area 1048 iou 0.082 -> 0.103 hd0 0.0 hand True
     15       55:14  0.1  0.0
  13830       03:40 97.5  3.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -40 /work/scratch/refine.log'
```
<details><summary>output</summary>

```
125 area 3735 iou 0.576 -> 0.593 hd0 0.002 hand True
150 area 2609 iou 0.575 -> 0.622 hd0 0.002 hand True
175 area 2725 iou 0.65 -> 0.652 hd0 0.002 hand True
200 area 3987 iou 0.074 -> 0.101 hd0 0.497 hand False
225 area 3754 iou 0.4 -> 0.556 hd0 0.042 hand True
250 area 1048 iou 0.082 -> 0.103 hd0 0.0 hand True
275 area 3816 iou 0.556 -> 0.606 hd0 0.001 hand True
300 area 3987 iou 0.383 -> 0.418 hd0 0.003 hand True
325 area 2113 iou 0.599 -> 0.641 hd0 0.05 hand True
350 area 2579 iou 0.565 -> 0.688 hd0 0.0 hand True
375 area 4108 iou 0.582 -> 0.61 hd0 0.002 hand True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -50 /work/scratch/refine.log'
```
<details><summary>output</summary>

```
125 area 3735 iou 0.576 -> 0.593 hd0 0.002 hand True
150 area 2609 iou 0.575 -> 0.622 hd0 0.002 hand True
175 area 2725 iou 0.65 -> 0.652 hd0 0.002 hand True
200 area 3987 iou 0.074 -> 0.101 hd0 0.497 hand False
225 area 3754 iou 0.4 -> 0.556 hd0 0.042 hand True
250 area 1048 iou 0.082 -> 0.103 hd0 0.0 hand True
275 area 3816 iou 0.556 -> 0.606 hd0 0.001 hand True
300 area 3987 iou 0.383 -> 0.418 hd0 0.003 hand True
325 area 2113 iou 0.599 -> 0.641 hd0 0.05 hand True
350 area 2579 iou 0.565 -> 0.688 hd0 0.0 hand True
375 area 4108 iou 0.582 -> 0.61 hd0 0.002 hand True
400 area 4313 iou 0.527 -> 0.624 hd0 0.002 hand True
425 area 3007 iou 0.578 -> 0.67 hd0 0.003 hand True
450 area 1145 iou 0.248 -> 0.364 hd0 0.027 hand True
475 area 2831 iou 0.614 -> 0.632 hd0 0.001 hand True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -60 /work/scratch/refine.log'
```
<details><summary>output</summary>

```
125 area 3735 iou 0.576 -> 0.593 hd0 0.002 hand True
150 area 2609 iou 0.575 -> 0.622 hd0 0.002 hand True
175 area 2725 iou 0.65 -> 0.652 hd0 0.002 hand True
200 area 3987 iou 0.074 -> 0.101 hd0 0.497 hand False
225 area 3754 iou 0.4 -> 0.556 hd0 0.042 hand True
250 area 1048 iou 0.082 -> 0.103 hd0 0.0 hand True
275 area 3816 iou 0.556 -> 0.606 hd0 0.001 hand True
300 area 3987 iou 0.383 -> 0.418 hd0 0.003 hand True
325 area 2113 iou 0.599 -> 0.641 hd0 0.05 hand True
350 area 2579 iou 0.565 -> 0.688 hd0 0.0 hand True
375 area 4108 iou 0.582 -> 0.61 hd0 0.002 hand True
400 area 4313 iou 0.527 -> 0.624 hd0 0.002 hand True
425 area 3007 iou 0.578 -> 0.67 hd0 0.003 hand True
450 area 1145 iou 0.248 -> 0.364 hd0 0.027 hand True
475 area 2831 iou 0.614 -> 0.632 hd0 0.001 hand True
500 area 3336  …[+355 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_all.py';s=open(p).read()
old=\"\"\"  j=np.searchsorted(ca,fi);j0=max(0,min(j-1,len(ca)-1));j1=min(len(ca)-1,j0+1)
  if ca[j0]==fi: a=0.;q=qs[j0];t=poses[j0,path[j0],:3,3]
  else:
   a=(fi-ca[j0])/(ca[j1]-ca[j0]);q=R.from_quat((1-a)*qs[j0]+a*qs[j1]).as_quat();t=(1-a)*poses[j0,path[j0],:3,3]+a*poses[j1,path[j1],:3,3]
  Traw[fi]=np.eye(4);Traw[fi,:3,:3]=R.from_quat(q).as_matrix();Traw[fi,:3,3]=t\"\"\"
new=\"\"\"  j=np.searchsorted(ca,fi);j0=max(0,min(j-1,len(ca)-1));j1=min(len(ca)-1,j0+1)
  if ca[j0]==fi:
   q=qs[j0];t=poses[j0,path[j0],:3,3]
  elif fi>ca[-1]:
   a=(fi-ca[-1])/(804-ca[-1]);q1=qs[-1];q2=R.from_matrix(Ts1[:3,:3]).as_quat()
   if q1@q2<0:q2=-q2
   q=R.from_quat((1-a)*q1+a*q2).as_quat();t=(1-a)*poses[-1,path[-1],:3,3]+a*Ts1[:3,3]
  else:
   a=(fi-ca[j0])/(ca[j1]-ca[j0]);q=R.from_quat((1-a)*qs[j0]+a*qs[j1]).as_quat();t=(1-a)*poses[j0,path[j0],:3,3]+a*poses[j1,path[j1],:3,3]
  Traw[fi]=np.eye(4);Traw[fi,:3,:3]=R.from_quat(q).as_matrix();Traw[fi,:3,3]=t\"\"\"
s=s.replace(old,new)
open(p,'w').write(s)
PY
python /work/scratch/refine_all.py > /work/scratch/refine.log
cat /work/scratch/refine.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/refine_all.py", line 74, in <module>
    rm=render_mask(Tsm[i]);m=torch.tensor(objs[i],device=dev);p=torch.tensor(pers[i],device=dev);rv=rm&~p;inter=(rv&m).sum().item();uni=(rv|m).sum().item();io.append(inter/max(uni,1))
       ^^^^^^^^^^^
NameError: name 'render_mask' is not defined
125 area 3735 iou 0.576 -> 0.593 hd0 0.002 hand True
150 area 2609 iou 0.575 -> 0.622 hd0 0.002 hand True
175 area 2725 iou 0.65 -> 0.652 hd0 0.002 hand True
200 area 3987 iou 0.074 -> 0.101 hd0 0.497 hand False
225 area 3754 iou 0.4 -> 0.556 hd0 0.042 hand True
250 area 1048 iou 0.082 -> 0.103 hd0 0.0 hand True
275 area 3816 iou 0.556 -> 0.606 hd0 0.001 hand True
300 area 3987 iou 0.383 -> 0.418 hd0 0.003 hand True
325 area 2113 iou 0.599 -> 0.641 hd0 0 …[+960 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,cv2,h5py
# load setup
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
T=np.load('/work/scratch/refined_all.npz')['T'];N=len(T)
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:];pers=f['person'][:]
io=[]
for i in range(N):
 rm=rend(torch.tensor(T[i],device=dev));m=cv2.resize(objs[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(pers[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);rv=rm&~p;inter=(rv&m).sum();uni=(rv|m).sum();io.append(inter/max(uni,1))
io=np.array(io);np.save('/work/scratch/refined_iou.npy',io)
print('all',io.mean(),'ph0',io[:105].mean(),'dyn',io[105:804].mean(),'ph1',io[804:].mean(),np.quantile(io,[.1,.5,.9]))
# raw identity half
io0=[]
for i in range(N):
 rm=rend(torch.tensor(raw[i],device=dev));m=cv2.resize(objs[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(pers[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);rv=rm&~p;inter=(rv&m).sum();uni=(rv|m).sum();io0.append(inter/max(uni,1))
io0=np.array(io0);print('raw half',io0.mean(),io0[:105].mean(),io0[105:804].mean(),io0[804:].mean(),np.quantile(io0,[.1,.5,.9]))
print('improved frac',np.mean(io>io0),'mean delta',(io-io0).mean())
PY"
```
<details><summary>output</summary>

```
all 0.4870013935151006 ph0 0.551600033374456 dyn 0.4792377738898886 ph1 0.4704637463540127 [0.22139213 0.54130133 0.63522749]
raw half 0.22740283307738007 0.2573135079536016 0.22170135883104944 0.23770417010398806 [0.03266684 0.23343706 0.39958699]
improved frac 0.9627539503386005 mean delta 0.2595985604377204

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
io=np.load('/work/scratch/refined_iou.npy');
with h5py.File('/work/inputs/masks.h5') as f:a=f['object'][:].sum((1,2))
for i in np.where((io<.25)&(a>700))[0]:print(i,int(a[i]),round(io[i],3))
PY"
```
<details><summary>output</summary>

```
191 3968 0.229
192 4269 0.133
193 4071 0.231
199 3845 0.214
200 3987 0.126
201 4010 0.137
202 4064 0.239
203 4051 0.207
204 3937 0.243
216 4490 0.229
217 3987 0.08
218 4123 0.024
219 4132 0.018
220 4472 0.012
221 4076 0.012
222 3929 0.021
223 4013 0.071
224 3643 0.129
237 1270 0.229
238 917 0.198
242 770 0.118
243 906 0.136
244 1073 0.109
245 1169 0.106
246 1105 0.094
247 1105 0.084
248 1115 0.088
249 1102 0.088
250 1048 0.1
251 835 0.113
258 1267 0.207
454 840 0.249
456 766 0.213
457 833 0.169
458 948 0.125
459 996 0.167
499 3723 0.166
500 3336 0.16
501 3225 0.16
502 3195 0.202
746 808 0.073
747 1003 0.0
753 712 0.0
755 740 0.241
756 1233 0.155
757 1685 0.099
758 1902 0.157
759 1981 0.227
794 2338 0.175
795 2344 0.126
796 2338 0.126
797 2344 0.216

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/second.log'
```
<details><summary>output</summary>

```
191 BEST fp fun 0.409 iou 0.598 hand 0.0
192 BEST fp fun 0.502 iou 0.51 hand 0.0
193 BEST current fun 0.519 iou 0.494 hand 0.007
194 BEST current fun 0.543 iou 0.467 hand 0.004
195 BEST fp fun 0.579 iou 0.453 hand 0.01
196 BEST current fun 0.623 iou 0.392 hand 0.006

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -30 /work/scratch/second.log'
```
<details><summary>output</summary>

```
191 BEST fp fun 0.409 iou 0.598 hand 0.0
192 BEST fp fun 0.502 iou 0.51 hand 0.0
193 BEST current fun 0.519 iou 0.494 hand 0.007
194 BEST current fun 0.543 iou 0.467 hand 0.004
195 BEST fp fun 0.579 iou 0.453 hand 0.01
196 BEST current fun 0.623 iou 0.392 hand 0.006
197 BEST id fun 0.446 iou 0.565 hand 0.004
198 BEST fp fun 0.734 iou 0.299 hand 0.696
199 BEST current fun 0.66 iou 0.353 hand 0.477
200 BEST fp fun 0.775 iou 0.264 hand 0.746
201 BEST id fun 0.471 iou 0.539 hand 0.002
202 BEST current fun 0.747 iou 0.289 hand 0.624
203 BEST current fun 0.783 iou 0.25 hand 0.633
204 BEST current fun 0.772 iou 0.262 hand 0.7
216 BEST xy fun 0.635 iou 0.38 hand 0.0
217 BEST current fun 0.723 iou 0.31 hand 0.662

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -30 /work/scratch/second.log'
```
<details><summary>output</summary>

```
191 BEST fp fun 0.409 iou 0.598 hand 0.0
192 BEST fp fun 0.502 iou 0.51 hand 0.0
193 BEST current fun 0.519 iou 0.494 hand 0.007
194 BEST current fun 0.543 iou 0.467 hand 0.004
195 BEST fp fun 0.579 iou 0.453 hand 0.01
196 BEST current fun 0.623 iou 0.392 hand 0.006
197 BEST id fun 0.446 iou 0.565 hand 0.004
198 BEST fp fun 0.734 iou 0.299 hand 0.696
199 BEST current fun 0.66 iou 0.353 hand 0.477
200 BEST fp fun 0.775 iou 0.264 hand 0.746
201 BEST id fun 0.471 iou 0.539 hand 0.002
202 BEST current fun 0.747 iou 0.289 hand 0.624
203 BEST current fun 0.783 iou 0.25 hand 0.633
204 BEST current fun 0.772 iou 0.262 hand 0.7
216 BEST xy fun 0.635 iou 0.38 hand 0.0
217 BEST current fun 0.723 iou 0.31 hand 0.662
218 BEST xy fun 0.84 iou 0.219 hand 0.007
219 BEST id fun 0.803 iou 0.218 hand 0.007
2 …[+389 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/second_pass.py <<'PY'
exec(open('/work/scratch/opt_compare.py').read().split(\"frames=list\")[0])
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
# Build five smoothed starts
R90=R.from_rotvec([0,0,np.pi/2]).as_matrix().astype(np.float32);R270=R.from_rotvec([0,0,3*np.pi/2]).as_matrix().astype(np.float32)
starts={'id':smooth_raw(flip_raw(raw,np.eye(3,dtype=np.float32)),2.0),'xy':smooth_raw(flip_raw(raw,np.diag([-1,-1,1]).astype(np.float32)),2.0),'90':smooth_raw(flip_raw(raw,R90),2.0),'270':smooth_raw(flip_raw(raw,R270),2.0)}
fpw=W@np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_foundationpose'].astype(np.float32);qfp=R.from_matrix(fpw[:,:3,:3]).as_quat().astype(np.float64)
for i in range(1,len(qfp)):
 if qfp[i]@qfp[i-1]<0:qfp[i]*=-1
qfp=gaussian_filter1d(qfp,2.0,axis=0,mode='nearest');qfp/=np.linalg.norm(qfp,axis=1,keepdims=True);Tb=smooth_raw(raw,2.0);cb=Tb[:,:3,:3]@cent+Tb[:,:3,3];Tfp=np.tile(np.eye(4),(len(raw),1,1));Tfp[:,:3,:3]=R.from_quat(qfp).as_matrix();Tfp[:,:3,3]=cb-Tfp[:,:3,:3]@cent;starts['fp']=Tfp.astype(np.float32)
# data
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];vv=np.array(g.vertices);Hp=torch.tensor(vv[vv[:,2]>.20][::10],dtype=torch.float32,device=dev)
rec=np.load('/work/inputs/reconstruction.npz');hk=torch.tensor(rec['human_keypoints_world'].astype(np.float32),device=dev);cp=torch.tensor(rec['contact_prob_right_left' …[+1904 chars]
```
<details><summary>output</summary>

```
191 BEST fp fun 0.409 iou 0.598 hand 0.0
192 BEST fp fun 0.502 iou 0.51 hand 0.0
193 BEST current fun 0.519 iou 0.494 hand 0.007
194 BEST current fun 0.543 iou 0.467 hand 0.004
195 BEST fp fun 0.579 iou 0.453 hand 0.01
196 BEST current fun 0.623 iou 0.392 hand 0.006
197 BEST id fun 0.446 iou 0.565 hand 0.004
198 BEST fp fun 0.734 iou 0.299 hand 0.696
199 BEST current fun 0.66 iou 0.353 hand 0.477
200 BEST fp fun 0.775 iou 0.264 hand 0.746
201 BEST id fun 0.471 iou 0.539 hand 0.002
202 BEST current fun 0.747 iou 0.289 hand 0.624
203 BEST current fun 0.783 iou 0.25 hand 0.633
204 BEST current fun 0.772 iou 0.262 hand 0.7
216 BEST xy fun 0.635 iou 0.38 hand 0.0
217 BEST current fun 0.723 iou 0.31 hand 0.662
218 BEST xy fun 0.84 iou 0.219 hand 0.007
219 BEST id fun 0.803 iou 0.218 hand 0.007
2 …[+561 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for f in ['/work/scratch/refined_all.npz','/work/scratch/refined_all2.npz']:
 T=np.load(f)['T'];print('\\n',f)
 for a,b in [(185,230),(490,510),(785,805)]:
  for i in range(a+1,b):
   rel=T[i,:3,:3].T@T[i-1,:3,:3];ang=R.from_matrix(rel).magnitude()*180/np.pi
   if ang>45: print(i,round(ang,1),round(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]),3))
PY"
```
<details><summary>output</summary>

```

 /work/scratch/refined_all.npz
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/refined_all2.npz'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -l /work/scratch/refined*; tail -5 /work/scratch/second_pass.py'
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 90704 Oct 20 19:53 /work/scratch/refined_all.npz
-rw-rw-r-- 1 hanzhou hanzhou 56832 Oct 20 20:00 /work/scratch/refined_all2.npz.npy
-rw-rw-r-- 1 hanzhou hanzhou  7216 Oct 20 19:53 /work/scratch/refined_iou.npy
  Tt=make(r.x);iou=loss_mask(rend(Tt),tm,pm)[1];hd=hand_loss(Tt,fi)[1].item();val=(r.fun,iou,hd,initname,Tt.detach().cpu().numpy())
  if best is None or val[0]<best[0]:best=val
 print(fi,'BEST',best[3],'fun',round(best[0],3),'iou',round(best[1],3),'hand',round(best[2],3))
 Tout[fi]=best[4]
np.save('/work/scratch/refined_all2.npz',Tout)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for f in ['/work/scratch/refined_all.npz','/work/scratch/refined_all2.npz.npy']:
 T=np.load(f)['T'] if f.endswith('.npz') else np.load(f);print('\\n',f)
 for a,b in [(185,230),(490,510),(785,805)]:
  for i in range(a+1,b):
   rel=T[i,:3,:3].T@T[i-1,:3,:3];ang=R.from_matrix(rel).magnitude()*180/np.pi
   if ang>45: print(i,round(ang,1),round(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]),3))
PY"
```
<details><summary>output</summary>

```

 /work/scratch/refined_all.npz

 /work/scratch/refined_all2.npz.npy
195 52.8 0.089
197 90.4 0.045
198 164.3 0.078
199 78.4 0.035
200 106.8 0.032
201 178.5 0.108
202 153.4 0.088
216 170.5 0.084
217 177.8 0.033
218 175.6 0.147
219 169.8 0.201
223 47.5 0.047
499 178.4 0.043
501 172.4 0.038
795 49.5 0.01
798 121.5 0.162

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
io=np.load('/work/scratch/refined_iou.npy');
with h5py.File('/work/inputs/masks.h5') as f:ar=f['object'][:].sum((1,2))
for a,b in [(180,230),(480,515),(780,805)]:
 print('\\n',a,b)
 for i in range(a,b):print(i,int(ar[i]),round(io[i],3),end=' | ')
 print()
PY"
```
<details><summary>output</summary>

```

 180 230
180 3012 0.683 | 181 3177 0.675 | 182 3386 0.652 | 183 3339 0.685 | 184 3464 0.663 | 185 3498 0.607 | 186 3740 0.548 | 187 3732 0.563 | 188 3964 0.553 | 189 3919 0.54 | 190 4193 0.485 | 191 3968 0.229 | 192 4269 0.133 | 193 4071 0.231 | 194 4213 0.332 | 195 4169 0.376 | 196 4272 0.351 | 197 4200 0.375 | 198 3857 0.377 | 199 3845 0.214 | 200 3987 0.126 | 201 4010 0.137 | 202 4064 0.239 | 203 4051 0.207 | 204 3937 0.243 | 205 3969 0.266 | 206 4022 0.27 | 207 4040 0.276 | 208 3985 0.279 | 209 3995 0.276 | 210 4058 0.282 | 211 3972 0.287 | 212 4030 0.297 | 213 3878 0.319 | 214 3817 0.339 | 215 4451 0.307 | 216 4490 0.229 | 217 3987 0.08 | 218 4123 0.024 | 219 4132 0.018 | 220 4472 0.012 | 221 4076 0.012 | 222 3929 0.021 | 223 4013 0.071 | 224 3643 0.129 | 225 3754 0.374 | 226 3406 0. …[+1093 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
# evaluate linear interpolation alternatives
import numpy as np,torch,cv2,h5py
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/refined_all.npz')['T'];cur=np.load('/work/scratch/refined_iou.npy')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:];pers=f['person'][:]
def ev(Ts,ids):
 out=[]
 for i,Ti in zip(ids,Ts):
  rm=rend(torch.tensor(Ti,device=dev));m=cv2.resize(objs[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(pers[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);rv=rm&~p;inter=(rv&m).sum();uni=(rv|m).sum();out.append(inter/max(uni,1))
 return np.array(out)
for a,b in [(190,226),(495,509),(790,800)]:
 ids=list(range(a+1,b));q0=R.from_matrix(T[a,:3,:3]).as_quat();q1=R.from_matrix(T[b,:3,:3]).as_quat();
 if q0@q1<0:q1=-q1
 Ts=[]
 for i in ids:
  u=(i-a)/(b-a);q=(1-u)*q0+u*q1;q/=np.linalg.norm(q);Ti=np.eye(4);Ti[:3,:3]=R.from_quat(q).as_matrix();Ti[:3,3]=(1-u)*T[a,:3,3]+u*T[b,:3,3];Ts.append(Ti)
 io=ev(Ts,ids);print(a,b,'interp mean',io.mean(),'cur',cur[ids].mean(),'per',np.round(io,3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 18, in <module>
  File "<stdin>", line 10, in ev
  File "<string>", line 18, in rend
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,cv2,h5py
exec(open('/work/scratch/opt_pose.py').read().split(\"if __name__\")[0])
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/refined_all.npz')['T'];cur=np.load('/work/scratch/refined_iou.npy')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:];pers=f['person'][:]
def ev(Ts,ids):
 out=[]
 for i,Ti in zip(ids,Ts):
  rm=rend(torch.tensor(Ti,device=dev,dtype=torch.float32));m=cv2.resize(objs[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(pers[i].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);rv=rm&~p;inter=(rv&m).sum();uni=(rv|m).sum();out.append(inter/max(uni,1))
 return np.array(out)
for a,b in [(190,226),(495,509),(790,800)]:
 ids=list(range(a+1,b));q0=R.from_matrix(T[a,:3,:3]).as_quat();q1=R.from_matrix(T[b,:3,:3]).as_quat();
 if q0@q1<0:q1=-q1
 Ts=[]
 for i in ids:
  u=(i-a)/(b-a);q=(1-u)*q0+u*q1;q/=np.linalg.norm(q);Ti=np.eye(4);Ti[:3,:3]=R.from_quat(q).as_matrix();Ti[:3,3]=(1-u)*T[a,:3,3]+u*T[b,:3,3];Ts.append(Ti)
 io=ev(Ts,ids);print(a,b,'interp mean',io.mean(),'cur',cur[ids].mean(),'per',np.round(io,3))
PY"
```
<details><summary>output</summary>

```
190 226 interp mean 0.011050894654213985 cur 0.22096819023432213 per [0.095 0.059 0.014 0.    0.    0.    0.    0.    0.    0.    0.    0.
 0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
 0.    0.    0.    0.    0.    0.001 0.    0.    0.023 0.055 0.14 ]
495 509 interp mean 0.07806214248907052 cur 0.31279010617986996 per [0.145 0.063 0.058 0.051 0.041 0.033 0.037 0.044 0.068 0.062 0.069 0.098
 0.247]
790 800 interp mean 0.2354609089800902 cur 0.2538739692467699 per [0.328 0.335 0.198 0.145 0.074 0.144 0.249 0.308 0.338]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ids=list(range(180,231,2));ims=[]
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:];per=f['person'][:]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=obj[i];y,x=np.where(m);pad=80;x0=max(0,x.min()-pad);x1=min(1536,x.max()+pad);y0=max(0,y.min()-pad);y1=min(1152,y.max()+pad);o=im[y0:y1,x0:x1];cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cnt=[c-[x0,y0] for c in cnt];cv2.drawContours(o,cnt,-1,(0,255,0),2);o=cv2.resize(o,(300,240));cv2.putText(o,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2);ims.append(o)
rows=[np.concatenate(ims[i:i+6],1) for i in range(0,len(ims),6)];cv2.imwrite('/work/scratch/frames180_230.jpg',cv2.cvtColor(np.concatenate(rows,0),cv2.COLOR_RGB2BGR))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1800 and the array at index 4 has size 600

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ids=list(range(180,231,2));ims=[]
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=obj[i];cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),2);im=cv2.resize(im,(384,288));cv2.putText(im,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2);ims.append(im)
rows=[np.concatenate(ims[i:i+6],1) for i in range(0,len(ims),6)];cv2.imwrite('/work/scratch/frames180_230.jpg',cv2.cvtColor(np.concatenate(rows,0),cv2.COLOR_RGB2BGR))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 2304 and the array at index 4 has size 768

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ids=list(range(180,231,2));ims=[]
with h5py.File('/work/inputs/masks.h5') as f:obj=f['object'][:]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=obj[i];cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),2);im=cv2.resize(im,(384,288));cv2.putText(im,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,0),2);ims.append(im)
rows=[]
for j in range(0,len(ims),6):
 a=ims[j:j+6]
 while len(a)<6:a.append(np.zeros((288,384,3),np.uint8))
 rows.append(np.concatenate(a,1))
cv2.imwrite('/work/scratch/frames180_230.jpg',cv2.cvtColor(np.concatenate(rows,0),cv2.COLOR_RGB2BGR))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
raw=np.load('/work/scratch/iou_id.npy');ref=np.load('/work/scratch/refined_iou.npy')
for a,b in [(190,226),(216,225),(495,510),(790,805)]:
 print(a,b,'raw',np.round(raw[a:b],3),'mean',raw[a:b].mean())
 print(' ref mean',ref[a:b].mean())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/iou_id.npy'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls /work/scratch/io_*.npy | head'
```
<details><summary>output</summary>

```
/work/scratch/io_id_0.npy
/work/scratch/io_id_2.npy
/work/scratch/io_xy_0.npy
/work/scratch/io_xy_2.npy
/work/scratch/io_xz_0.npy
/work/scratch/io_xz_2.npy
/work/scratch/io_yz_0.npy
/work/scratch/io_yz_2.npy

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
raw=np.load('/work/scratch/io_id_0.npy');ref=np.load('/work/scratch/refined_iou.npy')
for a,b in [(190,226),(216,225),(495,510),(790,805)]:
 print(a,b,'raw mean',raw[a:b].mean(),np.round(raw[a:b],3));print(' ref',ref[a:b].mean(),np.round(ref[a:b],3))
PY"
```
<details><summary>output</summary>

```
190 226 raw mean 0.0938344784938254 [0.421 0.305 0.088 0.039 0.021 0.051 0.029 0.049 0.063 0.068 0.054 0.028
 0.021 0.02  0.022 0.023 0.079 0.023 0.057 0.025 0.022 0.022 0.031 0.023
 0.029 0.021 0.042 0.023 0.03  0.031 0.052 0.107 0.199 0.297 0.556 0.406]
 ref 0.22830323884225098 [0.485 0.229 0.133 0.231 0.332 0.376 0.351 0.375 0.377 0.214 0.126 0.137
 0.239 0.207 0.243 0.266 0.27  0.276 0.279 0.276 0.282 0.287 0.297 0.319
 0.339 0.307 0.229 0.08  0.024 0.018 0.012 0.012 0.021 0.071 0.129 0.374]
216 225 raw mean 0.14862222328976593 [0.042 0.023 0.03  0.031 0.052 0.107 0.199 0.297 0.556]
 ref 0.06602539655869331 [0.229 0.08  0.024 0.018 0.012 0.012 0.021 0.071 0.129]
495 510 raw mean 0.1589773990613413 [0.295 0.233 0.147 0.033 0.025 0.248 0.28  0.044 0.07  0.14  0.118 0.136
 0.179 0.179 0.2 …[+369 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -60 /work/scratch/seqtrack.log'
```
<details><summary>output</summary>

```
  191 prev fun 0.497 iou 0.543 ang 6.4 hd 0.001
  191 idraw fun 0.523 iou 0.581 ang 38.5 hd 0.003
  191 fpraw fun 0.476 iou 0.591 ang 22.7 hd 0.002
BEST 191 fpraw 0.476 0.591 ang 22.7
  192 prev fun 0.48 iou 0.557 ang 6.1 hd 0.003
  192 vel fun 0.483 iou 0.576 ang 19.4 hd 0.002
  192 idraw fun 0.505 iou 0.549 ang 16.7 hd 0.004
  192 fpraw fun 0.575 iou 0.507 ang 31.8 hd 0.0
BEST 192 prev 0.48 0.557 ang 6.1
  193 prev fun 0.486 iou 0.546 ang 5.6 hd 0.0
  193 vel fun 0.493 iou 0.545 ang 5.9 hd 0.003
  193 idraw fun 0.664 iou 0.437 ang 39.3 hd 0.002
  193 fpraw fun 0.665 iou 0.483 ang 52.0 hd 0.01
BEST 193 prev 0.486 0.546 ang 5.6
  194 prev fun 0.569 iou 0.51 ang 22.9 hd 0.004
  194 vel fun 0.532 iou 0.504 ang 8.4 hd 0.004
  194 idraw fun 0.563 iou 0.557 ang 43.3 hd 0.01
  194 fpraw fun 0.79 …[+1522 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -70 /work/scratch/seqtrack.log'
```
<details><summary>output</summary>

```
  193 fpraw fun 0.665 iou 0.483 ang 52.0 hd 0.01
BEST 193 prev 0.486 0.546 ang 5.6
  194 prev fun 0.569 iou 0.51 ang 22.9 hd 0.004
  194 vel fun 0.532 iou 0.504 ang 8.4 hd 0.004
  194 idraw fun 0.563 iou 0.557 ang 43.3 hd 0.01
  194 fpraw fun 0.799 iou 0.415 ang 83.9 hd 0.0
BEST 194 vel 0.532 0.504 ang 8.4
  195 prev fun 0.568 iou 0.48 ang 11.3 hd 0.005
  195 vel fun 0.516 iou 0.521 ang 6.5 hd 0.008
  195 idraw fun 0.676 iou 0.465 ang 49.4 hd 0.002
  195 fpraw fun 0.794 iou 0.408 ang 76.2 hd 0.0
BEST 195 vel 0.516 0.521 ang 6.5
  196 prev fun 0.607 iou 0.431 ang 10.1 hd 0.0
  196 vel fun 0.568 iou 0.467 ang 7.9 hd 0.002
  196 idraw fun 0.599 iou 0.573 ang 55.1 hd 0.011
  196 fpraw fun 0.926 iou 0.333 ang 80.8 hd 0.46
BEST 196 vel 0.568 0.467 ang 7.9
  197 prev fun 0.604 iou 0.471 ang 21.1  …[+2389 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -40 /work/scratch/seqtrack.log'
```
<details><summary>output</summary>

```
  206 idraw fun 1.248 iou 0.107 ang 65.9 hd 0.001
  206 fpraw fun 1.248 iou 0.209 ang 119.5 hd 0.711
BEST 206 prev 0.441 0.573 ang 1.5
  207 prev fun 0.437 iou 0.571 ang 0.1 hd 0.0
  207 vel fun 0.432 iou 0.579 ang 1.1 hd 0.001
  207 idraw fun 1.228 iou 0.104 ang 62.2 hd 0.001
  207 fpraw fun 1.182 iou 0.271 ang 121.5 hd 0.688
BEST 207 vel 0.432 0.579 ang 1.1
  208 prev fun 0.434 iou 0.575 ang 1.0 hd 0.002
  208 vel fun 0.444 iou 0.57 ang 1.7 hd 0.002
  208 idraw fun 1.242 iou 0.092 ang 59.6 hd 0.004
  208 fpraw fun 1.189 iou 0.258 ang 118.7 hd 0.703
BEST 208 prev 0.434 0.575 ang 1.0
  209 prev fun 0.436 iou 0.58 ang 2.7 hd 0.002
  209 vel fun 0.437 iou 0.58 ang 2.8 hd 0.001
  209 idraw fun 1.231 iou 0.106 ang 60.8 hd 0.001
  209 fpraw fun 1.228 iou 0.216 ang 114.7 hd 0.719
BEST 209 prev 0 …[+1020 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/seq_track.py <<'PY'
exec(open('/work/scratch/opt_compare.py').read().split(\"frames=list\")[0])
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/refined_all.npz')['T'].copy();a,b=190,227
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['goodcane'];vv=np.array(g.vertices);Hp=torch.tensor(vv[vv[:,2]>.20][::10],dtype=torch.float32,device=dev)
rec=np.load('/work/inputs/reconstruction.npz');hk=torch.tensor(rec['human_keypoints_world'].astype(np.float32),device=dev);cp=torch.tensor(rec['contact_prob_right_left'].astype(np.float32),device=dev)
def hand_loss(T,fi):
 vw=Hp@T[:3,:3].T+T[:3,3];kp=torch.cat([hk[fi,21:42],hk[fi,42:63]],0);d=torch.cdist(vw,kp).min();return 0.2*(torch.clamp(d/.05,max=4)**2),d
Tprev=torch.tensor(T[a],device=dev)
for fi in range(a+1,b):
 m=objs[fi];area=m.sum();tm=cv2.resize(m.astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pers[fi].astype(np.uint8),(WW,HH),interpolation=cv2.INTER_NEAREST).astype(bool)
 # init from previous, extrapolated previous velocity, and raw smoothed identity
 inits=[('prev',Tprev.cpu().numpy())]
 if fi>=a+2:
  Tprev2=torch.tensor(T[fi-2],device=dev);Tve=Tprev+(Tprev-Tprev2);inits.append(('vel',Tve.cpu().numpy()))
 inits.append(('idraw',smooth_raw(raw,2.0)[fi]));inits.append(('fpraw',Tfp[fi]))
 best=None
 for nm,Tinit in inits:
  Ti=torch.tensor(Tinit,device=dev,dtype=torch.float32);hd0=hand_loss(Ti,fi) …[+2299 chars]
```
<details><summary>output</summary>

```
  211 prev fun 0.47 iou 0.559 ang 6.9 hd 0.001
  211 vel fun 0.447 iou 0.573 ang 3.2 hd 0.002
  211 idraw fun 1.233 iou 0.092 ang 61.3 hd 0.003
  211 fpraw fun 1.188 iou 0.249 ang 120.0 hd 0.732
BEST 211 vel 0.447 0.573 ang 3.2
  212 prev fun 0.479 iou 0.551 ang 6.6 hd 0.005
  212 vel fun 0.452 iou 0.568 ang 2.9 hd 0.0
  212 idraw fun 1.23 iou 0.101 ang 61.3 hd 0.014
  212 fpraw fun 1.192 iou 0.251 ang 126.3 hd 0.74
BEST 212 vel 0.452 0.568 ang 2.9
  213 prev fun 0.508 iou 0.531 ang 9.2 hd 0.007
  213 vel fun 0.456 iou 0.568 ang 3.9 hd 0.005
  213 idraw fun 1.111 iou 0.168 ang 58.2 hd 0.008
  213 fpraw fun 1.171 iou 0.261 ang 125.1 hd 0.729
BEST 213 vel 0.456 0.568 ang 3.9
  214 prev fun 0.536 iou 0.507 ang 11.0 hd 0.004
  214 vel fun 0.468 iou 0.556 ang 2.6 hd 0.001
  214 idraw fun 1.185  …[+2833 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -80 /work/scratch/seqint.log'
```
<details><summary>output</summary>

```

INTERVAL 236 260
BEST 237 prev 0.718 0.323 ang 5.2
BEST 238 prev 0.687 0.358 ang 2.6
BEST 239 vel 0.722 0.322 ang 5.7
BEST 240 prev 0.81 0.257 ang 5.8
BEST 241 vel 0.933 0.152 ang 3.0
BEST 242 prev 0.93 0.144 ang 1.2
BEST 243 prev 0.943 0.15 ang 3.1
BEST 244 prev 0.969 0.123 ang 3.0
BEST 245 prev 0.984 0.108 ang 2.6
BEST 246 vel 0.987 0.1 ang 0.8
BEST 247 vel 0.987 0.1 ang 1.0
BEST 248 prev 0.98 0.1 ang 1.1
BEST 249 vel 0.97 0.106 ang 1.4

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -30 /work/scratch/seqint.log'
```
<details><summary>output</summary>

```

INTERVAL 236 260
BEST 237 prev 0.718 0.323 ang 5.2
BEST 238 prev 0.687 0.358 ang 2.6
BEST 239 vel 0.722 0.322 ang 5.7
BEST 240 prev 0.81 0.257 ang 5.8
BEST 241 vel 0.933 0.152 ang 3.0
BEST 242 prev 0.93 0.144 ang 1.2
BEST 243 prev 0.943 0.15 ang 3.1
BEST 244 prev 0.969 0.123 ang 3.0
BEST 245 prev 0.984 0.108 ang 2.6
BEST 246 vel 0.987 0.1 ang 0.8
BEST 247 vel 0.987 0.1 ang 1.0
BEST 248 prev 0.98 0.1 ang 1.1
BEST 249 vel 0.97 0.106 ang 1.4
BEST 250 prev 0.964 0.114 ang 5.8
BEST 251 prev 0.935 0.139 ang 1.7
BEST 252 vel 0.894 0.175 ang 3.1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/seqint.log'
```
<details><summary>output</summary>

```
BEST 237 prev 0.718 0.323 ang 5.2
BEST 238 prev 0.687 0.358 ang 2.6
BEST 239 vel 0.722 0.322 ang 5.7
BEST 240 prev 0.81 0.257 ang 5.8
BEST 241 vel 0.933 0.152 ang 3.0
BEST 242 prev 0.93 0.144 ang 1.2
BEST 243 prev 0.943 0.15 ang 3.1
BEST 244 prev 0.969 0.123 ang 3.0
BEST 245 prev 0.984 0.108 ang 2.6
BEST 246 vel 0.987 0.1 ang 0.8
BEST 247 vel 0.987 0.1 ang 1.0
BEST 248 prev 0.98 0.1 ang 1.1
BEST 249 vel 0.97 0.106 ang 1.4
BEST 250 prev 0.964 0.114 ang 5.8
BEST 251 prev 0.935 0.139 ang 1.7
BEST 252 vel 0.894 0.175 ang 3.1
BEST 253 prev 0.802 0.246 ang 3.6
BEST 254 vel 0.783 0.271 ang 3.9
BEST 255 vel 0.618 0.422 ang 3.6
BEST 256 vel 0.598 0.447 ang 2.4

```
</details>
