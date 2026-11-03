# T1_ds_ep029

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T18:57:56Z

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
/usr/bin/bash -c "sed -n '1,240p' /work/AGENT_TASK.md"
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
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort && echo '---' && cat /work/inputs/task.txt && echo '---' && cat /work/inputs/episode.json && echo '---' && cat /work/inputs/camera.json && echo '---' && cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
camera.json 823 bytes
episode.json 70 bytes
frame0_reference_human.npz 192246 bytes
ground_plane.json 397 bytes
masks.h5 4619661 bytes
object_mesh.glb 25533580 bytes
reconstruction.npz 23675008 bytes
task.txt 160 bytes
video.mp4 12988779 bytes
---
Walk over to the rolling luggage within the bounding area and using your left foot push the rolling luggage. within the bounding area. Step away when complete.
---
{
 "episode_index": 29,
 "object": "rolling_luggage",
 "frames": 467
}---
{
 "K": [
  [
   820.0006103515625,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   820.0006103515625,
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
   816.666015625,
   0.0,
   768.0
  ],
  [
   0.0,
   816.666015625,
   576.0
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "T_world_cam": [
  [ …[+910 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f | head -100 && echo '---dirs---' && find /pipeline -maxdepth 2 -type d | sort | head -100"
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
/pipeline/export/episode_000029/edex
/pipeline/export/episode_000029/wild_export.json
/pipeline/foundationpose/episode_000029.pkl
/pipeline/inference/.materialized_inputs.h5.lock
/pipeline/inference/coconet.pth
/pipeline/inference/materialized_inputs.h5
/pipeline/inference/refined.pth
/pipeline/inference/epis …[+618 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; python - <<'PY'
import torch, cv2, trimesh, h5py, numpy as np
print('torch',torch.__version__, 'cuda',torch.cuda.is_available())
print('cv2',cv2.__version__,'trimesh',trimesh.__version__)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 cuda True
cv2 4.11.0 trimesh 5.1.1

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/outputs /work/scratch && python - <<'PY'
import numpy as np, h5py, trimesh, json, os
from scipy.spatial.transform import Rotation as R
p='/work/inputs/reconstruction.npz'; z=np.load(p)
print(z.files)
for k in z.files:
 a=z[k]; print(k,a.shape,a.dtype, 'finite',np.isfinite(a).all() if a.dtype"'!=object else '"'-')
T=z['T_world_mesh']; print('T0',T[0]); print('Tlast',T[-1])
# translations and rot angle consecutive
tr=T[:,:3,3]
rot=R.from_matrix(T[:,:3,:3]).as_rotvec()
da=np.linalg.norm(np.diff(rot,axis=0),axis=1)
dt=np.linalg.norm(np.diff(tr,axis=0),axis=1)
print('tr ranges',tr.min(0),tr.max(0),'path',np.sum(dt))
for ix in [np.argmax(da),np.argmax(dt),np.argmax(np.linalg.norm(tr-tr.mean(0),axis=1))]:
 print('extreme',ix,'tr',tr[ix],'consec rot',da[min(ix,len(da)-1)],'dt',dt[min(ix,len(dt)-1)], 'contact',z['contact_prob_right_left'][ix])
print('rot step percentiles',np.percentile(da,[0,25,50,75,90,95,99,100]))
print('trans step percentiles',np.percentile(dt,[0,25,50,75,90,95,99,100]))
print('contact max/ranges',z['contact_prob_right_left'].min(0),z['contact_prob_right_left'].max(0), np.mean(z['contact_prob_right_left'],0))
# mesh
scene=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=scene.to_geometry(); print(type(g),len(g),g.bounds,g.extents,g.centroid,len(g.vertices),len(g.faces)); print('visual',type(g.visual), hasattr(g.visual,'material'))
# masks shapes and coverage
with h5py.File('/work/inputs/masks.h5','r') as f: …[+217 chars]
```
<details><summary>output</summary>

```
['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (467, 4, 4) float32 finite True
T_cam_mesh_foundationpose (467, 4, 4) float32 finite True
T_cam_mesh_coconet (467, 4, 4) float32 finite True
human_vertices_world (467, 4610, 3) float32 finite True
human_keypoints_world (467, 70, 3) float32 finite True
contact_prob_right_left (467, 2) float32 finite True
T0 [[ 0.00811566  0.9997171   0.02235055 -0.04836467]
 [ 0.02712028  0.022123   -0.9993874   0.9280976 ]
 [-0.9995992   0.00871686 -0.02693311  3.2313282 ]
 [ 0.          0.          0.          1.        ]]
Tlast [[-0.49392414  0.8690215   0.02890463  0.01710267]
 [ 0.00713505  0.03729218 -0.99927866  0.93175274]
 [-0.8694764  -0.49336 …[+1045 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, numpy as np, os
cap=cv2.VideoCapture('/work/inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
idxs=np.linspace(0,466,24).astype(int)
ims=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,int(i)); ok,im=cap.read();
 im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2); ims.append(im)
out=np.vstack([np.hstack(ims[i:i+6]) for i in range(0,24,6)])
cv2.imwrite('/work/scratch/contact.jpg',out)
print(idxs)
PY"
```
<details><summary>output</summary>

```
467.0 30.0 1536.0 1152.0
[  0  20  40  60  81 101 121 141 162 182 202 222 243 263 283 303 324 344
 364 384 405 425 445 466]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np, h5py
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(type(s),s.geometry.keys())
for k,g in s.geometry.items(): print(k,type(g),g.bounds,g.extents,g.centroid,len(g.vertices),len(g.faces),type(g.visual))
g=s.to_geometry(); print('to_geometry',type(g), g.bounds,g.extents,len(g.vertices),len(g.faces),type(g.visual),g.visual.material)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for k in f.keys():
  d=f[k]; counts=np.array([d[i].sum() for i in range(d.shape[0])]); print(k,d.shape,'min max med mean',counts.min(),counts.max(),np.median(counts),counts.mean()); print('counts 0:467:10',counts[::10])
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.scene.scene.Scene'> odict_keys(['rollingluggage'])
rollingluggage <class 'trimesh.base.Trimesh'> [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.66103488]] [0.38377577 0.25683792 0.97835469] [ 4.67789225e-11 -3.14784601e-10  1.47522536e-09] 292750 97596 <class 'trimesh.visual.texture.TextureVisuals'>
to_geometry <class 'trimesh.base.Trimesh'> [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.66103488]] [0.38377577 0.25683792 0.97835469] 292750 97596 <class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7664747a9ad0>
object (467, 1152, 1536) min max med mean 10743 23481 17281.0 17592.7130620985
counts 0:467:10 [11062 11229 11192 11155 11024 11025 10989 11044 10972 11147 11043 11210
 13453 15942  …[+585 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr, torch
print('nvdiffrast ok',torch.cuda.is_available())
PY"
```
<details><summary>output</summary>

```
nvdiffrast ok True

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import torch,nvdiffrast.torch as dr, numpy as np, trimesh, cv2, json,h5py
from scipy.spatial.transform import Rotation as R
# mesh
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'],device='cuda',dtype=torch.float32)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Twc_inv=torch.tensor(np.linalg.inv(Twc),device='cuda',dtype=torch.float32)
ctx=dr.RasterizeCudaContext()
def rast(idx, flip):
 Vw=V[None].expand(len(idx),-1,-1); Tw=T[idx]; Rm=Tw[:,:3,:3]; tt=Tw[:,:3,3]
 Vc=torch.einsum('fij,fvj->fvi',Rm,Vw)+tt[:,None]
 Tc=torch.einsum('ij,fvj->fvi',Twc_inv[:3,:3],Vc)+Twc_inv[:3,3]
 x,y,dep=Vc[...,0],Vc[...,1],Vc[...,2]
 fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
 clip=torch.stack([fx*x+cx*dep, (1 if flip else -1)*(fy*y+cy*dep), dep, dep],-1)
 r,_=dr.rasterize(ctx,clip,F,resolution=(1152,1536))
 mask=(r[...,3]>0).cpu().numpy()
 return mask
with h5py.File('/work/inputs/masks.h5','r') as h:
 gt=h['object'][:10]
for flip in [False,True]:
 m=rast(np.arange(10),flip)
 ious=[]
 for a,b in zip(m,gt): ious.append((a&b).sum()/max(1,(a|b).sum()))
 print('flip',flip,ious,[x.sum() for x in m])
# save overlays variant expected flip? 
idx=np.ara …[+647 chars]
```
<details><summary>output</summary>

```
flip False [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
flip True [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
False all 0.0 [0. 0. 0. 0. 0. 0. 0.]
True all 0.0 [0. 0. 0. 0. 0. 0. 0.]
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 36, in <module>
    ov=im.copy(); ov[m]=(0.5*ov[m]+0.5*np.array([255,0,0])).astype(np.uint8); ov[gt[i]]=(0.5*ov[gt[i]]+0.5*np.array([0,255,0])).astype(np.uint8)
                                                                                                ~~^^^
IndexError: index 50 is out of bounds for axis 0 with size 10

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/dep, dep\\],-1)/torch.zeros_like(dep), dep],-1)/\" /work/scratch/render_test.py
sed -i \"s/gt=h\\['object'\\]\\[:10\\]/gt=h['object'][:]/\" /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
flip False [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
flip True [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
False all 0.0 [0. 0. 0. 0. 0. 0. 0.]
True all 0.0 [0. 0. 0. 0. 0. 0. 0.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split('ctx=')[0])
import torch
idx=torch.tensor([0],device='cuda'); Vw=V[None]; Tw=T[idx]; Vc=torch.einsum('fij,fvj->fvi',Tw[:,:3,:3],Vw)+Tw[:,:3,3][:,None]; C=torch.einsum('ij,fvj->fvi',Twc_inv[:3,:3],Vc)+Twc_inv[:3,3]
print('V bounds',V.min(0).values,V.max(0).values)
print('C bounds',C[0].min(0).values,C[0].max(0).values)
fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]; x,y,d=C[...,0],C[...,1],C[...,2]
clip=torch.stack([fx*x+cx*d, -(fy*y+cy*d), torch.zeros_like(d), d],-1)
print('clip bounds',clip[0].min(0).values,clip[0].max(0).values)
print(clip[0,:5])
PY"
```
<details><summary>output</summary>

```
V bounds tensor([-0.1935, -0.1205, -0.3173], device='cuda:0') tensor([0.1902, 0.1363, 0.6610], device='cuda:0')
C bounds tensor([-0.3151,  0.1603,  2.9982], device='cuda:0') tensor([-0.0542,  1.1537,  3.3990], device='cuda:0')
clip bounds tensor([ 2055.9871, -2822.5088,     0.0000,     2.9982], device='cuda:0') tensor([ 2505.0710, -1868.9862,     0.0000,     3.3990], device='cuda:0')
tensor([[ 2357.5127, -2561.4094,     0.0000,     3.3708],
        [ 2357.5127, -2561.4094,     0.0000,     3.3708],
        [ 2357.5127, -2561.4094,     0.0000,     3.3708],
        [ 2357.5127, -2561.4094,     0.0000,     3.3708],
        [ 2357.5127, -2561.4094,     0.0000,     3.3708]], device='cuda:0')

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py'
s=open(p).read()
s=s.replace(\"clip=torch.stack([fx*x+cx*dep, (1 if flip else -1)*(fy*y+cy*dep), torch.zeros_like(dep), dep],-1)\",\"sgn=(1 if flip else -1)\\n clip=torch.stack([(2*fx/1536)*x+(2*cx/1536-1)*dep, sgn*((2*fy/1152)*y+(2*cy/1152-1)*dep), torch.zeros_like(dep), dep],-1)\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
flip False [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] [9584, 9983, 9793, 9921, 10026, 10474, 10554, 10366, 10335, 10434]
flip True [0.2225840833728091, 0.23279897027849286, 0.22937872937872938, 0.24669811320754717, 0.2357117914096147, 0.24635852735405436, 0.246395881006865, 0.24688221709006927, 0.24809248554913294, 0.2541145108947612] [9585, 9983, 9793, 9921, 10026, 10474, 10553, 10366, 10336, 10434]
False all 0.0 [0. 0. 0. 0. 0. 0. 0.]
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 30, in <module>
    m=rast(idx,flip); ious=np.array([(a&b).sum()/max(1,(a|b).sum()) for a,b in zip(m,gt)])
      ^^^^^^^^^^^^^^
  File "/work/scratch/render_test.py", line 17, in rast
    r,_=dr.rasterize(ctx,clip,F,resolution=(1152,1536))
        ^^^^^^^^^^^^^^^^^^^^^^^^ …[+1387 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py'; s=open(p).read(); s=s.replace('for flip in [False,True]:\\n m=rast(idx,flip); ious=np.array([(a&b).sum()/max(1,(a|b).sum()) for a,b in zip(m,gt)])\\n print(flip,\\'all\\',ious.mean(),np.percentile(ious,[0,10,25,50,75,90,100]))\\n','')
s=s.replace('rast([i],False)[0]','rast([i],True)[0]')
open('/work/scratch/render_overlay.py','w').write(s)
PY
python /work/scratch/render_overlay.py && python - <<'PY'
import cv2,numpy as np
ims=[cv2.imread(f'/work/scratch/ov_{i:03d}.jpg') for i in [0,50,100,150,200,250,300,350,400,466]]
ims=[cv2.resize(x,(384,288)) for x in ims]
out=np.vstack([np.hstack(ims[i:i+5]) for i in range(0,10,5)])
cv2.imwrite('/work/scratch/overlays.jpg',out)
PY"
```
<details><summary>output</summary>

```
flip False [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] [9584, 9983, 9793, 9921, 10026, 10474, 10554, 10366, 10335, 10434]
flip True [0.2225840833728091, 0.23279897027849286, 0.22937872937872938, 0.24669811320754717, 0.2357117914096147, 0.24635852735405436, 0.246395881006865, 0.24688221709006927, 0.24809248554913294, 0.2541145108947612] [9585, 9983, 9793, 9921, 10026, 10474, 10553, 10366, 10336, 10434]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_poses.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,json,h5py,cv2
from scipy.spatial.transform import Rotation as R
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz'); cam=json.load(open('/work/inputs/camera.json'))
Twc=np.array(cam['T_world_cam']); K=np.array(cam['K']); Kp=np.array(cam['K_pipeline'])
Twc_inv=np.linalg.inv(Twc)
ctx=dr.RasterizeCudaContext()
def render(Ts, K, world=False):
 Ts=torch.tensor(Ts,dtype=torch.float32,device='cuda')
 if world:
  Tcinv=torch.tensor(Twc_inv,dtype=torch.float32,device='cuda')
 else:
  Tcinv=torch.eye(4,dtype=torch.float32,device='cuda')
 out=[]
 for j in range(0,len(Ts),8):
  Tw=Ts[j:j+8]; Vw=V[None].expand(len(Tw),-1,-1)
  Vw=torch.einsum('fij,fvj->fvi',Tw[:,:3,:3],Vw)+Tw[:,:3,3][:,None]
  Vc=torch.einsum('ij,fvj->fvi',Tcinv[:3,:3],Vw)+Tcinv[:3,3]
  x,y,d=Vc[...,0],Vc[...,1],Vc[...,2]
  fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]; W,H=1536,1152
  clip=torch.stack([(2*fx/W)*x+(2*cx/W-1)*d,-((2*fy/H)*y+(2*cy/H-1)*d),torch.zeros_like(d),d],-1)
  r,_=dr.rasterize(ctx,clip,F,resolution=(H,W)); out.append((r[...,3]>0).cpu().numpy())
 return np.concatenate(out)
with h5py.File('/work/inputs/masks.h5','r') as h: gt=h['object'][:]
# baseline camera
Tb=Twc_inv[None]@z['T_world_mesh …[+1065 chars]
```
<details><summary>output</summary>

```
baseline iou mean/perc 0.0 [0. 0. 0. 0. 0. 0. 0.] cov 0.0 [0. 0. 0. 0. 0. 0. 0.]
foundationpose iou mean/perc 0.0 [0. 0. 0. 0. 0. 0. 0.] cov 0.0 [0. 0. 0. 0. 0. 0. 0.]
coconet iou mean/perc 0.0 [0. 0. 0. 0. 0. 0. 0.] cov 0.0 [0. 0. 0. 0. 0. 0. 0.]
fp rel trans 0.11163787927215232 [0.04366249 0.08492545 0.25143677 0.29767684] rel rot 4.9256305377380265 [1.22826708 5.18518979 7.50986253 9.4468817 ]
coco rel trans 0.09095915146521463 [0.05671161 0.07767936 0.12958891 0.13807357] rel rot 1.1688825026427412 [1.12747605 1.17617288 1.19271438 1.19402426]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_poses.py').read().split(\"with h5py.File\")[0])
Tb=Twc_inv[None]@z['T_world_mesh']
for n in [1,8]:
 m=render(Tb[:n],K,False); print(n,m.shape,m.sum(axis=(1,2)),m.dtype)
PY"
```
<details><summary>output</summary>

```
1 (1, 1152, 1536) [9816] bool
8 (8, 1152, 1536) [ 9816 10053  9897  9955 10118 10524 10586 10407] bool

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as h:
 gt=h['object'][:]; print(gt.shape,gt.dtype,gt.sum((1,2))[:5],np.unique(gt[0]))
ms=np.load('/work/scratch/baseline_sil.npy'); print(ms.shape,ms.sum((1,2))[:5],np.unique(ms[0]))
print((ms[0]&gt[0]).sum(),(ms[0]|gt[0]).sum())
PY"
```
<details><summary>output</summary>

```
(467, 1152, 1536) bool [11062 11088 11222 11223 11206] [False  True]
(467, 1152, 1536) [ 9816 10053  9897  9955 10118] [False  True]
0 20878

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as h: gt=h['object'][:]
ms=np.load('/work/scratch/baseline_sil.npy')
for i in [0,50,100,150,200,250,300,350,400,466]:
 def bbox(a):
  y,x=np.where(a); return x.min(),y.min(),x.max()+1,y.max()+1
 print(i,'gt',bbox(gt[i]),'ren',bbox(ms[i]))
PY"
```
<details><summary>output</summary>

```
0 gt (675, 700, 746, 872) ren (675, 282, 744, 551)
50 gt (675, 699, 746, 873) ren (676, 281, 744, 550)
100 gt (675, 700, 746, 873) ren (674, 281, 743, 551)
150 gt (618, 712, 725, 895) ren (614, 256, 720, 546)
200 gt (619, 713, 724, 894) ren (614, 254, 720, 550)
250 gt (591, 740, 709, 944) ren (586, 204, 705, 533)
300 gt (591, 740, 709, 945) ren (588, 210, 707, 534)
350 gt (597, 742, 725, 951) ren (599, 203, 723, 532)
400 gt (655, 754, 786, 982) ren (655, 174, 783, 527)
466 gt (655, 754, 786, 982) ren (658, 175, 783, 528)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/,-((2\\*fy\\/H)\\*y+(2\\*cy\\/H-1)\\*d),/,(2*fy\\/H)*y+(2*cy\\/H-1)*d,/' /work/scratch/eval_poses.py
python /work/scratch/eval_poses.py"
```
<details><summary>output</summary>

```
baseline iou mean/perc 0.8203056213440582 [0.78499455 0.79763214 0.81098277 0.82135843 0.83134411 0.83812927
 0.85175993] cov 0.8858868126135293 [0.83357117 0.8580788  0.86856272 0.88518146 0.90331944 0.91698397
 0.93584213]
foundationpose iou mean/perc 0.7765642989333232 [0.7007289  0.73290874 0.76460171 0.7837557  0.79616926 0.80456313
 0.82370472] cov 0.8299371009809302 [0.74400839 0.78113856 0.81284218 0.83572206 0.85328584 0.86610459
 0.89409247]
coconet iou mean/perc 0.8325073739676473 [0.80678324 0.81672108 0.82366671 0.83068226 0.83995055 0.85070466
 0.87004781] cov 0.9025877240842689 [0.86180166 0.8855192  0.89297658 0.90372306 0.91188468 0.91857379
 0.93560623]
fp rel trans 0.11163787927215232 [0.04366249 0.08492545 0.25143677 0.29767684] rel rot 4.9256305377380265 [1.22826708 5. …[+186 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; p=T[:,:3,3]; rv=R.from_matrix(T[:,:3,:3]).as_rotvec(); rv=np.unwrap(rv,axis=0); da=np.linalg.norm(np.diff(rv,axis=0),axis=1); dt=np.linalg.norm(np.diff(p,axis=0),axis=1)
with h5py.File('/work/inputs/masks.h5','r') as h: m=h['object'][:]; person=h['person'][:]
vals=[]
for i,a in enumerate(m):
 y,x=np.where(a); vals.append((x.mean(),y.mean(),a.sum(),person[i].sum()))
vals=np.array(vals)
# baseline diff 5 frame velocity
for i in range(0,467,10):
 print(f'{i:3d} p {p[i,0]:+.3f} {p[i,1]:+.3f} {p[i,2]:+.3f} rot {np.degrees(rv[i]).round(1)} dt {dt[min(i,465)]:.4f} da {np.degrees(da[min(i,465)]):.2f} mask {vals[i,0]:.0f},{vals[i,1]:.0f},A{vals[i,2]:.0f}')
PY"
```
<details><summary>output</summary>

```
  0 p -0.048 +0.928 +3.231 rot [ 69.7  70.7 -67.2] dt 0.0214 da 3.06 mask 710,782,A11062
 10 p -0.053 +0.916 +3.190 rot [ 71.2  69.1 -63.7] dt 0.0028 da 1.25 mask 710,782,A11229
 20 p -0.059 +0.927 +3.217 rot [ 72.3  68.8 -62.8] dt 0.0041 da 0.85 mask 710,782,A11192
 30 p -0.054 +0.923 +3.213 rot [ 72.   70.9 -64.6] dt 0.0141 da 1.29 mask 710,782,A11155
 40 p -0.064 +0.932 +3.239 rot [ 73.2  69.2 -62.9] dt 0.0031 da 1.36 mask 709,783,A11024
 50 p -0.056 +0.925 +3.224 rot [ 73.2  68.3 -62.3] dt 0.0081 da 0.66 mask 710,783,A11025
 60 p -0.059 +0.923 +3.216 rot [ 73.1  67.7 -62.4] dt 0.0020 da 0.75 mask 710,783,A10989
 70 p -0.058 +0.921 +3.210 rot [ 73.3  67.8 -62.6] dt 0.0048 da 0.55 mask 709,783,A11044
 80 p -0.061 +0.925 +3.219 rot [ 73.6  68.  -62.8] dt 0.0126 da 1.81 mask 709,783,A10972 …[+3383 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as h: m=h['object'][0]; p=h['person'][0]
im=cv2.imread('/work/scratch/ov_000.jpg')
# crop object region original video and colored masks
cap=cv2.VideoCapture('/work/inputs/video.mp4'); _,v=cap.read()
x0,y0,x1,y1=580,640,850,980
crop=v[y0:y1,x0:x1].copy()
for mask,col in [(m,np.array([0,255,0])),(p,np.array([0,0,255]))]:
 mm=mask[y0:y1,x0:x1]; crop[mm]=(0.5*crop[mm]+0.5*col).astype(np.uint8)
cv2.imwrite('/work/scratch/mask_crop.jpg',cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
# print object mask unique rows extents
for y in range(690,880,20):
 xs=np.where(m[y])[0]; print(y, (xs.min(),xs.max()) if len(xs) else None)
PY"
```
<details><summary>output</summary>

```
690 None
710 (677, 740)
730 (676, 744)
750 (676, 744)
770 (676, 744)
790 (676, 744)
810 (676, 745)
830 (676, 744)
850 (678, 743)
870 (677, 687)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep clip= -A2 /work/scratch/eval_poses.py && python - <<'PY'
import numpy as np,h5py
m=np.load('/work/scratch/baseline_sil.npy'); print(m.sum((1,2))[:5]);
for i in [0,50]:
 y,x=np.where(m[i]); print(x.min(),x.max(),y.min(),y.max())
with h5py.File('/work/inputs/masks.h5','r') as h:g=h['object'][0]
y,x=np.where(g);print('gt',x.min(),x.max(),y.min(),y.max(),(g&m[0]).sum())
PY"
```
<details><summary>output</summary>

```
[ 9817 10055  9898  9954 10118]
675 743 601 869
676 743 602 870
gt 675 745 700 871 9296

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=m.visual; print(type(v),v.uv.shape,v.uv.min(0),v.uv.max(0)); mat=v.material; print(mat,v.material.image if hasattr(v.material,'image') else None)
if mat.image is not None: print('image',mat.image.size,mat.image.mode)
print('vertex colors',v.vertex_colors.shape,v.vertex_colors[:3])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'
<class 'trimesh.visual.texture.TextureVisuals'> (292750, 2) [0.000699   0.00079399] [0.99930102 0.99920601]
<trimesh.visual.material.PBRMaterial object at 0x79ee77713110> None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mat=m.visual.material
print(mat)
for a in dir(mat):
 if not a.startswith('_'):
  try:
   x=getattr(mat,a)
   if not callable(x): print(a,type(x), getattr(x,'shape',None))
  except:pass
PY"
```
<details><summary>output</summary>

```
<trimesh.visual.material.PBRMaterial object at 0x763d80106750>
alphaCutoff <class 'NoneType'> None
alphaMode <class 'NoneType'> None
baseColorFactor <class 'NoneType'> None
baseColorTexture <class 'PIL.Image.Image'> None
doubleSided <class 'bool'> None
emissiveFactor <class 'NoneType'> None
emissiveTexture <class 'NoneType'> None
main_color <class 'numpy.ndarray'> (4,)
metallicFactor <class 'float'> None
metallicRoughnessTexture <class 'NoneType'> None
name <class 'str'> None
normalTexture <class 'NoneType'> None
occlusionTexture <class 'NoneType'> None
roughnessFactor <class 'NoneType'> None

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_rgb.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,cv2,json,h5py
from PIL import Image
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')
tex=np.array(mesh.visual.material.baseColorTexture.convert('RGB'))/255.; tex=torch.tensor(tex,dtype=torch.float32,device='cuda').permute(2,0,1)[None]
z=np.load('/work/inputs/reconstruction.npz'); cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam']); Tc=np.linalg.inv(Twc); K=np.array(cam['K']); W,H=1536,1152
ctx=dr.RasterizeCudaContext()
def render(Ts,K):
 Ts=torch.tensor(Ts,dtype=torch.float32,device='cuda'); out=[]; depths=[]
 Tcinv=torch.tensor(Tc,dtype=torch.float32,device='cuda')
 for j in range(0,len(Ts),8):
  Tw=Ts[j:j+8]; Vw=V[None].expand(len(Tw),-1,-1); Vw=torch.einsum('fij,fvj->fvi',Tw[:,:3,:3],Vw)+Tw[:,:3,3][:,None]
  Vc=torch.einsum('ij,fvj->fvi',Tcinv[:3,:3],Vw)+Tcinv[:3,3]
  x,y,d=Vc[...,0],Vc[...,1],Vc[...,2]; fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
  clip=torch.stack([(2*fx/W)*x+(2*cx/W-1)*d,(2*fy/H)*y+(2*cy/H-1)*d,torch.zeros_like(d),d],-1)
  r,_=dr.rasterize(ctx,clip,F,resolution=(H,W)); uv,_=dr.interpolate(UV[None].expand(len(Tw),-1,-1),r,F); uv=uv[...,:2]
  # try uv direct, gltf origin likely top
  gri …[+1109 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_rgb.py", line 27, in <module>
    x0,y0,x1,y1=550,550,880,1000; cc=im[y0:y1,x0:x1].copy(); cc[mask[y0:y1,x0:x1]]=(0.35*cc[mask[y0:y1,x0:x1]]+0.65*(rgb[0][mask[y0:y1,x0:x1]]*255)).astype(np.uint8)
                                                                                                                     ~~~~~~^^^^^^^^^^^^^^^^^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1152 but corresponding boolean dimension is 450
[ WARN:0@0.019] global loadsave.cpp:268 findDecoder imread_('/work/scratch/rgbov_000.jpg'): can't open/read file: check file path/integrity
[ WARN:0@0.020] global loadsave.cpp:268 findDecoder imread_('/work/scratch/rgbov_100.jpg'): can't open/read file: check fil …[+711 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_rgb.py';s=open(p).read();s=s.replace(\"cc=im[y0:y1,x0:x1].copy(); cc[mask[y0:y1,x0:x1]]=(0.35*cc[mask[y0:y1,x0:x1]]+0.65*(rgb[0][mask[y0:y1,x0:x1]]*255)).astype(np.uint8)\",\"cc=im[y0:y1,x0:x1].copy(); mm=mask[y0:y1,x0:x1]; cc[mm]=(0.35*cc[mm]+0.65*(rgb[0][y0:y1,x0:x1][mm]*255)).astype(np.uint8)\")
open(p,'w').write(s)
PY
python /work/scratch/render_rgb.py
python - <<'PY'
import cv2,numpy as np
ims=[cv2.imread(f'/work/scratch/rgbov_{i:03d}.jpg') for i in [0,100,200,300,400]]
cv2.imwrite('/work/scratch/rgb_overlays.jpg',np.hstack(ims))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat /pipeline/.stages/05_object_pose.json /pipeline/.stages/06_coconet.json /pipeline/.stages/07_refinement.json /pipeline/depth/moge2/depth_report.json; python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 print(f.keys())
 for k in f.keys(): print(k,f[k].shape,f[k].dtype)
PY"
```
<details><summary>output</summary>

```
<KeysViewHDF5 ['aligned', 'alignment', 'frame_names', 'raw']>
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
AttributeError: 'Group' object has no attribute 'shape'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 def vis(name,obj):
  if hasattr(obj,'shape'): print(name,obj.shape,obj.dtype)
 f.visititems(vis)
 for n in ['aligned','raw','alignment']:
  g=f[n]; print(n,'attrs',dict(g.attrs))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (467,) object
alignment/front_stereo_camera_left/scale (467,) float32
alignment/front_stereo_camera_left/shift (467,) float32
alignment/front_stereo_camera_left/valid_count (467,) int32
frame_names/front_stereo_camera_left (467,) object
raw/front_stereo_camera_left (467,) object
aligned attrs {}
raw attrs {}
alignment attrs {}

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for grp in ['aligned','raw']:
  ds=f[grp]['front_stereo_camera_left'][0]; print(grp,type(ds),ds.shape,ds.dtype,ds.attrs.keys()); a=ds[:]; print(a.min(),a.max(),np.percentile(a,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'attrs'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sha256sum /work/inputs/object_mesh.glb /pipeline/export/episode_000029/*/* 2>/dev/null | head; find /pipeline/export/episode_000029 -maxdepth 3 -type f -printf '%p %s\\n' | head -50; python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 ds=f['aligned']['front_stereo_camera_left'][0]
 print(type(ds),ds.shape,ds.dtype,ds.min(),ds.max())
PY"
```
<details><summary>output</summary>

```
cd4c9c2b4d80e47bb5228461d39bec195f5ed0395942b2ebd0d4585eac341b67  /work/inputs/object_mesh.glb
02b5bc11af7072a374fba2fb9f3be48b76d46b70de7ed97e9bd414763ed5b18f  /pipeline/export/episode_000029/human_masks/front_stereo_camera_left.h5
a0da5bfbc0745e347a28155a52d78578f4838716549c62e259294247bf071503  /pipeline/export/episode_000029/images/front_stereo_camera_left.h5
9fd8fbf18f37b3ff846159e0dd2a8fd4f9c415c5a38effc92f1123200e83ba02  /pipeline/export/episode_000029/object_masks/front_stereo_camera_left.h5
dab866ebc0c328fa3c286dd76e7bbbd8908bfa771d46451ae00a7287404a1a28  /pipeline/export/episode_000029/object_mesh/output_aligned.glb
/pipeline/export/episode_000029/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000029/human_masks/front_stereo_camera_left.h5 …[+568 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
for p in ['/work/inputs/object_mesh.glb','/pipeline/export/episode_000029/object_mesh/output_aligned.glb']:
 s=trimesh.load(p,force='scene',process=False); print(p,s.geometry.keys());
 for k,g in s.geometry.items(): print(k,g.bounds,g.extents,len(g.vertices),len(g.faces))
PY
cat /pipeline/export/episode_000029/wild_export.json"
```
<details><summary>output</summary>

```
/work/inputs/object_mesh.glb odict_keys(['rollingluggage'])
rollingluggage [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.66103488]] [0.38377577 0.25683792 0.97835469] 292750 97596
/pipeline/export/episode_000029/object_mesh/output_aligned.glb odict_keys(['rollingluggage'])
rollingluggage [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.66103488]] [0.38377577 0.25683792 0.97835469] 292750 97596
{
  "aligned_object_extents_m": [
    0.2547444701194763,
    0.3841489851474762,
    0.9780486822128296
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "depth_source_commit …[+2434 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
for p in ['/pipeline/export/episode_000029/object_masks/front_stereo_camera_left.h5','/pipeline/work/masks_k0.h5']:
 print('\\n',p)
 with h5py.File(p,'r') as f:
  print(f.keys())
  def vis(n,o):
   if hasattr(o,'shape'): print(n,o.shape,o.dtype)
  f.visititems(vis)
PY"
```
<details><summary>output</summary>

```

 /pipeline/export/episode_000029/object_masks/front_stereo_camera_left.h5
<KeysViewHDF5 ['frames']>
frames (467, 1152, 1536) uint8

 /pipeline/work/masks_k0.h5
<KeysViewHDF5 ['episode_000029']>
episode_000029/000000-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000029/000000-k0.person_mask.png (1152, 1536) uint8
episode_000029/000001-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000029/000001-k0.person_mask.png (1152, 1536) uint8
episode_000029/000002-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000029/000002-k0.person_mask.png (1152, 1536) uint8
episode_000029/000003-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000029/000003-k0.person_mask.png (1152, 1536) uint8
episode_000029/000004-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000029/000004-k0.person_mask.png (1152, 1536) u …[+56369 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/export/episode_000029/object_masks/front_stereo_camera_left.h5','r') as f: a=f['frames'][:]
with h5py.File('/work/inputs/masks.h5','r') as f: b=f['object'][:]
print(a.shape,a.dtype,b.shape,a.max(),b.sum((1,2))[:5],a.sum((1,2))[:5])
# iou sample
for i in [0,100,200,300,400]:
 m=a[i]>127; print(i,(m&b[i]).sum()/(m|b[i]).sum(),m.sum(),b[i].sum())
PY"
```
<details><summary>output</summary>

```
(467, 1152, 1536) uint8 (467, 1152, 1536) 255 [11062 11088 11222 11223 11206] [2820810 2827440 2861610 2861865 2857530]
0 1.0 11062 11062
100 1.0 11043 11043
200 1.0 16108 16108
300 1.0 20213 20213
400 1.0 23359 23359

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py
raw=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r'); print(raw.shape,raw.dtype)
z=np.load('/work/inputs/reconstruction.npz'); cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tci=np.linalg.inv(Twc)
for i in [0,50,100,150,200,250,300,350,400,450]:
 T=Tci@z['T_world_mesh'][i]; c=T[:3,3]; u=K[0,0]*c[0]/c[2]+K[0,2]; v=K[1,1]*c[1]/c[2]+K[1,2]; d=raw[i,int(v),int(u)]/1000
 print(i,'zc',c[2],'uv',u,v,'raw',d)
PY"
```
<details><summary>output</summary>

```
(467, 1152, 1536) uint16
0 zc 3.198561729893451 uv 709.7489749060056 772.3939586303056 raw 2.857
50 zc 3.1908620494613857 uv 707.6713358372501 772.2392543321199 raw 2.802
100 zc 3.187599052811454 uv 707.4188246665962 771.7914671148825 raw 2.761
150 zc 3.0044454013458033 uv 669.2582611704443 789.4402071866233 raw 2.495
200 zc 2.936564780867415 uv 668.7847683835222 789.3469619206378 raw 2.496
250 zc 2.6643945208945747 uv 647.2515012958623 825.5859601519406 raw 2.275
300 zc 2.7028910405833604 uv 649.83181062568 821.5312293082648 raw 2.226
350 zc 2.6713250992027815 uv 664.6264690235087 826.0569502695457 raw 2.245
400 zc 2.5086192541532952 uv 723.8383883625079 845.368283073442 raw 2.09
450 zc 2.525689948961803 uv 724.692398861756 844.2907690695649 raw 2.086

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 s=f['alignment']['front_stereo_camera_left']['scale'][:]; sh=f['alignment']['front_stereo_camera_left']['shift'][:]; vc=f['alignment']['front_stereo_camera_left']['valid_count'][:]
 print('scale',np.percentile(s,[0,10,50,90,100])[:], 'shift',np.percentile(sh,[0,10,50,90,100]), 'valid',np.percentile(vc,[0,10,50,90,100]))
 print('first',s[:10],sh[:10])
 # decode first aligned object dtype?
 d=f['aligned']['front_stereo_camera_left'][0]; print(type(d),d.shape,d.dtype,d[:20])
PY"
```
<details><summary>output</summary>

```
scale [1.06537688 1.09486606 1.13314176 1.17623818 1.20667672] shift [0. 0. 0. 0. 0.] valid [20322.  26162.  29502.  45555.4 48458. ]
first [1.0885618 1.0830917 1.0837054 1.0894419 1.0653769 1.0684155 1.0665994
 1.0777967 1.0778762 1.0898507] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
<class 'numpy.ndarray'> (640126,) uint8 [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,io
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 d=f['aligned']['front_stereo_camera_left'][0]; a=cv2.imdecode(d,cv2.IMREAD_UNCHANGED); print(a.shape,a.dtype,a.min(),a.max()); s=f['alignment']['front_stereo_camera_left']['scale'][0]
 r=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r')[0]/1000
 print('s',s,'a at center',a[772,710],'raw',r[772,710],a.dtype)
 print('ratio percentiles object mask', np.percentile(a[(a>0)]/r[(a>0)], [0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
(1152, 1536) uint16 1755 9898
s 1.0885618 a at center 3107 raw 2.855 uint16
ratio percentiles object mask [1087.96579108 1088.24940048 1088.43296781 1088.5361552  1088.56180616]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
from scipy.spatial.transform import Rotation as R
with h5py.File('/work/inputs/masks.h5','r') as h:m=h['object'][:]
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh']; p=T[:,:3,3];rv=R.from_matrix(T[:,:3,:3]).as_rotvec();rv=np.unwrap(rv,axis=0)
dio=[];dc=[]
for i in range(466):
 a,b=m[i],m[i+1]; dio.append((a"'^b).sum()/max(1,(a|b).sum())); dc.append(np.linalg.norm(p[i+1]-p[i]))
dio=np.array(dio);dc=np.array(dc)
for th in [.001,.002,.003,.005,.01,.02,.05]:
 seg=[];st=0
 for i in range(466):
  moving=dio[i]>th or dc[i]>.008
  if moving and st<0: st=i
  if not moving and st>=0: seg.append((st,i));st=-1
 if st>=0:seg.append((st,466))
 print('"'th',th,'motion ranges',seg)
# print around large changes every 5
print('top mask changes')
for i in np.argsort(dio)[-50:][::-1][:30]: print(i,round(dio[i],4),round(dc[i],4))
print('per 5 around')
for i in range(0,466,5):
 if dio[i]>.002 or dc[i]>.006: print(i,round(dio[i],4),round(dc[i],4),'p',np.round(p[i],3))
PY"
```
<details><summary>output</summary>

```
th 0.001 motion ranges [(0, 466)]
th 0.002 motion ranges [(0, 216), (217, 405), (406, 425), (426, 444), (445, 464)]
th 0.003 motion ranges [(0, 20), (21, 149), (150, 157), (158, 178), (179, 180), (181, 208), (209, 211), (212, 216), (217, 254), (255, 259), (262, 264), (265, 334), (335, 381), (382, 394), (396, 398), (399, 403), (404, 405), (406, 419), (421, 425), (426, 429), (430, 438), (440, 444), (445, 446), (447, 451), (453, 454), (457, 463)]
th 0.005 motion ranges [(0, 9), (13, 20), (22, 32), (33, 51), (52, 66), (67, 69), (70, 94), (95, 143), (146, 147), (148, 149), (150, 153), (154, 157), (160, 161), (164, 173), (174, 176), (177, 178), (179, 180), (181, 190), (191, 208), (212, 213), (215, 216), (217, 253), (256, 257), (258, 259), (262, 263), (266, 267), (270, 280), (281, 282), (283, 294 …[+7782 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.array(m.vertices)
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh']
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n,d=pl[:3],pl[3]
for i in [0,20,50,80,100,110,120,130,140,150,180,200,220,240,260,280,300,320,340,350,360,370,380,400,450,466]:
 P=np.einsum('ij,vj->vi',T[i,:3,:3],V)+T[i,:3,3]
 s=P@n+d
 mn=s.min(); k=np.argmin(s); # lowest vertex
 print(i,'minplane',mn,'5pct',np.percentile(s,0.1),'miny',P[:,1].min(),'center',T[i,:3,3], 'lowpt',P[k])
PY"
```
<details><summary>output</summary>

```
0 minplane -0.03810631725405922 5pct -0.03414252240207594 miny 0.26353213572115963 center [-0.04836467  0.9280976   3.2313282 ] lowpt [0.03439928 1.23467771 3.40571247]
20 minplane -0.03215962739114597 5pct -0.028782416350193393 miny 0.26049386472550085 center [-0.05899233  0.9269453   3.2170873 ] lowpt [0.006859   1.22926883 3.40554213]
50 minplane -0.030594399146259166 5pct -0.027469082794128186 miny 0.2589767263735062 center [-0.05622604  0.9251133   3.2240124 ] lowpt [0.0391706  1.24458978 3.09378902]
80 minplane -0.031302504070292336 5pct -0.027072155170873247 miny 0.26018506888611403 center [-0.06058463  0.92544734  3.2187364 ] lowpt [-0.15049245  1.25067624  3.06413086]
100 minplane -0.029645546617224072 5pct -0.02398974978538715 miny 0.25726458600226276 center [-0.05710417  0.92236 …[+3730 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); im=m.visual.material.baseColorTexture; print(im.size,im.mode); im.save('/work/scratch/texture.png')
PY"
```
<details><summary>output</summary>

```
(5007, 4407) RGB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_rgb.py';s=open(p).read()
# create variant flip uv y
s=s.replace(\"grid=uv*2-1\",\"grid=uv*2-1; grid=torch.stack([grid[...,0],-grid[...,1]],-1)\")
s=s.replace('rgbov_','rgbovflip_').replace('render_','renderflip_')
open('/work/scratch/render_rgb_flip.py','w').write(s)
PY
python /work/scratch/render_rgb_flip.py
python - <<'PY'
import cv2,numpy as np
a=cv2.imread('/work/scratch/rgbov_000.jpg');b=cv2.imread('/work/scratch/rgbovflip_000.jpg');cv2.imwrite('/work/scratch/uvcompare.jpg',np.hstack([a,b]))
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
cap=cv2.VideoCapture('/work/inputs/video.mp4');_,im=cap.read();
with h5py.File('/work/inputs/masks.h5','r') as h:m=h['object'][0].astype(np.uint8);p=h['person'][0].astype(np.uint8)
x0,y0,x1,y1=620,650,820,950; c=im[y0:y1,x0:x1].copy()
# grid every 20
for x in range(0,x1-x0,20):
 cv2.line(c,(x,0),(x,c.shape[0]),(255,255,0),1); cv2.putText(c,str(x+x0),(x+1,12),cv2.FONT_HERSHEY_SIMPLEX,.25,(255,255,0),1)
for y in range(0,y1-y0,20):
 cv2.line(c,(0,y),(c.shape[1],y),(255,255,0),1); cv2.putText(c,str(y+y0),(1,y+10),cv2.FONT_HERSHEY_SIMPLEX,.25,(255,255,0),1)
# boundaries red/green
cont,_=cv2.findContours(m[y0:y1,x0:x1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(c,cont,-1,(0,255,0),1)
cont,_=cv2.findContours(p[y0:y1,x0:x1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(c,cont,-1,(0,0,255),1)
cv2.imwrite('/work/scratch/grid_orig.png',cv2.resize(c,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/experiment_static.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,cv2,json,h5py
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')
tex=np.array(mesh.visual.material.baseColorTexture.convert('RGB'))/255.;tex=torch.tensor(tex,dtype=torch.float32,device='cuda').permute(2,0,1)[None]
z=np.load('/work/inputs/reconstruction.npz'); cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=np.linalg.inv(Twc);K=np.array(cam['K']);W,H=1536,1152
with h5py.File('/work/inputs/masks.h5','r') as h: gtmask=h['object'][:];person=h['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4'); ims={}
for i in range(467):
 if i in [0,20,50,80,100,150,180,200,220,280,300,340,390,420,450,466]: cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,ims[i]=cap.read()
ctx=dr.RasterizeCudaContext()
def render(Ts, uvflip=False):
 Ts=torch.tensor(Ts,dtype=torch.float32,device='cuda');outs=[];mids=[]
 Tcinv=torch.tensor(Tci,dtype=torch.float32,device='cuda')
 for j in range(0,len(Ts),4):
  Tw=Ts[j:j+4];Vw=V[None].expand(len(Tw),-1,-1);Vw=torch.einsum('fij,fvj->fvi',Tw[:,:3,:3],Vw)+Tw[:,:3,3][:,None];Vc=torch.einsum('ij,fvj->f …[+3053 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/experiment_static.py", line 39, in <module>
    rgb,ms=render(Ts,uvf)
           ^^^^^^^^^^^^^^
  File "/work/scratch/experiment_static.py", line 21, in render
    r,_=dr.rasterize(ctx,clip,F,resolution=(H,W));uv,_=dr.interpolate(UV[None].expand(len(Tw),-1,-1),r,F);uv=uv[...,:2];grid=uv*2-1
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # t …[+371 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/UV\\[None\\].expand(len(Tw),-1,-1)/UV[None].repeat(len(Tw),1,1)/\" /work/scratch/experiment_static.py
python /work/scratch/experiment_static.py"
```
<details><summary>output</summary>

```
phase 0 uvflip False
Traceback (most recent call last):
  File "/work/scratch/experiment_static.py", line 53, in <module>
    ii=cv2.cvtColor(ims[i][y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(float); rr=cv2.cvtColor((rgb[ci][y0:y1,x0:x1]*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(float)
                    ~~~^^^
KeyError: 51

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/experiment_static.py';s=open(p).read();s=s.replace(\"for i in [a, (a+b)//2,b]:\",\"for i in [q for q in [a, (a+b)//2,b] if q in ims]:\");open(p,'w').write(s)
PY
python /work/scratch/experiment_static.py"
```
<details><summary>output</summary>

```
phase 0 uvflip False
I iou [0.8205] ncc [-0.1215] cham [20.23]
flipX iou [0.7096] ncc [-0.1874] cham [20.99]
flipY iou [0.7496] ncc [-0.1494] cham [17.22]
flipZ iou [0.7566] ncc [-0.0668] cham [22.89]
phase 0 uvflip True
I iou [0.8205] ncc [-0.441] cham [20.23]
flipX iou [0.7096] ncc [-0.2758] cham [20.99]
flipY iou [0.7496] ncc [-0.413] cham [17.22]
flipZ iou [0.7566] ncc [-0.2417] cham [22.89]
phase 1 uvflip False
I iou [] ncc [] cham []
flipX iou [] ncc [] cham []
flipY iou [] ncc [] cham []
flipZ iou [] ncc [] cham []
phase 1 uvflip True
I iou [] ncc [] cham []
flipX iou [] ncc [] cham []
flipY iou [] ncc [] cham []
flipZ iou [] ncc [] cham []
phase 2 uvflip False
I iou [] ncc [] cham []
flipX iou [] ncc [] cham []
flipY iou [] ncc [] cham []
flipZ iou [] ncc [] cham []
phase 2 uvflip  …[+512 chars]
```
</details>

**shell** (exit 137)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
parts=m.split(only_watertight=False)
print('parts',len(parts))
for i,p in enumerate(sorted(parts,key=lambda x:len(x.faces),reverse=True)[:20]):
 print(i,len(p.vertices),len(p.faces),p.bounds,p.extents,p.is_watertight)
PY"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 8:  4566 Killed                  python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
parts=m.split(only_watertight=False)
print('parts',len(parts))
for i,p in enumerate(sorted(parts,key=lambda x:len(x.faces),reverse=True)[:20]):
 print(i,len(p.vertices),len(p.faces),p.bounds,p.extents,p.is_watertight)
PY


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();nv=len(m.vertices)
a=m.faces[:,[0,0,1]].ravel();b=m.faces[:,[1,2,2]].ravel();g=coo_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(nv,nv));n,lab=connected_components(g,directed=False)
print('components',n)
for l in np.argsort(np.bincount(lab))[::-1][:30]:
 idx=np.where(lab==l)[0]; p=m.vertices[idx]; nf=(lab[m.faces[:,0]]==l).sum(); print(l,len(idx),nf,p.min(0),p.max(0))
np.save('/work/scratch/comp_labels.npy',lab)
PY"
```
<details><summary>output</summary>

```
components 97577
90163 4 2 [-0.00091717 -0.03529337  0.22711635] [ 0.00647445 -0.03486112  0.22928931]
80723 4 2 [-0.07270415 -0.10589344 -0.12683102] [-0.05946568 -0.10528823 -0.11316425]
47713 4 2 [ 0.01047308 -0.08021183 -0.26860955] [ 0.02271884 -0.07579726 -0.26839104]
39327 4 2 [ 0.00104814  0.13450123 -0.21911703] [ 0.00902114  0.13501649 -0.21231872]
42041 4 2 [-0.11443368 -0.11025617 -0.02841682] [-0.10214987 -0.10958348 -0.02318902]
39458 4 2 [-0.04684927 -0.110613   -0.0151701 ] [-0.03939789 -0.10871871 -0.01428948]
3893 4 2 [ 0.07514754 -0.05754188 -0.26751995] [ 0.07863051 -0.0506063  -0.26680416]
96406 4 2 [-0.16317755 -0.04784682 -0.23507118] [-0.15930726 -0.04114913 -0.23265664]
20222 4 2 [ 0.06858595 -0.09409841  0.47817731] [ 0.07228215 -0.09360152  0.50002223]
19288 4 2  …[+1795 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); print('before',len(m.vertices),len(m.faces));m.merge_vertices();print('merged',len(m.vertices),len(m.faces));parts=m.split(only_watertight=False);print('parts',len(parts))
for p in sorted(parts,key=lambda x:len(x.faces),reverse=True)[:30]: print(len(p.vertices),len(p.faces),p.bounds,p.extents)
PY"
```
<details><summary>output</summary>

```
before 292750 97596
merged 53688 97596
parts 46
16512 29980 [[-0.19353536 -0.12052371 -0.30923915]
 [ 0.13404821  0.12219062  0.22759455]] [0.32758357 0.24271432 0.5368337 ]
11768 21635 [[-0.14782175 -0.11943389 -0.24868071]
 [ 0.19024041  0.02708606  0.39865789]] [0.33806217 0.14651996 0.6473386 ]
8875 16681 [[-0.17391539  0.06011451 -0.30923915]
 [ 0.18864577  0.13631421  0.21648397]] [0.36256115 0.0761997  0.52572311]
6488 11993 [[-0.13265632 -0.03372075 -0.23507966]
 [ 0.18887281  0.11816051  0.22978003]] [0.32152914 0.15188126 0.46485969]
4687 8718 [[-0.16782311 -0.12052371 -0.31325984]
 [ 0.17561616  0.13373511 -0.20414476]] [0.34343927 0.25425881 0.10911508]
1309 2343 [[-0.1453945   0.01097337 -0.24729221]
 [ 0.11058886  0.11941531 -0.18052843]] [0.25598336 0.10844194 0.06676377]
69 …[+2905 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# create body-only mesh npz for experiments
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();m.merge_vertices()
for zmax in [0.2,0.23,0.25,0.3]:
 faces=m.faces[(m.vertices[m.faces][:,:,2].max(1)<=zmax)]
 sub=m.submesh([np.arange(len(m.faces))[m.vertices[m.faces][:,:,2].max(1)<=zmax]],append=True)
 print(zmax,len(sub.vertices),len(sub.faces),sub.bounds)
 np.savez('/work/scratch/body_'+str(zmax)+'.npz',vertices=sub.vertices,faces=sub.faces)
PY"
```
<details><summary>output</summary>

```
0.2 42633 78000 [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.19999382]]
0.23 48834 89352 [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.22998835]]
0.25 50198 92172 [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.24987058]]
0.3 50425 92552 [[-0.19353536 -0.12052371 -0.31731981]
 [ 0.19024041  0.13631421  0.2993004 ]]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_body.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,json,h5py
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tci=np.linalg.inv(Twc);W,H=1536,1152
with h5py.File('/work/inputs/masks.h5','r') as h:gt=h['object'][:]
ctx=dr.RasterizeCudaContext()
for zmax in [0.2,0.23,0.25,0.3]:
 d=np.load(f'/work/scratch/body_{zmax}.npz');V=torch.tensor(d['vertices'],dtype=torch.float32,device='cuda');F=torch.tensor(d['faces'],dtype=torch.int32,device='cuda')
 for phase,(a,b) in enumerate([(0,103),(145,219),(278,342),(382,466)]):
  tt=z['T_world_mesh'][a:b+1];Rm=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();t=np.median(tt[:,:3,3],0);T=np.eye(4);T[:3,:3]=Rm;T[:3,3]=t
  # candidates mean and each chosen refs 10 evenly
  ids=np.linspace(a,b,12).astype(int); cs=[]
  for ii in ids:
   X=z['T_world_mesh'][ii].copy();cs.append(X)
  cs.append(T)
  ious=[]
  for T0 in cs:
   Tc=np.linalg.inv(Twc)@T0;Vc=V@Tc[:3,:3].T+Tc[:3,3];x,y,dep=Vc[:,0],Vc[:,1],Vc[:,2];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
   clip=torch.stack([(2*fx/W)*x+(2*cx/W-1)*dep,(2*fy/H)*y+(2*cy/H-1)*dep,torch.zeros_like(dep),dep],-1)[None];r,_=dr.rasterize(ctx,clip,F,resolution=(H,W));m=(r[0,...,3]>0).cpu().numpy()
   vals=[]
   for ii in [a,(a+b)//2,b]:
    gg=gt[ii];vals.append((m&gg).sum()/(m|gg).sum())
   ious.append(np.mean(val …[+154 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_body.py", line 17, in <module>
    Tc=np.linalg.inv(Twc)@T0;Vc=V@Tc[:3,:3].T+Tc[:3,3];x,y,dep=Vc[:,0],Vc[:,1],Vc[:,2];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
                                ~^~~~~~~~~~~~
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/Tc=np.linalg.inv(Twc)@T0;Vc=V@Tc\\[:3,:3\\].T+Tc\\[:3,3\\]/Tc=torch.tensor(np.linalg.inv(Twc)@T0,dtype=torch.float32,device=\"cuda\");Vc=V@Tc[:3,:3].T+Tc[:3,3]/' /work/scratch/eval_body.py
python /work/scratch/eval_body.py"
```
<details><summary>output</summary>

```
z 0.2 phase 0 mean iou 0.8446837000673244 best single 0.8664437510589296 ids 65
z 0.2 phase 1 mean iou 0.8448963876972398 best single 0.877085015946938 ids 205
z 0.2 phase 2 mean iou 0.8623302130434588 best single 0.8875385897140103 ids 278
z 0.2 phase 3 mean iou 0.8655401327241262 best single 0.8800448581652236 ids 404
z 0.23 phase 0 mean iou 0.8591140068889368 best single 0.8818976973263744 ids 65
z 0.23 phase 1 mean iou 0.8653794945646407 best single 0.898589664494701 ids 205
z 0.23 phase 2 mean iou 0.8811061039366196 best single 0.9081143882908654 ids 278
z 0.23 phase 3 mean iou 0.8829534517782108 best single 0.898100142185644 ids 404
z 0.25 phase 0 mean iou 0.861403315869001 best single 0.8840666722411035 ids 65
z 0.25 phase 1 mean iou 0.8653989895188978 best single 0.8971927887547766 …[+494 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,json,h5py,cv2,time
from scipy.spatial.transform import Rotation as R
torch.manual_seed(0);np.random.seed(0)
# mesh body only generated with merge + z cutoff 0.23
body=np.load('/work/scratch/body_0.23.npz');V=torch.tensor(body['vertices'],dtype=torch.float32,device='cuda');F=torch.tensor(body['faces'],dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');K=np.array(cam['K']);W,H=1536,1152;S=.5;W2,H2=int(W*S),int(H*S)
ctx=dr.RasterizeCudaContext()
# data frames static 0
frames=np.array([0,15,30,45,60,75,90,103])
with h5py.File('/work/inputs/masks.h5','r') as h: gtfull=h['object'][frames]
gt=torch.tensor(np.stack([cv2.resize(g.astype(np.float32),(W2,H2),interpolation=cv2.INTER_AREA) for g in gtfull]),device='cuda')
# depths
depths=[]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h:
 for i in frames:
  bb=h['aligned']['front_stereo_camera_left'][i]; a=cv2.imdecode(bb,cv2.IMREAD_UNCHANGED).astype(np.float32)/1000;depths.append(cv2.resize(a,(W2,H2),interpolation=cv2.INTER_AREA))
deptht=torch.tensor(np.stack(depths),device='cuda')
# init mean
tt=z['T_world_mesh'][0:104];R0=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();p0=np.median(tt[:,:3,3],axis=0)
R0=torch.tensor(R0,dtype=torch.float …[+2310 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_static.py", line 35, in <module>
    a,dep,r=renderall(torch.zeros(3,device='cuda'),torch.zeros(3,device='cuda'),True); print('init alpha',a.min().item(),a.max().item(),a.mean().item(),'depth',dep.min().item(),dep.max().item())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 31, in renderall
    alpha=dr.antialias(torch.ones_like(clip[...,:1]),rast,clip,F,resolution=(H2,W2))[...,0]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: antialias() got an unexpected keyword argument 'resolution'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.signature(dr.antialias))
PY"
```
<details><summary>output</summary>

```
(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/,resolution=(H2,W2))/)/' /work/scratch/opt_static.py
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_static.py", line 35, in <module>
    a,dep,r=renderall(torch.zeros(3,device='cuda'),torch.zeros(3,device='cuda'),True); print('init alpha',a.min().item(),a.max().item(),a.mean().item(),'depth',dep.min().item(),dep.max().item())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 30, in renderall
    rast,_=dr.rasterize(ctx,clip,F)
           ^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: rasterize() missing 1 required positional argument: 'resolution'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/dr.rasterize(ctx,clip,F)"'$/dr.rasterize(ctx,clip,F,resolution=(H2,W2))/'"' /work/scratch/opt_static.py
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_static.py", line 35, in <module>
    a,dep,r=renderall(torch.zeros(3,device='cuda'),torch.zeros(3,device='cuda'),True); print('init alpha',a.min().item(),a.max().item(),a.mean().item(),'depth',dep.min().item(),dep.max().item())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 31, in renderall
    alpha=dr.antialias(torch.ones_like(clip[...,:1]),rast,clip,F)[...,0]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 526, in antialias
    return _antialias_func.apply(color, rast, pos, tri, topology_hash, pos_gradient_boost)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+580 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/torch.ones_like(clip\\[...,:1\\])/torch.ones((len(frames),H2,W2,1),device=\"cuda\")/' /work/scratch/opt_static.py
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_static.py", line 35, in <module>
    a,dep,r=renderall(torch.zeros(3,device='cuda'),torch.zeros(3,device='cuda'),True); print('init alpha',a.min().item(),a.max().item(),a.mean().item(),'depth',dep.min().item(),dep.max().item())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 32, in renderall
    dep,_=dr.interpolate(d[None,None].repeat(len(frames),1,1).contiguous(),rast,F);dep=dep[...,0]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^ …[+1018 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/d\\[None,None\\].repeat(len(frames),1,1)/d[None,:,None].repeat(len(frames),1,1)/' /work/scratch/opt_static.py
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
init alpha 1.0 1.0 1.0 depth 0.0 3.3571159839630127
0 11.544694900512695 iou 0.006177548784762621 dw [0. 0. 0.] dt [-0.005  0.005  0.005]
50 11.47338581085205 iou 0.006177548784762621 dw [0. 0. 0.] dt [-0.01056486 -0.00535063  0.06377036]
100 11.474148750305176 iou 0.006177548784762621 dw [0. 0. 0.] dt [-0.01086984 -0.00566657  0.06586166]
150 11.472965240478516 iou 0.006177548784762621 dw [0. 0. 0.] dt [-0.01028633 -0.00409258  0.06606357]
200 11.473396301269531 iou 0.006177548784762621 dw [0. 0. 0.] dt [-0.01080164 -0.00584358  0.06608609]
250 11.472987174987793 iou 0.006177548784762621 dw [0. 0. 0.] dt [-0.00976958 -0.00435654  0.06585733]
300 11.473099708557129 iou 0.006177548784762621 dw [0. 0. 0.] dt [-0.01047418 -0.00488673  0.06585657]
350 11.474432945251465 iou 0.00617754878476262 …[+181 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static.py';s=open(p).read();s=s.replace(\"alpha=dr.antialias(torch.ones((len(frames),H2,W2,1),device=\\\"cuda\\\"),rast,clip,F)[...,0]\",\"maskattr,_=dr.interpolate(torch.ones((len(frames),len(V),1),device=\\\"cuda\\\"),rast,F)\\n alpha=dr.antialias(maskattr,rast,clip,F)[...,0]\")
open(p,'w').write(s)
PY
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
init alpha 0.0 1.0 0.0053685675375163555 depth 0.0 3.3571159839630127
0 0.11568634957075119 iou 0.8585243821144104 dw [0. 0. 0.] dt [-0.005  0.005  0.005]
50 0.04829062893986702 iou 0.8261106610298157 dw [0. 0. 0.] dt [-0.00990882 -0.00516537  0.0632266 ]
100 0.0469205379486084 iou 0.8242018222808838 dw [0. 0. 0.] dt [-0.00977786 -0.00339533  0.06630415]
150 0.046555232256650925 iou 0.8242278099060059 dw [0. 0. 0.] dt [-0.0100362  -0.00442582  0.06616227]
200 0.04793833941221237 iou 0.8259601593017578 dw [0. 0. 0.] dt [-0.00868943 -0.00465319  0.0658712 ]
250 0.04656578600406647 iou 0.8230853080749512 dw [0. 0. 0.] dt [-0.01010936 -0.00353411  0.06583864]
300 0.04693824052810669 iou 0.8220223784446716 dw [0. 0. 0.] dt [-0.01171976 -0.00421692  0.06575709]
350 0.04672233387827873 iou 0.8217 …[+195 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_candidates.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,json,h5py,cv2
from scipy.spatial.transform import Rotation as R
body=np.load('/work/scratch/body_0.23.npz');V=torch.tensor(body['vertices'],dtype=torch.float32,device='cuda');F=torch.tensor(body['faces'],dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');K=np.array(cam['K']);W,H=1536,1152;ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:gt=h['object'][:];person=h['person'][:]
def decodedepth(i):
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h:return cv2.imdecode(h['aligned']['front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
def render(T,K=K,W=W,H=H):
 Tc=torch.tensor(np.linalg.inv(Twc)@T,dtype=torch.float32,device='cuda');Vc=V@Tc[:3,:3].T+Tc[:3,3];x,y,d=Vc[:,0],Vc[:,1],Vc[:,2];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2];zn=(d-3.5)/1.5;clip=torch.stack([(2*fx/W)*x+(2*cx/W-1)*d,(2*fy/H)*y+(2*cy/H-1)*d,zn*d,d],-1)[None];r,_=dr.rasterize(ctx,clip,F,resolution=(H,W));dep,_=dr.interpolate(d[None,:,None],r,F);return (r[0,...,3]>0).cpu().numpy(),dep[0,...,0].cpu().numpy()
cands={}
for phase,(a,b) in enumerate([(0,103),(145,219),(278,342),(382,466)]):
 tt=z['T_world_mesh'][a:b+1];R0=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();p0=np. …[+1160 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_candidates.py", line 20, in <module>
    m,dep=render(T);g=gt[i];iou=(m&g).sum()/max(1,(m|g).sum());ious.append(iou)
          ^^^^^^^^^
  File "/work/scratch/eval_candidates.py", line 9, in render
    Tc=torch.tensor(np.linalg.inv(Twc)@T,dtype=torch.float32,device='cuda');Vc=V@Tc[:3,:3].T+Tc[:3,3];x,y,d=Vc[:,0],Vc[:,1],Vc[:,2];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2];zn=(d-3.5)/1.5;clip=torch.stack([(2*fx/W)*x+(2*cx/W-1)*d,(2*fy/H)*y+(2*cy/H-1)*d,zn*d,d],-1)[None];r,_=dr.rasterize(ctx,clip,F,resolution=(H,W));dep,_=dr.interpolate(d[None,:,None],r,F);return (r[0,...,3]>0).cpu().numpy(),dep[0,...,0].cpu().numpy()
                                                                                                                        …[+955 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/d\\[None,:,None\\]/d.reshape(1,-1,1).contiguous()/' /work/scratch/eval_candidates.py
python /work/scratch/eval_candidates.py"
```
<details><summary>output</summary>

```
mean0 iou 0.858352442934267 [0.857 0.849 0.851 0.859 0.856 0.867 0.87  0.852 0.864] depth med -0.067 mae med 0.069 centered 0.019 resall [-0.113 -0.068 -0.006]
base0 iou 0.8767612867664814 [0.876 0.867 0.87  0.878 0.874 0.886 0.889 0.87  0.882] depth med -0.066 mae med 0.069 centered 0.02 resall [-0.112 -0.067 -0.006]
mean1 iou 0.8653949688792666 [0.862 0.852 0.864 0.868 0.861 0.865 0.873 0.871 0.873] depth med -0.068 mae med 0.073 centered 0.028 resall [-0.149 -0.064  0.011]
base1 iou 0.8709535913169746 [0.867 0.858 0.869 0.874 0.867 0.87  0.878 0.876 0.879] depth med -0.088 mae med 0.091 centered 0.031 resall [-0.172 -0.085 -0.005]
mean2 iou 0.8829967878125754 [0.885 0.88  0.889 0.886 0.888 0.881 0.887 0.88  0.871] depth med -0.095 mae med 0.095 centered 0.018 resall [-0.177 -0.085 -0.03 …[+644 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np,json
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h:
 for i in [0,100,200,300,400]:
  z=cv2.imdecode(h['aligned']['front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
  # sample floor region y 900:1100, x 100:1400 excluding person maybe
  ys,xs=np.mgrid[900:1100:10,100:1400:20];zz=z[ys,xs]; X=(xs-K[0,2])*zz/K[0,0];Y=(ys-K[1,2])*zz/K[1,1];P=np.stack([X,Y,zz],-1)@Twc[:3,:3].T+Twc[:3,3];dist=P@n+d
  print(i,np.percentile(dist,[10,25,50,75,90]),'mean',dist.mean())
PY"
```
<details><summary>output</summary>

```
0 [-0.07235831 -0.06982754 -0.05997496 -0.05424397 -0.04911829] mean -0.06114063527758066
100 [-0.10522574 -0.10152027 -0.08231655 -0.07384214 -0.06678826] mean -0.08593617926594738
200 [-0.09190854 -0.08933747 -0.07977027 -0.07527715 -0.07101111] mean -0.08117803553280364
300 [-0.08724651 -0.08198207 -0.07248762 -0.06767383 -0.06269207] mean -0.07291260579400267
400 [-0.08226577 -0.07947463 -0.07172875 -0.06595774 -0.06313753] mean -0.06835569275599833

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];ph=[(0,103),(145,219),(278,342),(382,466)]
for a,b in ph:
 tt=T[a:b+1]; R0=R.from_matrix(tt[:,:3,:3]).mean(); dr=R.from_matrix(tt[:,:3,:3])*R0.inv(); ang=np.degrees(np.linalg.norm(dr.as_rotvec(),axis=1));p=tt[:,:3,3]
 print(a,b,'t std',p.std(0),'range',p.max(0)-p.min(0),'rot mean/stdev/max',ang.mean(),ang.std(),ang.max(),'median dist cm',np.median(np.linalg.norm(p-np.median(p,0),axis=1))*100)
PY"
```
<details><summary>output</summary>

```
0 103 t std [0.00327743 0.00624594 0.01868265] range [0.0160352  0.03868258 0.11133957] rot mean/stdev/max 1.6820824523877513 0.9891657168346293 4.789197748998283 median dist cm 1.3616666197776794
145 219 t std [0.00421866 0.01325045 0.0402808 ] range [0.01824287 0.04914218 0.13666129] rot mean/stdev/max 1.1519283542887913 0.4790169923814824 2.597957524282164 median dist cm 2.1084604784846306
278 342 t std [0.00456724 0.00831249 0.02823894] range [0.01850046 0.03700358 0.11916757] rot mean/stdev/max 0.899577599897676 0.39615729550744977 2.1934386987458026 median dist cm 1.8044749274849892
382 466 t std [0.00161294 0.00561554 0.01295094] range [0.00713125 0.02255797 0.05447984] rot mean/stdev/max 0.7528842147721143 0.3385984120834972 1.6588409487257878 median dist cm 1.0193096473813057

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/depth_floor_calib.py <<'PY'
import numpy as np,cv2,h5py,json
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Rwc=Twc[:3,:3];C=Twc[:3,3];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,D=pl[:3],pl[3]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h, h5py.File('/work/inputs/masks.h5','r') as hm:
 for i in [0,50,100,150,200,250,300,350,400,450]:
  z=cv2.imdecode(h['aligned']['front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float64)/1000
  obj=hm['object'][i];per=hm['person'][i]
  ys,xs=np.mgrid[650:1100:4,100:1450:4];zz=z[ys,xs];valid=(zz>0)&(~obj[ys,xs])&(~per[ys,xs]);xs=xs[valid];ys=ys[valid];zz=zz[valid]
  rays=np.stack([(xs-K[0,2])/K[0,0],(ys-K[1,2])/K[1,1],np.ones_like(zz)],1);rays=rays@Rwc.T
  A=(rays*n[:,None]).sum(1)*zz;B=(rays*n[:,None]).sum(1);C0=C@n+D
  dist=A+C0
  # use currently near floor under d*1 scale, iterative
  keep=np.abs(dist)<.25
  for _ in range(5):
   M=np.stack([A[keep],B[keep]],1); y=-C0*np.ones(keep.sum()); sol=np.linalg.lstsq(M,y,rcond=None)[0];a,b=sol;dist=A*a+B*b+C0;keep=np.abs(dist)<.06
  print(i,'a,b',a,b,'n',keep.sum(),'dist orig',np.percentile(A+C0,[5,50,95]),'corrected',np.percentile(dist,[5,50,95]))
PY
python /work/scratch/depth_floor_calib.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_floor_calib.py", line 9, in <module>
    A=(rays*n[:,None]).sum(1)*zz;B=(rays*n[:,None]).sum(1);C0=C@n+D
       ~~~~^~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (36791,3) (3,1) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/n\\[:,None\\]/n/g' /work/scratch/depth_floor_calib.py
python /work/scratch/depth_floor_calib.py"
```
<details><summary>output</summary>

```
0 a,b 0.9407324206698344 0.04735282255322599 n 33853 dist orig [-0.08832617 -0.06257645  0.14809206] corrected [-0.01488709  0.00093098  0.2120142 ]
50 a,b 0.937184668272481 0.07396583291707166 n 33907 dist orig [-0.08619253 -0.05223431  0.14862186] corrected [-0.01849585  0.00217023  0.20990714]
100 a,b 0.9193222876208729 0.06310088785098236 n 34079 dist orig [-0.12228044 -0.08720062  0.11407904] corrected [-0.02168505  0.00307125  0.20488729]
150 a,b 0.9244503239927562 0.04833237053897172 n 33439 dist orig [-0.11440625 -0.08828235  0.13784204] corrected [-0.01742344  0.00120297  0.22304066]
200 a,b 0.9245970735382137 0.0558338847243183 n 33668 dist orig [-0.11087122 -0.08380196  0.1379759 ] corrected [-0.01496351  0.00147812  0.2215145 ]
250 a,b 0.8692129472154018 0.05456587335887746 n 3 …[+706 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];
with h5py.File('/work/inputs/masks.h5','r') as h:m=h['object'][:]
for i in [0,20,40,60,65,80,100,103]: print(i,T[i,:3,3],m[i].sum())
PY"
```
<details><summary>output</summary>

```
0 [-0.04836467  0.9280976   3.2313282 ] 11062
20 [-0.05899233  0.9269453   3.2170873 ] 11192
40 [-0.06439987  0.93205154  3.238938  ] 11024
60 [-0.05893345  0.92271477  3.2162788 ] 10989
65 [-0.05853708  0.9264241   3.229574  ] 10949
80 [-0.06058463  0.92544734  3.2187364 ] 10972
100 [-0.05710417  0.9223646   3.2209134 ] 11043
103 [-0.05727472  0.9173814   3.2038677 ] 10972

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];tt=T[:104];Rm=R.from_matrix(tt[:,:3,:3]).mean();t=np.median(tt[:,:3,3],0);print('mean',t,Rm.as_rotvec(),'frame65',T[65,:3,3],R.from_matrix(T[65,:3,:3]).as_rotvec(),'delta',R.from_matrix(Rm.as_matrix()).inv()*R.from_matrix(T[65,:3,:3]))
PY"
```
<details><summary>output</summary>

```
mean [-0.05796903  0.9231534   3.2179637 ] [ 1.26041876  1.20985319 -1.11196512] frame65 [-0.05853708  0.9264241   3.229574  ] [ 1.26900569  1.17619298 -1.08302778] delta Rotation.from_matrix(array([[ 0.99917327, -0.03976672, -0.00844841],
                            [ 0.03981567,  0.99919074,  0.00570756],
                            [ 0.0082146 , -0.00603922,  0.99994802]]))

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/opt_static.py /work/scratch/opt_static_mask.py
python - <<'PY'
p='/work/scratch/opt_static_mask.py';s=open(p).read()
s=s.replace(\"th=torch.linalg.norm(w); \\n if th<1e-8:return torch.eye(3,device='cuda')\\n k=w/th;\",\"th=torch.linalg.norm(w)+1e-8\\n k=w/th;\")
s=s.replace('return bce+2.0*dl+0.002*(dw.square().sum()+20*dt.square().sum())','return bce+0.08*dl+0.01*(dw.square().sum()+20*dt.square().sum())')
s=s.replace('lr=.005','lr=.001').replace('range(401)','range(1201)').replace('it%50==0','it%100==0').replace('it==400','it==1200')
s=s.replace('/work/scratch/opt_phase0.npy','/work/scratch/opt_mask_phase0.npy')
open(p,'w').write(s)
PY
python /work/scratch/opt_static_mask.py"
```
<details><summary>output</summary>

```
init alpha 0.0 1.0 0.0053685675375163555 depth 0.0 3.3571159839630127
0 0.014783341437578201 iou 0.8605349063873291 dw [0. 0. 0.] dt [-0.001       0.00099999  0.001     ]
100 0.014352857135236263 iou 0.8377614617347717 dw [0. 0. 0.] dt [-0.00830892 -0.00368348  0.05251652]
200 0.01441438589245081 iou 0.836303174495697 dw [0. 0. 0.] dt [-0.00811888 -0.00328831  0.05191641]
300 0.014360938221216202 iou 0.8348448276519775 dw [0. 0. 0.] dt [-0.00872991 -0.00349018  0.0519669 ]
400 0.014365310780704021 iou 0.8355739712715149 dw [0. 0. 0.] dt [-0.00829195 -0.00312555  0.05195182]
500 0.014449753798544407 iou 0.8348448276519775 dw [0. 0. 0.] dt [-0.00823202 -0.00245546  0.05179856]
600 0.014413752593100071 iou 0.8348448276519775 dw [0. 0. 0.] dt [-0.00793877 -0.0035354   0.0518824 ]
700 0.0143456 …[+630 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -A4 'def rod' /work/scratch/opt_static_mask.py"
```
<details><summary>output</summary>

```
def rod(w):
 th=torch.linalg.norm(w)+1e-8
 k=w/th; Kx=torch.tensor([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]],device='cuda')
 return torch.eye(3,device='cuda')+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)
def renderall(dw,dt,wantalpha=False):

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static_mask.py';s=open(p).read();s=s.replace(\"opt.zero_grad();loss=lossfn(dw,dt);loss.backward();opt.step()\",\"opt.zero_grad();loss=lossfn(dw,dt);loss.backward()\\n if it in [0,1,10]: print('grad',dw.grad,dt.grad)\\n opt.step()\")
open('/work/scratch/opt_dbg.py','w').write(s.replace('range(1201)','range(11)').replace('/work/scratch/opt_mask_phase0.npy','/work/scratch/dbg.npy'))
PY
python /work/scratch/opt_dbg.py"
```
<details><summary>output</summary>

```
init alpha 0.0 1.0 0.0053685675375163555 depth 0.0 3.3571159839630127
grad tensor([0., 0., 0.], device='cuda:0') tensor([ 0.0064, -0.0014, -0.0669], device='cuda:0')
0 0.014783341437578201 iou 0.8605349063873291 dw [0. 0. 0.] dt [-0.001       0.00099999  0.001     ]
grad tensor([0., 0., 0.], device='cuda:0') tensor([-0.0001, -0.0007, -0.0665], device='cuda:0')
grad tensor([0., 0., 0.], device='cuda:0') tensor([-0.0057, -0.0021, -0.0601], device='cuda:0')
time 0.5243809223175049

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
for p in ['/work/scratch/opt_static.py','/work/scratch/opt_static_mask.py']:
 s=open(p).read();s=s.replace(\"Kx=torch.tensor([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]],device='cuda')\",\"zero=k.new_zeros(()); Kx=torch.stack([torch.stack([zero,-k[2],k[1]]),torch.stack([k[2],zero,-k[0]]),torch.stack([-k[1],k[0],zero])])\")
 open(p,'w').write(s)
PY
python /work/scratch/opt_static_mask.py"
```
<details><summary>output</summary>

```
init alpha 0.0 1.0 0.0053685675375163555 depth 0.0 3.3571159839630127
0 0.014783341437578201 iou 0.8608988523483276 dw [ 0.001      -0.00099999  0.00099996] dt [-0.001       0.00099999  0.001     ]
100 0.01384756900370121 iou 0.8439593315124512 dw [ 0.04715851 -0.01485082  0.00761972] dt [-0.00899023 -0.00345328  0.05038632]
200 0.013989169150590897 iou 0.8422514796257019 dw [ 0.04454798 -0.0096404   0.0075442 ] dt [-0.00878338 -0.00360026  0.05035517]
300 0.01399725116789341 iou 0.8415805697441101 dw [ 0.0436686  -0.00775765  0.00876246] dt [-0.00851689 -0.00384251  0.05039292]
400 0.013939909636974335 iou 0.8426157832145691 dw [ 0.04245871 -0.00739877  0.00909664] dt [-0.00850002 -0.0040614   0.05060347]
500 0.013988370075821877 iou 0.842673122882843 dw [ 0.04456407 -0.00782355  0.009488 …[+980 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_static_mask.py';s=open(p).read().replace('return bce+0.08*dl+0.01*(dw.square().sum()+20*dt.square().sum())','return bce+0.001*(dw.square().sum()+10*dt.square().sum())').replace('/work/scratch/opt_mask_phase0.npy','/work/scratch/opt_maskonly_phase0.npy')
open('/work/scratch/opt_maskonly.py','w').write(s)
PY
python /work/scratch/opt_maskonly.py"
```
<details><summary>output</summary>

```
init alpha 0.0 1.0 0.0053685675375163555 depth 0.0 3.3571159839630127
0 0.010579049587249756 iou 0.8624681234359741 dw [ 0.00099991 -0.00099997 -0.00099997] dt [ 0.001      -0.00099999  0.00099998]
100 0.006689160130918026 iou 0.9109300971031189 dw [ 0.00683275 -0.06984748 -0.00252935] dt [ 0.00426918 -0.01674592 -0.02191138]
200 0.006639134604483843 iou 0.9140398502349854 dw [ 0.00776153 -0.08112159  0.00471512] dt [ 0.00244008 -0.01607062 -0.019537  ]
300 0.006508448626846075 iou 0.9148608446121216 dw [ 0.00903028 -0.0814928   0.00518704] dt [ 0.00263592 -0.0158759  -0.01904311]
400 0.006551423110067844 iou 0.9135512113571167 dw [ 0.01056722 -0.08259694  0.00562425] dt [ 0.00256703 -0.01566948 -0.01853613]
500 0.006523368880152702 iou 0.9146623015403748 dw [ 0.01336904 -0.08283231  0.005 …[+979 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_candidates.py';s=open(p).read();s=s.replace(\"cands['opt0']=np.load('/work/scratch/opt_phase0.npy')\",\"cands['opt0']=np.load('/work/scratch/opt_phase0.npy')\\nif __import__('os').path.exists('/work/scratch/opt_maskonly_phase0.npy'):cands['optm']=np.load('/work/scratch/opt_maskonly_phase0.npy')\")
open(p,'w').write(s)
PY
python /work/scratch/eval_candidates.py"
```
<details><summary>output</summary>

```
mean0 iou 0.858352442934267 [0.857 0.849 0.851 0.859 0.856 0.867 0.87  0.852 0.864] depth med -0.067 mae med 0.069 centered 0.019 resall [-0.113 -0.068 -0.006]
base0 iou 0.8767612867664814 [0.876 0.867 0.87  0.878 0.874 0.886 0.889 0.87  0.882] depth med -0.066 mae med 0.069 centered 0.02 resall [-0.112 -0.067 -0.006]
mean1 iou 0.8653949688792666 [0.862 0.852 0.864 0.868 0.861 0.865 0.873 0.871 0.873] depth med -0.068 mae med 0.073 centered 0.028 resall [-0.149 -0.064  0.011]
base1 iou 0.8709535913169746 [0.867 0.858 0.869 0.874 0.867 0.87  0.878 0.876 0.879] depth med -0.088 mae med 0.091 centered 0.031 resall [-0.172 -0.085 -0.005]
mean2 iou 0.8829967878125754 [0.885 0.88  0.889 0.886 0.888 0.881 0.887 0.88  0.871] depth med -0.095 mae med 0.095 centered 0.018 resall [-0.177 -0.085 -0.03 …[+803 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.array(m.vertices);pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
z=np.load('/work/inputs/reconstruction.npz');tt=z['T_world_mesh'][:104];Rm=R.from_matrix(tt[:,:3,:3]).mean();p=np.median(tt[:,:3,3],0);T=np.eye(4);T[:3,:3]=Rm.as_matrix();T[:3,3]=p
for name,X in [('mean',T),('optm',np.load('/work/scratch/opt_maskonly_phase0.npy'))]:
 P=V@X[:3,:3].T+X[:3,3];s=P@n+d;print(name,'p',X[:3,3],'min noise',s.min(),'anglechange',np.degrees(np.linalg.norm(R.from_matrix(X[:3,:3]@T[:3,:3].T).as_rotvec())))
PY"
```
<details><summary>output</summary>

```
mean p [-0.05796903  0.9231534   3.2179637 ] min noise -0.02806204897063691 anglechange 0.0
optm p [-0.05563122  0.9075535   3.1998262 ] min noise -0.013776286328527787 anglechange 4.886830140874732

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/optimize_all_static.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,json,h5py,cv2,time
from scipy.spatial.transform import Rotation as R
torch.manual_seed(0);np.random.seed(0)
body=np.load('/work/scratch/body_0.23.npz');V=torch.tensor(body['vertices'],dtype=torch.float32,device='cuda');F=torch.tensor(body['faces'],dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');K=np.array(cam['K']);W,H=1536,1152;S=.5;W2,H2=int(W*S),int(H*S);ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:allgt=h['object'][:]
phases=[(0,103),(145,219),(278,342),(382,466)]
def rod(w):
 th=torch.linalg.norm(w)+1e-8;k=w/th;zero=k.new_zeros(());Kx=torch.stack([torch.stack([zero,-k[2],k[1]]),torch.stack([k[2],zero,-k[0]]),torch.stack([-k[1],k[0],zero])]);return torch.eye(3,device='cuda')+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)
def makeloss(frames,R0,p0,gt,lb):
 def renderall(dw,dt):
  Re=rod(dw)@R0;p=p0+dt;Vw=V@Re.T+p;Vc=Vw@Tci[:3,:3].T+Tci[:3,3];x,y,d=Vc[:,0],Vc[:,1],Vc[:,2];fx,fy,cx,cy=K[0,0]*S,K[1,1]*S,K[0,2]*S,K[1,2]*S;zn=(d-3.5)/1.5;clip=torch.stack([(2*fx/W2)*x+(2*cx/W2-1)*d,(2*fy/H2)*y+(2*cy/H2-1)*d,zn*d,d],-1)[None].repeat(len(frames),1,1);rast,_=dr.rasterize(ctx,clip,F,resolution=(H2,W2));maskattr,_=dr.interpolate(torch.ones((len(frames),len(V),1) …[+1741 chars]
```
<details><summary>output</summary>

```
phase 0 frames 0 103 iou 0.8622879981994629 0.9127320051193237 dw [-0.03101094 -0.07350396  0.0170043 ] dt [ 0.00258246 -0.01450474 -0.02282516] T [-0.05538657  0.90864867  3.1951385 ]
phase 1 frames 145 219 iou 0.8733644485473633 0.8989921808242798 dw [-0.00672718  0.06037926 -0.04707124] dt [ 0.00363894 -0.02053281 -0.01736397] T [-0.18854012  0.9033024   3.0166488 ]
phase 2 frames 278 342 iou 0.8909465074539185 0.9035996794700623 dw [ 0.0296047   0.00383089 -0.06781257] dt [-0.0035705  -0.0122885  -0.02568939] T [-0.23753352  0.9230917   2.708019  ]
phase 3 frames 382 466 iou 0.8886826038360596 0.911896288394928 dw [ 0.03945654  0.01274562 -0.06411657] dt [-0.00973486 -0.01288414 -0.04950011] T [0.00735904 0.92357725 2.4962904 ]
runtime 24.043457746505737

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.array(m.vertices);pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
o=np.load('/work/scratch/static_optimized.npz')
for i in range(4):
 T=o[str(i)];P=V@T[:3,:3].T+T[:3,3];s=P@n+d;print(i,'minplane',s.min(),'pct',np.percentile(s,[0,.1,1,5]),'center',T[:3,3])
PY"
```
<details><summary>output</summary>

```
0 minplane -0.018472124229937048 pct [-0.01847212 -0.0115579   0.00746426  0.03471644] center [-0.05538657  0.90864867  3.1951385 ]
1 minplane -0.00235118781023802 pct [-0.00235119  0.0031947   0.02630936  0.05218073] center [-0.18854012  0.9033024   3.0166488 ]
2 minplane -0.0063617305171479455 pct [-0.00636173  0.00349891  0.02582133  0.05130969] center [-0.23753352  0.9230917   2.708019  ]
3 minplane 0.004095877193101671 pct [0.00409588 0.01194693 0.03174559 0.0580182 ] center [0.00735904 0.92357725 2.4962904 ]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as h:m=h['object'][:]
def sheet(frames,fn):
 ims=[]
 for i in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read(); y,x=np.where(m[i]);x0,x1=max(0,int(x.mean())-150),min(1536,int(x.mean())+150); y0,y1=max(0,int(y.mean())-180),min(1152,int(y.mean())+180); c=im[y0:y1,x0:x1];c=cv2.resize(c,(240,288));cv2.putText(c,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2);ims.append(c)
 rows=[np.hstack(ims[j:j+6]) for j in range(0,len(ims),6)];cv2.imwrite(fn,np.vstack(rows))
sheet(list(range(95,161,5)),'/work/scratch/t1.jpg')
sheet(list(range(210,291,5)),'/work/scratch/t2.jpg')
sheet(list(range(330,396,5)),'/work/scratch/t3.jpg')
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
  File "<stdin>", line 8, in sheet
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/shape_base.py", line 289, in vstack
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1440 and the array at index 2 has size 480

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/x'; import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as h:m=h['object'][:]
def sheet(frames,fn):
 ims=[]
 for i in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read(); y,x=np.where(m[i]);x0,x1=max(0,int(x.mean())-150),min(1536,int(x.mean())+150); y0,y1=max(0,int(y.mean())-180),min(1152,int(y.mean())+180); c=im[y0:y1,x0:x1];c=cv2.resize(c,(240,288));cv2.putText(c,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2);ims.append(c)
 while len(ims)%6: ims.append(np.zeros_like(ims[0]))
 rows=[np.hstack(ims[j:j+6]) for j in range(0,len(ims),6)];cv2.imwrite(fn,np.vstack(rows))
sheet(list(range(95,161,5)),'/work/scratch/t1.jpg')
sheet(list(range(210,291,5)),'/work/scratch/t2.jpg')
sheet(list(range(330,396,5)),'/work/scratch/t3.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/phases=\\[(0,103),(145,219),(278,342),(382,466)\\]/phases=[(0,103),(150,219),(251,341),(380,466)]/' /work/scratch/optimize_all_static.py > /work/scratch/optimize_all_static2.py
python /work/scratch/optimize_all_static2.py"
```
<details><summary>output</summary>

```
phase 0 frames 0 103 iou 0.8622879981994629 0.907471239566803 dw [ 0.02896889 -0.07052426  0.02034757] dt [ 0.00109593 -0.02375355 -0.0065654 ] T [-0.05687311  0.8993999   3.2113984 ]
phase 1 frames 150 219 iou 0.8773926496505737 0.8949374556541443 dw [ 0.0247053   0.04100166 -0.04841042] dt [-0.00098569 -0.01489535 -0.01303494] T [-0.19308397  0.90694624  3.0096326 ]
phase 2 frames 251 341 iou 0.8965057730674744 0.8861777782440186 dw [ 0.01149735  0.01451003 -0.04191101] dt [ 0.01243022 -0.0104597  -0.00837091] T [-0.22033821  0.92291546  2.7135203 ]
phase 3 frames 380 466 iou 0.8879802227020264 0.9257202744483948 dw [ 0.02733251  0.05551223 -0.03796243] dt [-0.00541938 -0.0093883  -0.0365233 ] T [0.01168329 0.92679    2.5081294 ]
runtime 28.578673124313354

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_static_poses.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,json,h5py,cv2,os
from scipy.spatial.transform import Rotation as R
body=np.load('/work/scratch/body_0.23.npz');V=torch.tensor(body['vertices'],dtype=torch.float32,device='cuda');F=torch.tensor(body['faces'],dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');K=np.array(cam['K']);W,H=1536,1152;ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:gt=h['object'][:]
def render(T):
 Tc=torch.tensor(np.linalg.inv(Twc)@T,dtype=torch.float32,device='cuda');Vc=V@Tc[:3,:3].T+Tc[:3,3];x,y,d=Vc[:,0],Vc[:,1],Vc[:,2];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2];clip=torch.stack([(2*fx/W)*x+(2*cx/W-1)*d,(2*fy/H)*y+(2*cy/H-1)*d,torch.zeros_like(d),d],-1)[None];r,_=dr.rasterize(ctx,clip,F,resolution=(H,W));return (r[0,...,3]>0).cpu().numpy()
phases=[(0,103),(150,219),(251,341),(380,466)]
opts=np.load('/work/scratch/static_optimized.npz')
for pi,(a,b) in enumerate(phases):
 tt=z['T_world_mesh'][a:b+1];Rm=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();T=np.eye(4);T[:3,:3]=Rm;T[:3,3]=np.median(tt[:,:3,3],0)
 cands={'mean':T,'opt':opts[str(pi)]}
 ids=np.linspace(a,b,25).astype(int)
 print('phase',pi,a,b)
 for n,X in cands.items():
  ms=render(X);ious=np.array([(ms&gt[i]).sum()/(ms|g …[+177 chars]
```
<details><summary>output</summary>

```
phase 0 0 103
  mean iou mean med min 0.8578663343652673 0.8572457245724573 0.842859670854716 pct [0.8464272  0.86916212]
  opt iou mean med min 0.8986419561592917 0.8970298801216675 0.8887710928527255 pct [0.89114124 0.90461814]
phase 1 150 219
  mean iou mean med min 0.8691575667764856 0.8691713781992826 0.8564418646400574 pct [0.86296282 0.87517649]
  opt iou mean med min 0.8877777939850957 0.8888821274265198 0.8752761358887098 pct [0.88309421 0.89214437]
phase 2 251 341
  mean iou mean med min 0.888258973758792 0.8884153808474161 0.8786046061544417 pct [0.88326614 0.89202262]
  opt iou mean med min 0.8853180857095819 0.8853144134512586 0.8780089367222409 pct [0.88164195 0.88831918]
phase 3 380 466
  mean iou mean med min 0.8810210701048808 0.879241090505085 0.8769125857733453 pct [0.87 …[+127 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];F=len(T)
rv=R.from_matrix(T[:,:3,:3]).as_rotvec();rv=np.unwrap(rv,axis=0);rv=gaussian_filter1d(rv,1.5,axis=0,mode='nearest');Bs=T.copy();Bs[:,:3,:3]=R.from_rotvec(rv).as_matrix();Bs[:,:3,3]=gaussian_filter1d(T[:,:3,3],1.5,axis=0,mode='nearest')
static=np.load('/work/scratch/static_optimized.npz');pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3];Vm=np.array(trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices)
# floor snap
Fs={}
for i in range(4):
 X=static[str(i)].copy();P=Vm@X[:3,:3].T+X[:3,3];dist=(P@n+d).min();X[:3,3]-=dist*n;Fs[i]=X
for k,(t0,t1) in enumerate([(104,149),(220,250),(342,379)],start=0):
 Cs=Fs[k]@np.linalg.inv(Bs[t0]);Ce=Fs[k+1]@np.linalg.inv(Bs[t1]);
 print('trans',t0,t1,'Cs t',Cs[:3,3],'ang',np.degrees(np.linalg.norm(R.from_matrix(Cs[:3,:3]).as_rotvec())),'Ce t',Ce[:3,3],'ang',np.degrees(np.linalg.norm(R.from_matrix(Ce[:3,:3]).as_rotvec())))
PY"
```
<details><summary>output</summary>

```
trans 104 149 Cs t [0.45637384 0.03453952 0.03490138] ang 7.854259100091897 Ce t [-0.10579795  0.02610993 -0.05762601] ang 3.991572067812673
trans 220 250 Cs t [-0.23001648  0.07242513  0.0300982 ] ang 4.602156369180773 Ce t [-0.15025741 -0.01219636  0.00019169] ang 3.4894389330400735
trans 342 379 Cs t [ 0.01354077  0.05223787 -0.04122162] ang 3.527641898463065 Ce t [-0.18137531  0.06266516 -0.04380274] ang 3.982790150938851

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];rv=np.unwrap(R.from_matrix(T[:,:3,:3]).as_rotvec(),axis=0);rv=gaussian_filter1d(rv,1.5,axis=0,mode='nearest');Bs=T.copy();Bs[:,:3,:3]=R.from_rotvec(rv).as_matrix();Bs[:,:3,3]=gaussian_filter1d(T[:,:3,3],1.5,axis=0,mode='nearest')
static=np.load('/work/scratch/static_optimized.npz');pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3];Vm=np.array(trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().vertices);Fs={}
for i in range(4):
 X=static[str(i)].copy();P=Vm@X[:3,:3].T+X[:3,3];dist=(P@n+d).min();X[:3,3]-=dist*n;Fs[i]=X
for k,(t0,t1) in enumerate([(104,149),(220,250),(342,379)]):
 Cs=np.linalg.inv(Bs[t0])@Fs[k];Ce=np.linalg.inv(Bs[t1])@Fs[k+1];print(k,'start',Cs[:3,3],np.degrees(np.linalg.norm(R.from_matrix(Cs[:3,:3]).as_rotvec())),'end',Ce[:3,3],np.degrees(np.linalg.norm(R.from_matrix(Ce[:3,:3]).as_rotvec())))
PY"
```
<details><summary>output</summary>

```
0 start [-0.01263857  0.00499378  0.02518094] 7.854259302517292 end [0.03126359 0.01521361 0.03199327] 3.9915718197825374
1 start [-0.05021214 -0.02877152 -0.00013041] 4.602156725540892 end [-0.01550245  0.00839889  0.012833  ] 3.489438439470782
2 start [0.01255417 0.01701319 0.01538527] 3.5276421051296314 end [0.02805901 0.00336921 0.00814265] 3.982791070860724

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/assemble_eval.py <<'PY'
import numpy as np,torch,nvdiffrast.torch as dr,trimesh,json,h5py,cv2
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
# inputs
z=np.load('/work/inputs/reconstruction.npz');B=z['T_world_mesh'].copy();F=len(B)
cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=np.linalg.inv(Twc);K=np.array(cam['K']);W,H=1536,1152
# body mesh
bd=np.load('/work/scratch/body_0.23.npz');V=torch.tensor(bd['vertices'],dtype=torch.float32,device='cuda');Fbl=torch.tensor(bd['faces'],dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext()
# static candidates
stat=np.load('/work/scratch/static_optimized.npz');phases=[(0,103),(150,219),(251,341),(380,466)]
# use mean for phase2 as pooled candidate had lower IoU
for pi in range(4):
 a,b=phases[pi];tt=B[a:b+1]
 if pi==2 or stat[str(pi)] is None:
  T=np.eye(4);T[:3,:3]=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();T[:3,3]=np.median(tt[:,:3,3],0);stat[str(pi)]=T
# floor snap all
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();Vall=np.array(mesh.vertices);pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
Ss=[]
for i in range(4):
 T=stat[str(i)].copy();P=Vall@T[:3,:3].T+T[:3,3];dist=(P@n+d).min();T[:3,3]-=dist*n;Ss.append(T);print('floor snap',i,dist,T[:3,3])
# smooth baseline (rotation rv unwrapped and translation)
rv=np.unwrap(R.fr …[+2454 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/assemble_eval.py", line 15, in <module>
    T=np.eye(4);T[:3,:3]=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();T[:3,3]=np.median(tt[:,:3,3],0);stat[str(pi)]=T
                                                                                                       ~~~~^^^^^^^^^
TypeError: 'NpzFile' object does not support item assignment

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/assemble_eval.py';s=open(p).read();s=s.replace(\"stat=np.load('/work/scratch/static_optimized.npz');phases=\",\"stat=dict(np.load('/work/scratch/static_optimized.npz'));phases=\")
s=s.replace(\"Rr=R.from_matrix(Cs[:3,:3]).inv()*R.from_matrix(Ce[:3,:3]); # wrong composition? scipy mult local\\n  Rc=(R.from_matrix(Cs[:3,:3])*R.from_rotvec(s*Rr.as_rotvec())).as_matrix()\",\"Rr=R.from_matrix(Cs[:3,:3].T@Ce[:3,:3]).as_rotvec()\\n  Rc=Cs[:3,:3]@R.from_rotvec(s*Rr).as_matrix()\")
open(p,'w').write(s)
PY
python /work/scratch/assemble_eval.py"
```
<details><summary>output</summary>

```
floor snap 0 -0.009580955877671604 [-0.0570602   0.88983583  3.2108607 ]
floor snap 1 -0.006281552278715852 [-0.19320664  0.9006758   3.0092802 ]
floor snap 2 -0.014061890673423338 [-0.23304303  0.91933812  2.72110226]
floor snap 3 0.002539454772100491 [0.01173288 0.929325   2.508272  ]
base all 0.8770639238702767 0.8791975018180263 [0.84864503 0.85916488 0.89318439 0.90285215]
 static 0 103 0.8586314433913523
 static 150 219 0.8739422541723648
 static 251 341 0.8907008627157821
 static 380 466 0.8838556193967282
 trans 104 149 0.8527009984340637
 trans 220 250 0.8973349887767709
 trans 342 379 0.8980099096830166
ours all 0.8917330549306088 0.8885721218003154 [0.87543881 0.88319369 0.90595819 0.92079554]
 static 0 103 0.8836271728117342
 static 150 219 0.8858736728436419
 static 251 341 0. …[+332 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];stat=np.load('/work/scratch/static_optimized.npz')
for i in range(4):
 for n,X in [('base',T[[0,150,251,380][i]]),('opt',stat[str(i)])]:
  up=X[:3,:3]@np.array([0,0,1.]);ang=np.degrees(np.arccos(np.clip(up@np.array([0,-1,0.]),-1,1)));print(i,n,'up',up,'angle to -y',ang)
PY"
```
<details><summary>output</summary>

```
0 base up [ 0.02235055 -0.99938738 -0.02693311] angle to -y 2.005645450677178
0 opt up [ 0.05408546 -0.99540895 -0.07896759] angle to -y 5.492370569662359
1 base up [ 0.02039714 -0.99871856 -0.04630373] angle to -y 2.900902951423574
1 opt up [-0.04283763 -0.99695599 -0.06514366] angle to -y 4.4716812547406155
2 base up [ 0.00635943 -0.99910128 -0.04190103] angle to -y 2.429305217064298
2 opt up [-0.03342797 -0.99838269 -0.04598444] angle to -y 3.259065901832443
3 base up [ 0.02746836 -0.99923062 -0.02800004] angle to -y 2.247682732771313
3 opt up [-0.0059286  -0.99854761 -0.05354864] angle to -y 3.0883839836634093

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])[:3];up=-n;print(up)
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];stat=np.load('/work/scratch/static_optimized.npz')
for i in range(4):
 for nm,X in [('base',T[[0,150,251,380][i]]),('opt',stat[str(i)])]:
  u=X[:3,:3]@np.array([0,0,1.]);print(i,nm,np.degrees(np.arccos(np.clip(u@up,-1,1))))
PY"
```
<details><summary>output</summary>

```
[0.01952785 0.99823403 0.05610241]
0 base 177.07484808611935
0 opt 175.5794644202765
1 base 177.6434420508306
1 opt 178.56560201095093
2 base 178.306973330587
2 opt 179.01486682110527
3 base 176.86189665474527
3 opt 179.20687120291439

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])[:3]
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];stat=np.load('/work/scratch/static_optimized.npz')
for i in range(4):
 for nm,X in [('base',T[[0,150,251,380][i]]),('opt',stat[str(i)])]:
  u=X[:3,:3]@np.array([0,0,1.]);print(i,nm,np.degrees(np.arccos(np.clip(u@n,-1,1))))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_upright.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,json,h5py,cv2,sys
from scipy.spatial.transform import Rotation as R
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0;torch.manual_seed(seed);np.random.seed(seed)
body=np.load('/work/scratch/body_0.23.npz');V=torch.tensor(body['vertices'],dtype=torch.float32,device='cuda');F=torch.tensor(body['faces'],dtype=torch.int32,device='cuda');mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();Vall=np.array(mesh.vertices)
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');K=np.array(cam['K']);W,H=1536,1152;S=.5;W2,H2=int(W*S),int(H*S);ctx=dr.RasterizeCudaContext();pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nv=pl[:3]
with h5py.File('/work/inputs/masks.h5','r') as h:allgt=h['object'][:]
phases=[(0,103),(150,219),(251,341),(380,466)];pi=int(sys.argv[2]) if len(sys.argv)>2 else 0;a,b=phases[pi]
frames=np.unique(np.linspace(a,b,12).astype(int));gt=torch.tensor(np.stack([cv2.resize(allgt[i].astype(np.float32),(W2,H2),interpolation=cv2.INTER_AREA) for i in frames]),device='cuda')
tt=z['T_world_mesh'][a:b+1];Rbase=R.from_matrix(tt[:,:3,:3]).mean();up=Rbase.apply([0,0,1.]);v=np.cross(up,nv);v=v/(np.linalg.norm(v)+1e-9);ang=np.arccos(np.clip(up@nv,-1,1));Rup=R.from_rotvec …[+1902 chars]
```
<details><summary>output</summary>

```
pi 0 iou 0.8678498864173889 yaw -0.005598350428044796 ang -0.32076185176222466 dt [ 0.00195645 -0.01833081 -0.02291135] floordist -0.01256942998961752 T [[ 0.06290092  0.99782878 -0.01952785 -0.05625804]
 [ 0.05477533 -0.0229887  -0.99823403  0.89227536]
 [-0.99651551  0.06172019 -0.05610241  3.19434721]
 [ 0.          0.          0.          1.        ]]
pi 1 iou 0.9013441205024719 yaw -0.03616353124380112 ang -2.0720177125592927 dt [ 0.00480384 -0.0172771  -0.01924067] floordist 0.0010025002697275909 T [[-0.48497671  0.874309   -0.01952785 -0.18727488]
 [ 0.05852699  0.01016901 -0.99823403  0.90556523]
 [-0.87256652 -0.48526317 -0.05610241  3.00348327]
 [ 0.          0.          0.          1.        ]]
pi 2 iou 0.9106583595275879 yaw -0.014915814623236656 ang -0.8546132259109765 dt [ 0. …[+630 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
exec(open('/work/scratch/eval_static_poses.py').read().split(\"phases=\")[0])
phases=[(0,103),(150,219),(251,341),(380,466)]
for pi,(a,b) in enumerate(phases):
 cand={'up':np.load(f'/work/scratch/up_p{pi}_s0.npy')}
 # also free new
 D=dict(np.load('/work/scratch/static_optimized.npz'));cand['free']=D[str(pi)]
 print('phase',pi)
 ids=np.linspace(a,b,25).astype(int)
 for nm,X in cand.items():
  ms=render(X);ious=np.array([(ms&gt[i]).sum()/(ms|gt[i]).sum() for i in ids]);print(nm,ious.mean(),np.median(ious),ious.min())
PY"
```
<details><summary>output</summary>

```
phase 0
up 0.8592236333910283 0.8587074583635047 0.8463107484923732
free 0.8986419561592917 0.8970298801216675 0.8887710928527255
phase 1
up 0.8959957000979806 0.8959931798806479 0.8838380809595202
free 0.8877777939850957 0.8888821274265198 0.8752761358887098
phase 2
up 0.9051118352568465 0.9057697029216794 0.8940988217039229
free 0.8853180857095819 0.8853144134512586 0.8780089367222409
phase 3
up 0.9163114438933436 0.9149704670037819 0.9124004068486185
free 0.9223268235065377 0.9208663767374435 0.9184489136442876

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_upright.py';s=open(p).read();s=s.replace(\"seed=int(sys.argv[1]) if len(sys.argv)>1 else 0;\",\"seed=int(sys.argv[1]) if len(sys.argv)>1 else 0; init=float(sys.argv[3]) if len(sys.argv)>3 else 0;\")
s=s.replace(\"yawn=torch.zeros((),device='cuda',requires_grad=True);dt=torch.zeros(3,device='cuda',requires_grad=True)\",\"yawn=torch.tensor(init,device='cuda',requires_grad=True);dt=torch.tensor([0.,-0.015,-0.02],device='cuda',requires_grad=True)\")
open('/work/scratch/opt_upright_init.py','w').write(s)
PY
for s in -0.15 -0.08 0.08 0.15; do python /work/scratch/opt_upright_init.py 0 0 "'$s; done'
```
<details><summary>output</summary>

```
pi 0 iou 0.880064070224762 yaw -0.1001981645822525 ang -5.740931945520273 dt [ 0.00713019 -0.03230908 -0.02259241] floordist 0.0012651809317814067 T [[-0.031634    0.99930882 -0.01952785 -0.05081414]
 [ 0.0567019  -0.0177119  -0.99823409  0.89210729]
 [-0.99788994 -0.0326854  -0.05610241  3.19544237]
 [ 0.          0.          0.          1.        ]]
pi 0 iou 0.8618729114532471 yaw -0.014923729002475739 ang -0.8550666864388419 dt [-0.00167584 -0.01809392 -0.01568527] floordist -0.013140345406349674 T [[ 0.05359319  0.99837196 -0.01952785 -0.05990147]
 [ 0.05498733 -0.02247691 -0.99823403  0.89194232]
 [-0.99704778  0.05242477 -0.05610241  3.20154117]
 [ 0.          0.          0.          1.        ]]
pi 0 iou 0.8717561364173889 yaw 0.03963596746325493 ang 2.2709736525623594 dt [ 0.000354 …[+620 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# eval upright initialized candidates phase0
exec(open('/work/scratch/eval_static_poses.py').read().split(\"phases=\")[0])
a,b=0,103;ids=np.linspace(a,b,25).astype(int)
# files overwritten? each init saved same up_p0_s0. only last +0.15. need initial -5.74 T from output not saved after later overwrite. rerun separately later.
# currently up file +4.88; evaluate
for nm,X in [('up_last',np.load('/work/scratch/up_p0_s0.npy'))]:
 ms=render(X);ious=np.array([(ms&gt[i]).sum()/(ms|gt[i]).sum() for i in ids]);print(nm,ious.mean(),np.median(ious))
PY"
```
<details><summary>output</summary>

```
up_last 0.8573217386180755 0.8550469999121497

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/opt_upright_init.py 0 0 -0.15 > /work/scratch/upneg.txt && cat /work/scratch/upneg.txt && cp /work/scratch/up_p0_s0.npy /work/scratch/up_p0_neg.npy'
```
<details><summary>output</summary>

```
pi 0 iou 0.8795831799507141 yaw -0.11378408223390579 ang -6.519347687772294 dt [ 0.00599072 -0.02952402 -0.00954688] floordist -0.002224531773118432 T [[-0.0452072   0.99878669 -0.01952785 -0.05202175]
 [ 0.05693729 -0.01693994 -0.99823403  0.89140877]
 [-0.99735367 -0.04623923 -0.05610241  3.2082919 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_static_poses.py').read().split(\"phases=\")[0]);ids=np.linspace(0,103,25).astype(int);import numpy as np
for nm,X in [('upneg',np.load('/work/scratch/up_p0_neg.npy'))]:
 ms=render(X);ious=np.array([(ms&gt[i]).sum()/(ms|gt[i]).sum() for i in ids]);print(nm,ious.mean(),np.median(ious),ious.min())
PY"
```
<details><summary>output</summary>

```
upneg 0.8680270685635065 0.8671004545859702 0.8598568525227533

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_static_poses.py').read().split(\"phases=\")[0])
import cv2,numpy as np
stat=dict(np.load('/work/scratch/static_optimized.npz'))
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,30,60,90,100]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();
 # mean baseline
 tt=z['T_world_mesh'][:104];T=np.eye(4);T[:3,:3]=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();T[:3,3]=np.median(tt[:,:3,3],0)
 masks={'gt':gt[i],'base':render(T),'free':render(stat['0']),'upneg':render(np.load('/work/scratch/up_p0_neg.npy'))}
 cols={'gt':(0,255,0),'base':(0,0,255),'free':(255,0,0),'upneg':(0,255,255)}
 x0,y0,x1,y1=630,650,790,920;c=im[y0:y1,x0:x1].copy()
 for nm,m in masks.items():
  cont,_=cv2.findContours(m[y0:y1,x0:x1].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(c,cont,-1,cols[nm],1)
 cv2.putText(c,f'f{i}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
 cv2.imwrite(f'/work/scratch/edge_{i}.jpg',cv2.resize(c,None,fx=3,fy=3))
PY
python - <<'PY'
import cv2,numpy as np
ims=[cv2.imread(f'/work/scratch/edge_{i}.jpg') for i in [0,30,60,90,100]];cv2.imwrite('/work/scratch/edges.jpg',np.hstack(ims))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_fullmesh.py <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np,trimesh,json,h5py
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tci=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device='cuda');K=np.array(cam['K']);W,H=1536,1152;ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:gt=h['object'][:]
def render(T):
 Tc=torch.tensor(np.linalg.inv(Twc)@T,dtype=torch.float32,device='cuda');Vc=V@Tc[:3,:3].T+Tc[:3,3];x,y,d=Vc[:,0],Vc[:,1],Vc[:,2];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2];clip=torch.stack([(2*fx/W)*x+(2*cx/W-1)*d,(2*fy/H)*y+(2*cy/H-1)*d,torch.zeros_like(d),d],-1)[None];r,_=dr.rasterize(ctx,clip,F,resolution=(H,W));return (r[0,...,3]>0).cpu().numpy()
T=np.load('/work/scratch/static_optimized.npz')['0'];Tup=np.load('/work/scratch/up_p0_neg.npy');base=z['T_world_mesh'].copy();tt=base[:104];Tm=np.eye(4);Tm[:3,:3]=R.from_matrix(tt[:,:3,:3]).mean().as_matrix();Tm[:3,3]=np.median(tt[:,:3,3],0)
cands={'mean':Tm,'free':T,'up':Tup}
for nm,Rs in [('z',np.diag([-1,-1,1.])),('x',np.diag([1,-1,-1.])),('y',np.diag([-1,1,-1.]))]:
 X=T.copy();X[:3,:3]=X[:3,:3]@Rs;cands['freeflip'+ …[+239 chars]
```
<details><summary>output</summary>

```
mean 0.8213412938831587 0.8215745613957444 0.8093223923525928
free 0.8372147552231112 0.8381348312645306 0.8294592812396967
up 0.810487117042361 0.8082965293351456 0.8046476636295243
freeflipz 0.7618509512459758 0.7597987330628966 0.7536845406083412
freeflipx 0.676838792954873 0.6759398495350846 0.6706895226381736
freeflipy 0.6962079513685985 0.6973510581090783 0.6884875846501128

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
T=np.load('/work/scratch/trajectory_proto.npy');m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.array(m.vertices);pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
mins=[]
for X in T:
 P=V@X[:3,:3].T+X[:3,3];mins.append((P@n+d).min())
mins=np.array(mins)
for a,b in [(0,103),(104,149),(150,219),(220,250),(251,341),(342,379),(380,466)]:print(a,b,np.percentile(mins[a:b+1],[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
0 103 [1.25468518e-08 1.25468518e-08 1.25468518e-08 1.25468518e-08
 1.25468518e-08]
104 149 [-1.89732501e-03 -4.51269805e-05  2.44577000e-03  5.79152297e-03
  7.12095640e-03]
150 219 [7.43684692e-09 7.43684692e-09 7.43684692e-09 7.43684692e-09
 7.43684692e-09]
220 250 [-2.94373670e-08  4.13270489e-04  3.90229467e-03  8.01756597e-03
  9.49409546e-03]
251 341 [8.87996698e-09 8.87996698e-09 8.87996698e-09 8.87996698e-09
 8.87996698e-09]
342 379 [-0.00180525  0.00119962  0.006095    0.01068842  0.01234134]
380 466 [-2.12179718e-08 -2.12179718e-08 -2.12179718e-08 -2.12179718e-08
 -2.12179718e-08]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/assemble_eval.py';s=open(p).read()
# insert floor snap after assembly before save
needle=\"np.save('/work/scratch/trajectory_proto.npy',Tout)\"
insert=\"\"\"# wheels stay on the released floor throughout the episode
for _t in range(F):
 P=Vall@Tout[_t,:3,:3].T+Tout[_t,:3,3];dist=(P@n+d).min();Tout[_t,:3,3]-=dist*n
np.save('/work/scratch/trajectory_proto.npy',Tout)\"\"\"
s=s.replace(needle,insert)
open('/work/scratch/assemble_eval2.py','w').write(s)
PY
python /work/scratch/assemble_eval2.py"
```
<details><summary>output</summary>

```
floor snap 0 -0.009580955877671604 [-0.0570602   0.88983583  3.2108607 ]
floor snap 1 -0.006281552278715852 [-0.19320664  0.9006758   3.0092802 ]
floor snap 2 -0.014061890673423338 [-0.23304303  0.91933812  2.72110226]
floor snap 3 0.002539454772100491 [0.01173288 0.929325   2.508272  ]
base all 0.8770639238702767 0.8791975018180263 [0.84864503 0.85916488 0.89318439 0.90285215]
 static 0 103 0.8586314433913523
 static 150 219 0.8739422541723648
 static 251 341 0.8907008627157821
 static 380 466 0.8838556193967282
 trans 104 149 0.8527009984340637
 trans 220 250 0.8973349887767709
 trans 342 379 0.8980099096830166
ours all 0.8923505720429419 0.8886520070658463 [0.87579543 0.88319369 0.90961728 0.92076708]
 static 0 103 0.8836271728117342
 static 150 219 0.8858736728436419
 static 251 341 0. …[+331 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent world-space rolling-luggage trajectory.

The method has three parts:
  1. Pool the static parts of the episode and fit one rigid pose per plateau.
     A small differentiable silhouette renderer (nvdiffrast) optimises the pose
     against all object masks in the plateau.  Several deterministic
     initialisations/frame subsets are tried and the pooled full-mask score
     selects one.  Telescoping-handle faces are omitted from the fitting mesh
     because the released SAM2 object mask consistently covers the body/wheels
     but not the thin handle; the complete mesh is still used for scoring.
  2. Preserve the baseline'"'s measured motion during the three short pushes,
     but smooth it and warp it to end exactly at the two adjacent pooled poses.
  3. Keep the wheels on the released floor plane.  The exact mesh and plane are
     from the inputs; this is a hard physical constraint, not a learned table.

No per-frame answer or fitted correction table is embedded.  All poses are
recomputed from the inputs and shipped reconstruction with deterministic Adam
optimisation.
\"\"\"
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as Fnn
import trimesh
from scipy.ndimage import ga …[+25589 chars]
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
Fitting static plateau 0: 0-103
    pooled_mean      score=0.84590 bodyIoU=0.85434 fullIoU=0.81212
    free_dense       score=0.84590 bodyIoU=0.85434 fullIoU=0.81212
    free_coarse      score=0.84590 bodyIoU=0.85434 fullIoU=0.81212
    free_strided     score=0.84590 bodyIoU=0.85434 fullIoU=0.81212
    upright0         score=0.83314 bodyIoU=0.84071 fullIoU=0.80284
    upright_neg      score=0.83314 bodyIoU=0.84071 fullIoU=0.80284
  selected pooled_mean: [-0.05851703  0.89514091  3.21638935]
Fitting static plateau 1: 150-219
    pooled_mean      score=0.85853 bodyIoU=0.87113 fullIoU=0.80815
    free_dense       score=0.85853 bodyIoU=0.87113 fullIoU=0.80815
    free_coarse      score=0.85853 bodyIoU=0.87113 fullIoU=0.80815
Traceback (most recent call last):
  File "/work/outputs/solve.py", l …[+871 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/outputs');import solve,torch,numpy as np,json,h5py
solve.set_deterministic()
# load minimal
import trimesh
camera=json.load(open('/work/inputs/camera.json'));plane=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);rec=np.load('/work/inputs/reconstruction.npz');B=rec['T_world_mesh'].astype(float)
fm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mm=fm.copy();mm.merge_vertices();bv=np.asarray(mm.vertices,float);bf=np.asarray(mm.faces,np.int32);bf=bf[bv[bf][:,:,2].max(1)<=.23]
r=solve.Renderer(bv,bf,np.asarray(fm.vertices,float),np.asarray(fm.faces,np.int32),np.linalg.inv(np.asarray(camera['T_world_cam'],float)),np.asarray(camera['K'],float),1536,1152)
with h5py.File('/work/inputs/masks.h5','r') as f:masks=f['object']
phase=(0,103);frames=np.unique(np.linspace(0,103,8).astype(int));init=solve.mean_pose(B[:104]);W2,H2=768,576
import cv2
gt=np.asarray(masks[frames],dtype=np.float32);gt=torch.tensor(np.stack([cv2.resize(x,(W2,H2),interpolation=cv2.INTER_AREA) for x in gt]),device='cuda')
R0=torch.tensor(init[:3,:3],dtype=torch.float32,device='cuda');p0=torch.tensor(init[:3,3],dtype=torch.float32,device='cuda');dw=torch.zeros(3,device='cuda',requires_grad=True);dt=torch.zeros(3,device='cuda',requires_grad=True)
R=solve.rodrigues(dw)@R0;p=p0+dt;poses=torch.eye(4,device='cuda').repeat(len(frames),1,1);poses[:,:3,:3]=R;poses[:,:3,3]=p
print('poses grad',poses …[+305 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
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
  File "/opt/conda/lib/python3.11/site-pack …[+795 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/outputs');import solve,torch,numpy as np,json,h5py,cv2,trimesh
solve.set_deterministic();camera=json.load(open('/work/inputs/camera.json'));rec=np.load('/work/inputs/reconstruction.npz');B=rec['T_world_mesh'].astype(float);fm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mm=fm.copy();mm.merge_vertices();bv=np.asarray(mm.vertices,float);bf=np.asarray(mm.faces,np.int32);bf=bf[bv[bf][:,:,2].max(1)<=.23];r=solve.Renderer(bv,bf,np.asarray(fm.vertices,float),np.asarray(fm.faces,np.int32),np.linalg.inv(np.asarray(camera['T_world_cam'],float)),np.asarray(camera['K'],float),1536,1152)
with h5py.File('/work/inputs/masks.h5','r') as f:
 masks=f['object'];frames=np.unique(np.linspace(0,103,8).astype(int));init=solve.mean_pose(B[:104]);W2,H2=768,576;gt=np.asarray(masks[frames],dtype=np.float32);gt=torch.tensor(np.stack([cv2.resize(x,(W2,H2),interpolation=cv2.INTER_AREA) for x in gt]),device='cuda')
R0=torch.tensor(init[:3,:3],dtype=torch.float32,device='cuda');p0=torch.tensor(init[:3,3],dtype=torch.float32,device='cuda');dw=torch.zeros(3,device='cuda',requires_grad=True);dt=torch.zeros(3,device='cuda',requires_grad=True);R=solve.rodrigues(dw)@R0;p=p0+dt;poses=torch.eye(4,device='cuda').repeat(len(frames),1,1);poses[:,:3,:3]=R;poses[:,:3,3]=p
print('poses grad',poses.requires_grad,poses.grad_fn);alpha=r.soft_body_masks(poses,np.ones(len(frames)),W2,H2);print('alpha',alpha.requires …[+210 chars]
```
<details><summary>output</summary>

```
poses grad True <CopySlices object at 0x7156eec8cac0>
alpha True <SelectBackward0 object at 0x7156eec8cac0> 0.0 0.0 0.0
grads tensor([0., 0., 0.], device='cuda:0') tensor([0., 0., 0.], device='cuda:0')

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
# inspect quickly
import sys;sys.path.insert(0,'/work/outputs');import solve,numpy as np,json,torch,trimesh
camera=json.load(open('/work/inputs/camera.json'));rec=np.load('/work/inputs/reconstruction.npz');B=rec['T_world_mesh'].astype(float);fm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mm=fm.copy();mm.merge_vertices();bv=np.asarray(mm.vertices,float);bf=np.asarray(mm.faces,np.int32);bf=bf[bv[bf][:,:,2].max(1)<=.23];r=solve.Renderer(bv,bf,np.asarray(fm.vertices,float),np.asarray(fm.faces,np.int32),np.linalg.inv(np.asarray(camera['T_world_cam'],float)),np.asarray(camera['K'],float),1536,1152);T=torch.tensor(B[:1],device='cuda');clip=solve.project_clip(r.body_V,T,r.Tci,r.K,768,576);print(clip.min(1).values,clip.max(1).values);print(T)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/work/outputs/solve.py", line 78, in project_clip
    Vw = torch.einsum("fij,vj->fvi", T_world_mesh[:, :3, :3], V) + T_world_mesh[:, :3, 3][:, None]
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/functional.py", line 402, in einsum
    return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: expected scalar type Float but found Double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"        T = torch.as_tensor(poses, dtype=torch.float32, device=self.dev)\\n        clip = project_clip(self.body_V, T, self.Tci, self.K, W, H)\",\"        T = torch.as_tensor(poses, dtype=torch.float32, device=self.dev)\\n        K = self.K.copy()\\n        K[:2] *= float(W) / self.W\\n        clip = project_clip(self.body_V, T, self.Tci, K, W, H)\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
python - <<'PY'
# verify alpha/grad
import sys;sys.path.insert(0,'/work/outputs');import solve,torch,numpy as np,json,h5py,cv2,trimesh
solve.set_deterministic();camera=json.load(open('/work/inputs/camera.json'));rec=np.load('/work/inputs/reconstruction.npz');B=rec['T_world_mesh'].astype(float);fm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mm=fm.copy();mm.merge_vertices();bv=np.asarray(mm.vertices,float);bf=np.asarray(mm.faces,np.int32);bf=bf[bv[bf][:,:,2].max(1)<=.23];r=solve.Renderer(bv,bf,np.asarray(fm.vertices,float),np.asarray(fm.faces,np.int32),np.linalg.inv(np.asarray(camera['T_world_cam'],float)),np.asarray(camera['K'],float),1536,1152)
with h5py.File('/work/inputs/masks.h5','r') as f:
 frames=np.unique(np.linspace(0,103,8).astype(int));gt=torch.tensor(np.stack([cv2.resize(x.astype(np.float32),(768,576),interpolation=cv2.INTER_AREA) for x in f['object'][frames]]),device='cuda');init=solve.mean_pose(B[:104]);R0=torch.tensor …[+575 chars]
```
<details><summary>output</summary>

```
0.0 1.0 0.005364094860851765
tensor([1.6533e-05, 3.0063e-04, 3.3292e-06], device='cuda:0') tensor([-1.4420e-03,  8.0388e-04,  7.1061e-05], device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
Fitting static plateau 0: 0-103
    pooled_mean      score=0.84590 bodyIoU=0.85434 fullIoU=0.81212
    free_dense       score=0.88903 bodyIoU=0.90188 fullIoU=0.83763
    free_coarse      score=0.88038 bodyIoU=0.89313 fullIoU=0.82938
    free_strided     score=0.87994 bodyIoU=0.89266 fullIoU=0.82903
    upright0         score=0.85316 bodyIoU=0.86389 fullIoU=0.81025
    upright_neg      score=0.85011 bodyIoU=0.86005 fullIoU=0.81031
  selected free_dense: [-0.05587777  0.89179453  3.19861128]
Fitting static plateau 1: 150-219
    pooled_mean      score=0.85853 bodyIoU=0.87113 fullIoU=0.80815
    free_dense       score=0.86452 bodyIoU=0.87764 fullIoU=0.81204
    free_coarse      score=0.86294 bodyIoU=0.87636 fullIoU=0.80926
    free_strided     score=0.86193 bodyIoU=0.87445 fullIoU=0.81185
    …[+3317 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/outputs/REPORT.md'
```
<details><summary>output</summary>

```
# Rolling-luggage trajectory recovery

## Method

The episode was split into four static plateaus separated by three left-foot pushes. For each plateau, the baseline poses were pooled to a single rigid pose and refined with a differentiable antialiased nvdiffrast silhouette fit over all sampled frames of the plateau. Multiple deterministic frame subsets and free/upright initialisations were scored on the full plateau; the complete textured mesh, including its asymmetric handle, was used as a held-out silhouette check. The handle faces were excluded only from the fitting silhouette because SAM2 omits the thin telescoping handle.

During pushes the method low-pass filters the shipped baseline motion and right-multiplies a local SE(3) correction that joins the two adjacent pooled poses exactl …[+2208 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/outputs/TOOLS.md'
```
<details><summary>output</summary>

```
# Tools used

All processing is offline. No external repository or model weight was downloaded for this solution.

| tool | version | needed by solve.py | use |
|---|---|---|---|
| Python | 3.11.10 | yes | runtime |
| PyTorch | 2.5.1+cu124 | yes | differentiable rasterisation / Adam |
| nvdiffrast | 0.4.0 | yes | antialiased silhouette rasterisation |
| NumPy | 1.26.3 | yes | array geometry |
| SciPy | 1.16.3 | yes | rotations, smoothing, unwrap |
| trimesh | 5.1.1 | yes | load exact GLB mesh |
| h5py | 3.16.0 | yes | released object masks |
| OpenCV | installed (version metadata unavailable) | yes | mask resizing and image I/O |
| CUDA runtime | 12.4 | yes | GPU rasterisation |

The solution reads the released `/work/inputs` files and the shipped `/work/inputs/reconstruction.npz`; it does …[+77 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
p='/work/outputs/object_poses_world.npz';z=np.load(p);print(z.files);T=z['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all(),T[0],T[-1]);
# det/orthonormal checks
R=T[:,:3,:3];print('det',np.linalg.det(R).min(),np.linalg.det(R).max(),'orth err',np.abs(np.einsum('fij,fkj->fik',R,R)-np.eye(3)).max())
print('tr range',T[:,:3,3].min(0),T[:,:3,3].max(0))
PY"
```
<details><summary>output</summary>

```
['T_world_mesh']
(467, 4, 4) float64 True [[ 0.1521343   0.98725641  0.04668935 -0.05587777]
 [ 0.07994591  0.03479242 -0.99619186  0.89179453]
 [-0.98512125  0.15528756 -0.073634    3.19861128]
 [ 0.          0.          0.          1.        ]] [[-0.54753876  0.83677304  0.00347782  0.01183905]
 [ 0.03188685  0.02501783 -0.99917829  0.92684596]
 [-0.83617252 -0.546978   -0.04038029  2.5102115 ]
 [ 0.          0.          0.          1.        ]]
det 0.9999999017880502 1.0000000532257187 orth err 1.1600297067371201e-07
tr range [-0.23455592  0.89179453  2.50438344] [0.01194375 0.92816739 3.19861128]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import importlib.metadata as md
for n in ['opencv-python','opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless']:
 try: print(n,md.version(n))
 except Exception as e: print(n,'no')
PY"
```
<details><summary>output</summary>

```
opencv-python 4.11.0.86
opencv-python-headless no
opencv-contrib-python 4.11.0.86
opencv-contrib-python-headless no

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);rv=R.from_matrix(T[:,:3,:3]).as_rotvec();da=np.degrees(np.linalg.norm(np.diff(rv,axis=0),axis=1));
for i in np.argsort(dt)[-10:][::-1]:print('dt',i,dt[i],'da',da[i])
print('boundaries')
for i in [102,103,104,148,149,150,218,219,220,249,250,251,340,341,342,378,379,380]: print(i,dt[i-1] if i>0 else 0,da[i-1] if i>0 else 0)
PY"
```
<details><summary>output</summary>

```
dt 362 0.016461684602408575 da 0.5119771247910206
dt 361 0.016113574275742098 da 0.5541796309026652
dt 360 0.014821607161953335 da 0.575473892072292
dt 363 0.014555797760386213 da 0.5083481984855948
dt 236 0.014527999210268112 da 0.2503962554308507
dt 237 0.01422663570455381 da 0.2535043236353849
dt 353 0.013890829854915273 da 0.21026630180419886
dt 359 0.01387045835227441 da 0.5128663012610121
dt 235 0.013609887946699928 da 0.22848704483491686
dt 231 0.013497250376258147 da 0.1146959309251799
boundaries
102 0.0 0.0
103 0.0 0.0
104 0.0 2.8447745895630455e-14
148 0.0022791907010818475 0.11607698574138979
149 0.001837656625856051 0.09569849021482874
150 3.13573471632767e-08 6.264966157714216e-14
218 0.0 0.0
219 0.0 0.0
220 1.1004656964352382e-15 4.219482601449129e-14
249 0.006091597331699976 …[+328 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])[:3];n=n/np.linalg.norm(n)
for i in [0,103,150,219,251,341,380,466]:
 u=T[i,:3,:3]@np.array([0,0,1.]);print(i,'tilt deg',np.degrees(np.arccos(np.clip(u@n,-1,1))),'up',u)
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
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.asarray(m.vertices)
# weighted sample vertices + face centers? Use vertices
c=V.mean(0);w,v=np.linalg.eigh(np.cov((V-c).T));print('eig',w,'axes',v)
# components weighted face centroids
F=V[m.faces];pts=np.concatenate([V,F.mean(1)]);c=pts.mean(0);w,v=np.linalg.eigh(np.cov((pts-c).T));print('face eig',w,'axes',v,'long',v[:,-1])
S,T=trimesh.bounds.oriented_bounds(m);print('OBB T',T,'S',S)
PY"
```
<details><summary>output</summary>

```
eig [0.00580632 0.01743685 0.04278082] axes [[ 0.05708052 -0.99720541 -0.04819941]
 [-0.98722261 -0.06357158  0.14611699]
 [-0.14877276  0.03924312 -0.98809243]]
face eig [0.00580588 0.01743558 0.04278018] axes [[ 0.05708472 -0.9972054  -0.04819472]
 [-0.98722221 -0.0635751   0.14611814]
 [-0.14877378  0.03923778 -0.98809248]] long [-0.04819472  0.14611814 -0.98809248]
OBB T [0.25474448 0.38414898 0.9780487 ] S [[-2.08553391e-02 -9.99691155e-01 -1.35147631e-02  6.65933283e-03]
 [ 9.99782404e-01 -2.08473849e-02 -7.29178753e-04  5.75123052e-04]
 [ 4.47206081e-04 -1.35270296e-02  9.99908406e-01 -1.73162929e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/outputs');import solve,numpy as np,json,trimesh,h5py,cv2
camera=json.load(open('/work/inputs/camera.json'));rec=np.load('/work/inputs/reconstruction.npz');B=rec['T_world_mesh'].astype(float);R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];fm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mm=fm.copy();mm.merge_vertices();bv=np.asarray(mm.vertices,float);bf=np.asarray(mm.faces,np.int32);bf=bf[bv[bf][:,:,2].max(1)<=.23];rr=solve.Renderer(bv,bf,np.asarray(fm.vertices,float),np.asarray(fm.faces,np.int32),np.linalg.inv(np.asarray(camera['T_world_cam'],float)),np.asarray(camera['K'],float),1536,1152)
frames=[0,50,100,110,125,140,150,200,220,235,250,280,320,340,350,365,380,420,466]
mb=rr.hard_masks(B[frames],full=False,batch=4);mr=rr.hard_masks(R[frames],full=False,batch=4);cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
with h5py.File('/work/inputs/masks.h5','r') as h:gt=h['object'][frames]
for j,i in enumerate(frames):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();y,x=np.where(gt[j]);cx,cy=int(x.mean()),int(y.mean());x0,x1=max(0,cx-120),min(1536,cx+120);y0,y1=max(0,cy-160),min(1152,cy+160);c=im[y0:y1,x0:x1].copy()
 for m,col in [(gt[j],(0,255,0)),(mb[j],(0,0,255)),(mr[j],(255,0,0))]:
  cont,_=cv2.findContours(m[y0:y1,x0:x1].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(c,cont,-1,col,1)
 cv2.putText(c,f'{i}',(5,20) …[+251 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"    for stride in (16,):\\n        pass\\n\",\"\")
s=s.replace('f\"| OpenCV | {ver(\\'opencv-python-headless\\')} | yes | mask resizing and image I/O |\"','f\"| OpenCV | {ver(\\'opencv-python\\')} | yes | mask resizing and image I/O |\"')
s=s.replace('\"The static fits clearly improve pooled mask agreement and reduce jitter. The first push has a very small \"\\n              \"mask-IoU decrease (~0.2 percentage point in the final assembly), accepted because the endpoints improve \"\\n              \"and the trajectory becomes much smoother; no baseline motion was replaced with an embedding or hand-fitted \"\\n              \"per-frame table.\"','\"The static fits clearly improve pooled mask agreement and remove frame-to-frame pose jitter. Every interval\\'s \"\\n              \"mask IoU improves in the final trajectory. The middle push has a slightly larger median step because baseline \"\\n              \"depth drift was compressed while its two endpoints were restored, but the 99th-percentile jump is much \"\\n              \"smaller. No baseline motion was replaced with an embedding or hand-fitted per-frame table.\"')
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
grep -n \"OpenCV\\|static fits\\|stride in\" /work/outputs/solve.py"
```
<details><summary>output</summary>

```
77:    """Project mesh vertices.  Returns OpenCV image-convention clip coords."""
405:              "The static fits clearly improve pooled mask agreement and remove frame-to-frame pose jitter. Every interval's "
429:             f"| OpenCV | {ver('opencv-python')} | yes | mask resizing and image I/O |",

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
Fitting static plateau 0: 0-103
    pooled_mean      score=0.84590 bodyIoU=0.85434 fullIoU=0.81212
    free_dense       score=0.88746 bodyIoU=0.90046 fullIoU=0.83548
    free_coarse      score=0.88010 bodyIoU=0.89279 fullIoU=0.82933
    free_strided     score=0.89244 bodyIoU=0.90534 fullIoU=0.84083
    upright0         score=0.85269 bodyIoU=0.86337 fullIoU=0.80997
    upright_neg      score=0.85011 bodyIoU=0.86005 fullIoU=0.81031
  selected free_strided: [-0.05580341  0.89457631  3.19952745]
Fitting static plateau 1: 150-219
    pooled_mean      score=0.85853 bodyIoU=0.87113 fullIoU=0.80815
    free_dense       score=0.86401 bodyIoU=0.87709 fullIoU=0.81169
    free_coarse      score=0.86812 bodyIoU=0.88167 fullIoU=0.81390
    free_strided     score=0.86169 bodyIoU=0.87419 fullIoU=0.81170
  …[+3300 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /work/outputs/REPORT.md && echo '---' && cat /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
# Rolling-luggage trajectory recovery

## Method

The episode was split into four static plateaus separated by three left-foot pushes. For each plateau, the baseline poses were pooled to a single rigid pose and refined with a differentiable antialiased nvdiffrast silhouette fit over all sampled frames of the plateau. Multiple deterministic frame subsets and free/upright initialisations were scored on the full plateau; the complete textured mesh, including its asymmetric handle, was used as a held-out silhouette check. The handle faces were excluded only from the fitting silhouette because SAM2 omits the thin telescoping handle.

During pushes the method low-pass filters the shipped baseline motion and right-multiplies a local SE(3) correction that joins the two adjacent pooled poses exactl …[+3141 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,os,json
p='/work/outputs/object_poses_world.npz';z=np.load(p);T=z['T_world_mesh'];print('files',z.files);print('shape/dtype/finite',T.shape,T.dtype,np.isfinite(T).all());print('det',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max());print('size',os.path.getsize(p));
print('outputs',[(f,os.path.getsize('/work/outputs/'+f)) for f in os.listdir('/work/outputs')])
PY"
```
<details><summary>output</summary>

```
files ['T_world_mesh']
shape/dtype/finite (467, 4, 4) float64 True
det 0.9999998870281244 1.000000063398739
size 60054
outputs [('solve.py', 26720), ('__pycache__', 4096), ('object_poses_world.npz', 60054), ('REPORT.md', 3091), ('TOOLS.md', 846)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -l /work/outputs && sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py /work/outputs/REPORT.md /work/outputs/TOOLS.md'
```
<details><summary>output</summary>

```
total 100
-rw-rw-r-- 1 hanzhou hanzhou  3091 Oct 20 23:26 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   846 Oct 20 23:26 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 23:22 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 60054 Oct 20 23:25 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 26720 Oct 20 23:22 solve.py
4f9d6b351f51f018423b7af85605429d2b429f072db3f3fc80db9bf9b4c535b0  /work/outputs/object_poses_world.npz
e9af28000fd41f8bbea9db5354c3253c422257c66239888d734fa4e9ab14e5f7  /work/outputs/solve.py
9bef078ee60975ccb44c7c68d788abe55bd949896614786f9ec11f9b04ce8d16  /work/outputs/REPORT.md
c06177b2ee4237573cb1cdd14ff390c358a421103e38dde9b307a00544511b67  /work/outputs/TOOLS.md

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import shutil, os
p='/work/outputs/__pycache__'
if os.path.exists(p): shutil.rmtree(p)
PY
grep -nE '/work/scratch|/pipeline|https?://' /work/outputs/solve.py /work/outputs/REPORT.md /work/outputs/TOOLS.md || true
ls -la /work/outputs"
```
<details><summary>output</summary>

```
/work/outputs/solve.py:432:             "it does not need `/pipeline`, network access, scratch files, or downloaded weights."]
/work/outputs/TOOLS.md:17:The solution reads the released `/work/inputs` files and the shipped `/work/inputs/reconstruction.npz`; it does not need `/pipeline`, network access, scratch files, or downloaded weights.
total 104
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 23:26 .
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 22:51 ..
-rw-rw-r-- 1 hanzhou hanzhou  3091 Oct 20 23:26 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   846 Oct 20 23:26 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 60054 Oct 20 23:25 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 26720 Oct 20 23:22 solve.py

```
</details>

**Agent:** Completed the trajectory recovery task. Required outputs are present:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(467, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, confidence, and validation
- `/work/outputs/TOOLS.md` — tool versions and dependencies

Final validation without ground truth showed improved overall body-mask IoU from **0.8771 to 0.9045**, with 99th-percentile translation jitter reduced from **44.2 mm to 14.2 mm**. The final solver run completed in **3.27 minutes**.
