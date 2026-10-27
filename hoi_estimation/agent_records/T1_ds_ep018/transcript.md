# T1_ds_ep018

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T17:40:33Z

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
/usr/bin/bash -c "sed -n '1,90p' /work/AGENT_TASK.md"
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
/usr/bin/bash -c "sed -n '91,180p' /work/AGENT_TASK.md"
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
/usr/bin/bash -c "find /work/inputs -maxdepth 1 -type f -printf '%f %s bytes\\n'; ls -la /work; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"
```
<details><summary>output</summary>

```
video.mp4 25867689 bytes
object_mesh.glb 16235908 bytes
ground_plane.json 399 bytes
task.txt 72 bytes
episode.json 69 bytes
camera.json 826 bytes
reconstruction.npz 43106909 bytes
frame0_reference_human.npz 192433 bytes
masks.h5 8446371 bytes
total 40
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:34 .
drwxr-xr-x 1 hanzhou hanzhou   200 Oct 20 21:40 ..
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:34 .codex_home
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:34 .net
-rw-rw-r-- 1 hanzhou hanzhou 10592 Oct 20 21:34 AGENT_TASK.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 17:53 inputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:34 logs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:34 outputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:34 scratch
NVIDIA RTX 6000 Ada Generation, 49140 MiB, 3236 MiB
 …[+53 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/inputs/task.txt; echo; cat /work/inputs/episode.json; echo; cat /work/inputs/camera.json; echo; cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk over to the blue trash can and slide the blue trash can side ways.

{
 "episode_index": 18,
 "object": "blue_trash_can",
 "frames": 858
}
{
 "K": [
  [
   820.00048828125,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   820.00048828125,
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
   843.0822143554688,
   0.0,
   768.0
  ],
  [
   0.0,
   843.0822143554688,
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
   0.9950919278405133,
   -0.008288567623105656,
   0.09860707273521717,
   -0.18020094592562447
  ],
  [
   0.008564690720473247,
   0.9999604965795403,
   -0.0023772575085875824,
   0.10260798737647048
  ],
  [
   -0.09858347335894513,
   0.0032101288380220905,
   0.9951236073238992,
   0.06627208061368874
  ],
  [
   0.0,
   0.0,
    …[+569 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100; echo '---'; find /work/outputs -maxdepth 1 -type f -ls"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1971
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2131
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/depth_report.json 1816
/pipeline/depth/moge2/aligned.h5 1056822936
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 91355
/pipeline/depth/moge2/raw.npy 3036414080
/pipeline/export/episode_000018/edex 563
/pipeline/export/episode_000018/wild_export.json 2808
/pipeline/foundationpose/episode_000018.pkl 244709
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 656207 …[+553 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, trimesh
r=np.load('/work/inputs/reconstruction.npz')
print('recon keys/shapes/dtypes')
for k in r.files:
 print(k, r[k].shape, r[k].dtype)
with h5py.File('/work/inputs/masks.h5') as f:
 print('masks')
 for k in f: print(k, f[k].shape, f[k].dtype, 'summean', f[k][:].sum(axis=(1,2)).mean())
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
print('mesh geom type',type(m), 'bounds',m.bounds,'extents',m.extents,'verts',len(m.vertices),'faces',len(m.faces),'visual',type(m.visual))
print('T0',r['T_world_mesh'][0]); print('F',len(r['T_world_mesh']))
PY"
```
<details><summary>output</summary>

```
recon keys/shapes/dtypes
T_world_mesh (858, 4, 4) float32
T_cam_mesh_foundationpose (858, 4, 4) float32
T_cam_mesh_coconet (858, 4, 4) float32
human_vertices_world (858, 4610, 3) float32
human_keypoints_world (858, 70, 3) float32
contact_prob_right_left (858, 2) float32
masks
object (858, 1152, 1536) bool summean 35589.91958041958
person (858, 1152, 1536) bool summean 53125.45221445221
mesh geom type <class 'trimesh.base.Trimesh'> bounds [[-0.19071645 -0.1941925  -0.28331217]
 [ 0.18710606  0.17963576  0.42995518]] extents [0.3778225  0.37382826 0.71326736] verts 255564 faces 85217 visual <class 'trimesh.visual.texture.TextureVisuals'>
T0 [[ 0.46281838 -0.8863954   0.01024711 -0.07775131]
 [ 0.06089677  0.04332434  0.99719995  0.80470306]
 [-0.88436085 -0.46089962  0.07402965  2.4799156 ]
 …[+58 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/analysis /work/outputs; python - <<'PY'
import numpy as np, cv2, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh']; c=r['contact_prob_right_left'];
# poses translation, rotation relative frame 0, deltas
tr=T[:,:3,3]
dtr=np.linalg.norm(np.diff(tr,axis=0),axis=1)
def relR(A,B): return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))
dro=np.array([relR(T[i],T[i+1]) for i in range(len(T)-1)])
for i in range(0,len(T),25):
 print(i,'tr',np.round(tr[i],3),'contact',np.round(c[i],2),'dtr',round(dtr[i] if i<len(dtr) else 0,4),'dro',round(dro[i] if i<len(dro) else 0,2))
print('trans ranges',tr.min(0),tr.max(0),'path',dtr.sum())
print('contact frames >.5 right',np.where(c[:,0]>.5)[0][[0,-1]] if (c[:,0]>.5).any() else None,'left',np.where(c[:,1]>.5)[0][[0,-1]] if (c[:,1]>.5).any() else None)
print('large jumps')
for i in np.where((dtr>.05)|(dro>20))[0]: print(i+1,'dtr',dtr[i],'dro',dro[i],'contact',c[i+1])
PY"
```
<details><summary>output</summary>

```
0 tr [-0.078  0.805  2.48 ] contact [0. 0.] dtr 0.0008 dro 0.06
25 tr [-0.077  0.809  2.494] contact [0. 0.] dtr 0.0033 dro 0.06
50 tr [-0.077  0.809  2.494] contact [0. 0.] dtr 1e-04 dro 0.06
75 tr [-0.076  0.814  2.512] contact [0. 0.] dtr 0.0 dro 0.06
100 tr [-0.076  0.814  2.513] contact [0. 0.] dtr 0.0002 dro 0.06
125 tr [-0.077  0.807  2.487] contact [0. 0.] dtr 0.0004 dro 0.06
150 tr [-0.078  0.806  2.483] contact [0. 0.] dtr 0.0003 dro 0.06
175 tr [-0.078  0.804  2.476] contact [0. 0.] dtr 0.0 dro 0.06
200 tr [-0.078  0.807  2.489] contact [0. 0.] dtr 0.0 dro 0.13
225 tr [-0.078  0.805  2.483] contact [0.94 0.77] dtr 1e-04 dro 0.13
250 tr [-0.084  0.804  2.482] contact [1. 1.] dtr 0.0005 dro 0.35
275 tr [-0.103  0.806  2.485] contact [1. 1.] dtr 0.0018 dro 0.62
300 tr [-0.146  0.67 …[+1641 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames,r_frame_rate,duration,width,height -of default=nw=1 /work/inputs/video.mp4; python - <<'PY'
import cv2, os
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,100,200,300,350,400,450,500,550,600,650,700,750,800,857]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 if ok: cv2.imwrite(f'/work/scratch/analysis/f{i:03d}.jpg',im)
PY
python - <<'PY'
import cv2, numpy as np
ims=[]
for i in [0,100,200,300,350,400,450,500,550,600,650,700,750,800,857]:
 im=cv2.imread(f'/work/scratch/analysis/f{i:03d}.jpg'); im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(10,30),0,1,(0,0,255),2); ims.append(im)
while len(ims)%4: ims.append(np.zeros_like(ims[0]))
out=np.vstack([np.hstack(ims[i:i+4]) for i in range(0,len(ims),4)])
cv2.imwrite('/work/scratch/analysis/contact_sheet.jpg',out)
PY"
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=28.600000
nb_frames=858

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/pipeline_report.json; echo '---POSE'; cat /pipeline/.stages/05_object_pose.json; echo '---COCO'; cat /pipeline/.stages/06_coconet.json; echo '---REFINE'; cat /pipeline/.stages/07_refinement.json"
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
    "mtime_ns": 1791232589968696004,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000018/inference/episode_000018_step200000_before_after.mp4",
    "size": 63679784
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+11885 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'];
# relative transform to start/static
for i in range(0,858,25):
 R=T[i,:3,:3]; t=T[i,:3,3]
 # axis-angle relative to T0
 Rrel=T[0,:3,:3].T@R
 ang=np.degrees(np.arccos(np.clip((np.trace(Rrel)-1)/2,-1,1)))
 ax=cv2=None
 print(i, 'pos', np.round(t,3), 'relrot',round(ang,2), 'Rdiag',np.round(np.diag(R),3))
PY"
```
<details><summary>output</summary>

```
0 pos [-0.078  0.805  2.48 ] relrot 0.06 Rdiag [0.463 0.043 0.074]
25 pos [-0.077  0.809  2.494] relrot 0.06 Rdiag [0.463 0.043 0.074]
50 pos [-0.077  0.809  2.494] relrot 0.06 Rdiag [0.463 0.043 0.074]
75 pos [-0.076  0.814  2.512] relrot 0.06 Rdiag [0.463 0.043 0.074]
100 pos [-0.076  0.814  2.513] relrot 0.06 Rdiag [0.463 0.043 0.074]
125 pos [-0.077  0.807  2.487] relrot 0.06 Rdiag [0.463 0.043 0.074]
150 pos [-0.078  0.806  2.483] relrot 0.06 Rdiag [0.463 0.043 0.074]
175 pos [-0.078  0.804  2.476] relrot 0.06 Rdiag [0.463 0.043 0.074]
200 pos [-0.078  0.807  2.489] relrot 0.44 Rdiag [0.457 0.046 0.078]
225 pos [-0.078  0.805  2.483] relrot 0.44 Rdiag [0.457 0.046 0.078]
250 pos [-0.084  0.804  2.482] relrot 0.97 Rdiag [0.452 0.054 0.079]
275 pos [-0.103  0.806  2.485] relrot 2.26 Rdi …[+1573 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['torch','nvdiffrast','trimesh','open3d','cv2','scipy','skimage','h5py','pyrender','pytorch3d','kornia']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'NO',type(e).__name__,str(e)[:120])
PY
pip list --format=freeze | grep -Ei 'torch|nvdiff|trimesh|open3d|opencv|scipy|scikit|h5py|pyrender|pytorch3d|kornia|roma|lie'"
```
<details><summary>output</summary>

```
torch OK 2.5.1+cu124
nvdiffrast OK 0.4.0
trimesh OK 5.1.1
open3d OK 0.20.0
cv2 OK 4.11.0
scipy OK 1.16.3
skimage OK 0.26.0
h5py OK 3.16.0
pyrender OK 0.1.45
pytorch3d OK 0.7.9
kornia OK 0.8.3
gradio_client==2.7.2
h5py==3.16.0
jupyter_client==7.4.9
kornia==0.8.3
kornia_rs==0.2.0
lief==0.14.1
nvdiffrast==0.4.0
open3d==0.20.0
opencv-contrib-python==4.11.0.86
opencv-python==4.11.0.86
pyrender==0.1.45
pytorch-lightning==2.6.6
pytorch3d==0.7.9
roma==1.6.1
scikit-image==0.26.0
scikit-learn==1.9.1
scipy==1.16.3
torch==2.5.1+cu124
torchaudio==2.5.1+cu124
torchelastic==0.2.2
torchmetrics==1.9.0
torchvision==0.20.1+cu124
trimesh==5.1.1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual
print(v)
for a in dir(v):
 if not a.startswith('_'):
  try:
   x=getattr(v,a)
   if not callable(x): print(a, type(x), getattr(x,'shape',''), str(x)[:100])
  except: pass
PY"
```
<details><summary>output</summary>

```
<trimesh.visual.texture.TextureVisuals object at 0x75f446f16750>
defined <class 'bool'>  True
face_materials <class 'NoneType'>  None
kind <class 'str'>  texture
material <class 'trimesh.visual.material.PBRMaterial'>  <trimesh.visual.material.PBRMaterial object at 0x75f446f15710>
mesh <class 'trimesh.base.Trimesh'>  <trimesh.Trimesh(vertices.shape=(255564, 3), faces.shape=(85217, 3))>
uv <class 'trimesh.caching.TrackedArray'> (255564, 2) [[0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 ...
 [0.41734797 0.90103
vertex_attributes <class 'trimesh.caching.DataStore'>  <trimesh.caching.DataStore object at 0x75f446f166d0>

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mt=m.visual.material
print(type(mt)); print(mt)
for a in ['baseColorTexture','baseColorFactor','image','metallicFactor','roughnessFactor']:
 try:
  x=getattr(mt,a); print(a,type(x),getattr(x,'shape',None),getattr(x,'size',None),str(x)[:200])
 except Exception as e: print(a,e)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.material.PBRMaterial'>
<trimesh.visual.material.PBRMaterial object at 0x71fdcf511d10>
baseColorTexture <class 'PIL.Image.Image'> None (5007, 4574) <PIL.Image.Image image mode=RGB size=5007x4574 at 0x71FE0DFC3250>
baseColorFactor <class 'NoneType'> None None None
image 'PBRMaterial' object has no attribute 'image'
metallicFactor <class 'float'> None None 0.0
roughnessFactor <class 'NoneType'> None None None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_check.py <<'PY'
import numpy as np, torch, trimesh, h5py, cv2, json, nvdiffrast.torch as dr, os, sys, time
D='/work/inputs'
cam=json.load(open(D+'/camera.json'))
K=np.array(cam['K'],np.float64); Twc=np.array(cam['T_world_cam'],np.float64)
Twc_inv=np.linalg.inv(Twc)
r=np.load(D+'/reconstruction.npz'); T=r['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
gl=dr.RasterizeCudaContext(device='cuda')
H,W=cam['height'],cam['width']
cap=cv2.VideoCapture(D+'/video.mp4')
with h5py.File(D+'/masks.h5') as hf:
 frames=[]
 for fi in [0,100,200,250,300,350,400,450,500,550,600,650,700,725,750,800,857]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ok,im=cap.read(); om=hf['object'][fi]; pm=hf['person'][fi]
  # render
  Tc2m=Twc_inv@T[fi].astype(np.float64)
  vh=np.concatenate([m.vertices,np.ones((len(V),1))],1)
  vc=(Tc2m@vh.T).T[:,:3]
  u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]; v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]
  xn=2*u/W-1; yn=1-2*v/H
  z=vc[:,2]; zn=(z-0.1)/(10-0.1)*2-1
  pos=np.stack([xn*zn,yn*zn,zn,np.ones_like(zn)],1) # not actual perspective? w should z
  pos[:,:3]=np.stack([xn,yn,zn],1); pos[:,3]=1
  pos=torch.tensor(pos,dtype=torch.float32,device='cuda')[None]
  rast,_=dr.rasterize(gl,pos,F,resolution=(H,W))
  mask=(rast[0,...,3]>0).cpu().numpy()
  for flip in [False,True]:
   mm=mask[::- …[+376 chars]
```
<details><summary>output</summary>

```
0 flip False iou 0.0 mask 38786 obj 37054 person 31043
0 flip True iou 0.9418271200327735 mask 38786 obj 37054 person 31043
100 flip False iou 0.0 mask 37731 obj 36860 person 61187
100 flip True iou 0.9548444584217837 mask 37731 obj 36860 person 61187
200 flip False iou 0.0 mask 38430 obj 35548 person 60839
200 flip True iou 0.9149409815696832 mask 38430 obj 35548 person 60839
250 flip False iou 0.0 mask 38545 obj 34870 person 66924
250 flip True iou 0.8963424084310585 mask 38545 obj 34870 person 66924
300 flip False iou 0.0 mask 34968 obj 28770 person 60831
300 flip True iou 0.8208776139869729 mask 34968 obj 28770 person 60831
350 flip False iou 0.0 mask 39808 obj 34049 person 63154
350 flip True iou 0.8154265909593688 mask 39808 obj 34049 person 63154
400 flip False iou 0.0 mask 33439 ob …[+1370 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/texture_check.py <<'PY'
import numpy as np, torch, trimesh, h5py, cv2, json, nvdiffrast.torch as dr
D='/work/inputs'; cam=json.load(open(D+'/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Ti=np.linalg.inv(Twc)
r=np.load(D+'/reconstruction.npz'); T=r['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(m.faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
imgs={}
for flipv in [False,True]:
 tex=np.asarray(m.visual.material.baseColorTexture).astype(np.float32)/255.
 if flipv: tex=tex[::-1]
 tex=torch.tensor(tex,device='cuda')[None]
 imgs[flipv]=tex
H,W=cam['height'],cam['width']; gl=dr.RasterizeCudaContext(device='cuda')
cap=cv2.VideoCapture(D+'/video.mp4')
with h5py.File(D+'/masks.h5') as hf:
 for fi in [0,100,200,250,300,350,400,450,500,550,600,650,700,725,750,800,857]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ok,im=cap.read(); om=hf['object'][fi]; pm=hf['person'][fi]
  Tc2m=Ti@T[fi].astype(np.float32); vh=np.c_[m.vertices,np.ones(len(V))]; vc=(Tc2m@vh.T).T[:,:3]
  u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]; v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]
  xn=2*u/W-1; yn=1-2*v/H; zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1)
  pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(V))],dtype=torch.float32,device='cuda')[None]
  rast,_=dr.rasterize(gl, …[+850 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/texture_check.py", line 10, in <module>
    tex=torch.tensor(tex,device='cuda')[None]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/tex=tex\\[::-1\\]/tex=tex[::-1].copy()/\" /work/scratch/analysis/texture_check.py; python /work/scratch/analysis/texture_check.py"
```
<details><summary>output</summary>

```
0 region 37053 out [(False, 48.52396117273455, 0.020419310197369703), (True, 66.24794213693897, 0.24984139119071358)]
100 region 36860 out [(False, 47.97595406040875, 0.03813198471895613), (True, 68.31191897268945, 0.2873726849126157)]
200 region 35548 out [(False, 48.78474175762349, 0.027243053797120626), (True, 71.72195341510071, 0.4020666084366792)]
250 region 34791 out [(False, 48.25303478869056, 0.002384201312272823), (True, 71.42440094660496, 0.3827867937320645)]
300 region 28754 out [(False, 49.69392084579537, -0.012118520932469465), (True, 74.6309151190559, 0.4249381886767556)]
350 region 34034 out [(False, 47.670858553211495, -0.011046150468412102), (True, 72.85522320816439, 0.24980861190429057)]
400 region 29176 out [(False, 47.62567407001188, 0.03215061629182753), (True, 78.0229 …[+1231 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); im=np.asarray(m.visual.material.baseColorTexture); print(im.shape, im.reshape(-1,3).mean(0), np.percentile(im.reshape(-1,3),[1,10,50,90,99],axis=0)); print(m.visual.material.emissiveTexture, m.visual.material.metallicRoughnessTexture)
PY"
```
<details><summary>output</summary>

```
(4574, 5007, 3) [28.20773872 39.20502818 51.3176962 ] [[  0.   0.   0.]
 [  0.   0.   0.]
 [  0.   0.   0.]
 [ 87. 118. 155.]
 [156. 155. 217.]]
None None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); uv=np.asarray(m.visual.uv); print(uv.min(0),uv.max(0),np.percentile(uv,[1,50,99],axis=0),uv[:10])
PY"
```
<details><summary>output</summary>

```
[0.00086537 0.00080532] [0.99870825 0.99755847] [[0.05696234 0.03023386]
 [0.56360042 0.42167902]
 [0.96454215 0.98164076]] [[0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 [0.16097389 0.3726449 ]
 [0.17756665 0.37791884]
 [0.17756665 0.37791884]
 [0.17756665 0.37791884]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/texture_check.py'
s=open(p).read()
s=s.replace(\"tex=np.asarray(m.visual.material.baseColorTexture).astype(np.float32)/255.\\n if flipv: tex=tex[::-1].copy()\\n tex=torch.tensor(tex,device='cuda')[None]\", \"tex=np.asarray(m.visual.material.baseColorTexture).astype(np.float32)/255.\\n if flipv: tex=tex[::-1].copy()\\n tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2)\\n tex=torch.tensor(tex,device='cuda')[None]\")
s=s.replace(\"col=dr.texture(tex,uv,filter_mode='linear')[0].cpu().numpy()[::-1]\", \"col=dr.texture(tex,uv,filter_mode='linear')[0,...,:3].cpu().numpy()[::-1]\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/texture_check.py"
```
<details><summary>output</summary>

```
0 region 37053 out [(False, 48.52396117273455, 0.020419310197369703), (True, 66.24794213693897, 0.24984139119071358)]
100 region 36860 out [(False, 47.97595406040875, 0.03813198471895613), (True, 68.31191897268945, 0.2873726849126157)]
200 region 35548 out [(False, 48.78474175762349, 0.027243053797120626), (True, 71.72195341510071, 0.4020666084366792)]
250 region 34791 out [(False, 48.25303478869056, 0.002384201312272823), (True, 71.42440094660496, 0.3827867937320645)]
300 region 28754 out [(False, 49.69392084579537, -0.012118520932469465), (True, 74.6309151190559, 0.4249381886767556)]
350 region 34034 out [(False, 47.670858553211495, -0.011046150468412102), (True, 72.85522320816439, 0.24980861190429057)]
400 region 29176 out [(False, 47.62567407001188, 0.03215061629182753), (True, 78.0229 …[+1231 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# modify to save raw texture exact quickly by re-run one; easiest create variant script from current
p='/work/scratch/analysis/texture_check.py'; s=open(p).read(); s=s.replace(\"for fi in [0,100,200,250,300,350,400,450,500,550,600,650,700,725,750,800,857]:\",\"for fi in [0]:\")
s=s.replace(\"vis=im.copy(); vis[mask]=rgb[mask]; vis=cv2.addWeighted(im,.35,vis,.65,0); cv2.imwrite\", \"vis=np.zeros_like(im); vis[mask]=rgb[mask]; cv2.imwrite\")
open('/work/scratch/analysis/texture_raw.py','w').write(s)
PY
python /work/scratch/analysis/texture_raw.py"
```
<details><summary>output</summary>

```
0 region 37053 out [(False, 48.52396117273455, 0.020419310197369703), (True, 66.24794213693897, 0.24984139119071358)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/texture_check.py'; s=open(p).read(); s=s.replace(\"col=dr.texture(tex,uv,filter_mode='linear')[0,...,:3].cpu().numpy()[::-1]\\n   rgb=(col*255).clip(0,255).astype(np.uint8)\", \"col=dr.texture(tex,uv,filter_mode='linear')[0,...,:3].cpu().numpy()[::-1]\\n   rgb=(col[...,::-1]*255).clip(0,255).astype(np.uint8)\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/texture_check.py"
```
<details><summary>output</summary>

```
0 region 37053 out [(False, 47.21124695256344, 0.020419310197369703), (True, 65.43041049307749, 0.24984139119071358)]
100 region 36860 out [(False, 46.84621993127148, 0.03813198471895613), (True, 67.61817688551275, 0.2873726849126157)]
200 region 35548 out [(False, 47.85203105659953, 0.027243053797120626), (True, 71.1249202955628, 0.4020666084366792)]
250 region 34791 out [(False, 47.26285533615015, 0.002384201312272823), (True, 70.82236785375528, 0.3827867937320645)]
300 region 28754 out [(False, 48.717523359068885, -0.012118520932469465), (True, 73.99466740859243, 0.4249381886767556)]
350 region 34034 out [(False, 46.80251121427592, -0.011046150468412102), (True, 72.16436504671799, 0.24980861190429057)]
400 region 29176 out [(False, 46.57481034640344, 0.03215061629182753), (True, 77.4867 …[+1231 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
r=np.load('/work/inputs/reconstruction.npz'); cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
B=r['T_world_mesh']; Fp=r['T_cam_mesh_foundationpose']; Co=r['T_cam_mesh_coconet']
def ang(A,B):
 return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))
def trans(A,B): return np.linalg.norm(A[:3,3]-B[:3,3])
for i in range(0,858,25):
 Wf=Twc@Fp[i]; Wc=Twc@Co[i]
 print(i,'B-Fp rot %.2f tr %.3f'%(ang(B[i],Wf),trans(B[i],Wf)),'B-Co rot %.2f tr %.3f'%(ang(B[i],Wc),trans(B[i],Wc)),'Fp-Co %.2f %.3f'%(ang(Wf,Wc),trans(Wf,Wc)))
PY"
```
<details><summary>output</summary>

```
0 B-Fp rot 5.61 tr 0.247 B-Co rot 1.52 tr 0.118 Fp-Co 4.87 0.136
25 B-Fp rot 2.03 tr 0.221 B-Co rot 1.52 tr 0.106 Fp-Co 0.99 0.127
50 B-Fp rot 2.39 tr 0.149 B-Co rot 1.52 tr 0.106 Fp-Co 1.24 0.050
75 B-Fp rot 2.77 tr 0.158 B-Co rot 1.52 tr 0.092 Fp-Co 1.38 0.078
100 B-Fp rot 2.94 tr 0.095 B-Co rot 1.52 tr 0.092 Fp-Co 1.57 0.008
125 B-Fp rot 2.78 tr 0.151 B-Co rot 1.52 tr 0.112 Fp-Co 1.30 0.046
150 B-Fp rot 2.94 tr 0.122 B-Co rot 1.52 tr 0.115 Fp-Co 1.51 0.011
175 B-Fp rot 2.84 tr 0.188 B-Co rot 1.52 tr 0.121 Fp-Co 1.47 0.074
200 B-Fp rot 3.25 tr 0.127 B-Co rot 1.52 tr 0.120 Fp-Co 2.01 0.012
225 B-Fp rot 3.11 tr 0.158 B-Co rot 1.52 tr 0.125 Fp-Co 1.74 0.038
250 B-Fp rot 3.30 tr 0.149 B-Co rot 1.53 tr 0.121 Fp-Co 1.84 0.031
275 B-Fp rot 3.94 tr 0.145 B-Co rot 1.53 tr 0.119 Fp-Co 2.45 0.030
3 …[+1540 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/.stages/04_depth_alignment.json; cat /pipeline/depth/moge2/depth_report.json'
```
<details><summary>output</summary>

```
{
  "name": "04_depth_alignment",
  "outputs": [
    {
      "mtime_ns": 1791228578725711306,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000018/depth/moge2/aligned.h5",
      "size": 1056822936
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/align_depth_to_mhr_wild.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000018/export/episode_000018",
      "--raw-depth",
      "/home/hanzhou/Projects/v2d/run/out/episode_000018/depth/moge2/raw.npy",
      "--depth-report",
      "/home/hanzhou/Projects/v2d/run/out/episode_000018/depth/moge2/depth_report.json",
      "--mhr-init",
      "/home/hanzhou/Projects/v2d/run/out/episode_000018/mhr_init/episode_000018.pkl",
      …[+2864 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, pprint
for p in ['/pipeline/foundationpose/episode_000018.pkl','/pipeline/mhr_init/episode_000018.pkl','/pipeline/inference/refinement_checkpoint.pth']:
 print('\\n',p)
 try:
  x=pickle.load(open(p,'rb'))
  print(type(x))
  if isinstance(x,dict):
   for k,v in x.items(): print(k,type(v),getattr(v,'shape',None),str(v)[:200])
  else: print(str(x)[:1000])
 except Exception as e: print('ERR',e)
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
frames <class 'list'> None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019',
camera_id <class 'int'> None 0
fp_poses_cam <class 'numpy.ndarray'> (858, 1, 4, 4) [[[[ 0.7820811  -0.6228696  -0.01955353 -0.17483196]
   [-0.08201327 -0.07177109 -0.9940435   0.76183546]
   [ 0.6177561   0.7790265  -0.1072143   2.6657727 ]
   [ 0.          0.          0.          
fp_candidate_count <class 'numpy.ndarray'> (858,) [252   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
   1   1   1   1   1   1   1   1   1   1   1   1   1  
fp_selected_candidate_ …[+10745 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/depth/moge2/raw.npy','/pipeline/work/masks_k0.h5']:
 print('\\n',p)
 if p.endswith('.h5'):
  with h5py.File(p,'r') as f:
   def pr(name,obj):
    if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype,obj.attrs)
   f.visititems(pr)
 else:
  a=np.load(p,mmap_mode='r'); print(a.shape,a.dtype)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
aligned/front_stereo_camera_left (858,) object <Attributes of HDF5 object at 124879802430576>
alignment/front_stereo_camera_left/scale (858,) float32 <Attributes of HDF5 object at 124879802424976>
alignment/front_stereo_camera_left/shift (858,) float32 <Attributes of HDF5 object at 124879802424976>
alignment/front_stereo_camera_left/valid_count (858,) int32 <Attributes of HDF5 object at 124879802424976>
frame_names/front_stereo_camera_left (858,) object <Attributes of HDF5 object at 124879802424976>
raw/front_stereo_camera_left (858,) object <Attributes of HDF5 object at 124879802430576>

 /pipeline/depth/moge2/raw.npy
(858, 1152, 1536) uint16

 /pipeline/work/masks_k0.h5
episode_000018/000000-k0.obj_rend_mask.png (1152, 1536) uint8 <Attributes of HDF5 ob …[+179782 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/pipeline/foundationpose/episode_000018.pkl','rb'))
for k,v in x.items():
 if k in ('frames','metadata'): print(k, type(v), v if k=='metadata' else f'list {len(v)}'); continue
 a=np.asarray(v) if not isinstance(v,(str,int,float)) else v
 print(k,type(v),getattr(a,'shape',None),getattr(a,'dtype',None), str(a)[:160].replace('\\n',' '))
r=np.load('/work/inputs/reconstruction.npz'); B=r['T_world_mesh']
if 'obj_pose_world' in x:
 A=x['obj_pose_world'];
 def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))
 for i in [0,100,200,250,300,350,400,450,500,600,700,800,857]:
  print(i,'obj_pose_world vs B',ang(A[i],B[i]),np.linalg.norm(A[i,:3,3]-B[i,:3,3]))
PY"
```
<details><summary>output</summary>

```
fp_tracking_state <class 'numpy.ndarray'> (858,) uint8 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 0 0 0
fp_confirmed_pose_source <class 'numpy.ndarray'> (858,) uint8 [6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7  7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7  7 7 7 7 7
fp_pending_pose_source <class 'numpy.ndarray'> (858,) uint8 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 0 0 0
fp_emitted_pose_source <class 'numpy.ndarray'> (858,) uint8 [6 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7  7 7  …[+3200 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np,json,trimesh
x=pickle.load(open('/pipeline/foundationpose/episode_000018.pkl','rb')); cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
r=np.load('/work/inputs/reconstruction.npz'); B=r['T_world_mesh']; Fp=r['T_cam_mesh_foundationpose']
A=x['obj_pose_world']; P=x['fp_poses_cam'][:,0]
def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))
for i in [0,1,100,400,857]:
 print('\\ni',i)
 print('A obj_world\\n',A[i]); print('Twc@P\\n',Twc@P[i]); print('Fp\\n',Fp[i]); print('B\\n',B[i])
 print('A-TwcP',ang(A[i],Twc@P[i]),np.linalg.norm(A[i,:3,3]-(Twc@P[i])[:3,3]))
 print('A-Fp',ang(A[i],Fp[i]),np.linalg.norm(A[i,:3,3]-Fp[i,:3,3]))
PY"
```
<details><summary>output</summary>

```

i 0
A obj_world
 [[ 0.7820811  -0.6228696  -0.01955353 -0.17483196]
 [-0.08201327 -0.07177109 -0.9940435   0.76183546]
 [ 0.6177561   0.7790265  -0.1072143   2.6657727 ]
 [ 0.          0.          0.          1.        ]]
Twc@P
 [[ 0.83983751 -0.54240012 -0.02179045 -0.0976253 ]
 [-0.07678031 -0.07895488 -0.99391685  0.85657874]
 [ 0.53738016  0.83640192 -0.10795484  2.73872653]
 [ 0.          0.          0.          1.        ]]
Fp
 [[ 0.6227401  -0.78208244  0.02327256 -0.17986135]
 [ 0.06758539  0.08340061  0.9942215   0.68966097]
 [-0.7795043  -0.6175687   0.10479417  2.6513839 ]
 [ 0.          0.          0.          1.        ]]
B
 [[ 0.46281838 -0.8863954   0.01024711 -0.07775131]
 [ 0.06089677  0.04332434  0.99719995  0.80470306]
 [-0.88436085 -0.46089962  0.07402965  2.4799156 ]
 …[+3859 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/hypothesis_check.py <<'PY'
import numpy as np, torch, trimesh, h5py, cv2, json, nvdiffrast.torch as dr
D='/work/inputs'; cam=json.load(open(D+'/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Ti=np.linalg.inv(Twc)
r=np.load(D+'/reconstruction.npz'); B=r['T_world_mesh']; Co=r['T_cam_mesh_coconet']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(m.faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.; tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2); tex=torch.tensor(tex,device='cuda')[None]
H,W=cam['height'],cam['width']; gl=dr.RasterizeCudaContext(device='cuda'); C=np.array(m.bounds).mean(0)
def flip(axis):
 R=np.eye(4); R[:3,:3]=np.diag([-1.,-1.,-1.]); R[axis,axis]=1.
 A=np.eye(4); A[:3,3]=C; Ai=np.eye(4); Ai[:3,3]=-C
 return A@R@Ai
def render(T):
 Tc2m=Ti@T.astype(np.float32); vh=np.c_[m.vertices,np.ones(len(V))]; vc=(Tc2m@vh.T).T[:,:3]
 u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]; v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]
 xn=2*u/W-1; yn=1-2*v/H; zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1)
 pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(V))],dtype=torch.float32,device='cuda')[None]
 rast,_=dr.rasterize(gl,pos,F,resolution=(H,W)); uv,_=dr.interpolate(UV[No …[+1486 chars]
```
<details><summary>output</summary>

```

FRAME 0
B [ 0.9418  0.2813 -0.0506]
Co [0.7884 0.1546 0.0273]
Bflip0 [ 0.8619 -0.1543 -0.0113]
Bflip1 [ 0.9036 -0.2235  0.0059]
Bflip2 [0.8996 0.0767 0.0142]

FRAME 100
B [ 0.9548  0.3189 -0.1088]
Co [0.7769 0.1737 0.0331]
Bflip0 [ 0.8806 -0.0515 -0.0136]
Bflip1 [ 0.9168 -0.1564  0.0026]
Bflip2 [0.907  0.1955 0.0256]

FRAME 200
B [ 0.9149  0.4306 -0.2188]
Co [0.7367 0.1919 0.0337]
Bflip0 [ 0.8495  0.1207 -0.0026]
Bflip1 [0.8809 0.0083 0.011 ]
Bflip2 [0.8705 0.2495 0.0075]

FRAME 250
B [ 0.8963  0.4185 -0.2183]
Co [0.7127 0.1703 0.0375]
Bflip0 [8.423e-01 1.056e-01 8.000e-04]
Bflip1 [ 0.868  -0.0311  0.0112]
Bflip2 [0.8543 0.1969 0.0253]

FRAME 300
B [0.8209 0.448  0.0021]
Co [ 0.6649  0.1643 -0.0235]
Bflip0 [ 0.7828  0.1797 -0.0117]
Bflip1 [0.7979 0.0978 0.0062]
Bflip2 [0.8258 0.2539 0.008 …[+1917 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4'); ids=list(range(680,858,10)); ims=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in ids:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); om=f['object'][i]; pm=f['person'][i]
  ys,xs=np.where(om)
  x0=max(0,xs.min()-80);x1=min(1536,xs.max()+80);y0=max(0,ys.min()-80);y1=min(1152,ys.max()+80)
  crop=im[y0:y1,x0:x1].copy(); crop=cv2.resize(crop,(240,300)); cv2.putText(crop,str(i),(8,25),0,.7,(0,0,255),2); ims.append(crop)
while len(ims)%6: ims.append(np.zeros_like(ims[0]))
out=np.vstack([np.hstack(ims[i:i+6]) for i in range(0,len(ims),6)]); cv2.imwrite('/work/scratch/analysis/final_phase.jpg',out)
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
with h5py.File('/work/inputs/masks.h5','r') as f:
 obj=f['object'][:]; per=f['person'][:]
for i in list(range(0,858,20))+list(range(720,858,5)):
 ys,xs=np.where(obj[i]);
 print(f'{i:3d} area {len(xs):5d} cen {xs.mean():7.2f} {ys.mean():7.2f} box {xs.min():4d},{ys.min():4d},{xs.max():4d},{ys.max():4d} person {per[i].sum():6d}')
PY"
```
<details><summary>output</summary>

```
  0 area 37054 cen  711.92  826.30 box  634, 690, 789, 976 person  31043
 20 area 36971 cen  712.33  826.34 box  635, 691, 789, 976 person  27329
 40 area 36881 cen  712.60  826.28 box  636, 691, 789, 976 person  28007
 60 area 37102 cen  712.61  826.32 box  633, 691, 790, 976 person  34886
 80 area 37175 cen  712.49  826.29 box  633, 691, 790, 976 person  43482
100 area 36860 cen  712.87  826.77 box  635, 691, 789, 975 person  61187
120 area 36960 cen  712.73  826.56 box  637, 691, 789, 976 person  71912
140 area 37020 cen  712.58  826.32 box  634, 691, 789, 976 person  69826
160 area 37013 cen  712.60  826.35 box  634, 691, 789, 976 person  61613
180 area 36968 cen  712.72  826.45 box  634, 691, 789, 976 person  55931
200 area 35548 cen  714.40  831.33 box  639, 691, 790, 976 person  608 …[+4383 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/work/inputs/masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
for i in range(170,261,2):
 ys,xs=np.where(obj[i]); print(i,len(xs),round(xs.mean(),2),round(ys.mean(),2),xs.min(),ys.min(),xs.max(),ys.max())
cap=cv2.VideoCapture('/work/inputs/video.mp4'); ids=list(range(180,261,5)); ims=[]
for i in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); y0,x0=500,550; crop=im[y0:1050,x0:1050]; crop=cv2.resize(crop,(300,330)); cv2.putText(crop,str(i),(8,25),0,.7,(0,0,255),2); ims.append(crop)
while len(ims)%4: ims.append(np.zeros_like(ims[0]))
out=np.vstack([np.hstack(ims[i:i+4]) for i in range(0,len(ims),4)]); cv2.imwrite('/work/scratch/analysis/start_phase.jpg',out)
PY"
```
<details><summary>output</summary>

```
170 37019 712.6 826.38 634 691 789 976
172 37031 712.63 826.4 634 691 789 976
174 37021 712.65 826.43 634 691 789 976
176 37013 712.65 826.4 634 691 789 976
178 36993 712.7 826.4 634 691 789 976
180 36968 712.72 826.45 634 691 789 976
182 37001 712.71 826.43 634 691 789 976
184 36991 712.72 826.37 634 691 789 976
186 37000 712.75 826.46 634 691 789 976
188 36745 713.25 827.17 638 691 789 976
190 36732 713.69 828.18 639 690 790 976
192 35981 714.31 829.84 640 691 789 976
194 35588 715.03 830.94 640 691 790 976
196 35362 715.19 831.81 640 691 790 976
198 35148 715.25 832.58 640 691 790 976
200 35548 714.4 831.33 639 691 790 976
202 35355 714.63 831.9 641 691 790 976
204 35103 714.46 832.27 642 691 789 976
206 34868 714.64 833.0 642 691 790 976
208 35055 714.82 832.72 642 690 790 976
210 3493 …[+1023 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); c=r['contact_prob_right_left'];
for i in range(170,280,5): print(i,np.round(c[i],3))
PY"
```
<details><summary>output</summary>

```
170 [0. 0.]
175 [0. 0.]
180 [0. 0.]
185 [0. 0.]
190 [0. 0.]
195 [0. 0.]
200 [0. 0.]
205 [0. 0.]
210 [0.002 0.   ]
215 [0.103 0.005]
220 [0.675 0.004]
225 [0.941 0.773]
230 [1. 1.]
235 [1. 1.]
240 [1. 1.]
245 [1. 1.]
250 [1. 1.]
255 [1. 1.]
260 [1. 1.]
265 [1. 1.]
270 [1. 1.]
275 [1. 1.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
c=np.load('/work/inputs/reconstruction.npz')['contact_prob_right_left']
for a,b in [(470,530),(620,680),(720,750)]:
 print('range',a,b)
 for i in range(a,b+1,2): print(i,np.round(c[i],2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/static_eval.py <<'PY'
import numpy as np, torch, trimesh, h5py, cv2, json, nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation
D='/work/inputs'; cam=json.load(open(D+'/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Ti=np.linalg.inv(Twc)
r=np.load(D+'/reconstruction.npz'); B=r['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry(); verts=m.vertices; faces=np.asarray(m.faces)
V=torch.tensor(verts,dtype=torch.float32,device='cuda'); F=torch.tensor(faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.; tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2); tex=torch.tensor(tex,device='cuda')[None]
H,W=cam['height'],cam['width']; gl=dr.RasterizeCudaContext(device='cuda')
def pose_median(T):
 R=Rotation.from_matrix(T[:,:3,:3]); Rm=R.mean().as_matrix(); t=np.median(T[:,:3,3],axis=0)
 out=np.eye(4); out[:3,:3]=Rm; out[:3,3]=t; return out
def render(T,scale=1.0):
 Tc2m=Ti@T.astype(np.float32); vh=np.c_[verts,np.ones(len(V))]; vc=(Tc2m@vh.T).T[:,:3]
 u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2]; v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2]
 xn=2*u/W-1; yn=1-2*v/H; zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1)
 pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(V))],dtype=torch.float32,device='cuda')[None]
 rast,_=dr.rasterize(gl,pos …[+1779 chars]
```
<details><summary>output</summary>

```

PHASE pre median pose [[ 0.46281002 -0.88639785  0.01028292 -0.07728334]
 [ 0.06088652  0.04335868  0.99720252  0.80768055]
 [-0.88436402 -0.46088923  0.07403651  2.49015141]
 [ 0.          0.          0.          1.        ]]
median iou 0.9501 ncc 0.3307 ioustd 0.0023459767659567386
mean iou 0.9512 ncc 0.3299 ioustd 0.002374820636572938
perframe B iou 0.9499 ncc 0.3264

PHASE mid median pose [[ 0.62163895 -0.7831979   0.01288691  0.76902664]
 [ 0.04987939  0.05599811  0.99718416  0.78640485]
 [-0.78171418 -0.61924573  0.07387604  2.49850702]
 [ 0.          0.          0.          1.        ]]
median iou 0.9265 ncc 0.4107 ioustd 0.007148099086520859
mean iou 0.9269 ncc 0.4118 ioustd 0.0071533220950917645
perframe B iou 0.9273 ncc 0.4094

PHASE final median pose [[ 0.57730582 -0.81646994   …[+323 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/score_all.py <<'PY'
import numpy as np, torch, trimesh, h5py, cv2, json, nvdiffrast.torch as dr, time
D='/work/inputs'; cam=json.load(open(D+'/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Ti=np.linalg.inv(Twc)
r=np.load(D+'/reconstruction.npz'); B=r['T_world_mesh']; Fcnt=len(B)
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry(); verts=m.vertices; faces=np.asarray(m.faces)
V=torch.tensor(verts,dtype=torch.float32,device='cuda'); F=torch.tensor(faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.; tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2); tex=torch.tensor(tex,device='cuda')[None]
H,W=576,768; gl=dr.RasterizeCudaContext(device='cuda')
cap=cv2.VideoCapture(D+'/video.mp4'); images=[]
for i in range(Fcnt):
 ok,im=cap.read(); assert ok; images.append(cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA))
with h5py.File(D+'/masks.h5') as hf:
 obj=np.stack([cv2.resize(hf['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0 for i in range(Fcnt)])
 per=np.stack([cv2.resize(hf['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0 for i in range(Fcnt)])
images=np.stack(images)
all_I=[]; all_N=[]; all_G=[]
for s in range(0,Fcnt,16):
 e=min(Fcnt,s+16); n=e-s
 vh=np.c_[verts,np.o …[+1974 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/score_all.py", line 23, in <module>
    rast,_=dr.rasterize(gl,pos,F,resolution=(H,W)); uv,_=dr.interpolate(UV[None].expand(n,-1,-1),rast,F)
                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, …[+228 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/UV\\[None\\].expand(n,-1,-1)/UV[None].repeat(n,1,1)/\" /work/scratch/analysis/score_all.py; PYTHONUNBUFFERED=1 python /work/scratch/analysis/score_all.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/score_all.py", line 24, in <module>
    col=dr.texture(tex.expand(n,-1,-1,-1),uv,filter_mode='linear')[...,:3].cpu().numpy()[...,::-1]*255
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 439, in texture
    return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/n …[+287 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/tex.expand(n,-1,-1,-1)/tex.repeat(n,1,1,1)/" /work/scratch/analysis/score_all.py; PYTHONUNBUFFERED=1 python /work/scratch/analysis/score_all.py'
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2898: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
done 16 / 858
done 32 / 858
done 48 / 858
done 64 / 858
done 80 / 858
done 96 / 858
done 112 / 858
done 128 / 858
done 144 / 858
done 160 / 858
done 176 / 858
done 192 / 858
done 2 …[+2533 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/score_all.py'; s=open(p).read(); s=s.replace(\"col=dr.texture(tex.repeat(n,1,1,1),uv,filter_mode='linear')[...,:3].cpu().numpy()[...,::-1]*255\\n masks=(rast[...,3]>0).cpu().numpy()\", \"col=dr.texture(tex.repeat(n,1,1,1),uv,filter_mode='linear')[...,:3].cpu().numpy()[...,::-1]*255\\n masks=(rast[...,3]>0).cpu().numpy()\\n col=col[:,::-1].copy(); masks=masks[:,::-1].copy()\")
open(p,'w').write(s)
PY
PYTHONUNBUFFERED=1 python /work/scratch/analysis/score_all.py"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
  return _methods._mean(a, axis=axis, dtype=dtype,
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2897: RuntimeWarning: invalid value encountered in divide
  c /= stddev[:, None]
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2898: RuntimeWarning: invalid value encountered in divide
  c /= stddev[None, :]
done 16 / 858
done 32 / 858
done 48 / 858
done 64 / 858
done 80 / 858
done 96 / 858
done 112 / 858
done 128 / 858
done 144 / 858
done 160 / 858
done 176 / 858
done 192 / 858
done 2 …[+2533 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# execute first part of score script in interactive no frames load
exec(open('/work/scratch/analysis/score_all.py').read().split(\"cap=cv2.VideoCapture\")[0])
import numpy as np
s=0;e=1;n=1;vh=np.c_[verts,np.ones(len(verts))];Tc=np.stack([Ti@B[i].astype(np.float32) for i in range(s,e)])
vc=torch.tensor(np.einsum('nij,vj->nvi',Tc,vh),dtype=torch.float32,device='cuda')
u=K[0,0]*vc[...,0]/vc[...,2]+K[0,2];v=K[1,1]*vc[...,1]/vc[...,2]+K[1,2]
xn=2*u/W-1;yn=1-2*v/H;zn=torch.clamp((vc[...,2]-.1)/9.9*2-1,-1,1)
pos=torch.stack([xn,yn,zn,torch.ones_like(zn)],-1)
print('pos ranges',pos[...,0].min().item(),pos[...,0].max().item(),pos[...,1].min().item(),pos[...,1].max().item(),pos[...,2].min().item(),pos[...,2].max().item())
rast,_=dr.rasterize(gl,pos,F,resolution=(H,W)); print('rast',rast.shape,rast[...,3].min().item(),rast[...,3].max().item(),(rast[...,3]>0).sum().item())
uv,_=dr.interpolate(UV[None].repeat(n,1,1),rast,F); print('uv',uv.min().item(),uv.max().item())
PY"
```
<details><summary>output</summary>

```
pos ranges 0.6505191326141357 1.0789833068847656 -2.4076523780822754 -1.3939197063446045 -0.5853348970413208 -0.4875139594078064
rast torch.Size([1, 576, 768, 4]) 0.0 0.0 0
uv 0.0 0.0

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/score_all.py'; s=open(p).read(); s=s.replace(\"H,W=576,768; gl=dr.RasterizeCudaContext(device='cuda')\", \"H,W=576,768; gl=dr.RasterizeCudaContext(device='cuda')\\nK[0]*=W/cam['width']; K[1]*=H/cam['height']\")
open(p,'w').write(s)
PY
PYTHONUNBUFFERED=1 python /work/scratch/analysis/score_all.py"
```
<details><summary>output</summary>

```
done 16 / 858
done 32 / 858
done 48 / 858
done 64 / 858
done 80 / 858
Traceback (most recent call last):
  File "/work/scratch/analysis/score_all.py", line 20, in <module>
    vc=torch.tensor(np.einsum('nij,vj->nvi',Tc,vh),dtype=torch.float32,device='cuda')
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/einsumfunc.py", line 1371, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/score_all.py'; s=open(p).read()
s=s.replace(\"tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.; tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2); tex=torch.tensor(tex,device='cuda')[None]\", \"tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.; tex=cv2.resize(tex,(1250,1142),interpolation=cv2.INTER_AREA); tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2); tex=torch.tensor(tex,device='cuda')[None]\")
s=s.replace('H,W=576,768','H,W=384,512')
open(p,'w').write(s)
PY
PYTHONUNBUFFERED=1 python /work/scratch/analysis/score_all.py"
```
<details><summary>output</summary>

```
done 16 / 858
done 32 / 858
done 48 / 858
done 64 / 858
done 80 / 858
done 96 / 858
done 112 / 858
done 128 / 858
done 144 / 858
done 160 / 858
done 176 / 858
done 192 / 858
done 208 / 858
done 224 / 858
done 240 / 858
done 256 / 858
done 272 / 858
done 288 / 858
done 304 / 858
done 320 / 858
done 336 / 858
done 352 / 858
done 368 / 858
done 384 / 858
done 400 / 858
done 416 / 858
done 432 / 858
done 448 / 858
done 464 / 858
done 480 / 858
done 496 / 858
done 512 / 858
done 528 / 858
done 544 / 858
done 560 / 858
done 576 / 858
done 592 / 858
done 608 / 858
done 624 / 858
done 640 / 858
done 656 / 858
done 672 / 858
done 688 / 858
done 704 / 858
done 720 / 858
done 736 / 858
done 752 / 858
done 768 / 858
done 784 / 858
done 800 / 858
done 816 / 858
done 832 / 858
done 848 / 858
done 858 /  …[+2462 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))
for a,b in [(0,30),(175,225),(495,525),(730,765)]:
 print('\\n',a,b)
 for i in range(a,b+1,5): print(i,np.round(T[i,:3,3],4),'rel',round(ang(T[0],T[i]) if i<200 else ang(T[a],T[i]),3))
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
s=np.loadtxt('/work/scratch/analysis/scores_all.csv',delimiter=',',skiprows=1)
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
from scipy.spatial.transform import Rotation
def med(T):
 o=np.eye(4);o[:3,:3]=Rotation.from_matrix(T[:,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[:,:3,3],0);return o
for name,(a,b) in {'pre':(0,187),'mid':(504,635),'final':(738,857)}.items():
 x=s[a:b+1]; sc=x[:,1]+0.2*x[:,2]+0.1*x[:,3]; top=x[np.argsort(sc)[-20:],0].astype(int)
 print('\\n',name,'top',top.tolist(),'scores',s[top,1:].mean(0))
 for label,ids in [('all',np.arange(a,b+1)),('top20',top),('best',[top[-1]])]:
  P=med(T[ids]); print(label,'t',np.round(P[:3,3],4),'rel0',np.round(np.degrees(np.arccos(np.clip((np.trace(T[ids[0],:3,:3].T@P[:3,:3])-1)/2,-1,1))),3))
PY"
```
<details><summary>output</summary>

```

 pre top [182, 166, 118, 164, 175, 179, 184, 121, 162, 161, 123, 185, 178, 177, 120, 167, 122, 180, 176, 105] scores [0.95388771 0.41473029 0.16324348]
all t [-0.0773  0.8077  2.4902] rel0 0.045
top20 t [-0.0776  0.8056  2.4829] rel0 0.045
best t [-0.0766  0.8123  2.5061] rel0 0.045

 mid top [632, 631, 630, 633, 530, 537, 547, 552, 548, 629, 634, 520, 549, 510, 517, 551, 516, 504, 514, 513] scores [0.93976554 0.46101268 0.03614741]
all t [0.769  0.7864 2.4985] rel0 0.528
top20 t [0.7634 0.7826 2.4857] rel0 0.163
best t [0.7607 0.7826 2.4864] rel0 0.161

 final top [756, 741, 757, 761, 759, 760, 753, 755, 754, 742, 747, 743, 744, 746, 752, 745, 750, 751, 749, 748] scores [0.95993654 0.43999529 0.26271837]
all t [0.3434 0.7866 2.3321] rel0 0.201
top20 t [0.3596 0.8079 2.4025] rel0 0.201
be …[+39 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/optimize_static.py <<'PY'
import numpy as np, torch, trimesh, h5py, cv2, json, nvdiffrast.torch as dr, time
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_erosion
D='/work/inputs'; cam=json.load(open(D+'/camera.json')); K0=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Ti=np.linalg.inv(Twc)
r=np.load(D+'/reconstruction.npz'); B=r['T_world_mesh']
scores=np.loadtxt('/work/scratch/analysis/scores_all.csv',delimiter=',',skiprows=1)
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry(); verts=m.vertices; faces=np.asarray(m.faces); C=np.array(m.bounds).mean(0)
V=torch.tensor(verts,dtype=torch.float32,device='cuda'); F=torch.tensor(faces,dtype=torch.int32,device='cuda'); UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.; tex=cv2.resize(tex,(1250,1142),interpolation=cv2.INTER_AREA); tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2); tex=torch.tensor(tex,device='cuda')[None]
H,W=384,512; K=K0.copy(); K[0]*=W/cam['width']; K[1]*=H/cam['height']; gl=dr.RasterizeCudaContext(device='cuda')
def pose_median(T):
 o=np.eye(4);o[:3,:3]=Rotation.from_matrix(T[:,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[:,:3,3],0);return o
def render(T):
 Tc2m=Ti@T.astype(np.float32); vh=np.c_[verts,np.ones(len(V) …[+3474 chars]
```
<details><summary>output</summary>

```

PHASE pre elapsed 98.02904105186462 params [-0.10747957  1.20245186  1.41386705 -0.00172336  0.01624089  0.04849799] metrics (0.9852657004830918, 0.5069621419343245, 0.5591870436775814) T [[ 0.4405075  -0.89756739  0.01805345 -0.08004043]
 [ 0.04098538  0.04019536  0.99835091  0.82167985]
 [-0.89681288 -0.43904114  0.0544935   2.53300365]
 [ 0.          0.          0.          1.        ]]

PHASE mid elapsed 65.34824228286743 params [-1.74128524  0.94488572  0.08299255  0.00652229  0.015978    0.04800968] metrics (0.9855217174238642, 0.558480983205604, 0.5733757709609478) T [[ 6.20271511e-01 -7.84386965e-01 -5.85597365e-04  7.70863985e-01]
 [ 3.33876280e-02  2.56561077e-02  9.99113122e-01  7.98180362e-01]
 [-7.83676285e-01 -6.19740957e-01  4.21025729e-02  2.53598883e+00]
 [ 0.00000000e+00 …[+536 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.c_[m.vertices,np.ones(len(m.vertices))]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']; opt=np.load('/work/scratch/analysis/static_opt.npz'); pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
def d(P): W=(P@V.T).T[:,:3]; vals=W@pl[:3]+pl[3]; return vals.min(),np.percentile(vals,[1,5,50,95,99]),W.min(0),W.max(0)
for name,ids in [('pre',range(0,188,20)),('mid',range(504,636,20)),('final',range(738,858,20))]:
 print('\\n',name)
 for i in ids:
  z=d(T[i]); print('b',i,round(z[0],3),'y',round(z[3][1],3))
 print('opt',d(opt[name+'_T']))
PY"
```
<details><summary>output</summary>

```

 pre
b 0 0.016 y 1.246
b 20 0.011 y 1.251
b 40 0.015 y 1.248
b 60 0.009 y 1.252
b 80 0.005 y 1.256
b 100 0.005 y 1.256
b 120 0.013 y 1.249
b 140 0.018 y 1.245
b 160 0.015 y 1.247
b 180 0.018 y 1.245
opt (-0.00010264065058884775, array([0.00387364, 0.05304036, 0.54737491, 0.70163885, 0.70790444]), array([-0.31346415,  0.52910371,  2.28565742]), array([0.15597657, 1.26056085, 2.75618886]))

 mid
b 504 0.016 y 1.229
b 524 0.025 y 1.222
b 544 0.023 y 1.223
b 564 0.012 y 1.232
b 584 0.018 y 1.228
b 604 0.018 y 1.228
b 624 0.01 y 1.234
opt (0.005658029556090849, array([0.0101598 , 0.06034037, 0.55476475, 0.71015077, 0.7171482 ]), array([0.53243899, 0.5075883 , 2.28391016]), array([1.02205509, 1.23490681, 2.77413032]))

 final
b 738 0.007 y 1.251
b 758 0.014 y 1.245
b 778 0.029 y 1.233
b 798 0.0 …[+270 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.c_[m.vertices,np.ones(len(m.vertices))]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']; pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
W=np.einsum('fij,vj->fvi',T.astype(np.float64),V)[...,:3]; vals=W@pl[:3]+pl[3]; mn=vals.min(1)
for i in range(0,858,20): print(i,'mindist',round(mn[i],3),'tr',np.round(T[i,:3,3],2))
print('range',mn.min(),mn.max());
for i in np.argsort(mn)[:20]: print('low',i,mn[i])
for i in np.argsort(mn)[-20:]: print('high',i,mn[i])
PY"
```
<details><summary>output</summary>

```
0 mindist 0.016 tr [-0.08  0.8   2.48]
20 mindist 0.011 tr [-0.08  0.81  2.49]
40 mindist 0.015 tr [-0.08  0.81  2.48]
60 mindist 0.009 tr [-0.08  0.81  2.5 ]
80 mindist 0.005 tr [-0.08  0.81  2.51]
100 mindist 0.005 tr [-0.08  0.81  2.51]
120 mindist 0.013 tr [-0.08  0.81  2.49]
140 mindist 0.018 tr [-0.08  0.8   2.48]
160 mindist 0.015 tr [-0.08  0.81  2.48]
180 mindist 0.018 tr [-0.08  0.8   2.48]
200 mindist 0.013 tr [-0.08  0.81  2.49]
220 mindist 0.015 tr [-0.08  0.81  2.48]
240 mindist 0.013 tr [-0.08  0.81  2.49]
260 mindist 0.017 tr [-0.1   0.8   2.48]
280 mindist 0.02 tr [-0.11  0.79  2.49]
300 mindist 0.125 tr [-0.15  0.67  2.59]
320 mindist 0.206 tr [-0.14  0.6   2.66]
340 mindist 0.254 tr [0.07 0.55 2.48]
360 mindist 0.28 tr [0.27 0.54 2.3 ]
380 mindist 0.262 tr [0.42 0.55 2.3 …[+2102 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
s=np.loadtxt('/work/scratch/analysis/scores_all.csv',delimiter=',',skiprows=1)
# print low score segments/outliers
for name,(a,b) in {'pre':(0,190),'move1':(190,505),'mid':(504,636),'move2':(636,738),'final':(738,858)}.items():
 x=s[a:b]
 print('\\n',name,'iou mean',x[:,1].mean(),'q',np.percentile(x[:,1],[5,10,50,90]),'ncc mean',x[:,2].mean())
 # lowest combined
 sc=x[:,1]+.2*x[:,2]
 for j in np.argsort(sc)[:20]: print(int(x[j,0]),np.round(x[j,1:],3).tolist(),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```

 pre iou mean 0.9577279738097962 q [0.94984895 0.95049734 0.95700916 0.9653638 ] ncc mean 0.3535907441385837
0 [0.95, 0.307, 0.163]; 1 [0.95, 0.312, 0.162]; 2 [0.951, 0.309, 0.168]; 34 [0.951, 0.31, 0.161]; 35 [0.951, 0.312, 0.158]; 3 [0.952, 0.307, 0.161]; 40 [0.953, 0.306, 0.152]; 32 [0.952, 0.31, 0.15]; 30 [0.952, 0.31, 0.141]; 33 [0.952, 0.313, 0.16]; 36 [0.951, 0.315, 0.163]; 31 [0.952, 0.315, 0.15]; 41 [0.952, 0.318, 0.15]; 39 [0.953, 0.312, 0.156]; 26 [0.956, 0.3, 0.146]; 27 [0.954, 0.309, 0.145]; 28 [0.954, 0.311, 0.148]; 37 [0.954, 0.314, 0.157]; 38 [0.955, 0.31, 0.157]; 42 [0.953, 0.321, 0.15]; 

 move1 iou mean 0.8569457884058077 q [0.6892499  0.78720862 0.87861137 0.90931192] ncc mean 0.4229284601368261
321 [0.664, 0.243, 0.338]; 333 [0.654, 0.349, 0.218]; 320 [0.681, 0.218, 0 …[+2430 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']; N=len(T)
tr=T[:,:3,3]; R=Rotation.from_matrix(T[:,:3,:3]); rv=R.as_rotvec(); dt=np.diff(tr,axis=0); dr=np.diff(rv,axis=0); dn=np.linalg.norm(dt,axis=1); da=np.degrees(np.linalg.norm(dr,axis=1))
# accel
an=np.abs(np.diff(dn)); aa=np.degrees(np.abs(np.diff(rv,axis=0)))
print('max speed',dn.max(),da.max())
print('top trans speeds')
for i in np.argsort(dn)[-30:][::-1]: print(i,'->',i+1,'d',dn[i],'a',da[i],'pos',tr[i])
print('top rot speeds')
for i in np.argsort(da)[-30:][::-1]: print(i,'->',i+1,'d',dn[i],'a',da[i],'pos',tr[i])
PY"
```
<details><summary>output</summary>

```
max speed 0.054431412 3.5986818230418676
top trans speeds
342 -> 343 d 0.054431412 a 1.9557404588503797 pos [0.10623258 0.54770243 2.4803615 ]
680 -> 681 d 0.033675157 a 0.7535464202630927 pos [0.7210331 0.778916  2.4903328]
332 -> 333 d 0.03154465 a 1.0462916233798472 pos [-0.03912057  0.57765007  2.5885623 ]
690 -> 691 d 0.030232089 a 0.5313832301382585 pos [0.5633339 0.7773424 2.424722 ]
393 -> 394 d 0.030053433 a 0.7088304983837029 pos [0.52908355 0.5237307  2.4317036 ]
682 -> 683 d 0.027871396 a 1.0371892575603385 pos [0.6866401  0.76897424 2.4606323 ]
339 -> 340 d 0.026558928 a 0.7744506586878309 pos [0.05262429 0.5551425  2.497945  ]
687 -> 688 d 0.025340429 a 0.4742754430159255 pos [0.61373496 0.7741612  2.4352102 ]
429 -> 430 d 0.023873892 a 0.5492867919690376 pos [0.788729   0.66 …[+4324 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_frames.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);T=np.load(D+'/reconstruction.npz')['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;V=torch.tensor(verts,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.;tex=cv2.resize(tex,(2504,2287),interpolation=cv2.INTER_AREA);tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2);tex=torch.tensor(tex,device='cuda')[None]
H,W=cam['height'],cam['width'];gl=dr.RasterizeCudaContext(device='cuda')
def render(T):
 Tc=Ti@T.astype(np.float32);vc=(Tc@np.c_[verts,np.ones(len(verts))].T).T[:,:3];u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2];v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2];xn=2*u/W-1;yn=1-2*v/H;zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1);pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(v))],dtype=torch.float32,device='cuda')[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));uv,_=dr.interpolate(UV[None],rast,F);col=(dr.texture(tex,uv,filter_mode='linear')[0,...,:3].cpu().numpy()[::-1][...,::-1]*255).clip(0,255).astype(np.uint8);mask=(rast[0,...,3]>0).cpu().numpy()[::-1]; …[+1156 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/opt_trans.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr,time
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K0=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);T=np.load(D+'/reconstruction.npz')['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;V=torch.tensor(verts,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
H,W=384,512;K=K0.copy();K[0]*=W/cam['width'];K[1]*=H/cam['height'];gl=dr.RasterizeCudaContext(device='cuda')
def render(T):
 Tc=Ti@T.astype(np.float32);vc=(Tc@np.c_[verts,np.ones(len(verts))].T).T[:,:3];u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2];v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2];xn=2*u/W-1;yn=1-2*v/H;zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1);pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(v))],dtype=torch.float32,device='cuda')[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));return (rast[0,...,3]>0).cpu().numpy()[::-1]
def optim(T0,om,pm):
 def sc(T):
  mask=render(T)&~pm; tar=om&~pm; inter=(mask&tar).sum();union=(mask|tar).sum();return inter/union
 best=sc(T0); tt=T0.copy(); steps=[.04,.02,.01,.005]
 for step in steps:
  for it in range(2):
   improved=False
   for ax in range(3):
    for sign in [-1,1]:
     C=tt.copy();C[ax,3]+=sign*step; s=sc(C)
     if s>best+1e-7: best=s;tt=C;improved=True
   if not improved: break
 return tt,best
cap=cv2.Vi …[+691 chars]
```
<details><summary>output</summary>

```
220 iou0 0.9535 iou1 0.9738 shift [-0.005  0.01   0.03 ] sec 1.72
250 iou0 0.9635 iou1 0.9747 shift [-0.005  0.005  0.01 ] sec 1.12
280 iou0 0.9531 iou1 0.9725 shift [-0.005  0.015  0.03 ] sec 1.05
300 iou0 0.9402 iou1 0.9839 shift [-0.015  0.005  0.025] sec 1.68
320 iou0 0.9527 iou1 0.9841 shift [-0.01   0.01   0.025] sec 1.52
340 iou0 0.9416 iou1 0.9905 shift [0.    0.01  0.055] sec 1.55
360 iou0 0.9176 iou1 0.9737 shift [0.    0.01  0.055] sec 1.41
400 iou0 0.9311 iou1 0.9831 shift [0.    0.015 0.035] sec 1.33
440 iou0 0.9365 iou1 0.9705 shift [-0.015  0.005 -0.005] sec 1.98
460 iou0 0.9319 iou1 0.9659 shift [0.   0.02 0.03] sec 1.8
480 iou0 0.9298 iou1 0.9686 shift [0.   0.02 0.04] sec 1.68
500 iou0 0.9428 iou1 0.9694 shift [-0.02  -0.005 -0.015] sec 2.09
510 iou0 0.9446 iou1 0.9792 sh …[+540 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left','alignment/front_stereo_camera_left/scale','alignment/front_stereo_camera_left/shift']:
  o=f[k]; print(k,type(o),getattr(o,'shape',None),getattr(o,'dtype',None),dict(o.attrs))
  if isinstance(o,h5py.Dataset): print(o[:3])
 g=f['aligned/front_stereo_camera_left']; x=g[0]; print('x',type(x),getattr(x,'shape',None),getattr(x,'dtype',None)); print(x[:200] if hasattr(x,'__getitem__') else x)
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (858,) object {}
[array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)]
raw/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (858,) object {}
[array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)]
alignment/front_stereo_camera_left/scale <class 'h5py._hl.dataset.Dataset'> (858,) float32 {}
[1.1360185 1.1182321 1.1254305]
alignment/front_stereo_camera_left/shift <class 'h5py._hl.dataset.Dataset'> (858,) float32 {}
[0. 0. 0.]
x <class 'numpy.ndarray'> (643041,) uint8
[137  80  78  71  …[+796 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  b=f[k][0]; im=cv2.imdecode(b,cv2.IMREAD_UNCHANGED); print(k,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[1,50,99]))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (1152, 1536) uint16 1827 10184 [1909. 4118. 8998.]
raw/front_stereo_camera_left (1152, 1536) uint16 1609 8965 [1681. 3625. 7921.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np,json
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for i in [0,100,200,300,400,500,600,700,800]:
  d=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
  with h5py.File('/work/inputs/masks.h5','r') as h:
   om=h['object'][i];pm=h['person'][i]
  v=d[om&~pm]
  print(i,'depth q',np.percentile(v,[5,25,50,75,95]),'mean',v.mean())
PY"
```
<details><summary>output</summary>

```
0 depth q [2.409 2.476 2.539 2.61  2.858] mean 2.56652003886325
100 depth q [2.28  2.343 2.403 2.475 2.651] mean 2.4215867607162234
200 depth q [2.308   2.367   2.419   2.479   2.65465] mean 2.4363763080904692
300 depth q [2.484 2.538 2.585 2.632 2.717] mean 2.5981561174097516
400 depth q [2.53  2.564 2.598 2.63  2.747] mean 2.6080853441184533
500 depth q [2.433 2.5   2.552 2.608 2.701] mean 2.5593837705205353
600 depth q [2.551 2.624 2.687 2.755 2.93 ] mean 2.7011134098544236
700 depth q [2.302 2.386 2.459 2.546 2.656] mean 2.4697992490244682
800 depth q [2.34  2.417 2.483 2.56  2.736] mean 2.4985841020075963

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Ti=np.linalg.inv(Twc)
for i in [0,100,200,300,400,500,600,700,800]:
 c=Ti@T[i,:3,3]; print(i,'world',np.round(T[i,:3,3],3),'cam',np.round(c,3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Ti=np.linalg.inv(Twc)
for i in [0,100,200,300,400,500,600,700,800]:
 c=Ti@np.r_[T[i,:3,3],1]; print(i,'world',np.round(T[i,:3,3],3),'cam',np.round(c[:3],3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/opt_trans_all.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr,time,csv
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K0=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);T=np.load(D+'/reconstruction.npz')['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;V=torch.tensor(verts,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda')
H,W=384,512;K=K0.copy();K[0]*=W/cam['width'];K[1]*=H/cam['height'];gl=dr.RasterizeCudaContext(device='cuda')
def render(T):
 Tc=Ti@T.astype(np.float32);vc=(Tc@np.c_[verts,np.ones(len(verts))].T).T[:,:3];u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2];v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2];xn=2*u/W-1;yn=1-2*v/H;zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1);pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(v))],dtype=torch.float32,device='cuda')[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));return (rast[0,...,3]>0).cpu().numpy()[::-1]
def optim(T0,om,pm):
 def sc(T):
  mask=render(T)&~pm; tar=om&~pm; inter=(mask&tar).sum();union=(mask|tar).sum();return inter/union
 best=sc(T0);tt=T0.copy();steps=[.04,.02,.01,.005]
 for step in steps:
  for it in range(2):
   improved=False
   for ax in range(3):
    for sign in [-1,1]:
     C=tt.copy();C[ax,3]+=sign*step;s=sc(C)
     if s>best+1e-7:best=s;tt=C;improved=True
   if not improved:break
 return tt,best
cap=cv2 …[+787 chars]
```
<details><summary>output</summary>

```
190 0.9459 0.9819 [-0.005  0.015  0.04 ]
194 0.9435 0.9693 [-0.005  0.01   0.03 ]
198 0.946 0.9567 [-0.005  0.     0.01 ]
202 0.9561 0.978 [-0.005  0.01   0.03 ]
206 0.9519 0.9707 [-0.005  0.005  0.02 ]
210 0.958 0.9666 [-0.005 -0.005  0.   ]
214 0.9542 0.9743 [-0.005  0.01   0.03 ]
218 0.9531 0.9734 [-0.005  0.01   0.03 ]
222 0.9456 0.9671 [-0.005  0.01   0.03 ]
226 0.9582 0.9779 [-0.005  0.01   0.03 ]
230 0.9586 0.9772 [-0.005  0.005  0.02 ]
234 0.9446 0.9661 [-0.005  0.01   0.03 ]
238 0.966 0.9872 [-0.005  0.01   0.03 ]
242 0.9586 0.9812 [-0.005  0.01   0.03 ]
246 0.9626 0.9852 [-0.005  0.01   0.03 ]
250 0.9635 0.9747 [-0.005  0.005  0.01 ]
254 0.9485 0.9729 [-0.005  0.01   0.03 ]
258 0.9521 0.9796 [-0.005  0.01   0.03 ]
262 0.9552 0.9711 [-0.005  0.005  0.01 ]
266 0.9657 0.9745 [-0.005 …[+4610 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];O=np.load('/work/scratch/analysis/static_opt.npz');n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n=n/np.linalg.norm(n)
def angcol(R):
 z=R[:3,2]; return np.degrees(np.arccos(np.clip(abs(z@n),-1,1)))
for name,a,b in [('pre',0,188),('mid',504,636),('final',738,858)]:
 vals=[angcol(T[i,:3,:3]) for i in range(a,b)]
 print(name,'B angle',np.mean(vals),np.min(vals),np.max(vals),'opt',angcol(O[name+'_T'][:3,:3]))
PY"
```
<details><summary>output</summary>

```
pre B angle 1.2082984275010453 1.0755824782095145 1.2898137160157395 opt 0.035137758823035506
mid B angle 1.1615757507192594 0.7412870271962727 1.482476726400974 opt 1.3182743396081251
final B angle 0.8940404228942248 0.8940404228942245 0.8940404228942245 opt 1.2471605849221001

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/static_variants.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K0=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);B=np.load(D+'/reconstruction.npz')['T_world_mesh'];O=np.load('/work/scratch/analysis/static_opt.npz')
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;V=torch.tensor(verts,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.;tex=cv2.resize(tex,(1000,914),interpolation=cv2.INTER_AREA);tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2);tex=torch.tensor(tex,device='cuda')[None]
H,W=384,512;K=K0.copy();K[0]*=W/cam['width'];K[1]*=H/cam['height'];gl=dr.RasterizeCudaContext(device='cuda')
def render(T):
 Tc=Ti@T.astype(np.float32);vc=(Tc@np.c_[verts,np.ones(len(verts))].T).T[:,:3];u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2];v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2];xn=2*u/W-1;yn=1-2*v/H;zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1);pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(v))],dtype=torch.float32,device='cuda')[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));uv,_=dr.interpolate(UV[None],rast,F);col=(dr.texture(tex,uv,filte …[+1479 chars]
```
<details><summary>output</summary>

```

 pre
Bmedian [0.95710303 0.35137313] ioustd 0.0023111005994733043
opt [0.98328852 0.37528468] ioustd 0.0020144731562333805
optT_BR [0.97952613 0.28301   ] ioustd 0.0017987599438089395
BR_optT [0.97952613 0.28301   ] ioustd 0.0017987599438089395

 mid
Bmedian [0.93551945 0.43089153] ioustd 0.007075225006353208
opt [0.98297911 0.48855425] ioustd 0.007078906869798946
optT_BR [0.97850999 0.44170472] ioustd 0.007924782718324072
BR_optT [0.97850999 0.44170472] ioustd 0.007924782718324072

 final
Bmedian [0.92358618 0.40395368] ioustd 0.003394112463342547
opt [0.98533117 0.42399981] ioustd 0.0018005361870263533
optT_BR [0.98112533 0.37908908] ioustd 0.0021308975517924304
BR_optT [0.98112533 0.37908908] ioustd 0.0021308975517924304

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/eval_shifts.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr
from scipy.signal import medfilt,savgol_filter
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K0=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);B=np.load(D+'/reconstruction.npz')['T_world_mesh'];N=len(B)
S=np.loadtxt('/work/scratch/analysis/trans_shift.csv',delimiter=',',skiprows=1); frames=S[:,0].astype(int); sh=S[:,3:6]
# interpolate then robust smooth on dynamic ranges
full=np.zeros((N,3));
for c in range(3): full[:,c]=np.interp(np.arange(N),frames,sh[:,c])
sm=full.copy()
for a,b in [(220,503),(642,737)]:
 for c in range(3):
  z=medfilt(full[a:b+1,c],kernel_size=7)
  z=savgol_filter(z,11,2,mode='interp')
  sm[a:b+1,c]=z
 # conservative clamps
 sm[a:b+1,0]=np.clip(sm[a:b+1,0],-.02,.02)
 sm[a:b+1,1]=np.clip(sm[a:b+1,1],-.01,.03)
 sm[a:b+1,2]=np.clip(sm[a:b+1,2],0,.07)
print('raw dynamic stats',np.percentile(np.r_[full[220:504],full[642:738]],[0,10,50,90,100],axis=0))
print('smooth stats',np.percentile(np.r_[sm[220:504],sm[642:738]],[0,10,50,90,100],axis=0))
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;V=torch.tensor(verts,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTextur …[+2174 chars]
```
<details><summary>output</summary>

```
raw dynamic stats [[-0.04500002 -0.01499999 -0.05999994]
 [-0.01887498  0.         -0.01249999]
 [-0.005       0.00999999  0.02499986]
 [ 0.          0.01999998  0.0449999 ]
 [ 0.005       0.03000003  0.07000017]]
smooth stats [[-0.02       -0.01        0.        ]
 [-0.01555826  0.00020979  0.        ]
 [-0.005       0.01036567  0.02342649]
 [ 0.0001049   0.01851603  0.04393154]
 [ 0.00181818  0.02155011  0.07      ]]
200 {'B': [0.917, 0.447], 'raw': [0.938, 0.467], 'smooth': [0.938, 0.467]}
240 {'B': [0.894, 0.448], 'raw': [0.909, 0.444], 'smooth': [0.909, 0.442]}
280 {'B': [0.903, 0.431], 'raw': [0.913, 0.431], 'smooth': [0.913, 0.429]}
320 {'B': [0.821, 0.389], 'raw': [0.823, 0.24], 'smooth': [0.822, 0.246]}
360 {'B': [0.758, 0.332], 'raw': [0.769, 0.19], 'smooth': [0.773, 0.19]}
400 { …[+704 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/opt_pose.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr,time
from scipy.spatial.transform import Rotation
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K0=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);T=np.load(D+'/reconstruction.npz')['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;C=np.array(m.bounds).mean(0);V=torch.tensor(verts,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.;tex=cv2.resize(tex,(1000,914),interpolation=cv2.INTER_AREA);tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2);tex=torch.tensor(tex,device='cuda')[None]
H,W=384,512;K=K0.copy();K[0]*=W/cam['width'];K[1]*=H/cam['height'];gl=dr.RasterizeCudaContext(device='cuda')
def render(T):
 Tc=Ti@T.astype(np.float32);vc=(Tc@np.c_[verts,np.ones(len(verts))].T).T[:,:3];u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2];v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2];xn=2*u/W-1;yn=1-2*v/H;zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1);pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(v))],dtype=torch.float32,device='cuda')[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));uv,_=dr.interpolate(UV[None],rast,F);col=(dr.texture(tex,uv,filter_mode='linear')[0,...,: …[+1688 chars]
```
<details><summary>output</summary>

```
260 before [0.89  0.423 1.017] after [0.909 0.495 1.057] p [-7.5e-01  7.5e-01  3.4e+00  0.0e+00  0.0e+00  2.0e-03] sec 3.91
300 before [0.82  0.454 0.956] after [0.848 0.556 1.014] p [0.00e+00 1.50e+00 5.25e+00 5.00e-03 0.00e+00 5.00e-03] sec 3.49
320 before [0.681 0.218 0.747] after [0.691 0.458 0.828] p [-1.50e+00  3.40e+00  1.55e+00  5.00e-03  0.00e+00  2.00e-03] sec 4.16
340 before [0.741 0.376 0.854] after [0.778 0.375 0.891] p [-1.5   3.    5.25  0.02  0.    0.04] sec 3.42
360 before [0.864 0.336 0.965] after [0.901 0.376 1.014] p [ 0.4   -0.4    0.75  -0.008  0.002  0.017] sec 4.1
380 before [0.877 0.297 0.966] after [0.927 0.365 1.037] p [3.5e-01 0.0e+00 4.5e+00 0.0e+00 0.0e+00 2.0e-03] sec 3.95
400 before [0.845 0.375 0.957] after [0.854 0.427 0.982] p [ 1.5    0.75   0.     0.    …[+944 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/opt_pose_all.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr,time
from scipy.spatial.transform import Rotation
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K0=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);T=np.load(D+'/reconstruction.npz')['T_world_mesh']
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;C=np.array(m.bounds).mean(0);V=torch.tensor(verts,dtype=torch.float32,device='cuda');F=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture)[::-1].copy().astype(np.float32)/255.;tex=cv2.resize(tex,(1000,914),interpolation=cv2.INTER_AREA);tex=np.concatenate([tex,np.ones_like(tex[...,:1])],2);tex=torch.tensor(tex,device='cuda')[None]
H,W=384,512;K=K0.copy();K[0]*=W/cam['width'];K[1]*=H/cam['height'];gl=dr.RasterizeCudaContext(device='cuda')
def render(T):
 Tc=Ti@T.astype(np.float32);vc=(Tc@np.c_[verts,np.ones(len(verts))].T).T[:,:3];u=K[0,0]*vc[:,0]/vc[:,2]+K[0,2];v=K[1,1]*vc[:,1]/vc[:,2]+K[1,2];xn=2*u/W-1;yn=1-2*v/H;zn=np.clip((vc[:,2]-.1)/9.9*2-1,-1,1);pos=torch.tensor(np.c_[xn,yn,zn,np.ones(len(v))],dtype=torch.float32,device='cuda')[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));uv,_=dr.interpolate(UV[None],rast,F);col=(dr.texture(tex,uv,filter_mode='linear')[0,. …[+1914 chars]
```
<details><summary>output</summary>

```
220 [0.888 0.449 1.022] [0.897 0.482 1.042] [0.   0.   2.6  0.   0.   0.01]
224 [0.879 0.447 1.013] [0.882 0.515 1.037] [-0.4    1.1    2.2   -0.005  0.    -0.008]
228 [0.893 0.443 1.026] [0.902 0.47  1.043] [0.   0.   3.   0.   0.   0.01]
232 [0.887 0.446 1.021] [0.89  0.52  1.046] [-4.0e-01  1.1e+00  2.6e+00 -2.0e-03  0.0e+00 -1.0e-02]
236 [0.899 0.442 1.031] [0.913 0.485 1.058] [ 4.00e-01  4.00e-01  2.25e+00 -2.00e-03  2.00e-03  1.20e-02]
240 [0.901 0.442 1.034] [0.917 0.495 1.065] [ 0.00e+00  8.00e-01  2.25e+00 -2.00e-03  2.00e-03  1.20e-02]
244 [0.905 0.447 1.039] [0.916 0.518 1.071] [ 0.     0.75   0.75  -0.005  0.002  0.005]
248 [0.904 0.449 1.038] [0.913 0.504 1.065] [-7.50e-01  4.00e-01  2.25e+00 -2.00e-03  0.00e+00  0.00e+00]
252 [0.888 0.455 1.024] [0.897 0.486 1.042] [0.    0.  …[+8063 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import medfilt,savgol_filter
s=np.loadtxt('/work/scratch/analysis/pose_opt.csv',delimiter=',',skiprows=1); N=858
frames=s[:,0].astype(int); p=s[:,7:13]
for a,b in [(220,500),(642,737)]:
 m=(frames>=a)&(frames<=b); ff=frames[m]; pp=p[m]
 for c in range(6):
  sm=savgol_filter(medfilt(pp[:,c],5),9,2,mode='interp')
  print(a,b,c,'raw range',np.round([pp[:,c].min(),pp[:,c].max()],2),'smooth range',np.round([sm.min(),sm.max()],2),'start/end',np.round([sm[0],sm[-1]],2))
 print('at every 20')
 for j in range(0,len(ff),20): print(ff[j],np.round(pp[j],2),np.round(savgol_filter(medfilt(pp[:,c],5),9,2,mode='interp')[j],2) if False else '')
PY"
```
<details><summary>output</summary>

```
220 500 0 raw range [-4.1   2.25] smooth range [-3.2   1.43] start/end [-0.1  -3.08]
220 500 1 raw range [-1.85  4.5 ] smooth range [-1.24  2.94] start/end [-0.05  0.65]
220 500 2 raw range [-9.    5.65] smooth range [-5.59  4.26] start/end [ 2.41 -3.64]
220 500 3 raw range [-0.02  0.02] smooth range [-0.01  0.01] start/end [0. 0.]
220 500 4 raw range [-0.01  0.01] smooth range [-0.    0.01] start/end [-0.  0.]
220 500 5 raw range [-0.02  0.04] smooth range [-0.01  0.04] start/end [-0.  0.]
at every 20
220 [0.   0.   2.6  0.   0.   0.01] 
300 [0.   1.5  5.25 0.   0.   0.  ] 
380 [0.35 0.   4.5  0.   0.   0.  ] 
460 [-0.75 -1.5  -5.6   0.   -0.01  0.  ] 
642 737 0 raw range [-3.   0.4] smooth range [-2.14 -0.49] start/end [-1.51 -0.84]
642 737 1 raw range [-0.4   2.25] smooth range [0.25 0. …[+458 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/eval_candidates.py <<'PY'
import numpy as np,torch,trimesh,h5py,cv2,json,nvdiffrast.torch as dr
from scipy.signal import medfilt,savgol_filter
from scipy.spatial.transform import Rotation
D='/work/inputs';cam=json.load(open(D+'/camera.json'));K0=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Ti=np.linalg.inv(Twc);B=np.load(D+'/reconstruction.npz')['T_world_mesh'];N=len(B);O=np.load('/work/scratch/analysis/static_opt.npz')
m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;C=np.array(m.bounds).mean(0)
S=np.loadtxt('/work/scratch/analysis/pose_opt.csv',delimiter=',',skiprows=1); pf=S[:,0].astype(int); pp=S[:,7:13]
# Interpolate then smooth optimizer corrections
P=np.zeros((N,6));
for c in range(6):P[:,c]=np.interp(np.arange(N),pf,pp[:,c])
PS=P.copy()
for a,b in [(220,503),(642,737)]:
 for c in range(6):PS[a:b+1,c]=savgol_filter(medfilt(P[a:b+1,c],5),9,2,mode='interp')
 # conservative clamps
 PS[a:b+1,0]=np.clip(PS[a:b+1,0],-4,4);PS[a:b+1,1]=np.clip(PS[a:b+1,1],-3,4.5);PS[a:b+1,2]=np.clip(PS[a:b+1,2],-6,6)
 PS[a:b+1,3]=np.clip(PS[a:b+1,3],-.03,.03);PS[a:b+1,4]=np.clip(PS[a:b+1,4],-.015,.03);PS[a:b+1,5]=np.clip(PS[a:b+1,5],-.02,.05)
def apply_delta(T,p):
 A=np.eye(4);A[:3,:3]=Rotation.from_rotvec(np.deg2rad(p[:3])).as_matrix();A[:3,3]=C;Ai=np.eye(4);Ai[:3,3]=-C;Dt=np.eye(4);Dt[:3,3]=p[3:];return Dt@T@A@Ai
def slerp_pose(A,Bb,w):
 R=Rotation.from_matrix([A[:3,:3], …[+4008 chars]
```
<details><summary>output</summary>

```

 B ALL [0.90760624 0.40056235]

 anchor ALL [0.94208494 0.40329369]

 opt ALL [0.94069098 0.44180056]

 blend ALL [0.92010655 0.41062351]

PHASE pre
B [0.9505 0.3651] 37
anchor [0.9767 0.3934] 37
opt [0.9767 0.3934] 37
blend [0.9505 0.3651] 37

PHASE dyn1
B [0.8498 0.4194] 47
anchor [0.8747 0.3755] 47
opt [0.8721 0.4674] 47
blend [0.8763 0.4351] 47

PHASE mid
B [0.9355 0.4283] 23
anchor [0.9826 0.4786] 23
opt [0.9826 0.4786] 23
blend [0.9355 0.4283] 23

PHASE dyn2
B [0.9132 0.3768] 16
anchor [0.9473 0.3726] 16
opt [0.9426 0.4466] 16
blend [0.9471 0.4205] 16

PHASE final
B [0.9277 0.409 ] 20
anchor [0.9855 0.425 ] 20
opt [0.9855 0.425 ] 20
blend [0.9277 0.409 ] 20
smooth B [array([0.0087, 0.8219, 0.0087, 1.0488]), array([0.0076, 0.4   , 0.0082, 0.5109])]
smooth anchor [array([0.0087, 0.825 …[+252 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
r=np.load('/work/inputs/reconstruction.npz');kp=r['human_keypoints_world'];c=r['contact_prob_right_left'];C=np.load('/work/scratch/analysis/candidates.npz');m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
# nearest vertex dist per hands, averaged min distance per hand, evaluate contact frames
for name in ['B','anchor','opt','blend']:
 ds=[]; vals=[]
 for i in range(858):
  if max(c[i])<.5: continue
  W=(C[name][i]@np.c_[v,np.ones(len(v))].T).T[:,:3];tree=cKDTree(W)
  hs=[]
  # right 21:42, left 42:63. filter finite, take min over points to object
  for sl in [slice(21,42),slice(42,63)]:
   pts=kp[i,sl];pts=pts[np.isfinite(pts).all(1)];
   if len(pts): hs.append(tree.query(pts)[0].min())
  vals.append(np.mean(hs));ds.append(np.min(hs))
 print(name,'contact frames',len(vals),'mean hand min median',np.median(vals),'mean',np.mean(vals),'close<5cm',np.mean(np.array(vals)<.05),'all min',np.min(ds))
PY"
```
<details><summary>output</summary>

```
B contact frames 376 mean hand min median 0.005583368338588476 mean 0.013157413938126429 close<5cm 0.9547872340425532 all min 0.0003692108610666723
anchor contact frames 376 mean hand min median 0.006972512119394267 mean 0.014867774985483164 close<5cm 0.9601063829787234 all min 0.00030948011405349294
opt contact frames 376 mean hand min median 0.00463060271597144 mean 0.01314672333143802 close<5cm 0.9574468085106383 all min 0.00032485529161634265
blend contact frames 376 mean hand min median 0.0034068175735337165 mean 0.012094172634734576 close<5cm 0.9574468085106383 all min 0.00028930951535604497

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
C=np.load('/work/scratch/analysis/candidates.npz')
def rel(A,B):return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1))),np.linalg.norm(A[:3,3]-B[:3,3])
for name in ['B','anchor','opt','blend']:
 T=C[name];print('\\n',name)
 for i in [219,503,641,737]:
  a=rel(T[i],T[i+1]);print(i,'->',i+1,'rot',a[0],'tr',a[1])
PY"
```
<details><summary>output</summary>

```

 B
219 -> 220 rot 0.13122123336572936 tr 0.0
503 -> 504 rot 0.2920830798394951 tr 0.0018996632
641 -> 642 rot 0.3333784053795558 tr 0.019039463
737 -> 738 rot 0.2853052730672655 tr 0.0

 anchor
219 -> 220 rot 0.08846932694105812 tr 5.738326e-06
503 -> 504 rot 0.16071255014426922 tr 6.83559e-06
641 -> 642 rot 0.19382665722055112 tr 9.664779e-06
737 -> 738 rot 0.201741189079591 tr 9.578561e-06

 opt
219 -> 220 rot 1.9737765059719299 tr 0.05023555
503 -> 504 rot 4.9222689709472105 tr 0.037847824
641 -> 642 rot 0.902656546429246 tr 0.021507487
737 -> 738 rot 2.15049489418546 tr 0.019503158

 blend
219 -> 220 rot 1.8811760486728182 tr 0.027798465
503 -> 504 rot 3.3719686301015206 tr 0.012293438
641 -> 642 rot 2.354014762826702 tr 0.013697903
737 -> 738 rot 1.2987501632904899 tr 0.010231097

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/eval_candidates.py';s=open(p).read()
s=s.replace(\"Tblend=B.copy()\", \"Tblend=B.copy();Ttaper=B.copy()\")
s=s.replace(\"for i in list(range(220,504))+list(range(642,738)):\\n Tblend[i]=slerp_pose(Ta[i],Tp[i],.5)\", \"for i in list(range(220,504))+list(range(642,738)):\\n Tblend[i]=slerp_pose(Ta[i],Tp[i],.5)\\nfor a,b in [(220,503),(642,737)]:\\n ramp=18 if a==220 else 12\\n for i in range(a,b+1):\\n  w=.5\\n  if i<a+ramp:w*=.5*(1-np.cos(np.pi*(i-a)/ramp))\\n  if i>b-ramp:w*=.5*(1-np.cos(np.pi*(b-i)/ramp))\\n  Ttaper[i]=slerp_pose(Ta[i],Tp[i],w)\")
s=s.replace(\"['B','anchor','opt','blend']\", \"['B','anchor','opt','blend','taper']\")
s=s.replace(\"('blend',Tblend)]\", \"('blend',Tblend),('taper',Ttaper)]\")
s=s.replace(\"[('B',B),('anchor',Ta),('opt',Tp),('blend',Tblend)]\", \"[('B',B),('anchor',Ta),('opt',Tp),('blend',Tblend),('taper',Ttaper)]\")
s=s.replace(\"np.savez('/work/scratch/analysis/candidates.npz',B=B,anchor=Ta,opt=Tp,blend=Tblend,PS=PS)\", \"np.savez('/work/scratch/analysis/candidates.npz',B=B,anchor=Ta,opt=Tp,blend=Tblend,taper=Ttaper,PS=PS)\")
open('/work/scratch/analysis/eval_candidates2.py','w').write(s)
PY
python /work/scratch/analysis/eval_candidates2.py"
```
<details><summary>output</summary>

```

 B ALL [0.90760624 0.40056235]

 anchor ALL [0.94208494 0.40329369]

 opt ALL [0.94069098 0.44180056]

 blend ALL [0.92010655 0.41062351]

 taper ALL [0.92018461 0.41038434]

PHASE pre
B [0.9505 0.3651] 37
anchor [0.9767 0.3934] 37
opt [0.9767 0.3934] 37
blend [0.9505 0.3651] 37
taper [0.9505 0.3651] 37

PHASE dyn1
B [0.8498 0.4194] 47
anchor [0.8747 0.3755] 47
opt [0.8721 0.4674] 47
blend [0.8763 0.4351] 47
taper [0.8764 0.435 ] 47

PHASE mid
B [0.9355 0.4283] 23
anchor [0.9826 0.4786] 23
opt [0.9826 0.4786] 23
blend [0.9355 0.4283] 23
taper [0.9355 0.4283] 23

PHASE dyn2
B [0.9132 0.3768] 16
anchor [0.9473 0.3726] 16
opt [0.9426 0.4466] 16
blend [0.9471 0.4205] 16
taper [0.9474 0.4188] 16

PHASE final
B [0.9277 0.409 ] 20
anchor [0.9855 0.425 ] 20
opt [0.9855 0.425 ] 20
blend [0.9277 0. …[+509 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
C=np.load('/work/scratch/analysis/candidates.npz')
def rel(A,B):return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1))),np.linalg.norm(A[:3,3]-B[:3,3])
T=C['taper'];
for i in [219,503,641,737]:print(i,rel(T[i],T[i+1]))
# max jumps and accel
tr=T[:,:3,3];rv=Rotation.from_matrix(T[:,:3,:3]).as_rotvec();dt=np.diff(tr,axis=0);dr=np.diff(rv,axis=0)
print('max dt',np.linalg.norm(dt,axis=1).max(),'max drdeg',np.degrees(np.linalg.norm(dr,axis=1)).max())
for seg in [(220,503),(642,737)]:
 a,b=seg;print(seg,'max',np.linalg.norm(dt[a:b],axis=1).max(),np.degrees(np.linalg.norm(dr[a:b],axis=1)).max())
PY"
```
<details><summary>output</summary>

```
219 (1.7769615580221998, 0.052909486)
503 (1.5303345524324345, 0.030818643)
641 (2.722179618747575, 0.020046052)
737 (1.8682313509776416, 0.018417401)
max dt 0.055029146 max drdeg 3.337182351729291
(220, 503) max 0.055029146 3.337182351729291
(642, 737) max 0.032546967 1.48031937514752

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'Ttaper\\|for a,b' /work/scratch/analysis/eval_candidates2.py | head -20; python - <<'PY'
import numpy as np
C=np.load('/work/scratch/analysis/candidates.npz');print(np.max(np.abs(C['taper'][219]-C['anchor'][219])));print(np.max(np.abs(C['taper'][220]-C['anchor'][220])));print(np.max(np.abs(C['taper'][220]-C['taper'][219])))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy,scipy,torch,cv2,trimesh,h5py,nvdiffrast
print('python',sys.version.split()[0]);print('numpy',numpy.__version__);print('scipy',scipy.__version__);print('torch',torch.__version__);print('opencv',cv2.__version__);print('trimesh',trimesh.__version__);print('h5py',h5py.__version__);print('nvdiffrast',nvdiffrast.__version__)
PY"
```
<details><summary>output</summary>

```
python 3.11.10
numpy 1.26.3
scipy 1.16.3
torch 2.5.1+cu124
opencv 4.11.0
trimesh 5.1.1
h5py 3.16.0
nvdiffrast 0.4.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Trajectory recovery for the blue trash-can episode.

The method combines:
  1) pooled, static-world-pose estimation for the three rest phases;
  2) local render-and-compare pose refinement during manipulation;
  3) conservative anchoring and temporal smoothing of the manipulation poses.

It is deterministic and uses only files under --inputs plus the installed stack.
"""
import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import binary_erosion
from scipy.optimize import differential_evolution
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation

# Phase boundaries were checked against the object mask track, hand/object contact
# probability, and the video.  Frames in the three "static" ranges have a fixed
# mask bounding box (up to hand occlusion), and the released floor plane is used
# as a physical contact check when pooling their pose.
STATIC_RANGES = ((0, 219), (504, 641), (738, 857))
DYNAMIC_RANGES = ((220, 503), (642, 737))
PHASE_NAMES = ("pre_static", "manipulation_1", "middle_static",
               "manipulation_2", "final_static")
RENDER_H, RENDER_W = 384, 512
SEED = 20261020


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('"'--inputs', required=True)
    p.add_argument('--out', required=True)
    return p …[+18497 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'PYTHONUNBUFFERED=1 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/outputs/solve_run.log'
```
<details><summary>output</summary>

```
[1/6] Decoding low-resolution video and masks
[2/6] Scoring baseline and pooling static-phase evidence
  static 0: frames 0-219, pooled IoU=0.9855, NCC=0.5073, delta=[-0.3426, 1.0789, 2.039, -0.0012, 0.0153, 0.0487]
  static 1: frames 504-641, pooled IoU=0.9840, NCC=0.5606, delta=[-2.3241, 1.1646, -0.2584, 0.0059, 0.0164, 0.0449]
  static 2: frames 738-857, pooled IoU=0.9858, NCC=0.5022, delta=[-0.9015, 0.9415, 0.7432, -0.0004, 0.0127, 0.0157]
[3/6] Refining manipulation poses every 4 frames
[4/6] Temporally smoothing and anchoring corrections
[5/6] Validating baseline versus result
VALIDATION_JSON {"final_static": {"baseline_iou": 0.9276899322833032, "baseline_ncc": 0.4090345042682199, "n_heldout": 20, "result_iou": 0.9829353759969237, "result_ncc": 0.4452576591854516}, "manipulation_1":  …[+793 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];print(T.dtype,T.shape,np.isfinite(T).all());print('orth err',np.abs(np.einsum('fij,fkj->fik',T[:,:3,:3],T[:,:3,:3])-np.eye(3)).max());print('det',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
PY"
```
<details><summary>output</summary>

```
float64 (858, 4, 4) True
orth err 8.881784197001252e-16
det 0.9999999999999991 1.000000000000001

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; O=np.load('/work/scratch/analysis/static_opt.npz');B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=np.linalg.norm(pl[:3]);pl=pl/n
def md(P):
 W=(P@V.T).T[:,:3];return (W@pl[:3]+pl[3]).min()
for name,a,b,on in [('pre',0,219,'pre'),('mid',504,641,'mid'),('final',738,857,'final')]:
 print(name,'out T',np.round(T[a,:3,3],4),'floor',md(T[a]),'scratch',np.round(O[on+'_T'][:3,3],4),'floor',md(O[on+'_T']),'B',np.round(B[a,:3,3],4),'floor',md(B[a]))
PY"
```
<details><summary>output</summary>

```
pre out T [-0.0795  0.822   2.5263] floor 0.0 scratch [-0.08    0.8217  2.533 ] floor -0.00010264065058884775 B [-0.0778  0.8047  2.4799] floor 0.016406533516751587
mid out T [0.7708 0.8022 2.5338] floor -2.220446049250313e-16 scratch [0.7709 0.7982 2.536 ] floor 0.005658029556090849 B [0.7682 0.7889 2.5067] floor 0.016269285492918284
final out T [0.3593 0.8184 2.4198] floor 2.220446049250313e-16 scratch [0.3593 0.8227 2.4253] floor -0.005218454099778391 B [0.3617 0.8106 2.4115] floor 0.007121931822462901

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# use code from static_variants by exec but override vars quickly: simpler import module? execute module runs loop.
# Write compact by loading definitions from previous script part before cap
s=open('/work/scratch/analysis/static_variants.py').read(); pre=s.split(\"cap=cv2.VideoCapture\")[0]; exec(pre)
Tout=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
cap=cv2.VideoCapture(D+'/video.mp4');
with h5py.File(D+'/masks.h5') as hf:
 for name,ranges in [('pre_clean',[range(0,188,6)]),('pre_contact',[range(190,220,2)]),('mid_clean',[range(504,636,4)]),('mid_end',[range(636,642,1)]),('final',[range(738,858,4)])]:
  print('\\n',name); ids=ranges[0]
  for lab,TT in [('B',B),('out',Tout)]:
   vals=[]
   for i in ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA);om=cv2.resize(hf['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0;pm=cv2.resize(hf['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
    vals.append(score(im,om,pm,*render(TT[i])))
   print(lab,np.round(np.mean(vals,0),4))
PY"
```
<details><summary>output</summary>

```

 pre_clean
B [0.9576 0.3512]
out [0.9783 0.3251]

 pre_contact
B [0.9072 0.4509]
out [0.9334 0.4383]

 mid_clean
B [0.936  0.4356]
out [0.9747 0.4035]

 mid_end
B [0.917  0.3145]
out [0.9546 0.3932]

 final
B [0.9281 0.4104]
out [0.9827 0.4457]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/check_top.py <<'PY'
import sys,cv2,h5py,numpy as np,trimesh,json
sys.path.insert(0,'/work/outputs'); import solve
D='/work/inputs';cam=json.load(open(D+'/camera.json'));B=np.load(D+'/reconstruction.npz')['T_world_mesh'];F=len(B);m=trimesh.load(D+'/object_mesh.glb',force='scene',process=False).to_geometry()
images=np.empty((F,384,512,3),np.uint8);cap=cv2.VideoCapture(D+'/video.mp4')
for i in range(F):ok,im=cap.read();images[i]=cv2.resize(im,(512,384),interpolation=cv2.INTER_AREA)
om=np.empty((F,384,512),bool);pm=np.empty_like(om)
with h5py.File(D+'/masks.h5') as hf:
 for i in range(F):om[i]=cv2.resize(hf['object'][i].astype(np.uint8),(512,384),interpolation=cv2.INTER_NEAREST)>0;pm[i]=cv2.resize(hf['person'][i].astype(np.uint8),(512,384),interpolation=cv2.INTER_NEAREST)>0
r=solve.Renderer(m,cam,(1250,1142));iou,ncc,grad=r.batch_baseline_scores(B,images,om,pm)
for a,b in solve.STATIC_RANGES:
 sc=iou[a:b+1]+.2*ncc[a:b+1]+.1*grad[a:b+1];top=a+np.argsort(sc)[-20:]
 print(a,b,top.tolist(),np.round([iou[top].mean(),ncc[top].mean(),grad[top].mean()],4))
PY
PYTHONUNBUFFERED=1 python /work/scratch/analysis/check_top.py"
```
<details><summary>output</summary>

```
0 219 [164, 181, 183, 175, 182, 162, 121, 161, 123, 179, 184, 120, 167, 185, 122, 105, 178, 177, 180, 176] [0.9532 0.4179 0.1617]
504 641 [632, 631, 530, 630, 633, 537, 547, 552, 548, 629, 634, 520, 510, 549, 551, 517, 516, 504, 514, 513] [0.9398 0.461  0.0324]
738 857 [741, 756, 757, 761, 759, 753, 742, 760, 754, 755, 747, 743, 744, 746, 745, 752, 750, 751, 749, 748] [0.9599 0.4399 0.236 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"n = np.asarray(plane[:3], dtype=np.float64)\\n    n /= np.linalg.norm(n)\\n    W = (T @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]\\n    dist = W @ n + plane[3]\", \"norm = float(np.linalg.norm(plane[:3]))\\n    n = np.asarray(plane[:3], dtype=np.float64) / norm\\n    d = float(plane[3]) / norm\\n    W = (T @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]\\n    dist = W @ n + d\")
s=s.replace(\"static_renderer, T0, target, target_mask, valid,\\n            seed=SEED + 17 * phase_idx)\", \"static_renderer, T0, target, target_mask, valid,\\n            seed=1234)\")
s=s.replace(\"print(f'  static {phase_idx}: frames {a}-{b}, pooled IoU={iou:.4f}, '\", \"print(f'  static {phase_idx}: frames {a}-{b}, top={top.tolist()}, pooled IoU={iou:.4f}, '\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py; grep -n 'seed=1234\\|norm =\\|top=' /work/outputs/solve.py"
```
<details><summary>output</summary>

```
70:    norm = float(np.linalg.norm(plane[:3]))
358:            seed=1234)
361:        print(f'  static {phase_idx}: frames {a}-{b}, top={top.tolist()}, pooled IoU={iou:.4f}, '

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'PYTHONUNBUFFERED=1 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/outputs/solve_run.log'
```
<details><summary>output</summary>

```
[1/6] Decoding low-resolution video and masks
[2/6] Scoring baseline and pooling static-phase evidence
  static 0: frames 0-219, top=[164, 181, 183, 175, 182, 162, 121, 161, 123, 179, 184, 120, 167, 185, 122, 105, 178, 177, 180, 176], pooled IoU=0.9850, NCC=0.5135, delta=[-0.1054, 1.1073, 1.4292, -0.0015, 0.0164, 0.0515]
  static 1: frames 504-641, top=[632, 631, 530, 630, 633, 537, 547, 552, 548, 629, 634, 520, 510, 549, 551, 517, 516, 504, 514, 513], pooled IoU=0.9855, NCC=0.5585, delta=[-1.7413, 0.9449, 0.083, 0.0065, 0.016, 0.048]
  static 2: frames 738-857, top=[741, 756, 757, 761, 759, 753, 742, 760, 754, 755, 747, 743, 744, 746, 745, 752, 750, 751, 749, 748], pooled IoU=0.9864, NCC=0.5021, delta=[-1.2248, 0.8926, 1.0739, -0.0009, 0.0151, 0.0209]
[3/6] Refining manipulation poses eve …[+1106 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"        T = floor_contact_adjust(T, vertices, plane)\\n        static_poses.append(T)\", \"        # The released plane fit has several-cm tolerance.  Keep the image/mask\\n        # optimum rather than snapping the mesh to the plane (ADD cares about\\n        # metric pose, not just floor contact).\\n        static_poses.append(T)\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py; grep -n 'floor_contact_adjust\\|released plane' /work/outputs/solve.py"
```
<details><summary>output</summary>

```
68:def floor_contact_adjust(T, vertices, plane):
359:        # The released plane fit has several-cm tolerance.  Keep the image/mask

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'PYTHONUNBUFFERED=1 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/outputs/solve_run.log'
```
<details><summary>output</summary>

```
[1/6] Decoding low-resolution video and masks
[2/6] Scoring baseline and pooling static-phase evidence
  static 0: frames 0-219, top=[164, 181, 183, 175, 182, 162, 121, 161, 123, 179, 184, 120, 167, 185, 122, 105, 178, 177, 180, 176], pooled IoU=0.9850, NCC=0.5135, delta=[-0.1054, 1.1073, 1.4292, -0.0015, 0.0164, 0.0515]
  static 1: frames 504-641, top=[632, 631, 530, 630, 633, 537, 547, 552, 548, 629, 634, 520, 510, 549, 551, 517, 516, 504, 514, 513], pooled IoU=0.9855, NCC=0.5585, delta=[-1.7413, 0.9449, 0.083, 0.0065, 0.016, 0.048]
  static 2: frames 738-857, top=[741, 756, 757, 761, 759, 753, 742, 760, 754, 755, 747, 743, 744, 746, 745, 752, 750, 751, 749, 748], pooled IoU=0.9864, NCC=0.5021, delta=[-1.2248, 0.8926, 1.0739, -0.0009, 0.0151, 0.0209]
[3/6] Refining manipulation poses eve …[+1107 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/outputs/solve_run.log; ls -lh /work/outputs/object_poses_world.npz /work/outputs/solve.py /work/outputs/solve_run.log'
```
<details><summary>output</summary>

```
[1/6] Decoding low-resolution video and masks
[2/6] Scoring baseline and pooling static-phase evidence
  static 0: frames 0-219, top=[164, 181, 183, 175, 182, 162, 121, 161, 123, 179, 184, 120, 167, 185, 122, 105, 178, 177, 180, 176], pooled IoU=0.9850, NCC=0.5135, delta=[-0.1054, 1.1073, 1.4292, -0.0015, 0.0164, 0.0515]
  static 1: frames 504-641, top=[632, 631, 530, 630, 633, 537, 547, 552, 548, 629, 634, 520, 510, 549, 551, 517, 516, 504, 514, 513], pooled IoU=0.9855, NCC=0.5585, delta=[-1.7413, 0.9449, 0.083, 0.0065, 0.016, 0.048]
  static 2: frames 738-857, top=[741, 756, 757, 761, 759, 753, 742, 760, 754, 755, 747, 743, 744, 746, 745, 752, 750, 751, 749, 748], pooled IoU=0.9864, NCC=0.5021, delta=[-1.2248, 0.8926, 1.0739, -0.0009, 0.0151, 0.0209]
[3/6] Refining manipulation poses eve …[+1336 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json, trimesh
from scipy.spatial.transform import Rotation
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];print('shape/dtype/finite',T.shape,T.dtype,np.isfinite(T).all());print('rot orth err',np.abs(np.einsum('fij,fkj->fik',T[:,:3,:3],T[:,:3,:3])-np.eye(3)).max(),'det',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
def rel(A,B):return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1))),np.linalg.norm(A[:3,3]-B[:3,3])
print('boundary jumps result')
for i in [218,219,502,503,640,641,736,737,856]:print(i,'->',i+1,np.round(rel(T[i],T[i+1]),5))
print('static pose variation')
for a,b in [(0,219),(504,641),(738,857)]:
 print(a,b,'max pos std from first',np.linalg.norm(T[a:b+1,:3,3]-T[a,:3,3],axis=1).max(),'max rot',max(rel(T[a],T[i])[0] for i in range(a,b+1)))
# floor distance
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);pl=pl/np.linalg.norm(pl[:3])
for name,X in [('B',B),('out',T)]:
 ds=[]
 for a,b in [(0,219),(504,641),(738,857)]:
  vals=[]
  for i in [a,a+(b-a)//2,b]:
   W=(X[i]@V.T).T[:,:3];vals.append((W@pl[:3]+pl[3]).min())
  ds.append(vals)
 print('floor min samples',name,np.round(ds,4))
# smoothness of whole trajectory excluding static? velocities …[+472 chars]
```
<details><summary>output</summary>

```
shape/dtype/finite (858, 4, 4) float64 True
rot orth err 8.881784197001252e-16 det 0.9999999999999987 1.000000000000001
boundary jumps result
218 -> 219 [0. 0.]
219 -> 220 [0.e+00 1.e-05]
502 -> 503 [6.7189e-01 6.2000e-04]
503 -> 504 [0.e+00 1.e-05]
640 -> 641 [0. 0.]
641 -> 642 [0.e+00 1.e-05]
736 -> 737 [0.01793 0.00016]
737 -> 738 [0.e+00 1.e-05]
856 -> 857 [0. 0.]
static pose variation
0 219 max pos std from first 0.0 max rot 2.0913097891518726e-06
504 641 max pos std from first 0.0 max rot 0.0
738 857 max pos std from first 0.0 max rot 2.9575586669421963e-06
floor min samples B [[ 0.0164  0.0097  0.0151]
 [ 0.0163  0.0126 -0.0029]
 [ 0.0071  0.0355  0.0426]]
floor min samples out [[ 0.0017  0.0017  0.0017]
 [ 0.0057  0.0057  0.0057]
 [-0.0052 -0.0052 -0.0052]]
B mean speed 0.004362161 …[+325 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
R=np.load('/work/inputs/reconstruction.npz');kp=R['human_keypoints_world'];c=R['contact_prob_right_left'];C={'baseline':R['T_world_mesh'],'result':np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']};m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::2]
for name,T in C.items():
 hs=[];mins=[]
 for i in range(858):
  if max(c[i])<.5:continue
  W=(T[i]@np.c_[v,np.ones(len(v))].T).T[:,:3];tree=cKDTree(W);cur=[]
  for sl in [slice(21,42),slice(42,63)]:
   p=kp[i,sl];p=p[np.isfinite(p).all(1)]
   if len(p):cur.append(tree.query(p)[0].min())
  hs.append(np.mean(cur));mins.append(np.min(cur))
 hs=np.array(hs);mins=np.array(mins)
 print(name,'n',len(hs),'mean hand',hs.mean(),'median',np.median(hs),'close<5cm',np.mean(hs<.05),'mean all-min',mins.mean())
PY"
```
<details><summary>output</summary>

```
baseline n 376 mean hand 0.013157413938126429 median 0.005583368338588476 close<5cm 0.9547872340425532 mean all-min 0.002966408500876509
result n 376 mean hand 0.01202210942852462 median 0.0033795922162445737 close<5cm 0.9574468085106383 mean all-min 0.003148517609516581

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
rows=[('pre',37,.950452542929197,.36511225126876584,.9764075307395093,.40806821520464437),('m1',47,.8497833225543634,.419374543611143,.8760639337346486,.4371532546471004),('mid',23,.935509940278181,.4283235383789008,.9826284399972434,.4785597052475444),('m2',16,.9131627799960349,.3767729710633715,.9473875962129166,.4187700921439732),('fin',20,.9276899322833032,.4090345042682199,.9854736122535035,.4250107797739758)]
n=sum(r[1] for r in rows)
for idx,label in [(2,'biou'),(3,'bncc'),(4,'riou'),(5,'rncc')]:print(label,sum(r[1]*r[idx] for r in rows)/n)
PY"
```
<details><summary>output</summary>

```
biou 0.9076062377652878
bncc 0.40056122973259517
riou 0.9424490309601654
rncc 0.43253242812211307

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Blue Trash-Can Trajectory Recovery

## Summary

The final trajectory is a temporally constrained revision of the supplied reconstruction. It keeps the baseline trajectory where the object is moving and image evidence is ambiguous, but replaces the three stationary-world segments with pooled rigid poses and lightly regularizes the two manipulation segments.

No ground truth or challenge annotations were used. All decisions were made from the supplied video, masks, reconstruction, mesh, camera, and released floor plane.

## Recovered phases

| Phase | Frames | Interpretation | Evidence |
|---|---:|---|---|
| Pre-static | 0-219 | Object rests on the floor while the person approaches | Object-mask centroid remains at approximately (712.6, 826.3) through frame 187; the first sustained hand contact probability above 0.5 occurs at frame 219 |
| Manipulation 1 | 220-503 | Person lifts/slides the can to the right and partly toward the camera | Contact probability is high; mask moves right from near x=715 to x=986; baseline trajectory rises off the floor and rotates smoothly |
| Middle static | 504-641 | Object is released and rests in the new location | Both hand-contact probabilities fall below 0.1 by frame 504; object-mask bbox is fixed at approximately (910, 678, 1066, 952) through frame 634 |
| Manipulation 2 | 642-737 | Person moves the can back toward image-left | Contact becomes high again at frame 642; object mask moves  …[+6139 chars]
```
<details><summary>output</summary>

```
total 148K
-rw-rw-r-- 1 hanzhou hanzhou 6.0K Oct 20 22:40 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.4K Oct 20 22:40 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 22:33 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 108K Oct 20 22:39 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  20K Oct 20 22:33 solve.py
-rw-rw-r-- 1 hanzhou hanzhou 1.9K Oct 20 22:39 solve_run.log
--- REPORT head
# Blue Trash-Can Trajectory Recovery

## Summary

The final trajectory is a temporally constrained revision of the supplied reconstruction. It keeps the baseline trajectory where the object is moving and image evidence is ambiguous, but replaces the three stationary-world segments with pooled rigid poses and lightly regularizes the two manipulation segments.

No ground truth or challenge annotations were used.  …[+1315 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import shutil, os, numpy as np
p='/work/outputs/__pycache__'
if os.path.isdir(p): shutil.rmtree(p)
z=np.load('/work/outputs/object_poses_world.npz')
assert list(z.files)==['T_world_mesh']
T=z['T_world_mesh']
assert T.shape==(858,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
print('verified', z.files, T.shape, T.dtype)
print('\\n'.join(sorted(os.listdir('/work/outputs'))))
PY"
```
<details><summary>output</summary>

```
verified ['T_world_mesh'] (858, 4, 4) float64
REPORT.md
TOOLS.md
object_poses_world.npz
solve.py
solve_run.log

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -nE '/work/scratch|candidates|trans_shift|pose_opt|static_opt' /work/outputs/solve.py || true; grep -n 'seed' /work/outputs/solve.py | head -20; wc -l /work/outputs/solve.py /work/outputs/REPORT.md /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
210:def optimize_static(renderer, T0, target, target_mask, valid_mask, seed):
237:        seed=seed, popsize=10, maxiter=30, tol=0.002,
303:    np.random.seed(SEED)
304:    torch.manual_seed(SEED)
358:            seed=1234)
  470 /work/outputs/solve.py
   61 /work/outputs/REPORT.md
   35 /work/outputs/TOOLS.md
  566 total

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import PIL; print(PIL.__version__)
PY"
```
<details><summary>output</summary>

```
10.2.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/TOOLS.md';s=open(p).read().replace('| Pillow | Installed as a trimesh dependency | GLB texture decoding through trimesh |','| Pillow | 10.2.0 | GLB texture decoding through trimesh |');open(p,'w').write(s)
PY
grep -n Pillow /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
17:| Pillow | 10.2.0 | GLB texture decoding through trimesh |

```
</details>

**Agent:** Completed the trajectory recovery.

Outputs:
- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(858, 4, 4)`, `float64`
- `/work/outputs/REPORT.md`
- `/work/outputs/TOOLS.md`
- `/work/outputs/solve.py`
- `/work/outputs/solve_run.log`

Validation on 143 held-out frames:

| Metric | Baseline | Result |
|---|---:|---:|
| Weighted mask IoU | 0.9076 | **0.9424** |
| Weighted texture NCC | 0.4006 | **0.4325** |

All five phases improved on both metrics. Static phases are now constant, phase boundaries are continuous, and contact/floor checks support the revised poses. Final deterministic solver runtime was **347.9 seconds**.
