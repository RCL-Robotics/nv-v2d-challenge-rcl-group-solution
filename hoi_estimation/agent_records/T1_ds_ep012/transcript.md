# T1_ds_ep012

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T17:13:39Z

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
/usr/bin/bash -c "sed -n '81,180p' /work/AGENT_TASK.md"
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
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort && echo '---' && cat /work/inputs/task.txt && echo '---' && cat /work/inputs/episode.json && echo '---' && cat /work/inputs/camera.json && echo '---' && cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
camera.json 817 bytes
episode.json 69 bytes
frame0_reference_human.npz 191490 bytes
ground_plane.json 400 bytes
masks.h5 5403936 bytes
object_mesh.glb 22048360 bytes
reconstruction.npz 35602027 bytes
task.txt 99 bytes
video.mp4 18737223 bytes
---
Walk over to the platform on the ground and turn the platform over. Place on the ground when done.
---
{
 "episode_index": 12,
 "object": "black_platform",
 "frames": 700
}---
{
 "K": [
  [
   819.9990234375,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9990234375,
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
   823.9595947265625,
   0.0,
   768.0
  ],
  [
   0.0,
   823.9595947265625,
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
   -0.9999074800211503,
   0.008349959757947473,
   -0.01073 …[+844 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ls -la /work && find /work -maxdepth 2 -type d -printf '%p\\n' | sort | head -100"
```
<details><summary>output</summary>

```
total 40
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:07 .
drwxr-xr-x 1 hanzhou hanzhou   200 Oct 20 21:13 ..
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:07 .codex_home
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:07 .net
-rw-rw-r-- 1 hanzhou hanzhou 10592 Oct 20 21:07 AGENT_TASK.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 17:49 inputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:07 logs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:07 outputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 21:07 scratch
/work
/work/.codex_home
/work/.codex_home/.tmp
/work/.codex_home/sessions
/work/.codex_home/shell_snapshots
/work/.codex_home/skills
/work/.codex_home/thread-writer-locks
/work/.codex_home/tmp
/work/.net
/work/inputs
/work/logs
/work/outputs
/work/scratch

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, os
v=cv2.VideoCapture('/work/inputs/video.mp4')
for p in ['fps','frame_count','frame_width','frame_height']:
 print(p, v.get(getattr(cv2,p.upper())))
PY
ls -lah /pipeline | head && find /pipeline -maxdepth 2 -type f -printf '%p %s\\n' | head -100"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AttributeError: module 'cv2' has no attribute 'FPS'
total 36K
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 15 16:08 .
drwxr-xr-x 1 hanzhou hanzhou  200 Oct 20 21:13 ..
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 15 16:08 .stages
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 15 15:06 depth
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 15 15:09 export
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 15 15:21 foundationpose
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 15 16:08 inference
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 15 15:12 mhr_init
-rw-rw-r-- 1 hanzhou hanzhou 6.1K Oct 15 16:08 pipeline_report.json
/pipeline/.stages/03_human_initialization.json 2295
/pipeline/.stages/01_depth.json 1911
/pipeline/.stages/02_export.json 2160
/pipeline/.stages/04_depth_alignme …[+796 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, trimesh, json
r=np.load('/work/inputs/reconstruction.npz')
print('keys', r.files)
for k in r.files:
 a=r[k]
 print(k, a.shape, a.dtype, 'minmax', (np.nanmin(a),np.nanmax(a)) if a.size and np.issubdtype(a.dtype,np.number) else '')
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
print('mesh type',type(m), 'verts',len(m.vertices),'faces',len(m.faces))
print('bounds',m.bounds,'extents',m.extents,'center',m.centroid)
print('visual',type(m.visual), getattr(m.visual,'kind',None))
if hasattr(m.visual,'material'):
 print('material',m.visual.material, type(m.visual.material))
PY"
```
<details><summary>output</summary>

```
keys ['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (700, 4, 4) float32 minmax (-0.99998224, 2.8043213)
T_cam_mesh_foundationpose (700, 4, 4) float32 minmax (-0.9949648, 4.2201843)
T_cam_mesh_coconet (700, 4, 4) float32 minmax (-0.9966729, 3.9281306)
human_vertices_world (700, 4610, 3) float32 minmax (-0.60093963, 3.1843817)
human_keypoints_world (700, 70, 3) float32 minmax (-0.5942313, 3.160217)
contact_prob_right_left (700, 2) float32 minmax (6.5398706e-13, 1.0)
mesh type <class 'trimesh.base.Trimesh'> verts 285889 faces 95307
bounds [[-0.29098165 -0.09970255 -0.39503556]
 [ 0.31388506  0.0434465   0.41024145]] extents [0.60486671 0.14314905 0.80527702] center [ 2.31521359e-10 -7 …[+212 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader && python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY"
```
<details><summary>output</summary>

```
2.5.1+cu124 True NVIDIA RTX 6000 Ada Generation

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/scratch/diag && python - <<'PY'
import cv2, h5py, numpy as np, os
from PIL import Image, ImageDraw
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
print('frames',len(frames), frames[0].shape)
with h5py.File('/work/inputs/masks.h5','r') as f:
 print(f.keys(), f['object'].shape, f['object'].dtype)
 ids=list(range(0,len(frames),20))
 thumbs=[]
 for i in ids:
  im=frames[i].copy()
  obj=f['object'][i].astype(bool); per=f['person'][i].astype(bool)
  # contours
  cnts,_=cv2.findContours(obj.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(im,cnts,-1,(0,255,0),3)
  cnts,_=cv2.findContours(per.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(im,cnts,-1,(0,0,255),2)
  im=cv2.resize(im,(384,288),interpolation=cv2.INTER_AREA)
  cv2.rectangle(im,(0,0),120,30,(0,0,0),-1)
  cv2.putText(im,f'{i}',(5,24),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
  thumbs.append(im)
 cols=5; rows=(len(thumbs)+cols-1)//cols
 sheet=np.zeros((rows*288,cols*384,3),np.uint8)
 for j,im in enumerate(thumbs):
  r,c=divmod(j,cols); sheet[r*288:(r+1)*288,c*384:(c+1)*384]=im
 cv2.imwrite('/work/scratch/diag/contact.jpg',sheet)
print('wrote',sheet.shape)
# Sample raw at 0,100,... high res
for i in [0,50,100,150,200,250,300,350,400,450,500,550,600,650,699]:
 cv2.imwrite(f'/work/scratch/diag/f{i:03d}.jpg',frames[i])
PY"
```
<details><summary>output</summary>

```
frames 700 (1152, 1536, 3)
<KeysViewHDF5 ['object', 'person']> (700, 1152, 1536) bool
Traceback (most recent call last):
  File "<stdin>", line 23, in <module>
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'rectangle'
> Overload resolution failed:
>  - Can't parse 'pt2'. Input argument doesn't provide sequence protocol
>  - Can't parse 'rec'. Expected sequence length 4, got 2
>  - Can't parse 'pt2'. Input argument doesn't provide sequence protocol
>  - Can't parse 'rec'. Expected sequence length 4, got 2


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, h5py, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
with h5py.File('/work/inputs/masks.h5','r') as f:
 ids=list(range(0,len(frames),20)); thumbs=[]
 for i in ids:
  im=frames[i].copy(); obj=f['object'][i].astype(bool); per=f['person'][i].astype(bool)
  cv2.drawContours(im,cv2.findContours(obj.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),3)
  cv2.drawContours(im,cv2.findContours(per.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,0,255),2)
  im=cv2.resize(im,(384,288),interpolation=cv2.INTER_AREA)
  cv2.rectangle(im,(0,0),(120,30),(0,0,0),-1); cv2.putText(im,str(i),(5,24),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
  thumbs.append(im)
 cols=5; rows=(len(thumbs)+cols-1)//cols; sheet=np.zeros((rows*288,cols*384,3),np.uint8)
 for j,im in enumerate(thumbs):
  r,c=divmod(j,cols); sheet[r*288:(r+1)*288,c*384:(c+1)*384]=im
 cv2.imwrite('/work/scratch/diag/contact.jpg',sheet)
print('wrote',sheet.shape)
for i in [0,50,100,150,200,250,300,350,400,450,500,550,600,650,699]: cv2.imwrite(f'/work/scratch/diag/f{i:03d}.jpg',frames[i])
PY"
```
<details><summary>output</summary>

```
wrote (2016, 1920, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json, cv2
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; P=T[:,:3,3]; Rm=T[:,:3,:3]
# geodesic adjacent ang
rel=np.einsum('fji,fjk->fik',Rm[:-1],Rm[1:])
ang=np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
vel=np.linalg.norm(np.diff(P,axis=0),axis=1)
acc=np.linalg.norm(np.diff(P,n=2,axis=0),axis=1)
with h5py.File('/work/inputs/masks.h5','r') as f:
 areas=f['object'][:].sum((1,2)); parea=f['person'][:].sum((1,2)); inter=(f['object'][:]&f['person'][:]).sum((1,2))
# print every 10 frame stats
print('frame center xyz, rot step, vel, objarea, iou_obj_person_overlap, contact')
for i in range(0,700,10):
 print(f'{i:3d}', ' '.join(f'{x: .3f}' for x in P[i]), f'rot {ang[min(i,698)]:6.1f} vel {vel[min(i,698)]:.4f} area {areas[i]:6d} ov {inter[i]:5d} c {r[\"contact_prob_right_left\"][i,0]:.2f}/{r[\"contact_prob_right_left\"][i,1]:.2f}')
print('top jumps')
for i in np.argsort(ang)[-30:][::-1]: print(i, i+1, ang[i], vel[i], P[i], P[i+1])
PY"
```
<details><summary>output</summary>

```
frame center xyz, rot step, vel, objarea, iou_obj_person_overlap, contact
  0 -0.169  1.128  2.737 rot    0.2 vel 0.0155 area  12265 ov     0 c 0.00/0.00
 10 -0.175  1.143  2.685 rot    0.6 vel 0.0134 area  12144 ov     0 c 0.00/0.00
 20 -0.174  1.152  2.668 rot    0.2 vel 0.0322 area  12239 ov     0 c 0.00/0.00
 30 -0.179  1.151  2.685 rot    0.3 vel 0.0037 area  12248 ov     0 c 0.00/0.00
 40 -0.175  1.148  2.684 rot    0.3 vel 0.0052 area  12260 ov     0 c 0.00/0.00
 50 -0.174  1.142  2.706 rot    0.2 vel 0.0100 area  12134 ov     0 c 0.00/0.00
 60 -0.175  1.150  2.677 rot    0.2 vel 0.0122 area  12034 ov     0 c 0.00/0.00
 70 -0.168  1.143  2.691 rot    0.2 vel 0.0029 area  12243 ov    75 c 0.00/0.00
 80 -0.181  1.152  2.669 rot    0.5 vel 0.0124 area  11959 ov     0 c 0.00/0.00
 90 -0 …[+7930 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr, torch
print('nvdiffrast', dr)
PY"
```
<details><summary>output</summary>

```
nvdiffrast <module 'nvdiffrast.torch' from '/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/__init__.py'>

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; F=len(T)
p=T[:,:3,3]; R=Rot.from_matrix(T[:,:3,:3]); e=R.as_euler('xyz',degrees=True)
rel=R[:-1].inv()*R[1:]; ang=rel.magnitude()*180/np.pi
vel=np.linalg.norm(np.diff(p,axis=0),axis=1)
fig,ax=plt.subplots(5,1,figsize=(16,14),sharex=True)
ax[0].plot(p); ax[0].legend(['x','y','z']); ax[0].set_ylabel('center')
ax[1].plot(e); ax[1].legend(['ex','ey','ez']); ax[1].set_ylabel('euler')
ax[2].plot(ang); ax[2].set_ylabel('rot step deg'); ax[2].set_ylim(0,5)
ax[3].plot(vel); ax[3].set_ylabel('trans step m'); ax[3].set_ylim(0,.1)
ax[4].plot(r['contact_prob_right_left']); ax[4].set_ylabel('contact'); ax[4].legend(['right','left'])
for a in ax: a.grid(); a.axvspan(0,460,alpha=.1); a.axvspan(610,699,alpha=.1)
plt.tight_layout(); plt.savefig('/work/scratch/diag/traj.png',dpi=120)
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
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; R=T[:,:3,:3]
axes=np.einsum('fij,jk->fik',R,np.eye(3))
for i in range(0,700,20):
 y=axes[i,:,1]; x=axes[i,:,0]; z=axes[i,:,2]
 print(i, 'axes x',np.round(x,2),'y',np.round(y,2),'z',np.round(z,2),'yawLong',np.degrees(np.arctan2(z[0],z[2])))
# relative frame 0 orientations
R0=R[0]; rel=np.einsum('ji,fjk->fik',R0,R); rr=Rot.from_matrix(rel); print('relative angle')
for i in range(0,700,20): print(i, np.degrees(rr[i].magnitude()), np.round(rr[i].as_rotvec(),2))
PY"
```
<details><summary>output</summary>

```
0 axes x [-1.    0.04  0.02] y [-0.04 -1.   -0.02] z [ 0.01 -0.02  1.  ] yawLong 0.8578983133407067
20 axes x [-1.    0.05  0.02] y [-0.05 -1.   -0.01] z [ 0.02 -0.01  1.  ] yawLong 1.0080805775798547
40 axes x [-1.    0.05  0.01] y [-0.05 -1.   -0.02] z [ 0.01 -0.02  1.  ] yawLong 0.5413516416005548
60 axes x [-1.    0.04  0.02] y [-0.04 -1.   -0.01] z [ 0.02 -0.01  1.  ] yawLong 0.9552127965578057
80 axes x [-1.    0.04  0.01] y [-0.04 -1.   -0.01] z [ 0.01 -0.01  1.  ] yawLong 0.6753886250559964
100 axes x [-1.    0.04  0.01] y [-0.04 -1.   -0.  ] z [ 0.01 -0.    1.  ] yawLong 0.5960133081700842
120 axes x [-1.    0.04  0.01] y [-0.04 -1.   -0.03] z [ 0.01 -0.03  1.  ] yawLong 0.5029255618001274
140 axes x [-1.    0.04  0.01] y [-0.04 -1.   -0.02] z [ 0.01 -0.02  1.  ] yawLong 0.4891534 …[+4163 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m.visual.uv.shape, m.visual.uv.min(0),m.visual.uv.max(0))
mat=m.visual.material
print('basecolor',mat.baseColorTexture, type(mat.baseColorTexture))
print('image',mat.baseColorTexture.size, mat.baseColorTexture.mode)
print('metallic',mat.metallicFactor,'rough',mat.roughnessFactor,'basefac',mat.baseColorFactor)
print('normals',m.vertex_normals[:3],m.face_normals[:3])
PY"
```
<details><summary>output</summary>

```
(285889, 2) [0.000699   0.00072598] [0.99930102 0.99927402]
basecolor <PIL.Image.Image image mode=RGB size=5007x4818 at 0x71C027306E10> <class 'PIL.Image.Image'>
image (5007, 4818) RGB
metallic 0.0 rough None basefac None
normals [[ 0.26444737  0.88419643  0.385051  ]
 [ 0.87272188 -0.42205419  0.24541145]
 [ 0.82239516  0.00931057 -0.5688405 ]] [[ 0.87272188 -0.42205419  0.24541145]
 [ 0.69885939  0.2883864   0.65454476]
 [ 0.48491635 -0.73861639  0.46829689]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_utils.py <<'PY'
import numpy as np, torch, trimesh, nvdiffrast.torch as dr
from PIL import Image

class Renderer:
    def __init__(self, mesh_path, width=1536, height=1152, device='cuda'):
        self.device=device; self.W=width; self.H=height
        m=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
        self.verts=torch.tensor(np.asarray(m.vertices),dtype=torch.float32,device=device)
        self.faces=torch.tensor(np.asarray(m.faces),dtype=torch.int32,device=device)
        self.uv=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device=device)
        tex=np.array(m.visual.material.baseColorTexture.convert('RGB'))
        self.tex=torch.tensor(tex,dtype=torch.float32,device=device).permute(2,0,1)[None]/255.
        self.ctx=dr.RasterizeCudaContext()
    def render(self,T,K,return_uv=False):
        # T: world mesh, K: 3x3
        if isinstance(T,np.ndarray): T=torch.tensor(T,dtype=torch.float32,device=self.device)
        if isinstance(K,np.ndarray): K=torch.tensor(K,dtype=torch.float32,device=self.device)
        R=T[:3,:3]; t=T[:3,3]
        vc=self.verts@R.T+t
        x,y,z=vc[:,0],vc[:,1],vc[:,2]
        # clip (x, -y, 0.5z, z) gives image y down and NDC z=0.5
        pos=torch.stack([K[0,0]*x+K[0,1]*y+K[0,2]*z,
                         -(K[1,0]*x+K[1,1]*y+K[1,2]*z),
                         0.5*z,z],dim=-1)[None]
        rast,rast_db=dr.rasterize(self.ctx,pos,self.faces,resolution=(self.H,self …[+1594 chars]
```
<details><summary>output</summary>

```
0 0
200 0
300 0
400 0
500 0
580 0
600 0
699 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np,torch
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'][0]
ren=Renderer('/work/inputs/object_mesh.glb')
R=T[:3,:3]; t=T[:3,3]
vc=ren.verts.cpu().numpy()@R.T+t
x,y,z=vc.T
u=K[0,0]*x+K[0,2]*z; v=K[1,1]*y+K[1,2]*z
print('cam bounds',vc.min(0),vc.max(0),'u/z', (u/z).min(),(u/z).max(), 'v',(v/z).min(),(v/z).max())
pos=torch.tensor(np.stack([u,-v,.5*z,z],-1)[None],dtype=torch.float32,device='cuda')
print('pos',pos.shape,pos[...,3].min(),pos[...,3].max())
from nvdiffrast import torch as dr
ctx=dr.RasterizeCudaContext(); faces=ren.faces
rast,_=dr.rasterize(ctx,pos,faces,resolution=(1152,1536))
print('rast',rast.shape,rast[...,3].max(),(rast[...,3]>0).sum())
# try z=0.5*? z should for projection perhaps x,y? Standard nvdiffrast expects openGL.
PY"
```
<details><summary>output</summary>

```
cam bounds [-0.4873978  1.0726597  2.3409405] [0.12526222 1.242133   3.1456065 ] u/z 589.37683 794.2016 v 842.2047 987.3678
pos torch.Size([1, 285889, 4]) tensor(2.3409, device='cuda:0') tensor(3.1456, device='cuda:0')
rast torch.Size([1, 1152, 1536, 4]) tensor(0., device='cuda:0') tensor(0, device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr, inspect
print(inspect.signature(dr.rasterize))
PY"
```
<details><summary>output</summary>

```
(glctx, pos, tri, resolution, ranges=None, grad_db=True)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext()
# tri in NDC
pos=torch.tensor([[[-0.5,-0.5,.5,1],[.5,-.5,.5,1],[0,.5,.5,1]]],device='cuda',dtype=torch.float32)
tri=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
r,_=dr.rasterize(ctx,pos,tri,resolution=(100,100)); print(r[...,3].sum())
pos2=torch.tensor([[[-.5,.5,.5,1],[.5,.5,.5,1],[0,-.5,.5,1]]],device='cuda',dtype=torch.float32)
r,_=dr.rasterize(ctx,pos2,tri,resolution=(100,100)); print(r[...,3].sum())
PY"
```
<details><summary>output</summary>

```
tensor(1250., device='cuda:0')
tensor(1250., device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_utils.py'
s=open(p).read(); s=s.replace(\"pos=torch.stack([K[0,0]*x+K[0,1]*y+K[0,2]*z,\\n                         -(K[1,0]*x+K[1,1]*y+K[1,2]*z),\\n                         0.5*z,z],dim=-1)[None]\", \"u=K[0,0]*x+K[0,1]*y+K[0,2]*z\\n        v=K[1,0]*x+K[1,1]*y+K[1,2]*z\\n        pos=torch.stack([(2*u/self.W-1)*z,\\n                         -(2*v/self.H-1)*z,\\n                         0.5*z,z],dim=-1)[None]\")
open(p,'w').write(s)
PY
python - <<'PY'
import sys, cv2, json, numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer,make_overlay
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
ren=Renderer('/work/inputs/object_mesh.glb')
for i in [0,200,300,400,500,580,600,699]:
 R=ren.render(T[i],K); ov=make_overlay(frames[i],R); cv2.imwrite(f'/work/scratch/diag/ov{i:03d}.jpg',ov)
 rgb=(R['rgb'].detach().cpu().numpy()*255).astype(np.uint8); m=R['mask'].detach().cpu().numpy()
 out=frames[i].copy(); out[m]=(0.5*out[m]+0.5*rgb[m]).astype(np.uint8); cv2.imwrite(f'/work/scratch/diag/tex{i:03d}.jpg',out)
 print(i,m.sum().item())
PY"
```
<details><summary>output</summary>

```
0 0
200 0
300 0
400 0
500 0
580 0
600 0
699 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'u=\\|pos=' /work/scratch/render_utils.py && python - <<'PY'
import sys,json,numpy as np,torch,nvdiffrast.torch as dr
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'][0]
ren=Renderer('/work/inputs/object_mesh.glb'); R=T[:3,:3]; t=T[:3,3]
vc=ren.verts@torch.tensor(R.T,dtype=torch.float32,device='cuda')+torch.tensor(t,dtype=torch.float32,device='cuda'); x,y,z=vc.T
u=K[0,0]*x+K[0,1]*y+K[0,2]*z; v=K[1,0]*x+K[1,1]*y+K[1,2]*z
pos=torch.stack([(2*u/ren.W-1)*z,-(2*v/ren.H-1)*z,.5*z,z],-1)[None]
print(pos.min(1).values, pos.max(1).values)
print('ndc', (pos[...,:2]/pos[...,3:4]).min(1).values,(pos[...,:2]/pos[...,3:4]).max(1).values)
rast,_=dr.rasterize(ren.ctx,pos,ren.faces,resolution=(ren.H,ren.W)); print(rast[...,3].max(),(rast[...,3]>0).sum())
# Try doubled tri?
tri=torch.tensor([[0,100,1000]],dtype=torch.int32,device='cuda')
r2,_=dr.rasterize(ren.ctx,pos,tri,resolution=(ren.H,ren.W)); print('single',r2[...,3].sum())
PY"
```
<details><summary>output</summary>

```
tensor([[  1.9064, -11.7384,   1.1705,   2.3409]], device='cuda:0') tensor([[ 6.9363, -6.6299,  1.5728,  3.1456]], device='cuda:0')
ndc tensor([[ 0.8104, -3.7750]], device='cuda:0') tensor([[ 2.2121, -2.8258]], device='cuda:0')
tensor(0., device='cuda:0') tensor(0, device='cuda:0')
single tensor(0., device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_utils.py'; s=open(p).read(); s=s.replace(\"pos=torch.stack([(2*u/self.W-1)*z,\\n                         -(2*v/self.H-1)*z,\", \"pos=torch.stack([2*u/self.W-z,\\n                         -(2*v/self.H-z),\")
open(p,'w').write(s)
PY
python - <<'PY'
import sys, cv2, json, numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer,make_overlay
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
ren=Renderer('/work/inputs/object_mesh.glb')
for i in [0,200,300,400,500,580,600,699]:
 R=ren.render(T[i],K); ov=make_overlay(frames[i],R); cv2.imwrite(f'/work/scratch/diag/ov{i:03d}.jpg',ov)
 rgb=(R['rgb'].detach().cpu().numpy()*255).astype(np.uint8); m=R['mask'].detach().cpu().numpy()
 out=frames[i].copy(); out[m]=(0.5*out[m]+0.5*rgb[m]).astype(np.uint8); cv2.imwrite(f'/work/scratch/diag/tex{i:03d}.jpg',out)
 print(i,m.sum().item())
PY"
```
<details><summary>output</summary>

```
0 21386
200 23624
300 17850
400 14064
500 18782
580 21893
600 22972
699 21054

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys, cv2, json, numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer,make_overlay
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); inv=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
ren=Renderer('/work/inputs/object_mesh.glb')
for i in [0,200,300,400,500,580,600,699]:
 R=ren.render(inv@T[i],K); ov=make_overlay(frames[i],R); cv2.imwrite(f'/work/scratch/diag/ov{i:03d}.jpg',ov)
 rgb=(R['rgb'].detach().cpu().numpy()*255).astype(np.uint8); m=R['mask'].detach().cpu().numpy()
 out=frames[i].copy(); out[m]=(0.5*out[m]+0.5*rgb[m]).astype(np.uint8); cv2.imwrite(f'/work/scratch/diag/tex{i:03d}.jpg',out)
 print(i,m.sum().item())
PY"
```
<details><summary>output</summary>

```
0 11533
200 11154
300 10663
400 9048
500 5754
580 12162
600 11851
699 12473

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
c=json.load(open('/work/inputs/camera.json')); Twc=np.array(c['T_world_cam']); inv=np.linalg.inv(Twc); K=np.array(c['K'])
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; p=T[0,:3,3]
for name,x in [('wc*p',Twc@p),('inv*p',inv@p)]: print(name,x, (K@x)[:2]/x[2])
print('camera center',Twc[:3,3])
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
import numpy as np, json
c=json.load(open('/work/inputs/camera.json')); Twc=np.array(c['T_world_cam']); inv=np.linalg.inv(Twc); K=np.array(c['K'])
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; p=np.r_[T[0,:3,3],1]
for name,x in [('wc*p',Twc@p),('inv*p',inv@p)]: print(name,x, (K@x[:3])[:2]/x[2])
print('camera center',Twc[:3,3])
PY"
```
<details><summary>output</summary>

```
wc*p [0.19100687 1.13101149 3.5312519  1.        ] [802.45406006 823.13442762]
inv*p [0.18185863 1.05008663 3.53148439 1.        ] [800.32694944 804.32664969]
camera center [ 0.04154909 -0.24309623  6.15529686]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext(); tri=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
for sign in [1,-1]:
 pos=torch.tensor([[[-.5,sign*.5,.5,1],[.5,sign*.5,.5,1],[0,sign*-.5,.5,1]]],device='cuda',dtype=torch.float32)
 r,_=dr.rasterize(ctx,pos,tri,resolution=(100,100)); m=r[0,...,3]>0; ys,xs=torch.where(m); print(sign,ys.min().item(),ys.max().item(),xs.min().item(),xs.max().item())
PY"
```
<details><summary>output</summary>

```
1 26 74 25 74
-1 25 73 25 74

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext(); tri=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
for sign in [1,-1]:
 pos=torch.tensor([[[-.5,sign*.5,.5,1],[.5,sign*.5,.5,1],[0,sign*-.5,.5,1]]],device='cuda',dtype=torch.float32)
 r,_=dr.rasterize(ctx,pos,tri,resolution=(100,100)); m=r[0,...,3]>0; counts=m.sum(1)
 print(sign,'top',counts[:5].tolist(),'bottom',counts[-5:].tolist(),'argmin',counts.argmin().item())
PY"
```
<details><summary>output</summary>

```
1 top [0, 0, 0, 0, 0] bottom [0, 0, 0, 0, 0] argmin 0
-1 top [0, 0, 0, 0, 0] bottom [0, 0, 0, 0, 0] argmin 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext(); tri=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda')
for sign in [1,-1]:
 pos=torch.tensor([[[-.5,sign*.5,.5,1],[.5,sign*.5,.5,1],[0,sign*-.5,.5,1]]],device='cuda',dtype=torch.float32)
 r,_=dr.rasterize(ctx,pos,tri,resolution=(100,100)); m=r[0,...,3]>0; ys,xs=torch.where(m); print(sign,ys.min().item(),ys.max().item(),m.sum().item());
 for y in [25,26,50,73,74]: print(y,m[y].sum().item(),end=';')
 print()
PY"
```
<details><summary>output</summary>

```
1 26 74 1250
25 0;26 2;50 26;73 48;74 50;
-1 25 73 1250
25 50;26 48;50 24;73 2;74 0;

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np,torch
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'][0]
ren=Renderer('/work/inputs/object_mesh.glb'); R=ren.render(np.linalg.inv(Twc)@T,K); m=R['mask'].cpu().numpy(); ys,xs=np.where(m); print(xs.mean(),ys.mean(),xs.min(),xs.max(),ys.min(),ys.max())
# expected vertices pixel range
M=np.linalg.inv(Twc)@T; v=ren.verts.cpu().numpy()@M[:3,:3].T+M[:3,3]; uv=(K@v.T).T; uv=uv[:,:2]/uv[:,2:]; print(uv.min(0),uv.max(0),uv.mean(0))
PY"
```
<details><summary>output</summary>

```
804.7548773085928 338.9271655250152 730 886 286 387
[729.95237364 763.96507636] [887.04999134 866.12753508] [795.77155063 803.98595126]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_utils.py'; s=open(p).read().replace(\"-(2*v/self.H-z),\", \"2*v/self.H-z,\"); open(p,'w').write(s)
PY
python - <<'PY'
import sys, cv2, json, numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer,make_overlay
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); inv=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
ren=Renderer('/work/inputs/object_mesh.glb')
for i in [0,200,300,400,500,580,600,699]:
 R=ren.render(inv@T[i],K); ov=make_overlay(frames[i],R); cv2.imwrite(f'/work/scratch/diag/ov{i:03d}.jpg',ov)
 rgb=(R['rgb'].detach().cpu().numpy()*255).astype(np.uint8); m=R['mask'].detach().cpu().numpy()
 out=frames[i].copy(); out[m]=(0.5*out[m]+0.5*rgb[m]).astype(np.uint8); cv2.imwrite(f'/work/scratch/diag/tex{i:03d}.jpg',out)
 print(i,m.sum().item())
PY"
```
<details><summary>output</summary>

```
0 11534
200 11151
300 10664
400 9048
500 5754
580 12161
600 11848
699 12470

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/pipeline_report.json && for f in /pipeline/.stages/*.json; do echo ==== $f; cat $f; done'
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
    "mtime_ns": 1792066136957042770,
    "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000012/inference/episode_000012_step200000_before_after.mp4",
    "size": 48393566
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.0 …[+24300 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz'); c=json.load(open('/work/inputs/camera.json')); Twc=np.array(c['T_world_cam']); iw=np.linalg.inv(Twc)
base=iw@r['T_world_mesh']; fp=r['T_cam_mesh_foundationpose']; cc=r['T_cam_mesh_coconet']
for name,T in [('fp',fp),('cc',cc),('base',base)]:
 p=T[:,:3,3]; R=Rot.from_matrix(T[:,:3,:3]); rel=R[:-1].inv()*R[1:]; ang=rel.magnitude()*180/np.pi
 print(name,'pos sample')
 for i in range(0,700,50): print(i,np.round(p[i],3), 'step',round(ang[min(i,698)],2))
# relative rotations fp/cc/base
for a,b,na,nb in [(base,fp,'base','fp'),(base,cc,'base','cc'),(fp,cc,'fp','cc')]:
 rel=Rot.from_matrix(a[:,:3,:3]).inv()*Rot.from_matrix(b[:,:3,:3]); ang=rel.magnitude()*180/np.pi
 print(na,nb,'angle percentiles',np.percentile(ang,[0,10,25,50,75,90,100]))
 for i in range(0,700,50): print(i,round(ang[i],1),np.round(np.linalg.norm(a[i,:3,3]-b[i,:3,3]),3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
fp pos sample
0 [0.131 1.037 3.762] step 0.22
50 [0.127 1.031 3.754] step 0.26
100 [0.139 1.089 3.963] step 0.22
150 [0.132 1.058 3.857] step 0.25
200 [0.138 1.117 4.075] step 0.39
250 [0.24  1.032 4.114] step 1.28
300 [0.187 0.92  3.904] step 1.06
350 [0.083 0.858 3.782] step 0.44
400 [0.03  0.842 3.753] step 0.59
450 [-0.082  0.84   3.779] step 0.2
500 [-0.148  0.865  3.75 ] step 1.63
550 [-0.299  0.947  3.545] step 1.39
600 [-0.33   1.056  3.584] step 0.51
650 [-0.344  1.109  3.753] step 0.05
cc pos sample
0 [0.141 0.996 3.593] step 0.23
50 [0.144 0.998 3.6  ] step 0.21
100 [0.125 1.011 3.671] step 0.43
150 [0.134 1.003 3.64 ] step 0.25
200 [0.12  1.031 3.824] step 0.33
250 [0.229 0.948 3.895] step 1.22
300 [0.186 0.871 3.734] step 0.54
350 [0.088 0.848 3.711] step 0.39
400 [0.029 0.816 …[+1643 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline/depth /pipeline/export /pipeline/foundationpose -type f -printf '%p %s\\n' | sort"
```
<details><summary>output</summary>

```
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 723195608
/pipeline/depth/moge2/depth_report.json 1879
/pipeline/depth/moge2/intrinsics.pkl 74607
/pipeline/depth/moge2/raw.npy 2477260928
/pipeline/export/episode_000012/edex 563
/pipeline/export/episode_000012/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000012/human_masks/front_stereo_camera_left.h5 16037152
/pipeline/export/episode_000012/images/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000012/images/front_stereo_camera_left.h5 558761872
/pipeline/export/episode_000012/object_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000012/object_masks/front_stereo_camera_left.h5 14959983
/pipeline/ …[+167 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,cv2,json,numpy as np,h5py
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer,make_overlay
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); iw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; ren=Renderer('/work/inputs/object_mesh.glb')
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
with h5py.File('/work/inputs/masks.h5','r') as f:
 ids=list(range(150,271,10)); th=[]
 for i in ids:
  im=frames[i].copy(); R=ren.render(iw@T[i],K); im=make_overlay(im,R,alpha=.3)
  m=f['object'][i].astype(bool); cv2.drawContours(im,cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),2)
  crop=im[580:1050,450:1100]; crop=cv2.resize(crop,(650,470)); cv2.rectangle(crop,(0,0),(100,30),(0,0,0),-1); cv2.putText(crop,str(i),(5,24),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2); th.append(crop)
 cols=4; rows=(len(th)+cols-1)//cols; sheet=np.zeros((rows*470,cols*650,3),np.uint8)
 for j,im in enumerate(th):
  rr,cc=divmod(j,cols); sheet[rr*470:(rr+1)*470,cc*650:(cc+1)*650]=im
 cv2.imwrite('/work/scratch/diag/crop150_270.jpg',sheet)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,cv2,json,numpy as np,h5py
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer,make_overlay
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; ren=Renderer('/work/inputs/object_mesh.glb')
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
with h5py.File('/work/inputs/masks.h5','r') as f:
 ids=list(range(540,700,10)); th=[]
 for i in ids:
  im=frames[i].copy(); R=ren.render(iw@T[i],K); im=make_overlay(im,R,alpha=.3)
  m=f['object'][i].astype(bool); cv2.drawContours(im,cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),2)
  crop=im[600:1000,450:1100]; crop=cv2.resize(crop,(650,400)); cv2.rectangle(crop,(0,0),(110,30),(0,0,0),-1); cv2.putText(crop,str(i),(5,24),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2); th.append(crop)
 cols=4; rows=(len(th)+cols-1)//cols; sheet=np.zeros((rows*400,cols*650,3),np.uint8)
 for j,im in enumerate(th):
  rr,cc=divmod(j,cols); sheet[rr*400:(rr+1)*400,cc*650:(cc+1)*650]=im
 cv2.imwrite('/work/scratch/diag/crop540_700.jpg',sheet)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.array(m.vertices)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; g=json.load(open('/work/inputs/ground_plane.json')); n=np.array(g['plane'][:3]);d=g['plane'][3]
for i in [0,50,100,150,200,210,220,250,300,400,500,550,600,610,650,699]:
 w=v@T[i,:3,:3].T+T[i,:3,3]
 dist=w@n+d
 print(i,'bbox y',w[:,1].min(),w[:,1].max(),'plane dist min/mean',dist.min(),dist.mean(),'world center',T[i,:3,3])
PY"
```
<details><summary>output</summary>

```
0 bbox y 1.072659792714841 1.2421330632695988 plane dist min/mean -0.004969800917320022 0.1177557159913294 world center [-0.16944642  1.128499    2.7374988 ]
50 bbox y 1.088614323459178 1.2534073793802603 plane dist min/mean -0.01695757552772581 0.1061456017088907 world center [-0.17439796  1.1419677   2.7062278 ]
100 bbox y 1.123912210496763 1.2840070723974246 plane dist min/mean -0.04849495480323873 0.07754073400640772 world center [-0.15822305  1.1762496   2.6028867 ]
150 bbox y 1.1026702942649487 1.2619255550486053 plane dist min/mean -0.024631382861306506 0.09725559126490066 world center [-0.16501747  1.1530591   2.6624546 ]
200 bbox y 1.1005478690349473 1.250148237715952 plane dist min/mean -0.02417890199964101 0.10477263494172999 world center [-0.1481581  1.1473333  2.6184165]
210 b …[+1741 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_iou.py <<'PY'
import sys,os,json,time,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); iw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); Ts=r['T_world_mesh']
scale=.5; W=int(cam['width']*scale); H=int(cam['height']*scale); Ks=K.copy(); Ks[:2]*=scale
ren=Renderer('/work/inputs/object_mesh.glb',width=W,height=H)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA))
with h5py.File('/work/inputs/masks.h5','r') as f:
 out=[]
 t0=time.time()
 for i in range(len(Ts)):
  rm=ren.render(iw@Ts[i],Ks)
  pred=rm['mask'].detach().cpu().numpy(); person=cv2.resize(f['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  obj=cv2.resize(f['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  predv=pred&~person; targ=obj&~person
  inter=(predv&targ).sum(); union=(predv|targ).sum(); iou=inter/union if union else 1
  # centroid and area
  ys,xs=np.where(predv); yt,xt=np.where(targ)
  dx=xs.mean()-xt.mean() if len(xs) and len(xt) else 0; dy=ys.mean()-yt.mean() if len(xs) and len(yt) else 0
  out.append([iou,dx*2,dy*2,predv.sum()*4,targ.sum()*4,obj.sum()*4,person.sum()*4])
  if i%50==0: print(i,out[-1])
 out=np.array(out …[+362 chars]
```
<details><summary>output</summary>

```
0 [0.8855784469096672, 3.214946461712202, 0.4122635963627772, 11524, 12272, 12272, 31896]
50 [0.8651044965167828, 4.387955462690002, 0.480250164582003, 11408, 12152, 12152, 23160]
100 [0.8677335075114304, -1.5944287501374674, -1.0774720251841927, 10752, 12124, 12124, 19348]
150 [0.8827766863130321, 0.9525154403725082, -1.1489551434084433, 11060, 11940, 11940, 16628]
200 [0.8186940669539278, -2.313370857986456, -4.252951365969352, 10308, 11640, 11704, 15312]
250 [0.7773068144766874, 0.22991904479101777, -5.459469258900413, 9844, 11960, 12060, 13808]
300 [0.8003651856360317, 0.20213328420993548, -4.818911451150825, 10588, 13076, 13108, 13696]
350 [0.8198757763975155, 0.06842165064233541, 0.09246221559487822, 9548, 11548, 11548, 11592]
400 [0.8978162340337865, -0.08964202435402058, -1.2332646 …[+861 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 print(list(f.keys()))
 def vis(name,obj):
  if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype, 'attrs',dict(obj.attrs))
 f.visititems(vis)
 for k in f.keys():
  a=f[k]
  if isinstance(a,h5py.Dataset):
   for i in [0,100,300,699]:
    z=a[i]; print(k,i,np.nanmin(z),np.nanmax(z),np.nanmedian(z))
PY"
```
<details><summary>output</summary>

```
['aligned', 'alignment', 'frame_names', 'raw']
aligned/front_stereo_camera_left (700,) object attrs {}
alignment/front_stereo_camera_left/scale (700,) float32 attrs {}
alignment/front_stereo_camera_left/shift (700,) float32 attrs {}
alignment/front_stereo_camera_left/valid_count (700,) int32 attrs {}
frame_names/front_stereo_camera_left (700,) object attrs {}
raw/front_stereo_camera_left (700,) object attrs {}

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 for key in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  a=f[key]; print(key,a.shape,a.dtype,type(a[0]))
  for i in [0,100,300,699]:
   z=f[a[i]]; print(i,z.shape,z.dtype,np.nanmin(z),np.nanmax(z),np.nanmedian(z),np.isfinite(z).mean())
 print('frames',f['frame_names/front_stereo_camera_left'][:3])
 print('scale',f['alignment/front_stereo_camera_left/scale'][:10])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (700,) object <class 'numpy.ndarray'>
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 407, in __getitem__
    return self._get(name)
           ^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 423, in _get
    raise TypeError("Accessing a group is done with bytes or str, "
TypeError: Accessing a group is done with bytes or str, not <class 'numpy.ndarray'>

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for key in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  a=f[key]; print(key,a.shape,a.dtype); print(type(a[0]),getattr(a[0],'shape',None),getattr(a[0],'dtype',None)); print(a[0])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (700,) object
<class 'numpy.ndarray'> (551711,) uint8
[137  80  78 ...  66  96 130]
raw/front_stereo_camera_left (700,) object
<class 'numpy.ndarray'> (481143,) uint8
[137  80  78 ...  66  96 130]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 a=f['aligned/front_stereo_camera_left']; print(a.shape,len(a[0]),f['alignment/front_stereo_camera_left/valid_count'][:5])
 for i in range(3): print(i,len(a[i]),a[i][:20].tobytes())
PY
cat /pipeline/depth/moge2/depth_report.json && cat /pipeline/.stages/04_depth_alignment.json"
```
<details><summary>output</summary>

```
(700,) 551711 [27956 27496 26702 26133 25863]
0 551711 b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x06\x00'
1 553277 b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x06\x00'
2 554805 b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x06\x00'
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 7.4410014152526855,
    "min": 1.5292445421218872
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 160.0455972859927,
  "firstFrameIntrinsicsPx": [
    [
      823.9595947265625,
      0.0,
      768.0
    ],
    [
      0.0,
      823.9595947265625,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "focalXPx": {
    "max": 838.8572387695312,
    "mean": 825.8202514648438,
    "min": 810.8 …[+3374 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for i in [0,100,300,699]:
  z=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)
  print(i,z.shape,z.dtype,z.min(),z.max(),np.median(z))
  if i==0:
   cv2.imwrite('/work/scratch/diag/depth0.png', (np.clip(z,0,7000)/7).astype(np.uint16))
PY"
```
<details><summary>output</summary>

```
0 (1152, 1536) uint16 1857 8610 3953.0
100 (1152, 1536) uint16 1990 8925 4289.0
300 (1152, 1536) uint16 1948 8807 4223.0
699 (1152, 1536) uint16 1842 8307 3992.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_utils.py'; s=open(p).read(); s=s.replace(\"out={'mask':mask[0], 'rgb':rgb.permute(1,2,0), 'rast':rast[0], 'uv':uv[0]}\", \"zmap,_=dr.interpolate(vc[:,2][None],rast,self.faces)\\n        out={'mask':mask[0], 'rgb':rgb.permute(1,2,0), 'rast':rast[0], 'uv':uv[0], 'z':zmap[0,:,:,0]}\")
open(p,'w').write(s)
PY
python - <<'PY'
import sys,json,numpy as np,h5py,cv2
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; ren=Renderer('/work/inputs/object_mesh.glb')
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f, h5py.File('/work/inputs/masks.h5','r') as g:
 for i in [0,100,200,300,400,500,580,600,650,699]:
  R=ren.render(iw@T[i],K); m=R['mask'].cpu().numpy(); z=R['z'].cpu().numpy(); dm=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)/1000.
  obj=g['object'][i]; per=g['person'][i]; vis=obj&~per&m
  diff=z[vis]-dm[vis]
  print(i,'vis',vis.sum(),'pred',np.median(z[vis]),'moge',np.median(dm[vis]),'diff med mean std',np.median(diff),diff.mean(),diff.std(),'p10/90',np.percentile(diff,[10,90]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/work/scratch/render_utils.py", line 33, in render
    zmap,_=dr.interpolate(vc[:,2][None],rast,self.faces)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _nvdiffrast_c.interpolate_fwd(attr,  …[+162 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/vc\\[:,2\\]\\[None\\]/vc[:,2][None].contiguous()/\" /work/scratch/render_utils.py
python - <<'PY'
import sys,json,numpy as np,h5py,cv2
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; ren=Renderer('/work/inputs/object_mesh.glb')
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f, h5py.File('/work/inputs/masks.h5','r') as g:
 for i in [0,100,200,300,400,500,580,600,650,699]:
  R=ren.render(iw@T[i],K); m=R['mask'].cpu().numpy(); z=R['z'].cpu().numpy(); dm=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)/1000.
  obj=g['object'][i]; per=g['person'][i]; vis=obj&~per&m
  diff=z[vis]-dm[vis]
  print(i,'vis',vis.sum(),'pred',round(np.median(z[vis]),3),'moge',round(np.median(dm[vis]),3),'diff med mean std',round(np.median(diff),3),round(diff.mean(),3),round(diff.std(),3),'p10/90',np.round(np.percentile(diff,[10,90]),3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/work/scratch/render_utils.py", line 33, in render
    zmap,_=dr.interpolate(vc[:,2][None].contiguous(),rast,self.faces)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _nvdiffras …[+674 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/vc\\[:\\\",2\\]\\[None\\].contiguous()/vc[:,2][None,:,None].contiguous()/\" /work/scratch/render_utils.py
grep -n zmap /work/scratch/render_utils.py
python - <<'PY'
import sys,json,numpy as np,h5py,cv2
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; ren=Renderer('/work/inputs/object_mesh.glb')
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f, h5py.File('/work/inputs/masks.h5','r') as g:
 for i in [0,100,200,300,400,500,580,600,650,699]:
  R=ren.render(iw@T[i],K); m=R['mask'].cpu().numpy(); z=R['z'].cpu().numpy(); dm=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)/1000.
  obj=g['object'][i]; per=g['person'][i]; vis=obj&~per&m
  diff=z[vis]-dm[vis]
  print(i,'vis',vis.sum(),'pred',round(np.median(z[vis]),3),'moge',round(np.median(dm[vis]),3),'diff med mean std',round(np.median(diff),3),round(diff.mean(),3),round(diff.std(),3),'p10/90',np.round(np.percentile(diff,[10,90]),3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/work/scratch/render_utils.py", line 33, in render
    zmap,_=dr.interpolate(vc[:,2][None].contiguous(),rast,self.faces)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _nvdiffras …[+674 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_utils.py'; s=open(p).read().replace('zmap,_=dr.interpolate(vc[:,2][None].contiguous(),rast,self.faces)','zmap,_=dr.interpolate(vc[:,2].reshape(1,-1,1).contiguous(),rast,self.faces)'); open(p,'w').write(s)
PY
grep -n zmap /work/scratch/render_utils.py
python - <<'PY'
import sys,json,numpy as np,h5py,cv2
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; ren=Renderer('/work/inputs/object_mesh.glb')
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f, h5py.File('/work/inputs/masks.h5','r') as g:
 for i in [0,100,200,300,400,500,580,600,650,699]:
  R=ren.render(iw@T[i],K); m=R['mask'].cpu().numpy(); z=R['z'].cpu().numpy(); dm=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)/1000.
  obj=g['object'][i]; per=g['person'][i]; vis=obj&~per&m
  diff=z[vis]-dm[vis]
  print(i,'vis',vis.sum(),'pred',round(np.median(z[vis]),3),'moge',round(np.median(dm[vis]),3),'diff med mean std',round(np.median(diff),3),round(diff.mean(),3),round(diff.std(),3),'p10/90',np.round(np.percentile(diff,[10,90]),3))
PY"
```
<details><summary>output</summary>

```
0 vis 11100 pred 3.475 moge 3.612 diff med mean std -0.192 -0.185 0.095 p10/90 [-0.268 -0.075]
100 vis 10688 pred 3.614 moge 3.812 diff med mean std -0.267 -0.252 0.104 p10/90 [-0.347 -0.113]
200 vis 9929 pred 3.513 moge 3.921 diff med mean std -0.433 -0.42 0.07 p10/90 [-0.489 -0.346]
300 vis 10558 pred 3.706 moge 3.784 diff med mean std -0.113 -0.094 0.102 p10/90 [-0.201  0.053]
400 vis 8700 pred 3.542 moge 3.545 diff med mean std -0.072 -0.055 0.105 p10/90 [-0.15   0.065]
500 vis 5051 pred 3.456 moge 3.425 diff med mean std -0.004 -0.009 0.329 p10/90 [-0.279  0.432]
580 vis 10403 pred 3.411 moge 3.336 diff med mean std 0.026 0.036 0.079 p10/90 [-0.036  0.134]
600 vis 10150 pred 3.447 moge 3.366 diff med mean std 0.034 0.048 0.082 p10/90 [-0.039  0.163]
650 vis 12118 pred 3.447 moge 3.639 …[+157 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json,numpy as np,h5py,cv2
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); iw=np.linalg.inv(Twc); g=json.load(open('/work/inputs/ground_plane.json')); n=np.array(g['plane'][:3]); d=g['plane'][3]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,100,200,300,400,500,580,600,650,699]:
  ys,xs=np.where(f['object'][i]&~f['person'][i]); ys=xs[:0]
  ys,xs=np.where(f['object'][i]&~f['person'][i]); pix=np.stack([xs,ys,np.ones_like(xs)]); rays=iw[:3,:3]@(np.linalg.inv(K)@pix); t=-d/(n@rays); pts=Twc[:3,3,None]+t[None]*rays; zc=(iw[:3,:3]@pts+iw[:3,3,None])[2]
  print(i,'floor z median',np.median(zc),'range',np.percentile(zc,[10,90]))
PY"
```
<details><summary>output</summary>

```
0 floor z median 4.093767572191409 range [3.66105827 4.68078299]
100 floor z median 4.090522522821277 range [3.65996631 4.66770377]
200 floor z median 4.074305027428668 range [3.66096677 4.6236393 ]
300 floor z median 4.734180576435568 range [3.78075975 5.90711136]
400 floor z median 4.87830715907123 range [3.7802329  6.36069831]
500 floor z median 4.570134444048266 range [3.7664359  5.99993939]
580 floor z median 3.746236027557492 range [3.42367336 4.32406264]
600 floor z median 3.7436313089604765 range [3.42233866 4.27495994]
650 floor z median 3.822805399471719 range [3.43585841 4.37770098]
699 floor z median 3.8239689058784605 range [3.43546289 4.37859737]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 print('\\n',p)
 d=torch.load(p,map_location='cpu',weights_only=False)
 print(type(d))
 if hasattr(d,'keys'):
  for k,v in d.items(): print(k,type(v),getattr(v,'shape',None), str(v)[:300] if not hasattr(v,'shape') else '')
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
<class 'dict'>
schema <class 'str'> None cari4d.mhr_wild_inference.v1
frames <class 'list'> None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029',
frame_meta <class 'list'> None [{'frame': '000000', 'src_frame': 0, 'kid': 0}, {'frame': '000001', 'src_frame': 1, 'kid': 0}, {'frame': '000002', 'src_frame': 2, 'kid': 0}, {'frame': '000003', 'src_frame': 3, 'kid': 0}, {'frame': '000004', 'src_frame': 4, 'kid': 0}, {'frame': '000005', 'src_frame': 5, 'kid': 0}, {'frame': '000006
kid <class 'int'> None 0
checkpoint  …[+7253 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 d=torch.load(p,map_location='cpu',weights_only=False)
 print('\\n',p)
 for grp in ['pr','pr_initial','in','observations','raw','metadata','postopt']:
  print('GROUP',grp)
  o=d.get(grp,{})
  if isinstance(o,dict):
   for k,v in o.items():
    if isinstance(v,(np.ndarray,torch.Tensor)): print(' ',k,getattr(v,'shape',None),getattr(v,'dtype',None), np.nanmin(v) if getattr(v,'size',0) else '')
    elif isinstance(v,(list,tuple)): print(' ',k,type(v).__name__,len(v),str(v[:2])[:200])
    else: print(' ',k,type(v).__name__,str(v)[:200])
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
GROUP pr
  pose_abs (700, 4, 4) float32 -0.9966729
  pose_abs_1st_delta (700, 4, 4) float32 -0.9966729
  mhr_global_rot6d (700, 6) float32 -0.9984316
  mhr_trans (700, 3) float32 -0.60053784
  mhr_body_pose_cont (700, 260) float32 -1.0109951
  mhr_hand (700, 108) float32 -0.514939
  mhr_shape (700, 45) float32 -2.413293
  mhr_scale (700, 28) float32 -2.4101858
  mhr_face (700, 72) float32 0.0
  contact_logits (700, 2) float32 -28.055689
GROUP pr_initial
  pose_abs (700, 4, 4) float32 -0.9966729
  pose_abs_1st_delta (700, 4, 4) float32 -0.9966729
  mhr_global_rot6d (700, 6) float32 -0.9984316
  mhr_trans (700, 3) float32 -0.60053784
  mhr_body_pose_cont (700, 260) float32 -1.0109951
  mhr_hand (700, 108) float32 -0.514939
  mhr_shape (700, 45) float32 -2.41 …[+1522 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import fftconvolve
p=np.zeros((5,5),bool); p[2,2]=1;p[2,3]=1
t=np.zeros((5,5),bool); t[1,2]=1;t[1,3]=1
c=fftconvolve(t.astype(int),p[::-1,::-1].astype(int),mode='full')
print(c.astype(int));
# predicted shifted dy=-1 dx=0 intersects 2, corr index?
PY"
```
<details><summary>output</summary>

```
[[0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_shift.py <<'PY'
import sys,os,json,time,cv2,h5py,numpy as np
from scipy.signal import fftconvolve
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
r=np.load('/work/inputs/reconstruction.npz'); Ts=r['T_world_mesh']
scale=.25; W=int(cam['width']*scale); H=int(cam['height']*scale); Ks=K.copy(); Ks[:2]*=scale
ren=Renderer('/work/inputs/object_mesh.glb',width=W,height=H)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
out=[]; shifts=[]; t0=time.time()
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(len(Ts)):
  rm=ren.render(iw@Ts[i],Ks)
  pred=rm['mask'].cpu().numpy()
  person=cv2.resize(f['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  obj=cv2.resize(f['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  predv=pred&~person; targ=obj&~person
  # centroid dx/dy to move pred to target
  ys,xs=np.where(predv); yt,xt=np.where(targ)
  cdx=xt.mean()-xs.mean() if len(xs) and len(xt) else 0; cdy=yt.mean()-ys.mean() if len(xs) and len(yt) else 0
  c=fftconvolve(targ.astype(np.float32),predv[::-1,::-1].astype(np.float32),mode='full')
  ap=predv.sum(); at=targ.sum(); best=(-1,None)
  for dy in range(-6,7):
   for dx in range(-6,7):
    inter=c[H-1+dy,W-1 …[+913 chars]
```
<details><summary>output</summary>

```
0 [0.8936705934671378, 0.9179485678688583, -0.4847994732774339, 0.01698966593193063, -1, 0, 724, 772]
100 [0.8567740473180722, 0.8567740473180722, 0.5314595362264924, 0.3761757310485905, 0, 0, 672, 767]
200 [0.790025959961905, 0.8284179313732597, 0.8271252680362693, 1.400437504323179, 0, 2, 632, 732]
300 [0.7854542812715611, 0.7897931731705808, 0.1257855525582272, 1.4318017082265158, 0, 2, 652, 821]
400 [0.8943890573454386, 0.8943890573454386, 0.3532211725749903, 0.5104450720685634, 0, 0, 544, 604]
500 [0.8153031377379666, 0.8153031377379666, 1.3512589684580973, 1.0313726817381905, 0, 0, 332, 356]
600 [0.8667598487013211, 0.8667598487013211, 0.5360994478641317, 0.4259100729688896, 0, 0, 629, 702]
time 58.662309646606445
IoU base/best 0.8558505322696908 0.8666830640725688 shifts med [0. 0.] …[+418 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/gen_candidates.py <<'PY'
import numpy as np, json
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; F=len(T)
# static means
def mean_pose(ids):
 R=Rot.from_matrix(T[ids,:3,:3]); Rm=R.mean().as_matrix(); p=np.median(T[ids,:3,3],axis=0)
 out=np.eye(4); out[:3,:3]=Rm; out[:3,3]=p; return out
S0=mean_pose(np.arange(0,211)); S1=mean_pose(np.arange(610,700))
print('S0',S0); print('S1',S1)
q=Rot.from_matrix(T[:,:3,:3]).as_quat()
# hemisphere align
for i in range(1,F):
 if np.dot(q[i],q[i-1])<0: q[i]*=-1
# fill static q constant then smooth moving only via full filter
for sigma in [0.75,1.0,1.5,2.0,2.5,3.0]:
 p=gaussian_filter1d(T[:,:3,3],sigma=sigma,axis=0,mode='nearest')
 qs=gaussian_filter1d(q,sigma=sigma,axis=0,mode='nearest'); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
 # static average overwrite
 q0=Rot.from_matrix(S0[:3,:3]).as_quat(); q1=Rot.from_matrix(S1[:3,:3]).as_quat()
 if np.dot(q0,qs[0])<0:q0=-q0
 if np.dot(q1,qs[-1])<0:q1=-q1
 qs[:211]=q0; qs[610:]=q1; p[:211]=S0[:3,3]; p[610:]=S1[:3,3]
 C=np.tile(np.eye(4),(F,1,1)); C[:,:3,:3]=Rot.from_quat(qs).as_matrix(); C[:,:3,3]=p
 np.save(f'/work/scratch/diag/cand_s{sigma}.npy',C.astype(np.float32))
 print(sigma,'static p std base',np.std(T[:211,:3,3],0),'rot angle base-mean med',np.median((Rot.from_matrix(T[:211,:3,:3]).inv()*Rot.from_matrix(S0[:3,:3])).magnitude()*180/np.pi))
 …[+42 chars]
```
<details><summary>output</summary>

```
S0 [[-0.99939622 -0.03422885  0.00596547 -0.16649526]
 [ 0.03418086 -0.9993839  -0.00796846  1.1505084 ]
 [ 0.00623454 -0.00775975  0.99995046  2.67283654]
 [ 0.          0.          0.          1.        ]]
S1 [[ 0.97569235  0.04367678 -0.21474816  0.29073605]
 [-0.04375488  0.99903264  0.00439224  1.19020891]
 [ 0.21473226  0.00511081  0.97665958  2.76929259]
 [ 0.          0.          0.          1.        ]]
0.75 static p std base [0.00886124 0.01025493 0.03641232] rot angle base-mean med 0.8052217206133594
1.0 static p std base [0.00886124 0.01025493 0.03641232] rot angle base-mean med 0.8052217206133594
1.5 static p std base [0.00886124 0.01025493 0.03641232] rot angle base-mean med 0.8052217206133594
2.0 static p std base [0.00886124 0.01025493 0.03641232] rot angle base-mean med 0. …[+217 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; F=700
# create static-only sigma0
from scipy.spatial.transform import Rotation as Rot
def mean_pose(ids):
 o=np.eye(4); o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix(); o[:3,3]=np.median(T[ids,:3,3],axis=0); return o
S0=mean_pose(np.arange(0,211)); S1=mean_pose(np.arange(610,700)); C=T.copy(); C[:211]=S0; C[610:]=S1; np.save('/work/scratch/diag/cand_s0.npy',C)
PY
cat > /work/scratch/eval_cand.py <<'PY'
import sys,os,json,time,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
path=sys.argv[1]
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
Ts=np.load(path).astype(np.float32)
scale=.25; W=int(cam['width']*scale); H=int(cam['height']*scale); Ks=K.copy(); Ks[:2]*=scale
ren=Renderer('/work/inputs/object_mesh.glb',width=W,height=H)
out=[]; t0=time.time()
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(len(Ts)):
  pred=ren.render(iw@Ts[i],Ks)['mask'].cpu().numpy()
  person=cv2.resize(f['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  obj=cv2.resize(f['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  predv=pred&~person; targ=obj&~person
  inter=(predv&targ).sum(); union=(predv|targ).sum(); iou=inter/union if union else 1
  ys,xs=np.where(predv); yt,xt=np.where(t …[+522 chars]
```
<details><summary>output</summary>

```
/work/scratch/diag/cand_s0.npy time 53.68644404411316 mean 0.863767559651209
0 211 0.8852376158004932 [0.24781286 0.32070561] 0.9180235615868078
211 610 0.8414848329084127 [0.45670196 0.78037465] 0.8802736858415533
610 700 0.9122189610165051 [0.42886195 0.15213463] 0.9418321397978067
/work/scratch/diag/cand_s1.0.npy time 53.88813877105713 mean 0.8642086366010332
0 211 0.8852376158004932 [0.24781286 0.32070561] 0.9180235615868078
211 610 0.8422586521186307 [0.45024532 0.78074559] 0.8803699777233356
610 700 0.9122189610165051 [0.42886195 0.15213463] 0.9418321397978067
/work/scratch/diag/cand_s2.0.npy time 53.80200242996216 mean 0.8644795084443008
0 211 0.8852376158004932 [0.24781286 0.32070561] 0.9180235615868078
211 610 0.8427338658787493 [0.44845964 0.77962342] 0.8805627592309452
610 700 0 …[+348 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; kp=r['human_keypoints_world']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); tree=cKDTree(np.array(m.vertices)[::5])
out=[]
for i in range(len(T)):
 inv=np.linalg.inv(T[i]); p=kp[i]@inv[:3,:3].T+inv[:3,3]
 d,_=tree.query(p,workers=-1); out.append([d[21:42].min(),d[42:63].min(),d.min()])
out=np.array(out); np.save('/work/scratch/diag/hand_obj_dist.npy',out)
for i in range(0,700,10): print(i,*np.round(out[i],3))
PY"
```
<details><summary>output</summary>

```
0 0.561 0.797 0.379
10 0.55 0.806 0.399
20 0.568 0.779 0.317
30 0.551 0.76 0.26
40 0.502 0.702 0.247
50 0.485 0.684 0.26
60 0.488 0.694 0.242
70 0.473 0.731 0.179
80 0.528 0.751 0.265
90 0.534 0.684 0.098
100 0.536 0.646 0.127
110 0.509 0.587 0.107
120 0.527 0.562 0.047
130 0.543 0.513 0.088
140 0.481 0.433 0.066
150 0.299 0.248 0.125
160 0.088 0.04 0.04
170 0.003 0.001 0.001
180 0.003 0.003 0.003
190 0.004 0.002 0.002
200 0.008 0.004 0.004
210 0.001 0.014 0.001
220 0.002 0.007 0.002
230 0.005 0.004 0.004
240 0.005 0.008 0.005
250 0.004 0.005 0.004
260 0.002 0.012 0.002
270 0.001 0.01 0.001
280 0.004 0.007 0.004
290 0.011 0.005 0.005
300 0.018 0.004 0.004
310 0.042 0.001 0.001
320 0.039 0.001 0.001
330 0.016 0.002 0.002
340 0.007 0.002 0.002
350 0.004 0.003 0.003
360 0.021 0.025 0.021
370  …[+712 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.array(m.vertices)
g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);d=g['plane'][3]
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
for win in [(0,120),(0,150),(0,170),(0,211),(610,700),(630,700)]:
 o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[win[0]:win[1],:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[win[0]:win[1],:3,3],axis=0)
 w=v@o[:3,:3].T+o[:3,3];dist=w@n+d
 print(win,'p',o[:3,3],'dist min/quantiles',dist.min(),np.percentile(dist,[.01,.1,.5]))
PY"
```
<details><summary>output</summary>

```
(0, 120) p [-0.17160079  1.14730763  2.68770695] dist min/quantiles -0.02221644308630477 [-0.02190912 -0.02015228 -0.01240167]
(0, 150) p [-0.17001382  1.1495676   2.68102646] dist min/quantiles -0.023672237876168456 [-0.02338253 -0.02169578 -0.01411359]
(0, 170) p [-0.16884536  1.14904058  2.67974544] dist min/quantiles -0.023222527129008963 [-0.0229478  -0.02129659 -0.01371617]
(0, 211) p [-0.16649526  1.1505084   2.67283654] dist min/quantiles -0.024390984989173248 [-0.02413538 -0.02265037 -0.01657227]
(610, 700) p [0.29073605 1.19020891 2.76929259] dist min/quantiles -0.02003909336898979 [-0.01921917 -0.01525081 -0.00989218]
(630, 700) p [0.29035923 1.1896137  2.7721169 ] dist min/quantiles -0.02101657550707725 [-0.02015091 -0.01625218 -0.01077773]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];ren=Renderer('/work/inputs/object_mesh.glb')
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,100,300,500,600,699]:
 for _ in range(i): pass
 # inefficient reopen seek
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); R=ren.render(iw@T[i],K); rgb=(R['rgb'].detach().cpu().numpy()*255).astype(np.uint8);m=R['mask'].detach().cpu().numpy()
 out=im.copy();out[m]=rgb[m]
 # bbox around mask expanded
 ys,xs=np.where(m);x0=max(0,xs.min()-20);x1=min(im.shape[1],xs.max()+20);y0=max(0,ys.min()-20);y1=min(im.shape[0],ys.max()+20)
 crop=np.hstack([im[y0:y1,x0:x1],out[y0:y1,x0:x1]])
 cv2.imwrite(f'/work/scratch/diag/texcmp{i}.jpg',crop)
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
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();m.visual.material.baseColorTexture.resize((1001,964)).save('/work/scratch/diag/texture.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];ren=Renderer('/work/inputs/object_mesh.glb')
frames=[]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
flips={'id':np.eye(3),'x':Rot.from_euler('x',180,degrees=True).as_matrix(),'y':Rot.from_euler('y',180,degrees=True).as_matrix(),'z':Rot.from_euler('z',180,degrees=True).as_matrix()}
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,50,100,150,600,650,699]:
  print('frame',i)
  obj=f['object'][i];per=f['person'][i];tg=obj&~per
  for n,Rf in flips.items():
   C=T[i].copy();C[:3,:3]=C[:3,:3]@Rf
   R=ren.render(iw@C,K);pm=R['mask'].cpu().numpy();p=pm&~per;inter=(p&tg).sum();un=(p|tg).sum();iou=inter/un
   # texture gradient NCC inside target & pred
   img=frames[i].astype(np.float32).mean(2); renrgb=(R['rgb'].cpu().numpy()*255).mean(2)
   m=tg&p
   x=img[m];y=renrgb[m]
   if len(x)>10:
    xs=(x-x.mean())/(x.std()+1e-6);ys=(y-y.mean())/(y.std()+1e-6); ncc=(xs*ys).mean()
   else:ncc=0
   print(n,round(iou,4),round(ncc,3))
PY"
```
<details><summary>output</summary>

```
frame 0
id 0.8741 0.103
x 0.7528 -0.011
y 0.8909 0.173
z 0.7499 -0.001
frame 50
id 0.853 0.084
x 0.7483 -0.01
y 0.905 0.153
z 0.7512 0.024
frame 100
id 0.8749 0.118
x 0.747 0.014
y 0.8056 0.051
z 0.721 -0.032
frame 150
id 0.89 0.162
x 0.7531 0.029
y 0.845 0.079
z 0.734 0.032
frame 600
id 0.8907 0.097
x 0.7463 0.069
y 0.7884 0.065
z 0.7069 0.027
frame 650
id 0.9139 0.149
x 0.7567 0.045
y 0.8089 0.049
z 0.7035 0.015
frame 699
id 0.8995 0.147
x 0.7522 0.031
y 0.7912 0.055
z 0.6936 0.01

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];ren=Renderer('/work/inputs/object_mesh.glb')
Rf=Rot.from_euler('y',180,degrees=True).as_matrix();cap=cv2.VideoCapture('/work/inputs/video.mp4');frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
imgs=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,50,100,150,600,650,699]:
  im=frames[i].copy(); m0=ren.render(iw@T[i],K)['mask'].cpu().numpy(); C=T[i].copy();C[:3,:3]=C[:3,:3]@Rf;m1=ren.render(iw@C,K)['mask'].cpu().numpy()
  ov=im.copy();ov[m0]=(.35*ov[m0]+.65*np.array([0,255,0])).astype(np.uint8);ov[m1]=(.35*ov[m1]+.65*np.array([255,0,255])).astype(np.uint8)
  targ=f['object'][i].astype(bool);cv2.drawContours(ov,cv2.findContours(targ.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),2)
  ys,xs=np.where(targ|m0|m1);x0=max(0,xs.min()-20);x1=min(im.shape[1],xs.max()+20);y0=max(0,ys.min()-20);y1=min(im.shape[0],ys.max()+20);crop=ov[y0:y1,x0:x1];crop=cv2.resize(crop,(crop.shape[1]*2,crop.shape[0]*2));cv2.putText(crop,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);imgs.append(crop)
 # concat variable sizes into 4x2 with padding
 maxw=max …[+284 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_static_windows.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
scale=.25; W=int(cam['width']*scale); H=int(cam['height']*scale); Ks=K.copy(); Ks[:2]*=scale; ren=Renderer('/work/inputs/object_mesh.glb',width=W,height=H)
def mean_pose(ids):
 o=np.eye(4); o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix(); o[:3,3]=np.median(T[ids,:3,3],axis=0); return o
windows=[(0,120),(0,150),(0,160),(0,180),(0,195),(0,210),(20,180),
         (610,700),(620,700),(630,700),(640,700),(650,700),(610,650),(0,700)]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for a,b in windows:
  P=mean_pose(np.arange(a,b)); vals=[]; areas=[]
  ids=range(0,211) if b<=211 else range(610,700)
  for i in ids:
   pred=ren.render(iw@P,Ks)['mask'].cpu().numpy(); person=cv2.resize(f['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool); obj=cv2.resize(f['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
   p=pred&~person; t=obj&~person; inter=(p&t).sum(); union=(p|t).sum(); vals.append(inter/union if union else 1); areas.append(p.sum()/max(t.sum(),1))
  print((a,b),'p',np.round(P[:3,3],4),'iou mean/med',np.m …[+159 chars]
```
<details><summary>output</summary>

```
(0, 120) p [-0.1716  1.1473  2.6877] iou mean/med 0.8881087459550018 0.8887381275440976 p10 0.8789473684210526 area 0.9175507117170707
(0, 150) p [-0.17    1.1496  2.681 ] iou mean/med 0.8805829159471801 0.882051282051282 p10 0.8735483870967742 area 0.9084804608301825
(0, 160) p [-0.1696  1.1488  2.681 ] iou mean/med 0.8785337233931673 0.8803641092327699 p10 0.871222076215506 area 0.9083581710126711
(0, 180) p [-0.1684  1.1498  2.678 ] iou mean/med 0.8760394174015271 0.8773946360153256 p10 0.8687258687258688 area 0.9082154867559677
(0, 195) p [-0.1676  1.1508  2.6743] iou mean/med 0.879065155406263 0.8807588075880759 p10 0.87146529562982 area 0.9132216927416833
(0, 210) p [-0.1665  1.1504  2.6729] iou mean/med 0.8852376158004932 0.8867924528301887 p10 0.8780804150453956 area 0.918023561586 …[+1071 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_static_clean.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); iw=np.linalg.inv(np.array(cam['T_world_cam']))
scale=.25; W=int(cam['width']*scale); H=int(cam['height']*scale); Ks=K.copy(); Ks[:2]*=scale; ren=Renderer('/work/inputs/object_mesh.glb',width=W,height=H)
def mean_pose(ids):
 o=np.eye(4); o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix(); o[:3,3]=np.median(T[ids,:3,3],axis=0); return o
windows=[(0,100),(0,120),(0,140),(0,150),(0,160),(0,165),(20,150),(50,150),(630,700),(600,700)]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for a,b in windows:
  P=mean_pose(np.arange(a,b)); vals=[]; areas=[]; ids=range(0,166) if b<=211 else range(610,700)
  for i in ids:
   pred=ren.render(iw@P,Ks)['mask'].cpu().numpy(); person=cv2.resize(f['person'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool); obj=cv2.resize(f['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
   p=pred&~person; t=obj&~person; inter=(p&t).sum(); union=(p|t).sum(); vals.append(inter/union if union else 1); areas.append(p.sum()/max(t.sum(),1))
  print((a,b),'p',np.round(P[:3,3],4),'iou',np.mean(vals),np.median(vals),'p10',np.percentile(vals,10),'area' …[+73 chars]
```
<details><summary>output</summary>

```
(0, 100) p [-0.1727  1.1469  2.6872] iou 0.8777534984046057 0.879948914431673 p10 0.8692808223859583 area 0.9119624170362385
(0, 120) p [-0.1716  1.1473  2.6877] iou 0.8897735297214631 0.8923076923076924 p10 0.8814346349745331 area 0.9171335609015864
(0, 140) p [-0.1706  1.1483  2.6833] iou 0.8864504549713069 0.8886043533930857 p10 0.8782552966389062 area 0.9158150362462464
(0, 150) p [-0.17    1.1496  2.681 ] iou 0.8821574124469382 0.8834512017939005 p10 0.8751623376623376 area 0.9079594465153912
(0, 160) p [-0.1696  1.1488  2.681 ] iou 0.8797341882709859 0.8810286756020818 p10 0.8727031928396622 area 0.9079515096295075
(0, 165) p [-0.1691  1.149   2.6798] iou 0.8800768525225101 0.8813018023301274 p10 0.8728700306440753 area 0.9103913214445385
(20, 150) p [-0.1693  1.15    2.6797] iou 0.8 …[+458 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];ren=Renderer('/work/inputs/object_mesh.glb');Rf=Rot.from_euler('y',180,degrees=True).as_matrix()
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);_,im=cap.read()
outs=[]
for nm,C in [('id',T[0]),('yflip',np.r_[np.c_[T[0][:3,:3]@Rf,T[0][:3,3]],[[0,0,0,1]]])]:
 R=ren.render(iw@C,K);rgb=(R['rgb'].cpu().numpy()*255).astype(np.uint8);m=R['mask'].cpu().numpy();o=im.copy();o[m]=rgb[m];outs.append(o)
ys,xs=np.where(m);x0=xs.min()-20;x1=xs.max()+20;y0=ys.min()-20;y1=ys.max()+20
sheet=np.hstack([im[y0:y1,x0:x1],outs[0][y0:y1,x0:x1],outs[1][y0:y1,x0:x1]])
sheet=cv2.resize(sheet,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST);cv2.imwrite('/work/scratch/diag/id_yflip_tex.jpg',sheet)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/render_utils.py /work/scratch/render_utils_save.py && sed -i 's/1-uv\\[...,1\\]\\*2/uv[...,1]*2-1/' /work/scratch/render_utils.py
grep -n 'grid=' /work/scratch/render_utils.py
python - <<'PY'
import sys,json,cv2,numpy as np
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];ren=Renderer('/work/inputs/object_mesh.glb')
cap=cv2.VideoCapture('/work/inputs/video.mp4');_,im=cap.read();R=ren.render(iw@T[0],K);rgb=(R['rgb'].cpu().numpy()*255).astype(np.uint8);m=R['mask'].cpu().numpy();o=im.copy();o[m]=rgb[m]
ys,xs=np.where(m);x0=xs.min()-20;x1=xs.max()+20;y0=ys.min()-20;y1=ys.max()+20;sheet=np.hstack([im[y0:y1,x0:x1],o[y0:y1,x0:x1]]);sheet=cv2.resize(sheet,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST);cv2.imwrite('/work/scratch/diag/uvflip_tex.jpg',sheet)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,numpy as np,h5py
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];ren=Renderer('/work/inputs/object_mesh.glb');Rf=Rot.from_euler('y',180,degrees=True).as_matrix();cap=cv2.VideoCapture('/work/inputs/video.mp4');frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,50,100,150,600,650,699]:
  obj=f['object'][i];per=f['person'][i];tg=obj&~per
  print(i,end=' ')
  for n,Rf0 in [('id',np.eye(3)),('y',Rf)]:
   C=T[i].copy();C[:3,:3]=C[:3,:3]@Rf0;R=ren.render(iw@C,K);p=R['mask'].cpu().numpy()&~per;m=tg&p
   x=frames[i].astype(np.float32).mean(2)[m];y=(R['rgb'].cpu().numpy()*255).mean(2)[m];xs=(x-x.mean())/(x.std()+1e-6);ys=(y-y.mean())/(y.std()+1e-6)
   print(n,'iou',round((p&tg).sum()/(p|tg).sum(),3),'ncc',round((xs*ys).mean(),3),end='; ')
  print()
PY"
```
<details><summary>output</summary>

```
0 id iou 0.874 ncc 0.034; y iou 0.891 ncc 0.017; 
50 id iou 0.853 ncc 0.063; y iou 0.905 ncc 0.029; 
100 id iou 0.875 ncc 0.026; y iou 0.806 ncc 0.022; 
150 id iou 0.89 ncc 0.036; y iou 0.845 ncc 0.039; 
600 id iou 0.891 ncc -0.08; y iou 0.788 ncc 0.007; 
650 id iou 0.914 ncc -0.049; y iou 0.809 ncc -0.016; 
699 id iou 0.9 ncc -0.055; y iou 0.791 ncc -0.005; 

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cp /work/scratch/render_utils_save.py /work/scratch/render_utils.py
python - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];F=len(T)
def mean_pose(ids):
 o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
S0=mean_pose(np.arange(0,120));S1=mean_pose(np.arange(630,700))
for sig in [0,1,1.5,2,2.5]:
 q=Rot.from_matrix(T[:,:3,:3]).as_quat()
 for i in range(1,F):
  if q[i]@q[i-1]<0:q[i]*=-1
 if sig:
  p=gaussian_filter1d(T[:,:3,3],sig,axis=0,mode='nearest');qs=gaussian_filter1d(q,sig,axis=0,mode='nearest');qs/=np.linalg.norm(qs,axis=1,keepdims=True);Rm=Rot.from_quat(qs).as_matrix()
 else: p=T[:,:3,3].copy();Rm=T[:,:3,:3].copy()
 C=np.tile(np.eye(4),(F,1,1));C[:,:3,:3]=Rm;C[:,:3,3]=p
 # blend S0 into frames 0-170, S1 605-699. anchor moving at boundaries from filtered
 # Hard static then small ramp anchored to raw-filtered
 C[:166]=S0
 for i in range(166,171):
  w=(i-165)/6 # at 166 1/6, 170 5/6; 171 full moving
  # interpolate rotation from S0 to filtered at 171
  rr=Rot.from_matrix(np.stack([S0[:3,:3],C[171,:3,:3]]))
  sl=rr[0]*(1-w)+rr[1]*w # Rotation multiplication not scale; use slerp
  sl=Rot.from_matrix(S0[:3,:3]).__class__.from_quat((1-w)*Rot.from_matrix(S0[:3,:3]).as_quat()+w*Rot.from_matrix(C[171,:3,:3]).as_quat())
  C[i,:3,:3]=sl.as_matrix();C[i,:3,3]=( …[+578 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 23, in <module>
TypeError: Argument 'other' has incorrect type (expected scipy.spatial.transform._rotation.Rotation, got float)
Traceback (most recent call last):
  File "/work/scratch/eval_cand.py", line 5, in <module>
    Ts=np.load(path).astype(np.float32)
       ^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/diag/cand2_s0.npy'
Traceback (most recent call last):
  File "/work/scratch/eval_cand.py", line 5, in <module>
    Ts=np.load(path).astype(np.float32)
       ^^^^^^^^^^^^^
  File "/opt/c …[+1625 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];F=len(T)
def mean_pose(ids):
 o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
def qlerp(qa,qb,w):
 if qa@qb<0: qb=-qb
 q=(1-w)*qa+w*qb; return q/np.linalg.norm(q)
S0=mean_pose(np.arange(0,120));S1=mean_pose(np.arange(630,700))
for sig in [0,1,1.5,2,2.5]:
 q=Rot.from_matrix(T[:,:3,:3]).as_quat()
 for i in range(1,F):
  if q[i]@q[i-1]<0:q[i]*=-1
 if sig:
  p=gaussian_filter1d(T[:,:3,3],sig,axis=0,mode='nearest');qs=gaussian_filter1d(q,sig,axis=0,mode='nearest');qs/=np.linalg.norm(qs,axis=1,keepdims=True);Rm=Rot.from_quat(qs).as_matrix()
 else: p=T[:,:3,3].copy();Rm=T[:,:3,:3].copy()
 C=np.tile(np.eye(4),(F,1,1));C[:,:3,:3]=Rm;C[:,:3,3]=p
 q0=Rot.from_matrix(S0[:3,:3]).as_quat();q1=Rot.from_matrix(S1[:3,:3]).as_quat()
 C[:166]=S0
 for i in range(166,171):
  w=(i-165)/6
  C[i,:3,:3]=Rot.from_quat(qlerp(q0,Rot.from_matrix(C[171,:3,:3]).as_quat(),w)).as_matrix()
  C[i,:3,3]=(1-w)*S0[:3,3]+w*C[171,:3,3]
 C[610:]=S1
 for i in range(605,610):
  w=(610-i)/5
  C[i,:3,:3]=Rot.from_quat(qlerp(Rot.from_matrix(C[i,:3,:3]).as_quat(),q1,w)).as_matrix()
  C[i,:3,3]=(1-w)*C[i,:3,3]+w*S1[:3,3]
 np.save(f'/work/scratch/diag/cand2_s{sig}.npy',C.astype(np.float32))
print('S0',S0);print('S1',S1)
PY
for s in 0 1 1.5 2 2.5; …[+79 chars]
```
<details><summary>output</summary>

```
S0 [[-0.99899016 -0.04342005  0.01154822 -0.17160079]
 [ 0.04326414 -0.99897348 -0.01342372  1.14730763]
 [ 0.01211922 -0.01291054  0.99984321  2.68770695]
 [ 0.          0.          0.          1.        ]]
S1 [[ 9.75544253e-01  4.50638188e-02 -2.15134058e-01  2.90359229e-01]
 [-4.44510696e-02  9.98981980e-01  7.68803132e-03  1.18961370e+00]
 [ 2.15261499e-01  2.06292422e-03  9.76554264e-01  2.77211690e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
/work/scratch/diag/cand2_s0.npy time 46.45520853996277 mean 0.859976246092523
0 211 0.8748546014320924 [0.47639113 0.4592319 ] 0.9104540085666346
211 610 0.8417551768394862 [0.45683052 0.77926387] 0.8806928354162713
610 700 0.9058748422626621 [0.33411146 0.15765407] 0.9406073949495558
/work/scratch/diag/cand2_s1.npy ti …[+1116 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,numpy as np,h5py
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));base=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cand=np.load('/work/scratch/diag/cand2_s2.npy');ren=Renderer('/work/inputs/object_mesh.glb')
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,100,150,200,250,300,400,500,600,650,699]:
  row=[]
  for n,T in [('b',base),('c',cand)]:
   p=ren.render(iw@T[i],K)['mask'].cpu().numpy()&~f['person'][i]; ys,xs=np.where(p); row += [xs.mean(),ys.mean(),p.sum()]
  t=f['object'][i]&~f['person'][i];yt,xt=np.where(t)
  print(i,'target',round(xt.mean(),2),round(yt.mean(),2),t.sum(),'base d',round(row[0]-xt.mean(),2),round(row[1]-yt.mean(),2),row[2],'cand d',round(row[3]-xt.mean(),2),round(row[4]-yt.mean(),2),row[5])
PY"
```
<details><summary>output</summary>

```
0 target 801.04 811.0 12265 base d 3.72 1.08 11534 cand d 3.49 0.66 11262
100 target 801.28 811.27 12127 base d -0.97 -0.71 10777 cand d 3.54 0.67 11193
150 target 801.14 811.18 11959 base d 1.51 -0.49 11093 cand d 3.4 0.52 11250
200 target 801.58 812.29 11633 base d -2.04 -4.06 10332 cand d -1.38 -4.2 10273
250 target 823.17 788.89 11925 base d 0.64 -5.15 9874 cand d 0.49 -5.41 9830
300 target 813.67 776.66 13070 base d 0.69 -4.56 10632 cand d 0.8 -3.84 10742
400 target 779.28 769.8 9688 base d 0.32 -0.74 8728 cand d -0.25 -0.39 8786
500 target 750.76 779.1 5782 base d -4.08 -3.31 5290 cand d -4.19 -3.67 5200
600 target 687.05 832.69 11286 base d -0.23 -0.56 10260 cand d 0.25 -0.43 10386
650 target 688.84 827.51 13187 base d -0.27 0.12 12191 cand d 0.0 0.89 12352
699 target 688.52 827.45  …[+52 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,numpy as np,h5py
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/scratch/diag/cand2_s2.npy');ren=Renderer('/work/inputs/object_mesh.glb',width=384,height=288);Ks=K.copy();Ks[:2]*=.25
with h5py.File('/work/inputs/masks.h5','r') as f:
 i=250;p=ren.render(iw@T[i],Ks)['mask'].cpu().numpy();person=cv2.resize(f['person'][i].astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool);obj=cv2.resize(f['object'][i].astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool);pv=p&~person;t=obj&~person;ys,xs=np.where(pv);yt,xt=np.where(t);print('quarter',xs.mean(),ys.mean(),xt.mean(),yt.mean(),xs.mean()-xt.mean(),ys.mean()-yt.mean(),pv.sum(),t.sum())
PY"
```
<details><summary>output</summary>

```
quarter 205.47524752475246 195.55280528052805 205.78418230563003 197.17426273458446 -0.3089347808775642 -1.6214574540564115 606 746

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/refine_centroid.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0,'/work/scratch'); from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc)
C0=np.load('/work/scratch/diag/cand2_s2.npy'); ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5
M=400; ds=np.zeros((M,2)); zs=np.zeros(M)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(160,610):
  pred=ren.render(iw@C0[i],Ks)['mask'].cpu().numpy(); person=cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool); obj=cv2.resize(f['object'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool)
  p=pred&~person;t=obj&~person;ys,xs=np.where(p);yt,xt=np.where(t)
  if len(xs) and len(xt): ds[i-160]=(xs.mean()-xt.mean(),ys.mean()-yt.mean())/.5
  zs[i-160]=(iw@C0[i])[2,3]
# Smooth corrections; median/mad reject outliers
for sig in [1,2,3,5]:
 d=gaussian_filter1d(ds,sig,axis=0,mode='nearest')
 C=C0.copy()
 for j,i in enumerate(range(160,610)):
  dx,dy=-d[j,0]*zs[j]/K[0,0],-d[j,1]*zs[j]/K[1,1]
  D=np.eye(4);D[:3,3]=Twc[:3,:3]@np.array([dx,dy,0.])
  C[i]=D@C[i]
 # anchors static; blend camera correction off near boundaries
 for i in list(range(160,166))+list(range(604,610)):
  pass
 C[:160]=C0[:160];C[610:]=C0[610:]
 np.save(f'/work/ …[+320 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/refine_centroid.py", line 11, in <module>
    if len(xs) and len(xt): ds[i-160]=(xs.mean()-xt.mean(),ys.mean()-yt.mean())/.5
                                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~
TypeError: unsupported operand type(s) for /: 'tuple' and 'float'
Traceback (most recent call last):
  File "/work/scratch/eval_cand.py", line 5, in <module>
    Ts=np.load(path).astype(np.float32)
       ^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/diag/cand3_c1.npy'
Traceback (most recent …[+1319 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_centroid.py';s=open(p).read().replace(\"ds[i-160]=(xs.mean()-xt.mean(),ys.mean()-yt.mean())/.5\",\"ds[i-160]=np.array([xs.mean()-xt.mean(),ys.mean()-yt.mean()])/.5\");open(p,'w').write(s)
PY
python /work/scratch/refine_centroid.py
for s in 1 2 3 5; do python /work/scratch/eval_cand.py /work/scratch/diag/cand3_c"'$s.npy; done'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/refine_centroid.py", line 11, in <module>
    if len(xs) and len(xt): ds[i-160]=np.array([xs.mean()-xt.mean(),ys.mean()-yt.mean()])/.5
                            ~~^^^^^^^
IndexError: index 400 is out of bounds for axis 0 with size 400
Traceback (most recent call last):
  File "/work/scratch/eval_cand.py", line 5, in <module>
    Ts=np.load(path).astype(np.float32)
       ^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/diag/cand3_c1.npy'
Traceback (most recent call last):
  File "/work/scratch/ev …[+1282 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/M=400/M=450/' /work/scratch/refine_centroid.py
python /work/scratch/refine_centroid.py
for s in 1 2 3 5; do python /work/scratch/eval_cand.py /work/scratch/diag/cand3_c"'$s.npy; done'
```
<details><summary>output</summary>

```
sig 1 corr pixels mean std [-0.75307771 -2.28774734] [1.25397017 1.96913206] m corr mean [0.0032578  0.01026535]
sig 2 corr pixels mean std [-0.75310919 -2.28791128] [1.23232903 1.95126247] m corr mean [0.00325763 0.01026546]
sig 3 corr pixels mean std [-0.75318903 -2.28823432] [1.20847108 1.93120646] m corr mean [0.00325752 0.01026604]
sig 5 corr pixels mean std [-0.75236193 -2.2886024 ] [1.16221778 1.89119185] m corr mean [0.00325279 0.01026568]
/work/scratch/diag/cand3_c1.npy time 63.33021664619446 mean 0.8779847644954095
0 211 0.8850203563608899 [0.41167944 0.3105321 ] 0.9158002853283114
211 610 0.8679731933609004 [0.28511845 0.28317135] 0.8904639873231299
610 700 0.9058748422626621 [0.33411146 0.15765407] 0.9406073949495558
/work/scratch/diag/cand3_c2.npy time 60.98702335357666 mean 0 …[+796 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,numpy as np,h5py
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/scratch/diag/cand3_c5.npy');ren=Renderer('/work/inputs/object_mesh.glb');cap=cv2.VideoCapture('/work/inputs/video.mp4');frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
with h5py.File('/work/inputs/masks.h5','r') as f:
 ids=[160,180,200,220,240,260,280,300,330,360,400,440,470,500,530,560,590];th=[]
 for i in ids:
  im=frames[i].copy();m=ren.render(iw@T[i],K)['mask'].cpu().numpy();ov=im.copy();ov[m]=(.25*ov[m]+.75*np.array([0,255,0])).astype(np.uint8);obj=f['object'][i];cv2.drawContours(ov,cv2.findContours(obj.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),2)
  ys,xs=np.where(m|obj);x0=max(0,xs.min()-15);x1=min(im.shape[1],xs.max()+15);y0=max(0,ys.min()-15);y1=min(im.shape[0],ys.max()+15);crop=ov[y0:y1,x0:x1];crop=cv2.resize(crop,(350,350));cv2.putText(crop,str(i),(5,28),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2);th.append(crop)
 cols=6;rows=3;sheet=np.zeros((rows*350,cols*350,3),np.uint8)
 for j,im in enumerate(th):
  rr,cc=divmod(j,cols);sheet[rr*350:(rr+1)*350,cc*350:(cc+1)*350]=im
 cv2.imwrite('/work/scratch/diag/cand3_motion.jpg',sheet)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_depth_scale.py <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc);C=np.load('/work/scratch/diag/cand3_c5.npy');ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5
ids=[160,180,200,220,240,260,280,300,330,360,400,440,470,500,530,560,590];scales=[1.0,1.01,1.02,1.03,1.04,1.05,1.06,1.08,1.10]
res={s:[] for s in scales}
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in ids:
  person=cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);obj=cv2.resize(f['object'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);tg=obj&~person
  for s in scales:
   S=np.eye(4);S[:3,:3]/=s;Cc=S@iw@C[i];pred=ren.render(Cc,Ks)['mask'].cpu().numpy()&~person;inter=(pred&tg).sum();un=(pred|tg).sum();res[s].append((inter/un,pred.sum()/max(tg.sum(),1)))
  if i%40==0:print(i,flush=True)
for s in scales:
 a=np.array(res[s]);print(s,'iou',a[:,0].mean(),'area',a[:,1].mean())
PY
python /work/scratch/test_depth_scale.py"
```
<details><summary>output</summary>

```
160
200
240
280
360
400
440
560
1.0 iou 0.8630964283439935 area 0.8890414700889655
1.01 iou 0.8630964283439935 area 0.8890414700889655
1.02 iou 0.8630964283439935 area 0.8890414700889655
1.03 iou 0.8630964283439935 area 0.8890414700889655
1.04 iou 0.8630964283439935 area 0.8890414700889655
1.05 iou 0.863078165838743 area 0.889023104953339
1.06 iou 0.8630964283439935 area 0.8890414700889655
1.08 iou 0.8630964283439935 area 0.8890414700889655
1.1 iou 0.8630964283439935 area 0.8890414700889655

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
C=np.load('/work/scratch/diag/cand3_c5.npy');cam=json.load(open('/work/inputs/camera.json'));iw=np.linalg.inv(np.array(cam['T_world_cam']))
for s in [1,1.1]:
 S=np.eye(4);S[:3,:3]/=s;Cc=S@iw@C[300];print(s,Cc[:3,3],np.linalg.det(Cc[:3,:3]))
PY"
```
<details><summary>output</summary>

```
1 [0.22900927 0.96702911 3.74244268] 1.000000022675162
1.1 [0.20819025 0.87911737 3.40222061] 0.7513148179377623

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));Ks=np.array(cam['K']);Ks[:2]*=.5;iw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.load('/work/scratch/diag/cand3_c5.npy');ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576)
for s in [1,1.05,1.1,1.2]:
 S=np.eye(4);S[:3,:3]/=s;Cc=S@iw@C[300];m=ren.render(Cc,Ks)['mask'];print(s,m.sum().item(),Cc[:3,3])
PY"
```
<details><summary>output</summary>

```
1 2707 [0.22900927 0.96702911 3.74244268]
1.05 2707 [0.21810407 0.9209801  3.56423112]
1.1 2707 [0.20819025 0.87911737 3.40222061]
1.2 2707 [0.19084106 0.80585759 3.11870223]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/test_depth_scale.py';s=open(p).read();s=s.replace(\"S=np.eye(4);S[:3,:3]/=s;Cc=S@iw@C[i];pred=ren.render(Cc,Ks)['mask'].cpu().numpy()&~person\", \"Cc=iw@C[i];Cc[:3,3]/=s;pred=ren.render(Cc,Ks)['mask'].cpu().numpy()&~person\");open(p,'w').write(s)
PY
python /work/scratch/test_depth_scale.py"
```
<details><summary>output</summary>

```
160
200
240
280
360
400
440
560
1.0 iou 0.8630964283439935 area 0.8890414700889655
1.01 iou 0.8756620679254081 area 0.9078418672835702
1.02 iou 0.8830480050431184 area 0.9243612477850474
1.03 iou 0.8882804236529144 area 0.9417738201921563
1.04 iou 0.8919557375670927 area 0.9592381645535868
1.05 iou 0.89174541769874 area 0.9759247586296305
1.06 iou 0.8881920783319477 area 0.9942202101867421
1.08 iou 0.8709216391855947 area 1.0295293659487696
1.1 iou 0.8464672323048903 area 1.0649814941148346

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_depth_static.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc);r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
def mean_pose(ids):
 o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5;scales=[1,1.01,1.02,1.03,1.04,1.05,1.06,1.08]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for a,b,ids in [(0,120,list(range(0,150,10))),(630,700,list(range(630,700,5)))]:
  P=mean_pose(np.arange(a,b));Tcam=iw@P;print('window',a,b)
  for s in scales:
   Cc=Tcam.copy();Cc[:3,3]/=s;vals=[]
   for i in ids:
    person=cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);obj=cv2.resize(f['object'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);tg=obj&~person;pred=ren.render(Cc,Ks)['mask'].cpu().numpy()&~person;vals.append(((pred&tg).sum()/(pred|tg).sum(),pred.sum()/tg.sum()))
   v=np.array(vals);print(s,v[:,0].mean(),v[:,1].mean())
PY
python /work/scratch/test_depth_static.py"
```
<details><summary>output</summary>

```
window 0 120
1 0.8833920071067247 0.9300081185983143
1.01 0.8849107136331102 0.9476988964363892
1.02 0.8794616981514074 0.9640717847146673
1.03 0.8714751972719201 0.9816318733119185
1.04 0.8611911584637677 1.0037874830047584
1.05 0.8527037801722481 1.0194279406768787
1.06 0.8434055893602115 1.0427963598682026
1.08 0.8202212250190454 1.081834231865284
window 630 700
1 0.9205543763813944 0.941721425132263
1.01 0.9183668418093093 0.9590593195872177
1.02 0.9148083934286194 0.9782222555637464
1.03 0.9020288032768204 0.99890605947492
1.04 0.8839948343889116 1.0186773426253066
1.05 0.8673101879348436 1.0366235842541192
1.06 0.8489272985174308 1.0530489579482867
1.08 0.8131935891044415 1.1001958639222855

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_depth_erode.py <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.load('/work/scratch/diag/cand3_c5.npy');ren=Renderer('/work/inputs/object_mesh.glb')
ids=[160,180,200,220,240,260,280,300,330,360,400,440,470,500,530,560,590];scales=[1,1.01,1.02,1.03,1.04,1.05,1.06];res={(s,e):[] for s in scales for e in [0,1,2]}
ker=np.ones((3,3),np.uint8)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in ids:
  obj=f['object'][i].astype(np.uint8);person=f['person'][i]
  for e in [0,1,2]:
   obj2=obj if e==0 else cv2.erode(obj,ker,iterations=e);tg=obj2.astype(bool)&~person
   for s in scales:
    Cc=iw@C[i];Cc[:3,3]/=s;pred=ren.render(Cc,K)['mask'].cpu().numpy()&~person;res[(s,e)].append(((pred&tg).sum()/(pred|tg).sum(),pred.sum()/tg.sum()))
  print(i,flush=True)
for e in [0,1,2]:
 print('erode',e)
 for s in scales:
  a=np.array(res[(s,e)]);print(s,round(a[:,0].mean(),4),round(a[:,1].mean(),4))
PY
python /work/scratch/test_depth_erode.py"
```
<details><summary>output</summary>

```
160
180
200
220
240
260
280
300
330
360
400
440
470
500
530
560
590
erode 0
1 0.8633 0.8908
1.01 0.873 0.9078
1.02 0.8802 0.9249
1.03 0.8853 0.9426
1.04 0.8881 0.96
1.05 0.8875 0.9778
1.06 0.8841 0.995
erode 1
1 0.8756 0.9465
1.01 0.8804 0.9646
1.02 0.8814 0.9827
1.03 0.88 1.0015
1.04 0.8755 1.02
1.05 0.8679 1.039
1.06 0.8577 1.0572
erode 2
1 0.8643 1.0131
1.01 0.8619 1.0325
1.02 0.8562 1.0519
1.03 0.8483 1.0721
1.04 0.8379 1.0919
1.05 0.8249 1.1122
1.06 0.8113 1.1317

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_depth_static_erode.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc);r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
def mean_pose(ids):o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
ren=Renderer('/work/inputs/object_mesh.glb');scales=[1,1.005,1.01,1.015,1.02,1.03];ker=np.ones((3,3),np.uint8)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for a,b,ids in [(0,120,list(range(0,150,10))),(630,700,list(range(630,700,5)))]:
  P=mean_pose(np.arange(a,b));print('window',a,b)
  for e in [0,1]:
   for s in scales:
    vals=[];Cc=iw@P;Cc[:3,3]/=s
    for i in ids:
     obj=f['object'][i].astype(np.uint8);obj2=obj if e==0 else cv2.erode(obj,ker,iterations=e);tg=obj2.astype(bool)&~f['person'][i];pred=ren.render(Cc,K)['mask'].cpu().numpy()&~f['person'][i];vals.append(((pred&tg).sum()/(pred|tg).sum(),pred.sum()/tg.sum()))
    v=np.array(vals);print('e',e,s,round(v[:,0].mean(),4),round(v[:,1].mean(),4))
PY
python /work/scratch/test_depth_static_erode.py"
```
<details><summary>output</summary>

```
window 0 120
e 0 1 0.8749 0.9271
e 0 1.005 0.8753 0.9362
e 0 1.01 0.8728 0.9462
e 0 1.015 0.8698 0.9566
e 0 1.02 0.8672 0.9653
e 0 1.03 0.8605 0.9841
e 1 1 0.8622 0.9714
e 1 1.005 0.8592 0.9809
e 1 1.01 0.8547 0.9913
e 1 1.015 0.8499 1.0022
e 1 1.02 0.8462 1.0114
e 1 1.03 0.838 1.0311
window 630 700
e 0 1 0.9087 0.9388
e 0 1.005 0.9109 0.9481
e 0 1.01 0.911 0.9566
e 0 1.015 0.9114 0.9674
e 0 1.02 0.9109 0.9768
e 0 1.03 0.9013 0.9958
e 1 1 0.9068 0.9826
e 1 1.005 0.9048 0.9924
e 1 1.01 0.9 1.0013
e 1 1.015 0.8925 1.0126
e 1 1.02 0.8845 1.0224
e 1 1.03 0.8676 1.0423

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_curve.py <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.load('/work/scratch/diag/cand3_c5.npy');ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5;scales=np.arange(1.0,1.0501,.005);ker=np.ones((3,3),np.uint8)
out=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(160,610,10):
  obj=cv2.erode(f['object'][i].astype(np.uint8),ker,iterations=1);tg=cv2.resize(obj,(768,576),interpolation=cv2.INTER_NEAREST).astype(bool)&~cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool)
  vals=[]
  for s in scales:
   Cc=iw@C[i];Cc[:3,3]/=s;pred=ren.render(Cc,Ks)['mask'].cpu().numpy()&~cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);vals.append((pred&tg).sum()/(pred|tg).sum())
  j=np.argmax(vals);out.append([i,scales[j],max(vals),*vals]);print(i,scales[j],max(vals),flush=True)
np.save('/work/scratch/diag/depth_curve.npy',np.array(out))
PY
python /work/scratch/depth_curve.py"
```
<details><summary>output</summary>

```
160 1.0099999999999998 0.9075602509078904
170 1.0149999999999997 0.8925092374874034
180 1.0149999999999997 0.8747447243022465
190 1.0199999999999996 0.8627450980392157
200 1.0099999999999998 0.8730756072528224
210 1.0349999999999993 0.873485868102288
220 1.0299999999999994 0.8605916354981299
230 1.0249999999999995 0.8646616541353384
240 1.0349999999999993 0.8632507634882932
250 1.049999999999999 0.8908908908908909
260 1.0399999999999991 0.8808322824716267
270 1.044999999999999 0.8824076946943841
280 1.049999999999999 0.8932275777913362
290 1.0349999999999993 0.8886855748703874
300 1.0399999999999991 0.9134157944814463
310 1.044999999999999 0.9120809614168248
320 1.0299999999999994 0.9100224143451809
330 1.0399999999999991 0.9216650278597182
340 1.049999999999999 0.9155304289990023
350 1.04 …[+906 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/apply_depth_centroid.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.ndimage import gaussian_filter1d
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc)
C=np.load('/work/scratch/diag/cand3_c5.npy').astype(np.float64)
f1=np.load('/work/scratch/diag/depth_curve.npy'); xx=f1[:,0]; ss=f1[:,1]
s=np.ones(len(C));s[160:610]=np.interp(np.arange(160,610),xx,ss);s=gaussian_filter1d(s,5,mode='nearest');s=np.clip(s,1,1.05)
# taper at ends to 1
for i in range(160,166):s[i]=1+(s[i]-1)*(i-159)/7
for i in range(604,610):s[i]=1+(s[i]-1)*(610-i)/7
print('depth scale stats',np.percentile(s[160:610],[0,10,50,90,100]))
C2=C.copy()
for i in range(160,610):
 Ct=iw@C[i];Ct[:3,3]/=s[i];C2[i]=Twc@Ct
# centroid at half res
ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5;M=450;ds=np.zeros((M,2));zs=np.zeros(M)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for idx,i in enumerate(range(160,610)):
  person=cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);obj=cv2.resize(f['object'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);tg=obj&~person
  pred=ren.render(iw@C2[i],Ks)['mask'].cpu().numpy()&~person
  ys,xs=np.where(pred);yt,xt=np.where(tg)
  if len(xs) and len(xt):ds[idx]=np.array([xs.mean()-xt …[+593 chars]
```
<details><summary>output</summary>

```
depth scale stats [1.         1.00140757 1.01741055 1.0467411  1.04994148]
Traceback (most recent call last):
  File "/work/scratch/apply_depth_centroid.py", line 30, in <module>
    np.save(f'/work/scratch/diag/cand4_d{d}+c{sig}.npy',C3.astype(np.float32))
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 542, in save
    file_ctx = open(file, "wb")
               ^^^^^^^^^^^^^^^^
OSError: [Errno 36] File name too long: '/work/scratch/diag/cand4_d[[ 4.68619048e-01  1.69739358e-01]\n [ 5.69127623e-01  2.21883134e-01]\n [ 6.99982272e-01  2.97721930e-01]\n [ 8.24279171e-01  3.81101301e-01]\n [ 8.86095264e-01  4.44350029e-01]\n [ 8.33000606e-01  4.60527430e-01]\n [ 6.41105361e-01  4.19485512e-01]\n [ 3.28295089e-01  3.35974130e-01]\n [-5.10153605e-02  2.39436256e-01]\n …[+17235 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/cand4_d{d}+c{sig}.npy/cand4_d5+c{sig}.npy/" /work/scratch/apply_depth_centroid.py
python /work/scratch/apply_depth_centroid.py
for s in 2 5 10; do python /work/scratch/eval_cand.py /work/scratch/diag/cand4_d5+c$s.npy; done'
```
<details><summary>output</summary>

```
depth scale stats [1.         1.00140757 1.01741055 1.0467411  1.04994148]
centroid sig 2 corr px [0.0220504  0.21760595] std [0.38544502 0.52341642]
centroid sig 5 corr px [0.0210478  0.21609549] std [0.32819761 0.47207998]
centroid sig 10 corr px [0.02299035 0.21510238] std [0.29207969 0.44064816]
/work/scratch/diag/cand4_d5+c2.npy time 55.298152923583984 mean 0.8957666975164172
0 211 0.8878504310723097 [0.406594   0.29549179] 0.9224623203823734
211 610 0.8976729611568799 [0.24508557 0.20789625] 0.9289461902377388
610 700 0.9058748422626621 [0.33411146 0.15765407] 0.9406073949495558
/work/scratch/diag/cand4_d5+c5.npy time 51.97769832611084 mean 0.895597345256282
0 211 0.8877504576238431 [0.40407981 0.29443216] 0.9225183893009136
211 610 0.8974287200930499 [0.24546687 0.20665075] 0.928860 …[+371 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,numpy as np,h5py
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/scratch/diag/cand4_d5+c2.npy');ren=Renderer('/work/inputs/object_mesh.glb');cap=cv2.VideoCapture('/work/inputs/video.mp4');frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
with h5py.File('/work/inputs/masks.h5','r') as f:
 ids=[160,180,200,220,240,260,280,300,330,360,400,440,470,500,530,560,590];th=[]
 for i in ids:
  im=frames[i].copy();m=ren.render(iw@T[i],K)['mask'].cpu().numpy();ov=im.copy();ov[m]=(.25*ov[m]+.75*np.array([0,255,0])).astype(np.uint8);obj=f['object'][i];cv2.drawContours(ov,cv2.findContours(obj.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),2)
  ys,xs=np.where(m|obj);x0=max(0,xs.min()-15);x1=min(im.shape[1],xs.max()+15);y0=max(0,ys.min()-15);y1=min(im.shape[0],ys.max()+15);crop=ov[y0:y1,x0:x1];crop=cv2.resize(crop,(350,350));cv2.putText(crop,str(i),(5,28),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2);th.append(crop)
 sheet=np.zeros((3*350,6*350,3),np.uint8)
 for j,im in enumerate(th):
  rr,cc=divmod(j,6);sheet[rr*350:(rr+1)*350,cc*350:(cc+1)*350]=im
 cv2.imwrite('/work/scratch/diag/cand4_motion.jpg',sheet)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# reuse depth_curve by replace input file and output
s=open('/work/scratch/depth_curve.py').read().replace(\"cand3_c5.npy\",\"cand4_d5+c2.npy\").replace(\"depth_curve.npy\",\"depth_curve2.npy\")
open('/work/scratch/depth_curve2.py','w').write(s)
PY
python /work/scratch/depth_curve2.py"
```
<details><summary>output</summary>

```
160 1.005 0.9113251411491199
170 1.0099999999999998 0.8904475617902472
180 1.0 0.8723259762308998
190 1.0 0.8639133378469871
200 1.005 0.8695212407282535
210 1.0 0.864765100671141
220 1.0 0.8532296246195469
230 1.0 0.8612618724559024
240 1.0 0.8631899762952929
250 1.0 0.886849132176235
260 1.005 0.8778816199376948
270 1.005 0.8849365914011754
280 1.005 0.8935523114355232
290 1.0 0.9062980030721967
300 1.005 0.9123304951119521
310 1.0099999999999998 0.9167979829814056
320 1.0 0.9124040920716112
330 1.0099999999999998 0.922876667751383
340 1.0 0.9216543375924681
350 1.0099999999999998 0.9304564907275321
360 1.0149999999999997 0.9314456035767511
370 1.0 0.927810650887574
380 1.0099999999999998 0.9241017359709326
390 1.0 0.9296482412060302
400 1.0 0.9401853411962932
410 1.005 0.941358024691358 …[+528 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/scratch/diag/cand4_d5+c2.npy');ren=Renderer('/work/inputs/object_mesh.glb');ker=np.ones((3,3),np.uint8)
def mom(m):
 y,x=np.where(m);c=np.cov(np.stack([x,y]));w,v=np.linalg.eigh(c);ang=np.degrees(np.arctan2(v[1,-1],v[0,-1]));return ang,np.sqrt(np.maximum(w,1e-6))
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [160,180,200,220,240,260,280,300,330,360,400,440,470,500,530,560,590]:
  obj=cv2.erode(f['object'][i].astype(np.uint8),ker,iterations=1).astype(bool)&~f['person'][i];p=ren.render(iw@T[i],K)['mask'].cpu().numpy()&~f['person'][i]
  a1,l1=mom(p);a2,l2=mom(obj);d=(a1-a2+90)%180-90
  print(i,'dang',round(d,2),'axes ratio',np.round(l1/l2,3),'areas',p.sum(),obj.sum())
PY"
```
<details><summary>output</summary>

```
160 dang 0.65 axes ratio [1.039 0.998] areas 11267 11603
180 dang -0.34 axes ratio [1.075 1.002] areas 11026 11099
200 dang -0.89 axes ratio [1.077 0.999] areas 11080 11107
220 dang 1.18 axes ratio [1.093 1.01 ] areas 11048 10922
240 dang -0.2 axes ratio [1.089 1.002] areas 11068 10950
260 dang -1.54 axes ratio [1.058 1.012] areas 11787 12229
280 dang -0.52 axes ratio [1.036 1.006] areas 11995 12771
300 dang -0.43 axes ratio [1.021 1.011] areas 11690 12454
330 dang -0.38 axes ratio [1.008 1.02 ] areas 11403 12008
360 dang 0.41 axes ratio [0.993 1.002] areas 9959 10537
400 dang -0.37 axes ratio [0.998 1.035] areas 9198 9207
440 dang -0.23 axes ratio [0.999 1.021] areas 7239 7131
470 dang 0.62 axes ratio [0.984 1.014] areas 6063 6034
500 dang 0.56 axes ratio [0.95  1.036] areas 5208 5224
530 …[+167 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_centroid.py <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc);C=np.load('/work/scratch/diag/cand4_d5+c2.npy');ren=Renderer('/work/inputs/object_mesh.glb')
M=117+near if False else None
PY
python - <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc);C=np.load('/work/scratch/diag/cand4_d5+c2.npy');ren=Renderer('/work/inputs/object_mesh.glb')
for a,b in [(0,166),(630,700)]:
 errs=[]
 with h5py.File('/work/inputs/masks.h5','r') as f:
  for i in range(a,b):
   p=ren.render(iw@C[i],K)['mask'].cpu().numpy()&~f['person'][i];t=f['object'][i]&~f['person'][i];ys,xs=np.where(p);yt,xt=np.where(t);errs.append([xs.mean()-xt.mean(),ys.mean()-yt.mean()])
 e=np.array(errs);print(a,b,'mean',e.mean(0),'med',np.median(e,0),'std',e.std(0),'mean abs',np.abs(e).mean(0))
PY"
```
<details><summary>output</summary>

```
0 166 mean [3.46814718 0.63911999] med [3.53455465 0.65566613] std [0.58506408 0.15175135] mean abs [3.46814718 0.63911999]
630 700 mean [0.12820397 0.92959733] med [0.15809322 0.92870333] std [0.12265548 0.03424536] mean abs [0.14440199 0.92959733]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/apply_static_centroid.py <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc);C=np.load('/work/scratch/diag/cand4_d5+c2.npy');ren=Renderer('/work/inputs/object_mesh.glb')
def mean_err(a,b):
 e=[]
 with h5py.File('/work/inputs/masks.h5','r') as f:
  for i in range(a,b):
   p=ren.render(iw@C[i],K)['mask'].cpu().numpy()&~f['person'][i];t=f['object'][i]&~f['person'][i];ys,xs=np.where(p);yt,xt=np.where(t);e.append([xs.mean()-xt.mean(),ys.mean()-yt.mean()])
 return np.median(np.array(e),axis=0)
e0=mean_err(0,120);e1=mean_err(630,700);print('errors pix',e0,e1)
C2=C.copy()
def shift(i,e):
 z=(iw@C2[i])[2,3];dx,dy=-e[0]*z/K[0,0],-e[1]*z/K[1,1]
 D=np.eye(4);D[:3,3]=Twc[:3,:3]@np.array([dx,dy,0.]);C2[i]=D@C2[i]
# constant static with ramp; ramps use static error fading out (moving correction near zero)
for i in range(0,166): shift(i,e0)
for i in range(166,176):
 w=(175-i)/10.; shift(i,e0*w)
for i in range(610,700): shift(i,e1)
for i in range(600,610):
 w=(i-599)/10.; shift(i,e1*w)
np.save('/work/scratch/diag/cand5.npy',C2.astype(np.float32))
PY
python /work/scratch/apply_static_centroid.py
python /work/scratch/eval_cand.py /work/scratch/diag/cand5.npy"
```
<details><summary>output</summary>

```
errors pix [3.56311269 0.71728283] [0.15809322 0.92870333]
/work/scratch/diag/cand5.npy time 61.028695821762085 mean 0.8918686019869936
0 211 0.8850031443386686 [0.4507521  0.39607783] 0.9209465206743213
211 610 0.8973506933708424 [0.24632985 0.21163607] 0.9287561196629189
610 700 0.8836603475607826 [0.43425221 0.46287767] 0.9381010479555624

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/diag/iou_base.npy')[:,0]
c=np.load('/work/scratch/diag/cand4_d5+c2_iou.npy')[:,0]
for a,z in [(0,166),(166,610),(610,700),(630,700),(0,700)]:
 print(a,z,'base',b[a:z].mean(),'cand',c[a:z].mean(),'delta',c[a:z].mean()-b[a:z].mean())
print('medians',np.median(b),np.median(c))
PY"
```
<details><summary>output</summary>

```
0 166 base 0.8823912381198574 cand 0.8901946662058015 delta 0.007803428085944031
166 610 base 0.8498273132986962 cand 0.8958009861884895 delta 0.04597367288979326
610 700 base 0.9079274517597071 cand 0.9058748422626621 delta -0.0020526094970449993
630 700 base 0.9132458126273418 cand 0.9062787846899183 delta -0.006967027937423542
0 700 base 0.8650196332727015 cand 0.8957666975164172 delta 0.03074706424371576
medians 0.8782967540774682 0.894840269020597

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# regenerate cand2_s2 with S1 610:700 only (copy of prior generation)
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];F=len(T)
def mean_pose(ids):o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
def qlerp(qa,qb,w):
 if qa@qb<0:qb=-qb
 q=(1-w)*qa+w*qb;return q/np.linalg.norm(q)
S0=mean_pose(np.arange(0,120));S1=mean_pose(np.arange(610,700));sig=2
q=Rot.from_matrix(T[:,:3,:3]).as_quat()
for i in range(1,F):
 if q[i]@q[i-1]<0:q[i]*=-1
p=gaussian_filter1d(T[:,:3,3],sig,axis=0,mode='nearest');qs=gaussian_filter1d(q,sig,axis=0,mode='nearest');qs/=np.linalg.norm(qs,axis=1,keepdims=True);C=np.tile(np.eye(4),(F,1,1));C[:,:3,:3]=Rot.from_quat(qs).as_matrix();C[:,:3,3]=p
q0=Rot.from_matrix(S0[:3,:3]).as_quat();q1=Rot.from_matrix(S1[:3,:3]).as_quat();C[:166]=S0
for i in range(166,171):
 w=(i-165)/6;C[i,:3,:3]=Rot.from_quat(qlerp(q0,Rot.from_matrix(C[171,:3,:3]).as_quat(),w)).as_matrix();C[i,:3,3]=(1-w)*S0[:3,3]+w*C[171,:3,3]
C[610:]=S1
np.save('/work/scratch/diag/cand2_s2_v2.npy',C.astype(np.float32))
print('S1',S1)
PY
# adapt candidate4 script input path and output prefix
sed -e \"s/cand3_c5.npy/cand2_s2_v2.npy/\" -e \"s/cand4_d5+c/cand4v2_d5+c/g\" /work/scratch/apply_depth_centroid.py > /work/scratch/apply_depth_centroid_v2.py
# first need cand3 centroid r …[+538 chars]
```
<details><summary>output</summary>

```
S1 [[ 0.97569235  0.04367678 -0.21474816  0.29073605]
 [-0.04375488  0.99903264  0.00439224  1.19020891]
 [ 0.21473226  0.00511081  0.97665958  2.76929259]
 [ 0.          0.          0.          1.        ]]
sig 1 corr pixels mean std [-0.75324193 -2.29222921] [1.25388519 1.96463431] m corr mean [0.00325862 0.01028463]
sig 2 corr pixels mean std [-0.75317162 -2.29221817] [1.23232737 1.9469849 ] m corr mean [0.00325803 0.01028402]
sig 3 corr pixels mean std [-0.75304916 -2.29235194] [1.20849894 1.92717017] m corr mean [0.00325706 0.01028382]
sig 5 corr pixels mean std [-0.75179344 -2.2925411 ] [1.16229706 1.88743548] m corr mean [0.00325049 0.01028272]
depth scale stats [1.         1.00140757 1.01741055 1.0467411  1.04994148]
centroid sig 2 corr px [-0.58350522 -1.75352464] std [1.3579254   …[+460 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('/work/scratch/diag/cand3_c5.npy');b=np.load('/work/scratch/diag/cand3v2_c5.npy');print('cand3 diff',np.abs(a[160:610]-b[160:610]).max(),np.linalg.norm(a[160:610,:3,3]-b[160:610,:3,3],axis=2).mean())
c=np.load('/work/scratch/diag/cand4_d5+c2.npy');d=np.load('/work/scratch/diag/cand4v2_d5+c2.npy');print('cand4 diff',np.abs(c[160:610]-d[160:610]).max(),np.linalg.norm(c[160:610,:3,3]-d[160:610,:3,3],axis=2).mean())
for i in [160,200,300,400,500,600]:print(i,np.round(c[i,:3,3],4),np.round(d[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/linalg/linalg.py", line 2583, in norm
    return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
numpy.exceptions.AxisError: axis 2 is out of bounds for array of dimension 2

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('/work/scratch/diag/cand3_c5.npy');b=np.load('/work/scratch/diag/cand3v2_c5.npy');print('cand3 diff',np.abs(a[160:610]-b[160:610]).max(),np.linalg.norm(a[160:610,:3,3]-b[160:610,:3,3],axis=1).mean())
c=np.load('/work/scratch/diag/cand4_d5+c2.npy');d=np.load('/work/scratch/diag/cand4v2_d5+c2.npy');print('cand4 diff',np.abs(c[160:610]-d[160:610]).max(),np.linalg.norm(c[160:610,:3,3]-d[160:610,:3,3],axis=1).mean())
for i in [160,200,300,400,500,600]:print(i,np.round(c[i,:3,3],4),np.round(d[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```
cand3 diff 0.030993223 0.00020520421
cand4 diff 0.030773163 0.0023291763
160 [-0.1582  1.146   2.6907] [-0.1588  1.146   2.6907]
200 [-0.1577  1.1483  2.6753] [-0.1558  1.1443  2.6749]
300 [-0.2093  1.0163  2.6603] [-0.2093  1.0137  2.66  ]
400 [-0.0595  0.9404  2.7419] [-0.0588  0.9401  2.7418]
500 [0.0898 0.9894 2.727 ] [0.0891 0.9904 2.7271]
600 [0.2902 1.1986 2.7489] [0.2904 1.1981 2.7489]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'diff -u /work/scratch/apply_depth_centroid.py /work/scratch/apply_depth_centroid_v2.py | head -100'
```
<details><summary>output</summary>

```
--- /work/scratch/apply_depth_centroid.py	2026-10-20 21:41:20.357518257 +0400
+++ /work/scratch/apply_depth_centroid_v2.py	2026-10-20 21:49:09.040529432 +0400
@@ -2,7 +2,7 @@
 from scipy.ndimage import gaussian_filter1d
 sys.path.insert(0,'/work/scratch');from render_utils import Renderer
 cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);iw=np.linalg.inv(Twc)
-C=np.load('/work/scratch/diag/cand3_c5.npy').astype(np.float64)
+C=np.load('/work/scratch/diag/cand2_s2_v2.npy').astype(np.float64)
 f1=np.load('/work/scratch/diag/depth_curve.npy'); xx=f1[:,0]; ss=f1[:,1]
 s=np.ones(len(C));s[160:610]=np.interp(np.arange(160,610),xx,ss);s=gaussian_filter1d(s,5,mode='nearest');s=np.clip(s,1,1.05)
 # taper at ends to 1
@@ -27,5 +27,5 @@
   dx,dy=-d[ …[+332 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
c=np.load('/work/scratch/diag/cand4_d5+c2.npy');v=np.load('/work/scratch/diag/cand2_s2_v2.npy');h=c.copy();h[610:]=v[610:];np.save('/work/scratch/diag/final_candidate.npy',h.astype(np.float32))
print('boundary pose delta',np.linalg.norm(c[609,:3,3]-h[610,:3,3]))
PY
python /work/scratch/eval_cand.py /work/scratch/diag/final_candidate.npy"
```
<details><summary>output</summary>

```
/work/scratch/diag/final_candidate.npy time 47.486154317855835 mean 0.8965823699276256
0 211 0.8878504310723097 [0.406594   0.29549179] 0.9224623203823734
211 610 0.8976729611568799 [0.24508557 0.20789625] 0.9289461902377388
610 700 0.9122189610165051 [0.42886195 0.15213463] 0.9418321397978067

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];S0=Rot.from_matrix(T[:120,:3,:3]).mean();p0=np.median(T[:120,:3,3],0)
for i in range(140,191):
 ang=(S0.inv()*Rot.from_matrix(T[i,:3,:3])).magnitude()*180/np.pi
 print(i,round(ang,3),round(np.linalg.norm(T[i,:3,3]-p0),4),np.round(T[i,:3,3]-p0,3))
PY"
```
<details><summary>output</summary>

```
140 0.502 0.0194 [ 0.007  0.011 -0.014]
141 0.365 0.049 [ 0.007  0.022 -0.043]
142 0.193 0.0278 [ 0.003  0.015 -0.023]
143 0.502 0.0651 [ 0.005  0.026 -0.06 ]
144 0.694 0.0324 [ 0.004  0.014 -0.029]
145 0.601 0.0366 [ 0.006  0.014 -0.033]
146 0.391 0.0126 [ 0.011 -0.     0.006]
147 0.592 0.0413 [ 0.011  0.013 -0.038]
148 0.714 0.0365 [ 0.008  0.011 -0.034]
149 0.599 0.0473 [ 0.005  0.013 -0.045]
150 0.681 0.0267 [ 0.007  0.006 -0.025]
151 0.691 0.0305 [ 0.01   0.006 -0.028]
152 0.219 0.028 [ 0.016 -0.01   0.021]
153 0.348 0.0212 [ 0.011 -0.011  0.015]
154 0.756 0.0329 [ 0.014 -0.016  0.025]
155 0.728 0.0283 [ 0.016 -0.014  0.019]
156 0.558 0.0226 [ 0.016 -0.012  0.01 ]
157 0.832 0.021 [ 0.015 -0.002 -0.015]
158 0.817 0.0268 [ 0.013  0.002 -0.023]
159 1.126 0.024 [ 0.015 -0.006 -0.018]
160  …[+1228 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));base=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];C=np.load('/work/scratch/diag/final_candidate.npy');ren=Renderer('/work/inputs/object_mesh.glb')
with h5py.File('/work/inputs/masks.h5','r') as f:
 print('i base static cand')
 for i in range(145,191):
  tg=f['object'][i]&~f['person'][i];vals=[]
  for T in [base,C,C]:
   p=ren.render(iw@T[i],K)['mask'].cpu().numpy()&~f['person'][i];vals.append((p&tg).sum()/(p|tg).sum())
  # candidate static actually set constant 0-165, so cand at 150 same static. after165 moving.
  print(i,round(vals[0],4),round(vals[1],4))
PY"
```
<details><summary>output</summary>

```
i base static cand
145 0.8918 0.8708
146 0.9045 0.8711
147 0.8938 0.8727
148 0.8908 0.8731
149 0.8812 0.8734
150 0.89 0.874
151 0.8926 0.8743
152 0.9122 0.8736
153 0.9025 0.8741
154 0.9037 0.8783
155 0.8995 0.8779
156 0.8962 0.8774
157 0.8872 0.8781
158 0.8876 0.8773
159 0.873 0.8769
160 0.881 0.9073
161 0.8675 0.9057
162 0.8883 0.9102
163 0.8812 0.9114
164 0.8899 0.9112
165 0.8948 0.9092
166 0.8989 0.9093
167 0.886 0.9015
168 0.8725 0.8933
169 0.8707 0.8914
170 0.8724 0.8883
171 0.8741 0.8799
172 0.8668 0.879
173 0.8706 0.8769
174 0.8583 0.8768
175 0.8538 0.8753
176 0.8635 0.8753
177 0.8612 0.8752
178 0.8738 0.8752
179 0.868 0.8766
180 0.8588 0.8757
181 0.8682 0.873
182 0.8584 0.87
183 0.8583 0.8697
184 0.8569 0.8706
185 0.8595 0.8677
186 0.859 0.8698
187 0.8483 0.8697
188 0.857 0.8714
18 …[+34 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');fr=[]
while True:
 ok,im=cap.read()
 if not ok:break
 fr.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
with h5py.File('/work/inputs/masks.h5','r') as f:
 d=[]
 for i in range(1,700):
  m=f['object'][i]&f['object'][i-1]&~f['person'][i]&~f['person'][i-1]
  # exclude boundary 3 px
  mm=cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
  if mm.sum()<100:d.append(np.nan);continue
  a=fr[i-1][mm].astype(float);b=fr[i][mm].astype(float);d.append(np.mean(np.abs(a-b)))
 d=np.array(d)
for i in range(130,201):print(i,round(d[i-1],2))
print('bounds high')
for a,b in [(0,150),(150,210),(210,250),(250,600),(600,630),(630,699)]:print(a,b,np.nanmedian(d[a:b]),np.nanmean(d[a:b]))
PY"
```
<details><summary>output</summary>

```
130 0.8
131 1.05
132 0.68
133 1.03
134 0.96
135 0.86
136 0.93
137 0.98
138 0.76
139 0.55
140 0.68
141 0.98
142 0.81
143 0.8
144 0.82
145 0.75
146 1.08
147 0.88
148 0.74
149 0.69
150 0.77
151 0.93
152 0.82
153 0.67
154 0.6
155 0.74
156 1.12
157 0.87
158 0.8
159 0.7
160 0.84
161 0.99
162 0.87
163 0.78
164 0.94
165 0.83
166 1.2
167 0.8
168 0.89
169 0.87
170 0.58
171 1.26
172 1.13
173 0.88
174 0.73
175 0.75
176 1.22
177 0.98
178 0.86
179 0.8
180 0.71
181 1.04
182 0.93
183 0.67
184 0.98
185 0.89
186 0.89
187 0.89
188 0.68
189 0.61
190 0.65
191 0.93
192 0.71
193 0.78
194 0.87
195 0.84
196 1.57
197 1.53
198 1.47
199 1.87
200 2.41
bounds high
0 150 0.7653651939942161 0.7886447026709494
150 210 0.8893520889780324 1.3482135084783846
210 250 4.302280276452052 4.417150385291377
250 600 3.0122974087754 …[+114 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.load('/work/scratch/diag/final_candidate.npy');S0=np.load('/work/scratch/diag/cand2_s2_v2.npy')[0];ren=Renderer('/work/inputs/object_mesh.glb')
with h5py.File('/work/inputs/masks.h5','r') as f:
 print('i static current diff')
 for i in range(155,211):
  tg=f['object'][i]&~f['person'][i];vals=[]
  for T in [S0,C[i]]:
   p=ren.render(iw@T,K)['mask'].cpu().numpy()&~f['person'][i];vals.append((p&tg).sum()/(p|tg).sum())
  print(i,round(vals[0],4),round(vals[1],4),round(vals[0]-vals[1],4))
PY"
```
<details><summary>output</summary>

```
i static current diff
155 0.8779 0.8779 0.0
156 0.8774 0.8774 0.0
157 0.8781 0.8781 0.0
158 0.8773 0.8773 0.0
159 0.8769 0.8769 0.0
160 0.8764 0.9073 -0.0309
161 0.8716 0.9057 -0.0341
162 0.8774 0.9102 -0.0328
163 0.8769 0.9114 -0.0346
164 0.877 0.9112 -0.0342
165 0.8768 0.9092 -0.0324
166 0.8772 0.9093 -0.0321
167 0.8747 0.9015 -0.0269
168 0.8716 0.8933 -0.0217
169 0.8747 0.8914 -0.0167
170 0.8766 0.8883 -0.0116
171 0.8767 0.8799 -0.0032
172 0.8771 0.879 -0.0019
173 0.8764 0.8769 -0.0005
174 0.8775 0.8768 0.0007
175 0.8755 0.8753 0.0002
176 0.8748 0.8753 -0.0005
177 0.8745 0.8752 -0.0007
178 0.8755 0.8752 0.0003
179 0.8752 0.8766 -0.0014
180 0.8749 0.8757 -0.0008
181 0.8747 0.873 0.0017
182 0.8748 0.87 0.0048
183 0.8743 0.8697 0.0045
184 0.8749 0.8706 0.0043
185 0.866 0.8677 -0.0017
186 0 …[+637 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
C=np.load('/work/scratch/diag/final_candidate.npy');S=C[0]
for i in range(155,211,5):
 a=(Rot.from_matrix(S[:3,:3]).inv()*Rot.from_matrix(C[i,:3,:3])).magnitude()*180/np.pi
 print(i,'angle',round(a,2),'trans mm',round(np.linalg.norm(C[i,:3,3]-S[:3,3])*1000,1),'delta',np.round((C[i,:3,3]-S[:3,3])*1000,1))
PY"
```
<details><summary>output</summary>

```
155 angle 0.0 trans mm 0.0 delta [0. 0. 0.]
160 angle 0.0 trans mm 13.8 delta [13.4 -1.3  3. ]
165 angle 0.0 trans mm 37.4 delta [ 12.5 -13.2  32.7]
170 angle 1.72 trans mm 15.2 delta [14.7 -1.4 -3.6]
175 angle 2.25 trans mm 25.4 delta [ 12.4   7.  -21.1]
180 angle 2.21 trans mm 18.7 delta [ 14.1   3.7 -11.7]
185 angle 2.38 trans mm 22.6 delta [ 15.1   5.2 -16. ]
190 angle 2.31 trans mm 27.7 delta [ 15.7   8.8 -21.1]
195 angle 2.24 trans mm 22.6 delta [ 14.8   4.3 -16.6]
200 angle 2.97 trans mm 18.7 delta [ 13.9   1.  -12.4]
205 angle 4.91 trans mm 23.4 delta [ 15.7  -5.5 -16.5]
210 angle 6.22 trans mm 26.3 delta [  3.7  -9.5 -24.2]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# quick rerun video only and print diff around end using masks h5 sequential maybe already 100s, but reuse no cache unavailable. Instead inspect candidate pose motion; skip video.
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
p=T[:,:3,3];R=Rot.from_matrix(T[:,:3,:3]);rel=R[:-1].inv()*R[1:];ang=rel.magnitude()*180/np.pi;vel=np.linalg.norm(np.diff(p,axis=0),axis=1)
for i in range(560,631): print(i,round(ang[i-1],2),round(vel[i-1]*100,2))
PY"
```
<details><summary>output</summary>

```
560 0.64 3.47
561 0.62 2.15
562 0.78 2.87
563 0.83 0.79
564 0.64 2.25
565 0.72 0.7
566 0.76 0.64
567 0.3 0.87
568 0.46 0.84
569 1.24 3.43
570 0.46 0.89
571 0.95 0.71
572 0.47 5.35
573 0.58 0.88
574 0.48 0.65
575 1.2 0.63
576 0.87 3.63
577 1.1 0.97
578 1.15 2.54
579 1.18 0.96
580 0.67 1.79
581 0.51 0.97
582 0.38 0.87
583 0.52 0.24
584 0.58 0.95
585 0.06 2.28
586 0.12 0.75
587 0.23 1.23
588 0.18 1.22
589 0.14 0.28
590 0.6 0.79
591 0.6 5.06
592 0.67 1.73
593 0.37 2.73
594 0.14 3.33
595 0.56 3.07
596 0.63 2.5
597 0.46 3.62
598 0.49 1.18
599 0.11 1.42
600 0.45 2.14
601 0.2 2.09
602 0.45 0.68
603 0.43 2.87
604 0.09 1.37
605 0.13 3.18
606 0.41 3.37
607 0.62 1.97
608 0.16 0.14
609 0.62 3.62
610 0.09 1.78
611 0.18 0.4
612 0.13 1.03
613 0.16 2.74
614 0.51 1.96
615 0.22 2.16
616 0.25 2.13
617 0.93 4. …[+182 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');prev=None;ds=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(550,641):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
  if prev is not None:
   m=f['object'][i]&f['object'][i-1]&~f['person'][i]&~f['person'][i-1];m=cv2.erode(m.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
   if m.sum()<100:ds.append(np.nan)
   else:ds.append(np.mean(np.abs(prev[m].astype(float)-g[m].astype(float))))
  prev=g
for j,d in enumerate(ds,551):print(j,round(d,2))
PY"
```
<details><summary>output</summary>

```
551 4.68
552 4.61
553 4.58
554 4.47
555 4.28
556 4.3
557 3.81
558 3.59
559 4.07
560 3.89
561 3.99
562 4.06
563 4.24
564 4.03
565 3.92
566 4.11
567 3.71
568 3.28
569 3.18
570 3.12
571 3.43
572 3.01
573 3.01
574 3.17
575 3.26
576 3.4
577 4.27
578 6.79
579 7.38
580 3.09
581 3.07
582 1.52
583 1.03
584 1.15
585 1.27
586 1.12
587 0.76
588 0.94
589 0.95
590 0.91
591 1.0
592 0.87
593 0.98
594 0.9
595 0.96
596 1.15
597 1.06
598 0.91
599 0.82
600 0.75
601 0.91
602 0.76
603 0.89
604 0.83
605 0.86
606 1.0
607 0.72
608 0.79
609 0.83
610 0.63
611 1.11
612 0.74
613 0.86
614 0.68
615 0.63
616 0.88
617 0.76
618 0.6
619 0.67
620 0.56
621 0.7
622 0.55
623 0.69
624 0.48
625 0.6
626 0.82
627 0.6
628 0.57
629 0.57
630 0.5
631 0.77
632 0.6
633 0.71
634 0.61
635 0.71
636 0.88
637 0.73
638 0.66
639 0.54
640 0.55

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_final_windows2.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5
def mean_pose(ids):o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
ker=np.ones((3,3),np.uint8)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for a,b in [(582,700),(590,700),(600,700),(610,700),(620,700),(630,700),(582,630),(0,700)]:
  P=mean_pose(np.arange(a,b));vals=[];vals1=[];areas=[]
  for i in range(582,700):
   o=f['object'][i].astype(np.uint8);p=f['person'][i];pred=ren.render(iw@P,Ks)['mask'].cpu().numpy()&~p
   for e,arr in [(0,vals),(1,vals1)]:
    t=(o if e==0 else cv2.erode(o,ker,iterations=1)).astype(bool)&~p;arr.append((pred&t).sum()/(pred|t).sum())
   t=o.astype(bool)&~p;areas.append(pred.sum()/t.sum())
  print((a,b),'iou',round(np.mean(vals),4),'erode',round(np.mean(vals1),4),'area',round(np.mean(areas),4),'p',np.round(P[:3,3],4),flush=True)
PY
python /work/scratch/eval_final_windows2.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_final_windows2.py", line 11, in <module>
    o=f['object'][i].astype(np.uint8);p=f['person'][i];pred=ren.render(iw@P,Ks)['mask'].cpu().numpy()&~p
                                                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~
ValueError: operands could not be broadcast together with shapes (576,768) (1152,1536) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_final_windows2.py';s=open(p).read().replace(\"o=f['object'][i].astype(np.uint8);p=f['person'][i];pred=ren.render(iw@P,Ks)['mask'].cpu().numpy()&~p\", \"o=cv2.resize(f['object'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST);p=cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);pred=ren.render(iw@P,Ks)['mask'].cpu().numpy()&~p\");open(p,'w').write(s)
PY
python /work/scratch/eval_final_windows2.py"
```
<details><summary>output</summary>

```
(582, 700) iou 0.9167 erode 0.8838 area 0.9294 p [0.2915 1.1918 2.7611]
(590, 700) iou 0.9179 erode 0.8832 area 0.9306 p [0.2913 1.1912 2.7629]
(600, 700) iou 0.9176 erode 0.8803 area 0.9304 p [0.2912 1.1909 2.7657]
(610, 700) iou 0.9192 erode 0.8787 area 0.9364 p [0.2907 1.1902 2.7693]
(620, 700) iou 0.9195 erode 0.8755 area 0.9393 p [0.2906 1.1897 2.7705]
(630, 700) iou 0.9188 erode 0.8712 area 0.944 p [0.2904 1.1896 2.7721]
(582, 630) iou 0.8963 erode 0.8998 area 0.9074 p [0.2936 1.1978 2.7385]
(0, 700) iou 0.0584 erode 0.0485 area 0.4362 p [-0.1277  1.1371  2.678 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_v3.py <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];F=len(T)
def mean_pose(ids):o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
def qlerp(qa,qb,w):
 if qa@qb<0:qb=-qb
 q=(1-w)*qa+w*qb;return q/np.linalg.norm(q)
S0=mean_pose(np.arange(20,150));S1=mean_pose(np.arange(590,700));q=Rot.from_matrix(T[:,:3,:3]).as_quat()
for i in range(1,F):
 if q[i]@q[i-1]<0:q[i]*=-1
p=gaussian_filter1d(T[:,:3,3],2,axis=0,mode='nearest');qs=gaussian_filter1d(q,2,axis=0,mode='nearest');qs/=np.linalg.norm(qs,axis=1,keepdims=True)
C=np.tile(np.eye(4),(F,1,1));C[:,:3,:3]=Rot.from_quat(qs).as_matrix();C[:,:3,3]=p
q0=Rot.from_matrix(S0[:3,:3]).as_quat();q1=Rot.from_matrix(S1[:3,:3]).as_quat();C[:196]=S0
for i in range(196,202):
 w=(i-195)/7;C[i,:3,:3]=Rot.from_quat(qlerp(q0,Rot.from_matrix(C[202,:3,:3]).as_quat(),w)).as_matrix();C[i,:3,3]=(1-w)*S0[:3,3]+w*C[202,:3,3]
C[582:]=S1
for i in range(577,582):
 w=(582-i)/5;C[i,:3,:3]=Rot.from_quat(qlerp(Rot.from_matrix(C[i,:3,:3]).as_quat(),q1,w)).as_matrix();C[i,:3,3]=(1-w)*C[i,:3,3]+w*S1[:3,3]
np.save('/work/scratch/diag/v3_base.npy',C.astype(np.float32));print('S0',S0);print('S1',S1)
PY
python /work/scratch/build_v3.py"
```
<details><summary>output</summary>

```
S0 [[-0.99905105 -0.04218451  0.01083828 -0.16928893]
 [ 0.0420294  -0.99901601 -0.01416115  1.14996195]
 [ 0.011425   -0.01369219  0.99984098  2.67974544]
 [ 0.          0.          0.          1.        ]]
S1 [[ 0.97564486  0.04148904 -0.21539677  0.29132664]
 [-0.04162514  0.99912566  0.00390632  1.19122982]
 [ 0.21537051  0.00515474  0.9765188   2.76291752]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_initial_windows2.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5
def mean_pose(ids):o=np.eye(4);o[:3,:3]=Rot.from_matrix(T[ids,:3,:3]).mean().as_matrix();o[:3,3]=np.median(T[ids,:3,3],axis=0);return o
ker=np.ones((3,3),np.uint8)
with h5py.File('/work/inputs/masks.h5','r') as f:
 for a,b in [(0,100),(0,120),(0,140),(0,150),(0,160),(0,170),(0,195),(20,150),(50,150),(100,195)]:
  P=mean_pose(np.arange(a,b));vals=[];vals1=[];areas=[]
  for i in range(0,196):
   o=cv2.resize(f['object'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST);p=cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);pred=ren.render(iw@P,Ks)['mask'].cpu().numpy()&~p
   for e,arr in [(0,vals),(1,vals1)]:
    t=(o if e==0 else cv2.erode(o,ker,iterations=1)).astype(bool)&~p;arr.append((pred&t).sum()/(pred|t).sum())
   t=o.astype(bool)&~p;areas.append(pred.sum()/t.sum())
  print((a,b),'iou',round(np.mean(vals),4),'erode',round(np.mean(vals1),4),'area',round(np.mean(areas),4),'p',np.round(P[:3,3],4),flush=True)
PY
python /work/scr …[+30 chars]
```
<details><summary>output</summary>

```
(0, 100) iou 0.8773 erode 0.8419 area 0.928 p [-0.1727  1.1469  2.6872]
(0, 120) iou 0.8837 erode 0.8482 area 0.9293 p [-0.1716  1.1473  2.6877]
(0, 140) iou 0.8887 erode 0.8538 area 0.9263 p [-0.1706  1.1483  2.6833]
(0, 150) iou 0.8913 erode 0.8573 area 0.9237 p [-0.17    1.1496  2.681 ]
(0, 160) iou 0.892 erode 0.8587 area 0.9264 p [-0.1696  1.1488  2.681 ]
(0, 170) iou 0.893 erode 0.8591 area 0.9246 p [-0.1688  1.149   2.6797]
(0, 195) iou 0.8916 erode 0.8589 area 0.923 p [-0.1676  1.1508  2.6743]
(20, 150) iou 0.8921 erode 0.86 area 0.9214 p [-0.1693  1.15    2.6797]
(50, 150) iou 0.8971 erode 0.867 area 0.9233 p [-0.1675  1.1516  2.6762]
(100, 195) iou 0.8928 erode 0.8802 area 0.9181 p [-0.1607  1.1593  2.6569]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
# make v3 centroid
s=open('/work/scratch/refine_centroid.py').read().replace(\"cand2_s2.npy\",\"v3_base.npy\").replace(\"range(160,610)\",\"range(196,582)\").replace(\"M=450\",\"M=386\").replace(\"cand3_c\",\"v3_c\")
open('/work/scratch/refine_centroid_v3.py','w').write(s)
# depth curve v3 input
s=open('/work/scratch/depth_curve.py').read().replace(\"cand3_c5.npy\",\"v3_c5.npy\").replace(\"range(160,610,10)\",\"range(196,582,10)\").replace(\"depth_curve.npy\",\"depth_curve_v3.npy\")
open('/work/scratch/depth_curve_v3.py','w').write(s)
PY
python /work/scratch/refine_centroid_v3.py
python /work/scratch/depth_curve_v3.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/refine_centroid_v3.py", line 11, in <module>
    if len(xs) and len(xt): ds[i-160]=np.array([xs.mean()-xt.mean(),ys.mean()-yt.mean()])/.5
                            ~~^^^^^^^
IndexError: index 386 is out of bounds for axis 0 with size 386
Traceback (most recent call last):
  File "/work/scratch/depth_curve_v3.py", line 3, in <module>
    cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.load('/work/scratch/diag/v3_c5.npy');ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5;scales=np.arange(1.0,1.0501,.005);ker=np.ones((3,3),np.uint8)
                                                                                              …[+360 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '160\\|196\\|range\\|M=' /work/scratch/refine_centroid_v3.py"
```
<details><summary>output</summary>

```
6:M=386; ds=np.zeros((M,2)); zs=np.zeros(M)
8: for i in range(196,582):
11:  if len(xs) and len(xt): ds[i-160]=np.array([xs.mean()-xt.mean(),ys.mean()-yt.mean()])/.5
12:  zs[i-160]=(iw@C0[i])[2,3]
17: for j,i in enumerate(range(196,582)):
22: for i in list(range(160,166))+list(range(604,610)):
24: C[:160]=C0[:160];C[610:]=C0[610:]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_centroid_v3.py';s=open(p).read().replace('i-160','i-196').replace('C[:160]=C0[:160];C[610:]=C0[610:]','C[:196]=C0[:196];C[582:]=C0[582:]');open(p,'w').write(s)
PY
python /work/scratch/refine_centroid_v3.py
python /work/scratch/depth_curve_v3.py"
```
<details><summary>output</summary>

```
sig 1 corr pixels mean std [-0.75604634 -2.41599756] [1.24600614 2.02766405] m corr mean [0.00325893 0.01085437]
sig 2 corr pixels mean std [-0.75509699 -2.41521583] [1.22747047 2.00614299] m corr mean [0.00325444 0.0108501 ]
sig 3 corr pixels mean std [-0.75323782 -2.41396792] [1.20848613 1.98215146] m corr mean [0.00324581 0.01084346]
sig 5 corr pixels mean std [-0.74732346 -2.40965476] [1.17097244 1.93403755] m corr mean [0.0032186  0.01082174]
196 1.0 0.8791923340177961
206 1.0149999999999997 0.866213921901528
216 1.0299999999999994 0.8677517802644964
226 1.0299999999999994 0.8680694722778891
236 1.0399999999999991 0.8765855330819335
246 1.0299999999999994 0.8778969214804566
256 1.0299999999999994 0.8804846103470858
266 1.049999999999999 0.886060229742316
276 1.0399999999999991 0.88502 …[+1089 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/apply_depth_centroid.py').read().replace(\"cand3_c5.npy\",\"v3_c5.npy\").replace(\"depth_curve.npy\",\"depth_curve_v3.npy\").replace(\"range(160,610)\",\"range(196,582)\").replace(\"range(160,610)\",\"range(196,582)\").replace(\"range(160,166)\",\"range(196,202)\").replace(\"range(604,610)\",\"range(576,582)\").replace(\"M=450\",\"M=386\").replace(\"i-160\",\"i-196\").replace(\"cand4_d5+c\",\"v3_final_c\")
open('/work/scratch/apply_depth_centroid_v3.py','w').write(s)
PY
grep -n 'range\\|196\\|582\\|M=' /work/scratch/apply_depth_centroid_v3.py
python /work/scratch/apply_depth_centroid_v3.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/apply_depth_centroid_v3.py", line 7, in <module>
    s=np.ones(len(C));s[160:610]=np.interp(np.arange(196,582),xx,ss);s=gaussian_filter1d(s,5,mode='nearest');s=np.clip(s,1,1.05)
                      ~^^^^^^^^^
ValueError: could not broadcast input array from shape (386,) into shape (450,)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/apply_depth_centroid_v3.py'
```
<details><summary>output</summary>

```
  if len(xs) and len(xt):ds[idx]=np.array([xs.mean()-xt.mean(),ys.mean()-yt.mean()])/.5
  zs[idx]=(iw@C2[i])[2,3]
for sig in [2,5,10]:
 d=gaussian_filter1d(ds,sig,axis=0,mode='nearest');C3=C2.copy()
 for j,i in enumerate(range(196,582)):
  dx,dy=-d[j,0]*zs[j]/K[0,0],-d[j,1]*zs[j]/K[1,1]
  D=np.eye(4);D[:3,3]=Twc[:3,:3]@np.array([dx,dy,0.])
  C3[i]=D@C3[i]
 np.save(f'/work/scratch/diag/v3_final_c{sig}.npy',C3.astype(np.float32))
 print('centroid sig',sig,'corr px',np.mean(d,0),'std',np.std(d,0))

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
base=np.load('/work/scratch/diag/v3_base.npy');C=np.load('/work/scratch/diag/v3_final_c2.npy');print(np.abs(base[582:]-C[582:]).max());print(base[650,:3,3],C[650,:3,3])
PY"
```
<details><summary>output</summary>

```
0.0
[0.29132664 1.1912298  2.7629175 ] [0.29132664 1.1912298  2.7629175 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));P=np.load('/work/scratch/diag/v3_base.npy')[650];ren=Renderer('/work/inputs/object_mesh.glb',width=384,height=288);Ks=K.copy();Ks[:2]*=.25
with h5py.File('/work/inputs/masks.h5','r') as f:
 vals=[]
 for i in range(582,700):
  o=cv2.resize(f['object'][i].astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(f['person'][i].astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool);t=o&~p;q=ren.render(iw@P,Ks)['mask'].cpu().numpy()&~p;vals.append((q&t).sum()/(q|t).sum())
 print(np.mean(vals))
 for i in [582,590,600,610,630,650,699]:
  o=cv2.resize(f['object'][i].astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool);p=cv2.resize(f['person'][i].astype(np.uint8),(384,288),interpolation=cv2.INTER_NEAREST).astype(bool);t=o&~p;q=ren.render(iw@P,Ks)['mask'].cpu().numpy()&~p;print(i,(q&t).sum()/(q|t).sum(),q.sum(),t.sum())
PY"
```
<details><summary>output</summary>

```
0.8914227000606546
582 0.8994334277620396 644 697
590 0.8899858956276445 646 694
600 0.8967921896792189 658 702
610 0.8884664131812421 715 775
630 0.8890214797136038 758 825
650 0.888622754491018 758 819
699 0.8943577430972389 758 820

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/apply_depth_centroid_v3.py';s=open(p).read().replace('s[160:610]','s[196:582]').replace('(i-159)/7','(i-195)/7').replace('(610-i)/7','(582-i)/6');open(p,'w').write(s)
PY
python /work/scratch/apply_depth_centroid_v3.py
for s in 2 5 10; do python /work/scratch/eval_cand.py /work/scratch/diag/v3_final_c"'$s.npy; done'
```
<details><summary>output</summary>

```
depth scale stats [1.00003407 1.00126995 1.01909126 1.04344729 1.04992961]
centroid sig 2 corr px [0.02358732 0.1673693 ] std [0.36811267 0.54891143]
centroid sig 5 corr px [0.02727794 0.16996253] std [0.32596237 0.48191269]
centroid sig 10 corr px [0.03582944 0.17623546] std [0.29085786 0.43093904]
/work/scratch/diag/v3_final_c2.npy time 58.25999426841736 mean 0.8906220730201572
0 211 0.8832631252922872 [0.4654904  0.28294439] 0.9118308770395199
211 610 0.894438727915617 [0.26218478 0.22691825] 0.9237193358359611
610 700 0.8909542137678473 [0.40535824 0.31748884] 0.9270783443213043
/work/scratch/diag/v3_final_c5.npy time 50.29099488258362 mean 0.8906757336050408
0 211 0.8832208742047036 [0.46660959 0.28463266] 0.9117662734081592
211 610 0.8945552125995735 [0.26092538 0.22904821] 0.9238119 …[+370 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/scale=.25/scale=.5/' /work/scratch/eval_cand.py > /work/scratch/eval_cand_half.py
python /work/scratch/eval_cand_half.py /work/scratch/diag/v3_final_c2.npy"
```
<details><summary>output</summary>

```
/work/scratch/diag/v3_final_c2.npy time 57.27320098876953 mean 0.8986042829263641
0 211 0.8908826955892943 [1.06533233 0.07216328] 0.9221047668426925
211 610 0.8980310087461935 [0.09403806 0.09937899] 0.9247467477605674
610 700 0.9192486309931389 [0.25913292 0.09101123] 0.9298483064794616

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/diag/iou_base.npy')[:,0];c=np.load('/work/scratch/diag/v3_final_c2_iou.npy')[:,0]
for a,z in [(0,196),(196,582),(582,700),(0,700)]: print(a,z,'base',b[a:z].mean(),'cand',c[a:z].mean(),'delta',c[a:z].mean()-b[a:z].mean())
PY"
```
<details><summary>output</summary>

```
0 196 base 0.8780437306428491 cand 0.8920921938930135 delta 0.0140484632501644
196 582 base 0.8459848516565769 cand 0.8960184655663718 delta 0.05003361390979488
582 700 base 0.9056527063174064 cand 0.9178796638712258 delta 0.012226957553819418
0 700 base 0.8650196332727015 cand 0.8986042829263641 delta 0.03358464965366259

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/rot_probe.py <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.load('/work/scratch/diag/v3_final_c2.npy');ren=Renderer('/work/inputs/object_mesh.glb',width=768,height=576);Ks=K.copy();Ks[:2]*=.5;ker=np.ones((3,3),np.uint8)
def iou(T,tg,per):
 p=ren.render(iw@T,Ks)['mask'].cpu().numpy()&~per;return (p&tg).sum()/(p|tg).sum()
res=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(200,582,20):
  o=cv2.erode(f['object'][i].astype(np.uint8),ker,iterations=1);o=cv2.resize(o,(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);per=cv2.resize(f['person'][i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);tg=o&~per;T=C[i].copy();base=iou(T,tg,per);cur=base;deltas=np.zeros(3)
  for rnd in range(2):
   for ax in range(3):
    vals=[]
    for d in [-2,-1,0,1,2]:
     R=np.eye(4);R[:3,:3]=Rot.from_euler('xyz',np.eye(3)[ax]*d,degrees=True).as_matrix();vals.append(iou(T@R,tg,per))
    j=int(np.argmax(vals));d=[-2,-1,0,1,2][j]
    if d:
     R=np.eye(4);R[:3,:3]=Rot.from_euler('xyz',np.eye(3)[ax]*d,degrees=True).as_matrix();T=T@R;deltas[ax]+=d;cur=vals[j]
  res.append([i,base,cur,*deltas]);print(res[-1],flush=True)
np.save('/work/scratch/diag/rot_probe.npy',np.array(res))
PY
pyt …[+31 chars]
```
<details><summary>output</summary>

```
[200, 0.8706365503080082, 0.903832505322924, -2.0, -1.0, -1.0]
[220, 0.8532296246195469, 0.9019815994338287, -2.0, -2.0, -1.0]
[240, 0.8627583869874619, 0.9052855622561192, -2.0, 0.0, 0.0]
[260, 0.8779106356198867, 0.9001292824822237, -2.0, 0.0, 0.0]
[280, 0.8883405305367057, 0.8883405305367057, 0.0, 0.0, 0.0]
[300, 0.9090909090909091, 0.9090909090909091, 0.0, 0.0, 0.0]
[320, 0.9105482526450785, 0.9108056265984654, 0.0, 0.0, -1.0]
[340, 0.9245602165087957, 0.9245602165087957, 0.0, 0.0, 0.0]
[360, 0.9218926023282012, 0.9225563909774436, 0.0, -1.0, 0.0]
[380, 0.9176518548715858, 0.9237978810105949, 0.0, 0.0, -1.0]
[400, 0.943500424808836, 0.9475926714955262, 0.0, 1.0, 0.0]
[420, 0.9229357798165138, 0.9378120744439401, 1.0, 1.0, 0.0]
[440, 0.9246794871794872, 0.9457406402604449, 0.0, 2.0, 0.0 …[+432 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.load('/work/scratch/diag/v3_final_c2.npy');res=np.load('/work/scratch/diag/rot_probe.npy');ren=Renderer('/work/inputs/object_mesh.glb');cap=cv2.VideoCapture('/work/inputs/video.mp4');ids=res[:,0].astype(int);frames={}
for i in ids:cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));_,frames[int(i)]=cap.read()
th=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for row in res:
  i=int(row[0]);T=C[i].copy();ang=row[3:]/2 # wait probe deltas summed, not /2
  ang=row[3:]
  R=np.eye(4);R[:3,:3]=Rot.from_euler('xyz',ang,degrees=True).as_matrix();Tc=T@R
  m0=ren.render(iw@T,K)['mask'].cpu().numpy();m1=ren.render(iw@Tc,K)['mask'].cpu().numpy();im=frames[i].copy();ov=im.copy();ov[m0]=(.4*ov[m0]+.6*np.array([0,255,0])).astype(np.uint8);ov[m1]=(.4*ov[m1]+.6*np.array([255,0,255])).astype(np.uint8);o=f['object'][i];cv2.drawContours(ov,cv2.findContours(o.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),2);ys,xs=np.where((m0|m1|o));x0=max(0,xs.min()-10);x1=min(im.shape[1],xs.max()+10);y0=max(0,ys.min()-10);y1=min(im.shape[0],ys.max()+10);crop=ov[y0:y1,x0:x1];crop=cv2.resize(crop,(280,280));cv2.putText(crop,f'{i} {row[3:]}',(3,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(25 …[+245 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/apply_rot_curve.py <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
C=np.load('/work/scratch/diag/v3_final_c2.npy');P=np.load('/work/scratch/diag/rot_probe.npy');frames=P[:,0].astype(int);delta=P[:,3:].astype(float)
# interpolation anchors
xx=np.r_[190,frames,590];dd=np.vstack([np.zeros((1,3)),delta,np.zeros((1,3))])
for sig in [3,6,10]:
 out=C.copy();ang=np.zeros((700,3));ang[196:582]=np.interp(np.arange(196,582),xx,dd,axis=0);ang=gaussian_filter1d(ang,sig,axis=0,mode='nearest')
 for i in range(196,582):
  R=np.eye(4);R[:3,:3]=Rot.from_euler('xyz',ang[i],degrees=True).as_matrix();out[i]=C[i]@R
 np.save(f'/work/scratch/diag/v3_rot_s{sig}.npy',out.astype(np.float32));print(sig,np.percentile(np.linalg.norm(ang[196:582],axis=1),[0,50,90,100]))
PY
python /work/scratch/apply_rot_curve.py
for s in 3 6 10; do python /work/scratch/eval_cand_half.py /work/scratch/diag/v3_rot_s"'$s.npy; done'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/apply_rot_curve.py", line 8, in <module>
    out=C.copy();ang=np.zeros((700,3));ang[196:582]=np.interp(np.arange(196,582),xx,dd,axis=0);ang=gaussian_filter1d(ang,sig,axis=0,mode='nearest')
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: interp() got an unexpected keyword argument 'axis'
Traceback (most recent call last):
  File "/work/scratch/eval_cand_half.py", line 5, in <module>
    Ts=np.load(path).astype(np.float32)
       ^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No suc …[+963 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/apply_rot_curve.py';s=open(p).read().replace(\"ang[196:582]=np.interp(np.arange(196,582),xx,dd,axis=0)\",\"ang[196:582]=np.stack([np.interp(np.arange(196,582),xx,dd[:,j]) for j in range(3)],axis=1)\");open(p,'w').write(s)
PY
python /work/scratch/apply_rot_curve.py
for s in 3 6 10; do python /work/scratch/eval_cand_half.py /work/scratch/diag/v3_rot_s"'$s.npy; done'
```
<details><summary>output</summary>

```
3 [2.78745385e-05 1.02628819e+00 2.63673699e+00 3.88573214e+00]
6 [0.01163131 0.97641783 2.57877987 3.52459445]
10 [0.0813087  1.00004087 2.43355389 3.0675873 ]
/work/scratch/diag/v3_rot_s3.npy time 61.47980833053589 mean 0.9011781874578463
0 211 0.8917954402030995 [1.0745063  0.09601806] 0.9197331282983379
211 610 0.9020639512487618 [0.16883713 0.25772046] 0.9179719265491487
610 700 0.9192486309931389 [0.25913292 0.09101123] 0.9298483064794616
/work/scratch/diag/v3_rot_s6.npy time 60.26279354095459 mean 0.9012573694570479
0 211 0.8918434995814675 [1.0729579  0.09523021] 0.9198825899296657
211 610 0.9021774521775976 [0.16850429 0.25461313] 0.9180779171967642
610 700 0.9192486309931389 [0.25913292 0.09101123] 0.9298483064794616
/work/scratch/diag/v3_rot_s10.npy time 61.30211353302002 mean 0 …[+223 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# centroid on rotated candidate
s=open('/work/scratch/refine_centroid_v3.py').read().replace('v3_base.npy','v3_rot_s10.npy').replace('v3_c','v3r_c');open('/work/scratch/refine_rot_centroid.py','w').write(s)
# depth curve on rotated+centroid
s=open('/work/scratch/depth_curve_v3.py').read().replace('v3_c5.npy','v3r_c5.npy').replace('depth_curve_v3.npy','depth_curve_v3r.npy');open('/work/scratch/depth_curve_v3r.py','w').write(s)
PY
python /work/scratch/refine_rot_centroid.py
python /work/scratch/depth_curve_v3r.py"
```
<details><summary>output</summary>

```
sig 1 corr pixels mean std [-0.10366916 -0.2887953 ] [0.36331354 0.54093767] m corr mean [0.00046031 0.00126134]
sig 2 corr pixels mean std [-0.10333532 -0.2880796 ] [0.35016068 0.52677953] m corr mean [0.00045883 0.00125808]
sig 3 corr pixels mean std [-0.10276446 -0.28731929] [0.34374676 0.51760352] m corr mean [0.0004563  0.00125463]
sig 5 corr pixels mean std [-0.10116734 -0.28554598] [0.3323022  0.50190466] m corr mean [0.00044928 0.00124671]
196 1.0149999999999997 0.9094693028095734
206 1.0199999999999996 0.8982817869415808
216 1.0099999999999998 0.9100731962356222
226 1.0099999999999998 0.9111796982167353
236 1.0199999999999996 0.9164032314717246
246 1.0099999999999998 0.9153682719546742
256 1.0149999999999997 0.9116479152878888
266 1.0149999999999997 0.9042852674382234
276 1.005 0. …[+956 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/apply_depth_centroid_v3.py').read().replace('v3_c5.npy','v3r_c5.npy').replace('depth_curve_v3.npy','depth_curve_v3r.npy').replace('v3_final_c','v3r_final_c');open('/work/scratch/apply_depth_centroid_v3r.py','w').write(s)
PY
python /work/scratch/apply_depth_centroid_v3r.py"
```
<details><summary>output</summary>

```
depth scale stats [1.         1.00007987 1.0045675  1.01588494 1.02179648]
centroid sig 2 corr px [-0.02181342  0.08755776] std [0.10233737 0.20829963]
centroid sig 5 corr px [-0.02030941  0.08820365] std [0.0805834 0.1859267]
centroid sig 10 corr px [-0.01809362  0.08903725] std [0.06651546 0.16570114]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for s in 2 5 10; do python /work/scratch/eval_cand_half.py /work/scratch/diag/v3r_final_c$s.npy; done'
```
<details><summary>output</summary>

```
/work/scratch/diag/v3r_final_c2.npy time 52.30342507362366 mean 0.9069878816910227
0 211 0.8932279715991166 [1.06009511 0.0668117 ] 0.9218884566339735
211 610 0.9114988430749871 [0.08634428 0.08382862] 0.9300483284554872
610 700 0.9192486309931389 [0.25913292 0.09101123] 0.9298483064794616
/work/scratch/diag/v3r_final_c5.npy time 58.77948451042175 mean 0.9069619257593952
0 211 0.8932175912448761 [1.06264179 0.06889419] 0.921927353803793
211 610 0.911458795713096 [0.09181399 0.09202566] 0.929965210438214
610 700 0.9192486309931389 [0.25913292 0.09101123] 0.9298483064794616
/work/scratch/diag/v3r_final_c10.npy time 61.51729655265808 mean 0.9069742307669358
0 211 0.8932382056885452 [1.06282997 0.06960816] 0.9219387029210389
211 610 0.9114694820731569 [0.09312692 0.09679858] 0.9299480890933522 …[+71 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as Rot
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];C=np.load('/work/scratch/diag/v3r_final_c2.npy')
def stats(T,a,b):
 R=Rot.from_matrix(T[a:b,:3,:3]);rel=R[:-1].inv()*R[1:];ang=rel.magnitude()*180/np.pi;p=T[a:b,:3,3];v=np.linalg.norm(np.diff(p,axis=0),axis=1)*100
 return np.median(ang),np.percentile(ang,95),ang.max(),np.median(v),np.percentile(v,95),v.max()
for n,T in [('base',B),('cand',C)]:
 print(n)
 for a,b in [(0,195),(196,582),(582,700)]:print(a,b,np.round(stats(T,a,b),3))
 print('max boundary',np.linalg.norm(C[196,:3,3]-C[195,:3,3]) if n=='cand' else '')
# orientation jumps around boundaries
for i in [195,196,197,580,581,582]:
 rel=(Rot.from_matrix(C[max(0,i-1),:3,:3]).inv()*Rot.from_matrix(C[i,:3,:3])).magnitude()*180/np.pi
 print(i,'step',rel,'vcm',np.linalg.norm(C[i,:3,3]-C[max(0,i-1),:3,3])*100)
# floor dists
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.array(m.vertices);g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);d=g['plane'][3]
for nm,T in [('base0',B[0]),('cand0',C[0]),('base699',B[699]),('cand699',C[699])]:
 w=v@T[:3,:3].T+T[:3,3];dist=w@n+d;print(nm,'plane min mean',dist.min(),dist.mean(),'bbox y',w[:,1].min(),w[:,1].max())
# camera z change moving
cam=json.load(open('/work/inputs/camera.json'));iw=np.linalg.inv(np.array(cam['T_world_cam']))
for a,b i …[+186 chars]
```
<details><summary>output</summary>

```
base
0 195 [0.284 0.616 0.823 1.204 4.109 5.204]
196 582 [ 0.673  1.391  2.022  1.623  7.777 16.597]
582 700 [0.22  0.603 0.93  0.902 3.419 5.059]
max boundary 
cand
0 195 [0. 0. 0. 0. 0. 0.]
196 582 [0.485 0.954 1.879 0.44  1.063 1.978]
582 700 [0. 0. 0. 0. 0. 0.]
max boundary 0.0109969275
195 step 0.0 vcm 0.0
196 step 0.8588973174300389 vcm 1.0996927507221699
197 step 0.42373185147681763 vcm 0.1565419603139162
580 step 0.2501725714000956 vcm 1.8552228808403015
581 step 0.24385529870608016 vcm 1.6932571306824684
582 step 0.35516484597473785 vcm 1.8713327124714851
base0 plane min mean -0.004969800917320022 0.1177557159913294 bbox y 1.072659792714841 1.2421330632695988
cand0 plane min mean -0.02389500573573411 0.09960549135411019 bbox y 1.0960093820051706 1.261925131373789
base699 plane min …[+338 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF world trajectory for one rigid object.

The released baseline contains useful per-frame appearance evidence but has
tracker jitter and depth drift while the object is carried.  This script:

  1. pools the two static phases into one rigid pose per phase;
  2. smoothly interpolates the baseline motion (SE(3)-compatible quaternion
     smoothing + translation smoothing);
  3. aligns the projected mesh silhouette to the SAM2 object mask, excluding
     pixels covered by the person, to remove 2-D trajectory drift;
  4. searches per-frame metric depth along the camera ray, using a mask eroded
     by one pixel to compensate SAM2'"'s approximately one-pixel dilation;
  5. tests small local rotation corrections on a sparse set of anchors and
     applies the resulting smooth rotation curve.

Only the supplied inputs are read.  Rendering is deterministic and uses the
installed nvdiffrast rasterizer.
\"\"\"
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr


# Phase boundaries were verified by frame differencing inside the visible
# object region (excluding the person): the image is static from frames 0-195
# (median consecutive abs-diffe …[+15647 chars]
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
wrote /work/outputs/object_poses_world.npz: (700, 4, 4), elapsed 70.2s
initial static pose:
 [[-0.99905105 -0.04218451  0.01083828 -0.16928893]
 [ 0.0420294  -0.99901601 -0.01416115  1.14996195]
 [ 0.011425   -0.01369219  0.99984098  2.67974544]
 [ 0.          0.          0.          1.        ]]
final static pose:
 [[ 0.97564486  0.04148904 -0.21539677  0.29132664]
 [-0.04162514  0.99912566  0.00390632  1.19122988]
 [ 0.21537051  0.00515474  0.9765188   2.76291752]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
out=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];ref=np.load('/work/scratch/diag/v3r_final_c2.npy')
d=np.linalg.norm(out[:,:3,3]-ref[:,:3,3],axis=1)
print(out.shape,out.dtype,'trans diff max/mean',d.max(),d.mean())
from scipy.spatial.transform import Rotation as Rot
a=(Rot.from_matrix(ref[:,:3,:3]).inv()*Rot.from_matrix(out[:,:3,:3])).magnitude()*180/np.pi
print('rot diff max/mean/med',a.max(),a.mean(),np.median(a))
print('max loc',np.argmax(d),d[np.argmax(d)],'rot max loc',np.argmax(a),a[np.argmax(a)])
PY"
```
<details><summary>output</summary>

```
(700, 4, 4) float64 trans diff max/mean 2.2403807479721473e-05 6.309482026561803e-07
rot diff max/mean/med 2.40583248674873e-06 3.548314842095038e-07 2.1254478099183614e-07
max loc 332 2.2403807479721473e-05 rot max loc 286 2.40583248674873e-06

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
b=np.load('/work/scratch/diag/iou_base.npy');c=np.load('/work/scratch/diag/v3r_final_c2_iou.npy')
phases=[('initial static',0,196),('manipulation',196,582),('final static',582,700)]
print('metric table')
for n,a,z in phases:
 bb=b[a:z];cc=c[a:z]
 print(n,'iou',bb[:,0].mean(),cc[:,0].mean(),'cent full',np.abs(bb[:,1:3]).mean(0),np.abs(cc[:,1:3]).mean(0)*2,'area', (bb[:,3]/bb[:,4]).mean(),(cc[:,3]/cc[:,4]).mean())
# temporal
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];C=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
def temp(T,a,z):
 R=Rot.from_matrix(T[a:z,:3,:3]);rel=R[:-1].inv()*R[1:];ang=rel.magnitude()*180/np.pi;p=T[a:z,:3,3];v=np.linalg.norm(np.diff(p,axis=0),axis=1)*100;return np.median(ang),np.percentile(ang,95),ang.max(),np.median(v),np.percentile(v,95),v.max()
for n,a,z in phases:
 print('temp',n,'base',np.round(temp(B,a,z),3),'cand',np.round(temp(C,a,z),3))
# global angle axis
rel=Rot.from_matrix(C[0,:3,:3]).inv()*Rot.from_matrix(C[-1,:3,:3]);print('global',rel.magnitude()*180/np.pi,rel.as_rotvec()/(rel.magnitude()+1e-12))
# camera z corr
cam=json.load(open('/work/inputs/camera.json'));iw=np.linalg.inv(np.array(cam['T_world_cam']));zb=(iw@B[196:582])[:,2,3];zc=(iw@C[196:582])[:,2,3];print('zcorr',np.median(zc-zb),np.percentile(zc-zb,[10,90]))
# overall
print('overall',b[:,0].mean(),c[:,0].mean())
PY"
```
<details><summary>output</summary>

```
metric table
initial static iou 0.8780437306428491 0.8920921938930135 cent full [2.11662947 0.84330342] [2.27444028 0.13689553] area 0.9189193966095939 0.9213832121601385
manipulation iou 0.8459848516565769 0.9112218830152351 cent full [1.23303725 2.5268821 ] [0.14809039 0.15179082] area 0.8791926151120235 0.9297578262013516
final static iou 0.9056527063174064 0.9178796638712258 cent full [0.51575081 0.72941672] [0.50808146 0.22075435] area 0.9285508449909992 0.9306480041487324
temp initial static base [0.284 0.616 0.823 1.201 4.107 5.204] cand [0. 0. 0. 0. 0. 0.]
temp manipulation base [ 0.673  1.391  2.022  1.623  7.777 16.597] cand [0.485 0.954 1.879 0.44  1.063 1.979]
temp final static base [0.22  0.603 0.93  0.902 3.419 5.059] cand [0. 0. 0. 0. 0. 0.]
global 179.90992479915468 [-0.113 …[+130 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=[('numpy','numpy'),('scipy','scipy'),('opencv-python','cv2'),('h5py','h5py'),('torch','torch'),('trimesh','trimesh'),('nvdiffrast','nvdiffrast')]
import importlib,importlib.metadata
for dist,mod in mods:
 m=importlib.import_module(mod)
 try:v=importlib.metadata.version(dist)
 except Exception:v=getattr(m,'__version__','unknown')
 print(dist,v,getattr(m,'__version__',''))
PY
git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true
find /work/tools -maxdepth 2 -type f 2>/dev/null | head || true"
```
<details><summary>output</summary>

```
scipy 1.16.3 1.16.3
opencv-python 4.11.0.86 4.11.0
h5py 3.16.0 3.16.0
torch 2.5.1+cu124 2.5.1+cu124
trimesh 5.1.1 5.1.1
nvdiffrast 0.4.0 0.4.0
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Rigid object trajectory reconstruction

## Result

"'`object_poses_world.npz` contains `T_world_mesh` with shape `(700, 4, 4)` and
dtype `float64`, mapping the supplied GLB mesh coordinates directly into the
released world frame.

The final trajectory preserves the baseline evidence in ambiguous cases, but
uses the video-level constraints that the single-frame reconstruction ignores:
two genuinely rigid static phases, smooth object motion while carried, and
metric depth inferred from a pooled silhouette fit.

No ground-truth or challenge annotation was used.

## Phases

Frame differencing inside the visible object region (person pixels excluded)
gave a median absolute gray-level difference below 1 during both static
phases.  It rose to about 2-4 levels during manipulation.

| Phase | Frames | Description | Confidence |
|---|---:|---|---|
| Initial static | 0-195 | Platform resting on the floor, then touched while still stationary | High |
| Manipulation | 196-581 | Platform lifted, rotated about 180 degrees, and lowered | Medium-high |
| Final static | 582-699 | Overturned platform at rest on the floor | High |

The final relative orientation is a 179.91 degree rotation from the initial
pose about world-frame axis approximately `[-0.114, 0.004, -0.993]`, which
matches the instructed turn-over action.  In both static phases the output is
one exactly constant rigid transform.

## Method

1. **Phase segmentation.**  The bo …[+5333 chars]
```
<details><summary>output</summary>

```
total 124K
-rw-rw-r-- 1 hanzhou hanzhou 5.3K Oct 20 22:18 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.3K Oct 20 22:18 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 22:16 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  88K Oct 20 22:17 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  17K Oct 20 22:16 solve.py
# Rigid object trajectory reconstruction

## Result

`object_poses_world.npz` contains `T_world_mesh` with shape `(700, 4, 4)` and
dtype `float64`, mapping the supplied GLB mesh coordinates directly into the
released world frame.

The final trajectory preserves the baseline evidence in ambiguous cases, but
uses the video-level constraints that the single-frame reconstruction ignores:
two genuinely rigid static phases, smooth object motion while carried, and
metric depth inferred from a  …[+293 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/rerun.npz > /work/scratch/rerun.log 2>&1 && python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
a=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];b=np.load('/work/scratch/rerun.npz')['T_world_mesh']
d=np.linalg.norm(a[:,:3,3]-b[:,:3,3],axis=1)
r=(Rot.from_matrix(a[:,:3,:3]).inv()*Rot.from_matrix(b[:,:3,:3])).magnitude()*180/np.pi
print('shape dtype',a.shape,a.dtype)
print('max translation diff m',d.max())
print('max rotation diff deg',r.max())
print('finite',np.isfinite(a).all(),'det range',np.linalg.det(a[:,:3,:3]).min(),np.linalg.det(a[:,:3,:3]).max())
print('last row max err',np.abs(a[:,3,:]-np.array([0,0,0,1])).max())
PY
cat /work/scratch/rerun.log"
```
<details><summary>output</summary>

```
shape dtype (700, 4, 4) float64
max translation diff m 0.0
max rotation diff deg 0.0
finite True det range 0.9999999999999982 1.0000000000000013
last row max err 0.0
wrote /work/scratch/rerun.npz: (700, 4, 4), elapsed 59.1s
initial static pose:
 [[-0.99905105 -0.04218451  0.01083828 -0.16928893]
 [ 0.0420294  -0.99901601 -0.01416115  1.14996195]
 [ 0.011425   -0.01369219  0.99984098  2.67974544]
 [ 0.          0.          0.          1.        ]]
final static pose:
 [[ 0.97564486  0.04148904 -0.21539677  0.29132664]
 [-0.04162514  0.99912566  0.00390632  1.19122988]
 [ 0.21537051  0.00515474  0.9765188   2.76291752]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read().replace('An independent rerun reproduced the tested trajectory to\\nwithin 0.03 mm translation and 0.000003 degrees rotation.','An independent rerun produced a bit-identical trajectory (0.0 mm translation\\nand 0.0 degree maximum difference).')
open(p,'w').write(s)
import shutil
shutil.rmtree('/work/outputs/__pycache__',ignore_errors=True)
PY
ls -la /work/outputs && tail -15 /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
total 128
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 22:19 .
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 21:07 ..
-rw-rw-r-- 1 hanzhou hanzhou  5393 Oct 20 22:19 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou  1262 Oct 20 22:18 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 89878 Oct 20 22:17 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 16946 Oct 20 22:16 solve.py

Command:

```bash
python /work/outputs/solve.py \
  --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The measured end-to-end runtime in this container was **70.2 seconds** on the
available CUDA GPU.  The script reads only `camera.json`,
`reconstruction.npz`, `masks.h5`, and `object_mesh.glb` from `--inputs`; it
does not use the network, scratch files, `/weights`, or `/pipeline`.  Random
seeds are fixed.  An independe …[+101 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,cv2,h5py,numpy as np
sys.path.insert(0,'/work/scratch');from render_utils import Renderer
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);iw=np.linalg.inv(np.array(cam['T_world_cam']));B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];C=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];ren=Renderer('/work/inputs/object_mesh.glb')
ids=list(range(0,700,14));res=[]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in ids:
  t=f['object'][i]&~f['person'][i];vals=[]
  for T in [B,C]:
   p=ren.render(iw@T[i],K)['mask'].cpu().numpy()&~f['person'][i];vals.append((p&t).sum()/(p|t).sum())
  res.append([i,*vals])
r=np.array(res)
for n,a,b in [('init',0,196),('move',196,582),('final',582,700)]:
 q=r[(r[:,0]>=a)&(r[:,0]<b)];print(n,'n',len(q),'base',q[:,1].mean(),'result',q[:,2].mean())
print('all',r[:,1].mean(),r[:,2].mean())
PY"
```
<details><summary>output</summary>

```
init n 14 base 0.8810976327124812 result 0.8866580253776013
move n 28 base 0.8505734473862321 result 0.9059682810438432
final n 8 base 0.9023822214909356 result 0.917442770112285
all 0.8674096231343342 0.902397327708246

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read();needle='| **All frames** | **0.8650** | **0.9070** | **+0.0420** |\\n';repl=needle+'\\nAn independent full-resolution check on every 14th frame (50 frames) gave\\nbaseline/result IoU 0.8674/0.9024 overall, with the same positive change in\\neach phase.\\n';s=s.replace(needle,repl);open(p,'w').write(s)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n -A8 -B2 'All frames' /work/outputs/REPORT.md && python - <<'PY'
import numpy as np, json, os
p='/work/outputs/object_poses_world.npz'
a=np.load(p)
print('keys',a.files,'shape',a['T_world_mesh'].shape,'dtype',a['T_world_mesh'].dtype)
print('files',sorted(os.listdir('/work/outputs')))
PY"
```
<details><summary>output</summary>

```
keys ['T_world_mesh'] shape (700, 4, 4) dtype float64
files ['REPORT.md', 'TOOLS.md', 'object_poses_world.npz', 'solve.py']

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(700, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — tools and versions

Validation improvements over the baseline:

| Metric | Baseline | Result |
|---|---:|---:|
| Mean mask IoU, all frames | 0.8650 | 0.9070 |
| Manipulation mask IoU | 0.8460 | 0.9112 |
| Full-resolution sampled IoU | 0.8674 | 0.9024 |
| Max translation step during motion | 16.6 cm | 1.98 cm |

The solver runs offline in about 60–70 seconds and reproduced bit-identical output on a second run.
