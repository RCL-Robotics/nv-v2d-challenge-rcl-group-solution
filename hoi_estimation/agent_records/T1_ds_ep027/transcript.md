# T1_ds_ep027

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T18:47:05Z

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
/usr/bin/bash -c "find /work -maxdepth 2 -type f -printf '%p\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/work/.codex_home/.sandbox_migration
/work/.codex_home/config.toml
/work/.codex_home/goals_1.sqlite
/work/.codex_home/goals_1.sqlite-shm
/work/.codex_home/goals_1.sqlite-wal
/work/.codex_home/installation_id
/work/.codex_home/logs_2.sqlite
/work/.codex_home/logs_2.sqlite-shm
/work/.codex_home/logs_2.sqlite-wal
/work/.codex_home/memories_1.sqlite
/work/.codex_home/memories_1.sqlite-shm
/work/.codex_home/memories_1.sqlite-wal
/work/.codex_home/model_catalog.json
/work/.codex_home/queue_1.sqlite
/work/.codex_home/queue_1.sqlite-shm
/work/.codex_home/queue_1.sqlite-wal
/work/.codex_home/state_5.sqlite
/work/.codex_home/state_5.sqlite-shm
/work/.codex_home/state_5.sqlite-wal
/work/.codex_home/thread_history_1.sqlite
/work/.codex_home/thread_history_1.sqlite-shm
/work/.codex_home/thread_history_ …[+420 chars]
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
/usr/bin/bash -c "python - <<'PY'
import json, numpy as np, h5py, os
for p in ['camera.json','episode.json','ground_plane.json']:
 print('\\n==',p); print(open('/work/inputs/'+p).read())
print('\\n== task'); print(open('/work/inputs/task.txt').read())
for p in ['reconstruction.npz','frame0_reference_human.npz']:
 print('\\n==',p)
 with np.load('/work/inputs/'+p) as z:
  for k in z.files:
   a=z[k]; print(k, a.shape, a.dtype, 'range', (float(a.min()),float(a.max())) if a.size and np.issubdtype(a.dtype,np.number) else '')
with h5py.File('/work/inputs/masks.h5','r') as f:
 print('\\n== masks')
 for k in f.keys(): print(k, f[k].shape, f[k].dtype, 'mean', f[k][:].mean())
PY"
```
<details><summary>output</summary>

```

== camera.json
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
   812.297607421875,
   0.0,
   768.0
  ],
  [
   0.0,
   812.297607421875,
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
   0.9993173472679673,
   -0.0038983030200155813,
   0.03673748334980032,
   0.03467302740598889
  ],
  [
   0.003273806724716094,
   0.999849385256584,
   0.0170437376053555,
   0.03554809192916546
  ],
  [
   -0.03679839179695117,
   -0.016911831231275428,
   0.9991795976327593,
   0.16779911426498725
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
 "note": "pinhole, OpenCV camera (x right, y down, z forward); world is the r …[+1454 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi; echo '---'; python - <<'PY'
import sys, cv2, torch, trimesh, open3d, nvdiffrast
print(sys.version)
print('torch',torch.__version__,'cuda',torch.cuda.is_available())
print('cv2',cv2.__version__,'trimesh',trimesh.__version__,'open3d',open3d.__version__)
PY
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 /work/inputs/video.mp4"
```
<details><summary>output</summary>

```
---
3.11.10 | packaged by conda-forge | (main, Oct 16 2024, 01:27:36) [GCC 13.3.0]
torch 2.5.1+cu124 cuda True
cv2 4.11.0 trimesh 5.1.1 open3d 0.20.0
width=1536
height=1152
r_frame_rate=30/1
nb_frames=565
duration=18.834000

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs /work/scratch/analysis /work/tools && python - <<'PY'
import numpy as np, trimesh, json
from scipy.spatial.transform import Rotation as R
from pathlib import Path
p=Path('/work/inputs')
z=np.load(p/'reconstruction.npz')
T=z['T_world_mesh'].astype(np.float64)
F=len(T)
Rr=T[:,:3,:3]; tt=T[:,:3,3]
print('F',F)
print('t min/max',tt.min(0),tt.max(0))
print('det',np.linalg.det(Rr).min(),np.linalg.det(Rr).max())
for i in range(F):
 if i%20==0 or i==F-1:
  print(i, 't',np.round(tt[i],4),'Rvec',np.round(R.from_matrix(Rr[i]).as_rotvec(),3))
# frame-to-frame
rots=[]
for i in range(F-1):
 d=Rr[i+1]@Rr[i].T
 rots.append(np.degrees(np.linalg.norm(R.from_matrix(d).as_rotvec())))
rots=np.array(rots); dt=np.linalg.norm(np.diff(tt,axis=0),axis=1)
print('rot diffs percentiles',np.percentile(rots,[0,25,50,75,90,95,99,100]))
print('trans diffs percentiles',np.percentile(dt,[0,25,50,75,90,95,99,100]))
print('large rot frames',[(int(i),round(float(rots[i]),1)) for i in np.where(rots>20)[0]])
print('large trans frames',[(int(i),round(float(dt[i]),3)) for i in np.where(dt>.03)[0]])
# static const via rolling diffs
for a,b in [(0,50),(0,100),(0,150),(400,565),(500,565)]:
 Rs=Rr[a:b]; ts=tt[a:b]
 # variation about median pose? pairwise around first
 print('seg',a,b,'t std',ts.std(0),'range',ts.max(0)-ts.min(0),'rot from first deg',np.degrees(np.linalg.norm(R.from_matrix(Rs@Rs[0].T).as_rotvec(),axis=1)).max())
# mesh
sc=trimesh.load(p/'object_mesh.glb',force='sce …[+786 chars]
```
<details><summary>output</summary>

```
F 565
t min/max [-0.50092858  0.19908504  2.94546342] [0.63490224 1.12331891 4.02510166]
det 0.9999959177955126 1.0000006888582142
0 t [0.5045 1.1054 3.7869] Rvec [ 0.073 -2.933 -0.073]
20 t [0.5038 1.1072 3.7954] Rvec [ 0.021 -3.052 -0.11 ]
40 t [0.5066 1.1137 3.8174] Rvec [ 0.021 -3.052 -0.11 ]
60 t [0.5083 1.1175 3.8301] Rvec [ 0.021 -3.052 -0.11 ]
80 t [0.509  1.1193 3.8361] Rvec [ 0.021 -3.052 -0.11 ]
100 t [0.5108 1.1233 3.8498] Rvec [ 0.021 -3.052 -0.11 ]
120 t [0.5044 1.1087 3.8002] Rvec [ 0.021 -3.052 -0.11 ]
140 t [0.5009 1.0939 3.7523] Rvec [ 0.142 -2.902 -0.12 ]
160 t [0.4941 0.9758 3.7495] Rvec [-0.039 -2.794 -0.102]
180 t [0.5245 0.569  3.7798] Rvec [ 2.000e-03 -2.425e+00  7.000e-03]
200 t [0.5134 0.2985 3.6575] Rvec [0.06  0.561 0.066]
220 t [0.4991 0.2687 3.6531] Rvec [0.01 …[+4415 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, cv2, h5py, json, trimesh
from pathlib import Path
p=Path('/work/inputs')
with h5py.File(p/'masks.h5','r') as f:
 obj=f['object'][:]; per=f['person'][:]
F=len(obj)
areas=obj.sum((1,2)); pa=per.sum((1,2)); inter=(obj&per).sum((1,2))
ys,xs=np.mgrid[:1152,:1536]
for i in [0,20,40,60,80,100,120,130,140,150,160,170,180,190,197,200,210,220,230,240,250,260,280,300,320,340,360,380,400,420,440,460,480,500,540,564]:
 m=obj[i]
 if m.any():
  yy,xx=np.where(m); print(i,'area',areas[i],'bbox',xx.min(),yy.min(),xx.max(),yy.max(),'cent',round(xx.mean(),1),round(yy.mean(),1),'person area',pa[i],'overlap',inter[i],round(inter[i]/areas[i],3))
 else: print(i,'EMPTY')
print('empty obj',np.where(areas==0)[0][:50], 'count',sum(areas==0))
print('area percentiles',np.percentile(areas,[0,1,5,10,25,50,75,90,95,99,100]))
# project baseline bounds 8 corners
cam=json.load(open(p/'camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
z=np.load(p/'reconstruction.npz'); T=z['T_world_mesh'].astype(float)
g=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices
# use points every 30 and corners
for i in [0,50,100,140,160,180,197,220,250,280,300,320,340,360,380,400,420,440,460,500,560]:
 v=V[::50]
 xyz=(T[i,:3,:3]@v.T+T[i,:3,3:]).T
 xyz=(Tcw[:3,:3]@xyz.T+Tcw[:3,3:]).T
 uv=xyz[:,:2]/xyz[:,2:3]; uv=uv*K[:2,:2]+K[:2,2]
 print('proj',i,'uv',uv.min(0).round(1),uv.max(0).round(1),'z',xyz[: …[+47 chars]
```
<details><summary>output</summary>

```
0 area 1762 bbox 805 767 852 808 cent 829.1 787.4 person area 73218 overlap 0 0.0
20 area 2002 bbox 806 768 861 808 cent 832.5 787.5 person area 66164 overlap 0 0.0
40 area 2013 bbox 805 768 862 808 cent 832.3 787.5 person area 56271 overlap 0 0.0
60 area 2041 bbox 806 767 862 809 cent 833.1 787.5 person area 43820 overlap 0 0.0
80 area 2058 bbox 806 767 863 809 cent 833.2 787.5 person area 34073 overlap 0 0.0
100 area 2054 bbox 806 768 863 809 cent 833.4 787.6 person area 29451 overlap 0 0.0
120 area 1991 bbox 806 768 862 808 cent 832.7 787.5 person area 31223 overlap 0 0.0
130 area 1944 bbox 806 768 862 808 cent 833.3 788.3 person area 29607 overlap 30 0.015
140 area 1659 bbox 805 767 862 808 cent 831.3 789.5 person area 26716 overlap 14 0.008
150 area 1725 bbox 805 765 861 808 cent 831. …[+2577 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/make_sheets.py <<'PY'
import cv2, h5py, numpy as np, json, trimesh
from pathlib import Path
p=Path('/work/inputs'); out=Path('/work/scratch/analysis')
cap=cv2.VideoCapture(str(p/'video.mp4'))
frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
cap.release(); F=len(frames)
with h5py.File(p/'masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
# baseline projected center
cam=json.load(open(p/'camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
z=np.load(p/'reconstruction.npz'); T=z['T_world_mesh'].astype(float)
g=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices[::20]
cent=[]; bbs=[]
for i in range(F):
 xyz=(T[i,:3,:3]@V.T+T[i,:3,3:]).T
 xyz=(Tcw[:3,:3]@xyz.T+Tcw[:3,3:]).T
 uv=(xyz[:,:2]/xyz[:,2:3])@K[:2,:2].T+K[:2,2]
 cent.append(np.median(uv,axis=0)); bbs.append([uv.min(0),uv.max(0)])
cent=np.array(cent)
# fill mask missing
mcent=[]
for i in range(F):
 yy,xx=np.where(obj[i]); mcent.append([xx.mean(),yy.mean()] if len(xx) else [np.nan,np.nan])
mcent=np.array(mcent)
for j in range(2):
 v=mcent[:,j]; good=np.isfinite(v); mcent[:,j]=np.interp(np.arange(F),np.arange(F)[good],v[good])
# object crop 320x240 scale 2
samples=list(range(0,F,10))
if F-1 not in samples:samples.append(F-1)
crops=[]
for i in samples:
 cx,cy=mcent[i]
 x0=int(round(cx-160)); y0=int(round(cy-120)); x1=x0+320; y1=y0+240
 # pad
 x0=max(0,min(x0,1536-320)); y0=m …[+1576 chars]
```
<details><summary>output</summary>

```
wrote 58 30
total 4.3M
-rw-rw-r-- 1 hanzhou hanzhou 1.5M Oct 20 22:41 full_frames.jpg
-rw-rw-r-- 1 hanzhou hanzhou 2.9K Oct 20 22:41 make_sheets.py
-rw-rw-r-- 1 hanzhou hanzhou 2.8M Oct 20 22:41 object_crops.jpg
-rw-rw-r-- 1 hanzhou hanzhou  37K Oct 20 22:41 projection_debug.npz

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual
print(type(v),v.__dict__.keys())
for k,a in v.__dict__.items():
 try: print(k,type(a),getattr(a,'shape',None),getattr(a,'dtype',None))
 except: pass
print('uv',v.uv.shape,v.uv.min(0),v.uv.max(0))
print('material',v.material)
im=v.material.image
print('image',type(im),im.size,im.mode)
print('base color',v.material.baseColorFactor,'rough',v.material.roughnessFactor)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> dict_keys(['vertex_attributes', '_cache', 'material', 'face_materials', 'mesh'])
vertex_attributes <class 'trimesh.caching.DataStore'> None None
_cache <class 'trimesh.caching.Cache'> None None
material <class 'trimesh.visual.material.PBRMaterial'> None None
face_materials <class 'NoneType'> None None
mesh <class 'trimesh.base.Trimesh'> None None
uv (293827, 2) [0.001521   0.00142401] [0.99847901 0.99857599]
material <trimesh.visual.material.PBRMaterial object at 0x7ae71efef790>
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=m.visual; mat=v.material
print(mat.__dict__.keys())
for k,a in mat.__dict__.items(): print(k,type(a),getattr(a,'shape',None),getattr(a,'size',None),str(a)[:200])
PY"
```
<details><summary>output</summary>

```
dict_keys(['_data'])
_data <class 'dict'> None None {'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=2301x2458 at 0x7F57728229D0>, 'doubleSided': True, 'name': '_texture'}

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_test.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py
from pathlib import Path
import nvdiffrast.torch as dr
p=Path('/work/inputs'); out=Path('/work/scratch/analysis')
m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(np.asarray(m.vertices),dtype=torch.float32,device='cuda'); F=torch.tensor(np.asarray(m.faces),dtype=torch.int32,device='cuda')
uv=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture).astype(np.float32)/255
tex=torch.tensor(tex,dtype=torch.float32,device='cuda')
cam=json.load(open(p/'camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
T=np.load(p/'reconstruction.npz')['T_world_mesh'].astype(np.float32)
ctx=dr.RasterizeCudaContext(); H,W=1152,1536
fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
def render(Twm):
 Tcm=Tcw@Twm
 vc=V@torch.tensor(Tcm[:3,:3].T,device='cuda').T + torch.tensor(Tcm[:3,3],device='cuda') # wrong? V @ R.T
 # Above Tcm[:3,:3].T -> V @ R ; good
 x,y,z=vc[:,0],vc[:,1],vc[:,2]
 clip=torch.stack([2*fx/W*x+(2*cx/W-1)*z, 2*fy/H*y+(2*cy/H-1)*z, z, z],1)[None]
 rast,_=dr.rasterize(ctx,clip,F,resolution=[H,W])
 uv_interp,_=dr.interpolate(uv[None],rast,F)
 # texture expects uv in [0,1], likely v not flipped
 rgb=dr.texture(tex[None],uv_interp,filter_mode='linear',boundary_mode='wrap')
 mask=rast[...,3]>0
 return rgb[0].clamp(0,1).cpu().nump …[+939 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/render_test.py", line 33, in <module>
    rgb,mask=render(T[i]); im=frames[i].copy()
             ^^^^^^^^^^^^
  File "/work/scratch/analysis/render_test.py", line 16, in render
    vc=V@torch.tensor(Tcm[:3,:3].T,device='cuda').T + torch.tensor(Tcm[:3,3],device='cuda') # wrong? V @ R.T
       ~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != double

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/render_test.py'
s=open(p).read()
s=s.replace(\"Tcm=Tcw@Twm\\n vc=V@torch.tensor(Tcm[:3,:3].T,device='cuda').T + torch.tensor(Tcm[:3,3],device='cuda') # wrong? V @ R.T\\n # Above Tcm[:3,:3].T -> V @ R ; good\", \"Tcm=(Tcw@Twm).astype(np.float32)\\n Rcm=torch.tensor(Tcm[:3,:3],device='cuda')\\n tcm=torch.tensor(Tcm[:3,3],device='cuda')\\n vc=V@Rcm.T+tcm\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/render_test.py"
```
<details><summary>output</summary>

```
0 mask area 1721 obj area 1762 iou 0.6998535871156661
100 mask area 1732 obj area 2054 iou 0.7568445475638051
160 mask area 1732 obj area 1430 iou 0.5754857997010463
180 mask area 1286 obj area 1332 iou 0.45444444444444443
200 mask area 1307 obj area 579 iou 0.23672131147540984
250 mask area 1423 obj area 806 iou 0.3042715038033938
300 mask area 980 obj area 931 iou 0.5498783454987834
350 mask area 1704 obj area 1951 iou 0.7572115384615384
400 mask area 2094 obj area 2312 iou 0.8677405680373039
440 mask area 2407 obj area 2744 iou 0.8330960854092526
500 mask area 2458 obj area 2900 iou 0.842503438789546
560 mask area 2435 obj area 2875 iou 0.8431100312391531

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,glob,os
ims=[]
for i in [0,100,160,180,200,250,300,350,400,440,500,560]:
 im=cv2.imread(f'/work/scratch/analysis/rendercrop_{i:03d}.jpg');
 if im is None: continue
 cv2.rectangle(im,(0,0),(100,30),(0,0,0),-1); cv2.putText(im,str(i),(5,24),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2); ims.append(im)
rows=3; cols=4; h,w=ims[0].shape[:2]; sheet=np.zeros((rows*h,cols*w,3),np.uint8)
for k,im in enumerate(ims):sheet[(k//cols)*h:(k//cols+1)*h,(k%cols)*w:(k%cols+1)*w]=im
cv2.imwrite('/work/scratch/analysis/render_sheet.jpg',sheet)
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
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); Tw=z['T_world_mesh'].astype(float); Tf=z['T_cam_mesh_foundationpose'].astype(float); Tc=z['T_cam_mesh_coconet'].astype(float)
for name,T in [('world',Tw),('fp',Tf),('coco',Tc)]:
 print('\\n',name)
 for i in range(len(T)-1):
  da=np.degrees(np.linalg.norm(R.from_matrix(T[i+1,:3,:3]@T[i,:3,:3].T).as_rotvec()))
  if da>15: print(i,'->',i+1,round(da,1),'t',round(np.linalg.norm(T[i+1,:3,3]-T[i,:3,3]),3))
 print('rot diff pct',np.percentile([np.degrees(np.linalg.norm(R.from_matrix(T[i+1,:3,:3]@T[i,:3,:3].T).as_rotvec())) for i in range(len(T)-1)],[50,90,95,99,100]))
# compare baseline world to fp/coco after world transformation rotations
Twc=np.eye(4)
import json
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for name,T in [('fp',Tf),('coco',Tc)]:
 Wrot=np.einsum('ij,fjk->fik',Twc[:3,:3][None],T[:,:3,:3])
 d=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('fji,fjk->fik',Wrot,Tw[:,:3,:3])).as_rotvec(),axis=1))
 print(name,'rot vs baseline pct',np.percentile(d,[0,25,50,75,90,95,99,100]))
# print angles relative from initial for all
print('\\nFrame rvec world and angle from frame100')
R0=Tw[100,:3,:3]
for i in range(0,565,5):
 d=R.from_matrix(Tw[i,:3,:3]@R0.T).as_rotvec(); ang=np.degrees(np.linalg.norm(d))
 print(i, np.round(R.from_matrix(Tw[i,:3,:3]).as_rotvec(),2), round(ang,1), np.round(Tw[i,:3,3],2))
PY"
```
<details><summary>output</summary>

```

 world
161 -> 162 18.3 t 0.022
179 -> 180 17.3 t 0.024
188 -> 189 22.4 t 0.046
189 -> 190 25.0 t 0.025
197 -> 198 122.8 t 0.039
233 -> 234 16.5 t 0.019
250 -> 251 35.6 t 0.044
251 -> 252 33.1 t 0.023
318 -> 319 15.0 t 0.02
319 -> 320 15.8 t 0.019
359 -> 360 19.4 t 0.054
361 -> 362 22.5 t 0.032
362 -> 363 20.3 t 0.049
372 -> 373 30.1 t 0.023
373 -> 374 26.6 t 0.013
374 -> 375 23.5 t 0.045
rot diff pct [  1.1879132    7.54453643  10.84453107  24.01603877 122.77451968]

 fp
86 -> 87 15.7 t 0.03
161 -> 162 22.6 t 0.283
173 -> 174 21.4 t 0.066
175 -> 176 24.4 t 2.085
179 -> 180 22.9 t 0.038
180 -> 181 17.3 t 0.048
188 -> 189 82.5 t 0.275
191 -> 192 94.6 t 0.33
192 -> 193 21.0 t 0.191
194 -> 195 165.7 t 0.137
195 -> 196 19.5 t 0.144
198 -> 199 20.5 t 0.076
203 -> 204 17.5 t 0.022
204 -> 205 149 …[+1340 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); Tw=z['T_world_mesh'].astype(float); Tf=z['T_cam_mesh_foundationpose'].astype(float); Tc=z['T_cam_mesh_coconet'].astype(float)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for name,T in [('fp',Tf),('coco',Tc)]:
 Wrot=Twc[None,:3,:3]@T[:,:3,:3]
 d=np.degrees(np.linalg.norm(R.from_matrix(np.transpose(Wrot,(0,2,1))@Tw[:,:3,:3]).as_rotvec(),axis=1))
 print(name,'rot vs baseline pct',np.percentile(d,[0,25,50,75,90,95,99,100]))
 print('near baseline',np.mean(d<5),'near 180',np.mean(np.abs(d-180)<20))
R0=Tw[100,:3,:3]
print('\\nFrame rvec world and angle from frame100')
for i in range(0,565,5):
 d=R.from_matrix(Tw[i,:3,:3]@R0.T).as_rotvec(); ang=np.degrees(np.linalg.norm(d))
 print(f'{i:3d}', np.round(R.from_matrix(Tw[i,:3,:3]).as_rotvec(),2), f'{ang:6.1f}', np.round(Tw[i,:3,3],2))
PY"
```
<details><summary>output</summary>

```
fp rot vs baseline pct [  0.67295252   3.44354353   5.67656238  11.0036149   22.53092837
  45.13653314 165.41930273 174.03387946]
near baseline 0.4389380530973451 near 180 0.01592920353982301
coco rot vs baseline pct [1.0622879  1.06458012 1.28899025 1.29653914 1.30538955 1.30643809
 1.30721701 1.30771213]
near baseline 1.0 near 180 0.0

Frame rvec world and angle from frame100
  0 [ 0.07 -2.93 -0.07]    7.2 [0.5  1.11 3.79]
  5 [ 0.08 -3.07 -0.08]    2.5 [0.5  1.1  3.78]
 10 [ 0.02 -3.05 -0.11]    0.0 [0.5  1.1  3.78]
 15 [ 0.02 -3.05 -0.11]    0.0 [0.5  1.1  3.79]
 20 [ 0.02 -3.05 -0.11]    0.0 [0.5  1.11 3.8 ]
 25 [ 0.02 -3.05 -0.11]    0.0 [0.5  1.11 3.8 ]
 30 [ 0.02 -3.05 -0.11]    0.0 [0.5  1.11 3.8 ]
 35 [ 0.02 -3.05 -0.11]    0.0 [0.51 1.11 3.81]
 40 [ 0.02 -3.05 -0.11]    0.0 [0.5 …[+5131 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image
import trimesh
im=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().visual.material.baseColorTexture
im.resize((575,614)).save('/work/scratch/analysis/texture.jpg')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_variants.py <<'PY'
exec(open('/work/scratch/analysis/render_test.py').read().split('cap=cv2.VideoCapture')[0])
for i in [0,100,160,180,200,250,300,350,400,440,500,560]:
 rgb,mask=render(T[i]);
 ys,xs=np.where(mask); cx=int(xs.mean());cy=int(ys.mean());x0=max(0,cx-130);y0=max(0,cy-100);x1=min(W,x0+260);y1=min(H,y0+200)
 out=np.zeros((y1-y0,x1-x0,3),np.uint8)
 out[mask[y0:y1,x0:x1]]=(rgb[mask[y0:y1,x0:x1]]*255).clip(0,255).astype(np.uint8)
 out=cv2.resize(out,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
 cv2.imwrite(str(outpath/f'rawrender_{i:03d}.png'),cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
PY
# fix variable name
sed -i \"s|str(out/|str(outpath/|\" /work/scratch/analysis/render_variants.py
python - <<'PY'
p='/work/scratch/analysis/render_variants.py';s=open(p).read();s=s.replace(\"exec(open('/work/scratch/analysis/render_test.py').read().split('cap=cv2.VideoCapture')[0])\",\"exec(open('/work/scratch/analysis/render_test.py').read().split('cap=cv2.VideoCapture')[0])\\noutpath=out\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/render_variants.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/render_variants.py", line 7, in <module>
    out[mask[y0:y1,x0:x1]]=(rgb[mask[y0:y1,x0:x1]]*255).clip(0,255).astype(np.uint8)
                            ~~~^^^^^^^^^^^^^^^^^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1152 but corresponding boolean dimension is 200

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/render_variants.py';s=open(p).read();s=s.replace(\"out[mask[y0:y1,x0:x1]]=(rgb[mask[y0:y1,x0:x1]]*255).clip(0,255).astype(np.uint8)\",\"mc=mask[y0:y1,x0:x1]; rc=rgb[y0:y1,x0:x1]; out[mc]=(rc[mc]*255).clip(0,255).astype(np.uint8)\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/render_variants.py
python - <<'PY'
import cv2,numpy as np
ims=[]
for i in [0,100,160,180,200,250,300,350,400,440,500,560]:
 im=cv2.imread(f'/work/scratch/analysis/rawrender_{i:03d}.png'); cv2.rectangle(im,(0,0),(80,30),(255,255,255),-1); cv2.putText(im,str(i),(5,24),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2); ims.append(im)
h,w=ims[0].shape[:2];sheet=np.zeros((3*h,4*w,3),np.uint8)
for k,im in enumerate(ims): sheet[(k//4)*h:(k//4+1)*h,(k%4)*w:(k%4+1)*w]=im
cv2.imwrite('/work/scratch/analysis/rawrender_sheet.jpg',sheet)
PY"
```
<details><summary>output</summary>

```
/work/scratch/analysis/render_variants.py:5: RuntimeWarning: Mean of empty slice.
  ys,xs=np.where(mask); cx=int(xs.mean());cy=int(ys.mean());x0=max(0,cx-130);y0=max(0,cy-100);x1=min(W,x0+260);y1=min(H,y0+200)
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
Traceback (most recent call last):
  File "/work/scratch/analysis/render_variants.py", line 5, in <module>
    ys,xs=np.where(mask); cx=int(xs.mean());cy=int(ys.mean());x0=max(0,cx-130);y0=max(0,cy-100);x1=min(W,x0+260);y1=min(H,y0+200)
                             ^^^^^^^^^^^^^^
ValueError: cannot convert float NaN to integer
[ WARN:0@0.032] global loadsave.cpp:268 findDecoder imread_('/work/scratch/analysis/rawrender_160 …[+1603 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/analysis/render_variants.py; nl -ba /work/scratch/analysis/render_test.py | sed -n '1,40p'; ls -l /work/scratch/analysis/rawrender*"
```
<details><summary>output</summary>

```
     1	exec(open('/work/scratch/analysis/render_test.py').read().split('cap=cv2.VideoCapture')[0])
     2	outpath=out
     3	for i in [0,100,160,180,200,250,300,350,400,440,500,560]:
     4	 rgb,mask=render(T[i]);
     5	 ys,xs=np.where(mask); cx=int(xs.mean());cy=int(ys.mean());x0=max(0,cx-130);y0=max(0,cy-100);x1=min(W,x0+260);y1=min(H,y0+200)
     6	 out=np.zeros((y1-y0,x1-x0,3),np.uint8)
     7	 mc=mask[y0:y1,x0:x1]; rc=rgb[y0:y1,x0:x1]; out[mc]=(rc[mc]*255).clip(0,255).astype(np.uint8)
     8	 out=cv2.resize(out,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
     9	 cv2.imwrite(str(outpath/f'rawrender_{i:03d}.png'),cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
     1	import numpy as np, torch, trimesh, json, cv2, h5py
     2	from pathlib import Path
     3	import nvdiffrast.torch as dr
     4 …[+2263 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_variants2.py <<'PY'
exec(open('/work/scratch/analysis/render_test.py').read().split('cap=cv2.VideoCapture')[0])
outpath=out
for idx in [0,100,160,180,200,250,300,350,400,440,500,560]:
 rgb,mask=render(T[idx]); print(idx,mask.sum(),rgb.min(),rgb.max())
 ys,xs=np.where(mask)
 if len(xs)==0: continue
 cx=int(xs.mean());cy=int(ys.mean());x0=max(0,cx-130);y0=max(0,cy-100);x1=min(W,x0+260);y1=min(H,y0+200)
 out_img=np.zeros((y1-y0,x1-x0,3),np.uint8)
 mc=mask[y0:y1,x0:x1]; rc=rgb[y0:y1,x0:x1]; out_img[mc]=(rc[mc]*255).clip(0,255).astype(np.uint8)
 out_img=cv2.resize(out_img,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
 cv2.imwrite(str(outpath/f'rawrender_{idx:03d}.png'),cv2.cvtColor(out_img,cv2.COLOR_RGB2BGR))
PY
python /work/scratch/analysis/render_variants2.py"
```
<details><summary>output</summary>

```
0 1721 0.0 0.77781194
100 1731 0.0 0.7795216
160 0 0.0 0.0
180 1289 0.0 0.7758662
200 540 0.0 0.76279455
250 0 0.0 0.0
300 0 0.0 0.0
350 0 0.0 0.0
400 0 0.0 0.0
440 0 0.0 0.0
500 0 0.0 0.0
560 0 0.0 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "for i in 160 250 300 350 400 440 500 560; do python - <<PY
exec(open('/work/scratch/analysis/render_test.py').read().split('cap=cv2.VideoCapture')[0])
rgb,mask=render(T["'$i]); print($i,mask.sum(),np.where(mask)[0].min() if mask.any() else None)
PY
done'
```
<details><summary>output</summary>

```
160 1732 739
250 1423 577
300 980 586
350 1704 580
400 2094 579
440 2407 609
500 2458 610
560 2435 610

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 4 -type f -printf '%p %s\\n' | sort | head -300"
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
/pipeline/depth/moge2/aligned.h5 697122020
/pipeline/depth/moge2/depth_report.json 1811
/pipeline/depth/moge2/intrinsics.pkl 60288
/pipeline/depth/moge2/raw.npy 1999503488
/pipeline/export/episode_000027/edex 561
/pipeline/export/episode_000027/human_masks/.front_stereo_camera_left.h5.storage-repack.lock 0
/pipeline/export/episode_000027/human_masks/front_stereo_camera_left.h5 13514379
/pipeline/export/ep …[+11447 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 3 -type f -printf '%p\\n' | sort | head -300"
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
/code/reconstruction/.venv/bin/numpy-config …[+7037 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace -maxdepth 3 -type f -name '*.py' -printf '%p\\n' | sort | head -300"
```
<details><summary>output</summary>

```
/workspace/third_party/kaolin/setup.py
/workspace/third_party/nvdiffrast/setup.py
/workspace/v2d_cari4d/lib/__init__.py
/workspace/v2d_cari4d/lib/download_weights.py
/workspace/v2d_cari4d/lib/pack_masks.py
/workspace/v2d_cari4d/lib/run_inference.py
/workspace/v2d_cari4d/lib/run_training.py
/workspace/v2d_common/__init__.py
/workspace/v2d_common/broadcast.py
/workspace/v2d_common/datatypes.py
/workspace/v2d_common/ffv1_sidecar.py
/workspace/v2d_common/hdf5_transcode.py
/workspace/v2d_common/result_bundle.py
/workspace/v2d_common/tests/test_ffv1_sidecar.py
/workspace/v2d_common/tests/test_result_bundle_gravity.py
/workspace/v2d_common/tests/test_video_frame_count.py
/workspace/v2d_common/tests/test_video_hdf5.py
/workspace/v2d_common/utils.py
/workspace/v2d_common/video.py
/workspace/v2d_fou …[+3886 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/.stages/05_object_pose.json; echo; cat /pipeline/.stages/06_coconet.json; echo; cat /pipeline/.stages/07_refinement.json; echo; cat /pipeline/pipeline_report.json'
```
<details><summary>output</summary>

```
{
  "name": "05_object_pose",
  "outputs": [
    {
      "mtime_ns": 1791268229554560050,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000027/foundationpose/episode_000027.pkl",
      "size": 163539
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000027/export/episode_000027",
      "--depth-root",
      "/home/hanzhou/Projects/v2d/run/out/episode_000027/depth/moge2/aligned.h5",
      "--out-file",
      "/home/hanzhou/Projects/v2d/run/out/episode_000027/foundationpose/episode_000027.pkl",
      "--camera-id",
      "0",
      "--iteration",
      "5",
      "--max-attempts",
      "5",
 …[+11854 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,260p' /workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py; echo '=== video_to_poses'; sed -n '1,300p' /workspace/v2d_foundation_pose/lib/run_video_to_poses.py"
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
    scale …[+19509 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,300p' /workspace/v2d_cari4d/lib/run_inference.py; echo '=== core files'; find /workspace/v2d_cari4d -type f -name '*.py' -printf '%p\\n' | sort"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the complete CARI4D wild-video inference pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from v2d.cari4d.lib.download_weights import CARI4D_CHECKPOINT_RELATIVE_PATH, CARI4D_CHECKPOINT_SHA256, CARI4D_CONFIG_RELATIVE_PATH, CARI4D_MANIFEST_RELATIVE_PATH, CARI4D_REVISION, download_weights, require_moge2_model, sha256_file


SOURCE_ROOT = Path(__file__).resolve().parent / "cari4d"
SAM3D_SOURCE_ROOT = Path("/workspace/v2d_sam3d_body/lib")
PIPELINE_SCHEMA = "v2d.cari4d.wild_inference.v1"


def _file_ide …[+24968 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,320p' /workspace/v2d_cari4d/lib/cari4d/prep/foundationpose_pose_selection.py; echo ====; sed -n '1,320p' /workspace/v2d_cari4d/lib/cari4d/lib_mhr/object_symmetry.py; echo ====; sed -n '1,260p' /workspace/v2d_cari4d/lib/cari4d/lib_mhr/object_pose_frame.py"
```
<details><summary>output</summary>

```
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag

import cv2
import numpy as np

from lib_mhr.object_symmetry import OBJECT_SYMMETRY_MODE_FINITE, OBJECT_SYMMETRY_MODE_FULL_SO3


@dataclass(frozen=True)
class OracleSelection:
    index: int
    centered_pose: np.ndarray
    output_pose: np.ndarray
    rotation_errors_deg: np.ndarray


@dataclass(frozen=True)
class CandidateFilter:
    iou_mask: np.ndarray
    keep_mask: np.ndarray
    rotation_distances_deg: np.ndarray
    rotation_threshold_deg: float
    translation_distances_m: np.ndarray
    translation_threshold_m: float
    selected_index: int | None


class TrackingState(IntEnum):
    TRACKING = 0
    OCCLUDED = 1
    REACQUIRING = 2


class PoseSource(IntEnum):
    NONE = 0
     …[+41492 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace/v2d_cari4d/lib/cari4d -type f -name '*.py' -print0 | xargs -0 grep -n \"pose_abs\\|refined.pth\\|symmetry\\|object_pose\" | head -300"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/Utils.py:832:def symmetry_spec_from_info(info, rot_angle_discrete=5):
/workspace/v2d_cari4d/lib/cari4d/Utils.py:833:  from lib_mhr.object_symmetry import bop_object_symmetry
/workspace/v2d_cari4d/lib/cari4d/Utils.py:835:  return bop_object_symmetry(info, continuous_angle_degrees=rot_angle_discrete)
/workspace/v2d_cari4d/lib/cari4d/estimater.py:49:def cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs, return_indices=False):
/workspace/v2d_cari4d/lib/cari4d/estimater.py:57:      symmetry_tfs (list of np.ndarray): List of 4x4 symmetry transformation matrices.
/workspace/v2d_cari4d/lib/cari4d/estimater.py:77:      for tf in symmetry_tfs:
/workspace/v2d_cari4d/lib/cari4d/estimater.py:95:  def __init__(self, model_pts, model_normals, symmetry_tfs=None,  …[+50706 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,pickle,json
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth','/pipeline/inference/refinement_checkpoint.pth','/pipeline/foundationpose/episode_000027.pkl']:
 print('\\n===',p)
 if p.endswith('.pkl'):
  with open(p,'rb') as f: x=pickle.load(f)
  print(type(x)); print(x if isinstance(x,(dict,list)) and len(str(x))<10000 else (x.keys() if hasattr(x,'keys') else len(x)))
 else:
  try:x=torch.load(p,map_location='cpu',weights_only=False)
  except Exception as e: print('ERR',e);continue
  print(type(x))
  if isinstance(x,dict):
   for k,v in x.items():
    print(k,type(v),getattr(v,'shape',None),str(v)[:150].replace('\\n',' '))
PY"
```
<details><summary>output</summary>

```

=== /pipeline/inference/coconet.pth
<class 'dict'>
schema <class 'str'> None cari4d.mhr_wild_inference.v1
frames <class 'list'> None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014',
frame_meta <class 'list'> None [{'frame': '000000', 'src_frame': 0, 'kid': 0}, {'frame': '000001', 'src_frame': 1, 'kid': 0}, {'frame': '000002', 'src_frame': 2, 'kid': 0}, {'frame'
kid <class 'int'> None 0
checkpoint <class 'dict'> None {'path': '/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth', 'step': 200000}
mhr_inference <class 'dict'> None {'schema': 'cari4d.mhr_inference_provenance.v1', 'checkpoint': {'path': '/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/ste …[+8536 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
p='/pipeline/foundationpose/episode_000027.pkl'
with open(p,'rb') as f:x=pickle.load(f)
for k,v in x.items():
 a=np.asarray(v)
 print(k,a.shape,a.dtype,'minmax', (float(a.min()),float(a.max())) if a.size and np.issubdtype(a.dtype,np.number) else '')
 if k not in ['fp_poses_cam','obj_pose_world','frames','obj_rot_init','obj_t_init']:
  print(' ',str(v)[:500].replace('\\n',' '))
print('metadata',x['metadata'])
PY"
```
<details><summary>output</summary>

```
  [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan n
fp_translation_threshold_m (565,) float32 minmax (nan, nan)
  [nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan nan  nan nan nan nan …[+19318 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,420p' /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py"
```
<details><summary>output</summary>

```
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio
import nvdiffrast.torch as dr
import numpy as np
import torch
import trimesh
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Utils
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, cluster_poses
from lib_mhr import ObjectSymmetry, load_optional_output_object_symmetry, load_optional_output_symmetry_tfs
from lib_mhr.object_texture import concatenate_object_mesh_parts, load_object_mesh_parts
from pre …[+25947 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 x=torch.load(p,map_location='cpu',weights_only=False)
 print('\\n',p)
 for sect in ['pr','pr_initial','in','raw','postopt']:
  d=x.get(sect); print(sect, type(d), list(d.keys()) if isinstance(d,dict) else '')
  if isinstance(d,dict):
   for k,v in d.items():
    try:a=np.asarray(v); print(' ',k,a.shape,a.dtype, (a.min(),a.max()) if a.size and np.issubdtype(a.dtype,np.number) else '')
    except: print(' ',k,type(v),str(v)[:100])
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
pr <class 'dict'> ['pose_abs', 'pose_abs_1st_delta', 'mhr_global_rot6d', 'mhr_trans', 'mhr_body_pose_cont', 'mhr_hand', 'mhr_shape', 'mhr_scale', 'mhr_face', 'contact_logits']
  pose_abs (565, 4, 4) float32 (-0.99998426, 17.346657)
  pose_abs_1st_delta (565, 4, 4) float32 (-0.99998426, 17.346657)
  mhr_global_rot6d (565, 6) float32 (-0.99747443, 0.99999756)
  mhr_trans (565, 3) float32 (-0.88842267, 3.7903926)
  mhr_body_pose_cont (565, 260) float32 (-0.8211088, 1.0602443)
  mhr_hand (565, 108) float32 (-0.5450705, 0.41084105)
  mhr_shape (565, 45) float32 (-1.6668898, 2.2418694)
  mhr_scale (565, 28) float32 (-2.4568713, 2.8568282)
  mhr_face (565, 72) float32 (0.0, 0.0)
  contact_logits (565, 2) float32 (-35.8667, 21.779278)
pr_initial <class 'dict'> ['p …[+5707 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/export/episode_000027/wild_export.json; echo; cat /pipeline/export/episode_000027/edex; echo; find /pipeline/export/episode_000027 -maxdepth 2 -type f -o -type l -printf '%p -> %l\\n' | sort | head -100"
```
<details><summary>output</summary>

```
{
  "aligned_object_extents_m": [
    0.07273773849010468,
    0.2511727809906006,
    0.3503861725330353
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
  "frame_count": 565,
  "height": 1152,
  "intrinsics": [
    [
      812.297607421875,
      0.0,
      768.0
    ],
    [
      0.0,
      812.297607421875,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "object_mesh_file": "/home/hanzhou/Projects/v2d/run/out/episode_000027/export/episode_000027/object_mesh/output_aligned.glb",
  "object_mesh_to_training_transfor …[+2558 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
with open('/pipeline/foundationpose/episode_000027.pkl','rb') as f:x=pickle.load(f)
for k in ['fp_poses_cam','obj_pose_world','obj_rot_init','obj_t_init','fp_confirmed_pose_source','fp_emitted_pose_source','fp_tracking_state','fp_selected_iou','fp_selected_visibility','fp_full_image_iou','fp_tolerant_precision','fp_tolerant_recall','fp_rendered_visibility']:
 a=np.asarray(x[k]); print('\\n',k,a.shape,a.dtype)
 if a.ndim==1:
  uniq,cnt=np.unique(a,return_counts=True); print('uniq',list(zip(uniq.tolist(),cnt.tolist()))[:30])
  print('samples',a[::20])
 else:
  print('samples',a[::50].reshape(-1,4)[:8] if a.ndim==4 else a[::50])
print('metadata selected keys')
for k,v in x['metadata'].items():
 if 'pose' in k or 'sym' in k or 'mesh' in k or 'camera' in k: print(k,str(v)[:500])
PY"
```
<details><summary>output</summary>

```

 fp_poses_cam (565, 1, 4, 4) float32
samples [[-0.06861859  0.99242747 -0.10187833  0.29186153]
 [ 0.991007    0.0795647   0.10758597  0.96738714]
 [ 0.11487724 -0.09357979 -0.98896205  3.7400458 ]
 [ 0.          0.          0.          1.        ]
 [-0.00939044  0.9991768   0.03947049  0.29529384]
 [ 0.992454    0.00448703  0.12252926  0.9634646 ]
 [ 0.12225137  0.04032328 -0.9916796   3.7160366 ]
 [ 0.          0.          0.          1.        ]]

 obj_pose_world (565, 4, 4) float32
samples [[[-6.86185881e-02  9.92427468e-01 -1.01878330e-01  2.91861534e-01]
  [ 9.91006970e-01  7.95646980e-02  1.07585974e-01  9.67387140e-01]
  [ 1.14877239e-01 -9.35797915e-02 -9.88962054e-01  3.74004579e+00]
  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

 [[-9.39044170e-03  9.9917 …[+13877 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
for p in ['/work/inputs/object_mesh.glb','/pipeline/export/episode_000027/object_mesh/output_aligned.glb']:
 m=trimesh.load(p,force='scene',process=False).to_geometry(); print(p, m.vertices.shape,m.faces.shape,m.bounds,m.extents,m.centroid)
 print('visual',type(m.visual),getattr(m.visual,'uv',None).shape if getattr(m.visual,'uv',None) is not None else None)
PY"
```
<details><summary>output</summary>

```
/work/inputs/object_mesh.glb (293827, 3) (97945, 3) [[-0.12257792 -0.04155331 -0.17304806]
 [ 0.12962829  0.03169774  0.17714772]] [0.25220621 0.07325106 0.35019578] [ 6.87076613e-11  2.77035445e-10 -3.70416435e-10]
visual <class 'trimesh.visual.texture.TextureVisuals'> (293827, 2)
/pipeline/export/episode_000027/object_mesh/output_aligned.glb (293827, 3) (97945, 3) [[-0.03636887 -0.12558639 -0.17519309]
 [ 0.03636887  0.12558639  0.17519309]] [0.07273774 0.25117278 0.35038618] [ 0.00533543  0.00416986 -0.00184554]
visual <class 'trimesh.visual.texture.TextureVisuals'> (293827, 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); Tw=z['T_world_mesh'].astype(float)
# for baseline input mesh local y is height
for i in range(0,565,10):
 Ry=Tw[i,:3,1]; tilt=np.degrees(np.arccos(np.clip(abs(Ry[1]),-1,1)))
 # angle from +y
 tiltp=np.degrees(np.arccos(np.clip(Ry[1],-1,1)))
 print(f'{i:3d} yaxis {Ry.round(2)} tilt_abs {tilt:5.1f} tiltp {tiltp:5.1f}')
PY"
```
<details><summary>output</summary>

```
  0 yaxis [-0.04  1.    0.05] tilt_abs   4.0 tiltp   4.0
 10 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 20 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 30 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 40 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 50 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 60 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 70 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 80 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
 90 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
100 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
110 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
120 yaxis [-0.01  1.    0.07] tilt_abs   4.2 tiltp   4.2
130 yaxis [-0.06  1.    0.07] tilt_abs   5.1 tiltp   5.1
14 …[+2368 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/unwrap_dp.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); F=len(T)
S=np.eye(4); S[:3,:3]=Rot.from_euler('y',np.pi).as_matrix()
C=np.stack([T,T@S],axis=1) # F,2,4,4
# costs
for cap in [10,20,30,45,90,180]:
 print('\\ncap',cap)
 for l in [0.1,0.3,1,3]:
  dp=np.zeros((F,2)); back=np.zeros((F,2),np.int8)
  for i in range(1,F):
   for b in range(2):
    costs=[]
    for a in range(2):
     ang=np.degrees(np.linalg.norm(Rot.from_matrix(C[i,b,:3,:3]@C[i-1,a,:3,:3].T).as_rotvec()))
     costs.append(dp[i-1,a]+l*min(ang,cap)**2)
    a=int(np.argmin(costs)); dp[i,b]=costs[a];back[i,b]=a
  path=np.zeros(F,int); path[-1]=int(np.argmin(dp[-1]))
  for i in range(F-1,0,-1): path[i-1]=back[i,path[i]]
  bd=np.where(np.diff(path)"'!=0)[0]+1
  rots=np.array([np.degrees(np.linalg.norm(Rot.from_matrix(C[i,path[i],:3,:3]@C[i-1,path[i-1],:3,:3].T).as_rotvec())) for i in range(1,F)])
  print('"'lambda',l,'flip frames',bd,'maxstep',rots.max(),'pct>10',np.mean(rots>10),'finalflip',path[-1],'cost',dp[-1].min())
PY
python /work/scratch/analysis/unwrap_dp.py"
```
<details><summary>output</summary>

```

cap 10
lambda 0.1 flip frames [  1 180 181 184 338 354] maxstep 174.02247029318525 pct>10 0.06028368794326241 finalflip 0 cost 873.6645240172231
lambda 0.3 flip frames [  1 180 320 354] maxstep 174.02247029318525 pct>10 0.06028368794326241 finalflip 0 cost 2620.993572051669
lambda 1 flip frames [162 180 181 184 200 215] maxstep 168.20261649078606 pct>10 0.06028368794326241 finalflip 0 cost 8736.645240172227
lambda 3 flip frames [  1 180 181 184 200 215 338 354] maxstep 174.02247029318525 pct>10 0.06028368794326241 finalflip 0 cost 26209.935720516678

cap 20
lambda 0.1 flip frames [375] maxstep 157.0822171424459 pct>10 0.06028368794326241 finalflip 1 cost 1371.3756283671464
lambda 0.3 flip frames [189] maxstep 158.02481739842565 pct>10 0.06028368794326241 finalflip 0 cost 4114.12688510144
 …[+2144 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float); F=len(T)
S=Rot.from_euler('y',np.pi).as_matrix(); Tf=T.copy(); Tf[198:,:3,:3]=Tf[198:,:3,:3]@S
# yaw decomposition, also y-axis
for i in range(0,F,5):
 def yaw(R):
  # rotate long local z to world and get atan2 x,z
  v=R@np.array([0,0,1.]); return np.degrees(np.arctan2(v[0],v[2]))
 print(f'{i:3d} base {yaw(T[i,:3,:3]):7.1f} flipfix {yaw(Tf[i,:3,:3]):7.1f} rel0 {np.degrees(np.linalg.norm(Rot.from_matrix(Tf[i,:3,:3]@Tf[100,:3,:3].T).as_rotvec())):6.1f}')
PY"
```
<details><summary>output</summary>

```
  0 base  -168.1 flipfix  -168.1 rel0    7.2
  5 base  -175.7 flipfix  -175.7 rel0    2.5
 10 base  -174.9 flipfix  -174.9 rel0    0.0
 15 base  -174.9 flipfix  -174.9 rel0    0.0
 20 base  -174.9 flipfix  -174.9 rel0    0.0
 25 base  -174.9 flipfix  -174.9 rel0    0.0
 30 base  -174.9 flipfix  -174.9 rel0    0.0
 35 base  -174.9 flipfix  -174.9 rel0    0.0
 40 base  -174.9 flipfix  -174.9 rel0    0.0
 45 base  -174.9 flipfix  -174.9 rel0    0.0
 50 base  -174.9 flipfix  -174.9 rel0    0.0
 55 base  -174.9 flipfix  -174.9 rel0    0.0
 60 base  -174.9 flipfix  -174.9 rel0    0.0
 65 base  -174.9 flipfix  -174.9 rel0    0.0
 70 base  -174.9 flipfix  -174.9 rel0    0.0
 75 base  -174.9 flipfix  -174.9 rel0    0.0
 80 base  -174.9 flipfix  -174.9 rel0    0.0
 85 base  -174.9 flipfix  -174.9 re …[+4285 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_flips.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr
p=Path('/work/inputs'); out=Path('/work/scratch/analysis')
m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(np.asarray(m.vertices),dtype=torch.float32,device='cuda'); F=torch.tensor(np.asarray(m.faces),dtype=torch.int32,device='cuda')
uv=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')
tex=np.asarray(m.visual.material.baseColorTexture).astype(np.float32)/255
tex=torch.tensor(tex,dtype=torch.float32,device='cuda')
cam=json.load(open(p/'camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
T=np.load(p/'reconstruction.npz')['T_world_mesh'].astype(np.float32)
cap=cv2.VideoCapture(str(p/'video.mp4')); frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
with h5py.File(p/'masks.h5','r') as f: obj=f['object'][:]; per=f['person'][:]
S=np.eye(4,dtype=np.float32); S[:3,:3]=Rot.from_euler('y',np.pi).as_matrix()
ctx=dr.RasterizeCudaContext(); H,W=1152,1536

def render(Twm,x0,y0,w,h):
 # crop intrinsics
 k=K.copy(); k[0,2]-=x0; k[1,2]-=y0
 Tcm=(Tcw@Twm).astype(np.float32); Rcm=torch.tensor(Tcm[:3,:3],device='cuda'); tcm=torch.tensor(Tcm[:3,3],device='cuda')
 vc=V@Rcm.T+tcm; x,y,z=vc[:,0],vc[:,1],vc[:,2]
 fx,fy=k[0,0],k[1,1]; cx,cy=k[0, …[+2368 chars]
```
<details><summary>output</summary>

```
100 base {'ncc': -0.09513103662320728, 'edge_ncc': -0.00038690007758401975, 'mad': 0.225193589925766, 'iou': 0.7568445475638051, 'vis': 1732, 'full': 1732} flip {'ncc': -0.03387432020132499, 'edge_ncc': 0.03195780348929552, 'mad': 0.23291119933128357, 'iou': 0.7259770114942529, 'vis': 1700, 'full': 1700}
140 base {'ncc': -0.1322082045797108, 'edge_ncc': -0.00011474554586555497, 'mad': 0.2187151163816452, 'iou': 0.5929808568824065, 'vis': 1454, 'full': 1850} flip {'ncc': -0.008079595507196773, 'edge_ncc': 0.0018626262890262387, 'mad': 0.20609933137893677, 'iou': 0.5715575620767495, 'vis': 1446, 'full': 1836}
150 base {'ncc': -0.18110984926921736, 'edge_ncc': 0.03463685057347822, 'mad': 0.21867993474006653, 'iou': 0.5608048993875766, 'vis': 1382, 'full': 1939} flip {'ncc': 0.0125408095336052 …[+6796 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
ims=[]
for i in [140,160,180,200,240,250,280,300,340,380,400,440,500]:
 im=cv2.imread(f'/work/scratch/analysis/flipcmp_{i:03d}.jpg');
 if im is None:continue
 im=cv2.resize(im,(576,288)); cv2.putText(im,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);ims.append(im)
h,w=ims[0].shape[:2];sheet=np.zeros((4*h,4*w,3),np.uint8)
for k,im in enumerate(ims):sheet[(k//4)*h:(k//4+1)*h,(k%4)*w:(k%4+1)*w]=im
cv2.imwrite('/work/scratch/analysis/flip_sheet.jpg',sheet)
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
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); uv=np.asarray(m.visual.uv); im=np.asarray(m.visual.material.baseColorTexture); print(im.shape)
for flip in [False,True]:
 v=uv[:,1].copy();
 if flip:v=1-v
 x=np.clip((uv[:,0]*(im.shape[1]-1)).astype(int),0,im.shape[1]-1); y=np.clip((v*(im.shape[0]-1)).astype(int),0,im.shape[0]-1)
 c=im[y,x]; print('flip',flip,'mean',c.mean(0),'black frac',(c.max(1)<5).mean(),'minmax',c.min(),c.max(),'unique',len(np.unique(c.reshape(-1,3),axis=0)))
# UV face coverage texture bbox
print('uv bbox',uv.min(0),uv.max(0),'mean',uv.mean(0))
PY"
```
<details><summary>output</summary>

```
(2458, 2301, 3)
flip False mean [75.75051646 44.99349958 14.35109775] black frac 0.465287397005721 minmax 0 255 unique 11493
flip True mean [150.46896984  89.70749455  27.8335313 ] black frac 3.743699523869488e-05 minmax 0 255 unique 16390
uv bbox [0.001521   0.00142401] [0.99847901 0.99857599] mean [0.52080547 0.50846564]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/uv=torch.tensor(np.asarray(m.visual.uv),dtype=torch.float32,device='cuda')/uv_np=np.asarray(m.visual.uv).copy(); uv_np[:,1]=1.0-uv_np[:,1]; uv=torch.tensor(uv_np,dtype=torch.float32,device='cuda')/\" /work/scratch/analysis/render_flips.py
python /work/scratch/analysis/render_flips.py"
```
<details><summary>output</summary>

```
100 base {'ncc': 0.22571609283869826, 'edge_ncc': 0.032223686627476045, 'mad': 0.09066250175237656, 'iou': 0.7568445475638051, 'vis': 1732, 'full': 1732} flip {'ncc': 0.14030406814252794, 'edge_ncc': -0.01214531248156523, 'mad': 0.08796539157629013, 'iou': 0.7259770114942529, 'vis': 1700, 'full': 1700}
140 base {'ncc': 0.1403250622708031, 'edge_ncc': 0.004927528020218432, 'mad': 0.13092753291130066, 'iou': 0.5929808568824065, 'vis': 1454, 'full': 1850} flip {'ncc': -0.03581745566725873, 'edge_ncc': 0.02375731956447904, 'mad': 0.1327773630619049, 'iou': 0.5715575620767495, 'vis': 1446, 'full': 1836}
150 base {'ncc': 0.22561686638085313, 'edge_ncc': 0.04563354634543452, 'mad': 0.12269800156354904, 'iou': 0.5608048993875766, 'vis': 1382, 'full': 1939} flip {'ncc': 0.03560726837793429, 'edge_n …[+6788 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
ims=[]
for i in [100,140,160,180,200,240,250,280,300,340,380,400,440,500]:
 im=cv2.imread(f'/work/scratch/analysis/flipcmp_{i:03d}.jpg');
 if im is None:continue
 im=cv2.resize(im,(576,288)); cv2.putText(im,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);ims.append(im)
h,w=ims[0].shape[:2];sheet=np.zeros((4*h,4*w,3),np.uint8)
for k,im in enumerate(ims):sheet[(k//4)*h:(k//4+1)*h,(k%4)*w:(k%4+1)*w]=im
cv2.imwrite('/work/scratch/analysis/flip_sheet_fixed.jpg',sheet)
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
from pathlib import Path
p=Path('/work/inputs');cap=cv2.VideoCapture(str(p/'video.mp4')); frames=[]
while True:
 ok,im=cap.read();
 if not ok:break
 frames.append(im)
with h5py.File(p/'masks.h5','r') as f:obj=f['object'][:]
idx=list(range(150,351,5));tw,th=240,180; cols=10;rows=(len(idx)+cols-1)//cols
sheet=np.zeros((rows*th,cols*tw,3),np.uint8)
for k,i in enumerate(idx):
 yy,xx=np.where(obj[i]);
 if len(xx):cx,cy=int(np.median(xx)),int(np.median(yy))
 else: cx,cy=768,576
 # fixed-ish box 160x120 then resize to 240x180
 x0=max(0,min(cx-80,1536-160));y0=max(0,min(cy-60,1152-120));crop=frames[i][y0:y0+120,x0:x0+160]
 crop=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_CUBIC)
 cv2.rectangle(crop,(0,0),(48,24),(0,0,0),-1);cv2.putText(crop,str(i),(3,19),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
 sheet[(k//cols)*th:(k//cols+1)*th,(k%cols)*tw:(k%cols+1)*tw]=crop
cv2.imwrite('/work/scratch/analysis/carry_150_350.jpg',sheet)
# 350-460
idx=list(range(350,461,5));rows=(len(idx)+cols-1)//cols;sheet=np.zeros((rows*th,cols*tw,3),np.uint8)
for k,i in enumerate(idx):
 yy,xx=np.where(obj[i]);
 if len(xx):cx,cy=int(np.median(xx)),int(np.median(yy))
 else: cx,cy=768,576
 x0=max(0,min(cx-80,1536-160));y0=max(0,min(cy-60,1152-120));crop=frames[i][y0:y0+120,x0:x0+160]
 crop=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_CUBIC);cv2.rectangle(crop,(0,0),(48,24),(0,0,0),-1);cv2.putText(crop,str(i),(3,19),cv2.FONT_HERSHEY_SIMPLEX,.65, …[+149 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/render_texture_sheet.py <<'PY'
# run render_flips setup without loop
src=open('/work/scratch/analysis/render_flips.py').read().split(\"rows=[]\")[0]
exec(src)
ims=[]
for i in [100,150,180,200,220,240,250,260,280,300,320,340,360,380,400,420,440,500]:
 c=project_center(T[i]); x0=max(0,int(c[0])-80); y0=max(0,int(c[1])-60); w,h=160,120
 if x0+w>W:x0=W-w
 if y0+h>H:y0=H-h
 img=frames[i][y0:y0+h,x0:x0+w]
 r0,m0=render(T[i],x0,y0,w,h); r1,m1=render(T[i]@S,x0,y0,w,h)
 def rr(rgb,m):
  z=np.zeros_like(img); z[m]=np.clip(rgb[m]*255,0,255).astype(np.uint8)[:,::-1]; return z
 row=np.concatenate([img,rr(r0,m0),rr(r1,m1)],axis=1)
 ims.append(row)
h,w=ims[0].shape[:2];cols=3;rows=(len(ims)+cols-1)//cols;sheet=np.zeros((rows*h,cols*w,3),np.uint8)
for k,im in enumerate(ims):
 cv2.rectangle(im,(0,0),(45,24),(0,0,0),-1);cv2.putText(im,str([100,150,180,200,220,240,250,260,280,300,320,340,360,380,400,420,440,500][k]),(3,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
 sheet[(k//cols)*h:(k//cols+1)*h,(k%cols)*w:(k%cols+1)*w]=im
cv2.imwrite('/work/scratch/analysis/texture_variants.jpg',sheet)
PY
python /work/scratch/analysis/render_texture_sheet.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np
x=torch.load('/pipeline/inference/refinement_checkpoint.pth',map_location='cpu',weights_only=False)
for k,v in x.items():
 if isinstance(v,torch.Tensor): print(k,v.shape,v.dtype,float(v.min()),float(v.max()))
 elif isinstance(v,dict): print(k,'dict',list(v.keys()))
 else: print(k,type(v),str(v)[:100])
print('\\npostopt config')
r=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for k,v in r['postopt'].items():
 print(k, type(v), str(v)[:1000].replace('\\n',' ') if not isinstance(v,(dict,list)) else (list(v.keys()) if isinstance(v,dict) else f'list {len(v)}'))
PY"
```
<details><summary>output</summary>

```
schema <class 'str'> cari4d.mhr_postopt_smplh_parity.v10
config dict ['num_steps', 'batch_size', 'device', 'frame_start', 'frame_limit', 'lr', 'warmup_steps', 'schedule_steps', 'object_surface_samples', 'penetration_surface_samples', 'penetration_point_reduction', 'penetration_human_mesh_mode', 'penetration_collision_proxy_path', 'penetration_bbox_rejection', 'penetration_bbox_margin_m', 'hand_surface_mode', 'hand_surface_spec_path', 'contact_activation_distance_m', 'w_contact', 'w_silhouette', 'w_penetration', 'w_human_pose_prior', 'human_pose_prior_beta', 'w_temporal', 'w_object_translation_prior', 'penetration_start_fraction', 'symmetric_object', 'random_seed', 'report_every', 'diagnostics_every', 'save_every', 'checkpoint_path', 'freeze_object_rotation', 'freeze_body_internal_translati …[+3376 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); Tf=z['T_cam_mesh_foundationpose'].astype(float); Tc=z['T_cam_mesh_coconet'].astype(float)
with open('/pipeline/foundationpose/episode_000027.pkl','rb') as f:x=pickle.load(f)
fp=x['fp_poses_cam'][:,0]
A=np.array(json.load(open('/pipeline/export/episode_000027/wild_export.json'))['source_object_mesh_to_aligned_transform'])
for form in ['fp','fpA','Afp']:
 P=fp if form=='fp' else (fp@A if form=='fpA' else A@fp)
 print(form, np.abs(P-Tf).max(), np.percentile(np.abs(P-Tf),[50,90,99]))
# Coco camera is likely aligned
print('coco vs pkl obj?',Tc[0],x['obj_pose_world'][0])
print('fpA0', (fp@A)[0]);print('Tf0',Tf[0]);print('coco0',Tc[0])
# transforms / projection check using input mesh and fpA vs baseline
PY"
```
<details><summary>output</summary>

```
fp 19.9167001247406 [ 0.1821386   1.07242434 15.98394149]
fpA 19.912119455028463 [1.86418373e-03 9.42208191e-01 1.59800746e+01]
Afp 19.892461790428534 [ 0.24945399  1.41359426 15.96006791]
coco vs pkl obj? [[-0.98614395 -0.04535127 -0.15957277  0.29267722]
 [-0.05807005  0.99541813  0.07596498  0.92778736]
 [ 0.15539657  0.08417883 -0.98425901  3.57949901]
 [ 0.          0.          0.          1.        ]] [[-0.06861859  0.99242747 -0.10187833  0.29186153]
 [ 0.991007    0.0795647   0.10758597  0.96738714]
 [ 0.11487724 -0.09357979 -0.98896205  3.7400458 ]
 [ 0.          0.          0.          1.        ]]
fpA0 [[-0.98983345 -0.06412219 -0.12695693  0.29582172]
 [-0.07736462  0.99174255  0.10228171  0.9728078 ]
 [ 0.11935012  0.11106388 -0.98662062  3.74209366]
 [ 0.          0.          …[+446 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/score_fp.py <<'PY'
src=open('/work/scratch/analysis/render_flips.py').read().split(\"rows=[]\")[0]
exec(src)
import pickle
with open('/pipeline/foundationpose/episode_000027.pkl','rb') as f:x=pickle.load(f)
A=np.array(json.load(open('/pipeline/export/episode_000027/wild_export.json'))['source_object_mesh_to_aligned_transform'])
Tfp=np.einsum('ij,fjk->fik',Twc, x['fp_poses_cam'][:,0]@A)
for i in [0,50,100,140,150,160,170,180,190,195,200,210,220,240,250,260,280,300,320,340,360,380,400,420,440,460,500,560]:
 c=project_center(T[i]); x0=max(0,int(c[0])-96);y0=max(0,int(c[1])-72);w,h=192,144
 if x0+w>W:x0=W-w
 if y0+h>H:y0=H-h
 img=frames[i][y0:y0+h,x0:x0+w];om=obj[i][y0:y0+h,x0:x0+w];pm=per[i][y0:y0+h,x0:x0+w]
 vals=[]
 for name,P in [('base',T[i]),('baseS',T[i]@S),('fp',Tfp[i]),('fpS',Tfp[i]@S)]:
  rgb,m=render(P,x0,y0,w,h); ss=score(img,rgb,m,om,pm); vals.append((name,ss))
 print(i,' '.join(f\"{n}:ncc={s['ncc']:.2f},mad={s['mad']:.3f},iou={s['iou']:.2f},vis={s['vis']}\" for n,s in vals),flush=True)
PY
python /work/scratch/analysis/score_fp.py"
```
<details><summary>output</summary>

```
0 base:ncc=0.23,mad=0.091,iou=0.70,vis=1512 baseS:ncc=0.05,mad=0.098,iou=0.67,vis=1479 fp:ncc=-0.17,mad=0.085,iou=0.33,vis=1705 fpS:ncc=-0.12,mad=0.090,iou=0.34,vis=1680
50 base:ncc=0.21,mad=0.091,iou=0.76,vis=1748 baseS:ncc=0.12,mad=0.094,iou=0.73,vis=1711 fp:ncc=-0.12,mad=0.086,iou=0.32,vis=1755 fpS:ncc=-0.16,mad=0.092,iou=0.33,vis=1729
100 base:ncc=0.23,mad=0.091,iou=0.76,vis=1732 baseS:ncc=0.14,mad=0.088,iou=0.73,vis=1700 fp:ncc=-0.14,mad=0.092,iou=0.31,vis=1795 fpS:ncc=-0.16,mad=0.094,iou=0.31,vis=1760
140 base:ncc=0.14,mad=0.131,iou=0.59,vis=1454 baseS:ncc=-0.04,mad=0.133,iou=0.57,vis=1446 fp:ncc=-0.07,mad=0.109,iou=0.20,vis=1240 fpS:ncc=-0.21,mad=0.114,iou=0.20,vis=1220
150 base:ncc=0.23,mad=0.123,iou=0.56,vis=1382 baseS:ncc=0.04,mad=0.127,iou=0.55,vis=1359 fp:ncc=-0.06,mad=0.106,io …[+2516 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/score_all_flips.py <<'PY'
src=open('/work/scratch/analysis/render_flips.py').read().split(\"rows=[]\")[0]
exec(src)
# additional phase-consistent flip candidates (R @ S only; S is mesh-frame)
def metrics(img,rgb,mask,om,pm):
 obs=om&~pm
 # candidate common visible pixels; allow 2px shift of candidate to reduce translation nuisance
 best=None
 g1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
 g2=cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
 # local highpass normalized
 h1=g1-cv2.GaussianBlur(g1,(0,0),1.2); h2=g2-cv2.GaussianBlur(g2,(0,0),1.2)
 for dx in range(-2,3):
  for dy in range(-2,3):
   M=np.float32([[1,0,dx],[0,1,dy]]); m2=cv2.warpAffine(mask.astype(np.uint8),M,(mask.shape[1],mask.shape[0]))>0; h2s=cv2.warpAffine(h2,M,(h2.shape[1],h2.shape[0]))
   vis=m2&obs
   if vis.sum()<40: continue
   a=h1[vis];b=h2s[vis]
   if a.std()<1e-6 or b.std()<1e-6:ncc=0
   else:ncc=float(np.corrcoef(a,b)[0,1])
   # precision: candidate visible in observed or 2px dilated obs; recall observed covered by candidate
   obsd=cv2.dilate(obs.astype(np.uint8),np.ones((5,5),np.uint8))>0
   prec=float((m2&obsd).sum()/max(1,m2.sum()))
   rec=float((obs&cv2.dilate(m2.astype(np.uint8),np.ones((5,5),np.uint8))>0).sum()/max(1,obs.sum()))
   # bbox/centroid
   if best is None or ncc>best[0]:best=(ncc,prec,rec,int(vis.sum()),int(m2.sum()),dx,dy)
 if best is None:return (0,0,0,0,0,0,0,0)
 # exact-v …[+713 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/score_all_flips.py", line 30, in <module>
    arr=np.zeros((F,2,9),np.float32)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: only integer tensors of a single element can be converted to an index

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/arr=np.zeros((F,2,9),np.float32)/NF=len(T); arr=np.zeros((NF,2,9),np.float32)/; s/for i in range(F):/for i in range(NF):/; s/if i%25==0:/if i%25==0:/' /work/scratch/analysis/score_all_flips.py
python /work/scratch/analysis/score_all_flips.py"
```
<details><summary>output</summary>

```
0 [1.048848e-01 8.930854e-01 8.921680e-01 1.405000e+03 1.721000e+03
 1.000000e+00 0.000000e+00 1.762000e+03 6.907767e-01] [ 9.1641121e-02  9.3125737e-01  9.4097614e-01  1.4930000e+03
  1.7020000e+03 -2.0000000e+00  1.0000000e+00  1.7620000e+03
  6.7020249e-01]
25 [ 1.4298995e-01  9.9550813e-01  9.6972656e-01  1.7370000e+03
  1.7810000e+03 -2.0000000e+00  2.0000000e+00  2.0480000e+03
  8.2159847e-01] [ 1.9312155e-01  1.0000000e+00  9.6484375e-01  1.7220000e+03
  1.7460000e+03 -2.0000000e+00  2.0000000e+00  2.0480000e+03
  7.7871543e-01]
50 [8.9125946e-02 1.0000000e+00 9.3600392e-01 1.6860000e+03 1.7480000e+03
 0.0000000e+00 1.0000000e+00 2.0470000e+03 8.1665868e-01] [ 1.522920e-01  1.000000e+00  9.526136e-01  1.685000e+03  1.711000e+03
 -2.000000e+00  2.000000e+00  2.047000e+03  7.701366e-0 …[+1642 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/score_all_flips.py';s=open(p).read()
s=s.replace(\"if best is None:return (0,0,0,0,0,0,0,0)\",\"if best is None:return (0,0,0,0,0,0,0,0,0)\")
s=s.replace(\"for i in range(NF):\\n c=project_center\", \"for i in range(NF):\\n if i % 20 == 0:\\n  ctx=dr.RasterizeCudaContext()\\n c=project_center\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/score_all_flips.py"
```
<details><summary>output</summary>

```
0 [1.048848e-01 8.930854e-01 8.921680e-01 1.405000e+03 1.721000e+03
 1.000000e+00 0.000000e+00 1.762000e+03 6.907767e-01] [ 9.1641121e-02  9.3125737e-01  9.4097614e-01  1.4930000e+03
  1.7020000e+03 -2.0000000e+00  1.0000000e+00  1.7620000e+03
  6.7020249e-01]
25 [ 1.4298995e-01  9.9550813e-01  9.6972656e-01  1.7370000e+03
  1.7810000e+03 -2.0000000e+00  2.0000000e+00  2.0480000e+03
  8.2159847e-01] [ 1.9312155e-01  1.0000000e+00  9.6484375e-01  1.7220000e+03
  1.7460000e+03 -2.0000000e+00  2.0000000e+00  2.0480000e+03
  7.7871543e-01]
50 [8.9125946e-02 1.0000000e+00 9.3600392e-01 1.6860000e+03 1.7480000e+03
 0.0000000e+00 1.0000000e+00 2.0470000e+03 8.1665868e-01] [ 1.522920e-01  1.000000e+00  9.526136e-01  1.685000e+03  1.711000e+03
 -2.000000e+00  2.000000e+00  2.047000e+03  7.701366e-0 …[+5493 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/analysis/flip_scores.npz'); s=z['scores'] # F,2, [ncc,prec,rec,vis,full,dx,dy,obs,iou]
d=s[:,0,0]-s[:,1,0]
print('ncc diff stats',np.percentile(d,[0,5,25,50,75,95,100]),'base wins',np.mean(d>0),'conf>0.03',np.mean(np.abs(d)>.03))
print('by 10 frame block: mean diff, base wins, mean IoU diff, mean base ncc')
for a in range(0,len(s),10):
 b=min(len(s),a+10);dd=d[a:b];print(f'{a:3d}-{b-1:3d} diff {dd.mean():+.3f} med {np.median(dd):+.3f} win {np.mean(dd>0):.2f} iou {np.mean(s[a:b,0,8]-s[a:b,1,8]):+.3f} ncc {s[a:b,0,0].mean():+.3f} vis {s[a:b,0,3].mean():.0f}')
# rolling means window 9,15,30 sign
for w in [5,9,15,30]:
 c=np.convolve(d,np.ones(w),'same')/np.convolve(np.ones(len(d)),np.ones(w),'same')
 print('\\nwindow',w,'sign changes',np.where(np.diff(np.sign(c))"'!=0)[0])
 for i in range(0,len(s),20):print(i,round(c[i],3),end='"'; ')
 print()
PY"
```
<details><summary>output</summary>

```
 10- 19 diff -0.057 med -0.059 win 0.00 iou +0.040 ncc +0.101 vis 1718
 20- 29 diff -0.062 med -0.067 win 0.00 iou +0.044 ncc +0.118 vis 1729
 30- 39 diff -0.070 med -0.071 win 0.00 iou +0.040 ncc +0.114 vis 1727
 40- 49 diff -0.067 med -0.068 win 0.00 iou +0.039 ncc +0.105 vis 1696
 50- 59 diff -0.073 med -0.067 win 0.00 iou +0.042 ncc +0.095 vis 1658
 60- 69 diff -0.067 med -0.067 win 0.00 iou +0.042 ncc +0.105 vis 1672
 70- 79 diff -0.048 med -0.064 win 0.20 iou +0.042 ncc +0.106 vis 1660
 80- 89 diff -0.070 med -0.070 win 0.00 iou +0.043 ncc +0.107 vis 1665
 90- 99 diff -0.067 med -0.071 win 0.00 iou +0.045 ncc +0.107 vis 1690
100-109 diff -0.064 med -0.066 win 0.00 iou +0.043 ncc +0.087 vis 1650
110-119 diff -0.055 med -0.052 win 0.00 iou +0.043 ncc +0.092 vis 1690
120-129 diff -0.034 …[+5207 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
plane=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=plane[:3];d=plane[3]
for i in [0,5,10,50,100,128,130,140,150,160,180,200,300,400,440,450,500,560]:
 w=V@T[i,:3,:3].T+T[i,:3,3]; dist=w@n+d
 print(i,'dist min/max/pcts',dist.min(),np.percentile(dist,[0.1,1,5,50,95,99,99.9]),dist.max(),'t',T[i,:3,3].round(3))
PY"
```
<details><summary>output</summary>

```
0 dist min/max/pcts 0.028852724261346552 [0.03018328 0.03389574 0.0399573  0.05465237 0.10074567 0.10901162
 0.11190976] 0.11397807671941584 t [0.505 1.105 3.787]
5 dist min/max/pcts 0.033957247527905965 [0.035246   0.03911348 0.045444   0.0602806  0.10657112 0.1150085
 0.11795063] 0.11973295750620339 t [0.499 1.1   3.78 ]
10 dist min/max/pcts 0.031939449232488926 [0.03339793 0.03828768 0.04383347 0.05630555 0.103142   0.10820591
 0.11033964] 0.11160473190767761 t [0.502 1.103 3.782]
50 dist min/max/pcts 0.016541975690299138 [0.01800046 0.02289021 0.028436   0.04090808 0.08774453 0.09280844
 0.09494217] 0.09620725836548805 t [0.508 1.116 3.825]
100 dist min/max/pcts 0.0077022644928288475 [0.00916074 0.0140505  0.01959628 0.03206837 0.07890482 0.08396873
 0.08610246] 0.08736754716801776 t [ …[+2155 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5']:
 print('\\n',p)
 with h5py.File(p,'r') as f:
  def vis(name,obj):
   if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype,obj.chunks)
   else: print(name,'GROUP',obj.keys())
  f.visititems(vis)
  print('attrs',dict(f.attrs))
PY"
```
<details><summary>output</summary>

```
aligned GROUP <KeysViewHDF5 ['front_stereo_camera_left']>
aligned/front_stereo_camera_left (565,) object None
alignment GROUP <KeysViewHDF5 ['front_stereo_camera_left']>
alignment/front_stereo_camera_left GROUP <KeysViewHDF5 ['scale', 'shift', 'valid_count']>
alignment/front_stereo_camera_left/scale (565,) float32 None
alignment/front_stereo_camera_left/shift (565,) float32 None
alignment/front_stereo_camera_left/valid_count (565,) int32 None
frame_names GROUP <KeysViewHDF5 ['front_stereo_camera_left']>
frame_names/front_stereo_camera_left (565,) object None
raw GROUP <KeysViewHDF5 ['front_stereo_camera_left']>
raw/front_stereo_camera_left (565,) object None
attrs {'complete': True, 'depth_alignment_input_identity_json': '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-norm …[+1453 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 for grp in ['raw','aligned']:
  d=f[grp]['front_stereo_camera_left']; a=d[0]; print(grp,type(a),getattr(a,'shape',None),getattr(a,'dtype',None),repr(a)[:500])
 print('scale shift',f['alignment/front_stereo_camera_left/scale'][:10],f['alignment/front_stereo_camera_left/shift'][:10])
PY"
```
<details><summary>output</summary>

```
raw <class 'numpy.ndarray'> (613367,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
aligned <class 'numpy.ndarray'> (656599,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
scale shift [1.0839474 1.0878392 1.0997002 1.1036792 1.1057994 1.1037314 1.104167
 1.0957928 1.1101794 1.1076763] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 for grp in ['raw','aligned']:
  for i in [0,100,300,500]:
   a=f[grp]['front_stereo_camera_left'][i]; im=cv2.imdecode(a,cv2.IMREAD_UNCHANGED); print(grp,i,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[1,10,50,90,99]))
PY"
```
<details><summary>output</summary>

```
raw 0 (1152, 1536) uint16 1599 8888 [1660. 1970. 3418. 5971. 8064.]
raw 100 (1152, 1536) uint16 1603 8780 [1670. 1982. 3519. 6062. 8022.]
raw 300 (1152, 1536) uint16 1556 8725 [1624. 1931. 3461. 5933. 7791.]
raw 500 (1152, 1536) uint16 1562 8821 [1626. 1933. 3453. 6021. 8009.]
aligned 0 (1152, 1536) uint16 1733 9634 [1799. 2135. 3704. 6472. 8740.]
aligned 100 (1152, 1536) uint16 1696 9293 [1767. 2097. 3724. 6416. 8491.]
aligned 300 (1152, 1536) uint16 1724 9667 [1799. 2139. 3834. 6573. 8632.]
aligned 500 (1152, 1536) uint16 1691 9554 [1761. 2093. 3740. 6521. 8674.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/score_all_flips.py';s=open(p).read()
# add direct ncc to best tuple: currently ncc highpass; set direct variable and prepend
s=s.replace(\"   if a.std()<1e-6 or b.std()<1e-6:ncc=0\\n   else:ncc=float(np.corrcoef(a,b)[0,1])\", \"   if a.std()<1e-6 or b.std()<1e-6:ncc=0\\n   else:ncc=float(np.corrcoef(a,b)[0,1])\\n   ad=g1[vis];bd=h2s[vis]\\n   direct=float(np.corrcoef(ad,bd)[0,1]) if ad.std()>1e-6 and bd.std()>1e-6 else 0\")
s=s.replace(\"if best is None or ncc>best[0]:best=(ncc,prec,rec,int(vis.sum()),int(m2.sum()),dx,dy)\",\"if best is None or ncc>best[0]:best=(ncc,direct,prec,rec,int(vis.sum()),int(m2.sum()),dx,dy)\")
s=s.replace(\"if best is None:return (0,0,0,0,0,0,0,0,0)\",\"if best is None:return (0,0,0,0,0,0,0,0,0,0)\")
s=s.replace(\"arr=np.zeros((NF,2,9),np.float32)\",\"arr=np.zeros((NF,2,10),np.float32)\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/score_all_flips.py"
```
<details><summary>output</summary>

```
0 [1.0488480e-01 6.6121253e-03 8.9308542e-01 8.9216799e-01 1.4050000e+03
 1.7210000e+03 1.0000000e+00 0.0000000e+00 1.7620000e+03 6.9077671e-01] [ 9.1641121e-02 -4.7006190e-02  9.3125737e-01  9.4097614e-01
  1.4930000e+03  1.7020000e+03 -2.0000000e+00  1.0000000e+00
  1.7620000e+03  6.7020249e-01]
25 [ 1.4298995e-01  2.9403033e-02  9.9550813e-01  9.6972656e-01
  1.7370000e+03  1.7810000e+03 -2.0000000e+00  2.0000000e+00
  2.0480000e+03  8.2159847e-01] [ 1.9312155e-01  3.1156700e-02  1.0000000e+00  9.6484375e-01
  1.7220000e+03  1.7460000e+03 -2.0000000e+00  2.0000000e+00
  2.0480000e+03  7.7871543e-01]
50 [8.9125946e-02 9.6963048e-03 1.0000000e+00 9.3600392e-01 1.6860000e+03
 1.7480000e+03 0.0000000e+00 1.0000000e+00 2.0470000e+03 8.1665868e-01] [ 1.522920e-01  1.461826e-02  1.000000e+00   …[+6216 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/depth_residual.py <<'PY'
import numpy as np,torch,trimesh,json,cv2,h5py
import nvdiffrast.torch as dr
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
p=Path('/work/inputs'); m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(np.asarray(m.vertices),dtype=torch.float32,device='cuda'); Faces=torch.tensor(np.asarray(m.faces),dtype=torch.int32,device='cuda')
cam=json.load(open(p/'camera.json')); K=np.array(cam['K_pipeline']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc); T=np.load(p/'reconstruction.npz')['T_world_mesh'].astype(np.float32)
S=np.eye(4,dtype=np.float32);S[:3,:3]=Rot.from_euler('y',np.pi).as_matrix()
with h5py.File(p/'masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ctx=dr.RasterizeCudaContext();H,W=1152,1536

def render(Twm,x0,y0,w,h):
 k=K.copy();k[0,2]-=x0;k[1,2]-=y0;Tcm=(Tcw@Twm).astype(np.float32);R=torch.tensor(Tcm[:3,:3],device='cuda');t=torch.tensor(Tcm[:3,3],device='cuda');vc=V@R.T+t;x,y,z=vc[:,0],vc[:,1],vc[:,2];fx,fy=k[0,0],k[1,1];cx,cy=k[0,2],k[1,2]
 clip=torch.stack([2*fx/w*x+(2*cx/w-1)*z,2*fy/h*y+(2*cy/h-1)*z,z,z],1)[None]
 rast,_=dr.rasterize(ctx,clip,Faces,resolution=[h,w]);dz,_=dr.interpolate(z[None],rast,Faces);mask=rast[0,...,3]>0
 return dz[0].cpu().numpy(),mask.cpu().numpy()
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: ds=f['aligned']['front_stereo_camera_left']
for i in list(range(0,150,10))+[1 …[+803 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/depth_residual.py", line 19, in <module>
    im=cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
                    ~~^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 870, in __getitem__
    if self._fast_read_ok and (new_dtype is None):
       ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/base.py", line 534, in __get__
    value = obj.__dict__[self.func.__name__] = self.func(obj)
                                               ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py …[+926 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/depth_residual.py';s=open(p).read();s=s.replace(\"with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: ds=f['aligned']['front_stereo_camera_left']\\nfor i in list(range(0,150,10))+[160,180,200,220,250,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560]:\\n im=cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000\", \"inds=list(range(0,150,10))+[160,180,200,220,250,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560]\\nwith h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:\\n ds=f['aligned']['front_stereo_camera_left']\\n depths={i:cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in inds}\\nfor i in inds:\\n im=depths[i]\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/depth_residual.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/depth_residual.py", line 27, in <module>
    rr,m0=render(T[i],x0,y0,w,h);rr1,m1=render(T[i]@S,x0,y0,w,h)
          ^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/depth_residual.py", line 15, in render
    rast,_=dr.rasterize(ctx,clip,Faces,resolution=[h,w]);dz,_=dr.interpolate(z[None],rast,Faces);mask=rast[0,...,3]>0
                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return  …[+406 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/depth_residual.py';s=open(p).read().replace(\"dz,_=dr.interpolate(z[None],rast,Faces)\",\"dz,_=dr.interpolate(z.contiguous()[None],rast,Faces)\")
s=s.replace(\"for i in inds:\\n im=depths[i]\", \"for kk,i in enumerate(inds):\\n if kk%15==0: ctx=dr.RasterizeCudaContext()\\n im=depths[i]\")
open(p,'w').write(s)
PY
python /work/scratch/analysis/depth_residual.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/depth_residual.py", line 31, in <module>
    if len(r): print(i,name,'n',len(r),'res med/mean/p10p90',round(float(np.median(r-z)),3),round(float(np.mean(r-z)),3),np.round(np.percentile(r-z,[10,90]),3),'zdiff std',round(float(np.std(r-z)),3))
                                                                                   ~^~
ValueError: operands could not be broadcast together with shapes (944,) (944,293827) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dr.interpolate(z.contiguous()\\[None\\],rast,Faces)/dr.interpolate(z.contiguous()[:,None],rast,Faces)/' /work/scratch/analysis/depth_residual.py
python /work/scratch/analysis/depth_residual.py"
```
<details><summary>output</summary>

```
0 base n 944 res med/mean/p10p90 -0.077 -0.072 [-0.196  0.106] zdiff std 0.143
0 flip n 917 res med/mean/p10p90 -0.076 -0.072 [-0.201  0.109] zdiff std 0.147
10 base n 1202 res med/mean/p10p90 -0.014 -0.004 [-0.136  0.15 ] zdiff std 0.11
10 flip n 1162 res med/mean/p10p90 -0.014 -0.004 [-0.141  0.153] zdiff std 0.112
20 base n 1167 res med/mean/p10p90 -0.084 -0.071 [-0.202  0.083] zdiff std 0.111
20 flip n 1114 res med/mean/p10p90 -0.082 -0.07 [-0.207  0.093] zdiff std 0.115
30 base n 1189 res med/mean/p10p90 -0.093 -0.081 [-0.215  0.074] zdiff std 0.112
30 flip n 1143 res med/mean/p10p90 -0.092 -0.081 [-0.219  0.079] zdiff std 0.113
40 base n 1156 res med/mean/p10p90 -0.073 -0.06 [-0.193  0.097] zdiff std 0.112
40 flip n 1114 res med/mean/p10p90 -0.071 -0.057 [-0.197  0.105] zdiff std 0.1 …[+4846 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py,cv2
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc)
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:ds=f['aligned']['front_stereo_camera_left']
import h5py as h
with h.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:]
for i in [0,10,100,300,500]:
 d=cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED)/1000
 p=T[i,:3,3];pc=Tcw[:3,:3]@p+Tcw[:3,3]
 yy,xx=np.where(obj[i]);vals=d[yy,xx]
 print(i,'Tc',pc.round(3),'mask depth',np.percentile(vals,[10,50,90]).round(3),'render bbox depth range?',)
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
import numpy as np,json,h5py,cv2
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc)
inds=[0,10,100,300,500]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 ds=f['aligned']['front_stereo_camera_left'];dsx={i:cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED)/1000 for i in inds}
with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:]
for i in inds:
 d=dsx[i];p=T[i,:3,3];pc=Tcw[:3,:3]@p+Tcw[:3,3];yy,xx=np.where(obj[i]);vals=d[yy,xx]
 print(i,'Tc',pc.round(3),'mask depth',np.percentile(vals,[10,50,90]).round(3))
PY"
```
<details><summary>output</summary>

```
0 Tc [0.34  1.007 3.652] mask depth [3.6   3.695 3.89 ]
10 Tc [0.338 1.005 3.646] mask depth [3.646 3.761 3.901]
100 Tc [0.344 1.023 3.715] mask depth [3.541 3.636 3.772]
300 Tc [-0.395  0.171  3.706] mask depth [3.036 3.095 3.669]
500 Tc [-0.488  0.229  2.808] mask depth [2.544 2.576 2.694]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py,cv2
z=np.load('/work/inputs/reconstruction.npz');VH=z['human_vertices_world'].astype(float);T=z['T_world_mesh'].astype(float)
cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);K=np.array(cam['K_pipeline'])
inds=[0,10,50,100,130,160,180,200,220,250,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 ds=f['aligned']['front_stereo_camera_left'];D={i:cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED)/1000 for i in inds}
with h5py.File('/work/inputs/masks.h5','r') as f:pm=f['person'][:];om=f['object'][:]
for i in inds:
 p=VH[i];pc=p@Tcw[:3,:3].T+Tcw[:3,3];uv=(pc[:,:2]/pc[:,2:3])@K[:2,:2].T+K[:2,2];u=np.round(uv[:,0]).astype(int);v=np.round(uv[:,1]).astype(int);ok=(u>=0)&(u<1536)&(v>=0)&(v<1152);ok&=pc[:,2]>0
 u=u[ok];v=v[ok];cz=pc[ok,2];d=D[i][v,u];
 # valid observed depth and points near person mask; depth should be >= vertex z for front faces; ratio for points close to surface choose quantiles
 val=(d>0)&(d<20)
 ratios=d[val]/cz[val]
 dif=d[val]-cz[val]
 # visible surface vertices tend min positive diff
 print(i,'hum ratio pct',np.round(np.percentile(ratios,[5,25,50,75,95]),3),'diff pct',np.round(np.percentile(dif,[5,25,50,75,95]),3),'camera center z',round((Tcw@T[i])[2,3],3))
PY"
```
<details><summary>output</summary>

```
0 hum ratio pct [0.898 0.921 0.965 1.028 3.035] diff pct [-0.266 -0.205 -0.089  0.072  5.226] camera center z 3.652
10 hum ratio pct [0.888 0.92  0.953 1.023 2.442] diff pct [-0.297 -0.21  -0.12   0.063  3.645] camera center z 3.646
50 hum ratio pct [0.906 0.927 0.954 0.997 2.102] diff pct [-0.288 -0.224 -0.14  -0.009  3.396] camera center z 3.69
100 hum ratio pct [0.837 0.89  0.94  0.988 2.351] diff pct [-0.611 -0.387 -0.208 -0.04   4.707] camera center z 3.715
130 hum ratio pct [0.837 0.889 0.937 1.013 1.227] diff pct [-0.576 -0.401 -0.217  0.043  0.782] camera center z 3.608
160 hum ratio pct [0.84  0.908 0.949 0.995 1.193] diff pct [-0.559 -0.331 -0.174 -0.018  0.707] camera center z 3.612
180 hum ratio pct [0.881 0.926 0.973 1.656 1.979] diff pct [-0.425 -0.26  -0.092  2.38   3.391] c …[+2145 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/depth_correct_test.py <<'PY'
# setup from depth_residual
src=open('/work/scratch/analysis/depth_residual.py').read().split(\"inds=list(\")[0]
exec(src)
inds=[0,50,100,130,160,180,200,220,250,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 ds=f['aligned']['front_stereo_camera_left'];D={i:cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in inds}
for kk,i in enumerate(inds):
 if kk%15==0:ctx=dr.RasterizeCudaContext()
 im=D[i];P=T[i].copy(); yy,xx=np.where(obj[i]);x0=max(0,xx.min()-20);x1=min(W,xx.max()+21);y0=max(0,yy.min()-20);y1=min(H,yy.max()+21);w=x1-x0;h=y1-y0
 out=[]
 for it in range(4):
  rr,m=render(P,x0,y0,w,h);vis=m&obj[i,y0:y1,x0:x1]&~per[i,y0:y1,x0:x1]
  obs=im[y0:y1,x0:x1][vis];rend=rr[vis];res=np.median(obs-rend) if len(obs)>20 else 0
  # tolerant shape metrics using original K? mask here Kpipeline
  mfull=m; ob=obj[i,y0:y1,x0:x1]&~per[i,y0:y1,x0:x1]; iou=(mfull&ob).sum()/max(1,(mfull|ob).sum())
  out.append((res,len(obs),iou,mfull.sum(),ob.sum()))
  P[:3,3]+=Twc[:3,:3]@np.array([0,0,res],np.float32)
 print(i,out)
PY
python /work/scratch/analysis/depth_correct_test.py"
```
<details><summary>output</summary>

```
0 [(-0.07684207, 944, 0.37475188566891626, 1701, 1762), (-0.0444808, 795, 0.28783490224475017, 1795, 1762), (-0.023394346, 702, 0.2438346648141716, 1819, 1762), (-0.013231039, 654, 0.22177009155645983, 1841, 1762)]
50 [(-0.12314224, 1183, 0.4578173374613003, 1720, 2047), (-0.07858133, 891, 0.2938654353562005, 1876, 2047), (-0.032019377, 672, 0.20537897310513448, 1897, 2047), (-0.006911993, 587, 0.17718080289767582, 1853, 2047)]
100 [(-0.20083928, 1194, 0.4669534610872116, 1697, 2054), (-0.09696841, 685, 0.20890515401036902, 1910, 2054), (-0.01123786, 416, 0.1224249558563861, 1760, 2054), (-0.000875473, 391, 0.11462914101436529, 1748, 2054)]
130 [(-0.17963028, 1155, 0.45760697305863707, 1765, 1914), (-0.06547022, 686, 0.2175015852885225, 1926, 1914), (-0.007314205, 505, 0.15596046942557135, …[+4360 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
s=np.load('/work/scratch/analysis/flip_scores.npz')['scores']
d=s[:,0,1]-s[:,1,1] # direct ncc base - flip
for a in range(0,565,10):
 b=min(565,a+10);dd=d[a:b]
 print(f'{a:3d}-{b-1:3d} directdiff {dd.mean():+.3f} med {np.median(dd):+.3f} win {np.mean(dd>0):.2f} | base ncc {s[a:b,0,1].mean():+.3f} flip {s[a:b,1,1].mean():+.3f} | ioudiff {np.mean(s[a:b,0,9]-s[a:b,1,9]):+.3f}')
print('\\nrolling 15')
c=np.convolve(d,np.ones(15),'same')/np.convolve(np.ones(len(d)),np.ones(15),'same')
for i in range(0,565,10):print(i,round(c[i],3),end='; ')
print('\\nsign changes',np.where(np.diff(np.sign(c))"'!=0)[0])
PY'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/one_large.py <<'PY'
src=open('/work/scratch/analysis/render_flips.py').read().split(\"rows=[]\")[0];exec(src)
for i in [300,400,500]:
 c=project_center(T[i]);x0=max(0,int(c[0])-80);y0=max(0,int(c[1])-60);w,h=160,120
 if x0+w>W:x0=W-w
 if y0+h>H:y0=H-h
 img=frames[i][y0:y0+h,x0:x0+w];r0,m0=render(T[i],x0,y0,w,h);r1,m1=render(T[i]@S,x0,y0,w,h)
 def rr(rgb,m):
  z=np.zeros_like(img);z[m]=np.clip(rgb[m]*255,0,255).astype(np.uint8)[:,::-1];return z
 row=np.concatenate([img,rr(r0,m0),rr(r1,m1)],1);row=cv2.resize(row,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC);cv2.putText(row,str(i),(5,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);cv2.imwrite(f'/work/scratch/analysis/large_{i}.jpg',row)
PY
python /work/scratch/analysis/one_large.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=np.asarray(m.vertices);t=cKDTree(v)
for axis in ['x','y','z']:
 S=R.from_euler(axis,np.pi).as_matrix();vv=v@S.T;d,_=t.query(vv)
 print(axis,'NN dist percentiles',np.percentile(d,[50,90,95,99,99.9,100]),'mean',d.mean())
# principal axes tests with oriented bounds? S from OBB
S,to_obb=trimesh.bounds.oriented_bounds(m);print('S',S,'det',np.linalg.det(S[:3,:3]))
for j in range(3):
 Rm=S[:3,:3].T@np.diag([1 if k"'!=j else -1 for k in range(3)])@S[:3,:3]
 vv=v@Rm.T;d,_=t.query(vv);print('"'obbaxis',j,np.percentile(d,[50,90,99,100]),d.mean())
PY"
```
<details><summary>output</summary>

```
x NN dist percentiles [0.01964698 0.02860795 0.03054713 0.03470808 0.03793309 0.04015209] mean 0.017606625687256197
y NN dist percentiles [0.00184652 0.00379234 0.00474009 0.00662634 0.0082101  0.00948819] mean 0.0021422601701386623
z NN dist percentiles [0.01972453 0.02879812 0.03078455 0.03460145 0.0387801  0.04089505] mean 0.017590094462726327
S [[ 0.00496272  0.99998235 -0.00326673  0.00533543]
 [-0.99966216  0.00487773 -0.02552984  0.00416986]
 [-0.02551345  0.00339232  0.99966872 -0.00184554]
 [ 0.          0.          0.          1.        ]] det 1.0
obbaxis 0 [0.01956865 0.02901711 0.03422733 0.03987667] 0.01758183200417291
obbaxis 1 [0.00172811 0.00354288 0.00691096 0.009827  ] 0.0020185915561616533
obbaxis 2 [0.00177423 0.00354874 0.00579384 0.00858714] 0.002024138198734427

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
s=np.load('/work/scratch/analysis/flip_scores.npz')['scores'];T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
S=Rot.from_euler('y',np.pi).as_matrix()
def yaw(R):
 v=R@np.array([0,0,1.]);return np.degrees(np.arctan2(v[0],v[2]))
for i in range(175,256):
 print(f'{i} yaw {yaw(T[i,:3,:3]):7.1f} | base ncc {s[i,0,1]:+.3f} hp {s[i,0,0]:+.3f} iou {s[i,0,9]:.3f} | flip ncc {s[i,1,1]:+.3f} hp {s[i,1,0]:+.3f} iou {s[i,1,9]:.3f} | bestN {\"B\" if s[i,0,1]>s[i,1,1] else \"F\"} bestI {\"B\" if s[i,0,9]>s[i,1,9] else \"F\"}')
PY"
```
<details><summary>output</summary>

```
175 yaw  -154.6 | base ncc +0.092 hp +0.154 iou 0.125 | flip ncc +0.158 hp +0.196 iou 0.126 | bestN F bestI F
176 yaw  -158.4 | base ncc +0.043 hp +0.083 iou 0.261 | flip ncc -0.025 hp +0.069 iou 0.266 | bestN B bestI F
177 yaw  -154.6 | base ncc +0.112 hp +0.117 iou 0.263 | flip ncc +0.068 hp +0.050 iou 0.266 | bestN B bestI F
178 yaw  -155.4 | base ncc -0.023 hp +0.087 iou 0.260 | flip ncc +0.013 hp +0.087 iou 0.257 | bestN F bestI B
179 yaw  -156.1 | base ncc +0.036 hp +0.112 iou 0.316 | flip ncc +0.058 hp +0.110 iou 0.309 | bestN F bestI B
180 yaw  -139.0 | base ncc +0.056 hp +0.028 iou 0.352 | flip ncc +0.088 hp +0.059 iou 0.344 | bestN F bestI B
181 yaw  -150.8 | base ncc -0.053 hp +0.031 iou 0.302 | flip ncc -0.043 hp +0.019 iou 0.300 | bestN F bestI B
182 yaw  -156.3 | base ncc +0. …[+8110 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/score_depthcorr.py <<'PY'
# reuse render_flips setup with original K and score direct
src=open('/work/scratch/analysis/render_flips.py').read().split(\"rows=[]\")[0];exec(src)
# render depth using Kpipeline
Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
def render_depth(Twm):
 Tcm=(Tcw@Twm).astype(np.float32);Rcm=torch.tensor(Tcm[:3,:3],device='cuda');tcm=torch.tensor(Tcm[:3,3],device='cuda');vc=V@Rcm.T+tcm;x,y,z=vc[:,0],vc[:,1],vc[:,2];fx,fy=Kp[0,0],Kp[1,1];cx,cy=Kp[0,2],Kp[1,2]
 clip=torch.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,z,z],1)[None]
 rast,_=dr.rasterize(ctx,clip,F,resolution=[H,W]);dz,_=dr.interpolate(z.contiguous()[:,None],rast,F);return dz[0,:,0].cpu().numpy(),(rast[0,...,3]>0).cpu().numpy()
def metrics2(img,rgb,mask,om,pm):
 obs=om&~pm;g1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;g2=cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
 h1=g1-cv2.GaussianBlur(g1,(0,0),1.2);h2=g2-cv2.GaussianBlur(g2,(0,0),1.2);best=None
 for dx in range(-3,4):
  for dy in range(-3,4):
   M=np.float32([[1,0,dx],[0,1,dy]]);m2=cv2.warpAffine(mask.astype(np.uint8),M,(mask.shape[1],mask.shape[0]))>0;h2s=cv2.warpAffine(h2,M,(h2.shape[1],h2.shape[0]));vis=m2&obs
   if vis.sum()<30:continue
   a=g1[vis];b=cv2.warpAffine(g2,M,(g2.shape[1],g2.shape[0]))[vis];ncc=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-6 and b.std()>1e-6 else 0
   aa=h1[vis];bb=h2s …[+1802 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/score_depthcorr.py", line 32, in <module>
    rd,md=render_depth(T[i]);vis=md&obj[i]&~per[i];res=float(np.median(im[vis]-rd[vis])) if vis.sum()>20 else 0
                                                                               ~~^^^^^
IndexError: boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 1536

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/yaw_grid.py <<'PY'
src=open('/work/scratch/analysis/render_flips.py').read().split(\"rows=[]\")[0];exec(src)
Kp=np.array(json.load(open('/work/inputs/camera.json'))['K_pipeline'])
def render_depth(Twm):
 Tcm=(Tcw@Twm).astype(np.float32);Rcm=torch.tensor(Tcm[:3,:3],device='cuda');tcm=torch.tensor(Tcm[:3,3],device='cuda');vc=V@Rcm.T+tcm;x,y,z=vc[:,0],vc[:,1],vc[:,2];fx,fy=Kp[0,0],Kp[1,1];cx,cy=Kp[0,2],Kp[1,2]
 clip=torch.stack([2*fx/W*x+(2*cx/W-1)*z,2*fy/H*y+(2*cy/H-1)*z,z,z],1)[None];rast,_=dr.rasterize(ctx,clip,F,resolution=[H,W]);dz,_=dr.interpolate(z.contiguous()[:,None],rast,F);return dz[0,:,0].cpu().numpy(),(rast[0,...,3]>0).cpu().numpy()
def score_yaw(img,rgb,mask,om,pm,use_edges=False):
 obs=om&~pm;g1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;g2=cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
 best=(-9,None)
 for dx in range(-3,4):
  for dy in range(-3,4):
   M=np.float32([[1,0,dx],[0,1,dy]]);m2=cv2.warpAffine(mask.astype(np.uint8),M,(mask.shape[1],mask.shape[0]))>0;vis=m2&obs
   if vis.sum()<60:continue
   a=g1[vis];b=cv2.warpAffine(g2,M,(g2.shape[1],g2.shape[0]))[vis]
   ncc=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-6 and b.std()>1e-6 else 0
   # local high-pass ncc and gradient orientation
   a1=cv2.GaussianBlur(a,(0,0),1)-a;b1=cv2.GaussianBlur(b,(0,0),1)-b
   hp=float(np.corrcoef(a1,b1)[0,1]) if a1.std()>1e-6 and b1.std()>1e-6 else 0
   iou=(m2&ob).sum()/ …[+1177 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/yaw_grid.py", line 26, in <module>
    ctx=dr.RasterizeCudaContext();im=D[i];yy,xx=np.where(obj[i]);rd,md=render_depth(T[i]);vis=md&obj[i]&~per[i];res=np.median(im[vis]-rd[vis]);P=T[i].copy();P[:3,3]+=Twc[:3,:3]@np.array([0,0,res],np.float32)
                                                                                                                                      ~~^^^^^
IndexError: boolean index did not match indexed array along dimension 1; dimension is 1 but corresponding boolean dimension is 1536

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/dz\\[0,:,0\\]/dz[0,...,0]/' /work/scratch/analysis/score_depthcorr.py
python /work/scratch/analysis/score_depthcorr.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/score_depthcorr.py", line 41, in <module>
    rgb,m=render(Q,cx0,cy0,w2,h2);vals.append((name,metrics2(img,rgb,m,om,pm)))
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/score_depthcorr.py", line 18, in metrics2
    obsd=cv2.dilate(obs.astype(np.uint8),np.ones((7,7),np.uint8))>0;prec=(m2&obsd).sum()/max(1,m2.sum());rec=(obs&(cv2.dilate(m2.astype(np.uint8),np.ones((7,7),np.uint8))>0)).sum()/max(1,obs.sum());iou=(m2&ob).sum()/max(1,(m2|ob).sum())
                                                                                                                                                                                                              ^^
NameError: name …[+42 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/dz\\[0,:,0\\]/dz[0,...,0]/' /work/scratch/analysis/yaw_grid.py
python /work/scratch/analysis/yaw_grid.py"
```
<details><summary>output</summary>

```

frame 300 depthres -0.6590905
Traceback (most recent call last):
  File "/work/scratch/analysis/yaw_grid.py", line 34, in <module>
    Rz=Rot.from_euler('y',np.deg2rad(deg)).as_matrix();Q=P.copy();Q[:3,:3]=Rz@P[:3,:3];rgb,m=render(Q,x0,y0,w,h);sc=score_yaw(img,rgb,m,om,pm);arr.append((deg,sc))
                                                                                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/yaw_grid.py", line 18, in score_yaw
    iou=(m2&ob).sum()/max(1,(m2|ob).sum())
            ^^
NameError: name 'ob' is not defined. Did you mean: 'om'?

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/obs=om\\&~pm;g1=/obs=om\\&~pm;ob=obs;g1=/' /work/scratch/analysis/score_depthcorr.py
python /work/scratch/analysis/score_depthcorr.py"
```
<details><summary>output</summary>

```
0 res -0.064 B:sc=0.473,ncc=0.251,hp=0.068,iou=0.792,vis=1577,sh=-3,-2 | F:sc=0.267,ncc=0.057,hp=0.065,iou=0.750,vis=1519,sh=-3,-3
10 res -0.003 B:sc=0.447,ncc=0.205,hp=0.103,iou=0.823,vis=1737,sh=0,1 | F:sc=0.331,ncc=0.062,hp=0.165,iou=0.846,vis=1756,sh=-2,2
20 res -0.07 B:sc=0.472,ncc=0.252,hp=0.045,iou=0.815,vis=1744,sh=0,-3 | F:sc=0.296,ncc=0.076,hp=0.046,iou=0.815,vis=1728,sh=-3,-3
30 res -0.079 B:sc=0.441,ncc=0.207,hp=0.089,iou=0.812,vis=1759,sh=1,-3 | F:sc=0.302,ncc=0.041,hp=0.113,iou=0.885,vis=1831,sh=-3,-2
40 res -0.061 B:sc=0.439,ncc=0.203,hp=0.042,iou=0.882,vis=1804,sh=-3,-1 | F:sc=0.331,ncc=0.084,hp=0.110,iou=0.833,vis=1732,sh=-3,-2
50 res -0.109 B:sc=0.398,ncc=0.160,hp=0.040,iou=0.899,vis=1862,sh=-3,-3 | F:sc=0.160,ncc=-0.062,hp=-0.007,iou=0.899,vis=1846,sh=-3,-3
60 res -0.146 …[+6797 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ps -o pid,etime,pcpu,pmem,cmd -C python | tail -20; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
```
<details><summary>output</summary>

```
    PID     ELAPSED %CPU %MEM CMD
     14       17:02  0.2  0.0 python /work/.net/allowlist_proxy.py inner --sock /work/.net/proxy.sock
   7371       04:40 2452  4.6 python /work/scratch/analysis/yaw_grid.py
7371, 680 MiB

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "sed -i 's/obs=om\\&~pm;g1=/obs=om\\&~pm;ob=obs;g1=/' /work/scratch/analysis/yaw_grid.py
python /work/scratch/analysis/yaw_grid.py"
```
<details><summary>output</summary>

```

frame 300 depthres -0.6590905
top [(10, (0.5383834716162561, (-1, -1), 0.21873960502496162, 1.0, 0.34821933295647256, 616)), (-10, (0.4919885416999349, (-3, -1), 0.17371156907513674, 0.9999999999999978, 0.3413848631239936, 636)), (-80, (0.48259061453045343, (3, -3), 0.1666856785158285, 0.9999999999999989, 0.32952468007312613, 721)), (-90, (0.47997404475715355, (1, -2), 0.16936253396578665, 1.0, 0.30305755395683454, 674)), (-110, (0.4752591670531568, (2, -2), 0.16293489338867695, 1.0, 0.31162136832239923, 665)), (-30, (0.4727949201619278, (-1, 0), 0.16233204648786823, 0.9999999999999993, 0.30231436837029896, 627)), (-40, (0.4705645364152793, (0, -1), 0.15843687684081126, 0.9999999999999998, 0.31063829787234043, 657)), (20, (0.46780276602808507, (3, -2), 0.14020964281891757, 0.9999999999999 …[+3811 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/yaw_grid_fast.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,torch,trimesh,json,cv2,h5py
import nvdiffrast.torch as dr
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
p=Path('/work/inputs');m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry();verts=np.asarray(m.vertices);faces=np.asarray(m.faces)
V=torch.tensor(verts,dtype=torch.float32,device='cuda');Tri=torch.tensor(faces,dtype=torch.int32,device='cuda');uvn=np.asarray(m.visual.uv).copy();uvn[:,1]=1-uvn[:,1];uv=torch.tensor(uvn,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(m.visual.material.baseColorTexture).astype(np.float32)/255,device='cuda')
cam=json.load(open(p/'camera.json'));K=np.array(cam['K']);Kp=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);T=np.load(p/'reconstruction.npz')['T_world_mesh'].astype(np.float32);H,W=1152,1536
cap=cv2.VideoCapture(str(p/'video.mp4'));frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
with h5py.File(p/'masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
ctx=dr.RasterizeCudaContext()
def render_batch(poses,x0,y0,w,h,Kuse):
 B=len(poses);Tcm=np.einsum('ij,bjk->bik',Tcw,poses).astype(np.float32);R=torch.tensor(Tcm[:,:3,:3],device='cuda');tt=torch.tensor(Tcm[:,:3,3],device='cuda')
 vc=torch.einsum('bnj,bkj->bn …[+2697 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/yaw_grid_fast.py", line 30, in <module>
    res=depth_res(T[i],im);P=T[i].copy();P[:3,3]+=Twc[:3,:3]@np.array([0,0,res],np.float32)
        ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/yaw_grid_fast.py", line 23, in depth_res
    rd,md,_=render_batch(P[None],0,0,W,H,Kp);vis=md[0]&obj[i]&~per[i];return float(np.median(im[vis]-rd[0][vis])) if vis.sum()>20 else 0.
    ^^^^^^^
ValueError: too many values to unpack (expected 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/rd,md,_=render_batch(P\\[None\\],0,0,W,H,Kp)/rd,md,_,_=render_batch(P[None],0,0,W,H,Kp)/' /work/scratch/analysis/yaw_grid_fast.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/yaw_grid_fast.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/yaw_grid_fast.py", line 30, in <module>
    res=depth_res(T[i],im);P=T[i].copy();P[:3,3]+=Twc[:3,:3]@np.array([0,0,res],np.float32)
        ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/yaw_grid_fast.py", line 23, in depth_res
    rd,md,_,_=render_batch(P[None],0,0,W,H,Kp);vis=md[0]&obj[i]&~per[i];return float(np.median(im[vis]-rd[0][vis])) if vis.sum()>20 else 0.
                                                                                               ~~~~~~~^~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (972,) (972,3) 

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/yaw_grid_fast.py';s=open(p).read()
s=s.replace(\"rgb=dr.texture(tex[None].expand(B,-1,-1,-1),uvi,filter_mode='linear',boundary_mode='wrap');mask=rast[...,3]>0\\n return rgb.clamp(0,1).cpu().numpy(),mask.cpu().numpy(),z,torch.einsum('bnj,bkj->bnk',V[None].expand(B,-1,-1),R)+tt[:,None,:]\", \"rgb=dr.texture(tex[None].expand(B,-1,-1,-1),uvi,filter_mode='linear',boundary_mode='wrap');mask=rast[...,3]>0\\ndmap,_=dr.interpolate(z[...,None],rast,Tri)\\n return rgb.clamp(0,1).cpu().numpy(),mask.cpu().numpy(),dmap[...,0].cpu().numpy(),vc\")
open(p,'w').write(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/yaw_grid_fast.py"
```
<details><summary>output</summary>

```
  File "/work/scratch/analysis/yaw_grid_fast.py", line 22
    return rgb.clamp(0,1).cpu().numpy(),mask.cpu().numpy(),dmap[...,0].cpu().numpy(),vc
IndentationError: unexpected indent

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/analysis/yaw_grid_fast.py | sed -n '15,27p'"
```
<details><summary>output</summary>

```
    15	with h5py.File(p/'masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
    16	ctx=dr.RasterizeCudaContext()
    17	def render_batch(poses,x0,y0,w,h,Kuse):
    18	 B=len(poses);Tcm=np.einsum('ij,bjk->bik',Tcw,poses).astype(np.float32);R=torch.tensor(Tcm[:,:3,:3],device='cuda');tt=torch.tensor(Tcm[:,:3,3],device='cuda')
    19	 vc=torch.einsum('bnj,bkj->bnk',V[None].expand(B,-1,-1),R)+tt[:,None,:];x,y,z=vc[...,0],vc[...,1],vc[...,2];k=Kuse.copy();k[0,2]-=x0;k[1,2]-=y0;fx,fy=k[0,0],k[1,1];cx,cy=k[0,2],k[1,2]
    20	 clip=torch.stack([2*fx/w*x+(2*cx/w-1)*z,2*fy/h*y+(2*cy/h-1)*z,z,z],-1).contiguous();rast,_=dr.rasterize(ctx,clip,Tri,resolution=[h,w]);uvi,_=dr.interpolate(uv[None].expand(B,-1,-1),rast,Tri);rgb=dr.texture(tex[None].expand(B,-1,-1,-1),uvi,filter_mode='linear',boundary_ …[+654 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/yaw_grid_fast.py';s=open(p).read();s=s.replace(\"\\ndmap,_=dr.interpolate\", \"\\n dmap,_=dr.interpolate\");s=s.replace(\"rd,md,_,_=render_batch(P[None],0,0,W,H,Kp);vis=md[0]&obj[i]&~per[i];return float(np.median(im[vis]-rd[0][vis]))\",\"rgb,md,rd,_=render_batch(P[None],0,0,W,H,Kp);vis=md[0]&obj[i]&~per[i];return float(np.median(im[vis]-rd[0][vis]))\")
open(p,'w').write(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/yaw_grid_fast.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/yaw_grid_fast.py", line 31, in <module>
    res=depth_res(T[i],im);P=T[i].copy();P[:3,3]+=Twc[:3,:3]@np.array([0,0,res],np.float32)
        ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/yaw_grid_fast.py", line 24, in depth_res
    rgb,md,rd,_=render_batch(P[None],0,0,W,H,Kp);vis=md[0]&obj[i]&~per[i];return float(np.median(im[vis]-rd[0][vis])) if vis.sum()>20 else 0.
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/yaw_grid_fast.py", line 21, in render_batch
    dmap,_=dr.interpolate(z[...,None],rast,Tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, r …[+383 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/dr.interpolate(z\\[...,None\\]/dr.interpolate(z.contiguous()[...,None]/' /work/scratch/analysis/yaw_grid_fast.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/yaw_grid_fast.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analysis/yaw_grid_fast.py", line 39, in <module>
    rgbs,masks,_,_=render_batch(np.stack(poses),x0,y0,w,h,K)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/analysis/yaw_grid_fast.py", line 20, in render_batch
    clip=torch.stack([2*fx/w*x+(2*cx/w-1)*z,2*fy/h*y+(2*cy/h-1)*z,z,z],-1).contiguous();rast,_=dr.rasterize(ctx,clip,Tri,resolution=[h,w]);uvi,_=dr.interpolate(uv[None].expand(B,-1,-1),rast,Tri);rgb=dr.texture(tex[None].expand(B,-1,-1,-1),uvi,filter_mode='linear',boundary_mode='wrap');mask=rast[...,3]>0
                                                                                                                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+719 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/uv\\[None\\].expand(B,-1,-1)/uv[None].expand(B,-1,-1).contiguous()/; s/tex\\[None\\].expand(B,-1,-1,-1)/tex[None].expand(B,-1,-1,-1).contiguous()/\" /work/scratch/analysis/yaw_grid_fast.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/yaw_grid_fast.py"
```
<details><summary>output</summary>

```
150 res -0.113 BEST [(20.0, 0.21319720763559297, 0.09594865498241661, 0.00025443974527483915, 0.5859247135842881, 1432), (40.0, 0.19282476617919356, 0.0869706275490937, -0.014272472234156633, 0.547111283443195, 1411), (30.0, 0.18444988460446493, 0.06801933600569436, 0.019556186489943887, 0.5577075098814229, 1411), (-30.0, 0.1669454287200774, 0.0622119416595211, 0.003980677214187683, 0.5186915887850467, 1221), (-10.0, 0.16547421637014126, 0.05931479668211853, -0.030351614609793413, 0.5687366167023554, 1328)]
175 res -0.515 BEST [(-130.0, 0.1585805156973113, 0.06535673606507332, 0.05407050879525122, 0.3985307621671258, 868), (110.0, 0.15435053558301737, 0.050883957776291164, 0.0903848969023355, 0.4043517679057117, 892), (20.0, 0.13436322447610916, 0.04827943920167695, 0.007416930523847146, 0 …[+10341 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/yaw_grid_fast.py';s=open(p).read()
s=s.replace(\"inds=[150,175,195,200,210,225,250,275,300,325,340,360,380,400,420,440,460,480,500,520,540,560]\", \"inds=list(range(0,130,10))+list(range(130,451,5))+list(range(460,565,10))+[564]\")
s=s.replace(\"print(i,'res',round(res,3),'BEST',sorted(c,key=lambda z:z[1],reverse=True)[:5],flush=True)\", \"print(i,'res',round(res,3),'BEST',sorted(c,key=lambda z:z[1],reverse=True)[:2],flush=True)\\n allc.append((i,c))\")
s=s.replace(\"for kk,i in enumerate(inds):\", \"allc=[]\\nfor kk,i in enumerate(inds):\")
s=s.replace(\"np.savez('/work/scratch/analysis/yaw_grid_fast.npz',inds=inds,curves=np.array([c])) if False else None\", \"np.savez('/work/scratch/analysis/yaw_grid_fast.npz',inds=np.array([x[0] for x in allc]),curves=np.array([x[1] for x in allc]),degs=degs)\")
open('/work/scratch/analysis/yaw_grid_all.py','w').write(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/yaw_grid_all.py > /work/scratch/analysis/yaw_grid_all.log 2>&1
wc -l /work/scratch/analysis/yaw_grid_all.log; tail -20 /work/scratch/analysis/yaw_grid_all.log"
```
<details><summary>output</summary>

```
90 /work/scratch/analysis/yaw_grid_all.log
415 res -0.261 BEST [(60.0, 0.29801380371147346, 0.1625356434549768, 0.09566816385367437, 0.5578055964653903, 1515), (70.0, 0.2906154410770955, 0.1678893989729746, 0.049306579870892056, 0.5519969856819894, 1465)]
420 res -0.173 BEST [(-40.0, 0.2676637518499186, 0.11179188818451388, 0.11162548485572382, 0.6398274622573688, 1780), (-50.0, 0.26044219420676945, 0.11299169954011369, 0.07989597722049135, 0.6373825018076645, 1763)]
425 res -0.111 BEST [(60.0, 0.412094443737006, 0.21854838865006587, 0.23600827200232063, 0.6727199354317999, 1667), (20.0, 0.3886820209543256, 0.17434498084597622, 0.2792659686525753, 0.7226027397260274, 1899)]
430 res -0.262 BEST [(70.0, 0.28766075216885023, 0.18426893078185794, -0.0020767761671365895, 0.5195550771438823, 144 …[+3547 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/analysis/yaw_grid_fast.npz');inds=z['inds'];cur=z['curves'] # N,36,6
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
def yaw_of(R):
 v=R@np.array([0,0,1.]);return np.degrees(np.arctan2(v[0],v[2]))
for k,i in enumerate(inds):
 c=cur[k];base=yaw_of(T[i,:3,:3]);
 # top combined, top direct, top hp, top iou
 tops={}
 tops['sc']=c[np.argmax(c[:,1])]
 tops['ncc']=c[np.argmax(c[:,2])]
 tops['hp']=c[np.argmax(c[:,3])]
 tops['iou']=c[np.argmax(c[:,4])]
 # top entries where direct ncc positive and local
 print(f'{i:3d} base {base:7.1f} | sc {base+tops[\"sc\"][0]:7.1f} ({tops[\"sc\"][1]:.2f}) | ncc {base+tops[\"ncc\"][0]:7.1f} ({tops[\"ncc\"][2]:.2f}) | hp {base+tops[\"hp\"][0]:7.1f} ({tops[\"hp\"][3]:.2f}) | iou {base+tops[\"iou\"][0]:7.1f} ({tops[\"iou\"][4]:.2f})')
PY"
```
<details><summary>output</summary>

```
  0 base  -168.1 | sc  -158.1 (0.35) | ncc  -158.1 (0.22) | hp   -58.1 (0.04) | iou  -158.1 (0.74)
 10 base  -174.9 | sc  -164.9 (0.40) | ncc  -164.9 (0.23) | hp  -164.9 (0.07) | iou  -184.9 (0.78)
 20 base  -174.9 | sc  -174.9 (0.35) | ncc  -174.9 (0.20) | hp  -274.9 (0.04) | iou  -164.9 (0.87)
 30 base  -174.9 | sc  -134.9 (0.33) | ncc  -134.9 (0.18) | hp   -64.9 (0.03) | iou  -164.9 (0.86)
 40 base  -174.9 | sc  -164.9 (0.39) | ncc  -164.9 (0.22) | hp   -54.9 (0.04) | iou  -164.9 (0.87)
 50 base  -174.9 | sc  -144.9 (0.24) | ncc  -114.9 (0.09) | hp   -34.9 (0.02) | iou  -164.9 (0.80)
 60 base  -174.9 | sc  -154.9 (0.19) | ncc  -114.9 (0.05) | hp   -24.9 (0.04) | iou  -164.9 (0.71)
 70 base  -174.9 | sc  -174.9 (0.35) | ncc  -174.9 (0.18) | hp   -64.9 (0.05) | iou  -174.9 (0.88)
 80 base …[+8113 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/scratch/analysis/yaw_grid_fast.npz');inds=z['inds'];cur=z['curves']
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
def yaw_of(R):
 v=R@np.array([0,0,1.]);return np.degrees(np.arctan2(v[0],v[2]))
degs=cur[0,:,0]
for name,mask in [('initial',(inds>=0)&(inds<=125)),('390-415',(inds>=390)&(inds<=415)),('final',(inds>=470)),('380-440',(inds>=380)&(inds<=440)),('300-430',(inds>=300)&(inds<=430))]:
 C=cur[mask];ii=inds[mask];w=C[:,:,5]/np.maximum(C[:,:,5],1) # vis
 # average direct, hp, iou weighted equally per frame (normalize ncc by reliability)
 mean_ncc=np.nanmean(C[:,:,2],axis=0);mean_hp=np.nanmean(C[:,:,3],axis=0);mean_iou=np.nanmean(C[:,:,4],axis=0);mean_sc=np.nanmean(C[:,:,1],axis=0)
 # absolute yaw average (base yaw changes); show relative mode too
 bases=np.array([yaw_of(T[i,:3,:3]) for i in ii]);
 abs_yaw=np.array([((bases[:,None]+degs[None,:]+180)%360)-180])
 # circular mean weighted by ncc pos? 
 print('\\n',name,'n',len(ii))
 for lab,a in [('ncc',mean_ncc),('hp',mean_hp),('iou',mean_iou),('sc',mean_sc)]:
  top=np.argsort(a)[-6:][::-1];print(lab,[(float(degs[j]),round(float(a[j]),3)) for j in top])
 # Select ncc-weighted circular yaw
 w=np.clip(mean_ncc,0,None);rad=np.deg2rad(abs_yaw);cm=np.angle(np.mean(np.exp(1j*rad)*w,axis=0));print('circ ncc peak deg',np.degrees(cm))
PY"
```
<details><summary>output</summary>

```

 initial n 13
ncc [(10.0, 0.134), (30.0, 0.13), (20.0, 0.122), (0.0, 0.118), (40.0, 0.114), (50.0, 0.113)]
hp [(110.0, 0.012), (-70.0, 0.006), (-100.0, 0.003), (120.0, 0.001), (90.0, 0.001), (100.0, 0.0)]
iou [(10.0, 0.793), (0.0, 0.785), (20.0, 0.776), (-10.0, 0.762), (30.0, 0.754), (-20.0, 0.734)]
sc [(10.0, 0.286), (30.0, 0.277), (0.0, 0.27), (20.0, 0.268), (40.0, 0.256), (-10.0, 0.254)]
circ ncc peak deg [[   0.            0.            0.            0.            0.
     0.            0.            0.            0.          101.90180047
     0.          121.90180047  131.90180047  141.90180047  151.90180047
   161.90180047  171.90180047 -178.09819953 -168.09819953 -158.09819953
  -148.09819953 -138.09819953 -128.09819953 -118.09819953 -108.09819953
   -98.09819953  -88.09819953  -78. …[+36183 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/scratch/analysis/yaw_grid_fast.npz');inds=z['inds'];cur=z['curves'];degs=cur[0,:,0]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
def yaw_of(R):
 v=R@np.array([0,0,1.]);return np.degrees(np.arctan2(v[0],v[2]))
windows=[('0-125',0,125),('130-190',130,190),('230-300',230,300),('300-340',300,340),('340-380',340,380),('380-415',380,415),('415-440',415,440),('450-564',450,564)]
for name,a,b in windows:
 m=(inds>=a)&(inds<=b);C=cur[m];ii=inds[m];bases=np.array([yaw_of(T[i,:3,:3]) for i in ii]);
 means={lab:C[:,:,k].mean(0) for k,lab in [(2,'ncc'),(3,'hp'),(4,'iou'),(1,'sc')]}
 print('\\n',name,'n',len(ii),'base range',bases.min(),bases.max())
 for lab,v in means.items():
  top=np.argsort(v)[-4:][::-1];print(lab,[(float(degs[j]),round(float(v[j]),3)) for j in top])
 # combined mean score absolute yaw relative to current baseline angle: unwrap each
 combined=0.55*means['ncc']+0.2*means['hp']+0.25*means['iou']
 top=np.argsort(combined)[-6:][::-1];print('comb',[(float(degs[j]),round(float(combined[j]),3)) for j in top])
PY"
```
<details><summary>output</summary>

```

 0-125 n 13 base range -174.92426104931886 -168.09819952886312
ncc [(10.0, 0.134), (30.0, 0.13), (20.0, 0.122), (0.0, 0.118)]
hp [(110.0, 0.012), (-70.0, 0.006), (-100.0, 0.003), (120.0, 0.001)]
iou [(10.0, 0.793), (0.0, 0.785), (20.0, 0.776), (-10.0, 0.762)]
sc [(10.0, 0.286), (30.0, 0.277), (0.0, 0.27), (20.0, 0.268)]
comb [(10.0, 0.267), (0.0, 0.257), (30.0, 0.257), (20.0, 0.253), (-10.0, 0.245), (40.0, 0.241)]

 130-190 n 13 base range -169.37946112793526 -138.96922388882479
ncc [(40.0, 0.027), (30.0, 0.022), (10.0, 0.021), (20.0, 0.019)]
hp [(140.0, 0.05), (100.0, 0.039), (110.0, 0.036), (-140.0, 0.035)]
iou [(10.0, 0.391), (0.0, 0.39), (20.0, 0.384), (-10.0, 0.38)]
sc [(20.0, 0.104), (10.0, 0.103), (40.0, 0.103), (30.0, 0.1)]
comb [(20.0, 0.113), (10.0, 0.113), (40.0, 0.108), (30.0, …[+2597 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/static_composite.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
exec(open('/work/scratch/analysis/yaw_grid_fast.py').read().split(\"inds=\")[0])
# read video into crops for windows
def make_comp(start,end,bbox):
 x0,y0,x1,y1=bbox;cap=cv2.VideoCapture(str(p/'video.mp4'));ims=[];oms=[];pms=[]
 for i in range(end+1):
  ok,im=cap.read()
  if not ok:break
  if i>=start:
   ims.append(im[y0:y1,x0:x1]);oms.append(obj[i,y0:y1,x0:x1]);pms.append(per[i,y0:y1,x0:x1])
 cap.release();return np.median(np.stack(ims),0).astype(np.uint8),np.mean(np.stack(oms),0)>=0.5,np.mean(np.stack(pms),0)>=0.5
for name,start,end,bbox,idx in [('initial',0,125,(740,700,940,850),100),('final',470,564,(480,540,730,710),520)]:
 img,om,pm=make_comp(start,end,bbox);x0,y0,x1,y1=bbox;w=x1-x0;h=y1-y0
 ctx=dr.RasterizeCudaContext();P=T[idx].copy()
 # depth correction from direct depth residual using median over frames? use representative frame depth residual
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
  ds=f['aligned']['front_stereo_camera_left'];dep=cv2.imdecode(ds[idx],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000
 rgb,md,rd,_=render_batch(P[None],0,0,W,H,Kp);vis=md[0]&obj[idx]&~per[idx];res=float(np.median(dep[vis]-rd[0][vis]));P[:3,3]+=Twc[:3,:3]@np.array([0,0,res],np.float32)
 print('\\n',name,'res',res,'baseyaw',end)
 g1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.f …[+1210 chars]
```
<details><summary>output</summary>

```

 initial res -0.18949925899505615 baseyaw 125
top [(70.0, 0.20414777006781476, 0.07829301029965144, 0.021951922721948052, 0.4858575008950949, 1357), (100.0, 0.18249753777232172, 0.054810292158722754, 0.05465244428421103, 0.46702702702702703, 1296), (60.0, 0.17283157704917143, 0.0411434814031863, 0.030730940657671308, 0.5021676300578035, 1390), (90.0, 0.1695880359770379, 0.041530434270139395, 0.05286853824100648, 0.4699355762347888, 1313), (110.0, 0.16344852521149694, 0.035359351280720375, 0.05677761974522363, 0.4669345999269273, 1278), (85.0, 0.16260120166714154, 0.03854173270268026, 0.03027951951036337, 0.47201426024955434, 1324), (75.0, 0.15772574054782917, 0.03385111517568579, 0.01981955543214547, 0.47964285714285715, 1343), (30.0, 0.1477425715805762, 0.009454176210256132, -0.009882907 …[+1267 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/static_opt.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
# setup render batch from yaw_grid_fast without execution loop
src=open('/work/scratch/analysis/yaw_grid_fast.py').read();exec(src.split(\"inds=\")[0])
def align_and_score(P,yawdeg,img,om,pm,bbox):
 x0,y0,x1,y1=bbox;w=x1-x0;h=y1-y0
 Q=P.copy();Q[:3,:3]=Rot.from_euler('y',np.deg2rad(yawdeg)).as_matrix()@P[:3,:3]
 # iterate depth-area + centroid
 for it in range(3):
  rgbs,masks,_,_=render_batch(Q[None],x0,y0,w,h,K);m=masks[0]
  area=m.sum();tarea=om.sum();camera_center=Tcw[:3,:3]@Q[:3,3]+Tcw[:3,3];zc=camera_center[2]
  if area<30:return (-9,)*8
  yy,xx=np.where(m);cy,cx=yy.mean(),xx.mean();yy2,xx2=np.where(om);cy2,cx2=yy2.mean(),xx2.mean()
  # depth via area ratio; damp
  scale=np.sqrt(area/max(1,tarea));zc_new=zc*scale
  # 2d centroid offsets in crop
  dx=(cx2-cx)*zc/K[0,0];dy=(cy2-cy)*zc/K[1,1]
  camera_center_new=camera_center.copy();camera_center_new[2]=zc_new;camera_center_new[0]+=dx;camera_center_new[1]+=dy
  Q[:3,3]=Twc[:3,:3]@camera_center_new+Twc[:3,3]
 rgbs,masks,_,_=render_batch(Q[None],x0,y0,w,h,K);rgb=rgbs[0];m=masks[0];vis=m&om&~pm
 if vis.sum()<100:return (-9,)*8
 g1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;g2=cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
 gh1=g1-cv2.GaussianBlur(g1,(0,0),1.2);gh2=g2-cv2.GaussianB …[+1823 chars]
```
<details><summary>output</summary>

```

 initial
top combined [(5.0, (0.5122543504204302, 0.20307805031629036, -0.03254987202832011, 0.9019607843137255, 0.996076508092202, 0.9995085995085995, 1932, 2039)), (-5.0, (0.493579822655506, 0.17326698095507778, 0.025253347078920303, 0.900749063670412, 0.9896296296296296, 1.0, 1924, 2025)), (-15.0, (0.48222130636626415, 0.18974606437626432, -0.044501369142160206, 0.8610729023383769, 0.9792490118577075, 0.9945945945945946, 1878, 2024)), (15.0, (0.46935599776660863, 0.163465960486668, -0.014762776563259607, 0.8824074074074074, 0.9871984244214672, 0.9965601965601966, 1906, 2031)), (25.0, (0.4646310466845698, 0.17294927941919727, -0.017954800036773712, 0.8436363636363636, 0.9693221177634834, 0.9911547911547911, 1856, 2021)), (20.0, (0.45758077698019406, 0.16440035967674307, -0.0314795163971 …[+5297 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/dyn_align_test.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
src=open('/work/scratch/analysis/yaw_grid_fast.py').read();exec(src.split(\"inds=\")[0])
def fit_frame(i,P,iters=4):
 yy,xx=np.where(obj[i]);
 if len(xx)<20:return P,None
 x0=max(0,xx.min()-25);x1=min(W,xx.max()+26);y0=max(0,yy.min()-25);y1=min(H,yy.max()+26);w=x1-x0;h=y1-y0
 obs=obj[i,y0:y1,x0:x1]&~per[i,y0:y1,x0:x1]
 vals=[]
 for it in range(iters+1):
  rgbs,masks,_,_=render_batch(P[None],x0,y0,w,h,K);m=masks[0];mv=m&~per[i,y0:y1,x0:x1]
  inter=(mv&obs).sum();iou=inter/max(1,(mv|obs).sum());prec=inter/max(1,mv.sum());rec=inter/max(1,obs.sum());vals.append((mv.sum(),obs.sum(),iou,prec,rec))
  if it==iters or mv.sum()<20:break
  cc=Tcw[:3,:3]@P[:3,3]+Tcw[:3,3];zc=cc[2];scale=np.sqrt(mv.sum()/max(1,obs.sum()));yy1,xx1=np.where(mv);yy2,xx2=np.where(obs);dx=(xx2.mean()-xx1.mean())*zc/K[0,0];dy=(yy2.mean()-yy1.mean())*zc/K[1,1];cn=cc.copy();cn[2]=zc*scale;cn[0]+=dx;cn[1]+=dy;P[:3,3]=Twc[:3,:3]@cn+Twc[:3,3]
 return P,vals
ctx=dr.RasterizeCudaContext()
for i in [0,100,130,140,150,160,170,180,190,195,200,210,220,230,240,250,260,280,300,320,340,360,380,400,420,440,450,470,500,560]:
 P,v=fit_frame(i,T[i].copy());print(i,v)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/dyn_align_test.py"
```
<details><summary>output</summary>

```
0 [(1512, 1762, 0.7793478260869565, 0.9484126984126984, 0.8138479001135074), (1513, 1762, 0.3714405360134003, 0.5862524785194977, 0.503405221339387), (1514, 1762, 0.37416107382550334, 0.5891677675033025, 0.5062429057888763), (1551, 1762, 0.38214434710054235, 0.5905867182462927, 0.5198637911464246), (1639, 1762, 0.45528455284552843, 0.649176327028676, 0.6038592508513053)]
100 [(1732, 2054, 0.7568445475638051, 0.9416859122401847, 0.7940603700097371), (2120, 2054, 0.2894655545257955, 0.4419811320754717, 0.4561830574488802), (2001, 2054, 0.7926613616268788, 0.8960519740129935, 0.8729308666017527), (2067, 2054, 0.8122251539138082, 0.8935655539429125, 0.8992210321324245), (2037, 2054, 0.9188555347091932, 0.9617083946980854, 0.9537487828627069)]
130 [(1668, 1914, 0.8072653884964682, 0.95923261390 …[+9529 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
p='/work/inputs';cap=cv2.VideoCapture(p+'/video.mp4');frames=[]
while True:
 ok,im=cap.read();
 if not ok:break
 frames.append(im)
with h5py.File(p+'/masks.h5','r') as f:o=f['object'][:]
idx=list(range(330,451,5));tw,th=240,180;cols=9;rows=(len(idx)+cols-1)//cols;sheet=np.zeros((rows*th,cols*tw,3),np.uint8)
for k,i in enumerate(idx):
 yy,xx=np.where(o[i]);
 if len(xx):cx,cy=int(np.median(xx)),int(np.median(yy))
 else:cx,cy=768,576
 x0=max(0,min(cx-90,1536-180));y0=max(0,min(cy-60,1152-120));crop=frames[i][y0:y0+120,x0:x0+180];crop=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_CUBIC);cv2.rectangle(crop,(0,0),(50,24),(0,0,0),-1);cv2.putText(crop,str(i),(3,19),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);sheet[(k//cols)*th:(k//cols+1)*th,(k%cols)*tw:(k%cols+1)*tw]=crop
cv2.imwrite('/work/scratch/analysis/carry_330_450_big.jpg',sheet)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/depth_sequence.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
src=open('/work/scratch/analysis/yaw_grid_fast.py').read();exec(src.split(\"inds=\")[0])
# load all depths lazily frame
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: ds=f['aligned']['front_stereo_camera_left']; deps=[cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(len(T))]
res=np.zeros(len(T),np.float32);cnt=np.zeros(len(T),np.int32);area=np.zeros(len(T),np.int32);overlap=np.zeros(len(T),np.float32)
ctx=dr.RasterizeCudaContext()
for i in range(len(T)):
 yy,xx=np.where(obj[i]);
 if len(xx)<20:continue
 x0=max(0,xx.min()-20);x1=min(W,xx.max()+21);y0=max(0,yy.min()-20);y1=min(H,yy.max()+21);w=x1-x0;h=y1-y0
 rgb,m,rd,_=render_batch(T[i][None],x0,y0,w,h,Kp);vis=m[0]&obj[i,y0:y1,x0:x1]&~per[i,y0:y1,x0:x1]
 if vis.sum()>20:res[i]=np.median(deps[i][y0:y1,x0:x1][vis]-rd[0][vis]);cnt[i]=vis.sum()
 area[i]=obj[i].sum();overlap[i]=(obj[i]&per[i]).sum()/max(1,area[i])
np.savez('/work/scratch/analysis/depth_sequence.npz',res=res,cnt=cnt,area=area,overlap=overlap)
for a in range(0,565,10):
 b=min(565,a+10);print(f'{a:3d}-{b-1:3d} res med {np.median(res[a:b]):+.3f} mean {res[a:b].mean():+.3f} range {res[a:b].min():+.3f},{res[a:b].max():+.3f} cnt {cnt[a:b].mean():.0f} ov {overlap[a:b].mean():.2f}')
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 p …[+47 chars]
```
<details><summary>output</summary>

```
  0-  9 res med +0.002 mean -0.011 range -0.064,+0.019 cnt 1172 ov 0.00
 10- 19 res med -0.037 mean -0.034 range -0.056,-0.003 cnt 1189 ov 0.00
 20- 29 res med -0.063 mean -0.056 range -0.090,-0.024 cnt 1180 ov 0.00
 30- 39 res med -0.072 mean -0.071 range -0.095,-0.044 cnt 1183 ov 0.00
 40- 49 res med -0.037 mean -0.040 range -0.074,+0.003 cnt 1171 ov 0.00
 50- 59 res med -0.140 mean -0.135 range -0.171,-0.092 cnt 1165 ov 0.00
 60- 69 res med -0.107 mean -0.114 range -0.158,-0.086 cnt 1182 ov 0.00
 70- 79 res med -0.064 mean -0.059 range -0.070,-0.034 cnt 1180 ov 0.00
 80- 89 res med -0.057 mean -0.066 range -0.122,-0.038 cnt 1186 ov 0.00
 90- 99 res med -0.199 mean -0.183 range -0.217,-0.141 cnt 1177 ov 0.00
100-109 res med -0.149 mean -0.140 range -0.189,-0.060 cnt 1188 ov 0.00
110-119  …[+3271 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc)
for name,idx in [('initial',100),('final',520)]:
 b=np.load(f'/work/scratch/analysis/staticbest_{name}.npz');P=b['P'];print(name,'deg',b['deg'],'val',b['val']);print('base t',T[idx,:3,3],'best t',P[:3,3],'delta world',P[:3,3]-T[idx,:3,3]);print('camera z', (Tcw@T[idx])[2,3], (Tcw@P)[2,3]);print('R delta',b['deg'])
PY"
```
<details><summary>output</summary>

```
initial deg 5.0 val [ 5.12254350e-01  2.03078050e-01 -3.25498720e-02  9.01960784e-01
  9.96076508e-01  9.99508600e-01  1.93200000e+03  2.03900000e+03]
base t [0.51082897 1.12331891 3.84983397] best t [0.47464824 1.063842   3.5794787 ] delta world [-0.03618073 -0.05947685 -0.27035522]
camera z 3.7150465524180314 3.4425702308725192
R delta 5.0
final deg -15.0 val [4.73359138e-01 1.22316946e-01 1.20582641e-01 9.34073327e-01
 9.94777159e-01 1.00000000e+00 2.77200000e+03 2.87200000e+03]
base t [-0.34732121  0.30814534  2.96504378] best t [-0.31678516  0.28745508  2.755637  ] delta world [ 0.03053606 -0.02069026 -0.20940685]
camera z 2.785562371794752 2.5770964953812294
R delta -15.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/depth_iou.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
src=open('/work/scratch/analysis/yaw_grid_fast.py').read();exec(src.split(\"inds=\")[0])
ctx=dr.RasterizeCudaContext()
offs=np.arange(-0.60,0.301,0.02)
for i in [0,50,100,130,140,150,160,170,180,190,200,220,240,250,260,280,300,320,340,360,380,400,420,440,450,470,500,520,560]:
 yy,xx=np.where(obj[i]);x0=max(0,xx.min()-30);x1=min(W,xx.max()+31);y0=max(0,yy.min()-30);y1=min(H,yy.max()+31);w=x1-x0;h=y1-y0
 obs=obj[i,y0:y1,x0:x1]&~per[i,y0:y1,x0:x1];poses=[]
 for off in offs:
  P=T[i].copy();cc=Tcw[:3,:3]@P[:3,3]+Tcw[:3,3];cc[2]+=off;P[:3,3]=Twc[:3,:3]@cc+Twc[:3,3];poses.append(P)
 _,ms,_,_=render_batch(np.stack(poses),x0,y0,w,h,K);vals=[]
 for off,m in zip(offs,ms):
  mv=m&~per[i,y0:y1,x0:x1];inter=(mv&obs).sum();vals.append((float(inter/max(1,(mv|obs).sum())),float(inter/max(1,mv.sum())),float(inter/max(1,obs.sum())),int(mv.sum())))
 best=max(vals);bi=int(np.argmax([v[0] for v in vals]));print(i,'obs',obs.sum(),'best iou',round(best[0],3),'off',round(float(offs[bi]),3),'pr',round(best[1],2),round(best[2],2),'area',best[3])
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/depth_iou.py"
```
<details><summary>output</summary>

```
0 obs 1762 best iou 0.844 off -0.04 pr 0.99 0.85 area 1513
50 obs 2047 best iou 0.881 off -0.06 pr 0.99 0.89 area 1827
100 obs 2054 best iou 0.877 off -0.06 pr 1.0 0.88 area 1807
130 obs 1914 best iou 0.884 off -0.04 pr 0.98 0.9 area 1759
140 obs 1645 best iou 0.82 off -0.04 pr 0.93 0.88 area 1551
150 obs 1629 best iou 0.851 off -0.06 pr 0.95 0.89 area 1524
160 obs 1290 best iou 0.828 off -0.04 pr 0.93 0.88 area 1229
170 obs 1074 best iou 0.726 off -0.22 pr 0.78 0.92 area 1266
180 obs 1123 best iou 0.421 off 0.1 pr 0.64 0.55 area 970
190 obs 137 best iou 0.047 off -0.18 pr 0.05 0.29 area 757
200 obs 358 best iou 0.344 off 0.3 pr 0.54 0.49 area 326
220 obs 587 best iou 0.539 off 0.3 pr 0.89 0.58 area 378
240 obs 437 best iou 0.73 off -0.12 pr 0.84 0.85 area 440
250 obs 806 best iou 0.63 off …[+923 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/static_apply_eval.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
src=open('/work/scratch/analysis/yaw_grid_fast.py').read();exec(src.split(\"inds=\")[0])
def miou(P,i):
 yy,xx=np.where(obj[i]);x0=max(0,xx.min()-25);x1=min(W,xx.max()+26);y0=max(0,yy.min()-25);y1=min(H,yy.max()+26);w=x1-x0;h=y1-y0
 _,ms,_,_=render_batch(P[None],x0,y0,w,h,K);m=ms[0];obs=obj[i,y0:y1,x0:x1]&~per[i,y0:y1,x0:x1];mv=m&~per[i,y0:y1,x0:x1];inter=(mv&obs).sum();return inter/max(1,(mv|obs).sum()),m.sum(),obs.sum()
Pi=np.load('/work/scratch/analysis/staticbest_initial.npz')['P'];Pf=np.load('/work/scratch/analysis/staticbest_final.npz')['P'];ctx=dr.RasterizeCudaContext()
print('initial')
for i in list(range(0,171,5)):
 print(i,tuple(round(x,3) if isinstance(x,float) else x for x in miou(Pi,i)))
print('final')
for i in list(range(420,565,5)):
 print(i,tuple(round(x,3) if isinstance(x,float) else x for x in miou(Pf,i)))
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/static_apply_eval.py"
```
<details><summary>output</summary>

```
initial
0 (0.888, 2039, 1762)
5 (0.891, 2039, 2088)
10 (0.893, 2039, 2055)
15 (0.891, 2039, 2058)
20 (0.886, 2039, 2002)
25 (0.892, 2039, 2048)
30 (0.894, 2039, 2058)
35 (0.892, 2039, 2060)
40 (0.891, 2039, 2013)
45 (0.898, 2039, 2066)
50 (0.9, 2039, 2047)
55 (0.887, 2039, 1995)
60 (0.906, 2039, 2041)
65 (0.896, 2039, 1996)
70 (0.911, 2039, 2057)
75 (0.908, 2039, 2056)
80 (0.903, 2039, 2058)
85 (0.897, 2039, 1995)
90 (0.906, 2039, 2045)
95 (0.906, 2039, 2033)
100 (0.905, 2039, 2054)
105 (0.892, 2039, 1988)
110 (0.891, 2039, 1984)
115 (0.893, 2039, 1986)
120 (0.894, 2039, 1991)
125 (0.89, 2039, 2016)
130 (0.91, 2039, 1914)
135 (0.874, 2039, 1722)
140 (0.856, 2039, 1645)
145 (0.878, 2039, 1620)
150 (0.811, 2039, 1629)
155 (0.479, 2039, 1513)
160 (0.159, 1932, 1290)
165 (0.0, 831, 1143)
170 ( …[+714 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
def yaw(R):
 v=R@np.array([0,0,1.]);return np.degrees(np.arctan2(v[0],v[2]))
y=np.array([yaw(R) for R in T[:,:3,:3]]);yu=np.rad2deg(np.unwrap(np.deg2rad(y)))
for i in range(0,565,5):print(i,round(y[i],1),round(yu[i],1))
print('max ydiff',np.max(np.abs(np.diff(yu))),np.where(np.abs(np.diff(yu))>20)[0])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/build_candidates.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,json,cv2,h5py,torch,trimesh
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
import nvdiffrast.torch as dr
p='/work/inputs';z=np.load(p+'/reconstruction.npz');T=z['T_world_mesh'].astype(float);F=len(T)
Pi=np.load('/work/scratch/analysis/staticbest_initial.npz')['P'].astype(float);Pf=np.load('/work/scratch/analysis/staticbest_final.npz')['P'].astype(float)
# candidates static only
B=T.copy();B[:153]=Pi;B[450:]=Pf
# decompose yaw and residual rotation; dynamic smoothing
yaw=[];res=[]
for R in T[:,:3,:3]:
 v=R@np.array([0,0,1.]);yy=np.arctan2(v[0],v[2]);yaw.append(yy);res.append(Rot.from_euler('y',-yy).as_matrix()@R)
yaw=np.rad2deg(np.unwrap(np.array(yaw)))
# robust spike repair using median filter then savgol
ym=median_filter(yaw,size=7,mode='nearest');ys=savgol_filter(ym,11,2,mode='interp')
# blend endpoints toward static yaw
# static yaw from optimized rotation
def getyaw(R):
 v=R@np.array([0,0,1.]);return np.rad2deg(np.arctan2(v[0],v[2]))
yPi=getyaw(Pi[:3,:3]);yPf=getyaw(Pf[:3,:3]);
# map static optimized yaw into unwrapped branch nearest ys at boundaries
def branch_near(a,ref):
 while a-ref>180:a-=360
 while a-ref<-180:a+=360
 return a
yPi=branch_near(yPi,ys[152]);yPf= …[+1819 chars]
```
<details><summary>output</summary>

```
Pi [[-0.98464733 -0.00434179 -0.17450519  0.47464824]
 [-0.01702402  0.99731213  0.07124327  1.06384206]
 [ 0.17372721  0.07312012 -0.98207498  3.57947874]
 [ 0.          0.          0.          1.        ]] Pf [[-0.60287821  0.01796891 -0.79763353 -0.31678516]
 [-0.0346376   0.99821126  0.04866817  0.28745508]
 [ 0.79708189  0.05696918 -0.60117954  2.75563693]
 [ 0.          0.          0.          1.        ]] yPi -169.92426109364902 yPf -487.0054775452741 yrange -169.733906844815 -486.75286034708427
base rot pct [  1.19   7.54  10.84  24.02 122.77] trans pct [0.007 0.027 0.036 0.06  0.139] energy dyn 5.480827463337673 0.019045331394355013
static rot pct [  0.52   7.65  11.07  24.02 122.77] trans pct [0.004 0.027 0.036 0.076 0.247] energy dyn 5.480827463337673 0.019045331394355013
smooth …[+271 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/dyn_yaw_align.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
src=open('/work/scratch/analysis/yaw_grid_fast.py').read();exec(src.split(\"inds=\")[0])
def fit_score(P,yawdeg,img,om,pm,bbox):
 x0,y0,x1,y1=bbox;w=x1-x0;h=y1-y0;Q=P.copy();Q[:3,:3]=Rot.from_euler('y',np.deg2rad(yawdeg)).as_matrix()@Q[:3,:3]
 obs=om&~pm
 for it in range(3):
  rgbs,masks,_,_=render_batch(Q[None],x0,y0,w,h,K);m=masks[0]&~pm
  if m.sum()<30:return (-9,)*9
  cc=Tcw[:3,:3]@Q[:3,3]+Tcw[:3,3];zc=cc[2];scale=(m.sum()/max(1,obs.sum()))**0.5;yy,xx=np.where(m);yy2,xx2=np.where(obs);dx=(xx2.mean()-xx.mean())*zc/K[0,0];dy=(yy2.mean()-yy.mean())*zc/K[1,1];cn=cc.copy();cn[2]=zc*(1+0.7*(scale-1));cn[0]+=0.8*dx;cn[1]+=0.8*dy;Q[:3,3]=Twc[:3,:3]@cn+Twc[:3,3]
 rgbs,masks,_,_=render_batch(Q[None],x0,y0,w,h,K);rgb=rgbs[0];m=masks[0]&~pm;vis=m&obs
 g1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255;g2=cv2.cvtColor((rgb*255).clip(0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)/255;gh1=g1-cv2.GaussianBlur(g1,(0,0),1.2);gh2=g2-cv2.GaussianBlur(g2,(0,0),1.2)
 if vis.sum()<50:ncc=hp=0
 else:ncc=float(np.corrcoef(g1[vis],g2[vis])[0,1]);hp=float(np.corrcoef(gh1[vis],gh2[vis])[0,1])
 iou=(m&obs).sum()/max(1,(m|obs).sum());sc=ncc+0.2*hp+0.3*iou
 return (float(sc),float(ncc),float(hp),float(iou),int(vis.sum()),int(m.sum()),int(obs.sum()),float(cc[2]),Q)
ctx=dr.RasterizeCudaContex …[+924 chars]
```
<details><summary>output</summary>

```

frame 250 obs 806 over 0.0
combined [(80.0, 0.347, 0.151, 0.631, 627, 814), (75.0, 0.345, 0.156, 0.631, 628, 818), (100.0, 0.302, 0.119, 0.571, 586, 807), (70.0, 0.284, 0.096, 0.63, 631, 827), (65.0, 0.275, 0.081, 0.636, 638, 835), (55.0, 0.265, 0.078, 0.622, 636, 852), (90.0, 0.258, 0.077, 0.61, 609, 801), (85.0, 0.257, 0.063, 0.624, 620, 807)]
ncc [(75.0, 0.156, 0.631), (80.0, 0.151, 0.631), (100.0, 0.119, 0.571), (70.0, 0.096, 0.63), (65.0, 0.081, 0.636)]
iou [(-85.0, 0.761, -0.061), (-80.0, 0.752, -0.093), (-90.0, 0.751, -0.121), (-75.0, 0.742, -0.028), (-95.0, 0.739, -0.132)]

frame 280 obs 804 over 0.0
combined [(170.0, 0.425, 0.174, 0.699, 677, 842), (-140.0, 0.417, 0.227, 0.593, 602, 814), (-125.0, 0.392, 0.225, 0.562, 588, 831), (-165.0, 0.376, 0.121, 0.672, 659, 835), (-90.0, 0. …[+5367 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/eval_candidates.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,cv2,h5py,torch,trimesh,json
import nvdiffrast.torch as dr
from pathlib import Path
p=Path('/work/inputs');m=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(np.asarray(m.vertices),dtype=torch.float32,device='cuda');Tri=torch.tensor(np.asarray(m.faces),dtype=torch.int32,device='cuda')
cam=json.load(open(p/'camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);H,W=1152,1536
with h5py.File(p/'masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
z=np.load('/work/scratch/analysis/candidates.npz');names=['T','B','C','D'];Ts=[z[n].astype(float) for n in names];F=len(Ts[0]);ctx=dr.RasterizeCudaContext();ious=np.zeros((4,F),np.float32);prec=np.zeros_like(ious);rec=np.zeros_like(ious)
for i in range(F):
 yy,xx=np.where(obj[i]);
 if len(xx)<20:continue
 x0=max(0,xx.min()-25);x1=min(W,xx.max()+26);y0=max(0,yy.min()-25);y1=min(H,yy.max()+26);w=x1-x0;h=y1-y0
 obs=obj[i,y0:y1,x0:x1]&~per[i,y0:y1,x0:x1];poses=np.stack([X[i] for X in Ts]);B=len(poses);Tcm=np.einsum('ij,bjk->bik',Tcw,poses).astype(np.float32);R=torch.tensor(Tcm[:,:3,:3],device='cuda');tt=torch.tensor(Tcm[:,:3,3],device='cuda');vc=torch.einsum('bnj,bkj->bnk',V[None].expand(B,-1,-1),R)+tt[:,None,:];x,y,zc=vc[...,0],vc[...,1],vc[...,2];k=K.c …[+1130 chars]
```
<details><summary>output</summary>

```

 all
T iou mean/med/p10 int64 0.7044 0.767 0.4151 prec 0.8831 rec 0.7532
B iou mean/med/p10 int64 0.7567 0.8707 0.4151 prec 0.8798 rec 0.8132
C iou mean/med/p10 int64 0.7575 0.8663 0.4396 prec 0.8836 rec 0.8134
D iou mean/med/p10 int64 0.7053 0.7661 0.4396 prec 0.8871 rec 0.7534

 initial
T iou mean/med/p10 int64 0.7599 0.7641 0.7361 prec 0.9316 rec 0.8047
B iou mean/med/p10 int64 0.8919 0.8938 0.8785 prec 0.9403 rec 0.9451
C iou mean/med/p10 int64 0.8919 0.8938 0.8785 prec 0.9403 rec 0.9451
D iou mean/med/p10 int64 0.7599 0.7641 0.7361 prec 0.9316 rec 0.8047

 dynamic
T iou mean/med/p10 int64 0.6173 0.6488 0.2721 prec 0.814 rec 0.6862
B iou mean/med/p10 int64 0.6173 0.6488 0.2721 prec 0.814 rec 0.6862
C iou mean/med/p10 int64 0.6189 0.6445 0.2782 prec 0.8213 rec 0.6866
D iou mean/med/p10 …[+612 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/build_E.py <<'PY'
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/scratch/analysis/candidates.npz');T=z['T'];B=z['B'];C=z['C'];Pi=z['Pi'];Pf=z['Pf'];F=len(T)
E=B.copy()
# copy smoothed dynamic
E[153:450]=C[153:450]
def slerp(A,Bmat,t):
 ra=Rot.from_matrix(A);rb=Rot.from_matrix(Bmat);delta=(ra.inv()*rb).as_rotvec();return (ra*Rot.from_rotvec(t*delta)).as_matrix()
# smoothstep endpoint blends, remain within static optimized pose at phase boundaries
for i in range(153,165):
 t=(i-152)/13;t=t*t*(3-2*t);E[i,:3,:3]=slerp(Pi[:3,:3],C[i,:3,:3],t);E[i,:3,3]=(1-t)*Pi[:3,3]+t*C[i,:3,3]
for i in range(438,450):
 t=(i-437)/13;t=t*t*(3-2*t);E[i,:3,:3]=slerp(C[i,:3,:3],Pf[:3,:3],t);E[i,:3,3]=(1-t)*C[i,:3,3]+t*Pf[:3,3]
np.save('/work/scratch/analysis/E.npy',E)
# metrics smooth
rr=np.degrees(np.linalg.norm(Rot.from_matrix(E[1:,:3,:3]@np.transpose(E[:-1,:3,:3],(0,2,1))).as_rotvec(),axis=1));tt=np.linalg.norm(np.diff(E[:,:3,3],axis=0),axis=1)
print('smooth',np.percentile(rr,[50,90,95,99,100]),np.percentile(tt,[50,90,95,99,100]),rr[153:449].mean(),tt[153:449].mean())
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/build_E.py
# modify eval to include E
python - <<'PY'
p='/work/scratch/analysis/eval_candidates.py';s=open(p).read();s=s.replace(\"z=np.load …[+676 chars]
```
<details><summary>output</summary>

```
smooth [ 0.35855382  2.97215598  4.29339453 15.09149426 29.06274594] [0.00390052 0.01872576 0.02295421 0.03098062 0.03609335] 2.35436068384371 0.012886357804465237

 all
T iou mean/med/p10 int64 0.7044 0.767 0.4151 prec 0.8831 rec 0.7532
B iou mean/med/p10 int64 0.7567 0.8707 0.4151 prec 0.8798 rec 0.8132
C iou mean/med/p10 int64 0.7575 0.8663 0.4396 prec 0.8836 rec 0.8134
E iou mean/med/p10 int64 0.7564 0.8715 0.4396 prec 0.8803 rec 0.8147

 initial
T iou mean/med/p10 int64 0.7599 0.7641 0.7361 prec 0.9316 rec 0.8047
B iou mean/med/p10 int64 0.8919 0.8938 0.8785 prec 0.9403 rec 0.9451
C iou mean/med/p10 int64 0.8919 0.8938 0.8785 prec 0.9403 rec 0.9451
E iou mean/med/p10 int64 0.8919 0.8938 0.8785 prec 0.9403 rec 0.9451

 dynamic
T iou mean/med/p10 int64 0.6173 0.6488 0.2721 prec 0.814 re …[+774 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
E=np.load('/work/scratch/analysis/E.npy');T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for X,n in [(T,'base'),(E,'E')]:
 d=np.degrees(np.linalg.norm(Rot.from_matrix(X[1:,:3,:3]@np.transpose(X[:-1,:3,:3],(0,2,1))).as_rotvec(),axis=1));
 print(n,sorted([(round(float(d[i]),1),i,i+1) for i in np.argsort(d)[-15:]],reverse=True))
PY"
```
<details><summary>output</summary>

```
base [(122.8, 197, 198), (35.6, 250, 251), (33.1, 251, 252), (30.1, 372, 373), (26.6, 373, 374), (25.0, 189, 190), (23.5, 374, 375), (22.5, 361, 362), (22.4, 188, 189), (20.3, 362, 363), (19.4, 359, 360), (18.3, 161, 162), (17.3, 179, 180), (16.5, 233, 234), (15.8, 319, 320)]
E [(29.1, 197, 198), (27.1, 196, 197), (26.3, 198, 199), (24.1, 195, 196), (21.3, 199, 200), (17.1, 194, 195), (13.9, 200, 201), (8.3, 193, 194), (8.1, 328, 329), (7.7, 329, 330), (7.7, 202, 203), (7.4, 327, 328), (7.3, 330, 331), (6.6, 326, 327), (6.3, 331, 332)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '"'^  def '"\\|"'^    def '"' /workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py | head -100; echo ===; grep -n '"'^  def '"\\|"'^    def '"' /workspace/v2d_cari4d/lib/cari4d/estimater.py | head -100; echo ===; sed -n '250,520p' /workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py"
```
<details><summary>output</summary>

```
118:    def __init__(
136:    def _init_fp(self, mesh: Mesh) -> None:
149:    def register(
168:    def track_one(
185:    def track_one_with_recovery(
221:    def track_one_particles(
369:    def _mask_iou(
395:    def reset_to_pose(self, pose: Transform3d) -> None:
408:    def rescale_mesh(self, scale: float) -> None:
412:    def rescale_to(self, scale: float) -> None:
427:    def _chamfer_score(
478:    def _score_scale(
536:    def align_depth_to_object(
621:    def estimate_scale_grid_search(
===
95:  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None,
124:  def init_refiner(self, refiner, cfg):
130:  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
166:  def get_tf_to_centered_mesh(self):
171:  def to_d …[+12319 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,260p' /workspace/v2d_mesh/lib/mesh.py; sed -n '1,220p' /workspace/v2d_common/datatypes.py | head -220"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
import numpy as np
import trimesh
from PIL import Image as PILImage

from v2d.common.datatypes import BoundingBox3d, Image  # re-exported for convenience


@dataclass
class Mesh:
    """
    Triangle mesh with optional texture (UV + image) or per-vertex RGBA colors.

    UV texture takes priority over vertex_colors when both are present.
    Vertices are in a right-handed coordinate system. The canonical camera
    convention used by mesh_render_* functions is OpenCV: camera at the origin
    looking along +Z, Y axis pointing down.

    Appearance priority (highest to lowest):
      1. uv + texture  — UV-mapped tex …[+10655 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /weights/foundationpose -maxdepth 4 -type f -printf '%p %s\\n' | sort; find /weights -maxdepth 2 -type d | sort | head -100"
```
<details><summary>output</summary>

```
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/config.yml 708
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth 68220109
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/config.yml 778
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth 190229389
/weights
/weights/cari4d
/weights/cari4d/.cache
/weights/cari4d/2026-08-25-09-35-57
/weights/foundationpose
/weights/foundationpose/nvlabs_pytorch
/weights/grounding_dino
/weights/hf_home
/weights/hf_home/hub
/weights/hf_home/xet
/weights/sam2
/weights/sam2/.cache
/weights/sam3d_body
/weights/sam3d_body/checkpoints
/weights/sam3d_body/torch_home

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,220p' /workspace/v2d_foundation_pose/lib/backends.py; sed -n '1,220p' /workspace/v2d_foundation_pose/lib/fp_utils.py; grep -R \"FoundationPoseTracker(\" -n /workspace /code/reconstruction/modules | head -20"
```
<details><summary>output</summary>

```
/workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py:6:    tracker = FoundationPoseTracker(mesh, weights_dir)
/workspace/v2d_foundation_pose/lib/run_align_depth_to_object.py:69:    tracker = FoundationPoseTracker(mesh, weights_dir, backend=backend)
/workspace/v2d_foundation_pose/lib/run_ekf_smoothing.py:263:    tracker = FoundationPoseTracker(mesh, weights_dir, backend=backend)
/workspace/v2d_foundation_pose/lib/run_estimate_mesh_scale.py:77:    tracker = FoundationPoseTracker(mesh, weights_dir, backend=backend)
/workspace/v2d_foundation_pose/lib/run_video_to_poses.py:62:    tracker = FoundationPoseTracker(mesh, weights_dir, backend=backend)
/code/reconstruction/modules/v2d_foundation_pose/lib/foundation_pose_tracker.py:6:    tracker = FoundationPoseTracker(mesh, weights_dir)
/cod …[+608 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '118,190p' /workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py; sed -n '90,180p' /workspace/v2d_cari4d/lib/cari4d/estimater.py; sed -n '1,130p' /workspace/v2d_foundation_pose/lib/run_video_to_poses.py"
```
<details><summary>output</summary>

```
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
        self._particles: np.ndarray | None = None  # (N, 4, 4) poses in pose_last frame
        self._weights: np.ndarray | None = None    # (N,) normalised
        self._init_fp(mesh)

    def _init_fp(self, mesh: Mesh) -> None:
        tm = mesh.to_trimesh()
        self._est = FoundationP …[+10315 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"scorer.predict\\|pose_last\\|tf_to_centered\" /workspace/v2d_cari4d/lib/cari4d/estimater.py | head -60; sed -n '330,430p' /workspace/v2d_cari4d/lib/cari4d/estimater.py; sed -n '230,340p' /workspace/v2d_cari4d/lib/cari4d/estimater.py"
```
<details><summary>output</summary>

```
116:    self.pose_last = None   # Used for tracking; per the centered mesh
166:  def get_tf_to_centered_mesh(self):
299:        center = self.pose_last[..., :3, 3].clone().cpu().numpy()
306:        center = self.pose_last[..., :3, 3].clone().cpu().numpy()
332:    score_result = self.scorer.predict(mesh=self.mesh, rgb=rgb_for_model, depth=depth_for_model, K=K, ob_in_cams=scorer_poses, normal_map=normal_map,
351:    if obj_vis is not None and self.pose_last is not None:
352:      pose_prev = self.pose_last[None].expand(len(poses), -1, -1)
361:    best_pose = poses[0]@self.get_tf_to_centered_mesh()
362:    self.pose_last = poses[0]
374:    if self.pose_last is None:
385:                                     ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None,
390:    s …[+9078 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '160,175p' /workspace/v2d_cari4d/lib/cari4d/estimater.py"
```
<details><summary>output</summary>

```
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")

  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center

  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/fp_score_test.py <<'PY'
import os,sys,json,cv2,h5py,numpy as np,torch,trimesh
sys.path.insert(0,'/workspace')
from scipy.spatial.transform import Rotation as Rot
from v2d.mesh.lib.mesh import Mesh
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
p='/work/inputs';T=np.load(p+'/reconstruction.npz')['T_world_mesh'].astype(np.float32);cam=json.load(open(p+'/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);K=np.array(cam['K_pipeline']);A=np.array(json.load(open('/pipeline/export/episode_000027/wild_export.json'))['source_object_mesh_to_aligned_transform'])
mesh=Mesh.load('/pipeline/export/episode_000027/object_mesh/output_aligned.glb',force_mesh=True)
tracker=FoundationPoseTracker(mesh,'/weights/foundationpose',backend='nvlabs_pytorch');print('loaded')
cap=cv2.VideoCapture(p+'/video.mp4');frames=[]
while True:
 ok,im=cap.read();
 if not ok:break
 frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
with h5py.File(p+'/masks.h5','r') as f:obj=f['object'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:ds=f['aligned']['front_stereo_camera_left']
for i in [200,250,300,340,400,440]:
 rgb=frames[i];depth=cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000;mask=obj[i]
 prepared=tracker._est.prepare_registration_frame(K,rgb,depth,mask);dt=prepared['depth'];poses=[]
 for deg in [-90,-30,0,30,90,180]:
  Q=T[i].copy();Q[:3,:3]=Rot.from_euler('y',np.deg2rad(deg)).as_m …[+544 chars]
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
[__init__()] Using pretrained model from /weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth
/workspace/v2d_cari4d/lib/cari4d/learning/training/predict_score.py:141: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the defa …[+3917 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/fp_score_test.py';s=open(p).read();s=s.replace(\"with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:ds=f['aligned']['front_stereo_camera_left']\\nfor i in [200,250,300,340,400,440]:\\n rgb=frames[i];depth=cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000;mask=obj[i]\", \"inds=[200,250,300,340,400,440]\\nwith h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:\\n ds=f['aligned']['front_stereo_camera_left']; deps={i:cv2.imdecode(ds[i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in inds}\\nfor i in inds:\\n rgb=frames[i];depth=deps[i];mask=obj[i]\")
open(p,'w').write(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/fp_score_test.py"
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
[__init__()] Using pretrained model from /weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth
/workspace/v2d_cari4d/lib/cari4d/learning/training/predict_score.py:141: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the defa …[+2491 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep -n "def predict" -A100 /workspace/v2d_cari4d/lib/cari4d/learning/training/predict_score.py | head -130; grep -n "erode_depth" -A20 /workspace/v2d_cari4d/lib/cari4d/Utils.py | head -60'
```
<details><summary>output</summary>

```
151:  def predict(self, rgb, depth, K, ob_in_cams, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, num_vis=None, no_text=False, rgb_only=False, return_pose_data=False):
152-    '''
153-    @rgb: np array (H,W,3)
154-    '''
155-    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')
156-
157-    if not self.cfg.use_normal:
158-      normal_map = None
159-
160-    if mesh_tensors is None:
161-      mesh_tensors = make_mesh_tensors(mesh)
162-
163-    rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
164-    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
165-
166-    if rgb_only:
167-      depth = torch.zeros_like(depth)
168-
169-    pose_data = make_crop_data_batch(self.cfg.input_resize, ob_in_c …[+3998 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analysis/fp_score_test.py';s=open(p).read()
s=s.replace(\"from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker\", \"from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker\\nfrom v2d.foundation_pose.lib.fp_utils import erode_depth,bilateral_filter_depth\")
s=s.replace(\"prepared=tracker._est.prepare_registration_frame(K,rgb,depth,mask);dt=prepared['depth'];poses=[]\", \"dt=bilateral_filter_depth(erode_depth(torch.as_tensor(depth,device='cuda',dtype=torch.float),radius=2,device='cuda'),radius=2,device='cuda');poses=[]\")
open(p,'w').write(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/fp_score_test.py"
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
[__init__()] Using pretrained model from /weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth
/workspace/v2d_cari4d/lib/cari4d/learning/training/predict_score.py:141: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the defa …[+3352 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'df -h / /work /tmp; du -sh /work/scratch /work/tools /tmp/* 2>/dev/null | sort -h | tail -30'
```
<details><summary>output</summary>

```
Filesystem                                               Size  Used Avail Use% Mounted on
overlay                                                   64M   64M  8.0K 100% /
10.127.102.71:/Group/hanzhou/v2d/agent_runs/T1_ds_ep027  139T   49T   90T  36% /work
tmpfs                                                     64M   64M  8.0K 100% /tmp
0	/tmp/foundationpose_debug
4.0K	/tmp/material.mtl
4.0K	/work/tools
5.3M	/tmp/material_0.png
17M	/tmp/233be8bc-8cea-423b-bd23-81e54f218e00.obj
18M	/work/scratch
22M	/tmp/33b33495-9b59-4b36-bfab-5bbbe4a7854e.obj
22M	/tmp/bf838b32-868e-473e-a5be-bf56e61488bc.obj

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
for pat in ('*.obj','material.mtl','material_*.png'):
 for p in Path('/tmp').glob(pat):
  try:p.unlink()
  except Exception as e:print(e)
for p in Path('/tmp/foundationpose_debug').glob('*'):
 try:p.unlink()
 except Exception as e:print(e)
PY
df -h /tmp /; du -sh /tmp"
```
<details><summary>output</summary>

```
Filesystem      Size  Used Avail Use% Mounted on
tmpfs            64M   20K   64M   1% /tmp
overlay          64M   20K   64M   1% /
0	/tmp

```
</details>

**shell** (exit 135)
```
/usr/bin/bash -c "mkdir -p /work/scratch/tmp
python - <<'PY'
p='/work/scratch/analysis/fp_score_test.py';s=open(p).read();s=s.replace(\"import os,sys,json,cv2,h5py,numpy as np,torch,trimesh\", \"import os,sys,json,cv2,h5py,numpy as np,torch,trimesh,tempfile\\nPath('/work/scratch/tmp').mkdir(parents=True,exist_ok=True);tempfile.tempdir='/work/scratch/tmp'\")
s=s.replace(\"import os,sys,json\", \"from pathlib import Path\\nimport os,sys,json\")
open(p,'w').write(s)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/fp_score_test.py"
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
[__init__()] Using pretrained model from /weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth
/workspace/v2d_cari4d/lib/cari4d/learning/training/predict_score.py:141: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the defa …[+2145 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'git -C /workspace/third_party/nvdiffrast rev-parse HEAD 2>/dev/null || true; git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true; git -C /workspace/v2d_foundation_pose rev-parse HEAD 2>/dev/null || true'
```
<details><summary>output</summary>

```
253ac4fcea7de5f396371124af597e6cc957bfae
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth /weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth'
```
<details><summary>output</summary>

```
81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26  /weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth
774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60  /weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy,scipy,cv2,trimesh,torch,h5py,torchvision
import nvdiffrast
print('python',sys.version.split()[0]);print('numpy',numpy.__version__);print('scipy',scipy.__version__);print('opencv',cv2.__version__);print('trimesh',trimesh.__version__);print('torch',torch.__version__);print('h5py',h5py.__version__);print('torchvision',torchvision.__version__);print('nvdiffrast',getattr(nvdiffrast,'__version__','unknown'),nvdiffrast.__file__)
PY"
```
<details><summary>output</summary>

```
python 3.11.10
numpy 1.26.3
scipy 1.16.3
opencv 4.11.0
trimesh 5.1.1
torch 2.5.1+cu124
h5py 3.16.0
torchvision 0.20.1+cu124
nvdiffrast 0.4.0 /opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF woven-basket trajectory.

The released baseline is strong on moving, clearly visible frames, but it has
two pathologies that are visible in this episode:
  * the basket has long stationary phases in which the per-frame tracker still
    drifts in depth/translation; and
  * hand/body occlusion produces short rotation/translation spikes.
The method below therefore (1) estimates one pooled pose for each stationary
phase directly from a temporal median image and a pooled silhouette, and
(2) applies robust temporal smoothing plus smooth endpoint blends only in the
carried interval.  No pose values are copied from a precomputed solution.

Run offline:
    python /work/outputs/solve.py --inputs /work/inputs '"\\
        --out /work/outputs/object_poses_world.npz
\"\"\"

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

# Keep small-matrix BLAS/OpenCV calls deterministic and avoid oversubscription.
os.environ.setdefault(\"OMP_NUM_THREADS\", \"1\")
os.environ.setdefault(\"OPENBLAS_NUM_THREADS\", \"1\")
os.environ.setdefault(\"MKL_NUM_THREADS\", \"1\")

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
import nvdiffrast.torch as dr …[+27421 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/REPORT.md
Runtime: 14.2s
Mask IoU mean baseline/result: 0.7044 / 0.7166

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/outputs/REPORT.md'
```
<details><summary>output</summary>

```
# Woven-basket 6-DoF trajectory recovery

## Result

`object_poses_world.npz` contains `T_world_mesh` as float64 with shape
`(565, 4, 4)`.  Each matrix maps the exact input mesh-file coordinates
to the released world frame.

## Recovered phases

| Phase | Frames | Motion model | Confidence |
|---|---:|---|---|
| Initial rest on floor | 0-152 | one constant pose | high |
| Pickup and carry | 153-437 | robustly smoothed rigid motion | medium |
| Placement transition | 438-449 | blend into the desk pose | medium-low |
| Final rest on desk | 450-564 | one constant pose | high |

The phase boundaries were selected from the pooled-mask test and verified in
the video.  The initial pooled pose retains mask IoU above 0.85 through frame
152 and drops at the lift; the final pooled pose reaches IoU ab …[+3809 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/outputs/object_poses_world.npz');R=z['T_world_mesh'];E=np.load('/work/scratch/analysis/E.npy')
print(R.shape,R.dtype)
print('static initial result',R[100]);print('scratch',E[100]);
print('static final result',R[520]);print('scratch',E[520])
print('diff',np.abs(R-E).max(),np.percentile(np.abs(R-E),[50,90,99,100]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace('BLEND_FRAMES = 13  # 153..164 and 438..449 in the released 565-frame clip', 'BLEND_FRAMES = 13  # 153..164 and 438..449 in the released 565-frame clip\\n\\n# Crop windows used for pooled evidence.  They were verified against the\\n# static-mask extents; retaining fixed windows makes raster quantization and\\n# texture scoring reproducible across runs.\\nINITIAL_COMPOSITE_BBOX = (740, 700, 940, 850)\\nFINAL_COMPOSITE_BBOX = (480, 540, 730, 710)')
s=s.replace('    for frame in range(dynamic_start, dynamic_start + BLEND_FRAMES):', '    for frame in range(dynamic_start, dynamic_start + BLEND_FRAMES - 1):')
s=s.replace('    release_blend_start = FINAL_STATIC_START - BLEND_FRAMES\\n    for frame in range(release_blend_start, FINAL_STATIC_START):\\n        fraction = (frame - release_blend_start + 1) / float(BLEND_FRAMES)', '    release_blend_start = FINAL_STATIC_START - BLEND_FRAMES + 1\\n    for frame in range(release_blend_start, FINAL_STATIC_START):\\n        fraction = (frame - release_blend_start + 1) / float(BLEND_FRAMES)')
start=s.index('    def pooled_bbox(start: int, end: int)')
end=s.index('    initial_end = min(INITIAL_POOL_END, frame_count - 1)', start)
s=s[:start]+s[end:]
s=s.replace('    initial_bbox = pooled_bbox(0, initial_end)\\n    final_bbox = pooled_bbox(final_start, frame_count - 1)\\n    initial_rgb, initial_mask, initial_person = make_static_composite(', '    initial_bb …[+365 chars]
```
<details><summary>output</summary>

```
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/REPORT.md
Runtime: 11.4s
Mask IoU mean baseline/result: 0.7044 / 0.7090

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -A8 'Validation without' /work/outputs/REPORT.md; grep 'Pooled-pose' -A2 /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
## Validation without ground truth

Visible silhouette IoU excludes person-mask pixels and compares the rendered
mesh with the SAM2 object mask.  Values are mean / median; higher is better.

| Phase | Frames | Baseline IoU | Result IoU | Delta |
|---|---:|---:|---:|---:|
| Initial rest on floor | 0-152 | 0.760 / 0.764 | 0.721 / 0.724 | -0.039 |
| Pickup / carried | 153-437 | 0.607 / 0.640 | 0.604 / 0.611 | -0.004 |
Pooled-pose texture evidence: initial NCC `0.095` with mask
IoU `0.725` at yaw offset `+45.0` degrees;
final NCC `0.119` with mask IoU `0.939`

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];E=np.load('/work/scratch/analysis/E.npy');B=np.load('/work/scratch/analysis/candidates.npz')['B'];
for i in [0,100,152,153,165,450,520,564]:
 print(i,'R-E',np.abs(R[i]-E[i]).max(),'R-B',np.abs(R[i]-B[i]).max())
print('max',np.abs(R-E).max())
PY"
```
<details><summary>output</summary>

```
0 R-E 0.5922746562653205 R-B 0.5922746562653205
100 R-E 0.5922746562653205 R-B 0.5922746562653205
152 R-E 0.5922746562653205 R-B 0.5922746562653205
153 R-E 0.5835734160449324 R-B 0.36355621416705364
165 R-E 0.0 R-B 0.051330029155753354
450 R-E 0.06723069287352379 R-B 0.06723069287352379
520 R-E 0.06723069287352379 R-B 0.06723069287352379
564 R-E 0.06723069287352379 R-B 0.06723069287352379
max 0.5922746562653205

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/test_solve_fit.py <<'PY'
import sys,importlib.util,json,numpy as np,h5py
spec=importlib.util.spec_from_file_location('sol','/work/outputs/solve.py');sol=importlib.util.module_from_spec(spec);spec.loader.exec_module(sol)
import trimesh
p='/work/inputs';K,Twc,Tcw=sol.read_camera(sol.Path(p));mesh=trimesh.load(p+'/object_mesh.glb',force='scene',process=False).to_geometry();r=sol.MeshRenderer(mesh)
z=np.load('/work/scratch/analysis/comp_initial.npz');P=z['P'];
pose,yaw,diag=sol.fit_static_pose(r,P,z['img'],z['om'].astype(bool),z['pm'].astype(bool),tuple(z['bbox'].astype(int)),K,Twc,Tcw)
print('yaw',yaw,diag,'pose',pose)
np.save('/work/scratch/analysis/solvefit_initial.npy',pose)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/scratch/analysis/test_solve_fit.py"
```
<details><summary>output</summary>

```
yaw 5.0 {'score': 0.5122543504204302, 'ncc': 0.20307805031629036, 'highpass_ncc': -0.03254987202832011, 'mask_iou': 0.9019607843137255} pose [[-0.98464733 -0.00434179 -0.17450519  0.47464824]
 [-0.01702402  0.9973121   0.07124327  1.063842  ]
 [ 0.17372721  0.07312012 -0.982075    3.5794787 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
# Insert helper after _image_ncc
marker='def fit_static_pose('
helper='''def anchor_depth_to_moge(
    renderer: MeshRenderer,
    pose: np.ndarray,
    frame: int,
    object_mask: np.ndarray,
    person_mask: np.ndarray,
    camera_from_world: np.ndarray,
    world_from_camera: np.ndarray,
    pipeline_intrinsics: np.ndarray,
    depth_h5: Path,
) -> tuple[np.ndarray, float]:
    \"\"\"Initialize pooled-pose depth from MoGe-2 camera depth.

    This is only an initialization.  The pooled silhouette and texture fit that
    follows determines the final pose.  MoGe depth in "'`/pipeline` is in the
    camera frame with `K_pipeline`; it is not assumed to be world-metric.
    """
    with h5py.File(depth_h5, "r") as handle:
        encoded = handle["aligned"]["front_stereo_camera_left"][frame]
    depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return pose, 0.0
    depth = depth.astype(np.float32) / 1000.0
    yy, xx = np.where(object_mask[frame])
    bbox = (
        max(0, int(yy.min()) - 20),
        max(0, int(yy.min()) - 20),  # overwritten below; keeps tuple ordering explicit
        min(1536, int(xx.max()) + 21),
        min(1152, int(yy.max()) + 21),
    )
    bbox = (
        max(0, int(xx.min()) - 20),
        max(0, int(yy.min()) - 20),
        min(1536, int(xx.max()) + 21),
        min(1152, int(yy.max()) + 21),
    )
    _, masks, rendered_depth = rendere …[+2617 chars]
```
<details><summary>output</summary>

```
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/REPORT.md
Runtime: 13.8s
Mask IoU mean baseline/result: 0.7044 / 0.7569

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -A12 'Validation without' /work/outputs/REPORT.md; grep 'Pooled-pose' -A2 /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
## Validation without ground truth

Visible silhouette IoU excludes person-mask pixels and compares the rendered
mesh with the SAM2 object mask.  Values are mean / median; higher is better.

| Phase | Frames | Baseline IoU | Result IoU | Delta |
|---|---:|---:|---:|---:|
| Initial rest on floor | 0-152 | 0.760 / 0.764 | 0.892 / 0.898 | +0.133 |
| Pickup / carried | 153-437 | 0.607 / 0.640 | 0.606 / 0.617 | -0.001 |
| Placement transition | 438-449 | 0.851 / 0.850 | 0.878 / 0.871 | +0.026 |
| Final rest on desk | 450-564 | 0.856 / 0.856 | 0.937 / 0.938 | +0.081 |

Pooled-pose texture evidence: initial NCC `0.173` with mask
Pooled-pose texture evidence: initial NCC `0.173` with mask
IoU `0.901` at yaw offset `-5.0` degrees;
final NCC `0.119` with mask IoU `0.938`

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];E=np.load('/work/scratch/analysis/E.npy')
print(R.shape,R.dtype,'finite',np.isfinite(R).all())
print('pose diff max',np.abs(R-E).max(),'pct',np.percentile(np.abs(R-E),[50,90,99,100]))
for i in [0,100,152,153,164,165,450,520,564]:print(i,np.abs(R[i]-E[i]).max())
print('det',np.linalg.det(R[:,:3,:3]).min(),np.linalg.det(R[:,:3,:3]).max())
PY"
```
<details><summary>output</summary>

```
(565, 4, 4) float64 finite True
pose diff max 0.17362152229332387 pct [0.         0.03511815 0.17362152 0.17362152]
0 0.17362152229332387
100 0.17362152229332387
152 0.17362152229332387
153 0.17063188162707218
164 0.0025816893359231763
165 0.0
450 0.12937436143151082
520 0.12937436143151082
564 0.12937436143151082
det 0.9999985766054498 1.0000001186477767

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/outputs/REPORT.md | sed -n '38,62p'"
```
<details><summary>output</summary>

```
    38	   two pooled static poses over 13 frames.  This removes the large translation
    39	   steps left by independent per-frame depth estimates at pickup and release.
    40	4. **Symmetry branch.**  Both the baseline and its 180-degree vertical symmetry
    41	   transform were scored with textured rendering.  Silhouette agreement favored
    42	   the baseline branch throughout this episode, so no per-frame flip was applied;
    43	   the temporal decoder still removes tracker jumps without changing branch.
    44	
    45	The large moving-frame changes are therefore supported by temporal motion and
    46	static anchors, not by a per-frame correction table.
    47	
    48	## Validation without ground truth
    49	
    50	Visible silhouette IoU excludes person-mask pixels and compares  …[+696 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "tail -15 /work/outputs/REPORT.md; echo '---'; cat /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```

* **Initial floor pose: high.**  The object is unobstructed, large enough for
  textured comparison, and all static observations support one pose.
* **Carried interval: medium.**  Translation and yaw are temporally constrained,
  but hand occlusion around frames 188-246 makes the absolute symmetry branch
  less certain than in the rest phases.
* **Final desk pose: high.**  The basket is visible from frame 450 onward and
  the pooled pose has the strongest silhouette support in the episode.

## Runtime

Deterministic solve runtime: **13.8 s** on the provided GPU.
The script uses no network access.  Besides the inputs, it reads the provided
`/pipeline/depth/moge2/aligned.h5` only to initialize depth for the two pooled
stationary poses; all final pose values are recomputed from rendered evid …[+1804 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "OPENBLAS_NUM_THREADS=1 python - <<'PY'
import numpy as np,h5py
z=np.load('/work/scratch/analysis/candidate_metrics.npz');b=z['ious'][0];e=z['ious'][3]
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
a=o.sum((1,2));ov=(o&p).sum((1,2))/np.maximum(1,a)
for lo,hi in [(0,500),(500,1000),(1000,1500),(1500,2200),(2200,99999)]:
 m=(a>=lo)&(a<hi)&(np.arange(565)>=153)&(np.arange(565)<=449)
 print(lo,hi,m.sum(),'base',b[m].mean() if m.any() else None,'E',e[m].mean() if m.any() else None,'delta',(e[m]-b[m]).mean() if m.any() else None)
for th in [.05,.1,.2,.3]:
 m=(ov<th)&(np.arange(565)>=153)&(np.arange(565)<=449)
 print('ov<',th,m.sum(),b[m].mean(),e[m].mean(),(e[m]-b[m]).mean())
print('best/worst delta frames',np.argsort(e-b)[:10],np.argsort(e-b)[-10:])
PY"
```
<details><summary>output</summary>

```
0 500 13 base 0.15015481 E 0.16585325 delta 0.015698444
500 1000 108 base 0.49947527 E 0.5048644 delta 0.005389083
1000 1500 52 base 0.53543836 E 0.54195285 delta 0.006514502
1500 2200 62 base 0.7721385 E 0.747134 delta -0.025004592
2200 99999 62 base 0.83445716 E 0.83874625 delta 0.0042891293
ov< 0.05 196 0.6863067 0.6834997 -0.0028069327
ov< 0.1 224 0.6829184 0.6782838 -0.0046345466
ov< 0.2 254 0.6638835 0.6614301 -0.0024533374
ov< 0.3 268 0.6511617 0.64956856 -0.0015931212
best/worst delta frames [157 158 159 360 156 362 361 373 370 372] [ 67  73 142 148 147 126 176 161 175 162]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
# remove duplicate bbox construction
old='''    bbox = (
        max(0, int(yy.min()) - 20),
        max(0, int(yy.min()) - 20),  # overwritten below; keeps tuple ordering explicit
        min(1536, int(xx.max()) + 21),
        min(1152, int(yy.max()) + 21),
    )
    bbox = (
'''
s=s.replace(old,'    bbox = (\\n')
# report blending wording
s=s.replace('two pooled static poses over 13 frames.', 'the two pooled static poses over short 12-frame transition windows.')
# tools pipeline usage
s=s.replace(\""'`solve.py` does not need `/pipeline`, `/weights`, or `/code`.", "`solve.py` also reads `/pipeline/depth/moge2/aligned.h5` only to initialize'"\\ncamera depth for the two pooled stationary-pose fits.  It does not need\\n"'`/weights` or `/code`.")
open(p,'"'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
grep -n \"bbox = (\" -A7 /work/outputs/solve.py | head -12
grep -n \"reads /pipeline\\|short 12-frame\" /work/outputs/solve.py"
```
<details><summary>output</summary>

```
232:    bbox = (
233-        max(0, int(xx.min()) - 20),
234-        max(0, int(yy.min()) - 20),
235-        min(1536, int(xx.max()) + 21),
236-        min(1152, int(yy.max()) + 21),
237-    )
238-    _, masks, rendered_depth = renderer.render(pose[None], bbox, pipeline_intrinsics, camera_from_world)
239-    x0, y0, x1, y1 = bbox
--
400:        bbox = (x0, y0, x1, y1)
401-        observed = object_masks[frame, y0:y1, x0:x1] & ~person_masks[frame, y0:y1, x0:x1]
402-        poses = np.stack([trajectories[name][frame] for name in names], axis=0)
503:   the two pooled static poses over short 12-frame transition windows.  This removes the large translation

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/outputs -maxdepth 2 -printf '%y %p %s\\n' | sort"
```
<details><summary>output</summary>

```
d /work/outputs 4096
d /work/outputs/__pycache__ 4096
f /work/outputs/REPORT.md 4754
f /work/outputs/TOOLS.md 1794
f /work/outputs/__pycache__/solve.cpython-311.pyc 47159
f /work/outputs/object_poses_world.npz 72598
f /work/outputs/solve.py 31047

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/REPORT.md
Runtime: 13.2s
Mask IoU mean baseline/result: 0.7044 / 0.7569

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
p=Path('/work/outputs/__pycache__')
if p.exists():
 for f in p.glob('*'):
  f.unlink()
 p.rmdir()
PY
python - <<'PY'
import json,numpy as np,h5py
from pathlib import Path
p=Path('/work/outputs');z=np.load(p/'object_poses_world.npz')
print('keys',z.files)
T=z['T_world_mesh'];print('shape',T.shape,'dtype',T.dtype,'finite',np.isfinite(T).all())
print('last rows',np.unique(T[450:],axis=0).shape[0], 'initial constant',np.unique(T[:153],axis=0).shape[0])
print('det range',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
print('files',[(x.name,x.stat().st_size) for x in sorted(p.iterdir())])
PY
cat /work/outputs/TOOLS.md | tail -20
tail -10 /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
keys ['T_world_mesh']
shape (565, 4, 4) dtype float64 finite True
last rows 1 initial constant 1
det range 0.9999985766054498 1.0000001186477767
files [('REPORT.md', 4782), ('TOOLS.md', 1903), ('object_poses_world.npz', 72598), ('solve.py', 31047)]
camera depth for the two pooled stationary-pose fits.  It does not need
`/weights` or `/code`.

## Tried but not required by `solve.py`

* NVIDIA CARI4D / FoundationPose wrappers under `/code/reconstruction`
  (repository commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`).  The
  provided FoundationPose outputs were inspected and used only as the released
  baseline evidence.  A direct FoundationPose scorer experiment was attempted
  but not used by the final deterministic script.
* Provided FoundationPose PyTorch weights (not downloaded here):
 …[+983 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/analysis/final_overlay.py <<'PY'
import importlib.util,json,cv2,h5py,numpy as np,trimesh
spec=importlib.util.spec_from_file_location('sol','/work/outputs/solve.py');sol=importlib.util.module_from_spec(spec);spec.loader.exec_module(sol)
p='/work/inputs';K,Twc,Tcw=sol.read_camera(sol.Path(p));T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];mesh=trimesh.load(p+'/object_mesh.glb',force='scene',process=False).to_geometry();r=sol.MeshRenderer(mesh)
cap=cv2.VideoCapture(p+'/video.mp4');frames=[]
while True:
 ok,im=cap.read();
 if not ok:break
 frames.append(im)
with h5py.File(p+'/masks.h5','r') as f:o=f['object'][:];pp=f['person'][:]
ims=[]
for i in [0,100,152,153,180,200,250,300,340,400,440,450,500,560]:
 yy,xx=np.where(o[i]);x0=max(0,xx.min()-45);x1=min(1536,xx.max()+46);y0=max(0,yy.min()-35);y1=min(1152,yy.max()+36);bbox=(x0,y0,x1,y1)
 rgb,m,_=r.render(T[i][None],bbox,K,Tcw);mask=m[0];im=frames[i][y0:y1,x0:x1].copy();im[mask]=(0.25*im[mask]+0.75*rgb[0][mask][:,::-1]*255).clip(0,255).astype(np.uint8);im=cv2.resize(im,(240,180),interpolation=cv2.INTER_CUBIC);cv2.putText(im,str(i),(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);ims.append(im)
cols=5;rows=(len(ims)+cols-1)//cols;sheet=np.zeros((rows*180,cols*240,3),np.uint8)
for k,im in enumerate(ims):sheet[(k//cols)*180:(k//cols+1)*180,(k%cols)*240:(k%cols+1)*240]=im
cv2.imwrite('/work/scratch/analysis/final_overlay.jpg',sheet)
PY
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MK …[+63 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '48,90p' /work/outputs/REPORT.md; sed -n '1,45p' /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
## Validation without ground truth

Visible silhouette IoU excludes person-mask pixels and compares the rendered
mesh with the SAM2 object mask.  Values are mean / median; higher is better.

| Phase | Frames | Baseline IoU | Result IoU | Delta |
|---|---:|---:|---:|---:|
| Initial rest on floor | 0-152 | 0.760 / 0.764 | 0.892 / 0.898 | +0.133 |
| Pickup / carried | 153-437 | 0.607 / 0.640 | 0.606 / 0.617 | -0.001 |
| Placement transition | 438-449 | 0.851 / 0.850 | 0.878 / 0.871 | +0.026 |
| Final rest on desk | 450-564 | 0.856 / 0.856 | 0.937 / 0.938 | +0.081 |

Pooled-pose texture evidence: initial NCC `0.173` with mask
IoU `0.901` at yaw offset `-5.0` degrees;
final NCC `0.119` with mask IoU `0.938`
at yaw offset `-25.0` degrees.

Dynamic-interval temporal diagnostics (frames 153-449):
 …[+3119 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'h1=$(sha256sum /work/outputs/object_poses_world.npz | cut -d'"' ' -f1); OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz >/work/scratch/final_solve_stdout.txt; h2="'$(sha256sum /work/outputs/object_poses_world.npz | cut -d'"' ' -f1); echo \"before="'$h1"; echo "after =$h2"; cat /work/scratch/final_solve_stdout.txt'
```
<details><summary>output</summary>

```
before=ac7fb4bfac44dcc232991300f62b22b7615a0b700560dbab992ace2cd3be7e2b
after =ac7fb4bfac44dcc232991300f62b22b7615a0b700560dbab992ace2cd3be7e2b
Wrote /work/outputs/object_poses_world.npz
Wrote /work/outputs/REPORT.md
Runtime: 9.9s
Mask IoU mean baseline/result: 0.7044 / 0.7569

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/outputs -maxdepth 2 -printf '%y %p %s\\n' | sort; python - <<'PY'
import numpy as np
z=np.load('/work/outputs/object_poses_world.npz');T=z['T_world_mesh'];print(T.shape,T.dtype,T.flags.c_contiguous,np.isfinite(T).all())
PY"
```
<details><summary>output</summary>

```
d /work/outputs 4096
d /work/outputs/__pycache__ 4096
f /work/outputs/REPORT.md 4781
f /work/outputs/TOOLS.md 1903
f /work/outputs/__pycache__/solve.cpython-311.pyc 47159
f /work/outputs/object_poses_world.npz 72598
f /work/outputs/solve.py 31047
(565, 4, 4) float64 True True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
p=Path('/work/outputs/__pycache__')
if p.exists():
 for f in p.glob('*'): f.unlink()
 p.rmdir()
PY
find /work/outputs -maxdepth 1 -type f -printf '%f %s\\n' | sort"
```
<details><summary>output</summary>

```
REPORT.md 4781
TOOLS.md 1903
object_poses_world.npz 72598
solve.py 31047

```
</details>

**Agent:** Completed and wrote all required outputs:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(565, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — tools, versions, revisions, and weight hashes

Method highlights:
- Pooled single poses for the initial floor-rest and final desk-rest phases using temporal median texture plus silhouette/depth alignment.
- Robust temporal smoothing and endpoint blending through the carried interval.
- Excluded person-occluded pixels from validation and retained the baseline symmetry branch where textured silhouette scoring supported it.

Validation:
- Full-video visible-mask IoU: `0.704 → 0.757`
- Initial static phase IoU: `0.760 → 0.892`
- Final static phase IoU: `0.856 → 0.937`
- Dynamic rotation max step: `122.77° → 29.06°`
- Dynamic translation max step: `0.139 m → 0.036 m`

The required command was run successfully, and a repeat run produced an identical NPZ hash. Runtime is approximately 10 seconds.
