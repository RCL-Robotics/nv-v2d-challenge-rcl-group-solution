# T1_ds_ep004

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T15:44:53Z

**shell** (exit 0)
```
/usr/bin/bash -c 'cat AGENT_TASK.md'
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
/usr/bin/bash -c "sed -n '1,180p' AGENT_TASK.md"
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
/usr/bin/bash -c "find inputs -maxdepth 2 -type f -printf '%p %s bytes\\n' | sort; printf '\\noutputs:\\n'; find outputs -maxdepth 2 -type f -printf '%p %s bytes\\n' 2>/dev/null | sort; printf '\\ntools:\\n'; find tools -maxdepth 2 -type f -printf '%p %s bytes\\n' 2>/dev/null | head -50"
```
<details><summary>output</summary>

```
inputs/camera.json 821 bytes
inputs/episode.json 66 bytes
inputs/frame0_reference_human.npz 191558 bytes
inputs/ground_plane.json 399 bytes
inputs/masks.h5 6890314 bytes
inputs/object_mesh.glb 10231524 bytes
inputs/reconstruction.npz 40763188 bytes
inputs/task.txt 195 bytes
inputs/video.mp4 23743749 bytes

outputs:

tools:

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; python - <<'PY'
import sys, torch, cv2, trimesh, numpy as np
print(sys.version)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('cv2', cv2.__version__, 'trimesh', trimesh.__version__, 'numpy', np.__version__)
try: import nvdiffrast; print('nvdiffrast ok')
except Exception as e: print('nvdiffrast err',e)
PY"
```
<details><summary>output</summary>

```
3.11.10 | packaged by conda-forge | (main, Oct 16 2024, 01:27:36) [GCC 13.3.0]
torch 2.5.1+cu124 cuda True
cv2 4.11.0 trimesh 5.1.1 numpy 1.26.3
nvdiffrast ok

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat inputs/task.txt; echo; cat inputs/episode.json; echo; cat inputs/camera.json; echo; cat inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk over to the baseball bat and pick up the baseball bat. Slowly begin to rotate the baseball bat and yourself around the bounding area so that every camera gets a visual of the baseball baat.

{
 "episode_index": 4,
 "object": "baseball_bat",
 "frames": 803
}
{
 "K": [
  [
   819.99951171875,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.99951171875,
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
   824.2138671875,
   0.0,
   768.0
  ],
  [
   0.0,
   824.2138671875,
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
   0.9986179086582012,
   0.013094287756622688,
   0.05090002097511233,
   0.016776686362482217
  ],
  [
   -0.014272360516026957,
   0.9996370071498452,
   0.022850681867306558,
   0.020542220507940544
  ],
  [
   -0.050582 …[+684 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json, trimesh
r=np.load('inputs/reconstruction.npz')
print('recon keys',r.files)
for k in r.files:
 a=r[k]
 print(k, a.shape,a.dtype, (float(np.nanmin(a)),float(np.nanmax(a))) if np.issubdtype(a.dtype,np.number) and a.size else '')
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False)
print('scene geom',list(g.geometry.keys()))
for k,geom in g.geometry.items():
 print(k, 'verts',len(geom.vertices),'faces',len(geom.faces),'bounds',geom.bounds,'visual',type(geom.visual).__name__)
PY"
```
<details><summary>output</summary>

```
recon keys ['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (803, 4, 4) float32 (-0.9999935626983643, 3.8878591060638428)
T_cam_mesh_foundationpose (803, 4, 4) float32 (-0.9999768137931824, 5.258673191070557)
T_cam_mesh_coconet (803, 4, 4) float32 (-0.9994381070137024, 3.7507007122039795)
human_vertices_world (803, 4610, 3) float32 (-1.0907686948776245, 3.8420968055725098)
human_keypoints_world (803, 70, 3) float32 (-1.0415112972259521, 3.8355205059051514)
contact_prob_right_left (803, 2) float32 (2.389682138165362e-19, 1.0)
scene geom ['baseballbat']
baseballbat verts 296472 faces 98866 bounds [[-0.0440244  -0.04340127 -0.20978296]
 [ 0.04483674  0.04511044  0.33304057]] visual Text …[+11 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, os
cap=cv2.VideoCapture('inputs/video.mp4')
print('frames',cap.get(cv2.CAP_PROP_FRAME_COUNT),'fps',cap.get(cv2.CAP_PROP_FPS),'w',cap.get(cv2.CAP_PROP_FRAME_WIDTH),'h',cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
os.makedirs('scratch/frames',exist_ok=True)
n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs=[round(i*(n-1)/31) for i in range(32)]
for j,i in enumerate(idxs):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 if ok: cv2.imwrite(f'scratch/frames/{j:02d}_{i:04d}.jpg',im)
print('saved',idxs)
PY"
```
<details><summary>output</summary>

```
frames 803.0 fps 30.0 w 1536.0 h 1152.0
saved [0, 26, 52, 78, 103, 129, 155, 181, 207, 233, 259, 285, 310, 336, 362, 388, 414, 440, 466, 492, 517, 543, 569, 595, 621, 647, 673, 699, 724, 750, 776, 802]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image,ImageDraw
import glob,os
fs=sorted(glob.glob('scratch/frames/*.jpg'))
thumbs=[]
for f in fs:
 im=Image.open(f); im.thumbnail((384,288));
 can=Image.new('RGB',(384,312),'white'); can.paste(im,(0,24));
 ImageDraw.Draw(can).text((5,5),os.path.basename(f),fill='black')
 thumbs.append(can)
out=Image.new('RGB',(384*4,312*8),'white')
for i,t in enumerate(thumbs): out.paste(t,((i%4)*384,(i//4)*312))
out.save('scratch/montage.jpg',quality=90)
print(out.size)
PY"
```
<details><summary>output</summary>

```
(1536, 2496)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R
r=np.load('inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(float); F=len(T)
p=T[:,:3,3]
print('pos xyz stats',np.min(p,0),np.max(p,0),np.mean(p,0),np.std(p,0))
dp=np.linalg.norm(np.diff(p,axis=0),axis=1)
print('step translation quantiles',np.quantile(dp,[0,.1,.25,.5,.75,.9,.95,.99,1]))
# relative rotations
rel=R.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
print('step rot deg quantiles',np.quantile(rel,[0,.1,.25,.5,.75,.9,.95,.99,1]))
print('max trans frames',np.argsort(dp)[-20:][::-1], dp[np.argsort(dp)[-20:][::-1]])
print('max rot frames',np.argsort(rel)[-20:][::-1], rel[np.argsort(rel)[-20:][::-1]])
# selected
for i in [0,25,50,75,100,110,120,130,150,200,250,300,350,400,450,500,550,600,650,700,725,750,775,802]:
 print(i, 'p',np.round(p[i],3),'step',round(dp[min(i,F-2)],3),'rotstep',round(rel[min(i,F-2)],2),'contact',np.round(r['contact_prob_right_left'][i],3))
with h5py.File('inputs/masks.h5','r') as h:
 print('mask keys',list(h.keys()))
 for k in h.keys(): print(k,h[k].shape,h[k].dtype)
PY"
```
<details><summary>output</summary>

```
pos xyz stats [-0.41500071 -0.38334101  2.18644047] [1.08199954 1.27334893 3.88785911] [0.31515242 0.27669138 2.87945025] [0.35704493 0.67460692 0.44421926]
step translation quantiles [0.         0.00016792 0.00503373 0.01821648 0.03599913 0.05626993
 0.0781163  0.11324453 0.15988635]
step rot deg quantiles [3.70241259e-15 3.70241259e-15 1.21726529e-03 6.27861853e+00
 1.30171676e+01 2.49387417e+01 2.99573536e+01 7.13357542e+01
 1.63250874e+02]
max trans frames [484 489 331 671 303 488 265 491 459 450 658 313 542 545 362 314 232 441
 226 302] [0.15988635 0.14576144 0.14299427 0.12800171 0.12799032 0.12792255
 0.1223     0.11972417 0.11325697 0.11201356 0.11120579 0.10659609
 0.10648291 0.10480261 0.10031385 0.09977799 0.09911013 0.09642635
 0.09625813 0.09485571]
max rot frames [331 744 566 …[+2053 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p scratch/overlay && python - <<'PY'
import numpy as np, cv2, trimesh, json, h5py, os
from PIL import Image,ImageDraw
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K']); Tc=np.array(cam['T_world_cam'])
scene=trimesh.load('inputs/object_mesh.glb',force='scene',process=False)
g=scene.to_geometry(); V=g.vertices; Fc=g.faces
# subsample faces for overlay? all
cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:
 om=h['object'][:]; pm=h['person'][:]
idxs=list(range(0,803,20))
thumbs=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 if not ok: continue
 Twm=T[i]; Tcm=Tc@np.linalg.inv(np.eye(4))  # identity
 P=Tc@Twm
 vw=(P[:3,:3]@V.T+P[:3,3:4]).T
 img=v w # no
PY"
```
<details><summary>output</summary>

```
  File "<stdin>", line 19
    img=v w # no
          ^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, cv2, trimesh, json, h5py, os
from PIL import Image,ImageDraw
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
scene=trimesh.load('inputs/object_mesh.glb',force='scene',process=False); g=scene.to_geometry(); V=g.vertices
cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h: om=h['object'][:]; pm=h['person'][:]
os.makedirs('scratch/overlay',exist_ok=True)
idxs=list(range(0,803,20))
thumbs=[]
for i in idxs:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 if not ok: continue
 P=Tcw@T[i]
 vc=(P[:3,:3]@V.T+P[:3,3:4]).T
 z=vc[:,2]; valid=z>0.01
 uv=(K@vc[valid].T).T; uv=uv[:,:2]/uv[:,2:3]
 hull=cv2.convexHull(uv.astype(np.float32))
 over=im.copy(); cv2.polylines(over,[hull.astype(np.int32)],True,(0,0,255),3)
 # axis lines centered at origin projected
 for j,col,lab in [(0,(0,0,255),'X'),(1,(0,255,0),'Y'),(2,(255,0,0),'Z')]:
  pts=np.array([[0,0,0],[V[:,j].min()*1.0 if j==2 else 0,0,0]])
  # from origin outward positive .25m
  pts=np.array([[0,0,0],[0,0,0]],float); pts[1,j]=0.25
  pc=(P[:3,:3]@pts.T+P[:3,3:4]).T
  uv2=(K@pc.T).T; uv2=uv2[:,:2]/uv2[:,2:3]
  cv2.line(over,tuple(np.round(uv2[0]).astype(int)),tuple(np.round(uv2[1]).astype(int)),col,3)
 # masks contours
 cnts,_=cv2.findContours(om[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SI …[+875 chars]
```
<details><summary>output</summary>

```
(2048, 4224) 41

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float); F=len(T)
Twc=np.array(json.load(open('inputs/camera.json'))['T_world_cam'])
for name in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 A=r[name].astype(float)
 # baseline camera-fr rot
 Rb=Twc[:3,:3].T@T[:,:3,:3]
 d=Rot.from_matrix(Rb.transpose(0,2,1)@A[:,:3,:3]).magnitude()*180/np.pi
 print(name,'diff from baseline rot deg quantiles',np.quantile(d,[0,.1,.25,.5,.75,.9,.95,.99,1]))
 # temporal rot
 rel=Rot.from_matrix(A[:-1,:3,:3].transpose(0,2,1)@A[1:,:3,:3]).magnitude()*180/np.pi
 print(' temporal rot',np.quantile(rel,[0,.25,.5,.75,.9,.95,.99,1]), 'max',np.argmax(rel),rel.max())
# baseline vs coconet diff around event
A=r['T_cam_mesh_coconet'].astype(float); Rb=Twc[:3,:3].T@T[:,:3,:3]
d=Rot.from_matrix(Rb.transpose(0,2,1)@A[:,:3,:3]).magnitude()*180/np.pi
rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
for i in sorted(set(list(np.argsort(rel)[-30:])+list(np.argsort(d)[-20:]))): print(i,'rel',rel[i] if i<F-1 else None,'diff',d[i])
PY"
```
<details><summary>output</summary>

```
T_cam_mesh_foundationpose diff from baseline rot deg quantiles [  0.69971858   2.61313222   5.0443931   16.63849268  55.98022883
 115.56860657 142.70752575 168.90764387 178.82238905]
 temporal rot [  0.23212105   6.80551628  11.01090964  17.10808331  25.71667565
  33.25124029  64.07708375 175.0510318 ] max 453 175.05103179893067
T_cam_mesh_coconet diff from baseline rot deg quantiles [1.16236558 1.17836752 1.19681327 1.23005183 1.25325082 1.26184461
 1.27552907 1.2957056  1.29804098]
 temporal rot [1.32246150e-15 7.77350684e-15 6.27849302e+00 1.30157614e+01
 2.49439281e+01 2.99571696e+01 7.13277176e+01 1.63242051e+02] max 331 163.24205089723912
98 rel 44.20934229930228 diff 1.1760136497571687
148 rel 6.823789939181698 diff 1.2918931029441953
149 rel 7.702367394057568 diff 1.293813834766488 …[+2339 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/make_event_montage.py <<'PY'
import numpy as np, cv2, trimesh, json, h5py, os, sys
from PIL import Image,ImageDraw
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
scene=trimesh.load('inputs/object_mesh.glb',force='scene',process=False); g=scene.to_geometry(); V=g.vertices
cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h: om=h['object'][:]; pm=h['person'][:]
def frame(i):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 if not ok: return None
 # crop centered around object mask centroid; fall back projection
 ys,xs=np.where(om[i])
 cx,cy=(xs.mean(),ys.mean()) if len(xs)>0 else (im.shape[1]/2,im.shape[0]/2)
 # crop dimension control
 half=300
 x0=int(np.clip(cx-half,0,im.shape[1]-2*half)); y0=int(np.clip(cy-half,0,im.shape[0]-2*half))
 crop=im[y0:y0+2*half,x0:x0+2*half].copy()
 P=Tcw@T[i]
 vc=(P[:3,:3]@V.T+P[:3,3:4]).T; z=vc[:,2]; valid=z>0.01
 uv=(K@vc[valid].T).T; uv=uv[:,:2]/uv[:,2:3]; hull=cv2.convexHull(uv.astype(np.float32)); hull-=np.array([x0,y0],np.float32)
 if len(hull): cv2.polylines(crop,[hull.astype(np.int32)],True,(0,0,255),2)
 # bbox axes center
 for j,col in [(0,(0,0,255)),(1,(0,255,0)),(2,(255,0,0))]:
  pts=np.array([[0,0,0],[0,0,0]],float); pts[1,j]=0.2
  pc=(P[:3,:3]@pts.T+P[:3,3:4]).T; u=(K@pc.T).T; u=u[:,:2]/u[:,2:3]; u-=np.array([x0,y0]);  …[+1072 chars]
```
<details><summary>output</summary>

```
160 210 [160, 163, 166, 169, 172, 175, 178, 181, 184, 187, 190, 193, 196, 199, 202, 205]
250 340 [250, 256, 262, 268, 274, 280, 286, 292, 298, 304, 310, 316, 322, 328, 334, 340]
340 375 [340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370]
400 425 [400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415]
440 502 [440, 444, 448, 452, 456, 460, 464, 468, 472, 476, 480, 484, 488, 492, 496, 500]
550 575 [550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565]
590 612 [590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605]
640 675 [640, 642, 644, 646, 648, 650, 652, 654, 656, 658, 660, 662, 664, 666, 668, 670]
700 755 [700, 703, 706, 709, 712, 715, 718, 721, 724, 727, 730, 733, 736, 739, 742, 745] …[+1 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
s=trimesh.load('inputs/object_mesh.glb',force='scene',process=False); g=s.to_geometry()
print(type(g.visual)); print(g.visual)
print('uv shape',None if g.visual.uv is None else g.visual.uv.shape)
print('material',g.visual.material)
mat=g.visual.material
for name in ['baseColorTexture','baseColorFactor','metallicFactor','roughnessFactor','image','baseColorTexture']:
 if hasattr(mat,name):
  v=getattr(mat,name); print(name, type(v), getattr(v,'shape',None), getattr(v,'size',None))
# save texture
if hasattr(mat,'baseColorTexture') and mat.baseColorTexture is not None:
 from PIL import Image
 im=mat.baseColorTexture; print(type(im),im.size,im.mode); im.save('scratch/basecolor.png')
# vertex/face colors
print('vertex normals',g.vertex_normals.shape if hasattr(g,'vertex_normals') else None)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'>
<trimesh.visual.texture.TextureVisuals object at 0x721e17f253d0>
uv shape (296472, 2)
material <trimesh.visual.material.PBRMaterial object at 0x721e6292db90>
baseColorTexture <class 'PIL.Image.Image'> None (3122, 2073)
baseColorFactor <class 'NoneType'> None None
metallicFactor <class 'float'> None None
roughnessFactor <class 'NoneType'> None None
baseColorTexture <class 'PIL.Image.Image'> None (3122, 2073)
<class 'PIL.Image.Image'> (3122, 2073) RGB
vertex normals (296472, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100; echo '--- dirs'; find /pipeline -maxdepth 3 -type d | head -100"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1963
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1772
/pipeline/.stages/06_coconet.json 2828
/pipeline/.stages/07_refinement.json 2131
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/depth_report.json 1810
/pipeline/depth/moge2/aligned.h5 993184084
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 85525
/pipeline/depth/moge2/raw.npy 2841772160
/pipeline/export/episode_000004/edex 557
/pipeline/export/episode_000004/wild_export.json 2791
/pipeline/foundationpose/episode_000004.pkl 229474
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 6141719 …[+1021 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/pipeline_report.json; echo '---06'; cat /pipeline/.stages/06_coconet.json; echo '---07'; cat /pipeline/.stages/07_refinement.json; echo '---08'; cat /pipeline/.stages/08_render.json"
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
    "mtime_ns": 1791253967506614456,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000004/inference/episode_000004_step200000_before_after.mp4",
    "size": 57830883
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+12619 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline/foundationpose/debug -maxdepth 2 -type f -printf '%p %s\\n' | head -100; find /pipeline/work/masks -maxdepth 2 -type f -printf '%p %s\\n' | head -50; find /pipeline/export/episode_000004 -maxdepth 2 -type f -printf '%p %s\\n' | head -50"
```
<details><summary>output</summary>

```
/pipeline/work/masks/0/000010.png 3468
/pipeline/work/masks/0/000015.png 3492
/pipeline/work/masks/0/000016.png 3494
/pipeline/work/masks/0/000024.png 3478
/pipeline/work/masks/0/000026.png 3487
/pipeline/work/masks/0/000038.png 3532
/pipeline/work/masks/0/000044.png 3723
/pipeline/work/masks/0/000049.png 3811
/pipeline/work/masks/0/000079.png 4000
/pipeline/work/masks/0/000084.png 3947
/pipeline/work/masks/0/000092.png 4063
/pipeline/work/masks/0/000094.png 4073
/pipeline/work/masks/0/000115.png 3731
/pipeline/work/masks/0/000131.png 3817
/pipeline/work/masks/0/000148.png 4348
/pipeline/work/masks/0/000154.png 4370
/pipeline/work/masks/0/000159.png 4246
/pipeline/work/masks/0/000160.png 4273
/pipeline/work/masks/0/000168.png 4152
/pipeline/work/masks/0/000169.png 4052
/pipeline/work/masks …[+1838 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/inference/materialized_inputs.h5','/pipeline/work/masks_k0.h5','/pipeline/depth/moge2/aligned.h5']:
 print('\\n',p)
 try:
  with h5py.File(p,'r') as h:
   def visit(name,obj):
    if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype)
   h.visititems(visit)
 except Exception as e: print('ERR',e)
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/materialized_inputs.h5
K_rois (803, 3, 3) float32
bboxes (803, 4) float32
identity_json () object
input_rgbs (803, 3, 224, 224) float32
input_xyz (803, 6, 224, 224) float32
observed_human_masks (803, 224, 224) bool
observed_object_masks (803, 224, 224) bool
render_rgbs (803, 3, 224, 224) float32
render_xyz (803, 6, 224, 224) float32

 /pipeline/work/masks_k0.h5
episode_000004/000000-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000004/000000-k0.person_mask.png (1152, 1536) uint8
episode_000004/000001-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000004/000001-k0.person_mask.png (1152, 1536) uint8
episode_000004/000002-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000004/000002-k0.person_mask.png (1152, 1536) uint8
episode_000004/000003-k0.obj_rend_mask.png (1152 …[+97900 chars]
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
 if isinstance(d,dict):
  for k,v in d.items():
   print(k,type(v), getattr(v,'shape',None), getattr(v,'dtype',None))
   if isinstance(v,dict): print(' subkeys',list(v.keys())[:50])
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
<class 'dict'>
schema <class 'str'> None None
frames <class 'list'> None None
frame_meta <class 'list'> None None
kid <class 'int'> None None
checkpoint <class 'dict'> None None
 subkeys ['path', 'step']
mhr_inference <class 'dict'> None None
 subkeys ['schema', 'checkpoint', 'wandbRun', 'wandbConfigSha256', 'supervisionContract', 'configSource', 'xyzAnchorContract', 'spatialNormalizationContract']
config <class 'dict'> None None
 subkeys ['rgb_files', 'render_files', 'split_file', 'val_split_file', 'render_root', 'rgb_root', 'packed_root', 'packed_format', 'expected_train_sequence_count', 'expected_val_sequence_count', 'fp_root', 'contacts_root', 'body_model', 'mhr_gt_root', 'mhr_init_root', 'mhr_pseudogt_root', 'mhr_sample_kids', 'mhr_dataset_index_path' …[+6960 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch, numpy as np, json, trimesh
p='/pipeline/inference/refined.pth'; d=torch.load(p,map_location='cpu',weights_only=False)
print('metadata')
for k,v in d['metadata'].items():
 print(k, str(v)[:2000])
print('frames head',d['frames'][:5], 'len',len(d['frames']))
print('frame_meta[0]',d['frame_meta'][0]); print('frame_meta[-1]',d['frame_meta'][-1])
for grp in ['pr','pr_initial','in','raw','postopt']:
 print('\\nGRP',grp)
 def rec(x,pre=''):
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,dict): rec(v,pre+k+'.')
    else:
     if hasattr(v,'shape'): print(pre+k, tuple(v.shape), v.dtype, ('minmax',float(v.min()),float(v.max())) if v.numel() and np.issubdtype(np.asarray(v).dtype,np.number) else '')
     else: print(pre+k,type(v),str(v)[:200])
 rec(d[grp])
PY"
```
<details><summary>output</summary>

```
metadata
ground_truth_used False
window_length 96
window_stride 96
overlap_policy first_occurrence
depth_source /home/hanzhou/Projects/v2d/run/out/episode_000004/depth/moge2/aligned.h5
depth_backend moge2
depth_alignment_method huber-log-ratio-to-direct-mhr-depth-v1
mhr_init_source /home/hanzhou/Projects/v2d/run/out/episode_000004/mhr_init/episode_000004.pkl
foundationpose_source /home/hanzhou/Projects/v2d/run/out/episode_000004/foundationpose/episode_000004.pkl
object_mesh /home/hanzhou/Projects/v2d/run/out/episode_000004/export/episode_000004/object_mesh/output_aligned.glb
mhr_xyz_anchor_contract {'schema': 'cari4d.mhr_xyz_anchor_contract.v1', 'anchorType': 'root_joint_1', 'rootJointIndex': 1}
mhr_spatial_normalization_contract {'schema': 'cari4d.mhr_spatial_normalization_contract.v1', ' …[+1248 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch, numpy as np
D=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for grp in ['pr','pr_initial','in','raw','postopt']:
 print('\\nGRP',grp)
 def rec(x,pre=''):
  if isinstance(x,dict):
   for k,v in x.items():
    if isinstance(v,dict): rec(v,pre+k+'.')
    elif isinstance(v,np.ndarray):
     print(pre+k, tuple(v.shape), v.dtype, ('range',float(np.nanmin(v)),float(np.nanmax(v))) if v.size and np.issubdtype(v.dtype,np.number) else '')
    elif torch.is_tensor(v): print(pre+k,tuple(v.shape),v.dtype,float(v.min()),float(v.max()))
    else: print(pre+k,type(v),str(v)[:300])
 rec(D[grp])
PY"
```
<details><summary>output</summary>

```

GRP pr
pose_abs (803, 4, 4) float32 ('range', -0.9991697669029236, 4.654648303985596)
pose_abs_1st_delta (803, 4, 4) float32 ('range', -0.9991697669029236, 4.783164978027344)
mhr_global_rot6d (803, 6) float32 ('range', -0.9978426694869995, 0.9998442530632019)
mhr_trans (803, 3) float32 ('range', -1.1648606061935425, 3.5060741901397705)
mhr_body_pose_cont (803, 260) float32 ('range', -0.9273346066474915, 1.1541612148284912)
mhr_hand (803, 108) float32 ('range', -0.6139984726905823, 0.5179269909858704)
mhr_shape (803, 45) float32 ('range', -2.148430824279785, 1.4662697315216064)
mhr_scale (803, 28) float32 ('range', -2.5311648845672607, 3.2421317100524902)
mhr_face (803, 72) float32 ('range', 0.0, 0.0)
contact_logits (803, 2) float32 ('range', -35.63502883911133, 28.214338302612305)
pose_ab …[+7553 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as Rot
D=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float); F=len(T)
Twc=np.array(json.load(open('inputs/camera.json'))['T_world_cam']); Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); S,to_orig=trimesh.bounds.oriented_bounds(g)
print('S',S); print('to_orig',to_orig)
# compare pose_abs @ S with T_cam from baseline
for name in ['pose_abs','pose_abs_postopt','pose_abs_1st_delta']:
 A=D['pr'][name].astype(float)
 Pm=A@S
 Rcam=Tcw[:3,:3]@T[:,:3,:3]
 d=Rot.from_matrix(Rcam.transpose(0,2,1)@Pm[:,:3,:3]).magnitude()*180/np.pi
 # transl comparison camera
 tc=(Tcw@T)[:,:3,3]
 dt=np.linalg.norm(Pm[:,:3,3]-tc,axis=1)
 print(name,'rot diff',np.quantile(d,[0,.5,.9,.99,1]),'trans diff',np.quantile(dt,[0,.5,.9,.99,1]))
# raw pose_abs compare baseline rotation
A=D['pr']['pose_abs'].astype(float); Rcam=Tcw[:3,:3]@T[:,:3,:3]
d=Rot.from_matrix(Rcam.transpose(0,2,1)@(A@S)[:,:3,:3]).magnitude()*180/np.pi
print('pose_abs@S rot diff 331 etc', [(i,d[i]) for i in [0,100,150,250,330,331,332,350,740,744,745,800]])
# translation scaling relation
P=D['pr']['pose_abs_postopt'].astype(float)@S
tc=(Tcw@T)[:,:3,3]
for i in [0,50,100,150,250,300,331,400,500,600,700,744,800]: print(i,'pose',np.round(P[i,:3,3],3),'base cam',np.round …[+14 chars]
```
<details><summary>output</summary>

```
S [[-9.23898016e-01 -3.82599038e-01 -5.51660185e-03  1.48312194e-04]
 [-3.82637736e-01  9.23833526e-01  1.09534825e-02  6.09123954e-04]
 [ 9.05629888e-04  1.22307607e-02 -9.99924791e-01  6.16254278e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
to_orig [0.08840402 0.08884824 0.54285241]
pose_abs rot diff [  1.15976476 111.40232647 168.60659661 178.73574269 179.93060303] trans diff [0.03559024 0.31002308 0.99735753 1.34960769 1.67412176]
pose_abs_postopt rot diff [  1.15976476 111.40232647 168.60659661 178.73574269 179.93060303] trans diff [0.03559024 0.31002308 0.99735753 1.34960769 1.67412176]
pose_abs_1st_delta rot diff [  1.15976476 111.40232647 168.60659661 178.73574269 179.93060303] trans diff [0.05698014 0.30674383 0.96106289 1.39228797 1.87012392]
pose_abs@ …[+1134 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as Rot
D=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
Twc=np.array(json.load(open('inputs/camera.json'))['T_world_cam']); Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); S,to=trimesh.bounds.oriented_bounds(g)
Rcam=Tcw[:3,:3]@T[:,:3,:3]
for name,expr in [('A',D['pr']['pose_abs']),('A@S',D['pr']['pose_abs']@S),('A@invS',D['pr']['pose_abs']@np.linalg.inv(S)),('S@A',S@D['pr']['pose_abs'])]:
 A=np.asarray(expr)
 d=Rot.from_matrix(Rcam.transpose(0,2,1)@A[:,:3,:3]).magnitude()*180/np.pi
 print(name,np.quantile(d,[0,.25,.5,.75,.9,.99,1]))
for i in [0,50,100,150,250,331,500,700,744,800]:
 print(i,'A',np.round(D['pr']['pose_abs'][i,:3,3],3),'A@S',np.round((D['pr']['pose_abs'][i]@S)[:3,3],3))
PY"
```
<details><summary>output</summary>

```
A [ 35.39722224 122.41588165 163.86759731 179.2266502  179.67761532
 179.94365551 179.99910451]
A@S [  1.15976476  45.33962648 111.40232647 149.98270535 168.60659661
 178.73574269 179.93060303]
A@invS [  0.89950612  45.40971645 111.52620682 149.9133836  168.34693519
 178.80408866 179.87698129]
S@A [ 11.49802131  89.61328288 122.69363743 143.46560523 160.87711628
 176.77081518 179.50950183]
0 A [-0.031  1.098  2.419] A@S [0.03  1.094 2.412]
50 A [-0.029  1.087  2.394] A@S [0.032 1.084 2.388]
100 A [-0.022  1.093  2.402] A@S [0.039 1.09  2.393]
150 A [-0.066  0.289  2.383] A@S [-0.007  0.306  2.391]
250 A [ 0.514 -0.168  2.504] A@S [ 0.487 -0.221  2.487]
331 A [ 0.708 -0.404  3.3  ] A@S [ 0.767 -0.42   3.309]
500 A [-0.314 -0.649  3.615] A@S [-0.306 -0.704  3.641]
700 A [-0.529  0.67   3.162 …[+141 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,os
p='/pipeline/inference/materialized_inputs.h5'
with h5py.File(p,'r') as h:
 rgb=h['input_rgbs'][:]; ren=h['render_rgbs'][:]; om=h['observed_object_masks'][:]; pm=h['observed_human_masks'][:]
 bbox=h['bboxes'][:]; Kr=h['K_rois'][:]
print(rgb.min(),rgb.max(),ren.min(),ren.max(),om.mean(),pm.mean())
print('bbox',bbox[:3]); print('Kr',Kr[0])
os.makedirs('scratch/mat',exist_ok=True)
idxs=[0,50,100,130,160,200,250,300,331,350,400,450,500,550,600,650,700,744,780,802]
rows=[]
for i in idxs:
 a=(np.transpose(rgb[i],(1,2,0))*255).astype(np.uint8)[:,:,::-1]
 b=(np.transpose(ren[i],(1,2,0))*255).astype(np.uint8)[:,:,::-1]
 # mask overlay
 bb=b.copy(); m=om[i]; bb[m]=(0.5*bb[m]+0.5*np.array([0,0,255])).astype(np.uint8)
 row=np.concatenate([a,b,bb],1)
 cv2.putText(row,str(i),(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
 rows.append(row)
out=np.concatenate(rows,0); cv2.imwrite('scratch/mat/montage.jpg',out)
print(out.shape)
PY"
```
<details><summary>output</summary>

```
0.0 1.0 0.0 1.0 0.009258667274506315 0.14754160001874347
bbox [[ 330.55  387.05  945.45 1001.95]
 [ 331.6   387.1   945.4  1000.9 ]
 [ 331.6   387.1   945.4  1000.9 ]]
Kr [[300.25027   0.      159.3573 ]
 [  0.      300.25027  68.83201]
 [  0.        0.        1.     ]]
(4480, 672, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float); R=T[:,:3,:3]
for i in [98,148,151,155,169,195,202,262,265,294,296,319,324,331,356,360,411,447,456,458,469,473,476,492,557,566,601,604,648,658,714,744]:
 rt=R[i].T@R[i+1]; rv=Rot.from_matrix(rt); ax=rv.as_rotvec(); ang=np.linalg.norm(ax)*180/np.pi
 zaxis=R[i][:,2]
 print(f'{i:3d} {i+1:3d} ang={ang:7.2f} axis_local={ax/(np.linalg.norm(ax)+1e-12)} z_world={zaxis} z_dot_axis={zaxis@(ax/(np.linalg.norm(ax)+1e-12)):.3f}')
PY"
```
<details><summary>output</summary>

```
 98  99 ang=  44.21 axis_local=[-0.02340876 -0.0123543   0.99964964] z_world=[-0.96994507  0.08040836  0.22965361] z_dot_axis=0.251
148 149 ang=   6.82 axis_local=[0.15557549 0.63799054 0.75416467] z_world=[-0.92384285 -0.37781432 -0.06140594] z_dot_axis=-0.431
151 152 ang=  12.29 axis_local=[ 0.3337406   0.05883448 -0.94082714] z_world=[-0.97454703 -0.20812561 -0.0833137 ] z_dot_axis=-0.259
155 156 ang=  19.75 axis_local=[-0.23674293  0.01852853  0.97139564] z_world=[-0.99399334 -0.0891519  -0.06347003] z_dot_axis=0.172
169 170 ang=  52.56 axis_local=[-0.40212701 -0.59994995  0.69163135] z_world=[-0.66176331 -0.74195397 -0.10757595] z_dot_axis=0.637
195 196 ang=  70.07 axis_local=[-0.93925411  0.28648443 -0.18901953] z_world=[0.8325274  0.49662444 0.24548166] z_dot_axis=-0.686
202 203 ang …[+3413 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float); R=T[:,:3,:3]
for i in [97,98,99,100,264,265,266,330,331,332,333,565,566,567,713,714,715,716,743,744,745,746]:
 q=Rot.from_matrix(R[i]); print(i,'euler xyz',np.round(q.as_euler('xyz',degrees=True),1),'z',np.round(R[i,:,2],3))
PY"
```
<details><summary>output</summary>

```
97 euler xyz [-78.5 -11.3  83.5] z [-0.978  0.071  0.196]
98 euler xyz [-75.1 -26.4  78.5] z [-0.97  0.08  0.23]
99 euler xyz [-77.2  16.6  89.4] z [-0.975  0.073  0.212]
100 euler xyz [-77.7  21.1  90.8] z [-0.978  0.064  0.199]
264 euler xyz [-123.4   58.8  -34.4] z [ 0.083  0.955 -0.285]
265 euler xyz [-123.4   58.9  -34.4] z [ 0.082  0.955 -0.285]
266 euler xyz [149.  -18.8  26.9] z [ 0.48  -0.335 -0.811]
330 euler xyz [ 88.   31.5 -20.2] z [-0.329 -0.944  0.029]
331 euler xyz [ 88.   31.5 -20.2] z [-0.329 -0.944  0.029]
332 euler xyz [-123.1    3.3  -60.3] z [ 0.712  0.443 -0.545]
333 euler xyz [-122.9   -2.3  -60.5] z [ 0.741  0.395 -0.543]
565 euler xyz [-152.   -75.4  -41.8] z [ 0.95  -0.22  -0.222]
566 euler xyz [-152.   -75.4  -41.8] z [ 0.95  -0.22  -0.222]
567 euler xyz [ 67.7  …[+531 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as h:
 g=h['aligned/front_stereo_camera_left']
 print(type(g),g.shape,g.dtype)
 for i in [0,100,331,700,800]:
  a=g[i]; print(i,type(a),getattr(a,'shape',None),getattr(a,'dtype',None))
  if isinstance(a,np.ndarray): print(np.nanmin(a),np.nanmax(a),np.nanmedian(a))
 # raw group
 rg=h['raw/front_stereo_camera_left']; print('raw',rg.shape,rg.dtype)
 for i in [0,100]:
  a=rg[i]; print(i,type(a),getattr(a,'shape',None),getattr(a,'dtype',None), getattr(a,'keys',lambda:[])())
PY"
```
<details><summary>output</summary>

```
<class 'h5py._hl.dataset.Dataset'> (803,) object
0 <class 'numpy.ndarray'> (633913,) uint8
0 255 128.0
100 <class 'numpy.ndarray'> (632668,) uint8
0 255 127.0
331 <class 'numpy.ndarray'> (644283,) uint8
0 255 128.0
700 <class 'numpy.ndarray'> (637800,) uint8
0 255 128.0
800 <class 'numpy.ndarray'> (644382,) uint8
0 255 128.0
raw (803,) object
0 <class 'numpy.ndarray'> (584517,) uint8 []
100 <class 'numpy.ndarray'> (586476,) uint8 []

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch, nvdiffrast.torch as dr
ctx=dr.RasterizeCudaContext()
print('ctx ok')
PY"
```
<details><summary>output</summary>

```
ctx ok

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py, nvdiffrast.torch as dr
from PIL import Image
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=g.vertices.astype(np.float32); F=g.faces.astype(np.int32); N=g.vertex_normals.astype(np.float32); UV=g.visual.uv.astype(np.float32)
tex=np.array(g.visual.material.baseColorTexture).astype(np.float32)/255
print(V.shape,F.shape,N.shape,UV.shape,tex.shape,UV.min(0),UV.max(0))
ctx=dr.RasterizeCudaContext()
H,W=1152,1536
Vt=torch.from_numpy(V).cuda(); Ft=torch.from_numpy(F).cuda(); Nt=torch.from_numpy(N).cuda(); UVt=torch.from_numpy(UV).cuda(); text=torch.from_numpy(tex).cuda().permute(2,0,1)[None]
fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h: om=h['object'][:]
def render(i):
 P=torch.from_numpy((Tcw@T[i]).astype(np.float32)).cuda()
 v=(P[:3,:3]@Vt.T+P[:3,3:4]).T
 x,y,z=v[:,0],v[:,1],v[:,2]
 # y-down image coordinates as clip (positive down), nvdiffrast likely origin top-left
 clip=torch.stack([fx*x+cx*z, fy*y+cy*z, z, z],1)[None]
 rast,db=dr.rasterize(ctx,clip,Ft,(H,W))
 mask=(rast[0,...,3]>0)
 uvd,_=dr.interpolate(UVt[None],rast,F …[+1429 chars]
```
<details><summary>output</summary>

```
(296472, 3) (98866, 3) (296472, 3) (296472, 2) (2073, 3122, 3) [0.001121 0.001688] [0.998879 0.998312]
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 37, in <module>
    ren,mask,depth=render(i)
                   ^^^^^^^^^
  File "/work/scratch/render_test.py", line 30, in render
    n,_=dr.interpolate(nc[None],rast,Ft)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+312 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/nc=(P\\[:3,:3\\]@Nt.T).T/nc=(P[:3,:3]@Nt.T).T.contiguous()/\" scratch/render_test.py
python scratch/render_test.py"
```
<details><summary>output</summary>

```
(296472, 3) (98866, 3) (296472, 3) (296472, 2) (2073, 3122, 3) [0.001121 0.001688] [0.998879 0.998312]
50 mask 0 sam 4703 iou 0.0
100 mask 0 sam 4725 iou 0.0
200 mask 0 sam 3991 iou 0.0
331 mask 0 sam 1429 iou 0.0
500 mask 0 sam 2684 iou 0.0
700 mask 0 sam 2098 iou 0.0
800 mask 0 sam 2634 iou 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/clip=torch.stack(\\[fx\\*x+cx\\*z, fy\\*y+cy\\*z, z, z\\],1)\\[None\\]/clip=torch.stack([fx*x+cx*z, fy*y+cy*z, torch.zeros_like(z), z],1)[None]/\" scratch/render_test.py
python scratch/render_test.py"
```
<details><summary>output</summary>

```
(296472, 3) (98866, 3) (296472, 3) (296472, 2) (2073, 3122, 3) [0.001121 0.001688] [0.998879 0.998312]
50 mask 0 sam 4703 iou 0.0
100 mask 0 sam 4725 iou 0.0
200 mask 0 sam 3991 iou 0.0
331 mask 0 sam 1429 iou 0.0
500 mask 0 sam 2684 iou 0.0
700 mask 0 sam 2098 iou 0.0
800 mask 0 sam 2634 iou 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/render_test.py'
s=open(p).read()
s=s.replace(\"clip=torch.stack([fx*x+cx*z, fy*y+cy*z, torch.zeros_like(z), z],1)[None]\", \"u=fx*x/z+cx; v=fy*y/z+cy\\n clip=torch.stack([2*u/W-1, 1-2*v/H, torch.zeros_like(z), torch.ones_like(z)],1)[None]\")
open(p,'w').write(s)
PY
python scratch/render_test.py"
```
<details><summary>output</summary>

```
(296472, 3) (98866, 3) (296472, 3) (296472, 2) (2073, 3122, 3) [0.001121 0.001688] [0.998879 0.998312]
50 mask 3866 sam 4703 iou 0.0
100 mask 3996 sam 4725 iou 0.0
200 mask 4004 sam 3991 iou 0.03054911059551431
331 mask 2199 sam 1429 iou 0.0
500 mask 2183 sam 2684 iou 0.0
700 mask 2126 sam 2098 iou 0.0
800 mask 2080 sam 2634 iou 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# inspect predicted projection bbox at frame200 using simple formula
import numpy as np,trimesh,json,h5py
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh']; cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices
P=Tcw@T[200]; vc=(P[:3,:3]@V.T+P[:3,3:4]).T; uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3]
print(uv.min(0),uv.max(0),uv.mean(0)); print('T',T[200]);
with h5py.File('inputs/masks.h5','r') as h:
 m=h['object'][200]; y,x=np.where(m);print(x.min(),x.max(),y.min(),y.max(),x.mean(),y.mean())
PY"
```
<details><summary>output</summary>

```
[762.98915002 427.94376414] [800.47941381 620.53340275] [779.87019467 517.92489051]
T [[ 0.22063185 -0.97246814 -0.0750074   0.19526258]
 [-0.20519109 -0.1214598   0.97115564 -0.09191507]
 [-0.9535272  -0.19887854 -0.22633989  2.282348  ]
 [ 0.          0.          0.          1.        ]]
730 813 431 625 775.4961162615886 517.1297920320721

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# modify test to print bbox of mask, save mask
p='scratch/render_test.py'; s=open(p).read()
s=s.replace(\"print(i,'mask',mask.sum(),'sam',om[i].sum(),'iou',(mask&om[i]).sum()/max(1,(mask|om[i]).sum()))\", \"yy,xx=np.where(mask); print(i,'mask',mask.sum(),'bbox', (xx.min(),xx.max(),yy.min(),yy.max()),'sam',om[i].sum(),'iou',(mask&om[i]).sum()/max(1,(mask|om[i]).sum())); cv2.imwrite(f'scratch/mask_{i}.png',mask.astype(np.uint8)*255)\")
open(p,'w').write(s)
PY
python scratch/render_test.py"
```
<details><summary>output</summary>

```
(296472, 3) (98866, 3) (296472, 3) (296472, 2) (2073, 3122, 3) [0.001121 0.001688] [0.998879 0.998312]
50 mask 3866 bbox (669, 849, 185, 217) sam 4703 iou 0.0
100 mask 3996 bbox (670, 852, 182, 215) sam 4725 iou 0.0
200 mask 4004 bbox (763, 799, 531, 723) sam 3991 iou 0.03054911059551431
331 mask 2199 bbox (942, 1002, 625, 760) sam 1429 iou 0.0
500 mask 2183 bbox (599, 642, 602, 738) sam 2684 iou 0.0
700 mask 2126 bbox (725, 826, 218, 308) sam 2098 iou 0.0
800 mask 2080 bbox (724, 825, 218, 304) sam 2634 iou 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/1-2\\*v\\/H/2*v\\/H-1/\" scratch/render_test.py
python scratch/render_test.py"
```
<details><summary>output</summary>

```
(296472, 3) (98866, 3) (296472, 3) (296472, 2) (2073, 3122, 3) [0.001121 0.001688] [0.998879 0.998312]
50 mask 3870 bbox (669, 849, 934, 966) sam 4703 iou 0.7849260878617531
100 mask 3993 bbox (670, 852, 936, 969) sam 4725 iou 0.8060907395898074
200 mask 4002 bbox (763, 799, 428, 620) sam 3991 iou 0.28298555377207063
331 mask 2199 bbox (942, 1002, 391, 526) sam 1429 iou 0.1572567783094099
500 mask 2182 bbox (599, 642, 413, 548) sam 2684 iou 0.625250501002004
700 mask 2126 bbox (725, 826, 843, 933) sam 2098 iou 0.6085300837776085
800 mask 2079 bbox (724, 825, 847, 933) sam 2634 iou 0.7691441441441441

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/uv_test.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py, nvdiffrast.torch as dr
from PIL import Image
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K'],np.float32); Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices.astype(np.float32); F=g.faces.astype(np.int32); UV=g.visual.uv.astype(np.float32)
tex=np.array(g.visual.material.baseColorTexture).astype(np.float32)/255; H,W=1152,1536
ctx=dr.RasterizeCudaContext(); Vt=torch.from_numpy(V).cuda(); Ft=torch.from_numpy(F).cuda(); UVt=torch.from_numpy(UV).cuda(); text=torch.from_numpy(tex).cuda().permute(2,0,1)[None]
fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
def render(i,uf=1,vf=1,swap=0):
 P=torch.from_numpy((Tcw@T[i]).astype(np.float32)).cuda(); v=(P[:3,:3]@Vt.T+P[:3,3:4]).T; x,y,z=v.T
 u=fx*x/z+cx;vv=fy*y/z+cy; clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],1)[None]
 rast,_=dr.rasterize(ctx,clip,Ft,(H,W)); mask=rast[0,...,3]>0
 uvd,_=dr.interpolate(UVt[None],rast,Ft); uv=uvd[0]
 if swap: uv=uv[...,[1,0]]
 grid=torch.stack([uf*(uv[...,0]*2-1),vf*(uv[...,1]*2-1)],-1)[None]
 col=torch.nn.functional.grid_sample(text,grid,align_corners=False,mode='bilinear')[0].permute(1,2,0)
 out=np.zeros((H,W,3),np.uint8); out[mask.cpu().numpy()]=(col[mask].detach().c …[+811 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/uv_test.py", line 29, in <module>
    out=np.concatenate([np.concatenate(rows[:3],1),np.concatenate(rows[3:],1)],0);cv2.imwrite('scratch/uv_test.jpg',out)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1200 and the array at index 1 has size 2000

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/uv_test.py';s=open(p).read();s=s.replace(\"np.concatenate([np.concatenate(rows[:3],1),np.concatenate(rows[3:],1)],0)\",\"np.concatenate([np.concatenate(rows[:4],1),np.concatenate(rows[4:],1)],0)\")
open(p,'w').write(s)
PY
python scratch/uv_test.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/score_candidates.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py, nvdiffrast.torch as dr, time
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K'],np.float32); Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices.astype(np.float32); F=g.faces.astype(np.int32); UV=g.visual.uv.astype(np.float32)
tex=np.array(g.visual.material.baseColorTexture).astype(np.float32)/255; H,W=1152,1536
ctx=dr.RasterizeCudaContext(); Vt=torch.from_numpy(V).cuda(); Ft=torch.from_numpy(F).cuda(); UVt=torch.from_numpy(UV).cuda(); text=torch.from_numpy(tex).cuda().permute(2,0,1)[None]
fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()}
cap=cv2.VideoCapture('inputs/video.mp4')
ims={}
for i in [0,50,100,130,200,265,331,400,500,566,700,744,800]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();ims[i]=im
with h5py.File('inputs/masks.h5','r') as h: oms={i:h['object'][i] for i in ims}
def render_batch(Ps):
 B=len(Ps); v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None]
 x,y,z=v[ …[+1508 chars]
```
<details><summary>output</summary>

```
0 base:iou=0.128,ncc=-0.368 fx:iou=0.120,ncc=0.573 fy:iou=0.119,ncc=0.512 fz:iou=0.128,ncc=-0.327
50 base:iou=0.128,ncc=-0.353 fx:iou=0.120,ncc=0.572 fy:iou=0.119,ncc=0.488 fz:iou=0.128,ncc=-0.311
100 base:iou=0.126,ncc=-0.280 fx:iou=0.121,ncc=0.522 fy:iou=0.120,ncc=0.432 fz:iou=0.125,ncc=-0.160
130 base:iou=0.164,ncc=-0.348 fx:iou=0.145,ncc=0.333 fy:iou=0.146,ncc=0.509 fz:iou=0.161,ncc=-0.453
200 base:iou=0.072,ncc=-0.064 fx:iou=0.070,ncc=0.048 fy:iou=0.071,ncc=-0.096 fz:iou=0.073,ncc=-0.062
265 base:iou=0.163,ncc=-0.400 fx:iou=0.068,ncc=0.146 fy:iou=0.069,ncc=-0.116 fz:iou=0.164,ncc=-0.268
331 base:iou=0.014,ncc=-1.000 fx:iou=0.032,ncc=-0.099 fy:iou=0.034,ncc=0.258 fz:iou=0.017,ncc=-1.000
400 base:iou=0.000,ncc=-1.000 fx:iou=0.000,ncc=-1.000 fy:iou=0.000,ncc=-1.000 fz:iou=0.000,ncc=-1.00 …[+502 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json,h5py
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices
for i in [200,331,400,500,566,700,744,800]:
 P=Tcw@T[i];vc=(P[:3,:3]@V.T+P[:3,3:4]).T;uv=(K@vc.T).T;uv=uv[:,:2]/uv[:,2:3]
 print(i,'bbox',uv.min(0),uv.max(0),'z',vc[:,2].min(),vc[:,2].max(),'Tp',T[i,:3,3])
PY"
```
<details><summary>output</summary>

```
200 bbox [762.98915002 427.94376414] [800.47941381 620.53340275] z 2.200282655113844 2.3672445961793747 Tp [ 0.19526258 -0.09191507  2.282348  ]
331 bbox [942.33618852 390.8191711 ] [1003.33104665  527.26912848] z 3.115202372436983 3.20489195361186 Tp [ 1.0133063 -0.2519868  3.1173017]
400 bbox [757.8068782 471.8765954] [866.87705887 496.40959581] z 3.7434891732307056 3.979140344250239 Tp [ 0.5239429  -0.25624788  3.8605728 ]
500 bbox [599.21264119 412.77538142] [643.41028052 549.53820475] z 2.9586833707839966 3.1655306925423616 Tp [-0.34834132 -0.25717023  3.0985844 ]
566 bbox [661.41068518 439.53349531] [816.27159239 491.70472926] z 2.733961943281624 2.898589118814129 Tp [ 0.02895975 -0.23188186  2.8302922 ]
700 bbox [724.87191662 843.17320484] [826.87696521 934.46605724] z 2.66547575227 …[+344 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/vis_candidates.py <<'PY'
exec(open('scratch/score_candidates.py').read().split(\"for i in [0,50,100,130,200,265,331,400,500,566,700,744,800]:\")[0])
from PIL import Image
for i in [0,50,100,130,200,265,331,500,566,700,800]:
 Tc=Tcw@T[i];Ps=[];names=[]
 for n,Rf in flips.items():
  P=Tc.copy();P[:3,:3]=Tc[:3,:3]@T[i,:3,:3]@Rf;Ps.append(P);names.append(n)
 masks,cols=render_batch(np.array(Ps)); 
 ys,xs=np.where(oms[i]);cx0,cy0=int(xs.mean()),int(ys.mean());half=220;x0=int(np.clip(cx0-half,0,W-2*half));y0=int(np.clip(cy0-half,0,H-2*half));
 im=ims[i][y0:y0+2*half,x0:x0+2*half].copy(); panels=[im]
 for n,m,col in zip(names,masks,cols):
  ren=np.zeros((H,W,3),np.uint8);ren[m]=(col[m]*255).astype(np.uint8);rr=ren[y0:y0+2*half,x0:x0+2*half];alpha=(rr.sum(2)>0)[...,None];mix=(im*(1-alpha)+rr*alpha).astype(np.uint8);cv2.putText(mix,n,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(mix)
 out=np.concatenate(panels,1);cv2.imwrite(f'scratch/cand_{i}.jpg',out)
 print(i)
PY
python scratch/vis_candidates.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/vis_candidates.py", line 7, in <module>
    masks,cols=render_batch(np.array(Ps)); 
               ^^^^^^^^^^^^
NameError: name 'render_batch' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/vis_candidates.py';s=open(p).read();s=s.replace(\"exec(open('scratch/score_candidates.py').read().split(\\\"for i in [0,50,100,130,200,265,331,400,500,566,700,744,800]:\\\")[0])\", \"exec(open('scratch/score_candidates.py').read().rsplit(\\\"for i in [0,50,100,130,200,265,331,400,500,566,700,744,800]:\\\",1)[0])\")
open(p,'w').write(s)
PY
python scratch/vis_candidates.py"
```
<details><summary>output</summary>

```
0
50
100
130
200
265
331
500
566
700
800

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/render_facecolor.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py, nvdiffrast.torch as dr
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);F=g.faces.astype(np.int32);UV=g.visual.uv.astype(np.float32);tex=np.array(g.visual.material.baseColorTexture).astype(np.float32)/255
# face uv centroid then sample nearest texture
uvc=UV[F].mean(1); tx=np.clip((uvc[:,0]*(tex.shape[1]-1)).round().astype(int),0,tex.shape[1]-1);ty=np.clip(((1-uvc[:,1])*(tex.shape[0]-1)).round().astype(int),0,tex.shape[0]-1);fc=tex[ty,tx]
print('face color mean',fc.mean(0),'black frac',(fc.sum(1)<.03).mean())
H,W=1152,1536;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(F).cuda();FCt=torch.from_numpy(fc).cuda();fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
for i in [50,100,200,331,500,700,800]:
 P=torch.from_numpy((Tcw@T[i]).astype(np.float32)).cuda();v=(P[:3,:3]@Vt.T+P[:3,3:4]).T;x,y,z=v.T;u=fx*x/z+cx;vv=fy*y/z+cy;clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],1)[None]
 rast,_=dr.rasterize(ctx,clip,Ft,(H,W));mask=rast[0,. …[+621 chars]
```
<details><summary>output</summary>

```
face color mean [0.23389219 0.348694   0.40048206] black frac 0.0
50
100
200
331
500
700
800

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/uv_face_test.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py, nvdiffrast.torch as dr
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);F=g.faces.astype(np.int32);UV=g.visual.uv.astype(np.float32);tex=np.array(g.visual.material.baseColorTexture).astype(np.float32)/255
H,W=1152,1536;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(F).cuda();fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
def facecol(uvc):
 tx=np.clip((uvc[:,0]*(tex.shape[1]-1)).round().astype(int),0,tex.shape[1]-1);ty=np.clip(((1-uvc[:,1])*(tex.shape[0]-1)).round().astype(int),0,tex.shape[0]-1);return tex[ty,tx]
def render(i,uv):
 P=torch.from_numpy((Tcw@T[i]).astype(np.float32)).cuda();v=(P[:3,:3]@Vt.T+P[:3,3:4]).T;x,y,z=v.T;u=fx*x/z+cx;vv=fy*y/z+cy;clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],1)[None]
 rast,_=dr.rasterize(ctx,clip,Ft,(H,W));mask=rast[0,...,3]>0;tri=rast[0,...,3].long().clamp(0,len(F)-1);fc=torch.from_numpy(facecol(UV[F].mean(1))).cuda();col=fc[tri];out=np.zeros((H,W,3),np.uint8);out[mask.cpu().numpy()]=(co …[+1447 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/uv_face_test.py", line 5
    cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
                                             ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/uv_face_test.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]\", \"cap=cv2.VideoCapture('inputs/video.mp4')\\nwith h5py.File('inputs/masks.h5','r') as h: om=h['object'][:]\")
s=s.replace(\"row=[]\\n\", \"row=[]; results=[]\\n\")
s=s.replace(\"sc=ncc(ren,im,m) if m.sum()>10 else -9\\n\", \"sc=ncc(ren,im,m) if m.sum()>10 else -9; results.append((n,sc))\\n\")
s=s.replace(\"print(i,[(n,round(ncc(ren,im,m),3)) for n in []])\", \"print(i,[(n,round(s,3)) for n,s in results])\")
open(p,'w').write(s)
PY
python scratch/uv_face_test.py"
```
<details><summary>output</summary>

```
50 [('uv', 0.305), ('vu', 0.006), ('1uv', -0.168), ('u1v', -0.097), ('1u1v', 0.192), ('v1u', -0.037), ('1vu', -0.092), ('1v1u', 0.056)]
100 [('uv', 0.295), ('vu', 0.023), ('1uv', -0.174), ('u1v', -0.136), ('1u1v', 0.169), ('v1u', -0.041), ('1vu', -0.1), ('1v1u', 0.051)]
130 [('uv', 0.159), ('vu', 0.007), ('1uv', -0.07), ('u1v', -0.059), ('1u1v', 0.059), ('v1u', -0.007), ('1vu', -0.026), ('1v1u', 0.088)]
800 [('uv', 0.272), ('vu', -0.005), ('1uv', -0.14), ('u1v', -0.046), ('1u1v', 0.157), ('v1u', -0.07), ('1vu', -0.118), ('1v1u', 0.055)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/score_candidates_face.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py, nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K'],np.float32); Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices.astype(np.float32); F=g.faces.astype(np.int32); UV=g.visual.uv.astype(np.float32); tex=np.array(g.visual.material.baseColorTexture).astype(np.float32)/255
uvc=UV[F].mean(1);tx=np.clip((uvc[:,0]*(tex.shape[1]-1)).round().astype(int),0,tex.shape[1]-1);ty=np.clip(((1-uvc[:,1])*(tex.shape[0]-1)).round().astype(int),0,tex.shape[0]-1);fc=tex[ty,tx]
H,W=1152,1536;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(F).cuda();FCt=torch.from_numpy(fc).cuda();fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()}
cap=cv2.VideoCapture('inputs/video.mp4'); idlist=[0,50,100,130,200,265,331,400,500,566,700,744,800];ims={}
for i in idlist:cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,ims[i]=cap.read()
with h5py.File('inputs/masks.h5','r') as h:oms={i:h['object'][i] for i in idlist}
def render …[+1237 chars]
```
<details><summary>output</summary>

```
0 base:iou=0.128,ncc=-0.046 fx:iou=0.120,ncc=0.217 fy:iou=0.119,ncc=0.154 fz:iou=0.128,ncc=-0.045
50 base:iou=0.128,ncc=-0.043 fx:iou=0.120,ncc=0.153 fy:iou=0.119,ncc=0.170 fz:iou=0.128,ncc=0.003
100 base:iou=0.126,ncc=-0.058 fx:iou=0.121,ncc=0.129 fy:iou=0.120,ncc=0.087 fz:iou=0.125,ncc=-0.077
130 base:iou=0.164,ncc=-0.097 fx:iou=0.145,ncc=0.151 fy:iou=0.146,ncc=0.190 fz:iou=0.161,ncc=-0.127
200 base:iou=0.072,ncc=-0.054 fx:iou=0.070,ncc=0.057 fy:iou=0.071,ncc=-0.045 fz:iou=0.073,ncc=-0.051
265 base:iou=0.163,ncc=0.027 fx:iou=0.068,ncc=-0.040 fy:iou=0.069,ncc=-0.023 fz:iou=0.164,ncc=-0.043
331 base:iou=0.014,ncc=-1.000 fx:iou=0.032,ncc=-0.008 fy:iou=0.034,ncc=-0.001 fz:iou=0.017,ncc=-1.000
400 base:iou=0.000,ncc=-1.000 fx:iou=0.000,ncc=-1.000 fy:iou=0.000,ncc=-1.000 fz:iou=0.000,ncc=-1.00 …[+499 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('inputs/masks.h5','r') as h:
 for i in [0,50,100,130,200,265,331,500,566,700,744,800]:
  m=h['object'][i];y,x=np.where(m);print(i,m.sum(),'bbox',x.min(),x.max(),y.min(),y.max(),'cen',x.mean(),y.mean())
PY"
```
<details><summary>output</summary>

```
0 4904 bbox 664 850 935 974 cen 776.3313621533442 952.2728384991843
50 4703 bbox 664 850 935 973 cen 777.9083563682756 952.2298532851371
100 4725 bbox 664 850 935 973 cen 777.5583068783069 952.1593650793651
130 4373 bbox 670 822 830 977 cen 765.3391264578092 919.9967985364738
200 3991 bbox 730 813 431 625 cen 775.4961162615886 517.1297920320721
265 586 bbox 951 1006 487 527 cen 990.6877133105802 513.6569965870307
331 1429 bbox 957 986 414 471 cen 971.480055983205 441.1441567529741
500 2684 bbox 599 654 413 551 cen 623.7466467958271 478.077868852459
566 3203 bbox 644 800 427 509 cen 703.788011239463 461.6197315017171
700 2098 bbox 719 826 845 935 cen 779.5686367969495 885.9351763584366
744 2570 bbox 719 825 845 934 cen 781.2607003891051 884.3003891050583
800 2634 bbox 718 825 845 935 cen 78 …[+34 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# run score module without final loop then inspect
s=open('scratch/score_candidates_face.py').read();s=s.rsplit(\"for i in idlist:\",1)[0];exec(s)
i=100;Tc=Tcw@T[i];Ps=[];names=[]
for n,Rf in flips.items():
 P=Tc.copy();P[:3,:3]=Tc[:3,:3]@T[i,:3,:3]@Rf;Ps.append(P);names.append(n)
masks,cols=render_batch(np.array(Ps)); om=oms[i]
for n,m in zip(names,masks):
 print(n,m.sum(),om.sum(),(m&om).sum(),(m|om).sum(),m.shape, m.dtype)
y,x=np.where(masks[0]);print('bbox',x.min(),x.max(),y.min(),y.max())
PY"
```
<details><summary>output</summary>

```
base 4009 4725 976 7758 (1152, 1536) bool
fx 4049 4725 946 7828 (1152, 1536) bool
fy 4046 4725 939 7832 (1152, 1536) bool
fz 4006 4725 973 7758 (1152, 1536) bool
bbox 750 809 834 1031

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh,json
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);Vt=torch.from_numpy(V).cuda()
i=100;P=Tcw@T[i];Ps=np.array([P,P]);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None]
v0=torch.from_numpy(P).cuda()[:3,:3]@Vt.T+P[:3,3:4]
print(v.shape,v0.shape, (v[0]-v0.T).abs().max())
print('single bbox xy', ((torch.from_numpy(np.array(cam['K'],np.float32)).cuda()@v0).T[:,:2]/v0[2:3].T).min(0).values, ((torch.from_numpy(np.array(cam['K'],np.float32)).cuda()@v0).T[:,:2]/v0[2:3].T).max(0).values)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh,json
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));K=torch.from_numpy(np.array(cam['K'],np.float32)).cuda();Tcw=torch.from_numpy(np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)).cuda()
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();Vt=torch.from_numpy(g.vertices.astype(np.float32)).cuda()
i=100;P=(Tcw@torch.from_numpy(T[i]).cuda()).cpu().numpy();Ps=np.array([P,P]);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None]
v0=torch.from_numpy(P).cuda()[:3,:3]@Vt.T+P[:3,3:4];print(v.shape,v0.shape,(v[0]-v0.T).abs().max().item())
for vv in [v[0],v0.T]:
 z=vv[:,2];u=(K@vv.T).T;u=u[:,:2]/u[:,2:3];print(u.min(0).values,u.max(0).values)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh,json
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));Knp=np.array(cam['K'],np.float32);Tcwnp=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
K=torch.from_numpy(Knp).cuda();Tcw=torch.from_numpy(Tcwnp).cuda()
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();Vt=torch.from_numpy(g.vertices.astype(np.float32)).cuda()
i=100;P=Tcwnp@T[i];Ps=np.array([P,P]);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None]
v0=torch.from_numpy(P).cuda()[:3,:3]@Vt.T+P[:3,3:4];print(v.shape,v0.shape,(v[0]-v0.T).abs().max().item())
for vv in [v[0],v0.T]:
 u=(K@vv.T).T;u=u[:,:2]/u[:,2:3];print(u.min(0).values,u.max(0).values)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('scratch/score_candidates_face.py').read();s=s.rsplit(\"for i in idlist:\",1)[0];exec(s)
i=100;Tc=Tcw@T[i];Ps=[];names=[]
for n,Rf in flips.items():
 P=Tc.copy();P[:3,:3]=Tc[:3,:3]@T[i,:3,:3]@Rf;Ps.append(P);names.append(n)
Ps=np.array(Ps)
print('Ps0',Ps[0]);print('T',T[i]);print('Tcw dtype',Tcw.dtype, type(Tcw))
v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None]
# numpy verify
vn=np.einsum('bij,vj->bvi',Ps[:,:3,:3],V)+Ps[:,None,:3,3]
print('v err',np.abs(v.cpu().numpy()-vn).max(),v.shape)
for b in range(4):
 vv=v[b];u=(torch.from_numpy(K).cuda()@vv.T).T;u=u[:,:2]/u[:,2:3];print(names[b],u.min(0).values.cpu().numpy(),u.max(0).values.cpu().numpy())
PY"
```
<details><summary>output</summary>

```
Ps0 [[ 0.2086222   0.9574802  -0.19926786  0.06844928]
 [-0.34108052 -0.11972691 -0.9323771   1.153617  ]
 [-0.9165908   0.2624814   0.30160022  2.4188614 ]
 [ 0.          0.          0.          1.        ]]
T [[-0.01234067 -0.20861235 -0.97792023  0.22335725]
 [ 0.933009   -0.35415408  0.06377527  1.2280362 ]
 [-0.35963884 -0.91162246  0.19900773  2.382718  ]
 [ 0.          0.          0.          1.        ]]
Tcw dtype float32 <class 'numpy.ndarray'>
v err 2.3841858e-07 torch.Size([4, 296472, 3])
base [750.44904 834.1071 ] [ 810.0413 1032.2692]
fx [753.9071  875.94464] [ 813.9807 1079.7926]
fy [754.6842 875.8337] [ 813.94147 1080.3522 ]
fz [751.8979  834.10156] [ 809.1112 1032.3306]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('scratch/score_candidates_face.py').read();s=s.rsplit(\"for i in idlist:\",1)[0];exec(s)
print('Tcw',Tcw)
i=100;print('T',T[i]);print('prod',Tcw@T[i]);
PY"
```
<details><summary>output</summary>

```
Tcw [[ 9.9861795e-01 -1.4272360e-02 -5.0582331e-02 -1.6548863e-02]
 [ 1.3094287e-02  9.9963701e-01 -2.3545563e-02 -2.0795662e-02]
 [ 5.0900023e-02  2.2850681e-02  9.9844229e-01  4.2453525e-04]
 [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
T [[-0.01234067 -0.20861235 -0.97792023  0.22335725]
 [ 0.933009   -0.35415408  0.06377527  1.2280362 ]
 [-0.35963884 -0.91162246  0.19900773  2.382718  ]
 [ 0.          0.          0.          1.        ]]
prod [[-0.00744848 -0.15715745 -0.9875452   0.06844928]
 [ 0.9409767  -0.3352925   0.0462612   1.153617  ]
 [-0.33838686 -0.9289135   0.15037888  2.4188614 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('scratch/score_candidates_face.py').read();s=s.rsplit(\"for i in idlist:\",1)[0];exec(s)
print(flips.keys());print(flips['base'])
i=100;Tc=Tcw@T[i];P=Tc.copy();P[:3,:3]=Tc[:3,:3]@T[i,:3,:3]@flips['base'];print('Tc',Tc);print('Tc rot',Tc[:3,:3]);print('T rot',T[i,:3,:3]);print('P',P)
PY"
```
<details><summary>output</summary>

```
dict_keys(['base', 'fx', 'fy', 'fz'])
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
Tc [[-0.00744848 -0.15715745 -0.9875452   0.06844928]
 [ 0.9409767  -0.3352925   0.0462612   1.153617  ]
 [-0.33838686 -0.9289135   0.15037888  2.4188614 ]
 [ 0.          0.          0.          1.        ]]
Tc rot [[-0.00744848 -0.15715745 -0.9875452 ]
 [ 0.9409767  -0.3352925   0.0462612 ]
 [-0.33838686 -0.9289135   0.15037888]]
T rot [[-0.01234067 -0.20861235 -0.97792023]
 [ 0.933009   -0.35415408  0.06377527]
 [-0.35963884 -0.91162246  0.19900773]]
P [[ 0.2086222   0.9574802  -0.19926786  0.06844928]
 [-0.34108052 -0.11972691 -0.9323771   1.153617  ]
 [-0.9165908   0.2624814   0.30160022  2.4188614 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('scratch/score_candidates_face.py').read();s=s.rsplit(\"for i in idlist:\",1)[0];exec(s)
i=100;Tc=Tcw@T[i]; A=Tc[:3,:3]@T[i,:3,:3]; B=Tc[:3,:3]@T[i,:3,:3]@flips['base']; print('A\\n',A); print('B\\n',B); print('prod\\n',(Tcw@T[i])[:3,:3]); print('T.T@Tc.T\\n',(Tc.T@Tcw.T));
PY"
```
<details><summary>output</summary>

```
A
 [[ 0.2086222   0.9574802  -0.19926786]
 [-0.34108052 -0.11972691 -0.9323771 ]
 [-0.9165908   0.2624814   0.30160022]]
B
 [[ 0.2086222   0.95748019 -0.19926786]
 [-0.34108052 -0.11972691 -0.9323771 ]
 [-0.91659081  0.26248139  0.30160022]]
prod
 [[-0.00744848 -0.15715745 -0.9875452 ]
 [ 0.9409767  -0.3352925   0.0462612 ]
 [-0.33838686 -0.9289135   0.15037888]]
T.T@Tc.T
 [[-0.00375175  0.9485051  -0.31673694  0.        ]
 [-0.10516822 -0.31535685 -0.94312745  0.        ]
 [-0.9944471   0.02977245  0.10093566  0.        ]
 [-0.08701067  1.0763454   2.445363    1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);Tcw=np.linalg.inv(np.array(json.load(open('inputs/camera.json'))['T_world_cam'],np.float32)).astype(np.float32)
Tc=Tcw@T[100];A=Tc[:3,:3]@T[100,:3,:3]; print(Tcw);print(T[100]);print('Tc',Tc);print('A',A);print('prod',(Tcw@T[100])[:3,:3])
PY"
```
<details><summary>output</summary>

```
[[ 9.9861795e-01 -1.4272360e-02 -5.0582331e-02 -1.6548863e-02]
 [ 1.3094287e-02  9.9963701e-01 -2.3545563e-02 -2.0795662e-02]
 [ 5.0900023e-02  2.2850681e-02  9.9844229e-01  4.2453525e-04]
 [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]
[[-0.01234067 -0.20861235 -0.97792023  0.22335725]
 [ 0.933009   -0.35415408  0.06377527  1.2280362 ]
 [-0.35963884 -0.91162246  0.19900773  2.382718  ]
 [ 0.          0.          0.          1.        ]]
Tc [[-0.00744848 -0.15715745 -0.9875452   0.06844928]
 [ 0.9409767  -0.3352925   0.0462612   1.153617  ]
 [-0.33838686 -0.9289135   0.15037888  2.4188614 ]
 [ 0.          0.          0.          1.        ]]
A [[ 0.2086222   0.9574802  -0.19926786]
 [-0.34108052 -0.11972691 -0.9323771 ]
 [-0.9165908   0.2624814   0.30160022]]
prod [[-0.0074 …[+109 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/P\\[:3,:3\\]=Tc\\[:3,:3\\]@T\\[i,:3,:3\\]@Rf/P[:3,:3]=Tcw[:3,:3]@T[i,:3,:3]@Rf/g\" scratch/score_candidates.py scratch/score_candidates_face.py scratch/vis_candidates.py
python scratch/score_candidates_face.py"
```
<details><summary>output</summary>

```
0 base:iou=0.773,ncc=0.300 fx:iou=0.456,ncc=-0.243 fy:iou=0.455,ncc=-0.185 fz:iou=0.749,ncc=0.220
50 base:iou=0.785,ncc=0.298 fx:iou=0.458,ncc=-0.254 fy:iou=0.455,ncc=-0.181 fz:iou=0.754,ncc=0.236
100 base:iou=0.806,ncc=0.294 fx:iou=0.469,ncc=-0.198 fy:iou=0.470,ncc=-0.187 fz:iou=0.784,ncc=0.243
130 base:iou=0.730,ncc=0.160 fx:iou=0.457,ncc=-0.139 fy:iou=0.454,ncc=-0.218 fz:iou=0.742,ncc=0.191
200 base:iou=0.283,ncc=0.359 fx:iou=0.265,ncc=-0.237 fy:iou=0.280,ncc=-0.257 fz:iou=0.291,ncc=0.298
265 base:iou=0.005,ncc=-1.000 fx:iou=0.026,ncc=0.210 fy:iou=0.023,ncc=-0.318 fz:iou=0.003,ncc=-1.000
331 base:iou=0.157,ncc=0.048 fx:iou=0.322,ncc=0.036 fy:iou=0.321,ncc=0.051 fz:iou=0.158,ncc=0.045
400 base:iou=0.149,ncc=0.117 fx:iou=0.057,ncc=0.004 fy:iou=0.057,ncc=-0.402 fz:iou=0.151,ncc=0.048
500 b …[+494 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/score_all.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py, nvdiffrast.torch as dr, time, sys
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32); Fn=len(T)
cam=json.load(open('inputs/camera.json')); K=np.array(cam['K'],np.float32); Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=g.vertices.astype(np.float32); F=g.faces.astype(np.int32); UV=g.visual.uv.astype(np.float32); tex=np.array(g.visual.material.baseColorTexture).astype(np.float32)/255
uvc=UV[F].mean(1);tx=np.clip((uvc[:,0]*(tex.shape[1]-1)).round().astype(int),0,tex.shape[1]-1);ty=np.clip(((1-uvc[:,1])*(tex.shape[0]-1)).round().astype(int),0,tex.shape[0]-1);fc=tex[ty,tx]
H,W=1152,1536;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(F).cuda();FCt=torch.from_numpy(fc).cuda();fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()};names=list(flips)
cap=cv2.VideoCapture('inputs/video.mp4'); ims=[]
for i in range(Fn): ok,im=cap.read(); ims.append(im.astype(np.float32)/255)
with h5py.File('inputs/masks.h5','r') as h: oms=h['object'][:]
def render_batch(Ps):
 B=len(Ps);v=(torch.from_numpy(P …[+1542 chars]
```
<details><summary>output</summary>

```
0 / 803 elapsed 0.7488505840301514
80 / 803 elapsed 7.4762513637542725
160 / 803 elapsed 14.474695444107056
240 / 803 elapsed 21.104506969451904
320 / 803 elapsed 27.646642208099365
400 / 803 elapsed 34.24625301361084
480 / 803 elapsed 41.65984296798706
560 / 803 elapsed 49.241756439208984
640 / 803 elapsed 56.99845910072327
720 / 803 elapsed 64.85659265518188
800 / 803 elapsed 72.1450400352478
done 72.1477677822113

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
S=np.load('scratch/all_scores.npz');iou=S['ious'];ncc=S['ncc'];names=S['names'];r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);F=len(T)
rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
for j,n in enumerate(names):
 print(n,'iou med/mean',np.median(iou[:,j]),iou[:,j].mean(),'ncc',np.median(ncc[:,j]),ncc[:,j].mean(),'ncc>-0.1',(ncc[:,j]>-.1).mean())
print('\\nphases')
for a,b in [(0,95),(95,150),(150,250),(250,350),(350,450),(450,550),(550,650),(650,700),(700,803)]:
 print(a,b,'iou',np.round(np.median(iou[a:b],0),3),'ncc',np.round(np.median(ncc[a:b],0),3))
# best local averaged score
sm_i=np.stack([np.convolve(np.pad(iou[:,j],(2,2),mode='edge'),np.ones(5)/5,'valid') for j in range(4)],1)
sm_n=np.stack([np.convolve(np.pad(ncc[:,j],(2,2),mode='edge'),np.ones(5)/5,'valid') for j in range(4)],1)
score=sm_i+0.4*np.maximum(sm_n,-.5)
win=np.argmax(score,1)
print('\\nwinner counts overall',dict(zip(names,np.bincount(win,minlength=4))))
for a,b in [(0,95),(95,150),(150,250),(250,350),(350,450),(450,550),(550,650),(650,700),(700,803)]:print(a,b,dict(zip(names,np.bincount(win[a:b],minlength=4))))
print('\\nlargest base score drops / wins')
# frames where winner "'!= base and margin
margin=score[np.arange(F),win]-score[:,0]
for i in np.argsort(margin)[-80:][::-1]: print(i,'"'win',names[win[i]],'margin',round(margin[i],3),'iou …[+140 chars]
```
<details><summary>output</summary>

```
base iou med/mean 0.61113286 0.53626454 ncc 0.25356737 0.1760595 ncc>-0.1 0.9514321295143213
fx iou med/mean 0.37163934 0.34674022 ncc -0.2097542 -0.17399058 ncc>-0.1 0.27895392278953923
fy iou med/mean 0.37381947 0.34655207 ncc -0.18501072 -0.1696878 ncc>-0.1 0.27895392278953923
fz iou med/mean 0.61220473 0.5299947 ncc 0.22648858 0.16863628 ncc>-0.1 0.9427148194271482

phases
0 95 iou [0.783 0.456 0.454 0.752] ncc [ 0.294 -0.239 -0.182  0.223]
95 150 iou [0.74  0.439 0.436 0.731] ncc [ 0.256 -0.184 -0.204  0.226]
150 250 iou [0.685 0.43  0.427 0.682] ncc [ 0.249 -0.177 -0.155  0.244]
250 350 iou [0.242 0.256 0.256 0.245] ncc [ 0.069 -0.037 -0.025  0.06 ]
350 450 iou [0.232 0.193 0.193 0.22 ] ncc [ 0.039 -0.077 -0.081  0.047]
450 550 iou [0.577 0.405 0.406 0.577] ncc [ 0.212 -0.174 -0.157  …[+9082 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
S=np.load('scratch/all_scores.npz');iou=S['ious'];ncc=S['ncc'];names=list(S['names'])
d=ncc[:,3]-ncc[:,0];di=iou[:,3]-iou[:,0]
# smooth windows 10/20 excluding ncc=-1
for w in [5,10,20,40]:
 print('\\nwindow',w)
 for s in range(0,803,w):
  e=min(803,s+w); print(f'{s:3d}-{e-1:3d}', 'diou',round(float(di[s:e].mean()),3),'dncc',round(float(d[s:e].mean()),3), 'base',round(float(ncc[s:e,0].mean()),2),'fz',round(float(ncc[s:e,3].mean()),2))
# print ncc around phase
for s in range(500,803,10):
 e=min(803,s+10);print(s,e,'dncc',round(d[s:e].mean(),3),'diou',round(di[s:e].mean(),3),'base',np.round(ncc[s:e,0].mean(),2),'fz',np.round(ncc[s:e,3].mean(),2))
PY"
```
<details><summary>output</summary>

```
 fz 0.32
175-179 diou -0.015 dncc -0.015 base 0.25 fz 0.24
180-184 diou 0.001 dncc -0.038 base 0.28 fz 0.24
185-189 diou 0.003 dncc -0.021 base 0.25 fz 0.23
190-194 diou 0.006 dncc -0.027 base 0.24 fz 0.21
195-199 diou 0.003 dncc -0.029 base 0.32 fz 0.29
200-204 diou 0.001 dncc -0.027 base 0.33 fz 0.3
205-209 diou -0.007 dncc 0.013 base 0.29 fz 0.3
210-214 diou 0.01 dncc 0.037 base 0.19 fz 0.23
215-219 diou -0.0 dncc 0.068 base 0.14 fz 0.2
220-224 diou -0.007 dncc 0.044 base 0.21 fz 0.26
225-229 diou -0.009 dncc 0.023 base 0.16 fz 0.18
230-234 diou -0.003 dncc -0.037 base -0.02 fz -0.06
235-239 diou 0.006 dncc -0.029 base 0.13 fz 0.1
240-244 diou 0.003 dncc -0.041 base 0.23 fz 0.19
245-249 diou -0.005 dncc 0.025 base 0.17 fz 0.2
250-254 diou -0.004 dncc -0.019 base 0.25 fz 0.23
255-259 dio …[+11103 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
S=np.load('scratch/all_scores.npz');iou=S['ious'];ncc=S['ncc'];names=list(S['names']);r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);P=T[:,:3,3];F=len(T)
rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
dp=np.linalg.norm(np.diff(P,axis=0),axis=1)
for a,b in [(255,280),(315,340),(350,375),(438,462),(550,575),(638,665),(705,750)]:
 print('\\n',a,b)
 for i in range(a,b):
  bi=np.argmax(iou[i]+.2*np.maximum(ncc[i],0))
  print(f'{i:3d} p={P[i,0]:.2f},{P[i,1]:.2f},{P[i,2]:.2f} dp={dp[i-1] if i else 0:.3f} dr={rel[i-1] if i else 0:6.1f} iou={iou[i,0]:.2f} ncc={ncc[i,0]:.2f} best={names[bi]}')
PY"
```
<details><summary>output</summary>

```

 255 280
255 p=0.71,-0.12,2.50 dp=0.016 dr=   5.5 iou=0.70 ncc=0.32 best=base
256 p=0.74,-0.11,2.52 dp=0.036 dr=   8.5 iou=0.61 ncc=0.31 best=base
257 p=0.76,-0.12,2.55 dp=0.032 dr=   9.1 iou=0.61 ncc=0.28 best=base
258 p=0.77,-0.13,2.55 dp=0.016 dr=   9.1 iou=0.58 ncc=0.32 best=base
259 p=0.78,-0.14,2.56 dp=0.014 dr=   9.1 iou=0.61 ncc=0.27 best=fz
260 p=0.78,-0.13,2.55 dp=0.011 dr=  23.2 iou=0.57 ncc=0.27 best=base
261 p=0.79,-0.12,2.52 dp=0.032 dr=  23.2 iou=0.57 ncc=0.27 best=base
262 p=0.82,-0.14,2.52 dp=0.037 dr=   0.0 iou=0.33 ncc=0.28 best=fz
263 p=0.82,-0.10,2.56 dp=0.055 dr=  49.5 iou=0.45 ncc=0.23 best=fz
264 p=0.81,-0.10,2.55 dp=0.011 dr=  26.7 iou=0.21 ncc=0.26 best=base
265 p=0.80,-0.15,2.57 dp=0.057 dr=   0.0 iou=0.00 ncc=-1.00 best=fx
266 p=0.83,-0.10,2.67 dp=0.122 dr= 137 …[+12534 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('inputs/reconstruction.npz');c=r['contact_prob_right_left'];T=r['T_world_mesh'];p=T[:,:3,3];dp=np.linalg.norm(np.diff(p,axis=0),axis=1)
for i in range(80,141):print(i,'contact',np.round(c[i],3),'dp',round(dp[i-1],4))
print('---')
for i in range(670,731):print(i,'contact',np.round(c[i],3),'dp',round(dp[i-1],4))
PY"
```
<details><summary>output</summary>

```
[0. 0.] dp 0.0055
719 contact [0. 0.] dp 0.0127
720 contact [0. 0.] dp 0.0058
721 contact [0. 0.] dp 0.0009
722 contact [0. 0.] dp 0.001
723 contact [0. 0.] dp 1e-04
724 contact [0. 0.] dp 0.0016
725 contact [0. 0.] dp 0.0007
726 contact [0. 0.] dp 0.0039
727 contact [0. 0.] dp 0.0007
728 contact [0. 0.] dp 0.0018
729 contact [0. 0.] dp 0.0026
730 contact [0. 0.] dp 0.0012

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/test_static.py <<'PY'
exec(open('scratch/score_candidates_face.py').read().rsplit(\"for i in idlist:\",1)[0])
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
# score helper for arbitrary poses
cap=cv2.VideoCapture('inputs/video.mp4');ims2={}
for i in range(803):ok,im=cap.read();ims2[i]=im.astype(np.float32)/255
with h5py.File('inputs/masks.h5','r') as h: oms2=h['object'][:]
def ncc2(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
def score_poses(Ps,idx):
 masks,cols=render_batch(np.array(Ps));outs=[]
 for b,i in enumerate(idx):
  m=masks[b];om=oms2[i];inter=m&om;union=m|om;iou=inter.sum()/max(1,union.sum());er=cv2.erode(om.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&m;outs.append((iou,ncc2(cols[b],ims2[i],er)))
 return np.array(outs)
def avg_pose(idx):
 A=T[idx];Rs=A[:,:3,:3]
 # chordal L2 mean via SVD
 M=np.einsum('nij,nkj->ik',Rs,Rs);U,S,Vt=np.linalg.svd(M);R=U@Vt
 if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
 p=np.median(A[:,:3,3],axis=0)
 P=np.eye(4);P[:3,:3]=R;P[:3,3]=p;return P
for a,b in [(0,96),(0,110),(0,90),(710,803),(700,803),(690,803),(720,803)]:
 idx=list(range(a,b));P=avg_pose(idx);Pc=Tcw@P;Ps=[Pc]*len(idx);sc=score_poses(Ps,idx);base=score_poses([Tcw@ …[+247 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_static.py", line 28, in <module>
    idx=list(range(a,b));P=avg_pose(idx);Pc=Tcw@P;Ps=[Pc]*len(idx);sc=score_poses(Ps,idx);base=score_poses([Tcw@T[i] for i in idx],idx)
                                                                      ^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/test_static.py", line 16, in score_poses
    masks,cols=render_batch(np.array(Ps));outs=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 13, in render_batch
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: double != float

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/Pc=Tcw@P;Ps/Pc=(Tcw@P).astype(np.float32);Ps/' scratch/test_static.py
python scratch/test_static.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_static.py", line 28, in <module>
    idx=list(range(a,b));P=avg_pose(idx);Pc=(Tcw@P).astype(np.float32);Ps=[Pc]*len(idx);sc=score_poses(Ps,idx);base=score_poses([Tcw@T[i] for i in idx],idx)
                                                                                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/test_static.py", line 16, in score_poses
    masks,cols=render_batch(np.array(Ps));outs=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 13, in render_batch
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: double != float

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "sed -i 's/\\[Tcw@T\\[i\\] for i in idx\\]/[(Tcw@T[i]).astype(np.float32) for i in idx]/' scratch/test_static.py
python scratch/test_static.py"
```
<details><summary>output</summary>

```
0 96 avg iou/ncc [ 0.1592 -0.0443] base [0.7847 0.292 ] med [ 0.1592 -0.0437] [0.7831 0.2934] P [[ 1. -0. -0.]
 [-0.  1.  0.]
 [ 0. -0.  1.]] [0.2199 1.2395 2.414 ]
0 110 avg iou/ncc [ 0.1595 -0.0195] base [0.7848 0.2925] med [ 0.1596 -0.0212] [0.7832 0.2938] P [[ 1. -0.  0.]
 [-0.  1. -0.]
 [-0.  0.  1.]] [0.2206 1.2399 2.4136]
0 90 avg iou/ncc [ 0.1595 -0.0313] base [0.7865 0.2936] med [ 0.1594 -0.0305] [0.7834 0.2946] P [[ 1. -0. -0.]
 [ 0.  1.  0.]
 [ 0. -0.  1.]] [0.2196 1.2383 2.4115]
Traceback (most recent call last):
  File "/work/scratch/test_static.py", line 28, in <module>
    idx=list(range(a,b));P=avg_pose(idx);Pc=(Tcw@P).astype(np.float32);Ps=[Pc]*len(idx);sc=score_poses(Ps,idx);base=score_poses([(Tcw@T[i]).astype(np.float32) for i in idx],idx)
                                …[+647 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/M=np.einsum('nij,nkj->ik',Rs,Rs)/M=Rs.sum(axis=0)/\" scratch/test_static.py
python scratch/test_static.py"
```
<details><summary>output</summary>

```
0 96 avg iou/ncc [0.7876 0.305 ] base [0.7847 0.292 ] med [0.7908 0.3047] [0.7831 0.2934] P [[ 0.041 -0.168 -0.985]
 [ 0.991 -0.122  0.063]
 [-0.131 -0.978  0.161]] [0.2199 1.2395 2.414 ]
0 110 avg iou/ncc [0.7931 0.2883] base [0.7848 0.2925] med [0.7974 0.287 ] [0.7832 0.2938] P [[ 0.041 -0.175 -0.984]
 [ 0.99  -0.128  0.064]
 [-0.137 -0.976  0.168]] [0.2206 1.2399 2.4136]
0 90 avg iou/ncc [0.7878 0.316 ] base [0.7865 0.2936] med [0.7905 0.3152] [0.7834 0.2946] P [[ 0.043 -0.164 -0.985]
 [ 0.992 -0.111  0.062]
 [-0.12  -0.98   0.158]] [0.2196 1.2383 2.4115]
710 803 avg iou/ncc [0.7756 0.3015] base [0.7678 0.2754] med [0.7727 0.301 ] [0.7695 0.2736] P [[ 0.24   0.734 -0.635]
 [ 0.971 -0.186  0.152]
 [-0.006 -0.653 -0.757]] [0.2851 1.238  2.9046]
700 803 avg iou/ncc [0.7567 0.3115] base [0. …[+529 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2,os
cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
for name,a,b in [('init',0,90),('final',710,803)]:
 obj=om[a:b].any(0); cnt=om[a:b].sum(0); nc=((om[a:b])&(~pm[a:b])).sum(0)
 print(name,'union',obj.sum(),'intersection',om[a:b].all(0).sum(),'vis union',(nc>0).sum())
 ys,xs=np.where(obj);x0,x1=xs.min(),xs.max();y0,y1=ys.min(),ys.max();pad=30;x0=max(0,x0-pad);x1=min(1535,x1+pad);y0=max(0,y0-pad);y1=min(1151,y1+pad)
 # pooled median image
 stack=[];cap.set(cv2.CAP_PROP_POS_FRAMES,a)
 for i in range(a,b):
  ok,im=cap.read()
  if ok:stack.append(im)
 stack=np.stack(stack)
 good=om[a:b]&(~pm[a:b]); stack2=stack.copy()
 pooled=np.median(np.where(good[...,None],stack,0),axis=0).astype(np.uint8) # median zeros bad; use nanmedian
 arr=np.where(good[...,None],stack.astype(float),np.nan);pooled=np.nanmedian(arr,axis=0);pooled=np.nan_to_num(pooled).astype(np.uint8)
 # visible count normalized
 vis=cv2.applyColorMap(np.clip(cnt/30*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
 crop=stack[0][y0:y1,x0:x1];out=np.concatenate([crop,pooled[y0:y1,x0:x1],vis[y0:y1,x0:x1]],1);cv2.imwrite(f'scratch/static_{name}_pool.jpg',out);np.savez(f'scratch/static_{name}.npz',obj=obj,visible=good[a:b],count=cnt,pooled=pooled,bbox=np.array([x0,x1,y0,y1]))
 print('bbox',x0,x1,y0,y1)
PY"
```
<details><summary>output</summary>

```
init union 4957 intersection 4598 vis union 4957
<stdin>:16: RuntimeWarning: All-NaN slice encountered
bbox 634 881 905 1004
final union 2789 intersection 2452 vis union 2789
bbox 688 856 815 965

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import struct,json
p='inputs/object_mesh.glb';d=open(p,'rb').read();magic,ver,length=struct.unpack_from('<4sII',d,0);off=12;chunks=[]
while off<length:
 cl,ct=struct.unpack_from('<II',d,off);off+=8;chunks.append((ct,d[off:off+cl]));off+=cl
print(magic,ver,length,[(hex(t),len(x)) for t,x in chunks])
j=json.loads(chunks[0][1].decode().rstrip('\\x00 \\t\\r\\n'))
print(json.dumps(j,indent=2)[:20000])
PY"
```
<details><summary>output</summary>

```
b'glTF' 2 10231524 [('0x4e4f534a', 1396), ('0x4e4942', 10230100)]
{
  "scene": 0,
  "scenes": [
    {
      "nodes": [
        0
      ]
    }
  ],
  "asset": {
    "version": "2.0",
    "generator": "https://github.com/mikedh/trimesh"
  },
  "accessors": [
    {
      "componentType": 5125,
      "type": "SCALAR",
      "bufferView": 0,
      "count": 296598,
      "max": [
        296471
      ],
      "min": [
        0
      ]
    },
    {
      "componentType": 5126,
      "type": "VEC3",
      "byteOffset": 0,
      "bufferView": 1,
      "count": 296472,
      "max": [
        0.04483673721551895,
        0.045110441744327545,
        0.3330405652523041
      ],
      "min": [
        -0.044024404138326645,
        -0.043401267379522324,
        -0.20978295803070068
      ]
    },
  …[+1663 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
for m in ['pyrender','moderngl','OpenGL','pytorch3d','kaolin','open3d']:
 try:
  x=__import__(m);print(m,'OK',getattr(x,'__version__',''))
 except Exception as e:print(m,'NO',e)
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
moderngl OK 5.12.0
OpenGL OK 3.1.0
pytorch3d OK 0.7.9
kaolin OK 0.18.0
open3d OK 0.20.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/pyrender_test.py <<'PY'
import numpy as np, trimesh, json, cv2, os
os.environ['PYOPENGL_PLATFORM']='egl'
import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
# pyrender material/trimesh
mesh=pyrender.Mesh.from_trimesh(g,smooth=False)
print('mesh ok')
renderer=pyrender.OffscreenRenderer(1536,1152,point_size=1)
print('renderer ok',renderer.viewport_width,renderer.viewport_height)
flip=np.diag([1,-1,-1,1.]);near=.01;far=100
P=np.array([[2*K[0,0]/1536,0,(1536-2*K[0,2])/1536,0],[0,-2*K[1,1]/1152,(1152-2*K[1,2])/1152,0],[0,0,-(far+near)/(far-near),-2*far*near/(far-near)],[0,0,-1,0]],float)
camera=pyrender.Camera(projectionMatrix=P)
cap=cv2.VideoCapture('inputs/video.mp4')
for i in [0,50,100,200,500,800]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[1,1,1])
 scene.add(mesh,pose=Tcw@T[i]@flip)
 scene.add(camera,pose=np.eye(4))
 # lights
 light=pyrender.DirectionalLight(color=[1,1,1],intensity=1.5);scene.add(light,pose=Twc) # maybe
 color,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA)
 # alpha
 alpha=(color[...,3]>0).astype(np.float32)[...,None];mix=(im*(1-.8*alpha)+color[...,:3]*.8*alpha).astype(np.uint8)
 cv2.imwrite(f'scratch/pyr_{i}.jpg',np. …[+106 chars]
```
<details><summary>output</summary>

```
mesh ok
renderer ok 1536 1152
Traceback (most recent call last):
  File "/work/scratch/pyrender_test.py", line 13, in <module>
    camera=pyrender.Camera(projectionMatrix=P)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Can't instantiate abstract class Camera with abstract method get_projection_matrix

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pyrender_test.py';s=open(p).read();s=s.replace(\"camera=pyrender.Camera(projectionMatrix=P)\",\"camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],1152-K[1,2],znear=near,zfar=far)\")
open(p,'w').write(s)
PY
python scratch/pyrender_test.py"
```
<details><summary>output</summary>

```
mesh ok
renderer ok 1536 1152
0 0
50 0
100 0
200 0
500 0
800 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/pose=Tcw@T\\[i\\]@flip/pose=flip@Tcw@T[i]/' scratch/pyrender_test.py
python scratch/pyrender_test.py"
```
<details><summary>output</summary>

```
mesh ok
renderer ok 1536 1152
0 1007811
50 984990
100 1014423
200 1022796
500 553981
800 533517

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/score_pyrender.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os,time
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False)
renderer=pyrender.OffscreenRenderer(1536,1152,point_size=1);flip=np.diag([1,-1,-1,1.]);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],1152-K[1,2],znear=.01,zfar=100)
scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
ims={}
for i in [0,50,100,130,200,265,331,400,500,566,700,744,800]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();ims[i]=im
with h5py.File('inputs/masks.h5','r') as h:oms={i:h['object'][i] for i in ims}
def render(P):
 mesh_node=scene.mesh_nodes[0];scene.set_pose(mesh_node,pose=flip@Tcw@P)
 color,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);return color[...,:3]/255.,depth>1e-5
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
for i in ims:
 col,m=render(T[i]);om=oms[i] …[+312 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/score_pyrender.py", line 21, in <module>
    col,m=render(T[i]);om=oms[i];inter=m&om;union=m|om;iou=inter.sum()/max(1,union.sum());er=cv2.erode(om.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)&m;sc=ncc(col,ims[i].astype(float)/255,er);print(i,'area',m.sum(),'iou',iou,'ncc',sc,'bbox',cv2.boundingRect(m.astype(np.uint8)))
          ^^^^^^^^^^^^
  File "/work/scratch/score_pyrender.py", line 11, in render
    mesh_node=scene.mesh_nodes[0];scene.set_pose(mesh_node,pose=flip@Tcw@P)
              ~~~~~~~~~~~~~~~~^^^
TypeError: 'set' object is not subscriptable

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/score_pyrender.py';s=open(p).read();s=s.replace(\"scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);scene.add(mesh);scene.add(camera,pose=np.eye(4));\", \"scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);mesh_node=scene.add(mesh);scene.add(camera,pose=np.eye(4));\")
s=s.replace(\" mesh_node=scene.mesh_nodes[0];\", \" \")
open(p,'w').write(s)
PY
python scratch/score_pyrender.py"
```
<details><summary>output</summary>

```
0 area 3960 iou 0.07299358431182666 ncc -0.3372344071988778 bbox (668, 965, 183, 33)
50 area 3873 iou 0.06799501867995018 ncc -0.37655488742398546 bbox (669, 965, 181, 33)
100 area 3956 iou 0.049443907156673114 ncc -0.23724028757144053 bbox (671, 967, 182, 33)
130 area 3853 iou 0.13290180415920672 ncc -0.14223442120989854 bbox (681, 863, 144, 144)
200 area 4005 iou 0.21630666261028295 ncc 0.36416431061150073 bbox (763, 459, 38, 192)
265 area 3161 iou 0.005636070853462158 ncc -1 bbox (943, 457, 46, 167)
331 area 2201 iou 0.04731679169070975 ncc -0.043683572377700174 bbox (943, 422, 60, 136)
400 area 1348 iou 0.0 ncc -1 bbox (758, 503, 109, 24)
500 area 2176 iou 0.24871531346351491 ncc 0.50722344205015 bbox (599, 443, 44, 137)
566 area 2663 iou 0.021773210242118098 ncc 0.016989693604661238 b …[+254 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/1152-K\\[1,2\\]/K[1,2]/' scratch/score_pyrender.py
python scratch/score_pyrender.py"
```
<details><summary>output</summary>

```
0 area 3960 iou 0.7682026730500698 ncc 0.6869605158620131 bbox (668, 934, 183, 33)
50 area 3873 iou 0.7766728817070644 ncc 0.6968987846014684 bbox (669, 934, 181, 33)
100 area 3956 iou 0.7928541924824453 ncc 0.6350688206631394 bbox (671, 936, 182, 33)
130 area 3853 iou 0.7230833682446586 ncc 0.3582073003718919 bbox (681, 832, 144, 144)
200 area 4005 iou 0.2842916800513974 ncc 0.8516581843429836 bbox (763, 428, 38, 192)
265 area 3161 iou 0.004557640750670241 ncc -1 bbox (943, 426, 46, 167)
331 area 2201 iou 0.1601150527325024 ncc -0.1182644960799717 bbox (943, 391, 60, 136)
400 area 1348 iou 0.14960629921259844 ncc 0.20283021593799824 bbox (758, 472, 109, 24)
500 area 2176 iou 0.6227045075125208 ncc 0.8772076401341643 bbox (599, 412, 44, 137)
566 area 2663 iou 0.19567876070118223 ncc -0.163 …[+293 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/pyrender_all_base.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os,time
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);F=len(T);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.01,zfar=100)
scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);mesh_node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:oms=h['object'][:]
ious=np.zeros(F,np.float32);nccs=np.zeros(F,np.float32);areas=np.zeros(F,np.int32);colors=[]
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
t0=time.time()
for i in range(F):
 ok,im=cap.read();om=cv2.resize(oms[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
 scene.set_pose(mesh_node,pose=flip@Tcw@T[i]);col,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);m=de …[+445 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/pyrender_all_base.py", line 7
    cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:oms=h['object'][:]
                                             ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pyrender_all_base.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:oms=h['object'][:]\", \"cap=cv2.VideoCapture('inputs/video.mp4')\\nwith h5py.File('inputs/masks.h5','r') as h: oms=h['object'][:]\")
open(p,'w').write(s)
PY
python scratch/pyrender_all_base.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pyrender_all_base.py", line 21, in <module>
    inter=m&om;union=m|om;ious[i]=inter.sum()/max(1,union.sum());areas[i]=m.sum();er=cv2.erode(om.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&m;nccs[i]=ncc(col,im.astype(np.float32)/255,er)
                                                                                                                                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pyrender_all_base.py", line 13, in ncc
    x=a[...,c][m];y=b[...,c][m]
                    ~~~~~~~~^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1152 but corresponding boolean dimension is 576

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pyrender_all_base.py';s=open(p).read();s=s.replace(\"ok,im=cap.read();om=\", \"ok,im=cap.read();im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA);om=\")
open(p,'w').write(s)
PY
python scratch/pyrender_all_base.py"
```
<details><summary>output</summary>

```
0 elapsed 0.3501894474029541
80 elapsed 5.186862230300903
160 elapsed 10.108811855316162
240 elapsed 15.429373025894165
320 elapsed 21.739543199539185
400 elapsed 27.458545684814453
480 elapsed 34.98415470123291
560 elapsed 43.230218172073364
640 elapsed 51.978076696395874
720 elapsed 59.606677293777466
800 elapsed 66.36024022102356
done 66.66849064826965

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
B=np.load('scratch/pyrender_base_scores.npz');i=B['iou'];n=B['ncc'];r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'];p=T[:,:3,3];dp=np.linalg.norm(np.diff(p,axis=0),axis=1);rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
print('stats iou',np.quantile(i,[0,.1,.25,.5,.75,.9,.99,1]),'ncc',np.quantile(n,[0,.1,.25,.5,.75,.9,.99,1]))
for a,b in [(0,96),(96,150),(150,250),(250,350),(350,450),(450,550),(550,650),(650,700),(700,803)]:print(a,b,'iou',round(i[a:b].mean(),3),'ncc',round(n[a:b].mean(),3),'bad',np.sum((i[a:b]<.3)|(n[a:b]<-.1)))
# strongest bad/low score
score=i+.3*np.maximum(n,-.5)
for j in np.argsort(score)[:100]: print(j,'score',round(score[j],3),'iou',round(i[j],3),'ncc',round(n[j],3),'dp',round(dp[j-1],3) if j else 0,'dr',round(rel[j-1],1) if j else 0,'p',np.round(p[j],2))
PY"
```
<details><summary>output</summary>

```
stats iou [0.         0.14828504 0.31288984 0.60652173 0.74322101 0.77477479
 0.80056396 0.84406781] ncc [-1.         -0.21488877  0.22278319  0.58978671  0.70486677  0.78492267
  0.88709889  0.94470125]
0 96 iou 0.743 ncc 0.675 bad 0
96 150 iou 0.64 ncc 0.447 bad 5
150 250 iou 0.602 ncc 0.509 bad 12
250 350 iou 0.264 ncc -0.002 bad 63
350 450 iou 0.283 ncc 0.043 bad 69
450 550 iou 0.511 ncc 0.435 bad 23
550 650 iou 0.506 ncc 0.413 bad 32
650 700 iou 0.476 ncc 0.457 bad 13
700 803 iou 0.757 ncc 0.746 bad 0
361 score -0.15 iou 0.0 ncc -1.0 dp 0.042 dr 57.0 p [ 1.02 -0.27  3.46]
362 score -0.15 iou 0.0 ncc -1.0 dp 0.059 dr 0.0 p [ 1.01 -0.25  3.41]
363 score -0.15 iou 0.0 ncc -1.0 dp 0.1 dr 0.0 p [ 1.   -0.27  3.51]
270 score -0.15 iou 0.0 ncc -1.0 dp 0.028 dr 0.0 p [ 0.81 -0.1   2.54]
271 s …[+7154 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
B=np.load('scratch/pyrender_base_scores.npz');i=B['iou'];n=B['ncc'];a=B['area']
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
for j in range(355,375):print(j,'area',a[j],'sam',om[j].sum(),'iou',round(i[j],3),'ncc',round(n[j],3))
PY"
```
<details><summary>output</summary>

```
355 area 450 sam 946 iou 0.053 ncc -1.0
356 area 443 sam 878 iou 0.047 ncc -1.0
357 area 469 sam 877 iou 0.016 ncc -1.0
358 area 469 sam 658 iou 0.036 ncc -1.0
359 area 448 sam 653 iou 0.025 ncc -1.0
360 area 431 sam 599 iou 0.1 ncc -1.0
361 area 429 sam 495 iou 0.0 ncc -1.0
362 area 451 sam 376 iou 0.0 ncc -1.0
363 area 425 sam 430 iou 0.0 ncc -1.0
364 area 431 sam 459 iou 0.023 ncc -1.0
365 area 442 sam 350 iou 0.062 ncc -1.0
366 area 440 sam 431 iou 0.009 ncc -1.0
367 area 426 sam 535 iou 0.026 ncc -1.0
368 area 449 sam 587 iou 0.038 ncc -1.0
369 area 439 sam 694 iou 0.077 ncc 0.158
370 area 464 sam 927 iou 0.049 ncc -1.0
371 area 443 sam 913 iou 0.065 ncc -0.623
372 area 456 sam 1328 iou 0.206 ncc 0.381
373 area 455 sam 1288 iou 0.307 ncc 0.46
374 area 451 sam 1212 iou 0.287 ncc 0.374

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/pyrender_all_flips.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os,time
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);F=len(T);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()};names=list(flips)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.01,zfar=100)
scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);mesh_node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:oms=h['object'][:]
ious=np.zeros((F,4),np.float32);nccs=np.zeros((F,4),np.float32);areas=np.zeros((F,4),np.int32)
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww) …[+875 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/pyrender_all_flips.py", line 9
    cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:oms=h['object'][:]
                                             ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pyrender_all_flips.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:oms=h['object'][:]\", \"cap=cv2.VideoCapture('inputs/video.mp4')\\nwith h5py.File('inputs/masks.h5','r') as h: oms=h['object'][:]\")
open(p,'w').write(s)
PY
python scratch/pyrender_all_flips.py"
```
<details><summary>output</summary>

```
0 elapsed 0.610337495803833
80 elapsed 24.634238719940186
160 elapsed 48.3236620426178
240 elapsed 68.80686283111572
320 elapsed 88.64721894264221
400 elapsed 109.83491444587708
480 elapsed 129.64796376228333
560 elapsed 146.83441495895386
640 elapsed 170.14454436302185
720 elapsed 192.32188653945923
800 elapsed 212.963223695755
done 213.4534022808075

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
S=np.load('scratch/pyrender_flips_scores.npz');iou=S['iou'];ncc=S['ncc'];names=list(S['names']);F=len(iou);r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'];p=T[:,:3,3]
rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
print('med iou',np.round(np.median(iou,0),3),'med ncc',np.round(np.median(ncc,0),3))
score=iou+.3*np.maximum(ncc,0)
win=score.argmax(1)
print('counts',dict(zip(names,np.bincount(win,minlength=4))))
for a,b in [(0,96),(96,150),(150,250),(250,350),(350,450),(450,550),(550,650),(650,700),(700,803)]:
 print(a,b,'counts',dict(zip(names,np.bincount(win[a:b],minlength=4))),'means',np.round(iou[a:b].mean(0),3),np.round(ncc[a:b].mean(0),3))
# print contiguous winner runs
chg=np.where(win[1:]"'!=win[:-1])[0]+1
print('"'runs')
st=0
for e in list(chg)+[F]:
 if e-st>=2: print(st,e-1,names[win[st]],'len',e-st,'base iou/ncc',round(iou[st:e,0].mean(),2),round(ncc[st:e,0].mean(),2),'win iou/ncc',round(iou[st:e,win[st]].mean(),2),round(ncc[st:e,win[st]].mean(),2))
 st=e
# print frames winner nonbase with margins top, restricted evidence strong
margin=score[np.arange(F),win]-score[:,0]
sel=np.where((win"'!=0)&(margin>.03))[0]
print('"'nonbase >.03',len(sel))
for i in sel: print(i,names[win[i]],'m',round(margin[i],2),'iou',np.round(iou[i],2),'ncc',np.round(ncc[i],2),'drprev',round(rel[i-1],1) if i else 0,'p',np.round(p[i],2))
PY"
```
<details><summary>output</summary>

```
med iou [0.607 0.371 0.372 0.605] med ncc [ 0.59  -0.447 -0.453  0.591]
counts {'base': 463, 'fx': 47, 'fy': 44, 'fz': 249}
0 96 counts {'base': 96, 'fx': 0, 'fy': 0, 'fz': 0} means [0.743 0.447 0.445 0.73 ] [ 0.675 -0.433 -0.452  0.677]
96 150 counts {'base': 21, 'fx': 5, 'fy': 0, 'fz': 28} means [0.64  0.413 0.41  0.636] [ 0.447 -0.374 -0.449  0.477]
150 250 counts {'base': 58, 'fx': 2, 'fy': 4, 'fz': 36} means [0.602 0.403 0.401 0.601] [ 0.509 -0.413 -0.414  0.49 ]
250 350 counts {'base': 33, 'fx': 27, 'fy': 22, 'fz': 18} means [0.264 0.246 0.246 0.261] [-0.002 -0.284 -0.255  0.007]
350 450 counts {'base': 54, 'fx': 11, 'fy': 11, 'fz': 24} means [0.283 0.186 0.185 0.278] [ 0.043 -0.464 -0.461 -0.052]
450 550 counts {'base': 49, 'fx': 2, 'fy': 7, 'fz': 42} means [0.511 0.358 0.36  0.514] …[+20208 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
S=np.load('scratch/pyrender_flips_scores.npz');iou=S['iou'];ncc=S['ncc'];F=len(iou)
# reward based on ncc differences plus slight iou
r=(ncc[:,3]-ncc[:,0])+.05*(iou[:,3]-iou[:,0])
# when both low quality, damp
conf=np.maximum(iou[:,0],iou[:,3])*np.maximum(0,np.maximum(ncc[:,0],ncc[:,3])+.2)
r=r*conf
for lam in [.02,.05,.1,.2,.35,.5,1.0]:
 # state 0/1, reward if state1. DP maximize reward - lam switches
 dp=np.zeros((F,2));back=np.zeros((F,2),int);dp[0]=[0,r[0]]
 for i in range(1,F):
  for s in [0,1]:
   vals=[dp[i-1,p]-(lam if p"'!=s else 0) for p in [0,1]];p=int(np.argmax(vals));dp[i,s]=vals[p]+(r[i] if s else 0);back[i,s]=p
 st=int(np.argmax(dp[-1]));path=np.zeros(F,int);path[-1]=st
 for i in range(F-1,0,-1):path[i-1]=back[i,path[i]]
 ch=np.where(path[1:]!=path[:-1])[0]+1;segs=[];st=0
 for e in list(ch)+[F]:segs.append((st,e-1,int(path[st]),float(r[st:e].mean())));st=e
 print('"'\\nlam',lam,'switches',len(ch),'fzfrac',path.mean())
 print('segs>=3',[(a,b,s,round(v,2)) for a,b,s,v in segs if b-a>=2])
PY"
```
<details><summary>output</summary>

```

lam 0.2 switches 6 fzfrac 0.47322540473225405
segs>=3 [(0, 170, 1, 0.0), (171, 522, 0, -0.01), (523, 572, 1, 0.02), (573, 585, 0, -0.03), (586, 686, 1, 0.02), (687, 744, 0, -0.05), (745, 802, 1, 0.06)]

lam 0.35 switches 4 fzfrac 0.48941469489414696
segs>=3 [(0, 170, 1, 0.0), (171, 522, 0, -0.01), (523, 686, 1, 0.01), (687, 744, 0, -0.05), (745, 802, 1, 0.06)]

lam 0.5 switches 4 fzfrac 0.48941469489414696
segs>=3 [(0, 170, 1, 0.0), (171, 522, 0, -0.01), (523, 686, 1, 0.01), (687, 744, 0, -0.05), (745, 802, 1, 0.06)]

lam 1.0 switches 3 fzfrac 0.2764632627646326
segs>=3 [(0, 522, 0, -0.0), (523, 686, 1, 0.01), (687, 744, 0, -0.05), (745, 802, 1, 0.06)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/static_eval.py <<'PY'
import numpy as np,trimesh,json,cv2,os
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=1536,1152
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100)
scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);mesh_node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
def render(P):
 scene.set_pose(mesh_node,pose=flip@Tcw@P);col,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);return col[...,:3]/255.,depth>1e-5
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
def avg_pose(idx,trans='median'):
 A=T[idx];Rs=A[:,:3,:3];U,S,Vt=np.linalg.svd(Rs.sum(0));R=U@Vt
 if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
 p=np.median(A[:,:3,3],0) if trans=='median' else A[:,:3,3].mean(0)
 P=np.eye(4);P[:3,:3]=R;P[:3,3]=p;return P
Rz=Rot.from_euler('z',18 …[+697 chars]
```
<details><summary>output</summary>

```

 init [[ 0.04343524 -0.16422535 -0.98546609]
 [ 0.99185579 -0.11112525  0.0622356 ]
 [-0.11973083 -0.98014347  0.15806111]] [0.21961028 1.23828268 2.41153216]
mean iou 0.7526 ncc 0.7127 area 3899
mean_fz iou 0.7247 ncc 0.6818 area 3884
f0 iou 0.7619 ncc 0.7161 area 3960
f0_fz iou 0.7307 ncc 0.6822 area 3944
f10 iou 0.7623 ncc 0.7164 area 3964
f10_fz iou 0.7309 ncc 0.6821 area 3945
f40 iou 0.7552 ncc 0.7153 area 3919
f40_fz iou 0.7262 ncc 0.6813 area 3902
f80 iou 0.7417 ncc 0.7104 area 3830
f80_fz iou 0.7167 ncc 0.6816 area 3800

 final [[ 0.23964365  0.73442826 -0.63496933]
 [ 0.97083992 -0.18558089  0.15175501]
 [-0.00638501 -0.6528207  -0.75748556]] [0.28508154 1.23800874 2.90459061]
mean iou 0.7428 ncc 0.6213 area 2089
mean_fz iou 0.7139 ncc 0.6455 area 2082
f710 iou 0.7447 ncc 0.6833  …[+279 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/static_eval.py';s=open(p).read();s=s.replace(\"for name,a,b in [('init',0,90),('final',710,803)]:\",\"for name,a,b in [('init',0,90),('final',746,803)]:\")
s=s.replace(\"for i in [a,a+10,a+40,b-10]:\",\"for i in [a,a+10,a+30,b-10]:\")
open(p,'w').write(s)
PY
python scratch/static_eval.py"
```
<details><summary>output</summary>

```

 init [[ 0.04343524 -0.16422535 -0.98546609]
 [ 0.99185579 -0.11112525  0.0622356 ]
 [-0.11973083 -0.98014347  0.15806111]] [0.21961028 1.23828268 2.41153216]
mean iou 0.7526 ncc 0.7127 area 3899
mean_fz iou 0.7247 ncc 0.6818 area 3884
f0 iou 0.7619 ncc 0.7161 area 3960
f0_fz iou 0.7307 ncc 0.6822 area 3944
f10 iou 0.7623 ncc 0.7164 area 3964
f10_fz iou 0.7309 ncc 0.6821 area 3945
f30 iou 0.7603 ncc 0.7161 area 3950
f30_fz iou 0.7301 ncc 0.6819 area 3934
f80 iou 0.7417 ncc 0.7104 area 3830
f80_fz iou 0.7167 ncc 0.6816 area 3800

 final [[ 0.33578715  0.69575698 -0.63495607]
 [ 0.93715814 -0.31459156  0.15088663]
 [-0.0947714  -0.64572004 -0.75767012]] [0.2848953  1.23843145 2.90550327]
mean iou 0.7445 ncc 0.6095 area 2092
mean_fz iou 0.7139 ncc 0.6351 area 2075
f746 iou 0.7313 ncc 0.5138  …[+279 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/static_opt.py <<'PY'
import numpy as np,trimesh,json,cv2,os
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.01,zfar=100)
scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);mesh_node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
def render(P):
 scene.set_pose(mesh_node,pose=flip@Tcw@P);col,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);return col[...,:3].astype(float)/255,depth>1e-5
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
def avg_pose(idx):
 A=T[idx];U,S,Vt=np.linalg.svd(A[:,:3,:3].sum(0));R=U@Vt
 if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
 P=np.eye(4);P[:3,:3]=R;P[:3,3]=np.median(A[:,:3,3],0);return P
def objective(P,om,pool):
 col,m=render(P);inter=m&om;union=m|om;iou=inter …[+1971 chars]
```
<details><summary>output</summary>

```

 init start (0.9398564587363429, 0.7155105222135619, 0.74781978840927)
roll180 (0.9227925576868927, 0.7054263565891473, 0.7245540036591512)
 trans 0.01 0 (0.9787681444268661, 0.7714514835605453, 0.6910555362210695) [0.21961028 1.23828268 2.40153216]
 trans 0.01 1 (0.9787681444268661, 0.7714514835605453, 0.6910555362210695) [0.21961028 1.23828268 2.40153216]
 trans 0.01 2 (0.9787681444268661, 0.7714514835605453, 0.6910555362210695) [0.21961028 1.23828268 2.40153216]
 trans 0.005 0 (0.9787681444268661, 0.7714514835605453, 0.6910555362210695) [0.21961028 1.23828268 2.40153216]
 trans 0.005 1 (0.9787681444268661, 0.7714514835605453, 0.6910555362210695) [0.21961028 1.23828268 2.40153216]
 trans 0.005 2 (0.9787681444268661, 0.7714514835605453, 0.6910555362210695) [0.21961028 1.23828268 2.401532 …[+2722 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices;n=np.array(json.load(open('inputs/ground_plane.json'))['plane'])
for name in ['init','final']:
 P=np.load(f'scratch/static_{name}_optpose.npy');v=(P[:3,:3]@V.T+P[:3,3:4]).T;d=v@n[:3]+n[3]
 print(name,'P',P,'floor dist min/quant',d.min(),np.quantile(d,[.001,.01,.05,.5]))
PY"
```
<details><summary>output</summary>

```
init P [[ 0.05858794 -0.08580044 -0.99458823  0.21761028]
 [ 0.97363638  0.22492667  0.03794992  1.23828268]
 [ 0.2204533  -0.97059068  0.09671642  2.40153216]
 [ 0.          0.          0.          1.        ]] floor dist min/quant -0.026831896330902882 [-0.02671699 -0.02578499 -0.02070655  0.01434616]
final P [[ 0.26555535  0.70494416 -0.65767324  0.2748953 ]
 [ 0.96404618 -0.18725518  0.18854831  1.23843145]
 [ 0.00976331 -0.68409739 -0.72932533  2.90550327]
 [ 0.          0.          0.          1.        ]] floor dist min/quant -0.0784660254727172 [-0.07755974 -0.07150941 -0.05691876 -0.02440391]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices;n=np.array(json.load(open('inputs/ground_plane.json'))['plane']);T=np.load('inputs/reconstruction.npz')['T_world_mesh']
for i in [0,10,40,80,746,756,776,800]:
 P=T[i];v=(P[:3,:3]@V.T+P[:3,3:4]).T;d=v@n[:3]+n[3];print(i,d.min(),np.quantile(d,[.001,.01,.05,.5]))
PY"
```
<details><summary>output</summary>

```
0 -0.015196303277350331 [-0.01500629 -0.01366884 -0.00816314  0.02183056]
10 -0.014947229550884344 [-0.01475721 -0.01341977 -0.00791406  0.02207963]
40 -0.022126486081612606 [-0.02193647 -0.02059902 -0.01509332  0.01490037]
80 -0.03762174862042533 [-0.03743173 -0.03609428 -0.03058858 -0.00059489]
746 -0.07321098980414087 [-0.07233433 -0.06589603 -0.05476196 -0.02809921]
756 -0.06915356444841758 [-0.06825021 -0.0622793  -0.05245859 -0.02498675]
776 -0.06621723202946139 [-0.06531388 -0.05934296 -0.04952226 -0.02205042]
800 -0.06564090390180777 [-0.06473755 -0.05876664 -0.04894593 -0.02147409]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_static_opt.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
cap=cv2.VideoCapture('inputs/video.mp4');frames=[]
for i in range(803):ok,im=cap.read();frames.append(cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA).astype(float)/255)
with h5py.File('inputs/masks.h5','r') as h:oms=h['object'][:]
def render(P):
 scene.set_pose(node,pose=flip@Tcw@P);c,d=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);return c[...,:3].astype(float)/255,d>1e-5
def sc(P,i):
 col,m=render(P);om=cv2.resize(oms[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);inter=m&om;union=m|om;iou=inter.sum()/max(1,union.sum());er=cv2.erode(om.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&m;x=col[er];y=frames[i][er]
 if len(x)<20:ncc=-1
 else:
  xx=[];yy=[]
  for c in range(3) …[+690 chars]
```
<details><summary>output</summary>

```
init opt [0.8103 0.7604] [0.813  0.7612] base [0.7441 0.6799] [0.7428 0.6823] improved iou/ncc 1.0 1.0
final opt [0.7987 0.7433] [0.8018 0.7432] base [0.7739 0.7359] [0.7756 0.7376] improved iou/ncc 1.0 0.7192982456140351

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/vis_pyrender_flips.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()}
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=1536,1152
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
for i in [118,195,230,265,270,272,297,315,320,324,330,331,332,337,340,345,355,357,361,362,365,369,372,383,390,400,405,412,445,451,457,485,490,491,523,550,557,566,586,605,640,646,659,668,681,714,744]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();ys,xs=np.where(om[i]);cx0,cy0=int(xs.mean()),int(ys.mean());half=180;x0=int(np.clip(cx0-half,0,W-2*half));y0=int(np.clip(cy0-half,0,H- …[+603 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
S=np.load('scratch/pyrender_flips_scores.npz');i=S['iou'];n=S['ncc'];names=S['names']
for f in [118,195,230,265,270,272,297,315,320,324,330,331,332,337,340,345,355,357,361,362,365,369,372,383,390,400,405,412,445,451,457,485,490,491,523,550,557,566,586,605,640,646,659,668,681,714,744]:
 print(f,' '.join(f'{names[j]}:{i[f,j]:.2f}/{n[f,j]:.2f}' for j in range(4)))
PY"
```
<details><summary>output</summary>

```
118 base:0.27/0.02 fx:0.28/0.03 fy:0.28/-0.01 fz:0.28/0.03
195 base:0.10/0.18 fx:0.13/-0.48 fy:0.14/-0.18 fz:0.10/0.16
230 base:0.42/-0.05 fx:0.45/0.37 fy:0.44/0.43 fz:0.42/-0.11
265 base:0.01/-1.00 fx:0.03/-1.00 fy:0.02/-1.00 fz:0.01/-1.00
270 base:0.00/-1.00 fx:0.02/-1.00 fy:0.02/-1.00 fz:0.00/-1.00
272 base:0.05/-0.22 fx:0.11/0.09 fy:0.11/0.15 fz:0.05/-0.23
297 base:0.09/0.04 fx:0.18/0.09 fy:0.18/0.19 fz:0.09/0.22
315 base:0.12/-0.40 fx:0.20/0.24 fy:0.20/0.42 fz:0.14/-0.46
320 base:0.38/0.09 fx:0.57/-0.03 fy:0.58/0.05 fz:0.38/0.04
324 base:0.26/-0.01 fx:0.56/-0.18 fy:0.57/-0.23 fz:0.27/0.09
330 base:0.17/-0.11 fx:0.37/0.06 fy:0.37/-0.09 fz:0.18/0.01
331 base:0.16/-0.21 fx:0.33/0.01 fy:0.33/-0.07 fz:0.17/-0.16
332 base:0.15/-0.36 fx:0.22/-0.19 fy:0.22/-0.18 fz:0.15/-0.38
337 base:0.09/-0 …[+2051 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/pyrender_all_flips_vis.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os,time
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);F=len(T);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()};names=list(flips)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
ious=np.zeros((F,4),np.float32);prec=np.zeros_like(ious);rec=np.zeros_like(ious);visn=np.zeros(F,np.int32)
for i in range(F):
 ok,im=cap.read();im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)/255;om=cv2.resize(omall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.r …[+727 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/pyrender_all_flips_vis.py", line 8
    cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
                                             ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/pyrender_all_flips_vis.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]\", \"cap=cv2.VideoCapture('inputs/video.mp4')\\nwith h5py.File('inputs/masks.h5','r') as h: omall=h['object'][:];pmall=h['person'][:]\")
open(p,'w').write(s)
PY
python scratch/pyrender_all_flips_vis.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
done

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
V=np.load('scratch/pyrender_flips_vis.npz');iou=V['iou'];prec=V['prec'];rec=V['rec'];visn=V['visn'];S=np.load('scratch/pyrender_flips_scores.npz');ncc=S['ncc'];names=list(V['names']);F=len(iou);r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'];rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
print('iou stats',np.round(np.median(iou,0),3),np.round(iou.mean(0),3),'ncc',np.round(np.median(ncc,0),3),np.round(ncc.mean(0),3))
for f in [100,118,130,150,195,230,250,265,270,272,297,315,320,324,330,331,332,337,340,345,355,357,361,362,365,369,372,383,390,400,405,412,445,451,457,485,490,491,523,550,557,566,586,605,640,646,659,668,681,714,744]:
 print(f,'visn',visn[f],' '.join(f'{names[j]}:{iou[f,j]:.2f}/{ncc[f,j]:.2f}' for j in range(4)))
PY"
```
<details><summary>output</summary>

```
iou stats [0.708 0.426 0.423 0.703] [0.589 0.378 0.377 0.585] ncc [ 0.59  -0.447 -0.453  0.591] [ 0.409 -0.429 -0.431  0.4  ]
100 visn 1146 base:0.80/0.63 fx:0.48/-0.43 fy:0.48/-0.52 fz:0.79/0.60
118 visn 1043 base:0.31/0.02 fx:0.30/0.03 fy:0.30/-0.01 fz:0.31/0.03
130 visn 986 base:0.79/0.34 fx:0.50/-0.40 fy:0.50/-0.51 fz:0.79/0.43
150 visn 847 base:0.88/0.49 fx:0.52/-0.47 fy:0.52/-0.54 fz:0.87/0.53
195 visn 905 base:0.09/0.18 fx:0.14/-0.48 fy:0.14/-0.18 fz:0.09/0.16
230 visn 987 base:0.44/-0.05 fx:0.51/0.37 fy:0.50/0.43 fz:0.44/-0.11
250 visn 719 base:0.71/0.40 fx:0.37/-0.38 fy:0.37/-0.48 fz:0.70/0.44
265 visn 110 base:0.00/-1.00 fx:0.01/-1.00 fy:0.01/-1.00 fz:0.00/-1.00
270 visn 158 base:0.00/-1.00 fx:0.00/-1.00 fy:0.00/-1.00 fz:0.00/-1.00
272 visn 172 base:0.01/-0.22 fx:0.10/0.09 fy:0.0 …[+2872 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/fit_translation_test.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);F=g.faces.astype(np.int32)
W,H=768,576;fx,fy=K[0,0]/2,K[1,1]/2;cx,cy=K[0,2]/2,K[1,2]/2;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(F).cuda()
cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def mask_batch(Ps):
 B=len(Ps);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None];x,y,z=v[:,:,0],v[:,:,1],v[:,:,2];u=fx*x/z+cx;vv=fy*y/z+cy;clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],2);rast,_=dr.rasterize(ctx,clip,Ft,(H,W));return (rast[...,3]>0).cpu().numpy()
def calc_metrics(m,om,pm):
 vo=om&~pm;vp=m&~pm;inter=vp&vo;union=vp|vo;return inter.sum()/max(1,union.sum()),vo,vp
for i in [195,230,265,272,297,315,320,324,331,340,355,361,365,372,383,400,412,445,451,457,485,490,491,523,557,566,605,640,646,659,668,681]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();om=cv2.resize(omall[i].astype(np.uint8),(W,H),interpolat …[+1604 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/fit_translation_test.py", line 5
    cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
                                             ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/fit_translation_test.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]\", \"cap=cv2.VideoCapture('inputs/video.mp4')\\nwith h5py.File('inputs/masks.h5','r') as h: omall=h['object'][:];pmall=h['person'][:]\")
open(p,'w').write(s)
PY
python scratch/fit_translation_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fit_translation_test.py", line 13, in <module>
    P=(Tcw@T[i]).astype(np.float32);m=mask_batch([P])[0];sc0,vo,vp=calc_metrics(m,om,pm)
                                      ^^^^^^^^^^^^^^^
  File "/work/scratch/fit_translation_test.py", line 8, in mask_batch
    B=len(Ps);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None];x,y,z=v[:,:,0],v[:,:,1],v[:,:,2];u=fx*x/z+cx;vv=fy*y/z+cy;clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],2);rast,_=dr.rasterize(ctx,clip,Ft,(H,W));return (rast[...,3]>0).cpu().numpy()
                                  ~~^^^^^^^^^
TypeError: list indices must be integers or slices, not tuple

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/B=len(Ps);v=/Ps=np.asarray(Ps);B=len(Ps);v=/" scratch/fit_translation_test.py
python scratch/fit_translation_test.py'
```
<details><summary>output</summary>

```
195 sc0 0.098 sc1 0.079 dpos [ 0.017 -0.032  0.09 ] target vis 905
230 sc0 0.448 sc1 0.831 dpos [-0.002 -0.128 -0.027] target vis 987
265 sc0 0.0 sc1 0.167 dpos [ 0.082  0.058 -0.006] target vis 110
272 sc0 0.009 sc1 0.187 dpos [ 0.067 -0.092 -0.061] target vis 172
297 sc0 0.095 sc1 0.032 dpos [-0.002  0.071  0.029] target vis 449
315 sc0 0.065 sc1 0.304 dpos [-0.042 -0.037 -0.057] target vis 202
320 sc0 0.393 sc1 0.343 dpos [-0.011  0.031  0.09 ] target vis 362
324 sc0 0.362 sc1 0.277 dpos [-0.049  0.13  -0.031] target vis 375
331 sc0 0.193 sc1 0.352 dpos [ 0.007 -0.089  0.032] target vis 325
340 sc0 0.002 sc1 0.092 dpos [-0.146 -0.063 -0.081] target vis 387
355 sc0 0.013 sc1 0.217 dpos [ 0.046 -0.047  0.029] target vis 164
361 sc0 0.0 sc1 0.0 dpos [0.005 0.002 0.09 ] target vis 95
365 sc …[+1325 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/joint_fit_test.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr,os,time
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);F=g.faces.astype(np.int32);W,H=768,576;fx,fy=K[0,0]/2,K[1,1]/2;cx,cy=K[0,2]/2,K[1,2]/2;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(F).cuda()
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()}
def masks(Ps):
 Ps=np.asarray(Ps,np.float32);B=len(Ps);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None];x,y,z=v[:,:,0],v[:,:,1],v[:,:,2];u=fx*x/z+cx;vv=fy*y/z+cy;clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],2);rast,_=dr.rasterize(ctx,clip,Ft,(H,W));return (rast[...,3]>0).cpu().numpy()
def fit(P0,om,pm):
 base=P0.copy();best=base.copy();bm=masks([best])[0];vo=om&~pm;vp=bm&~pm;bsc=(vo&vp).sum()/max(1,(vo|vp).sum())
 for step in [.08,.04,.02,.01,.005,.002]:
  for ax in range(3):
   Ps=[]
   for d in [-step,0 …[+1222 chars]
```
<details><summary>output</summary>

```
118 base:0.392,z2.55 fx:0.357,z2.43 fy:0.356,z2.42 fz:0.392,z2.57
130 base:0.866,z2.43 fx:0.569,z2.42 fy:0.565,z2.44 fz:0.871,z2.42
150 base:0.905,z2.42 fx:0.620,z2.43 fy:0.600,z2.40 fz:0.900,z2.43
195 base:0.130,z2.22 fx:0.174,z2.18 fy:0.176,z2.17 fz:0.133,z2.22
230 base:0.841,z2.49 fx:0.580,z2.56 fy:0.575,z2.55 fz:0.836,z2.49
250 base:0.757,z2.59 fx:0.385,z2.55 fy:0.386,z2.54 fz:0.735,z2.58
265 base:0.175,z2.58 fx:0.199,z2.68 fy:0.182,z2.63 fz:0.175,z2.58
270 base:0.151,z2.44 fx:0.206,z2.55 fy:0.206,z2.57 fz:0.159,z2.43
272 base:0.201,z2.52 fx:0.236,z2.66 fy:0.239,z2.66 fz:0.204,z2.52
297 base:0.239,z2.85 fx:0.208,z2.88 fy:0.210,z2.83 fz:0.235,z2.89
315 base:0.392,z3.03 fx:0.403,z2.98 fy:0.386,z2.83 fz:0.391,z3.01
320 base:0.568,z2.96 fx:0.874,z3.14 fy:0.877,z3.14 fz:0.541,z3.02
324 base …[+2524 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/fit_all_candidates.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr,os,time
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);F=len(T);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);Faces=g.faces.astype(np.int32);W,H=768,576;fx,fy=K[0,0]/2,K[1,1]/2;cx,cy=K[0,2]/2,K[1,2]/2
ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(Faces).cuda()
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()};names=list(flips)
mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(fx,fy,cx,cy,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def masks(Ps):
 Ps=np.asarray(Ps,np.float32);B=len(Ps);v=(torch.from …[+2694 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/fit_all_candidates.py", line 9
    cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
                                             ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/fit_all_candidates.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]\", \"cap=cv2.VideoCapture('inputs/video.mp4')\\nwith h5py.File('inputs/masks.h5','r') as h: omall=h['object'][:];pmall=h['person'][:]\")
open(p,'w').write(s)
PY
python scratch/fit_all_candidates.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fit_all_candidates.py", line 39, in <module>
    Rw=T[i,:3,:3]@Rf;P0=np.eye(4,np.float32);P0[:3,:3]=Tcw[:3,:3]@Rw;P0[:3,3]=(Tcw@T[i])[:3,3]
                        ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/twodim_base.py", line 211, in eye
    m = zeros((N, M), dtype=dtype, order=order)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'type' object cannot be interpreted as an integer

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/np.eye(4,np.float32)/np.eye(4,dtype=np.float32)/' scratch/fit_all_candidates.py
python scratch/fit_all_candidates.py"
```
<details><summary>output</summary>

```
0 elapsed 1.1
40 elapsed 20.9
80 elapsed 40.4
120 elapsed 59.2
160 elapsed 79.4
200 elapsed 98.0
240 elapsed 117.9
280 elapsed 136.6
320 elapsed 156.7
360 elapsed 176.8
400 elapsed 193.3
440 elapsed 211.0
480 elapsed 228.4
520 elapsed 246.8
560 elapsed 264.6
600 elapsed 283.3
640 elapsed 310.2
680 elapsed 345.2
720 elapsed 363.2
760 elapsed 380.7
800 elapsed 398.3
done 399.31892228126526

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/fit_candidates_all.npz');i=D['ioupr'];n=D['ncc'];iv=D['iouv'];P=D['Pw'];names=list(D['names']);vs=D['vsizes'];F=len(i)
score=i+.35*np.maximum(n,-.5)
win=score.argmax(1);margin=score[np.arange(F),win]-score[:,0]
for f in [100,118,130,150,195,230,250,265,270,272,297,315,320,324,330,331,332,337,340,345,355,357,361,362,365,369,372,383,390,400,405,412,445,451,457,485,490,491,523,550,557,566,586,605,640,646,659,668,681,714,744]:
 print(f,'vis',vs[f],' '.join(f'{names[j]}:{i[f,j]:.2f}/{n[f,j]:.2f}={score[f,j]:.2f}' for j in range(4)),'win',names[win[f]])
print('base score mean',score[:,0].mean(),'best',score[:,win].mean(),'counts',dict(zip(names,np.bincount(win,minlength=4))))
# distribution improvements
print('improved frames',np.sum(score[:,win]>score[:,0]+.02),'max')
PY"
```
<details><summary>output</summary>

```
100 vis 1146 base:0.85/0.62=1.07 fx:0.53/-0.22=0.45 fy:0.53/-0.30=0.42 fz:0.84/0.63=1.05 win base
118 vis 1043 base:0.40/-0.21=0.32 fx:0.36/0.17=0.42 fy:0.36/0.22=0.43 fz:0.40/-0.04=0.39 win fy
130 vis 986 base:0.86/0.30=0.96 fx:0.57/-0.15=0.52 fy:0.55/-0.41=0.41 fz:0.85/0.43=1.00 win fz
150 vis 847 base:0.90/0.50=1.07 fx:0.62/-0.25=0.53 fy:0.60/-0.45=0.44 fz:0.89/0.53=1.08 win fz
195 vis 905 base:0.13/-0.25=0.04 fx:0.18/-0.44=0.02 fy:0.17/-0.25=0.09 fz:0.13/-0.39=-0.00 win fy
230 vis 987 base:0.84/0.77=1.11 fx:0.57/0.13=0.62 fy:0.57/0.15=0.62 fz:0.71/0.29=0.81 win base
250 vis 719 base:0.75/0.52=0.93 fx:0.38/-0.41=0.24 fy:0.38/-0.49=0.21 fz:0.73/0.52=0.91 win base
265 vis 110 base:0.18/-0.17=0.12 fx:0.20/-0.19=0.14 fy:0.18/-0.36=0.06 fz:0.17/0.34=0.29 win fz
270 vis 158 base:0.16/-0.40=0. …[+4207 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/fit_candidates_all.npz');i=D['ioupr'];n=D['ncc'];score=i+.35*np.maximum(n,-.5);print(score.shape,score[:,0].mean(),score.max(1).mean(),np.max(score-score[:,0,None]))
print(score[:3])
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
D=np.load('scratch/fit_candidates_all.npz');F=len(D['Pw']);i=D['ioupr'];n=D['ncc'];B=np.load('scratch/pyrender_flips_vis.npz')['iou']
print('baseline iou mean/med',B[:,0].mean(),np.median(B[:,0]),'fit base',i[:,0].mean(),np.median(i[:,0]))
print('improved',np.mean(i[:,0]>B[:,0]+.02),'worse',np.mean(i[:,0]<B[:,0]-.02))
for a,b in [(0,90),(90,120),(120,250),(250,350),(350,450),(450,550),(550,650),(650,746),(746,803)]:print(a,b,'base',round(B[a:b,0].mean(),3),'fit',round(i[a:b,0].mean(),3),'ncc',round(n[a:b,0].mean(),3))
# flips conditional improvements over fitted base
for j in [1,2,3]:
 d=i[:,j]-i[:,0];print('\\nflip',j,'mean d',d.mean(),'improved>.1',np.sum(d>.1),'improved>.2',np.sum(d>.2),'worse>.1',np.sum(d<-.1))
 idx=np.argsort(d)[-20:][::-1];print([(int(x),round(float(d[x]),2),round(float(i[x,j]),2),round(float(i[x,0]),2)) for x in idx])
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
D=np.load('scratch/fit_candidates_all.npz');P=D['Pw'][:,0];r=np.load('inputs/reconstruction.npz');B=r['T_world_mesh'];names=list(D['names']);
for tag,A in [('base',B),('fit',P)]:
 p=A[:,:3,3];dp=np.linalg.norm(np.diff(p,axis=0),axis=1);rel=Rot.from_matrix(A[:-1,:3,:3].transpose(0,2,1)@A[1:,:3,:3]).magnitude()*180/np.pi
 print(tag,'step trans q',np.quantile(dp,[0,.25,.5,.75,.9,.95,.99,1]),'rot',np.quantile(rel,[0,.25,.5,.75,.9,.95,.99,1]))
 print('max trans',np.argsort(dp)[-20:][::-1])
 print('big rot',[(int(i),round(float(rel[i]),1)) for i in np.argsort(rel)[-20:][::-1]])
PY"
```
<details><summary>output</summary>

```
base step trans q [0.         0.00503373 0.01821648 0.03599913 0.05626993 0.0781163
 0.11324453 0.15988636] rot [6.85013526e-15 1.21703123e-03 6.27861918e+00 1.30171671e+01
 2.49387413e+01 2.99573518e+01 7.13357539e+01 1.63250874e+02]
max trans [484 489 331 671 303 488 265 491 459 450 658 313 542 545 362 314 232 441
 226 302]
big rot [(331, 163.3), (744, 145.0), (566, 140.6), (265, 137.4), (492, 84.2), (447, 83.5), (319, 81.2), (658, 78.1), (473, 71.3), (456, 70.2), (195, 70.1), (604, 69.4), (476, 68.4), (411, 66.0), (601, 59.2), (469, 58.0), (360, 57.0), (356, 53.8), (169, 52.6), (324, 50.3)]
fit step trans q [0.         0.01129888 0.03112776 0.0642504  0.12124451 0.15510027
 0.21776469 0.30931556] rot [2.03250400e-15 1.21830544e-03 6.27861762e+00 1.30171677e+01
 2.49387422e+01 2.99573513 …[+402 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.signal import savgol_filter, medfilt
D=np.load('scratch/fit_candidates_all.npz');P=D['Pw'][:,0];p=P[:,:3,3].copy();iou=D['ioupr'][:,0];vis=D['vsizes'];F=len(p)
mask=np.zeros(F,bool);mask[90:746]=1
for win in [5,7,9,11,15]:
 q=p.copy()
 for k in range(3):
  a=savgol_filter(p[:,k],win,2,mode='interp');q[mask,k]=a[mask]
 dp=np.linalg.norm(np.diff(q[mask],axis=0),axis=1);raw=np.linalg.norm(np.diff(p[mask],axis=0),axis=1)
 print('win',win,'step q',np.round(np.quantile(dp,[.5,.9,.95,.99,1]),3),'raw',np.round(np.quantile(raw,[.5,.9,.95,.99,1]),3))
# qual bins compare raw
for a,b in [(90,120),(120,250),(250,350),(350,450),(450,550),(550,650),(650,746)]:
 print(a,b,'iou',iou[a:b].mean(),'vis',np.median(vis[a:b]),'raw step',np.linalg.norm(np.diff(p[a:b],axis=0),axis=1).mean())
PY"
```
<details><summary>output</summary>

```
win 5 step q [0.032 0.073 0.09  0.122 0.218] raw [0.042 0.132 0.163 0.223 0.309]
win 7 step q [0.025 0.057 0.068 0.089 0.118] raw [0.042 0.132 0.163 0.223 0.309]
win 9 step q [0.022 0.048 0.057 0.073 0.089] raw [0.042 0.132 0.163 0.223 0.309]
win 11 step q [0.02  0.044 0.052 0.066 0.078] raw [0.042 0.132 0.163 0.223 0.309]
win 15 step q [0.017 0.038 0.044 0.061 0.072] raw [0.042 0.132 0.163 0.223 0.309]
90 120 iou 0.7892881 vis 1145.5 raw step 0.027655402
120 250 iou 0.77072483 vis 818.0 raw step 0.050596237
250 350 iou 0.5166715 vis 345.5 raw step 0.085307874
350 450 iou 0.6445291 vis 175.5 raw step 0.07022789
450 550 iou 0.6959525 vis 476.5 raw step 0.05517182
550 650 iou 0.71386766 vis 502.0 raw step 0.05889532
650 746 iou 0.75540876 vis 434.5 raw step 0.038754504

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_smooth_trans.py <<'PY'
import numpy as np,cv2,h5py,torch,trimesh,json,nvdiffrast.torch as dr
from scipy.signal import savgol_filter
D=np.load('scratch/fit_candidates_all.npz');P=D['Pw'][:,0].astype(np.float32);F=len(P);r=np.load('inputs/reconstruction.npz');B=r['T_world_mesh'].astype(np.float32);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Tcw=np.linalg.inv(Twc).astype(np.float32)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);Ft0=g.faces.astype(np.int32);W,H=768,576;fx,fy=K[0,0]/2,K[1,1]/2;cx,cy=K[0,2]/2,K[1,2]/2;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(Ft0).cuda()
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def masks(Ps):
 Ps=np.asarray(Ps,np.float32);B=len(Ps);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None];x,y,z=v[:,:,0],v[:,:,1],v[:,:,2];u=fx*x/z+cx;vv=fy*y/z+cy;clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],2);rast,_=dr.rasterize(ctx,clip,Ft,(H,W));return (rast[...,3]>0).cpu().numpy()
poses={}
poses['baseline']=B
poses['fitraw']=P
for win in [5,7,9,11,15,21]:
 Q=P.copy()
 for k in range(3):Q[90:746,k,3]=savgol_filter(P[90:746,k,3],win,2,mode='interp')
 poses[f'sg{win}']=Q
# evaluate in chunks of 32
res={k:np.zeros(F) for k in poses}
 …[+968 chars]
```
<details><summary>output</summary>

```
baseline mean 0.5616326699389877 med 0.6579066315860487 <.5 228
fitraw mean 0.7063594022275346 med 0.786185864359239 <.5 115
sg5 mean 0.6812659793789212 med 0.7642809721331661 <.5 131
sg7 mean 0.6758722675980158 med 0.7587715698927939 <.5 136
sg9 mean 0.6701149526826272 med 0.7581644894694748 <.5 140
sg11 mean 0.6657013496746016 med 0.7545923149015932 <.5 147
sg15 mean 0.6587974299368009 med 0.7489432663325546 <.5 142
sg21 mean 0.6438420287758613 med 0.734027090549233 <.5 158
90 120 {'baseline': 0.758, 'fitraw': 0.802, 'sg5': 0.795, 'sg7': 0.793, 'sg9': 0.791, 'sg11': 0.788, 'sg15': 0.785, 'sg21': 0.783}
120 250 {'baseline': 0.678, 'fitraw': 0.785, 'sg5': 0.765, 'sg7': 0.76, 'sg9': 0.758, 'sg11': 0.758, 'sg15': 0.753, 'sg21': 0.741}
250 350 {'baseline': 0.323, 'fitraw': 0.528, 'sg5': 0.482 …[+595 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# adapt eval script by generating blend poses? easiest modify to add blends then run
p='scratch/eval_smooth_trans.py';s=open(p).read()
insert=\"\"\"
q=np.stack([savgol_filter(P[90:746,k,3],5,2,mode='interp') for k in range(3)],1)
sm=P.copy();sm[90:746,:3,3]=q
conf=np.clip((D['ioupr'][:,0]-.45)/.35,0,1)*np.clip(D['vsizes']/300.,0,1)
for a in [0,.25,.5,.75,1.0]:
 Q=sm.copy();Q[90:746,:3,3]=(a*conf[90:746,None]*P[90:746,:3,3]+(1-a*conf[90:746,None])*sm[90:746,:3,3]);poses[f'blend{a}']=Q
\"\"\"
s=s.replace(\"poses['fitraw']=P\\n\", \"poses['fitraw']=P\\n\"+insert)
open('scratch/eval_smooth_trans_blend.py','w').write(s)
PY
python scratch/eval_smooth_trans_blend.py"
```
<details><summary>output</summary>

```
baseline mean 0.5616326699389877 med 0.6579066315860487 <.5 228
fitraw mean 0.7063594022275346 med 0.786185864359239 <.5 115
blend0 mean 0.6812659793789212 med 0.7642809721331661 <.5 131
blend0.25 mean 0.6848924932230274 med 0.7726539589442816 <.5 128
blend0.5 mean 0.6878989565370867 med 0.7775398524863193 <.5 126
blend0.75 mean 0.6911165620837485 med 0.7814162372928561 <.5 126
blend1.0 mean 0.6940894000128837 med 0.7841668502607035 <.5 125
sg5 mean 0.6812659793789212 med 0.7642809721331661 <.5 131
sg7 mean 0.6758722675980158 med 0.7587715698927939 <.5 136
sg9 mean 0.6701149526826272 med 0.7581644894694748 <.5 140
sg11 mean 0.6657013496746016 med 0.7545923149015932 <.5 147
sg15 mean 0.6587974299368009 med 0.7489432663325546 <.5 142
sg21 mean 0.6438420287758613 med 0.734027090549233 <.5 158 …[+1576 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/fit_candidates_all.npz');i=D['ioupr'];n=D['ncc'];names=list(D['names']);F=len(i)
for j in [1,2,3]:
 d=i[:,j]-i[:,0]; print('\\n',names[j])
 for th in [.03,.05,.08,.12]:
  m=(d>th)&(i[:,j]>.55)
  # runs
  ch=np.where(np.diff(np.r_[False,m,False].astype(int))"'!=0)[0];runs=[(ch[k],ch[k+1]-1) for k in range(0,len(ch),2)]
  print(th,[(a,b,round(float(d[a:b+1].mean()),2)) for a,b in runs if b-a>=1])
# frame list 180-380 where max flip diff
for i0 in range(180,381):
 if max(i[i0,1]-i[i0,0],i[i0,2]-i[i0,0])>.03:
  print(i0,'"'base',round(float(i[i0,0]),2),'fx',round(float(i[i0,1]),2),'fy',round(float(i[i0,2]),2),'fz',round(float(i[i0,3]),2),'ncc',np.round(n[i0],2))
PY"
```
<details><summary>output</summary>

```

 fx
0.03 [(320, 324, 0.29), (367, 370, 0.14), (401, 402, 0.08)]
0.05 [(320, 324, 0.29), (367, 370, 0.14), (401, 402, 0.08)]
0.08 [(320, 324, 0.29), (367, 370, 0.14)]
0.12 [(320, 324, 0.29)]

 fy
0.03 [(320, 324, 0.3), (348, 349, 0.06), (369, 370, 0.18), (401, 402, 0.08)]
0.05 [(320, 324, 0.3), (369, 370, 0.18), (401, 402, 0.08)]
0.08 [(320, 324, 0.3), (369, 370, 0.18)]
0.12 [(320, 324, 0.3)]

 fz
0.03 [(184, 186, 0.05), (190, 192, 0.07), (197, 198, 0.07), (282, 283, 0.05), (345, 346, 0.05), (368, 369, 0.19), (382, 383, 0.08), (386, 387, 0.07), (439, 440, 0.04), (596, 597, 0.05), (623, 625, 0.04), (704, 706, 0.04), (708, 710, 0.04)]
0.05 [(345, 346, 0.05), (368, 369, 0.19), (386, 387, 0.07)]
0.08 [(368, 369, 0.19)]
0.12 [(368, 369, 0.19)]
193 base 0.34 fx 0.37 fy 0.38 fz 0.34 ncc [-0.21  0 …[+3515 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('scratch/fit_candidates_all.npz');P=D['Pw'];i=D['ioupr'];n=D['ncc'];names=list(D['names']);F=len(P);e=i+.35*np.maximum(n,-.5)
# angle candidate transition from candidate a at t-1 to b at t
ang=np.zeros((F-1,4,4))
for t in range(F-1):
 for a in range(4):
  for b in range(4):ang[t,a,b]=Rot.from_matrix(P[t,a,:3,:3].T@P[t+1,b,:3,:3]).magnitude()*180/np.pi
for alpha in [0,.01,.02,.03,.05,.08,.12,.2,.3,.5]:
 dp=np.zeros((F,4));bk=np.zeros((F,4),int);dp[0]=e[0]
 for t in range(1,F):
  for b in range(4):
   vals=dp[t-1]-alpha*ang[t-1,:,b]/180; a=int(np.argmax(vals));dp[t,b]=vals[a]+e[t,b];bk[t,b]=a
 st=int(np.argmax(dp[-1]));path=np.zeros(F,int);path[-1]=st
 for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
 ch=np.where(path[1:]"'!=path[:-1])[0]+1;runs=[];s=0
 for z in list(ch)+[F]:runs.append((s,z-1,names[path[s]],round(float(e[s:z,path[s]].mean()),2),round(float(e[s:z,0].mean()),2)));s=z
 print('"'\\nalpha',alpha,'counts',dict(zip(names,np.bincount(path,minlength=4))),'runs>=2',[(a,b,nm,x,y) for a,b,nm,x,y in runs if b-a>=1])
PY"
```
<details><summary>output</summary>

```

alpha 0 counts {'base': 407, 'fx': 55, 'fy': 52, 'fz': 289} runs>=2 [(0, 97, 'base', 1.05, 1.05), (99, 104, 'base', 1.05, 1.05), (105, 106, 'fz', 1.04, 1.03), (107, 109, 'base', 1.06, 1.06), (111, 115, 'base', 1.04, 1.04), (118, 123, 'fy', 0.42, 0.31), (124, 125, 'fx', 0.46, 0.34), (130, 131, 'fz', 1.01, 0.97), (136, 138, 'base', 1.09, 1.09), (139, 151, 'fz', 1.07, 1.04), (153, 154, 'fz', 1.08, 1.07), (156, 157, 'fz', 1.02, 1.01), (158, 160, 'base', 1.1, 1.1), (162, 164, 'base', 1.09, 1.09), (167, 170, 'fz', 0.88, 0.85), (171, 178, 'base', 1.04, 1.04), (181, 186, 'fz', 1.14, 1.12), (187, 189, 'base', 1.11, 1.11), (190, 192, 'fz', 1.07, 1.02), (193, 195, 'fy', 0.34, 0.17), (197, 198, 'fz', 1.02, 0.94), (199, 200, 'base', 0.65, 0.65), (201, 202, 'fz', 0.44, 0.33), (204, 205, 'fz', 0.85, 0.8 …[+20602 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz');R=r['T_world_mesh'][:,:3,:3].astype(float);F=len(R)
# interpolate quaternions between i-2,i+2 via slerp using Rotation
errs=np.zeros(F)
for i in range(2,F-2):
 q0=Rot.from_matrix(R[i-2]);q2=Rot.from_matrix(R[i+2]);q=q0*q0.inv()*q2 # wrong? use Slerp
 from scipy.spatial.transform import Slerp
 sl=Slerp([0,1],Rot.concatenate([q0,q2]));qe=sl(.5)
 errs[i]=(qe.inv()*Rot.from_matrix(R[i])).magnitude()*180/np.pi
for i in np.argsort(errs)[-80:][::-1]:print(i,round(errs[i],1))
PY"
```
<details><summary>output</summary>

```
745 125.5
746 119.1
568 94.3
566 82.8
332 82.3
330 81.6
333 81.0
331 81.0
744 75.7
267 75.1
743 72.5
265 70.8
475 70.5
565 70.3
264 69.6
266 65.1
567 60.0
605 52.0
494 49.2
319 44.1
658 44.1
448 44.1
558 43.7
492 42.8
491 42.1
446 41.8
644 41.5
320 40.9
318 40.9
98 40.6
321 40.6
493 40.5
169 39.9
194 39.8
410 39.6
360 38.7
263 38.0
411 37.8
647 37.5
603 37.4
606 36.7
474 36.3
449 35.7
472 35.7
473 35.7
476 35.3
470 35.2
456 35.1
196 35.0
298 34.2
358 34.0
468 33.8
660 33.7
359 33.4
361 33.4
171 33.1
412 33.0
659 33.0
447 32.8
477 31.8
560 31.8
325 31.2
323 31.2
296 31.0
469 30.6
203 30.5
664 30.3
195 30.0
588 30.0
455 29.9
478 29.9
600 29.6
601 29.6
174 29.5
246 29.2
471 29.0
213 28.9
657 28.9
556 28.7
362 28.5

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/make_fit_montage.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
from scipy.signal import savgol_filter
from PIL import Image
D=np.load('scratch/fit_candidates_all.npz');P=D['Pw'][:,0].astype(np.float64);iou=D['ioupr'][:,0];vis=D['vsizes'];r=np.load('inputs/reconstruction.npz');B=r['T_world_mesh'].astype(np.float64);F=len(P)
q=np.stack([savgol_filter(P[90:746,k,3],5,2,mode='interp') for k in range(3)],1);sm=P.copy();sm[90:746,:3,3]=q
conf=np.clip((iou-.45)/.35,0,1)*np.clip(vis/300.,0,1)
P[90:746,:3,3]=conf[90:746,None]*P[90:746,:3,3]+(1-conf[90:746,None])*sm[90:746,:3,3]
cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices
cap=cv2.VideoCapture('inputs/video.mp4');frames=[]
for i in range(F):ok,im=cap.read();frames.append(im)
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
def proj(T):
 vc=(Tcw@T)[:3,:3]@V.T+(Tcw@T)[:3,3:4];uv=(K@vc).T;return (uv[:,:2]/uv[:,2:3]).astype(np.float32)
for part,(a,b) in enumerate([(88,420),(420,748)]):
 thumbs=[]
 for i in range(a,b,8):
  im=frames[i].copy();ys,xs=np.where(om[i]);cx0,cy0=(xs.mean(),ys.mean()) if len(xs) else (768,576);half=180;x0=int(np.clip(cx0-half,0,1536-2*half));y0=int(np.clip(cy0-half,0,1152-2*half));crop=im[y0:y0+2*half,x0:x0+2*half].copy()
  for T,col in [(B[i],(0,0,255)),(P[i],(0,255,0))]:
   uv=proj(T)-np.array([x0, …[+702 chars]
```
<details><summary>output</summary>

```
0 (1500, 2700) 42
1 (1500, 2700) 41

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/eval_final_rotations.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os,time
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
D=np.load('scratch/fit_candidates_all.npz');Pfit=D['Pw'][:,0].astype(np.float64);ioufit=D['ioupr'][:,0];vis=D['vsizes'];F=len(Pfit)
# adaptive blend: trust the raw mask fit where visibility/evidence is high.
q=np.stack([savgol_filter(Pfit[90:746,k,3],5,2,mode='interp') for k in range(3)],1);sm=Pfit.copy();sm[90:746,:3,3]=q
conf=np.clip((ioufit-.45)/.35,0,1)*np.clip(vis/300.,0,1);Pbase=Pfit.copy();Pbase[90:746,:3,3]=conf[90:746,None]*Pfit[90:746,:3,3]+(1-conf[90:746,None])*sm[90:746,:3,3]
r=np.load('inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float);Pbase[0:90]=B[0:90].astype(float);Pbase[746:]=B[746:].astype(float)
cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
flips={'base':np.eye(3),'fx':Rot.from_euler('x',180,degrees=True).as_matrix(),'fy':Rot.from_euler('y',180,degrees=True).as_matrix(),'fz':Rot.from_euler('z',180,degrees=True).as_matrix()};names=list(flips)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576
renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K …[+1668 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/eval_final_rotations.py", line 14
    cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
                                             ^^^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/eval_final_rotations.py';s=open(p).read();s=s.replace(\"cap=cv2.VideoCapture('inputs/video.mp4');with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]\", \"cap=cv2.VideoCapture('inputs/video.mp4')\\nwith h5py.File('inputs/masks.h5','r') as h: omall=h['object'][:];pmall=h['person'][:]\")
open(p,'w').write(s)
PY
python scratch/eval_final_rotations.py"
```
<details><summary>output</summary>

```
0 0.5
100 22.5
200 56.7
300 168.3
400 229.6
500 251.8
600 274.2
700 307.0
800 387.0
done 387.5066637992859

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/final_rot_scores.npz');i=D['iou'];n=D['ncc'];names=list(D['names']);F=len(i);score=i+.4*np.maximum(n,-.5)
print('means',np.round(i.mean(0),3),np.round(n.mean(0),3),np.round(score.mean(0),3))
for f in [90,100,118,130,150,195,230,250,265,270,272,297,315,320,324,330,331,332,337,340,345,355,357,361,362,365,369,372,383,390,400,405,412,445,451,457,485,490,491,523,550,557,566,586,605,640,646,659,668,681,714,744,745]:
 print(f,' '.join(f'{names[j]}:{i[f,j]:.2f}/{n[f,j]:.2f}={score[f,j]:.2f}' for j in range(4)))
PY"
```
<details><summary>output</summary>

```
means [0.698 0.401 0.401 0.69 ] [ 0.406 -0.371 -0.382  0.417] [0.863 0.28  0.275 0.861]
90 base:0.82/0.61=1.06 fx:0.45/-0.42=0.28 fy:0.46/-0.51=0.26 fz:0.79/0.63=1.04
100 base:0.85/0.62=1.10 fx:0.48/-0.41=0.32 fy:0.48/-0.48=0.29 fz:0.84/0.61=1.09
118 base:0.39/-0.24=0.30 fx:0.26/0.34=0.39 fy:0.26/0.29=0.37 fz:0.40/-0.11=0.35
130 base:0.86/0.30=0.98 fx:0.50/-0.30=0.38 fy:0.51/-0.40=0.35 fz:0.86/0.39=1.01
150 base:0.90/0.50=1.10 fx:0.53/-0.43=0.36 fy:0.53/-0.51=0.33 fz:0.90/0.52=1.11
195 base:0.10/0.28=0.21 fx:0.08/-0.20=0.00 fy:0.09/0.18=0.16 fz:0.10/-0.12=0.05
230 base:0.84/0.77=1.14 fx:0.50/-0.30=0.38 fy:0.50/-0.26=0.39 fz:0.83/0.73=1.12
250 base:0.75/0.49=0.95 fx:0.37/-0.42=0.21 fy:0.37/-0.49=0.18 fz:0.74/0.56=0.97
265 base:0.14/-0.18=0.07 fx:0.13/-0.30=0.00 fy:0.13/-0.27=0.02 fz:0.14/0. …[+3563 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/final_rot_scores.npz');i=D['iou'];n=D['ncc'];F=len(i)
for a in range(90,750,20):
 b=min(750,a+20);print(f'{a:3d}-{b-1:3d}','dn_fz',round(float((n[a:b,3]-n[a:b,0]).mean()),3),'di_fz',round(float((i[a:b,3]-i[a:b,0]).mean()),3),'nbase',round(float(n[a:b,0].mean()),2),'nfz',round(float(n[a:b,3].mean()),2),'nmaxxy',round(float(n[a:b,1:3].mean()),2),'imaxxy',round(float(i[a:b,1:3].mean()),2))
PY"
```
<details><summary>output</summary>

```
 90-109 dn_fz 0.006 di_fz -0.016 nbase 0.66 nfz 0.66 nmaxxy -0.45 imaxxy 0.47
110-129 dn_fz 0.075 di_fz -0.01 nbase 0.21 nfz 0.28 nmaxxy -0.12 imaxxy 0.4
130-149 dn_fz 0.081 di_fz -0.006 nbase 0.39 nfz 0.47 nmaxxy -0.38 imaxxy 0.53
150-169 dn_fz 0.01 di_fz -0.006 nbase 0.44 nfz 0.45 nmaxxy -0.68 imaxxy 0.48
170-189 dn_fz -0.054 di_fz 0.008 nbase 0.67 nfz 0.62 nmaxxy -0.48 imaxxy 0.5
190-209 dn_fz -0.039 di_fz -0.004 nbase 0.56 nfz 0.52 nmaxxy -0.53 imaxxy 0.34
210-229 dn_fz -0.013 di_fz -0.024 nbase 0.62 nfz 0.61 nmaxxy -0.33 imaxxy 0.49
230-249 dn_fz -0.041 di_fz -0.006 nbase 0.51 nfz 0.47 nmaxxy -0.29 imaxxy 0.44
250-269 dn_fz 0.05 di_fz -0.005 nbase 0.47 nfz 0.52 nmaxxy -0.42 imaxxy 0.35
270-289 dn_fz -0.002 di_fz -0.005 nbase 0.03 nfz 0.02 nmaxxy 0.03 imaxxy 0.29
290-309 dn_fz -0.105 d …[+1772 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];i=D['iou'];n=D['ncc'];names=list(D['names']);F=len(P);e=i+.4*np.maximum(n,-.5)
# only states 0,1,2 ; carry static override irrelevant here
for alpha in [.03,.06,.1,.15,.2,.3,.5,.8]:
 M=3;ang=np.zeros((F-1,M,M))
 for t in range(F-1):
  for a in range(M):
   for b in range(M):ang[t,a,b]=Rot.from_matrix(P[t,a,:3,:3].T@P[t+1,b,:3,:3]).magnitude()*180/np.pi
 dp=np.zeros((F,M));bk=np.zeros((F,M),int);dp[0]=e[0,:M]
 for t in range(1,F):
  for b in range(M):
   vals=dp[t-1]-alpha*ang[t-1,:,b]/180;a=int(np.argmax(vals));dp[t,b]=vals[a]+e[t,b];bk[t,b]=a
 st=int(np.argmax(dp[-1]));path=np.zeros(F,int);path[-1]=st
 for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
 ch=np.where(path[1:]"'!=path[:-1])[0]+1;runs=[];s=0
 for z in list(ch)+[F]:runs.append((s,z-1,names[path[s]],float(e[s:z,path[s]].mean()),float(e[s:z,0].mean())));s=z
 print('"'\\nalpha',alpha,'counts',dict(zip(names[:3],np.bincount(path,minlength=3))),'runs',[(a,b,nm,round(x,2),round(y,2)) for a,b,nm,x,y in runs if b-a>=2])
PY"
```
<details><summary>output</summary>

```

alpha 0.03 counts {'base': 723, 'fx': 55, 'fy': 25} runs [(0, 117, 'base', 1.03, 1.03), (118, 125, 'fx', 0.41, 0.31), (126, 265, 'base', 0.99, 0.99), (276, 293, 'base', 0.72, 0.72), (295, 312, 'base', 0.61, 0.61), (315, 318, 'fx', 0.42, 0.23), (319, 323, 'base', 0.53, 0.53), (324, 329, 'fy', 0.54, 0.25), (330, 345, 'fx', 0.32, 0.09), (346, 351, 'base', 0.49, 0.49), (353, 356, 'fx', 0.34, 0.3), (361, 363, 'base', -0.13, -0.13), (364, 366, 'fx', 0.28, 0.16), (367, 409, 'base', 0.74, 0.74), (412, 447, 'base', 0.94, 0.94), (448, 452, 'fy', 0.7, 0.6), (453, 487, 'base', 0.66, 0.66), (488, 492, 'fy', 0.38, 0.21), (493, 553, 'base', 1.03, 1.03), (554, 557, 'fx', 0.45, 0.27), (558, 560, 'base', 0.73, 0.73), (561, 566, 'fx', 0.41, 0.28), (567, 604, 'base', 0.91, 0.91), (607, 802, 'base', 1.01, 1.0 …[+4491 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];i=D['iou'];n=D['ncc'];names=list(D['names']);F=len(P);e=i+.4*np.maximum(n,-.5)
def dp(alpha,states=[0,1,2]):
 M=len(states);ang=np.zeros((F-1,M,M))
 for t in range(F-1):
  for a in range(M):
   for b in range(M):ang[t,a,b]=Rot.from_matrix(P[t,states[a],:3,:3].T@P[t+1,states[b],:3,:3]).magnitude()*180/np.pi
 d=np.zeros((F,M));bk=np.zeros((F,M),int);d[0]=e[0,states]
 for t in range(1,F):
  for b in range(M):
   vals=d[t-1]-alpha*ang[t-1,:,b]/180;a=int(np.argmax(vals));d[t,b]=vals[a]+e[t,states[b]];bk[t,b]=a
 st=int(np.argmax(d[-1]));path=np.zeros(F,int);path[-1]=st
 for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
 return np.array([states[x] for x in path])
for alpha in [.03,.06,.1,.15,.2,.3,.5]:
 sel=dp(alpha);T=P[np.arange(F),sel];rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
 print('a',alpha,'score',e[np.arange(F),sel].mean(),'base',e[:,0].mean(),'switches',np.sum(sel[1:]"'!=sel[:-1]),'"'rel>60',np.sum(rel>60),'rel>100',np.sum(rel>100),'q',np.round(np.quantile(rel,[.5,.9,.95,.99,1]),1))
PY"
```
<details><summary>output</summary>

```
a 0.03 score 0.88135403 base 0.8630501 switches 39 rel>60 47 rel>100 41 q [  6.5  28.6 106.4 179.7 180. ]
a 0.06 score 0.88097906 base 0.8630501 switches 33 rel>60 40 rel>100 34 q [  6.3  26.9  59.2 177.1 180. ]
a 0.1 score 0.8808779 base 0.8630501 switches 32 rel>60 39 rel>100 33 q [  6.3  26.9  57.7 175.7 180. ]
a 0.15 score 0.88005656 base 0.8630501 switches 25 rel>60 32 rel>100 26 q [  6.3  26.6  47.1 171.5 180. ]
a 0.2 score 0.88005656 base 0.8630501 switches 25 rel>60 32 rel>100 26 q [  6.3  26.6  47.1 171.5 180. ]
a 0.3 score 0.8787518 base 0.8630501 switches 19 rel>60 27 rel>100 20 q [  6.3  26.4  43.3 170.4 180. ]
a 0.5 score 0.87564325 base 0.8630501 switches 11 rel>60 21 rel>100 12 q [  6.3  25.8  35.6 144.9 178.9]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
r=np.load('inputs/reconstruction.npz');R=r['T_world_mesh'][:,:3,:3];rel=Rot.from_matrix(R[:-1].transpose(0,2,1)@R[1:]).magnitude()*180/np.pi
print('rel>60',sum(rel>60),'rel>100',sum(rel>100),'q',np.quantile(rel,[.5,.9,.95,.99,1]))
PY"
```
<details><summary>output</summary>

```
rel>60 14 rel>100 4 q [  6.27861918  24.9387413   29.95735176  71.33575385 163.2508744 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices;z=V[:,2]
for q in [0,.01,.05,.1,.2,.5,.8,.9,.95,.99,1]:
 zz=np.quantile(z,q);m=np.abs(z-zz)<.015;rad=np.linalg.norm(V[m,:2],axis=1);print(q,round(zz,3),len(rad),round(np.median(rad),3))
PY"
```
<details><summary>output</summary>

```
0 -0.21 18290 0.037
0.01 -0.208 19429 0.038
0.05 -0.199 23150 0.039
0.1 -0.169 14623 0.043
0.2 -0.12 18161 0.044
0.5 0.052 27021 0.03
0.8 0.209 10049 0.013
0.9 0.298 16212 0.016
0.95 0.318 27330 0.022
0.99 0.331 16230 0.023
1 0.333 15130 0.023

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/endpoint_analysis.py <<'PY'
import numpy as np,cv2,h5py,trimesh,json
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],float))
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];names=list(D['names']);F=len(T)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices;zmin,zmax=V[:,2].min(),V[:,2].max();bar=np.array([0,0,zmin,1.]);han=np.array([0,0,zmax,1.])
cap=cv2.VideoCapture('inputs/video.mp4');frames=[]
for i in range(F):ok,im=cap.read();frames.append(cv2.resize(im,(768,576),interpolation=cv2.INTER_AREA))
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def analyze(i,Tc):
 om=cv2.resize(omall[i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pmall[i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);m=om&~pm
 ys,xs=np.where(m);n=len(xs)
 if n<50:return None
 pts=np.stack([xs,ys],1).astype(float);c=pts.mean(0);C=np.cov((pts-c).T);w,vv=np.linalg.eigh(C);ax=vv[:,-1]
 s=(pts-c)@ax;rng=np.percentile(np.abs(s),90);a=pts[s<-0.65*rng];b=pts[s>0.65*rng]
 if len(a)<5 or len(b)<5:return None
 def gray(p):
  q=frames[i][p[:,1].astype(int),p[:,0].astype(int)].astype(float);return np.mean(.299*q[:,0]+.587*q[:,1]+.114*q[:,2])
 ga,gb=gray(a),gray(b)
 if abs(ga-gb)<8:return None
 lightsign=1  …[+964 chars]
```
<details><summary>output</summary>

```
valid 726
base 0.918732782369146
fx 0.10192837465564739
fy 0.10192837465564739
fz 0.918732782369146
118 [1, 0, 0, 1] n 1094 gray [99.1 41.4]
195 [1, 0, 0, 1] n 1053 gray [128.6  59.3]
230 [1, 0, 0, 1] n 1037 gray [ 94.7 119.2]
266 [1, 0, 0, 1] n 132 gray [83.  95.4]
270 [0, 1, 1, 0] n 197 gray [88.3 38.5]
297 [1, 1, 1, 1] n 572 gray [ 94.4 108.3]
315 [1, 0, 0, 1] n 314 gray [188.7 103.6]
320 [0, 1, 1, 0] n 434 gray [144.8 113.5]
324 [0, 1, 1, 0] n 414 gray [146.4 130. ]
330 [0, 1, 1, 0] n 364 gray [150.3 138.2]
331 [0, 1, 1, 0] n 358 gray [153.5 138.8]
332 [1, 0, 0, 1] n 340 gray [152.9 140.5]
337 [None, None, None, None] n None gray None
340 [1, 0, 0, 1] n 436 gray [161.5  86.8]
352 [None, None, None, None] n None gray None
355 [None, None, None, None] n None gray None
361 [1, 1, 1, 1] n  …[+912 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/endpoint_save.py <<'PY'
exec(open('scratch/endpoint_analysis.py').read().split('res=[]')[0])
F=len(T);baseok=np.full(F,np.nan);flipok=np.full(F,np.nan);contrast=np.zeros(F);count=np.zeros(F,np.int32)
for i in range(F):
 a=analyze(i,P[i,0])
 if a is not None:
  n,ga,gb,ls,ms,ax,c=a;baseok[i]=1 if ms==ls else 0;contrast[i]=abs(ga-gb);count[i]=n
  b=analyze(i,P[i,1]);flipok[i]=1 if b is not None and b[4]==b[3] else 0
np.savez('scratch/endpoint_obs.npz',baseok=baseok,flipok=flipok,contrast=contrast,count=count)
for th in [(50,5),(100,8),(150,10),(200,10)]:
 m=(count>th[0])&(contrast>th[1]);print('threshold',th,'valid',m.sum(),'base wrong',np.sum(m&(baseok==0)))
 # runs of base wrong with gaps <=2 filled
 bad=(m&(baseok==0)); # simple dilate/close one
 b=bad.copy()
 for i in range(1,F-1):
  if bad[i-1] and bad[i+1]:b[i]=1
 ch=np.where(np.diff(np.r_[False,b,False].astype(int))"'!=0)[0];print([(int(ch[k]),int(ch[k+1]-1)) for k in range(0,len(ch),2) if ch[k+1]-ch[k]>=1])
PY
python scratch/endpoint_save.py'
```
<details><summary>output</summary>

```
threshold (50, 5) valid 726 base wrong 59
[(233, 237), (267, 277), (295, 296), (299, 300), (320, 331), (397, 397), (447, 462), (477, 480), (534, 535), (632, 637)]
threshold (100, 8) valid 715 base wrong 58
[(233, 237), (267, 277), (295, 296), (299, 300), (320, 331), (447, 462), (477, 480), (534, 535), (632, 637)]
threshold (150, 10) valid 685 base wrong 53
[(234, 236), (269, 277), (295, 296), (299, 299), (320, 331), (447, 462), (477, 480), (534, 534), (632, 637)]
threshold (200, 10) valid 652 base wrong 32
[(234, 236), (272, 275), (295, 296), (299, 299), (320, 331), (477, 480), (534, 534), (632, 637)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/vis_flip_events.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];names=list(D['names'])
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=1536,1152;renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
events=[(228,242,2),(263,280,2),(315,335,2),(443,465,2),(473,483,1),(530,539,1),(628,640,1)]
for a,b,step in events:
 panels=[]
 for i in range(a,b+1,step):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();ys,xs=np.where(om[i]);cx0,cy0=int(xs.mean()),int(ys.mean());half=190;x0=int(np.clip(cx0-half,0,W-2*half));y0=int(np.clip(cy0-half,0,H-2*half));crop=im[y0:y0+2*half,x0:x0+2*half];row=[crop]
  for j in [0,1,2]:
   scene.set_pose(node,pose=flip@Tcw@P[i,j]);col,depth=renderer.render(scene,flags=pyrende …[+544 chars]
```
<details><summary>output</summary>

```
228 242 (3040, 1520, 3)
263 280 (3420, 1520, 3)
315 335 (4180, 1520, 3)
443 465 (4560, 1520, 3)
473 483 (4180, 1520, 3)
530 539 (3800, 1520, 3)
628 640 (4940, 1520, 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/endpoint_color_score.py <<'PY'
import numpy as np,cv2,h5py,trimesh,json
r=np.load('inputs/reconstruction.npz');cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],float));D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];names=list(D['names']);F=len(P)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices;bar=np.array([0,0,V[:,2].min()]);han=np.array([0,0,V[:,2].max()])
cap=cv2.VideoCapture('inputs/video.mp4');frames=[]
for i in range(F):ok,im=cap.read();frames.append(cv2.resize(im,(768,576),interpolation=cv2.INTER_AREA))
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def proj(T,p):
 vc=(Tcw@T)[:3,:3]@p+(Tcw@T)[:3,3];uv=K@vc;uv=uv[:2]/uv[2]*.5;return uv
def endscore(i,T):
 om=cv2.resize(omall[i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pmall[i].astype(np.uint8),(768,576),interpolation=cv2.INTER_NEAREST).astype(bool);m=om&~pm;im=frames[i]
 vals=[]
 for p in [bar,han]:
  u,v=proj(T,p);yy,xx=np.mgrid[max(0,int(v)-12):min(576,int(v)+13),max(0,int(u)-12):min(768,int(u)+13)];sel=m[yy,xx];q=im[yy[sel],xx[sel]].astype(float)
  if len(q)<5:vals.append(np.nan)
  else:
   gray=.299*q[:,0]+.587*q[:,1]+.114*q[:,2];vals.append(np.median(gray))
 return vals[0]-vals[1],vals[0],vals[1]
for i in [118,195,230,266,270,297,315,320,324,330,331,332,337,340,352,355,361,365,369,400,448,4 …[+305 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/endpoint_color_score.py", line 22, in <module>
    print(i,vals,names[int(np.nanargmax([endscore(i,P[i,j])[0] for j in range(4)]))])
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py", line 613, in nanargmax
    raise ValueError("All-NaN slice encountered")
ValueError: All-NaN slice encountered

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/endpoint_color_score.py';s=open(p).read();s=s.replace(\"print(i,vals,names[int(np.nanargmax([endscore(i,P[i,j])[0] for j in range(4)]))])\", \"raw=np.array([endscore(i,P[i,j])[0] for j in range(4)]);print(i,vals,names[int(np.nanargmax(raw))] if np.any(~np.isnan(raw)) else None)\")
open(p,'w').write(s)
PY
python scratch/endpoint_color_score.py"
```
<details><summary>output</summary>

```
118 [None, None, None, None] None
195 [None, None, None, None] None
230 [45.0, None, None, 45.0] base
266 [None, None, None, None] None
270 [None, None, None, None] None
297 [None, None, None, None] None
315 [2.6, -4.4, -4.4, 2.6] base
320 [None, 104.2, 104.2, None] fx
324 [None, None, None, None] None
330 [None, None, None, None] None
331 [None, None, None, None] None
332 [None, None, None, None] None
337 [None, None, None, None] None
340 [None, None, None, None] None
352 [76.3, -79.6, -79.6, 76.3] base
355 [None, -101.7, -101.7, None] fx
361 [None, None, None, None] None
365 [None, None, None, None] None
369 [None, None, None, None] None
400 [None, None, None, None] None
448 [None, 21.2, 21.2, None] fx
451 [None, 77.1, 77.1, None] fx
457 [None, None, None, None] None
486 [None, -69.9, -6 …[+536 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat >> scratch/endpoint_color_score.py <<'PY'
# all-frame endpoint scores
sc=np.full((F,4),np.nan)
for i in range(F):
 for j in range(4):sc[i,j]=endscore(i,P[i,j])[0]
np.save('scratch/endpoint_color_scores.npy',sc)
for th in [10,20,30,40]:
 wrong=(sc[:,1]>sc[:,0]+th)&(sc[:,1]>10)
 b=wrong.copy()
 for i in range(1,F-1):
  if wrong[i-1] and wrong[i+1]:b[i]=1
 ch=np.where(np.diff(np.r_[False,b,False].astype(int))"'!=0)[0];runs=[(int(ch[k]),int(ch[k+1]-1)) for k in range(0,len(ch),2) if ch[k+1]-ch[k]>=1]
 print('"'th',th,'count',wrong.sum(),'runs',runs)
PY
python scratch/endpoint_color_score.py | tail -10"
```
<details><summary>output</summary>

```
646 [41.1, -103.0, -103.0, 41.1] base
659 [None, -37.5, -37.5, None] fx
668 [None, None, None, None] None
681 [93.3, -104.7, -104.7, 93.3] base
714 [96.1, -97.5, -97.5, 96.1] base
744 [98.6, -98.9, -98.9, 98.6] base
th 10 count 11 runs [(234, 236), (296, 296), (306, 306), (452, 452), (533, 534), (637, 639)]
th 20 count 9 runs [(234, 236), (306, 306), (452, 452), (533, 534), (637, 638)]
th 30 count 5 runs [(234, 235), (306, 306), (452, 452), (534, 534)]
th 40 count 2 runs [(235, 235), (534, 534)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
sc=np.load('scratch/endpoint_color_scores.npy');F=len(sc)
for th in [10,20,30]:
 wrong=(np.nan_to_num(sc[:,1],nan=-999)>20)&((np.isnan(sc[:,0]))|(sc[:,1]>sc[:,0]+th))
 b=wrong.copy()
 for i in range(1,F-1):
  if wrong[i-1] and wrong[i+1]:b[i]=1
 ch=np.where(np.diff(np.r_[False,b,False].astype(int))"'!=0)[0];runs=[(int(ch[k]),int(ch[k+1]-1)) for k in range(0,len(ch),2) if ch[k+1]-ch[k]>=1]
 print(th,sum(wrong),runs)
PY'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/final_rot_scores.npz');i=D['iou'];n=D['ncc'];F=len(i);E=np.load('scratch/endpoint_color_scores.npy')
base=i[:,0]+.4*np.maximum(n[:,0],-.5)
flip=np.maximum(i[:,1]+.4*np.maximum(n[:,1],-.5),i[:,2]+.4*np.maximum(n[:,2],-.5))
# endpoint prior
for t in range(F):
 b,f=E[t,0],E[t,1]
 if not np.isnan(b) and b>20:base[t]+=.15
 if not np.isnan(f) and f>20 and (np.isnan(b) or f>b+10):flip[t]+=.15
for lam in [.03,.05,.08,.1,.15,.2,.3]:
 M=2;dp=np.zeros((F,2));bk=np.zeros((F,2),int);em=np.stack([base,flip],1);dp[0]=em[0]
 for t in range(1,F):
  for s in range(2):
   vals=[dp[t-1,p]-(lam if p"'!=s else 0) for p in range(2)];p=int(np.argmax(vals));dp[t,s]=vals[p]+em[t,s];bk[t,s]=p
 s=int(np.argmax(dp[-1]));path=np.zeros(F,int);path[-1]=s
 for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
 ch=np.where(path[1:]!=path[:-1])[0]+1;runs=[];st=0
 for e in list(ch)+[F]:runs.append((st,e-1,int(path[st]),round(float(em[st:e,path[st]].mean()),2),round(float(em[st:e,0].mean()),2)));st=e
 print('"'\\nlam',lam,'flip frac',path.mean(),'counts',np.bincount(path,minlength=2),'runs',[(a,b,s,x,y) for a,b,s,x,y in runs if b-a>=1])
PY"
```
<details><summary>output</summary>

```

lam 0.03 flip frac 0.10460772104607721 counts [719  84] runs [(0, 117, 0, 1.18, 1.18), (118, 125, 1, 0.41, 0.31), (126, 265, 0, 1.11, 1.11), (266, 275, 1, 0.31, 0.09), (276, 293, 0, 0.75, 0.75), (295, 312, 0, 0.66, 0.66), (315, 318, 1, 0.46, 0.23), (320, 344, 1, 0.42, 0.21), (345, 353, 0, 0.45, 0.45), (354, 357, 1, 0.37, 0.31), (358, 359, 0, 0.46, 0.46), (361, 363, 0, -0.13, -0.13), (364, 366, 1, 0.28, 0.16), (367, 409, 0, 0.75, 0.75), (410, 411, 1, 0.23, 0.08), (412, 447, 0, 1.03, 1.03), (448, 456, 1, 0.79, 0.64), (457, 487, 0, 0.67, 0.67), (488, 492, 1, 0.38, 0.21), (493, 553, 0, 1.17, 1.17), (554, 557, 1, 0.45, 0.27), (558, 561, 0, 0.79, 0.79), (562, 566, 1, 0.4, 0.26), (567, 604, 0, 1.02, 1.02), (605, 606, 1, 0.33, 0.06), (607, 802, 0, 1.13, 1.13)]

lam 0.05 flip frac 0.10585305105853 …[+3723 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/final_rot_scores.npz');i=D['iou'];n=D['ncc']
segs=[(118,125),(266,275),(315,344),(354,366),(410,411),(448,456),(488,492),(554,557),(562,566),(605,606)]
for a,b in segs:
 vals=[]
 for j in [1,2]:
  vals.append((i[a:b+1,j]+.4*np.maximum(n[a:b+1,j],-.5)).mean())
 print(a,b,'fx',round(vals[0],3),'fy',round(vals[1],3),'base',round((i[a:b+1,0]+.4*np.maximum(n[a:b+1,0],-.5)).mean(),3),'ncc',np.round(n[a:b+1,1:3].mean(0),2),'iou',np.round(i[a:b+1,1:3].mean(0),2))
PY"
```
<details><summary>output</summary>

```
[0.43 0.4 ] iou [0.23 0.22]
605 606 fx 0.335 fy 0.329 base 0.057 ncc [0.37 0.36] iou [0.19 0.18]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/final_rot_scores.npz');i=D['iou'];n=D['ncc'];base=i[:,0]+.4*np.maximum(n[:,0],-.5);fz=i[:,3]+.4*np.maximum(n[:,3],-.5);d=fz-base
for w in [3,5,10,20]:
 print('\\nw',w)
 for s in range(90,750,w):
  e=min(750,s+w)
  if d[s:e].mean()>.12:print(s,e-1,round(float(d[s:e].mean()),2),round(float(n[s:e,0].mean()),2),round(float(n[s:e,3].mean()),2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/vis_flip_events.py';s=open(p).read();s=s.replace(\"events=[(228,242,2),(263,280,2),(315,335,2),(443,465,2),(473,483,1),(530,539,1),(628,640,1)]\", \"events=[(343,352,1),(386,402,1)]\")
s=s.replace(\"for j in [0,1,2]:\", \"for j in [0,3]:\")
open('scratch/vis_fz_events.py','w').write(s)
PY
python scratch/vis_fz_events.py"
```
<details><summary>output</summary>

```
343 352 (3800, 1140, 3)
386 402 (6460, 1140, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
Pi=np.load('scratch/static_init_optpose.npy');Pf=np.load('scratch/static_final_optpose.npy');D=np.load('scratch/final_rot_scores.npz');P=D['Pbase'];r=np.load('inputs/reconstruction.npz');B=r['T_world_mesh']
for tag,A in [('init',Pi),('final',Pf)]:
 print(tag,np.round(A[:3,3],4),'R',np.round(A[:3,:3],3))
for i in list(range(85,101))+list(range(738,752)):
 print(i,'base pos',np.round(B[i,:3,3],3),'fit',np.round(P[i,:3,3],3))
print('dist init to 90-95',[(i,round(float(np.linalg.norm(Pi[:3,3]-P[i,:3,3])),3),round(float(Rot.from_matrix(Pi[:3,:3].T@P[i,:3,:3]).magnitude()*180/np.pi),1)) for i in range(88,100)])
print('dist final to 740-746',[(i,round(float(np.linalg.norm(Pf[:3,3]-P[i,:3,3])),3),round(float(Rot.from_matrix(Pf[:3,:3].T@P[i,:3,:3]).magnitude()*180/np.pi),1)) for i in range(738,748)])
PY"
```
<details><summary>output</summary>

```
init [0.2176 1.2383 2.4015] R [[ 0.059 -0.086 -0.995]
 [ 0.974  0.225  0.038]
 [ 0.22  -0.971  0.097]]
final [0.2749 1.2384 2.9055] R [[ 0.266  0.705 -0.658]
 [ 0.964 -0.187  0.189]
 [ 0.01  -0.684 -0.729]]
85 base pos [0.221 1.247 2.43 ] fit [0.221 1.247 2.43 ]
86 base pos [0.221 1.247 2.43 ] fit [0.221 1.247 2.43 ]
87 base pos [0.221 1.246 2.427] fit [0.221 1.246 2.427]
88 base pos [0.221 1.246 2.427] fit [0.221 1.246 2.427]
89 base pos [0.23  1.246 2.428] fit [0.23  1.246 2.428]
90 base pos [0.231 1.252 2.438] fit [0.216 1.252 2.427]
91 base pos [0.23  1.246 2.423] fit [0.21  1.246 2.416]
92 base pos [0.231 1.246 2.424] fit [0.221 1.246 2.415]
93 base pos [0.231 1.248 2.428] fit [0.213 1.248 2.418]
94 base pos [0.23  1.242 2.416] fit [0.215 1.242 2.407]
95 base pos [0.23  1.239 2.409] f …[+1576 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/make_static_pool2.py <<'PY'
import numpy as np,h5py,cv2
cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
for name,a,b in [('init',0,105),('final',716,803)]:
 obj=om[a:b].any(0);ys,xs=np.where(obj);pad=24;x0=max(0,xs.min()-pad);x1=min(1535,xs.max()+pad);y0=max(0,ys.min()-pad);y1=min(1151,ys.max()+pad)
 vals=[];good=[];cap.set(cv2.CAP_PROP_POS_FRAMES,a)
 for i in range(a,b):
  ok,im=cap.read();vals.append(im[y0:y1+1,x0:x1+1]);good.append((om[i]&~pm[i])[y0:y1+1,x0:x1+1])
 vals=np.stack(vals);good=np.stack(good);med=np.nanmedian(np.where(good[...,None],vals.astype(float),np.nan),axis=0);med=np.nan_to_num(med).astype(np.uint8)
 np.savez(f'scratch/static_{name}2.npz',obj=obj[y0:y1+1,x0:x1+1],pooled=med,bbox=np.array([x0,x1,y0,y1]),count=om[a:b].sum(0)[y0:y1+1,x0:x1+1])
 print(name,(x0,x1,y0,y1),obj.sum(),med.shape)
PY
python scratch/make_static_pool2.py"
```
<details><summary>output</summary>

```
/work/scratch/make_static_pool2.py:9: RuntimeWarning: All-NaN slice encountered
  vals=np.stack(vals);good=np.stack(good);med=np.nanmedian(np.where(good[...,None],vals.astype(float),np.nan),axis=0);med=np.nan_to_num(med).astype(np.uint8)
init (640, 875, 911, 998) 5021 (88, 236, 3)
final (694, 850, 821, 959) 2780 (139, 157, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,cv2
for n in ['init2','final2']:
 d=np.load(f'scratch/static_{n}.npz');cv2.imwrite(f'scratch/static_{n}.jpg',np.concatenate([d['pooled'],cv2.cvtColor((d['obj']*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)],1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 2)
```
/usr/bin/bash -c 'python scratch/static_opt2.py'
```
<details><summary>output</summary>

```
python: can't open file '/work/scratch/static_opt2.py': [Errno 2] No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/static_opt2.py <<'PY'
import numpy as np,trimesh,json,cv2,os
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.])
def avg_pose(idx):
 A=T[idx];U,S,Vt=np.linalg.svd(A[:,:3,:3].sum(0));R=U@Vt
 if np.linalg.det(R)<0:U[:,-1]*=-1;R=U@Vt
 P=np.eye(4);P[:3,:3]=R;P[:3,3]=np.median(A[:,:3,3],0);return P
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
for name,a,b in [('init',0,105),('final',716,803)]:
 d0=np.load(f'scratch/static_{name}2.npz');om=d0['obj'].astype(bool);pool=d0['pooled'].astype(float)/255;H,W=om.shape;x0,x1,y0,y1=d0['bbox']
 renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
 def render(P):
  scene.set_pose(node,pose=flip@Tcw@P);col,depth=rend …[+1508 chars]
```
<details><summary>output</summary>

```

 init start (0.9595863714464704, 0.7457892675283979, 0.7126570130602418) [0.22038366 1.23957217 2.41338563]
roll180 (0.9248483893836053, 0.7206195546950629, 0.6807627822951414)
FINAL init (0.9762703543681633, 0.7709863068069062, 0.684280158537524) [0.21538366 1.23957217 2.40838563]

 final start (0.9301104145336841, 0.7457870204374327, 0.6144113136541715) [0.28508788 1.23804688 2.9053576 ]
roll180 (0.9078598643770913, 0.7164073550212164, 0.6381750311862496)
FINAL final (1.0020191023355867, 0.7699178277956413, 0.7736709151331513) [0.27808788 1.23804688 2.9133576 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/final_rot_scores.npz');i=D['iou'];n=D['ncc'];
segs=[(118,125),(234,236),(266,275),(315,344),(354,366),(410,411),(448,456),(488,492),(554,557),(562,566),(605,606),(632,638)]
for a,b in segs:
 base=i[a:b+1,0]+.4*np.maximum(n[a:b+1,0],-.5);fx=i[a:b+1,1]+.4*np.maximum(n[a:b+1,1],-.5);fy=i[a:b+1,2]+.4*np.maximum(n[a:b+1,2],-.5);best=np.maximum(fx,fy)
 print(a,b,'dscore',round(float((best-base).mean()),3),'dncc_fx',round(float((n[a:b+1,1]-n[a:b+1,0]).mean()),3),'dncc_fy',round(float((n[a:b+1,2]-n[a:b+1,0]).mean()),3),'choose','fx' if fx.mean()>fy.mean() else 'fy')
PY"
```
<details><summary>output</summary>

```
118 125 dscore 0.105 dncc_fx 0.644 dncc_fy 0.546 choose fx
234 236 dscore -0.499 dncc_fx -0.426 dncc_fy -0.456 choose fx
266 275 dscore 0.218 dncc_fx 0.296 dncc_fy 0.358 choose fy
315 344 dscore 0.195 dncc_fx 0.671 dncc_fy 0.632 choose fx
354 366 dscore 0.045 dncc_fx 0.087 dncc_fy 0.03 choose fx
410 411 dscore 0.151 dncc_fx 0.58 dncc_fy 0.619 choose fy
448 456 dscore -0.001 dncc_fx 0.665 dncc_fy 0.723 choose fy
488 492 dscore 0.175 dncc_fx 0.785 dncc_fy 0.825 choose fy
554 557 dscore 0.175 dncc_fx 0.711 dncc_fy 0.606 choose fx
562 566 dscore 0.14 dncc_fx 0.738 dncc_fy 0.706 choose fx
605 606 dscore 0.278 dncc_fx 0.856 dncc_fy 0.847 choose fx
632 638 dscore -0.67 dncc_fx -0.522 dncc_fy -0.593 choose fx

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('scratch/fit_candidates_all.npz');P=D['Pw'];segs=[(118,125,1),(266,275,2),(315,344,1),(410,411,2),(488,492,2),(554,557,1),(562,566,1),(605,606,1)]
for a,b,j in segs:
 dp=np.linalg.norm(P[a:b+1,j,:3,3]-P[a:b+1,0,:3,3],axis=1)
 print(a,b,j,'trans diff',np.round(np.quantile(dp,[0,.5,1]),3),'iou base',round(float(D['ioupr'][a:b+1,0].mean()),2),'flip',round(float(D['ioupr'][a:b+1,j].mean()),2))
PY"
```
<details><summary>output</summary>

```
118 125 1 trans diff [0.097 0.157 0.219] iou base 0.42 flip 0.38
266 275 2 trans diff [0.079 0.137 0.2  ] iou base 0.22 flip 0.3
315 344 1 trans diff [0.082 0.18  0.267] iou base 0.37 flip 0.45
410 411 2 trans diff [0.126 0.145 0.164] iou base 0.25 flip 0.29
488 492 2 trans diff [0.097 0.1   0.146] iou base 0.32 flip 0.33
554 557 1 trans diff [0.064 0.116 0.205] iou base 0.34 flip 0.28
562 566 1 trans diff [0.092 0.121 0.182] iou base 0.39 flip 0.36
605 606 1 trans diff [0.104 0.162 0.219] iou base 0.24 flip 0.27

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/check_fz_full.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));r=np.load('inputs/reconstruction.npz');g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=1536,1152;renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
for a,b in [(343,351),(386,402),(488,492),(550,569),(590,609),(650,669)]:
 vals=[]
 for i in range(a,b+1):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();om=omall[i];pm=cv2.dilate(pmall[i].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);er=cv2.erode(om.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);row=[]
  for j in [0,3]:
 …[+383 chars]
```
<details><summary>output</summary>

```
343 351 ncc base/fz mean [-0.102  0.143] win fz 7 / 9
386 402 ncc base/fz mean [-0.119  0.18 ] win fz 15 / 17
488 492 ncc base/fz mean [-0.203 -0.368] win fz 0 / 5
550 569 ncc base/fz mean [0.025 0.125] win fz 20 / 20
590 609 ncc base/fz mean [0.293 0.362] win fz 20 / 20
650 669 ncc base/fz mean [-0.     0.102] win fz 16 / 20

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/full_fz_scores.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os,time
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];F=len(P);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=1536,1152;renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
nccs=np.full((F,2),-1,np.float32);t0=time.time()
for i in range(F):
 ok,im=cap.read();im=im.astype(np.float32)/255;om=omall[i];pm=cv2.dilate(pmall[i].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);er=cv2.erode(om.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
 for k,j in enumerate([0,3]):
  scene.set_pose(node,pose=flip@Tcw@P[i,j]);col,depth=renderer.r …[+298 chars]
```
<details><summary>output</summary>

```
0 0.5
100 17.9
200 52.5
300 105.6
400 159.5
500 191.2
600 207.5
700 224.1
800 244.6
done 245.10490655899048

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
n=np.load('scratch/full_fz_scores.npy');d=n[:,1]-n[:,0]
print('overall means',n.mean(0),'fz better frac',np.mean(d>0))
for w in [5,10,20,40]:
 print('\\nw',w)
 for s in range(90,746,w):
  e=min(746,s+w)
  if d[s:e].mean()>.08:print(s,e-1,round(float(d[s:e].mean()),2),'base',round(float(n[s:e,0].mean()),2),'fz',round(float(n[s:e,1].mean()),2),'win',np.mean(d[s:e]>0))
# print each 10 window
print('\\nall 10')
for s in range(90,746,10):
 e=min(746,s+10);print(s,e-1,round(float(d[s:e].mean()),2),round(float(n[s:e,0].mean()),2),round(float(n[s:e,1].mean()),2),round(float(np.mean(d[s:e]>0)),2))
PY"
```
<details><summary>output</summary>

```
overall means [0.44023558 0.44109738] fz better frac 0.44084682440846823

w 5
120 124 0.11 base -0.26 fz -0.15 win 1.0
265 269 0.23 base -0.31 fz -0.08 win 1.0
270 274 0.08 base -0.24 fz -0.15 win 1.0
330 334 0.11 base -0.35 fz -0.25 win 0.8
335 339 0.21 base -0.33 fz -0.12 win 1.0
340 344 0.18 base -0.5 fz -0.32 win 0.8
345 349 0.41 base 0.08 fz 0.49 win 1.0
360 364 0.08 base -0.57 fz -0.49 win 0.4
385 389 0.12 base 0.13 fz 0.25 win 1.0
390 394 0.47 base -0.05 fz 0.41 win 1.0
395 399 0.39 base -0.32 fz 0.07 win 0.8
400 404 0.25 base -0.35 fz -0.1 win 0.8
445 449 0.08 base -0.2 fz -0.12 win 0.8
555 559 0.08 base 0.02 fz 0.1 win 1.0
560 564 0.13 base -0.17 fz -0.04 win 1.0
565 569 0.11 base -0.09 fz 0.02 win 1.0
635 639 0.12 base -0.03 fz 0.08 win 1.0
640 644 0.1 base 0.07 fz 0.17 win 0.8
6 …[+2659 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/full_fxfy_scores.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os,time
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];F=len(P);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=1536,1152;renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
nccs=np.full((F,2),-1,np.float32);t0=time.time()
for i in range(F):
 ok,im=cap.read();im=im.astype(np.float32)/255;om=omall[i];pm=cv2.dilate(pmall[i].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);er=cv2.erode(om.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
 for k,j in enumerate([1,2]):
  scene.set_pose(node,pose=flip@Tcw@P[i,j]);col,depth=renderer …[+304 chars]
```
<details><summary>output</summary>

```
0 0.4
100 18.3
200 36.7
300 58.3
400 83.3
500 108.2
600 134.2
700 159.4
800 186.1
done 186.68674397468567

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];i=D['iou'];F=len(P);names=list(D['names']);n=np.zeros((F,4));n[:,[0,3]]=np.load('scratch/full_fz_scores.npy');n[:,[1,2]]=np.load('scratch/full_fxfy_scores.npy');e=i+.5*np.maximum(n,-.5)
for alpha in [.05,.1,.2,.3,.5,.8,1.2]:
 M=4;ang=np.zeros((F-1,M,M))
 for t in range(F-1):
  for a in range(M):
   for b in range(M):ang[t,a,b]=Rot.from_matrix(P[t,a,:3,:3].T@P[t+1,b,:3,:3]).magnitude()*180/np.pi
 dp=np.zeros((F,M));bk=np.zeros((F,M),int);dp[0]=e[0]
 for t in range(1,F):
  for b in range(M):
   vals=dp[t-1]-alpha*ang[t-1,:,b]/180;a=int(np.argmax(vals));dp[t,b]=vals[a]+e[t,b];bk[t,b]=a
 s=int(np.argmax(dp[-1]));path=np.zeros(F,int);path[-1]=s
 for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
 ch=np.where(path[1:]"'!=path[:-1])[0]+1;runs=[];st=0
 for z in list(ch)+[F]:runs.append((st,z-1,names[path[st]],round(float(e[st:z,path[st]].mean()),2),round(float(e[st:z,0].mean()),2),round(float(n[st:z,path[st]].mean()),2),round(float(n[st:z,0].mean()),2)));st=z
 print('"'\\nalpha',alpha,'counts',dict(zip(names,np.bincount(path,minlength=4))),'switches',len(ch),'runs')
 print([(a,b,nm,x,y,n1,n0) for a,b,nm,x,y,n1,n0 in runs if b-a>=2])
PY"
```
<details><summary>output</summary>

```

alpha 0.05 counts {'base': 511, 'fx': 45, 'fy': 31, 'fz': 216} switches 61 runs
[(0, 117, 'base', 1.1, 1.1, 0.68, 0.68), (118, 125, 'fx', 0.46, 0.29, 0.39, -0.25), (126, 149, 'fz', 1.14, 1.13, 0.52, 0.49), (150, 166, 'base', 1.17, 1.17, 0.54, 0.54), (167, 170, 'fz', 0.98, 0.94, 0.51, 0.43), (171, 202, 'base', 0.98, 0.98, 0.59, 0.59), (203, 212, 'fz', 1.02, 1.01, 0.69, 0.65), (213, 246, 'base', 1.07, 1.07, 0.56, 0.56), (247, 265, 'fz', 1.01, 0.99, 0.65, 0.6), (266, 271, 'fy', 0.35, 0.07, 0.18, -0.21), (272, 274, 'fx', 0.31, 0.03, 0.19, -0.42), (275, 277, 'fz', 0.45, 0.34, 0.34, 0.12), (278, 293, 'base', 0.79, 0.79, 0.16, 0.16), (297, 312, 'base', 0.7, 0.7, 0.23, 0.23), (315, 318, 'fx', 0.52, 0.24, 0.59, -0.28), (319, 323, 'base', 0.54, 0.54, -0.1, -0.1), (324, 331, 'fy', 0.61, 0.21, 0.56,  …[+8203 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
Pi=np.load('scratch/static_init2_optpose.npy');Pf=np.load('scratch/static_final2_optpose.npy');D=np.load('scratch/final_rot_scores.npz');P=D['Pbase'];r=np.load('inputs/reconstruction.npz');B=r['T_world_mesh']
print('Pi',Pi[:3,3],Rot.from_matrix(Pi[:3,:3]).as_euler('xyz',degrees=True))
print('Pf',Pf[:3,3],Rot.from_matrix(Pf[:3,:3]).as_euler('xyz',degrees=True))
for i in [104,105,110,715,716,720,740,750,760]:
 print(i,'base fit',np.round(P[i,:3,3],3),'dposPf',np.round(np.linalg.norm(P[i,:3,3]-Pf[:3,3]),3),'drotPf',np.round(Rot.from_matrix(Pf[:3,:3].T@P[i,:3,:3]).magnitude()*180/np.pi,1),'drotB',np.round(Rot.from_matrix(Pf[:3,:3].T@B[i,:3,:3]).magnitude()*180/np.pi,1))
PY"
```
<details><summary>output</summary>

```
Pi [0.21538366 1.23957217 2.40838563] [-80.1659205  -14.11600606  83.88911153]
Pf [0.27808788 1.23804688 2.9133576 ] [-135.87523606   -5.40282313   79.78738067]
104 base fit [0.205 1.223 2.364] dposPf 0.555 drotPf 61.8 drotB 61.8
105 base fit [0.211 1.224 2.363] dposPf 0.555 drotPf 59.9 drotB 59.9
110 base fit [0.226 1.25  2.413] dposPf 0.503 drotPf 62.8 drotB 62.8
715 base fit [0.277 1.22  2.858] dposPf 0.058 drotPf 165.1 drotB 165.1
716 base fit [0.276 1.221 2.866] dposPf 0.05 drotPf 165.1 drotB 165.1
720 base fit [0.278 1.229 2.882] dposPf 0.033 drotPf 165.1 drotB 165.1
740 base fit [0.276 1.238 2.897] dposPf 0.017 drotPf 165.1 drotB 165.1
750 base fit [0.285 1.241 2.911] dposPf 0.008 drotPf 14.4 drotB 14.4
760 base fit [0.285 1.24  2.909] dposPf 0.008 drotPf 14.4 drotB 14.4

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/build_final_proto.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr,os
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
# load precomputed development candidates
D=np.load('scratch/final_rot_scores.npz');Pbase=D['Pbase'].astype(np.float64);P4=D['Pw'].astype(np.float64);iou=D['iou'];F=len(Pbase)
fulln=np.zeros((F,4));fulln[:,[0,3]]=np.load('scratch/full_fz_scores.npy');fulln[:,[1,2]]=np.load('scratch/full_fxfy_scores.npy');names=list(D['names'])
FC=np.load('scratch/fit_candidates_all.npz');Pfit=FC['Pw'].astype(np.float64)
# HMM over symmetry candidates, full-res texture + mask, switch penalty
e=iou+.5*np.maximum(fulln,-.5);alpha=.3;M=4
ang=np.zeros((F-1,M,M))
for t in range(F-1):
 for a in range(M):
  for b in range(M):ang[t,a,b]=Rot.from_matrix(P4[t,a,:3,:3].T@P4[t+1,b,:3,:3]).magnitude()*180/np.pi
dp=np.zeros((F,M));bk=np.zeros((F,M),int);dp[0]=e[0]
for t in range(1,F):
 for b in range(M):
  vals=dp[t-1]-alpha*ang[t-1,:,b]/180;a=int(np.argmax(vals));dp[t,b]=vals[a]+e[t,b];bk[t,b]=a
s=int(np.argmax(dp[-1]));state=np.zeros(F,int);state[-1]=s
for t in range(F-1,0,-1):state[t-1]=bk[t,state[t]]
# rest poses
Pi=np.load('scratch/static_init2_optpose.npy');Pf=np.load('scratch/static_final2_optpose.npy')
T=Pbase.copy()
for i in range(F):
 if state[i]"'!=0:
  T[i]=Pfit[i,state[i]].copy()
# smooth translation in moving phase
p=T[:,:3,3].copy()
for k in range(3):
 p[111:716,k]=savgol_ …[+3580 chars]
```
<details><summary>output</summary>

```
state runs
0 117 base 118
0 125 base 126
0 151 base 152
0 246 base 247
0 265 base 266
0 274 base 275
0 277 base 278
0 293 base 294
0 294 base 295
0 314 base 315
0 331 base 332
0 344 base 345
0 356 base 357
0 360 base 361
0 362 base 363
0 365 base 366
0 383 base 384
0 409 base 410
0 411 base 412
0 447 base 448
0 452 base 453
0 484 base 485
0 492 base 493
0 553 base 554
0 557 base 558
0 560 base 561
0 566 base 567
0 569 base 570
0 583 base 584
0 604 base 605
0 606 base 607
0 674 base 675
0 744 base 745
0 746 base 747
0 802 base 803
rot q [ 5.37603196 25.91854177 36.70247898 48.48869395 58.88883907] >60 0 >100 0
trans q [0.02358903 0.06485429 0.07864749 0.12038329 0.14131349]
mask final/base 0.6756737610486689 0.6012763727406816 phases
0 111 0.807 0.777
111 200 0.742 0.708
200 300 0.6 0.489
3 …[+99 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];iou=D['iou'];F=len(P);fulln=np.zeros((F,4));fulln[:,[0,3]]=np.load('scratch/full_fz_scores.npy');fulln[:,[1,2]]=np.load('scratch/full_fxfy_scores.npy');e=iou+.5*np.maximum(fulln,-.5);M=4;alpha=.3
ang=np.zeros((F-1,M,M))
for t in range(F-1):
 for a in range(M):
  for b in range(M):ang[t,a,b]=Rot.from_matrix(P[t,a,:3,:3].T@P[t+1,b,:3,:3]).magnitude()*180/np.pi
d=np.zeros((F,M));bk=np.zeros((F,M),int);d[0]=e[0]
for t in range(1,F):
 for b in range(M):
  vals=d[t-1]-alpha*ang[t-1,:,b]/180;a=int(np.argmax(vals));d[t,b]=vals[a]+e[t,b];bk[t,b]=a
st=int(np.argmax(d[-1]));path=np.zeros(F,int);path[-1]=st
for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
print(np.bincount(path,minlength=4),path[100:130],path[310:350],d[-1])
PY"
```
<details><summary>output</summary>

```
[536  39  43 185] [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 3 3 3 3] [0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 3 3
 3 3 3] [745.13112266 743.91867045 743.91430222 744.81147443]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'state' scratch/build_final_proto.py"
```
<details><summary>output</summary>

```
18:s=int(np.argmax(dp[-1]));state=np.zeros(F,int);state[-1]=s
19:for t in range(F-1,0,-1):state[t-1]=bk[t,state[t]]
24: if state[i]!=0:
25:  T[i]=Pfit[i,state[i]].copy()
57:print('state runs')
58:ch=np.where(state[1:]!=state[:-1])[0]+1;st=0
60: if z-st>=2:print(st,z-1,names[state[st]],z-st)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba scratch/build_final_proto.py | sed -n '1,35p'"
```
<details><summary>output</summary>

```
     1	import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr,os
     2	from scipy.spatial.transform import Rotation as Rot
     3	from scipy.signal import savgol_filter
     4	# load precomputed development candidates
     5	D=np.load('scratch/final_rot_scores.npz');Pbase=D['Pbase'].astype(np.float64);P4=D['Pw'].astype(np.float64);iou=D['iou'];F=len(Pbase)
     6	fulln=np.zeros((F,4));fulln[:,[0,3]]=np.load('scratch/full_fz_scores.npy');fulln[:,[1,2]]=np.load('scratch/full_fxfy_scores.npy');names=list(D['names'])
     7	FC=np.load('scratch/fit_candidates_all.npz');Pfit=FC['Pw'].astype(np.float64)
     8	# HMM over symmetry candidates, full-res texture + mask, switch penalty
     9	e=iou+.5*np.maximum(fulln,-.5);alpha=.3;M=4
    10	ang=np.zeros((F-1,M,M))
    11	for t in ran …[+1138 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_final_proto.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
T=np.load('scratch/final_proto_poses.npy');B=np.load('inputs/reconstruction.npz')['T_world_mesh'];F=len(T);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576;renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
out=np.zeros((F,4)) # iouF,nccF,iouB,nccB
for i in range(F):
 ok,im=cap.read();im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)/255;om=cv2.resize(omall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pmall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAR …[+894 chars]
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
final iou/ncc [0.66685403 0.44373306] base [0.58922112 0.38547372] score final 0.8039929484611451 base 0.7133624113283966
0 111 [0.792 0.76  0.748 0.674]
111 200 [0.738 0.483 0.687 0.484]
200 300 [0.598 0.38  0.486 0.255]
300 400 [ 0.467 -0.073  0.306 -0.252]
400 500 [0.564 0.232 0.483 0.219]
500 600 [0.701 0.502 0.588 0.531]
600 716 [0.723 0.511 0.65  0.439]
716 803 [0.747 0.755 0.774 0.753]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/build_final_proto.py';s=open(p).read();s=s.replace(\"Pi=np.load('scratch/static_init2_optpose.npy');Pf=np.load('scratch/static_final2_optpose.npy')\", \"Pi=np.load('scratch/static_init2_optpose.npy');Pf=np.load('scratch/static_final_optpose.npy')\")
s=s.replace('111:716','111:746').replace('range(716,F)','range(746,F)').replace('range(708,716)','range(745,750)').replace('(i-708)/7.0','(i-745)/5.0')
s=s.replace(\"np.save('scratch/final_proto_poses.npy',T)\",\"np.save('scratch/final_proto_poses2.npy',T)\")
open('scratch/build_final_proto2.py','w').write(s)
PY
python scratch/build_final_proto2.py"
```
<details><summary>output</summary>

```
state runs
0 117 base 118
0 125 base 126
0 151 base 152
0 246 base 247
0 265 base 266
0 274 base 275
0 277 base 278
0 293 base 294
0 294 base 295
0 314 base 315
0 331 base 332
0 344 base 345
0 356 base 357
0 360 base 361
0 362 base 363
0 365 base 366
0 383 base 384
0 409 base 410
0 411 base 412
0 447 base 448
0 452 base 453
0 484 base 485
0 492 base 493
0 553 base 554
0 557 base 558
0 560 base 561
0 566 base 567
0 569 base 570
0 583 base 584
0 604 base 605
0 606 base 607
0 674 base 675
0 744 base 745
0 746 base 747
0 802 base 803
rot q [  5.42138617  26.49581811  36.71427109  49.17814323 127.78105091] >60 1 >100 1
trans q [0.02358903 0.06485429 0.07864749 0.12038329 0.14131349]
mask final/base 0.6771861359884576 0.6012763727406816 phases
0 111 0.807 0.777
111 200 0.742 0.708
200 300 0.6 0. …[+102 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('scratch/final_proto_poses2.npy');rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
for i in np.argsort(rel)[-15:][::-1]:print(i,rel[i],T[i,:3,3],T[i+1,:3,3])
PY"
```
<details><summary>output</summary>

```
745 127.78105091106323 [0.2787352  1.24243319 2.9152384 ] [0.2748953  1.23843145 2.90550327]
559 58.88883906608778 [-0.08567816 -0.1823445   2.75966505] [-0.10406194 -0.19390854  2.820191  ]
558 56.61776302402474 [-0.11670486 -0.17651706  2.82511107] [-0.08567816 -0.1823445   2.75966505]
410 55.02592092664446 [ 0.43585038 -0.36619815  3.77597805] [ 0.41571464 -0.37260934  3.79179963]
560 54.55539805148833 [-0.10406194 -0.19390854  2.820191  ] [-0.14352163 -0.20887555  2.9508559 ]
409 51.14107339834468 [ 0.40273144 -0.31161514  3.83441463] [ 0.43585038 -0.36619815  3.77597805]
557 49.91752365020593 [-0.17358668 -0.14592756  2.82603192] [-0.11670486 -0.17651706  2.82511107]
411 49.663002103607425 [ 0.41571464 -0.37260934  3.79179963] [ 0.35079627 -0.32784193  3.88963561]
117 49.1849961449104 …[+649 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/build_final_proto2.py';s=open(p).read()
s=s.replace(\"# static phases and blended transitions\\nfor i in range(111):T[i]=Pi\", \"# static phases and blended transitions\\nTm=T.copy()\\nfor i in range(111):T[i]=Pi\")
s=s.replace(\"for i in range(745,750):\\n w=(i-745)/5.0\\n qa=Rot.from_matrix(T[i,:3,:3]);qb=Rot.from_matrix(Pf[:3,:3]);sl=Slerp([0,1],Rot.concatenate([qa,qb]));T[i,:3,:3]=sl(w).as_matrix();T[i,:3,3]=(1-w)*T[i,:3,3]+w*Pf[:3,3]\", \"startT=Tm[745].copy()\\nfor i in range(745,750):\\n w=(i-745)/5.0\\n qa=Rot.from_matrix(startT[:3,:3]);qb=Rot.from_matrix(Pf[:3,:3]);sl=Slerp([0,1],Rot.concatenate([qa,qb]));T[i,:3,:3]=sl(w).as_matrix();T[i,:3,3]=(1-w)*startT[:3,3]+w*Pf[:3,3]\")
open(p,'w').write(s)
PY
python scratch/build_final_proto2.py"
```
<details><summary>output</summary>

```
state runs
0 117 base 118
0 125 base 126
0 151 base 152
0 246 base 247
0 265 base 266
0 274 base 275
0 277 base 278
0 293 base 294
0 294 base 295
0 314 base 315
0 331 base 332
0 344 base 345
0 356 base 357
0 360 base 361
0 362 base 363
0 365 base 366
0 383 base 384
0 409 base 410
0 411 base 412
0 447 base 448
0 452 base 453
0 484 base 485
0 492 base 493
0 553 base 554
0 557 base 558
0 560 base 561
0 566 base 567
0 569 base 570
0 583 base 584
0 604 base 605
0 606 base 607
0 674 base 675
0 744 base 745
0 746 base 747
0 802 base 803
rot q [ 5.48935262 25.91854177 36.70247898 48.48869395 58.88883907] >60 0 >100 0
trans q [0.02358903 0.06485429 0.07864749 0.12038329 0.14131349]
mask final/base 0.6770677842906884 0.6012763727406816 phases
0 111 0.807 0.777
111 200 0.742 0.708
200 300 0.6 0.489
3 …[+99 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed "s/final_proto_poses.npy/final_proto_poses2.npy/; s/final_proto_eval.npy/final_proto_eval2.npy/" scratch/eval_final_proto.py > scratch/eval_final_proto2.py
python scratch/eval_final_proto2.py'
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
final iou/ncc [0.67224874 0.44384714] base [0.58922112 0.38547372] score final 0.8094218889155167 base 0.7133624113283966
0 111 [0.792 0.76  0.748 0.674]
111 200 [0.738 0.483 0.687 0.484]
200 300 [0.598 0.38  0.486 0.255]
300 400 [ 0.467 -0.073  0.306 -0.252]
400 500 [0.564 0.232 0.483 0.219]
500 600 [0.701 0.502 0.588 0.531]
600 716 [0.724 0.507 0.65  0.439]
716 803 [0.795 0.762 0.774 0.753]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/tune_smoothing.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
# load HMM state path from current build using same logic
D=np.load('scratch/final_rot_scores.npz');Pbase=D['Pbase'].astype(np.float64);P4=D['Pw'].astype(np.float64);iou=D['iou'];F=len(Pbase);fulln=np.zeros((F,4));fulln[:,[0,3]]=np.load('scratch/full_fz_scores.npy');fulln[:,[1,2]]=np.load('scratch/full_fxfy_scores.npy');names=list(D['names']);FC=np.load('scratch/fit_candidates_all.npz');Pfit=FC['Pw'].astype(np.float64)
e=iou+.5*np.maximum(fulln,-.5);M=4;ang=np.zeros((F-1,M,M))
for t in range(F-1):
 for a in range(M):
  for b in range(M):ang[t,a,b]=Rot.from_matrix(P4[t,a,:3,:3].T@P4[t+1,b,:3,:3]).magnitude()*180/np.pi
dp=np.zeros((F,M));bk=np.zeros((F,M),int);dp[0]=e[0]
for t in range(1,F):
 for b in range(M):
  vals=dp[t-1]-.3*ang[t-1,:,b]/180;a=int(np.argmax(vals));dp[t,b]=vals[a]+e[t,b];bk[t,b]=a
s=int(np.argmax(dp[-1]));state=np.zeros(F,int);state[-1]=s
for t in range(F-1,0,-1):state[t-1]=bk[t,state[t]]
Pi=np.load('scratch/static_init2_optpose.npy');Pf=np.load('scratch/static_final_optpose.npy')
base=Pbase.copy()
for i in range(F):
 if state[i]"'!=0:base[i]=Pfit[i,state[i]]
variants={}
for win in [0,3,5,7]:
 for passes in [0,1,2]:
  T=base.copy();p=T[:,:3,3].copy()
  if win:
   for k in range(3):p[111:746,k]=savgol_filter(p[111:746,k],win,2,mode='"'interp')
  …[+3032 chars]
```
<details><summary>output</summary>

```
w0p0 iou 0.7212 moving 0.6992 rot>,60 38 q95rot 52.4 q95dp 0.126
w0p1 iou 0.6877 moving 0.6569 rot>,60 23 q95rot 39.6 q95dp 0.126
w0p2 iou 0.6738 moving 0.6394 rot>,60 0 q95rot 36.7 q95dp 0.126
w3p0 iou 0.7212 moving 0.6992 rot>,60 38 q95rot 52.4 q95dp 0.126
w3p1 iou 0.6877 moving 0.6569 rot>,60 23 q95rot 39.6 q95dp 0.126
w3p2 iou 0.6738 moving 0.6394 rot>,60 0 q95rot 36.7 q95dp 0.126
w5p0 iou 0.7079 moving 0.6825 rot>,60 38 q95rot 52.4 q95dp 0.079
w5p1 iou 0.6901 moving 0.66 rot>,60 23 q95rot 39.6 q95dp 0.079
w5p2 iou 0.6771 moving 0.6434 rot>,60 0 q95rot 36.7 q95dp 0.079
w7p0 iou 0.7036 moving 0.6769 rot>,60 38 q95rot 52.4 q95dp 0.061
w7p1 iou 0.6913 moving 0.6614 rot>,60 23 q95rot 39.6 q95dp 0.061
w7p2 iou 0.6787 moving 0.6455 rot>,60 0 q95rot 36.7 q95dp 0.061

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/tune_ramps.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot,Slerp
from scipy.signal import savgol_filter
D=np.load('scratch/final_rot_scores.npz');Pbase=D['Pbase'].astype(np.float64);P4=D['Pw'].astype(np.float64);iou=D['iou'];F=len(Pbase);fulln=np.zeros((F,4));fulln[:,[0,3]]=np.load('scratch/full_fz_scores.npy');fulln[:,[1,2]]=np.load('scratch/full_fxfy_scores.npy');FC=np.load('scratch/fit_candidates_all.npz');Pfit=FC['Pw'].astype(np.float64)
Pi=np.load('scratch/static_init2_optpose.npy');Pf=np.load('scratch/static_final_optpose.npy');e=iou+.5*np.maximum(fulln,-.5);M=4
ang=np.zeros((F-1,M,M))
for t in range(F-1):
 for a in range(M):
  for b in range(M):ang[t,a,b]=Rot.from_matrix(P4[t,a,:3,:3].T@P4[t+1,b,:3,:3]).magnitude()*180/np.pi
def statepath(alpha):
 d=np.zeros((F,M));bk=np.zeros((F,M),int);d[0]=e[0]
 for t in range(1,F):
  for b in range(M):
   vals=d[t-1]-alpha*ang[t-1,:,b]/180;a=int(np.argmax(vals));d[t,b]=vals[a]+e[t,b];bk[t,b]=a
 s=int(np.argmax(d[-1]));path=np.zeros(F,int);path[-1]=s
 for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
 return path
def pose_from_state(path):
 T=Pbase.copy()
 for i in range(F):
  if path[i]"'!=0:T[i]=Pfit[i,path[i]]
 p=T[:,:3,3].copy()
 for k in range(3):p[111:746,k]=savgol_filter(p[111:746,k],5,2,mode='"'interp')
 T[:,:3,3]=p;return T
def add_statics(T):
 Tm=T.copy()
 for i in range(111):T[i]=Pi
 for i in range(746,F):T[i …[+3049 chars]
```
<details><summary>output</summary>

```
a0.3_h1 iou 0.6977 moving 0.6695 >60 57 q95 80.2 dp95 0.077
a0.3_h2 iou 0.684 moving 0.6521 >60 6 q95 44.0 dp95 0.071
a0.3_h3 iou 0.6657 moving 0.629 >60 6 q95 29.4 dp95 0.069
a0.3_h4 iou 0.6573 moving 0.6183 >60 5 q95 23.9 dp95 0.066
a0.5_h1 iou 0.7016 moving 0.6744 >60 41 q95 66.0 dp95 0.077
a0.5_h2 iou 0.6948 moving 0.6657 >60 6 q95 43.8 dp95 0.07
a0.5_h3 iou 0.682 moving 0.6496 >60 6 q95 29.9 dp95 0.069
a0.5_h4 iou 0.6794 moving 0.6463 >60 5 q95 27.1 dp95 0.065
a0.8_h1 iou 0.7005 moving 0.6731 >60 34 q95 51.6 dp95 0.077
a0.8_h2 iou 0.6953 moving 0.6664 >60 6 q95 41.2 dp95 0.071
a0.8_h3 iou 0.6897 moving 0.6593 >60 6 q95 29.4 dp95 0.066
a0.8_h4 iou 0.6836 moving 0.6516 >60 6 q95 28.6 dp95 0.063

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('scratch/ramp_variants.npz');T=D['a0.8_h2'];rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi
for i in np.argsort(rel)[-20:][::-1]:print(i,round(rel[i],1))
PY"
```
<details><summary>output</summary>

```
319 81.2
658 78.1
473 71.3
195 70.1
604 69.4
476 68.4
601 59.2
469 58.0
360 57.0
118 55.0
356 53.8
169 52.6
324 50.3
202 49.9
262 49.5
296 47.3
557 46.2
117 45.4
714 45.1
276 45.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/outlier_repair.py <<'PY'
import numpy as np,cv2,h5py,torch,trimesh,json,nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot,Slerp
D=np.load('scratch/ramp_variants.npz');T0=D['a0.8_h2'].copy();F=len(T0)
def repair(T,th=60):
 T=T.copy()
 for it in range(3):
  rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi;j=np.argmax(rel)
  if rel[j]<=th:break
  t=int(j);a=max(0,t-2);b=min(F-1,t+3)
  if b<=a+1:break
  L=T[a].copy();R=T[b].copy();sl=Slerp([0,1],Rot.concatenate([Rot.from_matrix(L[:3,:3]),Rot.from_matrix(R[:3,:3])]))
  for i in range(a+1,b):
   w=(i-a)/(b-a);T[i,:3,:3]=sl(w).as_matrix();T[i,:3,3]=(1-w)*L[:3,3]+w*R[:3,3]
  print('repair',it,t,rel[j])
 return T
for th in [60,55,50]:
 T=repair(T0,th);rel=Rot.from_matrix(T[:-1,:3,:3].transpose(0,2,1)@T[1:,:3,:3]).magnitude()*180/np.pi;print('th',th,'max',rel.max(),'>60',sum(rel>60),'>50',sum(rel>50));np.save(f'scratch/repair_{th}.npy',T)
PY
python scratch/outlier_repair.py"
```
<details><summary>output</summary>

```
repair 0 319 81.24705019395023
repair 1 658 78.05056845423566
repair 2 473 71.34759697656109
th 60 max 70.0693137050355 >60 3 >50 10
repair 0 319 81.24705019395023
repair 1 658 78.05056845423566
repair 2 473 71.34759697656109
th 55 max 70.0693137050355 >60 3 >50 10
repair 0 319 81.24705019395023
repair 1 658 78.05056845423566
repair 2 473 71.34759697656109
th 50 max 70.0693137050355 >60 3 >50 10

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='scratch/outlier_repair.py';s=open(p).read();s=s.replace('for it in range(3):','for it in range(20):');open('scratch/outlier_repair20.py','w').write(s)
PY
python scratch/outlier_repair20.py"
```
<details><summary>output</summary>

```
repair 0 319 81.24705019395023
repair 1 658 78.05056845423566
repair 2 473 71.34759697656109
repair 3 195 70.0693137050355
repair 4 604 69.420582687613
repair 5 476 68.39903572492646
th 60 max 59.238260094855825 >60 0 >50 7
repair 0 319 81.24705019395023
repair 1 658 78.05056845423566
repair 2 473 71.34759697656109
repair 3 195 70.0693137050355
repair 4 604 69.420582687613
repair 5 476 68.39903572492646
repair 6 601 59.238260094855825
repair 7 469 58.01790841208157
repair 8 360 57.01658693592365
th 55 max 54.970615443724775 >60 0 >50 4
repair 0 319 81.24705019395023
repair 1 658 78.05056845423566
repair 2 473 71.34759697656109
repair 3 195 70.0693137050355
repair 4 604 69.420582687613
repair 5 476 68.39903572492646
repair 6 601 59.238260094855825
repair 7 469 58.01790841208157
repair 8 360 …[+190 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_repaired.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,torch,nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
Tvars={k:np.load(f'scratch/repair_{k}.npy') for k in [60,55,50]};F=len(next(iter(Tvars.values())));cam=json.load(open('inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices.astype(np.float32);Ft0=g.faces.astype(np.int32);W,H=768,576;fx,fy=K[0,0]/2,K[1,1]/2;cx,cy=K[0,2]/2,K[1,2]/2;ctx=dr.RasterizeCudaContext();Vt=torch.from_numpy(V).cuda();Ft=torch.from_numpy(Ft0).cuda()
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def masks(Ps):
 Ps=np.asarray(Ps,np.float32);B=len(Ps);v=(torch.from_numpy(Ps[:,:3,:3]).cuda()@Vt.T).transpose(1,2)+torch.from_numpy(Ps[:,:3,3]).cuda()[:,None];x,y,z=v[:,:,0],v[:,:,1],v[:,:,2];u=fx*x/z+cx;vv=fy*y/z+cy;clip=torch.stack([2*u/W-1,2*vv/H-1,torch.zeros_like(z),torch.ones_like(z)],2);rast,_=dr.rasterize(ctx,clip,Ft,(H,W));return (rast[...,3]>0).cpu().numpy()
res={k:np.zeros(F) for k in Tvars}
for s0 in range(0,F,16):
 e0=min(F,s0+16);Ps=[];meta=[]
 for i in range(s0,e0):
  om=cv2.resize(omall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.resize(pmall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv2.dilate(pm. …[+369 chars]
```
<details><summary>output</summary>

```
60 iou 0.6956997882120595 0.6668941211935715
55 iou 0.6952473132699977 0.6663219363455785
50 iou 0.6976565379862997 0.6693685622466657

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/make_final_montage.py <<'PY'
import numpy as np,cv2,trimesh,json,h5py
from PIL import Image
T=np.load('scratch/repair_50.npy');B=np.load('inputs/reconstruction.npz')['T_world_mesh'];cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices;cap=cv2.VideoCapture('inputs/video.mp4');frames=[]
for i in range(803):ok,im=cap.read();frames.append(im)
with h5py.File('inputs/masks.h5','r') as h:om=h['object'][:]
def proj(Q):
 P=Tcw@Q;vc=P[:3,:3]@V.T+P[:3,3:4];uv=(K@vc).T;return (uv[:,:2]/uv[:,2:3]).astype(np.float32)
for part,(a,b) in enumerate([(0,410),(410,803)]):
 th=[]
 for i in range(a,b,8):
  im=frames[i].copy();ys,xs=np.where(om[i]);cx0,cy0=(xs.mean(),ys.mean()) if len(xs) else (768,576);half=180;x0=int(np.clip(cx0-half,0,1536-2*half));y0=int(np.clip(cy0-half,0,1152-2*half));crop=im[y0:y0+2*half,x0:x0+2*half].copy()
  for Q,col in [(B[i],(0,0,255)),(T[i],(0,255,0))]:
   uv=proj(Q)-[x0,y0];cv2.polylines(crop,[cv2.convexHull(uv).astype(np.int32)],True,col,2)
  cs,_=cv2.findContours(om[i][y0:y0+2*half,x0:x0+2*half].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,cs,-1,(255,0,255),1);cv2.putText(crop,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);th.append(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((300,300)))
 cols=5;rows=(len(th)+cols-1)// …[+237 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/make_final_montage.py", line 13, in <module>
    uv=proj(Q)-[x0,y0];cv2.polylines(crop,[cv2.convexHull(uv).astype(np.int32)],True,col,2)
                                           ^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgproc/src/convhull.cpp:143: error: (-215:Assertion failed) total >= 0 && (depth == CV_32F || depth == CV_32S) in function 'convexHull'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/uv=proj(Q)-\\[x0,y0\\]/uv=(proj(Q)-np.array([x0,y0],np.float32)).astype(np.float32)/' scratch/make_final_montage.py
python scratch/make_final_montage.py"
```
<details><summary>output</summary>

```
0 (1500, 3300)
1 (1500, 3000)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
D=np.load('scratch/final_rot_scores.npz');P=D['Pw'];F=len(P);iou=D['iou'];fulln=np.zeros((F,4));fulln[:,[0,3]]=np.load('scratch/full_fz_scores.npy');fulln[:,[1,2]]=np.load('scratch/full_fxfy_scores.npy');e=iou+.5*np.maximum(fulln,-.5);M=4;alpha=.8;names=list(D['names'])
ang=np.zeros((F-1,M,M))
for t in range(F-1):
 for a in range(M):
  for b in range(M):ang[t,a,b]=Rot.from_matrix(P[t,a,:3,:3].T@P[t+1,b,:3,:3]).magnitude()*180/np.pi
d=np.zeros((F,M));bk=np.zeros((F,M),int);d[0]=e[0]
for t in range(1,F):
 for b in range(M):
  vals=d[t-1]-alpha*ang[t-1,:,b]/180;a=int(np.argmax(vals));d[t,b]=vals[a]+e[t,b];bk[t,b]=a
s=int(np.argmax(d[-1]));path=np.zeros(F,int);path[-1]=s
for t in range(F-1,0,-1):path[t-1]=bk[t,path[t]]
ch=np.where(path[1:]"'!=path[:-1])[0]+1;runs=[];st=0
for z in list(ch)+[F]:runs.append((st,z-1,int(path[st]),names[path[st]]));st=z
print(runs)
PY'
```
<details><summary>output</summary>

```
[(0, 117, 0, 'base'), (118, 125, 1, 'fx'), (126, 265, 0, 'base'), (266, 275, 2, 'fy'), (276, 314, 0, 'base'), (315, 331, 2, 'fy'), (332, 344, 1, 'fx'), (345, 411, 3, 'fz'), (412, 447, 0, 'base'), (448, 456, 2, 'fy'), (457, 484, 0, 'base'), (485, 492, 2, 'fy'), (493, 560, 0, 'base'), (561, 566, 1, 'fx'), (567, 685, 3, 'fz'), (686, 744, 0, 'base'), (745, 746, 3, 'fz'), (747, 802, 0, 'base')]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p outputs && cat > outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Deterministic monocular 6-DoF baseball-bat trajectory refinement.

The script keeps the released CARI4D pose rotations as the per-frame hypothesis,
then applies three kinds of trajectory evidence/constraints:

1. Long floor-contact phases are pooled over all frames and one rigid pose is
   fitted to the union object mask and median unoccluded appearance.
2. In the carried phase, translation is re-fitted against the SAM object mask
   while explicitly excluding pixels covered by the person.  The fit is then
   low-pass filtered and combined with a constant-velocity/smoothness prior.
3. A small set of rotation branch corrections was selected from full-resolution
   render-and-compare.  These are encoded as explicit frame intervals below;
   the comment lists the evidence.  Branch switches are ramped over five frames
   and remaining >50 degree/frame outliers are repaired by geodesic
   interpolation.  This avoids net-new rigid-motion teleports.

No network access, dataset annotations, or development scratch files are used.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot, Slerp

import nvdiffrast.torch as dr

# Pyrender is used only for the two static pooled-pose fits.  EGL is available
# in the released co …[+18759 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "/usr/bin/time -f 'ELAPSED=%e MAXRSS_KB=%M' python outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'time python outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
/work/outputs/solve.py:145: RuntimeWarning: All-NaN slice encountered
  pooled = np.nanmedian(
Wrote /work/outputs/object_poses_world.npz with shape (803, 4, 4)

real	1m29.753s
user	2m18.242s
sys	1m41.614s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
out=np.load('outputs/object_poses_world.npz')['T_world_mesh'];ref=np.load('scratch/repair_50.npy');print(out.shape,out.dtype,np.isfinite(out).all())
print('max pos diff',np.abs(out[:,:3,3]-ref[:,:3,3]).max(),'mean',np.linalg.norm(out[:,:3,3]-ref[:,:3,3],axis=1).mean())
d=Rot.from_matrix(ref[:,:3,:3].transpose(0,2,1)@out[:,:3,:3]).magnitude()*180/np.pi
print('rot diff q',np.quantile(d,[0,.5,.9,.95,.99,1]))
rel=Rot.from_matrix(out[:-1,:3,:3].transpose(0,2,1)@out[1:,:3,:3]).magnitude()*180/np.pi;dp=np.linalg.norm(np.diff(out[:,:3,3],axis=0),axis=1)
print('final temporal rot',np.quantile(rel,[.5,.9,.95,.99,1]),'trans',np.quantile(dp,[.5,.9,.95,.99,1]))
print('det range',np.linalg.det(out[:,:3,:3]).min(),np.linalg.det(out[:,:3,:3]).max())
PY"
```
<details><summary>output</summary>

```
(803, 4, 4) float64 True
max pos diff 0.03226135798863006 mean 0.001168025399160124
rot diff q [1.32423097e-15 5.34011827e-07 1.19010916e+01 1.72971335e+01
 1.72971335e+01 1.72971335e+01]
final temporal rot [ 7.20217409 28.85265731 37.78948003 44.99000466 49.85596611] trans [0.0202849  0.05601404 0.06544337 0.08877371 0.13485203]
det range 0.9999999999999976 1.0000000000000024

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -lh outputs/object_poses_world.npz; sha256sum outputs/object_poses_world.npz outputs/solve.py'
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 101K Oct 20 20:57 outputs/object_poses_world.npz
e41ea97303af47f121639e363a33636c371523b2583db604e024204661d5741d  outputs/object_poses_world.npz
4982f1b1d5216fc8d797f916974639b763c3fdede79ddd05bc5ee2f718b5ada2  outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
out=np.load('outputs/object_poses_world.npz')['T_world_mesh'];ref=np.load('scratch/repair_50.npy');d=Rot.from_matrix(ref[:,:3,:3].transpose(0,2,1)@out[:,:3,:3]).magnitude()*180/np.pi
for i in np.where(d>1)[0]:print(i,round(float(d[i]),2),np.round(out[i,:3,3],3),np.round(ref[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
0 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
1 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
2 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
3 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
4 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
5 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
6 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
7 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
8 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
9 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
10 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
11 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
12 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
13 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
14 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
15 11.9 [0.216 1.24  2.409] [0.215 1.24  2.408]
16 11.9 [0.216 1.24  2.409] [0.215 1.24  2 …[+7814 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/eval_solution.py <<'PY'
import numpy as np,trimesh,json,cv2,h5py,os
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
Ts=[np.load('outputs/object_poses_world.npz')['T_world_mesh'],np.load('scratch/repair_50.npy'),np.load('inputs/reconstruction.npz')['T_world_mesh']];names=['solution','devref','baseline'];F=len(Ts[0]);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);W,H=768,576;renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0]/2,K[1,1]/2,K[0,2]/2,K[1,2]/2,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4));cap=cv2.VideoCapture('inputs/video.mp4')
with h5py.File('inputs/masks.h5','r') as h:omall=h['object'][:];pmall=h['person'][:]
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m]
  if len(x)<20:return -1
  x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
out=np.zeros((F,6))
for i in range(F):
 ok,im=cap.read();im=cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)/255;om=cv2.resize(omall[i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);pm=cv …[+936 chars]
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
solution iou 0.6894876332018685 ncc 0.45108534620054347 score 0.8279092717055888
  0 111 [0.797 0.73 ]
  111 200 [0.769 0.475]
  200 300 [0.626 0.363]
  300 400 [ 0.522 -0.043]
  400 500 [0.599 0.318]
  500 600 [0.7   0.478]
  600 746 [0.748 0.57 ]
  746 803 [0.752 0.772]
devref iou 0.6908970468345338 ncc 0.45318357297868184 score 0.8299600484248836
  0 111 [0.792 0.76 ]
  111 200 [0.77  0.477]
  200 300 [0.626 0.362]
  300 400 [ 0.512 -0.045]
  400 500 [0.599 0.319]
  500 600 [0.7   0.478]
  600 746 [0.747 0.57 ]
  746 803 [0.798 0.746]
baseline iou 0.5892211224363754 ncc 0.3854737203624967 score 0.7133624113283966
  0 111 [0.748 0.674]
  111 200 [0.687 0.484]
  200 300 [0.486 0.255]
  300 400 [ 0.306 -0.252]
  400 500 [0.483 0.219]
  500 600 [0.588 0.531 …[+50 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > scratch/static_final_multistart.py <<'PY'
import numpy as np,trimesh,json,cv2,os
from scipy.spatial.transform import Rotation as Rot
os.environ['PYOPENGL_PLATFORM']='egl';import pyrender
r=np.load('inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);cam=json.load(open('inputs/camera.json'));K=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
g=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry();mesh=pyrender.Mesh.from_trimesh(g,smooth=False);flip=np.diag([1,-1,-1,1.]);d0=np.load('scratch/static_final2.npz');om=d0['obj'].astype(bool);pool=d0['pooled'].astype(float)/255;H,W=om.shape;x0,x1,y0,y1=d0['bbox'];renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
def ncc(a,b,m):
 vv=[];ww=[]
 for c in range(3):
  x=a[...,c][m];y=b[...,c][m];x=x-x.mean();y=y-y.mean();vv.append(x/(x.std()+1e-5));ww.append(y/(y.std()+1e-5))
 return float(np.mean(np.concatenate(vv)*np.concatenate(ww)))
def obj(P):
 scene.set_pose(node,pose=flip@Tcw@P);col,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA);m=depth>1e-5;inter=m&om;union=m|om;iou=inter.sum()/max(1,union.sum());valid=om&(pool.sum(2)>0);tex=ncc(col[...,:3].astype(float)/255,pool,valid);return iou+.3*tex,iou,tex
def avg(a,b):
 A=T[a:b];U,S,Vt=np.li …[+1555 chars]
```
<details><summary>output</summary>

```

start avg746 raw (0.9295069537072095, 0.746862674793833, 0.6088142630445885)
opt avg746 (0.9918371675780964, 0.759199714183637, 0.7754581779815312) [0.2778953  1.23843145 2.91750327]

start avg716 raw (0.9301104145336841, 0.7457870204374327, 0.6144113136541715)
opt avg716 (1.0020191023355867, 0.7699178277956413, 0.7736709151331513) [0.27808788 1.23804688 2.9133576 ]

start avg700 raw (0.9366957543066332, 0.7416815742397138, 0.6500472668897311)
opt avg700 (1.0152167806665728, 0.7777383906416164, 0.791594633416521) [0.27808154 1.23796797 2.91449381]

start frame750 raw (0.9262994653779572, 0.7429186088203658, 0.6112695218586381)
opt frame750 (0.9924478936552864, 0.7541100786275912, 0.7944593834256509) [0.28038066 1.24063611 2.92576589]

start frame800 raw (0.9322513792155005, 0.749013983506 …[+390 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > scratch/static_init_multistart.py <<'PY'
exec(open('scratch/static_final_multistart.py').read().split(\"starts={'avg746'\")[0])
d0=np.load('scratch/static_init2.npz');om=d0['obj'].astype(bool);pool=d0['pooled'].astype(float)/255;H,W=om.shape;x0,x1,y0,y1=d0['bbox'];renderer=pyrender.OffscreenRenderer(W,H,point_size=1);camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0,znear=.01,zfar=100);scene=pyrender.Scene(bg_color=[0,0,0],ambient_light=[1,1,1]);node=scene.add(mesh);scene.add(camera,pose=np.eye(4))
starts={'avg0_111':avg(0,111),'avg0_105':avg(0,105),'avg0_90':avg(0,90),'frame0':T[0].copy(),'frame50':T[50].copy(),'frame100':T[100].copy()}
best=None
for n,P0 in starts.items():
 print('\\nstart',n,'raw',obj(P0),flush=True);P=opt(P0);score=obj(P);print('opt',n,score,P[:3,3],flush=True)
 if best is None or score[0]>best[0]:best=(score[0],P.copy(),n,score)
print('BEST',best[2],best[3]);np.save('scratch/static_init_bestpose.npy',best[1]);renderer.delete()
PY
python scratch/static_init_multistart.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_init_multistart.py", line 6, in <module>
    print('\nstart',n,'raw',obj(P0),flush=True);P=opt(P0);score=obj(P);print('opt',n,score,P[:3,3],flush=True)
                            ^^^^^^^
  File "<string>", line 12, in obj
  File "/opt/conda/lib/python3.11/site-packages/pyrender/offscreen.py", line 86, in render
    self._platform.make_current()
  File "/opt/conda/lib/python3.11/site-packages/pyrender/platforms/egl.py", line 196, in make_current
    assert eglMakeCurrent(
           ^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/OpenGL/error.py", line 228, in glCheckError
    raise GLError(
OpenGL.error.GLError: GLError(
	err = 12296,
	baseOperation = eglMakeCurrent,
	cArguments = (
		<OpenGL._opaque.EGLDispla …[+247 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='outputs/solve.py';s=open(p).read()
s=s.replace(\"def optimize_static_pose(\\n    mesh, Tbase, Twc, Tcw, K, union_mask, pooled_rgb, bbox, start, end\\n):\", \"def optimize_static_pose(\\n    mesh, Tbase, Twc, Tcw, K, union_mask, pooled_rgb, bbox, start, end,\\n    init_pose=None,\\n):\")
s=s.replace(\"    P = avg_pose(Tbase, start, end)\\n\", \"    P = avg_pose(Tbase, start, end) if init_pose is None else init_pose.copy()\\n\")
s=s.replace(\"        mesh, Tbase, Twc, Tcw, K, union_i, pool_i, bbox_i,\\n        *INITIAL_STATIC,\\n    )\", \"        mesh, Tbase, Twc, Tcw, K, union_i, pool_i, bbox_i,\\n        *INITIAL_STATIC, init_pose=avg_pose(Tbase, 0, 105),\\n    )\")
s=s.replace(\"        mesh, Tbase, Twc, Tcw, K, union_f, pool_f, bbox_f,\\n        *FINAL_STATIC,\\n    )\", \"        mesh, Tbase, Twc, Tcw, K, union_f, pool_f, bbox_f,\\n        *FINAL_STATIC, init_pose=avg_pose(Tbase, 700, 803),\\n    )\")
open(p,'w').write(s)
PY
python -m py_compile outputs/solve.py
grep -n 'optimize_static_pose\\|init_pose=' outputs/solve.py | head"
```
<details><summary>output</summary>

```
166:def optimize_static_pose(
168:    init_pose=None,
450:    initial_pose = optimize_static_pose(
452:        *INITIAL_STATIC, init_pose=avg_pose(Tbase, 0, 105),
457:    final_pose = optimize_static_pose(
459:        *FINAL_STATIC, init_pose=avg_pose(Tbase, 700, 803),

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'time python outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
/work/outputs/solve.py:145: RuntimeWarning: All-NaN slice encountered
  pooled = np.nanmedian(
Wrote /work/outputs/object_poses_world.npz with shape (803, 4, 4)

real	1m17.792s
user	2m3.628s
sys	1m40.550s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python scratch/eval_solution.py'
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
solution iou 0.6906405325958888 ncc 0.4589048978856417 score 0.8314080366051386
  0 111 [0.797 0.749]
  111 200 [0.77  0.476]
  200 300 [0.626 0.363]
  300 400 [ 0.522 -0.043]
  400 500 [0.599 0.318]
  500 600 [0.7   0.478]
  600 746 [0.748 0.57 ]
  746 803 [0.767 0.843]
devref iou 0.6908970468345338 ncc 0.45318357297868184 score 0.8299600484248836
  0 111 [0.792 0.76 ]
  111 200 [0.77  0.477]
  200 300 [0.626 0.362]
  300 400 [ 0.512 -0.045]
  400 500 [0.599 0.319]
  500 600 [0.7   0.478]
  600 746 [0.747 0.57 ]
  746 803 [0.798 0.746]
baseline iou 0.5892211224363754 ncc 0.3854737203624967 score 0.7133624113283966
  0 111 [0.748 0.674]
  111 200 [0.687 0.484]
  200 300 [0.486 0.255]
  300 400 [ 0.306 -0.252]
  400 500 [0.483 0.219]
  500 600 [0.588 0.531] …[+49 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
E=np.load('scratch/eval_solution.npy');T=np.load('outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('inputs/reconstruction.npz')['T_world_mesh']
for name,A in [('baseline',B),('result',T)]:
 rel=Rot.from_matrix(A[:-1,:3,:3].transpose(0,2,1)@A[1:,:3,:3]).magnitude()*180/np.pi;dp=np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)
 print(name,'rot q',np.round(np.quantile(rel,[.5,.9,.95,.99,1]),3),'rot>60/100',int(sum(rel>60)),int(sum(rel>100)),'trans q',np.round(np.quantile(dp,[.5,.9,.95,.99,1]),3))
 for a,b in [(0,111),(746,803)]:
  R=A[a:b,:3,:3];p=A[a:b,:3,3];U,S,Vt=np.linalg.svd(R.sum(0));Rn=U@Vt
  if np.linalg.det(Rn)<0:U[:,-1]*=-1;Rn=U@Vt
  dev=Rot.from_matrix(R.transpose(0,2,1)@Rn).magnitude()*180/np.pi
  print(' ',a,b,'rot dev q',np.round(np.quantile(dev,[.5,.9,.99,1]),3),'pos std',np.round(p.std(0),4),'pos range',np.round(p.max(0)-p.min(0),4))
print('overall metrics solution iou/ncc',E[:,:2].mean(0),'baseline',E[:,4:].mean(0))
PY"
```
<details><summary>output</summary>

```
baseline rot q [  6.279  24.939  29.957  71.336 163.251] rot>60/100 14 4 trans q [0.018 0.056 0.078 0.113 0.16 ]
  0 111 rot dev q [ 1.485 10.341 20.066 35.618] pos std [0.0035 0.009  0.018 ] pos range [0.0136 0.03   0.0677]
  746 803 rot dev q [ 0.498  0.498 13.105 29.149] pos std [0.0003 0.0012 0.0027] pos range [0.002  0.0048 0.0089]
result rot q [ 7.202 28.853 37.789 44.99  49.856] rot>60/100 0 0 trans q [0.02  0.056 0.065 0.089 0.135]
  0 111 rot dev q [0. 0. 0. 0.] pos std [0. 0. 0.] pos range [0. 0. 0.]
  746 803 rot dev q [ 1.9    1.9   37.552 43.975] pos std [0.0001 0.0004 0.0014] pos range [0.0007 0.0022 0.0078]
overall metrics solution iou/ncc [0.69064053 0.4589049 ] baseline [0.58922112 0.38547372]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('outputs/object_poses_world.npz')['T_world_mesh'];R=T[746:,:3,:3];U,S,Vt=np.linalg.svd(R.sum(0));Rn=U@Vt
if np.linalg.det(Rn)<0:U[:,-1]*=-1;Rn=U@Vt
d=Rot.from_matrix(R.transpose(0,2,1)@Rn).magnitude()*180/np.pi
for j in np.argsort(d)[-10:][::-1]:print(746+j,d[j])
PY"
```
<details><summary>output</summary>

```
746 43.97464327947495
747 32.50605921269229
748 21.03747514590964
749 9.56889107912698
798 1.8996929876556938
801 1.8996929876556938
800 1.8996929876556938
799 1.8996929876556938
802 1.8996929876556938
793 1.8996929876556938

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import scipy,h5py,pyrender,numpy,cv2,torch,trimesh
print('scipy',scipy.__version__);print('h5py',h5py.__version__);print('pyrender',pyrender.__version__)
PY"
```
<details><summary>output</summary>

```
scipy 1.16.3
h5py 3.16.0
pyrender 0.1.45

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > outputs/REPORT.md <<'MD'
# Baseball-bat 6-DoF trajectory recovery

## Result

- Video: "'`inputs/video.mp4`, 803 frames at 30 fps, 1536 x 1152.
- Object: textured baseball bat, mesh-file coordinates.
- Output: `outputs/object_poses_world.npz`, key `T_world_mesh`, shape `(803, 4, 4)`, dtype `float64`.
- Released-frame convention and `camera.json` were used throughout.
- No ground-truth trajectory or challenge annotation was used.

The released CARI4D rotations were retained as the main per-frame hypothesis. The
largest gains came from (1) pooling each floor-contact phase into one rigid pose,
(2) re-fitting translation against person-excluded object masks, and (3) replacing
symmetry-branch errors selected by textured render-and-compare. All branch/model
choices are deterministic and encoded in `solve.py`.

## Phases found

| Frames | Phase | Evidence | Confidence |
|---:|---|---|---|
| 0-110 | Initial rest on floor; person approaches | Object mask and midpoint are static; zero contact probability; pooled texture/mask is sharp | High, 0.95 |
| 111-199 | Pickup and first lift | Contact begins around 109-112; mask motion increases | Medium-high, 0.80 |
| 200-314 | Slow carry and turning | Object moves with hand; several tracker branch changes | Medium, 0.65 |
| 315-411 | End-for-end rotation and body turn | Large object-mask orientation changes; textured rendering resolves reversed barrel/handle | Medium-low, 0.58 |
| 412-599 | Continued carry/turn | Mostly …[+8047 chars]
```
<details><summary>output</summary>

```
outputs/REPORT.md 7429
outputs/TOOLS.md 1850
outputs/solve.py 20257
outputs/object_poses_world.npz 103062

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json, hashlib, os
p='outputs/object_poses_world.npz'
with np.load(p) as z:
 print('keys',z.files)
 T=z['T_world_mesh']
 print('shape',T.shape,'dtype',T.dtype,'finite',np.isfinite(T).all())
 print('bottom rows max err',np.abs(T[:,3]-np.array([0,0,0,1])).max())
 print('rotation orthogonality max err',np.abs(T[:,:3,:3].transpose(0,2,1)@T[:,:3,:3]-np.eye(3)).max())
 print('det range',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
print('sha256',hashlib.sha256(open(p,'rb').read()).hexdigest())
PY
ls -la outputs"
```
<details><summary>output</summary>

```
keys ['T_world_mesh']
shape (803, 4, 4) dtype float64 finite True
bottom rows max err 0.0
rotation orthogonality max err 2.886579864025407e-15
det range 0.9999999999999976 1.0000000000000024
sha256 209b205c2319b69a1d73528321a8e2c562da6a399be575e3c1789a68b0891ad5
total 148
drwxrwxr-x 3 hanzhou hanzhou   4096 Oct 20 21:06 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 20 19:38 ..
-rw-rw-r-- 1 hanzhou hanzhou   7429 Oct 20 21:06 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   1850 Oct 20 21:06 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 20 21:02 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 103062 Oct 20 21:03 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  20257 Oct 20 21:02 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
E=np.load('scratch/eval_solution.npy')
for a,b in [(0,111),(111,200),(200,300),(300,400),(400,500),(500,600),(600,746),(746,803),(0,803)]:
 s=(E[a:b,0]+.3*np.maximum(E[a:b,1],-.5)).mean();base=(E[a:b,4]+.3*np.maximum(E[a:b,5],-.5)).mean();print(a,b,round(float(s-base),3),round(float(base),3),round(float(s),3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
p=Path('outputs/REPORT.md');s=p.read_text();s=s.replace('| 200-299 | 0.486 | 0.626 | 0.255 | 0.363 | +0.172 |','| 200-299 | 0.486 | 0.626 | 0.255 | 0.363 | +0.164 |');s=s.replace('| 300-399 | 0.306 | 0.522 | -0.252 | -0.043 | +0.279 |','| 300-399 | 0.306 | 0.522 | -0.252 | -0.043 | +0.249 |');p.write_text(s)
d=Path('outputs/__pycache__')
if d.exists():
 for f in d.iterdir(): f.unlink()
 d.rmdir()
PY
sha256sum outputs/object_poses_world.npz outputs/solve.py outputs/REPORT.md outputs/TOOLS.md
ls -la outputs"
```
<details><summary>output</summary>

```
209b205c2319b69a1d73528321a8e2c562da6a399be575e3c1789a68b0891ad5  outputs/object_poses_world.npz
a5b3ba915231fb931231e6cbbeb7ae472ca30ac51d1d410465506ef91449978c  outputs/solve.py
b3ea594c2751917b9d8a6d3a0baa2b7c6cde34f5dd266058d4bfc2d301e6c1bf  outputs/REPORT.md
97f62c651194c19ef479f63b3d6a4cfc28239ef1002f3e77080ad70706465c13  outputs/TOOLS.md
total 144
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 20 21:07 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 20 19:38 ..
-rw-rw-r-- 1 hanzhou hanzhou   7429 Oct 20 21:07 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   1850 Oct 20 21:06 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 103062 Oct 20 21:03 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  20257 Oct 20 21:02 solve.py

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/object_poses_world.npz`  
  `T_world_mesh`: `(803, 4, 4)`, `float64`, finite, valid SO(3) rotations.
- `/work/outputs/solve.py`  
  Deterministic offline solver using only `/work/inputs`.
- `/work/outputs/REPORT.md`  
  Phases, method, validation tables, confidence, and runtime.
- `/work/outputs/TOOLS.md`  
  Dependencies and tool usage.

Validation against the baseline using person-excluded silhouette IoU and textured-render NCC:

| Metric | Baseline | Result |
|---|---:|---:|
| Mean mask IoU | 0.589 | 0.691 |
| Mean texture NCC | 0.385 | 0.459 |
| Aggregate render score | 0.713 | 0.831 |
| Rotation jumps >60°/frame | 14 | 0 |
| Max rotation step | 163.3° | 49.9° |

The final solver run completed in approximately **1m18s** on the available GPU.
