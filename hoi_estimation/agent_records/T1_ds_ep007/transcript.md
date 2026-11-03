# T1_ds_ep007

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T16:22:52Z

**shell** (exit 0)
```
/usr/bin/bash -lc 'cat /work/AGENT_TASK.md'
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
/usr/bin/bash -lc "sed -n '1,220p' /work/AGENT_TASK.md"
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
/usr/bin/bash -lc "nl -ba /work/AGENT_TASK.md | sed -n '55,95p'"
```
<details><summary>output</summary>

```
    55	Iterate: diagnose the baseline, form a hypothesis, test it on a few frames, measure, and repeat. Work in small,
    56	fast experiments first; you have a fast model, so many short iterations are better than one long attempt.
    57	
    58	## Depth
    59	The baseline's distance from the camera was calibrated against the whole human body and the floor; depth from the
    60	object alone (silhouette scale on an object a few dozen pixels wide, with an estimated K) is far less reliable. You
    61	may correct depth, but only when several independent cues agree (hand contact with the reconstructed human, static
    62	object on the released floor plane, MoGe-2 depth, metric mesh size vs. image size), and quantify their disagreement.
    63	Do not snap the object onto the floor plane: it …[+3002 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "find /work/inputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort && printf '\\nTASK\\n' && cat /work/inputs/task.txt && printf '\\nEPISODE\\n' && cat /work/inputs/episode.json"
```
<details><summary>output</summary>

```
camera.json 831 bytes
episode.json 58 bytes
frame0_reference_human.npz 192445 bytes
ground_plane.json 397 bytes
masks.h5 7506760 bytes
object_mesh.glb 9155320 bytes
reconstruction.npz 45151533 bytes
task.txt 169 bytes
video.mp4 24989307 bytes

TASK
Walk over to the iron on the round table and using your right hand in the handle slide it back and forth horizontally on the desk. Stand it up vertically when finished.

EPISODE
{
 "episode_index": 7,
 "object": "iron",
 "frames": 896
}
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1929
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2132
/pipeline/.stages/08_render.json 2525
/pipeline/depth/moge2/depth_report.json 1808
/pipeline/depth/moge2/aligned.h5 1097696992
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 95383
/pipeline/depth/moge2/raw.npy 3170893952
/pipeline/export/episode_000007/edex 561
/pipeline/export/episode_000007/wild_export.json 2789
/pipeline/foundationpose/episode_000007.pkl 255235
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 685249 …[+549 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'ls -la /work /work/outputs /work/scratch /work/tools 2>/dev/null || true'
```
<details><summary>output</summary>

```
/work:
total 40
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 20:16 .
drwxr-xr-x 1 hanzhou hanzhou   200 Oct 20 20:22 ..
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 20:16 .codex_home
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 20:16 .net
-rw-rw-r-- 1 hanzhou hanzhou 10592 Oct 20 20:16 AGENT_TASK.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 17:43 inputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 20:16 logs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 20:16 outputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 20:16 scratch

/work/outputs:
total 8
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 20 20:16 .
drwxrwxr-x 8 hanzhou hanzhou 4096 Oct 20 20:16 ..

/work/scratch:
total 8
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 20 20:16 .
drwxrwxr-x 8 hanzhou hanzhou 4096 Oct 20 20:16 ..

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader && python - <<'PY'
import sys, torch
print(sys.version)
print('torch',torch.__version__,'cuda',torch.cuda.is_available())
PY"
```
<details><summary>output</summary>

```
3.11.10 | packaged by conda-forge | (main, Oct 16 2024, 01:27:36) [GCC 13.3.0]
torch 2.5.1+cu124 cuda True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import json, numpy as np, os, h5py, trimesh
for f in ['camera.json','ground_plane.json']:
 print('\\n',f); print(open('/work/inputs/'+f).read())
r=np.load('/work/inputs/reconstruction.npz')
print('keys',r.files)
for k in r.files:
 a=r[k]; print(k,a.shape,a.dtype, 'finite',np.isfinite(a).all() if a.dtype.kind in 'fc' else '')
P=r['T_world_mesh']; print('P first',P[:5]); print('trans ranges',P[:,:3,3].min(0),P[:,:3,3].max(0))
# rotation increments
def aa(R): 
 c=np.clip((np.trace(R)-1)/2,-1,1); return np.degrees(np.arccos(c))
angs=np.array([aa(P[i,:3,:3].T@P[i+1,:3,:3]) for i in range(len(P)-1)])
d=np.linalg.norm(P[1:,:3,3]-P[:-1,:3,3],axis=1)
print('rot step deg percentiles',np.percentile(angs,[0,10,25,50,75,90,95,99,100]))
print('top rot steps',np.argsort(angs)[-30:][::-1],angs[np.argsort(angs)[-30:][::-1]])
print('trans step',np.percentile(d,[0,10,25,50,75,90,95,99,100]))
print('contact percentiles',np.percentile(r['contact_prob_right_left'],[0,10,50,90,100],axis=0))
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
geom=mesh.to_geometry()
print('mesh',type(geom),len(geom.vertices),len(geom.faces),geom.bounds,geom.extents,geom.volume)
print('visual',type(geom.visual), hasattr(geom.visual,'material'), hasattr(geom.visual,'uv'))
PY"
```
<details><summary>output</summary>

```

 camera.json
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
   826.50732421875,
   0.0,
   768.0
  ],
  [
   0.0,
   826.5073852539062,
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
   0.9939386461001821,
   -0.00015970281720214667,
   0.10993608271876518,
   -0.14445048252465045
  ],
  [
   -0.0005195967274573216,
   0.9999809513329587,
   0.0061503650681558105,
   0.070514698820334
  ],
  [
   -0.10993497081355785,
   -0.006170207957674791,
   0.9939196299127918,
   -0.047044461572868075
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
 "note": "pinhole, OpenCV camera (x right, y down, z forward); world is …[+3358 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import json
for f in ['/pipeline/pipeline_report.json','/pipeline/.stages/08_render.json','/pipeline/.stages/07_refinement.json','/pipeline/.stages/06_coconet.json']:
 print('\\n###',f); print(open(f).read())
PY"
```
<details><summary>output</summary>

```

### /pipeline/pipeline_report.json
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
    "mtime_ns": 1791287980842484705,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000007/inference/episode_000007_step200000_before_after.mp4",
    "size": 60103469
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_d …[+12646 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=nw=1 /work/inputs/video.mp4'
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=29.866667
nb_frames=896

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py
for f in ['/work/inputs/masks.h5','/pipeline/work/masks_k0.h5']:
 print(f)
 with h5py.File(f) as h:
  def show(name,obj): print(name, getattr(obj,'shape',None), getattr(obj,'dtype',None))
  h.visititems(show)
PY"
```
<details><summary>output</summary>

```
/work/inputs/masks.h5
object (896, 1152, 1536) bool
person (896, 1152, 1536) bool
/pipeline/work/masks_k0.h5
episode_000007 None None
episode_000007/000000-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000007/000000-k0.person_mask.png (1152, 1536) uint8
episode_000007/000001-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000007/000001-k0.person_mask.png (1152, 1536) uint8
episode_000007/000002-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000007/000002-k0.person_mask.png (1152, 1536) uint8
episode_000007/000003-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000007/000003-k0.person_mask.png (1152, 1536) uint8
episode_000007/000004-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000007/000004-k0.person_mask.png (1152, 1536) uint8
episode_000007/000005-k0.obj_rend_mask.png (1152, 1536)  …[+108646 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/make_sheets.py <<'PY'
import cv2, numpy as np, h5py, os
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[]
while True:
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
print('loaded',len(frames))
os.makedirs('/work/scratch/sheets',exist_ok=True)
with h5py.File('/work/inputs/masks.h5','r') as h:
 om=h['object'][:]; pm=h['person'][:]
# overview every 8, 16x7 thumbnails
for s,e in [(0,224),(224,448),(448,672),(672,896)]:
 ids=list(range(s,e,8))
 tw,th=192,144; cols=14; rows=(len(ids)+cols-1)//cols
 out=np.ones((rows*th,cols*tw,3),np.uint8)*30
 for j,i in enumerate(ids):
  im=cv2.resize(frames[i],(tw,th),interpolation=cv2.INTER_AREA)
  cv2.putText(im,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
  out[(j//cols)*th:(j//cols+1)*th,(j%cols)*tw:(j%cols+1)*tw]=im
 cv2.imwrite(f'/work/scratch/sheets/overview_{s}_{e}.jpg',out)
# detail each 112 frame, step4, 7x4? 28 images 256x192
for s in range(0,896,112):
 ids=list(range(s,min(s+112,896),4)); tw,th=256,192; cols=7; rows=(len(ids)+cols-1)//cols
 out=np.ones((rows*th,cols*tw,3),np.uint8)*30
 for j,i in enumerate(ids):
  # crop around object mask + 50 margin, but also show wide person relation
  ys,xs=np.where(om[i])
  if len(xs):
   cx,cy=int(xs.mean()),int(ys.mean()); w=max(280,int((xs.max()-xs.min())*3)); h=max(220,int((ys.max()-ys.min())*3))
   x0=max(0,min(1536-w,cx-w//2)); y0=max(0,min(1152-h,cy-h//2)); crop=frames[i][y0:y0+h,x0:x0+w]
  else: crop=frames[i]
  im=cv2.resize …[+1254 chars]
```
<details><summary>output</summary>

```
loaded 896
total 5.8M
-rw-rw-r-- 1 hanzhou hanzhou 453K Oct 20 20:17 detail_0_112.jpg
-rw-rw-r-- 1 hanzhou hanzhou 451K Oct 20 20:17 detail_112_224.jpg
-rw-rw-r-- 1 hanzhou hanzhou 449K Oct 20 20:17 detail_224_336.jpg
-rw-rw-r-- 1 hanzhou hanzhou 459K Oct 20 20:17 detail_336_448.jpg
-rw-rw-r-- 1 hanzhou hanzhou 461K Oct 20 20:17 detail_448_560.jpg
-rw-rw-r-- 1 hanzhou hanzhou 457K Oct 20 20:17 detail_560_672.jpg
-rw-rw-r-- 1 hanzhou hanzhou 453K Oct 20 20:17 detail_672_784.jpg
-rw-rw-r-- 1 hanzhou hanzhou 449K Oct 20 20:17 detail_784_896.jpg
-rw-rw-r-- 1 hanzhou hanzhou 211K Oct 20 20:17 masks_0_224.jpg
-rw-rw-r-- 1 hanzhou hanzhou 212K Oct 20 20:17 masks_224_448.jpg
-rw-rw-r-- 1 hanzhou hanzhou 218K Oct 20 20:17 masks_448_672.jpg
-rw-rw-r-- 1 hanzhou hanzhou 212K Oct 20 20:17 masks_672_89 …[+276 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/analyze.py <<'PY'
import numpy as np, h5py, json, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=np.load('/work/inputs/reconstruction.npz'); P=r['T_world_mesh']; F=len(P)
with h5py.File('/work/inputs/masks.h5','r') as h:
 om=h['object'][:]; pm=h['person'][:]
areas=om.reshape(F,-1).sum(1); parea=pm.reshape(F,-1).sum(1)
# bbox
bbox=[]
for i in range(F):
 y,x=np.where(om[i]);
 bbox.append((x.min(),y.min(),x.max(),y.max()) if len(x) else (0,0,0,0))
bbox=np.array(bbox)
t=P[:,:3,3]; R=P[:,:3,:3]
def rv(R):
 # stable log
 v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
 s=np.linalg.norm(v); c=np.clip((np.trace(R)-1)/2,-1,1); th=np.arccos(c)
 return v*(th/np.sin(th)) if s>1e-8 and th>1e-6 else v
rv0=np.array([rv(R[0].T@x) for x in R]); rvn=np.array([rv(R[i].T@R[i+1]) for i in range(F-1)])
smooth=np.zeros(F); smooth[1:-1]=np.linalg.norm(rvn[:-1]+rvn[1:],axis=1)/2
# mesh vertices approx projected
print('areas',np.percentile(areas,[0,1,10,50,90,99,100]),'person',np.percentile(parea,[0,10,50,90,100]))
print('bbox starts',bbox[:20])
# core samples every 10
for i in list(range(0,F,10))+[F-1]:
 print(f'{i} t={t[i]} rv0={np.degrees(rv0[i]).round(2)} step={np.degrees(np.linalg.norm(rvn[i-1])).round(2) if i else 0} area={areas[i]} contact={r[\"contact_prob_right_left\"][i]}')
# plots
fig,axs=plt.subplots(5,1,figsize=(20,14),sharex=True)
axs[0].plot(t[:,0],label='x'); axs[0].plot(t[:,1],label='y'); axs[0].plot(t[:, …[+771 chars]
```
<details><summary>output</summary>

```
areas [2494.   2666.85 3024.   4417.   5214.   5298.1  5394.  ] person [32816.  34748.5 38974.  41616.  62306. ]
bbox starts [[679 478 766 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]
 [679 477 767 628]]
0 t=[0.24529311 0.01509904 3.1904907 ] rv0=[0. 0. 0.] step=0 area=5394 contact=[2.0732480e-07 7.2252665e-06]
10 t=[0.24401931 0.01528015 3.1799095 ] rv0=[0. 0. 0.] step=0.0 area=5297 contact=[8.039611e-08 1.572714e-06]
20 t=[0.24243487 0.01550544 3.1667478 ] rv0=[0. 0. 0.] step=0.0 area=5262 c …[+11475 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py
f='/pipeline/depth/moge2/aligned.h5'
with h5py.File(f) as h:
 def show(n,o): print(n,getattr(o,'shape',None),getattr(o,'dtype',None))
 h.visititems(show)
PY"
```
<details><summary>output</summary>

```
aligned None None
aligned/front_stereo_camera_left (896,) object
alignment None None
alignment/front_stereo_camera_left None None
alignment/front_stereo_camera_left/scale (896,) float32
alignment/front_stereo_camera_left/shift (896,) float32
alignment/front_stereo_camera_left/valid_count (896,) int32
frame_names None None
frame_names/front_stereo_camera_left (896,) object
raw None None
raw/front_stereo_camera_left (896,) object

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');P=r['T_world_mesh'];
# identify runs of near-identical rotation & translation
for a,b in [(0,110),(110,170),(170,260),(260,340),(340,450),(450,560),(560,670),(670,780),(780,896)]:
 print('\\n',a,b)
 for i in range(a,b,5):
  print(i, P[i,:3,3].round(3), np.degrees(np.arccos(np.clip((np.trace(P[i,:3,:3])-1)/2,-1,1))).round(1))
PY"
```
<details><summary>output</summary>

```

 0 110
0 [0.245 0.015 3.19 ] 174.6
5 [0.245 0.015 3.184] 174.6
10 [0.244 0.015 3.18 ] 174.6
15 [0.244 0.015 3.176] 174.6
20 [0.242 0.016 3.167] 174.6
25 [0.242 0.016 3.162] 174.6
30 [0.242 0.016 3.162] 174.6
35 [0.241 0.016 3.157] 174.6
40 [0.241 0.016 3.155] 174.6
45 [0.241 0.016 3.159] 174.6
50 [0.242 0.016 3.162] 174.6
55 [0.243 0.015 3.168] 174.6
60 [0.244 0.015 3.178] 174.6
65 [0.245 0.015 3.186] 174.6
70 [0.248 0.015 3.216] 174.6
75 [0.25  0.014 3.228] 174.6
80 [0.251 0.014 3.235] 174.6
85 [0.251 0.014 3.24 ] 174.6
90 [0.253 0.014 3.254] 174.6
95 [0.307 0.024 3.346] 179.9
100 [ 0.172 -0.054  3.17 ] 177.6
105 [ 0.206 -0.079  3.184] 177.6

 110 170
110 [ 0.187 -0.019  3.214] 169.9
115 [ 0.205 -0.072  3.222] 170.4
120 [ 0.205 -0.072  3.226] 170.4
125 [ 0.206 -0.072  3.23 ] 170.4
130 [  …[+5137 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
mods=['nvdiffrast','pyrender','pytorch3d','open3d','cv2','trimesh','torch3d','kornia']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__','?'))
 except Exception as e: print(m,'NO',e)
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
pytorch3d OK 0.7.9
open3d OK 0.20.0
cv2 OK 4.11.0
trimesh OK 5.1.1
torch3d NO No module named 'torch3d'
kornia OK 0.8.3

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m.visual.uv.shape,m.visual.uv.min(0),m.visual.uv.max(0),m.visual.material.image.size,m.visual.material.image.mode)
print(m.faces.shape,m.vertices.shape)
print(type(m.visual.material),m.visual.material.name)
print(m.visual.uv[:5],m.faces[:5])
# texture image stats
im=np.asarray(m.visual.material.image); print(im.shape,im.min(),im.max(),im.mean((0,1)))
# ground distances vertices from plane
import json
p=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(p[:3]); d=p[3]
dist=m.vertices@n+d
print('mesh local plane dist',np.percentile(dist,[0,1,5,25,50,75,95,99,100]))
# principal axes
for i in range(3): print(i,np.linalg.eigh(m.vertices.T@m.vertices)[1][:,i])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
mat=m.visual.material
print(type(mat)); print(mat.__dict__.keys()); print(mat)
for k,v in mat.__dict__.items(): print(k,type(v),getattr(v,'shape',None),getattr(v,'size',None))
print('uv',m.visual.uv.shape,m.visual.uv.min(0),m.visual.uv.max(0))
print('image',mat.baseColorTexture)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.material.PBRMaterial'>
dict_keys(['_data'])
<trimesh.visual.material.PBRMaterial object at 0x7f06c00df590>
_data <class 'dict'> None None
uv (298533, 2) [0.00206    0.00184399] [0.99794    0.99815601]
image <PIL.Image.Image image mode=RGB size=1699x1898 at 0x7F077CB1F250>

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/test_render.py <<'PY'
import numpy as np, torch, cv2, trimesh, nvdiffrast.torch as dr, json
from PIL import Image
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda')[None]
F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
uv=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')[None]
tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
r=np.load('/work/inputs/reconstruction.npz'); P=r['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tc2w=np.array(cam['T_world_cam']); w2c=np.linalg.inv(Tc2w)
i=0
Tm=torch.tensor(P[i],dtype=torch.float32,device='cuda')
T=torch.linalg.inv(torch.tensor(w2c,dtype=torch.float32,device='cuda'))@Tm
verts=V@T[:3,:3].T+T[:3,3]
# full image render
gl=dr.RasterizeCudaContext()
rast,interp=dr.rasterize(gl,verts,torch.tensor(K,dtype=torch.float32,device='cuda')[None],resolution=(1152,1536))
print('rast',rast.shape,rast.max(),(rast[...,3]>0).sum())
color=dr.interpolate(tex,interp,F)[0]
mask=(rast[...,3]>0)
out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
# background black, mask
out[~mask.cpu()]=0
cv2.imwrite('/work/scratch/render0.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
PY
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 16, in <module>
    rast,interp=dr.rasterize(gl,verts,torch.tensor(K,dtype=torch.float32,device='cuda')[None],resolution=(1152,1536))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Fi …[+403 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='''import numpy as np, torch, cv2, trimesh, nvdiffrast.torch as dr, json
from PIL import Image
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda')[None]
F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
uv=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')[None]
tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
r=np.load('/work/inputs/reconstruction.npz'); P=r['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tc2w=np.array(cam['T_world_cam']); w2c=np.linalg.inv(Tc2w)
i=0
Tm=torch.tensor(P[i],dtype=torch.float32,device='cuda')
T=torch.linalg.inv(torch.tensor(w2c,dtype=torch.float32,device='cuda'))@Tm
verts=V@T[:3,:3].T+T[:3,3]
xy=verts[...,:2]/verts[...,2:3]
uvpix=xy*torch.tensor([K[0,0],K[1,1]],device='cuda')+torch.tensor([K[0,2],K[1,2]],device='cuda')
z=verts[...,2:3]
pos=torch.cat([uvpix*z,z,z],dim=-1)
gl=dr.RasterizeCudaContext()
rast,interp=dr.rasterize(gl,pos,F,resolution=(1152,1536))
print('rast',rast.shape,rast.max(),(rast[...,3]>0).sum())
color=dr.interpolate(tex,interp,F)[0]
mask=(rast[...,3]>0)
out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8)
out[~mask.cpu()]=0
cv2.imwrite('/work/scratch/render0.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
'''
open('/work/scratch/test_render.py','w').write(p)
PY
python / …[+28 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 19, in <module>
    rast,interp=dr.rasterize(gl,pos,F,resolution=(1152,1536))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 78, in forward
    out, out_db = _nvd …[+285 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/dtype=torch.float32,device='cuda')+torch.tensor(\\[K\\[0,0\\],K\\[1,1\\]\\],device='cuda')/dtype=torch.float32,device='cuda')+torch.tensor([K[0,0],K[1,1]],dtype=torch.float32,device='cuda')/; s/torch.tensor(\\[K\\[0,2\\],K\\[1,2\\]\\],device='cuda')/torch.tensor([K[0,2],K[1,2]],dtype=torch.float32,device='cuda')/\" /work/scratch/test_render.py
# inspect relevant
grep uvpix -A2 /work/scratch/test_render.py
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 19, in <module>
    rast,interp=dr.rasterize(gl,pos,F,resolution=(1152,1536))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 78, in forward
    out, out_db = _nvd …[+285 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/uvpix=xy\\*torch.tensor(\\[K\\[0,0\\],K\\[1,1\\]\\],device='cuda')/uvpix=xy*torch.tensor([K[0,0],K[1,1]],dtype=torch.float32,device='cuda')/\" /work/scratch/test_render.py
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(0., device='cuda:0') tensor(0, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 21, in <module>
    color=dr.interpolate(tex,interp,F)[0]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _nvdiffrast_c.i …[+157 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
exec(open('/work/scratch/test_render.py').read().split('xy=')[0])
print('cam z',verts[...,2].min(),verts[...,2].max())
xy=verts[...,:2]/verts[...,2:3]; print('xy',xy.min((0,1)),xy.max((0,1)))
uvpix=xy*torch.tensor([K[0,0],K[1,1]],dtype=torch.float32,device='cuda')+torch.tensor([K[0,2],K[1,2]],dtype=torch.float32,device='cuda')
print('pix',uvpix.min((0,1)),uvpix.max((0,1)))
PY"
```
<details><summary>output</summary>

```
cam z tensor(3.0053, device='cuda:0') tensor(3.2048, device='cuda:0')
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
TypeError: min() received an invalid combination of arguments - got (tuple), but expected one of:
 * ()
      didn't match because some of the arguments have invalid types: (!tuple of (int, int)!)
 * (Tensor other)
      didn't match because some of the arguments have invalid types: (!tuple of (int, int)!)
 * (int dim, bool keepdim = False)
 * (name dim, bool keepdim = False)


```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/test_render.py'; s=open(p).read()
s=s.replace(\"pos=torch.cat([uvpix*z,z,z],dim=-1)\",\"ndc=torch.stack([2*uvpix[...,0]/1536-1,1-2*uvpix[...,1]/1152],dim=-1)\\npos=torch.cat([ndc*z,z,z],dim=-1)\")
open(p,'w').write(s)
PY
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99387., device='cuda:0') tensor(2060, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 22, in <module>
    color=dr.interpolate(tex,interp,F)[0]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _nvdiffr …[+164 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.getsource(dr.interpolate)[:2500])
PY"
```
<details><summary>output</summary>

```
def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Interpolate vertex attributes.

    All input tensors must be contiguous and reside in GPU memory. The output tensors
    will be contiguous and reside in GPU memory.

    Args:
        attr: Attribute tensor with dtype `torch.float32`. 
              Shape is [num_vertices, num_attributes] in range mode, or 
              [minibatch_size, num_vertices, num_attributes] in instanced mode.
              Broadcasting is supported along the minibatch axis.
        rast: Main output tensor from `rasterize()`.
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        rast_db: (Optional) Tensor containing image-space derivatives of barycentrics, 
                 i.e., the second output tenso …[+1701 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/test_render.py'; s=open(p).read().replace(\"uv=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')[None]\",\"uv=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')\")
s=s.replace(\"color=dr.interpolate(tex,interp,F)[0]\",\"icolor=dr.interpolate(uv,interp,F)[0]\\ncolor=dr.texture(tex,icolor[None],filter_mode='linear')[0]\")
open(p,'w').write(s)
PY
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99387., device='cuda:0') tensor(2060, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 23, in <module>
    color=dr.texture(tex,icolor[None],filter_mode='linear')[0]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 439, in texture
    return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Fil …[+305 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/dr.texture(tex,icolor\\[None\\],filter_mode='linear')/dr.texture(tex,icolor,filter_mode='linear')/\" /work/scratch/test_render.py
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99387., device='cuda:0') tensor(2060, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 26, in <module>
    out[~mask.cpu()]=0
    ~~~^^^^^^^^^^^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1152 but corresponding boolean dimension is 1

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/out=(color.clamp(0,1).cpu().numpy()\\*255).astype(np.uint8)/out=(color[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)/; s/out\\[~mask.cpu()\\]=0/out[~mask[0].cpu()]=0/\" /work/scratch/test_render.py
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99387., device='cuda:0') tensor(2060, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 26, in <module>
    out[~mask[0].cpu()]=0
    ~~~^^^^^^^^^^^^^^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1536 but corresponding boolean dimension is 1152

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/color=dr.texture(tex,icolor,filter_mode='linear')\\[0\\]/color=dr.texture(tex,icolor,filter_mode='linear')\\nprint('color',color.shape,'mask',mask.shape)/\" /work/scratch/test_render.py
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99387., device='cuda:0') tensor(2060, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 24, in <module>
    print('color',color.shape,'mask',mask.shape)
                                     ^^^^
NameError: name 'mask' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/test_render.py'; s=open(p).read(); s=s.replace(\"print('color',color.shape,'mask',mask.shape)\",\"print('color',color.shape)\"); s=s.replace(\"out=(color[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)\\nout[~mask[0].cpu()]=0\",\"out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8)[0]\\nout[~mask[0].cpu()]=0\"); open(p,'w').write(s)
PY
python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99387., device='cuda:0') tensor(2060, device='cuda:0')
color torch.Size([1, 1152, 1536, 3])

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
im=cv2.imread('/work/scratch/render0.png'); frame=cv2.imread('/work/inputs/video.mp4',0) if False else None
cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,fr=cap.read()
# bbox render
ys,xs=np.where(im.any(2)); print(xs.min(),xs.max(),ys.min(),ys.max())
cv2.imwrite('/work/scratch/render0_crop.png',im[max(0,ys.min()-40):ys.max()+40,max(0,xs.min()-40):xs.max()+40])
# overlay
ov=fr.copy(); alpha=.6; m=im.any(2); ov[m]=(im[m]*.8+ov[m]*.2).astype(np.uint8)
cv2.imwrite('/work/scratch/overlay0_crop.png',ov[max(0,ys.min()-40):ys.max()+40,max(0,xs.min()-40):xs.max()+40])
PY"
```
<details><summary>output</summary>

```
855 892 527 590

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
r=np.load('/work/inputs/reconstruction.npz'); P=r['T_world_mesh']; c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); w2c=np.linalg.inv(np.array(c['T_world_cam']))
for i in [0,100,200,400,700,800]:
 V=m.vertices; W=V@P[i,:3,:3].T+P[i,:3,3]; C=W@w2c[:3,:3].T+w2c[:3,3]; uv=C[:,:2]/C[:,2:3]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
 print(i,uv.min(0),uv.max(0),C[:,2].min(),C[:,2].max())
PY"
```
<details><summary>output</summary>

```
0 [744.30403668 507.05209749] [782.01292174 581.88648196] 3.178129229752935 3.37110117782343
100 [726.00863279 489.5285488 ] [762.38297168 561.2530488 ] 3.1335375430611827 3.3492039529514046
200 [700.50272527 476.3273863 ] [769.61989941 510.37467295] 3.5528430712553107 3.680331560799095
400 [727.24094293 517.55633165] [762.70491149 556.47716267] 3.230378964032216 3.53660095023765
700 [753.25355068 479.71796286] [787.64167335 537.37223162] 3.1589783457770686 3.4273506109736007
800 [721.06826525 500.56078336] [753.77980294 559.16824841] 3.4221742887579483 3.6752554234406714

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat /work/scratch/test_render.py && python - <<'PY'
import numpy as np, torch, trimesh, json, nvdiffrast.torch as dr
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda')[None]
r=np.load('/work/inputs/reconstruction.npz'); P=r['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']); w2c=np.linalg.inv(np.array(cam['T_world_cam']))
T=torch.linalg.inv(torch.tensor(w2c,dtype=torch.float32,device='cuda'))@torch.tensor(P[0],dtype=torch.float32,device='cuda')
verts=V@T[:3,:3].T+T[:3,3]
xy=verts[...,:2]/verts[...,2:3]; uvpix=xy*torch.tensor([K[0,0],K[1,1]],dtype=torch.float32,device='cuda')+torch.tensor([K[0,2],K[1,2]],dtype=torch.float32,device='cuda')
print(uvpix[0].min(0).values,uvpix[0].max(0).values)
PY"
```
<details><summary>output</summary>

```
tensor([855.1059, 551.8503], device='cuda:0') tensor([893.4125, 631.9638], device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/T=torch.linalg.inv(torch.tensor(w2c,dtype=torch.float32,device='cuda'))@Tm/T=torch.tensor(w2c,dtype=torch.float32,device='cuda')@Tm/\" /work/scratch/test_render.py
python /work/scratch/test_render.py
python - <<'PY'
import cv2,numpy as np
im=cv2.imread('/work/scratch/render0.png'); cap=cv2.VideoCapture('/work/inputs/video.mp4'); ok,fr=cap.read()
ys,xs=np.where(im.any(2)); print(xs.min(),xs.max(),ys.min(),ys.max())
x0=max(0,xs.min()-80);x1=min(1536,xs.max()+80);y0=max(0,ys.min()-80);y1=min(1152,ys.max()+80)
ov=fr.copy(); m=im.any(2); ov[m]=(im[m]*.8+ov[m]*.2).astype(np.uint8)
cv2.imwrite('/work/scratch/render0_crop.png',im[y0:y1,x0:x1]); cv2.imwrite('/work/scratch/overlay0_crop.png',ov[y0:y1,x0:x1])
PY"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99496., device='cuda:0') tensor(1888, device='cuda:0')
color torch.Size([1, 1152, 1536, 3])
747 780 579 643

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/1-2\\*uvpix\\[...,1\\]\\/1152/2*uvpix[...,1]\\/1152-1/\" /work/scratch/test_render.py
python /work/scratch/test_render.py
python - <<'PY'
import cv2,numpy as np
im=cv2.imread('/work/scratch/render0.png'); cap=cv2.VideoCapture('/work/inputs/video.mp4'); ok,fr=cap.read()
ys,xs=np.where(im.any(2)); print(xs.min(),xs.max(),ys.min(),ys.max())
x0=max(0,xs.min()-80);x1=min(1536,xs.max()+80);y0=max(0,ys.min()-80);y1=min(1152,ys.max()+80)
ov=fr.copy(); m=im.any(2); ov[m]=(im[m]*.8+ov[m]*.2).astype(np.uint8)
cv2.imwrite('/work/scratch/render0_crop.png',im[y0:y1,x0:x1]); cv2.imwrite('/work/scratch/overlay0_crop.png',ov[y0:y1,x0:x1])
PY"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99496., device='cuda:0') tensor(1888, device='cuda:0')
color torch.Size([1, 1152, 1536, 3])
746 781 510 576

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
im=cv2.imread('/work/scratch/render0.png'); m=(im.max(2)>0).astype(np.uint8)*255
cv2.imwrite('/work/scratch/rendermask.png',m)
# count rows/cols occupancy
print('nonzero',m.sum()//255, 'rowmax',[(i,int((m[i]>0).sum())) for i in range(1152) if (m[i]>0).sum()>20][:20])
PY"
```
<details><summary>output</summary>

```
nonzero 32.0 rowmax []

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
# modify to save mask
s=open('/work/scratch/test_render.py').read(); s=s.replace(\"cv2.imwrite('/work/scratch/render0.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))\",\"cv2.imwrite('/work/scratch/render0.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR))\\ncv2.imwrite('/work/scratch/rendermask.png',mask[0].cpu().numpy().astype(np.uint8)*255)\")
open('/work/scratch/test_render2.py','w').write(s)
PY
python /work/scratch/test_render2.py
python - <<'PY'
import cv2,numpy as np
m=cv2.imread('/work/scratch/rendermask.png',0); print((m>0).sum(),m.shape); ys,xs=np.where(m); print(xs.min(),xs.max(),ys.min(),ys.max())
cv2.imwrite('/work/scratch/rendermask_crop.png',m[400:700,600:900])
PY"
```
<details><summary>output</summary>

```
rast torch.Size([1, 1152, 1536, 4]) tensor(99496., device='cuda:0') tensor(1888, device='cuda:0')
color torch.Size([1, 1152, 1536, 3])
1888 (1152, 1536)
744 781 507 581

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,numpy as np
f='/pipeline/depth/moge2/aligned.h5'
with h5py.File(f) as h:
 g=h['aligned/front_stereo_camera_left']
 print('aligned group',g.keys())
 for k in g: print(k,g[k].shape,g[k].dtype, g[k][:3])
 print('raw group',h['raw/front_stereo_camera_left'].keys())
 for k in h['raw/front_stereo_camera_left']: print(k,h['raw/front_stereo_camera_left'][k].shape,h['raw/front_stereo_camera_left'][k].dtype,h['raw/front_stereo_camera_left'][k][:3])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
AttributeError: 'Dataset' object has no attribute 'keys'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,numpy as np,pickle
f='/pipeline/depth/moge2/aligned.h5'
with h5py.File(f) as h:
 for path in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left','frame_names/front_stereo_camera_left','alignment/front_stereo_camera_left/scale','alignment/front_stereo_camera_left/shift']:
  d=h[path]; print(path,d.shape,d.dtype)
  for i in [0,1,100]:
   v=d[i]; print(i,type(v),getattr(v,'shape',None),getattr(v,'dtype',None), repr(v)[:500])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (896,) object
0 <class 'numpy.ndarray'> (658091,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
1 <class 'numpy.ndarray'> (658686,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
100 <class 'numpy.ndarray'> (641185,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
raw/front_stereo_camera_left (896,) object
0 <class 'numpy.ndarray'> (596501,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
1 <class 'numpy.ndarray'> (595875,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
100 <class 'numpy.ndarray'> (587433,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
frame_names/front_stereo_camera_left (896,) object
0 <class 'bytes'> None None b'000000'
1 <class 'bytes'> None None b …[+428 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/depth_inspect.py <<'PY'
import h5py,numpy as np,cv2,json
f='/pipeline/depth/moge2/aligned.h5'
with h5py.File(f) as h:
 for i in [0,100,200,400,700,800]:
  raw=cv2.imdecode(h['raw/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)
  ali=cv2.imdecode(h['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)
  print(i,raw.shape,raw.dtype,np.percentile(raw,[0,1,50,99,100]),ali.shape,ali.dtype,np.percentile(ali,[0,1,50,99,100]))
  cv2.imwrite(f'/work/scratch/depth_{i}.png',ali)
PY
python /work/scratch/depth_inspect.py"
```
<details><summary>output</summary>

```
0 (1152, 1536) uint16 [1592. 1671. 3592. 8116. 8601.] (1152, 1536) uint16 [1812. 1902. 4089. 9239. 9791.]
100 (1152, 1536) uint16 [1576. 1654. 3597. 8064. 8590.] (1152, 1536) uint16 [1753.   1840.   4003.   8974.29 9559.  ]
200 (1152, 1536) uint16 [1564. 1640. 3558. 8022. 8733.] (1152, 1536) uint16 [ 1794.  1881.  4082.  9204. 10020.]
400 (1152, 1536) uint16 [1576.   1657.   3613.   8047.29 8590.  ] (1152, 1536) uint16 [1757.   1848.   4029.   8975.29 9580.  ]
700 (1152, 1536) uint16 [1579. 1657. 3601. 8128. 8776.] (1152, 1536) uint16 [1768. 1855. 4032. 9102. 9828.]
800 (1152, 1536) uint16 [1591. 1670. 3638. 8194. 8777.] (1152, 1536) uint16 [1768. 1856. 4044. 9109. 9758.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/pipeline/depth/moge2/aligned.h5') as h, h5py.File('/work/inputs/masks.h5') as m:
 for i in [0,50,100,150,200,300,400,500,600,700,800,895]:
  d=cv2.imdecode(h['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
  om=m['object'][i]; pm=m['person'][i]
  print(i,'obj med/range',np.percentile(d[om],[5,25,50,75,95]),'person',np.percentile(d[pm],[5,25,50,75,95]),'scale',h['alignment/front_stereo_camera_left/scale'][i])
PY"
```
<details><summary>output</summary>

```
0 obj med/range [7.42  7.856 8.289 8.606 9.143] person [2.4   2.432 2.488 2.583 2.702] scale 1.1384143
50 obj med/range [7.37555 7.861   8.263   8.52125 8.982  ] person [2.618 2.644 2.688 2.773 3.049] scale 1.1329101
100 obj med/range [6.8974 7.392  7.966  8.361  8.8672] person [2.83  2.95  3.031 3.086 3.264] scale 1.112899
150 obj med/range [7.0026 7.6    8.054  8.353  8.8048] person [2.87  2.989 3.056 3.168 3.335] scale 1.115617
200 obj med/range [7.1098 8.019  8.349  8.627  8.9546] person [2.976 3.071 3.135 3.242 3.401] scale 1.1474367
300 obj med/range [7.543 7.937 8.26  8.419 8.793] person [2.92  3.038 3.103 3.213 3.398] scale 1.1208518
400 obj med/range [7.505   7.93725 8.283   8.476   8.866  ] person [2.857 2.997 3.087 3.2   3.34 ] scale 1.1153269
500 obj med/range [5.947   6.232    …[+535 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); ok,fr=cap.read()
with h5py.File('/work/inputs/masks.h5') as h: om=h['object'][0];pm=h['person'][0]
ov=fr.copy();ov[om]=(ov[om]*.4+np.array([0,0,255])*.6).astype(np.uint8);ov[pm]=(ov[pm]*.6+np.array([0,255,0])*.4).astype(np.uint8)
ys,xs=np.where(om);x0=max(0,xs.min()-60);x1=xs.max()+60;y0=max(0,ys.min()-60);y1=ys.max()+60
cv2.imwrite('/work/scratch/frame0_objmask.png',ov[y0:y1,x0:x1]);cv2.imwrite('/work/scratch/frame0_obj.png',fr[y0:y1,x0:x1]);print(x0,x1,y0,y1)
PY"
```
<details><summary>output</summary>

```
619 826 418 688

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'find /code/reconstruction -maxdepth 4 -type f | head -100 && grep -R "sam2" -n /code/reconstruction | head -80'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "find /workspace -maxdepth 3 -type f -iname '*sam2*' -o -iname '*mask*' | head -100; find /weights/sam2 -maxdepth 3 -type f -printf '%p %s\\n' | head -100"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/pack_masks.py
/workspace/v2d_mesh/lib/mesh_render_mask.py
/workspace/v2d_mesh/lib/run_mesh_render_mask.py
/workspace/v2d_sam2/docker/run_mv_videos_to_masks.py
/workspace/v2d_sam2/docker/run_video_to_masks.py
/workspace/v2d_sam2/lib/image_to_mask.py
/workspace/v2d_sam2/lib/mv_videos_to_masks.py
/workspace/v2d_sam2/lib/mv_videos_to_masks.yaml
/workspace/v2d_sam2/lib/sam2_utils.py
/workspace/v2d_sam2/lib/video_to_masks.py
/weights/sam2/.cache/huggingface/CACHEDIR.TAG 191
/weights/sam2/.cache/huggingface/.gitignore 1
/weights/sam2/sam2.1_hiera_l.yaml 3798
/weights/sam2/.gitattributes 1519
/weights/sam2/README.md 19979
/weights/sam2/config.json 5705
/weights/sam2/processor_config.json 95
/weights/sam2/preprocessor_config.json 683
/weights/sam2/video_preprocessor_config …[+98 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'cat /pipeline/work/prompts.json; cat /pipeline/export/episode_000007/wild_export.json'
```
<details><summary>output</summary>

```
{
  "prompts": [
    {
      "frame_index": 0,
      "object_id": 0,
      "points": null,
      "point_labels": null,
      "box": {
        "x0": 428.0830993652344,
        "y0": 399.2731018066406,
        "x1": 615.5076904296875,
        "y1": 967.3934326171875
      },
      "mask_path": null
    },
    {
      "frame_index": 0,
      "object_id": 1,
      "points": null,
      "point_labels": null,
      "box": {
        "x0": 674.9259033203125,
        "y0": 476.0604553222656,
        "x1": 768.900634765625,
        "y1": 631.4425048828125
      },
      "mask_path": null
    }
  ]
}{
  "aligned_object_extents_m": [
    0.11979246884584427,
    0.13417744636535645,
    0.3051129877567291
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_p …[+2585 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '1,260p' /workspace/v2d_sam2/lib/video_to_masks.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SAM2 video to masks processing function.
Can be called directly from command line or imported as a function.
"""
from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem
from v2d.sam2.lib.datatypes import Sam2Prompts
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from v2d.common.video import FrameWriter

_predictor = None


def _resolve_prompt_mask_path(mask_path: str, prompts_path: str) -> Path:
    path = Path(mask_path)
    if not path.is_absolute():
        path = Path(prompts_path).resolve().parent / path
    return path


def _validate_prompt(prompt) -> None:
  …[+4517 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '1,240p' /workspace/v2d_sam2/docker/run_video_to_masks.py; sed -n '1,300p' /workspace/v2d_sam2/lib/sam2_utils.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import shutil
import tempfile

from v2d.docker.container import run_in_container
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR


def _rewrite_prompts_for_container(
    prompts_path: str,
) -> tuple[str, list[str], str | None]:
    """Rewrite host mask prompt paths to container-visible paths."""
    with open(prompts_path) as f:
        data = json.load(f)
    prompts = data.get("prompts") or []

    mask_dirs: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        mask_path = prompt.get("mask_path")
        if not mask_path:
            continue
        if not os.path.isabs(mask_path):
            …[+13568 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import v2d_sam2, inspect
print(v2d_sam2.__file__)
PY
find /weights/sam2 -maxdepth 2 -type f -name '*.yaml' -o -name '*.pt' -printf '%p\\n'"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'v2d_sam2'
/weights/sam2/sam2.1_hiera_large.pt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
for m in ['v2d.sam2.lib.video_to_masks','sam2']:
 try:
  x=__import__(m,fromlist=['*']); print(m,'OK',x.__file__)
 except Exception as e: print(m,'NO',e)
PY
echo PYTHONPATH="'$PYTHONPATH
find /workspace/v2d_sam2 -maxdepth 2 -type f -printf '"'%p\\n' | head"
```
<details><summary>output</summary>

```
v2d.sam2.lib.video_to_masks OK /workspace/v2d_sam2/lib/video_to_masks.py
sam2 OK /opt/conda/lib/python3.11/site-packages/sam2/__init__.py
PYTHONPATH=/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose
/workspace/v2d_sam2/.dockerignore
/workspace/v2d_sam2/.gitignore
/workspace/v2d_sam2/LICENSE
/workspace/v2d_sam2/__init__.py
/workspace/v2d_sam2/assets/test_prompts.json
/workspace/v2d_sam2/assets/test_video.mp4
/workspace/v2d_sam2/docker/Dockerfile
/workspace/v2d_sam2/docker/__init__.py
/workspace/v2d_sam2/docker/_config.py
/workspace/v2d_sam2/docker/build.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'grep -n "def build_sam2_video_predictor_low_mem" -A100 /workspace/v2d_sam2/lib/sam2_utils.py'
```
<details><summary>output</summary>

```
67:def build_sam2_video_predictor_low_mem(
68-    config_file,
69-    ckpt_path=None,
70-    device="cuda",
71-    mode="eval",
72-    hydra_overrides_extra=[],
73-    apply_postprocessing=True,
74-    **kwargs,
75-):
76-    hydra_overrides = [
77-        "++model._target_=sam2_utils.SAM2VideoPredictorLowMem",
78-    ]
79-
80-    if apply_postprocessing:
81-        hydra_overrides_extra = hydra_overrides_extra.copy()
82-        hydra_overrides_extra += [
83-            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
84-            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
85-            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
86-            "++model.binarize_mask_from_pts_for_mem_enc=true …[+3404 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/sam2_reseg.json <<'EOF'
{
 \"prompts\": [
  {\"frame_index\": 0, \"object_id\": 0, \"points\": null, \"point_labels\": null,
   \"box\": {\"x0\": 670.0, \"y0\": 550.0, \"x1\": 775.0, \"y1\": 645.0},
   \"mask_path\": null}
 ]
}
EOF
export HYDRA_CONFIG_SEARCH_PATH=/weights/sam2
export CONFIG_FILE=sam2.1_hiera_l.yaml
export CHECKPOINT_FILE=sam2.1_hiera_large.pt
python -m v2d.sam2.lib.video_to_masks --video_path /work/inputs/video.mp4 --prompts_path /work/scratch/sam2_reseg.json --masks_dir /work/scratch/sam2_iron --weights_dir /weights/sam2"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/workspace/v2d_sam2/lib/video_to_masks.py", line 134, in <module>
    video_to_masks(args.video_path, args.prompts_path, args.masks_dir, args.weights_dir)
  File "/workspace/v2d_sam2/lib/video_to_masks.py", line 80, in video_to_masks
    predictor = _get_predictor(weights_dir)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/v2d_sam2/lib/video_to_masks.py", line 52, in _get_predictor
    _predictor = build_sam2_video_predictor_low_mem(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/v2d_sam2/lib/sam2_utils.py", line 91, in build_sam2_video_predictor_low_mem
    cfg = compose(config_name=config_file, overrid …[+2583 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'export CONFIG_FILE=configs/sam2.1/sam2.1_hiera_l.yaml
export CHECKPOINT_FILE=sam2.1_hiera_large.pt
python -m v2d.sam2.lib.video_to_masks --video_path /work/inputs/video.mp4 --prompts_path /work/scratch/sam2_reseg.json --masks_dir /work/scratch/sam2_iron --weights_dir /weights/sam2'
```
<details><summary>output</summary>

```
/workspace/v2d_sam2/lib/sam2_utils.py:1058: UserWarning: cannot import name '_C' from 'sam2' (/opt/conda/lib/python3.11/site-packages/sam2/__init__.py)

Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
  pred_masks_gpu = fill_holes_in_mask_scores(
propagate in video: 0it [00:00, ?it/s]propagate in video: 0it [00:00, ?it/s]
propagate in video:   0%|          | 0/896 [00:00<?, ?it/s]propagate in video:   0%|          | 3/896 [00:00<00:33, 27.06it/s]propagate in video:   1%|          | 6/896 [00:00<00:35, 25.29it/s]propagate in video:   1%|    …[+20800 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "find /work/scratch/sam2_iron -type f | head; ls -lh /work/scratch/sam2_iron; python - <<'PY'
import h5py,numpy as np,glob,os
print(glob.glob('/work/scratch/sam2_iron/*'))
for f in glob.glob('/work/scratch/sam2_iron/*'):
 print(f)
 try:
  with h5py.File(f) as h: h.visititems(lambda n,o: print(n,getattr(o,'shape',None),getattr(o,'dtype',None)))
 except Exception as e: print(e)
PY"
```
<details><summary>output</summary>

```
['/work/scratch/sam2_iron/0']
/work/scratch/sam2_iron/0
[Errno 21] Unable to synchronously open file (file read failed: time = Tue Oct 20 20:28:53 2026
, filename = '/work/scratch/sam2_iron/0', file descriptor = 3, errno = 21, error message = 'Is a directory', buf = 0x7ffdc25c45f8, total read size = 8, bytes this sub-read = 8, offset = 0)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np,os
for i in [0,50,100,150,200,300,400,500,550,600,650,700,720,740,760,780,800,850,895]:
 p=f'/work/scratch/sam2_iron/0/{i:06d}.png'; m=cv2.imread(p,0)>0; ys,xs=np.where(m); print(i,m.sum(),(xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None)
# overlay sheet every 16 object red
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
for s,e in [(0,224),(224,448),(448,672),(672,896)]:
 ids=list(range(s,e,16));tw,th=256,192;cols=7;rows=4;out=np.ones((rows*th,cols*tw,3),np.uint8)*30
 for j,i in enumerate(ids):
  im=frames[i].copy(); m=cv2.imread(f'/work/scratch/sam2_iron/0/{i:06d}.png',0)>0
  im[m]=(im[m]*.4+np.array([0,0,255])*.6).astype(np.uint8)
  ys,xs=np.where(m)
  if len(xs):
   cx,cy=int(xs.mean()),int(ys.mean()); w=max(250,int((xs.max()-xs.min())*2.2));h=max(190,int((ys.max()-ys.min())*2.2));x0=max(0,min(1536-w,cx-w//2));y0=max(0,min(1152-h,cy-h//2));crop=im[y0:y0+h,x0:x0+w]
  else:crop=im
  im=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_AREA);cv2.putText(im,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
  out[(j//cols)*th:(j//cols+1)*th,(j%cols)*tw:(j%cols+1)*tw]=im
 cv2.imwrite(f'/work/scratch/sheets/sam2_{s}.jpg',out)
PY"
```
<details><summary>output</summary>

```
0 3802 (675, 557, 796, 632)
50 3995 (673, 557, 768, 633)
100 3529 (674, 558, 768, 633)
150 3451 (674, 557, 768, 633)
200 3133 (671, 547, 745, 631)
300 2754 (671, 557, 761, 632)
400 4334 (678, 557, 767, 630)
500 1729 (672, 559, 795, 632)
550 2345 (667, 562, 735, 631)
600 2550 (675, 563, 729, 630)
650 1980 (662, 557, 725, 632)
700 2420 (678, 558, 759, 632)
720 2646 (675, 557, 747, 632)
740 2555 (674, 558, 796, 633)
760 2418 (675, 550, 797, 629)
780 2653 (674, 558, 796, 629)
800 2520 (679, 558, 797, 629)
850 2601 (677, 558, 797, 629)
895 2574 (675, 559, 797, 629)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/pose_search0.py <<'PY'
import numpy as np, torch, cv2, trimesh, nvdiffrast.torch as dr, json, h5py, math, time
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0)
# data
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda')
F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],float); T_wc=np.array(cam['T_world_cam'],float); T_cw=np.linalg.inv(T_wc)
frame=0
cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,img=cap.read()
m=cv2.imread(f'/work/scratch/sam2_iron/0/{frame:06d}.png',0)>0
ys,xs=np.where(m); cx=(xs.min()+xs.max())/2; cy=(ys.min()+ys.max())/2
x0=max(0,int(xs.min())-45); y0=max(0,int(ys.min())-45); x1=min(1536,int(xs.max())+46); y1=min(1152,int(ys.max())+46)
# force dims even
W=x1-x0; H=y1-y0
Kc=K.copy(); Kc[0,2]-=x0; Kc[1,2]-=y0
tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.float32,device='cuda')
print('crop',x0,y0,W,H,'target',tgt.sum().item(),'center',cx,cy)
# Base local y -> world up(-Y); local z -> world +X; local x=+Z
B=np.array([[0,0,1],[0,-1,0],[1,0,0]],float)
# Actually above columns? np rows maps axes: ensure B[:,0]=x->[0,0,1], B[:,1]=y->[0,-1,0], B[:,2]=z->[1,0,0]
B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])])
print('B',B,'det',np.linalg.det(B))
yaws=np.de …[+2477 chars]
```
<details><summary>output</summary>

```
crop 630 512 212 166 target 3802.0 center 735.5 594.5
B [[ 0.  0.  1.]
 [ 0. -1.  0.]
 [ 1.  0.  0.]] det 1.0
candidates (648, 4, 4)
BEST 0.5209003 522 yaw 290.0 depth 1.4
BEST 0.5184692 523 yaw 290.0 depth 1.5
BEST 0.5060788 93 yaw 50.0 depth 1.7
BEST 0.5004768 542 yaw 300.0 depth 1.5999999999999999
BEST 0.50007457 109 yaw 59.99999999999999 depth 1.5
BEST 0.49752125 524 yaw 290.0 depth 1.5999999999999999
BEST 0.4973383 76 yaw 40.0 depth 1.7999999999999998
BEST 0.49242866 541 yaw 300.0 depth 1.5
BEST 0.4910421 92 yaw 50.0 depth 1.5999999999999999
BEST 0.4909245 543 yaw 300.0 depth 1.7
BEST 0.4873519 110 yaw 59.99999999999999 depth 1.5999999999999999
BEST 0.48622283 94 yaw 50.0 depth 1.7999999999999998
BEST 0.48608625 504 yaw 280.0 depth 1.4
BEST 0.4819023 59 yaw 29.999999999999996 depth 1. …[+277 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/render_candidates.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,math
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);T_cw=np.linalg.inv(T_wc);frame=0
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,frame);_,img=cap.read();m=cv2.imread(f'/work/scratch/sam2_iron/0/{frame:06d}.png',0)>0;ys,xs=np.where(m);cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;x0=int(xs.min())-45;y0=int(ys.min())-45;x1=int(xs.max())+46;y1=int(ys.max())+46;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext()
B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])])
def render(yaw,z,tilt=0,roll=0):
 R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@B
 X=(cx-K[0,2])/K[0,0]*z;Y=(cy-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];T=np …[+1171 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_candidates.py", line 13, in <module>
    out,mask=render(yaw,z);roi=img[y0:y1,x0:x1];ov=roi.copy();ov[mask]=(out[mask]*.8+ov[mask]*.2).astype(np.uint8);panel=np.concatenate([roi,out,ov],1);cv2.putText(panel,f'y{yaw} z{z}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);rows.append(panel)
             ^^^^^^^^^^^^^
  File "/work/scratch/render_candidates.py", line 9, in render
    X=(cx-K[0,2])/K[0,0]*z;Y=(cy-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=torch.tensor(T_cw,dtype=torch.float32,device='cuda')[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];zz=Vc[...,2:3];uv=Vc[...,:2]/zz*torch.tensor([Kc[0,0],K …[+1866 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/\" /work/scratch/render_candidates.py
python /work/scratch/render_candidates.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
# instrument source to print bbox
s=open('/work/scratch/render_candidates.py').read();s=s.replace(\"out,mask=render(yaw,z);roi=\", \"out,mask=render(yaw,z);print('render',yaw,z,np.where(mask)[1].min(),np.where(mask)[1].max(),np.where(mask)[0].min(),np.where(mask)[0].max());roi=\")
open('/work/scratch/render_candidates2.py','w').write(s)
PY
python /work/scratch/render_candidates2.py"
```
<details><summary>output</summary>

```
render 290 1.4 44 140 46 121
render 290 1.5 48 138 48 118
render 50 1.7 49 165 52 111
render 300 1.6 45 149 51 116
render 59 1.5 48 171 48 115
render 290 1.6 52 136 51 116
render 40 1.8 48 171 54 109
render 300 1.5 41 152 48 118

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,trimesh,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); im=np.asarray(m.visual.material.baseColorTexture);print(im.shape,im.dtype,im.min(),im.max(),im.mean((0,1)),np.percentile(im,[1,5,25,50,75,95,99],axis=(0,1)))
cv2.imwrite('/work/scratch/texture.png',cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
# uv stats
uv=m.visual.uv;print(uv.shape,np.percentile(uv,[0,1,5,25,50,75,95,99,100],axis=0))
PY"
```
<details><summary>output</summary>

```
(1898, 1699, 3) uint8 0 255 [52.12438824 69.03235431 73.26982307] [[  0.   0.   0.]
 [  0.   0.   0.]
 [  0.   0.   0.]
 [ 37.  71.  67.]
 [ 87. 125. 126.]
 [175. 178. 217.]
 [191. 191. 217.]]
(298533, 2) [[0.00206    0.00184399]
 [0.05902    0.03256851]
 [0.12964232 0.07679099]
 [0.3374666  0.24427402]
 [0.54290926 0.5395568 ]
 [0.71141124 0.79353601]
 [0.88683897 0.93223602]
 [0.95001314 0.97633302]
 [0.99794    0.99815601]]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/render_candidates.py').read()
s=s.replace(\"icolor=dr.interpolate(UV,interp,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];mask=rast[0,...,3]>0;out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8)\",\"icolor=dr.interpolate(UV,interp,F)[0];mask=rast[0,...,3]>0\\n masknp=mask.cpu().numpy(); print('uvr',icolor[mask].min(0).values,icolor[mask].max(0).values)\\n color=dr.texture(tex,icolor,filter_mode='linear')[0]\\n out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8)\")
open('/work/scratch/render_debug.py','w').write(s)
PY
python /work/scratch/render_debug.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_debug.py", line 16, in <module>
    out,mask=render(yaw,z);roi=img[y0:y1,x0:x1];ov=roi.copy();ov[mask]=(out[mask]*.8+ov[mask]*.2).astype(np.uint8);panel=np.concatenate([roi,out,ov],1);cv2.putText(panel,f'y{yaw} z{z}',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);rows.append(panel)
             ^^^^^^^^^^^^^
  File "/work/scratch/render_debug.py", line 10, in render
    masknp=mask.cpu().numpy(); print('uvr',icolor[mask].min(0).values,icolor[mask].max(0).values)
                                           ~~~~~~^^^^^^
IndexError: The shape of the mask [166, 212] at index 0 does not match the shape of the indexed tensor [1, 166, 212, 2] at index 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/icolor\\[mask\\]/icolor[0][mask]/g\" /work/scratch/render_debug.py
python /work/scratch/render_debug.py"
```
<details><summary>output</summary>

```
uvr tensor([-0.5698,  0.0000], device='cuda:0') tensor([0.8415, 1.2348], device='cuda:0')
uvr tensor([-1.3675, -0.6167], device='cuda:0') tensor([0.8920, 2.4768], device='cuda:0')
uvr tensor([-0.2163,  0.0000], device='cuda:0') tensor([0.9036, 1.1233], device='cuda:0')
uvr tensor([ 0.0000, -0.0981], device='cuda:0') tensor([1.0220, 0.6983], device='cuda:0')
uvr tensor([-12.1306,  -7.3665], device='cuda:0') tensor([0.8712, 0.6398], device='cuda:0')
uvr tensor([0., 0.], device='cuda:0') tensor([0.8992, 0.6536], device='cuda:0')
uvr tensor([0., 0.], device='cuda:0') tensor([0.9115, 0.8176], device='cuda:0')
uvr tensor([-0.6611,  0.0000], device='cuda:0') tensor([0.8685, 1.8491], device='cuda:0')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'grep -n "icolor=" /work/scratch/render_debug.py'
```
<details><summary>output</summary>

```
9: X=(cx-K[0,2])/K[0,0]*z;Y=(cy-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=torch.tensor(T_cw,dtype=torch.float32,device='cuda')[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];zz=Vc[...,2:3];uv=Vc[...,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32);ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*zz,zz,zz],-1);rast,interp=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,interp,F)[0];mask=rast[0,...,3]>0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/render_debug.py';s=open(p).read();s=s.replace(\"masknp=mask.cpu().numpy(); print('uvr',icolor[0][mask].min(0).values,icolor[0][mask].max(0).values)\",\"masknp=mask.cpu().numpy(); uvm=icolor[0][mask]; print('uvr',torch.quantile(uvm,torch.tensor([0,.1,.5,.9,1.],device='cuda'),dim=0))\")
open(p,'w').write(s)
PY
python /work/scratch/render_debug.py"
```
<details><summary>output</summary>

```
uvr tensor([[-0.5698,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.4269,  0.2935],
        [ 0.8415,  1.2348]], device='cuda:0')
uvr tensor([[-1.3675, -0.6167],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.4286,  0.2936],
        [ 0.8920,  2.4768]], device='cuda:0')
uvr tensor([[-0.2163,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.4253,  0.2936],
        [ 0.9036,  1.1233]], device='cuda:0')
uvr tensor([[ 0.0000, -0.0981],
        [ 0.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 0.4280,  0.2935],
        [ 1.0220,  0.6983]], device='cuda:0')
uvr tensor([[-12.1306,  -7.3665],
        [  0.0000,   0.0000],
        [  0.0000,   0.0000],
        [  0.4246,   0.2935],
        [  0.8712,   0.6398] …[+486 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();uv=m.visual.uv
print('zero',np.mean(np.all(uv<1e-6,1)),np.mean(np.any(uv<1e-6,1)))
print('faces zero vertices',np.mean(np.all(np.all(uv[m.faces]<1e-6,2),1)),np.mean(np.any(np.all(uv[m.faces]<1e-6,2),1)))
# face area and uv zero
print('UV valid used ratio', np.unique(m.faces).size, len(uv))
# Does GLB have multiple primitives merged with uv index misalignment?
print(m.visual.uv[:20])
PY"
```
<details><summary>output</summary>

```
zero 0.0 0.0
faces zero vertices 0.0 0.0
UV valid used ratio 298533 298533
[[0.432758   0.29426897]
 [0.432758   0.29426897]
 [0.432758   0.29426897]
 [0.432758   0.29426897]
 [0.432758   0.29426897]
 [0.432758   0.29426897]
 [0.432758   0.29426897]
 [0.4228777  0.29540443]
 [0.4228777  0.29540443]
 [0.4228777  0.29540443]
 [0.4228777  0.29540443]
 [0.4228777  0.29540443]
 [0.4228777  0.29540443]
 [0.426566   0.29451299]
 [0.426566   0.29451299]
 [0.426566   0.29451299]
 [0.426566   0.29451299]
 [0.426566   0.29451299]
 [0.422539   0.29252899]
 [0.422539   0.29252899]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.getsource(dr.texture)[:5000])
PY"
```
<details><summary>output</summary>

```
def texture(tex, uv, uv_da=None, mip_level_bias=None, mip=None, filter_mode='auto', boundary_mode='wrap', max_mip_level=None):
    """Perform texture sampling.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Args:
        tex: Texture tensor with dtype `torch.float32`. For 2D textures, must have shape
             [minibatch_size, tex_height, tex_width, tex_channels]. For cube map textures,
             must have shape [minibatch_size, 6, tex_height, tex_width, tex_channels] where
             tex_width and tex_height are equal. Note that `boundary_mode` must also be set
             to 'cube' to enable cube map mode. Broadcasting is supported along the minibatch axis.
        uv: Tensor containing  …[+4201 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/uvdebug.py <<'PY'
exec(open('/work/scratch/render_candidates.py').read().split('cands=')[0])
out,mask=render(290,1.4)
# rerender internals no; recover uv via new clipped version copied quickly
import torch
yaw,z=290,1.4;R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@B;X=(cx-K[0,2])/K[0,0]*z;Y=(cy-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=torch.tensor(T_cw,dtype=torch.float32,device='cuda')[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];zz=Vc[...,2:3];uv=Vc[...,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32);ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*zz,zz,zz],-1);rast,interp=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,interp,F)[0];mask=rast[0,...,3]>0
col=torch.cat([icolor,torch.zeros_like(icolor[...,:1])],-1); im=(col.clamp(0,1)[0].cpu().numpy()*255).astype(np.uint8); im[~mask[0].cpu().numpy()]=0
cv2.imwrite('/work/scratch/uvdebug.png',cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
# nearest texture
im2=dr.texture(tex,icolor,filter_mode='nearest')[0]; im2=(im2.clamp(0,1)[0].cpu().numpy()*255).astype(np.uint8); im2[~mask[0].cpu().numpy()]=0;cv2.imwrite('/work/scratch/texnearest.png',cv2.cvtColor(im2,cv2.COLOR_RGB2BGR))
PY
python /work/scratch/uvdebug.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/uvdebug.py", line 6, in <module>
    col=torch.cat([icolor,torch.zeros_like(icolor[...,:1])],-1); im=(col.clamp(0,1)[0].cpu().numpy()*255).astype(np.uint8); im[~mask[0].cpu().numpy()]=0
                                                                                                                            ~~^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 166 but corresponding boolean dimension is 212

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i 's/mask\\[0\\].cpu().numpy()/mask.cpu().numpy()/g' /work/scratch/uvdebug.py
python /work/scratch/uvdebug.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/uvdebug.py", line 9, in <module>
    im2=dr.texture(tex,icolor,filter_mode='nearest')[0]; im2=(im2.clamp(0,1)[0].cpu().numpy()*255).astype(np.uint8); im2[~mask.cpu().numpy()]=0;cv2.imwrite('/work/scratch/texnearest.png',cv2.cvtColor(im2,cv2.COLOR_RGB2BGR))
                                                                                                                     ~~~^^^^^^^^^^^^^^^^^^^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 212 but corresponding boolean dimension is 166

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/im2=(im2.clamp(0,1)\\[0\\].cpu()/im2=(im2.clamp(0,1).cpu()/' /work/scratch/uvdebug.py
python /work/scratch/uvdebug.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/uvdebug.py';s=open(p).read();s=s.replace(\"icolor=dr.interpolate(UV,interp,F)[0];mask=rast[0,...,3]>0\",\"icolor=dr.interpolate(UV,interp,F)[0];mask=rast[0,...,3]>0\\nprint('bary stats',rast[0][mask][:,:3].min(0).values,rast[0][mask][:,:3].max(0).values,(rast[0][mask][:,:3].sum(-1)-1).abs().max().item()); print('uv nonzero', (icolor[0][mask].abs().sum(-1)>1e-6).float().mean().item())\")
open(p,'w').write(s)
PY
python /work/scratch/uvdebug.py"
```
<details><summary>output</summary>

```
bary stats tensor([0., 0., 1.], device='cuda:0') tensor([1., 1., 1.], device='cuda:0') 1.000000238418579
uv nonzero 0.16995596885681152

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
for p in ['/work/scratch/pose_search0.py','/work/scratch/render_candidates.py','/work/scratch/render_candidates2.py','/work/scratch/render_debug.py','/work/scratch/uvdebug.py','/work/scratch/test_render.py','/work/scratch/test_render2.py']:
 try:
  s=open(p).read()
  s=s.replace(\"pos=torch.cat([ndc*z,z,z],-1)\",\"pos=torch.cat([ndc*z,z*z,z],-1)\")
  s=s.replace(\"pos=torch.cat([ndc*zz,zz,zz],-1)\",\"pos=torch.cat([ndc*zz,zz*zz,zz],-1)\")
  open(p,'w').write(s)
 except:pass
PY
python /work/scratch/render_candidates.py
python /work/scratch/pose_search0.py | head -25"
```
<details><summary>output</summary>

```
crop 630 512 212 166 target 3802.0 center 735.5 594.5
B [[ 0.  0.  1.]
 [ 0. -1.  0.]
 [ 1.  0.  0.]] det 1.0
candidates (648, 4, 4)
BEST 0.0 647 yaw 350.0 depth 3.1
BEST 0.0 646 yaw 350.0 depth 3.0
BEST 0.0 645 yaw 350.0 depth 2.9
BEST 0.0 644 yaw 350.0 depth 2.8
BEST 0.0 643 yaw 350.0 depth 2.7
BEST 0.0 642 yaw 350.0 depth 2.6
BEST 0.0 641 yaw 350.0 depth 2.5
BEST 0.0 640 yaw 350.0 depth 2.4
BEST 0.0 639 yaw 350.0 depth 2.3
BEST 0.0 638 yaw 350.0 depth 2.2
BEST 0.0 637 yaw 350.0 depth 2.1
BEST 0.0 636 yaw 350.0 depth 2.0
BEST 0.0 635 yaw 350.0 depth 1.9
BEST 0.0 634 yaw 350.0 depth 1.7999999999999998
BEST 0.0 633 yaw 350.0 depth 1.7
BEST 0.0 632 yaw 350.0 depth 1.5999999999999999
BEST 0.0 631 yaw 350.0 depth 1.5
BEST 0.0 630 yaw 350.0 depth 1.4
BEST 0.0 629 yaw 340.0 depth 3.1
BEST 0.0 6 …[+23 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
for p in ['/work/scratch/pose_search0.py','/work/scratch/render_candidates.py','/work/scratch/render_candidates2.py','/work/scratch/render_debug.py','/work/scratch/uvdebug.py','/work/scratch/test_render.py','/work/scratch/test_render2.py']:
 try:
  s=open(p).read().replace(\"pos=torch.cat([ndc*z,z*z,z],-1)\",\"pos=torch.cat([ndc*z,(z-3.0)*z,z],-1)\").replace(\"pos=torch.cat([ndc*zz,zz*zz,zz],-1)\",\"pos=torch.cat([ndc*zz,(zz-3.0)*zz,zz],-1)\")
  open(p,'w').write(s)
 except:pass
PY
python /work/scratch/pose_search0.py | head -25
python /work/scratch/render_candidates.py"
```
<details><summary>output</summary>

```
crop 630 512 212 166 target 3802.0 center 735.5 594.5
B [[ 0.  0.  1.]
 [ 0. -1.  0.]
 [ 1.  0.  0.]] det 1.0
candidates (648, 4, 4)
BEST 0.48029798 60 yaw 29.999999999999996 depth 2.0
BEST 0.46134853 42 yaw 20.0 depth 2.0
BEST 0.45294 78 yaw 40.0 depth 2.0
BEST 0.44812322 61 yaw 29.999999999999996 depth 2.1
BEST 0.4431662 43 yaw 20.0 depth 2.1
BEST 0.4392265 79 yaw 40.0 depth 2.1
BEST 0.43304604 96 yaw 50.0 depth 2.0
BEST 0.431639 25 yaw 10.0 depth 2.1
BEST 0.42795545 583 yaw 320.0 depth 2.1
BEST 0.42703825 601 yaw 330.0 depth 2.1
BEST 0.4246977 62 yaw 29.999999999999996 depth 2.2
BEST 0.42160526 44 yaw 20.0 depth 2.2
BEST 0.42033443 97 yaw 50.0 depth 2.1
BEST 0.41810668 565 yaw 310.0 depth 2.1
BEST 0.41783482 619 yaw 340.0 depth 2.1
BEST 0.4163816 80 yaw 40.0 depth 2.2
BEST 0.41630307 40 …[+135 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import pytorch3d, torch
from pytorch3d.renderer import SoftSilhouetteShader, RasterizationSettings, MeshRenderer, MeshRasterizer, PerspectiveCameras
from pytorch3d.structures import Meshes
print('ok',pytorch3d.__version__)
PY"
```
<details><summary>output</summary>

```
ok 0.7.9

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/pose_search_rgb.py <<'PY'
import numpy as np, torch, cv2, trimesh, nvdiffrast.torch as dr, json, math, time
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);T_cw=np.linalg.inv(T_wc);frame=0
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,frame);_,img=cap.read();m=cv2.imread(f'/work/scratch/sam2_iron/0/{frame:06d}.png',0)>0;ys,xs=np.where(m);cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;x0=int(xs.min())-35;y0=int(ys.min())-35;x1=int(xs.max())+36;y1=int(ys.max())+36;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.float32,device='cuda');roi=torch.tensor(cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255; gl=dr.RasterizeCudaContext()
B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])])
yaws=np.deg2rad(np.arange(0,360,5));depths=np.linspace(1.3,3.3,21); tilts=np.deg2rad([-8,0,8])
poses=[];meta=[]
f …[+2995 chars]
```
<details><summary>output</summary>

```
  File "/work/scratch/pose_search_rgb.py", line 20
    maski=(pred&Tg)[...,None].float(); l1=((color-roi[None]).abs()*maski).sum((1,2,3))/maski.sum((1,2,3)).clamp_min(1)/3
IndentationError: unexpected indent

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "nl -ba /work/scratch/pose_search_rgb.py | sed -n '15,25p'"
```
<details><summary>output</summary>

```
    15	   q=np.eye(4);q[:3,:3]=R;q[:3,3]=t;poses.append(q);meta.append((yaw,tilt,z))
    16	poses=torch.tensor(np.array(poses),dtype=torch.float32,device='cuda');Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda'); res=[];t0=time.time()
    17	for s in range(0,len(poses),12):
    18	 T=Tcw[None]@poses[s:s+12]; Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,(z-3.0)*z,z],-1);rast,interp=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,interp,F)[0];color=dr.texture(tex,icolor,filter_mode='linear');pred=rast[...,3]>0;Tg=tgt.bool()[None]; inter=(pred&Tg).sum((1,2)).float();un …[+644 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i '20s/"'^  / /;21s/''^  / /;22s/''^  / /'"' /work/scratch/pose_search_rgb.py
python /work/scratch/pose_search_rgb.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pose_search_rgb.py", line 18, in <module>
    T=Tcw[None]@poses[s:s+12]; Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,(z-3.0)*z,z],-1);rast,interp=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,interp,F)[0];color=dr.texture(tex,icolor,filter_mode='linear');pred=rast[...,3]>0;Tg=tgt.bool()[None]; inter=(pred&Tg).sum((1,2)).float();union=(pred|Tg).sum((1,2)).float();iou=inter/union.clamp_min(1)
                                                                                                                              …[+1090 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/pose_search_rgb.py
python /work/scratch/pose_search_rgb.py"
```
<details><summary>output</summary>

```
elapsed 1.616713285446167
BEST 0.487 0.372 0.035 yaw 25.0 tilt 0.0 z 2.0 1624
BEST 0.483 0.371 0.06 yaw 30.0 tilt -8.0 z 2.0 133
BEST 0.483 0.372 0.043 yaw 25.0 tilt 8.0 z 2.0 3136
BEST 0.483 0.372 0.031 yaw 25.0 tilt -8.0 z 2.0 112
BEST 0.48 0.375 0.034 yaw 30.0 tilt 0.0 z 2.0 1645
BEST 0.477 0.375 0.054 yaw 35.0 tilt -8.0 z 2.0 154
BEST 0.478 0.375 0.028 yaw 30.0 tilt 8.0 z 2.0 3157
BEST 0.468 0.379 0.022 yaw 35.0 tilt 0.0 z 2.0 1666
BEST 0.465 0.378 0.036 yaw 20.0 tilt 8.0 z 2.0 3115
BEST 0.462 0.377 0.038 yaw 35.0 tilt 8.0 z 2.0 3178
BEST 0.461 0.379 0.029 yaw 20.0 tilt 0.0 z 2.0 1603
BEST 0.463 0.384 0.022 yaw 40.0 tilt -8.0 z 2.0 175
BEST 0.462 0.38 0.006 yaw 20.0 tilt -8.0 z 2.0 91
BEST 0.458 0.388 0.028 yaw 45.0 tilt -8.0 z 2.0 196
BEST 0.448 0.365 0.025 yaw 30.0 tilt 8.0 z 2.1 315 …[+781 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
for p in ['/work/scratch/render_cands.png','/work/scratch/rgbsearch_top.png']:
 im=cv2.imread(p); print(p,im.shape,im.mean(),im.max(),(im>10).sum())
PY
grep -o \"color=dr.texture["'^;]*" /work/scratch/pose_search_rgb.py /work/scratch/render_candidates.py'
```
<details><summary>output</summary>

```
/work/scratch/render_cands.png (1328, 636, 3) 69.35572952186102 255 1630852
/work/scratch/rgbsearch_top.png (2920, 576, 3) 66.62373438292745 255 3215228
/work/scratch/pose_search_rgb.py:color=dr.texture(tex,icolor,filter_mode='linear')
/work/scratch/pose_search_rgb.py:color=dr.texture(tex,icolor,filter_mode='linear')[0]
/work/scratch/render_candidates.py:color=dr.texture(tex,icolor,filter_mode='linear')[0]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
im=cv2.imread('/work/scratch/rgbsearch_top.png');print(im.shape)
# first row height=146? detect actual crop H=146
H=146;W=192
for j in range(3):
 c=im[:H,j*W:(j+1)*W];print(j,c.mean(),(c>10).sum(),c.max())
 cv2.imwrite(f'/work/scratch/top_{j}.png',c)
PY"
```
<details><summary>output</summary>

```
(2920, 576, 3)
0 104.88962614155251 80497 255
1 0.3114892503805175 111 255
2 94.72883371385083 80080 255

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n \"def render\\|icolor\\|rast,interp\" /work/scratch/render_candidates.py | tail -5; grep -n \"for iou\" -A5 /work/scratch/pose_search_rgb.py"
```
<details><summary>output</summary>

```
7:def render(yaw,z,tilt=0,roll=0):
9: X=(cx-K[0,2])/K[0,0]*z;Y=(cy-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=torch.tensor(T_cw,dtype=torch.float32,device='cuda')[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];zz=Vc[...,2:3];uv=Vc[...,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32);ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*zz,(zz-3.0)*zz,zz],-1);rast,interp=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,interp,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];mask=rast[0,...,3]>0;out=(color.clamp(0,1).cpu().numpy()*255) …[+1229 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed \"s/cands=\\[(290,1.4).*/cands=[(25,2.0)]/\" /work/scratch/render_candidates.py > /work/scratch/render_one.py
python /work/scratch/render_one.py
python - <<'PY'
import cv2
im=cv2.imread('/work/scratch/render_cands.png');print(im.shape,im.mean(),(im>10).sum())
PY"
```
<details><summary>output</summary>

```
(166, 636, 3) 66.42390315980904 202510

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python /work/scratch/uvdebug.py | head -10
python - <<'PY'
import cv2
for f in ['uvdebug','texnearest']:
 im=cv2.imread('/work/scratch/'+f+'.png');print(f,im.mean(),(im>10).sum())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/uvdebug.py", line 6, in <module>
    print('bary stats',rast[0][mask][:,:3].min(0).values,rast[0][mask][:,:3].max(0).values,(rast[0][mask][:,:3].sum(-1)-1).abs().max().item()); print('uv nonzero', (icolor[0][mask].abs().sum(-1)>1e-6).float().mean().item())
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: min(): Expected reduction dim 0 to have non-zero size.
uvdebug 1.8159146018034402 2083
texnearest 0.04698037432749867 40

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import glob
for p in glob.glob('/work/scratch/*.py'):
 s=open(p).read(); n=s.replace(\"(z-3.0)*z\",\"((z-5.0)/5.0)*z\").replace(\"(zz-3.0)*zz\",\"((zz-5.0)/5.0)*zz\")
 if n"'!=s: open(p,'"'w').write(n); print(p)
PY
python /work/scratch/pose_search_rgb.py
python /work/scratch/uvdebug.py | head"
```
<details><summary>output</summary>

```
/work/scratch/pose_search0.py
/work/scratch/render_candidates.py
/work/scratch/render_candidates2.py
/work/scratch/render_debug.py
/work/scratch/uvdebug.py
/work/scratch/pose_search_rgb.py
/work/scratch/render_one.py
elapsed 3.054920196533203
BEST 0.529 0.362 0.024 yaw 290.0 tilt -8.0 z 1.5 1220
BEST 0.526 0.36 0.007 yaw 285.0 tilt -8.0 z 1.4 1198
BEST 0.522 0.357 0.025 yaw 285.0 tilt 0.0 z 1.4 2710
BEST 0.521 0.355 0.021 yaw 290.0 tilt 0.0 z 1.4 2731
BEST 0.523 0.361 0.015 yaw 290.0 tilt -8.0 z 1.4 1219
BEST 0.519 0.358 0.036 yaw 290.0 tilt 0.0 z 1.5 2732
BEST 0.517 0.362 0.023 yaw 295.0 tilt -8.0 z 1.5 1241
BEST 0.514 0.363 0.032 yaw 295.0 tilt -8.0 z 1.6 1242
BEST 0.513 0.357 0.002 yaw 285.0 tilt -8.0 z 1.3 1197
BEST 0.511 0.356 0.004 yaw 295.0 tilt 0.0 z 1.5 2753
BEST 0.509 0.355 0.013 …[+1198 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'python /work/scratch/uvdebug.py 2>&1 | head -5'
```
<details><summary>output</summary>

```
bary stats tensor([ 0.0000,  0.0000, -0.7465], device='cuda:0') tensor([ 1.0000,  1.0000, -0.6869], device='cuda:0') 1.7369740009307861
uv nonzero 0.05981794372200966

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/pyrender_test.py <<'PY'
import pyrender,trimesh,numpy as np,cv2,json
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[1,1,1])
node=scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);W,H=cam['width'],cam['height'];T_wc=np.array(cam['T_world_cam']);w2c=np.linalg.inv(T_wc)
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0];Tc=w2c@P
# OpenGL cam world: rows 0,-1,2 of camera
pose=np.eye(4);pose[:3,:3]=np.diag([1,-1,1])@Tc[:3,:3];pose[:3,3]=np.diag([1,-1,1])@Tc[:3,3]
camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=100)
scene.add(camera,pose=pose)
r=pyrender.OffscreenRenderer(W,H);color,depth=r.render(scene);cv2.imwrite('/work/scratch/pyrender0.png',cv2.cvtColor(color,cv2.COLOR_RGB2BGR));print(color.shape,color.max(),(color>0).sum())
PY
python /work/scratch/pyrender_test.py"
```
<details><summary>output</summary>

```
(1152, 1536, 3) 0 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/pyrender_test.py';s=open(p).read();s=s.replace(\"pose=np.eye(4);pose[:3,:3]=np.diag([1,-1,1])@Tc[:3,:3];pose[:3,3]=np.diag([1,-1,1])@Tc[:3,3]\",\"Mc2w=np.linalg.inv(Tc); pose=np.eye(4);pose[:3,:3]=Mc2w[:3,:3]@np.diag([1,-1,1]);pose[:3,3]=Mc2w[:3,3]\")
open(p,'w').write(s)
PY
python /work/scratch/pyrender_test.py"
```
<details><summary>output</summary>

```
(1152, 1536, 3) 0 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/pyrender_test.py';s=open(p).read();a=s.index('Mc2w=');b=s.index('\\ncamera=',a);s=s[:a]+\"Mc2w=T_wc@np.diag([1,-1,-1,1]); pose=Mc2w\"+s[b:];open(p,'w').write(s)
PY
python /work/scratch/pyrender_test.py"
```
<details><summary>output</summary>

```
(1152, 1536, 3) 12 517899

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np,h5py
im=cv2.imread('/work/scratch/pyrender0.png');m=im.max(2)>0;ys,xs=np.where(m);print(xs.min(),xs.max(),ys.min(),ys.max(),m.sum())
cap=cv2.VideoCapture('/work/inputs/video.mp4');ok,fr=cap.read();
x0=max(0,xs.min()-50);x1=xs.max()+50;y0=max(0,ys.min()-50);y1=ys.max()+50
ov=fr[y0:y1,x0:x1].copy();r=im[y0:y1,x0:x1];mm=r.max(2)>0;ov[mm]=(r[mm]*.8+ov[mm]*.2).astype(np.uint8)
cv2.imwrite('/work/scratch/pyrendercrop.png',np.concatenate([fr[y0:y1,x0:x1],r,ov],1))
PY"
```
<details><summary>output</summary>

```
1095 1535 0 464 172633

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.getsource(dr.rasterize)[:4500])
PY"
```
<details><summary>output</summary>

```
def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    '''Rasterize triangles.

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
        …[+1402 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import glob
for p in glob.glob('/work/scratch/*.py'):
 s=open(p).read(); n=s.replace('rast,interp=dr.rasterize','rast,rast_db=dr.rasterize').replace('dr.interpolate(UV,interp,F)','dr.interpolate(UV,rast,F)')
 if n"'!=s: open(p,'"'w').write(n); print(p)
PY
python /work/scratch/pose_search_rgb.py"
```
<details><summary>output</summary>

```
/work/scratch/test_render.py
/work/scratch/test_render2.py
/work/scratch/render_candidates.py
/work/scratch/render_candidates2.py
/work/scratch/render_debug.py
/work/scratch/uvdebug.py
/work/scratch/pose_search_rgb.py
/work/scratch/render_one.py
elapsed 3.1379823684692383
BEST 0.505 0.213 0.78 yaw 55.0 tilt 0.0 z 1.6 1746
BEST 0.507 0.215 0.772 yaw 50.0 tilt 0.0 z 1.7 1726
BEST 0.501 0.211 0.787 yaw 60.0 tilt 0.0 z 1.5 1766
BEST 0.505 0.219 0.78 yaw 55.0 tilt -8.0 z 1.6 234
BEST 0.501 0.213 0.79 yaw 60.0 tilt -8.0 z 1.5 254
BEST 0.503 0.224 0.767 yaw 50.0 tilt -8.0 z 1.7 214
BEST 0.494 0.208 0.788 yaw 65.0 tilt 0.0 z 1.4 1786
BEST 0.495 0.213 0.774 yaw 55.0 tilt 8.0 z 1.6 3258
BEST 0.494 0.212 0.775 yaw 60.0 tilt 8.0 z 1.5 3278
BEST 0.498 0.22 0.766 yaw 45.0 tilt 0.0 z 1.7 1705
BEST 0.493  …[+1028 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
from pytorch3d.renderer import PerspectiveCameras
import inspect
print(inspect.signature(PerspectiveCameras.__init__))
PY"
```
<details><summary>output</summary>

```
(self, focal_length: Union[float, Sequence[Tuple[float]], Sequence[Tuple[float, float]], torch.Tensor] = 1.0, principal_point=((0.0, 0.0),), R: torch.Tensor = tensor([[[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]]]), T: torch.Tensor = tensor([[0., 0., 0.]]), K: Optional[torch.Tensor] = None, device: Union[str, torch.device] = 'cpu', in_ndc: bool = True, image_size: Union[List, Tuple, torch.Tensor, NoneType] = None) -> None

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/p3d_test.py <<'PY'
import numpy as np,torch,cv2,trimesh,json
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras,MeshRenderer,MeshRasterizer,RasterizationSettings,SoftSilhouetteShader
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32);F=torch.tensor(mesh.faces,dtype=torch.int64)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);w2c=np.linalg.inv(T_wc);P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0];Tc=w2c@P
D=np.diag([-1.,-1.,1.]);R=D@Tc[:3,:3];T=D@Tc[:3,3]
Kt=torch.tensor(K,dtype=torch.float32)[None]
cams=PerspectiveCameras(K=Kt,R=R[None],T=T[None],in_ndc=False,image_size=((1152,1536),),device='cuda')
mesh=Meshes(verts=[V.cuda()],faces=[F.cuda()])
raster=MeshRasterizer(cameras=cams,raster_settings=RasterizationSettings(image_size=(1152,1536),blur_radius=1e-4,faces_per_pixel=1,bin_size=0))
ren=MeshRenderer(rasterizer=raster,shader=SoftSilhouetteShader(blend_params=torch.tensor([1e-4,1e-4])))
alpha=ren(mesh)[0,...,3]
print(alpha.shape,alpha.min(),alpha.max(),(alpha>.5).sum())
ys,xs=torch.where(alpha>.5);print(xs.min(),xs.max(),ys.min(),ys.max())
cv2.imwrite('/work/scratch/p3d_alpha.png',(alpha.clamp(0,1).cpu().numpy()*255).astype(np.uint8))
PY
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 12, in <module>
    alpha=ren(mesh)[0,...,3]
          ^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/mesh/renderer.py", line 63, in forward
    fragments = self.rasterizer(meshes_world, **kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/mo …[+1494 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/Kt=torch.tensor(K,dtype=torch.float32)\\[None\\]/Kt=torch.tensor(np.block([[K,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]]),dtype=torch.float32)[None]/\" /work/scratch/p3d_test.py
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 12, in <module>
    alpha=ren(mesh)[0,...,3]
          ^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/mesh/renderer.py", line 64, in forward
    images = self.shader(fragments, meshes_world, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/tor …[+935 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/from pytorch3d.renderer import PerspectiveCameras,MeshRenderer,MeshRasterizer,RasterizationSettings,SoftSilhouetteShader/from pytorch3d.renderer import PerspectiveCameras,MeshRenderer,MeshRasterizer,RasterizationSettings,SoftSilhouetteShader,BlendParams/; s/SoftSilhouetteShader(blend_params=torch.tensor(\\[1e-4,1e-4\\]))/SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4,gamma=1e-4))/\" /work/scratch/p3d_test.py
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
torch.Size([1152, 1536]) tensor(0., device='cuda:0') tensor(0., device='cuda:0') tensor(0, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 14, in <module>
    ys,xs=torch.where(alpha>.5);print(xs.min(),xs.max(),ys.min(),ys.max())
                                      ^^^^^^^^
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/p3d_test.py';s=open(p).read();s=s.replace(\"D=np.diag([-1.,-1.,1.]);R=D@Tc[:3,:3];T=D@Tc[:3,3]\",\"D=np.diag([-1.,-1.,1.]);V=torch.tensor((D@(Tc[:3,:3]@mesh.vertices.T+Tc[:3,3,None])).T,dtype=torch.float32);R=np.eye(3);T=np.zeros(3)\")
open(p,'w').write(s)
PY
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
torch.Size([1152, 1536]) tensor(0., device='cuda:0') tensor(0., device='cuda:0') tensor(0, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 14, in <module>
    ys,xs=torch.where(alpha>.5);print(xs.min(),xs.max(),ys.min(),ys.max())
                                      ^^^^^^^^
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -R \"class PerspectiveCameras\" -n /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/cameras.py | head; grep -n \"class FoVOrthographicCameras\" -A30 /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/cameras.py | head -40; grep -n \"get_ndc_camera_transform\" -A60 /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/cameras.py | grep -n \"z\\|x\\|y\" | head"
```
<details><summary>output</summary>

```
1030:class PerspectiveCameras(CamerasBase):
793:class FoVOrthographicCameras(CamerasBase):
794-    """
795-    A class which stores a batch of parameters to generate a batch of
796-    projection matrices by specifying the field of view.
797-    The definitions of the parameters follow the OpenGL orthographic camera.
798-    """
799-
800-    # For __getitem__
801-    _FIELDS = (
802-        "K",
803-        "znear",
804-        "zfar",
805-        "R",
806-        "T",
807-        "max_y",
808-        "min_y",
809-        "max_x",
810-        "min_x",
811-        "scale_xyz",
812-    )
813-
814-    def __init__(
815-        self,
816-        znear: _BatchFloatType = 1.0,
817-        zfar: _BatchFloatType = 100.0,
818-        max_y: _BatchFloatType = 1.0,
819-        min_y: _BatchFloatType  …[+853 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '45,90p' /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/cameras.py"
```
<details><summary>output</summary>

```
        The transformation from world --> view happens after applying a rotation (R)
        and translation (T)
    - NDC coordinate system: This is the normalized coordinate system that confines
        points in a volume the rendered part of the object or scene, also known as
        view volume. For square images, given the PyTorch3D convention, (+1, +1, znear)
        is the top left near corner, and (-1, -1, zfar) is the bottom right far
        corner of the volume.
        The transformation from view --> NDC happens after applying the camera
        projection matrix (P) if defined in NDC space.
        For non square images, we scale the points such that smallest side
        has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    - Screen coordinate system: Thi …[+1864 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n \"def get_projection_transform\" -A110 /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/cameras.py | sed -n '1,140p'"
```
<details><summary>output</summary>

```
96:    def get_projection_transform(self, **kwargs):
97-        """
98-        Calculate the projective transformation matrix.
99-
100-        Args:
101-            **kwargs: parameters for the projection can be passed in as keyword
102-                arguments to override the default values set in `__init__`.
103-
104-        Return:
105-            a `Transform3d` object which represents a batch of projection
106-            matrices of shape (N, 3, 3)
107-        """
108-        raise NotImplementedError()
109-
110-    def unproject_points(self, xy_depth: torch.Tensor, **kwargs):
111-        """
112-        Transform input points from camera coordinates (NDC or screen)
113-        to the world / camera coordinates.
114-
115-        Each of the input points `xy_depth` of shape (..., 3)  …[+5431 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '1090,1185p' /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/cameras.py"
```
<details><summary>output</summary>

```
            principal_point=principal_point,
            R=R,
            T=T,
            K=K,
            _in_ndc=in_ndc,
            **kwargs,  # pyre-ignore
        )
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        # When focal length is provided as one value, expand to
        # create (N, 2) shape tensor
        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(-1, 2)  # (N, 2)

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the projection matrix usin …[+2642 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/p3d_test.py';s=open(p).read();import re
s=s.replace(\"Kt=torch.tensor(np.block([[K,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]]),dtype=torch.float32)[None]\",\"K4=np.array([[K[0,0],0,K[0,2],0],[0,K[1,1],K[1,2],0],[0,0,0,1],[0,0,1,0]],np.float32); Kt=torch.tensor(K4)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
torch.Size([1152, 1536]) tensor(0., device='cuda:0') tensor(0.4992, device='cuda:0') tensor(0, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 14, in <module>
    ys,xs=torch.where(alpha>.5);print(xs.min(),xs.max(),ys.min(),ys.max())
                                      ^^^^^^^^
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i 's/sigma=1e-4,gamma=1e-4/sigma=1e-2,gamma=1e-4/' /work/scratch/p3d_test.py
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
torch.Size([1152, 1536]) tensor(0., device='cuda:0') tensor(0.5000, device='cuda:0') tensor(0, device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 14, in <module>
    ys,xs=torch.where(alpha>.5);print(xs.min(),xs.max(),ys.min(),ys.max())
                                      ^^^^^^^^
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc 'sed -i "s/print(alpha.shape,alpha.min(),alpha.max(),(alpha>.5).sum())/print(alpha.shape,alpha.min(),alpha.max(),torch.unique(alpha).numel(),torch.unique(alpha)[:20])/" /work/scratch/p3d_test.py
python /work/scratch/p3d_test.py'
```
<details><summary>output</summary>

```
torch.Size([1152, 1536]) tensor(0., device='cuda:0') tensor(0.5000, device='cuda:0') 2257 tensor([0.0000, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975,
        0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975,
        0.4975, 0.4975], device='cuda:0')
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 14, in <module>
    ys,xs=torch.where(alpha>.5);print(xs.min(),xs.max(),ys.min(),ys.max())
                                      ^^^^^^^^
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i 's/alpha>.5/alpha>.1/g' /work/scratch/p3d_test.py
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
torch.Size([1152, 1536]) tensor(0., device='cuda:0') tensor(0.5000, device='cuda:0') 2257 tensor([0.0000, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975,
        0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975, 0.4975,
        0.4975, 0.4975], device='cuda:0')
tensor(739, device='cuda:0') tensor(787, device='cuda:0') tensor(501, device='cuda:0') tensor(587, device='cuda:0')

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/opt_static.py <<'PY'
import numpy as np, torch, cv2, trimesh, nvdiffrast.torch as dr, json, time, math, sys
from scipy.spatial.transform import Rotation as Rot
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras,MeshRenderer,MeshRasterizer,RasterizationSettings,SoftSilhouetteShader,BlendParams
np.random.seed(0);torch.manual_seed(0)
# object and camera
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
Vnp=np.array(mesh.vertices,np.float32);Fnp=np.array(mesh.faces,np.int64)
V=torch.tensor(Vnp,device='cuda');F=torch.tensor(Fnp.astype(np.int32),device='cuda');UV=torch.tensor(mesh.visual.uv,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);T_cw=np.linalg.inv(T_wc)
frames=[0,30,60,90]
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs={}; masks={}
for i in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read(); imgs[i]=im
 masks[i]=cv2.imread(f'/work/scratch/sam2_iron/0/{i:06d}.png',0)>0
# common crop from union frame0 target plus margin
ys,xs=np.where(masks[0]); pad=30;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0
# initial pose from grid best: yaw55, tilt0, z1.6, centered on bbox center frame0
cx=(xs.min()+xs.m …[+4417 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_static.py", line 62, in <module>
    alpha,color,pm=render_frame(i,R,t);tg=tgts[i]
                   ^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_static.py", line 53, in render_frame
    rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];pm=rast[0,...,3]>0
                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return s …[+397 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/UV=torch.tensor(mesh.visual.uv,device='cuda')/UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')/\" /work/scratch/opt_static.py
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
0 loss 13.034666061401367 mask 11.706558227539062 photo 0.8854049444198608 rv [0. 0. 0.] dt [ 0.01  0.01 -0.01]
10 loss 49.519493103027344 mask 47.932823181152344 photo 1.0564457178115845 rv [0. 0. 0.] dt [ 0.08846474 -0.00568172 -0.06463711]
../aten/src/ATen/native/cuda/Loss.cu:94: operator(): block: [34,0,0], thread: [0,0,0] Assertion `input_val >= zero && input_val <= one` failed.
../aten/src/ATen/native/cuda/Loss.cu:94: operator(): block: [34,0,0], thread: [1,0,0] Assertion `input_val >= zero && input_val <= one` failed.
../aten/src/ATen/native/cuda/Loss.cu:94: operator(): block: [34,0,0], thread: [2,0,0] Assertion `input_val >= zero && input_val <= one` failed.
../aten/src/ATen/native/cuda/Loss.cu:94: operator(): block: [34,0,0], thread: [3,0,0] Assertion `input_val >= zero && input_v …[+102583 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/mask_stats_candidate.py <<'PY'
exec(open('/work/scratch/render_candidates.py').read().split('cands=')[0])
for yaw,z in [(55,1.6),(50,1.7),(60,1.5),(290,1.4),(55,1.9),(55,1.4),(45,1.7),(65,1.5)]:
 out,mask=render(yaw,z);ys,xs=np.where(mask);ty,tx=np.where(m[y0:y1,x0:x1]);
 print(yaw,z,'pred',xs.mean(),ys.mean(),xs.min(),xs.max(),ys.min(),ys.max(),'tgt',tx.mean(),ty.mean(),tx.min(),tx.max(),ty.min(),ty.max())
PY
python /work/scratch/mask_stats_candidate.py"
```
<details><summary>output</summary>

```
55 1.6 pred 100.14968424645843 82.17016555726234 49 168 50 113 tgt 87.52630194634403 82.36927932667017 45 166 45 120
50 1.7 pred 99.99963656187533 82.08758858804289 49 165 52 111 tgt 87.52630194634403 82.36927932667017 45 166 45 120
60 1.5 pred 100.67070642640728 82.23042576941477 49 171 48 115 tgt 87.52630194634403 82.36927932667017 45 166 45 120
290 1.4 pred 94.9231144343303 84.01885565669701 44 140 46 121 tgt 87.52630194634403 82.36927932667017 45 166 45 120
55 1.9 pred 100.62758786711603 82.07968223399133 57 158 55 108 tgt 87.52630194634403 82.36927932667017 45 166 45 120
55 1.4 pred 99.89630885613668 82.2298160949524 41 177 46 117 tgt 87.52630194634403 82.36927932667017 45 166 45 120
45 1.7 pred 100.38137624101336 82.076857240671 47 170 52 111 tgt 87.52630194634403 82.36927932667017 4 …[+129 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/search_tilt.py <<'PY'
# adapted from render_candidates definitions
exec(open('/work/scratch/render_candidates.py').read().split('cands=')[0])
import numpy as np
rows=[]
for tilt in [-30,-20,-10,0,10,20,30]:
 for yaw,z in [(55,1.6),(55,1.5),(55,1.4),(50,1.5),(60,1.5)]:
  R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@B
  X=(cx-K[0,2])/K[0,0]*z;Y=(cy-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=torch.tensor(T_cw,dtype=torch.float32,device='cuda')[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];zz=Vc[...,2:3];uv=Vc[...,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32);ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];pm=rast[0,...,3]>0;out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8);out[~pm.cpu().numpy()]=0;inter=(pm&tgt.bool()).sum().item();union=(pm|tgt.bool()).sum().item();iou=inter/union;roi=img[y0:y1,x0:x1];mm=pm.cpu().numpy();ov=roi.copy();ov[mm]=(out[mm]*.8+ov[mm]*.2).astype(np.uint8);p=np.concatenate([roi,out,ov],1);cv2.putText(p,f'{iou:.3f} y{yaw} t{tilt} z{z}',(4,15),cv2.FONT_HERSH …[+293 chars]
```
<details><summary>output</summary>

```
0 55 1.6 0.5052976004985977
-10 55 1.6 0.5016221226633709
0 60 1.5 0.5014160083469966
-10 60 1.5 0.4973269973269973
10 55 1.6 0.4909725383098164
10 60 1.5 0.488096976778151
0 55 1.5 0.48624503686897336
-10 55 1.5 0.4842979861991269
-20 55 1.6 0.4781701683802712
10 55 1.5 0.47493769039047357
-20 60 1.5 0.4700732022391273
-30 55 1.6 0.4654384461582405
0 50 1.5 0.4653825431034483
20 55 1.6 0.46165062700495774
-30 60 1.5 0.4613156805957799
-10 50 1.5 0.4602016449986734
20 60 1.5 0.45819304152637486
-20 55 1.5 0.4548770602539854
0 55 1.4 0.45250759878419455
10 50 1.5 0.4508239602406487

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
m=cv2.imread('/work/scratch/sam2_iron/0/000000.png',0);ys,xs=np.where(m>0);x0=xs.min()-20;y0=ys.min()-20;x1=xs.max()+20;y1=ys.max()+20
cv2.imwrite('/work/scratch/targetmask0.png',m[y0:y1,x0:x1])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/clean_masks.py <<'PY'
import cv2,numpy as np,os
for i in range(896):
 p=f'/work/scratch/sam2_iron/0/{i:06d}.png';m=cv2.imread(p,0)>0
 n,lab,stats,cent=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
 if n>1:
  areas=stats[1:,cv2.CC_STAT_AREA]; ma=areas.max(); keep=np.where(areas>=max(20,0.015*ma))[0]+1
  out=np.isin(lab,keep)
 else:out=m
 os.makedirs('/work/scratch/iron_masks',exist_ok=True);cv2.imwrite(f'/work/scratch/iron_masks/{i:06d}.png',out.astype(np.uint8)*255)
PY
python /work/scratch/clean_masks.py
python - <<'PY'
import cv2,numpy as np
for i in [0,50,100,200,400,600,700,760,800,895]:
 m=cv2.imread(f'/work/scratch/iron_masks/{i:06d}.png',0)>0;ys,xs=np.where(m);print(i,m.sum(),(xs.min(),ys.min(),xs.max(),ys.max()))
PY"
```
<details><summary>output</summary>

```
0 3735 (675, 557, 769, 632)
50 3995 (673, 557, 768, 633)
100 3494 (678, 558, 768, 630)
200 3133 (671, 547, 745, 631)
400 4334 (678, 557, 767, 630)
600 2550 (675, 563, 729, 630)
700 2414 (678, 558, 759, 632)
760 2400 (675, 550, 771, 629)
800 2495 (679, 558, 771, 629)
895 2555 (675, 559, 770, 629)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/pose_search_rgb.py';s=open(p).read();s=s.replace(\"f'/work/scratch/sam2_iron/0/{frame:06d}.png'\",\"f'/work/scratch/iron_masks/{frame:06d}.png'\")
s=s.replace('yaws=np.deg2rad(np.arange(0,360,5));depths=np.linspace(1.3,3.3,21); tilts=np.deg2rad([-8,0,8])','yaws=np.deg2rad(np.arange(0,360,5));depths=np.linspace(1.6,4.0,25); tilts=np.deg2rad([-12,-6,0,6,12])')
s=s.replace(\"sorted(res,key=lambda x:-(x[0]-0.4*x[1]+0.1*x[2]))\",\"sorted(res,key=lambda x:-(x[0]-0.4*x[1]+0.2*x[2]))\")
open('/work/scratch/pose_search_rgb2.py','w').write(s)
PY
python /work/scratch/pose_search_rgb2.py"
```
<details><summary>output</summary>

```
elapsed 4.902421951293945
BEST 0.566 0.208 0.798 yaw 75.0 tilt -12.0 z 1.6 375
BEST 0.559 0.206 0.796 yaw 70.0 tilt -12.0 z 1.7 351
BEST 0.561 0.214 0.785 yaw 70.0 tilt -12.0 z 1.6 350
BEST 0.551 0.206 0.805 yaw 75.0 tilt -6.0 z 1.6 2175
BEST 0.556 0.212 0.788 yaw 70.0 tilt -6.0 z 1.6 2150
BEST 0.554 0.209 0.791 yaw 70.0 tilt -6.0 z 1.7 2151
BEST 0.55 0.214 0.784 yaw 65.0 tilt -12.0 z 1.8 327
BEST 0.553 0.219 0.779 yaw 65.0 tilt -12.0 z 1.7 326
BEST 0.547 0.211 0.791 yaw 70.0 tilt 0.0 z 1.6 3950
BEST 0.541 0.201 0.803 yaw 70.0 tilt -12.0 z 1.8 352
BEST 0.541 0.203 0.803 yaw 75.0 tilt -12.0 z 1.7 376
BEST 0.548 0.213 0.785 yaw 65.0 tilt -6.0 z 1.8 2127
BEST 0.548 0.217 0.782 yaw 65.0 tilt -6.0 z 1.7 2126
BEST 0.538 0.203 0.798 yaw 65.0 tilt -12.0 z 1.9 328
BEST 0.539 0.208 0.792 yaw 70.0 ti …[+808 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
# use render_candidates load and render yaw70 tilt-12 z1.7 but need function support tilt
s=open('/work/scratch/render_candidates.py').read().split('cands=')[0]
exec(s)
for yaw,tilt,z in [(75,-12,1.6),(70,-12,1.7),(70,-12,2.0),(70,-12,2.5),(70,0,2.0),(70,0,2.5),(290,0,2.0)]:
 # inline render code via direct
 R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@B;X=(cx-K[0,2])/K[0,0]*z;Y=(cy-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=torch.tensor(T_cw,dtype=torch.float32,device='cuda')[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];zz=Vc[...,2:3];uv=Vc[...,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32);ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));pm=rast[0,...,3]>0;ys,xs=np.where(pm.cpu());ty,tx=np.where(m[y0:y1,x0:x1]);print(yaw,tilt,z,'pred',len(xs),(xs.min(),xs.max(),ys.min(),ys.max()),'tgt',len(tx),(tx.min(),tx.max(),ty.min(),ty.max()))
PY"
```
<details><summary>output</summary>

```
75 -12 1.6 pred 4259 (61, 158, 46, 110) tgt 3802 (45, 166, 45, 120)
70 -12 1.7 pred 4153 (59, 158, 49, 108) tgt 3802 (45, 166, 45, 120)
70 -12 2.0 pred 3001 (66, 149, 54, 104) tgt 3802 (45, 166, 45, 120)
70 -12 2.5 pred 1913 (74, 140, 60, 100) tgt 3802 (45, 166, 45, 120)
70 0 2.0 pred 3011 (69, 150, 56, 106) tgt 3802 (45, 166, 45, 120)
70 0 2.5 pred 1937 (76, 141, 61, 101) tgt 3802 (45, 166, 45, 120)
290 0 2.0 pred 2950 (63, 130, 57, 108) tgt 3802 (45, 166, 45, 120)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/pose_search_rgb2.py').read()
s=s.replace(\"depths=np.linspace(1.6,4.0,25); tilts=np.deg2rad([-12,-6,0,6,12])\",\"depths=np.linspace(1.5,3.8,24); tilts=np.deg2rad([-30,-22,-15,-8,0,8,15,22,30])\")
open('/work/scratch/pose_search_rgb3.py','w').write(s)
PY
python /work/scratch/pose_search_rgb3.py"
```
<details><summary>output</summary>

```
elapsed 6.609578371047974
BEST 0.593 0.207 0.796 yaw 75.0 tilt -30.0 z 1.6 361
BEST 0.595 0.211 0.792 yaw 70.0 tilt -30.0 z 1.6 337
BEST 0.592 0.208 0.794 yaw 70.0 tilt -30.0 z 1.7 338
BEST 0.593 0.213 0.788 yaw 75.0 tilt -30.0 z 1.5 360
BEST 0.588 0.211 0.798 yaw 80.0 tilt -30.0 z 1.5 384
BEST 0.591 0.216 0.781 yaw 65.0 tilt -30.0 z 1.7 314
BEST 0.588 0.215 0.783 yaw 65.0 tilt -30.0 z 1.8 315
BEST 0.58 0.204 0.801 yaw 75.0 tilt -22.0 z 1.6 2089
BEST 0.581 0.209 0.797 yaw 80.0 tilt -22.0 z 1.5 2112
BEST 0.58 0.209 0.792 yaw 70.0 tilt -22.0 z 1.6 2065
BEST 0.581 0.21 0.79 yaw 75.0 tilt -22.0 z 1.5 2088
BEST 0.576 0.204 0.798 yaw 70.0 tilt -22.0 z 1.7 2066
BEST 0.576 0.213 0.787 yaw 65.0 tilt -22.0 z 1.7 2042
BEST 0.573 0.207 0.792 yaw 70.0 tilt -30.0 z 1.8 339
BEST 0.572 0.208 0.795 yaw 65. …[+824 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/local_search.py <<'PY'
# reuse definitions from pose_search_rgb3 but stop before poses generation
s=open('/work/scratch/pose_search_rgb3.py').read().split('poses=[];meta=[]')[0]
exec(s)
# candidate orientation/depth list around best including 180 alternatives + local shifts
cands=[]
for yaw in [55,60,65,70,75,80,250,255,260,265]:
 for tilt in [-35,-30,-25,-20,-15,-10,0]:
  for z in np.arange(1.5,2.61,.1): cands.append((yaw,tilt,z))
poses=[];meta=[]
for yaw,tilt,z in cands:
 for du in range(-15,16,3):
  for dv in range(-12,13,3):
   R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@B
   X=(cx+du-K[0,2])/K[0,0]*z;Y=(cy+dv-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];q=np.eye(4);q[:3,:3]=R;q[:3,3]=t;poses.append(q);meta.append((yaw,tilt,z,du,dv))
poses=torch.tensor(np.array(poses),dtype=torch.float32,device='cuda');Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');res=[]
for s in range(0,len(poses),16):
 T=Tcw[None]@poses[s:s+16];Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear');pred=rast[...,3]>0;Tg=tgt.bool()[None];inter=(pr …[+722 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/local_search.py", line 17, in <module>
    T=Tcw[None]@poses[s:s+16];Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear');pred=rast[...,3]>0;Tg=tgt.bool()[None];inter=(pred&Tg).sum((1,2)).float();union=(pred|Tg).sum((1,2)).float();iou=inter/union.clamp_min(1);maski=(pred&Tg)[...,None].float();l1=((color-roi[None]).abs()*maski).sum((1,2,3))/maski.sum((1,2,3)).clamp_min(1)/3;a=(color*roi[None]* …[+1345 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/local_search.py
python /work/scratch/local_search.py"
```
<details><summary>output</summary>

```
0.5986 0.21 0.797 (75, -35, 1.5, 0, 0)
0.6002 0.214 0.792 (70, -35, 1.6, 0, 0)
0.5984 0.211 0.794 (70, -35, 1.6, -3, 0)
0.5969 0.211 0.796 (75, -30, 1.5, -3, 0)
0.5987 0.212 0.786 (70, -30, 1.6, -3, 0)
0.5882 0.2 0.807 (75, -30, 1.6, -3, -3)
0.5916 0.202 0.796 (70, -25, 1.7000000000000002, -3, 0)
0.5918 0.204 0.798 (70, -30, 1.7000000000000002, -3, 0)
0.5954 0.21 0.796 (70, -30, 1.6, 0, 3)
0.5934 0.207 0.796 (75, -30, 1.6, 0, 0)
0.5935 0.206 0.795 (75, -25, 1.6, -3, 0)
0.5951 0.21 0.791 (70, -25, 1.6, -3, 3)
0.5944 0.212 0.796 (75, -35, 1.5, -3, 0)
0.5947 0.211 0.792 (70, -30, 1.6, 0, 0)
0.5874 0.203 0.808 (80, -10, 1.5, -3, 0)
0.5972 0.214 0.785 (65, -30, 1.7000000000000002, -3, 0)
0.5923 0.208 0.794 (70, -30, 1.7000000000000002, 0, 0)
0.5955 0.214 0.793 (70, -35, 1.6, 0, 3)
0.5879 0.203  …[+69 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
m=cv2.imread('/work/scratch/iron_masks/000000.png',0);ys,xs=np.where(m>0);x0=xs.min()-20;y0=ys.min()-20;x1=xs.max()+20;y1=ys.max()+20;cv2.imwrite('/work/scratch/targetmask0_clean.png',m[y0:y1,x0:x1])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
im=cv2.imread('/work/scratch/frame0_obj.png');m=cv2.imread('/work/scratch/iron_masks/000000.png',0)>0
# frame crop origin x619,y418
roi=m[418:688,619:826];ov=im.copy();ov[roi]=(ov[roi]*.4+np.array([0,0,255])*.6).astype(np.uint8);cv2.imwrite('/work/scratch/frame0_cleanoverlay.png',ov)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/sam2_points.json <<'EOF'
{
 \"prompts\": [
  {\"frame_index\": 0, \"object_id\": 0,
   \"points\": [{\"x\":720.0,\"y\":600.0},{\"x\":740.0,\"y\":500.0},{\"x\":700.0,\"y\":540.0},{\"x\":760.0,\"y\":540.0}],
   \"point_labels\": [1,0,0,0], \"box\": null, \"mask_path\": null}
 ]
}
EOF
export CONFIG_FILE=configs/sam2.1/sam2.1_hiera_l.yaml
export CHECKPOINT_FILE=sam2.1_hiera_large.pt
python -m v2d.sam2.lib.video_to_masks --video_path /work/inputs/video.mp4 --prompts_path /work/scratch/sam2_points.json --masks_dir /work/scratch/sam2_iron2 --weights_dir /weights/sam2"
```
<details><summary>output</summary>

```
/workspace/v2d_sam2/lib/sam2_utils.py:1058: UserWarning: cannot import name '_C' from 'sam2' (/opt/conda/lib/python3.11/site-packages/sam2/__init__.py)

Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
  pred_masks_gpu = fill_holes_in_mask_scores(
propagate in video: 0it [00:00, ?it/s]propagate in video: 0it [00:00, ?it/s]
propagate in video:   0%|          | 0/896 [00:00<?, ?it/s]propagate in video:   0%|          | 4/896 [00:00<00:24, 35.81it/s]propagate in video:   1%|          | 8/896 [00:00<00:31, 28.37it/s]propagate in video:   1%|    …[+20942 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
m=cv2.imread('/work/scratch/sam2_iron2/0/000000.png',0)>0;ys,xs=np.where(m);print(m.sum(),xs.min(),xs.max(),ys.min(),ys.max())
im=cv2.imread('/work/scratch/frame0_obj.png');roi=m[418:688,619:826];ov=im.copy();ov[roi]=(ov[roi]*.4+np.array([0,0,255])*.6).astype(np.uint8);cv2.imwrite('/work/scratch/frame0_pts2overlay.png',ov)
PY"
```
<details><summary>output</summary>

```
5509 446 765 470 820

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/grabcut0.py <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');_,im=cap.read()
mask=np.zeros(im.shape[:2],np.uint8);bgdModel=np.zeros((1,65),np.float64);fgdModel=np.zeros((1,65),np.float64)
rect=(668,548,110,90)
# seed inside rect: definite fg at body, definite bg at upper corners/tripod
mask[:]=cv2.GC_BGD
mask[548:638,668:778]=cv2.GC_PR_BGD
mask[575:633,676:772]=cv2.GC_PR_FGD
for p in [(700,600),(730,610),(750,610),(690,585),(745,590)]: mask[p[1],p[0]]=cv2.GC_FGD
for p in [(730,480),(700,510),(755,520),(710,555),(765,550),(680,560),(770,575),(700,635),(760,635)]: mask[p[1],p[0]]=cv2.GC_BGD
cv2.grabCut(im,mask,rect,bgdModel,fgdModel,8,cv2.GC_INIT_WITH_MASK)
out=np.isin(mask,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
# keep components near expected
n,lab,stats,c=cv2.connectedComponentsWithStats(out,8);areas=stats[1:,4];keep=np.where(areas>max(15,areas.max()*.02))[0]+1;out=np.isin(lab,keep).astype(np.uint8)
cv2.imwrite('/work/scratch/grabcut0.png',out*255)
ov=im.copy();mm=out>0;ov[mm]=(ov[mm]*.4+np.array([0,0,255])*.6).astype(np.uint8);cv2.imwrite('/work/scratch/grabcut0_overlay.png',ov[418:688,619:826])
PY
python /work/scratch/grabcut0.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2
im=cv2.imread('/work/scratch/frame0_obj.png');# crop x619,y418 -> full coords
# upscale 3x, draw grid every10px full => every30
out=cv2.resize(im,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
for X in range(620,826,10):
 x=(X-619)*3;cv2.line(out,(x,0),(x,out.shape[0]),(0,255,0),1)
 if X%20==0: cv2.putText(out,str(X),(x+1,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
for Y in range(420,688,10):
 y=(Y-418)*3;cv2.line(out,(0,y),(out.shape[1],y),(0,255,0),1)
 if Y%20==0: cv2.putText(out,str(Y),(1,y+12),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
cv2.imwrite('/work/scratch/frame0_grid.png',out)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/sam2_points3.json <<'EOF'
{\"prompts\":[{\"frame_index\":0,\"object_id\":0,
\"points\":[
{\"x\":700,\"y\":600},{\"x\":730,\"y\":610},{\"x\":755,\"y\":610},{\"x\":690,\"y\":585},
{\"x\":730,\"y\":480},{\"x\":700,\"y\":510},{\"x\":755,\"y\":520},{\"x\":710,\"y\":555},{\"x\":765,\"y\":550},{\"x\":780,\"y\":580},{\"x\":745,\"y\":565}],
\"point_labels\":[1,1,1,1,0,0,0,0,0,0,0],\"box\":null,\"mask_path\":null}]}
EOF
export CONFIG_FILE=configs/sam2.1/sam2.1_hiera_l.yaml
export CHECKPOINT_FILE=sam2.1_hiera_large.pt
python -m v2d.sam2.lib.video_to_masks --video_path /work/inputs/video.mp4 --prompts_path /work/scratch/sam2_points3.json --masks_dir /work/scratch/sam2_iron3 --weights_dir /weights/sam2"
```
<details><summary>output</summary>

```
/workspace/v2d_sam2/lib/sam2_utils.py:1058: UserWarning: cannot import name '_C' from 'sam2' (/opt/conda/lib/python3.11/site-packages/sam2/__init__.py)

Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
  pred_masks_gpu = fill_holes_in_mask_scores(
propagate in video: 0it [00:00, ?it/s]propagate in video: 0it [00:00, ?it/s]
propagate in video:   0%|          | 0/896 [00:00<?, ?it/s]propagate in video:   0%|          | 4/896 [00:00<00:27, 32.28it/s]propagate in video:   1%|          | 8/896 [00:00<00:34, 26.06it/s]propagate in video:   1%|    …[+25282 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/sam2_ironbox.json <<'EOF'
{\"prompts\":[{\"frame_index\":0,\"object_id\":0,\"points\":null,\"point_labels\":null,
\"box\":{\"x0\":665.0,\"y0\":610.0,\"x1\":780.0,\"y1\":675.0},\"mask_path\":null}]}
EOF
export CONFIG_FILE=configs/sam2.1/sam2.1_hiera_l.yaml
export CHECKPOINT_FILE=sam2.1_hiera_large.pt
python -m v2d.sam2.lib.video_to_masks --video_path /work/inputs/video.mp4 --prompts_path /work/scratch/sam2_ironbox.json --masks_dir /work/scratch/sam2_iron4 --weights_dir /weights/sam2"
```
<details><summary>output</summary>

```
/workspace/v2d_sam2/lib/sam2_utils.py:1058: UserWarning: cannot import name '_C' from 'sam2' (/opt/conda/lib/python3.11/site-packages/sam2/__init__.py)

Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
  pred_masks_gpu = fill_holes_in_mask_scores(
propagate in video: 0it [00:00, ?it/s]propagate in video: 0it [00:00, ?it/s]
propagate in video:   0%|          | 0/896 [00:00<?, ?it/s]propagate in video:   0%|          | 2/896 [00:00<00:45, 19.73it/s]propagate in video:   0%|          | 4/896 [00:00<01:05, 13.61it/s]propagate in video:   1%|    …[+26103 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
m=cv2.imread('/work/scratch/sam2_iron4/0/000000.png',0)>0;ys,xs=np.where(m);print(m.sum(),xs.min(),xs.max(),ys.min(),ys.max())
im=cv2.imread('/work/scratch/frame0_obj.png');roi=m[418:688,619:826];ov=im.copy();ov[roi]=(ov[roi]*.4+np.array([0,0,255])*.6).astype(np.uint8);cv2.imwrite('/work/scratch/frame0_boxoverlay.png',ov)
PY"
```
<details><summary>output</summary>

```
2466 685 754 613 657

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/pose_search_rgb3.py').read();s=s.replace(\"f'/work/scratch/iron_masks/{frame:06d}.png'\",\"f'/work/scratch/sam2_iron4/0/{frame:06d}.png'\")
s=s.replace('depths=np.linspace(1.5,3.8,24)','depths=np.linspace(2.5,5.0,26)')
open('/work/scratch/pose_search_iron4.py','w').write(s)
PY
python /work/scratch/pose_search_iron4.py"
```
<details><summary>output</summary>

```
elapsed 4.8127760887146
BEST 0.854 0.326 0.691 yaw 230.0 tilt -30.0 z 2.8 1199
BEST 0.853 0.327 0.693 yaw 235.0 tilt -30.0 z 2.7 1224
BEST 0.854 0.33 0.694 yaw 235.0 tilt -30.0 z 2.8 1225
BEST 0.852 0.331 0.694 yaw 240.0 tilt -30.0 z 2.7 1250
BEST 0.787 0.238 0.825 yaw 55.0 tilt 8.0 z 2.5 9646
BEST 0.849 0.329 0.692 yaw 240.0 tilt -30.0 z 2.6 1249
BEST 0.782 0.235 0.827 yaw 50.0 tilt 8.0 z 2.6 9621
BEST 0.841 0.318 0.701 yaw 230.0 tilt -30.0 z 2.7 1198
BEST 0.85 0.333 0.685 yaw 245.0 tilt -30.0 z 2.6 1275
BEST 0.846 0.329 0.691 yaw 230.0 tilt -30.0 z 2.9 1200
BEST 0.841 0.33 0.7 yaw 235.0 tilt -22.0 z 2.7 3096
BEST 0.84 0.325 0.692 yaw 235.0 tilt -30.0 z 2.6 1223
BEST 0.775 0.233 0.829 yaw 50.0 tilt 8.0 z 2.5 9620
BEST 0.837 0.326 0.703 yaw 230.0 tilt -22.0 z 2.8 3071
BEST 0.84 0.332 0.703 …[+832 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/search_refine.py <<'PY'
# definitions
s=open('/work/scratch/pose_search_rgb3.py').read().split('poses=[];meta=[]')[0]
s=s.replace(\"f'/work/scratch/iron_masks/{frame:06d}.png'\",\"f'/work/scratch/sam2_iron4/0/{frame:06d}.png'\")
exec(s)
# crop is based on correct mask; target
tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.float32,device='cuda')
# target bbox center within crop
yy,xx=np.where(m[y0:y1,x0:x1]);tuc=(xx.min()+xx.max())/2+x0;tvc=(yy.min()+yy.max())/2+y0
cands=[];rots=[]
B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])])
for yaw in np.arange(45,76,3):
 for tilt in np.arange(-15,21,5):
  for roll in np.arange(-20,21,10):
   for z in np.arange(2.2,3.01,.1):
    R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@Rot.from_euler('z',np.deg2rad(roll)).as_matrix()@B
    cands.append((yaw,tilt,roll,z));rots.append(R)
rots=np.array(rots,np.float32);print('cands',len(cands))
Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext(); allres=[]
def render_batch(Rs,ts):
 poses=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32)
 T=Tcw[None]@torch.tensor(poses,device='cuda')
 Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/ …[+1718 chars]
```
<details><summary>output</summary>

```
cands 3960
Traceback (most recent call last):
  File "/work/scratch/search_refine.py", line 27, in <module>
    pred,color=render_batch(Rs,np.array(ts));# center shift by bbox means
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/search_refine.py", line 22, in render_batch
    Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear');pred=rast[...,3]>0;return pred,color
                                                                  …[+1120 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/search_refine.py
python /work/scratch/search_refine.py"
```
<details><summary>output</summary>

```
cands 3960
BEST 0.784 0.241 0.823 (60, 5, 0, 2.3000000000000003)
BEST 0.7815 0.239 0.827 (57, 10, 0, 2.4000000000000004)
BEST 0.7889 0.25 0.807 (57, 0, -10, 2.4000000000000004)
BEST 0.7888 0.25 0.804 (60, 0, -10, 2.4000000000000004)
BEST 0.7805 0.24 0.821 (63, 5, 0, 2.2)
BEST 0.7796 0.24 0.82 (60, 10, 0, 2.3000000000000003)
BEST 0.7798 0.241 0.819 (57, 10, 0, 2.3000000000000003)
BEST 0.7847 0.249 0.809 (60, 0, -10, 2.3000000000000003)
BEST 0.7784 0.242 0.822 (57, 5, 0, 2.4000000000000004)
BEST 0.7858 0.252 0.801 (63, 0, -10, 2.3000000000000003)
BEST 0.778 0.243 0.821 (57, 5, 0, 2.3000000000000003)
BEST 0.7748 0.239 0.825 (54, 10, 0, 2.4000000000000004)
BEST 0.7742 0.239 0.825 (63, 5, 0, 2.3000000000000003)
BEST 0.7775 0.243 0.811 (60, 5, -10, 2.3000000000000003)
BEST 0.779 0.249 0.807 (54, …[+837 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/sheet_iron4.py <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');frames=[]
while True:
 ok,im=cap.read()
 if not ok:break
 frames.append(im)
for s,e in [(0,224),(224,448),(448,672),(672,896)]:
 ids=list(range(s,e,16));tw,th=256,192;cols=7;rows=4;out=np.ones((rows*th,cols*tw,3),np.uint8)*30
 for j,i in enumerate(ids):
  im=frames[i].copy();m=cv2.imread(f'/work/scratch/sam2_iron4/0/{i:06d}.png',0)>0;im[m]=(im[m]*.3+np.array([0,0,255])*.7).astype(np.uint8)
  ys,xs=np.where(m)
  if len(xs):
   cx,cy=int(xs.mean()),int(ys.mean());w=max(240,int((xs.max()-xs.min())*2.4));h=max(180,int((ys.max()-ys.min())*2.4));x0=max(0,min(1536-w,cx-w//2));y0=max(0,min(1152-h,cy-h//2));crop=im[y0:y0+h,x0:x0+w]
  else:crop=im
  im=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_AREA);cv2.putText(im,str(i),(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);out[(j//cols)*th:(j//cols+1)*th,(j%cols)*tw:(j%cols+1)*tw]=im
 cv2.imwrite(f'/work/scratch/sheets/iron4_{s}.jpg',out)
PY
python /work/scratch/sheet_iron4.py
python - <<'PY'
import cv2,numpy as np
for i in range(0,896,32):
 m=cv2.imread(f'/work/scratch/sam2_iron4/0/{i:06d}.png',0)>0;ys,xs=np.where(m);print(i,m.sum(),(xs.min(),xs.max(),ys.min(),ys.max()) if len(xs) else None)
PY"
```
<details><summary>output</summary>

```
0 2466 (685, 754, 613, 657)
32 2473 (685, 754, 613, 656)
64 2424 (685, 753, 613, 656)
96 2431 (685, 753, 613, 656)
128 2207 (685, 754, 617, 656)
160 2128 (689, 758, 616, 655)
192 1821 (727, 791, 611, 647)
224 1699 (745, 806, 609, 643)
256 1720 (741, 801, 608, 644)
288 1919 (715, 776, 612, 650)
320 2166 (685, 752, 614, 655)
352 2363 (648, 724, 619, 659)
384 2562 (615, 697, 623, 665)
416 2426 (630, 708, 624, 663)
448 2204 (666, 736, 618, 658)
480 1893 (699, 762, 616, 653)
512 1907 (719, 781, 611, 649)
544 1834 (736, 795, 609, 646)
576 1696 (747, 806, 608, 644)
608 1707 (736, 793, 610, 645)
640 1737 (716, 777, 612, 647)
672 2053 (718, 784, 610, 649)
704 2032 (716, 776, 600, 650)
736 2587 (704, 760, 575, 652)
768 2898 (683, 734, 569, 654)
800 3346 (681, 732, 569, 656)
832 3319 (682, 732, 569,  …[+35 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/opt_static.py').read()
s=s.replace(\"frames=[0,30,60,90]\",\"frames=[0,32,64,96]\")
s=s.replace(\"f'/work/scratch/sam2_iron/0/{i:06d}.png'\",\"f'/work/scratch/sam2_iron4/0/{i:06d}.png'\")
s=s.replace(\"cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2; z0=1.6;yaw=np.deg2rad(55)\",\"cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2; z0=2.5;yaw=np.deg2rad(55);tilt=np.deg2rad(8)\")
s=s.replace(\"R0=Rot.from_euler('y',yaw).as_matrix()@B\",\"R0=Rot.from_euler('y',yaw).as_matrix()@Rot.from_euler('x',tilt).as_matrix()@B\")
s=s.replace(\"lr=0.01\",\"lr=0.003\").replace(\"range(301)\",\"range(181)\").replace(\"it%10==0 or it==300\",\"it%10==0 or it==180\")
s=s.replace(\"sigma=1e-2,gamma=1e-4\",\"sigma=1e-3,gamma=1e-4\")
open('/work/scratch/opt_static_iron.py','w').write(s)
PY
python /work/scratch/opt_static_iron.py"
```
<details><summary>output</summary>

```
0 loss 9.089622497558594 mask 7.623661518096924 photo 0.9773075580596924 rv [0. 0. 0.] dt [ 0.003  0.003 -0.003]
10 loss 9.149176597595215 mask 7.580507278442383 photo 1.0457446575164795 rv [0. 0. 0.] dt [0.00654176 0.00722517 0.01445085]
20 loss 10.327157974243164 mask 8.846452713012695 photo 0.986966073513031 rv [0. 0. 0.] dt [1.26035475e-05 6.41000876e-03 3.70480604e-02]
30 loss 8.993600845336914 mask 7.484482288360596 photo 1.005677342414856 rv [0. 0. 0.] dt [0.0076253  0.00929214 0.05568093]
40 loss 12.32990550994873 mask 10.772795677185059 photo 1.0373338460922241 rv [0. 0. 0.] dt [0.0039926  0.01909886 0.0735193 ]
50 loss 14.881013870239258 mask 13.232745170593262 photo 1.0978633165359497 rv [0. 0. 0.] dt [0.01493204 0.02705775 0.08022608]
60 loss 24.321361541748047 mask 22.78530311 …[+1877 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n \"pos=torch.cat\\|icolor=\" /work/scratch/opt_static_iron.py; grep -n \"interpolate\" /work/scratch/opt_static_iron.py"
```
<details><summary>output</summary>

```
52: ndc=torch.stack([2*uv[:,0]/W-1,2*uv[:,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1)[None]
53: rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];pm=rast[0,...,3]>0
53: rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];pm=rast[0,...,3]>0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/opt_photo.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,time
from scipy.spatial.transform import Rotation as Rot
np.random.seed(0);torch.manual_seed(0)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);T_cw=np.linalg.inv(T_wc)
frame=0;cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,frame);_,img=cap.read();m=cv2.imread(f'/work/scratch/sam2_iron4/0/{frame:06d}.png',0)>0;ys,xs=np.where(m);pad=50;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;roi=torch.tensor(cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255;tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.bool,device='cuda')
cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;z0=2.5;yaw=np.deg2rad(55);tilt=np.deg2rad(8);B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])]);R0=Rot.from_euler('y',yaw).as_matrix()@Rot.from_euler('x',tilt). …[+2635 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_photo.py", line 19, in <module>
    R=R0t@rod(rv);t=t0t+dt;c,p=render(R,t);print('INIT',metrics(c,p),t.detach().cpu().numpy())
                               ^^^^^^^^^^^
  File "/work/scratch/opt_photo.py", line 14, in render
    Tm=torch.eye(4,device='cuda');Tm[:3,:3]=R;Tm[:3,3]=t;Tc=Tcw@Tm;Vc=V@Tc[:3,:3].T+Tc[:3,3];z=Vc[:,2:3];uv=Vc[:,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[:,0]/W-1,2*uv[:,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1)[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];return color,rast[0,...,3]>0
                                          …[+1179 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/opt_photo.py
python /work/scratch/opt_photo.py"
```
<details><summary>output</summary>

```
INIT (0.7873730043541364, 0.23780793217113164, 0.8245350195235495) [0.013384  0.3130814 2.4492905]
0 0.3255404233932495 (0.7873730043541364, 0.23780793217113164, 0.8245350195235495) rv [0. 0. 0.] dt [ 0.0002  0.0002 -0.0002]
20 0.3232557475566864 (0.7922266618234654, 0.23607366722401632, 0.8256358773872035) rv [0. 0. 0.] dt [ 0.00090653  0.0004571  -0.00084995]
40 0.3231416940689087 (0.7927272727272727, 0.2360107794817835, 0.8257381206211777) rv [0. 0. 0.] dt [ 0.00098285  0.00056148 -0.00010441]
60 0.3246957063674927 (0.7909255898366606, 0.2367882461967587, 0.8241850469690316) rv [0. 0. 0.] dt [0.0009849  0.00080979 0.00065258]
80 0.3237733244895935 (0.7927272727272727, 0.2362460354784983, 0.8249453724255953) rv [0. 0. 0.] dt [0.00104613 0.00072436 0.00145404]
100 0.3235776126384735 (0.79 …[+889 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '1,180p' /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/blending.py; grep -n \"class SoftSilhouetteShader\" -A100 /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/mesh/shader.py"
```
<details><summary>output</summary>

```
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import NamedTuple, Sequence, Union

import torch
from pytorch3d import _C
from pytorch3d.common.datatypes import Device

# Example functions for blending the top K colors per pixel using the outputs
# from rasterization.
# NOTE: All blending function should return an RGBA image per batch element


class BlendParams(NamedTuple):
    """
    Data class to store blending params with defaults

    Members:
        sigma (float): For SoftmaxPhong, controls the width of the sigmoid
            function used to calculate the 2D distance based probability. Determine …[+10590 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -n '120,210p' /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/blending.py"
```
<details><summary>output</summary>

```
      - **A** - blend based on the 2D distance based probability map [1].

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [1] Liu et al, 'Soft Rasterizer: A Differentiable Renderer for Image-based
        3D Reasoning', ICCV 2019
    """
    N, H, …[+3358 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/search_upright.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,time
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);T_cw=np.linalg.inv(T_wc);frame=800
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,frame);_,img=cap.read();m=cv2.imread(f'/work/scratch/sam2_iron4/0/{frame:06d}.png',0)>0;ys,xs=np.where(m);pad=35;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;roi=torch.tensor(cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255;tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.bool,device='cuda');yy,xx=np.where(m[y0:y1,x0:x1]);tuc=(xx.min()+xx.max())/2+x0;tvc=(yy.min()+yy.max())/2+y0;print('crop',x0,y0,W,H,'center',tuc,tvc)
B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])]);Rx90=Rot.from_euler('x',-90,degrees=True).as_matrix();cands=[ …[+3144 chars]
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
cands 7350
Traceback (most recent call last):
  File "/work/scratch/search_upright.py", line 19, in <module>
    pred,color=render_batch(Rs,np.array(ts));ts2=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/search_upright.py", line 14, in render_batch
    poses=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(poses,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.in …[+1442 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/search_upright.py
python /work/scratch/search_upright.py"
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
cands 7350
BEST 0.8513 0.245 0.741 (45, 21, 10, 2.6)
BEST 0.8516 0.248 0.735 (45, 21, 10, 2.5)
BEST 0.8493 0.248 0.737 (48, 21, 10, 2.6)
BEST 0.8499 0.249 0.734 (48, 21, 10, 2.5)
BEST 0.8439 0.242 0.743 (42, 21, 10, 2.5)
BEST 0.8419 0.24 0.745 (42, 21, 10, 2.6)
BEST 0.8434 0.248 0.742 (51, 21, 10, 2.6)
BEST 0.8396 0.241 0.742 (39, 21, 10, 2.6)
BEST 0.8355 0.244 0.752 (54, 21, 20, 2.5)
BEST 0.8382 0.247 0.746 (51, 21, 10, 2.5)
BEST 0.8376 0.244 0.735 (39, 21, 10, 2.5)
BEST 0.8338 0.247 0.748 (54, 21, 20, 2.6)
BEST 0.8331 0.25 0.751 (60, 21, 20, 2.5)
BEST 0.8312 0.244 0.748 (48, 15, 10, 2.6)
BEST 0.8327 0.246 0.746 (51, 21, 20, 2.6)
BEST 0.8315 0.244 0.745 (45, 15, 10, 2.6)
BEST 0.8331 0.248 0.746 (57, 21, 20, 2.5)
BEST 0.8309 0.245 0.746 (51, 21, 20,  …[+507 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/search_upright.py').read()
# replace R construction and loops
old=\"\"\"for yaw in np.arange(30,91,3):\\n for tilt in np.arange(-15,22,6):\\n  for roll in np.arange(-20,21,10):\\n   for z in np.arange(2.4,3.31,.1):\\n    R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@Rot.from_euler('z',np.deg2rad(roll)).as_matrix()@B@Rx90;cands.append((yaw,tilt,roll,z));rots.append(R)\"\"\"
new=\"\"\"Bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])])\\nfor yaw in np.arange(0,360,5):\\n for tilt in np.arange(-20,21,10):\\n  for roll in np.arange(-20,21,10):\\n   for z in np.arange(2.4,3.31,.1):\\n    R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@Rot.from_euler('z',np.deg2rad(roll)).as_matrix()@Bu;cands.append((yaw,tilt,roll,z));rots.append(R)\"\"\"
assert old in s
s=s.replace(old,new)
open('/work/scratch/search_upright2.py','w').write(s)
PY
python /work/scratch/search_upright2.py"
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
cands 18000
BEST 0.8362 0.218 0.748 (225, -20, -10, 2.7)
BEST 0.849 0.248 0.739 (315, 10, -20, 2.6)
BEST 0.8475 0.249 0.736 (315, 10, -20, 2.5)
BEST 0.8424 0.243 0.746 (310, 10, -20, 2.6)
BEST 0.8347 0.223 0.739 (225, -20, -10, 2.6)
BEST 0.8401 0.242 0.746 (310, 10, -20, 2.5)
BEST 0.8235 0.216 0.755 (230, -20, -10, 2.6)
BEST 0.8367 0.233 0.722 (210, -20, 0, 2.7)
BEST 0.8343 0.23 0.725 (215, -20, 0, 2.6)
BEST 0.8268 0.22 0.737 (220, -20, -10, 2.7)
BEST 0.8221 0.218 0.751 (230, -20, -10, 2.5)
BEST 0.8363 0.232 0.716 (210, -20, 0, 2.6)
BEST 0.8119 0.207 0.773 (240, -20, -20, 2.6)
BEST 0.8277 0.224 0.735 (220, -10, 0, 2.6)
BEST 0.8269 0.225 0.736 (220, -10, 0, 2.7)
BEST 0.8332 0.243 0.738 (305, 10, -20, 2.6)
BEST 0.8303 0.243 0.743 (310, 10, -20, 2.7)
BE …[+612 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_up_candidates.py <<'PY'
# load common definitions by executing search_upright2 before loop
s=open('/work/scratch/search_upright2.py').read().split('for yaw in np.arange')[0]
exec(s)
# render helper
Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext()
def evalR(R,z,name,du=0,dv=0):
 X=(tuc+du-K[0,2])/K[0,0]*z;Y=(tvc+dv-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];Tm=torch.eye(4,device='cuda');Tm[:3,:3]=torch.tensor(R,dtype=torch.float32,device='cuda');Tm[:3,3]=torch.tensor(t,dtype=torch.float32,device='cuda');Tc=Tcw@Tm;Vc=V@Tc[:3,:3].T+Tc[:3,3];zz=Vc[:,2:3];uv=Vc[:,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[:,0]/W-1,2*uv[:,1]/H-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1)[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];pm=rast[0,...,3]>0;inter=(pm&tgt).sum().item();union=(pm|tgt).sum().item();mi=(pm&tgt)[...,None].float();l1=((color-roi).abs()*mi).sum().item()/(mi.sum().item()*3+1e-6);ncc=(color*roi*mi).sum().item()/((color**2*mi).sum().sqrt()*(roi**2*mi).sum().sqrt()).item();out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8);p=pm.cpu().numpy();out[~p]=0;act=cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB);ov=act.copy();ov[p]=(.7*out[p]+.3*ov[p]).astype(np.uint8);cv2.imwrite(f'/work/scratch/up_{name}.png' …[+800 chars]
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
Traceback (most recent call last):
  File "/work/scratch/eval_up_candidates.py", line 11, in <module>
    evalR(Rflat@Rot.from_euler('x',th,degrees=True).as_matrix(),2.7,f'flat_rx{th}')
  File "/work/scratch/eval_up_candidates.py", line 7, in evalR
    X=(tuc+du-K[0,2])/K[0,0]*z;Y=(tvc+dv-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];Tm=torch.eye(4,device='cuda');Tm[:3,:3]=torch.tensor(R,dtype=torch.float32,device='cuda');Tm[:3,3]=torch.tensor(t,dtype=torch.float32,device='cuda');Tc=Tcw@Tm;Vc=V@Tc[:3,:3].T+Tc[:3,3];zz=Vc[:,2:3];uv=Vc[:,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[:,0]/W-1,2*uv[:,1]/H-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1)[None];rast, …[+2148 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/device='cuda')+torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/\" /work/scratch/eval_up_candidates.py
python /work/scratch/eval_up_candidates.py"
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
flat_rx-90 0.775644541963796 0.2650090657469933 0.7141925757280381
flat_rx90 0.7580557072637902 0.22503206248517416 0.7217091002985293
y225t-20r-10 0.7869674185463659 0.21969353236122097 0.7411313546747178
y240t-20r-20 0.7848918843021623 0.2077813163605061 0.7715777025327536
y315t10r-20 0.8217117856166256 0.25664414557954585 0.7184420389482155
y55t8r0 0.7427184466019418 0.2466926657533032 0.7099212319442603
y45t21r10 0.7933188484519283 0.24233382275706564 0.7447007300212358

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,800);_,im=cap.read();cv2.imwrite('/work/scratch/frame800_obj.png',im[520:700,630:780])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/search_upright2.py').read()
s=s.replace('frame=800','frame=895')
s=s.replace('for tilt in np.arange(-20,21,10):','for tilt in np.arange(-30,31,10):')
s=s.replace('for roll in np.arange(-20,21,10):','for roll in np.arange(-30,31,10):')
s=s.replace('for z in np.arange(2.4,3.31,.1):','for z in np.arange(2.4,3.11,.1):')
s=s.replace('/work/scratch/upright_top.png','/work/scratch/upright895_top.png').replace('/work/scratch/upright_best.npy','/work/scratch/upright895_best.npy')
open('/work/scratch/search_upright895.py','w').write(s)
PY
python /work/scratch/search_upright895.py"
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
cands 28224
BEST 0.8775 0.249 0.735 (310, 10, -30, 2.5)
BEST 0.8772 0.247 0.733 (305, 10, -30, 2.5)
BEST 0.8743 0.248 0.731 (305, 10, -30, 2.4)
BEST 0.86 0.237 0.747 (300, 10, -30, 2.4)
BEST 0.8631 0.241 0.739 (300, 10, -30, 2.5)
BEST 0.8509 0.22 0.74 (220, -30, -10, 2.5)
BEST 0.8605 0.249 0.744 (310, 20, -30, 2.4)
BEST 0.8599 0.252 0.745 (315, 20, -30, 2.4)
BEST 0.8474 0.221 0.74 (220, -30, -10, 2.6)
BEST 0.8581 0.249 0.735 (310, 10, -30, 2.6)
BEST 0.8578 0.247 0.732 (305, 10, -30, 2.6)
BEST 0.8434 0.22 0.747 (225, -30, -10, 2.5)
BEST 0.8595 0.239 0.711 (205, -30, 0, 2.6)
BEST 0.8558 0.235 0.718 (210, -30, 0, 2.6)
BEST 0.8539 0.249 0.744 (315, 10, -20, 2.6)
BEST 0.8408 0.22 0.747 (225, -20, -10, 2.7)
BEST 0.8561 0.252 0.73 (310, 10, -30, 2.4)
BEST 0 …[+561 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,895);_,im=cap.read();cv2.imwrite('/work/scratch/frame895_obj.png',im[520:710,630:790])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/sift_eval.py <<'PY'
# exec common definitions up to evaluation
s=open('/work/scratch/eval_up_candidates.py').read().split('B=np.column_stack')[0]
exec(s)
sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.005,edgeThreshold=10)
def ev(R,z,name):
 X=(tuc-K[0,2])/K[0,0]*z;Y=(tvc-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];Tm=torch.eye(4,device='cuda');Tm[:3,:3]=torch.tensor(R,dtype=torch.float32,device='cuda');Tm[:3,3]=torch.tensor(t,dtype=torch.float32,device='cuda');Tc=Tcw@Tm;Vc=V@Tc[:3,:3].T+Tc[:3,3];zz=Vc[:,2:3];uv=Vc[:,:2]/zz*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[:,0]/W-1,2*uv[:,1]/H-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1)[None];rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];pm=rast[0,...,3]>0;out=(color.clamp(0,1).cpu().numpy()*255).astype(np.uint8);p=pm.cpu().numpy();out[~p]=0
 act=cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
 # Mask background out of actual
 actm=act.copy(); actm[~tgt.cpu().numpy()]=0
 # use CLAHE gray
 g1=cv2.cvtColor(actm,cv2.COLOR_RGB2GRAY);g2=cv2.cvtColor(out,cv2.COLOR_RGB2GRAY)
 g1=cv2.resize(g1,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC);g2=cv2.resize(g2,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
 k1,d1=sift.detectAndCompute(g1,None);k2,d2=sift.detectAndCompute(g2,None);good=[]
 if d1 is not None and d2 is no …[+817 chars]
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
Traceback (most recent call last):
  File "/work/scratch/sift_eval.py", line 22, in <module>
    R=Rot.from_euler('y',yaw,degrees=True).as_matrix()@Rot.from_euler('x',tilt,degrees=True).as_matrix()@Rot.from_euler('z',roll,degrees=True).as_matrix()@Bu;ev(R,z,f'y{yaw}t{tilt}r{roll}')
                                                                                                                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/sift_eval.py", line 6, in ev
    X=(tuc-K[0,2])/K[0,0]*z;Y=(tvc-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];Tm=torch.eye(4,device='cuda');Tm[:3,:3]=torch.tensor(R,dtype=torch.float32,device='cuda');Tm[:3,3]=torch.tensor(t,dtype=torch.float32,device= …[+1919 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/device='cuda')+torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/\" /work/scratch/sift_eval.py
python /work/scratch/sift_eval.py"
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
y225t-20r-10 kp 406 904 good 27 mean 213.83035956488715
y240t-20r-20 kp 406 875 good 18 mean 243.9660390218099
y235t-30r-20 kp 406 947 good 27 mean 210.0026513558847
y310t10r-30 kp 406 891 good 35 mean 203.70756988525392
y305t10r-30 kp 406 886 good 26 mean 205.69241450383112
y315t20r-30 kp 406 910 good 42 mean 229.24542872111002
y55t8r0 kp 406 812 good 35 mean 218.5133298601423
y230t-30r-20 kp 406 934 good 23 mean 222.21417667554772

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/sift_eval.py';s=open(p).read();s=s.replace(\"good=[m for m,n in mm if m.distance<.75*n.distance]\\n print(name,'kp',len(k1) if k1 else 0,len(k2) if k2 else 0,'good',len(good),'mean',np.mean([m.distance for m in good]) if good else 0)\",\"good=[m for m,n in mm if m.distance<.78*n.distance]\\n inl=0; med=999\\n if len(good)>=3:\\n  src=np.float32([k2[m.queryIdx].pt for m in good]);dst=np.float32([k1[m.trainIdx].pt for m in good]);M,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=2000);inl=int(inliers.sum()) if inliers is not None else 0; med=float(np.median(np.linalg.norm(dst-src,axis=1)))\\n print(name,'kp',len(k1) if k1 else 0,len(k2) if k2 else 0,'good',len(good),'inl',inl,'mean',np.mean([m.distance for m in good]) if good else 0)\")
open('/work/scratch/sift_eval2.py','w').write(s)
PY
python /work/scratch/sift_eval2.py"
```
<details><summary>output</summary>

```
crop 646 534 122 158 center 706.5 612.5
y225t-20r-10 kp 406 904 good 33 inl 6 mean 224.78193294640744
y240t-20r-20 kp 406 875 good 31 inl 5 mean 239.91199468797254
y235t-30r-20 kp 406 947 good 39 inl 5 mean 229.74915235470502
y310t10r-30 kp 406 891 good 46 inl 21 mean 216.8103732233462
y305t10r-30 kp 406 886 good 42 inl 10 mean 210.0467067900158
y315t20r-30 kp 406 910 good 53 inl 30 mean 230.43422727764778
y55t8r0 kp 406 812 good 43 inl 13 mean 216.59239090320676
y230t-30r-20 kp 406 934 good 29 inl 3 mean 227.5714692740605

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/fine_flat.py <<'PY'
# common
s=open('/work/scratch/pose_search_rgb3.py').read().split('poses=[];meta=[]')[0]
s=s.replace(\"f'/work/scratch/iron_masks/{frame:06d}.png'\",\"f'/work/scratch/sam2_iron4/0/{frame:06d}.png'\")
exec(s);tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.float32,device='cuda');yy,xx=np.where(m[y0:y1,x0:x1]);tuc=(xx.min()+xx.max())/2+x0;tvc=(yy.min()+yy.max())/2+y0
B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])]);cands=[];rots=[]
for yaw in np.arange(51,61,1):
 for tilt in np.arange(3,14,1):
  for roll in np.arange(-8,9,2):
   for z in np.arange(2.3,2.71,.05):
    R=Rot.from_euler('y',np.deg2rad(yaw)).as_matrix()@Rot.from_euler('x',np.deg2rad(tilt)).as_matrix()@Rot.from_euler('z',np.deg2rad(roll)).as_matrix()@B;cands.append((yaw,tilt,roll,z));rots.append(R)
rots=np.array(rots,np.float32);print('cands',len(cands));Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();res=[];poses=[]
def render_batch(Rs,ts):
 PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W …[+1618 chars]
```
<details><summary>output</summary>

```
cands 8910
Traceback (most recent call last):
  File "/work/scratch/fine_flat.py", line 18, in <module>
    pred,color=render_batch(Rs,np.array(ts));ts2=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/fine_flat.py", line 13, in render_batch
    PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filt …[+1330 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/fine_flat.py
python /work/scratch/fine_flat.py"
```
<details><summary>output</summary>

```
cands 8910
BEST 0.7855 0.236 0.825 (51, 3, 0, 2.549999999999999)
BEST 0.7828 0.235 0.828 (51, 4, 0, 2.549999999999999)
BEST 0.7791 0.232 0.834 (51, 5, -2, 2.549999999999999)
BEST 0.7781 0.232 0.836 (51, 4, -2, 2.549999999999999)
BEST 0.7911 0.244 0.816 (51, 5, 2, 2.6999999999999984)
BEST 0.782 0.235 0.826 (51, 6, 0, 2.549999999999999)
BEST 0.7819 0.237 0.828 (52, 7, -2, 2.5999999999999988)
BEST 0.79 0.246 0.816 (51, 8, 0, 2.6999999999999984)
BEST 0.7882 0.243 0.817 (51, 4, 2, 2.6999999999999984)
BEST 0.7834 0.239 0.824 (51, 10, -2, 2.6499999999999986)
BEST 0.7773 0.234 0.832 (52, 3, 0, 2.549999999999999)
BEST 0.7868 0.243 0.818 (51, 3, 2, 2.6999999999999984)
BEST 0.7832 0.24 0.822 (52, 8, -2, 2.5999999999999988)
BEST 0.7807 0.237 0.826 (52, 5, 0, 2.5999999999999988)
BEST 0.7778 0.234 0.83  …[+854 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/fine_flat_sift.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_cw=np.linalg.inv(np.array(cam['T_world_cam']));frame=0;cap=cv2.VideoCapture('/work/inputs/video.mp4');_,img=cap.read();m=cv2.imread(f'/work/scratch/sam2_iron4/0/{frame:06d}.png',0)>0;ys,xs=np.where(m);pad=35;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;roi=torch.tensor(cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255;tgt=m[y0:y1,x0:x1];gl=dr.RasterizeCudaContext();Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');poses=np.load('/work/scratch/fine_flat_poses.npy');meta=np.load('/work/scratch/fine_flat_meta.npy');sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.005,edgeThreshold=10)
def draw(T):
 T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=Tcw[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,: …[+1399 chars]
```
<details><summary>output</summary>

```
0 inl 3 good 21 mask [0.78553253 0.23604934 0.82543355]
1 inl 3 good 26 mask [0.7827661  0.23472241 0.827874  ]
2 inl 3 good 18 mask [0.7791121  0.23245314 0.83381391]
3 inl 3 good 16 mask [0.77806121 0.2319877  0.83554852]
4 inl 3 good 24 mask [0.79114163 0.24436411 0.81574261]
5 inl 3 good 23 mask [0.78202814 0.23528478 0.82591182]
6 inl 3 good 10 mask [0.78191292 0.23684534 0.82779664]
7 inl 3 good 18 mask [0.78997707 0.24570557 0.81593084]
8 inl 3 good 18 mask [0.78816795 0.24323003 0.81690174]
9 inl 3 good 14 mask [0.7834084  0.23903362 0.82386482]
10 inl 6 good 25 mask [0.77733237 0.23390952 0.83190089]
11 inl 4 good 22 mask [0.78677875 0.24320796 0.81764734]
12 inl 3 good 20 mask [0.78322196 0.23974892 0.82246315]
13 inl 2 good 14 mask [0.78066915 0.2374253  0.82641637]
14 inl 3 goo …[+900 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np,json
F=896;st=[]
for i in range(F):
 m=cv2.imread(f'/work/scratch/sam2_iron4/0/{i:06d}.png',0)>0;ys,xs=np.where(m);st.append([xs.mean(),ys.mean(),xs.min(),xs.max(),ys.min(),ys.max(),m.sum()])
st=np.array(st);np.save('/work/scratch/maskstats4.npy',st)
for i in range(0,F,10): print(i,st[i].round(1))
PY"
```
<details><summary>output</summary>

```
0 [ 718.8  635.9  685.   754.   613.   657.  2466. ]
10 [ 718.4  635.9  685.   753.   613.   656.  2421. ]
20 [ 719.   635.8  685.   754.   613.   656.  2472. ]
30 [ 719.   635.9  685.   754.   613.   656.  2469. ]
40 [ 718.9  635.8  685.   754.   613.   657.  2473. ]
50 [ 718.9  635.9  685.   754.   613.   657.  2490. ]
60 [ 718.4  635.9  685.   753.   613.   656.  2432. ]
70 [ 718.4  635.9  685.   753.   613.   656.  2420. ]
80 [ 718.4  635.9  685.   753.   613.   656.  2416. ]
90 [ 718.5  635.9  685.   753.   613.   656.  2412. ]
100 [ 718.8  635.7  685.   754.   613.   656.  2476. ]
110 [ 718.2  636.8  685.   754.   615.   656.  2300. ]
120 [ 718.3  637.3  685.   754.   617.   656.  2221. ]
130 [ 718.5  637.5  685.   754.   617.   656.  2200. ]
140 [ 718.4  637.6  685.   754.   617.    …[+4132 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();n=m.face_normals;a=m.area_faces
# histogram normals rounded .1
from collections import Counter
for axis in range(3):
 vals=n[:,axis]; print('axis',axis,'weighted abs>0.8',a[np.abs(vals)>.8].sum()/a.sum(),'pos',a[vals>.8].sum()/a.sum(),'neg',a[vals<-.8].sum()/a.sum())
# top normals signs
for sgn in [1,-1]:
 idx=(n*sgn>.9).any(1); print('dir group',sgn,a[idx].sum()/a.sum())
# min/max extents counts around faces
idx=np.argsort(a)[-20:]
for i in idx: print(a[i],n[i],m.triangles_center[i])
PY"
```
<details><summary>output</summary>

```
axis 0 weighted abs>0.8 0.47575282559216914 pos 0.18689097604012736 neg 0.2888618495520418
axis 1 weighted abs>0.8 0.282128435142108 pos 0.13377698992004716 neg 0.1483514452220609
axis 2 weighted abs>0.8 0.10161841307877252 pos 0.017440710331047687 neg 0.08417770274772483
dir group 1 0.24800608025809479
dir group -1 0.4422269878706227
1.853738571163201e-05 [-0.98423148  0.07500906  0.16019374] [-0.03458941  0.00621943  0.09843393]
1.8787945485212446e-05 [-0.97714093  0.078775    0.1974591 ] [-0.0170067  -0.01647856 -0.01627369]
1.892569357252696e-05 [-0.02374409 -0.98345908  0.17956741] [-0.03212266 -0.05117172  0.05886205]
1.894730382698527e-05 [-0.97814015  0.09226035  0.18635951] [-0.01165466  0.01598027 -0.00335427]
1.9045321350183993e-05 [-0.96962973  0.07413515  0.23307115] [-0.01527 …[+1522 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/search_flat_plane.py <<'PY'
s=open('/work/scratch/pose_search_rgb3.py').read().split('poses=[];meta=[]')[0]
s=s.replace(\"f'/work/scratch/iron_masks/{frame:06d}.png'\",\"f'/work/scratch/sam2_iron4/0/{frame:06d}.png'\")
exec(s);tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.float32,device='cuda');yy,xx=np.where(m[y0:y1,x0:x1]);tuc=(xx.min()+xx.max())/2+x0;tvc=(yy.min()+yy.max())/2+y0
# plane normal up from released floor
pl=json.load(open('/work/inputs/ground_plane.json'))['plane'];up=np.array(pl[:3],float);up=up/np.linalg.norm(up);e1=np.array([1.,0,0]);e1=e1-up*np.dot(up,e1);e1/=np.linalg.norm(e1);e2=np.cross(up,e1);print('up',up,'e1',e1,'e2',e2,np.linalg.norm(np.cross(e1,e2)-up))
cands=[];rots=[]
for yaw in np.arange(0,360,2):
 d=np.cos(np.deg2rad(yaw))*e1+np.sin(np.deg2rad(yaw))*e2;x=np.cross(up,d);R=np.column_stack([x,up,d]);print('det',np.linalg.det(R)) if yaw==0 else None
 for z in np.arange(2.0,4.01,.05):cands.append((yaw,z));rots.append(R)
rots=np.array(rots,np.float32);print('cands',len(cands));Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();res=[]
def render_batch(Rs,ts):
 PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc …[+1454 chars]
```
<details><summary>output</summary>

```
up [-0.01977445 -0.998203   -0.05656619] e1 [ 0.99980447 -0.01974278 -0.00111878] e2 [ 2.16840434e-19 -5.65772490e-02  9.98398225e-01] 1.1102230246251565e-16
det 1.0000000000000002
cands 7380
Traceback (most recent call last):
  File "/work/scratch/search_flat_plane.py", line 17, in <module>
    pred,color=render_batch(Rs,np.array(ts));ts2=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/search_flat_plane.py", line 12, in render_batch
    PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.st …[+1577 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/search_flat_plane.py
python /work/scratch/search_flat_plane.py"
```
<details><summary>output</summary>

```
up [-0.01977445 -0.998203   -0.05656619] e1 [ 0.99980447 -0.01974278 -0.00111878] e2 [ 2.16840434e-19 -5.65772490e-02  9.98398225e-01] 1.1102230246251565e-16
det 1.0000000000000002
cands 7380
BEST 0.7864 0.234 0.826 (312, 2.599999999999998)
BEST 0.7864 0.237 0.82 (312, 2.6499999999999977)
BEST 0.785 0.236 0.821 (314, 2.6499999999999977)
BEST 0.7874 0.24 0.817 (310, 2.6499999999999977)
BEST 0.7787 0.232 0.828 (314, 2.599999999999998)
BEST 0.7816 0.238 0.824 (308, 2.599999999999998)
BEST 0.7775 0.234 0.828 (310, 2.599999999999998)
BEST 0.7743 0.23 0.826 (310, 2.4999999999999982)
BEST 0.7805 0.24 0.821 (306, 2.599999999999998)
BEST 0.778 0.237 0.823 (308, 2.549999999999998)
BEST 0.7746 0.233 0.826 (312, 2.549999999999998)
BEST 0.7797 0.24 0.821 (308, 2.6499999999999977)
BEST 0.7791 0.24 0.821 …[+860 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/search_up_plane.py <<'PY'
s=open('/work/scratch/pose_search_rgb3.py').read().split('poses=[];meta=[]')[0]
s=s.replace(\"frame=0\",\"frame=895\").replace(\"f'/work/scratch/iron_masks/{frame:06d}.png'\",\"f'/work/scratch/sam2_iron4/0/{frame:06d}.png'\")
exec(s);tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.float32,device='cuda');yy,xx=np.where(m[y0:y1,x0:x1]);tuc=(xx.min()+xx.max())/2+x0;tvc=(yy.min()+yy.max())/2+y0
pl=json.load(open('/work/inputs/ground_plane.json'))['plane'];up=np.array(pl[:3],float);up/=np.linalg.norm(up);e1=np.array([1.,0,0]);e1=e1-up*np.dot(up,e1);e1/=np.linalg.norm(e1);e2=np.cross(up,e1)
cands=[];rots=[]
for yaw in np.arange(0,360,1):
 d=np.cos(np.deg2rad(yaw))*e1+np.sin(np.deg2rad(yaw))*e2;y=np.cross(up,d);R=np.column_stack([d,y,up])
 for z in np.arange(2.2,3.31,.025):cands.append((yaw,z));rots.append(R)
rots=np.array(rots,np.float32);print('cands',len(cands));Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();res=[]
def render_batch(Rs,ts):
 PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,re …[+1468 chars]
```
<details><summary>output</summary>

```
cands 16200
Traceback (most recent call last):
  File "/work/scratch/search_up_plane.py", line 16, in <module>
    pred,color=render_batch(Rs,np.array(ts));ts2=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/search_up_plane.py", line 11, in render_batch
    PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(te …[+1393 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/search_up_plane.py
python /work/scratch/search_up_plane.py"
```
<details><summary>output</summary>

```
cands 16200
BEST 0.8089 0.247 0.748 (41, 2.6999999999999984)
BEST 0.8022 0.244 0.753 (41, 2.6749999999999985)
BEST 0.8022 0.245 0.751 (40, 2.7499999999999982)
BEST 0.8036 0.246 0.75 (40, 2.6999999999999984)
BEST 0.8081 0.249 0.743 (43, 2.6999999999999984)
BEST 0.8048 0.247 0.747 (42, 2.7249999999999983)
BEST 0.8086 0.249 0.741 (44, 2.6999999999999984)
BEST 0.8057 0.247 0.745 (43, 2.7249999999999983)
BEST 0.8039 0.246 0.747 (42, 2.6749999999999985)
BEST 0.8066 0.248 0.743 (42, 2.6999999999999984)
BEST 0.8022 0.245 0.748 (43, 2.6749999999999985)
BEST 0.7987 0.244 0.751 (42, 2.6499999999999986)
BEST 0.8014 0.246 0.746 (44, 2.7499999999999982)
BEST 0.8031 0.247 0.744 (44, 2.6749999999999985)
BEST 0.7974 0.245 0.752 (40, 2.6749999999999985)
BEST 0.7961 0.245 0.754 (37, 2.6749999999999985)
BEST  …[+1164 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/fit_trans_fixed.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,time,sys
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);T_cw=np.linalg.inv(T_wc);Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');up=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);up/=np.linalg.norm(up);e1=np.array([1.,0,0]);e1=e1-up*np.dot(up,e1);e1/=np.linalg.norm(e1);e2=np.cross(up,e1);gl=dr.RasterizeCudaContext()
def R_yaw_flat(yaw):
 d=np.cos(np.deg2rad(yaw))*e1+np.sin(np.deg2rad(yaw))*e2;x=np.cross(up,d);return torch.tensor(np.column_stack([x,up,d]),dtype=torch.float32,device='cuda')
def R_yaw_up(yaw):
 d=np.cos(np.deg2rad(yaw))*e1+np.sin(np.deg2rad(yaw))*e2;y=np.cross(up,d);return torch.tensor(np.column_stack([d,y,up]),dtype=torch.float32,device='cuda')
def render_batch(R,Ts,Kc,W,H):
 # Ts world mesh transforms B,3,4
 Tcw4=torch.eye(4,device='cuda')[None].repeat(len(Ts),1,1);Tcw4[:,:3,:4]=Tcw[None,None,:3,:4]@Ts
 Vc=V[None]@Tcw4[:,:3,:3].transpose(1,2)+Tcw …[+2959 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fit_trans_fixed.py", line 42, in <module>
    ss=fit(i,R); # score temporal prior none, photo+ncc
       ^^^^^^^^
  File "/work/scratch/fit_trans_fixed.py", line 27, in fit
    zz=zs[st:st+12];ts=make_ts(zz,tuc,tvc);pred,color=render_batch(R,np.stack(ts),Kc,W,H)
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/fit_trans_fixed.py", line 10, in render_batch
    Tcw4=torch.eye(4,device='cuda')[None].repeat(len(Ts),1,1);Tcw4[:,:3,:4]=Tcw[None,None,:3,:4]@Ts
                                                                            ~~~~~~~~~~~~~~~~~~~~^~~
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
        …[+136 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/fit_trans_fixed.py';s=open(p).read();s=s.replace(\"Tcw4[:,:3,:4]=Tcw[None,None,:3,:4]@Ts\",\"Ts=torch.tensor(Ts,dtype=torch.float32,device='cuda');Tcw4[:,:3,:4]=Tcw[None,None,:3,:4]@Ts\")
open(p,'w').write(s)
PY
python /work/scratch/fit_trans_fixed.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fit_trans_fixed.py", line 42, in <module>
    ss=fit(i,R); # score temporal prior none, photo+ncc
       ^^^^^^^^
  File "/work/scratch/fit_trans_fixed.py", line 27, in fit
    zz=zs[st:st+12];ts=make_ts(zz,tuc,tvc);pred,color=render_batch(R,np.stack(ts),Kc,W,H)
                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/fit_trans_fixed.py", line 11, in render_batch
    Vc=V[None]@Tcw4[:,:3,:3].transpose(1,2)+Tcw4[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W …[+1307 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/fit_trans_fixed.py
python /work/scratch/fit_trans_fixed.py"
```
<details><summary>output</summary>

```
flat 0 iou 0.786 l1 0.234 ncc 0.826 z 2.6 t [0.018 0.328 2.549]
flat 160 iou 0.856 l1 0.235 ncc 0.813 z 2.75 t [0.041 0.344 2.698]
flat 200 iou 0.822 l1 0.234 ncc 0.828 z 3.15 t [0.226 0.355 3.079]
flat 240 iou 0.826 l1 0.231 ncc 0.831 z 3.15 t [0.268 0.347 3.075]
flat 300 iou 0.841 l1 0.236 ncc 0.82 z 2.95 t [0.095 0.357 2.893]
flat 360 iou 0.838 l1 0.245 ncc 0.793 z 2.65 t [-0.109  0.35   2.613]
flat 400 iou 0.864 l1 0.229 ncc 0.797 z 2.35 t [-0.18   0.327  2.32 ]
flat 480 iou 0.831 l1 0.238 ncc 0.82 z 2.95 t [0.081 0.36  2.894]
flat 560 iou 0.793 l1 0.229 ncc 0.835 z 3.0 t [0.232 0.332 2.928]
flat 620 iou 0.806 l1 0.235 ncc 0.834 z 3.2 t [0.193 0.361 3.133]
flat 680 iou 0.728 l1 0.248 ncc 0.803 z 2.95 t [0.147 0.339 2.887]
up 800 iou 0.812 l1 0.245 ncc 0.745 z 2.7 t [-0.012  0.28   2.65 …[+203 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
up=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);up/=np.linalg.norm(up);e1=np.array([1.,0,0]);e1=e1-up*np.dot(up,e1);e1/=np.linalg.norm(e1);e2=np.cross(up,e1)
def flat(yaw):
 d=np.cos(np.deg2rad(yaw))*e1+np.sin(np.deg2rad(yaw))*e2;x=np.cross(up,d);return np.column_stack([x,up,d])
def upright(yaw):
 d=np.cos(np.deg2rad(yaw))*e1+np.sin(np.deg2rad(yaw))*e2;y=np.cross(up,d);return np.column_stack([d,y,up])
A=flat(312);B=upright(41);C=A@R.from_euler('x',-90,degrees=True).as_matrix();print('rel',np.degrees(np.linalg.norm(R.from_matrix(C.T@B).as_rotvec())),C, B)
# projected local x direction etc
print('A cols',A,'B cols',B,'C cols',C)
PY"
```
<details><summary>output</summary>

```
rel 0.9999999999999662 [[ 0.74299952 -0.66899977 -0.01977445]
 [-0.05252931 -0.02883459 -0.998203  ]
 [ 0.66722739  0.74270309 -0.05656619]] [[ 0.75456201 -0.65593075 -0.01977445]
 [-0.05201808 -0.02974696 -0.998203  ]
 [ 0.65416381  0.75423469 -0.05656619]]
A cols [[ 0.74299952 -0.01977445  0.66899977]
 [-0.05252931 -0.998203    0.02883459]
 [ 0.66722739 -0.05656619 -0.74270309]] B cols [[ 0.75456201 -0.65593075 -0.01977445]
 [-0.05201808 -0.02974696 -0.998203  ]
 [ 0.65416381  0.75423469 -0.05656619]] C cols [[ 0.74299952 -0.66899977 -0.01977445]
 [-0.05252931 -0.02883459 -0.998203  ]
 [ 0.66722739  0.74270309 -0.05656619]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
s=np.load('/work/scratch/maskstats4.npy')
for i in range(660,811):
 print(i, 'wh',int(s[i,3]-s[i,2]+1),int(s[i,5]-s[i,4]+1),'c',round(s[i,0],1),round(s[i,1],1),'a',int(s[i,6]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/aspect_curve.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_cw=np.linalg.inv(np.array(cam['T_world_cam']));Tcw=torch.tensor(T_cw,dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();up=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);up/=np.linalg.norm(up);e1=np.array([1.,0,0]);e1=e1-up*np.dot(up,e1);e1/=np.linalg.norm(e1);e2=np.cross(up,e1);yaw=np.deg2rad(312);d=np.cos(yaw)*e1+np.sin(yaw)*e2;x=np.cross(up,d);Rflat=np.column_stack([x,up,d]);
def bbox(beta,z=2.7):
 R=torch.tensor(Rflat@Rot.from_euler('x',beta,degrees=True).as_matrix(),dtype=torch.float32,device='cuda');t=torch.tensor(T_wc[:3,:3]@np.array([0,0,z])+T_wc[:3,3],dtype=torch.float32,device='cuda');Tm=torch.eye(4,device='cuda');Tm[:3,:3]=R;Tm[:3,3]=t;Tc=Tcw@Tm;Vc=V@Tc[:3,:3].T+Tc[:3,3];zz=Vc[:,2:3];uv=Vc[:,:2]/zz*torch.tensor([K[0,0],K[1,1]],device='cuda')+torch.tensor([K[0,2],K[1,2]],device='cuda');ndc=torch.stack([2*uv[:,0]/1536-1,2*uv[:,1]/1152-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1)[None];rast,_=dr.rasterize(gl,pos,F,resolution=(1152,1536));pm=rast[0,...,3]>0;yy,xx …[+198 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/aspect_curve.py", line 6, in <module>
    for b in list(range(-90,1,5)): print(b,bbox(b))
                                           ^^^^^^^
  File "/work/scratch/aspect_curve.py", line 5, in bbox
    R=torch.tensor(Rflat@Rot.from_euler('x',beta,degrees=True).as_matrix(),dtype=torch.float32,device='cuda');t=torch.tensor(T_wc[:3,:3]@np.array([0,0,z])+T_wc[:3,3],dtype=torch.float32,device='cuda');Tm=torch.eye(4,device='cuda');Tm[:3,:3]=R;Tm[:3,3]=t;Tc=Tcw@Tm;Vc=V@Tc[:3,:3].T+Tc[:3,3];zz=Vc[:,2:3];uv=Vc[:,:2]/zz*torch.tensor([K[0,0],K[1,1]],device='cuda')+torch.tensor([K[0,2],K[1,2]],device='cuda');ndc=torch.stack([2*uv[:,0]/1536-1,2*uv[:,1]/1152-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1)[None];rast,_=dr.rasterize(gl,pos,F,resol …[+321 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/T_cw=np.linalg.inv(np.array(cam\\['T_world_cam'\\]))/T_wc=np.array(cam['T_world_cam']);T_cw=np.linalg.inv(T_wc)/\" /work/scratch/aspect_curve.py
python /work/scratch/aspect_curve.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/aspect_curve.py", line 6, in <module>
    for b in list(range(-90,1,5)): print(b,bbox(b))
                                           ^^^^^^^
  File "/work/scratch/aspect_curve.py", line 5, in bbox
    R=torch.tensor(Rflat@Rot.from_euler('x',beta,degrees=True).as_matrix(),dtype=torch.float32,device='cuda');t=torch.tensor(T_wc[:3,:3]@np.array([0,0,z])+T_wc[:3,3],dtype=torch.float32,device='cuda');Tm=torch.eye(4,device='cuda');Tm[:3,:3]=R;Tm[:3,3]=t;Tc=Tcw@Tm;Vc=V@Tc[:3,:3].T+Tc[:3,3];zz=Vc[:,2:3];uv=Vc[:,:2]/zz*torch.tensor([K[0,0],K[1,1]],device='cuda')+torch.tensor([K[0,2],K[1,2]],device='cuda');ndc=torch.stack([2*uv[:,0]/1536-1,2*uv[:,1]/1152-1],-1);pos=torch.cat([ndc*zz,((zz-5)/5)*zz,zz],-1)[None];rast,_=dr.rasterize(gl,pos,F,resol …[+1584 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[K\\[0,0\\],K\\[1,1\\]\\],device='cuda')/torch.tensor([K[0,0],K[1,1]],device='cuda',dtype=torch.float32)/; s/torch.tensor(\\[K\\[0,2\\],K\\[1,2\\]\\],device='cuda')/torch.tensor([K[0,2],K[1,2]],device='cuda',dtype=torch.float32)/\" /work/scratch/aspect_curve.py
python /work/scratch/aspect_curve.py"
```
<details><summary>output</summary>

```
-90 (44, 95, 2.159090909090909, 3152)
-85 (43, 95, 2.2093023255813953, 3155)
-80 (45, 95, 2.111111111111111, 3155)
-75 (49, 95, 1.9387755102040816, 3144)
-70 (53, 94, 1.7735849056603774, 3118)
-65 (56, 92, 1.6428571428571428, 3078)
-60 (59, 89, 1.5084745762711864, 3027)
-55 (63, 86, 1.3650793650793651, 2973)
-50 (65, 82, 1.2615384615384615, 2892)
-45 (68, 78, 1.1470588235294117, 2817)
-40 (70, 72, 1.0285714285714285, 2721)
-35 (71, 66, 0.9295774647887324, 2629)
-30 (73, 61, 0.8356164383561644, 2540)
-25 (74, 56, 0.7567567567567568, 2461)
-20 (74, 50, 0.6756756756756757, 2384)
-15 (75, 45, 0.6, 2326)
-10 (76, 42, 0.5526315789473685, 2281)
-5 (76, 38, 0.5, 2242)
0 (76, 38, 0.5, 2265)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/bbox_cands.py <<'PY'
# use fit_trans_fixed render/functions
import h5py
exec(open('/work/scratch/fit_trans_fixed.py').read().split(\"if __name__=='__main__':\")[0])
# evaluate specific rotations at frame0 using global image etc via fit? easier compute rot and make helper
cap=cv2.VideoCapture('/work/inputs/video.mp4');_,img=cap.read();m=cv2.imread('/work/scratch/sam2_iron4/0/000000.png',0)>0;ys,xs=np.where(m);pad=50;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;yy,xx=np.where(m);tuc=(xx.min()+xx.max())/2;tvc=(yy.min()+yy.max())/2
from scipy.spatial.transform import Rotation as Rot
B=np.column_stack([np.array([0,0,1.]),np.array([0,-1.,0]),np.array([1.,0,0])])
def calc(R,z,name):
 X=(tuc-K[0,2])/K[0,0]*z;Y=(tvc-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];Tm=np.eye(4);Tm[:3,:3]=R;Tm[:3,3]=t;pred,color=render_batch(torch.tensor(R,dtype=torch.float32,device='cuda'),np.array([Tm]),Kc,W,H);pm=pred[0];yy2,xx2=torch.where(pm);print(name,'bbox',int(xx2.max()-xx2.min()+1),int(yy2.max()-yy2.min()+1),'area',len(xx2),'c',float((xx2.min()+xx2.max())/2+x0),float((yy2.min()+yy2.max())/2+y0))
calc(R_yaw_flat(312).cpu().numpy(),2.6,'plane312')
for yaw,tilt,roll in [(55,8,0),(51,3,0),(60,5,0),(70,0,0),(55,15,0),(55,0,0)]:calc(Rot.from_euler('y',yaw,degrees=True).as_matrix()@Rot.from_euler('x',tilt,degrees=True).as_matrix()@Rot.from_euler('z',roll,deg …[+92 chars]
```
<details><summary>output</summary>

```
plane312 bbox 77 44 area 2457 c 720.0 633.5
y55t8r0 bbox 74 46 area 2279 c 720.5 633.5
y51t3r0 bbox 76 43 area 2351 c 720.5 634.0
y60t5r0 bbox 70 44 area 2111 c 721.5 633.5
y70t0r0 bbox 63 42 area 1789 c 722.0 633.5
y55t15r0 bbox 74 48 area 2334 c 720.5 633.5
y55t0r0 bbox 74 42 area 2225 c 720.5 633.5

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_flat_multiframe.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);Tcw=torch.tensor(np.linalg.inv(np.array(cam['T_world_cam'])),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();poses=np.load('/work/scratch/fine_flat_poses.npy');meta=np.load('/work/scratch/fine_flat_meta.npy');frames=[0,32,64,96]
# common crop
m0=cv2.imread('/work/scratch/sam2_iron4/0/000000.png',0)>0;ys,xs=np.where(m0);pad=40;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0
cap=cv2.VideoCapture('/work/inputs/video.mp4');rois={};tgts={}
for i in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();rois[i]=torch.tensor(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255;tgts[i]=torch.tensor(cv2.imread(f'/work/scratch/sam2_iron4/0/{i:06d}.png',0)[y0:y1,x0:x1]>0,device='cuda')
for k,T in enumerate(poses[:30]):
 ss=[]
 for i in frames:
  Tt=to …[+947 chars]
```
<details><summary>output</summary>

```
0 avg [0.7916 0.2343 0.8239] meta [0.78553253 0.23604934 0.82543355]
1 avg [0.7881 0.2331 0.8263] meta [0.7827661  0.23472241 0.827874  ]
2 avg [0.7846 0.2304 0.8326] meta [0.7791121  0.23245314 0.83381391]
3 avg [0.7831 0.2299 0.8342] meta [0.77806121 0.2319877  0.83554852]
4 avg [0.7983 0.2424 0.8144] meta [0.79114163 0.24436411 0.81574261]
5 avg [0.7862 0.2333 0.8249] meta [0.78202814 0.23528478 0.82591182]
6 avg [0.7871 0.2346 0.8268] meta [0.78191292 0.23684534 0.82779664]
7 avg [0.7954 0.2435 0.8148] meta [0.78997707 0.24570557 0.81593084]
8 avg [0.7954 0.2414 0.8154] meta [0.78816795 0.24323003 0.81690174]
9 avg [0.788  0.2363 0.8236] meta [0.7834084  0.23903362 0.82386482]
10 avg [0.7837 0.2319 0.8303] meta [0.77733237 0.23390952 0.83190089]
11 avg [0.7941 0.2415 0.816 ] meta [0.78 …[+1290 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/fit_all_fixed.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,time,h5py
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);Tcw=torch.tensor(np.linalg.inv(T_wc),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();Ftot=896
# fixed rotations from fine searches
Rflat=torch.tensor(np.load('/work/scratch/fine_flat_poses.npy')[0,:3,:3],dtype=torch.float32,device='cuda')
upv=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);upv/=np.linalg.norm(upv);e1=np.array([1.,0,0]);e1-=upv*np.dot(upv,e1);e1/=np.linalg.norm(e1);e2=np.cross(upv,e1);yaw=41;d=np.cos(np.deg2rad(yaw))*e1+np.sin(np.deg2rad(yaw))*e2;y=np.cross(upv,d);Rup=torch.tensor(np.column_stack([d,y,upv]),dtype=torch.float32,device='cuda')

def render_batch(R,Ts,Kc,W,H):
 Ts=torch.tensor(np.asarray(Ts),dtype=torch.float32,device='cuda');Tcw4=torch.eye(4,device='cuda')[None].repeat(len(Ts),1,1);Tcw4[:,:3,:4]=Tcw[None,None,:3,:4]@Ts;Vc=V[None]@Tcw4[:,:3,:3].transpose(1,2)+Tcw4[:,None,:3 …[+3035 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fit_all_fixed.py", line 44, in <module>
    run((0,679),Rflat,2.6,'flat')
  File "/work/scratch/fit_all_fixed.py", line 40, in run
    b=fit(fi,R,img,obj,person,zprev if fi>frames[0] else None);res.append((fi,*b));zprev=b[4]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/fit_all_fixed.py", line 25, in fit
    zz=zs[st:st+17];pred,color=render_batch(R,mk(zz),Kc,W,H)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/fit_all_fixed.py", line 9, in render_batch
    Ts=torch.tensor(np.asarray(Ts),dtype=torch.float32,device='cuda');Tcw4=torch.eye(4,device='cuda')[None].repeat(len(Ts),1,1);Tcw4[:,:3,:4]=Tcw[None,None,:3,:4]@Ts;Vc=V[None]@Tcw4[:,:3,:3].transpose(1,2)+Tcw4[ …[+1742 chars]
```
</details>

**shell** (exit 130)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/fit_all_fixed.py
python /work/scratch/fit_all_fixed.py"
```
<details><summary>output</summary>

```
flat 0 score/iou/l1/ncc/z (0.9143269305004619, 0.7772115034583181, 0.2343066367766988, 0.8296296723006134, 2.5600000000000005) t [0.014 0.324 2.51 ]
flat 25 score/iou/l1/ncc/z (0.5252313187329549, 0.40613645538958415, 0.270592064385469, 0.8010899306360458, 3.96) t [0.106 0.457 3.907]
flat 50 score/iou/l1/ncc/z (0.5234547523257491, 0.4036144578313253, 0.2720711221630776, 0.805846524573388, 3.96) t [0.106 0.462 3.907]
flat 75 score/iou/l1/ncc/z (0.5339989275249787, 0.41477507222451504, 0.2708564112130244, 0.801923114657484, 3.96) t [0.101 0.457 3.908]
flat 100 score/iou/l1/ncc/z (0.5273943303435016, 0.4063004846526656, 0.26527086252936893, 0.8027004177985869, 3.96) t [0.106 0.457 3.907]
flat 125 score/iou/l1/ncc/z (0.5753067341660404, 0.45311091073038773, 0.2654260840097053, 0.80729459455425 …[+695 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/fit_all_fixed.py';s=open(p).read()
s=s.replace(\"if zprev is None: zs=np.arange(2.0,3.61,.08)\\n else:\\n  zests=np.clip(zprev*(95.0/w),1.8,4.2);zs=np.linspace(zests-.24,zests+.24,17)\",\"zs=np.arange(2.0,3.81,.05)\")
s=s.replace(\"for st in range(0,len(zs),17):\\n  zz=zs[st:st+17]\",\"for st in range(0,len(zs),16):\\n  zz=zs[st:st+16]\")
s=s.replace(\"pen=0 if zprev is None else .2*((z-zprev)/.15)**2;score=iou+.25*ncc-.3*l1-pen\",\"score=iou+.25*ncc-.3*l1\")
open('/work/scratch/fit_all_fixed2.py','w').write(s)
PY
python /work/scratch/fit_all_fixed2.py"
```
<details><summary>output</summary>

```
flat 0 score/iou/l1/ncc/z (0.9210761004676022, 0.7855325336241367, 0.23604941375494765, 0.8254335638797995, 2.549999999999998) t [0.015 0.323 2.499]
flat 25 score/iou/l1/ncc/z (0.9090181012804476, 0.7717782577393808, 0.2347397305707222, 0.8306470508491337, 2.549999999999998) t [0.015 0.319 2.499]
flat 50 score/iou/l1/ncc/z (0.919235135378351, 0.7838814600650524, 0.2350049555645835, 0.8234206479306945, 2.549999999999998) t [0.015 0.323 2.499]
flat 75 score/iou/l1/ncc/z (0.9152322128827108, 0.7767247915087188, 0.23144519331123795, 0.8317639174694539, 2.6499999999999977) t [0.02  0.331 2.599]
flat 100 score/iou/l1/ncc/z (0.9147925631039608, 0.7765342960288809, 0.22925903751846868, 0.8281439133224823, 2.549999999999998) t [0.015 0.319 2.499]
flat 125 score/iou/l1/ncc/z (0.9601743901983002, 0.8 …[+4096 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
# modify bbox script execute then print Rup yaw41
s=open('/work/scratch/bbox_cands.py').read();s=s.split(\"calc(R_yaw_flat\")[0];exec(s)
upv=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);upv/=np.linalg.norm(upv);e1=np.array([1.,0,0]);e1-=upv*np.dot(upv,e1);e1/=np.linalg.norm(e1);e2=np.cross(upv,e1);d=np.cos(np.deg2rad(41))*e1+np.sin(np.deg2rad(41))*e2;y=np.cross(upv,d);calc(np.column_stack([d,y,upv]),2.7,'up41')
calc(R_yaw_flat(312).cpu().numpy()@Rot.from_euler('x',-90,degrees=True).as_matrix(),2.7,'flatrx90')
PY"
```
<details><summary>output</summary>

```
up41 bbox 45 96 area 3193 c 718.0 628.5
flatrx90 bbox 44 96 area 3204 c 717.5 628.5

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/refine_up_sift.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);Tcw=torch.tensor(np.linalg.inv(T_wc),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();frame=895;cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,frame);_,img=cap.read();m=cv2.imread(f'/work/scratch/sam2_iron4/0/{frame:06d}.png',0)>0;ys,xs=np.where(m);pad=35;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;roi=torch.tensor(cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255;tgt=torch.tensor(m[y0:y1,x0:x1],dtype=torch.bool,device='cuda');yy,xx=np.where(m[y0:y1,x0:x1]);tuc=(xx.min()+xx.max())/2+x0;tvc=(yy.min()+yy.max())/2+y0
Bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])]);cands=[];rots=[]
for yaw in np …[+3624 chars]
```
<details><summary>output</summary>

```
cands 8064
Traceback (most recent call last):
  File "/work/scratch/refine_up_sift.py", line 18, in <module>
    pred,color=rend(Rs,np.array(ts));ts2=[]
               ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/refine_up_sift.py", line 13, in rend
    PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linea …[+1366 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/refine_up_sift.py
python /work/scratch/refine_up_sift.py"
```
<details><summary>output</summary>

```
cands 8064
Traceback (most recent call last):
  File "/work/scratch/refine_up_sift.py", line 28, in <module>
    pred,color=rend(np.array([allT[j,:3,:3]]),np.array([allT[j]]));pm=pred[0];out=(color[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8);out[~pm.cpu().numpy()]=0;g2=cv2.resize(cv2.cvtColor(out,cv2.COLOR_RGB2GRAY),None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC);k2,d2=sift.detectAndCompute(g2,None);good=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/refine_up_sift.py", line 13, in rend
    PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],K …[+2469 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/rend(np.array(\\[allT\\[j,:3,:3\\]\\]),np.array(\\[allT\\[j\\]\\]))/rend(np.array([allT[j,:3,:3]]),np.array([allT[j,:3,3]]))/\" /work/scratch/refine_up_sift.py
python /work/scratch/refine_up_sift.py"
```
<details><summary>output</summary>

```
cands 8064
SIFT 26 76 rank 247 metrics [8.465e-01 2.499e-01 7.493e-01 5.519e+03] meta (319, 10, -25, 2.6)
SIFT 23 64 rank 187 metrics [8.521e-01 2.512e-01 7.473e-01 5.518e+03] meta (319, 10, -25, 2.5)
SIFT 21 80 rank 120 metrics [8.635e-01 2.505e-01 7.404e-01 4.278e+03] meta (313, 20, -30, 2.4)
SIFT 21 73 rank 244 metrics [8.451e-01 2.490e-01 7.517e-01 5.175e+03] meta (317, 20, -30, 2.5)
SIFT 21 69 rank 153 metrics [8.589e-01 2.515e-01 7.428e-01 4.623e+03] meta (315, 10, -25, 2.6)
SIFT 21 69 rank 119 metrics [8.590e-01 2.490e-01 7.484e-01 5.119e+03] meta (317, 15, -30, 2.5)
SIFT 21 67 rank 295 metrics [8.414e-01 2.494e-01 7.522e-01 5.623e+03] meta (319, 20, -30, 2.5)
SIFT 21 66 rank 221 metrics [8.508e-01 2.504e-01 7.447e-01 4.727e+03] meta (315, 20, -30, 2.5)
SIFT 21 58 rank 218 metrics [ …[+2057 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as Rot
Rf=np.load('/work/scratch/fine_flat_poses.npy')[0,:3,:3]
Bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])])
for y,t,r in [(319,10,-25),(315,15,-30),(313,20,-30)]:
 Ru=Rot.from_euler('y',y,degrees=True).as_matrix()@Rot.from_euler('x',t,degrees=True).as_matrix()@Rot.from_euler('z',r,degrees=True).as_matrix()@Bu
 rel=Rot.from_matrix(Rf.T@Ru); aa=np.degrees(rel.as_rotvec());print(y,t,r,'rel angle',np.linalg.norm(aa),'axis',aa/(np.linalg.norm(aa)+1e-9),'axis world',Rf@(aa/(np.linalg.norm(aa)+1e-9)))
PY"
```
<details><summary>output</summary>

```
319 10 -25 rel angle 82.06511813528901 axis [-0.95784114 -0.20763683  0.19856307] axis world [-0.60995729  0.2574818  -0.74943658]
315 15 -30 rel angle 77.35514439383961 axis [-0.93303139 -0.23960063  0.26841008] axis world [-0.54544671  0.28810335 -0.78707326]
313 20 -30 rel angle 72.05893105673869 axis [-0.92864959 -0.2415016   0.28157932] axis world [-0.53368109  0.28977239 -0.79449129]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_up_multiframe.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);Tcw=torch.tensor(np.linalg.inv(T_wc),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();frames=[800,830,860,895]
# crop based on static target frame895
m0=cv2.imread('/work/scratch/sam2_iron4/0/000895.png',0)>0;ys,xs=np.where(m0);pad=35;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;yy,xx=np.where(m0);tuc=(xx.min()+xx.max())/2;tvc=(yy.min()+yy.max())/2
cap=cv2.VideoCapture('/work/inputs/video.mp4');rois={};tgts={}
for i in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();rois[i]=torch.tensor(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255;tgts[i]=torch.tensor(cv2.imread(f'/work/scratch/sam2_iron4/0/{i:06d}.png',0)[y0:y1,x0:x1]>0,device='cuda') …[+2793 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_up_multiframe.py", line 16, in <module>
    act=cv2.cvtColor(rois[i].cpu().numpy(),cv2.COLOR_RGB2GRAY);act[~tgts[i].cpu().numpy()]=0;g=cv2.resize(act,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC);actual[i]=sift.detectAndCompute(g,None)
                                                                                                                                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) /io/opencv/modules/features2d/src/sift.dispatch.cpp:512: error: (-5:Bad argument) image is empty or has incorrect depth (!=CV_8U) in function 'detectAndCompute'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/act=cv2.cvtColor(rois\\[i\\].cpu().numpy(),cv2.COLOR_RGB2GRAY)/act=cv2.cvtColor((rois[i].cpu().numpy()*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)/\" /work/scratch/eval_up_multiframe.py
python /work/scratch/eval_up_multiframe.py"
```
<details><summary>output</summary>

```
RES (90, [36, 6, 22, 26], 0.8234141801202395, 0.25742094286674544, 0.7293810280382502, (319.0, 10.0, -25.0, 2.6))
RES (86, [36, 10, 16, 24], 0.8152743809771625, 0.256976487963243, 0.7391435239617827, (321.0, 20.0, -25.0, 2.4))
RES (80, [35, 9, 16, 20], 0.819973272423346, 0.25550440971405064, 0.7436469929681047, (323.0, 20.0, -25.0, 2.5))
RES (78, [26, 5, 27, 20], 0.8455195279388356, 0.2567983526041402, 0.7238257090217634, (313.0, 10.0, -30.0, 2.5))
RES (77, [31, 7, 20, 19], 0.8280673769768765, 0.25762779222389376, 0.7365770216639456, (321.0, 15.0, -25.0, 2.5))
RES (74, [28, 6, 21, 19], 0.8415883713603477, 0.25473417596415837, 0.7339196856424011, (315.0, 15.0, -30.0, 2.5))
RES (74, [28, 5, 23, 18], 0.8358024857840267, 0.2563797138574846, 0.7288182013060359, (313.0, 15.0, -30.0, 2.4))
RES (7 …[+2602 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
m=np.load('/work/scratch/fine_flat_meta.npy');print(m[:15,:])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
Rf=np.load('/work/scratch/fine_flat_poses.npy')[0,:3,:3];Bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])]);Ru=Rot.from_euler('y',319,degrees=True).as_matrix()@Rot.from_euler('x',10,degrees=True).as_matrix()@Rot.from_euler('z',-25,degrees=True).as_matrix()@Bu
rel=Rot.from_matrix(Rf.T@Ru);v=rel.as_rotvec();print(np.degrees(v),np.degrees(np.linalg.norm(v)), 'axis local',v/np.linalg.norm(v),'axis world',Rf@(v/np.linalg.norm(v)))
print('Rup',Ru)
PY"
```
<details><summary>output</summary>

```
[-78.60534648 -17.03974133  16.29510156] 82.06511813528901 axis local [-0.95784114 -0.20763683  0.19856307] axis world [-0.60995729  0.2574818  -0.74943658]
Rup [[ 0.7321453  -0.64609202 -0.21570434]
 [-0.41619774 -0.17364818 -0.89253894]
 [ 0.53920562  0.74324385 -0.39603774]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/fit_up_chosen.py <<'PY'
s=open('/work/scratch/fit_all_fixed2.py').read().split(\"run((0,679)\")[0]
exec(s)
from scipy.spatial.transform import Rotation as Rot
Bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])])
Rup=torch.tensor(Rot.from_euler('y',319,degrees=True).as_matrix()@Rot.from_euler('x',10,degrees=True).as_matrix()@Rot.from_euler('z',-25,degrees=True).as_matrix()@Bu,dtype=torch.float32,device='cuda')
run((780,895),Rup,2.6,'up_chosen')
PY
python /work/scratch/fit_up_chosen.py"
```
<details><summary>output</summary>

```
up_chosen 800 score/iou/l1/ncc/z (0.9494917543489654, 0.8388829215896885, 0.2508639483117009, 0.7434720690111486, 2.4999999999999982) t [-0.028  0.252  2.454]
up_chosen 825 score/iou/l1/ncc/z (0.9520642430723594, 0.8418297051529346, 0.2525040862978037, 0.743943055235063, 2.549999999999998) t [-0.024  0.256  2.504]
up_chosen 850 score/iou/l1/ncc/z (0.9553333959183072, 0.844559585492228, 0.25188264538400273, 0.7453544161651208, 2.599999999999998) t [-0.023  0.258  2.554]
up_chosen 875 score/iou/l1/ncc/z (0.9543173607299817, 0.8428914377167245, 0.2507502945905063, 0.7466040455616366, 2.4999999999999982) t [-0.028  0.252  2.454]
up_chosen elapsed 35.55906820297241

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/probe_transition.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,h5py
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);Tcw=torch.tensor(np.linalg.inv(T_wc),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext()
Rf=Rot.from_matrix(np.load('/work/scratch/fine_flat_poses.npy')[0,:3,:3]);Bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])]);Ru=Rot.from_matrix(Rot.from_euler('y',319,degrees=True).as_matrix()@Rot.from_euler('x',10,degrees=True).as_matrix()@Rot.from_euler('z',-25,degrees=True).as_matrix()@Bu)
def render_batch(Rs,ts,Kc,W,H):
 PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[. …[+2833 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/probe_transition.py", line 24, in <module>
    pred,color=render_batch(Rs,np.array(ts),Kc,W,H);ts2=[]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/probe_transition.py", line 7, in render_batch
    PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture …[+1396 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/probe_transition.py
python /work/scratch/probe_transition.py"
```
<details><summary>output</summary>

```
(0.8967348841347743, 0.7924843423799582, 0.2808034807254033, 0.7539663438897486, 0.55, 2.7000000000000006)
(0.8926122608924965, 0.7845213849287169, 0.27330059866765305, 0.7603242222563015, 0.5, 2.6000000000000005)
(0.8894540787882685, 0.7843959731543624, 0.27950056482456537, 0.7556331003251028, 0.525, 2.7000000000000006)
(0.8891287311584595, 0.7825736183945139, 0.2771379503121326, 0.7587859914303412, 0.525, 2.6000000000000005)
(0.8873568416724564, 0.7833614864864865, 0.2799030368572726, 0.7518650649726067, 0.5, 2.7000000000000006)
(0.8837236043025265, 0.7776006377042647, 0.27861369415168696, 0.7588282993750718, 0.55, 2.6000000000000005)
(0.8816122715816241, 0.7736928104575164, 0.2766335851765868, 0.7636381467083349, 0.5750000000000001, 2.7000000000000006)
(0.8800182149697129, 0.77453921988 …[+3867 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/fit_transition.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,h5py,time
from scipy.spatial.transform import Rotation as Rot
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);Tcw=torch.tensor(np.linalg.inv(T_wc),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext()
Rf=Rot.from_matrix(np.load('/work/scratch/fine_flat_poses.npy')[0,:3,:3]);Bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])]);Ru=Rot.from_matrix(np.array([[0.7321453,-0.64609202,-0.21570434],[-0.41619774,-0.17364818,-0.89253894],[0.53920562,0.74324385,-0.39603774]]))
# slerp rotations
def slerp(a,b,s):
 qa=a.as_quat();qb=b.as_quat();
 if np.dot(qa,qb)<0:qb=-qb
 d=np.clip(np.dot(qa,qb),-1,1);th=np.arccos(d)
 if th<1e-7:return Rot.from_quat(qa)
 q=(np.sin((1-s)*th)*qa+np.sin(s*th)*qb)/np.sin(th);return Rot.from_quat(q)
ss=np.arange(0,1.001,.025);zs=np.arange(2.2,3.31,.1);Rs=np.array([slerp(Rf,Ru,s).as_matrix() for s in ss],np.float32)
def rend(Rs,ts,Kc,W,H):
 PP=np. …[+3195 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fit_transition.py", line 26, in <module>
    pr,co=rend(Rs[[a for a,b in cc]],np.array(ts),Kc,W,H)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/fit_transition.py", line 15, in rend
    PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');Vc=V[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,i …[+2121 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/fit_transition.py';s=open(p).read();old=\"PP=np.array([np.block([[R,t[:,None]],[np.zeros((1,3)),np.ones((1,1))]]) for R,t in zip(Rs,ts)],np.float32);T=Tcw[None]@torch.tensor(PP,device='cuda');\";new=\"T=Tcw[None]@torch.tensor(np.asarray(ts),dtype=torch.float32,device='cuda');\";assert old in s;s=s.replace(old,new);s=s.replace(\"torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')\",\"torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)\").replace(\"torch.tensor([Kc[0,2],Kc[1,2]],device='cuda')\",\"torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)\");open(p,'w').write(s)
PY
python /work/scratch/fit_transition.py"
```
<details><summary>output</summary>

```
660 s 0.0 z 3.000000000000001 score 0.95873594 t [0.145 0.343 2.938] elapsed 0.9559295177459717
670 s 0.025 z 2.8000000000000007 score 0.98525715 t [0.136 0.325 2.738] elapsed 8.074716567993164
680 s 0.07500000000000001 z 2.8000000000000007 score 0.90249914 t [0.131 0.322 2.738] elapsed 15.31280517578125
690 s 0.1 z 2.6000000000000005 score 0.85477835 t [0.11  0.296 2.539] elapsed 22.84559965133667
700 s 0.07500000000000001 z 2.8000000000000007 score 0.8241035 t [0.121 0.315 2.739] elapsed 33.18212342262268
710 s 0.47500000000000003 z 2.7000000000000006 score 0.83199716 t [0.111 0.285 2.64 ] elapsed 43.49541711807251
720 s 0.55 z 2.7000000000000006 score 0.8967349 t [0.098 0.273 2.641] elapsed 54.043023109436035
730 s 0.6000000000000001 z 2.5000000000000004 score 0.90801376 t [0.063 0.249  …[+702 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');fr=[]
for i in range(660,730):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();mm=cv2.imread(f'/work/scratch/sam2_iron4/0/{i:06d}.png',0)>0;ys,xs=np.where(mm);x0=max(0,xs.min()-50);x1=min(1536,xs.max()+50);y0=max(0,ys.min()-50);y1=min(1152,ys.max()+50);crop=im[y0:y1,x0:x1];crop=cv2.resize(crop,(180,180));cv2.putText(crop,str(i),(3,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);fr.append(crop)
rows=[np.concatenate(fr[i:i+10],1) for i in range(0,70,10)];cv2.imwrite('/work/scratch/transition660_730.jpg',np.concatenate(rows,0))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
D=np.load('/work/scratch/transition_fit.npz');S=D['scores'];ss=D['ss'];zs=D['zs'];fr=D['frames']
for i,f in enumerate(fr):
 if f%5==0:
  a,b=np.unravel_index(np.argmax(S[i]),S[i].shape)
  print(f,round(ss[a],3),round(zs[b],2),round(S[i,a,b],3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/final_orientation.py <<'PY'
# use shared defs from transition fit file before frames loop
s0=open('/work/scratch/fit_transition.py').read().split('frames=list(range')[0]
exec(s0)
cap=cv2.VideoCapture('/work/inputs/video.mp4');frames=[800,830,860,895];tgts={};rois={};acts={}
# use crop based final obj
obj0=cv2.imread('/work/scratch/sam2_iron4/0/000895.png',0)>0;ys,xs=np.where(obj0);pad=35;x0=int(xs.min())-pad;y0=int(ys.min())-pad;x1=int(xs.max())+pad+1;y1=int(ys.max())+pad+1;W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;tuc=(xs.min()+xs.max())/2;tvc=(ys.min()+ys.max())/2
for f in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);_,im=cap.read();obj=cv2.imread(f'/work/scratch/sam2_iron4/0/{f:06d}.png',0)>0;tgts[f]=torch.tensor(obj[y0:y1,x0:x1],device='cuda');rois[f]=torch.tensor(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB),dtype=torch.float32,device='cuda')/255;acts[f]=(im[y0:y1,x0:x1],obj[y0:y1,x0:x1])
# candidates s,z
cands=[];rots=[]
for sv in np.arange(.75,1.001,.01):
 R=slerp(Rf,Ru,sv).as_matrix()
 for z in np.arange(2.2,2.81,.05):cands.append((sv,z));rots.append(R)
rots=np.array(rots,np.float32);res=[];TT=[]
for st in range(0,len(cands),16):
 cc=cands[st:st+16];ts=[]
 for sv,z in cc:
  X=(tuc-K[0,2])/K[0,0]*z;Y=(tvc-K[1,2])/K[1,1]*z;t=T_wc[:3,:3]@np.array([X,Y,z])+T_wc[:3,3];Tm=np.eye(4);Tm[:3,:3]=rots[st+len(ts)];Tm[:3,3]=t;ts.append(Tm)
 pred,color=rend(rots[st:st+16],np.array(ts),Kc,W,H);ts2=[]
 for j,(sv,z) in enumerate(cc):
  …[+2509 chars]
```
<details><summary>output</summary>

```
top
[9.76325409e-01 8.50113981e-01 2.47917891e-01 7.51261947e-01
 2.67000000e+02] (0.9500000000000002, 2.549999999999999)
[  0.97531202   0.8501213    0.24820103   0.74823712 241.        ] (0.9300000000000002, 2.549999999999999)
[9.74070047e-01 8.50280245e-01 2.50952922e-01 7.47236569e-01
 2.92000000e+02] (0.9700000000000002, 2.499999999999999)
[9.73420219e-01 8.50055498e-01 2.51661906e-01 7.46764946e-01
 2.79000000e+02] (0.9600000000000002, 2.499999999999999)
[9.73344316e-01 8.47349570e-01 2.48392501e-01 7.51172487e-01
 2.53000000e+02] (0.9400000000000002, 2.499999999999999)
[9.72809190e-01 8.47240979e-01 2.48641913e-01 7.50083254e-01
 2.54000000e+02] (0.9400000000000002, 2.549999999999999)
[  0.97114209   0.84577905   0.2481236    0.74870827 228.        ] (0.9200000000000002, 2.549999999 …[+3307 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/out\\[~pm\\[0\\].cpu().numpy()\\]=0/out[~pm.cpu().numpy()]=0/\" /work/scratch/final_orientation.py
python /work/scratch/final_orientation.py > /work/scratch/final_orientation.txt
head -35 /work/scratch/final_orientation.txt"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/final_orientation.py", line 38, in <module>
    color,pm=rend(np.array([T[:3,:3]]),np.array([T]),Kc,W,H);out=(color[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8);out[~pm.cpu().numpy()]=0;gg=cv2.resize(cv2.cvtColor(out,cv2.COLOR_RGB2GRAY),None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC);k2,d2=sift.detectAndCompute(gg,None);k1,d1=ad[f];good=[]
                                                                                                                              ^^^^^^^^^^^^^^^^^
TypeError: ufunc 'invert' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
top
[9.76325409e-01 8.50113981e-01 2.47917891e-01 7.51261947e-01
 2.67000000e+02] (0.950 …[+2363 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/dp_transition.py <<'PY'
import numpy as np
D=np.load('/work/scratch/transition_fit.npz');S=D['scores'];ss=D['ss'];zs=D['zs'];frames=D['frames']
nF,nS,nZ=S.shape
# unary favor evidence; enforce s=0 before 690, final target near .95 after 795
U=-8*S.copy()
for i,f in enumerate(frames):
 if f<=689: U[i,1:,:]+=20
 if f>=795:
  for a in range(nS): U[i,a,:]+=8*((ss[a]-.95)/.05)**2
# transition monotonic + smoothness
inf=1e9;dp=np.full((nS,nZ),inf);back=np.zeros((nF,nS,nZ,2),np.int16);dp[0,:]=U[0,0,:]
for i in range(1,nF):
 new=np.full((nS,nZ),inf)
 for a in range(nS):
  for b in range(nZ):
   best=inf;ba=0;bb=0
   for pa in range(a+1): # nondecreasing s
    ds=(a-pa)
    for pb in range(nZ):
     cost=dp[pa,pb]+0.35*(ds/1.0)**2+0.08*((b-pb)/1.0)**2
     if cost<best:best=cost;ba=pa;bb=pb
   new[a,b]=best+U[i,a,b];back[i,a,b]=[ba,bb]
 dp=new
 # progress
end=np.unravel_index(np.argmin(dp),dp.shape);print('end',end,ss[end[0]],zs[end[1]],dp[end])
path=[]
for i in range(nF-1,-1,-1):
 path.append(end);end=tuple(back[i,end[0],end[1]])
path=path[::-1]
for i,f in enumerate(frames):
 if f%5==0:print(f,round(ss[path[i][0]],3),round(zs[path[i][1]],2),round(S[i,path[i][0],path[i][1]],3))
np.save('/work/scratch/tpath.npy',np.array([(frames[i],ss[a],zs[b],a,b) for i,(a,b) in enumerate(path)]))
PY
python /work/scratch/dp_transition.py"
```
<details><summary>output</summary>

```
end (38, 4) 0.9500000000000001 2.6000000000000005 -995.505375442504
660 0.0 3.0 0.959
665 0.0 3.0 0.959
670 0.0 2.9 0.975
675 0.0 2.9 0.941
680 0.0 2.9 0.86
685 0.0 2.8 0.805
690 0.025 2.8 0.823
695 0.075 2.8 0.84
700 0.175 2.8 0.798
705 0.3 2.7 0.812
710 0.375 2.7 0.816
715 0.475 2.7 0.882
720 0.525 2.6 0.889
725 0.575 2.6 0.946
730 0.6 2.5 0.908
735 0.625 2.5 0.895
740 0.65 2.5 0.879
745 0.65 2.5 0.901
750 0.725 2.6 0.919
755 0.775 2.6 0.953
760 0.775 2.6 0.949
765 0.825 2.5 0.922
770 0.875 2.5 0.921
775 0.9 2.5 0.924
780 0.9 2.5 0.934
785 0.9 2.5 0.934
790 0.9 2.5 0.937
795 0.95 2.5 0.954

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/fit_final_chosen.py <<'PY'
s=open('/work/scratch/fit_all_fixed2.py').read().split(\"run((0,679)\")[0]
exec(s)
# construct final rotation via slerp flat->RU at .95
bu=np.column_stack([np.array([1.,0,0]),np.array([0,0,1.]),np.array([0,-1.,0])])
RU=Rot.from_matrix np if False else None
from scipy.spatial.transform import Rotation as Rot
Ru=Rot.from_matrix(np.array([[0.7321453,-0.64609202,-0.21570434],[-0.41619774,-0.17364818,-0.89253894],[0.53920562,0.74324385,-0.39603774]]))
Rfm=Rot.from_matrix(Rflat.cpu().numpy())
def slerp(a,b,s):
 qa=a.as_quat();qb=b.as_quat();
 if np.dot(qa,qb)<0: qb=-qb
 th=np.arccos(np.clip(np.dot(qa,qb),-1,1));q=(np.sin((1-s)*th)*qa+np.sin(s*th)*qb)/np.sin(th);return Rot.from_quat(q)
Rfinal=torch.tensor(slerp(Rfm,Ru,.95).as_matrix(),dtype=torch.float32,device='cuda')
run((795,895),Rfinal,2.55,'final_chosen')
PY
python /work/scratch/fit_final_chosen.py"
```
<details><summary>output</summary>

```
  File "/work/scratch/fit_final_chosen.py", line 5
    RU=Rot.from_matrix np if False else None
                       ^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i '/RU=Rot.from_matrix/d' /work/scratch/fit_final_chosen.py
python /work/scratch/fit_final_chosen.py"
```
<details><summary>output</summary>

```
final_chosen 800 score/iou/l1/ncc/z (0.9621687767816489, 0.8489208633093526, 0.24682354582451124, 0.749179908878599, 2.549999999999998) t [-0.024  0.256  2.504]
final_chosen 825 score/iou/l1/ncc/z (0.9618517100386443, 0.8500953938402834, 0.2498018591262101, 0.7467874957448959, 2.4999999999999982) t [-0.024  0.252  2.454]
final_chosen 850 score/iou/l1/ncc/z (0.959334991293197, 0.8485200232153222, 0.25258974976101917, 0.7463675720247229, 2.599999999999998) t [-0.021  0.256  2.554]
final_chosen 875 score/iou/l1/ncc/z (0.9591114556331545, 0.8448700410396717, 0.2469348808377796, 0.753287515379267, 2.549999999999998) t [-0.024  0.256  2.504]
final_chosen elapsed 22.738750457763672

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
for name in ['flat','up','final_chosen']:
 A=np.load(f'/work/scratch/fit_{name}.npy');print(name,A.shape);print('frames',A[0,:8]);print('score min/med',np.percentile(A[:,1],[0,10,50,90,100]),'iou',np.percentile(A[:,2],[0,10,50,90,100]));print('z',np.percentile(A[:,5],[0,10,50,90,100]));print('world y',np.percentile(A[:,8],[0,10,50,90,100]));
PY"
```
<details><summary>output</summary>

```
[0.90183775 0.91122113 0.92209702 0.93294488 0.94388839] iou [0.78767302 0.79978986 0.81004429 0.8208583  0.83149418]
z [2.6   2.675 2.7   2.8   2.8  ]
world y [2.55328156 2.62829935 2.65348525 2.75331717 2.7537031 ]
final_chosen (101, 9)
frames [ 7.95000000e+02  9.53890972e-01  8.40993434e-01  2.46469712e-01
  7.47353804e-01  2.50000000e+00 -2.75152893e-02  2.52128393e-01]
score min/med [0.9503617  0.95665809 0.96233999 0.97297275 0.97833548] iou [0.83922652 0.84464641 0.85056834 0.86047157 0.86726629]
z [2.45 2.5  2.55 2.6  2.6 ]
world y [2.40402107 2.45385642 2.50404538 2.55389246 2.55408631]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
A=np.load('/work/scratch/fit_flat.npy');B=np.load('/work/scratch/fit_final_chosen.npy');
# print every 20
for i in range(0,len(A),20): print('F',int(A[i,0]),*[round(x,3) for x in A[i,1:6]],*[round(x,3) for x in A[i,7:10]])
for i in range(0,len(B),10): print('U',int(B[i,0]),*[round(x,3) for x in B[i,1:6]],*[round(x,3) for x in B[i,7:10]])
PY"
```
<details><summary>output</summary>

```
F 0 0.921 0.786 0.236 0.825 2.55 0.323 2.499
F 20 0.907 0.77 0.235 0.831 2.55 0.319 2.499
F 40 0.92 0.785 0.236 0.824 2.55 0.323 2.499
F 60 0.916 0.777 0.231 0.832 2.65 0.331 2.599
F 80 0.915 0.777 0.232 0.832 2.65 0.331 2.599
F 100 0.915 0.777 0.229 0.828 2.55 0.319 2.499
F 120 0.959 0.825 0.233 0.816 2.8 0.351 2.749
F 140 0.965 0.83 0.231 0.815 2.7 0.342 2.649
F 160 0.979 0.845 0.234 0.815 2.7 0.339 2.648
F 180 0.954 0.818 0.232 0.822 2.85 0.337 2.788
F 200 0.956 0.818 0.232 0.831 3.15 0.353 3.079
F 220 0.966 0.829 0.232 0.828 3.15 0.345 3.075
F 240 0.966 0.829 0.234 0.829 3.1 0.343 3.025
F 260 0.935 0.797 0.232 0.831 2.95 0.331 2.879
F 280 0.947 0.812 0.238 0.825 2.95 0.34 2.886
F 300 0.968 0.832 0.236 0.827 2.95 0.355 2.893
F 320 0.966 0.829 0.231 0.824 2.7 0.336 2.65
F 340 0.977 0.848 …[+1277 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
A=np.load('/work/scratch/fit_flat.npy');
for col,name in [(6,'x'),(7,'y'),(8,'z')]:
 a=A[:,col]; v=np.diff(a); print(name,'range',a.min(),a.max(),'step pct',np.percentile(np.abs(v),[50,90,95,99,100]))
 # print largest jumps
 ix=np.argsort(np.abs(v))[-20:][::-1];print(list(zip(ix[:10],v[ix[:10]].round(4))))
# smoothness baseline vs ours
P=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][:680]; print('baseline translation range',P[:,:3,3].min(0),P[:,:3,3].max(0))
PY"
```
<details><summary>output</summary>

```
baseline translation range [ 0.08132219 -0.26363975  2.9784193 ] [0.3154996  0.04087893 3.5729728 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
A=np.load('/work/scratch/fit_flat.npy');
for a,b in [(0,100),(100,680),(0,680)]:print(a,b,np.median(A[a:b,7]),np.percentile(A[a:b,7],[5,95]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/eval_plane_trans.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,h5py
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);Twc3=T_wc[:3,:3];twc=T_wc[:3,3];Tcw=torch.tensor(np.linalg.inv(T_wc),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();Rf=torch.tensor(np.load('/work/scratch/fine_flat_poses.npy')[0,:3,:3],dtype=torch.float32,device='cuda')
def rend(T,Kc,W,H):
 T=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=Tcw[None]@T;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];return color,rast[0,...,3]>0
def T_at(u,v,y):
 # world y fixed, object origin projects to u,v; solve cam Z
 a=(u-K[0,2 …[+1609 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_plane_trans.py", line 13, in <module>
    T=T_at(tuc,tvc,.335);color,pm=rend(T,Kc,W,H);yy2,xx2=torch.where(pm);pu=(xx2.min()+xx2.max()).float()/2+x0;pv=(yy2.min()+yy2.max()).float()/2+y0;T=T_at(tuc-(pu.item()-tuc),tvc-(pv.item()-tvc),.335);color,pm=rend(T,Kc,W,H);inter=(pm&tgt).sum().item();fp=(pm&~tgt&~pers).sum().item();fn=(tgt&~pm).sum().item();iou=inter/max(inter+fp+fn,1);mi=(pm&tgt)[...,None].float();l1=((color-roi).abs()*mi).sum().item()/(mi.sum().item()*3+1e-6);ncc=(color*roi*mi).sum().item()/((color**2*mi).sum().sqrt()*(roi**2*mi).sum().sqrt()).item();print(f,'iou',round(iou,3),'l1',round(l1,3),'ncc',round(ncc,3),'t',T[:3,3].round(3),'z',float((np.linalg.inv(T_wc)@T)[2,3]))
                                  ^^^^^^^^^^^^^ …[+1834 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/\" /work/scratch/eval_plane_trans.py
python /work/scratch/eval_plane_trans.py"
```
<details><summary>output</summary>

```
0 iou 0.762 l1 0.235 ncc 0.831 t [0.022 0.335 2.641] z 2.692049866295778
80 iou 0.767 l1 0.234 ncc 0.83 t [0.021 0.335 2.642] z 2.692041184981177
120 iou 0.817 l1 0.231 ncc 0.817 t [0.019 0.335 2.592] z 2.642843442034249
160 iou 0.829 l1 0.231 ncc 0.82 t [0.033 0.335 2.607] z 2.659112247972073
200 iou 0.81 l1 0.232 ncc 0.831 t [0.204 0.335 2.88 ] z 2.949181243015851
240 iou 0.81 l1 0.235 ncc 0.826 t [0.25  0.335 2.937] z 3.010816583917766
260 iou 0.787 l1 0.235 ncc 0.827 t [0.223 0.335 2.94 ] z 3.0106537079316174
300 iou 0.805 l1 0.231 ncc 0.829 t [0.077 0.335 2.687] z 2.743402835440738
340 iou 0.827 l1 0.239 ncc 0.813 t [-0.058  0.335  2.537] z 2.579591786234311
380 iou 0.856 l1 0.236 ncc 0.79 t [-0.179  0.335  2.364] z 2.393930616049082
400 iou 0.845 l1 0.232 ncc 0.793 t [-0.182  0.335   …[+632 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/assemble.py <<'PY'
import numpy as np
from scipy.signal import medfilt,savgol_filter
from scipy.spatial.transform import Rotation as Rot
F=896
Pf=np.load('/work/scratch/fit_flat.npy');Pu=np.load('/work/scratch/fit_final_chosen.npy');D=np.load('/work/scratch/transition_fit.npz');TP=np.load('/work/scratch/tpath.npy')
# rotations
Rflat=np.load('/work/scratch/fine_flat_poses.npy')[0,:3,:3]
Rfinal=np.load('/work/scratch/final_orient_T.npy')[:3,:3]
# static translation medians
tflat=np.median(Pf[:160,6:9],axis=0);tfinal=np.median(Pu[:,6:9],axis=0)
T=np.zeros((F,4,4));T[:,3,3]=1
# flat dynamic 160-659 raw translations
raw=Pf[160:660,6:9].copy()
# clean outliers 5frame median, smooth 15
sm=np.stack([savgol_filter(medfilt(raw[:,j],5),15,2) for j in range(3)],1)
# force static frames
T[:160,:3,:3]=Rflat;T[:160,:3,3]=tflat
T[160:660,:3,:3]=Rflat;T[160:660,:3,3]=sm
# transition 660-794
tr=[]
for i in range(len(TP)):
 f,s,z,a,b=TP[i];TM=D['TT'][i,int(a),int(b)];tr.append(TM)
# raw transition translations smooth lightly
tr=np.array(tr);tt=tr[:,:3,3];tt=np.stack([savgol_filter(medfilt(tt[:,j],5),11,2) for j in range(3)],1)
for i in range(135):
 f=int(TP[i,0]);T[f,:3,:3]=tr[i,:3,:3];T[f,:3,3]=tt[i]
# final static
T[795:,:3,:3]=Rfinal;T[795:,:3,3]=tfinal
np.save('/work/scratch/T_assemble.npy',T)
# stats transition between neighbors: angle and translation
def ang(R1,R2):return np.degrees(np.linalg.norm(Rot.from_matrix(R1.T@R2).as_rotvec()))
da=np.array([a …[+522 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/assemble.py", line 8, in <module>
    Rfinal=np.load('/work/scratch/final_orient_T.npy')[:3,:3]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/final_orient_T.npy'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/assemble.py';s=open(p).read();s=s.replace(\"Rfinal=np.load('/work/scratch/final_orient_T.npy')[:3,:3]\",\"Ru=Rot.from_matrix(np.array([[0.7321453,-0.64609202,-0.21570434],[-0.41619774,-0.17364818,-0.89253894],[0.53920562,0.74324385,-0.39603774]]))\\nqa=Rot.from_matrix(Rflat).as_quat();qb=Ru.as_quat();qb=qb if np.dot(qa,qb)>=0 else -qb;th=np.arccos(np.clip(np.dot(qa,qb),-1,1));qv=(np.sin(.05*th)*qa+np.sin(.95*th)*qb)/np.sin(th);Rfinal=Rot.from_quat(qv).as_matrix()\")
open(p,'w').write(s)
PY
python /work/scratch/assemble.py"
```
<details><summary>output</summary>

```
rot steps pct [0.00000000e+00 4.49306994e-15 7.82356568e-15 1.63797399e-14
 2.05162843e+00 2.05162984e+00]
trans steps pct [0.         0.00420327 0.01762743 0.02058446 0.02677245 0.06384256]
0 160 rot max 4.493069939485388e-15 trans max 0.0 trans range [0.01964144 0.33086873 2.59940604] [0.01964144 0.33086873 2.59940604]
160 660 rot max 4.493069939485388e-15 trans max 0.03459553938326345 trans range [-0.18290443  0.31306347  2.19707805] [0.26593123 0.36140725 3.06686135]
660 795 rot max 2.0516298419229786 trans max 0.02105777597761417 trans range [-0.02645949  0.2394543   2.43641925] [0.1460731  0.34427354 2.95135713]
795 896 rot max 0.0 trans max 0.0 trans range [-0.02363113  0.25575824  2.50404538] [-0.02363113  0.25575824  2.50404538]
static flat y/z std [2.42861287e-17 8.88178420e-16 5 …[+76 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/evaluate_poses.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,h5py
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);Tcw=torch.tensor(np.linalg.inv(np.array(cam['T_world_cam'])),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();Tours=np.load('/work/scratch/T_assemble.npy');Tbase=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def render(T,frame,img,obj,person):
 ys,xs=np.where(obj);pad=35;x0=max(0,int(xs.min())-pad);y0=max(0,int(ys.min())-pad);x1=min(1536,int(xs.max())+pad+1);y1=min(1152,int(ys.max())+pad+1);W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;Tt=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=Tcw[None]@Tt;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32);ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution …[+1490 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/evaluate_poses.py", line 10, in <module>
    a=render(Tours[i],i,img,obj,person);b=render(Tbase[i],i,img,obj,person);ours.append(a[:3]+(Tours[i][:3,3],));base.append(b[:3]+(Tbase[i][:3,3],))
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/evaluate_poses.py", line 5, in render
    ys,xs=np.where(obj);pad=35;x0=max(0,int(xs.min())-pad);y0=max(0,int(ys.min())-pad);x1=min(1536,int(xs.max())+pad+1);y1=min(1152,int(ys.max())+pad+1);W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;Tt=torch.tensor(T,dtype=torch.float32,device='cuda')[None];Tc=Tcw[None]@Tt;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=t …[+2203 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/evaluate_poses.py';s=open(p).read();s=s.replace(\"ncc=(color*roi*mi).sum().item()/((color**2*mi).sum().sqrt()*(roi**2*mi).sum().sqrt()).item();return\",\"den=((color**2*mi).sum().sqrt()*(roi**2*mi).sum().sqrt()).item();ncc=(color*roi*mi).sum().item()/(den+1e-6);return\");open(p,'w').write(s)
PY
python /work/scratch/evaluate_poses.py"
```
<details><summary>output</summary>

```
0 (0.7655841731989549, 0.23364075288489097, 0.8327897868850137) (0.0, 0.0, 0.0)
100 (0.7679970160387914, 0.22852145477394625, 0.8303544185019787) (0.0, 0.0, 0.0)
200 (0.8209909429941395, 0.23265664179986606, 0.8327979773038473) (0.0, 0.0, 0.0)
300 (0.8165623495426095, 0.23129617938408092, 0.835143213343078) (0.0, 0.0, 0.0)
400 (0.8578506097560976, 0.22732894173295884, 0.8006842489503443) (0.0, 0.0, 0.0)
500 (0.7983614019116978, 0.23618280978265244, 0.826881960624443) (0.0, 0.0, 0.0)
600 (0.8268305718866916, 0.23668284165539583, 0.8240903956677794) (0.0, 0.0, 0.0)
700 (0.6812443642921551, 0.24796948026668994, 0.7800644541262386) (0.0, 0.0, 0.0)
800 (0.8489208633093526, 0.2468231744676284, 0.7491772085647209) (0.0, 0.0, 0.0)
Traceback (most recent call last):
  File "/work/scratch/evaluate_p …[+362 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/ours.append(a\\[:3\\]+(Tours\\[i\\]\\[:3,3\\],));base.append(b\\[:3\\]+(Tbase\\[i\\]\\[:3,3\\],))/ours.append(a[:3]);base.append(b[:3])/\" /work/scratch/evaluate_poses.py
python /work/scratch/evaluate_poses.py"
```
<details><summary>output</summary>

```
0 (0.7655841731989549, 0.23364075288489097, 0.8327897868850137) (0.0, 0.0, 0.0)
100 (0.7679970160387914, 0.22852145477394625, 0.8303544185019787) (0.0, 0.0, 0.0)
200 (0.8209909429941395, 0.23265664179986606, 0.8327979773038473) (0.0, 0.0, 0.0)
300 (0.8165623495426095, 0.23129617938408092, 0.835143213343078) (0.0, 0.0, 0.0)
400 (0.8578506097560976, 0.22732894173295884, 0.8006842489503443) (0.0, 0.0, 0.0)
500 (0.7983614019116978, 0.23618280978265244, 0.826881960624443) (0.0, 0.0, 0.0)
600 (0.8268305718866916, 0.23668284165539583, 0.8240903956677794) (0.0, 0.0, 0.0)
700 (0.6812443642921551, 0.24796948026668994, 0.7800644541262386) (0.0, 0.0, 0.0)
800 (0.8489208633093526, 0.2468231744676284, 0.7491772085647209) (0.0, 0.0, 0.0)
static flat ours iou/l1/ncc [0.78388937 0.22949995 0.8309025 ] base …[+311 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/refine_translation.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,h5py,time
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);T_wc=np.array(cam['T_world_cam'],float);Tcw=torch.tensor(np.linalg.inv(T_wc),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();Tall=np.load('/work/scratch/T_assemble.npy');Tout=Tall.copy()
def rend(R,ts,Kc,W,H):
 Tcw4=torch.eye(4,device='cuda')[None].repeat(len(ts),1,1);Tm=torch.eye(4,device='cuda')[None].repeat(len(ts),1,1);Tm[:,:3,:3]=R[None];Tm[:,:3,3]=torch.tensor(np.asarray(ts),dtype=torch.float32,device='cuda');Tcw4=Tcw[None]@Tm;Vc=V[None]@Tcw4[:,:3,:3].transpose(1,2)+Tcw4[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear');return rast[...,3]>0,color
ca …[+1892 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/refine_translation.py", line 18, in <module>
    pr,co=rend(R,ts[st:st+20],Kc,W,H)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/refine_translation.py", line 5, in rend
    Tcw4=torch.eye(4,device='cuda')[None].repeat(len(ts),1,1);Tm=torch.eye(4,device='cuda')[None].repeat(len(ts),1,1);Tm[:,:3,:3]=R[None];Tm[:,:3,3]=torch.tensor(np.asarray(ts),dtype=torch.float32,device='cuda');Tcw4=Tcw[None]@Tm;Vc=V[None]@Tcw4[:,:3,:3].transpose(1,2)+Tcw4[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,_=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr …[+1521 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/g; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/g\" /work/scratch/refine_translation.py
python /work/scratch/refine_translation.py"
```
<details><summary>output</summary>

```
200 score 0.9900082478524588 z 3.092630736861966 duv 2 1 t [0.229 0.348 3.021]
250 score 0.9828436725053243 z 3.056082001506501 duv 2 1 t [0.26  0.339 2.981]
300 score 1.0084648472092057 z 2.8950525872568216 duv 2 1 t [0.098 0.35  2.837]
350 score 0.9981980584337997 z 2.529993312783874 duv 2 1 t [-0.077  0.33   2.489]
400 score 1.0110003432265364 z 2.2464905518937193 duv 1 -1 t [-0.173  0.309  2.215]
450 score 0.9852801864331823 z 2.6672623561807067 duv 2 2 t [-0.023  0.342  2.621]
500 score 0.9708826325072609 z 2.8321364614345477 duv 2 2 t [0.12  0.34  2.771]
550 score 0.9799159628983531 z 2.958449218036927 duv 2 1 t [0.223 0.332 2.887]
600 score 0.9889666883177728 z 3.050883153484433 duv 2 2 t [0.244 0.34  2.978]
650 score 0.9920306808680739 z 2.930000000000013 duv 2 1 t [0.147 0.337 2.8 …[+191 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/scratch/T_refined.npy');print(T.shape)
for a,b,name in [(0,160,'static'),(160,660,'slide'),(660,795,'stand'),(795,896,'up')]:
 tt=T[a:b,:3,3];print(name,'range',tt.min(0),tt.max(0),'std',tt.std(0))
da=np.array([np.degrees(np.linalg.norm(Rot.from_matrix(T[i,:3,:3].T@T[i+1,:3,:3]).as_rotvec())) for i in range(895)])
dt=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1);print('steps',np.percentile(da,[50,90,99,100]),np.percentile(dt,[50,90,99,100]))
for i in [158,159,160,161,658,659,660,661,693,694,695,793,794,795,796]:print(i,T[i,:3,3].round(3))
PY"
```
<details><summary>output</summary>

```
(896, 4, 4)
static range [0.01964144 0.33086873 2.59940604] [0.01964144 0.33086873 2.59940604] std [2.42861287e-17 8.88178420e-16 5.77315973e-15]
slide range [-0.17791896  0.30393767  2.1664706 ] [0.27527613 0.36356057 3.07741824] std [0.14032328 0.01076908 0.23116959]
stand range [-0.02047285  0.24228888  2.42231753] [0.15839667 0.35093421 2.98008957] std [0.06264348 0.03486535 0.14336117]
up range [-0.02363113  0.25575824  2.50404538] [-0.02363113  0.25575824  2.50404538] std [4.16333634e-17 1.66533454e-16 3.55271368e-15]
steps [4.49306994e-15 7.82356568e-15 2.05162843e+00 2.05162984e+00] [0.00755905 0.04031744 0.0667743  0.10450304]
158 [0.02  0.331 2.599]
159 [0.02  0.331 2.599]
160 [0.043 0.334 2.608]
161 [0.048 0.336 2.635]
658 [0.148 0.338 2.877]
659 [0.148 0.338 2.877]
660 [0.158 0 …[+216 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
T=np.load('/work/scratch/T_refined.npy')
for i in range(655,696):print(i,T[i,:3,3].round(4), 'rot0',np.round(T[i,:3,:3],3).tolist())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/make_final.py <<'PY'
import numpy as np
from scipy.signal import medfilt,savgol_filter
T=np.load('/work/scratch/T_refined.npy');F=len(T);out=T.copy()
# Smooth dynamic translation inside flat and stand segments; preserve static anchors.
for a,b in [(160,660),(660,795)]:
 tt=T[a:b,:3,3].copy();tt=np.stack([savgol_filter(medfilt(tt[:,j],7),15,2) for j in range(3)],1);out[a:b,:3,3]=tt
# Explicit static anchor equality
out[:160]=T[0];out[795:]=T[-1]
# Round rotations to orthonormal after no-op (already)
np.save('/work/scratch/T_final.npy',out)
# check boundary diffs
for i in [158,159,160,161,657,658,659,660,661,662,792,793,794,795,796]:print(i,out[i,:3,3].round(4))
PY
python /work/scratch/make_final.py"
```
<details><summary>output</summary>

```
158 [0.0196 0.3309 2.5994]
159 [0.0196 0.3309 2.5994]
160 [0.042  0.3313 2.5996]
161 [0.0468 0.3323 2.6147]
657 [0.1476 0.3375 2.8739]
658 [0.148  0.3377 2.8768]
659 [0.1484 0.338  2.8802]
660 [0.153  0.3416 2.924 ]
661 [0.1527 0.3405 2.9092]
662 [0.1523 0.3395 2.8954]
792 [-0.0195  0.2513  2.4629]
793 [-0.0195  0.2514  2.4635]
794 [-0.0196  0.2515  2.4643]
795 [-0.0236  0.2558  2.504 ]
796 [-0.0236  0.2558  2.504 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/scratch/make_final.py';s=open(p).read();s=s.replace(\"for a,b in [(160,660),(660,795)]:\",\"for a,b in [(160,795)]:\");open(p,'w').write(s)
PY
python /work/scratch/make_final.py"
```
<details><summary>output</summary>

```
158 [0.0196 0.3309 2.5994]
159 [0.0196 0.3309 2.5994]
160 [0.042  0.3313 2.5996]
161 [0.0468 0.3323 2.6147]
657 [0.1482 0.338  2.8788]
658 [0.1491 0.3385 2.8842]
659 [0.15   0.3389 2.8894]
660 [0.1508 0.3394 2.8955]
661 [0.1514 0.3396 2.8984]
662 [0.1519 0.3395 2.8965]
792 [-0.0195  0.2513  2.4629]
793 [-0.0195  0.2514  2.4635]
794 [-0.0196  0.2515  2.4643]
795 [-0.0236  0.2558  2.504 ]
796 [-0.0236  0.2558  2.504 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/scratch/make_final2.py <<'PY'
import numpy as np
from scipy.signal import medfilt,savgol_filter
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/scratch/T_refined.npy');out=T.copy()
# smooth continuous dynamic segment
tt=T[160:795,:3,3].copy();tt=np.stack([savgol_filter(medfilt(tt[:,j],7),15,2) for j in range(3)],1);out[160:795,:3,3]=tt
# static anchors
out[:160]=T[0];out[795:]=T[-1]
# smoothstep blend entering manipulation 155..170
for f in range(155,171):
 a=(f-155)/15.;s=a*a*(3-2*a);out[f,:3,3]=(1-s)*out[155,:3,3]+s*out[170,:3,3]
# blend ending manipulation into final static pose 785..800 (rotation slerp + translation)
Ra=Rot.from_matrix(out[785,:3,:3]);Rb=Rot.from_matrix(out[800,:3,:3]);
for f in range(785,801):
 a=(f-785)/15.;s=a*a*(3-2*a);out[f,:3,3]=(1-s)*out[785,:3,3]+s*out[800,:3,3];out[f,:3,:3]= (Ra*(Rb)).as_matrix() if False else None
# slerp implementation
def slerp(a,b,s):
 qa=a.as_quat();qb=b.as_quat();
 if np.dot(qa,qb)<0:qb=-qb
 th=np.arccos(np.clip(np.dot(qa,qb),-1,1));q=(np.sin((1-s)*th)*qa+np.sin(s*th)*qb)/np.sin(max(th,1e-8));return Rot.from_quat(q)
for f in range(785,801):
 a=(f-785)/15.;s=a*a*(3-2*a);out[f,:3,:3]=slerp(Ra,Rb,s).as_matrix()
# re-orthonormalize all rotations
for f in range(896):out[f,:3,:3]=Rot.from_matrix(out[f,:3,:3]).as_matrix()
np.save('/work/scratch/T_final2.npy',out)
for i in [154,155,159,160,164,169,170,171,658,659,660,784,785,789,790,794,795,799,800,801]:print(i,out[i,:3,3].round(4 …[+42 chars]
```
<details><summary>output</summary>

```
154 [0.0196 0.3309 2.5994]
155 [0.0196 0.3309 2.5994]
159 [0.033  0.3324 2.6253]
160 [0.0394 0.3331 2.6377]
164 [0.0689 0.3364 2.6951]
169 [0.0947 0.3393 2.7453]
170 [0.0957 0.3395 2.7471]
171 [0.1013 0.34   2.7587]
658 [0.1491 0.3385 2.8842]
659 [0.15   0.3389 2.8894]
660 [0.1508 0.3394 2.8955]
784 [-0.0188  0.2517  2.4676]
785 [-0.0191  0.2515  2.4661]
789 [-0.0199  0.2523  2.4728]
790 [-0.0203  0.2526  2.4759]
794 [-0.022   0.2543  2.4907]
795 [-0.0224  0.2547  2.4942]
799 [-0.0236  0.2557  2.5036]
800 [-0.0236  0.2558  2.504 ]
801 [-0.0236  0.2558  2.504 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed 's/T_assemble.npy/T_final2.npy/; s/eval_ours.npy/eval_final.npy/; s/eval_base.npy/eval_base2.npy/' /work/scratch/evaluate_poses.py > /work/scratch/evaluate_final.py
python /work/scratch/evaluate_final.py"
```
<details><summary>output</summary>

```
0 (0.7655841731989549, 0.23364075288489097, 0.8327897868850137) (0.0, 0.0, 0.0)
100 (0.7679970160387914, 0.22852145477394625, 0.8303544185019787) (0.0, 0.0, 0.0)
200 (0.8599892299407647, 0.2399329573342083, 0.8154849380415019) (0.0, 0.0, 0.0)
300 (0.8689892051030422, 0.23818920898549514, 0.8172589990117729) (0.0, 0.0, 0.0)
400 (0.8746635909265668, 0.22292496562667766, 0.8099720175067506) (0.0, 0.0, 0.0)
500 (0.8401486988847584, 0.23712288731167072, 0.8141406801374229) (0.0, 0.0, 0.0)
600 (0.8517316017316018, 0.23455058266157336, 0.820528096271051) (0.0, 0.0, 0.0)
700 (0.812354853692522, 0.2416383812983465, 0.7994483096080303) (0.0, 0.0, 0.0)
800 (0.8489208633093526, 0.2468231744676284, 0.7491772085647209) (0.0, 0.0, 0.0)
static flat ours iou/l1/ncc [0.78467388 0.22956392 0.83076852] base [ …[+309 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/fine_flat.py').read()
s=s.replace(\"for yaw in np.arange(51,61,1):\\n for tilt in np.arange(3,14,1):\\n  for roll in np.arange(-8,9,2):\\n   for z in np.arange(2.3,2.71,.05):\",\"for yaw in np.arange(49,53.01,.5):\\n for tilt in np.arange(1,6.01,.5):\\n  for roll in np.arange(-2,2.01,1):\\n   for z in np.arange(2.47,2.631,.02):\")
s=s.replace('/work/scratch/fine_flat_meta.npy','/work/scratch/fine_flat2_meta.npy').replace('/work/scratch/fine_flat_poses.npy','/work/scratch/fine_flat2_poses.npy')
open('/work/scratch/fine_flat2.py','w').write(s)
PY
python /work/scratch/fine_flat2.py"
```
<details><summary>output</summary>

```
cands 4455
BEST 0.7878 0.232 0.83 (49.0, 1.0, 1.0, 2.5900000000000003)
BEST 0.7871 0.232 0.83 (49.0, 2.5, 1.0, 2.5700000000000003)
BEST 0.7889 0.235 0.829 (51.0, 1.5, 1.0, 2.6100000000000003)
BEST 0.7865 0.231 0.831 (49.5, 1.0, 1.0, 2.5700000000000003)
BEST 0.7894 0.234 0.825 (49.0, 1.0, 2.0, 2.5900000000000003)
BEST 0.7874 0.232 0.829 (49.0, 2.0, 1.0, 2.5700000000000003)
BEST 0.786 0.231 0.831 (49.0, 1.0, 1.0, 2.5700000000000003)
BEST 0.787 0.233 0.83 (51.0, 1.5, 1.0, 2.5500000000000003)
BEST 0.7846 0.23 0.832 (51.5, 1.0, 2.0, 2.5500000000000003)
BEST 0.7864 0.232 0.829 (51.0, 1.0, 2.0, 2.5700000000000003)
BEST 0.7897 0.236 0.826 (51.0, 2.0, 1.0, 2.6100000000000003)
BEST 0.7875 0.234 0.829 (49.5, 3.0, 1.0, 2.5900000000000003)
BEST 0.7854 0.231 0.831 (50.0, 2.5, 1.0, 2.5700000000000003)
BE …[+1026 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
T=np.load('/work/scratch/T_final2.npy');print(T[0]);print(T[895]);
A=np.load('/work/scratch/fit_flat.npy');print('fit0',A[0])
PY"
```
<details><summary>output</summary>

```
[[ 7.76080921e-01 -4.06726786e-02  6.29320377e-01  1.96414390e-02]
 [-5.23359576e-02 -9.98629535e-01  2.18504506e-10  3.30868735e-01]
 [ 6.28457915e-01 -3.29360847e-02 -7.77145973e-01  2.59940604e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
[[ 0.74415064 -0.64849277 -0.16030267 -0.02363113]
 [-0.40003519 -0.24042582 -0.88440221  0.25575824]
 [ 0.53498754  0.72225518 -0.43833296  2.50404538]
 [ 0.          0.          0.          1.        ]]
fit0 [0.         0.9210761  0.78553253 0.23604941 0.82543356 2.55
 0.01499449 0.32259809 2.49935946]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
A=np.load('/work/scratch/eval_final.npy')
print('min iou frames')
for i in np.argsort(A[:,0])[:30]:print(i,round(A[i,0],3),round(A[i,1],3),round(A[i,2],3))
for a,b,name in [(0,160,'static flat'),(160,660,'slide'),(660,795,'stand'),(795,896,'static upright')]:
 x=A[a:b];print(name,'iou pct',np.percentile(x[:,0],[0,10,50,90,100]),'ncc',np.percentile(x[:,2],[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
s=open('/work/scratch/dp_transition.py').read();s=s.replace(\" if f<=689: U[i,1:,:]+=20\\n\",\"\")
open('/work/scratch/dp_transition2.py','w').write(s)
PY
python /work/scratch/dp_transition2.py"
```
<details><summary>output</summary>

```
end (38, 4) 0.9500000000000001 2.6000000000000005 -1000.7938420677176
660 0.0 3.0 0.959
665 0.0 3.0 0.959
670 0.025 2.8 0.985
675 0.05 2.8 0.975
680 0.075 2.8 0.902
685 0.1 2.6 0.852
690 0.1 2.6 0.855
695 0.1 2.7 0.834
700 0.175 2.8 0.798
705 0.3 2.7 0.812
710 0.375 2.7 0.816
715 0.475 2.7 0.882
720 0.525 2.6 0.889
725 0.575 2.6 0.946
730 0.6 2.5 0.908
735 0.625 2.5 0.895
740 0.65 2.5 0.879
745 0.65 2.5 0.901
750 0.725 2.6 0.919
755 0.775 2.6 0.953
760 0.775 2.6 0.949
765 0.825 2.5 0.922
770 0.875 2.5 0.921
775 0.9 2.5 0.924
780 0.9 2.5 0.934
785 0.9 2.5 0.934
790 0.9 2.5 0.937
795 0.95 2.5 0.954

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
T=np.load('/work/scratch/T_refined.npy');D=np.load('/work/scratch/transition_fit.npz');TP=np.load('/work/scratch/tpath.npy')
for i in range(len(TP)):
 f,s,z,a,b=TP[i];f=int(f)
 T[f,:3,:3]=D['TT'][i,int(a),int(b),:3,:3]
 if f<=700:T[f,:3,3]=D['TT'][i,int(a),int(b),:3,3]
np.save('/work/scratch/T_transnew_base.npy',T)
print('saved')
PY
# modify refine script range/T file/output
sed 's/for f in range(160,795):/for f in range(660,701):/; s#T_assemble.npy#T_transnew_base.npy#; s#T_refined.npy#T_refine660.npy#' /work/scratch/refine_translation.py > /work/scratch/refine_660.py
python /work/scratch/refine_660.py"
```
<details><summary>output</summary>

```
saved
700 score 0.9396563618207697 z 2.7699999305617333 duv 2 2 t [0.126 0.317 2.708]
elapsed 16.972225189208984

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
T=np.load('/work/scratch/T_refined.npy');R=np.load('/work/scratch/T_refine660.npy');T[660:701]=R[660:701];np.save('/work/scratch/T_combined.npy',T)
p='/work/scratch/make_final2.py';s=open(p).read().replace('T_refined.npy','T_combined.npy').replace(\"T_final2.npy\",\"T_final3.npy\");open('/work/scratch/make_final3.py','w').write(s)
PY
python /work/scratch/make_final3.py
sed 's/T_final2.npy/T_final3.npy/; s/eval_final.npy/eval_final3.npy/; s/eval_base2.npy/eval_base3.npy/' /work/scratch/evaluate_final.py > /work/scratch/evaluate_final3.py
python /work/scratch/evaluate_final3.py"
```
<details><summary>output</summary>

```
154 [0.0196 0.3309 2.5994]
155 [0.0196 0.3309 2.5994]
159 [0.033  0.3324 2.6253]
160 [0.0394 0.3331 2.6377]
164 [0.0689 0.3364 2.6951]
169 [0.0947 0.3393 2.7453]
170 [0.0957 0.3395 2.7471]
171 [0.1013 0.34   2.7587]
658 [0.1492 0.3384 2.8837]
659 [0.1503 0.3392 2.892 ]
660 [0.1512 0.34   2.902 ]
784 [-0.0188  0.2517  2.4676]
785 [-0.0191  0.2515  2.4661]
789 [-0.0199  0.2523  2.4728]
790 [-0.0203  0.2526  2.4759]
794 [-0.022   0.2543  2.4907]
795 [-0.0224  0.2547  2.4942]
799 [-0.0236  0.2557  2.5036]
800 [-0.0236  0.2558  2.504 ]
801 [-0.0236  0.2558  2.504 ]
0 (0.7655841731989549, 0.23364075288489097, 0.8327897868850137) (0.0, 0.0, 0.0)
100 (0.7679970160387914, 0.22852145477394625, 0.8303544185019787) (0.0, 0.0, 0.0)
200 (0.8599892299407647, 0.2399329573342083, 0.8154849380415019) (0.0,  …[+878 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc "cat > /work/scratch/render_validation.py <<'PY'
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);Tcw=torch.tensor(np.linalg.inv(np.array(cam['T_world_cam'])),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();T=np.load('/work/scratch/T_final3.npy');ev=np.load('/work/scratch/eval_final3.npy');cap=cv2.VideoCapture('/work/inputs/video.mp4')
for s,e in [(0,224),(224,448),(448,672),(672,896)]:
 ids=list(range(s,e,16));tw,th=256,192;cols=7;rows=4;out=np.ones((rows*th,cols*tw,3),np.uint8)*30
 for j,i in enumerate(ids):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,img=cap.read();m=cv2.imread(f'/work/scratch/sam2_iron4/0/{i:06d}.png',0)>0;ys,xs=np.where(m);pad=30;x0=max(0,int(xs.min())-pad);y0=max(0,int(ys.min())-pad);x1=min(1536,int(xs.max())+pad+1);y1=min(1152,int(ys.max())+pad+1);W=x1-x0;H=y1-y0;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0
  Tt=torch.tensor(T[i],dtype=torch.float32,device='cuda')[None];Tc=Tcw[None]@Tt;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];z=Vc[...,2:3];uv=Vc …[+992 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_validation.py", line 8, in <module>
    Tt=torch.tensor(T[i],dtype=torch.float32,device='cuda')[None];Tc=Tcw[None]@Tt;Vc=V[None]@Tc[:,:3,:3].transpose(1,2)+Tc[:,None,:3,3];z=Vc[...,2:3];uv=Vc[...,:2]/z*torch.tensor([Kc[0,0],Kc[1,1]],device='cuda')+torch.tensor([Kc[0,2],Kc[1,2]],device='cuda');ndc=torch.stack([2*uv[...,0]/W-1,2*uv[...,1]/H-1],-1);pos=torch.cat([ndc*z,((z-5)/5)*z,z],-1);rast,interp=dr.rasterize(gl,pos,F,resolution=(H,W));icolor=dr.interpolate(UV,rast,F)[0];color=dr.texture(tex,icolor,filter_mode='linear')[0];pm=rast[0,...,3]>0
                                                                                                                                                                                             …[+1082 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed -i \"s/torch.tensor(\\[Kc\\[0,0\\],Kc\\[1,1\\]\\],device='cuda')/torch.tensor([Kc[0,0],Kc[1,1]],device='cuda',dtype=torch.float32)/; s/torch.tensor(\\[Kc\\[0,2\\],Kc\\[1,2\\]\\],device='cuda')/torch.tensor([Kc[0,2],Kc[1,2]],device='cuda',dtype=torch.float32)/\" /work/scratch/render_validation.py
python /work/scratch/render_validation.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "mkdir -p /work/outputs /work/tools
cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a monocular iron trajectory with re-segmentation and render-and-compare.

The supplied baseline mask segmented a tripod standing behind the iron.  This
solver therefore re-prompts SAM2 on the iron itself, then exploits the task'"'s
physical structure: a fixed lying pose while sliding, a smooth slerp to the
final upright pose, and constant poses in the two untouched intervals.
\"\"\"
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation as Rot

# The package is installed in the container but is not always on sys.path when
# this file is invoked from /work/outputs.
sys.path.insert(0, \"/workspace/v2d_sam2\")
from v2d.sam2.lib.video_to_masks import video_to_masks  # noqa: E402


def set_deterministic(seed=7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setRNGSeed(seed)


def slerp(a, b, s):
    qa, qb = a.as_quat(), b.as_quat()
    if np.dot(qa, qb) < 0:
        qb = -qb
    th = np.arccos(np.clip(np.dot(qa, qb), -1.0, 1.0))
    if th < 1e-8:
        return Rot.from_quat(qa)
   …[+18401 chars]
```
<details><summary>output</summary>

```
428 /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"            first_T = {}\\n\", \"            first_T = {}\\n            first_pred = {}\\n\")
s=s.replace(\"                for j, cell in enumerate(cc):\\n                    first_T[cell] = ts[j]\\n\", \"                for j, cell in enumerate(cc):\\n                    first_T[cell] = ts[j]\\n                    first_pred[cell] = pr[j]\\n\")
s=s.replace(\"            for a, b in cells:\\n                z = zs[b]\\n                yy, xx = torch.where(render(rotations[[a]], [first_T[(a, b)]], Kc, W, H)[0][0])\\n\", \"            for a, b in cells:\\n                z = zs[b]\\n                yy, xx = torch.where(first_pred[(a, b)])\\n\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
grep -n \"first_pred\\|first_T\" /work/outputs/solve.py"
```
<details><summary>output</summary>

```
295:            first_T = {}
296:            first_pred = {}
308:                    first_T[cell] = ts[j]
309:                    first_pred[cell] = pr[j]
313:                yy, xx = torch.where(first_pred[(a, b)])

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```
<details><summary>output</summary>

```
15051 MiB, 49140 MiB
17 MiB, 49140 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import ast
ast.parse(open('/work/outputs/solve.py').read());print('syntax ok')
PY"
```
<details><summary>output</summary>

```
syntax ok

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; python - <<'"'PY'
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i,torch.cuda.get_device_name(i))
PY"
```
<details><summary>output</summary>

```
1
0 NVIDIA RTX 6000 Ada Generation

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc '/usr/bin/time -p python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "TIMEFORMAT='elapsed %R user %U sys %S'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/scratch/solve_run.log"
```
<details><summary>output</summary>

```
/workspace/v2d_sam2/lib/sam2_utils.py:1058: UserWarning: cannot import name '_C' from 'sam2' (/opt/conda/lib/python3.11/site-packages/sam2/__init__.py)

Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
  pred_masks_gpu = fill_holes_in_mask_scores(
propagate in video: 0it [00:00, ?it/s]propagate in video: 0it [00:00, ?it/s]
propagate in video:   0%|          | 0/896 [00:00<?, ?it/s]propagate in video:   0%|          | 3/896 [00:00<00:39, 22.47it/s]propagate in video:   1%|          | 6/896 [00:00<00:45, 19.45it/s]propagate in video:   1%|    …[+31083 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed \"s#T=np.load('/work/scratch/T_final3.npy')#T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']#; s/eval_final3.npy/eval_output.npy/; s/eval_base3.npy/eval_base_output.npy/\" /work/scratch/evaluate_final3.py > /work/scratch/evaluate_output.py
python /work/scratch/evaluate_output.py"
```
<details><summary>output</summary>

```
0 (0.7655841731989549, 0.23364075288489097, 0.8327897868850137) (0.0, 0.0, 0.0)
100 (0.7679970160387914, 0.22852145477394625, 0.8303544185019787) (0.0, 0.0, 0.0)
200 (0.8599892299407647, 0.2399329573342083, 0.8154849380415019) (0.0, 0.0, 0.0)
300 (0.8689892051030422, 0.23818920898549514, 0.8172589990117729) (0.0, 0.0, 0.0)
400 (0.8746635909265668, 0.22292496562667766, 0.8099720175067506) (0.0, 0.0, 0.0)
500 (0.8401486988847584, 0.23712288731167072, 0.8141406801374229) (0.0, 0.0, 0.0)
600 (0.8517316017316018, 0.23455058266157336, 0.820528096271051) (0.0, 0.0, 0.0)
700 (0.8133395090319593, 0.24340020872461793, 0.7969487327286083) (0.0, 0.0, 0.0)
800 (0.8489208633093526, 0.2468231744676284, 0.7491772085647209) (0.0, 0.0, 0.0)
static flat ours iou/l1/ncc [0.78467388 0.22956392 0.83076852] base …[+311 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def stats(P):
 da=np.array([np.degrees(np.linalg.norm(Rot.from_matrix(P[i,:3,:3].T@P[i+1,:3,:3]).as_rotvec())) for i in range(len(P)-1)])
 dt=np.linalg.norm(P[1:,:3,3]-P[:-1,:3,3],axis=1)
 return np.percentile(da,[50,90,99,100]),np.percentile(dt,[50,90,99,100])
print('ours',stats(T));print('base',stats(B))
print('shape',T.shape,T.dtype,np.isfinite(T).all())
# orthonormal residual
print('orth',np.abs(np.einsum('nij,nkj->nik',T[:,:3,:3],T[:,:3,:3])-np.eye(3)).max(),'det',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
PY"
```
<details><summary>output</summary>

```
ours (array([0.        , 0.        , 2.05162828, 2.05163063]), array([0.00497884, 0.0156771 , 0.02829889, 0.04318018]))
base (array([1.61133992e-14, 2.13559363e+01, 9.61087185e+01, 1.53176933e+02]), array([0.00158025, 0.0666437 , 0.12098043, 0.3231664 ]))
shape (896, 4, 4) float64 True
orth 6.661338147750939e-16 det 0.9999999999999992 1.0000000000000007

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "git -C /workspace rev-parse HEAD 2>/dev/null || true; git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true; git -C /workspace/v2d_sam2 rev-parse HEAD 2>/dev/null || true; python - <<'PY'
import torch,cv2,numpy,scipy,trimesh,h5py,nvdiffrast,pytorch3d,sam2
for n,m in [('torch',torch),('opencv',cv2),('numpy',numpy),('scipy',scipy),('trimesh',trimesh),('h5py',h5py),('nvdiffrast',nvdiffrast),('pytorch3d',pytorch3d),('sam2',sam2)]:print(n,getattr(m,'__version__','unknown'))
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124
opencv 4.11.0
numpy 1.26.3
scipy 1.16.3
trimesh 5.1.1
h5py 3.16.0
nvdiffrast 0.4.0
pytorch3d 0.7.9
sam2 unknown

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sha256sum /weights/sam2/sam2.1_hiera_large.pt /weights/sam2/sam2.1_hiera_l.yaml 2>/dev/null; stat -c '%n %s' /weights/sam2/sam2.1_hiera_large.pt /weights/sam2/sam2.1_hiera_l.yaml"
```
<details><summary>output</summary>

```
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318  /weights/sam2/sam2.1_hiera_large.pt
545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636  /weights/sam2/sam2.1_hiera_l.yaml
/weights/sam2/sam2.1_hiera_large.pt 898083611
/weights/sam2/sam2.1_hiera_l.yaml 3798

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import PIL, matplotlib, skimage
for n,m in [('Pillow',PIL),('matplotlib',matplotlib),('scikit-image',skimage)]: print(n,getattr(m,'__version__','unknown'))
PY
ffmpeg -version | head -1; ffprobe -version | head -1"
```
<details><summary>output</summary>

```
Pillow 10.2.0
matplotlib 3.11.2
scikit-image 0.26.0
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
ffprobe version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2007-2021 the FFmpeg developers

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "cat > /work/outputs/REPORT.md <<'MD'
# Monocular iron trajectory recovery

## Summary

The supplied reconstruction was not usable as a pose prior for this episode. Its
object mask at frame 0 was a box prompt covering both the iron and the tripod
behind it; SAM2 selected the tripod. The baseline pose consequently renders on
the tripod/background and has no overlap with the actual iron. I therefore
re-segmented the iron and recomputed the complete trajectory rather than
blending with the baseline.

Final output: "'`object_poses_world.npz`, key `T_world_mesh`, shape `(896,4,4)`,
dtype `float64`.

## Phases found (verified from the corrected masks and video)

| Frames | Phase | Treatment |
|---|---|---|
| 0–159 | Iron lies untouched on the round table | One constant world pose |
| 160–659 | Right-hand slide back and forth | Fixed lying rotation; per-frame translation/depth |
| 660–794 | Iron is rotated up to stand vertically | Monotone slerp from lying orientation to final upright orientation; Viterbi decoding over rotation progress and depth |
| 795–895 | Iron stands still after release | One constant world pose |

The object mask is unchanged to within SAM2 jitter through frame 159. The first
visible object displacement is at frame 160. The stand-up starts subtly near
frame 670 and is essentially complete by frame 795.

## Method

1. **Object re-segmentation.** SAM2.1-Hiera-L was prompted on frame 0 with a
   tight box `(665,610)-(780,675)`, which contains th …[+6793 chars]
```
<details><summary>output</summary>

```
total 152K
-rw-rw-r-- 1 hanzhou hanzhou 6.3K Oct 20 21:23 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.8K Oct 20 21:23 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:13 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 113K Oct 20 21:21 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  20K Oct 20 21:13 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n \"T=np.load\" /work/scratch/evaluate_output.py; python - <<'PY'
import numpy as np
A=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/scratch/T_final3.npy');print('maxdiff',np.abs(A-B).max(),'mean',np.abs(A-B).mean(),'allclose',np.allclose(A,B,atol=1e-6));print(A[200]);print(B[200])
PY"
```
<details><summary>output</summary>

```
maxdiff 0.15176269750222504 mean 0.005421366682212075 allclose False
[[ 0.77608091 -0.04067268  0.62932039  0.22648979]
 [-0.05233596 -0.99862953  0.          0.34946108]
 [ 0.62845793 -0.03293608 -0.77714596  2.98255759]
 [ 0.          0.          0.          1.        ]]
[[ 7.76080921e-01 -4.06726786e-02  6.29320377e-01  2.30168237e-01]
 [-5.23359576e-02 -9.98629535e-01  2.18504506e-10  3.47043779e-01]
 [ 6.28457915e-01 -3.29360847e-02 -7.77145973e-01  3.01051977e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -lc 'head -20 /work/scratch/evaluate_output.py; grep -n "object_poses" /work/scratch/evaluate_output.py'
```
<details><summary>output</summary>

```
import numpy as np,torch,cv2,trimesh,nvdiffrast.torch as dr,json,h5py
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda');tex=torch.tensor(np.asarray(mesh.visual.material.baseColorTexture),dtype=torch.float32,device='cuda')[None]/255
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);Tcw=torch.tensor(np.linalg.inv(np.array(cam['T_world_cam'])),dtype=torch.float32,device='cuda');gl=dr.RasterizeCudaContext();Tours=np.load('/work/scratch/T_final3.npy');Tbase=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def render(T,frame,img,obj,p …[+2070 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "sed \"s#Tours=np.load('/work/scratch/T_final3.npy')#Tours=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']#; s/eval_output.npy/eval_output_real.npy/; s/eval_base_output.npy/eval_base_real.npy/\" /work/scratch/evaluate_output.py > /work/scratch/evaluate_output_real.py
python /work/scratch/evaluate_output_real.py"
```
<details><summary>output</summary>

```
0 (0.7855325336241367, 0.23604933843777196, 0.8254335638797995) (0.0, 0.0, 0.0)
100 (0.7852738483859267, 0.23138896653721877, 0.8223681064973103) (0.0, 0.0, 0.0)
200 (0.8416622830089426, 0.24426462804155943, 0.8073989699017933) (0.0, 0.0, 0.0)
300 (0.8641490926924963, 0.23562364577397868, 0.822032584630099) (0.0, 0.0, 0.0)
400 (0.841042584434655, 0.23101255162914483, 0.7937487122817266) (0.0, 0.0, 0.0)
500 (0.836027713625866, 0.23411896703072163, 0.8188392904387759) (0.0, 0.0, 0.0)
600 (0.8467518045530261, 0.23912362849110683, 0.8178816076914339) (0.0, 0.0, 0.0)
700 (0.7537414965986394, 0.2517584636870345, 0.7753493454563132) (0.0, 0.0, 0.0)
800 (0.8569871969490602, 0.2468477973587521, 0.7470769708773588) (0.0, 0.0, 0.0)
static flat ours iou/l1/ncc [0.79887885 0.23212952 0.82374608] base [ …[+309 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def s(P):
 da=np.array([np.degrees(np.linalg.norm(Rot.from_matrix(P[i,:3,:3].T@P[i+1,:3,:3]).as_rotvec())) for i in range(len(P)-1)]);dt=np.linalg.norm(P[1:,:3,3]-P[:-1,:3,3],axis=1);return np.percentile(da,[50,90,99,100]),np.percentile(dt,[50,90,99,100])
print('ours',s(T));print('base',s(B));print('orth',np.abs(np.einsum('nij,nkj->nik',T[:,:3,:3],T[:,:3,:3])-np.eye(3)).max());print(T.shape,T.dtype,np.isfinite(T).all())
PY"
```
<details><summary>output</summary>

```
ours (array([0.        , 0.        , 2.05162828, 2.05163063]), array([0.00497884, 0.0156771 , 0.02829889, 0.04318018]))
base (array([1.61133992e-14, 2.13559363e+01, 9.61087185e+01, 1.53176933e+02]), array([0.00158025, 0.0666437 , 0.12098043, 0.3231664 ]))
orth 6.661338147750939e-16
(896, 4, 4) float64 True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read()
s=s.replace('| 0–159 static lying | **Result** | **0.785** | **0.230** | **0.831** |','| 0–159 static lying | **Result** | **0.799** | **0.232** | **0.824** |')
s=s.replace('| 160–659 sliding | **Result** | **0.855** | **0.237** | **0.814** |','| 160–659 sliding | **Result** | **0.840** | **0.239** | **0.811** |')
s=s.replace('| 660–794 stand-up | **Result** | **0.833** | **0.249** | **0.769** |','| 660–794 stand-up | **Result** | **0.828** | **0.250** | **0.769** |')
s=s.replace('| 795–895 static upright | **Result** | **0.848** | **0.248** | **0.751** |','| 795–895 static upright | **Result** | **0.856** | **0.248** | **0.750** |')
s=s.replace('| All 896 frames | **Result** | **0.839** | **0.239** | **0.803** |','| All 896 frames | **Result** | **0.833** | **0.241** | **0.800** |')
s=s.replace('silhouette IoU is limited mainly by SAM\\n  boundary differences, while RGB correlation is 0.83.','silhouette IoU is 0.799 and RGB correlation is 0.824.')
s=s.replace('The largest uncertainty is the\\n  metric depth, which is constrained by the known 0.305 m mesh and silhouette\\n  size rather than the unreliable object-only MoGe depth.','The largest uncertainty is the\\n  metric depth, which is constrained by the known 0.305 m mesh and silhouette\\n  size rather than the unreliable object-only MoGe depth; the pooled phase IoU\\n  is 0.840.')
s=s.replace('monotone trajectory and slerp path prevent isolate …[+419 chars]
```
<details><summary>output</summary>

```
## Validation without ground truth

Metrics below are computed on every frame against the corrected SAM2 iron mask.
Person-covered pixels are ignored when charging rendered false positives. RGB
values are measured only on the visible intersection of rendered and target
masks. Baseline and result use identical rendering, camera, mesh, masks, and
scoring code.

| Phase | Method | Mean mask IoU | Mean RGB L1 | Mean RGB NCC |
|---|---|---:|---:|---:|
| 0–159 static lying | Baseline | 0.000 | 0.000 | 0.000 |
| 0–159 static lying | **Result** | **0.799** | **0.232** | **0.824** |
| 160–659 sliding | Baseline | 0.000 | 0.000 | 0.000 |
| 160–659 sliding | **Result** | **0.840** | **0.239** | **0.811** |

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "python - <<'PY'
import shutil,os
p='/work/outputs/__pycache__'
if os.path.isdir(p): shutil.rmtree(p)
PY
find /work/outputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort
python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz';d=np.load(p)
print('keys',d.files)
T=d['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all(),T[:,3,:].min(),T[:,3,:].max())
print('det range',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
PY"
```
<details><summary>output</summary>

```
keys ['T_world_mesh']
(896, 4, 4) float64 True 0.0 1.0
det range 0.9999999999999992 1.0000000000000007

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n '/work/scratch\\|/pipeline\\|http' /work/outputs/solve.py || true"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n 'Result\\|Baseline\\|Confidence\\|Runtime' /work/outputs/REPORT.md; grep -n 'SHA\\|Git\\|git\\|PyTorch3D' /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
65:masks. Baseline and result use identical rendering, camera, mesh, masks, and
70:| 0–159 static lying | Baseline | 0.000 | 0.000 | 0.000 |
71:| 0–159 static lying | **Result** | **0.799** | **0.232** | **0.824** |
72:| 160–659 sliding | Baseline | 0.000 | 0.000 | 0.000 |
73:| 160–659 sliding | **Result** | **0.840** | **0.239** | **0.811** |
74:| 660–794 stand-up | Baseline | 0.000 | 0.000 | 0.000 |
75:| 660–794 stand-up | **Result** | **0.828** | **0.250** | **0.769** |
76:| 795–895 static upright | Baseline | 0.000 | 0.000 | 0.000 |
77:| 795–895 static upright | **Result** | **0.856** | **0.248** | **0.750** |
78:| All 896 frames | Baseline | 0.000 | 0.000 | 0.000 |
79:| All 896 frames | **Result** | **0.833** | **0.241** | **0.800** |
90:| Baseline | 0.000 / 21.36 / 96.11 / 153.18 | 0 …[+442 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -lc "grep -n \"0.833\\|0.848\\|worst\\|0.785\\|0.855\" /work/outputs/REPORT.md || true; sed -n '60,125p' /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
79:| All 896 frames | **Result** | **0.833** | **0.241** | **0.800** |
## Validation without ground truth

Metrics below are computed on every frame against the corrected SAM2 iron mask.
Person-covered pixels are ignored when charging rendered false positives. RGB
values are measured only on the visible intersection of rendered and target
masks. Baseline and result use identical rendering, camera, mesh, masks, and
scoring code.

| Phase | Method | Mean mask IoU | Mean RGB L1 | Mean RGB NCC |
|---|---|---:|---:|---:|
| 0–159 static lying | Baseline | 0.000 | 0.000 | 0.000 |
| 0–159 static lying | **Result** | **0.799** | **0.232** | **0.824** |
| 160–659 sliding | Baseline | 0.000 | 0.000 | 0.000 |
| 160–659 sliding | **Result** | **0.840** | **0.239** | **0.811** |
| 660–794 stand-up | Bas …[+2365 chars]
```
</details>

**Agent:** Completed the task and produced the required outputs:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, `(896, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, method, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — tools, versions, weights, hashes, dependencies

Key result: the supplied baseline mask had segmented the tripod behind the iron. The solver re-segments the iron with SAM2, recomputes lying/sliding poses, globally decodes the stand-up rotation, and imposes constant poses during both static phases.

Validation against the corrected iron mask:

- Mean mask IoU: **0.833**
- Mean visible RGB NCC: **0.800**
- Maximum adjacent-frame rotation step: **2.05°**
- Maximum adjacent-frame translation step: **0.0432 m**
- Final offline run time: **422.6 seconds**
