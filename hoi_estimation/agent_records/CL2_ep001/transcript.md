# CL2_ep001

Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T05:05:00Z

**Bash**
```
cat /work/AGENT_TASK.md; ls -la /work
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
and translation both matter; a  …[+11642 chars]
```
</details>

**Bash**
```
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude/projects/-work/memory/ 2>/dev/null; nvidia-smi; ls /pipeline /code/reconstruction /weights
```

<details><summary>output</summary>

```
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 08:58 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 08:58 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 08:58 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 08:58 proxy.sock

inputs:
total 79340
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 08:58 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 08:58 ..
-rw-rw-r-- 1 hanzhou hanzhou      826 Oct 21 08:58 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       60 Oct 21 08:58 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   190750 Oct 21 08:58 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      398 Oct 21 08:58 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  6623945 Oct 21 08:58 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 14688800 Oct 21 08:58 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 3790 …[+4719 chars]
```
</details>

**Bash**
```
cd /work; echo $CUDA_VISIBLE_DEVICES; cat logs/run_manifest.txt; cat /pipeline/pipeline_report.json | head -50; find /pipeline -maxdepth 3 | head -60
```

<details><summary>output</summary>

```
0
date_utc: 2026-10-21T05:05:00Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2_ep001:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000001:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
f86a72563a2cc08cfb49c636ce88b4b94984d03832f95ca01e16496c348d6e50  ./camera.json
6973fc3c45848f8543419296c5f44d458b057f0a32a1c78d59a75d8569c3093a  ./episode.json
7f88abfa45a2a83f841f3f9bbf3fd5aa4d999a15c031dc600dc03e …[+3993 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, trimesh, h5py
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m.vertices.shape, m.faces.shape, m.bounds, m.extents)
print(type(m.visual), getattr(m.visual,'material',None))
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
r=np.load('inputs/reconstruction.npz'); 
for k in r.files: print(k, r[k].shape, r[k].dtype)
f=h5py.File('inputs/masks.h5'); print({k:f[k].shape for k in f})
"; ls /pipeline/export/episode_000001/images | head -3; ls /pipeline/export/episode_000001/images | wc -l; python -c "import torch,nvdiffrast;print(torch.__version__)"; pip list 2>/dev/null | grep -i -E "torch|opencv|trimesh|open3d|kornia|scipy|nvdiff|pytorch3d|lietorch|roma"
```

<details><summary>output</summary>

```
(283294, 3) (94444, 3) [[-0.12395333 -0.07117505 -0.38321212]
 [ 0.12843189  0.07261943  0.66588682]] [0.25238522 0.14379448 1.04909894]
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7c380a706b50>
[[ 1.77420933e-03 -9.99998398e-01  2.36184807e-04  7.21731583e-04]
 [-9.99820737e-01 -1.76944178e-03  1.88510521e-02  9.27468548e-03]
 [-1.88506040e-02 -2.69588181e-04 -9.99822275e-01  1.40408676e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] [0.14402921 0.25239922 1.05070154]
T_world_mesh (747, 4, 4) float32
T_cam_mesh_foundationpose (747, 4, 4) float32
T_cam_mesh_coconet (747, 4, 4) float32
human_vertices_world (747, 4610, 3) float32
human_keypoints_world (747, 70, 3) float32
contact_prob_right_left (747, 2) float32
{' …[+879 chars]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/frames; python -c "
import cv2,numpy as np
cap=cv2.VideoCapture('inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
fr=[]
while True:
  ok,im=cap.read()
  if not ok: break
  fr.append(cv2.resize(im,(768,576),interpolation=cv2.INTER_AREA))
fr=np.stack(fr); print(fr.shape); np.save('scratch/frames/half.npy',fr)
"; df -h /work | tail -1; free -g | head -2; nproc
```

<details><summary>output</summary>

```
30.0 747.0
(747, 576, 768, 3)
10.127.102.71:/Group/hanzhou/v2d/agent_runs/CL2_ep001  139T   49T   90T  36% /work
               total        used        free      shared  buff/cache   available
Mem:             125          20           9           0          96         103
48
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, h5py, json
from scipy.spatial.transform import Rotation as R
r=np.load('inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
f=h5py.File('inputs/masks.h5')
om=f['object']; pm=f['person']
cp=r['contact_prob_right_left']
prev=None
for i in range(0,747,10):
  o=om[i]; a=o.sum(); ys,xs=np.nonzero(o)
  bb=(xs.min(),ys.min(),xs.max(),ys.max()) if a else None
  rot=R.from_matrix(T[i,:3,:3])
  dr = (R.from_matrix(T[i,:3,:3])*R.from_matrix(T[i-10,:3,:3]).inv()).magnitude()*57.3 if i>0 else 0
  print(i, 'area',a, bb, 'pa',pm[i].sum(), 'tc',np.round(Tc[i,:3,3],3), 'dR10',round(dr,1), 'cp',np.round(cp[i],2))
"

```

<details><summary>output</summary>

```
0 area 7958 (745, 603, 873, 942) pa 37853 tc [0.176 0.762 2.572] dR10 0 cp [0. 0.]
10 area 8072 (745, 603, 873, 943) pa 33010 tc [0.176 0.766 2.583] dR10 0.0 cp [0. 0.]
20 area 8045 (745, 603, 873, 943) pa 31765 tc [0.181 0.767 2.601] dR10 9.2 cp [0. 0.]
30 area 7824 (745, 605, 873, 943) pa 26049 tc [0.182 0.788 2.612] dR10 1.7 cp [0. 0.]
40 area 7650 (744, 603, 873, 943) pa 25055 tc [0.18  0.765 2.666] dR10 8.1 cp [0. 0.]
50 area 8010 (744, 603, 873, 943) pa 29991 tc [0.178 0.77  2.59 ] dR10 8.6 cp [0. 0.]
60 area 8011 (744, 603, 873, 943) pa 35510 tc [0.179 0.774 2.603] dR10 0.0 cp [0. 0.]
70 area 8014 (744, 603, 873, 943) pa 44483 tc [0.18  0.776 2.612] dR10 0.0 cp [0. 0.]
80 area 7997 (744, 603, 873, 943) pa 55449 tc [0.179 0.771 2.595] dR10 0.0 cp [0. 0.]
90 area 8007 (744, 603, 873,  …[+5889 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import trimesh,numpy as np
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual; print(v.uv.shape, v.material.baseColorTexture.size if v.material.baseColorTexture else None, v.material.baseColorFactor)
"; ls /code/reconstruction/modules /workspace; cat /code/reconstruction/CLAUDE.md | head -80
```

<details><summary>output</summary>

```
(283294, 2) (4171, 4189) None
/code/reconstruction/modules:
v2d_anycalib
v2d_bundlesdf
v2d_cari4d
v2d_common
v2d_cusfm
v2d_depth
v2d_depth_anything
v2d_detectron2
v2d_docker
v2d_droid_slam
v2d_ego_hand_reconstruction
v2d_face_detector
v2d_foundation_pose
v2d_foundation_stereo
v2d_geocalib
v2d_grounding_dino
v2d_gsplat_refinement
v2d_hamer
v2d_hand_alignment
v2d_hawor
v2d_hoi_object_reconstruction
v2d_mediapipe
v2d_mesh
v2d_moge
v2d_mv
v2d_mv_calibration
v2d_mv_postprocess
v2d_mv_preprocess
v2d_pipelines
v2d_rosbag
v2d_sam2
v2d_sam3d
v2d_sam3d_body
v2d_task_library_loader
v2d_unidepth
v2d_viz
v2d_wilor

/workspace:
third_party
v2d_cari4d
v2d_common
v2d_foundation_pose
v2d_mesh
v2d_moge
v2d_mv
v2d_sam2
v2d_sam3d_body
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) wh …[+2902 chars]
```
</details>

**Agent:** Let me write a shared rendering library and look at overlays of the baseline.

**Write**
```
/work/outputs/lib.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/lib.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/viz; cat > scratch/t_overlay.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
fr=np.load('/work/scratch/frames/half.npy',mmap_mode='r')
s=0.5; Ks=K.copy(); Ks[:2]*=s
H,W=fr.shape[1:3]
rd=Renderer('/work/inputs/object_mesh.glb',Ks,H,W)
T=rec['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)@T
ids=[int(a) for a in sys.argv[1:]]
ms=load_masks('/work/inputs',idx=ids)
for j,i in enumerate(ids):
    o=rd.render(Tc[i:i+1])
    m=o['mask'][0].cpu().numpy(); c=o['color'][0].cpu().numpy()
    om=ms['object'][j]
    iou=(m&om).sum()/max((m|om).sum(),1)
    im=fr[i].copy().astype(np.float32)
    ov=im.copy(); ov[m]=0.4*im[m]+0.6*c[m][:,::-1]*255
    ys,xs=np.nonzero(om|m); x0,x1,y0,y1=max(xs.min()-60,0),xs.max()+60,max(ys.min()-60,0),ys.max()+60
    cont=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    ov=ov.astype(np.uint8); cv2.drawContours(ov,cont,-1,(0,255,0),1)
    crop=np.concatenate([im[y0:y1,x0:x1].astype(np.uint8),ov[y0:y1,x0:x1]],1)
    crop=cv2.resize(crop,None,fx=2,fy=2)
    cv2.putText(crop,f'{i} iou {iou:.2f}',(5,20),0,0.6,(0,0,255),2)
    cv2.imwrite(f'/work/scratch/viz/ov_{i:04d}.png',crop); print(i,iou)
EOF
python scratch/t_overlay.py 0 130 200 300 400 600 700
```

<details><summary>output</summary>

```
0 0.46493414026343893
130 0.5396419437340153
200 0.3086900129701686
300 0.2283640424343942
400 0.3727175080558539
600 0.4961923847695391
700 0.40625
```
</details>

**Read**
```
/work/scratch/viz/ov_0000.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/viz/ov_0700.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/t_views.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from scipy.spatial.transform import Rotation as R
K=np.array([[800,0,200],[0,800,300],[0,0,1.]])
rd=Renderer('/work/inputs/object_mesh.glb',K,600,400)
c=rd.mesh.bounds.mean(0)
ims=[]
for ax,ang in [('y',0),('y',90),('y',180),('y',270),('x',90),('x',-90)]:
    Rm=R.from_euler(ax,ang,degrees=True).as_matrix()
    T=np.eye(4); T[:3,:3]=Rm; T[:3,3]=-Rm@c+np.array([0,0,3.0])
    o=rd.render(T[None]); im=(o['color'][0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8)
    im[~o['mask'][0].cpu().numpy()]=255
    cv2.putText(im,f'{ax}{ang}',(5,20),0,0.6,(0,0,0),2); ims.append(im)
cv2.imwrite('/work/scratch/viz/views.png',np.concatenate(ims,1))
EOF
python scratch/t_views.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/views.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/t_ov2.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
cam,K,Twc,rec=load_inputs('/work/inputs')
key=sys.argv[1]
if key=='base': T=rec['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)@T
else: Tc=np.load(key)['T_cam']
cap=cv2.VideoCapture('/work/inputs/video.mp4')
rd=Renderer('/work/inputs/object_mesh.glb',K,1152,1536)
ids=[int(a) for a in sys.argv[3:]]
ms=load_masks('/work/inputs',scale=1,idx=ids)
rows=[]
for j,i in enumerate(ids):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
    o=rd.render(Tc[i:i+1]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[:,:,::-1]*255)
    om=ms['object'][j]
    ys,xs=np.nonzero(om|m); cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
    hs=max(xs.max()-xs.min(),ys.max()-ys.min())//2+30
    x0,y0=max(cx-hs,0),max(cy-hs,0); x1,y1=x0+2*hs,y0+2*hs
    rimg=np.full_like(im,255); rimg[m]=c[m]
    ov=im.copy(); ov[m]=(0.5*im[m]+0.5*c[m]).astype(np.uint8)
    cont=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]; cv2.drawContours(ov,cont,-1,(0,255,0),1)
    row=np.concatenate([im[y0:y1,x0:x1],rimg[y0:y1,x0:x1],ov[y0:y1,x0:x1]],1)
    row=cv2.resize(row,(900,300)); cv2.putText(row,str(i),(5,20),0,0.7,(0,0,255),2); rows.append(row)
cv2.imwrite(sys.argv[2],np.concatenate(rows,0))
EOF
python scratch/t_ov2.py base scratch/viz/b1.png 0 60 115 125 135 145; python scratch/t_ov2.py base scratch/viz/b2.png 160 200 250 300 350 400
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

> *thinking:* Setup complete — frames decoded, renderer ready, and the episode shows three phases (resting ~0-115, carried to ~635, then resting again), with the main risk being a 180° flip around the vacuum's long axis. Now checking the baseline overlays.

**Read**
```
/work/scratch/viz/b1.png
```

**Read**
```
/work/scratch/viz/b2.png
```

<details><summary>output</summary>

```
[image]
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python -c "
import sys; sys.path.insert(0,'outputs')
from lib import *
cam,K,Twc,rec=load_inputs('inputs')
Tb=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(np.float64)
Tf=rec['T_cam_mesh_foundationpose'].astype(np.float64); Tn=rec['T_cam_mesh_coconet'].astype(np.float64)
a1=rot_angle(Tb[:,:3,:3],Tf[:,:3,:3]); a2=rot_angle(Tb[:,:3,:3],Tn[:,:3,:3]); a3=rot_angle(Tf[:,:3,:3],Tn[:,:3,:3])
d=rot_angle(Tb[1:,:3,:3],Tb[:-1,:3,:3])
for i in range(0,747,10): print(i, round(a1[i],1), round(a2[i],1), round(a3[i],1), 'tb',np.round(Tb[i,:3,3],2),'tf',np.round(Tf[i,:3,3],2),'tn',np.round(Tn[i,:3,3],2), 'maxstep', round(d[i:i+10].max(),1))
"

```

<details><summary>output</summary>

```
0 5.5 1.4 4.3 tb [0.18 0.76 2.57] tf [0.15 0.76 2.7 ] tn [0.15 0.73 2.65] maxstep 0.0
10 5.3 1.4 4.0 tb [0.18 0.77 2.58] tf [0.16 0.79 2.79] tn [0.15 0.73 2.65] maxstep 7.4
20 9.5 1.4 8.2 tb [0.18 0.77 2.6 ] tf [0.15 0.79 2.86] tn [0.15 0.72 2.66] maxstep 7.3
30 6.4 1.4 5.0 tb [0.18 0.79 2.61] tf [0.14 0.79 2.83] tn [0.15 0.74 2.66] maxstep 7.8
40 9.6 1.4 8.6 tb [0.18 0.76 2.67] tf [0.15 0.78 2.85] tn [0.15 0.72 2.71] maxstep 10.9
50 1.6 1.4 1.4 tb [0.18 0.77 2.59] tf [0.15 0.78 2.74] tn [0.15 0.72 2.62] maxstep 0.1
60 3.5 1.4 2.6 tb [0.18 0.77 2.6 ] tf [0.15 0.78 2.71] tn [0.15 0.72 2.62] maxstep 0.1
70 5.5 1.4 4.1 tb [0.18 0.78 2.61] tf [0.14 0.76 2.66] tn [0.15 0.72 2.62] maxstep 0.1
80 5.4 1.4 4.0 tb [0.18 0.77 2.59] tf [0.14 0.76 2.66] tn [0.15 0.72 2.62] maxstep 0.1
90 5.9 1.4 4.6 tb …[+6056 chars]
```
</details>

**Bash**
```
ls /workspace/v2d_sam2 /workspace/third_party /weights/sam2 /weights/foundationpose /weights/hf_home; python -c "import sam2; print(sam2.__file__)"; cat /pipeline/work/prompts.json | head -30
```

<details><summary>output</summary>

```
/weights/foundationpose:
nvlabs_pytorch

/weights/hf_home:
hub
xet

/weights/sam2:
README.md
config.json
model.safetensors
preprocessor_config.json
processor_config.json
sam2.1_hiera_l.yaml
sam2.1_hiera_large.pt
video_preprocessor_config.json

/workspace/third_party:
kaolin
nvdiffrast

/workspace/v2d_sam2:
LICENSE
__init__.py
assets
docker
lib
/opt/conda/lib/python3.11/site-packages/sam2/__init__.py
{
  "prompts": [
    {
      "frame_index": 0,
      "object_id": 0,
      "points": null,
      "point_labels": null,
      "box": {
        "x0": 837.010009765625,
        "y0": 378.51824951171875,
        "x1": 978.0130615234375,
        "y1": 896.3482666015625
      },
      "mask_path": null
    },
    {
      "frame_index": 0,
      "object_id": 1,
      "points": null,
      "point_label …[+199 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np,cv2
fr=np.load('scratch/frames/half.npy',mmap_mode='r')
ims=[]
for i in [0,700]:
  im=fr[i].copy()
  for x in range(0,768,50): cv2.line(im,(x,0),(x,575),(0,255,255),1); cv2.putText(im,str(x*2),(x,10),0,0.3,(0,255,255),1)
  for y in range(0,576,50): cv2.line(im,(0,y),(767,y),(0,255,255),1); cv2.putText(im,str(y*2),(0,y),0,0.3,(0,255,255),1)
  ims.append(im)
cv2.imwrite('scratch/viz/full0.png',np.concatenate(ims,1))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/full0.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/cache; cat > scratch/t_chair.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import glob
cfg='/weights/sam2/sam2.1_hiera_l.yaml'
model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/weights/sam2/sam2.1_hiera_large.pt',device='cuda')
pred=SAM2ImagePredictor(model)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
ms=load_masks('/work/inputs',scale=1,idx=[0,700])
out=[]
for j,i in enumerate([0,700]):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
    pred.set_image(im[:,:,::-1].copy())
    m,s,_=pred.predict(box=np.array([625,650,835,1005]),multimask_output=False)
    m=m[0]>0; m&=~ms['object'][j]
    print(i,s,m.sum()); out.append(m)
    v=im.copy(); v[m]=(0.5*v[m]+[0,0,127]).astype(np.uint8); cv2.imwrite(f'/work/scratch/viz/chair{i}.png',v[600:1050,550:900])
print('iou',(out[0]&out[1]).sum()/(out[0]|out[1]).sum())
EOF
python scratch/t_chair.py 2>&1 | tail -5
```

<details><summary>output</summary>

```
0 [0.9248642] 23778
700 [0.9257799] 24443
iou 0.9426718233824832
```
</details>

**Bash**
```
cd /work/scratch/viz; python -c "
import cv2,numpy as np; a=cv2.imread('chair0.png'); b=cv2.imread('chair700.png'); cv2.imwrite('chairs.png',np.concatenate([a,b],1))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/chairs.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Agent:** Chair mask from SAM2 looks clean (the chair barely moves between frames 0 and 700, IoU 0.94), so I can treat it as a static occluder. Next: a batched, crop-based render-and-compare fitter.

**Write**
```
/work/outputs/fit.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/fit.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work; cat > scratch/prep.py <<'EOF'
# cache full-res frames, masks, chair mask for experiments
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
fr=load_video('/work/inputs',scale=1.0); np.save('/work/scratch/cache/full.npy',fr)
f=h5py.File('/work/inputs/masks.h5','r')
np.save('/work/scratch/cache/obj.npy',f['object'][:]); np.save('/work/scratch/cache/person.npy',f['person'][:])
EOF
python scratch/prep.py; ls -la scratch/cache
```

<details><summary>output</summary>

```
total 6454112
drwxrwxr-x 2 hanzhou hanzhou       4096 Oct 21 09:05 .
drwxrwxr-x 7 hanzhou hanzhou       4096 Oct 21 09:03 ..
-rw-rw-r-- 1 hanzhou hanzhou 3965386880 Oct 21 09:05 full.npy
-rw-rw-r-- 1 hanzhou hanzhou 1321795712 Oct 21 09:05 obj.npy
-rw-rw-r-- 1 hanzhou hanzhou 1321795712 Oct 21 09:05 person.npy
```
</details>

**Bash**
```
cd /work; cat >> scratch/t_chair.py <<'EOF'
ch=out[0]|out[1]
np.save('/work/scratch/cache/chair.npy',ch)
EOF
python scratch/t_chair.py 2>&1 | tail -2
```

<details><summary>output</summary>

```
700 [0.9257799] 24443
iou 0.9426718233824832
```
</details>

**Bash**
```
cd /work; cat > scratch/common_exp.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from fit import *
cam,K,Twc,rec=load_inputs('/work/inputs')
FULL=np.load('/work/scratch/cache/full.npy',mmap_mode='r')
OBJ=np.load('/work/scratch/cache/obj.npy',mmap_mode='r')
PER=np.load('/work/scratch/cache/person.npy',mmap_mode='r')
CHAIR=np.load('/work/scratch/cache/chair.npy')
Tb=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(np.float64)
Tfp=rec['T_cam_mesh_foundationpose'].astype(np.float64); Tcn=rec['T_cam_mesh_coconet'].astype(np.float64)
kern=np.ones((7,7),np.uint8)
def ignore_mask(i):
    return cv2.dilate((PER[i]|CHAIR).astype(np.uint8),kern)>0
def make_fd(fit,ids,Tinit,R=192):
    boxes=[]
    for i,T in zip(ids,Tinit):
        corners=trimesh.bounds.corners(fit.mesh.bounds)
        uv=project(K,T,corners)
        boxes.append(box_from(OBJ[i],uv,pad=0.15))
    return FrameData([FULL[i] for i in ids],[OBJ[i] for i in ids],[ignore_mask(i) for i in ids],boxes,K,R)
EOF
cat > scratch/t_refine.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
import time
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ids=list(range(0,747,3))
t0=time.time()
fd=make_fd(fit,ids,Tb[ids])
print('prep',time.time()-t0)
R0,p0=fit.to_rp(Tb[ids])
with torch.no_grad():
    L0=losses(fit,torch.tensor(R0,dtype=torch.float32,device='cuda'),torch.tensor(p0,dtype=torch.float32,device='cuda'),fd,w_col=1.0)
t0=time.time()
R1,p1,L1=refine(fit,R0,p0,fd,iters=80)
print('refine',tim …[+446 chars]
```

<details><summary>output</summary>

```
prep 5.2178635597229
refine 11.910902261734009 4.593027072
0 iou 0.52->0.69 ncc 0.23->0.46 dR 10.4 dp 0.178 dz -0.175
9 iou 0.53->0.69 ncc 0.24->0.43 dR 14.0 dp 0.201 dz -0.199
18 iou 0.53->0.71 ncc 0.35->0.46 dR 11.8 dp 0.171 dz -0.169
27 iou 0.57->0.70 ncc 0.31->0.48 dR 13.3 dp 0.177 dz -0.173
36 iou 0.50->0.71 ncc 0.23->0.42 dR 15.2 dp 0.271 dz -0.264
45 iou 0.53->0.70 ncc 0.27->0.46 dR 13.5 dp 0.221 dz -0.218
54 iou 0.52->0.70 ncc 0.28->0.39 dR 12.0 dp 0.216 dz -0.211
63 iou 0.52->0.71 ncc 0.27->0.32 dR 11.5 dp 0.269 dz -0.265
72 iou 0.52->0.70 ncc 0.28->0.35 dR 12.0 dp 0.264 dz -0.260
81 iou 0.52->0.71 ncc 0.28->0.30 dR 18.5 dp 0.294 dz -0.291
90 iou 0.52->0.71 ncc 0.28->0.34 dR 12.9 dp 0.257 dz -0.254
99 iou 0.53->0.70 ncc 0.28->0.30 dR 15.0 dp 0.257 dz -0.254
108 iou 0.48->0.70 ncc  …[+4347 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np,cv2
F=np.load('scratch/cache/full.npy',mmap_mode='r'); O=np.load('scratch/cache/obj.npy',mmap_mode='r')
rows=[]
for i in [60,700]:
  im=F[i].copy(); c=cv2.findContours(O[i].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]; cv2.drawContours(im,c,-1,(0,255,0),1)
  rows.append(cv2.resize(im[600:950,700:900],None,fx=2.5,fy=2.5,interpolation=cv2.INTER_NEAREST))
cv2.imwrite('scratch/viz/maskzoom.png',np.concatenate(rows,1))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/maskzoom.png
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* Free 6-DoF refinement boosts mean mask IoU from 0.45 to 0.65 but shifts the object ~20 cm toward the camera each frame—likely a mask/cord bias rather than true depth change, so I'll keep depth pinned to baseline. Next I'll fit one shared pose per static phase using multiple hypotheses, including symmetry flips.

**Bash**
```
cd /work; cat >> outputs/fit.py <<'EOF'


def refine_grouped(fit, R0, p0, fd, group, iters=80, lr_r=0.02, lr_t=0.01, w_dt=2.0, w_col=0.0,
                   fix_depth=True, frame_w=None):
    """Like refine but parameters are shared within groups. R0,p0: per-group init (G,..);
    group: (B,) index of the group of each frame. Depth (camera z of the centre) kept fixed if fix_depth."""
    dev = fit.dev
    g = torch.as_tensor(np.asarray(group), device=dev, dtype=torch.long)
    R0t = torch.tensor(R0, dtype=torch.float32, device=dev)
    p0t = torch.tensor(p0, dtype=torch.float32, device=dev)
    G = len(p0)
    w = torch.zeros(G, 3, device=dev, requires_grad=True)
    dp = torch.zeros(G, 3, device=dev, requires_grad=True)
    fw = torch.ones(len(group), device=dev) if frame_w is None else torch.as_tensor(frame_w, device=dev, dtype=torch.float32)
    opt = torch.optim.Adam([{"params": [w], "lr": lr_r}, {"params": [dp], "lr": lr_t}])

    def cur():
        Rm = so3_exp(w) @ R0t
        d = torch.cat([dp[:, :2], torch.zeros_like(dp[:, 2:])], 1) if fix_depth else dp
        # keep image position consistent when moving x,y at fixed z
        return Rm, p0t + d

    for it in range(iters):
        Rm, p = cur()
        L = losses(fit, Rm[g], p[g], fd, w_col)
        loss = (1 - L["iou"]) + w_dt * L["dt"]
        if w_col > 0:
            loss = loss + w_col * (1 - L["ncc"])
        opt.zero_grad()
        (loss * fw).sum().backward()
        opt.step()
    with torch.no_grad():
    …[+1739 chars]
```

<details><summary>output</summary>

```
base wcol 0.0 iou 0.673 ncc 0.436 dt 0.0007 dR_from_init 19.9 dR_from_base 19.9 dp [-0.009  0.004 -0.   ]
base wcol 0.5 iou 0.647 ncc 0.492 dt 0.0007 dR_from_init 15.1 dR_from_base 15.1 dp [-0.007  0.002 -0.   ]
flipx wcol 0.0 iou 0.568 ncc 0.088 dt 0.0024 dR_from_init 31.0 dR_from_base 173.1 dp [ 0.013  0.025 -0.   ]
flipx wcol 0.5 iou 0.487 ncc 0.203 dt 0.0023 dR_from_init 36.7 dR_from_base 173.8 dp [-0.008  0.008 -0.   ]
flipy wcol 0.0 iou 0.493 ncc 0.072 dt 0.0039 dR_from_init 24.9 dR_from_base 171.5 dp [-0.009  0.026 -0.   ]
flipy wcol 0.5 iou 0.382 ncc 0.100 dt 0.0067 dR_from_init 32.2 dR_from_base 152.9 dp [-0.009 -0.043 -0.   ]
flipz wcol 0.0 iou 0.552 ncc -0.168 dt 0.0032 dR_from_init 27.6 dR_from_base 154.6 dp [ 0.001 -0.013 -0.   ]
flipz wcol 0.5 iou 0.212 ncc 0.237 dt 0.0320 dR …[+1113 chars]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np
cam_T=None
for a,name in [(0,'base_0.0'),(0,'base_0.5'),(640,'base_0.0')]:
    T=np.load(f'scratch/static_{a}_{name}.npy')
    Tc=np.repeat(T[None],747,0); np.savez(f'scratch/st_{a}_{name}.npz',T_cam=Tc)
EOF
python scratch/t_ov2.py scratch/st_0_base_0.0.npz scratch/viz/s0a.png 20 60 100; python scratch/t_ov2.py base scratch/viz/s0b.png 20 60 100; python scratch/t_ov2.py scratch/st_640_base_0.0.npz scratch/viz/s1a.png 660 700; python scratch/t_ov2.py base scratch/viz/s1b.png 660 700
python -c "
import cv2,numpy as np
for a,b,o in [('s0a','s0b','s0'),('s1a','s1b','s1')]:
  cv2.imwrite(f'/work/scratch/viz/{o}.png',np.concatenate([cv2.imread(f'/work/scratch/viz/{a}.png')[:,300:],cv2.imread(f'/work/scratch/viz/{b}.png')[:,300:]],1))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/s0.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/viz/s1.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python -c "
import h5py,numpy as np,pickle
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def p(n,o):
  if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype)
f.visititems(p)
print(dict(f.attrs))
k=pickle.load(open('/pipeline/depth/moge2/intrinsics.pkl','rb')); print(type(k), k if not hasattr(k,'shape') else k.shape)
"; cat /pipeline/depth/moge2/depth_report.json | head -40; python -c "import numpy as np; a=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r'); print(a.shape,a.dtype)"
```

<details><summary>output</summary>

```
<persisted-output>
Output too large (77KB). Full output saved to: /work/scratch/.claude/projects/-work/80b7169f-f370-4f06-ac96-e912fe20c019/tool-results/bdhhb3116.txt

Preview (first 2KB):
aligned/front_stereo_camera_left (747,) object
alignment/front_stereo_camera_left/scale (747,) float32
alignment/front_stereo_camera_left/shift (747,) float32
alignment/front_stereo_camera_left/valid_count (747,) int32
frame_names/front_stereo_camera_left (747,) object
raw/front_stereo_camera_left (747,) object
{'complete': True, 'depth_alignment_input_identity_json': '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-normal","depth_model_revision":"b135031bae30b5ac2ae141a0e68717795ce38340","depth_source_commit":"925b8ed835a7a9cdb7578ba15c658a0afc969030","direct_mhr":{"mtime_ns":17912008340 …[+747 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, cv2, h5py
a=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r'); print(a.shape,a.dtype)
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
x=f['aligned/front_stereo_camera_left'][0]; print(type(x), len(x))
im=cv2.imdecode(np.frombuffer(x,np.uint8),cv2.IMREAD_UNCHANGED); print(im.shape,im.dtype,im.min(),im.max())
print(f['alignment/front_stereo_camera_left/scale'][:10], f['alignment/front_stereo_camera_left/shift'][:10])
print(a[0].min(),a[0].max(), a[0][700,800])
"
```

<details><summary>output</summary>

```
(747, 1152, 1536) uint16
<class 'numpy.ndarray'> 676571
(1152, 1536) uint16 1636 10016
[1.1155448 1.1191528 1.1380609 1.1507937 1.1431127 1.1507826 1.1561503
 1.1427715 1.1445017 1.1609173] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
1467 8979 5119
```
</details>

**Bash**
```
cd /work; cat > scratch/t_geo.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
import json
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n,d=pl[:3],pl[3]
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=np.asarray(m.vertices)
def info(Tc,tag):
    Tw=Twc@Tc; X=V@Tw[:3,:3].T+Tw[:3,3]; s=X@n+d
    ax=Tc[:3,:3]@np.array([0,0,1.])
    print(tag,'floor signed dist min %.3f (pctl1 %.3f)'%(s.min(),np.percentile(s,0.5)),'axis_cam',np.round(ax,3),'z %.3f'%Tc[2,3])
for i in [0,60,100]: info(Tb[i],f'base{i}')
info(np.load('scratch/static_0_base_0.0.npy'),'ref0'); info(np.load('scratch/static_0_base_0.5.npy'),'ref0c')
for i in [660,700]: info(Tb[i],f'base{i}')
info(np.load('scratch/static_640_base_0.0.npy'),'ref640')
# sign: which side is "above"? check human feet
hv=rec['human_vertices_world'][0]; print('human min/max dist',(hv@n+d).min(),(hv@n+d).max())
EOF
python scratch/t_geo.py
```

<details><summary>output</summary>

```
base0 floor signed dist min -0.021 (pctl1 -0.012) axis_cam [-0.284 -0.959  0.016] z 2.572
base60 floor signed dist min -0.034 (pctl1 -0.026) axis_cam [-0.295 -0.955 -0.019] z 2.603
base100 floor signed dist min -0.027 (pctl1 -0.019) axis_cam [-0.295 -0.955 -0.019] z 2.584
ref0 floor signed dist min -0.030 (pctl1 -0.024) axis_cam [-0.306 -0.952  0.014] z 2.588
ref0c floor signed dist min -0.028 (pctl1 -0.022) axis_cam [-0.306 -0.952  0.004] z 2.590
base660 floor signed dist min -0.005 (pctl1 0.001) axis_cam [-0.212 -0.958 -0.192] z 2.631
base700 floor signed dist min -0.021 (pctl1 -0.015) axis_cam [-0.212 -0.958 -0.192] z 2.675
ref640 floor signed dist min -0.027 (pctl1 -0.022) axis_cam [-0.24  -0.956 -0.165] z 2.649
human min/max dist -0.019747420502842505 1.7096733986330344
```
</details>

> *thinking:* The shared-pose fit keeps the long-axis within ~2° of baseline across both static phases, mainly varying roll (15–20° in phase 1, 8° in phase 2), with the vacuum's foot sitting on the floor plane within 2–3 cm—consistent with the human's feet. Next I'll check whether the reconstructed left hand tracks the handle while the vacuum is carried.

**Bash**
```
cd /work; cat > scratch/t_hand.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=np.asarray(m.vertices); top=V[V[:,2]>0.60].mean(0); print('top',top)
kp=rec['human_keypoints_world'].astype(np.float64)
Tw=rec['T_world_mesh'].astype(np.float64)
gw=np.einsum('fij,j->fi',Tw[:,:3,:3],top)+Tw[:,:3,3]
for name,rng in [('R',range(21,42)),('L',range(42,63))]:
    h=kp[:,list(rng)].mean(1)
    dd=np.linalg.norm(h-gw,axis=1)
    # camera frame depth comparison
    hc=(np.linalg.inv(Twc)[:3,:3]@h.T).T+np.linalg.inv(Twc)[:3,3]; gc=(np.linalg.inv(Twc)[:3,:3]@gw.T).T+np.linalg.inv(Twc)[:3,3]
    print(name,[ (i,round(dd[i],3), round(hc[i,2]-gc[i,2],3)) for i in range(100,660,20)])
EOF
python scratch/t_hand.py
```

<details><summary>output</summary>

```
top [ 4.67110069e-04 -2.01432951e-02  6.34324025e-01]
R [(100, 0.805, -0.287), (120, 0.842, -0.194), (140, 0.745, -0.428), (160, 0.727, -0.579), (180, 0.673, -0.584), (200, 0.749, -0.691), (220, 0.479, -0.445), (240, 0.407, -0.229), (260, 0.754, -0.238), (280, 0.713, -0.176), (300, 0.718, -0.36), (320, 0.557, -0.534), (340, 0.559, -0.514), (360, 0.681, -0.621), (380, 0.418, -0.373), (400, 0.272, -0.233), (420, 0.307, 0.0), (440, 0.396, 0.029), (460, 0.534, -0.055), (480, 0.636, -0.242), (500, 0.657, -0.626), (520, 0.768, -0.738), (540, 0.802, -0.725), (560, 0.502, -0.278), (580, 0.132, -0.101), (600, 0.109, -0.082), (620, 0.105, -0.09), (640, 0.295, 0.066)]
L [(100, 0.069, 0.049), (120, 0.123, -0.122), (140, 0.028, -0.0), (160, 0.124, -0.095), (180, 0.036, -0.032), (200, 0.062, -0.055), (2 …[+472 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/t_hyp.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
from scipy.spatial.transform import Rotation as Rot
import time
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
F=747; ids=np.arange(F)
Rb,pb=fit.to_rp(Tb); Rf,_=fit.to_rp(Tfp)
def flip(Rm,ax):
    e=np.zeros(3); e[ax]=np.pi; return Rm@Rot.from_rotvec(e).as_matrix()
H={'base':Rb,'fp':Rf,'base_fz':flip(Rb,2),'fp_fz':flip(Rf,2),'base_fy':flip(Rb,1)}
res={}
t0=time.time()
for c0 in range(0,F,125):
    cid=ids[c0:c0+125]
    fd=make_fd(fit,cid,Tb[cid])
    for k,Rh in H.items():
        R1,p1,L=refine(fit,Rh[cid],pb[cid],fd,iters=80,w_col=0.3,fix_depth=True)
        r=res.setdefault(k,{'R':[],'p':[],'iou':[],'ncc':[],'dt':[]})
        r['R'].append(R1); r['p'].append(p1); r['iou'].append(L['iou']); r['ncc'].append(L['ncc']); r['dt'].append(L['dt'])
    print(c0,time.time()-t0,flush=True)
out={}
for k,r in res.items():
    for q in r: out[f'{k}/{q}']=np.concatenate(r[q])
np.savez('/work/scratch/hyp.npz',**out)
EOF
python scratch/t_hyp.py
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bzhmdp66k). Output is being written to: /tmp/claude-1003/-work/80b7169f-f370-4f06-ac96-e912fe20c019/tasks/bzhmdp66k.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 100; cat /tmp/claude-1003/-work/80b7169f-f370-4f06-ac96-e912fe20c019/tasks/bzhmdp66k.output
```

<details><summary>output</summary>

```
0 37.218406200408936
125 75.41797471046448
250 107.46678733825684
375 139.6145203113556
500 176.70679688453674
625 213.65192532539368

[exited with code 0]
```
</details>

**Bash**
```
cd /work; cat > scratch/t_ana.py <<'EOF'
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from lib import rot_angle
h=np.load('/work/scratch/hyp.npz'); ks=['base','fp','base_fz','fp_fz','base_fy']
S=np.stack([h[f'{k}/iou']+0.3*h[f'{k}/ncc']-2*h[f'{k}/dt'] for k in ks],1)
R=np.stack([h[f'{k}/R'] for k in ks],1)
best=S.argmax(1)
for i in range(0,747,8):
    print(i, ' '.join('%s:%.2f/%.2f'%(k[:7],h[f'{k}/iou'][i],h[f'{k}/ncc'][i]) for k in ks), 'best',ks[best[i]], 'ang(base,fp)=%.0f ang(best,base)=%.0f'%(rot_angle(R[i,0],R[i,1]),rot_angle(R[i,best[i]],R[i,0])))
EOF
python scratch/t_ana.py
```

<details><summary>output</summary>

```
0 base:0.66/0.48 fp:0.65/0.51 base_fz:0.38/0.14 fp_fz:0.42/-0.16 base_fy:0.40/0.16 best base ang(base,fp)=1 ang(best,base)=0
8 base:0.64/0.46 fp:0.63/0.48 base_fz:0.50/-0.04 fp_fz:0.51/-0.19 base_fy:0.36/0.14 best fp ang(base,fp)=1 ang(best,base)=1
16 base:0.65/0.53 fp:0.58/0.40 base_fz:0.43/0.14 fp_fz:0.49/0.14 base_fy:0.45/0.07 best base ang(base,fp)=2 ang(best,base)=0
24 base:0.66/0.51 fp:0.65/0.52 base_fz:0.25/0.21 fp_fz:0.40/0.18 base_fy:0.39/0.18 best base ang(base,fp)=1 ang(best,base)=0
32 base:0.64/0.52 fp:0.63/0.53 base_fz:0.39/0.17 fp_fz:0.41/0.08 base_fy:0.47/0.16 best base ang(base,fp)=3 ang(best,base)=0
40 base:0.58/0.42 fp:0.64/0.41 base_fz:0.07/0.38 fp_fz:0.39/-0.01 base_fy:0.37/0.11 best fp ang(base,fp)=4 ang(best,base)=4
48 base:0.64/0.48 fp:0.67/0.45 base_fz:0.42/-0.04 fp …[+11136 chars]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
p='/work/outputs/fit.py'; s=open(p).read()
s=s.replace("    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)\n","    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)\n")
open(p,'w').write(s)
EOF
grep -n "sqrt((w" outputs/fit.py
```

<details><summary>output</summary>

```
11:    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
```
</details>

**Bash**
```
cd /work; cat > scratch/t_cmp.py <<'EOF'
# usage: t_cmp.py out.png hyps.npz key1,key2,.. frames...
import sys; sys.path.insert(0,'/work/outputs')
from lib import *
from fit import CropFitter
cam,K,Twc,rec=load_inputs('/work/inputs')
FULL=np.load('/work/scratch/cache/full.npy',mmap_mode='r'); OBJ=np.load('/work/scratch/cache/obj.npy',mmap_mode='r')
out,hp,keys=sys.argv[1],sys.argv[2],sys.argv[3].split(',')
ids=[int(a) for a in sys.argv[4:]]
h=np.load(hp)
fit=CropFitter('/work/inputs/object_mesh.glb',res=64)
rd=Renderer('/work/inputs/object_mesh.glb',K,1152,1536)
def getT(k,i):
    if k=='base0': return np.linalg.inv(Twc)@rec['T_world_mesh'][i].astype(np.float64)
    if k+'/T' in h: return h[k+'/T'][i]
    return fit.to_T(h[k+'/R'][i:i+1],h[k+'/p'][i:i+1])[0]
rows=[]
for i in ids:
    im=FULL[i]; om=OBJ[i]
    ys,xs=np.nonzero(om); cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
    hs=max(xs.max()-xs.min(),ys.max()-ys.min())//2+40
    x0,y0=max(cx-hs,0),max(cy-hs,0); x1,y1=x0+2*hs,y0+2*hs
    cols=[im[y0:y1,x0:x1]]
    cont=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    for k in keys:
        T=getT(k,i); o=rd.render(T[None]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[:,:,::-1]*255)
        r=np.full_like(im,255); r[m]=c[m]; cv2.drawContours(r,cont,-1,(0,200,0),1)
        cols.append(r[y0:y1,x0:x1])
    row=np.concatenate([cv2.resize(c,(260,260)) for c in cols],1)
    cv2.putText(row,str(i),(5,20),0,0.7,(0,0,255),2); …[+172 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/c1.png
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* I found a real baseline failure around frames 256–296: the image shows the vacuum's back view while the baseline shows the front, a 180° flip my per-channel color metric missed since normalization erased the red-vs-grey difference. I'm switching to a chromaticity-based color score to catch this.

**Bash**
```
cd /work; cat >> outputs/fit.py <<'EOF'


def chroma(x):
    s = x.sum(-1, keepdim=True) + 0.05
    return x / s


def chroma_score(col, img, w):
    """mean L1 distance of rg-chromaticity (gain invariant) between render and image over weight w.
    returns (B,) distance (lower better) and the weight mass."""
    d = (chroma(col) - chroma(img)).abs().sum(-1)
    ws = w.sum((1, 2))
    return (d * w).sum((1, 2)) / ws.clamp_min(1.0), ws
EOF
cat > scratch/t_chroma.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
h=np.load('/work/scratch/hyp.npz'); ks=['base','fp','base_fz','fp_fz','base_fy']
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ids=list(range(0,747,8))
fd=make_fd(fit,ids,Tb[ids])
for k in ks:
    Rm=torch.tensor(np.nan_to_num(h[k+'/R'][ids]),dtype=torch.float32,device='cuda'); p=torch.tensor(h[k+'/p'][ids],dtype=torch.float32,device='cuda')
    with torch.no_grad():
        o=fit.render(Rm,p,fd.K); w=o['hard']*fd.M*fd.V
        d,ws=chroma_score(o['color'],fd.img,w)
    h_=dict(np.load('/work/scratch/hyp.npz')) if False else None
    globals()['C_'+k]=d.cpu().numpy()
for j,i in enumerate(ids):
    print(i,' '.join('%s:%.3f'%(k[:7],globals()['C_'+k][j]) for k in ks))
EOF
python scratch/t_chroma.py 2>&1 | sed -n '25,45p'
```

<details><summary>output</summary>

```
192 base:0.241 fp:0.222 base_fz:0.198 fp_fz:0.198 base_fy:0.246
200 base:0.200 fp:0.205 base_fz:0.175 fp_fz:0.239 base_fy:0.277
208 base:0.218 fp:0.209 base_fz:0.188 fp_fz:0.189 base_fy:0.275
216 base:0.254 fp:0.277 base_fz:0.171 fp_fz:0.244 base_fy:0.210
224 base:0.251 fp:0.314 base_fz:0.271 fp_fz:0.245 base_fy:0.195
232 base:0.240 fp:0.204 base_fz:0.857 fp_fz:0.168 base_fy:0.300
240 base:0.248 fp:0.234 base_fz:0.218 fp_fz:0.238 base_fy:0.292
248 base:0.253 fp:0.250 base_fz:0.181 fp_fz:0.158 base_fy:0.225
256 base:0.292 fp:0.302 base_fz:0.159 fp_fz:0.147 base_fy:0.301
264 base:0.202 fp:0.284 base_fz:0.198 fp_fz:0.170 base_fy:0.229
272 base:0.178 fp:0.281 base_fz:0.250 fp_fz:0.238 base_fy:0.166
280 base:0.158 fp:0.344 base_fz:0.327 fp_fz:0.167 base_fy:0.164
288 base:0.182 fp:0.313 base_fz: …[+543 chars]
```
</details>

**Bash**
```
cd /work; python scratch/t_cmp.py scratch/viz/c2.png scratch/hyp.npz base0,base,fp,base_fz,fp_fz 160 184 200 216 232 248
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/c2.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
p='/work/outputs/fit.py'; s=open(p).read()
old='''    if w_col > 0:
        w = out["hard"] * M * V
        res["ncc"] = color_residual(out["color"], fd.img, w)
        res["color"] = out["color"]
    return res'''
new='''    if w_col > 0:
        w = out["hard"] * M * V
        res["chr"], _ = chroma_score(out["color"], fd.img, w)
        res["color"] = out["color"]
    return res'''
assert old in s; s=s.replace(old,new)
s=s.replace('''            loss = loss + w_col * (1 - L["ncc"])''','''            loss = loss + w_col * L["chr"]''')
s=s.replace('''if k in ("iou", "dt", "ncc")}''','''if k in ("iou", "dt", "chr")}''')
open(p,'w').write(s)
EOF
grep -n 'chr' outputs/fit.py
```

<details><summary>output</summary>

```
171:        res["chr"], _ = chroma_score(out["color"], fd.img, w)
196:            loss = loss + w_col * L["chr"]
210:        {k: v.detach().cpu().numpy() for k, v in L.items() if k in ("iou", "dt", "chr")}
238:            loss = loss + w_col * L["chr"]
246:        {k: v.cpu().numpy() for k, v in L.items() if k in ("iou", "dt", "chr")}
249:def chroma(x):
254:def chroma_score(col, img, w):
255:    """mean L1 distance of rg-chromaticity (gain invariant) between render and image over weight w.
257:    d = (chroma(col) - chroma(img)).abs().sum(-1)
```
</details>

**Bash**
```
cd /work; cat > scratch/t_static2.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
from scipy.spatial.transform import Rotation as Rot
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
import sys
a,b=int(sys.argv[1]),int(sys.argv[2])
ids=list(range(a,b,4))
Rb,pb=fit.to_rp(Tb[ids])
Rm0=Rot.from_matrix(Rb).mean().as_matrix(); pm0=np.median(pb,0)
T0=fit.to_T(Rm0[None],pm0[None])[0]
fd=make_fd(fit,ids,[T0]*len(ids))
hyps={f'rz{d}':Rm0@Rot.from_euler('z',d,degrees=True).as_matrix() for d in [0,-30,30,90,180,270]}
hyps['fx']=Rm0@Rot.from_euler('x',180,degrees=True).as_matrix(); hyps['fy']=Rm0@Rot.from_euler('y',180,degrees=True).as_matrix()
for name,Rh in hyps.items():
  for wc in [1.0]:
    R1,p1,L=refine_grouped(fit,Rh[None],pm0[None],fd,np.zeros(len(ids),int),iters=100,w_col=wc)
    print(name,'wc',wc,'iou %.3f chr %.3f dt %.4f'%(L['iou'].mean(),L['chr'].mean(),L['dt'].mean()),'dR_base %.1f'%rot_angle(R1[0],Rm0),'roll-ish: axis dev %.1f'%np.degrees(np.arccos(np.clip(R1[0][:,2]@Rm0[:,2],-1,1))),'dp',np.round(p1[0]-pm0,3))
    np.save(f'/work/scratch/static2_{a}_{name}_{wc}.npy',fit.to_T(R1,p1)[0])
with torch.no_grad():
    L=losses(fit,torch.tensor(Rb,dtype=torch.float32,device='cuda'),torch.tensor(pb,dtype=torch.float32,device='cuda'),fd,w_col=1)
print('baseline perframe iou %.3f chr %.3f'%(L['iou'].mean(),L['chr'].mean()))
EOF
python scratch/t_static2.py 0 112; python scratch/t_static2.py 640 747
```

<details><summary>output</summary>

```
rz0 wc 1.0 iou 0.136 chr 0.357 dt 0.0180 dR_base 21.0 roll-ish: axis dev 13.3 dp [-0.093 -0.039 -0.   ]
rz-30 wc 1.0 iou 0.230 chr 0.288 dt 0.0179 dR_base 37.0 roll-ish: axis dev 10.4 dp [-0.062 -0.05  -0.   ]
rz30 wc 1.0 iou 0.357 chr 0.200 dt 0.0068 dR_base 24.9 roll-ish: axis dev 8.6 dp [-0.049  0.06  -0.   ]
rz90 wc 1.0 iou 0.446 chr 0.223 dt 0.0037 dR_base 77.5 roll-ish: axis dev 18.5 dp [ 0.014 -0.012 -0.   ]
rz180 wc 1.0 iou 0.123 chr 0.358 dt 0.0264 dR_base 161.1 roll-ish: axis dev 4.6 dp [ 0.046 -0.111 -0.   ]
rz270 wc 1.0 iou 0.258 chr 0.314 dt 0.0275 dR_base 67.9 roll-ish: axis dev 24.4 dp [ 0.11   0.052 -0.   ]
fx wc 1.0 iou 0.192 chr 0.211 dt 0.0282 dR_base 164.6 roll-ish: axis dev 162.2 dp [ 0.074  0.013 -0.   ]
fy wc 1.0 iou 0.308 chr 0.312 dt 0.0049 dR_base 162.8 roll-ish:  …[+953 chars]
```
</details>

**Bash**
```
cd /work; sed -i 's/  for wc in \[1.0\]:/  for wc in [0.0]:/' scratch/t_static2.py; python scratch/t_static2.py 0 112; python scratch/t_static2.py 640 747
```

<details><summary>output</summary>

```
rz0 wc 0.0 iou 0.679 chr 0.200 dt 0.0007 dR_base 21.6 roll-ish: axis dev 3.4 dp [-0.011 -0.002 -0.   ]
rz-30 wc 0.0 iou 0.682 chr 0.193 dt 0.0006 dR_base 20.8 roll-ish: axis dev 0.9 dp [-0.012  0.001 -0.   ]
rz30 wc 0.0 iou 0.656 chr 0.205 dt 0.0006 dR_base 29.9 roll-ish: axis dev 1.6 dp [-0.007  0.016 -0.   ]
rz90 wc 0.0 iou 0.511 chr 0.240 dt 0.0011 dR_base 97.0 roll-ish: axis dev 25.7 dp [ 0.035  0.143 -0.   ]
rz180 wc 0.0 iou 0.561 chr 0.226 dt 0.0028 dR_base 154.6 roll-ish: axis dev 9.5 dp [ 0.    -0.016 -0.   ]
rz270 wc 0.0 iou 0.633 chr 0.250 dt 0.0013 dR_base 81.4 roll-ish: axis dev 10.0 dp [-0.006 -0.018 -0.   ]
fx wc 0.0 iou 0.562 chr 0.260 dt 0.0025 dR_base 173.0 roll-ish: axis dev 162.6 dp [ 0.01  0.02 -0.  ]
fy wc 0.0 iou 0.494 chr 0.291 dt 0.0038 dR_base 172.6 roll-ish: axis  …[+944 chars]
```
</details>

> *thinking:* Static phase poses are finalized with silhouette optimization—mean IoU improved to 0.68 (phase 1) and 0.66 (phase 2), mostly via roll correction. Now running the multi-hypothesis pass for the carried phase.

**Bash**
```
cd /work; cat > scratch/t_hyp2.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
from scipy.spatial.transform import Rotation as Rot
import time
fit=CropFitter('/work/inputs/object_mesh.glb',res=128)
F=747; ids=np.arange(100,656)
Rb,pb=fit.to_rp(Tb); Rf,_=fit.to_rp(Tfp)
H={}
for src,Rs in [('b',Rb),('f',Rf)]:
    for d in [0,90,180,270]:
        H[f'{src}{d}']=Rs@Rot.from_euler('z',d,degrees=True).as_matrix()
res={}
t0=time.time()
for c0 in range(0,len(ids),140):
    cid=ids[c0:c0+140]
    fd=make_fd(fit,cid,Tb[cid],R=128)
    for k,Rh in H.items():
        R1,p1,L=refine(fit,Rh[cid],pb[cid],fd,iters=50,w_col=0.0,fix_depth=True,lr_r=0.03)
        r=res.setdefault(k,{'R':[],'p':[],'iou':[],'chr':[],'dt':[]})
        for q,v in [('R',R1),('p',p1),('iou',L['iou']),('chr',L['chr']),('dt',L['dt'])]: r[q].append(v)
    print(c0,time.time()-t0,flush=True)
out={'ids':ids}
for k,r in res.items():
    for q in r: out[f'{k}/{q}']=np.concatenate(r[q])
np.savez('/work/scratch/hyp2.npz',**out)
EOF
python scratch/t_hyp2.py
```

<details><summary>output</summary>

```
0 23.408385038375854
140 45.96871256828308
280 70.5170328617096
420 92.45448899269104
```
</details>

**Bash**
```
cd /work; cat > scratch/t_vit.py <<'EOF'
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from lib import rot_angle
h=np.load('/work/scratch/hyp2.npz'); ids=h['ids']
ks=[k[:-2] for k in h.files if k.endswith('/R')]
R=np.stack([np.nan_to_num(h[k+'/R']) for k in ks],1); P=np.stack([h[k+'/p'] for k in ks],1)
lc=float(sys.argv[1]) if len(sys.argv)>1 else 1.5; g=float(sys.argv[2]) if len(sys.argv)>2 else 0.02
U=np.stack([(1-h[k+'/iou'])+2*h[k+'/dt']+lc*h[k+'/chr'] for k in ks],1)
U[~np.isfinite(U)]=10
T,N=U.shape
D=np.zeros((T,N)); bp=np.zeros((T,N),int); D[0]=U[0]
for t in range(1,T):
    ang=rot_angle(R[t-1][:,None],R[t][None,:])  # (N,N)
    C=g*np.maximum(ang-3,0)
    tot=D[t-1][:,None]+C
    bp[t]=tot.argmin(0); D[t]=tot.min(0)+U[t]
path=np.zeros(T,int); path[-1]=D[-1].argmin()
for t in range(T-1,0,-1): path[t-1]=bp[t,path[t]]
Rs=R[np.arange(T),path]
np.savez('/work/scratch/vit.npz',ids=ids,R=Rs,p=P[np.arange(T),path],path=path,ks=np.array(ks))
step=rot_angle(Rs[1:],Rs[:-1])
print('mean unary chosen %.3f vs b0 %.3f'%(U[np.arange(T),path].mean(),U[:,0].mean()))
for t in range(0,T,6): print(ids[t],ks[path[t]],'U %.3f b0U %.3f'%(U[t,path[t]],U[t,0]),'ang_to_b0 %.0f'%rot_angle(Rs[t],R[t,0]),'step %.1f'%(step[t] if t<T-1 else 0))
EOF
python scratch/t_vit.py 1.5 0.02 | head -100
```

<details><summary>output</summary>

```
mean unary chosen 0.753 vs b0 0.803
100 f0 U 0.750 b0U 0.720 ang_to_b0 5 step 1.8
106 b0 U 0.829 b0U 0.829 ang_to_b0 0 step 24.7
112 f0 U 0.759 b0U 0.718 ang_to_b0 9 step 1.0
118 f0 U 0.668 b0U 0.784 ang_to_b0 5 step 5.0
124 b0 U 0.764 b0U 0.764 ang_to_b0 0 step 0.5
130 f0 U 0.787 b0U 0.721 ang_to_b0 7 step 1.9
136 b0 U 0.785 b0U 0.785 ang_to_b0 0 step 1.1
142 b0 U 0.782 b0U 0.782 ang_to_b0 0 step 3.5
148 f0 U 0.828 b0U 0.788 ang_to_b0 13 step 7.1
154 f0 U 0.729 b0U 0.790 ang_to_b0 8 step 4.8
160 b0 U 0.656 b0U 0.656 ang_to_b0 0 step 6.5
166 b0 U 0.578 b0U 0.578 ang_to_b0 0 step 5.0
172 b0 U 0.821 b0U 0.821 ang_to_b0 0 step 6.1
178 b0 U 0.882 b0U 0.882 ang_to_b0 0 step 3.2
184 f0 U 0.923 b0U 0.930 ang_to_b0 11 step 2.0
190 f0 U 0.843 b0U 1.076 ang_to_b0 6 step 39.2
196 f0 U 0.828 b0U 0.863 …[+3575 chars]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from fit import CropFitter
v=np.load('/work/scratch/vit.npz'); fit=CropFitter('/work/inputs/object_mesh.glb',res=32)
T=np.tile(np.eye(4),(747,1,1)); T[v['ids']]=fit.to_T(v['R'],v['p'])
np.savez('/work/scratch/vit_T.npz',**{'vit/T':T})
EOF
python scratch/t_cmp.py scratch/viz/c3.png scratch/vit_T.npz base0,vit 238 246 262 304 316 328 364 388; python scratch/t_cmp.py scratch/viz/c4.png scratch/vit_T.npz base0,vit 196 400 544 556
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/c3.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/viz/c4.png
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The Viterbi decode underperforms the baseline—frames 316, 328, 388, and 400 are misclassified as back-facing despite showing the front—likely because the chroma cue is too noisy on dark pixels. I'll build a stronger appearance feature using blurred redness and darkness maps and test it against manually verified frames.

**Bash**
```
cd /work; cat >> outputs/fit.py <<'EOF'


def app_feat(x):
    """2-channel appearance feature from BGR in [0,1]: redness and darkness."""
    b, g, r = x[..., 0], x[..., 1], x[..., 2]
    red = (r - torch.maximum(g, b)).clamp(0, 1) / (r + 0.05)
    v = x.max(-1).values
    return torch.stack([red, v], -1)


def blur(x, k=5):
    # x (B,H,W,C) box blur
    xc = x.permute(0, 3, 1, 2)
    xc = torch.nn.functional.avg_pool2d(xc, k, stride=1, padding=k // 2, count_include_pad=False)
    return xc.permute(0, 2, 3, 1)


def app_score(col, hard, img, M, V):
    """appearance distance inside rendered∩mask (valid). Brightness normalised by median ratio per frame."""
    fr = app_feat(col)
    fi = app_feat(img)
    w = hard * M * V
    ws = w.sum((1, 2)).clamp_min(1.0)
    # gain for brightness: ratio of means
    gain = (fi[..., 1] * w).sum((1, 2)) / (fr[..., 1] * w).sum((1, 2)).clamp_min(1e-3)
    fr = torch.stack([fr[..., 0], fr[..., 1] * gain[:, None, None]], -1)
    fr, fi = blur(fr * w[..., None]), blur(fi * w[..., None])
    wb = blur(w[..., None]).clamp_min(1e-3)
    fr, fi = fr / wb, fi / wb
    d = (fr - fi).abs()
    return (d[..., 0] * w).sum((1, 2)) / ws, (d[..., 1] * w).sum((1, 2)) / ws
EOF
cat > scratch/t_app.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
h=np.load('/work/scratch/hyp2.npz'); hid=list(h['ids'])
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ids=[int(a) for a in sys.argv[1].split(',')]
ks=['b0','b90','b180','b270','f0','f180']
fd=make …[+645 chars]
```

<details><summary>output</summary>

```
200 b0: 0.50 0.176 0.127 | b90: 0.54 0.194 0.134 | b180: 0.48 0.202 0.214 | b270: 0.47 0.193 0.249 | f0: 0.58 0.144 0.173 | f180: 0.52 0.141 0.201 |
256 b0: 0.69 0.232 0.071 | b90: 0.64 0.107 0.080 | b180: 0.69 0.049 0.068 | b270: 0.50 0.187 0.075 | f0: 0.52 0.271 0.079 | f180: 0.60 0.066 0.076 |
262 b0: 0.41 0.212 0.073 | b90: 0.57 0.268 0.092 | b180: 0.62 0.122 0.082 | b270: 0.66 0.076 0.067 | f0: 0.56 0.265 0.090 | f180: 0.65 0.064 0.075 |
280 b0: 0.71 0.055 0.066 | b90: 0.22 0.188 0.067 | b180: 0.65 0.299 0.069 | b270: 0.60 0.142 0.081 | f0: 0.64 0.301 0.072 | f180: 0.62 0.077 0.079 |
304 b0: 0.54 0.136 0.075 | b90: 0.51 0.139 0.084 | b180: 0.36 0.305 0.094 | b270: 0.52 0.227 0.091 | f0: 0.44 0.178 0.092 | f180: 0.28 0.236 0.091 |
316 b0: 0.63 0.115 0.096 | b90: 0.64 0.113 0.096 | b180 …[+689 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/t_app2.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
h=np.load('/work/scratch/hyp2.npz'); hid=list(h['ids'])
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ks=['b0','b90','b180','b270','f0','f90','f180','f270']
allS={k:[] for k in ks}
for c0 in range(0,len(hid),140):
    ids=hid[c0:c0+140]; fd=make_fd(fit,ids,Tb[ids]); j=list(range(c0,c0+len(ids)))
    for k in ks:
        Rm=torch.tensor(np.nan_to_num(h[k+'/R'][j]),dtype=torch.float32,device='cuda'); p=torch.tensor(h[k+'/p'][j],dtype=torch.float32,device='cuda')
        with torch.no_grad():
            o=fit.render(Rm,p,fd.K); L=losses(fit,Rm,p,fd,w_col=0)
            dr_,dv=app_score(o['color'],o['hard'],fd.img,fd.M,fd.V)
        allS[k].append(np.stack([L['iou'].cpu().numpy(),L['dt'].cpu().numpy(),dr_.cpu().numpy(),dv.cpu().numpy()],1))
np.savez('/work/scratch/hyp2_scores.npz',ids=hid,**{k:np.concatenate(v) for k,v in allS.items()})
EOF
python scratch/t_app2.py; python - <<'EOF'
import numpy as np
s=np.load('/work/scratch/hyp2_scores.npz'); ids=s['ids']
ks=['b0','b90','b180','b270','f0','f90','f180','f270']
U=np.stack([1-s[k][:,0]+2*s[k][:,1]+s[k][:,2]+s[k][:,3] for k in ks],1); U=np.nan_to_num(U,nan=9)
for t in range(0,len(ids),3):
    o=np.argsort(U[t]); print(ids[t],' '.join('%s%.2f'%(k,U[t,i]) for i,k in enumerate(ks)),'| best',ks[o[0]],'margin_vs_b0 %.2f'%(U[t,0]-U[t,o[0]]))
EOF
```

<details><summary>output</summary>

```
100 b00.56 b900.70 b1800.74 b2700.65 f00.60 f900.70 f1800.78 f2700.66 | best b0 margin_vs_b0 0.00
103 b00.62 b900.68 b1800.74 b2700.63 f00.53 f900.77 f1800.87 f2700.66 | best f0 margin_vs_b0 0.09
106 b00.65 b900.67 b1800.74 b2700.66 f00.67 f900.70 f1800.80 f2700.63 | best f270 margin_vs_b0 0.02
109 b00.58 b900.73 b1800.76 b2700.65 f00.54 f900.71 f1800.79 f2700.65 | best f0 margin_vs_b0 0.04
112 b00.58 b900.69 b1800.74 b2700.63 f00.57 f900.70 f1800.80 f2700.64 | best f0 margin_vs_b0 0.01
115 b00.52 b900.73 b1800.75 b2700.73 f00.57 f900.73 f1800.89 f2700.66 | best b0 margin_vs_b0 0.00
118 b00.59 b900.74 b1800.95 b2700.62 f00.52 f900.68 f1800.80 f2700.65 | best f0 margin_vs_b0 0.07
121 b00.59 b900.67 b1800.81 b2700.63 f00.61 f900.76 f1800.78 f2700.62 | best b0 margin_vs_b0 0.00
124 b00.58 b90 …[+17499 chars]
```
</details>

> *thinking:* The refined appearance score cleanly separates hypotheses, with the rotated version winning only in frames 241–268 and ~307–322 (where the vacuum turns to show its back). Next, I'll automatically detect the static phases and run a joint temporal optimization with smoothness priors.

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np, cv2
O=np.load('/work/scratch/cache/obj.npy',mmap_mode='r'); P=np.load('/work/scratch/cache/person.npy',mmap_mode='r')
F=O.shape[0]
small=np.stack([cv2.resize(O[i].astype(np.uint8),(384,288),interpolation=cv2.INTER_AREA)>0 for i in range(F)])
ps=np.stack([cv2.resize(P[i].astype(np.uint8),(384,288),interpolation=cv2.INTER_AREA)>0 for i in range(F)])
def iou(a,b): return (a&b).sum()/max((a|b).sum(),1)
s=np.array([iou(small[i],small[max(i-3,0)]) for i in range(F)])
k=np.ones((9,9),np.uint8)
touch=np.array([(cv2.dilate(small[i].astype(np.uint8),k)>0)[ps[i]].sum() for i in range(F)])
np.save('/work/scratch/stab.npy',np.stack([s,touch],1))
for i in list(range(90,150,2))+list(range(600,660,2)): print(i,'%.2f'%s[i],touch[i])
EOF
```

<details><summary>output</summary>

```
90 0.99 5
92 0.99 6
94 0.99 7
96 0.95 17
98 0.99 16
100 0.99 17
102 0.99 26
104 0.99 30
106 0.99 33
108 0.98 36
110 0.98 38
112 0.93 39
114 0.95 31
116 0.96 33
118 0.96 39
120 0.95 21
122 0.97 21
124 0.98 21
126 0.90 24
128 0.86 24
130 0.79 19
132 0.67 25
134 0.59 22
136 0.57 21
138 0.60 19
140 0.63 21
142 0.61 18
144 0.58 21
146 0.61 15
148 0.62 16
600 0.64 205
602 0.63 217
604 0.63 225
606 0.60 252
608 0.50 269
610 0.60 265
612 0.82 283
614 0.92 281
616 0.94 283
618 0.94 279
620 0.98 271
622 0.93 266
624 0.98 269
626 0.97 263
628 0.95 286
630 0.96 286
632 0.96 283
634 0.96 304
636 0.95 299
638 0.97 298
640 0.97 293
642 0.96 288
644 0.97 273
646 0.94 262
648 0.94 263
650 0.94 230
652 0.96 227
654 0.95 201
656 0.94 140
658 0.93 81
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np, cv2
O=np.load('/work/scratch/cache/obj.npy',mmap_mode='r'); P=np.load('/work/scratch/cache/person.npy',mmap_mode='r')
def iou(a,b): return (a&b).sum()/max((a|b).sum(),1)
ref=O[720]; ch=np.load('/work/scratch/cache/chair.npy')
for i in range(596,680,4):
    vis=~(P[i]|P[720])  # compare only where person absent in both
    print(i,'%.2f'%iou(O[i]&vis,ref&vis))
ref=O[50]
for i in range(96,136,4):
    vis=~(P[i]|P[50]); print(i,'%.2f'%iou(O[i]&vis,ref&vis))
EOF
```

<details><summary>output</summary>

```
596 0.06
600 0.19
604 0.38
608 0.78
612 0.92
616 0.95
620 0.95
624 0.95
628 0.95
632 0.95
636 0.95
640 0.95
644 0.94
648 0.94
652 0.94
656 0.95
660 0.97
664 0.98
668 0.98
672 0.98
676 0.98
96 0.95
100 0.99
104 0.98
108 0.98
112 0.95
116 0.96
120 0.97
124 0.97
128 0.82
132 0.56
```
</details>

**Write**
```
/work/outputs/traj.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/traj.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work; cat > scratch/t_joint.py <<'EOF'
exec(open('/work/scratch/common_exp.py').read())
from traj import *
from scipy.ndimage import median_filter, gaussian_filter1d
import time
holdout=int(sys.argv[1]); tag=sys.argv[2]
lam=float(sys.argv[3]) if len(sys.argv)>3 else 5.0
fit=CropFitter('/work/inputs/object_mesh.glb',res=160)
h=np.load('/work/scratch/hyp2.npz'); s=np.load('/work/scratch/hyp2_scores.npz'); hid=list(h['ids'])
ks=['b0','b90','b180','b270','f0','f90','f180','f270']
a,b=126,612
j=[hid.index(i) for i in range(a,b)]
U=np.stack([1-s[k][j,0]+2*s[k][j,1]+s[k][j,2]+s[k][j,3] for k in ks],1); U=np.nan_to_num(U,nan=9)
R=np.stack([np.nan_to_num(h[k+'/R'][j]) for k in ks],1); P=np.stack([h[k+'/p'][j] for k in ks],1)
path=viterbi(U,R,P)
print('viterbi choices',{k:int((path==i).sum()) for i,k in enumerate(ks)})
R0=R[np.arange(len(j)),path]; p0=P[np.arange(len(j)),path]
Rb,pb=fit.to_rp(Tb)
z=gaussian_filter1d(median_filter(pb[:,2],9,mode='nearest'),3,mode='nearest')
p0[:,2]=z[a:b]
S1=np.load('/work/scratch/static2_0_rz-30_0.0.npy'); S2=np.load('/work/scratch/static2_640_rz0_0.0.npy')
R1,p1=fit.to_rp(S1[None]); R2,p2=fit.to_rp(S2[None])
ids=list(range(a,b))
fd=make_fd(fit,ids,Tb[ids],R=160)
iw=np.ones(len(ids)); 
if holdout: iw[::5]=0
t0=time.time()
Rj,pj=joint_refine(fit,fd,R0,p0,iw,np.repeat(R1,2,0),np.repeat(p1,2,0),np.repeat(R2,2,0),np.repeat(p2,2,0),iters=150,lam_acc_r=lam,log=print)
print('time',time.time()-t0)
E_j=eval_frames(fit,fd,Rj,pj); E_b=eval_frames(fit,fd,Rb[a: …[+472 chars]
```

<details><summary>output</summary>

```
viterbi choices {'b0': 245, 'b90': 9, 'b180': 3, 'b270': 29, 'f0': 138, 'f90': 11, 'f180': 21, 'f270': 30}
  it 0 img 165.679 acc_r 74.8268 vel_r 35.8940 acc_p 0.757946
  it 25 img 165.182 acc_r 7.1510 vel_r 13.7017 acc_p 0.029131
  it 50 img 163.673 acc_r 2.3002 vel_r 9.1865 acc_p 0.010357
  it 75 img 163.416 acc_r 1.1937 vel_r 7.0322 acc_p 0.008414
  it 100 img 163.208 acc_r 0.7679 vel_r 5.9182 acc_p 0.007831
  it 125 img 163.219 acc_r 0.5601 vel_r 5.2568 acc_p 0.007582
  it 149 img 163.281 acc_r 0.4458 vel_r 4.8373 acc_p 0.007477
time 55.54663825035095
frames 98 iou base 0.391 vit 0.574 joint 0.582 U base 0.847 vit 0.646 joint 0.636
```
</details>

> *thinking:* The held-out check shows mask IoU of 0.58 for the joint trajectory versus 0.39 for the baseline. I'm now investigating whether fitting at fixed depth while masks run ~8% too big could skew the long-axis tilt, and testing a shared depth-scale factor to absorb that size mismatch before removing it.

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
exec(open('/work/scratch/common_exp.py').read())
d=np.load('/work/scratch/joint_ho5.npz'); a,b=int(d['a']),int(d['b'])
fit=CropFitter('/work/inputs/object_mesh.glb',res=32)
Rb,pb=fit.to_rp(Tb)
def oop(Rm,p):
    ax=Rm[:,:,2]; ray=p/np.linalg.norm(p,axis=1,keepdims=True)
    return np.degrees(np.arcsin(np.abs((ax*ray).sum(1))))
ob=oop(Rb[a:b],pb[a:b]); oj=oop(d['R'],d['p'])
print('out-of-plane angle of long axis: base mean %.1f median %.1f | joint mean %.1f median %.1f'%(ob.mean(),np.median(ob),oj.mean(),np.median(oj)))
print('static1 base',oop(Rb[:110],pb[:110]).mean(),'static2 base',oop(Rb[620:],pb[620:]).mean())
for nm in ['static2_0_rz-30_0.0','static2_640_rz0_0.0']:
    T=np.load(f'/work/scratch/{nm}.npy'); R_,p_=fit.to_rp(T[None]); print(nm,oop(R_,p_))
EOF
```

<details><summary>output</summary>

```
out-of-plane angle of long axis: base mean 16.1 median 15.2 | joint mean 21.2 median 15.9
static1 base 13.489857457267046 static2 base 24.1550395582257
static2_0_rz-30_0.0 [12.85038677]
static2_640_rz0_0.0 [21.96968812]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
p='/work/outputs/traj.py'; s=open(p).read()
s=s.replace("lam_prior=0.0, chunk=130, w_dt=2.0, fix_depth=True, log=None):","lam_prior=0.0, chunk=130, w_dt=2.0, fix_depth=True, global_scale=False, log=None):")
s=s.replace("""    iw = torch.tensor(img_w""","""    ls = torch.zeros(1, device=dev, requires_grad=global_scale)
    iw = torch.tensor(img_w""")
s=s.replace("""    opt = torch.optim.Adam([{"params": [w], "lr": lr_r}, {"params": [dp], "lr": lr_t}])
    tt =""","""    groups = [{"params": [w], "lr": lr_r}, {"params": [dp], "lr": lr_t}]
    if global_scale:
        groups.append({"params": [ls], "lr": 0.003})
    opt = torch.optim.Adam(groups)
    tt =""")
s=s.replace("""            L = losses(fit, Rm[idx], p[idx], sub, 0.0)""","""            # global scale: the silhouette sees the trajectory scaled along the viewing rays; the
            # scale absorbs a constant mask-size bias and is discarded afterwards (depth re-anchoring)
            L = losses(fit, Rm[idx], p[idx] * torch.exp(ls), sub, 0.0)""")
s=s.replace("""            log(f"  it {it} img""","""            log(f"  it {it} scale {torch.exp(ls).item():.4f} img""")
open(p,'w').write(s)
EOF
sed -i 's/iters=150,lam_acc_r=lam,log=print)/iters=150,lam_acc_r=lam,log=print,global_scale=bool(int(sys.argv[4])) if len(sys.argv)>4 else False)/' scratch/t_joint.py
python scratch/t_joint.py 1 ho5s 5.0 1
```

<details><summary>output</summary>

```
viterbi choices {'b0': 245, 'b90': 9, 'b180': 3, 'b270': 29, 'f0': 138, 'f90': 11, 'f180': 21, 'f270': 30}
  it 0 scale 0.9970 img 165.679 acc_r 74.8268 vel_r 35.8940 acc_p 0.757946
  it 25 scale 0.9685 img 158.548 acc_r 7.1545 vel_r 13.7167 acc_p 0.029373
  it 50 scale 0.9526 img 153.125 acc_r 2.3166 vel_r 9.2290 acc_p 0.010562
  it 75 scale 0.9398 img 148.944 acc_r 1.2028 vel_r 7.0527 acc_p 0.008716
  it 100 scale 0.9293 img 146.434 acc_r 0.7775 vel_r 5.9485 acc_p 0.008068
  it 125 scale 0.9210 img 144.501 acc_r 0.5726 vel_r 5.3203 acc_p 0.007705
  it 149 scale 0.9143 img 142.978 acc_r 0.4670 vel_r 4.9409 acc_p 0.007558
time 78.982492685318
frames 98 iou base 0.391 vit 0.574 joint 0.548 U base 0.847 vit 0.646 joint 0.650
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from lib import rot_angle
a=np.load('/work/scratch/joint_ho5.npz'); b=np.load('/work/scratch/joint_ho5s.npz')
d=rot_angle(a['R'],b['R']); print('rot diff scale vs no-scale: mean %.1f median %.1f p90 %.1f max %.1f'%(d.mean(),np.median(d),np.percentile(d,90),d.max()))
dv=rot_angle(a['R'],a['R0']); print('joint vs viterbi init: mean %.1f median %.1f p90 %.1f'%(dv.mean(),np.median(dv),np.percentile(dv,90)))
EOF
```

<details><summary>output</summary>

```
rot diff scale vs no-scale: mean 4.0 median 3.7 p90 6.9 max 11.3
joint vs viterbi init: mean 6.6 median 4.4 p90 16.2
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np
exec(open('/work/scratch/common_exp.py').read())
d=np.load('/work/inputs/frame0_reference_human.npz'); print({k:d[k].shape for k in d.files})
V=d['vertices_world']; Tcw=np.linalg.inv(Twc); X=V@Tcw[:3,:3].T+Tcw[:3,3]
uv=X[:,:2]/X[:,2:]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
m=np.zeros((1152,1536),np.uint8)
for u,v in uv.astype(int):
    if 0<=u<1536 and 0<=v<1152: m[v,u]=1
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
pm=PER[0]
print('proj area',m.sum(),'mask area',pm.sum(),'iou',(m&pm).sum()/(m|pm).sum())
ys,xs=np.nonzero(m); ys2,xs2=np.nonzero(pm); print('bbox proj',xs.min(),xs.max(),ys.min(),ys.max(),'mask',xs2.min(),xs2.max(),ys2.min(),ys2.max())
print('human depth median',np.median(X[:,2]))
EOF
```

<details><summary>output</summary>

```
{'vertices_world': (18056, 3)}
proj area 29635 mask area 37853 iou 0.7612150630235653
bbox proj 848 976 389 880 mask 841 975 380 896
human depth median 2.9719815923928303
```
</details>

**Bash**
```
cd /work; python scratch/t_joint.py 1 ho5_l15 15.0 0 2>&1 | tail -2; python scratch/t_joint.py 1 ho5_l2 2.0 0 2>&1 | tail -1
```

<details><summary>output</summary>

```
time 44.969210624694824
frames 98 iou base 0.391 vit 0.574 joint 0.581 U base 0.847 vit 0.646 joint 0.638
frames 98 iou base 0.391 vit 0.574 joint 0.582 U base 0.847 vit 0.646 joint 0.637
```
</details>

> *thinking:* Smoothness weight has little effect (IoU ~0.581–0.582 across λ=2,5,15), so I'll keep 5; the ~9% size mismatch traces to SAM2 mask bloat rather than depth, so I'll estimate one global silhouette scale from static phases for fitting only, then discard it. Writing `solve.py` now.

**Bash**
```
cd /work; python - <<'EOF'
p='/work/outputs/fit.py'; s=open(p).read()
# add scale option to refine and refine_grouped
s=s.replace("""           fix_depth=False, z_prior=None, w_z=0.0, smooth=None):""","""           fix_depth=False, z_prior=None, w_z=0.0, smooth=None, scale=1.0):""")
s=s.replace("""        p = p0t + d
        L = losses(fit, Rm, p, fd, w_col)
        loss = (1 - L["iou"]) + w_dt * L["dt"]
        if w_col > 0:
            loss = loss + w_col * L["chr"]
        if z_prior""","""        p = p0t + d
        L = losses(fit, Rm, p * scale, fd, w_col)
        loss = (1 - L["iou"]) + w_dt * L["dt"]
        if w_col > 0:
            loss = loss + w_col * L["chr"]
        if z_prior""")
s=s.replace("""        p = p0t + d
        L = losses(fit, Rm, p, fd, w_col=1.0)
    return Rm.detach()""","""        p = p0t + d
        L = losses(fit, Rm, p * scale, fd, w_col=1.0)
    return Rm.detach()""")
open(p,'w').write(s)
p='/work/outputs/traj.py'; s=open(p).read()
s=s.replace("global_scale=False, log=None):","global_scale=False, scale=1.0, log=None):")
s=s.replace("ls = torch.zeros(1, device=dev, requires_grad=global_scale)","ls = torch.full((1,), float(np.log(scale)), device=dev, requires_grad=global_scale)")
s=s.replace("""def eval_frames(fit, fd, Rm, p, chunk=130):""","""def eval_frames(fit, fd, Rm, p, chunk=130, scale=1.0):""")
s=s.replace("""        pt = torch.tensor(p[ids], dtype=torch.float32, device=dev)""","""        pt = torch.tensor(p[ids] * scale, dtype=torch.floa …[+103 chars]
```

<details><summary>output</summary>

```
outputs/fit.py:177:           fix_depth=False, z_prior=None, w_z=0.0, smooth=None, scale=1.0):
outputs/fit.py:193:        L = losses(fit, Rm, p * scale, fd, w_col)
outputs/fit.py:208:        L = losses(fit, Rm, p * scale, fd, w_col=1.0)
outputs/traj.py:30:                 lam_prior=0.0, chunk=130, w_dt=2.0, fix_depth=True, global_scale=False, scale=1.0, log=None):
outputs/traj.py:40:    ls = torch.full((1,), float(np.log(scale)), device=dev, requires_grad=global_scale)
outputs/traj.py:43:    if global_scale:
outputs/traj.py:62:            # global scale: the silhouette sees the trajectory scaled along the viewing rays; the
outputs/traj.py:63:            # scale absorbs a constant mask-size bias and is discarded afterwards (depth re-anchoring)
outputs/traj.py:83:            log(f"  it {it}  …[+295 chars]
```
</details>

**Write**
```
/work/outputs/solve.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/solve.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work; python outputs/solve.py --inputs /work/inputs --out /work/scratch/val/object_poses_world.npz --holdout 5 2>&1 | grep -v Warning | tee scratch/val_log.txt | tail -50
```

<details><summary>output</summary>

```
09:35:48 loading video / masks
09:36:07 static phases: [0,124] and [657,746]; carried: [125,656]
09:36:10   static [0,124] hyp rz0: U 0.488 iou 0.685 dR_base 15.4 scale 0.902
09:36:12   static [0,124] hyp rz-30: U 0.492 iou 0.670 dR_base 11.6 scale 0.927
09:36:15   static [0,124] hyp rz30: U 0.484 iou 0.719 dR_base 33.3 scale 0.854
09:36:17   static [0,124] hyp rz90: U 0.877 iou 0.440 dR_base 94.7 scale 0.880
09:36:19   static [0,124] hyp rz180: U 0.685 iou 0.575 dR_base 155.1 scale 0.913
09:36:21   static [0,124] hyp rz270: U 0.616 iou 0.659 dR_base 83.2 scale 0.916
09:36:23   static [0,124] hyp fx: U 0.714 iou 0.558 dR_base 175.5 scale 0.930
09:36:27   static [0,124] hyp fy: U 0.736 iou 0.552 dR_base 178.2 scale 0.833
09:36:27   -> static [0,124] chose rz30 U 0.484 (baseline per-frame U  …[+1988 chars]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
p='/work/outputs/fit.py'; s=open(p).read()
s=s.replace("""                   fix_depth=True, frame_w=None):""","""                   fix_depth=True, frame_w=None, scale=1.0):""")
s=s.replace("""        Rm, p = cur()
        L = losses(fit, Rm[g], p[g], fd, w_col)""","""        Rm, p = cur()
        L = losses(fit, Rm[g], p[g] * scale, fd, w_col)""")
s=s.replace("""        Rm, p = cur()
        L = losses(fit, Rm[g], p[g], fd, w_col=1.0)""","""        Rm, p = cur()
        L = losses(fit, Rm[g], p[g] * scale, fd, w_col=1.0)""")
open(p,'w').write(s)
p='/work/outputs/solve.py'; s=open(p).read()
old_start=s.index("    static = []\n")
old_end=s.index("    # ------------------------------------------------------------ carried phase: hypotheses")
new='''    # (a) silhouette scale (mask-size bias): free-depth fit of the baseline mean pose of each static phase.
    #     The masks are slightly bloated (SAM2 person mask vs. the true frame-0 human: ~3-5 % larger), so the
    #     silhouette prefers the object closer; one global factor absorbs this and is discarded at the end.
    sdata = []
    for (a, b) in [(0, s1), (s2, F - 1)]:
        ids = [i for i in range(a, b + 1, 3) if OBJ[i].sum() > 200]
        Rm0 = Rot.from_matrix(Rb[a:b + 1]).mean().as_matrix()
        pm0 = np.median(pb[a:b + 1], 0)
        T0 = fit.to_T(Rm0[None], pm0[None])[0]
        fd = make_fd(ids, [T0] * len(ids), RES_FIT)
        R1, p1, _ = refine_grouped(fit, Rm0[None], pm0[None], fd …[+1869 chars]
```

<details><summary>output</summary>

```
153:    gscale = float(np.clip(np.mean([d[-1] for d in sdata]), 0.85, 1.0))
154:    log(f"global silhouette scale (mask-size bias) {gscale:.3f}")
155:    # (b) one pose per static phase, several hypotheses, depth fixed at the baseline median (x gscale)
164:                                       fix_depth=True, scale=gscale)
165:            E = eval_frames(fit, fd, np.repeat(R1, len(ids), 0), np.repeat(p1, len(ids), 0), scale=gscale)
170:        Eb = eval_frames(fit, fd, Rb[ids], pb[ids], scale=gscale)
196:                               lr_r=0.03, scale=gscale)
198:            E = eval_frames(fit, fds, R1, p1, scale=gscale)
221:                          lam_acc_r=5.0, scale=gscale, log=log)
242:        Ej = eval_frames(fit, fd, Rj, pj, scale=gscale)
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np,json
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
T=np.load('/work/scratch/val/object_poses_world.npz')['T_world_mesh']
np.savez('/work/scratch/val_T.npz',**{'val/T':np.linalg.inv(Twc)@T})
EOF
python scratch/t_cmp.py scratch/viz/v1.png scratch/val_T.npz base0,val 140 200 250 262 290 316 400 470 560 620
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/viz/v1.png
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* Held-out test shows mask IoU of 0.65 vs 0.39 baseline for frames with removed image terms, and overlays look correct except one uncertain frame at a transition (316). Running the final `solve.py` now.

**Bash**
```
cd /work; python outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | grep -v Warning | tee outputs/solve_log.txt | grep -v "  it \|hypotheses "
```

<details><summary>output</summary>

```
09:41:45 loading video / masks
09:42:03 static phases: [0,124] and [657,746]; carried: [125,656]
09:42:07   static [0,124] free-depth scale 0.897
09:42:10   static [657,746] free-depth scale 0.898
09:42:10 global silhouette scale (mask-size bias) 0.898
09:42:12   static [0,124] hyp rz0: U 0.508 iou 0.688 dR_base 16.2
09:42:15   static [0,124] hyp rz-30: U 0.539 iou 0.675 dR_base 14.9
09:42:17   static [0,124] hyp rz30: U 0.517 iou 0.689 dR_base 33.1
09:42:19   static [0,124] hyp rz90: U 0.593 iou 0.630 dR_base 95.1
09:42:22   static [0,124] hyp rz180: U 0.695 iou 0.592 dR_base 149.9
09:42:24   static [0,124] hyp rz270: U 0.660 iou 0.650 dR_base 88.7
09:42:26   static [0,124] hyp fx: U 0.700 iou 0.561 dR_base 178.0
09:42:29   static [0,124] hyp fy: U 0.762 iou 0.519 dR_base 177.1
09:42:29   …[+1026 chars]
```
</details>

> *thinking:* The final run finished in 244 s, with both static phases selecting rolls closest to the baseline mean (16° and 9° off baseline) and depth re-anchoring factors near 1.00. Next, I'll compute the per-phase validation table comparing baseline and result.

**Bash**
```
cd /work; cat > outputs/validate.py <<'EOF'
"""Validation without ground truth: per-phase table baseline vs result (mask IoU, appearance, smoothness,
floor contact, depth difference). Usage: python validate.py --inputs /work/inputs --res object_poses_world.npz"""
import argparse, json, os, sys
import numpy as np, cv2, h5py, torch, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_inputs, load_video, rot_angle
from fit import CropFitter, FrameData, project, box_from
from traj import eval_frames

ap = argparse.ArgumentParser(); ap.add_argument("--inputs", default="/work/inputs")
ap.add_argument("--res", default="/work/outputs/object_poses_world.npz"); a = ap.parse_args()
cam, K, Twc, rec = load_inputs(a.inputs); Tcw = np.linalg.inv(Twc)
Tb = Tcw @ rec["T_world_mesh"].astype(np.float64)
To = Tcw @ np.load(a.res)["T_world_mesh"]
diag = np.load(os.path.splitext(a.res)[0] + "_diag.npz"); ph = diag["phases"]; gs = float(diag["gscale"])
frames = load_video(a.inputs, 1.0)
with h5py.File(os.path.join(a.inputs, "masks.h5"), "r") as f:
    OBJ = f["object"][:]; PER = f["person"][:]
CH = np.zeros(OBJ.shape[1:], bool)
CH[650:1005, 625:835] = False  # stool not excluded here: identical treatment for baseline and result
fit = CropFitter(os.path.join(a.inputs, "object_mesh.glb"), res=160)
corners = trimesh.bounds.corners(fit.mesh.bounds)
pl = np.array(json.load(open(os.path.join(a.inputs, "ground_plane.json")))["plane"])
V = np.asarray(fit.mesh.vertic …[+2366 chars]
```

<details><summary>output</summary>

```
| phase | frames | IoU base / ours (scale 1) | IoU base / ours (scale 0.898) | appearance dist base / ours | rot step med / p95 deg base | ours | centre accel median cm base / ours | lowest point to floor cm base / ours | depth diff ours-base mean / median cm | rot change vs base median / p90 deg |
|---|---|---|---|---|---|---|---|---|---|---|
| static_start | 0-124 | 0.451 / 0.465 | 0.388 / 0.558 | 0.254 / 0.192 | 0.07 / 5.6 | 0.00 / 0.0 | 0.14 / 0.00 | -3.1 / -6.2 | -0.82 / 0.00 | 13.3 / 23.5 |
| carried | 125-656 | 0.311 / 0.390 | 0.296 / 0.469 | 0.236 / 0.197 | 3.47 / 17.6 | 2.05 / 8.4 | 3.43 / 0.22 | -5.0 / -7.2 | 0.84 / 0.84 | 22.2 / 76.9 |
| static_end | 657-746 | 0.354 / 0.306 | 0.321 / 0.430 | 0.213 / 0.145 | 0.12 / 0.1 | 0.00 / 0.0 | 0.02 / 0.00 | -2.0 / -4.7 | 0.93 / 0.00 | 9.3  …[+7 chars]
```
</details>

> *thinking:* I found that my result sinks the vacuum's foot 3–4 cm further into the floor than the baseline in both static phases, while the true frame-0 soles sit at a realistic −2.0 cm, pointing to a slight rotation/position error. I'll add a soft floor-contact term (equality when at rest near the floor, one-sided no-penetration while carried) and re-run.

**Bash**
```
cd /work; cat >> outputs/fit.py <<'EOF'


class FloorTerm:
    """Signed distance of the mesh's lowest point to the floor (camera-frame computation).
    h0: floor level of resting contacts (signed plane distance of the true frame-0 human soles).
    mode 'contact': (d_min - h0)^2 ; mode 'above': relu(h0 - tol - d_min)^2."""

    def __init__(self, fit, plane_w, T_world_cam, h0, w, mode, tol=0.01, n_sub=6000):
        n = np.asarray(plane_w[:3], np.float64)
        d = float(plane_w[3])
        Rwc, twc = T_world_cam[:3, :3], T_world_cam[:3, 3]
        self.n = torch.tensor(Rwc.T @ n, dtype=torch.float32, device=fit.dev)
        self.d = float(n @ twc + d)
        step = max(1, len(fit.v) // n_sub)
        self.v = fit.v[::step]  # centred mesh coords
        self.h0, self.w, self.mode, self.tol = h0, w, mode, tol

    def dmin(self, Rm, p, tau=0.004):
        X = torch.einsum("bij,nj->bni", Rm, self.v) + p[:, None]
        s = X @ self.n + self.d
        return -tau * torch.logsumexp(-s / tau, 1)

    def __call__(self, Rm, p):
        m = self.dmin(Rm, p)
        if self.mode == "contact":
            return self.w * (m - self.h0) ** 2
        return self.w * torch.relu(self.h0 - self.tol - m) ** 2
EOF
python - <<'EOF'
p='/work/outputs/fit.py'; s=open(p).read()
s=s.replace("""                   fix_depth=True, frame_w=None, scale=1.0):""","""                   fix_depth=True, frame_w=None, scale=1.0, floor=None):""")
s=s.replace("""        if w_col > 0:
            loss = l …[+3146 chars]
```

<details><summary>output</summary>

```
134:    # floor: released plane + level of the true frame-0 human soles (signed distance of its lowest vertices);
135:    # a resting object touches the floor at that level, a carried one must not go below it.
143:    log(f"floor contact level h0 {h0 * 100:.1f} cm; baseline lowest point median {np.median(dmin_base) * 100:.1f} cm")
169:        # resting-contact term only if the baseline itself has the object on the floor in this phase (< 5 cm)
171:        floor = FloorTerm(fit, plane, Twc, h0, 300.0, "contact") if rest else \
173:        log(f"  static [{a},{b}] floor mode {'contact' if rest else 'above'}")
180:                                       fix_depth=True, scale=gscale, floor=floor)
237:                          lam_acc_r=5.0, scale=gscale, floor=FloorTerm(fit, plane, Twc, h0, 300. …[+21 chars]
```
</details>

**Bash**
```
cd /work; python outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | grep -v Warning | tee outputs/solve_log.txt | grep -v "hypotheses " | grep -v "static \[.*hyp r\|hyp f"; python outputs/validate.py 2>&1 | grep -v Warning | tee outputs/validation_table.md
```

<details><summary>output</summary>

```
09:47:28 loading video / masks
09:47:47 floor contact level h0 -0.0 cm; baseline lowest point median -5.2 cm
09:47:48 static phases: [0,124] and [657,746]; carried: [125,656]
09:47:52   static [0,124] free-depth scale 0.907
09:47:54   static [657,746] free-depth scale 0.902
09:47:54 global silhouette scale (mask-size bias) 0.904
09:47:54   static [0,124] floor mode contact
09:48:23   -> static [0,124] chose rz-30 U 0.535 (baseline per-frame U 0.804)
09:48:23   static [657,746] floor mode contact
09:48:48   -> static [657,746] chose rz0 U 0.593 (baseline per-frame U 0.694)
09:50:48 viterbi choices {'b0': 261, 'b90': 23, 'b180': 2, 'b270': 3, 'f0': 183, 'f90': 36, 'f180': 24, 'f270': 0}
09:50:53   it 0 scale 0.9043 img 192.397 acc_r 66.5087 vel_r 31.3747 acc_p 0.756146
09:50:59   it 25 scale …[+1501 chars]
```
</details>

> *thinking:* Reverting the floor contact term since enforcing it hurt fit quality (static-end IoU dropped 0.430→0.368, carried IoU 0.469→0.414), suggesting the baseline depth is off rather than the rotations being wrong. I'll report floor contact as an observation instead of forcing it, and re-run the image-only version.

**Bash**
```
cd /work; python - <<'EOF'
p='/work/outputs/solve.py'; s=open(p).read()
s=s.replace("""SEED = 0
""","""SEED = 0
# Floor terms are implemented but disabled: at the baseline depth (which we keep, see REPORT) the floor contact
# and the image evidence disagree by several cm; enforcing contact at fixed depth distorted rotations / image fit
# (static-end mask IoU 0.43 -> 0.37). The floor is therefore only reported as a depth cue (validate.py).
USE_FLOOR = False
""")
s=s.replace("""        rest = abs(np.median(dmin_base[a:b + 1]) - h0) < 0.05
        floor =""","""        rest = abs(np.median(dmin_base[a:b + 1]) - h0) < 0.05
        floor = None if not USE_FLOOR else""")
s=s.replace("""floor=FloorTerm(fit, plane, Twc, h0, 300.0, "above"), log=log)""","""floor=FloorTerm(fit, plane, Twc, h0, 300.0, "above") if USE_FLOOR else None, log=log)""")
s=s.replace("""        log(f"  static [{a},{b}] floor mode {'contact' if rest else 'above'}")""","""        log(f"  static [{a},{b}] floor term: {('contact' if rest else 'above') if USE_FLOOR else 'off'}")""")
open(p,'w').write(s)
EOF
python outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | grep -v Warning | tee outputs/solve_log.txt | grep -v "hypotheses \|  it " | grep -v "hyp r\|hyp f"; python outputs/validate.py 2>&1 | grep -v Warning | tee outputs/validation_table.md
```

<details><summary>output</summary>

```
09:52:27 loading video / masks
09:52:40 floor contact level h0 -0.0 cm; baseline lowest point median -5.2 cm
09:52:42 static phases: [0,124] and [657,746]; carried: [125,656]
09:52:46   static [0,124] free-depth scale 0.904
09:52:49   static [657,746] free-depth scale 0.902
09:52:49 global silhouette scale (mask-size bias) 0.903
09:52:49   static [0,124] floor term: off
09:53:13   -> static [0,124] chose rz-30 U 0.486 (baseline per-frame U 0.805)
09:53:13   static [657,746] floor term: off
09:53:22   -> static [657,746] chose rz0 U 0.453 (baseline per-frame U 0.696)
09:55:00 viterbi choices {'b0': 259, 'b90': 27, 'b180': 4, 'b270': 14, 'f0': 187, 'f90': 18, 'f180': 21, 'f270': 2}
09:55:43 depth re-anchor static_start: factor 1.0000
09:55:43 depth re-anchor carried: factor 1.0029
09:55:43 d …[+912 chars]
```
</details>

**Bash**
```
cd /work; cat > outputs/depth_cues.py <<'EOF'
"""Depth cues (reported only): suggested shift along the viewing ray (cm, negative = closer to camera)."""
import json, os, sys
import numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_inputs
inp = sys.argv[1] if len(sys.argv) > 1 else "/work/inputs"
res = sys.argv[2] if len(sys.argv) > 2 else "/work/outputs/object_poses_world.npz"
cam, K, Twc, rec = load_inputs(inp)
Tcw = np.linalg.inv(Twc)
pl = np.array(json.load(open(os.path.join(inp, "ground_plane.json")))["plane"]); n, d = pl[:3], pl[3]
hv = np.load(os.path.join(inp, "frame0_reference_human.npz"))["vertices_world"]
h0 = np.percentile(hv @ n + d, 0.1)
V = np.asarray(trimesh.load(os.path.join(inp, "object_mesh.glb"), force="scene", process=False).to_geometry().vertices)
diag = np.load(os.path.splitext(res)[0] + "_diag.npz"); ph = diag["phases"]; gs = float(diag["gscale"])
To = np.load(res)["T_world_mesh"]; Tb = rec["T_world_mesh"].astype(np.float64)
print(f"true-human sole level (plane distance, 0.1 pct): {h0 * 100:.1f} cm")
for tag, T in [("baseline", Tb), ("ours", To)]:
    for name, (a, b) in zip(["static_start", "static_end"], [ph[0], ph[2]]):
        # scale camera-frame vertex positions by f (move along rays); find f s.t. lowest point = h0
        Tc = Tcw @ T[a]
        Xc = V @ Tc[:3, :3].T + Tc[:3, 3]
        fs = np.linspace(0.85, 1.15, 601)
        lows = [(((f * Xc) @ Twc[:3, :3].T + Twc[:3, 3]) @ n + d).min() f …[+1393 chars]
```

<details><summary>output</summary>

```
true-human sole level (plane distance, 0.1 pct): -0.0 cm
floor cue baseline static_start: lowest point -2.1 cm -> depth shift -3.9 cm
floor cue baseline static_end: lowest point -1.5 cm -> depth shift -2.9 cm
floor cue ours static_start: lowest point -5.6 cm -> depth shift -10.4 cm
floor cue ours static_end: lowest point -4.6 cm -> depth shift -9.0 cm
silhouette-size cue: scale 0.903 -> -29.6 cm (uncorrected); with 3-5 % mask bloat measured on the true frame-0 human: -21.3 .. -15.7 cm
hand cue (right hand, 30% frames closest to the handle top): hand-handle depth diff median -3.3 cm, MAD 6.4 cm, 3D dist median 20.9 cm
hand cue (left hand, 30% frames closest to the handle top): hand-handle depth diff median -3.1 cm, MAD 2.2 cm, 3D dist median 5.7 cm
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np,json,sys; sys.path.insert(0,'/work/outputs')
from lib import rot_angle
o=np.load('/work/outputs/object_poses_world.npz'); T=o['T_world_mesh']; print(o.files,T.shape,T.dtype,np.isfinite(T).all())
R=T[:,:3,:3]; print('orthonormal err',np.abs(np.einsum('fij,fkj->fik',R,R)-np.eye(3)).max(),'det',np.linalg.det(R).min())
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
np.savez('/work/scratch/fin_T.npz',**{'fin/T':np.linalg.inv(Twc)@T})
EOF
python scratch/t_cmp.py scratch/viz/f1.png scratch/fin_T.npz base0,fin 60 180 256 300 340 450 600 700
```

<details><summary>output</summary>

```
['T_world_mesh'] (747, 4, 4) float64 True
orthonormal err 2.04546807092143e-06 det 0.9999984755798736
```
</details>

**Read**
```
/work/scratch/viz/f1.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python -c "
import importlib
for m in ['numpy','torch','nvdiffrast','cv2','trimesh','scipy','h5py','sam2','PIL']:
    try:
        x=importlib.import_module(m); print(m, getattr(x,'__version__','?'))
    except Exception as e: print(m,'ERR',e)
"; pip show sam2 2>/dev/null | head -3; sha256sum /weights/sam2/sam2.1_hiera_large.pt; ls /work/tools 2>/dev/null; ls outputs
```

<details><summary>output</summary>

```
numpy 1.26.3
torch 2.5.1+cu124
nvdiffrast 0.4.0
cv2 4.11.0
trimesh 5.1.1
scipy 1.16.3
h5py 3.16.0
sam2 ?
PIL 10.2.0
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318  /weights/sam2/sam2.1_hiera_large.pt
__pycache__
depth_cues.py
depth_cues.txt
fit.py
lib.py
object_poses_world.npz
object_poses_world_diag.npz
solve.py
solve_log.txt
traj.py
validate.py
validation_table.md
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
p='/work/outputs/validate.py'; s=open(p).read()
s=s.replace('''CH = np.zeros(OBJ.shape[1:], bool)
CH[650:1005, 625:835] = False  # stool not excluded here: identical treatment for baseline and result
''','''# note: only the person is excluded here (not the stool) - identical treatment for baseline and result
''')
open(p,'w').write(s)
EOF
rm -rf outputs/__pycache__; grep -c "it " outputs/solve_log.txt; grep "viterbi\|done" outputs/solve_log.txt
```

<details><summary>output</summary>

```
7
09:55:00 viterbi choices {'b0': 259, 'b90': 27, 'b180': 4, 'b270': 14, 'f0': 187, 'f90': 18, 'f180': 21, 'f270': 2}
09:55:43 done in 196.0s -> /work/outputs/object_poses_world.npz
```
</details>

**Agent:** The final output exists and is checked: 747 finite, orthonormal poses. Writing the report and the tools list now.

**Write**
```
/work/outputs/TOOLS.md
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/TOOLS.md (file state is current in your context — no need to Read it back)
```
</details>

**Write**
```
/work/outputs/REPORT.md
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/REPORT.md (file state is current in your context — no need to Read it back)
```
</details>

**Agent:** I've written a full 747-frame trajectory to `/work/outputs/object_poses_world.npz`. Mask fit and smoothness are clearly better than the baseline, but our static poses sit about 2.5 cm lower against the floor than the baseline does. That trade-off is unresolved.

The final `solve.py` run took 196 s offline and the output is finite with valid rotations. Total GPU time across all experiments was about 35 minutes (estimated), under 9 GB.

**Phases** (detected automatically from the masks, checked by eye):
- **Frames 0–124:** the vacuum leans on the bar stool. I fit one pose for the whole phase instead of the baseline's per-frame rotation, which wanders by up to 11°. Mine is 13° from the baseline, mostly a roll about the long axis.
- **Frames 125–656:** carried and used.
  - **Flip fixed:** at frames 244–268 the baseline shows the vacuum's red front while the video shows its grey back with the cord. My result shows the back.
  - **Jitter reduced:** rotation change per frame drops from a median of 3.5° (95th percentile 17.6°) to 1.9° (7.6°).
- **Frames 657–746:** back against the stool in a different pose. One pose, 8.7° from the baseline.

**Validation** (mask IoU with person-occluded pixels excluded; the stool is not excluded here):

| phase | IoU baseline | IoU mine |
|---|---|---|
| start at rest | 0.40 | 0.55 |
| carried | 0.30 | 0.47 |
| end at rest | 0.32 | 0.43 |

The strongest check: on carried frames where I hid the image and let the pose come only from neighbouring frames, mask IoU is 0.65 against the baseline's 0.39. The appearance score also improves in every phase.

**Depth:** kept on the baseline, with a correction of 0. The three cues disagree far beyond the 2 cm rule:

| cue | suggested shift |
|---|---|
| floor contact (my static poses) | 9–10 cm closer |
| object size vs mask size | 16–21 cm closer |
| hand position | about 3 cm closer |

Masks are about 10% too large relative to the mesh, partly because SAM2 masks run 3–5% larger than the true silhouette (measured on the frame-0 human). I fit with one global correction for that and then discard it, so it does not move depth.

**Main caveat:** at baseline depth, my static poses push the vacuum's foot about 2.5 cm further below the floor than the baseline does. Forcing floor contact during the fit hurt the image fit badly (IoU 0.43 → 0.37), so I left it off. The likely explanation is that the baseline depth is a few cm too far, but I can't confirm that without ground truth.

**Confidence:**
- **Start at rest:** medium; the roll is uncertain by about ±10–15°, though the long-axis direction is reliable.
- **End at rest:** medium-high.
- **Carried:** medium; turns such as around frame 316 and frames hidden behind the stool are still doubtful.

Nothing was downloaded; the only extra model used is the provided SAM2, once, to segment the stool.

Files are in `/work/outputs`:
- object_poses_world.npz
- REPORT.md
- TOOLS.md
- solve.py

## Final

I've written a full 747-frame trajectory to `/work/outputs/object_poses_world.npz`. Mask fit and smoothness are clearly better than the baseline, but our static poses sit about 2.5 cm lower against the floor than the baseline does. That trade-off is unresolved.

The final `solve.py` run took 196 s offline and the output is finite with valid rotations. Total GPU time across all experiments was about 35 minutes (estimated), under 9 GB.

**Phases** (detected automatically from the masks, checked by eye):
- **Frames 0–124:** the vacuum leans on the bar stool. I fit one pose for the whole phase instead of the baseline's per-frame rotation, which wanders by up to 11°. Mine is 13° from the baseline, mostly a roll about the long axis.
- **Frames 125–656:** carried and used.
  - **Flip fixed:** at frames 244–268 the baseline shows the vacuum's red front while the video shows its grey back with the cord. My result shows the back.
  - **Jitter reduced:** rotation change per frame drops from a median of 3.5° (95th percentile 17.6°) to 1.9° (7.6°).
- **Frames 657–746:** back against the stool in a different pose. One pose, 8.7° from the baseline.

**Validation** (mask IoU with person-occluded pixels excluded; the stool is not excluded here):

| phase | IoU baseline | IoU mine |
|---|---|---|
| start at rest | 0.40 | 0.55 |
| carried | 0.30 | 0.47 |
| end at rest | 0.32 | 0.43 |

The strongest check: on carried frames where I hid the image and let the pose come only from neighbouring frames, mask IoU is 0.65 against the baseline's 0.39. The appearance score also improves in every phase.

**Depth:** kept on the baseline, with a correction of 0. The three cues disagree far beyond the 2 cm rule:

| cue | suggested shift |
|---|---|
| floor contact (my static poses) | 9–10 cm closer |
| object size vs mask size | 16–21 cm closer |
| hand position | about 3 cm closer |

Masks are about 10% too large relative to the mesh, partly because SAM2 masks run 3–5% larger than the true silhouette (measured on the frame-0 human). I fit with one global correction for that and then discard it, so it does not move depth.

**Main caveat:** at baseline depth, my static poses push the vacuum's foot about 2.5 cm further below the floor than the baseline does. Forcing floor contact during the fit hurt the image fit badly (IoU 0.43 → 0.37), so I left it off. The likely explanation is that the baseline depth is a few cm too far, but I can't confirm that without ground truth.

**Confidence:**
- **Start at rest:** medium; the roll is uncertain by about ±10–15°, though the long-axis direction is reliable.
- **End at rest:** medium-high.
- **Carried:** medium; turns such as around frame 316 and frames hidden behind the stool are still doubtful.

Nothing was downloaded; the only extra model used is the provided SAM2, once, to segment the stool.

Files are in `/work/outputs`:
- object_poses_world.npz
- REPORT.md
- TOOLS.md
- solve.py
