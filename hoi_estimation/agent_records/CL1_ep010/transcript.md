# CL1_ep010

Claude Opus 完整轨迹，第一批  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-20T21:10:39Z

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
and translation both matter; a  …[+10236 chars]
```
</details>

**Bash**
```
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; nvidia-smi; ls /pipeline /weights /code/reconstruction
```

<details><summary>output</summary>

```
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 01:04 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 01:04 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 01:04 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 01:04 proxy.sock

inputs:
total 88928
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 01:04 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 01:04 ..
-rw-rw-r-- 1 hanzhou hanzhou      830 Oct 21 01:04 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       65 Oct 21 01:04 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   192843 Oct 21 01:04 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      397 Oct 21 01:04 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  8114342 Oct 21 01:04 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 11006244 Oct 21 01:04 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 4021 …[+4631 chars]
```
</details>

**Bash**
```
ls -R /work/scratch | head; cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null
```

<details><summary>output</summary>

```
Exit code 1
/work/scratch:
```
</details>

**Bash**
```
cd /work; cat logs/run_manifest.txt; echo $CUDA_VISIBLE_DEVICES; python -c "
import numpy as np, trimesh
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m, m.bounds, m.extents, len(m.vertices), len(m.faces))
print(m.visual.kind if hasattr(m.visual,'kind') else None)
r=np.load('inputs/reconstruction.npz')
for k in r.files: print(k, r[k].shape, r[k].dtype)
"; find /pipeline -maxdepth 3 | head -50; cat /pipeline/pipeline_report.json | head -100
```

<details><summary>output</summary>

```
date_utc: 2026-10-20T21:10:39Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL1_ep010:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000010:/pipeline:ro
task_sha256: ed1b646f904146c1d2901e69c6c66646f6d940bf18f332777f03f95517c92df6
cd85cf01df57ebc6ac0769e00ca0ad731c105d35b478d94d571dac47a688be47  ./camera.json
56bbe57298133c7ca8a309ddb410f4c2652393800107e16d21ebcfcb8eada195  ./episode.json
6ab22fb0521bff82de39ffc5916711380a329bb89fea24a998381808 …[+6012 chars]
```
</details>

**Bash**
```
mkdir -p /work/scratch/frames /work/outputs /work/tools; cd /work/scratch; python -c "
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
i=0
while True:
  ok,f=cap.read()
  if not ok: break
  if i%50==0: cv2.imwrite(f'frames/{i:04d}.jpg', cv2.resize(f,(768,576)))
  i+=1
print(i)
"; ls frames
```

<details><summary>output</summary>

```
30.0 794.0
794
0000.jpg
0050.jpg
0100.jpg
0150.jpg
0200.jpg
0250.jpg
0300.jpg
0350.jpg
0400.jpg
0450.jpg
0500.jpg
0550.jpg
0600.jpg
0650.jpg
0700.jpg
0750.jpg
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import cv2,numpy as np
ims=[cv2.imread(f'frames/{i:04d}.jpg') for i in range(0,800,50)]
ims=[cv2.resize(x,(384,288)) for x in ims]
for k,i in enumerate(range(0,800,50)): cv2.putText(ims[k],str(i),(10,30),0,1,(0,0,255),2)
rows=[np.hstack(ims[r*4:r*4+4]) for r in range(4)]
cv2.imwrite('grid.jpg',np.vstack(rows))
"
```

**Read**
```
/work/scratch/grid.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(float)
cam=np.array(__import__('json').load(open('/work/inputs/camera.json'))['T_world_cam'])
Tc=np.linalg.inv(cam)@T
for i in range(0,794,20):
  rv=R.from_matrix(T[i,:3,:3]).as_rotvec()
  d=np.degrees(np.linalg.norm(R.from_matrix(T[i,:3,:3]@T[max(i-1,0),:3,:3].T).as_rotvec()))
  fp=r['T_cam_mesh_foundationpose'][i]; cc=r['T_cam_mesh_coconet'][i]
  print(i, np.round(T[i,:3,3],3), np.round(np.degrees(rv),1), round(d,2), 'camz',round(Tc[i,2,3],3),'fpz',round(fp[2,3],3),'ccz',round(cc[2,3],3), np.round(r['contact_prob_right_left'][i],2))
"
```

<details><summary>output</summary>

```
0 [0.492 0.475 2.872] [38.  33.8 86.1] 0.0 camz 2.74 fpz 2.735 ccz 2.663 [0. 0.]
20 [0.488 0.473 2.874] [38.1 34.1 85.9] 0.07 camz 2.742 fpz 2.78 ccz 2.672 [0. 0.]
40 [0.493 0.472 2.859] [37.7 33.9 86.6] 0.33 camz 2.727 fpz 2.78 ccz 2.676 [0. 0.]
60 [0.492 0.477 2.883] [37.7 33.9 86.4] 0.18 camz 2.751 fpz 2.732 ccz 2.683 [0. 0.]
80 [0.498 0.477 2.892] [37.5 33.5 87. ] 0.37 camz 2.76 fpz 2.752 ccz 2.681 [0. 0.]
100 [0.495 0.481 2.879] [37.4 33.  86.6] 0.12 camz 2.747 fpz 2.778 ccz 2.671 [0. 0.]
120 [0.499 0.476 2.865] [36.8 33.  86.8] 0.29 camz 2.733 fpz 2.755 ccz 2.654 [0. 0.]
140 [0.494 0.48  2.879] [37.3 33.2 86.7] 0.45 camz 2.747 fpz 2.699 ccz 2.648 [0.   0.06]
160 [0.492 0.478 2.876] [37.6 33.1 86.4] 0.2 camz 2.744 fpz 2.736 ccz 2.645 [0.98 0.98]
180 [0.477 0.476 2.863] [38.7 33.7 86.1 …[+2638 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import h5py
f=h5py.File('/work/inputs/masks.h5','r')
for k in f: print(k,f[k].shape,f[k].dtype,f[k].chunks,f[k].compression)
import numpy as np
o=f['object']
a=[o[i].sum() for i in range(0,794,20)]
print(a)
"; python -c "import nvdiffrast, torch; print(torch.__version__, nvdiffrast.__version__)"; ls /pipeline/depth/moge2 /pipeline/export/episode_000010/*| head -30; cat /pipeline/depth/moge2/depth_report.json
```

<details><summary>output</summary>

```
object (794, 1152, 1536) bool (25, 72, 96) gzip
person (794, 1152, 1536) bool (25, 72, 96) gzip
[29554, 28891, 28918, 28935, 29096, 21490, 14707, 19801, 23292, 21940, 21340, 18545, 17773, 17879, 17419, 18062, 18231, 18971, 19882, 21195, 21448, 22602, 24724, 30152, 33022, 34214, 33788, 31618, 31201, 31691, 31646, 31859, 33674, 34041, 34031, 34043, 32140, 26788, 21435, 18891]
2.5.1+cu124 0.4.0
/pipeline/export/episode_000010/edex
/pipeline/export/episode_000010/wild_export.json

/pipeline/depth/moge2:
aligned.h5
depth_report.json
intrinsics.pkl
raw.npy

/pipeline/export/episode_000010/human_masks:
front_stereo_camera_left.h5

/pipeline/export/episode_000010/images:
front_stereo_camera_left.h5

/pipeline/export/episode_000010/object_masks:
front_stereo_camera_left.h5

/pipeline/export/episode …[+1849 chars]
```
</details>

**Write**
```
/work/outputs/hoi_common.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/hoi_common.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/scratch; cat > overlay.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
r=np.load(f'{I}/reconstruction.npz')
src=sys.argv[1] if len(sys.argv)>1 else 'base'
frames=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [0,200,300,400,450,500,550,600,750]
if src=='base': T=np.linalg.inv(Twc)@r['T_world_mesh'].astype(float); KK=K
elif src=='fp': T=r['T_cam_mesh_foundationpose'].astype(float); KK=Kp
elif src=='cc': T=r['T_cam_mesh_coconet'].astype(float); KK=Kp
else: T=np.linalg.inv(Twc)@np.load(src)['T_world_mesh']; KK=K
R=Renderer(v,f,uv,tex)
ims=read_video(f'{I}/video.mp4',frames)
om=load_masks(I,frames)
tiles=[]
for k,i in enumerate(frames):
    mk,d,c=R.render(torch.tensor(T[i:i+1],dtype=torch.float32,device='cuda'),KK,W,H,True)
    mk=mk[0].cpu().numpy()>0.5
    im=ims[i].copy().astype(float)
    cnt,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    im=im.astype(np.uint8).copy()
    cnt2,_=cv2.findContours(om[k].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im,cnt2,-1,(0,255,0),2)
    cv2.drawContours(im,cnt,-1,(255,0,0),2)
    ys,xs=np.where(om[k]|mk)
    if len(ys)==0: ys,xs=np.array([0,H-1]),np.array([0,W-1])
    y0,y1,x0,x1=max(ys.min()-60,0),min(ys.max()+60,H),max(xs.min()-60,0),min(xs.max()+60,W)
    crop=im[y0:y1,x0:x1]
    s=400/max(crop.shape[:2]); crop=cv2.resize(crop,None,fx=s,fy=s)
     …[+355 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/ov_base.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
m,v,f,uv,tex=load_mesh(I)
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
r=np.load(I+'/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
vs=v[::20].astype(float)
for i in range(0,794,25):
  p=vs@T[i,:3,:3].T+T[i,:3,3]; h=p@n+d
  print(i, round(h.min(),3), round(np.percentile(h,1),3))
hv=np.load(I+'/frame0_reference_human.npz')['vertices_world']; print('human0 min h', (hv@n+d).min())
hr=r['human_vertices_world']; print('recon human min h per frame', [round(float((hr[i]@n+d).min()),3) for i in range(0,794,50)])
"

```

<details><summary>output</summary>

```
0 0.025 0.033
25 0.034 0.041
50 0.024 0.033
75 0.024 0.035
100 0.02 0.027
125 0.021 0.028
150 0.022 0.029
175 0.024 0.032
200 0.03 0.037
225 0.019 0.027
250 0.019 0.026
275 0.001 0.009
300 0.006 0.025
325 0.027 0.051
350 0.016 0.048
375 -0.001 0.033
400 -0.001 0.032
425 -0.002 0.029
450 -0.003 0.035
475 -0.026 0.017
500 -0.016 0.035
525 0.02 0.055
550 0.017 0.058
575 -0.006 0.01
600 0.004 0.013
625 0.009 0.015
650 -0.021 -0.016
675 -0.019 -0.014
700 -0.031 -0.026
725 -0.009 -0.004
750 -0.001 0.007
775 0.002 0.017
human0 min h 0.01438993721254156
recon human min h per frame [0.011, -0.003, 0.011, 0.017, 0.011, 0.015, 0.011, 0.034, 0.022, 0.031, 0.054, 0.007, 0.003, 0.024, 0.03, 0.011]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import pickle,h5py,numpy as np
d=pickle.load(open('/pipeline/depth/moge2/intrinsics.pkl','rb')); print(type(d), d.keys() if hasattr(d,'keys') else len(d))
for k in (d.keys() if hasattr(d,'keys') else []): 
  x=d[k]; print(k, getattr(x,'shape',x) if not isinstance(x,(int,float,str)) else x)
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def p(n,o):
  if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype)
f.visititems(p)
for k in f.attrs: print(k,f.attrs[k])
"; ls -la /pipeline/foundationpose /pipeline/foundationpose/debug | head; ls /pipeline/work/masks/0 | head
```

<details><summary>output</summary>

```
<persisted-output>
Output too large (81.2KB). Full output saved to: /work/scratch/.claude/projects/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tool-results/bo3boxoid.txt

Preview (first 2KB):
<class 'dict'> dict_keys(['fx', 'fy', 'cx', 'cy', 'focals', 'per_frame_intrinsics', 'H', 'W', 'depth_backend', 'camera_policy', 'model_id', 'model_revision', 'source_commit', 'batch_size', 'devices'])
fx 797.921630859375
fy 797.921630859375
cx 768.0
cy 576.0
focals [797.921630859375, 796.6846923828125, 796.721923828125, 799.8682861328125, 797.6160278320312, 802.0446166992188, 799.952880859375, 802.4078369140625, 802.187255859375, 802.23046875, 802.09814453125, 802.1585083007812, 806.25341796875, 804.033203125, 803.7257080078125, 804.0782470703125, 804.8970336914062, 803.267578125, 804.59619140625, 803. …[+1415 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import h5py
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def p(n,o):
  if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype)
f.visititems(p)
print(list(f.attrs.items())[:20])
" 2>&1 | head -30; ls -la /pipeline/foundationpose /pipeline/foundationpose/debug | head; ls /pipeline/work/masks/0 | head -3; ls /pipeline/work/masks/0 | wc -l
```

<details><summary>output</summary>

```
aligned/front_stereo_camera_left (794,) object
alignment/front_stereo_camera_left/scale (794,) float32
alignment/front_stereo_camera_left/shift (794,) float32
alignment/front_stereo_camera_left/valid_count (794,) int32
frame_names/front_stereo_camera_left (794,) object
raw/front_stereo_camera_left (794,) object
[('complete', True), ('depth_alignment_input_identity_json', '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-normal","depth_model_revision":"b135031bae30b5ac2ae141a0e68717795ce38340","depth_source_commit":"925b8ed835a7a9cdb7578ba15c658a0afc969030","direct_mhr":{"mtime_ns":1791211510344776417,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000010/mhr_init/episode_000010.pkl","size":3543729},"export":{"mtime_ns":1791211383715776900,"path":"/home/hanzhou/Projects/v2 …[+931 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > phase_probe.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
om=load_masks(I,None,0.25); pm=load_masks(I,None,0.25,'person')
np.save('om_q.npy',om); np.save('pm_q.npy',pm)
def iou(a,b,valid):
    a=a&valid; b=b&valid
    return (a&b).sum()/max((a|b).sum(),1)
ref0=om[0]; refL=om[-1]
for i in range(0,794,10):
    v=~(pm[i]|pm[0]); vL=~(pm[i]|pm[-1]); vp=~(pm[i]|pm[max(i-10,0)])
    print(i, round(iou(om[i],ref0,v),3), round(iou(om[i],refL,vL),3), round(iou(om[i],om[max(i-10,0)],vp),3), om[i].sum(), (pm[i]&cv2.dilate(om[i].astype(np.uint8),np.ones((9,9))).astype(bool)).sum())
EOF
python phase_probe.py
```

<details><summary>output</summary>

```
0 1.0 0.019 1.0 1817 0
10 0.977 0.019 0.977 1833 0
20 0.972 0.019 0.972 1782 0
30 0.979 0.019 0.972 1833 0
40 0.971 0.019 0.972 1782 0
50 0.978 0.019 0.978 1823 0
60 0.97 0.019 0.977 1781 0
70 0.976 0.021 0.976 1825 0
80 0.972 0.022 0.992 1801 44
90 0.965 0.022 0.98 1699 243
100 0.967 0.001 0.963 1318 309
110 0.967 0.002 0.973 1001 244
120 0.969 0.003 0.966 904 164
130 0.949 0.002 0.97 1069 330
140 0.943 0.014 0.98 1206 469
150 0.963 0.019 0.941 1428 436
160 0.961 0.018 0.958 1434 467
170 0.957 0.017 0.954 1409 456
180 0.857 0.012 0.882 1321 440
190 0.848 0.013 0.971 1304 463
200 0.817 0.011 0.939 1303 465
210 0.812 0.009 0.98 1245 489
220 0.784 0.006 0.951 1128 524
230 0.746 0.006 0.936 1072 537
240 0.746 0.003 0.972 1060 513
250 0.779 0.006 0.939 1118 536
260 0.772 0.002 0.962 1096 531
2 …[+1626 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > montage.py <<'EOF'
import sys,cv2,numpy as np
sys.path.insert(0,'/work/outputs')
from hoi_common import read_video
fr=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
x0,y0,x1,y1=[int(v) for v in sys.argv[4].split(',')]
ims=read_video('/work/inputs/video.mp4',fr)
t=[]
for i in fr:
    c=ims[i][y0:y1,x0:x1][:,:,::-1].copy(); s=300/max(c.shape[:2]); c=cv2.resize(c,None,fx=s,fy=s)
    cv2.putText(c,str(i),(5,25),0,0.8,(0,255,255),2); t.append(c)
nc=int(sys.argv[5]) if len(sys.argv)>5 else 5
while len(t)%nc: t.append(np.zeros_like(t[0]))
cv2.imwrite(sys.argv[6] if len(sys.argv)>6 else 'mont.jpg',np.vstack([np.hstack(t[j:j+nc]) for j in range(0,len(t),nc)]))
EOF
python montage.py 160 310 10 250,250,1100,1050 5
```

**Read**
```
/work/scratch/mont.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_common.py <<'EOF'


def so3_exp(w):
    """w (...,3) -> (...,3,3) rotation, differentiable."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)[..., None]
    k = w / th[..., 0]
    Kx = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2]; Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def compose(T0, w, dt, c):
    """T0 (B,4,4) torch; perturb rotation about mesh point c (3,) (in camera frame, left-multiplied)."""
    R0 = T0[:, :3, :3]; t0 = T0[:, :3, 3]
    Rd = so3_exp(w)
    pc = (R0 @ c[:, None])[..., 0] + t0  # centre in cam
    R = Rd @ R0
    t = pc + dt - (R @ c[:, None])[..., 0]
    T = torch.zeros_like(T0)
    T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
    return T


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.transpose(0, 2, 1) @ Rb, axis1=1, axis2=2) - 1) / 2 if Ra.ndim == 3 else (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
EOF
cd /work/scratch; cat > fit_static.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'; s=0.5
K,Twc,Kp=load_camera(I); Ks=scale_K(K,s)
m,v,f,uv,tex=load_mesh(I)
r=np.load(f'{I}/reconstruction.npz')
Tb=np.linalg.inv(Twc)@r['T_world_mesh'] …[+1465 chars]
```

<details><summary>output</summary>

```
baseline iou 0.7361165881156921 per-frame baseline mean iou 0.7528894320130348
0 1800.8465576171875 0.7361165881156921
50 911.4910278320312 0.8516238331794739
100 904.14111328125 0.8531923294067383
150 916.0654296875 0.8515045046806335
200 914.2097778320312 0.8514364361763
250 907.0011596679688 0.8519647121429443
final iou 0.8522747755050659 dt [[-0.01807364  0.00335794 -0.03867409]] rot deg 1.4422417209638536
cam t new [0.36037725 0.39883819 2.7220974 ] base [0.37888222 0.39596234 2.76029784]
```
</details>

**Bash**
```
cd /work/scratch; cat > floorcheck.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
for p in sys.argv[1:]:
    T=Twc@np.load(p)
    h=(v.astype(float)@T[:3,:3].T+T[:3,3])@n+d
    print(p, 'min h',h.min(), 'p0.5',np.percentile(h,0.5), 'p2', np.percentile(h,2))
EOF
python floorcheck.py static_0_160.npy
```

<details><summary>output</summary>

```
static_0_160.npy min h 0.023896550267149097 p0.5 0.029417502852129074 p2 0.034290450774953074
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
T=Twc@np.load('static_0_160.npy')
print('up dir in mesh coords', -T[:3,:3].T@np.array([0,1,0.]), 'plane normal in mesh', T[:3,:3].T@n)
h=(v.astype(float)@T[:3,:3].T+T[:3,3])@n+d
low=v[h<h.min()+0.02]
print(len(low)); 
import collections
print(np.round(low[::max(1,len(low)//40)],3))
"

```

<details><summary>output</summary>

```
up dir in mesh coords [-0.99756983  0.0566565   0.04050865] plane normal in mesh [-0.99942716  0.03116901 -0.01305081]
15528
[[ 0.727  0.246 -0.327]
 [ 0.698 -0.185 -0.327]
 [ 0.701 -0.287 -0.385]
 [ 0.699 -0.19  -0.394]
 [ 0.703 -0.277  0.376]
 [ 0.696 -0.04   0.406]
 [ 0.695 -0.189  0.347]
 [ 0.697 -0.29   0.4  ]
 [ 0.699 -0.228 -0.307]
 [ 0.698 -0.279 -0.315]
 [ 0.727  0.198 -0.331]
 [ 0.7   -0.212 -0.392]
 [ 0.696 -0.143  0.406]
 [ 0.698 -0.273  0.405]
 [ 0.718  0.158 -0.339]
 [ 0.696 -0.293 -0.333]
 [ 0.703 -0.224 -0.314]
 [ 0.719  0.244 -0.333]
 [ 0.686 -0.301  0.366]
 [ 0.721  0.155 -0.334]
 [ 0.705 -0.251 -0.355]
 [ 0.699  0.136  0.345]
 [ 0.698 -0.212 -0.314]
 [ 0.696 -0.011  0.406]
 [ 0.722  0.199 -0.391]
 [ 0.707 -0.253 -0.357]
 [ 0.702  0.228  0.346]
 [ 0.701 -0.276 -0.332]
 [  …[+309 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3])
# mesh support plane: convex hull facets
hull=m.convex_hull
fn=hull.face_normals; ar=hull.area_faces
big=fn[:,0]>0.9
print('support normal(s) near +x:', (fn[big]*ar[big,None]).sum(0)/ar[big].sum(), ar[big].sum())
for p in ['static_0_160.npy']:
  T=Twc@np.load(p); up=T[:3,:3]@np.array([-1,0,0.])
  print('desk up in world',up,'angle to -y',np.degrees(np.arccos(-up[1])),'angle to plane n',np.degrees(np.arccos(up@n)))
r=np.load(I+'/reconstruction.npz'); Tb=r['T_world_mesh'][0].astype(float); up=Tb[:3,:3]@np.array([-1,0,0.])
print('baseline desk up',up,np.degrees(np.arccos(-up[1])),np.degrees(np.arccos(up@n)))
print('angle plane n vs -y', np.degrees(np.arccos(-n[1])))
# human ref: 
"

```

<details><summary>output</summary>

```
support normal(s) near +x: [ 0.99775508 -0.03336904  0.01333642] 0.45104272258360234
desk up in world [ 0.01258945 -0.99756983 -0.06850129] angle to -y 3.995252792572566 angle to plane n 1.9394358573834478
baseline desk up [ 0.00470445 -0.99872065 -0.05034959] 2.8985401755535634 1.3934622583818959
angle plane n vs -y 3.4006447666509754
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
z=np.load(I+'/frame0_reference_human.npz'); print(z.files)
hv=z['vertices_world']; print(hv.shape)
Tcw=np.linalg.inv(Twc); pc=hv@Tcw[:3,:3].T+Tcw[:3,3]
uvp=pc@K.T; uvp=uvp[:,:2]/uvp[:,2:]
pm=load_masks(I,[0],1,'person')[0]
u=np.round(uvp).astype(int); ok=(u[:,0]>=0)&(u[:,0]<W)&(u[:,1]>=0)&(u[:,1]<H)
print('frac verts in person mask', pm[u[ok,1],u[ok,0]].mean(), 'depth', pc[:,2].mean())
ys,xs=np.where(pm); print('mask bbox',xs.min(),xs.max(),ys.min(),ys.max()); print('proj bbox',uvp[:,0].min(),uvp[:,0].max(),uvp[:,1].min(),uvp[:,1].max())
"

```

<details><summary>output</summary>

```
['vertices_world']
(18056, 3)
frac verts in person mask 0.943785999113868 depth 2.578587411197946
mask bbox 448 601 384 946
proj bbox 453.0504404513873 598.2695033245806 390.3413512540486 944.3682818581008
```
</details>

**Bash**
```
cd /work/scratch; cat > vis_static.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
fr=int(sys.argv[2]); T=np.load(sys.argv[1])
R=Renderer(v,f,uv,tex)
mk,d,c=R.render(torch.tensor(T[None],dtype=torch.float32,device='cuda'),K,W,H,True)
mk=mk[0].cpu().numpy()>0.5; c=(c[0].cpu().numpy()*255).astype(np.uint8)
im=read_video(I+'/video.mp4',[fr])[fr]
om=load_masks(I,[fr])[0]; pm=load_masks(I,[fr],1,'person')[0]
diff=np.zeros_like(im); diff[om&~mk]=(0,255,0); diff[mk&~om]=(255,0,0); diff[mk&om]=(255,255,255); diff[pm&~om]=(diff[pm&~om]//2+[0,0,100])
ys,xs=np.where(om|mk); y0,y1,x0,x1=ys.min()-30,ys.max()+30,xs.min()-30,xs.max()+30
blend=im.copy(); blend[mk]=(0.5*im[mk]+0.5*c[mk]).astype(np.uint8)
out=np.hstack([im[y0:y1,x0:x1],blend[y0:y1,x0:x1],diff[y0:y1,x0:x1]])
cv2.imwrite(sys.argv[3],out[:,:,::-1])
EOF
python vis_static.py static_0_160.npy 60 vs.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vs.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > eval_iou.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'; s=0.5
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
r=np.load(f'{I}/reconstruction.npz')
srcs={'base':(np.linalg.inv(Twc)@r['T_world_mesh'].astype(float),K),'fp':(r['T_cam_mesh_foundationpose'].astype(float),Kp),'cc':(r['T_cam_mesh_coconet'].astype(float),Kp)}
for p in sys.argv[1:]:
    srcs[p]=(np.linalg.inv(Twc)@np.load(p)['T_world_mesh'],K)
R=Renderer(v,f)
h,w=int(H*s),int(W*s)
fr=list(range(0,794))
res={k:np.zeros(794) for k in srcs}
fh=h5py.File(I+'/masks.h5','r')
for a in range(0,794,50):
    idx=list(range(a,min(a+50,794)))
    om=torch.tensor(load_masks(I,idx,s),device='cuda'); pm=torch.tensor(load_masks(I,idx,s,'person'),device='cuda')
    valid=~pm|om
    for k,(T,KK) in srcs.items():
        mk,_,_=R.render(torch.tensor(T[idx],dtype=torch.float32,device='cuda'),scale_K(KK,s),w,h)
        mk=(mk>0.5)&valid
        o=om&valid
        res[k][idx]=((mk&o).sum((1,2))/(mk|o).sum((1,2)).clamp(min=1)).cpu().numpy()
np.savez('iou_eval.npz',**res)
for a in range(0,794,25):
    print(a, ' '.join(f'{k}:{res[k][a:a+25].mean():.3f}' for k in res))
print('all', ' '.join(f'{k}:{res[k].mean():.3f}' for k in res))
EOF
python eval_iou.py
```

<details><summary>output</summary>

```
0 base:0.755 fp:0.743 cc:0.748
25 base:0.751 fp:0.692 cc:0.729
50 base:0.768 fp:0.765 cc:0.755
75 base:0.768 fp:0.734 cc:0.766
100 base:0.752 fp:0.679 cc:0.748
125 base:0.764 fp:0.701 cc:0.773
150 base:0.777 fp:0.702 cc:0.791
175 base:0.793 fp:0.696 cc:0.798
200 base:0.780 fp:0.688 cc:0.786
225 base:0.712 fp:0.663 cc:0.727
250 base:0.714 fp:0.665 cc:0.731
275 base:0.695 fp:0.601 cc:0.707
300 base:0.721 fp:0.570 cc:0.731
325 base:0.696 fp:0.667 cc:0.730
350 base:0.696 fp:0.681 cc:0.746
375 base:0.677 fp:0.677 cc:0.743
400 base:0.682 fp:0.752 cc:0.750
425 base:0.766 fp:0.801 cc:0.816
450 base:0.797 fp:0.851 cc:0.858
475 base:0.797 fp:0.846 cc:0.854
500 base:0.888 fp:0.872 cc:0.900
525 base:0.899 fp:0.880 cc:0.894
550 base:0.869 fp:0.866 cc:0.875
575 base:0.903 fp:0.877 cc:0.894
600 base:0.88 …[+283 chars]
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_common.py <<'EOF'


class SilhouetteData:
    """Object / person masks at reduced scale on the GPU (uint8)."""

    def __init__(self, inputs, scale=0.5, device="cuda"):
        self.s = scale
        om = load_masks(inputs, None, scale)
        pm = load_masks(inputs, None, scale, "person")
        self.obj = torch.tensor(om, device=device)
        self.valid = torch.tensor(~pm | om, device=device)
        self.h, self.w = om.shape[1:]

    def loss(self, mk, idx):
        """mk (B,h,w) soft render; idx frame indices. Returns per-frame loss (B,) normalised by mask area."""
        t = self.obj[idx].float(); v = self.valid[idx].float()
        e = ((mk - t) ** 2 * v).sum((1, 2))
        # coarse level widens the basin of attraction
        mk4 = torch.nn.functional.avg_pool2d(mk[:, None], 4)[:, 0]
        t4 = torch.nn.functional.avg_pool2d(t[:, None], 4)[:, 0]
        v4 = torch.nn.functional.avg_pool2d(v[:, None], 4)[:, 0]
        e4 = ((mk4 - t4) ** 2 * (v4 > 0.99)).sum((1, 2)) * 16
        area = t.sum((1, 2)).clamp(min=50)
        return (e + e4) / area

    def iou(self, mk, idx):
        o = self.obj[idx].bool(); v = self.valid[idx].bool(); m = (mk > 0.5) & v
        o = o & v
        return (m & o).sum((1, 2)).float() / (m | o).sum((1, 2)).clamp(min=1).float()
EOF
cd /work/scratch; cat > track.py <<'EOF'
import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'; s=0.5
K,Twc,Kp=load_camera(I); Ks=scale_K(K,s …[+1905 chars]
```

<details><summary>output</summary>

```
masks loaded 64.50189876556396
0 0.36381031572818756 65.88631010055542
50 0.16520018130540848 75.80972051620483
100 0.16263890862464905 85.75258731842041
150 0.16278601288795472 96.44237613677979
199 0.16274591386318207 106.5128984451294
280 0.68 0.833 [-2.5 -1.9 -8.5] 1.6
290 0.711 0.832 [-2.2 -2.1 -7.9] 3.0
300 0.718 0.818 [ 0.1 -0.9 -0.7] 3.0
310 0.729 0.824 [-1.5 -0.1 -0.1] 0.9
320 0.701 0.811 [-1.   0.3 -1.4] 1.8
330 0.692 0.806 [-1.4 -0.2 -3.8] 2.3
340 0.704 0.809 [-1.5 -1.  -4.9] 2.2
350 0.694 0.817 [-1.8 -0.9 -4.9] 1.5
360 0.71 0.823 [-0.9 -1.1 -6.1] 1.5
370 0.677 0.818 [-1.8 -0.8 -8.6] 3.8
380 0.685 0.816 [ -1.6  -1.4 -10.2] 2.4
390 0.666 0.818 [ -1.1  -1.6 -11.5] 3.4
400 0.683 0.821 [ -0.8  -2.  -12.5] 3.1
410 0.679 0.826 [ -1.2  -2.5 -12.2] 3.4
420 0.695 0.83 [ -0.6  -2.7 -13.2] …[+662 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import h5py,numpy as np,cv2
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
a=f['aligned/front_stereo_camera_left'][0]; print(type(a), len(a), a[:8])
im=cv2.imdecode(np.frombuffer(a,np.uint8),cv2.IMREAD_UNCHANGED); print(im.shape,im.dtype,im.min(),im.max())
print(f['alignment/front_stereo_camera_left/scale'][:10], f['alignment/front_stereo_camera_left/shift'][:10])
print(f['frame_names/front_stereo_camera_left'][:3])
"
```

<details><summary>output</summary>

```
<class 'numpy.ndarray'> 637984 [137  80  78  71  13  10  26  10]
(1152, 1536) uint16 1672 9155
[1.0577121 1.0542014 1.057247  1.056216  1.0607339 1.059262  1.0639185
 1.0657406 1.0632007 1.065914 ] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
[b'000000' b'000001' b'000002']
```
</details>

**Bash**
```
cd /work/scratch; cat > depthcmp.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
r=np.load(f'{I}/reconstruction.npz')
Tb=np.linalg.inv(Twc)@r['T_world_mesh'].astype(float)
tr=np.load(sys.argv[1]); Tt=dict(zip(tr['fr'],tr['T']))
R=Renderer(v,f)
fh=h5py.File('/pipeline/depth/moge2/aligned.h5','r')['aligned/front_stereo_camera_left']
frs=[int(x) for x in sys.argv[2].split(',')]
om=load_masks(I,frs); pm=load_masks(I,frs,1,'person')
for k,i in enumerate(frs):
    D=cv2.imdecode(np.frombuffer(fh[i],np.uint8),cv2.IMREAD_UNCHANGED)/1000.
    # MoGe depth is for K_pipeline; z-depth per pixel is ~independent of focal; compare z at the same pixels
    er=cv2.erode(om[k].astype(np.uint8),np.ones((5,5)))>0
    out=[]
    for name,T in [('base',Tb[i]),('fit',Tt[i])]:
        mk,d,_=R.render(torch.tensor(T[None],dtype=torch.float32,device='cuda'),K,W,H)
        d=d[0].cpu().numpy(); sel=er&(d>0)&~pm[k]
        out.append(f'{name}: render med {np.median(d[sel]):.3f} moge med {np.median(D[sel]):.3f} ratio {np.median(D[sel]/d[sel]):.3f} n={sel.sum()}')
    hm=pm[k]; print(i,' | '.join(out), 'human moge med', np.median(D[hm]) if hm.sum() else -1)
EOF
python depthcmp.py track_280_600_0.0.npz 300,350,400,450,500,550,590
```

<details><summary>output</summary>

```
300 base: render med 2.553 moge med 2.658 ratio 1.005 n=12619 | fit: render med 2.536 moge med 2.669 ratio 1.014 n=13481 human moge med 2.159
350 base: render med 2.700 moge med 2.598 ratio 0.960 n=13497 | fit: render med 2.630 moge med 2.597 ratio 0.981 n=14343 human moge med 2.326
400 base: render med 2.840 moge med 2.678 ratio 0.947 n=14676 | fit: render med 2.683 moge med 2.690 ratio 0.998 n=16475 human moge med 2.805
450 base: render med 3.121 moge med 2.845 ratio 0.907 n=22832 | fit: render med 3.050 moge med 2.835 ratio 0.931 n=23539 human moge med 3.109
500 base: render med 3.404 moge med 3.112 ratio 0.896 n=29085 | fit: render med 3.330 moge med 3.113 ratio 0.918 n=30281 human moge med 3.183
550 base: render med 3.463 moge med 3.416 ratio 0.972 n=26925 | fit: render med 3.369 moge …[+193 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np
T=np.load('static_0_160.npy'); np.savez('sA.npz',fr=np.arange(0,280),T=np.repeat(T[None],280,0))"; python depthcmp.py sA.npz 0,60,120,200,250
```

<details><summary>output</summary>

```
0 base: render med 2.524 moge med 2.564 ratio 1.011 n=22416 | fit: render med 2.535 moge med 2.570 ratio 1.011 n=23858 human moge med 2.319
60 base: render med 2.543 moge med 2.565 ratio 0.999 n=22256 | fit: render med 2.547 moge med 2.563 ratio 1.007 n=23400 human moge med 2.061
120 base: render med 2.898 moge med 2.920 ratio 1.036 n=10862 | fit: render med 2.902 moge med 2.922 ratio 1.031 n=11606 human moge med 1.926
200 base: render med 2.698 moge med 2.764 ratio 1.018 n=16510 | fit: render med 2.730 moge med 2.766 ratio 1.009 n=16483 human moge med 2.087
250 base: render med 2.848 moge med 2.760 ratio 0.972 n=12990 | fit: render med 2.824 moge med 2.760 ratio 0.983 n=13717 human moge med 2.089
```
</details>

**Bash**
```
cd /work/scratch; python fit_static.py 600 794 && python floorcheck.py static_600_794.npy && python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
r=np.load(I+'/reconstruction.npz'); Tb=r['T_world_mesh'].astype(float)
T=Twc@np.load('static_600_794.npy')
for i in [600,650,700,750,790]: print(i, np.round((T[:3,3]-Tb[i,:3,3])*100,1), rot_angle_deg(T[:3,:3],Tb[i,:3,:3]))
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3])
m,v,f,uv,tex=load_mesh(I)
print('plane normal in mesh', T[:3,:3].T@n)
"
```

<details><summary>output</summary>

```
baseline iou 0.853005588054657 per-frame baseline mean iou 0.8663817286491394
0 1071.093994140625 0.853005588054657
50 436.8545227050781 0.93881756067276
100 408.84857177734375 0.9423138499259949
150 383.16876220703125 0.945216715335846
200 377.7747802734375 0.9463072419166565
250 373.0958251953125 0.9465036988258362
final iou 0.9453096389770508 dt [[-0.00754991 -0.0465049  -0.14977409]] rot deg 0.8915376020900357
cam t new [-0.34748402  0.75547194  3.40876436] base [-0.33972439  0.80265866  3.55884024]
static_600_794.npy min h 0.03755109385511046 p0.5 0.042548382969768817 p2 0.046369911094575444
600 [-1.1 -2.8 -8. ] 0.8809819615723031
650 [ -0.4  -4.8 -14.1] 1.032717797142798
700 [ -0.5  -5.5 -16.3] 1.0412759884088585
750 [ -2.   -2.7 -11.3] 0.8900347791544008
790 [-2.  -1.8 -6.1] 1.30402 …[+70 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
hull=m.convex_hull; fn=hull.face_normals; ar=hull.area_faces
sel=fn[:,1]<-0.95; print('support normal -y side', (fn[sel]*ar[sel,None]).sum(0)/ar[sel].sum(), ar[sel].sum())
for x in np.unique(np.round(hull.vertices[hull.vertices[:,1]<-0.31],2),axis=0)[:40]: pass
hv=hull.vertices; lo=hv[hv[:,1]<hv[:,1].min()+0.01]; print(np.round(lo,3))
T=Twc@np.load('static_600_794.npy')
h=(v.astype(float)@T[:3,:3].T+T[:3,3])@n+d
for q in [0.03,0.04,0.05]:
  s=h<h.min()+0.005*0+q-0.03+0.005
lo=v[h<h.min()+0.01]; print('low verts in fit pose (mesh coords) x-range',lo[:,0].min(),lo[:,0].max(),'z-range',lo[:,2].min(),lo[:,2].max(), len(lo))
import collections
# height of support extremes
for name,pt in [('top-edge zmin',[-0.18,-0.318,-0.5]),('top-edge zmax',[-0.18,-0.318,0.5]),('foot zmin',[0.72,-0.30,-0.35]),('foot zmax',[0.72,-0.30,0.38])]:
  idx=np.argmin(((v-pt)**2).sum(1)); print(name, v[idx], h[idx])
"
```

<details><summary>output</summary>

```
support normal -y side [ 0.01472256 -0.99920253  0.00157769] 0.7562685978486202
[[-0.178 -0.309 -0.489]
 [-0.178 -0.31  -0.486]
 [-0.18  -0.314 -0.475]
 [-0.181 -0.315 -0.466]
 [-0.18  -0.31  -0.486]
 [-0.183 -0.318 -0.432]
 [-0.167 -0.313  0.495]
 [ 0.673 -0.308 -0.373]
 [-0.177 -0.311  0.484]
 [-0.177 -0.313  0.479]
 [-0.176 -0.313  0.485]
 [-0.174 -0.316  0.476]
 [-0.174 -0.316  0.483]
 [-0.175 -0.315  0.479]
 [-0.177 -0.311  0.48 ]
 [-0.172 -0.317  0.48 ]
 [-0.172 -0.318  0.466]
 [-0.171 -0.318  0.472]
 [-0.173 -0.317  0.472]
 [-0.17  -0.318  0.47 ]
 [-0.171 -0.312  0.496]
 [-0.171 -0.316  0.487]
 [-0.168 -0.315  0.489]
 [-0.17  -0.315  0.491]
 [-0.17  -0.318  0.463]
 [-0.176 -0.314  0.475]
 [-0.169 -0.317  0.481]
 [-0.17  -0.318  0.476]
 [-0.171 -0.317  0.479]
 [-0.171 -0.317  0.483]
 …[+1411 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
hull=m.convex_hull; fn=hull.face_normals; ar=hull.area_faces
sel=fn[:,1]>0.95; print('support normal +y side', (fn[sel]*ar[sel,None]).sum(0)/ar[sel].sum(), ar[sel].sum())
hv=hull.vertices; lo=hv[hv[:,1]>hv[:,1].max()-0.01]; print(np.round(lo[::3],3))
for nm,T in [('fit',Twc@np.load('static_600_794.npy')),('base700',np.load(I+'/reconstruction.npz')['T_world_mesh'][700].astype(float))]:
  h=(v.astype(float)@T[:3,:3].T+T[:3,3])@n+d
  lo=v[h<h.min()+0.01]
  print(nm,'min h',h.min(),'low verts x',lo[:,0].min(),lo[:,0].max(),'z',lo[:,2].min(),lo[:,2].max())
  for q in [(-0.2,-0.4),(-0.2,0.4),(0.75,-0.4),(0.75,0.4)]:
    s=(np.abs(v[:,0]-q[0])<0.1)&(np.abs(v[:,2]-q[1])<0.15); print('  region',q,'min h',h[s].min())
"
```

<details><summary>output</summary>

```
support normal +y side [ 0.03591877  0.99776991 -0.00936125] 0.770732060424954
[[-0.152  0.303  0.416]
 [-0.152  0.301  0.466]
 [-0.152  0.302  0.461]
 [-0.132  0.303  0.462]
 [-0.138  0.309  0.162]
 [-0.131  0.302  0.465]
 [-0.133  0.306  0.383]]
fit min h 0.03755109385511046 low verts x -0.15617825 0.71588147 z -0.39659703 0.49846473
  region (-0.2, -0.4) min h 0.042337450651863895
  region (-0.2, 0.4) min h 0.0398395240255951
  region (0.75, -0.4) min h 0.052965036224782436
  region (0.75, 0.4) min h 0.041005421807699616
base700 min h -0.03137486996565264 low verts x -0.16470124 -0.13018493 z -0.09111432 0.5098879
  region (-0.2, -0.4) min h -0.019145547858951062
  region (-0.2, 0.4) min h -0.03137486996565264
  region (0.75, -0.4) min h 0.0009979199791672055
  region (0.75, 0.4) min h  …[+21 chars]
```
</details>

**Bash**
```
cd /work/scratch; python vis_static.py static_600_794.npy 760 vsB.jpg; python -c "
import cv2; im=cv2.imread('vsB.jpg'); print(im.shape)"
```

<details><summary>output</summary>

```
(313, 1161, 3)
```
</details>

**Read**
```
/work/scratch/vsB.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > depthscan.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'; s=float(sys.argv[4]) if len(sys.argv)>4 else 0.5
K,Twc,Kp=load_camera(I); Ks=scale_K(K,s)
m,v,f,uv,tex=load_mesh(I)
T=np.load(sys.argv[1]); a,b=int(sys.argv[2]),int(sys.argv[3])
fr=list(range(a,b,3))
om=load_masks(I,fr,s); pm=load_masks(I,fr,s,'person'); valid=~pm|om
nobj=torch.tensor((om&valid).sum(0),dtype=torch.float32,device='cuda'); nbg=torch.tensor((~om&valid).sum(0),dtype=torch.float32,device='cuda')
R=Renderer(v,f); h,w=int(H*s),int(W*s)
c=v.mean(0)
for eps in [-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04]:
    # scale about camera centre: move the object along the viewing ray of its centre, then re-fit nothing (pure scale test)
    T2=T.copy(); T2[:3,3]=T[:3,3]*(1+eps)
    mk,_,_=R.render(torch.tensor(T2[None],dtype=torch.float32,device='cuda'),Ks,w,h); mk=mk[0]>0.5
    inter=(nobj*mk).sum(); fp=(nbg*mk).sum(); fn=(nobj*~mk).sum()
    print(eps, round((inter/(inter+fp+fn)).item(),4), 'fp',int(fp/len(fr)),'fn',int(fn/len(fr)))
EOF
python depthscan.py static_600_794.npy 700 794 1.0; python depthscan.py static_0_160.npy 0 90 1.0
```

<details><summary>output</summary>

```
-0.04 0.8049 fp 4010 fn 1775
-0.03 0.8373 fp 3231 fn 1465
-0.02 0.8735 fp 2393 fn 1153
-0.01 0.9071 fp 1647 fn 887
0 0.9351 fp 1012 fn 719
0.01 0.9338 fp 808 fn 944
0.02 0.9169 fp 812 fn 1385
0.03 0.8904 fp 963 fn 1953
0.04 0.8638 fp 1103 fn 2541
-0.04 0.7232 fp 6090 fn 3725
-0.03 0.7663 fp 4863 fn 3136
-0.02 0.7987 fp 3901 fn 2797
-0.01 0.8249 fp 3064 fn 2615
0 0.8434 fp 2429 fn 2550
0.01 0.8368 fp 2234 fn 2923
0.02 0.8101 fp 2391 fn 3642
0.03 0.7715 fp 2770 fn 4575
0.04 0.7295 fp 3240 fn 5583
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
r=np.load(I+'/reconstruction.npz')
hv=np.load(I+'/frame0_reference_human.npz')['vertices_world']; hr=r['human_vertices_world'][0]
Tcw=np.linalg.inv(Twc)
pc=lambda p: p@Tcw[:3,:3].T+Tcw[:3,3]
a,b=pc(hv),pc(hr)
print('true human cam centroid',a.mean(0),'recon',b.mean(0))
print('true extents',np.ptp(a,0),'recon',np.ptp(b,0))
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
print('true min h',(hv@n+d).min(),'recon min h',(hr@n+d).min())
kp=r['human_keypoints_world']
for i in [0,100,700,790]: print(i,'recon min h', (r['human_vertices_world'][i]@n+d).min())
"
```

<details><summary>output</summary>

```
true human cam centroid [-0.6925507   0.240674    2.57858741] recon [-0.69721017  0.20362789  2.56483396]
true extents [0.48606902 1.68349046 0.51331611] recon [0.48530463 1.68859108 0.51443904]
true min h 0.01438993721254156 recon min h 0.011383810994245902
0 recon min h 0.011383810994245902
100 recon min h 0.011104882957227558
700 recon min h 0.03026267809412242
790 recon min h 0.0036845263943863227
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
import re
s=open('hoi_common.py').read()
s=s.replace('''    def __init__(self, inputs, scale=0.5, device="cuda"):
        self.s = scale
        om = load_masks(inputs, None, scale)
        pm = load_masks(inputs, None, scale, "person")''','''    def __init__(self, inputs, scale=0.5, device="cuda", cache=None):
        self.s = scale
        import os
        if cache and os.path.exists(cache):
            z = np.load(cache); om, pm = z["om"], z["pm"]
        else:
            om = load_masks(inputs, None, scale)
            pm = load_masks(inputs, None, scale, "person")
            if cache:
                np.savez(cache, om=om, pm=pm)''')
open('hoi_common.py','w').write(s)
EOF
cd /work/scratch; sed -i "s#SD=SilhouetteData(I,s)#SD=SilhouetteData(I,s,cache='/work/scratch/masks_half.npz')#" track.py; python track.py 0 794 0 150 > track_all.log; tail -3 track_all.log
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b6xaxmmp9). Output is being written to: /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/b6xaxmmp9.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 100; cat /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/b6xaxmmp9.output; tail -5 /work/scratch/track_all.log
```

<details><summary>output</summary>

```
770 0.815 0.92 [-1.8 -2.5 -8.2] 0.5
780 0.8 0.917 [-2.3 -1.8 -5.3] 1.0
790 0.798 0.914 [-2.3 -2.3 -7.1] 0.8

[exited with code 0]
750 0.846 0.931 [ -2.3  -2.8 -12.6] 0.7
760 0.836 0.927 [-1.4 -3.  -9.5] 0.8
770 0.815 0.92 [-1.8 -2.5 -8.2] 0.5
780 0.8 0.917 [-2.3 -1.8 -5.3] 1.0
790 0.798 0.914 [-2.3 -2.3 -7.1] 0.8
```
</details>

**Agent:** Let me analyze the per-frame fit relative to the two static poses to locate phase boundaries.

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
z=np.load('track_0_794_0.0.npz'); T=z['T']; iou=z['iou']
A=np.load('static_0_160.npy'); B=np.load('static_600_794.npy')
m,v,f,uv,tex=load_mesh('/work/inputs'); vs=v[::50].astype(float)
def add(T1,T2): return np.linalg.norm((vs@T1[:3,:3].T+T1[:3,3])-(vs@T2[:3,:3].T+T2[:3,3]),axis=1).mean()
for i in list(range(0,794,10)):
  print(i, 'addA %.3f addB %.3f'%(add(T[i],A),add(T[i],B)), 'iou %.3f base %.3f'%(iou[i],z['iou0'][i]), 'dprev %.3f'%add(T[i],T[max(i-1,0)]))
" > perframe.txt; awk 'NR%2==1' perframe.txt | head -80
```

<details><summary>output</summary>

```
0 addA 0.002 addB 1.069 iou 0.850 base 0.751 dprev 0.000
20 addA 0.001 addB 1.068 iou 0.846 base 0.755 dprev 0.000
40 addA 0.001 addB 1.070 iou 0.843 base 0.770 dprev 0.003
60 addA 0.002 addB 1.068 iou 0.845 base 0.772 dprev 0.003
80 addA 0.004 addB 1.072 iou 0.855 base 0.744 dprev 0.001
100 addA 0.005 addB 1.069 iou 0.867 base 0.767 dprev 0.003
120 addA 0.017 addB 1.081 iou 0.870 base 0.722 dprev 0.003
140 addA 0.007 addB 1.067 iou 0.855 base 0.770 dprev 0.006
160 addA 0.007 addB 1.069 iou 0.871 base 0.776 dprev 0.002
180 addA 0.029 addB 1.075 iou 0.901 base 0.803 dprev 0.003
200 addA 0.032 addB 1.067 iou 0.884 base 0.804 dprev 0.001
220 addA 0.033 addB 1.066 iou 0.866 base 0.747 dprev 0.002
240 addA 0.033 addB 1.060 iou 0.869 base 0.714 dprev 0.002
260 addA 0.033 addB 1.064 iou 0.861 bas …[+1553 chars]
```
</details>

**Bash**
```
cd /work/scratch; sed -n '15,32p;56,64p' perframe.txt
```

<details><summary>output</summary>

```
140 addA 0.007 addB 1.067 iou 0.855 base 0.770 dprev 0.006
150 addA 0.006 addB 1.068 iou 0.874 base 0.758 dprev 0.002
160 addA 0.007 addB 1.069 iou 0.871 base 0.776 dprev 0.002
170 addA 0.008 addB 1.068 iou 0.872 base 0.779 dprev 0.002
180 addA 0.029 addB 1.075 iou 0.901 base 0.803 dprev 0.003
190 addA 0.029 addB 1.072 iou 0.899 base 0.805 dprev 0.002
200 addA 0.032 addB 1.067 iou 0.884 base 0.804 dprev 0.001
210 addA 0.032 addB 1.067 iou 0.882 base 0.806 dprev 0.003
220 addA 0.033 addB 1.066 iou 0.866 base 0.747 dprev 0.002
230 addA 0.032 addB 1.063 iou 0.878 base 0.722 dprev 0.002
240 addA 0.033 addB 1.060 iou 0.869 base 0.714 dprev 0.002
250 addA 0.031 addB 1.063 iou 0.873 base 0.716 dprev 0.000
260 addA 0.033 addB 1.064 iou 0.861 base 0.709 dprev 0.002
270 addA 0.041 addB 1.056 iou 0.8 …[+792 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
z=np.load('track_0_794_0.0.npz'); T=z['T']
A=np.load('static_0_160.npy')
for i in range(150,300,5):
  print(i, np.round((T[i,:3,3]-A[:3,3])*100,1), round(rot_angle_deg(T[i,:3,:3],A[:3,:3]),2), np.round(np.degrees(__import__('scipy.spatial.transform',fromlist=['x']).Rotation.from_matrix(T[i,:3,:3]@A[:3,:3].T).as_rotvec()),2))
"
```

<details><summary>output</summary>

```
150 [-0.4  0.  -0.4] 0.45 [-0.3  -0.01 -0.29]
155 [-0.4  0.1 -0.4] 0.47 [-0.12  0.04 -0.42]
160 [-0.4  0.1 -0.5] 0.51 [-0.21  0.05 -0.43]
165 [-0.4  0.1 -0.3] 0.47 [-0.21  0.08 -0.38]
170 [-0.5 -0.1 -0.5] 0.46 [-0.29  0.18 -0.25]
175 [-1.4 -0.1 -2.4] 0.9 [ 0.14  0.84 -0.23]
180 [-1.4 -0.1 -2.5] 0.98 [ 0.1   0.9  -0.34]
185 [-1.4 -0.1 -2.4] 1.01 [-0.07  0.96 -0.28]
190 [-1.6 -0.2 -2.3] 1.37 [-0.01  1.35 -0.15]
195 [-2.2  0.  -2.1] 1.71 [ 0.    1.62 -0.52]
200 [-2.2  0.1 -2.2] 1.72 [ 0.12  1.63 -0.53]
205 [-2.2  0.1 -2.2] 1.72 [ 0.16  1.63 -0.53]
210 [-2.2  0.2 -2.1] 1.71 [ 0.16  1.59 -0.57]
215 [-2.3  0.2 -2.1] 1.75 [-0.01  1.62 -0.63]
220 [-2.4  0.1 -2.2] 1.84 [-0.19  1.65 -0.77]
225 [-2.4  0.1 -2.1] 1.81 [-0.26  1.63 -0.72]
230 [-2.5  0.2 -1.9] 1.94 [-0.16  1.72 -0.86]
235 [-2.6  0.4 -1.6 …[+578 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
fr=[100,150,170,174,176,180,200,220,240,260,270,280]
ims=read_video('/work/inputs/video.mp4',fr)
om=load_masks('/work/inputs',[100])[0]
ys,xs=np.where(om); print(xs.min(),xs.max(),ys.min(),ys.max())
" 
```

<details><summary>output</summary>

```
783 1018 605 963
```
</details>

**Bash**
```
cd /work/scratch; python montage.py 150 290 10 600,350,1300,1000 5 m2.jpg
```

**Read**
```
/work/scratch/m2.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
fr=list(range(100,300,10))+[175,178]
ims=read_video('/work/inputs/video.mp4',fr)
def patch(i): return cv2.cvtColor(ims[i][600:890,850:1030],cv2.COLOR_RGB2GRAY).astype(np.float32)
ref=patch(150)
win=cv2.createHanningWindow(ref.shape[::-1],cv2.CV_32F)
for i in sorted(fr):
  p=patch(i)
  sh,resp=cv2.phaseCorrelate(ref,p,win)
  # also ECC
  wm=np.eye(2,3,dtype=np.float32)
  try:
    cc,wm=cv2.findTransformECC(ref,p,wm,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),None,5)
  except Exception as e: cc=-1
  print(i, np.round(sh,2), round(resp,3), np.round(wm[:,2],2), round(cc,3))
"
```

<details><summary>output</summary>

```
100 [ 0.02 -0.04] 0.901 [0.02 0.02] 0.997
110 [ 0.04 -0.05] 0.881 [0.07 0.  ] 0.987
120 [ 0.06 -0.01] 0.854 [0.29 0.02] 0.975
130 [0.03 0.02] 0.875 [0.02 0.01] 0.987
140 [-0.   -0.02] 0.921 [ 0.07 -0.03] 0.998
150 [-0. -0.] 1.0 [0. 0.] 1.0
160 [ 0.08 -0.  ] 0.947 [-0.02  0.02] 0.997
170 [ 0.03 -0.11] 0.924 [-0.21 -0.1 ] 0.996
175 [-0.3  -0.09] 0.891 [-0.58 -0.06] 0.996
178 [-0.35 -0.08] 0.883 [-0.66 -0.05] 0.996
180 [-0.34 -0.12] 0.878 [-0.62 -0.09] 0.995
190 [-0.49 -0.13] 0.814 [-1.05 -0.19] 0.99
200 [-0.55 -0.09] 0.804 [-1.31 -0.04] 0.991
210 [-0.56 -0.11] 0.797 [-1.28 -0.02] 0.992
220 [-0.57 -0.12] 0.757 [-1.14 -0.12] 0.969
230 [-0.54 -0.08] 0.733 [-1.09 -0.11] 0.947
240 [-0.55 -0.13] 0.735 [-1.  -0.1] 0.956
250 [-0.52 -0.12] 0.752 [-1.06 -0.14] 0.96
260 [-0.65 -0.14] 0.695 [-1.31 -0.25 …[+140 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > track2.py <<'EOF'
import sys, time, argparse; sys.path.insert(0,'/work/outputs')
from hoi_common import *
p=argparse.ArgumentParser()
p.add_argument('--a',type=int); p.add_argument('--b',type=int); p.add_argument('--lam',type=float,default=1e4)
p.add_argument('--iters',type=int,default=200); p.add_argument('--init',default='track_0_794_0.0.npz')
p.add_argument('--A',default='static_0_160.npy'); p.add_argument('--B',default='static_600_794.npy'); p.add_argument('--out')
args=p.parse_args()
I='/work/inputs'; s=0.5
K,Twc,Kp=load_camera(I); Ks=scale_K(K,s)
m,v,f,uv,tex=load_mesh(I)
t0=time.time()
SD=SilhouetteData(I,s,cache='/work/scratch/masks_half.npz')
R=Renderer(v,f)
a,b=args.a,args.b
fr=np.arange(a,b); N=len(fr)
z=np.load(args.init); Ti=dict(zip(z['fr'],z['T']))
T0=torch.tensor(np.stack([Ti[i] for i in fr]),dtype=torch.float32,device='cuda')
TA=torch.tensor(np.load(args.A),dtype=torch.float32,device='cuda'); TB=torch.tensor(np.load(args.B),dtype=torch.float32,device='cuda')
c=torch.tensor(v.mean(0),device='cuda')
bb=torch.tensor(trimesh.bounds.corners(np.stack([v.min(0),v.max(0)])),dtype=torch.float32,device='cuda')
w=torch.zeros(N,3,device='cuda',requires_grad=True); dt=torch.zeros(N,3,device='cuda',requires_grad=True)
opt=torch.optim.Adam([w,dt],lr=2e-3)
CH=40
def corners(T): return torch.einsum('nij,kj->nki',T[:,:3,:3],bb)+T[:,None,:3,3]
for it in range(args.iters):
    opt.zero_grad(); tot=0
    for k in range(0,N,CH):
        T=compose(T0[k:k+CH] …[+876 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bh92ravhp). Output is being written to: /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/bh92ravhp.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/bh92ravhp.output
```

<details><summary>output</summary>

```
199 0.16521838430822247 0.0007075949533451238 88.45957112312317
mean iou 0.8782361
199 0.16664683677741057 0.0021295147181968014 91.87908458709717
mean iou 0.87749016
```
</details>

**Bash**
```
cd /work/scratch; cat > klt.py <<'EOF'
import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
a,b=int(sys.argv[1]),int(sys.argv[2])
t0=time.time()
fr=list(range(a,b+1))
ims=read_video(I+'/video.mp4',fr)
om=load_masks(I,fr); pm=load_masks(I,fr,1,'person')
print('loaded',time.time()-t0)
out=[]
for k in range(len(fr)-1):
    i=fr[k]
    g0=cv2.cvtColor(ims[i],cv2.COLOR_RGB2GRAY); g1=cv2.cvtColor(ims[i+1],cv2.COLOR_RGB2GRAY)
    allow=cv2.erode(om[k].astype(np.uint8),np.ones((11,11)))&~cv2.dilate(pm[k].astype(np.uint8),np.ones((15,15))).astype(bool)
    allow=allow.astype(np.uint8)
    pts=cv2.goodFeaturesToTrack(g0,400,0.01,6,mask=allow,blockSize=7)
    if pts is None: continue
    p1,st,err=cv2.calcOpticalFlowPyrLK(g0,g1,pts,None,winSize=(15,15),maxLevel=3)
    p0b,st2,err2=cv2.calcOpticalFlowPyrLK(g1,g0,p1,None,winSize=(15,15),maxLevel=3)
    fb=np.linalg.norm(pts-p0b,axis=-1)[:,0]
    ok=(st[:,0]==1)&(st2[:,0]==1)&(fb<0.5)
    # target must also lie in object mask (not person) in frame i+1
    q=np.round(p1[:,0]).astype(int); q[:,0]=q[:,0].clip(0,W-1); q[:,1]=q[:,1].clip(0,H-1)
    ok&=om[k+1][q[:,1],q[:,0]]&~pm[k+1][q[:,1],q[:,0]]
    for p,qq in zip(pts[ok,0],p1[ok,0]): out.append((i,p[0],p[1],qq[0],qq[1]))
out=np.array(out)
np.save(f'klt_{a}_{b}.npy',out)
cnt=np.bincount(out[:,0].astype(int)-a,minlength=b-a)
print('time',time.time()-t0,'tracks',len(out)); print('per-frame counts (every 10):',cnt[::10])
mot=np.linalg.norm(out[:,3:5]-o …[+82 chars]
```

<details><summary>output</summary>

```
loaded 32.15400433540344
time 46.2833993434906 tracks 9430
per-frame counts (every 10): [17 11 13 12  9 21 16 16 15 19 16 21 17 23 14 16 15 17 13 29 34 32 42 39
 46 49 44 43 43 38 39 43 40 39 41]
median motion 0.4825455017249918
```
</details>

**Bash**
```
sleep 20; cat /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/bh92ravhp.output; cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
m,v,f,uv,tex=load_mesh('/work/inputs'); vs=v[::50].astype(float)
def add(T1,T2): return np.linalg.norm((vs@T1[:3,:3].T+T1[:3,3])-(vs@T2[:3,:3].T+T2[:3,3]),axis=1).mean()
z0=np.load('track_0_794_0.0.npz'); 
zs={l:np.load(f'tr2_{l}.npz') for l in ['1e3','1e4','1e5']}
for l,z in zs.items():
  T=z['T']; jit=[add(T[i],T[i-1]) for i in range(1,len(T))]; acc=[np.linalg.norm(((vs@T[i+1,:3,:3].T+T[i+1,:3,3])-2*(vs@T[i,:3,:3].T+T[i,:3,3])+(vs@T[i-1,:3,:3].T+T[i-1,:3,3])),axis=1).mean() for i in range(1,len(T)-1)]
  print(l,'iou',z['iou'].mean(),'mean step',np.mean(jit),'mean acc',np.mean(acc))
T=z0['T'][262:600]; acc=[np.linalg.norm(((vs@T[i+1,:3,:3].T+T[i+1,:3,3])-2*(vs@T[i,:3,:3].T+T[i,:3,3])+(vs@T[i-1,:3,:3].T+T[i-1,:3,3])),axis=1).mean() for i in range(1,len(T)-1)]
print('perframe iou',z0['iou'][262:600].mean(),'acc',np.mean(acc))
for i in range(0,338,15): print(262+i, ' '.join('%.3f'%add(zs[l]['T'][i],z0['T'][262+i]) for l in zs), ' '.join('%.3f'%zs[l]['iou'][i] for l in zs), '%.3f'%z0['iou'][262+i])
"
```

<details><summary>output</summary>

```
199 0.16521838430822247 0.0007075949533451238 88.45957112312317
mean iou 0.8782361
199 0.16664683677741057 0.0021295147181968014 91.87908458709717
mean iou 0.87749016
199 0.168112036039138 0.014127904846823428 87.75103235244751
mean iou 0.87671006

[exited with code 0]
1e3 iou 0.8782361 mean step 0.004016858329743274 mean acc 0.00044454327710993796
1e4 iou 0.87749016 mean step 0.0039493804534814665 mean acc 0.000300505491113101
1e5 iou 0.87671006 mean step 0.00393058204779735 mean acc 0.00026787776306152957
perframe iou 0.8785635 acc 0.006235176578053635
262 0.026 0.032 0.033 0.779 0.746 0.728 0.858
277 0.002 0.003 0.004 0.840 0.840 0.845 0.837
292 0.002 0.002 0.002 0.840 0.844 0.844 0.844
307 0.008 0.009 0.009 0.829 0.827 0.827 0.832
322 0.003 0.005 0.005 0.823 0.820 0.818 0.823
337 0.001 …[+818 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
K,Twc,Kp=load_camera('/work/inputs')
z=np.load('tr2_1e4.npz'); A=np.load('static_0_160.npy'); B=np.load('static_600_794.npy')
T=np.zeros((794,4,4)); T[:262]=A; T[600:]=B; T[262:600]=z['T']
np.savez('cand1.npz',T_world_mesh=Twc@T)
"; python overlay.py cand1.npz 270,300,330,360,390,420,440,470,560
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/ov_cand1.npz.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The silhouette refit improved mask IoU from ~0.79 to ~0.88, placing the object 4–15 cm closer to the camera in the static phases than the baseline, consistent with MoGe depth and corner-resting heights. Next I'll check flip hypotheses, add feature-track constraints for the occluded moving phase, and fuse in depth cues.

**Bash**
```
cat >> /work/outputs/hoi_common.py <<'EOF'


def pooled_counts(SD, frames):
    """For a static phase: per-pixel number of frames where the pixel is visible object / visible background."""
    o = SD.obj[frames].float(); v = SD.valid[frames].float()
    return (o * v).sum(0), ((1 - o) * v).sum(0)


def fit_static_pose(R, Ks, h, w, nobj, nbg, T_init, iters=250, lr=2e-3, extra=None):
    """Fit one pose to pooled silhouette evidence of a static phase. Returns (T (4,4) np, iou)."""
    dev = nobj.device
    T0 = torch.tensor(T_init[None], dtype=torch.float32, device=dev)
    c = R.v.mean(0)
    wv = torch.zeros(1, 3, device=dev, requires_grad=True); dt = torch.zeros(1, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([wv, dt], lr=lr)
    nf = (nobj + nbg).max()
    for it in range(iters):
        T = compose(T0, wv, dt, c)
        mk, _, _ = R.render(T, Ks, w, h)
        m = mk[0]
        L = (nobj * (1 - m) ** 2 + nbg * m ** 2).sum() / nf
        m4 = torch.nn.functional.avg_pool2d(m[None, None], 4)[0, 0]
        o4 = torch.nn.functional.avg_pool2d(nobj[None, None], 4)[0, 0]; b4 = torch.nn.functional.avg_pool2d(nbg[None, None], 4)[0, 0]
        L = L + 16 * (o4 * (1 - m4) ** 2 + b4 * m4 ** 2).sum() / nf
        if extra is not None:
            L = L + extra(T)
        opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        T = compose(T0, wv, dt, c)
        mk, _, _ = R.render(T, Ks, w, h); m = mk[0] > 0.5
        inter = (nobj * m).sum(); iou  …[+1216 chars]
```

<details><summary>output</summary>

```
A None iou 0.8297 moved [-0.5 -0.  -0.7] 0.3
A 0 iou 0.8228 moved [-1.   0.5  2.6] 179.5
A 1 iou 0.3669 moved [ 5.1 56.6 38.4] 164.2
A 2 iou 0.3685 moved [ 1.1 48.  31.4] 166.8
B None iou 0.9447 moved [-0.   0.1  0.4] 0.4
B 0 iou 0.9212 moved [ 0.9 -0.4  0.9] 178.9
B 1 iou 0.2933 moved [ 36.    6.4 -13.9] 172.5
B 2 iou 0.3141 moved [ 36.6   2.9 -18. ] 179.5
```
</details>

**Bash**
```
cd /work/scratch; cat > textest.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I)
m,v,f,uv,tex=load_mesh(I)
R=Renderer(v,f,uv,tex); ctr=(v.min(0)+v.max(0))/2
def score(T,frs):
    ims=read_video(I+'/video.mp4',frs); pm=load_masks(I,frs,1,'person'); om=load_masks(I,frs)
    out=[]
    mk,d,c=R.render(torch.tensor(T[None],dtype=torch.float32,device='cuda'),K,W,H,True)
    mk=mk[0].cpu().numpy()>0.5; c=c[0].cpu().numpy()
    rg=cv2.cvtColor((c*255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(float)
    for k,i in enumerate(frs):
        g=cv2.cvtColor(ims[i],cv2.COLOR_RGB2GRAY).astype(float)
        sel=cv2.erode((mk&om[k]&~pm[k]).astype(np.uint8),np.ones((5,5)))>0
        a=g[sel]-g[sel].mean(); b=rg[sel]-rg[sel].mean()
        out.append((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
    return np.mean(out)
for name,fr,Tp in [('A',list(range(0,90,10)),'static_0_160.npy'),('B',list(range(700,794,10)),'static_600_794.npy')]:
    T=np.load(Tp)
    print(name,'orig',score(T,fr),'flipx',score(flip_about_axis(T,0,ctr),fr))
EOF
python textest.py
```

<details><summary>output</summary>

```
A orig -0.2679697023461965 flipx -0.26052501342986856
B orig 0.4680556792090972 flipx 0.10189540487441295
```
</details>

**Bash**
```
cd /work/scratch; cat > flipvis.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'; s=0.5
K,Twc,Kp=load_camera(I); Ks=scale_K(K,s)
m,v,f,uv,tex=load_mesh(I)
SD=SilhouetteData(I,s,cache='/work/scratch/masks_half.npz'); R=Renderer(v,f,uv,tex)
ctr=(v.min(0)+v.max(0))/2
fr=list(range(0,262,2)); no,nb=pooled_counts(SD,fr); T=np.load('static_0_160.npy')
Tf,iou=fit_static_pose(R,Ks,SD.h,SD.w,no,nb,flip_about_axis(T,0,ctr),iters=300)
np.save('static_A_flip.npy',Tf)
i=60; im=read_video(I+'/video.mp4',[i])[i]
tiles=[]
for TT in [T,Tf]:
    mk,d,c=R.render(torch.tensor(TT[None],dtype=torch.float32,device='cuda'),K,W,H,True)
    c=(c[0].cpu().numpy()*255).astype(np.uint8); mk=mk[0].cpu().numpy()>0.5
    b=im.copy(); b[mk]=c[mk]; tiles.append(b[380:1000,550:1250])
tiles.insert(0,im[380:1000,550:1250])
cv2.imwrite('flipvis.jpg',np.hstack(tiles)[:,:,::-1])
EOF
python flipvis.py
```

**Read**
```
/work/scratch/flipvis.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_common.py <<'EOF'


def klt_tracks(video_frames, om, pm, frames):
    """Pairwise KLT tracks on visible object texture between consecutive frames.
    video_frames: dict i->RGB; om/pm: dict or array indexable by position in `frames`.
    Returns array (n,5): frame i, x_i, y_i, x_{i+1}, y_{i+1} (full-res pixels)."""
    out = []
    for k in range(len(frames) - 1):
        i = frames[k]
        g0 = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2GRAY); g1 = cv2.cvtColor(video_frames[i + 1], cv2.COLOR_RGB2GRAY)
        allow = (cv2.erode(om[k].astype(np.uint8), np.ones((11, 11))) > 0) & ~(cv2.dilate(pm[k].astype(np.uint8), np.ones((15, 15))) > 0)
        pts = cv2.goodFeaturesToTrack(g0, 400, 0.01, 6, mask=allow.astype(np.uint8), blockSize=7)
        if pts is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(15, 15), maxLevel=3)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(15, 15), maxLevel=3)
        fb = np.linalg.norm(pts - p0b, axis=-1)[:, 0]
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (fb < 0.5)
        q = np.round(p1[:, 0]).astype(int); q[:, 0] = q[:, 0].clip(0, W - 1); q[:, 1] = q[:, 1].clip(0, H - 1)
        ok &= om[k + 1][q[:, 1], q[:, 0]] & ~pm[k + 1][q[:, 1], q[:, 0]]
        for p, qq in zip(pts[ok, 0], p1[ok, 0]):
            out.append((i, p[0], p[1], qq[0], qq[1]))
    return np.array(out, np.float64).reshape(-1, 5)


def klt_lift(R, K, Tcam, tracks, scal …[+2482 chars]
```

<details><summary>output</summary>

```
base         all med 0.80 mean 0.90 260:0.96 300:0.99 350:1.14 400:0.74 450:0.93 500:0.69 550:0.66
cand1.npz    all med 0.14 mean 0.21 260:0.22 300:0.15 350:0.14 400:0.15 450:0.16 500:0.11 550:0.13
perframe     all med 0.35 mean 0.43 260:0.49 300:0.48 350:0.33 400:0.35 450:0.35 500:0.31 550:0.33
```
</details>

**Bash**
```
cd /work/scratch; cat > depthcues.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I); m,v,f,uv,tex=load_mesh(I); R=Renderer(v,f)
pl=json.load(open(I+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
fh=h5py.File('/pipeline/depth/moge2/aligned.h5','r')['aligned/front_stereo_camera_left']
vv=v.astype(float)
for name,Tp,frs in [('A','static_0_160.npy',range(0,170,10)),('B','static_600_794.npy',range(600,794,10))]:
    T=np.load(Tp)
    def minh(eps):
        T2=T.copy(); T2[:3,3]*=1+eps; Tw=Twc@T2
        return ((vv@Tw[:3,:3].T+Tw[:3,3])@n+d).min()
    eps=np.linspace(-0.05,0.05,201); hs=np.array([minh(e) for e in eps])
    e_floor=eps[np.argmin(np.abs(hs-0.014))]
    frs=list(frs); om=load_masks(I,frs); pm=load_masks(I,frs,1,'person')
    mk,dd,_=R.render(torch.tensor(T[None],dtype=torch.float32,device='cuda'),K,W,H); dd=dd[0].cpu().numpy()
    rat=[]
    for k,i in enumerate(frs):
        D=cv2.imdecode(np.frombuffer(fh[i],np.uint8),cv2.IMREAD_UNCHANGED)/1000.
        sel=(cv2.erode(om[k].astype(np.uint8),np.ones((5,5)))>0)&(dd>0)&~pm[k]
        rat.append(np.median(D[sel]/dd[sel]))
    print(name,'minh(0)=%.3f'%minh(0),'dh/deps=%.2f'%((minh(0.01)-minh(-0.01))/0.02),'eps_floor(h0=0.014)=%.4f'%e_floor,'moge ratio med %.4f iqr %.4f'%(np.median(rat),np.subtract(*np.percentile(rat,[75,25]))))
EOF
python depthcues.py
```

<details><summary>output</summary>

```
A minh(0)=0.024 dh/deps=-0.63 eps_floor(h0=0.014)=0.0155 moge ratio med 1.0148 iqr 0.0234
B minh(0)=0.038 dh/deps=-1.03 eps_floor(h0=0.014)=0.0230 moge ratio med 0.9738 iqr 0.0213
```
</details>

**Bash**
```
cd /work/scratch; cat > edgeprof.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I); m,v,f,uv,tex=load_mesh(I); R=Renderer(v,f)
def sdist(mk):
    mk=mk.astype(np.uint8)
    return cv2.distanceTransform(mk,cv2.DIST_L2,5)-cv2.distanceTransform(1-mk,cv2.DIST_L2,5)
for name,Tp,frs in [('A','static_0_160.npy',range(0,90,15)),('B','static_600_794.npy',range(700,794,15)),('human',None,[0])]:
    frs=list(frs); ims=read_video(I+'/video.mp4',frs); om=load_masks(I,frs,1,'person' if name=='human' else 'object'); pm=load_masks(I,frs,1,'person')
    if Tp:
        T=np.load(Tp); rm,_,_=R.render(torch.tensor(T[None],dtype=torch.float32,device='cuda'),K,W,H); rm=rm[0].cpu().numpy()>0.5
    bins=np.arange(-5,5.5,0.5); accm=np.zeros(len(bins)-1); accr=np.zeros(len(bins)-1)
    for k,i in enumerate(frs):
        g=cv2.cvtColor(ims[i],cv2.COLOR_RGB2GRAY).astype(np.float32)
        gm=np.hypot(cv2.Sobel(g,cv2.CV_32F,1,0),cv2.Sobel(g,cv2.CV_32F,0,1))
        excl=cv2.dilate(pm[k].astype(np.uint8),np.ones((9,9)))>0 if name!='human' else np.zeros_like(pm[k])
        for mk,acc in ([(om[k],accm)]+([(rm,accr)] if Tp else [])):
            sd=sdist(mk); sel=(np.abs(sd)<5)&~excl
            h1,_=np.histogram(sd[sel],bins,weights=gm[sel]); h0,_=np.histogram(sd[sel],bins)
            acc+=h1/np.maximum(h0,1)
    c=(bins[:-1]+bins[1:])/2
    def peak(a): 
        j=np.argmax(a); return c[j]
    print(name,'mask: grad profile',np.rou …[+185 chars]
```

<details><summary>output</summary>

```
A mask: grad profile [0.2  0.2  0.19 0.24 0.26 0.42 0.33 0.65 0.77 0.   0.   0.   1.   0.
 0.68 0.65 0.24 0.3  0.15 0.24] peak at 1.25
A render: grad profile [0.6  0.63 0.51 0.61 0.61 0.86 0.75 1.   0.82 0.   0.   0.   0.78 0.
 0.69 0.86 0.53 0.74 0.44 0.52] peak at -1.25
B mask: grad profile [0.23 0.24 0.22 0.25 0.24 0.3  0.28 0.44 0.64 0.   0.   0.   1.   0.
 0.78 0.59 0.43 0.38 0.27 0.27] peak at 1.25
B render: grad profile [0.3  0.29 0.26 0.32 0.31 0.43 0.37 0.48 0.55 0.   0.   0.   1.   0.
 0.93 0.75 0.51 0.45 0.38 0.36] peak at 1.25
human mask: grad profile [0.15 0.16 0.18 0.2  0.22 0.26 0.3  0.51 0.72 0.   0.   0.   1.   0.
 0.69 0.4  0.29 0.28 0.22 0.23] peak at 1.25
bins centres [-4.75 -4.25 -3.75 -3.25 -2.75 -2.25 -1.75 -1.25 -0.75 -0.25  0.25  0.75
  1.25  1.75  2.25  2.75  3.25 …[+19 chars]
```
</details>

**Bash**
```
cd /work/scratch; python klt.py 140 610 >/dev/null; python - <<'EOF'
s=open('track2.py').read()
s=s.replace("p.add_argument('--out')","p.add_argument('--out'); p.add_argument('--mu',type=float,default=0); p.add_argument('--klt',default='klt_140_610.npy')")
s=s.replace("CH=40\n","""CH=40
tr=np.load(args.klt); tr=tr[(tr[:,0]>=a)&(tr[:,0]+1<b)]
ti=torch.tensor(tr[:,0].astype(int)-a,device='cuda'); P0=torch.tensor(tr[:,1:3],dtype=torch.float32,device='cuda'); P1=torch.tensor(tr[:,3:5],dtype=torch.float32,device='cuda')
Kt=torch.tensor(K,dtype=torch.float32,device='cuda')
def proj(T,X):
    Xc=torch.einsum('nij,nj->ni',T[:,:3,:3],X)+T[:,:3,3]; q=Xc@Kt.T; return q[:,:2]/q[:,2:]
Xm=None
""")
s=s.replace("""    T=compose(T0,w,dt,c)
    P=torch.cat""","""    T=compose(T0,w,dt,c)
    if args.mu>0:
        if it%40==0:
            Tc=T.detach().cpu().numpy().astype(float); Xnp,val=klt_lift(R,K,{i:Tc[i-a] for i in fr},tr)
            Xm=torch.tensor(Xnp,dtype=torch.float32,device='cuda'); vm=torch.tensor(val,device='cuda')
        e=torch.nn.functional.huber_loss(proj(T[ti],Xm),P0,reduction='none',delta=1.0).sum(-1)+torch.nn.functional.huber_loss(proj(T[ti+1],Xm),P1,reduction='none',delta=1.0).sum(-1)
        Lk=args.mu*(e*vm).sum(); Lk.backward(retain_graph=True)
    P=torch.cat""")
open('track3.py','w').write(s)
EOF
grep -n "Lk\|klt_lift" track3.py | head; python track3.py --a 150 --b 600 --lam 1e4 --mu 0 --iters 250 --out t3_mu0.npz | tail -1 && python track3.py --a 150 --b 600 --lam  …[+53 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bs8p794ke). Output is being written to: /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/bs8p794ke.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 150; cat /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/bs8p794ke.output
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/bs8p794ke.output
```

<details><summary>output</summary>

```
40:            Tc=T.detach().cpu().numpy().astype(float); Xnp,val=klt_lift(R,K,{i:Tc[i-a] for i in fr},tr)
43:        Lk=args.mu*(e*vm).sum(); Lk.backward(retain_graph=True)
mean iou 0.8785253
mean iou 0.0

[exited with code 0]
```
</details>

**Bash**
```
cd /work/scratch; sed -i 's/        Lk=args.mu\*(e\*vm).sum(); Lk.backward(retain_graph=True)/        Lk=args.mu*(e*vm).sum(); Lk.backward(retain_graph=True)\n        if it%10==0: print("klt",it,Lk.item(),torch.isnan(e).sum().item(),vm.sum().item(),(e*vm).sum().item()\/vm.sum().item(),flush=True)/' track3.py; timeout 100 python track3.py --a 150 --b 600 --lam 1e4 --mu 0.01 --iters 21 --out /dev/null 2>&1 | grep -v "^$" | head
```

<details><summary>output</summary>

```
klt 0 nan 0 10401 nan
0 0.16201246844397652 0.593778076171875 3.8821725845336914
klt 10 nan 639 10401 nan
klt 20 nan 1313 10401 nan
20 0.43851125346289743 nan 10.628527164459229
mean iou 0.7319728
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
s=open('track3.py').read()
s=s.replace("Xm=torch.tensor(Xnp,dtype=torch.float32,device='cuda'); vm=torch.tensor(val,device='cuda')",
"vm=torch.tensor(val,device='cuda'); Xm=torch.tensor(Xnp,dtype=torch.float32,device='cuda'); Xm[~vm]=torch.tensor(v.mean(0),device='cuda')")
s=s.replace("Lk=args.mu*(e*vm).sum()","e=torch.where(vm,e,torch.zeros_like(e)); Lk=args.mu*e.sum()")
s=s.replace('(e*vm).sum().item()','e.sum().item()')
open('track3.py','w').write(s)
EOF
python track3.py --a 150 --b 600 --lam 1e4 --mu 0.01 --iters 250 --out t3_mu01.npz | grep -v klt | tail -3
```

<details><summary>output</summary>

```
200 0.16254155145751106 0.0021254824267493352 62.4015793800354
249 0.1627909012635549 0.0020193832450442844 83.13768768310547
mean iou 0.8783965
```
</details>

**Bash**
```
cd /work/scratch; cat > cmpstatic.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
m,v,f,uv,tex=load_mesh('/work/inputs'); vs=v[::50].astype(float)
def add(T1,T2): return np.linalg.norm((vs@T1[:3,:3].T+T1[:3,3])-(vs@T2[:3,:3].T+T2[:3,3]),axis=1).mean()
A=np.load('static_0_160.npy')
zs=[np.load(p) for p in sys.argv[1:]]
for i in range(150,300,10):
    print(i,' '.join('%.3f'%add(z['T'][i-z['fr'][0]],A) for z in zs))
EOF
python cmpstatic.py t3_mu0.npz t3_mu01.npz; grep klt /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/bs8p794ke.output | tail -2
```

<details><summary>output</summary>

```
150 0.001 0.000
160 0.006 0.006
170 0.012 0.012
180 0.028 0.028
190 0.029 0.029
200 0.032 0.032
210 0.032 0.032
220 0.032 0.033
230 0.032 0.031
240 0.032 0.033
250 0.030 0.031
260 0.034 0.033
270 0.040 0.039
280 0.047 0.048
290 0.090 0.087
40:            Tc=T.detach().cpu().numpy().astype(float); Xnp,val=klt_lift(R,K,{i:Tc[i-a] for i in fr},tr)
```
</details>

**Bash**
```
cd /work/scratch; python track3.py --a 150 --b 600 --lam 1e4 --mu 1 --iters 250 --out t3_mu1.npz | grep -E "klt (0|240)|mean"; python cmpstatic.py t3_mu0.npz t3_mu1.npz
```

<details><summary>output</summary>

```
klt 0 1405.044677734375 0 10401 0.13508746060324728
klt 240 413.3562316894531 0 10396 0.03976108423330638
mean iou 0.87794477
150 0.001 0.002
160 0.006 0.007
170 0.012 0.012
180 0.028 0.027
190 0.029 0.032
200 0.032 0.032
210 0.032 0.032
220 0.032 0.032
230 0.032 0.032
240 0.032 0.032
250 0.030 0.032
260 0.034 0.032
270 0.040 0.040
280 0.047 0.048
290 0.090 0.085
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
K,Twc,Kp=load_camera('/work/inputs'); m,v,f,uv,tex=load_mesh('/work/inputs')
A=np.load('static_0_160.npy'); z=np.load('t3_mu0.npz')
def pr(T):
  X=v@T[:3,:3].T+T[:3,3]; q=X@K.T; return q[:,:2]/q[:,2:]
qa=pr(A)
sel=(qa[:,0]>850)&(qa[:,0]<1030)&(qa[:,1]>600)&(qa[:,1]<890)
for i in [180,200,240,260]:
  qb=pr(z['T'][i-150]); print(i, (qb[sel]-qa[sel]).mean(0), sel.sum())
"
```

<details><summary>output</summary>

```
180 [-1.99731332  0.49318745] 134770
200 [-3.61490913  0.71616283] 134770
240 [-4.72600044  1.07045967] 134770
260 [-4.48180576  0.7400352 ] 134770
```
</details>

**Bash**
```
cd /work/scratch; cat > occtest.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'; s=0.5
K,Twc,Kp=load_camera(I); Ks=scale_K(K,s)
m,v,f,uv,tex=load_mesh(I); vs=v[::50].astype(float)
def add(T1,T2): return np.linalg.norm((vs@T1[:3,:3].T+T1[:3,3])-(vs@T2[:3,:3].T+T2[:3,3]),axis=1).mean()
z=np.load('masks_half.npz'); om,pm=z['om'],z['pm']
R=Renderer(v,f); A=np.load('static_0_160.npy')
fr=list(range(180,262,3))
for dil in [0,3,6,10]:
    o=om[fr]; p=pm[fr]
    if dil: p=np.stack([cv2.dilate(x.astype(np.uint8),np.ones((2*dil+1,2*dil+1)))>0 for x in p])
    valid=~p|o
    # object pixels adjacent to the person are unreliable too (hands); keep them
    no=torch.tensor((o&valid).sum(0),dtype=torch.float32,device='cuda'); nb=torch.tensor((~o&valid).sum(0),dtype=torch.float32,device='cuda')
    T,iou=fit_static_pose(R,Ks,int(H*s),int(W*s),no,nb,A,iters=250)
    print(dil,'iou',round(iou,4),'ADD to A %.4f'%add(T,A))
EOF
python occtest.py
```

<details><summary>output</summary>

```
0 iou 0.8715 ADD to A 0.0315
3 iou 0.8838 ADD to A 0.0320
6 iou 0.8947 ADD to A 0.0329
10 iou 0.8971 ADD to A 0.0350
```
</details>

**Bash**
```
cd /work/scratch; cat > flowcheck.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I); m,v,f,uv,tex=load_mesh(I); R=Renderer(v,f)
A=np.load('static_0_160.npy'); z=np.load('t3_mu0.npz'); a0=int(z['fr'][0])
i0=int(sys.argv[1]); fr=[int(x) for x in sys.argv[2].split(',')]
ims=read_video(I+'/video.mp4',[i0]+fr); om=load_masks(I,[i0]+fr); pm=load_masks(I,[i0]+fr,1,'person')
g0=cv2.cvtColor(ims[i0],cv2.COLOR_RGB2GRAY)
mk,d,_=R.render(torch.tensor(A[None],dtype=torch.float32,device='cuda'),K,W,H); d=d[0].cpu().numpy()
for k,i in enumerate(fr):
    g1=cv2.cvtColor(ims[i],cv2.COLOR_RGB2GRAY)
    fl=cv2.calcOpticalFlowFarneback(g0,g1,None,0.5,4,21,5,7,1.5,0)
    sel=(cv2.erode(om[0].astype(np.uint8),np.ones((7,7)))>0)&~(cv2.dilate((pm[0]|pm[k+1]).astype(np.uint8),np.ones((21,21)))>0)&(d>0)
    ys,xs=np.where(sel)
    zz=d[ys,xs]; X=np.stack([(xs-K[0,2])/K[0,0]*zz,(ys-K[1,2])/K[1,1]*zz,zz],1)
    Xm=(X-A[:3,3])@A[:3,:3]
    T=z['T'][i-a0]; Xc=Xm@T[:3,:3].T+T[:3,3]; q=Xc@K.T; q=q[:,:2]/q[:,2:]
    pred=q-np.stack([xs,ys],1); meas=fl[ys,xs]
    # split into regions by x
    for nm,s in [('left',xs<np.percentile(xs,33)),('mid',(xs>=np.percentile(xs,33))&(xs<np.percentile(xs,66))),('right',xs>=np.percentile(xs,66))]:
        print(i,nm,'meas',np.round(np.median(meas[s],0),2),'pred(silh fit)',np.round(np.median(pred[s],0),2), s.sum())
EOF
python flowcheck.py 160 200,240
```

<details><summary>output</summary>

```
200 left meas [-8.57  0.51] pred(silh fit) [-7.85  2.81] 4629
200 mid meas [-0.59 -0.  ] pred(silh fit) [-4.3   1.02] 4448
200 right meas [-1.2 -0. ] pred(silh fit) [-2.42  0.89] 5007
240 left meas [-8.96  1.36] pred(silh fit) [-7.55  3.61] 3512
240 mid meas [-1.35 -0.17] pred(silh fit) [-2.94  1.31] 3416
240 right meas [-1.24 -0.05] pred(silh fit) [-3.82  0.91] 3805
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
fr=[600,620,640,700,760,790]
ims=read_video(I+'/video.mp4',fr); om=load_masks(I,fr); pm=load_masks(I,fr,1,'person')
g0=cv2.cvtColor(ims[790],cv2.COLOR_RGB2GRAY)
for k,i in enumerate(fr[:-1]):
    g1=cv2.cvtColor(ims[i],cv2.COLOR_RGB2GRAY)
    fl=cv2.calcOpticalFlowFarneback(g0,g1,None,0.5,4,21,5,7,1.5,0)
    sel=(cv2.erode(om[-1].astype(np.uint8),np.ones((7,7)))>0)&~(cv2.dilate((pm[-1]|pm[k]).astype(np.uint8),np.ones((21,21)))>0)
    ys,xs=np.where(sel)
    for nm,s in [('left',xs<np.percentile(xs,33)),('mid',(xs>=np.percentile(xs,33))&(xs<np.percentile(xs,66))),('right',xs>=np.percentile(xs,66))]:
        print(i,nm,np.round(np.median(fl[ys[s],xs[s]],0),2),s.sum())
EOF
```

<details><summary>output</summary>

```
600 left [-61.97 -21.76] 4628
600 mid [-46.68 -19.61] 4608
600 right [-44.35 -14.41] 4956
620 left [-76.76 -26.31] 4628
620 mid [-70.61 -27.7 ] 4608
620 right [-63.54 -18.52] 4956
640 left [-76.92 -23.19] 4628
640 mid [-69.34 -22.11] 4608
640 right [-61.   -17.56] 4956
700 left [-74.77 -22.98] 4628
700 mid [-60.24 -24.13] 4608
700 right [-57.68 -16.81] 4956
760 left [-1.12  0.02] 4628
760 mid [-0.01  0.01] 4608
760 right [-0.02 -0.  ] 4956
```
</details>

**Bash**
```
cd /work/scratch; cat > eccpatch.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'
K,Twc,Kp=load_camera(I); m,v,f,uv,tex=load_mesh(I); R=Renderer(v,f)
A=np.load(sys.argv[1]); z=np.load(sys.argv[2]); a0=int(z['fr'][0])
i0=int(sys.argv[3]); fr=[int(x) for x in sys.argv[4].split(',')]
ims=read_video(I+'/video.mp4',[i0]+fr); pm=load_masks(I,[i0]+fr,1,'person'); om=load_masks(I,[i0])[0]
mk,d,_=R.render(torch.tensor(A[None],dtype=torch.float32,device='cuda'),K,W,H); d=d[0].cpu().numpy()
g0=cv2.cvtColor(ims[i0],cv2.COLOR_RGB2GRAY).astype(np.float32)
gm=np.hypot(cv2.Sobel(g0,cv2.CV_32F,1,0),cv2.Sobel(g0,cv2.CV_32F,0,1))
# candidate patch centres: object boundary pixels with strong gradient, far from person in all frames
P=21
anyp=np.any(pm,0); farp=~(cv2.dilate(anyp.astype(np.uint8),np.ones((61,61)))>0)
cand=np.argwhere(om&farp&(gm>50)&(d>0))
rng=np.random.default_rng(0); cand=cand[rng.choice(len(cand),min(len(cand),3000),replace=False)]
sel=[]
for y,x in cand:
    if all((y-yy)**2+(x-xx)**2>60**2 for yy,xx in sel) and P<y<H-P and P<x<W-P: sel.append((y,x))
print('patches',len(sel))
for i in fr:
    g1=cv2.cvtColor(ims[i],cv2.COLOR_RGB2GRAY).astype(np.float32)
    T=z['T'][i-a0]
    rows=[]
    for y,x in sel:
        p0=g0[y-P:y+P+1,x-P:x+P+1]
        best=None
        res=cv2.matchTemplate(g1[y-P-12:y+P+13,x-P-12:x+P+13],p0,cv2.TM_CCOEFF_NORMED)
        _,mv,_,ml=cv2.minMaxLoc(res)
        meas=np.array([ml[0]-12,ml[1]-12])
         …[+607 chars]
```

<details><summary>output</summary>

```
patches 8
180 n 8 median |meas| [0. 0.] median |pred| [1.05 1.39] median |meas-pred| 1.81 |meas-0| 0.5
200 n 8 median |meas| [1. 0.] median |pred| [1.72 1.73] median |meas-pred| 1.77 |meas-0| 1.0
240 n 8 median |meas| [1. 0.] median |pred| [1.99 2.13] median |meas-pred| 2.59 |meas-0| 1.0
260 n 8 median |meas| [1. 0.] median |pred| [1.64 2.14] median |meas-pred| 2.25 |meas-0| 1.21
```
</details>

**Bash**
```
cd /work/scratch; python track3.py --a 172 --b 600 --lam 1e4 --mu 20 --iters 250 --out t3_mu20.npz | grep -E "klt (0|240)|mean"; python cmpstatic.py t3_mu1.npz t3_mu20.npz
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b7x0q0rvl). Output is being written to: /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/b7x0q0rvl.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 45; cat /tmp/claude-1003/-work/f2c3f6bd-9189-4a46-b9be-8aa555f00d38/tasks/b7x0q0rvl.output
```

<details><summary>output</summary>

```
klt 0 27404.501953125 0 10108 0.13555847820105363
klt 240 8231.7734375 0 10104 0.04073522209837997
mean iou 0.8783951
150 0.002 1.054
160 0.007 1.058
170 0.012 1.065
180 0.027 0.029
190 0.032 0.032
200 0.032 0.032
210 0.032 0.032
220 0.032 0.033
230 0.032 0.032
240 0.032 0.032
250 0.032 0.032
260 0.032 0.032
270 0.040 0.041
280 0.048 0.047
290 0.085 0.085

[exited with code 0]
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
cd /work/outputs; python - <<'EOF'
s=open('hoi_common.py').read()
s=s.replace("""    res = np.full(len(tracks), np.nan)
    for n in np.where(val)[0]:
        pass
""","")
open('hoi_common.py','w').write(s)
EOF
cd /work; (time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz) > /work/outputs/solve_log.txt 2>&1 &
echo started
```

<details><summary>output</summary>

```
started
```
</details>

**Agent:** Meanwhile, I'll write the validation script.

**Write**
```
/work/outputs/validate.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/validate.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
sleep 30; cat /work/outputs/solve_log.txt
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
sleep 90; cat /work/outputs/solve_log.txt
```

<details><summary>output</summary>

```
[   60.1s] masks loaded
[   64.7s] static A: hypothesis flip=None pooled IoU 0.8514
[   68.1s] static A: hypothesis flip=0 pooled IoU 0.8545
[   71.5s] static A: hypothesis flip=1 pooled IoU 0.3837
[   74.7s] static A: hypothesis flip=2 pooled IoU 0.3731
[   74.7s] static A: chose flip=0, margin to 2nd 0.0030
[   76.9s] static A: depth cues eps silhouette 0, floor -0.0040 (h(0)=0.012, h0=0.014), moge +0.0023 -> fused -0.0000 (-0.0 cm)
[   79.1s] static B: hypothesis flip=None pooled IoU 0.9446
[   81.9s] static B: hypothesis flip=0 pooled IoU 0.9210
[   85.5s] static B: hypothesis flip=1 pooled IoU 0.3028
[   88.8s] static B: hypothesis flip=2 pooled IoU 0.2962
[   88.8s] static B: chose flip=None, margin to 2nd 0.0236
[   90.5s] static B: depth cues eps silhouette 0, floor +0.0210 (h(0)=0 …[+108 chars]
```
</details>

**Bash**
```
pkill -f solve.py; sleep 1; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

<details><summary>output</summary>

```
Exit code 144
```
</details>

**Agent:** Now restructure so the static-A orientation is decided by continuity with the moving-phase tracking when the silhouette margin is ambiguous.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
old_start=s.index("    def fit_phase(name, fit_frames, all_frames):")
old_end=s.index("    # ------------------------------------------------------------ moving phase")
new='''    def fit_hypotheses(name, fit_frames):
        """Pooled silhouette fit from the baseline pose and its 180-degree flips about the 3 mesh axes."""
        no, nb = pooled_counts(SD, fit_frames)
        Tinit = Tb[fit_frames[len(fit_frames) // 2]]
        hyps = []
        for ax in [None, 0, 1, 2]:
            Ti = Tinit if ax is None else flip_about_axis(Tinit, ax, ctr)
            T, iou = fit_static_pose(R, Ks, h, w, no, nb, Ti, iters=300)
            hyps.append((iou, ax, T))
            log(f"static {name}: hypothesis flip={ax} pooled IoU {iou:.4f}")
        return sorted(hyps, key=lambda x: -x[0])

    def choose(name, hyps, T_track):
        """Silhouette decides if the IoU margin is clear (>= 0.01); otherwise (near-symmetric silhouette) the
        hypothesis that is continuous with the tracked moving phase at the phase boundary wins."""
        rot = [rot_angle_deg(hh[2][:3, :3], T_track[:3, :3]) for hh in hyps]
        log(f"static {name}: rotation to tracked boundary pose: " + ", ".join(f"flip={hh[1]}: {r:.1f}deg" for hh, r in zip(hyps, rot)))
        margin = hyps[0][0] - hyps[1][0]
        if margin >= 0.01:
            k = 0
        else:
            cands = [j for j in range(len(hyps)) if hyps[0][0] - hyps[j][0] < 0.01]
       …[+2625 chars]
```

<details><summary>output</summary>

```
138:        tA = torch.tensor(TA[None], dtype=torch.float32, device="cuda"); tB = torch.tensor(TB[None], dtype=torch.float32, device="cuda")
183:    TA = TB = None
186:    TA = depth_fuse("A", choose("A", hypA, T1[0]), list(range(0, A_END)))
187:    TB = depth_fuse("B", choose("B", hypB, T1[-1]), list(range(B_START, F)))
192:    Tcam[:a] = TA; Tcam[a:b] = T2; Tcam[b:] = TB
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace('''        tA = torch.tensor(TA[None], dtype=torch.float32, device="cuda"); tB = torch.tensor(TB[None], dtype=torch.float32, device="cuda")''','''        if lam > 0:
            tA = torch.tensor(TA[None], dtype=torch.float32, device="cuda"); tB = torch.tensor(TB[None], dtype=torch.float32, device="cuda")''')
s=s.replace("    TA = TB = None\n","")
open('solve.py','w').write(s)
EOF
python -c "import ast;ast.parse(open('/work/outputs/solve.py').read())"; sed -n 100,200p solve.py
```

<details><summary>output</summary>

```
        e_floor = float(epss[np.argmin(np.abs(hs - h0))])
        dh = abs(minh(0.01) - minh(-0.01)) / 0.02
        mk, dd, _ = R.render(torch.tensor(T[None], dtype=torch.float32, device="cuda"), K, W, H)
        dd = dd[0].cpu().numpy()
        rat = []
        for i in all_frames[::10]:
            D = cv2.imdecode(np.frombuffer(moge[i], np.uint8), cv2.IMREAD_UNCHANGED) / 1000.0
            o = cv2.resize(SD.obj[i].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            vis = cv2.resize(SD.valid[i].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            sel = (cv2.erode(o, np.ones((5, 5))) > 0) & (vis > 0) & (dd > 0)
            if sel.sum() > 500:
                rat.append(np.median(D[sel] / dd[sel]))
        e_moge = float(np.m …[+4936 chars]
```
</details>

**Bash**
```
cd /work; (time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz) > /work/outputs/solve_log.txt 2>&1; tail -30 /work/outputs/solve_log.txt
```

<details><summary>output</summary>

```
[   58.1s] masks loaded
[   60.1s] static A: hypothesis flip=None pooled IoU 0.8536
[   61.1s] static A: hypothesis flip=0 pooled IoU 0.8555
[   62.0s] static A: hypothesis flip=1 pooled IoU 0.3833
[   63.1s] static A: hypothesis flip=2 pooled IoU 0.3742
[   64.0s] static B: hypothesis flip=None pooled IoU 0.9445
[   65.1s] static B: hypothesis flip=0 pooled IoU 0.9220
[   66.1s] static B: hypothesis flip=1 pooled IoU 0.3033
[   67.1s] static B: hypothesis flip=2 pooled IoU 0.2978
[  106.2s] KLT tracks: 8934 (4500 used for fitting)
[  151.3s] moving phase: per-frame silhouette fit done
[  151.3s] static A: rotation to tracked boundary pose: flip=0: 178.0deg, flip=None: 2.5deg, flip=1: 164.6deg, flip=2: 165.2deg
[  151.3s] static A: chose flip=None (IoU margin best-2nd 0.0019)
[  153.6s] st …[+616 chars]
```
</details>

**Bash**
```
free -g | head -2; cd /work/outputs && timeout 1500 python validate.py object_poses_world.npz 2>&1 | tail -15
```

<details><summary>output</summary>

```
               total        used        free      shared  buff/cache   available
Mem:             125          24          28           0          72          99
| phase | frames | traj | mask IoU | held-out KLT px (median / mean<=5) | patch i->i+5 px (median / mean<=10) | accel mm/frame^2 | median lowest-point height above plane (m) |
|---|---|---|---|---|---|---|---|
| static A | 0-264 | baseline | 0.760 | 0.66 / 0.74 (n=1658) | 0.92 / 1.07 (n=536) | 8.55 | 0.023 |
| static A | 0-264 | result | 0.820 | 0.02 / 0.07 (n=1658) | 0.02 / 0.11 (n=536) | 0.05 | 0.022 |
| moving | 265-599 | baseline | 0.774 | 0.83 / 0.94 (n=4472) | 1.73 / 2.21 (n=710) | 18.24 | 0.002 |
| moving | 265-599 | result | 0.878 | 0.34 / 0.42 (n=4472) | 0.81 / 1.56 (n=710) | 1.82 | 0.031 |
| static B | 600-793 | baseline …[+239 chars]
```
</details>

**Bash**
```
timeout 1500 python validate.py /work/scratch/cand1.npz 2>&1 | grep "| result\|all-frames"
```

<details><summary>output</summary>

```
| static A | 0-264 | result | 0.825 | 0.03 / 0.09 (n=1658) | 0.02 / 0.14 (n=536) | 0.03 | 0.024 |
| moving | 265-599 | result | 0.878 | 0.14 / 0.22 (n=4472) | 0.75 / 1.51 (n=710) | 0.30 | 0.031 |
| static B | 600-793 | result | 0.944 | 0.01 / 0.02 (n=3899) | 0.02 / 0.02 (n=422) | 0.00 | 0.038 |
baseline all-frames IoU 0.78994733
result all-frames IoU 0.87647146
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
a=s.index("    # KLT tracks (full-resolution masks for the moving phase)")
b=s.index("    T1 = optimise(")
s=s[:a]+s[b:]
s=s.replace("T2 = optimise(T1, iters=250, lam=1e4, mu=1.0, tracks=tracks_fit)","T2 = optimise(T1, iters=250, lam=1e4, mu=0)")
s=s.replace("""     static poses as fixed boundary conditions.""","""     static poses as fixed boundary conditions.  (A KLT reprojection term is implemented in optimise(mu>0) but
     disabled: in tests it did not remove the occlusion bias and, fitted on alternate frame pairs, it made the
     trajectory less smooth and increased the held-out KLT residual; KLT tracks are used for validation only.)""")
s=s.replace("""     all frames with silhouette + constant-velocity (acceleration) prior on the mesh bounding-box corners +
     KLT feature-track reprojection (even frames only; odd frames are held out for validation), with the""","""     all frames with silhouette + constant-velocity (acceleration) prior on the mesh bounding-box corners, with the""")
open('solve.py','w').write(s)
EOF
sed -n 1,25p solve.py; grep -n "tracks" solve.py
```

<details><summary>output</summary>

```
"""Recover the 6-DoF trajectory of the white desk (episode 10) in the released world frame.

Method (see REPORT.md):
  1. Phases (constants verified by eye + automatic checks, see below): static A (desk upright, untouched / only
     held), moving (desk turned, tilted and laid on its side), static B (desk lying on the floor).
  2. Static phases: ONE pose per phase, fitted by render-and-compare (nvdiffrast) against the SAM2 object mask
     pooled over all frames of the phase (pixels covered by the person are ignored).  Hypotheses: the baseline
     pose and its 180-degree flips about the three mesh axes; the best pooled IoU wins.  Depth along the viewing
     ray is then fused (inverse variance) with two weak cues: floor contact and MoGe-2 depth.
  3. Moving phase: per-frame silhouette fit …[+1773 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace("pose and its 180-degree flips about the three mesh axes; the best pooled IoU wins.","pose and its 180-degree flips about the three mesh axes; the best pooled IoU wins unless its margin is\n     < 0.01 (near-symmetric silhouette), then the hypothesis continuous with the tracked moving phase wins.")
open('solve.py','w').write(s)
EOF
cd /work; (time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz) > /work/outputs/solve_log.txt 2>&1; tail -12 /work/outputs/solve_log.txt; python -c "
import numpy as np; z=np.load('/work/outputs/object_poses_world.npz'); T=z['T_world_mesh']; print(T.shape,T.dtype,np.isfinite(T).all(), np.abs(np.linalg.det(T[:,:3,:3])-1).max())"
```

<details><summary>output</summary>

```
[  154.3s] static A: rotation to tracked boundary pose: flip=0: 178.0deg, flip=None: 2.5deg, flip=1: 164.5deg, flip=2: 165.5deg
[  154.3s] static A: chose flip=None (IoU margin best-2nd 0.0017)
[  157.4s] static A: depth cues eps silhouette 0, floor +0.0150 (h(0)=0.024, h0=0.014), moge +0.0121 -> fused +0.0027 (+0.7 cm)
[  157.4s] static B: rotation to tracked boundary pose: flip=None: 0.4deg, flip=0: 179.6deg, flip=1: 177.3deg, flip=2: 178.4deg
[  157.4s] static B: chose flip=None (IoU margin best-2nd 0.0221)
[  160.3s] static B: depth cues eps silhouette 0, floor +0.0210 (h(0)=0.036, h0=0.014), moge -0.0267 -> fused +0.0009 (+0.3 cm)
[  285.8s] moving phase: joint fit done; peak GPU mem 3.52 GB
[  285.8s] wrote /work/outputs/object_poses_world.npz  shape (794, 4, 4)

real	4m48.505s
user	 …[+70 chars]
```
</details>

**Bash**
```
cd /work/outputs && timeout 1500 python validate.py object_poses_world.npz 2>&1 | tail -10 | tee validation_table.md; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_common import *
I='/work/inputs'; m,v,f,uv,tex=load_mesh(I); vs=v[::50].astype(float)
b=np.load(I+'/reconstruction.npz')['T_world_mesh'].astype(float); r=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
d=np.array([np.linalg.norm((vs@b[i,:3,:3].T+b[i,:3,3])-(vs@r[i,:3,:3].T+r[i,:3,3]),axis=1).mean() for i in range(794)])
ang=np.array([rot_angle_deg(b[i,:3,:3],r[i,:3,:3]) for i in range(794)])
for a,e in [(0,265),(265,600),(600,794)]: print(a,e,'ADD(base,result) mean %.3f max %.3f; rot diff mean %.1f max %.1f'%(d[a:e].mean(),d[a:e].max(),ang[a:e].mean(),ang[a:e].max()))
"
```

<details><summary>output</summary>

```
| phase | frames | traj | mask IoU | held-out KLT px (median / mean<=5) | patch i->i+5 px (median / mean<=10) | accel mm/frame^2 | median lowest-point height above plane (m) |
|---|---|---|---|---|---|---|---|
| static A | 0-264 | baseline | 0.760 | 0.66 / 0.74 (n=1658) | 0.92 / 1.07 (n=536) | 8.55 | 0.023 |
| static A | 0-264 | result | 0.821 | 0.03 / 0.07 (n=1658) | 0.02 / 0.09 (n=536) | 0.02 | 0.022 |
| moving | 265-599 | baseline | 0.774 | 0.83 / 0.94 (n=4472) | 1.73 / 2.21 (n=710) | 18.24 | 0.002 |
| moving | 265-599 | result | 0.877 | 0.14 / 0.22 (n=4472) | 0.80 / 1.54 (n=710) | 0.31 | 0.031 |
| static B | 600-793 | baseline | 0.859 | 0.73 / 0.80 (n=3899) | 0.98 / 1.23 (n=422) | 14.84 | -0.004 |
| static B | 600-793 | result | 0.941 | 0.01 / 0.02 (n=3899) | 0.02 / 0.02 (n=422) | 0.00 …[+294 chars]
```
</details>

**Bash**
```
python -c "
import torch,nvdiffrast,cv2,trimesh,h5py,numpy,scipy; print(torch.__version__,nvdiffrast.__version__,cv2.__version__,trimesh.__version__,h5py.__version__,numpy.__version__,scipy.__version__)"; python -c "
import numpy as np, json
I='/work/inputs'; c=json.load(open(I+'/camera.json')); Tcw=np.linalg.inv(np.array(c['T_world_cam']))
b=Tcw@np.load(I+'/reconstruction.npz')['T_world_mesh'].astype(float); r=Tcw@np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for a,e in [(0,265),(265,400),(400,500),(500,600),(600,794)]:
  db=np.linalg.norm(b[a:e,:3,3],axis=1); dr=np.linalg.norm(r[a:e,:3,3],axis=1); print(a,e,'range base %.3f result %.3f diff %.3f (%.1f%%)'%(db.mean(),dr.mean(),(dr-db).mean(),100*((dr-db)/db).mean()))
"; ls /work/tools /work/scratch/pyuser 2>&1 | head
```

<details><summary>output</summary>

```
2.5.1+cu124 0.4.0 4.11.0 5.1.1 3.16.0 1.26.3 1.16.3
0 265 range base 2.794 result 2.781 diff -0.013 (-0.5%)
265 400 range base 2.935 result 2.872 diff -0.063 (-2.1%)
400 500 range base 3.306 result 3.173 diff -0.133 (-4.0%)
500 600 range base 3.615 result 3.494 diff -0.121 (-3.3%)
600 794 range base 3.623 result 3.516 diff -0.106 (-2.9%)
ls: cannot access '/work/scratch/pyuser': No such file or directory
/work/tools:
```
</details>

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

**Bash**
```
ls -la /work/outputs; rm -rf /work/outputs/__pycache__
```

<details><summary>output</summary>

```
total 168
drwxrwxr-x 3 hanzhou hanzhou   4096 Oct 21 02:03 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 21 01:04 ..
-rw-rw-r-- 1 hanzhou hanzhou   9321 Oct 21 02:03 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   1438 Oct 21 02:03 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 21 02:00 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  11649 Oct 21 01:44 hoi_common.py
-rw-rw-r-- 1 hanzhou hanzhou 101910 Oct 21 02:00 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou  11000 Oct 21 01:55 solve.py
-rw-rw-r-- 1 hanzhou hanzhou   1365 Oct 21 02:00 solve_log.txt
-rw-rw-r-- 1 hanzhou hanzhou   5052 Oct 21 01:44 validate.py
-rw-rw-r-- 1 hanzhou hanzhou    878 Oct 21 02:02 validation_table.md
```
</details>

**Agent:** I've written the full trajectory to `/work/outputs/object_poses_world.npz` (794 poses, float64), along with `REPORT.md`, `TOOLS.md` and `solve.py`. The final `solve.py` run completed offline in 4 min 49 s using 3.5 GB of GPU memory. Without ground truth, my own checks favour the result over the baseline in every phase; the frames where the person covers the desk are the least certain.

**Phases** (boundaries are constants in `solve.py`, checked by eye and by automatic tests):
- **Frames 0–264, desk upright:** one constant pose. The person grabs the desk at about frame 175. Image patches the person doesn't cover move by at most 1 px, so the 2–3 cm the per-frame fits drift here comes from the person blocking part of the desk, not from real motion.
- **Frames 265–599, moving:** the desk is turned, tipped and laid on its front edge.
- **Frames 600–793, lying on the floor:** one constant pose.

**Method:**
- **Static phases:** the mesh is rendered and matched to the object mask over all frames of the phase, ignoring person pixels. I tried the baseline pose and its 180° flips about each mesh axis.
- **Symmetry in the upright phase:** the outline can't tell the right orientation from a 180° turn about the vertical axis (scores differ by 0.002). `solve.py` picks the one that connects smoothly to the tracked motion, which leads to the lying pose. That pose is unambiguous: outline score 0.944 vs 0.922 for the flip, texture match 0.47 vs 0.10.
- **Moving phase:** each frame is fitted to its outline starting from the baseline, then all frames are optimised together with a smoothness penalty, using the two static poses as fixed endpoints.

**Validation, baseline vs result:**

| Phase | Mask overlap (IoU) | Held-out feature-track error, px (median) | Patch-match error, px (median) |
|---|---|---|---|
| Upright (0–264) | 0.760 → 0.821 | 0.66 → 0.03 | 0.92 → 0.02 |
| Moving (265–599) | 0.774 → 0.877 | 0.83 → 0.14 | 1.73 → 0.80 |
| Lying (600–793) | 0.859 → 0.941 | 0.73 → 0.01 | 0.98 → 0.02 |

Over all frames the overlap goes from 0.790 to 0.874. In the static phases the two error columns mainly show that the baseline jitters while the desk is still; the moving phase is the informative row. The feature tracks used for validation are not used by `solve.py`.

**Depth is the biggest change.** The result puts the desk 2–4% closer to the camera than the baseline in the moving and lying phases, up to about 13 cm. The camera intrinsics reproduce the true frame-0 human's outline within a few pixels, and the masks sit on the image edges, so I trust the outline-based depth. The MoGe-2 depth agrees with it. The floor contact only partly does: lying on the floor, the desk sits level but 3.5 cm above the plane. The floor would put it about 2% farther and MoGe about 2.7% closer, so I kept the outline depth, with only a 0.3–0.7 cm fused correction in the static phases.

**Confidence:**
- **Upright phase:** high.
- **Lying phase:** high, but only medium-high on depth.
- **Moving phase:** medium. Frames 280–440, where the bending person covers much of the desk, probably still have errors of a few cm, like the bias I saw on the known-static frames.

**Other notes:**
- I tried adding a feature-tracking term to the moving-phase optimisation and dropped it. It didn't remove the occlusion bias, and it made the trajectory jumpier.
- GPU rendering isn't bit-exact, so reruns of `solve.py` differ by well under a millimetre; seeds are fixed and the decisions don't change.
- My GPU use across all experiments is an estimate from run times, roughly 35–40 minutes, against the 45-minute limit.
- No external tools or weights were downloaded.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoi_common.py`
- `validate.py`
- `validation_table.md`
- `solve_log.txt`

## Final

I've written the full trajectory to `/work/outputs/object_poses_world.npz` (794 poses, float64), along with `REPORT.md`, `TOOLS.md` and `solve.py`. The final `solve.py` run completed offline in 4 min 49 s using 3.5 GB of GPU memory. Without ground truth, my own checks favour the result over the baseline in every phase; the frames where the person covers the desk are the least certain.

**Phases** (boundaries are constants in `solve.py`, checked by eye and by automatic tests):
- **Frames 0–264, desk upright:** one constant pose. The person grabs the desk at about frame 175. Image patches the person doesn't cover move by at most 1 px, so the 2–3 cm the per-frame fits drift here comes from the person blocking part of the desk, not from real motion.
- **Frames 265–599, moving:** the desk is turned, tipped and laid on its front edge.
- **Frames 600–793, lying on the floor:** one constant pose.

**Method:**
- **Static phases:** the mesh is rendered and matched to the object mask over all frames of the phase, ignoring person pixels. I tried the baseline pose and its 180° flips about each mesh axis.
- **Symmetry in the upright phase:** the outline can't tell the right orientation from a 180° turn about the vertical axis (scores differ by 0.002). `solve.py` picks the one that connects smoothly to the tracked motion, which leads to the lying pose. That pose is unambiguous: outline score 0.944 vs 0.922 for the flip, texture match 0.47 vs 0.10.
- **Moving phase:** each frame is fitted to its outline starting from the baseline, then all frames are optimised together with a smoothness penalty, using the two static poses as fixed endpoints.

**Validation, baseline vs result:**

| Phase | Mask overlap (IoU) | Held-out feature-track error, px (median) | Patch-match error, px (median) |
|---|---|---|---|
| Upright (0–264) | 0.760 → 0.821 | 0.66 → 0.03 | 0.92 → 0.02 |
| Moving (265–599) | 0.774 → 0.877 | 0.83 → 0.14 | 1.73 → 0.80 |
| Lying (600–793) | 0.859 → 0.941 | 0.73 → 0.01 | 0.98 → 0.02 |

Over all frames the overlap goes from 0.790 to 0.874. In the static phases the two error columns mainly show that the baseline jitters while the desk is still; the moving phase is the informative row. The feature tracks used for validation are not used by `solve.py`.

**Depth is the biggest change.** The result puts the desk 2–4% closer to the camera than the baseline in the moving and lying phases, up to about 13 cm. The camera intrinsics reproduce the true frame-0 human's outline within a few pixels, and the masks sit on the image edges, so I trust the outline-based depth. The MoGe-2 depth agrees with it. The floor contact only partly does: lying on the floor, the desk sits level but 3.5 cm above the plane. The floor would put it about 2% farther and MoGe about 2.7% closer, so I kept the outline depth, with only a 0.3–0.7 cm fused correction in the static phases.

**Confidence:**
- **Upright phase:** high.
- **Lying phase:** high, but only medium-high on depth.
- **Moving phase:** medium. Frames 280–440, where the bending person covers much of the desk, probably still have errors of a few cm, like the bias I saw on the known-static frames.

**Other notes:**
- I tried adding a feature-tracking term to the moving-phase optimisation and dropped it. It didn't remove the occlusion bias, and it made the trajectory jumpier.
- GPU rendering isn't bit-exact, so reruns of `solve.py` differ by well under a millimetre; seeds are fixed and the decisions don't change.
- My GPU use across all experiments is an estimate from run times, roughly 35–40 minutes, against the 45-minute limit.
- No external tools or weights were downloaded.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoi_common.py`
- `validate.py`
- `validation_table.md`
- `solve_log.txt`
