# CL2_ep002

Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T05:05:21Z

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
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 08:59 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 08:59 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 08:59 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 08:59 proxy.sock

inputs:
total 94808
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 08:59 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 08:59 ..
-rw-rw-r-- 1 hanzhou hanzhou      839 Oct 21 08:59 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       72 Oct 21 08:59 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   194239 Oct 21 08:59 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      398 Oct 21 08:59 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  7970917 Oct 21 08:59 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou  9153188 Oct 21 08:59 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 4943 …[+4811 chars]
```
</details>

**Bash**
```
cd /work; cat logs/run_manifest.txt; echo $CUDA_VISIBLE_DEVICES; python -c "import torch;print(torch.__version__, torch.cuda.device_count())"; cat /pipeline/pipeline_report.json | head -50; find /pipeline -maxdepth 3 | head -50
```

<details><summary>output</summary>

```
date_utc: 2026-10-21T05:05:21Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2_ep002:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000002:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
a2ca757e215299b113f2b6e284d5485280c6d2f2a41c65e09672b9ef1dc72a20  ./camera.json
f951f310d26afa061d8f0006dd4ab8aaf91bf53d96fc45c9194a48b9633225df  ./episode.json
b0e38e53034a3764d2aa43b028bd35e227b14beeaf25cdf8bd62ee11 …[+4022 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, h5py, trimesh
r=np.load('inputs/reconstruction.npz')
for k in r.files: print(k, r[k].shape, r[k].dtype)
m=h5py.File('inputs/masks.h5','r')
for k in m: print(k, m[k].shape, m[k].dtype)
s=trimesh.load('inputs/object_mesh.glb',force='scene',process=False)
g=s.to_geometry(); print(g, g.bounds, g.extents)
print(trimesh.bounds.oriented_bounds(g))
"
ls /pipeline/export/episode_000002/images | head -3; ls /pipeline/export/episode_000002/images | wc -l
```

<details><summary>output</summary>

```
T_world_mesh (980, 4, 4) float32
T_cam_mesh_foundationpose (980, 4, 4) float32
T_cam_mesh_coconet (980, 4, 4) float32
human_vertices_world (980, 4610, 3) float32
human_keypoints_world (980, 70, 3) float32
contact_prob_right_left (980, 2) float32
object (980, 1152, 1536) bool
person (980, 1152, 1536) bool
<trimesh.Trimesh(vertices.shape=(263964, 3), faces.shape=(88020, 3))> [[-0.11327863 -0.06662077 -0.12387678]
 [ 0.12049808  0.08906516  0.1194884 ]] [0.23377671 0.15568593 0.24336518]
(array([[-0.0045892 ,  0.99997997, -0.00435865, -0.01143537],
       [ 0.99976099,  0.00468128,  0.0213555 , -0.00378538],
       [ 0.02137548, -0.0042596 , -0.99976244, -0.00212534],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]), array([0.15508843, 0.23449096, 0.24362373]))
front_stereo_camer …[+11 chars]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/frames outputs tools; python -c "
import cv2
cap=cv2.VideoCapture('inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
i=0
while True:
  ok,f=cap.read()
  if not ok: break
  cv2.imwrite(f'scratch/frames/{i:04d}.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 92]); i+=1
print(i)
"; python -c "import nvdiffrast, open3d; print('ok')"; pip list 2>/dev/null | grep -i -E "torch|kornia|opencv|trimesh|scipy|pytorch3d|lightglue|roma|raft|cotracker"
```

<details><summary>output</summary>

```
30.0 980.0
980
ok
kornia                                   0.8.3
kornia_rs                                0.2.0
opencv-contrib-python                    4.11.0.86
opencv-python                            4.11.0.86
pytorch-lightning                        2.6.6
pytorch3d                                0.7.9
roma                                     1.6.1
scipy                                    1.16.3
torch                                    2.5.1+cu124
torchaudio                               2.5.1+cu124
torchelastic                             0.2.2
torchmetrics                             1.9.0
torchvision                              0.20.1+cu124
trimesh                                  5.1.1
```
</details>

**Agent:** Frames extracted (980 @ 30fps). Now making a contact sheet to understand the episode phases.

**Bash**
```
cd /work; mkdir -p scratch/vis; python -c "
import cv2, numpy as np, h5py
m=h5py.File('inputs/masks.h5','r')
idx=list(range(0,980,35))
tiles=[]
for i in idx:
  f=cv2.imread(f'scratch/frames/{i:04d}.jpg')
  om=m['object'][i]; pm=m['person'][i]
  f[om]=(0.5*f[om]+[0,0,127]).astype(np.uint8)
  f[pm]=(0.7*f[pm]+[76,0,0]).astype(np.uint8)
  f=cv2.resize(f,(384,288)); cv2.putText(f,str(i),(5,25),0,0.8,(0,255,255),2); tiles.append(f)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[k:k+6]) for k in range(0,len(tiles),6)]
cv2.imwrite('scratch/vis/sheet.jpg', np.vstack(rows))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/sheet.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, h5py
m=h5py.File('inputs/masks.h5','r')
O=m['object'][:]
area=O.sum((1,2))
ys,xs=[],[]
for i in range(980):
  yy,xx=np.nonzero(O[i]); 
  if len(yy): ys.append(yy.mean()); xs.append(xx.mean())
  else: ys.append(np.nan); xs.append(np.nan)
np.savez('scratch/maskstats.npz',area=area,cx=np.array(xs),cy=np.array(ys))
for i in range(0,980,10): print(i, area[i], round(xs[i],1) if area[i] else '-', round(ys[i],1) if area[i] else '-')
"
```

<details><summary>output</summary>

```
0 2371 831.5 608.9
10 2342 831.2 609.0
20 2343 831.3 609.0
30 2342 831.2 609.0
40 2337 831.3 609.0
50 2344 831.3 609.0
60 2329 831.3 609.0
70 2337 831.3 609.0
80 2333 831.3 609.0
90 2280 831.9 608.9
100 2252 832.1 609.7
110 2203 832.6 609.6
120 2195 832.0 609.7
130 2216 828.4 608.6
140 2327 823.1 590.1
150 2084 803.4 568.2
160 1664 772.9 574.7
170 1501 752.8 600.6
180 499 746.1 614.8
190 1507 745.0 675.4
200 1495 743.5 737.3
210 1709 744.5 761.7
220 1814 744.3 765.1
230 1813 743.8 765.8
240 1817 743.8 765.5
250 1859 743.3 765.4
260 1907 742.7 765.3
270 2211 741.4 764.4
280 2168 740.7 764.5
290 2220 741.5 764.6
300 2142 740.6 764.4
310 2135 740.7 764.3
320 2216 741.5 764.7
330 2154 740.8 764.4
340 2153 740.7 764.4
350 2234 741.5 764.6
360 2224 741.5 764.6
370 2173 740.8 764.6
380 2218 741.5 …[+1009 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import cv2, numpy as np
tiles=[]
for i in [700,720,740,760,780,800,820,835,845,850,860,870,890,920,950,979]:
  f=cv2.imread(f'scratch/frames/{i:04d}.jpg')[450:1000,650:1250]
  f=cv2.resize(f,(400,367)); cv2.putText(f,str(i),(5,25),0,0.8,(0,255,255),2); tiles.append(f)
rows=[np.hstack(tiles[k:k+4]) for k in range(0,16,4)]
cv2.imwrite('scratch/vis/late.jpg', np.vstack(rows))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/late.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python -c "
import trimesh, numpy as np
s=trimesh.load('inputs/object_mesh.glb',force='scene',process=False)
print(s.geometry.keys())
g=s.to_geometry()
print(type(g.visual), getattr(g.visual,'material',None))
v=g.visual
print(v.uv.shape if hasattr(v,'uv') and v.uv is not None else None)
img=v.material.baseColorTexture if hasattr(v.material,'baseColorTexture') else v.material.image
print(img.size if img is not None else None)
img.save('scratch/vis/tex.png')
print(g.vertices.mean(0), g.center_mass if g.is_watertight else 'not watertight')
"
```

<details><summary>output</summary>

```
odict_keys(['blueeasterbasket01'])
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7e895591cfd0>
(263964, 2)
(3066, 2181)
[-0.01496485  0.01280887 -0.01614906] not watertight
```
</details>

> *thinking:* I've traced the basket's trajectory: static on the table early on, moved to the floor, picked up again for a handover, then placed down and occluded by the person. Now I'll build a renderer to inspect the baseline against this.

**Write**
```
/work/outputs/hoilib.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/hoilib.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
mkdir -p /work/scratch/exp; cat > /work/scratch/exp/overlay.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
key=sys.argv[1]; frames=[int(x) for x in sys.argv[2].split(',')]
if key=='base': Tcm=Tcw@rec['T_world_mesh'].astype(np.float64)
else:
    Tcm=np.load(key)['T_world_mesh']; Tcm=Tcw@Tcm
m=h5py.File('/work/inputs/masks.h5','r')
tiles=[]
for i in frames:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    p=K@Tcm[i,:3,3]; u,v=p[:2]/p[2]
    S=200; x0=int(u-S/2); y0=int(v-S/2)
    col,mk,_=R.render(Tcm[i:i+1],crop=(x0,y0,S,S),scale=2)
    col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mk[0].cpu().numpy()
    crop=np.zeros((S,S,3),np.uint8)
    ys,xs=slice(max(y0,0),min(y0+S,1152)),slice(max(x0,0),min(x0+S,1536))
    crop[ys.start-y0:ys.stop-y0, xs.start-x0:xs.stop-x0]=img[ys,xs]
    crop=cv2.resize(crop,(2*S,2*S))
    om=np.zeros((S,S),np.uint8); om[ys.start-y0:ys.stop-y0, xs.start-x0:xs.stop-x0]=m['object'][i][ys,xs]
    om=cv2.resize(om,(2*S,2*S),interpolation=cv2.INTER_NEAREST)
    a=crop.copy(); b=crop.copy(); b[mk]=col[mk]
    cnt,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(a,cnt,-1,(0,255,0),1)
    cnt,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(a,cnt,-1 …[+365 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/ov1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > /work/scratch/exp/traj.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
key=sys.argv[1] if len(sys.argv)>1 else 'T_world_mesh'
if key in rec.files and key!='T_world_mesh':
    T=Twc@rec[key].astype(np.float64)
else: T=rec['T_world_mesh'].astype(np.float64)
ms=np.load('/work/scratch/maskstats.npz')
Tcw=np.linalg.inv(Twc)
for i in range(0,980,int(sys.argv[2]) if len(sys.argv)>2 else 10):
    R=T[i,:3,:3]; t=T[i,:3,3]
    up=R[:,1]  # mesh y in world
    tilt=np.degrees(np.arccos(np.clip(-up[1],-1,1)))  # angle between mesh y and world up(-y)... sign unknown
    x=R[:,0]; yaw=np.degrees(np.arctan2(x[2],x[0]))
    tc=(Tcw@T[i])[:3,3]
    dR=np.degrees(np.arccos(np.clip((np.trace(T[i-1,:3,:3].T@R)-1)/2,-1,1))) if i>0 else 0
    print(f"{i:4d} pos {t[0]:6.3f} {t[1]:6.3f} {t[2]:6.3f} zc {tc[2]:5.2f} tilt {tilt:6.1f} yaw {yaw:7.1f} dR {dR:5.1f} area {ms['area'][i]}")
EOF
python /work/scratch/exp/traj.py T_world_mesh 10
```

<details><summary>output</summary>

```
   0 pos  0.476  0.307  3.350 zc  3.31 tilt    3.6 yaw    64.9 dR   0.0 area 2371
  10 pos  0.476  0.307  3.350 zc  3.31 tilt    3.6 yaw    64.9 dR   0.1 area 2342
  20 pos  0.476  0.307  3.351 zc  3.31 tilt    3.6 yaw    64.9 dR   0.1 area 2343
  30 pos  0.476  0.307  3.350 zc  3.31 tilt    3.6 yaw    64.9 dR   0.1 area 2342
  40 pos  0.475  0.306  3.343 zc  3.30 tilt    3.6 yaw    64.9 dR   0.1 area 2337
  50 pos  0.475  0.306  3.342 zc  3.30 tilt    3.6 yaw    64.9 dR   0.1 area 2344
  60 pos  0.475  0.306  3.343 zc  3.30 tilt    3.6 yaw    64.9 dR   0.1 area 2329
  70 pos  0.476  0.307  3.348 zc  3.31 tilt    3.6 yaw    64.9 dR   0.1 area 2337
  80 pos  0.475  0.306  3.342 zc  3.30 tilt    3.6 yaw    64.9 dR   0.1 area 2333
  90 pos  0.476  0.307  3.352 zc  3.31 tilt    3.6 yaw    64.9 …[+7166 chars]
```
</details>

**Bash**
```
cat > /work/scratch/exp/zoom.py <<'EOF'
import numpy as np, cv2, sys
ms=np.load('/work/scratch/maskstats.npz')
fr=[int(x) for x in sys.argv[1].split(',')]; tiles=[]
S=int(sys.argv[3]) if len(sys.argv)>3 else 110
for i in fr:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    cx,cy=ms['cx'][i],ms['cy'][i]
    if np.isnan(cx): cx,cy=float(sys.argv[4]),float(sys.argv[5])
    x0=int(cx-S/2); y0=int(cy-S/2)
    c=cv2.resize(img[y0:y0+S,x0:x0+S],(400,400),interpolation=cv2.INTER_CUBIC)
    cv2.putText(c,str(i),(5,25),0,0.8,(0,255,255),2); tiles.append(c)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))
EOF
python /work/scratch/exp/zoom.py 0,60,300,350,400,520,560,600,620,640,660,690 /work/scratch/vis/z1.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/z1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import cv2; t=cv2.imread('/work/scratch/vis/tex.png'); cv2.imwrite('/work/scratch/vis/tex_s.jpg', cv2.resize(t,(1090,1533)))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/tex_s.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > /work/scratch/exp/yawgrid.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
from scipy.spatial.transform import Rotation as Rot
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
i=int(sys.argv[1]); T0=Tcw@rec['T_world_mesh'][i].astype(np.float64)
c=V.mean(0)*0  # rotate about mesh origin y axis
tiles=[]
img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
p=K@T0[:3,3]; u,v=p[:2]/p[2]; S=110; x0=int(u-S/2); y0=int(v-S/2)
tiles.append(cv2.resize(img[y0:y0+S,x0:x0+S],(330,330),interpolation=cv2.INTER_CUBIC))
for k in range(int(sys.argv[2]) if len(sys.argv)>2 else 11):
    a=k*360/11
    Ry=np.eye(4); Ry[:3,:3]=Rot.from_euler('y',a,degrees=True).as_matrix()
    T=T0@Ry
    col,mk,_=R.render(T[None],crop=(x0,y0,S,S),scale=3)
    col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8)
    cv2.putText(col,f'{a:.0f}',(5,25),0,0.8,(0,255,255),2); tiles.append(col)
cv2.imwrite(sys.argv[3] if len(sys.argv)>3 else '/work/scratch/vis/yaw.jpg', np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,12,6)]))
EOF
python /work/scratch/exp/yawgrid.py 0 11 /work/scratch/vis/yaw0.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/yaw0.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > /work/scratch/exp/plotyaw.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
def yawtilt(T):
    R=T[:,:3,:3]; up=R[:,:,1]; x=R[:,:,0]
    tilt=np.degrees(np.arccos(np.clip(-up[:,1],-1,1)))
    # yaw: project mesh x onto plane perpendicular to world up
    yaw=np.degrees(np.arctan2(x[:,2],x[:,0]))
    return yaw,tilt
srcs={'base':rec['T_world_mesh'].astype(np.float64),'fp':Twc@rec['T_cam_mesh_foundationpose'].astype(np.float64),'cc':Twc@rec['T_cam_mesh_coconet'].astype(np.float64)}
for a in sys.argv[1:]:
    srcs[a.split('/')[-1]]=np.load(a)['T_world_mesh']
fig,ax=plt.subplots(4,1,figsize=(18,14),sharex=True)
ms=np.load('/work/scratch/maskstats.npz')
for k,T in srcs.items():
    y,t=yawtilt(T); ax[0].plot(y,'.',ms=2,label=k); ax[1].plot(t,'.',ms=2,label=k)
    Tc=np.linalg.inv(Twc)@T; ax[2].plot(Tc[:,2,3],label=k)
ax[3].plot(ms['area']); 
for a in ax: a.grid(); a.legend()
ax[0].set_ylabel('yaw'); ax[1].set_ylabel('tilt'); ax[2].set_ylabel('zcam'); ax[2].set_ylim(1.5,6)
ax[0].set_xticks(range(0,980,20)); plt.tight_layout(); plt.savefig('/work/scratch/vis/yawplot.png',dpi=60)
EOF
python /work/scratch/exp/plotyaw.py
```

<details><summary>output</summary>

```
/work/scratch/exp/plotyaw.py:20: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  for a in ax: a.grid(); a.legend()
```
</details>

**Read**
```
/work/scratch/vis/yawplot.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
ls /workspace /workspace/*/ | head -60; ls /weights/foundationpose /weights/foundationpose/*; ls /code/reconstruction/modules /code/reconstruction/scripts
```

<details><summary>output</summary>

```
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

/workspace/third_party/:
kaolin
nvdiffrast

/workspace/v2d_cari4d/:
lib

/workspace/v2d_common/:
LICENSE
__init__.py
broadcast.py
datatypes.py
ffv1_sidecar.py
hdf5_transcode.py
pyproject.toml
result_bundle.py
tests
utils.py
v2d_common.egg-info
video.py

/workspace/v2d_foundation_pose/:
lib

/workspace/v2d_mesh/:
lib

/workspace/v2d_moge/:
lib

/workspace/v2d_mv/:
LICENSE
__init__.py
cli
io
math
pyproject.toml
rig
v2d_mv.egg-info
vis

/workspace/v2d_sam2/:
LICENSE
__init__.py
assets
docker
lib

/workspace/v2d_sam3d_body/:
/weights/foundationpose:
nvlabs_pytorch

/weights/foundationpose/nvlabs_pytorch:
2023-10-28-18-33-37
2024-01-11-20-02-45
/code/reconstruction/modules:
v2d_an …[+948 chars]
```
</details>

**Bash**
```
find /workspace/v2d_foundation_pose/lib -maxdepth 3 | grep -v __pycache__ | head -50; cat /pipeline/.stages/05_object_pose.json | head -60
```

<details><summary>output</summary>

```
/workspace/v2d_foundation_pose/lib
/workspace/v2d_foundation_pose/lib/FoundationPose
/workspace/v2d_foundation_pose/lib/__init__.py
/workspace/v2d_foundation_pose/lib/backends.py
/workspace/v2d_foundation_pose/lib/depth_alignment.py
/workspace/v2d_foundation_pose/lib/download_weights.py
/workspace/v2d_foundation_pose/lib/foundation_pose_tracker.py
/workspace/v2d_foundation_pose/lib/fp_utils.py
/workspace/v2d_foundation_pose/lib/multiview_tracker.py
/workspace/v2d_foundation_pose/lib/mv_videos_to_poses.py
/workspace/v2d_foundation_pose/lib/mv_videos_to_poses.yaml
/workspace/v2d_foundation_pose/lib/pyproject.toml
/workspace/v2d_foundation_pose/lib/render_overlay.py
/workspace/v2d_foundation_pose/lib/run_align_depth_to_object.py
/workspace/v2d_foundation_pose/lib/run_align_depth_to_reference_ …[+3915 chars]
```
</details>

**Bash**
```
ls /workspace/v2d_cari4d/lib/cari4d/prep/; wc -l /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py; grep -n "def \|FoundationPose(\|ScorePredictor\|PoseRefinePredictor\|register\|track_one\|est\.\b" /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py | head -60
```

<details><summary>output</summary>

```
__init__.py
align_depth_to_mhr_wild.py
align_utils.py
foundationpose_pose_selection.py
mhr_depth_alignment.py
mhr_depth_backend.py
mhr_depth_filter.py
mhr_depth_h5.py
mhr_effective_masks.py
mhr_export_utils.py
mhr_ffv1_sidecar.py
mhr_foundationpose_diagnostics.py
mhr_foundationpose_training_tiers.py
mhr_geometry_certificate.py
mhr_geometry_crop.py
mhr_gt_object_visibility.py
mhr_mesh_visibility.py
mhr_packed_h5.py
mhr_preprocess_manifest.py
mhr_refit_cache.py
mhr_refit_metadata.py
mhr_render_shards.py
mhr_rgb_h5.py
mhr_sam3d_cache.py
mhr_sensor_depth_h5.py
mhr_wild_depth.py
prepare_mhr_wild_export.py
run_foundationpose_mhr_export.py
run_sam3d_mhr_export.py
1060 /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py
27:from estimater import FoundationPose, PoseRefinePredicto …[+7148 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
g,V,F,UV,TEX=load_mesh('/work/inputs')
import numpy as np
y=V[:,1]; print(y.min(),y.max())
for lo,hi in [(-0.067,-0.06),(-0.03,-0.02),(0.02,0.03),(0.07,0.08),(0.08,0.09)]:
  s=V[(y>lo)&(y<hi)]
  # circle fit
  A=np.c_[2*s[:,0],2*s[:,2],np.ones(len(s))]; b=s[:,0]**2+s[:,2]**2
  c=np.linalg.lstsq(A,b,rcond=None)[0]; r=np.sqrt(c[2]+c[0]**2+c[1]**2)
  print(lo,hi,len(s),'center',c[:2],'r',r, 'xrange',s[:,0].min(),s[:,0].max(),'zrange',s[:,2].min(),s[:,2].max())
"

```

<details><summary>output</summary>

```
-0.06662077456712723 0.08906515687704086
-0.067 -0.06 37436 center [0.00531509 0.00334448] r 0.08915438309053322 xrange -0.085862897336483 0.0953073650598526 zrange -0.08718760311603546 0.09369972348213196
-0.03 -0.02 24259 center [-0.00332787 -0.00896663] r 0.09937987197319154 xrange -0.11289902776479721 0.09988883882761002 zrange -0.11611330509185791 0.09910093247890472
0.02 0.03 10410 center [ 0.00186933 -0.0027821 ] r 0.10952007139718374 xrange -0.1063094288110733 0.1072477325797081 zrange -0.12166684120893478 0.10726961493492126
0.07 0.08 12190 center [0.00319985 0.00379899] r 0.11220988236087102 xrange -0.10730316489934921 0.11451101303100586 zrange -0.10735582560300827 0.1135357916355133
0.08 0.09 62610 center [0.00380674 0.00332777] r 0.11380777984003951 xrange -0.11327863484621048 …[+68 chars]
```
</details>

**Bash**
```
cat > /work/scratch/exp/yawscore.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch, torch.nn.functional as Fn
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
m=h5py.File('/work/inputs/masks.h5','r')
AX=np.array([0.004,0,0.003])
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    M[:3,3]=AX-M[:3,:3]@AX; return M
def grads(x):  # x (B,1,H,W)
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=x.device)[None,None]/8
    gx=Fn.conv2d(x,kx,padding=1); gy=Fn.conv2d(x,kx.transpose(2,3),padding=1); return gx,gy
frames=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
src=sys.argv[4] if len(sys.argv)>4 else 'base'
yaws=np.radians(np.arange(0,360,10))
S=96; SC=4
res=[]
for i in frames:
    T0=Tcw@rec['T_world_mesh'][i].astype(np.float64)
    if src!='base': T0=Tcw@np.load(src)['T_world_mesh'][i]
    p=K@T0[:3,3]; u,v=p[:2]/p[2]; x0=int(u-S/2); y0=int(v-S/2)
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S]
    om=m['object'][i][y0:y0+S,x0:x0+S]; pm=m['person'][i][y0:y0+S,x0:x0+S]
    Ts=np.stack([T0@yawrot(a) for a in yaws])
    col,mk,_=R.render(Ts,crop=(x0,y0,S,S),scale=SC)
    col=col.permute(0,3,1,2); mk=mk[:,None].float()
    col=Fn.avg_pool2d(col,2); mk=Fn.avg_pool2d(mk,2)   # 2x res
    real=torch.tensor(cv2.resize(img,( …[+1856 chars]
```

<details><summary>output</summary>

```
iou -0.45 -0.36 -0.36 -0.29 -0.36 -0.43 -0.47 -0.49 -0.12 -0.02 -0.11 +0.18 +0.69 +1.01 +1.23 +1.50 +1.77 +2.04 +2.01 +1.65 +1.05 +0.53 +0.07 -0.46 -0.88 -1.13 -1.01 -1.08 -1.35 -1.19 -1.01 -0.80 -0.42 -0.19 -0.31 -0.43
   argbest hist 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 17 10 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
edge +0.31 +0.09 +0.46 +0.68 +0.99 +1.09 +1.68 +1.32 +1.14 +1.58 +0.96 +0.15 -0.55 -0.74 -1.24 -1.29 -1.56 -1.90 -1.49 -1.08 -0.94 -0.73 -0.11 +0.39 +1.34 +0.22 -0.02 +0.63 +0.32 -0.26 -0.63 -0.36 +0.03 -0.13 -0.03 -0.33
   argbest hist 0 0 0 0 0 0 18 1 0 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 2 0 0 0 0 0 0 0 0
ncc +0.39 +0.32 +0.05 -0.09 -0.20 -0.46 -0.77 -1.06 -1.11 -1.26 -1.35 -1.52 -1.49 -1.45 -1.31 -1.36 -1.04 -0.78 -0.50 -0.22 +0.11 +0.45 +0.73 +0.91 +1.06 +1.25 +1.28 +1.39 +1.34 +1 …[+439 chars]
```
</details>

**Bash**
```
cat > /work/scratch/exp/yawvis.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
AX=np.array([0.004,0,0.003])
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    M[:3,3]=AX-M[:3,:3]@AX; return M
i=int(sys.argv[1]); offs=[float(x) for x in sys.argv[2].split(',')]
T0=Tcw@rec['T_world_mesh'][i].astype(np.float64)
if len(sys.argv)>4: T0=Tcw@np.load(sys.argv[4])['T_world_mesh'][i]
S=96; p=K@T0[:3,3]; u,v=p[:2]/p[2]; x0=int(u-S/2); y0=int(v-S/2)
img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S]
tiles=[cv2.resize(img,(300,300),interpolation=cv2.INTER_CUBIC)]
for a in offs:
    col,mk,_=R.render((T0@yawrot(np.radians(a)))[None],crop=(x0,y0,S,S),scale=4)
    c=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mk[0].cpu().numpy()
    c=cv2.resize(c,(S,S),interpolation=cv2.INTER_AREA); mk=cv2.resize(mk.astype(np.uint8),(S,S),interpolation=cv2.INTER_AREA)>0
    b=img.copy(); b[mk]=c[mk]
    b=cv2.resize(b,(300,300),interpolation=cv2.INTER_CUBIC)
    cv2.putText(b,f'{a:.0f}',(5,25),0,0.8,(0,255,255),2); tiles.append(b)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3], np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))
EOF
python /work/scratch/exp/yawvis.py 350 0,60,120,170,220,27 …[+33 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/yv350.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > /work/scratch/exp/iou.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
src=sys.argv[1]
T=rec['T_world_mesh'].astype(np.float64) if src=='base' else np.load(src)['T_world_mesh']
Tc=Tcw@T
m=h5py.File('/work/inputs/masks.h5','r'); O=m['object']; P=m['person']
out=[]
for i in range(980):
    om=O[i]; pm=P[i]
    _,mk,_=R.render(Tc[i:i+1]); mk=mk[0].cpu().numpy()
    v=~pm
    inter=(mk&om&v).sum(); uni=((mk|om)&v).sum()
    iou=inter/uni if uni>0 else np.nan
    ys,xs=np.nonzero(mk&v); yo,xo=np.nonzero(om)
    d=(xs.mean()-xo.mean(), ys.mean()-yo.mean()) if len(xs) and len(xo) else (np.nan,np.nan)
    out.append([iou,(mk&v).sum(),om.sum(),d[0],d[1]])
out=np.array(out); np.save(sys.argv[2],out)
for a in range(0,980,20):
    s=out[a:a+20]; print(a, np.nanmean(s,0).round(2))
EOF
python /work/scratch/exp/iou.py base /work/scratch/iou_base.npy
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: ba4i6nq5w). Output is being written to: /tmp/claude-1003/-work/8189bfce-9305-4501-9089-bb725c28b792/tasks/ba4i6nq5w.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
```
</details>

**Bash**
```
cat > /work/scratch/exp/hands.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
kp=rec['human_keypoints_world']; T=rec['T_world_mesh']
c=T[:,:3,3]+np.einsum('fij,j->fi',T[:,:3,:3],np.array([0.004,0.01,0.003]))
rh=kp[:,21:42].mean(1); lh=kp[:,42:63].mean(1)
cp=rec['contact_prob_right_left']
fig,ax=plt.subplots(3,1,figsize=(18,10),sharex=True)
ax[0].plot(np.linalg.norm(rh-c,axis=1),label='R-obj'); ax[0].plot(np.linalg.norm(lh-c,axis=1),label='L-obj'); ax[0].set_ylim(0,1)
ax[1].plot(cp[:,0],label='cR'); ax[1].plot(cp[:,1],label='cL')
for k,l in [(0,'x'),(1,'y'),(2,'z')]:
    ax[2].plot(rh[:,k],label='rh'+l); ax[2].plot(lh[:,k],'--',label='lh'+l)
for a in ax: a.grid(); a.legend()
ax[0].set_xticks(range(0,980,20)); plt.tight_layout(); plt.savefig('/work/scratch/vis/hands.png',dpi=60)
EOF
python /work/scratch/exp/hands.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/hands.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/8189bfce-9305-4501-9089-bb725c28b792/tasks/ba4i6nq5w.output
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import h5py,time
m=h5py.File('/work/inputs/masks.h5','r'); d=m['object']; print(d.chunks, d.compression)
t=time.time(); x=d[0:50]; print(time.time()-t)
"; cat /tmp/claude-1003/-work/8189bfce-9305-4501-9089-bb725c28b792/tasks/ba4i6nq5w.output
```

<details><summary>output</summary>

```
(31, 72, 96) gzip
0.20726633071899414
0 [ 8.70000e-01  2.25390e+03  2.34245e+03  2.76000e+00 -2.00000e-01]
20 [ 8.70000e-01  2.25525e+03  2.34005e+03  2.74000e+00 -2.00000e-01]
40 [ 8.70000e-01  2.27530e+03  2.33765e+03  2.78000e+00 -2.90000e-01]
60 [ 8.70000e-01  2.26365e+03  2.33450e+03  2.77000e+00 -2.40000e-01]
80 [ 8.70000e-01  2.24585e+03  2.30000e+03  2.43000e+00 -2.90000e-01]
100 [ 8.70000e-01  2.20235e+03  2.21455e+03  1.91000e+00 -7.70000e-01]
120 [ 8.70000e-01  2.13005e+03  2.21235e+03  6.60000e-01 -1.10000e+00]
140 [ 7.90000e-01  1.77055e+03  2.07760e+03 -1.29000e+00 -3.22000e+00]
160 [ 7.7000e-01  1.4214e+03  1.3619e+03  1.1300e+00 -7.8000e-01]
180 [ 4.8000e-01  1.3921e+03  1.0479e+03  3.0000e-02 -4.0300e+00]
200 [ 7.50000e-01  1.46235e+03  1.66475e+03 -1.60000e-01 -3.61000e+0 …[+2449 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp && python overlay.py base 600,640,660,680,700,720 /work/scratch/vis/ov2.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/ov2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The baseline tracking is noticeably off during hand-held frames (640–720, ~6–9 cm lateral error) and completely breaks down during the occluded carry and rest segments (740–979, errors of meters and flipped orientation). Next I'll build a differentiable silhouette fitter using nvdiffrast with temporal regularization to fix this.

**Bash**
```
cd /work/scratch/exp; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
import open3d as o3d, numpy as np, time
g,V,F,UV,TEX=load_mesh('/work/inputs')
t=time.time()
m=trimesh.Trimesh(V,F,process=False); m.merge_vertices(); print(m.vertices.shape,m.faces.shape)
mo=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices),o3d.utility.Vector3iVector(m.faces))
md=mo.simplify_quadric_decimation(6000)
print(len(md.vertices),len(md.triangles),time.time()-t)
d=trimesh.Trimesh(np.asarray(md.vertices),np.asarray(md.triangles))
print(d.bounds, m.bounds)
"

```

<details><summary>output</summary>

```
(44260, 3) (88020, 3)
3042 6000 1.7612435817718506
[[-0.11326338 -0.06653326 -0.12383926]
 [ 0.1204943   0.08906516  0.11939317]] [[-0.11327863 -0.06662077 -0.12387678]
 [ 0.12049808  0.08906516  0.1194884 ]]
```
</details>

**Bash**
```
cat >> /work/outputs/hoilib.py <<'EOF'


# ---------------------------------------------------------------------------------------------
# geometry helpers
def so3_exp(w):
    """w: (...,3) torch -> (...,3,3)"""
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    K = torch.zeros(w.shape[:-1] + (3, 3), dtype=w.dtype, device=w.device)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def rot_angle_deg(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def decimated_mesh(verts, faces, n_faces=6000):
    import open3d as o3d
    m = trimesh.Trimesh(verts, faces, process=False)
    m.merge_vertices()
    mo = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    md = mo.simplify_quadric_decimation(n_faces)
    return np.asarray(md.vertices, np.float64), np.asarray(md.triangles, np.int64)


class SilRenderer:
    """Differentiable (antialiased) silhouette renderer for per-frame crops."""

    def __init__(self, verts, faces, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v …[+1195 chars]
```

<details><summary>output</summary>

```
ok
```
</details>

**Bash**
```
cat >> /work/outputs/hoilib.py <<'EOF'


def load_mask_crops(inp, centers, sizes, res=128, chunk=62):
    """Crop object/person masks around per-frame centers (u,v) with square side sizes (px), area-resampled to res.
    Returns obj (F,res,res) float32, per (F,res,res) float32, inimg (F,res,res) float32, boxes (F,3)=(x0,y0,size)."""
    import h5py
    import cv2
    f = h5py.File(f"{inp}/masks.h5", "r")
    O, P = f["object"], f["person"]
    F, H, W = O.shape
    obj = np.zeros((F, res, res), np.float32)
    per = np.zeros((F, res, res), np.float32)
    inim = np.zeros((F, res, res), np.float32)
    boxes = np.zeros((F, 3), np.float64)
    for a in range(0, F, chunk):
        oc = O[a:a + chunk]
        pc = P[a:a + chunk]
        for k in range(oc.shape[0]):
            i = a + k
            s = int(np.ceil(sizes[i]))
            x0 = int(np.round(centers[i][0] - s / 2))
            y0 = int(np.round(centers[i][1] - s / 2))
            boxes[i] = (x0, y0, s)
            for src, dst in ((oc[k], obj), (pc[k], per), (None, inim)):
                buf = np.zeros((s, s), np.float32)
                xa, xb = max(x0, 0), min(x0 + s, W)
                ya, yb = max(y0, 0), min(y0 + s, H)
                if xb > xa and yb > ya:
                    buf[ya - y0:yb - y0, xa - x0:xb - x0] = 1.0 if src is None else src[ya:yb, xa:xb]
                dst[i] = cv2.resize(buf, (res, res), interpolation=cv2.INTER_AREA)
    return obj, per, inim, boxes


def mask_stats(inp, chunk=62):
    "" …[+1248 chars]
```

<details><summary>output</summary>

```
6.484295845031738
```
</details>

**Bash**
```
cat >> /work/outputs/hoilib.py <<'EOF'


def fit_silhouettes(sil, R0, anchor0, Kc, obj, valid, Pc, K, iters=300, lr=0.01, w_rot_prior=1.0,
                    w_smooth_t=0.0, w_smooth_r=0.0, free_depth=False, fit_rot=True, frame_w=None,
                    static=False, res=128, w_depth_prior=0.0, rot_mask=None):
    """Joint silhouette fit of a sequence of frames (or one shared pose if static=True).
    R0 (F,3,3) init camera rotations, anchor0 (F,3) init camera-frame position of mesh point Pc.
    Optimises a rotation increment w (camera frame, R = exp(w) R0), the image position of the anchor
    (pixel offsets) and optionally log-depth. Returns R (F,3,3), anchor (F,3), per-frame loss."""
    dev = "cuda"
    F = len(R0)
    tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
    R0t, a0, Kct, objt, valt = tt(R0), tt(anchor0), tt(Kc), tt(obj), tt(valid)
    Pct = tt(Pc)
    fw = tt(np.ones(F) if frame_w is None else frame_w)
    n = 1 if static else F
    w = torch.zeros(n, 3, device=dev, requires_grad=True)
    duv = torch.zeros(n, 2, device=dev, requires_grad=True)
    ldz = torch.zeros(n, 1, device=dev, requires_grad=True)
    params = [w, duv] + ([ldz] if free_depth else [])
    opt = torch.optim.Adam([{"params": [w], "lr": lr if fit_rot else 0.0},
                            {"params": [duv], "lr": lr * 30},
                            {"params": [ldz], "lr": lr * 0.3 if free_depth else 0.0}])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, …[+2643 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
cat > /work/scratch/exp/prep.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tc=Tcw@rec['T_world_mesh'].astype(np.float64)
Pc=np.array([0.004,0.011,0.003])
ms=np.load('/work/scratch/mstats.npz')
anc=np.einsum('fij,j->fi',Tc[:,:3,:3],Pc)+Tc[:,:3,3]
proj=(K@anc.T).T; proj=proj[:,:2]/proj[:,2:]
diam=K[0,0]*0.26/anc[:,2]
area,bbox=ms['area'],ms['bbox']
cen=np.where((area>200)[:,None], (bbox[:,:2]+bbox[:,2:])/2, proj)
bsz=np.nan_to_num(np.max(bbox[:,2:]-bbox[:,:2],1))
size=np.maximum(2.0*diam,1.5*bsz)
obj,per,inim,boxes=load_mask_crops('/work/inputs',cen,size,128)
np.savez('/work/scratch/crops.npz',obj=obj,per=per,inim=inim,boxes=boxes)
EOF
python /work/scratch/exp/prep.py && ls -la /work/scratch/crops.npz
```

<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 192700340 Oct 21 09:11 /work/scratch/crops.npz
```
</details>

**Bash**
```
cat > /work/scratch/exp/fit1.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
import time
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
Vd,Fd=decimated_mesh(V,F)
sil=SilRenderer(Vd,Fd)
Tcw=np.linalg.inv(Twc)
Tc=Tcw@rec['T_world_mesh'].astype(np.float64)
Pc=np.array([0.004,0.011,0.003])
c=np.load('/work/scratch/crops.npz')
a,b=int(sys.argv[1]),int(sys.argv[2]); fr=np.arange(a,b)
free=len(sys.argv)>3 and sys.argv[3]=='free'
anc=np.einsum('fij,j->fi',Tc[:,:3,:3],Pc)+Tc[:,:3,3]
Kc=crop_K(K,c['boxes'],128)
valid=c['inim']*(1-c['per'])
t=time.time()
R,A,iou=fit_silhouettes(sil,Tc[fr,:3,:3],anc[fr],Kc[fr],c['obj'][fr],valid[fr],Pc,K,iters=300,w_rot_prior=0.02,free_depth=free)
print('time',time.time()-t)
# iou of init
R0,A0,iou0=fit_silhouettes(sil,Tc[fr,:3,:3],anc[fr],Kc[fr],c['obj'][fr],valid[fr],Pc,K,iters=0)
dang=rot_angle_deg(R,Tc[fr,:3,:3])
for k in range(0,len(fr),5):
    du=(K@A[k])[:2]/A[k,2]-(K@anc[fr[k]])[:2]/anc[fr[k],2]
    print(fr[k], f'iou {iou0[k]:.2f}->{iou[k]:.2f} drot {dang[k]:5.1f} duv {du[0]:6.1f} {du[1]:6.1f} z {anc[fr[k],2]:.3f}->{A[k,2]:.3f}')
out=Tc.copy()
for k,i in enumerate(fr):
    out[i,:3,:3]=R[k]; out[i,:3,3]=A[k]-R[k]@Pc
np.savez(f'/work/scratch/fit1_{a}_{b}.npz',T_world_mesh=Twc@out)
EOF
python /work/scratch/exp/fit1.py 600 740
```

<details><summary>output</summary>

```
time 7.928295612335205
600 iou 0.72->0.93 drot   6.2 duv    1.8    8.1 z 2.816->2.816
605 iou 0.78->0.88 drot   7.8 duv    1.6    4.7 z 2.781->2.781
610 iou 0.74->0.93 drot   5.2 duv    0.0    6.8 z 2.713->2.713
615 iou 0.71->0.87 drot  13.3 duv    0.9    7.5 z 2.668->2.668
620 iou 0.72->0.81 drot   8.7 duv    0.4    5.4 z 2.608->2.608
625 iou 0.66->0.91 drot   5.7 duv    4.3    8.1 z 2.559->2.559
630 iou 0.72->0.91 drot   5.3 duv    3.4    7.1 z 2.564->2.564
635 iou 0.60->0.93 drot   1.7 duv    4.0   13.3 z 2.531->2.531
640 iou 0.58->0.93 drot   1.4 duv    4.4   14.3 z 2.528->2.528
645 iou 0.65->0.78 drot  12.8 duv    1.9    6.6 z 2.493->2.493
650 iou 0.58->0.92 drot   2.2 duv    5.8   14.4 z 2.478->2.478
655 iou 0.54->0.93 drot   5.3 duv    6.9   17.0 z 2.465->2.465
660 iou 0.51->0.84 dr …[+986 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; python fit1.py 600 740 free | awk 'NR%2==1'; python fit1.py 270 400 free | head -12
```

<details><summary>output</summary>

```
time 5.352678537368774
605 iou 0.78->0.91 drot   5.8 duv    1.3    5.6 z 2.781->2.612
615 iou 0.71->0.94 drot  13.1 duv    1.8    9.0 z 2.668->2.510
625 iou 0.66->0.92 drot  13.7 duv    4.5   10.0 z 2.559->2.433
635 iou 0.60->0.94 drot   1.9 duv    4.4   13.5 z 2.531->2.472
645 iou 0.65->0.95 drot   1.5 duv    3.3   11.3 z 2.493->2.430
655 iou 0.54->0.96 drot   3.5 duv    6.9   16.9 z 2.465->2.372
665 iou 0.47->0.93 drot   2.6 duv    5.8   20.0 z 2.506->2.469
675 iou 0.32->0.83 drot   6.1 duv   13.8   20.7 z 2.598->2.324
685 iou 0.35->0.83 drot   7.4 duv    9.5   19.9 z 2.537->2.475
695 iou 0.32->0.84 drot  10.8 duv   17.9   17.3 z 2.550->2.426
705 iou 0.27->0.79 drot  11.2 duv   20.6   21.3 z 2.534->2.631
715 iou 0.22->0.83 drot   9.8 duv   21.7   31.4 z 2.532->2.881
725 iou 0.35->0.84 dr …[+1090 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoilib.py').read()
s=s.replace("""        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        Kc[:2] *= scale""","""        Kc[0, 2] -= x0 - 0.5
        Kc[1, 2] -= y0 - 0.5
        Kc[:2] *= scale""")
open('hoilib.py','w').write(s)
EOF
grep -n "x0 - 0.5" hoilib.py; cd /work/scratch/exp; python fit1.py 600 740 > /dev/null; sed -i 's/yaws=np.radians(np.arange(0,360,10))/yaws=np.radians(np.arange(0,360,5))/; s/np.arange(0,360,10)/np.arange(0,360,5)/' yawscore.py; grep -n "minlength\|src\b" yawscore.py | head
```

<details><summary>output</summary>

```
62:        Kc[0, 2] -= x0 - 0.5
17:src=sys.argv[4] if len(sys.argv)>4 else 'base'
23:    if src!='base': T0=Tcw@np.load(src)['T_world_mesh'][i]
55:    best=np.bincount(np.argmax(res[:,:,k]*(1 if k<3 else -1),1),minlength=36)
```
</details>

**Bash**
```
cd /work/scratch/exp; python - <<'EOF'
s=open('yawscore.py').read()
i=s.index("names=['iou'")
s=s[:i]+"""names=['iou','edge','ncc','negcdist']
res[:,:,3]*=-1
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,4,figsize=(20,6))
for k in range(4):
    z=(res[:,:,k]-res[:,:,k].mean(1,keepdims=True))/(res[:,:,k].std(1,keepdims=True)+1e-9)
    ax[k].imshow(z,aspect='auto',extent=[0,360,frames[-1],frames[0]]); ax[k].set_title(names[k])
    print(names[k],'mean-z argmax',np.degrees(yaws[np.argmax(z.mean(0))]), 'top3', np.degrees(yaws[np.argsort(-z.mean(0))[:6]]))
plt.savefig(sys.argv[5] if len(sys.argv)>5 else '/work/scratch/vis/yawscore.png',dpi=50)
"""
open('yawscore.py','w').write(s)
EOF
python yawscore.py 600 735 3 /work/scratch/fit1_600_740.npz /work/scratch/vis/ys_carry.png
```

<details><summary>output</summary>

```
iou mean-z argmax 250.00000000000003 top3 [250. 255. 245. 285. 260. 240.]
edge mean-z argmax 280.0 top3 [280. 275. 285. 165. 170. 160.]
ncc mean-z argmax 100.0 top3 [100.  95. 110. 105.  80. 115.]
negcdist mean-z argmax 250.00000000000003 top3 [250. 260. 255. 245. 270. 275.]
```
</details>

**Read**
```
/work/scratch/vis/ys_carry.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > shiftplot.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
res=np.load(sys.argv[1]); a,b,st=[int(x) for x in sys.argv[2:5]]; src=sys.argv[5]
T=rec['T_world_mesh'].astype(np.float64) if src=='base' else np.load(src)['T_world_mesh']
fr=np.arange(a,b,st)
x=T[fr,:3,0]; yaw=np.degrees(np.arctan2(x[:,2],x[:,0]))
# rotation about mesh +y by delta: mesh x -> cos*x - sin*z ... sign check numerically
from scipy.spatial.transform import Rotation as Rot
Ry=Rot.from_euler('y',10,degrees=True).as_matrix()
x2=(T[fr[0],:3,:3]@Ry)[:,0]; dyaw=np.degrees(np.arctan2(x2[2],x2[0]))-yaw[0]
sign=np.sign(((dyaw+180)%360)-180); print('sign',sign)
ny=res.shape[1]; step=360/ny
fig,ax=plt.subplots(1,4,figsize=(20,6))
names=['iou','edge','ncc','negcdist']
for k in range(4):
    z=(res[:,:,k]-res[:,:,k].mean(1,keepdims=True))/(res[:,:,k].std(1,keepdims=True)+1e-9)
    out=np.zeros_like(z)
    for j in range(len(fr)):
        for yi in range(ny):
            absy=(yaw[j]+sign*yi*step)%360
            out[j,int(round(absy/step))%ny]=z[j,yi]
    ax[k].imshow(out,aspect='auto',extent=[0,360,fr[-1],fr[0]]); ax[k].set_title(names[k])
    m=out.mean(0); print(names[k],'abs-yaw best',np.arange(ny)[np.argsort(-m)[:5]]*step)
plt.savefig(sys.argv[6],dpi=50)
EOF
python shiftplot.py /work/scratch/yawscore_600_734.npy 600 735  …[+67 chars]
```

<details><summary>output</summary>

```
Exit code 1
Traceback (most recent call last):
  File "/work/scratch/exp/shiftplot.py", line 5, in <module>
    res=np.load(sys.argv[1]); a,b,st=[int(x) for x in sys.argv[2:5]]; src=sys.argv[5]
        ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/yawscore_600_734.npy'
```
</details>

**Bash**
```
cd /work/scratch/exp; ls /work/scratch/*.npy; python shiftplot.py /work/scratch/yawscore_600_732.npy 600 735 3 /work/scratch/fit1_600_740.npz /work/scratch/vis/ys_carry_abs.png
```

<details><summary>output</summary>

```
/work/scratch/iou_base.npy
/work/scratch/yawscore_280_425.npy
/work/scratch/yawscore_600_732.npy
sign 1.0
iou abs-yaw best [190. 185. 180. 195. 210.]
edge abs-yaw best [305. 310. 300. 315. 295.]
ncc abs-yaw best [50. 40. 75. 45. 85.]
negcdist abs-yaw best [  0.   5. 355.  10. 350.]
```
</details>

**Read**
```
/work/scratch/vis/ys_carry_abs.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); kp=r['human_keypoints_world']
h=kp[650,21:42]; 
D=np.linalg.norm(h[:,None]-h[None],axis=-1)
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print((D*100).round(0).astype(int))
print(kp[650,:21]); 
"
```

<details><summary>output</summary>

```
[[ 0  2  6  9  3  2  3  5  4  3  4  6  4  4  5  7  5  6  7  8 12]
 [ 2  0  3  6  5  4  3  3  6  5  5  4  6  5  5  6  6  6  7  7 10]
 [ 6  3  0  3  9  7  6  4  9  8  7  5  8  8  7  6  8  8  8  7  7]
 [ 9  6  3  0 12 10  9  6 12 11 10  6 11 10  9  7 10 10  9  7  4]
 [ 3  5  9 12  0  2  4  8  2  2  4  8  3  4  6  9  5  6  8 10 15]
 [ 2  4  7 10  2  0  2  6  3  2  3  6  4  4  4  7  5  6  7  9 13]
 [ 3  3  6  9  4  2  0  4  5  4  2  4  6  5  4  6  6  6  6  7 11]
 [ 5  3  4  6  8  6  4  0  8  7  5  3  8  7  6  4  8  7  7  6  8]
 [ 4  6  9 12  2  3  5  8  0  2  5  8  2  3  5  8  4  5  7  9 14]
 [ 3  5  8 11  2  2  4  7  2  0  3  6  3  2  3  7  3  4  5  8 13]
 [ 4  5  7 10  4  3  2  5  5  3  0  4  5  3  2  5  4  4  4  6 12]
 [ 6  4  5  6  8  6  4  3  8  6  4  0  7  6  4  2  6  5  4  4  8]
 [ 4  6  …[+1028 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > handrot.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
kp=rec['human_keypoints_world'].astype(np.float64)
def hand_frames(kp,base):
    h=kp[:,base:base+21]; wr=h[:,20]; mcp=h[:,[3,7,11,15,19]]
    x=mcp.mean(1)-wr; x/=np.linalg.norm(x,axis=1,keepdims=True)
    a=mcp[:,0]-mcp[:,4]; z=np.cross(x,a); z/=np.linalg.norm(z,axis=1,keepdims=True); y=np.cross(z,x)
    return np.stack([x,y,z],-1), wr
T=rec['T_world_mesh'].astype(np.float64) if sys.argv[1]=='base' else np.load(sys.argv[1])['T_world_mesh']
base=int(sys.argv[2]); a,b=int(sys.argv[3]),int(sys.argv[4])
Rh,wr=hand_frames(kp,base)
rel=np.einsum('fji,fjk->fik',Rh,T[:,:3,:3])
def yaw(R): x=R[:,:,0]; return np.degrees(np.arctan2(x[:,2],x[:,0]))
hy=np.degrees(np.arctan2(Rh[:,2,0],Rh[:,0,0]))
oy=yaw(T[:,:3,:3])
ref=rel[(a+b)//2]
for i in range(a,b,5):
    print(i, f'handyaw {hy[i]:7.1f} objyaw {oy[i]:7.1f} diff {((oy[i]-hy[i]+180)%360)-180:7.1f}  rel-vs-mid {rot_angle_deg(rel[i],ref):6.1f}')
EOF
python handrot.py /work/scratch/fit1_600_740.npz 42 560 740
```

<details><summary>output</summary>

```
560 handyaw   -53.3 objyaw   -48.9 diff     4.4  rel-vs-mid   70.4
565 handyaw   -58.2 objyaw   -62.6 diff    -4.4  rel-vs-mid   60.4
570 handyaw   -53.2 objyaw   -41.3 diff    11.9  rel-vs-mid   75.2
575 handyaw   -48.2 objyaw   -24.7 diff    23.5  rel-vs-mid   88.1
580 handyaw   -51.3 objyaw   -31.7 diff    19.7  rel-vs-mid   84.9
585 handyaw   -54.1 objyaw   -29.4 diff    24.7  rel-vs-mid   91.1
590 handyaw   -52.9 objyaw   -32.6 diff    20.3  rel-vs-mid   85.9
595 handyaw   -51.8 objyaw   -20.5 diff    31.3  rel-vs-mid   99.9
600 handyaw   -52.3 objyaw    16.7 diff    69.0  rel-vs-mid  135.6
605 handyaw   -49.8 objyaw    14.2 diff    64.0  rel-vs-mid  131.8
610 handyaw   -46.1 objyaw    17.1 diff    63.2  rel-vs-mid  130.5
615 handyaw   -43.7 objyaw    22.5 diff    66.2  rel-vs-mid  13 …[+1611 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
g,V,F,UV,TEX=load_mesh('/work/inputs')
c=np.array([0.004,0.003])
r=np.linalg.norm(V[:,[0,2]]-c,axis=1); y=V[:,1]
ang=np.degrees(np.arctan2(V[:,2]-c[1],V[:,0]-c[0]))
# body radius profile: linear between bottom r=.089 at y=-.067 and r=.114 at y=.089
rb=0.089+(y+0.067)/(0.156)*(0.114-0.089)
out=r>rb+0.006
print(out.sum())
H,xe,ye=np.histogram2d(ang[out],y[out],bins=[36,8],range=[[-180,180],[-0.07,0.09]])
np.set_printoptions(linewidth=200)
print(ye.round(3))
for k in range(36): print(int(xe[k]), H[k].astype(int))
"
```

<details><summary>output</summary>

```
74208
[-0.07 -0.05 -0.03 -0.01  0.01  0.03  0.05  0.07  0.09]
-180 [   0   19 2202    0    0    0    0    0]
-170 [   0 2474 5439    0    0    0    0    0]
-160 [   0  932 1156    0    0    0    0    0]
-150 [   0 1226 1308    0    0    0    0    0]
-140 [   0 2843 2824    0    0    0    0    0]
-130 [    0 11250  6265     0     0     0     0     0]
-120 [   0 1808 3943    0    0    0    0    0]
-110 [   0    0 3875  229    0    0    0    0]
-100 [   0    0  507 3758    0    0    0    0]
-90 [   0    0    0 2054 2735    0    0    0]
-80 [   0    0    0    0 2520 1640    0    0]
-70 [   0    0    0    0    0 2267    0    0]
-60 [  0   0   0   0   0   0   0 350]
-50 [0 0 0 0 0 0 0 0]
-40 [0 0 0 0 0 0 0 0]
-30 [0 0 0 0 0 0 0 0]
-20 [0 0 0 0 0 0 0 0]
-10 [0 0 0 0 0 0 0 0]
0 [0 0 0 0 0 0 0 0]
1 …[+499 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; sed -i 's/S=96; p=K@T0/S=int(sys.argv[5]) if len(sys.argv)>5 else 96; p=K@T0/' yawvis.py; python yawvis.py 660 0,30,60,90,120,150,180,210,240,270,300 /work/scratch/vis/yv660.jpg /work/scratch/fit1_600_740.npz 110
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/yv660.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import pickle,numpy as np
d=pickle.load(open('/pipeline/foundationpose/episode_000002.pkl','rb'))
print(type(d)); 
if isinstance(d,dict):
  for k,v in d.items(): print(k, type(v), getattr(v,'shape',None) if not isinstance(v,(list,dict)) else (len(v)))
" 2>&1 | head -40; ls /pipeline/foundationpose/debug | head
```

<details><summary>output</summary>

```
<class 'dict'>
frames <class 'list'> 980
camera_id <class 'int'> None
fp_poses_cam <class 'numpy.ndarray'> (980, 1, 4, 4)
fp_candidate_count <class 'numpy.ndarray'> (980,)
fp_selected_candidate_index <class 'numpy.ndarray'> (980,)
fp_selected_iou <class 'numpy.ndarray'> (980,)
fp_selected_visibility <class 'numpy.ndarray'> (980,)
fp_retry_count <class 'numpy.ndarray'> (980,)
fp_reliable <class 'numpy.ndarray'> (980,)
fp_oracle_rotation_error_deg <class 'numpy.ndarray'> (980,)
fp_translation_threshold_m <class 'numpy.ndarray'> (980,)
fp_selected_translation_distance_m <class 'numpy.ndarray'> (980,)
gt_object_visibility_ratio <class 'numpy.ndarray'> (980,)
gt_object_visibility_valid <class 'numpy.ndarray'> (980,)
gt_object_rendered_pixels <class 'numpy.ndarray'> (980,)
gt_object_visible_pixe …[+994 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > diffplot.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
kp=rec['human_keypoints_world'].astype(np.float64)
def hand_frames(kp,base):
    h=kp[:,base:base+21]; wr=h[:,20]; mcp=h[:,[3,7,11,15,19]]
    x=mcp.mean(1)-wr; x/=np.linalg.norm(x,axis=1,keepdims=True)
    a=mcp[:,0]-mcp[:,4]; z=np.cross(x,a); z/=np.linalg.norm(z,axis=1,keepdims=True); y=np.cross(z,x)
    return np.stack([x,y,z],-1), wr
def yaw(R): x=R[:,:,0]; return np.degrees(np.arctan2(x[:,2],x[:,0]))
srcs={'base':rec['T_world_mesh'].astype(np.float64),'fp':Twc@rec['T_cam_mesh_foundationpose'].astype(np.float64),'cc':Twc@rec['T_cam_mesh_coconet'].astype(np.float64)}
fig,ax=plt.subplots(3,1,figsize=(18,12),sharex=True)
for hi,base in enumerate([21,42]):
    Rh,wr=hand_frames(kp,base); hy=np.degrees(np.arctan2(Rh[:,2,0],Rh[:,0,0]))
    ax[2].plot(hy,label=f'handyaw {base}')
    for k,T in srcs.items():
        d=((yaw(T[:,:3,:3])-hy+180)%360)-180
        ax[hi].plot(d,'.',ms=3,label=k)
for k,T in srcs.items(): ax[2].plot(yaw(T[:,:3,:3]),'.',ms=2,label=k)
for a in ax: a.grid(); a.legend()
ax[0].set_title('objyaw - R hand yaw'); ax[1].set_title('objyaw - L hand yaw')
ax[0].set_xticks(range(0,980,20)); plt.tight_layout(); plt.savefig('/work/scratch/vis/diff.png',dpi=55)
EOF
python diffplot.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/diff.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import numpy as np
m=np.load('/work/scratch/mstats.npz'); a=m['area']; c=m['cen']; b=m['bbox']
for r in [range(118,140),range(196,290,3),range(425,452),range(836,875)]:
  for i in r: print(i, int(a[i]), c[i].round(1), b[i])
  print()
"
```

<details><summary>output</summary>

```
118 2206 [832.  609.8] [805. 587. 860. 634.]
119 2173 [832.7 609.6] [806. 587. 860. 633.]
120 2195 [832.  609.7] [805. 587. 860. 634.]
121 2166 [832.6 609.6] [806. 587. 860. 633.]
122 2213 [831.7 609.8] [805. 587. 859. 634.]
123 2210 [831.4 609.8] [805. 587. 859. 634.]
124 2212 [831.2 609.8] [805. 587. 858. 634.]
125 2214 [830.5 609.8] [804. 587. 858. 634.]
126 2217 [830.2 609.9] [804. 587. 858. 634.]
127 2169 [830.2 609.6] [804. 587. 857. 634.]
128 2199 [829.3 609.6] [803. 587. 856. 634.]
129 2154 [829.4 609.1] [803. 586. 856. 633.]
130 2216 [828.4 608.6] [802. 586. 856. 633.]
131 2190 [828.6 607.8] [802. 585. 856. 632.]
132 2206 [828.2 607. ] [802. 583. 855. 632.]
133 2257 [827.8 605.5] [802. 581. 854. 631.]
134 2325 [826.7 603.8] [800. 579. 853. 630.]
135 2307 [826.2 601.7] [800. 577. 8 …[+4383 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > fitall.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
Vd,Fd=decimated_mesh(V,F); sil=SilRenderer(Vd,Fd)
Tcw=np.linalg.inv(Twc); Tc=Tcw@rec['T_world_mesh'].astype(np.float64)
Pc=np.array([0.004,0.011,0.003])
c=np.load('/work/scratch/crops.npz'); ms=np.load('/work/scratch/mstats.npz')
anc=np.einsum('fij,j->fi',Tc[:,:3,:3],Pc)+Tc[:,:3,3]
Kc=crop_K(K,c['boxes'],128); valid=c['inim']*(1-c['per'])
fr=np.nonzero(ms['area']>300)[0]
out=Tc.copy(); ious=np.full(980,np.nan); ious0=np.full(980,np.nan)
for s in range(0,len(fr),200):
    f=fr[s:s+200]
    R,A,iou=fit_silhouettes(sil,Tc[f,:3,:3],anc[f],Kc[f],c['obj'][f],valid[f],Pc,K,iters=300,w_rot_prior=0.02)
    _,_,iou0=fit_silhouettes(sil,Tc[f,:3,:3],anc[f],Kc[f],c['obj'][f],valid[f],Pc,K,iters=0)
    for k,i in enumerate(f): out[i,:3,:3]=R[k]; out[i,:3,3]=A[k]-R[k]@Pc
    ious[f]=iou; ious0[f]=iou0
np.savez('/work/scratch/fitA.npz',T_world_mesh=Twc@out,iou=ious,iou0=ious0)
print(np.nanmean(ious0),np.nanmean(ious))
EOF
python fitall.py
```

<details><summary>output</summary>

```
0.7281998250151844 0.871082433904853
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > pano.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoilib import *
import cv2, torch, nvdiffrast.torch as dr
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
AXC=np.array([0.004,0.0,0.003])
# per-face attributes (mesh is unshared: 3 verts per face?)
fn=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]); fn/=np.linalg.norm(fn,axis=1,keepdims=True)+1e-12
fc=V[F].mean(1); rad=fc[:,[0,2]]-AXC[[0,2]]; r=np.linalg.norm(rad,axis=1); rad/=r[:,None]+1e-12
radial=(fn[:,0]*rad[:,0]+fn[:,2]*rad[:,1])
y=fc[:,1]; rb=0.089+(y+0.067)/0.156*(0.114-0.089)
body=(np.abs(r-rb)<0.006)&(y>-0.06)&(y<0.085)
outer=(radial>0.5)&body
print('outer faces',outer.sum(),'of',len(F))
# unshare vertices for per-face attributes
Vu=V[F].reshape(-1,3); Fu=np.arange(len(Vu)).reshape(-1,3)
phi=np.arctan2(Vu[:,2]-AXC[2],Vu[:,0]-AXC[0])
attr=np.stack([np.cos(phi),np.sin(phi),Vu[:,1],np.repeat(outer.astype(np.float32),3)],1)
dev='cuda'
glctx=dr.RasterizeCudaContext()
vt=torch.tensor(Vu,dtype=torch.float32,device=dev); ft=torch.tensor(Fu,dtype=torch.int32,device=dev)
at=torch.tensor(attr,dtype=torch.float32,device=dev)
nf=torch.tensor(np.repeat(fn,3,0),dtype=torch.float32,device=dev)
def render_attr(T, x0,y0,S,scale):
    Kc=K.copy(); Kc[0,2]=(K[0,2]-x0+0.5)*scale; Kc[1,2]=(K[1,2]-y0+0.5)*scale; Kc[:2,:2]*=scale
    res=int(S*scale)
    Tt=torch.tensor(T,dtype=torch.float32,device=dev)
    vc=vt@Tt[:3,:3].T+Tt[:3,3]
    fx,fy,cx,cy=Kc[0,0],Kc[1 …[+2647 chars]
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
```
</details>

**Read**
```
/work/scratch/vis/pano1.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat >> pano.py <<'EOF'

def pano_img(gimg, x0, y0, S, T, scale=4, permask=None):
    g=cv2.resize(gimg,(S*scale,S*scale),interpolation=cv2.INTER_CUBIC)
    a,m=render_attr(T,x0,y0,S,scale)
    ok=m&(a[...,3]>0.99)&(a[...,4]>0.35)
    ph=np.degrees(np.arctan2(a[...,1],a[...,0]))%360; h=a[...,2]
    w=ok.astype(np.float32); k=int(6*scale)|1
    mu=cv2.GaussianBlur(g*w,(k,k),0)/(cv2.GaussianBlur(w,(k,k),0)+1e-6)
    hp=g-mu
    pi=(ph[ok]/360*NPHI).astype(int)%NPHI; hi=np.clip(((h[ok]-HB[0])/(HB[1]-HB[0])*NH).astype(int),0,NH-1)
    sm=np.zeros((NH,NPHI)); wt=np.zeros((NH,NPHI))
    np.add.at(sm,(hi,pi),hp[ok]); np.add.at(wt,(hi,pi),1)
    return sm,wt

def texture_pano(dist=2.6, nview=12):
    R=Renderer(V,F,UV,TEX,K,1152,1536)
    from scipy.spatial.transform import Rotation as Rot
    sm=np.zeros((NH,NPHI)); wt=np.zeros((NH,NPHI))
    for k in range(nview):
        T=np.eye(4); T[:3,:3]=Rot.from_euler('x',-10,degrees=True).as_matrix()@Rot.from_euler('y',k*360/nview,degrees=True).as_matrix()
        T[:3,3]=[0,0,dist]; T[:3,3]-=T[:3,:3]@AXC
        p=K@T[:3,3]; u,v=p[:2]/p[2]; S=int(K[0,0]*0.4/p[2])+8; x0=int(u-S/2); y0=int(v-S/2)
        col,mk,_=R.render(T[None],crop=(x0,y0,S,S),scale=1)
        gimg=(col[0].cpu().numpy()@np.array([0.299,0.587,0.114])*255).astype(np.float32)
        s,w=pano_img(gimg,x0,y0,S,T); sm+=s; wt+=w
    return sm,wt
EOF
cat > pano_test2.py <<'EOF'
from pano import *
sm,wt=texture_pano()
np.savez('/work/scratch/texpano.npz',sm=s …[+224 chars]
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
```
</details>

**Read**
```
/work/scratch/vis/texpano.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat >> pano.py <<'EOF'

TXP=None
def corr_shift(sm,wt,sign=-1.0):
    """circular correlation of a frame panorama with the texture panorama; returns score per shift bin s,
    where s means: texture content at phi appears in the frame panorama at phi+s."""
    global TXP
    if TXP is None:
        d=np.load('/work/scratch/texpano.npz'); TXP=np.where(d['wt']>0,d['sm']/np.maximum(d['wt'],1),0)
    P=np.where(wt>0,sm/np.maximum(wt,1),0); W=(wt>0).astype(float)
    P=P-(P*W).sum(1,keepdims=True)/np.maximum(W.sum(1,keepdims=True),1); P*=W
    sc=np.zeros(NPHI)
    for s in range(NPHI):
        Ts=np.roll(TXP,s,axis=1)
        num=(P*Ts).sum(); den=np.sqrt((P**2).sum()*((Ts**2)*W).sum())+1e-9
        sc[s]=sign*num/den
    return sc
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    M[:3,3]=AXC-M[:3,:3]@AXC; return M
EOF
cat > pano_synth.py <<'EOF'
from pano import *
# synthetic: render texture at true pose; unwrap with pose rotated by +d (mesh-frame y rotation)
Rr=Renderer(V,F,UV,TEX,K,1152,1536)
Tc=np.linalg.inv(Twc)@np.load('/work/scratch/fitA.npz')['T_world_mesh']
T=Tc[660]
p=K@T[:3,3]; u,v=p[:2]/p[2]; S=int(K[0,0]*0.4/p[2])+8; x0=int(u-S/2); y0=int(v-S/2)
col,mk,_=Rr.render(T[None],crop=(x0,y0,S,S),scale=1)
gimg=(col[0].cpu().numpy()@np.array([0.299,0.587,0.114])*255).astype(np.float32)
for d in [0,20,60]:
    sm,wt=pano_img(gimg,x0,y0,S,T@yawrot(np.radians(d)))
    sc=corr_shift(sm,wt,sign=1.0)
    print(d,  …[+91 chars]
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
0 best shift deg [  0. 282.  99. 189.] 0.57
20 best shift deg [ 21. 336. 210. 300.] 0.36
60 best shift deg [ 60.  18. 342. 159.] 0.48
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > pano_real.py <<'EOF'
from pano import *
import h5py
src=sys.argv[1]; a,b,st=[int(x) for x in sys.argv[2:5]]; sign=float(sys.argv[5])
T=np.load(src)['T_world_mesh'] if src!='base' else rec['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)@T
P=h5py.File('/work/inputs/masks.h5','r')['person']
def wyaw(Tw): x=Tw[:3,0]; return np.degrees(np.arctan2(x[2],x[0]))
out=[]
for i in range(a,b,st):
    sm,wt=pano(i,Tc[i],P[i])
    sc=corr_shift(sm,wt,sign)
    top=np.argsort(-sc)[:3]
    absy=[(wyaw(Twc@Tc[i]@yawrot(np.radians(-s*360/NPHI))))%360 for s in top]
    out.append(sc)
    print(i, f'baseyaw {wyaw(T[i])%360:6.1f}', 'top shifts',top*3, 'abs yaw', np.round(absy,0), 'sc',sc[top].round(2), 'cov',(wt.sum(0)>0).mean().round(2))
np.save(f'/work/scratch/pscore_{a}_{b}.npy',np.array(out))
EOF
python pano_real.py base 214 436 12 -1
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
214 baseyaw  281.4 top shifts [279  42  45] abs yaw [  2. 240. 237.] sc [0.12 0.1  0.1 ] cov 0.4
226 baseyaw  211.0 top shifts [357 312  96] abs yaw [214. 259. 115.] sc [0.14 0.13 0.11] cov 0.42
238 baseyaw  208.8 top shifts [  3 297  72] abs yaw [206. 272. 137.] sc [0.13 0.09 0.07] cov 0.42
250 baseyaw  208.8 top shifts [297 264 210] abs yaw [272. 305. 359.] sc [0.11 0.11 0.1 ] cov 0.44
262 baseyaw  171.5 top shifts [336  84 339] abs yaw [196.  88. 193.] sc [0.08 0.06 0.06] cov 0.4
274 baseyaw  227.6 top shifts [210  24  21] abs yaw [ 18. 204. 207.] sc [0.12 0.08 0.08] cov 0.42
286 baseyaw  227.6 top shifts [210 120 285] abs yaw [ 18. 108. 303.] sc [0.15 0.12 0.11] cov 0.42
298 baseyaw  227.6 top shifts [210  12 330] abs yaw [ 18. 216. 258.] sc [0.13 0.11 0.1 ]  …[+1086 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; python pano_real.py base 214 436 24 1 | cut -c1-120
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
214 baseyaw  281.4 top shifts [318  36 273] abs yaw [323. 246.   8.] sc [0.14 0.11 0.09] cov 0.4
238 baseyaw  208.8 top shifts [195 303  78] abs yaw [ 14. 266. 131.] sc [0.08 0.08 0.08] cov 0.42
262 baseyaw  171.5 top shifts [ 60 303 180] abs yaw [112. 229. 352.] sc [0.08 0.06 0.06] cov 0.4
286 baseyaw  227.6 top shifts [234 189   0] abs yaw [354.  39. 228.] sc [0.16 0.14 0.14] cov 0.42
310 baseyaw  227.6 top shifts [222  87 132] abs yaw [  6. 141.  96.] sc [0.12 0.09 0.09] cov 0.42
334 baseyaw  227.6 top shifts [222 186   6] abs yaw [  6.  42. 222.] sc [0.12 0.12 0.11] cov 0.43
358 baseyaw  227.6 top shifts [189 141 324] abs yaw [ 39.  87. 264.] sc [0.1  0.1  0.09] cov 0.42
382 baseyaw  227.6 top shifts [  0 219 309] abs yaw [228.   9. 279.] sc [0.09 0.08 0.08]  …[+204 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > pano_all.py <<'EOF'
from pano import *
import h5py, time
src=sys.argv[1]
T=np.load(src)['T_world_mesh'] if src!='base' else rec['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)@T
P=h5py.File('/work/inputs/masks.h5','r')['person']
ms=np.load('/work/scratch/mstats.npz')
def wyaw(Tw): x=Tw[:3,0]; return np.degrees(np.arctan2(x[2],x[0]))
A=np.full((980,NPHI),np.nan); cov=np.zeros(980)
t=time.time()
for i in range(980):
    if ms['area'][i]<300: continue
    sm,wt=pano(i,Tc[i],P[i])
    sc=corr_shift(sm,wt,-1)
    cov[i]=(wt.sum(0)>0).mean()
    for s in range(NPHI):
        ay=wyaw(Twc@Tc[i]@yawrot(np.radians(-s*360/NPHI)))%360
        A[i,int(round(ay/3))%NPHI]=sc[s]
print(time.time()-t)
np.savez(sys.argv[2],A=A,cov=cov)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.figure(figsize=(10,20)); Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
plt.imshow(np.nan_to_num(Z),aspect='auto',extent=[0,360,980,0],cmap='viridis',vmin=-2.5,vmax=2.5)
x=T[:,:3,0]; by=np.degrees(np.arctan2(x[:,2],x[:,0]))%360
plt.plot(by,np.arange(980),'r.',ms=1); plt.yticks(range(0,980,20)); plt.grid(alpha=0.3)
plt.savefig(sys.argv[3],dpi=50)
EOF
python pano_all.py /work/scratch/fitA.npz /work/scratch/pscoreA.npz /work/scratch/vis/pscoreA.png
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
56.6018648147583
/work/scratch/exp/pano_all.py:22: RuntimeWarning: Mean of empty slice
  plt.figure(figsize=(10,20)); Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
  var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
/work/scratch/exp/pano_all.py:22: RuntimeWarning: invalid value encountered in divide
  plt.figure(figsize=(10,20)); Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
```
</details>

**Read**
```
/work/scratch/vis/pscoreA.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import numpy as np
d=np.load('/work/scratch/pscoreA.npz'); A=d['A']
Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
for a,b in [(0,124),(214,436),(560,600),(600,640),(640,680),(680,720),(847,868)]:
  m=np.nanmean(Z[a:b],0); o=np.argsort(-m)
  pk=[k for k in o if m[k]>=m[(k-1)%120] and m[k]>=m[(k+1)%120]][:6]
  print(a,b,'peaks',[(k*3,round(m[k],2)) for k in pk])
"
```

<details><summary>output</summary>

```
<string>:4: RuntimeWarning: Mean of empty slice
/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
  var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
<string>:4: RuntimeWarning: invalid value encountered in divide
0 124 peaks [(222, 1.75), (294, 1.33), (339, 1.24), (249, 0.93), (33, 0.86), (105, 0.78)]
214 436 peaks [(171, 0.79), (213, 0.77), (270, 0.76), (51, 0.7), (276, 0.69), (312, 0.67)]
560 600 peaks [(342, 0.81), (30, 0.8), (267, 0.69), (216, 0.66), (261, 0.6), (126, 0.59)]
600 640 peaks [(246, 0.62), (15, 0.55), (204, 0.54), (324, 0.49), (102, 0.48), (279, 0.45)]
640 680 peaks [(342, 0.75), (117, 0.63), (246, 0.57), (324, 0.53), (315, 0.52), (216, 0.5)]
680 720 peaks [(300, 0.47), (345, 0.42), (243, 0 …[+132 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > pano_avg.py <<'EOF'
from pano import *
import h5py
T=np.load('/work/scratch/fitA.npz')['T_world_mesh']; Tc=np.linalg.inv(Twc)@T
P=h5py.File('/work/inputs/masks.h5','r')['person']
rows=[]
d=np.load('/work/scratch/texpano.npz'); tp=np.where(d['wt']>0,d['sm']/np.maximum(d['wt'],1),0)
def show(im,s=2.0):
    return cv2.resize(np.clip(im*s+128,0,255).astype(np.uint8),(NPHI*8,NH*8),interpolation=cv2.INTER_NEAREST)
rows.append(show(-tp,3))
for a,b,st in [(0,124,4),(280,410,4),(600,604,1),(660,664,1)]:
    S=np.zeros((NH,NPHI)); W=np.zeros((NH,NPHI))
    for i in range(a,b,st):
        sm,wt=pano(i,Tc[i],P[i]); S+=sm; W+=wt
    im=np.where(W>0,S/np.maximum(W,1),0); r=show(im,6); r[np.repeat(np.repeat(W==0,8,0),8,1)]=0
    cv2.putText(r,f'{a}',(3,15),0,0.5,255,1); rows.append(r)
cv2.imwrite('/work/scratch/vis/pano_avg.png',np.vstack(rows))
EOF
python pano_avg.py
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
```
</details>

**Read**
```
/work/scratch/vis/pano_avg.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; python -c "
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
d=np.load('/work/scratch/pscoreA.npz'); A=d['A']
Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
a,b=500,745
T=rec['T_world_mesh']; x=T[:,:3,0]; by=np.degrees(np.arctan2(x[:,2],x[:,0]))%360
fp=Twc@rec['T_cam_mesh_foundationpose']; x=fp[:,:3,0]; fy=np.degrees(np.arctan2(x[:,2],x[:,0]))%360
plt.figure(figsize=(14,14)); plt.imshow(np.nan_to_num(Z[a:b]),aspect='auto',extent=[0,360,b,a],vmin=-2.5,vmax=2.5)
plt.plot(by[a:b],np.arange(a,b),'r.',ms=3); plt.plot(fy[a:b],np.arange(a,b),'w.',ms=2)
plt.yticks(range(a,b,10)); plt.xticks(range(0,361,15)); plt.grid(alpha=0.3)
plt.savefig('/work/scratch/vis/pscore_c2.png',dpi=50)
" 2>/dev/null
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/pscore_c2.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > lugs.py <<'EOF'
from pano import *
T=np.load('/work/scratch/fitA.npz')['T_world_mesh']
def wyaw(Tw): x=Tw[:3,0]; return np.degrees(np.arctan2(x[2],x[0]))
AX=AXC
lug=[AX+np.array([0.114*np.cos(np.radians(a)),0.075,0.114*np.sin(np.radians(a))]) for a in (-60,120)]
for i in [int(x) for x in sys.argv[1].split(',')]:
    Tc=np.linalg.inv(Twc)@T[i]; y0=wyaw(T[i])
    res=[]
    for ay in range(0,360,15):
        d=np.radians(-(ay-y0))  # approx: rotation about mesh y changes world yaw by -angle? check sign
        Tn=Tc@yawrot(d)
        if abs(((wyaw(Twc@Tn)-ay+180)%360)-180)>5: Tn=Tc@yawrot(-d)
        c=Tn[:3,:3]@AX+Tn[:3,3]; uc=K[0,0]*c[0]/c[2]
        fr=[]
        for L in lug:
            p=Tn[:3,:3]@L+Tn[:3,3]; fr.append((K[0,0]*p[0]/p[2]-uc)/(K[0,0]*0.114/c[2]))
        res.append(f'{ay}:{fr[0]:+.1f}/{fr[1]:+.1f}')
    print(i,' '.join(res))
EOF
python lugs.py 640
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
640 0:+0.4/-0.3 15:+0.1/-0.0 30:-0.1/+0.3 45:-0.3/+0.5 60:-0.6/+0.8 75:-0.7/+0.9 90:-0.9/+1.0 105:-0.9/+1.1 120:-0.9/+1.1 135:-0.8/+1.0 150:-0.7/+0.8 165:-0.5/+0.6 180:-0.3/+0.4 195:-0.0/+0.1 210:+0.3/-0.1 225:+0.5/-0.3 240:+0.8/-0.6 255:+0.9/-0.7 270:+1.0/-0.9 285:+1.1/-0.9 300:+1.1/-0.9 315:+1.0/-0.8 330:+0.8/-0.7 345:+0.6/-0.5
```
</details>

**Bash**
```
cd /work/scratch/exp; python lugs.py 60,350,850; python zoom.py 0,100,230,350,420,555,575,595 /work/scratch/vis/z2.jpg 80
```

<details><summary>output</summary>

```
outer faces 31779 of 88020
60 0:+0.4/-0.4 15:+0.1/-0.1 30:-0.1/+0.2 45:-0.4/+0.4 60:-0.6/+0.7 75:-0.8/+0.8 90:-0.9/+1.0 105:-1.0/+1.0 120:-1.0/+1.0 135:-0.9/+0.9 150:-0.8/+0.8 165:-0.6/+0.6 180:-0.4/+0.4 195:-0.1/+0.1 210:+0.2/-0.1 225:+0.4/-0.4 240:+0.7/-0.6 255:+0.8/-0.8 270:+1.0/-0.9 285:+1.0/-1.0 300:+1.0/-1.0 315:+0.9/-0.9 330:+0.8/-0.8 345:+0.6/-0.6
350 0:+0.5/-0.5 15:+0.2/-0.2 30:-0.0/+0.1 45:-0.3/+0.4 60:-0.5/+0.6 75:-0.7/+0.8 90:-0.9/+0.9 105:-1.0/+1.0 120:-1.0/+1.0 135:-1.0/+1.0 150:-0.9/+0.8 165:-0.7/+0.7 180:-0.5/+0.5 195:-0.2/+0.2 210:+0.1/-0.0 225:+0.4/-0.3 240:+0.6/-0.5 255:+0.8/-0.7 270:+0.9/-0.9 285:+1.0/-1.0 300:+1.0/-1.0 315:+1.0/-1.0 330:+0.8/-0.9 345:+0.7/-0.7
850 0:+1.1/-0.4 15:+1.0/-0.3 30:+0.8/-0.1 45:+0.6/+0.1 60:+0.4/+0.4 75:+0.2/+0.6 90:-0.0/+0.8 105:-0.2/+1.0 12 …[+221 chars]
```
</details>

**Read**
```
/work/scratch/vis/z2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > dp_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
T=np.load('/work/scratch/fitA.npz')['T_world_mesh']
d=np.load('/work/scratch/pscoreA.npz'); A=d['A']
Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
AXC=np.array([0.004,0,0.003])
def wyaw(R): x=R[...,:,0]; return np.degrees(np.arctan2(x[...,2],x[...,0]))
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]; M[:3,3]=AXC-M[:3,:3]@AXC; return M
NB=120
a,b=int(sys.argv[1]),int(sys.argv[2]); lo,hi=float(sys.argv[3]),float(sys.argv[4])
lugdir=np.array([np.cos(np.radians(120)),0,np.sin(np.radians(120))])
Tcw=np.linalg.inv(Twc)
data=np.zeros((b-a,NB))
for k,i in enumerate(range(a,b)):
    y0=wyaw(T[i,:3,:3])
    for j in range(NB):
        ay=j*3; dl=np.radians(ay-y0)
        Tn=T[i]@yawrot(-dl)
        if abs(((wyaw(Tn[:3,:3])-ay+180)%360)-180)>4: Tn=T[i]@yawrot(dl)
        Tcn=Tcw@Tn; ray=Tcn[:3,:3]@AXC+Tcn[:3,3]; ray/=np.linalg.norm(ray)
        ld=Tcn[:3,:3]@lugdir; lugc=(ld@ray)**2
        z=Z[i,j] if not np.isnan(Z[i,j]) else 0
        inrange = (lo<=ay<=hi)
        data[k,j]=-z+3.0*lugc+(0 if inrange else 100)
# viterbi
maxd=2; cost=data[0].copy(); bp=np.zeros((b-a,NB),int)
for k in range(1,b-a):
    best=np.full(NB,np.inf); arg=np.zeros(NB,int)
    for dd in range(-maxd,maxd+1):
        c=np.roll(cost,dd)+0.3*dd*dd
        m=c<best; best[m]=c[m];  …[+460 chars]
```

<details><summary>output</summary>

```
/work/scratch/exp/dp_test.py:7: RuntimeWarning: Mean of empty slice
  Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py:1879: RuntimeWarning: Degrees of freedom <= 0 for slice.
  var = nanvar(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
/work/scratch/exp/dp_test.py:7: RuntimeWarning: invalid value encountered in divide
  Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
total -222.1 mean texture z on path 1.46
[309 300 300 306 297 294 288 273 279 291 294 294 294 288 279 291 297 285
 300 306]
/work/scratch/exp/dp_test.py:7: RuntimeWarning: Mean of empty slice
  Z=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py:1879 …[+390 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; sed -i 's/data\[k,j\]=-z+3.0\*lugc/data[k,j]=-z+LW*lugc/; s/^NB=120/NB=120\nLW=float(sys.argv[5]) if len(sys.argv)>5 else 3.0/' dp_test.py; python dp_test.py 545 740 0 360 0 2>/dev/null; python dp_test.py 545 740 0 360 3 2>/dev/null
```

<details><summary>output</summary>

```
total -282.7 mean texture z on path 1.693
[  9  33  30  39  30  24  15   6 345 342 342 345 324 318 321 342 333 318
 309 306]
total -222.1 mean texture z on path 1.46
[309 300 300 306 297 294 288 273 279 291 294 294 294 288 279 291 297 285
 300 306]
```
</details>

**Bash**
```
cd /work/scratch/exp; for r in "0 124" "125 214" "214 436" "436 545"; do echo $r; python dp_test.py $r 240 340 3 2>/dev/null; python dp_test.py $r 60 160 3 2>/dev/null; python dp_test.py $r 0 360 0 2>/dev/null; done
```

<details><summary>output</summary>

```
0 124
total -213.5 mean texture z on path 1.869
[294 297 294 294 297 294 297 297 297 294 297 294 294]
total -122.8 mean texture z on path 1.12
[114 105 102 105 120 117 120 117 117 105 108 108 105]
total -240.3 mean texture z on path 2.027
[219 222 222 222 222 222 222 222 219 219 219 219 222]
125 214
total -70.3 mean texture z on path 1.023
[303 294 294 306 300 303 306 303 306]
total -61.0 mean texture z on path 1.085
[111 102 105  99 120 123 129 117  87]
total -96.5 mean texture z on path 1.341
[237 222 216 222 234 225 255 261 255]
214 436
total -287.8 mean texture z on path 1.667
[291 297 291 291 291 309 309 297 276 276 273 270 270 273 276 273 270 294
 297 294 297 300 300]
total -193.7 mean texture z on path 1.076
[126 129 123 123 126 123 111 111 102 111 114 111 117 129 135 138 114 114
 1 …[+426 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; cat > absvis.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
AXC=np.array([0.004,0,0.003])
def wyaw(R): x=R[:,0]; return np.degrees(np.arctan2(x[2],x[0]))
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]; M[:3,3]=AXC-M[:3,:3]@AXC; return M
i=int(sys.argv[1]); yaws=[float(x) for x in sys.argv[2].split(',')]
T=np.load('/work/scratch/fitA.npz')['T_world_mesh'][i]
S=int(sys.argv[4]) if len(sys.argv)>4 else 80
p=K@(Tcw@T)[:3,3]; u,v=p[:2]/p[2]; x0=int(u-S/2); y0=int(v-S/2)
img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S]
Z=320
tiles=[cv2.resize(img,(Z,Z),interpolation=cv2.INTER_CUBIC)]
for ay in yaws:
    y0w=wyaw(T[:3,:3]); dl=np.radians(ay-y0w); Tn=T@yawrot(-dl)
    if abs(((wyaw(Tn[:3,:3])-ay+180)%360)-180)>4: Tn=T@yawrot(dl)
    col,mk,_=R.render((Tcw@Tn)[None],crop=(x0,y0,S,S),scale=4)
    c=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); m=mk[0].cpu().numpy()
    big=cv2.resize(img,(4*S,4*S),interpolation=cv2.INTER_CUBIC); big[m]=c[m]
    t=cv2.resize(big,(Z,Z),interpolation=cv2.INTER_AREA); cv2.putText(t,f'{ay:.0f}',(5,25),0,0.8,(0,255,255),2); tiles.append(t)
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+3]) for k in range(0, …[+158 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/abs30.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/vis/abs350.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /workspace/v2d_foundation_pose/lib/FoundationPose; grep -n "def \|class " estimater.py | head -40; grep -n "def \|class " learning/training/predict_score.py | head; sed -n 1,40p /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py | grep -n import
```

<details><summary>output</summary>

```
18:class FoundationPose:
19:  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
44:  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
82:  def get_tf_to_centered_mesh(self):
88:  def to_device(self, s='cuda:0'):
106:  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
132:  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
142:  def guess_translation(self, depth, mask, K):
164:  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
248:  def compute_add_err_to_gt_pose(self, poses):
255:  def track_one(self, rgb, depth, K, iteration, extra={}) …[+2490 chars]
```
</details>

**Bash**
```
cd /workspace/v2d_foundation_pose/lib/FoundationPose; sed -n 117,200p learning/training/predict_score.py; sed -n 18,45p estimater.py; sed -n 100,125p /workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py
```

<details><summary>output</summary>

```
class ScorePredictor:
  def __init__(self, amp=True):
    self.amp = amp
    self.run_name = "2024-01-11-20-02-45"

    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    weights_base_dir = os.environ.get("FOUNDATIONPOSE_WEIGHTS_DIR", f'{code_dir}/../../weights')
    ckpt_dir = f'{weights_base_dir}/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{weights_base_dir}/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = np.inf
    if 'c_in' not in  …[+4789 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import h5py
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def p(n,o):
  print(n, getattr(o,'shape',None), getattr(o,'dtype',None))
f.visititems(p)
" | head; ls /weights/foundationpose/nvlabs_pytorch/*; cat /pipeline/depth/moge2/depth_report.json | head -30
```

<details><summary>output</summary>

```
aligned None None
aligned/front_stereo_camera_left (980,) object
alignment None None
alignment/front_stereo_camera_left None None
alignment/front_stereo_camera_left/scale (980,) float32
alignment/front_stereo_camera_left/shift (980,) float32
alignment/front_stereo_camera_left/valid_count (980,) int32
frame_names None None
frame_names/front_stereo_camera_left (980,) object
raw None None
Traceback (most recent call last):
  File "<string>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 771, in visititems
    return h5o.visit(self.id, proxy)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5o.pyx", line …[+944 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import h5py, numpy as np
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
x=f['aligned/front_stereo_camera_left'][0]; print(type(x), len(x), x[:16])
print(f['aligned/front_stereo_camera_left'].attrs.keys(), dict(f.attrs))
for k in f['aligned/front_stereo_camera_left'].attrs: print(k, f['aligned/front_stereo_camera_left'].attrs[k])
"; grep -rn "def .*decode\|imdecode\|ffv1\|exr" /workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py | head
```

<details><summary>output</summary>

```
<class 'numpy.ndarray'> 630914 [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82]
<KeysViewHDF5 []> {'complete': True, 'depth_alignment_input_identity_json': '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-normal","depth_model_revision":"b135031bae30b5ac2ae141a0e68717795ce38340","depth_source_commit":"925b8ed835a7a9cdb7578ba15c658a0afc969030","direct_mhr":{"mtime_ns":1791201009198816476,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000002/mhr_init/episode_000002.pkl","size":4373475},"export":{"mtime_ns":1791200589502818077,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000002/export/episode_000002/wild_export.json","size":2804},"raw_depth":{"mtime_ns":1791200314659819125,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000002/depth/moge2/raw.npy", …[+827 chars]
```
</details>

**Bash**
```
sed -n 140,182p /workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py
```

<details><summary>output</summary>

```
        raw_payload = np.asarray(record.raw_payload, dtype=np.uint8)
        raw_shape = _canonical_depth_shape(raw_payload)
        if raw_shape is None or raw_shape != tuple(np.asarray(record.raw_depth_m).shape):
            raise ValueError(f"Preserved raw depth payload is not canonical or differs from decoded shape: payload={raw_shape}, decoded={np.asarray(record.raw_depth_m).shape}")
    return raw_payload, _encode_metric_depth(record.aligned_depth_m)


def decode_depth_png_uint16(payload: np.ndarray) -> np.ndarray:
    from PIL import Image

    value = np.asarray(payload)
    if value.dtype != np.dtype("uint8") or value.ndim != 1:
        raise TypeError(f"Depth payload must be a one-dimensional uint8 array, got shape={value.shape}, dtype={value.dtype}")
    encoded = value.tobytes( …[+1573 chars]
```
</details>

**Bash**
```
cd /workspace/v2d_foundation_pose/lib/FoundationPose; sed -n 44,82p estimater.py; grep -n "scorer.predict\|refiner.predict" -A3 estimater.py | head -20; sed -n 1,30p estimater.py | grep import
```

<details><summary>output</summary>

```
  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarr …[+2137 chars]
```
</details>

**Bash**
```
mkdir -p /work/scratch/fp; cat > /work/scratch/exp/fpscore.py <<'EOF'
import sys, os
os.environ['FOUNDATIONPOSE_WEIGHTS_DIR']='/weights/foundationpose/nvlabs_pytorch'
sys.path.insert(0,'/workspace/v2d_foundation_pose/lib/FoundationPose')
sys.path.insert(0,'/work/outputs')
import logging
import numpy as np, cv2, h5py, io, torch
from PIL import Image
from estimater import ScorePredictor, make_mesh_tensors, compute_mesh_diameter
import nvdiffrast.torch as dr
from hoilib import load_inputs, load_mesh
logging.disable(logging.INFO)
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
mc=(V.max(0)+V.min(0))/2
gm=g.copy(); gm.vertices=gm.vertices-mc
mt=make_mesh_tensors(gm); diam=compute_mesh_diameter(model_pts=gm.vertices,n_sample=10000)
scorer=ScorePredictor(); glctx=dr.RasterizeCudaContext()
D=h5py.File('/pipeline/depth/moge2/aligned.h5','r')['aligned/front_stereo_camera_left']
AXC=np.array([0.004,0,0.003])
def wyaw(R): x=R[:,0]; return np.degrees(np.arctan2(x[2],x[0]))
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]; M[:3,3]=AXC-M[:3,:3]@AXC; return M
Tw=np.load('/work/scratch/fitA.npz')['T_world_mesh']; Tcw=np.linalg.inv(Twc)
Kp=np.array(cam['K_pipeline'])
Tmc=np.eye(4); Tmc[:3,3]=mc
step=float(sys.argv[2]) if len(sys.argv)>2 else 15
for i in [int(x) for x in sys.argv[1].split(',')]:
    rgb=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[...,::-1].copy()
    dep=np.asarray(Image.open(io.BytesIO(D[i].tobyte …[+1052 chars]
```

<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/exp/fpscore.py", line 16, in <module>
    mt=make_mesh_tensors(gm); diam=compute_mesh_diameter(model_pts=gm.vertices,n_sample=10000)
       ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/v2d_foundation_pose/lib/FoundationPose/Utils.py", line 107, in make_mesh_tensors
    img = np.array(mesh.visual.material.image.convert('RGB'))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'PBRMaterial' object has no attribute 'image'
```
</details>

**Bash**
```
cd /work/scratch/exp; sed -i 's/^gm=g.copy(); gm.vertices=gm.vertices-mc/gm=g.copy(); gm.vertices=gm.vertices-mc; gm.visual=gm.visual.copy(); gm.visual.material=gm.visual.material.to_simple() if hasattr(gm.visual.material,"to_simple") else gm.visual.material/' fpscore.py; python -c "
import trimesh; print(hasattr(trimesh.visual.material.PBRMaterial,'to_simple'))"; python fpscore.py 30,350,580,620,660,700 2>&1 | grep -v Warning | tail -8
```

<details><summary>output</summary>

```
True
  _C._set_default_tensor_type(t)
  with torch.cuda.amp.autocast(enabled=self.amp):
30 top [(210, 138.44), (285, 138.44), (300, 138.44), (0, 138.41), (240, 138.41), (255, 138.41)] sc@290-300 [138.41, 138.44, 138.44, 138.41] sc@90-135 [138.38, 138.38, 138.38, 138.38]
350 top [(15, 140.34), (30, 140.34), (225, 140.34), (210, 140.34), (300, 140.34), (285, 140.34)] sc@290-300 [140.34, 140.34, 140.34, 140.34] sc@90-135 [140.31, 140.31, 140.31, 140.28]
580 top [(300, 138.62), (30, 138.59), (270, 138.56), (75, 138.56), (345, 138.56), (315, 138.56)] sc@290-300 [138.56, 138.53, 138.62, 138.56] sc@90-135 [138.5, 138.47, 138.53, 138.5]
620 top [(120, 146.91), (315, 146.91), (300, 146.91), (285, 146.91), (60, 146.88), (0, 146.88)] sc@290-300 [146.88, 146.91, 146.91, 146.91] sc@90-135 [146.88, 146. …[+384 chars]
```
</details>

> *thinking:* The texture-unwrapping and bail-lug cues consistently point to a world yaw of ~290-300° throughout the episode, unlike the baseline's inconsistent jumps between 65°, 227°, and 280° — and the FoundationPose scorer proved flat on these small crops so I'm skipping it. Now writing `solve.py`.

**Bash**
```
cd /work/scratch/exp; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
T=np.load('/work/scratch/fitA.npz')['T_world_mesh']
kp=rec['human_keypoints_world']
Pc=np.array([0.004,0.011,0.003])
anc=np.einsum('fij,j->fi',T[:,:3,:3],Pc)+T[:,:3,3]
for i in range(690,746,5):
    for b in (21,42):
        h=kp[i,b:b+21]; print(i,b, 'wrist d %.3f'%np.linalg.norm(h[20]-anc[i]), 'handc d %.3f'%np.linalg.norm(h.mean(0)-anc[i]), end=' | ')
    print()
for i in range(840,870,4):
    for b in (21,42):
        h=kp[i,b:b+21]; print(i,b, h.mean(0).round(3), end=' | ')
    print()
EOF
```

<details><summary>output</summary>

```
690 21 wrist d 0.589 handc d 0.586 | 690 42 wrist d 0.220 handc d 0.183 | 
695 21 wrist d 0.571 handc d 0.572 | 695 42 wrist d 0.215 handc d 0.173 | 
700 21 wrist d 0.558 handc d 0.563 | 700 42 wrist d 0.205 handc d 0.139 | 
705 21 wrist d 0.556 handc d 0.562 | 705 42 wrist d 0.202 handc d 0.122 | 
710 21 wrist d 0.534 handc d 0.546 | 710 42 wrist d 0.190 handc d 0.108 | 
715 21 wrist d 0.536 handc d 0.547 | 715 42 wrist d 0.220 handc d 0.161 | 
720 21 wrist d 0.499 handc d 0.515 | 720 42 wrist d 0.150 handc d 0.100 | 
725 21 wrist d 0.578 handc d 0.594 | 725 42 wrist d 0.195 handc d 0.126 | 
730 21 wrist d 0.567 handc d 0.585 | 730 42 wrist d 0.164 handc d 0.089 | 
735 21 wrist d 0.582 handc d 0.601 | 735 42 wrist d 0.158 handc d 0.096 | 
740 21 wrist d 0.616 handc d 0.637 | 740 42 wrist  …[+570 chars]
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
s=open('solve.py').read()
old="""    # ---------------- pass 1: per-frame silhouette fit from the baseline ----------------
    fr = np.nonzero(visible)[0]
    T1_c = Tb_c.copy()"""
new="""    # ---------------- pass 1: per-frame silhouette fit from the baseline ----------------
    # static phases: start upright (support = floor / table parallel to the floor) with the baseline's median yaw,
    # at the mask centre with the baseline's median depth (the baseline tilt in S3 is up to 80 deg off)
    T0_c = Tb_c.copy()
    anc0 = anc_b.copy()
    for nm, a, b, kind in PHASES:
        if kind != "static":
            continue
        ym = np.degrees(np.angle(np.exp(1j * np.radians(wyaw(Tb_w[a:b + 1, :3, :3]))).mean()))
        Rw0 = rot_from_axis_yaw(n_up, ym)
        zm = np.median(anc_b[a:b + 1, 2])
        for i in range(a, b + 1):
            T0_c[i, :3, :3] = Tcw[:3, :3] @ Rw0
            if area[i] > 200:
                c = (bbox[i, :2] + bbox[i, 2:]) / 2
                anc0[i] = np.array([(c[0] - K[0, 2]) / K[0, 0], (c[1] - K[1, 2]) / K[1, 1], 1.0]) * zm
            else:
                anc0[i] = anc_b[i] / anc_b[i, 2] * zm
    fr = np.nonzero(visible)[0]
    T1_c = Tb_c.copy()"""
assert old in s; s=s.replace(old,new)
s=s.replace("""        R, A, iou = fit_silhouettes(sil, Tb_c[f, :3, :3], anc_b[f], Kc[f], obj[f], valid[f], PC, K, iters=300,
                                    w_rot_prior=0.02, res=RES)
        _, _, iou0 = fit_silhoue …[+2410 chars]
```

<details><summary>output</summary>

```
500:    depth_scale = {}
508:        depth_scale[nm] = float(s)
533:    log["depth_rescale_per_phase"] = depth_scale
534:    T_w = Twc @ T_c
```
</details>

**Bash**
```
python solve.py --inputs /work/inputs --out /work/scratch/run1.npz 2>&1 | grep -v Warn | tail -20
```

<details><summary>output</summary>

```
[pass1] mean IoU baseline 0.728 -> 0.869  (36s)
[yaw] texture scores for 738 frames (95s)
[yaw] decoded yaw per phase: [('S1_table_static', 294.0), ('C1_carry', 266.0), ('S2_floor_static', 276.0), ('C2_carry_handover', 294.9), ('S3_floor_static_occluded', 288.0)]
[fit] final mean IoU (visible) 0.837  (124s)
[done] wrote /work/scratch/run1.npz  (124s)
```
</details>

**Bash**
```
cd /work/scratch/exp; python plotyaw.py /work/scratch/run1.npz 2>/dev/null; python overlay.py /work/scratch/run1.npz 0,170,200,350,470,520,600,660,700,730,760,800,850,900,960 /work/scratch/vis/ov_run1.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/yawplot.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/vis/ov_run1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* Found two issues: the basket stays tilted 40° during occluded frames 743–846 due to extrapolation from unreliable sliver masks, and partially-occluded frames are weighted too heavily. I'll fix this by adding a per-frame reliability score (visible vs. expected silhouette area) and anchoring each carry phase's tilt to upright at the start and end, since the basket sits on flat surfaces there.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
old="""    T1_w = Twc @ T1_c
"""
new="""    T1_w = Twc @ T1_c
    # reliability: visible object-mask area relative to the unoccluded rendered silhouette area of the pass-1 pose
    exp_area = np.zeros(F)
    with torch.no_grad():
        for s in range(0, len(fr), 200):
            f = fr[s:s + 200]
            Rt = torch.tensor(T1_c[f, :3, :3], dtype=torch.float32, device="cuda")
            tt = torch.tensor(T1_c[f, :3, 3], dtype=torch.float32, device="cuda")
            sl, _ = sil.render(Rt, tt, torch.tensor(Kc[f], dtype=torch.float32, device="cuda"), RES)
            exp_area[f] = sl.sum((1, 2)).cpu().numpy() * (boxes[f, 2] / RES) ** 2
    vis_frac = np.where(exp_area > 0, area / np.maximum(exp_area, 1), 0)
    reliable = visible & (vis_frac > 0.6) & (iou1 > 0.6)
    print(f"[pass1] reliable frames: {reliable.sum()} / visible {visible.sum()}")
"""
assert old in s; s=s.replace(old,new,1)
old=s[s.index("    axis_s = np.tile(n_up, (F, 1))"):s.index("    for i in range(F):\n        R_w[i] = rot_from_axis_yaw")]
new='''    axis_s = np.tile(n_up, (F, 1))
    for k, (_, a, b, kind) in enumerate(PHASES):
        if kind != "carry":
            continue
        idx = np.arange(a, b + 1)
        ax = axis1[idx].copy()
        # reliable frames only; reject axes tilted more than 45 deg from up (flips)
        good = reliable[idx] & (ax @ n_up > np.cos(np.radians(45)))
        # the basket is picked up from / put down on a …[+730 chars]
```

<details><summary>output</summary>

```
[pass1] reliable frames: 690 / visible 758
[pass1] mean IoU baseline 0.728 -> 0.870  (46s)
[yaw] texture scores for 740 frames (120s)
[yaw] decoded yaw per phase: [('S1_table_static', 294.0), ('C1_carry', 267.5), ('S2_floor_static', 252.0), ('C2_carry_handover', 274.1), ('S3_floor_static_occluded', 291.0)]
[fit] final mean IoU (visible) 0.836  (142s)
[done] wrote /work/scratch/run2.npz  (142s)
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
old="""    np.savez(args.out, T_world_mesh=T_w.astype(np.float64))"""
new="""    np.savez(args.out, T_world_mesh=T_w.astype(np.float64))
    np.savez(os.path.join(os.path.dirname(args.out), "solve_diag.npz"), data=data, Ztex=Ztex, has_tex=has_tex,
             yaw_dp=yaw_dp, yaw_final=yaw_final, iou_base=iou_base, iou1=iou1, final_iou=final_iou,
             reliable=reliable, vis_frac=vis_frac, T1_w=T1_w, axis_s=axis_s)"""
s=s.replace(old,new)
open('solve.py','w').write(s)
EOF
python solve.py --inputs /work/inputs --out /work/scratch/run2.npz 2>&1 | grep -v Warn | tail -3; cp solve_diag.npz /work/scratch/diag2.npz
python - <<'EOF'
import numpy as np
d=np.load('/work/scratch/diag2.npz'); data=d['data']; Z=d['Ztex']; ht=d['has_tex']; rel=d['reliable']
for nm,a,b in [('S1',0,124),('S2',214,436),('S3',847,979)]:
    tot=data[a:b+1].sum(0); tz=Z[a:b+1][ht[a:b+1]].mean(0)
    o=np.argsort(tot)[:8]
    print(nm,'best total',[(int(k*3),round(tot[k],2)) for k in o])
    print('   tex mean z at those',[round(tz[k],2) for k in o])
EOF
```

<details><summary>output</summary>

```
Exit code 1
[yaw] decoded yaw per phase: [('S1_table_static', 294.0), ('C1_carry', 268.9), ('S2_floor_static', 276.0), ('C2_carry_handover', 293.0), ('S3_floor_static_occluded', 288.0)]
[fit] final mean IoU (visible) 0.833  (128s)
[done] wrote /work/scratch/run2.npz  (128s)
cp: cannot stat 'solve_diag.npz': No such file or directory
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/work/scratch/diag2.npz'
```
</details>

**Bash**
```
cp /work/scratch/solve_diag.npz /work/scratch/diag2.npz; python - <<'EOF'
import numpy as np
d=np.load('/work/scratch/diag2.npz'); data=d['data']; Z=d['Ztex']; ht=d['has_tex']; rel=d['reliable']
for nm,a,b in [('S1',0,124),('C1',125,213),('S2',214,436),('C2',437,742),('S3',847,979)]:
    tot=data[a:b+1].sum(0); tz=Z[a:b+1][ht[a:b+1]].mean(0)
    o=np.argsort(tot)[:8]
    print(nm,'best total',[(int(k*3),round(tot[k],2)) for k in o])
    print('   tex mean z at those',[round(tz[k],2) for k in o])
print(d['yaw_dp'][::10])
EOF
```

<details><summary>output</summary>

```
S1 best total [(294, -14.66), (105, -10.16), (297, -9.44), (117, -7.48), (108, -6.62), (300, -6.02), (120, -5.41), (123, -4.13)]
   tex mean z at those [1.66, 0.83, 1.15, 0.59, 0.46, 0.83, 0.44, 0.37]
C1 best total [(267, -48.81), (273, -45.84), (264, -41.63), (237, -36.25), (309, -35.22), (303, -31.55), (183, -31.29), (228, -28.78)]
   tex mean z at those [0.36, 0.32, 0.28, 0.26, 0.23, 0.17, 0.38, 0.19]
S2 best total [(315, -9.54), (312, -7.66), (318, -5.61), (276, -4.26), (309, -3.97), (273, -3.63), (129, -3.49), (126, -2.44)]
   tex mean z at those [1.21, 0.93, 0.92, 0.65, 0.49, 0.68, 0.44, 0.31]
C2 best total [(279, -44.72), (294, -41.64), (288, -27.5), (282, -22.16), (291, -12.64), (297, -11.15), (306, -9.56), (285, -7.66)]
   tex mean z at those [0.15, 0.08, 0.02, 0.04, -0.03, -0.0,  …[+705 chars]
```
</details>

> *thinking:* The yaw decode is unstable—nvdiffrast's non-deterministic atomics make S2's near-tie evidence (spread across ~270–315°) flip between runs under hard Viterbi argmax. I'm switching to forward–backward posteriors and using the circular mean of the dominant mode, which minimizes expected error and varies smoothly with inputs.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
a=s.index("    # Viterbi\n"); b=s.index("    yaw_final = yaw_dp.copy()")
new='''    # Forward-backward over yaw (costs act as negative log-likelihoods). The posterior per frame is summarised by the
    # circular mean of its dominant mode (+-60 deg around the MAP bin): minimum expected error, and stable against
    # near-ties (a hard Viterbi argmax jumped by 20-40 deg between near-identical runs in static phase S2).
    def trans_list(i):
        static_step = PHASES[ph[i]][3] == "static" and ph[i] == ph[i - 1]
        md = 0 if static_step else MAXD
        return [(dd, W_TRANS * dd * dd) for dd in range(-md, md + 1)]

    def lse(x, axis=0):
        m = x.max(axis)
        return m + np.log(np.exp(x - np.expand_dims(m, axis)).sum(axis))

    la = np.zeros((F, NB))
    la[0] = -data[0]
    la[0] -= lse(la[0])
    for i in range(1, F):
        tl = trans_list(i)
        st = np.stack([np.roll(la[i - 1], dd) - c for dd, c in tl])
        la[i] = lse(st) - data[i]
        la[i] -= lse(la[i])
    lb = np.zeros((F, NB))
    for i in range(F - 2, -1, -1):
        tl = trans_list(i + 1)
        st = np.stack([np.roll(lb[i + 1] - data[i + 1], -dd) - c for dd, c in tl])
        lb[i] = lse(st)
        lb[i] -= lse(lb[i])
    post = la + lb
    post = np.exp(post - post.max(1, keepdims=True))
    post /= post.sum(1, keepdims=True)
    yaw_dp = np.zeros(F)
    for i in range(F):
        jm = int(np.argmax(post[i]))
        d …[+956 chars]
```

<details><summary>output</summary>

```
62:W_LUG = 3.0               # weight of the lug-side cost (cos^2 of the angle lug-axis / viewing ray)
65:W_TRANS = 0.3             # quadratic transition cost per bin^2
409:    # Forward-backward over yaw (costs act as negative log-likelihoods). The posterior per frame is summarised by the
[yaw] texture scores for 740 frames (105s)
[yaw] decoded yaw per phase: [('S1_table_static', 294.3), ('C1_carry', 270.2), ('S2_floor_static', 276.7), ('C2_carry_handover', 293.7), ('S3_floor_static_occluded', 283.0)]
[fit] final mean IoU (visible) 0.834  (120s)
[done] wrote /work/scratch/run3.npz  (120s)
{
 "yaw_posterior_mass_in_mode_per_phase": {
  "S1_table_static": 0.9999999999987773,
  "C1_carry": 0.9999828304013644,
  "S2_floor_static": 0.9999581813217115,
  "C2_carry_handover": 0.9857453749701462 …[+482 chars]
```
</details>

**Bash**
```
cp /work/scratch/solve_diag.npz /work/scratch/diag3.npz; cat > /work/scratch/exp/fb.py <<'EOF'
import numpy as np, sys
d=np.load(sys.argv[1]); data=d['data']/float(sys.argv[2])
F,NB=data.shape; yaws=np.arange(NB)*3.0
PH=[(0,124,'s'),(125,213,'c'),(214,436,'s'),(437,846,'c'),(847,979,'s')]
ph=np.zeros(F,int)
for k,(a,b,_) in enumerate(PH): ph[a:b+1]=k
def tl(i):
    md=0 if (PH[ph[i]][2]=='s' and ph[i]==ph[i-1]) else 2
    return [(dd,0.3*dd*dd/float(sys.argv[2])) for dd in range(-md,md+1)]
def lse(x,axis=0):
    m=x.max(axis); return m+np.log(np.exp(x-np.expand_dims(m,axis)).sum(axis))
la=np.zeros((F,NB)); la[0]=-data[0]; la[0]-=lse(la[0])
for i in range(1,F):
    st=np.stack([np.roll(la[i-1],dd)-c for dd,c in tl(i)]); la[i]=lse(st)-data[i]; la[i]-=lse(la[i])
lb=np.zeros((F,NB))
for i in range(F-2,-1,-1):
    st=np.stack([np.roll(lb[i+1]-data[i+1],-dd)-c for dd,c in tl(i+1)]); lb[i]=lse(st); lb[i]-=lse(lb[i])
p=la+lb; p=np.exp(p-p.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
for i in [60,170,300,600,700,800,900]:
    o=np.argsort(-p[i])[:5]; print(i,[(int(yaws[k]),round(p[i][k],2)) for k in o])
EOF
for t in 1 3 10; do echo T=$t; python /work/scratch/exp/fb.py /work/scratch/diag3.npz $t; done
```

<details><summary>output</summary>

```
T=1
60 [(294, 0.9), (297, 0.08), (300, 0.02), (282, 0.0), (291, 0.0)]
170 [(270, 0.18), (273, 0.17), (258, 0.12), (255, 0.11), (264, 0.1)]
300 [(276, 0.62), (273, 0.33), (315, 0.04), (312, 0.01), (285, 0.0)]
600 [(288, 0.67), (285, 0.25), (291, 0.05), (282, 0.01), (294, 0.01)]
700 [(300, 0.36), (297, 0.29), (294, 0.15), (291, 0.05), (285, 0.03)]
800 [(285, 0.04), (282, 0.04), (288, 0.04), (279, 0.04), (291, 0.04)]
900 [(282, 0.04), (291, 0.04), (288, 0.04), (270, 0.04), (279, 0.03)]
T=3
60 [(294, 0.64), (297, 0.23), (300, 0.11), (282, 0.01), (291, 0.01)]
170 [(258, 0.08), (270, 0.08), (261, 0.07), (273, 0.07), (264, 0.07)]
300 [(276, 0.25), (273, 0.18), (315, 0.14), (312, 0.09), (288, 0.05)]
600 [(288, 0.31), (291, 0.17), (285, 0.17), (294, 0.09), (297, 0.09)]
700 [(297, 0.13), (294, 0.12) …[+670 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace("STATIC_TOTAL_W = 10.0     # total data weight of one static phase (frames are near-identical images)\n",
"STATIC_TOTAL_W = 10.0     # total data weight of one static phase (frames are near-identical images)\nTEMP = 3.0                # posterior temperature: per-frame evidence is strongly correlated between frames\n")
s=s.replace("""    def trans_list(i):""","""    data = data / TEMP

    def trans_list(i):""")
s=s.replace("return [(dd, W_TRANS * dd * dd) for dd in range(-md, md + 1)]","return [(dd, W_TRANS * dd * dd / TEMP) for dd in range(-md, md + 1)]")
open('solve.py','w').write(s)
EOF
for r in a b; do python solve.py --inputs /work/inputs --out /work/scratch/run4$r.npz 2>&1 | grep "decoded"; done; python -c "
import numpy as np, sys; sys.path.insert(0,'/work/outputs'); from hoilib import rot_angle_deg
a=np.load('/work/scratch/run4a.npz')['T_world_mesh']; b=np.load('/work/scratch/run4b.npz')['T_world_mesh']
print('max trans diff cm', np.abs(a[:,:3,3]-b[:,:3,3]).max()*100, 'max rot diff deg', rot_angle_deg(a[:,:3,:3],b[:,:3,:3]).max())"
```

<details><summary>output</summary>

```
[yaw] decoded yaw per phase: [('S1_table_static', 294.5), ('C1_carry', 277.0), ('S2_floor_static', 287.5), ('C2_carry_handover', 292.1), ('S3_floor_static_occluded', 279.3)]
[yaw] decoded yaw per phase: [('S1_table_static', 294.7), ('C1_carry', 280.5), ('S2_floor_static', 292.4), ('C2_carry_handover', 292.0), ('S3_floor_static_occluded', 285.8)]
max trans diff cm 2.622275103586136 max rot diff deg 6.588526413837428
```
</details>

**Bash**
```
python -c "
import numpy as np, sys; sys.path.insert(0,'/work/outputs'); from hoilib import rot_angle_deg
a=np.load('/work/scratch/run4a.npz')['T_world_mesh']; b=np.load('/work/scratch/run4b.npz')['T_world_mesh']
dt=np.linalg.norm(a[:,:3,3]-b[:,:3,3],axis=1)*100; dr=rot_angle_deg(a[:,:3,:3],b[:,:3,:3])
for s in range(0,980,40): print(s, dt[s:s+40].max().round(3), dr[s:s+40].max().round(2))
"
```

<details><summary>output</summary>

```
0 0.032 0.23
40 0.032 0.23
80 0.032 0.23
120 0.26 1.78
160 1.945 6.38
200 0.563 6.59
240 0.157 4.94
280 0.157 4.94
320 0.157 4.94
360 0.157 4.94
400 0.216 4.94
440 0.699 6.08
480 0.395 3.05
520 0.406 1.94
560 0.216 2.04
600 0.413 1.86
640 1.019 3.36
680 2.314 4.36
720 2.715 5.26
760 0.207 2.83
800 0.157 1.96
840 0.216 6.45
880 0.216 6.45
920 0.216 6.45
960 0.216 6.45
```
</details>

> *thinking:* Determinism is still failing (runs diverge up to 6.6° and 2.7 cm) due to nvdiffrast's non-deterministic atomic gradient accumulation. I'm switching to a Levenberg–Marquardt fitter using deterministic forward-only finite differences with supersampling, followed by temporal smoothing.

**Bash**
```
cat >> /work/outputs/hoilib.py <<'EOF'


class CovRenderer:
    """Deterministic (forward-only) silhouette coverage renderer: supersampled rasterisation + average pooling."""

    def __init__(self, verts, faces, ss=4, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(verts, dtype=torch.float32, device=device)
        self.f = torch.tensor(faces, dtype=torch.int32, device=device)
        self.ss = ss

    @torch.no_grad()
    def render(self, R, t, Kc, res):
        ss = self.ss
        r2 = res * ss
        vc = torch.einsum("bij,nj->bni", R, self.v) + t[:, None, :]
        near, far = 0.05, 20.0
        fx, fy = Kc[:, 0, 0] * ss, Kc[:, 1, 1] * ss
        # crop pixel j centre <-> crop coordinate j+0.5 (see crop_K); supersampled pixel centres accordingly
        cx, cy = Kc[:, 0, 2] * ss, Kc[:, 1, 2] * ss
        x, y, z = vc[..., 0], vc[..., 1], vc[..., 2]
        cxn = (2 * fx[:, None] * x + (2 * cx[:, None] - r2) * z) / r2
        cyn = (-2 * fy[:, None] * y - (2 * cy[:, None] - r2) * z) / r2
        czn = (far + near) / (far - near) * z - 2 * far * near / (far - near)
        clip = torch.stack([cxn, cyn, czn, z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, resolution=[r2, r2])
        m = (rast[..., 3] > 0).float().flip(1)
        return torch.nn.functional.avg_pool2d(m[:, None], ss)[:, 0]


def fit_lm(cov, R0, anchor0, Kc, obj, valid, Pc, K, iters=12, fit_rot …[+6752 chars]
```

<details><summary>output</summary>

```
5.252392530441284 0.6784452 [[0.13811, 0.20963, 2.81578], [0.23292, 0.28105, 2.60766], [0.33408, 0.30686, 2.52757]] [0.97367284 0.03574558 0.22512863]
4.6948771476745605 0.6784452 [[0.13811, 0.20963, 2.81578], [0.23292, 0.28105, 2.60766], [0.33408, 0.30686, 2.52757]] [0.97367284 0.03574558 0.22512863]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoilib.py').read()
old="""    # per-frame normalisation (fixed): visible object + initial silhouette area"""
new="""    # coarse init of the image offset: match centroids of visible object mask and visible silhouette (2 rounds)
    if init_centroid:
        for _ in range(2):
            for b in range(0, F, batch):
                idx = np.arange(b, min(F, b + batch))
                par = P[np.zeros(len(idx), int)] if static else P[idx]
                _, s = residuals(par, idx)
                vm = torch.tensor(valid[idx], dtype=torch.float32, device=dev)
                o = torch.tensor(obj[idx], dtype=torch.float32, device=dev)
                yy, xx = torch.meshgrid(torch.arange(res, device=dev, dtype=torch.float32),
                                        torch.arange(res, device=dev, dtype=torch.float32), indexing="ij")
                ws, wo = (s * vm).sum((1, 2)), (o * vm).sum((1, 2))
                ok = ((ws > 10) & (wo > 10)).cpu().numpy()
                dx = ((o * vm * xx).sum((1, 2)) / wo.clamp_min(1) - (s * vm * xx).sum((1, 2)) / ws.clamp_min(1)).cpu().numpy()
                dy = ((o * vm * yy).sum((1, 2)) / wo.clamp_min(1) - (s * vm * yy).sum((1, 2)) / ws.clamp_min(1)).cpu().numpy()
                scl = Kc[idx, 0, 0] / K[0, 0]                    # crop px per image px
                dpx = np.stack([dx, dy], 1) / scl[:, None] * ok[:, None]
                if static:
                    P[0, -2:] += dpx[ok].mean(0) …[+589 chars]
```

<details><summary>output</summary>

```
1.5960578918457031 0.910561 [[0.13826, 0.20495, 2.81578], [0.23324, 0.28247, 2.60766], [0.33666, 0.31938, 2.52757]] [0.96387961 0.02581716 0.26508328]
1.3914604187011719 0.910561 [[0.13826, 0.20495, 2.81578], [0.23324, 0.28247, 2.60766], [0.33666, 0.31938, 2.52757]] [0.96387961 0.02581716 0.26508328]
13.282833814620972 0.9131155 [[0.1383, 0.20494, 2.81578], [0.23324, 0.28247, 2.60766], [0.33659, 0.31945, 2.52757]] [0.96387993 0.02585033 0.26507888]
16.375144958496094 0.9131155 [[0.1383, 0.20494, 2.81578], [0.23324, 0.28247, 2.60766], [0.33659, 0.31945, 2.52757]] [0.96387993 0.02585033 0.26507888]
```
</details>

**Agent:** LM fitter is deterministic (bit-identical across runs), 3× faster, and gets better IoU (0.91 vs 0.88 on 600–740). Wiring it into `solve.py`.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoilib.py').read()
old=s[s.index("    if init_centroid:\n"):s.index("    # per-frame normalisation (fixed)")]
new='''    if init_centroid:
        yy, xx = torch.meshgrid(torch.arange(res, device=dev, dtype=torch.float32),
                                torch.arange(res, device=dev, dtype=torch.float32), indexing="ij")
        for _ in range(2):
            dpx_all = np.zeros((F, 2))
            ok_all = np.zeros(F, bool)
            for b in range(0, F, batch):
                idx = np.arange(b, min(F, b + batch))
                par = P[np.zeros(len(idx), int)] if static else P[idx]
                _, s = residuals(par, idx)
                vm = torch.tensor(valid[idx], dtype=torch.float32, device=dev)
                o = torch.tensor(obj[idx], dtype=torch.float32, device=dev)
                ws, wo = (s * vm).sum((1, 2)), (o * vm).sum((1, 2))
                ok_all[idx] = ((ws > 10) & (wo > 10)).cpu().numpy()
                dx = ((o * vm * xx).sum((1, 2)) / wo.clamp_min(1) - (s * vm * xx).sum((1, 2)) / ws.clamp_min(1))
                dy = ((o * vm * yy).sum((1, 2)) / wo.clamp_min(1) - (s * vm * yy).sum((1, 2)) / ws.clamp_min(1))
                scl = Kc[idx, 0, 0] / K[0, 0]                    # crop px per image px
                dpx_all[idx] = np.stack([dx.cpu().numpy(), dy.cpu().numpy()], 1) / scl[:, None]
            if static:
                if ok_all.any():
                    P[0, -2:] += dpx_all[ok_all].mean(0)
        …[+2870 chars]
```

<details><summary>output</summary>

```
[pass1] reliable frames: 695 / visible 758
[pass1] mean IoU baseline 0.729 -> 0.906  (43s)
[yaw] texture scores for 738 frames (91s)
[yaw] decoded yaw per phase: [('S1_table_static', 292.4), ('C1_carry', 273.0), ('S2_floor_static', 274.8), ('C2_carry_handover', 288.9), ('S3_floor_static_occluded', 288.5)]
[fit] final mean IoU (visible) 0.881  (108s)
[done] wrote /work/scratch/run5a.npz  (108s)
[pass1] reliable frames: 695 / visible 758
[pass1] mean IoU baseline 0.729 -> 0.906  (26s)
[yaw] texture scores for 738 frames (76s)
[yaw] decoded yaw per phase: [('S1_table_static', 292.4), ('C1_carry', 273.0), ('S2_floor_static', 274.8), ('C2_carry_handover', 288.9), ('S3_floor_static_occluded', 288.5)]
[fit] final mean IoU (visible) 0.881  (87s)
[done] wrote /work/scratch/run5b.npz  (87s)
max abs  …[+8 chars]
```
</details>

> *thinking:* The run is now bit-for-bit deterministic, with mean silhouette IoU improving from 0.729 to 0.881 after rotation constraints. Next I'll write a validation script covering per-phase IoU, texture/lug yaw evidence, smoothness, floor contact, hand distance, and depth cues versus baseline.

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
cd /work/outputs; cp /work/scratch/solve_diag.npz /work/scratch/diag5.npz; python validate.py --result /work/scratch/run5a.npz --diag /work/scratch/diag5.npz 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
| phase | frames | vis | IoU base | IoU ours | centroid err px base/ours | texture z base/ours | lug cos^2 base/ours | median dR deg/frame base/ours | median acc cm/f^2 base/ours | floor height cm base/ours | tilt vs floor deg base/ours | hand dist cm base/ours | dz mean/median cm (ours-base) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1_table_static | 0-124 | 125 | 0.85 | 0.92 | 3.27/0.43 | -0.45/1.57 | 0.51/0.00 | 0.13/0.00 | 0.04/0.00 | 81.82/81.90 | 0.84/0.00 | 78.97/78.10 | -0.41/0.00 |
| C1_carry | 125-213 | 86 | 0.75 | 0.85 | 2.64/0.79 | -0.05/0.15 | -/- | 6.67/1.11 | 2.64/0.19 | - | 8.14/14.76 | 15.88/14.63 | -0.92/-0.53 |
| S2_floor_static | 214-436 | 223 | 0.80 | 0.84 | 3.75/0.66 | 0.13/-0.85 | 0.77/0.14 | 0.00/0.00 | 0.10/0.00 | 1.08/0.05 | 2.59/0.00 | 61.44/6 …[+1930 chars]
```
</details>

**Bash**
```
python - <<'EOF'
import numpy as np
d=np.load('/work/scratch/diag5.npz'); data=d['data']; Z=d['Ztex']; ht=d['has_tex']; post=d['post']
for nm,a,b in [('S1',0,124),('S2',214,436),('C2v',545,742)]:
    tot=data[a:b+1].sum(0); tz=Z[a:b+1][ht[a:b+1]].mean(0)
    o=np.argsort(tot)[:8]
    print(nm,'best total',[(int(k*3),round(tot[k],2)) for k in o])
    print('   tex z top',[(int(k*3),round(tz[k],2)) for k in np.argsort(-tz)[:8]])
    print('   post', [(int(k*3),round(post[(a+b)//2][k],2)) for k in np.argsort(-post[(a+b)//2])[:6]])
EOF
```

<details><summary>output</summary>

```
Exit code 1
S1 best total [(291, -4.54), (102, -3.8), (105, -3.72), (294, -3.46), (297, -3.35), (117, -2.31), (111, -2.02), (120, -0.91)]
   tex z top [(219, 1.88), (291, 1.57), (339, 1.5), (246, 1.45), (336, 1.37), (294, 1.24), (168, 1.22), (297, 1.21)]
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
IndexError: index 62 is out of bounds for axis 0 with size 5
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace("""    post = np.arange(o1 + 1, o1 + 6)
    off1 = (anc_w[post] - hand[post]).mean(0)""","""    after = np.arange(o1 + 1, o1 + 6)
    off1 = (anc_w[after] - hand[after]).mean(0)""")
open('solve.py','w').write(s)
EOF
python - <<'EOF'
import numpy as np
d=np.load('/work/scratch/diag5.npz'); data=d['data']; Z=d['Ztex']; ht=d['has_tex']
for nm,a,b in [('S1',0,124),('S2',214,436),('C2v',545,742)]:
    tot=data[a:b+1].sum(0); tz=Z[a:b+1][ht[a:b+1]].mean(0)
    o=np.argsort(tot)[:8]
    print(nm,'best total',[(int(k*3),round(tot[k],2)) for k in o])
    print('   tex z top',[(int(k*3),round(tz[k],2)) for k in np.argsort(-tz)[:8]], 'z@270-300',[round(tz[k],2) for k in range(88,102,2)])
EOF
```

<details><summary>output</summary>

```
S1 best total [(291, -4.54), (102, -3.8), (105, -3.72), (294, -3.46), (297, -3.35), (117, -2.31), (111, -2.02), (120, -0.91)]
   tex z top [(219, 1.88), (291, 1.57), (339, 1.5), (246, 1.45), (336, 1.37), (294, 1.24), (168, 1.22), (297, 1.21)] z@270-300 [0.11, -0.58, -0.05, 0.23, -0.02, 1.24, -0.03]
S2 best total [(297, -1.86), (288, -1.78), (306, -1.72), (285, -0.84), (273, -0.73), (294, -0.67), (309, -0.52), (138, -0.39)]
   tex z top [(210, 1.3), (213, 1.0), (72, 0.93), (3, 0.84), (258, 0.84), (207, 0.82), (0, 0.73), (216, 0.73)] z@270-300 [-0.2, 0.2, -0.85, -0.27, 0.51, 0.14, -0.01]
C2v best total [(282, -17.55), (294, -12.68), (291, -6.56), (114, -6.12), (279, -3.8), (102, -3.34), (285, -1.54), (105, 1.0)]
   tex z top [(204, 0.35), (12, 0.31), (57, 0.29), (282, 0.28), (159, 0.27), (14 …[+90 chars]
```
</details>

**Bash**
```
cd /work/scratch/exp; python - <<'EOF' 2>/dev/null
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=np.load('/work/scratch/diag5.npz'); Z=d['Ztex'].copy(); Z[~d['has_tex']]=0
A=np.load('/work/scratch/pscoreA.npz')['A']; ZA=(A-np.nanmean(A,1,keepdims=True))/np.nanstd(A,1,keepdims=True)
fig,ax=plt.subplots(1,2,figsize=(16,16))
for a,M,t in [(ax[0],np.nan_to_num(ZA),'adam fitA'),(ax[1],Z,'LM run5')]:
    a.imshow(M[:760],aspect='auto',extent=[0,360,760,0],vmin=-2.5,vmax=2.5); a.set_title(t); a.set_yticks(range(0,760,20))
    a.plot(d['yaw_final'][:760],np.arange(760),'r.',ms=1)
plt.savefig('/work/scratch/vis/ztex_cmp.png',dpi=45)
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/ztex_cmp.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
old=s[s.index("        row = np.full(NB, np.nan)\n"):s.index("        Ztex[i] = (row - row.mean())")]
new='''        ays = np.array([wyaw((T1_w[i] @ yawrot(np.radians(-s_ * 3.0)))[:3, :3]) % 360 for s_ in range(NB)])
        o = np.argsort(ays)
        row = np.interp(yaws, ays[o], sc[o], period=360)       # scores on the absolute-yaw grid
        from scipy.ndimage import gaussian_filter1d
        row = gaussian_filter1d(row, 1.0, mode="wrap")         # remove 3-deg binning comb
'''
s=s.replace(old,new)
# yaws defined later; move definition up
s=s.replace("""    Ztex = np.zeros((F, NB))""","""    yaws = np.arange(NB) * 3.0
    Ztex = np.zeros((F, NB))""")
open('solve.py','w').write(s)
EOF
python solve.py --inputs /work/inputs --out /work/scratch/run6.npz 2>&1 | grep -v Warn | grep "\["; cp /work/scratch/solve_diag.npz /work/scratch/diag6.npz
python - <<'EOF'
import numpy as np
d=np.load('/work/scratch/diag6.npz'); data=d['data']; Z=d['Ztex']; ht=d['has_tex']; post=d['post']
for nm,a,b in [('S1',0,124),('S2',214,436),('S3',847,979)]:
    tot=data[a:b+1].sum(0); tz=Z[a:b+1][ht[a:b+1]].mean(0)
    o=np.argsort(tot)[:6]
    print(nm,'best total',[(int(k*3),round(tot[k],2)) for k in o])
    print('   tex z top',[(int(k*3),round(tz[k],2)) for k in np.argsort(-tz)[:8]])
    m=(a+b)//2; print('   post', [(int(k*3),round(post[m][k],2)) for k in np.argsort(-post[m])[:6]])
EOF
```

<details><summary>output</summary>

```
[pass1] reliable frames: 695 / visible 758
[pass1] mean IoU baseline 0.729 -> 0.906  (57s)
[yaw] texture scores for 738 frames (117s)
[yaw] decoded yaw per phase: [('S1_table_static', 293.8), ('C1_carry', 270.4), ('S2_floor_static', 267.3), ('C2_carry_handover', 288.9), ('S3_floor_static_occluded', 291.2)]
[fit] final mean IoU (visible) 0.881  (127s)
[done] wrote /work/scratch/run6.npz  (127s)
S1 best total [(294, -6.28), (291, -4.59), (297, -4.32), (105, -4.19), (102, -3.8), (108, -2.55)]
   tex z top [(294, 2.08), (336, 1.7), (291, 1.58), (339, 1.51), (297, 1.5), (219, 1.36), (12, 1.32), (171, 1.31)]
   post [(294, 0.76), (291, 0.15), (297, 0.09), (288, 0.0), (300, 0.0), (279, 0.0)]
S2 best total [(288, -1.69), (297, -1.63), (294, -1.53), (291, -1.52), (306, -1.19), (300, -1.16)]
   tex  …[+670 chars]
```
</details>

> *thinking:* Yaw results are stable at ~290° for S1, C2, S3, and ~267° for S2 (with S2 genuinely ambiguous in the 258–297° range, which I'll note). Now checking depth cues for the final occluded rest (S3): the basket sits 7.7 cm below the floor and 38 cm from the releasing hand.

**Bash**
```
cd /work/outputs; python - <<'EOF' 2>&1 | grep -v Warn
import numpy as np, json, sys
sys.path.insert(0,'.')
from hoilib import *
import solve as S
K,Twc,rec,cam=load_inputs('/work/inputs'); Tcw=np.linalg.inv(Twc)
T=np.load('/work/scratch/run6.npz')['T_world_mesh']; Tb=rec['T_world_mesh']
kp=rec['human_keypoints_world'].astype(np.float64)
plane=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n,d=plane[:3],plane[3]
if n[1]>0: n,d=-n,-d
g,V,F,UV,TEX=load_mesh('/work/inputs')
for a,nm in [(214,'S2'),(847,'S3')]:
    Tw=T[a]; anc=Tw[:3,:3]@S.PC+Tw[:3,3]; cc=Twc[:3,3]; ray=(anc-cc)/np.linalg.norm(anc-cc)
    pts=(Tw[:3,:3]@V[::5].T).T+Tw[:3,3]; h0=(pts@n+d).min()
    print(nm,'floor cue shift cm', -h0/(ray@n)*100)
    for b in (21,42):
        H=kp[a:a+6,b:b+21].mean(1).mean(0); W=kp[a:a+6,b+20].mean(0)
        ss=np.linspace(-0.5,0.5,1001); dist=np.linalg.norm(anc[None]+ss[:,None]*ray[None]-H[None],axis=1)
        print('  hand',b,'dist now %.1f cm'%(np.linalg.norm(anc-H)*100),'closest-approach shift %.1f cm (dist %.1f)'%(ss[dist.argmin()]*100,dist.min()*100),
              ' shift for dist=12cm:', [round(s*100,1) for s in ss[np.abs(dist-0.12)<0.002]][:3], [round(s*100,1) for s in ss[np.abs(dist-0.12)<0.002]][-3:])
# what about the hand-object distance during visible C2 at last frames, and S2 pickup
for i in [440,445,700,720,740]:
    Tw=T[i]; anc=Tw[:3,:3]@S.PC+Tw[:3,3]
    print(i,[round(np.linalg.norm(anc-kp[i,b:b+21].mean(0))*100,1) for b in (21,42)])
EOF
```

<details><summary>output</summary>

```
S2 floor cue shift cm 0.10302476903006545
  hand 21 dist now 10.8 cm closest-approach shift -4.8 cm (dist 9.7)  shift for dist=12cm: [-12.1, -12.0, -11.9] [2.3, 2.4, 2.5]
  hand 42 dist now 86.7 cm closest-approach shift -50.0 cm (dist 69.4)  shift for dist=12cm: [] []
S3 floor cue shift cm -18.77670533449792
  hand 21 dist now 106.8 cm closest-approach shift -50.0 cm (dist 61.2)  shift for dist=12cm: [] []
  hand 42 dist now 39.6 cm closest-approach shift -37.7 cm (dist 12.2)  shift for dist=12cm: [-38.7, -38.6, -38.5] [-36.9, -36.8, -36.7]
440 [78.1, 11.2]
445 [74.9, 12.8]
700 [57.7, 16.4]
720 [55.7, 13.8]
740 [60.5, 13.9]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF' 2>&1 | grep -v Warn
import numpy as np, sys, torch
sys.path.insert(0,'.')
from hoilib import *
import solve as S
K,Twc,rec,cam=load_inputs('/work/inputs'); Tcw=np.linalg.inv(Twc)
T=np.load('/work/scratch/run6.npz')['T_world_mesh']
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
ms=np.load('/work/scratch/mstats.npz')
for i in [214,300,400,848,852,856,860,864]:
    Tc=Tcw@T[i]
    _,mk,_=R.render(Tc[None]); mk=mk[0].cpu().numpy()
    ys,xs=np.nonzero(mk); bb=ms['bbox'][i]
    print(i,'render y %d-%d (h %d) x %d-%d'%(ys.min(),ys.max()+1,ys.max()+1-ys.min(),xs.min(),xs.max()+1),' mask y %d-%d (h %d) x %d-%d'%(bb[1],bb[3],bb[3]-bb[1],bb[0],bb[2]), 'z %.3f'%Tc[2,3])
EOF
```

<details><summary>output</summary>

```
214 render y 743-789 (h 46) x 717-769  mask y 739-786 (h 47) x 720-768 z 3.898
300 render y 743-789 (h 46) x 717-769  mask y 741-791 (h 50) x 716-766 z 3.898
400 render y 743-789 (h 46) x 717-769  mask y 740-791 (h 51) x 716-767 z 3.898
848 render y 820-882 (h 62) x 891-958  mask y 814-877 (h 63) x 900-915 z 3.137
852 render y 820-882 (h 62) x 891-958  mask y 819-876 (h 57) x 896-914 z 3.137
856 render y 820-882 (h 62) x 891-958  mask y 820-876 (h 56) x 895-913 z 3.137
860 render y 820-882 (h 62) x 891-958  mask y 821-876 (h 55) x 895-912 z 3.137
864 render y 820-882 (h 62) x 891-958  mask y 821-875 (h 54) x 895-912 z 3.137
```
</details>

> *thinking:* S3's depth cues conflict (floor, hand, and mask estimates all differ beyond the 2 cm agreement threshold), so I'm keeping its baseline depth and reporting all three cues, while S2's floor and hand cues agree and confirm its baseline. Next I'll check overlays and GPU memory.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace('''    log["runtime_s"] = time.time() - t_start''','''    log["runtime_s"] = time.time() - t_start
    log["torch_max_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9''')
open('solve.py','w').write(s)
EOF
(python solve.py --inputs /work/inputs --out /work/scratch/run7.npz > /work/scratch/run7.log 2>&1 &) ; for k in 1 2 3 4 5 6; do sleep 20; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; done; sleep 30; tail -3 /work/scratch/run7.log; cat /work/scratch/solve_log.json | tail -3
```

<details><summary>output</summary>

```
11903, 2210 MiB
11903, 4150 MiB
11903, 4150 MiB
11903, 4148 MiB
[yaw] decoded yaw per phase: [('S1_table_static', 293.8), ('C1_carry', 270.4), ('S2_floor_static', 267.3), ('C2_carry_handover', 288.9), ('S3_floor_static_occluded', 291.2)]
[fit] final mean IoU (visible) 0.881  (85s)
[done] wrote /work/scratch/run7.npz  (85s)
 "runtime_s": 84.93564701080322,
 "torch_max_mem_GB": 2.143529472
}
```
</details>

**Bash**
```
cd /work/scratch/exp; python overlay.py /work/scratch/run7.npz 60,150,190,300,450,500,560,640,720,742,790,830,850,920,979 /work/scratch/vis/ov_run7.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis/ov_run7.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Agent:** Overlays look right across all phases. Running the final `solve.py` to the official output path and then the validation.

**Bash**
```
cd /work; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | grep "\["; python -c "
import numpy as np; d=np.load('/work/outputs/object_poses_world.npz'); T=d['T_world_mesh']; print(T.shape,T.dtype,np.isfinite(T).all(), np.abs(T[:,3]-[0,0,0,1]).max(), np.abs(np.einsum('fji,fjk->fik',T[:,:3,:3],T[:,:3,:3])-np.eye(3)).max())
r=np.load('/work/scratch/run7.npz')['T_world_mesh']; print('same as run7:', np.abs(T-r).max())"; cd /work/outputs && python validate.py 2>&1 | grep -v Warn > /work/outputs/validation.md; cat /work/outputs/validation.md; cat solve_log.json
```

<details><summary>output</summary>

```
[pass1] reliable frames: 695 / visible 758
[pass1] mean IoU baseline 0.729 -> 0.906  (23s)
[yaw] texture scores for 738 frames (71s)
[yaw] decoded yaw per phase: [('S1_table_static', 293.8), ('C1_carry', 270.4), ('S2_floor_static', 267.3), ('C2_carry_handover', 288.9), ('S3_floor_static_occluded', 291.2)]
[fit] final mean IoU (visible) 0.881  (79s)
[done] wrote /work/outputs/object_poses_world.npz  (79s)

real	1m21.058s
user	1m24.632s
sys	0m10.805s
(980, 4, 4) float64 True 0.0 6.661338147750939e-16
same as run7: 0.0
| phase | frames | vis | IoU base | IoU ours | centroid err px base/ours | texture z base/ours | lug cos^2 base/ours | median dR deg/frame base/ours | median acc cm/f^2 base/ours | floor height cm base/ours | tilt vs floor deg base/ours | hand dist cm base/ours | dz mean/median …[+3273 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF' 2>&1 | grep -v Warn
import numpy as np, sys
sys.path.insert(0,'.')
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs'); Tcw=np.linalg.inv(Twc)
Tb=Tcw@rec['T_world_mesh']; Tr=Tcw@np.load('object_poses_world.npz')['T_world_mesh']
for nm,a,b in [('C2 visible part',437,742),('C2 occluded carry',743,846)]:
    dz=(Tr[a:b+1,2,3]-Tb[a:b+1,2,3])*100; print(nm, 'dz mean %.2f median %.2f'%(dz.mean(),np.median(dz)), 'base z med %.2f ours %.2f'%(np.median(Tb[a:b+1,2,3]),np.median(Tr[a:b+1,2,3])))
EOF
python -c "import torch, numpy, scipy, cv2, trimesh, open3d, h5py, nvdiffrast; print('torch',torch.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__,'opencv',cv2.__version__,'trimesh',trimesh.__version__,'open3d',open3d.__version__,'h5py',h5py.__version__,'nvdiffrast',nvdiffrast.__version__)"; python -c "import matplotlib, PIL, omegaconf; print(matplotlib.__version__, PIL.__version__)" 2>&1 | tail -1; ls /work/tools; ls /work/outputs
```

<details><summary>output</summary>

```
C2 visible part dz mean 1.03 median 0.98 base z med 2.96 ours 2.96
C2 occluded carry dz mean -81.54 median -54.39 base z med 3.69 ours 3.09
torch 2.5.1+cu124 numpy 1.26.3 scipy 1.16.3 opencv 4.11.0 trimesh 5.1.1 open3d 0.20.0 h5py 3.16.0 nvdiffrast 0.4.0
3.11.2 10.2.0
__pycache__
hoilib.py
object_poses_world.npz
solve.py
solve_diag.npz
solve_log.json
validate.py
validation.md
```
</details>

**Agent:** Final output is written and verified: 980×4×4 float64, all rotations orthonormal, identical to the previous run. Writing REPORT.md and TOOLS.md.

**Write**
```
/work/outputs/REPORT.md
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/REPORT.md (file state is current in your context — no need to Read it back)
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

**Bash**
```
rm -rf /work/outputs/__pycache__; ls -la /work/outputs
```

<details><summary>output</summary>

```
total 3176
drwxrwxr-x 2 hanzhou hanzhou    4096 Oct 21 10:01 .
drwxrwxr-x 8 hanzhou hanzhou    4096 Oct 21 08:59 ..
-rw-rw-r-- 1 hanzhou hanzhou   12110 Oct 21 10:01 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou    1533 Oct 21 10:01 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou   21093 Oct 21 09:47 hoilib.py
-rw-rw-r-- 1 hanzhou hanzhou  125718 Oct 21 09:59 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou   28023 Oct 21 09:55 solve.py
-rw-rw-r-- 1 hanzhou hanzhou 3023560 Oct 21 09:59 solve_diag.npz
-rw-rw-r-- 1 hanzhou hanzhou     717 Oct 21 09:59 solve_log.json
-rw-rw-r-- 1 hanzhou hanzhou    8480 Oct 21 09:51 validate.py
-rw-rw-r-- 1 hanzhou hanzhou    2834 Oct 21 10:00 validation.md
```
</details>

**Agent:** I've written `object_poses_world.npz` (980 frames, float64) plus `REPORT.md` and `TOOLS.md`. Position is better than the baseline in every phase; the biggest open risk is the rotation of the basket around its own axis (yaw), which is mostly settled by indirect evidence.

`solve.py` runs offline in about 80 s, peaks at about 4.1 GB of GPU memory, and gives identical output on repeated runs.

**What the video shows:**
- 0–124: basket on the table.
- 125–213: carried and put on the floor.
- 214–436: resting on the floor.
- 437–846: picked up, handed to the other hand, carried. It is hidden in front of the body from 743.
- 847–979: resting on the floor behind the person's legs. Only a thin sliver is visible, in 847–867.

**What was wrong with the baseline, and what I changed:**
- **Image alignment.** The baseline was 15–30 px off the mask at 640–735, which is 6–9 cm sideways. I refit the image position against the object mask, ignoring pixels covered by the person. Mean mask overlap (IoU) on visible frames goes from 0.73 to 0.88; it reaches 0.89 for 437–846, up from 0.64.
- **Lost tracking.** At 743–846 the baseline puts the object 0.5–1.9 m too far away and upside down. I attached it to the reconstructed hand that was holding it.
- **Yaw.** The basket has 8 identical bunnies around it, and the baseline flips between them: 65° on the table and several values within the floor rest. I unwrapped the video onto the basket's wall, correlated it with the mesh texture, and added the position of the handle's attachment points, which I checked by eye. I then decoded one smooth yaw track for the whole episode. It comes out at about 290° almost throughout.
- **Static phases** now have one constant, upright pose; the baseline's tilt in the final rest reached 80°.
- **Depth** is re-anchored per phase to the baseline's median, as instructed. Medians match within 1 cm, except the hidden carry.

**Where it could go wrong:**
- **Table phase yaw.** I chose 294°, which is 131° from the baseline. The main alternative is about 105°, the same view turned 180°. FoundationPose's own first registration gave that alternative. If it is the truth, this phase gets worse than the baseline.
- **Floor rest (214–436) yaw** is weakly determined, anywhere in 258–315°; the output is 267°. Its depth is well confirmed: the basket bottom sits 0.1 cm from the floor plane.
- **Final rest (847–979) depth** is left at the baseline even though the basket ends up 7.7 cm below the floor. The independent cues disagree: floor contact says 19 cm closer, the hand says 38 cm closer, and the visible size says slightly farther. That fails the rule that two cues must agree within about 2 cm.
- **Hidden carry (743–846)** rotation is a smooth interpolation, so confidence is low.

FoundationPose's scorer gave flat scores on these small crops, so I didn't use it.

Files are in `/work/outputs` (`hoilib.py` is a helper that `solve.py` needs):
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoilib.py`
- `validate.py`
- `validation.md`

## Final

I've written `object_poses_world.npz` (980 frames, float64) plus `REPORT.md` and `TOOLS.md`. Position is better than the baseline in every phase; the biggest open risk is the rotation of the basket around its own axis (yaw), which is mostly settled by indirect evidence.

`solve.py` runs offline in about 80 s, peaks at about 4.1 GB of GPU memory, and gives identical output on repeated runs.

**What the video shows:**
- 0–124: basket on the table.
- 125–213: carried and put on the floor.
- 214–436: resting on the floor.
- 437–846: picked up, handed to the other hand, carried. It is hidden in front of the body from 743.
- 847–979: resting on the floor behind the person's legs. Only a thin sliver is visible, in 847–867.

**What was wrong with the baseline, and what I changed:**
- **Image alignment.** The baseline was 15–30 px off the mask at 640–735, which is 6–9 cm sideways. I refit the image position against the object mask, ignoring pixels covered by the person. Mean mask overlap (IoU) on visible frames goes from 0.73 to 0.88; it reaches 0.89 for 437–846, up from 0.64.
- **Lost tracking.** At 743–846 the baseline puts the object 0.5–1.9 m too far away and upside down. I attached it to the reconstructed hand that was holding it.
- **Yaw.** The basket has 8 identical bunnies around it, and the baseline flips between them: 65° on the table and several values within the floor rest. I unwrapped the video onto the basket's wall, correlated it with the mesh texture, and added the position of the handle's attachment points, which I checked by eye. I then decoded one smooth yaw track for the whole episode. It comes out at about 290° almost throughout.
- **Static phases** now have one constant, upright pose; the baseline's tilt in the final rest reached 80°.
- **Depth** is re-anchored per phase to the baseline's median, as instructed. Medians match within 1 cm, except the hidden carry.

**Where it could go wrong:**
- **Table phase yaw.** I chose 294°, which is 131° from the baseline. The main alternative is about 105°, the same view turned 180°. FoundationPose's own first registration gave that alternative. If it is the truth, this phase gets worse than the baseline.
- **Floor rest (214–436) yaw** is weakly determined, anywhere in 258–315°; the output is 267°. Its depth is well confirmed: the basket bottom sits 0.1 cm from the floor plane.
- **Final rest (847–979) depth** is left at the baseline even though the basket ends up 7.7 cm below the floor. The independent cues disagree: floor contact says 19 cm closer, the hand says 38 cm closer, and the visible size says slightly farther. That fails the rule that two cues must agree within about 2 cm.
- **Hidden carry (743–846)** rotation is a smooth interpolation, so confidence is low.

FoundationPose's scorer gave flat scores on these small crops, so I didn't use it.

Files are in `/work/outputs` (`hoilib.py` is a helper that `solve.py` needs):
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoilib.py`
- `validate.py`
- `validation.md`
