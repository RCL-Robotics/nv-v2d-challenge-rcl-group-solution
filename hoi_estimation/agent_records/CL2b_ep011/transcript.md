# CL2b_ep011

Claude Opus v2（ep11-21）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T09:57:22Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude/projects/-work/memory/ 2>/dev/null && cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null; nvidia-smi; ls /pipeline /weights /workspace /code/reconstruction
```

<details><summary>output</summary>

```
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 13:51 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 13:51 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 13:51 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 13:51 proxy.sock

inputs:
total 93764
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 13:51 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 13:51 ..
-rw-rw-r-- 1 hanzhou hanzhou      822 Oct 21 13:51 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       70 Oct 21 13:51 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   193877 Oct 21 13:51 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      395 Oct 21 13:51 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  6116444 Oct 21 13:51 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 10541188 Oct 21 13:51 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 4589 …[+4673 chars]
```
</details>

**Bash**
```
cd /work; echo $CUDA_VISIBLE_DEVICES; cat logs/run_manifest.txt; python -c "
import numpy as np, h5py, trimesh
r=np.load('inputs/reconstruction.npz'); 
for k in r: print(k, r[k].shape, r[k].dtype)
f=h5py.File('inputs/masks.h5'); print({k:f[k].shape for k in f})
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry(); print(m, m.bounds, m.extents)
print(trimesh.bounds.oriented_bounds(m))
"; cat /pipeline/pipeline_report.json | head -100; find /pipeline -maxdepth 3 | head -50
```

<details><summary>output</summary>

```
0
date_utc: 2026-10-21T09:57:22Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2b_ep011:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000011:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
45f7d8c8d17cc21190f7a0cdd3b90a98c330420689d6a19915acf56bf4e46960  ./camera.json
ee19e46c20c013dcb9b14898d966f1ca417181f8515e2381dded25333ddde474  ./episode.json
27f695d586d79e2213a2ff3b3bd077b4f4b5b5a64774492137e4e …[+6395 chars]
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
30.0 901.0
901
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
0800.jpg
0850.jpg
0900.jpg
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import cv2,numpy as np
ims=[cv2.resize(cv2.imread(f'frames/{i:04d}.jpg'),(384,288)) for i in range(0,901,50)]
ims+= [np.zeros_like(ims[0])]*(20-len(ims))
rows=[np.hstack(ims[r*5:(r+1)*5]) for r in range(4)]
g=np.vstack(rows)
for k in range(19): cv2.putText(g,str(k*50),((k%5)*384+5,(k//5)*288+25),0,0.8,(0,0,255),2)
cv2.imwrite('grid.jpg',g)
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/grid.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np, h5py, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
f=h5py.File('/work/inputs/masks.h5')
om=f['object']
from scipy.spatial.transform import Rotation as R
cp=r['contact_prob_right_left']
prev=None
for i in range(0,901,10):
  m=om[i]; a=m.sum(); ys,xs=np.nonzero(m)
  bb=(xs.min(),ys.min(),xs.max(),ys.max()) if a else None
  rot=R.from_matrix(Tc[i,:3,:3])
  d = np.degrees((R.from_matrix(T[i,:3,:3])*R.from_matrix(T[i-10,:3,:3]).inv()).magnitude()) if i>=10 else 0
  print(i,a,bb,np.round(T[i,:3,3],3),np.round(Tc[i,2,3],3),'drot10=%.1f'%d, np.round(cp[i],2))
"

```

<details><summary>output</summary>

```
0 1929 (768, 598, 851, 626) [0.375 0.284 3.108] 3.144 drot10=0.0 [0. 0.]
10 1944 (768, 598, 852, 626) [0.375 0.284 3.109] 3.145 drot10=0.0 [0. 0.]
20 1972 (768, 598, 853, 626) [0.375 0.284 3.108] 3.144 drot10=0.0 [0. 0.]
30 1966 (768, 598, 853, 626) [0.374 0.283 3.102] 3.138 drot10=0.0 [0. 0.]
40 1964 (768, 598, 853, 626) [0.373 0.282 3.085] 3.121 drot10=0.0 [0. 0.]
50 1871 (769, 598, 852, 626) [0.372 0.282 3.084] 3.12 drot10=0.0 [0. 0.]
60 1957 (768, 598, 852, 626) [0.372 0.282 3.081] 3.117 drot10=0.0 [0. 0.]
70 1811 (776, 598, 852, 626) [0.387 0.29  3.132] 3.169 drot10=5.1 [1. 1.]
80 1823 (776, 596, 852, 626) [0.391 0.29  3.143] 3.18 drot10=1.6 [1. 1.]
90 2026 (777, 590, 850, 625) [0.383 0.272 3.11 ] 3.147 drot10=6.6 [1. 1.]
100 2175 (772, 580, 843, 620) [0.355 0.241 3.139] 3.174 drot10= …[+6217 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > /work/scratch/crops.py <<'EOF'
import cv2,numpy as np,h5py,sys
f=h5py.File('/work/inputs/masks.h5'); om=f['object']; pm=f['person']
frames=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
cap=cv2.VideoCapture('/work/inputs/video.mp4')
out=[]; i=0; last=(640,700)
want=set(frames)
while i<=max(frames):
    ok,fr=cap.read()
    if i in want:
        m=om[i]; ys,xs=np.nonzero(m)
        if len(xs): last=(int(xs.mean()),int(ys.mean()))
        cx,cy=last; s=100
        x0=max(cx-s,0); y0=max(cy-s,0)
        c=fr[y0:y0+2*s, x0:x0+2*s].copy()
        mm=m[y0:y0+2*s, x0:x0+2*s]
        cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        c2=cv2.resize(c,(400,400),interpolation=cv2.INTER_CUBIC)
        cnt=[x*2 for x in cnt]
        cv2.drawContours(c2,cnt,-1,(0,255,0),1)
        cv2.putText(c2,str(i),(5,25),0,0.8,(0,0,255),2)
        out.append(c2)
    i+=1
while len(out)%6: out.append(np.zeros_like(out[0]))
g=np.vstack([np.hstack(out[r*6:(r+1)*6]) for r in range(len(out)//6)])
cv2.imwrite(sys.argv[4],g)
EOF
python crops.py 0 240 10 c1.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/c1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python crops.py 240 480 10 c2.jpg; python crops.py 480 720 10 c3.jpg; python crops.py 720 901 8 c4.jpg
```

**Read**
```
/work/scratch/c2.jpg
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

> *thinking:* The video shows a repeated table-floor-table-floor-table cycle, with segments mapped out so far (rest, carried, floor, carried, table). I'll check the remaining portion now.

**Read**
```
/work/scratch/c3.jpg
```

**Read**
```
/work/scratch/c4.jpg
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
cd /work/scratch; python -c "
import trimesh,numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(s.geometry)
m=s.to_geometry()
print(type(m.visual), getattr(m.visual,'uv',None) is not None)
mat=m.visual.material; print(type(mat))
img=getattr(mat,'baseColorTexture',None) or getattr(mat,'image',None); print(img.size if img else None)
print(m.vertices.mean(0))
"; python -c "import nvdiffrast, torch; print(torch.__version__, torch.cuda.is_available())"; pip list 2>/dev/null | grep -i -E "torch|opencv|trimesh|open3d|kornia|scipy|pytorch3d|roma|lietorch|h5py|nvdiff"
```

<details><summary>output</summary>

```
OrderedDict([('lightbluebook01', <trimesh.Trimesh(vertices.shape=(271275, 3), faces.shape=(90566, 3))>)])
<class 'trimesh.visual.texture.TextureVisuals'> True
<class 'trimesh.visual.material.PBRMaterial'>
(2168, 2332)
[ 7.13411626e-05 -1.25369364e-03 -1.28963237e-03]
2.5.1+cu124 True
h5py                                     3.16.0
kornia                                   0.8.3
kornia_rs                                0.2.0
nvdiffrast                               0.4.0
open3d                                   0.20.0
opencv-contrib-python                    4.11.0.86
opencv-python                            4.11.0.86
pytorch-lightning                        2.6.6
pytorch3d                                0.7.9
roma                                     1.6.1
scipy                               …[+313 chars]
```
</details>

**Bash**
```
mkdir -p /work/outputs; cat > /work/outputs/hoi_lib.py <<'EOF'
"""Shared helpers: data loading, nvdiffrast textured rendering in image crops."""
import json, numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr

DEV = 'cuda'


def load_inputs(inp):
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K'], np.float64)
    Twc = np.array(cam['T_world_cam'], np.float64)
    rec = np.load(f'{inp}/reconstruction.npz')
    return K, Twc, rec


def load_video(path, scale=1.0):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(f[:, :, ::-1].copy())  # RGB
    return frames


class MeshRenderer:
    def __init__(self, glb):
        m = trimesh.load(glb, force='scene', process=False).to_geometry()
        self.v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=DEV)
        self.f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=DEV)
        uv = np.asarray(m.visual.uv, np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=DEV)
        tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'), np.float32) / 255.0
        self.tex = torch.tensor(tex, device=DEV)[None].contiguous()
        self.verts_np = np.asarray(m.vertices)
        self.ctx = dr.RasterizeCudaContex …[+3208 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/ov1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > iou.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
K,Twc,rec=load_inputs('/work/inputs')
cam=json.load(open('/work/inputs/camera.json')); Kp=np.array(cam['K_pipeline'])
R=MeshRenderer('/work/inputs/object_mesh.glb')
F=901
f=h5py.File('/work/inputs/masks.h5'); OM=f['object'][:]; PM=f['person'][:]
srcs={'base':(mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64)),K),
      'fp':(rec['T_cam_mesh_foundationpose'].astype(np.float64),Kp),
      'cc':(rec['T_cam_mesh_coconet'].astype(np.float64),Kp)}
res={k:np.zeros(F) for k in srcs}
for i in range(F):
    ys,xs=np.nonzero(OM[i])
    if len(xs)==0:
        for k in res: res[k][i]=np.nan
        continue
    cx,cy=xs.mean(),ys.mean()
    crop=(int(cx)-128,int(cy)-96,256,192)
    x0,y0=crop[0],crop[1]
    om=np.zeros((192,256),bool); pm=np.zeros((192,256),bool)
    sl=(slice(max(y0,0),min(y0+192,1152)),slice(max(x0,0),min(x0+256,1536)))
    dsl=(slice(sl[0].start-y0,sl[0].stop-y0),slice(sl[1].start-x0,sl[1].stop-x0))
    om[dsl]=OM[i][sl]; pm[dsl]=PM[i][sl]
    valid=~(pm&~om)
    for k,(T,KK) in srcs.items():
        _,m,_=R.render(T[i:i+1],KK,crop,(256,192),with_tex=False)
        m=m[0].cpu().numpy()&valid
        res[k][i]=(m&om).sum()/max((m|om).sum(),1)
np.savez('iou_src.npz',**res)
for a in range(0,F,30):
    print(a, ' '.join('%s=%.2f'%(k,np.nanmean(v[a:a+30])) for k,v in res.items()))
print('all', ' '.join('%s=%.3f'%(k,np.nanmean(v)) for k,v in res.items()))
EOF
python iou.py
```

<details><summary>output</summary>

```
0 base=0.78 fp=0.82 cc=0.79
30 base=0.79 fp=0.81 cc=0.79
60 base=0.83 fp=0.84 cc=0.83
90 base=0.78 fp=0.81 cc=0.80
120 base=0.38 fp=0.47 cc=0.39
150 base=0.58 fp=0.70 cc=0.58
180 base=0.82 fp=0.81 cc=0.82
210 base=0.87 fp=0.84 cc=0.86
240 base=0.88 fp=0.85 cc=0.87
270 base=0.84 fp=0.85 cc=0.84
300 base=0.69 fp=0.76 cc=0.68
330 base=0.47 fp=0.48 cc=0.47
360 base=0.73 fp=0.75 cc=0.75
390 base=0.83 fp=0.85 cc=0.84
420 base=0.81 fp=0.85 cc=0.83
450 base=0.74 fp=0.78 cc=0.75
480 base=0.78 fp=0.76 cc=0.80
510 base=0.92 fp=0.93 cc=0.95
540 base=0.80 fp=0.84 cc=0.82
570 base=0.25 fp=0.46 cc=0.24
600 base=0.70 fp=0.70 cc=0.70
630 base=0.82 fp=0.78 cc=0.81
660 base=0.87 fp=0.84 cc=0.86
690 base=0.95 fp=0.91 cc=0.94
720 base=0.95 fp=0.91 cc=0.93
750 base=0.90 fp=0.88 cc=0.89
780 base=0.81 fp=0.82 cc= …[+157 chars]
```
</details>

**Bash**
```
cat > cmp.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rr
K,Twc,rec=load_inputs('/work/inputs')
B=mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64)); C=rec['T_cam_mesh_coconet'].astype(np.float64); P=rec['T_cam_mesh_foundationpose'].astype(np.float64)
def rd(A,Bm): return np.degrees(Rr.from_matrix(np.einsum('nji,njk->nik',A[:,:3,:3],Bm[:,:3,:3])).magnitude())
r1=rd(B,C); r2=rd(B,P); r3=rd(C,P)
for a in range(0,901,30): print(a,'b-c %.1f b-fp %.1f c-fp %.1f'%(r1[a:a+30].mean(),r2[a:a+30].mean(),r3[a:a+30].mean()), 'zB %.3f zC %.3f zP %.3f'%(B[a,2,3],C[a,2,3],P[a,2,3]))
EOF
python cmp.py
```

<details><summary>output</summary>

```
0 b-c 1.3 b-fp 2.4 c-fp 1.6 zB 3.144 zC 3.094 zP 3.213
30 b-c 1.3 b-fp 2.4 c-fp 1.7 zB 3.138 zC 3.094 zP 3.270
60 b-c 1.3 b-fp 2.7 c-fp 1.9 zB 3.117 zC 3.094 zP 3.177
90 b-c 1.3 b-fp 7.3 c-fp 6.6 zB 3.147 zC 3.131 zP 3.193
120 b-c 1.3 b-fp 30.2 c-fp 29.6 zB 3.303 zC 3.232 zP 3.209
150 b-c 1.3 b-fp 7.6 c-fp 7.4 zB 3.394 zC 3.456 zP 7.038
180 b-c 1.3 b-fp 2.3 c-fp 2.3 zB 3.393 zC 3.370 zP 3.047
210 b-c 1.3 b-fp 2.0 c-fp 1.9 zB 3.369 zC 3.422 zP 3.509
240 b-c 1.3 b-fp 1.9 c-fp 1.7 zB 3.375 zC 3.446 zP 3.567
270 b-c 1.3 b-fp 1.8 c-fp 1.0 zB 3.392 zC 3.394 zP 3.373
300 b-c 1.3 b-fp 5.5 c-fp 5.3 zB 3.395 zC 3.425 zP 3.476
330 b-c 1.3 b-fp 11.4 c-fp 11.0 zB 3.421 zC 3.406 zP 4.482
360 b-c 1.3 b-fp 18.9 c-fp 18.5 zB 3.455 zC 3.403 zP 2.993
390 b-c 1.3 b-fp 3.0 c-fp 1.9 zB 3.329 zC 3.286 zP 3.375
4 …[+979 chars]
```
</details>

> *thinking:* The rendering pipeline works, and the baseline (essentially CoCoNet with a constant rotation offset and depth correction) gives decent silhouette IoU in static phases but performs poorly around frames 120–150, 300–360, 570–600, and 810–900. I'll examine those problem frames next.

**Bash**
```
cd /work/scratch; python ov.py ov2.jpg 130,330,570,600,820,850,880,900
```

**Read**
```
/work/scratch/ov2.jpg
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
cd /work/scratch; cat > views.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rr
R=MeshRenderer('/work/inputs/object_mesh.glb')
K=np.array([[800,0,200],[0,800,200],[0,0,1.]])
outs=[]
for name,rv in [('+y view',[90,0,0]),('-y view',[-90,0,0]),('+x',[0,-90,0]),('-x',[0,90,0]),('+z',[0,180,0]),('-z',[0,0,0])]:
    T=np.eye(4); T[:3,:3]=Rr.from_euler('xyz',rv,degrees=True).as_matrix(); T[2,3]=0.8
    rgb,m,_=R.render(T[None],K,(0,0,400,400),(400,400))
    im=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8).copy()
    cv2.putText(im,name,(5,25),0,0.8,(0,0,255),2); outs.append(im)
cv2.imwrite('views.jpg',np.hstack(outs))
EOF
python views.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/views.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
free -g; nproc; ls /pipeline/depth/moge2/ -la; python -c "
import h5py; f=h5py.File('/pipeline/depth/moge2/aligned.h5'); print({k:(f[k].shape,f[k].dtype) for k in f})"
```

<details><summary>output</summary>

```
Exit code 1
               total        used        free      shared  buff/cache   available
Mem:             125          27          25           0          72          96
Swap:              1           1           0
48
total 4203504
drwxrwxr-x 2 hanzhou hanzhou       4096 Oct  5 20:35 .
drwxrwxr-x 3 hanzhou hanzhou       4096 Oct  5 20:29 ..
-rw-rw-r-- 1 hanzhou hanzhou          0 Oct  5 20:35 .aligned.h5.storage-repack.lock
-rw-rw-r-- 1 hanzhou hanzhou 1115675348 Oct  5 20:44 aligned.h5
-rw-rw-r-- 1 hanzhou hanzhou       1810 Oct  5 20:32 depth_report.json
-rw-rw-r-- 1 hanzhou hanzhou      95913 Oct  5 20:32 intrinsics.pkl
-rw-rw-r-- 1 hanzhou hanzhou 3188588672 Oct  5 20:32 raw.npy
Traceback (most recent call last):
  File "<string>", line 2, in <module>
  File "<string>", line 2, in  …[+66 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > motion.py <<'EOF'
import numpy as np, h5py, cv2
f=h5py.File('/work/inputs/masks.h5'); OM=f['object'][:]; PM=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4'); G=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    G.append(cv2.GaussianBlur(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY),(5,5),0).astype(np.float32))
G=np.stack(G); F=len(G)
k=3
d=np.full(F,np.nan); cen=np.full((F,2),np.nan); area=OM.reshape(F,-1).sum(1)
for t in range(F):
    ys,xs=np.nonzero(OM[t])
    if len(xs): cen[t]=[xs.mean(),ys.mean()]
for t in range(k,F):
    reg=(OM[t]|OM[t-k]); reg=cv2.dilate(reg.astype(np.uint8),np.ones((5,5)))>0
    reg&=~PM[t]&~PM[t-k]
    if reg.sum()>200: d[t]=np.abs(G[t]-G[t-k])[reg].mean()
np.savez('motion.npz',d=d,cen=cen,area=area)
for t in range(0,F,5): print(t, '%.2f'%d[t], np.round(cen[t],1), area[t], int(PM[t][cv2.dilate(OM[t].astype(np.uint8),np.ones((15,15)))>0].sum()))
EOF
python motion.py
```

<details><summary>output</summary>

```
0 nan [809.5 611.3] 1929 0
5 0.45 [810.4 611.4] 1975 0
10 0.56 [809.8 611.3] 1944 0
15 0.63 [810.3 611.4] 1970 0
20 0.59 [810.4 611.4] 1972 0
25 0.39 [810.5 611.4] 1965 0
30 0.47 [810.4 611.3] 1966 0
35 0.55 [810.4 611.3] 1964 0
40 0.53 [810.4 611.3] 1964 0
45 0.42 [810.3 611.3] 1961 0
50 1.06 [810.4 611.2] 1871 103
55 1.88 [809.7 611.3] 1979 146
60 1.78 [810.2 611.3] 1957 246
65 1.98 [810.4 611.3] 1881 380
70 1.35 [813.1 611.4] 1811 343
75 1.97 [812.7 611.7] 1756 373
80 5.67 [813.7 610.7] 1823 412
85 7.15 [813.  609.7] 1880 403
90 19.17 [812.2 607.5] 2026 347
95 10.74 [812. 604.] 2261 570
100 20.78 [807.  601.1] 2175 553
105 23.48 [794.1 592.2] 1891 749
110 23.54 [776.7 586.4] 1432 900
115 22.04 [749.6 582. ] 1094 1626
120 15.07 [720.5 585.4] 1136 1523
125 46.69 [699.2 594.3] 1292 1138
13 …[+4964 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > fixcrops.py <<'EOF'
import cv2,numpy as np,sys
fr=[int(a) for a in sys.argv[2].split(',')]; cx,cy=int(sys.argv[3]),int(sys.argv[4]); s=int(sys.argv[5]) if len(sys.argv)>5 else 60
cap=cv2.VideoCapture('/work/inputs/video.mp4'); out=[]; i=0
while i<=max(fr):
    ok,f=cap.read()
    if i in fr:
        c=cv2.resize(f[cy-s*3//4:cy+s*3//4,cx-s:cx+s],(400,300),interpolation=cv2.INTER_CUBIC); cv2.putText(c,str(i),(5,25),0,0.8,(0,0,255),2); out.append(c)
    i+=1
while len(out)%5: out.append(np.zeros_like(out[0]))
cv2.imwrite(sys.argv[1],np.vstack([np.hstack(out[r*5:r*5+5]) for r in range(len(out)//5)]))
EOF
python fixcrops.py e1.jpg 836,842,848,854,860,866,872,878,884,890,894,898,900 688 615 50
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/e1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import h5py
f=h5py.File('/pipeline/depth/moge2/aligned.h5')
def p(n,o):
  print(n, getattr(o,'shape',''), getattr(o,'dtype',''))
f.visititems(p)
print(dict(f.attrs))
"; cat /pipeline/depth/moge2/depth_report.json
```

<details><summary>output</summary>

```
aligned  
aligned/front_stereo_camera_left (901,) object
alignment  
alignment/front_stereo_camera_left  
alignment/front_stereo_camera_left/scale (901,) float32
alignment/front_stereo_camera_left/shift (901,) float32
alignment/front_stereo_camera_left/valid_count (901,) int32
frame_names  
frame_names/front_stereo_camera_left (901,) object
raw  
raw/front_stereo_camera_left (901,) object
{'complete': True, 'depth_alignment_input_identity_json': '{"depth_backend":"moge2","depth_model_id":"Ruicheng/moge-2-vitl-normal","depth_model_revision":"b135031bae30b5ac2ae141a0e68717795ce38340","depth_source_commit":"925b8ed835a7a9cdb7578ba15c658a0afc969030","direct_mhr":{"mtime_ns":1791218142484751117,"path":"/home/hanzhou/Projects/v2d/run/out/episode_000011/mhr_init/episode_000011.pkl","size":4021056 …[+2423 chars]
```
</details>

**Bash**
```
python -c "
import h5py, numpy as np, cv2
f=h5py.File('/pipeline/depth/moge2/aligned.h5')
a=f['aligned/front_stereo_camera_left'][0]; print(type(a), len(a), a[:8])
im=cv2.imdecode(np.frombuffer(a,np.uint8),cv2.IMREAD_UNCHANGED); print(im.shape, im.dtype, im.min(), im.max())
print(f['alignment/front_stereo_camera_left/scale'][:5], f['alignment/front_stereo_camera_left/shift'][:5])
print(f['frame_names/front_stereo_camera_left'][:2])
r=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r'); print(r.shape,r.dtype, r[0].min(), r[0].max(), r[0,611,810], im[611,810])
"
```

<details><summary>output</summary>

```
<class 'numpy.ndarray'> 648767 [137  80  78  71  13  10  26  10]
(1152, 1536) uint16 1812 10078
[1.1243744 1.121233  1.1227219 1.1203622 1.1217712] [0. 0. 0. 0. 0.]
[b'000000' b'000001']
(901, 1152, 1536) uint16 1612 8964 2726 3065
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


# ---------------------------------------------------------------- geometry helpers
def so3_exp(w):
    """w (B,3) torch -> (B,3,3)"""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def rot_angle_deg(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def flip_T(axis):
    """180 deg rotation about mesh axis (0=x,1=y,2=z) through the mesh origin."""
    T = np.eye(4)
    d = -np.ones(3); d[axis] = 1
    T[:3, :3] = np.diag(d)
    return T


SYM_FLIPS = [np.eye(4), flip_T(0), flip_T(1), flip_T(2)]


# ---------------------------------------------------------------- observations
class Obs:
    """Per-frame cropped observations for render-and-compare."""

    def __init__(self, frames, OM, PM, bg_depth, K, crop_wh=(192, 144)):
        self.frames, self.OM, self.PM, self.bg, self.K = frames, OM, PM, bg_depth, K
        self.cw, self.ch = crop_wh

    def crop_for(self, i, center):
        cx, cy = center
        x0 = int(round(cx - self.cw / 2)); y0 = int(round(cy - s …[+2945 chars]
```

<details><summary>output</summary>

```
ok
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


def gray(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def eval_terms(rgb, m, depth, img, om, pm, bg, occ_margin=0.05):
    """Per-item losses. Pixels covered by the person (and not object) are ignored; rendered pixels
    lying behind the static background depth (+margin) are treated as occluded (e.g. by the table)."""
    occ_bg = (depth > bg + occ_margin).float() * (1 - om)
    valid = (1 - pm * (1 - om)) * (1 - occ_bg)
    mv = m * valid
    inter = (mv * om).sum((1, 2)); union = (mv + om - mv * om).sum((1, 2))
    sil = ((mv - om * valid) ** 2).sum((1, 2)) / om.sum((1, 2)).clamp_min(50)
    siou = inter / union.clamp_min(1)
    # photometric NCC on overlap
    w = (m > 0.5).float() * om * (1 - pm)
    gi = gray(blur(img, 5, 1.0)); gr = gray(blur(rgb * m[..., None], 5, 1.0)) / blur(m[..., None], 5, 1.0)[..., 0].clamp_min(1e-3)
    n = w.sum((1, 2)).clamp_min(1)
    mi = (gi * w).sum((1, 2)) / n; mr = (gr * w).sum((1, 2)) / n
    di = (gi - mi[:, None, None]) * w; drr = (gr - mr[:, None, None]) * w
    ncc = (di * drr).sum((1, 2)) / ((di ** 2).sum((1, 2)) * (drr ** 2).sum((1, 2))).sqrt().clamp_min(1e-6)
    return dict(sil=sil, siou=siou, ncc=ncc, nov=n)


class PoseFitter:
    def __init__(self, R, obs, K, size=(192, 144)):
        self.R, self.obs, self.K, self.size = R, obs, K, size

    def crops_for(self, T, idx):
        cs = []
        for T_, i in zip(T, idx):
            p = self.K @ T_[:3, 3]
 …[+3376 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Agent:** Building the render-and-compare fitter now (silhouette + texture NCC, with person and table/background occlusion handled via the MoGe background depth). Testing it on a few frames first.

**Bash**
```
cd /work/scratch; cat > t1.py <<'EOF'
exec(open('setup.py').read())
t=time.time()
idx=list(range(0,901,1))
sc={}
for k,Fl in enumerate(SYM_FLIPS):
    T=Tb@Fl
    res={'siou':[], 'ncc':[], 'sil':[]}
    for a in range(0,901,64):
        s=fit.score(T[a:a+64], idx[a:a+64])
        for kk in res: res[kk].append(s[kk])
    sc[k]={kk:np.concatenate(v) for kk,v in res.items()}
print('score time',time.time()-t)
np.savez('flipscores.npz',**{f'{k}_{kk}':v for k in sc for kk,v in sc[k].items()})
for a in range(0,901,30):
    print(a, ' | '.join('f%d iou %.2f ncc %.2f'%(k,sc[k]['siou'][a:a+30].mean(),sc[k]['ncc'][a:a+30].mean()) for k in sc))
EOF
python t1.py
```

<details><summary>output</summary>

```
load 25.451849460601807
score time 4.71337890625
0 f0 iou 0.78 ncc 0.68 | f1 iou 0.78 ncc 0.44 | f2 iou 0.78 ncc 0.27 | f3 iou 0.77 ncc 0.40
30 f0 iou 0.78 ncc 0.68 | f1 iou 0.78 ncc 0.46 | f2 iou 0.78 ncc 0.28 | f3 iou 0.77 ncc 0.42
60 f0 iou 0.81 ncc 0.73 | f1 iou 0.82 ncc 0.52 | f2 iou 0.81 ncc 0.24 | f3 iou 0.80 ncc 0.40
90 f0 iou 0.76 ncc 0.22 | f1 iou 0.76 ncc 0.05 | f2 iou 0.76 ncc -0.19 | f3 iou 0.76 ncc -0.04
120 f0 iou 0.36 ncc 0.25 | f1 iou 0.36 ncc 0.26 | f2 iou 0.35 ncc -0.06 | f3 iou 0.35 ncc -0.04
150 f0 iou 0.69 ncc 0.00 | f1 iou 0.68 ncc -0.11 | f2 iou 0.68 ncc -0.17 | f3 iou 0.69 ncc -0.05
180 f0 iou 0.86 ncc -0.20 | f1 iou 0.85 ncc -0.42 | f2 iou 0.84 ncc -0.32 | f3 iou 0.86 ncc -0.19
210 f0 iou 0.89 ncc -0.10 | f1 iou 0.87 ncc -0.34 | f2 iou 0.87 ncc -0.25 | f3 iou 0.88 …[+2226 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
s=s.replace("""        self.verts_np = np.asarray(m.vertices)
""","""        self.verts_np = np.asarray(m.vertices)
        fn = np.asarray(m.face_normals, np.float32)
        vn = np.zeros((len(m.vertices), 3), np.float32)
        vn[np.asarray(m.faces).reshape(-1)] = np.repeat(fn, 3, 0)
        self.vn = torch.tensor(vn, device=DEV)
""")
s=s.replace("""    depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, R.f)
    feat = torch.cat([rgb * m, m, depth * m], -1)
    feat = dr.antialias(feat.contiguous(), rast, clip, R.f)
    if ss > 1:
        feat = torch.nn.functional.avg_pool2d(feat.permute(0, 3, 1, 2), ss).permute(0, 2, 3, 1)
    m = feat[..., 3]
    rgb = feat[..., :3] / m[..., None].clamp_min(1e-4)
    depth = feat[..., 4] / m.clamp_min(1e-4)
    return rgb, m, depth""","""    depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, R.f)
    nc = torch.einsum('bij,nj->bni', T_cam[:, :3, :3], R.vn).contiguous()
    nrm, _ = dr.interpolate(nc, rast, R.f)
    feat = torch.cat([rgb * m, m, depth * m, nrm * m], -1)
    feat = dr.antialias(feat.contiguous(), rast, clip, R.f)
    if ss > 1:
        feat = torch.nn.functional.avg_pool2d(feat.permute(0, 3, 1, 2), ss).permute(0, 2, 3, 1)
    m = feat[..., 3]
    mm = m[..., None].clamp_min(1e-4)
    rgb = feat[..., :3] / mm
    depth = feat[..., 4] / m.clamp_min(1e-4)
    nrm = feat[..., 5:8] / mm
    return rgb, m, depth, nrm""")
# new eval_terms with shadi …[+2438 chars]
```

<details><summary>output</summary>

```
268:            rgb, m, d, nr = render_soft(self.R, T[pid_t], self.K, crops, self.size)
277:            rgb, m, d, nr = render_soft(self.R, T[pid_t], self.K, crops, self.size)
288:            rgb, m, d, nr = render_soft(self.R, Tt, self.K, crops, self.size)
load 27.282470703125
score time 12.998873949050903
0 f0 iou 0.78 ncc 0.53 | f1 iou 0.78 ncc 0.41 | f2 iou 0.78 ncc 0.42 | f3 iou 0.77 ncc 0.43
30 f0 iou 0.78 ncc 0.54 | f1 iou 0.78 ncc 0.45 | f2 iou 0.78 ncc 0.45 | f3 iou 0.77 ncc 0.45
60 f0 iou 0.81 ncc 0.64 | f1 iou 0.82 ncc 0.53 | f2 iou 0.81 ncc 0.54 | f3 iou 0.80 ncc 0.55
90 f0 iou 0.76 ncc 0.49 | f1 iou 0.76 ncc 0.44 | f2 iou 0.76 ncc 0.42 | f3 iou 0.76 ncc 0.46
120 f0 iou 0.36 ncc 0.72 | f1 iou 0.36 ncc 0.72 | f2 iou 0.35 ncc 0.69 | f3 iou 0.35 ncc 0.71
150 f0 iou 0.69 ncc 0.20 | …[+2417 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > ovh.py <<'EOF'
# overlay hypotheses: python ovh.py out.jpg frame posesfile(npz with T (N,4,4) cam frame) 
exec(open('setup.py').read().replace("frames=load_video('/work/inputs/video.mp4')","frames=None"))
import sys
out=sys.argv[1]; fr=int(sys.argv[2]); Ts=np.load(sys.argv[3])['T']
cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,fr); ok,img=cap.read()
rows=[]
for T in Ts:
    P=K@T[:3,3]; cx,cy=P[:2]/P[2]
    crop=(int(cx)-60,int(cy)-45,120,90)
    rgb,m,d=R.render(T[None],K,crop,(480,360))
    im=cv2.resize(img[crop[1]:crop[1]+90,crop[0]:crop[0]+120],(480,360),interpolation=cv2.INTER_CUBIC)
    r=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8); mm=m[0].cpu().numpy()
    bl=im.copy(); cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(bl,cnt,-1,(0,0,255),1)
    rows.append(np.hstack([im,bl,np.where(mm[...,None],r,0)]))
cv2.imwrite(out,np.vstack(rows))
EOF
python -c "
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
K,Twc,rec=load_inputs('/work/inputs'); Tb=mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64))
np.savez('h720.npz',T=np.stack([Tb[720]@F for F in SYM_FLIPS]))
np.savez('h240.npz',T=np.stack([Tb[240]@F for F in SYM_FLIPS]))
"; python ovh.py h720.jpg 720 h720.npz
```

<details><summary>output</summary>

```
load 9.974748134613037
```
</details>

**Read**
```
/work/scratch/h720.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python fixcrops.py d1.jpg 640,650,660,668,672,680,690,700,720,745,750,756 612 808 50; python fixcrops.py b1.jpg 200,208,214,220,240,270,290,296,300,304,306,310 645 820 50
```

**Read**
```
/work/scratch/d1.jpg
```

**Read**
```
/work/scratch/b1.jpg
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

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat >> /work/outputs/hoi_lib.py <<'EOF'


def mean_pose(Ts):
    """Chordal L2 mean rotation + median translation of (N,4,4)."""
    M = Ts[:, :3, :3].sum(0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = np.median(Ts[:, :3, 3], 0)
    return T


def pick_frames(PM, OM, frames, n):
    """Pick up to n frames of a phase, preferring frames where the object is least covered by the person."""
    frames = np.asarray(frames)
    occ = np.array([(PM[i] & cv2.dilate(OM[i].astype(np.uint8), np.ones((9, 9))).astype(bool)).sum()
                    / max(OM[i].sum(), 1) for i in frames])
    order = np.argsort(occ, kind='stable')
    keep = np.sort(frames[order[:max(n * 2, 1)]])
    if len(keep) > n:
        keep = keep[np.linspace(0, len(keep) - 1, n).round().astype(int)]
    return keep
EOF
cat > t2.py <<'EOF'
exec(open('setup.py').read())
phases={'A':(0,78),'B':(200,303),'C':(402,468),'D1':(645,669),'D2':(692,746),'E':(845,900)}
for name,(a,b) in phases.items():
    fr=pick_frames(PM,OM,range(a,b+1),16)
    T0=mean_pose(Tb[a:b+1])
    t=time.time()
    res=[]
    for k,Fl in enumerate(SYM_FLIPS):
        Tf,terms=fit.fit((T0@Fl)[None],fr,pid=np.zeros(len(fr),int),iters=80)
        loss=terms['sil']+0.3*(1-terms['ncc'])
        res.append((k,loss.mean(),terms['siou'].mean(),terms['ncc'].mean(),loss,Tf[0]))
    L=np.stack([r[4] for r in res])
    wins=np.bincount(L.argmin(0),minle …[+327 chars]
```

<details><summary>output</summary>

```
load 22.079579830169678
A time 12.7 f0 L 0.286 iou 0.790 r2 0.726 | f1 L 0.179 iou 0.889 r2 0.718 | f2 L 0.193 iou 0.872 r2 0.737 | f3 L 0.217 iou 0.879 r2 0.625 wins [ 0 16  0  0] drot vs T0 [6.0, 3.7, 3.9, 3.9]
B time 13.5 f0 L 0.298 iou 0.895 r2 0.327 | f1 L 0.348 iou 0.891 r2 0.189 | f2 L 0.307 iou 0.897 r2 0.295 | f3 L 0.334 iou 0.895 r2 0.221 wins [15  0  1  0] drot vs T0 [3.1, 7.9, 4.9, 10.4]
C time 9.0 f0 L 0.164 iou 0.880 r2 0.809 | f1 L 0.141 iou 0.910 r2 0.778 | f2 L 0.153 iou 0.902 r2 0.762 | f3 L 0.149 iou 0.897 r2 0.793 wins [ 0 11  0  5] drot vs T0 [5.3, 5.0, 4.6, 5.7]
D1 time 7.6 f0 L 0.215 iou 0.885 r2 0.644 | f1 L 0.253 iou 0.868 r2 0.577 | f2 L 0.257 iou 0.848 r2 0.655 | f3 L 0.256 iou 0.862 r2 0.613 wins [15  0  0  1] drot vs T0 [16.8, 11.3, 12.4, 10.1]
D2 time 9.4 f0 L …[+365 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np
s=np.load('static_A.npy'); np.savez('hA.npz',T=s)
s=np.load('static_D2.npy'); np.savez('hD2.npz',T=s)
"; python ovh.py hA.jpg 30 hA.npz; python ovh.py hD2.jpg 720 hD2.npz
```

**Read**
```
/work/scratch/hA.jpg
```

<details><summary>output</summary>

```
load 8.78507113456726
load 8.72657561302185
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > prof.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
import time
K,Twc,rec=load_inputs('/work/inputs')
R=MeshRenderer('/work/inputs/object_mesh.glb')
Tb=mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64))
T=torch.tensor(Tb[:64],dtype=torch.float32,device=DEV)
crops=[(700,550,192,144)]*64
for ss in [1,2]:
  for it in range(3):
    torch.cuda.synchronize(); t=time.time()
    out=render_soft(R,T,K,crops,(192,144),ss=ss)
    torch.cuda.synchronize(); print(ss,time.time()-t)
print(torch.cuda.max_memory_allocated()/1e9)
EOF
python prof.py
```

<details><summary>output</summary>

```
1 0.16339540481567383
1 0.02929854393005371
1 0.040107011795043945
2 0.041059017181396484
2 0.036322593688964844
2 0.02803206443786621
4.882834432
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


def rot_grid(step_deg, half_deg):
    from scipy.spatial.transform import Rotation as Rr
    v = np.arange(-half_deg, half_deg + 1e-6, step_deg)
    g = np.stack(np.meshgrid(v, v, v, indexing='ij'), -1).reshape(-1, 3)
    return Rr.from_rotvec(np.radians(g)).as_matrix()


def solve_static(fit, T0, frames, n_frames=16, n_grid_frames=4, w_photo=0.3, iters=100, topk=2,
                 grid=(7.5, 15.0), PM=None, OM=None):
    """One constant camera-frame pose for a static phase. For each of the 4 box-symmetry flips of T0:
    coarse rotation grid around it (silhouette+texture on a few frames), then gradient refinement of the
    best candidates on n_frames frames. Returns list (per flip) of (T, mean loss, terms)."""
    fr = pick_frames(PM, OM, frames, n_frames)
    frg = pick_frames(PM, OM, frames, n_grid_frames)
    p = fit.K @ T0[:3, 3]
    crop = fit.obs.crop_for(0, (p[0] / p[2], p[1] / p[2]))
    G = rot_grid(*grid)
    out = []
    c = fit.R.verts_np.mean(0)
    for k, Fl in enumerate(SYM_FLIPS):
        Tk = T0 @ Fl
        cands = []
        for g in G:
            T = Tk.copy()
            T[:3, :3] = Tk[:3, :3] @ g
            T[:3, 3] = Tk[:3, 3] + Tk[:3, :3] @ c - T[:3, :3] @ c  # rotate about mesh centroid
            cands.append(T)
        cands = np.stack(cands)
        losses = np.zeros(len(cands))
        for i in frg:
            for a in range(0, len(cands), 64):
                s = fit.score(cands[a:a + 64], [i] * le …[+1205 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b691u7vie). Output is being written to: /tmp/claude-1003/-work/9189d8f3-53b7-480e-8b44-ca357b834506/tasks/b691u7vie.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/9189d8f3-53b7-480e-8b44-ca357b834506/tasks/b691u7vie.output
```

<details><summary>output</summary>

```
load 24.995824575424194
A 43.1s f0 L 0.171 iou 0.883 r2 0.763 | f1 L 0.180 iou 0.891 r2 0.710 | f2 L 0.215 iou 0.871 r2 0.660 | f3 L 0.226 iou 0.880 r2 0.593 wins [16  0  0  0]
B 48.0s f0 L 0.335 iou 0.876 r2 0.285 | f1 L 0.331 iou 0.907 r2 0.180 | f2 L 0.303 iou 0.907 r2 0.274 | f3 L 0.334 iou 0.898 r2 0.215 wins [ 1  0 15  0]
C 38.8s f0 L 0.159 iou 0.895 r2 0.771 | f1 L 0.144 iou 0.904 r2 0.790 | f2 L 0.157 iou 0.890 r2 0.789 | f3 L 0.154 iou 0.890 r2 0.799 wins [ 0 11  5  0]
D1 25.9s f0 L 0.207 iou 0.888 r2 0.662 | f1 L 0.241 iou 0.873 r2 0.597 | f2 L 0.223 iou 0.873 r2 0.672 | f3 L 0.238 iou 0.870 r2 0.639 wins [12  1  3  0]
```
</details>

**Bash**
```
sleep 45; cat /tmp/claude-1003/-work/9189d8f3-53b7-480e-8b44-ca357b834506/tasks/b691u7vie.output | tail -2; cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
K,Twc,rec=load_inputs('/work/inputs')
for name,key in [('base','T_world_mesh'),('fp','T_cam_mesh_foundationpose')]:
  T=rec[key].astype(np.float64)
  print(name)
  for t in range(1,901):
    a=[rot_angle_deg(T[t,:3,:3],T[t-1,:3,:3]@F[:3,:3]) for F in SYM_FLIPS]
    if np.argmin(a)!=0 or a[0]>20: print(t, np.round(a,1))
"

```

<details><summary>output</summary>

```
[exited with code 0]
base
146 [ 23.2 176.4 157.2 177.6]
147 [ 26.9 179.7 153.5 175.3]
151 [ 29.9 178.9 151.2 172.3]
156 [ 29.1 176.  151.7 174.7]
157 [ 25.7 176.6 154.7 177.4]
159 [ 22.6 176.7 157.6 179.4]
328 [ 26.2 179.3 153.8 178.9]
331 [ 23.3 178.3 157.5 174.1]
362 [ 25.4 156.7 170.7 176.4]
363 [ 35.  157.3 155.5 170.7]
573 [ 22.8 176.7 162.5 165.9]
575 [ 24.3 176.4 161.  165.4]
580 [ 20.2 160.3 177.2 176.5]
582 [ 27.4 178.5 153.2 174.6]
585 [ 23.5 174.7 157.4 176.7]
587 [ 34.3 179.4 145.7 179.7]
813 [ 23.5 168.8 159.8 176.2]
896 [ 27.2 171.7 155.1 172.9]
897 [ 27.2 171.7 155.1 172.9]
fp
117 [ 28.9 178.8 151.2 178.4]
138 [145.8 130.  100.5 110.4]
141 [139.7 142.4 101.8 104. ]
143 [ 45.2 178.1 135.5 172.3]
144 [ 21.6 179.7 159.8 172.3]
334 [ 22.4 177.6 158.4 174.4]
358 [ 27.7 179.  152. …[+818 chars]
```
</details>

> *thinking:* The baseline tracks identity consistently across frames and matches the book's spine orientation during the first static phase, though the texture score struggles with top/bottom-cover flips. I'll visually verify the other static phases before relying on it.

**Bash**
```
cd /work/scratch; for p in C:430 D1:660 B:240 D2:720; do n=${p%%:*}; f=${p##*:}; python ovh.py st_$n.jpg $f st_$n.npz 2>/dev/null | tail -0; done; python -c "
import cv2,numpy as np
a=[cv2.imread(f'st_{n}.jpg') for n in ['C','D1']]
cv2.imwrite('stCD.jpg',np.hstack(a))
a=[cv2.imread(f'st_{n}.jpg') for n in ['B','D2']]
cv2.imwrite('stBD.jpg',np.hstack(a))
"
```

**Read**
```
/work/scratch/stCD.jpg
```

<details><summary>output</summary>

```
Exit code 1
[ WARN:0@0.028] global loadsave.cpp:268 findDecoder imread_('st_C.jpg'): can't open/read file: check file path/integrity
[ WARN:0@0.030] global loadsave.cpp:268 findDecoder imread_('st_D1.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "<string>", line 4, in <module>
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'imwrite'
> Overload resolution failed:
>  - img data type = object is not supported
>  - Expected Ptr<cv::UMat> for argument 'img'
```
</details>

<details><summary>output</summary>

```
File does not exist. Note: your current working directory is /work/scratch.
```
</details>

**Bash**
```
cd /work/scratch; python ovh.py st_C.jpg 430 st_C.npz 2>&1 | tail -3; ls st_*
```

<details><summary>output</summary>

```
load 8.93268632888794
st_A.npz
st_B.npz
st_C.jpg
st_C.npz
st_D1.npz
st_D2.npz
st_E.npz
```
</details>

**Bash**
```
cd /work/scratch; python ovh.py st_D1.jpg 660 st_D1.npz >/dev/null; python ovh.py st_B.jpg 240 st_B.npz >/dev/null; python ovh.py st_D2.jpg 720 st_D2.npz >/dev/null; python -c "
import cv2,numpy as np
a=[cv2.imread(f'st_{n}.jpg') for n in ['C','D1']]
cv2.imwrite('stCD.jpg',np.hstack(a))
a=[cv2.imread(f'st_{n}.jpg') for n in ['B','D2']]
cv2.imwrite('stBD.jpg',np.hstack(a))
"
```

**Read**
```
/work/scratch/stCD.jpg
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

**Read**
```
/work/scratch/stBD.jpg
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
from hoi_lib import *
import json
K,Twc,rec=load_inputs('/work/inputs')
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n,d=pl[:3],pl[3]
print('n',n,'d',d, 'cam center dist', n@Twc[:3,3]+d)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.asarray(m.vertices)[::20]
Tw=rec['T_world_mesh'].astype(np.float64)
for name,(a,b) in {'A':(0,78),'B':(200,303),'C':(402,468),'D1':(645,669),'D2':(692,746),'E':(845,900)}.items():
  ds=[]; ups=[]
  for i in range(a,b+1):
    P=V@Tw[i,:3,:3].T+Tw[i,:3,3]; s=P@n+d
    ds.append([s.min(),s.max(), Tw[i,:3,3]@n+d])
    ups.append(abs(Tw[i,:3,1]@n))
  ds=np.array(ds); print(name,'plane dist min/max/center (median) %.3f %.3f %.3f'%tuple(np.median(ds,0)), 'std center %.3f'%ds[:,2].std(), '|y.n| %.3f'%np.median(ups))
"

```

<details><summary>output</summary>

```
n [-0.01881248 -0.99828219 -0.05548658] d 1.3896661071987289 cam center dist 1.3347325757242718
A plane dist min/max/center (median) 0.893 0.960 0.927 std center 0.004 |y.n| 1.000
B plane dist min/max/center (median) -0.005 0.071 0.033 std center 0.010 |y.n| 0.998
C plane dist min/max/center (median) 0.888 0.950 0.919 std center 0.002 |y.n| 1.000
D1 plane dist min/max/center (median) 0.019 0.084 0.050 std center 0.005 |y.n| 0.999
D2 plane dist min/max/center (median) 0.016 0.085 0.049 std center 0.006 |y.n| 1.000
E plane dist min/max/center (median) 0.775 0.863 0.817 std center 0.006 |y.n| 0.995
```
</details>

**Bash**
```
cd /work/scratch; cat > t4.py <<'EOF'
exec(open('setup.py').read())
from scipy.spatial.transform import Rotation as Rr
a,b=469,644
idx=np.arange(a,b+1)
t=time.time()
Tr=np.zeros((len(idx),4,4)); TT={}
for s in range(0,len(idx),32):
    ii=idx[s:s+32]
    Tf,terms=fit.fit(Tb[ii],ii,iters=50)
    Tr[s:s+32]=Tf
    for k,v in terms.items(): TT.setdefault(k,[]).append(v)
TT={k:np.concatenate(v) for k,v in TT.items()}
sb=fit.score(Tb[idx[:64]],idx[:64])
print('time',time.time()-t)
S0={k:[] for k in TT}
for s in range(0,len(idx),64):
    sc=fit.score(Tb[idx[s:s+64]],idx[s:s+64])
    for k in S0: S0[k].append(sc[k])
S0={k:np.concatenate(v) for k,v in S0.items()}
dr_=rot_angle_deg(Tr[:,:3,:3],Tb[idx,:3,:3])
for s in range(0,len(idx),10):
    sl=slice(s,s+10)
    print(idx[s],'base iou %.2f r2 %.2f | fit iou %.2f r2 %.2f | drot %.1f'%(S0['siou'][sl].mean(),S0['ncc'][sl].mean(),TT['siou'][sl].mean(),TT['ncc'][sl].mean(),dr_[sl].mean()))
vel=lambda T: rot_angle_deg(T[1:,:3,:3],T[:-1,:3,:3])
print('rot speed base median %.2f mean %.2f, fit median %.2f mean %.2f'%(np.median(vel(Tb[idx])),vel(Tb[idx]).mean(),np.median(vel(Tr)),vel(Tr).mean()))
np.save('m3_fit.npy',Tr)
EOF
python t4.py
```

<details><summary>output</summary>

```
load 21.798065662384033
time 12.468576669692993
469 base iou 0.70 r2 0.60 | fit iou 0.81 r2 0.75 | drot 7.9
479 base iou 0.65 r2 0.50 | fit iou 0.77 r2 0.52 | drot 10.7
489 base iou 0.75 r2 0.41 | fit iou 0.84 r2 0.40 | drot 10.9
499 base iou 0.89 r2 0.83 | fit iou 0.90 r2 0.86 | drot 5.5
509 base iou 0.91 r2 0.85 | fit iou 0.90 r2 0.87 | drot 2.7
519 base iou 0.92 r2 0.66 | fit iou 0.89 r2 0.69 | drot 5.8
529 base iou 0.90 r2 0.68 | fit iou 0.94 r2 0.63 | drot 7.5
539 base iou 0.87 r2 0.76 | fit iou 0.95 r2 0.65 | drot 8.6
549 base iou 0.88 r2 0.63 | fit iou 0.94 r2 0.62 | drot 6.8
559 base iou 0.77 r2 0.48 | fit iou 0.92 r2 0.52 | drot 8.2
569 base iou 0.17 r2 0.77 | fit iou 0.21 r2 0.81 | drot 13.5
579 base iou 0.14 r2 0.66 | fit iou 0.40 r2 0.66 | drot 15.1
589 base iou 0.35 r2 0.59 |  …[+401 chars]
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


def fit_traj(fit, T_init, idx, iters=60, w_photo=0.3, lam_acc_r=30.0, lam_acc_t=3000.0, lam_prior_r=0.5,
             lam_prior_t=0.0, img_mask=None, chunk=32, lr_r=0.01, lr_t=0.002, fixed_ends=None, z_free=False):
    """Joint optimisation of a trajectory segment: per-frame image terms (silhouette + texture) plus
    second-difference smoothness on rotation (chordal) and translation, plus a weak prior towards T_init.
    img_mask: optional bool per frame (False -> frame contributes no image term, e.g. held-out frames).
    fixed_ends: optional (T_before, T_after) poses adjacent to the segment (e.g. static phase poses)."""
    N = len(idx)
    T0 = torch.tensor(np.asarray(T_init), dtype=torch.float32, device=DEV)
    w = torch.zeros(N, 3, device=DEV, requires_grad=True)
    dt = torch.zeros(N, 3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
    zmask = torch.tensor([1., 1., 1. if z_free else 0.], device=DEV)
    crops = fit.crops_for(np.asarray(T_init), idx)
    if img_mask is None:
        img_mask = np.ones(N, bool)
    chunks = [np.arange(s, min(s + chunk, N)) for s in range(0, N, chunk)]
    obs_c = [batch_obs(fit.obs, [idx[j] for j in c], [crops[j] for j in c]) for c in chunks]
    ends = None
    if fixed_ends is not None:
        ends = [None if e is None else torch.tensor(e, dtype=torch.float32, device=DEV) for e in fixed_ends]

    def poses():
        …[+2675 chars]
```

<details><summary>output</summary>

```
load 24.84923791885376
lam_r 30 lam_p 0.5 time 17.5 HELDOUT iou base 0.699 fit 0.733 | r2 base 0.625 fit 0.664 || train iou 0.698->0.738 | drot mean 4.0 | rot speed med 1.77 mean 1.84 (base 2.63 4.28)
lam_r 100 lam_p 0.5 time 16.4 HELDOUT iou base 0.699 fit 0.729 | r2 base 0.625 fit 0.669 || train iou 0.698->0.732 | drot mean 3.5 | rot speed med 1.82 mean 1.79 (base 2.63 4.28)
lam_r 300 lam_p 0.5 time 20.5 HELDOUT iou base 0.699 fit 0.727 | r2 base 0.625 fit 0.665 || train iou 0.698->0.730 | drot mean 3.4 | rot speed med 1.82 mean 1.80 (base 2.63 4.28)
```
</details>

**Bash**
```
cd /work/scratch; cat > t6.py <<'EOF'
exec(open('setup.py').read())
a,b=469,644
idx=np.arange(a,b+1); N=len(idx)
vel=lambda T: rot_angle_deg(T[1:,:3,:3],T[:-1,:3,:3])
S0=score_traj(fit,Tb[idx],idx)
train=(np.arange(N)%2==0); te=~train
for kw in [dict(iters=120,lr_r=0.02,lr_t=0.003,lam_acc_r=30,lam_prior_r=0.1),dict(iters=120,lr_r=0.02,lr_t=0.003,lam_acc_r=10,lam_acc_t=1000,lam_prior_r=0.1)]:
    t=time.time()
    Tr=fit_traj(fit,Tb[idx],idx,img_mask=train,**kw)
    S=score_traj(fit,Tr,idx)
    dr_=rot_angle_deg(Tr[:,:3,:3],Tb[idx,:3,:3])
    print(kw,'time %.1f'%(time.time()-t),
      'HELDOUT iou base %.3f fit %.3f | r2 base %.3f fit %.3f || train iou %.3f | drot mean %.1f | rot speed med %.2f mean %.2f'%(
      S0['siou'][te].mean(),S['siou'][te].mean(),S0['ncc'][te].mean(),S['ncc'][te].mean(),S['siou'][train].mean(),dr_.mean(),
      np.median(vel(Tr)),vel(Tr).mean()))
    for s in range(0,N,20): print('  ',idx[s],'iou %.2f->%.2f drot %.1f'%(S0['siou'][s:s+20].mean(),S['siou'][s:s+20].mean(),dr_[s:s+20].mean()))
EOF
python t6.py
```

<details><summary>output</summary>

```
load 21.04977536201477
{'iters': 120, 'lr_r': 0.02, 'lr_t': 0.003, 'lam_acc_r': 30, 'lam_prior_r': 0.1} time 52.5 HELDOUT iou base 0.699 fit 0.746 | r2 base 0.625 fit 0.665 || train iou 0.754 | drot mean 5.2 | rot speed med 1.79 mean 1.73
   469 iou 0.68->0.76 drot 3.7
   489 iou 0.82->0.83 drot 4.6
   509 iou 0.91->0.90 drot 2.1
   529 iou 0.88->0.93 drot 4.3
   549 iou 0.82->0.92 drot 4.5
   569 iou 0.16->0.14 drot 13.6
   589 iou 0.49->0.62 drot 6.5
   609 iou 0.73->0.80 drot 3.1
   629 iou 0.82->0.88 drot 3.6
{'iters': 120, 'lr_r': 0.02, 'lr_t': 0.003, 'lam_acc_r': 10, 'lam_acc_t': 1000, 'lam_prior_r': 0.1} time 32.9 HELDOUT iou base 0.699 fit 0.763 | r2 base 0.625 fit 0.638 || train iou 0.774 | drot mean 6.0 | rot speed med 1.74 mean 1.84
   469 iou 0.68->0.77 drot 4.7
   489 iou 0.82 …[+233 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > dbg.py <<'EOF'
exec(open('setup.py').read())
import sys
fr=[int(x) for x in sys.argv[2].split(',')]
T=Tb if len(sys.argv)<4 else None
rows=[]
for i in fr:
    p=K@Tb[i,:3,3]; crop=obs.crop_for(i,(p[0]/p[2],p[1]/p[2]))
    img,om,pm,bg=obs.get(i,crop)
    rgb,m,d,nr=render_soft(R,torch.tensor(Tb[i:i+1],dtype=torch.float32,device=DEV),K,[crop],fit.size)
    m=m[0].cpu().numpy(); d=d[0].cpu().numpy()
    occ=(d>bg+0.05)&(m>0.5)
    vis=(img*255).astype(np.uint8)[:,:,::-1].copy()
    a=np.zeros_like(vis); a[...,1]=om*255; a[...,2]=(m>0.5)*255; a[...,0]=pm*120
    b=np.zeros_like(vis); b[occ]=(0,0,255); 
    bgv=np.clip((bg-2.5)/2*255,0,255).astype(np.uint8); bgv=cv2.cvtColor(bgv,cv2.COLOR_GRAY2BGR)
    row=np.hstack([vis,a,np.maximum(b,bgv//2)])
    cv2.putText(row,str(i),(3,15),0,0.5,(0,0,255),1)
    rows.append(cv2.resize(row,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
cv2.imwrite(sys.argv[1],np.vstack(rows))
EOF
python dbg.py dbg1.jpg 568,575,582,588
```

<details><summary>output</summary>

```
load 22.694952487945557
```
</details>

**Read**
```
/work/scratch/dbg1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* I found that the baseline badly mislocates the book in occluded frames (e.g. ~575–590, placing it above the table edge instead of below), and joint trajectory optimization already boosts held-out IoU from 0.70 to 0.76 on segment 469–644. I'm now adding a distance-transform term plus a 2D shift search to fix these larger positional offsets.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
# batch_obs returns DT too
s=s.replace("""def batch_obs(obs, idx, crops):
    imgs, oms, pms, bgs = zip(*[obs.get(i, c) for i, c in zip(idx, crops)])
    t = lambda a: torch.tensor(np.stack(a), device=DEV)
    return t(imgs), t(oms).float(), t(pms).float(), t(bgs).float()""","""def batch_obs(obs, idx, crops):
    imgs, oms, pms, bgs = zip(*[obs.get(i, c) for i, c in zip(idx, crops)])
    dts = [np.minimum(cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3), 30.0) if o.any()
           else np.full(o.shape, 30.0, np.float32) for o in oms]
    t = lambda a: torch.tensor(np.stack(a), device=DEV)
    return t(imgs), t(oms).float(), t(pms).float(), t(bgs).float(), t(dts).float()""")
s=s.replace("def eval_terms(rgb, m, depth, nrm, img, om, pm, bg, occ_margin=0.05):","def eval_terms(rgb, m, depth, nrm, img, om, pm, bg, dtm, occ_margin=0.05):")
s=s.replace("""    siou = inter / union.clamp_min(1)
    w = (m > 0.5)""","""    siou = inter / union.clamp_min(1)
    # chamfer-like term: rendered visible pixels far from the observed mask (gives a pull when not overlapping)
    cham = (mv * dtm).sum((1, 2)) / mv.sum((1, 2)).clamp_min(1) / 30.0 * (om.sum((1, 2)) > 20).float()
    w = (m > 0.5)""")
s=s.replace("return dict(sil=sil, siou=siou, ncc=r2, nov=n)","return dict(sil=sil, siou=siou, ncc=r2, nov=n, cham=cham)")
s=s.replace("img, om, pm, bg = batch_obs(self.obs, idx, crops)","ob = batch_obs(self.obs, idx, crops)")
s …[+541 chars]
```

<details><summary>output</summary>

```
6:W_CHAM = 1.0
145:def batch_obs(obs, idx, crops):
205:def eval_terms(rgb, m, depth, nrm, img, om, pm, bg, dtm, occ_margin=0.05):
259:        ob = batch_obs(self.obs, idx, crops)
274:            terms = eval_terms(rgb, m, d, nr, *ob)
275:            li = terms['sil'] + w_photo * (1 - terms['ncc']) + W_CHAM * terms['cham']
283:            terms = eval_terms(rgb, m, d, nr, *ob)
291:            ob = batch_obs(self.obs, idx, crops)
294:            terms = eval_terms(rgb, m, d, nr, *ob)
380:    obs_c = [batch_obs(fit.obs, [idx[j] for j in c], [crops[j] for j in c]) for c in chunks]
395:            terms = eval_terms(rgb, m, d, nr, *ob)
396:            li = (terms['sil'] + w_photo * (1 - terms['ncc']) + W_CHAM * terms['cham']) * torch.tensor(img_mask[c], device=DEV).float()
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


def shift_search(fit, T, idx, max_shift=40, bs=32, w_cham=1.0):
    """For each frame, find the image-plane shift (pixels) of the rendered silhouette of pose T that best
    explains the object mask (same loss as eval_terms' silhouette + chamfer, evaluated for all integer shifts
    by correlation). Returns shifted poses (moved parallel to the image plane at constant z), loss at the
    best shift, and loss at zero shift."""
    import torch.nn.functional as Fn
    T = np.asarray(T)
    out_T = T.copy(); Lbest = np.zeros(len(T)); L0 = np.zeros(len(T)); shifts = np.zeros((len(T), 2))
    S = max_shift
    for s in range(0, len(T), bs):
        ii = list(idx[s:s + bs]); Ts = T[s:s + bs]
        crops = fit.crops_for(Ts, ii)
        img, om, pm, bg, dtm = batch_obs(fit.obs, ii, crops)
        with torch.no_grad():
            _, m, d, _ = render_soft(fit.R, torch.tensor(Ts, dtype=torch.float32, device=DEV), fit.K, crops, fit.size)
        H, W = m.shape[1:]
        for j in range(len(ii)):
            mj = m[j]
            if mj.sum() < 5 or om[j].sum() < 20:
                continue
            z = float(Ts[j, 2, 3])
            occ = ((z > bg[j] + 0.05).float()) * (1 - om[j])
            valid = (1 - pm[j] * (1 - om[j])) * (1 - occ)
            ys, xs = torch.nonzero(mj > 0.01, as_tuple=True)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            k = mj[y0:y1, x0:x1]
            pad  …[+1730 chars]
```

<details><summary>output</summary>

```
load 23.532139778137207
time 8.230671882629395
0 L0 0.21 Lb 0.17 [-1.  2.] [2. 2.]
10 L0 0.21 Lb 0.17 [-1.5  2. ] [2. 2.]
20 L0 0.21 Lb 0.17 [-1.5  2. ] [2. 2.]
30 L0 0.21 Lb 0.16 [-2.  2.] [3. 2.]
40 L0 0.21 Lb 0.15 [-2.  2.] [3. 2.]
50 L0 0.21 Lb 0.15 [-2.  2.] [3. 2.]
60 L0 0.19 Lb 0.17 [-2.  1.] [3. 2.]
70 L0 0.16 Lb 0.14 [-1.  1.] [2. 2.]
80 L0 0.16 Lb 0.13 [-1.  1.] [2. 2.]
90 L0 0.16 Lb 0.14 [-1.  1.] [2. 3.]
100 L0 0.22 Lb 0.21 [0. 1.] [5. 2.]
110 L0 0.29 Lb 0.28 [0. 0.] [4. 3.]
120 L0 0.27 Lb 0.25 [-2. -1.] [4. 2.]
130 L0 0.72 Lb 0.32 [ 0. -1.] [ 4. 22.]
140 L0 1.89 Lb 0.28 [-4.5 23.5] [21. 31.]
150 L0 0.35 Lb 0.17 [-7.  -1.5] [10. 13.]
160 L0 0.30 Lb 0.13 [-7. -3.] [8. 4.]
170 L0 0.34 Lb 0.12 [-7.5 -3. ] [9. 4.]
180 L0 0.16 Lb 0.10 [-1.5  0. ] [3. 7.]
190 L0 0.12 Lb 0.09 [0. 2.]  …[+2757 chars]
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


def viterbi(unary, trans):
    """unary (N,C); trans(t) -> (C,C) cost from candidate at t-1 to t. Returns best path."""
    N, C = unary.shape
    cost = unary[0].copy(); back = np.zeros((N, C), int)
    for t in range(1, N):
        tot = cost[:, None] + trans(t)
        back[t] = tot.argmin(0)
        cost = tot.min(0) + unary[t]
    path = np.zeros(N, int); path[-1] = cost.argmin()
    for t in range(N - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth_series(y, w, lam, groups=None):
    """Weighted 2nd-difference smoothing: min sum w (x-y)^2 + lam sum (x[t+1]-2x[t]+x[t-1])^2.
    groups: list of (a,b) index ranges forced to a constant value (static phases)."""
    import scipy.sparse as sp, scipy.sparse.linalg as spl
    N = len(y)
    # parametrise x = P u, where frames of a static group share one variable
    var = np.arange(N)
    if groups:
        for a, b in groups:
            var[a:b + 1] = var[a]
    uniq, var = np.unique(var, return_inverse=True)
    P = sp.csr_matrix((np.ones(N), (np.arange(N), var)), shape=(N, len(uniq)))
    D = sp.diags([np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N))
    Wm = sp.diags(w)
    A = P.T @ (Wm + lam * D.T @ D) @ P
    u = spl.spsolve(A.tocsc(), P.T @ (w * y))
    return P @ u
EOF
cat > /work/outputs/solve.py <<'EOF'
#!/usr/bin/env python
"""Trajectory-level 6-DoF object pose recovery for one monocular HOI video (light-blue  …[+7253 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: buda3i28n). Output is being written to: /tmp/claude-1003/-work/9189d8f3-53b7-480e-8b44-ca357b834506/tasks/buda3i28n.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
cd /work/scratch; cat > /work/outputs/validate.py <<'EOF'
#!/usr/bin/env python
"""Ground-truth-free validation: per-phase table baseline vs result.
usage: python validate.py --inputs /work/inputs --result object_poses_world.npz [--md out.md]"""
import argparse, os, sys, json
import numpy as np, h5py, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoi_lib as H
from solve import STATIC

ap = argparse.ArgumentParser()
ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--result', required=True)
ap.add_argument('--pipeline', default='/pipeline'); ap.add_argument('--md', default=None)
args = ap.parse_args()
K, Twc, rec = H.load_inputs(args.inputs)
Tb = H.mesh_to_cam(Twc, rec['T_world_mesh'].astype(np.float64))
Tr = H.mesh_to_cam(Twc, np.load(args.result)['T_world_mesh'])
F = len(Tb)
frames = H.load_video(f'{args.inputs}/video.mp4')
with h5py.File(f'{args.inputs}/masks.h5') as f:
    OM = f['object'][:]; PM = f['person'][:]
with h5py.File(f'{args.pipeline}/depth/moge2/aligned.h5') as fa:
    ds = [cv2.imdecode(np.frombuffer(fa['aligned/front_stereo_camera_left'][i], np.uint8), cv2.IMREAD_UNCHANGED)
          for i in range(0, F, 10)]
bg = (np.median(np.stack(ds), 0) / 1000.0).astype(np.float32)
R = H.MeshRenderer(f'{args.inputs}/object_mesh.glb')
fit = H.PoseFitter(R, H.Obs(frames, OM, PM, bg, K), K)
idx = np.arange(F)
Sb = H.score_traj(fit, Tb, idx); Sr = H.score_traj(fit, Tr, idx)
pl = np.array(json.load(open(f'{args.inputs}/gr …[+2533 chars]
```

<details><summary>output</summary>

```
loaded inputs in 22.0s
static A 0-79: loss 0.197 rot change vs baseline mean 2.8 deg  [29s]
static B 212-303: loss 0.289 rot change vs baseline mean 4.8 deg  [34s]
static C 405-466: loss 0.184 rot change vs baseline mean 4.9 deg  [39s]
static D1 656-669: loss 0.194 rot change vs baseline mean 18.5 deg  [44s]
static D2 695-747: loss 0.134 rot change vs baseline mean 2.7 deg  [48s]
moving 80-211: shifted 0.79 of frames, mean rot change vs baseline 7.7 deg  [69s]
moving 304-404: shifted 0.52 of frames, mean rot change vs baseline 8.8 deg  [85s]
moving 467-655: shifted 0.86 of frames, mean rot change vs baseline 8.9 deg  [111s]
moving 670-694: shifted 0.52 of frames, mean rot change vs baseline 4.0 deg  [115s]
moving 748-900: shifted 0.76 of frames, mean rot change vs baseline 6.6 deg  [137s]
 …[+60 chars]
```
</details>

**Agent:** First full pipeline run finished in 137 s. One warning sign: the short floor phase D1 (656–669) moved 18.5° away from the baseline, which I need to check. Running the validation table now.

**Bash**
```
cd /work/scratch; python /work/outputs/validate.py --result run1.npz 2>&1 | tail -15
```

<details><summary>output</summary>

```
| phase | frames | type | IoU base | IoU ours | tex R2 base | tex R2 ours | rot speed base (med/p95 deg/fr) | rot speed ours | trans acc base (cm/fr^2, p95) | ours | dz mean (cm) | dz median (cm) | rot diff to base (mean deg) | bottom-to-floor base/ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0-79 | static | 0.786 | 0.854 | 0.569 | 0.774 | 0.06/1.48 | 0.00/0.00 | 1.91 | 0.00 | +0.27 | -0.00 | 2.9 | 89.3/88.4 |
| M0 | 80-211 | moving | 0.694 | 0.873 | 0.466 | 0.449 | 1.92/18.73 | 1.29/4.63 | 8.90 | 0.53 | -0.70 | -0.69 | 7.7 | 71.7/65.6 |
| B | 212-303 | static | 0.891 | 0.910 | 0.337 | 0.304 | 0.10/1.32 | 0.00/0.00 | 7.14 | 0.00 | -0.38 | +0.00 | 5.1 | -0.5/-0.9 |
| M1 | 304-404 | moving | 0.703 | 0.885 | 0.487 | 0.465 | 2.53/11.01 | 1.64/3.05 | 9.46 | 0. …[+941 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
import json
K,Twc,rec=load_inputs('/work/inputs'); Tb=mesh_to_cam(Twc,rec['T_world_mesh'].astype(np.float64))
d=np.load('run1_diag.npz'); T=d['T_cam']
np.savez('h660.npz',T=np.stack([Tb[660],T[660]]))
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3])
for nm in ['A','B','C','D1','D2']:
  Ts=d['static_'+nm]; Rw=Twc[:3,:3]@Ts[:3,:3]; print(nm,'|y.n| %.4f'%abs(Rw[:,1]@n), 'tilt deg %.1f'%np.degrees(np.arccos(abs(Rw[:,1]@n))))
"; python ovh.py h660.jpg 660 h660.npz >/dev/null
```

**Read**
```
/work/scratch/h660.jpg
```

<details><summary>output</summary>

```
A |y.n| 0.9986 tilt deg 3.1
B |y.n| 0.9959 tilt deg 5.2
C |y.n| 0.9975 tilt deg 4.0
D1 |y.n| 0.9841 tilt deg 10.2
D2 |y.n| 0.9981 tilt deg 3.5
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > ovcmp.py <<'EOF'
# python ovcmp.py out.jpg frames result.npz : rows per frame: image | baseline overlay | result overlay
exec(open('setup.py').read().replace("frames=load_video('/work/inputs/video.mp4')","frames=None"))
import sys
fr=[int(x) for x in sys.argv[2].split(',')]
Tr=mesh_to_cam(Twc,np.load(sys.argv[3])['T_world_mesh'])
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs={}; i=0
while i<=max(fr):
    ok,f=cap.read()
    if i in fr: imgs[i]=f
    i+=1
rows=[]
def draw(img,T,crop,col):
    rgb,m,d=R.render(T[None],K,crop,(360,270)); mm=m[0].cpu().numpy().astype(np.uint8)
    cnt,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cnt,-1,col,1)
    r=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8); return r
for i in fr:
    P=K@Tb[i,:3,3]; cx,cy=P[:2]/P[2]; crop=(int(cx)-60,int(cy)-45,120,90)
    im=cv2.resize(imgs[i][crop[1]:crop[1]+90,crop[0]:crop[0]+120],(360,270),interpolation=cv2.INTER_CUBIC)
    a=im.copy(); b=im.copy(); ra=draw(a,Tb[i],crop,(0,0,255)); rb=draw(b,Tr[i],crop,(0,255,0))
    row=np.hstack([im,a,ra,b,rb]); cv2.putText(row,str(i),(5,25),0,0.8,(0,0,255),2); rows.append(row)
cv2.imwrite(sys.argv[1],np.vstack(rows))
EOF
python ovcmp.py cmp1.jpg 140,330,580,600,820,880,895 run1.npz >/dev/null
```

**Read**
```
/work/scratch/cmp1.jpg
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
cd /work/scratch; python fixcrops.py e2.jpg 780,800,840,860,880,900 690 610 110
```

**Read**
```
/work/scratch/e2.jpg
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
import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from hoi_lib import *
K,Twc,rec=load_inputs('/work/inputs')
kp=rec['human_keypoints_world']; Tb=rec['T_world_mesh']; Tr=np.load('run1.npz')['T_world_mesh']
rh=kp[:,21:42]; lh=kp[:,42:63]
for f in [100,120,180,330,380,520,600,780,800,840,860,880,895]:
  out=[]
  for T in [Tb,Tr]:
    c=T[f,:3,3]; out.append((np.linalg.norm(rh[f]-c,axis=1).min(),np.linalg.norm(lh[f]-c,axis=1).min()))
  mid=(rh[f].mean(0)+lh[f].mean(0))/2
  print(f,'base dist R/L %.3f %.3f'%out[0],' ours %.3f %.3f'%out[1], 'hands mid', np.round(mid,3),'base c',np.round(Tb[f,:3,3],3),'ours c',np.round(Tr[f,:3,3],3))
"

```

<details><summary>output</summary>

```
100 base dist R/L 0.094 0.059  ours 0.094 0.068 hands mid [0.304 0.227 3.203] base c [0.355 0.241 3.139] ours c [0.355 0.248 3.131]
120 base dist R/L 0.055 0.054  ours 0.058 0.058 hands mid [0.046 0.246 3.318] base c [0.044 0.21  3.284] ours c [0.045 0.217 3.269]
180 base dist R/L 0.032 0.119  ours 0.019 0.122 hands mid [-0.189  0.956  3.338] base c [-0.233  1.02   3.376] ours c [-0.244  1.007  3.372]
330 base dist R/L 0.030 0.089  ours 0.022 0.073 hands mid [-0.211  0.788  3.378] base c [-0.235  0.768  3.407] ours c [-0.249  0.819  3.378]
380 base dist R/L 0.071 0.023  ours 0.078 0.040 hands mid [0.119 0.255 3.345] base c [0.144 0.253 3.353] ours c [0.143 0.242 3.369]
520 base dist R/L 0.009 0.033  ours 0.014 0.022 hands mid [0.055 0.114 3.282] base c [0.036 0.065 3.281] ours c [0.042 0.0 …[+996 chars]
```
</details>

> *thinking:* Visual check revealed two issues: the book was lifted ~5cm near the end due to over-permissive occlusion handling, and it tilted 10° off the floor during static phase D1. I'll fix these by restricting occluder pixels to near hand keypoints, treating other person pixels as negative evidence, adding a lie-flat constraint for static phases, and loosening the Viterbi translation cost to allow the correct 17px shift at frame 580.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
# Obs: optional hand mask per frame; person pixels: near hands -> occluder (ignored), elsewhere -> weight PERSON_NEG
s=s.replace("""    def __init__(self, frames, OM, PM, bg_depth, K, crop_wh=(192, 144)):
        self.frames, self.OM, self.PM, self.bg, self.K = frames, OM, PM, bg_depth, K
        self.cw, self.ch = crop_wh""","""    def __init__(self, frames, OM, PM, bg_depth, K, crop_wh=(192, 144), hand_uv=None, hand_r=14):
        self.frames, self.OM, self.PM, self.bg, self.K = frames, OM, PM, bg_depth, K
        self.cw, self.ch = crop_wh
        self.hand_uv, self.hand_r = hand_uv, hand_r   # (F,J,2) projected hand keypoints or None""")
s=s.replace("""        om = cut(self.OM[i]); pm = cut(self.PM[i]); bg = cut(self.bg, 100.0)
        return img, om, pm, bg""","""        om = cut(self.OM[i]); pm = cut(self.PM[i]); bg = cut(self.bg, 100.0)
        # person pixels: 1 = treated as occluder (ignored). Without hand info every person pixel is an occluder;
        # with hand keypoints only person pixels near the hands are occluders, the rest of the body counts as
        # partial negative evidence (value PERSON_NEG in [0,1): weight of 'object absent' there is 1-value).
        pm = pm.astype(np.float32)
        if self.hand_uv is not None:
            hm = np.zeros((h, w), np.uint8)
            for u, v in self.hand_uv[i]:
                cv2.circle(hm, (int(round(u - x0)), int(round(v - y0))), self.hand_r, 1, -1) …[+336 chars]
```

<details><summary>output</summary>

```
221:    occ_bg = (depth > bg + occ_margin).float() * (1 - om)
222:    valid = (1 - pm * (1 - om)) * (1 - occ_bg)
229:    w = (m > 0.5).float() * om * (1 - pm)
459:            valid = (1 - pm[j] * (1 - om[j])) * (1 - occ)
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
s=s.replace("""            T_prior=None, w_prior=0.0, crops=None):""","""            T_prior=None, w_prior=0.0, crops=None, flat_n=None, w_flat=0.0):""")
s=s.replace("""            if T_prior is not None and w_prior > 0:
                loss = loss + w_prior * ((w ** 2).sum() + 100 * (dt ** 2).sum())""","""            if T_prior is not None and w_prior > 0:
                loss = loss + w_prior * ((w ** 2).sum() + 100 * (dt ** 2).sum())
            if flat_n is not None and w_flat > 0:
                # object resting on a support plane: thin axis (mesh y) parallel to the plane normal
                nt = torch.tensor(flat_n, dtype=torch.float32, device=DEV)
                c = (T[:, :3, 1] * nt).sum(-1)
                loss = loss + w_flat * (1 - c.abs()).sum()""")
# solve_static flat mode
s=s.replace("""                 grid=(7.5, 15.0), PM=None, OM=None, flips=(0, 1, 2, 3)):""","""                 grid=(7.5, 15.0), PM=None, OM=None, flips=(0, 1, 2, 3), flat_n=None, w_flat=20.0,
                 yaw_half=20.0, yaw_step=2.5):""")
s=s.replace("""    G = rot_grid(*grid)
    out = []
    c = fit.R.verts_np.mean(0)
    for k in flips:
        Tk = T0 @ SYM_FLIPS[k]
        cands = []
        for g in G:
            T = Tk.copy()
            T[:3, :3] = Tk[:3, :3] @ g
            T[:3, 3] = Tk[:3, 3] + Tk[:3, :3] @ c - T[:3, :3] @ c  # rotate about mesh centroid
            cands.append(T)""","""    G = rot_grid(*grid …[+3504 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bv2w5lkpd). Output is being written to: /tmp/claude-1003/-work/9189d8f3-53b7-480e-8b44-ca357b834506/tasks/bv2w5lkpd.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/outputs; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 100; cat /tmp/claude-1003/-work/9189d8f3-53b7-480e-8b44-ca357b834506/tasks/bv2w5lkpd.output
```

<details><summary>output</summary>

```
loaded inputs in 25.4s
static A 0-79: loss 0.215 rot change vs baseline mean 1.9 deg  [32s]
static B 212-303: loss 0.303 rot change vs baseline mean 6.6 deg  [37s]
static C 405-466: loss 0.187 rot change vs baseline mean 3.1 deg  [41s]
static D1 656-669: loss 0.273 rot change vs baseline mean 14.7 deg  [45s]
static D2 695-747: loss 0.141 rot change vs baseline mean 2.3 deg  [49s]
moving 80-211: shifted 0.86 of frames, mean rot change vs baseline 8.1 deg  [69s]
moving 304-404: shifted 0.81 of frames, mean rot change vs baseline 8.8 deg  [84s]
moving 467-655: shifted 0.92 of frames, mean rot change vs baseline 8.7 deg  [113s]
moving 670-694: shifted 0.52 of frames, mean rot change vs baseline 3.4 deg  [117s]
moving 748-900: shifted 0.77 of frames, mean rot change vs baseline 6.4 deg  [141s]
 …[+1800 chars]
```
</details>

**Bash**
```
cd /work/scratch; python ovcmp.py cmp2.jpg 660,668,240,300 run2.npz >/dev/null
```

**Read**
```
/work/scratch/cmp2.jpg
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
cd /work/scratch; cat > t8.py <<'EOF'
exec(open('setup.py').read())
import sys; sys.path.insert(0,'/work/outputs'); from solve import STATIC
d1=np.load('run1_diag.npz'); d2=np.load('run2_diag.npz')
for n,a,b in STATIC:
    idx=np.arange(a,b+1)
    Tm=mean_pose(Tb[a:b+1])
    res=[]
    for lab,T in [('base/frame',Tb[idx]),('base mean',np.repeat(Tm[None],len(idx),0)),('run1',np.repeat(d1['static_'+n][None],len(idx),0)),('run2',np.repeat(d2['static_'+n][None],len(idx),0))]:
        S=score_traj(fit,T,idx)
        res.append('%s iou %.3f r2 %.3f'%(lab,S['siou'].mean(),S['ncc'].mean()))
    print(n,' | '.join(res))
EOF
python t8.py
```

<details><summary>output</summary>

```
load 30.684003353118896
A base/frame iou 0.786 r2 0.569 | base mean iou 0.782 r2 0.571 | run1 iou 0.855 r2 0.774 | run2 iou 0.831 r2 0.808
B base/frame iou 0.891 r2 0.337 | base mean iou 0.886 r2 0.355 | run1 iou 0.909 r2 0.309 | run2 iou 0.870 r2 0.375
C base/frame iou 0.785 r2 0.707 | base mean iou 0.795 r2 0.714 | run1 iou 0.850 r2 0.798 | run2 iou 0.859 r2 0.785
D1 base/frame iou 0.786 r2 0.638 | base mean iou 0.783 r2 0.643 | run1 iou 0.895 r2 0.681 | run2 iou 0.863 r2 0.522
D2 base/frame iou 0.942 r2 0.559 | base mean iou 0.941 r2 0.551 | run1 iou 0.961 r2 0.637 | run2 iou 0.947 r2 0.658
```
</details>

> *thinking:* The static-phase fit was inflating IoU by cropping off the book end hidden under the hand, which the baseline actually handles correctly. I'll anchor static-phase fits to the pooled baseline pose with a rotation prior and soft lie-flat term, and for carried phases use held-out validation (fitting on even frames, scoring on odd) to objectively tune the prior strength.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
s=s.replace("""            if T_prior is not None and w_prior > 0:
                loss = loss + w_prior * ((w ** 2).sum() + 100 * (dt ** 2).sum())""","""            if T_prior is not None and w_prior > 0:
                # prior towards the initial rotation (T0); translation only very weakly
                loss = loss + w_prior * ((w ** 2).sum() + 1.0 * (dt ** 2).sum())""")
open('hoi_lib.py','w').write(s)
s=open('solve.py').read()
s=s.replace("""    ap.add_argument('--diag', default=None, help='optional path for diagnostics npz')""","""    ap.add_argument('--diag', default=None, help='optional path for diagnostics npz')
    ap.add_argument('--holdout', action='store_true',
                    help='validation mode: odd frames of moving segments get no image term (scored afterwards)')
    ap.add_argument('--lam_prior_r', type=float, default=LAM_PRIOR_R)""")
s=s.replace("""STATIC = [('A', 0, 79),""","""# trajectory optimisation weights (chosen with the --holdout validation, see REPORT.md)
LAM_PRIOR_R = 1.0      # moving phases: prior towards the decoded initial rotation (baseline)
STATIC_W_PRIOR = 4.0   # static phases: prior towards the pooled baseline rotation (~0.03 loss per 5 deg)
STATIC_W_FLAT = 5.0    # static phases: soft 'lies flat on the support plane' term
STATIC = [('A', 0, 79),""")
s=s.replace("""        T0 = H.mean_pose(T_init[a:b + 1])
        T0[:3, 3] *= zt[a] / T0[2, 3]
        out, fr = H.solve_s …[+1969 chars]
```

<details><summary>output</summary>

```
131:        Tsh, Lb, L0, sh = H.shift_search(fit, Tc0, idx)
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace("""        Tsh, Lb, L0, sh = H.shift_search(fit, Tc0, idx)
""","""        Tsh, Lb, L0, sh = H.shift_search(fit, Tc0, idx)
        if args.holdout:   # held-out frames must not see their own image: use the neighbours' shift
            for j in range(1, len(idx), 2):
                nb = [k for k in (j - 1, j + 1) if k < len(idx)]
                du, dv = sh[nb].mean(0)
                Tsh[j] = Tc0[j]; Tsh[j, 0, 3] += du * Tc0[j, 2, 3] / K[0, 0]; Tsh[j, 1, 3] += dv * Tc0[j, 2, 3] / K[1, 1]
                L0[j] = Lb[j] = 0.0
""")
open('solve.py','w').write(s)
v=open('validate.py').read()
v=v.replace("""ap.add_argument('--pipeline', default='/pipeline'); ap.add_argument('--md', default=None)""","""ap.add_argument('--pipeline', default='/pipeline'); ap.add_argument('--md', default=None)
ap.add_argument('--holdout_eval', action='store_true', help='also report odd frames of moving segments only')""")
v=v.replace("""txt = '\\n'.join(rows)""","""if args.holdout_eval:
    rows.append('')
    rows.append('| moving segment | held-out (odd) frames | IoU base | IoU ours | tex R2 base | tex R2 ours | chamfer base | ours |')
    rows.append('|---|---|---|---|---|---|---|---|')
    allm = []
    for n, a, b, typ in ph:
        if typ != 'moving': continue
        o = np.arange(a + 1, b + 1, 2)
        o = o[OM[o].reshape(len(o), -1).sum(1) > 50]
        allm.append(o)
        rows.append(f'| {n} | {len(o)} | {Sb["siou" …[+777 chars]
```

<details><summary>output</summary>

```
started
```
</details>

**Bash**
```
python -c "
import numpy, scipy, torch, cv2, trimesh, h5py, nvdiffrast, PIL
for m in [numpy,scipy,torch,cv2,trimesh,h5py,nvdiffrast,PIL]: print(m.__name__, m.__version__)
"; python --version; nvidia-smi --query-gpu=name,memory.used --format=csv
```

<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
torch 2.5.1+cu124
cv2 4.11.0
trimesh 5.1.1
h5py 3.16.0
nvdiffrast 0.4.0
PIL 10.2.0
Python 3.11.10
name, memory.used [MiB]
NVIDIA RTX 6000 Ada Generation, 4209 MiB
NVIDIA RTX 6000 Ada Generation, 19 MiB
```
</details>

**Bash**
```
sleep 240; cd /work/scratch; cat ho_0.1.log; cat ho_0.1.val | tail -10
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
cd /work/scratch; cat ho_0.1.log; tail -10 ho_0.1.val 2>/dev/null
```

<details><summary>output</summary>

```
Exit code 1
loaded inputs in 19.9s
static A 0-79: IoU 0.826 rot change vs pooled baseline 2.5 deg  [25s]
static B 212-303: IoU 0.879 rot change vs pooled baseline 5.1 deg  [28s]
static C 405-466: IoU 0.834 rot change vs pooled baseline 3.9 deg  [31s]
static D1 656-669: IoU 0.858 rot change vs pooled baseline 2.5 deg  [33s]
static D2 695-747: IoU 0.956 rot change vs pooled baseline 2.2 deg  [39s]
moving 80-211: shifted 0.79 of frames, mean rot change vs baseline 7.0 deg  [88s]
moving 304-404: shifted 0.75 of frames, mean rot change vs baseline 7.7 deg  [131s]
```
</details>

**Bash**
```
sleep 110; cd /work/scratch; tail -4 ho_0.1.log; tail -10 ho_0.1.val 2>/dev/null
```

<details><summary>output</summary>

```
moving 467-655: shifted 0.88 of frames, mean rot change vs baseline 6.0 deg  [200s]
moving 670-694: shifted 0.60 of frames, mean rot change vs baseline 1.9 deg  [204s]
moving 748-900: shifted 0.78 of frames, mean rot change vs baseline 5.8 deg  [233s]
done in 233s -> /work/scratch/ho_0.1.npz
```
</details>

> *thinking:* Validation runs are underway — prior 0.1 is done and prior 1.0 is running (~4 min each), with about 25 of 45 allowed GPU minutes used. After this comparison, I'll move straight to the final run plus validation.

**Bash**
```
sleep 100; cd /work/scratch; tail -20 ho_0.1.val
```

<details><summary>output</summary>

```
| A | 0-79 | static | 0.786 | 0.833 | 0.569 | 0.793 | 0.06/1.48 | 0.02/0.02 | 1.91 | 0.00 | +0.27 | -0.00 | 1.8 | 89.3/88.7 |
| M0 | 80-211 | moving | 0.694 | 0.867 | 0.466 | 0.451 | 1.92/18.73 | 1.18/4.76 | 8.90 | 0.45 | -0.70 | -0.69 | 7.0 | 71.7/66.1 |
| B | 212-303 | static | 0.891 | 0.872 | 0.337 | 0.383 | 0.10/1.32 | 0.03/0.03 | 7.14 | 0.00 | -0.38 | +0.00 | 3.2 | -0.5/-0.7 |
| M1 | 304-404 | moving | 0.703 | 0.877 | 0.487 | 0.494 | 2.53/11.01 | 1.51/3.20 | 9.46 | 0.37 | +0.58 | +0.49 | 7.7 | 62.8/65.9 |
| C | 405-466 | static | 0.785 | 0.827 | 0.707 | 0.752 | 0.70/1.85 | 0.00/0.00 | 1.82 | 0.00 | -0.70 | +0.00 | 4.2 | 88.8/88.4 |
| M2 | 467-655 | moving | 0.737 | 0.853 | 0.620 | 0.616 | 2.51/13.74 | 1.90/3.17 | 8.56 | 0.43 | -0.61 | -0.59 | 6.0 | 81.2/80.0 |
| D1 | 656-669 | static  …[+1140 chars]
```
</details>

**Bash**
```
sleep 115; cd /work/scratch; tail -3 ho_1.0.log
```

<details><summary>output</summary>

```
moving 304-404: shifted 0.75 of frames, mean rot change vs baseline 6.0 deg  [115s]
moving 467-655: shifted 0.88 of frames, mean rot change vs baseline 4.5 deg  [177s]
moving 670-694: shifted 0.60 of frames, mean rot change vs baseline 1.4 deg  [186s]
```
</details>

**Bash**
```
sleep 90; cd /work/scratch; tail -8 ho_1.0.val; grep "ALL | 0-900" ho_1.0.val
```

<details><summary>output</summary>

```
| moving segment | held-out (odd) frames | IoU base | IoU ours | tex R2 base | tex R2 ours | chamfer base | ours |
|---|---|---|---|---|---|---|---|
| M0 | 65 | 0.700 | 0.855 | 0.467 | 0.440 | 0.0426 | 0.0027 |
| M1 | 49 | 0.698 | 0.855 | 0.490 | 0.525 | 0.0076 | 0.0028 |
| M2 | 90 | 0.737 | 0.847 | 0.610 | 0.618 | 0.0590 | 0.0230 |
| M3 | 12 | 0.905 | 0.922 | 0.610 | 0.624 | 0.0213 | 0.0161 |
| M4 | 72 | 0.823 | 0.868 | 0.369 | 0.527 | 0.0091 | 0.0061 |
| ALL | 288 | 0.751 | 0.859 | 0.497 | 0.539 | 0.0325 | 0.0105 |
| ALL | 0-900 |  | 0.783 | 0.862 | 0.509 | 0.561 | 1.25/11.76 | 0.65/3.12 | 8.73 | 0.50 | -0.21 | -0.14 | 4.0 | 67.6/67.4 |
```
</details>

> *thinking:* The held-out test confirms the refinement helps (IoU up, texture R² improved, chamfer error ~3x lower), and since both prior strengths perform similarly I'm keeping the more conservative 1.0 to stay closer to baseline. I'll also add an accept/reject rule for static phases that falls back to the pooled baseline pose whenever the refined pose doesn't improve IoU without hurting texture agreement, which should revert B and D1.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace("""    obs = H.Obs(frames, OM, PM, bg, K, hand_uv=hand_uv)""","""    obs = H.Obs(frames, OM, PM, bg, K, hand_uv=hand_uv)
    # second scorer with the plain criterion (all person pixels ignored) used for the static accept/reject test
    fit_plain = H.PoseFitter(R, H.Obs(frames, OM, PM, bg, K), K)""")
s=s.replace("""        Ts = Tf[0]
        Ts[:3, 3] *= zt[a] / Ts[2, 3]
        T_out[a:b + 1] = Ts""","""        Ts = Tf[0]
        Ts[:3, 3] *= zt[a] / Ts[2, 3]
        # accept the refined pose only if it explains ALL frames of the phase better than the pooled baseline
        # pose under the plain criterion: IoU gain > 0.01 and texture R^2 not lower by more than 0.01
        allf = np.arange(a, b + 1)
        allf = allf[OM[allf].reshape(len(allf), -1).sum(1) > 50]
        s_fit = H.score_traj(fit_plain, np.repeat(Ts[None], len(allf), 0), allf)
        s_b = H.score_traj(fit_plain, np.repeat(T0[None], len(allf), 0), allf)
        accept = (s_fit['siou'].mean() - s_b['siou'].mean() > 0.01) and (s_fit['ncc'].mean() - s_b['ncc'].mean() > -0.01)
        print(f'   static {name}: pooled baseline IoU {s_b["siou"].mean():.3f} R2 {s_b["ncc"].mean():.3f} | refined IoU '
              f'{s_fit["siou"].mean():.3f} R2 {s_fit["ncc"].mean():.3f} -> {"refined" if accept else "pooled baseline"}')
        if not accept:
            Ts = T0
        T_out[a:b + 1] = Ts""")
s=s.replace("""    print(f'done in {time.time() - …[+449 chars]
```

<details><summary>output</summary>

```
36:LAM_PRIOR_R = 1.0      # moving phases: prior towards the decoded initial rotation (baseline)
ok
```
</details>

**Bash**
```
cat > /work/outputs/depth_cues.py <<'EOF'
#!/usr/bin/env python
"""Independent depth cues for the depth re-anchoring decision (report only; solve.py applies 0 shift).
For each static phase: (1) floor contact: depth shift along the viewing ray that puts the lowest mesh point
on the released floor plane (floor phases only); (2) MoGe-2 aligned depth vs rendered object depth on visible
object pixels (median); (3) for carried phases: camera-depth offset hand keypoints vs object centre."""
import argparse, os, sys, json
import numpy as np, h5py, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoi_lib as H
from solve import STATIC
ap = argparse.ArgumentParser()
ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--result', required=True)
ap.add_argument('--pipeline', default='/pipeline')
args = ap.parse_args()
K, Twc, rec = H.load_inputs(args.inputs)
Tb = H.mesh_to_cam(Twc, rec['T_world_mesh'].astype(np.float64))
Tr = H.mesh_to_cam(Twc, np.load(args.result)['T_world_mesh'])
pl = np.array(json.load(open(f'{args.inputs}/ground_plane.json'))['plane'])
Tcw = np.linalg.inv(Twc)
n_c = Tcw[:3, :3] @ pl[:3]; d_c = pl[:3] @ Twc[:3, 3] + pl[3]   # plane in camera frame: n_c.x + d_c = 0
R = H.MeshRenderer(f'{args.inputs}/object_mesh.glb')
V = R.verts_np[::20]
with h5py.File(f'{args.inputs}/masks.h5') as f:
    OM = f['object']; PM = f['person']
    fa = h5py.File(f'{args.pipeline}/depth/moge2/aligned.h5')
    print('| phase | frames | floor-co …[+2300 chars]
```

<details><summary>output</summary>

```
loaded inputs in 26.5s
   static A: pooled baseline IoU 0.765 R2 0.548 | refined IoU 0.842 R2 0.796 -> refined
static A 0-79: IoU 0.831 rot change vs pooled baseline 2.2 deg  [33s]
   static B: pooled baseline IoU 0.848 R2 0.348 | refined IoU 0.873 R2 0.362 -> refined
static B 212-303: IoU 0.879 rot change vs pooled baseline 2.3 deg  [38s]
   static C: pooled baseline IoU 0.783 R2 0.708 | refined IoU 0.812 R2 0.793 -> refined
static C 405-466: IoU 0.821 rot change vs pooled baseline 4.5 deg  [42s]
   static D1: pooled baseline IoU 0.771 R2 0.622 | refined IoU 0.860 R2 0.615 -> refined
static D1 656-669: IoU 0.860 rot change vs pooled baseline 3.3 deg  [44s]
   static D2: pooled baseline IoU 0.922 R2 0.508 | refined IoU 0.954 R2 0.623 -> refined
static D2 695-747: IoU 0.955 rot change vs po …[+28 chars]
```
</details>

**Bash**
```
sleep 150; cat /work/scratch/final.log | tail -7
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
tail -7 /work/scratch/final.log
```

<details><summary>output</summary>

```
static D2 695-747: IoU 0.955 rot change vs pooled baseline 2.1 deg  [49s]
moving 80-211: shifted 0.86 of frames, mean rot change vs baseline 6.3 deg  [84s]
moving 304-404: shifted 0.81 of frames, mean rot change vs baseline 7.1 deg  [105s]
moving 467-655: shifted 0.92 of frames, mean rot change vs baseline 5.9 deg  [140s]
moving 670-694: shifted 0.52 of frames, mean rot change vs baseline 1.7 deg  [144s]
moving 748-900: shifted 0.77 of frames, mean rot change vs baseline 5.3 deg  [172s]
done in 172s -> /work/outputs/object_poses_world.npz; peak GPU memory 4.92 GB
```
</details>

**Agent:** Final `solve.py` run finished: 172 s wall time, peak GPU memory 4.9 GB. Running the validation table, the depth cues and a visual spot check.

**Bash**
```
cd /work/scratch; timeout 300 python /work/outputs/validate.py --result /work/outputs/object_poses_world.npz --md /work/scratch/final_val.md > final_val.log 2>&1; timeout 200 python /work/outputs/depth_cues.py --result /work/outputs/object_poses_world.npz > cues.log 2>&1; cat final_val.md; cat cues.log; python -c "
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(T.shape,T.dtype,np.isfinite(T).all(), np.abs(T[:,3]-[0,0,0,1]).max())
print('det',np.abs(np.linalg.det(T[:,:3,:3])-1).max())"
```

<details><summary>output</summary>

```
| phase | frames | type | IoU base | IoU ours | tex R2 base | tex R2 ours | rot speed base (med/p95 deg/fr) | rot speed ours | trans acc base (cm/fr^2, p95) | ours | dz mean (cm) | dz median (cm) | rot diff to base (mean deg) | bottom-to-floor base/ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0-79 | static | 0.786 | 0.840 | 0.569 | 0.794 | 0.06/1.48 | 0.01/0.01 | 1.91 | 0.00 | +0.27 | -0.00 | 1.6 | 89.3/88.6 |
| M0 | 80-211 | moving | 0.694 | 0.878 | 0.466 | 0.431 | 1.92/18.73 | 1.30/4.54 | 8.90 | 0.54 | -0.70 | -0.69 | 6.3 | 71.7/66.4 |
| B | 212-303 | static | 0.891 | 0.874 | 0.337 | 0.352 | 0.10/1.32 | 0.00/0.00 | 7.14 | 0.00 | -0.38 | +0.00 | 3.9 | -0.5/-0.3 |
| M1 | 304-404 | moving | 0.703 | 0.883 | 0.487 | 0.471 | 2.53/11.01 | 1.40/3.35 | 9.46 | 0. …[+1417 chars]
```
</details>

**Bash**
```
cd /work/scratch; python ovcmp.py cmpF.jpg 140,330,582,600,700,880 /work/outputs/object_poses_world.npz >/dev/null
```

**Read**
```
/work/scratch/cmpF.jpg
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
cd /work/scratch; mv /work/outputs/object_poses_world_scores.npz /work/scratch/final_scores.npz; python -c "
import numpy as np
s=np.load('final_scores.npz')
for a in range(580,640,5): print(a, 'iou %.2f->%.2f r2 %.2f->%.2f'%(s['b_siou'][a:a+5].mean(),s['r_siou'][a:a+5].mean(),s['b_ncc'][a:a+5].mean(),s['r_ncc'][a:a+5].mean()))
"
```

<details><summary>output</summary>

```
580 iou 0.28->0.71 r2 0.32->0.24
585 iou 0.00->0.65 r2 0.95->0.45
590 iou 0.19->0.86 r2 0.58->0.50
595 iou 0.64->0.87 r2 0.58->0.48
600 iou 0.62->0.83 r2 0.70->0.33
605 iou 0.64->0.87 r2 0.51->0.28
610 iou 0.73->0.84 r2 0.57->0.46
615 iou 0.72->0.84 r2 0.61->0.50
620 iou 0.73->0.82 r2 0.63->0.53
625 iou 0.78->0.85 r2 0.54->0.54
630 iou 0.82->0.85 r2 0.57->0.65
635 iou 0.82->0.87 r2 0.62->0.66
```
</details>

> *thinking:* Spot checks confirm fixes at most frames, but 585–620 show rising IoU with falling texture R², suggesting a rotation drift. I'll re-run the trajectory fit with higher texture weight and adopt it only if it improves held-out texture agreement without hurting IoU.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace("""    ap.add_argument('--lam_prior_r', type=float, default=LAM_PRIOR_R)""","""    ap.add_argument('--lam_prior_r', type=float, default=LAM_PRIOR_R)
    ap.add_argument('--w_photo', type=float, default=W_PHOTO_MOVING)""")
s=s.replace("""LAM_PRIOR_R = 1.0 ""","""W_PHOTO_MOVING = 0.3   # moving phases: weight of the texture term relative to the silhouette term
LAM_PRIOR_R = 1.0 """)
s=s.replace("lam_prior_r=args.lam_prior_r, fixed_ends=ends, img_mask=img_mask)","lam_prior_r=args.lam_prior_r, fixed_ends=ends, img_mask=img_mask, w_photo=args.w_photo)")
open('solve.py','w').write(s)
EOF
cd /work/scratch; nohup sh -c "python /work/outputs/solve.py --holdout --w_photo 1.0 --out /work/scratch/ho_wp1.npz > ho_wp1.log 2>&1; python /work/outputs/validate.py --holdout_eval --result ho_wp1.npz > ho_wp1.val 2>&1; mv ho_wp1_scores.npz ho_wp1_sc.npz" > /dev/null 2>&1 &
echo started
```

<details><summary>output</summary>

```
started
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
sleep 100; tail -3 /work/scratch/ho_wp1.log
```

<details><summary>output</summary>

```
static D2 695-747: IoU 0.952 rot change vs pooled baseline 2.1 deg  [38s]
moving 80-211: shifted 0.79 of frames, mean rot change vs baseline 5.5 deg  [63s]
moving 304-404: shifted 0.75 of frames, mean rot change vs baseline 6.6 deg  [94s]
```
</details>

**Bash**
```
sleep 115; tail -2 /work/scratch/ho_wp1.log
```

<details><summary>output</summary>

```
moving 748-900: shifted 0.78 of frames, mean rot change vs baseline 5.9 deg  [162s]
done in 163s -> /work/scratch/ho_wp1.npz; peak GPU memory 4.92 GB
```
</details>

**Bash**
```
sleep 75; cd /work/scratch; tail -8 ho_wp1.val; grep "ALL | 0-900" ho_wp1.val ho_1.0.val
```

<details><summary>output</summary>

```
| moving segment | held-out (odd) frames | IoU base | IoU ours | tex R2 base | tex R2 ours | chamfer base | ours |
|---|---|---|---|---|---|---|---|
| M0 | 65 | 0.700 | 0.794 | 0.467 | 0.475 | 0.0426 | 0.0037 |
| M1 | 49 | 0.698 | 0.825 | 0.490 | 0.583 | 0.0076 | 0.0033 |
| M2 | 90 | 0.737 | 0.817 | 0.610 | 0.655 | 0.0590 | 0.0253 |
| M3 | 12 | 0.905 | 0.901 | 0.610 | 0.620 | 0.0213 | 0.0155 |
| M4 | 72 | 0.823 | 0.830 | 0.369 | 0.557 | 0.0091 | 0.0080 |
| ALL | 288 | 0.751 | 0.820 | 0.497 | 0.576 | 0.0325 | 0.0119 |
ho_wp1.val:| ALL | 0-900 |  | 0.783 | 0.838 | 0.509 | 0.601 | 1.25/11.76 | 0.89/3.55 | 8.73 | 0.69 | -0.21 | -0.14 | 4.8 | 67.6/66.9 |
ho_1.0.val:| ALL | 0-900 |  | 0.783 | 0.862 | 0.509 | 0.561 | 1.25/11.76 | 0.65/3.12 | 8.73 | 0.50 | -0.21 | -0.14 | 4.0 | 67.6/67.4 |
```
</details>

> *thinking:* Comparing texture weights on held-out carried-phase frames, weight 1.0 (IoU 0.820, R² 0.576) outperforms 0.3 almost everywhere, so I'll adopt 1.0—since texture is the key cue for resolving rotation on box-shaped silhouettes—and proceed with the final run.

**Bash**
```
cd /work/outputs; sed -i 's/^W_PHOTO_MOVING = 0.3 /W_PHOTO_MOVING = 1.0 /' solve.py; grep -n "^W_PHOTO_MOVING\|^LAM_PRIOR_R\|^STATIC_W" solve.py; cd /work/scratch; nohup sh -c "/usr/bin/time -v python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz --diag /work/scratch/final_diag.npz > final.log 2> final.time; python /work/outputs/validate.py --result /work/outputs/object_poses_world.npz --md /work/scratch/final_val.md > final_val.log 2>&1; mv /work/outputs/object_poses_world_scores.npz /work/scratch/final_scores.npz; python /work/outputs/depth_cues.py --result /work/outputs/object_poses_world.npz > cues.log 2>&1" > /dev/null 2>&1 & echo started
```

<details><summary>output</summary>

```
36:W_PHOTO_MOVING = 1.0   # moving phases: weight of the texture term relative to the silhouette term
37:LAM_PRIOR_R = 1.0      # moving phases: prior towards the decoded initial rotation (baseline)
38:STATIC_W_PRIOR = 4.0   # static phases: prior towards the pooled baseline rotation (~0.03 loss per 5 deg)
39:STATIC_W_FLAT = 5.0    # static phases: soft 'lies flat on the support plane' term
started
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace("""# Phase boundaries (inclusive frame ranges). Static phases were found from the frame-difference signal inside
# the object region (H.motion_signal, low = static) and the mask centroid / bbox stability, and then verified
# by eye on fixed-position crops:""","""# Phase boundaries (inclusive frame ranges). Static phases were found from the frame-difference signal inside
# the object region (mean |I_t - I_{t-3}| over object-mask pixels not covered by the person; < ~3 grey levels
# = static) and the mask centroid / bbox stability, and then verified by eye on fixed-position crops:""")
open('solve.py','w').write(s)
EOF
sleep 90; cat /work/scratch/final.log
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
sleep 60; cat /work/scratch/final.log; tail -3 /work/scratch/final.time
```

<details><summary>output</summary>

```
sh: 1: /usr/bin/time: not found
```
</details>

**Bash**
```
pkill -f validate.py; pkill -f depth_cues.py; sleep 2; ps aux | grep -E "solve|validate|depth_cues" | grep -v grep | wc -l; cd /work/scratch; nohup sh -c "python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz --diag /work/scratch/final_diag.npz > final.log 2>&1; python /work/outputs/validate.py --result /work/outputs/object_poses_world.npz --md /work/scratch/final_val.md > final_val.log 2>&1; mv /work/outputs/object_poses_world_scores.npz /work/scratch/final_scores.npz; python /work/outputs/depth_cues.py --result /work/outputs/object_poses_world.npz > cues.log 2>&1" > /dev/null 2>&1 & echo started
```

<details><summary>output</summary>

```
Exit code 144
```
</details>

**Bash**
```
ps aux | grep -E "solve.py|validate.py|depth_cues.py" | grep -v grep | wc -l
```

<details><summary>output</summary>

```
0
```
</details>

**Bash**
```
cd /work/scratch; nohup sh -c "python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz --diag /work/scratch/final_diag.npz > final.log 2>&1; python /work/outputs/validate.py --result /work/outputs/object_poses_world.npz --md /work/scratch/final_val.md > final_val.log 2>&1; mv /work/outputs/object_poses_world_scores.npz /work/scratch/final_scores.npz; python /work/outputs/depth_cues.py --result /work/outputs/object_poses_world.npz > cues.log 2>&1" > /dev/null 2>&1 &
echo started
```

<details><summary>output</summary>

```
started
```
</details>

> *thinking:* The timing tool wasn't installed, so I had to restart the final run—it's running now with the chosen settings while I draft the report in parallel.

**Bash**
```
sleep 115; cat /work/scratch/final.log
```

<details><summary>output</summary>

```
loaded inputs in 20.0s
   static A: pooled baseline IoU 0.765 R2 0.548 | refined IoU 0.828 R2 0.718 -> refined
static A 0-79: IoU 0.822 rot change vs pooled baseline 2.7 deg  [24s]
   static B: pooled baseline IoU 0.848 R2 0.348 | refined IoU 0.844 R2 0.391 -> pooled baseline
static B 212-303: IoU 0.855 rot change vs pooled baseline 0.0 deg  [27s]
   static C: pooled baseline IoU 0.783 R2 0.708 | refined IoU 0.870 R2 0.809 -> refined
static C 405-466: IoU 0.870 rot change vs pooled baseline 3.3 deg  [30s]
   static D1: pooled baseline IoU 0.771 R2 0.622 | refined IoU 0.847 R2 0.630 -> refined
static D1 656-669: IoU 0.847 rot change vs pooled baseline 2.3 deg  [32s]
   static D2: pooled baseline IoU 0.922 R2 0.508 | refined IoU 0.953 R2 0.649 -> refined
static D2 695-747: IoU 0.955 rot chan …[+369 chars]
```
</details>

**Bash**
```
sleep 100; tail -2 /work/scratch/final.log; cat /work/scratch/final_val.md 2>/dev/null | head -3
```

<details><summary>output</summary>

```
moving 748-900: shifted 0.77 of frames, mean rot change vs baseline 6.1 deg  [129s]
done in 129s -> /work/outputs/object_poses_world.npz; peak GPU memory 4.92 GB
| phase | frames | type | IoU base | IoU ours | tex R2 base | tex R2 ours | rot speed base (med/p95 deg/fr) | rot speed ours | trans acc base (cm/fr^2, p95) | ours | dz mean (cm) | dz median (cm) | rot diff to base (mean deg) | bottom-to-floor base/ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0-79 | static | 0.786 | 0.827 | 0.569 | 0.811 | 0.06/1.48 | 0.00/0.00 | 1.91 | 0.00 | +0.27 | -0.00 | 2.0 | 89.3/88.8 |
```
</details>

**Bash**
```
sleep 70; cd /work/scratch; ls -la --time-style=+%T final_val.md cues.log final_scores.npz /work/outputs/object_poses_world.npz; ps aux | grep -E "validate.py|depth_cues.py" | grep -v grep | wc -l
```

<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 115606 14:57:44 /work/outputs/object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou    423 14:58:26 cues.log
-rw-rw-r-- 1 hanzhou hanzhou  38490 14:58:14 final_scores.npz
-rw-rw-r-- 1 hanzhou hanzhou   1738 14:58:14 final_val.md
0
```
</details>

**Bash**
```
cd /work/scratch; cat final_val.md cues.log; python -c "
import numpy as np, sys; sys.path.insert(0,'/work/outputs')
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(T.shape,T.dtype,np.isfinite(T).all())
s=np.load('final_scores.npz')
for a,b in [(580,600),(600,620)]: print(a,b,'iou %.2f->%.2f r2 %.2f->%.2f'%(s['b_siou'][a:b].mean(),s['r_siou'][a:b].mean(),s['b_ncc'][a:b].mean(),s['r_ncc'][a:b].mean()))
d=np.load('final_diag.npz'); print(d['names'], d['phases'].tolist())
"
```

<details><summary>output</summary>

```
| phase | frames | type | IoU base | IoU ours | tex R2 base | tex R2 ours | rot speed base (med/p95 deg/fr) | rot speed ours | trans acc base (cm/fr^2, p95) | ours | dz mean (cm) | dz median (cm) | rot diff to base (mean deg) | bottom-to-floor base/ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0-79 | static | 0.786 | 0.827 | 0.569 | 0.811 | 0.06/1.48 | 0.00/0.00 | 1.91 | 0.00 | +0.27 | -0.00 | 2.0 | 89.3/88.8 |
| M0 | 80-211 | moving | 0.694 | 0.820 | 0.466 | 0.528 | 1.92/18.73 | 1.58/4.98 | 8.90 | 1.00 | -0.70 | -0.69 | 6.7 | 71.7/66.1 |
| B | 212-303 | static | 0.891 | 0.850 | 0.337 | 0.364 | 0.10/1.32 | 0.00/0.00 | 7.14 | 0.00 | -0.38 | +0.00 | 3.5 | -0.5/0.4 |
| M1 | 304-404 | moving | 0.703 | 0.843 | 0.487 | 0.555 | 2.53/11.01 | 1.92/4.73 | 9.46 | 0.8 …[+1625 chars]
```
</details>

> *thinking:* Results look solid (IoU 0.783→0.843, texture R² improved, rotation jumps reduced), but I noticed the fallback for rejected static refinements uses a lie-flat pose that underperforms the un-flattened pooled pose (0.850 vs 0.886 IoU). I'll switch the fallback to the un-flattened version and rerun—GPU budget (~31 min used) allows it.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
old="""        T0 = H.mean_pose(T_init[a:b + 1])
        T0[:3, 3] *= zt[a] / T0[2, 3]
        T0 = H.flatten_pose(T0, n_cam, R.verts_np.mean(0))"""
new="""        T_pool = H.mean_pose(T_init[a:b + 1])
        T_pool[:3, 3] *= zt[a] / T_pool[2, 3]
        T0 = H.flatten_pose(T_pool, n_cam, R.verts_np.mean(0))"""
assert old in s; s=s.replace(old,new)
old="""        s_b = H.score_traj(fit_plain, np.repeat(T0[None], len(allf), 0), allf)"""
assert old in s; s=s.replace(old,"""        s_b = H.score_traj(fit_plain, np.repeat(T_pool[None], len(allf), 0), allf)""")
old="""        if not accept:
            Ts = T0"""
assert old in s; s=s.replace(old,"""        if not accept:      # fall back to the pooled baseline pose (not flattened)
            Ts = T_pool""")
s=s.replace("""        # accept the refined pose only if it explains ALL frames of the phase better than the pooled baseline
        # pose under""","""        # accept the refined pose only if it explains ALL frames of the phase better than the pooled baseline
        # pose (chordal-mean rotation, median position; not flattened) under""")
s=s.replace("""f'{H.rot_angle_deg(T0[:3, :3], Ts[:3, :3]):.1f} deg""","""f'{H.rot_angle_deg(T_pool[:3, :3], Ts[:3, :3]):.1f} deg""")
open('solve.py','w').write(s)
EOF
cd /work/scratch; nohup sh -c "python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz --diag /work/scratch/final_diag.npz >  …[+357 chars]
```

<details><summary>output</summary>

```
started
```
</details>

**Bash**
```
sleep 118; cat /work/scratch/final.log | grep -E "static|done"
```

<details><summary>output</summary>

```
   static A: pooled baseline IoU 0.783 R2 0.567 | refined IoU 0.732 R2 0.592 -> pooled baseline
static A 0-79: IoU 0.723 rot change vs pooled baseline 0.0 deg  [28s]
   static B: pooled baseline IoU 0.882 R2 0.351 | refined IoU 0.874 R2 0.360 -> pooled baseline
static B 212-303: IoU 0.886 rot change vs pooled baseline 0.0 deg  [32s]
   static C: pooled baseline IoU 0.796 R2 0.708 | refined IoU 0.843 R2 0.747 -> refined
static C 405-466: IoU 0.854 rot change vs pooled baseline 3.4 deg  [35s]
   static D1: pooled baseline IoU 0.783 R2 0.621 | refined IoU 0.847 R2 0.610 -> pooled baseline
static D1 656-669: IoU 0.847 rot change vs pooled baseline 0.0 deg  [38s]
   static D2: pooled baseline IoU 0.941 R2 0.553 | refined IoU 0.951 R2 0.621 -> refined
static D2 695-747: IoU 0.952 rot change vs p …[+29 chars]
```
</details>

> *thinking:* The static refinement proved unstable across runs—this time phase A got stuck in a bad local minimum (IoU 0.73 vs. 0.83–0.84 previously) due to non-deterministic GPU gradients and a large step size, though the fallback check caught it. I'm stopping this run to make the fit more robust: using smaller steps and three joint yaw starts (−4°, 0°, +4°), keeping whichever performs best.

**Bash**
```
pkill -f "outputs/solve.py" ; sleep 3; ps aux | grep -E "outputs/(solve|validate|depth_cues).py" | grep -v grep | awk '{print $2, $12, $13}'
```

<details><summary>output</summary>

```
Exit code 144
```
</details>

**Bash**
```
ps aux | grep -E "(solve|validate|depth_cues)\.py" | grep -v grep | awk '{print $2, $13, $14}'
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
old="""        fr = H.pick_frames(PM, OM, range(a, b + 1), 16)
        p = K @ T0[:3, 3]
        crop = obs.crop_for(0, (p[0] / p[2], p[1] / p[2]))
        Tf, terms = fit.fit(T0[None], fr, pid=np.zeros(len(fr), int), iters=100, crops=[crop] * len(fr),
                            T_prior=True, w_prior=STATIC_W_PRIOR, flat_n=n_cam, w_flat=STATIC_W_FLAT)
        Ts = Tf[0]"""
new="""        # multi-start (yaw -4/0/+4 deg about the support normal), small steps: the GPU rasteriser gradients are
        # not bit-deterministic, a single start occasionally ends in a poor local minimum
        fr = H.pick_frames(PM, OM, range(a, b + 1), 12)
        p = K @ T0[:3, 3]
        crop = obs.crop_for(0, (p[0] / p[2], p[1] / p[2]))
        c0 = R.verts_np.mean(0)
        starts = []
        for ang in STATIC_YAW_STARTS:
            g = H.axis_angle(n_cam, np.radians(ang))
            Tsd = T0.copy(); Tsd[:3, :3] = g @ T0[:3, :3]
            Tsd[:3, 3] = T0[:3, 3] + T0[:3, :3] @ c0 - Tsd[:3, :3] @ c0
            starts.append(Tsd)
        nS = len(starts)
        Tf, terms = fit.fit(np.stack(starts), np.tile(fr, nS), pid=np.repeat(np.arange(nS), len(fr)), iters=120,
                            lr_r=0.01, lr_t=0.002, crops=[crop] * (len(fr) * nS), T_prior=True,
                            w_prior=STATIC_W_PRIOR, flat_n=n_cam, w_flat=STATIC_W_FLAT)
        Ls = (terms['sil'] + 0.3 * (1 - terms['ncc']) + H.W_CHAM * terms['cham']).resh …[+1124 chars]
```

<details><summary>output</summary>

```
127:        Ls = (terms['sil'] + 0.3 * (1 - terms['ncc']) + H.W_CHAM * terms['cham']).reshape(nS, len(fr)).mean(1)
loaded inputs in 22.5s
   static A: pooled baseline IoU 0.783 R2 0.567 | refined IoU 0.838 R2 0.816 -> refined
static A 0-79: start losses [0.266 0.203 0.223] rot change vs pooled baseline 1.9 deg  [31s]
   static B: pooled baseline IoU 0.882 R2 0.351 | refined IoU 0.868 R2 0.362 -> pooled baseline
static B 212-303: start losses [0.316 0.327 0.328] rot change vs pooled baseline 0.0 deg  [39s]
   static C: pooled baseline IoU 0.796 R2 0.708 | refined IoU 0.853 R2 0.825 -> refined
static C 405-466: start losses [0.174 0.19  0.179] rot change vs pooled baseline 3.9 deg  [46s]
   static D1: pooled baseline IoU 0.783 R2 0.621 | refined IoU 0.861 R2 0.610 -> pooled baseline
static D …[+88 chars]
```
</details>

**Bash**
```
sleep 110; tail -7 /work/scratch/final.log
```

<details><summary>output</summary>

```
static D1 656-669: start losses [0.287 0.301 0.314] rot change vs pooled baseline 0.0 deg  [52s]
   static D2: pooled baseline IoU 0.941 R2 0.553 | refined IoU 0.953 R2 0.619 -> refined
static D2 695-747: start losses [0.158 0.146 0.166] rot change vs pooled baseline 0.6 deg  [59s]
moving 80-211: shifted 0.86 of frames, mean rot change vs baseline 6.8 deg  [88s]
moving 304-404: shifted 0.81 of frames, mean rot change vs baseline 7.4 deg  [110s]
moving 467-655: shifted 0.92 of frames, mean rot change vs baseline 6.3 deg  [146s]
moving 670-694: shifted 0.52 of frames, mean rot change vs baseline 4.0 deg  [150s]
```
</details>

**Bash**
```
sleep 100; tail -1 /work/scratch/final.log; ls -la --time-style=+%T /work/scratch/final_val.md /work/scratch/cues.log
```

<details><summary>output</summary>

```
done in 181s -> /work/outputs/object_poses_world.npz; peak GPU memory 5.39 GB
-rw-rw-r-- 1 hanzhou hanzhou  423 15:07:45 /work/scratch/cues.log
-rw-rw-r-- 1 hanzhou hanzhou 1740 15:07:31 /work/scratch/final_val.md
```
</details>

**Bash**
```
cd /work/scratch; date +%T; cat final_val.md cues.log; python -c "
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(T.shape,T.dtype,np.isfinite(T).all(), 'max |det-1|', np.abs(np.linalg.det(T[:,:3,:3])-1).max())
"; ls /work/outputs
```

<details><summary>output</summary>

```
15:14:35
| phase | frames | type | IoU base | IoU ours | tex R2 base | tex R2 ours | rot speed base (med/p95 deg/fr) | rot speed ours | trans acc base (cm/fr^2, p95) | ours | dz mean (cm) | dz median (cm) | rot diff to base (mean deg) | bottom-to-floor base/ours (cm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0-79 | static | 0.786 | 0.837 | 0.569 | 0.815 | 0.06/1.48 | 0.01/0.01 | 1.91 | 0.00 | +0.27 | -0.00 | 2.1 | 89.3/88.6 |
| M0 | 80-211 | moving | 0.694 | 0.806 | 0.466 | 0.506 | 1.92/18.73 | 1.65/5.82 | 8.90 | 1.07 | -0.70 | -0.69 | 6.8 | 71.7/65.9 |
| B | 212-303 | static | 0.891 | 0.884 | 0.337 | 0.353 | 0.10/1.32 | 0.00/0.00 | 7.14 | 0.00 | -0.38 | +0.00 | 1.6 | -0.5/-0.4 |
| M1 | 304-404 | moving | 0.703 | 0.839 | 0.487 | 0.561 | 2.53/11.01 | 1.86/4.30 |  …[+1520 chars]
```
</details>

**Agent:** Final output is written and checked: 901×4×4 float64, all finite, proper rotations. GPU use is about 38 of 45 minutes by my estimate, so I'm stopping GPU work and writing the report.

**Bash**
```
cd /work/scratch; grep -A9 "moving segment" ho_wp1.val; grep -A9 "moving segment" ho_1.0.val | tail -1; rm -rf /work/outputs/__pycache__
```

<details><summary>output</summary>

```
| moving segment | held-out (odd) frames | IoU base | IoU ours | tex R2 base | tex R2 ours | chamfer base | ours |
|---|---|---|---|---|---|---|---|
| M0 | 65 | 0.700 | 0.794 | 0.467 | 0.475 | 0.0426 | 0.0037 |
| M1 | 49 | 0.698 | 0.825 | 0.490 | 0.583 | 0.0076 | 0.0033 |
| M2 | 90 | 0.737 | 0.817 | 0.610 | 0.655 | 0.0590 | 0.0253 |
| M3 | 12 | 0.905 | 0.901 | 0.610 | 0.620 | 0.0213 | 0.0155 |
| M4 | 72 | 0.823 | 0.830 | 0.369 | 0.557 | 0.0091 | 0.0080 |
| ALL | 288 | 0.751 | 0.820 | 0.497 | 0.576 | 0.0325 | 0.0119 |
| ALL | 288 | 0.751 | 0.859 | 0.497 | 0.539 | 0.0325 | 0.0105 |
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
cd /work/outputs; grep -n "scratch" solve.py hoi_lib.py; ls -la; python -c "import ast,sys; [ast.parse(open(f).read()) for f in ['solve.py','hoi_lib.py','validate.py','depth_cues.py']]; print('syntax ok')"
```

<details><summary>output</summary>

```
total 196
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 21 15:09 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 21 13:51 ..
-rw-rw-r-- 1 hanzhou hanzhou  14039 Oct 21 15:09 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   2237 Oct 21 14:47 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou   3717 Oct 21 14:42 depth_cues.py
-rw-rw-r-- 1 hanzhou hanzhou  25787 Oct 21 14:31 hoi_lib.py
-rw-rw-r-- 1 hanzhou hanzhou 115606 Oct 21 15:06 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou  11652 Oct 21 15:03 solve.py
-rw-rw-r-- 1 hanzhou hanzhou   4907 Oct 21 14:31 validate.py
syntax ok
```
</details>

**Agent:** I've written the result to `/work/outputs/object_poses_world.npz` (901×4×4 float64). Against the baseline it improves image agreement overall, and on frames held out from the fitting. It isn't uniformly better: some frames around 585–620 may have slightly worse rotation than the baseline.

**What the video actually shows.** The task says "table to floor", but the book goes table → floor → table → floor (with a slide on the floor) and ends up held behind the far table edge. That gives five static phases and five carried segments. I set the phase boundaries as commented constants in `solve.py`, from a frame-difference signal and checked by eye.

**What was wrong with the baseline.**
- **Symmetry flips:** none. It never flips the book 180° between frames, and the visible spine logo matches its orientation, so I kept its orientation everywhere.
- **Position errors:** where the book passes behind or under the table edge (around frames 140, 330 and 580), the baseline is off by up to about 7 cm. Near 580 it hides the book behind the table top while the video shows it underneath.
- **Jumps:** rotation jumps up to 19° per frame during occlusions, and 1–3 cm of depth jitter in static phases.

**What the solver does.**
- **Static phases:** one constant pose per phase, fitted over many frames by comparing renders of the textured mesh with the image, kept flat on the floor or table. An automatic rule keeps the refined pose only if it beats the averaged baseline pose. It did for three phases; the other two (floor phases B and D1) use the averaged baseline pose.
- **Carried segments:** a search for image shifts that explain the mask, then a joint smooth trajectory fit. Hands are treated as occluders, and so is the table, using the existing MoGe depth maps.
- **Depth:** every phase is re-anchored to the baseline's median depth. The independent cues (floor contact, MoGe depth, hand position) disagree with each other, so the applied depth shift is 0.

**Validation (no ground truth).**

| measure | baseline | result |
|---|---|---|
| Mask IoU, all frames | 0.783 | 0.842 |
| Texture agreement (R²), all frames | 0.509 | 0.596 |
| 95th-percentile rotation change per frame | 11.8° | 4.3° |
| Held-out carried frames (288): IoU | 0.751 | 0.820 |
| Held-out carried frames: texture R² | 0.497 | 0.576 |

- Held-out carried frames improve on both measures in every segment, except a negligible −0.004 IoU in one 12-frame segment.
- Static-phase numbers are measured on the frames used for fitting, so they are not an independent test.
- Mean depth difference from the baseline is −0.2 cm overall and within ±0.7 cm per phase.

**Weak spots.**
- **Frames ~585–620:** silhouette fit improved but texture agreement is below the baseline, so the rotation may be slightly worse there.
- **Frames 830–900:** the book is mostly hidden by the table; the result stays near the baseline and the hand positions, but confidence is low.
- **Cover side:** front versus back cover can't be told apart at this resolution, so I kept the baseline's choice.
- **Reproducibility:** repeated runs are not bit-identical, because the renderer's gradients use GPU atomic additions. Differences are typically under 1–2°. Multi-start fitting and the accept rule keep this from changing the result much.

The final `solve.py` run took 181 s with 5.4 GB peak GPU memory. My estimate of total GPU time across all development runs is about 38 of the allowed 45 minutes, counted as process run time rather than measured. Nothing external was downloaded.

Files are in `/work/outputs`:
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoi_lib.py`
- `validate.py`
- `depth_cues.py`
- `object_poses_world.npz`

## Final

I've written the result to `/work/outputs/object_poses_world.npz` (901×4×4 float64). Against the baseline it improves image agreement overall, and on frames held out from the fitting. It isn't uniformly better: some frames around 585–620 may have slightly worse rotation than the baseline.

**What the video actually shows.** The task says "table to floor", but the book goes table → floor → table → floor (with a slide on the floor) and ends up held behind the far table edge. That gives five static phases and five carried segments. I set the phase boundaries as commented constants in `solve.py`, from a frame-difference signal and checked by eye.

**What was wrong with the baseline.**
- **Symmetry flips:** none. It never flips the book 180° between frames, and the visible spine logo matches its orientation, so I kept its orientation everywhere.
- **Position errors:** where the book passes behind or under the table edge (around frames 140, 330 and 580), the baseline is off by up to about 7 cm. Near 580 it hides the book behind the table top while the video shows it underneath.
- **Jumps:** rotation jumps up to 19° per frame during occlusions, and 1–3 cm of depth jitter in static phases.

**What the solver does.**
- **Static phases:** one constant pose per phase, fitted over many frames by comparing renders of the textured mesh with the image, kept flat on the floor or table. An automatic rule keeps the refined pose only if it beats the averaged baseline pose. It did for three phases; the other two (floor phases B and D1) use the averaged baseline pose.
- **Carried segments:** a search for image shifts that explain the mask, then a joint smooth trajectory fit. Hands are treated as occluders, and so is the table, using the existing MoGe depth maps.
- **Depth:** every phase is re-anchored to the baseline's median depth. The independent cues (floor contact, MoGe depth, hand position) disagree with each other, so the applied depth shift is 0.

**Validation (no ground truth).**

| measure | baseline | result |
|---|---|---|
| Mask IoU, all frames | 0.783 | 0.842 |
| Texture agreement (R²), all frames | 0.509 | 0.596 |
| 95th-percentile rotation change per frame | 11.8° | 4.3° |
| Held-out carried frames (288): IoU | 0.751 | 0.820 |
| Held-out carried frames: texture R² | 0.497 | 0.576 |

- Held-out carried frames improve on both measures in every segment, except a negligible −0.004 IoU in one 12-frame segment.
- Static-phase numbers are measured on the frames used for fitting, so they are not an independent test.
- Mean depth difference from the baseline is −0.2 cm overall and within ±0.7 cm per phase.

**Weak spots.**
- **Frames ~585–620:** silhouette fit improved but texture agreement is below the baseline, so the rotation may be slightly worse there.
- **Frames 830–900:** the book is mostly hidden by the table; the result stays near the baseline and the hand positions, but confidence is low.
- **Cover side:** front versus back cover can't be told apart at this resolution, so I kept the baseline's choice.
- **Reproducibility:** repeated runs are not bit-identical, because the renderer's gradients use GPU atomic additions. Differences are typically under 1–2°. Multi-start fitting and the accept rule keep this from changing the result much.

The final `solve.py` run took 181 s with 5.4 GB peak GPU memory. My estimate of total GPU time across all development runs is about 38 of the allowed 45 minutes, counted as process run time rather than measured. Nothing external was downloaded.

Files are in `/work/outputs`:
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoi_lib.py`
- `validate.py`
- `depth_cues.py`
- `object_poses_world.npz`
