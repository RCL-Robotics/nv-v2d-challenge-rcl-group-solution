# CL1_ep011

Claude Opus 完整轨迹，第一批  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-20T21:10:59Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; nvidia-smi; ls /pipeline /workspace /code/reconstruction /weights
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
total 93764
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 01:04 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 01:04 ..
-rw-rw-r-- 1 hanzhou hanzhou      822 Oct 21 01:04 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       70 Oct 21 01:04 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   193877 Oct 21 01:04 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      395 Oct 21 01:04 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  6116444 Oct 21 01:04 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 10541188 Oct 21 01:04 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 4589 …[+4673 chars]
```
</details>

**Bash**
```
cd /work; echo $CUDA_VISIBLE_DEVICES; cat logs/run_manifest.txt; cat /pipeline/pipeline_report.json | head -100; find /pipeline -maxdepth 3 | head -80
```

<details><summary>output</summary>

```
0
date_utc: 2026-10-20T21:10:59Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL1_ep011:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000011:/pipeline:ro
task_sha256: ed1b646f904146c1d2901e69c6c66646f6d940bf18f332777f03f95517c92df6
45f7d8c8d17cc21190f7a0cdd3b90a98c330420689d6a19915acf56bf4e46960  ./camera.json
ee19e46c20c013dcb9b14898d966f1ca417181f8515e2381dded25333ddde474  ./episode.json
27f695d586d79e2213a2ff3b3bd077b4f4b5b5a64774492137e4e5 …[+5517 chars]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/code tools; which python; python -c "import torch, nvdiffrast, cv2, trimesh, h5py; print(torch.__version__, cv2.__version__, trimesh.__version__, torch.cuda.is_available())"; nproc; free -g; df -h /work | tail -1
```

**Bash**
```
cd /work; python -I - <<'EOF'
import numpy as np, trimesh, h5py
m = trimesh.load('inputs/object_mesh.glb', force='scene', process=False).to_geometry()
print(m, m.bounds, m.extents)
print(type(m.visual), getattr(m.visual,'material',None))
S,ext = trimesh.bounds.oriented_bounds(m); print(S, ext)
r = np.load('inputs/reconstruction.npz')
for k in r.files: print(k, r[k].shape, r[k].dtype)
h = h5py.File('inputs/masks.h5','r')
for k in h: print(k, h[k].shape, h[k].dtype, h[k].chunks)
EOF
```

<details><summary>output</summary>

```
/opt/conda/bin/python
2.5.1+cu124 4.11.0 5.1.1 True
48
               total        used        free      shared  buff/cache   available
Mem:             125          22          10           0          91         101
Swap:              1           1           0
10.127.102.71:/Group/hanzhou/v2d/agent_runs/CL1_ep011  139T   49T   90T  36% /work
```
</details>

<details><summary>output</summary>

```
<trimesh.Trimesh(vertices.shape=(271275, 3), faces.shape=(90566, 3))> [[-0.09479323 -0.02947056 -0.13077636]
 [ 0.09639004  0.02936917  0.1300355 ]] [0.19118327 0.05883973 0.26081187]
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x798bf7e1ba90>
[[ 1.22464680e-16  1.00000000e+00  0.00000000e+00  5.06974757e-05]
 [ 1.00000000e+00 -1.22464680e-16  0.00000000e+00 -7.98404217e-04]
 [ 0.00000000e+00  0.00000000e+00 -1.00000000e+00 -3.70427966e-04]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] [0.05883973 0.19118327 0.26081187]
T_world_mesh (901, 4, 4) float32
T_cam_mesh_foundationpose (901, 4, 4) float32
T_cam_mesh_coconet (901, 4, 4) float32
human_vertices_world (901, 4610, 3) float32
human_keypoints_world (901, 70, 3) floa …[+130 chars]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/frames; ffprobe -v error -show_streams inputs/video.mp4 | grep -E "r_frame_rate|nb_frames|width|height|codec_name"; time ffmpeg -v error -i inputs/video.mp4 -q:v 2 scratch/frames/%05d.jpg; ls scratch/frames | wc -l
```

<details><summary>output</summary>

```
codec_name=h264
width=1536
height=1152
coded_width=1536
coded_height=1152
r_frame_rate=30/1
nb_frames=901

real	0m7.909s
user	0m22.743s
sys	0m0.582s
901
```
</details>

**Bash**
```
cd /work; cat > scratch/code/stats.py <<'EOF'
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R
r = np.load('inputs/reconstruction.npz')
T = r['T_world_mesh'].astype(np.float64)
h = h5py.File('inputs/masks.h5','r')
F = T.shape[0]
oa = np.zeros(F); pa=np.zeros(F); bb=np.zeros((F,4))
for s in range(0,F,29):
    o = h['object'][s:s+29]; p=h['person'][s:s+29]
    for i in range(o.shape[0]):
        oa[s+i]=o[i].sum(); pa[s+i]=p[i].sum()
        ys,xs=np.nonzero(o[i])
        if len(xs): bb[s+i]=[xs.min(),ys.min(),xs.max(),ys.max()]
np.savez('scratch/mask_stats.npz', oa=oa, pa=pa, bb=bb)
c = r['contact_prob_right_left']
dR = np.r_[0, np.degrees(np.linalg.norm(R.from_matrix(T[1:,:3,:3].transpose(0,2,1)@T[:-1,:3,:3]).as_rotvec(),axis=1))]
dt = np.r_[0, np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)]*100
for f in range(0,F,10):
    print(f, int(oa[f]), bb[f].astype(int), np.round(c[f],2), np.round(T[f,:3,3],3), round(dR[f],1), round(dt[f],2))
EOF
python scratch/code/stats.py
```

<details><summary>output</summary>

```
0 1929 [768 598 851 626] [0. 0.] [0.375 0.284 3.108] 0.0 0.0
10 1944 [768 598 852 626] [0. 0.] [0.375 0.284 3.109] 0.0 0.01
20 1972 [768 598 853 626] [0. 0.] [0.375 0.284 3.108] 0.0 0.0
30 1966 [768 598 853 626] [0. 0.] [0.374 0.283 3.102] 0.0 0.04
40 1964 [768 598 853 626] [0. 0.] [0.373 0.282 3.085] 0.0 0.0
50 1871 [769 598 852 626] [0. 0.] [0.372 0.282 3.084] 0.0 0.01
60 1957 [768 598 852 626] [0. 0.] [0.372 0.282 3.081] 0.0 0.0
70 1811 [776 598 852 626] [1. 1.] [0.387 0.29  3.132] 1.2 0.73
80 1823 [776 596 852 626] [1. 1.] [0.391 0.29  3.143] 1.3 1.75
90 2026 [777 590 850 625] [1. 1.] [0.383 0.272 3.11 ] 1.2 0.95
100 2175 [772 580 843 620] [1. 1.] [0.355 0.241 3.139] 1.4 1.45
110 1432 [742 575 806 604] [1. 1.] [0.231 0.195 3.204] 4.2 2.43
120 1136 [691 575 752 596] [1. 1.] [0.044 0.21  …[+5195 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/sheet.py <<'EOF'
import numpy as np, cv2, sys
frames = list(range(0,901,30))
imgs=[]
for f in frames:
    im = cv2.imread(f'scratch/frames/{f+1:05d}.jpg')
    im = cv2.resize(im,(384,288)); cv2.putText(im,str(f),(5,25),0,0.8,(0,0,255),2); imgs.append(im)
while len(imgs)%6: imgs.append(np.zeros_like(imgs[0]))
rows=[np.hstack(imgs[i:i+6]) for i in range(0,len(imgs),6)]
cv2.imwrite('scratch/sheet_full.jpg', np.vstack(rows))
EOF
python scratch/code/sheet.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sheet_full.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/common.py <<'EOF'
import numpy as np, json, trimesh, torch, nvdiffrast.torch as dr, h5py, cv2
INP='/work/inputs'
def load_cam(inp=INP):
    c=json.load(open(f'{inp}/camera.json'))
    return np.array(c['K']), np.array(c['T_world_cam']), np.array(c['K_pipeline'])
def load_mesh(inp=INP):
    m = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    v = np.asarray(m.vertices, np.float32); f = np.asarray(m.faces, np.int32)
    uv = np.asarray(m.visual.uv, np.float32)
    mat = m.visual.material
    tex = np.asarray(mat.baseColorTexture.convert('RGB'), np.float32)/255. if getattr(mat,'baseColorTexture',None) is not None else None
    return m, v, f, uv, tex
class Renderer:
    def __init__(self, v, f, uv, tex, dev='cuda'):
        self.dev=dev
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(v, device=dev); self.f=torch.tensor(f, device=dev)
        self.uv = torch.tensor(uv, device=dev)
        self.uv_t = self.uv.clone(); self.uv_t[:,1] = 1-self.uv_t[:,1]
        self.tex = torch.tensor(tex, device=dev)[None].contiguous()
    def render(self, T_cam_mesh, K, H, W, x0=0, y0=0, need_rgb=True):
        """T_cam_mesh (B,4,4) torch; K 3x3 np; render crop window [x0,x0+W)x[y0,y0+H) of the full image."""
        B = T_cam_mesh.shape[0]
        fx,fy,cx,cy = K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0
        n,fa=0.05,20.
        P = torch.tensor([[2*fx/W,0,2*cx/W-1,0],[0,2*fy/H,2*cy/H-1,0],[0,0,(fa+n)/(fa-n …[+2528 chars]
```

<details><summary>output</summary>

```
(2332, 2168, 3)
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

> *thinking:* The book is small and distant in frame, and while the baseline silhouette matches well when static, some carried frames (like 150) show noticeable mismatches. I'll now check whether the cover texture reveals a front/back flip.

**Bash**
```
cd /work; cat > scratch/code/ov2.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp = load_cam(); m,v,f,uv,tex = load_mesh()
R = Renderer(v,f,uv,tex)
r = np.load(f'{INP}/reconstruction.npz'); Tw = r['T_world_mesh'].astype(np.float64)
h = h5py.File(f'{INP}/masks.h5','r')
frames=[int(x) for x in sys.argv[1].split(',')]
tiles=[]
for fr in frames:
    Tc = np.linalg.inv(Twc)@Tw[fr]
    om = h['object'][fr]; ys,xs = np.nonzero(om)
    cx,cy = int(xs.mean()), int(ys.mean()); S=100
    x0,y0 = cx-S//2, cy-S//2
    out = R.render(torch.tensor(Tc[None],device='cuda'), K, S,S,x0,y0)
    rgb = (out['rgb'][0].cpu().numpy()*255).astype(np.uint8)
    img = load_frame(fr)[y0:y0+S,x0:x0+S]
    t = np.vstack([img,rgb]); t=cv2.resize(t,(S*4,S*8),interpolation=cv2.INTER_CUBIC)
    cv2.putText(t,str(fr),(3,20),0,0.7,(255,255,0),2); tiles.append(t)
cv2.imwrite(sys.argv[2], np.hstack(tiles)[...,::-1])
EOF
python scratch/code/ov2.py 0,100,210,260,420 scratch/ov2.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/ov2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/canon.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
from scipy.spatial.transform import Rotation as Rot
m,v,f,uv,tex = load_mesh(); R=Renderer(v,f,uv,tex)
cv2.imwrite('/work/scratch/tex.jpg', cv2.resize((tex[...,::-1]*255).astype(np.uint8),(1084,1166)))
K=np.array([[1500,0,200],[0,1500,200],[0,0,1.]])
tiles=[]
# view along each of +-x,+-y,+-z axes (camera looking at face)
for ax,lab in [((0,0,0),'cam z->mesh+z'),((0,180,0),'mesh -z'),((0,90,0),'y90'),((0,-90,0),'y-90'),((90,0,0),'x90'),((-90,0,0),'x-90'),((30,30,0),'oblique')]:
    T=np.eye(4); T[:3,:3]=Rot.from_euler('xyz',ax,degrees=True).as_matrix(); T[2,3]=1.2
    o=R.render(torch.tensor(T[None],device='cuda'),K,400,400)
    im=(o['rgb'][0].cpu().numpy()*255).astype(np.uint8)[...,::-1].copy()
    cv2.putText(im,lab,(3,20),0,0.6,(0,0,255),2); tiles.append(im)
cv2.imwrite('/work/scratch/canon.jpg',np.hstack(tiles))
EOF
python scratch/code/canon.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/canon.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/tex.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/cropgrid.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp = load_cam()
r = np.load(f'{INP}/reconstruction.npz'); Tw = r['T_world_mesh'].astype(np.float64)
h = h5py.File(f'{INP}/masks.h5','r')
a,b,st = map(int,sys.argv[1:4]); S=int(sys.argv[5]) if len(sys.argv)>5 else 140
tiles=[]
for fr in range(a,b,st):
    Tc=np.linalg.inv(Twc)@Tw[fr]; p=K@Tc[:3,3]; cx,cy=int(p[0]/p[2]),int(p[1]/p[2])
    img=load_frame(fr)
    x0,y0=min(max(cx-S//2,0),1536-S),min(max(cy-S//2,0),1152-S)
    c=img[y0:y0+S,x0:x0+S].copy()
    om=h['object'][fr][y0:y0+S,x0:x0+S].astype(np.uint8)
    cnt,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(c,cnt,-1,(0,255,0),1)
    c=cv2.resize(c,(2*S,2*S)); cv2.putText(c,str(fr),(3,22),0,0.8,(255,255,0),2); tiles.append(c)
n=10
while len(tiles)%n: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[4], np.vstack([np.hstack(tiles[i:i+n]) for i in range(0,len(tiles),n)])[...,::-1])
EOF
python scratch/code/cropgrid.py 0 300 10 scratch/g1.jpg; python scratch/code/cropgrid.py 300 600 10 scratch/g2.jpg; python scratch/code/cropgrid.py 600 901 10 scratch/g3.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/g1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/g2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/g3.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/rel.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
from scipy.spatial.transform import Rotation as Rot
K,Twc,Kp = load_cam()
r = np.load(f'{INP}/reconstruction.npz'); Tw = r['T_world_mesh'].astype(np.float64)
Tb = np.linalg.inv(Twc)@Tw
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64); Tco=r['T_cam_mesh_coconet'].astype(np.float64)
def ang(A,B): return np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum('nji,njk->nik',A[:,:3,:3],B[:,:3,:3])).as_rotvec(),axis=1))
a1=ang(Tb,Tco); a2=ang(Tb,Tf); a3=ang(Tco,Tf)
for f in range(0,901,25):
    pb=K@Tb[f,:3,3]; pc=Kp@Tco[f,:3,3]; pf=Kp@Tf[f,:3,3]
    print(f, 'rot b-co %.1f b-fp %.1f co-fp %.1f'%(a1[f],a2[f],a3[f]), 'z b %.2f co %.2f fp %.2f'%(Tb[f,2,3],Tco[f,2,3],Tf[f,2,3]), 'px b',(pb[:2]/pb[2]).astype(int),'co',(pc[:2]/pc[2]).astype(int),'fp',(pf[:2]/pf[2]).astype(int))
print('median rot diffs', np.median(a1), np.median(a2), np.percentile(a1,90))
EOF
python scratch/code/rel.py
```

<details><summary>output</summary>

```
0 rot b-co 1.3 b-fp 2.4 co-fp 1.6 z b 3.14 co 3.09 fp 3.21 px b [813 609] co [813 609] fp [809 610]
25 rot b-co 1.3 b-fp 2.2 co-fp 1.3 z b 3.14 co 3.09 fp 3.22 px b [813 609] co [813 609] fp [810 610]
50 rot b-co 1.3 b-fp 2.2 co-fp 1.6 z b 3.12 co 3.09 fp 3.19 px b [813 609] co [813 609] fp [809 610]
75 rot b-co 1.3 b-fp 2.9 co-fp 2.0 z b 3.17 co 3.16 fp 3.21 px b [814 611] co [814 611] fp [811 610]
100 rot b-co 1.3 b-fp 3.6 co-fp 2.7 z b 3.17 co 3.14 fp 3.21 px b [807 597] co [807 597] fp [805 598]
125 rot b-co 1.3 b-fp 11.8 co-fp 11.2 z b 3.27 co 3.20 fp 3.20 px b [704 595] co [704 595] fp [704 595]
150 rot b-co 1.3 b-fp 26.0 co-fp 25.3 z b 3.39 co 3.46 fp 7.04 px b [659 653] co [659 653] fp [635 659]
175 rot b-co 1.3 b-fp 8.1 co-fp 7.8 z b 3.39 co 3.37 fp 2.91 px b [660 764] co [660 764 …[+3064 chars]
```
</details>

**Bash**
```
cd /work; cat >> scratch/code/common.py <<'EOF'

import torch.nn.functional as Fn
from scipy.spatial.transform import Rotation as Rot
FLIPS = {'id':np.eye(4)}
for ax,name in [(0,'fx'),(1,'fy'),(2,'fz')]:
    M=np.eye(4); M[:3,:3]=Rot.from_rotvec(np.pi*np.eye(3)[ax]).as_matrix(); FLIPS[name]=M
class Scorer:
    """Score pose hypotheses against frame evidence in a crop window."""
    def __init__(self, R, K, ss=3):
        self.R=R; self.K=K; self.ss=ss
    def render_crop(self, Tc, x0,y0,S):
        ss=self.ss; Ks=self.K.copy(); Ks[:2]*=ss; Ks[0,2]+=(ss-1)/2; Ks[1,2]+=(ss-1)/2  # pixel-center consistent
        o=self.R.render(Tc, Ks, S*ss,S*ss, x0*ss, y0*ss)
        m=Fn.avg_pool2d(o['mask'].float()[:,None],ss)[:,0]
        rgb=Fn.avg_pool2d(o['rgb'].permute(0,3,1,2),ss)
        rgb=rgb/m[:,None].clamp(min=1e-6)
        d=Fn.avg_pool2d(o['depth'][:,None],ss)[:,0]/m.clamp(min=1e-6)
        return m,rgb,d
    def score(self, Tc, img, om, pm, x0,y0,S):
        """Tc (B,4,4) torch; img (S,S,3) float 0-1 torch; om, pm (S,S) bool torch."""
        m,rgb,_=self.render_crop(Tc,x0,y0,S)
        valid=(~pm).float()
        omf=om.float()
        inter=(m*omf*valid).sum((1,2)); uni=((m+omf-m*omf)*valid).sum((1,2))
        iou=inter/uni.clamp(min=1)
        # appearance NCC on gray, inside rendered & observed & not person
        g_r=rgb.mean(1); g_i=img.mean(-1)[None]
        w=(m>0.99).float()*omf*valid
        def ncc(a,b,w):
            ws=w.sum((1,2)).clamp(min=1)
            am=(a* …[+1274 chars]
```

<details><summary>output</summary>

```
0 id:iou0.79/ncc0.58 fx:iou0.80/ncc0.42 fy:iou0.79/ncc0.20 fz:iou0.78/ncc0.31 1538
20 id:iou0.80/ncc0.56 fx:iou0.80/ncc0.35 fy:iou0.79/ncc0.20 fz:iou0.79/ncc0.30 1554
40 id:iou0.80/ncc0.53 fx:iou0.80/ncc0.35 fy:iou0.80/ncc0.19 fz:iou0.79/ncc0.28 1569
60 id:iou0.81/ncc0.55 fx:iou0.81/ncc0.42 fy:iou0.80/ncc0.30 fz:iou0.80/ncc0.36 1565
80 id:iou0.85/ncc0.55 fx:iou0.86/ncc0.46 fy:iou0.84/ncc0.09 fz:iou0.83/ncc0.15 1513
100 id:iou0.80/ncc0.07 fx:iou0.80/ncc-0.22 fy:iou0.80/ncc-0.54 fz:iou0.80/ncc-0.42 1720
120 id:iou0.89/ncc0.45 fx:iou0.89/ncc0.38 fy:iou0.85/ncc0.19 fz:iou0.85/ncc0.03 779
140 id:iou0.00/ncc0.00 fx:iou0.00/ncc0.00 fy:iou0.00/ncc0.00 fz:iou0.00/ncc0.00 0
160 id:iou0.59/ncc-0.14 fx:iou0.59/ncc-0.19 fy:iou0.58/ncc-0.10 fz:iou0.59/ncc-0.13 1162
180 id:iou0.74/ncc-0.29 fx:iou0.74/ncc …[+3097 chars]
```
</details>

**Bash**
```
cd /work; python scratch/code/overlay.py world 150,340,480,580,600,620,820,840,860,880 scratch/ov3.jpg
```

<details><summary>output</summary>

```
(2332, 2168, 3)
```
</details>

**Read**
```
/work/scratch/ov3.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* I've identified clear baseline failures around frames 140–160, 330–350, 480, and 570–900, and found that the floor scenes need a lighting-insensitive high-pass score instead of plain grey-level comparison. Next, I'll automatically detect static phases and fit one shared pose per phase using render-and-compare.

**Bash**
```
cd /work; cat > scratch/code/motion.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
h=h5py.File(f'{INP}/masks.h5','r'); ms=np.load('/work/scratch/mask_stats.npz'); bb=ms['bb']
F=901
prev=None; d1=np.zeros(F); d5=np.zeros(F); cen=np.zeros((F,2))
buf={}
for fr in range(F):
    img=cv2.cvtColor(load_frame(fr),cv2.COLOR_RGB2GRAY).astype(np.float32)
    om=h['object'][fr]; pm=h['person'][fr]
    buf[fr]=(img,om,pm)
    ys,xs=np.nonzero(om); cen[fr]=[xs.mean(),ys.mean()] if len(xs) else [np.nan,np.nan]
    for lag,arr in [(1,d1),(5,d5)]:
        if fr-lag in buf:
            i0,o0,p0=buf[fr-lag]
            reg=(om|o0)&~pm&~p0
            reg=cv2.dilate(reg.astype(np.uint8),np.ones((5,5)))>0
            reg&=~pm&~p0
            arr[fr]=np.abs(img-i0)[reg].mean() if reg.sum()>50 else np.nan
    buf.pop(fr-6,None)
np.savez('/work/scratch/motion.npz',d1=d1,d5=d5,cen=cen)
for fr in range(0,F,5): print(fr, round(d1[fr],1), round(d5[fr],1), np.round(cen[fr],0), int(ms['oa'][fr]))
EOF
python scratch/code/motion.py
```

<details><summary>output</summary>

```
0 0.0 0.0 [810. 611.] 1929
5 1.1 1.9 [810. 611.] 1975
10 0.9 1.5 [810. 611.] 1944
15 0.7 1.4 [810. 611.] 1970
20 0.9 1.5 [810. 611.] 1972
25 0.7 1.5 [810. 611.] 1965
30 1.0 1.3 [810. 611.] 1966
35 0.6 1.2 [810. 611.] 1964
40 0.9 1.2 [810. 611.] 1964
45 0.9 1.3 [810. 611.] 1961
50 1.1 1.9 [810. 611.] 1871
55 1.2 3.1 [810. 611.] 1979
60 1.6 3.2 [810. 611.] 1957
65 1.5 3.8 [810. 611.] 1881
70 1.3 2.8 [813. 611.] 1811
75 1.4 3.7 [813. 612.] 1756
80 3.1 10.7 [814. 611.] 1823
85 7.0 11.2 [813. 610.] 1880
90 8.8 22.7 [812. 608.] 2026
95 7.4 23.4 [812. 604.] 2261
100 14.6 27.7 [807. 601.] 2175
105 16.4 38.8 [794. 592.] 1891
110 14.6 36.8 [777. 586.] 1432
115 6.3 34.4 [750. 582.] 1094
120 15.0 15.7 [721. 585.] 1136
125 27.6 55.8 [699. 594.] 1292
130 30.5 58.0 [680. 610.] 1394
135 20.4 53.1 [647. 62 …[+4556 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/zoom.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
fr_list=[int(x) for x in sys.argv[1].split(',')]; cx,cy,S=map(int,sys.argv[2:5])
tiles=[]
for fr in fr_list:
    c=load_frame(fr)[cy-S//2:cy+S//2,cx-S//2:cx+S//2]
    c=cv2.resize(c,(S*4,S*4),interpolation=cv2.INTER_CUBIC).copy(); cv2.putText(c,str(fr),(3,22),0,0.8,(255,255,0),2); tiles.append(c)
n=int(sys.argv[6]) if len(sys.argv)>6 else 5
while len(tiles)%n: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[5], np.vstack([np.hstack(tiles[i:i+n]) for i in range(0,len(tiles),n)])[...,::-1])
EOF
python scratch/code/zoom.py 640,655,670,680,690,700,730,760 612 808 100 scratch/z1.jpg 4
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/z1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/axes.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp=load_cam()
r=np.load(f'{INP}/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
camc=Twc[:3,3]
for fr in [0,40,250,290,420,450,700,730,860,890]:
    Rm=Tw[fr,:3,:3]; up=np.array([0,-1,0.]); 
    tocam=camc-Tw[fr,:3,3]; tocam/=np.linalg.norm(tocam)
    print(fr,'mesh axes in world up-comp:',np.round(Rm.T@up,2),' toward-cam comp:',np.round(Rm.T@tocam,2))
EOF
python scratch/code/axes.py
```

<details><summary>output</summary>

```
0 mesh axes in world up-comp: [-0.07  1.    0.05]  toward-cam comp: [ 0.61  0.16 -0.77]
40 mesh axes in world up-comp: [-0.07  1.    0.05]  toward-cam comp: [ 0.61  0.16 -0.77]
250 mesh axes in world up-comp: [0.06 0.99 0.1 ]  toward-cam comp: [-0.73  0.41 -0.55]
290 mesh axes in world up-comp: [0.04 1.   0.08]  toward-cam comp: [-0.73  0.39 -0.56]
420 mesh axes in world up-comp: [-0.04  1.    0.04]  toward-cam comp: [ 0.61  0.12 -0.78]
450 mesh axes in world up-comp: [-0.05  1.    0.04]  toward-cam comp: [ 0.59  0.13 -0.79]
700 mesh axes in world up-comp: [-0.07 -1.    0.06]  toward-cam comp: [ 0.87 -0.38 -0.31]
730 mesh axes in world up-comp: [-0.06 -1.    0.06]  toward-cam comp: [ 0.88 -0.38 -0.3 ]
860 mesh axes in world up-comp: [ 0.04 -0.98  0.17]  toward-cam comp: [ 0.08 -0.26 -0.96] …[+90 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/floor.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp=load_cam(); m,v,f,uv,tex=load_mesh()
gp=json.load(open(f'{INP}/ground_plane.json')); n=np.array(gp['plane'][:3]); d=gp['plane'][3]
r=np.load(f'{INP}/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
vs=v[::50].astype(np.float64)
for fr in list(range(0,80,20))+list(range(200,310,15))+list(range(400,470,20))+list(range(640,770,15))+list(range(840,901,15)):
    pw=vs@Tw[fr,:3,:3].T+Tw[fr,:3,3]; s=pw@n+d
    # sign: n.x+d ; floor points 0; compute signed dist min/max
    print(fr, 'plane dist min %.3f max %.3f'%(s.min(),s.max()), 'cam dist %.3f'%np.linalg.norm(Tw[fr,:3,3]-Twc[:3,3]))
EOF
python scratch/code/floor.py
```

<details><summary>output</summary>

```
0 plane dist min 0.893 max 0.959 cam dist 3.157
20 plane dist min 0.893 max 0.960 cam dist 3.156
40 plane dist min 0.896 max 0.963 cam dist 3.134
60 plane dist min 0.897 max 0.963 cam dist 3.130
200 plane dist min 0.037 max 0.113 cam dist 3.524
215 plane dist min -0.004 max 0.076 cam dist 3.562
230 plane dist min -0.004 max 0.076 cam dist 3.562
245 plane dist min -0.005 max 0.071 cam dist 3.568
260 plane dist min -0.006 max 0.063 cam dist 3.565
275 plane dist min -0.008 max 0.060 cam dist 3.579
290 plane dist min -0.002 max 0.065 cam dist 3.564
305 plane dist min -0.013 max 0.095 cam dist 3.581
400 plane dist min 0.887 max 0.952 cam dist 3.301
420 plane dist min 0.887 max 0.947 cam dist 3.281
440 plane dist min 0.891 max 0.954 cam dist 3.281
460 plane dist min 0.882 max 0.950 cam dist 3.33 …[+701 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/precrop.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp=load_cam()
r=np.load(f'{INP}/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64); Tb=np.linalg.inv(Twc)@Tw
h=h5py.File(f'{INP}/masks.h5','r'); S=160; F=901
X0=np.zeros(F,int); Y0=np.zeros(F,int); IM=np.zeros((F,S,S,3),np.uint8); OM=np.zeros((F,S,S),bool); PM=np.zeros((F,S,S),bool)
for fr in range(F):
    om=h['object'][fr]; ys,xs=np.nonzero(om)
    p=K@Tb[fr,:3,3]; pc=p[:2]/p[2]
    c=np.array([xs.mean(),ys.mean()]) if len(xs)>100 else pc
    x0=int(np.clip(c[0]-S/2,0,1536-S)); y0=int(np.clip(c[1]-S/2,0,1152-S))
    X0[fr]=x0;Y0[fr]=y0
    IM[fr]=load_frame(fr)[y0:y0+S,x0:x0+S]; OM[fr]=om[y0:y0+S,x0:x0+S]; PM[fr]=h['person'][fr][y0:y0+S,x0:x0+S]
np.savez_compressed('/work/scratch/crops.npz',X0=X0,Y0=Y0,IM=IM,OM=OM,PM=PM,S=S)
EOF
python scratch/code/precrop.py; ls -la scratch/crops.npz
```

<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 39746513 Oct 21 01:13 scratch/crops.npz
```
</details>

**Bash**
```
cd /work; cat > scratch/code/fit.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
def gauss_k(sig):
    r=int(3*sig+0.5); x=torch.arange(-r,r+1,dtype=torch.float32); g=torch.exp(-x**2/(2*sig**2)); return g/g.sum()
def blur(x,sig):  # x (B,H,W)
    g=gauss_k(sig).to(x.device); r=len(g)//2
    x=Fn.conv2d(Fn.pad(x[:,None],(r,r,0,0),mode='replicate'),g[None,None,None,:])
    x=Fn.conv2d(Fn.pad(x,(0,0,r,r),mode='replicate'),g[None,None,:,None]); return x[:,0]
def so3_exp(w):
    th=w.norm(dim=-1,keepdim=True).clamp(min=1e-8); k=w/th
    Kx=torch.zeros(w.shape[:-1]+(3,3),device=w.device)
    Kx[...,0,1]=-k[...,2];Kx[...,0,2]=k[...,1];Kx[...,1,0]=k[...,2];Kx[...,1,2]=-k[...,0];Kx[...,2,0]=-k[...,1];Kx[...,2,1]=k[...,0]
    s=torch.sin(th)[...,None]; c=torch.cos(th)[...,None]
    return torch.eye(3,device=w.device)+s*Kx+(1-c)*Kx@Kx
class Fitter:
    def __init__(self, inp='/work/inputs', crops='/work/scratch/crops.npz', ss=2):
        self.K,self.Twc,self.Kp=load_cam(inp)
        m,v,f,uv,tex=load_mesh(inp); self.R=Renderer(v,f,uv,tex); self.ss=ss
        c=np.load(crops); self.S=int(c['S']); self.X0=c['X0']; self.Y0=c['Y0']
        d='cuda'
        self.IM=torch.tensor(c['IM'],device=d).float()/255; self.OM=torch.tensor(c['OM'],device=d); self.PM=torch.tensor(c['PM'],device=d)
        self.G=self.IM.mean(-1)
        self.Ghp=self.G-blur(self.G,2.0)
    def proj(self, idx):
        S=self.S*self.ss; ss=self.ss; K=self.K; n,fa=0.05,20.
        P=np. …[+4644 chars]
```

<details><summary>output</summary>

```
0 id:iou0.79 lb0.07 ncc0.68 fx:iou0.78 lb0.07 ncc0.62 fy:iou0.78 lb0.07 ncc0.66 fz:iou0.77 lb0.08 ncc0.73
40 id:iou0.79 lb0.06 ncc0.65 fx:iou0.79 lb0.06 ncc0.59 fy:iou0.79 lb0.07 ncc0.11 fz:iou0.78 lb0.07 ncc0.69
100 id:iou0.79 lb0.09 ncc0.81 fx:iou0.79 lb0.08 ncc0.69 fy:iou0.80 lb0.08 ncc0.65 fz:iou0.79 lb0.08 ncc0.71
120 id:iou0.87 lb0.04 ncc0.50 fx:iou0.88 lb0.04 ncc0.55 fy:iou0.84 lb0.05 ncc0.48 fz:iou0.84 lb0.05 ncc0.44
200 id:iou0.87 lb0.03 ncc0.64 fx:iou0.85 lb0.04 ncc0.71 fy:iou0.84 lb0.05 ncc0.71 fz:iou0.85 lb0.04 ncc0.67
250 id:iou0.86 lb0.03 ncc0.69 fx:iou0.84 lb0.04 ncc0.67 fy:iou0.83 lb0.05 ncc0.66 fz:iou0.85 lb0.04 ncc0.63
290 id:iou0.82 lb0.07 ncc0.68 fx:iou0.80 lb0.07 ncc0.68 fy:iou0.79 lb0.08 ncc0.66 fz:iou0.81 lb0.08 ncc0.65
400 id:iou0.81 lb0.04 ncc0.68 fx:iou0.81 lb0.04 …[+816 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/app.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from fit import *
def nblur(x,w,sig): return blur(x*w,sig)/blur(w,sig).clamp(min=1e-4)
def ncc(a,b,w):
    n=w.sum((1,2)).clamp(min=1)
    a0=a-(a*w).sum((1,2))[:,None,None]/n[:,None,None]; b0=b-(b*w).sum((1,2))[:,None,None]/n[:,None,None]
    return (a0*b0*w).sum((1,2))/torch.sqrt((a0*a0*w).sum((1,2))*(b0*b0*w).sum((1,2))+1e-9)
def render_rgb(Fi,Tc,idx):
    R=Fi.R; S=Fi.S*Fi.ss
    vh=torch.cat([R.v,torch.ones_like(R.v[:,:1])],1); vc=torch.einsum('bij,nj->bni',Tc.float(),vh)
    clip=torch.einsum('bij,bnj->bni',Fi.proj(idx),vc).contiguous()
    rast,_=dr.rasterize(R.glctx,clip,R.f,(S,S)); msk=(rast[...,3:4]>0).float()
    uvi,_=dr.interpolate(R.uv_t[None].contiguous(),rast,R.f); col=dr.texture(R.tex,uvi,filter_mode='linear')*msk
    out=Fn.avg_pool2d(torch.cat([msk,col],-1).permute(0,3,1,2),Fi.ss)
    m=out[:,0]; return m, out[:,1:]/m[:,None].clamp(min=1e-4)
def app_scores(Fi,Tc,idx,sig=1.5):
    it=torch.tensor(list(idx),device='cuda')
    m,rgb=render_rgb(Fi,Tc,idx)
    valid=(~Fi.PM[it]).float()*Fi.OM[it].float()
    inner=(-Fn.max_pool2d(-(m>0.99).float()[:,None],5,1,2)[:,0])*valid  # eroded 2px
    g=rgb.mean(1); I=Fi.G[it]
    ghp=g-nblur(g,inner,sig); ihp=I-nblur(I,inner,sig)
    s_hp=ncc(ghp,ihp,inner)
    s_g=ncc(g,I,inner)
    ch_r=rgb[:,2]-rgb[:,0]; ch_i=Fi.IM[it][...,2]-Fi.IM[it][...,0]
    s_ch=ncc(ch_r,ch_i,inner)
    return s_hp,s_g,s_ch,inner.sum((1,2))
if __name_ …[+500 chars]
```

<details><summary>output</summary>

```
0 1183 id: hp0.02 g0.60 ch0.76 fx: hp0.06 g0.46 ch0.19 fy: hp0.16 g0.21 ch0.12 fz: hp0.17 g0.27 ch0.68
40 1206 id: hp-0.02 g0.59 ch0.77 fx: hp0.04 g0.47 ch0.18 fy: hp0.16 g0.19 ch0.13 fz: hp0.16 g0.24 ch0.68
100 1378 id: hp0.32 g0.05 ch0.79 fx: hp-0.01 g-0.21 ch0.08 fy: hp-0.00 g-0.54 ch0.21 fz: hp-0.01 g-0.46 ch0.72
120 609 id: hp-0.24 g-0.24 ch0.16 fx: hp0.26 g0.16 ch0.06 fy: hp0.15 g0.07 ch0.31 fz: hp-0.05 g0.02 ch0.58
200 1677 id: hp-0.01 g-0.28 ch0.78 fx: hp-0.09 g-0.43 ch0.23 fy: hp-0.06 g-0.29 ch0.13 fz: hp-0.02 g-0.24 ch0.46
250 1648 id: hp0.06 g-0.14 ch0.72 fx: hp-0.08 g-0.38 ch0.18 fy: hp0.04 g-0.20 ch0.17 fz: hp0.04 g-0.14 ch0.52
290 1561 id: hp0.05 g-0.10 ch0.67 fx: hp0.03 g-0.29 ch0.33 fy: hp-0.06 g-0.22 ch-0.11 fz: hp0.02 g-0.12 ch0.40
400 905 id: hp0.33 g0.67 ch0.61 fx: hp-0 …[+826 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/viewhyp.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from app import *
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
frames=[int(x) for x in sys.argv[1].split(',')]; names=list(FLIPS)
rows=[]
for fr in frames:
    Ts=torch.tensor(np.stack([Tb[fr]@FLIPS[n] for n in names]),device='cuda')
    m,rgb=render_rgb(Fi,Ts,[fr]*4)
    img=Fi.IM[fr].cpu().numpy()
    tiles=[img]
    for i in range(4):
        mm=m[i].cpu().numpy()[...,None]; tiles.append(mm*rgb[i].permute(1,2,0).cpu().numpy()+(1-mm)*img*0.3)
    t=np.hstack(tiles); 
    ys,xs=np.nonzero(Fi.OM[fr].cpu().numpy()); c=(int(ys.mean()),int(xs.mean()))
    rows.append(cv2.resize((t*255).clip(0,255).astype(np.uint8)[max(c[0]-50,0):c[0]+50],None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC))
cv2.imwrite(sys.argv[2],np.vstack(rows)[...,::-1])
EOF
python scratch/code/viewhyp.py 0,250,700 scratch/vh1.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vh1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import re
s=open('/work/scratch/code/fit.py').read()
a=s.index('        # appearance: composite'); b=s.index("        return {'iou'")
new='''        # appearance: interior high-pass NCC (rendered texture vs image), lighting-insensitive
        it2=it
        inner=(-Fn.max_pool2d(-(m>0.99).float()[:,None],5,1,2)[:,0])*valid*om
        I=self.G[it2]
        ghp=g-nblur(g.detach()*0+g,inner,1.5); ihp=I-nblur(I,inner,1.5)
        ncc_=ncc(ghp,ihp,inner)
        ncc_=torch.where(inner.sum((1,2))>30,ncc_,torch.zeros_like(ncc_))
'''
s=s[:a]+new+s[b:]
s=s.replace("return {'iou':iou,'lb':lb,'ncc':ncc,'loss':(1-iou)+lb+w_app*(1-ncc)}","return {'iou':iou,'lb':lb,'ncc':ncc_,'loss':(1-iou)+lb+w_app*(1-ncc_)}")
s=s.replace("class Fitter:",'''def nblur(x,w,sig): return blur(x*w,sig)/blur(w,sig).clamp(min=1e-4)
def ncc(a,b,w):
    n=w.sum((1,2)).clamp(min=1)
    a0=a-(a*w).sum((1,2))[:,None,None]/n[:,None,None]; b0=b-(b*w).sum((1,2))[:,None,None]/n[:,None,None]
    return (a0*b0*w).sum((1,2))/torch.sqrt((a0*a0*w).sum((1,2))*(b0*b0*w).sum((1,2))+1e-9)
class Fitter:''')
open('/work/scratch/code/fit.py','w').write(s)
EOF
grep -n "ncc" scratch/code/fit.py | head
```

<details><summary>output</summary>

```
16:def ncc(a,b,w):
66:        ncc_=ncc(ghp,ihp,inner)
67:        ncc_=torch.where(inner.sum((1,2))>30,ncc_,torch.zeros_like(ncc_))
68:        return {'iou':iou,'lb':lb,'ncc':ncc_,'loss':(1-iou)+lb+w_app*(1-ncc_)}
```
</details>

**Bash**
```
cd /work; sed -i 's/nblur(g.detach()\*0+g,inner,1.5)/nblur(g,inner,1.5)/' scratch/code/fit.py; cat > scratch/code/t2.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from fit import *
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
a,b=int(sys.argv[1]),int(sys.argv[2]); idx=list(range(a,b,3))
mid=idx[len(idx)//2]
for nm,Fl in FLIPS.items():
    t0=time.time()
    T,L=Fi.optimize(np.stack([Tb[mid]@Fl]),idx,iters=150,shared=True,lr_r=0.01,lr_t=0.003)
    T1=T[0]
    dR=np.degrees(np.linalg.norm(Rot.from_matrix(T1[:3,:3].T@(Tb[mid]@Fl)[:3,:3]).as_rotvec()))
    print(nm,'iou %.3f lb %.3f ncc %.3f loss %.3f'%(L['iou'].mean(),L['lb'].mean(),L['ncc'].mean(),L['loss'].mean()),'dR %.1f dt'%dR,np.round((T1[:3,3]-Tb[mid,:3,3])*100,1),'%.1fs'%(time.time()-t0))
EOF
python scratch/code/t2.py 0 75
```

<details><summary>output</summary>

```
id iou 0.853 lb 0.028 ncc 0.513 loss 0.662 dR 4.5 dt [-1.8  0.2 -2.8] 4.9s
fx iou 0.822 lb 0.039 ncc 0.209 loss 1.008 dR 3.8 dt [-1.4 -0.1 -2.9] 1.7s
fy iou 0.702 lb 0.111 ncc -0.067 loss 1.476 dR 5.8 dt [-2.5 -0.2 -0. ] 1.9s
fz iou 0.866 lb 0.027 ncc -0.053 loss 1.214 dR 15.9 dt [-0.2  0.6 -1. ] 1.8s
```
</details>

**Bash**
```
cd /work; for p in "215 300" "400 466" "650 750" "845 901"; do echo "== $p"; python scratch/code/t2.py $p; done
```

<details><summary>output</summary>

```
== 215 300
id iou 0.771 lb 0.097 ncc 0.167 loss 1.159 dR 4.2 dt [ 0.8  0.1 -0.1] 3.2s
fx iou 0.797 lb 0.082 ncc 0.047 loss 1.238 dR 7.6 dt [-1.9  0.7 -0.4] 2.1s
fy iou 0.806 lb 0.059 ncc 0.002 loss 1.251 dR 2.5 dt [-2.1 -0.1 -0.1] 3.8s
fz iou 0.847 lb 0.057 ncc 0.064 loss 1.145 dR 3.3 dt [-0.9 -0.5  1.1] 4.1s
== 400 466
id iou 0.849 lb 0.034 ncc 0.359 loss 0.827 dR 2.6 dt [-0.2 -0.3 -1.3] 2.5s
fx iou 0.756 lb 0.092 ncc 0.263 loss 1.073 dR 4.3 dt [-0.5 -0.7  0.1] 1.7s
fy iou 0.812 lb 0.051 ncc -0.015 loss 1.254 dR 3.8 dt [-2.5  0.8 -1.9] 1.6s
fz iou 0.771 lb 0.096 ncc -0.015 loss 1.340 dR 7.9 dt [-4.1  0.7 -0.7] 3.2s
== 650 750
id iou 0.894 lb 0.041 ncc 0.277 loss 0.871 dR 1.5 dt [ 0.4 -0.1 -0.1] 5.9s
fx iou 0.884 lb 0.037 ncc 0.092 loss 1.062 dR 3.6 dt [-0.  -0.3 -0.3] 4.6s
fy iou 0.856 lb …[+449 chars]
```
</details>

**Bash**
```
cd /work; python scratch/code/zoom.py 826,832,838,845,855,865,875,885,893,900 688 615 100 scratch/z2.jpg 5
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/z2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python scratch/code/zoom.py 20,420,840,890 720 640 260 scratch/z3.jpg 2
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/z3.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* I've mapped out the book's resting phases (table, floor, table, floor) and found that during 830–900 it's held just below the tabletop edge, causing the low IoU due to occlusion rather than a wrong pose. I'll now model the tabletop as an occluder plane at ~0.89 m, using the pooled texture score (which favors the baseline orientation in S1/S3/S4) to resolve the remaining ambiguity in S2.

**Bash**
```
cd /work; cat > scratch/code/bg.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
h=h5py.File(f'{INP}/masks.h5','r')
fr=list(range(0,901,15))
stack=[]; 
for f in fr:
    im=load_frame(f).astype(np.float32); m=h['person'][f]|h['object'][f]
    m=cv2.dilate(m.astype(np.uint8),np.ones((15,15)))>0
    im[m]=np.nan; stack.append(im)
bg=np.nanmedian(np.stack(stack),0)
np.save('/work/scratch/bg.npy',bg.astype(np.float32))
cv2.imwrite('/work/scratch/bg.jpg',np.nan_to_num(bg)[...,::-1].astype(np.uint8))
EOF
python scratch/code/bg.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
cd /work; cat > scratch/code/table.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
bg=np.load('/work/scratch/bg.npy'); g=bg.mean(-1); sat=bg.max(-1)-bg.min(-1)
th=(g>185)&(sat<35)
th=cv2.morphologyEx(th.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3)))
n,lab,st,cen=cv2.connectedComponentsWithStats(th)
seed=lab[640,720]; print('seed label',seed, st[seed])
tab=(lab==seed)
tab=cv2.morphologyEx(tab.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((7,7)))>0
np.save('/work/scratch/table_mask.npy',tab)
vis=bg.copy().astype(np.uint8); vis[tab]=(0.5*vis[tab]+[0,0,127]).astype(np.uint8)
cv2.imwrite('/work/scratch/table_vis.jpg',vis[400:1000,350:1100][...,::-1])
EOF
python scratch/code/table.py
```

<details><summary>output</summary>

```
seed label 26 [ 616  606  328   58 9603]
```
</details>

**Read**
```
/work/scratch/table_vis.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat /pipeline/depth/moge2/depth_report.json; python -c "
import h5py,pickle,numpy as np
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
def p(n,o):
    if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype)
h.visititems(p)
print(dict(h.attrs))
k=pickle.load(open('/pipeline/depth/moge2/intrinsics.pkl','rb')); print(type(k), (k.keys() if hasattr(k,'keys') else len(k)))
"

```

<details><summary>output</summary>

```
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 9.283381462097168,
    "min": 1.5831542015075684
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 140.15602144016884,
  "firstFrameIntrinsicsPx": [
    [
      825.64306640625,
      0.0,
      768.0
    ],
    [
      0.0,
      825.6430053710938,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "focalXPx": {
    "max": 835.900390625,
    "mean": 823.6602783203125,
    "min": 814.3628540039062
  },
  "focalYPx": {
    "max": 835.9003295898438,
    "mean": 823.6602783203125,
    "min": 814.3628540039062
  },
  "fps": 30.0,
  "frames": 901,
  "framesPerSecond": 6.428550059724888,
  "height": 1152,
  "intrinsicsFile": {
    …[+2545 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import h5py,numpy as np,cv2
h=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
a=h['aligned/front_stereo_camera_left'][0]; print(type(a), len(a), bytes(a[:8]))
d=cv2.imdecode(np.frombuffer(a,np.uint8),cv2.IMREAD_UNCHANGED); print(d.shape,d.dtype,d.min(),d.max())
print(h['alignment/front_stereo_camera_left/scale'][:10], h['alignment/front_stereo_camera_left/shift'][:10])
"

```

<details><summary>output</summary>

```
<class 'numpy.ndarray'> 648767 b'\x89PNG\r\n\x1a\n'
(1152, 1536) uint16 1812 10078
[1.1243744 1.121233  1.1227219 1.1203622 1.1217712 1.1275384 1.1280569
 1.1285108 1.1232473 1.1216615] [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/mogecmp.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp=load_cam(); m,v,f,uv,tex=load_mesh(); R=Renderer(v,f,uv,tex)
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Twc)@r['T_world_mesh'].astype(np.float64)
Tco=r['T_cam_mesh_coconet'].astype(np.float64)
h=h5py.File(f'{INP}/masks.h5','r'); dh=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
tm=np.load('/work/scratch/table_mask.npy')
res=[]
for fr in list(range(0,901,10)):
    om=h['object'][fr]&~h['person'][fr]; om=cv2.erode(om.astype(np.uint8),np.ones((5,5)))>0
    if om.sum()<50: continue
    D=cv2.imdecode(np.frombuffer(dh['aligned/front_stereo_camera_left'][fr],np.uint8),-1).astype(np.float32)/1000
    o=R.render(torch.tensor(Tb[fr][None],device='cuda'),K,1152,1536,need_rgb=False)
    rd=o['depth'][0].cpu().numpy(); rm=o['mask'][0].cpu().numpy()
    sel=om&rm
    oc=R.render(torch.tensor(Tco[fr][None],device='cuda'),Kp,1152,1536,need_rgb=False)
    cd=oc['depth'][0].cpu().numpy(); cm=oc['mask'][0].cpu().numpy()
    sel2=om&cm
    tz=np.median(D[tm]) 
    res.append((fr,np.median(D[om]),np.median(rd[sel]) if sel.sum() else np.nan,np.median(cd[sel2]) if sel2.sum() else np.nan,tz))
    print('%d moge %.3f base %.3f coco %.3f  ratio base/moge %.3f table %.3f'%(fr,res[-1][1],res[-1][2],res[-1][3],res[-1][2]/res[-1][1],tz))
EOF
python scratch/code/mogecmp.py
```

<details><summary>output</summary>

```
0 moge 3.140 base 3.057 coco 3.006  ratio base/moge 0.974 table 2.968
10 moge 3.149 base 3.057 coco 3.006  ratio base/moge 0.971 table 2.978
20 moge 3.142 base 3.057 coco 3.007  ratio base/moge 0.973 table 2.965
30 moge 3.200 base 3.051 coco 3.007  ratio base/moge 0.953 table 3.019
40 moge 3.168 base 3.034 coco 3.007  ratio base/moge 0.958 table 2.983
50 moge 3.114 base 3.031 coco 3.005  ratio base/moge 0.973 table 2.961
60 moge 3.102 base 3.030 coco 3.006  ratio base/moge 0.977 table 2.940
70 moge 3.134 base 3.078 coco 3.059  ratio base/moge 0.982 table 2.988
80 moge 3.116 base 3.093 coco 3.078  ratio base/moge 0.992 table 2.977
90 moge 3.118 base 3.059 coco 3.043  ratio base/moge 0.981 table 2.968
100 moge 3.134 base 3.082 coco 3.046  ratio base/moge 0.983 table 2.948
110 moge 3.045 base …[+5518 chars]
```
</details>

> *thinking:* Baseline depth holds up during the static floor phases, but MoGe's S4 estimate places the book below the floor, so I'll disregard MoGe here. Next I'll cross-check depth during the table phases using the table legs' floor contact to estimate table height independently.

**Bash**
```
cd /work; python -c "
import numpy as np,cv2
bg=np.load('/work/scratch/bg.npy').astype(np.uint8)
c=bg[580:960,600:1000].copy()
for y in range(0,380,20): cv2.line(c,(0,y),(399,y),(255,0,0) if y%100==0 else (0,255,255),1)
for x in range(0,400,20): cv2.line(c,(x,0),(x,379),(255,0,0) if x%100==0 else (0,255,255),1)
cv2.imwrite('/work/scratch/legs.jpg',cv2.resize(c,None,fx=2.5,fy=2.5)[...,::-1])
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/legs.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/tableh.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp=load_cam(); gp=json.load(open(f'{INP}/ground_plane.json')); n=np.array(gp['plane'][:3]); d=gp['plane'][3]
C=Twc[:3,3]; Rwc=Twc[:3,:3]
def ray(u,v): r=Rwc@np.linalg.solve(K,[u,v,1.]); return r/np.linalg.norm(r)
def proj(P): pc=np.linalg.inv(Twc)@np.r_[P,1]; p=K@pc[:3]; return p[:2]/p[2]
up=n/np.linalg.norm(n)  # n.x+d: positive above floor? check
print('cam height', C@n+d)
for name,(u,vb,vt) in {'left':(693,903,664),'right':(861,850,646)}.items():
    for barh in [0.0,0.03]:
        r=ray(u,vb); t=-(C@n+d-barh)/(r@n); P=C+t*r
        hs=np.linspace(0,1.5,15001); ys=np.array([proj(P+h*n)[1] for h in hs[::10]])
        h=hs[::10][np.argmin(abs(ys-vt))]
        print(name,'barh',barh,'depth %.2f'%t,'column height above floor %.3f'%(h+barh))
EOF
python scratch/code/tableh.py
```

<details><summary>output</summary>

```
cam height 1.3347325757242718
left barh 0.0 depth 2.98 column height above floor 0.808
left barh 0.03 depth 2.91 column height above floor 0.820
right barh 0.0 depth 3.38 column height above floor 0.794
right barh 0.03 depth 3.30 column height above floor 0.806
```
</details>

**Bash**
```
cd /work; cat > scratch/code/legs.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
bg=np.load('/work/scratch/bg.npy'); g=bg.mean(-1); sat=bg.max(-1)-bg.min(-1)
tab=np.load('/work/scratch/table_mask.npy')
cand=((g>140)&(sat<30)).astype(np.uint8)
cand=cv2.morphologyEx(cand,cv2.MORPH_OPEN,np.ones((3,3)))
tabd=cv2.dilate(tab.astype(np.uint8),np.ones((9,9)))
n,lab,st,_=cv2.connectedComponentsWithStats(cand|tabd)
l=lab[640,720]; comp=(lab==l)
legs=comp&~(tabd>0)
legs=cv2.morphologyEx(legs.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((5,5)))>0
# table-top with front edge strip (dilate downward)
k=np.zeros((9,1),np.uint8); k[4:]=1
top=cv2.dilate(tab.astype(np.uint8),k)>0
np.save('/work/scratch/legs_mask.npy',legs); np.save('/work/scratch/top_mask.npy',top)
vis=bg.copy().astype(np.uint8); vis[legs]=(0.5*vis[legs]+[127,0,0]).astype(np.uint8); vis[top]=(0.5*vis[top]+[0,0,127]).astype(np.uint8)
cv2.imwrite('/work/scratch/legs_vis.jpg',vis[500:1000,450:1050][...,::-1])
EOF
python scratch/code/legs.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/legs_vis.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np
bg=np.load('/work/scratch/bg.npy')
for p in [(780,693),(850,690),(930,700),(750,861),(870,900),(700,640),(700,760)]: print(p,bg[p])
"
```

<details><summary>output</summary>

```
(780, 693) [125. 127. 113.]
(850, 690) [114. 115. 101.]
(930, 700) [57. 58. 50.]
(750, 861) [114. 116. 102.]
(870, 900) [128. 131. 124.]
(700, 640) [92. 94. 83.]
(700, 760) [92. 95. 89.]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
s=open('/work/scratch/code/fit.py').read()
s=s.replace('''        self.G=self.IM.mean(-1)''','''        self.G=self.IM.mean(-1)
        self.set_table(0.89)
    def set_table(self, h_table, top_mask='/work/scratch/top_mask.npy'):
        """Static occluders: table-top (plane at h_table above floor, occludes points behind it) and leg columns."""
        gp=json.load(open('/work/inputs/ground_plane.json')); n=np.array(gp['plane'][:3]); d=gp['plane'][3]
        top=np.load(top_mask); Tcw=np.linalg.inv(self.Twc)
        nc=Tcw[:3,:3]@n; dc=d-n@self.Twc[:3,3]*0  # plane in cam: n.(Rwc x + C) + d - h = 0
        C=self.Twc[:3,3]; nC=self.Twc[:3,:3].T@n; dC=n@C+d-h_table
        ys,xs=np.mgrid[0:1152,0:1536]; rays=np.stack([(xs-self.K[0,2])/self.K[0,0],(ys-self.K[1,2])/self.K[1,1],np.ones_like(xs,float)],-1)
        z=-dC/(rays@nC); z[(z<0)|~top]=1e9
        legs=np.zeros((1152,1536),bool)
        # leg columns of the table, read off the background image by eye (x0,x1,y0,y1), see REPORT
        for x0,x1,y0,y1 in [(678,709,664,908),(847,875,646,852)]: legs[y0:y1,x0:x1]=True
        S=self.S; F=len(self.X0); TZ=np.zeros((F,S,S),np.float32); LG=np.zeros((F,S,S),bool)
        for f in range(F):
            TZ[f]=z[self.Y0[f]:self.Y0[f]+S,self.X0[f]:self.X0[f]+S]; LG[f]=legs[self.Y0[f]:self.Y0[f]+S,self.X0[f]:self.X0[f]+S]
        self.TZ=torch.tensor(TZ,device='cuda'); self.LG=torch.tensor(LG,device='cuda')''')
# render depth too
s=s.replace('''        msk=(r …[+1630 chars]
```

<details><summary>output</summary>

```
0 [0.79 0.06 1.22]	10 [0.78 0.05 1.23]	20 [0.78 0.04 1.24]	30 [0.79 0.03 1.25]	40 [0.79 0.01 1.26]	50 [0.81 0.01 1.23]
60 [0.8  0.01 1.25]	70 [0.82 0.34 0.88]	80 [ 0.85 -0.02  1.2 ]	90 [0.83 0.22 1.  ]	100 [0.79 0.29 1.01]	110 [0.68 0.32 1.17]
120 [ 0.87 -0.14  1.3 ]	130 [0.6  0.1  1.49]	140 [0.   0.   2.88]	150 [0.14 0.14 2.56]	160 [0.72 0.01 1.39]	170 [ 0.65 -0.07  1.61]
180 [ 0.85 -0.09  1.29]	190 [0.89 0.19 0.94]	200 [ 0.91 -0.05  1.15]	210 [0.91 0.09 1.02]	220 [0.91 0.14 0.97]	230 [ 0.92 -0.04  1.14]
240 [0.91 0.04 1.06]	250 [0.9  0.07 1.05]	260 [0.92 0.02 1.07]	270 [0.9  0.22 0.9 ]	280 [0.89 0.16 0.97]	290 [0.88 0.02 1.12]
300 [0.9  0.11 1.02]	310 [0.88 0.33 0.82]	320 [ 0.79 -0.04  1.33]	330 [ 0.37 -0.01  2.14]	340 [ 0.64 -0.04  1.62]	350 [0.64 0.04 1.56]
360 [ 0.    0.   36.98]	370  …[+1165 chars]
```
</details>

> *thinking:* Correcting my earlier assessment: with the tabletop modeled as an occluder, baseline poses at 840–890 actually fit the visible book well (IoU 0.94–1.0), meaning the book lies flat behind the table edge rather than standing up—the real problem frames are the carried segments (130–175, 320–360, 470–490, 570–630, 780–830). Next I'll run per-frame multi-hypothesis refinement across baseline, FoundationPose, and 180° flips, optimizing each via render-and-compare.

**Bash**
```
cd /work; python - <<'EOF'
s=open('/work/scratch/code/fit.py').read()
s=s.replace("occ=self.PM[it]|self.LG[it]|((self._z>self.TZ[it])&(self._z>0))","occ=self.PM[it]|self.LG[it]|((self._z>self.TZ[it]+0.02)&(self._z>0))")
s=s.replace("lb=(((mb-ob)**2)*valid).sum((1,2))/((ob*valid).sum((1,2)).clamp(min=1))","lb=(((mb-ob)**2)*valid).sum((1,2))/((ob*valid).sum((1,2)).clamp(min=20))")
open('/work/scratch/code/fit.py','w').write(s)
EOF
cat > scratch/code/perframe.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from fit import *
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
Tco=r['T_cam_mesh_coconet'].astype(np.float64); Tfp=r['T_cam_mesh_foundationpose'].astype(np.float64)
M=np.einsum('nij,nkj->ik',Tb[:,:3,:3],Tco[:,:3,:3]); U,_,Vt=np.linalg.svd(M); A=U@Vt
print('A angle',np.degrees(np.linalg.norm(Rot.from_matrix(A).as_rotvec())))
F=901
def fp_hyp(f):
    T=np.eye(4); T[:3,:3]=A@Tfp[f,:3,:3]
    p=Fi.Kp@Tfp[f,:3,3]; uv=p[:2]/p[2]; ray=np.linalg.solve(Fi.K,[uv[0],uv[1],1]); T[:3,3]=ray*Tb[f,2,3]
    return T
H=['base','fp','fx','fy','fz']
hyps=np.zeros((F,len(H),4,4)); res=np.zeros((F,len(H),4))
t0=time.time()
B=24
for s in range(0,F,B):
    idx=list(range(s,min(s+B,F)))
    T0=[];ii=[]
    for f in idx:
        for h in H:
            T0.append(Tb[f] if h=='base' else fp_hyp(f) if h=='fp' else Tb[f]@FLIPS[h]); ii.append(f)
    T,L=Fi.optimize(np.stack(T0),ii,iters=80,lr_r=0.02,lr_t=0.004,fix_dept …[+421 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bqukelfrb). Output is being written to: /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bqukelfrb.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bqukelfrb.output; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

<details><summary>output</summary>

```
A angle 1.2657222856449133
pid, used_gpu_memory [MiB]
4329, 5478 MiB
```
</details>

**Bash**
```
cat > /work/scratch/NOTES.md <<'EOF'
# Working notes (ep011 light_blue_book)
- GPU time used (approx): stats/overlays ~1 min, static fits ~1 min, perframe.py run1 (started 01:2x)...
- Phases: S1 table 0-77; carry 78-211; S2 floor 212-300 (hand touching); carry 301-397; S3 table 398-466; carry 467-640 (in-hand rotation 500-550);
  S4 floor 645-760; carry 761-830; 830-900 held behind table far edge (occluded by tabletop), slides.
- Mesh: x=19cm (spine at +x), y=5.9cm thickness (front cover -y, back +y), z=26cm (logo end +z).
- Baseline = CoCoNet * const 1.27deg rot, per-frame depth rescale.
- Table occluder plane at h=0.89 above floor (book-bottom height in S1/S3); leg columns rects by eye.
- MoGe depth disagrees (S4 20cm farther -> below floor); keep baseline depth.
EOF
sleep 45; cat /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bqukelfrb.output | head -5
```

<details><summary>output</summary>

```
A angle 1.2657222856449133
```
</details>

**Bash**
```
sleep 120; cat /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bqukelfrb.output | head -5
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
A angle 1.2657222856449133
time 275.4391529560089
0 fp base:0.53/-0.03 fp:0.81/0.18 fx:0.77/0.13 fy:0.85/0.11 fz:0.90/0.02
10 fp base:0.79/0.12 fp:0.81/0.29 fx:0.82/-0.06 fy:0.86/0.06 fz:0.63/0.01
20 base base:0.78/0.42 fp:0.61/0.10 fx:0.77/0.07 fy:0.55/-0.03 fz:0.78/0.04
```
</details>

**Bash**
```
cat /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bqukelfrb.output | tail -88 | paste - - | head -50
```

<details><summary>output</summary>

```
50 base base:0.88/0.41 fp:0.81/0.06 fx:0.61/0.09 fy:0.59/0.03 fz:0.38/-0.04	60 base base:0.77/0.43 fp:0.81/0.28 fx:0.80/0.10 fy:0.54/0.13 fz:0.86/-0.01
70 fp base:0.81/0.17 fp:0.82/0.17 fx:0.77/0.17 fy:0.80/0.04 fz:0.86/0.03	80 fy base:0.71/0.08 fp:0.77/0.03 fx:0.71/0.06 fy:0.83/0.24 fz:0.61/-0.01
90 fy base:0.64/-0.00 fp:0.68/0.05 fx:0.75/0.05 fy:0.77/0.08 fz:0.81/-0.06	100 fy base:0.44/0.03 fp:0.81/-0.00 fx:0.62/0.12 fy:0.85/0.01 fz:0.76/-0.03
110 base base:0.84/0.02 fp:0.73/0.02 fx:0.31/-0.04 fy:0.71/-0.02 fz:0.49/-0.00	120 fx base:0.77/-0.00 fp:0.86/0.08 fx:0.94/0.03 fy:0.90/0.06 fz:0.90/0.07
130 fx base:0.32/-0.11 fp:0.78/-0.01 fx:0.79/0.08 fy:0.68/0.03 fz:0.70/0.01	140 base base:0.37/0.02 fp:0.16/-0.04 fx:0.19/0.06 fy:0.38/0.08 fz:0.27/0.15
150 fp base:0.59/-0.01 fp:0.74/-0.05 fx:0.4 …[+5731 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/dbg.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from fit import *
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
idx=[0,100,200,700]
for w_app in [0.0,1.0]:
  for lr in [0.02,0.005]:
    T0=torch.tensor(Tb[idx],device='cuda',dtype=torch.float32)
    w=torch.zeros(4,3,device='cuda',requires_grad=True); t=torch.zeros(4,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr},{'params':[t],'lr':lr/5}])
    hist=[]
    for i in range(80):
        T=torch.eye(4,device='cuda').repeat(4,1,1); T[:,:3,:3]=so3_exp(w)@T0[:,:3,:3]; T[:,:3,3]=T0[:,:3,3]+t
        L=Fi.losses(T,idx,w_app); opt.zero_grad(); L['loss'].sum().backward(); opt.step()
        if i%10==0 or i==79: hist.append(np.round(L['iou'].detach().cpu().numpy(),3))
    print('w_app',w_app,'lr',lr); print(np.array(hist).T)
EOF
python scratch/code/dbg.py
```

<details><summary>output</summary>

```
w_app 0.0 lr 0.02
[[0.786 0.912 0.916 0.921 0.928 0.93  0.935 0.939 0.943]
 [0.79  0.913 0.932 0.932 0.934 0.936 0.939 0.939 0.94 ]
 [0.913 0.945 0.945 0.946 0.946 0.947 0.947 0.946 0.947]
 [0.945 0.959 0.959 0.961 0.962 0.962 0.961 0.962 0.96 ]]
w_app 0.0 lr 0.005
[[0.786 0.893 0.927 0.929 0.931 0.928 0.928 0.928 0.926]
 [0.79  0.875 0.898 0.911 0.92  0.923 0.927 0.929 0.932]
 [0.913 0.944 0.945 0.946 0.946 0.946 0.945 0.947 0.945]
 [0.945 0.959 0.961 0.961 0.962 0.961 0.962 0.962 0.961]]
w_app 1.0 lr 0.02
[[0.786 0.651 0.65  0.698 0.692 0.689 0.683 0.635 0.661]
 [0.79  0.731 0.714 0.686 0.654 0.666 0.666 0.725 0.743]
 [0.913 0.905 0.874 0.835 0.835 0.821 0.817 0.835 0.85 ]
 [0.945 0.919 0.935 0.934 0.945 0.936 0.929 0.94  0.931]]
w_app 1.0 lr 0.005
[[0.786 0.811 0.815 0.797 0.801 0.8   0 …[+189 chars]
```
</details>

> *thinking:* Found that adding the texture term to per-frame gradient optimization causes divergence (IoU drops to ~0.65), whereas silhouette-only refinement converges well (IoU 0.79→0.93–0.96). I'll refine poses using silhouette alone, and use texture as a discrete per-hypothesis score pooled over time to resolve flips.

**Bash**
```
cd /work; python - <<'EOF'
s=open('/work/scratch/code/fit.py').read()
s=s.replace('''    def losses(self, Tc, idx, w_app=1.0):
        idx=list(idx); it=torch.tensor(idx,device='cuda')
        m,g=self.render(Tc,idx)''','''    def losses(self, Tc, idx, w_app=1.0, rgb=True):
        idx=list(idx); it=torch.tensor(idx,device='cuda')
        m,g=self.render(Tc,idx,rgb=rgb)''')
s=s.replace('''        # appearance: interior high-pass NCC (rendered texture vs image), lighting-insensitive''','''        if not rgb: return {'iou':iou,'lb':lb,'loss':(1-iou)+lb}
        # appearance: interior high-pass NCC (rendered texture vs image), lighting-insensitive''')
s=s.replace('''            T=build(); L=self.losses(T,idx,w_app)
            loss=L['loss'].sum()''','''            T=build(); L=self.losses(T,idx,w_app,rgb=w_app>0)
            loss=L['loss'].sum()''')
open('/work/scratch/code/fit.py','w').write(s)
EOF
sed -i 's/T,L=Fi.optimize(np.stack(T0),ii,iters=80,lr_r=0.02,lr_t=0.004,fix_depth=True)/T,L=Fi.optimize(np.stack(T0),ii,iters=60,lr_r=0.02,lr_t=0.004,fix_depth=True,w_app=0.0)/' scratch/code/perframe.py
grep -n "optimize(" scratch/code/perframe.py; cd /work && timeout 900 python scratch/code/perframe.py > scratch/perframe.log 2>&1; tail -3 scratch/perframe.log | head -1; grep time scratch/perframe.log
```

<details><summary>output</summary>

```
23:    T,L=Fi.optimize(np.stack(T0),ii,iters=60,lr_r=0.02,lr_t=0.004,fix_depth=True,w_app=0.0)
880 fp base:0.96/-0.11 fp:0.97/-0.01 fx:0.87/0.43 fy:0.88/0.00 fz:0.88/0.15
time 96.2440242767334
```
</details>

**Bash**
```
cd /work; sed -n 3,95p scratch/perframe.log | paste - - 
```

<details><summary>output</summary>

```
0 fx base:0.93/-0.18 fp:0.93/-0.20 fx:0.94/0.04 fy:0.93/-0.03 fz:0.93/-0.03	10 fx base:0.93/-0.19 fp:0.93/-0.16 fx:0.93/0.01 fy:0.92/-0.03 fz:0.92/-0.05
20 fx base:0.92/-0.16 fp:0.92/-0.17 fx:0.93/0.01 fy:0.92/-0.01 fz:0.92/-0.06	30 fx base:0.92/-0.17 fp:0.92/-0.17 fx:0.93/0.01 fy:0.92/-0.00 fz:0.92/-0.06
40 fx base:0.92/-0.13 fp:0.92/-0.12 fx:0.93/0.01 fy:0.92/0.00 fz:0.92/-0.06	50 fx base:0.92/-0.21 fp:0.92/-0.16 fx:0.93/0.10 fy:0.92/-0.09 fz:0.92/-0.00
60 fx base:0.93/-0.09 fp:0.93/-0.09 fx:0.93/-0.00 fy:0.92/0.05 fz:0.92/-0.04	70 fy base:0.93/-0.11 fp:0.92/-0.07 fx:0.94/0.23 fy:0.94/0.02 fz:0.93/0.01
80 fy base:0.95/0.04 fp:0.95/-0.02 fx:0.96/-0.08 fy:0.96/0.06 fz:0.96/-0.11	90 fy base:0.95/0.03 fp:0.95/0.01 fx:0.96/-0.08 fy:0.96/0.07 fz:0.95/-0.05
100 fx base:0.93/-0.05 fp:0.93/-0.05  …[+6232 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/appscale.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from fit import *
from app import render_rgb
Fi=Fitter(); P=np.load('/work/scratch/perframe.npz'); hyps=P['hyps']; H=list(P['H'])
def scores(T,idx):
    it=torch.tensor(idx,device='cuda')
    with torch.no_grad():
        Fi.losses(T,idx,rgb=False); valid=Fi._valid
        m,rgb=render_rgb(Fi,T,idx)
        om=Fi.OM[it].float(); inner=(-Fn.max_pool2d(-(m>0.99).float()[:,None],5,1,2)[:,0])*valid*om
        g=rgb.mean(1); I=Fi.G[it]; out={}
        for name,(s1,s2) in {'hp1.5':(0,1.5),'hp3':(0,3.0),'dog1-3':(1.0,3.0),'dog1.5-5':(1.5,5.0),'raw':(None,None)}.items():
            if s1 is None: a,b=g,I
            else:
                a=(nblur(g,inner,s1) if s1>0 else g)-nblur(g,inner,s2); b=(nblur(I,inner,s1) if s1>0 else I)-nblur(I,inner,s2)
            out[name]=ncc(a,b,inner).cpu().numpy()
    return out
for a,b in [(0,70),(215,300),(400,466),(650,750),(845,900)]:
    idx=list(range(a,b,4)); res={}
    for hi,h in enumerate(H):
        res[h]=scores(torch.tensor(hyps[idx,hi],device='cuda'),idx)
    print(a,b)
    for k in res['base']:
        print('   ',k,' '.join(f"{h}:{res[h][k].mean():.3f}(win{np.mean([res[h][k][j]>=max(res[x][k][j] for x in H if x not in('fp',h)) for j in range(len(idx))]):.2f})" for h in H))
EOF
python scratch/code/appscale.py
```

<details><summary>output</summary>

```
0 70
    hp1.5 base:-0.138(win0.00) fp:-0.127(win0.00) fx:0.034(win0.61) fy:0.003(win0.39) fz:-0.035(win0.00)
    hp3 base:-0.080(win0.00) fp:-0.071(win0.00) fx:0.080(win1.00) fy:-0.027(win0.00) fz:-0.055(win0.00)
    dog1-3 base:0.009(win0.00) fp:0.015(win0.00) fx:0.158(win1.00) fy:-0.068(win0.00) fz:-0.087(win0.00)
    dog1.5-5 base:0.188(win0.56) fp:0.189(win0.39) fx:0.178(win0.44) fy:-0.118(win0.00) fz:0.022(win0.00)
    raw base:0.330(win0.83) fp:0.338(win0.56) fx:0.316(win0.17) fy:0.045(win0.00) fz:0.063(win0.00)
215 300
    hp1.5 base:-0.015(win0.55) fp:-0.019(win0.23) fx:-0.043(win0.00) fy:-0.006(win0.45) fz:-0.032(win0.00)
    hp3 base:-0.065(win0.86) fp:-0.065(win0.41) fx:-0.163(win0.00) fy:-0.119(win0.14) fz:-0.102(win0.00)
    dog1-3 base:-0.131(win1.00) fp:-0.126(win0.59) fx:- …[+1865 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/viewhyp2.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from fit import *
from app import render_rgb
Fi=Fitter(); P=np.load(sys.argv[3] if len(sys.argv)>3 else '/work/scratch/perframe.npz'); hyps=P['hyps']; H=list(P['H'])
frames=[int(x) for x in sys.argv[1].split(',')]
rows=[]
for fr in frames:
    T=torch.tensor(hyps[fr],device='cuda')
    with torch.no_grad(): m,rgb=render_rgb(Fi,T,[fr]*len(H))
    img=Fi.IM[fr].cpu().numpy(); tiles=[img]
    for i in range(len(H)):
        mm=m[i].cpu().numpy()[...,None]; tiles.append(mm*rgb[i].permute(1,2,0).cpu().numpy()+(1-mm)*img*0.3)
    t=np.hstack(tiles); ys,xs=np.nonzero(Fi.OM[fr].cpu().numpy()); c=int(ys.mean()) if len(ys) else 80
    t=t[max(c-40,0):max(c-40,0)+80]
    rows.append(cv2.resize((t*255).clip(0,255).astype(np.uint8),None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC))
cv2.imwrite(sys.argv[2],np.vstack(rows)[...,::-1])
EOF
python scratch/code/viewhyp2.py 0,40,420,700 scratch/vh2.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vh2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* Visual checks at frames 0, 40, 420, and 700 confirm the baseline orientation is correct during static phases, but the fine-scale texture score falsely favors a flip at frame 0 — so I'll rely on the coarse band-pass and raw-intensity scores instead, which match the visual check. Next, I'll test whether silhouette refinement reduces pose jitter during static phases as a ground-truth-free validation.

**Bash**
```
cd /work; cat > scratch/code/spread.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
K,Twc,Kp=load_cam()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Twc)@r['T_world_mesh'].astype(np.float64)
P=np.load('/work/scratch/perframe.npz'); hy=P['hyps'][:,0]
def spread(T):
    Rm=Rot.from_matrix(T[:,:3,:3]).mean().as_matrix()
    ang=np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum('ij,njk->nik',Rm.T,T[:,:3,:3])).as_rotvec(),axis=1))
    t=T[:,:3,3]; dt=np.linalg.norm(t-np.median(t,0),axis=1)*100
    # lateral (perp to ray) only
    ray=np.median(t,0)/np.linalg.norm(np.median(t,0)); tl=t-np.outer(t@ray,ray); dl=np.linalg.norm(tl-np.median(tl,0),axis=1)*100
    return np.median(ang),np.percentile(ang,90),np.median(dl),np.median(dt)
for a,b in [(0,70),(215,300),(400,466),(650,750),(845,900)]:
    s1=spread(Tb[a:b]); s2=spread(hy[a:b])
    print(a,b,'base rot med/p90 %.2f/%.2f lat %.2fcm all %.2fcm'%s1,'| refined %.2f/%.2f lat %.2fcm all %.2fcm'%s2)
EOF
python scratch/code/spread.py
```

<details><summary>output</summary>

```
0 70 base rot med/p90 0.40/2.00 lat 0.00cm all 1.66cm | refined 1.08/2.41 lat 0.12cm all 1.66cm
215 300 base rot med/p90 1.37/1.76 lat 0.41cm all 0.80cm | refined 1.52/2.60 lat 0.17cm all 0.72cm
400 466 base rot med/p90 2.15/3.50 lat 0.27cm all 0.92cm | refined 3.80/7.15 lat 0.13cm all 0.92cm
650 750 base rot med/p90 1.09/2.05 lat 0.21cm all 1.77cm | refined 5.14/12.78 lat 0.17cm all 2.17cm
845 900 base rot med/p90 11.81/23.67 lat 1.01cm all 3.72cm | refined 10.46/17.97 lat 2.04cm all 3.47cm
```
</details>

> *thinking:* Independent per-frame refinement actually increases pose noise during static phases (S4 rotation scatter jumps from 1.1° to 5.1°), so I'll instead use one pooled pose per static phase plus joint temporal optimization (silhouette + smoothness) for the carried segments. Next I'll examine where the baseline fails specifically in those carried segments.

**Bash**
```
cd /work; cat > scratch/code/ovseq.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from fit import *
from app import render_rgb
Fi=Fitter()
src=sys.argv[3] if len(sys.argv)>3 else 'base'
if src=='base':
    r=np.load(f'{INP}/reconstruction.npz'); T=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
else:
    T=np.linalg.inv(Fi.Twc)@np.load(src)['T_world_mesh']
a,b,s=map(int,sys.argv[1].split(','))
frames=list(range(a,b,s)); tiles=[]
with torch.no_grad(): m,rgb=render_rgb(Fi,torch.tensor(T[frames],device='cuda'),frames)
for i,fr in enumerate(frames):
    img=Fi.IM[fr].cpu().numpy(); mm=m[i].cpu().numpy()
    ren=mm[...,None]*rgb[i].permute(1,2,0).cpu().numpy()+(1-mm[...,None])*img
    ov=(img*255).astype(np.uint8).copy()
    for msk,col in [(Fi.OM[fr].cpu().numpy(),(0,255,0)),(mm>0.5,(255,0,0))]:
        cnt,_=cv2.findContours(msk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cnt,-1,col,1)
    t=np.vstack([(ren*255).astype(np.uint8),ov]); t=cv2.resize(t,None,fx=1.5,fy=1.5); cv2.putText(t,str(fr),(3,18),0,0.6,(255,255,0),2); tiles.append(t)
n=10
while len(tiles)%n: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(tiles[i:i+n]) for i in range(0,len(tiles),n)])[...,::-1])
EOF
python scratch/code/ovseq.py 130,180,5 scratch/sq1.jpg
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sq1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np
from scipy.ndimage import median_filter, binary_opening, binary_closing
m=np.load('/work/scratch/motion.npz'); d5=np.nan_to_num(m['d5'],nan=99.)
ms=np.load('/work/scratch/mask_stats.npz'); oa=ms['oa']
s=median_filter(d5,9)<6.5
s=binary_closing(s,np.ones(9)); s=binary_opening(s,np.ones(25))
segs=[];i=0
while i<len(s):
    if s[i]:
        j=i
        while j<len(s) and s[j]: j+=1
        segs.append((i,j-1)); i=j
    else: i+=1
print(segs)
EOF
```

<details><summary>output</summary>

```
[(4, 77), (215, 294), (405, 465), (658, 747), (848, 883)]
```
</details>

> *thinking:* Automatic static-phase detection confirms the four static intervals I identified visually, plus frames 848–883 which I'll treat as slow motion rather than a fixed pose since the book is being lowered there. Next, I'll fit one pooled pose per static phase.

**Bash**
```
cd /work; cat > scratch/code/static_fit.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from fit import *
from app import render_rgb
def rot_mean(Rs): return Rot.from_matrix(Rs).mean().as_matrix()
def coarse_app(Fi,T,idx):
    it=torch.tensor(idx,device='cuda')
    with torch.no_grad():
        Fi.losses(T,idx,rgb=False); valid=Fi._valid
        m,rgb=render_rgb(Fi,T,idx)
        om=Fi.OM[it].float(); inner=(-Fn.max_pool2d(-(m>0.99).float()[:,None],5,1,2)[:,0])*valid*om
        g=rgb.mean(1); I=Fi.G[it]
        a=nblur(g,inner,1.5)-nblur(g,inner,5.0); b=nblur(I,inner,1.5)-nblur(I,inner,5.0)
        s1=ncc(a,b,inner); s2=ncc(g,I,inner)
    return (s1+s2).cpu().numpy()/2
def fit_static(Fi,Tb,a,b,step=3):
    idx=list(range(a,b+1,step))
    T0=np.eye(4); T0[:3,:3]=rot_mean(Tb[a:b+1,:3,:3]); T0[:3,3]=np.median(Tb[a:b+1,:3,3],0)
    out={}
    for nm,Fl in FLIPS.items():
        T,L=Fi.optimize(np.stack([T0@Fl]),idx,iters=120,shared=True,lr_r=0.01,lr_t=0.003,fix_depth=True,w_app=0.0)
        Tt=torch.tensor(np.repeat(T[:1],len(idx),0),device='cuda')
        ap=coarse_app(Fi,Tt,idx)
        out[nm]=(T[0],float(L['iou'].mean()),float(ap.mean()),ap)
    return out
if __name__=='__main__':
    Fi=Fitter()
    r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
    res={}
    for a,b in [(0,77),(215,294),(405,465),(658,747)]:
        t0=time.time(); o=fit_static(Fi,Tb,a,b); res[(a,b)]=o
        for nm,(T,iou,ap,aps)  …[+531 chars]
```

<details><summary>output</summary>

```
0 77 id iou 0.923 app 0.266 (wins 0.92) dR_base 15.8 dt_cm [-0.7  0.7  0. ]
0 77 fx iou 0.929 app 0.227 (wins 0.08) dR_base 176.5 dt_cm [-0.9  0.7  0. ]
0 77 fy iou 0.925 app -0.018 (wins 0.00) dR_base 165.2 dt_cm [-1.2  0.6  0. ]
0 77 fz iou 0.920 app 0.065 (wins 0.00) dR_base 176.5 dt_cm [-1.2  0.6  0. ]
time 10.7
215 294 id iou 0.944 app -0.221 (wins 1.00) dR_base 4.2 dt_cm [-0.5  0.3 -0.2]
215 294 fx iou 0.938 app -0.446 (wins 0.00) dR_base 178.9 dt_cm [-0.5  0.3 -0.2]
215 294 fy iou 0.937 app -0.419 (wins 0.00) dR_base 178.2 dt_cm [-0.7  0.4 -0.2]
215 294 fz iou 0.940 app -0.301 (wins 0.00) dR_base 175.7 dt_cm [-0.7  0.4 -0.2]
time 7.3
405 465 id iou 0.949 app 0.297 (wins 0.76) dR_base 16.1 dt_cm [-1.1  0.2  0. ]
405 465 fx iou 0.955 app 0.222 (wins 0.24) dR_base 175.8 dt_cm [-1.2  0. …[+505 chars]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
import pickle
K,Twc,Kp=load_cam(); st=pickle.load(open('/work/scratch/static_fits.pkl','rb'))
r=np.load(f'{INP}/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
gp=json.load(open(f'{INP}/ground_plane.json')); n=np.array(gp['plane'][:3])
for (a,b),o in st.items():
    T=Twc@o['id']; Tm=Tw[(a+b)//2]
    for nm,X in [('fit',T),('base',Tm)]:
        ax=X[:3,:3]  # columns: mesh axes in world
        tilt=np.degrees(np.arccos(abs(ax[:,1]@n)))
        yaw=np.degrees(np.arctan2(ax[2,2],ax[0,2]))
        print(a,b,nm,'tilt of thickness axis from vertical %.1f'%tilt,'yaw of long axis %.1f'%yaw)
EOF
```

<details><summary>output</summary>

```
0 77 fit tilt of thickness axis from vertical 6.7 yaw of long axis 137.4
0 77 base tilt of thickness axis from vertical 1.7 yaw of long axis 122.1
215 294 fit tilt of thickness axis from vertical 6.1 yaw of long axis 43.2
215 294 base tilt of thickness axis from vertical 3.4 yaw of long axis 42.7
405 465 fit tilt of thickness axis from vertical 7.0 yaw of long axis 139.1
405 465 base tilt of thickness axis from vertical 0.4 yaw of long axis 124.1
658 747 fit tilt of thickness axis from vertical 3.3 yaw of long axis 26.6
658 747 base tilt of thickness axis from vertical 2.1 yaw of long axis 26.9
```
</details>

> *thinking:* The unconstrained fit produces an unrealistic 7° tilt and 15° yaw shift for the resting phases, which isn't physical for a book flat on a table. I'll constrain the pose so the cover stays flat on the plane, leaving only yaw and position free to fit.

**Bash**
```
cd /work; python - <<'EOF'
s=open('/work/scratch/code/fit.py').read()
s=s.replace("fix_depth=False, shared=False, extra=None):","fix_depth=False, shared=False, extra=None, yaw_axis=None):")
s=s.replace('''        w=torch.zeros(B,3,device='cuda',requires_grad=True); t=torch.zeros(B,3,device='cuda',requires_grad=True)''','''        w=torch.zeros(B,1 if yaw_axis is not None else 3,device='cuda',requires_grad=True); t=torch.zeros(B,3,device='cuda',requires_grad=True)
        ax=torch.tensor(np.asarray(yaw_axis),device='cuda',dtype=torch.float32) if yaw_axis is not None else None''')
s=s.replace('''            Rm=so3_exp(w)@T0[:B,:3,:3]; tt=T0[:B,:3,3]+t''','''            ww=w*ax[None] if ax is not None else w
            Rm=so3_exp(ww)@T0[:B,:3,:3]; tt=T0[:B,:3,3]+t''')
open('/work/scratch/code/fit.py','w').write(s)
EOF
cat > scratch/code/static_fit2.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from static_fit import *
def flatten(Rc, n_cam, ax_idx=1):
    """rotate Rc minimally so mesh axis ax_idx is parallel to +-n_cam"""
    a=Rc[:,ax_idx]; tgt=n_cam*np.sign(a@n_cam)
    v=np.cross(a,tgt); s=np.linalg.norm(v); c=a@tgt
    if s<1e-9: return Rc
    Rfix=Rot.from_rotvec(v/s*np.arctan2(s,c)).as_matrix(); return Rfix@Rc
def fit_static_flat(Fi,Tb,a,b,step=3,yaw_grid=None):
    gp=json.load(open('/work/inputs/ground_plane.json')); n=np.array(gp['plane'][:3]); n_cam=Fi.Twc[:3,:3].T@n
    idx=list(range(a,b+1,step))
    T0=np.eye(4); T0[:3,:3]=flatten(rot_mean(Tb …[+1215 chars]
```

<details><summary>output</summary>

```
0 77 id iou 0.802 app 0.599 (wins 1.00) dR_base 10.6 dt_cm [-0.8  0.7  0. ]
0 77 fx iou 0.808 app 0.466 (wins 0.00) dR_base 179.1 dt_cm [-1.   0.8  0. ]
0 77 fy iou 0.812 app 0.171 (wins 0.00) dR_base 168.7 dt_cm [-1.2  0.6  0. ]
0 77 fz iou 0.803 app 0.179 (wins 0.00) dR_base 178.6 dt_cm [-1.2  0.7  0. ]
215 294 id iou 0.883 app -0.122 (wins 0.56) dR_base 6.0 dt_cm [-0.5  0.3 -0.2]
215 294 fx iou 0.862 app -0.463 (wins 0.00) dR_base 177.0 dt_cm [-0.4  0.5 -0.2]
215 294 fy iou 0.861 app -0.194 (wins 0.00) dR_base 176.0 dt_cm [-0.7  0.4 -0.2]
215 294 fz iou 0.886 app -0.132 (wins 0.44) dR_base 178.4 dt_cm [-0.8  0.3 -0.2]
405 465 id iou 0.838 app 0.462 (wins 1.00) dR_base 9.6 dt_cm [-1.1  0.5  0. ]
405 465 fx iou 0.844 app 0.370 (wins 0.00) dR_base 179.6 dt_cm [-1.2  0.5  0. ]
405 465 fy io …[+466 chars]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
import pickle
K,Twc,Kp=load_cam(); st=pickle.load(open('/work/scratch/static_fits.pkl','rb'))
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Twc)@r['T_world_mesh'].astype(np.float64)
gp=json.load(open(f'{INP}/ground_plane.json')); n=np.array(gp['plane'][:3]); nc=Twc[:3,:3].T@n
print('floor normal in cam',np.round(nc,3))
for (a,b),o in st.items():
    T=o['id']; u=T[:3,1]*np.sign(T[:3,1]@nc); ub=Tb[(a+b)//2,:3,1]; ub=ub*np.sign(ub@nc)
    print(a,b,'fit up',np.round(u,3),'base up',np.round(ub,3), 'cent dir',np.round(T[:3,3]/np.linalg.norm(T[:3,3]),3))
EOF
```

<details><summary>output</summary>

```
floor normal in cam [ 0.002 -0.998 -0.07 ]
0 77 fit up [-0.015 -0.983 -0.184] base up [-0.007 -0.995 -0.098] cent dir [0.064 0.062 0.996]
215 294 fit up [ 0.048 -0.985 -0.166] base up [-0.014 -0.992 -0.127] cent dir [-0.128  0.3    0.945]
405 465 fit up [-0.003 -0.982 -0.19 ] base up [ 0.007 -0.997 -0.074] cent dir [0.029 0.057 0.998]
658 747 fit up [ 0.014 -0.992 -0.127] base up [-0.    -0.994 -0.107] cent dir [-0.162  0.285  0.945]
```
</details>

> *thinking:* The free silhouette fit consistently shows the book tilted ~6° toward the camera, but texture scores favor the flat-constrained pose across all samples, suggesting this tilt is mask bias on a thin silhouette rather than real geometry. I'll now compare these results against the baseline on the same score before concluding.

**Bash**
```
cd /work; cat > scratch/code/static_cmp.py <<'EOF'
import sys,time,pickle; sys.path.insert(0,'/work/scratch/code')
from static_fit2 import *
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
free=pickle.load(open('/work/scratch/static_fits.pkl','rb')); flat=pickle.load(open('/work/scratch/static_fits_flat.pkl','rb'))
gp=json.load(open('/work/inputs/ground_plane.json')); n=np.array(gp['plane'][:3]); n_cam=Fi.Twc[:3,:3].T@n
for (a,b) in free:
    idx=list(range(a,b+1,3))
    # flat + appearance
    T,L=Fi.optimize(np.stack([flat[(a,b)]['id']]),idx,iters=100,shared=True,lr_r=0.005,lr_t=0.002,fix_depth=True,w_app=1.0,yaw_axis=n_cam)
    cands={'base':Tb[idx],'free':np.repeat(free[(a,b)]['id'][None],len(idx),0),'flat':np.repeat(flat[(a,b)]['id'][None],len(idx),0),'flat+app':np.repeat(T[:1],len(idx),0)}
    s=[]
    for k,TT in cands.items():
        Tt=torch.tensor(TT,device='cuda'); ap=coarse_app(Fi,Tt,idx)
        with torch.no_grad(): L=Fi.losses(Tt,idx)
        s.append(f"{k}: iou {L['iou'].mean():.3f} app {ap.mean():.3f} hp {L['ncc'].mean():.3f}")
    print(a,b,' | '.join(s), 'flat+app dyaw %.1f'%np.degrees(np.linalg.norm(Rot.from_matrix(T[0,:3,:3].T@flat[(a,b)]['id'][:3,:3]).as_rotvec())))
    flat[(a,b)]['id_app']=T[0]
pickle.dump(flat,open('/work/scratch/static_fits_flat.pkl','wb'))
EOF
python scratch/code/static_cmp.py
```

<details><summary>output</summary>

```
0 77 base: iou 0.794 app 0.680 hp 0.057 | free: iou 0.923 app 0.266 hp -0.171 | flat: iou 0.802 app 0.599 hp 0.253 | flat+app: iou 0.798 app 0.632 hp 0.234 flat+app dyaw 2.8
215 294 base: iou 0.905 app -0.086 hp 0.090 | free: iou 0.944 app -0.221 hp -0.052 | flat: iou 0.883 app -0.122 hp 0.061 | flat+app: iou 0.872 app -0.003 hp 0.191 flat+app dyaw 0.5
405 465 base: iou 0.793 app 0.612 hp 0.237 | free: iou 0.949 app 0.297 hp -0.080 | flat: iou 0.838 app 0.462 hp 0.034 | flat+app: iou 0.794 app 0.643 hp 0.395 flat+app dyaw 5.3
658 747 base: iou 0.910 app 0.156 hp 0.305 | free: iou 0.924 app -0.005 hp 0.055 | flat: iou 0.898 app 0.114 hp 0.041 | flat+app: iou 0.874 app 0.084 hp 0.336 flat+app dyaw 2.1
```
</details>

> *thinking:* The baseline actually outperforms silhouette-driven refinement on texture accuracy in static phases, so I'll instead use a single robust pose per static phase (averaging rotation and median translation) to remove jitter while keeping the baseline's evidence. For carried phases, I'll run a joint temporal optimization combining a silhouette term, a robust prior toward the baseline, smoothness, and the static poses as anchors to fix only gross tracking errors.

**Bash**
```
cd /work; cat > scratch/code/segopt.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from fit import *
def vee_skew(M): return torch.stack([M[...,2,1]-M[...,1,2],M[...,0,2]-M[...,2,0],M[...,1,0]-M[...,0,1]],-1)/2
def seg_optimize(Fi, Tinit, Tprior, idx, anchor_l=None, anchor_r=None, iters=150, lr_r=0.01, lr_t=0.003,
                 w_sil=1.0, s_acc_t=0.006, s_acc_r=0.03, c_r=np.radians(15), c_t=0.05, w_prior=0.3, s_depth=0.03, chunk=80, has_ev=None):
    """Joint optimisation of a frame sequence. Tinit/Tprior (N,4,4) camera frame. anchors: fixed 4x4 before/after."""
    dev='cuda'; N=len(idx)
    T0=torch.tensor(Tinit,device=dev,dtype=torch.float32); TP=torch.tensor(Tprior,device=dev,dtype=torch.float32)
    w=torch.zeros(N,3,device=dev,requires_grad=True); t=torch.zeros(N,3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr_r},{'params':[t],'lr':lr_t}])
    ev=torch.tensor(has_ev if has_ev is not None else np.ones(N),device=dev,dtype=torch.float32)
    AL=torch.tensor(anchor_l,device=dev,dtype=torch.float32)[None] if anchor_l is not None else None
    AR=torch.tensor(anchor_r,device=dev,dtype=torch.float32)[None] if anchor_r is not None else None
    ray=TP[:,:3,3]/TP[:,:3,3].norm(dim=-1,keepdim=True)
    def build():
        R=so3_exp(w)@T0[:,:3,:3]; tt=T0[:,:3,3]+t
        return R,tt
    for i in range(iters):
        R,tt=build()
        T=torch.eye(4,device=dev).repeat(N,1,1); T[:,:3,:3]=R; T[:,:3,3]=tt
        # silhouett …[+2434 chars]
```

<details><summary>output</summary>

```
78 214 done 15.8
295 404 done 28.3
466 657 done 47.5
748 900 done 59.1
```
</details>

**Bash**
```
cd /work; python scratch/code/ovseq.py 130,180,5 scratch/sq2.jpg /work/scratch/seg_v1.npz
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sq2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* Joint optimisation stalled because poses rendered behind the tabletop get zero cost from the occlusion model, so there's no gradient toward the visible book. I'll add a per-frame 2D shift search to fix gross misplacements first, then rerun the joint optimisation from that starting point.

**Bash**
```
cd /work; cat > scratch/code/shiftinit.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from fit import *
def shift_search(Fi, Tc, idx, maxs=48, lam=0.5):
    """For each frame, find 2D shift of rendered silhouette that best matches the observed mask (occluders = person, legs).
    Returns (N,2) shifts in px, IoU before/after."""
    it=torch.tensor(idx,device='cuda')
    with torch.no_grad():
        m,_=Fi.render(torch.tensor(Tc,device='cuda'),idx,rgb=False)
        m=(m>0.5).float()
        om=Fi.OM[it].float(); occ=(Fi.PM[it]|Fi.LG[it]).float()
        bg=(1-om)*(1-occ)
        S=m.shape[-1]; P=S+2*maxs
        def corr(a,b):  # sum_x a(x-d) b(x) for all d
            A=torch.fft.rfft2(Fn.pad(a,(0,P-S,0,P-S))); B=torch.fft.rfft2(Fn.pad(b,(0,P-S,0,P-S)))
            c=torch.fft.irfft2(torch.conj(A)*B,s=(P,P))
            # index d (dy,dx) -> c[dy mod P, dx mod P]
            ds=torch.arange(-maxs,maxs+1,device='cuda')%P
            return c[:,ds][:,:,ds]
        ov=corr(m,om); fp=corr(m,bg); area_m=m.sum((1,2))[:,None,None]
        nocc=area_m-corr(m,occ)  # visible rendered area (approx)
        iou=ov/(om.sum((1,2))[:,None,None]+fp+ov*0).clamp(min=1)   # inter/(om + false-positive)
        best=iou.flatten(1).argmax(1); dy=best//(2*maxs+1)-maxs; dx=best%(2*maxs+1)-maxs
        i0=iou[:,maxs,maxs]; i1=iou.flatten(1).max(1).values
    return torch.stack([dx,dy],1).cpu().numpy(), i0.cpu().numpy(), i1.cpu().numpy()
def apply_shift(Fi,T,dxy):
    T=T.copy() …[+631 chars]
```

<details><summary>output</summary>

```
135 [  3.   -10.     0.12   0.44]
140 [-15.    25.     0.     0.43]
145 [-9.   25.    0.    0.64]
150 [-7.   14.    0.12  0.62]
155 [-7.   -1.    0.78  0.88]
160 [-7.   -2.    0.72  0.88]
165 [-7.   -3.    0.7   0.91]
170 [-9.   -4.    0.65  0.88]
175 [-7.   -3.    0.69  0.91]
320 [-4.    4.    0.8   0.94]
325 [-6.   14.    0.43  0.87]
330 [-4.   13.    0.39  0.8 ]
335 [-2.   -6.    0.66  0.88]
340 [-8.   -5.    0.65  0.83]
355 [-20.    -2.     0.36   0.47]
365 [ 0.   -4.    0.74  0.86]
460 [-5.    3.    0.74  0.84]
480 [-1.    5.    0.61  0.79]
565 [-4.  -5.   0.6  0.9]
570 [  1.   -12.     0.26   0.43]
585 [ 1.   26.    0.    0.64]
590 [12.   17.    0.1   0.83]
595 [-9.   -6.    0.47  0.87]
600 [-3.   -4.    0.65  0.79]
605 [-5.   -3.    0.58  0.77]
610 [-8.   -1.    0.55  0.77]
615 [-4. …[+609 chars]
```
</details>

**Bash**
```
cd /work; sed -i 's/        bg=(1-om)\*(1-occ)/        top=(Fi.TZ[it]<1e8).float()\n        bg=(1-om)*(1-occ)*(1-top)/' scratch/code/shiftinit.py; python scratch/code/shiftinit.py
```

<details><summary>output</summary>

```
135 [ 3.   -5.    0.39  0.76]
140 [ 1.   23.    0.    0.59]
145 [-8.   25.    0.    0.64]
150 [-7.   14.    0.14  0.62]
155 [-7.   -1.    0.78  0.88]
160 [-7.   -2.    0.72  0.88]
165 [-7.   -3.    0.7   0.91]
170 [-9.   -4.    0.65  0.88]
175 [-7.   -3.    0.69  0.91]
320 [-4.    4.    0.8   0.94]
325 [-6.   14.    0.43  0.87]
330 [-4.   13.    0.39  0.8 ]
335 [-2.   -6.    0.66  0.88]
340 [-8.   -5.    0.65  0.83]
355 [-2.   -5.    0.43  0.56]
460 [-5.    3.    0.74  0.85]
480 [-1.    6.    0.61  0.81]
565 [-3.   -4.    0.77  0.96]
580 [-8.   -6.    0.46  0.6 ]
585 [ 1.   26.    0.    0.64]
590 [12.   17.    0.1   0.83]
595 [-9.   -6.    0.47  0.87]
600 [-3.   -4.    0.65  0.79]
605 [-5.   -3.    0.58  0.77]
610 [-8.   -1.    0.55  0.77]
615 [-4.    0.    0.72  0.82]
620 [-3.    1.    0. …[+155 chars]
```
</details>

**Bash**
```
cd /work; cat > scratch/code/run_seg2.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from segopt import *
from shiftinit import shift_search, apply_shift
from static_fit import rot_mean
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
F=901; static=[(0,77),(215,294),(405,465),(658,747)]
out=Tb.copy(); SP={}
for a,b in static:
    T=np.eye(4); T[:3,:3]=rot_mean(Tb[a:b+1,:3,:3]); T[:3,3]=np.median(Tb[a:b+1,:3,3],0); out[a:b+1]=T; SP[(a,b)]=T
ms=np.load('/work/scratch/mask_stats.npz'); hasev=(ms['oa']>100).astype(float)
sh=np.load('/work/scratch/shifts.npy')
Tinit=Tb.copy()
g=(sh[:,3]-sh[:,2]>0.1)
Tinit[g]=apply_shift(Fi,Tb[g],sh[g,:2])
segs=[(78,214,(0,77),(215,294)),(295,404,(215,294),(405,465)),(466,657,(405,465),(658,747)),(748,900,(658,747),None)]
t0=time.time()
kw=eval('dict(%s)'%(sys.argv[2] if len(sys.argv)>2 else ''))
for a,b,l,rr in segs:
    idx=list(range(a,b+1))
    T=seg_optimize(Fi,Tinit[idx],Tinit[idx],idx,anchor_l=SP[l],anchor_r=SP[rr] if rr else None,has_ev=hasev[idx],**kw)
    out[idx]=T; print(a,b,'done %.1f'%(time.time()-t0))
np.savez(sys.argv[1],T_world_mesh=np.einsum('ij,njk->nik',Fi.Twc,out))
EOF
timeout 1200 python scratch/code/run_seg2.py /work/scratch/seg_v2.npz && python scratch/code/ovseq.py 130,180,5 scratch/sq2.jpg /work/scratch/seg_v2.npz
```

<details><summary>output</summary>

```
78 214 done 22.0
295 404 done 35.2
466 657 done 52.0
748 900 done 62.3
```
</details>

**Read**
```
/work/scratch/sq2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python scratch/code/ovseq.py 300,400,5 scratch/sq3.jpg /work/scratch/seg_v2.npz
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sq3.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The shift initialisation and joint optimisation fixes resolved the gross offsets at 135–150 and 325–330, and 300–400 now checks out; remaining mismatches like 350–355 trace back to incomplete SAM2 masks rather than tracking errors. Next I'll check the two remaining carried segments.

**Bash**
```
cd /work; python scratch/code/ovseq.py 470,670,10 scratch/sq4.jpg /work/scratch/seg_v2.npz
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sq4.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work; python scratch/code/ovseq.py 750,901,8 scratch/sq5.jpg /work/scratch/seg_v2.npz
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sq5.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The 470–900 render checks out visually—areas outside the mask are just covered by the hand or tabletop. Next I'll run per-phase validation numbers against baseline, test a held-out frame, and check for remaining rotation jumps.

**Bash**
```
cd /work; cat > scratch/code/jumps.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from common import *
r=np.load(f'{INP}/reconstruction.npz'); Tb=r['T_world_mesh'].astype(np.float64)
Tv=np.load(sys.argv[1])['T_world_mesh']
def dr(T): return np.r_[0,np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum('nji,njk->nik',T[:-1,:3,:3],T[1:,:3,:3])).as_rotvec(),axis=1))]
a=dr(Tb); b=dr(Tv)
print('frames with >8deg/frame: base',np.nonzero(a>8)[0].tolist()); print('new',np.nonzero(b>8)[0].tolist())
d=np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum('nji,njk->nik',Tb[:,:3,:3],Tv[:,:3,:3])).as_rotvec(),axis=1))
dt=np.linalg.norm(Tb[:,:3,3]-Tv[:,:3,3],axis=1)*100
for s in range(0,901,25): print(s, 'rot diff base-new %.1f'%d[s:s+25].mean(), 'max %.1f'%d[s:s+25].max(), 'trans diff %.1fcm'%dt[s:s+25].mean())
EOF
python scratch/code/jumps.py /work/scratch/seg_v2.npz
```

<details><summary>output</summary>

```
frames with >8deg/frame: base [109, 117, 123, 129, 136, 137, 138, 142, 146, 147, 148, 149, 150, 151, 156, 157, 159, 160, 166, 323, 324, 327, 328, 329, 331, 332, 334, 335, 352, 355, 362, 363, 373, 378, 487, 490, 491, 492, 493, 498, 564, 570, 573, 575, 580, 581, 582, 585, 586, 587, 588, 592, 593, 594, 607, 608, 812, 813, 814, 819, 820, 822, 825, 828, 829, 830, 870, 872, 873, 876, 877, 879, 880, 882, 883, 885, 886, 893, 894, 896, 897]
new []
0 rot diff base-new 0.8 max 0.8 trans diff 0.1cm
25 rot diff base-new 0.8 max 0.8 trans diff 1.6cm
50 rot diff base-new 2.0 max 4.7 trans diff 2.6cm
75 rot diff base-new 2.5 max 4.3 trans diff 1.1cm
100 rot diff base-new 5.2 max 11.9 trans diff 0.9cm
125 rot diff base-new 6.1 max 16.3 trans diff 5.8cm
150 rot diff base-new 6.6 max 18.9 trans diff 3.7cm
17 …[+1536 chars]
```
</details>

**Bash**
```
cd /work; sed -i "s/^ms=np.load('\/work\/scratch\/mask_stats.npz'); hasev=(ms\['oa'\]>100).astype(float)/ms=np.load('\/work\/scratch\/mask_stats.npz'); hasev=(ms['oa']>100).astype(float)\nimport os\nif os.environ.get('HOLDOUT'): hasev[np.arange(901)%4==2]=0/" scratch/code/run_seg2.py; grep -n HOLDOUT scratch/code/run_seg2.py; HOLDOUT=1 timeout 1200 python scratch/code/run_seg2.py /work/scratch/seg_v2_ho.npz
```

<details><summary>output</summary>

```
13:if os.environ.get('HOLDOUT'): hasev[np.arange(901)%4==2]=0
78 214 done 14.8
295 404 done 31.2
466 657 done 60.4
748 900 done 83.0
```
</details>

**Bash**
```
cd /work; cat > scratch/code/evaltab.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from fit import *
from static_fit import coarse_app
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tw={'base':r['T_world_mesh'].astype(np.float64)}
for a in sys.argv[1:]:
    k,p=a.split('='); Tw[k]=np.load(p)['T_world_mesh']
ms=np.load('/work/scratch/mask_stats.npz'); hasev=ms['oa']>100
gp=json.load(open(f'{INP}/ground_plane.json')); n=np.array(gp['plane'][:3]); d=gp['plane'][3]
m,v,f,uv,tex=load_mesh(); vs=v[::200].astype(np.float64)
phases=[('S1 table',0,77),('C1 carry',78,214),('S2 floor',215,294),('C2 carry',295,404),('S3 table',405,465),('C3 carry',466,657),('S4 floor',658,747),('C4 carry',748,829),('E held',830,900)]
met={}
for k,T in Tw.items():
    Tc=np.einsum('ij,njk->nik',np.linalg.inv(Fi.Twc),T)
    iou=np.zeros(901); app=np.zeros(901)
    for s in range(0,901,60):
        idx=list(range(s,min(s+60,901)))
        with torch.no_grad(): L=Fi.losses(torch.tensor(Tc[idx],device='cuda'),idx,rgb=False)
        iou[idx]=L['iou'].cpu().numpy(); app[idx]=coarse_app(Fi,torch.tensor(Tc[idx],device='cuda'),idx)
    rel=Rot.from_matrix(np.einsum('nji,njk->nik',T[:-1,:3,:3],T[1:,:3,:3])).as_rotvec()
    accr=np.r_[0,np.degrees(np.linalg.norm(rel[1:]-rel[:-1],axis=1)),0]
    acct=np.r_[0,np.linalg.norm(T[2:,:3,3]-2*T[1:-1,:3,3]+T[:-2,:3,3],axis=1)*100,0]
    low=np.array([ (vs@T[i,:3,:3].T+T[i,:3,3])@n+d for i in range(901)]).min(1)*100
    met[k]=(iou,app,accr,acct,low)
ho= …[+1140 chars]
```

<details><summary>output</summary>

```
| phase | frames | IoU base | IoU new | IoU ho | app base | app new | app ho | rot-acc base (deg) | rot-acc new (deg) | rot-acc ho (deg) | trans-acc base (cm) | trans-acc new (cm) | trans-acc ho (cm) | lowest pt above floor base (cm) | lowest pt above floor new (cm) | lowest pt above floor ho (cm) |
| S1 table | 0-77 | 0.793 | 0.790 | 0.790 | 0.684 | 0.676 | 0.676 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 89.4 | 89.4 | 89.4 |
| C1 carry | 78-214 | 0.710 | 0.836 | 0.830 | 0.087 | -0.039 | -0.021 | 2.66 | 0.18 | 0.17 | 1.83 | 0.14 | 0.14 | 71.4 | 63.1 | 63.1 |
| S2 floor | 215-294 | 0.904 | 0.906 | 0.906 | -0.085 | -0.081 | -0.081 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 | -0.5 | -0.5 | -0.5 |
| C2 carry | 295-404 | 0.717 | 0.843 | 0.835 | 0.171 | 0.111 | 0.115 | 3.87 | 0.22 | 0.22 | 1.92 …[+1106 chars]
```
</details>

> *thinking:* Held-out frame IoU improved from ~0.71–0.74 to ~0.83–0.85 in the carried phases, confirming the temporal model generalizes rather than just fitting masks, while static phases stayed essentially unchanged. However, texture scores dropped in C1 and C2, suggesting rotation may have worsened in parts of those segments, so I'll locate exactly where.

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np
m=np.load('/work/scratch/evalmet.npz'); b=m['base']; n=m['new']
for s in range(78,410,8):
    print(s, 'iou b %.2f n %.2f'%(b[0][s:s+8].mean(),n[0][s:s+8].mean()), 'app b %.2f n %.2f'%(b[1][s:s+8].mean(),n[1][s:s+8].mean()))
EOF
```

<details><summary>output</summary>

```
78 iou b 0.84 n 0.89 app b 0.63 n 0.49
86 iou b 0.83 n 0.90 app b 0.39 n 0.11
94 iou b 0.82 n 0.85 app b 0.17 n -0.02
102 iou b 0.78 n 0.82 app b 0.50 n 0.17
110 iou b 0.78 n 0.83 app b 0.22 n 0.21
118 iou b 0.78 n 0.87 app b 0.19 n 0.28
126 iou b 0.64 n 0.70 app b 0.16 n 0.16
134 iou b 0.06 n 0.31 app b 0.03 n 0.09
142 iou b 0.03 n 0.62 app b 0.02 n 0.24
150 iou b 0.64 n 0.79 app b 0.26 n 0.08
158 iou b 0.70 n 0.89 app b -0.08 n -0.49
166 iou b 0.69 n 0.92 app b -0.07 n -0.43
174 iou b 0.73 n 0.92 app b -0.18 n -0.33
182 iou b 0.85 n 0.92 app b -0.14 n -0.18
190 iou b 0.89 n 0.92 app b -0.16 n -0.32
198 iou b 0.90 n 0.93 app b -0.25 n -0.39
206 iou b 0.90 n 0.92 app b -0.21 n -0.31
214 iou b 0.91 n 0.90 app b -0.14 n -0.14
222 iou b 0.90 n 0.90 app b -0.14 n -0.11
230 iou b 0.92 n 0.91 ap …[+925 chars]
```
</details>

> *thinking:* The texture score drop correlates with the largest silhouette-driven rotation changes (frames 78–110, 158–210), mirroring the static-phase bias. I'll now test two fixes on segment C1: a stronger rotation prior toward baseline, and incorporating the texture term into the joint optimization for potential stability gains.

**Bash**
```
cd /work; python - <<'EOF'
s=open('/work/scratch/code/segopt.py').read()
s=s.replace("chunk=80, has_ev=None):","chunk=80, has_ev=None, w_app=0.0, c_r_deg=None):")
s=s.replace('''            L=Fi.losses(T2,idx[sl],rgb=False)''','''            L=Fi.losses(T2,idx[sl],w_app=w_app,rgb=w_app>0)''')
s=s.replace('''    dev='cuda'; N=len(idx)''','''    dev='cuda'; N=len(idx)
    if c_r_deg is not None: c_r=np.radians(c_r_deg)''')
open('/work/scratch/code/segopt.py','w').write(s)
EOF
cat > scratch/code/c1test.py <<'EOF'
import sys,time; sys.path.insert(0,'/work/scratch/code')
from segopt import *
from shiftinit import apply_shift
from static_fit import rot_mean, coarse_app
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
SP={}
for a,b in [(0,77),(215,294)]:
    T=np.eye(4); T[:3,:3]=rot_mean(Tb[a:b+1,:3,:3]); T[:3,3]=np.median(Tb[a:b+1,:3,3],0); SP[(a,b)]=T
ms=np.load('/work/scratch/mask_stats.npz'); hasev=(ms['oa']>100).astype(float)
ho=np.arange(901)%4==2; hev=hasev.copy(); hev[ho]=0
sh=np.load('/work/scratch/shifts.npy'); Tinit=Tb.copy(); g=(sh[:,3]-sh[:,2]>0.1); Tinit[g]=apply_shift(Fi,Tb[g],sh[g,:2])
a,b=78,214; idx=list(range(a,b+1))
for name,kw in [('wp1',dict(w_prior=1.0)),('wp3_c8',dict(w_prior=3.0,c_r_deg=8)),('app0.5',dict(w_app=0.5)),('app1_wp1',dict(w_app=1.0,w_prior=1.0))]:
    t0=time.time()
    T=seg_optimize(Fi,Tinit[idx],Tinit[idx],idx,anchor_l=SP[(0,77)],anchor_r=SP[(215,294)],has_ev=hev[idx],**kw)
     …[+376 chars]
```

<details><summary>output</summary>

```
wp1 iou 0.827  heldout-iou 0.840  app -0.005  (10s)
wp3_c8 iou 0.817  heldout-iou 0.830  app 0.018  (9s)
app0.5 iou 0.803  heldout-iou 0.816  app 0.099  (11s)
app1_wp1 iou 0.795  heldout-iou 0.808  app 0.135  (11s)
```
</details>

**Bash**
```
cd /work; cat >> scratch/code/c1test.py <<'EOF'
EOF
python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch/code')
from fit import *
from static_fit import coarse_app
Fi=Fitter()
r=np.load(f'{INP}/reconstruction.npz'); Tb=np.linalg.inv(Fi.Twc)@r['T_world_mesh'].astype(np.float64)
ms=np.load('/work/scratch/mask_stats.npz'); hasev=ms['oa']>100; ho=np.arange(901)%4==2
idx=list(range(78,215)); h=(hasev&ho)[idx]
v2=np.einsum('ij,njk->nik',np.linalg.inv(Fi.Twc),np.load('/work/scratch/seg_v2_ho.npz')['T_world_mesh'])[idx]
for name,T in [('base',Tb[idx]),('v2ho',v2)]+[(n,np.load(f'/work/scratch/c1_{n}.npy')) for n in ['wp1','wp3_c8','app0.5','app1_wp1']]:
    ap=coarse_app(Fi,torch.tensor(T,device='cuda'),idx); print(name,'heldout app %.3f'%ap[h].mean())
EOF
```

<details><summary>output</summary>

```
base heldout app 0.107
v2ho heldout app -0.012
wp1 heldout app 0.012
wp3_c8 heldout app 0.030
app0.5 heldout app 0.106
app1_wp1 heldout app 0.124
```
</details>

> *thinking:* The texture term in the joint optimization outperforms the baseline on held-out C1 frames (IoU 0.81 vs 0.72, texture 0.124 vs 0.107), while silhouette-only improved IoU but hurt texture agreement. I'll now run the full segment set with this setting and generate a held-out variant for the validation table.

**Bash**
```
cd /work; timeout 1500 python scratch/code/run_seg2.py /work/scratch/seg_v3.npz "w_app=1.0,w_prior=1.0" && HOLDOUT=1 timeout 1500 python scratch/code/run_seg2.py /work/scratch/seg_v3_ho.npz "w_app=1.0,w_prior=1.0" && python scratch/code/evaltab.py new=/work/scratch/seg_v3.npz ho=/work/scratch/seg_v3_ho.npz
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bw60gbzwn). Output is being written to: /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bw60gbzwn.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Agent:** While that runs, I'll write `solve.py` as a single self-contained script that recomputes everything from the inputs.

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
cat /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bw60gbzwn.output
```

<details><summary>output</summary>

```
78 214 done 12.6
295 404 done 30.7
466 657 done 58.7
748 900 done 82.1
78 214 done 22.0
295 404 done 39.1
466 657 done 73.1
748 900 done 95.1
| phase | frames | IoU base | IoU new | IoU ho | app base | app new | app ho | rot-acc base (deg) | rot-acc new (deg) | rot-acc ho (deg) | trans-acc base (cm) | trans-acc new (cm) | trans-acc ho (cm) | lowest pt above floor base (cm) | lowest pt above floor new (cm) | lowest pt above floor ho (cm) |
| S1 table | 0-77 | 0.793 | 0.790 | 0.790 | 0.684 | 0.676 | 0.676 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 89.4 | 89.4 | 89.4 |
| C1 carry | 78-214 | 0.710 | 0.802 | 0.798 | 0.087 | 0.138 | 0.116 | 2.66 | 0.34 | 0.29 | 1.83 | 0.20 | 0.18 | 71.4 | 63.1 | 62.9 |
| S2 floor | 215-294 | 0.904 | 0.906 | 0.906 | -0.085 | -0.081 | -0.081 | 0.00 | 0.00 | 0.00  …[+1268 chars]
```
</details>

> *thinking:* The texture-term run (v3) consistently boosts IoU across all phases and most texture scores, with only a slight dip in C2. Before the final run, I'll check the phase-detection edge handling in `solve.py` on the CPU.

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np
from scipy.ndimage import median_filter, binary_opening, binary_closing
d5=np.load('/work/scratch/motion.npz')['d5'].copy(); d5[:5]=np.nan
F=901
d5f=np.nan_to_num(d5,nan=99.); d5f[:5]=d5f[5]
st=median_filter(d5f,9)<6.5
st=binary_opening(binary_closing(st,np.ones(9)),np.ones(25))
ph=[];i=0
while i<F:
    if st[i]:
        j=i
        while j<F and st[j]: j+=1
        ph.append((i,j-1)); i=j
    else: i+=1
print(ph)
EOF
```

<details><summary>output</summary>

```
[(4, 77), (215, 294), (405, 465), (658, 747), (848, 883)]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
s=open('/work/outputs/solve.py').read()
old="""    st = median_filter(d5f, 9) < 6.5
    st = binary_opening(binary_closing(st, np.ones(9)), np.ones(25))"""
new="""    st = median_filter(d5f, 9, mode='nearest') < 6.5
    pad = 30   # edge padding so the morphology does not erode the first/last phase
    st = np.r_[np.repeat(st[0], pad), st, np.repeat(st[-1], pad)]
    st = binary_opening(binary_closing(st, np.ones(9)), np.ones(25))[pad:-pad]"""
assert old in s; s=s.replace(old,new); open('/work/outputs/solve.py','w').write(s)
EOF
python - <<'EOF'
import numpy as np
from scipy.ndimage import median_filter, binary_opening, binary_closing
d5=np.load('/work/scratch/motion.npz')['d5'].copy(); d5[:5]=np.nan
d5f=np.nan_to_num(d5,nan=99.); d5f[:5]=d5f[5]
st = median_filter(d5f, 9, mode='nearest') < 6.5
pad=30; st=np.r_[np.repeat(st[0],pad),st,np.repeat(st[-1],pad)]
st=binary_opening(binary_closing(st,np.ones(9)),np.ones(25))[pad:-pad]
print(np.nonzero(np.diff(st.astype(int)))[0], st[0], st[-1])
EOF
```

<details><summary>output</summary>

```
[ 77 214 294 404 465 657 747 847 883 898] True True
```
</details>

**Agent:** Phase edges are now handled correctly. Running the final `solve.py` once.

**Bash**
```
cd /work; (time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz) > /work/outputs/solve_log.txt 2>&1; tail -30 /work/outputs/solve_log.txt
```

<details><summary>output</summary>

```
Command running in background with ID: bqf4joqov. Output is being written to: /tmp/claude-1003/-work/c7ecb743-31dc-46df-9ac6-127602874e19/tasks/bqf4joqov.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Agent:** While it runs, I'll gather package versions for TOOLS.md.

**Bash**
```
python -c "
import torch,cv2,trimesh,h5py,numpy,scipy,nvdiffrast
print('torch',torch.__version__,'cuda',torch.version.cuda); print('opencv',cv2.__version__); print('trimesh',trimesh.__version__); print('h5py',h5py.__version__); print('numpy',numpy.__version__); print('scipy',scipy.__version__); print('nvdiffrast',nvdiffrast.__version__)
"; ffmpeg -version | head -1; python --version; ls /work/tools; pip list --user 2>/dev/null | head
```

<details><summary>output</summary>

```
torch 2.5.1+cu124 cuda 12.4
opencv 4.11.0
trimesh 5.1.1
h5py 3.16.0
numpy 1.26.3
scipy 1.16.3
nvdiffrast 0.4.0
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
Python 3.11.10
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
sleep 100; cat /work/outputs/solve_log.txt
```

<details><summary>output</summary>

```
[92s] decoded 901 frames
```
</details>

**Bash**
```
sleep 110; cat /work/outputs/solve_log.txt
```

<details><summary>output</summary>

```
[92s] decoded 901 frames
rest phases [(0, 77, 0.894), (215, 294, -0.005), (405, 465, 0.888), (658, 747, 0.015)]
table-top height (baseline rest-on-table poses) 0.891 m
[129s] shift-corrected frames: 165
[156s] segment 78-214 optimised
[177s] segment 295-404 optimised
[210s] segment 466-657 optimised
[227s] segment 748-900 optimised
```
</details>

**Bash**
```
sleep 45; cat /work/outputs/solve_log.txt
```

<details><summary>output</summary>

```
[92s] decoded 901 frames
rest phases [(0, 77, 0.894), (215, 294, -0.005), (405, 465, 0.888), (658, 747, 0.015)]
table-top height (baseline rest-on-table poses) 0.891 m
[129s] shift-corrected frames: 165
[156s] segment 78-214 optimised
[177s] segment 295-404 optimised
[210s] segment 466-657 optimised
[227s] segment 748-900 optimised
baseline: mean IoU 0.793  mean texture score 0.202
result: mean IoU 0.838  mean texture score 0.252
wrote /work/outputs/object_poses_world.npz  (901 frames)  total 230s

real	3m51.947s
user	3m38.983s
sys	0m59.734s
```
</details>

> *thinking:* `solve.py` completed in 3m52s (2.5m on GPU), improving mean IoU from 0.793 to 0.838 and texture score from 0.202 to 0.252. I'll now check the output file and build the final per-phase table.

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np
from scipy.spatial.transform import Rotation as R
o=np.load('/work/outputs/object_poses_world.npz'); T=o['T_world_mesh']; print(o.files, T.shape, T.dtype, np.isfinite(T).all())
print('orthonormal err', np.abs(np.einsum('nji,njk->nik',T[:,:3,:3],T[:,:3,:3])-np.eye(3)).max(), 'det', np.linalg.det(T[:,:3,:3]).min())
v=np.load('/work/scratch/seg_v3.npz')['T_world_mesh']
d=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik',T[:,:3,:3],v[:,:3,:3])).as_rotvec(),axis=1))
print('vs scratch v3: rot diff median %.2f max %.2f, trans diff max %.2f cm'%(np.median(d),d.max(),np.abs(T[:,:3,3]-v[:,:3,3]).max()*100))
b=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
db=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik',T[:,:3,:3],b[:,:3,:3])).as_rotvec(),axis=1))
print('vs baseline: rot diff median %.1f p90 %.1f max %.1f; trans median %.1f cm p90 %.1f max %.1f'%(np.median(db),np.percentile(db,90),db.max(),*(np.percentile(np.linalg.norm(T[:,:3,3]-b[:,:3,3],axis=1)*100,[50,90,100]))))
EOF
python scratch/code/evaltab.py final=/work/outputs/object_poses_world.npz ho=/work/scratch/seg_v3_ho.npz 2>&1 | tail -22
```

<details><summary>output</summary>

```
['T_world_mesh'] (901, 4, 4) float64 True
orthonormal err 5.874475090705822e-06 det 0.9999954730365116
vs scratch v3: rot diff median 0.25 max 3.28, trans diff max 2.17 cm
vs baseline: rot diff median 1.9 p90 7.0 max 23.3; trans median 1.4 cm p90 4.2 max 14.0
| phase | frames | IoU base | IoU final | IoU ho | app base | app final | app ho | rot-acc base (deg) | rot-acc final (deg) | rot-acc ho (deg) | trans-acc base (cm) | trans-acc final (cm) | trans-acc ho (cm) | lowest pt above floor base (cm) | lowest pt above floor final (cm) | lowest pt above floor ho (cm) |
| S1 table | 0-77 | 0.793 | 0.790 | 0.790 | 0.684 | 0.676 | 0.676 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 89.4 | 89.4 | 89.4 |
| C1 carry | 78-214 | 0.710 | 0.800 | 0.798 | 0.087 | 0.140 | 0.116 | 2.66 | 0.33 | 0.29 | 1.83 |  …[+1392 chars]
```
</details>

**Bash**
```
cd /work; python - <<'EOF'
import numpy as np
m=np.load('/work/scratch/evalmet.npz'); ms=np.load('/work/scratch/mask_stats.npz'); ev=ms['oa']>100; ho=np.arange(901)%4==2
for nm,a,b in [('C1',78,214),('C2',295,404),('C3',466,657),('C4',748,829),('E',830,900)]:
    s=np.zeros(901,bool); s[a:b+1]=True; s&=ev&ho
    print(nm,'held-out texture: base %.3f ho %.3f'%(m['base'][1][s].mean(),m['ho'][1][s].mean()))
EOF
```

<details><summary>output</summary>

```
C1 held-out texture: base 0.107 ho 0.126
C2 held-out texture: base 0.184 ho 0.196
C3 held-out texture: base 0.245 ho 0.349
C4 held-out texture: base 0.003 ho 0.067
E held-out texture: base 0.118 ho 0.177
```
</details>

> *thinking:* The held-out frames confirm the texture score outperforms baseline across all carried phases (C1–C4 and the end segment). Now I'll write up the report.

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
cd /work; python scratch/code/jumps.py /work/outputs/object_poses_world.npz | head -2 | python -c "
import sys,ast
l=sys.stdin.read().splitlines(); b=ast.literal_eval(l[0].split('base',1)[1].strip()); print(len(b), l[1])"; ls -la /work/outputs
```

<details><summary>output</summary>

```
81 new []
Traceback (most recent call last):
  File "/work/scratch/code/jumps.py", line 10, in <module>
    for s in range(0,901,25): print(s, 'rot diff base-new %.1f'%d[s:s+25].mean(), 'max %.1f'%d[s:s+25].max(), 'trans diff %.1fcm'%dt[s:s+25].mean())
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BrokenPipeError: [Errno 32] Broken pipe
total 172
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 21 01:53 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 21 01:05 ..
-rw-rw-r-- 1 hanzhou hanzhou  13241 Oct 21 01:53 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   1440 Oct 21 01:48 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 115606 Oct 21 01:51 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou  20553 Oct 21 01:47 solve.p …[+64 chars]
```
</details>

**Agent:** I produced the full 901-frame trajectory, and on my own checks it beats the baseline in every phase where the book is moving. I can't measure the actual ADD score, since there is no ground truth. `solve.py` ran once at the end with no errors, in 3 min 52 s. The output has all 901 frames as valid rigid transforms. Total GPU use was about 24 of the 45 allowed minutes, never above about 5.5 GB.

**What happens in the video.** The book is moved four times, not once:
- It lies on the table (frames 0–77), on the floor (215–294), on the table again (405–465) and on the floor again (658–747), with carrying in between.
- During 466–657 it is turned over in the hands.
- From 830 to the end it is held flat just behind the table's far edge, so the tabletop hides most of it.

The four resting phases are detected automatically from image motion.

**What the baseline gets wrong, and what I changed:**
- **Resting phases:** the baseline is already right here. Its orientation beats all three 180° flips on a texture comparison pooled over each phase, and I confirmed it by eye from the spine logo and the covers. So there is no symmetry flip in this episode. I only replaced each resting phase with one averaged pose, which removes the frame-to-frame jitter.
- **Carried phases:** the baseline lags and jumps, sometimes 5–15 cm away from the visible book (e.g. frames 135–150, 325–330, 585–590), with 81 frames jumping more than 8° between frames.
  - I fix gross misplacements with a 2-D shift search of the rendered outline against the mask.
  - Then I optimise each carried stretch as a whole: outline and texture match, smoothness, a robust pull toward the baseline, and the resting poses as fixed end points.
- **The tabletop as an occluder:** I model it explicitly, because the book is behind it in several stretches. Without that, the baseline looks wrong at 830–900 when it isn't.

**Validation without ground truth.** For the held-out test I re-ran the method with every 4th frame's image evidence removed, then scored those frames:

| | carried phases, baseline | carried phases, result |
|---|---|---|
| Mask overlap (IoU), held-out frames | 0.71–0.93 | 0.81–0.96 |
| Texture agreement, held-out frames | 0.00–0.25 | 0.07–0.35 |

The texture score also improves in every carried phase. In the resting phases both scores stay within ±0.01 of the baseline.

**Things I tested and rejected:**
- Refining each frame on its own made the poses noisier.
- Fitting resting poses to the outline alone tilts the book about 6°. The masks seem to be biased by a pixel or two, and the texture score gets worse.
- Correcting depth: the depth cues (MoGe-2, floor contact, table height from the legs, outline size) disagree by up to 20 cm and in different directions. So I kept the baseline's distance from the camera, only averaged or smoothed.

**Confidence:**
- **Resting phases:** high, though the gain there is small.
- **Carried phases:** medium. The large position errors are clearly fixed, but rotation is a compromise. Frames where the book is almost fully hidden are interpolated, and the fast turn-over around frames 500–550 is the least certain.
- **Final held segment (830–900):** medium-low, because only a thin strip of the book is visible.

**Things to know:**
- The two table-leg rectangles in `solve.py` are constants I read off the background image by eye. They are commented as such.
- Reruns can differ by well under a millimetre because of the GPU renderer.
- No external tools, packages or weights were downloaded. TOOLS.md lists the container software used.

Files are in `/work/outputs`:
- object_poses_world.npz
- solve.py
- REPORT.md
- TOOLS.md
- solve_log.txt

## Final

I produced the full 901-frame trajectory, and on my own checks it beats the baseline in every phase where the book is moving. I can't measure the actual ADD score, since there is no ground truth. `solve.py` ran once at the end with no errors, in 3 min 52 s. The output has all 901 frames as valid rigid transforms. Total GPU use was about 24 of the 45 allowed minutes, never above about 5.5 GB.

**What happens in the video.** The book is moved four times, not once:
- It lies on the table (frames 0–77), on the floor (215–294), on the table again (405–465) and on the floor again (658–747), with carrying in between.
- During 466–657 it is turned over in the hands.
- From 830 to the end it is held flat just behind the table's far edge, so the tabletop hides most of it.

The four resting phases are detected automatically from image motion.

**What the baseline gets wrong, and what I changed:**
- **Resting phases:** the baseline is already right here. Its orientation beats all three 180° flips on a texture comparison pooled over each phase, and I confirmed it by eye from the spine logo and the covers. So there is no symmetry flip in this episode. I only replaced each resting phase with one averaged pose, which removes the frame-to-frame jitter.
- **Carried phases:** the baseline lags and jumps, sometimes 5–15 cm away from the visible book (e.g. frames 135–150, 325–330, 585–590), with 81 frames jumping more than 8° between frames.
  - I fix gross misplacements with a 2-D shift search of the rendered outline against the mask.
  - Then I optimise each carried stretch as a whole: outline and texture match, smoothness, a robust pull toward the baseline, and the resting poses as fixed end points.
- **The tabletop as an occluder:** I model it explicitly, because the book is behind it in several stretches. Without that, the baseline looks wrong at 830–900 when it isn't.

**Validation without ground truth.** For the held-out test I re-ran the method with every 4th frame's image evidence removed, then scored those frames:

| | carried phases, baseline | carried phases, result |
|---|---|---|
| Mask overlap (IoU), held-out frames | 0.71–0.93 | 0.81–0.96 |
| Texture agreement, held-out frames | 0.00–0.25 | 0.07–0.35 |

The texture score also improves in every carried phase. In the resting phases both scores stay within ±0.01 of the baseline.

**Things I tested and rejected:**
- Refining each frame on its own made the poses noisier.
- Fitting resting poses to the outline alone tilts the book about 6°. The masks seem to be biased by a pixel or two, and the texture score gets worse.
- Correcting depth: the depth cues (MoGe-2, floor contact, table height from the legs, outline size) disagree by up to 20 cm and in different directions. So I kept the baseline's distance from the camera, only averaged or smoothed.

**Confidence:**
- **Resting phases:** high, though the gain there is small.
- **Carried phases:** medium. The large position errors are clearly fixed, but rotation is a compromise. Frames where the book is almost fully hidden are interpolated, and the fast turn-over around frames 500–550 is the least certain.
- **Final held segment (830–900):** medium-low, because only a thin strip of the book is visible.

**Things to know:**
- The two table-leg rectangles in `solve.py` are constants I read off the background image by eye. They are commented as such.
- Reruns can differ by well under a millimetre because of the GPU renderer.
- No external tools, packages or weights were downloaded. TOOLS.md lists the container software used.

Files are in `/work/outputs`:
- object_poses_world.npz
- solve.py
- REPORT.md
- TOOLS.md
- solve_log.txt
