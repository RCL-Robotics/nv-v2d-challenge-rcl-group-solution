# CL2b_ep022

Claude Opus v2（ep11-21）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T13:24:09Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude -R | head; nvidia-smi; ls /pipeline -R | head -50
```

<details><summary>output</summary>

```
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 17:17 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 17:17 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 17:17 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 17:17 proxy.sock

inputs:
total 74824
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 17:17 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 17:17 ..
-rw-rw-r-- 1 hanzhou hanzhou      845 Oct 21 17:17 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       66 Oct 21 17:17 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   194478 Oct 21 17:17 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      397 Oct 21 17:17 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  7202856 Oct 21 17:17 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 12306000 Oct 21 17:17 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 3631 …[+5331 chars]
```
</details>

**Bash**
```
cd /work; echo $CUDA_VISIBLE_DEVICES; ls /work/scratch/.claude/projects/-work/ 2>/dev/null; cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null; cat logs/run_manifest.txt; cat /pipeline/pipeline_report.json | head -50; python -c "import torch,nvdiffrast;print(torch.__version__, torch.cuda.device_count())"; nproc; free -g
```

<details><summary>output</summary>

```
0
9c970dd7-9b9a-465b-a801-f693e8bb11a4.jsonl
memory
date_utc: 2026-10-21T13:24:09Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2b_ep022:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000022:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
5ccb21d394cdc5b2df46bbb69849fa205d0def02b2903de81a697217d06487f5  ./camera.json
63de52842fb945be0b021862d9b43fd2d678fa26bd08df8f0849997a5be8017a  ./episode.json
d50 …[+2653 chars]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/an outputs tools; python -c "
import numpy as np, trimesh, h5py
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m, m.bounds, m.extents)
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
print(type(m.visual), getattr(m.visual,'material',None))
r=np.load('inputs/reconstruction.npz'); 
for k in r: print(k, r[k].shape, r[k].dtype)
f=h5py.File('inputs/masks.h5'); print({k:f[k].shape for k in f})
"

```

<details><summary>output</summary>

```
<trimesh.Trimesh(vertices.shape=(179840, 3), faces.shape=(59956, 3))> [[-0.21632116 -0.1382958  -0.26096481]
 [ 0.21404727  0.21074353  0.34480789]] [0.43036842 0.34903933 0.6057727 ]
[[-0.01506124 -0.99988002  0.00361986  0.03968606]
 [-0.99912787  0.01519072  0.03889399  0.00598699]
 [-0.03894431 -0.00303091 -0.99923679  0.03888342]
 [ 0.          0.          0.          1.        ]] [0.34991471 0.43019378 0.61279299]
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x740a97907050>
T_world_mesh (719, 4, 4) float32
T_cam_mesh_foundationpose (719, 4, 4) float32
T_cam_mesh_coconet (719, 4, 4) float32
human_vertices_world (719, 4610, 3) float32
human_keypoints_world (719, 70, 3) float32
contact_prob_right_left (719, 2) float32
{'object': (719, 11 …[+39 chars]
```
</details>

**Bash**
```
cd /work/scratch; mkdir -p frames; ffmpeg -loglevel error -i /work/inputs/video.mp4 -q:v 2 frames/%04d.jpg; ls frames | wc -l; ffprobe -v error -show_entries stream=r_frame_rate,nb_frames /work/inputs/video.mp4
```

<details><summary>output</summary>

```
719
[STREAM]
r_frame_rate=30/1
nb_frames=719
[/STREAM]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import cv2,numpy as np,h5py
f=h5py.File('/work/inputs/masks.h5')
idx=list(range(0,719,30))
tiles=[]
for i in idx:
  im=cv2.imread('frames/%04d.jpg'%(i+1))
  o=f['object'][i]; p=f['person'][i]
  im[o]=(im[o]*0.5+np.array([0,0,255])*0.5).astype(np.uint8)
  im[p]=(im[p]*0.7+np.array([255,0,0])*0.3).astype(np.uint8)
  im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(10,30),0,1,(0,255,0),2); tiles.append(im)
while len(tiles)%5: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[k:k+5]) for k in range(0,len(tiles),5)]
cv2.imwrite('an/sheet.jpg',np.vstack(rows))
a=f['object'][:].reshape(719,-1).sum(1); print(a[::10])
"

```

<details><summary>output</summary>

```
[11041 11214 10889 10707 10558 10653 10450 10286 10294 10227 10729 10735
 10218 10757 10125 10158  8601  7560  6909  6907  6991  7432  9477 12330
 14083 13924 13695 15178 14996 16746 17239 16409 18416 17825 16710 15399
 14434 14749 14273 13339 11407 12044 11399 10232  9504  9005  8710  8178
  8614  7716  7326  6726  6120  5638  5730  5531  4557  4881  4817  4476
  4493  4367  4860  4902  4887  4823  5137  4926  4874  5155  5234  5219]
```
</details>

**Read**
```
/work/scratch/an/sheet.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
mkdir -p /work/outputs /work/scratch/code; cat > /work/scratch/code/common.py <<'EOF'
import numpy as np, trimesh, json, torch, h5py, cv2
import nvdiffrast.torch as dr
INP='/work/inputs'
def load_mesh():
    m=trimesh.load(f'{INP}/object_mesh.glb',force='scene',process=False).to_geometry()
    return m
def load_cam():
    c=json.load(open(f'{INP}/camera.json'))
    return np.array(c['K']),np.array(c['T_world_cam']),np.array(c['K_pipeline'])
class Renderer:
    def __init__(self,mesh,K,H=1152,W=1536,dev='cuda'):
        self.glctx=dr.RasterizeCudaContext()
        self.dev=dev
        self.v=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev)
        self.f=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
        uv=mesh.visual.uv
        self.uv=torch.tensor(uv,dtype=torch.float32,device=dev)
        img=mesh.visual.material.baseColorTexture
        tex=np.asarray(img.convert('RGB')).astype(np.float32)/255.
        self.tex=torch.tensor(tex,device=dev)[None]  # trimesh uv origin bottom-left -> flip
        self.tex=torch.flip(self.tex,[1]).contiguous()
        self.K=K;self.H=H;self.W=W
    def proj(self,K,H,W,n=0.01,f=20.):
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
        # opencv cam -> clip ; y down in image; nvdiffrast clip y up -> row 0 at bottom; we flip later
        P=np.array([[2*fx/W,0,(2*cx-W)/W,0],[0,2*fy/H,(2*cy-H)/H,0],[0,0,(f+n)/(f-n),-2*f*n/(f-n)],[0,0,1,0]],dtype=np.float32)
        return torch.tensor(P,device=self.dev)
    def render(self …[+2043 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/an/base_ov.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/an/base_ov2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
from common import *
from scipy.spatial.transform import Rotation as Rot
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
T=r['T_world_mesh'].astype(np.float64)
def rdiff(A,B): return np.degrees(Rot.from_matrix(A[:,:3,:3].transpose(0,2,1)@B[:,:3,:3]).magnitude())
dr_=rdiff(T[:-1],T[1:]); dt=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)*100
Tc=Tcw@T
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
np.set_printoptions(precision=1,suppress=True,linewidth=250)
print('rot step deg (every 5 frames sum)'); print(np.add.reduceat(dr_,range(0,718,5)))
print('trans step cm'); print(np.add.reduceat(dt,range(0,718,5)))
print('cam z'); print(Tc[::10,2,3])
print('fp z'); print(fp[::10,2,3]); print('cc z'); print(cc[::10,2,3])
print('rot base vs cc', rdiff(Tc,cc)[::10]); print('rot base vs fp', rdiff(Tc,fp)[::10])
print('contact', r['contact_prob_right_left'][::10].T)
"

```

<details><summary>output</summary>

```
rot step deg (every 5 frames sum)
[ 2.1  1.9  2.6  1.5  1.6  1.7  1.5  1.6  1.3  1.8  1.4  1.9  1.9  1.7  2.   2.4  1.9  1.8  1.   2.3  2.4  1.9  1.3  1.2  1.3  1.   1.6  2.1  2.4  1.3  2.   1.8  2.7  1.7  1.8  2.   1.1  2.7  3.3  2.1  3.   4.2  8.4 18.7 22.  28.6 25.2 13.5  8.7
  5.   4.3  5.6  6.4  5.7  3.8  6.8  6.9  7.8  5.1  5.1  8.3 10.8  8.4  9.4 14.4 10.7  6.2  2.9  2.5  4.7  3.5  4.   2.5  2.7  3.9  5.5  7.9  8.3  9.5 10.5  9.1  8.9  6.9  4.   7.1  5.8  3.5  4.9  3.   4.2  5.7  6.2  4.9  9.7 11.8 11.8  9.8  5.
  1.8  2.8  3.2  4.4  3.4  3.3  6.4  3.   3.3  5.3 11.1 15.8 16.1  5.2  2.3  3.8  2.5  2.7  2.8  3.7  2.4  2.7  3.9  2.9  2.1  3.7  1.9  2.3  2.1  2.5  2.4  2.8  2.5  3.8  4.1  3.8  4.6  2.7  3.1  5.3  4.5  4.1  4.1  6.4  3.6  3.1]
trans step cm
[ 4.5  5.7  6.7  3.1  5.5  9. …[+2788 chars]
```
</details>

**Bash**
```
python -c "
from common import *
f=h5py.File('/work/inputs/masks.h5')
O=f['object'][:]; 
np.save('/work/scratch/an/objmask_ds.npy',O[:,::4,::4])
P=f['person'][:]; np.save('/work/scratch/an/permask_ds.npy',P[:,::4,::4])
cen=[];bb=[]
for o in O:
  ys,xs=np.nonzero(o); cen.append([xs.mean(),ys.mean()]); bb.append([xs.min(),ys.min(),xs.max(),ys.max()])
cen=np.array(cen);bb=np.array(bb); np.save('/work/scratch/an/cen.npy',cen); np.save('/work/scratch/an/bb.npy',bb)
np.set_printoptions(precision=0,suppress=True,linewidth=250)
print(np.c_[np.arange(0,719,10),cen[::10],bb[::10]])
"

```

<details><summary>output</summary>

```
[[   0.  668.  818.  604.  713.  753.  926.]
 [  10.  668.  818.  603.  714.  753.  926.]
 [  20.  669.  818.  603.  713.  753.  926.]
 [  30.  669.  819.  603.  714.  752.  927.]
 [  40.  669.  819.  603.  714.  752.  927.]
 [  50.  669.  819.  603.  713.  753.  926.]
 [  60.  669.  819.  603.  713.  752.  926.]
 [  70.  669.  820.  603.  713.  752.  926.]
 [  80.  669.  820.  603.  714.  752.  926.]
 [  90.  669.  821.  603.  714.  752.  926.]
 [ 100.  669.  820.  603.  714.  753.  926.]
 [ 110.  669.  820.  604.  714.  753.  926.]
 [ 120.  669.  820.  603.  713.  753.  926.]
 [ 130.  669.  819.  604.  713.  753.  926.]
 [ 140.  668.  821.  603.  714.  753.  927.]
 [ 150.  669.  821.  603.  714.  753.  927.]
 [ 160.  658.  819.  603.  713.  751.  927.]
 [ 170.  654.  825.  603.  712.  75 …[+2440 chars]
```
</details>

**Bash**
```
python -c "
from common import *
r=load_rec(); T=r['T_world_mesh'].astype(np.float64)
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n=pl[:3]; d=pl[3]
print('n',n, np.linalg.norm(n))
m=load_mesh(); V=m.vertices
for i in [0,100,150,200,300,400,500,600,700]:
  R=T[i,:3,:3]; print(i, 'mesh axes dot n:', np.round(R.T@n,3), end=' ')
  W=V@R.T+T[i,:3,3]; s=W@n+d
  print('signed dist min/5pct/max', np.round([s.min(),np.percentile(s,1),s.max()],3))
"

```

<details><summary>output</summary>

```
n [-0.01742591 -0.99852898 -0.0513441 ] 1.0
0 mesh axes dot n: [-0.011 -0.002  1.   ] signed dist min/5pct/max [0.007 0.013 0.616]
100 mesh axes dot n: [-0.006  0.006  1.   ] signed dist min/5pct/max [0.011 0.016 0.619]
150 mesh axes dot n: [-0.008  0.011  1.   ] signed dist min/5pct/max [0.011 0.017 0.62 ]
200 mesh axes dot n: [-0.035  0.019  0.999] signed dist min/5pct/max [-0.021 -0.014  0.593]
300 mesh axes dot n: [-0.014  0.486  0.874] signed dist min/5pct/max [-0.037 -0.026  0.549]
400 mesh axes dot n: [0.002 0.427 0.904] signed dist min/5pct/max [-0.036 -0.029  0.557]
500 mesh axes dot n: [0.021 0.399 0.917] signed dist min/5pct/max [-0.027 -0.019  0.571]
600 mesh axes dot n: [ 0.013 -0.011  1.   ] signed dist min/5pct/max [-0.     0.007  0.611]
700 mesh axes dot n: [-0.007 -0.002   …[+51 chars]
```
</details>

**Bash**
```
cat >> /work/scratch/code/common.py <<'EOF'

def load_masks(scale=0.5):
    f=h5py.File(f'{INP}/masks.h5')
    O=f['object'];P=f['person']
    H,W=O.shape[1:]
    h,w=int(H*scale),int(W*scale)
    Os=np.zeros((O.shape[0],h,w),bool);Ps=np.zeros_like(Os)
    for i in range(0,O.shape[0],64):
        o=torch.tensor(O[i:i+64]).float()[:,None];p=torch.tensor(P[i:i+64]).float()[:,None]
        Os[i:i+64]=(torch.nn.functional.interpolate(o,(h,w),mode='area')[:,0]>0.5).numpy()
        Ps[i:i+64]=(torch.nn.functional.interpolate(p,(h,w),mode='area')[:,0]>0.5).numpy()
    return Os,Ps

def iou_scores(R,Tc,Om,Pm,K=None,scale=0.5,bs=32):
    """Tc (F,4,4) cam poses; Om,Pm (F,h,w) bool numpy. IoU excluding person pixels"""
    out=[]
    for i in range(0,len(Tc),bs):
        rm=R.render(Tc[i:i+bs],K=K,scale=scale,tex=False)['mask']
        o=torch.tensor(Om[i:i+bs],device='cuda');p=torch.tensor(Pm[i:i+bs],device='cuda')
        valid=~p | o
        inter=(rm&o&valid).sum((1,2)).float();uni=((rm|o)&valid).sum((1,2)).float()
        out.append((inter/uni.clamp(min=1)).cpu().numpy())
    return np.concatenate(out)
EOF
python -c "
from common import *
import time
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
t=time.time();Om,Pm=load_masks();print('masks',time.time()-t)
np.save('/work/scratch/an/Om.npy',Om);np.save('/work/scratch/an/Pm.npy',Pm)
R=Renderer(load_mesh(),K)
Tb=Tcw@r['T_world_mesh'].astype(np.float64)
ib=iou_scores(R,Tb,Om,Pm); ifp=iou_scores(R,r['T_cam_mesh_foundationpose …[+302 chars]
```

<details><summary>output</summary>

```
masks 6.634398937225342
0 160 0.68062466 0.6219924 0.679878
160 215 0.6972604 0.6611628 0.71434706
215 560 0.56742173 0.5729852 0.5811381
560 719 0.6770129 0.6420095 0.67164725
[0.61 0.62 0.62 0.67 0.7  0.7  0.66 0.73 0.72 0.71 0.69 0.69 0.68 0.69 0.68 0.73 0.7  0.69 0.65 0.73 0.74 0.71 0.72 0.69 0.82 0.84 0.79 0.86 0.81 0.78 0.81 0.75 0.68 0.64 0.67 0.55 0.53 0.51 0.42 0.4  0.47 0.41 0.35 0.36 0.43 0.41 0.39 0.34 0.4
 0.41 0.47 0.53 0.53 0.58 0.52 0.48 0.67 0.62 0.63 0.67 0.65 0.69 0.63 0.67 0.65 0.67 0.7  0.65 0.72 0.72 0.67 0.72]
```
</details>

**Bash**
```
cat > /work/scratch/code/viz.py <<'EOF'
from common import *
def overlay(R,Tc,idx,out,K=None,cols=4,width=500,extra=None):
    f=h5py.File(f'{INP}/masks.h5')
    tiles=[]
    for j,i in enumerate(idx):
        o=R.render(np.asarray(Tc[j])[None],K=K)
        im=cv2.imread('/work/scratch/frames/%04d.jpg'%(i+1))
        mk=o['mask'][0].cpu().numpy(); om=f['object'][i]; pm=f['person'][i]
        ov=im.copy()
        c,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,c,-1,(0,255,0),1)
        c,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,c,-1,(0,0,255),1)
        ys,xs=np.where(om|mk); y0,y1,x0,x1=max(ys.min()-40,0),ys.max()+40,max(xs.min()-40,0),xs.max()+40
        t=ov[y0:y1,x0:x1]; s=width/t.shape[1]; t=cv2.resize(t,(width,int(t.shape[0]*s)))
        t=cv2.copyMakeBorder(t,0,max(0,width*1.3-t.shape[0]).__int__(),0,0,cv2.BORDER_CONSTANT)[:int(width*1.3)]
        cv2.putText(t,str(i),(10,30),0,1,(0,255,255),2); tiles.append(t)
    while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out,np.vstack([np.hstack(tiles[k:k+cols]) for k in range(0,len(tiles),cols)]))
EOF
python -c "
from viz import *
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
R=Renderer(load_mesh(),K)
Tb=Tcw@r['T_world_mesh'].astype(np.float64)
idx=[0,100,240,300,370,400,450,600]
overlay(R,Tb[idx],idx,'/work/scratch/an/ov_base.jpg')
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/an/ov_base.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The ladder stays still (0-160), gets dragged (160-555), then settles again, with the baseline fitting silhouettes well overall but jittering during still phases and misaligning at frames 370 and 450. Next I'll automatically pin down the exact phase boundaries.

**Bash**
```
python -c "
import numpy as np,cv2
Om=np.load('/work/scratch/an/Om.npy');Pm=np.load('/work/scratch/an/Pm.npy')
def g(i): return cv2.cvtColor(cv2.resize(cv2.imread('/work/scratch/frames/%04d.jpg'%(i+1)),(768,576)),cv2.COLOR_BGR2GRAY).astype(np.float32)
for ref in [0,718]:
  R=g(ref); reg=cv2.dilate(Om[ref].astype(np.uint8),np.ones((9,9)))>0
  out=[]
  for i in range(719):
    I=g(i); v=reg&~cv2.dilate(Pm[i].astype(np.uint8),np.ones((7,7))).astype(bool)&~cv2.dilate(Pm[ref].astype(np.uint8),np.ones((7,7))).astype(bool)
    out.append([np.abs(I-R)[v].mean() if v.sum()>50 else -1, v.sum()/reg.sum()])
  out=np.array(out); np.save('/work/scratch/an/diff_ref%d.npy'%ref,out)
  np.set_printoptions(precision=1,suppress=True,linewidth=250)
  print(ref); print(np.c_[np.arange(0,719,5),out[::5]].T)
"

```

<details><summary>output</summary>

```
0
[[  0.    5.   10.   15.   20.   25.   30.   35.   40.   45.   50.   55.   60.   65.   70.   75.   80.   85.   90.   95.  100.  105.  110.  115.  120.  125.  130.  135.  140.  145.  150.  155.  160.  165.  170.  175.  180.  185.  190.  195.  200.
  205.  210.  215.  220.  225.  230.  235.  240.  245.  250.  255.  260.  265.  270.  275.  280.  285.  290.  295.  300.  305.  310.  315.  320.  325.  330.  335.  340.  345.  350.  355.  360.  365.  370.  375.  380.  385.  390.  395.  400.  405.
  410.  415.  420.  425.  430.  435.  440.  445.  450.  455.  460.  465.  470.  475.  480.  485.  490.  495.  500.  505.  510.  515.  520.  525.  530.  535.  540.  545.  550.  555.  560.  565.  570.  575.  580.  585.  590.  595.  600.  605.  610.
  615.  620.  625.  630.  635.  640.  645.  650.  655.  6 …[+4435 chars]
```
</details>

**Bash**
```
python -c "
import numpy as np,cv2
Om=np.load('/work/scratch/an/Om.npy');Pm=np.load('/work/scratch/an/Pm.npy')
def g(i): return cv2.cvtColor(cv2.resize(cv2.imread('/work/scratch/frames/%04d.jpg'%(i+1)),(768,576)),cv2.COLOR_BGR2GRAY).astype(np.float32)
for ref,rng in [(150,range(150,216)),(718,range(540,575))]:
  R=g(ref); reg=cv2.dilate(Om[ref].astype(np.uint8),np.ones((9,9)))>0
  s=''
  for i in rng:
    I=g(i); v=reg&~cv2.dilate(Pm[i].astype(np.uint8),np.ones((7,7))).astype(bool)&~cv2.dilate(Pm[ref].astype(np.uint8),np.ones((7,7))).astype(bool)
    s+='%d:%.1f/%.2f '%(i,np.abs(I-R)[v].mean(),v.sum()/reg.sum())
  print(s)
"

```

<details><summary>output</summary>

```
150:0.0/0.95 151:0.6/0.94 152:0.9/0.91 153:1.1/0.87 154:1.1/0.83 155:1.3/0.81 156:1.3/0.80 157:1.5/0.79 158:1.5/0.78 159:1.6/0.77 160:1.8/0.76 161:1.9/0.75 162:1.9/0.75 163:2.0/0.74 164:2.0/0.73 165:1.9/0.71 166:2.0/0.70 167:2.1/0.68 168:2.1/0.67 169:2.1/0.66 170:2.1/0.65 171:2.2/0.65 172:2.2/0.63 173:2.2/0.63 174:2.1/0.61 175:2.2/0.60 176:2.3/0.60 177:2.4/0.59 178:2.4/0.59 179:2.4/0.58 180:2.4/0.58 181:2.4/0.57 182:2.4/0.57 183:2.7/0.58 184:2.5/0.56 185:2.5/0.56 186:2.5/0.56 187:2.6/0.55 188:2.7/0.56 189:2.7/0.56 190:2.6/0.55 191:2.7/0.55 192:2.7/0.55 193:2.8/0.56 194:2.8/0.56 195:2.9/0.57 196:2.9/0.56 197:3.0/0.57 198:3.0/0.56 199:3.2/0.56 200:3.8/0.56 201:5.0/0.55 202:7.0/0.55 203:9.2/0.55 204:11.4/0.56 205:13.5/0.54 206:15.1/0.54 207:16.6/0.54 208:17.7/0.54 209:18.6/0.53 210:19.4/0.53  …[+542 chars]
```
</details>

**Bash**
```
python -c "
from common import *
from scipy.spatial.transform import Rotation as Rot
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
Tb=Tcw@r['T_world_mesh'].astype(np.float64)
m=load_mesh(); c=m.vertices.mean(0); print('centroid',c)
for a,b in [(0,200),(560,719)]:
  P=Tb[a:b,:3,:3]@c+Tb[a:b,:3,3]
  print(a,b,'center cam mean',P.mean(0),'std cm',P.std(0)*100,'range',np.ptp(P,0)*100)
  rv=Rot.from_matrix(Tb[a:b,:3,:3]); mr=rv.mean(); print(' rot dev deg mean/max',np.degrees((mr.inv()*rv).magnitude()).mean(),np.degrees((mr.inv()*rv).magnitude()).max())
  # pixel position
  u=(K@P.T).T; u=u[:,:2]/u[:,2:]; print(' pix std',u.std(0))
"

```

<details><summary>output</summary>

```
centroid [0.01119669 0.00360527 0.01379611]
0 200 center cam mean [-0.30560184  0.86498964  2.82767907] std cm [0.5022492  0.82146946 2.14966895] range [2.33057766 3.77465691 9.77719609]
 rot dev deg mean/max 0.6766389488199368 2.4380045463616584
 pix std [0.92925707 0.97755786]
560 719 center cam mean [0.32360128 0.7762627  3.83702442] std cm [0.95741773 0.96991361 6.13950944] range [ 3.8592036   3.50691554 20.19962812]
 rot dev deg mean/max 1.2696581375754488 2.8999294764410775
 pix std [1.15697267 0.95762717]
```
</details>

**Bash**
```
cat > /work/scratch/code/fitlib.py <<'EOF'
from common import *
import torch.nn.functional as F
def so3_exp(w):
    th=w.norm(dim=-1,keepdim=True).clamp(min=1e-9)
    k=w/th; Kx=torch.zeros(w.shape[:-1]+(3,3),device=w.device,dtype=w.dtype)
    Kx[...,0,1]=-k[...,2];Kx[...,0,2]=k[...,1];Kx[...,1,0]=k[...,2];Kx[...,1,2]=-k[...,0];Kx[...,2,0]=-k[...,1];Kx[...,2,1]=k[...,0]
    th=th[...,None]
    I=torch.eye(3,device=w.device,dtype=w.dtype).expand_as(Kx)
    return I+torch.sin(th)*Kx+(1-torch.cos(th))*Kx@Kx

def sample_points(mesh,n,seed=0):
    pts,_=trimesh.sample.sample_surface(mesh,n,seed=seed)
    return pts

class FrameData:
    """per-frame targets at given scale: dt_fwd (distance to object|person), object pixel samples"""
    def __init__(self,frames,scale=0.5,nq=2000,seed=0,person_dil=3):
        f=h5py.File(f'{INP}/masks.h5')
        rng=np.random.RandomState(seed)
        self.frames=list(frames);self.scale=scale
        dts=[];qs=[];qn=[]
        for i in self.frames:
            o=f['object'][i];p=f['person'][i]
            if scale!=1:
                H,W=o.shape;h,w=int(H*scale),int(W*scale)
                o=cv2.resize(o.astype(np.uint8),(w,h),interpolation=cv2.INTER_AREA)>0
                p=cv2.resize(p.astype(np.uint8),(w,h),interpolation=cv2.INTER_AREA)>0
            if person_dil>0: p=cv2.dilate(p.astype(np.uint8),np.ones((2*person_dil+1,)*2))>0
            ok=(o|p)
            dt=cv2.distanceTransform((~ok).astype(np.uint8),cv2.DIST_L2,5)
            dts.ap …[+4199 chars]
```

<details><summary>output</summary>

```
Exit code 1
Traceback (most recent call last):
  File "/work/scratch/code/exp_static.py", line 23, in <module>
    print('init',evalT(T0))
                 ^^^^^^^^^
  File "/work/scratch/code/exp_static.py", line 21, in evalT
    fw,rv_=sil_loss(uv,fd,torch.arange(len(frames)))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/code/fitlib.py", line 49, in sil_loss
    d=F.grid_sample(fd.dt[idx][:,None],g,align_corners=False,padding_mode='border')[:,0,:,0]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/functional.py", line 4910, in grid_sample
    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+102 chars]
```
</details>

**Bash**
```
sed -i 's/d=F.grid_sample(fd.dt\[idx\]\[:,None\],g,/d=F.grid_sample(fd.dt[idx][:,None].to(g.dtype),g,/; s/q=fd.q\[idx\]/q=fd.q[idx].to(uv.dtype)/' fitlib.py && python exp_static.py 0 200 && python exp_static.py 560 719
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b3acw21r1). Output is being written to: /tmp/claude-1003/-work/9c970dd7-9b9a-465b-a801-f693e8bb11a4/tasks/b3acw21r1.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
```
</details>

**Bash**
```
sleep 5; cat /tmp/claude-1003/-work/9c970dd7-9b9a-465b-a801-f693e8bb11a4/tasks/b3acw21r1.output; nvidia-smi --query-gpu=memory.used --format=csv
```

<details><summary>output</summary>

```
init (0.07701785058854459, 0.6729408848462753)
baseline per-frame 0.07504054066951646 0.66924349708471
0 0.07701785058854459 0.6729408848462753 [[-0.003 -0.003  0.003]] [[-0.3  0.3 -0.3]]
50 0.009749932791572446 0.5080411776944325 [[-0.00265131  0.00157006  0.00057854]] [[ 2.44435855e-03  3.13681048e-01 -3.43041222e+00]]
100 0.010139824063473576 0.5035615022053022 [[-0.00248172  0.00399957  0.00096206]] [[ 0.16317316 -0.16417109 -4.81597765]]
150 0.010372212955642127 0.5032348611718691 [[-0.00183663  0.00458441  0.00048362]] [[ 0.18253231 -0.24251061 -5.03581411]]
200 0.012056944344413597 0.5005726091266607 [[ 0.00241078  0.00581704 -0.00157033]] [[ 0.24996649 -0.44506455 -5.74307715]]
250 0.011954170889254132 0.5006807903496466 [[ 0.00215903  0.00595553 -0.00149804]] [[ 0.2385788  -0.4464 …[+815 chars]
```
</details>

**Bash**
```
pkill -f exp_static.py; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv
```

<details><summary>output</summary>

```
Exit code 144
```
</details>

**Bash**
```
python - <<'EOF'
s=open('/work/scratch/code/fitlib.py').read()
old=s[s.index('def sil_loss'):s.index('class PoseParam')]
new='''def sil_loss(uv,fd,idx,clamp=25.0,rev_idx=None):
    """uv (B,N,2) pixel coords at fd scale; idx indices into fd arrays. returns per-frame fwd, rev (pixels)"""
    uv=uv.float()
    g=torch.stack([uv[...,0]/fd.W*2-1,uv[...,1]/fd.H*2-1],-1)[:,:,None]
    d=F.grid_sample(fd.dt[idx][:,None],g,align_corners=False,padding_mode='border')[:,0,:,0]
    fwd=torch.clamp(d,max=clamp).mean(1)
    q=fd.q[idx]
    up=uv if rev_idx is None else uv[:,rev_idx]
    dd=torch.cdist(q,up).min(2).values
    rev=torch.clamp(dd,max=clamp).mean(1)
    return fwd,rev

'''
s=s.replace(old,new)
s=s.replace("def __init__(self,frames,scale=0.5,nq=2000","def __init__(self,frames,scale=0.5,nq=1000")
open('/work/scratch/code/fitlib.py','w').write(s)
s=open('/work/scratch/code/exp_static.py').read()
s=s.replace("fw,rv_=sil_loss(uv,fd,torch.arange(len(frames)))\n    loss","fw,rv_=sil_loss(uv,fd,torch.arange(len(frames)),rev_idx=torch.randperm(5000,device='cuda')[:2000])\n    loss")
s=s.replace("print('final',evalT(T1),time.time()-t0)","print('final',evalT(T1),time.time()-t0,torch.cuda.max_memory_allocated()/1e9)")
open('/work/scratch/code/exp_static.py','w').write(s)
EOF
grep -n rev_idx exp_static.py; timeout 600 python exp_static.py 0 200
```

<details><summary>output</summary>

```
35:    fw,rv_=sil_loss(uv,fd,torch.arange(len(frames)),rev_idx=torch.randperm(5000,device='cuda')[:2000])
init (0.07701780647039413, 0.6642696261405945)
baseline per-frame 0.0750405415892601 0.6619572639465332
0 0.07701780647039413 0.8869488835334778 [[0.003 0.003 0.003]] [[-0.3  0.3 -0.3]]
50 0.01290174201130867 0.7282233834266663 [[ 0.0058075   0.00220425 -0.00212711]] [[ 0.01708047  0.21581668 -3.63258462]]
100 0.013330355286598206 0.7260705828666687 [[0.0096758  0.00461118 0.00199907]] [[ 0.2604577  -0.45702152 -5.66215101]]
150 0.014297432266175747 0.7344256043434143 [[ 0.00977479  0.01104023 -0.00143752]] [[ 0.34453441 -0.61113476 -6.41183605]]
200 0.014485294930636883 0.7253061532974243 [[ 0.00885353  0.00935924 -0.00094386]] [[ 0.35518493 -0.70207105 -6.44179338]]
250 0.01366658322 …[+274 chars]
```
</details>

**Bash**
```
cat >> /work/scratch/code/fitlib.py <<'EOF'

def so3_log(R):
    tr=(R[...,0,0]+R[...,1,1]+R[...,2,2]).clamp(-1+1e-7,3-1e-7)
    th=torch.acos(((tr-1)/2).clamp(-1+1e-7,1-1e-7))
    v=torch.stack([R[...,2,1]-R[...,1,2],R[...,0,2]-R[...,2,0],R[...,1,0]-R[...,0,1]],-1)
    s=torch.where(th[...,None]<1e-4,0.5+th[...,None]**2/12,th[...,None]/(2*torch.sin(th[...,None]).clamp(min=1e-9)))
    return v*s

def smooth_terms(R,t,c):
    """R (B,3,3), t (B,3) -> per-step rot second difference (rad), center second difference (m)"""
    rel=so3_log(R[:-1].transpose(1,2)@R[1:])  # body-frame angular velocity
    # express in camera frame for comparability
    relc=torch.einsum('bij,bj->bi',R[:-1],rel)
    acc_r=relc[1:]-relc[:-1]
    p=torch.einsum('bij,j->bi',R,c)+t
    acc_t=p[2:]-2*p[1:-1]+p[:-2]
    return acc_r,acc_t,relc,p
EOF
cat > /work/scratch/code/exp_moving.py <<'EOF'
from fitlib import *
import sys,time
torch.manual_seed(0)
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
Tb=Tcw@r['T_world_mesh'].astype(np.float64)
mesh=load_mesh(); c=mesh.vertices.mean(0)
NP=4000
pts=torch.tensor(sample_points(mesh,NP),device='cuda')
a,b=int(sys.argv[1]),int(sys.argv[2]); frames=np.arange(a,b)
sr,st=float(sys.argv[3]),float(sys.argv[4]); iters=int(sys.argv[5]); tag=sys.argv[6]
init=sys.argv[7] if len(sys.argv)>7 else None
scale=0.5
fd=FrameData(frames,scale=scale,nq=800)
Ks=torch.tensor(K*np.array([[scale],[scale],[1]]),device='cuda')
T0=Tb[frames] if init is None else np.load(init)[frame …[+1353 chars]
```

<details><summary>output</summary>

```
init data 0.141 1.048 smooth r 0.0241 t 0.0130
0 0.141 1.339 0.0241 0.0130
100 0.039 1.097 0.0024 0.0007
200 0.034 1.093 0.0021 0.0006
300 0.031 1.074 0.0021 0.0006
final data 0.028 0.759 smooth r 0.0020 t 0.0006 22.708529472351074 6.969235968
```
</details>

**Bash**
```
cat > /work/scratch/code/cmp.py <<'EOF'
from viz import *
import sys
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
Tb=Tcw@r['T_world_mesh'].astype(np.float64)
R=Renderer(load_mesh(),K)
Om=np.load('/work/scratch/an/Om.npy');Pm=np.load('/work/scratch/an/Pm.npy')
T=np.load(sys.argv[1]); a,b=int(sys.argv[2]),int(sys.argv[3])
ib=np.load('/work/scratch/an/iou_base.npy')
i1=iou_scores(R,T[a:b],Om[a:b],Pm[a:b])
np.set_printoptions(precision=2,suppress=True,linewidth=250)
print('mean iou base %.3f new %.3f'%(ib[a:b].mean(),i1.mean()))
for s in range(a,b,40): print(s, 'base %.3f new %.3f'%(ib[s:s+40].mean(),i1[s-a:s-a+40].mean()))
if len(sys.argv)>4:
    idx=[int(x) for x in sys.argv[4].split(',')]
    overlay(R,T[idx],idx,sys.argv[5])
EOF
python cmp.py /work/scratch/an/mov_a.npy 200 560 250,300,340,370,400,430,450,480 /work/scratch/an/ov_mov_a.jpg
```

<details><summary>output</summary>

```
mean iou base 0.573 new 0.689
200 base 0.718 new 0.787
240 base 0.820 new 0.862
280 base 0.787 new 0.826
320 base 0.612 new 0.714
360 base 0.430 new 0.651
400 base 0.377 new 0.591
440 base 0.384 new 0.555
480 base 0.478 new 0.609
520 base 0.553 new 0.609
```
</details>

**Read**
```
/work/scratch/an/ov_mov_a.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import cv2,numpy as np
t=[]
for i in [330,380,420,440,455,470,500,540]:
  im=cv2.imread('/work/scratch/frames/%04d.jpg'%(i+1)); bb=np.load('/work/scratch/an/bb.npy')[i]
  x0,y0,x1,y1=bb; c=im[max(y0-60,0):y1+60,max(x0-80,0):x1+80]; c=cv2.resize(c,(480,int(480*c.shape[0]/c.shape[1])))[:480]
  c=cv2.copyMakeBorder(c,0,480-c.shape[0],0,0,0); cv2.putText(c,str(i),(10,30),0,1,(0,255,255),2); t.append(c)
cv2.imwrite('/work/scratch/an/raw.jpg',np.vstack([np.hstack(t[:4]),np.hstack(t[4:])]))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/an/raw.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat >> /work/scratch/code/fitlib.py <<'EOF'

def reanchor(Tc,Tb,frames,c):
    """scale camera-frame object-centre positions of Tc[frames] so median z equals baseline's"""
    Tc=Tc.copy()
    pc=np.einsum('bij,j->bi',Tc[frames,:3,:3],c)+Tc[frames,:3,3]
    pb=np.einsum('bij,j->bi',Tb[frames,:3,:3],c)+Tb[frames,:3,3]
    s=np.median(pb[:,2])/np.median(pc[:,2])
    Tc[frames,:3,3]+=(s-1)*pc
    return Tc,s

def floor_height(Tw,V,plane):
    n,d=np.asarray(plane[:3]),plane[3]
    out=[]
    for T in Tw:
        W=V@T[:3,:3].T+T[:3,3]; s=W@n+d; out.append([s.min(),np.percentile(s,0.5)])
    return np.array(out)
EOF
python -c "
from fitlib import *
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
Tb=Tcw@r['T_world_mesh'].astype(np.float64)
m=load_mesh();c=m.vertices.mean(0); V=m.vertices[::20]
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']
T=np.load('/work/scratch/an/mov_a.npy'); fr=np.arange(200,560)
Ta,s=reanchor(T,Tb,fr,c); print('scale',s)
hb=floor_height(Twc@Tb[fr],V,pl); ha=floor_height(Twc@Ta[fr],V,pl)
pc=np.einsum('bij,j->bi',Ta[fr,:3,:3],c)+Ta[fr,:3,3]; pb=np.einsum('bij,j->bi',Tb[fr,:3,:3],c)+Tb[fr,:3,3]
np.set_printoptions(precision=1,suppress=True,linewidth=250)
print('base minh cm',hb[::8,0]*100); print('fit minh cm',ha[::8,0]*100)
print('dz cm', (pc[::8,2]-pb[::8,2])*100)
"

```

<details><summary>output</summary>

```
scale 1.009212836249806
base minh cm [-2.1 -1.6 -1.7 -0.8 -0.8 -1.2 -1.8 -2.2 -1.7 -1.5 -0.6 -2.9 -1.9 -2.9 -2.5 -1.7  0.8 -0.5 -1.1 -2.5 -2.6 -2.1 -2.6 -3.4 -3.5 -3.5 -3.9 -2.4 -2.3 -1.6 -2.1 -1.7 -1.1 -3.2 -3.2 -2.1 -2.1 -2.2 -2.3 -2.1 -0.7 -1.1 -1.8 -1.4 -0.5]
fit minh cm [-3.3 -3.3 -2.7 -1.  -1.2 -1.9 -1.7 -1.8 -1.5 -1.3 -1.7 -2.9 -3.3 -3.5 -3.6 -3.4 -2.2 -1.1 -1.8 -2.9 -2.9 -2.4 -2.  -2.5 -2.8 -3.1 -3.9 -3.1 -2.3 -2.  -2.2 -2.4 -2.5 -3.9 -3.4 -2.5 -2.7 -3.3 -3.2 -2.4 -1.8 -1.8 -2.3 -2.8 -2.6]
dz cm [ 1.8  2.1  1.2  1.   1.5  1.8  0.9  1.4  1.9  0.9  1.7  1.   2.   1.8  2.1  1.2  1.7 -1.4  0.5 -0.3 -0.6 -0.2 -1.1 -0.5 -0.  -0.9 -0.8  1.5 -1.1  0.2 -1.3 -0.6  2.1  1.5 -0.6  1.3  1.8  2.5  1.5  0.9  2.9  2.9  1.3  2.6  3.3]
```
</details>

**Bash**
```
cat > /work/scratch/code/exp_flips.py <<'EOF'
from fitlib import *
import time
torch.manual_seed(0)
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
mesh=load_mesh(); c=mesh.vertices.mean(0)
NP=3000; pts=torch.tensor(sample_points(mesh,NP),device='cuda')
T=np.load('/work/scratch/an/mov_a.npy')
frames=np.arange(200,560,4)
scale=0.5; fd=FrameData(frames,scale=scale,nq=600); Ks=torch.tensor(K*np.array([[scale],[scale],[1]]),device='cuda')
S,_=trimesh.bounds.oriented_bounds(mesh)  # principal axes
res={}
for name,ax in [('none',None),('x',0),('y',1),('z',2)]:
    T0=T[frames].copy()
    if ax is not None:
        # rotate 180 about mesh principal axis 'ax' through centroid: use mesh axes (x,y,z of mesh file)
        Rf=np.diag([1.,1.,1.]); Rf[[i for i in range(3) if i!=ax],[i for i in range(3) if i!=ax]]=-1
        F4=np.eye(4);F4[:3,:3]=Rf;F4[:3,3]=c-Rf@c
        T0=T0@F4
    P=PoseParam(T0,c); opt=torch.optim.Adam([{'params':[P.w],'lr':5e-3},{'params':[P.dt],'lr':3e-3}])
    idx=torch.arange(len(frames),device='cuda')
    for it in range(300):
        R,t=P.RT(); X=torch.einsum('bij,nj->bni',R,pts)+t[:,None]
        fw,rv=sil_loss(project(Ks,X),fd,idx,rev_idx=torch.randperm(NP,device='cuda')[:1000])
        loss=(fw+rv).sum(); opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        R,t=P.RT(); X=torch.einsum('bij,nj->bni',R,pts)+t[:,None]; fw,rv=sil_loss(project(Ks,X),fd,idx)
    res[name]=(fw+rv).cpu().numpy()
    print(name,res[name].mean())
L=np.s …[+231 chars]
```

<details><summary>output</summary>

```
none 0.7503614
x 3.4425564
y 3.4716003
z 1.480867
[[200.     0.58   1.75   1.92   1.26]
 [212.     0.55   1.84   2.05   1.39]
 [224.     0.63   1.01   2.02   2.1 ]
 [236.     0.64   3.48   4.2    2.05]
 [248.     0.67   3.35   3.39   1.39]
 [260.     0.67   3.42   3.66   1.4 ]
 [272.     0.69   3.19   3.23   1.51]
 [284.     0.7    3.53   3.7    1.22]
 [296.     0.77   4.06   4.28   1.2 ]
 [308.     0.72   4.47   4.09   1.24]
 [320.     0.82   5.52   4.43   2.25]
 [332.     0.87   7.37   4.4    2.41]
 [344.     0.91   4.32   4.21   2.21]
 [356.     0.85   3.97   3.5    1.64]
 [368.     0.85   2.62   3.17   1.51]
 [380.     0.85   3.07   3.14   1.32]
 [392.     0.8    2.32   2.11   0.87]
 [404.     0.78   2.71   2.97   0.9 ]
 [416.     0.97   2.96   3.36   1.16]
 [428.     0.78   3.05   3.1 …[+425 chars]
```
</details>

**Bash**
```
cat > /work/scratch/code/exp_static_flips.py <<'EOF'
from fitlib import *
torch.manual_seed(0)
r=load_rec();K,Twc,Kp=load_cam();Tcw=np.linalg.inv(Twc)
Tb=Tcw@r['T_world_mesh'].astype(np.float64)
mesh=load_mesh(); c=mesh.vertices.mean(0)
NP=3000; pts=torch.tensor(sample_points(mesh,NP),device='cuda')
from scipy.spatial.transform import Rotation as Rot
for a,b in [(0,200),(561,719)]:
    frames=np.arange(a,b,5); fd=FrameData(frames,scale=0.5,nq=600); Ks=torch.tensor(K*np.array([[.5],[.5],[1]]),device='cuda')
    T0=np.eye(4);T0[:3,:3]=Rot.from_matrix(Tb[a:b,:3,:3]).mean().as_matrix();T0[:3,3]=np.median(Tb[a:b,:3,3],0)
    for name,ax in [('none',None),('x',0),('y',1),('z',2)]:
        Ti=T0.copy()
        if ax is not None:
            Rf=-np.eye(3); Rf[ax,ax]=1; F4=np.eye(4);F4[:3,:3]=Rf;F4[:3,3]=c-Rf@c; Ti=Ti@F4
        P=PoseParam(Ti[None],c); opt=torch.optim.Adam([{'params':[P.w],'lr':5e-3},{'params':[P.dt],'lr':3e-3}])
        idx=torch.arange(len(frames),device='cuda')
        for it in range(300):
            R,t=P.RT(); X=(torch.einsum('bij,nj->bni',R,pts)+t[:,None]).expand(len(frames),-1,-1)
            fw,rv=sil_loss(project(Ks,X),fd,idx); loss=(fw+rv).mean(); opt.zero_grad();loss.backward();opt.step()
        print(a,b,name,'%.3f'%loss.item())
EOF
python exp_static_flips.py
```

<details><summary>output</summary>

```
0 200 none 0.619
0 200 x 4.219
0 200 y 2.742
0 200 z 1.609
561 719 none 0.514
561 719 x 2.079
561 719 y 2.642
561 719 z 0.795
```
</details>

> *thinking:* I've identified three motion phases (still, dragged, still), confirmed the baseline orientation beats all flip alternatives, and improved dragged-phase mask IoU from 0.57 to 0.69 via silhouette refinement—though this pulled depth 6–12cm too close, which I'll correct afterward. Now writing the joint solver.

**Bash**
```
cat > /work/outputs/solve.py <<'EOF'
#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF object pose for one HOI video.

Method (see REPORT.md):
 1. phase segmentation (static / moving) from image differences inside the object region,
 2. per-frame silhouette evidence: robust 2-D chamfer between projected mesh surface points and
    the SAM2 object mask (points that fall on the person are not penalised = occlusion aware),
 3. joint optimisation over the whole episode: ONE pose per static phase (evidence pooled over all
    its frames), one pose per moving frame, SE(3) second-difference smoothness, initialised from
    the baseline (180-deg flip hypotheses were tested offline and always lost by a clear margin; the
    same test is re-run here as a sanity check and logged),
 4. re-anchoring: per phase, camera-frame positions are scaled along the viewing rays so that the
    median camera depth equals the baseline's (silhouette-only depth is biased, see REPORT.md),
    with a smooth blend of the scale across the moving phase so the trajectory stays continuous.
Deterministic: fixed seeds, fixed point samples, fixed iteration counts.
"""
import argparse, json, os, time
import numpy as np, h5py, cv2, trimesh, torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------------- utils
def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = w /  …[+11598 chars]
```

<details><summary>output</summary>

```
phases: static [0,76)  moving [76,564)  static [564,719)
it 0 data 2.335 smooth 2107.2  13s
it 100 data 1.854 smooth 14.3  21s
it 200 data 1.843 smooth 12.2  29s
it 300 data 1.803 smooth 11.8  36s
it 400 data 1.760 smooth 10.9  44s
it 500 data 1.806 smooth 10.8  52s
it 599 data 1.787 smooth 10.6  60s
re-anchor scales {'S1': 1.0124031255915373, 'M': 1.005951305665394, 'S2': 0.9971325294501536}
done (719, 4, 4) 60s
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from solve import detect_phases
import h5py,numpy as np
h=h5py.File('/work/inputs/masks.h5'); o=h['object'][:];p=h['person'][:]
a,b,d=detect_phases('/work/inputs/video.mp4',o,p,719)
np.save('an/dmot.npy',d)
np.set_printoptions(precision=1,suppress=True,linewidth=250)
print(a,b); print(np.c_[np.arange(0,719,1),d][60:220].T); print(d[540:].round(1))
"
```

<details><summary>output</summary>

```
76 564
[[ 60.   61.   62.   63.   64.   65.   66.   67.   68.   69.   70.   71.   72.   73.   74.   75.   76.   77.   78.   79.   80.   81.   82.   83.   84.   85.   86.   87.   88.   89.   90.   91.   92.   93.   94.   95.   96.   97.   98.   99.  100.
  101.  102.  103.  104.  105.  106.  107.  108.  109.  110.  111.  112.  113.  114.  115.  116.  117.  118.  119.  120.  121.  122.  123.  124.  125.  126.  127.  128.  129.  130.  131.  132.  133.  134.  135.  136.  137.  138.  139.  140.  141.
  142.  143.  144.  145.  146.  147.  148.  149.  150.  151.  152.  153.  154.  155.  156.  157.  158.  159.  160.  161.  162.  163.  164.  165.  166.  167.  168.  169.  170.  171.  172.  173.  174.  175.  176.  177.  178.  179.  180.  181.  182.
  183.  184.  185.  186.  187.  188.  189.  190.  19 …[+2041 chars]
```
</details>

**Bash**
```
cd /work/outputs && python - <<'EOF'
s=open('solve.py').read()
s=s.replace("def detect_phases(video, obj, per, F_, lag=5, thr=3.0, min_static=30):","def detect_phases(video, obj, per, F_, lag=5, thr=4.0, min_run=5, min_static=30):")
s=s.replace("""    mov = np.nan_to_num(d, nan=0.0) > thr
""","""    mov = np.nan_to_num(d, nan=0.0) > thr
    # keep only sustained motion (>= min_run consecutive frames): brief blips come from the
    # person's legs / shadow passing next to the object (e.g. frames 80-82 here, d~3.3)
    lab = np.zeros(F_, bool); i = 0
    while i < F_:
        if mov[i]:
            j = i
            while j < F_ and mov[j]: j += 1
            if j - i >= min_run: lab[i:j] = True
            i = j
        else: i += 1
    mov = lab
""")
s=s.replace("Static phases are the leading / trailing runs with d below `thr` grey levels (sensor noise +\n    slow exposure drift are ~1; a moving ladder gives >10).","Static phases are the leading / trailing runs with d below `thr` grey levels (sensor noise +\n    slow exposure drift are ~1-1.5; a moving ladder gives 5-20).")
open('solve.py','w').write(s)
EOF
cd /work/scratch && python /work/outputs/solve.py --out /work/scratch/an/sol_v1.npz 2>&1 | tail -12
```

<details><summary>output</summary>

```
phases: static [0,198)  moving [198,564)  static [564,719)
it 0 data 2.346 smooth 1899.1  18s
it 100 data 1.857 smooth 15.0  25s
it 200 data 1.843 smooth 12.5  32s
it 300 data 1.802 smooth 11.6  39s
it 400 data 1.760 smooth 10.8  47s
it 500 data 1.806 smooth 10.8  54s
it 599 data 1.789 smooth 10.6  61s
re-anchor scales {'S1': 1.021535777978744, 'M': 1.0128748547801008, 'S2': 0.997148809982114}
done (719, 4, 4) 61s
```
</details>

**Bash**
```
cat > /work/outputs/validate.py <<'EOF'
#!/usr/bin/env python
"""No-GT validation of a trajectory vs the baseline: per phase mask IoU (person pixels excluded),
silhouette chamfer residuals, temporal smoothness, static-phase jitter, floor height, hand-object
consistency, and camera-depth difference to the baseline.
usage: python validate.py --inputs /work/inputs --pred X.npz [--pred2 Y.npz] [--phases a,b]"""
import argparse, json, numpy as np, h5py, cv2, trimesh, torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot

def render_masks(glctx, V, Fc, T, K, H, W):
    v = torch.tensor(V, dtype=torch.float32, device='cuda'); f = torch.tensor(Fc, dtype=torch.int32, device='cuda')
    n, fr = 0.01, 20.
    P = torch.tensor([[2 * K[0, 0] / W, 0, (2 * K[0, 2] - W) / W, 0], [0, 2 * K[1, 1] / H, (2 * K[1, 2] - H) / H, 0],
                      [0, 0, (fr + n) / (fr - n), -2 * fr * n / (fr - n)], [0, 0, 1, 0]], dtype=torch.float32, device='cuda')
    T = torch.tensor(T, dtype=torch.float32, device='cuda')
    vc = torch.einsum('bij,nj->bni', T[:, :3, :3], v) + T[:, None, :3, 3]
    clip = torch.einsum('ij,bnj->bni', P, torch.cat([vc, torch.ones_like(vc[..., :1])], -1))
    rast, _ = dr.rasterize(glctx, clip.contiguous(), f, (H, W))
    return rast[..., 3] > 0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--pred', required=True)
    ap.add_argument('--phases', default=None); ap.add_ar …[+3959 chars]
```

<details><summary>output</summary>

```
| phase | n | iou_base | iou_pred | iou_p10_base | iou_p10_pred | racc_base | racc_pred | tacc_base | tacc_pred | jitter_base | jitter_pred | hmin_base | hmin_pred | dz_mean | dz_median | dpos_mean | drot_mean | drot_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 static | 198 | 0.675 | 0.745 | 0.625 | 0.724 | 0.594 | 0.000 | 1.272 | 0.011 | 1.976 | 0.000 | 0.802 | -0.659 | 0.175 | -0.000 | 2.406 | 0.961 | 2.714 |
| M moving | 366 | 0.573 | 0.655 | 0.371 | 0.529 | 1.368 | 0.130 | 1.300 | 0.091 | 68.295 | 68.241 | -1.896 | -2.260 | 0.808 | 1.150 | 2.558 | 1.883 | 4.615 |
| S2 static | 155 | 0.668 | 0.669 | 0.630 | 0.641 | 1.118 | 0.000 | 2.174 | 0.043 | 5.387 | 0.000 | -0.042 | -0.531 | 1.809 | -0.000 | 5.508 | 2.141 | 4.190 |
| all | 719 | 0.621 | 0 …[+130 chars]
```
</details>

**Agent:** You've hit your session limit · resets 2:50pm (UTC)

## Final

You've hit your session limit · resets 2:50pm (UTC)
