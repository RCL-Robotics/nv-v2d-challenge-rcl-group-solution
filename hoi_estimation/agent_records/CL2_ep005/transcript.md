# CL2_ep005

Claude Opus 完整轨迹 v2（tier3_opus 发布用，ep0-10）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T06:08:23Z

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
cd /work/inputs; ls -la; cat task.txt episode.json camera.json ground_plane.json; ls /work/logs /work/.net /work/scratch; ls /pipeline; nvidia-smi; cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null
```

<details><summary>output</summary>

```
Exit code 1
total 107168
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 10:02 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 10:02 ..
-rw-rw-r-- 1 hanzhou hanzhou      827 Oct 21 10:02 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       68 Oct 21 10:02 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   192935 Oct 21 10:02 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      399 Oct 21 10:02 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  8166311 Oct 21 10:02 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 19197156 Oct 21 10:02 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 50516207 Oct 21 10:02 reconstruction.npz
-rw-rw-r-- 1 hanzhou hanzhou       69 Oct 21 10:02 task.txt
-rw-rw-r-- 1 hanzhou hanzhou 31632938 Oct 21 10:02 video.mp4
Walk over to the G1 Box and rotate the box around the bounding area.
{
 "epi …[+3552 chars]
```
</details>

**Bash**
```
ls -la /pipeline /code /weights /workspace 2>&1 | head -50; ls -R /work/scratch | head -50; cat /work/scratch/pipeline_report.json | head -50; echo $CUDA_VISIBLE_DEVICES
```

<details><summary>output</summary>

```
/code:
total 100
drwxrwxr-x 11 hanzhou hanzhou  4096 Oct  4 16:57 .
drwxr-xr-x  1 hanzhou hanzhou   200 Oct 21 10:08 ..
drwxrwxr-x  3 hanzhou hanzhou  4096 Oct  4 16:57 .claude
drwxrwxr-x  3 hanzhou hanzhou  4096 Oct  4 16:57 .codex
drwxrwxr-x  9 hanzhou hanzhou  4096 Oct  4 23:34 .git
-rw-rw-r--  1 hanzhou hanzhou  1182 Oct  4 16:57 .gitattributes
drwxrwxr-x  4 hanzhou hanzhou  4096 Oct  4 16:57 .github
-rw-rw-r--  1 hanzhou hanzhou   434 Oct  4 16:57 .gitignore
-rw-rw-r--  1 hanzhou hanzhou 30209 Oct  4 16:57 LICENSE
-rw-rw-r--  1 hanzhou hanzhou 11425 Oct  4 16:57 README.md
-rw-rw-r--  1 hanzhou hanzhou  1917 Oct  4 16:57 SECURITY.md
drwxrwxr-x  9 hanzhou hanzhou  4096 Oct  4 16:57 docs
drwxrwxr-x  9 hanzhou hanzhou  4096 Oct  4 17:06 reconstruction
drwxrwxr-x 10 hanzhou hanzhou  4096 O …[+1589 chars]
```
</details>

**Bash**
```
cat /pipeline/pipeline_report.json; find /pipeline -maxdepth 3 | head -60; du -sh /pipeline/*
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
    "mtime_ns": 1791285789451493065,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000005/inference/episode_000005_step200000_before_after.mp4",
    "size": 79817739
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+6824 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import trimesh, numpy as np
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m, m.bounds, m.extents, len(m.vertices), len(m.faces))
print(type(m.visual), getattr(m.visual,'material',None))
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
r=np.load('inputs/reconstruction.npz'); 
for k in r.files: print(k, r[k].shape, r[k].dtype)
import h5py; f=h5py.File('inputs/masks.h5'); print({k:f[k].shape for k in f})
"; python -c "import cv2; c=cv2.VideoCapture('/work/inputs/video.mp4'); print(c.get(5), c.get(7))"; ls /pipeline/export/episode_000005/images | head -3; ls /pipeline/export/episode_000005/images | wc -l
```

<details><summary>output</summary>

```
<trimesh.Trimesh(vertices.shape=(289270, 3), faces.shape=(96454, 3))> [[-0.25669852 -0.20899488 -0.3920801 ]
 [ 0.26427466  0.18703176  0.46906903]] [0.52097318 0.39602664 0.86114913] 289270 96454
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x79fae592b490>
[[-0.01327811  0.99987957 -0.00803306  0.01364348]
 [-0.99989845 -0.01331908 -0.00506915  0.00247045]
 [-0.00517554  0.00796493  0.99995489 -0.03821753]
 [ 0.          0.          0.          1.        ]] [0.39077532 0.52324755 0.86084622]
T_world_mesh (1003, 4, 4) float32
T_cam_mesh_foundationpose (1003, 4, 4) float32
T_cam_mesh_coconet (1003, 4, 4) float32
human_vertices_world (1003, 4610, 3) float32
human_keypoints_world (1003, 70, 3) float32
contact_prob_right_left (1003, 2) float32
 …[+102 chars]
```
</details>

**Agent:** Starting exploration: the object is a big black box (52×40×86 cm), 1003 frames. Let me look at the video and baseline.

**Bash**
```
mkdir -p /work/scratch/frames /work/outputs /work/tools; cd /work/scratch; python -c "
import cv2
c=cv2.VideoCapture('/work/inputs/video.mp4'); i=0
while True:
    ok,f=c.read()
    if not ok: break
    if i%50==0: cv2.imwrite(f'frames/{i:04d}.jpg', cv2.resize(f,(768,576)))
    i+=1
print(i)
"; python -c "
import cv2,numpy as np,glob
fs=sorted(glob.glob('frames/*.jpg'))
ims=[cv2.putText(cv2.resize(cv2.imread(f),(384,288)),f[-8:-4],(10,30),0,1,(0,255,0),2) for f in fs]
while len(ims)%5: ims.append(np.zeros_like(ims[0]))
rows=[np.hstack(ims[i:i+5]) for i in range(0,len(ims),5)]
cv2.imwrite('contact.jpg',np.vstack(rows))
"
```

<details><summary>output</summary>

```
1003
```
</details>

**Read**
```
/work/scratch/contact.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import nvdiffrast.torch as dr, torch; print(torch.__version__, torch.cuda.is_available())
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual; print(v.uv.shape, v.material.baseColorTexture.size if v.material.baseColorTexture else None, v.material.baseColorFactor)
"; pip list 2>/dev/null | grep -i -E "torch|kornia|opencv|open3d|trimesh|scipy|pytorch3d|roma|lietorch|h5py|cotracker|raft"
```

<details><summary>output</summary>

```
2.5.1+cu124 True
(289270, 2) (5007, 4408) None
h5py                                     3.16.0
kornia                                   0.8.3
kornia_rs                                0.2.0
open3d                                   0.20.0
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
trimesh            …[+28 chars]
```
</details>

**Write**
```
/work/outputs/hoi_lib.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/hoi_lib.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
a=s.index('        n, fr = 0.05, 50.0'); b=s.index('        rast, _ = dr.rasterize')
new='''        n, fr = 0.05, 50.0
        # OpenCV camera -> clip; nvdiffrast row 0 is ndc y=-1, so map pixel row v to ndc y = 2(v+0.5)/h-1
        P = torch.tensor([[2 * fx / w, 0, 2 * (cx + 0.5) / w - 1, 0],
                          [0, 2 * fy / h, 2 * (cy + 0.5) / h - 1, 0],
                          [0, 0, (fr + n) / (fr - n), -2 * fr * n / (fr - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)
        clip = torch.einsum("ij,bnj->bni", P, vc)
'''
s=s[:a]+new+s[b:]
open('hoi_lib.py','w').write(s)
EOF
grep -n "scale - 0.5" hoi_lib.py
```

<details><summary>output</summary>

```
43:        fx, fy, cx, cy = K[0, 0] * scale, K[1, 1] * scale, (K[0, 2] + 0.5) * scale - 0.5, (K[1, 2] + 0.5) * scale - 0.5
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import cv2, numpy as np, h5py
c=cv2.VideoCapture('/work/inputs/video.mp4'); fr=[]
while True:
    ok,f=c.read()
    if not ok: break
    fr.append(cv2.resize(f,(768,576),interpolation=cv2.INTER_AREA)[...,::-1])
np.save('frames_half.npy',np.stack(fr))
f=h5py.File('/work/inputs/masks.h5')
om=f['object'][:]; pm=f['person'][:]
np.save('objmask_half.npy', om[:,::2,::2]); np.save('permask_half.npy', pm[:,::2,::2])
area=om.reshape(len(om),-1).sum(1); print('obj area min/med/max',area.min(),np.median(area),area.max(), (area==0).sum())
EOF
```

<details><summary>output</summary>

```
obj area min/med/max 22758 29210.0 58822 0
```
</details>

**Bash**
```
cd /work/scratch; cat > /work/scratch/overlay.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
frames=np.load('frames_half.npy', mmap_mode='r'); om=np.load('objmask_half.npy', mmap_mode='r')
def overlay(T_wm, idx, out, label=''):
    Tc=cam_from_world(d['T_wc'], T_wm[idx])
    ims=[]
    for k,i in enumerate(idx):
        r=R.render(Tc[k:k+1], d['K'], d['H'], d['W'], scale=0.5)
        rgb=r['rgb'][0].cpu().numpy(); m=r['mask'][0].cpu().numpy()
        im=frames[i].astype(np.float32)/255
        im2=im.copy(); im2[m]=0.4*im[m]+0.6*np.clip(rgb[m]*2.5,0,1)
        cnt,_=cv2.findContours(om[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        im2=(im2*255).astype(np.uint8).copy(); cv2.drawContours(im2,cnt,-1,(0,255,0),1)
        ys,xs=np.where(om[i]|m); 
        y0,y1,x0,x1=max(ys.min()-30,0),ys.max()+30,max(xs.min()-30,0),xs.max()+30
        crop=cv2.resize(im2[y0:y1,x0:x1],(256,256))
        cv2.putText(crop,f'{label}{i}',(5,20),0,0.6,(255,255,0),2)
        ims.append(crop)
    while len(ims)%6: ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(out, np.vstack([np.hstack(ims[j:j+6]) for j in range(0,len(ims),6)])[...,::-1])
if __name__=='__main__':
    T=d['rec']['T_world_mesh'].astype(np.float64)
    idx=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else list(range(0,1003,42))
    overlay(T, idx, sys.argv[1] …[+35 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/base_ov.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
K=np.array([[800,0,256],[0,800,256],[0,0,1.]])
views={'+x':[0,-90,0],'-x':[0,90,0],'+y':[90,0,0],'-y':[-90,0,0],'+z':[0,180,0],'-z':[0,0,0]}
ims=[]
for name,e in views.items():
    T=np.eye(4); T[:3,:3]=Rot.from_euler('xyz',e,degrees=True).as_matrix(); T[:3,3]=[0,0,2.2]
    # camera looks along +z; view from axis: rotate so that axis points toward camera (-z)
    r=R.render(T[None],K,512,512)
    im=(np.clip(r['rgb'][0].cpu().numpy()*3,0,1)*255).astype(np.uint8).copy()
    # draw mesh axes
    for ax,col in zip(np.eye(3),[(255,0,0),(0,255,0),(0,0,255)]):
        p0=K@(T[:3,3]); p1=K@(T[:3,:3]@(ax*0.3)+T[:3,3])
        cv2.arrowedLine(im,tuple((p0[:2]/p0[2]).astype(int)),tuple((p1[:2]/p1[2]).astype(int)),col,3)
    cv2.putText(im,'facing '+name,(5,25),0,0.8,(255,255,0),2); ims.append(im)
cv2.imwrite('mesh_views.jpg',np.vstack([np.hstack(ims[:3]),np.hstack(ims[3:])])[...,::-1])
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/mesh_views.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
d=load_inputs('/work/inputs'); rec=d['rec']
T=rec['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64); cc=rec['T_cam_mesh_coconet'].astype(np.float64)
dR=rot_angle(T[:-1,:3,:3],T[1:,:3,:3]); dt=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)*100
print('rot step deg: pctiles',np.percentile(dR,[50,90,99,100]).round(2)); print('big jumps',[(i,round(dR[i],1)) for i in np.where(dR>5)[0]])
print('trans step cm pctiles',np.percentile(dt,[50,90,99,100]).round(2))
print('baseline vs fp rot', np.percentile(rot_angle(Tc[:,:3,:3],fp[:,:3,:3]),[10,50,90,100]).round(1))
print('baseline vs cc rot', np.percentile(rot_angle(Tc[:,:3,:3],cc[:,:3,:3]),[10,50,90,100]).round(1))
print('z base/fp/cc', np.median(Tc[:,2,3]),np.median(fp[:,2,3]),np.median(cc[:,2,3]))
for i in range(0,1003,25):
    print(i, 'pos',Tc[i,:3,3].round(3),'fpz',fp[i,2,3].round(3),'ccz',cc[i,2,3].round(3),'rfp',rot_angle(Tc[i,:3,:3],fp[i,:3,:3]).round(1),'rcc',rot_angle(Tc[i,:3,:3],cc[i,:3,:3]).round(1),'contact',rec['contact_prob_right_left'][i].round(2))
EOF
```

<details><summary>output</summary>

```
rot step deg: pctiles [0.57 1.46 2.51 4.95]
big jumps []
trans step cm pctiles [ 0.94  3.28  7.31 10.85]
baseline vs fp rot [2.4 3.6 5.2 9.8]
baseline vs cc rot [1.4 1.4 1.5 1.5]
z base/fp/cc 2.8456006776877634 3.016542434692383 2.9844706058502197
0 pos [-0.273  0.645  2.524] fpz 2.698 ccz 2.667 rfp 3.4 rcc 1.4 contact [0. 0.]
25 pos [-0.275  0.654  2.569] fpz 2.784 ccz 2.723 rfp 2.8 rcc 1.4 contact [0. 0.]
50 pos [-0.277  0.653  2.583] fpz 2.768 ccz 2.727 rfp 3.4 rcc 1.4 contact [0. 0.]
75 pos [-0.269  0.649  2.565] fpz 2.809 ccz 2.735 rfp 3.8 rcc 1.4 contact [0. 0.]
100 pos [-0.266  0.644  2.543] fpz 2.782 ccz 2.722 rfp 3.8 rcc 1.4 contact [0. 0.]
125 pos [-0.265  0.644  2.512] fpz 2.721 ccz 2.686 rfp 4.5 rcc 1.4 contact [1. 1.]
150 pos [-0.269  0.649  2.554] fpz 2.774 ccz 2.734 rfp 5.0  …[+2806 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
d=load_inputs('/work/inputs'); rec=d['rec']; mesh=load_mesh('/work/inputs')
V=np.asarray(mesh.vertices)[::20]
T=rec['T_world_mesh'].astype(np.float64)
n,dd=d['plane'][:3],d['plane'][3]
h=np.einsum('fij,nj->fni',T[:,:3,:3],V)+T[:,None,:3,3]
hh=h@n+dd
mn=hh.min(1)
for i in range(0,1003,25): print(i, round(mn[i]*100,1), end=' | ')
print(); print('median',np.median(mn)*100)
# human feet height
kp=rec['human_keypoints_world']; hv=rec['human_vertices_world']
hf=(hv@n+dd).min(1); print('human min height pct', np.percentile(hf*100,[5,50,95]))
ref=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; print('ref human min h',(ref@n+dd).min()*100, 'recon frame0 min', hf[0]*100)
EOF
```

<details><summary>output</summary>

```
0 -0.8 | 25 -2.3 | 50 -2.2 | 75 -1.6 | 100 -0.9 | 125 -0.8 | 150 -1.7 | 175 -1.7 | 200 -1.2 | 225 -1.9 | 250 -0.3 | 275 0.0 | 300 2.5 | 325 2.6 | 350 -0.1 | 375 0.5 | 400 3.4 | 425 3.5 | 450 -1.7 | 475 1.4 | 500 0.2 | 525 3.3 | 550 1.8 | 575 3.9 | 600 -0.8 | 625 -1.1 | 650 0.5 | 675 -0.4 | 700 -1.3 | 725 -0.3 | 750 -1.0 | 775 -0.6 | 800 -1.5 | 825 -0.2 | 850 -1.8 | 875 -0.8 | 900 -2.2 | 925 -2.2 | 950 -1.9 | 975 -2.0 | 1000 -1.9 | 
median -0.7042970402806548
human min height pct [-4.13228433 -1.75255264  2.14363011]
ref human min h 2.9927336093976242 recon frame0 min -0.16507651620860653
```
</details>

**Bash**
```
cd /work/scratch; cat > iou.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
om=torch.tensor(np.load('objmask_half.npy')).cuda(); pm=torch.tensor(np.load('permask_half.npy')).cuda()
def ious(T_wm, scale=0.5):
    Tc=cam_from_world(d['T_wc'],T_wm); out=[]
    for s in range(0,len(Tc),32):
        r=R.render(Tc[s:s+32],d['K'],d['H'],d['W'],scale=scale,need_rgb=False)['mask']
        o=om[s:s+32]; valid=~(pm[s:s+32]&~o)
        inter=(r&o&valid).sum((1,2)).float(); uni=((r|o)&valid).sum((1,2)).float()
        out.append((inter/uni).cpu().numpy())
    return np.concatenate(out)
if __name__=='__main__':
    T=d['rec']['T_world_mesh'].astype(np.float64)
    i=ious(T); np.save('iou_base.npy',i)
    print('mean',i.mean()); 
    for s in range(0,1003,25): print(s, i[s:s+25].mean().round(3), i[s:s+25].min().round(3), end=' | ')
EOF
python iou.py
```

<details><summary>output</summary>

```
mean 0.93153054
0 0.951 0.944 | 25 0.962 0.955 | 50 0.931 0.91 | 75 0.932 0.919 | 100 0.94 0.931 | 125 0.941 0.935 | 150 0.936 0.919 | 175 0.921 0.911 | 200 0.909 0.891 | 225 0.905 0.884 | 250 0.922 0.906 | 275 0.93 0.914 | 300 0.949 0.939 | 325 0.943 0.926 | 350 0.932 0.922 | 375 0.946 0.932 | 400 0.945 0.932 | 425 0.951 0.939 | 450 0.949 0.938 | 475 0.925 0.908 | 500 0.904 0.885 | 525 0.901 0.881 | 550 0.923 0.905 | 575 0.931 0.913 | 600 0.951 0.937 | 625 0.94 0.924 | 650 0.937 0.928 | 675 0.943 0.935 | 700 0.941 0.932 | 725 0.952 0.944 | 750 0.947 0.941 | 775 0.941 0.933 | 800 0.919 0.891 | 825 0.909 0.899 | 850 0.916 0.892 | 875 0.917 0.898 | 900 0.914 0.903 | 925 0.915 0.903 | 950 0.913 0.902 | 975 0.928 0.914 | 1000 0.928 0.926 |
```
</details>

**Bash**
```
cd /work/scratch; cat > sbs.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def getframe(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read(); return f[...,::-1]
def sbs(T_wm_list, idx, out):
    rows=[]
    for i in idx:
        im=getframe(i).astype(np.float32)/255; tiles=[]
        rs=[]
        for T_wm in T_wm_list:
            Tc=cam_from_world(d['T_wc'],T_wm[i:i+1]); r=R.render(Tc,d['K'],d['H'],d['W'])
            rs.append((r['rgb'][0].cpu().numpy(),r['mask'][0].cpu().numpy()))
        m=rs[0][1]; ys,xs=np.where(m); y0,y1,x0,x1=max(ys.min()-40,0),ys.max()+40,max(xs.min()-40,0),xs.max()+40
        def prep(a): 
            a=a[y0:y1,x0:x1]; a=np.clip(a/ max(np.percentile(a[a>0],95),1e-3) ,0,1) if a.max()>0 else a
            return cv2.resize((a**0.7*255).astype(np.uint8),(320,320))
        t=prep(im.copy()); cv2.putText(t,str(i),(5,20),0,0.7,(255,255,0),2); tiles.append(t)
        for rgb,mm in rs: tiles.append(prep(rgb))
        rows.append(np.hstack(tiles))
    cv2.imwrite(out,np.vstack(rows)[...,::-1])
if __name__=='__main__':
    T=d['rec']['T_world_mesh'].astype(np.float64)
    idx=[int(x) for x in sys.argv[2].split(',')]
    sbs([T],idx,sys.argv[1])
EOF
python sbs.py sbs1.jpg 0,150,250,300,400,500; python sbs.py sbs2.jpg 550,650,750,850,900,1000
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sbs1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/sbs2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; nproc; free -g; df -h /work | tail -1; time python - <<'EOF'
import numpy as np, cv2, time
fr=np.load('frames_half.npy', mmap_mode='r'); om=np.load('objmask_half.npy', mmap_mode='r'); pm=np.load('permask_half.npy', mmap_mode='r')
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
F=len(fr); med=np.zeros(F); p90=np.zeros(F); bg=np.zeros(F)
g=[cv2.cvtColor(np.ascontiguousarray(f),cv2.COLOR_RGB2GRAY) for f in fr]
flows=np.zeros((F,576,768,2),np.float16)
for i in range(F-1):
    fl=dis.calc(g[i],g[i+1],None); flows[i]=fl
    m=om[i]&~pm[i]
    m=cv2.erode(m.astype(np.uint8),np.ones((5,5)))>0
    mag=np.linalg.norm(fl,axis=2)
    if m.sum()>50: med[i]=np.median(mag[m]); p90[i]=np.percentile(mag[m],90)
    bgm=~(cv2.dilate((om[i]|pm[i]).astype(np.uint8),np.ones((15,15)))>0)
    bg[i]=np.median(mag[bgm])
np.save('flow_half.npy',flows); np.save('objflow_stats.npy',np.stack([med,p90,bg]))
for s in range(0,F,10): print(s, med[s:s+10].round(2).max(), end=' | ')
print(); print('bg',np.percentile(bg,[50,99]))
EOF
```

<details><summary>output</summary>

```
48
               total        used        free      shared  buff/cache   available
Mem:             125          24          36           0          64          99
Swap:              1           1           0
10.127.102.71:/Group/hanzhou/v2d/agent_runs/CL2_ep005  139T   49T   90T  36% /work
0 0.35 | 10 0.43 | 20 0.48 | 30 0.3 | 40 0.36 | 50 0.43 | 60 0.37 | 70 0.36 | 80 0.37 | 90 0.42 | 100 0.4 | 110 0.42 | 120 0.53 | 130 0.53 | 140 0.43 | 150 0.35 | 160 0.52 | 170 0.71 | 180 0.75 | 190 1.07 | 200 1.29 | 210 0.73 | 220 0.35 | 230 0.4 | 240 0.81 | 250 1.0 | 260 1.1 | 270 0.57 | 280 0.56 | 290 0.37 | 300 0.24 | 310 0.12 | 320 0.24 | 330 0.2 | 340 0.15 | 350 0.11 | 360 0.12 | 370 0.13 | 380 0.08 | 390 0.07 | 400 0.08 | 410 0.14 | 420 0.16 | 430 0.15 | 440 0.13 | 450 0.21 | 460 0.85 | 470 0.8 …[+659 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2
fr=np.load('frames_half.npy', mmap_mode='r'); om=np.load('objmask_half.npy', mmap_mode='r'); pm=np.load('permask_half.npy', mmap_mode='r')
F=len(fr); k=5
diff=np.zeros(F); miou=np.zeros(F)
for i in range(k,F):
    a=fr[i].astype(np.float32).mean(2); b=fr[i-k].astype(np.float32).mean(2)
    m=om[i]&om[i-k]&~pm[i]&~pm[i-k]
    m=cv2.erode(m.astype(np.uint8),np.ones((7,7)))>0
    diff[i]=np.abs(a-b)[m].mean() if m.sum()>100 else -1
    va=~pm[i]&~pm[i-k]
    miou[i]=(om[i]&om[i-k]&va).sum()/max(((om[i]|om[i-k])&va).sum(),1)
np.save('static_feats.npy',np.stack([diff,miou]))
for s in range(0,F,10): print(s, diff[s:s+10].max().round(1), miou[s:s+10].min().round(3), end=' | ')
EOF
```

<details><summary>output</summary>

```
0 1.4 0.0 | 10 1.3 0.991 | 20 1.4 0.987 | 30 1.4 0.989 | 40 1.4 0.989 | 50 1.6 0.989 | 60 1.6 0.99 | 70 1.6 0.995 | 80 1.5 0.996 | 90 1.4 0.981 | 100 1.4 0.991 | 110 1.5 0.995 | 120 1.6 0.996 | 130 1.5 0.993 | 140 1.4 0.995 | 150 1.4 0.995 | 160 1.7 0.985 | 170 2.1 0.962 | 180 1.9 0.948 | 190 2.4 0.947 | 200 3.2 0.958 | 210 2.9 0.959 | 220 2.7 0.961 | 230 2.6 0.968 | 240 5.5 0.965 | 250 10.7 0.946 | 260 10.1 0.944 | 270 8.2 0.964 | 280 5.4 0.972 | 290 4.1 0.978 | 300 3.3 0.983 | 310 2.2 0.99 | 320 3.3 0.963 | 330 3.7 0.955 | 340 2.6 0.981 | 350 2.4 0.986 | 360 2.3 0.991 | 370 2.2 0.986 | 380 2.1 0.977 | 390 1.6 0.986 | 400 1.5 0.99 | 410 2.0 0.983 | 420 2.3 0.98 | 430 2.5 0.985 | 440 2.9 0.989 | 450 2.7 0.982 | 460 5.8 0.954 | 470 6.6 0.922 | 480 5.5 0.951 | 490 10.3 0.884 | 500 9.5 0.894  …[+803 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2
fr=np.load('frames_half.npy', mmap_mode='r'); om=np.load('objmask_half.npy', mmap_mode='r')
def sheet(idx,out,sz=150,cols=10):
    tiles=[]
    for i in idx:
        ys,xs=np.where(om[i]); cy,cx=int(ys.mean()),int(xs.mean()); h=130
        y0,x0=max(cy-h,0),max(cx-h,0)
        t=cv2.resize(np.ascontiguousarray(fr[i][y0:y0+2*h,x0:x0+2*h]),(sz,sz))[...,::-1].copy()
        cv2.putText(t,str(i),(3,15),0,0.5,(0,255,255),1); tiles.append(t)
    while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out,np.vstack([np.hstack(tiles[j:j+cols]) for j in range(0,len(tiles),cols)]))
sheet(range(0,500,5),'sheetA.jpg'); sheet(range(500,1003,5),'sheetB.jpg')
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/sheetA.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/sheetB.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
d=load_inputs('/work/inputs'); rec=d['rec']
T=rec['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64)
for a,b in [(0,165),(340,450),(580,750),(920,1003)]:
    R=Rot.from_matrix(Tc[a:b,:3,:3]); m=R.mean(); ang=(R*m.inv()).magnitude()*57.3
    Rf=Rot.from_matrix(fp[a:b,:3,:3]); mf=Rf.mean(); angf=(Rf*mf.inv()).magnitude()*57.3
    t=Tc[a:b,:3,3]; 
    uv=t[:,:2]/t[:,2:]*820
    print(a,b,'rot dev base med/max',np.median(ang).round(2),ang.max().round(2),'fp',np.median(angf).round(2),angf.max().round(2),'| t std cm',(t.std(0)*100).round(2),'uv std px',uv.std(0).round(2), 'fp t std', (fp[a:b,:3,3].std(0)*100).round(2))
EOF
```

<details><summary>output</summary>

```
0 165 rot dev base med/max 0.73 2.24 fp 1.1 2.73 | t std cm [0.42 0.42 2.03] uv std px [1.02 1.38] fp t std [0.41 0.63 2.87]
340 450 rot dev base med/max 0.67 3.35 fp 0.93 3.07 | t std cm [0.3 1.1 3.7] uv std px [0.81 1.23] fp t std [0.31 1.07 4.03]
580 750 rot dev base med/max 0.73 3.03 fp 0.79 3.41 | t std cm [0.67 0.99 2.64] uv std px [1.53 1.6 ] fp t std [0.93 2.23 8.04]
920 1003 rot dev base med/max 0.56 1.62 fp 0.65 1.5 | t std cm [0.14 0.27 0.83] uv std px [0.36 0.6 ] fp t std [0.25 0.79 2.09]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=load_inputs('/work/inputs'); rec=d['rec']
T=rec['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
fp=rec['T_cam_mesh_foundationpose'].astype(np.float64); cc=rec['T_cam_mesh_coconet'].astype(np.float64)
st=np.load('static_feats.npy')
fig,ax=plt.subplots(5,1,figsize=(16,16),sharex=True)
ax[0].plot(Tc[:,2,3],label='base z'); ax[0].plot(cc[:,2,3]*np.median(Tc[:,2,3]/cc[:,2,3]),label='cc z scaled'); ax[0].plot(fp[:,2,3]*np.median(Tc[:,2,3]/fp[:,2,3]),label='fp z scaled'); ax[0].legend()
ax[1].plot(Tc[:,2,3]/cc[:,2,3],label='base/cc z ratio'); ax[1].legend()
u=Tc[:,0,3]/Tc[:,2,3]*820+758; v=Tc[:,1,3]/Tc[:,2,3]*820+560
uc=cc[:,0,3]/cc[:,2,3]*834.8+768; vc=cc[:,1,3]/cc[:,2,3]*834.8+576
ax[2].plot(u-uc,label='du base-cc'); ax[2].plot(v-vc,label='dv'); ax[2].legend()
ax[3].plot(st[0],label='img diff'); ax[3].legend()
ax[4].plot(st[1],label='mask iou k5'); ax[4].plot(np.load('iou_base.npy'),label='render iou'); ax[4].legend(); ax[4].set_ylim(0.85,1)
for a in ax: a.grid(); a.set_xticks(range(0,1003,50))
plt.tight_layout(); plt.savefig('plot_z.png',dpi=60)
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/plot_z.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=load_inputs('/work/inputs'); rec=d['rec']; mesh=load_mesh('/work/inputs')
V=np.asarray(mesh.vertices)[::10]
T=rec['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
cc=rec['T_cam_mesh_coconet'].astype(np.float64)
n,dd=d['plane'][:3],d['plane'][3]
def minh(Tcam):
    Tw=d['T_wc'][None]@Tcam
    h=np.einsum('fij,nj->fni',Tw[:,:3,:3],V)+Tw[:,None,:3,3]
    return (h@n+dd).min(1)
s=np.median(Tc[:,2,3]/cc[:,2,3])
Tcs=Tc.copy(); Tcs[:,:3,3]*= (cc[:,2,3]*s/Tc[:,2,3])[:,None]
h1=minh(Tc); h2=minh(Tcs)
# sensitivity: dh per unit along-ray scale
Tp=Tc.copy(); Tp[:,:3,3]*=1.01; dh=(minh(Tp)-h1)/(0.01*Tc[:,2,3])
print('dh/dz pct',np.percentile(dh,[5,50,95]))
plt.figure(figsize=(16,5)); plt.plot(h1*100,label='base'); plt.plot(h2*100,label='cc*const'); plt.legend(); plt.grid(); plt.xticks(range(0,1003,50)); plt.savefig('plot_h.png',dpi=60)
EOF
```

<details><summary>output</summary>

```
dh/dz pct [-0.3372425  -0.29725101 -0.27026717]
```
</details>

**Read**
```
/work/scratch/plot_h.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
d=load_inputs('/work/inputs'); rec=d['rec']
z=np.load('/work/inputs/frame0_reference_human.npz'); print(z.files, z['vertices_world'].shape)
ref=z['vertices_world']; hv=rec['human_vertices_world'][0]
print('ref mean',ref.mean(0),'rec mean',hv.mean(0))
Tcw=np.linalg.inv(d['T_wc'])
rc=ref@Tcw[:3,:3].T+Tcw[:3,3]; hc=hv@Tcw[:3,:3].T+Tcw[:3,3]
print('cam ref mean',rc.mean(0),'rec',hc.mean(0))
if len(ref)==len(hv) or len(ref)//4==len(hv): pass
n,dd=d['plane'][:3],d['plane'][3]
print('ref height range',(ref@n+dd).min(),(ref@n+dd).max(),'rec',(hv@n+dd).min(),(hv@n+dd).max())
EOF
```

<details><summary>output</summary>

```
['vertices_world'] (18056, 3)
ref mean [0.50223947 0.2537904  3.5740573 ] rec mean [0.49303982 0.22622815 3.4560487 ]
cam ref mean [0.32799227 0.13010036 3.41548893] rec [0.32222268 0.10094944 3.29763904]
ref height range 0.029927336093976242 1.6794949746841128 rec -0.0016507651620860653 1.6979543020929284
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


def so3_exp(w):
    """w: (B,3) torch -> (B,3,3)"""
    th = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def render_sil_soft(R, T, K, H, W, scale):
    """Differentiable antialiased silhouette for torch poses T (B,4,4)."""
    h, w = int(round(H * scale)), int(round(W * scale))
    fx, fy = K[0, 0] * scale, K[1, 1] * scale
    cx, cy = (K[0, 2] + 0.5) * scale - 0.5, (K[1, 2] + 0.5) * scale - 0.5
    n, fr = 0.05, 50.0
    P = torch.tensor([[2 * fx / w, 0, 2 * (cx + 0.5) / w - 1, 0],
                      [0, 2 * fy / h, 2 * (cy + 0.5) / h - 1, 0],
                      [0, 0, (fr + n) / (fr - n), -2 * fr * n / (fr - n)],
                      [0, 0, 1, 0]], dtype=torch.float32, device=R.dev)
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("ij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (h, w))
    sil = (rast[..., 3:4] > 0).float()
    sil = dr.antialias(sil, rast, clip, R.f)
    return sil[..., 0], rast, vc
EOF
echo ok
```

<details><summary>output</summary>

```
ok
```
</details>

**Bash**
```
cat > /work/scratch/refine_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, torch.nn.functional as Fn
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
om=np.load('objmask_half.npy', mmap_mode='r'); pm=np.load('permask_half.npy', mmap_mode='r')
def gblur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-((torch.arange(k,device=x.device)-k//2)**2)/(2*s*s)); g=g/g.sum()
    x=Fn.conv2d(x[:,None],g[None,None,None,:],padding=(0,k//2)); x=Fn.conv2d(x,g[None,None,:,None],padding=(k//2,0)); return x[:,0]
def refine(Tc0, idx, iters=150, scale=0.5, fix_z=True, lr=2e-3):
    B=len(idx); T0=torch.tensor(Tc0,dtype=torch.float32,device='cuda')
    m=torch.tensor(np.stack([om[i] for i in idx]),device='cuda').float()
    p=torch.tensor(np.stack([pm[i] for i in idx]),device='cuda')
    valid=(~(p & (m<0.5))).float()
    w=torch.zeros(B,3,device='cuda',requires_grad=True); dxy=torch.zeros(B,2,device='cuda',requires_grad=True); dlz=torch.zeros(B,1,device='cuda',requires_grad=True)
    params=[w,dxy]+([] if fix_z else [dlz])
    opt=torch.optim.Adam(params,lr=lr)
    t0=T0[:,:3,3]; xy0=t0[:,:2]/t0[:,2:]
    mb=[gblur(m,s) for s in (4,1.5)]
    for it in range(iters):
        Rn=so3_exp(w)@T0[:,:3,:3]; z=t0[:,2:]*torch.exp(dlz); t=torch.cat([(xy0+dxy)*z,z],1)
        T=torch.zeros_like(T0); T[:,:3,:3]=Rn; T[:,:3,3]=t; T[:,3,3]=1
        sil,_,_=render_sil_soft(R,T …[+1529 chars]
```

<details><summary>output</summary>

```
time 22.469023942947388 mem 1.42134528
base rot dev med/max 0.71 2.22 t std cm [0.41 0.42 2.09]
refined rot dev med/max 1.11 3.57 t std cm [0.25 0.74 2.09]
iou refined 0.9710535 base 0.94289273
rot change base->refined 3.852266416901336 z ratio 1.000000000714662
```
</details>

**Bash**
```
python refine_test.py 0 165 0
```

<details><summary>output</summary>

```
time 21.706672191619873 mem 1.421346304
base rot dev med/max 0.71 2.22 t std cm [0.41 0.42 2.09]
refined rot dev med/max 0.64 2.28 t std cm [0.14 0.4  0.94]
iou refined 0.97294694 base 0.94289273
rot change base->refined 3.424255134294252 z ratio 1.0042536670282989
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
d=load_inputs('/work/inputs'); T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
idx=list(range(0,165,3)); Tr=np.load('ref_0_165_0.npy')
dR=Rot.from_matrix(Tr[:,:3,:3]@np.swapaxes(Tc[idx][:,:3,:3],1,2)).as_rotvec()*57.3
print('delta rotvec cam frame (deg) median',np.median(dR,0), 'std',dR.std(0))
EOF
```

<details><summary>output</summary>

```
delta rotvec cam frame (deg) median [ 0.20083976 -3.29809218 -0.22029219] std [0.72331352 0.74775503 0.16979445]
```
</details>

**Bash**
```
cd /work/scratch; cat > cmp_contours.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
def getframe(i):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,f=cap.read(); return f
def draw(i, Tcs, cols, out=None, pad=60):
    im=getframe(i); im=np.clip(im.astype(np.float32)*2.2,0,255).astype(np.uint8)
    allm=None
    for Tc,col in zip(Tcs,cols):
        r=R.render(Tc[None],d['K'],d['H'],d['W']); m=r['mask'][0].cpu().numpy().astype(np.uint8)
        # depth edges (internal box edges) via depth discontinuity + silhouette
        cnt,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,col,1)
        allm=m if allm is None else allm|m
    ys,xs=np.where(allm); y0,y1,x0,x1=max(ys.min()-pad,0),ys.max()+pad,max(xs.min()-pad,0),xs.max()+pad
    crop=im[y0:y1,x0:x1]; crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    if out: cv2.imwrite(out,crop)
    return crop
EOF
python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch'); 
from cmp_contours import *
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T); Tr=np.load('/work/scratch/ref_0_165_0.npy')
draw(60,[Tc[60],Tr[20]],[(0,0,255),(255,128,0)],'/work/scratch/c60.jpg')
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/c60.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat >> cmp_contours.py <<'EOF'
import torch, nvdiffrast.torch as dr
fn=torch.tensor(np.asarray(mesh.face_normals),dtype=torch.float32,device='cuda')
def normal_edges(Tc):
    T=torch.tensor(Tc[None],dtype=torch.float32,device='cuda')
    K=d['K']; H,W=d['H'],d['W']
    P=torch.tensor([[2*K[0,0]/W,0,2*(K[0,2]+0.5)/W-1,0],[0,2*K[1,1]/H,2*(K[1,2]+0.5)/H-1,0],[0,0,1.002,-0.1],[0,0,1,0]],dtype=torch.float32,device='cuda')
    vh=torch.cat([R.v,torch.ones_like(R.v[:,:1])],1); vc=torch.einsum('bij,nj->bni',T,vh); clip=torch.einsum('ij,bnj->bni',P,vc).contiguous()
    rast,_=dr.rasterize(R.glctx,clip,R.f,(H,W))
    tri=rast[0,...,3].long()-1; nm=torch.zeros(H,W,3,device='cuda'); ok=tri>=0
    nm[ok]=fn[tri[ok]]@T[0,:3,:3].T
    nm=nm.cpu().numpy(); 
    # quantize to dominant box axes -> label image
    Rm=Tc[:3,:3]; lab=np.zeros((H,W),np.int32)
    dots=np.einsum('hwc,ck->hwk',nm,np.concatenate([Rm,-Rm],1))
    lab=np.argmax(dots,2)+1; lab[~ok.cpu().numpy()]=0
    e=np.zeros((H,W),np.uint8)
    e[:,1:]|=(lab[:,1:]!=lab[:,:-1]); e[1:,:]|=(lab[1:,:]!=lab[:-1,:])
    return e
def draw_edges(i,Tcs,cols,out,pad=60):
    im=getframe(i); im=np.clip(im.astype(np.float32)*2.2,0,255).astype(np.uint8); allm=None
    for Tc,col in zip(Tcs,cols):
        e=normal_edges(Tc); im[e>0]=col; allm=e if allm is None else allm|e
    ys,xs=np.where(allm); y0,y1,x0,x1=max(ys.min()-pad,0),ys.max()+pad,max(xs.min()-pad,0),xs.max()+pad
    crop=cv2.resize(im[y0:y1,x0:x1],None,fx=2,fy=2,inte …[+354 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/e60.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2
fr=np.load('frames_half.npy', mmap_mode='r'); om=np.load('objmask_half.npy', mmap_mode='r'); pm=np.load('permask_half.npy', mmap_mode='r')
def iou_vs(ref, rng):
    out=[]
    for i in rng:
        v=~pm[i]&~pm[ref]
        out.append(((om[i]&om[ref]&v).sum()/max(((om[i]|om[ref])&v).sum(),1)).round(3))
    return out
for ref,rng in [(50,range(0,200,10)),(380,range(290,470,10)),(650,range(540,800,10)),(980,range(880,1003,10))]:
    print(ref, list(zip(rng,iou_vs(ref,rng))))
EOF
```

<details><summary>output</summary>

```
50 [(0, 0.99), (10, 0.993), (20, 0.993), (30, 0.993), (40, 0.994), (50, 1.0), (60, 0.99), (70, 0.991), (80, 0.99), (90, 0.99), (100, 0.988), (110, 0.987), (120, 0.992), (130, 0.988), (140, 0.983), (150, 0.983), (160, 0.983), (170, 0.97), (180, 0.931), (190, 0.881)]
380 [(290, 0.846), (300, 0.907), (310, 0.928), (320, 0.932), (330, 0.974), (340, 0.98), (350, 0.978), (360, 0.974), (370, 0.982), (380, 1.0), (390, 0.966), (400, 0.964), (410, 0.963), (420, 0.973), (430, 0.978), (440, 0.973), (450, 0.969), (460, 0.978)]
650 [(540, 0.902), (550, 0.902), (560, 0.915), (570, 0.925), (580, 0.992), (590, 0.993), (600, 0.995), (610, 0.994), (620, 0.997), (630, 0.997), (640, 0.997), (650, 1.0), (660, 0.997), (670, 0.993), (680, 0.995), (690, 0.996), (700, 0.995), (710, 0.997), (720, 0.987), (730, 0.967 …[+271 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np
om=np.load('objmask_half.npy'); pm=np.load('permask_half.npy')
F=len(om)
def iou(a,b):
    v=~pm[a]&~pm[b]; return (om[a]&om[b]&v).sum()/max(((om[a]|om[b])&v).sum(),1)
k=10
s=np.array([min(iou(i,max(i-k,0)),iou(i,min(i+k,F-1))) for i in range(F)])
np.save('iou_pm10.npy',s)
for thr in [0.98,0.985]:
    st=s>=thr; runs=[]; i=0
    while i<F:
        if st[i]:
            j=i
            while j+1<F and st[j+1]: j+=1
            runs.append((i,j)); i=j+1
        else: i+=1
    print(thr,[(a,b) for a,b in runs if b-a>=20])
EOF
```

<details><summary>output</summary>

```
0.98 [(0, 158), (422, 448), (583, 713), (938, 1002)]
0.985 [(0, 86), (98, 157), (617, 711)]
```
</details>

> *thinking:* Rotations and image position look accurate, but depth jitters 2–4 cm along the viewing ray during static phases; silhouette refinement raises mask IoU to 0.97, halves that jitter, and reveals a ~3.3° yaw correction. Next I'll build a joint, temporally regularised render-and-compare optimizer with static-phase pose sharing.

**Write**
```
/work/outputs/traj_opt.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/traj_opt.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cat > /work/scratch/exp1.py <<'EOF'
import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, h5py
from hoi_lib import *
from traj_opt import *
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0)
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T); F=len(Tc)
f=h5py.File('/work/inputs/masks.h5')
S=192
import os
if os.path.exists('crops.npz'):
    z=np.load('crops.npz'); crops,om,vm=z['crops'],z['om'],z['vm']
else:
    t=time.time(); OM=f['object'][:]; PM=f['person'][:]
    crops,om,vm=make_crops(Tc,lambda i:OM[i],lambda i:PM[i],d['K'],d['H'],d['W'],R,S=S)
    np.savez('crops.npz',crops=crops,om=om,vm=vm); print('crops',time.time()-t)
static=[(0,158),(583,713),(938,1002)] if len(sys.argv)<3 else eval(sys.argv[2])
node_of=np.zeros(F,int); init=[]; n=0; i=0
while i<F:
    g=[s for s in static if s[0]==i]
    if g:
        a,b=g[0]; Rm=Rot.from_matrix(Tc[a:b+1,:3,:3]).mean().as_matrix(); tt=Tc[a:b+1,:3,3]
        xy=np.median(tt[:,:2]/tt[:,2:],0); z=np.median(tt[:,2])
        Ti=np.eye(4); Ti[:3,:3]=Rm; Ti[:3,3]=[xy[0]*z,xy[1]*z,z]; init.append(Ti); node_of[a:b+1]=n; n+=1; i=b+1
    else:
        init.append(Tc[i]); node_of[i]=n; n+=1; i+=1
init=np.stack(init)
lam=float(sys.argv[1])
def smooth(Rn,t):
    aR=Rn[:-2]-2*Rn[1:-1]+Rn[2:]; at=t[:-2]-2*t[1:-1]+t[2:]
    return lam*((aR**2).sum()+(at**2).sum())
fw=np.ones(F)
fw=fw/np.bincount …[+690 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bmz7c0kjc). Output is being written to: /tmp/claude-1003/-work/5937f8de-5fa2-45a9-afb0-76b646866b27/tasks/bmz7c0kjc.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
cat > /work/scratch/edge_score.py <<'EOF'
"""Held-out validation: chamfer distance from rendered INTERNAL crease edges (box face boundaries, not the silhouette)
to Canny edges of the video frame (person pixels excluded)."""
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch, h5py
import nvdiffrast.torch as dr
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
fn=torch.tensor(np.asarray(mesh.face_normals),dtype=torch.float32,device='cuda')
S, O = trimesh_obb = None, None
import trimesh
Sobb,_=trimesh.bounds.oriented_bounds(mesh); Robb=np.linalg.inv(Sobb)[:3,:3]  # obb axes in mesh frame (columns)
fr=None
def get_frames():
    global fr
    if fr is None: fr=np.load('/work/scratch/frames_full_gray.npy',mmap_mode='r')
    return fr
f5=h5py.File('/work/inputs/masks.h5')
def labels(Tc_batch):
    T=torch.tensor(Tc_batch,dtype=torch.float32,device='cuda'); B=len(T)
    K=d['K']; H,W=d['H'],d['W']
    P=torch.tensor([[2*K[0,0]/W,0,2*(K[0,2]+0.5)/W-1,0],[0,2*K[1,1]/H,2*(K[1,2]+0.5)/H-1,0],[0,0,1.002,-0.1],[0,0,1,0]],dtype=torch.float32,device='cuda')
    vh=torch.cat([R.v,torch.ones_like(R.v[:,:1])],1); vc=torch.einsum('bij,nj->bni',T,vh); clip=torch.einsum('ij,bnj->bni',P,vc).contiguous()
    rast,_=dr.rasterize(R.glctx,clip,R.f,(H,W))
    tri=rast[...,3].long()-1; ok=tri>=0
    A=torch.tensor(np.concatenate([Robb,-Robb],1),dtype=torch.float32,device='cuda')  # 3x6 obb face normals (mesh frame)
    lab=tor …[+1540 chars]
```

<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 1774780544 Oct 21 10:16 frames_full_gray.npy
```
</details>

**Bash**
```
cat /tmp/claude-1003/-work/5937f8de-5fa2-45a9-afb0-76b646866b27/tasks/bmz7c0kjc.output
```

<details><summary>output</summary>

```
crops 11.921427965164185
  it 0 sil 18.1766 smooth 71.8104
  it 25 sil 10.4388 smooth 2.3014
  it 50 sil 9.0914 smooth 0.3762
  it 75 sil 8.2916 smooth 0.1936
  it 100 sil 7.7954 smooth 0.1593
  it 125 sil 7.5047 smooth 0.1436
  it 150 sil 7.3376 smooth 0.1359
  it 175 sil 7.2518 smooth 0.1322
  it 199 sil 7.1958 smooth 0.1296
opt time 127.67588472366333 mem 2.2253056
iou init 0.933719 final 0.9669373
0 0.949 0.969 rot chg 2.19 z ratio 1.0004
100 0.938 0.969 rot chg 2.1 z ratio 1.0099
200 0.918 0.966 rot chg 1.01 z ratio 1.0066
300 0.942 0.969 rot chg 1.5 z ratio 1.0064
400 0.943 0.971 rot chg 1.46 z ratio 1.0064
500 0.919 0.954 rot chg 1.67 z ratio 1.0124
600 0.944 0.969 rot chg 1.93 z ratio 1.0033
700 0.942 0.969 rot chg 0.64 z ratio 1.0022
800 0.918 0.959 rot chg 0.84 z ratio 1.0177
900 …[+107 chars]
```
</details>

**Bash**
```
cat > eval_edges.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from edge_score import *
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
idx=list(range(0,1003,10))
res={'base':score(Tc,idx)}
for fn_ in sys.argv[1:]:
    res[fn_]=score(np.load(fn_),idx)
names=list(res)
print('frames   '+'  '.join(names))
for a in range(0,1003,100):
    sl=slice(a//10,(a+100)//10); print(a, '  '.join(f'{np.nanmean(res[n][sl]):.3f}' for n in names))
print('ALL', '  '.join(f'{np.nanmean(res[n]):.3f}' for n in names))
EOF
python eval_edges.py exp1_50.0.npy
```

<details><summary>output</summary>

```
frames   base  exp1_50.0.npy
0 3.127  2.759
100 3.336  2.918
200 3.223  2.659
300 2.089  1.919
400 2.412  2.236
500 2.387  2.286
600 1.904  1.677
700 1.893  1.588
800 2.374  1.883
900 2.541  1.965
1000 2.256  1.865
ALL 2.526  2.186
```
</details>

> *thinking:* Joint silhouette optimization raised mean IoU from 0.934 to 0.967, and box-edge alignment error on held-out data dropped from 2.53 to 2.19 px across every segment. Next, I'll speed things up with a decimated mesh and tune the smoothness weight using held-out frames.

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, trimesh, open3d as o3d, time
from hoi_lib import load_mesh
m=load_mesh('/work/inputs')
t=time.time()
mm=trimesh.Trimesh(m.vertices,m.faces,process=True)  # merge duplicate verts
print(mm.vertices.shape, mm.faces.shape, mm.is_watertight)
om=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mm.vertices),o3d.utility.Vector3iVector(mm.faces))
s=om.simplify_quadric_decimation(8000)
print(len(s.vertices),len(s.triangles),time.time()-t)
v=np.asarray(s.vertices); print(v.min(0),v.max(0), m.vertices.min(0), m.vertices.max(0))
EOF
```

<details><summary>output</summary>

```
(49591, 3) (96454, 3) False
4259 7999 0.80906081199646
[-0.25571437 -0.20899488 -0.39184707] [0.2628936  0.18599446 0.46862598] [-0.25669852 -0.20899488 -0.3920801 ] [0.26427466 0.18703176 0.46906903]
```
</details>

**Bash**
```
cat >> /work/outputs/hoi_lib.py <<'EOF'


class SilMesh:
    """Decimated untextured mesh (same mesh-file frame) for fast silhouette rendering."""

    def __init__(self, mesh, n_faces=8000, device="cuda", glctx=None):
        import open3d as o3d
        mm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mm.vertices), o3d.utility.Vector3iVector(mm.faces))
        s = om.simplify_quadric_decimation(n_faces)
        self.dev = device
        self.glctx = glctx if glctx is not None else dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(s.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(s.triangles), dtype=torch.int32, device=device)
EOF
cd /work/scratch; sed -i 's/^opt=TrajOpt(R,/SM=SilMesh(mesh,glctx=R.glctx)\nopt=TrajOpt(SM,/' exp1.py
python - <<'EOF'
s=open('/work/scratch/exp1.py').read()
s=s.replace("fw=fw/np.bincount(node_of)[node_of]**0.5  # static groups: total weight ~ sqrt(n) frames",
"fw=fw/np.bincount(node_of)[node_of]**0.5  # static groups: total weight ~ sqrt(n) frames\nho=np.zeros(F,bool)\nif '--holdout' in sys.argv:\n    mv=np.where(np.bincount(node_of)[node_of]==1)[0]; ho[mv[2::5]]=True; fw[ho]=0\n")
s=s.replace("print('iou init',i0.mean(),'final',i1.mean())","print('iou init',i0.mean(),'final',i1.mean(),'holdout init',i0[ho].mean(),'final',i1[ho].mean())")
s=s.replace("np.save(f'exp1_{lam}.npy',Tout)","np.save(f'exp1_ …[+338 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bjvr6yfl1). Output is being written to: /tmp/claude-1003/-work/5937f8de-5fa2-45a9-afb0-76b646866b27/tasks/bjvr6yfl1.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/5937f8de-5fa2-45a9-afb0-76b646866b27/tasks/bjvr6yfl1.output
```

<details><summary>output</summary>

```
opt time 71.39116716384888 mem 1.010034688
iou init 0.9335206 final 0.9694334 holdout init 0.9287297 final 0.9669894
opt time 77.4383282661438 mem 1.010034688
iou init 0.9335206 final 0.9629884 holdout init 0.9287297 final 0.9573565
```
</details>

**Bash**
```
sleep 45; cat /tmp/claude-1003/-work/5937f8de-5fa2-45a9-afb0-76b646866b27/tasks/bjvr6yfl1.output
```

<details><summary>output</summary>

```
opt time 71.39116716384888 mem 1.010034688
iou init 0.9335206 final 0.9694334 holdout init 0.9287297 final 0.9669894
opt time 77.4383282661438 mem 1.010034688
iou init 0.9335206 final 0.9629884 holdout init 0.9287297 final 0.9573565
opt time 71.21803426742554 mem 1.010034688
iou init 0.9335206 final 0.95409507 holdout init 0.9287297 final 0.94996464

[exited with code 0]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
d=load_inputs('/work/inputs'); T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
for name,X in [('base',Tc),('l10',np.load('exp1_10.0_ho.npy')),('l50',np.load('exp1_50.0_ho.npy')),('l250',np.load('exp1_250.0_ho.npy'))]:
    z=X[:,2,3]; az=np.abs(z[2:]-2*z[1:-1]+z[:-2])*100
    ar=rot_angle(X[1:,:3,:3],X[:-1,:3,:3])
    print(name,'z accel cm med/p90',np.median(az).round(3),np.percentile(az,90).round(3),'rot step med/p90',np.median(ar).round(3),np.percentile(ar,90).round(3))
EOF
```

<details><summary>output</summary>

```
base z accel cm med/p90 1.367 4.874 rot step med/p90 0.569 1.462
l10 z accel cm med/p90 0.025 0.15 rot step med/p90 0.276 1.047
l50 z accel cm med/p90 0.019 0.109 rot step med/p90 0.27 1.084
l250 z accel cm med/p90 0.016 0.101 rot step med/p90 0.262 1.092
```
</details>

**Bash**
```
cat >> /work/outputs/traj_opt.py <<'EOF'


def make_edge_maps(crops, gray_fn, person_fn, E=320, trunc=10.0):
    """Per-frame distance transform (crop px at res E) to Canny edges of the contrast-equalised frame crop,
    plus a person mask (dilated) at the same res. Returns float16 arrays (F,E,E)."""
    import cv2
    F = len(crops)
    dts = np.zeros((F, E, E), np.float16)
    pms = np.zeros((F, E, E), np.uint8)
    for i in range(F):
        x0, y0, c = crops[i]
        M = np.array([[E / c, 0, -x0 * E / c], [0, E / c, -y0 * E / c]])
        g = cv2.warpAffine(gray_fn(i), M, (E, E), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        g = cv2.equalizeHist(g)
        ed = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.2), 40, 100)
        dts[i] = np.minimum(cv2.distanceTransform((ed == 0).astype(np.uint8), cv2.DIST_L2, 3), trunc)
        p = cv2.warpAffine(person_fn(i).astype(np.uint8), M, (E, E), flags=cv2.INTER_NEAREST)
        pms[i] = cv2.dilate(p, np.ones((7, 7))) > 0
    return dts, pms


@torch.no_grad()
def extract_edge_points(SM, fn_obb, T, P, E, n_pts=256, gen=None):
    """Internal crease edges of the box at pose T, as mesh-frame 3D points (B,n_pts,3) + validity (B,n_pts).
    fn_obb: (Fc,) label of each face = dominant OBB face direction (1..6)."""
    vh = torch.cat([SM.v, torch.ones_like(SM.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("bij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(SM.glctx, clip, SM.f,  …[+1622 chars]
```

<details><summary>output</summary>

```
done
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('traj_opt.py').read()
old_start=s.index("    def run(self, iters, frame_w, smooth_fn")
old_end=s.index("    @torch.no_grad()\n    def ious")
new='''    def run(self, iters, frame_w, smooth_fn, lr=2e-3, chunk=96, log=None, edge=None, edge_w=None, reextract=50):
        """edge: dict(dts, pms, E, fn_obb, mu) enables the crease-edge chamfer term (weights edge_w per frame)."""
        opt = torch.optim.Adam([self.w, self.dxy, self.dlz], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.1)
        frame_w = torch.as_tensor(frame_w, dtype=torch.float32, device=self.dev)
        if edge is not None:
            E = edge["E"]
            Pe = crop_proj(self.K, self.crops, E, self.dev)
            edge_w = torch.as_tensor(edge_w, dtype=torch.float32, device=self.dev)
            gen = torch.Generator().manual_seed(0)
        for it in range(iters):
            if edge is not None and it % reextract == 0:
                pts, val = [], []
                for s in range(0, self.F, chunk):
                    idx = torch.arange(s, min(s + chunk, self.F), device=self.dev)
                    p, v = extract_edge_points(self.R, edge["fn_obb"], self.frame_T(idx).detach(), Pe[idx], E, gen=gen)
                    pts.append(p)
                    val.append(v)
                pts, val = torch.cat(pts), torch.cat(val)
            opt.zero_grad()
            tot, tote = 0.0, 0.0
            for s in range(0, se …[+1074 chars]
```

<details><summary>output</summary>

```
94:        self.crops = crops
128:            Pe = crop_proj(self.K, self.crops, E, self.dev)
```
</details>

**Bash**
```
cat > /work/scratch/exp2.py <<'EOF'
import sys, time, os; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, h5py, trimesh
from hoi_lib import *
from traj_opt import *
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0)
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh); SM=SilMesh(mesh,glctx=R.glctx)
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T); F=len(Tc)
z=np.load('/work/scratch/crops.npz'); crops,om,vm=z['crops'],z['om'],z['vm']; S=192
static=[(0,158),(583,713),(938,1002)]
node_of=np.zeros(F,int); init=[]; n=0; i=0
while i<F:
    g=[s for s in static if s[0]==i]
    if g:
        a,b=g[0]; Rm=Rot.from_matrix(Tc[a:b+1,:3,:3]).mean().as_matrix(); tt=Tc[a:b+1,:3,3]
        xy=np.median(tt[:,:2]/tt[:,2:],0); zz=np.median(tt[:,2])
        Ti=np.eye(4); Ti[:3,:3]=Rm; Ti[:3,3]=[xy[0]*zz,xy[1]*zz,zz]; init.append(Ti); node_of[a:b+1]=n; n+=1; i=b+1
    else:
        init.append(Tc[i]); node_of[i]=n; n+=1; i+=1
init=np.stack(init)
lam=10.0
def smooth(Rn,t):
    aR=Rn[:-2]-2*Rn[1:-1]+Rn[2:]; at=t[:-2]-2*t[1:-1]+t[2:]
    return lam*((aR**2).sum()+(at**2).sum())
fw=np.ones(F)/np.bincount(node_of)[node_of]**0.5
opt=TrajOpt(SM,d['K'],init,node_of,crops,om,vm,S)
if os.path.exists('/work/scratch/stage1.npz'):
    st=np.load('/work/scratch/stage1.npz'); [p.data.copy_(torch.tensor(st[k],device='cuda')) for k,p in zip(['w','dxy','dlz'],[opt.w,opt.dxy,opt.dlz])]
else:
    t=time.time(); opt.run(150,f …[+1587 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bvzpyjywz). Output is being written to: /tmp/claude-1003/-work/5937f8de-5fa2-45a9-afb0-76b646866b27/tasks/bvzpyjywz.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/outputs; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
cat > /work/scratch/eval_edges2.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from edge_score import *
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
idx=list(range(int(sys.argv[1]),1003,10))
res={'base':score(Tc,idx)}
for fn_ in sys.argv[2:]:
    res[fn_.split('/')[-1]]=score(np.load(fn_),idx)
names=list(res)
print('frames   '+'  '.join(names))
for a in range(0,1003,100):
    sl=[k for k,i in enumerate(idx) if a<=i<a+100]; print(a, '  '.join(f'{np.nanmean(res[n][sl]):.3f}' for n in names))
print('ALL', '  '.join(f'{np.nanmean(res[n]):.3f}' for n in names))
EOF
sleep 60; cat /tmp/claude-1003/-work/5937f8de-5fa2-45a9-afb0-76b646866b27/tasks/bvzpyjywz.output
```

<details><summary>output</summary>

```
  it 0 data 20.2749 (edge 0.0000) smooth 14.3621
  it 25 data 11.2132 (edge 0.0000) smooth 0.4533
  it 50 data 9.4321 (edge 0.0000) smooth 0.0900
  it 75 data 8.5096 (edge 0.0000) smooth 0.0544
  it 100 data 8.0398 (edge 0.0000) smooth 0.0466
  it 125 data 7.8044 (edge 0.0000) smooth 0.0442
  it 149 data 7.6938 (edge 0.0000) smooth 0.0426
stage1 48.29827165603638
edges 51.17712473869324
  it 0 data 17.7009 (edge 10.0117) smooth 0.0426
  it 25 data 16.9662 (edge 9.2721) smooth 0.0859
  it 50 data 16.6767 (edge 9.2198) smooth 0.0565
  it 75 data 16.4275 (edge 9.1266) smooth 0.0526
  it 99 data 16.3870 (edge 9.1052) smooth 0.0494
stage2 35.63142991065979 mem 1.452256768
iou 0.9685597

[exited with code 0]
```
</details>

**Bash**
```
cd /work/scratch && python eval_edges2.py 5 stage1_T.npy stage2_0.01.npy
```

<details><summary>output</summary>

```
frames   base  stage1_T.npy  stage2_0.01.npy
0 3.229  2.694  2.161
100 3.217  2.515  2.136
200 3.326  2.551  2.337
300 2.161  1.937  1.726
400 2.461  2.313  2.109
500 2.397  2.268  1.985
600 1.892  1.682  1.437
700 1.902  1.578  1.459
800 2.504  1.929  1.906
900 2.576  2.021  1.880
/work/scratch/eval_edges2.py:11: RuntimeWarning: Mean of empty slice
  sl=[k for k,i in enumerate(idx) if a<=i<a+100]; print(a, '  '.join(f'{np.nanmean(res[n][sl]):.3f}' for n in names))
1000 nan  nan  nan
ALL 2.567  2.149  1.914
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
d=load_inputs('/work/inputs'); T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
A=np.load('stage1_T.npy'); B=np.load('stage2_0.01.npy')
for a in range(0,1003,100):
    s=slice(a,a+100)
    print(a,'rot base-s1',np.median(rot_angle(Tc[s,:3,:3],A[s,:3,:3])).round(2),'s1-s2',np.median(rot_angle(A[s,:3,:3],B[s,:3,:3])).round(2),'base-s2',np.median(rot_angle(Tc[s,:3,:3],B[s,:3,:3])).round(2),
      'z s1/base',np.median(A[s,2,3]/Tc[s,2,3]).round(4),'z s2/base',np.median(B[s,2,3]/Tc[s,2,3]).round(4))
EOF
```

<details><summary>output</summary>

```
0 rot base-s1 2.73 s1-s2 2.11 base-s2 1.11 z s1/base 1.0076 z s2/base 1.01
100 rot base-s1 2.6 s1-s2 2.11 base-s2 1.54 z s1/base 1.0148 z s2/base 1.0172
200 rot base-s1 1.06 s1-s2 1.37 base-s2 2.12 z s1/base 1.0097 z s2/base 1.0128
300 rot base-s1 1.83 s1-s2 1.1 base-s2 2.3 z s1/base 1.0076 z s2/base 1.0081
400 rot base-s1 1.76 s1-s2 0.73 base-s2 1.75 z s1/base 1.0096 z s2/base 1.0096
500 rot base-s1 2.01 s1-s2 0.78 base-s2 2.57 z s1/base 1.0163 z s2/base 1.0283
600 rot base-s1 1.92 s1-s2 0.66 base-s2 1.65 z s1/base 1.009 z s2/base 1.0173
700 rot base-s1 0.85 s1-s2 0.6 base-s2 0.67 z s1/base 1.007 z s2/base 1.0115
800 rot base-s1 0.82 s1-s2 1.28 base-s2 1.63 z s1/base 1.0202 z s2/base 1.0252
900 rot base-s1 1.14 s1-s2 1.75 base-s2 1.53 z s1/base 1.0386 z s2/base 1.0446
1000 rot base-s1 1.4 …[+59 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
B=np.load('stage2_0.01.npy')
for a,b in [(330,460),(420,450),(890,940),(150,175),(560,600),(700,760)]:
    ref=B[a]; print(a,b,'rot from start',[rot_angle(ref[:3,:3],B[i,:3,:3]).round(1) for i in range(a,b,5)])
    print('   trans from start cm',[ (np.linalg.norm(B[i,:3,3]-ref[:3,3])*100).round(1) for i in range(a,b,5)])
EOF
```

<details><summary>output</summary>

```
330 460 rot from start [0.2, 0.8, 0.9, 1.8, 2.0, 1.9, 1.9, 1.9, 2.1, 2.3, 2.5, 2.1, 1.7, 1.7, 1.9, 1.8, 1.5, 1.7, 2.1, 2.1, 2.1, 1.9, 1.9, 1.9, 1.9, 2.2]
   trans from start cm [0.0, 1.2, 2.0, 3.9, 3.8, 2.4, 3.6, 4.2, 5.2, 6.1, 6.1, 3.9, 2.7, 2.3, 0.8, 0.6, 0.9, 2.1, 3.2, 2.5, 2.6, 2.9, 2.8, 3.5, 4.9, 5.8]
420 450 rot from start [0.3, 0.4, 0.6, 0.4, 0.3, 0.6]
   trans from start cm [0.0, 0.8, 0.6, 0.3, 0.4, 0.4]
890 940 rot from start [0.3, 2.5, 5.5, 7.5, 7.6, 7.3, 7.2, 7.5, 7.4, 6.9]
   trans from start cm [0.0, 2.3, 2.9, 3.5, 3.6, 4.0, 3.7, 3.3, 3.3, 3.2]
150 175 rot from start [0.0, 0.0, 0.5, 1.7, 2.7]
   trans from start cm [0.0, 0.0, 0.6, 1.5, 2.7]
560 600 rot from start [0.2, 1.6, 3.4, 4.9, 4.9, 4.3, 4.3, 4.3]
   trans from start cm [0.0, 1.4, 1.5, 3.9, 4.0, 5.1, 5.1, 5.1]
700 760 ro …[+157 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
d=load_inputs('/work/inputs'); T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
B=np.load('stage2_0.01.npy'); K=d['K']
for i in range(300,470,10):
    t=B[i,:3,3]; u=K[0,0]*t[0]/t[2]+K[0,2]; v=K[1,1]*t[1]/t[2]+K[1,2]
    tb=Tc[i,:3,3]; ub=K[0,0]*tb[0]/tb[2]+K[0,2]; vb=K[1,1]*tb[1]/tb[2]+K[1,2]
    print(i,'s2 u,v,z',round(u,1),round(v,1),round(t[2],3),' base',round(ub,1),round(vb,1),round(tb[2],3))
EOF
```

<details><summary>output</summary>

```
300 s2 u,v,z 751.6 767.2 3.08  base 756.8 765.9 3.043
310 s2 u,v,z 752.3 770.6 3.116  base 758.0 767.0 3.05
320 s2 u,v,z 752.4 770.5 3.104  base 757.4 767.5 3.056
330 s2 u,v,z 755.2 771.5 3.095  base 760.7 767.5 3.033
340 s2 u,v,z 755.9 772.3 3.114  base 760.7 767.9 3.058
350 s2 u,v,z 755.8 773.7 3.129  base 761.4 768.6 3.145
360 s2 u,v,z 756.5 773.1 3.128  base 761.3 768.0 3.139
370 s2 u,v,z 756.4 774.0 3.142  base 760.7 770.0 3.12
380 s2 u,v,z 756.1 773.9 3.151  base 759.7 770.9 3.165
390 s2 u,v,z 755.9 772.3 3.121  base 760.2 770.5 3.124
400 s2 u,v,z 756.5 772.7 3.099  base 760.3 770.7 3.053
410 s2 u,v,z 756.5 772.1 3.102  base 760.0 769.5 3.09
420 s2 u,v,z 756.0 773.1 3.124  base 760.2 770.7 3.042
430 s2 u,v,z 756.3 773.2 3.118  base 759.4 770.1 3.103
440 s2 u,v,z 756.2 772.9 3.121  ba …[+128 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2
G=np.load('frames_full_gray.npy',mmap_mode='r')
z=np.load('crops.npz'); x0,y0,c=z['crops'][400].astype(int)
tiles=[]
for i in [310,340,370,400,430,460]:
    t=cv2.equalizeHist(np.ascontiguousarray(G[i][y0:y0+c,x0:x0+c])); t=cv2.resize(t,(300,300)); cv2.putText(t,str(i),(5,20),0,0.7,255,2); tiles.append(t)
ref=cv2.equalizeHist(np.ascontiguousarray(G[400][y0:y0+c,x0:x0+c]))
dif=[cv2.resize(cv2.absdiff(cv2.equalizeHist(np.ascontiguousarray(G[i][y0:y0+c,x0:x0+c])),ref),(300,300)) for i in [310,340,370,400,430,460]]
cv2.imwrite('q330.jpg',np.vstack([np.hstack(tiles),np.hstack(dif)]))
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/q330.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* Edge term validation held up well on held-out frames (internal-edge error dropped from 2.57 to 1.91 px). I spotted a remaining issue where depth wanders ±2.5 cm during the quasi-static lying phase (330–460), so I'll add a motion-adaptive velocity penalty for that and start writing `solve.py`.

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
a=s.index("    # exact per-phase median match (small residual factors), blended at boundaries")
b=s.index("    out[:, :3, 3] *= np.exp(sm)[:, None]\n    return out, rho * np.exp(sm)")
new='''    # exact per-phase median match (small residual factors); inside moving phases the factor is ramped over
    # 15 frames from the neighbouring static phase's factor, so no jump is created at phase boundaries
    lf = np.zeros(len(phases))
    for p, (_, a, b) in enumerate(phases):
        lf[p] = np.log(np.median(Tc_base[a:b + 1, 2, 3]) / np.median(out[a:b + 1, 2, 3]))
    sm = np.zeros(F)
    for p, (kind, a, b) in enumerate(phases):
        sm[a:b + 1] = lf[p]
        if kind == "moving":
            L = min(15, (b - a + 1) // 2)
            if p > 0:
                sm[a:a + L] = np.linspace(lf[p - 1], lf[p], L + 1)[1:]
            if p + 1 < len(phases):
                sm[b - L + 1:b + 1] = np.linspace(lf[p], lf[p + 1], L + 1)[:-1]
'''
s=s[:a]+new+s[b:]
s=s.replace("RATIO_SMOOTH = 2000.0   # smoothness of the depth re-anchoring ratio (2nd differences of log ratio)",
"RATIO_SMOOTH = 3e7      # smoothness of the depth re-anchoring ratio (2nd differences of log ratio; ~75-frame scale)")
open('solve.py','w').write(s)
EOF
sed -n '/^def reanchor_depth/,/^def depth_cues/p' solve.py
```

<details><summary>output</summary>

```
def reanchor_depth(Tc_new, Tc_base, phases, node_of):
    """Scale camera-frame translations along the viewing ray by a smooth ratio rho(t) (constant inside static
    phases) fitted to log(z_base/z_new); afterwards each phase's median z is matched exactly to the baseline's
    by a per-phase factor that is blended linearly over 15 frames across phase boundaries."""
    F = len(Tc_new)
    r = np.log(Tc_base[:, 2, 3] / Tc_new[:, 2, 3])
    # unknowns: one per node
    N = node_of.max() + 1
    A = np.zeros((F, N))
    A[np.arange(F), node_of] = 1
    D = np.zeros((N - 2, N))
    for k in range(N - 2):
        D[k, k:k + 3] = [1, -2, 1]
    x = np.linalg.solve(A.T @ A + RATIO_SMOOTH * D.T @ D, A.T @ r)
    rho = np.exp(x[node_of])
    out = Tc_new.copy()
    out[:, :3, 3] *= rho[:, None]
   …[+865 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, json
from solve import *
d=load_inputs('/work/inputs'); d['inp']='/work/inputs'; rec=d['rec']; mesh=load_mesh('/work/inputs')
Tc_base=cam_from_world(d['T_wc'],rec['T_world_mesh'].astype(np.float64))
runs=[(0,158),(422,448),(583,713),(938,1002)]; phases=phases_from_runs(runs,1003)
init,node_of=build_nodes(Tc_base,runs)
Tc_opt=np.load('stage2_0.01.npy')
import time; t=time.time()
Tn,ratio=reanchor_depth(Tc_opt,Tc_base,phases,node_of); print('reanchor',time.time()-t)
for k,a,b in phases:
    dz=(Tn[a:b+1,2,3]-Tc_base[a:b+1,2,3])*100
    print(k,a,b,'dz mean/med cm',dz.mean().round(2),np.median(dz).round(2),'ratio range',ratio[a:b+1].min().round(4),ratio[a:b+1].max().round(4))
z=Tn[:,2,3]; print('z accel med/p99 cm', np.median(np.abs(z[2:]-2*z[1:-1]+z[:-2]))*100, np.percentile(np.abs(z[2:]-2*z[1:-1]+z[:-2]),99)*100, 'max step cm', np.abs(np.diff(z)).max()*100)
t=time.time(); cues,allc=depth_cues(Tn,d,mesh,phases,rec); print('cues',time.time()-t); print(json.dumps(cues,indent=0)); print(allc)
EOF
```

<details><summary>output</summary>

```
reanchor 6.013341665267944
static 0 158 dz mean/med cm 0.04 0.0 ratio range 0.9867 0.9867
moving 159 421 dz mean/med cm -0.73 -0.46 ratio range 0.9836 0.9962
static 422 448 dz mean/med cm 0.68 0.02 ratio range 0.9969 0.9969
moving 449 582 dz mean/med cm 2.73 3.08 ratio range 0.9831 0.9965
static 583 713 dz mean/med cm -0.24 0.0 ratio range 0.983 0.983
moving 714 937 dz mean/med cm -5.25 -5.77 ratio range 0.9466 0.9818
static 938 1002 dz mean/med cm 0.16 0.0 ratio range 0.9563 0.9563
z accel med/p99 cm 0.023958744918228803 0.27316130123300475 max step cm 1.1950362866043562
cues 15.79342246055603
{
"static 0-158": {
"floor_cm": -19.544347584476125,
"hand_cm": -5.499999999999991
},
"moving 159-421": {
"floor_cm": -7.452722303147836,
"hand_cm": -3.99999999999999
},
"static 422-448": {
"floor_c …[+466 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from solve import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=load_inputs('/work/inputs'); rec=d['rec']
Tc_base=cam_from_world(d['T_wc'],rec['T_world_mesh'].astype(np.float64))
cc=rec['T_cam_mesh_coconet'].astype(np.float64); fp=rec['T_cam_mesh_foundationpose'].astype(np.float64)
runs=[(0,158),(422,448),(583,713),(938,1002)]; phases=phases_from_runs(runs,1003)
init,node_of=build_nodes(Tc_base,runs)
Tc_opt=np.load('stage2_0.01.npy'); Tn,ratio=reanchor_depth(Tc_opt,Tc_base,phases,node_of)
fig,ax=plt.subplots(2,1,figsize=(16,9),sharex=True)
ax[0].plot(Tc_base[:,2,3],label='base',lw=0.8); ax[0].plot(Tc_opt[:,2,3],label='opt (silhouette scale)'); ax[0].plot(Tn[:,2,3],label='opt reanchored')
ax[0].plot(cc[:,2,3]*np.median(Tc_base[:,2,3]/cc[:,2,3]),label='cc*const',lw=0.8)
ax[1].plot(Tc_opt[:,2,3]/Tc_base[:,2,3],label='opt/base'); ax[1].plot(cc[:,2,3]*np.median(Tc_base[:,2,3]/cc[:,2,3])/Tc_base[:,2,3],label='cc*c/base'); ax[1].plot(fp[:,2,3]*np.median(Tc_base[:,2,3]/fp[:,2,3])/Tc_base[:,2,3],label='fp*c/base',lw=0.6)
for a in ax:
    a.legend(); a.grid(); a.set_xticks(range(0,1003,50))
    for k,s,e in phases:
        if k=='static': a.axvspan(s,e,alpha=0.1)
plt.tight_layout(); plt.savefig('plot_z2.png',dpi=60)
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/plot_z2.png
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
a=s.index("    # exact per-phase median match (small residual factors)")
b=s.index("    out[:, :3, 3] *= np.exp(sm)[:, None]\n    return out, rho * np.exp(sm)")
s=s[:a]+s[b:].replace("    out[:, :3, 3] *= np.exp(sm)[:, None]\n    return out, rho * np.exp(sm)","    return out, rho",1)
s=s.replace('''    """Scale camera-frame translations along the viewing ray by a smooth ratio rho(t) (constant inside static
    phases) fitted to log(z_base/z_new); afterwards each phase's median z is matched exactly to the baseline's
    by a per-phase factor that is blended linearly over 15 frames across phase boundaries."""''','''    """Scale camera-frame translations along the viewing ray (image position unchanged) by a smooth ratio rho(t),
    constant inside static phases, fitted to log(z_base/z_new) (~75-frame smoothing scale). The trajectory thus
    keeps the baseline's low-frequency depth (its calibrated placement, per phase) and our relative motion."""''')
open('solve.py','w').write(s)
EOF
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from solve import *
d=load_inputs('/work/inputs'); rec=d['rec']
Tc_base=cam_from_world(d['T_wc'],rec['T_world_mesh'].astype(np.float64))
runs=[(0,158),(422,448),(583,713),(938,1002)]; phases=phases_from_runs(runs,1003)
init,node_of=build_nodes(Tc_base,runs)
Tc_opt=np.load('stage2_0.01.npy')
for a in [1e6,3e7,1e9]:
    import solve; solve.RA …[+248 chars]
```

<details><summary>output</summary>

```
1000000.0 [('static', 0, 158, -0.11, -0.11), ('moving', 159, 421, 0.29, 0.38), ('static', 422, 448, -1.36, -1.34), ('moving', 449, 582, -2.52, 1.09), ('static', 583, 713, -0.58, -0.58), ('moving', 714, 937, 5.0, -0.82), ('static', 938, 1002, 0.23, 0.23)]
30000000.0 [('static', 0, 158, -0.11, -0.11), ('moving', 159, 421, 0.93, 0.27), ('static', 422, 448, -2.6, -2.58), ('moving', 449, 582, -0.72, 2.35), ('static', 583, 713, -0.23, -0.23), ('moving', 714, 937, 4.45, -1.76), ('static', 938, 1002, 1.0, 1.0)]
1000000000.0 [('static', 0, 158, 0.65, 0.65), ('moving', 159, 421, 0.55, 0.22), ('static', 422, 448, -3.63, -3.61), ('moving', 449, 582, -1.34, 1.56), ('static', 583, 713, -0.84, -0.84), ('moving', 714, 937, 4.15, -1.7), ('static', 938, 1002, 2.49, 2.49)]
```
</details>

**Bash**
```
cd /work/outputs; sed -i 's/^RATIO_SMOOTH = 3e7 .*/RATIO_SMOOTH = 3e6      # smoothness of the depth re-anchoring ratio (2nd differences of log ratio; ~40-frame scale)/' solve.py; sed -i 's/(~75-frame smoothing scale)/(~40-frame smoothing scale)/' solve.py; grep -n "RATIO_SMOOTH\|frame smoothing" solve.py; cd /work/scratch && nohup python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/holdout/object_poses_world.npz --holdout > holdout.log 2>&1 &
```

<details><summary>output</summary>

```
42:RATIO_SMOOTH = 3e6      # smoothness of the depth re-anchoring ratio (2nd differences of log ratio; ~40-frame scale)
125:    constant inside static phases, fitted to log(z_base/z_new) (~40-frame smoothing scale). The trajectory thus
136:    x = np.linalg.solve(A.T @ A + RATIO_SMOOTH * D.T @ D, A.T @ r)
```
</details>

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
sleep 20; cat /work/scratch/holdout.log
```

<details><summary>output</summary>

```
loaded 1003 frames  10.5s
phases: [('static', 0, 158), ('moving', 159, 345), ('static', 346, 365), ('moving', 366, 421), ('static', 422, 448), ('moving', 449, 582), ('static', 583, 713), ('moving', 714, 937), ('static', 938, 1002)]
crops/edges ready 26.3s
  it 0 data 15.5586 (edge 0.0000) smooth 25.5831
  it 25 data 9.1522 (edge 0.0000) smooth 0.9011
  it 50 data 7.9651 (edge 0.0000) smooth 0.2339
  it 75 data 7.3505 (edge 0.0000) smooth 0.1481
```
</details>

**Bash**
```
sleep 75; cat /work/scratch/holdout.log
```

<details><summary>output</summary>

```
loaded 1003 frames  10.5s
phases: [('static', 0, 158), ('moving', 159, 345), ('static', 346, 365), ('moving', 366, 421), ('static', 422, 448), ('moving', 449, 582), ('static', 583, 713), ('moving', 714, 937), ('static', 938, 1002)]
crops/edges ready 26.3s
  it 0 data 15.5586 (edge 0.0000) smooth 25.5831
  it 25 data 9.1522 (edge 0.0000) smooth 0.9011
  it 50 data 7.9651 (edge 0.0000) smooth 0.2339
  it 75 data 7.3505 (edge 0.0000) smooth 0.1481
  it 100 data 6.9931 (edge 0.0000) smooth 0.1241
  it 125 data 6.8189 (edge 0.0000) smooth 0.1135
  it 149 data 6.7376 (edge 0.0000) smooth 0.1089
stage 1 done 77.1s
  it 0 data 22.1679 (edge 15.4336) smooth 0.1087
  it 25 data 21.0974 (edge 14.1179) smooth 0.1412
  it 50 data 20.5268 (edge 13.9047) smooth 0.0909
  it 75 data 20.2449 (edge 13.7956)  …[+83 chars]
```
</details>

**Bash**
```
sleep 25; tail -3 /work/scratch/holdout.log; cat /work/scratch/holdout/object_poses_world_diag.json | head -60
```

<details><summary>output</summary>

```
stage 2 done 125.8s
depth cues (cm, + = farther): {'floor_cm': -10.585814216738951, 'hand_cm': 1.3877787807814457e-14} silhouette: 4.97 applied: 0.0
saved /work/scratch/holdout/object_poses_world.npz  total 143.8s
{
 "phases": [
  [
   "static",
   0,
   158
  ],
  [
   "moving",
   159,
   345
  ],
  [
   "static",
   346,
   365
  ],
  [
   "moving",
   366,
   421
  ],
  [
   "static",
   422,
   448
  ],
  [
   "moving",
   449,
   582
  ],
  [
   "static",
   583,
   713
  ],
  [
   "moving",
   714,
   937
  ],
  [
   "static",
   938,
   1002
  ]
 ],
 "runs": [
  [
   0,
   158
  ],
  [
   346,
   365
  ],
  [
   422,
   448
```
</details>

**Bash**
```
cd /work/scratch && nohup python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > /work/outputs/solve_run.log 2>&1 &
sleep 5; echo started
```

<details><summary>output</summary>

```
started
```
</details>

**Bash**
```
python -c "import nvdiffrast, numpy, matplotlib, torch; print(nvdiffrast.__version__, numpy.__version__, matplotlib.__version__, torch.version.cuda)"; python --version
```

<details><summary>output</summary>

```
0.4.0 1.26.3 3.11.2 12.4
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
sleep 100; cat /work/outputs/solve_run.log | tail -5
```

<details><summary>output</summary>

```
  it 75 data 8.7161 (edge 0.0000) smooth 0.1520
  it 100 data 8.2890 (edge 0.0000) smooth 0.1290
  it 125 data 8.0793 (edge 0.0000) smooth 0.1195
  it 149 data 7.9760 (edge 0.0000) smooth 0.1144
stage 1 done 118.9s
```
</details>

**Bash**
```
sleep 90; tail -4 /work/outputs/solve_run.log
```

<details><summary>output</summary>

```
  it 99 data 24.6174 (edge 16.8787) smooth 0.0877
stage 2 done 167.4s
depth cues (cm, + = farther): {'floor_cm': -10.831471635044913, 'hand_cm': 1.3877787807814457e-14} silhouette: 5.38 applied: 0.0
saved /work/outputs/object_poses_world.npz  total 185.2s
```
</details>

**Bash**
```
python validate.py --inputs /work/inputs --result /work/outputs/object_poses_world.npz --holdout /work/scratch/holdout/object_poses_world.npz > /work/outputs/validation_table.md 2>/work/scratch/val.err; cat validation_table.md; tail -3 /work/scratch/val.err
```

<details><summary>output</summary>

```
| phase | frames | IoU base | IoU ours | held-out IoU | edge px base | edge px ours | held-out edge px | rot step deg base/ours | z-accel cm base/ours | static spread cm base/ours | floor h cm base/ours (std) | dz mean / median cm | diff of median z cm | ADD(base,ours) cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| static 0-158 | 159 | 0.943 | 0.956 | - | 3.19 | 2.13 | - | 0.44 / 0.02 | 0.94 / 0.00 | 1.80 / 0.00 | -1.4 (0.6) / -2.3 (0.0) | 0.07 / 0.04 | 0.04 | 2.47 |
| moving 159-345 | 187 | 0.926 | 0.963 | 0.962 | 2.91 | 2.18 | 2.22 | 0.81 / 0.42 | 1.30 / 0.04 | - | -0.4 (1.9) / 0.4 (1.7) | 0.39 / 0.30 | 0.11 | 3.19 |
| static 346-365 | 20 | 0.931 | 0.970 | - | 2.19 | 1.73 | - | 0.74 / 0.00 | 3.30 / 0.00 | 2.77 / 0.00 | 2.1 (1.3) / 1.4 (0.0) | -0.37 / -0.94 | -0.94 | …[+1280 chars]
```
</details>

> *thinking:* The final run (185s) shows our method outperforming the baseline on every image-space metric across all phases, even on held-out frames (IoU 0.926→0.953, edge error 2.50→2.04 px). Now I'll visually spot-check a fast-rotation frame and the final static phase.

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from cmp_contours import *
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
To=cam_from_world(d['T_wc'],np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'])
for i in [520,960,300]:
    draw_edges(i,[Tc[i],To[i]],[(0,0,255),(255,128,0)],f'/work/scratch/fin_{i}.jpg')
a=cv2.imread('/work/scratch/fin_520.jpg'); b=cv2.imread('/work/scratch/fin_960.jpg'); c=cv2.imread('/work/scratch/fin_300.jpg')
h=500; f=lambda x: cv2.resize(x,(int(x.shape[1]*h/x.shape[0]),h))
cv2.imwrite('/work/scratch/fin.jpg',np.hstack([f(a),f(b),f(c)]))
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/fin.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > flipcheck.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, trimesh, h5py, json
from validate import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
To=cam_from_world(d['T_wc'],np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'])
f5=h5py.File('/work/inputs/masks.h5'); 
gray=np.load('frames_full_gray.npy',mmap_mode='r')
Sobb,_=trimesh.bounds.oriented_bounds(mesh); Sinv=np.linalg.inv(Sobb); Robb=Sinv[:3,:3]; c=Sinv[:3,3]
fn_lab=torch.tensor(np.argmax(np.asarray(mesh.face_normals)@np.concatenate([Robb,-Robb],1),1)+1,device='cuda')
idx=list(range(4,1003,20)); OM=f5['object'][idx]; PM=f5['person'][idx]
PMfull={i:f5['person'][i] for i in idx}
def flip(T,ax):
    # 180 deg about OBB axis ax through the OBB centre (mesh frame)
    Fm=np.eye(4); Rf=Robb@np.diag([1 if k==ax else -1 for k in range(3)])@Robb.T
    Fm[:3,:3]=Rf; Fm[:3,3]=c-Rf@c
    return T@Fm
res={}
for name,ax in [('ours',None),('flipA',0),('flipB',1),('flipC',2)]:
    T=To[idx] if ax is None else flip(To[idx],ax)
    io=mask_iou(R,d,T,OM,PM)
    class G: pass
    ed=edge_chamfer(R,fn_lab,d,np.concatenate([T]),list(range(len(idx))),[gray[i] for i in idx],[PMfull[i] for i in idx])
    res[name]=(io,ed)
phases=json.load(open('/work/outputs/object_poses_world_diag.json'))['phases']
for k,a,b in phases:
    sel=[j for j,i in enumerate(idx) if a<=i<=b]
    print(k,a,b,' '.join(f"{n}: IoU {res[n][0][sel].mean():.3f} edge { …[+93 chars]
```

<details><summary>output</summary>

```
static 0 158 ours: IoU 0.956 edge 2.10 | flipA: IoU 0.855 edge 2.82 | flipB: IoU 0.861 edge 4.13 | flipC: IoU 0.923 edge 4.31 |
moving 159 345 ours: IoU 0.964 edge 2.10 | flipA: IoU 0.909 edge 2.74 | flipB: IoU 0.893 edge 4.44 | flipC: IoU 0.938 edge 4.02 |
static 346 365 ours: IoU 0.974 edge 1.70 | flipA: IoU 0.915 edge 2.41 | flipB: IoU 0.886 edge 4.20 | flipC: IoU 0.944 edge 3.00 |
moving 366 421 ours: IoU 0.970 edge 1.77 | flipA: IoU 0.908 edge 2.29 | flipB: IoU 0.877 edge 4.32 | flipC: IoU 0.939 edge 3.00 |
static 422 448 ours: IoU 0.967 edge 2.07 | flipA: IoU 0.875 edge 2.66 | flipB: IoU 0.839 edge 4.29 | flipC: IoU 0.918 edge 2.95 |
moving 449 582 ours: IoU 0.952 edge 2.13 | flipA: IoU 0.880 edge 2.80 | flipB: IoU 0.867 edge 3.78 | flipC: IoU 0.919 edge 2.92 |
static 583 713 ours: I …[+368 chars]
```
</details>

**Bash**
```
cd /work/outputs; python -c "
import json; d=json.load(open('object_poses_world_diag.json'))
for k,v in d['cues_per_phase'].items(): print(k, {a:(round(b,1) if b is not None else None) for a,b in v.items()})
print(d['cues_all'], d['silhouette_scale_cue_cm'], d['applied_shift_cm'], d['runtime_s'], d['iou_crop_opt_mean'])
"; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from solve import *
d=load_inputs('/work/inputs'); Tb=cam_from_world(d['T_wc'],d['rec']['T_world_mesh'].astype(np.float64))
z=np.load('/work/outputs/object_poses_world_diag.npz'); Topt=z['Tc_opt']
import json; ph=json.load(open('/work/outputs/object_poses_world_diag.json'))['phases']
for k,a,b in ph:
    print(k,a,b,'silhouette-only depth vs base (median per-frame, cm):', round(np.median(Topt[a:b+1,2,3]-Tb[a:b+1,2,3])*100,1))
o=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(o.shape,o.dtype,np.isfinite(o).all(), np.abs(np.linalg.det(o[:,:3,:3])-1).max())
EOF
ls -la /work/outputs
```

<details><summary>output</summary>

```
static 0-158 {'floor_cm': -19.9, 'hand_cm': -4.5}
moving 159-345 {'floor_cm': -11.7, 'hand_cm': -3.0}
static 346-365 {'floor_cm': -6.9, 'hand_cm': -8.5}
moving 366-421 {'floor_cm': -5.4, 'hand_cm': -3.0}
static 422-448 {'floor_cm': -5.7, 'hand_cm': 5.0}
moving 449-582 {'floor_cm': -6.6, 'hand_cm': 2.0}
static 583-713 {'floor_cm': -9.2, 'hand_cm': 3.0}
moving 714-937 {'floor_cm': -13.3, 'hand_cm': 3.0}
static 938-1002 {'floor_cm': -16.8, 'hand_cm': 9.0}
{'floor_cm': -10.831471635044913, 'hand_cm': 1.3877787807814457e-14} 5.384232855147925 0.0 185.1553282737732 0.9657880663871765
static 0 158 silhouette-only depth vs base (median per-frame, cm): 4.1
moving 159 345 silhouette-only depth vs base (median per-frame, cm): 3.7
static 346 365 silhouette-only depth vs base (median per-frame, cm): 1. …[+1320 chars]
```
</details>

**Bash**
```
cd /work/outputs; sed -i '/^FLOOR_EXPECTED = None/d' solve.py; grep -n "FLOOR_EXPECTED\|rot_angle" solve.py; python -c "import ast;ast.parse(open('/work/outputs/solve.py').read())"; rm -rf __pycache__
```

<details><summary>output</summary>

```
30:from hoi_lib import Renderer, SilMesh, cam_from_world, load_inputs, load_mesh, rot_angle  # noqa: E402
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
cat > /work/scratch/.claude/projects/-work/memory/hoi-trajectory-task.md <<'EOF'
---
name: hoi-trajectory-task
description: Object 6-DoF trajectory recovery task (AGENT_TASK.md, CARI4D baseline) — what worked on episode 5
metadata:
  type: project
---
Episode 5 (big_black_box) solved 2026-10-21; outputs in /work/outputs (solve.py, traj_opt.py, hoi_lib.py, validate.py).
Baseline flaws found: 2-4 cm depth jitter along the ray (even in static phases), ~1-3 deg yaw; no flips.
What worked: joint nvdiffrast silhouette + crease-edge chamfer optimisation over all frames with static-phase node
sharing, acceleration + mask-driven velocity penalty, then smooth depth re-anchoring to baseline z.
**Why:** held-out-frame IoU/edge metrics improved in every phase.
**How to apply:** for similar episodes reuse this pipeline; watch the 45-min GPU budget (full run ~3 min, GPU is shared and timings vary).
EOF
echo "- [HOI trajectory task](hoi-trajectory-task.md) — episode-5 object-pose pipeline & lessons" >> /work/scratch/.claude/projects/-work/memory/MEMORY.md; ls /work/outputs
```

<details><summary>output</summary>

```
REPORT.md
TOOLS.md
hoi_lib.py
object_poses_world.npz
object_poses_world_diag.json
object_poses_world_diag.npz
solve.py
solve_run.log
traj_opt.py
validate.py
validation_table.md
```
</details>

**Agent:** I've written the full trajectory (1003 poses) to `/work/outputs/object_poses_world.npz`, along with `REPORT.md` and `TOOLS.md`. It comes from one run of `solve.py` (185 s, under 2.5 GB GPU memory), and it beats the baseline on every image check in every phase. One thing it can't fix: any constant depth error the baseline already has.

**What was wrong with the baseline**
- **No flips or jumps.** The trolley handle and castors land on the right sides, and flipping 180° about any of the box's three axes scores worse in every phase.
- **Depth jitter is the main error.** The distance from the camera wobbles 2–4 cm from frame to frame, even while the box is standing still.
- **Rotation is off by 1–3°**, mostly a turn about the vertical axis. The image position is accurate to about 1 px.

**What `solve.py` does**
1. **Finds the phases automatically from the masks:** five still phases (0–158, 346–365, 422–448, 583–713, 938–1002) and the moving stretches between them. I checked the boundaries by eye.
2. **Refines all frames in one optimisation.** It matches the rendered mesh outline to the object mask and the box's internal edges to edges detected in the video. Each still phase gets a single pose, and a smoothness penalty is applied over time, much stronger when the masks say the box isn't moving.
3. **Puts the depth back on the baseline's level.** Positions are scaled along the line of sight with a slowly varying factor, so the image position doesn't change. The median per-frame depth difference from the baseline is at most 1.75 cm in every phase, and the episode mean is −0.02 cm.
4. **Checks the depth against physical cues.** Floor contact suggests −10.8 cm, the human's hand contact 0 cm, and the object's on-screen size +5.4 cm. They disagree, so I applied no shift, as the task's rule requires.

**Validation (no ground truth)**

| | baseline | ours |
|---|---|---|
| Mask overlap (IoU), all frames | 0.932 | 0.954 |
| Mask overlap on 120 frames hidden from the optimiser (separate run) | 0.926 | 0.953 |
| Box-edge misalignment (px), all frames | 2.53 | 2.00 |
| Box-edge misalignment (px), hidden frames | 2.50 | 2.04 |
| Pose jitter within still phases (cm) | 0.65–3.3 | 0 |

The edge check isn't used by the outline term, and the hidden frames improve as much as the fitted ones, so this isn't just overfitting the masks. Rotation changed by 1.5° median (4.6° max). On average the new poses sit 3.2 cm from the baseline's.

**Confidence**
- **High** for the still phases and most of the motion: rotation, image position and smoothness.
- **Medium** for frames 449–582, where the box is spun quickly with the hands covering much of it. Depth there ended up 1.6 cm farther than the baseline on average, a side effect of the depth smoothing.
- **Phase 714–937:** a medians-only comparison makes it look 4.6 cm off the baseline. Frame by frame it's −1.0 cm; depth drops steeply in that phase, so comparing the two medians is misleading.

**Other notes**
- All my GPU work came to about 20 minutes, under the 45-minute limit.
- Reruns differ by about 1e-5 because GPU arithmetic isn't fully deterministic; seeds are fixed.
- No external tools or weights were downloaded. `solve.py` reads only `/work/inputs`.
- After the final run I deleted one unused constant from `solve.py`; this doesn't change the output.

Files are in `/work/outputs`:
- object_poses_world.npz
- REPORT.md
- TOOLS.md
- solve.py
- validation_table.md

## Final

I've written the full trajectory (1003 poses) to `/work/outputs/object_poses_world.npz`, along with `REPORT.md` and `TOOLS.md`. It comes from one run of `solve.py` (185 s, under 2.5 GB GPU memory), and it beats the baseline on every image check in every phase. One thing it can't fix: any constant depth error the baseline already has.

**What was wrong with the baseline**
- **No flips or jumps.** The trolley handle and castors land on the right sides, and flipping 180° about any of the box's three axes scores worse in every phase.
- **Depth jitter is the main error.** The distance from the camera wobbles 2–4 cm from frame to frame, even while the box is standing still.
- **Rotation is off by 1–3°**, mostly a turn about the vertical axis. The image position is accurate to about 1 px.

**What `solve.py` does**
1. **Finds the phases automatically from the masks:** five still phases (0–158, 346–365, 422–448, 583–713, 938–1002) and the moving stretches between them. I checked the boundaries by eye.
2. **Refines all frames in one optimisation.** It matches the rendered mesh outline to the object mask and the box's internal edges to edges detected in the video. Each still phase gets a single pose, and a smoothness penalty is applied over time, much stronger when the masks say the box isn't moving.
3. **Puts the depth back on the baseline's level.** Positions are scaled along the line of sight with a slowly varying factor, so the image position doesn't change. The median per-frame depth difference from the baseline is at most 1.75 cm in every phase, and the episode mean is −0.02 cm.
4. **Checks the depth against physical cues.** Floor contact suggests −10.8 cm, the human's hand contact 0 cm, and the object's on-screen size +5.4 cm. They disagree, so I applied no shift, as the task's rule requires.

**Validation (no ground truth)**

| | baseline | ours |
|---|---|---|
| Mask overlap (IoU), all frames | 0.932 | 0.954 |
| Mask overlap on 120 frames hidden from the optimiser (separate run) | 0.926 | 0.953 |
| Box-edge misalignment (px), all frames | 2.53 | 2.00 |
| Box-edge misalignment (px), hidden frames | 2.50 | 2.04 |
| Pose jitter within still phases (cm) | 0.65–3.3 | 0 |

The edge check isn't used by the outline term, and the hidden frames improve as much as the fitted ones, so this isn't just overfitting the masks. Rotation changed by 1.5° median (4.6° max). On average the new poses sit 3.2 cm from the baseline's.

**Confidence**
- **High** for the still phases and most of the motion: rotation, image position and smoothness.
- **Medium** for frames 449–582, where the box is spun quickly with the hands covering much of it. Depth there ended up 1.6 cm farther than the baseline on average, a side effect of the depth smoothing.
- **Phase 714–937:** a medians-only comparison makes it look 4.6 cm off the baseline. Frame by frame it's −1.0 cm; depth drops steeply in that phase, so comparing the two medians is misleading.

**Other notes**
- All my GPU work came to about 20 minutes, under the 45-minute limit.
- Reruns differ by about 1e-5 because GPU arithmetic isn't fully deterministic; seeds are fixed.
- No external tools or weights were downloaded. `solve.py` reads only `/work/inputs`.
- After the final run I deleted one unused constant from `solve.py`; this doesn't change the output.

Files are in `/work/outputs`:
- object_poses_world.npz
- REPORT.md
- TOOLS.md
- solve.py
- validation_table.md
