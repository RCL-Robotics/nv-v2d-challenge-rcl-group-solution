# CL2b_ep012

Claude Opus v2（ep11-21）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T09:57:42Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude/projects/-work/memory/ 2>/dev/null && cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null; nvidia-smi; python -V
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
total 80092
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 13:51 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 13:51 ..
-rw-rw-r-- 1 hanzhou hanzhou      817 Oct 21 13:51 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       69 Oct 21 13:51 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   191490 Oct 21 13:51 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      400 Oct 21 13:51 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  5403936 Oct 21 13:51 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 22048360 Oct 21 13:51 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 3560 …[+4338 chars]
```
</details>

**Bash**
```
cd /work; cat logs/run_manifest.txt; echo; echo CVD=$CUDA_VISIBLE_DEVICES; ls /pipeline /pipeline/*; ls /code/reconstruction; nproc; free -g
```

<details><summary>output</summary>

```
date_utc: 2026-10-21T09:57:42Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2b_ep012:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000012:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
0c5f097c9aee67d5ac2d6b29bd1f8690ec1f80fdffb16b91b2fa76b9e54299c7  ./camera.json
8bac93d034f96070cd8a7409b6e1b40b2787eb276bcf1108803c846e1c292be7  ./episode.json
1acaad99f6890cf8436ca5da3ae64761881ad10a2 …[+1346 chars]
```
</details>

**Bash**
```
cd /work; cat /pipeline/pipeline_report.json | head -80; ls -R /pipeline/depth | head; ls /pipeline/export/episode_000012 /pipeline/foundationpose/debug | head -30
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
    "mtime_ns": 1792066136957042770,
    "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000012/inference/episode_000012_step200000_before_after.mp4",
    "size": 48393566
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.0 …[+2400 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -I - <<'EOF'
import numpy as np, trimesh, h5py
r=np.load('/work/inputs/reconstruction.npz')
for k in r.files: print(k, r[k].shape, r[k].dtype)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m, m.bounds, m.extents)
print(type(m.visual), getattr(m.visual,'material',None))
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
with h5py.File('/work/inputs/masks.h5') as f:
    for k in f: print(k,f[k].shape,f[k].dtype)
    om=f['object'][:].reshape(700,-1).sum(1); pm=f['person'][:].reshape(700,-1).sum(1)
np.set_printoptions(linewidth=200)
print(om[::10]); print(pm[::10])
print(r['contact_prob_right_left'][::10].round(2).T)
EOF
```

<details><summary>output</summary>

```
T_world_mesh (700, 4, 4) float32
T_cam_mesh_foundationpose (700, 4, 4) float32
T_cam_mesh_coconet (700, 4, 4) float32
human_vertices_world (700, 4610, 3) float32
human_keypoints_world (700, 70, 3) float32
contact_prob_right_left (700, 2) float32
<trimesh.Trimesh(vertices.shape=(285889, 3), faces.shape=(95307, 3))> [[-0.29098165 -0.09970255 -0.39503556]
 [ 0.31388506  0.0434465   0.41024145]] [0.60486671 0.14314905 0.80527702]
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x73e03092aa10>
[[-6.12323400e-17 -1.00000000e+00 -6.12323400e-17 -2.81280279e-02]
 [ 1.00000000e+00 -6.12323400e-17  0.00000000e+00 -1.14517063e-02]
 [-3.74939946e-33 -6.12323400e-17  1.00000000e+00 -7.60294497e-03]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000 …[+1659 chars]
```
</details>

**Bash**
```
mkdir -p /work/scratch/frames /work/outputs /work/tools; cd /work/scratch; python -I - <<'EOF'
import cv2, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames=[]
i=0
while True:
    ok,f=cap.read()
    if not ok: break
    frames.append(cv2.resize(f,(768,576)))
    i+=1
print(i)
np.save('/work/scratch/frames_half.npy', np.stack(frames))
for j in [0,150,170,200,250,300,350,400,450,500,550,600,620,699]:
    cv2.imwrite(f'/work/scratch/frames/{j:04d}.jpg', frames[j])
EOF
```

<details><summary>output</summary>

```
30.0 700.0
700
```
</details>

**Bash**
```
cd /work/scratch; python -I -c "
import cv2,numpy as np
F=np.load('frames_half.npy')
ids=[0,170,250,350,450,550,620,699]
im=[cv2.putText(F[i].copy(),str(i),(20,40),0,1.2,(0,0,255),3) for i in ids]
g=np.vstack([np.hstack(im[k:k+4]) for k in (0,4)])
cv2.imwrite('grid.jpg',cv2.resize(g,None,fx=0.6,fy=0.6))
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
cd /work/scratch; python -I - <<'EOF'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.vertices; print(v.mean(0), np.median(v,0))
mat=m.visual.material; print(mat.baseColorTexture.size if mat.baseColorTexture else None, mat.baseColorFactor)
uv=m.visual.uv; print(uv.shape, uv.min(0), uv.max(0))
# PCA axes
c=v-v.mean(0); w,U=np.linalg.eigh(c.T@c/len(c)); print(w,U)
# histogram along y (thin axis)
h,e=np.histogram(v[:,1],20); print(h, e.round(3))
import torch, nvdiffrast.torch as dr
print(torch.__version__, torch.cuda.is_available())
glc=dr.RasterizeCudaContext(); print('ok')
EOF
```

<details><summary>output</summary>

```
[-0.01901435  0.00309212 -0.01205958] [-0.0478005   0.00977107 -0.0234835 ]
(5007, 4818) None
(285889, 2) [0.000699   0.00072598] [0.99930102 0.99927402]
[0.00080438 0.0323732  0.06391555] [[-0.01491707 -0.99286265  0.11832687]
 [-0.99987335  0.01546844  0.00374261]
 [-0.00554623 -0.11825606 -0.99296765]]
[ 6573  1925  1868  2048  2607  2615  3713  3803  4194  4573  4850  5468
  7967 43616 36324 33852 41395 59612 17213  1673] [-0.1   -0.093 -0.085 -0.078 -0.071 -0.064 -0.057 -0.05  -0.042 -0.035
 -0.028 -0.021 -0.014 -0.007  0.001  0.008  0.015  0.022  0.029  0.036
  0.043]
2.5.1+cu124 True
ok
```
</details>

**Write**
```
/work/scratch/lib.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/lib.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
R=Renderer(m,tex,768,576); Kh=K.copy(); Kh[:2]*=0.5
F=np.load('frames_half.npy')
ids=[0,170,250,350,450,550,620,699]
out=R.render(Kh,Tc[ids])
om=load_masks('object')[ids][:,::2,::2]
ims=[]
for j,i in enumerate(ids):
    im=F[i].copy()
    msk=out['mask'][j].cpu().numpy()
    rgb=(out['rgb'][j].cpu().numpy()[...,::-1]*255).astype(np.uint8)
    im[msk]=(0.5*im[msk]+0.5*rgb[msk]).astype(np.uint8)
    cnt,_=cv2.findContours(om[j].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im,cnt,-1,(0,255,0),1)
    cnt,_=cv2.findContours(msk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im,cnt,-1,(0,0,255),1)
    ys,xs=np.nonzero(om[j]); cy,cx=int(ys.mean()),int(xs.mean())
    crop=im[max(cy-100,0):cy+100, max(cx-130,0):cx+130]
    crop=cv2.resize(crop,(520,400)); cv2.putText(crop,str(i),(10,30),0,1,(0,0,255),2)
    ims.append(crop)
g=np.vstack([np.hstack(ims[k:k+4]) for k in (0,4)])
cv2.imwrite('base_ov.jpg',g)
print(Tc[ids,:3,3])
EOF
```

<details><summary>output</summary>

```
[[ 0.18185863  1.05008663  3.53148439]
 [ 0.1666872   1.07214855  3.64244515]
 [ 0.26642769  0.98780006  3.74780602]
 [ 0.13471682  0.93253241  3.75294141]
 [-0.02608971  0.86079363  3.52811164]
 [-0.25160037  1.02938764  3.49634522]
 [-0.27423813  1.11502706  3.51604585]
 [-0.27502659  1.11739729  3.49394858]]
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
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tfp=r['T_cam_mesh_foundationpose'].astype(np.float64); Tcc=r['T_cam_mesh_coconet'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
np.set_printoptions(precision=3,suppress=True,linewidth=200)
dR=rotdist(Tw[1:,:3,:3],Tw[:-1,:3,:3]); dt=np.linalg.norm(Tw[1:,:3,3]-Tw[:-1,:3,3],axis=1)
print('frame  yaxis_world  xaxis_world  t_world  dR dt  rot(base,fp) rot(base,cc) z base fp cc')
for i in range(0,700,10):
    print(i, Tw[i,:3,1], Tw[i,:3,0], Tw[i,:3,3], round(dR[min(i,698)],2), round(dt[min(i,698)]*100,2), rotdist(Tc[i,:3,:3],Tfp[i,:3,:3]).round(1), rotdist(Tc[i,:3,:3],Tcc[i,:3,:3]).round(1), Tc[i,2,3].round(3),Tfp[i,2,3].round(3),Tcc[i,2,3].round(3))
big=np.nonzero(dR>5)[0]; print('big jumps',big, dR[big].round(1))
EOF
```

<details><summary>output</summary>

```
frame  yaxis_world  xaxis_world  t_world  dR dt  rot(base,fp) rot(base,cc) z base fp cc
0 [-0.044 -0.999 -0.017] [-0.999  0.044  0.016] [-0.169  1.128  2.737] 0.23 1.55 2.0 1.3 3.531 3.762 3.593
10 [-0.046 -0.999 -0.014] [-0.999  0.046  0.014] [-0.175  1.143  2.685] 0.58 1.34 2.1 1.3 3.585 3.862 3.636
20 [-0.045 -0.999 -0.013] [-0.999  0.045  0.018] [-0.174  1.152  2.668] 0.24 3.22 1.8 1.3 3.603 3.799 3.622
30 [-0.046 -0.999 -0.021] [-0.999  0.045  0.018] [-0.179  1.151  2.685] 0.29 0.37 2.0 1.3 3.586 3.746 3.59
40 [-0.046 -0.999 -0.015] [-0.999  0.046  0.01 ] [-0.175  1.148  2.684] 0.27 0.52 1.8 1.3 3.587 3.761 3.598
50 [-0.04  -0.999 -0.014] [-0.999  0.04   0.011] [-0.174  1.142  2.706] 0.23 1.0 1.9 1.3 3.564 3.754 3.6
60 [-0.044 -0.999 -0.014] [-0.999  0.044  0.017] [-0.175  1.15   2.67 …[+6732 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
om=load_masks('object'); pm=load_masks('person')
np.save('om_half.npy',om[:,::2,::2]); np.save('pm_half.npy',pm[:,::2,::2])
def iou(a,b): return (a&b).sum()/max((a|b).sum(),1)
# visible-only IoU, ignoring person pixels
ref0=om[0]; refL=om[-1]
r=np.load(INP+'/reconstruction.npz'); c=r['contact_prob_right_left']
for i in range(0,700,5):
    ys,xs=np.nonzero(om[i])
    print(i, om[i].sum(), round(iou(om[i],ref0),3), round(iou(om[i],refL),3), round(iou(om[i],om[i-5]) if i>=5 else 1,3), xs.mean().round(1), ys.mean().round(1), c[i].round(2), (pm[i]&np.roll(om[i],0)).sum())
EOF
```

<details><summary>output</summary>

```
0 12265 1.0 0.095 1 801.0 811.0 [0. 0.] 0
5 12155 0.987 0.096 0.987 800.6 811.1 [0. 0.] 0
10 12144 0.987 0.095 0.998 800.7 811.0 [0. 0.] 0
15 12229 0.995 0.094 0.989 801.0 810.9 [0. 0.] 0
20 12239 0.995 0.095 0.998 801.0 810.9 [0. 0.] 0
25 12229 0.994 0.095 0.998 801.0 810.9 [0. 0.] 0
30 12248 0.995 0.095 0.998 801.0 810.9 [0. 0.] 0
35 12250 0.995 0.094 0.998 801.0 810.9 [0. 0.] 0
40 12260 0.995 0.095 0.998 801.0 810.9 [0. 0.] 0
45 12252 0.996 0.095 0.997 801.0 810.9 [0. 0.] 0
50 12134 0.985 0.096 0.985 800.5 811.1 [0. 0.] 0
55 12155 0.986 0.096 0.996 800.5 811.0 [0. 0.] 0
60 12034 0.98 0.093 0.988 800.9 810.8 [0. 0.] 0
65 12216 0.982 0.095 0.985 800.3 810.5 [0. 0.] 20
70 12243 0.991 0.095 0.988 800.8 810.8 [0. 0.] 75
75 12210 0.99 0.095 0.996 800.9 810.9 [0. 0.] 76
80 11959 0.974 0.094 0. …[+6226 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2
F=np.load('frames_half.npy').astype(np.float32); om=np.load('om_half.npy'); pm=np.load('pm_half.npy')
def diffs(ref):
    reg=cv2.dilate(om[ref].astype(np.uint8),np.ones((9,9)))>0
    out=[]
    for i in range(700):
        v=reg&~cv2.dilate(pm[i].astype(np.uint8),np.ones((7,7))).astype(bool)
        out.append((np.abs(F[i]-F[ref]).mean(-1)[v].mean(), v.sum()/reg.sum()))
    return np.array(out)
d0=diffs(0); dL=diffs(699)
for i in list(range(150,230,2))+list(range(570,640,2)): print(i, d0[i].round(2), dL[i].round(2))
EOF
```

<details><summary>output</summary>

```
150 [4.96 0.94] [43.61  1.  ]
152 [5.15 0.94] [43.64  1.  ]
154 [5.09 0.94] [43.69  1.  ]
156 [5.25 0.94] [43.7  1. ]
158 [5.14 0.94] [43.63  1.  ]
160 [5.25 0.93] [43.7  1. ]
162 [5.16 0.91] [43.66  1.  ]
164 [5.17 0.9 ] [43.74  1.  ]
166 [5.19 0.88] [43.65  1.  ]
168 [5.32 0.88] [43.62  1.  ]
170 [5.13 0.86] [43.56  0.99]
172 [5.09 0.85] [43.39  0.99]
174 [5.02 0.85] [43.3   0.99]
176 [5.09 0.85] [43.37  0.99]
178 [5.04 0.84] [43.41  0.99]
180 [4.98 0.83] [43.34  0.99]
182 [5.08 0.85] [43.36  0.99]
184 [5.   0.83] [43.33  0.99]
186 [4.9  0.83] [43.32  0.99]
188 [4.86 0.83] [43.32  0.99]
190 [4.89 0.83] [43.37  0.99]
192 [4.95 0.83] [43.34  0.99]
194 [5.03 0.83] [43.29  0.99]
196 [5.45 0.84] [43.21  0.99]
198 [6.1  0.84] [43.3   0.99]
200 [7.79 0.84] [43.26  0.99]
202 [9.87 0.85] [43.33   …[+1475 chars]
```
</details>

> *thinking:* The pallet stays still, gets flipped on edge, then settles legs-up; silhouettes and rotations track well, but translation jitters by several cm even during static segments. Next I'll check for the 180° yaw ambiguity, which silhouette matching alone can't resolve.

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
# symmetric center: bbox center
c=(m.bounds[0]+m.bounds[1])/2
def flip(T,axis):
    R=np.eye(4); R[:3,:3]=cv2.Rodrigues(np.eye(3)[axis]*np.pi)[0]
    C=np.eye(4); C[:3,3]=c; Ci=np.linalg.inv(C)
    return T@C@R@Ci
R2=Renderer(m,tex,1536,1152)
F=[]
cap=cv2.VideoCapture(INP+'/video.mp4')
for i in range(700):
    ok,f=cap.read()
    if i in (0,100,650,699): F.append(f)
rows=[]
for fi,i in zip([0,3],[0,699]):
    T=Tc[i]; hyps=[T,flip(T,1),flip(T,0),flip(T,2)]
    out=R2.render(K,np.stack(hyps))
    ys,xs=np.nonzero(out['mask'][0].cpu().numpy()); y0,y1,x0,x1=ys.min()-20,ys.max()+20,xs.min()-20,xs.max()+20
    crops=[F[fi][y0:y1,x0:x1]]
    for h in range(4):
        rgb=(out['rgb'][h].cpu().numpy()[...,::-1]*255).astype(np.uint8)
        rgb=np.clip(rgb.astype(float)*2.5,0,255).astype(np.uint8)
        crops.append(rgb[y0:y1,x0:x1])
    crops[0]=np.clip(crops[0].astype(float)*2.5,0,255).astype(np.uint8)
    rows.append(np.hstack(crops))
w=max(r.shape[1] for r in rows)
rows=[np.pad(r,((0,0),(0,w-r.shape[1]),(0,0))) for r in rows]
cv2.imwrite('flip_cmp.jpg',np.vstack(rows))
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/flip_cmp.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
c=(m.bounds[0]+m.bounds[1])/2
def flip(T,axis):
    R=np.eye(4); R[:3,:3]=cv2.Rodrigues(np.eye(3)[axis]*np.pi)[0]
    C=np.eye(4); C[:3,3]=c
    return T@C@R@np.linalg.inv(C)
R2=Renderer(m,tex,1536,1152)
cap=cv2.VideoCapture(INP+'/video.mp4'); F={}
for i in range(700):
    ok,f=cap.read()
    if i in (0,699): F[i]=f
def norm(g,msk):
    g=g.astype(float); v=g[msk]; return np.clip((g-v.mean())/v.std()*50+128,0,255).astype(np.uint8)
for i in [0,699]:
    out=R2.render(K,np.stack([Tc[i],flip(Tc[i],1)]))
    msk=out['mask'][0].cpu().numpy()
    ys,xs=np.nonzero(msk); y0,y1,x0,x1=ys.min()-5,ys.max()+5,xs.min()-5,xs.max()+5
    g=cv2.cvtColor(F[i],cv2.COLOR_BGR2GRAY); 
    crops=[norm(g,msk)[y0:y1,x0:x1]]
    for h in range(2):
        rg=(out['rgb'][h].cpu().numpy().mean(-1)*255); mk=out['mask'][h].cpu().numpy()
        crops.append(norm(rg,mk)[y0:y1,x0:x1]*mk[y0:y1,x0:x1])
    im=np.hstack(crops); im=cv2.resize(im,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'flipzoom_{i}.jpg',im)
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/flipzoom_0.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/flipzoom_699.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import trimesh, numpy as np, open3d as o3d
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
P=m.sample(200000, seed=0) if 'seed' in trimesh.sample.sample_surface.__code__.co_varnames else m.sample(200000)
c=(m.bounds[0]+m.bounds[1])/2
def flipR(axis):
    R=np.diag([1.,1,1]); 
    for a in range(3):
        if a!=axis: R[a,a]=-1
    return R
src=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
for ax in range(3):
    Q=(P-c)@flipR(ax).T+c
    tgt=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q))
    reg=o3d.pipelines.registration.registration_icp(tgt,src,0.03,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
    tq=np.asarray(tgt.transform(reg.transformation).points)
    d=np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tq)).compute_point_cloud_distance(src))
    print(ax, 'fitness',reg.fitness,'rmse',reg.inlier_rmse, 'dist mean/90/99', d.mean(), np.percentile(d,[90,99]), '\n',reg.transformation.round(4))
    np.save(f'symdist_{ax}.npy', np.c_[P, np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector((P-c)@flipR(ax).T+c)).compute_point_cloud_distance(src))])
EOF
```

<details><summary>output</summary>

```
0 fitness 0.605925 rmse 0.015953529529560837 dist mean/90/99 0.02631318248556008 [0.05406684 0.07156634] 
 [[ 9.998e-01 -1.840e-02  1.230e-02 -2.000e-04]
 [ 1.900e-02  9.983e-01 -5.570e-02  2.920e-02]
 [-1.120e-02  5.590e-02  9.984e-01 -3.100e-03]
 [ 0.000e+00  0.000e+00  0.000e+00  1.000e+00]]
1 fitness 0.99617 rmse 0.005735254353368628 dist mean/90/99 0.004252400240675412 [0.00864324 0.02377006] 
 [[ 9.998e-01  1.840e-02 -9.000e-04  7.400e-03]
 [-1.840e-02  9.998e-01 -1.160e-02  3.000e-04]
 [ 7.000e-04  1.160e-02  9.999e-01 -5.500e-03]
 [ 0.000e+00  0.000e+00  0.000e+00  1.000e+00]]
2 fitness 0.470815 rmse 0.015153195244195266 dist mean/90/99 0.033197213427497124 [0.06374748 0.07700159] 
 [[ 0.9999 -0.0029  0.013   0.0065]
 [ 0.0034  0.9993 -0.0379  0.012 ]
 [-0.0128  0.038   0.9992  0.0 …[+40 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import trimesh, numpy as np, open3d as o3d
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
P=m.sample(200000)
c=(m.bounds[0]+m.bounds[1])/2
R=np.diag([-1.,1,-1])
src=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
Q=(P-c)@R.T+c
tgt=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q))
reg=o3d.pipelines.registration.registration_icp(tgt,src,0.02,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
reg=o3d.pipelines.registration.registration_icp(tgt,src,0.01,reg.transformation,o3d.pipelines.registration.TransformationEstimationPointToPoint())
Tsym=reg.transformation.copy()
tgt.transform(Tsym)
# distances of original points to flipped cloud: where the original has material not present in flipped
d=np.asarray(src.compute_point_cloud_distance(tgt))
print(np.percentile(d,[50,90,95,99,99.9]))
sel=d>0.012
print('n asym', sel.sum())
import collections
Ps=P[sel]
# cluster
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Ps)); lab=np.array(pc.cluster_dbscan(0.015,30))
for l in range(lab.max()+1):
    q=Ps[lab==l]; print(l, len(q), q.mean(0).round(3), (q.max(0)-q.min(0)).round(3), d[sel][lab==l].mean().round(3))
# full symmetry transform in mesh coords: x -> Tsym @ (C R C^-1)
C=np.eye(4); C[:3,3]=c; Rf=np.eye(4); Rf[:3,:3]=R
S=Tsym@C@Rf@np.linalg.inv(C); np.save('Tsym_y.npy',S); print(S.round(4))
EOF
```

<details><summary>output</summary>

```
[0.00275913 0.00869512 0.01293174 0.02399532 0.03649156]
n asym 11550
0 374 [-0.028  0.004  0.155] [0.08  0.022 0.059] 0.017
1 971 [-0.172 -0.031 -0.334] [0.113 0.041 0.053] 0.022
2 646 [0.1   0.005 0.214] [0.041 0.031 0.122] 0.019
3 189 [ 0.252  0.006 -0.154] [0.04  0.023 0.026] 0.016
4 1248 [-0.152  0.01   0.01 ] [0.126 0.052 0.074] 0.019
5 664 [ 0.198 -0.01  -0.347] [0.114 0.055 0.063] 0.016
6 548 [-0.204  0.015  0.246] [0.087 0.036 0.051] 0.019
7 710 [-0.169 -0.004 -0.269] [0.103 0.059 0.032] 0.017
8 132 [ 0.191  0.011 -0.211] [0.04  0.017 0.035] 0.014
9 1167 [0.208 0.023 0.335] [0.134 0.021 0.063] 0.024
10 351 [-0.126  0.01   0.163] [0.041 0.034 0.051] 0.018
11 633 [0.215 0.039 0.002] [0.119 0.015 0.057] 0.015
12 45 [-0.031 -0.002  0.351] [0.018 0.009 0.015] 0.015
13 1703 [-0.156  0.0 …[+1150 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -I -c "
import sys; sys.path.insert(0,'/work/scratch')
import trimesh,cv2,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
t=np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
print(t.mean((0,1)), np.percentile(t.mean(-1),[5,50,95,99]))
cv2.imwrite('tex.jpg',cv2.resize(t[...,::-1],(1000,962)))
"
```

<details><summary>output</summary>

```
[22.01243643 24.83662234 24.37050197] [ 0.          0.33333333 79.         99.33333333]
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
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2, h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
A=[];B=[];PA=[];PB=[]
with h5py.File('/work/inputs/masks.h5') as f:
    pm=f['person']
    for i in range(700):
        ok,fr=cap.read()
        if i<=190 and i%3==0: A.append(fr); PA.append(pm[i])
        if i>=585 and i%3==0: B.append(fr); PB.append(pm[i])
def med(I,P):
    I=np.stack(I).astype(np.float32); P=np.stack(P)
    P=np.stack([cv2.dilate(p.astype(np.uint8),np.ones((15,15))) for p in P]).astype(bool)
    I[P]=np.nan
    return np.nanmedian(I,0)
mA=med(A,PA); mB=med(B,PB)
print(np.isnan(mA).sum(), np.isnan(mB).sum())
np.save('medA.npy',np.nan_to_num(mA)); np.save('medB.npy',np.nan_to_num(mB))
cv2.imwrite('medA.jpg',np.nan_to_num(mA).astype(np.uint8)[650:950,600:1000]); cv2.imwrite('medB.jpg',np.nan_to_num(mB).astype(np.uint8)[650:950,500:900])
EOF
```

<details><summary>output</summary>

```
<stdin>:14: RuntimeWarning: All-NaN slice encountered
0 22980
```
</details>

**Bash**
```
cd /work/scratch; python -I -c "
import cv2,numpy as np
a=cv2.imread('medA.jpg');b=cv2.imread('medB.jpg')
cv2.imwrite('med_both.jpg',cv2.resize(np.hstack([a,b]),None,fx=1.6,fy=1.6,interpolation=cv2.INTER_CUBIC))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/med_both.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2, h5py
mA=np.load('medA.npy'); g=cv2.cvtColor(mA.astype(np.uint8),cv2.COLOR_BGR2GRAY)
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][0]
ring=cv2.dilate(om.astype(np.uint8),np.ones((41,41)))>0
inner=cv2.erode(om.astype(np.uint8),np.ones((3,3)))>0
print('obj',np.percentile(g[inner],[5,25,50,75,95])); print('ring bg',np.percentile(g[ring&~om],[1,5,25,50,75]))
EOF
```

<details><summary>output</summary>

```
obj [ 4. 12. 19. 37. 53.]
ring bg [40. 50. 69. 79. 86.]
```
</details>

**Write**
```
/work/scratch/fit.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/fit.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/scratch; cat >> lib.py <<'EOF'


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    R = U @ Vt
    if np.linalg.det(R) < 0: R = U @ np.diag([1, 1, -1]) @ Vt
    return R
EOF
python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from fit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
S=np.load('Tsym_y.npy')
c=(m.bounds[0]+m.bounds[1])/2
def flip(axis):
    R=np.eye(4); R[:3,:3]=cv2.Rodrigues(np.eye(3)[axis]*np.pi)[0]; C=np.eye(4); C[:3,3]=c
    return C@R@np.linalg.inv(C)
Rr=Renderer(m,tex,1536,1152)
om=load_masks('object',[0,699])
for name,rng,img,omk in [('A',range(0,190),np.load('medA.npy'),om[0]),('B',range(590,700),np.load('medB.npy'),om[1])]:
    T0=np.eye(4); T0[:3,:3]=rot_mean(Tc[list(rng),:3,:3]); T0[:3,3]=np.median(Tc[list(rng),:3,3],0)
    p=dark_prob(img)
    roi=(cv2.dilate(omk.astype(np.uint8),np.ones((61,61)))>0).astype(np.float32)
    if name=='B':
        valid=(img.sum(-1)>0).astype(np.float32); roi*=valid
    for hn,H in [('base',np.eye(4)),('Sy',S),('flipx',flip(0)),('flipz',flip(2))]:
        T,l=fit_pose(Rr,K,T0@H,p,roi,iters=200)
        print(name,hn,'loss %.4f'%l,'drot %.2f'%rotdist(T[:3,:3],(T0@H)[:3,:3]),'dt',((T[:3,3]-T0[:3,3])*100).round(2), 'rot vs base %.1f'%rotdist(T[:3,:3],T0[:3,:3]))
EOF
```

<details><summary>output</summary>

```
A base loss 0.0591 drot 1.71 dt [-2.14 -2.86 -9.83] rot vs base 1.7
A Sy loss 0.0602 drot 1.95 dt [ 0.88 -2.42 -9.92] rot vs base 180.0
A flipx loss 0.1174 drot 7.13 dt [-2.01  1.18 -1.17] rot vs base 173.0
A flipz loss 0.1204 drot 6.32 dt [2.76 3.67 6.25] rot vs base 179.0
B base loss 0.0470 drot 0.86 dt [-0.09 -0.18 -0.96] rot vs base 0.9
B Sy loss 0.0479 drot 0.81 dt [-2.9  -0.92 -3.99] rot vs base 179.4
B flipx loss 0.1003 drot 5.97 dt [-1.45  4.54 14.7 ] rot vs base 176.9
B flipz loss 0.1009 drot 5.62 dt [-4.91  4.79 17.12] rot vs base 176.4
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from fit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
S=np.load('Tsym_y.npy')
Rr=Renderer(m,tex,1536,1152)
om=load_masks('object',[0,699])
res={}
for name,rng,img,omk in [('A',range(0,190),np.load('medA.npy'),om[0]),('B',range(590,700),np.load('medB.npy'),om[1])]:
    T0=np.eye(4); T0[:3,:3]=rot_mean(Tc[list(rng),:3,:3]); T0[:3,3]=np.median(Tc[list(rng),:3,3],0)
    p=dark_prob(img)
    roi=(cv2.dilate(omk.astype(np.uint8),np.ones((61,61)))>0).astype(np.float32)
    if name=='B': roi*=(img.sum(-1)>0)
    for thr in [30,38,46]:
      p=dark_prob(img,thr)
      out=[]
      for hn,H in [('base',np.eye(4)),('Sy',S)]:
        for fd in [False,True]:
            T,l=fit_pose(Rr,K,T0@H,p,roi,iters=200,fix_depth=fd); out.append(l)
            res[(name,hn,fd)]=T
      print(name,thr,'base/Sy free %.4f %.4f   fixdepth %.4f %.4f'%(out[0],out[2],out[1],out[3]))
    ys,xs=np.nonzero(omk); y0,y1,x0,x1=ys.min()-30,ys.max()+30,xs.min()-30,xs.max()+30
    a=Rr.render(K,np.stack([res[(name,'base',True)],res[(name,'Sy',True)]]),rgb=False)['mask'].cpu().numpy()
    vis=np.zeros(a[0].shape+(3,),np.uint8); vis[...,1]=a[0]*255; vis[...,2]=a[1]*255
    gi=cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    im=np.hstack([cv2.cvtColor(gi,cv2.COLOR_GRAY2BGR)[y0:y1,x0:x1], vis[y0:y1,x0:x1], (cv2.cvtColor( …[+161 chars]
```

<details><summary>output</summary>

```
A 30 base/Sy free 0.0860 0.0871   fixdepth 0.0899 0.0897
A 38 base/Sy free 0.0589 0.0599   fixdepth 0.0644 0.0656
A 46 base/Sy free 0.0410 0.0423   fixdepth 0.0487 0.0497
B 30 base/Sy free 0.0657 0.0662   fixdepth 0.0662 0.0657
B 38 base/Sy free 0.0466 0.0477   fixdepth 0.0475 0.0481
B 46 base/Sy free 0.0352 0.0374   fixdepth 0.0379 0.0409
```
</details>

**Read**
```
/work/scratch/sym_A.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/sym_B.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
c=(m.bounds[0]+m.bounds[1])/2
pc=(Tc[:,:3,:3]@c)+Tc[:,:3,3]  # bbox centre in cam
uv=pc[:,:2]/pc[:,2:]*K[0,0]+K[:2,2]
for nm,sl in [('A',slice(0,196)),('B',slice(585,700))]:
    print(nm,'z mean/std/min/max',pc[sl,2].mean().round(3),pc[sl,2].std().round(3),pc[sl,2].min().round(3),pc[sl,2].max().round(3),'uv std',uv[sl].std(0).round(2), 'rot spread', rotdist(Tc[sl,:3,:3],rot_mean(Tc[sl,:3,:3])[None]).mean().round(2))
# floor distance of lowest mesh points
pl=json.load(open(INP+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
V=m.vertices[::20]
for i in [0,50,100,150,190,600,650,699]:
    Vw=V@Tw[i,:3,:3].T+Tw[i,:3,3]; s=Vw@n+d
    print(i,'plane signed dist min/1pct', s.min().round(3), np.percentile(s,1).round(3), np.percentile(s,99).round(3))
H=np.load(INP+'/frame0_reference_human.npz'); print(H.files)
hv=H['vertices_world']; s=hv@n+d; print('ref human plane dist min/max',s.min(),s.max())
hr=r['human_vertices_world'][0]; s2=hr@n+d; print('recon human f0',s2.min(),s2.max())
EOF
```

<details><summary>output</summary>

```
A z mean/std/min/max 3.596 0.033 3.526 3.675 uv std [2.07 1.33] rot spread 0.76
B z mean/std/min/max 3.5 0.021 3.47 3.557 uv std [0.63 0.73] rot spread 0.58
0 plane signed dist min/1pct -0.005 0.01 0.163
50 plane signed dist min/1pct -0.017 -0.001 0.151
100 plane signed dist min/1pct -0.048 -0.03 0.125
150 plane signed dist min/1pct -0.025 -0.009 0.142
190 plane signed dist min/1pct -0.041 -0.026 0.137
600 plane signed dist min/1pct -0.037 -0.027 0.121
650 plane signed dist min/1pct -0.023 -0.011 0.141
699 plane signed dist min/1pct -0.022 -0.01 0.146
['vertices_world']
ref human plane dist min/max -0.009620737536536694 1.5815759648545527
recon human f0 -0.03882000746958347 1.5957703957986744
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
Tfp=r['T_cam_mesh_foundationpose'].astype(np.float64); Tcc=r['T_cam_mesh_coconet'].astype(np.float64)
om=np.load('om_half.npy'); pm=np.load('pm_half.npy')
Rr=Renderer(m,tex,768,576); Kh=K.copy(); Kh[:2]*=0.5; Kph=Kp.copy(); Kph[:2]*=0.5
def ious(K_,T):
    out=[]
    for s in range(0,700,50):
        rm=Rr.render(K_,T[s:s+50],rgb=False)['mask'].cpu().numpy()
        for j in range(len(rm)):
            i=s+j; v=~pm[i]
            a=rm[j]&v; b=om[i]&v
            out.append((a&b).sum()/max((a|b).sum(),1))
    return np.array(out)
ib=ious(Kh,Tc); ifp=ious(Kph,Tfp); icc=ious(Kph,Tcc)
np.save('iou_base.npy',ib)
for s in range(0,700,25): print(s, ib[s:s+25].mean().round(3), ib[s:s+25].min().round(3), ifp[s:s+25].mean().round(3), icc[s:s+25].mean().round(3))
EOF
```

<details><summary>output</summary>

```
0 0.862 0.837 0.813 0.862
25 0.848 0.827 0.833 0.85
50 0.862 0.838 0.827 0.863
75 0.879 0.831 0.81 0.877
100 0.885 0.867 0.829 0.889
125 0.881 0.855 0.83 0.885
150 0.886 0.865 0.782 0.861
175 0.85 0.81 0.748 0.799
200 0.799 0.77 0.729 0.748
225 0.804 0.784 0.742 0.756
250 0.774 0.749 0.749 0.748
275 0.808 0.765 0.795 0.81
300 0.835 0.803 0.835 0.853
325 0.838 0.819 0.825 0.851
350 0.819 0.798 0.832 0.855
375 0.857 0.8 0.852 0.88
400 0.893 0.867 0.877 0.884
425 0.902 0.88 0.842 0.882
450 0.876 0.863 0.821 0.858
475 0.881 0.842 0.817 0.868
500 0.876 0.843 0.788 0.87
525 0.897 0.877 0.826 0.891
550 0.884 0.863 0.867 0.9
575 0.9 0.884 0.873 0.907
600 0.895 0.883 0.868 0.904
625 0.895 0.881 0.824 0.887
650 0.9 0.89 0.838 0.893
675 0.894 0.886 0.835 0.886
```
</details>

**Bash**
```
cd /work/scratch; python -I -c "
import numpy as np
om=np.load('om_half.npy')
d=[]
for i in range(700):
    ys,xs=np.nonzero(om[i]); d.append((xs.max()-xs.min(), ys.max()-ys.min()))
d=np.array(d)*2; print(d.max(0), d.argmax(0))
"
```

<details><summary>output</summary>

```
[166 184] [578 359]
```
</details>

**Write**
```
/work/scratch/bfit.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/bfit.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/scratch; cat > t_static.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
c=(m.bounds[0]+m.bounds[1])/2
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
ids=list(range(0,196,4))+list(range(588,700,4))
with h5py.File(INP+'/masks.h5') as f:
    om=f['object'][ids]; pm=f['person'][ids]
Ks=[];tg=[];wt=[]
for j,i in enumerate(ids):
    ys,xs=np.nonzero(om[j]); kc,x0,y0=crop_K(K,xs.mean(),ys.mean()); Ks.append(kc)
    tg.append(crop_img(om[j],x0,y0)); p=cv2.dilate(pm[j].astype(np.uint8),np.ones((5,5)))
    wt.append(1-crop_img(p,x0,y0))
tg=np.stack(tg).astype(np.float32); wt=np.stack(wt).astype(np.float32)
T0=Tc[ids]
def centre(T): return T[:,:3,:3]@c+T[:,:3,3]
for fd in [False,True]:
    T,per,_=bf.fit(T0,Ks,tg,wt,iters=150,fix_depth=fd)
    for nm,sel in [('A',np.array(ids)<300),('B',np.array(ids)>300)]:
        pb=centre(T0[sel]); pf=centre(T[sel])
        print('fixdepth',fd,nm,'base: c std(cm)',(pb.std(0)*100).round(2),'rotspread',rotdist(T0[sel,:3,:3],rot_mean(T0[sel,:3,:3])[None]).mean().round(2),
              '| fit: c std',(pf.std(0)*100).round(2),'rotspread',rotdist(T[sel,:3,:3],rot_mean(T[sel,:3,:3])[None]).mean().round(2),
              'mean dz(cm)',((pf-pb)[:,2].mean()*100).round(1),'rot fit-vs-basemean',rotdist(rot_mean(T[sel,:3,:3]),rot_mean(T0[sel,:3,:3])).round(2), 'loss',per[sel].mea …[+36 chars]
```

<details><summary>output</summary>

```
fixdepth False A base: c std(cm) [0.83 0.79 3.4 ] rotspread 0.8 | fit: c std [0.11 0.27 0.5 ] rotspread 0.37 mean dz(cm) -14.1 rot fit-vs-basemean 1.22 loss 0.0076
fixdepth False B base: c std(cm) [0.4  0.52 2.03] rotspread 0.56 | fit: c std [0.15 0.41 1.3 ] rotspread 0.28 mean dz(cm) -10.7 rot fit-vs-basemean 0.7 loss 0.009
fixdepth True A base: c std(cm) [0.83 0.79 3.4 ] rotspread 0.8 | fit: c std [0.32 1.03 3.25] rotspread 0.62 mean dz(cm) 0.1 rot fit-vs-basemean 0.55 loss 0.0126
fixdepth True B base: c std(cm) [0.4  0.52 2.03] rotspread 0.56 | fit: c std [0.3  0.64 1.95] rotspread 0.54 mean dz(cm) 0.0 rot fit-vs-basemean 0.44 loss 0.0124
```
</details>

**Bash**
```
cd /work/scratch; cat > t_all.py <<'EOF'
import sys, time; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
t0=time.time()
with h5py.File(INP+'/masks.h5') as f:
    OM=f['object'][:]; PM=f['person'][:]
Ks=[];tg=[];wt=[]
for i in range(700):
    ys,xs=np.nonzero(OM[i]); kc,x0,y0=crop_K(K,xs.mean(),ys.mean()); Ks.append(kc)
    tg.append(crop_img(OM[i],x0,y0)); p=cv2.dilate(PM[i].astype(np.uint8),np.ones((5,5)))
    wt.append(1-crop_img(p,x0,y0))
tg=np.stack(tg).astype(np.float32); wt=np.stack(wt).astype(np.float32)
np.savez_compressed('crops.npz',Ks=np.stack(Ks),tg=tg.astype(bool),wt=wt.astype(bool))
Tf=[];L=[];L0=[]
for s in range(0,700,50):
    sl=slice(s,s+50)
    T,per,_=bf.fit(Tc[sl],Ks[sl],tg[sl],wt[sl],iters=150)
    _,per0,_=bf.fit(Tc[sl],Ks[sl],tg[sl],wt[sl],iters=0)
    Tf.append(T);L.append(per);L0.append(per0)
Tf=np.concatenate(Tf);L=np.concatenate(L);L0=np.concatenate(L0)
np.savez('fit_free.npz',T=Tf,loss=L,loss0=L0)
print('time',time.time()-t0)
EOF
python t_all.py
```

<details><summary>output</summary>

```
time 48.50801372528076
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
d=np.load('fit_free.npz'); Tf=d['T']
c=(m.bounds[0]+m.bounds[1])/2
pb=Tc[:,:3,:3]@c+Tc[:,:3,3]; pf=Tf[:,:3,:3]@c+Tf[:,:3,3]
rd=rotdist(Tf[:,:3,:3],Tc[:,:3,:3])
dRb=np.r_[0,rotdist(Tc[1:,:3,:3],Tc[:-1,:3,:3])]; dRf=np.r_[0,rotdist(Tf[1:,:3,:3],Tf[:-1,:3,:3])]
for i in range(0,700,10):
    print(i,'loss0 %.4f loss %.4f'%(d['loss0'][i],d['loss'][i]),'zb %.3f zf %.3f ratio %.3f'%(pb[i,2],pf[i,2],pb[i,2]/pf[i,2]),'rotdiff %.1f'%rd[i],'dRb %.2f dRf %.2f'%(dRb[i],dRf[i]))
EOF
```

<details><summary>output</summary>

```
0 loss0 0.0179 loss 0.0077 zb 3.526 zf 3.445 ratio 1.023 rotdiff 0.8 dRb 0.00 dRf 0.00
10 loss0 0.0192 loss 0.0073 zb 3.580 zf 3.456 ratio 1.036 rotdiff 1.0 dRb 0.32 dRf 0.41
20 loss0 0.0169 loss 0.0079 zb 3.597 zf 3.452 ratio 1.042 rotdiff 1.0 dRb 0.26 dRf 0.27
30 loss0 0.0219 loss 0.0078 zb 3.580 zf 3.450 ratio 1.038 rotdiff 0.7 dRb 0.46 dRf 0.11
40 loss0 0.0193 loss 0.0079 zb 3.581 zf 3.448 ratio 1.039 rotdiff 0.9 dRb 0.22 dRf 0.24
50 loss0 0.0221 loss 0.0077 zb 3.559 zf 3.456 ratio 1.030 rotdiff 0.9 dRb 0.25 dRf 0.12
60 loss0 0.0181 loss 0.0078 zb 3.589 zf 3.467 ratio 1.035 rotdiff 1.0 dRb 0.34 dRf 0.46
70 loss0 0.0153 loss 0.0078 zb 3.574 zf 3.451 ratio 1.036 rotdiff 1.2 dRb 0.31 dRf 0.15
80 loss0 0.0215 loss 0.0076 zb 3.596 zf 3.465 ratio 1.038 rotdiff 1.2 dRb 0.27 dRf 0.28
90 loss0  …[+5418 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
Tf=np.load('fit_free.npz')['T']
pl=json.load(open(INP+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
V=m.vertices[::10]
kp=r['human_keypoints_world']
import scipy.spatial as sp
def stats(Tcam):
    Tw_=Twc[None]@Tcam
    out=[]
    for i in range(700):
        Vw=V@Tw_[i,:3,:3].T+Tw_[i,:3,3]; s=Vw@n+d
        tree=sp.cKDTree(Vw)
        hd=[tree.query(kp[i,h])[0].min() for h in (list(range(21,42)),list(range(42,63)))]
        out.append((s.min(),hd[0],hd[1]))
    return np.array(out)
Tfs=Tf.copy(); Tfs[:,:3,3]*=1.036
sb=stats(Tc); sf=stats(Tfs)
np.save('phys_base.npy',sb); np.save('phys_fit.npy',sf)
for i in range(150,620,10): print(i,'base floor %.3f handR %.3f handL %.3f'%tuple(sb[i]),' | fit*1.036 floor %.3f handR %.3f handL %.3f'%tuple(sf[i]))
EOF
```

<details><summary>output</summary>

```
150 base floor -0.025 handR 0.299 handL 0.249  | fit*1.036 floor -0.013 handR 0.311 handL 0.263
160 base floor -0.031 handR 0.088 handL 0.040  | fit*1.036 floor -0.011 handR 0.122 handL 0.077
170 base floor -0.037 handR 0.004 handL 0.001  | fit*1.036 floor -0.013 handR 0.008 handL 0.004
180 base floor -0.042 handR 0.004 handL 0.003  | fit*1.036 floor -0.013 handR 0.026 handL 0.004
190 base floor -0.041 handR 0.004 handL 0.002  | fit*1.036 floor -0.016 handR 0.019 handL 0.004
200 base floor -0.024 handR 0.008 handL 0.004  | fit*1.036 floor -0.008 handR 0.001 handL 0.010
210 base floor -0.038 handR 0.001 handL 0.015  | fit*1.036 floor -0.002 handR 0.023 handL 0.007
220 base floor -0.031 handR 0.002 handL 0.007  | fit*1.036 floor -0.008 handR 0.002 handL 0.003
230 base floor -0.039 handR 0.00 …[+3706 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
from scipy.optimize import brentq
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
Tf=np.load('fit_free.npz')['T']
pl=json.load(open(INP+'/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
V=m.vertices[::10]
c=(m.bounds[0]+m.bounds[1])/2
zb=(Tc[:,:3,:3]@c+Tc[:,:3,3])[:,2]; zf=(Tf[:,:3,:3]@c+Tf[:,:3,3])[:,2]
ratio=zb/zf
print('episode median ratio',np.median(ratio), 'median zb/median zf', np.median(zb)/np.median(zf))
def floor_at(i,s):
    T=Tf[i].copy(); T[:3,3]*=s; Tw_=Twc@T; Vw=V@Tw_[:3,:3].T+Tw_[:3,3]; return (Vw@n+d).min()
static=list(range(0,196))+list(range(585,700))
sg=np.median(ratio)
h0=np.median([floor_at(i,sg) for i in static[::5]]); print('h0 at global scale',h0)
sfl=np.array([brentq(lambda s: floor_at(i,s)-h0, 0.8,1.3) for i in range(700)])
np.save('s_floor.npy',sfl)
for i in range(0,700,10): print(i, 'ratio_base %.3f  s_floor %.3f  -> dz floor vs global %.1f cm, base vs global %.1f cm'%(ratio[i],sfl[i],(sfl[i]-sg)*zf[i]*100,(ratio[i]-sg)*zf[i]*100))
for nm,sl in [('A',slice(0,196)),('M',slice(196,585)),('B',slice(585,700))]:
    print(nm,'median ratio base',np.median(ratio[sl]).round(4),'median s_floor',np.median(sfl[sl]).round(4), 'std s_floor', sfl[sl].std().round(4))
EOF
```

<details><summary>output</summary>

```
episode median ratio 1.0386068625422316 median zb/median zf 1.037442826389097
h0 at global scale -0.01658681511187554
0 ratio_base 1.023  s_floor 1.043  -> dz floor vs global 1.6 cm, base vs global -5.2 cm
10 ratio_base 1.036  s_floor 1.040  -> dz floor vs global 0.4 cm, base vs global -1.0 cm
20 ratio_base 1.042  s_floor 1.041  -> dz floor vs global 0.9 cm, base vs global 1.2 cm
30 ratio_base 1.038  s_floor 1.042  -> dz floor vs global 1.3 cm, base vs global -0.2 cm
40 ratio_base 1.039  s_floor 1.044  -> dz floor vs global 1.7 cm, base vs global 0.0 cm
50 ratio_base 1.030  s_floor 1.040  -> dz floor vs global 0.4 cm, base vs global -3.1 cm
60 ratio_base 1.035  s_floor 1.037  -> dz floor vs global -0.6 cm, base vs global -1.2 cm
70 ratio_base 1.036  s_floor 1.040  -> dz floor vs global 0.4 …[+5820 chars]
```
</details>

> *thinking:* The silhouette fitting method proves far steadier than the baseline (0.5 cm/0.3° vs 3.4 cm/0.8° frame-to-frame spread) and closely matches it during still phases, though during the flip both methods diverge from the baseline by up to 15 cm around frames 270–370, where the baseline sinks into the floor. Next I'll check the 180° symmetry ambiguity frame by frame.

**Bash**
```
cd /work/scratch; cat > t_sym.py <<'EOF'
import sys, time; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
S=np.load('Tsym_y.npy')
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg'].astype(np.float32); wt=d['wt'].astype(np.float32)
Tf=np.load('fit_free.npz')['T']
Ts=[];L=[]
for s in range(0,700,50):
    sl=slice(s,s+50)
    T,per,_=bf.fit(Tf[sl]@S,Ks[sl],tg[sl],wt[sl],iters=150)
    Ts.append(T);L.append(per)
Ts=np.concatenate(Ts);L=np.concatenate(L)
np.savez('fit_sym.npz',T=Ts,loss=L)
Lb=np.load('fit_free.npz')['loss']
for s in range(0,700,25): print(s, 'base %.4f sym %.4f  frac base better %.2f  rot sym-vs-(base@S) %.1f'%(Lb[s:s+25].mean(),L[s:s+25].mean(),(Lb[s:s+25]<L[s:s+25]).mean(), rotdist(Ts[s,:3,:3],(Tf[s]@S)[:3,:3])))
print('overall frac base better',(Lb<L).mean(), 'mean rel diff',((L-Lb)/Lb).mean())
EOF
python t_sym.py
```

<details><summary>output</summary>

```
0 base 0.0076 sym 0.0077  frac base better 0.84  rot sym-vs-(base@S) 0.4
25 base 0.0078 sym 0.0080  frac base better 0.88  rot sym-vs-(base@S) 0.4
50 base 0.0078 sym 0.0078  frac base better 0.68  rot sym-vs-(base@S) 0.6
75 base 0.0076 sym 0.0075  frac base better 0.40  rot sym-vs-(base@S) 0.8
100 base 0.0076 sym 0.0074  frac base better 0.24  rot sym-vs-(base@S) 0.9
125 base 0.0075 sym 0.0072  frac base better 0.08  rot sym-vs-(base@S) 0.6
150 base 0.0076 sym 0.0076  frac base better 0.44  rot sym-vs-(base@S) 0.8
175 base 0.0076 sym 0.0074  frac base better 0.16  rot sym-vs-(base@S) 1.2
200 base 0.0080 sym 0.0077  frac base better 0.00  rot sym-vs-(base@S) 1.5
225 base 0.0080 sym 0.0073  frac base better 0.00  rot sym-vs-(base@S) 1.7
250 base 0.0116 sym 0.0110  frac base better 0.00  rot  …[+1364 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > vis_sym.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from bfit import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
R=Renderer(m,tex,CROP,CROP)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg']; wt=d['wt']
Tb=np.load('fit_free.npz')['T']; Ts=np.load('fit_sym.npz')['T']
cap=cv2.VideoCapture(INP+'/video.mp4'); ids=[int(a) for a in sys.argv[1:]]; F={}
for i in range(max(ids)+1):
    ok,f=cap.read()
    if i in ids: F[i]=f
rows=[]
for i in ids:
    a=R.render(Ks[i],np.stack([Tb[i],Ts[i]]),rgb=False)['mask'].cpu().numpy()
    x0=int(K[0,2]-Ks[i][0,2]); y0=int(K[1,2]-Ks[i][1,2])
    img=crop_img(F[i],x0,y0)
    def col(r):
        v=np.zeros((CROP,CROP,3),np.uint8); v[...,1]=tg[i]*255; v[...,2]=r*255; v[~wt[i]]=(80,0,0); return v
    rows.append(np.hstack([img,col(a[0]),col(a[1])]))
cv2.imwrite('vis_sym.png',np.vstack(rows))
EOF
python vis_sym.py 300 650
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/vis_sym.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > t_dark.py <<'EOF'
import sys, time; sys.path.insert(0,'/work/scratch')
from bfit import *
from fit import dark_prob
m,tex=load_mesh(); K,Twc,Kp=load_cam()
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg'].astype(np.float32); wt=d['wt'].astype(np.float32)
cap=cv2.VideoCapture(INP+'/video.mp4'); IM=[]
for i in range(700):
    ok,f=cap.read(); x0=int(K[0,2]-Ks[i][0,2]); y0=int(K[1,2]-Ks[i][1,2]); IM.append(crop_img(f,x0,y0))
IM=np.stack(IM); np.save('imcrops.npy',IM)
DP=np.stack([dark_prob(im) for im in IM]).astype(np.float32)
# ROI: dilated SAM mask, minus person
roi=np.stack([cv2.dilate(t.astype(np.uint8),np.ones((21,21))) for t in tg]).astype(np.float32)*wt
Tb=np.load('fit_free.npz')['T']; Ts=np.load('fit_sym.npz')['T']
out={}
for nm,T0 in [('b',Tb),('s',Ts)]:
    L0=[];L1=[];TT=[]
    for s in range(0,700,50):
        sl=slice(s,s+50)
        _,p0,_=bf.fit(T0[sl],Ks[sl],DP[sl],roi[sl],iters=0)
        T,p1,_=bf.fit(T0[sl],Ks[sl],DP[sl],roi[sl],iters=60,lr_r=0.003,lr_t=0.002)
        L0.append(p0);L1.append(p1);TT.append(T)
    out[nm]=(np.concatenate(L0),np.concatenate(L1),np.concatenate(TT))
np.savez('fit_dark.npz',Lb0=out['b'][0],Lb1=out['b'][1],Tb=out['b'][2],Ls0=out['s'][0],Ls1=out['s'][1],Ts=out['s'][2])
for s in range(0,700,25):
    sl=slice(s,s+25)
    print(s,'eval-only base %.4f sym %.4f fracB %.2f | refined base %.4f sym %.4f fracB %.2f'%(out['b'][0][sl].mean(),out['s'][0][sl].mean(),(out['b'][0][s …[+136 chars]
```

<details><summary>output</summary>

```
0 eval-only base 0.1402 sym 0.1460 fracB 1.00 | refined base 0.1219 sym 0.1260 fracB 1.00
25 eval-only base 0.1440 sym 0.1482 fracB 1.00 | refined base 0.1236 sym 0.1274 fracB 1.00
50 eval-only base 0.1343 sym 0.1399 fracB 1.00 | refined base 0.1171 sym 0.1210 fracB 1.00
75 eval-only base 0.1260 sym 0.1276 fracB 0.72 | refined base 0.1069 sym 0.1095 fracB 1.00
100 eval-only base 0.0962 sym 0.0957 fracB 0.44 | refined base 0.0853 sym 0.0865 fracB 0.92
125 eval-only base 0.0906 sym 0.0915 fracB 0.72 | refined base 0.0857 sym 0.0853 fracB 0.40
150 eval-only base 0.0916 sym 0.0923 fracB 0.72 | refined base 0.0851 sym 0.0845 fracB 0.36
175 eval-only base 0.0893 sym 0.0883 fracB 0.24 | refined base 0.0852 sym 0.0828 fracB 0.00
200 eval-only base 0.0844 sym 0.0813 fracB 0.04 | refined base 0.0782 …[+1770 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > evalq.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from bfit import *
from fit import dark_prob
m,tex=load_mesh(); K,Twc,Kp=load_cam()
R=Renderer(m,tex,CROP,CROP); bf=BatchFitter(R)
d=np.load('crops.npz'); Ks=d['Ks']; tg=d['tg'].astype(np.float32); wt=d['wt'].astype(np.float32)
IM=np.load('imcrops.npy'); DP=np.stack([dark_prob(im) for im in IM]).astype(np.float32)
roi=np.stack([cv2.dilate(t.astype(np.uint8),np.ones((21,21))) for t in tg]).astype(np.float32)*wt
def evaluate(T):
    Ld=[];Ls=[];IoU=[]
    for s in range(0,700,50):
        sl=slice(s,s+50)
        _,pd,mm=bf.fit(T[sl],Ks[sl],DP[sl],roi[sl],iters=0)
        _,ps,_=bf.fit(T[sl],Ks[sl],tg[sl],wt[sl],iters=0)
        Ld.append(pd);Ls.append(ps)
        a=(mm>0.5)&(wt[sl]>0); b=(tg[sl]>0)&(wt[sl]>0)
        IoU.append((a&b).sum((1,2))/np.maximum((a|b).sum((1,2)),1))
    return np.concatenate(Ld),np.concatenate(Ls),np.concatenate(IoU)
if __name__=='__main__':
    r=np.load(INP+'/reconstruction.npz'); Tw=r['T_world_mesh'].astype(np.float64)
    Tc=np.linalg.inv(Twc)[None]@Tw
    res={'base':evaluate(Tc),'fitsam':evaluate(np.load('fit_free.npz')['T']),'fitdark':evaluate(np.load('fit_dark.npz')['Tb'])}
    np.savez('evalq.npz',**{k:np.stack(v) for k,v in res.items()})
    for s in range(0,700,50):
        sl=slice(s,s+50)
        print(s,'  '.join('%s dark %.4f sam %.4f iou %.3f'%(k,v[0][sl].mean(),v[1][sl].mean(),v[2][sl].mean()) for k,v in res.items()))
EOF
python evalq.py
```

<details><summary>output</summary>

```
0 base dark 0.1737 sam 0.0190 iou 0.854  fitsam dark 0.1421 sam 0.0077 iou 0.939  fitdark dark 0.1228 sam 0.0107 iou 0.918
50 base dark 0.1615 sam 0.0170 iou 0.868  fitsam dark 0.1301 sam 0.0077 iou 0.939  fitdark dark 0.1120 sam 0.0100 iou 0.923
100 base dark 0.1156 sam 0.0153 iou 0.881  fitsam dark 0.0934 sam 0.0076 iou 0.940  fitdark dark 0.0855 sam 0.0087 iou 0.932
150 base dark 0.1280 sam 0.0170 iou 0.868  fitsam dark 0.0904 sam 0.0076 iou 0.940  fitdark dark 0.0851 sam 0.0087 iou 0.934
200 base dark 0.1692 sam 0.0259 iou 0.800  fitsam dark 0.0839 sam 0.0080 iou 0.936  fitdark dark 0.0768 sam 0.0091 iou 0.929
250 base dark 0.1707 sam 0.0302 iou 0.788  fitsam dark 0.0909 sam 0.0118 iou 0.915  fitdark dark 0.0811 sam 0.0131 iou 0.907
300 base dark 0.1235 sam 0.0210 iou 0.843  fitsam dar …[+946 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh(); K,Twc,Kp=load_cam()
Tf=np.load('fit_free.npz')['T']; L=np.load('fit_free.npz')['loss']
c=(m.bounds[0]+m.bounds[1])/2
p=Tf[:,:3,:3]@c+Tf[:,:3,3]
RB=rot_mean(Tf[640:700,:3,:3]); pB=np.median(p[640:700],0)
RA=rot_mean(Tf[0:150,:3,:3]); pA=np.median(p[0:150],0)
for i in list(range(150,215,3))+list(range(560,650,3)):
    print(i,'vsA rot %.2f dp(cm) %s'%(rotdist(Tf[i,:3,:3],RA),((p[i]-pA)*100).round(1)),'  vsB rot %.2f dp %s'%(rotdist(Tf[i,:3,:3],RB),((p[i]-pB)*100).round(1)), 'loss %.4f'%L[i])
EOF
```

<details><summary>output</summary>

```
150 vsA rot 0.58 dp(cm) [0.1 0.4 0.5]   vsB rot 179.79 dp [45.1 -0.1  6.5] loss 0.0076
153 vsA rot 0.36 dp(cm) [0.2 0.5 0.8]   vsB rot 179.87 dp [45.2 -0.   6.8] loss 0.0077
156 vsA rot 0.28 dp(cm) [0.  0.2 0.1]   vsB rot 179.67 dp [45.1 -0.3  6. ] loss 0.0079
159 vsA rot 0.32 dp(cm) [-0.   0.  -0.3]   vsB rot 179.83 dp [45.  -0.4  5.6] loss 0.0076
162 vsA rot 0.25 dp(cm) [-0.   0.2  0.1]   vsB rot 179.80 dp [45.  -0.3  6. ] loss 0.0077
165 vsA rot 0.50 dp(cm) [0.1 0.2 0.2]   vsB rot 179.47 dp [45.1 -0.2  6.2] loss 0.0078
168 vsA rot 0.34 dp(cm) [0.  0.6 0.8]   vsB rot 179.86 dp [45.   0.1  6.8] loss 0.0079
171 vsA rot 0.38 dp(cm) [ 0.1  0.1 -0.1]   vsB rot 179.88 dp [45.1 -0.4  5.9] loss 0.0073
174 vsA rot 0.20 dp(cm) [-0.1  0.3  0.3]   vsB rot 179.86 dp [45.  -0.1  6.3] loss 0.0076
177 v …[+3935 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tc=np.linalg.inv(Twc)[None]@r['T_world_mesh'].astype(np.float64)
c=(m.bounds[0]+m.bounds[1])/2
for nm,T in [('fit',np.load('fit_free.npz')['T']),('base',Tc)]:
    q=Rot.from_matrix(T[:,:3,:3]).as_quat()
    for i in range(1,700):
        if q[i]@q[i-1]<0: q[i]*=-1
    p=T[:,:3,:3]@c+T[:,:3,3]
    for sg in [2,4]:
        qs=gaussian_filter1d(q,sg,axis=0); qs/=np.linalg.norm(qs,axis=1,keepdims=True)
        ps=gaussian_filter1d(p,sg,axis=0)
        rr=rotdist(Rot.from_quat(qs).as_matrix(),T[:,:3,:3]); dp=np.linalg.norm((ps-p)[:,:2],axis=1)*100; dz=np.abs(ps-p)[:,2]*100
        sl=slice(200,585)
        print(nm,'sigma',sg,'M: rot resid mean %.2f p95 %.2f | xy resid %.2f cm | z resid %.2f cm p95 %.2f'%(rr[sl].mean(),np.percentile(rr[sl],95),dp[sl].mean(),dz[sl].mean(),np.percentile(dz[sl],95)))
EOF
```

<details><summary>output</summary>

```
fit sigma 2 M: rot resid mean 0.34 p95 0.55 | xy resid 0.21 cm | z resid 0.53 cm p95 1.49
fit sigma 4 M: rot resid mean 0.46 p95 0.88 | xy resid 0.34 cm | z resid 0.77 cm p95 2.34
base sigma 2 M: rot resid mean 0.41 p95 0.71 | xy resid 0.43 cm | z resid 1.44 cm p95 4.12
base sigma 4 M: rot resid mean 0.54 p95 0.92 | xy resid 0.52 cm | z resid 1.66 cm p95 4.53
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tc=np.linalg.inv(Twc)[None]@r['T_world_mesh'].astype(np.float64)
d=np.load('fit_free.npz'); Tf=d['T']
dRf=np.r_[0,rotdist(Tf[1:,:3,:3],Tf[:-1,:3,:3])]; rd=rotdist(Tf[:,:3,:3],Tc[:,:3,:3])
print('top dRf', np.argsort(-dRf)[:15], np.sort(-dRf)[:15].round(2))
print('top rotdiff vs base', np.argsort(-rd)[:15], np.sort(-rd)[:15].round(2))
print('top loss', np.argsort(-d['loss'])[:15], np.sort(-d['loss'])[:15].round(4))
EOF
```

<details><summary>output</summary>

```
top dRf [502 514 547 578 250 525 236 533 542 512 576 504 528 346 426] [-2.66 -2.25 -1.47 -1.41 -1.41 -1.4  -1.39 -1.36 -1.36 -1.36 -1.33 -1.33
 -1.3  -1.29 -1.26]
top rotdiff vs base [217 209 207 214 208 218 220 213 221 206 210 239 222 226 223] [-4.06 -3.9  -3.82 -3.8  -3.78 -3.75 -3.73 -3.72 -3.65 -3.54 -3.5  -3.46
 -3.44 -3.42 -3.41]
top loss [266 267 280 265 264 268 263 269 282 281 270 271 278 283 277] [-0.0128 -0.0128 -0.0128 -0.0128 -0.0127 -0.0127 -0.0127 -0.0126 -0.0126
 -0.0126 -0.0125 -0.0125 -0.0125 -0.0124 -0.0124]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
import scipy.spatial as sp
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tc=np.linalg.inv(Twc)[None]@r['T_world_mesh'].astype(np.float64)
Tf=np.load('fit_free.npz')['T']; sfl=np.load('s_floor.npy')
kp=r['human_keypoints_world']; cp=r['contact_prob_right_left']
Tcw=np.linalg.inv(Twc)
V=m.vertices[::5]
c=(m.bounds[0]+m.bounds[1])/2
zb=(Tc[:,:3,:3]@c+Tc[:,:3,3])[:,2]; zf=(Tf[:,:3,:3]@c+Tf[:,:3,3])[:,2]
ss=np.linspace(0.95,1.12,69)
res=[]
for i in range(170,605,5):
    hands=[]
    if cp[i,0]>0.5: hands.append(kp[i,21:42])
    if cp[i,1]>0.5: hands.append(kp[i,42:63])
    hc=[(Tcw[:3,:3]@h.T).T+Tcw[:3,3] for h in hands]  # cam frame
    Vc0=V@Tf[i,:3,:3].T
    best=[]
    for s in ss:
        tree=sp.cKDTree(Vc0+Tf[i,:3,3]*s)
        best.append(sum(tree.query(h)[0].min() for h in hc))
    sh=ss[np.argmin(best)]
    res.append((i,sh,zb[i]/zf[i],sfl[i],min(best)))
    print(i,'s_hand %.3f (resid %.3f)  s_base %.3f  s_floor %.3f'%(sh,min(best),zb[i]/zf[i],sfl[i]))
res=np.array(res); M=(res[:,0]>=200)
print('M medians: hand',np.median(res[M,1]),'base',np.median(res[M,2]),'floor',np.median(res[M,3]))
EOF
```

<details><summary>output</summary>

```
170 s_hand 1.065 (resid 0.004)  s_base 1.053  s_floor 1.039
175 s_hand 1.048 (resid 0.004)  s_base 1.062  s_floor 1.040
180 s_hand 1.055 (resid 0.005)  s_base 1.059  s_floor 1.039
185 s_hand 1.055 (resid 0.003)  s_base 1.054  s_floor 1.036
190 s_hand 1.042 (resid 0.003)  s_base 1.056  s_floor 1.037
195 s_hand 1.070 (resid 0.003)  s_base 1.055  s_floor 1.038
200 s_hand 1.085 (resid 0.005)  s_base 1.055  s_floor 1.044
205 s_hand 1.055 (resid 0.003)  s_base 1.065  s_floor 1.048
210 s_hand 1.090 (resid 0.003)  s_base 1.073  s_floor 1.049
215 s_hand 1.068 (resid 0.003)  s_base 1.069  s_floor 1.045
220 s_hand 1.093 (resid 0.003)  s_base 1.066  s_floor 1.044
225 s_hand 1.083 (resid 0.004)  s_base 1.075  s_floor 1.048
230 s_hand 1.060 (resid 0.004)  s_base 1.071  s_floor 1.041
235 s_hand 1.062 (re …[+4491 chars]
```
</details>

> *thinking:* Now I'll write solve.py: it auto-detects phases, fits poses to silhouettes per frame and per still phase, smooths the moving phase, re-anchors depth using baseline medians and floor-contact cues, then outputs validation numbers.

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
old=s[s.index("    V5 = mesh.vertices[::5]"):s.index("    shift_hand =")]
new='''    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)
    s_hand = []
    for i in sM[::5]:
        hands = [kp[i, 21:42]] * (cp[i, 0] > .5) + [kp[i, 42:63]] * (cp[i, 1] > .5)
        if not hands: continue
        best = []
        for g in grid[::3]:
            Tm = np.linalg.inv(Twc @ apply_scale(T_sm[i:i + 1], [g])[0])  # world -> mesh
            best.append(sum(tree.query(h @ Tm[:3, :3].T + Tm[:3, 3])[0].min() for h in hands))
        s_hand.append(grid[::3][int(np.argmin(best))])
'''
s=s.replace(old,new)
old2=s[s.index("    rng = np.random.default_rng(seed)"):s.index("    c = (mesh.bounds[0]")]
s=s.replace(old2,"    P = trimesh.sample.sample_surface(mesh, 100000, seed=seed)[0]\n")
open('solve.py','w').write(s)
EOF
python -c "import trimesh,inspect; print(trimesh.__version__, inspect.signature(trimesh.sample.sample_surface))"
grep -n "P = trimesh" solve.py
```

<details><summary>output</summary>

```
5.1.1 (mesh, count: int | numpy.integer, face_weight: Union[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]], numpy._typing._nested_sequence._NestedSequence[numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]], bool, int, float, complex, str, bytes, numpy._typing._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]], NoneType] = None, sample_color=False, return_barycentric: bool = False, seed: int | numpy.integer | collections.abc.Sequence[int] | numpy.random.bit_generator.SeedSequence | numpy.random._generator.Generator | numpy.random.bit_generator.BitGenerator | None = None)
68:    P = trimesh.sample.sample_surface(mesh, 100000, seed=seed)[0]
```
</details>

**Bash**
```
cd /work/scratch && python -I /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/test_run/object_poses_world.npz 2>&1 | tail -150
```

<details><summary>output</summary>

```
/work/outputs/solve.py:325: DeprecationWarning: In future, it will be an error for 'np.bool_' scalars to be interpreted as an index
  hands = [kp[i, 21:42]] * (cp[i, 0] > .5) + [kp[i, 42:63]] * (cp[i, 1] > .5)
phases {'A': (0, 62), 'M': (63, 616), 'B': (617, 699)} thrA 3.02 thrB 2.25
symmetry votes {"A": {"frac_base_better_dark": 1.0, "frac_base_better_sam": 0.7619047619047619, "mean_dark_base": 0.14093531668186188, "mean_dark_sym": 0.14628998935222626}, "M": {"frac_base_better_dark": 0.5595667870036101, "frac_base_better_sam": 0.35018050541516244, "mean_dark_base": 0.08900041878223419, "mean_dark_sym": 0.08924049139022827}, "B": {"frac_base_better_dark": 0.7349397590361446, "frac_base_better_sam": 1.0, "mean_dark_base": 0.09079549461603165, "mean_dark_sym": 0.09187685698270798}} flip Fals …[+3349 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,tex=load_mesh()
Tf=np.load('fit_free.npz')['T']; F=700
ctr=(m.bounds[0]+m.bounds[1])/2
c=Tf[:,:3,:3]@ctr+Tf[:,:3,3]
def dev(ref):
    R0=rot_mean(Tf[ref,:3,:3]); c0=np.median(c[ref],0)
    return rotdist(Tf[:,:3,:3],R0[None]), np.linalg.norm(c-c0,axis=1)*100
for thr_r,thr_c in [(1.0,2.5),(1.5,3.0),(0.8,2.0)]:
    rA,cA=dev(slice(0,30)); mv=(rA>thr_r)|(cA>thr_c)
    a_end=next(i for i in range(F-2) if mv[i:i+3].all())-1
    rB,cB=dev(slice(F-30,F)); mv=(rB>thr_r)|(cB>thr_c)
    b=F-1
    while not mv[b-3:b].all(): b-=1
    print(thr_r,thr_c,'A_end',a_end,'B_start',b, 'max dev in A',rA[:a_end+1].max().round(2),cA[:a_end+1].max().round(2),'in B',rB[b:].max().round(2),cB[b:].max().round(2))
rB,cB=dev(slice(F-30,F))
print(np.c_[np.arange(570,620),rB[570:620].round(2),cB[570:620].round(2)].tolist())
EOF
```

<details><summary>output</summary>

```
1.0 2.5 A_end 200 B_start 588 max dev in A 1.03 1.54 in B 0.55 4.06
1.5 3.0 A_end 201 B_start 588 max dev in A 1.39 1.54 in B 0.55 4.06
0.8 2.0 A_end 117 B_start 588 max dev in A 0.83 1.41 in B 0.55 4.06
[[570.0, 6.62, 4.61], [571.0, 5.58, 3.07], [572.0, 4.98, 3.66], [573.0, 4.63, 3.07], [574.0, 3.81, 2.64], [575.0, 3.54, 5.67], [576.0, 2.74, 6.46], [577.0, 2.35, 7.49], [578.0, 1.04, 3.82], [579.0, 0.44, 3.83], [580.0, 0.65, 2.99], [581.0, 0.52, 2.07], [582.0, 0.63, 4.1], [583.0, 0.56, 4.94], [584.0, 0.55, 4.22], [585.0, 0.32, 3.47], [586.0, 0.32, 3.28], [587.0, 0.57, 3.47], [588.0, 0.43, 1.87], [589.0, 0.55, 4.06], [590.0, 0.45, 3.15], [591.0, 0.29, 1.09], [592.0, 0.3, 1.1], [593.0, 0.37, 2.9], [594.0, 0.44, 0.18], [595.0, 0.36, 0.14], [596.0, 0.27, 0.22], [597.0, 0.45, 0.99], [598.0, 0.3 …[+444 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2
F=np.load('frames_half.npy'); G=np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in F]).astype(np.float32)
om=np.load('om_half.npy'); pm=np.load('pm_half.npy')
def diff(ref):
    reg=cv2.dilate(om[ref].astype(np.uint8),np.ones((9,9)))>0
    out=[]
    for i in range(700):
        v=reg&~(cv2.dilate(pm[i].astype(np.uint8),np.ones((7,7)))>0)
        out.append(np.abs(G[i]-G[ref])[v].mean())
    return np.array(out)
d0=diff(0); dL=diff(699)
np.set_printoptions(precision=2,linewidth=220)
print('d0 0-100',d0[:100]); print('d0 180-210',d0[180:210]); print('dL 570-640',dL[570:640]); print('dL 640-699',dL[640:])
EOF
```

<details><summary>output</summary>

```
d0 0-100 [0.   1.09 1.3  1.47 1.76 1.68 1.73 1.8  1.98 1.95 1.99 2.16 2.36 2.27 2.25 2.31 2.55 2.44 2.51 2.49 2.49 2.48 2.49 2.5  2.53 2.53 2.53 2.51 2.55 2.51 2.5  2.48 2.55 2.52 2.5  2.5  2.56 2.55 2.54 2.53 2.56 2.51 2.53
 2.56 2.59 2.54 2.6  2.62 2.65 2.65 2.69 2.74 2.81 2.78 2.82 2.84 2.89 2.84 2.87 2.93 3.08 2.98 2.95 3.05 3.11 3.18 3.08 3.2  3.33 3.22 3.14 3.19 3.21 3.19 3.24 3.16 3.23 3.2  3.24 3.19 3.18 3.24 3.24 3.22 3.3  3.36
 3.58 3.54 3.52 3.59 3.68 3.63 3.7  3.65 3.68 3.69 3.78 3.67 3.68 3.69]
d0 180-210 [ 4.84  4.85  4.95  4.81  4.9   4.76  4.81  4.74  4.78  4.76  4.78  4.79  4.85  4.81  4.9   4.96  5.33  5.5   6.02  6.68  7.71  8.78  9.8  10.77 11.71 12.4  13.13 13.56 14.08 14.75]
dL 570-640 [13.61 13.26 13.07 12.68 12.28 11.88 11.37 10.21  7.48  3.56  4.08  3.27  3.19  3.0 …[+652 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2
F=np.load('frames_half.npy'); G=np.stack([cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in F]).astype(np.float32)
om=np.load('om_half.npy'); pm=np.load('pm_half.npy')
k=2; e=np.zeros(700)
for i in range(700):
    a,b=max(i-k,0),min(i+k,699)
    reg=cv2.dilate((om[a]|om[b]).astype(np.uint8),np.ones((9,9)))>0
    v=reg&~(cv2.dilate((pm[a]|pm[b]).astype(np.uint8),np.ones((7,7)))>0)
    e[i]=np.abs(G[b]-G[a])[v].mean() if v.sum()>50 else np.nan
np.save('motion_e.npy',e)
np.set_printoptions(precision=2,linewidth=220)
print('median static', np.median(e[:150]), np.percentile(e[:150],[90,99,100]), np.median(e[620:]), np.percentile(e[620:],[90,99,100]))
print('150-215',e[150:215]); print('560-620',e[560:620]); print('M min', np.nanmin(e[205:575]), np.argsort(e[205:575])[:10]+205)
EOF
```

<details><summary>output</summary>

```
median static 1.2539681792259216 [1.47 1.86 1.92] 0.9633733928203583 [1.09 1.16 1.16]
150-215 [ 1.29  1.16  1.2   1.16  1.47  1.33  1.46  1.41  1.42  1.25  1.25  1.24  1.48  1.29  1.43  1.27  1.43  1.34  1.26  1.36  1.43  1.18  1.17  1.19  1.46  1.29  1.34  1.29  1.43  1.26  1.18  1.11  1.37  1.3   1.35  1.25
  1.4   1.13  1.14  1.12  1.25  1.05  1.22  1.3   1.81  2.06  2.65  3.22  4.11  4.68  5.53  6.03  6.65  6.96  7.34  7.7   8.39  8.4   8.72  8.87  9.37  9.62 10.06 10.27 10.54]
560-620 [ 8.67  8.9   9.05  8.88  8.87  8.35  8.11  7.64  7.44  7.21  6.97  6.98  7.13  6.95  7.19  7.54  9.37 10.53  9.8   8.97  5.99  1.67  2.52  1.37  1.42  1.18  1.32  1.2   1.25  1.25  1.33  1.24  1.17  1.14  1.31  1.19
  1.27  1.27  1.3   1.1   1.02  1.04  1.16  1.06  1.32  1.19  1.26  1.12  1.07  1.22  1. …[+124 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
a=s.index("    def img_diff(ref):"); b=s.index("    phases = {")
new='''    # motion energy: mean |I(i+2) - I(i-2)| inside the (dilated) object masks, person pixels excluded
    e = np.zeros(F)
    for i in range(F):
        a_, b_ = max(i - 2, 0), min(i + 2, F - 1)
        reg = cv2.dilate((om_h[a_] | om_h[b_]).astype(np.uint8), np.ones((9, 9))) > 0
        v = reg & ~(cv2.dilate((pm_h[a_] | pm_h[b_]).astype(np.uint8), np.ones((7, 7))) > 0)
        e[i] = np.abs(GREY[b_] - GREY[a_])[v].mean() if v.sum() > 50 else np.inf
    # static-A ends 2 frames before the first run of 3 frames whose energy exceeds 1.6x the energy level
    # of the first 60 frames (the object is untouched there); static-B starts 3 frames after the last
    # such run (threshold from the last 60 frames).  Verified by eye: lift starts ~196, put down ~582.
    thrA = 1.6 * np.median(e[:60]); thrB = 1.6 * np.median(e[-60:])
    i0 = next(i for i in range(F - 2) if (e[i:i + 3] > thrA).all())
    a_end = i0 + 1
    i1 = next(i for i in range(F - 1, 2, -1) if (e[i - 2:i + 1] > thrB).all())
    b_start = i1 + 3
'''
s=s[:a]+new+s[b:]
s=s.replace("hands = [kp[i, 21:42]] * (cp[i, 0] > .5) + [kp[i, 42:63]] * (cp[i, 1] > .5)","hands = [kp[i, 21:42]] * int(cp[i, 0] > .5) + [kp[i, 42:63]] * int(cp[i, 1] > .5)")
# floor: use convex-hull vertices (the lowest point of a rigid body is always a hull vertex)
s=s.replace("        V = mesh.vertices[::10]\n","         …[+198 chars]
```

<details><summary>output</summary>

```
207:    thrA = 1.6 * np.median(e[:60]); thrB = 1.6 * np.median(e[-60:])
208:    i0 = next(i for i in range(F - 2) if (e[i:i + 3] > thrA).all())
213:    print('phases', phases, 'thrA %.2f thrB %.2f' % (thrA, thrB))
260:    print('symmetry votes', json.dumps(sym_votes), 'flip', flip)
297:    HULL = mesh.convex_hull.vertices
300:        V = HULL
353:    print('depth cues', json.dumps(depth_cues))
393:    print(json.dumps(table, indent=1))
394:    print('done in %.1f s' % (time.time() - t_start))
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
reps=[("    GREY = np.stack(GREY)\n","    GREY = np.stack(GREY)\n    print('[%.0fs] video read' % (time.time() - t_start))\n"),
("    T_fit, L_fit = fit_all(Tc_base, FIT_ITERS)\n","    T_fit, L_fit = fit_all(Tc_base, FIT_ITERS)\n    print('[%.0fs] per-frame fit' % (time.time() - t_start))\n"),
("    print('symmetry votes'","    print('[%.0fs] symmetry' % (time.time() - t_start))\n    print('symmetry votes'"),
("    # ---- 5. depth re-anchoring","    print('[%.0fs] static + smoothing' % (time.time() - t_start))\n    # ---- 5. depth re-anchoring"),
("    print('depth cues'","    print('[%.0fs] depth' % (time.time() - t_start))\n    print('depth cues'")]
for a,b in reps:
    assert a in s, a; s=s.replace(a,b)
open('solve.py','w').write(s)
EOF
cd /work/scratch && python -I /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/test_run/object_poses_world.npz 2>&1 | grep -v '^ ' | head -30; python -c "
import json; d=json.load(open('/work/scratch/test_run/object_poses_world_diagnostics.json'))
for k,v in d['table'].items(): print(k, {a:(round(b,4) if isinstance(b,float) else b) for a,b in v.items()})"
```

<details><summary>output</summary>

```
[8s] video read
phases {'A': (0, 196), 'M': (197, 584), 'B': (585, 699)} thrA 1.94 thrB 1.53
[44s] per-frame fit
[110s] symmetry
symmetry votes {"A": {"frac_base_better_dark": 0.766497461928934, "frac_base_better_sam": 0.40609137055837563, "mean_dark_base": 0.11428474634885788, "mean_dark_sym": 0.11656971275806427}, "M": {"frac_base_better_dark": 0.520618556701031, "frac_base_better_sam": 0.3324742268041237, "mean_dark_base": 0.0849471464753151, "mean_dark_sym": 0.08496078103780746}, "B": {"frac_base_better_dark": 0.7391304347826086, "frac_base_better_sam": 1.0, "mean_dark_base": 0.08870673179626465, "mean_dark_sym": 0.08997352421283722}} flip False
[120s] static + smoothing
[126s] depth
depth cues {"shift_floor_cm": -2.3571867613406985, "shift_size_constancy_cm": -4.841945708885721, "shif …[+2307 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
a=s.index("    disagreement = abs(shift_floor - shift_size)"); b=s.index("    T_final_c = apply_scale(T_sm, s_t)")
new='''    disagreement = abs(shift_floor - shift_size)
    correction = 0.5 * (shift_floor + shift_size)
    # Deviate from the baseline depth of the moving phase only if the two independent cues agree
    # (within ~2 cm; we use 2.5 cm as the hard limit) and the correction exceeds their disagreement.
    apply_corr = disagreement <= 2.5 and abs(correction) > disagreement
    s_target = s_phase['M'] + (correction / 100 / np.median(zr[sM]) if apply_corr else 0.0)
    # moving-phase scale: size-constancy ramp between the static phases, offset so that the phase
    # median matches the target; the offset is tapered to 0 over RAMP frames at both phase ends so the
    # trajectory stays continuous with the static phases
    taper = np.clip(np.minimum(sM - sM[0] + 1, sM[-1] - sM + 1) / RAMP, 0, 1)
    s_t = s_static.copy()
    off = s_target - np.median(ramp)
    for _ in range(20):  # adjust the offset so that the (tapered) median hits the target exactly
        off += s_target - np.median(ramp + off * taper)
    s_t[sM] = ramp + off * taper
'''
s=s[:a]+new+s[b:]
open('solve.py','w').write(s)
EOF
sed -n 330,360p solve.py
```

<details><summary>output</summary>

```
        if not hands: continue
        best = []
        for g in grid[::3]:
            Tm = np.linalg.inv(Twc @ apply_scale(T_sm[i:i + 1], [g])[0])  # world -> mesh
            best.append(sum(tree.query(h @ Tm[:3, :3].T + Tm[:3, 3])[0].min() for h in hands))
        s_hand.append(grid[::3][int(np.argmin(best))])
    shift_hand = (np.median(s_hand) - s_phase['M']) * np.median(zr[sM]) * 100 if s_hand else float('nan')
    disagreement = abs(shift_floor - shift_size)
    correction = 0.5 * (shift_floor + shift_size)
    # Deviate from the baseline depth of the moving phase only if the two independent cues agree
    # (within ~2 cm; we use 2.5 cm as the hard limit) and the correction exceeds their disagreement.
    apply_corr = disagreement <= 2.5 and abs(correction) > disagreement
    s_ta …[+1262 chars]
```
</details>

**Bash**
```
cd /work/scratch && python -I /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/test_run/object_poses_world.npz 2>&1 | grep -v '^ ' | grep -v '^[{}]' ; python -c "
import json; d=json.load(open('/work/scratch/test_run/object_poses_world_diagnostics.json'))
for k,v in d['table'].items(): print(k, {a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items() if 'spread' not in a})"
```

<details><summary>output</summary>

```
[12s] video read
phases {'A': (0, 196), 'M': (197, 584), 'B': (585, 699)} thrA 1.94 thrB 1.53
[85s] per-frame fit
[162s] symmetry
symmetry votes {"A": {"frac_base_better_dark": 0.751269035532995, "frac_base_better_sam": 0.45685279187817257, "mean_dark_base": 0.1141972616314888, "mean_dark_sym": 0.11670609563589096}, "M": {"frac_base_better_dark": 0.5489690721649485, "frac_base_better_sam": 0.3324742268041237, "mean_dark_base": 0.08489872515201569, "mean_dark_sym": 0.08486902713775635}, "B": {"frac_base_better_dark": 0.7739130434782608, "frac_base_better_sam": 1.0, "mean_dark_base": 0.08867767453193665, "mean_dark_sym": 0.08995752781629562}} flip False
[178s] static + smoothing
[185s] depth
depth cues {"shift_floor_cm": -2.188591286001623, "shift_size_constancy_cm": -4.700360109274248, "shi …[+1896 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
old=s[s.index("    apply_corr = disagreement <= 2.5"):s.index("    s_t[sM] = ramp + off * taper\n")]
new='''    apply_corr = disagreement <= 2.0 and abs(correction) > disagreement
    z_target = np.median(zb[sM]) + (correction / 100 if apply_corr else 0.0)
    # moving-phase scale: size-constancy ramp between the static phases, offset so that the phase
    # median camera depth equals z_target; the offset is tapered to 0 over RAMP frames at both phase
    # ends so the trajectory stays continuous with the static phases
    taper = np.clip(np.minimum(sM - sM[0] + 1, sM[-1] - sM + 1) / RAMP, 0, 1)
    s_t = s_static.copy()
    off = 0.0
    for _ in range(50):  # fixed-point iteration: median depth -> z_target
        off += (z_target - np.median(zr[sM] * (ramp + off * taper))) / np.median(zr[sM])
'''
s=s.replace(old,new)
s=s.replace("    # Deviate from the baseline depth of the moving phase only if the two independent cues agree\n    # (within ~2 cm; we use 2.5 cm as the hard limit) and the correction exceeds their disagreement.\n",
"    # Deviate from the baseline depth of the moving phase only if the two independent cues agree\n    # within 2 cm and the correction exceeds their disagreement (task rule); otherwise 0.\n")
open('solve.py','w').write(s)
EOF
sed -n 336,356p solve.py
```

<details><summary>output</summary>

```
    shift_hand = (np.median(s_hand) - s_phase['M']) * np.median(zr[sM]) * 100 if s_hand else float('nan')
    disagreement = abs(shift_floor - shift_size)
    correction = 0.5 * (shift_floor + shift_size)
    # Deviate from the baseline depth of the moving phase only if the two independent cues agree
    # within 2 cm and the correction exceeds their disagreement (task rule); otherwise 0.
    apply_corr = disagreement <= 2.0 and abs(correction) > disagreement
    z_target = np.median(zb[sM]) + (correction / 100 if apply_corr else 0.0)
    # moving-phase scale: size-constancy ramp between the static phases, offset so that the phase
    # median camera depth equals z_target; the offset is tapered to 0 over RAMP frames at both phase
    # ends so the trajectory stays continuous with the stati …[+719 chars]
```
</details>

> *thinking:* solve.py now runs fully in ~3 minutes, auto-detecting phases (A: 0–196, M: 197–584, B: 585–699); the symmetry cue confirms the baseline yaw in both still phases, but the moving-phase depth cues disagree by 2.5 cm (just over the 2 cm limit), so I'm keeping the baseline depth there. Now re-running and checking overlays.

**Bash**
```
cd /work/scratch && python -I /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/test_run/object_poses_world.npz 2>&1 | grep -E "phases|depth cues|done"; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
import json
m,tex=load_mesh(); K,Twc,Kp=load_cam()
r=np.load(INP+'/reconstruction.npz'); Tb=np.linalg.inv(Twc)[None]@r['T_world_mesh'].astype(np.float64)
To=np.linalg.inv(Twc)[None]@np.load('/work/scratch/test_run/object_poses_world.npz')['T_world_mesh']
d=json.load(open('/work/scratch/test_run/object_poses_world_diagnostics.json')); print(d['table']['M']['dz_median_cm'], d['table']['M']['dz_mean_cm'])
R=Renderer(m,tex,768,576); Kh=K.copy(); Kh[:2]*=0.5
F=np.load('frames_half.npy'); om=np.load('om_half.npy')
ids=[100,230,270,330,380,450,520,560]
ob=R.render(Kh,Tb[ids],rgb=False)['mask'].cpu().numpy(); oo=R.render(Kh,To[ids],rgb=False)['mask'].cpu().numpy()
ims=[]
for j,i in enumerate(ids):
    im=F[i].copy()
    for msk,col in [(om[i],(0,255,0)),(ob[j],(0,0,255)),(oo[j],(255,255,0))]:
        cnt,_=cv2.findContours(msk.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cnt,-1,col,1)
    ys,xs=np.nonzero(om[i]); cy,cx=int(ys.mean()),int(xs.mean())
    crop=cv2.resize(im[max(cy-80,0):cy+80, max(cx-100,0):cx+100],(500,400),interpolation=cv2.INTER_NEAREST); cv2.putText(crop,str(i),(10,30),0,1,(0,0,255),2); ims.append(crop)
cv2.imwrite('final_ov.jpg',np.vstack([np.hstack(ims[:4]),np.hstack(ims[4:])]))
EOF
```

<details><summary>output</summary>

```
phases {'A': (0, 196), 'M': (197, 584), 'B': (585, 699)} thrA 1.94 thrB 1.53
depth cues {"shift_floor_cm": -2.2211311387135755, "shift_size_constancy_cm": -4.684921790511652, "shift_hand_contact_cm_reported_only": -3.910871684648229, "disagreement_cm": 2.4637906517980763, "applied": false, "applied_median_shift_cm": 0.0, "s_phase": {"A": 1.0386732540025, "M": 1.0477846675428082, "B": 1.029859763283425}, "h0_floor_m": -0.01772788918399648, "s_floor_median": 1.0413756803679666}
done in 124.5 s
-1.159438037105276 -0.852212891735497
```
</details>

**Read**
```
/work/scratch/final_ov.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work && python -I /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > /work/outputs/solve_log.txt 2>&1; tail -3 /work/outputs/solve_log.txt; grep -E "phases|symmetry votes|depth cues" /work/outputs/solve_log.txt; python -I -c "
import numpy as np, json
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(T.shape,T.dtype, np.isfinite(T).all(), np.abs(np.linalg.det(T[:,:3,:3])-1).max())
d=json.load(open('/work/outputs/object_poses_world_diagnostics.json'))
for k,v in d['table'].items(): print(k, {a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items()})
print(d['runtime_s'])
"
```

<details><summary>output</summary>

```
 }
}
done in 163.2 s
phases {'A': (0, 196), 'M': (197, 584), 'B': (585, 699)} thrA 1.94 thrB 1.53
symmetry votes {"A": {"frac_base_better_dark": 0.751269035532995, "frac_base_better_sam": 0.43147208121827413, "mean_dark_base": 0.11444035917520523, "mean_dark_sym": 0.116734579205513}, "M": {"frac_base_better_dark": 0.520618556701031, "frac_base_better_sam": 0.33762886597938147, "mean_dark_base": 0.08500711619853973, "mean_dark_sym": 0.08490061014890671}, "B": {"frac_base_better_dark": 0.7478260869565218, "frac_base_better_sam": 1.0, "mean_dark_base": 0.08871490508317947, "mean_dark_sym": 0.08989279717206955}} flip False
depth cues {"shift_floor_cm": -2.0478176363159806, "shift_size_constancy_cm": -4.7195202352433565, "shift_hand_contact_cm_reported_only": -4.429097263049218, "disagreement_c …[+2215 chars]
```
</details>

**Bash**
```
cd /work; python -I -c "
import torch, nvdiffrast, cv2, trimesh, open3d, scipy, h5py, numpy
for m in [torch, nvdiffrast, cv2, trimesh, open3d, scipy, h5py, numpy]: print(m.__name__, getattr(m,'__version__','?'))
"; ls /work/tools; ls /work/outputs
```

<details><summary>output</summary>

```
torch 2.5.1+cu124
nvdiffrast 0.4.0
cv2 4.11.0
trimesh 5.1.1
open3d 0.20.0
scipy 1.16.3
h5py 3.16.0
numpy 1.26.3
object_poses_world.npz
object_poses_world_diagnostics.json
solve.py
solve_log.txt
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
cd /work/scratch; python -I -c "
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
a=np.load('/work/scratch/test_run/object_poses_world.npz')['T_world_mesh']; b=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
m,_=load_mesh(); V=m.vertices[::50]
add=np.array([np.linalg.norm((V@a[i,:3,:3].T+a[i,:3,3])-(V@b[i,:3,:3].T+b[i,:3,3]),axis=1).mean() for i in range(700)])*100
print('run-to-run ADD cm mean %.3f max %.3f'%(add.mean(),add.max()))
r=np.load(INP+'/reconstruction.npz')['T_world_mesh']
addb=np.array([np.linalg.norm((V@r[i,:3,:3].T+r[i,:3,3])-(V@b[i,:3,:3].T+b[i,:3,3]),axis=1).mean() for i in range(700)])*100
for nm,sl in [('A',slice(0,197)),('M',slice(197,585)),('B',slice(585,700))]: print(nm,'ADD(ours, baseline) cm mean %.2f median %.2f max %.2f'%(addb[sl].mean(),np.median(addb[sl]),addb[sl].max()))
"
```

<details><summary>output</summary>

```
run-to-run ADD cm mean 0.126 max 0.661
A ADD(ours, baseline) cm mean 3.41 median 2.84 max 8.46
M ADD(ours, baseline) cm mean 6.59 median 6.34 max 18.25
B ADD(ours, baseline) cm mean 2.01 median 1.60 max 6.67
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
cd /work/scratch; python -I -c "
import sys; sys.path.insert(0,'/work/scratch')
from lib import *
m,_=load_mesh(); K,Twc,Kp=load_cam(); c=(m.bounds[0]+m.bounds[1])/2
Ti=np.linalg.inv(Twc)
b=Ti[None]@np.load(INP+'/reconstruction.npz')['T_world_mesh'].astype(float); o=Ti[None]@np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
dz=((o[:,:3,:3]@c+o[:,:3,3])-(b[:,:3,:3]@c+b[:,:3,3]))[:,2]*100
print('min dz',dz.min().round(1),'at',dz.argmin(), 'max', dz.max().round(1), dz.argmax()); print(dz[260:380:10].round(1))
"
```

<details><summary>output</summary>

```
min dz -15.8 at 365 max 17.6 507
[ -6.2 -13.4 -12.7  -1.3  -9.3  -2.6  -3.2  -3.3  -4.8 -11.4 -15.  -10. ]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('REPORT.md').read()
a="frames 270–370 move up to ~15 cm closer than the baseline. That removes most of the baseline's\npenetration into the floor."
assert a in s
s=s.replace(a,"frames 270–370 move up to ~16 cm closer than the baseline, where the baseline swings back and sinks into the floor.\nAround frames 505–510 they move up to ~18 cm farther, where the baseline briefly jumps toward the camera.")
a="Peak GPU memory is about 2 GB\n(batches of 50 crops at 320×320)."
assert a in s
s=s.replace(a,"Peak GPU memory was not measured; I estimate it is well under the 9 GB limit\n(batches of 50 crops at 320×320 and a 95k-face mesh).")
open('REPORT.md','w').write(s)
EOF
ls -la /work/outputs
```

<details><summary>output</summary>

```
total 148
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 21 14:41 .
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 21 13:52 ..
-rw-rw-r-- 1 hanzhou hanzhou  9574 Oct 21 14:41 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou  1149 Oct 21 14:40 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 89878 Oct 21 14:40 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou  4157 Oct 21 14:40 object_poses_world_diagnostics.json
-rw-rw-r-- 1 hanzhou hanzhou 21762 Oct 21 14:35 solve.py
-rw-rw-r-- 1 hanzhou hanzhou  3996 Oct 21 14:40 solve_log.txt
```
</details>

**Agent:** I've produced the new trajectory, and by every check I could run without ground truth it beats the baseline. The final `solve.py` was run once with the required command and wrote poses for all 700 frames (valid rotations, no NaNs), plus `REPORT.md`, `TOOLS.md` and a diagnostics file.

**What happens in the video** (phases are detected automatically from image motion; I confirmed the boundaries by eye):
- **Frames 0–196:** the pallet stands still on its legs; the person grabs it around frame 168 but it doesn't move yet.
- **Frames 197–584:** it is tipped up over one long edge, stood on edge, and lowered onto its other face. It stays on the floor throughout.
- **Frames 585–699:** it lies still, legs up.

**Method:**
- **Per-frame fit:** each frame's pose is fitted by rendering the exact mesh and matching it to the object mask, ignoring pixels covered by the person.
- **Still phases:** one pose per phase, fitted to all its frames together.
- **Moving phase:** the per-frame fits are lightly smoothed over time.
- **Depth:** re-anchored along the camera rays so each phase's median depth equals the baseline's.

**Validation, baseline → result:**

| Check | Baseline | Result |
|---|---|---|
| Mask overlap (IoU) | 0.865 | 0.889 |
| Dark-pixel check (an image cue not used in fitting; lower is better) | 0.122 | 0.102 |
| Frame-to-frame jitter of object position (cm per frame²) | 3.4 | 0.07 |
| Worst sinking into the floor, 5th percentile | −5.6 cm | −3.8 cm |

While the object sits still, the baseline's depth wobbles by 2–3 cm per frame and its rotation by 0.6–0.8°. One pose per still phase removes that.

**Two judgement calls:**
- **180° yaw flip:** the pallet looks almost the same turned 180° about its deck-normal axis, and it's all black. The image cue held out from fitting prefers the baseline's orientation in 75% of frames in both still phases, so I kept it. I'm fairly but not fully confident: if this choice is wrong, every frame is wrong.
- **Moving-phase depth:**
  - Two independent cues say the object is closer than the baseline puts it: floor contact (−2.0 cm) and silhouette size (−4.7 cm). Hand contact, which isn't independent of the baseline, says −4.4 cm.
  - Those two cues disagree by 2.7 cm, over the task's 2 cm rule, so the applied median shift is 0. Had it passed I'd have applied about −3.4 cm, so this decision is borderline.
  - Even so, frame-by-frame depth now follows the silhouette fits instead of the baseline's swings, which reach about 16–18 cm.

**Limits:**
- Repeated runs aren't bit-identical: the renderer's GPU gradients are slightly non-deterministic, giving about 1.3 mm mean and 6.6 mm max difference between two runs.
- `solve.py` takes about 2–3 minutes. Total GPU use for the session was roughly 25 minutes, under the 45-minute limit.
- I didn't measure peak GPU memory; I estimate it is well under the 9 GB limit.
- No external tools or weights were downloaded. The script uses only installed packages and reads only the input files.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `object_poses_world_diagnostics.json`
- `solve_log.txt`

## Final

I've produced the new trajectory, and by every check I could run without ground truth it beats the baseline. The final `solve.py` was run once with the required command and wrote poses for all 700 frames (valid rotations, no NaNs), plus `REPORT.md`, `TOOLS.md` and a diagnostics file.

**What happens in the video** (phases are detected automatically from image motion; I confirmed the boundaries by eye):
- **Frames 0–196:** the pallet stands still on its legs; the person grabs it around frame 168 but it doesn't move yet.
- **Frames 197–584:** it is tipped up over one long edge, stood on edge, and lowered onto its other face. It stays on the floor throughout.
- **Frames 585–699:** it lies still, legs up.

**Method:**
- **Per-frame fit:** each frame's pose is fitted by rendering the exact mesh and matching it to the object mask, ignoring pixels covered by the person.
- **Still phases:** one pose per phase, fitted to all its frames together.
- **Moving phase:** the per-frame fits are lightly smoothed over time.
- **Depth:** re-anchored along the camera rays so each phase's median depth equals the baseline's.

**Validation, baseline → result:**

| Check | Baseline | Result |
|---|---|---|
| Mask overlap (IoU) | 0.865 | 0.889 |
| Dark-pixel check (an image cue not used in fitting; lower is better) | 0.122 | 0.102 |
| Frame-to-frame jitter of object position (cm per frame²) | 3.4 | 0.07 |
| Worst sinking into the floor, 5th percentile | −5.6 cm | −3.8 cm |

While the object sits still, the baseline's depth wobbles by 2–3 cm per frame and its rotation by 0.6–0.8°. One pose per still phase removes that.

**Two judgement calls:**
- **180° yaw flip:** the pallet looks almost the same turned 180° about its deck-normal axis, and it's all black. The image cue held out from fitting prefers the baseline's orientation in 75% of frames in both still phases, so I kept it. I'm fairly but not fully confident: if this choice is wrong, every frame is wrong.
- **Moving-phase depth:**
  - Two independent cues say the object is closer than the baseline puts it: floor contact (−2.0 cm) and silhouette size (−4.7 cm). Hand contact, which isn't independent of the baseline, says −4.4 cm.
  - Those two cues disagree by 2.7 cm, over the task's 2 cm rule, so the applied median shift is 0. Had it passed I'd have applied about −3.4 cm, so this decision is borderline.
  - Even so, frame-by-frame depth now follows the silhouette fits instead of the baseline's swings, which reach about 16–18 cm.

**Limits:**
- Repeated runs aren't bit-identical: the renderer's GPU gradients are slightly non-deterministic, giving about 1.3 mm mean and 6.6 mm max difference between two runs.
- `solve.py` takes about 2–3 minutes. Total GPU use for the session was roughly 25 minutes, under the 45-minute limit.
- I didn't measure peak GPU memory; I estimate it is well under the 9 GB limit.
- No external tools or weights were downloaded. The script uses only installed packages and reads only the input files.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `object_poses_world_diagnostics.json`
- `solve_log.txt`
