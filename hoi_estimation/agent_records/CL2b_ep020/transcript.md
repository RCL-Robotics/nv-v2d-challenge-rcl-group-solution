# CL2b_ep020

Claude Opus v2（ep11-21）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T12:45:38Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude/projects/-work/memory/ 2>/dev/null && cat /work/scratch/.claude/projects/-work/memory/MEMORY.md
```

<details><summary>output</summary>

```
Exit code 1
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 16:39 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 16:39 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 16:39 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 16:39 proxy.sock

inputs:
total 60116
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 16:39 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 16:39 ..
-rw-rw-r-- 1 hanzhou hanzhou      834 Oct 21 16:39 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       66 Oct 21 16:39 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   192425 Oct 21 16:39 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      394 Oct 21 16:39 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  4931672 Oct 21 16:39 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 11571184 Oct 21 16:39 object_mesh.glb
-rw-rw-r-- 1 hanzhou  …[+2421 chars]
```
</details>

**Bash**
```
cd /work; cat logs/run_manifest.txt; cat logs/proxy.log; ls /pipeline /pipeline/*; nvidia-smi; python -c "import torch, nvdiffrast; print(torch.__version__)"; nproc; free -g
```

<details><summary>output</summary>

```
date_utc: 2026-10-21T12:45:38Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2b_ep020:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000020:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
5659cc9e2d2ad4040ba9fe7ad5a68b321c8889128bf65891eaf81631246ff1e2  ./camera.json
58b010fce1a7cbbb0b9168cfa644378655cb61cd17442a04e4311f682a8bc8fb  ./episode.json
c96fbc606ae5341a21d14ec791e4e1c1346a7fd89e67db7dc3bb44d …[+3548 chars]
```
</details>

**Bash**
```
cd /work; echo CVD=$CUDA_VISIBLE_DEVICES; cat /pipeline/pipeline_report.json | head -100; ls /pipeline/depth/moge2 /pipeline/export/episode_000020 /pipeline/foundationpose/debug | head -40; ls /pipeline/work/masks | head
```

<details><summary>output</summary>

```
CVD=0
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
    "mtime_ns": 1791236187379682281,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000020/inference/episode_000020_step200000_before_after.mp4",
    "size": 38954331
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt …[+2826 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, trimesh
r=np.load('inputs/reconstruction.npz')
for k in r.files: print(k, r[k].shape, r[k].dtype)
m=trimesh.load('inputs/object_mesh.glb', force='scene', process=False).to_geometry()
print(m, m.bounds, m.extents)
print(m.visual.kind)
import trimesh.bounds
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
"

```

<details><summary>output</summary>

```
T_world_mesh (553, 4, 4) float32
T_cam_mesh_foundationpose (553, 4, 4) float32
T_cam_mesh_coconet (553, 4, 4) float32
human_vertices_world (553, 4610, 3) float32
human_keypoints_world (553, 70, 3) float32
contact_prob_right_left (553, 2) float32
<trimesh.Trimesh(vertices.shape=(295487, 3), faces.shape=(98533, 3))> [[-0.10614258 -0.11585464 -0.12354461]
 [ 0.1142482   0.10137311  0.10902679]] [0.22039078 0.21722775 0.2325714 ]
texture
[[-1.03569680e-02  9.99851835e-01 -1.37492109e-02  7.29859772e-03]
 [ 9.99946085e-01  1.03456997e-02 -8.90428324e-04 -4.02778551e-03]
 [-7.48051186e-04 -1.37576918e-02 -9.99905079e-01 -7.64841381e-03]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] [0.217078   0.22033025 0.23193262]
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
"; ls /pipeline/export/episode_000020/images | head -3; cat /pipeline/export/episode_000020/wild_export.json
```

<details><summary>output</summary>

```
30.0 553.0
553
front_stereo_camera_left.h5
{
  "aligned_object_extents_m": [
    0.21707800030708313,
    0.22033025324344635,
    0.2319326251745224
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
  "frame_count": 553,
  "height": 1152,
  "intrinsics": [
    [
      831.392822265625,
      0.0,
      768.0
    ],
    [
      0.0,
      831.392822265625,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "object_mesh_file": "/home/hanzhou/Projects/v2d/run/out/episode_000020/export/episode_000020/object_mesh/output_align …[+2040 chars]
```
</details>

> *thinking:* I've extracted the 553 frames and confirmed the clip shows someone walking up to a rice cooker, picking it up, rotating it, and setting it back down. Next, I'll examine the video and masks to split it into distinct phases.

**Bash**
```
cd /work; python -c "
import cv2, numpy as np, h5py
f=h5py.File('inputs/masks.h5','r'); print({k:f[k].shape for k in f})
om=f['object']; pm=f['person']
tiles=[]
for i in range(0,553,20):
    im=cv2.imread(f'scratch/frames/{i:04d}.jpg')
    o=om[i]; p=pm[i]
    im[o]=(0.5*im[o]+[0,0,127]).astype(np.uint8)
    im[p]=(0.6*im[p]+[100,0,0]).astype(np.uint8)
    im=cv2.resize(im,(384,288)); cv2.putText(im,str(i),(5,25),0,0.8,(0,255,255),2); tiles.append(im)
while len(tiles)%6: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[j:j+6]) for j in range(0,len(tiles),6)]
cv2.imwrite('scratch/montage.jpg', np.vstack(rows))
"

```

<details><summary>output</summary>

```
{'object': (553, 1152, 1536), 'person': (553, 1152, 1536)}
```
</details>

**Read**
```
/work/scratch/montage.jpg
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
f=h5py.File('inputs/masks.h5','r'); om=f['object'][:]; pm=f['person'][:]
np.save('scratch/om_area.npy', om.sum((1,2)))
for i in range(0,553,10):
    ys,xs=np.nonzero(om[i])
    print(i, om[i].sum(), (xs.min(),xs.max(),ys.min(),ys.max()) if len(xs) else None)
"

```

<details><summary>output</summary>

```
0 3396 (726, 785, 576, 640)
10 3429 (726, 785, 576, 641)
20 3346 (727, 785, 576, 640)
30 3426 (726, 785, 576, 640)
40 3361 (726, 785, 576, 640)
50 3441 (726, 785, 576, 640)
60 3432 (726, 785, 576, 640)
70 3436 (726, 785, 576, 640)
80 3360 (727, 785, 576, 640)
90 3360 (727, 785, 576, 640)
100 3335 (727, 785, 576, 640)
110 3329 (727, 785, 576, 640)
120 3320 (727, 785, 577, 640)
130 3341 (727, 785, 577, 640)
140 3328 (727, 785, 577, 640)
150 3316 (727, 785, 578, 640)
160 3290 (727, 785, 580, 640)
170 3265 (727, 785, 580, 640)
180 3268 (727, 785, 580, 640)
190 3261 (727, 785, 580, 640)
200 3266 (727, 785, 580, 640)
210 3255 (727, 785, 580, 640)
220 3266 (726, 784, 580, 640)
230 3160 (727, 784, 580, 640)
240 3140 (726, 782, 579, 639)
250 3190 (725, 782, 578, 640)
260 3252 (724, 783, 578, 640)
2 …[+868 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import cv2, numpy as np
tiles=[]
for i in range(0,553,8):
    im=cv2.imread(f'scratch/frames/{i:04d}.jpg')[540:680,680:840]
    im=cv2.resize(im,(240,210),interpolation=cv2.INTER_CUBIC); cv2.putText(im,str(i),(3,18),0,0.6,(0,255,255),2); tiles.append(im)
while len(tiles)%10: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[j:j+10]) for j in range(0,len(tiles),10)]
cv2.imwrite('scratch/crops.jpg', np.vstack(rows))
"

```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/crops.jpg
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
s=trimesh.load('inputs/object_mesh.glb', force='scene', process=False)
print(s.geometry)
m=s.to_geometry()
v=m.visual; print(type(v), v.uv.shape if v.uv is not None else None)
mat=v.material; print(type(mat)); 
img=getattr(mat,'baseColorTexture',None) or getattr(mat,'image',None); print(img.size if img else None)
img.save('scratch/tex.png')
"

```

<details><summary>output</summary>

```
OrderedDict([('ricecooker', <trimesh.Trimesh(vertices.shape=(295487, 3), faces.shape=(98533, 3))>)])
<class 'trimesh.visual.texture.TextureVisuals'> (295487, 2)
<class 'trimesh.visual.material.PBRMaterial'>
(2320, 2344)
```
</details>

> *thinking:* The rice cooker stays on the table throughout, just getting spun in place by the right hand (frames ~150-460) before settling still again—though its small size (~60×65 px) limits texture detail. Next, I'll build a renderer for the textured mesh to check the baseline's rotations against this.

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
mkdir -p /work/scratch/py; cd /work/scratch/py; cat > render_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
K,Twc,Kp,W,H=load_camera('/work/inputs')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Tcm=np.linalg.inv(Twc)@Twm
R=Renderer(v,f,uv,tex)
x0,y0,w,h=640,520,240,180
frames=[int(a) for a in sys.argv[1:]] or [0,100,200,300,400,500]
om=h5py.File('/work/inputs/masks.h5','r')['object']
tiles=[]
for i in frames:
    rgb,m,d=R.render(Tcm[i],K,x0,y0,w,h)
    rgb=(rgb[0].cpu().numpy()*255).astype(np.uint8)[...,::-1]; m=m[0].cpu().numpy()
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+h,x0:x0+w]
    ov=im.copy(); ov[m]=rgb[m]
    mk=om[i][y0:y0+h,x0:x0+w].astype(np.uint8)
    cs,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,cs,-1,(0,255,0),1)
    t=np.hstack([im,ov]); t=cv2.resize(t,None,fx=2,fy=2); cv2.putText(t,str(i),(3,20),0,0.7,(0,255,255),2); tiles.append(t)
cv2.imwrite('/work/scratch/render_test.jpg', np.vstack([np.hstack(tiles[j:j+2]) for j in range(0,len(tiles),2)]))
EOF
python render_test.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/render_test.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/tex.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Agent:** Need v flip. Fix and re-render.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
s=s.replace("""        self.uv = torch.tensor(uv, device=device)""","""        uv = uv.copy(); uv[:, 1] = 1.0 - uv[:, 1]  # glTF uv (v up from bottom in trimesh) -> texture rows
        self.uv = torch.tensor(uv, device=device)""")
open('hoi_lib.py','w').write(s)
EOF
cd /work/scratch/py; python render_test.py
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/render_test.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/py; cat > analyze.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
K,Twc,Kp,W,H=load_camera('/work/inputs')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Tcm=np.linalg.inv(Twc)@Twm
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
Rw=Twm[:,:3,:3]
# which mesh axis is aligned with world up (-y)
up=np.array([0,-1,0.])
ax=np.einsum('fij,i->fj',Rw,up)  # mesh-frame coords of world-up
print('mesh-frame up vector at frames:'); 
for i in range(0,553,25): 
    print(i, np.round(ax[i],3), 'ang0 %.1f'%rot_angle_deg(Rw[0],Rw[i]), 'step %.2f'%(rot_angle_deg(Rw[i-1],Rw[i]) if i else 0),
      'tcam',np.round(Tcm[i,:3,3],3),'fp',np.round(fp[i,:3,3],3),'cc',np.round(cc[i,:3,3],3),
      'R(base vs fp) %.1f'%rot_angle_deg(Tcm[i,:3,:3],fp[i,:3,:3]),'R(base vs cc) %.1f'%rot_angle_deg(Tcm[i,:3,:3],cc[i,:3,:3]))
st=np.array([0]+[rot_angle_deg(Rw[i-1],Rw[i]) for i in range(1,553)])
print('big steps', [(i,round(s,1)) for i,s in enumerate(st) if s>3])
print('contact', np.round(r['contact_prob_right_left'][::25],2).tolist())
EOF
python analyze.py
```

<details><summary>output</summary>

```
mesh-frame up vector at frames:
0 [-0.064 -0.022 -0.998] ang0 0.0 step 0.00 tcam [-0.004  0.169  3.049] fp [-0.051  0.123  3.274] cc [-0.042  0.115  3.191] R(base vs fp) 3.7 R(base vs cc) 1.3
25 [-0.069 -0.022 -0.997] ang0 0.4 step 0.53 tcam [-0.006  0.169  3.025] fp [-0.05   0.122  3.192] cc [-0.044  0.115  3.163] R(base vs fp) 2.6 R(base vs cc) 1.3
50 [-0.052 -0.022 -0.998] ang0 1.2 step 0.32 tcam [-3.000e-03  1.710e-01  3.044e+00] fp [-0.049  0.12   3.141] cc [-0.04   0.114  3.11 ] R(base vs fp) 3.3 R(base vs cc) 1.3
75 [-0.039 -0.017 -0.999] ang0 1.8 step 0.28 tcam [-1.000e-03  1.760e-01  3.091e+00] fp [-0.049  0.121  3.146] cc [-0.039  0.118  3.133] R(base vs fp) 3.9 R(base vs cc) 1.3
100 [-0.036 -0.01  -0.999] ang0 2.1 step 0.38 tcam [-1.000e-03  1.830e-01  3.106e+00] fp [-0.048  0.1 …[+3697 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; cat > yaw.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
K,Twc,Kp,W,H=load_camera('/work/inputs')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Rw=Twm[:,:3,:3]
# yaw: direction of mesh x axis projected to floor plane (world xz)
def yaw(R):
    a=R[:,:,0]; return np.degrees(np.arctan2(a[:,2],a[:,0]))
y=np.unwrap(np.radians(yaw(Rw))); y=np.degrees(y)
fp=np.linalg.inv(Twc)[None]  # cam->... we want world rot of fp: Twc@Tcm
for name,T in [('base',Twm),('fp',Twc@r['T_cam_mesh_foundationpose']),('cc',Twc@r['T_cam_mesh_coconet'])]:
    yy=np.degrees(np.unwrap(np.radians(yaw(T[:,:3,:3]))))
    print(name,' '.join('%d:%.0f'%(i,yy[i]-yy[0]) for i in range(0,553,10)))
EOF
python yaw.py
```

<details><summary>output</summary>

```
base 0:0 10:-1 20:-0 30:-0 40:-1 50:-1 60:-0 70:-0 80:-2 90:-1 100:-1 110:-1 120:-1 130:-1 140:-0 150:-0 160:-1 170:-0 180:-1 190:-1 200:-1 210:-2 220:-14 230:-24 240:-39 250:-53 260:-72 270:-91 280:-102 290:-102 300:-102 310:-102 320:-105 330:-123 340:-134 350:-163 360:-181 370:-194 380:-211 390:-229 400:-265 410:-281 420:-309 430:-309 440:-309 450:-309 460:-309 470:-308 480:-308 490:-308 500:-309 510:-309 520:-309 530:-309 540:-309 550:-309
fp 0:0 10:-0 20:-0 30:0 40:-0 50:-0 60:0 70:0 80:-1 90:-1 100:-1 110:-0 120:-1 130:-0 140:-0 150:0 160:-0 170:0 180:-0 190:-0 200:-0 210:-1 220:-13 230:-24 240:-38 250:-53 260:-71 270:-91 280:-102 290:-102 300:-102 310:-102 320:-105 330:-122 340:-133 350:-162 360:-181 370:-194 380:-211 390:-228 400:-264 410:-279 420:-308 430:-308 440:-309 450:-310 460 …[+531 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; cat > trans.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
K,Twc,Kp,W,H=load_camera('/work/inputs')
v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Tcm=np.linalg.inv(Twc)@Twm
c=v.mean(0)
vw=np.einsum('fij,nj->fni',Twm[:,:3,:3],v[::50].astype(np.float64))+Twm[:,None,:3,3]
bottom=vw[:,:,1].max(1)  # y-down: max y = lowest
top=vw[:,:,1].min(1)
pl=np.array(__import__('json').load(open('/work/inputs/ground_plane.json'))['plane'])
cw=vw.mean(1)
for i in range(0,553,10):
    pc=Tcm[i,:3,:3]@c+Tcm[i,:3,3]
    u=K@pc; u=u[:2]/u[2]
    print(i,'world ctr',np.round(cw[i],3),'bottom y %.3f'%bottom[i],'floor dist bottom %.3f'%(-(pl[:3]@vw[i].T+pl[3])).min(), 'cam z %.3f'%pc[2],'uv',np.round(u,1))
EOF
python trans.py
```

<details><summary>output</summary>

```
0 world ctr [0.146 0.292 3.12 ] bottom y 0.399 floor dist bottom -1.047 cam z 3.069 uv [754.3 607. ]
10 world ctr [0.146 0.29  3.1  ] bottom y 0.396 floor dist bottom -1.051 cam z 3.049 uv [754.7 606.8]
20 world ctr [0.145 0.291 3.106] bottom y 0.398 floor dist bottom -1.049 cam z 3.054 uv [754.3 607. ]
30 world ctr [0.145 0.294 3.111] bottom y 0.401 floor dist bottom -1.046 cam z 3.060 uv [754.3 607.8]
40 world ctr [0.147 0.298 3.146] bottom y 0.404 floor dist bottom -1.040 cam z 3.095 uv [754.3 607.9]
50 world ctr [0.147 0.293 3.114] bottom y 0.399 floor dist bottom -1.047 cam z 3.063 uv [754.7 607.5]
60 world ctr [0.147 0.295 3.136] bottom y 0.401 floor dist bottom -1.044 cam z 3.085 uv [754.5 607.4]
70 world ctr [0.149 0.298 3.157] bottom y 0.403 floor dist bottom -1.040 cam z 3.106 uv …[+4956 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; cat > viz.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoi_lib import *
v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
K,Twc,Kp,W,H=load_camera('/work/inputs')
R=Renderer(v,f,uv,tex)
def show(Tcm_list, frames, out, labels=None, S=4):
    x0,y0,w,h=690,545,128,112
    Ks=K.copy(); Ks[:2]*=S; Ks[0,2]=(K[0,2]+0.5)*S-0.5; Ks[1,2]=(K[1,2]+0.5)*S-0.5
    rows=[]
    for i in frames:
        im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+h,x0:x0+w]
        im=cv2.resize(im,(w*S,h*S),interpolation=cv2.INTER_CUBIC)
        row=[im]
        for T in Tcm_list:
            rgb,m,d=R.render(T[i],Ks,x0*S,y0*S,w*S,h*S)
            rgb=(rgb[0].cpu().numpy()*255).astype(np.uint8)[...,::-1]; m=m[0].cpu().numpy()
            o=im.copy(); o[m]=rgb[m]; row.append(o)
        row=np.hstack(row); cv2.putText(row,str(i),(5,30),0,1,(0,255,255),2); rows.append(row)
    cv2.imwrite(out,np.vstack(rows))
if __name__=='__main__':
    r=np.load('/work/inputs/reconstruction.npz')
    Tcm=np.linalg.inv(Twc)@r['T_world_mesh'].astype(np.float64)
    show([Tcm],[int(a) for a in sys.argv[2:]],sys.argv[1])
EOF
python viz.py /work/scratch/v1.jpg 0 290 500
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/v1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
s=s.replace("""        # OpenCV y-down; nvdiffrast clip y-up -> rows from bottom. Flip y so row 0 = top.
        clip = clip * torch.tensor([1, -1, 1, 1], device=self.dev)
""","""        # nvdiffrast output row 0 corresponds to NDC y=-1, which proj() maps to image row y0 (top): no flip.
""")
s=s.replace("""        # rows: clip y flipped means NDC y=-1 is top... nvdiffrast row 0 = ndc y=-1
""","")
open('hoi_lib.py','w').write(s)
EOF
cd /work/scratch/py; python viz.py /work/scratch/v1.jpg 0 290 500
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/v1.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* Fixed the vertical-flip bug, and the baseline overlay now aligns well at frames 0, 290, and 500. I noticed depth jitter in the first still phase (9 cm spread while the object is stationary), so next I'll build an image-matching score to refine the turning angle frame by frame.

**Bash**
```
cd /work/outputs; cat >> hoi_lib.py <<'EOF'


# ---------------- image evidence ----------------
import torch.nn.functional as F


def blur_t(x, sigma):
    """x: (B,C,H,W) gaussian blur."""
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    k = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2); k = k / k.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


class Scorer:
    """Scores candidate camera-frame poses against one video frame in a crop window."""

    def __init__(self, renderer, K, crop=(690, 545, 128, 112), ss=2, blur_sigma=1.0):
        self.R = renderer
        self.K = K
        self.x0, self.y0, self.w, self.h = crop
        self.ss = ss
        Ks = K.copy(); Ks[:2] *= ss
        Ks[0, 2] = (K[0, 2] + 0.5) * ss - 0.5; Ks[1, 2] = (K[1, 2] + 0.5) * ss - 0.5
        self.Ks = Ks
        self.blur_sigma = blur_sigma

    def set_frame(self, img_bgr, obj_mask, person_mask):
        x0, y0, w, h = self.x0, self.y0, self.w, self.h
        dev = self.R.dev
        im = img_bgr[y0:y0 + h, x0:x0 + w, ::-1].astype(np.float32) / 255.0
        self.img = torch.tensor(im.copy(), device=dev)  # h,w,3 RGB
        self.om = torch.tensor(obj_mask[y0:y0 + h, x0:x0 + w].copy(), device=dev).float()
        …[+3331 chars]
```

<details><summary>output</summary>

```
0 iou@0 0.870 ncc@0 0.599 best ncc [(-12, 0.611, 0.869), (-10, 0.611, 0.869), (-8, 0.61, 0.869), (-6, 0.608, 0.869), (-14, 0.606, 0.869)]
   ncc curve every 10deg 0.22 0.18 0.15 0.17 0.19 0.27 0.35 0.39 0.41 0.45 0.48 0.49 0.48 0.47 0.48 0.55 0.58 0.61 0.60 0.59 0.57 0.54 0.51 0.49 0.49 0.48 0.47 0.48 0.46 0.45 0.44 0.42 0.37 0.32 0.31 0.28
100 iou@0 0.879 ncc@0 0.629 best ncc [(-10, 0.641, 0.886), (-8, 0.639, 0.885), (-6, 0.637, 0.882), (-4, 0.635, 0.88), (-2, 0.633, 0.88)]
   ncc curve every 10deg 0.15 0.11 0.10 0.12 0.16 0.23 0.31 0.34 0.35 0.39 0.43 0.46 0.47 0.47 0.51 0.55 0.59 0.64 0.63 0.63 0.62 0.59 0.57 0.54 0.52 0.48 0.45 0.42 0.40 0.37 0.36 0.33 0.28 0.23 0.21 0.19
290 iou@0 0.897 ncc@0 0.515 best ncc [(90, 0.594, 0.899), (92, 0.593, 0.897), (94, 0.591, 0.895), (96, 0.587, 0.894 …[+567 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; cat > vizyaw.py <<'EOF'
from common import *
from viz import show
# args: out frame deg1 deg2 ...
out=sys.argv[1]; i=int(sys.argv[2]); degs=[float(a) for a in sys.argv[3:]]
L=[]
for d in degs:
    T=Tcm.copy(); T[i]=yaw_pose(Tcm[i],up_cam,center,np.radians(d)); L.append(T)
show(L,[i],out)
EOF
python vizyaw.py /work/scratch/v2.jpg 290 0 90 -12
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/v2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('hoi_lib.py').read()
s=s.replace('''            out_iou.append(iou); out_ncc.append(ncc.mean(1))
        return torch.cat(out_iou).cpu().numpy(), torch.cat(out_ncc).cpu().numpy()''','''            out_iou.append(iou); out_ncc.append(ncc.mean(1))
        return torch.cat(out_iou).cpu().numpy(), torch.cat(out_ncc).cpu().numpy()

    def feat(self, x, kind):
        """x: (B,3,h,w) rgb -> feature maps."""
        if kind == "rgb":
            return x
        if kind == "dog":
            return blur_t(x, 0.7) - blur_t(x, 3.0)
        if kind == "dogg":
            g = x.mean(1, keepdim=True)
            return blur_t(g, 0.7) - blur_t(g, 3.0)
        if kind == "chroma":
            s = x.sum(1, keepdim=True) + 1e-3
            return x / s
        raise ValueError(kind)

    def score_feat(self, Tcm, kind="dog", chunk=64, erode=True):
        out_iou, out_ncc = [], []
        Tcm = np.asarray(Tcm)
        img = self.img.permute(2, 0, 1)[None]
        fi = self.feat(img, kind)
        for s in range(0, len(Tcm), chunk):
            rgb, m = self.render_batch(Tcm[s:s + chunk])
            v = self.valid[None]
            inter = (m * self.om[None] * v).sum((1, 2))
            union = ((m + self.om[None] - m * self.om[None]) * v).sum((1, 2))
            iou = inter / union.clamp(min=1)
            # composite render over the real background so that features at the silhouette are comparable
            comp = rgb + img * (1 - m[:, None])
   …[+1528 chars]
```

<details><summary>output</summary>

```
   0 | rgb: best -12 (0.614) @0 0.596 2nd 20 (0.560) | dog: best -8 (0.263) @0 0.198 2nd -60 (0.213) | dogg: best -8 (0.271) @0 0.205 2nd -60 (0.218) | chroma: best -138 (0.110) @0 0.067 2nd -168 (0.094)
 100 | rgb: best -10 (0.618) @0 0.605 2nd 24 (0.581) | dog: best -60 (0.240) @0 0.162 2nd 92 (0.233) | dogg: best -60 (0.246) @0 0.169 2nd 92 (0.239) | chroma: best 10 (0.146) @0 0.124 2nd -136 (0.044)
 200 | rgb: best -6 (0.644) @0 0.627 2nd 28 (0.596) | dog: best -6 (0.357) @0 0.314 2nd 60 (0.323) | dogg: best -6 (0.372) @0 0.326 2nd 60 (0.333) | chroma: best 6 (0.210) @0 0.195 2nd -138 (0.157)
 250 | rgb: best -4 (0.708) @0 0.667 2nd 50 (0.543) | dog: best -4 (0.546) @0 0.391 2nd -50 (0.313) | dogg: best -4 (0.561) @0 0.401 2nd -50 (0.324) | chroma: best 70 (0.052) @0 -0.058 2nd -96 (0. …[+1408 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; cat > motion.py <<'EOF'
from common import *
x0,y0,w,h=700,560,110,95
prev=None; res=[]
ref0=frame(0)[y0:y0+h,x0:x0+w].astype(np.float32)
refE=frame(552)[y0:y0+h,x0:x0+w].astype(np.float32)
for i in range(553):
    im=cv2.GaussianBlur(frame(i)[y0:y0+h,x0:x0+w].astype(np.float32),(5,5),1)
    pm=cv2.dilate(PM[i][y0:y0+h,x0:x0+w].astype(np.uint8),np.ones((9,9),np.uint8))>0
    om=OM[i][y0:y0+h,x0:x0+w]
    val=om&~pm
    d1=np.abs(im-prev).mean(-1)[val].mean() if prev is not None else 0
    d0=np.abs(im-cv2.GaussianBlur(ref0,(5,5),1)).mean(-1)[val].mean()
    dE=np.abs(im-cv2.GaussianBlur(refE,(5,5),1)).mean(-1)[val].mean()
    res.append((d1,d0,dE,val.sum(),om.sum()-val.sum()))
    prev=im
res=np.array(res); np.save('/work/scratch/motion.npy',res)
for i in range(0,553,4): print(i,' '.join('%.1f'%x for x in res[i]))
EOF
python motion.py | head -150
```

<details><summary>output</summary>

```
0 0.0 0.0 15.7 3396.0 0.0
4 0.5 0.9 16.1 3452.0 0.0
8 0.8 1.2 15.3 3349.0 0.0
12 0.4 1.2 15.7 3426.0 0.0
16 0.4 1.3 15.8 3429.0 0.0
20 0.4 1.6 15.1 3346.0 0.0
24 0.4 2.1 15.5 3430.0 0.0
28 0.9 2.5 15.2 3427.0 0.0
32 0.5 2.8 14.6 3347.0 0.0
36 0.3 2.9 14.9 3431.0 0.0
40 0.3 3.0 14.4 3361.0 0.0
44 0.4 2.9 15.0 3441.0 0.0
48 0.7 2.9 14.3 3353.0 0.0
52 0.5 3.0 14.8 3441.0 0.0
56 0.2 2.8 14.4 3363.0 0.0
60 0.4 2.7 15.0 3432.0 0.0
64 0.4 2.5 14.6 3354.0 0.0
68 0.8 2.5 14.7 3357.0 0.0
72 0.3 2.4 14.7 3356.0 0.0
76 0.4 2.4 14.7 3358.0 0.0
80 0.3 2.4 14.8 3360.0 0.0
84 0.4 2.6 14.9 3408.0 0.0
88 0.9 2.8 14.5 3356.0 0.0
92 0.5 3.1 14.5 3349.0 0.0
96 0.6 3.6 14.3 3337.0 0.0
100 0.4 3.6 14.1 3335.0 0.0
104 0.4 3.6 14.2 3321.0 0.0
108 0.7 3.5 14.3 3334.0 0.0
112 0.4 3.5 14.3 3334.0 0.0
116 0.6 3.6 14.4 …[+3255 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
import numpy as np; r=np.load('/work/scratch/motion.npy')
for a,b in [(200,232),(268,290),(305,335),(405,432)]:
    print(a, ' '.join('%d:%.1f'%(i,r[i,0]) for i in range(a,b)))
"
```

<details><summary>output</summary>

```
200 200:0.2 201:0.5 202:0.3 203:0.6 204:0.2 205:0.5 206:0.6 207:0.4 208:0.8 209:0.4 210:0.5 211:0.8 212:0.7 213:0.9 214:0.8 215:0.9 216:0.8 217:0.8 218:1.1 219:1.1 220:1.2 221:1.0 222:0.8 223:1.1 224:0.8 225:0.9 226:0.9 227:1.1 228:0.9 229:0.9 230:1.0 231:1.1
268 268:1.8 269:1.7 270:1.8 271:2.1 272:1.9 273:1.8 274:1.9 275:1.2 276:0.6 277:0.6 278:0.8 279:0.8 280:0.5 281:0.5 282:0.7 283:1.0 284:0.8 285:0.6 286:0.7 287:0.7 288:0.7 289:0.5
305 305:0.5 306:0.6 307:0.5 308:0.6 309:0.4 310:0.5 311:0.4 312:0.3 313:0.6 314:0.7 315:0.9 316:1.0 317:0.9 318:1.2 319:1.2 320:1.1 321:1.1 322:1.2 323:1.6 324:1.5 325:1.9 326:1.9 327:1.9 328:2.0 329:1.6 330:1.8 331:1.4 332:1.6 333:1.8 334:1.3
405 405:1.4 406:1.5 407:1.7 408:2.0 409:2.0 410:2.1 411:2.5 412:2.4 413:2.4 414:2.4 415:2.6 416:2.8 417:2.7 418:2.6  …[+103 chars]
```
</details>

> *thinking:* I identified five motion phases from frame differences (still, turning, pause, turning, still), and the turning angles match the baseline at about −102° and −207° (≈−309° total). Next I'll check whether the cooker sits flat on the floor plane to constrain its tilt.

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
import json
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3])
Rw=Twm[:,:3,:3]
u=np.einsum('fij,i->fj',Rw,n)
for i in [0,50,100,150,200,250,290,350,400,450,500,550]: print(i,np.round(u[i],3), '%.1f deg from -z'%np.degrees(np.arccos(-u[i,2])))
print('median S1',np.median(u[:209],0),'S2',np.median(u[421:],0),'P',np.median(u[276:313],0))
"
```

<details><summary>output</summary>

```
0 [-0.006 -0.013 -1.   ] 0.8 deg from -z
50 [ 0.006 -0.013 -1.   ] 0.9 deg from -z
100 [ 0.023 -0.002 -1.   ] 1.3 deg from -z
150 [ 0.003 -0.023 -1.   ] 1.3 deg from -z
200 [ 0.008 -0.018 -1.   ] 1.1 deg from -z
250 [ 0.01  -0.014 -1.   ] 1.0 deg from -z
290 [-0.013 -0.038 -0.999] 2.3 deg from -z
350 [-0.009 -0.026 -1.   ] 1.6 deg from -z
400 [-0.021  0.006 -1.   ] 1.2 deg from -z
450 [ 0.008 -0.008 -1.   ] 0.7 deg from -z
500 [-0.006 -0.002 -1.   ] 0.5 deg from -z
550 [-0.026 -0.011 -1.   ] 1.6 deg from -z
median S1 [ 0.00236992 -0.01216747 -0.99985308] S2 [-0.01252451 -0.00821381 -0.99986221] P [-0.00590723 -0.04778762 -0.99882749]
```
</details>

> *thinking:* The cooker's mesh stays upright (within 1° of vertical) throughout, so I'll constrain it to table-upright and solve only for turning angle plus 3D position; since depth medians and mask positions disagree on near/far motion, I'll resolve this with a differentiable render-and-compare fit pooled over each still phase.

**Bash**
```
cd /work/outputs; cat >> hoi_lib.py <<'EOF'


# ---------------- differentiable fitting ----------------
def rotz_t(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


class DiffFitter:
    """Differentiable render-and-compare over a set of frames sharing one crop window."""

    def __init__(self, renderer, K, crop=(690, 545, 128, 112), ss=2):
        self.R = renderer
        self.x0, self.y0, self.w, self.h = crop
        self.ss = ss
        Ks = K.copy(); Ks[:2] *= ss
        Ks[0, 2] = (K[0, 2] + 0.5) * ss - 0.5; Ks[1, 2] = (K[1, 2] + 0.5) * ss - 0.5
        self.P = torch.tensor(renderer.proj(Ks, self.x0 * ss, self.y0 * ss, self.w * ss, self.h * ss), device=renderer.dev)

    def load_frames(self, imgs, oms, pms):
        import cv2
        x0, y0, w, h = self.x0, self.y0, self.w, self.h
        dev = self.R.dev
        I, O, V = [], [], []
        for im, om, pm in zip(imgs, oms, pms):
            I.append(im[y0:y0 + h, x0:x0 + w, ::-1].astype(np.float32) / 255.0)
            O.append(om[y0:y0 + h, x0:x0 + w].astype(np.float32))
            p = cv2.dilate(pm[y0:y0 + h, x0:x0 + w].astype(np.uint8), np.ones((5, 5), np.uint8))
            V.append(1.0 - p.astype(np.float32))
        self.img = torch.tensor(np.stack(I), device=dev).permute(0, 3, 1, 2).contiguous()
        self.om = torch.tensor(np. …[+1917 chars]
```

<details><summary>output</summary>

```
ok
```
</details>

**Bash**
```
cd /work/scratch/py; cat >> common.py <<'EOF'
import json, torch
nup=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]); nup/=np.linalg.norm(nup)
def upright_basis(R0):
    # minimal rotation taking R0 @ (-e_z) onto nup
    a=R0@np.array([0,0,-1.]); ax=np.cross(a,nup); s=np.linalg.norm(ax); c=a@nup
    Rc=rotvec_to_R(ax/s*np.arctan2(s,c)) if s>1e-9 else np.eye(3)
    return Rc@R0
B=upright_basis(Twm[0,:3,:3])
def yaw_of(Rwm):
    M=B.T@Rwm  # ~ Rz(theta)
    return np.arctan2(M[...,1,0],M[...,0,0])
yaw_base=np.unwrap(yaw_of(Twm[:,:3,:3]))
Rcw=Tcw[:3,:3]; tcw=Tcw[:3,3]
EOF
cat > static_fit.py <<'EOF'
from common import *
import time
a,b,step=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); fixz=len(sys.argv)>4 and sys.argv[4]=='fixz'
fr=list(range(a,b+1,step))
D=DiffFitter(Rd,K)
D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
th0=np.median(yaw_base[a:b+1])
tc0=np.median(Tcm[a:b+1,:3,3],0)
dev='cuda'
Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
th=torch.tensor(th0,device=dev,dtype=torch.float32,requires_grad=True)
dt=torch.zeros(3,device=dev,requires_grad=True)
opt=torch.optim.Adam([{'params':[th],'lr':0.01},{'params':[dt],'lr':0.002}])
tc0t=torch.tensor(tc0,device=dev,dtype=torch.float32)
t0=time.time()
for it in range(150):
    Rcm=(Bc@rotz_t(th))[None].expand(len(fr),3,3)
    d=dt*torch.tensor([1,1,0. if fixz else 1.],device=dev)
    tcm=(tc0t+d)[None].expand(len(fr),3)
    iou,ncc=D.losses(Rcm,tcm,torch.arange( …[+906 chars]
```

<details><summary>output</summary>

```
0 iou 0.8778 ncc 0.1736 dth 0.57deg dt(mm) [0. 0. 0.]
30 iou 0.9580 ncc 0.4942 dth 2.92deg dt(mm) [ -8.    3.1 -59.8]
60 iou 0.9433 ncc 0.6261 dth 1.24deg dt(mm) [  -7.5    1.3 -110. ]
90 iou 0.9337 ncc 0.6521 dth 0.77deg dt(mm) [  -8.3   -0.7 -142. ]
120 iou 0.9306 ncc 0.6572 dth 0.77deg dt(mm) [  -8.7   -1.2 -153.1]
149 iou 0.9281 ncc 0.6546 dth 0.93deg dt(mm) [  -8.7   -1.1 -155.7]
time 4.256953239440918
baseline per-frame: iou 0.8755 ncc 0.1644
fitted cam t [-0.0121  0.1722  2.9198]  baseline median [-0.0034  0.1733  3.0755]  baseline mean z 3.0694
0 iou 0.8853 ncc 0.2460 dth 0.57deg dt(mm) [0. 0. 0.]
30 iou 0.9340 ncc 0.5371 dth -1.17deg dt(mm) [ -4.9   5.2 -27.1]
60 iou 0.9304 ncc 0.5412 dth -0.72deg dt(mm) [ -4.8   3.4 -43.7]
90 iou 0.9260 ncc 0.5889 dth -0.06deg dt(mm) [ -4.6   4.3 …[+883 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; for r in "0 130 3" "465 552 3" "280 312 2" "140 208 3"; do python static_fit.py $r fixz | tail -4; done
```

<details><summary>output</summary>

```
149 iou 0.9344 ncc 0.2966 dth 3.50deg dt(mm) [-6.2  1.8  0. ]
time 2.8705291748046875
baseline per-frame: iou 0.8755 ncc 0.1644
fitted cam t [-0.0096  0.1751  3.0755]  baseline median [-0.0034  0.1733  3.0755]  baseline mean z 3.0694
149 iou 0.9387 ncc 0.4949 dth 0.33deg dt(mm) [-5.1  5.   0. ]
time 1.9651715755462646
baseline per-frame: iou 0.8852 ncc 0.2702
fitted cam t [-2.3000e-03  1.6990e-01  3.0493e+00]  baseline median [2.8000e-03 1.6490e-01 3.0493e+00]  baseline mean z 3.0547
149 iou 0.9368 ncc 0.5247 dth -2.12deg dt(mm) [-9.6 -6.4  0. ]
time 1.9390482902526855
baseline per-frame: iou 0.8900 ncc 0.2839
fitted cam t [-0.015   0.1714  3.0801]  baseline median [-0.0053  0.1778  3.0801]  baseline mean z 3.0782
149 iou 0.9453 ncc 0.4647 dth -2.19deg dt(mm) [-6.9  3.8  0. ]
time 2.146643 …[+157 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
x0,y0,w,h=690,545,128,112
ims=[frame(i)[y0:y0+h,x0:x0+w] for i in [0,100,200,460,552]]
d=[cv2.absdiff(ims[0],ims[1])*4,cv2.absdiff(ims[0],ims[2])*4,cv2.absdiff(ims[3],ims[4])*4]
top=np.hstack(ims[:3]+[ims[4]]); bot=np.hstack(d+[np.zeros_like(ims[0])])
o=np.vstack([top,bot]); o=cv2.resize(o,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST); cv2.imwrite('/work/scratch/diff.jpg',o)
for i in [0,50,100,130,200]:
    m=OM[i]; ys,xs=np.nonzero(m); print(i, xs.mean(), ys.mean(), m.sum())
"

```

<details><summary>output</summary>

```
0 755.8103651354535 609.1637220259129 3396
50 755.7268235977914 609.0563789596048 3441
100 756.1586206896552 609.2146926536732 3335
130 756.0859024244238 609.2253816222687 3341
200 756.1044090630741 609.9810165339866 3266
```
</details>

**Read**
```
/work/scratch/diff.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
fits={'S1':np.array([-0.0096,0.1751,3.0755]),'S1b':np.array([-0.0107,0.185,3.0965]),'S2':np.array([-0.0023,0.1699,3.0493]),'P':np.array([-0.015,0.1714,3.0801])}
# mesh origin in cam; object bottom point = origin + R*(0,0,zmax) ~ origin + 0.109*(-up)
for k,t in fits.items():
    pw=Twc[:3,:3]@t+Twc[:3,3]
    h=nup@pw+d
    # ray direction for this point
    ray=Twc[:3,:3]@(t/t[2])
    # depth needed for height equal to S1 height
    print(k,'height of origin %.4f'%h)
hS1=nup@(Twc[:3,:3]@fits['S1']+Twc[:3,3])+d
for k,t in fits.items():
    # find scale s so that height(s*t)=hS1 : nup@(Rwc s t + twc)+d = hS1
    s=(hS1-d-nup@Twc[:3,3])/(nup@(Twc[:3,:3]@t))
    print(k,'z for equal height %.4f (baseline-anchored %.4f) diff %.1f cm'%(s*t[2],t[2],(s-1)*t[2]*100))
"

```

<details><summary>output</summary>

```
S1 height of origin 0.9163
S1b height of origin 0.9047
S2 height of origin 0.9236
P height of origin 0.9197
S1 z for equal height 3.0755 (baseline-anchored 3.0755) diff 0.0 cm
S1b z for equal height 3.0150 (baseline-anchored 3.0965) diff -8.1 cm
S2 z for equal height 3.1017 (baseline-anchored 3.0493) diff 5.2 cm
P z for equal height 3.1047 (baseline-anchored 3.0801) diff 2.5 cm
```
</details>

> *thinking:* Depth can't be reliably recovered from the object alone, so I'll keep the baseline's per-phase median depth as advised. The fits do improve angle and position estimates per still phase (IoU ~0.93–0.94 vs 0.88–0.90) and suggest a ~6 mm sideways shift, which I'll verify visually before trusting it.

**Bash**
```
cd /work/scratch/py; cat > contour.py <<'EOF'
from common import *
def contour_img(i, poses, colors, out, S=6):
    x0,y0,w,h=700,560,110,95
    im=cv2.resize(frame(i)[y0:y0+h,x0:x0+w],(w*S,h*S),interpolation=cv2.INTER_CUBIC)
    Ks=K.copy(); Ks[:2]*=S; Ks[0,2]=(K[0,2]+0.5)*S-0.5; Ks[1,2]=(K[1,2]+0.5)*S-0.5
    mk=cv2.resize(OM[i][y0:y0+h,x0:x0+w].astype(np.uint8),(w*S,h*S),interpolation=cv2.INTER_NEAREST)
    cs,_=cv2.findContours(mk,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,(0,255,0),1)
    for T,c in zip(poses,colors):
        _,m,_=Rd.render(T,Ks,x0*S,y0*S,w*S,h*S,need_rgb=False)
        m=m[0].cpu().numpy().astype(np.uint8)
        cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,c,1)
    cv2.imwrite(out,im)
if __name__=='__main__':
    i=int(sys.argv[1]); T2=Tcm[i].copy(); T2[:3,3]+=[float(x) for x in sys.argv[2:5]]
    contour_img(i,[Tcm[i],T2],[(0,0,255),(255,0,0)],'/work/scratch/contour.jpg')
EOF
python contour.py 0 -0.006 0.002 0
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/contour.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The sideways offset is confirmed—blue (fitted) outline aligns with the cooker while the red baseline is off by ~2px—so I'll use image-position refinement at fixed depth. Next I'll check how precisely the texture score constrains the turning angle during a still phase.

**Bash**
```
cd /work/scratch/py; cat > pooled_sweep.py <<'EOF'
from common import *
a,b,step=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); dx,dy=float(sys.argv[4]),float(sys.argv[5])
fr=list(range(a,b+1,step))
D=DiffFitter(Rd,K)
D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
th0=np.median(yaw_base[a:b+1]); tc0=np.median(Tcm[a:b+1,:3,3],0)+[dx,dy,0]
dev='cuda'
Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
grid=np.radians(np.arange(-180,180,2.))
res=[]
with torch.no_grad():
    for g in grid:
        th=torch.tensor(th0+g,device=dev,dtype=torch.float32)
        Rcm=(Bc@rotz_t(th))[None].expand(len(fr),3,3); tcm=torch.tensor(tc0,device=dev,dtype=torch.float32)[None].expand(len(fr),3)
        iou,ncc=D.losses(Rcm,tcm,torch.arange(len(fr)))
        iou2,ncc2=D.losses(Rcm,tcm,torch.arange(len(fr)),feat='rgb')
        res.append((iou.mean().item(),ncc.mean().item(),ncc2.mean().item()))
res=np.array(res)
for j in range(0,len(grid),3): print('%4d iou %.3f dog %.3f rgb %.3f'%(np.degrees(grid[j]),*res[j]))
k=np.argmax(res[:,1]); print('best dog',np.degrees(grid[k]), 'best rgb', np.degrees(grid[np.argmax(res[:,2])]),'best iou',np.degrees(grid[np.argmax(res[:,0])]))
EOF
python pooled_sweep.py 0 130 3 -0.006 0.002
```

<details><summary>output</summary>

```
-180 iou 0.898 dog 0.122 rgb 0.116
-174 iou 0.904 dog 0.115 rgb 0.134
-168 iou 0.909 dog 0.140 rgb 0.160
-162 iou 0.910 dog 0.143 rgb 0.189
-156 iou 0.908 dog 0.152 rgb 0.217
-150 iou 0.905 dog 0.149 rgb 0.251
-144 iou 0.901 dog 0.112 rgb 0.281
-138 iou 0.898 dog 0.100 rgb 0.312
-132 iou 0.896 dog 0.112 rgb 0.356
-126 iou 0.895 dog 0.131 rgb 0.380
-119 iou 0.894 dog 0.155 rgb 0.394
-114 iou 0.893 dog 0.152 rgb 0.392
-108 iou 0.892 dog 0.162 rgb 0.401
-102 iou 0.891 dog 0.192 rgb 0.410
 -96 iou 0.893 dog 0.208 rgb 0.406
 -90 iou 0.905 dog 0.181 rgb 0.403
 -84 iou 0.902 dog 0.158 rgb 0.412
 -78 iou 0.891 dog 0.141 rgb 0.425
 -72 iou 0.882 dog 0.131 rgb 0.434
 -66 iou 0.876 dog 0.111 rgb 0.453
 -59 iou 0.874 dog 0.084 rgb 0.463
 -54 iou 0.875 dog 0.050 rgb 0.476
 -48 iou 0.881 dog 0.030 rgb 0 …[+1343 chars]
```
</details>

> *thinking:* Found a single clear peak 2° from the baseline's median angle with nothing else competitive across the full sweep. I'll fit shared pose/angle for still phases and per-frame angle/position with temporal smoothness for turning phases, starting with a per-frame 360° sweep through the turning phases to check for any baseline flips.

**Bash**
```
cd /work/scratch/py; cat > flipcheck.py <<'EOF'
from common import *
fr=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
D=DiffFitter(Rd,K)
D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
dev='cuda'
Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
grid=np.radians(np.arange(-180,180,3.))
S=np.zeros((len(grid),len(fr)))
with torch.no_grad():
    for gi,g in enumerate(grid):
        th=torch.tensor(yaw_base[fr]+g,device=dev,dtype=torch.float32)
        Rcm=Bc[None]@rotz_t(th); tcm=torch.tensor(Tcm[fr,:3,3]+[-0.006,0.002,0],device=dev,dtype=torch.float32)
        iou,ncc=D.losses(Rcm,tcm,torch.arange(len(fr)))
        S[gi]=(iou+0.5*ncc).cpu().numpy()
np.save('/work/scratch/flip_%s_%s.npy'%(sys.argv[1],sys.argv[2]),S)
for j,i in enumerate(fr):
    k=np.argmax(S[:,j]); d=np.abs((np.degrees(grid-grid[k])+180)%360-180); k2=np.argmax(np.where(d>30,S[:,j],-9))
    print(i,'best %4d (%.3f)  @0 %.3f  2nd %4d (%.3f)'%(np.degrees(grid[k]),S[k,j],S[60,j],np.degrees(grid[k2]),S[k2,j]))
EOF
python flipcheck.py 205 425 5
```

<details><summary>output</summary>

```
205 best   -6 (1.176)  @0 1.174  2nd  -39 (1.107)
210 best    0 (1.160)  @0 1.160  2nd  -33 (1.113)
215 best    0 (1.136)  @0 1.136  2nd   54 (1.087)
220 best    0 (1.155)  @0 1.155  2nd  -33 (1.121)
225 best    3 (1.153)  @0 1.149  2nd   51 (1.107)
230 best    0 (1.152)  @0 1.152  2nd   51 (1.098)
235 best    3 (1.161)  @0 1.156  2nd  -36 (1.142)
240 best    0 (1.208)  @0 1.208  2nd   39 (1.092)
245 best    0 (1.215)  @0 1.215  2nd   39 (1.098)
250 best    0 (1.129)  @0 1.129  2nd   39 (1.046)
255 best    3 (1.197)  @0 1.186  2nd   36 (1.098)
260 best    0 (1.177)  @0 1.177  2nd  108 (1.082)
265 best    0 (1.189)  @0 1.189  2nd  108 (1.067)
270 best    0 (1.175)  @0 1.175  2nd  -36 (1.055)
275 best    0 (1.174)  @0 1.174  2nd  -39 (1.081)
280 best    0 (1.125)  @0 1.125  2nd  -84 (1.098)
 …[+1399 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
z=Tcm[:,2,3]
for n,(a,b) in dict(S1=(0,209),M1=(210,275),P=(276,313),M2=(314,420),S2=(421,552)).items():
    s=z[a:b+1]; print(n,'median %.4f mean %.4f std %.4f min %.3f max %.3f'%(np.median(s),s.mean(),s.std(),s.min(),s.max()))
from scipy.ndimage import gaussian_filter1d
zs=gaussian_filter1d(z,15,mode='nearest'); print(' '.join('%.3f'%x for x in zs[::15]))
"

```

<details><summary>output</summary>

```
S1 median 3.0921 mean 3.0808 std 0.0320 min 2.988 max 3.164
M1 median 3.0962 mean 3.0956 std 0.0203 min 3.060 max 3.175
P median 3.0727 mean 3.0787 std 0.0324 min 3.031 max 3.151
M2 median 3.1180 mean 3.1104 std 0.0253 min 3.049 max 3.165
S2 median 3.0577 mean 3.0738 std 0.0410 min 2.987 max 3.184
3.040 3.033 3.035 3.048 3.067 3.082 3.094 3.104 3.106 3.102 3.099 3.099 3.100 3.100 3.099 3.099 3.098 3.094 3.087 3.082 3.080 3.081 3.089 3.098 3.108 3.116 3.123 3.131 3.135 3.124 3.101 3.084 3.070 3.058 3.050 3.043 3.039
```
</details>

**Bash**
```
cd /work/scratch/py; cat > zfit.py <<'EOF'
from common import *
# IoU-only pooled fit of (theta,x,y,z) per segment
def fit(a,b,step,w_ncc=0.0,iters=200):
    fr=list(range(a,b+1,step))
    D=DiffFitter(Rd,K); D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
    th0=np.median(yaw_base[a:b+1]); tc0=np.median(Tcm[a:b+1,:3,3],0)
    dev='cuda'; Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
    th=torch.tensor(th0,device=dev,dtype=torch.float32,requires_grad=True)
    dt=torch.zeros(3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([{'params':[th],'lr':0.005},{'params':[dt],'lr':0.001}])
    tc0t=torch.tensor(tc0,device=dev,dtype=torch.float32)
    for it in range(iters):
        Rcm=(Bc@rotz_t(th))[None].expand(len(fr),3,3); tcm=(tc0t+dt)[None].expand(len(fr),3)
        iou,ncc=D.losses(Rcm,tcm,torch.arange(len(fr)))
        loss=-(iou.mean()+w_ncc*ncc.mean()); opt.zero_grad(); loss.backward(); opt.step()
    c=(tc0t+dt).detach().cpu().numpy()
    print(a,b,'w_ncc',w_ncc,'iou %.4f'%iou.mean().item(),'dth %.1f'%np.degrees(th.item()-th0),'z %.4f (base med %.4f) diff %.1fcm'%(c[2],tc0[2],100*(c[2]-tc0[2])))
for seg in [(0,130,3),(140,208,3),(276,313,2),(465,552,3),(421,460,2)]:
    fit(*seg)
EOF
python zfit.py
```

<details><summary>output</summary>

```
0 130 w_ncc 0.0 iou 0.9635 dth 8.6 z 3.0037 (base med 3.0755) diff -7.2cm
140 208 w_ncc 0.0 iou 0.9709 dth 4.0 z 3.0239 (base med 3.0965) diff -7.3cm
276 313 w_ncc 0.0 iou 0.9628 dth -5.7 z 3.0223 (base med 3.0727) diff -5.0cm
465 552 w_ncc 0.0 iou 0.9541 dth 8.3 z 3.0338 (base med 3.0493) diff -1.6cm
421 460 w_ncc 0.0 iou 0.9434 dth 12.9 z 3.0076 (base med 3.1121) diff -10.5cm
```
</details>

> *thinking:* Silhouette-only fits put all phases within ~3 cm of each other but run 2–10 cm too close to baseline, likely due to oversized blurred masks, so I won't treat them as absolute depth; the baseline's per-phase medians vary by up to 6 cm. I'll now check a third cue—constant table height—before deciding how to handle depth across phases.

**Bash**
```
cd /work/scratch/py; cat >> zfit.py <<'EOF'
EOF
python - <<'EOF'
import sys; sys.argv=['x']
exec(open('/work/scratch/py/common.py').read())
src=open('/work/scratch/py/zfit.py').read().split('for seg in')[0]
src=src.replace("    print(a,b,'w_ncc'","    return c,th.item()\n    print(a,b,'w_ncc'")
exec(src)
d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
for seg in [(0,130,3),(140,208,3),(276,313,2),(465,552,3),(421,460,2)]:
    c,th=fit(*seg)
    u=K@c; u=u[:2]/u[2]
    # bottom centre point of the cooker: mesh point (cx,cy,zmax) ; in world below origin along -up
    bot_m=np.array([center[0],center[1],v[:,2].max()])
    Rwm=B@np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
    Rcm_=Rcw@Rwm
    pb=Rcm_@bot_m+c
    ub=K@pb; ub=ub[:2]/ub[2]
    hw=nup@(Twc[:3,:3]@pb+Twc[:3,3])+d
    print(seg,'uv origin',np.round(u,2),'uv bottom',np.round(ub,2),'z %.4f'%c[2],'bottom height %.4f'%hw)
EOF
```

<details><summary>output</summary>

```
(0, 130, 3) uv origin [754.93 608.19] uv bottom [756.98 637.88] z 3.0032 bottom height 0.8137
(140, 208, 3) uv origin [754.98 608.27] uv bottom [757.14 637.74] z 3.0240 bottom height 0.8105
(276, 313, 2) uv origin [754.54 607.38] uv bottom [753.49 636.94] z 3.0224 bottom height 0.8140
(465, 552, 3) uv origin [756.3  606.44] uv bottom [758.98 635.5 ] z 3.0342 bottom height 0.8157
(421, 460, 2) uv origin [755.98 606.07] uv bottom [758.69 635.41] z 3.0081 bottom height 0.8207
```
</details>

> *thinking:* I'll use the pooled silhouette fits for relative depth shape across phases, then re-anchor the whole episode's median depth to the baseline's (allowed by the brief). Next, I'm writing the joint optimizer and automatic phase detection.

**Bash**
```
cd /work/scratch/py; python -c "
import numpy as np
from scipy.ndimage import median_filter
d1=np.load('/work/scratch/motion.npy')[:,0]
s=median_filter(d1,5,mode='nearest')
mv=s>0.8
def runs(m):
    r=[];a=0
    for i in range(1,len(m)+1):
        if i==len(m) or m[i]!=m[a]: r.append([a,i-1,bool(m[a])]); a=i
    return r
print(runs(mv))
"

```

<details><summary>output</summary>

```
[[0, 212, False], [213, 213, True], [214, 216, False], [217, 275, True], [276, 314, False], [315, 421, True], [422, 552, False]]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
print(' '.join('%d:%.1f'%(i,np.degrees(yaw_base[i]-yaw_base[0])) for i in range(200,232)))
print(' '.join('%d:%.1f'%(i,np.degrees(yaw_base[i]-yaw_base[0])) for i in range(408,440)))
x0,y0,w,h=700,560,110,95
t=[cv2.resize(frame(i)[y0:y0+h,x0:x0+w],None,fx=2,fy=2) for i in range(204,224,2)]
cv2.imwrite('/work/scratch/s.jpg',np.vstack([np.hstack(t[:5]),np.hstack(t[5:])]))
"

```

<details><summary>output</summary>

```
200:1.0 201:1.1 202:1.3 203:1.4 204:1.0 205:1.4 206:1.2 207:1.5 208:1.6 209:2.1 210:2.2 211:3.3 212:4.0 213:5.0 214:6.6 215:6.8 216:8.1 217:9.9 218:10.7 219:12.0 220:13.9 221:15.4 222:15.6 223:17.5 224:18.3 225:19.6 226:20.8 227:21.0 228:22.1 229:22.7 230:24.4 231:24.7
408:278.1 409:279.2 410:280.5 411:283.5 412:285.6 413:287.7 414:291.0 415:292.8 416:297.5 417:302.4 418:306.1 419:308.9 420:309.1 421:309.5 422:309.0 423:308.4 424:308.8 425:308.8 426:308.8 427:309.5 428:309.1 429:308.9 430:308.8 431:308.7 432:308.5 433:309.2 434:310.1 435:309.4 436:309.8 437:309.6 438:309.3 439:309.3
```
</details>

> *thinking:* I noticed motion starts gradually around frame 209, before the frame-difference signal crosses its threshold, so I'll pad each detected turning phase by 6 frames on each side (safer than mislabeling moving frames as still). Now writing the joint optimiser.

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
cd /work/outputs; python - <<'EOF'
s=open('traj_opt.py').read()
s=s.replace("""        self.off = torch.zeros(U, 4, device=dev, requires_grad=True)""","""        self.raw = torch.zeros(U, 4, device=dev, requires_grad=True)
        self.step = torch.tensor([0.5, 0.15, 0.15, 0.5], device=dev)  # Adam step size per dim (deg, px, px, cm)""")
s=s.replace("""    def unit_params(self):
        th""","""    @property
    def off(self):
        return self.raw * self.step

    def unit_params(self):
        th""")
s=s.replace("""        lrs = torch.tensor(lr, device=self.dev)
        opt = torch.optim.Adam([self.off], lr=1.0)""","""        self.step = torch.tensor(lr, device=self.dev)
        opt = torch.optim.Adam([self.raw], lr=1.0)""")
s=s.replace("""            with torch.no_grad():
                self.off.grad *= lrs  # per-dimension step scaling (Adam is scale invariant; this sets units)
                if fix_z:
                    self.off.grad[:, 3] = 0
""","""            if fix_z:
                with torch.no_grad():
                    self.raw.grad[:, 3] = 0
""")
open('traj_opt.py','w').write(s)
EOF
grep -n "raw\|step" traj_opt.py | head -20
```

<details><summary>output</summary>

```
69:        self.raw = torch.zeros(U, 4, device=dev, requires_grad=True)
70:        self.step = torch.tensor([0.5, 0.15, 0.15, 0.5], device=dev)  # Adam step size per dim (deg, px, px, cm)
76:        return self.raw * self.step
116:        self.step = torch.tensor(lr, device=self.dev)
117:        opt = torch.optim.Adam([self.raw], lr=1.0)
133:                    self.raw.grad[:, 3] = 0
134:            opt.step(); sched.step()
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('traj_opt.py').read()
s=s.replace("""        self.step = torch.tensor(lr, device=self.dev)
""","""        new = torch.tensor(lr, device=self.dev)
        with torch.no_grad():
            self.raw *= self.step / new  # keep current offsets when changing the step scale
        self.step = new
""")
open('traj_opt.py','w').write(s)
EOF
cd /work/scratch/py; cat > run_joint.py <<'EOF'
from common import *
sys.path.insert(0,'/work/outputs')
from traj_opt import *
import time
d1=np.load('/work/scratch/motion.npy')[:,0]
phases=segment_phases(d1); print(phases)
crop=(690,545,128,112)
t0=time.time()
imgs=[frame(i) for i in range(553)]; oms=[OM[i] for i in range(553)]; pms=[PM[i] for i in range(553)]
print('load',time.time()-t0)
# init: baseline yaw (unwrapped), baseline uv of origin, baseline z
o=Tcm[:,:3,3]; uv=(o@K.T); uv=uv[:,:2]/uv[:,2:]
init_uvz=np.concatenate([uv,o[:,2:]],1)
opt=TrajectoryOptimizer(Rd,K,Rcw,B,crop,imgs,oms,pms,phases,yaw_base,init_uvz)
fix_z='fixz' in sys.argv
opt.optimize(iters=int(sys.argv[1]),fix_z=fix_z)
print('time',time.time()-t0)
T,th,uvz=opt.result()
iou,ncc=opt.eval_frames()
np.savez('/work/scratch/joint_%s.npz'%('fixz' if fix_z else 'free'),T=T,th=th,uvz=uvz,iou=iou,ncc=ncc,phases=np.array(phases))
for a,b,m in phases:
    print(a,b,m,'iou %.4f ncc %.4f'%(iou[a:b+1].mean(),ncc[a:b+1].mean()),'z med %.4f'%np.median(uvz[a:b+1,2]),'base z med %.4f'%np.median(Tcm[a:b+1,2,3]),
          'dyaw med %.1f'%np.degrees(np. …[+68 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bt1avr1lw). Output is being written to: /tmp/claude-1003/-work/bca75961-5f31-445f-a8fa-85b4a814bcaa/tasks/bt1avr1lw.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/outputs; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/bca75961-5f31-445f-a8fa-85b4a814bcaa/tasks/bt1avr1lw.output; nvidia-smi --query-gpu=memory.used --format=csv
```

<details><summary>output</summary>

```
[(0, 210, False), (211, 281, True), (282, 308, False), (309, 427, True), (428, 552, False)]
load 35.33386492729187
it    0 iou 0.8963 ncc 0.2676 smooth 75.417
it   25 iou 0.9324 ncc 0.4309 smooth 72.724
it   50 iou 0.9370 ncc 0.4209 smooth 72.555
it   75 iou 0.9365 ncc 0.4291 smooth 72.514
it  100 iou 0.9367 ncc 0.4270 smooth 72.508
it  125 iou 0.9368 ncc 0.4281 smooth 72.505
it  149 iou 0.9368 ncc 0.4277 smooth 72.504
time 133.04355359077454
0 210 False iou 0.9370 ncc 0.2615 z med 3.0921 base z med 3.0921 dyaw med 3.4
211 281 True iou 0.9430 ncc 0.4645 z med 3.0958 base z med 3.0958 dyaw med 1.4
282 308 False iou 0.9416 ncc 0.5294 z med 3.0801 base z med 3.0801 dyaw med -2.4
309 427 True iou 0.9340 ncc 0.5823 z med 3.1182 base z med 3.1182 dyaw med 1.2
428 552 False iou 0.9344 ncc 0.5182  …[+99 chars]
```
</details>

> *thinking:* The joint optimiser's first full run (depth fixed at baseline) automatically identified phases S1 0–210, M1 211–281, P 282–308, M2 309–427, S2 428–552, raising mask IoU from 0.896 to 0.937, converging in ~50 iterations using 2.5 GB GPU memory. Next, I'll compare its turning angle to the baseline frame by frame.

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
J=np.load('/work/scratch/joint_fixz.npz')
th=J['th']; uvz=J['uvz']
o=Tcm[:,:3,3]; uv=(o@K.T); uv=uv[:,:2]/uv[:,2:]
for i in range(200,440,4): print(i,'yaw base %.1f ours %.1f diff %.1f'%(np.degrees(yaw_base[i]-yaw_base[0]),np.degrees(th[i]-yaw_base[0]),np.degrees(th[i]-yaw_base[i])),' du %.2f dv %.2f'%tuple(uvz[i,:2]-uv[i]),' iou %.3f ncc %.3f'%(J['iou'][i],J['ncc'][i]))
"
```

<details><summary>output</summary>

```
200 yaw base 1.0 ours 4.1 diff 3.1  du -1.51 dv -1.21  iou 0.944 ncc 0.254
204 yaw base 1.0 ours 4.1 diff 3.0  du -1.44 dv -1.38  iou 0.945 ncc 0.260
208 yaw base 1.6 ours 4.1 diff 2.5  du -1.39 dv -1.32  iou 0.946 ncc 0.263
212 yaw base 4.0 ours 6.8 diff 2.8  du -1.79 dv -0.04  iou 0.953 ncc 0.290
216 yaw base 8.1 ours 10.7 diff 2.5  du -1.57 dv 0.02  iou 0.953 ncc 0.325
220 yaw base 13.9 ours 15.9 diff 2.0  du -1.43 dv -0.69  iou 0.933 ncc 0.320
224 yaw base 18.3 ours 19.5 diff 1.2  du -1.68 dv -0.54  iou 0.938 ncc 0.340
228 yaw base 22.1 ours 23.3 diff 1.2  du -1.85 dv -0.43  iou 0.953 ncc 0.376
232 yaw base 25.8 ours 27.3 diff 1.5  du -1.90 dv -1.44  iou 0.957 ncc 0.415
236 yaw base 32.5 ours 33.9 diff 1.5  du -2.10 dv -2.00  iou 0.958 ncc 0.469
240 yaw base 38.9 ours 39.9 diff 1.0  du …[+3898 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; timeout 600 python run_joint.py 150 free 2>&1 | tail -8
```

<details><summary>output</summary>

```
it  125 iou 0.9441 ncc 0.5170 smooth 0.230
it  149 iou 0.9441 ncc 0.5163 smooth 0.223
time 95.87913608551025
0 210 False iou 0.9527 ncc 0.4886 z med 3.0442 base z med 3.0921 dyaw med 1.5
211 281 True iou 0.9467 ncc 0.4819 z med 3.0827 base z med 3.0958 dyaw med 1.2
282 308 False iou 0.9445 ncc 0.5365 z med 3.0553 base z med 3.0801 dyaw med -2.2
309 427 True iou 0.9377 ncc 0.5856 z med 3.1014 base z med 3.1182 dyaw med 1.1
428 552 False iou 0.9344 ncc 0.5121 z med 3.0579 base z med 3.0553 dyaw med 0.1
```
</details>

> *thinking:* The silhouette depth cue proved fragile, shifting results by 2-3 cm depending on fitted angle across phases. I'm now running a per-frame silhouette depth sweep to check whether this cue holds steady during still phases, given a baseline drift of 7 cm.

**Bash**
```
cd /work/scratch/py; cat > zsweep.py <<'EOF'
from common import *
J=np.load('/work/scratch/joint_free.npz'); th=J['th']; uvz=J['uvz']
D=DiffFitter(Rd,K)
dev='cuda'; Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32); Kinv=torch.tensor(np.linalg.inv(K),device=dev,dtype=torch.float32)
zg=np.arange(-0.12,0.1201,0.005)
best=np.zeros(553); curv=np.zeros(553)
for s in range(0,553,79):
    fr=list(range(s,min(553,s+79)))
    D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
    S=np.zeros((len(zg),len(fr)))
    with torch.no_grad():
        R=Bc[None]@rotz_t(torch.tensor(th[fr],device=dev,dtype=torch.float32))
        ray=torch.tensor(np.c_[uvz[fr,:2],np.ones(len(fr))],device=dev,dtype=torch.float32)@Kinv.T
        for k,dz in enumerate(zg):
            t=ray*torch.tensor(uvz[fr,2]+dz,device=dev,dtype=torch.float32)[:,None]
            iou,_=D.losses(R,t,torch.arange(len(fr)),w_ncc=0)
            S[k]=iou.cpu().numpy()
    k=S.argmax(0)
    # parabolic refinement
    for j,i in enumerate(fr):
        kk=min(max(k[j],1),len(zg)-2); y0,y1,y2=S[kk-1,j],S[kk,j],S[kk+1,j]
        den=y0-2*y1+y2; off=0.5*(y0-y2)/den if den<0 else 0
        best[i]=uvz[i,2]+zg[kk]+off*0.005
np.save('/work/scratch/zbest_iou.npy',best)
for i in range(0,553,8): print(i,'iou-z %.3f  base z %.3f'%(best[i],Tcm[i,2,3]))
EOF
python zsweep.py
```

<details><summary>output</summary>

```
0 iou-z 3.036  base z 3.049
8 iou-z 3.044  base z 3.042
16 iou-z 3.021  base z 3.038
24 iou-z 3.021  base z 3.017
32 iou-z 3.044  base z 3.037
40 iou-z 3.044  base z 3.076
48 iou-z 3.044  base z 3.118
56 iou-z 3.043  base z 3.066
64 iou-z 3.045  base z 3.080
72 iou-z 3.039  base z 3.091
80 iou-z 3.044  base z 3.098
88 iou-z 3.042  base z 3.086
96 iou-z 3.045  base z 3.106
104 iou-z 3.049  base z 3.109
112 iou-z 3.048  base z 3.105
120 iou-z 3.052  base z 3.102
128 iou-z 3.052  base z 3.104
136 iou-z 3.047  base z 3.096
144 iou-z 3.050  base z 3.095
152 iou-z 3.048  base z 3.097
160 iou-z 3.047  base z 3.087
168 iou-z 3.048  base z 3.110
176 iou-z 3.048  base z 3.092
184 iou-z 3.049  base z 3.089
192 iou-z 3.049  base z 3.099
200 iou-z 3.048  base z 3.095
208 iou-z 3.049  base z 3.091
216 i …[+1284 chars]
```
</details>

> *thinking:* Silhouette depth is stable within each still phase, but silhouette and baseline estimates disagree on whether S2 is farther or closer than S1 by about 6 cm, which matters for the thresholds in question. I'll check an independent cue—reconstructed hand position during contact—to resolve this.

**Bash**
```
cd /work/scratch/py; cat > handcue.py <<'EOF'
from common import *
from scipy.spatial import cKDTree
kp=r['human_keypoints_world'].astype(np.float64); cp=r['contact_prob_right_left']
vs=v[::20].astype(np.float64); tree=cKDTree(vs)
J=np.load('/work/scratch/joint_free.npz'); Tours=J['T']
def hand_dist(Tc, i, dz=0.0):
    # Tc camera-frame pose; shift along ray by dz (scale)
    T=Tc.copy(); s=(T[2,3]+dz)/T[2,3]; T[:3,3]*=s
    Tw=Twc@T; Tinv=np.linalg.inv(Tw)
    pts=np.r_[kp[i,21:42],kp[i,42:63]]
    pm=(Tinv[:3,:3]@pts.T).T+Tinv[:3,3]
    d,_=tree.query(pm); return d.min()
dzs=np.arange(-0.15,0.1501,0.005)
for i in list(range(130,560,10)):
    if i>552: break
    db=[hand_dist(Tcm[i],i,dz) for dz in dzs]
    k=int(np.argmin(db))
    print(i,'contact %.2f %.2f'%tuple(cp[i]),'hand-surface dist base %.3f, best dz %.3f (dist %.3f)  => z %.3f ; base z %.3f'%(db[30],dzs[k],db[k],Tcm[i,2,3]+dzs[k],Tcm[i,2,3]))
EOF
python handcue.py
```

<details><summary>output</summary>

```
130 contact 0.00 0.00 hand-surface dist base 0.011, best dz -0.125 (dist 0.001)  => z 2.981 ; base z 3.106
140 contact 0.00 0.98 hand-surface dist base 0.002, best dz -0.025 (dist 0.001)  => z 3.067 ; base z 3.092
150 contact 0.00 1.00 hand-surface dist base 0.002, best dz -0.075 (dist 0.001)  => z 3.034 ; base z 3.109
160 contact 0.00 1.00 hand-surface dist base 0.003, best dz -0.075 (dist 0.001)  => z 3.012 ; base z 3.087
170 contact 0.00 1.00 hand-surface dist base 0.002, best dz 0.075 (dist 0.001)  => z 3.188 ; base z 3.113
180 contact 0.00 1.00 hand-surface dist base 0.002, best dz -0.135 (dist 0.001)  => z 2.957 ; base z 3.092
190 contact 0.00 1.00 hand-surface dist base 0.004, best dz -0.060 (dist 0.001)  => z 3.058 ; base z 3.118
200 contact 0.00 1.00 hand-surface dist base 0.003,  …[+3790 chars]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
ref=np.load('/work/inputs/frame0_reference_human.npz'); print(ref.files)
vr=ref['vertices_world']; print(vr.shape)
hv=r['human_vertices_world'][0]; print(hv.shape)
vr4=vr[::4] if len(vr)//4+1>=len(hv) else vr
print(len(vr4))
n=min(len(vr4),len(hv))
d=hv[:n]-vr4[:n]; print('mean offset world',d.mean(0), 'rms',np.sqrt((d**2).sum(1).mean()))
# in camera frame
dc=(Tcw[:3,:3]@d.T).T; print('mean offset cam',dc.mean(0))
print('ref centroid cam', Tcw[:3,:3]@vr.mean(0)+Tcw[:3,3], 'recon centroid cam', Tcw[:3,:3]@hv.mean(0)+Tcw[:3,3])
"

```

<details><summary>output</summary>

```
['vertices_world']
(18056, 3)
(4610, 3)
18056
mean offset world [ 0.00963731 -0.02283254 -0.05512504] rms 0.71899736
mean offset cam [ 0.01277042 -0.02140494 -0.05506102]
ref centroid cam [-0.74979446  0.18428917  2.86774749] recon centroid cam [-0.73738481  0.15166193  2.81114978]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
vr=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
hv=r['human_vertices_world'][0].astype(np.float64)
def cam(x): return (Tcw[:3,:3]@x.T).T+Tcw[:3,3]
a=cam(vr); b=cam(hv)
for q in [2,10,50,90,98]: print(q, np.round(np.percentile(a,q,0),3), np.round(np.percentile(b,q,0),3))
# projected image bbox
pa=a@K.T; pa=pa[:,:2]/pa[:,2:]; pb=b@K.T; pb=pb[:,:2]/pb[:,2:]
print('img bbox ref',pa.min(0),pa.max(0),' recon',pb.min(0),pb.max(0))
"

```

<details><summary>output</summary>

```
2 [-0.986 -0.513  2.671] [-0.981 -0.553  2.618]
10 [-0.953 -0.467  2.735] [-0.941 -0.5    2.688]
50 [-0.755  0.288  2.875] [-0.74   0.246  2.793]
90 [-0.547  1.07   2.967] [-0.542  1.075  2.929]
98 [-0.516  1.108  3.074] [-0.505  1.137  2.998]
img bbox ref [446.28543291 385.69449229] [616.54214794 896.48184131]  recon [450.66711929 374.62958031] [617.08819772 908.93160077]
```
</details>

> *thinking:* The hand-distance cue isn't useful since it's pinned near zero, so instead I used table geometry: the cooker base shifted up ~2.4px between S1 and S2, implying S2 is ~4.8cm farther, which matches the silhouette cue but contradicts the baseline. I'll now compute this table-plane cue precisely for each still phase.

**Bash**
```
cd /work/scratch/py; cat > tablecue.py <<'EOF'
from common import *
J=np.load('/work/scratch/joint_fixz.npz'); th=J['th']; uvz=J['uvz']
zs=np.load('/work/scratch/zbest_iou.npy')
d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
bot_m=np.array([center[0],center[1],v[:,2].max()])
Kinv=np.linalg.inv(K)
def bottom_height(i,z):
    Rwm=B@np.array([[np.cos(th[i]),-np.sin(th[i]),0],[np.sin(th[i]),np.cos(th[i]),0],[0,0,1]])
    t=z*(Kinv@np.r_[uvz[i,:2],1]); pb=Rcw@Rwm@bot_m+t
    return nup@(Twc[:3,:3]@pb+Twc[:3,3])+d, pb
def z_for_height(i,h):
    # height linear in z (t scales along ray, rotation fixed)
    h1,_=bottom_height(i,3.0); h2,_=bottom_height(i,3.1)
    return 3.0+(h-h1)*(0.1)/(h2-h1)
occ=np.array([PM[i][545:657,690:818].sum() for i in range(553)])
for name,(a,b) in dict(S1=(0,130),S1b=(140,210),P=(282,308),S2a=(428,460),S2=(465,552)).items():
    fr=np.arange(a,b+1)
    zsil=np.median(zs[fr])
    hs=[bottom_height(i,zsil)[0] for i in fr]
    print(name,'sil z %.4f  bottom height at sil z %.4f  base z med %.4f  uv %s'%(zsil,np.median(hs),np.median(Tcm[fr,2,3]),np.round(np.median(uvz[fr,:2],0),2)))
    hs_ref=0.8137
    print('    z for height %.4f: %.4f'%(hs_ref, np.median([z_for_height(i,hs_ref) for i in fr])))
EOF
python tablecue.py
```

<details><summary>output</summary>

```
S1 sil z 3.0390  bottom height at sil z 0.8107  base z med 3.0755  uv [755.7  607.63]
    z for height 0.8137: 3.0176
S1b sil z 3.0483  bottom height at sil z 0.8094  base z med 3.0965  uv [755.7  607.63]
    z for height 0.8137: 3.0176
P sil z 3.0061  bottom height at sil z 0.8200  base z med 3.0801  uv [753.94 606.39]
    z for height 0.8137: 3.0513
S2a sil z 3.0694  bottom height at sil z 0.8104  base z med 3.1086  uv [757.57 606.56]
    z for height 0.8137: 3.0453
S2 sil z 3.0700  bottom height at sil z 0.8103  base z med 3.0493  uv [757.57 606.56]
    z for height 0.8137: 3.0453
```
</details>

> *thinking:* The silhouette and table-plane cues agree closely (within 0.5 cm) that S2 is deeper than the baseline suggests, so I'll use that relative offset while keeping the episode anchored to the baseline's median depth. For the ambiguous held pause, I'll average the two conflicting cue values, then check both cues frame by frame through the turning phases.

**Bash**
```
cd /work/scratch/py; cat >> tablecue.py <<'EOF'
from scipy.ndimage import median_filter, gaussian_filter1d
zt=np.array([z_for_height(i,0.8137) for i in range(553)])
np.save('/work/scratch/ztable.npy',zt)
zsm=gaussian_filter1d(median_filter(zs,9,mode='nearest'),3,mode='nearest'); ztm=gaussian_filter1d(median_filter(zt,9,mode='nearest'),3,mode='nearest')
for i in range(200,470,6): print(i,'sil %.3f table %.3f  | smoothed sil %.3f table %.3f  base %.3f occ %d'%(zs[i],zt[i],zsm[i],ztm[i],Tcm[i,2,3],occ[i]))
EOF
python tablecue.py | tail -45
```

<details><summary>output</summary>

```
200 sil 3.048 table 3.018  | smoothed sil 3.048 table 3.018  base 3.095 occ 1503
206 sil 3.048 table 3.018  | smoothed sil 3.049 table 3.017  base 3.099 occ 1510
212 sil 3.069 table 3.000  | smoothed sil 3.055 table 3.007  base 3.101 occ 1651
218 sil 3.018 table 3.004  | smoothed sil 3.042 table 3.005  base 3.098 occ 1694
224 sil 3.014 table 3.017  | smoothed sil 3.030 table 3.015  base 3.097 occ 1587
230 sil 3.041 table 3.022  | smoothed sil 3.037 table 3.022  base 3.105 occ 1603
236 sil 3.036 table 3.031  | smoothed sil 3.036 table 3.030  base 3.092 occ 1606
242 sil 3.032 table 3.037  | smoothed sil 3.034 table 3.036  base 3.103 occ 1585
248 sil 3.044 table 3.035  | smoothed sil 3.025 table 3.037  base 3.092 occ 1547
254 sil 2.990 table 3.040  | smoothed sil 3.004 table 3.040  base 3.156 …[+2843 chars]
```
</details>

> *thinking:* Both depth cues agree on the still phases, with the silhouette cue noisier during hand movement, so I'll fuse them by aligning offsets on still phases, smoothing through turns, and anchoring to the baseline's median depth. Now writing `solve.py` to run the full pipeline end to end.

**Write**
```
/work/outputs/solve.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/solve.py (file state is current in your context — no need to Read it back)
```
</details>

**Edit**
```
/work/outputs/solve.py
```

<details><summary>output</summary>

```
The file /work/outputs/solve.py has been updated successfully. (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/outputs; timeout 1200 python solve.py --inputs /work/inputs --out /work/scratch/test_out.npz 2>&1 | grep -v "^it" | tail -40
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: b40m9kdxb). Output is being written to: /tmp/claude-1003/-work/bca75961-5f31-445f-a8fa-85b4a814bcaa/tasks/b40m9kdxb.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch/py; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 240; cat /tmp/claude-1003/-work/bca75961-5f31-445f-a8fa-85b4a814bcaa/tasks/b40m9kdxb.output
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
cat /tmp/claude-1003/-work/bca75961-5f31-445f-a8fa-85b4a814bcaa/tasks/b40m9kdxb.output | grep -v " it "
```

<details><summary>output</summary>

```
[17:11:50] crop (693, 551, 128, 112)
[17:11:52] loaded 553 frames
[17:11:53] phases (start, end, moving): [(0, 211, False), (212, 281, True), (282, 310, False), (311, 426, True), (427, 552, False)]
[17:11:53] baseline tilt from upright: median 1.15 deg, max 4.50 deg
[17:13:06] depth re-anchoring scale 1.0225 (shift 6.8 cm)
[17:13:06]   phase 0-211 moving=False: z_sil med 3.014 z_tab med 3.084 final z med 3.082 baseline z med 3.092
[17:13:06]   phase 212-281 moving=True: z_sil med 3.023 z_tab med 3.090 final z med 3.086 baseline z med 3.095
[17:13:06]   phase 282-310 moving=False: z_sil med 3.010 z_tab med 3.110 final z med 3.093 baseline z med 3.080
[17:13:06]   phase 311-426 moving=True: z_sil med 3.048 z_tab med 3.100 final z med 3.109 baseline z med 3.118
[17:13:06]   phase 427-552 movi …[+2404 chars]
```
</details>

> *thinking:* solve.py completed in 121s, with fitted poses outperforming the baseline on both mask IoU (0.93–0.95 vs 0.88–0.90) and texture score across all phases. Depth remains the key uncertainty—S2 lands 6.4 cm farther than baseline, consistent in direction with the table (+1.7 cm) and silhouette (+5.6 cm) cues—so I'll cross-check it with a model-free pixel measurement of the cooker's bottom edge against the table.

**Bash**
```
cd /work/scratch/py; cat > bottomedge.py <<'EOF'
from common import *
# vertical intensity profile in central columns; locate cooker->table transition near the bottom with subpixel accuracy
def edge_rows(i, cols, rlo=625, rhi=650):
    im=frame(i).astype(np.float32)
    # use blue channel minus red: cooker is white (b~r), table is tan (r>>b)
    ch=im[:,:,0]-im[:,:,2]
    out=[]
    for c in cols:
        p=ch[rlo:rhi,c-1:c+2].mean(1)
        g=np.diff(p)  # going down: white (b-r ~0) -> tan (b-r negative): g negative
        k=np.argmin(g)
        if 0<k<len(g)-1:
            y0,y1,y2=g[k-1],g[k],g[k+1]; den=y0-2*y1+y2; off=0.5*(y0-y2)/den if den!=0 else 0
        else: off=0
        out.append(rlo+k+0.5+off)
    return np.array(out)
for name,fr in dict(S1=range(0,130,4),S1b=range(140,210,4),P=range(282,310,2),S2=range(465,552,4)).items():
    for cols in [range(740,772,2)]:
        e=np.array([edge_rows(i,cols) for i in fr])
        print(name,'bottom edge row median %.2f  per-col mean %s'%(np.median(e),np.round(np.median(e,0),1)))
EOF
python bottomedge.py
```

<details><summary>output</summary>

```
S1 bottom edge row median 639.35  per-col mean [635.6 639.4 639.4 639.4 639.4 639.4 639.2 639.  639.1 639.  639.2 639.3
 639.3 639.3 639.4 639.5]
S1b bottom edge row median 639.32  per-col mean [635.5 639.4 639.4 639.4 639.4 639.3 639.  639.  639.  639.  639.2 639.2
 639.3 639.4 639.4 639.5]
P bottom edge row median 635.55  per-col mean [635.9 635.5 635.6 635.7 636.3 636.4 635.1 635.6 635.6 639.3 639.4 635.4
 635.5 635.3 635.6 634.5]
S2 bottom edge row median 635.58  per-col mean [635.  634.5 634.5 634.6 634.8 635.7 635.9 635.1 635.5 636.3 636.2 636.2
 636.3 636.3 639.4 635.4]
```
</details>

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
a=frame(50)[600:660,720:800]; b=frame(500)[600:660,720:800]; c=frame(295)[600:660,720:800]
o=np.hstack([a,b,c]); o=cv2.resize(o,None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST)
for y in range(0,o.shape[0],80): cv2.line(o,(0,y),(o.shape[1],y),(0,0,255) if (600+y//8)%10==0 else (80,80,80),1)
for y in range(0,o.shape[0],80): cv2.putText(o,str(600+y//8),(2,y+14),0,0.5,(0,0,255),1)
cv2.imwrite('/work/scratch/bottom.jpg',o)
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/bottom.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The camera is confirmed static since the table's front edge hasn't shifted; now I'll convert the measured pixel shifts (cooker contact line ~2.2px, P ~0.7px) into depth values.

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
Kinv=np.linalg.inv(K)
def depth_at(u,vv,h):
    ray=Kinv@np.array([u,vv,1.]) # cam
    rw=Twc[:3,:3]@ray; ow=Twc[:3,3]
    # nup.(ow+s rw)+d = h
    s=(h-d-nup@ow)/(nup@rw); return s*ray[2]
# find h so that the S1 contact row 639.75 at column 755 has depth 3.08
from scipy.optimize import brentq
h=brentq(lambda h: depth_at(755,639.75,h)-3.03,0.3,1.3)
print('h',h)
for vv in [639.75,639.0,637.5]: print(vv, depth_at(755,vv,h))
"

```

<details><summary>output</summary>

```
h 0.8027880191308584
639.75 3.03
639.0 3.045513432870965
637.5 3.077021796623051
```
</details>

> *thinking:* The contact-edge measurement confirms S2 is farther than S1 (4.7 cm vs. P's 1.5 cm), consistent with other depth cues but contradicting the baseline's estimate. I'll add this as a third depth cue in `solve.py`, starting with a per-frame detector and noise check.

**Bash**
```
cd /work/scratch/py; cat > contact.py <<'EOF'
from common import *
J=np.load('/work/scratch/test_out.npz'); Tw=J['T_world_mesh']; Tc=Tcw@Tw
o=Tc[:,:3,3]; pu=o@K.T; pu=pu[:,:2]/pu[:,2:]
def contact_row(img, uc, vguess):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    g=cv2.GaussianBlur(g,(3,3),0.7)
    rows=[]
    for c in range(int(round(uc))-6,int(round(uc))+7):
        lo=int(vguess)-8; p=g[lo:lo+17,c]
        k=int(np.argmin(p))
        if 0<k<len(p)-1:
            y0,y1,y2=p[k-1],p[k],p[k+1]; den=y0-2*y1+y2; off=0.5*(y0-y2)/den if den>0 else 0
            rows.append(lo+k+off)
    return np.median(rows) if rows else np.nan
vc=np.array([contact_row(frame(i),pu[i,0]+2,639) for i in range(553)])
np.save('/work/scratch/contact_v.npy',vc)
for a,b in [(0,211),(212,281),(282,310),(311,426),(427,552)]:
    s=vc[a:b+1]; print(a,b,'median %.2f std %.2f p10 %.2f p90 %.2f'%(np.nanmedian(s),np.nanstd(s),np.nanpercentile(s,10),np.nanpercentile(s,90)))
print(' '.join('%d:%.1f'%(i,vc[i]) for i in range(200,440,4)))
EOF
python contact.py
```

<details><summary>output</summary>

```
0 211 median 639.41 std 0.02 p10 639.38 p90 639.44
212 281 median 638.95 std 0.20 p10 638.64 p90 639.29
282 310 median 638.59 std 0.01 p10 638.58 p90 638.60
311 426 median 637.41 std 0.48 p10 637.06 p90 638.47
427 552 median 637.12 std 0.01 p10 637.11 p90 637.13
200:639.4 204:639.4 208:639.4 212:639.4 216:639.3 220:639.2 224:639.1 228:639.1 232:639.0 236:639.0 240:638.9 244:638.9 248:639.0 252:638.9 256:638.9 260:639.0 264:638.9 268:638.9 272:638.7 276:638.6 280:638.6 284:638.6 288:638.6 292:638.6 296:638.6 300:638.6 304:638.6 308:638.6 312:638.6 316:638.6 320:638.6 324:638.4 328:638.2 332:638.0 336:637.9 340:637.9 344:637.8 348:637.5 352:637.2 356:637.0 360:637.1 364:637.1 368:637.1 372:637.1 376:637.6 380:637.8 384:637.7 388:637.7 392:637.3 396:637.4 400:637.4 404:637.4 408:637.4 412:637 …[+62 chars]
```
</details>

> *thinking:* The contact-edge measurements show the cooker steadily moving away from the camera (~1.7 cm by P, ~4.8 cm by S2), so I'll use this as the primary depth cue in solve.py, keeping silhouette and fitted-position as cross-checks.

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
a=s.index("    # ---------------- 4b. table-plane depth ----------------")
b=s.index("    # ---------------- 5. re-anchor")
new='''    # ---------------- 4b. table-plane depth cues ----------------
    # bottom centre of the cooker in mesh coordinates (mesh -z is up, so the bottom is at max z)
    bot_m = np.array([(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].max()])

    def pose_np(i, z):
        c, s_ = np.cos(th1[i]), np.sin(th1[i])
        Rcm_i = Rcw @ B @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        return Rcm_i, z * (Kinv @ np.r_[uvz1[i, :2], 1])

    def plane_depth(u_, v_, h):
        """camera depth of the point at pixel (u_,v_) lying at height h above the floor plane"""
        ray = Kinv @ np.array([u_, v_, 1.0]); rw = Twc[:3, :3] @ ray
        s_ = (h - d_pl - nup @ Twc[:3, 3]) / (nup @ rw)
        return s_ * ray[2]

    # (i) contact line: the dark contact shadow between the round base and the table top. Its image row at the
    # base centre column is measured to ~0.02 px in static phases. The table top is a plane parallel to the floor
    # (assumption), so the row gives the depth of the base's front point up to a constant (unknown table height,
    # base radius) that the re-anchoring removes.
    gray = [cv2.GaussianBlur(cv2.cvtColor(im[y0:y0 + CROP_H, x0:x0 + CROP_W], cv2.COLOR_BGR2GRAY).astype(np.float32),
                             (3, 3), 0.7)  …[+3532 chars]
```

<details><summary>output</summary>

```
145:    z_sil = np.zeros(F)
161:            z_sil[idx] = uvz1[idx, 2] + zg[k] + off * 0.005
235:    for name, zc in (("contact", z_ct), ("silhouette", z_sil), ("fit_table", z_tab), ("baseline", o[:, 2])):
297:                   z_sil=z_sil.tolist(), z_tab=z_tab.tolist(), z_contact=z_ct.tolist(), z_final=z_fin.tolist()),
```
</details>

**Bash**
```
cd /work/outputs; timeout 1200 python solve.py --inputs /work/inputs --out /work/scratch/test_out2.npz > /work/scratch/run2.log 2>&1; grep -v " it " /work/scratch/run2.log | cut -c1-400
```

<details><summary>output</summary>

```
Command running in background with ID: bih4h4ddb. Output is being written to: /tmp/claude-1003/-work/bca75961-5f31-445f-a8fa-85b4a814bcaa/tasks/bih4h4ddb.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work/outputs; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Agent:** While that runs, I'm writing a held-out validation. It refits with every 4th frame excluded from the data term, then scores the interpolated poses on those unseen frames against the baseline.

**Bash**
```
mkdir -p /work/outputs/validation; cat > /work/outputs/validation/heldout.py <<'EOF'
"""Held-out validation: refit theta,u,v with every 4th frame removed from the data term (weight 0) and compare
image agreement on those held-out frames with the baseline. Depth fixed to the final solve.py depths."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, cv2, h5py
from hoi_lib import Renderer, load_camera, load_mesh
from traj_opt import TrajectoryOptimizer
from solve import upright_basis, yaw_of
inp = "/work/inputs"
res = np.load(sys.argv[1]); valj = json.load(open(sys.argv[2]))
K, Twc, _, W, H = load_camera(inp); Tcw = np.linalg.inv(Twc); Rcw = Tcw[:3, :3]
plane = json.load(open(f"{inp}/ground_plane.json"))["plane"]; nup = np.array(plane[:3]); nup /= np.linalg.norm(nup)
if plane[3] < 0: nup = -nup
rec = np.load(f"{inp}/reconstruction.npz"); Twm_b = rec["T_world_mesh"].astype(np.float64); F = len(Twm_b)
Tcm_b = Tcw[None] @ Twm_b; Tcm_o = Tcw[None] @ res["T_world_mesh"]
v, f, uv, tex = load_mesh(f"{inp}/object_mesh.glb"); Rd = Renderer(v, f, uv, tex)
o = Tcm_b[:, :3, 3]; pu = o @ K.T; pu = pu[:, :2] / pu[:, 2:]
cu, cv_ = np.median(pu, 0); x0 = int(round(cu - 64)); y0 = int(round(cv_ - 56)); crop = (x0, y0, 128, 112)
cap = cv2.VideoCapture(f"{inp}/video.mp4"); imgs = []
while True:
    ok, fr = cap.read()
    if not ok: break
    full = np.zeros((H, W, 3), np.uint8); full[y0:y0+112, x0:x0+128] = fr[y0 …[+1948 chars]
```

<details><summary>output</summary>

```
[17:18:16] crop (693, 551, 128, 112)
[17:18:17] loaded 553 frames
[17:18:18] phases (start, end, moving): [(0, 211, False), (212, 281, True), (282, 310, False), (311, 426, True), (427, 552, False)]
[17:18:18] baseline tilt from upright: median 1.15 deg, max 4.50 deg
[17:20:02] cue contact   : phase depth relative to first static phase (cm): +0.0 +1.0 +1.8 +4.1 +4.8
[17:20:02] cue silhouette: phase depth relative to first static phase (cm): +0.0 +0.7 -0.5 +3.4 +5.4
[17:20:02] cue fit_table : phase depth relative to first static phase (cm): +0.0 +0.8 +2.8 +1.8 +1.9
[17:20:02] cue baseline  : phase depth relative to first static phase (cm): +0.0 +0.3 -1.2 +2.6 -3.6
[17:20:02] depth re-anchoring scale 1.0078 (shift 2.4 cm)
[17:20:02]   phase 0-211 moving=False: final z med 3.076 baseline z med …[+324 chars]
```
</details>

> *thinking:* I cross-checked depth cues for S2 against S1, and three independent measures (contact edge, silhouette, fitted table position) consistently place S2 farther back, with contact edge and silhouette agreeing within 0.6 cm. Anchoring to the baseline's episode median, S2 is 6.7 cm farther than the baseline's S2 while S1 is 1.6 cm closer.

**Bash**
```
sleep 45; grep -v " it " /work/scratch/run2.log | tail -7 | cut -c1-500
```

<details><summary>output</summary>

```
[17:20:58] wrote /work/scratch/test_out2.npz (553, 4, 4)
[17:20:59] {'phase': '0-211', 'moving': False, 'iou_base': 0.8875399827957153, 'iou_ours': 0.9413352608680725, 'ncc_base': 0.20386163890361786, 'ncc_ours': 0.2807104289531708, 'rot_step_base_deg': 0.5060127988360633, 'rot_step_ours_deg': 0.0262929686087781, 'trans_step_base_mm': 6.747222229033753, 'trans_step_ours_mm': 0.0, 'dz_mean_cm': -0.5345418563115358, 'dz_median_cm': -1.656316150837145, 'yaw_diff_median_deg': 3.123445829031245}
[17:20:59] {'phase': '212-281', 'moving': True, 'iou_base': 0.9015529155731201, 'iou_ours': 0.9472641944885254, 'ncc_base': 0.27783727645874023, 'ncc_ours': 0.4798642694950104, 'rot_step_base_deg': 1.7268692658853884, 'rot_step_ours_deg': 1.4220380850040943, 'trans_step_base_mm': 6.529716401026874, 'tra …[+1498 chars]
```
</details>

**Bash**
```
cd /work/outputs; (timeout 900 python validation/heldout.py /work/scratch/test_out2.npz /work/scratch/solve_validation.json > /work/scratch/heldout.log 2>&1 &) ; ls /work/scratch/*.json; cd /work/scratch/py; python -c "
from common import *
from viz import show
O=np.load('/work/scratch/test_out2.npz')['T_world_mesh']; To=Tcw@O
show([Tcm,To],[100,230,260,295,340,370,400,445,520],'/work/scratch/cmp.jpg')
"
```

<details><summary>output</summary>

```
/work/scratch/solve_validation.json
```
</details>

**Read**
```
/work/scratch/cmp.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
sleep 150; cat /work/scratch/heldout.log
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
sleep 60; cat /work/scratch/heldout.log
```

<details><summary>output</summary>

```
phase | held-out frames | IoU baseline | IoU ours(held-out fit) | NCC baseline | NCC ours(held-out fit)
0-211 (static) | 53 | 0.886 | 0.946 | 0.199 | 0.311
212-281 (moving) | 17 | 0.898 | 0.947 | 0.270 | 0.482
282-310 (static) | 8 | 0.888 | 0.939 | 0.293 | 0.521
311-426 (moving) | 29 | 0.902 | 0.931 | 0.416 | 0.584
427-552 (static) | 31 | 0.882 | 0.926 | 0.265 | 0.418
all | 138 | 0.890 | 0.938 | 0.274 | 0.426
fraction of held-out frames where ours(held-out) IoU > baseline IoU: 0.99
fraction of held-out frames where ours(held-out) NCC > baseline NCC: 0.98
```
</details>

> *thinking:* My poses beat the baseline on held-out validation frames (99% on mask IoU, 98% on texture score), including hand-occluded turns. Now I'll verify the cooker's bottom height stays constant since it's resting on the table.

**Bash**
```
cd /work/scratch/py; cat > /work/outputs/validation/physics.py <<'EOF'
"""Physical plausibility: height of the object's lowest point above the floor plane and its world-position spread
per phase (object rests on a table: height should be constant; static phases: position constant)."""
import sys, json, numpy as np, trimesh
inp = "/work/inputs"
res = np.load(sys.argv[1]); valj = json.load(open(sys.argv[2]))
rec = np.load(f"{inp}/reconstruction.npz"); Tb = rec["T_world_mesh"].astype(np.float64); To = res["T_world_mesh"]
pl = json.load(open(f"{inp}/ground_plane.json"))["plane"]; n = np.array(pl[:3]); d = pl[3]
m = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry(); v = np.asarray(m.vertices)[::30]
def heights(T):
    w = np.einsum("fij,nj->fni", T[:, :3, :3], v) + T[:, None, :3, 3]
    s = w @ n + d
    return np.abs(s).min(1)  # distance of the lowest point to the floor
hb, ho = heights(Tb), heights(To)
print("phase | bottom height above floor (cm) baseline mean±std | ours mean±std | world centre spread (cm, std) baseline | ours")
for a, b, mv in valj["phases"]:
    sl = slice(a, b + 1)
    sb = np.linalg.norm(Tb[sl, :3, 3].std(0)) * 100; so = np.linalg.norm(To[sl, :3, 3].std(0)) * 100
    print(f"{a}-{b} {'moving' if mv else 'static'} | {100*hb[sl].mean():.1f}±{100*hb[sl].std():.1f} | {100*ho[sl].mean():.1f}±{100*ho[sl].std():.1f} | {sb:.1f} | {so:.1f}")
print(f"all | {100*hb.mean():.1f}±{100*hb.std():.1f} | {100*ho.mean():.1f}±{100*ho.std():.1f …[+114 chars]
```

<details><summary>output</summary>

```
phase | bottom height above floor (cm) baseline mean±std | ours mean±std | world centre spread (cm, std) baseline | ours
0-211 static | 80.4±0.8 | 80.7±0.0 | 3.2 | 0.0
212-281 moving | 80.1±0.4 | 80.6±0.2 | 2.2 | 0.5
282-310 static | 80.3±0.5 | 80.8±0.0 | 3.3 | 0.0
311-426 moving | 80.1±0.4 | 80.4±0.2 | 2.7 | 1.2
427-552 static | 81.6±0.7 | 80.4±0.0 | 3.9 | 0.0
all | 80.6±0.8 | 80.6±0.2
```
</details>

> *thinking:* My trajectory keeps the cooker's bottom consistently 80.4–80.8 cm above the floor, while the baseline shows it floating 1.2 cm higher in S2 than S1, confirming its depth issue. Now running the official `solve.py`.

**Bash**
```
cd /work; rm -f /work/outputs/object_poses_world.npz; ( time timeout 1500 python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > /work/outputs/solve_log.txt 2>&1 ) 2> /work/scratch/time.txt; cat /work/scratch/time.txt; grep -v " it " /work/outputs/solve_log.txt | cut -c1-200
```

<details><summary>output</summary>

```
Command running in background with ID: b3a73mfvg. Output is being written to: /tmp/claude-1003/-work/bca75961-5f31-445f-a8fa-85b4a814bcaa/tasks/b3a73mfvg.output. You will be notified when it completes. To check interim output, use Read on that file path.
Session cwd remains /work/scratch/py; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
python -c "
import torch, numpy, scipy, cv2, trimesh, h5py, nvdiffrast, PIL
for m in [torch,numpy,scipy,cv2,trimesh,h5py,PIL]: print(m.__name__, m.__version__)
print('nvdiffrast', getattr(nvdiffrast,'__version__','?'))
"; python --version; pip list 2>/dev/null | grep -i -E "nvdiffrast|opencv|torch " 
```

<details><summary>output</summary>

```
torch 2.5.1+cu124
numpy 1.26.3
scipy 1.16.3
cv2 4.11.0
trimesh 5.1.1
h5py 3.16.0
PIL 10.2.0
nvdiffrast 0.4.0
Python 3.11.10
nvdiffrast                               0.4.0
opencv-contrib-python                    4.11.0.86
opencv-python                            4.11.0.86
torch                                    2.5.1+cu124
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
sleep 170; cat /work/scratch/time.txt; grep -v " it " /work/outputs/solve_log.txt | cut -c1-200
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
cat /work/scratch/time.txt; grep -v " it " /work/outputs/solve_log.txt | cut -c1-220; python -c "
import numpy as np
a=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; b=np.load('/work/scratch/test_out2.npz')['T_world_mesh']
print(a.shape,a.dtype,np.isfinite(a).all(), 'max diff vs previous run', np.abs(a-b).max())
R=a[:,:3,:3]; print('orthonormal err', np.abs(np.einsum('fji,fjk->fik',R,R)-np.eye(3)).max(), 'det', np.linalg.det(R).min())
"; python /work/outputs/validation/physics.py /work/outputs/object_poses_world.npz /work/outputs/solve_validation.json
```

<details><summary>output</summary>

```
real	2m5.245s
user	2m8.234s
sys	0m11.883s
[17:24:48] crop (693, 551, 128, 112)
[17:24:50] loaded 553 frames
[17:24:50] phases (start, end, moving): [(0, 211, False), (212, 281, True), (282, 310, False), (311, 426, True), (427, 552, False)]
[17:24:50] baseline tilt from upright: median 1.15 deg, max 4.50 deg
[17:26:07] cue contact   : phase depth relative to first static phase (cm): +0.0 +1.0 +1.8 +4.1 +4.8
[17:26:07] cue silhouette: phase depth relative to first static phase (cm): +0.0 +0.5 -0.6 +3.1 +3.4
[17:26:07] cue fit_table : phase depth relative to first static phase (cm): +0.0 +0.9 +2.7 +1.8 +1.8
[17:26:07] cue baseline  : phase depth relative to first static phase (cm): +0.0 +0.3 -1.2 +2.6 -3.6
[17:26:07] depth re-anchoring scale 1.0078 (shift 2.4 cm)
[17:26:07]   phase 0-211 movi …[+2084 chars]
```
</details>

> *thinking:* The run completed in 125s with a valid output of 553 finite poses, but I noticed up to 0.009 variance in matrix entries compared to the previous identical run, likely due to nondeterministic GPU gradient accumulation in nvdiffrast. Now I'll measure how significant this discrepancy actually is.

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
a=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; b=np.load('/work/scratch/test_out2.npz')['T_world_mesh']
pts=v[::100].astype(np.float64)
def add(A,B): return np.linalg.norm(np.einsum('fij,nj->fni',A[:,:3,:3],pts)+A[:,None,:3,3]-np.einsum('fij,nj->fni',B[:,:3,:3],pts)-B[:,None,:3,3],axis=2).mean(1)
d=add(a,b); print('ADD between runs: mean %.2f mm max %.2f mm'%(d.mean()*1000,d.max()*1000))
d=add(a,Twm); print('ADD ours vs baseline: mean %.1f mm median %.1f max %.1f'%(d.mean()*1000,np.median(d)*1000,d.max()*1000))
for p,(s,e) in dict(S1=(0,211),M1=(212,281),P=(282,310),M2=(311,426),S2=(427,552)).items(): print(p,'%.1f mm'%(1000*d[s:e+1].mean()))
"
```

<details><summary>output</summary>

```
ADD between runs: mean 0.44 mm max 1.50 mm
ADD ours vs baseline: mean 34.0 mm median 25.2 max 138.1
S1 29.8 mm
M1 22.3 mm
P 35.1 mm
M2 20.7 mm
S2 59.6 mm
```
</details>

> *thinking:* Run-to-run variation (0.44 mm mean, 1.5 mm max) is negligible compared to my trajectory's 34 mm average shift from baseline, mostly in S2 (60 mm). I'll now check the largest differences (up to 138 mm) to confirm they're baseline outliers rather than mistakes on my part.

**Bash**
```
cd /work/scratch/py; python -c "
from common import *
a=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; To=Tcw@a
pts=v[::100].astype(np.float64)
d=np.linalg.norm(np.einsum('fij,nj->fni',a[:,:3,:3],pts)+a[:,None,:3,3]-np.einsum('fij,nj->fni',Twm[:,:3,:3],pts)-Twm[:,None,:3,3],axis=2).mean(1)
for i in np.argsort(-d)[:8]: print(i,'%.0f mm'%(d[i]*1000),'dz %.1f cm'%(100*(To[i,2,3]-Tcm[i,2,3])),'rot diff %.1f'%rot_angle_deg(a[i,:3,:3],Twm[i,:3,:3]))
"
```

<details><summary>output</summary>

```
483 138 mm dz 13.7 cm rot diff 1.3
491 125 mm dz 12.4 cm rot diff 0.8
481 104 mm dz 10.3 cm rot diff 0.6
534 103 mm dz 10.3 cm rot diff 1.5
484 102 mm dz 10.1 cm rot diff 1.0
533 102 mm dz 10.1 cm rot diff 1.8
531 97 mm dz 9.7 cm rot diff 2.0
532 96 mm dz 9.5 cm rot diff 1.0
```
</details>

> *thinking:* These S2 frame spikes to 2.99–3.02 m are baseline outliers during a still phase, not issues with my output. I'll save the validation logs and write up the report now.

**Bash**
```
cp /work/scratch/heldout.log /work/outputs/validation/heldout_result.txt; python /work/outputs/validation/physics.py /work/outputs/object_poses_world.npz /work/outputs/solve_validation.json > /work/outputs/validation/physics_result.txt; python -c "
import json; j=json.load(open('/work/outputs/solve_validation.json'))
print('| phase | IoU base | IoU ours | NCC base | NCC ours | rot step base/ours (deg, median) | trans step base/ours (mm, median) | dz mean / median (cm) | yaw diff (deg) |')
for r in j['validation']:
    print('| %s %s | %.3f | %.3f | %.3f | %.3f | %.2f / %.2f | %.1f / %.1f | %+.1f / %+.1f | %+.1f |'%(r['phase'],'moving' if r['moving'] else 'static',r['iou_base'],r['iou_ours'],r['ncc_base'],r['ncc_ours'],r['rot_step_base_deg'],r['rot_step_ours_deg'],r['trans_step_base_mm'],r['trans_step_ours_mm'],r['dz_mean_cm'],r['dz_median_cm'],r['yaw_diff_median_deg']))
print(j['runtime_s'])
"; ls /work/outputs
```

<details><summary>output</summary>

```
| phase | IoU base | IoU ours | NCC base | NCC ours | rot step base/ours (deg, median) | trans step base/ours (mm, median) | dz mean / median (cm) | yaw diff (deg) |
| 0-211 static | 0.888 | 0.942 | 0.204 | 0.278 | 0.51 / 0.02 | 6.7 / 0.0 | -0.5 / -1.7 | +3.1 |
| 212-281 moving | 0.902 | 0.947 | 0.278 | 0.480 | 1.73 / 1.46 | 6.5 / 0.5 | -1.0 / -1.2 | +1.2 |
| 282-310 static | 0.894 | 0.940 | 0.292 | 0.521 | 0.78 / 0.03 | 28.4 / 0.0 | +1.5 / +1.3 | -2.1 |
| 311-426 moving | 0.902 | 0.933 | 0.411 | 0.587 | 1.82 / 1.64 | 6.7 / 0.7 | +0.3 / -0.1 | +1.2 |
| 427-552 static | 0.884 | 0.927 | 0.280 | 0.417 | 0.71 / 0.00 | 10.1 / 0.0 | +5.3 / +6.8 | +0.3 |
122.69750308990479
TOOLS.md
__pycache__
hoi_lib.py
object_poses_world.npz
solve.py
solve_log.txt
solve_validation.json
traj_opt.py
validation
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

**Agent:** I've produced a full trajectory that I expect to beat the baseline. It's in `/work/outputs/object_poses_world.npz`, with `REPORT.md` and `TOOLS.md` alongside. I ran `solve.py` once at the end as required: it took about 2 minutes and used about 2.5 GB of GPU memory. I estimate total GPU use during development at roughly 20 minutes. No outside tools or model weights were used.

**What happens in the video:** the person turns the rice cooker in place on the table, never visibly lifting it. It turns about −100°, pauses while held, then turns about −210° more. The script splits the video into five phases from the image itself: still (0–211), turning (212–281), held (282–310), turning (311–426), still (427–552). The baseline's orientation was basically right, and I found no 180° flips.

**What I changed:**
- **Still phases:** one fixed pose per phase, fitted against all its frames together. The baseline jitters there: its depth varies by 3–4 cm (standard deviation) and drifts 7 cm while nothing moves.
- **Orientation:** the cooker is kept upright relative to the floor, and its turning angle is fitted with the textured mesh against every frame, smoothly in time.
- **Position in the image:** shifted 1.5–3 px left, about 1 cm sideways; the baseline sits visibly off the cooker's outline.
- **Depth:** the overall distance to the camera is kept at the baseline's median, as the brief asked. Only the relative depth between phases changed.

**The decision with the most at stake:** where the cooker ends up. Three measurements from the image all say it was pushed 2–5 cm away from the camera while being turned; the baseline says it ended 3.6 cm closer. The main one is where the cooker's base meets the table in the image, measured very precisely in still frames. The cooker's size in the image agrees within 1.4 cm. As a result, the final still phase is about 6.7 cm farther away than the baseline has it. That rests on the table being level and the base's contact shadow not changing as it turns. If either fails, part of this change could be wrong, though the size cue needs neither assumption.

**Checks (no ground truth available):**

| Check | Baseline | Mine |
|---|---|---|
| Mask overlap, frames left out of the fit (IoU) | 0.890 | 0.938 |
| Texture match, same frames (NCC) | 0.274 | 0.426 |
| Cooker's bottom height, first vs final still phase | 80.4 vs 81.6 cm | 80.7 vs 80.5 cm |

On the left-out frames my poses are better in 99% (mask) and 98% (texture) of frames. The baseline leaves the cooker floating 1.2 cm above the table at the end; that check partly assumes a level table, like the depth cue.

**Confidence:** high for orientation (about ±2–3°) and image position, medium-high for the depth change between phases. The overall distance to the camera is the baseline's and I couldn't verify it.

**Reproducibility:** two complete runs differ by 0.44 mm on average (1.5 mm at most). The cause is nondeterministic GPU arithmetic in the renderer's gradient step, despite fixed seeds.

Files are in `/work/outputs`:
- object_poses_world.npz
- REPORT.md
- TOOLS.md
- solve.py

## Final

I've produced a full trajectory that I expect to beat the baseline. It's in `/work/outputs/object_poses_world.npz`, with `REPORT.md` and `TOOLS.md` alongside. I ran `solve.py` once at the end as required: it took about 2 minutes and used about 2.5 GB of GPU memory. I estimate total GPU use during development at roughly 20 minutes. No outside tools or model weights were used.

**What happens in the video:** the person turns the rice cooker in place on the table, never visibly lifting it. It turns about −100°, pauses while held, then turns about −210° more. The script splits the video into five phases from the image itself: still (0–211), turning (212–281), held (282–310), turning (311–426), still (427–552). The baseline's orientation was basically right, and I found no 180° flips.

**What I changed:**
- **Still phases:** one fixed pose per phase, fitted against all its frames together. The baseline jitters there: its depth varies by 3–4 cm (standard deviation) and drifts 7 cm while nothing moves.
- **Orientation:** the cooker is kept upright relative to the floor, and its turning angle is fitted with the textured mesh against every frame, smoothly in time.
- **Position in the image:** shifted 1.5–3 px left, about 1 cm sideways; the baseline sits visibly off the cooker's outline.
- **Depth:** the overall distance to the camera is kept at the baseline's median, as the brief asked. Only the relative depth between phases changed.

**The decision with the most at stake:** where the cooker ends up. Three measurements from the image all say it was pushed 2–5 cm away from the camera while being turned; the baseline says it ended 3.6 cm closer. The main one is where the cooker's base meets the table in the image, measured very precisely in still frames. The cooker's size in the image agrees within 1.4 cm. As a result, the final still phase is about 6.7 cm farther away than the baseline has it. That rests on the table being level and the base's contact shadow not changing as it turns. If either fails, part of this change could be wrong, though the size cue needs neither assumption.

**Checks (no ground truth available):**

| Check | Baseline | Mine |
|---|---|---|
| Mask overlap, frames left out of the fit (IoU) | 0.890 | 0.938 |
| Texture match, same frames (NCC) | 0.274 | 0.426 |
| Cooker's bottom height, first vs final still phase | 80.4 vs 81.6 cm | 80.7 vs 80.5 cm |

On the left-out frames my poses are better in 99% (mask) and 98% (texture) of frames. The baseline leaves the cooker floating 1.2 cm above the table at the end; that check partly assumes a level table, like the depth cue.

**Confidence:** high for orientation (about ±2–3°) and image position, medium-high for the depth change between phases. The overall distance to the camera is the baseline's and I couldn't verify it.

**Reproducibility:** two complete runs differ by 0.44 mm on average (1.5 mm at most). The cause is nondeterministic GPU arithmetic in the renderer's gradient step, despite fixed seeds.

Files are in `/work/outputs`:
- object_poses_world.npz
- REPORT.md
- TOOLS.md
- solve.py
