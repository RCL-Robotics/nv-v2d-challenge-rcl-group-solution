# CL2c_ep025

Claude Opus v2（ep22-29）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T15:42:58Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude/projects/-work/memory/ 2>/dev/null && cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null; nvidia-smi; python -c "import torch,nvdiffrast;print(torch.__version__)"
```

<details><summary>output</summary>

```
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 19:36 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 19:36 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 19:36 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 19:36 proxy.sock

inputs:
total 79808
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 19:36 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 19:36 ..
-rw-rw-r-- 1 hanzhou hanzhou        0 Oct 21 19:36 .preexported
-rw-rw-r-- 1 hanzhou hanzhou      802 Oct 21 19:36 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       72 Oct 21 19:36 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   194710 Oct 21 19:36 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      396 Oct 21 19:36 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  6954392 Oct 21 19:36 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou  797560 …[+4526 chars]
```
</details>

**Bash**
```
cd /work; cat logs/run_manifest.txt; echo $CUDA_VISIBLE_DEVICES; ls /pipeline /pipeline/*; python -c "
import numpy as np
r=np.load('inputs/reconstruction.npz')
for k in r: print(k, r[k].shape, r[k].dtype)
"
```

<details><summary>output</summary>

```
date_utc: 2026-10-21T15:42:58Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2c_ep025:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d_failed/episode_000025_out:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
ae5309669e9b659df57415fa8d241cf0742fbbe3ecdce4b748aaede178f1b4c7  ./camera.json
dd8502a3c652ed40144b48a9d59f0079953c7855f1be75aff7cab9554c458427  ./episode.json
4f7f44da6f2ec62a35d09d01a4ec2abd0d989463d57d …[+1147 chars]
```
</details>

**Bash**
```
cd /pipeline; find . -maxdepth 4 | head -50; du -sh *; ls /code/reconstruction
```

<details><summary>output</summary>

```
.
./.stages
./depth
./export
./foundationpose
./inference
./mhr_init
./.stages/04_depth_alignment.json
./.stages/01_depth.json
./.stages/02_export.json
./.stages/03_human_initialization.json
./.stages/05_object_pose.json
./depth/moge2
./export/episode_000025
./foundationpose/debug
./foundationpose/episode_000025.pkl
./mhr_init/episode_000025.sam3d.h5
./mhr_init/episode_000025.pkl
./mhr_init/episode_000025.stats.pkl
./depth/moge2/depth_report.json
./depth/moge2/aligned.h5
./depth/moge2/.aligned.h5.storage-repack.lock
./depth/moge2/intrinsics.pkl
./depth/moge2/raw.npy
./export/episode_000025/human_masks
./export/episode_000025/images
./export/episode_000025/object_masks
./export/episode_000025/object_mesh
./export/episode_000025/edex
./export/episode_000025/wild_export.json
./export/episode_ …[+671 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import trimesh,numpy as np
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m, m.vertices.shape, m.faces.shape, m.bounds, m.extents)
print(type(m.visual), getattr(m.visual,'material',None))
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
v=m.vertices-m.vertices.mean(0); w,U=np.linalg.eigh(np.cov(v.T)); print(w,U)
"
```

<details><summary>output</summary>

```
<trimesh.Trimesh(vertices.shape=(259484, 3), faces.shape=(86534, 3))> (259484, 3) (86534, 3) [[-0.04554274 -0.04614785 -0.13553183]
 [ 0.04659176  0.04503646  0.14767875]] [0.09213449 0.09118432 0.28321058]
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7c41a7502510>
[[ 5.59746318e-03  9.99984334e-01 -4.40622709e-06  5.54326156e-04]
 [ 9.99984330e-01 -5.59746277e-03  8.62717918e-05 -5.19255292e-04]
 [ 8.62457766e-05 -4.88906123e-06 -9.99999996e-01  6.07192734e-03]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] [0.09117972 0.09214041 0.28321354]
[0.0006806  0.00083271 0.00979073] [[-0.28372941  0.95834538 -0.03273751]
 [ 0.95863575  0.28429264  0.01397097]
 [-0.02269604  0.02741937  0.99936633]]
```
</details>

**Bash**
```
cat /code/reconstruction/CLAUDE.md | head -100; ls /code/reconstruction/modules /code/reconstruction/scripts /code/reconstruction/workflows; cat /pipeline/.stages/*.json
```

<details><summary>output</summary>

```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## HOI object reconstruction skills

Use the focused companion skills under the monorepo's `.claude/skills/` for
`v2d_hoi_object_reconstruction`:

- **`hoi-object-reconstruction-setup`** prepares host packages, Docker/GPU,
  images, weights, and calibrated stereo input.
- **`hoi-object-reconstruction-run`** launches, monitors, resumes, and verifies
  BundleSDF or SAM3D. It uses the included basketball capture and sensible
  defaults when run details are omitted instead of starting a questionnaire.
- **`hoi-object-reconstruction-doctor`** diagnoses the first failing setup,
  input, pipeline, or output stage and applies the narrowest repair.

For the exact input contract, comm …[+13968 chars]
```
</details>

**Agent:** Starting exploration: the object is a ~28 cm × 9 cm cyan bottle (near-rotationally symmetric about its long axis), 859 frames. Next I'll look at the video, masks, and baseline trajectory.

**Bash**
```
mkdir -p /work/scratch/frames /work/tools /work/outputs; cd /work/scratch; python -c "
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
i=0
while True:
    ok,f=cap.read()
    if not ok: break
    cv2.imwrite(f'frames/{i:04d}.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 92]); i+=1
print(i)
"; python -c "
import h5py; f=h5py.File('/work/inputs/masks.h5'); print({k:(f[k].shape,f[k].dtype) for k in f})
import numpy as np
o=f['object'][:]; a=o.reshape(len(o),-1).sum(1); print(a[::20])
"
```

<details><summary>output</summary>

```
30.0 859.0
859
{'object': ((859, 1152, 1536), dtype('bool')), 'person': ((859, 1152, 1536), dtype('bool'))}
[2590 2820 2738 2660 2610 2665 2745 2296 2077 2295 2422 2424 2429 2473
 2499 2505 2359 2166 2109 2001 2136 2282 2203 2285 2247 2571 2078 1398
  606 1215 1118 1158  566 1428 1634 1535 2073 2176 2687 2825 2867 3008
 3004]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import cv2,numpy as np
ims=[]
for i in range(0,859,43):
    f=cv2.imread(f'frames/{i:04d}.jpg'); f=cv2.resize(f,(384,288)); cv2.putText(f,str(i),(5,25),0,0.8,(0,0,255),2); ims.append(f)
while len(ims)%5: ims.append(np.zeros_like(ims[0]))
rows=[np.hstack(ims[i:i+5]) for i in range(0,len(ims),5)]
cv2.imwrite('sheet.jpg',np.vstack(rows))
"
```

**Read**
```
/work/scratch/sheet.jpg
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
cd /work/scratch; cat > an0.py <<'EOF'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
cam=json.load(open('/work/inputs/camera.json'))
Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
T=r['T_world_mesh'].astype(np.float64)
Tc=Tcw[None]@T
z=T[:,:3,2]  # bottle axis in world
np.set_printoptions(precision=3,suppress=True,linewidth=200)
cp=r['contact_prob_right_left']
for i in range(0,859,10):
    print(i, Tc[i,:3,3], 'axisW',z[i], 'cp',cp[i])
EOF
python an0.py
```

<details><summary>output</summary>

```
0 [0.275 0.285 2.513] axisW [ 0.045 -0.998 -0.053] cp [0. 0.]
10 [0.266 0.285 2.503] axisW [ 0.082 -0.995 -0.058] cp [0. 0.]
20 [0.262 0.279 2.449] axisW [ 0.029 -0.996 -0.09 ] cp [0. 0.]
30 [0.263 0.277 2.446] axisW [ 0.028 -0.988 -0.149] cp [0. 0.]
40 [0.272 0.285 2.471] axisW [ 0.051 -0.988 -0.147] cp [0. 0.]
50 [0.273 0.285 2.473] axisW [ 0.051 -0.988 -0.147] cp [0. 0.]
60 [0.274 0.286 2.483] axisW [ 0.051 -0.988 -0.147] cp [0. 0.]
70 [0.274 0.287 2.486] axisW [ 0.051 -0.988 -0.147] cp [0. 0.]
80 [0.274 0.286 2.484] axisW [ 0.051 -0.988 -0.147] cp [0. 0.]
90 [0.273 0.286 2.48 ] axisW [ 0.051 -0.988 -0.147] cp [0.    0.038]
100 [0.273 0.286 2.479] axisW [ 0.051 -0.988 -0.147] cp [0. 1.]
110 [0.273 0.286 2.476] axisW [ 0.051 -0.988 -0.147] cp [0. 1.]
120 [0.259 0.27  2.518] axisW [ 0.137 …[+5082 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m.visual.uv.shape, m.visual.material.baseColorTexture.size if m.visual.material.baseColorTexture else None)
v=m.vertices
for zz in np.linspace(-0.135,0.147,15):
    s=v[np.abs(v[:,2]-zz)<0.005]; r=np.linalg.norm(s[:,:2]-[0.0005,-0.0005],axis=1)
    print(round(zz,3), len(s), r.max().round(4) if len(s) else None, s[:,:2].mean(0).round(4) if len(s) else None)
"
```

<details><summary>output</summary>

```
(259484, 2) (1464, 1440)
-0.135 22651 0.0443 [ 0.0002 -0.0016]
-0.115 4122 0.0461 [-0.0012 -0.0027]
-0.095 3555 0.0446 [ 0.0011 -0.0003]
-0.075 3681 0.0446 [ 0.0011 -0.0018]
-0.054 3683 0.0445 [ 0.003  -0.0004]
-0.034 3868 0.0447 [ 0.0019 -0.0003]
-0.014 3841 0.045 [0.0027 0.0001]
0.006 3563 0.0453 [0.0016 0.002 ]
0.026 3327 0.0456 [0.0013 0.0037]
0.046 3173 0.0459 [0.0007 0.0043]
0.066 4627 0.0468 [-0.0128  0.0048]
0.087 20377 0.0489 [-0.0066  0.0021]
0.107 11523 0.0507 [-0.0138  0.003 ]
0.127 31447 0.0469 [-0.0087 -0.0003]
0.147 9157 0.0378 [0.0176 0.0029]
```
</details>

**Bash**
```
mkdir -p /work/outputs; cat > /work/scratch/rutil.py <<'EOF'
import numpy as np, torch, trimesh, nvdiffrast.torch as dr, json
class Renderer:
    def __init__(self, mesh_path, device='cuda'):
        m=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
        self.mesh=m
        self.dev=device
        self.v=torch.tensor(m.vertices,dtype=torch.float32,device=device)
        self.f=torch.tensor(m.faces,dtype=torch.int32,device=device)
        uv=np.array(m.visual.uv,dtype=np.float32).copy(); uv[:,1]=1-uv[:,1]
        self.uv=torch.tensor(uv,device=device)
        tex=np.array(m.visual.material.baseColorTexture.convert('RGB'),dtype=np.float32)/255.
        self.tex=torch.tensor(tex,device=device)[None].contiguous()
        self.glctx=dr.RasterizeCudaContext()
    def render(self, T_cam_mesh, K, H, W, x0=0, y0=0):
        """T (B,4,4) torch cam<-mesh; renders crop window [x0,x0+W)x[y0,y0+H) of full image. returns rgb(B,H,W,3), mask, depth"""
        T=torch.as_tensor(T_cam_mesh,dtype=torch.float32,device=self.dev)
        if T.ndim==2: T=T[None]
        B=T.shape[0]
        K=torch.as_tensor(K,dtype=torch.float32,device=self.dev)
        pc=self.v@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3]
        z=pc[...,2]
        u=K[0,0]*pc[...,0]/z+K[0,2]-x0; v=K[1,1]*pc[...,1]/z+K[1,2]-y0
        n,f=0.05,20.
        xc=2*u/W-1; yc=2*v/H-1
        zc=(z*(f+n)/(f-n)-2*f*n/(f-n))
        clip=torch.stack([xc*z,yc*z,zc,z],-1)
        rast,_=dr.rasterize(self.glctx,clip.contigu …[+1117 chars]
```

**Read**
```
/work/scratch/mesh_views.jpg
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
cd /work/scratch; cat > viz.py <<'EOF'
import sys, numpy as np, cv2, h5py
from rutil import *
R=Renderer('/work/inputs/object_mesh.glb')
K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz')
mf=h5py.File('/work/inputs/masks.h5')
def crop_panel(i, poses, Ks, S=200, scale=2):
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    m=mf['object'][i]
    ys,xs=np.nonzero(m)
    if len(xs)==0:
        # use projection of first pose
        p=Ks[0]@poses[0][:3,3]; cx,cy=p[0]/p[2],p[1]/p[2]
    else: cx,cy=xs.mean(),ys.mean()
    x0=int(cx-S/2); y0=int(cy-S/2)
    x0=max(0,min(1536-S,x0)); y0=max(0,min(1152-S,y0))
    crop=img[y0:y0+S,x0:x0+S].copy()
    out=[crop]
    mc=(m[y0:y0+S,x0:x0+S]*255).astype(np.uint8)
    pc=(mf['person'][i][y0:y0+S,x0:x0+S]*120).astype(np.uint8)
    out.append(np.dstack([mc,pc,np.zeros_like(mc)]))
    for T,KK in zip(poses,Ks):
        c,mm,d=R.render(T,KK,S,S,x0,y0)
        c=(c[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=mm[0].cpu().numpy()
        o=crop.copy(); o[mm]=(0.3*o[mm]+0.7*c[mm]).astype(np.uint8)
        out.append(o)
    pan=np.hstack(out)
    pan=cv2.resize(pan,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    cv2.putText(pan,str(i),(5,25),0,0.8,(0,0,255),2)
    return pan
if __name__=='__main__':
    frames=[int(a) for a in sys.argv[2].split(',')]
    pans=[]
    for i in frames:
        Tb=Tcw@r['T_world_mesh'][i].astype(np.float64)
        pans.append(crop_panel(i,[Tb,r['T_cam_m …[+174 chars]
```

**Read**
```
/work/scratch/v1.jpg
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
cd /work/scratch; cat > an1.py <<'EOF'
import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz')
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64); Tc=r['T_cam_mesh_coconet'].astype(np.float64)
def ang(A,B): 
    R=np.einsum('nij,nik->njk',A[:,:3,:3],B[:,:3,:3]); return np.degrees(np.arccos(np.clip((np.trace(R,axis1=1,axis2=2)-1)/2,-1,1)))
def axang(A,B): return np.degrees(np.arccos(np.clip((A[:,:3,2]*B[:,:3,2]).sum(1),-1,1)))
db=ang(Tb[1:],Tb[:-1]); dax=axang(Tb[1:],Tb[:-1])
bf=ang(Tb,Tf); bc=ang(Tb,Tc); fc=ang(Tf,Tc)
np.set_printoptions(precision=1,suppress=True,linewidth=250)
print('frames with baseline step >15deg:', [(i+1,round(db[i],1),round(dax[i],1)) for i in np.nonzero(db>15)[0]])
for i in range(0,859,10):
    print(i, 'step%.1f'%db[min(i,857)], 'b-f %.1f b-c %.1f f-c %.1f'%(bf[i],bc[i],fc[i]), 'axis b-f %.1f b-c %.1f'%(axang(Tb[i:i+1],Tf[i:i+1])[0],axang(Tb[i:i+1],Tc[i:i+1])[0]), 'z b %.3f f %.3f c %.3f'%(Tb[i,2,3],Tf[i,2,3],Tc[i,2,3]))
EOF
python an1.py
```

<details><summary>output</summary>

```
frames with baseline step >15deg: [(25, 23.3, 23.3), (26, 20.4, 20.4), (130, 17.1, 3.3), (133, 24.7, 7.0), (142, 38.0, 3.7), (144, 18.0, 5.7), (145, 17.5, 1.6), (153, 17.1, 1.6), (165, 16.7, 1.7), (167, 28.6, 1.2), (171, 28.5, 11.8), (180, 25.0, 3.5), (182, 60.7, 8.1), (238, 26.2, 26.0), (240, 23.8, 22.4), (241, 25.0, 21.7), (242, 25.0, 21.7), (243, 30.9, 30.5), (244, 20.4, 16.7), (247, 34.8, 32.2), (249, 37.9, 27.5), (251, 68.5, 51.1), (256, 21.5, 3.0), (272, 23.6, 5.9), (275, 21.8, 10.5), (276, 34.8, 20.1), (282, 22.0, 15.3), (283, 43.3, 38.1), (285, 23.6, 6.1), (287, 20.7, 5.0), (288, 18.3, 4.5), (289, 18.3, 4.5), (292, 54.8, 54.2), (303, 20.0, 6.9), (305, 18.6, 8.5), (306, 21.2, 19.5), (307, 36.9, 26.0), (318, 49.6, 47.6), (320, 17.7, 3.8), (321, 17.7, 3.8), (323, 16.8, 2.0), (325, 31. …[+9054 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > evalm.py <<'EOF'
import numpy as np, h5py, torch, json
from rutil import *
R=Renderer('/work/inputs/object_mesh.glb')
K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
mf=h5py.File('/work/inputs/masks.h5')
OM=mf['object'][:]; PM=mf['person'][:]
F=len(OM)
cent=np.zeros((F,2)); area=OM.reshape(F,-1).sum(1)
for i in range(F):
    ys,xs=np.nonzero(OM[i])
    cent[i]=(xs.mean(),ys.mean()) if len(xs) else (np.nan,np.nan)
S=256
def iou_all(Tc, KK=K):
    out=np.zeros(F)
    for i in range(F):
        p=KK@Tc[i,:3,3]; c=cent[i] if area[i]>0 else p[:2]/p[2]
        x0=int(np.clip(c[0]-S/2,0,1536-S)); y0=int(np.clip(c[1]-S/2,0,1152-S))
        _,m,_=R.render(Tc[i],KK,S,S,x0,y0); m=m[0].cpu().numpy()
        o=OM[i,y0:y0+S,x0:x0+S]; pm=PM[i,y0:y0+S,x0:x0+S]&~o
        valid=~pm
        inter=(m&o).sum(); uni=((m|o)&valid).sum()
        out[i]=inter/max(uni,1)
    return out
if __name__=='__main__':
    r=np.load('/work/inputs/reconstruction.npz')
    Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
    ib=iou_all(Tb); ic=iou_all(r['T_cam_mesh_coconet'].astype(np.float64),Kp); iff=iou_all(r['T_cam_mesh_foundationpose'].astype(np.float64),Kp)
    np.savez('iou_base.npz',b=ib,c=ic,f=iff,area=area)
    for s in range(0,F,30):
        e=slice(s,s+30); print(s, 'area %5d'%area[e].mean(), 'base %.3f coco %.3f fp %.3f'%(ib[e].mean(),ic[e].mean(),iff[e].mean()))
    print('mean',ib.mean(),ic.mean(),iff.mean())
EOF
python evalm.py
```

<details><summary>output</summary>

```
0 area  2684 base 0.861 coco 0.864 fp 0.924
30 area  2720 base 0.804 coco 0.804 fp 0.959
60 area  2659 base 0.807 coco 0.808 fp 0.955
90 area  2626 base 0.816 coco 0.817 fp 0.941
120 area  2416 base 0.829 coco 0.828 fp 0.855
150 area  2081 base 0.746 coco 0.744 fp 0.864
180 area  2382 base 0.752 coco 0.753 fp 0.763
210 area  2311 base 0.798 coco 0.797 fp 0.729
240 area  2420 base 0.743 coco 0.742 fp 0.651
270 area  2503 base 0.745 coco 0.747 fp 0.902
300 area  2426 base 0.773 coco 0.778 fp 0.887
330 area  2125 base 0.763 coco 0.764 fp 0.840
360 area  2042 base 0.775 coco 0.775 fp 0.644
390 area  2161 base 0.716 coco 0.717 fp 0.595
420 area  2296 base 0.834 coco 0.838 fp 0.918
450 area  2301 base 0.825 coco 0.831 fp 0.954
480 area  2541 base 0.855 coco 0.858 fp 0.847
510 area  1914 base 0.7 …[+590 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > an2.py <<'EOF'
from evalm import *
import cv2
r=np.load('/work/inputs/reconstruction.npz')
Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
for i in [0,60,100,830,850]:
    c=cent[i]; x0=int(c[0]-64); y0=int(c[1]-64)
    _,m,_=R.render(Tb[i],K,128,128,x0,y0); m=m[0].cpu().numpy()
    o=OM[i,y0:y0+128,x0:x0+128]
    ys,xs=np.nonzero(m); yo,xo=np.nonzero(o)
    print(i,'render area',m.sum(),'mask',o.sum(),'render bbox',xs.min(),xs.max(),ys.min(),ys.max(),'mask bbox',xo.min(),xo.max(),yo.min(),yo.max())
    img=cv2.imread(f'frames/{i:04d}.jpg')[y0:y0+128,x0:x0+128]
    cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cnt,-1,(0,0,255),1)
    cnt,_=cv2.findContours(o.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cnt,-1,(0,255,0),1)
    cv2.imwrite(f'cont_{i}.png',cv2.resize(img,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST))
EOF
python an2.py
```

**Read**
```
/work/scratch/cont_60.png
```

<details><summary>output</summary>

```
0 render area 2653 mask 2590 render bbox 51 82 17 111 mask bbox 50 80 19 109
60 render area 2707 mask 2660 render bbox 51 84 19 114 mask bbox 49 79 16 110
100 render area 2714 mask 2665 render bbox 51 84 17 112 mask bbox 48 80 19 108
830 render area 2954 mask 2975 render bbox 50 81 15 116 mask bbox 48 80 13 113
850 render area 2949 mask 2976 render bbox 50 81 14 115 mask bbox 48 80 11 112
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python viz.py v2.jpg 600,640,660,670,680,690,700,710 && python viz.py v3.jpg 240,250,290,360,370,400,740,780
```

**Read**
```
/work/scratch/v2.jpg
```

**Read**
```
/work/scratch/v3.jpg
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
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2, h5py
from rutil import *
R=Renderer('/work/inputs/object_mesh.glb')
K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz')
mf=h5py.File('/work/inputs/masks.h5')
for i in [0,830]:
    T=Tcw@r['T_world_mesh'][i]
    img=cv2.imread(f'frames/{i:04d}.jpg')[...,::-1]/255.
    c,m,d=R.render(T,K,1152,1536); c=c[0].cpu().numpy(); m=m[0].cpu().numpy()
    o=mf['object'][i]
    print(i,'render mean rgb',c[m].mean(0),'img in mask',img[o].mean(0), 'img in mask p10/p90 lum', np.percentile(img[o].mean(1),[10,50,90]), np.percentile(c[m].mean(1),[10,50,90]))
EOF
```

<details><summary>output</summary>

```
0 render mean rgb [0.22023419 0.462161   0.45222348] img in mask [0.27098342 0.4954743  0.52464078] img in mask p10/p90 lum [0.2848366  0.45228758 0.52418301] [0.23414753 0.40286425 0.45882356]
830 render mean rgb [0.20250028 0.4439401  0.46368334] img in mask [0.24340616 0.46541176 0.52327171] img in mask p10/p90 lum [0.24444444 0.44052288 0.52418301] [0.23231963 0.39673577 0.45881839]
```
</details>

> *thinking:* The baseline matches CoCoNet's output: good silhouette IoU (~0.80) in static phases but dropping to 0.42–0.48 during manipulation (frames 660–720), with axis errors and frozen poses. I'll now build a batched nvdiffrast render-and-compare refiner that scores silhouette and colour across multiple hypotheses per frame.

**Bash**
```
cd /work/scratch; cat > refine.py <<'EOF'
import numpy as np, torch, torch.nn.functional as Fn, h5py, cv2
import nvdiffrast.torch as dr
from rutil import Renderer, load_cam
dev='cuda'
def so3_exp(w):
    th=w.norm(dim=-1,keepdim=True).clamp(min=1e-8); k=w/th
    Kx=torch.zeros(w.shape[:-1]+(3,3),device=w.device,dtype=w.dtype)
    Kx[...,0,1]=-k[...,2];Kx[...,0,2]=k[...,1];Kx[...,1,0]=k[...,2];Kx[...,1,2]=-k[...,0];Kx[...,2,0]=-k[...,1];Kx[...,2,1]=k[...,0]
    th=th[...,None]; I=torch.eye(3,device=w.device,dtype=w.dtype)
    return I+torch.sin(th)*Kx+(1-torch.cos(th))*Kx@Kx
class Data:
    def __init__(self, S=192):
        mf=h5py.File('/work/inputs/masks.h5')
        self.OM=mf['object']; self.PM=mf['person']; self.S=S
        self.F=self.OM.shape[0]
        self.cache={}
    def get(self, i, fallback_uv):
        if i in self.cache: return self.cache[i]
        S=self.S
        o=self.OM[i]; ys,xs=np.nonzero(o)
        c=(xs.mean(),ys.mean()) if len(xs)>30 else fallback_uv
        x0=int(np.clip(c[0]-S/2,0,1536-S)); y0=int(np.clip(c[1]-S/2,0,1152-S))
        img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S,::-1].astype(np.float32)/255.
        oc=o[y0:y0+S,x0:x0+S]; pc=self.PM[i][y0:y0+S,x0:x0+S]
        d=(img,oc.astype(np.float32),(~(pc&~oc)).astype(np.float32),x0,y0,len(xs))
        self.cache[i]=d; return d
class Refiner:
    def __init__(self, R, K):
        self.R=R; self.K=torch.tensor(K,dtype=torch.float32,device=dev)
    def render(self, Rm, t,  …[+3925 chars]
```

<details><summary>output</summary>

```
time 3.842209815979004 mem 1.511721472
60 id init sil 0.602 col 0.244 -> sil 0.030 col 0.225  dR 10.8 dt [-0.016 -0.016 -0.039]
60 fx init sil 0.617 col 0.259 -> sil 0.072 col 0.295  dR 27.8 dt [-0.026 -0.018 -0.156]
60 fy init sil 0.618 col 0.260 -> sil 0.095 col 0.291  dR 28.9 dt [-0.025 -0.02  -0.16 ]
60 fz init sil 0.573 col 0.267 -> sil 0.055 col 0.262  dR 10.9 dt [-0.015 -0.018 -0.056]
240 id init sil 0.926 col 0.279 -> sil 0.047 col 0.230  dR 33.2 dt [-0.035  0.038 -0.215]
240 fx init sil 0.971 col 0.340 -> sil 0.025 col 0.326  dR 83.0 dt [-0.051  0.032 -0.358]
240 fy init sil 0.926 col 0.313 -> sil 0.039 col 0.296  dR 66.0 dt [-0.048  0.029 -0.333]
240 fz init sil 0.944 col 0.273 -> sil 0.023 col 0.269  dR 24.3 dt [-0.057  0.055 -0.314]
250 id init sil 2.052 col 0.350 -> sil 0.070  …[+1034 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > viz2.py <<'EOF'
# viz arbitrary poses: python viz2.py out.jpg file.npz  (npz with R,t,frames)
import sys
from viz import *
d=np.load(sys.argv[2]); fr=d['frames']
pans=[]; 
groups={}
for k,f in enumerate(fr): groups.setdefault(int(f),[]).append(k)
for f,ks in groups.items():
    poses=[]
    for k in ks:
        T=np.eye(4); T[:3,:3]=d['R'][k]; T[:3,3]=d['t'][k]; poses.append(T)
    pans.append(crop_panel(f,poses,[K]*len(poses),S=160,scale=1.5))
cv2.imwrite(sys.argv[1],np.vstack(pans))
EOF
python viz2.py r1.jpg test_ref_out.npz
```

**Read**
```
/work/scratch/r1.jpg
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
import numpy as np; d=np.load('iou_base.npz'); a=d['area']; print(np.nonzero(a<400)[0], a.min())"
```

<details><summary>output</summary>

```
[] 428
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import re
s=open('refine.py').read()
s=s.replace("w=(a*om)[...,None]","w=(a*valid)[...,None]")
open('refine.py','w').write(s)
EOF
cat > stageA.py <<'EOF'
import time, pickle
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam(); Tcw=np.linalg.inv(Twc)
r=np.load('/work/inputs/reconstruction.npz'); Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64)
D=Data(); RF=Refiner(R,K); F=D.F
flips={'id':np.eye(3),'fx':np.diag([1,-1,-1.]),'fy':np.diag([-1,1,-1.]),'fz':np.diag([-1,-1,1.])}
jobs=[]
for i in range(F):
    for src,Rs in [('b',Tb[i,:3,:3]),('f',Tf[i,:3,:3])]:
        for fn,Fm in flips.items(): jobs.append((i,src+fn,Rs@Fm,Tb[i,:3,3]))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
res=[]; BS=128; t1=time.time()
for s in range(0,len(jobs),BS):
    J=jobs[s:s+BS]; dd=[]
    for j in J:
        p=K@j[3]; dd.append(D.get(j[0],p[:2]/p[2]))
    img=tt([d[0] for d in dd]);om=tt([d[1] for d in dd]);val=tt([d[2] for d in dd]);x0=tt([d[3] for d in dd]);y0=tt([d[4] for d in dd])
    Rm,t,ls,lc=RF.optimize(tt([j[2] for j in J]),tt([j[3] for j in J]),img,om,val,x0,y0,D.S,iters=60)
    for k,j in enumerate(J): res.append(dict(f=j[0],src=j[1],R=Rm[k].cpu().numpy(),t=t[k].cpu().numpy(),ls=ls[k].item(),lc=lc[k].item()))
    if s==0: torch.cuda.synchronize(); print('first batch',time.time()-t1,flush=True)
print('total',time.time()-t1)
pickle.dump(res,open('s …[+38 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bg1pkh4go). Output is being written to: /tmp/claude-1003/-work/ca513c79-b1d1-4bff-b8dd-ab8ffbdff8aa/tasks/bg1pkh4go.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 30; cat /tmp/claude-1003/-work/ca513c79-b1d1-4bff-b8dd-ab8ffbdff8aa/tasks/bg1pkh4go.output
```

<details><summary>output</summary>

```
first batch 6.711466312408447
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, h5py
mf=h5py.File('/work/inputs/masks.h5'); OM=mf['object']
F=OM.shape[0]
c=np.zeros((F,2)); a=np.zeros(F); prev=None; iou=np.zeros(F)
for i in range(F):
    o=OM[i]; ys,xs=np.nonzero(o); c[i]=xs.mean(),ys.mean(); a[i]=len(xs)
    if prev is not None: iou[i]=(o&prev).sum()/(o|prev).sum()
    prev=o
np.savez('maskstats.npz',c=c,a=a,iou=iou)
for i in list(range(0,140,4))+list(range(740,859,4)):
    print(i,c[i].round(1),a[i],round(iou[i],3))
EOF
```

<details><summary>output</summary>

```
0 [844.9 652.9] 2590.0 0.0
4 [845.1 652.5] 2656.0 0.992
8 [844.5 652.9] 2640.0 0.988
12 [844.4 653. ] 2648.0 0.992
16 [845.2 651.5] 2744.0 0.993
20 [845.6 651.4] 2820.0 0.928
24 [844.5 651.7] 2731.0 0.992
28 [845.  651.9] 2652.0 0.954
32 [844.9 651.4] 2741.0 0.993
36 [844.3 651.5] 2697.0 0.95
40 [844.5 651.8] 2738.0 0.968
44 [844.  651.2] 2791.0 0.981
48 [844.3 651.2] 2681.0 0.987
52 [845. 651.] 2724.0 0.996
56 [844.4 651.6] 2649.0 0.996
60 [844.4 651.5] 2660.0 0.998
64 [844.3 651.7] 2659.0 0.996
68 [844.3 651.6] 2663.0 0.952
72 [844.4 652.1] 2645.0 0.993
76 [845.1 651.7] 2674.0 0.991
80 [844.3 652.3] 2610.0 0.951
84 [844.9 651.9] 2682.0 0.997
88 [844.9 652.1] 2678.0 0.999
92 [844.9 652.3] 2671.0 0.958
96 [844.1 653.1] 2602.0 0.954
100 [844.7 653. ] 2665.0 0.997
104 [844.2 653.6] 2585.0 0. …[+1176 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=m.vertices[::50]
for i in [0,30,60,100,110,780,800,830,850]:
    w=v@T[i,:3,:3].T+T[i,:3,3]; s=w@n+d
    print(i,'plane dist min %.3f max %.3f'%(s.min(),s.max()), 'center', T[i,:3,3].round(3))
mc=np.load('maskstats.npz')['c']
c=np.load('maskstats.npz')['c']
for i in range(108,124): print(i,c[i].round(1))
EOF
```

<details><summary>output</summary>

```
0 plane dist min 0.802 max 1.086 center [-0.195  0.266  3.369]
30 plane dist min 0.814 max 1.099 center [-0.18   0.249  3.434]
60 plane dist min 0.801 max 1.087 center [-0.193  0.263  3.399]
100 plane dist min 0.802 max 1.088 center [-0.192  0.262  3.403]
110 plane dist min 0.803 max 1.089 center [-0.192  0.261  3.406]
780 plane dist min 0.803 max 1.087 center [-0.208  0.262  3.339]
800 plane dist min 0.798 max 1.084 center [-0.203  0.265  3.408]
830 plane dist min 0.804 max 1.088 center [-0.18   0.255  3.53 ]
850 plane dist min 0.803 max 1.088 center [-0.18   0.255  3.528]
108 [844.9 653.3]
109 [844.9 653.3]
110 [845.  653.3]
111 [845.  653.3]
112 [844.9 653.3]
113 [844.9 653.3]
114 [844.1 653.4]
115 [843.7 653.3]
116 [843.1 652.8]
117 [842.1 651.6]
118 [841.7 650.3]
119 [839.8 649.2]
120 …[+66 chars]
```
</details>

**Bash**
```
sleep 200; cat /tmp/claude-1003/-work/ca513c79-b1d1-4bff-b8dd-ab8ffbdff8aa/tasks/bg1pkh4go.output
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
sleep 100; cat /tmp/claude-1003/-work/ca513c79-b1d1-4bff-b8dd-ab8ffbdff8aa/tasks/bg1pkh4go.output
```

<details><summary>output</summary>

```
first batch 6.711466312408447
```
</details>

**Bash**
```
sleep 90; cat /tmp/claude-1003/-work/ca513c79-b1d1-4bff-b8dd-ab8ffbdff8aa/tasks/bg1pkh4go.output; ls -la /work/scratch/stageA.pkl
```

<details><summary>output</summary>

```
first batch 6.711466312408447
total 452.3696291446686

[exited with code 0]
-rw-rw-r-- 1 hanzhou hanzhou 1049661 Oct 21 19:49 /work/scratch/stageA.pkl
```
</details>

**Bash**
```
cd /work/scratch; cat > anA.py <<'EOF'
import pickle, numpy as np
res=pickle.load(open('stageA.pkl','rb'))
F=859
by=[[] for _ in range(F)]
for x in res: by[x['f']].append(x)
def ang(R1,R2): return np.degrees(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
def axang(R1,R2): return np.degrees(np.arccos(np.clip(R1[:,2]@R2[:,2],-1,1)))
wc=1.0
best=[]
for i in range(F):
    L=[x['ls']+wc*x['lc'] for x in by[i]]; k=int(np.argmin(L)); best.append(by[i][k])
    if i%10==0:
        s=sorted(range(len(L)),key=lambda j:L[j])
        print(i, ' '.join('%s:%.3f/%.3f'%(by[i][j]['src'],by[i][j]['ls'],by[i][j]['lc']) for j in s[:4]), '| 2nd-best ang %.0f ax %.0f'%(ang(by[i][s[0]]['R'],by[i][s[1]]['R']),axang(by[i][s[0]]['R'],by[i][s[1]]['R'])))
EOF
python anA.py
```

<details><summary>output</summary>

```
0 ffz:0.081/0.200 bfz:0.099/0.201 bid:0.108/0.222 fid:0.109/0.225 | 2nd-best ang 3 ax 1
10 fid:0.106/0.222 bid:0.102/0.228 ffx:0.059/0.298 ffz:0.167/0.197 | 2nd-best ang 1 ax 1
20 fid:0.121/0.229 bid:0.119/0.236 bfz:0.122/0.236 ffz:0.126/0.241 | 2nd-best ang 1 ax 1
30 fid:0.044/0.223 ffz:0.060/0.208 bid:0.046/0.224 bfz:0.055/0.217 | 2nd-best ang 123 ax 3
40 bid:0.052/0.224 ffz:0.072/0.206 fid:0.058/0.222 bfz:0.066/0.273 | 2nd-best ang 120 ax 2
50 fid:0.037/0.227 bid:0.055/0.222 ffz:0.053/0.270 bfz:0.071/0.268 | 2nd-best ang 3 ax 1
60 bid:0.040/0.224 fid:0.043/0.227 ffz:0.080/0.208 bfz:0.078/0.265 | 2nd-best ang 3 ax 0
70 bid:0.044/0.224 fid:0.047/0.223 ffz:0.048/0.263 bfz:0.064/0.265 | 2nd-best ang 2 ax 1
80 bid:0.033/0.219 fid:0.042/0.221 ffz:0.057/0.244 bfz:0.055/0.263 | 2nd-best ang 9 a …[+7016 chars]
```
</details>

> *thinking:* Stage A's render-and-compare refinement is eating 7.5 min of GPU time, too slow for the 45 min budget—likely because the texture is being copied per batch element. I'm profiling now to confirm.

**Bash**
```
cd /work/scratch; python - <<'EOF'
s=open('refine.py').read()
s=s.replace("texc,_=dr.interpolate(R.uv[None].expand(B,-1,-1).contiguous(),rast,R.f)\n        col=dr.texture(R.tex.expand(B,-1,-1,-1).contiguous(),texc,filter_mode='linear')",
"texc,_=dr.interpolate(R.uv,rast,R.f)\n        col=dr.texture(R.tex,texc,filter_mode='linear')")
open('refine.py','w').write(s)
EOF
grep -n "dr.texture\|dr.interpolate" refine.py; python - <<'EOF'
import time
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
RF=Refiner(R,K)
B=128;S=192
Rm=torch.eye(3,device=dev).repeat(B,1,1); t=torch.tensor([0,0,2.5],device=dev).repeat(B,1)
x0=torch.full((B,),758.-96,device=dev); y0=torch.full((B,),560.-96,device=dev)
img=torch.rand(B,S,S,3,device=dev); om=torch.zeros(B,S,S,device=dev); val=torch.ones(B,S,S,device=dev)
for k in range(3):
    torch.cuda.synchronize(); t1=time.time()
    Rr,tt,ls,lc=RF.optimize(Rm,t,img,om,val,x0,y0,S,iters=20)
    torch.cuda.synchronize(); print('20 iters',time.time()-t1)
EOF
```

<details><summary>output</summary>

```
38:        texc,_=dr.interpolate(R.uv,rast,R.f)
39:        col=dr.texture(R.tex,texc,filter_mode='linear')
20 iters 2.4925169944763184
20 iters 1.28267240524292
20 iters 1.2782068252563477
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
key=np.hstack([m.vertices,m.visual.uv])
u,inv=np.unique(key.round(7),axis=0,return_inverse=True)
print(len(u), len(np.unique(m.vertices.round(7),axis=0)))
EOF
```

<details><summary>output</summary>

```
48173 43269
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
s=open('rutil.py').read()
s=s.replace("""        self.v=torch.tensor(m.vertices,dtype=torch.float32,device=device)
        self.f=torch.tensor(m.faces,dtype=torch.int32,device=device)
        uv=np.array(m.visual.uv,dtype=np.float32).copy(); uv[:,1]=1-uv[:,1]""","""        # merge duplicated vertices (same position and uv) to cut per-vertex cost
        key=np.hstack([m.vertices,m.visual.uv]).round(7)
        _,first,inv=np.unique(key,axis=0,return_index=True,return_inverse=True)
        inv=inv.reshape(-1)
        self.v=torch.tensor(m.vertices[first],dtype=torch.float32,device=device)
        self.f=torch.tensor(inv[m.faces],dtype=torch.int32,device=device)
        uv=np.array(m.visual.uv[first],dtype=np.float32).copy(); uv[:,1]=1-uv[:,1]""")
s=s.replace("""        texc,_=dr.interpolate(self.uv[None].expand(B,-1,-1).contiguous(),rast,self.f)
        col=dr.texture(self.tex.expand(B,-1,-1,-1).contiguous(),texc,filter_mode='linear')""","""        texc,_=dr.interpolate(self.uv,rast,self.f)
        col=dr.texture(self.tex,texc,filter_mode='linear')""")
open('rutil.py','w').write(s)
EOF
python - <<'EOF'
import time
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
print(R.v.shape)
RF=Refiner(R,K)
B=128;S=192
Rm=torch.eye(3,device=dev).repeat(B,1,1); t=torch.tensor([0,0,2.5],device=dev).repeat(B,1)
x0=torch.full((B,),758.-96,device=dev); y0=torch.full((B,),560.-96,device=dev)
img=torch.rand(B,S,S,3,device=dev); …[+482 chars]
```

<details><summary>output</summary>

```
torch.Size([48173, 3])
20 iters 1.4381389617919922
20 iters 0.48392462730407715
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      triangleSetupKernel(CR::CRParams)         0.00%       0.000us         0.00%       0.000us       0.000us …[+2511 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import pickle, numpy as np, json
res=pickle.load(open('stageA.pkl','rb'))
r=np.load('/work/inputs/reconstruction.npz'); Tcw=np.linalg.inv(np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']))
Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
F=859; by=[[] for _ in range(F)]
for x in res: by[x['f']].append(x)
zb=[];zm=[]
for i in range(F):
    L=[x['ls']+x['lc'] for x in by[i]]; k=int(np.argmin(L)); zm.append(by[i][k]['t'][2]); zb.append(Tb[i,2,3])
zb=np.array(zb); zm=np.array(zm)
for s in range(0,F,40): print(s, 'base z %.3f  fit z %.3f  ratio %.3f'%(np.median(zb[s:s+40]),np.median(zm[s:s+40]),np.median(zm[s:s+40]/zb[s:s+40])))
EOF
```

<details><summary>output</summary>

```
0 base z 2.468  fit z 2.441  ratio 0.988
40 base z 2.483  fit z 2.456  ratio 0.990
80 base z 2.480  fit z 2.461  ratio 0.992
120 base z 2.659  fit z 2.590  ratio 0.968
160 base z 2.746  fit z 2.572  ratio 0.930
200 base z 2.721  fit z 2.612  ratio 0.955
240 base z 2.636  fit z 2.565  ratio 0.974
280 base z 2.612  fit z 2.505  ratio 0.954
320 base z 2.698  fit z 2.574  ratio 0.950
360 base z 2.823  fit z 2.588  ratio 0.929
400 base z 2.716  fit z 2.576  ratio 0.950
440 base z 2.641  fit z 2.509  ratio 0.960
480 base z 2.641  fit z 2.465  ratio 0.928
520 base z 2.986  fit z 2.867  ratio 0.975
560 base z 3.264  fit z 3.120  ratio 0.965
600 base z 3.224  fit z 3.070  ratio 0.949
640 base z 3.239  fit z 3.077  ratio 0.953
680 base z 2.980  fit z 2.819  ratio 0.955
720 base z 2.684  fit z 2.472  …[+141 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > vit.py <<'EOF'
import numpy as np
def rot_angle_matrix(Ra, Rb):
    # Ra (n,3,3), Rb (m,3,3) -> (n,m) degrees
    tr=np.einsum('aij,bij->ab',Ra,Rb)
    return np.degrees(np.arccos(np.clip((tr-1)/2,-1,1)))
def viterbi(cands, unary, lam_r=0.004, lam_t=1.0, static=None, cap=None):
    """cands: list per frame of (R (n,3,3), t (n,3)); unary list of (n,) arrays."""
    F=len(cands); back=[]; acc=unary[0].copy()
    for i in range(1,F):
        Ra,ta=cands[i-1]; Rb,tb=cands[i]
        A=rot_angle_matrix(Ra,Rb); 
        Td=np.linalg.norm(ta[:,None]-tb[None],axis=-1)
        lr=lam_r*(static[i] if static is not None else 1)
        C=lr*A+lam_t*Td
        tot=acc[:,None]+C
        b=np.argmin(tot,0); back.append(b)
        acc=tot[b,np.arange(len(b))]+unary[i]
    path=[int(np.argmin(acc))]
    for b in back[::-1]: path.append(int(b[path[-1]]))
    return path[::-1]
EOF
cat > runvit.py <<'EOF'
import pickle, numpy as np, sys
from vit import *
res=pickle.load(open(sys.argv[1],'rb'))
F=859; by=[[] for _ in range(F)]
for x in res: by[x['f']].append(x)
wc=float(sys.argv[2]); lr=float(sys.argv[3])
cands=[(np.array([x['R'] for x in b]),np.array([x['t'] for x in b])) for b in by]
un=[np.array([x['ls']+wc*x['lc'] for x in b]) for b in by]
path=viterbi(cands,un,lam_r=lr)
sel=[by[i][k] for i,k in enumerate(path)]
R=np.array([s['R'] for s in sel]); t=np.array([s['t'] for s in sel])
steps=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',R[1:],R[:-1])-1)/2,-1,1)))
pri …[+236 chars]
```

<details><summary>output</summary>

```
big steps [(118, 29), (138, 25), (151, 150), (167, 26), (181, 159), (193, 32), (197, 25), (251, 158), (254, 29), (255, 22), (256, 142), (257, 23), (258, 32), (266, 24), (267, 26), (272, 34), (273, 28), (287, 24), (288, 52), (292, 22), (343, 30), (356, 24), (359, 28), (360, 21), (361, 33), (364, 27), (370, 58), (410, 30), (414, 30), (420, 32), (431, 23), (492, 161), (507, 22), (514, 178), (544, 180), (555, 139), (556, 24), (557, 110), (558, 39), (560, 49), (573, 27), (576, 43), (577, 32), (578, 21), (592, 21), (593, 26), (596, 76), (607, 66), (623, 38), (642, 163), (645, 30), (651, 24), (655, 70), (689, 35), (699, 104), (705, 39), (713, 102), (718, 93), (733, 21), (735, 22), (736, 29), (737, 31), (738, 30), (745, 27), (750, 52)]
srcs id id id id id id id id id id id id id id id id fz fz fz  …[+200 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > track.py <<'EOF'
import time, pickle, sys
from refine import *
from vit import rot_angle_matrix
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
D=Data(); RF=Refiner(R,K); F=D.F
resA=pickle.load(open('stageA.pkl','rb'))
byA=[[] for _ in range(F)]
for x in resA: byA[x['f']].append(x)
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
WC=1.0; NKEEP=8; NFRESH=3; ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 20
def nms(cs):
    cs=sorted(cs,key=lambda x:x['ls']+WC*x['lc']); out=[]
    for c in cs:
        if all(rot_angle_matrix(c['R'][None],o['R'][None])[0,0]>8 for o in out): out.append(c)
        if len(out)>=NKEEP: break
    return out
def run(order, tag):
    pool=[]; H=None; t1=time.time()
    for n,i in enumerate(order):
        fresh=sorted(byA[i],key=lambda x:x['ls']+WC*x['lc'])[:NFRESH]
        inits=[(c['R'],c['t']) for c in fresh]
        if H is not None: inits+= [(h['R'],h['t']) for h in H]
        d=D.get(i,None)
        B=len(inits)
        img=tt([d[0]]*B);om=tt([d[1]]*B);val=tt([d[2]]*B);x0=tt([d[3]]*B);y0=tt([d[4]]*B)
        Rm,t,ls,lc=RF.optimize(tt([a[0] for a in inits]),tt([a[1] for a in inits]),img,om,val,x0,y0,D.S,iters=ITERS)
        cs=[dict(f=i,src=tag,R=Rm[k].cpu().numpy(),t=t[k].cpu().numpy(),ls=ls[k].item(),lc=lc[k].item()) for k in range(B)]
        pool+=cs; H=nms(cs)
        if n%100==0: print(tag,n,time.time()-t1,flush=True)
    return pool
pf=run(range(F),'fw'); pb=run(range(F-1,-1,-1),'b …[+108 chars]
```

<details><summary>output</summary>

```
fw 0 1.2763097286224365
fw 100 22.223303079605103
fw 200 43.72468590736389
fw 300 66.07398986816406
fw 400 91.9930362701416
fw 500 116.71441578865051
fw 600 141.95763301849365
fw 700 166.34514451026917
fw 800 192.35023379325867
bw 0 0.17253541946411133
bw 100 15.810124397277832
bw 200 30.974567890167236
bw 300 46.2516393661499
bw 400 61.346381187438965
bw 500 76.79889297485352
bw 600 92.40785908699036
bw 700 107.69682335853577
bw 800 121.92108869552612
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import pickle
a=pickle.load(open('stageA.pkl','rb')); b=pickle.load(open('track1.pkl','rb'))
pickle.dump(a+b,open('poolAT.pkl','wb')); print(len(a),len(b))
EOF
python runvit.py poolAT.pkl 1.0 0.004 vitAT.npz
```

<details><summary>output</summary>

```
6872 18352
big steps [(118, 29), (138, 25), (151, 150), (167, 26), (181, 159), (193, 32), (197, 25), (258, 32), (266, 24), (267, 26), (287, 24), (288, 52), (292, 22), (343, 30), (356, 24), (359, 28), (360, 21), (361, 33), (364, 27), (370, 58), (410, 30), (414, 30), (420, 32), (431, 23), (492, 161), (514, 178), (544, 180), (555, 139), (556, 24), (557, 110), (558, 39), (560, 49), (573, 27), (576, 43), (577, 32), (578, 21), (596, 74), (607, 73), (623, 33), (642, 147), (645, 27), (655, 64), (659, 35), (661, 24), (662, 28), (672, 26), (675, 21), (688, 23), (697, 25), (700, 165), (705, 39), (713, 102), (718, 93), (733, 21), (735, 54), (745, 27), (750, 52)]
srcs id id id id id id id id id id id id id id id id fz fz fz id id w w w id id fz fz id id id id id id id id id id id id id id id id id id i …[+117 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np
d=np.load('vitAT.npz'); fr=list(range(144,190,4))
np.savez('tmpv.npz',R=d['R'][fr],t=d['t'][fr],frames=fr)
EOF
python viz2.py r2.jpg tmpv.npz
```

**Read**
```
/work/scratch/r2.jpg
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
from rutil import *; import cv2
from scipy.spatial.transform import Rotation as Rot
R=Renderer('/work/inputs/object_mesh.glb')
K=np.array([[1500,0,150],[0,1500,250],[0,0,1.]])
ims=[]
for ang in range(0,360,45):
    Rm=Rot.from_euler('xz',[90,ang],degrees=True).as_matrix()  # axis +z -> -y (cap up)
    T=np.eye(4); T[:3,:3]=Rm; T[:3,3]=[0,0,1.0]
    c,m,d=R.render(T,K,500,300)
    im=(c[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); cv2.putText(im,str(ang),(5,25),0,0.8,(0,0,255),2); ims.append(im)
cv2.imwrite('spin_views.jpg',np.hstack(ims))
"
```

**Read**
```
/work/scratch/spin_views.jpg
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
from rutil import *; import cv2
from scipy.spatial.transform import Rotation as Rot
R=Renderer('/work/inputs/object_mesh.glb')
K=np.array([[1500,0,100],[0,1500,250],[0,0,1.]])
ims=[]
for ang in range(0,360,30):
    Rm=Rot.from_euler('XZ',[90,ang],degrees=True).as_matrix()
    T=np.eye(4); T[:3,:3]=Rm; T[:3,3]=[0,0,1.0]
    c,m,d=R.render(T,K,500,200)
    im=(c[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); cv2.putText(im,str(ang),(5,25),0,0.8,(0,0,255),2); ims.append(im)
cv2.imwrite('spin_views.jpg',np.hstack(ims))
"
```

**Read**
```
/work/scratch/spin_views.jpg
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
cd /work/scratch; cat > viz3.py <<'EOF'
# python viz3.py out.jpg poses.npz frames [S] [scale]  -> image | selected | spin+90 | spin+180 | spin-90
import sys
from viz import *
d=np.load(sys.argv[2]); fr=[int(a) for a in sys.argv[3].split(',')]
S=int(sys.argv[4]) if len(sys.argv)>4 else 110; sc=float(sys.argv[5]) if len(sys.argv)>5 else 3
pans=[]
def Rz(a):
    c,s=np.cos(np.radians(a)),np.sin(np.radians(a)); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
for f in fr:
    poses=[]
    for a in [0,90,180,270]:
        T=np.eye(4); T[:3,:3]=d['R'][f]@Rz(a); T[:3,3]=d['t'][f]; poses.append(T)
    pans.append(crop_panel(f,poses,[K]*4,S=S,scale=sc))
cv2.imwrite(sys.argv[1],np.vstack(pans))
EOF
python viz3.py r3.jpg vitAT.npz 140,150,160,170,180,190
```

**Read**
```
/work/scratch/r3.jpg
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

> *thinking:* Spin around the bottle's long axis is critical for ADD accuracy since a 180° rotation shifts surface points ~9 cm, and the blue strap near the cap is the only reliable spin cue—but my color loss sometimes misjudges it. I'm now testing a dedicated feature loss targeting the blue strap and dark cap to fix this.

**Bash**
```
cd /work/scratch; cat > spinsweep.py <<'EOF'
import sys
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
D=Data(); RF=Refiner(R,K)
d=np.load(sys.argv[1])
def Rz(a):
    c,s=np.cos(np.radians(a)),np.sin(np.radians(a)); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def feats(x):  # x (...,3) rgb in [0,1]
    blue=(x[...,2]-x[...,1]).clamp(min=0)
    dark=(0.45-x.mean(-1)).clamp(min=0)
    return torch.stack([blue,dark],-1)
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
for f in [int(a) for a in sys.argv[2].split(',')]:
    img,om,val,x0,y0,n=D.get(f,None)
    angs=np.arange(0,360,20)
    Rs=tt([d['R'][f]@Rz(a) for a in angs]); ts=tt([d['t'][f]]*len(angs)); B=len(angs)
    I=tt([img]*B);O=tt([om]*B);V=tt([val]*B)
    with torch.no_grad():
        ls,lc,a=RF.losses(Rs,ts,I,O,V,tt([x0]*B),tt([y0]*B),D.S)
        col,a=RF.render(Rs,ts,tt([x0]*B),tt([y0]*B),D.S)
        fr=feats(col); fi=feats(I)
        w=(a*V)[...,None]
        lf=((fr-fi).abs()*w).sum((1,2,3))/(w.sum((1,2,3))+1)
        imb=(fi[0,...,0]*O[0]).sum()/O[0].sum()
    print(f,'img blue %.3f'%imb.item())
    print('  lc', ' '.join('%d:%.3f'%(a_,l) for a_,l in zip(angs,lc.cpu().numpy())))
    print('  lf', ' '.join('%d:%.4f'%(a_,l) for a_,l in zip(angs,lf.cpu().numpy())))
EOF
python spinsweep.py vitAT.npz 60,140,160,170,180,190
```

<details><summary>output</summary>

```
60 img blue 0.036
  lc 0:0.228 20:0.239 40:0.235 60:0.249 80:0.266 100:0.243 120:0.227 140:0.235 160:0.257 180:0.269 200:0.260 220:0.234 240:0.219 260:0.210 280:0.201 300:0.188 320:0.190 340:0.216
  lf 0:0.0764 20:0.0847 40:0.0982 60:0.1212 80:0.1347 100:0.1198 120:0.1019 140:0.0902 160:0.0877 180:0.0927 200:0.0982 220:0.0996 240:0.1005 260:0.0944 280:0.0867 300:0.0793 320:0.0743 340:0.0769
140 img blue 0.027
  lc 0:0.189 20:0.207 40:0.240 60:0.260 80:0.257 100:0.257 120:0.248 140:0.235 160:0.227 180:0.232 200:0.243 220:0.254 240:0.251 260:0.234 280:0.212 300:0.206 320:0.197 340:0.187
  lf 0:0.0634 20:0.0667 40:0.0779 60:0.0946 80:0.1044 100:0.1129 120:0.1096 140:0.0999 160:0.0874 180:0.0786 200:0.0722 220:0.0753 240:0.0811 260:0.0853 280:0.0849 300:0.0796 320:0.0699 340:0.0643
160 img blu …[+1568 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > spinstat.py <<'EOF'
import sys
from refine import *
R=Renderer('/work/inputs/object_mesh.glb'); K,Twc,Kp=load_cam()
D=Data(); RF=Refiner(R,K)
d=np.load('vitAT.npz')
def Rz(a):
    c,s=np.cos(np.radians(a)),np.sin(np.radians(a)); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
angs=np.arange(0,360,10)
def feats(x):
    blue=(x[...,2]-x[...,1]).clamp(min=0); dark=(0.45-x.mean(-1)).clamp(min=0)
    return torch.stack([blue,dark],-1)
for ref,frames in [(60,range(0,114,3)),(830,range(785,859,3))]:
    Rs=tt([d['R'][ref]@Rz(a) for a in angs]); ts=tt([d['t'][ref]]*len(angs)); B=len(angs)
    out={k:[] for k in ['ls','lc','lf','blur_lc']}
    for f in frames:
        img,om,val,x0,y0,n=D.get(f,None)
        I=tt([img]*B);O=tt([om]*B);V=tt([val]*B)
        with torch.no_grad():
            ls,lc,_=RF.losses(Rs,ts,I,O,V,tt([x0]*B),tt([y0]*B),D.S)
            col,a=RF.render(Rs,ts,tt([x0]*B),tt([y0]*B),D.S)
            w=(a*V)[...,None]
            lf=((feats(col)-feats(I)).abs()*w).sum((1,2,3))/(w.sum((1,2,3))+1)
        for k,v in [('ls',ls),('lc',lc),('lf',lf)]: out[k].append(v.cpu().numpy())
    for k in ['ls','lc','lf']:
        A=np.array(out[k]); am=angs[A.argmin(1)]
        print(ref,k,'argmins',np.bincount(am//10,minlength=36).nonzero()[0]*10, np.bincount(am//10)[np.bincount(am//10)>0], 'mean curve min at',angs[A.mean(0).argmin()], 'contrast', (A.mean(0).max()-A.mean(0).min()).round(4))
    …[+120 chars]
```

<details><summary>output</summary>

```
60 ls argmins [  0  10  20  30  40 140 280 290 300 320 330 340 350] [3 9 2 2 4 1 1 1 2 1 4 7 1] mean curve min at 340 contrast 0.0419
   mean curve 0:0.065 30:0.064 60:0.088 90:0.089 120:0.102 150:0.100 180:0.097 210:0.095 240:0.088 270:0.083 300:0.065 330:0.062
60 lc argmins [310] [38] mean curve min at 310 contrast 0.0877
   mean curve 0:0.228 30:0.241 60:0.255 90:0.260 120:0.230 150:0.250 180:0.272 210:0.253 240:0.222 270:0.207 300:0.189 330:0.203
60 lf argmins [  0 320 330] [15 22  1] mean curve min at 320 contrast 0.0632
   mean curve 0:0.077 30:0.094 60:0.125 90:0.132 120:0.104 150:0.091 180:0.094 210:0.101 240:0.102 270:0.091 300:0.080 330:0.076
830 ls argmins [ 20 120 130 160 170 220 260 270 340] [11  1  3  1  4  1  2  1  1] mean curve min at 20 contrast 0.0408
   mean curve 0:0.06 …[+486 chars]
```
</details>

> *thinking:* The spin-sweep test confirms the colour loss consistently picks the right spin across all sampled frames, so the issue is the gradient optimizer failing to explore spin space. I'll now implement a discrete spin sweep with Viterbi smoothing plus continuous refinement as a reusable library for `solve.py`, keeping GPU usage modest (~15 of 45 minutes used so far).

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
mkdir -p /work/scratch/dev; cd /work/scratch/dev; cat > s1.py <<'EOF'
import sys, time, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
t0=time.time()
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
D=FrameData('/work/inputs/video.mp4','/work/inputs/masks.h5'); F=D.F
print('data',time.time()-t0)
pickle.dump(D,open('D.pkl','wb'),protocol=4)
R=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(R,K,D.S)
r=np.load('/work/inputs/reconstruction.npz'); Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
Tf=r['T_cam_mesh_foundationpose'].astype(np.float64)
FX=np.diag([1,-1,-1.])
jobs=[]
for i in range(F):
    for Rs in [Tb[i,:3,:3],Tb[i,:3,:3]@FX,Tf[i,:3,:3],Tf[i,:3,:3]@FX]: jobs.append((i,Rs,Tb[i,:3,3]))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
BS=192; out=[]
torch.cuda.synchronize(); t1=time.time()
for s in range(0,len(jobs),BS):
    J=jobs[s:s+BS]; idx=[j[0] for j in J]
    Rm,t,ls,lc=RF.optimize(tt([j[1] for j in J]),tt([j[2] for j in J]),*D.batch(idx),iters=40)
    for k,j in enumerate(J): out.append((j[0],Rm[k].cpu().numpy().astype(np.float64),t[k].cpu().numpy().astype(np.float64),ls[k].item(),lc[k].item()))
torch.cuda.synchronize(); print('S1 gpu',time.time()-t1)
pickle.dump(out,open('s1.pkl','wb'))
EOF
python s1.py
```

<details><summary>output</summary>

```
data 85.0143551826477
S1 gpu 29.25882887840271
```
</details>

**Bash**
```
cd /work/scratch/dev; cat > s2.py <<'EOF'
import sys, time, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
D=pickle.load(open('D.pkl','rb')); F=D.F
R=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(R,K,D.S)
s1=pickle.load(open(sys.argv[1],'rb'))
by=[[] for _ in range(F)]
for x in s1: by[x[0]].append(x)
NSP=24; angs=np.arange(NSP)*360/NSP
SP=np.array([Rz(a) for a in angs])
# axis hypotheses: dedupe by axis direction
hyps=[]
for i in range(F):
    cs=sorted(by[i],key=lambda x:x[3]+x[4]); keep=[]
    for c in cs:
        if all(np.degrees(np.arccos(np.clip(c[1][:,2]@k[1][:,2],-1,1)))>12 for k in keep): keep.append(c)
    hyps.append(keep[:4])
jobs=[]
for i in range(F):
    for h,c in enumerate(hyps[i]):
        for a in range(NSP): jobs.append((i,h,c[1]@SP[a],c[2]))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
BS=384; L=[]
torch.cuda.synchronize(); t1=time.time()
for s in range(0,len(jobs),BS):
    J=jobs[s:s+BS]
    ls,lc=RF.evaluate(tt([j[2] for j in J]),tt([j[3] for j in J]),*D.batch([j[0] for j in J]))
    L+=list(zip(ls.cpu().numpy(),lc.cpu().numpy()))
torch.cuda.synchronize(); print('S2 gpu',time.time()-t1, len(jobs))
states=[[] for _ in range(F)]
for j,l in zip(jobs,L): states[j[0]].append(dict(h=j[1],R=j[2],t=j[3],ls=float(l[0]),lc=float(l[1])))
pickle.dump(states,open(sys.argv[2],'wb'))
EOF
python s2.py s1.pkl s2.pkl
```

<details><summary>output</summary>

```
S2 gpu 11.432700872421265 64128
```
</details>

**Bash**
```
cd /work/scratch/dev; cat > s3.py <<'EOF'
import sys, pickle, numpy as np
sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from vit import viterbi
states=pickle.load(open(sys.argv[1],'rb')); F=len(states)
wc=float(sys.argv[2]); lr=float(sys.argv[3])
cands=[(np.array([s['R'] for s in st]),np.array([s['t'] for s in st])) for st in states]
un=[np.array([s['ls']+wc*s['lc'] for s in st]) for st in states]
path=viterbi(cands,un,lam_r=lr,lam_t=1.0)
R=np.array([cands[i][0][k] for i,k in enumerate(path)]); t=np.array([cands[i][1][k] for i,k in enumerate(path)])
steps=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',R[1:],R[:-1])-1)/2,-1,1)))
print('big steps', [(i+1,round(s)) for i,s in enumerate(steps) if s>25])
np.savez(sys.argv[4],R=R,t=t,frames=np.arange(F))
EOF
python s3.py s2.pkl 1.0 0.004 v2.npz
```

<details><summary>output</summary>

```
big steps [(30, 25), (38, 26), (54, 26), (106, 28), (116, 33), (117, 37), (178, 26), (185, 151), (189, 177), (192, 170), (199, 157), (244, 41), (245, 53), (246, 34), (247, 33), (250, 36), (251, 49), (252, 169), (256, 161), (270, 151), (276, 168), (290, 29), (307, 172), (318, 173), (349, 48), (350, 27), (355, 32), (358, 32), (360, 28), (364, 152), (365, 151), (369, 41), (370, 42), (372, 34), (378, 33), (383, 28), (384, 41), (385, 39), (388, 28), (402, 25), (405, 25), (409, 29), (417, 28), (418, 28), (443, 158), (444, 150), (488, 168), (496, 177), (508, 167), (518, 175), (536, 173), (539, 161), (546, 174), (553, 164), (556, 151), (557, 56), (558, 28), (569, 163), (570, 168), (572, 131), (573, 54), (574, 30), (575, 47), (577, 151), (578, 165), (586, 147), (589, 147), (595, 31), (596, 61), (59 …[+619 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
import pickle, numpy as np
states=pickle.load(open('s2.pkl','rb'))
v=np.load('v2.npz')
for i in range(182,202):
    st=states[i]; hs=sorted(set(s['h'] for s in st))
    out=[]
    for h in hs:
        ss=[s for s in st if s['h']==h]; b=min(ss,key=lambda s:s['ls']+s['lc'])
        out.append('h%d ax%s u%.3f(ls%.3f)'%(h,np.round(b['R'][:,2],2),b['ls']+b['lc'],b['ls']))
    print(i,'sel ax',np.round(v['R'][i][:,2],2),' | '.join(out))
EOF
```

<details><summary>output</summary>

```
182 sel ax [ 0.69 -0.72  0.03] h0 ax[ 0.69 -0.72  0.03] u0.773(ls0.477) | h1 ax[-0.69  0.73 -0.04] u0.859(ls0.527) | h2 ax[-0.78  0.61  0.15] u2.119(ls1.775)
183 sel ax [ 0.75 -0.65  0.12] h0 ax[ 0.75 -0.65  0.12] u1.120(ls0.818) | h1 ax[-0.68  0.69  0.24] u1.288(ls0.984) | h2 ax[ 0.69 -0.72 -0.08] u1.372(ls1.051)
184 sel ax [ 0.76 -0.64  0.06] h0 ax[-0.7   0.65  0.31] u1.364(ls1.013) | h1 ax[ 0.76 -0.64  0.06] u1.495(ls1.177)
185 sel ax [-0.69  0.53  0.49] h0 ax[-0.69  0.53  0.49] u1.034(ls0.707) | h1 ax[-0.81  0.49  0.31] u1.306(ls0.972) | h2 ax[ 0.84 -0.55  0.07] u2.241(ls1.895)
186 sel ax [-0.77  0.45  0.45] h0 ax[-0.77  0.45  0.45] u0.831(ls0.470) | h1 ax[ 0.83 -0.52  0.2 ] u1.925(ls1.593)
187 sel ax [-0.82  0.51  0.24] h0 ax[-0.82  0.51  0.24] u1.060(ls0.710) | h1 ax[ 0.9  -0.42 -0.1 …[+2298 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
import pickle, numpy as np
s1=pickle.load(open('s1.pkl','rb'))
for x in s1:
    if x[0] in (60,190,195): print(x[0], np.round(x[1][:,2],2), np.round(x[2],3), round(x[3],3), round(x[4],3))
st=pickle.load(open('s2.pkl','rb'))[190]
print([round(s['ls'],3) for s in st if s['h']==0])
EOF
```

<details><summary>output</summary>

```
60 [-0.12 -0.98  0.17] [0.244 0.257 2.52 ] 1.532 0.307
60 [-0.15  0.97  0.21] [0.245 0.324 2.427] 1.64 0.352
60 [-0.11 -0.99 -0.05] [0.262 0.29  2.495] 0.544 0.263
60 [-0.57  0.72  0.38] [0.161 0.437 2.323] 5.966 0.444
190 [ 0.9  -0.42 -0.1 ] [ 0.117 -0.274  2.699] 0.358 0.297
190 [-0.94  0.17  0.3 ] [ 0.087 -0.274  2.656] 1.791 0.361
190 [ 0.95 -0.28 -0.11] [ 0.111 -0.282  2.693] 1.134 0.304
190 [-0.97  0.22  0.12] [ 0.083 -0.262  2.67 ] 1.488 0.331
195 [0.98 0.18 0.12] [ 0.179 -0.313  2.698] 2.599 0.418
195 [-0.85 -0.44  0.3 ] [-0.059 -0.338  2.655] 6.074 0.239
195 [ 0.96  0.23 -0.16] [ 0.166 -0.305  2.72 ] 1.845 0.386
195 [-0.93 -0.16  0.33] [ 0.113 -0.292  2.675] 1.888 0.359
[0.358, 0.351, 0.325, 0.324, 0.351, 0.36, 0.347, 0.355, 0.374, 0.388, 0.385, 0.387, 0.409, 0.41, 0.425, 0.421, 0 …[+54 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
import sys, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
import hoi_lib
sys.path.insert(0,'/work/scratch')
import refine as old
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
D=pickle.load(open('D.pkl','rb'))
r=np.load('/work/inputs/reconstruction.npz'); T=Tcw@r['T_world_mesh'][60]
R=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(R,K,D.S)
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
b=D.batch([60])
print([x.shape for x in b], b[3],b[4], b[1].sum(), b[2].mean(), b[0].mean())
print(RF.evaluate(tt([T[:3,:3]]),tt([T[:3,3]]),*b))
Do=old.Data(); d=Do.get(60,None)
print(d[3],d[4],d[1].sum(),d[2].mean(),d[0].mean(), np.abs(d[0]-b[0][0].cpu().numpy()).mean())
EOF
```

<details><summary>output</summary>

```
[torch.Size([1, 192, 192, 3]), torch.Size([1, 192, 192]), torch.Size([1, 192, 192]), torch.Size([1]), torch.Size([1])] tensor([748.], device='cuda:0') tensor([555.], device='cuda:0') tensor(2660., device='cuda:0') tensor(0.6853, device='cuda:0') tensor(0.4026, device='cuda:0')
(tensor([0.5793], device='cuda:0'), tensor([0.2486], device='cuda:0'))
748 555 2660.0 0.6852756 0.40258145 0.0051702526
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
import sys, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
sys.path.insert(0,'/work/scratch')
import refine as old
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
D=pickle.load(open('D.pkl','rb'))
r=np.load('/work/inputs/reconstruction.npz'); T=Tcw@r['T_world_mesh'][60]
R=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(R,K,D.S); RO=old.Refiner(R,K)
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
b=D.batch([60])
for it in [20,40,80]:
    _,_,ls,lc=RF.optimize(tt([T[:3,:3]]),tt([T[:3,3]]),*b,iters=it)
    _,_,ls2,lc2=RO.optimize(tt([T[:3,:3]]),tt([T[:3,3]]),b[0],b[1],b[2],b[3],b[4],192,iters=it)
    print(it, ls.item(), lc.item(), ls2.item(), lc2.item())
EOF
```

<details><summary>output</summary>

```
20 0.9556655287742615 0.2781594395637512 0.9556655287742615 0.2781594395637512
40 0.8916258811950684 0.27778980135917664 0.6260368227958679 0.2774520814418793
80 0.8424667119979858 0.2644580006599426 0.6132659912109375 0.24056603014469147
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
import sys, pickle, json, time
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
D=pickle.load(open('D.pkl','rb'))
r=np.load('/work/inputs/reconstruction.npz'); T=Tcw@r['T_world_mesh'][60]
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
b=D.batch([60])
class R2(Renderer):
    def __init__(s,mode):
        m=load_mesh('/work/inputs/object_mesh.glb'); s.glctx=dr.RasterizeCudaContext()
        uv=np.array(m.visual.uv,dtype=np.float32).copy(); uv[:,1]=1-uv[:,1]
        tex=np.array(m.visual.material.baseColorTexture.convert('RGB'),dtype=np.float32)/255.
        s.tex=torch.tensor(tex,device=dev)[None].contiguous()
        if mode=='raw':
            s.v=torch.tensor(m.vertices,dtype=torch.float32,device=dev); s.f=torch.tensor(m.faces,dtype=torch.int32,device=dev)
            s.uv=torch.tensor(uv,device=dev); s.fuv=s.f
        else:
            _,first,inv=np.unique(m.vertices.round(7),axis=0,return_index=True,return_inverse=True); inv=inv.reshape(-1)
            s.v=torch.tensor(m.vertices[first],dtype=torch.float32,device=dev); s.f=torch.tensor(inv[m.faces],dtype=torch.int32,device=dev)
            s.uv=torch.tensor(uv,device=dev); s.fuv=torch.tensor(m.faces,dtype=torch.int32,device=dev)
class RF2(Refiner):
    def render(self, Rm, t, x0, y0):
        R=self.R;S=self.S
        pc=torch.einsum('bi …[+1015 chars]
```

<details><summary>output</summary>

```
raw torch.Size([259484, 3])
  20 0.09025955945253372 0.2208246886730194
  40 0.052528999745845795 0.2261408269405365
  80 0.03594261780381203 0.22841447591781616
time 20it B128 1.2761132717132568
pos torch.Size([43269, 3])
  20 0.8623808026313782 0.26906853914260864
  40 2.3292076587677 0.3353055715560913
  80 4.849786758422852 0.35177725553512573
time 20it B128 0.4754769802093506
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF' 2>&1 | tail -5
import sys, pickle, json, time
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
import trimesh
m=load_mesh('/work/inputs/object_mesh.glb')
_,first,inv=np.unique(m.vertices.round(7),axis=0,return_index=True,return_inverse=True); inv=inv.reshape(-1)
mm=trimesh.Trimesh(m.vertices[first],inv[m.faces],process=False)
print('watertight',mm.is_watertight,'winding consistent',mm.is_winding_consistent,'volume',mm.volume, 'euler',mm.euler_number)
EOF
```

<details><summary>output</summary>

```
watertight True winding consistent True volume 0.001565824962160384 euler 2
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
import re
s=open('/work/outputs/hoi_lib.py').read()
s=s.replace("""        # merge duplicated vertices (same position and uv) to cut per-vertex cost
        key = np.hstack([m.vertices, m.visual.uv]).round(7)
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        self.v = torch.tensor(m.vertices[first], dtype=torch.float32, device=dev)
        self.f = torch.tensor(inv[m.faces], dtype=torch.int32, device=dev)
        uv = np.array(m.visual.uv[first], dtype=np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=dev)""","""        # merge duplicated vertex positions -> watertight topology (needed for correct silhouette
        # gradients in dr.antialias, and 6x fewer vertices); uvs keep the original per-face indexing.
        _, first, inv = np.unique(m.vertices.round(7), axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        self.v = torch.tensor(m.vertices[first], dtype=torch.float32, device=dev)
        # winding reversed because our clip space has y pointing down (image rows), which mirrors facing
        self.f = torch.tensor(inv[m.faces][:, [0, 2, 1]], dtype=torch.int32, device=dev)
        self.fuv = torch.tensor(m.faces[:, [0, 2, 1]], dtype=torch.int32, device=dev)
        uv = np.array(m.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, devi …[+830 chars]
```

<details><summary>output</summary>

```
Exit code 1
Traceback (most recent call last):
  File "<stdin>", line 12, in <module>
  File "/work/outputs/hoi_lib.py", line 141, in optimize
    ls, lc = self.losses(Rm, t, img, om, valid, x0, y0)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/hoi_lib.py", line 119, in losses
    col, a = self.render(Rm, t, x0, y0)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/hoi_lib.py", line 109, in render
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (S, S))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+636 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; sed -i 's/\[:, \[0, 2, 1\]\], dtype=torch.int32, device=dev)/[:, [0, 2, 1]].copy(), dtype=torch.int32, device=dev)/' /work/outputs/hoi_lib.py; grep -n "0, 2, 1" /work/outputs/hoi_lib.py; python - <<'EOF'
import sys, pickle, json, time
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
D=pickle.load(open('/work/scratch/dev/D.pkl','rb'))
r=np.load('/work/inputs/reconstruction.npz')
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
RR=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(RR,K,192)
for f in [60,240]:
    T=Tcw@r['T_world_mesh'][f]; b=D.batch([f])
    for it in [20,40,80]:
        _,_,ls,lc=RF.optimize(tt([T[:3,:3]]),tt([T[:3,3]]),*b,iters=it)
        print(f,it, ls.item(), lc.item())
EOF
```

<details><summary>output</summary>

```
28:        self.f = torch.tensor(inv[m.faces][:, [0, 2, 1]].copy(), dtype=torch.int32, device=dev)
29:        self.fuv = torch.tensor(m.faces[:, [0, 2, 1]].copy(), dtype=torch.int32, device=dev)
60 20 0.4975552558898926 0.24426469206809998
60 40 2.879314422607422 0.3720838129520416
60 80 6.9505934715271 0.26856398582458496
240 20 0.20394234359264374 0.25811767578125
240 40 0.3298141062259674 0.2377064824104309
240 80 0.1089830994606018 0.22637470066547394
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
s=open('/work/outputs/hoi_lib.py').read()
a=s.index("        # merge duplicated vertex positions"); b=s.index("        tex = np.array")
s=s[:a]+"""        # NOTE: the raw (unwelded) triangle soup is used on purpose: with the welded watertight mesh the
        # dr.antialias silhouette gradients made the pose optimisation diverge (tested), with the soup it converges.
        self.v = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
        self.f = torch.tensor(m.faces, dtype=torch.int32, device=dev)
        self.fuv = self.f
        uv = np.array(m.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=dev)
"""+s[b:]
open('/work/outputs/hoi_lib.py','w').write(s)
EOF
sed -n 15,35p /work/outputs/hoi_lib.py
```

<details><summary>output</summary>

```
    return trimesh.load(path, force="scene", process=False).to_geometry()


class Renderer:
    def __init__(self, mesh_path):
        m = load_mesh(mesh_path)
        self.mesh = m
        # NOTE: the raw (unwelded) triangle soup is used on purpose: with the welded watertight mesh the
        # dr.antialias silhouette gradients made the pose optimisation diverge (tested), with the soup it converges.
        self.v = torch.tensor(m.vertices, dtype=torch.float32, device=dev)
        self.f = torch.tensor(m.faces, dtype=torch.int32, device=dev)
        self.fuv = self.f
        uv = np.array(m.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=dev)
        tex = np.array(m.visual.material.baseColorTexture.convert('RGB'), dtype=np.flo …[+145 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; sed -i 's/^D=FrameData.*$/D=pickle.load(open("D.pkl","rb")); F=D.F/; s/^pickle.dump(D,open.*$//' s1.py; python s1.py && python s2.py s1.pkl s2.pkl && python s3.py s2.pkl 1.0 0.004 v2.npz
```

<details><summary>output</summary>

```
data 0.09057378768920898
S1 gpu 99.42883801460266
S2 gpu 31.91026496887207 50088
big steps [(159, 28), (166, 30), (167, 29), (177, 36), (180, 35), (415, 29), (420, 34), (548, 33), (557, 82), (596, 45), (597, 39), (621, 34), (622, 28), (642, 153), (651, 32), (655, 59), (699, 108)]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, json
d=np.load('dev/v2.npz'); 
fr=[100,160,175,190,250,300,370,420,480,540,557,600,640,650,660,700,720,760]
np.savez('tmpv.npz',R=d['R'][fr],t=d['t'][fr],frames=fr)
EOF
python viz2.py r4.jpg tmpv.npz
```

**Read**
```
/work/scratch/r4.jpg
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
cd /work/scratch; cat > viz4.py <<'EOF'
# grid viz: python viz4.py out.jpg poses.npz frames [poses2.npz]  -> per frame: image | overlay1 [| overlay2]
import sys
from viz import *
ds=[np.load(a) for a in sys.argv[4:]] if len(sys.argv)>4 else []
d=np.load(sys.argv[2]); fr=[int(a) for a in sys.argv[3].split(',')]
tiles=[]
for f in fr:
    img=cv2.imread(f'/work/scratch/frames/{f:04d}.jpg'); S=120
    c=np.load('/work/scratch/maskstats.npz')['c'][f]
    x0=int(np.clip(c[0]-S/2,0,1536-S)); y0=int(np.clip(c[1]-S/2,0,1152-S))
    crop=img[y0:y0+S,x0:x0+S]; out=[crop]
    for dd in [d]+ds:
        T=np.eye(4); T[:3,:3]=dd['R'][f]; T[:3,3]=dd['t'][f]
        col,mm,_=R.render(T,K,S,S,x0,y0); col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=mm[0].cpu().numpy()
        o=crop.copy(); o[mm]=col[mm]; out.append(o)
    p=cv2.resize(np.hstack(out),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST); cv2.putText(p,str(f),(4,20),0,0.7,(0,0,255),2)
    tiles.append(np.pad(p,((2,2),(2,2),(0,0))))
nc=3 if len(ds)==0 else 2
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1],np.vstack([np.hstack(tiles[i:i+nc]) for i in range(0,len(tiles),nc)]))
EOF
python viz4.py r4.jpg dev/v2.npz 100,160,175,190,250,300,370,420,480,540,557,600
```

**Read**
```
/work/scratch/r4.jpg
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
cd /work/scratch/dev; python - <<'EOF'
import pickle, numpy as np
states=pickle.load(open('s2.pkl','rb')); v=np.load('v2.npz')
for i in [20,60,100,190]:
    st=states[i]
    for h in sorted(set(s['h'] for s in st)):
        ss=[s for s in st if s['h']==h]
        print(i,'h',h,'ax',np.round(ss[0]['R'][:,2],2),' '.join('%.3f'%(s['ls']+s['lc']) for s in ss))
    k=[j for j,s in enumerate(st) if np.allclose(s['R'],v['R'][i])][0]; print('  sel',st[k]['h'],k%24, st[k]['ls']+st[k]['lc'])
EOF
```

<details><summary>output</summary>

```
20 h 0 ax [ 0.01 -0.98 -0.19] 0.363 0.377 0.382 0.395 0.421 0.429 0.415 0.401 0.392 0.396 0.405 0.416 0.418 0.414 0.401 0.382 0.370 0.361 0.354 0.332 0.318 0.315 0.330 0.358
20 h 1 ax [0.03 0.91 0.41] 0.449 0.455 0.464 0.494 0.525 0.563 0.596 0.602 0.601 0.594 0.553 0.522 0.482 0.463 0.457 0.473 0.497 0.527 0.545 0.573 0.567 0.549 0.489 0.453
  sel 0 20 0.3183160275220871
60 h 0 ax [ 0.01 -0.99 -0.14] 0.266 0.279 0.274 0.290 0.315 0.325 0.329 0.334 0.328 0.334 0.346 0.363 0.357 0.346 0.312 0.283 0.271 0.271 0.262 0.233 0.226 0.229 0.251 0.269
60 h 1 ax [0.03 0.93 0.37] 0.380 0.384 0.402 0.421 0.455 0.482 0.522 0.524 0.530 0.521 0.479 0.459 0.431 0.405 0.396 0.423 0.444 0.466 0.485 0.505 0.494 0.462 0.416 0.386
  sel 0 20 0.22556599602103233
100 h 0 ax [ 0.01 -0.99 -0.13] 0.320 0.321 0.336  …[+706 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
import pickle, numpy as np, cv2, sys
sys.path.insert(0,'/work/scratch')
states=pickle.load(open('s2.pkl','rb'))
from viz import R, K
f=100; st=[s for s in states[f] if s['h']==0]
img=cv2.imread(f'/work/scratch/frames/{f:04d}.jpg'); c=np.load('/work/scratch/maskstats.npz')['c'][f]; S=110
x0=int(c[0]-S/2); y0=int(c[1]-S/2); crop=img[y0:y0+S,x0:x0+S]
tiles=[crop]
for k in range(0,24,2):
    T=np.eye(4); T[:3,:3]=st[k]['R']; T[:3,3]=st[k]['t']
    col,mm,_=R.render(T,K,S,S,x0,y0); col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=mm[0].cpu().numpy()
    o=crop.copy(); o[mm]=col[mm]; cv2.putText(o,'%d %.2f'%(k*15,st[k]['ls']+st[k]['lc']),(2,12),0,0.4,(0,0,255),1); tiles.append(o)
tiles.append(crop)
cv2.imwrite('/work/scratch/spin100.jpg',cv2.resize(np.vstack([np.hstack(tiles[:7]),np.hstack(tiles[7:])]),None,fx=2,fy=2))
EOF
```

**Read**
```
/work/scratch/spin100.jpg
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
cd /work/scratch/dev; cat > losstest.py <<'EOF'
import sys, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
D=pickle.load(open('D.pkl','rb'))
RR=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(RR,K,192)
states=pickle.load(open('s2.pkl','rb'))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
def blur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-(torch.arange(k,device=dev)-k//2)**2/(2*s*s)); g=g/g.sum()
    x=x.permute(0,3,1,2); C=x.shape[1]
    x=Fn.conv2d(x,g.view(1,1,1,k).repeat(C,1,1,1),padding=(0,k//2),groups=C)
    x=Fn.conv2d(x,g.view(1,1,k,1).repeat(C,1,1,1),padding=(k//2,0),groups=C)
    return x.permute(0,2,3,1)
def variants(col,a,img,om,val):
    out={}
    w=(a*val)[...,None]
    for s in [0,1.5,3]:
        cb=blur(col,s) if s else col; ib=blur(img*(1),s) if s else img
        ab=blur(a[...,None],s)[...,0] if s else a
        # compare only where render alpha (blurred) is substantial
        ww=w
        g=((ib*cb*ww).sum((1,2))/((cb*cb*ww).sum((1,2))+1e-6)).clamp(0.7,1.5)
        out['lc_s%.1f'%s]=((cb*g[:,None,None]-ib).abs()*ww).sum((1,2,3))/(ww.sum((1,2,3))+1)
        bl=lambda x:(x[...,2]-x[...,1])
        out['blue_s%.1f'%s]=((bl(cb)-bl(ib)).abs()*ww[...,0]).sum((1,2))/(ww[...,0].sum((1,2))+1)
    return out
frames=[int(a) for a in sys.argv[1].split(',')]
for f in frames:
    st=[s for s in states[f] if s['h']==0]; B=len(st)
    b=D.batch([f]*B)
    w …[+350 chars]
```

<details><summary>output</summary>

```
20 lc_s0.0 argmin 300 contrast 0.348 0.237 0.253 0.249 0.246 0.226 0.245 0.273 0.252 0.208 0.201 0.192 0.210
20 blue_s0.0 argmin 15 contrast 0.619 0.054 0.050 0.070 0.088 0.090 0.080 0.071 0.064 0.061 0.064 0.070 0.070
20 lc_s1.5 argmin 300 contrast 0.447 0.214 0.221 0.203 0.215 0.201 0.218 0.244 0.215 0.171 0.161 0.154 0.181
20 blue_s1.5 argmin 15 contrast 0.654 0.051 0.045 0.061 0.082 0.085 0.076 0.068 0.062 0.058 0.061 0.068 0.067
20 lc_s3.0 argmin 285 contrast 0.438 0.205 0.196 0.177 0.204 0.197 0.209 0.220 0.186 0.156 0.141 0.143 0.178
20 blue_s3.0 argmin 15 contrast 0.669 0.048 0.037 0.050 0.072 0.077 0.070 0.064 0.057 0.054 0.058 0.064 0.063
100 lc_s0.0 argmin 315 contrast 0.413 0.214 0.237 0.224 0.245 0.229 0.231 0.260 0.257 0.205 0.185 0.175 0.180
100 blue_s0.0 argmin 30 contrast  …[+519 chars]
```
</details>

> *thinking:* The spin failure traces to the colour term being dominated by body shading, which wrongly favors "strap hidden" even when the strap is visible; a blueness feature matches the correct angle better. I'm switching the unary to silhouette + blueness + cap-darkness with the colour term down-weighted.

**Bash**
```
cd /work/scratch/dev; python - <<'EOF'
s=open('/work/outputs/hoi_lib.py').read()
s=s.replace('''        w = (a * valid)[..., None]
        g = ((img * col * w).sum((1, 2)) / ((col * col * w).sum((1, 2)) + 1e-6)).detach().clamp(0.7, 1.5)
        lcol = ((col * g[:, None, None] - img).abs() * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) + 1)
        return lsil, lcol
''','''        w = (a * valid)[..., None]
        g = ((img * col * w).sum((1, 2)) / ((col * col * w).sum((1, 2)) + 1e-6)).detach().clamp(0.7, 1.5)
        lcol = ((col * g[:, None, None] - img).abs() * w).sum((1, 2, 3)) / (w.sum((1, 2, 3)) + 1)
        # appearance features robust to shading: blueness (B-G, the blue strap near the cap, the only strong
        # cue for the spin about the long axis) and darkness (the dark cap, cue for the end-over-end flip),
        # compared after a light blur so that 1-2 px misalignment is tolerated.
        cb = blur(col, 1.5); ib = blur(img, 1.5); w0 = w[..., 0]; nw = w0.sum((1, 2)) + 1
        blue = lambda x: x[..., 2] - x[..., 1]
        dark = lambda x: (0.35 - x.mean(-1)).clamp(min=0)
        lblue = ((blue(cb) - blue(ib)).abs() * w0).sum((1, 2)) / nw
        ldark = ((dark(cb * g[:, None, None]) - dark(ib)).abs() * w0).sum((1, 2)) / nw
        return lsil, lcol, lblue, ldark

    def unary(self, ls, lc, lb, ld):
        return ls + self.wc * lc + self.wb * lb + self.wd * ld
''')
s=s.replace('''    def __init__(self, R, K, S):
        self.R = R; self.S = S''','''    def __init_ …[+1714 chars]
```

<details><summary>output</summary>

```
14:def load_mesh(path):
15:    return trimesh.load(path, force="scene", process=False).to_geometry()
19:    def __init__(self, mesh_path):
35:def so3_exp(w):
43:    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
46:def blur(x, s):
56:    return x[..., 0] if sq else x
59:def Rz(a_deg):
61:    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
64:def rot_angle(Ra, Rb):
67:    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
73:    def __init__(self, video, masks_h5, S=192):
99:    def batch(self, idx):
102:        return (t(self.img[idx]).float() / 255., t(self.om[idx]).float(), t(self.val[idx]).float(),
107:    def __init__(self, R, K, S, wc=0.3, wb=4.0, wd=4.0):
111:    def render(self, Rm, t, x0, y0):
125:        return out[..., :3], out[..., 3]
127:    def loss …[+485 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; cat > utest.py <<'EOF'
import sys, pickle, json
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'])
D=pickle.load(open('D.pkl','rb'))
RR=Renderer('/work/inputs/object_mesh.glb'); RF=Refiner(RR,K,192)
states=pickle.load(open('s2.pkl','rb'))
tt=lambda a:torch.tensor(np.array(a),dtype=torch.float32,device=dev)
def ev(f,h):
    st=[s for s in states[f] if s['h']==h]; B=len(st); b=D.batch([f]*B)
    L=RF.evaluate(tt([s['R'] for s in st]),tt([s['t'] for s in st]),*b)
    return L.cpu().numpy(), RF.unary(*L.T).cpu().numpy()
for f in [0,30,60,100,800,830,850]:
    L,u=ev(f,0); print(f,'unary argmin',np.argmin(u)*15,'blue argmin',np.argmin(L[:,2])*15,'dark',np.argmin(L[:,3])*15, 'lc',np.argmin(L[:,1])*15, 'u contrast %.3f'%(u.max()-u.min()))
for f in [190,250,300,700]:
    for h in range(len(set(s['h'] for s in states[f]))):
        L,u=ev(f,h); k=np.argmin(u); print(f,'h',h,'ax',np.round(states[f][h*24]['R'][:,2],2),'best u %.3f'%u[k],'comps',L[k].round(3),'spin',k*15)
EOF
python utest.py
```

<details><summary>output</summary>

```
0 unary argmin 15 blue argmin 15 dark 225 lc 285 u contrast 0.241
30 unary argmin 15 blue argmin 15 dark 0 lc 300 u contrast 0.267
60 unary argmin 15 blue argmin 15 dark 0 lc 300 u contrast 0.254
100 unary argmin 30 blue argmin 30 dark 15 lc 315 u contrast 0.196
800 unary argmin 0 blue argmin 0 dark 0 lc 0 u contrast 0.376
830 unary argmin 0 blue argmin 0 dark 0 lc 0 u contrast 0.478
850 unary argmin 0 blue argmin 0 dark 0 lc 0 u contrast 0.483
190 h 0 ax [ 0.9  -0.41 -0.11] best u 0.622 comps [0.217 0.278 0.062 0.018] spin 345
190 h 1 ax [-0.79  0.34  0.51] best u 0.651 comps [0.162 0.361 0.065 0.03 ] spin 195
250 h 0 ax [-0.81  0.52 -0.29] best u 0.513 comps [0.073 0.303 0.065 0.022] spin 345
250 h 1 ax [-0.76  0.63  0.15] best u 0.513 comps [0.129 0.306 0.053 0.02 ] spin 225
250 h 2 ax  …[+579 chars]
```
</details>

**Bash**
```
cd /work/scratch/dev; sed -i 's/wc=0.3, wb=4.0, wd=4.0/wc=0.6, wb=4.0, wd=4.0/' /work/outputs/hoi_lib.py
python - <<'EOF'
s=open('s2.py').read()
s=s.replace("""    ls,lc=RF.evaluate(tt([j[2] for j in J]),tt([j[3] for j in J]),*D.batch([j[0] for j in J]))
    L+=list(zip(ls.cpu().numpy(),lc.cpu().numpy()))""","""    Lb=RF.evaluate(tt([j[2] for j in J]),tt([j[3] for j in J]),*D.batch([j[0] for j in J]))
    L+=list(Lb.cpu().numpy())""")
s=s.replace("states[j[0]].append(dict(h=j[1],R=j[2],t=j[3],ls=float(l[0]),lc=float(l[1])))","states[j[0]].append(dict(h=j[1],R=j[2],t=j[3],L=l,u=float(RF.unary(*torch.tensor(l)))))")
open('s2.py','w').write(s)
s=open('s3.py').read()
s=s.replace("un=[np.array([s['ls']+wc*s['lc'] for s in st]) for st in states]","un=[np.array([s['u'] for s in st]) for st in states]")
open('s3.py','w').write(s)
EOF
python s2.py s1.pkl s2b.pkl && python s3.py s2b.pkl 1.0 0.004 v3.npz
```

<details><summary>output</summary>

```
S2 gpu 18.400120735168457 50088
big steps [(160, 31), (166, 30), (167, 29), (177, 35), (180, 35), (203, 59), (288, 57), (414, 29), (420, 32), (482, 25), (514, 97), (548, 33), (557, 53), (596, 41), (597, 41), (604, 111), (621, 34), (622, 28), (642, 175), (650, 25), (651, 147), (661, 81), (696, 93), (709, 63)]
```
</details>

**Bash**
```
cd /work/scratch; python viz4.py r5.jpg dev/v3.npz 100,160,190,203,250,288,300,370,480,514,540,557
```

**Read**
```
/work/scratch/r5.jpg
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
cd /work/scratch; python viz4.py r6.jpg dev/v3.npz 596,604,615,630,642,646,651,661,680,696,709,740
```

**Read**
```
/work/scratch/r6.jpg
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
cd /work/scratch; python viz4.py r7.jpg dev/v3.npz 600,602,604,606,636,639,641,642,644,648,650,652
```

**Read**
```
/work/scratch/r7.jpg
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

> *thinking:* Spin now matches the strap in most frames, but there's still toward/away tilt ambiguity (e.g. frames 636–641). I'll add tilt-mirrored axis hypotheses and write the full `solve.py`, covering phase segmentation, hypotheses, spin sweep, Viterbi, joint smoothing, and depth re-anchoring.

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
mkdir -p /work/scratch/cache && cd /work/scratch/dev && python -c "
import pickle,sys; sys.path.insert(0,'/work/outputs')
import hoi_lib
s1=pickle.load(open('s1.pkl','rb'))
pickle.dump([(x[0],x[1],x[2],x[3]+x[4]) for x in s1],open('/work/scratch/cache/s1.pkl','wb'))
" && ln -sf /work/scratch/dev/D.pkl /work/scratch/cache/data.pkl && cd /work/scratch && python /work/outputs/solve.py --out /work/scratch/run1/object_poses_world.npz --cache /work/scratch/cache 2>&1 | tail -20
```

<details><summary>output</summary>

```
[    0.2s] data loaded 859 frames
[    0.6s] phases [('static_start', 0, 115), ('carried', 115, 777), ('static_end', 777, 859)]
[    0.8s] stage 1 (refined hypotheses) done 3436
[   65.2s] stage 2 (spin sweep) done 84552 states
[   65.7s] viterbi done
[  103.5s] joint refinement done
[  103.8s] saved /work/scratch/run1/object_poses_world.npz
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
cd /work/scratch; python -c "
import numpy as np; r=np.load('/work/inputs/reconstruction.npz'); np.savez('baseline.npz',T_world_mesh=r['T_world_mesh'].astype(np.float64))"
python /work/outputs/validate.py --traj baseline.npz --traj run1/object_poses_world.npz --names base,run1 --data_cache cache/data.pkl --phases '[["static_start",0,115],["carried",115,777],["static_end",777,859],["all",0,859]]' --out_json run1/val.json
```

<details><summary>output</summary>

```
| traj | phase | iou | iou_p10 | sil | col | blue | dark | step_med | n_step_gt20 | trans_acc_mm | dz_mean_cm | dz_median_cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | static_start | 0.814 | 0.786 | 0.508 | 0.267 | 0.050 | 0.020 | 0.082 | 2 | 4.535 | 0.000 | 0.000 |
| run1 | static_start | 0.951 | 0.934 | 0.089 | 0.230 | 0.044 | 0.014 | 0.031 | 0 | 0.029 | 0.080 | 0.000 |
| base | carried | 0.727 | 0.535 | 0.891 | 0.323 | 0.059 | 0.029 | 6.699 | 88 | 23.375 | 0.000 | 0.000 |
| run1 | carried | 0.902 | 0.866 | 0.208 | 0.260 | 0.054 | 0.017 | 4.017 | 18 | 7.028 | 0.074 | -0.075 |
| base | static_end | 0.807 | 0.720 | 0.518 | 0.219 | 0.066 | 0.020 | 0.000 | 4 | 11.850 | 0.000 | 0.000 |
| run1 | static_end | 0.937 | 0.926 | 0.108 | 0.144 | 0.043 | 0.011 | 0.048 | 0 | 1.43 …[+224 chars]
```
</details>

> *thinking:* The full run shows clear improvements over baseline on mask IoU, cap loss, and rotation jump count (94→18), while depth stays consistent. Now I'll visually inspect the remaining rotation jumps.

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, json
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
T=Tcw[None]@np.load('run1/object_poses_world.npz')['T_world_mesh']
st=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',T[1:,:3,:3],T[:-1,:3,:3])-1)/2,-1,1)))
print([(i+1,round(s)) for i,s in enumerate(st) if s>15])
np.savez('run1/cam.npz',R=T[:,:3,:3],t=T[:,:3,3])
EOF
```

<details><summary>output</summary>

```
[(202, 18), (203, 22), (204, 18), (287, 18), (288, 22), (289, 21), (290, 18), (512, 18), (513, 27), (514, 40), (515, 25), (622, 18), (623, 29), (624, 63), (625, 28), (626, 16), (641, 26), (642, 81), (643, 21), (655, 32), (699, 25), (700, 44), (701, 29), (708, 19), (709, 29), (710, 24), (711, 19), (745, 16)]
```
</details>

**Bash**
```
cd /work/scratch; python viz4.py r8.jpg run1/cam.npz 510,513,516,620,623,627,636,640,644,697,700,703
```

**Read**
```
/work/scratch/r8.jpg
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
cd /work/scratch; python - <<'EOF'
import numpy as np, json
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
r=np.load('/work/inputs/reconstruction.npz')
kp=r['human_keypoints_world'].astype(np.float64); cp=r['contact_prob_right_left']
kc=kp@Tcw[:3,:3].T+Tcw[:3,3]
Tb=Tcw[None]@r['T_world_mesh'].astype(np.float64)
d=np.load('run1/object_poses_world_diag.npz'); tref=d['t_refined']; tan=d['t_anchored']
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); ctr=m.vertices.mean(0)
def center(R,t): return np.einsum('nij,j->ni',R,ctr)+t
cb=center(Tb[:,:3,:3],Tb[:,:3,3]); cm=center(d['R_refined'],tref); ca=center(d['R_refined'],tan)
hands={'R':kc[:,21:42].mean(1),'L':kc[:,42:63].mean(1)}
print('frames, hand, mean dz(obj-hand) base / mine-raw / mine-anchored ; 3D dist')
for a,b in [(130,200),(200,260),(260,330),(330,400),(400,470),(470,540),(540,610),(610,680),(680,750)]:
    for hn,hi in [('R',0),('L',1)]:
        sel=[i for i in range(a,b) if cp[i,hi]>0.9]
        if len(sel)<10: continue
        h=hands[hn][sel]
        f=lambda c: (np.mean(c[sel,2]-h[:,2]), np.median(np.linalg.norm(c[sel]-h,axis=1)))
        print(a,b,hn,len(sel),' base %.3f %.3f | raw %.3f %.3f | anch %.3f %.3f'%(*f(cb),*f(cm),*f(ca)))
EOF
```

<details><summary>output</summary>

```
frames, hand, mean dz(obj-hand) base / mine-raw / mine-anchored ; 3D dist
130 200 L 63  base -0.153 0.769 | raw -0.186 0.783 | anch -0.138 0.775
200 260 R 45  base -0.085 0.182 | raw -0.155 0.263 | anch -0.106 0.225
200 260 L 43  base 0.062 0.115 | raw -0.018 0.095 | anch 0.040 0.105
260 330 R 64  base -0.192 0.608 | raw -0.247 0.606 | anch -0.193 0.591
260 330 L 10  base 0.016 0.066 | raw -0.066 0.106 | anch 0.022 0.078
330 400 L 66  base 0.098 0.607 | raw -0.031 0.559 | anch 0.103 0.608
400 470 R 31  base -0.043 0.189 | raw -0.106 0.222 | anch -0.039 0.171
400 470 L 40  base 0.027 0.226 | raw -0.046 0.223 | anch 0.025 0.246
470 540 R 69  base -0.132 0.349 | raw -0.202 0.377 | anch -0.138 0.351
470 540 L 19  base 0.093 0.099 | raw 0.034 0.061 | anch 0.101 0.112
540 610 R 62  base -0.006 0 …[+389 chars]
```
</details>

> *thinking:* The hand-contact depth cue isn't reliable enough to adjust the baseline, so I'll keep the depth re-anchored per phase/window without extra shifts. Next, I'll do a broad visual audit of the carried phase.

**Bash**
```
cd /work/scratch; python viz4.py r9.jpg run1/cam.npz 130,170,210,230,270,330,350,390,410,440,460,500 && python viz4.py r10.jpg run1/cam.npz 530,560,580,590,610,660,670,690,720,730,760,800
```

**Read**
```
/work/scratch/r9.jpg
```

**Read**
```
/work/scratch/r10.jpg
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
cd /work/scratch; python - <<'EOF'
import pickle, numpy as np, sys
sys.path.insert(0,'/work/outputs')
Rs,ts,un=pickle.load(open('cache/s2.pkl','rb'))
d=np.load('run1/object_poses_world_diag.npz'); R0=d['R_viterbi']
for f in [170,610,720,730]:
    k=[j for j in range(len(Rs[f])) if np.allclose(Rs[f][j],R0[f])][0]
    nh=len(Rs[f])//24
    print(f,'sel hyp',k//24,'spin',k%24*15,'u %.3f'%un[f][k])
    for h in range(nh):
        u=un[f][h*24:(h+1)*24]; print('   h',h,'ax',np.round(Rs[f][h*24][:,2],2),'min %.3f at %d'%(u.min(),u.argmin()*15),' '.join('%.2f'%x for x in u[::2]))
EOF
```

<details><summary>output</summary>

```
170 sel hyp 0 spin 15 u 0.593
   h 0 ax [ 0.16 -0.99  0.02] min 0.589 at 0 0.59 0.60 0.65 0.66 0.68 0.70 0.73 0.76 0.77 0.74 0.68 0.60
   h 1 ax [-0.11  0.81  0.58] min 0.615 at 45 0.68 0.62 0.63 0.70 0.75 0.73 0.67 0.63 0.65 0.68 0.71 0.73
   h 2 ax [-0.11  0.81 -0.58] min 0.741 at 270 0.81 0.79 0.82 0.88 0.94 0.95 0.89 0.78 0.77 0.74 0.80 0.82
610 sel hyp 0 spin 180 u 0.571
   h 0 ax [0.13 0.83 0.54] min 0.539 at 0 0.54 0.58 0.64 0.68 0.67 0.61 0.57 0.59 0.63 0.69 0.65 0.59
   h 1 ax [ 0.13  0.83 -0.54] min 0.605 at 15 0.61 0.62 0.66 0.71 0.72 0.73 0.74 0.77 0.76 0.78 0.73 0.68
   h 2 ax [-0.2  -0.98 -0.04] min 0.847 at 285 0.90 0.94 0.97 0.97 1.01 1.02 1.05 0.98 0.94 0.89 0.85 0.89
720 sel hyp 0 spin 75 u 0.561
   h 0 ax [0.27 0.96 0.04] min 0.561 at 75 0.57 0.60 0.57 0.57 0.59 0.63 0.6 …[+900 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > spinviz.py <<'EOF'
import pickle, numpy as np, cv2, sys
from viz import R, K
Rs,ts,un=pickle.load(open('cache/s2.pkl','rb'))
f=int(sys.argv[1]); h=int(sys.argv[2])
img=cv2.imread(f'/work/scratch/frames/{f:04d}.jpg'); c=np.load('/work/scratch/maskstats.npz')['c'][f]; S=110
x0=int(c[0]-S/2); y0=int(c[1]-S/2); crop=img[y0:y0+S,x0:x0+S]
tiles=[crop]
for k in range(0,24,2):
    j=h*24+k; T=np.eye(4); T[:3,:3]=Rs[f][j]; T[:3,3]=ts[f][j]
    col,mm,_=R.render(T,K,S,S,x0,y0); col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=mm[0].cpu().numpy()
    o=crop.copy(); o[mm]=col[mm]; cv2.putText(o,'%d %.2f'%(k*15,un[f][j]),(2,12),0,0.4,(0,0,255),1); tiles.append(o)
tiles.append(crop)
cv2.imwrite(sys.argv[3],cv2.resize(np.vstack([np.hstack(tiles[:7]),np.hstack(tiles[7:])]),None,fx=2,fy=2))
EOF
python spinviz.py 730 0 sp730.jpg
```

**Read**
```
/work/scratch/sp730.jpg
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
cd /work/scratch; python - <<'EOF'
import pickle, numpy as np, sys
sys.path.insert(0,'/work/outputs')
D=pickle.load(open('cache/data.pkl','rb'))
for f in [730,100,830,60]:
    img=D.img[f].astype(float)/255; bg=img[...,2]-img[...,1]
    om=D.om[f]; val=D.val[f]
    for th in [0.1,0.15,0.2]:
        b=bg>th
        print(f,th,'blue px',b.sum(),'in obj',(b&om).sum(),'invalid(person)',(b&~val).sum(),'neither',(b&val&~om).sum())
EOF
```

<details><summary>output</summary>

```
730 0.1 blue px 124 in obj 124 invalid(person) 0 neither 0
730 0.15 blue px 56 in obj 56 invalid(person) 0 neither 0
730 0.2 blue px 0 in obj 0 invalid(person) 0 neither 0
100 0.1 blue px 212 in obj 198 invalid(person) 1 neither 13
100 0.15 blue px 180 in obj 174 invalid(person) 0 neither 6
100 0.2 blue px 96 in obj 96 invalid(person) 0 neither 0
830 0.1 blue px 572 in obj 572 invalid(person) 0 neither 0
830 0.15 blue px 460 in obj 460 invalid(person) 0 neither 0
830 0.2 blue px 396 in obj 396 invalid(person) 0 neither 0
60 0.1 blue px 208 in obj 206 invalid(person) 2 neither 0
60 0.15 blue px 184 in obj 182 invalid(person) 2 neither 0
60 0.2 blue px 108 in obj 108 invalid(person) 0 neither 0
```
</details>

> *thinking:* The strap at frame 730 is motion-blurred, which washes out the blueness term's discriminative power between spins. I'm now adding a normalized strap-overlap metric using a soft-thresholded blue map, scoring mismatch relative to total strap energy to better capture presence/absence and partial overlap.

**Bash**
```
cd /work/scratch; python - <<'EOF'
s=open('/work/outputs/hoi_lib.py').read()
s=s.replace('''        return lsil, lcol, lblue, ldark

    def unary(self, ls, lc, lb, ld):
        return ls + self.wc * lc + self.wb * lb + self.wd * ld
''','''        # strap overlap: soft-thresholded blue maps (strap ~1, cyan body ~0), mismatch normalised by the total strap
        # energy -> ~0 if both show no strap, ~1 if only one of them shows it or they are disjoint
        sb = lambda x: torch.sigmoid((blue(x) - 0.08) / 0.02)
        reg = valid * torch.clamp(om + a, max=1)
        br = sb(cb) * a; bi = sb(ib) * reg
        lstrap = (((br - bi) ** 2) * reg).sum((1, 2)) / ((br ** 2 * reg).sum((1, 2)) + (bi ** 2).sum((1, 2)) + 20.)
        return lsil, lcol, lblue, ldark, lstrap

    def unary(self, ls, lc, lb, ld, lst):
        return ls + self.wc * lc + self.wb * lb + self.wd * ld + self.ws * lst
''')
s=s.replace("def __init__(self, R, K, S, wc=0.6, wb=4.0, wd=4.0):\n        self.R = R; self.S = S; self.wc = wc; self.wb = wb; self.wd = wd",
"def __init__(self, R, K, S, wc=0.6, wb=4.0, wd=4.0, ws=0.3):\n        self.R = R; self.S = S; self.wc = wc; self.wb = wb; self.wd = wd; self.ws = ws")
open('/work/outputs/hoi_lib.py','w').write(s)
EOF
grep -n "ws" /work/outputs/hoi_lib.py | head; cat > /work/scratch/strtest.py <<'EOF'
import pickle, numpy as np, sys, json, torch
sys.path.insert(0,'/work/outputs')
from hoi_lib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'] …[+665 chars]
```

<details><summary>output</summary>

```
107:    def __init__(self, R, K, S, wc=0.6, wb=4.0, wd=4.0, ws=0.3):
108:        self.R = R; self.S = S; self.wc = wc; self.wb = wb; self.wd = wd; self.ws = ws
150:        # energy -> ~0 if both show no strap, ~1 if only one of them shows it or they are disjoint
158:        return ls + self.wc * lc + self.wb * lb + self.wd * ld + self.ws * lst
20 strap argmin 15 range 0.08-0.93 | unary argmin 15 0.20 0.21 0.60 0.90 0.92 0.92 0.91 0.91 0.92 0.92 0.91 0.88
100 strap argmin 30 range 0.05-0.90 | unary argmin 30 0.53 0.05 0.37 0.84 0.90 0.89 0.90 0.90 0.90 0.90 0.90 0.88
830 strap argmin 0 range 0.05-0.96 | unary argmin 0 0.05 0.39 0.83 0.96 0.96 0.95 0.95 0.96 0.96 0.95 0.72 0.31
730 strap argmin 345 range 0.17-0.88 | unary argmin 345 0.24 0.79 0.84 0.84 0.84 0.85 0.85 0.86 0.88 0.86 0.61 0.26 …[+337 chars]
```
</details>

**Bash**
```
cd /work/scratch; mv cache/s2.pkl cache/s2_old.pkl; python /work/outputs/solve.py --out /work/scratch/run2/object_poses_world.npz --cache /work/scratch/cache 2>&1 | tail -8 && python /work/outputs/validate.py --traj baseline.npz --traj run1/object_poses_world.npz --traj run2/object_poses_world.npz --names base,run1,run2 --data_cache cache/data.pkl --phases '[["static_start",0,115],["carried",115,777],["static_end",777,859],["all",0,859]]' --out_json run2/val.json
```

<details><summary>output</summary>

```
[    0.1s] data loaded 859 frames
[    0.6s] phases [('static_start', 0, 115), ('carried', 115, 777), ('static_end', 777, 859)]
[    0.7s] stage 1 (refined hypotheses) done 3436
[   65.6s] stage 2 (spin sweep) done 84552 states
[   66.0s] viterbi done
[  103.5s] joint refinement done
[  104.0s] saved /work/scratch/run2/object_poses_world.npz
| traj | phase | iou | iou_p10 | sil | col | blue | dark | step_med | n_step_gt20 | trans_acc_mm | dz_mean_cm | dz_median_cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | static_start | 0.814 | 0.786 | 0.508 | 0.267 | 0.050 | 0.020 | 0.082 | 2 | 4.535 | 0.000 | 0.000 |
| run1 | static_start | 0.951 | 0.934 | 0.089 | 0.230 | 0.044 | 0.014 | 0.031 | 0 | 0.029 | 0.080 | 0.000 |
| run2 | static_start | 0.953 | 0.935 | 0.088 | 0.230 | 0.0 …[+991 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
s=open('/work/outputs/validate.py').read()
s=s.replace("dark=float(L[sl, 3].mean()), ","dark=float(L[sl, 3].mean()), strap=float(L[sl, 4].mean()), ")
s=s.replace("keys = ['iou', 'iou_p10', 'sil', 'col', 'blue', 'dark',","keys = ['iou', 'iou_p10', 'sil', 'col', 'blue', 'dark', 'strap',")
open('/work/outputs/validate.py','w').write(s)
import numpy as np, json
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
A=Tcw[None]@np.load('run1/object_poses_world.npz')['T_world_mesh']; B=Tcw[None]@np.load('run2/object_poses_world.npz')['T_world_mesh']
ang=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',A[:,:3,:3],B[:,:3,:3])-1)/2,-1,1)))
print('frames differing >30deg:',(ang>30).sum(), 'segments:')
idx=np.nonzero(ang>30)[0]
segs=[]; 
for i in idx:
    if segs and i-segs[-1][1]<=2: segs[-1][1]=i
    else: segs.append([i,i])
print(segs)
np.savez('run2/cam.npz',R=B[:,:3,:3],t=B[:,:3,3])
EOF
```

<details><summary>output</summary>

```
frames differing >30deg: 245 segments:
[[177, 203], [226, 290], [479, 492], [498, 506], [509, 512], [567, 624], [640, 666], [698, 700], [709, 746]]
```
</details>

**Bash**
```
cd /work/scratch; python viz4.py r11.jpg run1/cam.npz 185,195,240,270,485,580,600,650,720,735 run2/cam.npz
```

**Read**
```
/work/scratch/r11.jpg
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

> *thinking:* Run 2 (with the strap term) correctly matches the visible strap across all 10 tested frames, so I'm adopting it. With about 26 of 45 GPU minutes used, I'll now do a single clean end-to-end `solve.py` run with no caches, logging peak GPU memory.

**Bash**
```
cd /work/scratch; python - <<'EOF'
s=open('/work/outputs/solve.py').read()
s=s.replace("    log('saved', args.out)","    log('saved', args.out, '| peak GPU memory %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9))")
open('/work/outputs/solve.py','w').write(s)
EOF
mkdir -p /work/scratch/final_run; (time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz) > /work/scratch/final_run/log.txt 2>&1; tail -12 /work/scratch/final_run/log.txt
```

<details><summary>output</summary>

```
[   81.5s] data loaded 859 frames
[   81.9s] phases [('static_start', 0, 115), ('carried', 115, 777), ('static_end', 777, 859)]
[  213.4s] stage 1 (refined hypotheses) done 3436
[  246.8s] stage 2 (spin sweep) done 87312 states
[  247.3s] viterbi done
[  284.8s] joint refinement done
[  285.2s] saved /work/outputs/object_poses_world.npz | peak GPU memory 5.41 GB

real	4m48.151s
user	4m47.018s
sys	0m14.379s
```
</details>

**Bash**
```
cd /work/scratch; python /work/outputs/validate.py --traj baseline.npz --traj run2/object_poses_world.npz --traj /work/outputs/object_poses_world.npz --names base,run2,final --data_cache cache/data.pkl --phases '[["static_start",0,115],["carried",115,777],["static_end",777,859],["all",0,859]]' --out_json final_run/val.json; python - <<'EOF'
import numpy as np, json
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
A=Tcw[None]@np.load('run2/object_poses_world.npz')['T_world_mesh']; B=Tcw[None]@np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
ang=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',A[:,:3,:3],B[:,:3,:3])-1)/2,-1,1)))
print('final vs run2: frames >30deg',(ang>30).sum(),'median ang',np.median(ang), np.nonzero(ang>30)[0])
d=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(d.shape,d.dtype,np.isfinite(d).all())
np.savez('final_run/cam.npz',R=B[:,:3,:3],t=B[:,:3,3])
EOF
```

<details><summary>output</summary>

```
| traj | phase | iou | iou_p10 | sil | col | blue | dark | strap | step_med | n_step_gt20 | trans_acc_mm | dz_mean_cm | dz_median_cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | static_start | 0.814 | 0.786 | 0.508 | 0.267 | 0.050 | 0.020 | 0.193 | 0.082 | 2 | 4.535 | 0.000 | 0.000 |
| run2 | static_start | 0.953 | 0.935 | 0.088 | 0.230 | 0.044 | 0.014 | 0.044 | 0.017 | 0 | 0.034 | 0.080 | 0.000 |
| final | static_start | 0.951 | 0.936 | 0.088 | 0.229 | 0.044 | 0.014 | 0.046 | 0.029 | 0 | 0.041 | 0.080 | 0.000 |
| base | carried | 0.727 | 0.535 | 0.891 | 0.323 | 0.059 | 0.029 | 0.528 | 6.699 | 88 | 23.375 | 0.000 | 0.000 |
| run2 | carried | 0.902 | 0.869 | 0.211 | 0.269 | 0.052 | 0.018 | 0.315 | 3.894 | 14 | 6.996 | 0.065 | -0.085 |
| final | carried | 0.901 | 0.86 …[+959 chars]
```
</details>

**Bash**
```
cd /work/scratch; python viz4.py r12.jpg run2/cam.npz 226,230,234,566,640,644,648,697 final_run/cam.npz
```

**Read**
```
/work/scratch/r12.jpg
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
cd /work/scratch; python - <<'EOF'
import numpy as np, json, trimesh
d=np.load('/work/outputs/object_poses_world_diag.npz'); dj=json.load(open('/work/outputs/object_poses_world_diag.json'))
cam=json.load(open('/work/inputs/camera.json')); Tcw=np.linalg.inv(np.array(cam['T_world_cam'])); Twc=np.array(cam['T_world_cam'])
zb=(Tcw[None]@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64))[:,2,3]
for n,a,b in dj['phases']:
    print(n,a,b,'silhouette-only depth shift vs baseline median (cm): %.1f'%((np.median(d['t_refined'][a:b,2])-np.median(zb[a:b]))*100), 'baseline z med %.3f'%np.median(zb[a:b]))
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); dd=pl[3]
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=m.vertices[::20]
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
for i in [50,840]:
    for nm,TT in [('final',T),('base',Tb)]:
        w=v@TT[i,:3,:3].T+TT[i,:3,3]; s=w@n+dd
        print(i,nm,'lowest point above floor %.3f m; axis tilt from floor normal %.1f deg'%(s.min(), np.degrees(np.arccos(abs(TT[i,:3,2]@n)))))
EOF
```

<details><summary>output</summary>

```
static_start 0 115 silhouette-only depth shift vs baseline median (cm): -1.6 baseline z med 2.480
carried 115 777 silhouette-only depth shift vs baseline median (cm): -7.9 baseline z med 2.730
static_end 777 859 silhouette-only depth shift vs baseline median (cm): -0.3 baseline z med 2.355
50 final lowest point above floor 0.814 m; axis tilt from floor normal 4.7 deg
50 base lowest point above floor 0.803 m; axis tilt from floor normal 6.6 deg
840 final lowest point above floor 0.810 m; axis tilt from floor normal 2.0 deg
840 base lowest point above floor 0.804 m; axis tilt from floor normal 3.0 deg
```
</details>

**Bash**
```
python -c "
import torch,nvdiffrast,cv2,trimesh,h5py,scipy,numpy,PIL
for m in [torch,nvdiffrast,cv2,trimesh,h5py,scipy,numpy,PIL]: print(m.__name__, m.__version__)
"; python --version; ls /work/tools /work/scratch/pyuser 2>&1 | head; ls -la /work/outputs
```

<details><summary>output</summary>

```
torch 2.5.1+cu124
nvdiffrast 0.4.0
cv2 4.11.0
trimesh 5.1.1
h5py 3.16.0
scipy 1.16.3
numpy 1.26.3
PIL 10.2.0
Python 3.11.10
ls: cannot access '/work/scratch/pyuser': No such file or directory
/work/tools:
total 460
drwxrwxr-x 3 hanzhou hanzhou   4096 Oct 21 20:25 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 21 19:37 ..
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 21 20:17 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou   8257 Oct 21 20:17 hoi_lib.py
-rw-rw-r-- 1 hanzhou hanzhou 110230 Oct 21 20:25 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou 126183 Oct 21 20:25 object_poses_world_diag.json
-rw-rw-r-- 1 hanzhou hanzhou 187921 Oct 21 20:25 object_poses_world_diag.npz
-rw-rw-r-- 1 hanzhou hanzhou  12696 Oct 21 20:20 solve.py
-rw-rw-r-- 1 hanzhou hanzhou   4029 Oct 21 20:19 validate.py
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
grep -n "scratch" /work/outputs/solve.py /work/outputs/hoi_lib.py; rm -rf /work/outputs/__pycache__; ls /work/outputs; cat > /work/scratch/.claude/projects/-work/memory/hoi-bottle-pose-insights.md <<'EOF'
---
name: hoi-bottle-pose-insights
description: Lessons from the CARI4D object-trajectory task (episode 25, cyan bottle) — render-and-compare pitfalls and spin cues
metadata:
  type: project
---
Task (2026-10-21): recover per-frame 6-DoF object poses for a monocular HOI video; outputs in /work/outputs (solve.py, hoi_lib.py, validate.py, REPORT.md).

- nvdiffrast: texture/uv must be broadcast (batch 1), not expanded per batch (6x slower). A welded watertight mesh made dr.antialias silhouette gradients diverge; the raw triangle soup converges.
- Near-cylindrical bottle: spin about the long axis dominates ADD error (180 deg spin = 9 cm). Plain colour L1 picks the wrong spin; a normalised blue-strap overlap term (soft-thresholded B-G maps) is the reliable cue. Gradient refinement does not explore spin, so use a discrete 24-spin sweep + Viterbi.
- Silhouette-only depth was biased ~5% too close while carried (motion blur / hand), so anchor depth to the baseline.

**Why:** these cost most of the GPU budget to discover.
**How to apply:** start similar episodes with the spin sweep + strap/feature unary + Viterbi + joint smoothing pipeline.
EOF
echo "- [HOI bottle pose insights](hoi-bottle-pose-insights.md) — nvdiffrast pitfalls, spin cue, depth anchoring for CARI4D trajectory task" > …[+55 chars]
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
validate.py
```
</details>

**Agent:** I ran `solve.py` once from start to finish with no caches. It wrote `/work/outputs/object_poses_world.npz`: `T_world_mesh`, 859×4×4, float64, all finite. The run took 4 min 48 s, about 3.4 min of it on the GPU, with peak GPU memory 5.4 GB. Total GPU use across development was about 30 of the 45 allowed minutes. `REPORT.md` and `TOOLS.md` are written. I don't have ground truth, so "better" below means it explains the video better on my own checks.

**Phases**, found automatically from where the object mask's centre sits:
- **Frames 0–114:** the bottle rests untouched on the table.
- **Frames 115–776:** it is carried, rotated and passed between hands.
- **Frames 777–858:** it is back on the table.

**Method:**
1. **Scoring a pose:** I render the textured mesh and compare it with the frame. I score the silhouette against the object mask, ignoring pixels covered by the person, plus colour, the dark cap and the blue strap.
2. **Hypotheses:** each frame starts from the baseline and FoundationPose poses and their end-to-end flips, each refined by optimisation. Then I add the toward/away mirror of each axis and 24 spin angles about the bottle's long axis.
3. **Choosing over time:** a Viterbi pass picks one hypothesis per frame, with jumps penalised 20 times more in the resting phases.
4. **Smoothing:** all frames are refined jointly with smoothness terms, and each resting phase shares one pose.
5. **Depth:** positions are slid along the viewing ray so the median depth matches the baseline's, per resting phase and in a 61-frame window while carried.

**Spin about the long axis** turned out to be the key accuracy issue. A 180° spin error moves surface points by about 9 cm. The only strong cue is a blue strap near the cap, and a plain colour comparison picked the wrong spin. A dedicated strap-matching term fixed this: in the resting phases every sampled frame now agrees on the spin, and it matches what I see in the video.

**Checks, baseline → result:**

| | Mask overlap (IoU) | Rotation jumps >20° | Strap mismatch |
|---|---|---|---|
| Start, resting | 0.81 → 0.95 | 2 → 0 | 0.19 → 0.05 |
| Carried | 0.73 → 0.90 | 88 → 12 | 0.53 → 0.31 |
| End, resting | 0.81 → 0.95 | 4 → 0 | 0.30 → 0.05 |

The strap term is part of what the method optimises, so that column is supporting rather than independent evidence. I also visually compared the poses against the video crops on over 100 frames.

- **Upright pose:** the bottle now stands closer to vertical on the level table (4.7° / 2.0° tilt, versus the baseline's 6.6° / 3.0°).
- **Table height:** its base is at the same height at the start and the end, within 4 mm.

**Depth:** median depth equals the baseline's in every phase, and I applied no extra shift. The silhouette size suggests −1.6, −7.9 and −0.3 cm by phase; the large value while carried is biased by motion blur and the hand. The hand-contact cue was inconclusive: hand-to-bottle distances were 0.06–0.77 m even when contact probability was high. No two independent cues agreed on a non-zero correction.

**Remaining weak spots:**
- **Frames ~636–641:** the bottle points roughly at the camera with the cap hidden by the hand, and my result still tilts the cap toward the camera instead of away.
- **Strap hidden:** in long stretches where the strap is out of view, the spin comes only from continuity with neighbouring frames.

Confidence is high for rotation in both resting phases (depth there is the baseline's), and medium-high for the carried phase.

No external tools, weights or packages were downloaded; everything used came with the container.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoi_lib.py`
- `validate.py`

## Final

I ran `solve.py` once from start to finish with no caches. It wrote `/work/outputs/object_poses_world.npz`: `T_world_mesh`, 859×4×4, float64, all finite. The run took 4 min 48 s, about 3.4 min of it on the GPU, with peak GPU memory 5.4 GB. Total GPU use across development was about 30 of the 45 allowed minutes. `REPORT.md` and `TOOLS.md` are written. I don't have ground truth, so "better" below means it explains the video better on my own checks.

**Phases**, found automatically from where the object mask's centre sits:
- **Frames 0–114:** the bottle rests untouched on the table.
- **Frames 115–776:** it is carried, rotated and passed between hands.
- **Frames 777–858:** it is back on the table.

**Method:**
1. **Scoring a pose:** I render the textured mesh and compare it with the frame. I score the silhouette against the object mask, ignoring pixels covered by the person, plus colour, the dark cap and the blue strap.
2. **Hypotheses:** each frame starts from the baseline and FoundationPose poses and their end-to-end flips, each refined by optimisation. Then I add the toward/away mirror of each axis and 24 spin angles about the bottle's long axis.
3. **Choosing over time:** a Viterbi pass picks one hypothesis per frame, with jumps penalised 20 times more in the resting phases.
4. **Smoothing:** all frames are refined jointly with smoothness terms, and each resting phase shares one pose.
5. **Depth:** positions are slid along the viewing ray so the median depth matches the baseline's, per resting phase and in a 61-frame window while carried.

**Spin about the long axis** turned out to be the key accuracy issue. A 180° spin error moves surface points by about 9 cm. The only strong cue is a blue strap near the cap, and a plain colour comparison picked the wrong spin. A dedicated strap-matching term fixed this: in the resting phases every sampled frame now agrees on the spin, and it matches what I see in the video.

**Checks, baseline → result:**

| | Mask overlap (IoU) | Rotation jumps >20° | Strap mismatch |
|---|---|---|---|
| Start, resting | 0.81 → 0.95 | 2 → 0 | 0.19 → 0.05 |
| Carried | 0.73 → 0.90 | 88 → 12 | 0.53 → 0.31 |
| End, resting | 0.81 → 0.95 | 4 → 0 | 0.30 → 0.05 |

The strap term is part of what the method optimises, so that column is supporting rather than independent evidence. I also visually compared the poses against the video crops on over 100 frames.

- **Upright pose:** the bottle now stands closer to vertical on the level table (4.7° / 2.0° tilt, versus the baseline's 6.6° / 3.0°).
- **Table height:** its base is at the same height at the start and the end, within 4 mm.

**Depth:** median depth equals the baseline's in every phase, and I applied no extra shift. The silhouette size suggests −1.6, −7.9 and −0.3 cm by phase; the large value while carried is biased by motion blur and the hand. The hand-contact cue was inconclusive: hand-to-bottle distances were 0.06–0.77 m even when contact probability was high. No two independent cues agreed on a non-zero correction.

**Remaining weak spots:**
- **Frames ~636–641:** the bottle points roughly at the camera with the cap hidden by the hand, and my result still tilts the cap toward the camera instead of away.
- **Strap hidden:** in long stretches where the strap is out of view, the spin comes only from continuity with neighbouring frames.

Confidence is high for rotation in both resting phases (depth there is the baseline's), and medium-high for the carried phase.

No external tools, weights or packages were downloaded; everything used came with the container.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `hoi_lib.py`
- `validate.py`
