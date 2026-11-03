# CL2b_ep015

Claude Opus v2（ep11-21）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T10:58:44Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude/projects/-work/memory/ 2>/dev/null && cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null; nvidia-smi
```

<details><summary>output</summary>

```
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 14:52 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 14:52 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 14:52 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 14:52 proxy.sock

inputs:
total 75840
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 14:52 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 14:52 ..
-rw-rw-r-- 1 hanzhou hanzhou      829 Oct 21 14:52 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       69 Oct 21 14:52 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   193840 Oct 21 14:52 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      398 Oct 21 14:52 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  6682914 Oct 21 14:52 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 13022116 Oct 21 14:52 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 3616 …[+4383 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, h5py
r=np.load('inputs/reconstruction.npz')
for k in r: print(k, r[k].shape, r[k].dtype)
f=h5py.File('inputs/masks.h5'); 
for k in f: print(k, f[k].shape, f[k].dtype, f[k].chunks, f[k].compression)
import trimesh
m=trimesh.load('inputs/object_mesh.glb', force='scene', process=False).to_geometry()
print(m.vertices.shape, m.faces.shape, m.bounds, m.extents)
print(type(m.visual), getattr(m.visual,'material',None))
"; ls /pipeline /pipeline/*; ls /code/reconstruction
```

<details><summary>output</summary>

```
T_world_mesh (715, 4, 4) float32
T_cam_mesh_foundationpose (715, 4, 4) float32
T_cam_mesh_coconet (715, 4, 4) float32
human_vertices_world (715, 4610, 3) float32
human_keypoints_world (715, 70, 3) float32
contact_prob_right_left (715, 2) float32
object (715, 1152, 1536) bool (23, 72, 96) gzip
person (715, 1152, 1536) bool (23, 72, 96) gzip
(199398, 3) (66486, 3) [[-0.13955034 -0.14493057 -0.30348039]
 [ 0.14996175  0.13710172  0.34234235]] [0.2895121  0.2820323  0.64582273]
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7c2946790110>
/pipeline/pipeline_report.json

/pipeline:
depth
export
foundationpose
inference
mhr_init
pipeline_report.json
work

/pipeline/depth:
moge2

/pipeline/export:
episode_000015

/pipeline/foundationpose:
debug
epi …[+429 chars]
```
</details>

**Bash**
```
cd /work; cat /pipeline/pipeline_report.json | head -100; ls -R /pipeline/depth | head; ls /pipeline/export/*; ls /pipeline/foundationpose/debug | head; ls /pipeline/work/masks | head; cat logs/run_manifest.txt
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
    "mtime_ns": 1791258297305597939,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000015/inference/episode_000015_step200000_before_after.mp4",
    "size": 51892741
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+4110 chars]
```
</details>

**Agent:** Let me extract frames and view the video at a coarse sampling.

**Bash**
```
mkdir -p /work/scratch/frames /work/outputs; cd /work/scratch; python -c "
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT))
i=0
while True:
  ok,f=cap.read()
  if not ok: break
  cv2.imwrite(f'frames/{i:04d}.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 92]); i+=1
print(i)
"; python -c "
import cv2, numpy as np
ims=[cv2.resize(cv2.imread(f'/work/scratch/frames/{i:04d}.jpg'),(384,288)) for i in range(0,715,30)]
while len(ims)%6: ims.append(np.zeros_like(ims[0]))
rows=[np.hstack(ims[i:i+6]) for i in range(0,len(ims),6)]
for j,i in enumerate(range(0,715,30)): pass
g=np.vstack(rows)
cv2.imwrite('/work/scratch/contact.jpg', g)
"
```

<details><summary>output</summary>

```
30.0 715.0
715
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
cd /work/scratch; cat > /work/scratch/ana1.py <<'EOF'
import numpy as np, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
cp=r['contact_prob_right_left']
def ang(A,B): 
    return np.degrees(np.linalg.norm(R.from_matrix(A[...,:3,:3].transpose(0,2,1)@B[...,:3,:3]).as_rotvec(),axis=-1))
dR=ang(T[:-1],T[1:]); dt=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)
for i in range(0,715,5):
    print(i, 'pos',np.round(T[i,:3,3],3),'camz %.3f'%Tc[i,2,3], 'fpz %.3f ccz %.3f'%(fp[i,2,3],cc[i,2,3]), 'dR %.1f dt %.3f'%(dR[min(i,713)],dt[min(i,713)]), 'cont',np.round(cp[i],2),
      'up',np.round(T[i,:3,:3]@np.array([0,0,1.]),2), 'R-fp %.1f R-cc %.1f'%(ang(Tc[i:i+1],fp[i:i+1])[0],ang(Tc[i:i+1],cc[i:i+1])[0]))
EOF
python ana1.py
```

<details><summary>output</summary>

```
0 pos [0.07  0.86  3.616] camz 3.579 fpz 3.668 ccz 3.513 dR 0.0 dt 0.001 cont [0. 0.] up [-0.02 -1.   -0.01] R-fp 3.0 R-cc 1.2
5 pos [0.07  0.859 3.614] camz 3.577 fpz 3.660 ccz 3.513 dR 0.0 dt 0.001 cont [0. 0.] up [-0.02 -1.   -0.01] R-fp 4.0 R-cc 1.2
10 pos [0.07  0.857 3.607] camz 3.570 fpz 3.682 ccz 3.513 dR 0.0 dt 0.004 cont [0. 0.] up [-0.02 -1.   -0.01] R-fp 3.8 R-cc 1.2
15 pos [0.07  0.858 3.61 ] camz 3.573 fpz 3.709 ccz 3.513 dR 0.0 dt 0.001 cont [0. 0.] up [-0.02 -1.   -0.01] R-fp 4.4 R-cc 1.2
20 pos [0.069 0.854 3.595] camz 3.558 fpz 3.731 ccz 3.513 dR 0.0 dt 0.000 cont [0. 0.] up [-0.02 -1.   -0.01] R-fp 3.7 R-cc 1.2
25 pos [0.069 0.854 3.593] camz 3.556 fpz 3.707 ccz 3.513 dR 0.0 dt 0.000 cont [0. 0.] up [-0.02 -1.   -0.01] R-fp 3.8 R-cc 1.2
30 pos [0.069 0.853 3.591] camz 3. …[+17992 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import trimesh, numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False)
print(s.geometry.keys(), s.graph.nodes_geometry)
m=s.to_geometry()
v=m.vertices
print(m.visual.uv.shape, m.visual.material.baseColorTexture.size if m.visual.material.baseColorTexture else None)
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
# distribution along z
import numpy as np
for z in np.linspace(-0.3,0.34,9):
  sel=np.abs(v[:,2]-z)<0.02
  if sel.sum(): print('z %.2f'%z, sel.sum(), 'x[%.3f %.3f] y[%.3f %.3f]'%(v[sel,0].min(),v[sel,0].max(),v[sel,1].min(),v[sel,1].max()))
print('centroid', v.mean(0), m.center_mass if m.is_watertight else 'nw')
"
python -c "import nvdiffrast, torch; print(torch.__version__, nvdiffrast.__version__)"; pip list 2>/dev/null | grep -i -E "torch|opencv|trimesh|open3d|scipy|kornia|pytorch3d|lietorch|roma|h5py|imageio|av"
```

<details><summary>output</summary>

```
odict_keys(['wetfloorsign01']) ['wetfloorsign01']
(199398, 2) (4755, 4626)
[[-6.03289009e-08  9.99999999e-01  4.02355273e-05  3.92596758e-03]
 [ 9.99994608e-01 -7.17965332e-08  3.28379215e-03 -4.22300917e-03]
 [ 3.28379215e-03  4.02355084e-05 -9.99994608e-01  1.96533364e-02]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] [0.28203242 0.28952054 0.64536667]
z -0.30 7758 x[-0.140 0.150] y[-0.145 0.137]
z -0.22 21232 x[-0.131 0.139] y[-0.143 0.132]
z -0.14 8455 x[-0.119 0.126] y[-0.138 0.127]
z -0.06 9553 x[-0.105 0.108] y[-0.136 0.124]
z 0.02 9866 x[-0.085 0.090] y[-0.132 0.120]
z 0.10 7637 x[-0.069 0.071] y[-0.128 0.115]
z 0.18 13562 x[-0.051 0.053] y[-0.124 0.121]
z 0.26 8731 x[-0.029 0.027] y[-0.119 0.117]
z 0.34 12811 x[-0.016 0.015] y[-0.101 0.093]
centroid [ 0.00232 …[+974 chars]
```
</details>

**Bash**
```
mkdir -p /work/outputs; cat > /work/outputs/common.py <<'EOF'
"""Shared utilities: mesh loading, nvdiffrast rendering, data loading."""
import json, numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr

H, W = 1152, 1536

def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    return m

class Renderer:
    def __init__(self, mesh, K, device="cuda", H=H, W=W):
        self.dev = device
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=device)
        uv = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.K = np.asarray(K, dtype=np.float64)
        self.H, self.W = H, W

    def proj_mat(self, K, H, W, n=0.05, f=20.0):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        # OpenCV camera -> clip space (nvdiffrast, y up in NDC -> we flip so row 0 = top)
        P = np.array([[2 * fx / W, 0, (2 * cx - W) / W, 0],
                      [0, 2 * fy / H, (2 * cy - H) / H, 0],
                      [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                       …[+2692 chars]
```

<details><summary>output</summary>

```
Exit code 1
Traceback (most recent call last):
  File "/work/scratch/texview.py", line 17, in <module>
    cv2.putText(im,f'yaw{yaw}',(10,30),0,1,(0,0,255),2); ims.append(im)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'putText'
> Overload resolution failed:
>  - Layout of the output array img is incompatible with cv::Mat
>  - Expected Ptr<cv::UMat> for argument 'img'
```
</details>

**Bash**
```
sed -i 's/\[:,:,::-1\]$/[:,:,::-1].copy()/' texview.py && sed -i "s/astype(np.uint8)\[:,:,::-1\]/astype(np.uint8)[:,:,::-1].copy()/" texview.py && python texview.py
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
cat > overlay.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import argparse
p=argparse.ArgumentParser(); p.add_argument('--poses',default='/work/inputs/reconstruction.npz'); p.add_argument('--key',default='T_world_mesh')
p.add_argument('--frames',default='0,100,200'); p.add_argument('--out',default='ov.jpg'); p.add_argument('--cam',default='world')
p.add_argument('--size',type=int,default=320); p.add_argument('--K',default='K')
a=p.parse_args()
K,Kp,Twc=load_camera('/work/inputs')
m=load_mesh('/work/inputs/object_mesh.glb')
T=np.load(a.poses)[a.key].astype(np.float64)
Tc=np.linalg.inv(Twc)@T if a.cam=='world' else T
KK=K if a.K=='K' else Kp
r=Renderer(m,KK)
f=h5py.File('/work/inputs/masks.h5')
frames=[int(x) for x in a.frames.split(',')]
tiles=[]
for i in frames:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    om=f['object'][i]
    mk,rgb,d,_=r.render(Tc[i][None])
    mk=mk[0].cpu().numpy(); rgb=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8)
    ys,xs=np.nonzero(mk|om)
    if len(xs)==0: ys,xs=np.array([576]),np.array([768])
    cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2; s=max(xs.max()-xs.min(),ys.max()-ys.min())*1.3+40
    x0,y0=int(cx-s/2),int(cy-s/2); s=int(s)
    blend=img.copy(); blend[mk]=(0.5*img[mk]+0.5*rgb[mk]).astype(np.uint8)
    cs,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(blend,cs,-1,(0,255,0),1)
    cs,_=cv2.findContours(mk.astype(np.uint8),cv2.R …[+659 chars]
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
python overlay.py --frames 470,480,488,492,495,498,502,506,512 --out ov2.jpg --size 280; python overlay.py --frames 520,560,580,600,610,618,622,630,700 --out ov3.jpg --size 280
```

**Read**
```
/work/scratch/ov2.jpg
```

**Read**
```
/work/scratch/ov3.jpg
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
cat > iou.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs')
m=load_mesh('/work/inputs/object_mesh.glb')
r=Renderer(m,K)
f=h5py.File('/work/inputs/masks.h5')
def iou_all(Tw, name):
    Tc=np.linalg.inv(Twc)@Tw
    out=[]
    for i in range(len(Tw)):
        om=torch.from_numpy(f['object'][i]).cuda(); pm=torch.from_numpy(f['person'][i]).cuda()
        mk=r.render(Tc[i][None],want_tex=False)[0][0]
        vis=mk&~pm
        inter=(vis&om).sum().item(); uni=(vis|om).sum().item()
        out.append((inter/max(uni,1), mk.sum().item(), vis.sum().item(), om.sum().item()))
    return np.array(out)
if __name__=='__main__':
    rr=np.load('/work/inputs/reconstruction.npz')
    a=iou_all(rr['T_world_mesh'].astype(np.float64),'base')
    np.save('iou_base.npy',a)
    for i in range(0,715,5): print(i, np.round(a[i],3))
EOF
python iou.py | awk '{printf "%s  ", $0; if (NR%3==0) print ""}'
```

<details><summary>output</summary>

```
0 [8.280e-01 8.172e+03 8.172e+03 8.610e+03]  5 [8.230e-01 8.182e+03 8.182e+03 8.613e+03]  10 [8.250e-01 8.214e+03 8.214e+03 8.702e+03]  
15 [8.280e-01 8.199e+03 8.199e+03 8.449e+03]  20 [8.280e-01 8.275e+03 8.275e+03 8.433e+03]  25 [8.280e-01 8.281e+03 8.281e+03 8.435e+03]  
30 [8.270e-01 8.290e+03 8.290e+03 8.448e+03]  35 [8.280e-01 8.281e+03 8.281e+03 8.437e+03]  40 [8.290e-01 8.268e+03 8.268e+03 8.439e+03]  
45 [8.290e-01 8.230e+03 8.230e+03 8.418e+03]  50 [8.280e-01 8.186e+03 8.186e+03 8.427e+03]  55 [8.290e-01 8.169e+03 8.169e+03 8.429e+03]  
60 [8.290e-01 8.071e+03 8.071e+03 8.421e+03]  65 [8.260e-01 8.033e+03 8.033e+03 8.477e+03]  70 [8.290e-01 8.022e+03 8.022e+03 8.425e+03]  
75 [8.270e-01 7.993e+03 7.993e+03 8.505e+03]  80 [8.280e-01 7.960e+03 7.960e+03 8.486e+03]  85 [8.280e-01 7 …[+6052 chars]
```
</details>

> *thinking:* The object is a wet-floor A-frame sign; baseline overlays are decent but show depth jitter during the static start, a rotation jump during full occlusion (~488–500), and low mask IoU in edge-on views (375–400, 560–714). I'll run a differentiable silhouette refinement and examine those problem frames more closely.

**Bash**
```
python overlay.py --frames 390,650,560 --out ov4.jpg --size 600
```

**Read**
```
/work/scratch/ov4.jpg
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
cat >> /work/outputs/common.py <<'EOF'

def crop_K(K, crop, S):
    x0, y0, cw, ch = crop
    Kc = np.asarray(K, dtype=np.float64).copy()
    Kc[0, 2] -= x0; Kc[1, 2] -= y0
    Kc[0] *= S / cw; Kc[1] *= S / ch
    return Kc

def proj_from_K(K, H, W, n=0.05, f=20.0):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([[2 * fx / W, 0, (2 * cx - W) / W, 0],
                     [0, 2 * fy / H, (2 * cy - H) / H, 0],
                     [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                     [0, 0, 1, 0]], dtype=np.float64)

def rotvec_to_mat(w):
    """Rodrigues, torch (B,3) -> (B,3,3), differentiable."""
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1] = -k[:, 2]; Kx[:, 0, 2] = k[:, 1]
    Kx[:, 1, 0] = k[:, 2]; Kx[:, 1, 2] = -k[:, 0]
    Kx[:, 2, 0] = -k[:, 1]; Kx[:, 2, 1] = k[:, 0]
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)

def render_batch(r, T_cam_mesh, Ps, S, with_tex=False):
    """T_cam_mesh torch (B,4,4), Ps torch (B,4,4) per-frame projection; returns soft silhouette (B,S,S) [, rgb]."""
    vh = torch.cat([r.v, torch.ones_like(r.v[:, :1])], 1)
    vc = torch.einsum('bij,nj->bni', T_cam_mesh, vh)
    clip = torch.einsum('bij,bnj->bni', Ps, vc).contiguous()
    rast, _ = dr.rasterize(r.ctx, clip, r.f,  …[+3134 chars]
```

<details><summary>output</summary>

```
6.575788259506226
0 iou 0.827 -> 0.877 dR 8.2 dt [-0.003  0.069  0.178]
100 iou 0.833 -> 0.876 dR 5.7 dt [-0.002  0.053  0.117]
200 iou 0.871 -> 0.950 dR 6.0 dt [-0.001 -0.023 -0.173]
300 iou 0.708 -> 0.883 dR 6.3 dt [-0.03   0.092  0.199]
305 iou 0.618 -> 0.841 dR 6.3 dt [-0.044  0.102  0.226]
350 iou 0.912 -> 0.961 dR 4.4 dt [-0.015 -0.012 -0.051]
380 iou 0.523 -> 0.621 dR 1.7 dt [0.017 0.041 0.047]
390 iou 0.447 -> 0.729 dR 14.1 dt [0.025 0.052 0.076]
400 iou 0.490 -> 0.595 dR 9.2 dt [0.003 0.031 0.04 ]
450 iou 0.785 -> 0.870 dR 5.6 dt [0.013 0.029 0.127]
560 iou 0.576 -> 0.779 dR 8.8 dt [-0.025  0.071  0.077]
580 iou 0.534 -> 0.617 dR 3.6 dt [-0.022  0.034  0.059]
650 iou 0.590 -> 0.754 dR 11.5 dt [-0.023  0.062  0.073]
700 iou 0.607 -> 0.761 dR 10.5 dt [-0.024  0.065  0.084]
```
</details>

**Bash**
```
python -c "
import numpy as np, json
d=np.load('ref_test.npz'); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64).copy()
for k,i in enumerate(d['idx']): T[i]=Twc@d['T'][k]
np.savez('ref_test_world.npz',T_world_mesh=T)
"; python overlay.py --poses ref_test_world.npz --frames 0,650,390 --out ov5.jpg --size 600; python overlay.py --frames 0 --out ov5b.jpg --size 900; python overlay.py --poses ref_test_world.npz --frames 0 --out ov5c.jpg --size 900
```

**Read**
```
/work/scratch/ov5b.jpg
```

**Read**
```
/work/scratch/ov5c.jpg
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
cat > refine2.py <<'EOF'
# per-frame silhouette + photometric refinement test
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import time, torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
torch.manual_seed(0)
K,Kp,Twc=load_camera('/work/inputs')
m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
f=h5py.File('/work/inputs/masks.h5')
rr=np.load('/work/inputs/reconstruction.npz'); Tw=rr['T_world_mesh'].astype(np.float64)
Tc0=np.linalg.inv(Twc)@Tw
S=384
def gblur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-(torch.arange(k,device=x.device)-k//2)**2/(2*s*s)); g=g/g.sum()
    C=x.shape[1]
    x=F.conv2d(x,g.view(1,1,1,k).repeat(C,1,1,1),padding=(0,k//2),groups=C)
    return F.conv2d(x,g.view(1,1,k,1).repeat(C,1,1,1),padding=(k//2,0),groups=C)
def prep(idx, Tc):
    crops=[];tg=[];vd=[];ims=[]
    for i in idx:
        om=f['object'][i]; pm=f['person'][i]
        img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[:,:,::-1].astype(np.float32)/255
        mk=r.render(Tc[i][None],want_tex=False)[0][0].cpu().numpy()
        ys,xs=np.nonzero(om|mk)
        cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2; s=max(xs.max()-xs.min(),ys.max()-ys.min())*1.25+20
        crop=(cx-s/2,cy-s/2,s,s); crops.append(crop)
        M=np.array([[S/s,0,-(cx-s/2)*S/s],[0,S/s,-(cy-s/2)*S/s]])
        fl=cv2.INTER_AREA if s>S else cv2.INTER_LINEAR
        o=cv2.warpAffine(om.astype(np.float32),M,(S,S),flags=fl)
        p=cv2.warpAffine(pm.astype(np.float32), …[+2701 chars]
```

<details><summary>output</summary>

```
0 iou 0.827 -> 0.837 photo 1.125 -> 0.942 dR 5.5 dt [-0.009  0.044  0.191]
100 iou 0.833 -> 0.856 photo 1.139 -> 0.976 dR 4.7 dt [-0.004  0.022  0.078]
200 iou 0.871 -> 0.942 photo 0.944 -> 0.648 dR 4.0 dt [ 0.    -0.023 -0.132]
300 iou 0.708 -> 0.835 photo 1.363 -> 1.222 dR 7.2 dt [-0.011  0.045  0.093]
305 iou 0.618 -> 0.796 photo 1.614 -> 1.459 dR 8.8 dt [-0.031  0.066  0.171]
350 iou 0.912 -> 0.959 photo 0.897 -> 0.637 dR 3.5 dt [-0.013 -0.013 -0.041]
380 iou 0.523 -> 0.555 photo 1.291 -> 1.066 dR 2.5 dt [ 0.004 -0.001  0.035]
390 iou 0.447 -> 0.536 photo 1.565 -> 1.386 dR 5.5 dt [0.003 0.01  0.019]
400 iou 0.490 -> 0.547 photo 1.498 -> 1.244 dR 5.3 dt [-0.005  0.001  0.023]
450 iou 0.785 -> 0.759 photo 1.656 -> 1.708 dR 4.8 dt [-0.01   0.003  0.078]
560 iou 0.576 -> 0.735 photo 1.460  …[+1349 chars]
```
</details>

**Bash**
```
sed -i 's/^cols=3/cols=min(3,len(tiles))/' overlay.py; python overlay.py --frames 650 --out a.jpg --size 700; python overlay.py --poses ref2_world_0.3.npz --frames 650 --out b.jpg --size 700; python -c "
import cv2,numpy as np; a=cv2.imread('a.jpg'); b=cv2.imread('b.jpg'); cv2.imwrite('ab.jpg', np.hstack([a,b[:,700:]]))"
```

**Read**
```
/work/scratch/ab.jpg
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
cat > ana2.py <<'EOF'
import numpy as np, json
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
kp=r['human_keypoints_world'].astype(np.float64)
z=Tc[:,2,3]; zs=savgol_filter(z,21,2)
handle=np.array([0,0,0.33])  # mesh-frame top
hw=(T[:,:3,:3]@handle)+T[:,:3,3]
lh=kp[:,42:63].mean(1); rh=kp[:,21:42].mean(1)
hc=(np.linalg.inv(Twc)[None,:3,:3]@lh[...,None])[...,0]+np.linalg.inv(Twc)[:3,3]
for i in range(150,640,6):
    print(i,'z %.3f hf %.3f'%(z[i],z[i]-zs[i]),'handle-lh',np.round(hw[i]-lh[i],3),'|%.3f|'%np.linalg.norm(hw[i]-lh[i]), 'lhz %.3f'%hc[i,2], 'cp',np.round(r['contact_prob_right_left'][i],2))
print('hf std', np.std((z-zs)[165:600]))
EOF
python ana2.py
```

<details><summary>output</summary>

```
150 z 3.601 hf 0.001 handle-lh [ 0.222  0.17  -0.508] |0.579| lhz 4.079 cp [0. 1.]
156 z 3.598 hf -0.001 handle-lh [ 0.259  0.171 -0.481] |0.572| lhz 4.048 cp [0. 1.]
162 z 3.598 hf 0.002 handle-lh [ 0.231  0.168 -0.479] |0.558| lhz 4.047 cp [0. 1.]
168 z 3.636 hf 0.036 handle-lh [ 0.199  0.174 -0.478] |0.546| lhz 4.087 cp [0. 1.]
174 z 3.663 hf 0.027 handle-lh [ 0.201  0.145 -0.457] |0.519| lhz 4.104 cp [0. 1.]
180 z 3.707 hf 0.048 handle-lh [ 0.133  0.069 -0.448] |0.472| lhz 4.127 cp [0. 1.]
186 z 3.590 hf -0.040 handle-lh [ 0.127 -0.004 -0.501] |0.517| lhz 4.062 cp [0. 1.]
192 z 3.632 hf 0.017 handle-lh [ 0.022 -0.063 -0.488] |0.492| lhz 4.094 cp [0. 1.]
198 z 3.592 hf -0.000 handle-lh [-0.091 -0.114 -0.471] |0.493| lhz 4.046 cp [0. 1.]
204 z 3.559 hf 0.016 handle-lh [-0.173 -0.13  -0.5 …[+6142 chars]
```
</details>

**Bash**
```
cat > ana3.py <<'EOF'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tcw=np.linalg.inv(Twc)
ref=np.load('/work/inputs/frame0_reference_human.npz'); print({k:ref[k].shape for k in ref})
hv=r['human_vertices_world'].astype(np.float64); kp=r['human_keypoints_world'].astype(np.float64)
rv=ref['vertices_world']
print('ref human centroid', rv.mean(0), 'recon f0 centroid (subsampled)', hv[0].mean(0), rv[::4].mean(0))
c_ref=(Tcw[:3,:3]@rv.mean(0))+Tcw[:3,3]; c_rec=(Tcw[:3,:3]@hv[0].mean(0))+Tcw[:3,3]
print('cam ref', c_ref, 'cam rec', c_rec)
if rv.shape[0]//4+1>=hv.shape[1]:
    d=hv[0]-rv[::4][:hv.shape[1]]; print('per-vertex diff mean', d.mean(0), np.linalg.norm(d,axis=1).mean())
handle=np.array([0,0,0.33]); hw=(T[:,:3,:3]@handle)+T[:,:3,3]
for i in [200,300,350,400,450,550]:
    dd=np.linalg.norm(hv[i]-hw[i],axis=1); j=dd.argmin()
    print(i,'nearest human vert to handle %.3f'%dd[j], 'idx',j, 'rh %.3f lh %.3f'%(np.linalg.norm(kp[i,21:42].mean(0)-hw[i]),np.linalg.norm(kp[i,42:63].mean(0)-hw[i])),
      'all kp min %.3f at %d'%(np.linalg.norm(kp[i]-hw[i],axis=1).min(), np.linalg.norm(kp[i]-hw[i],axis=1).argmin()))
EOF
python ana3.py
```

<details><summary>output</summary>

```
{'vertices_world': (18056, 3)}
ref human centroid [-0.34604895  0.31231642  2.6219149 ] recon f0 centroid (subsampled) [-0.35593617  0.27973988  2.63938789] [-0.3462454   0.31292462  2.6214113 ]
cam ref [-0.46959611  0.24748141  2.55487843] cam rec [-0.47979332  0.21424024  2.57085768]
200 nearest human vert to handle 0.057 idx 3580 rh 0.084 lh 0.533 all kp min 0.068 at 22
300 nearest human vert to handle 0.003 idx 3756 rh 0.022 lh 0.608 all kp min 0.020 at 36
350 nearest human vert to handle 0.009 idx 3617 rh 0.047 lh 0.630 all kp min 0.014 at 32
400 nearest human vert to handle 0.002 idx 3810 rh 0.036 lh 0.649 all kp min 0.012 at 30
450 nearest human vert to handle 0.026 idx 3707 rh 0.059 lh 0.627 all kp min 0.035 at 36
550 nearest human vert to handle 0.027 idx 3963 rh 0.061 lh 0.629 al …[+20 chars]
```
</details>

**Bash**
```
cat > ana4.py <<'EOF'
import numpy as np, json, trimesh
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
v=np.asarray(m.vertices)[::20]
for i in list(range(0,170,20))+list(range(580,715,15)):
    w=(T[i,:3,:3]@v.T).T+T[i,:3,3]; s=w@n+d   # signed dist; n points to -y (up) => positive above floor
    print(i, 'min signed dist to floor %.3f'%s.min(), 'max %.3f'%s.max())
ref=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']; s=ref@n+d; print('ref human feet min dist', s.min())
hv=r['human_vertices_world'].astype(np.float64)
print('recon human min dist per frame (sample)', [round(float((hv[i]@n+d).min()),3) for i in range(0,715,60)])
EOF
python ana4.py
```

<details><summary>output</summary>

```
0 min signed dist to floor 0.022 max 0.675
20 min signed dist to floor 0.029 max 0.681
40 min signed dist to floor 0.028 max 0.681
60 min signed dist to floor 0.015 max 0.668
80 min signed dist to floor 0.007 max 0.660
100 min signed dist to floor -0.002 max 0.651
120 min signed dist to floor 0.011 max 0.664
140 min signed dist to floor 0.015 max 0.668
160 min signed dist to floor 0.017 max 0.669
580 min signed dist to floor -0.016 max 0.638
595 min signed dist to floor -0.014 max 0.635
610 min signed dist to floor 0.004 max 0.652
625 min signed dist to floor 0.002 max 0.651
640 min signed dist to floor 0.004 max 0.653
655 min signed dist to floor 0.009 max 0.658
670 min signed dist to floor 0.012 max 0.661
685 min signed dist to floor 0.011 max 0.660
700 min signed dist to floor 0.012 max …[+171 chars]
```
</details>

**Bash**
```
cat > static_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch'); sys.path.insert(0,'/work/outputs')
import refine2 as R2
from refine2 import *
def refine_fz(idx, Tc, iters=200, lam=0.3, shared=False, fixz=True, lr=(3e-3,2e-3)):
    crops,tg,vd,img=prep(idx,Tc)
    Ps=torch.tensor(np.array([proj_from_K(crop_K(K,c,S),S,S) for c in crops]),dtype=torch.float32).cuda()
    T0=torch.tensor(Tc[idx],dtype=torch.float32).cuda()
    B=len(idx); nb=1 if shared else B
    w=torch.zeros(nb,3,device='cuda',requires_grad=True); t=torch.zeros(nb,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr[0]},{'params':[t],'lr':lr[1]}])
    c0=T0[:,:3,3].clone(); msk=torch.tensor([1.,1.,0. if fixz else 1.],device='cuda')
    for it in range(iters):
        Rd=rotvec_to_mat(w).expand(B,3,3)
        T=torch.cat([torch.cat([Rd@T0[:,:3,:3], (c0+t*msk)[...,None]],2), T0[:,3:]],1)
        sil,rgb,_=render_batch(r,T,Ps,S,with_tex=True)
        ls=(((sil-tg)**2)*vd).sum((1,2))/tg.sum((1,2)).clamp(min=50)
        lp=photo_loss(rgb,sil,img,vd) if lam>0 else torch.zeros_like(ls)
        l=ls+lam*lp
        opt.zero_grad(); l.sum().backward(); opt.step()
    with torch.no_grad():
        inter=((sil>0.5)&(tg>0.5)&(vd>0)).sum((1,2)).float(); uni=(((sil>0.5)|(tg>0.5))&(vd>0)).sum((1,2)).float()
    return T.detach().cpu().numpy().astype(np.float64), (inter/uni.clamp(min=1)).cpu().numpy(), ls.detach().cpu().numpy(), lp.detach().cpu().numpy()
if __name__=='__main__':
    l …[+1042 chars]
```

<details><summary>output</summary>

```
range 0 152 iou 0.832 -> 0.850
 delta rotvec(deg, world) mean [1.28 2.92 0.64] std [1.13 0.77 0.67]
 yaw(about world y) of mean delta 2.92
 dt cam mean [-0.005  0.002 -0.   ] std [0.002 0.002 0.   ]
range 630 714 iou 0.601 -> 0.730
 delta rotvec(deg, world) mean [-12.74   2.13  -3.38] std [0.76 0.37 0.4 ]
 yaw(about world y) of mean delta 2.13
 dt cam mean [-0.001  0.033  0.   ] std [0.002 0.004 0.   ]
range 0 152 iou 0.832 -> 0.872
 delta rotvec(deg, world) mean [-0.14  2.8   1.06] std [0.42 0.57 0.27]
 yaw(about world y) of mean delta 2.80
 dt cam mean [-0.004  0.02  -0.   ] std [0.002 0.003 0.   ]
range 630 714 iou 0.601 -> 0.746
 delta rotvec(deg, world) mean [-10.24   2.15  -3.42] std [0.82 0.19 0.18]
 yaw(about world y) of mean delta 2.15
 dt cam mean [0.    0.047 0.   ] std [0.001 0 …[+420 chars]
```
</details>

**Bash**
```
sed -i 's/^cols=min(3,len(tiles))/import os; cols=min(int(os.environ.get("COLS",3)),len(tiles))/' overlay.py; COLS=4 python overlay.py --frames $(seq -s, 165 10 395) --out c1.jpg --size 240; COLS=4 python overlay.py --frames $(seq -s, 405 10 625) --out c2.jpg --size 240
```

**Read**
```
/work/scratch/c1.jpg
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

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > flip_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch'); sys.path.insert(0,'/work/outputs')
from static_test import *
cxy=np.array([0.005,-0.004,0])
Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy  # p -> Rz180 (p - c) + c
idx=list(range(0,715,12))
Tb=Tc0.copy(); Tf=Tc0@Fz
res={}
for name,Tc in [('base',Tb),('flip',Tf)]:
    _,iou0,ls0,lp0=refine_fz(idx,Tc,iters=1,lam=0.3)
    Tn,iou,ls,lp=refine_fz(idx,Tc,iters=100,lam=0.0)
    # evaluate photo at refined
    Tfull=Tc.copy(); Tfull[idx]=Tn
    _,iou2,ls2,lp2=refine_fz(idx,Tfull,iters=1,lam=0.3)
    res[name]=(iou0,lp0,iou2,lp2)
for k,i in enumerate(idx):
    b=res['base']; f_=res['flip']
    print(i,'raw: iou %.3f/%.3f photo %.3f/%.3f | refined: iou %.3f/%.3f photo %.3f/%.3f'%(b[0][k],f_[0][k],b[1][k],f_[1][k],b[2][k],f_[2][k],b[3][k],f_[3][k]))
EOF
python flip_test.py
```

<details><summary>output</summary>

```
0 raw: iou 0.827/0.784 photo 1.125/1.214 | refined: iou 0.864/0.868 photo 1.189/1.342
12 raw: iou 0.822/0.779 photo 1.135/1.221 | refined: iou 0.856/0.857 photo 1.194/1.323
24 raw: iou 0.828/0.793 photo 1.142/1.228 | refined: iou 0.857/0.859 photo 1.124/1.271
36 raw: iou 0.829/0.793 photo 1.149/1.230 | refined: iou 0.857/0.858 photo 1.138/1.282
48 raw: iou 0.829/0.793 photo 1.154/1.231 | refined: iou 0.859/0.858 photo 1.155/1.268
60 raw: iou 0.826/0.792 photo 1.151/1.210 | refined: iou 0.861/0.866 photo 1.236/1.343
72 raw: iou 0.823/0.781 photo 1.142/1.204 | refined: iou 0.864/0.869 photo 1.260/1.388
84 raw: iou 0.830/0.789 photo 1.131/1.179 | refined: iou 0.866/0.871 photo 1.275/1.370
96 raw: iou 0.832/0.790 photo 1.141/1.181 | refined: iou 0.871/0.873 photo 1.278/1.366
108 raw: iou 0.831 …[+4469 chars]
```
</details>

> *thinking:* The flip test shows the baseline pose fits best for frames 0–340, but a 180° flip about the vertical axis fits better from ~frame 400 onward, suggesting a symmetry flip occurred mid-episode. I'll now verify this visually.

**Bash**
```
cat > sidebyside.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
f=h5py.File('/work/inputs/masks.h5')
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
cxy=np.array([0.005,-0.004,0]); Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy
frames=[int(x) for x in sys.argv[1].split(',')]; out=sys.argv[2]; sz=int(sys.argv[3]) if len(sys.argv)>3 else 400
rows=[]
for i in frames:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg'); om=f['object'][i]
    ys,xs=np.nonzero(om); cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2; s=int(max(xs.ptp(),ys.ptp())*1.15)+10
    x0,y0=cx-s//2,cy-s//2
    def cr(a): 
        p=cv2.copyMakeBorder(a,s,s,s,s,cv2.BORDER_CONSTANT); return cv2.resize(p[y0+s:y0+2*s,x0+s:x0+2*s],(sz,sz))
    tiles=[cr(img)]
    for T in [Tc0[i],Tc0[i]@Fz]:
        mk,rgb,_,_=r.render(T[None]); rgb=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8).copy()
        tiles.append(cr(rgb))
    cv2.putText(tiles[0],str(i),(5,25),0,0.8,(0,0,255),2); cv2.putText(tiles[1],'base',(5,25),0,0.8,(0,0,255),2); cv2.putText(tiles[2],'flip',(5,25),0,0.8,(0,0,255),2)
    rows.append(np.hstack(tiles))
cv2.imwrite(out,np.vstack(rows))
EOF
python sidebyside.py 650,420,40 sb1.jpg 330
```

**Read**
```
/work/scratch/sb1.jpg
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
cat > texview2.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
from scipy.spatial.transform import Rotation as R
m=load_mesh('/work/inputs/object_mesh.glb')
K=np.array([[2400,0,300],[0,2400,400],[0,0,1.]])
r=Renderer(m,K,H=800,W=600)
ims=[]
for yaw in range(0,360,45):
    Rb=np.array([[1,0,0],[0,0,-1],[0,1,0.]])  # mesh z -> cam -y (up), mesh y -> cam z
    Ry=R.from_euler('y',yaw,degrees=True).as_matrix()
    T=np.eye(4); T[:3,:3]=Ry@Rb; T[:3,3]=[0,0,2.2]
    mk,rgb,d,_=r.render(T[None])
    im=(rgb[0].cpu().numpy()*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.putText(im,f'yaw{yaw}',(10,30),0,1,(0,0,255),2); ims.append(im)
cv2.imwrite('views2.jpg', np.vstack([np.hstack(ims[:4]),np.hstack(ims[4:])]))
EOF
python texview2.py
```

**Read**
```
/work/scratch/views2.jpg
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
cat > ana5.py <<'EOF'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
kp=r['human_keypoints_world'].astype(np.float64)
def yaw(Rm): # angle of mesh x-axis in world x-z plane
    a=Rm[...,:3,0]; return np.degrees(np.arctan2(a[...,2],a[...,0]))
def yawc(Rm): a=Rm[...,:3,0]; return np.degrees(np.arctan2(a[...,2],a[...,0]))
yb=yaw(T); yf=yawc(fp); yc=yawc(cc); ybc=yawc(Tc)
# body yaw from keypoints: try indices for hips (MHR70: 9,10 = left/right hip?) - print a few kp to guess
print(kp[0,:15].round(2))
for i in range(0,715,8): print(i,'world yaw base %.0f | cam yaw base %.0f fp %.0f cc %.0f'%(yb[i],ybc[i],yf[i],yc[i]))
EOF
python ana5.py
```

<details><summary>output</summary>

```
[[-0.37 -0.3   2.71]
 [-0.4  -0.34  2.68]
 [-0.34 -0.34  2.68]
 [-0.45 -0.32  2.59]
 [-0.3  -0.32  2.58]
 [-0.55 -0.13  2.57]
 [-0.21 -0.13  2.51]
 [-0.58  0.17  2.64]
 [-0.14  0.17  2.54]
 [-0.44  0.36  2.62]
 [-0.26  0.37  2.57]
 [-0.45  0.77  2.69]
 [-0.27  0.78  2.6 ]
 [-0.51  1.11  2.51]
 [-0.29  1.16  2.56]]
0 world yaw base -34 | cam yaw base -33 fp -32 cc -32
8 world yaw base -34 | cam yaw base -33 fp -32 cc -32
16 world yaw base -34 | cam yaw base -33 fp -33 cc -32
24 world yaw base -34 | cam yaw base -33 fp -32 cc -32
32 world yaw base -34 | cam yaw base -33 fp -31 cc -32
40 world yaw base -34 | cam yaw base -33 fp -32 cc -32
48 world yaw base -34 | cam yaw base -33 fp -32 cc -32
56 world yaw base -34 | cam yaw base -33 fp -31 cc -32
64 world yaw base -34 | cam yaw base -33 fp -3 …[+4643 chars]
```
</details>

**Bash**
```
cat > yawscan.py <<'EOF'
import sys; sys.path.insert(0,'/work/scratch'); sys.path.insert(0,'/work/outputs')
from static_test import *
import time
cxy=np.array([0.005,-0.004,0])
def Rz(phi):
    T=np.eye(4); c,s=np.cos(phi),np.sin(phi); T[:2,:2]=[[c,-s],[s,c]]; T[:3,3]=cxy-T[:3,:3]@cxy; return T
yaws=np.radians(np.arange(0,360,10))
idx=list(range(0,715,4))
t0=time.time()
IOU=np.zeros((len(idx),len(yaws))); PH=np.zeros_like(IOU); LS=np.zeros_like(IOU)
for k0 in range(0,len(idx),30):
    sub=idx[k0:k0+30]
    for j,yw in enumerate(yaws):
        Tc=Tc0@Rz(yw)
        _,iou,ls,lp=refine_fz(sub,Tc,iters=1,lam=0.3)
        IOU[k0:k0+len(sub),j]=iou; PH[k0:k0+len(sub),j]=lp; LS[k0:k0+len(sub),j]=ls
print(time.time()-t0)
np.savez('yawscan.npz',IOU=IOU,PH=PH,LS=LS,idx=idx,yaws=yaws)
EOF
python yawscan.py
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: buvfqa0na). Output is being written to: /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/buvfqa0na.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
```
</details>

**Bash**
```
sleep 100; cat /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/buvfqa0na.output
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/buvfqa0na.output
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
cat > /work/outputs/evidence.py <<'EOF'
"""Per-frame image evidence: crops, masks, and a batched render-and-compare scorer."""
import numpy as np, torch, cv2, h5py, torch.nn.functional as F
from common import *

def gblur(x, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-(torch.arange(k, device=x.device, dtype=x.dtype) - k // 2) ** 2 / (2 * s * s)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, k).repeat(C, 1, 1, 1), padding=(0, k // 2), groups=C)
    return F.conv2d(x, g.view(1, 1, k, 1).repeat(C, 1, 1, 1), padding=(k // 2, 0), groups=C)

class Evidence:
    """Holds for every frame a square crop around the object (SAM mask bbox, or the baseline render if the mask is
    empty), resampled to S x S: image, object mask, valid mask (= not covered by the person unless object)."""
    def __init__(self, inputs, frames, renderer, K, Tc_ref, S=320, margin=1.3):
        self.S = S; self.K = K; self.r = renderer
        f = h5py.File(f"{inputs}/masks.h5", "r")
        F_ = len(frames)
        self.crops = np.zeros((F_, 4)); self.img = np.zeros((F_, S, S, 3), np.float32)
        self.obj = np.zeros((F_, S, S), np.float32); self.valid = np.zeros((F_, S, S), np.float32)
        self.area = np.zeros(F_)
        for i, fr in enumerate(frames):
            om = f["object"][i]; pm = f["person"][i]
            mk = renderer.render(Tc_ref[i][None], want_tex=False)[0][0].cpu().numpy()
            self.area[i] = om.sum()
            box = om if om.sum() > 200 else (o …[+3458 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
cat > build_ev.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
t0=time.time()
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
frames=read_video('/work/inputs/video.mp4')
ev=Evidence('/work/inputs',frames,r,K,Tc0,S=320)
ev.r=None
pickle.dump(ev,open('/work/scratch/ev320.pkl','wb'),protocol=4)
print(time.time()-t0)
EOF
sleep 30; cat /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/buvfqa0na.output
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
sleep 120; cat /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/buvfqa0na.output; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```

<details><summary>output</summary>

```
Exit code 143
Command timed out after 2m 0s
```
</details>

**Bash**
```
cat /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/buvfqa0na.output; ls -la /work/scratch/yawscan.npz 2>&1; ps aux | grep yawscan | head -3
```

<details><summary>output</summary>

```
ls: cannot access '/work/scratch/yawscan.npz': No such file or directory
hanzhou     4987  0.0  0.0   4500  3368 ?        SNs  15:10   0:00 /bin/bash -c source /work/scratch/.claude/shell-snapshots/snapshot-bash-1792580330649-0kcqzo.sh 2>/dev/null || true && shopt -u extglob 2>/dev/null || true && { \builtin unalias -- 'unsetenv'; \builtin unset -f -- 'unsetenv'; } >/dev/null 2>&1 || true && eval 'cat > yawscan.py <<'"'"'EOF'"'"' import sys; sys.path.insert(0,'"'"'/work/scratch'"'"'); sys.path.insert(0,'"'"'/work/outputs'"'"') from static_test import * import time cxy=np.array([0.005,-0.004,0]) def Rz(phi):     T=np.eye(4); c,s=np.cos(phi),np.sin(phi); T[:2,:2]=[[c,-s],[s,c]]; T[:3,3]=cxy-T[:3,:3]@cxy; return T yaws=np.radians(np.arange(0,360,10)) idx=list(range(0,715,4)) t0=time.time() IO …[+1093 chars]
```
</details>

**Bash**
```
pkill -f yawscan.py; cd /work/scratch && timeout 600 python build_ev.py
```

<details><summary>output</summary>

```
Exit code 144
```
</details>

**Agent:** Quick status: I killed a slow brute-force scan (it was re-reading masks for every hypothesis) and am now caching per-frame crops once, so hypothesis scoring becomes cheap.

**Bash**
```
python build_ev.py
```

<details><summary>output</summary>

```
Command running in background with ID: bhb302yok. Output is being written to: /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/bhb302yok.output. You will be notified when it completes. To check interim output, use Read on that file path.
```
</details>

**Bash**
```
cat > yawscan2.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
cxy=np.array([0.005,-0.004,0])
def Rz(phi):
    T=np.eye(4); c,s=np.cos(phi),np.sin(phi); T[:2,:2]=[[c,-s],[s,c]]; T[:3,3]=cxy-T[:3,:3]@cxy; return T
step=int(sys.argv[1]) if len(sys.argv)>1 else 4
iters=int(sys.argv[2]) if len(sys.argv)>2 else 0
yaws=np.radians(np.arange(0,360,10)); idx=np.arange(0,715,step)
t0=time.time()
pairs=[(i,j) for i in idx for j in range(len(yaws))]
IOU=np.zeros((len(idx),len(yaws))); PH=np.zeros_like(IOU); LS=np.zeros_like(IOU); TT=np.zeros((len(idx),len(yaws),4,4))
B=72
for b in range(0,len(pairs),B):
    pb=pairs[b:b+B]; ids=np.array([p[0] for p in pb])
    Tc=np.array([Tc0[i]@Rz(yaws[j]) for i,j in pb])
    T,iou,ls,lp=score_and_refine(r,ev,ids,Tc,iters=iters,lam=0.3)
    for k,(i,j) in enumerate(pb):
        ii=i//step; IOU[ii,j]=iou[k]; PH[ii,j]=lp[k]; LS[ii,j]=ls[k]; TT[ii,j]=T[k]
print('time',time.time()-t0)
np.savez(f'yawscan_s{step}_it{iters}.npz',IOU=IOU,PH=PH,LS=LS,idx=idx,yaws=yaws,T=TT)
EOF
sleep 45; cat /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/bhb302yok.output
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
sleep 90; cat /tmp/claude-1003/-work/cb0a3ca9-1f9c-4acd-a122-dcc11c47d72e/tasks/bhb302yok.output
```

<details><summary>output</summary>

```
96.39414191246033

[exited with code 0]
```
</details>

**Bash**
```
python yawscan2.py 4 0 && python -c "
import numpy as np
d=np.load('yawscan_s4_it0.npz'); IOU=d['IOU']; PH=d['PH']; idx=d['idx']
for k,i in enumerate(idx[::2]):
    k*=2
    o=np.argsort(-IOU[k])[:4]; p=np.argsort(PH[k])[:4]
    print(i,'iou best yaws',(o*10).tolist(),np.round(IOU[k,o],2).tolist(),'| base %.2f flip %.2f'%(IOU[k,0],IOU[k,18]),' photo best',(p*10).tolist(),np.round(PH[k,p],2).tolist(),'base %.2f flip %.2f'%(PH[k,0],PH[k,18]))
"
```

<details><summary>output</summary>

```
time 9.05354642868042
0 iou best yaws [0, 280, 100, 350] [0.83, 0.82, 0.82, 0.82] | base 0.83 flip 0.78  photo best [0, 350, 180, 170] [1.1, 1.11, 1.2, 1.2] base 1.10 flip 1.20
8 iou best yaws [0, 280, 100, 350] [0.82, 0.82, 0.81, 0.81] | base 0.82 flip 0.78  photo best [0, 350, 180, 170] [1.1, 1.12, 1.19, 1.2] base 1.10 flip 1.19
16 iou best yaws [0, 280, 350, 170] [0.83, 0.81, 0.81, 0.81] | base 0.83 flip 0.79  photo best [0, 350, 180, 170] [1.1, 1.12, 1.2, 1.21] base 1.10 flip 1.20
24 iou best yaws [0, 280, 350, 170] [0.83, 0.81, 0.81, 0.8] | base 0.83 flip 0.80  photo best [0, 350, 180, 170] [1.12, 1.14, 1.21, 1.23] base 1.12 flip 1.21
32 iou best yaws [0, 280, 350, 170] [0.83, 0.81, 0.81, 0.8] | base 0.83 flip 0.79  photo best [0, 350, 180, 170] [1.12, 1.15, 1.21, 1.23] base 1.12 flip …[+13535 chars]
```
</details>

**Bash**
```
python sidebyside.py 650,600,560 sb2.jpg 450
```

**Read**
```
/work/scratch/sb2.jpg
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
python sidebyside.py 0,140,336 sb3.jpg 450
```

**Read**
```
/work/scratch/sb3.jpg
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
cat > hinge.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
cxy=np.array([0.005,-0.004,0]); Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy
rows=[]
for i in [int(x) for x in sys.argv[1].split(',')]:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    mk,_,_,_=r.render(Tc0[i][None]); ys,xs=np.nonzero(mk[0].cpu().numpy())
    x0,y0=xs.min()-15,ys.min()-15; w=xs.max()-xs.min()+30; h=int((ys.max()-ys.min())*0.45)
    tiles=[cv2.resize(img[y0:y0+h,x0:x0+w],(w*4,h*4),interpolation=cv2.INTER_CUBIC)]
    for T in [Tc0[i],Tc0[i]@Fz]:
        # supersampled render
        _,rgb,_,_=r.render(T[None],crop=(x0,y0,w,h),H=h*4,W=w*4); tiles.append((rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8))
    rows.append(np.hstack(tiles))
    cv2.imwrite(f'hinge_{i}.jpg',rows[-1])
EOF
python hinge.py 650,390,0,336
```

**Read**
```
/work/scratch/hinge_650.jpg
```

**Read**
```
/work/scratch/hinge_390.jpg
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
cat > static_flip.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
cxy=np.array([0.005,-0.004,0])
def Rz(phi):
    T=np.eye(4); c,s=np.cos(phi),np.sin(phi); T[:2,:2]=[[c,-s],[s,c]]; T[:3,3]=cxy-T[:3,:3]@cxy; return T
for name,ids in [('start',np.arange(0,100,4)),('start_late',np.arange(110,160,4)),('end',np.arange(630,715,4))]:
    ref=Tc0[ids[0]]  # same pose for all (static): use the first frame's baseline pose
    for yaw in [0,180]:
        Tc=np.array([ref@Rz(np.radians(yaw))]*len(ids))
        for lam in [0.3]:
            T,iou,ls,lp=score_and_refine(r,ev,ids,Tc,iters=150,lam=lam,shared=True)
            _,iou2,ls2,lp2=score_and_refine(r,ev,ids,T,iters=0,lam=0.3)
            dR=np.degrees(np.linalg.norm(R.from_matrix(T[0,:3,:3]@(ref@Rz(np.radians(yaw)))[:3,:3].T).as_rotvec()))
            print(name,'yaw',yaw,'lam',lam,'iou %.3f sil %.4f photo %.3f'%(iou2.mean(),ls2.mean(),lp2.mean()),'dR from init %.1f'%dR, 'dt',np.round(T[0,:3,3]-ref[:3,3],3))
EOF
python static_flip.py
```

<details><summary>output</summary>

```
start yaw 0 lam 0.3 iou 0.844 sil 0.1515 photo 0.935 dR from init 3.0 dt [-0.006  0.005  0.   ]
start yaw 180 lam 0.3 iou 0.839 sil 0.1553 photo 1.069 dR from init 1.9 dt [-0.013  0.001 -0.012]
start_late yaw 0 lam 0.3 iou 0.868 sil 0.1255 photo 0.974 dR from init 3.8 dt [-0.009 -0.     0.   ]
start_late yaw 180 lam 0.3 iou 0.867 sil 0.1250 photo 0.971 dR from init 1.8 dt [-0.014 -0.    -0.012]
end yaw 0 lam 0.3 iou 0.732 sil 0.2643 photo 1.375 dR from init 13.2 dt [-0.     0.031  0.   ]
end yaw 180 lam 0.3 iou 0.737 sil 0.2665 photo 1.360 dR from init 3.7 dt [-0.003  0.022  0.006]
```
</details>

**Bash**
```
cat > ana6.py <<'EOF'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
kp=r['human_keypoints_world'].astype(np.float64)
def ang(v): return np.degrees(np.arctan2(v[...,2],v[...,0]))
ys=ang(T[:,:3,0])   # mesh x axis yaw
lat=(kp[:,5]-kp[:,6])+(kp[:,9]-kp[:,10]); yb=ang(lat)
rel=(ys-yb+180)%360-180
hand=kp[:,21:42].mean(1); 
for i in range(150,640,6):
    print(i,'sign %.0f body %.0f rel %.0f'%(ys[i],yb[i],rel[i]), 'hand-pelvis lateral offset %.2f'%np.dot(hand[i]-(kp[i,9]+kp[i,10])/2, lat[i]/np.linalg.norm(lat[i])))
EOF
python ana6.py
```

<details><summary>output</summary>

```
150 sign -34 body 88 rel -123 hand-pelvis lateral offset -0.34
156 sign -34 body 89 rel -123 hand-pelvis lateral offset -0.35
162 sign -34 body 90 rel -124 hand-pelvis lateral offset -0.35
168 sign -37 body 89 rel -126 hand-pelvis lateral offset -0.34
174 sign -42 body 87 rel -129 hand-pelvis lateral offset -0.31
180 sign -50 body 81 rel -131 hand-pelvis lateral offset -0.26
186 sign -58 body 76 rel -135 hand-pelvis lateral offset -0.27
192 sign -69 body 64 rel -133 hand-pelvis lateral offset -0.26
198 sign -79 body 58 rel -137 hand-pelvis lateral offset -0.26
204 sign -88 body 53 rel -141 hand-pelvis lateral offset -0.30
210 sign -101 body 46 rel -147 hand-pelvis lateral offset -0.31
216 sign -112 body 39 rel -152 hand-pelvis lateral offset -0.28
222 sign -132 body 35 rel -166 hand-pelvis …[+4384 chars]
```
</details>

> *thinking:* The sign's yaw relative to the person's body stays steady at ~170–180° across the whole carry, with no 180° jump even as the person turns a full loop — so the photometric evidence suggesting a flip after frame ~408 is likely unreliable. The only remaining window where a flip could be hidden is the pick-up (165–222), which I'm checking now.

**Bash**
```
COLS=5 python overlay.py --frames $(seq -s, 160 4 236) --out c3.jpg --size 200
```

**Read**
```
/work/scratch/c3.jpg
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
cat > ana7.py <<'EOF'
import numpy as np, h5py
f=h5py.File('/work/inputs/masks.h5')
O=f['object']; P=f['person']
prev=None; ref0=O[0]; refE=O[700]
for i in list(range(140,180,2))+list(range(590,640,2)):
    o=O[i]; p=P[i]
    vis=~p
    def iou(a,b): 
        a=a&vis; b=b&vis; return (a&b).sum()/max((a|b).sum(),1)
    ys,xs=np.nonzero(o)
    print(i,'area',o.sum(),'iou w/ f0 %.3f w/ f700 %.3f'%(iou(o,ref0),iou(o,refE)),'bottom y %d cx %.1f'%(ys.max(),xs.mean()))
EOF
python ana7.py
```

<details><summary>output</summary>

```
140 area 8442 iou w/ f0 0.978 w/ f700 0.000 bottom y 821 cx 739.3
142 area 8340 iou w/ f0 0.975 w/ f700 0.000 bottom y 821 cx 738.8
144 area 8336 iou w/ f0 0.975 w/ f700 0.000 bottom y 821 cx 738.8
146 area 8318 iou w/ f0 0.969 w/ f700 0.000 bottom y 821 cx 738.8
148 area 8187 iou w/ f0 0.955 w/ f700 0.000 bottom y 821 cx 739.4
150 area 8191 iou w/ f0 0.963 w/ f700 0.000 bottom y 821 cx 739.4
152 area 8189 iou w/ f0 0.956 w/ f700 0.000 bottom y 821 cx 739.4
154 area 8367 iou w/ f0 0.975 w/ f700 0.000 bottom y 821 cx 739.3
156 area 8379 iou w/ f0 0.975 w/ f700 0.000 bottom y 821 cx 739.3
158 area 8267 iou w/ f0 0.967 w/ f700 0.000 bottom y 821 cx 738.8
160 area 8299 iou w/ f0 0.975 w/ f700 0.000 bottom y 821 cx 738.8
162 area 8297 iou w/ f0 0.975 w/ f700 0.000 bottom y 821 cx 738.8
164 area …[+2169 chars]
```
</details>

**Bash**
```
cat > ana8.py <<'EOF'
import numpy as np, h5py
f=h5py.File('/work/inputs/masks.h5')
O=f['object']; P=f['person']; refE=O[714]; ref0=O[0]
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
for i in list(range(560,600,1)):
    o=O[i]; vis=~P[i]
    a=o&vis; b=refE&vis; print(i,'iou end %.3f'%((a&b).sum()/max((a|b).sum(),1)), 'base pos',np.round(T[i,:3,3],3))
EOF
python ana8.py
```

<details><summary>output</summary>

```
560 iou end 0.076 base pos [-0.646  0.791  3.081]
561 iou end 0.075 base pos [-0.65   0.799  3.073]
562 iou end 0.072 base pos [-0.656  0.806  3.083]
563 iou end 0.069 base pos [-0.665  0.811  3.078]
564 iou end 0.066 base pos [-0.67   0.822  3.076]
565 iou end 0.065 base pos [-0.667  0.829  3.071]
566 iou end 0.058 base pos [-0.68   0.825  3.07 ]
567 iou end 0.055 base pos [-0.68   0.835  3.06 ]
568 iou end 0.055 base pos [-0.684  0.837  3.059]
569 iou end 0.061 base pos [-0.696  0.841  3.046]
570 iou end 0.076 base pos [-0.701  0.847  3.066]
571 iou end 0.105 base pos [-0.698  0.864  3.048]
572 iou end 0.141 base pos [-0.696  0.863  3.037]
573 iou end 0.179 base pos [-0.702  0.871  3.033]
574 iou end 0.217 base pos [-0.709  0.878  3.031]
575 iou end 0.263 base pos [-0.711  0.882  3.029]
 …[+1199 chars]
```
</details>

**Bash**
```
cat > support.py <<'EOF'
import numpy as np, trimesh, json
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry()
v=np.asarray(m.vertices)
hull=m.convex_hull
# find the hull facet whose normal is closest to -z and is a support plane
best=None
for n,o in zip(hull.facets_normal, hull.facets_origin):
    if n[2]<-0.9:
        d=(v-o)@n
        print('facet normal',np.round(n,4),'area', 'max dist', d.max().round(4))
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; nf=-np.array(pl[:3])  # up normal (world -y)
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
for i in [0,100,160,600,650,714]:
    up=T[i,:3,:3]@np.array([0,0,1.]); print(i,'angle mesh z vs floor-up %.2f deg'%np.degrees(np.arccos(np.clip(up@-np.array(pl[:3]),-1,1))), 'vs world -y %.2f'%np.degrees(np.arccos(-up[1])))
EOF
python support.py
```

<details><summary>output</summary>

```
facet normal [ 0.2133 -0.0155 -0.9769] area max dist 0.0
facet normal [ 0.004  -0.018  -0.9998] area max dist 0.0
facet normal [ 0.0033  0.     -1.    ] area max dist 0.0
facet normal [ 0.0053  0.1291 -0.9916] area max dist 0.0
0 angle mesh z vs floor-up 177.69 deg vs world -y 1.33
100 angle mesh z vs floor-up 177.69 deg vs world -y 1.33
160 angle mesh z vs floor-up 177.69 deg vs world -y 1.33
600 angle mesh z vs floor-up 179.10 deg vs world -y 2.57
650 angle mesh z vs floor-up 179.03 deg vs world -y 2.30
714 angle mesh z vs floor-up 179.03 deg vs world -y 2.30
```
</details>

**Bash**
```
cat > ana9.py <<'EOF'
import numpy as np, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
Rw=R.from_matrix(T[:,:3,:3])
w=(Rw[:-1].inv()*Rw[1:]).as_rotvec()  # body-frame increments
wd=np.degrees(np.linalg.norm(w,axis=1))
acc=np.degrees(np.linalg.norm(w[1:]-w[:-1],axis=1))
for a in range(165,600,15):
    print(a, 'mean |w| %.2f deg/f, mean |dw| %.2f'%(wd[a:a+15].mean(), acc[a:a+15].mean()), ' max |w| %.1f'%wd[a:a+15].max())
EOF
python ana9.py
```

<details><summary>output</summary>

```
165 mean |w| 1.36 deg/f, mean |dw| 1.36  max |w| 2.7
180 mean |w| 1.88 deg/f, mean |dw| 1.46  max |w| 3.1
195 mean |w| 2.04 deg/f, mean |dw| 1.79  max |w| 4.6
210 mean |w| 2.45 deg/f, mean |dw| 1.33  max |w| 4.3
225 mean |w| 1.00 deg/f, mean |dw| 1.31  max |w| 2.3
240 mean |w| 1.52 deg/f, mean |dw| 1.62  max |w| 2.3
255 mean |w| 1.46 deg/f, mean |dw| 1.26  max |w| 2.4
270 mean |w| 1.49 deg/f, mean |dw| 1.38  max |w| 2.7
285 mean |w| 2.11 deg/f, mean |dw| 1.45  max |w| 3.9
300 mean |w| 1.86 deg/f, mean |dw| 1.34  max |w| 2.9
315 mean |w| 2.04 deg/f, mean |dw| 1.87  max |w| 4.1
330 mean |w| 2.49 deg/f, mean |dw| 1.84  max |w| 4.1
345 mean |w| 2.61 deg/f, mean |dw| 1.89  max |w| 4.5
360 mean |w| 3.34 deg/f, mean |dw| 2.10  max |w| 4.5
375 mean |w| 2.27 deg/f, mean |dw| 3.06  max |w| 5.2
390 m …[+740 chars]
```
</details>

**Bash**
```
cat > jit_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, time
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
a,b=int(sys.argv[1]),int(sys.argv[2]); lam=float(sys.argv[3]); it=int(sys.argv[4])
ids=np.arange(a,b)
t0=time.time(); out=[]
for k in range(0,len(ids),40):
    out.append(score_and_refine(r,ev,ids[k:k+40],Tc0[ids[k:k+40]],iters=it,lam=lam))
print('t',time.time()-t0)
T=np.concatenate([o[0] for o in out]); iou=np.concatenate([o[1] for o in out]); lp=np.concatenate([o[3] for o in out])
_,iou0,_,lp0=score_and_refine(r,ev,ids[:40],Tc0[ids[:40]],iters=0,lam=lam)
def jit(Tc):
    Rw=R.from_matrix(Tc[:,:3,:3]); w=(Rw[:-1].inv()*Rw[1:]).as_rotvec(); return np.degrees(np.linalg.norm(w[1:]-w[:-1],axis=1)).mean(), np.degrees(np.linalg.norm(w,axis=1)).mean()
print('baseline jitter %.2f vel %.2f'%jit(Tc0[ids]), ' refined jitter %.2f vel %.2f'%jit(T))
print('iou refined mean %.3f'%iou.mean())
d=np.array([np.degrees(np.linalg.norm(R.from_matrix(T[k,:3,:3]@Tc0[i,:3,:3].T).as_rotvec())) for k,i in enumerate(ids)])
print('mean rot change %.2f'%d.mean())
np.save(f'jit_{a}_{b}_{lam}_{it}.npy',T)
EOF
python jit_test.py 225 345 0.3 100
```

<details><summary>output</summary>

```
t 11.84545636177063
baseline jitter 1.50 vel 1.74  refined jitter 1.42 vel 1.68
iou refined mean 0.877
mean rot change 3.41
```
</details>

**Bash**
```
cat > ana10.py <<'EOF'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); kp=r['human_keypoints_world'].astype(np.float64)
i=300
print('elbow7',kp[i,7].round(3),'elbow8',kp[i,8].round(3))
for j in range(21,42): print(j, kp[i,j].round(3), 'd to elbow8 %.3f'%np.linalg.norm(kp[i,j]-kp[i,8]), 'd to elbow7 %.3f'%np.linalg.norm(kp[i,j]-kp[i,7]))
for j in range(62,70): print(j, kp[i,j].round(3), 'd to elbow8 %.3f'%np.linalg.norm(kp[i,j]-kp[i,8]))
EOF
python ana10.py
```

<details><summary>output</summary>

```
elbow7 [0.273 0.093 3.132] elbow8 [-0.121  0.12   2.912]
21 [-0.141  0.477  2.826] d to elbow8 0.367 d to elbow7 0.642
22 [-0.14   0.454  2.827] d to elbow8 0.345 d to elbow7 0.627
23 [-0.136  0.422  2.831] d to elbow8 0.313 d to elbow7 0.605
24 [-0.141  0.396  2.842] d to elbow8 0.285 d to elbow7 0.589
25 [-0.171  0.508  2.835] d to elbow8 0.398 d to elbow7 0.676
26 [-0.177  0.492  2.826] d to elbow8 0.386 d to elbow7 0.675
27 [-0.182  0.471  2.818] d to elbow8 0.369 d to elbow7 0.670
28 [-0.174  0.437  2.815] d to elbow8 0.335 d to elbow7 0.647
29 [-0.169  0.503  2.862] d to elbow8 0.389 d to elbow7 0.660
30 [-0.181  0.491  2.852] d to elbow8 0.380 d to elbow7 0.665
31 [-0.193  0.474  2.839] d to elbow8 0.368 d to elbow7 0.669
32 [-0.193  0.436  2.829] d to elbow8 0.334 d to elbow7 0.653 …[+901 chars]
```
</details>

**Bash**
```
cat > /work/outputs/handframe.py <<'EOF'
import numpy as np
def hand_frames(kp):
    """Right-hand frame (MHR70: wrist 41, index base 28, middle base 32, pinky base 40). Returns (F,4,4) world."""
    o = kp[:, 41]; a = kp[:, 32] - o; a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = kp[:, 28] - kp[:, 40]; b -= (b * a).sum(1, keepdims=True) * a; b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = np.cross(a, b)
    T = np.tile(np.eye(4), (len(kp), 1, 1)); T[:, :3, 0] = a; T[:, :3, 1] = b; T[:, :3, 2] = c; T[:, :3, 3] = o
    return T
EOF
cat > ana11.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from handframe import hand_frames
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); kp=r['human_keypoints_world'].astype(np.float64); T=r['T_world_mesh'].astype(np.float64)
H=hand_frames(kp); Rel=np.linalg.inv(H)@T
ref=Rel[300]
for i in range(170,625,8):
    d=np.degrees(np.linalg.norm(R.from_matrix(ref[:3,:3].T@Rel[i,:3,:3]).as_rotvec()))
    print(i,'rel rot vs f300 %.1f'%d,'rel t',np.round(Rel[i,:3,3],3), ' rotvec',np.round(np.degrees(R.from_matrix(ref[:3,:3].T@Rel[i,:3,:3]).as_rotvec()),0))
EOF
python ana11.py
```

<details><summary>output</summary>

```
170 rel rot vs f300 35.5 rel t [ 0.374 -0.14  -0.259]  rotvec [-9. -3. 34.]
178 rel rot vs f300 28.0 rel t [ 0.359 -0.154 -0.261]  rotvec [-13.  -2.  25.]
186 rel rot vs f300 30.5 rel t [ 0.289 -0.18  -0.25 ]  rotvec [-19.   4.  23.]
194 rel rot vs f300 29.1 rel t [ 0.239 -0.185 -0.301]  rotvec [-21.   9.  18.]
202 rel rot vs f300 36.5 rel t [ 0.187 -0.212 -0.288]  rotvec [-24.  15.  23.]
210 rel rot vs f300 43.0 rel t [ 0.102 -0.263 -0.266]  rotvec [-33.  17.  22.]
218 rel rot vs f300 38.5 rel t [ 0.129 -0.289 -0.215]  rotvec [-35.  10.  13.]
226 rel rot vs f300 39.8 rel t [ 0.132 -0.309 -0.187]  rotvec [-38.   6.  11.]
234 rel rot vs f300 34.6 rel t [ 0.166 -0.291 -0.21 ]  rotvec [-32.   5.  13.]
242 rel rot vs f300 29.8 rel t [ 0.197 -0.28  -0.194]  rotvec [-28.   2.  10.]
250 rel rot v …[+3580 chars]
```
</details>

> *thinking:* Keeping the baseline orientation since continuity through the pick-up and the body-locked yaw rule out a flip, though texture evidence for a whole-episode flip remains inconclusive. Now building the trajectory solver with constant poses for the static phases and robust SE(3) smoothing for the carry, interpolating low-weight occluded frames.

**Bash**
```
cat > /work/outputs/trajectory.py <<'EOF'
"""Phase segmentation and robust temporal smoothing of object poses."""
import numpy as np, torch
from scipy.spatial.transform import Rotation as Rot

def chordal_mean(Rs, w=None):
    w = np.ones(len(Rs)) if w is None else np.asarray(w)
    M = (Rs * w[:, None, None]).sum(0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

def static_pose_cam(Tc):
    """One camera-frame pose for a static phase: chordal mean rotation; translation = median depth along the
    median viewing ray (keeps the baseline's median image position and median depth)."""
    Rm = chordal_mean(Tc[:, :3, :3])
    t = Tc[:, :3, 3]; z = np.median(t[:, 2]); u = np.median(t[:, 0] / t[:, 2]); v = np.median(t[:, 1] / t[:, 2])
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = [u * z, v * z, z]
    return T

def mask_iou_vis(a, b, vis):
    a = a & vis; b = b & vis
    return (a & b).sum() / max((a | b).sum(), 1)

def detect_static_phases(obj_masks, person_masks, thr=0.85, ref_n=10):
    """Start phase: longest prefix whose object mask (outside the person) overlaps the mask of the first frames with
    IoU >= thr; end phase: same with the last frames. Returns (start_end_exclusive, end_start)."""
    F = len(obj_masks)
    ref0 = np.median(np.stack([obj_masks[i] for i in range(ref_n)]), 0) > 0.5
    refE = np.median(np.stack([obj_masks[F - 1 - i] for i in range(ref_n)]), 0) > 0.5
    s_end = 0
    for i in range(F):
 …[+2983 chars]
```

<details><summary>output</summary>

```
ok
```
</details>

**Bash**
```
cat > vis_ratio.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
f=h5py.File('/work/inputs/masks.h5')
out=[]
for i in range(715):
    mk=r.render(Tc0[i][None],want_tex=False)[0][0].cpu().numpy()
    om=f['object'][i]; pm=f['person'][i]
    out.append((om.sum(), mk.sum(), (mk&pm&~om).sum()))
out=np.array(out,float); np.save('vis.npy',out)
ratio=out[:,0]/np.maximum(out[:,1],1); occ=out[:,2]/np.maximum(out[:,1],1)
for i in range(160,640,5): print(i,'mask/render %.2f  render-occluded-by-person %.2f'%(ratio[i],occ[i]))
EOF
python vis_ratio.py | awk '{printf "%s | ", $0; if (NR%4==0) print ""}'
```

<details><summary>output</summary>

```
160 mask/render 1.03  render-occluded-by-person 0.03 | 165 mask/render 1.02  render-occluded-by-person 0.03 | 170 mask/render 1.05  render-occluded-by-person 0.03 | 175 mask/render 1.04  render-occluded-by-person 0.02 | 
180 mask/render 1.05  render-occluded-by-person 0.02 | 185 mask/render 1.05  render-occluded-by-person 0.01 | 190 mask/render 1.07  render-occluded-by-person 0.02 | 195 mask/render 1.10  render-occluded-by-person 0.02 | 
200 mask/render 1.10  render-occluded-by-person 0.02 | 205 mask/render 1.10  render-occluded-by-person 0.02 | 210 mask/render 1.07  render-occluded-by-person 0.03 | 215 mask/render 1.06  render-occluded-by-person 0.05 | 
220 mask/render 1.09  render-occluded-by-person 0.08 | 225 mask/render 1.07  render-occluded-by-person 0.09 | 230 mask/render 1.07  rende …[+4502 chars]
```
</details>

**Bash**
```
cat > draft1.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from trajectory import *
import numpy as np, json, h5py, time
rr=np.load('/work/inputs/reconstruction.npz'); Tw=rr['T_world_mesh'].astype(np.float64)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); Tc0=np.linalg.inv(Twc)@Tw
f=h5py.File('/work/inputs/masks.h5'); O=f['object']; P=f['person']
F=len(Tw)
t0=time.time()
s_end,e_start=detect_static_phases(O,P)
print('static start [0,%d)  end [%d,%d)'%(s_end,e_start,F), time.time()-t0)
vis=np.load('vis.npy'); occ=vis[:,2]/np.maximum(vis[:,1],1)
w=np.clip(1-occ,0.02,1)**2
TA=static_pose_cam(Tc0[:s_end]); TB=static_pose_cam(Tc0[e_start:])
fixed={k:TA for k in range(s_end)}; fixed.update({k:TB for k in range(e_start,F)})
lam=float(sys.argv[1]) if len(sys.argv)>1 else 30
Ts=smooth_se3(Tc0,w,fixed=fixed,lam_rot=lam,lam_tr=lam)
np.save(f'draft1_cam_{lam}.npy',Ts)
from scipy.spatial.transform import Rotation as R
dr=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik',Tc0[:,:3,:3],Ts[:,:3,:3])).as_rotvec(),axis=1))
dt=np.linalg.norm(Ts[:,:3,3]-Tc0[:,:3,3],axis=1)
for a,b in [(0,s_end),(s_end,e_start),(e_start,F),(430,510),(380,405)]:
    print('[%d,%d) mean dR %.2f deg, mean dt %.3f m, max dR %.1f'%(a,b,dr[a:b].mean(),dt[a:b].mean(),dr[a:b].max()))
print(time.time()-t0)
EOF
python draft1.py 30
```

<details><summary>output</summary>

```
static start [0,171)  end [591,715) 29.96831727027893
[0,171) mean dR 0.16 deg, mean dt 0.026 m, max dR 4.6
[171,591) mean dR 2.00 deg, mean dt 0.013 m, max dR 23.4
[591,715) mean dR 0.89 deg, mean dt 0.012 m, max dR 2.2
[430,510) mean dR 5.19 deg, mean dt 0.020 m, max dR 23.4
[380,405) mean dR 2.64 deg, mean dt 0.009 m, max dR 9.2
57.445777893066406
```
</details>

**Bash**
```
cat > cv.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from trajectory import *
import numpy as np, json
from scipy.spatial.transform import Rotation as R
rr=np.load('/work/inputs/reconstruction.npz'); Tw=rr['T_world_mesh'].astype(np.float64)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); Tc0=np.linalg.inv(Twc)@Tw
vis=np.load('vis.npy'); occ=vis[:,2]/np.maximum(vis[:,1],1); w=np.clip(1-occ,0.02,1)**2
a,b=171,591
sub=Tc0[a:b]; ww=w[a:b].copy()
ho=np.zeros(len(sub),bool); ho[2::5]=True; ho&=ww>0.8
ww2=ww.copy(); ww2[ho]=0
for lam in [1,3,10,30,100]:
    Ts=smooth_se3(sub,ww2,lam_rot=lam,lam_tr=lam,iters=1500)
    dr=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik',sub[:,:3,:3],Ts[:,:3,:3])).as_rotvec(),axis=1))
    dt=np.linalg.norm(Ts[:,:3,3]-sub[:,:3,3],axis=1)
    print('lam',lam,'heldout rot err %.2f deg  trans err %.2f cm | train rot %.2f trans %.2f'%(dr[ho].mean(),100*dt[ho].mean(),dr[~ho&(ww>0.8)].mean(),100*dt[~ho&(ww>0.8)].mean()))
EOF
python cv.py
```

<details><summary>output</summary>

```
lam 1 heldout rot err 0.95 deg  trans err 1.05 cm | train rot 0.64 trans 0.71
lam 3 heldout rot err 1.00 deg  trans err 1.03 cm | train rot 0.77 trans 0.82
lam 10 heldout rot err 1.07 deg  trans err 1.08 cm | train rot 0.97 trans 0.96
lam 30 heldout rot err 1.28 deg  trans err 1.17 cm | train rot 1.26 trans 1.11
lam 100 heldout rot err 1.80 deg  trans err 1.34 cm | train rot 1.81 trans 1.33
```
</details>

**Bash**
```
cat > hue.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
cxy=np.array([0.005,-0.004,0]); Fz=np.eye(4); Fz[:3,:3]=np.diag([-1,-1,1.]); Fz[:3,3]=2*cxy
f=h5py.File('/work/inputs/masks.h5')
def stats(rgb,mask):
    R_,G,B=[rgb[...,k].astype(float)+1 for k in range(3)]
    sel=mask&(R_>1.5*G)&(R_>60)
    if sel.sum()<5: return (0,0,0)
    return sel.sum(), np.median(G[sel]/R_[sel]), np.median(B[sel]/R_[sel])
for i in [int(x) for x in sys.argv[1].split(',')]:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[:,:,::-1]
    om=f['object'][i]&~f['person'][i]
    s=[stats(img,om)]
    for T in [Tc0[i],Tc0[i]@Fz]:
        mk,rgb,_,_=r.render(T[None]); rgb=(rgb[0].cpu().numpy()*255); mk=mk[0].cpu().numpy()
        s.append(stats(rgb,mk&~f['person'][i]))
    print(i,'img n=%d G/R=%.2f B/R=%.2f'%s[0],'| base n=%d G/R=%.2f B/R=%.2f'%s[1],'| flip n=%d G/R=%.2f B/R=%.2f'%s[2])
EOF
python hue.py 0,40,100,200,250,300,336,390,420,520,560,600,650,700
```

<details><summary>output</summary>

```
0 img n=758 G/R=0.45 B/R=0.03 | base n=723 G/R=0.44 B/R=0.05 | flip n=773 G/R=0.46 B/R=0.09
40 img n=768 G/R=0.45 B/R=0.03 | base n=734 G/R=0.44 B/R=0.05 | flip n=790 G/R=0.46 B/R=0.09
100 img n=782 G/R=0.45 B/R=0.03 | base n=693 G/R=0.44 B/R=0.05 | flip n=751 G/R=0.46 B/R=0.09
200 img n=1000 G/R=0.47 B/R=0.02 | base n=818 G/R=0.44 B/R=0.04 | flip n=818 G/R=0.46 B/R=0.08
250 img n=1015 G/R=0.45 B/R=0.04 | base n=881 G/R=0.45 B/R=0.06 | flip n=850 G/R=0.45 B/R=0.07
300 img n=1167 G/R=0.51 B/R=0.11 | base n=1115 G/R=0.45 B/R=0.08 | flip n=1105 G/R=0.44 B/R=0.08
336 img n=2222 G/R=0.42 B/R=0.03 | base n=1940 G/R=0.44 B/R=0.05 | flip n=1815 G/R=0.45 B/R=0.07
390 img n=1056 G/R=0.40 B/R=0.11 | base n=1123 G/R=0.43 B/R=0.09 | flip n=1059 G/R=0.45 B/R=0.12
420 img n=1062 G/R=0.48 B/R=0.15 | base  …[+526 chars]
```
</details>

> *thinking:* The color ratio check confirms the baseline orientation is correct, not the flipped one, based on pinkness matching in both static phases. Next I'll look at the edge-on segment 380–405, where the baseline render covers only about half the mask area.

**Bash**
```
COLS=4 python overlay.py --frames $(seq -s, 376 2 406) --out c4.jpg --size 220
```

**Read**
```
/work/scratch/c4.jpg
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
cat >> /work/outputs/evidence.py <<'EOF'

def refine_yaw(r, ev, ids, Tc, iters=100, lam=0.3, tilt_sigma_deg=2.0, tilt_w=0.002, lr=(4e-3, 2e-3)):
    """Refine poses with rotation parametrised in the MESH frame, R = R0 exp([a, b, yaw]): yaw about the mesh's
    vertical axis is free, tilts a, b carry a Gaussian prior (sigma tilt_sigma_deg) towards the input pose;
    image-plane translation free, camera depth fixed. Data term: silhouette + lam * photometric."""
    P, tg, vd, img = ev.batch(ids)
    T0 = torch.tensor(Tc, dtype=torch.float32, device="cuda")
    B = len(ids)
    w = torch.zeros(B, 3, device="cuda", requires_grad=True); t = torch.zeros(B, 2, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr[0]}, {"params": [t], "lr": lr[1]}])
    S = ev.S; sig = np.radians(tilt_sigma_deg)
    for it in range(iters + 1):
        Rd = rotvec_to_mat(w)
        tt = torch.cat([T0[:, :2, 3] + t, T0[:, 2:3, 3]], 1)
        T = torch.cat([torch.cat([T0[:, :3, :3] @ Rd, tt[..., None]], 2), T0[:, 3:]], 1)
        with torch.set_grad_enabled(it < iters):
            sil, rgb, _ = render_batch(r, T, P, S, with_tex=lam > 0)
            ls = (((sil - tg) ** 2) * vd).sum((1, 2)) / tg.sum((1, 2)).clamp(min=50)
            lp = photo_loss(rgb, sil, img, vd) if lam > 0 else torch.zeros_like(ls)
            prior = tilt_w * (w[:, :2] ** 2).sum(1) / sig ** 2
            l = ls + lam * lp + prior
            if it < iters:
                opt.zero_grad(); l.sum( …[+1655 chars]
```

<details><summary>output</summary>

```
t 57.1
171 iou 0.869 -> 0.897 tilt -1.2 1.1 yaw -0.5 | 174 iou 0.851 -> 0.878 tilt -1.5 0.5 yaw -1.6 | 177 iou 0.869 -> 0.902 tilt -2.1 0.2 yaw -0.8 | 
180 iou 0.885 -> 0.901 tilt -0.9 -1.0 yaw -3.5 | 183 iou 0.876 -> 0.912 tilt -0.1 -1.0 yaw -4.8 | 186 iou 0.858 -> 0.900 tilt -0.3 -0.2 yaw -3.3 | 
189 iou 0.863 -> 0.914 tilt -0.5 -0.5 yaw -0.3 | 192 iou 0.847 -> 0.893 tilt -0.2 1.0 yaw -2.6 | 195 iou 0.853 -> 0.882 tilt -0.6 1.3 yaw -1.2 | 
198 iou 0.878 -> 0.885 tilt -0.0 1.7 yaw 0.3 | 201 iou 0.867 -> 0.897 tilt -0.3 2.4 yaw 1.6 | 204 iou 0.895 -> 0.918 tilt 0.4 1.2 yaw -4.0 | 
207 iou 0.871 -> 0.892 tilt -0.7 1.5 yaw -0.9 | 210 iou 0.847 -> 0.903 tilt -0.6 0.0 yaw 0.6 | 213 iou 0.853 -> 0.915 tilt -1.8 1.4 yaw 3.2 | 
216 iou 0.846 -> 0.907 tilt -1.1 0.4 yaw 0.7 | 219 iou 0.829 -> 0.900 …[+5870 chars]
```
</details>

> *thinking:* The yaw corrections form coherent temporal patterns (consistent shifts across frame ranges, with IoU improving notably in the 405–480 range), suggesting genuine systematic baseline errors rather than noise. Next I'll test this on the static phases, then run a held-out check by fitting on even frames and evaluating on odd frames.

**Bash**
```
cat > yawref_static.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
from trajectory import chordal_mean
import pickle, time
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
for a,b in [(0,169),(591,715)]:
    ids=np.arange(a,b,3)
    T,iou,ls,lp,W=refine_yaw(r,ev,ids,Tc0[ids],iters=100,lam=0.3)
    _,iou0,_,_,_=refine_yaw(r,ev,ids,Tc0[ids],iters=0,lam=0.3)
    W=np.degrees(W)
    print(a,b,'iou %.3f -> %.3f'%(iou0.mean(),iou.mean()),'tilt mean %.2f %.2f std %.2f %.2f | yaw mean %.2f std %.2f'%(W[:,0].mean(),W[:,1].mean(),W[:,0].std(),W[:,1].std(),W[:,2].mean(),W[:,2].std()))
    print('  dxy mean (cm)', np.round(100*(T[:,:2,3]-Tc0[ids,:2,3]).mean(0),2), 'std', np.round(100*(T[:,:2,3]-Tc0[ids,:2,3]).std(0),2))
EOF
python yawref_static.py
```

<details><summary>output</summary>

```
0 169 iou 0.833 -> 0.853 tilt mean 0.16 0.77 std 0.43 0.83 | yaw mean -2.93 std 0.65
  dxy mean (cm) [-0.58  0.22] std [0.2 0.2]
591 715 iou 0.587 -> 0.707 tilt mean 7.32 1.19 std 1.22 0.28 | yaw mean -2.21 std 1.02
  dxy mean (cm) [-0.28  2.12] std [0.25 0.73]
```
</details>

**Bash**
```
sed -i 's/outs.append(refine_yaw(r,ev,ids\[k:k+60\],Tc0\[ids\[k:k+60\]\],iters=100,lam=lam))/outs.append(refine_yaw(r,ev,ids[k:k+60],Tc0[ids[k:k+60]],iters=100,lam=lam,tilt_w=float(sys.argv[4])))/' yawref_test.py && sed -i "s/np.savez(f'yawref_{a}_{b}_{lam}.npz'/np.savez(f'yawref_{a}_{b}_{lam}_{sys.argv[4]}.npz'/" yawref_test.py && python yawref_test.py 0 715 0.3 1.0 > yawref2.log; head -1 yawref2.log; awk 'NR>1 && NR%2==0{printf "%s | ", $0; if (NR%6==0) print ""}' yawref2.log | sed -n '28,80p'
```

<details><summary>output</summary>

```
t 88.3
486 iou 0.808 -> 0.830 tilt -0.0 0.0 yaw -3.4 | 492 iou 0.000 -> 0.000 tilt 0.0 -0.0 yaw 2.0 | 498 iou 0.090 -> 0.000 tilt -0.0 0.1 yaw -0.5 | 
504 iou 0.741 -> 0.823 tilt -0.1 0.0 yaw 8.4 | 510 iou 0.870 -> 0.909 tilt -0.0 -0.0 yaw 1.6 | 516 iou 0.872 -> 0.912 tilt 0.0 0.0 yaw -0.9 | 
522 iou 0.830 -> 0.893 tilt 0.1 -0.0 yaw -0.8 | 528 iou 0.803 -> 0.858 tilt 0.0 -0.0 yaw -1.2 | 534 iou 0.721 -> 0.819 tilt 0.1 -0.1 yaw -1.3 | 
540 iou 0.709 -> 0.790 tilt 0.0 0.0 yaw -2.5 | 546 iou 0.674 -> 0.782 tilt -0.0 0.1 yaw -2.2 | 552 iou 0.641 -> 0.778 tilt -0.0 0.1 yaw -4.7 | 
558 iou 0.587 -> 0.706 tilt 0.0 0.0 yaw -1.1 | 564 iou 0.484 -> 0.641 tilt 0.0 0.1 yaw -3.7 | 570 iou 0.499 -> 0.563 tilt 0.0 0.1 yaw -0.4 | 
576 iou 0.507 -> 0.587 tilt 0.0 0.0 yaw 3.7 | 582 iou 0.496 -> 0.550 tilt 0 …[+1058 chars]
```
</details>

**Bash**
```
awk 'NR>1 && NR%2==0{printf "%s | ", $0; if (NR%6==0) print ""}' yawref2.log | sed -n '1,27p'
```

<details><summary>output</summary>

```
0 iou 0.825 -> 0.847 tilt 0.0 -0.0 yaw -3.6 | 6 iou 0.822 -> 0.841 tilt -0.0 0.1 yaw -3.2 | 12 iou 0.821 -> 0.836 tilt -0.0 -0.0 yaw -3.0 | 
18 iou 0.826 -> 0.842 tilt 0.0 -0.0 yaw -2.9 | 24 iou 0.826 -> 0.838 tilt -0.0 0.0 yaw -3.4 | 30 iou 0.826 -> 0.839 tilt -0.0 0.0 yaw -2.8 | 
36 iou 0.828 -> 0.842 tilt 0.0 0.0 yaw -3.1 | 42 iou 0.824 -> 0.839 tilt -0.0 -0.0 yaw -2.8 | 48 iou 0.826 -> 0.842 tilt -0.0 0.0 yaw -3.3 | 
54 iou 0.826 -> 0.846 tilt -0.0 0.0 yaw -3.3 | 60 iou 0.827 -> 0.843 tilt -0.0 0.1 yaw -3.6 | 66 iou 0.824 -> 0.841 tilt -0.0 0.0 yaw -3.2 | 
72 iou 0.822 -> 0.836 tilt 0.0 -0.0 yaw -3.7 | 78 iou 0.828 -> 0.846 tilt -0.0 0.0 yaw -3.9 | 84 iou 0.828 -> 0.844 tilt -0.0 -0.0 yaw -3.8 | 
90 iou 0.828 -> 0.843 tilt 0.0 -0.0 yaw -3.7 | 96 iou 0.830 -> 0.843 tilt 0.0 -0.0 yaw -3. …[+3055 chars]
```
</details>

> *thinking:* With tilt fixed, the start static phase shows a consistent yaw correction of −3.0 to −3.9°, while the end phase stays near 0° once the hand is removed. Now I'll build the full solve.py pipeline.

**Bash**
```
cd /work/outputs && python - <<'EOF'
import re
s=open('evidence.py').read()
s=s.replace('''    def __init__(self, inputs, frames, renderer, K, Tc_ref, S=320, margin=1.3):
        self.S = S; self.K = K; self.r = renderer
        f = h5py.File(f"{inputs}/masks.h5", "r")
        F_ = len(frames)''','''    def __init__(self, inputs, frames, renderer, K, Tc_ref, S=320, margin=1.3):
        """frames: iterable over the video frames (BGR, in order); Tc_ref: (F,4,4) reference camera poses."""
        self.S = S; self.K = K; self.r = renderer
        f = h5py.File(f"{inputs}/masks.h5", "r")
        F_ = len(Tc_ref)''')
s=s.replace('''        self.area = np.zeros(F_)
        for i, fr in enumerate(frames):''','''        self.area = np.zeros(F_); self.occ = np.zeros(F_)
        for i, fr in enumerate(frames):
            if i >= F_: break''')
s=s.replace('''            self.obj[i] = o; self.valid[i] = ((p < 0.5) | (o > 0.5)).astype(np.float32)
        self.P''','''            self.obj[i] = o; self.valid[i] = ((p < 0.5) | (o > 0.5)).astype(np.float32)
            self.occ[i] = (mk & pm & ~om).sum() / max(mk.sum(), 1)   # fraction of the reference render hidden by the person
        self.P''')
open('evidence.py','w').write(s)
s=open('common.py').read()
s=s.replace('''def read_video(path):
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    return frames''','''def read_video(path):
    cap = cv2.Vid …[+365 chars]
```

<details><summary>output</summary>

```
22:        self.area = np.zeros(F_); self.occ = np.zeros(F_)
39:            self.occ[i] = (mk & pm & ~om).sum() / max(mk.sum(), 1)   # fraction of the reference render hidden by the person
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
cd /work/scratch && python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/test_out.npz --diag /work/scratch/test_diag.npz 2>&1 | tail -20
```

<details><summary>output</summary>

```
15:37:59 building evidence crops
15:39:17 per-frame refinement
15:40:44 refinement done; mean |dyaw| 3.21 deg
15:41:08 phases: static [0,169)  carried [169,592)  static [592,715)
15:41:17 CV lam=1.0: held-out rot 1.20 deg, trans 1.01 cm
15:41:24 CV lam=3.0: held-out rot 1.32 deg, trans 0.99 cm
15:41:32 CV lam=10.0: held-out rot 1.44 deg, trans 1.03 cm
15:41:39 CV lam=30.0: held-out rot 1.70 deg, trans 1.12 cm
15:41:39 chosen lambda 1.0
15:41:52 phase [0,169) depth scale 1.00000
15:41:52 phase [169,592) depth scale 1.00011
15:41:52 phase [592,715) depth scale 1.00000
15:41:52 saved /work/scratch/test_out.npz  (234s)
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
python /work/outputs/validate.py --result test_out.npz --diag test_diag.npz 2>&1 | tail -8
```

<details><summary>output</summary>

```
phase=static start | frames=0-168 | n=169 | iou_base=0.832 | iou_res=0.853 | photo_base=1.156 | photo_res=0.982 | jit_base=0.033 | jit_res=0.000 | dR_mean=3.361 | dt_mean_cm=2.713 | dz_mean_cm=0.060 | dz_median_cm=0.000 | ho_base=0.832 | ho_res=0.853 | floor_base_cm=1.574 | floor_res_cm=1.182
phase=carried | frames=169-591 | n=423 | iou_base=0.774 | iou_res=0.831 | photo_base=1.242 | photo_res=1.068 | jit_base=2.648 | jit_res=0.434 | dR_mean=3.959 | dt_mean_cm=1.859 | dz_mean_cm=0.032 | dz_median_cm=0.058 | ho_base=0.771 | ho_res=0.829 | floor_base_cm=14.145 | floor_res_cm=13.384
phase=static end | frames=592-714 | n=123 | iou_base=0.588 | iou_res=0.658 | photo_base=1.398 | photo_res=1.335 | jit_base=0.213 | jit_res=0.000 | dR_mean=0.836 | dt_mean_cm=1.738 | dz_mean_cm=-0.256 | dz_median_c …[+579 chars]
```
</details>

> *thinking:* The first full run shows IoU and photo residuals improving across all phases (static start, carried, static end), with held-out frames improving just as much as fitted ones, confirming this isn't overfitting. Next I'll check the overlays.

**Bash**
```
COLS=3 python overlay.py --poses test_out.npz --frames 0,390,438,462,495,502,560,650,300 --out r1.jpg --size 260
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
cat > cues.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, json, trimesh
from handframe import hand_frames
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); d=pl[3]
m=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry(); v=np.asarray(m.vertices)[::5]
Tw=np.load(sys.argv[1])['T_world_mesh']; Tc=np.linalg.inv(Twc)@Tw
rec=np.load('/work/inputs/reconstruction.npz'); kp=rec['human_keypoints_world'].astype(np.float64)
def floor_min(Tcam):
    w=(Twc[:3,:3]@((Tcam[:3,:3]@v.T).T+Tcam[:3,3]).T).T+Twc[:3,3]; return (w@n+d).min()
for name,(a,b) in [('start',(0,169)),('end',(592,715))]:
    i=(a+b)//2; T=Tc[i]
    for target in [0.0,0.018]:
        # find depth scale s (move along ray) s.t. floor_min = target
        ss=np.linspace(0.9,1.1,2001); vals=[]
        for s in ss:
            T2=T.copy(); T2[:3,3]*=s; vals.append(floor_min(T2))
        vals=np.array(vals); k=np.argmin(np.abs(vals-target))
        print(name,'floor target %.3f: depth shift %.1f cm (cur floor dist %.2f cm)'%(target,(ss[k]-1)*T[2,3]*100, 100*floor_min(T)))
# hand cue: handle (top of mesh) vs right-hand grip point (mean of finger bases)
handle=np.array([0,0,0.32])
shifts=[]
for i in range(200,590):
    g=kp[i,[28,32,36,40]].mean(0); gc=np.linalg.inv(Twc)[:3,:3]@g+np.linalg.inv(Twc)[:3,3]
    hc=Tc[i,:3,:3]@handle+Tc[i,:3,3]
    # move object alo …[+796 chars]
```

<details><summary>output</summary>

```
start floor target 0.000: depth shift 3.9 cm (cur floor dist 1.17 cm)
start floor target 0.018: depth shift -2.1 cm (cur floor dist 1.17 cm)
end floor target 0.000: depth shift -0.8 cm (cur floor dist -0.30 cm)
end floor target 0.018: depth shift -5.6 cm (cur floor dist -0.30 cm)
hand cue: depth shift along ray median -5.9 cm, IQR [-8.0, -2.9]
frame-0 human depth: reconstructed median z 2.570 vs reference 2.548 (diff 2.2 cm)
```
</details>

> *thinking:* Depth cues conflict (floor-contact estimates vs. hand-grip estimates don't agree), so I'll leave depth unchanged for now. Next I'll run a second refinement pass starting from the smoothed trajectory to see if anything still shifts.

**Bash**
```
cat > pass2.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle
from scipy.spatial.transform import Rotation as R
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
d=np.load('test_diag.npz'); Ts=d['Ts']; Tref=d['Tref']
ids=np.arange(169,592)
outs=[refine_yaw(r,ev,ids[k:k+60],Ts[ids[k:k+60]],iters=100,lam=0.3,tilt_w=1.0) for k in range(0,len(ids),60)]
T2=np.concatenate([o[0] for o in outs]); iou2=np.concatenate([o[1] for o in outs])
dy=np.degrees(np.concatenate([o[4] for o in outs]))[:,2]
d1=np.array([np.degrees(R.from_matrix(T2[k,:3,:3]@Tref[i,:3,:3].T).magnitude()) for k,i in enumerate(ids)])
print('pass2 yaw change from smoothed: mean |%.2f|, max %.1f; diff to pass1 refined obs mean %.2f'%(np.abs(dy).mean(),np.abs(dy).max(),d1.mean()))
print('IoU pass1 refined %.3f, smoothed %.3f, pass2 %.3f'%(d['iou1'][ids].mean(), np.nan, iou2.mean()))
for k in range(0,len(ids),12): print(ids[k],'dyaw %.1f iou %.3f/%.3f'%(dy[k],d['iou1'][ids[k]],iou2[k]))
EOF
python pass2.py | head -3; python pass2.py | tail -36 | awk '{printf "%s | ", $0; if (NR%4==0) print ""}'
```

<details><summary>output</summary>

```
pass2 yaw change from smoothed: mean |1.04|, max 17.6; diff to pass1 refined obs mean 1.41
IoU pass1 refined 0.817, smoothed nan, pass2 0.820
169 dyaw -0.5 iou 0.889/0.878
Traceback (most recent call last):
  File "/work/scratch/pass2.py", line 15, in <module>
    for k in range(0,len(ids),12): print(ids[k],'dyaw %.1f iou %.3f/%.3f'%(dy[k],d['iou1'][ids[k]],iou2[k]))
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
BrokenPipeError: [Errno 32] Broken pipe
169 dyaw -0.6 iou 0.889/0.877 | 181 dyaw -0.5 iou 0.895/0.903 | 193 dyaw -0.2 iou 0.884/0.882 | 205 dyaw -0.3 iou 0.906/0.904 | 
217 dyaw 1.4 iou 0.889/0.890 | 229 dyaw 0.3 iou 0.908/0.909 | 241 dyaw 0.1 iou 0.896/0.895 | 253 dyaw -0.4 iou 0.868/0.877 | 
265 dyaw 0.2 iou 0.897/0.8 …[+857 chars]
```
</details>

**Bash**
```
cat > robust.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('/work/scratch/ev320.pkl','rb'))
ids=np.arange(0,715,3); Y={}
for lam in [0.0,0.3,1.0]:
    Y[lam]=np.degrees(np.concatenate([refine_yaw(r,ev,ids[k:k+60],Tc0[ids[k:k+60]],iters=100,lam=lam,tilt_w=1.0)[4] for k in range(0,len(ids),60)]))[:,2]
segs=[(0,169),(169,375),(375,405),(405,486),(505,592),(592,715)]
for a,b in segs:
    s=(ids>=a)&(ids<b)
    print('[%d,%d) mean yaw corr: sil-only %.2f, lam0.3 %.2f, lam1 %.2f | corr(sil,lam1)=%.2f'%(a,b,Y[0.0][s].mean(),Y[0.3][s].mean(),Y[1.0][s].mean(),np.corrcoef(Y[0.0][s],Y[1.0][s])[0,1]))
EOF
python robust.py
```

<details><summary>output</summary>

```
[0,169) mean yaw corr: sil-only -3.50, lam0.3 -3.29, lam1 -3.51 | corr(sil,lam1)=0.26
[169,375) mean yaw corr: sil-only 2.49, lam0.3 1.00, lam1 0.77 | corr(sil,lam1)=0.91
[375,405) mean yaw corr: sil-only -2.03, lam0.3 -3.16, lam1 -6.64 | corr(sil,lam1)=0.88
[405,486) mean yaw corr: sil-only 4.85, lam0.3 7.67, lam1 8.16 | corr(sil,lam1)=0.69
[505,592) mean yaw corr: sil-only -0.62, lam0.3 -0.46, lam1 -1.21 | corr(sil,lam1)=0.68
[592,715) mean yaw corr: sil-only -1.56, lam0.3 -0.76, lam1 -1.56 | corr(sil,lam1)=0.53
```
</details>

> *thinking:* The robustness check holds up—silhouette-only and texture-weighted refinements agree on direction and magnitude of yaw corrections, and a second pass only shifts things ~1°, confirming one pass suffices. Now running the final solve.py.

**Bash**
```
cd /work && (time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz --diag /work/outputs/diag.npz) 2>&1 | tail -16
```

<details><summary>output</summary>

```
15:50:34 per-frame refinement
15:52:04 refinement done; mean |dyaw| 3.18 deg
15:52:27 phases: static [0,169)  carried [169,592)  static [592,715)
15:52:33 CV lam=1.0: held-out rot 1.24 deg, trans 1.02 cm
15:52:39 CV lam=3.0: held-out rot 1.30 deg, trans 0.98 cm
15:52:44 CV lam=10.0: held-out rot 1.36 deg, trans 1.00 cm
15:52:49 CV lam=30.0: held-out rot 1.62 deg, trans 1.08 cm
15:52:49 chosen lambda 1.0
15:52:57 phase [0,169) depth scale 1.00000
15:52:57 phase [169,592) depth scale 1.00013
15:52:57 phase [592,715) depth scale 1.00000
15:52:58 saved /work/outputs/object_poses_world.npz  (221s)

real	3m44.153s
user	15m5.882s
sys	0m9.127s
```
</details>

**Bash**
```
cd /work/scratch && python -c "
import numpy as np
from scipy.spatial.transform import Rotation as R
a=np.load('/work/outputs/object_poses_world.npz'); T=a['T_world_mesh']; print(list(a.keys()), T.shape, T.dtype, np.isfinite(T).all(), np.abs(T[:,3]-[0,0,0,1]).max())
print('det', np.abs(np.linalg.det(T[:,:3,:3])-1).max())
b=np.load('test_out.npz')['T_world_mesh']
dr=np.degrees(R.from_matrix(np.einsum('nji,njk->nik',T[:,:3,:3],b[:,:3,:3])).magnitude()); dt=np.linalg.norm(T[:,:3,3]-b[:,:3,3],axis=1)
print('run-to-run diff: rot mean %.3f max %.3f deg; trans mean %.2f max %.2f mm'%(dr.mean(),dr.max(),1000*dt.mean(),1000*dt.max()))
" && python /work/outputs/validate.py --result /work/outputs/object_poses_world.npz --diag /work/outputs/diag.npz 2>&1 | tail -6
```

<details><summary>output</summary>

```
['T_world_mesh'] (715, 4, 4) float64 True 0.0
det 3.78920292198881e-06
run-to-run diff: rot mean 0.126 max 1.446 deg; trans mean 1.00 max 10.96 mm
phase=static start | frames=0-168 | n=169 | iou_base=0.832 | iou_res=0.852 | photo_base=1.156 | photo_res=0.978 | jit_base=0.033 | jit_res=0.000 | dR_mean=3.309 | dt_mean_cm=2.713 | dz_mean_cm=0.060 | dz_median_cm=0.000 | ho_base=0.832 | ho_res=0.852 | floor_base_cm=1.574 | floor_res_cm=1.195
phase=carried | frames=169-591 | n=423 | iou_base=0.774 | iou_res=0.830 | photo_base=1.242 | photo_res=1.068 | jit_base=2.648 | jit_res=0.431 | dR_mean=3.920 | dt_mean_cm=1.861 | dz_mean_cm=0.039 | dz_median_cm=0.067 | ho_base=0.771 | ho_res=0.830 | floor_base_cm=14.145 | floor_res_cm=13.369
phase=static end | frames=592-714 | n=123 | iou_base=0.588 | iou_r …[+726 chars]
```
</details>

**Bash**
```
python -c "
import numpy, scipy, torch, cv2, trimesh, h5py, nvdiffrast
print(numpy.__version__, scipy.__version__, torch.__version__, cv2.__version__, trimesh.__version__, h5py.__version__, nvdiffrast.__version__, torch.version.cuda)"; python --version; ls /work/outputs; python -c "
import numpy as np
d=np.load('/work/outputs/diag.npz'); print(d['lam'], d['cv'])"
```

<details><summary>output</summary>

```
1.26.3 1.16.3 2.5.1+cu124 4.11.0 5.1.1 3.16.0 0.4.0 12.4
Python 3.11.10
__pycache__
common.py
diag.npz
diag_val.npy
evidence.py
handframe.py
object_poses_world.npz
solve.py
trajectory.py
validate.py
1.0 [0.46046788 0.46055742 0.47681521 0.54034775]
```
</details>

**Bash**
```
mkdir -p /work/outputs/analysis && cp cues.py /work/outputs/analysis/depth_cues.py && cp ana6.py /work/outputs/analysis/body_relative_yaw.py && cp hue.py /work/outputs/analysis/flip_hue_check.py && cp yawscan2.py /work/outputs/analysis/yaw_scan.py && cp robust.py /work/outputs/analysis/yaw_robustness.py && sed -i "1i # Analysis script (not used by solve.py). Reads frames extracted to /work/scratch/frames/%04d.jpg." /work/outputs/analysis/flip_hue_check.py && sed -i "1i # Analysis script (not used by solve.py). Needs the evidence cache built by an earlier scratch script (ev320.pkl)." /work/outputs/analysis/yaw_scan.py /work/outputs/analysis/yaw_robustness.py && sed -i "1i # Analysis script (not used by solve.py). Usage: python depth_cues.py object_poses_world.npz" /work/outputs/analysis/depth_cues.py && rm -rf /work/outputs/__pycache__ && ls /work/outputs/analysis
```

<details><summary>output</summary>

```
body_relative_yaw.py
depth_cues.py
flip_hue_check.py
yaw_robustness.py
yaw_scan.py
```
</details>

**Bash**
```
python /work/outputs/analysis/depth_cues.py /work/outputs/object_poses_world.npz
```

<details><summary>output</summary>

```
start floor target 0.000: depth shift 3.9 cm (cur floor dist 1.18 cm)
start floor target 0.018: depth shift -2.1 cm (cur floor dist 1.18 cm)
end floor target 0.000: depth shift -0.8 cm (cur floor dist -0.32 cm)
end floor target 0.018: depth shift -5.6 cm (cur floor dist -0.32 cm)
hand cue: depth shift along ray median -5.9 cm, IQR [-8.0, -2.9]
frame-0 human depth: reconstructed median z 2.570 vs reference 2.548 (diff 2.2 cm)
```
</details>

**Agent:** Now the report and tools list.

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
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from evidence import *
import pickle, torch
K,Kp,Twc=load_camera('/work/inputs'); m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
rr=np.load('/work/inputs/reconstruction.npz'); Tc0=np.linalg.inv(Twc)@rr['T_world_mesh'].astype(np.float64)
ev=pickle.load(open('ev320.pkl','rb'))
ids=np.arange(200,260); refine_yaw(r,ev,ids,Tc0[ids],iters=5,lam=0.3,tilt_w=1.0)
print('peak GB', torch.cuda.max_memory_allocated()/1e9, torch.cuda.max_memory_reserved()/1e9)
"
```

<details><summary>output</summary>

```
peak GB 3.014537216 4.051697664
```
</details>

**Bash**
```
cd /work/outputs && sed -i 's/peak GPU memory < 3 GB)/peak GPU memory about 3 GB allocated \/ 4 GB reserved, measured on one 60-frame batch)/' REPORT.md && grep -n "peak GPU" REPORT.md && ls -la /work/outputs
```

<details><summary>output</summary>

```
5:(run time 3 min 44 s wall, about 2.5 min of it on the GPU; peak GPU memory about 3 GB allocated / 4 GB reserved, measured on one 60-frame batch). Validation: `validate.py` (numbers below).
total 468
drwxrwxr-x 4 hanzhou hanzhou   4096 Oct 21 15:51 .
drwxrwxr-x 7 hanzhou hanzhou   4096 Oct 21 14:52 ..
-rw-rw-r-- 1 hanzhou hanzhou  11499 Oct 21 15:51 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   1201 Oct 21 15:50 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 21 15:51 __pycache__
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 21 15:48 analysis
-rw-rw-r-- 1 hanzhou hanzhou   5154 Oct 21 15:31 common.py
-rw-rw-r-- 1 hanzhou hanzhou 299934 Oct 21 15:46 diag.npz
-rw-rw-r-- 1 hanzhou hanzhou   2412 Oct 21 15:48 diag_val.npy
-rw-rw-r-- 1 hanzhou hanzhou   6967 Oct 21 15:31 evidence.py
-rw-rw-r-- 1 hanzhou …[+295 chars]
```
</details>

**Bash**
```
rm -rf /work/outputs/__pycache__ /work/outputs/analysis/__pycache__; ls /work/outputs
```

<details><summary>output</summary>

```
REPORT.md
TOOLS.md
analysis
common.py
diag.npz
diag_val.npy
evidence.py
handframe.py
object_poses_world.npz
solve.py
trajectory.py
validate.py
```
</details>

**Agent:** I ran `solve.py` once as specified. It wrote `object_poses_world.npz` (715 × 4 × 4, float64, all values finite) in 3 min 44 s, and `REPORT.md` and `TOOLS.md` are written. By my checks the result explains the video better than the baseline in all three phases, and it keeps the baseline's median depth exactly. There is no ground truth, so how much ADD actually improves can't be measured here.

**Phases** (found automatically from the masks):
- **Static start, 0–168:** the sign stands untouched on the floor.
- **Carried, 169–591:** the person lifts it by the handle, walks a full loop, and it is completely hidden behind them at 488–500.
- **Static end, 592–714:** the sign is back on the floor.

**What I changed:**
- **Static phases get one pose each.** The baseline's depth wandered between 3.55 and 3.66 m while the sign sat untouched (2.7 cm mean, up to about 6 cm). The start pose also turns by −3.5°; every variant of the image fit agreed on that.
- **Carried phase: per-frame yaw and image position re-fitted** by rendering the textured mesh and comparing it with the mask and image. Tilt and depth stay at the baseline values. Letting tilt move produced a 7–13° tilt that would lift two of the sign's feet off the floor, so I treat silhouette tilt as unreliable here.
- **Carried phase is then smoothed over time**, with heavily occluded frames given little weight. This replaces the baseline's 44° jump while the sign is hidden, and the smoothing strength was chosen by leaving out frames and predicting them.
- **Depth along the viewing ray: no change.** The independent depth cues disagree by more than 2 cm (floor contact +3.9/−2.1 cm at the start, −0.8/−5.6 cm at the end; hand grip −5.9 cm). Per-phase median depth matches the baseline exactly.
- **No 180° flip.** A photometric "flipped or not" test gave contradictory answers, sometimes within a single static phase, so I didn't trust it. Three stronger cues say the baseline orientation is right: the sign's yaw stays locked to the person's body through a full 360° turn, the pick-up can't flip it without an edge-on view that never happens, and the colour of the asymmetric tape matches the baseline in both static phases. A flip of the whole episode can't be completely ruled out, but the tape colour argues against it.

**Validation** (overlap of the rendered sign with the object mask; "held-out" means the trajectory was refitted without those frames):

| phase | overlap, baseline → result | held-out frames, baseline → result |
|---|---|---|
| static start | 0.832 → 0.852 | 0.832 → 0.852 |
| carried | 0.774 → 0.830 | 0.771 → 0.830 |
| static end | 0.588 → 0.658 | 0.589 → 0.658 |

The colour residual also drops in every phase.

**Confidence:**
- **Static phases:** high.
- **Carried phase:** medium-high overall. It is lowest for the fully hidden frames 488–500, which are interpolated, and the edge-on stretch 375–405, where the outline is very thin.

`solve.py` is not bit-exactly reproducible even with fixed seeds, because the GPU renderer's gradient accumulation isn't deterministic. Two runs differ by 0.13° and 1 mm on average, and by at most 1.4° and 11 mm on hidden frames. GPU use stayed around 3–4 GB and well under the 45-minute budget, and I used no external tools, downloads or weights.

Everything is in `/work/outputs`; `REPORT.md` has the full tables and `validate.py` regenerates them.

## Final

I ran `solve.py` once as specified. It wrote `object_poses_world.npz` (715 × 4 × 4, float64, all values finite) in 3 min 44 s, and `REPORT.md` and `TOOLS.md` are written. By my checks the result explains the video better than the baseline in all three phases, and it keeps the baseline's median depth exactly. There is no ground truth, so how much ADD actually improves can't be measured here.

**Phases** (found automatically from the masks):
- **Static start, 0–168:** the sign stands untouched on the floor.
- **Carried, 169–591:** the person lifts it by the handle, walks a full loop, and it is completely hidden behind them at 488–500.
- **Static end, 592–714:** the sign is back on the floor.

**What I changed:**
- **Static phases get one pose each.** The baseline's depth wandered between 3.55 and 3.66 m while the sign sat untouched (2.7 cm mean, up to about 6 cm). The start pose also turns by −3.5°; every variant of the image fit agreed on that.
- **Carried phase: per-frame yaw and image position re-fitted** by rendering the textured mesh and comparing it with the mask and image. Tilt and depth stay at the baseline values. Letting tilt move produced a 7–13° tilt that would lift two of the sign's feet off the floor, so I treat silhouette tilt as unreliable here.
- **Carried phase is then smoothed over time**, with heavily occluded frames given little weight. This replaces the baseline's 44° jump while the sign is hidden, and the smoothing strength was chosen by leaving out frames and predicting them.
- **Depth along the viewing ray: no change.** The independent depth cues disagree by more than 2 cm (floor contact +3.9/−2.1 cm at the start, −0.8/−5.6 cm at the end; hand grip −5.9 cm). Per-phase median depth matches the baseline exactly.
- **No 180° flip.** A photometric "flipped or not" test gave contradictory answers, sometimes within a single static phase, so I didn't trust it. Three stronger cues say the baseline orientation is right: the sign's yaw stays locked to the person's body through a full 360° turn, the pick-up can't flip it without an edge-on view that never happens, and the colour of the asymmetric tape matches the baseline in both static phases. A flip of the whole episode can't be completely ruled out, but the tape colour argues against it.

**Validation** (overlap of the rendered sign with the object mask; "held-out" means the trajectory was refitted without those frames):

| phase | overlap, baseline → result | held-out frames, baseline → result |
|---|---|---|
| static start | 0.832 → 0.852 | 0.832 → 0.852 |
| carried | 0.774 → 0.830 | 0.771 → 0.830 |
| static end | 0.588 → 0.658 | 0.589 → 0.658 |

The colour residual also drops in every phase.

**Confidence:**
- **Static phases:** high.
- **Carried phase:** medium-high overall. It is lowest for the fully hidden frames 488–500, which are interpolated, and the edge-on stretch 375–405, where the outline is very thin.

`solve.py` is not bit-exactly reproducible even with fixed seeds, because the GPU renderer's gradient accumulation isn't deterministic. Two runs differ by 0.13° and 1 mm on average, and by at most 1.4° and 11 mm on hidden frames. GPU use stayed around 3–4 GB and well under the 45-minute budget, and I used no external tools, downloads or weights.

Everything is in `/work/outputs`; `REPORT.md` has the full tables and `validate.py` regenerates them.
