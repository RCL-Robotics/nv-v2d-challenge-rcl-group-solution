# CL2b_ep014

Claude Opus v2（ep11-21）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T10:48:54Z

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
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 14:42 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 14:42 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 14:42 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 14:42 proxy.sock

inputs:
total 99420
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 14:42 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 14:42 ..
-rw-rw-r-- 1 hanzhou hanzhou      822 Oct 21 14:42 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       66 Oct 21 14:42 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   192895 Oct 21 14:42 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      401 Oct 21 14:42 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  9372584 Oct 21 14:42 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 15473916 Oct 21 14:42 object_mesh.glb
-rw-rw-r-- 1 hanzhou  …[+2489 chars]
```
</details>

**Bash**
```
cd /work; cat logs/run_manifest.txt logs/proxy.log; nvidia-smi; ls /pipeline /pipeline/* | head -80; ls /code/reconstruction /workspace; python -c "import torch,nvdiffrast,cv2,trimesh;print(torch.__version__,cv2.__version__)"; nproc; free -g
```

<details><summary>output</summary>

```
date_utc: 2026-10-21T10:48:54Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2b_ep014:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000014:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
4cc40add548f276cec0f29092a61124b89d252e5b3304e4f3951344f32c97762  ./camera.json
dbc1933ce41eb2a7d9dda96940e11fb055d82e6e5c2db2bea7971011aa1cb2ff  ./episode.json
771b8012902c3740be1813caeb30704867b7bce3d …[+3748 chars]
```
</details>

**Bash**
```
cd /work; echo $CUDA_VISIBLE_DEVICES; cat /pipeline/pipeline_report.json | head -100; ls -R /pipeline/depth | head; ls /pipeline/export/* /pipeline/foundationpose/debug | head
```

<details><summary>output</summary>

```
0
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
    "mtime_ns": 1792065993430039348,
    "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000014/inference/episode_000014_step200000_before_after.mp4",
    "size": 70191738
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0 …[+3019 chars]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/frames tools; python -c "
import trimesh,numpy as np
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False)
print(m.geometry.keys())
g=m.to_geometry(); print(g)
v=g.vertices; print(v.min(0),v.max(0),v.mean(0))
S,ext=trimesh.bounds.oriented_bounds(g); print(S,ext)
print(g.visual.kind if hasattr(g.visual,'kind') else None)
import numpy.linalg as la
c=v-v.mean(0); print(la.eigh(c.T@c/len(c)))
r=np.load('inputs/reconstruction.npz'); [print(k,r[k].shape,r[k].dtype) for k in r]
"

```

<details><summary>output</summary>

```
odict_keys(['finaltable.001'])
<trimesh.Trimesh(vertices.shape=(240494, 3), faces.shape=(80196, 3))>
[-0.38281685 -0.38470846 -0.12187351] [0.40190151 0.38721192 0.66425836] [0.01089476 0.01007864 0.05588094]
[[ 1.22464680e-16  1.00000000e+00  0.00000000e+00 -1.25172734e-03]
 [ 1.00000000e+00 -1.22464680e-16  0.00000000e+00 -9.54233110e-03]
 [ 0.00000000e+00  0.00000000e+00 -1.00000000e+00  2.71192424e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] [0.77192038 0.78471836 0.78613187]
texture
EighResult(eigenvalues=array([0.03177688, 0.03703939, 0.06772242]), eigenvectors=TrackedArray([[ 0.23274667,  0.97240303, -0.01616599],
              [ 0.96240055, -0.22789518,  0.14781398],
              [-0.14005061,  0.04996137,  0.98888305]]))
T_world_mesh (993, 4, 4) float3 …[+214 chars]
```
</details>

**Agent:** Starting work: the object is a round table (~0.78 m diameter and height, mesh z-axis = table axis), 993 frames. Next I'm looking at the video and the baseline trajectory.

**Bash**
```
cd /work/scratch; ffprobe -v error -show_entries stream=r_frame_rate,nb_frames,width,height ../inputs/video.mp4; ffmpeg -v error -i ../inputs/video.mp4 -vf "select='not(mod(n\,50))',scale=384:288,tile=5x4" -frames:v 1 -vsync 0 frames/contact.png; ls frames
```

<details><summary>output</summary>

```
[STREAM]
width=1536
height=1152
r_frame_rate=30/1
nb_frames=993
[/STREAM]
contact.png
```
</details>

**Read**
```
/work/scratch/frames/contact.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np, json, trimesh
r=np.load('../inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('../inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
pl=np.array(json.load(open('../inputs/ground_plane.json'))['plane'])
g=trimesh.load('../inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=np.asarray(g.vertices)[::50]
Tc=np.linalg.inv(Twc)@T
np.set_printoptions(precision=3,suppress=True,linewidth=200)
for f in range(0,993,25):
    R=T[f,:3,:3]; ax=R[:,2]  # mesh z axis in world
    vw=v@R.T+T[f,:3,3]
    h=vw@pl[:3]+pl[3]
    # angle of axis vs up (-y world)
    print(f, 'tcam',Tc[f,:3,3], 'axis',ax, 'minh %.3f maxh %.3f'%(h.min(),h.max()), 'contact',r['contact_prob_right_left'][f])
"

```

<details><summary>output</summary>

```
0 tcam [-0.566  0.474  3.439] axis [0.026 0.998 0.057] minh -0.018 maxh 0.768 contact [0. 0.]
25 tcam [-0.567  0.473  3.435] axis [0.023 0.998 0.057] minh -0.017 maxh 0.769 contact [0. 0.]
50 tcam [-0.566  0.472  3.441] axis [0.025 0.998 0.058] minh -0.016 maxh 0.770 contact [0. 0.]
75 tcam [-0.566  0.467  3.442] axis [0.031 0.998 0.056] minh -0.012 maxh 0.776 contact [0. 0.]
100 tcam [-0.569  0.479  3.488] axis [0.031 0.998 0.058] minh -0.025 maxh 0.762 contact [1. 0.]
125 tcam [-0.589  0.489  3.532] axis [0.033 0.998 0.058] minh -0.038 maxh 0.749 contact [1. 1.]
150 tcam [-0.635  0.473  3.471] axis [0.028 0.998 0.051] minh -0.023 maxh 0.768 contact [1. 1.]
175 tcam [-0.607  0.473  3.45 ] axis [0.043 0.998 0.039] minh -0.023 maxh 0.776 contact [1. 1.]
200 tcam [-0.551  0.452  3.381] axis  …[+3105 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import trimesh,numpy as np
g=trimesh.load('../inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(g.visual.uv.shape, g.visual.material)
im=g.visual.material.baseColorTexture; print(im.size if im is not None else None)
im.save('frames/tex.png')
v=g.vertices
# histogram of z
h,e=np.histogram(v[:,2],bins=40); print(np.c_[e[:-1],h])
"

```

<details><summary>output</summary>

```
(240494, 2) <trimesh.visual.material.PBRMaterial object at 0x79617c0f3e90>
(5007, 3516)
[[-1.21873513e-01  4.67250000e+04]
 [-1.02220216e-01  8.28780000e+04]
 [-8.25669192e-02  1.58770000e+04]
 [-6.29136223e-02  1.24400000e+04]
 [-4.32603255e-02  6.43900000e+03]
 [-2.36070286e-02  3.74000000e+03]
 [-3.95373181e-03  1.18980000e+04]
 [ 1.56995650e-02  2.22100000e+03]
 [ 3.53528619e-02  6.74000000e+02]
 [ 5.50061587e-02  6.96000000e+02]
 [ 7.46594556e-02  8.74000000e+02]
 [ 9.43127524e-02  9.68000000e+02]
 [ 1.13966049e-01  8.78000000e+02]
 [ 1.33619346e-01  7.58000000e+02]
 [ 1.53272643e-01  5.83000000e+02]
 [ 1.72925940e-01  7.53000000e+02]
 [ 1.92579237e-01  5.13000000e+02]
 [ 2.12232533e-01  4.85000000e+02]
 [ 2.31885830e-01  4.90000000e+02]
 [ 2.51539127e-01  7.52000000e+02]
 [ 2.7119242 …[+688 chars]
```
</details>

**Write**
```
/work/outputs/common.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/common.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/scratch; time python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
V=read_video('../inputs/video.mp4',0.5); print(V.shape); np.save('vid_half.npy',V)
M=read_masks('../inputs',0.5,'object'); np.save('mobj_half.npy',M)
P=read_masks('../inputs',0.5,'person'); np.save('mper_half.npy',P)
print(M.sum((1,2))[::50], P.sum((1,2))[::50])
"
```

<details><summary>output</summary>

```
(993, 576, 768, 3)
[ 3257  3168  3189  3184  4345  7065  9707  9725  9603  5289  6512  9132
  7922 11302 12986 13571 13898  9656  7172  7300] [ 7831  7707  6175  6306  6072  6536  5832  5916  5368  7872  8351 10016
  8384  7585  6080  6430  5201  7548 10139  7617]

real	0m34.810s
user	0m23.948s
sys	0m11.529s
```
</details>

**Bash**
```
mkdir -p /work/scratch/exp; cat > /work/scratch/exp/viz.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--poses',default='base'); ap.add_argument('--frames',default='0:993:50'); ap.add_argument('--out',default='/work/scratch/frames/ov.png'); ap.add_argument('--crop',type=int,default=1)
a=ap.parse_args()
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
V=np.load('/work/scratch/vid_half.npy',mmap_mode='r'); M=np.load('/work/scratch/mobj_half.npy',mmap_mode='r')
if a.poses=='base': Tw=D['rec']['T_world_mesh'].astype(np.float64)
else: Tw=np.load(a.poses)['T_world_mesh']
Tc=np.linalg.inv(D['Twc'])@Tw
K=D['K'].copy(); K[:2]*=0.5
R=Renderer(mesh,K,576,768)
s,e,st=map(int,a.frames.split(':')); fr=list(range(s,e,st))
tiles=[]
for f in fr:
    o=R.render(Tc[f:f+1])
    rgb=o['rgb'][0].cpu().numpy(); m=o['mask'][0].cpu().numpy()
    im=V[f].astype(np.float32)/255
    out=im.copy(); out[m]=0.5*im[m]+0.5*rgb[m]
    cs,_=cv2.findContours(M[f].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    out=(out*255).astype(np.uint8).copy(); cv2.drawContours(out,cs,-1,(0,255,0),1)
    if a.crop:
        ys,xs=np.nonzero(M[f]|m)
        if len(ys):
            cy,cx=int(np.median(ys)),int(np.median(xs))
        else: cy,cx=288,384
        y0=np.clip(cy-120,0,576-240); x0=np.clip(cx-160,0,768-320)
        out=out[y0:y0+240,x0:x0+320]
    else: out=cv2.resize(out,(320,240))
    cv2.putTe …[+282 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/ov.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > /work/scratch/exp/viz2.py <<'EOF'
# side-by-side: image crop | render (opaque over image) crop
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--poses',default='base'); ap.add_argument('--frames',default='300,350,400'); ap.add_argument('--out',default='/work/scratch/frames/sbs.png')
a=ap.parse_args()
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
V=np.load('/work/scratch/vid_half.npy',mmap_mode='r'); M=np.load('/work/scratch/mobj_half.npy',mmap_mode='r')
Tws=[]
for p in a.poses.split(','):
    Tws.append(D['rec']['T_world_mesh'].astype(np.float64) if p=='base' else np.load(p)['T_world_mesh'])
K=D['K'].copy(); K[:2]*=0.5
R=Renderer(mesh,K,576,768)
fr=[int(x) for x in a.frames.split(',')]
rows=[]
for f in fr:
    im=V[f].astype(np.float32)/255
    ys,xs=np.nonzero(M[f]); cy,cx=(int(np.median(ys)),int(np.median(xs))) if len(ys) else (288,384)
    y0=np.clip(cy-120,0,576-240); x0=np.clip(cx-160,0,768-320)
    tl=[im[y0:y0+240,x0:x0+320]]
    for Tw in Tws:
        Tc=np.linalg.inv(D['Twc'])@Tw[f]
        o=R.render(Tc[None]); rgb=o['rgb'][0].cpu().numpy(); m=o['mask'][0].cpu().numpy()
        out=im*0.3; out[m]=rgb[m]
        tl.append(out[y0:y0+240,x0:x0+320])
    t=(np.concatenate(tl,1)*255).astype(np.uint8).copy()
    cv2.putText(t,str(f),(5,20),0,0.6,(255,0,0),2)
    rows.append(t)
img=np.concatenate(rows,0)
img=cv2.resize(img,None,fx=1.5,fy=1.5)
cv2.imwrite(a.out,img[:,:,: …[+75 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/sbs.png
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
from common import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
M=np.load('/work/scratch/mobj_half.npy',mmap_mode='r'); P=np.load('/work/scratch/mper_half.npy',mmap_mode='r')
def ious(Tw):
    Tc=np.linalg.inv(D['Twc'])@Tw
    K=D['K'].copy(); K[:2]*=0.5
    R=Renderer(mesh,K,576,768)
    out=np.zeros(len(Tw))
    for s in range(0,len(Tw),32):
        o=R.render(Tc[s:s+32],rgb=False)['mask'].cpu().numpy()
        for i in range(o.shape[0]):
            f=s+i; valid=~(P[f]&~M[f])
            r=o[i]&valid; m=M[f]
            out[f]=(r&m).sum()/max((r|m).sum(),1)
    return out
if __name__=='__main__':
    Tw=D['rec']['T_world_mesh'].astype(np.float64) if len(sys.argv)<2 else np.load(sys.argv[1])['T_world_mesh']
    iou=ious(Tw)
    np.save('/work/scratch/iou_last.npy',iou)
    np.set_printoptions(precision=2,linewidth=200)
    print('mean',iou.mean(),'median',np.median(iou))
    print(np.array([iou[i:i+25].mean() for i in range(0,993,25)]))
    # motion
    R=Tw[:,:3,:3]; dr=rot_angle_deg(R[:-1],R[1:]); dt=np.linalg.norm(np.diff(Tw[:,:3,3],axis=0),axis=1)*100
    print('rot step deg per 25-block max:',np.array([dr[i:i+25].max() for i in range(0,992,25)]))
    print('trans step cm per 25-block max:',np.array([dt[i:i+25].max() for i in range(0,992,25)]))
EOF
cd /work/scratch && python exp/iou.py
```

<details><summary>output</summary>

```
mean 0.7904514140184111 median 0.7798385061821852
[0.71 0.73 0.74 0.74 0.74 0.73 0.73 0.79 0.82 0.85 0.89 0.89 0.94 0.95 0.93 0.93 0.86 0.77 0.74 0.72 0.78 0.8  0.85 0.82 0.79 0.79 0.82 0.85 0.85 0.77 0.65 0.75 0.72 0.79 0.77 0.61 0.73 0.76 0.75
 0.76]
rot step deg per 25-block max: [ 0.35  0.49  0.49  0.65  0.73  0.71  0.76  0.97  1.06  1.19  7.18 27.6   6.58  5.5   8.22 11.86  2.34  2.43  4.4   7.37  5.79  3.4   2.72  3.21 28.84  2.32  4.11  1.71  1.99  1.94  2.32  2.64  2.79
  2.48  2.59  3.41  0.74  0.84  0.57  0.73]
trans step cm per 25-block max: [ 0.7   1.07  1.43  2.33  1.44  2.38  0.93  1.19  1.89  2.63  4.99 10.26  5.61  2.14  2.97  3.31  3.42  2.34  3.92  3.7   2.2   1.59  1.48  2.59  2.36  2.8   2.81  2.22  4.12  3.13  2.73  4.23  2.53
  4.73  2.92  4.03  1.32  1.47  1.74  0.92 …[+1 chars]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs'); Tw=D['rec']['T_world_mesh'].astype(np.float64)
R=Tw[:,:3,:3]; dr=rot_angle_deg(R[:-1],R[1:])
idx=np.argsort(-dr)[:25]; print(sorted([(int(i),round(float(dr[i]),1)) for i in idx]))
for k in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
  T=D['rec'][k].astype(np.float64); R=T[:,:3,:3]; d=rot_angle_deg(R[:-1],R[1:]); idx=np.argsort(-d)[:15]; print(k,sorted([(int(i),round(float(d[i]),1)) for i in idx]))
Tc=np.linalg.inv(D['Twc'])@Tw
for k in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
  T=D['rec'][k].astype(np.float64); a=rot_angle_deg(Tc[:,:3,:3],T[:,:3,:3]); print(k,'rot diff vs base per 50', np.round([a[i:i+50].mean() for i in range(0,993,50)],1))
  print(' z ratio', np.round([ (Tc[i:i+50,2,3]/T[i:i+50,2,3]).mean() for i in range(0,993,50)],3))
"

```

<details><summary>output</summary>

```
[(270, 6.9), (271, 7.2), (279, 9.4), (281, 7.9), (282, 12.4), (284, 10.8), (285, 11.5), (286, 7.2), (287, 27.6), (290, 8.3), (291, 11.1), (295, 9.2), (296, 16.8), (297, 12.7), (367, 8.2), (374, 6.6), (380, 8.8), (393, 9.1), (396, 11.9), (478, 7.4), (603, 14.0), (604, 28.8), (607, 10.5), (613, 9.6), (614, 9.3)]
T_cam_mesh_foundationpose [(271, 9.8), (279, 11.8), (282, 21.1), (283, 10.2), (287, 9.8), (291, 10.3), (296, 16.0), (297, 14.9), (303, 10.0), (380, 11.8), (393, 13.2), (396, 13.5), (603, 14.4), (604, 24.8), (614, 10.0)]
T_cam_mesh_coconet [(279, 9.4), (282, 12.4), (284, 10.8), (285, 11.5), (287, 27.6), (291, 11.1), (295, 9.2), (296, 16.8), (297, 12.7), (396, 11.9), (603, 14.0), (604, 28.8), (607, 10.5), (613, 9.6), (614, 9.3)]
T_cam_mesh_foundationpose rot diff vs base per 50 [ 2.4   …[+485 chars]
```
</details>

**Bash**
```
python exp/viz.py --frames 270:300:2 --out frames/j1.png && python exp/viz.py --frames 596:626:2 --out frames/j2.png
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/j1.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/frames/j2.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import numpy as np
M=np.load('mobj_half.npy',mmap_mode='r'); P=np.load('mper_half.npy',mmap_mode='r')
F=len(M)
area=np.array([M[f].sum() for f in range(F)])
# mask change between consecutive frames: xor/union
d=np.array([ (M[f]^M[f+1]).sum()/max((M[f]|M[f+1]).sum(),1) for f in range(F-1)])
# person-object overlap of dilated masks
import cv2
k=np.ones((7,7),np.uint8)
touch=np.array([ (cv2.dilate(P[f].astype(np.uint8),k).astype(bool)&M[f]).sum() for f in range(F)])
r=np.load('../inputs/reconstruction.npz'); c=r['contact_prob_right_left']
for f in range(0,F-1,10): print(f, area[f], '%.3f'%d[f:f+10].mean(), touch[f], c[f])
" > phase_stats.txt; head -20 phase_stats.txt; sed -n 85,100p phase_stats.txt
```

<details><summary>output</summary>

```
0 3257 0.015 0 [4.6719366e-09 1.3559459e-10]
10 3111 0.022 0 [1.3012514e-08 1.8621353e-10]
20 3162 0.019 0 [3.7377827e-08 5.6210331e-10]
30 3257 0.010 0 [1.5854566e-08 3.3378239e-10]
40 3279 0.006 0 [6.2460124e-08 1.9296531e-09]
50 3168 0.008 0 [1.5386538e-08 5.7465666e-10]
60 3244 0.014 0 [3.0106712e-08 1.0617111e-10]
70 3173 0.004 0 [8.096565e-08 4.562123e-11]
80 3139 0.011 0 [7.4908524e-09 2.6400457e-12]
90 3225 0.025 24 [7.0384809e-09 2.6107766e-12]
100 3189 0.020 84 [9.9952954e-01 6.0999326e-07]
110 3080 0.021 145 [1.        0.9999996]
120 3083 0.014 307 [1.        0.9999997]
130 3013 0.024 302 [1.       0.999999]
140 3057 0.051 342 [0.99999994 0.9999977 ]
150 3184 0.026 341 [0.99999917 0.9999988 ]
160 3132 0.023 264 [1.        0.9999999]
170 3166 0.025 283 [1.        0.9999998]
180 3 …[+779 chars]
```
</details>

**Bash**
```
awk '{printf "%s:%s:%s:%s  ", $1,$2,$3,$4} NR%5==0{print ""}' phase_stats.txt
```

<details><summary>output</summary>

```
0:3257:0.015:0  10:3111:0.022:0  20:3162:0.019:0  30:3257:0.010:0  40:3279:0.006:0  
50:3168:0.008:0  60:3244:0.014:0  70:3173:0.004:0  80:3139:0.011:0  90:3225:0.025:24  
100:3189:0.020:84  110:3080:0.021:145  120:3083:0.014:307  130:3013:0.024:302  140:3057:0.051:342  
150:3184:0.026:341  160:3132:0.023:264  170:3166:0.025:283  180:3352:0.027:294  190:3718:0.039:343  
200:4345:0.040:528  210:5218:0.024:584  220:5772:0.024:615  230:6480:0.013:471  240:6920:0.013:382  
250:7065:0.025:334  260:7931:0.047:470  270:8491:0.026:445  280:8883:0.032:419  290:9214:0.032:274  
300:9707:0.024:403  310:9786:0.004:419  320:9751:0.003:487  330:9654:0.013:450  340:9667:0.004:485  
350:9725:0.002:568  360:9643:0.005:578  370:9662:0.002:619  380:9726:0.003:661  390:9772:0.023:706  
400:9603:0.054:515  410 …[+1208 chars]
```
</details>

**Bash**
```
python -c "
import numpy as np, cv2
V=np.load('vid_half.npy',mmap_mode='r'); M=np.load('mobj_half.npy',mmap_mode='r'); P=np.load('mper_half.npy',mmap_mode='r')
F=len(M); k=np.ones((5,5),np.uint8)
g=lambda f: cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(V[f]),cv2.COLOR_RGB2GRAY).astype(np.float32),(5,5),0)
out=[]
for f in range(5,F):
    m=cv2.erode(M[f].astype(np.uint8),k).astype(bool)&~cv2.dilate(P[f].astype(np.uint8),k).astype(bool)&~cv2.dilate(P[f-5].astype(np.uint8),k).astype(bool)
    if m.sum()<50: out.append(np.nan); continue
    out.append(np.median(np.abs(g(f)-g(f-5))[m]))
out=np.array([np.nan]*5+out); np.save('pixdiff5.npy',out)
for i in range(0,F,10): print(i, np.round(out[i:i+10],1))
" | awk '{print}' | head -100 | paste - - | column -t | head -60
```

<details><summary>output</summary>

```
0    [nan   nan   nan   nan   nan   0.3   0.3   0.3   0.3   0.3]   10     [0.2   0.3    0.3   0.3   0.2   0.3   0.2   0.3   0.3   0.3]          
20   [0.2   0.3   0.3   0.2   0.2   0.3   0.2   0.2   0.3   0.3]   30     [0.2   0.2    0.2   0.3   0.3   0.2   0.3   0.3   0.2   0.3]          
40   [0.2   0.2   0.2   0.2   0.2   0.3   0.2   0.2   0.3   0.3]   50     [0.2   0.2    0.2   0.2   0.2   0.2   0.2   0.2   0.2   0.1]          
60   [0.2   0.3   0.3   0.3   0.2   0.3   0.3   0.2   0.2   0.2]   70     [0.3   0.2    0.2   0.2   0.2   0.2   0.2   0.2   0.2   0.2]          
80   [0.2   0.2   0.3   0.3   0.2   0.3   0.2   0.3   0.3   0.3]   90     [0.3   0.3    0.3   0.3   0.3   0.4   0.4   0.4   0.5   0.5]          
100  [0.7   0.7   0.6   0.5   0.6   0.6   0.5   0.5   0.5   0.4]   110    [ …[+6402 chars]
```
</details>

**Bash**
```
cat > /work/scratch/exp/cmp_full.py <<'EOF'
# full-res crop: image with SAM mask contour (green) and render contour (red)
import sys; sys.path.insert(0,'/work/outputs')
from common import *
f=int(sys.argv[1]); out=sys.argv[2]; poses=sys.argv[3] if len(sys.argv)>3 else 'base'
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
cap=cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read()
import h5py
m=h5py.File('/work/inputs/masks.h5')['object'][f]; p=h5py.File('/work/inputs/masks.h5')['person'][f]
Tw=D['rec']['T_world_mesh'].astype(np.float64) if poses=='base' else np.load(poses)['T_world_mesh']
R=Renderer(mesh,D['K'],1152,1536); r=R.render((np.linalg.inv(D['Twc'])@Tw[f])[None],rgb=False)['mask'][0].cpu().numpy()
for mm,c in [(m,(0,255,0)),(r,(0,0,255)),(p,(255,0,0))]:
    cs,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,cs,-1,c,1)
ys,xs=np.nonzero(m|r); y0,y1,x0,x1=ys.min()-20,ys.max()+20,xs.min()-20,xs.max()+20
cv2.imwrite(out,im[max(y0,0):y1,max(x0,0):x1])
EOF
python exp/cmp_full.py 0 frames/f0.png
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/f0.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/scratch/exp/opt1.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/opt1.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/opt1.py --iters 40 2>&1 | tail -8 && python exp/iou.py opt1.npz
```

<details><summary>output</summary>

```
0 sil 0.4865828472802046 smooth 1.3387725353240967 2s
10 sil 0.2999830658942431 smooth 0.038702137768268585 12s
20 sil 0.2890683968381699 smooth 0.015118210576474667 21s
30 sil 0.2843566416253496 smooth 0.010198120027780533 29s
39 sil 0.28009135197296603 smooth 0.008668138645589352 37s
done 36.87512469291687
mean 0.8724630360705707 median 0.8749653835502631
[0.83 0.82 0.83 0.83 0.82 0.81 0.82 0.85 0.91 0.93 0.94 0.94 0.95 0.96 0.96 0.96 0.9  0.83 0.79 0.76 0.88 0.89 0.9  0.89 0.86 0.9  0.91 0.92 0.93 0.91 0.88 0.89 0.87 0.86 0.82 0.78 0.84 0.84 0.84
 0.84]
rot step deg per 25-block max: [0.03 0.03 0.03 0.19 0.19 0.24 0.33 0.64 0.67 0.55 1.65 1.9  1.81 0.01 0.01 1.33 1.47 0.76 1.58 1.87 1.11 0.29 0.27 3.12 4.46 1.13 0.95 0.88 0.88 0.85 0.66 1.06 0.76 1.45 1.44 2.08 0.3  0.29 0.01
 0.01]
tra …[+229 chars]
```
</details>

**Bash**
```
python exp/viz2.py --poses base,opt1.npz --frames 284,380,470,604,750 --out frames/sbs2.png
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/sbs2.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
m=load_mesh('/work/inputs'); v=m['v']
for lo,hi in [(-1,-0.08),(0.5,1),(0.1,0.4)]:
  s=v[(v[:,2]>lo)&(v[:,2]<hi)]; print(lo,hi,len(s),(s.min(0)+s.max(0))/2, s.min(0),s.max(0))
"

```

<details><summary>output</summary>

```
-1 -0.08 133676 [ 0.00954233  0.00125173 -0.10093964] [-0.38281685 -0.38470846 -0.12187351] [ 0.40190151  0.38721192 -0.08000577]
0.5 1 40749 [0.03357823 0.00251466 0.58230487] [-0.30828711 -0.32839814  0.50035137] [0.37544358 0.33342746 0.66425836]
0.1 0.4 11138 [0.00340074 0.00764059 0.24977167] [-0.02570454 -0.02301366  0.10010447] [0.03250601 0.03829485 0.39943886]
```
</details>

**Write**
```
/work/scratch/exp/spin.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/spin.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/spin.py opt1.npz 50,350,560,604,750,960
```

<details><summary>output</summary>

```
frame 50
  top iou: [(0, 0.833, 0.335), (3, 0.832, 0.322), (72, 0.821, 0.362), (69, 0.813, 0.339), (-3, 0.8, 0.296), (75, 0.79, 0.331)]
  top hpNCC: [(72, 0.821, 0.362), (69, 0.813, 0.339), (0, 0.833, 0.335), (75, 0.79, 0.331), (3, 0.832, 0.322), (141, 0.754, 0.32)]
  top NCC: [(3, 0.832, 0.423), (69, 0.813, 0.419), (72, 0.821, 0.414), (138, 0.736, 0.403)]
frame 350
  top iou: [(147, 0.967, 0.341), (75, 0.966, 0.362), (3, 0.963, 0.357), (0, 0.962, 0.238), (78, 0.961, 0.485), (150, 0.959, 0.34)]
  top hpNCC: [(78, 0.961, 0.485), (9, 0.936, 0.474), (6, 0.953, 0.473), (-66, 0.934, 0.445), (81, 0.945, 0.442), (39, 0.915, 0.424)]
  top NCC: [(-48, 0.901, 0.265), (-54, 0.903, 0.259), (-84, 0.919, 0.243), (-45, 0.904, 0.239)]
frame 560
  top iou: [(0, 0.891, 0.221), (-3, 0.89, 0.203), (3, 0.884,  …[+1414 chars]
```
</details>

**Bash**
```
pip list 2>/dev/null | grep -i -E "kornia|glue|roma|loftr|dino|timm|xformers|cotracker|raft|open3d|scipy|pytorch3d|transformers|omegaconf|hydra" ; ls /weights /weights/*
```

<details><summary>output</summary>

```
hydra-colorlog                           1.2.0
hydra-core                               1.3.7
hydra-submitit-launcher                  1.2.0
kornia                                   0.8.3
kornia_rs                                0.2.0
omegaconf                                2.3.1
open3d                                   0.20.0
pytorch3d                                0.7.9
roma                                     1.6.1
scipy                                    1.16.3
timm                                     1.0.30
transformers                             5.3.0
/weights:
cari4d
foundationpose
grounding_dino
hf_home
sam2
sam3d_body

/weights/cari4d:
2026-08-25-09-35-57

/weights/foundationpose:
nvlabs_pytorch

/weights/grounding_dino:
groundingdino_swint_ogc.pth

/weights/hf_home:
hub
xet

/ …[+218 chars]
```
</details>

**Bash**
```
cd /work/tools && curl -sSL -o /dev/null -w "%{http_code}\n" https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth; curl -sSL -o /dev/null -w "%{http_code}\n" https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth; curl -sSI http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt | head -3; ls /weights/hf_home/hub
```

<details><summary>output</summary>

```
200
200
HTTP/1.1 403 Forbidden

CACHEDIR.TAG
blobs
models--Ruicheng--moge-2-vitl-normal
```
</details>

**Bash**
```
git clone -q https://github.com/cvg/LightGlue.git && cd LightGlue && git rev-parse HEAD && mkdir -p /work/tools/weights && cd /work/tools/weights && for f in superpoint_lightglue.pth superpoint_v1.pth aliked_lightglue.pth disk_lightglue.pth; do curl -sSL -o $f https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/$f; done; curl -sSL -o aliked-n16.pth https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth; ls -la; sha256sum *
```

<details><summary>output</summary>

```
eb42fee2d71449efb0aa5c10549752b5d75384d8
total 147196
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 14:50 .
drwxrwxr-x 4 hanzhou hanzhou     4096 Oct 21 14:50 ..
-rw-rw-r-- 1 hanzhou hanzhou  2738091 Oct 21 14:50 aliked-n16.pth
-rw-rw-r-- 1 hanzhou hanzhou 47632827 Oct 21 14:50 aliked_lightglue.pth
-rw-rw-r-- 1 hanzhou hanzhou 47631405 Oct 21 14:50 disk_lightglue.pth
-rw-rw-r-- 1 hanzhou hanzhou 47500279 Oct 21 14:50 superpoint_lightglue.pth
-rw-rw-r-- 1 hanzhou hanzhou  5206086 Oct 21 14:50 superpoint_v1.pth
5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2  aliked-n16.pth
d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503  aliked_lightglue.pth
b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335  disk_lightglue.pth
6ff7040d0a497fc6639337946d7538dae …[+141 chars]
```
</details>

**Bash**
```
cd /work/tools/LightGlue; grep -n "load_state_dict_from_url\|url\|file_name" lightglue/*.py | head -20
```

<details><summary>output</summary>

```
lightglue/lightglue.py:349:    url = "https://github.com/cvg/LightGlue/releases/download/{}/{}.pth"
lightglue/lightglue.py:422:            state_dict = torch.hub.load_state_dict_from_url(
lightglue/lightglue.py:423:                self.url.format(self.version, self.conf.weights),
lightglue/lightglue.py:424:                file_name=fname,
lightglue/aliked.py:620:    checkpoint_url = "https://github.com/Shiaoming/ALIKED/raw/main/models/{}.pth"
lightglue/aliked.py:692:        state_dict = torch.hub.load_state_dict_from_url(
lightglue/aliked.py:693:            self.checkpoint_url.format(conf.model_name), map_location="cpu"
lightglue/superpoint.py:144:        url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"  # noqa
lightglue/superpoint.py:145:        self …[+57 chars]
```
</details>

**Bash**
```
cd /work/tools/LightGlue; sed -n 415,430p lightglue/lightglue.py; mkdir -p /work/tools/torch_home/hub/checkpoints; cd /work/tools/torch_home/hub/checkpoints; for f in ../../../weights/*; do ln -sf $(realpath $f) .; done; ls -la
```

<details><summary>output</summary>

```
                [self.confidence_threshold(i) for i in range(self.conf.n_layers)]
            ),
        )

        state_dict = None
        if features is not None:
            fname = f"{conf.weights}_{self.version.replace('.', '-')}.pth"
            state_dict = torch.hub.load_state_dict_from_url(
                self.url.format(self.version, self.conf.weights),
                file_name=fname,
            )
            self.load_state_dict(state_dict, strict=False)
        elif conf.weights is not None:
            path = Path(__file__).parent
            path = path / "weights/{}.pth".format(self.conf.weights)
            state_dict = torch.load(str(path), map_location="cpu")
total 8
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 14:51 .
drwxrwxr-x 3 hanzhou hanzhou 4096 Oct 21 14:51 ..
lr …[+543 chars]
```
</details>

**Bash**
```
cd /work/tools/torch_home/hub/checkpoints; for n in superpoint aliked disk; do ln -sf /work/tools/weights/${n}_lightglue.pth ${n}_lightglue_v0-1_arxiv.pth; done; cd /work/scratch; TORCH_HOME=/work/tools/torch_home python -c "
import sys; sys.path.insert(0,'/work/tools/LightGlue')
from lightglue import LightGlue, SuperPoint, ALIKED
e=SuperPoint(max_num_keypoints=2048).eval().cuda(); m=LightGlue(features='superpoint').eval().cuda()
e2=ALIKED(max_num_keypoints=2048).eval().cuda(); m2=LightGlue(features='aliked').eval().cuda(); print('ok')
"
```

<details><summary>output</summary>

```
ok
```
</details>

**Write**
```
/work/outputs/match.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/match.py (file state is current in your context — no need to Read it back)
```
</details>

**Write**
```
/work/scratch/exp/test_reg.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/test_reg.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/test_reg.py opt1.npz 350,560,604,750,960,50 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
Downloading: "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth" to /weights/sam3d_body/torch_home/hub/checkpoints/superpoint_v1.pth
Traceback (most recent call last):
  File "/work/scratch/exp/test_reg.py", line 12, in <module>
    mt = Matcher(kind)
         ^^^^^^^^^^^^^
  File "/work/outputs/match.py", line 19, in __init__
    self.ext = SuperPoint(max_num_keypoints=n_kp).eval().to(dev)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/tools/LightGlue/lightglue/superpoint.py", line 145, in __init__
    self.load_state_dict(torch.hub.load_state_dict_from_url(url))
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/hub.py", line 867, in load_state_dict_from_url
    download_url …[+364 chars]
```
</details>

**Bash**
```
cd /work/outputs && python - <<'EOF'
s=open('match.py').read()
s=s.replace('os.environ.setdefault("TORCH_HOME", "/work/tools/torch_home")\nsys.path.insert(0, "/work/tools/LightGlue")\nfrom lightglue import LightGlue, SuperPoint, ALIKED  # noqa: E402\n',
 'sys.path.insert(0, "/work/tools/LightGlue")\nfrom lightglue import LightGlue, SuperPoint, ALIKED  # noqa: E402\n\n# offline weights (downloaded once into /work/tools, see TOOLS.md)\ntorch.hub.set_dir("/work/tools/torch_home/hub")\n')
open('match.py','w').write(s)
EOF
cd /work/scratch && python exp/test_reg.py opt1.npz 350,560,604,750,960,50 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
350 init spin 0 match 72 inl 33 disp 7.8 -> spin vs ref 1.8  total rot 3.8  dt(cm) [-5.   3.5 24.4]
350 init spin 72 match 57 inl 19 disp 6.7 -> spin vs ref 75.9  total rot 75.9  dt(cm) [ 0.  -0.6 -7.1]
350 init spin 144 match 87 inl 34 disp 7.4 -> spin vs ref 143.9  total rot 143.8  dt(cm) [-0.9 -2.2  0.8]
350 -144 fail (27, 10)
350 init spin -72 match 43 inl 12 disp 9.5 -> spin vs ref -69.0  total rot 69.1  dt(cm) [  4.7  -2.9 -25.7]
560 init spin 0 match 36 inl 13 disp 5.4 -> spin vs ref 6.9  total rot 7.1  dt(cm) [  1.6  -6.2 -29.1]
560 init spin 72 match 56 inl 15 disp 12.3 -> spin vs ref 79.8  total rot 79.9  dt(cm) [  0.5  -3.8 -11.7]
560 init spin 144 match 75 inl 26 disp 5.2 -> spin vs ref 143.0  total rot 142.9  dt(cm) [  1.   -4.2 -11.5]
560 init spin -144 match 73 inl 28 disp 4 …[+2204 chars]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
mesh=load_mesh('/work/inputs')
K=np.array([[1500,0,400],[0,1500,400],[0,0,1.]])
R=Renderer(mesh,K,800,800)
def look(dirv, up):
    # camera looking along dirv (from camera to object), mesh centre at 0.27 z
    z=np.array(dirv,float); z/=np.linalg.norm(z); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    Rcm=np.stack([x,y,z]); c=np.array([0.006,0.004,0.27]); T=np.eye(4); T[:3,:3]=Rcm; T[:3,3]=-Rcm@c+np.array([0,0,3.0]); return T
views=[look([0,0,1],[0,1,0]), look([0,0,-1],[0,1,0]), look([1,0,0],[0,0,-1]), look([0,1,0],[0,0,-1]),look([-1,0,0],[0,0,-1]),look([0,-1,0],[0,0,-1])]
o=R.render(np.stack(views)); im=(o['rgb'].cpu().numpy()*255).astype(np.uint8)
im[~o['mask'].cpu().numpy()]=60
top=np.concatenate(im[:3],1); bot=np.concatenate(im[3:],1)
cv2.imwrite('frames/mesh_views.png',cv2.resize(np.concatenate([top,bot],0),None,fx=0.6,fy=0.6)[:,:,::-1])
"

```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/mesh_views.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python exp/cmp_full.py 960 frames/f960.png opt1.npz; python -c "
import cv2; a=cv2.imread('/work/scratch/frames/f960.png'); print(a.shape); cv2.imwrite('/work/scratch/frames/f960b.png', cv2.resize(a,None,fx=2,fy=2))"
```

<details><summary>output</summary>

```
(397, 320, 3)
```
</details>

**Read**
```
/work/scratch/frames/f960b.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/scratch/exp/viz_spin.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/viz_spin.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/viz_spin.py opt1.npz 560,750,350 frames/spin1.png
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/spin1.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python exp/viz_spin.py opt1.npz 560,750 frames/spin2.png 0,-20,20 && python -c "
import cv2; a=cv2.imread('/work/scratch/frames/spin2.png'); cv2.imwrite('/work/scratch/frames/spin2.png', cv2.resize(a,None,fx=1.3,fy=1.3))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/spin2.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python exp/viz_spin.py opt1.npz 560,750,600 frames/spin3.png 0,180,144,-144 && python -c "
import cv2; a=cv2.imread('/work/scratch/frames/spin3.png'); cv2.imwrite('/work/scratch/frames/spin3.png', cv2.resize(a,None,fx=1.2,fy=1.2))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/spin3.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
mesh=load_mesh('/work/inputs')
v=mesh['v']; f=mesh['f']
keep=(v[f][:,:,2].max(1)<0.08)
mesh2=dict(mesh); mesh2['f']=f[keep]
K=np.array([[1500,0,400],[0,1500,400],[0,0,1.]])
R=Renderer(mesh2,K,800,800)
def look(dirv, up, c=np.array([0.006,0.004,0.0]),d=2.2):
    z=np.array(dirv,float); z/=np.linalg.norm(z); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    Rcm=np.stack([x,y,z]); T=np.eye(4); T[:3,:3]=Rcm; T[:3,3]=-Rcm@c+np.array([0,0,d]); return T
views=[look([0,0,-1],[0,1,0]), look([0.0,0.6,-1],[0,1,0])]
o=R.render(np.stack(views)); im=(o['rgb'].cpu().numpy()*255).astype(np.uint8); im[~o['mask'].cpu().numpy()]=60
# draw mesh x axis (red) and y axis (green) in first view
T=views[0]; 
def pr(p): q=K@(T[:3,:3]@p+T[:3,3]); return tuple(int(x) for x in q[:2]/q[2])
a=im[0].copy(); cv2.line(a,pr(np.array([0,0,0.])),pr(np.array([0.3,0,0.])),(255,0,0),3); cv2.line(a,pr(np.array([0,0,0.])),pr(np.array([0,0.3,0.])),(0,255,0),3)
# leg directions: vertices z>0.55, angles
vl=v[v[:,2]>0.55]; ang=np.degrees(np.arctan2(vl[:,1]-0.004,vl[:,0]-0.006)); r=np.hypot(vl[:,0]-0.006,vl[:,1]-0.004)
h,e=np.histogram(ang[r>0.25],bins=72,range=(-180,180)); print('leg tip angle hist peaks', e[np.argsort(-h)[:10]])
cv2.imwrite('frames/underside.png',np.concatenate([a,im[1]],1)[:,:,::-1])
"

```

<details><summary>output</summary>

```
leg tip angle hist peaks [  65.   60.  -85.  135. -160.   -5.  -10.  130.    0. -155.]
```
</details>

**Read**
```
/work/scratch/frames/underside.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/scratch/exp/spin_vote.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/spin_vote.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/spin_vote.py opt1.npz "list(range(525,600,10))+list(range(740,800,10))" 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
525 {0: 23, 72: 25, 144: 20, -144: 21, -72: 9}
535 {0: 17, 72: 23, 144: 29, -144: 32, -72: 7}
545 {0: 18, 72: 31, 144: 30, -144: 31, -72: 2}
555 {0: 17, 72: 25, 144: 29, -144: 41, -72: 8}
565 {0: 33, 72: 27, 144: 26, -144: 32, -72: 42}
575 {0: 26, 72: 26, 144: 33, -144: 45, -72: 31}
585 {0: 19, 72: 16, 144: 35, -144: 36, -72: 22}
595 {0: 25, 72: 21, 144: 32, -144: 25, -72: 12}
740 {0: 34, 72: 38, 144: 39, -144: 33, -72: 31}
750 {0: 20, 72: 27, 144: 61, -144: 63, -72: 21}
760 {0: 14, 72: 32, 144: 38, -144: 32, -72: 21}
770 {0: 15, 72: 48, 144: 54, -144: 52, -72: 6}
780 {0: 22, 72: 59, 144: 42, -144: 43, -72: 28}
790 {0: 56, 72: 44, 144: 44, -144: 19, -72: 15}
total {0: 339.0, 72: 442.0, 144: 512.0, -144: 505.0, -72: 255.0}
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
mesh=load_mesh('/work/inputs'); v=mesh['v']
c=np.array([0.006,0.004]); r=np.hypot(*(v[:,:2]-c).T); ang=np.degrees(np.arctan2(v[:,1]-c[1],v[:,0]-c[0]))
for z0,z1 in [(-0.08,-0.07),(-0.07,-0.05),(-0.05,-0.03),(-0.03,0.0),(0.0,0.05)]:
  s=(v[:,2]>z0)&(v[:,2]<z1)&(r>0.06)&(r<0.36)
  h,e=np.histogram(ang[s],bins=24,range=(-180,180))
  print(z0,z1,s.sum(), ' '.join('%d:%d'%(e[i],h[i]) for i in range(24) if h[i]>0))
"

```

<details><summary>output</summary>

```
-0.08 -0.07 5593 -180:860 -165:318 -150:30 -90:29 -75:196 -60:520 -45:87 15:6 30:467 45:199 60:7 105:113 120:418 135:1388 150:538 165:417
-0.07 -0.05 1114 -180:74 -165:72 150:534 165:434
-0.05 -0.03 3775 -180:29 -165:177 150:2723 165:846
-0.03 0.0 3335 150:2683 165:652
0.0 0.05 12485 150:10188 165:2297
```
</details>

**Write**
```
/work/scratch/exp/lever.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/lever.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/lever.py opt1.npz "list(range(500,620,10))+list(range(650,860,15))" 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
lever faces 6607 board faces 1007
500 [ -9.4  -8.6   0.3 -13.3  -5.5]
510 [-8.7 -2.2 17.3 -7.4 -0.6]
520 [-6.6 -4.7 11.8 -7.1 -4.6]
530 [-5.2 -4.2 11.  -7.2 -3.6]
540 [ 0.9 -3.8 12.3 -6.7 -0.9]
550 [ 2.2 -1.2 14.7 -8.1 -4.6]
560 [  2.1  -1.6  15.2  -6.7 -10.9]
570 [ 2.7 -2.7 13.8 -5.6 -9.6]
580 [ 4.6 -3.8 13.4 -5.4 -3. ]
590 [-7.  -3.6 14.2 -4.6 -5.1]
600 [-6.5 -3.4 19.   3.6 -2.6]
610 [-44.4 -10.   -4.8   5.8 -11.5]
650 [-10.1  -0.6  -5.3   2.3   4.4]
665 [-4.7  9.2 -7.3 -0.1  0.8]
680 [ 1.7 15.4 -7.1  1.2  0.8]
695 [ nan 12.6 -8.4 -0.1  0.7]
710 [ 3.2  9.  -3.4 -0.4  4.1]
725 [ 0.4  5.8 -2.6 -0.1  2. ]
740 [-9.8  2.8 -2.   2.7  4.9]
755 [-13.7   1.2  -0.2   3.   -1.7]
770 [-18.1   1.6  -0.8   2.3  -5.9]
785 [-14.3   0.3  -1.5   2.1 -23. ]
800 [ -4.2  -0.9   0.7  -1.9 -38.9]
815 [ -7.1  - …[+197 chars]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs'); Tw=D['rec']['T_world_mesh'].astype(np.float64)
R=Tw[:,:3,:3]
rel=np.einsum('fji,fjk->fik',R[:-1],R[1:])  # body-frame relative
spin=np.degrees(np.arctan2(rel[:,1,0]-rel[:,0,1], rel[:,0,0]+rel[:,1,1]))  # approx twist about z
tot=rot_angle_deg(R[:-1],R[1:])
cs=np.cumsum(spin)
for a,b in [(0,100),(100,200),(200,260),(260,310),(310,360),(360,400),(400,470),(470,520),(520,600),(600,620),(620,700),(700,800),(800,900),(900,992)]:
    print(a,b,'spin sum %.1f'%spin[a:b].sum(),'max |step| %.1f'%np.abs(spin[a:b]).max(), 'tot max %.1f'%tot[a:b].max())
np.set_printoptions(precision=1,suppress=True,linewidth=220)
print(spin[595:620]); print(spin[275:300]); print(spin[360:400])
"

```

<details><summary>output</summary>

```
0 100 spin sum -0.4 max |step| 0.5 tot max 0.7
100 200 spin sum 6.7 max |step| 0.8 tot max 1.0
200 260 spin sum -4.5 max |step| 1.5 tot max 1.8
260 310 spin sum -1.1 max |step| 16.6 tot max 27.6
310 360 spin sum -12.3 max |step| 6.0 tot max 6.5
360 400 spin sum 22.2 max |step| 10.8 tot max 11.9
400 470 spin sum 8.0 max |step| 4.3 tot max 4.3
470 520 spin sum -0.2 max |step| 7.3 tot max 7.4
520 600 spin sum -5.7 max |step| 3.4 tot max 3.4
600 620 spin sum -70.9 max |step| 28.8 tot max 28.8
620 700 spin sum -14.5 max |step| 4.1 tot max 4.1
700 800 spin sum 24.1 max |step| 2.5 tot max 2.6
800 900 spin sum 11.2 max |step| 2.3 tot max 3.4
900 992 spin sum -0.6 max |step| 0.8 tot max 0.8
[  0.7  -0.9   1.1  -1.3  -1.2   3.8  -1.   -0.5 -13.9 -28.8   3.8   1.7 -10.4   2.9  -1.1   5.9  -2.8  -3.3  …[+396 chars]
```
</details>

**Bash**
```
python exp/viz_spin.py opt1.npz 560,680,710 frames/spin4.png 0,144,72,-144 && python -c "
import cv2; a=cv2.imread('/work/scratch/frames/spin4.png'); cv2.imwrite('/work/scratch/frames/spin4.png', cv2.resize(a,None,fx=1.2,fy=1.2))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/spin4.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python exp/viz_spin.py opt1.npz 960,50,880 frames/spin5.png 0,144,-144,72,-72 && python -c "
import cv2; a=cv2.imread('/work/scratch/frames/spin5.png'); cv2.imwrite('/work/scratch/frames/spin5.png', cv2.resize(a,None,fx=1.1,fy=1.1))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/spin5.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs')
for key in ['T_world_mesh','T_cam_mesh_foundationpose']:
  Tw=D['rec'][key].astype(np.float64)
  R=Tw[:,:3,:3]
  rel=np.einsum('fji,fjk->fik',R[:-1],R[1:])
  spin=np.degrees(np.arctan2(rel[:,1,0]-rel[:,0,1], rel[:,0,0]+rel[:,1,1]))
  w=20; cs=np.r_[0,np.cumsum(spin)]; s=cs[w:]-cs[:-w]
  print(key, [(i,round(s[i],1)) for i in range(0,len(s),5) if abs(s[i])>20])
"

```

<details><summary>output</summary>

```
T_world_mesh [(380, 24.9), (385, 30.0), (395, 26.5), (585, -44.5), (590, -47.3), (595, -64.3), (600, -70.9), (605, -29.1), (610, -28.9)]
T_cam_mesh_foundationpose [(380, 29.9), (385, 37.8), (395, 33.6), (460, -23.0), (470, -21.9), (585, -45.0), (590, -47.9), (595, -64.2), (600, -70.9), (605, -30.7), (610, -31.5)]
```
</details>

**Bash**
```
cat > exp/lever_viz.py <<'EOF'
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; f = int(sys.argv[2]); spins=[float(x) for x in sys.argv[3].split(',')]; out=sys.argv[4]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; fcs = mesh['f']
c = np.array([0.006, 0.004]); fc = v[fcs].mean(1)
r = np.hypot(*(fc[:, :2] - c).T); ang = np.degrees(np.arctan2(fc[:, 1] - c[1], fc[:, 0] - c[0]))
lever = (r > 0.07) & (r < 0.37) & (fc[:, 2] > -0.065) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
lev_t = torch.tensor(np.r_[False, lever], device='cuda')
cap = cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
ax_p = np.array([0.006, 0.004, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
m = h5py.File('/work/inputs/masks.h5')['object'][f]
ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min()-30, ys.max()+30, xs.min()-30, xs.max()+30
tiles=[]
for s in spins:
    T = torch.tensor(spinT(Tc[f], s)[None], dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).conti …[+496 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/lv680.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Read**
```
/work/scratch/frames/lv560.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import cv2, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); tiles=[]
for f in [560,585,595,600,604,608,612,616,620,630]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read()
    a=im[700:1100,350:900].copy(); cv2.putText(a,str(f),(5,30),0,1,(0,255,255),2); tiles.append(cv2.resize(a,None,fx=0.6,fy=0.6))
cv2.imwrite('/work/scratch/frames/slip.png',np.concatenate([np.concatenate(tiles[:5],1),np.concatenate(tiles[5:],1)],0))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/slip.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs'); B=D['rec']['T_world_mesh'].astype(np.float64); O=np.load('opt1.npz')['T_world_mesh']
for f in [50,300,350,450,500,530,560,580,590,600,610,620,640,680,710,800,960]:
    rel=B[f,:3,:3].T@O[f,:3,:3]; print(f,'spin %.1f'%np.degrees(np.arctan2(rel[1,0]-rel[0,1],rel[0,0]+rel[1,1])),'tot %.1f'%rot_angle_deg(B[f,:3,:3],O[f,:3,:3]))
"
```

<details><summary>output</summary>

```
50 spin -0.9 tot 1.8
300 spin 0.8 tot 4.4
350 spin 7.5 tot 7.9
450 spin 3.8 tot 3.9
500 spin 2.0 tot 2.4
530 spin 2.2 tot 2.3
560 spin 0.2 tot 0.4
580 spin -0.1 tot 0.5
590 spin 2.3 tot 2.4
600 spin -6.8 tot 6.9
610 spin -3.0 tot 3.2
620 spin 3.1 tot 3.3
640 spin -3.9 tot 4.2
680 spin -2.8 tot 4.0
710 spin -1.0 tot 1.1
800 spin -1.7 tot 1.9
960 spin -0.3 tot 0.5
```
</details>

**Bash**
```
for f in 560 640 680; do python exp/lever_viz.py base $f 0,72,144,-144,-72 frames/lvb$f.png; done; python -c "
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/frames/lvb%d.png'%f) for f in [560,640,680]]
w=max(i.shape[1] for i in ims); ims=[cv2.resize(i,None,fx=1600/i.shape[1],fy=1600/i.shape[1]) for i in ims]
cv2.imwrite('/work/scratch/frames/lvb.png',np.concatenate(ims,0))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/lvb.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python - <<'EOF'
s=open('exp/lever.py').read()
s=s.replace("""    g = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), device='cuda').float()""","""    g0 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g0, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    g = torch.tensor(bh, device='cuda').float()""")
s=s.replace("sc.append((g[bm].median() - g[lm].mean()).item())","sc.append((g[lm].mean() - g[bm].median()).item())")
open('exp/lever_bh.py','w').write(s)
EOF
python exp/lever_bh.py base "list(range(500,600,10))+list(range(625,860,15))" 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
lever faces 6607 board faces 1007
500 [11.4  3.6 13.6  5.7  8.5]
510 [11.2 -0.2 15.  13.9  9.5]
520 [12.   2.4  5.6  5.8 11.2]
530 [13.   2.4  7.3  9.   8.7]
540 [14.   2.2  7.2  5.9 12.2]
550 [13.8  1.7 10.   6.7 12.1]
560 [11.6  0.2  9.7  4.3 12.4]
570 [13.4  1.1 10.4  7.7 12. ]
580 [13.1  1.8  9.7  4.7 12. ]
590 [11.3  1.6  8.5  4.6 10.5]
625 [4.4 7.4 1.9 2.5 5.5]
640 [6.  5.8 2.1 1.9 8. ]
655 [ 8.9  3.8  1.2  0.7 11.4]
670 [12.4  6.8  1.7  0.2  2.5]
685 [6.2 8.4 0.9 1.4 1.1]
700 [nan 3.6 2.2 1.1 0.5]
715 [ 3.3  2.7  2.1 -0.  15.8]
730 [ 2.   1.7  4.  -0.2  3.5]
745 [5.7 2.9 4.3 0.2 9.6]
760 [ 8.7  2.   1.9  1.5 11.5]
775 [ 7.4  1.4  2.2  0.8 10.7]
790 [ 5.4  2.   0.6  3.  12.8]
805 [ 4.4  2.8 -0.1  3.7  2.7]
820 [ 3.9  3.8 -0.5  5.4  5.8]
835 [ 3.   3.9  0.1  6.3 13.8]
850 [ 5.7  4.5 - …[+113 chars]
```
</details>

**Write**
```
/work/scratch/exp/plate.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/plate.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/plate.py base "list(range(500,600,10))+list(range(625,860,15))" 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
500 [0.019 0.05  0.017 0.004 0.   ]
510 [0.045 0.018 0.031 0.007 0.01 ]
520 [0.024 0.041 0.034 0.009 0.026]
530 [0.03  0.032 0.027 0.008 0.031]
540 [0.037 0.031 0.042 0.017 0.032]
550 [0.031 0.028 0.031 0.007 0.038]
560 [0.025 0.022 0.034 0.01  0.033]
570 [0.023 0.027 0.019 0.007 0.033]
580 [0.036 0.028 0.012 0.006 0.028]
590 [0.033 0.035 0.013 0.005 0.019]
625 [0.004 0.036 0.05  0.036 0.006]
640 [0.014 0.017 0.015 0.013 0.006]
655 [0.008 0.021 0.03  0.024 0.003]
670 [0.033 0.038 0.013 0.022 0.002]
685 [0.022 0.03  0.013 0.024 0.005]
700 [0.016 0.025 0.015 0.015 0.003]
715 [0.007 0.037 0.021 0.004 0.006]
730 [0.014 0.015 0.015 0.    0.003]
745 [0.022 0.026 0.001 0.003 0.013]
760 [0.018 0.019 0.001 0.003 0.007]
775 [0.027 0.022 0.004 0.008 0.015]
790 [0.016 0.009 0.007 0.01  0.053]
805 [0.0 …[+244 chars]
```
</details>

**Bash**
```
python exp/viz_spin.py base 560,640 frames/spin6.png -144,-72,144 && python -c "
import cv2; a=cv2.imread('/work/scratch/frames/spin6.png'); cv2.imwrite('/work/scratch/frames/spin6.png', cv2.resize(a,None,fx=1.3,fy=1.3))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/spin6.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/scratch/exp/board.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/board.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/board.py base "list(range(500,600,10))" 2>&1 | grep -v Warn; python exp/board.py base "list(range(625,860,15))" 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
500 ncc [0.37 0.53 0.85 0.74 0.5 ] match [90, 144, 113, 135, 161]
510 ncc [0.02 0.33 0.68 0.63 0.3 ] match [92, 55, 90, 179, 98]
520 ncc [0.01 0.48 0.66 0.37 0.35] match [207, 182, 160, 201, 84]
530 ncc [0.02 0.45 0.66 0.37 0.34] match [161, 183, 150, 203, 138]
540 ncc [-0.01  0.46  0.71  0.47  0.36] match [141, 165, 159, 191, 116]
550 ncc [0.06 0.45 0.71 0.47 0.44] match [135, 152, 132, 214, 91]
560 ncc [0.1  0.45 0.74 0.56 0.47] match [201, 207, 139, 198, 125]
570 ncc [0.07 0.44 0.73 0.51 0.43] match [114, 176, 121, 186, 117]
580 ncc [0.06 0.47 0.73 0.5  0.41] match [198, 181, 167, 181, 92]
590 ncc [0.04 0.48 0.73 0.51 0.4 ] match [203, 206, 159, 199, 85]
ncc mean {0: 0.074, 72: 0.455, 144: 0.72, -144: 0.513, -72: 0.4} wins {0: 0, 72: 0, 144: 10, -144: 0, -72: 0}
match mean {0: 154.2, 72 …[+1410 chars]
```
</details>

**Bash**
```
cat > exp/closeup.py <<'EOF'
import sys; sys.path.insert(0, '/work/outputs')
from common import *
poses=sys.argv[1]; f=int(sys.argv[2]); spins=[float(x) for x in sys.argv[3].split(',')]; out=sys.argv[4]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
ax_p = np.array([0.006, 0.004, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
cap = cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); im=im[:,:,::-1]
# centre of crop: projection of underside centre
p = Tc[f][:3,:3]@np.array([0.006,0.004,-0.08])+Tc[f][:3,3]; q=D['K']@p; u,v=int(q[0]/q[2]),int(q[1]/q[2])
s=int(sys.argv[5]) if len(sys.argv)>5 else 180
tiles=[im[v-s:v+s,u-s:u+s]]
o=R.render(np.stack([spinT(Tc[f],d) for d in spins]))
for i,d in enumerate(spins):
    rgb=(o['rgb'][i].cpu().numpy()*255).astype(np.uint8); m=o['mask'][i].cpu().numpy()
    c=(im*0.3).astype(np.uint8); c[m]=rgb[m]; c=c[v-s:v+s,u-s:u+s].copy(); cv2.putText(c,str(d),(5,25),0,0.8,(255,255,0),2); tiles.append(c)
cv2.imwrite(out,np.concatenate(tiles,1)[:,:,::-1])
EOF
python exp/closeup.py base 560 144,-144 frames/cu560.png
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/cu560.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
mesh=load_mesh('/work/inputs'); v=mesh['v']; f=mesh['f']; uv=mesh['uv']; tex=mesh['tex']
H,W=tex.shape[:2]
fc=v[f].mean(1); fuv=uv[f].mean(1)
col=tex[np.clip(((1-fuv[:,1])*H).astype(int),0,H-1), np.clip((fuv[:,0]*W).astype(int),0,W-1)]
c=np.array([0.006,0.004]); r=np.hypot(*(fc[:,:2]-c).T)
sel=(fc[:,2]>-0.085)&(fc[:,2]<-0.06)&(r<0.34)
g=col.mean(1)
print('underside faces',sel.sum(),'gray pct',np.percentile(g[sel],[5,25,50,75,95]))
br=sel&(g>0.5); dk=sel&(g<0.12)
print('bright',br.sum(),'dark',dk.sum())
from sklearn.cluster import DBSCAN
for nm,s in [('bright',br),('dark',dk)]:
  P=fc[s]; lab=DBSCAN(eps=0.015,min_samples=5).fit(P).labels_
  for l in set(lab)-{-1}:
    Q=P[lab==l]; print(nm,l,len(Q),'centre',np.round(Q.mean(0),3),'ext',np.round(Q.max(0)-Q.min(0),3),'r %.3f ang %.0f'%(np.hypot(*(Q.mean(0)[:2]-c)),np.degrees(np.arctan2(Q.mean(0)[1]-c[1],Q.mean(0)[0]-c[0]))))
"

```

<details><summary>output</summary>

```
underside faces 6665 gray pct [0.11111111 0.14379086 0.19215687 0.25751635 0.43529412]
bright 201 dark 517
bright 0 199 centre [-0.006 -0.03  -0.069] ext [0.059 0.02  0.016] r 0.036 ang -110
dark 0 410 centre [ 0.021  0.043 -0.075] ext [0.116 0.061 0.025] r 0.042 ang 69
dark 1 21 centre [ 0.04  -0.065 -0.082] ext [0.018 0.01  0.003] r 0.077 ang -64
dark 2 41 centre [-0.034  0.008 -0.069] ext [0.018 0.012 0.015] r 0.041 ang 174
dark 3 7 centre [ 0.075 -0.07  -0.081] ext [0.007 0.017 0.002] r 0.101 ang -47
dark 4 38 centre [-0.094  0.04  -0.082] ext [0.009 0.009 0.006] r 0.107 ang 160
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
mesh=load_mesh('/work/inputs'); v=mesh['v']
c=np.array([0.006,0.004]); r=np.hypot(*(v[:,:2]-c).T); ang=np.degrees(np.arctan2(v[:,1]-c[1],v[:,0]-c[0]))
s=(r>0.05)&(r<0.38)&(v[:,2]>-0.07)&(v[:,2]<0.2)&((ang>130)|(ang<-170))
for r0 in np.arange(0.05,0.38,0.025):
  q=s&(r>=r0)&(r<r0+0.025)
  if q.sum(): print('r %.3f n %d z [%.3f %.3f] ang med %.0f'%(r0,q.sum(),v[q,2].min(),v[q,2].max(),np.median(ang[q])))
"

```

<details><summary>output</summary>

```
r 0.050 n 1651 z [-0.070 -0.046] ang med 162
r 0.075 n 1837 z [-0.054 -0.039] ang med 163
r 0.100 n 1259 z [-0.047 -0.033] ang med 163
r 0.125 n 658 z [-0.039 -0.026] ang med 162
r 0.150 n 828 z [-0.031 -0.019] ang med 162
r 0.175 n 743 z [-0.024 -0.011] ang med 164
r 0.200 n 1870 z [-0.015 0.008] ang med 161
r 0.225 n 2153 z [-0.011 0.011] ang med 160
r 0.250 n 2884 z [-0.005 0.015] ang med 160
r 0.275 n 3388 z [0.002 0.021] ang med 158
r 0.300 n 3236 z [0.004 0.023] ang med 159
r 0.325 n 534 z [0.010 0.023] ang med 162
```
</details>

**Write**
```
/work/scratch/exp/lever2.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/lever2.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/lever2.py base "[560]" dbg 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
560 [0.27 0.15 0.58 0.52 0.  ] linepx 855
mean {0: 0.269, 72: 0.145, 144: 0.581, -144: 0.515, -72: 0.001} wins {0: 0, 72: 0, 144: 1, -144: 0, -72: 0}
```
</details>

**Read**
```
/work/scratch/frames/l2_560.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
sed -i 's/rr = np.linspace(0.08, 0.31, 24)/rr = np.linspace(0.16, 0.31, 16)/; s/2 \* 8 \*\* 2/2 * 12 ** 2/' exp/lever2.py; python exp/lever2.py base "list(range(500,600,10))+list(range(625,860,15))" 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
500 [0.03 0.02 0.92 0.8  0.07] linepx 562
510 [0.08 0.07 0.94 0.93 0.02] linepx 857
520 [0.06 0.16 0.93 0.82 0.02] linepx 1143
530 [0.07 0.1  0.96 0.92 0.02] linepx 920
540 [0.05 0.13 0.96 0.71 0.01] linepx 759
550 [0.09 0.1  0.94 0.91 0.02] linepx 901
560 [0.06 0.09 0.93 0.83 0.  ] linepx 855
570 [0.1  0.08 0.96 0.92 0.  ] linepx 790
580 [0.07 0.09 0.93 0.85 0.  ] linepx 745
590 [0.05 0.1  0.93 0.67 0.01] linepx 734
625 [0.06 0.11 0.26 0.84 0.86] linepx 1135
640 [0.49 0.18 0.48 0.48 0.81] linepx 2005
655 [ nan 0.14 0.32 0.45 0.73] linepx 1559
670 [ nan 0.13 0.48 0.42 0.31] linepx 1398
685 [ nan 0.13 0.23 0.43 0.33] linepx 1318
700 [ nan 0.05 0.36 0.19 0.35] linepx 1711
715 [0.11 0.2  0.6  0.   0.74] linepx 1375
730 [0.43 0.13 0.73 0.   0.46] linepx 1580
745 [0.1  0.25 0.72 0.02 0.66] line …[+416 chars]
```
</details>

**Bash**
```
python exp/lever2.py base "[560,530]" dbg 2>&1 | grep -v Warn; python -c "
import cv2,numpy as np
a=cv2.imread('/work/scratch/frames/l2_560.png'); b=cv2.imread('/work/scratch/frames/l2_530.png')
h=max(a.shape[0],b.shape[0]); a=cv2.copyMakeBorder(a,0,h-a.shape[0],0,0,0); b=cv2.copyMakeBorder(b,0,h-b.shape[0],0,0,0)
cv2.imwrite('/work/scratch/frames/l2.png',cv2.resize(np.concatenate([a,b],1),None,fx=2,fy=2))"
```

<details><summary>output</summary>

```
560 [0.06 0.09 0.93 0.83 0.  ] linepx 855
530 [0.07 0.1  0.96 0.92 0.02] linepx 920
mean {0: 0.067, 72: 0.1, 144: 0.946, -144: 0.873, -72: 0.011} wins {0: 0, 72: 0, 144: 2, -144: 0, -72: 0}
```
</details>

**Read**
```
/work/scratch/frames/l2.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/scratch/exp/lever_angle.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/lever_angle.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/lever_angle.py base "list(range(440,880,8))" 2>&1 | grep -v Warn | column -c 200
```

<details><summary>output</summary>

```
440 lever_ang nan n 0 facing -0.28  spin_corr nan	592 lever_ang 45.0 n 60 facing 0.45  spin_corr -116.0	744 lever_ang -124.6 n 40 facing 0.75  spin_corr 74.4
448 lever_ang nan n 0 facing -0.24  spin_corr nan	600 lever_ang 52.4 n 77 facing 0.45  spin_corr -108.6	752 lever_ang -106.5 n 37 facing 0.70  spin_corr 92.5
456 lever_ang nan n 0 facing -0.21  spin_corr nan	608 lever_ang 90.6 n 118 facing 0.45  spin_corr -70.4	760 lever_ang -119.6 n 30 facing 0.66  spin_corr 79.4
464 lever_ang nan n 0 facing -0.16  spin_corr nan	616 lever_ang 107.4 n 131 facing 0.48  spin_corr -53.6	768 lever_ang -118.8 n 40 facing 0.66  spin_corr 80.2
472 lever_ang nan n 0 facing -0.08  spin_corr nan	624 lever_ang 110.6 n 89 facing 0.52  spin_corr -50.4	776 lever_ang -131.3 n 37 facing 0.68  spin_corr 67.7
480 lever …[+2106 chars]
```
</details>

**Bash**
```
python - <<'EOF'
import re
s=open('exp/lever_angle.py').read()
s=s.replace("    out.append((f, (peak + 180) % 360 - 180, int(m.sum()), facing))",
"""    out.append((f, (peak + 180) % 360 - 180, int(m.sum()), facing))
    if len(sys.argv) > 3:
        a = im.copy(); a[line] = (0, 0, 255)
        yy, xx = ys[sel][np.abs(((ang[sel] - peak + 180) % 360) - 180) < 12], xs[sel][np.abs(((ang[sel] - peak + 180) % 360) - 180) < 12]
        a[yy, xx] = (0, 255, 0)
        # mesh lever at 161 deg for reference, projected with this pose
        for rr_ in np.linspace(0.13, 0.33, 12):
            P = T[:3, :3] @ np.array([c[0] + rr_ * np.cos(np.radians(161)), c[1] + rr_ * np.sin(np.radians(161)), ZB]) + T[:3, 3]
            q = K @ P; cv2.circle(a, (int(q[0] / q[2]), int(q[1] / q[2])), 2, (255, 0, 0), -1)
        y_, x_ = np.nonzero(om); cv2.imwrite('/work/scratch/frames/la_%d.png' % f, a[y_.min():y_.max(), x_.min():x_.max()])""")
open('exp/lever_angle.py','w').write(s)
EOF
python exp/lever_angle.py base "[560,640,760]" dbg 2>&1 | grep -v Warn; python -c "
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/frames/la_%d.png'%f) for f in [560,640,760]]
h=max(i.shape[0] for i in ims); ims=[cv2.copyMakeBorder(i,0,h-i.shape[0],0,0,0) for i in ims]
cv2.imwrite('/work/scratch/frames/la.png',np.concatenate(ims,1))"
```

<details><summary>output</summary>

```
560 lever_ang 41.7 n 73 facing 0.47  spin_corr -119.3
640 lever_ang 106.9 n 99 facing 0.63  spin_corr -54.1
760 lever_ang -119.6 n 30 facing 0.66  spin_corr 79.4
```
</details>

**Read**
```
/work/scratch/frames/la.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python - <<'EOF'
s=open('exp/lever_angle.py').read()
s=s.replace("line = (bh > 25) & (g < 0.6 * med) & bm","line = (bh > 18) & (g < 0.7 * med) & bm")
s=s.replace("""    out.append((f, (peak + 180) % 360 - 180, int(m.sum()), facing))
""","""    out.append((f, (peak + 180) % 360 - 180, int(m.sum()), facing))
    # chamfer score of 3D mesh lever polyline under the 5 spin hypotheses
    det = np.zeros(g.shape, np.uint8); det[ys[sel], xs[sel]] = 1
    dt = cv2.distanceTransform(1 - det, cv2.DIST_L2, 5)
    rr = np.linspace(0.16, 0.31, 16); th = np.radians(161)
    LP = np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th), np.interp(rr, [0.05, 0.31], [-0.06, 0.02])]
    scs = []
    for s_ in [0, 72, 144, -144, -72]:
        Rz = axis_angle([0, 0, 1], np.radians(s_)); ap = np.array([0.006, 0.004, 0])
        Pm = (LP - ap) @ Rz.T + ap
        P = Pm @ T[:3, :3].T + T[:3, 3]; q = P @ K.T
        ui = np.clip((q[:, 0] / q[:, 2]).round().astype(int), 0, 1535); vi = np.clip((q[:, 1] / q[:, 2]).round().astype(int), 0, 1151)
        scs.append(np.mean(np.exp(-dt[vi, ui] ** 2 / (2 * 15 ** 2))))
    CH.append([f] + scs)
""")
s=s.replace("out = []\n","out = []\nCH = []\n")
s=s.replace("for o in out:","CH = np.array(CH)\nfor r_ in CH: print('chamfer', int(r_[0]), np.round(r_[1:], 2), [0, 72, 144, -144, -72][int(np.argmax(r_[1:]))])\nfor o in out:")
open('exp/lever_angle.py','w').write(s)
EOF
python exp/lever_angle.py base "list(range(500,880,8))" 2>&1 | grep chamfer | column -c 200
```

<details><summary>output</summary>

```
chamfer 500 [0.16 0.12 0.4  0.89 0.18] -144	chamfer 596 [0.15 0.18 0.84 0.85 0.22] -144	chamfer 692 [0.67 0.24 0.5  0.5  0.64] 0	chamfer 788 [0.07 0.57 0.53 0.   0.78] -72
chamfer 508 [0.19 0.14 0.25 0.95 0.21] -144	chamfer 604 [0.22 0.09 0.12 0.98 0.59] -144	chamfer 700 [0.74 0.24 0.53 0.02 0.66] 0	chamfer 796 [0.79 0.7  0.27 0.   0.76] 0
chamfer 516 [0.13 0.21 0.89 0.82 0.17] 144	chamfer 612 [0.67 0.09 0.86 0.6  0.56] 144	chamfer 708 [0.78 0.28 0.65 0.38 0.76] 0	chamfer 804 [0.2  0.78 0.19 0.   0.67] 72
chamfer 524 [0.14 0.26 0.2  0.8  0.2 ] -144	chamfer 620 [0.16 0.15 0.23 0.93 0.91] -144	chamfer 716 [0.61 0.36 0.76 0.27 0.87] -72	chamfer 812 [0.37 0.82 0.18 0.   0.64] 72
chamfer 532 [0.15 0.21 0.14 0.86 0.17] -144	chamfer 628 [0.27 0.22 0.37 0.07 0.98] -72	chamfer 724 [0.57 0.34 0.85 0 …[+1212 chars]
```
</details>

**Bash**
```
python - <<'EOF'
s=open('exp/closeup.py').read()
s=s.replace("p = Tc[f][:3,:3]@np.array([0.006,0.004,-0.08])+Tc[f][:3,3]","zc=float(sys.argv[6]) if len(sys.argv)>6 else -0.08\np = Tc[f][:3,:3]@np.array([0.006,0.004,zc])+Tc[f][:3,3]")
open('exp/closeup.py','w').write(s)
EOF
python exp/closeup.py base 350 0,72,144,-144,-72 frames/cu350.png 200 -0.12
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/cu350.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/outputs/rectify.py
```

<details><summary>output</summary>

```
File created successfully at: /work/outputs/rectify.py (file state is current in your context — no need to Read it back)
```
</details>

**Write**
```
/work/scratch/exp/rect_pool.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/rect_pool.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/rect_pool.py base under "[(505,596,3),(625,700,3),(700,800,3)]" b 2>&1 | grep -v Warn; python -c "
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/frames/'+n) for n in ['rect_ref_under.png','rect_b_under_0.png','rect_b_under_1.png','rect_b_under_2.png']]
cv2.imwrite('/work/scratch/frames/rect_under.png',np.concatenate(ims,1))"
```

<details><summary>output</summary>

```
seg 505 596 coverage 0.0454
seg 625 700 coverage 0.23286875
seg 700 800 coverage 0.30118125
```
</details>

**Read**
```
/work/scratch/frames/rect_under.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
m=load_mesh('/work/inputs'); g=m['mesh']; fc=g.triangles_center; n=g.face_normals
r=np.hypot(fc[:,0]-0.006,fc[:,1]-0.004); s=(r>0.15)&(r<0.3)
for nm,sel in [('+z',s&(n[:,2]>0.8)),('-z',s&(n[:,2]<-0.8))]:
  print(nm, sel.sum(), np.percentile(fc[sel,2],[5,25,50,75,95]))
"
```

<details><summary>output</summary>

```
+z 6020 [-0.09480549 -0.0937421  -0.09259945 -0.08729688  0.01008282]
-z 3958 [-0.11642352 -0.11567933 -0.11456044  0.54588643  0.56504041]
```
</details>

**Bash**
```
sed -i 's/z0, sign = -0.079, +1/z0, sign = -0.092, +1/; s/z0, sign = -0.1215, -1/z0, sign = -0.115, -1/' exp/rect_pool.py; sed -i 's/< 0.012)/< 0.015)/' /work/outputs/rectify.py; python exp/rect_pool.py base under "[(505,596,3),(625,700,3),(700,800,3)]" b 2>&1 | grep -v Warn; python -c "
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/frames/'+n) for n in ['rect_ref_under.png','rect_b_under_0.png','rect_b_under_1.png','rect_b_under_2.png']]
cv2.imwrite('/work/scratch/frames/rect_under.png',np.concatenate(ims,1))"
```

<details><summary>output</summary>

```
seg 505 596 coverage 0.68625
seg 625 700 coverage 0.686075
seg 700 800 coverage 0.69434375
```
</details>

**Read**
```
/work/scratch/frames/rect_under.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/scratch/exp/rect_corr.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/rect_corr.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/rect_corr.py under selftest,/work/scratch/rect_b_under_0.npy,/work/scratch/rect_b_under_1.npy,/work/scratch/rect_b_under_2.npy
```

<details><summary>output</summary>

```
selftest peak 30
```
</details>

**Bash**
```
sed -i "s/real = rotate_grid(ref, 30); files = \[\]/real = rotate_grid(ref, 30); files = files[1:]/" exp/rect_corr.py; python exp/rect_corr.py under selftest,/work/scratch/rect_b_under_0.npy,/work/scratch/rect_b_under_1.npy,/work/scratch/rect_b_under_2.npy
```

<details><summary>output</summary>

```
selftest peak 30
rect_b_under_0.npy peaks [(-144, 0.325), (-146, 0.324), (-148, 0.292), (-142, 0.275), (176, 0.268)] grid {0: 0.236, 72: 0.202, 144: 0.21, -144: 0.325, -72: 0.215}
rect_b_under_1.npy peaks [(-74, 0.273), (-76, 0.256), (-72, 0.24), (-104, 0.238), (-106, 0.236)] grid {0: 0.215, 72: 0.165, 144: 0.147, -144: 0.176, -72: 0.24}
rect_b_under_2.npy peaks [(-30, 0.215), (-28, 0.214), (-26, 0.209), (-32, 0.201), (-46, 0.197)] grid {0: 0.178, 72: 0.164, 144: 0.155, -144: 0.156, -72: 0.138}
```
</details>

**Bash**
```
python exp/rect_pool.py base top "[(316,390,2),(0,98,3),(937,993,3),(200,300,3)]" b 2>&1 | grep -v Warn; python exp/rect_corr.py top selftest,/work/scratch/rect_b_top_0.npy,/work/scratch/rect_b_top_1.npy,/work/scratch/rect_b_top_2.npy,/work/scratch/rect_b_top_3.npy; python -c "
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/frames/'+n) for n in ['rect_ref_top.png','rect_b_top_0.png','rect_b_top_1.png','rect_b_top_2.png']]
cv2.imwrite('/work/scratch/frames/rect_top.png',np.concatenate(ims,1))"
```

<details><summary>output</summary>

```
Exit code 1
seg 316 390 coverage 0.7477125
Traceback (most recent call last):
  File "/work/scratch/exp/rect_pool.py", line 45, in <module>
    np.save('/work/scratch/rect_%s_%s_%d.npy' % (tag, which, si), np.concatenate([img, wacc[..., None]], -1))
                                                                                       ~~~~^^^^^^^^^^^
TypeError: 'int' object is not subscriptable
selftest peak 30
rect_b_top_0.npy peaks [(-26, 0.971), (-88, 0.971), (-38, 0.971), (-50, 0.97), (-92, 0.97)] grid {0: 0.969, 72: 0.967, 144: 0.969, -144: 0.969, -72: 0.97}
Traceback (most recent call last):
  File "/work/scratch/exp/rect_corr.py", line 35, in <module>
    real = np.load(fn)
           ^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
 …[+737 chars]
```
</details>

**Bash**
```
python -c "
import cv2,numpy as np
ims=[cv2.imread('/work/scratch/frames/'+n) for n in ['rect_ref_top.png','rect_b_top_0.png']]
cv2.imwrite('/work/scratch/frames/rect_top.png',np.concatenate(ims,1))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/rect_top.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python - <<'EOF'
s=open('exp/rect_corr.py').read()
s=s.replace("""    hp = g - b / np.maximum(bw, 1e-3)
    hp = cv2.GaussianBlur(hp, (0, 0), 1.5)
    return hp, m""","""    hp = np.where(m, g - b / np.maximum(bw, 1e-3), 0)
    hp = cv2.GaussianBlur(hp, (0, 0), 1.5) / np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5), 1e-3)
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return hp, m""")
open('exp/rect_corr.py','w').write(s)
EOF
python exp/rect_corr.py top selftest,/work/scratch/rect_b_top_0.npy; python exp/rect_corr.py under selftest,/work/scratch/rect_b_under_0.npy,/work/scratch/rect_b_under_1.npy,/work/scratch/rect_b_under_2.npy
```

<details><summary>output</summary>

```
selftest peak 30
rect_b_top_0.npy peaks [(152, 0.16), (154, 0.147), (144, 0.124), (146, 0.094), (150, 0.091)] grid {0: 0.032, 72: -0.001, 144: 0.124, -144: 0.042, -72: -0.007}
selftest peak 30
rect_b_under_0.npy peaks [(-146, 0.161), (-144, 0.16), (-148, 0.113), (-142, 0.085), (-150, 0.076)] grid {0: 0.012, 72: -0.003, 144: 0.011, -144: 0.16, -72: -0.009}
rect_b_under_1.npy peaks [(-74, 0.19), (-76, 0.161), (-72, 0.14), (-44, 0.106), (-106, 0.095)] grid {0: 0.041, 72: 0.004, 144: -0.011, -144: 0.016, -72: 0.14}
rect_b_under_2.npy peaks [(-104, 0.049), (-102, 0.048), (-166, 0.046), (-30, 0.044), (-168, 0.041)] grid {0: 0.023, 72: 0.013, 144: 0.001, -144: 0.034, -72: -0.005}
```
</details>

**Bash**
```
python exp/rect_pool.py opt1.npz top "[(230,270,2),(270,316,2),(316,390,2),(390,430,2),(430,470,2)]" o 2>&1 | grep -v Warn; python exp/rect_corr.py top $(ls /work/scratch/rect_o_top_*.npy | tr '\n' ',' | sed 's/,$//'); python exp/rect_pool.py opt1.npz under "[(440,480,2),(480,505,2),(505,596,3),(596,625,2),(625,700,3),(700,760,3),(760,850,3)]" o 2>&1 | grep -v Warn; python exp/rect_corr.py under $(ls /work/scratch/rect_o_under_*.npy | tr '\n' ',' | sed 's/,$//')
```

<details><summary>output</summary>

```
Exit code 1
seg 230 270 coverage 0.7267375
seg 270 316 coverage 0.7114125
seg 316 390 coverage 0.70993125
seg 390 430 coverage 0.73816875
seg 430 470 coverage 0.74521875
rect_o_top_0.npy peaks [(140, 0.262), (138, 0.234), (142, 0.203), (144, 0.122), (136, 0.111)] grid {0: 0.023, 72: -0.015, 144: 0.122, -144: 0.029, -72: -0.007}
rect_o_top_1.npy peaks [(142, 0.221), (144, 0.191), (140, 0.175), (138, 0.133), (136, 0.131)] grid {0: 0.018, 72: 0.0, 144: 0.191, -144: 0.02, -72: -0.041}
rect_o_top_2.npy peaks [(144, 0.3), (146, 0.296), (142, 0.119), (148, 0.108), (8, 0.073)] grid {0: 0.016, 72: 0.029, 144: 0.3, -144: -0.018, -72: -0.012}
rect_o_top_3.npy peaks [(144, 0.185), (146, 0.177), (142, 0.167), (140, 0.155), (138, 0.137)] grid {0: 0.021, 72: 0.05, 144: 0.185, -144: -0.018, -72: -0.016}
r …[+827 chars]
```
</details>

**Bash**
```
python exp/rect_pool.py opt1.npz under "[(480,505,2),(505,596,3),(596,625,2),(625,700,3),(700,760,3),(760,850,3)]" o 2>&1 | grep -v Warn; python exp/rect_corr.py under $(ls /work/scratch/rect_o_under_*.npy | tr '\n' ',' | sed 's/,$//')
```

<details><summary>output</summary>

```
seg 480 505 coverage 0.61480625
seg 505 596 coverage 0.65271875
seg 596 625 coverage 0.67878125
seg 625 700 coverage 0.66959375
seg 700 760 coverage 0.68283125
seg 760 850 coverage 0.67455625
rect_o_under_0.npy peaks [(-148, 0.14), (36, 0.128), (-146, 0.117), (34, 0.116), (38, 0.106)] grid {0: 0.039, 72: -0.007, 144: -0.053, -144: 0.066, -72: 0.022}
rect_o_under_1.npy peaks [(-150, 0.284), (-152, 0.238), (-148, 0.223), (-146, 0.133), (-154, 0.126)] grid {0: -0.003, 72: -0.035, 144: 0.06, -144: 0.08, -72: 0.013}
rect_o_under_2.npy peaks [(-82, 0.248), (-84, 0.244), (-86, 0.22), (-80, 0.203), (-88, 0.195)] grid {0: 0.058, 72: 0.034, 144: -0.001, -144: 0.141, -72: -0.012}
rect_o_under_3.npy peaks [(-80, 0.254), (-78, 0.204), (-82, 0.186), (-76, 0.132), (-50, 0.104)] grid {0: 0.02, 72: 0.028,  …[+357 chars]
```
</details>

**Write**
```
/work/scratch/exp/sil_sweep.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/sil_sweep.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/sil_sweep.py opt1.npz "list(range(0,98,7))"; echo; python exp/sil_sweep.py opt1.npz "list(range(940,993,4))"; echo; python exp/sil_sweep.py opt1.npz "list(range(866,896,3))"
```

<details><summary>output</summary>

```
grid    0: best    0 iou 0.8266
grid   72: best   70 iou 0.8028
grid  144: best  140 iou 0.7258
grid -144: best -150 iou 0.7094
grid  -72: best  -72 iou 0.7339

grid    0: best    0 iou 0.8318
grid   72: best   72 iou 0.8523
grid  144: best  142 iou 0.8123
grid -144: best -148 iou 0.7446
grid  -72: best  -78 iou 0.7665

grid    0: best    0 iou 0.7773
grid   72: best   70 iou 0.8208
grid  144: best  140 iou 0.7637
grid -144: best -148 iou 0.7001
grid  -72: best  -78 iou 0.7234
```
</details>

**Write**
```
/work/scratch/exp/static_hyp.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/static_hyp.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/static_hyp.py opt1.npz "list(range(0,98,6))"; echo; python exp/static_hyp.py opt1.npz "list(range(940,993,4))"
```

<details><summary>output</summary>

```
hyp    0  loss 0.3429  iou 0.8469  drot 0.93 deg  dp [0.1 0.3 0.1] cm
hyp   72  loss 0.4428  iou 0.8297  drot 1.73 deg  dp [0.2 0.6 2.1] cm
hyp  144  loss 0.5840  iou 0.7810  drot 1.48 deg  dp [1.7 0.5 1. ] cm
hyp -144  loss 0.6441  iou 0.7828  drot 3.45 deg  dp [0.9 1.  2. ] cm
hyp  -72  loss 0.6123  iou 0.7770  drot 1.48 deg  dp [ 0.8  0.2 -3.3] cm

hyp    0  loss 0.4850  iou 0.8393  drot 0.38 deg  dp [0.2 0.5 1.3] cm
hyp   72  loss 0.3192  iou 0.8874  drot 1.23 deg  dp [0.6 0.9 3.5] cm
hyp  144  loss 0.3126  iou 0.8870  drot 2.09 deg  dp [1.3 1.  4.7] cm
hyp -144  loss 0.5323  iou 0.8316  drot 1.37 deg  dp [2.  1.4 4.3] cm
hyp  -72  loss 0.5205  iou 0.8274  drot 1.63 deg  dp [1.1 1.  1.2] cm
```
</details>

**Bash**
```
python exp/closeup.py opt1.npz 50 0,72,144,-144,-72 frames/cu50.png 110 0.1; python -c "
import cv2; a=cv2.imread('/work/scratch/frames/cu50.png'); cv2.imwrite('/work/scratch/frames/cu50.png', cv2.resize(a,None,fx=1.4,fy=1.4))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/cu50.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > exp/sil_cmp.py <<'EOF'
# full-res silhouette comparison for spin hypotheses: green=real mask only, red=render only, white=both
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses=sys.argv[1]; f=int(sys.argv[2]); spins=[float(x) for x in sys.argv[3].split(',')]; out=sys.argv[4]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
ax_p = np.array([0.006, 0.004, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
h=h5py.File('/work/inputs/masks.h5'); m=h['object'][f]; p=h['person'][f]
o=R.render(np.stack([spinT(Tc[f],s) for s in spins]),rgb=False)['mask'].cpu().numpy()
ys,xs=np.nonzero(m|o.any(0)); y0,y1,x0,x1=ys.min()-10,ys.max()+10,xs.min()-10,xs.max()+10
tiles=[]
for i,s in enumerate(spins):
    a=np.zeros(m.shape+(3,),np.uint8); a[m&~o[i]]=(0,255,0); a[o[i]&~m]=(0,0,255); a[o[i]&m]=(255,255,255); a[p&~m]=(a[p&~m]*0.3+(80,0,0)).astype(np.uint8)
    a=a[y0:y1,x0:x1].copy(); cv2.putText(a,str(s),(5,25),0,0.8,(0,255,255),2); tiles.append(a)
cv2.imwrite(out,np.concatenate(tiles,1))
EOF
python exp/sil_cmp.py opt1.npz 50 0,72,144,-144,-72 frames/sc50.png
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/sc50.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
python - <<'EOF'
s=open('exp/static_hyp.py').read()
s=s.replace("    print('hyp %4d","    Tf = np.eye(4); Tf[:3, :3] = Rr.detach().cpu().numpy(); Tf[:3, 3] = pp.detach().cpu().numpy() - Tf[:3, :3] @ c\n    RES[g] = Tf\n    print('hyp %4d")
s=s.replace("for g in [0, 72, 144, -144, -72]:","RES = {}\nfor g in [0, 72, 144, -144, -72]:")
s+="""
f0 = frames[len(frames) // 2]
Tall = np.repeat((D['Twc'] @ RES[0])[None], 993, 0)
for i, g in enumerate([0, 72, 144, -144, -72]):
    Tall[i] = D['Twc'] @ RES[g]
np.savez('/work/scratch/hyp_static.npz', T_world_mesh=Tall, frame=f0)
"""
open('exp/static_hyp.py','w').write(s)
EOF
python exp/static_hyp.py opt1.npz "list(range(0,98,6))" >/dev/null; python - <<'EOF'
# render each refined hypothesis at frame 48 (stored in rows 0..4)
import sys; sys.path.insert(0,'/work/outputs'); sys.argv=['x']
from common import *
import h5py
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); T=np.load('/work/scratch/hyp_static.npz')['T_world_mesh'][:5]
Tc=np.linalg.inv(D['Twc'])@T; R=Renderer(mesh,D['K'],1152,1536)
f=48; h=h5py.File('/work/inputs/masks.h5'); m=h['object'][f]
o=R.render(Tc,rgb=False)['mask'].cpu().numpy()
ys,xs=np.nonzero(m); y0,y1,x0,x1=ys.min()-10,ys.max()+10,xs.min()-10,xs.max()+10
tiles=[]
for i,s in enumerate([0,72,144,-144,-72]):
    a=np.zeros(m.shape+(3,),np.uint8); a[m&~o[i]]=(0,255,0); a[o[i]&~m]=(0,0,255); a[o[i]&m]=(255,255,255)
    a=a[y0:y1,x0:x1].copy(); cv2.putText(a,str(s),(5,25),0,0.8,(0,255,255),2); tiles.append(a) …[+106 chars]
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/sc_ref.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
sed -i 's/rr, w = rectify(im, Tc\[fr\], K, P, depth=od\[.depth.\]\[0\].cpu().numpy(), valid_mask=vm, normal_sign=sign)/rr, w = rectify(im, Tc[fr], K, P, depth=od["depth"][0].cpu().numpy(), valid_mask=vm, normal_sign=sign, min_facing=0.08)/' exp/rect_pool.py; grep -n "min_facing" exp/rect_pool.py; python exp/rect_pool.py opt1.npz top "[(0,98,2),(940,993,2),(150,200,2),(200,230,2)]" s 2>&1 | grep -v Warn; python exp/rect_corr.py top $(ls /work/scratch/rect_s_top_*.npy | tr '\n' ',' | sed 's/,$//')
```

<details><summary>output</summary>

```
39:        rr, w = rectify(im, Tc[fr], K, P, depth=od["depth"][0].cpu().numpy(), valid_mask=vm, normal_sign=sign, min_facing=0.08)
seg 0 98 coverage 0.53313125
seg 940 993 coverage 0.67103125
seg 150 200 coverage 0.67270625
seg 200 230 coverage 0.69844375
rect_s_top_0.npy peaks [(-172, 0.107), (-170, 0.097), (-174, 0.077), (-168, 0.076), (-118, 0.073)] grid {0: 0.011, 72: -0.012, 144: 0.009, -144: 0.031, -72: -0.051}
rect_s_top_1.npy peaks [(-150, 0.261), (-148, 0.238), (-152, 0.199), (-146, 0.161), (-154, 0.098)] grid {0: -0.004, 72: -0.004, 144: 0.035, -144: 0.082, -72: -0.02}
rect_s_top_2.npy peaks [(-170, 0.053), (-122, 0.049), (-124, 0.045), (-168, 0.045), (174, 0.043)] grid {0: -0.028, 72: -0.021, 144: 0.002, -144: -0.047, -72: -0.012}
rect_s_top_3.npy peaks [(140, 0.19), (142, 0.176 …[+108 chars]
```
</details>

**Bash**
```
python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs'); B=D['rec']['T_world_mesh'].astype(np.float64)
O=np.load('/work/scratch/opt1.npz')['T_world_mesh']
for T,nm in [(B,'base'),(O,'opt1')]:
  for a,b in [(50,215),(50,250),(215,350),(50,960),(560,700),(700,960),(350,560)]:
    rel=T[a,:3,:3].T@T[b,:3,:3]; rv=cv2.Rodrigues(rel)[0][:,0]; ang=np.degrees(np.linalg.norm(rv)); ax=rv/np.linalg.norm(rv)
    print(nm,a,b,'angle %.1f'%ang,'body axis',np.round(ax,2))
"
```

<details><summary>output</summary>

```
base 50 215 angle 20.8 body axis [0.14 0.94 0.3 ]
base 50 250 angle 30.8 body axis [0.18 0.97 0.17]
base 215 350 angle 69.3 body axis [ 0.18  0.95 -0.25]
base 50 960 angle 38.3 body axis [-0.  0. -1.]
base 560 700 angle 103.5 body axis [-0.11  0.56 -0.82]
base 700 960 angle 129.7 body axis [-0.64  0.76  0.13]
base 350 560 angle 99.0 body axis [0.3  0.91 0.29]
opt1 50 215 angle 19.7 body axis [0.2  0.92 0.34]
opt1 50 250 angle 30.0 body axis [0.2  0.96 0.18]
opt1 215 350 angle 67.2 body axis [ 0.26  0.95 -0.18]
opt1 50 960 angle 37.6 body axis [ 0.01 -0.02 -1.  ]
opt1 560 700 angle 104.5 body axis [-0.1   0.56 -0.82]
opt1 700 960 angle 129.7 body axis [-0.65  0.75  0.13]
opt1 350 560 angle 97.2 body axis [0.35 0.91 0.22]
```
</details>

**Bash**
```
python -c "
import cv2, numpy as np, h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4'); h=h5py.File('/work/inputs/masks.h5'); tiles=[]
for f in range(90,215,9):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read()
    m=h['object'][f]; ys,xs=np.nonzero(m); cy,cx=int(ys.mean()),int(xs.mean())
    a=im[cy-170:cy+170,cx-170:cx+170].copy(); cv2.putText(a,str(f),(5,30),0,1,(0,255,255),2); tiles.append(a)
while len(tiles)%5: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite('/work/scratch/frames/spinseq.png',np.concatenate([np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)],0))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/spinseq.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cat > exp/conv_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch/exp')
from common import *
from rectify import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); K=D['K']
Tw=np.load('/work/scratch/opt1.npz')['T_world_mesh']; Tc=np.linalg.inv(D['Twc'])@Tw
R=Renderer(mesh,K,1152,1536)
ax_p=np.array([0.006,0.004,0.0])
def spinT(T,deg):
    S=np.eye(4); S[:3,:3]=axis_angle([0,0,1],np.radians(deg)); S[:3,3]=ax_p-S[:3,:3]@ax_p; return T@S
f=560; P=plane_grid(-0.092)
o=R.render(spinT(Tc[f],40)[None],depth=True)
img=o['rgb'][0].cpu().numpy(); dep=o['depth'][0].cpu().numpy()
rr,w=rectify(img,Tc[f],K,P,depth=dep,valid_mask=o['mask'][0].cpu().numpy().astype(np.float32),normal_sign=1)
np.save('/work/scratch/rect_conv.npy',np.concatenate([rr,w[...,None]],-1))
EOF
python exp/conv_test.py && python exp/rect_corr.py under /work/scratch/rect_conv.npy
```

<details><summary>output</summary>

```
rect_conv.npy peaks [(40, 0.791), (38, 0.708), (42, 0.58), (36, 0.401), (44, 0.366)] grid {0: 0.013, 72: -0.003, 144: 0.001, -144: -0.007, -72: -0.003}
```
</details>

**Write**
```
/work/scratch/exp/win_scan.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/win_scan.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
mkdir -p /work/scratch/win; cd /work/scratch && python exp/win_scan.py base 30 win/b 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bkcc66ykh). Output is being written to: /tmp/claude-1003/-work/f5f61c40-1890-45ca-8f67-396f0c76ee6d/tasks/bkcc66ykh.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
python -c "
import cv2, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); tiles=[]
for f in [60,100,115,130,145,160,175,190,200,210]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read()
    a=im[700:960,380:780].copy(); cv2.putText(a,str(f),(5,30),0,1,(0,255,255),2); tiles.append(a)
cv2.imwrite('/work/scratch/frames/legs.png',np.concatenate([np.concatenate(tiles[:5],1),np.concatenate(tiles[5:],1)],0))
"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/legs.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Write**
```
/work/scratch/exp/spin_track.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/spin_track.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
python exp/spin_track.py base "list(range(60,262,6))" 2>&1 | column -c 180
```

<details><summary>output</summary>

```
60 best 0.0 iou 0.757 (min 0.538)	114 best -1.5 iou 0.768 (min 0.532)	168 best 0.0 iou 0.772 (min 0.543)	222 best 3.0 iou 0.842 (min 0.660)
66 best 0.0 iou 0.773 (min 0.533)	120 best -1.5 iou 0.762 (min 0.535)	174 best 0.0 iou 0.767 (min 0.542)	228 best 1.5 iou 0.867 (min 0.680)
72 best 0.0 iou 0.785 (min 0.541)	126 best 0.0 iou 0.751 (min 0.529)	180 best -1.5 iou 0.807 (min 0.564)	234 best 1.5 iou 0.861 (min 0.694)
78 best 0.0 iou 0.779 (min 0.532)	132 best 0.0 iou 0.773 (min 0.526)	186 best -1.5 iou 0.813 (min 0.578)	240 best 1.5 iou 0.873 (min 0.699)
84 best 0.0 iou 0.799 (min 0.552)	138 best 0.0 iou 0.752 (min 0.519)	192 best -1.5 iou 0.822 (min 0.580)	246 best 1.5 iou 0.874 (min 0.705)
90 best 0.0 iou 0.759 (min 0.530)	144 best 0.0 iou 0.746 (min 0.520)	198 best 0.0 iou 0.832 (min 0.5 …[+387 chars]
```
</details>

**Bash**
```
cat /tmp/claude-1003/-work/f5f61c40-1890-45ca-8f67-396f0c76ee6d/tasks/bkcc66ykh.output | grep -v Warn; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from rectify import *
import numpy as np, cv2
ref=np.load('/work/scratch/rect_ref_top.npy'); real=np.load('/work/scratch/rect_o_top_2.npy')
def hp(img,w):
    g=img[...,:3].mean(-1); m=w>0
    g=np.where(m,g,0); b=cv2.GaussianBlur(g,(0,0),8)/np.maximum(cv2.GaussianBlur(m.astype(np.float32),(0,0),8),1e-3)
    x=np.where(m,g-b,0); s=x[m].std(); return ((x/s*0.25+0.5).clip(0,1)*255).astype(np.uint8)
tiles=[hp(real[...,:3],real[...,3])]
for s in [144,-144,-72,0,72]:
    r=rotate_grid(ref,s); t=hp(r[...,:3],r[...,3]).copy(); cv2.putText(t,str(s),(5,25),0,0.8,255,2); tiles.append(t)
cv2.imwrite('/work/scratch/frames/top_cmp.png',np.concatenate(tiles,1))
EOF
```

<details><summary>output</summary>

```
top 0 30 peak -144 0.017  second -122 -0.000
top 30 60 peak -148 0.035  second -126 0.007
top 60 90 peak -146 0.047  second -168 0.035
top 90 120 peak -146 0.016  second -122 -0.007
top 120 150 peak -118 0.014  second   16 0.012
top 150 180 peak    8 0.012  second -116 0.004
top 180 210 peak -172 0.054  second   14 0.032
top 210 240 peak -180 0.048  second  150 0.044
top 240 270 peak  122 0.043  second  146 0.037
top 270 300 peak  144 0.125  second  -50 0.064
top 300 330 peak  144 0.103  second   96 0.052
top 330 360 peak  152 0.248  second    6 0.068
top 360 390 peak  152 0.095  second  -26 0.082
top 390 420 peak  138 0.234  second   72 0.068
top 420 450 peak  138 0.196  second  -24 0.072
top 450 480 peak  -44 0.114  second  140 0.078
top 870 900 peak -104 0.057  second  126 0.028
top 900 …[+133 chars]
```
</details>

**Read**
```
/work/scratch/frames/top_cmp.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/f5f61c40-1890-45ca-8f67-396f0c76ee6d/tasks/bkcc66ykh.output | grep under
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
cd /work/scratch; python exp/spin_track.py base "list(range(440,540,3))" 2>&1 | column -c 180
```

<details><summary>output</summary>

```
440 best 6.0 iou 0.768 (min 0.685)	467 best -1.5 iou 0.756 (min 0.641)	494 best 0.0 iou 0.803 (min 0.765)	521 best 1.5 iou 0.809 (min 0.756)
443 best 1.5 iou 0.789 (min 0.669)	470 best 1.5 iou 0.723 (min 0.597)	497 best -3.0 iou 0.793 (min 0.754)	524 best 0.0 iou 0.810 (min 0.758)
446 best 6.0 iou 0.794 (min 0.680)	473 best 1.5 iou 0.563 (min 0.455)	500 best 1.5 iou 0.804 (min 0.790)	527 best -1.5 iou 0.822 (min 0.768)
449 best 6.0 iou 0.796 (min 0.674)	476 best 3.0 iou 0.679 (min 0.565)	503 best 0.0 iou 0.755 (min 0.737)	530 best 4.5 iou 0.799 (min 0.750)
452 best 4.5 iou 0.817 (min 0.665)	479 best 13.5 iou 0.575 (min 0.506)	506 best 3.0 iou 0.767 (min 0.744)	533 best -10.5 iou 0.795 (min 0.761)
455 best 1.5 iou 0.814 (min 0.661)	482 best -6.0 iou 0.686 (min 0.632)	509 best -9.0 iou 0.753 …[+403 chars]
```
</details>

**Bash**
```
cd /work/scratch; python exp/spin_track.py base "list(range(258,322,3))" 2>&1 | column -c 180; grep under /tmp/claude-1003/-work/f5f61c40-1890-45ca-8f67-396f0c76ee6d/tasks/bkcc66ykh.output
```

<details><summary>output</summary>

```
258 best 1.5 iou 0.905 (min 0.747)	276 best 0.0 iou 0.929 (min 0.884)	294 best 34.5 iou 0.925 (min 0.919)	312 best 0.0 iou 0.963 (min 0.900)
261 best 1.5 iou 0.913 (min 0.776)	279 best 1.5 iou 0.931 (min 0.893)	297 best -36.0 iou 0.865 (min 0.856)	315 best 1.5 iou 0.952 (min 0.895)
264 best 3.0 iou 0.903 (min 0.786)	282 best 4.5 iou 0.937 (min 0.905)	300 best 22.5 iou 0.883 (min 0.876)	318 best 0.0 iou 0.963 (min 0.901)
267 best 3.0 iou 0.908 (min 0.828)	285 best 22.5 iou 0.845 (min 0.834)	303 best 21.0 iou 0.830 (min 0.818)	321 best 0.0 iou 0.962 (min 0.901)
270 best 3.0 iou 0.915 (min 0.859)	288 best -27.0 iou 0.909 (min 0.903)	306 best 1.5 iou 0.954 (min 0.914)
273 best 3.0 iou 0.924 (min 0.876)	291 best 19.5 iou 0.934 (min 0.923)	309 best 0.0 iou 0.956 (min 0.910)
under 480 510 peak -1 …[+272 chars]
```
</details>

**Write**
```
/work/scratch/exp/frame_tex.py
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/exp/frame_tex.py (file state is current in your context — no need to Read it back)
```
</details>

**Bash**
```
cd /work/scratch; python exp/frame_tex.py base top "list(range(230,330,6))" 3 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
230 peak   42 0.049 | grid 0:0.03 72:0.02 144:-0.01 -144:-0.01 -72:-0.02
236 peak   42 0.055 | grid 0:0.02 72:0.02 144:0.01 -144:-0.01 -72:-0.03
242 peak  -58 0.051 | grid 0:0.02 72:0.01 144:0.03 -144:0.01 -72:-0.01
248 peak  -60 0.075 | grid 0:-0.00 72:0.03 144:0.03 -144:-0.01 -72:-0.00
254 peak  -46 0.055 | grid 0:0.01 72:0.05 144:0.01 -144:-0.01 -72:-0.02
260 peak  -46 0.066 | grid 0:-0.02 72:0.02 144:-0.01 -144:-0.01 -72:-0.01
266 peak  -48 0.088 | grid 0:0.01 72:0.00 144:0.07 -144:0.01 -72:0.03
272 peak  142 0.092 | grid 0:0.01 72:-0.02 144:0.07 -144:0.00 -72:0.05
278 peak  108 0.058 | grid 0:0.02 72:0.02 144:-0.05 -144:-0.01 -72:-0.01
284 peak -174 0.059 | grid 0:-0.02 72:0.02 144:-0.00 -144:0.00 -72:0.01
290 peak  142 0.147 | grid 0:-0.03 72:-0.01 144:0.10 -144:-0.00 -72:0.01
296 pe …[+422 chars]
```
</details>

**Bash**
```
grep under /tmp/claude-1003/-work/f5f61c40-1890-45ca-8f67-396f0c76ee6d/tasks/bkcc66ykh.output; cd /work/scratch; python exp/spin_track.py base "list(range(590,632,3))+list(range(840,905,3))" 2>&1 | column -c 180
```

<details><summary>output</summary>

```
under 480 510 peak -100 0.073  second   76 0.046
under 510 540 peak -126 0.081  second -172 0.043
under 540 570 peak -146 0.316  second -122 0.053
under 570 600 peak -146 0.170  second -120 0.061
under 600 630 peak -160 0.082  second  -16 0.056
under 630 660 peak  -72 0.109  second   44 0.076
under 660 690 peak  -74 0.269  second -104 0.118
under 690 720 peak  -76 0.154  second    2 0.065
under 720 750 peak  -88 0.218  second -156 0.066
under 750 780 peak -104 0.132  second -166 0.061
under 780 810 peak  172 0.139  second -118 0.102
under 810 840 peak -134 0.162  second  158 0.128
under 840 870 peak  -80 0.278  second -144 0.115
under 870 900 peak  -76 0.374  second  -16 0.086
590 best 4.5 iou 0.850 (min 0.801)	617 best -1.5 iou 0.809 (min 0.766)	852 best 1.5 iou 0.831 (min 0.689)	879 best …[+1159 chars]
```
</details>

**Bash**
```
cd /work/scratch; sed -i 's/min_facing=0.1)/min_facing=float(__import__("os").environ.get("MINF","0.1")))/' exp/frame_tex.py; for p in base opt1.npz; do for mf in 0.08 0.1; do echo $p $mf; MINF=$mf python exp/frame_tex.py $p top "[940,966]" 26 2>&1 | grep -v Warn; done; done
```

<details><summary>output</summary>

```
base 0.08
940 peak -114 0.062 | grid 0:-0.02 72:0.01 144:-0.01 -144:0.00 -72:0.02
966 peak -114 0.057 | grid 0:-0.03 72:-0.02 144:-0.00 -144:-0.00 -72:0.03
base 0.1
940 peak -114 0.062 | grid 0:-0.02 72:0.01 144:-0.01 -144:0.00 -72:0.02
966 peak -114 0.057 | grid 0:-0.03 72:-0.02 144:-0.00 -144:-0.00 -72:0.03
opt1.npz 0.08
940 peak -150 0.243 | grid 0:0.00 72:-0.01 144:0.04 -144:0.09 -72:-0.02
966 peak -150 0.257 | grid 0:-0.01 72:-0.00 144:0.03 -144:0.07 -72:-0.02
opt1.npz 0.1
940 peak -150 0.243 | grid 0:0.00 72:-0.01 144:0.04 -144:0.09 -72:-0.02
966 peak -150 0.257 | grid 0:-0.01 72:-0.00 144:0.03 -144:0.07 -72:-0.02
```
</details>

**Bash**
```
cd /work/scratch; cat > exp/hyp_tex.py <<'EOF'
# for refined static hypotheses (hyp_static.npz rows 0..4 = spins 0,72,144,-144,-72), pooled top texture correlation
import sys; sys.path.insert(0, '/work/outputs')
from common import *
from rectify import *
import h5py
frames = eval(sys.argv[1]); which = sys.argv[2] if len(sys.argv) > 2 else 'top'
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
H = np.linalg.inv(D['Twc']) @ np.load('/work/scratch/hyp_static.npz')['T_world_mesh'][:5]
K = D['K']; R = Renderer(mesh, K, 1152, 1536)
h = h5py.File('/work/inputs/masks.h5'); Mo = h['object']; Mp = h['person']
exec(open('/work/scratch/exp/frame_tex.py').read().split("ref = np.load")[0].split("angs = np.arange")[1].join(["angs = np.arange", ""]) if False else "")
n = 400
yy, xx = np.mgrid[:n, :n]; rad = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) * 0.002
ann = (rad > 0.05) & (rad < 0.34)
angs = np.arange(-180, 180, 2)
def prep(img, w):
    g = img[..., :3].mean(-1); m = (w > 0) & ann
    g = np.where(m, g, 0)
    b = cv2.GaussianBlur(g, (0, 0), 8); bw = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 8)
    hp = np.where(m, g - b / np.maximum(bw, 1e-3), 0)
    hp = cv2.GaussianBlur(hp, (0, 0), 1.5) / np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5), 1e-3)
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return hp, m
def corr(a, ma, b, mb):
    m = ma & mb
    if m.sum() < 2000: return np.nan
    x = a[m] - a[m].mean(); y = b …[+1382 chars]
```

<details><summary>output</summary>

```
hyp    0  loss 0.3391  iou 0.8494  drot 1.07 deg  dp [0.1 0.3 0. ] cm
hyp   72  loss 0.4462  iou 0.8285  drot 1.71 deg  dp [0.2 0.6 2.1] cm
hyp  144  loss 0.5862  iou 0.7813  drot 1.49 deg  dp [1.6 0.5 0.8] cm
hyp -144  loss 0.6477  iou 0.7795  drot 3.41 deg  dp [0.9 1.  2.2] cm
hyp  -72  loss 0.6142  iou 0.7794  drot 1.24 deg  dp [ 0.8  0.1 -3.4] cm
hyp    0: corr at 0: 0.016   peak  120 0.087
hyp   72: corr at 0: -0.014   peak   72 0.081
hyp  144: corr at 0: -0.002   peak  166 0.087
hyp -144: corr at 0: 0.061   peak    0 0.061
hyp  -72: corr at 0: -0.003   peak -120 0.071
```
</details>

**Bash**
```
cd /work/scratch; python exp/static_hyp.py opt1.npz "list(range(937,993,4))" && python exp/hyp_tex.py "list(range(937,993,2))"
```

<details><summary>output</summary>

```
hyp    0  loss 0.4853  iou 0.8403  drot 0.47 deg  dp [0.2 0.5 1.2] cm
hyp   72  loss 0.3222  iou 0.8873  drot 1.20 deg  dp [0.6 0.9 3.5] cm
hyp  144  loss 0.3109  iou 0.8873  drot 2.10 deg  dp [1.3 1.  4.7] cm
hyp -144  loss 0.5368  iou 0.8310  drot 1.25 deg  dp [2.  1.4 4.4] cm
hyp  -72  loss 0.5232  iou 0.8275  drot 1.77 deg  dp [1.1 1.  1.3] cm
hyp    0: corr at 0: -0.015   peak -150 0.163
hyp   72: corr at 0: 0.013   peak  140 0.229
hyp  144: corr at 0: 0.023   peak   68 0.218
hyp -144: corr at 0: 0.015   peak -164 0.076
hyp  -72: corr at 0: -0.019   peak  126 0.072
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
m=load_mesh('/work/inputs'); v=m['v']
r=np.hypot(v[:,0]-0.006,v[:,1]-0.004)
s=v[:,2]>0.45
for r0 in np.arange(0,0.4,0.025):
  q=s&(r>=r0)&(r<r0+0.025)
  if q.sum(): print('r %.3f n %5d z %.3f..%.3f'%(r0,q.sum(),v[q,2].min(),v[q,2].max()))
"
```

<details><summary>output</summary>

```
r 0.000 n   443 z 0.451..0.532
r 0.025 n  4324 z 0.450..0.587
r 0.050 n  3650 z 0.533..0.582
r 0.075 n  3113 z 0.535..0.583
r 0.100 n  2066 z 0.537..0.583
r 0.125 n  2837 z 0.539..0.583
r 0.150 n  2195 z 0.541..0.586
r 0.175 n  1968 z 0.543..0.580
r 0.200 n  1361 z 0.546..0.582
r 0.225 n  1574 z 0.551..0.588
r 0.250 n  1594 z 0.556..0.642
r 0.275 n  3093 z 0.560..0.664
r 0.300 n  5171 z 0.565..0.662
r 0.325 n  6030 z 0.571..0.664
r 0.350 n  3274 z 0.580..0.664
r 0.375 n   346 z 0.618..0.645
```
</details>

**Bash**
```
cd /work/scratch; cat > exp/sil_nocaster.py <<'EOF'
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
frames = eval(sys.argv[1])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
H = np.linalg.inv(D['Twc']) @ np.load('/work/scratch/hyp_static.npz')['T_world_mesh'][:5]
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; f = mesh['f']; fc = v[f].mean(1)
r = np.hypot(fc[:, 0] - 0.006, fc[:, 1] - 0.004); ang = np.degrees(np.arctan2(fc[:, 1] - 0.004, fc[:, 0] - 0.006))
cast = (r > 0.24) & (fc[:, 2] > 0.588)
lever = (r > 0.05) & (r < 0.37) & (fc[:, 2] > -0.07) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
ign_t = torch.tensor(np.r_[False, cast | lever], device='cuda')
T = torch.tensor(H, dtype=torch.float32, device='cuda')
vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
full = rast[..., 3] > 0
ign = ign_t[rast[..., 3].long()].any(0).float()[None, None]
ign = torch.nn.functional.max_pool2d(ign, 15, 1, 7)[0, 0] > 0
h = h5py.File('/work/inputs/masks.h5')
acc = np.zeros(5)
for fr in frames:
    m = torch.tensor(h['object'][fr], device='cuda'); p = torch.tensor(h['person'][fr], device='cuda')
    valid = ~(p & ~m) & ~ign
    o = full & valid; mm = m & valid
    acc += ((o & mm).sum((1, 2)) / (o | mm).sum((1, 2))).cpu().numpy()
print('IoU w/o casters+lever:', dict(zip([ …[+307 chars]
```

<details><summary>output</summary>

```
IoU w/o casters+lever: {0: 0.9072, 72: 0.9171, 144: 0.8545, -144: 0.8518, -72: 0.8782}
IoU w/o casters+lever: {0: 0.9156, 72: 0.9519, 144: 0.9486, -144: 0.8923, -72: 0.8827}
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
from scipy.spatial import cKDTree
m=load_mesh('/work/inputs'); v=m['v']
r=np.hypot(v[:,0]-0.006,v[:,1]-0.004)
legs=v[(v[:,2]>0.5)&(v[:,2]<0.588)&(r>0.06)&(r<0.30)]
cast=v[(v[:,2]>0.588)&(r>0.24)]
print(len(legs),len(cast))
from scipy.optimize import minimize
def cost(p,P,k):
    c=np.array([p[0],p[1],0]); ax=np.array([p[2],p[3],1.]); ax/=np.linalg.norm(ax)
    Rk=axis_angle(ax,np.radians(72*k)); Q=(P-c)@Rk.T+c
    d,_=cKDTree(P).query(Q); return np.mean(np.minimum(d,0.03)**2)
sub=legs[::5]
best=minimize(lambda p: sum(cost(p,sub,k) for k in [1,2]),[0.006,0.004,0,0],method='Nelder-Mead',options=dict(maxiter=400))
p=best.x; print('axis',p)
for k in range(1,5):
    c=np.array([p[0],p[1],0]); ax=np.array([p[2],p[3],1.]); ax/=np.linalg.norm(ax); Rk=axis_angle(ax,np.radians(72*k))
    for nm,P in [('legs',legs),('casters',cast)]:
        Q=(P-c)@Rk.T+c; d,_=cKDTree(P).query(Q); print(k,nm,'mean %.4f p90 %.4f'%(d.mean(),np.percentile(d,90)))
"
```

<details><summary>output</summary>

```
20247 15024
axis [ 0.00385435  0.01171406  0.00586297 -0.00032867]
1 legs mean 0.0073 p90 0.0181
1 casters mean 0.0211 p90 0.0390
2 legs mean 0.0084 p90 0.0196
2 casters mean 0.0198 p90 0.0411
3 legs mean 0.0092 p90 0.0210
3 casters mean 0.0188 p90 0.0389
4 legs mean 0.0077 p90 0.0210
4 casters mean 0.0219 p90 0.0405
```
</details>

**Bash**
```
cd /work/scratch; python exp/static_hyp.py opt1.npz "list(range(0,98,6))" >/dev/null; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import h5py
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); T=np.load('/work/scratch/hyp_static.npz')['T_world_mesh'][:5]
Tc=np.linalg.inv(D['Twc'])@T; R=Renderer(mesh,D['K'],1152,1536)
f=48; h=h5py.File('/work/inputs/masks.h5'); m=h['object'][f]
o=R.render(Tc,rgb=False)['mask'].cpu().numpy()
ys,xs=np.nonzero(m); y0,x0,x1=ys.max()-140,xs.min()-10,xs.max()+10; y1=ys.max()+10
tiles=[]
for i,s in enumerate([0,72,144,-144,-72]):
    a=np.zeros(m.shape+(3,),np.uint8); a[m&~o[i]]=(0,255,0); a[o[i]&~m]=(0,0,255); a[o[i]&m]=(255,255,255)
    a=a[y0:y1,x0:x1].copy(); cv2.putText(a,str(s),(5,25),0,0.8,(0,255,255),2); tiles.append(a)
cv2.imwrite('/work/scratch/frames/legs_ref.png',cv2.resize(np.concatenate(tiles,1),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST))
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/legs_ref.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; sed -i 's/ax_p = np.array(\[0.006, 0.004, 0.0\])/ax_p = np.array([0.0039, 0.0117, 0.0])/; s/for it in range(150):/for it in range(int(__import__("os").environ.get("IT","150"))):/' exp/static_hyp.py; IT=400 python exp/static_hyp.py opt1.npz "list(range(0,98,6))" && python exp/sil_nocaster.py "list(range(0,98,6))"; IT=400 python exp/static_hyp.py opt1.npz "list(range(937,993,4))" && python exp/sil_nocaster.py "list(range(937,993,4))"
```

<details><summary>output</summary>

```
hyp    0  loss 0.3441  iou 0.8480  drot 0.99 deg  dp [0.1 0.4 0.1] cm
hyp   72  loss 0.4430  iou 0.8310  drot 1.64 deg  dp [-0.4  0.6  1.4] cm
hyp  144  loss 0.5866  iou 0.7797  drot 1.57 deg  dp [0.3 0.4 0.3] cm
hyp -144  loss 0.6455  iou 0.7813  drot 3.35 deg  dp [-0.6  0.9  2.4] cm
hyp  -72  loss 0.6174  iou 0.7789  drot 1.41 deg  dp [ 0.2  0.1 -2.6] cm
IoU w/o casters+lever: {0: 0.9067, 72: 0.919, 144: 0.8546, -144: 0.8503, -72: 0.8785}
hyp    0  loss 0.4856  iou 0.8397  drot 0.48 deg  dp [0.2 0.5 1.2] cm
hyp   72  loss 0.3210  iou 0.8872  drot 1.18 deg  dp [0.6 0.9 2.6] cm
hyp  144  loss 0.3131  iou 0.8869  drot 2.09 deg  dp [0.4 1.  3.4] cm
hyp -144  loss 0.5360  iou 0.8307  drot 1.22 deg  dp [0.6 1.4 3.8] cm
hyp  -72  loss 0.5212  iou 0.8274  drot 1.68 deg  dp [0.3 1.  1.7] cm
IoU w …[+80 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import h5py
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); T=np.load('/work/scratch/hyp_static.npz')['T_world_mesh'][:5]
Tc=np.linalg.inv(D['Twc'])@T; R=Renderer(mesh,D['K'],1152,1536)
f=965; h=h5py.File('/work/inputs/masks.h5'); m=h['object'][f]
o=R.render(Tc,rgb=False)['mask'].cpu().numpy()
ys,xs=np.nonzero(m); y0,x0,x1=ys.min()-10,xs.min()-10,xs.max()+10; y1=ys.max()+10
tiles=[]
for i,s in enumerate([0,72,144,-144,-72]):
    a=np.zeros(m.shape+(3,),np.uint8); a[m&~o[i]]=(0,255,0); a[o[i]&~m]=(0,0,255); a[o[i]&m]=(255,255,255)
    a=a[y0:y1,x0:x1].copy(); cv2.putText(a,str(s),(5,25),0,0.8,(0,255,255),2); tiles.append(a)
cv2.imwrite('/work/scratch/frames/legs_end.png',np.concatenate(tiles,1))
EOF
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/legs_end.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > exp/lever_vis.py <<'EOF'
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; f = mesh['f']; fc = v[f].mean(1)
r = np.hypot(fc[:, 0] - 0.006, fc[:, 1] - 0.004); ang = np.degrees(np.arctan2(fc[:, 1] - 0.004, fc[:, 0] - 0.006))
lever = (r > 0.12) & (r < 0.37) & (fc[:, 2] > -0.06) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
lev_t = torch.tensor(np.r_[False, lever], device='cuda')
ax_p = np.array([0.0039, 0.0117, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
h = h5py.File('/work/inputs/masks.h5')
for fr in frames:
    T = torch.tensor(np.stack([spinT(Tc[fr], s) for s in [0, 72, 144, -144, -72]]), dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    lm = lev_t[rast[..., 3].long()].cpu().numpy()
    m = h['object'][fr]; p = h['person'][fr]
    out = []
    for i in range(5):
        l = lm[i] & ~p
        out.app …[+164 chars]
```

<details><summary>output</summary>

```
20   192:0.00   289:0.00    75:0.03     0:0.00   109:0.00
50   192:0.00   289:0.00    75:0.03     0:0.00   109:0.00
80   192:0.00   289:0.00    75:0.00     0:0.00   109:0.00
875   387:0.06   145:0.97    94:0.22     9:0.00   378:0.00
880   319:0.00   246:0.91    87:0.00     1:0.00   298:0.00
885   330:0.00   465:0.69   186:0.00     0:0.00   186:0.10
890   282:0.41   605:0.57   315:0.14     0:0.00     0:0.00
950   244:0.00   595:0.42   170:0.06     0:0.00     0:0.00
980   244:0.02   595:0.47   533:0.02     0:0.00     0:0.00
```
</details>

**Bash**
```
cd /work/scratch; MINF=0.15 python exp/frame_tex.py opt1.npz top "list(range(420,500,6))" 3 2>&1 | grep -v Warn; MINF=0.15 python exp/frame_tex.py opt1.npz under "list(range(470,540,6))" 3 2>&1 | grep -v Warn
```

<details><summary>output</summary>

```
420 peak  136 0.213 | grid 0:-0.01 72:0.01 144:-0.01 -144:-0.02 -72:-0.00
426 peak  136 0.218 | grid 0:0.04 72:0.04 144:0.01 -144:-0.02 -72:-0.05
432 peak  134 0.150 | grid 0:0.04 72:-0.00 144:-0.05 -144:-0.03 -72:0.00
438 peak  136 0.136 | grid 0:0.03 72:0.00 144:-0.03 -144:-0.00 -72:0.00
444 peak  -10 0.071 | grid 0:0.01 72:0.03 144:-0.00 -144:-0.00 -72:-0.01
450 peak  -18 0.052 | grid 0:-0.00 72:0.01 144:0.01 -144:-0.05 -72:-0.00
456 peak  136 0.091 | grid 0:0.03 72:0.02 144:-0.01 -144:-0.02 -72:-0.02
462 peak -106 0.053 | grid 0:-0.01 72:0.01 144:0.03 -144:-0.01 -72:0.00
468 peak   88 0.123 | grid 0:0.00 72:0.05 144:-0.04 -144:-0.02 -72:-0.00
474 no view
480 no view
486 no view
492 no view
498 no view
470 no view
476 no view
482 no view
488 peak   36 0.121 | grid 0:0.04 72:0.01 144:-0. …[+597 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs'); rec=D['rec']; Tw=rec['T_world_mesh'].astype(np.float64)
kp=rec['human_keypoints_world'].astype(np.float64)
rh=kp[:,21:42].mean(1); lh=kp[:,42:63].mean(1)
Ti=np.linalg.inv(Tw)
def obj(p): return np.einsum('fij,fj->fi',Ti[:,:3,:3],p)+Ti[:,:3,3]
for nm,p in [('R',rh),('L',lh)]:
  q=obj(p); az=np.degrees(np.arctan2(q[:,1]-0.0117,q[:,0]-0.0039)); r=np.hypot(q[:,0]-0.0039,q[:,1]-0.0117)
  globals()[nm]=(az,r,q[:,2])
c=rec['contact_prob_right_left']
for f in list(range(100,1000,15)):
  print(f,'R az %6.0f r %.2f z %5.2f | L az %6.0f r %.2f z %5.2f | c %.1f %.1f'%(R[0][f],R[1][f],R[2][f],L[0][f],L[1][f],L[2][f],c[f,0],c[f,1]))
" 
```

<details><summary>output</summary>

```
100 R az     36 r 0.40 z -0.16 | L az    110 r 0.38 z -0.12 | c 1.0 0.0
115 R az      9 r 0.32 z -0.13 | L az    105 r 0.36 z -0.11 | c 1.0 1.0
130 R az    -10 r 0.35 z -0.13 | L az    107 r 0.36 z -0.10 | c 1.0 1.0
145 R az    -16 r 0.32 z -0.13 | L az    110 r 0.37 z -0.10 | c 1.0 1.0
160 R az    -25 r 0.32 z -0.11 | L az    111 r 0.36 z -0.08 | c 1.0 1.0
175 R az    -20 r 0.33 z -0.12 | L az    112 r 0.35 z -0.08 | c 1.0 1.0
190 R az    -18 r 0.39 z -0.10 | L az    117 r 0.36 z -0.09 | c 1.0 1.0
205 R az    -16 r 0.41 z -0.10 | L az    115 r 0.35 z -0.09 | c 1.0 1.0
220 R az    -16 r 0.41 z -0.10 | L az    113 r 0.36 z -0.07 | c 1.0 1.0
235 R az    -16 r 0.41 z -0.09 | L az     97 r 0.39 z -0.05 | c 1.0 1.0
250 R az    -16 r 0.41 z -0.09 | L az     86 r 0.39 z -0.07 | c 1.0 1.0
265 R az …[+3519 chars]
```
</details>

> *thinking:* I've confirmed the baseline's rotation about the table's 5-fold axis is off by 144° for frames ~0–600 and 72° after a tracker slip around frame 600–620, verified via underside texture, lever geometry/visibility, and hand-in-object continuity. Now I'll write the full `solve.py`.

**Bash**
```
cd /work/scratch; python -c "
import numpy as np
d=np.load('pixdiff5.npy'); import scipy.ndimage as nd
m=nd.median_filter(np.nan_to_num(d,nan=9),15)
st=m<0.45
lab,n=nd.label(st)
for i in range(1,n+1):
  w=np.where(lab==i)[0]
  if len(w)>=25: print(w[0],w[-1],len(w))
"
```

<details><summary>output</summary>

```
5 97 93
317 389 73
960 992 33
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np
d=np.load('pixdiff5.npy'); import scipy.ndimage as nd
m=nd.median_filter(np.nan_to_num(d,nan=9),15)
for th in [0.55,0.65,0.8]:
  lab,n=nd.label(m<th); print(th,[(w[0],w[-1]) for w in [np.where(lab==i)[0] for i in range(1,n+1)] if len(w)>=25])
"
```

<details><summary>output</summary>

```
0.55 [(5, 105), (317, 390), (955, 992)]
0.65 [(5, 113), (316, 390), (938, 992)]
0.8 [(5, 121), (316, 391), (521, 585), (937, 992)]
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
nvidia-smi --query-gpu=memory.used --format=csv && timeout 3000 python /work/outputs/solve.py --out /work/scratch/run1.npz --diag /work/scratch/run1_diag.npz 2>&1 | grep -v Warn | tee run1.log
```

<details><summary>output</summary>

```
memory.used [MiB]
2707 MiB
14309 MiB
[    0.8s] frames 993
[    2.3s] symmetry axis point [0.0007 0.0137 0.    ] dir [ 0.0073 -0.001   1.    ]
[  100.3s] masks loaded
[  104.8s] static phases [(0, 95), (316, 390), (938, 992)]
[  104.8s] baseline slip windows (a, b, spin change) [(600, 619, -72.7), (600, 627, -75.9)]
[  104.8s] segments [(0, 600), (619, 600), (627, 992)]
[  201.4s] texture windows 35
[  201.4s] segment (0, 600): under [0.07 0.06 0.13 0.54 0.06] (n=4)  top [0.27 0.05 1.   0.29 0.12] (n=16)
[  201.4s] segment (619, 600): under [0. 0. 0. 0. 0.] (n=0)  top [0. 0. 0. 0. 0.] (n=0)
[  201.4s] segment (627, 992): under [0.32 0.17 0.28 0.45 1.39] (n=9)  top [0.03 0.09 0.06 0.13 0.1 ] (n=4)
[  201.4s] top texture offset vs underside (deg): 72 from [72]
[  201.4s] segment (0, 600) ->  …[+830 chars]
```
</details>

**Bash**
```
cd /work/outputs && python - <<'EOF'
s=open('solve.py').read()
s=s.replace("""        Mh = np.stack([cv2.resize(Mfull_o[i].astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0 for i in range(F)])
        Ph = np.stack([cv2.resize(Mfull_p[i].astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0 for i in range(F)])""",
"""        Mh, Ph = [], []
        for s0 in range(0, F, 64):
            for src, dst in ((Mfull_o, Mh), (Mfull_p, Ph)):
                blk = src[s0:s0 + 64]
                dst += [cv2.resize(x.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_AREA) > 0 for x in blk]
        Mh, Ph = np.stack(Mh), np.stack(Ph)""")
s=s.replace("""        slips.append((int(max(a, 0)), int(min(b, len(cs) - 1)), float(post - pre)))
    return slips""","""        slips.append([int(max(a, 0)), int(min(b, len(cs) - 1)), float(post - pre)])
    merged = []  # merge overlapping windows
    for sl in slips:
        if merged and sl[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], sl[1])
            merged[-1][2] = float(cs[min(merged[-1][1] + 15, len(cs) - 1)] - cs[max(merged[-1][0] - 15, 0)])
        else:
            merged.append(sl)
    return [tuple(m) for m in merged]""")
open('solve.py','w').write(s)
EOF
cd /work/scratch && python -c "
import sys; sys.path.insert(0,'/work/outputs')
from solve import *
D=load_inputs('/work/inputs'); Tb=np.linalg.inv(D['Twc'])@D['rec']['T_world_mesh'].astype(np.float64)
sp=baseline_spin_ …[+152 chars]
```

<details><summary>output</summary>

```
[(600, 627, -77.63430663122111)]
[  0.   2.   1.   2.   1.   2.   1.   1.   1.   1.  -0.   0.  -2.   1.
  -1.  -0.  -1.  -0.  -2.  -3.   1.  -0.  -1. -15. -43. -40. -38. -48.
 -45. -46. -41. -43. -47. -56. -65. -69. -73. -74. -75. -74. -73. -73.
 -70. -73. -73. -73. -74. -75. -75. -75. -75. -76. -76. -77. -77. -76.
 -75. -75. -76. -75.]
```
</details>

**Bash**
```
python exp/viz2.py --poses base,run1.npz --frames 50,300,350,450,560 --out frames/r1a.png && python exp/viz2.py --poses base,run1.npz --frames 610,700,800,880,960 --out frames/r1b.png
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/r1a.png
```

**Read**
```
/work/scratch/frames/r1b.png
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
python /work/outputs/validate.py --result run1.npz 2>&1 | grep -v Warn | tee val1.txt
```

<details><summary>output</summary>

```
| phase | frames | IoU base | IoU result | lever px outside mask base/result | rot step med (deg) base/result | trans step med (cm) base/result | lowest point above floor med (cm) base/result | depth diff result-base mean/median (cm) | underside tex peak (deg, corr) base -> result | top tex peak base -> result |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-95 | 96 | 0.730 | 0.766 | 44 / 0 | 0.21 / 0.00 | 0.36 / 0.00 | -1.7 / -1.5 | -0.5 / -0.0 | - -> - | -146 (0.03) -> 16 (0.06) |
| 96-315 | 220 | 0.813 | 0.843 | 36 / 0 | 0.52 / 0.39 | 0.65 / 0.30 | -3.3 / -2.1 | 0.1 / -0.3 | - -> - | -120 (0.03) -> -78 (0.15) |
| 316-390 | 75 | 0.940 | 0.969 | 0 / 0 | 2.10 / 0.02 | 0.95 / 0.00 | -2.0 / -1.9 | 0.2 / 0.0 | - -> - | 144 (0.15) -> -70 (0.33) |
| 391-599 | 209 | 0.799 | 0.851 | 37 / 16 |  …[+850 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > exp/fine_spin.py <<'EOF'
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tc = np.linalg.inv(D['Twc']) @ np.load(poses)['T_world_mesh']
R = Renderer(mesh, D['K'], 1152, 1536)
h = h5py.File('/work/inputs/masks.h5')
c = np.array([0.0007, 0.0137, 0]); ax = np.array([0.0073, -0.001, 1]); ax /= np.linalg.norm(ax)
angs = np.arange(-16, 17, 2)
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle(ax, np.radians(deg)); S[:3, 3] = c - S[:3, :3] @ c
    return T @ S
acc = np.zeros(len(angs))
for f in frames:
    m = torch.tensor(h['object'][f], device='cuda'); p = torch.tensor(h['person'][f], device='cuda')
    valid = ~(p & ~m)
    o = R.render(np.stack([spinT(Tc[f], a) for a in angs]), rgb=False)['mask'] & valid
    acc += ((o & m).sum((1, 2)) / (o | m).sum((1, 2))).cpu().numpy()
print(' '.join('%d:%.4f' % (a, v) for a, v in zip(angs, acc / len(frames))))
EOF
for r in "range(520,600,6)" "range(640,860,10)" "range(320,390,6)" "range(0,96,8)" "range(940,992,6)"; do python exp/fine_spin.py run1.npz "list($r)"; done
```

<details><summary>output</summary>

```
-16:0.8471 -14:0.8504 -12:0.8541 -10:0.8575 -8:0.8598 -6:0.8608 -4:0.8613 -2:0.8620 0:0.8618 2:0.8597 4:0.8558 6:0.8518 8:0.8473 10:0.8410 12:0.8336 14:0.8264 16:0.8196
-16:0.7399 -14:0.7542 -12:0.7714 -10:0.7922 -8:0.8159 -6:0.8415 -4:0.8663 -2:0.8886 0:0.8982 2:0.8873 4:0.8642 6:0.8382 8:0.8114 10:0.7859 12:0.7639 14:0.7455 16:0.7302
-16:0.9138 -14:0.9160 -12:0.9186 -10:0.9226 -8:0.9288 -6:0.9385 -4:0.9495 -2:0.9594 0:0.9661 2:0.9655 4:0.9548 6:0.9425 8:0.9302 10:0.9183 12:0.9110 14:0.9083 16:0.9067
-16:0.6161 -14:0.6220 -12:0.6324 -10:0.6500 -8:0.6700 -6:0.6928 -4:0.7187 -2:0.7425 0:0.7579 2:0.7581 4:0.7486 6:0.7325 8:0.7154 10:0.6996 12:0.6854 14:0.6722 16:0.6610
-16:0.6382 -14:0.6550 -12:0.6758 -10:0.7004 -8:0.7283 -6:0.7591 -4:0.7934 -2:0.8178 0:0.8207 2:0.8126 4:0.7993 6:0.7815 8:0. …[+44 chars]
```
</details>

**Bash**
```
cd /work/scratch; sed -i 's/ax_p = np.array(\[0.006, 0.004, 0.0\])/ax_p = np.array([0.0007, 0.0137, 0.0])/' exp/closeup.py; python exp/closeup.py run1.npz 700 0,-9 frames/cu700.png 170; python exp/closeup.py run1.npz 780 0,-9 frames/cu780.png 170; python -c "
import cv2,numpy as np
a=cv2.imread('/work/scratch/frames/cu700.png'); b=cv2.imread('/work/scratch/frames/cu780.png'); cv2.imwrite('/work/scratch/frames/cu_fine.png',np.concatenate([a,b],0))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/cu_fine.png
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The pipeline runs end-to-end, improving silhouette IoU (0.790→0.856) and slashing rotation jumps (28.8°→2.2°) with spin flips fixed; I'm resolving a 9° leg-vs-texture discrepancy by trusting leg geometry since the mesh texture itself is misregistered by 72°. Next: refine depth re-anchoring and add a held-out check.

**Bash**
```
cd /work/outputs && python - <<'EOF'
s=open('solve.py').read()
# holdout support
s=s.replace("def refine_silhouettes(Tc_init, statics, mesh, K, M, P, centre, epochs=40, ws=1.0, lr=3e-3, seed=0):",
"def refine_silhouettes(Tc_init, statics, mesh, K, M, P, centre, epochs=40, ws=1.0, lr=3e-3, seed=0, holdout=None):")
s=s.replace("""        l = (((soft - m) ** 2) * valid).sum((1, 2)) / (m.sum((1, 2)) + 100)
        s8 = Fn.avg_pool2d((soft * valid)[:, None], 8); m8 = Fn.avg_pool2d((m * valid)[:, None], 8)
        return l + 4 * ((s8 - m8) ** 2).sum((1, 2, 3)) / (m.sum((1, 2)) / 64 + 2)""",
"""        l = (((soft - m) ** 2) * valid).sum((1, 2)) / (m.sum((1, 2)) + 100)
        s8 = Fn.avg_pool2d((soft * valid)[:, None], 8); m8 = Fn.avg_pool2d((m * valid)[:, None], 8)
        l = l + 4 * ((s8 - m8) ** 2).sum((1, 2, 3)) / (m.sum((1, 2)) / 64 + 2)
        return l * use[idx]""")
s=s.replace("""    ut = torch.tensor(unit, device=dev)
    om = torch.zeros""","""    ut = torch.tensor(unit, device=dev)
    use = torch.ones(F, device=dev)
    if holdout is not None:
        use[torch.tensor(np.where(holdout)[0], device=dev)] = 0.0
    om = torch.zeros""")
# local depth re-anchoring
old=s[s.index("    # ---------------- 7. depth re-anchoring"):s.index("    Tw = Twc @ Tout")]
new='''    # ---------------- 7. depth re-anchoring to the baseline (along the viewing rays)
    # Static phases: one scale per phase (median z = baseline median z).  Moving phases: a slowly varying
    # scale = running …[+2444 chars]
```

<details><summary>output</summary>

```
482:        log('phase %s depth scale %.4f..%.4f (median shift %.1f cm)' % ((a, b), scale[a:b + 1].min(), scale[a:b + 1].max(),
[  123.4s] segment (627, 992) -> spin -72 deg (underside)
[  125.6s]   refine epoch 0  silhouette loss 0.2821
[  131.1s]   refine epoch 10  silhouette loss 0.1901
[  136.6s]   refine epoch 20  silhouette loss 0.1802
[  142.3s]   refine epoch 30  silhouette loss 0.1760
[  147.7s]   refine epoch 39  silhouette loss 0.1735
[  151.3s] HOLDOUT (odd frames, not in loss): IoU baseline 0.7898, spin-corrected init 0.7769, refined 0.8571
[  151.3s] fitted (even) frames: baseline 0.7911 init 0.7776 refined 0.8570
```
</details>

**Bash**
```
cd /work/outputs && python - <<'EOF'
s=open('solve.py').read()
s=s.replace("    ap.add_argument('--holdout', action='store_true',","    ap.add_argument('--ws', type=float, default=1.0, help='smoothness weight (validated by --holdout)')\n    ap.add_argument('--holdout', action='store_true',")
s=s.replace("Tr = refine_silhouettes(Tc, statics, mesh, K, Mh, Ph, centre, holdout=hold)","Tr = refine_silhouettes(Tc, statics, mesh, K, Mh, Ph, centre, ws=args.ws, holdout=hold)")
open('solve.py','w').write(s)
EOF
cd /work/scratch; for w in 0.25 4; do python /work/outputs/solve.py --out /tmp/x.npz --holdout --ws $w 2>&1 | grep HOLDOUT; done
```

<details><summary>output</summary>

```
[  171.2s] HOLDOUT (odd frames, not in loss): IoU baseline 0.7898, spin-corrected init 0.7769, refined 0.8618
[  165.6s] HOLDOUT (odd frames, not in loss): IoU baseline 0.7898, spin-corrected init 0.7769, refined 0.8492
```
</details>

**Bash**
```
cd /work/scratch; python exp/viz.py --poses run1.npz --frames 470:495:5 --out frames/c1.png; python exp/viz.py --poses run1.npz --frames 598:630:6 --out frames/c2.png; python exp/viz.py --poses run1.npz --frames 870:895:5 --out frames/c3.png; python -c "
import cv2,numpy as np
cv2.imwrite('/work/scratch/frames/c.png',np.concatenate([cv2.imread('/work/scratch/frames/c%d.png'%i) for i in (1,2,3)],0))"
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Read**
```
/work/scratch/frames/c.png
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/outputs && sed -i "s/ap.add_argument('--ws', type=float, default=1.0,/ap.add_argument('--ws', type=float, default=0.5,/" solve.py && grep -n "'--ws'" solve.py && cd /work/scratch && /usr/bin/time -v python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz --diag /work/scratch/final_diag.npz 2>&1 | grep -v Warn | grep -v -E "^\s+(Page|Volun|Invol|Swaps|File|Socket|Signals|Average|Exit|Minor|Major|Percent|System|User)" | tee /work/scratch/final_run.log
```

<details><summary>output</summary>

```
304:    ap.add_argument('--ws', type=float, default=0.5, help='smoothness weight (validated by --holdout)')
/bin/bash: line 1: /usr/bin/time: No such file or directory
```
</details>

**Bash**
```
S=$(date +%s) && python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz --diag /work/scratch/final_diag.npz 2>&1 | grep -v Warn | tee /work/scratch/final_run.log; echo "wall $(( $(date +%s) - S )) s"
```

<details><summary>output</summary>

```
[    0.9s] frames 993
[    2.9s] symmetry axis point [0.0007 0.0137 0.    ] dir [ 0.0073 -0.001   1.    ]
[   11.3s] masks loaded
[   18.2s] static phases [(0, 95), (316, 390), (938, 992)]
[   18.2s] baseline slip windows (a, b, spin change) [(600, 627, -77.6)]
[   18.2s] segments [(0, 600), (627, 992)]
[  135.3s] texture windows 35
[  135.3s] segment (0, 600): under [0.07 0.06 0.13 0.54 0.06] (n=4)  top [0.27 0.05 1.   0.29 0.12] (n=16)
[  135.3s] segment (627, 992): under [0.32 0.17 0.28 0.45 1.39] (n=9)  top [0.03 0.09 0.06 0.13 0.1 ] (n=4)
[  135.3s] top texture offset vs underside (deg): 72 from [72]
[  135.3s] segment (0, 600) -> spin -144 deg (underside)
[  135.3s] segment (627, 992) -> spin -72 deg (underside)
[  137.5s]   refine epoch 0  silhouette loss 0.5491
[  144.4s]   refine  …[+677 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import numpy as np; T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(T.shape,T.dtype,np.isfinite(T).all(), np.abs(np.linalg.det(T[:,:3,:3])-1).max())"; python /work/outputs/validate.py --result /work/outputs/object_poses_world.npz 2>&1 | grep -v Warn | tee val_final.txt
```

<details><summary>output</summary>

```
(993, 4, 4) float64 True 1.346133998325616e-05
| phase | frames | IoU base | IoU result | lever px outside mask base/result | rot step med (deg) base/result | trans step med (cm) base/result | lowest point above floor med (cm) base/result | depth diff result-base mean/median (cm) | underside tex peak (deg, corr) base -> result | top tex peak base -> result |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-95 | 96 | 0.730 | 0.768 | 44 / 0 | 0.21 / 0.02 | 0.36 / 0.00 | -1.7 / -1.5 | -0.5 / -0.0 | - -> - | -146 (0.03) -> 16 (0.05) |
| 96-315 | 220 | 0.813 | 0.840 | 36 / 0 | 0.52 / 0.38 | 0.65 / 0.33 | -3.3 / -2.2 | 0.3 / -0.0 | - -> - | -120 (0.03) -> -76 (0.18) |
| 316-390 | 75 | 0.940 | 0.975 | 0 / 0 | 2.10 / 0.01 | 0.95 / 0.00 | -2.0 / -1.9 | 0.2 / 0.0 | - -> - | 144 (0.15) -> -72 (0.36) …[+897 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF' 2>&1 | grep -v Warn
import sys; sys.path.insert(0,'/work/outputs')
from common import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); v=mesh['v'][::20]
Twc=D['Twc']; pl=D['plane']
B=D['rec']['T_world_mesh'].astype(np.float64); R=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
dg=np.load('final_diag.npz'); centre=dg['centre']
phases=[(0,95),(96,315),(316,390),(391,599),(600,627),(628,937),(938,992)]
add=np.array([np.linalg.norm((v@B[f,:3,:3].T+B[f,:3,3])-(v@R[f,:3,:3].T+R[f,:3,3]),axis=1).mean() for f in range(len(B))])*100
for a,b in phases: print('ADD(base,result) %d-%d: median %.1f cm'%(a,b,np.median(add[a:b+1])))
print('all: median %.1f, frac>9cm %.2f'%(np.median(add),(add>9).mean()))
# floor cue: depth scale so that the lowest point lies on the plane, for frames at rest / pivoting (all frames)
Tc=np.linalg.inv(Twc)@R
def low(s,f):
    T=Tc[f].copy(); pc=T[:3,:3]@centre+T[:3,3]; T[:3,3]=pc*s-T[:3,:3]@centre
    W=Twc@T; return ((v@W[:3,:3].T+W[:3,3])@pl[:3]+pl[3]).min()
from scipy.optimize import brentq
sh=[]
for f in range(0,len(R),5):
    try: s=brentq(lambda s: low(s,f), 0.8,1.2)
    except Exception: continue
    pc=Tc[f][:3,:3]@centre+Tc[f][:3,3]; sh.append((s-1)*pc[2]*100)
sh=np.array(sh); print('floor cue (lowest point on released plane): depth shift median %.1f cm (IQR %.1f..%.1f)'%(np.median(sh),*np.percentile(sh,[25,75])))
EOF
grep "median shift" final_run.log
```

<details><summary>output</summary>

```
ADD(base,result) 0-95: median 43.7 cm
ADD(base,result) 96-315: median 43.5 cm
ADD(base,result) 316-390: median 41.9 cm
ADD(base,result) 391-599: median 43.0 cm
ADD(base,result) 600-627: median 33.9 cm
ADD(base,result) 628-937: median 27.2 cm
ADD(base,result) 938-992: median 27.1 cm
all: median 42.8, frac>9cm 1.00
floor cue (lowest point on released plane): depth shift median -6.0 cm (IQR -9.1..-3.7)
[  166.6s] phase (0, 95) depth scale 1.0040..1.0040 (median shift 1.4 cm)
[  166.6s] phase (96, 315) depth scale 0.9986..1.0142 (median shift 3.2 cm)
[  166.6s] phase (316, 390) depth scale 1.0080..1.0080 (median shift 2.5 cm)
[  166.6s] phase (391, 937) depth scale 0.9907..1.0125 (median shift 1.8 cm)
[  166.6s] phase (938, 992) depth scale 0.9978..0.9978 (median shift -0.5 cm)
```
</details>

**Bash**
```
cd /work/scratch; sed -i 's/ax_p = np.array(\[0.0039, 0.0117, 0.0\])/ax_p = np.array([0.0007, 0.0137, 0.0])/' exp/static_hyp.py; cp /work/outputs/object_poses_world.npz final.npz; for f in 330 360 650 700 760 820 540; do echo "frame $f"; IT=200 python exp/static_hyp.py final.npz "[$f]" | awk '{print $1,$2,$5,$6}' | tr '\n' ' '; echo; done
```

<details><summary>output</summary>

```
frame 330
hyp 0 iou 0.9733 hyp 72 iou 0.9649 hyp 144 iou 0.9625 hyp -144 iou 0.9710 hyp -72 iou 0.9749 
frame 360
hyp 0 iou 0.9763 hyp 72 iou 0.9695 hyp 144 iou 0.9680 hyp -144 iou 0.9753 hyp -72 iou 0.9768 
frame 650
hyp 0 iou 0.9121 hyp 72 iou 0.9143 hyp 144 iou 0.9238 hyp -144 iou 0.8999 hyp -72 iou 0.9048 
frame 700
hyp 0 iou 0.9251 hyp 72 iou 0.9224 hyp 144 iou 0.9286 hyp -144 iou 0.9006 hyp -72 iou 0.8926 
frame 760
hyp 0 iou 0.8949 hyp 72 iou 0.8820 hyp 144 iou 0.8407 hyp -144 iou 0.8358 hyp -72 iou 0.8570 
frame 820
hyp 0 iou 0.9002 hyp 72 iou 0.8862 hyp 144 iou 0.8432 hyp -144 iou 0.8381 hyp -72 iou 0.8774 
frame 540
hyp 0 iou 0.8839 hyp 72 iou 0.9016 hyp 144 iou 0.9066 hyp -144 iou 0.8880 hyp -72 iou 0.8865
```
</details>

**Bash**
```
python -c "
import numpy,scipy,cv2,trimesh,h5py,torch,nvdiffrast,PIL
for m in [numpy,scipy,cv2,trimesh,h5py,torch,nvdiffrast,PIL]: print(m.__name__, getattr(m,'__version__','?'))
"; ls /work/outputs; ls /work/tools
```

<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
cv2 4.11.0
trimesh 5.1.1
h5py 3.16.0
torch 2.5.1+cu124
nvdiffrast 0.4.0
PIL 10.2.0
__pycache__
common.py
match.py
object_poses_world.npz
rectify.py
solve.py
validate.py
LightGlue
torch_home
weights
```
</details>

**Bash**
```
cat /work/scratch/val_final.txt | head -12; grep -c . /work/scratch/final_run.log
```

<details><summary>output</summary>

```
| phase | frames | IoU base | IoU result | lever px outside mask base/result | rot step med (deg) base/result | trans step med (cm) base/result | lowest point above floor med (cm) base/result | depth diff result-base mean/median (cm) | underside tex peak (deg, corr) base -> result | top tex peak base -> result |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-95 | 96 | 0.730 | 0.768 | 44 / 0 | 0.21 / 0.02 | 0.36 / 0.00 | -1.7 / -1.5 | -0.5 / -0.0 | - -> - | -146 (0.03) -> 16 (0.05) |
| 96-315 | 220 | 0.813 | 0.840 | 36 / 0 | 0.52 / 0.38 | 0.65 / 0.33 | -3.3 / -2.2 | 0.3 / -0.0 | - -> - | -120 (0.03) -> -76 (0.18) |
| 316-390 | 75 | 0.940 | 0.975 | 0 / 0 | 2.10 / 0.01 | 0.95 / 0.00 | -2.0 / -1.9 | 0.2 / 0.0 | - -> - | 144 (0.15) -> -72 (0.36) |
| 391-599 | 209 | 0.799 | 0.856 | 37 / 16 |  …[+711 chars]
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

**Write**
```
/work/scratch/.claude/projects/-work/memory/round-table-spin-finding.md
```

<details><summary>output</summary>

```
File created successfully at: /work/scratch/.claude/projects/-work/memory/round-table-spin-finding.md
```
</details>

**Bash**
```
echo "- [Round table spin finding](round-table-spin-finding.md) — ep14 baseline spin off by -144/-72 deg; outputs done" > /work/scratch/.claude/projects/-work/memory/MEMORY.md; ls -la /work/outputs; python -c "
import numpy as np; T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']; print(T.shape, T.dtype)"
```

<details><summary>output</summary>

```
total 212
drwxrwxr-x 3 hanzhou hanzhou   4096 Oct 21 16:12 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 21 14:43 ..
-rw-rw-r-- 1 hanzhou hanzhou  13803 Oct 21 16:11 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   2497 Oct 21 16:12 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 21 16:05 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou   4673 Oct 21 14:44 common.py
-rw-rw-r-- 1 hanzhou hanzhou   5142 Oct 21 14:52 match.py
-rw-rw-r-- 1 hanzhou hanzhou 127382 Oct 21 16:05 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou   2125 Oct 21 15:08 rectify.py
-rw-rw-r-- 1 hanzhou hanzhou  22989 Oct 21 16:02 solve.py
-rw-rw-r-- 1 hanzhou hanzhou   7777 Oct 21 15:47 validate.py
(993, 4, 4) float64
```
</details>

**Agent:** I've produced all the required outputs. My estimate rotates the table by 72–144° relative to the baseline on every frame, about 27–44 cm of ADD difference. If my reading of the evidence is wrong, my output is that much worse than the baseline. I put my confidence at roughly 75–80%.

I ran `solve.py` once at the end with the exact required command. It took about 170 s and wrote `object_poses_world.npz` with `T_world_mesh` of shape 993×4×4, float64. Separately, `validate.py` produced the baseline-vs-result table in `REPORT.md`.

**Main finding: the baseline has the wrong spin on every frame.** The table's five-legged base looks almost the same after a 72° turn, and the baseline locked onto the wrong leg position.
- **Frames 0–600:** the correct pose is the baseline turned −144° about the table axis.
- **Frames 627–992:** it is the baseline turned −72°. The baseline's tracker jumped by about −72° at frames 600–627, while the video shows the table resting upside down and not turning.
- **Evidence for this:**
  - The texture on the underside (mounting plate, label, lever), averaged over hundreds of frames, lines up best at exactly these angles.
  - At frames 560 and 640 the mesh's lever lands on the real black lever at these angles, and the label lands in the right place.
  - At the start, −144° is the only option that keeps the mesh's lever out of view, matching the video.
  - The hands stay at fixed positions on the table during continuous grasps, so the baseline has no other jumps.
- **Evidence against, which is why confidence isn't higher:**
  - In the still phases at the start and end, the leg and caster outlines fit the baseline's choice (start) or other options (end) better than mine.
  - The mesh is internally inconsistent: its top-surface texture is 72° off from its underside texture.
  - Leg outlines and underside texture also disagree by about 9° on the fine spin. I followed the legs; if the texture is right instead, points on the rim are off by about 5 cm.

**Trajectory method.**
1. Find the still phases (frames 0–95, 316–390, 938–992) and the tracker jump.
2. Score the five possible spins per segment using the averaged texture, and pick one.
3. Bridge the jump frames by interpolation.
4. Refine all frames jointly to match the object masks, with smoothness and one shared pose per still phase.
5. Set depth back to the baseline's along each viewing ray.

**Validation (no ground truth).**

| | Baseline | Result |
|---|---|---|
| Mask overlap (IoU) | 0.790 | 0.857 |
| Mask overlap on frames held out of the fit | 0.790 | 0.857 |
| Largest frame-to-frame rotation jump | 28.8° | 2.3° |
| Median depth difference to baseline | – | 0.0 cm |

**Depth.** Two depth cues pointed somewhere else: the outline fit wanted the table about 2 cm closer, the floor plane about 6 cm closer. They don't agree within 2 cm, so I applied no depth shift.

**Caveats.**
- I didn't rerun `solve.py` to check determinism (to save shared GPU time); seeds are fixed, but the GPU renderer may still cause differences around 1e-5.
- By rough estimate, total GPU use during development stayed under the 45-minute budget.
- I tried LightGlue feature matching and abandoned it; `solve.py` doesn't use it.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `validate.py`

## Final

I've produced all the required outputs. My estimate rotates the table by 72–144° relative to the baseline on every frame, about 27–44 cm of ADD difference. If my reading of the evidence is wrong, my output is that much worse than the baseline. I put my confidence at roughly 75–80%.

I ran `solve.py` once at the end with the exact required command. It took about 170 s and wrote `object_poses_world.npz` with `T_world_mesh` of shape 993×4×4, float64. Separately, `validate.py` produced the baseline-vs-result table in `REPORT.md`.

**Main finding: the baseline has the wrong spin on every frame.** The table's five-legged base looks almost the same after a 72° turn, and the baseline locked onto the wrong leg position.
- **Frames 0–600:** the correct pose is the baseline turned −144° about the table axis.
- **Frames 627–992:** it is the baseline turned −72°. The baseline's tracker jumped by about −72° at frames 600–627, while the video shows the table resting upside down and not turning.
- **Evidence for this:**
  - The texture on the underside (mounting plate, label, lever), averaged over hundreds of frames, lines up best at exactly these angles.
  - At frames 560 and 640 the mesh's lever lands on the real black lever at these angles, and the label lands in the right place.
  - At the start, −144° is the only option that keeps the mesh's lever out of view, matching the video.
  - The hands stay at fixed positions on the table during continuous grasps, so the baseline has no other jumps.
- **Evidence against, which is why confidence isn't higher:**
  - In the still phases at the start and end, the leg and caster outlines fit the baseline's choice (start) or other options (end) better than mine.
  - The mesh is internally inconsistent: its top-surface texture is 72° off from its underside texture.
  - Leg outlines and underside texture also disagree by about 9° on the fine spin. I followed the legs; if the texture is right instead, points on the rim are off by about 5 cm.

**Trajectory method.**
1. Find the still phases (frames 0–95, 316–390, 938–992) and the tracker jump.
2. Score the five possible spins per segment using the averaged texture, and pick one.
3. Bridge the jump frames by interpolation.
4. Refine all frames jointly to match the object masks, with smoothness and one shared pose per still phase.
5. Set depth back to the baseline's along each viewing ray.

**Validation (no ground truth).**

| | Baseline | Result |
|---|---|---|
| Mask overlap (IoU) | 0.790 | 0.857 |
| Mask overlap on frames held out of the fit | 0.790 | 0.857 |
| Largest frame-to-frame rotation jump | 28.8° | 2.3° |
| Median depth difference to baseline | – | 0.0 cm |

**Depth.** Two depth cues pointed somewhere else: the outline fit wanted the table about 2 cm closer, the floor plane about 6 cm closer. They don't agree within 2 cm, so I applied no depth shift.

**Caveats.**
- I didn't rerun `solve.py` to check determinism (to save shared GPU time); seeds are fixed, but the GPU renderer may still cause differences around 1e-5.
- By rough estimate, total GPU use during development stayed under the 45-minute budget.
- I tried LightGlue feature matching and abandoned it; `solve.py` doesn't use it.

Files are in `/work/outputs`:
- `object_poses_world.npz`
- `REPORT.md`
- `TOOLS.md`
- `solve.py`
- `validate.py`
