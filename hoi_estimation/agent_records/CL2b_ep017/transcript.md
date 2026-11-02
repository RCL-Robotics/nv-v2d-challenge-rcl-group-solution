# CL2b_ep017

Claude Opus v2（ep11-21）  
模型: claude-opus-5-5 effort=high net=open  
日期: 2026-10-21T11:58:00Z

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
cd /work; ls -la inputs logs outputs scratch .net; cat inputs/task.txt inputs/episode.json inputs/camera.json inputs/ground_plane.json; ls /work/scratch/.claude/projects/-work/memory/ 2>/dev/null && cat /work/scratch/.claude/projects/-work/memory/MEMORY.md 2>/dev/null; nvidia-smi; ls /pipeline /weights /code/reconstruction
```

<details><summary>output</summary>

```
.net:
total 12
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 21 15:51 .
drwxrwxr-x 7 hanzhou hanzhou 4096 Oct 21 15:51 ..
-rw-rw-r-- 1 hanzhou hanzhou 3421 Oct 21 15:51 allowlist_proxy.py
srwxrwxr-x 1 hanzhou hanzhou    0 Oct 21 15:51 proxy.sock

inputs:
total 51384
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 21 15:51 .
drwxrwxr-x 7 hanzhou hanzhou     4096 Oct 21 15:51 ..
-rw-rw-r-- 1 hanzhou hanzhou      824 Oct 21 15:51 camera.json
-rw-rw-r-- 1 hanzhou hanzhou       67 Oct 21 15:51 episode.json
-rw-rw-r-- 1 hanzhou hanzhou   195260 Oct 21 15:51 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou      395 Oct 21 15:51 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou  4577166 Oct 21 15:51 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou 11317244 Oct 21 15:51 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou 2232 …[+4608 chars]
```
</details>

**Bash**
```
cd /work; echo $CUDA_VISIBLE_DEVICES; cat logs/run_manifest.txt; cat /pipeline/pipeline_report.json | head -50; ls -R /pipeline | head -80; python -c "import torch;print(torch.__version__, torch.cuda.device_count())"
```

<details><summary>output</summary>

```
0
date_utc: 2026-10-21T11:58:00Z
claude: 2.1.303 (Claude Code)
model: claude-opus-5-5 effort=high net=open
sif: /mnt/nas/Group/hanzhou/v2d/images/v2d_cari4d.sif 16860184576 bytes
binds: -B /mnt/nas/Group/hanzhou/v2d/agent_runs/CL2b_ep017:/work -B /home/hanzhou/.local/share/claude/versions/2.1.303:/opt/claude/claude:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B /mnt/nas/Group/hanzhou/v2d/weights:/weights:ro -B /mnt/nas/Group/hanzhou/v2d/outputs/cari4d/episode_000017:/pipeline:ro
task_sha256: 5d5091be818abc88b4513b225931c3f47ccc2a8cedb94b5d4692de619aa5b7a8
5611ad0e5238aa3dbc6fe3cba2e2d157f6a6f571b109a9ba4336878949afaf80  ./camera.json
ae4c1cc5c730acbb2d333f3e53cdecfd006ebf09dadf8e4322061d245f9b1994  ./episode.json
56bd2491278e16a8075146513886878a9ec6e8f2f97db691d7dba …[+3563 chars]
```
</details>

**Bash**
```
cd /work; python -c "
import numpy as np, trimesh, h5py
m=trimesh.load('inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m, m.bounds, m.extents, len(m.vertices), len(m.faces))
print(m.visual.kind if hasattr(m.visual,'kind') else None)
S,ext=trimesh.bounds.oriented_bounds(m); print(S, ext)
r=np.load('inputs/reconstruction.npz'); 
for k in r: print(k, r[k].shape, r[k].dtype)
f=h5py.File('inputs/masks.h5'); 
for k in f: print(k,f[k].shape)
import numpy.linalg as la
T=r['T_world_mesh']; print(T[0]); print(T[-1])
"

```

<details><summary>output</summary>

```
<trimesh.Trimesh(vertices.shape=(282928, 3), faces.shape=(94338, 3))> [[-0.09974834 -0.1046155  -0.09771387]
 [ 0.10502636  0.09852374  0.28576073]] [0.20477471 0.20313925 0.3834746 ] 282928 94338
texture
[[ 0.         -1.          0.         -0.00304588]
 [-1.          0.          0.          0.00263901]
 [ 0.          0.         -1.          0.09402343]
 [ 0.          0.          0.          1.        ]] [0.20313925 0.20477471 0.3834746 ]
T_world_mesh (440, 4, 4) float32
T_cam_mesh_foundationpose (440, 4, 4) float32
T_cam_mesh_coconet (440, 4, 4) float32
human_vertices_world (440, 4610, 3) float32
human_keypoints_world (440, 70, 3) float32
contact_prob_right_left (440, 2) float32
object (440, 1152, 1536)
person (440, 1152, 1536)
[[ 2.1013025e-02  9.9840003e-01 -5.2490044e-02  1.9354897e- …[+398 chars]
```
</details>

**Bash**
```
cd /work; mkdir -p scratch/frames outputs tools; python -c "
import cv2
cap=cv2.VideoCapture('inputs/video.mp4'); print(cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS))
i=0
while True:
    ok,f=cap.read()
    if not ok: break
    if i%20==0: cv2.imwrite(f'scratch/frames/{i:04d}.jpg', cv2.resize(f,(768,576)))
    i+=1
print(i)
"; ls scratch/frames | head -30
```

<details><summary>output</summary>

```
440.0 30.0
440
0000.jpg
0020.jpg
0040.jpg
0060.jpg
0080.jpg
0100.jpg
0120.jpg
0140.jpg
0160.jpg
0180.jpg
0200.jpg
0220.jpg
0240.jpg
0260.jpg
0280.jpg
0300.jpg
0320.jpg
0340.jpg
0360.jpg
0380.jpg
0400.jpg
0420.jpg
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import cv2, numpy as np
ims=[cv2.imread(f'frames/{i:04d}.jpg') for i in range(0,440,20)]
ims=[cv2.resize(im,(384,288)) for im in ims]
for k,im in zip(range(0,440,20),ims): cv2.putText(im,str(k),(10,30),0,1,(0,255,0),2)
while len(ims)%6: ims.append(np.zeros_like(ims[0]))
rows=[np.hstack(ims[i:i+6]) for i in range(0,len(ims),6)]
cv2.imwrite('grid.jpg',np.vstack(rows))
"
```

**Read**
```
/work/scratch/grid.jpg
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
import numpy as np, h5py, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
f=h5py.File('/work/inputs/masks.h5')
om=f['object']; pm=f['person']
cp=r['contact_prob_right_left']
prev=None
for i in range(0,440,4):
    m=om[i]; a=m.sum(); ys,xs=np.nonzero(m) 
    c=(xs.mean(),ys.mean()) if a else (0,0)
    ax=T[i,:3,2]
    eul=R.from_matrix(T[i,:3,:3])
    d=0 if prev is None else np.degrees((R.from_matrix(prev).inv()*eul).magnitude())
    prev=T[i,:3,:3]
    p=Tc[i,:3,3]; K=np.array(cam['K']); uv=K@p; uv=uv[:2]/uv[2]
    print(i, 'area',a,'mc %.0f %.0f'%c,'proj %.0f %.0f'%tuple(uv),'z %.3f'%p[2],'Wt',np.round(T[i,:3,3],3),'ax',np.round(ax,2),'dR4 %.1f'%d,'cp',np.round(cp[i],2))
"

```

<details><summary>output</summary>

```
0 area 9735 mc 765 1029 proj 771 1032 z 1.921 Wt [0.019 1.185 2.036] ax [-0.05 -1.    0.  ] dR4 0.0 cp [0. 0.]
4 area 9766 mc 764 1029 proj 771 1032 z 1.921 Wt [0.019 1.184 2.035] ax [-0.05 -1.    0.  ] dR4 0.0 cp [0. 0.]
8 area 9845 mc 765 1029 proj 771 1032 z 1.921 Wt [0.019 1.184 2.035] ax [-0.05 -1.    0.  ] dR4 0.0 cp [0. 0.]
12 area 9850 mc 765 1029 proj 771 1032 z 1.921 Wt [0.019 1.184 2.035] ax [-0.05 -1.    0.  ] dR4 0.0 cp [0. 0.]
16 area 9835 mc 765 1029 proj 771 1032 z 1.922 Wt [0.019 1.185 2.036] ax [-0.05 -1.    0.  ] dR4 0.0 cp [0. 0.]
20 area 9852 mc 765 1029 proj 771 1032 z 1.922 Wt [0.019 1.185 2.036] ax [-0.05 -1.    0.  ] dR4 0.0 cp [0. 0.]
24 area 9648 mc 765 1029 proj 771 1032 z 1.932 Wt [0.021 1.191 2.045] ax [-0.05 -1.    0.  ] dR4 0.0 cp [0. 0.]
28 area 9652 mc 765 …[+11568 chars]
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import cv2, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames={}
want=[0,100,118,122,126,130,136,144,152,160,168,176,184,190,196,248,256,264,272,280,290,300,310,320,330,340,380,439]
i=0
while True:
    ok,f=cap.read()
    if not ok: break
    if i in want: frames[i]=f
    i+=1
np.save('sel.npy',np.stack([frames[k] for k in want]))
import h5py
m=h5py.File('/work/inputs/masks.h5')['object']
crops=[]
cx={}
for k in want:
    mm=m[k]; ys,xs=np.nonzero(mm)
    if len(xs)==0: cxy=(830,590)
    else: cxy=(int(xs.mean()),int(ys.mean()))
    x0=max(0,cxy[0]-100); y0=max(0,cxy[1]-100)
    c=frames[k][y0:y0+200,x0:x0+200].copy()
    c=cv2.resize(c,(300,300),interpolation=cv2.INTER_CUBIC)
    cv2.putText(c,str(k),(5,25),0,0.8,(0,255,0),2)
    crops.append(c)
while len(crops)%7: crops.append(np.zeros_like(crops[0]))
cv2.imwrite('crops.jpg',np.vstack([np.hstack(crops[i:i+7]) for i in range(0,len(crops),7)]))
"
```

**Read**
```
/work/scratch/crops.jpg
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
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.vertices; c=np.array([0.0026,-0.003,0])
pts=m.sample(20000)
for a in [5,10,20,45,90,180]:
    Rm=R.from_euler('z',a,degrees=True).as_matrix()
    p2=(pts-c)@Rm.T+c
    print(a, np.linalg.norm(p2-pts,axis=1).mean())
# symmetry check: nearest-neighbour distance after rotation
from scipy.spatial import cKDTree
t=cKDTree(pts)
for a in [0,45,90,180]:
    Rm=R.from_euler('z',a,degrees=True).as_matrix()
    for cc in [c, np.zeros(3)]:
        p2=(pts-cc)@Rm.T+cc
        print('sym',a,cc, t.query(p2)[0].mean())
print(m.visual.material)
img=m.visual.material.baseColorTexture; print(img.size if img is not None else None)
img.save('tex.png')
"

```

<details><summary>output</summary>

```
5 0.005707660016834395
10 0.011404455178424557
20 0.02272211556674299
45 0.05007468141909213
90 0.0925259465202449
180 0.13085144844033803
sym 0 [ 0.0026 -0.003   0.    ] 5.268690949200709e-19
sym 0 [0. 0. 0.] 0.0
sym 45 [ 0.0026 -0.003   0.    ] 0.004247503675157296
sym 45 [0. 0. 0.] 0.005089389873927937
sym 90 [ 0.0026 -0.003   0.    ] 0.004586623935240814
sym 90 [0. 0. 0.] 0.006404204669249215
sym 180 [ 0.0026 -0.003   0.    ] 0.006000322743348123
sym 180 [0. 0. 0.] 0.008732381287820015
<trimesh.visual.material.PBRMaterial object at 0x75668a192a90>
(2232, 2280)
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

> *thinking:* The object is a traffic cone with near 4-fold symmetric texture, risking 90° yaw errors (~9cm ADD); it's on the floor until frame 117, carried and hidden around 197–247, then on the table from frame 336 onward. Next I'll check the baseline's yaw over time.

**Bash**
```
mkdir -p /work/outputs; cd /work/scratch; cat > ana_twist.py <<'EOF'
import numpy as np, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
def swing_twist(Rrel):
    q=R.from_matrix(Rrel).as_quat() # x y z w
    tw=np.array([0,0,q[2],q[3]]); tw/=np.linalg.norm(tw)
    ang=2*np.degrees(np.arctan2(tw[2],tw[3]))
    sw=R.from_quat(q)*R.from_quat(tw).inv()
    return (ang+180)%360-180, np.degrees(sw.magnitude())
for key in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
    T=r[key].astype(np.float64)
    print(key)
    s=[]
    for i in range(0,440,2):
        tw,sw=swing_twist(T[0,:3,:3].T@T[i,:3,:3])
        s.append('%d:%.0f/%.0f'%(i,tw,sw))
    print(' '.join(s))
EOF
python ana_twist.py
```

<details><summary>output</summary>

```
T_world_mesh
0:0/0 2:0/0 4:0/0 6:0/0 8:0/0 10:0/0 12:0/0 14:0/0 16:0/0 18:0/0 20:0/0 22:0/0 24:0/0 26:0/0 28:0/0 30:0/0 32:0/0 34:0/0 36:0/0 38:0/0 40:0/0 42:0/0 44:0/0 46:0/0 48:0/0 50:0/0 52:0/0 54:0/0 56:0/0 58:0/0 60:0/0 62:0/0 64:0/0 66:0/0 68:0/0 70:0/0 72:0/0 74:0/0 76:0/0 78:0/0 80:0/0 82:0/0 84:0/0 86:0/0 88:0/0 90:0/0 92:0/0 94:0/0 96:0/0 98:0/0 100:0/0 102:0/0 104:0/0 106:0/0 108:0/0 110:0/0 112:0/0 114:0/0 116:0/0 118:89/2 120:89/2 122:87/2 124:89/1 126:87/5 128:92/9 130:101/11 132:135/8 134:135/8 136:138/11 138:148/17 140:148/17 142:150/17 144:140/8 146:138/11 148:134/15 150:149/19 152:150/17 154:146/18 156:159/20 158:148/25 160:156/28 162:153/24 164:144/21 166:166/12 168:154/3 170:171/4 172:146/2 174:150/3 176:-180/4 178:155/4 180:-180/4 182:-160/5 184:-151/5 186:-172/1 188:- …[+5435 chars]
```
</details>

**Bash**
```
cd /code/reconstruction; cat CLAUDE.md | head -80; grep -rn "keypoint" --include=*.py -il . | head -20
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

For the exact input contract, comm …[+3361 chars]
```
</details>

**Bash**
```
cd /code/reconstruction; grep -rn -i "shoulder\|wrist\|pelvis\|hip" modules/v2d_cari4d/lib/cari4d/lib_mhr/schema.py modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py | head -30; grep -rn "pose_info\|KEYPOINT_NAMES\|keypoint_names" -r modules/v2d_sam3d_body/lib --include=*.py | head
```

<details><summary>output</summary>

```
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:11:MHR_WRIST_KEYPOINTS: Mapping[str, int] = {
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:12:    "left_wrist": 62,
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:13:    "right_wrist": 41,
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:15:MHR_WRIST_NAMES = tuple(MHR_WRIST_KEYPOINTS.keys())
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:16:MHR_WRIST_INDICES = tuple(MHR_WRIST_KEYPOINTS[name] for name in MHR_WRIST_NAMES)
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:26:    wrist_points_world: np.ndarray
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:95:def mhr_wrist_points_from_keypoints(mhr_keypoints: Any) -> np.ndarray:
modules/v2d_cari4d/lib/cari4d/lib_mhr/contact.py:97:    min_keypoints = max(MHR_WRIST_INDICES) + 1
modules/v2 …[+2482 chars]
```
</details>

**Bash**
```
cd /code/reconstruction; sed -n 800,900p modules/v2d_sam3d_body/lib/sam_3d_body/metadata/mhr70.py; sed -n 1,77p modules/v2d_sam3d_body/lib/sam_3d_body/metadata/mhr70.py | grep -n "name\|^ *[0-9]" | head -80
```

<details><summary>output</summary>

```
        ),
        64: dict(
            link=("right_pinky_finger3", "right_pinky_finger4"),
            id=64,
            color=[0, 255, 0],
        ),
    },
    joint_weights=[1.0] * 70,
    body_keypoint_names=[
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    foot_keypoint_names=[
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    ],
    left_hand_keypoint_names= …[+1599 chars]
```
</details>

**Bash**
```
cd /code/reconstruction; sed -n 1,78p modules/v2d_sam3d_body/lib/sam_3d_body/metadata/mhr70.py | tr -s ' \n' ' ' | head -c 3000
```

<details><summary>output</summary>

```
# Copyright (c) Meta Platforms, Inc. and affiliates. """The first 70 of 308 MHR keypoints, ignoring the rest for face keypoints""" mhr_names = [ "nose", "left-eye", "right-eye", "left-ear", "right-ear", "left-shoulder", "right-shoulder", "left-elbow", "right-elbow", "left-hip", "right-hip", "left-knee", "right-knee", "left-ankle", "right-ankle", "left-big-toe-tip", "left-small-toe-tip", "left-heel", "right-big-toe-tip", "right-small-toe-tip", "right-heel", "right-thumb-tip", "right-thumb-first-joint", "right-thumb-second-joint", "right-thumb-third-joint", "right-index-tip", "right-index-first-joint", "right-index-second-joint", "right-index-third-joint", "right-middle-tip", "right-middle-first-joint", "right-middle-second-joint", "right-middle-third-joint", "right-ring-tip", "right-ring-fi …[+853 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > ana_body.py <<'EOF'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz')
K=r['human_keypoints_world'].astype(np.float64); T=r['T_world_mesh'].astype(np.float64)
def yaw(v): # world y-down; horizontal plane x-z; angle about -y (up)
    return np.degrees(np.arctan2(v[...,0],v[...,2]))
sh=K[:,5]-K[:,6]; hp=K[:,9]-K[:,10]
up=np.array([0,-1.,0])
face=np.cross(up, sh+hp)   # direction
hw=K[:,62]-K[:,41]
def handyaw(w,i,p):  # hand plane x-axis: index MCP - pinky MCP
    return yaw(K[:,i]-K[:,p])
ry=handyaw(None,28,40); ly=handyaw(None,49,61)
obj=T[:,:3,3]
for i in range(100,345,3):
    dR=np.linalg.norm(K[i,41]-obj[i]); dL=np.linalg.norm(K[i,62]-obj[i])
    print(i,'face %.0f'%yaw(face[i]),'LRwrist %.0f'%yaw(hw[i]),'rh %.0f lh %.0f'%(ry[i],ly[i]),'dR %.2f dL %.2f'%(dR,dL), 'cp',np.round(r['contact_prob_right_left'][i],2))
EOF
python ana_body.py
```

<details><summary>output</summary>

```
100 face 21 LRwrist 111 rh 179 lh -127 dR 0.77 dL 0.73 cp [0. 0.]
103 face 15 LRwrist 102 rh -180 lh -130 dR 0.68 dL 0.67 cp [0. 0.]
106 face 9 LRwrist 100 rh 173 lh -133 dR 0.61 dL 0.60 cp [0. 0.]
109 face 5 LRwrist 94 rh 173 lh -142 dR 0.53 dL 0.53 cp [0. 0.]
112 face 1 LRwrist 92 rh 165 lh -149 dR 0.45 dL 0.45 cp [0. 0.]
115 face -1 LRwrist 86 rh 165 lh -156 dR 0.38 dL 0.38 cp [0. 0.]
118 face -1 LRwrist 85 rh 159 lh -154 dR 0.32 dL 0.31 cp [0.18 0.01]
121 face -1 LRwrist 88 rh 161 lh -153 dR 0.27 dL 0.27 cp [1. 1.]
124 face -2 LRwrist 93 rh 160 lh -154 dR 0.27 dL 0.27 cp [1. 1.]
127 face -3 LRwrist 92 rh 159 lh -154 dR 0.28 dL 0.27 cp [1. 1.]
130 face -5 LRwrist 83 rh 162 lh -168 dR 0.28 dL 0.27 cp [1. 1.]
133 face -7 LRwrist 87 rh 162 lh -165 dR 0.29 dL 0.30 cp [1. 1.]
136 face -10 LR …[+4675 chars]
```
</details>

> *thinking:* I found that FoundationPose's yaw jumps +90° at frame 104 while the cone is untouched, and the baseline carries that error from frame 118 onward—though the carried phase's overall rotation roughly matches the body turn, suggesting a likely 90° offset rather than a fundamental tracking failure. Next I'll build a render-and-compare scorer using nvdiffrast silhouettes against the SAM2 masks to verify this.

**Bash**
```
cd /work/scratch; python -c "
import nvdiffrast.torch as dr, torch; print(dr.__file__)
ctx=dr.RasterizeCudaContext(); print('ok')
"; pip list 2>/dev/null | grep -i -E "torch|nvdiff|opencv|trimesh|open3d|scipy|kornia|pytorch3d|h5py|numpy"
```

<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/__init__.py
ok
h5py                                     3.16.0
kornia                                   0.8.3
kornia_rs                                0.2.0
numpy                                    1.26.3
nvdiffrast                               0.4.0
open3d                                   0.20.0
opencv-contrib-python                    4.11.0.86
opencv-python                            4.11.0.86
pytorch-lightning                        2.6.6
pytorch3d                                0.7.9
scipy                                    1.16.3
torch                                    2.5.1+cu124
torchaudio                               2.5.1+cu124
torchelastic                             0.2.2
torchmetrics                             1.9.0 …[+101 chars]
```
</details>

**Bash**
```
free -g; nproc
```

<details><summary>output</summary>

```
               total        used        free      shared  buff/cache   available
Mem:             125          32          15           0          77          91
Swap:              1           1           0
48
```
</details>

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
cd /work/scratch; cat > vis_base.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
ep=Episode('/work/inputs'); fr=ep.frames()
rd=Renderer(ep.mesh)
want=[int(a) for a in sys.argv[2].split(',')]
src=sys.argv[1]
T = ep.Tc_base if src=='base' else ep.T_cam_world[None]@np.load(src)['T_world_mesh']
tiles=[]
for i in want:
    c=proj(ep.K,T[i:i+1,:3,3])[0]
    crop=crop_box(c,110); S=256
    m,rgb=rd.render(T[i:i+1],ep.K,crop,S,color=True)
    m=m[0].cpu().numpy(); rgb=rgb[0].cpu().numpy()
    im=cut(fr[i],crop,S).astype(np.float32)/255
    om=cut(ep.obj_mask[i],crop,S,cv2.INTER_NEAREST); pm=cut(ep.person_mask[i],crop,S,cv2.INTER_NEAREST)
    a=im.copy(); a=np.where(m[...,None]>0.5, 0.5*a+0.5*rgb, a)
    cs,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    a=(a*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.drawContours(a,cs,-1,(0,255,0),1)
    cs,_=cv2.findContours(pm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a,cs,-1,(255,0,0),1)
    b=(im*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.putText(b,str(i),(5,20),0,0.6,(0,255,0),2)
    tiles.append(np.hstack([b,a]))
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
EOF
python vis_base.py base 0,60,110,118,124,130,140,150,160,170,180,190,250,270,290,310,330,400 vb.jpg
```

**Read**
```
/work/scratch/vb.jpg
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
cd /work/scratch; cat > yaw_sweep.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
ep=Episode('/work/inputs')
rd=Renderer(ep.mesh)
c=np.array([0.0026,-0.003,0])
def about_axis(T,deg):
    A=np.eye(4);A[:3,3]=c; B=np.eye(4);B[:3,:3]=Rz(deg); C=np.eye(4);C[:3,3]=-c
    return T@A@B@C
frames=[int(a) for a in sys.argv[1].split(',')]
ths=np.arange(-45,46,3)
S=192
tot=np.zeros(len(ths))
for i in frames:
    T0=ep.Tc_base[i]
    ctr=proj(ep.K,T0[None,:3,3])[0]
    crop=crop_box(ctr,100)
    Ts=np.stack([about_axis(T0,t) for t in ths])
    m,_=rd.render(Ts,ep.K,crop,S)
    om=torch.tensor(cut(ep.obj_mask[i].astype(np.uint8)*255,crop,S,cv2.INTER_AREA)/255.,device='cuda').float()
    pm=torch.tensor(cut(ep.person_mask[i].astype(np.uint8)*255,crop,S,cv2.INTER_AREA)/255.,device='cuda').float()
    valid=1-pm*(1-om)  # ignore person pixels that are not object
    inter=(m*om*valid).sum((1,2)); uni=((m+om-m*om)*valid).sum((1,2))
    iou=(inter/uni).cpu().numpy(); tot+=iou
    print(i,' '.join('%d:%.3f'%(t,v) for t,v in zip(ths,iou)))
print('mean',' '.join('%d:%.4f'%(t,v) for t,v in zip(ths,tot/len(frames))))
EOF
python yaw_sweep.py 0,20,40,60,80,100,110 ; python yaw_sweep.py 340,360,380,400,420,439 | tail -1
```

<details><summary>output</summary>

```
0 -45:0.790 -42:0.790 -39:0.791 -36:0.792 -33:0.794 -30:0.796 -27:0.800 -24:0.806 -21:0.812 -18:0.820 -15:0.826 -12:0.833 -9:0.830 -6:0.829 -3:0.826 0:0.824 3:0.821 6:0.818 9:0.813 12:0.805 15:0.796 18:0.788 21:0.778 24:0.771 27:0.765 30:0.759 33:0.754 36:0.750 39:0.747 42:0.744 45:0.742
20 -45:0.794 -42:0.794 -39:0.795 -36:0.796 -33:0.798 -30:0.800 -27:0.805 -24:0.811 -21:0.818 -18:0.824 -15:0.831 -12:0.838 -9:0.836 -6:0.835 -3:0.833 0:0.830 3:0.828 6:0.825 9:0.818 12:0.809 15:0.801 18:0.792 21:0.783 24:0.776 27:0.770 30:0.764 33:0.759 36:0.755 39:0.752 42:0.749 45:0.747
40 -45:0.792 -42:0.793 -39:0.792 -36:0.793 -33:0.795 -30:0.798 -27:0.802 -24:0.808 -21:0.815 -18:0.823 -15:0.830 -12:0.837 -9:0.838 -6:0.836 -3:0.833 0:0.829 3:0.827 6:0.823 9:0.815 12:0.807 15:0.799 18:0.790 21:0.782 24: …[+1876 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat >> /work/outputs/hoilib.py <<'EOF'


# ----------------------------------------------------------------------------- differentiable fitting
AXIS_C = np.array([0.0026, -0.003, 0.0])  # cone axis centre in mesh-file coords (OBB centre, from oriented_bounds)


def axis_rot(T, deg):
    """Rotate pose T (cam/world-from-mesh) by deg about the cone's own axis (mesh z through AXIS_C)."""
    A = np.eye(4); A[:3, 3] = AXIS_C
    B = np.eye(4); B[:3, :3] = Rz(deg)
    C = np.eye(4); C[:3, 3] = -AXIS_C
    return T @ A @ B @ C


def so3_exp(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2]; Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


class MaskTargets:
    """Per-frame crops of object/person masks at fixed size, stored on GPU."""

    def __init__(self, ep, frames, Tc_init, half=None, size=160):
        self.frames = list(frames)
        self.size = size
        crops, om, val = [], [], []
        for k, i in enumerate(self.frames):
            ctr = proj(ep.K, Tc_init[k][None, :3, 3])[0]
            z = Tc_init[k][2, 3]
            h = half if half is n …[+3421 chars]
```

<details><summary>output</summary>

```
init -45 iou 0.9242 twist -91.5 swing 3.0 dt(cm) [-0.85 -0.53 -0.33]
init -30 iou 0.9203 twist -1.4 swing 1.7 dt(cm) [-0.98 -0.62  0.06]
init -15 iou 0.9201 twist -1.5 swing 1.7 dt(cm) [-1.01 -0.55  0.04]
init 0 iou 0.9206 twist -1.5 swing 1.6 dt(cm) [-1.  -0.6  0. ]
init 15 iou 0.9194 twist -1.3 swing 1.8 dt(cm) [-0.98 -0.55  0.06]
init 30 iou 0.9205 twist -1.3 swing 1.7 dt(cm) [-1.04 -0.63  0.04]
init 45 iou 0.8783 twist 9.2 swing 6.4 dt(cm) [-1.22 -0.92 -0.91]
```
</details>

> *thinking:* The pooled silhouette fit over 29 floor frames matches the baseline yaw within 1.5°, improving lateral correction by ~1 cm and IoU from 0.83 to 0.92, but on the table the edge-on view gives no yaw info, so yaw there must come from carry propagation. Next I'll run per-frame fits over the carried phase.

**Bash**
```
cd /work/scratch; cat >> /work/outputs/hoilib.py <<'EOF'


def compose(T0, w, t, zscale=0.0):
    """Apply rotation exp(w) about the cone centre (in camera frame) and translation t to poses T0 (B,4,4)."""
    R = so3_exp(w)
    cl = torch.tensor(list(AXIS_C[:2]) + [0.09, 1.0], device=T0.device, dtype=T0.dtype)
    cc = (T0 @ cl)[:, :3]
    Rn = R @ T0[:, :3, :3]
    tm = torch.tensor([1.0, 1.0, zscale], device=T0.device, dtype=T0.dtype)
    tn = (R @ (T0[:, :3, 3] - cc)[..., None])[..., 0] + cc + t * tm
    top = torch.cat([Rn, tn[..., None]], 2)
    bot = torch.tensor([0, 0, 0, 1.0], device=T0.device, dtype=T0.dtype).expand(T0.shape[0], 1, 4)
    return torch.cat([top, bot], 1)


def fit_frames(rd, ep, tg, T0, idx, iters=120, lr=0.01, zscale=0.0, chunk=96):
    """Independent per-hypothesis silhouette fit. T0 (N,4,4) numpy init, idx (N,) index into tg frames.
    Returns refined poses (N,4,4) and IoU (N,)."""
    outT, outI = [], []
    for s in range(0, len(T0), chunk):
        T0c = torch.tensor(T0[s:s + chunk], device="cuda", dtype=torch.float32)
        ic = idx[s:s + chunk]
        crops = [tg.crops[k] for k in ic]
        n = len(T0c)
        w = torch.zeros(n, 3, device="cuda", requires_grad=True)
        t = torch.zeros(n, 3, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([w, t], lr=lr)
        for it in range(iters):
            T = compose(T0c, w, t, zscale)
            m = render_masks(rd, T, ep.K, crops, tg.size)
            iou = soft_iou(m, t …[+1160 chars]
```

<details><summary>output</summary>

```
time 25.770946502685547
```
</details>

**Bash**
```
cd /work/scratch; cat > ana_carry.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
d=np.load('carry_fit.npz'); frames=d['frames']; Tf=d['Tf'].reshape(len(frames),-1,4,4); iou=d['iou'].reshape(len(frames),-1)
import json
ep=Episode('/work/inputs')
R0=ep.T_base[0,:3,:3]
for k,i in enumerate(frames):
    if i%2: continue
    s=[]
    for h in range(Tf.shape[1]):
        Tw=ep.T_world_cam@Tf[k,h]
        tw,sw=twist_swing(R0.T@Tw[:3,:3])
        s.append('%4.0f(%3.0f)/%2.0f %.3f'%(tw,(tw+45)%90-45,sw,iou[k,h]))
    twb,swb=twist_swing(R0.T@ep.T_base[i,:3,:3])
    print(i,'area %5d'%d['area'][k],'base %4.0f/%2.0f |'%(twb,swb),' | '.join(s))
EOF
python ana_carry.py
```

<details><summary>output</summary>

```
110 area  2050 base    0/ 0 |   -2( -2)/ 2 0.914 |   -2( -2)/ 2 0.912 |   -2( -2)/ 2 0.915 |   89( -1)/ 2 0.908
112 area  2047 base    0/ 0 |   -2( -2)/ 2 0.916 |   -2( -2)/ 2 0.917 |   -1( -1)/ 2 0.914 |   89( -1)/ 2 0.910
114 area  2045 base    0/ 0 |   -2( -2)/ 2 0.919 |   -2( -2)/ 2 0.919 |   -1( -1)/ 2 0.913 |   89( -1)/ 2 0.911
116 area  2004 base    0/ 0 |   -2( -2)/ 2 0.926 |   -2( -2)/ 2 0.924 |   -2( -2)/ 2 0.925 |   89( -1)/ 2 0.918
118 area  2026 base   89/ 2 |   89( -1)/ 2 0.908 |   88( -2)/ 2 0.911 |   88( -2)/ 2 0.909 |  177( -3)/ 2 0.923
120 area  2040 base   89/ 2 |   89( -1)/ 2 0.911 |   89( -1)/ 2 0.912 |   89( -1)/ 1 0.912 |  178( -2)/ 2 0.924
122 area  1949 base   87/ 2 |   89( -1)/ 2 0.917 |   89( -1)/ 2 0.920 |   89( -1)/ 2 0.920 |  177( -3)/ 2 0.924
124 area  1947 b …[+12415 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > tex_views.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
tiles=[]
for i,zoom in [(0,1),(400,1)]:
    T0=ep.Tc_base[i]
    ctr=proj(ep.K,T0[None,:3,3])[0]; crop=crop_box(ctr,int(70*1.92/T0[2,3]*1.0)+10)
    Ts=np.stack([axis_rot(T0,a) for a in [0,90,180,270]])
    m,rgb=rd.render(Ts,ep.K,crop,384,color=True)
    rgb=rgb.cpu().numpy()
    im=cut(ep.frames()[i],crop,384).astype(np.float32)/255
    row=[im]+[rgb[k]+(1-m[k].cpu().numpy()[...,None])*im*0.3 for k in range(4)]
    tiles.append(np.hstack(row))
cv2.imwrite('texviews.jpg',(np.vstack(tiles)[:,:,::-1]*255).clip(0,255).astype(np.uint8))
EOF
python tex_views.py
```

**Read**
```
/work/scratch/texviews.jpg
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

> *thinking:* I spotted a pale scuff on the lower cone body and a seam in the reflective band that could break the yaw symmetry. I'll check against the video to see if the scuff shows up there.

**Bash**
```
cd /work/scratch; cat > tex_views2.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
fl=[int(a) for a in sys.argv[1].split(',')]
tiles=[]
for i in fl:
    T0=ep.Tc_base[i]
    ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]; crop=crop_box(ctr,int(0.25*ep.K[0,0]/T0[2,3]))
    Ts=np.stack([axis_rot(T0,a) for a in [0,90,180,270]])
    m,rgb=rd.render(Ts,ep.K,crop,256,color=True)
    rgb=rgb.cpu().numpy()
    im=cut(ep.frames()[i],crop,256,cv2.INTER_CUBIC).astype(np.float32)/255
    row=[im]+[rgb[k]+(1-m[k].cpu().numpy()[...,None])*im*0.3 for k in range(4)]
    r=np.hstack(row); cv2.putText(r,str(i),(5,20),0,0.6,(0,1,0),2); tiles.append(r)
cv2.imwrite(sys.argv[2],(np.vstack(tiles)[:,:,::-1]*255).clip(0,255).astype(np.uint8))
EOF
python tex_views2.py 50,140,150,160,170 tv1.jpg; python tex_views2.py 260,280,290,300,320,400 tv2.jpg
```

**Read**
```
/work/scratch/tv1.jpg
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
/work/scratch/tv2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

**Bash**
```
cd /work/scratch; cat > photo_score.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
fl=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
offs=np.arange(0,360,15)
S=192
def sat(x):
    mx=x.max(-1)[0]; mn=x.min(-1)[0]; return (mx-mn)/(mx+1e-3)
tot=np.zeros(len(offs)); cnt=0
rows=[]
for i in fl:
    T0=ep.Tc_base[i]
    if ep.obj_mask[i].sum()<500: continue
    ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]; crop=crop_box(ctr,int(0.25*ep.K[0,0]/T0[2,3]))
    Ts=np.stack([axis_rot(T0,a) for a in offs])
    m,rgb=rd.render(Ts,ep.K,crop,S,color=True)
    im=torch.tensor(cut(ep.frames()[i],crop,S,cv2.INTER_AREA)/255.,device='cuda').float()
    om=torch.tensor(cut(ep.obj_mask[i].astype(np.uint8)*255,crop,S)/255.,device='cuda').float()
    pm=torch.tensor(cut(ep.person_mask[i].astype(np.uint8)*255,crop,S)/255.,device='cuda').float()
    w=(m>0.99).float()*(om>0.99).float()*(pm<0.01).float()
    # erode
    w=-torch.nn.functional.max_pool2d(-w[:,None],5,1,2)[:,0]
    si=sat(im)[None]; sr=sat(rgb)
    # compare also lightness-normalised colour
    d=((si-sr).abs()*w).sum((1,2))/w.sum((1,2)).clamp_min(1)
    d=d.cpu().numpy(); tot+=d; cnt+=1
    rows.append((i,d))
for i,d in rows[::max(1,len(rows)//12)]:
    print(i,' '.join('%d:%.3f'%(o,v) for o,v in zip(offs,d)))
print('MEAN',' '.join('%d:%.4f'%(o,v) for o,v in zip(offs,tot/cnt)))
EOF
python photo_score. …[+43 chars]
```

<details><summary>output</summary>

```
0 0:0.106 15:0.106 30:0.106 45:0.107 60:0.105 75:0.098 90:0.093 105:0.090 120:0.088 135:0.092 150:0.094 165:0.096 180:0.095 195:0.093 210:0.093 225:0.094 240:0.097 255:0.102 270:0.106 285:0.107 300:0.107 315:0.108 330:0.108 345:0.107
9 0:0.104 15:0.103 30:0.104 45:0.106 60:0.103 75:0.096 90:0.090 105:0.087 120:0.085 135:0.088 150:0.091 165:0.094 180:0.092 195:0.091 210:0.090 225:0.091 240:0.093 255:0.098 270:0.103 285:0.104 300:0.105 315:0.105 330:0.106 345:0.105
18 0:0.105 15:0.104 30:0.104 45:0.105 60:0.104 75:0.097 90:0.092 105:0.088 120:0.087 135:0.090 150:0.092 165:0.095 180:0.093 195:0.092 210:0.092 225:0.093 240:0.096 255:0.100 270:0.105 285:0.105 300:0.106 315:0.106 330:0.107 345:0.106
27 0:0.104 15:0.104 30:0.104 45:0.105 60:0.104 75:0.097 90:0.092 105:0.088 120:0.087 135:0.089 15 …[+6787 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > avg_static.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
for (a,b,ref) in [(0,95,50),(345,440,400)]:
    fr=ep.frames()[a:b].astype(np.float32)
    # median to remove transient occluders
    T0=ep.Tc_base[ref]
    ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]; h=int(0.22*ep.K[0,0]/T0[2,3]); crop=crop_box(ctr,h)
    x0,y0,s=crop
    med=np.median(fr[:,y0:y0+s,x0:x0+s],0)
    big=cv2.resize(med,(512,512),interpolation=cv2.INTER_CUBIC)
    Ts=np.stack([axis_rot(T0,o) for o in [0,90,180,270]])
    m,rgb=rd.render(Ts,ep.K,crop,512,color=True)
    rgb=rgb.cpu().numpy()*255
    cv2.imwrite(f'avg_{a}.jpg',np.hstack([big[:,:,::-1]]+[rgb[k][:,:,::-1] for k in range(4)]).astype(np.uint8))
EOF
python avg_static.py
```

**Read**
```
/work/scratch/avg_0.jpg
```

**Read**
```
/work/scratch/avg_345.jpg
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
cd /work/scratch; cat > montage.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
ep=Episode('/work/inputs')
fl=list(range(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])))
tiles=[]
for i in fl:
    m=ep.obj_mask[i]; ys,xs=np.nonzero(m)
    T0=ep.Tc_base[i]
    if len(xs)<200: ctr=proj(ep.K,(T0@np.array([0,0,0.09,1]))[None,:3])[0]
    else: ctr=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
    h=int(0.24*ep.K[0,0]/T0[2,3]); crop=crop_box(ctr,h)
    c=cut(ep.frames()[i],crop,200,cv2.INTER_CUBIC)[:,:,::-1].copy()
    cv2.putText(c,str(i),(3,18),0,0.6,(0,255,0),2); tiles.append(c)
nc=int(sys.argv[5]) if len(sys.argv)>5 else 8
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[4],np.vstack([np.hstack(tiles[k:k+nc]) for k in range(0,len(tiles),nc)]))
EOF
python montage.py 118 198 3 m1.jpg 9; python montage.py 246 340 3 m2.jpg 8
```

**Read**
```
/work/scratch/m1.jpg
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
/work/scratch/m2.jpg
```

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* The scuff mark isn't visible in any frame, so it's inconclusive, but the base corner rotation in frames 118–196 confirms the cone turns smoothly by ~45–90°, proving the baseline's +90° jump at frame 118 is spurious. Next I'll check if the table pose's yaw can be read from the base flange width.

**Bash**
```
cd /work/scratch; cat > fit_static2.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
a,b,zs=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
frames=list(range(a,b,6))
Tm=ep.Tc_base[frames[len(frames)//2]].copy(); Tm[:3,3]=np.median(ep.Tc_base[frames][:,:3,3],0)
tg=MaskTargets(ep,frames,np.repeat(Tm[None],len(frames),0),size=192)
offs=np.arange(-45,45,5)
T0=np.array([axis_rot(Tm,o) for o in offs])
# shared pose per hypothesis: optimise each hypothesis jointly over all frames
res=[]
for h in range(len(offs)):
    T0c=torch.tensor(T0[h:h+1],device='cuda',dtype=torch.float32)
    w=torch.zeros(1,3,device='cuda',requires_grad=True); t=torch.zeros(1,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.01)
    for it in range(120):
        T=compose(T0c,w,t,zs).expand(len(frames),4,4)
        iou=soft_iou(render_masks(rd,T,ep.K,tg.crops,tg.size),tg)
        loss=1-iou.mean(); opt.zero_grad(); loss.backward(); opt.step()
        if it==80:
            for g in opt.param_groups: g['lr']=0.003
    Tf=T[0].detach().cpu().numpy().astype(np.float64)
    tw,sw=twist_swing(ep.T_base[0,:3,:3].T@(ep.T_world_cam@Tf)[:3,:3])
    print('off %4d -> twist %6.1f swing %4.1f iou %.4f dt(cm) %s'%(offs[h],tw,sw,iou.mean().item(),np.round((Tf[:3,3]-Tm[:3,3])*100,2)))
EOF
python fit_static2.py 345 440 0; python fit_static2.py 0 116 0 
```

<details><summary>output</summary>

```
off  -45 -> twist -124.0 swing  6.7 iou 0.8339 dt(cm) [-0.8   0.58  0.39]
off  -40 -> twist  -84.1 swing  5.2 iou 0.8485 dt(cm) [-0.64  0.93  0.19]
off  -35 -> twist  -67.2 swing  6.1 iou 0.8605 dt(cm) [-0.53  1.08  0.4 ]
off  -30 -> twist  -65.9 swing  5.8 iou 0.8626 dt(cm) [-0.65  0.95  0.42]
off  -25 -> twist  -67.5 swing  6.1 iou 0.8630 dt(cm) [-0.77  0.97  0.43]
off  -20 -> twist  -69.0 swing  6.4 iou 0.8640 dt(cm) [-0.56  0.94  0.46]
off  -15 -> twist  -68.3 swing  6.1 iou 0.8630 dt(cm) [-0.57  0.99  0.43]
off  -10 -> twist  -68.2 swing  6.0 iou 0.8628 dt(cm) [-0.76  0.93  0.41]
off   -5 -> twist  -68.5 swing  6.2 iou 0.8643 dt(cm) [-0.7   1.01  0.43]
off    0 -> twist  -68.6 swing  6.0 iou 0.8637 dt(cm) [-0.64  0.93  0.41]
off    5 -> twist  -68.6 swing  5.9 iou 0.8637 dt(cm) [-0.75 …[+1863 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import re
p='/work/outputs/hoilib.py'
s=open(p).read()
s=s.replace('''def compose(T0, w, t, zscale=0.0):
    """Apply rotation exp(w) about the cone centre (in camera frame) and translation t to poses T0 (B,4,4)."""
    R = so3_exp(w)''','''def compose(T0, w, t, zscale=0.0, fix_twist=False):
    """Apply rotation exp(w) about the cone centre (in camera frame) and translation t to poses T0 (B,4,4).
    fix_twist: remove the component of w along the cone axis (twist is then held at its initial value)."""
    if fix_twist:
        a = T0[:, :3, 2]
        w = w - (w * a).sum(-1, keepdim=True) * a
    R = so3_exp(w)''')
s=s.replace('''def fit_frames(rd, ep, tg, T0, idx, iters=120, lr=0.01, zscale=0.0, chunk=96):''','''def fit_frames(rd, ep, tg, T0, idx, iters=120, lr=0.01, zscale=0.0, chunk=96, fix_twist=False):''')
s=s.replace('''            T = compose(T0c, w, t, zscale)
            m = render_masks''','''            T = compose(T0c, w, t, zscale, fix_twist)
            m = render_masks''')
s=s.replace('''            T = compose(T0c, w, t, zscale)
            iou = soft_iou''','''            T = compose(T0c, w, t, zscale, fix_twist)
            iou = soft_iou''')
open(p,'w').write(s)
EOF
cat > profile_carry.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, time
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
frames=[i for i in range(110,346) if ep.obj_mask[i].sum()>300]
tg=MaskTargets(ep …[+566 chars]
```

<details><summary>output</summary>

```
time 63.67855262756348
```
</details>

**Bash**
```
cd /work/scratch; cat > ana_prof.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs')
d=np.load('carry_profile.npz'); fr=d['frames']; g=d['grid']; iou=d['iou'].reshape(len(fr),len(g)); Tf=d['Tf'].reshape(len(fr),len(g),4,4)
R0=ep.T_base[0,:3,:3]
for k,i in enumerate(fr):
    tws=[twist_swing(R0.T@(ep.T_world_cam@Tf[k,j])[:3,:3])[0] for j in range(len(g))]
    tws=np.array(tws)%90
    j=iou[k].argmax(); 
    prof=''.join(' .:-=+*#%@'[min(9,int(max(0,(iou[k,jj]-iou[k].max()+0.02))/0.02*9))] for jj in range(len(g)))
    print(i,'best %5.1f'%tws[j],'iou %.3f range %.3f'%(iou[k].max(),iou[k].max()-iou[k].min()),prof)
EOF
python ana_prof.py
```

<details><summary>output</summary>

```
110 best   0.0 iou 0.913 range 0.097 @=               -
111 best   0.0 iou 0.911 range 0.097 @-               :
112 best   0.0 iou 0.913 range 0.096 @-               :
113 best   0.0 iou 0.915 range 0.095 @:               -
114 best   0.0 iou 0.912 range 0.094 @=               -
115 best   0.0 iou 0.919 range 0.100 @:                
116 best   0.0 iou 0.923 range 0.099 @:               :
117 best  90.0 iou 0.913 range 0.099 @-               =
118 best  90.0 iou 0.919 range 0.101 @.                
119 best  90.0 iou 0.919 range 0.100 @-               -
120 best  90.0 iou 0.919 range 0.101 @:               :
121 best  90.0 iou 0.921 range 0.105 @=               .
122 best  90.0 iou 0.923 range 0.109 @:               -
123 best   0.0 iou 0.925 range 0.106 @-               =
124 best  90.0 i …[+9559 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
tex=np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255
uv=m.visual.uv; H,W=tex.shape[:2]
px=(uv[:,0]*(W-1)).round().astype(int).clip(0,W-1); py=((1-uv[:,1])*(H-1)).round().astype(int).clip(0,H-1)
col=tex[py,px]; mx=col.max(1); mn=col.min(1); sat=(mx-mn)/(mx+1e-3)
v=m.vertices; z=v[:,2]; r=np.hypot(v[:,0]-AXIS_C[0],v[:,1]-AXIS_C[1]); az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
# outer surface only: r near max for given z
for zz in np.arange(-0.1,0.29,0.02):
    s=(z>=zz)&(z<zz+0.02)
    if s.sum()==0: continue
    rm=np.percentile(r[s],90); o=s&(r>rm-0.01)
    print('z %.2f n %6d sat_mean %.2f frac_low %.3f'%(zz,o.sum(),sat[o].mean(),(sat[o]<0.5).mean()))
low=(sat<0.5)&(z<0.1)&(z>-0.08)
print('low cnt',low.sum())
h,_=np.histogram(az[low],bins=24,range=(-180,180)); print('az hist',h)
h2,_=np.histogram(z[low],bins=10,range=(-0.08,0.1)); print('z hist',h2)
EOF
```

<details><summary>output</summary>

```
z -0.10 n  18948 sat_mean 0.72 frac_low 0.279
z -0.08 n  20850 sat_mean 0.91 frac_low 0.000
z -0.06 n  20846 sat_mean 0.91 frac_low 0.000
z -0.04 n  14423 sat_mean 0.90 frac_low 0.000
z -0.02 n  13233 sat_mean 0.90 frac_low 0.000
z 0.00 n  13556 sat_mean 0.90 frac_low 0.000
z 0.02 n  12660 sat_mean 0.91 frac_low 0.000
z 0.04 n  10774 sat_mean 0.91 frac_low 0.000
z 0.06 n  12186 sat_mean 0.76 frac_low 0.193
z 0.08 n   3891 sat_mean 0.13 frac_low 1.000
z 0.10 n   3446 sat_mean 0.12 frac_low 1.000
z 0.12 n   3194 sat_mean 0.09 frac_low 1.000
z 0.14 n   3189 sat_mean 0.08 frac_low 1.000
z 0.16 n   5290 sat_mean 0.86 frac_low 0.040
z 0.18 n   3891 sat_mean 0.90 frac_low 0.000
z 0.20 n   2947 sat_mean 0.90 frac_low 0.000
z 0.22 n   2381 sat_mean 0.90 frac_low 0.000
z 0.24 n   2205 sat_mean 0.91  …[+283 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
tex=np.asarray(m.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255
uv=m.visual.uv; H,W=tex.shape[:2]
px=(uv[:,0]*(W-1)).round().astype(int).clip(0,W-1); py=((1-uv[:,1])*(H-1)).round().astype(int).clip(0,H-1)
col=tex[py,px]; mx=col.max(1); mn=col.min(1); sat=(mx-mn)/(mx+1e-3)
v=m.vertices; z=v[:,2]; r=np.hypot(v[:,0]-AXIS_C[0],v[:,1]-AXIS_C[1]); az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
sel=(z>-0.08)&(z<0.05)
# outer: r > 0.85 * max r at that z
rb=np.zeros_like(r)
for zz in np.arange(-0.08,0.05,0.01):
    s=sel&(z>=zz)&(z<zz+0.01); rb[s]=np.percentile(r[s],95)
o=sel&(r>rb-0.012)
for th in [0.6,0.7,0.75,0.8,0.85]:
    low=o&(sat<th)
    h,_=np.histogram(az[low],bins=24,range=(-180,180)); print(th,low.sum(),'az hist',h)
low=o&(sat<0.75)
print('z of low', np.percentile(z[low],[5,50,95]), 'mean col', col[low].mean(0), 'orange col', col[o&(sat>0.85)].mean(0))
EOF
```

<details><summary>output</summary>

```
0.6 0 az hist [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
0.7 103 az hist [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  8 94  1  0  0]
0.75 468 az hist [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  32 224 152   8  52   0]
0.8 1652 az hist [  0   0   0   0   0   0   7   0   0   0   0   0   0   0   0   0   0   0
 282 753 345  98 167   0]
0.85 4152 az hist [  16    0  119   45   77   57  334  223   10    0    0    0    7   22
    0    0    0   15  601 1258  745  366  228   29]
z of low [-0.04422871 -0.01959853  0.01004061] mean col [0.66181606 0.31236836 0.1843217 ] orange col [0.7416868  0.17757647 0.06474584]
```
</details>

**Bash**
```
cd /work/scratch; cat > blotch_test.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
import nvdiffrast.torch as dr
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
v=np.asarray(ep.mesh.vertices); az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
attr=np.stack([az, v[:,2]],1).astype(np.float32)
attr_t=torch.tensor(attr,device='cuda')
def render_attr(Tc,crop,size):
    x0,y0,s=crop; sc=size/s; K=ep.K
    fx,fy=K[0,0]*sc,K[1,1]*sc; cx,cy=(K[0,2]-x0)*sc,(K[1,2]-y0)*sc; n,fz=0.05,20.
    P=torch.tensor([[2*fx/size,0,2*cx/size-1,0],[0,2*fy/size,2*cy/size-1,0],[0,0,(fz+n)/(fz-n),-2*fz*n/(fz-n)],[0,0,1,0]],device='cuda')
    Tc=torch.tensor(Tc,device='cuda',dtype=torch.float32)
    vh=torch.cat([rd.v,torch.ones_like(rd.v[:,:1])],1)
    clip=torch.einsum('ij,bvj->bvi',P,torch.einsum('bij,vj->bvi',Tc,vh)).contiguous()
    rast,_=dr.rasterize(rd.ctx,clip,rd.f,(size,size))
    a,_=dr.interpolate(attr_t[None],rast,rd.f)
    return a.cpu().numpy(), (rast[...,3]>0).cpu().numpy()
a,b,T=int(sys.argv[1]),int(sys.argv[2]),np.load(sys.argv[3])
fr=ep.frames()[a:b].astype(np.float32)
ctr=proj(ep.K,(T@np.array([0,0,0.09,1]))[None,:3])[0]; h=int(0.2*ep.K[0,0]/T[2,3]); crop=crop_box(ctr,h)
x0,y0,s=crop
med=np.median(fr[:,y0:y0+s,x0:x0+s],0)/255.
S=s*2
med=cv2.resize(med,(S,S),interpolation=cv2.INTER_CUBIC)
mx=med.max(-1); mn=med.min(-1); vs=(mx-mn)/(mx+1e-3)
for off in [0,90,180,270]:
    att,cov=render_attr(axis_rot(T,off)[None] …[+912 chars]
```

<details><summary>output</summary>

```
Exit code 1
Traceback (most recent call last):
  File "/work/scratch/blotch_test.py", line 28, in <module>
    att,cov=render_attr(axis_rot(T,off)[None],crop,S)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/blotch_test.py", line 15, in render_attr
    clip=torch.einsum('ij,bvj->bvi',P,torch.einsum('bij,vj->bvi',Tc,vh)).contiguous()
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/functional.py", line 402, in einsum
    return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: expected scalar type Float but found Double
Traceback (most recent call last):
  File "/work/scratch/blotch_test.py", line 28, in <module>
    at …[+585 chars]
```
</details>

**Bash**
```
cd /work/scratch; sed -i "s/\[0,0,1,0\]\],device='cuda')/[0,0,1,0]],device='cuda',dtype=torch.float32)/" blotch_test.py; python blotch_test.py 0 95 Tm_0.npy; python blotch_test.py 345 440 Tm_345.npy
```

<details><summary>output</summary>

```
off 0 blotch px 0 video sat blotch nan rest 0.918 value blotch nan rest 0.676
off 90 blotch px 2 video sat blotch 0.922 rest 0.920 value blotch 0.635 rest 0.675
off 180 blotch px 1452 video sat blotch 0.954 rest 0.916 value blotch 0.641 rest 0.679
off 270 blotch px 1808 video sat blotch 0.905 rest 0.929 value blotch 0.699 rest 0.669
off 0 blotch px 705 video sat blotch 0.841 rest 0.898 value blotch 0.936 rest 0.933
off 90 blotch px 0 video sat blotch nan rest 0.881 value blotch nan rest 0.929
off 180 blotch px 0 video sat blotch nan rest 0.879 value blotch nan rest 0.925
off 270 blotch px 805 video sat blotch 0.934 rest 0.879 value blotch 0.924 rest 0.934
```
</details>

**Bash**
```
cd /work/scratch; python -c "
import cv2,numpy as np
a=cv2.imread('sat_345.png',0); b=cv2.imread('sat_0.png',0)
a=cv2.resize(a,(400,400)); b=cv2.resize(b,(400,400))
# stretch contrast in 0.7-1 range
f=lambda x: np.clip((x.astype(float)-180)/(255-180)*255,0,255).astype(np.uint8)
cv2.imwrite('sat_both.png',np.hstack([f(b),f(a)]))
"
```

**Read**
```
/work/scratch/sat_both.png
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
cd /work/scratch; cat >> blotch_test.py <<'EOF'
f=lambda x: np.clip((x-0.7)/0.3,0,1)
tiles=[]
for off in [0,90,180,270]:
    att,cov=render_attr(axis_rot(T,off)[None],crop,S)
    azr=att[0,...,0]; zr=att[0,...,1]; cv0=cov[0]
    bl=cv0&(azr>95)&(azr<140)&(zr>-0.045)&(zr<0.01)
    im=np.repeat((f(vs)*255).astype(np.uint8)[...,None],3,2).copy()
    cs,_=cv2.findContours(bl.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im,cs,-1,(0,0,255),2)
    cs,_=cv2.findContours(cv0.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im,cs,-1,(0,255,0),1)
    tiles.append(cv2.resize(im,(300,300)))
cv2.imwrite(f'blotch_ov_{a}.png',np.hstack(tiles))
EOF
python blotch_test.py 0 95 Tm_0.npy >/dev/null; python blotch_test.py 345 440 Tm_345.npy > /dev/null; python -c "
import cv2,numpy as np; cv2.imwrite('blotch_ov.png',np.vstack([cv2.imread('blotch_ov_0.png'),cv2.imread('blotch_ov_345.png')]))"
```

**Read**
```
/work/scratch/blotch_ov.png
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
cd /work/scratch; cat > hmm_twist.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs')
d=np.load('carry_profile.npz'); fr=list(d['frames']); g=d['grid']; iou=d['iou'].reshape(len(fr),len(g)); Tf=d['Tf'].reshape(len(fr),len(g),4,4)
R0=ep.T_base[0,:3,:3]
sig_iou=float(sys.argv[1]) if len(sys.argv)>1 else 0.01
sig_step=float(sys.argv[2]) if len(sys.argv)>2 else 6.0
drift_occ=float(sys.argv[3]) if len(sys.argv)>3 else 85.0
X=np.arange(-360,360,2.5)  # absolute twist states
F0,F1=116,345
unary=np.zeros((F1-F0+1,len(X)))
for k,i in enumerate(fr):
    if i<F0 or i>F1: continue
    tws=np.array([twist_swing(R0.T@(ep.T_world_cam@Tf[k,j])[:3,:3])[0] for j in range(len(g))])
    # periodic interpolation of iou over twist mod 90
    o=np.argsort(tws%90); xs=np.concatenate([(tws%90)[o]-90,(tws%90)[o],(tws%90)[o]+90]); ys=np.tile(iou[k][o],3)
    prof=np.interp(X%90,xs,ys)
    unary[i-F0]=-0.5*((prof-prof.max())/sig_iou)**2
# anchors: A static twist ~ -1.3 at frame 116 (strong), B static: -68 mod 90 at 345
def wrapd(a): return (a+45)%90-45
unary[0]+= -0.5*((X-(-1.3))/3.0)**2
unary[-1]+= -0.5*(wrapd(X-(-68.6))/3.0)**2
# occluded frames: no unary.  transitions
n=F1-F0+1
face=None
logp=np.zeros((n,len(X))); bp=np.zeros((n,len(X)),int)
D=X[None,:]-X[:,None]  # to - from
alpha=unary[0].copy(); A=[alpha]
for t in range(1,n):
    i=F0+t
    mu=drift_occ/51.0 if 197<=i<=247 else 0.0
    tr=-0.5*((D-mu)/sig_step)**2
    m=alpha …[+820 chars]
```

<details><summary>output</summary>

```
B branch -158.6  P=0.000
B branch  -68.6  P=0.001
B branch   21.4  P=0.026
B branch  111.4  P=0.250
B branch  201.4  P=0.620
B branch  291.4  P=0.104
116:0 121:0 126:0 131:0 136:10 141:22 146:35 151:52 156:60 161:72 166:85 171:85 176:88 181:88 186:82 191:80 196:75 201:72 206:72 211:72 216:78 221:90 226:102 231:115 236:128 241:140 246:152 251:160 256:168 261:172 266:178 271:175 276:168 281:160 286:165 291:165 296:165 301:165 306:178 311:185 316:188 321:195 326:198 331:200 336:200 341:200
B branch -158.6  P=0.001
B branch  -68.6  P=0.022
B branch   21.4  P=0.228
B branch  111.4  P=0.625
B branch  201.4  P=0.120
B branch  291.4  P=0.004
Traceback (most recent call last):
  File "/work/scratch/hmm_twist.py", line 50, in <module>
    print(' '.join('%d:%.0f'%(F0+t,path[t]) for t in range(0,n,5) …[+393 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > fitB_vis.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
a,b=int(sys.argv[1]),int(sys.argv[2])
frames=list(range(a,b,6))
Tm=np.load(f'Tm_{a}.npy')
tg=MaskTargets(ep,frames,np.repeat(Tm[None],len(frames),0),size=192)
offs=np.arange(0,90,10)
R0=ep.T_base[0,:3,:3]
tw0=twist_swing(R0.T@(ep.T_world_cam@Tm)[:3,:3])[0]
outs=[]
for o in offs:
    T0c=torch.tensor(axis_rot(Tm,o-tw0+(-90 if a>300 else 0))[None],device='cuda',dtype=torch.float32)
    w=torch.zeros(1,3,device='cuda',requires_grad=True); t=torch.zeros(1,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.01)
    for it in range(120):
        T=compose(T0c,w,t,0.0,True).expand(len(frames),4,4)
        iou=soft_iou(render_masks(rd,T,ep.K,tg.crops,tg.size),tg)
        loss=1-iou.mean(); opt.zero_grad(); loss.backward(); opt.step()
    Tf=T[0].detach().cpu().numpy().astype(np.float64)
    tw,sw=twist_swing(R0.T@(ep.T_world_cam@Tf)[:3,:3])
    print('twist %.1f swing %.1f iou %.4f'%(tw,sw,iou.mean().item()))
    outs.append((iou.mean().item(),Tf,tw))
best=max(outs,key=lambda x:x[0])
# visual of median image + mask contours of best and of best+45
fr=ep.frames()[a:b:6]
crop=tg.crops[0]; x0,y0,s=crop
med=np.median(np.stack([cut(f,crop,384,cv2.INTER_CUBIC) for f in fr]),0).astype(np.uint8)[:,:,::-1].copy()
for (T,col) in [(best[1],(0,255,0)),(axis_rot(best[1],45),(0,0,255))]:
   …[+448 chars]
```

**Read**
```
/work/scratch/fitvis_345.png
```

<details><summary>output</summary>

```
twist -90.0 swing 5.4 iou 0.8424
twist -80.1 swing 6.1 iou 0.8538
twist -70.0 swing 6.1 iou 0.8624
twist -60.0 swing 6.3 iou 0.8563
twist -50.0 swing 5.9 iou 0.8529
twist -40.0 swing 5.8 iou 0.8457
twist -30.0 swing 6.3 iou 0.8416
twist -20.1 swing 7.1 iou 0.8496
twist -10.1 swing 6.0 iou 0.8518
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
import re
s=open('hmm_twist.py').read()
s=s.replace("face=None","""r=np.load('/work/inputs/reconstruction.npz'); Kp=r['human_keypoints_world'].astype(np.float64)
mode=sys.argv[4] if len(sys.argv)>4 else 'occ'
up=np.array([0,-1.,0])
if mode=='wrist': vv=Kp[:,62]-Kp[:,41]; yy=np.degrees(np.arctan2(vv[:,0],vv[:,2]))
else:
    fc=np.cross(up,(Kp[:,5]-Kp[:,6])+(Kp[:,9]-Kp[:,10])); yy=np.degrees(np.arctan2(fc[:,0],fc[:,2]))
yy=np.degrees(np.unwrap(np.radians(yy)))
from scipy.ndimage import uniform_filter1d
yy=uniform_filter1d(yy,5)
dface=-np.diff(yy,prepend=yy[0])  # predicted twist increment per frame""")
s=s.replace("mu=drift_occ/51.0 if 197<=i<=247 else 0.0","mu=(drift_occ/51.0 if 197<=i<=247 else 0.0) if mode=='occ' else dface[i]")
s=s.replace("i=F0+t; mu=drift_occ/51.0 if 197<=i<=247 else 0.0","i=F0+t; mu=(drift_occ/51.0 if 197<=i<=247 else 0.0) if mode=='occ' else dface[i]")
open('hmm_twist.py','w').write(s)
EOF
for m in face wrist; do for sg in 4 6 10; do echo "== $m sig_step $sg"; python hmm_twist.py 0.01 $sg 0 $m | head -6 | tr '\n' ' '; echo; done; done; python hmm_twist.py 0.01 6 0 face | tail -1
```

<details><summary>output</summary>

```
== face sig_step 4
B branch -158.6  P=0.000 B branch  -68.6  P=0.000 B branch   21.4  P=0.000 B branch  111.4  P=0.003 B branch  201.4  P=0.862 B branch  291.4  P=0.135 Traceback (most recent call last):
  File "/work/scratch/hmm_twist.py", line 59, in <module>
    print(' '.join('%d:%.0f'%(F0+t,path[t]) for t in range(0,n,5)))
BrokenPipeError: [Errno 32] Broken pipe

== face sig_step 6
B branch -158.6  P=0.000 B branch  -68.6  P=0.000 B branch   21.4  P=0.000 B branch  111.4  P=0.040 B branch  201.4  P=0.591 B branch  291.4  P=0.369 Traceback (most recent call last):
  File "/work/scratch/hmm_twist.py", line 59, in <module>
    print(' '.join('%d:%.0f'%(F0+t,path[t]) for t in range(0,n,5)))
BrokenPipeError: [Errno 32] Broken pipe

== face sig_step 10
B branch -158.6  P=0.000 B branch  -68 …[+1773 chars]
```
</details>

> *thinking:* The HMM analysis favors a table yaw of −158° over the baseline's −68° (which only gets 10–37% posterior support), suggesting the baseline's +90° jump was spurious. The scuff mark is the only evidence against −158°, so I'm now checking each frame to see if it actually appears in the video.

**Bash**
```
cd /work/scratch; cat > blotch_frames.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoilib import *
import nvdiffrast.torch as dr
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
v=np.asarray(ep.mesh.vertices); az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
attr_t=torch.tensor(np.stack([az,v[:,2]],1).astype(np.float32),device='cuda')
def render_attr(Tc,crop,size):
    x0,y0,s=crop; sc=size/s; K=ep.K
    fx,fy=K[0,0]*sc,K[1,1]*sc; cx,cy=(K[0,2]-x0)*sc,(K[1,2]-y0)*sc; n,fz=0.05,20.
    P=torch.tensor([[2*fx/size,0,2*cx/size-1,0],[0,2*fy/size,2*cy/size-1,0],[0,0,(fz+n)/(fz-n),-2*fz*n/(fz-n)],[0,0,1,0]],device='cuda',dtype=torch.float32)
    Tc=torch.tensor(Tc,device='cuda',dtype=torch.float32)
    vh=torch.cat([rd.v,torch.ones_like(rd.v[:,:1])],1)
    clip=torch.einsum('ij,bvj->bvi',P,torch.einsum('bij,vj->bvi',Tc,vh)).contiguous()
    rast,_=dr.rasterize(rd.ctx,clip,rd.f,(size,size))
    a,_=dr.interpolate(attr_t[None],rast,rd.f)
    return a.cpu().numpy(), (rast[...,3]>0).cpu().numpy()
d=np.load('carry_profile.npz'); fr=list(d['frames']); g=d['grid']; iou=d['iou'].reshape(len(fr),len(g)); Tf=d['Tf'].reshape(len(fr),len(g),4,4)
R0=ep.T_base[0,:3,:3]
S=160
rows=[]
for k,i in enumerate(fr):
    if i%2: continue
    j=iou[k].argmax(); T=Tf[k,j]
    ctr=proj(ep.K,(T@np.array([0,0,0.09,1]))[None,:3])[0]; h=int(0.2*ep.K[0,0]/T[2,3]); crop=crop_box(ctr,h)
    im=cut(ep.frames()[i],crop,S,cv2.INTER_AREA).astype(np.float32)/2 …[+1354 chars]
```

<details><summary>output</summary>

```
0 0 
90 34 [-0.234 -0.136 -0.082  0.007  0.064]
180 90 [-0.029  0.006  0.034  0.058  0.099]
270 56 [-0.142 -0.05  -0.022 -0.012  0.053]
```
</details>

**Bash**
```
cd /work/scratch; head -3 blotch_frames.txt; grep -c . blotch_frames.txt; awk '{print $1,$2,$3,$0}' blotch_frames.txt | grep -E "^(1[2-9][0-9]|2[4-9][0-9]|3[0-4][0-9]) " | cut -c1-200 | head -120
```

<details><summary>output</summary>

```
110 tw     0 off  0 n   0 c nan | off 90 n   1 c nan | off180 n 210 c 0.018 | off270 n 254 c-0.020
112 tw     0 off  0 n   0 c nan | off 90 n   1 c nan | off180 n 211 c 0.010 | off270 n 255 c 0.002
114 tw     0 off  0 n   0 c nan | off 90 n   0 c nan | off180 n 207 c 0.017 | off270 n 255 c-0.019
92
120 tw -0 120 tw    -0 off  0 n   0 c nan | off 90 n   0 c nan | off180 n 214 c 0.014 | off270 n 257 c-0.022
122 tw -0 122 tw    -0 off  0 n   0 c nan | off 90 n   1 c nan | off180 n 211 c 0.008 | off270 n 259 c-0.025
124 tw -0 124 tw    -0 off  0 n   0 c nan | off 90 n   0 c nan | off180 n 217 c 0.005 | off270 n 266 c-0.031
126 tw -0 126 tw    -0 off  0 n   0 c nan | off 90 n   0 c nan | off180 n 214 c 0.025 | off270 n 267 c-0.011
128 tw -0 128 tw    -0 off  0 n   0 c nan | off 90 n   1 c nan | …[+8959 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, trimesh
from hoilib import AXIS_C
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
tex=np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
uv=m.visual.uv; H,W=tex.shape[:2]
v=m.vertices; az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
s=(az>100)&(az<135)&(v[:,2]>-0.04)&(v[:,2]<0.005)
px=(uv[s,0]*(W-1)); py=((1-uv[s,1])*(H-1))
print(px.min(),px.max(),py.min(),py.max())
# cluster by patches: show bounding boxes of histogram
h,xe,ye=np.histogram2d(px,py,bins=20,range=[[0,W],[0,H]])
idx=np.argwhere(h>50); print(idx[:20], h[h>50][:20])
x0=int(np.percentile(px,2)); x1=int(np.percentile(px,98)); y0=int(np.percentile(py,2)); y1=int(np.percentile(py,98))
EOF
```

<details><summary>output</summary>

```
4.950588776730001 2217.775607585907 97.5637703537941 2258.7920491695404
[[ 0 18]
 [ 1 19]
 [ 2 18]
 [ 2 19]
 [ 6 18]
 [ 7 18]
 [ 8 18]
 [ 9 18]
 [ 9 19]
 [10 19]
 [11  0]
 [11  1]
 [12 12]
 [12 17]
 [17 18]
 [18 18]
 [19 18]] [ 68. 124. 135.  87.  97.  52. 113. 136.  58. 108.  54. 218.  68. 328.
  99. 106. 106.]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import numpy as np, cv2, trimesh, sys
sys.path.insert(0,'/work/outputs'); from hoilib import AXIS_C
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
tex=np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
uv=m.visual.uv; H,W=tex.shape[:2]
v=m.vertices; az=np.degrees(np.arctan2(v[:,1]-AXIS_C[1],v[:,0]-AXIS_C[0]))
s=(az>100)&(az<135)&(v[:,2]>-0.04)&(v[:,2]<0.005)
px=(uv[s,0]*(W-1)).astype(int); py=((1-uv[s,1])*(H-1)).astype(int)
mask=np.zeros((H,W),np.uint8); mask[py,px]=255
mask=cv2.dilate(mask,np.ones((9,9)))
n,lab,st,_=cv2.connectedComponentsWithStats(mask)
o=np.argsort(-st[1:,4])[:4]+1
tiles=[]
for k in o:
    x,y,w,h,a=st[k]; c=tex[max(0,y-20):y+h+20,max(0,x-20):x+w+20]
    c=cv2.resize(c,(300,int(300*c.shape[0]/c.shape[1])) if c.shape[1]>c.shape[0] else (int(300*c.shape[1]/c.shape[0]),300))
    print(st[k]); tiles.append(c)
H2=max(t.shape[0] for t in tiles)
tiles=[np.pad(t,((0,H2-t.shape[0]),(0,10),(0,0))) for t in tiles]
cv2.imwrite('blotch_tex.png',np.hstack(tiles)[:,:,::-1])
EOF
```

**Read**
```
/work/scratch/blotch_tex.png
```

<details><summary>output</summary>

```
[1238   93   52   80 1606]
[1392 1433   75   69 1136]
[1422 1984   34   39  966]
[ 961 2139   36   32  684]
```
</details>

<details><summary>output</summary>

```
[image]
```
</details>

> *thinking:* I've confirmed the scuff is just a scan artifact from a small texture atlas chart, so I'll disregard it. Now I'll implement `solve.py` with automatic phase detection, per-phase pose fitting for floor/table, HMM-decoded yaw profiles for the carried phase, hand-based cone tracking for the hidden gap, and smoothed depth re-anchored to each phase's baseline median.

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs')
om=ep.obj_mask; pm=ep.person_mask
refA=om[0]; refB=np.median(om[-30:].astype(np.uint8),0)>0
for ref,name,rng in [(refA,'A',range(100,140)),(refB,'B',range(300,350))]:
    s=[]
    for t in rng:
        v=~pm[t]|om[t]
        i=(om[t]&ref&v).sum(); u=((om[t]|ref)&v).sum()
        s.append('%d:%.2f'%(t,i/max(u,1)))
    print(name,' '.join(s))
# floor normal vs cone axis
n=ep.plane[:3]; n=n/np.linalg.norm(n)
for t in [0,400]:
    ax=ep.T_base[t,:3,2]; print(t,'angle axis-floor normal',np.degrees(np.arccos(abs(ax@n))))
EOF
```

<details><summary>output</summary>

```
A 100:0.98 101:0.98 102:0.99 103:0.99 104:0.99 105:0.99 106:0.99 107:0.99 108:0.99 109:0.99 110:0.98 111:0.98 112:0.98 113:0.98 114:0.99 115:0.98 116:0.98 117:0.98 118:0.98 119:0.98 120:0.98 121:0.99 122:0.98 123:0.98 124:0.97 125:0.91 126:0.83 127:0.74 128:0.60 129:0.49 130:0.39 131:0.31 132:0.24 133:0.18 134:0.13 135:0.10 136:0.07 137:0.04 138:0.02 139:0.01
B 300:0.00 301:0.00 302:0.00 303:0.00 304:0.00 305:0.01 306:0.06 307:0.14 308:0.22 309:0.33 310:0.46 311:0.55 312:0.65 313:0.73 314:0.79 315:0.84 316:0.88 317:0.89 318:0.88 319:0.87 320:0.92 321:0.88 322:0.89 323:0.89 324:0.90 325:0.88 326:0.87 327:0.86 328:0.89 329:0.89 330:0.91 331:0.93 332:0.89 333:0.90 334:0.95 335:0.96 336:0.93 337:0.93 338:0.93 339:0.98 340:0.95 341:0.98 342:0.95 343:0.95 344:0.96 345:0.96 346:0.99 347:0.96 348: …[+104 chars]
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
p='solve.py'; s=open(p).read()
s=s.replace("""    axis_c = np.array([(Tcw @ good[t])[:3, 2] for t in ts])
    pos_c = np.array([(Tcw @ good[t])[:3, 3] for t in ts])""","""    cl = np.array(list(AXIS_C[:2]) + [0.09])      # cone centre (mesh coords); positions below are centres
    ctr = lambda T: T[:3, :3] @ cl + T[:3, 3]      # noqa: E731
    axis_c = np.array([(Tcw @ good[t])[:3, 2] for t in ts])
    pos_c = np.array([ctr(Tcw @ good[t]) for t in ts])""")
s=s.replace("""    pos_all = np.concatenate([[TA[:3, 3]], pos_c, [(Tcw @ TB_w)[:3, 3]]])""","""    pos_all = np.concatenate([[ctr(TA)], pos_c, [ctr(Tcw @ TB_w)]])""")
s=s.replace("""        # the swing rotates about the cone centre; place the centre at pos_i
        cl = np.array(list(AXIS_C[:2]) + [0.09])
        Tc[:3, 3] = pos_i[s] - Tc[:3, :3] @ cl
        Tc0 = Tcw @ TA_w
        Tc[:3, 3] = pos_i[s] - Tc[:3, :3] @ cl + 0.0
""","""        Tc[:3, 3] = pos_i[s] - Tc[:3, :3] @ cl     # place the cone centre at pos_i
""")
s=s.replace("""    # positions were stored as the cone-centre position in the fits; convert consistently:
    # (good[] poses carry their own translation; recompute the centre-based translation for them)
""","")
open(p,'w').write(s)
EOF
grep -n "set_twist(Tw_t" solve.py
```

<details><summary>output</summary>

```
318:        Tw[t] = set_twist(Tw_t, R0, tw_path_s[s])
```
</details>

**Bash**
```
cd /work/scratch && python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/run1.npz --debug /work/scratch/run1_dbg.npz 2>&1 | tail -20
```

<details><summary>output</summary>

```
phases: static-floor [0,125]  carried [126,334]  static-table [335,439]
static floor: IoU 0.9299, twist -1.2 (baseline 0.0)
static table: IoU 0.8543, twist mod 90 = 15.0 (profile range 0.023)
table-phase twist branch posterior: -165.0: 0.397, -75.0: 0.587, 15.0: 0.001, 105.0: 0.016
(baseline table twist -71.3)
decoded table twist 287.5
depth re-anchoring scale factors: floor: 1.0002, carried: 1.0002, table: 0.9988
done in 109.5s -> /work/scratch/run1.npz
```
</details>

**Bash**
```
python -c "
import numpy as np
d=np.load('run1_dbg.npz'); p=d['tw_path']; s=d['seq']
print(' '.join('%d:%.0f'%(a,b) for a,b in zip(s[::4],p[::4])))
"
```

<details><summary>output</summary>

```
125:0 129:0 133:0 137:10 141:18 145:28 149:42 153:58 157:65 161:72 165:82 169:85 173:85 177:88 181:88 185:88 189:80 193:75 197:82 201:90 205:98 209:98 213:100 217:100 221:100 225:100 229:105 233:115 237:122 241:125 245:135 249:148 253:158 257:172 261:172 265:180 269:190 273:210 277:240 281:250 285:255 289:258 293:258 297:255 301:252 305:255 309:260 313:268 317:270 321:272 325:270 329:275 333:282
```
</details>

> *thinking:* Shifting the table yaw estimate by 6° flips the posterior to a near coin-flip (0.59/0.40 vs 0.62 before), so since evidence isn't decisive I'll keep the baseline branch for the table phase, switching only if another branch exceeds posterior 0.7. Now building the validation (mask IoU per phase, smoothness, depth difference).

**Bash**
```
cd /work/outputs; python - <<'EOF'
p='solve.py'; s=open(p).read()
s=s.replace('''SIG_STEP = 6.0           # deg/frame std of twist change w.r.t. the body-yaw prediction''','''SIG_STEP = 6.0           # deg/frame std of twist change w.r.t. the body-yaw prediction
BRANCH_SWITCH_P = 0.7    # table-phase 90-deg branch differs from the baseline's only if its posterior exceeds this''')
s=s.replace('''    # Viterbi (max-product) for the MAP twist path
    delta = unary[0].copy(); BP = []''','''    # table-phase branch: keep the baseline's branch unless the HMM clearly favours another one
    best_b = max(branches, key=branches.get)
    base_b = min(branches, key=lambda k: abs(((k - base_twB) + 180) % 360 - 180))
    chosen = best_b if branches[best_b] > BRANCH_SWITCH_P else base_b
    log(f"table branch: HMM-best {best_b:.1f} (p={branches[best_b]:.3f}), baseline-branch {base_b:.1f} "
        f"(p={branches[base_b]:.3f}) -> using {chosen:.1f}")
    unary[-1] += -0.5 * ((((X - chosen) + 180) % 360 - 180) / 3.0) ** 2
    # Viterbi (max-product) for the MAP twist path
    delta = unary[0].copy(); BP = []''')
open(p,'w').write(s)
EOF
cat > /work/scratch/evaluate.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
res=np.load(sys.argv[1])['T_world_mesh']; a_end=int(sys.argv[2]); b_start=int(sys.argv[3])
F=ep.F; Tc_new=ep.T_cam_world[None]@res; Tc_b=ep.Tc_base
def ious(Tc):
    out=np …[+2700 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bc0fdh0kg). Output is being written to: /tmp/claude-1003/-work/1b245680-4fe9-485a-b2b1-64fef49ff78f/tasks/bc0fdh0kg.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/scratch; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
sleep 100; cat /tmp/claude-1003/-work/1b245680-4fe9-485a-b2b1-64fef49ff78f/tasks/bc0fdh0kg.output
```

<details><summary>output</summary>

```
(Bash completed with no output)
```
</details>

**Bash**
```
sleep 60; cat /tmp/claude-1003/-work/1b245680-4fe9-485a-b2b1-64fef49ff78f/tasks/bc0fdh0kg.output
```

<details><summary>output</summary>

```
static floor: IoU 0.9297, twist -1.2 (baseline 0.0)
static table: IoU 0.8557, twist mod 90 = 20.0 (profile range 0.022)
table-phase twist branch posterior: -160.0: 0.401, -70.0: 0.584, 20.0: 0.000, 110.0: 0.015
(baseline table twist -71.3)
table branch: HMM-best -70.0 (p=0.584), baseline-branch -70.0 (p=0.584) -> using -70.0
decoded table twist 290.0
depth re-anchoring scale factors: floor: 1.0003, carried: 1.0003, table: 0.9991
done in 258.1s -> /work/scratch/run2.npz
| phase | frames | IoU base | IoU ours | n vis | med rot step base/ours (deg) | p95 rot step base/ours | med trans step base/ours (cm) | dz mean / median (cm) | mean ADD(base,ours) cm |
|---|---|---|---|---|---|---|---|---|---|
| floor | 0-125 | 0.852 | 0.941 | 126 | 0.00 / 0.00 | 0.8 / 0.0 | 0.01 / 0.00 | -0.21 / 0.00 | 2.4 …[+5319 chars]
```
</details>

> *thinking:* Mask IoU and rotation stability improved over baseline across all phases (floor, carried, table), while median depth stayed consistent. I'll now visually inspect the carried frames to confirm.

**Bash**
```
cd /work/scratch; python vis_base.py run2.npz 120,130,140,150,160,170,180,190,196,250,256,264,272,284,296,306,316,330 vr2.jpg
```

**Read**
```
/work/scratch/vr2.jpg
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
import h5py; f=h5py.File('/pipeline/depth/moge2/aligned.h5'); 
def p(n,o):
    print(n,getattr(o,'shape',''),getattr(o,'dtype',''))
f.visititems(p)
"; cat /pipeline/depth/moge2/depth_report.json | head -40
```

<details><summary>output</summary>

```
aligned  
aligned/front_stereo_camera_left (440,) object
alignment  
alignment/front_stereo_camera_left  
alignment/front_stereo_camera_left/scale (440,) float32
alignment/front_stereo_camera_left/shift (440,) float32
alignment/front_stereo_camera_left/valid_count (440,) int32
frame_names  
frame_names/front_stereo_camera_left (440,) object
raw  
raw/front_stereo_camera_left (440,) object
{
  "backend": "moge2",
  "batchSize": 8,
  "cameraPolicy": "first_frame_export_per_frame_depth",
  "depthM": {
    "max": 10.041121482849121,
    "min": 1.3992319107055664
  },
  "deviceCount": 1,
  "devices": [
    "cuda"
  ],
  "elapsedSeconds": 51.123739999951795,
  "firstFrameIntrinsicsPx": [
    [
      841.2771606445312,
      0.0,
      768.0
    ],
    [
      0.0,
      841.277099609375,
      5 …[+280 chars]
```
</details>

**Bash**
```
cd /work/scratch; cat > depth_cues.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
res=np.load(sys.argv[1])['T_world_mesh']; a_end,b_start=125,335
Tc=ep.T_cam_world[None]@res
n=ep.plane[:3]; d=ep.plane[3]; nn=np.linalg.norm(n); n=n/nn; d=d/nn
V=np.asarray(ep.mesh.vertices)
# (a) floor contact in static floor phase: lowest vertex (along gravity / plane normal) distance to plane
T=res[0]; Vw=(T[:3,:3]@V.T).T+T[:3,3]
sd=Vw@n+d   # signed distance
print('floor phase: signed plane distance of mesh points: min %.3f max %.3f (n=%s)'%(sd.min(),sd.max(),np.round(n,3)))
# side of the camera: camera centre
cc=ep.T_world_cam[:3,3]; print('camera signed dist %.3f'%(cc@n+d))
# shift along viewing ray needed to bring the lowest point onto the plane
ctr_c=Tc[0,:3,3]; ray=ctr_c/np.linalg.norm(ctr_c); ray_w=ep.T_world_cam[:3,:3]@ray
# moving by s along ray changes plane distance by s*(ray_w.n); choose sign so that the contact distance (the extreme point toward the floor) becomes 0
contact=sd.min() if abs(sd.min())<abs(sd.max()) else sd.max()
# the base is the extreme on the floor side: floor is on the side opposite the camera
base_side = sd.max() if (cc@n+d)<0 else sd.min()
s=-base_side/(ray_w@n)
print('floor cue: base-to-plane distance %.3f m -> along-ray shift %.3f m (dz %.3f m)'%(base_side,s,s*ray[2]))
# (b) silhouette scale: pooled fit with z free
for name,a,b in [('floor',0,a_end+1),( …[+1680 chars]
```

<details><summary>output</summary>

```
floor phase: signed plane distance of mesh points: min -0.013 max 0.376 (n=[-0.019 -0.998 -0.056])
camera signed dist 1.375
floor cue: base-to-plane distance -0.013 m -> along-ray shift -0.023 m (dz -0.020 m)
floor silhouette-scale cue: dz -0.043 m (iou 0.9364)
table silhouette-scale cue: dz -0.191 m (iou 0.9222)
carry: wrist-to-cone-surface distance median 0.107 m, p25 0.080 p75 0.126
carry: cone centre z - wrist-mid z: median 0.013 m
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
res=np.load('run2.npz')['T_world_mesh']; Tc=ep.T_cam_world[None]@res
for t in [10,60,400,430]:
    m=ep.obj_mask[t]; ys,xs=np.nonzero(m)
    crop=(0,0,1536); 
    r,_=rd.render(Tc[t:t+1],ep.K,(0,192-192,1536),1536)  # square 1536 crop covering y 0..1536
    r=r[0].cpu().numpy()[:1152]>0.5
    yr,xr=np.nonzero(r)
    print(t,'mask bbox w %d h %d'%(xs.max()-xs.min(),ys.max()-ys.min()),'render bbox w %d h %d'%(xr.max()-xr.min(),yr.max()-yr.min()), 'area mask %d render %d'%(m.sum(),r.sum()))
EOF
```

<details><summary>output</summary>

```
10 mask bbox w 94 h 194 render bbox w 95 h 193 area mask 9847 render 9483
60 mask bbox w 92 h 193 render bbox w 95 h 193 area mask 9683 render 9483
400 mask bbox w 71 h 127 render bbox w 74 h 123 area mask 4404 render 3866
430 mask bbox w 69 h 126 render bbox w 74 h 123 area mask 4253 render 3866
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs')
res=np.load('run2.npz')['T_world_mesh']; Tc=ep.T_cam_world[None]@res
cl=np.array(list(AXIS_C[:2])+[0.09,1.0])
for j,name in [(41,'R'),(62,'L')]:
    w=(ep.T_cam_world[:3,:3]@ep.kp[:,j].T).T+ep.T_cam_world[:3,3]
    for a,b in [(126,160),(160,197),(248,300),(300,316),(316,335)]:
        d=np.array([w[t]-(Tc[t]@cl)[:3] for t in range(a,b)])
        print(name,a,b,'wrist - cone centre (cam xyz, cm) median',np.round(np.median(d,0)*100,1))
EOF
```

<details><summary>output</summary>

```
R 126 160 wrist - cone centre (cam xyz, cm) median [-10.9 -15.5  -4.7]
R 160 197 wrist - cone centre (cam xyz, cm) median [ -2.5  -1.7 -18.6]
R 248 300 wrist - cone centre (cam xyz, cm) median [ 2.1 -4.1  5. ]
R 300 316 wrist - cone centre (cam xyz, cm) median [ 5.  -6.3 11.8]
R 316 335 wrist - cone centre (cam xyz, cm) median [ 7.6 -8.2 12. ]
L 126 160 wrist - cone centre (cam xyz, cm) median [  5.9 -17.9  10.2]
L 160 197 wrist - cone centre (cam xyz, cm) median [-11.7  -1.1   3.3]
L 248 300 wrist - cone centre (cam xyz, cm) median [14.  -1.6 -7.9]
L 300 316 wrist - cone centre (cam xyz, cm) median [16.4 -4.8 -4.3]
L 316 335 wrist - cone centre (cam xyz, cm) median [15.5 -7.2 -6. ]
```
</details>

> *thinking:* Both floor and table depth cues fail the "two independent cues within 2 cm" rule, so I'm keeping the baseline depth throughout. Next, I'll run a CPU-only sensitivity check of the table-branch posterior over the HMM's noise parameters.

**Bash**
```
cd /work/scratch; cat > sens.py <<'EOF'
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import *
import solve
ep=Episode('/work/inputs')
d=np.load('run2_dbg.npz'); seq=list(d['seq']); vis=list(d['vis']); iouf=d['iouf']; tw_f=d['tw_f']; twA=float(d['twA'])
X=solve.HMM_STATES; yaw=solve.body_yaw(ep); dpred=-np.diff(yaw,prepend=yaw[0]); D=X[None,:]-X[:,None]
vk={t:k for k,t in enumerate(vis)}
for sig_iou in [0.01,0.02,0.03]:
  for sig_step in [4,6,10]:
    for prior in [1.0,0.5,0.0]:
      for twBm in [15,20]:
        n=len(seq); un=np.zeros((n,len(X)))
        for s,t in enumerate(seq):
            if t in vk:
                prof=solve.periodic_interp(X,tw_f[vk[t]],iouf[vk[t]]); un[s]=-0.5*((prof-prof.max())/sig_iou)**2
        un[0]+=-0.5*((X-twA)/3)**2; un[-1]+=-0.5*(solve.wrap90(X-twBm)/3)**2
        al=un[0]-un[0].max()
        for s in range(1,n):
            m=al[:,None]-0.5*((D-prior*dpred[seq[s]])/sig_step)**2; mx=m.max(0); al=mx+np.log(np.exp(m-mx).sum(0))+un[s]; al-=al.max()
        p=np.exp(al); p/=p.sum()
        out=[]
        for c in [twBm-180,twBm-90,twBm,twBm+90]:
            sel=np.abs(((X-c)+180)%360-180)<45; out.append('%4d:%.2f'%(c,p[sel].sum()))
        print('sig_iou %.2f step %2d prior %.1f B%2d | '%(sig_iou,sig_step,prior,twBm),' '.join(out))
EOF
python sens.py
```

<details><summary>output</summary>

```
sig_iou 0.01 step  4 prior 1.0 B15 |  -165:0.62  -75:0.38   15:0.00  105:0.00
sig_iou 0.01 step  4 prior 1.0 B20 |  -160:0.65  -70:0.35   20:0.00  110:0.00
sig_iou 0.01 step  4 prior 0.5 B15 |  -165:0.37  -75:0.57   15:0.00  105:0.06
sig_iou 0.01 step  4 prior 0.5 B20 |  -160:0.37  -70:0.56   20:0.00  110:0.06
sig_iou 0.01 step  4 prior 0.0 B15 |  -165:0.05  -75:0.00   15:0.04  105:0.92
sig_iou 0.01 step  4 prior 0.0 B20 |  -160:0.05  -70:0.00   20:0.04  110:0.91
sig_iou 0.01 step  6 prior 1.0 B15 |  -165:0.40  -75:0.59   15:0.00  105:0.01
sig_iou 0.01 step  6 prior 1.0 B20 |  -160:0.40  -70:0.58   20:0.00  110:0.01
sig_iou 0.01 step  6 prior 0.5 B15 |  -165:0.41  -75:0.46   15:0.01  105:0.12
sig_iou 0.01 step  6 prior 0.5 B20 |  -160:0.41  -70:0.46   20:0.01  110:0.12
sig_iou 0.01 step  6 …[+3411 chars]
```
</details>

> *thinking:* The table-branch posterior is sensitive to noise parameters—the -160° branch leads (0.35–0.79) but nothing robustly clears 0.7, so I'll keep the baseline branch. I'm also replacing the fine-yaw anchor, which jittered between 15° and 20° on a 5° grid argmax, with the free-twist pooled fit that converges consistently regardless of starting yaw.

**Bash**
```
cd /work/outputs; python - <<'EOF'
p='solve.py'; s=open(p).read()
old_start=s.index("    twB_prof, iouB_prof, TB_list = static_profile(ep, rd, framesB, R0)")
old_end=s.index("    # 3. carried phase")
s=s[:old_start]+'''    # table phase: free-twist pooled fit from several initial twists; the silhouette only fixes twist mod 90
    resB, TmB = fit_static(ep, rd, framesB, np.arange(-40, 41, 10))
    iouB, TB = max(resB, key=lambda r: r[0])
    twB_mod = twist_swing(R0.T @ (Twc @ TB)[:3, :3])[0]
    twB_all = np.array([twist_swing(R0.T @ (Twc @ T)[:3, :3])[0] for _, T in resB])
    log(f"static table: IoU {iouB:.4f}, twist mod 90 = {twB_mod % 90:.1f} "
        f"(all inits: {np.round(twB_all % 90, 1).tolist()})")

'''+s[old_end:]
s=s.replace('''    profB = periodic_interp(X, twB_prof, iouB_prof)
    unary[-1] += -0.5 * ((profB - profB.max()) / SIG_IOU) ** 2      # table anchor (mod 90 only)''','''    unary[-1] += -0.5 * (wrap90(X - twB_mod) / 4.0) ** 2           # table anchor (mod 90 only)''')
open(p,'w').write(s)
EOF
grep -n "static_profile\|twB_prof\|iouB_prof" solve.py
```

<details><summary>output</summary>

```
108:def static_profile(ep, rd, frames, R0):
```
</details>

**Bash**
```
cd /work/outputs; cp hoilib.py /work/scratch/hoilib.py; python - <<'EOF'
s=open('/work/outputs/solve.py').read()
a=s.index("def static_profile("); b=s.index("# ----------------------------------------------------------------------------- helpers")
s=s[:a]+s[b:]
lib=open('/work/outputs/hoilib.py').read()
lib=lib[lib.index('import json'):]
# drop the lib's own imports that solve already has; keep all
imp_start=s.index("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))")
imp_end=s.index("from scipy.ndimage import")
s=s[:imp_start]+"\n# ============================================================================= library (data, rendering, fitting)\n"+lib.replace('import json\nimport os\n','')+"\n\n# ============================================================================= solver\n"+s[imp_end:]
s=s.replace("from scipy.ndimage import gaussian_filter1d, uniform_filter1d  # noqa: E402","from scipy.ndimage import gaussian_filter1d, uniform_filter1d")
s=s.replace("from scipy.spatial.transform import Rotation as Rot, Slerp  # noqa: E402\n","")
open('/work/outputs/solve.py','w').write(s)
EOF
rm hoilib.py; rm -rf __pycache__; python -m pyflakes solve.py 2>/dev/null || pip list 2>/dev/null | grep -i flake; python -c "import ast;ast.parse(open('solve.py').read());print('ok')"; sed -n 1,60p solve.py | head -60
```

<details><summary>output</summary>

```
ok
#!/usr/bin/env python
"""Trajectory-level re-estimation of the traffic-cone 6-DoF poses for one HOI video.

Usage: python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

Pipeline (see REPORT.md for the reasoning):
 1. phase segmentation from the SAM2 object mask (static on floor / carried / static on table);
 2. static phases: ONE shared pose per phase, fitted by render-and-compare (silhouette IoU against
    the object mask, person pixels ignored) pooled over the whole phase; depth fixed to the baseline;
 3. carried phase: per-frame silhouette fits on a grid of twist angles about the cone axis
    (the cone is ~4-fold symmetric, so the silhouette only measures twist modulo 90 deg);
 4. an HMM over the absolute twist decodes the trajectory jointly: data term = …[+2133 chars]
```
</details>

**Bash**
```
cd /work/outputs; python - <<'EOF'
s=open('solve.py').read()
s=s.replace('''import numpy as np
import torch


# ============================================================================= library (data, rendering, fitting)

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
''','''import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial.transform import Rotation as Rot

# ============================================================================= library (data, rendering, fitting)
''')
s=s.replace('''# ============================================================================= solver
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
''','''# ============================================================================= solver
''')
s=s.replace("Deterministic: fixed seeds, deterministic grid inits, no randomness.","Deterministic up to GPU floating-point atomics in nvdiffrast's antialias gradients: fixed seeds,\ndeterministic grid initialisations, no sampling. Needs only the inputs (no /pipeline, no network).")
open('solve.py','w').write(s)
EOF
python -c "import ast;ast.parse(open('solve.py').read())"; grep -n "^import\|^from" solve.py; cd /work/scratch && timeout 900 python /work/outputs/solve.py --out /work/scratch/run3.npz --debug /work/s …[+35 chars]
```

<details><summary>output</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bpvyq1q95). Output is being written to: /tmp/claude-1003/-work/1b245680-4fe9-485a-b2b1-64fef49ff78f/tasks/bpvyq1q95.output. You will be notified when it completes. If it is still running after 10m in the background, it will be stopped and you will be notified. To check interim output, use Read on that file path.
Session cwd remains /work/outputs; directory changes made by the backgrounded command do not apply to subsequent commands.
```
</details>

**Bash**
```
python --version; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

<details><summary>output</summary>

```
Python 3.11.10
NVIDIA RTX 6000 Ada Generation
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
sleep 90; cat /tmp/claude-1003/-work/1b245680-4fe9-485a-b2b1-64fef49ff78f/tasks/bpvyq1q95.output
```

<details><summary>output</summary>

```
23:import argparse
24:import json
25:import os
26:import sys
27:import time
29:import cv2
30:import h5py
31:import numpy as np
32:import torch
33:import trimesh
34:import nvdiffrast.torch as dr
35:from scipy.ndimage import gaussian_filter1d, uniform_filter1d
36:from scipy.spatial.transform import Rotation as Rot
phases: static-floor [0,125]  carried [126,334]  static-table [335,439]
static floor: IoU 0.9298, twist -1.3 (baseline 0.0)
static table: IoU 0.8629, twist mod 90 = 22.0 (all inits: [22.5, 22.8, 22.0, 21.5, 23.4, 22.2, 23.0, 21.3, 26.5])
table-phase twist branch posterior: -158.0: 0.503, -68.0: 0.476, 22.0: 0.000, 112.0: 0.021
(baseline table twist -71.3)
table branch: HMM-best -158.0 (p=0.503), baseline-branch -68.0 (p=0.476) -> using -68.0
decoded table twist 292.5
depth re-ancho …[+128 chars]
```
</details>

**Bash**
```
cd /work && (time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz) > /work/scratch/final_log.txt 2>&1; tail -15 /work/scratch/final_log.txt
```

<details><summary>output</summary>

```
phases: static-floor [0,125]  carried [126,334]  static-table [335,439]
static floor: IoU 0.9294, twist -1.3 (baseline 0.0)
static table: IoU 0.8625, twist mod 90 = 22.5 (all inits: [22.6, 22.5, 21.3, 23.2, 22.8, 22.5, 21.7, 22.3, 47.0])
table-phase twist branch posterior: -157.5: 0.486, -67.5: 0.492, 22.5: 0.000, 112.5: 0.022
(baseline table twist -71.3)
table branch: HMM-best -67.5 (p=0.492), baseline-branch -67.5 (p=0.492) -> using -67.5
decoded table twist 292.5
depth re-anchoring scale factors: floor: 1.0005, carried: 1.0002, table: 0.9984
done in 200.8s -> /work/outputs/object_poses_world.npz

real	3m23.671s
user	3m24.949s
sys	0m4.561s
```
</details>

> *thinking:* The final solve.py run (3m24s) gave a table-branch posterior of 0.49 vs 0.49, so the baseline branch stays selected—the 0.7 threshold rule consistently confirms this despite minor GPU noise. Now evaluating the final output.

**Bash**
```
cd /work/scratch; python -c "
import numpy as np; d=np.load('/work/outputs/object_poses_world.npz'); T=d['T_world_mesh']; print(list(d.keys()),T.shape,T.dtype,np.isfinite(T).all()); 
R=T[:,:3,:3]; print('orthonormality err',np.abs(R@R.transpose(0,2,1)-np.eye(3)).max(), 'det',np.linalg.det(R).min(),np.linalg.det(R).max())
r3=np.load('run3.npz')['T_world_mesh']; print('max diff vs run3 (m / rot)',np.abs(T[:,:3,3]-r3[:,:3,3]).max())
"; python evaluate.py /work/outputs/object_poses_world.npz 125 335 eval_final.npz v > eval_final.txt; head -6 eval_final.txt
```

<details><summary>output</summary>

```
['T_world_mesh'] (440, 4, 4) float64 True
orthonormality err 4.274684109351057e-06 det 0.9999983374600551 0.9999993195742777
max diff vs run3 (m / rot) 0.006364620156774414
| phase | frames | IoU base | IoU ours | n vis | med rot step base/ours (deg) | p95 rot step base/ours | med trans step base/ours (cm) | dz mean / median (cm) | mean ADD(base,ours) cm |
|---|---|---|---|---|---|---|---|---|---|
| floor | 0-125 | 0.852 | 0.940 | 126 | 0.00 / 0.00 | 0.8 / 0.0 | 0.01 / 0.00 | -0.21 / 0.00 | 2.4 |
| carried | 126-334 | 0.786 | 0.870 | 159 | 4.65 / 1.80 | 27.0 / 4.5 | 2.62 / 1.95 | 0.93 / 1.31 | 10.1 |
| table | 335-439 | 0.839 | 0.880 | 105 | 0.00 / 0.00 | 0.0 / 0.0 | 0.00 / 0.00 | 0.14 / -0.00 | 1.5 |
| all | 0-439 | 0.821 | 0.895 | 390 | 0.01 / 0.00 | 25.1 / 3.5 | 0.44 / 0.00 | 0.41 / 0.4 …[+9 chars]
```
</details>

**Bash**
```
cd /work/scratch; python - <<'EOF'
import sys; sys.path.insert(0,'/work/scratch')
import numpy as np
from hoilib import *
ep=Episode('/work/inputs')
e=np.load('eval_final.npz'); res=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
Tc=ep.T_cam_world[None]@res
segs=[('floor (static)',0,126),('pick-up/carry, visible',126,197),('carry, occluded',197,248),('carry, visible + placement',248,335),('table (static)',335,440)]
print('| segment | frames | IoU base | IoU ours | twist base (start→end) | twist ours (start→end) | mean ADD(base,ours) cm | dz mean / median cm |')
print('|---|---|---|---|---|---|---|---|')
for n,a,b in segs:
    s=slice(a,b); dz=(Tc[s,2,3]-ep.Tc_base[s,2,3])*100
    ib=e['ib'][s]; inw=e['inw'][s]
    f=lambda x: '%.3f'%np.nanmean(x) if np.isfinite(x).any() else 'n/a'
    print(f"| {n} | {a}-{b-1} | {f(ib)} | {f(inw)} | {e['twb'][a]:.0f} → {e['twb'][b-1]:.0f} | {e['twn'][a]:.0f} → {e['twn'][b-1]:.0f} | {e['addd'][s].mean():.1f} | {dz.mean():.2f} / {np.median(dz):.2f} |")
# floor plane check
n=ep.plane[:3]/np.linalg.norm(ep.plane[:3]); d=ep.plane[3]/np.linalg.norm(ep.plane[:3])
V=np.asarray(ep.mesh.vertices)
for name,T in [('ours',res[0]),('base',ep.T_base[0]),('base60',ep.T_base[60])]:
    sd=(T[:3,:3]@V.T).T@n+T[:3,3]@n+d; print(name,'min plane dist %.3f'%sd.min(),'axis tilt vs normal %.1f'%np.degrees(np.arccos(abs(T[:3,2]@n))))
print('table axis tilt vs floor normal ours %.1f base %.1f'%(np.degrees(np.arccos(abs(res[400,:3,2]@n))),np.degrees(np. …[+41 chars]
```

<details><summary>output</summary>

```
| segment | frames | IoU base | IoU ours | twist base (start→end) | twist ours (start→end) | mean ADD(base,ours) cm | dz mean / median cm |
|---|---|---|---|---|---|---|---|
| floor (static) | 0-125 | 0.852 | 0.940 | 0 → 87 | -1 → -1 | 2.4 | -0.21 / 0.00 |
| pick-up/carry, visible | 126-196 | 0.797 | 0.901 | 87 → -159 | 0 → 79 | 10.1 | 1.07 / 1.33 |
| carry, occluded | 197-247 | 0.446 | 0.733 | -161 → -12 | 82 → 155 | 12.6 | -0.12 / 0.11 |
| carry, visible + placement | 248-334 | 0.788 | 0.849 | -15 → -63 | 159 → -73 | 8.7 | 1.43 / 1.36 |
| table (static) | 335-439 | 0.839 | 0.880 | -66 → -71 | -67 → -67 | 1.5 | 0.14 / -0.00 |
ours min plane dist -0.014 axis tilt vs normal 3.6
base min plane dist -0.009 axis tilt vs normal 3.8
base60 min plane dist -0.030 axis tilt vs normal 3.8
table axis …[+39 chars]
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
mkdir -p /work/scratch/.claude/projects/-work/memory; cat > /work/scratch/.claude/projects/-work/memory/cone-ep17-trajectory.md <<'EOF'
---
name: cone-ep17-trajectory
description: Episode 17 traffic-cone pose task (2026-10-21) - outputs done; key findings on 4-fold twist ambiguity and branch decision
metadata:
  type: project
---
Task in /work/AGENT_TASK.md (traffic cone, 440 frames) was completed 2026-10-21: solve.py, REPORT.md, TOOLS.md and object_poses_world.npz are in /work/outputs.
Key facts: the cone is ~4-fold symmetric about its axis, so the silhouette gives twist only mod 90. The baseline has a spurious +90 twist jump at frame 118. The table-phase 90-deg branch is a ~50/50 posterior, so the baseline branch was kept by a rule (switch only if p>0.7). The mesh texture's pale blotch is a scan artifact.
**Why:** if the task is resumed, avoid redoing the analysis. **How to apply:** read REPORT.md first; analysis scripts are in /work/scratch.
EOF
echo "- [Cone ep17 trajectory task](cone-ep17-trajectory.md) — finished outputs + twist-branch findings" >> /work/scratch/.claude/projects/-work/memory/MEMORY.md; ls -la /work/outputs
```

<details><summary>output</summary>

```
total 112
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 21 16:38 .
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 21 15:52 ..
-rw-rw-r-- 1 hanzhou hanzhou 12465 Oct 21 16:38 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou  1514 Oct 21 16:32 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 56598 Oct 21 16:37 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou 26874 Oct 21 16:30 solve.py
```
</details>

**Agent:** I've produced a new trajectory for all 440 frames. It fixes the baseline where the evidence is clear, but the cone's yaw on the table is still a coin flip. The final `solve.py` ran once offline (3 min 24 s, about 25 min of GPU time in total) and its output checks out: shape 440×4×4, float64, finite, valid rotations.

The cone is almost exactly 4-fold symmetric about its axis, so the silhouette only pins its yaw to within 90°. With ADD as the metric, a wrong 90° branch costs about 9 cm. That ambiguity is the main thing I worked on.

**What was wrong in the baseline:**
- **Spurious yaw jump:** FoundationPose jumps +90° in yaw at frame 104 while the cone sits untouched on the floor. The baseline inherits this from frame 118, so the start of the carry is on a different 90° branch from its own floor pose.
- **Noisy carry:** the rotations jitter (95th-percentile step 27° per frame), and they are pure noise while the cone is hidden behind the person (frames 197–247).
- **Drifting static pose:** on the floor the baseline's depth drifts by 3 cm even though the cone doesn't move.

**What I did:**
- **Phases:** found automatically: floor 0–125, carried 126–334, table 335–439.
- **Static phases:** one pose each, fitted to the object mask over the whole phase. On the floor this confirms the baseline's yaw to within 1.3° and removes the drift and the jump.
- **Carried phase:**
  - **Yaw decoding:** each visible frame gets a yaw score from the mask. A model over absolute yaw then links the floor pose to the table pose, assuming the cone roughly turns with the person's torso.
  - **Hidden frames:** the cone follows the person's hands.
- **Depth:** set to the baseline's median per phase. The floor and table depth cues disagree with each other by more than the 2 cm rule allows, so the extra depth shift is 0.

| Phase | Mask IoU, baseline → mine | Confidence |
|---|---|---|
| Floor | 0.852 → 0.940 | High |
| Pick-up and carry, visible (126–196) | 0.797 → 0.901 | Medium-high |
| Hidden behind person (197–247) | no real image evidence | Low |
| Carry back and placement (248–334) | 0.788 → 0.849 | Medium-low |
| Table | 0.839 → 0.880 | ≈50% on the yaw branch |

Over the whole episode IoU goes from 0.821 to 0.895, and the 95th-percentile rotation step drops from 25° to 3.5°.

**The open risk is the table yaw.** The model gives the baseline's branch 0.49 and the branch 90° away 0.49. Across 54 parameter settings, the other branch is often on top, with probability up to 0.79. Because nothing is clearly favoured, I kept the baseline's branch by a written rule (switch only above 0.7). If that branch is wrong, the table phase and the frames leading into it lose about 9 cm of ADD either way.

I didn't use the pale smear in the mesh texture to pick a yaw branch. It never appears in the video and is almost certainly a scan artifact, which the report explains.

Files are in /work/outputs:
- object_poses_world.npz
- solve.py
- REPORT.md
- TOOLS.md

## Final

I've produced a new trajectory for all 440 frames. It fixes the baseline where the evidence is clear, but the cone's yaw on the table is still a coin flip. The final `solve.py` ran once offline (3 min 24 s, about 25 min of GPU time in total) and its output checks out: shape 440×4×4, float64, finite, valid rotations.

The cone is almost exactly 4-fold symmetric about its axis, so the silhouette only pins its yaw to within 90°. With ADD as the metric, a wrong 90° branch costs about 9 cm. That ambiguity is the main thing I worked on.

**What was wrong in the baseline:**
- **Spurious yaw jump:** FoundationPose jumps +90° in yaw at frame 104 while the cone sits untouched on the floor. The baseline inherits this from frame 118, so the start of the carry is on a different 90° branch from its own floor pose.
- **Noisy carry:** the rotations jitter (95th-percentile step 27° per frame), and they are pure noise while the cone is hidden behind the person (frames 197–247).
- **Drifting static pose:** on the floor the baseline's depth drifts by 3 cm even though the cone doesn't move.

**What I did:**
- **Phases:** found automatically: floor 0–125, carried 126–334, table 335–439.
- **Static phases:** one pose each, fitted to the object mask over the whole phase. On the floor this confirms the baseline's yaw to within 1.3° and removes the drift and the jump.
- **Carried phase:**
  - **Yaw decoding:** each visible frame gets a yaw score from the mask. A model over absolute yaw then links the floor pose to the table pose, assuming the cone roughly turns with the person's torso.
  - **Hidden frames:** the cone follows the person's hands.
- **Depth:** set to the baseline's median per phase. The floor and table depth cues disagree with each other by more than the 2 cm rule allows, so the extra depth shift is 0.

| Phase | Mask IoU, baseline → mine | Confidence |
|---|---|---|
| Floor | 0.852 → 0.940 | High |
| Pick-up and carry, visible (126–196) | 0.797 → 0.901 | Medium-high |
| Hidden behind person (197–247) | no real image evidence | Low |
| Carry back and placement (248–334) | 0.788 → 0.849 | Medium-low |
| Table | 0.839 → 0.880 | ≈50% on the yaw branch |

Over the whole episode IoU goes from 0.821 to 0.895, and the 95th-percentile rotation step drops from 25° to 3.5°.

**The open risk is the table yaw.** The model gives the baseline's branch 0.49 and the branch 90° away 0.49. Across 54 parameter settings, the other branch is often on top, with probability up to 0.79. Because nothing is clearly favoured, I kept the baseline's branch by a written rule (switch only above 0.7). If that branch is wrong, the table phase and the frames leading into it lose about 9 cm of ADD either way.

I didn't use the pale smear in the mesh texture to pick a yaw branch. It never appears in the video and is almost certainly a scan artifact, which the report explains.

Files are in /work/outputs:
- object_poses_world.npz
- solve.py
- REPORT.md
- TOOLS.md
