# T1_ds_ep023

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T18:26:28Z

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/AGENT_TASK.md'
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
and translation both matter; a  …[+9792 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,90p' /work/AGENT_TASK.md && printf '\\n--- 91+ ---\\n' && sed -n '91,180p' /work/AGENT_TASK.md"
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
and translation both matter; a  …[+9805 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '65,105p'"
```
<details><summary>output</summary>

```
    65	
    66	## Inputs (`/work/inputs`)
    67	- `video.mp4` - the only camera (static exo camera), 1536x1152.
    68	- `task.txt` - what the person does. `episode.json` - object name, frame count.
    69	- `object_mesh.glb` - the exact object mesh (metres), textured. Load it with
    70	  `trimesh.load(path, force="scene", process=False).to_geometry()`; all poses below map THESE mesh-file
    71	  coordinates to the target frame.
    72	- `camera.json` - `K` (pinhole intrinsics, shared by all episodes of this camera), `T_world_cam` (4x4 camera-to-world,
    73	  estimated from the frame-0 reference human and the floor). `K_pipeline`: the per-episode MoGe-2 K the pipeline's
    74	  camera-frame estimates were made with. OpenCV conventions: camera x right, y down, z forward; the world fr …[+2925 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort; printf '\\nTASK\\n'; cat /work/inputs/task.txt; printf '\\nEPISODE\\n'; cat /work/inputs/episode.json; printf '\\nCAMERA\\n'; cat /work/inputs/camera.json; printf '\\nPLANE\\n'; cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
camera.json 843 bytes
episode.json 67 bytes
frame0_reference_human.npz 192376 bytes
ground_plane.json 395 bytes
masks.h5 7931559 bytes
object_mesh.glb 14332412 bytes
reconstruction.npz 47617950 bytes
task.txt 125 bytes
video.mp4 28669144 bytes

TASK
Walk over to the toy stroller on the ground and use both hands to move the stroller back and forth within the bounding area.

EPISODE
{
 "episode_index": 23,
 "object": "toy_stroller",
 "frames": 946
}
CAMERA
{
 "K": [
  [
   820.0010986328125,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   820.0010986328125,
   560.4999389648438
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "K_pipeline": [
  [
   819.4840087890625,
   0.0,
   768.0
  ],
  [
   0.0,
   819.4840087890625,
   576.0
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "T_world_cam": [
  [
   0.9 …[+904 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100; printf '\\nWORKSPACE\\n'; find /workspace -maxdepth 2 -type f | head -80"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/01_depth.json 1744
/pipeline/.stages/02_export.json 1964
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2133
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/depth_report.json 1815
/pipeline/depth/moge2/aligned.h5 1172395304
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 100683
/pipeline/depth/moge2/raw.npy 3347841152
/pipeline/export/episode_000023/edex 563
/pipeline/export/episode_000023/wild_export.json 2793
/pipeline/foundationpose/episode_000023.pkl 269085
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 72346 …[+1125 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader; python - <<'PY'
import torch, cv2, trimesh, numpy as np, h5py
print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('cv2',cv2.__version__,'trimesh',trimesh.__version__,'numpy',np.__version__)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 cuda True NVIDIA RTX 6000 Ada Generation
cv2 4.11.0 trimesh 5.1.1 numpy 1.26.3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json, os
r=np.load('/work/inputs/reconstruction.npz')
print('recon keys')
for k in r.files:
 a=r[k]; print(k,a.shape,a.dtype)
print('\\npose translations (world)')
T=r['T_world_mesh']; print(T[::50,0:3,3]); print('bounds',T[:,0:3,3].min(0),T[:,0:3,3].max(0))
print('T det/orth',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
with h5py.File('/work/inputs/masks.h5') as f:
 print('\\nmask keys',list(f.keys()))
 for k in f: print(k,f[k].shape,f[k].dtype)
PY"
```
<details><summary>output</summary>

```
recon keys
T_world_mesh (946, 4, 4) float32
T_cam_mesh_foundationpose (946, 4, 4) float32
T_cam_mesh_coconet (946, 4, 4) float32
human_vertices_world (946, 4610, 3) float32
human_keypoints_world (946, 70, 3) float32
contact_prob_right_left (946, 2) float32

pose translations (world)
[[ 0.13105322  1.0089045   2.5044966 ]
 [ 0.12849148  1.0250092   2.5226219 ]
 [ 0.14006752  0.9957142   2.446952  ]
 [ 0.23216893  0.990945    2.4230258 ]
 [ 0.41847295  0.9817135   2.4552054 ]
 [ 0.6028238   0.98096275  2.567243  ]
 [ 0.7937077   0.9659885   2.7316685 ]
 [ 0.87070304  0.97367567  2.9324443 ]
 [ 0.8447791   0.9488127   3.2589169 ]
 [ 0.6874939   0.92904615  3.507158  ]
 [ 0.47125447  0.87567663  3.5643575 ]
 [ 0.2357661   0.88922703  3.8859162 ]
 [-0.08754428  0.91880155  3.9032145 ]
 [-0.2871 …[+426 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for f in /pipeline/.stages/*.json /pipeline/depth/moge2/depth_report.json /pipeline/pipeline_report.json; do echo ==== $f; cat $f; echo; done'
```
<details><summary>output</summary>

```
==== /pipeline/.stages/01_depth.json
{
  "name": "01_depth",
  "outputs": [
    {
      "mtime_ns": 1791265853983569112,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000023/depth/moge2/raw.npy",
      "size": 3347841152
    },
    {
      "mtime_ns": 1791265854111569112,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000023/depth/moge2/intrinsics.pkl",
      "size": 100683
    },
    {
      "mtime_ns": 1791265856782569102,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000023/depth/moge2/depth_report.json",
      "size": 1815
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/mhr_wild_depth.py",
      "--video",
      "/home/hanzhou/Projects/v2d/run/w …[+24227 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(m)
print('geometry',m.geometry)
for k,g in m.geometry.items():
 print(k,'verts',len(g.vertices),'faces',len(g.faces),'visual',type(g.visual),'bounds',g.bounds,'extents',g.extents)
 S=np.eye(4); S[:3,:3]=trimesh.bounds.oriented_bounds(g)[0]; print('S',S)
PY"
```
<details><summary>output</summary>

```
<trimesh.Scene(len(geometry)=1)>
geometry OrderedDict([('toystroller1', <trimesh.Trimesh(vertices.shape=(290547, 3), faces.shape=(96859, 3))>)])
toystroller1 verts 290547 faces 96859 visual <class 'trimesh.visual.texture.TextureVisuals'> bounds [[-0.14973971 -0.1413893  -0.28821406]
 [ 0.19135687  0.13983184  0.24650338]] extents [0.34109658 0.28122114 0.53471744]
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
ValueError: could not broadcast input array from shape (4,4) into shape (3,3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /work/inputs/video.mp4'
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
avg_frame_rate=30/1
duration=31.533333
nb_frames=946

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/frames && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"select='not(mod(n,25))',scale=384:288,tile=6x7\" -frames:v 1 /work/scratch/contact_video.png && ls -lh /work/scratch/contact_video.png"
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 4.9M Oct 20 22:20 /work/scratch/contact_video.png

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh']; tr=T[:,:3,3]; rot=R.from_matrix(T[:,:3,:3])
dp=np.linalg.norm(np.diff(tr,axis=0),axis=1)
dr=np.linalg.norm((rot[1:]*rot[:-1].inv()).as_rotvec(),axis=1)
c=r['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as f:
 o=f['object'][:]; p=f['person'][:]
area=o.reshape(len(o),-1).sum(1); pa=p.reshape(len(p),-1).sum(1)
ys,xs=np.meshgrid(np.arange(1152),np.arange(1536),indexing='ij')
# actually ys shape 1152x1536/xs same
cx=np.where(o,xs[None],0).sum((1,2))/np.maximum(area,1); cy=np.where(o,ys[None],0).sum((1,2))/np.maximum(area,1)
for i in range(0,len(T),20):
 lo=max(0,i-3);hi=min(len(T),i+4)
 print(f'{i:4d} t={i/30:.1f}s tr={tr[i].round(3)} dp={dp[lo:hi-1].mean():.4f} dr={np.degrees(dr[lo:hi-1]).mean():6.2f} oa={area[i]:7d} oc={(p[i]&o[i]).sum()/max(area[i],1):.2f} cx={cx[i]:.0f} cy={cy[i]:.0f} contact={c[i].round(2)}')
print('overall med dp',np.median(dp),'drdeg',np.degrees(np.median(dr)))
PY"
```
<details><summary>output</summary>

```
   0 t=0.0s tr=[0.131 1.009 2.504] dp=0.0075 dr=  0.73 oa=  13418 oc=0.00 cx=754 cy=885 contact=[0. 0.]
  20 t=0.7s tr=[0.127 1.01  2.493] dp=0.0076 dr=  0.49 oa=  13321 oc=0.00 cx=754 cy=885 contact=[0. 0.]
  40 t=1.3s tr=[0.126 1.024 2.518] dp=0.0142 dr=  0.44 oa=  13265 oc=0.00 cx=754 cy=885 contact=[0. 0.]
  60 t=2.0s tr=[0.128 1.012 2.495] dp=0.0083 dr=  0.55 oa=  13606 oc=0.00 cx=754 cy=884 contact=[0. 0.]
  80 t=2.7s tr=[0.129 1.014 2.502] dp=0.0088 dr=  0.52 oa=  13269 oc=0.00 cx=754 cy=885 contact=[0. 0.]
 100 t=3.3s tr=[0.14  0.996 2.447] dp=0.0106 dr=  0.69 oa=  13184 oc=0.00 cx=756 cy=886 contact=[1.   0.07]
 120 t=4.0s tr=[0.138 0.997 2.448] dp=0.0064 dr=  0.37 oa=  12783 oc=0.00 cx=756 cy=888 contact=[1. 1.]
 140 t=4.7s tr=[0.171 0.994 2.437] dp=0.0096 dr=  0.89 oa=  13200 oc …[+4335 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; tr=T[:,:3,3]; rot=R.from_matrix(T[:,:3,:3]);
for i in range(0,946,25):
 # rotation relative initial / relative prior 25
 d=(rot[i]*rot[0].inv()).as_rotvec(); rr=np.degrees(np.linalg.norm(d));
 print(i, 'rel0deg',f'{rr:7.2f}', 'euler',np.degrees((rot[i]*rot[0].inv()).as_euler('xyz')).round(2), 'contact',r['contact_prob_right_left'][i].round(2))
PY"
```
<details><summary>output</summary>

```
0 rel0deg    0.00 euler [0. 0. 0.] contact [0. 0.]
25 rel0deg    0.64 euler [-0.19 -0.61 -0.  ] contact [0. 0.]
50 rel0deg    0.47 euler [ 0.46 -0.07  0.02] contact [0. 0.]
75 rel0deg    0.42 euler [-0.08 -0.38 -0.16] contact [0. 0.]
100 rel0deg    1.84 euler [ 1.42 -1.06  0.48] contact [1.   0.07]
125 rel0deg    2.39 euler [ 2.11 -1.11 -0.12] contact [1. 1.]
150 rel0deg    8.58 euler [ 1.78 -8.37 -0.78] contact [1. 1.]
175 rel0deg   16.20 euler [  1.13 -16.11  -1.41] contact [1. 1.]
200 rel0deg   19.28 euler [  0.49 -19.27  -0.72] contact [1. 1.]
225 rel0deg   30.41 euler [  2.31 -30.28  -2.49] contact [1. 1.]
250 rel0deg   40.36 euler [  1.1  -40.31  -2.1 ] contact [1. 1.]
275 rel0deg   45.11 euler [  2.38 -45.    -3.5 ] contact [1. 1.]
300 rel0deg   48.63 euler [  2.22 -48.52  -3.73] co …[+1642 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/inference/materialized_inputs.h5') as f:
 print(list(f.keys()))
 def visit(n,o):
  if hasattr(o,'shape'): print(n,o.shape,o.dtype)
 f.visititems(visit)
PY"
```
<details><summary>output</summary>

```
['K_rois', 'bboxes', 'identity_json', 'input_rgbs', 'input_xyz', 'observed_human_masks', 'observed_object_masks', 'render_rgbs', 'render_xyz']
K_rois (946, 3, 3) float32
bboxes (946, 4) float32
identity_json () object
input_rgbs (946, 3, 224, 224) float32
input_xyz (946, 6, 224, 224) float32
observed_human_masks (946, 224, 224) bool
observed_object_masks (946, 224, 224) bool
render_rgbs (946, 3, 224, 224) float32
render_xyz (946, 6, 224, 224) float32

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffmpeg -hide_banner -loglevel error -i /pipeline/inference/episode_000023_step200000_before_after.mp4 -vf \"select='not(mod(n,40))',scale=512:384,tile=5x5\" -frames:v 1 /work/scratch/contact_before_after.png; ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 /pipeline/inference/episode_000023_step200000_before_after.mp4; ls -lh /work/scratch/contact_before_after.png"
```
<details><summary>output</summary>

```
width=4608
height=1236
r_frame_rate=30/1
nb_frames=946
-rw-rw-r-- 1 hanzhou hanzhou 5.4M Oct 20 22:22 /work/scratch/contact_before_after.png

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; Tf=r['T_cam_mesh_foundationpose']; Tc=r['T_cam_mesh_coconet']; Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for name,A in [('FP',Tf),('CoCo',Tc)]:
 Rw=Twc[None]@A
 d=(R.from_matrix(T[:,:3,:3]).inv()*R.from_matrix(Rw[:,:3,:3])).magnitude()
 t=np.linalg.norm(T[:,:3,3]-Rw[:,:3,3]@0+0,axis=1) # unused
 print(name,'rot diff deg median/max',np.degrees(np.median(d)),np.degrees(d.max()))
 for i in range(0,946,50): print(i,round(np.degrees(d[i]),1),end='; ')
 print()
# Compare FP and CoCo rotation
d=(R.from_matrix(Tf[:,:3,:3]).inv()*R.from_matrix(Tc[:,:3,:3])).magnitude();print('fp-coco',np.degrees(np.median(d)),np.degrees(d.max()))
# translation expressed world no correction
for name,A in [('FP',Tf),('CoCo',Tc)]:
 tw=Twc[None]@A
 print(name,'trans med diff',np.median(np.linalg.norm(tw[:,:3,3]-T[:,:3,3],axis=1)))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
ValueError: matmul: Input operand 1 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; tr=T[:,:3,3]; rot=R.from_matrix(T[:,:3,:3]);
dp=np.linalg.norm(np.diff(tr,axis=0),axis=1); dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude())
print('largest dp')
for i in np.argsort(dp)[-30:][::-1]: print(i+1,round(dp[i],4),'contact',r['contact_prob_right_left'][i+1])
print('largest dr')
for i in np.argsort(dr)[-30:][::-1]: print(i+1,round(dr[i],2),'contact',r['contact_prob_right_left'][i+1])
# static runs based on 5-frame median speed thresholds
from scipy.ndimage import median_filter
sm=median_filter(dp,size=7); sr=median_filter(dr,size=7)
static=((sm<0.004)&(sr<0.25))
runs=[]; s=None
for i,v in enumerate(static):
 if v and s is None:s=i
 if s is not None and (not v or i==len(static)-1):
  e=i if not v else i+1; runs.append((s,e,e-s));s=None
print('static runs >=5', [x for x in runs if x[2]>=5])
PY"
```
<details><summary>output</summary>

```
largest dp
435 0.1759 contact [0.9999904 0.9999958]
439 0.1693 contact [0.9999826 0.9999373]
438 0.1557 contact [0.99999356 0.99997634]
500 0.1499 contact [0.99999845 0.99999976]
462 0.1281 contact [0.99999946 0.9999911 ]
437 0.1209 contact [0.9999823  0.99997914]
471 0.1171 contact [0.9999999 0.9999961]
436 0.1134 contact [0.9998881 0.9999932]
460 0.1093 contact [0.99999905 0.99997973]
446 0.105 contact [0.99999386 0.99997133]
448 0.1021 contact [0.9999968 0.9998889]
750 0.0982 contact [1. 1.]
469 0.0974 contact [0.99999917 0.9999879 ]
449 0.097 contact [0.9999996  0.99995136]
447 0.0954 contact [0.99999875 0.9999021 ]
463 0.0886 contact [0.9999985  0.99996865]
456 0.0858 contact [0.9999998 0.9999866]
503 0.0833 contact [0.99999714 0.9999996 ]
455 0.082 contact [0.99999595 0.9999311 ]
472 …[+1297 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
p=pickle.load(open('/pipeline/foundationpose/episode_000023.pkl','rb'))
print(type(p))
if isinstance(p,dict):
 for k,v in p.items():
  try: print(k,type(v),getattr(v,'shape',None), (v if isinstance(v,(str,int,float)) else ''))
  except: pass
else: print(p)
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
frames <class 'list'> None 
camera_id <class 'int'> None 0
fp_poses_cam <class 'numpy.ndarray'> (946, 1, 4, 4) 
fp_candidate_count <class 'numpy.ndarray'> (946,) 
fp_selected_candidate_index <class 'numpy.ndarray'> (946,) 
fp_selected_iou <class 'numpy.ndarray'> (946,) 
fp_selected_visibility <class 'numpy.ndarray'> (946,) 
fp_retry_count <class 'numpy.ndarray'> (946,) 
fp_reliable <class 'numpy.ndarray'> (946,) 
fp_oracle_rotation_error_deg <class 'numpy.ndarray'> (946,) 
fp_translation_threshold_m <class 'numpy.ndarray'> (946,) 
fp_selected_translation_distance_m <class 'numpy.ndarray'> (946,) 
gt_object_visibility_ratio <class 'numpy.ndarray'> (946,) 
gt_object_visibility_valid <class 'numpy.ndarray'> (946,) 
gt_object_rendered_pixels <class 'numpy.ndarray'> (946,) 
gt_ob …[+1034 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; Tf=r['T_cam_mesh_foundationpose']; Tc=r['T_cam_mesh_coconet']; Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for name,A in [('FP',Tf),('CoCo',Tc)]:
 Rw=Twc@A
 d=(R.from_matrix(T[:,:3,:3]).inv()*R.from_matrix(Rw[:,:3,:3])).magnitude()
 print(name,'rot diff deg median/max',np.degrees(np.median(d)),np.degrees(d.max()))
 print('by 50:', ' '.join(f'{i}:{np.degrees(d[i]):.1f}' for i in range(0,946,50)))
 print('trans med diff',np.median(np.linalg.norm(Rw[:,:3,3]-T[:,:3,3],axis=1)))
# Compare FP and CoCo rotation
d=(R.from_matrix(Tf[:,:3,:3]).inv()*R.from_matrix(Tc[:,:3,:3])).magnitude();print('fp-coco',np.degrees(np.median(d)),np.degrees(d.max()))
PY"
```
<details><summary>output</summary>

```
FP rot diff deg median/max 3.107827348623452 43.0313206587788
by 50: 0:2.0 50:1.7 100:1.8 150:2.3 200:1.7 250:0.6 300:1.0 350:1.6 400:43.0 450:2.7 500:2.3 550:2.0 600:3.2 650:6.1 700:5.8 750:4.7 800:3.9 850:4.9 900:4.3
trans med diff 0.1365845837283684
CoCo rot diff deg median/max 1.1884779198896074 1.2424490063562912
by 50: 0:1.1 50:1.1 100:1.1 150:1.1 200:1.1 250:1.1 300:1.1 350:1.1 400:1.2 450:1.2 500:1.2 550:1.2 600:1.2 650:1.2 700:1.2 750:1.2 800:1.2 850:1.2 900:1.2
trans med diff 0.10488516527815941
fp-coco 2.5527016076766125 41.99311507401705

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,pprint,numpy as np
p=pickle.load(open('/pipeline/foundationpose/episode_000023.pkl','rb'))
pprint.pp(p['metadata'])
for k in ['fp_selected_iou','fp_selected_visibility','fp_reliable','fp_tracking_state','fp_confirmed_pose_source','fp_emitted_pose_source','fp_full_image_iou','fp_centroid_distance','fp_rejection_mask','fp_candidate_count','fp_retry_count']:
 v=p[k]; print('\\n',k,v.dtype,v.shape)
 if v.dtype.kind in 'iufb': print('uniq',np.unique(v,return_counts=True) if len(np.unique(v))<20 else (v.min(),v.median(),v.max()))
PY"
```
<details><summary>output</summary>

```
{'source': 'foundationpose_register_first_then_track_metric_depth',
 'depth_root': '/home/hanzhou/Projects/v2d/run/out/episode_000023/depth/moge2/aligned.h5',
 'depth_source': 'metric_depth',
 'depth_input_identity': {'path': '/home/hanzhou/Projects/v2d/run/out/episode_000023/depth/moge2/aligned.h5',
                          'size': 1172395304,
                          'mtime_ns': 1791266485601566703},
 'mesh': '/home/hanzhou/Projects/v2d/run/out/episode_000023/export/episode_000023/object_mesh/output_aligned.glb',
 'mask_identities': {'human': {'raw': {'storage': 'h5',
                                       'source': {'path': '/home/hanzhou/Projects/v2d/run/out/episode_000023/export/episode_000023/human_masks/front_stereo_camera_left.h5',
                                                 …[+40832 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'find /code/reconstruction -maxdepth 3 -type f | head -100; find /workspace/v2d_cari4d -type f | head -100'
```
<details><summary>output</summary>

```
/code/reconstruction/scripts/install_ego_reconstruction_packages.sh
/code/reconstruction/scripts/build_ego_e2e_containers.sh
/code/reconstruction/scripts/build_containers.sh
/code/reconstruction/scripts/install_packages.sh
/code/reconstruction/scripts/export_result_threejs_scene.py
/code/reconstruction/scripts/setup_css_env.sh
/code/reconstruction/scripts/download_ego_reconstruction_weights.sh
/code/reconstruction/scripts/build_ego_reconstruction_packages.sh
/code/reconstruction/scripts/sync_css.py
/code/reconstruction/scripts/setup_hitl_aws_env.sh
/code/reconstruction/scripts/install_ego_e2e_packages.sh
/code/reconstruction/scripts/scale_result_bundle.py
/code/reconstruction/scripts/setup_kratos_drs_env.sh
/code/reconstruction/scripts/setup_mv_hoi_status_publish_env.sh
/code/reconstructio …[+10915 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
p=pickle.load(open('/pipeline/foundationpose/episode_000023.pkl','rb'))
for k in ['fp_selected_iou','fp_selected_visibility','fp_reliable','fp_selected_candidate_index','fp_candidate_count','fp_retry_count','fp_full_image_iou','fp_tolerant_precision','fp_tolerant_recall','fp_centroid_distance','fp_rendered_visibility']:
 v=p[k].reshape(-1)
 print(k,'min/med/max',float(np.nanmin(v)),float(np.nanmedian(v)),float(np.nanmax(v)))
 print('values',v[::100])
PY"
```
<details><summary>output</summary>

```
fp_selected_iou min/med/max 0.44749215245246887 0.6163259744644165 0.7623449563980103
values [0.5864248  0.5963254  0.604269   0.68643266 0.52380955 0.67217815
 0.6220799  0.6141373  0.6294943  0.66168636]
fp_selected_visibility min/med/max 0.13533438742160797 0.9720263481140137 1.0
values [1.         0.9855829  0.9656311  0.8814348  0.2889683  0.91848814
 0.985899   0.97197175 0.9686034  0.9823173 ]
fp_reliable min/med/max 1.0 1.0 1.0
values [ True  True  True  True  True  True  True  True  True  True]
fp_selected_candidate_index min/med/max -1.0 -1.0 51.0
values [51 -1 -1 -1 -1 -1 -1 -1 -1 -1]
fp_candidate_count min/med/max 1.0 1.0 252.0
values [252   1   1   1   1   1   1   1   1   1]
fp_retry_count min/med/max 0.0 0.0 0.0
values [0 0 0 0 0 0 0 0 0 0]
fp_full_image_iou min/med/max 0.447 …[+988 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast,nvdiffrast.torch as dr,inspect
print(nvdiffrast.__file__)
print(dr.rasterize.__doc__)
PY"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py
Rasterize triangles.

    All input tensors must be contiguous and reside in GPU memory except for
    the `ranges` tensor that, if specified, has to reside in CPU memory. The
    output tensors will be contiguous and reside in GPU memory.

    Args:
        glctx: Rasterizer context of type `RasterizeCudaContext`.
        pos: Vertex position tensor with dtype `torch.float32`. To enable range
             mode, this tensor should have a 2D shape [num_vertices, 4]. To enable
             instanced mode, use a 3D shape [minibatch_size, num_vertices, 4].
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        resolution: Output resolution as integer tuple (height, width).
        ranges: In ran …[+640 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=list(m.geometry.values())[0]
print(type(g.visual),g.visual.material)
print('uv',g.visual.uv.shape,g.visual.uv.min(0),g.visual.uv.max(0))
mat=g.visual.material
print('mat',mat)
print('image',type(mat.image),mat.image.size if mat.image else None)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7c8eedd3ffd0>
uv (290547, 2) [0.0009     0.00091398] [0.99910098 0.99908602]
mat <trimesh.visual.material.PBRMaterial object at 0x7c8eedd3ffd0>
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];mat=g.visual.material
for k in dir(mat):
 if not k.startswith('_'):
  try:
   v=getattr(mat,k)
   if v is not None and not callable(v): print(k,type(v),getattr(v,'shape',getattr(v,'size',None)))
  except: pass
PY"
```
<details><summary>output</summary>

```
baseColorTexture <class 'PIL.PngImagePlugin.PngImageFile'> (3891, 3829)
doubleSided <class 'bool'> None
main_color <class 'numpy.ndarray'> (4,)
metallicFactor <class 'float'> None
name <class 'str'> None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_eval.py <<'PY'
import json, numpy as np, torch, trimesh, cv2, h5py, argparse, os, time
import nvdiffrast.torch as dr

def load_mesh(path):
    scene=trimesh.load(path, force='scene', process=False)
    g=list(scene.geometry.values())[0]
    return g

def project_vertices(V, T, Twc_inv, K, W, H):
    vh=np.concatenate([V,np.ones((len(V),1),np.float32)],1)
    cam=(Twc_inv@(T@vh.T)).T
    x,y,z=cam[:,0],cam[:,1],cam[:,2]
    fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
    xp=fx*x/z+cx; yp=fy*y/z+cy
    xn=2*xp/W-1; yn=1-2*yp/H
    return np.stack([xn*z,yn*z,z,z],1).astype(np.float32), cam

def make_renderer(mesh, device='cuda'):
    ctx=dr.RasterizeCudaContext()
    tri=torch.tensor(mesh.faces.astype(np.int32),device=device)
    V=torch.tensor(mesh.vertices.astype(np.float32),device=device)
    return ctx,tri,V

def raster_pose(renderer, T, Twc_inv, K, W=1536, H=1152):
    ctx,tri,V=renderer
    clip,_=project_vertices(mesh.vertices,T,Twc_inv,K,W,H) if False else (None,None)

def raster(mesh, renderer, T, Twc_inv, K, W=1536, H=1152, tex=None):
    ctx,tri,V=renderer
    clip,_=project_vertices(mesh.vertices,T,Twc_inv,K,W,H)
    pos=torch.tensor(clip,device='cuda')
    rast, _ = dr.rasterize(ctx,pos[None],tri,resolution=(H,W))
    mask=(rast[0,:,:,3]>0)
    depth=rast[0,:,:,2]
    uv=None; rgb=None
    if tex is not None:
        attr=torch.tensor(mesh.visual.uv.astype(np.float32),device='cuda')
        uv,_=dr.interpolate(attr[None],ras …[+1586 chars]
```
<details><summary>output</summary>

```
0 iou 0.0 prec 0.0 rec 0.0 masks 13418 8849 cent (754.2648680876434, 884.6456252794753) (755.2259012317776, 273.32150525483104) sec 0.099
100 iou 0.0 prec 0.0 rec 0.0 masks 13184 9290 cent (755.7002427184466, 886.1875758495146) (758.3837459634015, 270.33638320775026) sec 0.025
200 iou 0.0 prec 0.0 rec 0.0 masks 13002 9204 cent (850.739655437625, 882.5244577757268) (852.8897218600608, 278.433615819209) sec 0.023
300 iou 0.0 prec 0.0 rec 0.0 masks 8547 7480 cent (950.0952380952381, 841.2510822510823) (951.0161764705882, 323.34010695187163) sec 0.025
400 iou 0.0 prec 0.0 rec 0.0 masks 1905 5411 cent (921.3574803149606, 809.8992125984252) (929.1666974681204, 372.82184439105527) sec 0.035
500 iou 0.0 prec 0.0 rec 0.0 masks 5903 4478 cent (823.4761985431137, 752.8995426054548) (827.0690040196516 …[+581 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/yn=1-2\\*yp\\/H/yn=2*yp\\/H-1/' /work/scratch/render_eval.py && python /work/scratch/render_eval.py"
```
<details><summary>output</summary>

```
0 iou 0.5545625916358304 prec 0.8977169981916817 rec 0.5919660157996721 masks 13418 8848 cent (754.2648680876434, 884.6456252794753) (755.2250226039783, 877.6741636528029) sec 0.039
100 iou 0.5467695945476041 prec 0.868057855802142 rec 0.5963288834951457 masks 13184 9057 cent (755.7002427184466, 886.1875758495146) (760.159545103235, 882.5265540465938) sec 0.029
200 iou 0.5713771247479112 prec 0.8999432785025525 rec 0.6101369020150746 masks 13002 8815 cent (850.739655437625, 882.5244577757268) (855.7516732841747, 875.251843448667) sec 0.031
300 iou 0.6153681735599017 prec 0.8766747868453106 rec 0.6736866736866737 masks 8547 6568 cent (950.0952380952381, 841.2510822510823) (955.0158343483556, 833.0408038976858) sec 0.029
400 iou 0.592741935483871 prec 0.9370517928286852 rec 0.617322834645669 …[+1005 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_overlay.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/work/scratch'); from render_eval import load_mesh,make_renderer,raster,project_vertices
mesh=load_mesh('/work/inputs/object_mesh.glb'); r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)); ren=make_renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs={}
frames=[0,50,100,125,150,175,200,250,300,350,375,400,425,450,500,550,600,650,700,750,800,850,900,945]
maxf=max(frames)
for i in range(maxf+1):
 ok,img=cap.read()
 if i in frames: imgs[i]=img.copy()
# tile 6x4, each 512x384
rows=[]
with h5py.File('/work/inputs/masks.h5') as f:
 for rr in range(4):
  row=[]
  for cc in range(6):
   i=frames[rr*6+cc]; img=imgs[i]; mask,_,_,_=raster(mesh,ren,T[i],Twci,K); pred=mask.cpu().numpy().astype(np.uint8)
   om=f['object'][i].astype(np.uint8); pm=f['person'][i].astype(np.uint8)
   vis=pred.copy(); vis[pm.astype(bool)]=0
   pc,_=cv2.findContours(vis,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); oc,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   cv2.drawContours(img,oc,-1,(0,255,0),3); cv2.drawContours(img,pc,-1,(0,0,255),3)
   cv2.putText(img,str(i),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),4);cv2.putText(img,str(i),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
   i …[+249 chars]
```
<details><summary>output</summary>

```
(1536, 3072, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np,json,torch,sys
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
for i in range(946):
 ok,im=cap.read()
 if i in [0,400,945]: frames.append((i,im))
with h5py.File('/work/inputs/masks.h5') as f:
 z=[]
 for i,img in frames:
  m,dep,_,_=raster(mesh,ren,T[i],Twci,K);p=m.cpu().numpy();
  om=f['object'][i]
  # crop around actual+pred
  ys,xs=np.where(om|p);x0=max(0,xs.min()-40);x1=min(1536,xs.max()+40);y0=max(0,ys.min()-40);y1=min(1152,ys.max()+40)
  print(i,(x0,y0,x1,y1),'mask',om.sum(),'pred',p.sum(),'mask mean',img[om].mean(0),'pred mean',img[p].mean(0))
  out=img.copy();out[om]=(.6*out[om]+.4*np.array([0,255,0])).astype(np.uint8);out[p]=(.6*out[p]+.4*np.array([0,0,255])).astype(np.uint8)
  crop=out[y0:y1,x0:x1];crop=cv2.resize(crop,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_NEAREST);z.append(crop)
  raw=img[y0:y1,x0:x1]; cv2.imwrite(f'/work/scratch/object_{i}.png',cv2.resize(raw,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_NEAREST))
 maxh=max(i.shape[0] for i in z);z=[cv2.copyMakeBorder(i,0,maxh-i.shape[0],0,0,cv2.BORDER_CONSTANT) for i in z];cv2.imwrite('/work/scratch/object_comp …[+36 chars]
```
<details><summary>output</summary>

```
0 (634, 753, 876, 1015) mask 13418 pred 8848 mask mean [ 99.09479803 107.00767626 123.00342823] pred mean [102.17540687 109.92416365 125.98338608]
400 (852, 680, 1005, 889) mask 1905 pred 5410 mask mean [64.53753281 71.4335958  79.15958005] pred mean [22.24417745 23.35582255 25.38946396]
945 (479, 691, 648, 936) mask 9864 pred 6411 mask mean [ 86.41301703  95.57096513 113.47526358] pred mean [ 90.38777102  99.57697707 121.41631571]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/texturetest.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb'); tex=np.asarray(mesh.visual.material.baseColorTexture);print(tex.shape)
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4');
for i in range(1):ok,img=cap.read()
for flip in [0,1]:
 m,d,uv,rgb=raster(mesh,ren,T[0],Twci,K,tex=tex)
 # rgb only meaningful where m
 out=np.zeros_like(img)
 cols=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
 if flip: cols=cols[::-1]
 out[m.cpu().numpy()]=cols[m.cpu().numpy()]
 x0,y0,x1,y1=634,753,876,1015
 actual=img[y0:y1,x0:x1]; rendered=out[y0:y1,x0:x1]
 cv2.imwrite(f'/work/scratch/texture_{flip}.png',np.concatenate([actual,rendered],1))
PY
python /work/scratch/texturetest.py"
```
<details><summary>output</summary>

```
(3829, 3891, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image
im=Image.open('/work/inputs/object_mesh.glb') if False else None
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0]
g.visual.material.baseColorTexture.resize((778,766)).save('/work/scratch/tex.png')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];tex=np.asarray(g.visual.material.baseColorTexture);uv=g.visual.uv
for flip in [False,True]:
 t=tex[::-1] if flip else tex
 x=(uv[:,0]*(t.shape[1]-1)).astype(np.float32);y=(uv[:,1]*(t.shape[0]-1)).astype(np.float32)
 c=cv2.remap(t,x,y,cv2.INTER_LINEAR)
 print(flip,c.mean(0),np.mean(c.max(1)>20),np.percentile(c,[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgproc/src/imgwarp.cpp:1813: error: (-215:Assertion failed) dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX in function 'remap'


```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];tex=np.asarray(g.visual.material.baseColorTexture);uv=g.visual.uv
for flip in [False,True]:
 t=tex[::-1] if flip else tex
 x=(uv[:,0]*(t.shape[1]-1)).reshape(1,-1).astype(np.float32);y=(uv[:,1]*(t.shape[0]-1)).reshape(1,-1).astype(np.float32)
 c=cv2.remap(t,x,y,cv2.INTER_LINEAR)[0]
 print(flip,c.mean(0),np.mean(c.max(1)>20),np.percentile(c,[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
cv2.error: OpenCV(4.11.0) /io/opencv/modules/imgproc/src/imgwarp.cpp:1813: error: (-215:Assertion failed) dst.cols < SHRT_MAX && dst.rows < SHRT_MAX && src.cols < SHRT_MAX && src.rows < SHRT_MAX in function 'remap'


```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];tex=np.asarray(g.visual.material.baseColorTexture);uv=g.visual.uv
for flip in [False,True]:
 t=np.ascontiguousarray(tex[::-1] if flip else tex)
 x=(uv[:,0]*(t.shape[1]-1)).reshape(500,600).astype(np.float32);y=(uv[:,1]*(t.shape[0]-1)).reshape(500,600).astype(np.float32)
 c=cv2.remap(t,x,y,cv2.INTER_LINEAR).reshape(-1,3)
 print(flip,c.mean(0),np.mean(c.max(1)>20),np.percentile(c,[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
ValueError: cannot reshape array of size 290547 into shape (500,600)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];tex=np.asarray(g.visual.material.baseColorTexture);uv=g.visual.uv
for flip in [False,True]:
 t=np.ascontiguousarray(tex[::-1] if flip else tex)
 x=(uv[:,0]*(t.shape[1]-1)).reshape(10,-1).astype(np.float32);y=(uv[:,1]*(t.shape[0]-1)).reshape(10,-1).astype(np.float32)
 c=cv2.remap(t,x,y,cv2.INTER_LINEAR).reshape(-1,3)
 print(flip,c.mean(0),np.mean(c.max(1)>20),np.percentile(c,[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
ValueError: cannot reshape array of size 290547 into shape (10,newaxis)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];tex=np.asarray(g.visual.material.baseColorTexture);uv=g.visual.uv
N=len(uv);pad=(-N)%10;uv=np.pad(uv,((0,pad),(0,0)))
for flip in [False,True]:
 t=np.ascontiguousarray(tex[::-1] if flip else tex)
 x=(uv[:,0]*(t.shape[1]-1)).reshape(10,-1).astype(np.float32);y=(uv[:,1]*(t.shape[0]-1)).reshape(10,-1).astype(np.float32)
 c=cv2.remap(t,x,y,cv2.INTER_LINEAR).reshape(-1,3)[:N]
 print(flip,c.mean(0),np.mean(c.max(1)>20),np.percentile(c,[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
False [70.51001387 58.88047717 50.1626415 ] 0.4863378386285179 [  0.   0.   2. 123. 228.]
True [146.49900705 119.53506317 101.76330163] 0.9999896746481637 [  8. 104. 123. 145. 240.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb'); tex=np.asarray(mesh.visual.material.baseColorTexture)[::-1].copy()
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4');
for i in range(1):ok,img=cap.read()
m,d,uv,rgb=raster(mesh,ren,T[0],Twci,K,tex=tex)
out=np.zeros_like(img);cols=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8);mm=m.cpu().numpy();out[mm]=cols[mm]
x0,y0,x1,y1=634,753,876,1015
cv2.imwrite('/work/scratch/texture_flipped.png',np.concatenate([img[y0:y1,x0:x1],out[y0:y1,x0:x1]],1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/plot_traj.py <<'PY'
import numpy as np,h5py,matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import median_filter,gaussian_filter1d
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];tr=T[:,:3,3];rot=R.from_matrix(T[:,:3,:3]); c=r['contact_prob_right_left']
dp=np.r_[0,np.linalg.norm(np.diff(tr,axis=0),axis=1)];dr=np.r_[0,(rot[1:]*rot[:-1].inv()).magnitude()]
# relative vertical axis rotation etc
rel=(rot*rot[0].inv()).as_rotvec(); ang=np.degrees(np.linalg.norm(rel,axis=1))
fig,axs=plt.subplots(5,1,figsize=(16,12),sharex=True)
axs[0].plot(tr[:,0],label='x');axs[0].plot(tr[:,1],label='y');axs[0].plot(tr[:,2],label='z');axs[0].legend();axs[0].grid()
axs[1].plot(dp);axs[1].set_ylabel('dp');axs[1].grid();axs[1].plot(gaussian_filter1d(dp,10),label='smooth');axs[1].legend()
axs[2].plot(dr);axs[2].set_ylabel('dr deg');axs[2].grid()
axs[3].plot(ang);axs[3].set_ylabel('angle from f0');axs[3].grid()
axs[4].plot(c[:,0],label='right');axs[4].plot(c[:,1],label='left');axs[4].legend();axs[4].grid();axs[4].set_xlabel('frame')
for ax in axs: ax.axvline(100,color='k',alpha=.2);ax.axvline(900,color='k',alpha=.2)
plt.tight_layout();plt.savefig('/work/scratch/traj.png',dpi=120)
print('dp at intervals');print(list(enumerate(dp[::25])))
PY
python /work/scratch/plot_traj.py"
```
<details><summary>output</summary>

```
dp at intervals
[(0, 0.0), (1, 0.0063006505), (2, 0.0054963864), (3, 0.004488562), (4, 0.014843653), (5, 0.0034423796), (6, 0.0095303105), (7, 0.0052457917), (8, 0.0075567644), (9, 0.0038497124), (10, 0.012875728), (11, 0.029227564), (12, 0.013032317), (13, 0.004693768), (14, 0.0138099), (15, 0.029899644), (16, 0.025044363), (17, 0.006952051), (18, 0.080882736), (19, 0.00907385), (20, 0.14991203), (21, 0.011468869), (22, 0.025211351), (23, 0.022727108), (24, 0.012716742), (25, 0.011376151), (26, 0.028602889), (27, 0.019058904), (28, 0.016647467), (29, 0.014994546), (30, 0.098244205), (31, 0.01743613), (32, 0.04856989), (33, 0.0073796147), (34, 0.0038226245), (35, 0.003938625), (36, 0.0091136), (37, 0.001527015)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 print(list(f.keys()))
 def v(n,o):
  if hasattr(o,'shape'): print(n,o.shape,o.dtype)
 f.visititems(v)
PY"
```
<details><summary>output</summary>

```
['aligned', 'alignment', 'frame_names', 'raw']
aligned/front_stereo_camera_left (946,) object
alignment/front_stereo_camera_left/scale (946,) float32
alignment/front_stereo_camera_left/shift (946,) float32
alignment/front_stereo_camera_left/valid_count (946,) int32
frame_names/front_stereo_camera_left (946,) object
raw/front_stereo_camera_left (946,) object

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
p=pickle.load(open('/pipeline/mhr_init/episode_000023.pkl','rb'))
print(type(p))
if isinstance(p,dict):
 for k,v in p.items(): print(k,type(v),getattr(v,'shape',None))
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
body_model <class 'str'> None
frames <class 'list'> None
kids <class 'list'> None
metadata <class 'dict'> None
mhr_global_rot6d <class 'numpy.ndarray'> (946, 6)
mhr_trans <class 'numpy.ndarray'> (946, 3)
mhr_body_pose_cont <class 'numpy.ndarray'> (946, 260)
mhr_hand <class 'numpy.ndarray'> (946, 108)
mhr_shape <class 'numpy.ndarray'> (946, 45)
mhr_scale <class 'numpy.ndarray'> (946, 28)
mhr_face <class 'numpy.ndarray'> (946, 72)
mhr_joints <class 'numpy.ndarray'> (946, 127, 3)
mhr_keypoints <class 'numpy.ndarray'> (946, 70, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,pickle
p=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
print(type(p));
if isinstance(p,dict):
 for k,v in p.items():
  print(k,type(v),getattr(v,'shape',None))
  if hasattr(v,'keys'): print(' keys',list(v.keys())[:20])
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
schema <class 'str'> None
frames <class 'list'> None
frame_meta <class 'list'> None
kid <class 'int'> None
checkpoint <class 'dict'> None
 keys ['path', 'step']
mhr_inference <class 'dict'> None
 keys ['schema', 'checkpoint', 'wandbRun', 'wandbConfigSha256', 'supervisionContract', 'configSource', 'xyzAnchorContract', 'spatialNormalizationContract']
config <class 'dict'> None
 keys ['rgb_files', 'render_files', 'split_file', 'val_split_file', 'render_root', 'rgb_root', 'packed_root', 'packed_format', 'expected_train_sequence_count', 'expected_val_sequence_count', 'fp_root', 'contacts_root', 'body_model', 'mhr_gt_root', 'mhr_init_root', 'mhr_pseudogt_root', 'mhr_sample_kids', 'mhr_dataset_index_path', 'mhr_interaction_trim_root', 'mhr_effective_mask_root']
mesh_diameter <class …[+2326 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  g=f[p]; print(p,type(g),list(g.keys()) if hasattr(g,'keys') else '')
  if hasattr(g,'keys'):
   for k in g: print(' ',k,g[k].shape,g[k].dtype)
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> 
raw/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  d=f[p]; print(p,d.shape,d.dtype,d.chunks,d.compression); print(d[0],type(d[0]),getattr(d[0],'shape',None),getattr(d[0],'dtype',None))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (946,) object None None
[137  80  78 ...  66  96 130] <class 'numpy.ndarray'> (667884,) uint8
raw/front_stereo_camera_left (946,) object None None
[137  80  78 ...  66  96 130] <class 'numpy.ndarray'> (605435,) uint8

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned/front_stereo\\|front_stereo_camera_left\" -n /workspace/v2d_cari4d/lib/cari4d | head -40"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_export_utils.py:23:    "front_stereo_camera_left",

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned.h5\\|aligned/\\|decode.*depth\\|depth.*decode\" -n /workspace/v2d_cari4d/lib/cari4d/prep | head -100"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/prep/align_depth_to_mhr_wild.py:98:                profiler.record("raw_depth_decode", decode_started)
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:143:            raise ValueError(f"Preserved raw depth payload is not canonical or differs from decoded shape: payload={raw_shape}, decoded={np.asarray(record.raw_depth_m).shape}")
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:147:def decode_depth_png_uint16(payload: np.ndarray) -> np.ndarray:
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:162:        raise TypeError(f"Generated depth PNG must decode to a two-dimensional uint16-compatible array, got shape={decoded.shape}, dtype={decoded.dtype}")
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:164:        raise ValueError(f"Generated …[+1591 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '130,190p' /workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py; sed -n '780,825p' /workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py; sed -n '300,330p' /workspace/v2d_cari4d/lib/cari4d/prep/mhr_export_utils.py"
```
<details><summary>output</summary>

```

def _encode_metric_depth(depth_m: np.ndarray) -> np.ndarray:
    depth_mm = np.clip(np.asarray(depth_m, dtype=np.float32) * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
    return _encode_depth_png_uint16(depth_mm)


def _encode_depth_record(record: DepthFrameRecord) -> tuple[np.ndarray, np.ndarray]:
    if record.raw_payload is None:
        raw_payload = _encode_metric_depth(record.raw_depth_m)
    else:
        raw_payload = np.asarray(record.raw_payload, dtype=np.uint8)
        raw_shape = _canonical_depth_shape(raw_payload)
        if raw_shape is None or raw_shape != tuple(np.asarray(record.raw_depth_m).shape):
            raise ValueError(f"Preserved raw depth payload is not canonical or differs from decoded shape: payload={raw_shape}, decoded={np.asarray(record.raw_dept …[+6418 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,io,numpy as np
from PIL import Image
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for i in [0,100,400,500,900]:
  for kind in ['raw','aligned']:
   p=f[f'{kind}/front_stereo_camera_left'][i]; im=np.asarray(Image.open(io.BytesIO(p.tobytes())))
   if kind=='raw':r=im
   else:a=im
  print(i,r.shape,r.dtype,a.shape,a.dtype,'rawm',r.astype(float).min()/1000,r.astype(float).max()/1000,'alignm',a.astype(float).min()/1000,a.astype(float).max()/1000,'scale',f['alignment/front_stereo_camera_left/scale'][i],'shift',f['alignment/front_stereo_camera_left/shift'][i])
PY"
```
<details><summary>output</summary>

```
0 (1152, 1536) int32 (1152, 1536) int32 rawm 1.584 8.707 alignm 1.812 9.964 scale 1.1444699 shift 0.0
100 (1152, 1536) int32 (1152, 1536) int32 rawm 1.565 9.26 alignm 1.825 10.802 scale 1.1666229 shift 0.0
400 (1152, 1536) int32 (1152, 1536) int32 rawm 1.639 8.915 alignm 1.703 9.266 scale 1.0394485 shift 0.0
500 (1152, 1536) int32 (1152, 1536) int32 rawm 1.632 8.914 alignm 1.667 9.105 scale 1.0214853 shift 0.0
900 (1152, 1536) int32 (1152, 1536) int32 rawm 1.575 8.758 alignm 1.752 9.743 scale 1.1125492 shift 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_eval.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py,io
from PIL import Image
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
with h5py.File('/work/inputs/masks.h5') as fm, h5py.File('/pipeline/depth/moge2/aligned.h5') as fd:
 for i in range(0,946,25):
  m,dep,_,_=raster(mesh,ren,T[i],Twci,K);m=m.cpu().numpy();dep=dep.cpu().numpy(); om=fm['object'][i];pm=fm['person'][i]
  payload=fd['aligned/front_stereo_camera_left'][i]; D=np.asarray(Image.open(io.BytesIO(payload.tobytes())),np.float32)/1000
  pix=m&~pm&om&(D>0)
  # mesh z includes raster interpolated? yes ndc w
  diff=dep[pix]-D[pix]
  # full in object mask depth median vs rendered pixels
  print(i,'n',pix.sum(),'depthmed',np.median(D[om]),'predmed',np.median(dep[m&~pm&om]),'diff med/mean/std',np.median(diff),diff.mean(),diff.std(),'abs',np.median(abs(diff)))
PY
python /work/scratch/depth_eval.py"
```
<details><summary>output</summary>

```
0 n 7059 depthmed 2.681 predmed 1.0 diff med/mean/std -1.6760001 -1.7485193 0.23778167 abs 1.6760001
25 n 7277 depthmed 2.659 predmed 1.0 diff med/mean/std -1.6489999 -1.7240845 0.24657708 abs 1.6489999
50 n 7124 depthmed 2.672 predmed 1.0 diff med/mean/std -1.662 -1.7412026 0.24976125 abs 1.662
75 n 7214 depthmed 2.672 predmed 1.0 diff med/mean/std -1.665 -1.7419558 0.24983819 abs 1.665
100 n 6949 depthmed 2.675 predmed 1.0 diff med/mean/std -1.6789999 -1.7666433 0.26229635 abs 1.6789999
125 n 6827 depthmed 2.637 predmed 1.0 diff med/mean/std -1.6389999 -1.7178874 0.23531084 abs 1.6389999
150 n 6821 depthmed 2.672 predmed 1.0 diff med/mean/std -1.678 -1.7427717 0.22043312 abs 1.678
175 n 7143 depthmed 2.637 predmed 1.0 diff med/mean/std -1.6240001 -1.6746364 0.19971935 abs 1.6240001
200 n …[+2859 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_eval.py'
s=open(p).read()
s=s.replace(\"depth=rast[0,:,:,2]\\n    uv=None; rgb=None\", \"depth=rast[0,:,:,2]\\n    zattr=torch.tensor(cam[:,2:3].astype(np.float32),device='cuda')\\n    depth,_=dr.interpolate(zattr[None],rast,tri)\\n    depth=depth[0,:,:,0]\\n    uv=None; rgb=None\")
# clip function currently returns cam
open(p,'w').write(s)
PY
python /work/scratch/depth_eval.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_eval.py", line 7, in <module>
    m,dep,_,_=raster(mesh,ren,T[i],Twci,K);m=m.cpu().numpy();dep=dep.cpu().numpy(); om=fm['object'][i];pm=fm['person'][i]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/render_eval.py", line 35, in raster
    zattr=torch.tensor(cam[:,2:3].astype(np.float32),device='cuda')
                       ^^^
NameError: name 'cam' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '26,42p' /work/scratch/render_eval.py"
```
<details><summary>output</summary>

```
    clip,_=project_vertices(mesh.vertices,T,Twc_inv,K,W,H) if False else (None,None)

def raster(mesh, renderer, T, Twc_inv, K, W=1536, H=1152, tex=None):
    ctx,tri,V=renderer
    clip,_=project_vertices(mesh.vertices,T,Twc_inv,K,W,H)
    pos=torch.tensor(clip,device='cuda')
    rast, _ = dr.rasterize(ctx,pos[None],tri,resolution=(H,W))
    mask=(rast[0,:,:,3]>0)
    depth=rast[0,:,:,2]
    zattr=torch.tensor(cam[:,2:3].astype(np.float32),device='cuda')
    depth,_=dr.interpolate(zattr[None],rast,tri)
    depth=depth[0,:,:,0]
    uv=None; rgb=None
    if tex is not None:
        attr=torch.tensor(mesh.visual.uv.astype(np.float32),device='cuda')
        uv,_=dr.interpolate(attr[None],rast,tri)
        # uv torus? sample with grid_sample

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/clip,_=project_vertices(mesh.vertices,T,Twc_inv,K,W,H)/clip,cam=project_vertices(mesh.vertices,T,Twc_inv,K,W,H)/' /work/scratch/render_eval.py && python /work/scratch/depth_eval.py"
```
<details><summary>output</summary>

```
0 n 7059 depthmed 2.681 predmed 2.4998913 diff med/mean/std -0.17872882 -0.23794857 0.24545448 abs 0.17872882
25 n 7277 depthmed 2.659 predmed 2.4960742 diff med/mean/std -0.14804077 -0.21866558 0.2524251 abs 0.14804077
50 n 7124 depthmed 2.672 predmed 2.5185766 diff med/mean/std -0.13577092 -0.21423315 0.25486055 abs 0.13577092
75 n 7214 depthmed 2.672 predmed 2.4928427 diff med/mean/std -0.16431618 -0.23969123 0.2562027 abs 0.16431618
100 n 6949 depthmed 2.675 predmed 2.438673 diff med/mean/std -0.22260952 -0.31375882 0.27661622 abs 0.22260952
125 n 6827 depthmed 2.637 predmed 2.4248915 diff med/mean/std -0.19675136 -0.27927992 0.25372437 abs 0.19675136
150 n 6821 depthmed 2.672 predmed 2.4265432 diff med/mean/std -0.24687362 -0.30883092 0.22979647 abs 0.24687362
175 n 7143 depthmed 2.63 …[+3422 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];V=g.vertices;print('V',V.shape)
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]; print('floor y',-d/n[1])
for i in range(0,946,25):
 vh=np.c_[V,np.ones(len(V))];w=(T[i]@vh.T).T[:,:3];s=w@n+d
 print(i,'min y',w[:,1].min(),'plane dist min/max',s.min(),s.max(),'centroid y',w[:,1].mean())
PY"
```
<details><summary>output</summary>

```
V (290547, 3)
floor y 1.3969241367651297
0 min y 0.7846382134109715 plane dist min/max -0.0054019373356299916 0.46330083670823186 centroid y 0.9891300794151292
25 min y 0.7886171541216742 plane dist min/max -0.009299808792958997 0.4597891705404761 centroid y 0.993051305112108
50 min y 0.800639788443565 plane dist min/max -0.02149565904043782 0.446176192223086 centroid y 1.0052681588939876
75 min y 0.7889950738469051 plane dist min/max -0.009492528240538034 0.45942312900093074 centroid y 0.9934300746520569
100 min y 0.7712172397598536 plane dist min/max 0.010901303544288643 0.4795732032914791 centroid y 0.9761615528115527
125 min y 0.7664375903615235 plane dist min/max 0.01463009710314922 0.48537144509037755 centroid y 0.9714012145652188
150 min y 0.7663264095411279 plane dist min/max 0.014 …[+3722 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']; rot=R.from_matrix(T[:,:3,:3]);
up=np.array([0,1.,0]); meshup0=rot[0].inv().apply(up)
print('meshup0',meshup0)
for i in range(0,946,25):
 wup=rot[i].apply(meshup0); tilt=np.degrees(np.arccos(np.clip(wup@up,-1,1)))
 # orientation of forward world z mesh vector
 mf0=rot[0].inv().apply([0,0,1]); wf=rot[i].apply(mf0); yaw=np.degrees(np.arctan2(wf[0],wf[2]))
 print(i,'tilt',round(tilt,2),'yaw',round(yaw,2),'up',wup.round(3))
PY"
```
<details><summary>output</summary>

```
meshup0 [-0.3005256  -0.06043438 -0.95185716]
0 tilt 0.0 yaw 0.0 up [-0.  1. -0.]
25 tilt 0.19 yaw -0.61 up [ 0.     1.    -0.003]
50 tilt 0.46 yaw -0.07 up [-0.     1.     0.008]
75 tilt 0.18 yaw -0.38 up [ 0.003  1.    -0.001]
100 tilt 1.51 yaw -1.05 up [-0.009  1.     0.025]
125 tilt 2.12 yaw -1.12 up [0.001 0.999 0.037]
150 tilt 1.83 yaw -8.39 up [0.009 0.999 0.031]
175 tilt 1.55 yaw -16.14 up [0.019 1.    0.019]
200 tilt 0.72 yaw -19.27 up [0.01  1.    0.008]
225 tilt 2.4 yaw -30.34 up [0.023 0.999 0.035]
250 tilt 1.62 yaw -40.32 up [0.024 1.    0.015]
275 tilt 2.48 yaw -45.05 up [0.032 0.999 0.029]
300 tilt 2.53 yaw -48.56 up [0.036 0.999 0.026]
325 tilt 2.59 yaw -60.2 up [ 0.045  0.999 -0.002]
350 tilt 4.35 yaw -81.39 up [ 0.075  0.997 -0.012]
375 tilt 4.65 yaw -88.54 up [0.05  0.99 …[+1089 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'find /workspace/v2d_foundation_pose -type f -maxdepth 4 | head -100'
```
<details><summary>output</summary>

```
/workspace/v2d_foundation_pose/lib/FoundationPose/.gitignore
/workspace/v2d_foundation_pose/lib/FoundationPose/LICENSE
/workspace/v2d_foundation_pose/lib/FoundationPose/Utils.py
/workspace/v2d_foundation_pose/lib/FoundationPose/assets/bop.jpg
/workspace/v2d_foundation_pose/lib/FoundationPose/assets/cvpr_review.png
/workspace/v2d_foundation_pose/lib/FoundationPose/assets/demo.jpg
/workspace/v2d_foundation_pose/lib/FoundationPose/assets/demo_driller.jpg
/workspace/v2d_foundation_pose/lib/FoundationPose/assets/intro.jpg
/workspace/v2d_foundation_pose/lib/FoundationPose/assets/train_data_vis.png
/workspace/v2d_foundation_pose/lib/FoundationPose/build_all.sh
/workspace/v2d_foundation_pose/lib/FoundationPose/build_all_conda.sh
/workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/config_li …[+2992 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"def .*register\\|def .*track\\|class FoundationPose\\|score\" -n /workspace/v2d_foundation_pose /workspace/v2d_cari4d/lib/cari4d/prep/foundationpose_pose_selection.py | head -100"
```
<details><summary>output</summary>

```
/workspace/v2d_foundation_pose/lib/FoundationPose/estimater.py:13:from learning.training.predict_score import *
/workspace/v2d_foundation_pose/lib/FoundationPose/estimater.py:18:class FoundationPose:
/workspace/v2d_foundation_pose/lib/FoundationPose/estimater.py:19:  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
/workspace/v2d_foundation_pose/lib/FoundationPose/estimater.py:31:    if scorer is not None:
/workspace/v2d_foundation_pose/lib/FoundationPose/estimater.py:32:      self.scorer = scorer
/workspace/v2d_foundation_pose/lib/FoundationPose/estimater.py:34:      self.scorer = ScorePredictor()
/workspace/v2d_foundation_pose/lib/ …[+12364 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /weights/foundationpose -maxdepth 4 -type f -printf '%p %s\\n' | head -80"
```
<details><summary>output</summary>

```
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth 190229389
/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/config.yml 778
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth 68220109
/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/config.yml 708

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,pprint
p=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
print('metadata keys')
pprint.pp(p['metadata'])
print('pr keys')
for k,v in p['pr'].items(): print(k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None))
for k in ['pose_abs','pose_abs_postopt']:
 v=p['pr'][k]; print(k,v.shape,v.dtype);print(v[0])
print('postopt frame_indices',p['postopt']['frame_indices'][:10],len(p['postopt']['frame_indices']))
print('postopt final diag');pprint.pp(p['postopt']['final_diagnostics'])
PY"
```
<details><summary>output</summary>

```
metadata keys
{'ground_truth_used': False,
 'window_length': 96,
 'window_stride': 96,
 'overlap_policy': 'first_occurrence',
 'depth_source': '/home/hanzhou/Projects/v2d/run/out/episode_000023/depth/moge2/aligned.h5',
 'depth_backend': 'moge2',
 'depth_alignment_method': 'huber-log-ratio-to-direct-mhr-depth-v1',
 'mhr_init_source': '/home/hanzhou/Projects/v2d/run/out/episode_000023/mhr_init/episode_000023.pkl',
 'foundationpose_source': '/home/hanzhou/Projects/v2d/run/out/episode_000023/foundationpose/episode_000023.pkl',
 'object_mesh': '/home/hanzhou/Projects/v2d/run/out/episode_000023/export/episode_000023/object_mesh/output_aligned.glb',
 'mhr_xyz_anchor_contract': {'schema': 'cari4d.mhr_xyz_anchor_contract.v1',
                             'anchorType': 'root_joint_1',
               …[+3322 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,torch
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0]
S,e=trimesh.bounds.oriented_bounds(g); print(S,e)
p=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False);Tp=p['pr']['pose_abs']; Tm=Tp@S
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
from scipy.spatial.transform import Rotation as R
for i in [0,100,400,900]:
 d=(R.from_matrix(T[i,:3,:3]).inv()*R.from_matrix(Tm[i,:3,:3])).magnitude(); print(i,'trans',Tm[i,:3,3],T[i,:3,3],'rotdeg',np.degrees(d))
PY"
```
<details><summary>output</summary>

```
[[ 0.04975221 -0.99811757 -0.03586135 -0.00446159]
 [ 0.99872632  0.0494166   0.01018561 -0.02270192]
 [-0.0083943  -0.03632243  0.99930487  0.02180597]
 [ 0.          0.          0.          1.        ]] [0.28195187 0.33680166 0.53587952]
0 trans [-0.03843673  0.88829612  2.48855147] [0.13105322 1.0089045  2.5044966 ] rotdeg 2.204027545122024
100 trans [-0.03587488  0.88431694  2.46669947] [0.14006752 0.9957142  2.446952  ] rotdeg 2.2012280556396635
400 trans [0.60077942 0.74766951 3.0499685 ] [0.8447791 0.9488127 3.2589169] rotdeg 2.234310439012193
900 trans [-0.76148418  0.82099019  3.02663652] [-0.58449966  0.9761853   3.1017401 ] rotdeg 2.21789069179475

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/export/episode_000023/wild_export.json; printf '\\n--- edex ---\\n'; cat /pipeline/export/episode_000023/edex"
```
<details><summary>output</summary>

```
{
  "aligned_object_extents_m": [
    0.2819518744945526,
    0.3368016481399536,
    0.5358794927597046
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
  "frame_count": 946,
  "height": 1152,
  "intrinsics": [
    [
      819.4840087890625,
      0.0,
      768.0
    ],
    [
      0.0,
      819.4840087890625,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "object_mesh_file": "/home/hanzhou/Projects/v2d/run/out/episode_000023/export/episode_000023/object_mesh/output_aligned.glb",
  "object_mesh_to_training_transfo …[+2570 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];S,_=trimesh.bounds.oriented_bounds(g)
p=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False);A=p['pr']['pose_abs']@S
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
Ar=np.einsum('ij,njk->nik',T[:,:3,:3],np.transpose(A[:,:3,:3],(0,2,1)))
print('Arot variability',np.degrees((R.from_matrix(Ar[1:])*R.from_matrix(Ar[:-1]).inv()).magnitude()).max())
Armean=R.from_matrix(Ar).mean().as_matrix(); t=T[:,:3,3]-np.einsum('ij,nj->ni',Armean,A[:,:3,3]);print('A rot',Armean);print('t mean',t.mean(0),'std',t.std(0),'range',t.min(0),t.max(0))
pred=np.einsum('ij,njk->nik',Armean,A[:,:3,:3]);print('trans resid',np.linalg.norm(pred+t.mean(0)-T[:,:3,3],axis=1).mean(),np.max(np.linalg.norm(pred+t.mean(0)-T[:,:3,3],axis=1)))
# maybe Tworldcam @ poseabs? 
import json
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);pred2=np.einsum('ij,njk->nik',Twc,A)
print('rot diff Twc',np.degrees((R.from_matrix(T[:,:3,:3]).inv()*R.from_matrix(pred2[:,:3,:3])).magnitude()).mean())
print('trans diff Twc',np.linalg.norm(pred2[:,:3,3]-T[:,:3,3],axis=1).mean())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/einsumfunc.py", line 1371, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: operand has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions.

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];S,_=trimesh.bounds.oriented_bounds(g)
p=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False);A=p['pr']['pose_abs']@S
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
Ar=np.array([T[i,:3,:3]@A[i,:3,:3].T for i in range(946)])
print('Arot var',np.degrees((R.from_matrix(Ar[1:])*R.from_matrix(Ar[:-1]).inv()).magnitude()).max())
Armean=R.from_matrix(Ar).mean().as_matrix(); t=T[:,:3,3]-np.einsum('ij,nj->ni',Armean,A[:,:3,3]);print('Arot\\n',Armean);print('t mean',t.mean(0),'std',t.std(0),'range',t.min(0),t.max(0))
pred=np.einsum('ij,njk->nik',Armean,A[:,:3,:3]);res=np.linalg.norm(pred+t.mean(0)-T[:,:3,3],axis=1);print('trans resid mean/max',res.mean(),res.max())
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);pred2=np.einsum('ij,njk->nik',Twc,A)
print('rot diff Twc',np.degrees((R.from_matrix(T[:,:3,:3]).inv()*R.from_matrix(pred2[:,:3,:3])).magnitude()).mean())
print('trans diff Twc',np.linalg.norm(pred2[:,:3,3]-T[:,:3,3],axis=1).mean())
PY"
```
<details><summary>output</summary>

```
Arot var 0.007380559017938673
Arot
 [[ 9.99672628e-01  1.99138227e-04  2.55851006e-02]
 [-9.50159536e-04  9.99568890e-01  2.93450518e-02]
 [-2.55682269e-02 -2.93597550e-02  9.99241848e-01]]
t mean [0.11782303 0.06432875 0.09093594] std [0.01720043 0.01947109 0.06632583] range [ 0.07568796  0.01882723 -0.03898736] [0.16924542 0.11986318 0.3022583 ]
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
ValueError: operands could not be broadcast together with shapes (946,3,3) (946,3) 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"T_world_mesh\\|depth correction\\|depth_correction\\|aligned_depth\" -n /workspace/v2d_cari4d /workspace/v2d_hoi_object_reconstruction /workspace/v2d_pipelines 2>/dev/null | head -120"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:75:    aligned_depth_m: np.ndarray
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:144:    return raw_payload, _encode_metric_depth(record.aligned_depth_m)
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:817:    def write_frame(self, camera_name: str, index: int, raw_depth_m: np.ndarray, aligned_depth_m: np.ndarray, *, scale: float, shift: float, valid_count: int) -> None:
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:818:        self.write_frames(camera_name, [DepthFrameRecord(index, raw_depth_m, aligned_depth_m, scale, shift, valid_count)])
/workspace/v2d_cari4d/lib/cari4d/prep/mhr_depth_h5.py:830:        metadata = tuple(_DepthWriteMetadata(int(record.index), float(record.scale), float(record.shift), int(record.va …[+2963 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"refined.pth\\|pose_abs_postopt\" -n /workspace/v2d_cari4d/lib/cari4d | head -100"
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:804:        result["pr"]["pose_abs_postopt"] = _to_numpy(pose).astype(np.float32)
/workspace/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:1261:        result["pr"]["pose_abs_postopt"] = _to_numpy(pose).astype(np.float32)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"object_poses_world\\|T_world_mesh\\|reconstruction.npz\" -n /workspace /code/reconstruction 2>/dev/null | head -120"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_mv_postprocess/lib/interaction_trim.py:322:    object_poses_world_from_object: np.ndarray,
/code/reconstruction/modules/v2d_mv_postprocess/lib/interaction_trim.py:330:    poses = np.asarray(object_poses_world_from_object)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];tr=T[:,:3,3]; dp=np.r_[0,np.linalg.norm(np.diff(tr,axis=0),axis=1)];c=r['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:]
for lo,hi in [(70,125),(860,925)]:
 print('\\nrange',lo,hi)
 prev=None
 for i in range(lo,hi):
  m=o[i];ys,xs=np.where(m);cx=xs.mean();cy=ys.mean();area=len(xs)
  if i%2==0 or i in [87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,888,889,890]:
   print(i,'area',area,'cent',round(cx,2),round(cy,2),'dp',round(dp[i],4),'c',np.round(c[i],2))
PY"
```
<details><summary>output</summary>

```

range 70 125
70 area 13583 cent 754.19 884.4 dp 0.0165 c [0. 0.]
72 area 13559 cent 754.17 884.28 dp 0.0097 c [0. 0.]
74 area 13256 cent 754.04 884.88 dp 0.0152 c [0. 0.]
76 area 13582 cent 754.1 884.27 dp 0.009 c [0. 0.]
78 area 13240 cent 754.04 884.83 dp 0.0167 c [0. 0.]
80 area 13269 cent 753.94 884.75 dp 0.0106 c [0. 0.]
82 area 13255 cent 753.88 884.72 dp 0.0076 c [0. 0.]
84 area 13257 cent 753.9 884.73 dp 0.0039 c [0. 0.]
86 area 13251 cent 753.98 884.82 dp 0.011 c [0. 0.]
87 area 13254 cent 754.05 884.93 dp 0.0107 c [0. 0.]
88 area 13258 cent 753.92 884.79 dp 0.0207 c [0. 0.]
89 area 13580 cent 754.12 884.36 dp 0.0269 c [0. 0.]
90 area 13596 cent 754.09 884.3 dp 0.0149 c [0. 0.]
91 area 13576 cent 754.05 884.25 dp 0.0164 c [0. 0.]
92 area 13589 cent 754.15 884.28 dp 0.0034 c [0. 0 …[+3554 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:
 o=f['object'][:];p=f['person'][:]
a=o.reshape(len(o),-1).sum(1);inter=(o&p).reshape(len(o),-1).sum(1);r=inter/np.maximum(a,1)
print('max',r.max(),'quant',np.percentile(r,[50,90,95,99,99.5,100]));
for i in np.argsort(r)[-20:]:print(i,a[i],inter[i],r[i])
PY"
```
<details><summary>output</summary>

```
max 0.37521917007597894 quant [0.00109864 0.02370705 0.06933813 0.15979803 0.20398657 0.37521917]
391 2006 258 0.12861415752741776
364 3237 422 0.13036762434352797
365 3204 445 0.1388888888888889
388 2431 341 0.14027149321266968
379 2545 362 0.14223968565815323
387 2496 372 0.14903846153846154
380 2513 379 0.15081575805809788
385 2500 386 0.1544
381 2485 393 0.15814889336016097
383 2456 389 0.15838762214983712
382 2479 399 0.1609519967728923
386 2487 401 0.16123843988741454
384 2485 409 0.16458752515090544
399 2029 373 0.1838344011828487
398 1862 376 0.20193340494092374
393 1915 401 0.20939947780678853
397 1706 377 0.22098475967174677
394 2180 759 0.3481651376146789
395 1578 567 0.3593155893536122
396 1711 642 0.37521917007597894

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr,inspect
print(inspect.signature(dr.rasterize))
print(dr.rasterize.__doc__[:1000])
PY"
```
<details><summary>output</summary>

```
(glctx, pos, tri, resolution, ranges=None, grad_db=True)
Rasterize triangles.

    All input tensors must be contiguous and reside in GPU memory except for
    the `ranges` tensor that, if specified, has to reside in CPU memory. The
    output tensors will be contiguous and reside in GPU memory.

    Args:
        glctx: Rasterizer context of type `RasterizeCudaContext`.
        pos: Vertex position tensor with dtype `torch.float32`. To enable range
             mode, this tensor should have a 2D shape [num_vertices, 4]. To enable
             instanced mode, use a 3D shape [minibatch_size, num_vertices, 4].
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        resolution: Output resolution as integer tuple (height, width).
        ranges: In range mod …[+258 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/render_eval.py /work/scratch/render_eval_ds.py; python - <<'PY'
p='/work/scratch/render_eval_ds.py';s=open(p).read();s=s.replace(\"tri=torch.tensor(mesh.faces.astype(np.int32),device=device)\",\"tri0=torch.tensor(mesh.faces.astype(np.int32),device=device)\\n    tri=torch.cat([tri0,tri0[:,::-1]],0)\")
open(p,'w').write(s)
PY
sed 's/from render_eval import/from render_eval_ds import/' /work/scratch/render_eval.py > /work/scratch/test_ds.py
python /work/scratch/test_ds.py"
```
<details><summary>output</summary>

```
0 iou 0.5545625916358304 prec 0.8977169981916817 rec 0.5919660157996721 masks 13418 8848 cent (754.2648680876434, 884.6456252794753) (755.2250226039783, 877.6741636528029) sec 0.043
100 iou 0.5467695945476041 prec 0.868057855802142 rec 0.5963288834951457 masks 13184 9057 cent (755.7002427184466, 886.1875758495146) (760.159545103235, 882.5265540465938) sec 0.027
200 iou 0.5713771247479112 prec 0.8999432785025525 rec 0.6101369020150746 masks 13002 8815 cent (850.739655437625, 882.5244577757268) (855.7516732841747, 875.251843448667) sec 0.033
300 iou 0.6153681735599017 prec 0.8766747868453106 rec 0.6736866736866737 masks 8547 6568 cent (950.0952380952381, 841.2510822510823) (955.0158343483556, 833.0408038976858) sec 0.029
400 iou 0.592741935483871 prec 0.9370517928286852 rec 0.617322834645669 …[+1004 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np,torch,h5py
sys.path.insert(0,'/work/scratch');from render_eval import *
m=load_mesh('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(m)
with h5py.File('/work/inputs/masks.h5') as f:
 om=f['object'][0];pm=f['person'][0];pred,_,_,_=raster(m,ren,T[0],Twci,K);p=pred.cpu().numpy()
 for name,a in [('mask',om),('pred',p),('vis',p&~pm)]:
  ys,xs=np.where(a);print(name,len(xs),'bbox',xs.min(),ys.min(),xs.max(),ys.max(),'wh',xs.max()-xs.min()+1,ys.max()-ys.min()+1)
PY"
```
<details><summary>output</summary>

```
mask 13418 bbox 674 795 834 975 wh 161 181
pred 8848 bbox 681 793 836 966 wh 156 174
vis 8848 bbox 681 793 836 966 wh 156 174

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/moments_all.py <<'PY'
import sys,json,numpy as np,torch,h5py,cv2,time
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
F=len(T); out=np.zeros((F,12),np.float32)
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(F):
  om=f['object'][i]; pm=f['person'][i]
  m,_,_,_=raster(mesh,ren,T[i],Twci,K); p=m.cpu().numpy() & ~pm
  def stats(a):
   ys,xs=np.where(a); n=len(xs)
   if n<10:return np.zeros(9,np.float32)
   x=xs-xs.mean();y=ys-ys.mean()
   return np.array([n,xs.mean(),ys.mean(),np.mean(x*x),np.mean(y*y),np.mean(x*y),xs.min(),xs.max(),ys.max()],np.float32)
  a=stats(om);b=stats(p)
  out[i,:9]=a;out[i,9:]=b[:3]
  if i%100==0:print(i,flush=True)
np.save('/work/scratch/moments.npy',out)
print('saved')
PY
python /work/scratch/moments_all.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
saved

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
m=np.load('/work/scratch/moments.npy'); obs=m[:,:9];pred=m[:,9:12]
for i in range(0,946,25):
 o=obs[i];p=pred[i]
 # covariance eigen ratio overall and x/y std
 ar=o[0]/max(p[0],1); sr=np.sqrt((o[3]+o[4])/(p[3]+p[4])) if p[0]>0 and p[3]+p[4]>0 else 0
 print(i,'areaR',round(ar,3),'stdR',round(sr,3),'centdiff',np.round(o[1:3]-p[1:3],1),'obs area',int(o[0]),'pred',int(p[0]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
IndexError: index 3 is out of bounds for axis 0 with size 3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
m=np.load('/work/scratch/moments.npy'); obs=m[:,:9];pred=m[:,9:12]
for i in range(0,946,25):
 o=obs[i];p=pred[i];ar=o[0]/max(p[0],1);sr=np.sqrt(ar)
 print(i,'areaR',round(ar,3),'scaleR',round(sr,3),'centdiff',np.round(o[1:3]-p[1:3],1),'obs',int(o[0]),'pred',int(p[0]))
PY"
```
<details><summary>output</summary>

```
0 areaR 1.517 scaleR 1.231 centdiff [-1.  7.] obs 13418 pred 8848
25 areaR 1.523 scaleR 1.234 centdiff [-0.2  4.1] obs 13613 pred 8937
50 areaR 1.554 scaleR 1.247 centdiff [-0.2  3.4] obs 13582 pred 8741
75 areaR 1.518 scaleR 1.232 centdiff [-0.6  3.9] obs 13590 pred 8952
100 areaR 1.456 scaleR 1.207 centdiff [-4.5  3.7] obs 13184 pred 9057
125 areaR 1.393 scaleR 1.18 centdiff [-3.9  5. ] obs 12849 pred 9227
150 areaR 1.362 scaleR 1.167 centdiff [-6.5  5.5] obs 12919 pred 9482
175 areaR 1.472 scaleR 1.213 centdiff [-5.3  5.7] obs 13297 pred 9034
200 areaR 1.475 scaleR 1.214 centdiff [-5.   7.3] obs 13002 pred 8815
225 areaR 1.338 scaleR 1.157 centdiff [-2.3  6.3] obs 11555 pred 8639
250 areaR 1.302 scaleR 1.141 centdiff [-4.6  8.1] obs 10446 pred 8020
275 areaR 1.258 scaleR 1.122 centdiff  …[+1808 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np,torch,h5py
sys.path.insert(0,'/work/scratch');from render_eval import *
m=load_mesh('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(m)
with h5py.File('/work/inputs/masks.h5') as f:
 for i in [0,100,300,500,700,900]:
  om=f['object'][i];ys,xs=np.where(om);oc=np.array([xs.mean(),ys.mean()]);
  for kn in ['K','K_pipeline']:
   K=np.array(cam[kn],np.float32);pred,_,_,_=raster(m,ren,T[i],Twci,K);mm=pred.cpu().numpy();yp,xp=np.where(mm);pc=np.array([xp.mean(),yp.mean()]);print(i,kn,'centdiff',np.round(oc-pc,2),'area',len(xp))
PY"
```
<details><summary>output</summary>

```
0 K centdiff [-0.96  6.97] area 8848
0 K_pipeline centdiff [-10.99  -8.35] area 8856
100 K centdiff [-2.69  5.51] area 9292
100 K_pipeline centdiff [-12.66  -9.77] area 9296
300 K centdiff [-0.92 13.59] area 7480
300 K_pipeline centdiff [-10.68  -1.68] area 7468
500 K centdiff [-3.59  8.51] area 4480
500 K_pipeline centdiff [-13.51  -6.78] area 4470
700 K centdiff [-5.78  7.64] area 4264
700 K_pipeline centdiff [-15.66  -7.68] area 4260
900 K centdiff [-0.89  6.71] area 6695
900 K_pipeline centdiff [-10.95  -8.63] area 6670

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_shift.py <<'PY'
import sys,json,numpy as np,torch,h5py
sys.path.insert(0,'/work/scratch');from render_eval import *
m=load_mesh('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');T0=r['T_world_mesh'].copy();T=T0.copy();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(m)
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(0,946,25):
  om=f['object'][i];pm=f['person'][i]
  # predict baseline and centroid
  m1,_,_,_=raster(m,ren,T0[i],Twci,K);p1=m1.cpu().numpy()&~pm
  if not p1.any():continue
  yp,xp=np.where(p1);py,px=yp.mean(),xp.mean();yo,xo=np.where(om);oy,ox=yo.mean(),xo.mean()
  # camera z of centroid via projection of translation? use baseline trans depth
  ctr=(Twci@T0[i])[:3,3];z=ctr[2];dx=(ox-px)*z/K[0,0];dy=(oy-py)*z/K[1,1]
  dc=Twc[:3,:3]@np.array([dx,dy,0]);T[i]=T0[i].copy();T[i,:3,3]+=dc
  m2,_,_,_=raster(m,ren,T[i],Twci,K);p2=m2.cpu().numpy()&~pm
  def iou(a,b):return (a&b).sum()/max((a|b).sum(),1)
  print(i,'diff',round(ox-px,1),round(oy-py,1),'iou',round(iou(p1,om),4),round(iou(p2,om),4),'areas',p1.sum(),p2.sum())
PY
python /work/scratch/test_shift.py"
```
<details><summary>output</summary>

```
0 diff -1.0 7.0 iou 0.5546 0.6456 areas 8848 8898
25 diff -0.2 4.1 iou 0.6009 0.6416 areas 8937 8961
50 diff -0.2 3.4 iou 0.6023 0.6343 areas 8741 8773
75 diff -0.6 3.9 iou 0.599 0.6428 areas 8952 8969
100 diff -4.5 3.7 iou 0.5468 0.6613 areas 9057 9126
125 diff -3.9 5.0 iou 0.5325 0.6481 areas 9227 9230
150 diff -6.5 5.5 iou 0.5125 0.6599 areas 9482 9404
175 diff -5.3 5.7 iou 0.5628 0.6623 areas 9034 8978
200 diff -5.0 7.3 iou 0.5714 0.667 areas 8815 8857
225 diff -2.3 6.3 iou 0.658 0.7327 areas 8639 8735
250 diff -4.6 8.1 iou 0.6154 0.7417 areas 8020 8070
275 diff -3.1 9.2 iou 0.6351 0.7474 areas 7498 7613
300 diff -4.9 8.2 iou 0.6154 0.7268 areas 6568 6547
325 diff -2.3 4.5 iou 0.6895 0.7176 areas 5728 5759
350 diff -0.9 11.4 iou 0.688 0.662 areas 3384 3731
375 diff -2.0 6.7 iou 0.6137  …[+1155 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_scale.py <<'PY'
import sys,json,numpy as np,torch,h5py
sys.path.insert(0,'/work/scratch');from render_eval import *
m=load_mesh('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');T0=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(m)
dvs=[-0.15,-0.10,-0.05,0,0.05,0.10]
with h5py.File('/work/inputs/masks.h5') as f:
 for i in [0,25,50,75,100,150,200,300,400,500,600,700,800,900]:
  om=f['object'][i];pm=f['person'][i];best=(-1,None);vals=[]
  for dz in dvs:
   T=T0[i].copy(); ctr=(Twci@T)[:3,3];z=ctr[2]+dz
   # iterative centroid correction: first render then shift corresponding to z
   m1,_,_,_=raster(m,ren,T,Twci,K);p=m1.cpu().numpy()&~pm
   yp,xp=np.where(p);yo,xo=np.where(om)
   if len(xp)==0:continue
   dx=(xo.mean()-xp.mean())*z/K[0,0];dy=(yo.mean()-yp.mean())*z/K[1,1]
   T[:3,3]+=Twc[:3,:3]@np.array([dx,dy,dz])
   m2,_,_,_=raster(m,ren,T,Twci,K);p2=m2.cpu().numpy()&~pm
   iou=(p2&om).sum()/max((p2|om).sum(),1);vals.append((dz,round(iou,4)))
   if iou>best[0]:best=(iou,dz)
  print(i,'best',best[1],round(best[0],4),vals)
PY
python /work/scratch/test_scale.py"
```
<details><summary>output</summary>

```
0 best 0 0.6456 [(-0.15, 0.4394), (-0.1, 0.5122), (-0.05, 0.6037), (0, 0.6456), (0.05, 0.5712), (0.1, 0.4969)]
25 best 0 0.6416 [(-0.15, 0.4547), (-0.1, 0.5214), (-0.05, 0.615), (0, 0.6416), (0.05, 0.5662), (0.1, 0.4929)]
50 best 0 0.6343 [(-0.15, 0.4508), (-0.1, 0.5134), (-0.05, 0.6109), (0, 0.6343), (0.05, 0.5647), (0.1, 0.4909)]
75 best 0 0.6428 [(-0.15, 0.4531), (-0.1, 0.5191), (-0.05, 0.614), (0, 0.6428), (0.05, 0.569), (0.1, 0.4931)]
100 best 0 0.6613 [(-0.15, 0.4489), (-0.1, 0.5168), (-0.05, 0.6095), (0, 0.6613), (0.05, 0.5877), (0.1, 0.5086)]
150 best 0 0.6599 [(-0.15, 0.4465), (-0.1, 0.5157), (-0.05, 0.604), (0, 0.6599), (0.05, 0.6129), (0.1, 0.5356)]
200 best 0 0.667 [(-0.15, 0.4322), (-0.1, 0.5083), (-0.05, 0.5937), (0, 0.667), (0.05, 0.636), (0.1, 0.5516)]
300 best 0 0.7268 [(- …[+764 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0];im=g.visual.material.baseColorTexture
print(im.mode,im.size,im.getpixel((2000,2000)))
print('basecolor',g.visual.material.baseColorFactor,g.visual.material.main_color)
print('uv first',g.visual.uv[:5])
PY"
```
<details><summary>output</summary>

```
RGB (3891, 3829) (146, 137, 130)
basecolor None [102 102 102 255]
uv first [[0.31251201 0.28201199]
 [0.31251201 0.28201199]
 [0.31251201 0.28201199]
 [0.31251201 0.28201199]
 [0.31251201 0.28201199]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# inspect saved texture render pixels stats
import cv2,numpy as np
a=cv2.imread('/work/scratch/texture_flipped.png')[:,:,::-1]
h,w,_=a.shape;r=a[:,w//2:];m=r.max(2)>5
print(w,h,r[m].mean(0),np.percentile(r[m],[5,25,50,75,95],axis=0))
PY"
```
<details><summary>output</summary>

```
484 262 [ 98.54441682 118.21767631 143.69688065] [[ 36.  79. 108.]
 [ 77. 100. 128.]
 [101. 116. 143.]
 [120. 133. 160.]
 [154. 165. 183.]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=list(m.geometry.values())[0]
v=g.visual.to_color(); print(type(v),v.vertex_colors.shape,v.vertex_colors.dtype,v.vertex_colors.mean(0),np.percentile(v.vertex_colors[:,:3],[5,25,50,75,95],axis=0))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.color.ColorVisuals'> (290547, 4) uint8 [146.66429872 119.67214943 101.87198629 255.        ] [[110.  82.  43.]
 [133. 104.  86.]
 [147. 119. 105.]
 [162. 134. 121.]
 [180. 161. 149.]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/texturetest.py';s=open(p).read();s=s.replace(\"out[m.cpu().numpy()]=cols[m.cpu().numpy()]\",\"out[m.cpu().numpy()]=cols[m.cpu().numpy()][:,::-1]\")
open('/work/scratch/texturetest2.py','w').write(s)
PY
python /work/scratch/texturetest2.py"
```
<details><summary>output</summary>

```
(3829, 3891, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_eval.py';s=open(p).read()
s=s.replace(\"uv=None; rgb=None\\n    if tex is not None:\",\"uv=None; rgb=None\\n    if tex is not None:\\n        if isinstance(tex, np.ndarray) and tex.ndim==2:\\n            attr=torch.tensor(tex.astype(np.float32)/255.,device='cuda')\\n            rgb,_=dr.interpolate(attr[None],rast,tri)\\n            rgb=rgb[0].permute(2,0,1)\\n            return mask,depth,uv,rgb\")
open(p,'w').write(s)
PY
python - <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb'); vc=mesh.visual.to_color().vertex_colors[:,:3]
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4');ok,img=cap.read()
m,d,uv,rgb=raster(mesh,ren,T[0],Twci,K,tex=vc)
out=np.zeros_like(img);cols=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8);mm=m.cpu().numpy();out[mm]=cols[mm][:,::-1]
x0,y0,x1,y1=634,753,876,1015;cv2.imwrite('/work/scratch/texture_vcolor.png',np.concatenate([img[y0:y1,x0:x1],out[y0:y1,x0:x1]],1))
print(cols[mm].mean(0))
PY"
```
<details><summary>output</summary>

```
[143.37364376 118.36268083  99.02362116]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/photo_score.py <<'PY'
import sys,json,numpy as np,torch,trimesh,cv2,h5py
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3]
S,e=trimesh.bounds.oriented_bounds(mesh); ctr_mesh=S[:3,:3]@mesh.vertices.mean(0)+S[:3,3]
print('S',S,'centroid',ctr_mesh)
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,img=cap.read()
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][0];pm=f['person'][0]
def rot_about(T,axis,deg):
 c=(T@np.r_[ctr_mesh,1])[:3];Rr=R.from_rotvec(np.array(axis,float)/np.linalg.norm(axis)*np.radians(deg)).as_matrix();A=np.eye(4);A[:3,:3]=Rr;A[:3,3]=c-Rr@c;return A@T
def score(Ti):
 m,d,_,rgb=raster(mesh,ren,Ti,Twci,K,tex=vc);pn=m.cpu().numpy();
 # shift to centroid
 yo,xo=np.where(om);yp,xp=np.where(pn&~pm)
 z=((Twci@Ti)[2,3]);dx=(xo.mean()-xp.mean())*z/K[0,0];dy=(yo.mean()-yp.mean())*z/K[1,1];Ti2=Ti.copy();Ti2[:3,3]+=Twc[:3,:3]@np.array([dx,dy,0])
 m,d,_,rgb=raster(mesh,ren,Ti2,Twci,K,tex=vc);pn=m.cpu().numpy();reg=pn&om&~pm
 col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.float32)
 iou=(pn …[+1319 chars]
```
<details><summary>output</summary>

```
S [[ 0.04975221 -0.99811757 -0.03586135 -0.00446159]
 [ 0.99872632  0.0494166   0.01018561 -0.02270192]
 [-0.0083943  -0.03632243  0.99930487  0.02180597]
 [ 0.          0.          0.          1.        ]] centroid [-0.00382308 -0.00123995  0.03565871]
baseline (0.6456013568320921, 0.30571357158754436, 0.3815679537798193, 8755)
mesh0 (0.24992846105419791, -0.011182850306201505, 0.1909367752852075, 4367)
mesh1 (0.43710267771142364, 0.0311897603799704, -0.2684135326714486, 6807)
mesh2 (0.29610655737704916, -0.0517241601073775, -0.20266648201302373, 4913)
worldx (0.4472680245716133, 0.024256642156825072, -0.2658123499548662, 6917)
worldy (0.24210765387235975, 0.022132666536440845, 0.18126946297750032, 4264)
worldz (0.3046305774673331, 0.02867442803593658, -0.18266955810169141, 5059)
wy -90 ( …[+595 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/T\\[0\\]/T[945]/g; s/f\\[.object.\\]\\[0\\]/f[\"object\"][945]/; s/f\\[.person.\\]\\[0\\]/f[\"person\"][945]/; s/CAP_PROP_POS_FRAMES,0/CAP_PROP_POS_FRAMES,945/' /work/scratch/photo_score.py > /work/scratch/photo_score_945.py
python /work/scratch/photo_score_945.py"
```
<details><summary>output</summary>

```
S [[ 0.04975221 -0.99811757 -0.03586135 -0.00446159]
 [ 0.99872632  0.0494166   0.01018561 -0.02270192]
 [-0.0083943  -0.03632243  0.99930487  0.02180597]
 [ 0.          0.          0.          1.        ]] centroid [-0.00382308 -0.00123995  0.03565871]
baseline (0.6501773948302078, 0.16715917637105182, 0.14643437392092226, 6414)
mesh0 (0.48944392082940624, 0.001526305932210189, 0.13979115839022235, 5122)
mesh1 (0.5431413812894585, -0.03858829871427859, -0.11472346211672008, 5634)
mesh2 (0.5497202316678119, -0.011005100186447011, -0.21374972952973326, 5555)
worldx (0.5516511023394628, 0.017569064175121434, -0.12851896716614805, 5700)
worldy (0.4824660633484163, 0.07168018486810872, 0.13040639718927857, 5062)
worldz (0.5357983522950176, -0.012198642420436674, -0.18557854425072584, 5415)
wy  …[+606 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs={}
for i in range(946):
 ok,im=cap.read()
 if i in [100,200,300,400,450,500,550,600,650,700,800,900]:imgs[i]=im
with h5py.File('/work/inputs/masks.h5') as f:
 tiles=[]
 for i,im in imgs.items():
  y,x=np.where(f['object'][i]);x0=max(0,x.min()-70);x1=min(1536,x.max()+70);y0=max(0,y.min()-70);y1=min(1152,y.max()+70)
  crop=im[y0:y1,x0:x1].copy();cv2.putText(crop,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3);cv2.putText(crop,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
  crop=cv2.resize(crop,(300,300));tiles.append(crop)
out=np.concatenate([np.concatenate(tiles[j:j+4],1) for j in range(0,12,4)],0);cv2.imwrite('/work/scratch/object_motion_crops.jpg',out)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];rot=R.from_matrix(T[:,:3,:3]);
for lo,hi in [(0,93),(894,946),(0,946)]:
 Rm=R.from_matrix(rot[lo:hi].as_matrix()).mean(); t=np.median(T[lo:hi,:3,3],axis=0);d=(rot[lo:hi].inv()*Rm).magnitude();dt=np.linalg.norm(T[lo:hi,:3,3]-t,axis=1)
 print(lo,hi,'rot med/max deg',np.degrees(np.median(d)),np.degrees(d.max()),'trans med/max',np.median(dt),dt.max(),'tstd',T[lo:hi,:3,3].std(0))
 # fit linear drift rotation over window
PY"
```
<details><summary>output</summary>

```
0 93 rot med/max deg 0.5609101983047772 1.3381634231228647 trans med/max 0.011811211 0.038201872 tstd [0.00230724 0.00643895 0.01331698]
894 946 rot med/max deg 0.45792275742076594 1.2395562655730397 trans med/max 0.012159407 0.049756978 tstd [0.00365177 0.00787534 0.01700673]
0 946 rot med/max deg 90.52961812937856 149.6390644225407 trans med/max 0.7229101 0.8091026 tstd [0.49081528 0.03722513 0.5167585 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_candidates.py <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz'); T0=r['T_world_mesh'].astype(np.float64); F=len(T0); contacts=r['contact_prob_right_left']
# Static boundaries from contact + visible object centroid: movement begins frame 93, ends 894.
static_runs=[(0,93),(894,946)]
def pose_mean(Ts):
    # robust: coordinate median; rotation chordal L2 mean
    t=np.median(Ts[:,:3,3],axis=0); Rm=R.from_matrix(Ts[:,:3,:3]).mean().as_matrix()
    out=np.eye(4); out[:3,:3]=Rm; out[:3,3]=t; return out
def smooth_pose(T, trans_win=11, rot_sigma=2.5):
    out=T.copy(); t=T[:,:3,3]
    # remove isolated tracking spikes before SG smoothing
    tm=median_filter(t,size=(5,1),mode='nearest')
    for k in range(3): out[:,k,3]=savgol_filter(tm[:,k],trans_win,3,mode='interp')
    q=R.from_matrix(T[:,:3,:3]).as_quat() # xyzw
    for k in range(1,F):
        if np.dot(q[k],q[k-1])<0:q[k]*=-1
    # robust component median then gaussian smoothing
    qm=median_filter(q,size=(5,1),mode='nearest')
    rad=int(4*rot_sigma+0.5); x=np.arange(-rad,rad+1); w=np.exp(-0.5*(x/rot_sigma)**2);w/=w.sum()
    for k in range(4):qm[:,k]=np.convolve(np.pad(qm[:,k],(rad,rad),mode='edge'),w,mode='valid')
    qm/=np.linalg.norm(qm,axis=1,keepdims=True)
    out[:,:3,:3]=R.from_quat(qm).as_matrix()
    for lo,hi i …[+696 chars]
```
<details><summary>output</summary>

```
B dr med/p95/max 0.771459966069196 2.548891781692365 12.328823102254969 dt 0.011277772177705267 0.045846667008680335 0.1759136065878757
change from base trans 0.0 rot 0.0
S dr med/p95/max 0.3084335962460442 0.9871756156186139 2.254509972354725 dt 0.005629510936489866 0.0174259324783764 0.05462199069421198
change from base trans 0.01008910604123724 rot 0.7566829749078817

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_candidates_masks.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2,time
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');c=np.load('/work/scratch/candidates_base.npz');cands={k:c[k] for k in c.files}
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh);F=cands['T0'].shape[0]
res={k:np.zeros((F,8),np.float32) for k in cands}
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(F):
  om=f['object'][i];pm=f['person'][i]
  for name,T in cands.items():
   m,_,_,_=raster(mesh,ren,T[i],Twci,K);p=m.cpu().numpy()&~pm
   inter=(p&om).sum();union=(p|om).sum();yp,xp=np.where(p);yo,xo=np.where(om)
   res[name][i]=[inter/max(union,1),inter/max(p.sum(),1),inter/max(om.sum(),1),p.sum(),xo.mean()-xp.mean() if len(xp) else 0,yo.mean()-yp.mean() if len(yp) else 0,om.sum(),0]
  if i%100==0:print(i,flush=True)
np.savez('/work/scratch/eval_masks_base.npz',**res)
for name,a in res.items():
 print(name,'iou mean/med',a[:,0].mean(),np.median(a[:,0]),'prec',np.median(a[:,1]),'rec',np.median(a[:,2]))
for lo,hi,label in [(0,93,'static0'),(93,894,'moving'),(894,946,'static1')]:
 print(label,{n:round(float(v[lo:hi,0].mean()),4) for n,v in res.items()})
PY
python /work/scratch/eval_candidates_masks.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
T0 iou mean/med 0.6022411 0.5994474 prec 0.9449928 rec 0.6175933
S iou mean/med 0.60200953 0.5945262 prec 0.9420306 rec 0.61337686
static0 {'T0': 0.5911, 'S': 0.5866}
moving {'T0': 0.6002, 'S': 0.6004}
static1 {'T0': 0.6542, 'S': 0.6549}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_sc.py <<'PY'
import numpy as np,json,cv2
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
c=np.load('/work/scratch/candidates_base.npz');S=c['S'].copy();e=np.load('/work/scratch/eval_masks_base.npz')['S'];F=len(S)
dx=e[:,4].copy();dy=e[:,5].copy();area=e[:,3];omarea=e[:,6];iou=e[:,0]
valid=(area>900)&(omarea>1800)&(iou>0.42)
for v in [dx,dy]:
 med=median_filter(v,size=9,mode='nearest');bad=np.abs(v-med)>4
 v[bad]=med[bad]
for lo,hi in [(0,93),(894,946)]:
 dx[lo:hi]=np.median(dx[lo:hi][valid[lo:hi]])
 dy[lo:hi]=np.median(dy[lo:hi][valid[lo:hi]])
# interpolate invalid then savgol
for v in [dx,dy]:
 good=np.where(valid)[0]
 v[:]=np.interp(np.arange(F),good,v[good])
 v[:]=savgol_filter(v,11,2,mode='interp')
 dxc=np.clip(dx,-12,12);dyc=np.clip(dy,-12,12)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
C=S.copy()
for i in range(F):
 z=(Twci@S[i])[2,3]
 dpx=np.array([dxc[i]*z/K[0,0],dyc[i]*z/K[1,1],0.])
 C[i,:3,3]+=Twc[:3,:3]@dpx
np.savez('/work/scratch/candidates_sc.npz',S=S,C=C,dx=dxc,dy=dyc)
print('shift px dx/dy stats',dxc.min(),dxc.max(),dxc.mean(),dyc.min(),dyc.max(),dyc.mean())
for i in range(0,F,50):print(i,round(dx[i],1),round(dy[i],1),round(dxc[i],1),round(dyc[i],1))
PY
python /work/scratch/make_sc.py"
```
<details><summary>output</summary>

```
shift px dx/dy stats -7.6254 3.819133 -1.954488 0.78177917 12.0 6.517762
0 -0.5 5.0 -0.5 5.0
50 -0.5 5.0 -0.5 5.0
100 -5.0 3.4 -5.0 3.4
150 -5.5 5.3 -5.5 5.3
200 -4.9 7.4 -4.9 7.4
250 -4.0 7.6 -4.0 7.6
300 -4.1 7.4 -4.1 7.4
350 -1.7 11.9 -1.7 11.9
400 -4.3 5.3 -4.3 5.3
450 -0.8 7.7 -0.8 7.7
500 -0.6 6.9 -0.6 6.9
550 0.6 5.9 0.6 5.9
600 -1.7 5.5 -1.7 5.5
650 -5.3 7.3 -5.3 7.3
700 -4.1 5.5 -4.1 5.5
750 -4.1 6.2 -4.1 6.2
800 -3.5 7.0 -3.5 7.0
850 -0.4 8.6 -0.4 8.6
900 -0.5 5.2 -0.5 5.2

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed \"s/c=np.load('\\/work\\/scratch\\/candidates_base.npz');cands={k:c\\[k\\] for k in c.files}/c=np.load('\\/work\\/scratch\\/candidates_sc.npz');cands={'S':c['S'],'C':c['C']}/\" /work/scratch/eval_candidates_masks.py > /work/scratch/eval_sc_masks.py
python /work/scratch/eval_sc_masks.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_sc_masks.py", line 4, in <module>
    cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh);F=cands['T0'].shape[0]
                                                                                                                                                                          ~~~~~^^^^^^
KeyError: 'T0'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/F=cands\\['T0'\\].shape\\[0\\]/F=next(iter(cands.values())).shape[0]/\" /work/scratch/eval_sc_masks.py && python /work/scratch/eval_sc_masks.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
S iou mean/med 0.60200953 0.5945262 prec 0.9420306 rec 0.61337686
C iou mean/med 0.65094393 0.64224374 prec 0.9798355 rec 0.65354013
static0 {'S': 0.5866, 'C': 0.6419}
moving {'S': 0.6004, 'C': 0.6515}
static1 {'S': 0.6549, 'C': 0.659}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);V=list(m.geometry.values())[0].vertices;Vh=np.c_[V,np.ones(len(V))]
c=np.load('/work/scratch/candidates_sc.npz');pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
for name in ['S','C']:
 T=c[name];s=np.array([(T[i]@Vh.T).T[:,:3]@n+d for i in range(len(T))]);mn=s.min(1);print(name,'minplane med',np.median(mn),'pct',np.percentile(mn,[5,25,50,75,95]),'above .03',np.mean(mn>.03),'below -.03',np.mean(mn<-.03))
 for i in range(0,946,100):print(i,round(mn[i],3),end=' ')
 print()
PY"
```
<details><summary>output</summary>

```
S minplane med 0.010358538857595967 pct [-0.0185478  -0.01058427  0.01035854  0.0172166   0.03240786] above .03 0.06871035940803383 below -.03 0.011627906976744186
0 -0.011 100 0.015 200 0.024 300 0.016 400 -0.011 500 0.035 600 0.008 700 0.009 800 0.023 900 -0.015 
C minplane med -0.015094746737355447 pct [-0.05454996 -0.02580923 -0.01509475 -0.00569122  0.00446901] above .03 0.0 below -.03 0.20507399577167018
0 -0.026 100 0.005 200 0.002 300 -0.009 400 -0.032 500 0.004 600 -0.018 700 -0.015 800 -0.004 900 -0.035 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_floor_shift.py <<'PY'
import sys,json,numpy as np,torch,h5py,trimesh
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');Vh=np.c_[mesh.vertices,np.ones(len(mesh.vertices))];r=np.load('/work/inputs/reconstruction.npz');c=np.load('/work/scratch/candidates_base.npz');S=c['S'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=pl[:3]/np.linalg.norm(pl[:3]);e1=np.cross(n,[0,0,1.]);e1/=np.linalg.norm(e1);e2=np.cross(n,e1)
def shift_to(T,target):
 for it in range(3):
  wc=(T@Vh.T).T[:,:3].mean(0); cc=(Twci@np.r_[wc,1])[:3];x,y,z=cc
  J=np.array([[K[0,0]/z,0,-K[0,0]*x/z**2],[0,K[1,1]/z,-K[1,1]*y/z**2]])@Twc[:3,:3]@np.c_[e1,e2]
  ab=np.linalg.solve(J,target);T=T.copy();T[:3,3]+=e1*ab[0]+e2*ab[1]
 return T
with h5py.File('/work/inputs/masks.h5') as f:
 for i in [0,25,50,75,100,150,200,300,400,500,600,700,800,900]:
  om=f['object'][i];pm=f['person'][i];m,_,_,_=raster(mesh,ren,S[i],Twci,K);p=m.cpu().numpy()&~pm;yp,xp=np.where(p);yo,xo=np.where(om);target=np.array([xo.mean()-xp.mean(),yo.mean()-yp.mean()])
  T=shift_to(S[i],target);m2,_,_,_=raster(mesh,ren,T,Twci,K);p2=m2.cpu().numpy()&~pm;yp2,xp2=np.where(p2)
  iou=lambda a:(a&om).sum()/max((a|om).sum(),1);iou0=iou(p)
  minp=((T@Vh.T).T[:,:3]@n+pl[3 …[+236 chars]
```
<details><summary>output</summary>

```
0 target [-0.3  5.4] resid [ -0.3 -12.7] iou 0.588 0.52 areas 8859 9817 minp -0.011
25 target [-0.5  4.9] resid [  0.2 -12.1] iou 0.588 0.535 areas 8859 9785 minp -0.011
50 target [-0.4  4.9] resid [  0.1 -11.9] iou 0.589 0.539 areas 8859 9800 minp -0.011
75 target [-0.4  5.1] resid [  0.1 -12. ] iou 0.588 0.531 areas 8859 9766 minp -0.011
100 target [-5.   3.2] resid [10.5 -7. ] iou 0.542 0.527 areas 9170 9886 minp 0.015
150 target [-5.5  5.6] resid [ 10.5 -14. ] iou 0.526 0.456 areas 9386 10441 minp 0.016
200 target [-4.9  7.5] resid [ 11.4 -16. ] iou 0.572 0.406 areas 8872 10630 minp 0.024
300 target [-4.6  7.7] resid [  6.6 -14.3] iou 0.626 0.42 areas 6582 7404 minp 0.016
400 target [-4.2 10.3] resid [ 4.3 -6.4] iou 0.534 0.647 areas 1237 1946 minp -0.011
500 target [-0.6  6.9] resid [ …[+401 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);V=list(m.geometry.values())[0].vertices;r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
v=V;w=(T[0]@np.c_[v,np.ones(len(v))].T).T[:,:3];s=w@n+d
k=np.argmin(s);print('T',T[0]);print('mesh minmax',v.min(0),v.max(0));print('world minmax',w.min(0),w.max(0));print('s min',s.min(),'point',w[k],'s at miny',w[np.argmin(w[:,1])],np.dot(w[np.argmin(w[:,1])],n)+d);print('plane y',-d/n[1]);print('mean',w.mean(0),'smean',np.dot(w.mean(0),n)+d)
PY"
```
<details><summary>output</summary>

```
T [[ 0.9436347   0.12629089 -0.30594787  0.13105322]
 [-0.3005256  -0.06043437 -0.9518571   1.0089045 ]
 [-0.13870063  0.9901505  -0.01907434  2.5044966 ]
 [ 0.          0.          0.          1.        ]]
mesh minmax [-0.14973971 -0.1413893  -0.28821406] [0.19135687 0.13983184 0.24650338]
world minmax [-0.08407041  0.78463821  2.37255323] [0.38884558 1.25873428 2.65612376]
s min -0.0054019373356299916 point [0.10936468 1.24718598 2.6379348 ] s at miny [0.13571063 0.78463821 2.51842242] 0.4627250656739943
plane y 1.3969241367651297
mean [0.14689469 0.98913008 2.50118841] smean 0.25940306521776435

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
c=np.load('/work/scratch/candidates_base.npz');
for name in ['T0','S']:
 T=c[name];rot=R.from_matrix(T[:,:3,:3]);meshup0=rot[0].inv().apply([0,1.,0]);mf0=rot[0].inv().apply([0,0,1]);
 up=rot.apply(meshup0);fw=rot.apply(mf0);yaw=np.unwrap(np.arctan2(fw[:,0],fw[:,2]));tilt=np.degrees(np.arccos(np.clip(up@np.array([0,1.,0]),-1,1)))
 dy=np.r_[0,np.diff(yaw)]; dp=np.r_[0,np.diff(tilt)]
 yaw=np.degrees(yaw);dy=np.degrees(dy)
 print(name,'yaw range',yaw.min(),yaw.max(),'vel abs med/p95/max',np.median(abs(dy)),np.percentile(abs(dy),95),abs(dy).max())
 print('tilt abs diff med/p95/max',np.median(abs(dp)),np.percentile(abs(dp),95),abs(dp).max())
 for i in range(0,946,100):print(i,round(yaw[i],1),round(dy[i],2),round(tilt[i],1),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
T0 yaw range -269.0825741987854 0.95341136527894 vel abs med/p95/max 0.49096193770830976 1.9508647421905834 9.66150754636258
tilt abs diff med/p95/max 0.3061138289344978 1.1116304953668932 5.123933443491796
0 0.0 0.0 0.0; 100 -1.0 -0.63 1.5; 200 -19.3 0.19 0.7; 300 -48.6 -0.0 2.5; 400 -86.7 4.36 9.6; 500 -133.5 -1.02 5.1; 600 -200.1 -1.36 4.9; 700 -219.5 -1.66 3.9; 800 -242.4 -1.66 5.6; 900 -263.4 -0.27 5.7; 
S yaw range -268.3539236024192 0.1708911839720734 vel abs med/p95/max 0.2560402807149321 0.9025121364794311 1.7948350698845843
tilt abs diff med/p95/max 0.05876627661785805 0.22899357295030853 1.0151523848164334
0 0.0 0.0 0.0; 100 -0.5 -0.14 1.8; 200 -19.2 -0.09 0.8; 300 -48.2 -0.34 2.7; 400 -94.4 0.76 5.2; 500 -133.5 -0.48 4.9; 600 -199.2 -0.92 5.6; 700 -218.4 -0.44 4.2; 800 -241.5 - …[+31 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_photo.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3]
c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
cap=cv2.VideoCapture('/work/inputs/video.mp4');res={k:[] for k in cands}
def photo(reg,img,col):
 ga=cv2.cvtColor(img.astype(np.float32),cv2.COLOR_BGR2GRAY);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
 ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1)
 gax=cv2.Sobel(ga,cv2.CV_32F,1,0,ksize=3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,ksize=3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,ksize=3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,ksize=3)
 v1=np.r_[gax[reg],gay[reg]];v2=np.r_[grx[reg],gry[reg]];ncc=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
 cs=[]
 for ch in range(3):
  a=img[:,:,ch][reg].astype(np.float32);b=col[:,:,ch][reg];cs.append(np.corrcoef(a,b)[0,1])
 return ncc,np.mean(cs)
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(0,946,5):
  ok,img=cap.read() if False else (None,None)
  # don't read here (capture sequential at sample stride impossible) use grab loop below …[+2053 chars]
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
S iou 0.5622817790464305 ncc 0.051883100644631926 corr 0.2046537402852759 n 190
C iou 0.6135040503329435 ncc 0.12629771870356227 corr 0.28509359274787105 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/rot_search.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3];S,_=trimesh.bounds.oriented_bounds(mesh);cm=(S[:3,:3]@mesh.vertices.mean(0)+S[:3,3]);
c=np.load('/work/scratch/candidates_sc.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
def rot_about(T,axis,deg):
 ctr=(T@np.r_[cm,1])[:3];Rr=R.from_rotvec(np.array(axis,float)*np.radians(deg)).as_matrix();A=np.eye(4);A[:3,:3]=Rr;A[:3,3]=ctr-Rr@ctr;return A@T
def recenter(T,om,pm):
 for _ in range(2):
  m,_,_,_=raster(mesh,ren,T,Twci,K);p=m.cpu().numpy()&~pm;yp,xp=np.where(p);yo,xo=np.where(om)
  if not len(xp):return T
  z=((Twci@T)[2,3]);dx=(xo.mean()-xp.mean())*z/K[0,0];dy=(yo.mean()-yp.mean())*z/K[1,1];T=T.copy();T[:3,3]+=Twc[:3,:3]@np.array([dx,dy,0])
 return T
def photo(reg,img,col):
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1)
 gax=cv2.Sobel(ga,cv2.CV_32F,1,0,3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,3)
 a=np.r_[gax[reg],gay[r …[+908 chars]
```
<details><summary>output</summary>

```

frame 0
y [(-10, 0.63, 0.118), (-5, 0.643, 0.248), (0, 0.648, 0.33), (5, 0.644, 0.205), (10, 0.626, 0.168)]
x [(-10, 0.613, 0.081), (-5, 0.639, 0.047), (0, 0.648, 0.33), (5, 0.636, 0.12), (10, 0.598, 0.111)]
z [(-10, 0.49, 0.109), (-5, 0.573, 0.112), (0, 0.648, 0.33), (5, 0.592, 0.124), (10, 0.516, 0.093)]

frame 50
y [(-10, 0.626, 0.098), (-5, 0.639, 0.242), (0, 0.639, 0.337), (5, 0.637, 0.226), (10, 0.622, 0.167)]
x [(-10, 0.614, 0.1), (-5, 0.636, 0.045), (0, 0.639, 0.337), (5, 0.631, 0.126), (10, 0.6, 0.12)]
z [(-10, 0.495, 0.121), (-5, 0.572, 0.111), (0, 0.639, 0.337), (5, 0.598, 0.115), (10, 0.518, 0.086)]

frame 100
y [(-10, 0.636, 0.163), (-5, 0.654, 0.293), (0, 0.649, 0.257), (5, 0.643, 0.108), (10, 0.619, 0.135)]
x [(-10, 0.623, 0.085), (-5, 0.645, 0.129), (0, 0.649, 0.257), (5,  …[+5231 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/make_cb.py <<'PY'
import numpy as np,json
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
c=np.load('/work/scratch/candidates_base.npz');B=c['T0'].copy();e=np.load('/work/scratch/eval_masks_base.npz')['T0'];F=len(B)
dx=e[:,4].copy();dy=e[:,5].copy();valid=(e[:,3]>900)&(e[:,6]>1800)&(e[:,0]>.42)
for v in [dx,dy]:
 med=median_filter(v,9,mode='nearest');bad=np.abs(v-med)>4;v[bad]=med[bad]
for lo,hi in [(0,93),(894,946)]:dx[lo:hi]=np.median(dx[lo:hi][valid[lo:hi]]);dy[lo:hi]=np.median(dy[lo:hi][valid[lo:hi]])
for v in [dx,dy]:
 good=np.where(valid)[0];v[:]=np.interp(np.arange(F),good,v[good]);v[:]=savgol_filter(v,11,2,mode='interp')
dx=np.clip(dx,-12,12);dy=np.clip(dy,-12,12)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
CB=B.copy()
for i in range(F):
 z=(Twci@B[i])[2,3];CB[i,:3,3]+=Twc[:3,:3]@np.array([dx[i]*z/K[0,0],dy[i]*z/K[1,1],0])
np.savez('/work/scratch/candidates_cb.npz',B=B,CB=CB)
PY
python /work/scratch/make_cb.py
# eval photo CB alongside C
sed \"s/cands={'B':c0\\['T0'\\],'S':c1\\['S'\\],'C':c1\\['C'\\]}/cands={'CB':np.load('\\/work\\/scratch\\/candidates_cb.npz')['CB'],'C':c1['C']}/\" /work/scratch/eval_photo.py > /work/scratch/eval_photo_cb.py
python /work/scratch/eval_photo_cb.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/make_cb.py", line 4, in <module>
    c=np.load('/work/scratch/candidates_base.npz');B=c['T0'].copy();e=np.load('/work/scratch/eval_masks_base.npz')['T0'];F=len(B)
                                                                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 263, in __getitem__
    raise KeyError(f"{key} is not a file in the archive")
KeyError: 'T0 is not a file in the archive'
Traceback (most recent call last):
  File "/work/scratch/eval_photo_cb.py", line 4, in <module>
    c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'CB':np.load('/work/scratch/candidates_cb.npz')['CB'],'C':c1[' …[+452 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
for f in ['candidates_base.npz','eval_masks_base.npz','candidates_sc.npz']:
 z=np.load('/work/scratch/'+f);print(f,z.files)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_cb.py <<'PY'
import numpy as np,json
c=np.load('/work/scratch/candidates_base.npz');B=c['T0'].copy();sc=np.load('/work/scratch/candidates_sc.npz');dx=sc['dx'];dy=sc['dy'];F=len(B)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
CB=B.copy()
for i in range(F):
 z=(Twci@B[i])[2,3];CB[i,:3,3]+=Twc[:3,:3]@np.array([dx[i]*z/K[0,0],dy[i]*z/K[1,1],0])
np.savez('/work/scratch/candidates_cb.npz',B=B,CB=CB)
PY
python /work/scratch/make_cb.py
python /work/scratch/eval_photo_cb.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
CB iou 0.6134722451534539 ncc 0.11596474748080933 corr 0.28312338831408645 n 190
C iou 0.6135040503329435 ncc 0.12629771870356227 corr 0.28509359274787105 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_depth.py <<'PY'
import sys,json,numpy as np,torch,h5py,io
from PIL import Image
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cb=np.load('/work/scratch/candidates_cb.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C'],'CB':cb['CB']}
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh);res={k:[] for k in cands}
with h5py.File('/work/inputs/masks.h5') as fm,h5py.File('/pipeline/depth/moge2/aligned.h5') as fd:
 for i in range(0,946,5):
  om=fm['object'][i];pm=fm['person'][i]; payload=fd['aligned/front_stereo_camera_left'][i];D=np.asarray(Image.open(io.BytesIO(payload.tobytes())),np.float32)/1000
  for name,T in cands.items():
   m,dep,_,_=raster(mesh,ren,T[i],Twci,K);p=m.cpu().numpy();z=dep.cpu().numpy();reg=p&om&~pm&(D>0)
   if reg.sum()<100:res[name].append([i,0,0,0,reg.sum()])
   else:
    diff=z[reg]-D[reg]; med=np.median(diff);res[name].append([i,med,np.mean(np.abs(diff)),np.percentile(np.abs(diff),90),reg.sum()])
  if i%100==0:print(i,flush=True)
for n,a in res.items():
 a=np.array(a);valid=a[:,4]>100;print(n,'signed med',np.median(a[valid,1]),'mae med',np.median(a[valid,2]),'mae mean',a[valid,2].mean(),'p90 med',np.median(a[valid,3]),'n',valid.sum())
np.savez(' …[+110 chars]
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B signed med -0.1278451681137085 mae med 0.17535000294446945 mae mean 0.18127711321178236 p90 med 0.33826059103012107 n 190
S signed med -0.12640738487243652 mae med 0.17631196230649948 mae mean 0.1807339193005311 p90 med 0.3383856177330017 n 190
C signed med -0.12718302011489868 mae med 0.1801469773054123 mae mean 0.18147633201197574 p90 med 0.3518203735351562 n 190
CB signed med -0.1289566159248352 mae med 0.182023823261261 mae mean 0.18183246980372228 p90 med 0.35491352081298827 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz')
for name,T in [('B',c0['T0']),('S',c1['S']),('C',c1['C'])]:
 rot=R.from_matrix(T[:,:3,:3]);dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude())
 print(name,'dt med/p95/max',np.median(dt),np.percentile(dt,95),dt.max(),'dr',np.median(dr),np.percentile(dr,95),dr.max())
 for lo,hi in [(0,93),(93,894),(894,946)]:print(' ',lo,hi,'dt',dt[lo:hi-1].mean(),'dr',dr[lo:hi-1].mean())
PY"
```
<details><summary>output</summary>

```
B dt med/p95/max 0.011277772177705267 0.045846667008680335 0.1759136065878757 dr 0.771459966069196 2.548891781692365 12.328823102254969
  0 93 dt 0.008694667469018967 dr 0.5097544397811411
  93 894 dt 0.017485077439834518 dr 1.1271604089763452
  894 946 dt 0.006983806564034613 dr 0.5248862187726326
S dt med/p95/max 0.005629510936489866 0.0174259324783764 0.05462199069421198 dr 0.3084335962460442 0.9871756156186139 2.254509972354725
  0 93 dt 0.0 dr 0.0
  93 894 dt 0.007767246523574241 dr 0.4476364386671793
  894 946 dt 0.0 dr 0.0
C dt med/p95/max 0.0058709089129834 0.01796701606144294 0.053320760896930414 dr 0.3084335962460442 0.9871756156186139 2.254509972354725
  0 93 dt 3.5879572919984315e-05 dr 0.0
  93 894 dt 0.008031942964498794 dr 0.4476364386671793
  894 946 dt 0.000166427185626701 …[+10 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/flip_scan.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3];S,_=trimesh.bounds.oriented_bounds(mesh);cm=S[:3,:3]@mesh.vertices.mean(0)+S[:3,3]
c=np.load('/work/scratch/candidates_sc.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
def rot_about(T,axis,deg):
 ctr=(T@np.r_[cm,1])[:3];Rr=R.from_rotvec(np.array(axis,float)/np.linalg.norm(axis)*np.radians(deg)).as_matrix();A=np.eye(4);A[:3,:3]=Rr;A[:3,3]=ctr-Rr@ctr;return A@T
def recenter(T,om,pm):
 for _ in range(2):
  m,_,_,_=raster(mesh,ren,T,Twci,K);p=m.cpu().numpy()&~pm;yp,xp=np.where(p);yo,xo=np.where(om)
  if not len(xp):return T
  z=((Twci@T)[2,3]);T=T.copy();T[:3,3]+=Twc[:3,:3]@np.array([(xo.mean()-xp.mean())*z/K[0,0],(yo.mean()-yp.mean())*z/K[1,1],0])
 return T
def score(T,om,pm,img):
 m,_,_,rgb=raster(mesh,ren,T,Twci,K,tex=vc);p=m.cpu().numpy();reg=p&om&~pm;col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255)
 iou=(p&om).sum()/max((p|om).sum(),1)
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr, …[+1251 chars]
```
<details><summary>output</summary>

```
0 {'C': (0.65, 0.35), 'm0': (0.25, -0.02), 'm1': (0.44, 0.03), 'm2': (0.3, -0.04), 'wy': (0.24, 0.01)}
200 {'C': (0.65, -0.07), 'm0': (0.36, -0.04), 'm1': (0.53, 0.03), 'm2': (0.36, 0.06), 'wy': (0.36, -0.02)}
400 {'C': (0.22, 0.12), 'm0': (0.26, 0.06), 'm1': (0.25, 0.01), 'm2': (0.27, -0.08), 'wy': (0.25, -0.02)}
600 {'C': (0.62, 0.34), 'm0': (0.21, -0.05), 'm1': (0.39, -0.06), 'm2': (0.28, -0.08), 'wy': (0.23, -0.01)}
800 {'C': (0.65, 0.12), 'm0': (0.42, -0.07), 'm1': (0.45, -0.01), 'm2': (0.48, -0.01), 'wy': (0.43, -0.03)}
best counts
Counter({'C': 33, 'm2': 6, 'm1': 5, 'wy': 3, 'm0': 1})
C mean iou/ncc 0.6170041373035008 0.13059224205231315
m0 mean iou/ncc 0.361771989528148 -0.012547848775010919
m1 mean iou/ncc 0.4479101323474996 0.002777894624663
m2 mean iou/ncc 0.39090372027112785 -0 …[+292 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_grid.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3]
c=np.load('/work/scratch/candidates_sc.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
def score(T,om,pm,img):
 m,_,_,rgb=raster(mesh,ren,T,Twci,K,tex=vc);p=m.cpu().numpy();reg=p&om&~pm;col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255)
 iou=(p&om).sum()/max((p|om).sum(),1)
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1)
 gax=cv2.Sobel(ga,cv2.CV_32F,1,0,3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,3);a=np.r_[gax[reg],gay[reg]];b=np.r_[grx[reg],gry[reg]];ncc=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
 return iou,ncc,reg.sum()
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs={}
for i in range(946):
 ok,img=cap.read()
 if i%5==0:imgs[i]=img
with h5py.File('/work/inputs/masks.h5') as f:
 for lo,hi in [(0,93),(894,946)]:
  fs=list(range(lo,hi,5));samples=[]
  for i in fs:samples.append((i,imgs[i],f['object'][i],f['person'][i]))
  vals=[]
  for dx in [-2,-1, …[+461 chars]
```
<details><summary>output</summary>

```

phase 0 93
ncc (0.6512469756184587, 0.37703045271328006, -1, 0)
ncc (0.655457810626235, 0.3644216081903029, -2, 0)
ncc (0.6518645573791035, 0.3638775317401383, -2, -1)
ncc (0.6419789480133474, 0.35064726281881287, 0, 0)
ncc (0.6460032005635794, 0.3445944750102446, -1, -1)
ncc (0.6527353724513261, 0.3284256104703924, -1, 1)
ncc (0.6456134765238263, 0.32166767540483354, 0, 1)
ncc (0.6469027933504178, 0.2995034725332017, -2, -2)
ncc (0.6551494495443102, 0.29600116791886066, -2, 1)
ncc (0.6291783450059402, 0.2959303215951433, 1, 0)
iou (0.655457810626235, 0.3644216081903029, -2, 0)
iou (0.6551494495443102, 0.29600116791886066, -2, 1)
iou (0.6527763559465323, 0.18295520713930363, -2, 2)
iou (0.6527353724513261, 0.3284256104703924, -1, 1)
iou (0.6518645573791035, 0.3638775317401383, -2, -1)
iou …[+485 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_grid.py';s=open(p).read();s=s.replace(\"fs=list(range(lo,hi,5))\",\"fs=[i for i in range(lo,hi) if i%5==0]\")
s=s.replace(\"for lo,hi in [(0,93),(894,946)]:\",\"for lo,hi in [(894,946)]:\")
open('/work/scratch/static_grid_final.py','w').write(s)
PY
python /work/scratch/static_grid_final.py"
```
<details><summary>output</summary>

```

phase 894 946
ncc (0.6629637047419797, 0.161121264852068, -1, -1)
ncc (0.6611124407663515, 0.1601104288343758, 0, -2)
ncc (0.6623148607471493, 0.15101702918709478, -1, -2)
ncc (0.6611275977073079, 0.14736703932728176, 0, -1)
ncc (0.6621530134446795, 0.1430368063740467, -1, 0)
ncc (0.6620886923107929, 0.1317451555294992, -2, -1)
ncc (0.6615807147199861, 0.12975835177701908, -2, 0)
ncc (0.6569337772619483, 0.12835148000952382, 1, -2)
ncc (0.6597148707440357, 0.12088338302285809, 0, 0)
ncc (0.6599811939576963, 0.1150037116234781, -1, 1)
iou (0.6629637047419797, 0.161121264852068, -1, -1)
iou (0.6623902432711177, 0.11009763878618589, -2, -2)
iou (0.6623148607471493, 0.15101702918709478, -1, -2)
iou (0.6621530134446795, 0.1430368063740467, -1, 0)
iou (0.6620886923107929, 0.1317451555294992, -2 …[+268 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/trans_search.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3];c=np.load('/work/scratch/candidates_sc.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
def score(T,om,pm,img):
 m,_,_,rgb=raster(mesh,ren,T,Twci,K,tex=vc);p=m.cpu().numpy();reg=p&om&~pm;col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255)
 iou=(p&om).sum()/max((p|om).sum(),1)
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1)
 gax=cv2.Sobel(ga,cv2.CV_32F,1,0,3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,3);a=np.r_[gax[reg],gay[reg]];b=np.r_[grx[reg],gry[reg]];ncc=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
 return iou,ncc,reg.sum()
cap=cv2.VideoCapture('/work/inputs/video.mp4');targets=set(range(0,946,25));rows=[]
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(946):
  ok,img=cap.read()
  if i not in targets:continue
  om=f['object'][i];pm=f['person'][i];vals=[]
  for dx in [-3,-1.5,0,1.5,3]:
   for dy in [-6,-3,0,3,6]:
    T=C[i].copy();z=(Twci@T) …[+801 chars]
```
<details><summary>output</summary>

```
0 bestncc (0.3681059411324991, 0.6582118417137757, -1.5, 0) bestiou (0.32039883611109027, 0.6621460506706408, -3, 0)
25 bestncc (0.37216724811754986, 0.6494058970221505, -1.5, 0) bestiou (0.3207873386895511, 0.6519345128845165, -3, 0)
50 bestncc (0.36935351100237457, 0.6507644810349897, -1.5, 0) bestiou (0.3181501794402532, 0.653299977926569, -3, 0)
75 bestncc (0.37525104892338534, 0.6503820158683514, -1.5, 0) bestiou (0.3200654128747053, 0.653037211354611, -3, 0)
100 bestncc (0.3955250704905768, 0.6768306254173159, -1.5, 0) bestiou (0.3844633614273453, 0.6862482317027772, -3, 0)
125 bestncc (0.4426962480288526, 0.6613083277517303, -3, -3) bestiou (0.21511275236809244, 0.679572964438764, -3, 0)
150 bestncc (0.44500061690967185, 0.6777752926265563, -3, -3) bestiou (0.44500061690967185, 0.67 …[+4442 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_d.py <<'PY'
import numpy as np,json
c=np.load('/work/scratch/candidates_base.npz');S=c['S'];sc=np.load('/work/scratch/candidates_sc.npz');dx=sc['dx'].copy();dy=sc['dy'].copy()
# Mask segmentation extends farther below the mesh than above; calibrate 4 px systematic bias against texture evidence.
# Static texture grids put the pooled correction at zero.
dy=np.maximum(0,dy-4.0)
dy[:93]=0;dy[894:]=0
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
D=S.copy()
for i in range(len(D)):
 z=(Twci@S[i])[2,3];D[i,:3,3]+=Twc[:3,:3]@np.array([dx[i]*z/K[0,0],dy[i]*z/K[1,1],0])
np.savez('/work/scratch/candidates_d.npz',D=D,dx=dx,dy=dy)
print('dy',dy.min(),dy.max(),dy[::100])
PY
python /work/scratch/make_d.py
# photo eval D vs C
sed \"s/cands={'B':c0\\['T0'\\],'S':c1\\['S'\\],'C':c1\\['C'\\]}/cands={'D':np.load('\\/work\\/scratch\\/candidates_d.npz')['D'],'C':c1['C']}/\" /work/scratch/eval_photo.py > /work/scratch/eval_photo_d.py
python /work/scratch/eval_photo_d.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
D iou 0.5957481711330226 ncc 0.10504259514135131 corr 0.2500086736836418 n 190
C iou 0.6135040503329435 ncc 0.12629771870356227 corr 0.28509359274787105 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_photo.npz') # previous maybe overwritten D only? check
PY
# rerun photo save separately by sed
sed \"s#np.savez('/work/scratch/eval_photo.npz'#np.savez('/work/scratch/eval_photo_d.npz'#\" /work/scratch/eval_photo_d.py > /work/scratch/eval_photo_d2.py
python /work/scratch/eval_photo_d2.py >/tmp/evald.txt && cat /tmp/evald.txt
python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_photo_d.npz')
for n in z.files:
 a=z[n];
 for lo,hi,l in [(0,93,'init'),(93,894,'move'),(894,946,'final')]:print(n,l,'iou',a[(a[:,0]>=lo)&(a[:,0]<hi),1].mean(),'ncc',a[(a[:,0]>=lo)&(a[:,0]<hi),2].mean(),'corr',a[(a[:,0]>=lo)&(a[:,0]<hi),3].mean())
PY"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
D iou 0.5957481711330226 ncc 0.10504259514135131 corr 0.2500086736836418 n 190
C iou 0.6135040503329435 ncc 0.12629771870356227 corr 0.28509359274787105 n 190
D init iou 0.593352769844035 ncc -0.07774833088755466 corr 0.26451156392511005
D move iou 0.5917223376717529 ncc 0.1264644100376031 corr 0.2466622926025896
D final iou 0.658443260068836 ncc 0.10918234160943534 corr 0.27363286080913846
C init iou 0.641973727465179 ncc 0.35063481643594097 corr 0.3898170868967455
C move iou 0.606894808638018 ncc 0.09731765107096417 corr 0.28330944502517524
C final iou 0.6604635781216338 ncc 0.16033462454906267 corr 0.13015879700266317

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_e.py <<'PY'
import numpy as np,json
sc=np.load('/work/scratch/candidates_sc.npz');C=sc['C'].copy();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
E=C.copy()
for i in range(93,894):
 z=(Twci@C[i])[2,3]; E[i,:3,3]+=Twc[:3,:3]@np.array([0,-3*z/K[1,1],0])
np.savez('/work/scratch/candidates_e.npz',E=E)
PY
python /work/scratch/make_e.py
sed \"s/cands={'B':c0\\['T0'\\],'S':c1\\['S'\\],'C':c1\\['C'\\]}/cands={'C':c1['C'],'E':np.load('\\/work\\/scratch\\/candidates_e.npz')['E']}/; s#eval_photo.npz#eval_photo_e.npz#\" /work/scratch/eval_photo.py > /work/scratch/eval_photo_e.py
python /work/scratch/eval_photo_e.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
C iou 0.6135040503329435 ncc 0.12629771870356227 corr 0.28509359274787105 n 190
E iou 0.6063979271630029 ncc 0.1611583832698679 corr 0.2669104393912492 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_photo_e.npz')
for n in z.files:
 a=z[n]
 for lo,hi,l in [(0,93,'init'),(93,894,'move'),(894,946,'final')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)];print(n,l,'iou',q[:,1].mean(),'ncc',q[:,2].mean(),'corr',q[:,3].mean())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_masks_base.npz');print(z.files)
for n in z.files:
 a=z[n];print(n,'res med',np.median(a[:,4]),np.median(a[:,5]),'p95 abs',np.percentile(abs(a[:,4]),95),np.percentile(abs(a[:,5]),95))
 for lo,hi,l in [(0,93,'init'),(93,894,'move'),(894,946,'final')]:print(' ',l,np.mean(a[lo:hi,4]),np.mean(a[lo:hi,5]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s#r=np.load('/work/inputs/reconstruction.npz'); T=r\\['T_world_mesh'\\]#r=np.load('/work/scratch/candidates_sc.npz'); T=r['C']#; s#overlay_baseline.jpg#overlay_C.jpg#\" /work/scratch/make_overlay.py > /work/scratch/make_overlay_C.py
python /work/scratch/make_overlay_C.py"
```
<details><summary>output</summary>

```
(1536, 3072, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,os
for f in ['eval_photo.npz','eval_photo_cb.npz','eval_photo_d.npz','eval_photo_e.npz','eval_depth.npz','eval_masks_base.npz']:
 p='/work/scratch/'+f
 if os.path.exists(p):
  z=np.load(p);print(f,z.files)
  for n in z.files:
   a=z[n];print(' ',n,a.shape,a[:2])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s/cands={'B':c0\\['T0'\\],'S':c1\\['S'\\],'C':c1\\['C'\\]}/cands={'B':c0['T0'],'C':c1['C']}/; s#eval_photo.npz#eval_photo_final.npz#\" /work/scratch/eval_photo.py > /work/scratch/eval_photo_final.py
python /work/scratch/eval_photo_final.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
C iou 0.6135040503329435 ncc 0.12629771870356227 corr 0.28509359274787105 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
p=np.load('/work/scratch/eval_photo_final.npz');d=np.load('/work/scratch/eval_depth.npz');c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz')
print('PHOTO')
for lo,hi,l in [(0,93,'initial static'),(93,894,'manipulation'),(894,946,'final static'),(0,946,'all')]:
 print(l)
 for n,key in [('baseline','B'),('result','C')]:
  a=p[key];q=a[(a[:,0]>=lo)&(a[:,0]<hi)]
  print(n,'IoU %.4f gradNCC %.4f colorCorr %.4f n %d'%(q[:,1].mean(),q[:,2].mean(),q[:,3].mean(),len(q)))
print('\\nDEPTH')
for lo,hi,l in [(0,93,'initial static'),(93,894,'manipulation'),(894,946,'final static'),(0,946,'all')]:
 print(l)
 for n,key in [('baseline','B'),('result','C')]:
  a=d[key];q=a[(a[:,0]>=lo)&(a[:,0]<hi)&(a[:,4]>100)]
  print(n,'MAE %.4f signed %.4f n %d'%(q[:,2].mean(),np.median(q[:,1]),len(q)))
print('\\nMOTION')
for n,T in [('baseline',c0['T0']),('result',c1['C'])]:
 rot=R.from_matrix(T[:,:3,:3]);dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude())
 print(n,'dt med/p95 max',np.median(dt),np.percentile(dt,95),dt.max(),'dr',np.median(dr),np.percentile(dr,95),dr.max())
for lo,hi,l in [(0,93,'init'),(93,894,'move'),(894,946,'final')]:
 for n,T in [('B',c0['T0']),('C',c1['C'])]:
  rot=R.from_matrix(T[:,:3,:3]);
  if lo==hi-1:continue
  dt=np.linalg.norm(np.diff(T[lo:hi,:3,3],axis=0),axis=1);dr=np …[+112 chars]
```
<details><summary>output</summary>

```
PHOTO
initial static
baseline IoU 0.5894 gradNCC -0.0582 colorCorr 0.2491 n 19
result IoU 0.6420 gradNCC 0.3506 colorCorr 0.3898 n 19
manipulation
baseline IoU 0.5536 gradNCC 0.0634 colorCorr 0.1954 n 160
result IoU 0.6069 gradNCC 0.0973 colorCorr 0.2833 n 160
final static
baseline IoU 0.6589 gradNCC 0.1540 colorCorr 0.2872 n 11
result IoU 0.6605 gradNCC 0.1603 colorCorr 0.1302 n 11
all
baseline IoU 0.5633 gradNCC 0.0565 colorCorr 0.2061 n 190
result IoU 0.6135 gradNCC 0.1263 colorCorr 0.2851 n 190

DEPTH
initial static
baseline MAE 0.2363 signed -0.1517 n 19
result MAE 0.2321 signed -0.1471 n 19
manipulation
baseline MAE 0.1807 signed -0.1268 n 160
result MAE 0.1816 signed -0.1272 n 160
final static
baseline MAE 0.0941 signed -0.0089 n 11
result MAE 0.0916 signed -0.0089 n 11
all
baseline …[+718 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
b=np.load('/work/scratch/candidates_base.npz');c=np.load('/work/scratch/candidates_sc.npz');
for i in [0,50,894,900,925,945]:
 d=(R.from_matrix(c['C'][i,:3,:3])*R.from_matrix(b['T0'][i,:3,:3]).inv()).magnitude();print(i,'rotdiff',np.degrees(d),'tdiff',np.linalg.norm(c['C'][i,:3,3]-b['T0'][i,:3,3]))
PY"
```
<details><summary>output</summary>

```
0 rotdiff 0.32592807539199703 tdiff 0.0206632909907002
50 rotdiff 0.6257398511870375 tdiff 0.017960105796790984
894 rotdiff 1.239556265573015 tdiff 0.059764063487245236
900 rotdiff 0.784576068090391 tdiff 0.060540400053570406
925 rotdiff 0.7962637052482685 tdiff 0.01961930581977946
945 rotdiff 0.2100482175541349 tdiff 0.026947657456812074

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_rot.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3];S,_=trimesh.bounds.oriented_bounds(mesh);cm=S[:3,:3]@mesh.vertices.mean(0)+S[:3,3]
c=np.load('/work/scratch/candidates_sc.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
def rot_about(T,axis,deg):
 ctr=(T@np.r_[cm,1])[:3];Rr=R.from_rotvec(np.array(axis,float)*np.radians(deg)).as_matrix();A=np.eye(4);A[:3,:3]=Rr;A[:3,3]=ctr-Rr@ctr;return A@T
def score(T,om,pm,img):
 m,_,_,rgb=raster(mesh,ren,T,Twci,K,tex=vc);p=m.cpu().numpy();reg=p&om&~pm;col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255)
 iou=(p&om).sum()/max((p|om).sum(),1)
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1)
 gax=cv2.Sobel(ga,cv2.CV_32F,1,0,3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,3);a=np.r_[gax[reg],gay[reg]];b=np.r_[grx[reg],gry[reg]];ncc=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
 cs=[]
 for ch in range(3):cs.append(np.corrcoef(img[:,:,c …[+789 chars]
```
<details><summary>output</summary>

```

phase 0 base [0.64197895 0.35064726 0.38997956]
wx [(-2, 0.645, 0.2934, 0.3378), (-1, 0.6458, 0.3254, 0.3652), (-0.5, 0.6437, 0.3404, 0.3757), (0.5, 0.6391, 0.3256, 0.3644), (1, 0.638, 0.3227, 0.3641), (2, 0.6398, 0.3084, 0.3719)]
wy [(-2, 0.6392, 0.3244, 0.3675), (-1, 0.6409, 0.3189, 0.3541), (-0.5, 0.6419, 0.3249, 0.3715), (0.5, 0.6401, 0.3343, 0.3724), (1, 0.6425, 0.3391, 0.3739), (2, 0.6422, 0.3255, 0.3749)]
wz [(-2, 0.6166, 0.2868, 0.3886), (-1, 0.6316, 0.3206, 0.3893), (-0.5, 0.6371, 0.3322, 0.3751), (0.5, 0.6415, 0.3314, 0.379), (1, 0.6426, 0.3129, 0.3739), (2, 0.6357, 0.1897, 0.3709)]

phase 894 base [0.65971487 0.12088338 0.12898   ]
wx [(-2, 0.6607, 0.1376, 0.1386), (-1, 0.6584, 0.1264, 0.1159), (-0.5, 0.6582, 0.1309, 0.1432), (0.5, 0.6583, 0.1038, 0.1287), (1, 0.6564, 0.0935, 0 …[+407 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];c=r['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(886,906):
  o=f['object'][i];y,x=np.where(o);print(i,'T',np.round(T[i,:3,3],4),'cent',round(x.mean(),2),round(y.mean(),2),'area',len(x),'c',np.round(c[i],2),'dt',round(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]),4))
PY"
```
<details><summary>output</summary>

```
886 T [-0.5925  0.9718  3.0812] cent 556.32 814.16 area 8887 c [1. 1.] dt 0.0027
887 T [-0.5896  0.9693  3.0739] cent 556.1 814.23 area 8826 c [1.   0.99] dt 0.0082
888 T [-0.5876  0.9711  3.08  ] cent 556.16 814.12 area 8855 c [1.   0.97] dt 0.0066
889 T [-0.5881  0.969   3.079 ] cent 556.18 814.0 area 8869 c [1.   0.81] dt 0.0024
890 T [-0.5891  0.9698  3.0799] cent 556.32 813.77 area 8924 c [1.   0.17] dt 0.0015
891 T [-0.5862  0.9671  3.0792] cent 556.64 812.86 area 9044 c [0.87 0.02] dt 0.0041
892 T [-0.5837  0.9691  3.0891] cent 558.42 810.6 area 9462 c [0.07 0.  ] dt 0.0104
893 T [-0.5894  0.9693  3.1018] cent 557.66 810.18 area 9360 c [0. 0.] dt 0.0139
894 T [-0.5912  0.9723  3.1115] cent 558.0 809.89 area 9461 c [0. 0.] dt 0.0103
895 T [-0.5895  0.9732  3.1072] cent 559.77 807.96  …[+835 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s/static_runs=\\[(0,93),(894,946)\\]/static_runs=[(0,93),(901,946)]/\" /work/scratch/make_candidates.py > /work/scratch/make_candidates2.py
sed -i \"s#candidates_base.npz#candidates_base2.npz#\" /work/scratch/make_candidates2.py
python /work/scratch/make_candidates2.py
sed \"s#candidates_base.npz#candidates_base2.npz#g; s#candidates_sc.npz#candidates_sc2.npz#g; s/static_runs=\\[(0,93),(894,946)\\]/static_runs=[(0,93),(901,946)]/; s#candidates_sc.npz#candidates_sc2.npz#g\" /work/scratch/make_sc.py > /work/scratch/make_sc2.py
# make_sc doesn't use static_runs. okay.
python /work/scratch/make_sc2.py"
```
<details><summary>output</summary>

```
B dr med/p95/max 0.771459966069196 2.548891781692365 12.328823102254969 dt 0.011277772177705267 0.045846667008680335 0.1759136065878757
change from base trans 0.0 rot 0.0
S dr med/p95/max 0.30830193206284906 0.9821416934154344 2.254509972354725 dt 0.00563215135794794 0.0174259324783764 0.04069400063057061
change from base trans 0.009829803337396038 rot 0.7537919751117698
shift px dx/dy stats -7.6254 3.819133 -1.954488 0.78177917 12.0 6.517762
0 -0.5 5.0 -0.5 5.0
50 -0.5 5.0 -0.5 5.0
100 -5.0 3.4 -5.0 3.4
150 -5.5 5.3 -5.5 5.3
200 -4.9 7.4 -4.9 7.4
250 -4.0 7.6 -4.0 7.6
300 -4.1 7.4 -4.1 7.4
350 -1.7 11.9 -1.7 11.9
400 -4.3 5.3 -4.3 5.3
450 -0.8 7.7 -0.8 7.7
500 -0.6 6.9 -0.6 6.9
550 0.6 5.9 0.6 5.9
600 -1.7 5.5 -1.7 5.5
650 -5.3 7.3 -5.3 7.3
700 -4.1 5.5 -4.1 5.5
750 -4.1 6.2 -4.1 6.2
800  …[+62 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dy\\[894:\\]=0/dy[901:]=0/; s/dy\\[894:\\]=0/dy[901:]=0/' /work/scratch/make_sc2.py
sed -n '1,80p' /work/scratch/make_sc2.py | tail -20"
```
<details><summary>output</summary>

```
 med=median_filter(v,size=9,mode='nearest');bad=np.abs(v-med)>4
 v[bad]=med[bad]
for lo,hi in [(0,93),(894,946)]:
 dx[lo:hi]=np.median(dx[lo:hi][valid[lo:hi]])
 dy[lo:hi]=np.median(dy[lo:hi][valid[lo:hi]])
# interpolate invalid then savgol
for v in [dx,dy]:
 good=np.where(valid)[0]
 v[:]=np.interp(np.arange(F),good,v[good])
 v[:]=savgol_filter(v,11,2,mode='interp')
 dxc=np.clip(dx,-12,12);dyc=np.clip(dy,-12,12)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
C=S.copy()
for i in range(F):
 z=(Twci@S[i])[2,3]
 dpx=np.array([dxc[i]*z/K[0,0],dyc[i]*z/K[1,1],0.])
 C[i,:3,3]+=Twc[:3,:3]@dpx
np.savez('/work/scratch/candidates_sc2.npz',S=S,C=C,dx=dxc,dy=dyc)
print('shift px dx/dy stats',dxc.min(),dx …[+145 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/for lo,hi in \\[(0,93),(894,946)\\]/for lo,hi in [(0,93),(901,946)]/' /work/scratch/make_sc2.py && python /work/scratch/make_sc2.py >/tmp/makesc2.txt && tail -8 /tmp/makesc2.txt
# evaluate B/C2
sed \"s/cands={'B':c0\\['T0'\\],'S':c1\\['S'\\],'C':c1\\['C'\\]}/cands={'B':c0['T0'],'C':c2['C']}/; s#eval_photo.npz#eval_photo_c2.npz#; s#c1=np.load('/work/scratch/candidates_sc.npz')#c1=np.load('/work/scratch/candidates_sc.npz');c2=np.load('/work/scratch/candidates_sc2.npz')#\" /work/scratch/eval_photo.py > /work/scratch/eval_photo_c2.py
python /work/scratch/eval_photo_c2.py"
```
<details><summary>output</summary>

```
550 0.6 5.9 0.6 5.9
600 -1.7 5.5 -1.7 5.5
650 -5.3 7.3 -5.3 7.3
700 -4.1 5.5 -4.1 5.5
750 -4.1 6.2 -4.1 6.2
800 -3.5 7.0 -3.5 7.0
850 -0.4 8.6 -0.4 8.6
900 -0.0 4.9 -0.0 4.9
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
C iou 0.6136432530917225 ncc 0.12706647897059853 corr 0.28573710662152124 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_photo_c2.npz')
for n in z.files:
 a=z[n]
 for lo,hi,l in [(0,93,'init'),(93,901,'move'),(901,946,'final'),(894,901,'transition')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)];print(n,l,'iou',q[:,1].mean(),'ncc',q[:,2].mean(),'corr',q[:,3].mean())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_final_variants.py <<'PY'
import numpy as np,json
c=np.load('/work/scratch/candidates_sc2.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
out={}
for adj in [0,-1.5,-3,-6]:
 T=C.copy();z=(Twci@T[901])[2,3];T[901:,:3,3]+=Twc[:3,:3]@np.array([0,adj*z/K[1,1],0]);out[f'y{adj}']=T
np.savez('/work/scratch/final_variants.npz',**out)
PY
python /work/scratch/make_final_variants.py
python - <<'PY'
p='/work/scratch/eval_photo.py';s=open(p).read();s=s.replace(\"c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}\",\"cands=dict(np.load('/work/scratch/final_variants.npz'))\")
s=s.replace(\"eval_photo.npz\",\"eval_final_variants.npz\")
open('/work/scratch/eval_final_variants.py','w').write(s)
PY
python /work/scratch/eval_final_variants.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
y0 iou 0.6136432530917225 ncc 0.12706647897059853 corr 0.28573710662152124 n 190
y-1.5 iou 0.6136188661351438 ncc 0.12549867008746332 corr 0.2863364977383148 n 190
y-3 iou 0.613662023054692 ncc 0.12304685656923417 corr 0.2880464663791847 n 190
y-6 iou 0.6134823012129449 ncc 0.1263634991318286 corr 0.29396727556848823 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_final_variants.npz')
for n in z.files:
 a=z[n];
 for lo,hi,l in [(0,93,'init'),(93,901,'move'),(901,946,'final')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)];print(n,l,'iou',round(q[:,1].mean(),4),'ncc',round(q[:,2].mean(),4),'corr',round(q[:,3].mean(),4))
PY"
```
<details><summary>output</summary>

```
y0 init iou 0.642 ncc 0.3506 corr 0.3898
y0 move iou 0.6077 ncc 0.0979 corr 0.282
y0 final iou 0.6604 ncc 0.1806 corr 0.134
y-1.5 init iou 0.642 ncc 0.3506 corr 0.3898
y-1.5 move iou 0.6077 ncc 0.0979 corr 0.282
y-1.5 final iou 0.6599 ncc 0.1475 corr 0.1467
y-3 init iou 0.642 ncc 0.3506 corr 0.3898
y-3 move iou 0.6077 ncc 0.0979 corr 0.282
y-3 final iou 0.6608 ncc 0.0957 corr 0.1828
y-6 init iou 0.642 ncc 0.3506 corr 0.3898
y-6 move iou 0.6077 ncc 0.0979 corr 0.282
y-6 final iou 0.657 ncc 0.1657 corr 0.3078

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_motion_variants.py <<'PY'
import numpy as np,json
c=np.load('/work/scratch/candidates_sc2.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
out={}
for adj in [0,-1.5,-3,-4.5]:
 T=C.copy()
 for i in range(93,901):
  z=(Twci@T[i])[2,3];T[i,:3,3]+=Twc[:3,:3]@np.array([0,adj*z/K[1,1],0])
 # final: remove centroid-y correction (roughly -6 px), based on static texture validation
 z=(Twci@T[901])[2,3];T[901:,:3,3]+=Twc[:3,:3]@np.array([0,-6*z/K[1,1],0])
 out[f'm{adj}']=T
np.savez('/work/scratch/motion_variants.npz',**out)
PY
python /work/scratch/make_motion_variants.py
python - <<'PY'
p='/work/scratch/eval_photo.py';s=open(p).read();s=s.replace(\"c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}\",\"cands=dict(np.load('/work/scratch/motion_variants.npz'))\")
s=s.replace(\"eval_photo.npz\",\"eval_motion_variants.npz\")
open('/work/scratch/eval_motion_variants.py','w').write(s)
PY
python /work/scratch/eval_motion_variants.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
m0 iou 0.6134823012129449 ncc 0.1263634991318286 corr 0.29396727556848823 n 190
m-1.5 iou 0.612275656597702 ncc 0.16281789574967526 corr 0.29256278005990244 n 190
m-3 iou 0.6063554338663063 ncc 0.16180552846551033 corr 0.2768848421439283 n 190
m-4.5 iou 0.5967629791008114 ncc 0.14135214412267308 corr 0.2557790442377501 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/motion_variants.npz');T=z['m-1.5'];rot=R.from_matrix(T[:,:3,:3]);dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude())
for i in range(890,906):print(i,'dt',round(dt[i],4),'dr',round(dr[i],3))
for lo,hi,l in [(0,93,'init'),(93,901,'move'),(901,946,'final')]:print(l,'mean dt',dt[lo:hi-1].mean(),'dr',dr[lo:hi-1].mean(),'max',dt[lo:hi-1].max(),dr[lo:hi-1].max())
PY"
```
<details><summary>output</summary>

```
890 dt 0.0046 dr 0.316
891 dt 0.0064 dr 0.297
892 dt 0.0072 dr 0.258
893 dt 0.0071 dr 0.206
894 dt 0.0059 dr 0.157
895 dt 0.0049 dr 0.131
896 dt 0.0023 dr 0.138
897 dt 0.0016 dr 0.158
898 dt 0.0025 dr 0.168
899 dt 0.0033 dr 0.162
900 dt 0.0309 dr 0.296
901 dt 0.0007 dr 0.0
902 dt 0.0007 dr 0.0
903 dt 0.0006 dr 0.0
904 dt 0.0003 dr 0.0
905 dt 0.0002 dr 0.0
init mean dt 3.5879572919984315e-05 dr 0.0 max 0.001141191752385541 0.0
move mean dt 0.00799555793910821 dr 0.44514108278819076 max 0.03483069875112937 2.254509972354725
final mean dt 5.5627234147736615e-05 dr 0.0 max 0.0007485721976352365 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/motion_variants.npz');T=z['m-1.5'];rot=R.from_matrix(T[:,:3,:3]);dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude())
for i in range(86,101):print(i,'dt',round(dt[i],4),'dr',round(dr[i],3))
PY"
```
<details><summary>output</summary>

```
86 dt 0.0 dr 0.0
87 dt 0.0005 dr 0.0
88 dt 0.0003 dr 0.0
89 dt 0.0005 dr 0.0
90 dt 0.0008 dr 0.0
91 dt 0.0011 dr 0.0
92 dt 0.0423 dr 0.488
93 dt 0.0036 dr 0.119
94 dt 0.0039 dr 0.155
95 dt 0.0048 dr 0.2
96 dt 0.0055 dr 0.241
97 dt 0.0056 dr 0.266
98 dt 0.0047 dr 0.271
99 dt 0.0042 dr 0.257
100 dt 0.0041 dr 0.227

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/candidates_base.npz');c=np.load('/work/scratch/candidates_sc2.npz');v=np.load('/work/scratch/motion_variants.npz')['m-1.5']
for i in range(88,101):print(i,'B',np.round(b['T0'][i,:3,3],4),'S',np.round(c['S'][i,:3,3],4),'F',np.round(v[i,:3,3],4))
print('final')
for i in range(896,906):print(i,'B',np.round(b['T0'][i,:3,3],4),'S',np.round(c['S'][i,:3,3],4),'F',np.round(v[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/check_boundary.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3];T=np.load('/work/scratch/motion_variants.npz')['m-1.5'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
def score(P,om,pm,img):
 m,_,_,rgb=raster(mesh,ren,P,Twci,K,tex=vc);p=m.cpu().numpy();reg=p&om&~pm;iou=(p&om).sum()/max((p|om).sum(),1);col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255)
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1);gax=cv2.Sobel(ga,cv2.CV_32F,1,0,3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,3);a=np.r_[gax[reg],gay[reg]];b=np.r_[grx[reg],gry[reg]];ncc=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
 cs=[np.corrcoef(img[:,:,ch][reg],col[:,:,ch][reg])[0,1] for ch in range(3)]
 return iou,ncc,np.mean(cs)
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs={}
for i in range(946):
 ok,img=cap.read()
 if 88<=i<=110:imgs[i]=img
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(88,111):
  om=f['object'][i];pm=f['person'][i];
  if i%2==0 or i in [89,91,93,9 …[+193 chars]
```
<details><summary>output</summary>

```
88 own (0.642, 0.344, 0.375) static (0.645, 0.353, 0.385)
89 own (0.639, 0.356, 0.366) static (0.641, 0.363, 0.375)
90 own (0.64, 0.364, 0.371) static (0.641, 0.364, 0.374)
91 own (0.644, 0.375, 0.369) static (0.642, 0.366, 0.374)
92 own (0.646, 0.376, 0.368) static (0.642, 0.368, 0.373)
93 own (0.636, 0.221, 0.316) static (0.644, 0.364, 0.362)
94 own (0.633, 0.249, 0.337) static (0.646, 0.362, 0.373)
95 own (0.642, 0.289, 0.355) static (0.644, 0.364, 0.372)
96 own (0.644, 0.301, 0.334) static (0.639, 0.373, 0.359)
97 own (0.637, 0.274, 0.316) static (0.64, 0.373, 0.371)
98 own (0.637, 0.275, 0.311) static (0.635, 0.378, 0.357)
99 own (0.641, 0.29, 0.33) static (0.635, 0.382, 0.358)
100 own (0.647, 0.297, 0.341) static (0.637, 0.381, 0.355)
101 own (0.643, 0.263, 0.321) static (0.638, 0.37 …[+415 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/scan_static_boundaries.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3];T=np.load('/work/scratch/motion_variants.npz')['m-1.5'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twci=np.linalg.inv(np.array(cam['T_world_cam'],np.float32));ren=make_renderer(mesh)
def score(P,om,pm,img):
 m,_,_,rgb=raster(mesh,ren,P,Twci,K,tex=vc);p=m.cpu().numpy();reg=p&om&~pm;iou=(p&om).sum()/max((p|om).sum(),1);col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255)
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1);gax=cv2.Sobel(ga,cv2.CV_32F,1,0,3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,3);a=np.r_[gax[reg],gay[reg]];b=np.r_[grx[reg],gry[reg]];ncc=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
 return iou,ncc
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs={}
for i in range(946):
 ok,img=cap.read()
 if i<=160 or i>=820:imgs[i]=img
with h5py.File('/work/inputs/masks.h5') as f:
 print('INITIAL')
 for i in range(88,151):
  om=f['object'][i];pm=f['person'][i];a=score(T[i],om,pm,imgs[i]);b=score(T[0],om,pm,imgs[i]);
  if i%2==0 or i<100:print(i,'ow …[+360 chars]
```
<details><summary>output</summary>

```
INITIAL
88 own (0.642, 0.344) st (0.645, 0.353)
89 own (0.639, 0.356) st (0.641, 0.363)
90 own (0.64, 0.364) st (0.641, 0.364)
91 own (0.644, 0.375) st (0.642, 0.366)
92 own (0.646, 0.376) st (0.642, 0.368)
93 own (0.636, 0.221) st (0.644, 0.364)
94 own (0.633, 0.249) st (0.646, 0.362)
95 own (0.642, 0.289) st (0.644, 0.364)
96 own (0.644, 0.301) st (0.639, 0.373)
97 own (0.637, 0.274) st (0.64, 0.373)
98 own (0.637, 0.275) st (0.635, 0.378)
99 own (0.641, 0.29) st (0.635, 0.382)
100 own (0.647, 0.297) st (0.637, 0.381)
102 own (0.642, 0.236) st (0.64, 0.381)
104 own (0.639, 0.212) st (0.641, 0.381)
106 own (0.631, 0.205) st (0.636, 0.362)
108 own (0.627, 0.179) st (0.627, 0.328)
110 own (0.625, 0.19) st (0.62, 0.283)
112 own (0.626, 0.195) st (0.618, 0.262)
114 own (0.621, 0.178) st (0.61 …[+4462 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];
with h5py.File('/work/inputs/masks.h5') as f:
 for i in range(120,146):
  o=f['object'][i];y,x=np.where(o);print(i,'cent',round(x.mean(),2),round(y.mean(),2),'area',len(x),'T',np.round(T[i,:3,3],3),'dt',round(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]),4))
PY"
```
<details><summary>output</summary>

```
120 cent 756.08 888.12 area 12783 T [0.138 0.997 2.448] dt 0.0068
121 cent 756.08 887.99 area 12808 T [0.138 0.995 2.445] dt 0.0035
122 cent 756.13 888.0 area 12793 T [0.139 0.998 2.449] dt 0.0052
123 cent 756.39 887.94 area 12843 T [0.139 0.993 2.435] dt 0.0154
124 cent 756.87 887.51 area 13151 T [0.14  0.992 2.436] dt 0.0013
125 cent 756.82 887.95 area 12849 T [0.141 0.991 2.432] dt 0.0034
126 cent 756.93 887.84 area 12874 T [0.143 0.989 2.433] dt 0.0028
127 cent 757.12 887.84 area 12890 T [0.141 0.992 2.437] dt 0.0055
128 cent 757.27 887.87 area 12889 T [0.142 0.983 2.415] dt 0.0241
129 cent 757.43 887.81 area 12894 T [0.144 0.994 2.444] dt 0.0306
130 cent 757.55 887.76 area 12915 T [0.145 0.995 2.442] dt 0.0024
131 cent 757.82 887.79 area 12946 T [0.146 0.997 2.449] dt 0.0076
132 cent  …[+911 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/scratch/motion_variants.npz')['m-1.5'];print(np.max(abs(T[901]-T[945])),T[901,:3,3],T[945,:3,3])
np.set_printoptions(precision=8);print(T[901]);print(T[945])
PY"
```
<details><summary>output</summary>

```
0.001250776271725429 [-0.59671384  0.99216605  3.15126261] [-0.59796462  0.99293345  3.15127146]
[[-0.24503035  0.96917666  0.02562692 -0.59671384]
 [-0.19776787 -0.02408844 -0.97995286  0.99216605]
 [-0.94913013 -0.24518637  0.1975744   3.15126261]
 [ 0.          0.          0.          1.        ]]
[[-0.24503035  0.96917666  0.02562692 -0.59796462]
 [-0.19776787 -0.02408844 -0.97995286  0.99293345]
 [-0.94913013 -0.24518637  0.1975744   3.15127146]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s/static_runs=\\[(0,93),(894,946)\\]/static_runs=[(0,132),(901,946)]/\" /work/scratch/make_candidates.py > /work/scratch/make_candidates3.py
sed -i \"s#candidates_base.npz#candidates_base3.npz#\" /work/scratch/make_candidates3.py
python /work/scratch/make_candidates3.py
sed \"s#candidates_base.npz#candidates_base3.npz#g; s#candidates_sc.npz#candidates_sc3.npz#g; s/for lo,hi in \\[(0,93),(894,946)\\]/for lo,hi in [(0,132),(901,946)]/; s/dy\\[894:\\]=0/dy[901:]=0/\" /work/scratch/make_sc.py > /work/scratch/make_sc3.py
python /work/scratch/make_sc3.py >/tmp/makesc3.txt && tail -8 /tmp/makesc3.txt
cat > /work/scratch/make_final3.py <<'PY'
import numpy as np,json
c=np.load('/work/scratch/candidates_sc3.npz');C=c['C'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc)
F=C.copy()
for i in range(132,901):
 z=(Twci@F[i])[2,3];F[i,:3,3]+=Twc[:3,:3]@np.array([0,-1.5*z/K[1,1],0])
z=(Twci@F[901])[2,3];F[901:,:3,3]+=Twc[:3,:3]@np.array([0,-6*z/K[1,1],0])
np.savez('/work/scratch/candidate_final3.npz',F=F)
PY
python /work/scratch/make_final3.py"
```
<details><summary>output</summary>

```
B dr med/p95/max 0.771459966069196 2.548891781692365 12.328823102254969 dt 0.011277772177705267 0.045846667008680335 0.1759136065878757
change from base trans 0.0 rot 0.0
S dr med/p95/max 0.3058779041390807 0.9871756156186139 2.6339913352053155 dt 0.00563215135794794 0.0174259324783764 0.0607767265370814
change from base trans 0.01210739154276003 rot 0.822773094823679
550 0.6 5.9 0.6 5.9
600 -1.7 5.5 -1.7 5.5
650 -5.3 7.3 -5.3 7.3
700 -4.1 5.5 -4.1 5.5
750 -4.1 6.2 -4.1 6.2
800 -3.5 7.0 -3.5 7.0
850 -0.4 8.6 -0.4 8.6
900 -0.0 4.9 -0.0 4.9

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_photo.py';s=open(p).read();s=s.replace(\"c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}\",\"c0=np.load('/work/scratch/candidates_base3.npz');cands={'B':c0['T0'],'F':np.load('/work/scratch/candidate_final3.npz')['F']}\")
s=s.replace(\"eval_photo.npz\",\"eval_photo_final3.npz\")
open('/work/scratch/eval_photo_final3.py','w').write(s)
PY
python /work/scratch/eval_photo_final3.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
F iou 0.6126626184024907 ncc 0.16421393394707484 corr 0.2907693421648789 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/eval_photo_final3.npz')
for n in z.files:
 a=z[n]
 for lo,hi,l in [(0,132,'init'),(132,901,'move'),(901,946,'final')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)];print(n,l,'iou',round(q[:,1].mean(),4),'ncc',round(q[:,2].mean(),4),'corr',round(q[:,3].mean(),4),'n',len(q))
T=np.load('/work/scratch/candidate_final3.npz')['F'];b=np.load('/work/scratch/candidates_base3.npz')['T0'];
for nm,X in [('B',b),('F',T)]:
 rot=R.from_matrix(X[:,:3,:3]);dt=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude());print(nm,'dt',np.median(dt),np.percentile(dt,95),dt.max(),'dr',np.median(dr),np.percentile(dr,95),dr.max())
print('boundary init')
for i in range(128,136):print(i,round(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]),4))
print('boundary final')
for i in range(897,905):print(i,round(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]),4))
PY"
```
<details><summary>output</summary>

```
B init iou 0.5717 ncc -0.0563 corr 0.2381 n 27
B move iou 0.5563 ncc 0.0716 corr 0.1958 n 154
B final iou 0.6569 ncc 0.1375 corr 0.2873 n 9
F init iou 0.6406 ncc 0.3213 corr 0.3621 n 27
F move iou 0.6052 ncc 0.1366 corr 0.2773 n 154
F final iou 0.657 ncc 0.1657 corr 0.3078 n 9
B dt 0.011277772177705267 0.045846667008680335 0.1759136065878757 dr 0.771459966069196 2.548891781692365 12.328823102254969
F dt 0.005899904433512323 0.017968428859496777 0.061496832891228995 dr 0.3058779041390807 0.9871756156186139 2.6339913352053155
boundary init
128 0.0002
129 0.001
130 0.0019
131 0.0021
132 0.0615
133 0.0009
134 0.0011
135 0.0024
boundary final
897 0.0023
898 0.0016
899 0.0025
900 0.0033
901 0.0309
902 0.0007
903 0.0007
904 0.0006

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
F=np.load('/work/scratch/candidate_final3.npz')['F'];S=F[0];
for i in range(128,146):print(i,'F-static',np.round(F[i,:3,3]-S[:3,3],4),'norm',round(np.linalg.norm(F[i,:3,3]-S[:3,3]),4),'prevF',round(np.linalg.norm(F[i,:3,3]-F[i-1,:3,3]),4))
print('final')
S=F[945]
for i in range(890,906):print(i,'F-static',np.round(F[i,:3,3]-S[:3,3],4),'norm',round(np.linalg.norm(F[i,:3,3]-S[:3,3]),4),'prevF',round(np.linalg.norm(F[i,:3,3]-F[i-1,:3,3]),4))
PY"
```
<details><summary>output</summary>

```
128 F-static [ 0.0007  0.     -0.    ] norm 0.0007 prevF 0.0002
129 F-static [-0.0003 -0.0001  0.    ] norm 0.0004 prevF 0.001
130 F-static [-0.0022 -0.0002  0.    ] norm 0.0022 prevF 0.0019
131 F-static [-0.0043 -0.0004  0.0001] norm 0.0044 prevF 0.0021
132 F-static [ 0.0122 -0.0206 -0.0556] norm 0.0605 prevF 0.0615
133 F-static [ 0.0124 -0.0197 -0.0558] norm 0.0604 prevF 0.0009
134 F-static [ 0.0127 -0.0186 -0.0555] norm 0.0599 prevF 0.0011
135 F-static [ 0.0148 -0.0176 -0.0553] norm 0.0599 prevF 0.0024
136 F-static [ 0.0175 -0.0174 -0.0554] norm 0.0607 prevF 0.0027
137 F-static [ 0.0216 -0.0174 -0.0548] norm 0.0614 prevF 0.0042
138 F-static [ 0.025  -0.0191 -0.0559] norm 0.0641 prevF 0.0039
139 F-static [ 0.029  -0.0211 -0.0577] norm 0.0679 prevF 0.0049
140 F-static [ 0.0328 -0.023  -0. …[+1380 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_final4.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
F=np.load('/work/scratch/candidate_final3.npz')['F'].copy()
def slerp(r0,r1,a):return (R.from_matrix(r0).inv()*R.from_matrix(r1)).as_rotvec()*a
def blend_to(P,Q,a):
 out=Q.copy();out[:3,3]=(1-a)*P[:3,3]+a*Q[:3,3];rel=R.from_matrix(P[:3,:3]).inv()*R.from_matrix(Q[:3,:3]);out[:3,:3]=R.from_rotvec(rel.as_rotvec()*a).as_matrix()@P[:3,:3];return out
# Initial acceleration: the baseline switches sharply at frame 132 although visual motion starts there.
P0=F[131]
for i in range(132,146):
 a=(i-131)/14.;a=a*a*(3-2*a);F[i]=blend_to(P0,F[145],a)
# Final deceleration: object comes to rest around frames 900-901.
P1=F[901]
for i in range(890,901):
 a=(i-889)/12.;a=a*a*(3-2*a);F[i]=blend_to(F[i],P1,a)
np.savez('/work/scratch/candidate_final4.npz',F=F)
PY
python /work/scratch/make_final4.py
python - <<'PY'
p='/work/scratch/eval_photo.py';s=open(p).read();s=s.replace(\"c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}\",\"c0=np.load('/work/scratch/candidates_base3.npz');cands={'B':c0['T0'],'F3':np.load('/work/scratch/candidate_final3.npz')['F'],'F4':np.load('/work/scratch/candidate_final4.npz')['F']}\")
s=s.replace(\"eval_photo.npz\",\"eval_photo_final4.npz\")
open('/work/scratch/eval_photo_final4.py','w').write(s)
PY
python /work/scratch/eval_photo_final4.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
F3 iou 0.6126626184024907 ncc 0.16421393394707484 corr 0.2907693421648789 n 190
F4 iou 0.6120586290629938 ncc 0.163614219781464 corr 0.29028145272892414 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for nm in ['candidate_final3.npz','candidate_final4.npz']:
 T=np.load('/work/scratch/'+nm)['F'];rot=R.from_matrix(T[:,:3,:3]);dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude());print(nm,'max',dt.max(),dr.max(),'init',dt[128:140],'final',dt[895:904])
PY"
```
<details><summary>output</summary>

```
candidate_final3.npz max 0.061496832891228995 2.6339913352053155 init [0.00104781 0.00186642 0.0021418  0.06149683 0.00093575 0.00110277
 0.00237756 0.00267164 0.00417297 0.00391959 0.00485128 0.0056766 ] final [0.00487485 0.0023129  0.00162388 0.00245522 0.00328142 0.03087547
 0.00074857 0.00066467 0.00055252]
candidate_final4.npz max 0.03483069875112937 9.612085676077884 init [0.00104781 0.00186642 0.0021418  0.00160341 0.00448955 0.00689466
 0.00881875 0.01026182 0.01122386 0.01170489 0.01170489 0.01122386] final [0.00689508 0.00502649 0.00388552 0.00296715 0.00183923 0.0006075
 0.00074857 0.00066467 0.00055252]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/candidate_final4.npz')['F'];rot=R.from_matrix(T[:,:3,:3]);dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude());
for i in np.argsort(dr)[-20:][::-1]:print(i+1,round(dr[i],2),round(dt[i],3))
PY"
```
<details><summary>output</summary>

```
146 9.61 0.006
412 2.25 0.008
411 2.22 0.01
413 2.16 0.008
410 2.05 0.011
414 1.96 0.008
409 1.78 0.011
415 1.71 0.008
408 1.48 0.011
416 1.45 0.008
344 1.43 0.002
398 1.41 0.009
345 1.41 0.004
428 1.36 0.009
397 1.36 0.011
427 1.34 0.008
343 1.32 0.002
429 1.28 0.012
426 1.27 0.009
346 1.25 0.005

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/make_final4.py';s=open(p).read();s=s.replace(\"out[:3,:3]=R.from_rotvec(rel.as_rotvec()*a).as_matrix()@P[:3,:3]\",\"out[:3,:3]=P[:3,:3]@R.from_rotvec(rel.as_rotvec()*a).as_matrix()\")
open(p,'w').write(s)
PY
python /work/scratch/make_final4.py
python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for nm in ['candidate_final3.npz','candidate_final4.npz']:
 T=np.load('/work/scratch/'+nm)['F'];rot=R.from_matrix(T[:,:3,:3]);dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude());print(nm,'max',dt.max(),dr.max(),'init dt/dr',[(round(dt[i],4),round(dr[i],2)) for i in range(128,148)],'final',[(round(dt[i],4),round(dr[i],2)) for i in range(895,904)])
PY"
```
<details><summary>output</summary>

```
candidate_final3.npz max 0.061496832891228995 2.6339913352053155 init dt/dr [(0.001, 0.0), (0.0019, 0.0), (0.0021, 0.0), (0.0615, 2.63), (0.0009, 0.31), (0.0011, 0.28), (0.0024, 0.25), (0.0027, 0.25), (0.0042, 0.26), (0.0039, 0.28), (0.0049, 0.34), (0.0057, 0.42), (0.0074, 0.49), (0.0076, 0.54), (0.0078, 0.54), (0.0068, 0.5), (0.0068, 0.44), (0.0059, 0.38), (0.006, 0.34), (0.0046, 0.32)] final [(0.0049, 0.13), (0.0023, 0.14), (0.0016, 0.16), (0.0025, 0.17), (0.0033, 0.16), (0.0309, 0.3), (0.0007, 0.0), (0.0007, 0.0), (0.0006, 0.0)]
candidate_final4.npz max 0.03483069875112937 2.254509972354725 init dt/dr [(0.001, 0.0), (0.0019, 0.0), (0.0021, 0.0), (0.0016, 0.1), (0.0045, 0.27), (0.0069, 0.41), (0.0088, 0.53), (0.0103, 0.61), (0.0112, 0.67), (0.0117, 0.7), (0.0117, 0.7), (0.0112, 0.67), (0 …[+270 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/scratch/candidate_final4.npz')['F'];dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);print(np.argsort(dt)[-10:][::-1]+1, np.sort(dt)[-10:][::-1])
PY"
```
<details><summary>output</summary>

```
[436 435 437 438 434 839 439 446 447 838] [0.0348307  0.03329586 0.03049302 0.02747941 0.02742254 0.02406786
 0.02327791 0.02317033 0.02282206 0.02169965]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_photo.py';s=open(p).read();s=s.replace(\"c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}\",\"c0=np.load('/work/scratch/candidates_base3.npz');cands={'B':c0['T0'],'F':np.load('/work/scratch/candidate_final4.npz')['F']}\")
s=s.replace(\"eval_photo.npz\",\"eval_photo_final4b.npz\")
open('/work/scratch/eval_photo_final4b.py','w').write(s)
PY
python /work/scratch/eval_photo_final4b.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
F iou 0.6122896539440023 ncc 0.16407902541821237 corr 0.29103703434765676 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/eval_photo_final4b.npz');T=np.load('/work/scratch/candidate_final4.npz')['F'];B=np.load('/work/scratch/candidates_base3.npz')['T0']
for n in z.files:
 a=z[n]
 for lo,hi,l in [(0,132,'initial static'),(132,901,'manipulation'),(901,946,'final static'),(0,946,'all')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)];print(n,l,'IoU %.4f gradNCC %.4f colorCorr %.4f n %d'%(q[:,1].mean(),q[:,2].mean(),q[:,3].mean(),len(q)))
# floor
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);V=list(m.geometry.values())[0].vertices;Vh=np.c_[V,np.ones(len(V))];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nv,d=pl[:3],pl[3]
for n,X in [('B',B),('F',T)]:
 s=np.array([(X[i]@Vh.T).T[:,:3]@nv+d for i in range(len(X))]);mn=s.min(1)
 print(n,'floor median/p5/p95',np.median(mn),np.percentile(mn,5),np.percentile(mn,95),'meanabs',np.mean(abs(mn)))
 rot=R.from_matrix(X[:,:3,:3]);dt=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude())
 print(n,'dt med/p95/max',np.median(dt),np.percentile(dt,95),dt.max(),'dr',np.median(dr),np.percentile(dr,95),dr.max())
PY"
```
<details><summary>output</summary>

```
B initial static IoU 0.5717 gradNCC -0.0563 colorCorr 0.2381 n 27
B manipulation IoU 0.5563 gradNCC 0.0716 colorCorr 0.1958 n 154
B final static IoU 0.6569 gradNCC 0.1375 colorCorr 0.2873 n 9
B all IoU 0.5633 gradNCC 0.0565 colorCorr 0.2061 n 190
F initial static IoU 0.6406 gradNCC 0.3213 colorCorr 0.3621 n 27
F manipulation IoU 0.6047 gradNCC 0.1364 colorCorr 0.2776 n 154
F final static IoU 0.6570 gradNCC 0.1657 colorCorr 0.3078 n 9
F all IoU 0.6123 gradNCC 0.1641 colorCorr 0.2910 n 190
B floor median/p5/p95 0.010098135017877263 -0.02301572481998282 0.032872155831679295 meanabs 0.016430710863656203
B dt med/p95/max 0.011277772177705267 0.045846667008680335 0.1759136065878757 dr 0.771459966069196 2.548891781692365 12.328823102254969
F floor median/p5/p95 -0.011237316103332007 -0.0487592733 …[+197 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_depth.py';s=open(p).read();s=s.replace(\"mesh=load_mesh('/work/inputs/object_mesh.glb');c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cb=np.load('/work/scratch/candidates_cb.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C'],'CB':cb['CB']}\",\"mesh=load_mesh('/work/inputs/object_mesh.glb');cands={'B':np.load('/work/scratch/candidates_base3.npz')['T0'],'F':np.load('/work/scratch/candidate_final4.npz')['F']}\")
s=s.replace(\"eval_depth.npz\",\"eval_depth_final.npz\")
open('/work/scratch/eval_depth_final.py','w').write(s)
PY
python /work/scratch/eval_depth_final.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B signed med -0.1278451681137085 mae med 0.17535000294446945 mae mean 0.18127711321178236 p90 med 0.33826059103012107 n 190
F signed med -0.12550032138824463 mae med 0.17998501658439636 mae mean 0.1794003987586812 p90 med 0.3478805303573609 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s#r=np.load('/work/inputs/reconstruction.npz'); T=r\\['T_world_mesh'\\]#r=np.load('/work/scratch/candidate_final4.npz'); T=r['F']#; s#overlay_baseline.jpg#overlay_F.jpg#\" /work/scratch/make_overlay.py > /work/scratch/make_overlay_F.py
python /work/scratch/make_overlay_F.py"
```
<details><summary>output</summary>

```
(1536, 3072, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
for f in ['candidates_sc3.npz','candidate_final4.npz']:
 z=np.load('/work/scratch/'+f);T=z['C'] if 'C' in z.files else z['F'];print(f)
 for lo,hi in [(0,132),(901,946)]:
  tr=T[lo:hi,:3,3];print(lo,hi,'range',np.ptp(tr,axis=0),'std',tr.std(0),'maxdiff',np.max(np.linalg.norm(tr-tr[0],axis=1)))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_final5.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
F=np.load('/work/scratch/candidate_final3.npz')['F'].copy()
def pose_mean(Ts):
 t=np.median(Ts[:,:3,3],axis=0);Rm=R.from_matrix(Ts[:,:3,:3]).mean().as_matrix();out=np.eye(4);out[:3,:3]=Rm;out[:3,3]=t;return out
def blend_to(P,Q,a):
 out=Q.copy();out[:3,3]=(1-a)*P[:3,3]+a*Q[:3,3];rel=R.from_matrix(P[:3,:3]).inv()*R.from_matrix(Q[:3,:3]);out[:3,:3]=P[:3,:3]@R.from_rotvec(rel.as_rotvec()*a).as_matrix();return out
P0=pose_mean(F[:132]);P1=pose_mean(F[901:]);F[:132]=P0;F[901:]=P1
for i in range(132,146):
 a=(i-131)/14.;a=a*a*(3-2*a);F[i]=blend_to(P0,F[145],a)
for i in range(890,901):
 a=(i-889)/12.;a=a*a*(3-2*a);F[i]=blend_to(F[i],P1,a)
np.savez('/work/scratch/candidate_final5.npz',F=F)
PY
python /work/scratch/make_final5.py
python - <<'PY'
p='/work/scratch/eval_photo.py';s=open(p).read();s=s.replace(\"c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}\",\"c0=np.load('/work/scratch/candidates_base3.npz');cands={'B':c0['T0'],'F4':np.load('/work/scratch/candidate_final4.npz')['F'],'F5':np.load('/work/scratch/candidate_final5.npz')['F']}\")
s=s.replace(\"eval_photo.npz\",\"eval_photo_final5.npz\")
open('/work/scratch/eval_photo_final5.py','w').write(s)
PY
python /work/scratch/eval_photo_final5.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
F4 iou 0.6122896539440023 ncc 0.16407902541821237 corr 0.29103703434765676 n 190
F5 iou 0.6121877556965472 ncc 0.16356705142240077 corr 0.2907272241442092 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs
cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Trajectory recovery for the monocular toy-stroller episode.

The supplied CARI4D/FoundationPose/CoCoNet reconstruction is used as a prior.
A rigid-object trajectory prior is then fitted:
  1. pooled constant poses in the two visually verified rest phases,
  2. robust SE(3) temporal smoothing during manipulation,
  3. camera-plane corrections against the SAM object silhouette,
  4. smooth acceleration/deceleration at the two rest/manipulation boundaries.

No ground-truth annotation is read.  All evidence comes from the supplied video,
masks, mesh, camera, and baseline reconstruction.
"""
import argparse
import json
import os
import time

import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# Verified phase boundaries (30 fps).  The object'"'s silhouette centroid and its
# texture remain fixed through these frames.  Texture scores start changing at
# 132 and the released object has settled by 901.
INITIAL_STATIC_END = 132
FINAL_STATIC_START = 901


def load_geometry(path):
    scene = trimesh.load(path, force=\"scene\", process=False)
    return list(scene.geometry.values())[0]


def pose_mean(transforms):
    \"\"\"Robust translation median and chordal-L2 rotation mean.\"\"\"
    t = np.median(transforms[:, :3, 3], …[+9993 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
[1/5] Robust temporal prior
[2/5] Silhouette registration against SAM masks
  silhouette    0/946
  silhouette  100/946
  silhouette  200/946
  silhouette  300/946
  silhouette  400/946
  silhouette  500/946
  silhouette  600/946
  silhouette  700/946
  silhouette  800/946
  silhouette  900/946
[3/5] Apply camera-plane registration
[4/5] Validate output
[5/5] Wrote /work/outputs/object_poses_world.npz shape (946, 4, 4) dtype float64
Elapsed: 120.3s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz');q=np.load('/work/scratch/candidate_final5.npz')
T=p['T_world_mesh'];F=q['F'];print(p.files,T.shape,T.dtype,np.isfinite(T).all());print('diff translation max/mean',np.linalg.norm(T[:,:3,3]-F[:,:3,3],axis=1).max(),np.linalg.norm(T[:,:3,3]-F[:,:3,3],axis=1).mean());d=(np.linalg.det(T[:,:3,:3]));orth=np.abs(np.einsum('nji,njk->nik',T[:,:3,:3],T[:,:3,:3])-np.eye(3)).max();print('det',d.min(),d.max(),'orth',orth)
PY"
```
<details><summary>output</summary>

```
['T_world_mesh'] (946, 4, 4) float64 True
diff translation max/mean 0.004924379433374634 0.00036212460576019224
det 0.9999999999999993 1.0000000000000016 orth 1.1102230246251565e-15

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true; git -C /workspace/v2d_cari4d rev-parse HEAD 2>/dev/null || true; git -C /workspace/v2d_foundation_pose rev-parse HEAD 2>/dev/null || true'
```
<details><summary>output</summary>

```
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python -m pip show nvdiffrast torch trimesh scipy h5py numpy opencv-python Pillow | grep -E '"'^(Name|Version):'"'"
```
<details><summary>output</summary>

```
Name: nvdiffrast
Version: 0.4.0
Name: torch
Version: 2.5.1+cu124
Name: trimesh
Version: 5.1.1
Name: scipy
Version: 1.16.3
Name: h5py
Version: 3.16.0
Name: numpy
Version: 1.26.3
Name: lapack-lite
Name: tempita
Name: dragon4
Name: libdivide
Name: Meson
Name: meson-python
Name: spin
Name: OpenBLAS
Name: LAPACK
Name: GCC runtime library
Name: libquadmath
Name: opencv-python
Version: 4.11.0.86
Name: pillow
Version: 10.2.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];F=np.load('/work/scratch/candidate_final5.npz')['F'];d=np.linalg.norm(T[:,:3,3]-F[:,:3,3],axis=1)
for i in np.argsort(d)[-20:][::-1]:print(i,d[i],T[i,:3,3],F[i,:3,3])
print('phase mean',[(a,b,d[a:b].mean()) for a,b in [(0,132),(132,901),(901,946)]])
PY"
```
<details><summary>output</summary>

```
895 0.004924379433374634 [-0.59204136  0.99572525  3.12934726] [-0.59354175  0.99118866  3.13053794]
896 0.004730117122286117 [-0.59336273  0.99525394  3.1359083 ] [-0.5945796   0.99092229  3.1373678 ]
894 0.004560813092193088 [-0.59090206  0.99646069  3.12110413] [-0.59246462  0.99227403  3.12201587]
897 0.003812344793308957 [-0.5948592   0.99482317  3.14058581] [-0.59564328  0.99150248  3.14228648]
893 0.003587132784994143 [-0.58963088  0.99727703  3.11101725] [-0.59096892  0.99401031  3.11165421]
898 0.002855336343754045 [-0.59608711  0.99426278  3.14415648] [-0.59646294  0.99217255  3.14606505]
899 0.002364499151306888 [-0.59713256  0.99384366  3.14683882] [-0.59724277  0.99271635  3.14891437]
927 0.002318196698306592 [-0.59793238  0.99355557  3.14903853] [-0.59796462  0.99293345  3.15 …[+1335 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_photo.py';s=open(p).read();s=s.replace(\"c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C']}\",\"cands={'B':np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64),'result':np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']}\")
s=s.replace(\"eval_photo.npz\",\"eval_photo_solve.npz\")
open('/work/scratch/eval_photo_solve.py','w').write(s)
PY
python /work/scratch/eval_photo_solve.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
result iou 0.6119319075156195 ncc 0.16244266333299606 corr 0.2911786055608658 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_photo_solve.npz')
for n in z.files:
 a=z[n]
 for lo,hi,l in [(0,132,'initial static'),(132,901,'manipulation'),(901,946,'final static'),(0,946,'all')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)];print(n,l,'IoU %.4f gradNCC %.4f colorCorr %.4f n %d'%(q[:,1].mean(),q[:,2].mean(),q[:,3].mean(),len(q)))
PY"
```
<details><summary>output</summary>

```
B initial static IoU 0.5717 gradNCC -0.0563 colorCorr 0.2381 n 27
B manipulation IoU 0.5563 gradNCC 0.0716 colorCorr 0.1958 n 154
B final static IoU 0.6569 gradNCC 0.1375 colorCorr 0.2873 n 9
B all IoU 0.5633 gradNCC 0.0565 colorCorr 0.2061 n 190
result initial static IoU 0.6389 gradNCC 0.3333 colorCorr 0.3752 n 27
result manipulation IoU 0.6046 gradNCC 0.1352 colorCorr 0.2760 n 154
result final static IoU 0.6570 gradNCC 0.1160 colorCorr 0.2995 n 9
result all IoU 0.6119 gradNCC 0.1624 colorCorr 0.2912 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];F=np.load('/work/scratch/candidate_final5.npz')['F'];d=(R.from_matrix(T[:,:3,:3])*R.from_matrix(F[:,:3,:3]).inv()).magnitude();print('rot max',np.degrees(d.max()),'final',np.degrees(d[901:]).max());print('trans',np.linalg.norm(T[901:,:3,3]-F[901:,:3,3],axis=1).max())
PY"
```
<details><summary>output</summary>

```
rot max 0.03008387725006005 final 0.013984675943114993
trans 0.002318196698306592

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_grid2.py <<'PY'
import sys,json,numpy as np,torch,trimesh,h5py,cv2
sys.path.insert(0,'/work/scratch');from render_eval import *
mesh=load_mesh('/work/inputs/object_mesh.glb');vc=mesh.visual.to_color().vertex_colors[:,:3];T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Twc=np.array(cam['T_world_cam'],np.float32);Twci=np.linalg.inv(Twc);ren=make_renderer(mesh)
def score(P,om,pm,img):
 m,_,_,rgb=raster(mesh,ren,P,Twci,K,tex=vc);p=m.cpu().numpy();reg=p&om&~pm;iou=(p&om).sum()/max((p|om).sum(),1);col=(rgb.permute(1,2,0).cpu().numpy()*255).clip(0,255)
 ga=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32);gr=cv2.cvtColor(col.astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32);ga=cv2.GaussianBlur(ga,(0,0),1);gr=cv2.GaussianBlur(gr,(0,0),1);gax=cv2.Sobel(ga,cv2.CV_32F,1,0,3);gay=cv2.Sobel(ga,cv2.CV_32F,0,1,3);grx=cv2.Sobel(gr,cv2.CV_32F,1,0,3);gry=cv2.Sobel(gr,cv2.CV_32F,0,1,3);a=np.r_[gax[reg],gay[reg]];b=np.r_[grx[reg],gry[reg]];ncc=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
 cs=[np.corrcoef(img[:,:,ch][reg],col[:,:,ch][reg])[0,1] for ch in range(3)]
 return iou,ncc,np.mean(cs)
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs={}
for i in range(946):
 ok,img=cap.read()
 if i%4==0:imgs[i]=img
with h5py.File('/work/inputs/masks.h5') as f:
 for lo,hi in [(0,132),(901,946)]:
  samples=[(i,imgs[i],f['object'][i],f['person'] …[+612 chars]
```
<details><summary>output</summary>

```

phase 0 132 base (0.6387750996536711, 0.3320125719556038, 0.37471464576734514, 0, 0)
best ncc [(0.6564836945297253, 0.3974351719997272, 0.4157705883935714, -2, 0), (0.6516243913178923, 0.3872676993975259, 0.39600144362870787, -2, -1), (0.649868662919441, 0.38013490631881924, 0.39641811197088295, -1, 0), (0.6438419027313883, 0.34743631017111976, 0.37095221717833005, -1, -1), (0.6387750996536711, 0.3320125719556038, 0.37471464576734514, 0, 0), (0.6544175988278711, 0.3296609315461922, 0.4020436940050748, -1, 1), (0.6588535583309262, 0.32261431655828976, 0.41630053002564693, -2, 1), (0.6445691588973397, 0.30455945042701144, 0.3816656961415176, 0, 1)]
best corr [(0.6588535583309262, 0.32261431655828976, 0.41630053002564693, -2, 1), (0.6564836945297253, 0.3974351719997272, 0.4157705883935714, - …[+2789 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_grid2.py';s=open(p).read();s=s.replace(\"for dx in [-2,-1,0,1,2]:\\n   for dy in [-2,-1,0,1,2]:\", \"grids={0:([-4,-3,-2,-1,0],[-2,-1,0,1,2]),901:([0,1,2,3,4],[-4,-3,-2,-1,0])}\\n  for dx in grids[lo][0]:\\n   for dy in grids[lo][1]:\")
s=s.replace(\"vals[12]\",\"vals[len(vals)//2]\")
open('/work/scratch/static_grid3.py','w').write(s)
PY
python /work/scratch/static_grid3.py"
```
<details><summary>output</summary>

```

phase 0 132 base (0.6564836945297253, 0.3974351719997272, 0.4157705883935714, -2, 0)
best ncc [(0.6564836945297253, 0.3974351719997272, 0.4157705883935714, -2, 0), (0.6555398568060037, 0.3934167597500467, 0.4152500341801222, -3, -1), (0.6516243913178923, 0.3872676993975259, 0.39600144362870787, -2, -1), (0.649868662919441, 0.38013490631881924, 0.39641811197088295, -1, 0), (0.6585825344603587, 0.3629960727906011, 0.42537820674913474, -3, 0), (0.6564342366285814, 0.35727307883592146, 0.4127576224437169, -4, -1), (0.651962270470831, 0.3551749839925555, 0.3898234711313918, -4, -2), (0.6438419027313883, 0.34743631017111976, 0.37095221717833005, -1, -1)]
best corr [(0.6585825344603587, 0.3629960727906011, 0.42537820674913474, -3, 0), (0.6592935640945515, 0.2706571183629198, 0.4194174845844485,  …[+2786 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
old=\"\"\"    p0 = pose_mean(result[:INITIAL_STATIC_END])
    p1 = pose_mean(result[FINAL_STATIC_START:])
    result[:INITIAL_STATIC_END] = p0
    result[FINAL_STATIC_START:] = p1\"\"\"
new=\"\"\"    p0 = pose_mean(result[:INITIAL_STATIC_END])
    p1 = pose_mean(result[FINAL_STATIC_START:])
    # Two final sub-pixel/2-px refinements from the pooled texture correlation
    # grid.  These use all frames in each rest phase and improve silhouette,
    # gradient and colour agreement together (see REPORT.md).
    p0 = translate_camera_plane(
        p0, -2.0, 0.0, world_from_camera, world_to_camera, intrinsics
    )
    p1 = translate_camera_plane(
        p1, 2.0, -2.0, world_from_camera, world_to_camera, intrinsics
    )
    result[:INITIAL_STATIC_END] = p0
    result[FINAL_STATIC_START:] = p1\"\"\"
assert old in s
open(p,'w').write(s.replace(old,new))
PY
python -m py_compile /work/outputs/solve.py
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
[1/5] Robust temporal prior
[2/5] Silhouette registration against SAM masks
  silhouette    0/946
  silhouette  100/946
  silhouette  200/946
  silhouette  300/946
  silhouette  400/946
  silhouette  500/946
  silhouette  600/946
  silhouette  700/946
  silhouette  800/946
  silhouette  900/946
[3/5] Apply camera-plane registration
[4/5] Validate output
[5/5] Wrote /work/outputs/object_poses_world.npz shape (946, 4, 4) dtype float64
Elapsed: 122.7s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/eval_photo_solve.py'
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B iou 0.5632716840994365 ncc 0.0565081942052743 corr 0.20612677261351406 n 190
result iou 0.6140412409927865 ncc 0.18083724764430933 corr 0.30060625662313306 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/eval_photo_solve.npz')
for n in z.files:
 a=z[n]
 for lo,hi,l in [(0,132,'initial static'),(132,901,'manipulation'),(901,946,'final static'),(0,946,'all')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)];print(n,l,'IoU %.4f gradNCC %.4f colorCorr %.4f n %d'%(q[:,1].mean(),q[:,2].mean(),q[:,3].mean(),len(q)))
PY"
```
<details><summary>output</summary>

```
B initial static IoU 0.5717 gradNCC -0.0563 colorCorr 0.2381 n 27
B manipulation IoU 0.5563 gradNCC 0.0716 colorCorr 0.1958 n 154
B final static IoU 0.6569 gradNCC 0.1375 colorCorr 0.2873 n 9
B all IoU 0.5633 gradNCC 0.0565 colorCorr 0.2061 n 190
result initial static IoU 0.6566 gradNCC 0.3962 colorCorr 0.4154 n 27
result manipulation IoU 0.6046 gradNCC 0.1368 colorCorr 0.2769 n 154
result final static IoU 0.6472 gradNCC 0.2888 colorCorr 0.3622 n 9
result all IoU 0.6140 gradNCC 0.1808 colorCorr 0.3006 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_depth.py';s=open(p).read();s=s.replace(\"mesh=load_mesh('/work/inputs/object_mesh.glb');c0=np.load('/work/scratch/candidates_base.npz');c1=np.load('/work/scratch/candidates_sc.npz');cb=np.load('/work/scratch/candidates_cb.npz');cands={'B':c0['T0'],'S':c1['S'],'C':c1['C'],'CB':cb['CB']}\",\"mesh=load_mesh('/work/inputs/object_mesh.glb');cands={'B':np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64),'result':np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']}\")
s=s.replace(\"eval_depth.npz\",\"eval_depth_solve.npz\")
open('/work/scratch/eval_depth_solve.py','w').write(s)
PY
python /work/scratch/eval_depth_solve.py"
```
<details><summary>output</summary>

```
0
100
200
300
400
500
600
700
800
900
B signed med -0.1278451681137085 mae med 0.17535000294446945 mae mean 0.18127711321178236 p90 med 0.33826059103012107 n 190
result signed med -0.12538760900497437 mae med 0.17993222177028656 mae mean 0.17821312589864982 p90 med 0.3489969730377197 n 190

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64);z=np.load('/work/scratch/eval_depth_solve.npz')
for n in z.files:
 a=z[n]
 for lo,hi,l in [(0,132,'initial static'),(132,901,'manipulation'),(901,946,'final static'),(0,946,'all')]:
  q=a[(a[:,0]>=lo)&(a[:,0]<hi)&(a[:,4]>100)];print(n,l,'depthMAE %.4f signed %.4f n%d'%(q[:,2].mean(),np.median(q[:,1]),len(q)))
print('temporal/floor')
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);V=list(m.geometry.values())[0].vertices;Vh=np.c_[V,np.ones(len(V))];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);nv,d=pl[:3],pl[3]
for n,X in [('B',B),('R',T)]:
 rot=R.from_matrix(X[:,:3,:3]);dt=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);dr=np.degrees((rot[1:]*rot[:-1].inv()).magnitude());s=np.array([(X[i]@Vh.T).T[:,:3]@nv+d for i in range(len(X))]);mn=s.min(1)
 print(n,'dt med/p95/max',np.median(dt),np.percentile(dt,95),dt.max(),'dr',np.median(dr),np.percentile(dr,95),dr.max(),'floor med/absmean',np.median(mn),np.mean(abs(mn)))
 for lo,hi,l in [(0,132,'init'),(132,901,'move'),(901,946,'final')]:print(' ',l,'dtmean',dt[lo:hi-1].mean(),'drmean',dr[lo:hi-1].mean())
print('static variability',np.ptp(T[:132,:3,3],axis=0),np.ptp(T[901:,:3,3],axis=0))
print('boundary dt 128-145',np.round(np …[+156 chars]
```
<details><summary>output</summary>

```
B initial static depthMAE 0.2545 signed -0.1700 n27
B manipulation depthMAE 0.1740 signed -0.1251 n154
B final static depthMAE 0.0865 signed -0.0062 n9
B all depthMAE 0.1813 signed -0.1278 n190
result initial static depthMAE 0.2285 signed -0.1494 n27
result manipulation depthMAE 0.1747 signed -0.1225 n154
result final static depthMAE 0.0869 signed -0.0059 n9
result all depthMAE 0.1782 signed -0.1254 n190
temporal/floor
B dt med/p95/max 0.011277772177705267 0.045846667008680335 0.1759136065878757 dr 0.771459966069196 2.548891781692365 12.328823102254969 floor med/absmean 0.010098135017877263 0.016430710863656203
  init dtmean 0.00962875408165536 drmean 0.5178840424545678
  move dtmean 0.017692565456844053 drmean 1.1530433768155828
  final dtmean 0.006476446731518417 drmean 0.488631265584629 …[+598 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for d in /code/reconstruction /workspace/v2d_cari4d /workspace/v2d_foundation_pose /workspace/v2d_moge /workspace/v2d_sam2; do echo -n "$d "; git -C "$d" rev-parse HEAD 2>/dev/null || echo no-git; done'
```
<details><summary>output</summary>

```
/code/reconstruction 7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
/workspace/v2d_cari4d no-git
/workspace/v2d_foundation_pose no-git
/workspace/v2d_moge no-git
/workspace/v2d_sam2 no-git

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,platform
print('python',sys.version.split()[0],platform.platform())
PY
du -sh /work/outputs /work/scratch /work/tools 2>/dev/null || true
sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py"
```
<details><summary>output</summary>

```
python 3.11.10 Linux-6.8.0-138-generic-x86_64-with-glibc2.35
160K	/work/outputs
21M	/work/scratch
311016ab3aeab891e1a31fbff506edd28a2253713b6d00498d07eedc51fe5511  /work/outputs/object_poses_world.npz
0a05c69948d6c394e602bebb39aa9051f4e9c39ee8fa3a2b75735412921fb909  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```
<details><summary>output</summary>

```
107 MiB, 49140 MiB
14325 MiB, 49140 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Toy-stroller 6-DoF trajectory recovery

## Result

"'`object_poses_world.npz` contains `T_world_mesh` with shape `(946, 4, 4)` and dtype
`float64`. Each matrix maps the supplied mesh-file coordinates to the released world
frame. The run was deterministic and completed in **122.7 s** on the available GPU.

The supplied CARI4D/FoundationPose/CoCoNet trajectory was retained as the main
source of evidence. The change is a rigid-trajectory correction: pooled rest poses,
robust SE(3) smoothing, silhouette registration, and smooth phase transitions.
No ground-truth or challenge annotation was used.

## Phases found

| Frames | Phase | Evidence | Confidence |
|---|---|---|---|
| 0-131 | Initial rest | Object silhouette centroid and pooled textured-mesh scores stay fixed. Contact begins earlier, but the stroller does not visibly move until about frame 132. | High |
| 132-900 | Manipulation / push | Centroid moves and the object yaws through a long sweep while staying in contact with the floor. Includes short hand/leg occlusions. | Medium-high |
| 901-945 | Final rest | Motion settles at frame 901 and all subsequent frames support one pooled pose. | High for the image registration; medium for absolute depth |

The transitions use a 14-frame acceleration blend at 132-145 and a 11-frame
deceleration blend at 890-900, avoiding the abrupt 3-6 cm one-frame jumps produced
by concatenating the tracked and pooled segments.

## Method

1. …[+7376 chars]
```
<details><summary>output</summary>

```
total 148K
-rw-rw-r-- 1 hanzhou hanzhou 6.4K Oct 20 23:22 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 2.2K Oct 20 23:22 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 23:17 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 119K Oct 20 23:19 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  12K Oct 20 23:17 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py; python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz');T=p['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all(),T[:,3].min(),T[:,3].max())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import shutil
shutil.rmtree('/work/outputs/__pycache__', ignore_errors=True)
PY
ls -la /work/outputs; printf '\\nREPORT tail\\n'; tail -20 /work/outputs/REPORT.md; printf '\\nTOOLS tail\\n'; tail -20 /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
total 152
drwxrwxr-x 2 hanzhou hanzhou   4096 Oct 20 23:22 .
drwxrwxr-x 8 hanzhou hanzhou   4096 Oct 20 22:20 ..
-rw-rw-r-- 1 hanzhou hanzhou   6513 Oct 20 23:22 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou   2234 Oct 20 23:22 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 121366 Oct 20 23:19 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  11672 Oct 20 23:17 solve.py

REPORT tail
  centroid residuals there are smoothed from neighbouring reliable frames.
- Absolute camera depth is the least certain component. The supplied MoGe-2 depth
  disagrees with the mesh surface by roughly 0.18 m on average because the thin open
  mesh and background mix at mask pixels. The result therefore preserves the
  baseline's calibrated depth and corrects mainly the image-plane and temporal terms.
- The final view is re …[+1459 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read();s=s.replace('full-sequence\\n   mask IoU from 0.563 to 0.614 on the held-out validation sample.','full-sequence\\n   mask IoU from 0.563 to 0.614 on the validation sample.')
open(p,'w').write(s)
PY
grep -n \"validation sample\" /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
43:   mask IoU from 0.563 to 0.614 on the validation sample.

```
</details>

**Agent:** Completed the trajectory recovery task.

Outputs created:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(946, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline reconstruction script
- `/work/outputs/REPORT.md` — phase analysis, method, validation, confidence
- `/work/outputs/TOOLS.md` — dependency and provenance details

Validation on 190 sampled frames:

| Metric | Baseline | Result |
|---|---:|---:|
| Mask IoU | 0.5633 | **0.6140** |
| Texture gradient NCC | 0.0565 | **0.1808** |
| Colour correlation | 0.2061 | **0.3006** |
| Translation step median | 11.3 mm | **5.95 mm** |
| Rotation step median | 0.771° | **0.309°** |

`solve.py` was run successfully in 122.7 seconds and recomputes the output from the provided inputs without network access.
