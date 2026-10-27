# T1_ds_ep009

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T16:33:03Z

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
/usr/bin/bash -c "sed -n '1,200p' /work/AGENT_TASK.md"
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
/usr/bin/bash -c "wc -l -c /work/AGENT_TASK.md && nl -ba /work/AGENT_TASK.md | sed -n '35,80p'"
```
<details><summary>output</summary>

```
  120 10592 /work/AGENT_TASK.md
    35	   compare with the image: silhouette agreement with the object mask (excluding pixels covered by the person),
    36	   edges / texture / colour agreement, consistency with the MoGe-2 depth map; optical flow or feature tracks between
    37	   neighbouring frames (a correct trajectory must explain the image motion of the object's texture).
    38	3. **Several hypotheses per frame, not one**: the baseline, the FoundationPose and CoCoNet estimates, their 180 deg
    39	   symmetry flips about each principal axis of the mesh, poses propagated from neighbouring frames, and fresh
    40	   registrations (e.g. render-and-compare optimisation from several initialisations, or FoundationPose
    41	   re-registration on chosen frames).
    42	4. **Global deco …[+3593 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%p %s bytes\\n' | sort"
```
<details><summary>output</summary>

```
/work/inputs/camera.json 818 bytes
/work/inputs/episode.json 66 bytes
/work/inputs/frame0_reference_human.npz 192794 bytes
/work/inputs/ground_plane.json 398 bytes
/work/inputs/masks.h5 7317562 bytes
/work/inputs/object_mesh.glb 10271176 bytes
/work/inputs/reconstruction.npz 44113824 bytes
/work/inputs/task.txt 253 bytes
/work/inputs/video.mp4 25204920 bytes

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/inputs/task.txt; echo; cat /work/inputs/episode.json; echo; cat /work/inputs/camera.json; echo; cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk over to the big red bowl on the skinny wood chair and pick up the big red bowl using both hands and proceed to place the big red bowl on the ground in front of the skinny wood chair and then slide the big red bowl underneath the skinny wood chair.

{
 "episode_index": 9,
 "object": "big_red_bowl",
 "frames": 870
}
{
 "K": [
  [
   820.0006103515625,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   820.0006103515625,
   560.5
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "K_pipeline": [
  [
   807.0595703125,
   0.0,
   768.0
  ],
  [
   0.0,
   807.0595703125,
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
   0.9998747856499233,
   0.01296888275244168,
   -0.0090675852173233,
   0.22330019385202945
  ],
  [
   -0.012517244470275248,
   0.998758985165105,
   0.048205 …[+738 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1963
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2131
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/depth_report.json 1790
/pipeline/depth/moge2/aligned.h5 1096979728
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 92627
/pipeline/depth/moge2/raw.npy 3078881408
/pipeline/export/episode_000009/edex 557
/pipeline/export/episode_000009/wild_export.json 2756
/pipeline/foundationpose/episode_000009.pkl 248033
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 665378 …[+549 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work -maxdepth 2 -type d -printf '%p\\n' | sort | head -100; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"
```
<details><summary>output</summary>

```
/work
/work/.codex_home
/work/.codex_home/.tmp
/work/.codex_home/sessions
/work/.codex_home/shell_snapshots
/work/.codex_home/skills
/work/.codex_home/thread-writer-locks
/work/.codex_home/tmp
/work/.net
/work/inputs
/work/logs
/work/outputs
/work/scratch
NVIDIA RTX 6000 Ada Generation, 49140 MiB, 1177 MiB
NVIDIA RTX 6000 Ada Generation, 49140 MiB, 17 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
print(r.files)
for k in r.files:
 a=r[k]
 print(k,a.shape,a.dtype, 'minmax', (float(np.nanmin(a)),float(np.nanmax(a))) if np.issubdtype(a.dtype,np.number) else '')
print('contacts first/last nonzero/quantiles')
c=r['contact_prob_right_left']
nz=np.where((c>.25).any(1))[0]
print(nz[:20], nz[-20:] if len(nz) else [], np.quantile(c,[0,.1,.5,.9,.99,1],axis=0))
PY"
```
<details><summary>output</summary>

```
['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (870, 4, 4) float32 minmax (-0.9999918341636658, 3.259047746658325)
T_cam_mesh_foundationpose (870, 4, 4) float32 minmax (-0.9997876286506653, 3.8506410121917725)
T_cam_mesh_coconet (870, 4, 4) float32 minmax (-0.9999740719795227, 3.033494234085083)
human_vertices_world (870, 4610, 3) float32 minmax (-0.9300755262374878, 4.0531392097473145)
human_keypoints_world (870, 70, 3) float32 minmax (-0.904833972454071, 4.020415782928467)
contact_prob_right_left (870, 2) float32 minmax (5.694751311138835e-13, 1.0)
contacts first/last nonzero/quantiles
[166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183
 184 185] [750 751 752 …[+270 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 print(list(f.keys()))
 for k in f.keys():
  a=f[k]; print(k,a.shape,a.dtype)
  x=a[:] if a.size<1e9 else None
  if x is not None:
   s=x.sum((1,2)); print(' sum q',np.quantile(s,[0,.01,.1,.5,.9,.99,1]),'nonzero',np.count_nonzero(s))
PY"
```
<details><summary>output</summary>

```
['object', 'person']
object (870, 1152, 1536) bool
person (870, 1152, 1536) bool

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for f in /pipeline/.stages/*.json /pipeline/pipeline_report.json /pipeline/depth/moge2/depth_report.json; do echo ==== $f; cat $f; done'
```
<details><summary>output</summary>

```
{
  "name": "06_coconet",
  "outputs": [
    {
      "mtime_ns": 1791217500034753568,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000009/inference/coconet.pth",
      "size": 665378324
    },
    {
      "mtime_ns": 1791217474830753664,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000009/inference/materialized_inputs.h5",
      "size": 3230407288
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000009/export/episode_000009",
      "--depth-h5",
      "/home/hanzhou/Projects/v2d/run/out/episode_000009/depth/moge2/aligned.h5",
      "--mhr-init",
      "/home/hanzhou/Projects/v2d/ru …[+14464 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
p='/work/inputs/object_mesh.glb'
sc=trimesh.load(p,force='scene',process=False)
print(type(sc), sc.geometry.keys())
for n,g in sc.geometry.items():
 print(n, type(g), 'verts',len(g.vertices),'faces',len(g.faces),'bounds',g.bounds,'extents',g.extents,'visual',type(g.visual), 'tex',getattr(g.visual,'uv',None).shape if hasattr(g.visual,'uv') and getattr(g.visual,'uv',None) is not None else None)
 m=g.copy()
 try:
  S,b=trimesh.bounds.oriented_bounds(m)
  print('S',S,'b',b)
 except Exception as e: print(e)
print('scene extents',sc.extents, sc.bounds)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.scene.scene.Scene'> odict_keys(['redbowl'])
redbowl <class 'trimesh.base.Trimesh'> verts 295188 faces 98420 bounds [[-0.14736827 -0.06377962 -0.14680047]
 [ 0.14807813  0.06895825  0.15066887]] extents [0.2954464  0.13273787 0.29746935] visual <class 'trimesh.visual.texture.TextureVisuals'> tex (295188, 2)
S [[-6.12323400e-17 -1.00000000e+00 -6.12323400e-17  2.58931518e-03]
 [-1.00000000e+00  6.12323400e-17 -1.22464680e-16  3.54930758e-04]
 [ 1.22464680e-16  6.12323400e-17 -1.00000000e+00  1.93420053e-03]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]] b [0.13273787 0.2954464  0.29746935]
scene extents [0.2954464  0.13273787 0.29746935] [[-0.14736827 -0.06377962 -0.14680047]
 [ 0.14807813  0.06895825  0.15066887]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh']; F=len(T)
p=T[:,:3,3]; q=R.from_matrix(T[:,:3,:3]).as_rotvec()
dv=np.linalg.norm(np.diff(p,axis=0),axis=1); da=np.linalg.norm(np.diff(q,axis=0),axis=1)
for name,x in [('pos',p),('step',dv),('angstep',da)]:
 print(name,'q',np.quantile(x,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1],axis=0))
# print every 10 frames poses / steps
print('f pxyz rotvec step ang')
for i in range(0,F,10):
 print(i, np.round(p[i],3), np.round(q[i],3), '' if i==0 else f'{dv[i-1]:.4f} {da[i-1]:.4f}')
print('top jumps')
for i in np.argsort(dv)[-30:][::-1]: print('pos',i,dv[i])
for i in np.argsort(da)[-30:][::-1]: print('rot',i,da[i])
PY"
```
<details><summary>output</summary>

```
pos q [[-0.56566733  0.17361197  2.26462531]
 [-0.55605499  0.20091641  2.27442171]
 [-0.44875895  0.21701758  2.35245587]
 [-0.35587121  0.22392154  2.45631232]
 [-0.16447295  0.23665231  2.7779544 ]
 [ 0.03549327  0.64868906  3.10589802]
 [ 0.12735161  1.00047664  3.15304643]
 [ 0.17620123  1.19158396  3.18766713]
 [ 0.17697556  1.20828605  3.20937546]
 [ 0.17709918  1.21481114  3.23847701]
 [ 0.18166178  1.21770942  3.25904775]]
step q [0.         0.         0.         0.         0.0022931  0.00910783
 0.01624028 0.02554776 0.03376609 0.05025215 0.08651954]
angstep q [0.         0.         0.         0.         0.         0.07391317
 0.23919416 6.20543506 6.26168594 6.27531108 6.28104424]
f pxyz rotvec step ang
0 [0.127 0.681 3.154] [-2.369 -0.022  2.057] 
10 [0.127 0.68  3.147] [-2.369 …[+6720 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, pprint
p='/pipeline/foundationpose/episode_000009.pkl'
with open(p,'rb') as f: o=pickle.load(f)
print(type(o))
if isinstance(o,dict):
 print(o.keys())
 for k,v in o.items():
  print(k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None))
  if isinstance(v,(str,int,float,bool,list,tuple)): print(repr(v)[:1000])
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
dict_keys(['frames', 'camera_id', 'fp_poses_cam', 'fp_candidate_count', 'fp_selected_candidate_index', 'fp_selected_iou', 'fp_selected_visibility', 'fp_retry_count', 'fp_reliable', 'fp_oracle_rotation_error_deg', 'fp_translation_threshold_m', 'fp_selected_translation_distance_m', 'gt_object_visibility_ratio', 'gt_object_visibility_valid', 'gt_object_rendered_pixels', 'gt_object_visible_pixels', 'obj_pose_world', 'obj_rot_init', 'obj_t_init', 'metadata', 'fp_tracking_state', 'fp_confirmed_pose_source', 'fp_pending_pose_source', 'fp_emitted_pose_source', 'fp_gap_length', 'fp_observed_visible_pixels', 'fp_rendered_visible_pixels', 'fp_rendered_visibility', 'fp_full_image_iou', 'fp_tolerant_precision', 'fp_tolerant_recall', 'fp_centroid_distance', 'fp_confirmation_count', 'fp_re …[+3068 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/frames && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=nw=1 /work/inputs/video.mp4 && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"select='not(mod(n,30))',scale=384:288,tile=5x6\" -frames:v 1 /work/scratch/contact_30.jpg && ls -lh /work/scratch/contact_30.jpg"
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=29.000000
nb_frames=870
-rw-rw-r-- 1 hanzhou hanzhou 208K Oct 20 20:27 /work/scratch/contact_30.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 for k in ['object','person']:
  x=f[k][:]
  s=x.sum((1,2))
  print(k,'area quant',np.quantile(s,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
  print('every10')
  print(list(zip(range(0,len(s),10),s[::10].tolist())))
PY"
```
<details><summary>output</summary>

```
object area quant [ 177.    236.07  663.8   992.3  1952.25 3034.   3346.   3972.8  4643.55
 5436.62 5614.  ]
every10
[(0, 3194), (10, 3217), (20, 3186), (30, 3182), (40, 3190), (50, 3175), (60, 3127), (70, 3119), (80, 3156), (90, 3161), (100, 3135), (110, 3100), (120, 3105), (130, 3153), (140, 3147), (150, 3160), (160, 3060), (170, 2318), (180, 2148), (190, 2229), (200, 2142), (210, 1894), (220, 1902), (230, 1887), (240, 1627), (250, 1782), (260, 2011), (270, 2001), (280, 2373), (290, 2453), (300, 2823), (310, 2816), (320, 3085), (330, 3283), (340, 3605), (350, 3798), (360, 3858), (370, 4187), (380, 4528), (390, 4595), (400, 4731), (410, 5377), (420, 5535), (430, 5068), (440, 4340), (450, 3989), (460, 3416), (470, 2677), (480, 2278), (490, 2757), (500, 1160), (510, 300), (520, 299), (530,  …[+1765 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
for key in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 T=r[key]; F=len(T)
 dR=np.einsum('fji,fjk->fik',T[:-1,:3,:3],T[1:,:3,:3])
 ang=R.from_matrix(dR).magnitude()
 dt=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)
 print('\\n',key)
 print('ang deg q',np.quantile(np.degrees(ang),[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
 print('dt q',np.quantile(dt,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]))
 print('top angle')
 for i in np.argsort(ang)[-25:][::-1]: print(i,round(np.degrees(ang[i]),2),round(dt[i],4))
PY"
```
<details><summary>output</summary>

```

 T_world_mesh
ang deg q [7.19382946e-15 7.19382946e-15 7.19382946e-15 1.28279105e-14
 1.28279105e-14 2.38242877e+00 6.34860950e+00 1.32007092e+01
 2.08690586e+01 3.72611305e+01 1.10258613e+02]
dt q [0.         0.         0.         0.         0.0022931  0.00910783
 0.01624028 0.02554776 0.03376609 0.05025215 0.08651954]
top angle
446 110.26 0.0173
619 97.79 0.0365
608 85.64 0.0165
654 84.98 0.0125
537 71.92 0.0264
437 51.95 0.0064
155 45.31 0.0392
293 45.09 0.0119
424 39.63 0.0062
294 36.15 0.0056
161 35.44 0.012
326 34.61 0.0274
445 31.13 0.013
389 30.18 0.0106
382 29.42 0.0192
280 28.65 0.0199
279 28.64 0.0123
266 28.49 0.0077
225 28.42 0.0209
226 28.42 0.0069
523 28.24 0.0096
492 28.16 0.0088
383 27.93 0.0321
388 27.64 0.0156
324 27.4 0.0255

 T_cam_mesh_foundationpose
ang deg q [7.555 …[+1549 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh']
print('frame pos x y z | R euler xyz deg')
for i in range(0,870,10):
 e=R.from_matrix(T[i,:3,:3]).as_euler('xyz',degrees=True)
 print(f'{i:3d}',*[f'{x:.3f}' for x in T[i,:3,3]],'|',*[f'{x:.1f}' for x in e])
PY"
```
<details><summary>output</summary>

```
frame pos x y z | R euler xyz deg
  0 0.127 0.681 3.154 | -175.0 81.9 5.4
 10 0.127 0.680 3.147 | -175.0 81.9 5.4
 20 0.127 0.679 3.145 | -175.0 81.9 5.4
 30 0.127 0.679 3.145 | -175.0 81.9 5.4
 40 0.127 0.680 3.148 | -175.0 81.9 5.4
 50 0.127 0.680 3.149 | -175.0 81.9 5.4
 60 0.127 0.680 3.146 | -175.0 81.9 5.4
 70 0.127 0.675 3.128 | -175.0 81.9 5.4
 80 0.127 0.675 3.127 | -175.0 81.9 5.4
 90 0.128 0.673 3.119 | -175.0 81.9 5.4
100 0.128 0.673 3.117 | -175.0 81.9 5.4
110 0.128 0.669 3.101 | -175.0 81.9 5.4
120 0.128 0.672 3.113 | -175.0 81.9 5.4
130 0.128 0.672 3.114 | -175.0 81.9 5.4
140 0.128 0.673 3.119 | -175.0 81.9 5.4
150 0.128 0.673 3.118 | -175.0 81.9 5.4
160 0.122 0.689 3.164 | -1.8 57.8 179.6
170 0.118 0.688 3.165 | -1.2 -5.9 -179.4
180 0.119 0.692 3.183 | -2.4 -33.4 -179.1
190 …[+2851 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh']; p=T[:,:3,3]
# speed smoothed
v=np.linalg.norm(np.diff(p,axis=0),axis=1)
for w in [1,3,5,10]:
 if w>1:
  vv=np.convolve(v,np.ones(w)/w,'same')
 else: vv=v
 print('w',w,'q',np.quantile(vv,[0,.25,.5,.75,.9,.95,.99,1]))
 print('frames low<2mm',np.where(vv<.002)[0][:20], 'count',np.sum(vv<.002))
# segment runs where speed < threshold
for th in [.001,.002,.003,.005,.01]:
 b=v<th; runs=[]; s=None
 for i,x in enumerate(b):
  if x and s is None:s=i
  if not x and s is not None:runs.append((s,i));s=None
 if s is not None:runs.append((s,len(b)))
 print('th',th,[x for x in runs if x[1]-x[0]>=3][:30])
PY"
```
<details><summary>output</summary>

```
w 1 q [0.         0.0022931  0.00910783 0.01624028 0.02554776 0.03376609
 0.05025215 0.08651954]
frames low<2mm [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 20] count 213
w 3 q [0.         0.00280194 0.01083598 0.01687526 0.02380015 0.02811284
 0.0412651  0.059904  ]
frames low<2mm [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] count 198
w 5 q [0.         0.00268321 0.01132032 0.0174944  0.02275417 0.02639155
 0.03726672 0.05173224]
frames low<2mm [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] count 196
w 10 q [0.         0.00244659 0.01171821 0.01691922 0.02204145 0.02637843
 0.0335023  0.03802895]
frames low<2mm [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19] count 204
th 0.001 [(0, 5), (15, 19), (20, 31), (34, 60), (66, 80), (84,  …[+874 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work /pipeline -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' | head -100"
```
<details><summary>output</summary>

```
/work/scratch/contact_30.jpg
/work/.codex_home/skills/.system/imagegen/assets/imagegen.png
/work/.codex_home/skills/.system/openai-docs/assets/openai.png
/work/.codex_home/skills/.system/plugin-creator/assets/plugin-creator.png
/work/.codex_home/skills/.system/skill-creator/assets/skill-creator.png
/work/.codex_home/skills/.system/skill-installer/assets/skill-installer.png
/work/.codex_home/.tmp/plugins/plugins/airtable/assets/logo.png
/work/.codex_home/.tmp/plugins/plugins/atlassian-rovo/assets/app-icon.png
/work/.codex_home/.tmp/plugins/plugins/atlassian-rovo/assets/logo-dark.png
/work/.codex_home/.tmp/plugins/plugins/atlassian-rovo/assets/logo.png
/work/.codex_home/.tmp/plugins/plugins/boltz-api-cli/assets/app-icon.png
/work/.codex_home/.tmp/plugins/plugins/build-ios-apps/assets/app-ico …[+8027 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
with open('/pipeline/foundationpose/episode_000009.pkl','rb') as f:o=pickle.load(f)
for k in ['fp_selected_iou','fp_selected_visibility','fp_reliable','fp_oracle_rotation_error_deg','gt_object_visibility_ratio','fp_tracking_state','fp_confirmed_pose_source','fp_emitted_pose_source','fp_rendered_visibility','fp_full_image_iou','fp_tolerant_precision','fp_tolerant_recall']:
 a=np.asarray(o[k]); print('\\n',k,a.shape)
 try: print('q',np.quantile(a,[0,.01,.1,.25,.5,.75,.9,.99,1]))
 except: print(np.unique(a,return_counts=True))
print('metadata'); print(o['metadata'])
PY"
```
<details><summary>output</summary>

```

 fp_selected_iou (870,)
q [0.         0.         0.         0.         0.83538488 0.91516539
 0.93538441 0.95548145 0.97914147]

 fp_selected_visibility (870,)
q [0.00153441 0.00349668 0.01525155 0.39979664 0.75837174 0.98893975
 1.         1.         1.        ]

 fp_reliable (870,)
(array([ True]), array([870]))

 fp_oracle_rotation_error_deg (870,)
q [nan nan nan nan nan nan nan nan nan]

 gt_object_visibility_ratio (870,)
q [nan nan nan nan nan nan nan nan nan]

 fp_tracking_state (870,)
q [0. 0. 0. 0. 0. 0. 0. 0. 0.]

 fp_confirmed_pose_source (870,)
q [6. 7. 7. 7. 7. 7. 7. 7. 7.]

 fp_emitted_pose_source (870,)
q [6. 7. 7. 7. 7. 7. 7. 7. 7.]

 fp_rendered_visibility (870,)
q [0.00153441 0.00349668 0.01525155 0.39979664 0.75837174 0.98893975
 1.         1.         1.        ]

 fp_fu …[+8556 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=sc.geometry['redbowl']; v=g.visual
print(type(v),v)
for attr in dir(v):
 if not attr.startswith('_'):
  try:
   x=getattr(v,attr)
   if not callable(x): print(attr,type(x),getattr(x,'shape',None),repr(x)[:300])
  except Exception as e: pass
print('materials',g.visual.material)
print('uv bounds',g.visual.uv.min(0),g.visual.uv.max(0))
print('unique uv count',len(np.unique(g.visual.uv.round(6),axis=0)))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.texture.TextureVisuals object at 0x79d8f913a250>
defined <class 'bool'> None True
face_materials <class 'NoneType'> None None
kind <class 'str'> None 'texture'
material <class 'trimesh.visual.material.PBRMaterial'> None <trimesh.visual.material.PBRMaterial object at 0x79d8f6c79310>
mesh <class 'trimesh.base.Trimesh'> None <trimesh.Trimesh(vertices.shape=(295188, 3), faces.shape=(98420, 3))>
uv <class 'trimesh.caching.TrackedArray'> (295188, 2) TrackedArray([[0.66838098, 0.952196  ],
              [0.66838098, 0.952196  ],
              [0.66838098, 0.952196  ],
              ...,
              [0.152035  , 0.035375  ],
              [0.152035  , 0.035375  ],
              [0.152035  , 0.035375  ]])
vertex_attributes <class 'tr …[+237 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np, os
sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
g=sc.geometry['redbowl']; m=g.visual.material
for attr in ['baseColorTexture','baseColorFactor','metallicFactor','roughnessFactor','image','name']:
 x=getattr(m,attr,None)
 print(attr,type(x),repr(x)[:200])
 if x is not None and hasattr(x,'size'):
  print(' size',x.size,'mode',x.mode)
  x.save('/work/scratch/mesh_texture.png')
PY
ls -lh /work/scratch/mesh_texture.png || true"
```
<details><summary>output</summary>

```
baseColorTexture <class 'PIL.PngImagePlugin.PngImageFile'> <PIL.PngImagePlugin.PngImageFile image mode=RGB size=2263x2435 at 0x7B6116023350>
 size (2263, 2435) mode RGB
baseColorFactor <class 'NoneType'> None
metallicFactor <class 'float'> 0.0
roughnessFactor <class 'NoneType'> None
image <class 'NoneType'> None
name <class 'str'> '_texture'
-rw-rw-r-- 1 hanzhou hanzhou 3.1M Oct 20 20:28 /work/scratch/mesh_texture.png

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 3 -type f | head -100; grep -R \"nvdiffrast\\|RasterizeCudaContext\\|render_mesh\" -n /code/reconstruction /workspace/v2d_cari4d 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
253:        self.glctx = dr.RasterizeCudaContext()
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/align_depth_to_mhr_wild.py:74:    import nvdiffrast.torch as dr
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/align_depth_to_mhr_wild.py:76:    glctx = dr.RasterizeCudaContext()
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py:16:import nvdiffrast.torch as dr
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py:111:    glctx = dr.RasterizeCudaContext()
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py:237:        raise ValueError(f"nvdiffrast raster must have shape [N,H,W,4], got {tuple(raster.shape)}")
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/prep/mhr …[+11564 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['nvdiffrast','torch','torchvision','cv2','trimesh','open3d','pytorch3d','pyrender','skimage','scipy','h5py']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'NO',repr(e))
PY"
```
<details><summary>output</summary>

```
nvdiffrast OK 0.4.0
torch OK 2.5.1+cu124
torchvision OK 0.20.1+cu124
cv2 OK 4.11.0
trimesh OK 5.1.1
open3d OK 0.20.0
pytorch3d OK 0.7.9
pyrender OK 0.1.45
skimage OK 0.26.0
scipy OK 1.16.3
h5py OK 3.16.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,260p' /code/reconstruction/modules/v2d_foundation_pose/lib/fp_utils.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Re-export commonly used FoundationPose utility functions.

This wrapper lets other packages import GPU-accelerated depth filtering
and visualization helpers via a clean package path instead of manipulating
sys.path themselves:

    from v2d.foundation_pose.lib.fp_utils import erode_depth, bilateral_filter_depth, depth2xyzmap
"""
import os
import sys

_FP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FoundationPose")
if _FP_DIR not in sys.path:
    sys.path.insert(0, _FP_DIR)

from Utils import (  # noqa: E402
    erode_depth,
    bilateral_filter_depth,
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    nvdiffrast_render …[+27 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '120,260p' /code/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py"
```
<details><summary>output</summary>

```
    if mesh.visual.vertex_colors is None:
      logging.info(f"WARN: mesh doesn't have vertex_colors, assigning a pure color")
      mesh.visual.vertex_colors = np.tile(np.array([128,128,128]).reshape(1,3), (len(mesh.vertices), 1))
    mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[...,:3], device=device, dtype=torch.float)/255.0

  mesh_tensors.update({
    'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
    'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
    'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
  })
  return mesh_tensors


def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None …[+4689 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,180p' /code/reconstruction/modules/v2d_foundation_pose/lib/run_render_poses.py"
```
<details><summary>output</summary>

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
GPU-batched mesh overlay renderer using nvdiffrast.

Renders a mesh at every pose in poses_dir and composites over the matching
background frames from frames_dir.  All poses in a batch are rasterised in a
single nvdiffrast call, so throughput scales with GPU parallelism rather than
frame count.

Supports both UV-textured meshes and vertex-coloured meshes (detected
automatically via make_mesh_tensors).
"""
import argparse
import logging
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

from v2d.common.datatypes import CameraIntrinsics, Transform3d
from v2d.foundation_pose.lib.fp_utils import nvdiffrast_render, make_me …[+4962 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,140p' /code/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py"
```
<details><summary>output</summary>

```
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, sys, time,torch,pickle,trimesh,itertools,pdb,zipfile,datetime,imageio,gzip,logging,joblib,importlib,uuid,signal,multiprocessing,psutil,subprocess,tarfile,scipy,argparse
from pytorch3d.transforms import so3_log_map,so3_exp_map,se3_exp_map,se3_log_map,matrix_to_axis_angle,matrix_to_euler_angles,euler_angles_to_matrix, rotation_6d_to_matrix
from pytorch3d.render …[+4634 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"def projection_matrix_from_intrinsics\\|def transform_pts\\|def to_homo_torch\" /code/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py; sed -n '620,700p' /code/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py"
```
<details><summary>output</summary>

```
520:def to_homo_torch(pts):
529:def transform_pts(pts,tf):
752:def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
    tfs = compute_tf_batch(left, right, top, bottom)
    return tfs

  else:
    raise RuntimeError

  return tf



def cv_draw_text(img,text,uv_top_left,color=(255, 255, 255),fontScale=0.5,thickness=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,outline_color=None,line_spacing=1.5):
  H,W = img.shape[:2]
  uv_top_left = np.array(uv_top_left, dtype=float)
  assert uv_top_left.shape == (2,)

  for line in text.splitlines():
    (w, h), _ = cv2.getTextSize(text=line,fontFace=fontFace,fontScale=fontScale,thickness=thickness,)
    uv_bottom_left_i = uv_top_left + [0, h]

    ############# Ensure inside image
    while uv_bottom_left_i[0]<0:
      uv_ …[+2322 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '510,545p' /code/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py; sed -n '745,795p' /code/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py"
```
<details><summary>output</summary>

```

def to_homo(pts):
  '''
  @pts: (N,3 or 2) will homogeneliaze the last dimension
  '''
  assert len(pts.shape)==2, f'pts.shape: {pts.shape}'
  homo = np.concatenate((pts, np.ones((pts.shape[0],1))),axis=-1)
  return homo


def to_homo_torch(pts):
  '''
  @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
  ones = torch.ones((*pts.shape[:-1],1), dtype=torch.float, device=pts.device)
  homo = torch.cat((pts, ones),dim=-1)
  return homo


def transform_pts(pts,tf):
  """Transform 2d or 3d points
  @pts: (...,N_pts,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=pts.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]


def transform_dirs(dirs,tf):
  """
  @dirs: (...,3)
  @tf: (...,4,4)
   …[+1866 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '795,825p' /code/reconstruction/modules/v2d_foundation_pose/lib/FoundationPose/Utils.py"
```
<details><summary>output</summary>

```
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])
  else:
    raise NotImplementedError

  return proj



def symmetry_tfs_from_info(info, rot_angle_discrete=5):
  symmetry_tfs = [np.eye(4)]
  if 'symmetries_discrete' in info:
    tfs = np.array(info['symmetries_discrete']).reshape(-1,4,4)
    tfs[...,:3,3] *= 0.001
    symmetry_tfs = [np.eye(4)]
    symmetry_tfs += list(tfs)
  if 'symmetries_continuous' in info:
    axis = np.array(info['symmetries_continuous'][0]['axis']).reshape(3)
    offset = info['symmetries_continuous'][0]['offset']
    rxs = [0]
    rys = [0]
    rzs = [0]
    if axis[0]>0:
      rxs = np.arange(0,360,rot_angle_discrete)/180.0*np.pi
    elif axis[1]>0:
       …[+153 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
for i in range(0,870,10):
 ay=T[i,:3,1]; ax=T[i,:3,0]; az=T[i,:3,2]
 yaw=np.degrees(np.arctan2(ax[2],ax[0]))
 print(f'{i:3d} ay {np.round(ay,2)} dotY {ay[1]:+.2f} horiz {np.linalg.norm(ay[[0,2]]):.2f} yawXZ {yaw:7.1f}')
PY"
```
<details><summary>output</summary>

```
  0 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 10 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 20 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 30 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 40 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 50 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 60 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 70 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 80 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
 90 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
100 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
110 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz 0.01 yawXZ   -81.9
120 ay [ 0.01 -1.   -0.01] dotY -1.00 horiz  …[+4669 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_key_montage.py <<'PY'
import cv2, h5py, numpy as np
keys=[0,100,150,160,170,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,800,830,860]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames={}
idxs=set(keys)
i=0
while True:
 ok,img=cap.read()
 if not ok:break
 if i in idxs:frames[i]=img
 i+=1
cap.release()
with h5py.File('/work/inputs/masks.h5','r') as f:
 om=f['object'][:]; pm=f['person'][:]
tiles=[]
for k in keys:
 img=frames[k].copy(); o=om[k].astype(np.uint8); p=pm[k].astype(np.uint8)
 ys,xs=np.where(o)
 if len(xs):
  x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
  cx,cy=(x0+x1)//2,(y0+y1)//2
 else: cx,cy=768,576
 s=max(140,int(max(x1-x0+1,y1-y0+1)*2.6))
 s=min(s,450)
 x0=max(0,cx-s//2); y0=max(0,cy-s//2); x1=min(img.shape[1],x0+s); y1=min(img.shape[0],y0+s)
 x0=max(0,x1-s);y0=max(0,y1-s)
 crop=img[y0:y1,x0:x1].copy()
 oc=o[y0:y1,x0:x1]; pc=p[y0:y1,x0:x1]
 co,_=cv2.findContours(oc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(crop,co,-1,(0,255,0),2)
 cp,_=cv2.findContours(pc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(crop,cp,-1,(0,0,255),1)
 crop=cv2.resize(crop,(320,320),interpolation=cv2.INTER_CUBIC)
 cv2.putText(crop,f'f{k} area={o.sum()}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
 tiles.append(crop)
rows=[]
for r in range(0,len(tiles),5):
 row=tiles[r:r+5]
 while len(row)<5:row.append(np.zeros_like(tiles[0 …[+208 chars]
```
<details><summary>output</summary>

```
(2560, 1600, 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/score_baseline.py <<'PY'
import os,sys,time
import numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr
from PIL import Image

@torch.no_grad()
def main():
 device='cuda'
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
 mesh=sc.geometry['redbowl']
 verts=torch.tensor(mesh.vertices,device=device,dtype=torch.float)
 faces=torch.tensor(mesh.faces,device=device,dtype=torch.int32)
 V=torch.cat([verts,torch.ones(len(verts),1,device=device)],1)
 H,W=384,512
 K=np.array([[820.0006103515625,0,758.0999755859375],[0,820.0006103515625,560.5],[0,0,1.]])
 K[0]*=W/1536; K[1]*=H/1152
 # projection from FP
 nc=.001; fc=100.; depth=fc-nc; q=-(fc+nc)/depth; qn=-2*fc*nc/depth
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],dtype=np.float32)
 glcam=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32)
 with open('/work/inputs/camera.json') as f:import json;c=json.load(f)
 Twc=np.array(c['T_world_cam']); Twc_inv=np.linalg.inv(Twc)
 r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']
 # baseline camera pose
 Tcm=Twc_inv[None]@T
 Tct=torch.tensor(Tcm,device=device,dtype=torch.float)
 mtx=torch.tensor(proj@glcam,device=device,dtype=torch.float)[None]@Tct
 pos_clip=mtx[:,None]@V[None,...,None]
 glctx=dr.RasterizeCudaContext()
 with h5py.File('/work/inputs/masks.h5','r') as f:
  obj=f['object'][:]; per=f['person'][:]
 keys=np.arang …[+1073 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/score_baseline.py", line 54, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/score_baseline.py", line 37, in main
    rast,_=dr.rasterize(glctx,pc,faces,resolution=[H,W])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/ …[+571 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/pos_clip=mtx\\[:,None\\]@V\\[None,...,None\\]/pos_clip=(mtx[:,None]@V[None,...,None])[...,0]/\" /work/scratch/score_baseline.py
python /work/scratch/score_baseline.py"
```
<details><summary>output</summary>

```
  0 obj=  353 pred=  311 visIoU=0.000 rawIoU=0.000
 10 obj=  357 pred=  311 visIoU=0.000 rawIoU=0.000
 20 obj=  355 pred=  311 visIoU=0.000 rawIoU=0.000
 30 obj=  353 pred=  311 visIoU=0.000 rawIoU=0.000
 40 obj=  352 pred=  311 visIoU=0.000 rawIoU=0.000
 50 obj=  352 pred=  311 visIoU=0.000 rawIoU=0.000
 60 obj=  345 pred=  311 visIoU=0.000 rawIoU=0.000
 70 obj=  345 pred=  316 visIoU=0.000 rawIoU=0.000
 80 obj=  351 pred=  317 visIoU=0.000 rawIoU=0.000
 90 obj=  349 pred=  320 visIoU=0.000 rawIoU=0.000
100 obj=  347 pred=  321 visIoU=0.000 rawIoU=0.000
110 obj=  344 pred=  328 visIoU=0.000 rawIoU=0.000
120 obj=  344 pred=  322 visIoU=0.000 rawIoU=0.000
130 obj=  350 pred=  302 visIoU=0.000 rawIoU=0.000
140 obj=  350 pred=  273 visIoU=0.000 rawIoU=0.000
150 obj=  352 pred=  245 visIoU=0.0 …[+3666 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/score_baseline.py').read().split(\"if __name__\")[0])
# skip
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/pred=rast\\[...,3\\]>0/pred=torch.flip(rast[...,3],dims=[1])>0/\" /work/scratch/score_baseline.py
python /work/scratch/score_baseline.py"
```
<details><summary>output</summary>

```
  0 obj=  353 pred=  311 visIoU=0.870 rawIoU=0.870
 10 obj=  357 pred=  311 visIoU=0.856 rawIoU=0.856
 20 obj=  355 pred=  311 visIoU=0.860 rawIoU=0.860
 30 obj=  353 pred=  311 visIoU=0.860 rawIoU=0.860
 40 obj=  352 pred=  311 visIoU=0.862 rawIoU=0.862
 50 obj=  352 pred=  311 visIoU=0.862 rawIoU=0.862
 60 obj=  345 pred=  311 visIoU=0.885 rawIoU=0.885
 70 obj=  345 pred=  316 visIoU=0.899 rawIoU=0.899
 80 obj=  351 pred=  317 visIoU=0.876 rawIoU=0.876
 90 obj=  349 pred=  320 visIoU=0.879 rawIoU=0.879
100 obj=  347 pred=  321 visIoU=0.887 rawIoU=0.887
110 obj=  344 pred=  328 visIoU=0.893 rawIoU=0.893
120 obj=  344 pred=  322 visIoU=0.897 rawIoU=0.897
130 obj=  350 pred=  322 visIoU=0.877 rawIoU=0.877
140 obj=  350 pred=  320 visIoU=0.877 rawIoU=0.877
150 obj=  352 pred=  320 visIoU=0.8 …[+3680 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/overlay_baseline.py <<'PY'
import os,sys,time,json
import numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr

def main():
 device='cuda'; H,W=1152,1536
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); mesh=sc.geometry['redbowl']
 verts=torch.tensor(mesh.vertices,device=device,dtype=torch.float); faces=torch.tensor(mesh.faces,device=device,dtype=torch.int32)
 V=torch.cat([verts,torch.ones(len(verts),1,device=device)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']); Twc=np.array(c['T_world_cam']); Twc_inv=np.linalg.inv(Twc)
 nc=.001;fc=100.;depth=fc-nc;q=-(fc+nc)/depth;qn=-2*fc*nc/depth
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],dtype=np.float32)
 glcam=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32)
 r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; Tcm=Twc_inv[None]@T; Tct=torch.tensor(Tcm,device=device,dtype=torch.float)
 mtx=torch.tensor(proj@glcam,device=device,dtype=torch.float)[None]@Tct
 pos_clip=(mtx[:,None]@V[None,...,None])[...,0]
 glctx=dr.RasterizeCudaContext()
 keys=[0,100,160,180,200,240,280,320,360,400,420,440,480,500,520,560,600,640,680,720,760,800,860]
 cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames={};i=0
 while True:
  ok,img=cap.read()
  if not ok:break
  if i in keys:frames[i]=img
  i+=1
 cap.release()
 with h5py.File('/work/inp …[+1683 chars]
```
<details><summary>output</summary>

```
(1920, 1280, 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_score.py <<'PY'
import numpy as np, torch, trimesh, cv2, h5py, json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR

def main():
 dev='cuda'; H,W=576,768
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); mesh=sc.geometry['redbowl']
 tex=np.array(mesh.visual.material.image.convert('RGB'))
 tex=torch.tensor(cv2.resize(tex,(1024,1024))[None],device=dev,dtype=torch.float).permute(0,3,1,2)/255
 tex=tex.permute(0,2,3,1)
 verts=torch.tensor(mesh.vertices,device=dev,dtype=torch.float);faces=torch.tensor(mesh.faces,device=dev,dtype=torch.int32)
 uv=torch.tensor(mesh.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1]
 V=torch.cat([verts,torch.ones(len(verts),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;Twc=np.array(c['T_world_cam']);Ti=np.linalg.inv(Twc)
 nc=.001;fc=100.;dz=fc-nc;q=-(fc+nc)/dz;qn=-2*fc*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32)
 gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32)
 glctx=dr.RasterizeCudaContext();r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
 cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames={};i=0
 keys=[0,100,200,240,280,320,360,400,420,440,480,720,760,800,860]
 while True:
  ok,img=cap.read()
  if not ok:break
  if i in keys:frames[ …[+2190 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/yaw_score.py", line 65, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/work/scratch/yaw_score.py", line 8, in main
    tex=np.array(mesh.visual.material.image.convert('RGB'))
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/mesh.visual.material.image/mesh.visual.material.baseColorTexture/" /work/scratch/yaw_score.py
python /work/scratch/yaw_score.py'
```
<details><summary>output</summary>

```
frame 0 n 24 best [(90, 0.695), (60, 0.702), (45, 0.712), (75, 0.718), (105, 0.754), (30, 0.785)]
frame 100 n 24 best [(75, 0.696), (90, 0.7), (45, 0.726), (60, 0.728), (105, 0.747), (120, 0.795)]
frame 200 n 24 best [(-15, 0.876), (0, 0.888), (-45, 0.899), (-30, 0.904), (15, 0.915), (-60, 0.965)]
frame 240 n 24 best [(-120, 0.906), (-105, 0.947), (-150, 1.014), (-135, 1.016), (165, 1.025), (-180, 1.062)]
frame 280 n 24 best [(150, 0.996), (90, 1.028), (105, 1.062), (165, 1.065), (60, 1.073), (-165, 1.091)]
frame 320 n 24 best [(-30, 0.885), (-15, 0.91), (-45, 0.928), (0, 0.96), (-60, 0.963), (-75, 0.983)]
frame 360 n 24 best [(-15, 0.816), (-30, 0.851), (0, 0.858), (-45, 0.863), (-60, 0.863), (15, 0.889)]
frame 400 n 24 best [(-45, 0.809), (-60, 0.81), (-30, 0.823), (-15, 0.846), (0, 0.85 …[+731 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world']; c=r['contact_prob_right_left']
for i in range(0,870,10):
 rh=K[i,21:42].mean(0); lh=K[i,42:63].mean(0); rw=K[i,22] if False else K[i,21]; lw=K[i,42]
 v=rh-lh; p=T[i,:3,3]
 print(f'{i:3d} c {c[i,0]:.2f} {c[i,1]:.2f} rh {np.round(rh,2)} lh {np.round(lh,2)} d {np.linalg.norm(v):.3f} vec {np.round(v,2)} br {np.round(rh-p,2)} bl {np.round(lh-p,2)}')
PY"
```
<details><summary>output</summary>

```
  0 c 0.00 0.00 rh [-0.65  0.5   3.67] lh [-0.26  0.51  3.93] d 0.475 vec [-0.39 -0.01 -0.26] br [-0.78 -0.18  0.51] bl [-0.39 -0.17  0.78]
 10 c 0.00 0.00 rh [-0.63  0.49  3.56] lh [-0.25  0.49  3.77] d 0.435 vec [-0.38  0.   -0.21] br [-0.75 -0.19  0.41] bl [-0.37 -0.19  0.62]
 20 c 0.00 0.00 rh [-0.64  0.5   3.5 ] lh [-0.25  0.5   3.73] d 0.450 vec [-0.39  0.01 -0.22] br [-0.77 -0.18  0.36] bl [-0.38 -0.18  0.58]
 30 c 0.00 0.00 rh [-0.68  0.52  3.41] lh [-0.27  0.48  3.61] d 0.452 vec [-0.4   0.04 -0.2 ] br [-0.8  -0.16  0.27] bl [-0.4  -0.2   0.46]
 40 c 0.00 0.00 rh [-0.71  0.52  3.36] lh [-0.3   0.47  3.55] d 0.454 vec [-0.41  0.05 -0.2 ] br [-0.84 -0.16  0.21] bl [-0.43 -0.21  0.41]
 50 c 0.00 0.00 rh [-0.72  0.52  3.34] lh [-0.29  0.48  3.57] d 0.484 vec [-0.43  0.04 -0.23] br [-0 …[+11290 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_candidates.py <<'PY'
import numpy as np, torch, trimesh, cv2, json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR

def main():
 dev='cuda';H,W=1152,1536
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl']
 tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);f=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1]
 V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as ff:c=json.load(ff)
 K=np.array(c['K']);nc=.001;fc=100.;dz=fc-nc;q=-(fc+nc)/dz;qn=-2*fc*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32)
 gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
 cap=cv2.VideoCapture('/work/inputs/video.mp4');frames={};i=0;keys=[0,400,800]
 while 1:
  ok,x=cap.read()
  if not ok:break
  if i in keys:frames[i]=x
  i+=1
 cap.release();glctx=dr.RasterizeCudaContext();tiles=[]
 for k in keys:
  for mode in ['pre','post']:
   for a in [-90,-45,0,45,90]:
    Tp=T[k].copy();R=SR.from_euler('y',a,degrees=True).as_matrix() …[+1234 chars]
```
<details><summary>output</summary>

```
(1440, 1200, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/hand_yaw.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as SR
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world'];c=r['contact_prob_right_left']
rh=K[:,21:42].mean(1);lh=K[:,42:63].mean(1);v=rh-lh
u=v[:,[0,2]];ang=np.degrees(np.arctan2(u[:,1],u[:,0])) # atan2(z,x)
# unwrap u angle in grasped region; near 0 means +x, near -90 means -z
ang=np.degrees(np.unwrap(np.radians(ang)))
# base local x projected
bx=T[:,:3,0]; base=np.degrees(np.arctan2(bx[:,2],bx[:,0]))
base=np.degrees(np.unwrap(np.radians(base)))
# delta phase median using 170:190
mask=(c[:,0]>.8)&(c[:,1]>.8)
delta=np.median((ang-base)[170:191])
phi=ang-delta
# smooth u angle over phase using median filters
from scipy.ndimage import median_filter, gaussian_filter1d
for w in [1,5,11,21]:
 a2=ang.copy() if w==1 else median_filter(ang,size=w,mode='nearest')
 if w>1:a2=gaussian_filter1d(a2,sigma=w/4)
 p=a2-delta
 print('\\nw',w,'delta',delta)
 for i in range(0,870,10):
  if (i>=150 and i<=790) or i%100==0:
   print(f'{i:3d} u {ang[i]:7.1f} base {base[i]:7.1f} phi {p[i]:7.1f} est {base[i]-p[i]:7.1f}')
np.savez('/work/scratch/hand_yaw.npz',rh=rh,lh=lh,ang=ang,base=base,delta=delta,phi=phi,mask=mask)
PY
python /work/scratch/hand_yaw.py"
```
<details><summary>output</summary>

```

w 1 delta 127.15814
  0 u  -146.2 base   -81.9 phi  -273.3 est   191.4
100 u  -143.7 base   -81.9 phi  -270.8 est   188.9
150 u   -81.4 base   -81.9 phi  -208.6 est   126.7
160 u   -82.7 base  -122.2 phi  -209.9 est    87.7
170 u   -85.6 base  -185.9 phi  -212.7 est    26.8
180 u   -85.2 base  -213.4 phi  -212.4 est    -1.0
190 u   -84.9 base  -211.1 phi  -212.1 est     0.9
200 u   -84.8 base  -218.1 phi  -212.0 est    -6.1
210 u   -79.7 base  -284.8 phi  -206.9 est   -77.9
220 u   -75.7 base  -314.7 phi  -202.9 est  -111.8
230 u   -75.8 base  -348.0 phi  -203.0 est  -145.0
240 u   -75.7 base  -331.9 phi  -202.9 est  -129.0
250 u   -85.0 base  -343.4 phi  -212.1 est  -131.3
260 u   -91.3 base  -397.4 phi  -218.5 est  -178.9
270 u   -90.9 base  -431.1 phi  -218.1 est  -213.0
280 u   -90.6  …[+13158 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world'];c=r['contact_prob_right_left']
rh=K[:,21:42].mean(1);lh=K[:,42:63].mean(1);mid=(rh+lh)/2; p=T[:,:3,3]; d=p-mid
for i in range(0,870,20):
 print(f'{i:3d} c {c[i,0]:.2f}/{c[i,1]:.2f} p {np.round(p[i],3)} mid {np.round(mid[i],3)} d {np.round(d[i],3)}')
mask=(c[:,0]>.9)&(c[:,1]>.9)&(np.arange(870)>=175)&(np.arange(870)<=750)
print('d held mean',d[mask].mean(0),'std',d[mask].std(0),'q',np.quantile(d[mask],[.05,.5,.95],axis=0))
# per-axis smoothness residual std baseline displacement vs hand midpoint
for name,x in [('obj',p),('mid',mid),('d',d)]:
 v=np.diff(x,axis=0); sp=np.linalg.norm(v,axis=1)
 print(name,'speed held q',np.quantile(sp[175:750],[0,.1,.25,.5,.75,.9,1]))
 # window 11 accel
 acc=np.diff(x,n=2,axis=0); aa=np.linalg.norm(acc,axis=1)
 print(' acc q',np.quantile(aa[175:749],[0,.1,.25,.5,.75,.9,1]))
PY"
```
<details><summary>output</summary>

```
  0 c 0.00/0.00 p [0.127 0.681 3.154] mid [-0.457  0.507  3.801] d [ 0.584  0.174 -0.647]
 20 c 0.00/0.00 p [0.127 0.679 3.145] mid [-0.446  0.501  3.614] d [ 0.573  0.179 -0.469]
 40 c 0.00/0.00 p [0.127 0.68  3.148] mid [-0.506  0.495  3.456] d [ 0.633  0.185 -0.307]
 60 c 0.00/0.00 p [0.127 0.68  3.146] mid [-0.481  0.502  3.38 ] d [ 0.607  0.178 -0.234]
 80 c 0.00/0.00 p [0.127 0.675 3.127] mid [-0.338  0.506  3.265] d [ 0.466  0.17  -0.138]
100 c 0.00/0.00 p [0.128 0.673 3.117] mid [-0.255  0.52   3.161] d [ 0.383  0.152 -0.044]
120 c 0.00/0.00 p [0.128 0.672 3.113] mid [-0.168  0.521  3.061] d [0.296 0.151 0.052]
140 c 0.00/0.00 p [0.128 0.673 3.119] mid [-0.173  0.527  3.097] d [0.301 0.146 0.022]
160 c 0.00/0.00 p [0.122 0.689 3.164] mid [-0.038  0.559  3.085] d [0.16  0.13  0.079] …[+3874 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
keys=list(range(150,211,10))+list(range(730,811,5))
cap=cv2.VideoCapture('/work/inputs/video.mp4');frames={};i=0
while 1:
 ok,x=cap.read()
 if not ok:break
 if i in keys:frames[i]=x
 i+=1
cap.release()
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
tiles=[]
for k in keys:
 img=frames[k].copy(); om=o[k].astype(np.uint8); pm=p[k].astype(np.uint8)
 ys,xs=np.where(om|pm)
 # center around object
 ys,xs=np.where(om)
 if len(xs):cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
 else:
  ys,xs=np.where(pm);cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
 s=400;x0=max(0,cx-s//2);y0=max(0,cy-s//2);x1=min(1536,x0+s);y1=min(1152,y0+s);x0=max(0,x1-s);y0=max(0,y1-s)
 cr=img[y0:y1,x0:x1].copy()
 co,_=cv2.findContours(om[y0:y1,x0:x1],cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(cr,co,-1,(0,255,0),2)
 cr=cv2.resize(cr,(300,300));cv2.putText(cr,f'{k} a{om.sum()}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(cr)
rows=[np.concatenate(tiles[i:i+6],1) for i in range(0,len(tiles),6)]
out=np.concatenate(rows,0);cv2.imwrite('/work/scratch/transitions.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95]);print(out.shape,len(keys))
PY"
```
<details><summary>output</summary>

```
(1200, 1800, 3) 24

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_yaw_fine.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR

def main():
 dev='cuda';H,W=576,768
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl']
 tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None].permute(0,3,1,2)/255;tex=tex.permute(0,2,3,1)
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1]
 V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32)
 gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
 cap=cv2.VideoCapture('/work/inputs/video.mp4');keys=[0,30,60,90,120,150,800,820,840,860];frames={};i=0
 while 1:
  ok,x=cap.read()
  if not ok:break
  if i in keys:frames[i]=x
  i+=1
 cap.release()
 with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
 an …[+2343 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_yaw_fine.py", line 59, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/work/scratch/static_yaw_fine.py", line 44, in main
    x=predL[j][mask];y=L[mask]
      ~~~~~~~~^^^^^^
IndexError: too many indices for tensor of dimension 1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_yaw_fine.py'
s=open(p).read();s=s.replace(\"col=torch.flip(col,dims=[1])[...,0,:,:] # luma RGB rough\",\"col=torch.flip(col,dims=[1])\")
open(p,'w').write(s)
PY
python /work/scratch/static_yaw_fine.py"
```
<details><summary>output</summary>

```
0 tex [(50, 0.836), (45, 0.846), (65, 0.851), (70, 0.856), (55, 0.856), (60, 0.86), (90, 0.861), (85, 0.864)]
0 iou [(-30, 0.887), (5, 0.884), (-20, 0.884), (0, 0.883), (10, 0.883), (15, 0.883), (-5, 0.883), (-15, 0.882)]
30 tex [(50, 0.844), (70, 0.857), (65, 0.857), (45, 0.86), (55, 0.86), (115, 0.863), (85, 0.863), (100, 0.866)]
30 iou [(-25, 0.882), (-45, 0.882), (-35, 0.881), (-40, 0.88), (-30, 0.88), (-20, 0.88), (-55, 0.879), (-50, 0.879)]
60 tex [(70, 0.859), (50, 0.861), (65, 0.862), (100, 0.865), (85, 0.871), (115, 0.872), (55, 0.872), (90, 0.874)]
60 iou [(-90, 0.9), (-60, 0.899), (-45, 0.898), (-85, 0.896), (-75, 0.896), (-35, 0.896), (-30, 0.896), (-50, 0.896)]
90 tex [(55, 0.835), (70, 0.837), (95, 0.839), (85, 0.839), (50, 0.842), (90, 0.843), (65, 0.843), (75, 0.847)]
90 io …[+1973 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_static_candidates.py <<'PY'
import numpy as np,torch,trimesh,cv2,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR

def main():
 dev='cuda';H,W=1152,1536
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl']
 tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cap=cv2.VideoCapture('/work/inputs/video.mp4');frames={};i=0;keys=[0,100,800,860]
 while 1:
  ok,x=cap.read()
  if not ok:break
  if i in keys:frames[i]=x
  i+=1
 cap.release();glctx=dr.RasterizeCudaContext();angs=[-45,-30,-15,0,20,35,50,65,80,95];tiles=[]
 for k in keys:
  for a in angs:
   Tp=T[k].copy();Tp[:3,:3]=SR.from_euler('y',a,degrees=True).as …[+1205 chars]
```
<details><summary>output</summary>

```
(1920, 1200, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'];F=r['T_cam_mesh_foundationpose'];C=r['T_cam_mesh_coconet']
with open('/work/inputs/camera.json') as f:c=json.load(f)
Twc=np.array(c['T_world_cam']);Ti=np.linalg.inv(Twc)
FW=Twc[None]@F; CW=Twc[None]@C
for i in [0,100,150,160,170,180,200,240,300,400,500,600,700,760,800,860]:
 for name,X in [('base',T),('fp',FW),('coco',CW)]:
  ax=X[i,:3,0]; phi=np.degrees(np.arctan2(ax[2],ax[0])); ay=X[i,:3,1]
  print(f'{i:3d} {name:4s} Rphi {phi:7.1f} ay {np.round(ay,2)} t {np.round(X[i,:3,3],2)}')
 print()
# relative base to fp *positions at initial
for i in [0,100,150,800,860]:
 dR=T[i,:3,:3].T@FW[i,:3,:3];ang=np.degrees(R.from_matrix(dR).magnitude())
 print('angle base-fp',i,ang)
PY"
```
<details><summary>output</summary>

```
  0 base Rphi   -81.9 ay [ 0.01 -1.   -0.01] t [0.13 0.68 3.15]
  0 fp   Rphi   -55.6 ay [ 0.01 -1.    0.05] t [0.08 0.64 3.14]
  0 coco Rphi   -81.2 ay [ 0.01 -1.   -0.03] t [0.09 0.6  3.  ]

100 base Rphi   -81.9 ay [ 0.01 -1.   -0.01] t [0.13 0.67 3.12]
100 fp   Rphi  -127.1 ay [ 0.02 -1.   -0.03] t [0.09 0.61 3.07]
100 coco Rphi   -81.2 ay [ 0.01 -1.   -0.03] t [0.09 0.6  3.  ]

150 base Rphi   -81.9 ay [ 0.01 -1.   -0.01] t [0.13 0.67 3.12]
150 fp   Rphi  -125.1 ay [ 0.01 -1.   -0.02] t [0.09 0.59 2.98]
150 coco Rphi   -81.2 ay [ 0.01 -1.   -0.03] t [0.09 0.6  3.  ]

160 base Rphi  -122.2 ay [ 0.02 -1.   -0.02] t [0.12 0.69 3.16]
160 fp   Rphi  -122.7 ay [ 0.01 -1.   -0.  ] t [0.09 0.6  2.99]
160 coco Rphi  -121.5 ay [ 0.02 -1.   -0.03] t [0.09 0.61 3.03]

170 base Rphi   174.1 ay [ 0 …[+2506 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/detail_candidates.py <<'PY'
# based on render_static_candidates but output actual + candidates only mesh for 0,860
exec(open('/work/scratch/render_static_candidates.py').read().replace(\"keys=[0,100,800,860]\",\"keys=[0,860]\").replace(\"angs=[-45,-30,-15,0,20,35,50,65,80,95]\",\"angs=[-30,0,30,50,70,90]\").replace(\"cr=cv2.resize(img[y0:y1,x0:x1],(240,240))\",\"cr=cv2.resize(img[y0:y1,x0:x1],(400,400))\").replace(\"cv2.imwrite('/work/scratch/static_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])\",\"cv2.imwrite('/work/scratch/detail_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97])\").replace(\"rows=[np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)]\",\"rows=[np.concatenate(tiles[i:i+7],1) for i in range(0,len(tiles),7)]\"))
PY
python /work/scratch/detail_candidates.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/detail_candidates.py", line 2, in <module>
    exec(open('/work/scratch/render_static_candidates.py').read().replace("keys=[0,100,800,860]","keys=[0,860]").replace("angs=[-45,-30,-15,0,20,35,50,65,80,95]","angs=[-30,0,30,50,70,90]").replace("cr=cv2.resize(img[y0:y1,x0:x1],(240,240))","cr=cv2.resize(img[y0:y1,x0:x1],(400,400))").replace("cv2.imwrite('/work/scratch/static_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])","cv2.imwrite('/work/scratch/detail_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97])").replace("rows=[np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)]","rows=[np.concatenate(tiles[i:i+7],1) for i in range(0,len(tiles),7)]"))
  File "<string>", line 33, in <module>
  File "<string>", line 32, in main
Va …[+192 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c 'sed -i "s/while len(row)<5: row.append/while len(row)<7: row.append/" /work/scratch/detail_candidates.py
python /work/scratch/detail_candidates.py'
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/detail_candidates.py", line 2, in <module>
    exec(open('/work/scratch/render_static_candidates.py').read().replace("keys=[0,100,800,860]","keys=[0,860]").replace("angs=[-45,-30,-15,0,20,35,50,65,80,95]","angs=[-30,0,30,50,70,90]").replace("cr=cv2.resize(img[y0:y1,x0:x1],(240,240))","cr=cv2.resize(img[y0:y1,x0:x1],(400,400))").replace("cv2.imwrite('/work/scratch/static_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])","cv2.imwrite('/work/scratch/detail_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97])").replace("rows=[np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)]","rows=[np.concatenate(tiles[i:i+7],1) for i in range(0,len(tiles),7)]"))
  File "<string>", line 33, in <module>
  File "<string>", line 32, in main
Va …[+192 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/scratch/detail_candidates.py'
```
<details><summary>output</summary>

```
# based on render_static_candidates but output actual + candidates only mesh for 0,860
exec(open('/work/scratch/render_static_candidates.py').read().replace("keys=[0,100,800,860]","keys=[0,860]").replace("angs=[-45,-30,-15,0,20,35,50,65,80,95]","angs=[-30,0,30,50,70,90]").replace("cr=cv2.resize(img[y0:y1,x0:x1],(240,240))","cr=cv2.resize(img[y0:y1,x0:x1],(400,400))").replace("cv2.imwrite('/work/scratch/static_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])","cv2.imwrite('/work/scratch/detail_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97])").replace("rows=[np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)]","rows=[np.concatenate(tiles[i:i+7],1) for i in range(0,len(tiles),7)]"))

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/detail_candidates.py';s=open(p).read();s=s.replace(\".replace(\\\"rows=\", \".replace(\\\"while len(row)<5: row.append(np.zeros_like(tiles[0]))\\\",\\\"while len(row)<7: row.append(np.zeros_like(tiles[0]))\\\").replace(\\\"rows=\")
open(p,'w').write(s)
PY
python /work/scratch/detail_candidates.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/detail_candidates.py", line 2, in <module>
    exec(open('/work/scratch/render_static_candidates.py').read().replace("keys=[0,100,800,860]","keys=[0,860]").replace("angs=[-45,-30,-15,0,20,35,50,65,80,95]","angs=[-30,0,30,50,70,90]").replace("cr=cv2.resize(img[y0:y1,x0:x1],(240,240))","cr=cv2.resize(img[y0:y1,x0:x1],(400,400))").replace("cv2.imwrite('/work/scratch/static_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])","cv2.imwrite('/work/scratch/detail_candidates.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97])").replace("while len(row)<5: row.append(np.zeros_like(tiles[0]))","while len(row)<7: row.append(np.zeros_like(tiles[0]))").replace("rows=[np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)]","rows=[np.concatenate(tiles[i: …[+313 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/render_static_candidates.py'
```
<details><summary>output</summary>

```
   rast,_=dr.rasterize(glctx,pc,fc,resolution=[H,W]);uvm,_=dr.interpolate(uv,rast,fc);col=dr.texture(tex,uvm,filter_mode='linear')
   col=torch.flip(col,dims=[1]).cpu().numpy()[0];alpha=np.flip(rast[...,3].cpu().numpy()[0],0)>0
   img=frames[k].copy();img[alpha]=np.clip(.65*col[alpha]*255+.35*img[alpha],0,255).astype(np.uint8)[:,::-1]
   ys,xs=np.where(alpha)
   if len(xs):cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2;s=max(220,int(max(xs.max()-xs.min(),ys.max()-ys.min())*2.2))
   else:cx,cy,s=768,576,300
   x0=max(0,cx-s//2);y0=max(0,cy-s//2);x1=min(W,x0+s);y1=min(H,y0+s);x0=max(0,x1-s);y0=max(0,y1-s)
   cr=cv2.resize(img[y0:y1,x0:x1],(240,240));cv2.putText(cr,f'k{k} {a:+}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);tiles.append(cr)
 rows=[np.concatenate(tiles[i:i+5],1) for …[+189 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_static_candidates.py';s=open(p).read()
s=s.replace(\"keys=[0,100,800,860]\",\"keys=[0,860]\").replace(\"angs=[-45,-30,-15,0,20,35,50,65,80,95]\",\"angs=[-30,0,30,50,70,90]\").replace(\"cr=cv2.resize(img[y0:y1,x0:x1],(240,240))\",\"cr=cv2.resize(img[y0:y1,x0:x1],(400,400))\").replace(\"static_candidates.jpg\",\"detail_candidates.jpg\")
s=s.replace(\"rows=[np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)];out=np.concatenate(rows,0)\",\"rows=[]\\n for rr in range(0,len(tiles),6):\\n  row=tiles[rr:rr+6]\\n  while len(row)<6: row.append(np.zeros_like(tiles[0]))\\n  rows.append(np.concatenate(row,1))\\n out=np.concatenate(rows,0)\")
open('/work/scratch/detail_candidates.py','w').write(s)
PY
python /work/scratch/detail_candidates.py"
```
<details><summary>output</summary>

```
(800, 2400, 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/hand_global_score.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR
from scipy.ndimage import gaussian_filter1d

def wrap(a):return (a+180)%360-180

def main():
 dev='cuda';H,W=384,512
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl']
 tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];Kp=r['human_keypoints_world'];cp=r['contact_prob_right_left']
 rh=Kp[:,21:42].mean(1);lh=Kp[:,42:63].mean(1);u=rh-lh;hang=np.degrees(np.unwrap(np.arctan2(u[:,2],u[:,0])))
 hang=gaussian_filter1d(hang,sigma=5)
 bx=T[:,:3,0];bphi=np.degrees(np.unwrap …[+2655 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/hand_global_score.py", line 60, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/work/scratch/hand_global_score.py", line 23, in main
    phi[i]=initial+hang[i]-hang[anchor]
                           ~~~~^^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/anchor=180\\./anchor=180/' /work/scratch/hand_global_score.py
python /work/scratch/hand_global_score.py"
```
<details><summary>output</summary>

```
200 n 102 best [(-110, 0.937), (-140, 0.941), (-120, 0.945), (-130, 0.951), (-150, 0.952)]
220 n 64 best [(-180, nan), (-170, nan), (-160, nan), (-150, nan), (-140, nan)]
240 n 84 best [(-10, 0.831), (20, 0.843), (60, 0.879), (50, 0.899), (10, 0.909)]
260 n 85 best [(100, 0.971), (70, 1.006), (80, 1.011), (50, 1.03), (60, 1.068)]
280 n 141 best [(-160, 0.809), (-150, 0.85), (-70, 0.852), (-170, 0.87), (150, 0.885)]
300 n 153 best [(-100, 0.744), (-150, 0.753), (-130, 0.773), (-90, 0.797), (-110, 0.803)]
320 n 168 best [(-60, 0.782), (-120, 0.782), (-80, 0.786), (-90, 0.819), (-70, 0.821)]
340 n 180 best [(-130, 0.68), (-120, 0.749), (-140, 0.774), (-150, 0.802), (-170, 0.813)]
360 n 166 best [(-50, 0.71), (-10, 0.811), (-110, 0.827), (-70, 0.837), (-40, 0.839)]
380 n 209 best [(-70, 0.695) …[+1985 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
r=np.load('/work/inputs/reconstruction.npz');K=r['human_keypoints_world'];c=r['contact_prob_right_left'];T=r['T_world_mesh']
for mode in ['all','wrist','first7','centroid_ends']:
 if mode=='all':rh=K[:,21:42].mean(1);lh=K[:,42:63].mean(1)
 elif mode=='wrist':rh=K[:,21];lh=K[:,42]
 elif mode=='first7':rh=K[:,21:28].mean(1);lh=K[:,42:49].mean(1)
 else:rh=(K[:,21]+K[:,26]).mean(1);lh=(K[:,42]+K[:,47]).mean(1)
 u=rh-lh; a=np.degrees(np.unwrap(np.arctan2(u[:,2],u[:,0]))); a=gaussian_filter1d(a,3)
 print(mode,'angles',[(i,round(a[i],1)) for i in [170,180,200,240,300,360,400,440,480,520,560,600,640,680,720,760]])
# differences from anchor
PY"
```
<details><summary>output</summary>

```
all angles [(170, -84.7), (180, -85.3), (200, -84.8), (240, -77.8), (300, -105.3), (360, -151.7), (400, -152.6), (440, -96.5), (480, -60.6), (520, -24.7), (560, -31.3), (600, -24.6), (640, -34.8), (680, -38.9), (720, -48.6), (760, -42.4)]
wrist angles [(170, -85.0), (180, -87.3), (200, -85.3), (240, -75.9), (300, -105.2), (360, -148.8), (400, -150.7), (440, -95.4), (480, -57.4), (520, -22.1), (560, -30.7), (600, -16.8), (640, -29.5), (680, -37.6), (720, -50.0), (760, -48.5)]
first7 angles [(170, -84.6), (180, -85.7), (200, -84.6), (240, -78.0), (300, -105.9), (360, -151.4), (400, -152.1), (440, -96.6), (480, -60.2), (520, -23.8), (560, -31.3), (600, -23.2), (640, -33.5), (680, -38.6), (720, -49.2), (760, -45.0)]
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
Index …[+78 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_traj.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def wrap(a):return (a+180)%360-180
def smoothstep(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)

def get_trans(T,mode):
 p=T[:,:3,3].copy()
 if mode=='base':return p
 ps=savgol_filter(p,21,3,axis=0)
 p0=np.median(p[:195],0);p1=np.median(p[790:],0)
 q=p.copy()
 q[:]=p0
 a=smoothstep((np.arange(870)-195)/25);q=np.where((np.arange(870)[:,None]>=195)[:,:],p0[None]+a[:,None]*(ps-p0[None]),q)
 # dynamic uses ps, then blend to final 760:810
 dyn=ps.copy()
 a2=smoothstep((np.arange(870)-760)/50)[:,None]
 dyn=(1-a2)*dyn+a2*p1[None]
 q=np.where((np.arange(870)>=220)[:,None],dyn,q)
 q=np.where((np.arange(870)>=810)[:,None],p1[None],q)
 return q

def main():
 dev='cuda';H,W=384,512
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl']
 tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0] …[+4144 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_traj.py", line 89, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/work/scratch/eval_traj.py", line 55, in main
    X=T.copy();X[:,:3,3]=pos;X[:,:3,:3]=np.einsum('ij,fjk->fik',np.stack([SR.from_euler('y',d,degrees=True).as_matrix() for d in delta]),T[0,:3,:3])
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/einsumfunc.py", line 1371, in einsum
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: operand has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provid …[+38 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/einsum('ij,fjk->fik'/einsum('fij,jk->fik'/\" /work/scratch/eval_traj.py
python /work/scratch/eval_traj.py"
```
<details><summary>output</summary>

```
('base', 'base') ang q [  2.3824288   13.20070941  20.86905898  37.26113047 110.25861377] step q [0.00910783 0.02554776 0.03376609 0.05025215 0.08651954]
('base', 'handtilt') ang q [  4.27807156  24.9824977   39.22415764  73.98154368 169.9133353 ] step q [0.00910783 0.02554776 0.03376609 0.05025215 0.08651954]
('base', 'yawonly') ang q [0.14873616 1.14000787 1.48450188 2.06238151 2.35276761] step q [0.00910783 0.02554776 0.03376609 0.05025215 0.08651954]
('smooth', 'base') ang q [  2.3824288   13.20070941  20.86905898  37.26113047 110.25861377] step q [0.00496603 0.01228758 0.01566026 0.02286354 0.02543372]
('smooth', 'handtilt') ang q [  4.27807156  24.9824977   39.22415764  73.98154368 169.9133353 ] step q [0.00496603 0.01228758 0.01566026 0.02286354 0.02543372]
('smooth', 'yawonly') ang …[+3398 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];bx=T[:,:3,0];phi=np.degrees(np.arctan2(bx[:,2],bx[:,0]))
# unwrap modulo180
p180=np.degrees(np.unwrap(np.radians(phi),period=np.pi))
# raw unwrap period360
p360=np.degrees(np.unwrap(np.radians(phi)))
for i in range(0,870,10):print(f'{i:3d} raw {phi[i]:7.1f} u180 {p180[i]:8.1f} u360 {p360[i]:8.1f} delta180 {p180[i]-p180[0]:8.1f}')
print('step180 q',np.quantile(np.abs(np.diff(p180)),[0,.25,.5,.75,.9,.95,.99,1]))
print('step360 q',np.quantile(np.abs(np.diff(p360)),[0,.25,.5,.75,.9,.95,.99,1]))
# smooth p180 and fit
for sig in [2,4,8,15]:
 s=gaussian_filter1d(p180,sig)
 print('sig',sig,'final delta',s[-1]-s[0], 'step q',np.quantile(np.abs(np.diff(s)),[.5,.9,.95,.99,1]))
PY"
```
<details><summary>output</summary>

```
  0 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 10 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 20 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 30 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 40 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 50 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 60 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 70 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 80 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
 90 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
100 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
110 raw   -81.9 u180    -81.9 u360    -81.9 delta180      0.0
120 raw   -81.9 u180    -81.9 u360    -81.9 delta180     …[+5199 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/static_yaw_fine.npz');a=z['angs']
for k in ['0','30','60','90','120','150','800','820','840','860']:
 r=z[k]
 print('\\nk',k)
 for off in [-90,-60,-45,-30,-15,0,15,30,45,50,60,75,90,120]:
  i=np.where(a==off)[0][0];print(off,round(r[i,0],3),round(r[i,2],3),end='; ')
 print()
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
z=np.load('/work/scratch/hand_global_score.npz');offs=z['offs'];res=z['res'];phi=z['phi']
keys=[200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,560,580,600,620,640,660,680,700,720,740,760]
for k,r in zip(keys,res):
 s=r[:,0];n=r[:,1]
 if n.max()<80:print(k,'none');continue
 idx=np.argsort(np.nan_to_num(s,nan=99))[:8]
 vals=[]
 for i in idx:
  ab=phi[k]+offs[i];rel=ab-phi[0]
  vals.append((int(offs[i]),round(float(ab),1),round(float(rel),1),round(float(s[i]),3),int(n[i])))
 print(k,vals)
PY"
```
<details><summary>output</summary>

```
200 [(-110, -191.3, -109.3, 0.937, 98), (-140, -221.3, -139.3, 0.941, 101), (-120, -201.3, -119.3, 0.945, 98), (-130, -211.3, -129.3, 0.951, 100), (-150, -231.3, -149.3, 0.952, 101), (-90, -171.3, -89.3, 0.986, 98), (-70, -151.3, -69.3, 1.013, 96), (-100, -181.3, -99.3, 1.028, 97)]
220 none
240 [(-10, -85.2, -3.2, 0.831, 81), (20, -55.2, 26.8, 0.843, 82), (60, -15.2, 66.8, 0.879, 81), (50, -25.2, 56.8, 0.899, 81), (10, -65.2, 16.8, 0.909, 84), (0, -75.2, 6.8, 0.952, 83), (40, -35.2, 46.8, 1.034, 80), (30, -45.2, 36.8, 1.073, 82)]
260 [(100, 12.7, 94.6, 0.971, 80), (70, -17.3, 64.6, 1.006, 83), (80, -7.3, 74.6, 1.011, 81), (50, -37.3, 44.6, 1.03, 82), (60, -27.3, 54.6, 1.068, 83), (40, -47.3, 34.6, 1.082, 83), (30, -57.3, 24.6, 1.12, 85), (-10, -97.3, -15.4, 1.147, 82)]
280 [(-160, -248.9,  …[+3862 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
from skimage.morphology import binary_erosion
cap=cv2.VideoCapture('/work/inputs/video.mp4');frames={};keys=[0,100,400,800,860];i=0
while 1:
 ok,x=cap.read()
 if not ok:break
 if i in keys:frames[i]=x
 i+=1
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
for k in keys:
 m=o[k]&~p[k];m=binary_erosion(m,iterations=3)
 rgb=frames[k][...,::-1][m];lab=cv2.cvtColor(frames[k],cv2.COLOR_BGR2LAB)[m]
 print(k,len(rgb),'rgb mean',rgb.mean(0),'std',rgb.std(0),'L std',lab[:,0].std(),'red/gr ratio',np.median(rgb[:,0]/np.maximum(rgb[:,1],1)))
PY"
```
<details><summary>output</summary>

```
<stdin>:11: FutureWarning: `binary_erosion` is deprecated since version 0.26 and will be removed in version 0.28. Use `skimage.morphology.erosion` instead. Note the pixel shift by 1 for even-sized footprints (see docstring notes).
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
  File "/opt/conda/lib/python3.11/site-packages/skimage/morphology/misc.py", line 47, in func_out
    return func(image, footprint=footprint, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/skimage/_shared/utils.py", line 690, in wrapped
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
TypeError: binary_erosion() got an unexpected keyword argument 'iterations'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');frames={};keys=[0,100,400,800,860];i=0
while 1:
 ok,x=cap.read()
 if not ok:break
 if i in keys:frames[i]=x
 i+=1
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:];p=f['person'][:]
for k in keys:
 m=o[k]&~p[k];m=cv2.erode(m.astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool)
 rgb=frames[k][...,::-1][m];lab=cv2.cvtColor(frames[k],cv2.COLOR_BGR2LAB)[m]
 print(k,len(rgb),'rgb mean',np.round(rgb.mean(0),1),'std',np.round(rgb.std(0),1),'L std',round(lab[:,0].std(),1),'rg ratio',round(float(np.median(rgb[:,0]/np.maximum(rgb[:,1],1))),2))
PY"
```
<details><summary>output</summary>

```
0 2444 rgb mean [95.1 13.4 16.2] std [22.  12.8 12.8] L std 16.8 rg ratio 8.5
100 2391 rgb mean [91.7 12.5 14.3] std [21.1 12.7 12.6] L std 16.4 rg ratio 9.44
400 3779 rgb mean [90.5 15.2 15.1] std [13.9 10.5 10.4] L std 11.5 rg ratio 7.09
800 2806 rgb mean [69.1 11.4 12.6] std [18.  12.4 12.1] L std 14.2 rg ratio 8.1
860 2882 rgb mean [68.6 11.5 12.2] std [18.5 12.3 12. ] L std 14.4 rg ratio 7.65

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_yaw_variants.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def wrap(a):return (a+180)%360-180
def main():
 dev='cuda';H,W=384,512
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl'];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);Kp=r['human_keypoints_world'];cp=r['contact_prob_right_left']
 rh=Kp[:,21:42].mean(1);lh=Kp[:,42:63].mean(1);u=rh-lh;hang=np.degrees(np.unwrap(np.arctan2(u[:,2],u[:,0])));hang=gaussian_filter1d(han …[+3654 chars]
```
<details><summary>output</summary>

```
base IoU 0.576 0.5362 tex 0.8587 0.8666 weighted 0.8573 nvis 22
[(0, 0.87, 0.92, 310), (50, 0.87, 0.93, 308), (100, 0.88, 0.93, 309), (150, 0.87, 0.92, 308), (180, 0.34, None, 83), (220, 0.29, None, 60), (240, 0.37, None, 68), (260, 0.42, None, 94), (280, 0.47, 1.0, 128), (300, 0.39, 0.86, 144), (320, 0.48, 0.8, 163), (340, 0.43, 0.87, 170), (360, 0.39, 0.92, 166), (380, 0.41, 0.87, 209), (400, 0.52, 0.78, 269), (420, 0.6, 0.8, 367), (440, 0.65, 0.87, 318), (460, 0.57, 0.94, 217), (700, 0.54, 0.84, 201), (720, 0.54, 0.9, 206), (740, 0.47, 0.87, 163), (760, 0.55, 0.73, 158), (780, 0.6, 0.73, 209), (800, 0.84, 0.8, 349), (830, 0.83, 0.81, 349), (860, 0.83, 0.81, 351)]
hand IoU 0.5703 0.5469 tex 0.9311 0.8955 weighted 0.9234 nvis 22
[(0, 0.87, 0.83, 310), (50, 0.87, 0.83, 308), (100, 0.88, 0. …[+2605 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/eval_yaw_variants.npz')
for n in z.files:
 X=z[n];d=np.einsum('fji,fjk->fik',X[:-1,:3,:3],X[1:,:3,:3]);a=np.degrees(R.from_matrix(d).magnitude());dt=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1)
 print(n,'ang',np.quantile(a,[.5,.75,.9,.95,.99,1]),'vel',np.quantile(dt,[.5,.9,.99,1]))
 # static rotation spread
 for sl in [(0,155),(800,870)]:
  d=np.einsum('fji,fjk->fik',X[sl[0]:sl[1]-1,:3,:3],X[sl[0]+1:sl[1],:3,:3]);print(' ',sl,np.degrees(R.from_matrix(d).magnitude()).max())
PY"
```
<details><summary>output</summary>

```
base ang [  2.3824288    6.34860949  13.20070941  20.86905898  37.26113047
 110.25861377] vel [0.00496603 0.01228758 0.02286354 0.02543372]
  (0, 155) 6.86829249224686e-15
  (800, 870) 2.320643454062832e-15
hand ang [  2.37970332   6.40720083  13.46338876  20.8915      37.32784247
 110.97102279] vel [0.00496603 0.01228758 0.02286354 0.02543372]
  (0, 155) 6.023144916747663e-15
  (800, 870) 9.905985889061502e-15
handres ang [  2.37206518   6.37603375  13.36151101  20.77110227  37.26160078
 110.76398031] vel [0.00496603 0.01228758 0.02286354 0.02543372]
  (0, 155) 6.023144916747663e-15
  (800, 870) 1.5180467445779352e-14
linear ang [  2.43232499   6.34061934  13.23765853  20.95119263  37.12422382
 110.12111786] vel [0.00496603 0.01228758 0.02286354 0.02543372]
  (0, 155) 6.023144916747663e-1 …[+253 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/eval_yaw_variants.npz');r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
for n in ['base','handres']:
 X=z[n];d=np.einsum('fji,fjk->fik',X[:-1,:3,:3],X[1:,:3,:3]);a=np.degrees(R.from_matrix(d).magnitude())
 print(n)
 for i in np.argsort(a)[-20:][::-1]:print(i,round(a[i],1),'ay before/after',np.round(T[i,:3,1],2),np.round(T[i+1,:3,1],2))
PY"
```
<details><summary>output</summary>

```
base
446 110.3 ay before/after [ 0.03 -1.   -0.07] [ 0.05 -1.   -0.03]
619 97.8 ay before/after [-0.01 -1.   -0.08] [ 0.08 -1.    0.06]
608 85.6 ay before/after [-0.   -1.   -0.04] [ 0.01 -1.   -0.03]
654 85.0 ay before/after [ 0.15 -0.98  0.09] [ 0.02 -1.    0.03]
537 71.9 ay before/after [-0.22 -0.97  0.05] [ 0.02 -1.    0.05]
437 52.0 ay before/after [-0.01 -1.   -0.1 ] [ 0.01 -0.99 -0.11]
155 45.3 ay before/after [ 0.01 -1.   -0.01] [ 0.02 -1.   -0.03]
293 45.1 ay before/after [ 0.05 -1.   -0.09] [ 0.02 -0.99 -0.12]
424 39.6 ay before/after [ 0.  -1.  -0.1] [-0.01 -0.99 -0.1 ]
294 36.1 ay before/after [ 0.02 -0.99 -0.12] [ 0.03 -0.99 -0.13]
161 35.4 ay before/after [ 0.   -1.   -0.02] [ 0.05 -1.   -0.04]
326 34.6 ay before/after [-0.11 -0.99  0.08] [-0.07 -0.99 -0.13]
445 31.1 ay befor …[+1809 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world'];rh=K[:,21:42].mean(1);lh=K[:,42:63].mean(1);u=rh-lh;ha=np.degrees(np.arctan2(u[:,2],u[:,0]))
for i in range(440,451):
 bx=T[i,:3,0];bp=np.degrees(np.arctan2(bx[2],bx[0]));print(i,bp,ha[i], 'Rlocal delta',np.degrees(R.from_matrix(T[i-1,:3,:3].T@T[i,:3,:3]).magnitude()))
PY"
```
<details><summary>output</summary>

```
440 72.98159 -94.38479 Rlocal delta 5.929747145560586
441 60.52817 -93.558 Rlocal delta 12.427388034971694
442 59.318993 -93.19609 Rlocal delta 1.7088945163036553
443 61.464344 -91.40258 Rlocal delta 2.190647787932065
444 58.43178 -92.59772 Rlocal delta 3.149334445372673
445 59.732445 -91.44472 Rlocal delta 1.5944905614591325
446 28.692745 -89.88303 Rlocal delta 31.128659599590893
447 -81.46893 -89.86478 Rlocal delta 110.25861406206286
448 -81.46695 -89.39474 Rlocal delta 0.005057645500187369
449 -87.2013 -88.49312 Rlocal delta 5.954335223133725
450 -81.06043 -88.71911 Rlocal delta 6.20484877254324

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_yaw_variants.py';s=open(p).read()
old=\"\"\"for name,off in offs.items():
  X=T.copy();X[:,:3,3]=trans
  for i in range(n):
   bx=T[i,:3,0];bp=np.degrees(np.arctan2(bx[2],bx[0]));theta=wrap(off[i]);X[i,:3,:3]=SR.from_euler('y',theta,degrees=True).as_matrix()@T[i,:3,:3]
  poses[name]=X\"\"\"
new=\"\"\"bx=T[:,:3,0];bphi=np.degrees(np.arctan2(bx[:,2],bx[:,0]))
 for name,off in offs.items():
  X=T.copy();X[:,:3,3]=trans
  for i in range(n):
   theta=wrap(off[i]-bphi[i]);X[i,:3,:3]=SR.from_euler('y',theta,degrees=True).as_matrix()@T[i,:3,:3]
  poses[name]=X\"\"\"
assert old in s
open(p,'w').write(s.replace(old,new))
PY
python /work/scratch/eval_yaw_variants.py"
```
<details><summary>output</summary>

```
base IoU 0.5819 0.5403 tex 0.9461 0.9046 weighted 0.9358 nvis 22
[(0, 0.87, 0.86, 309), (50, 0.86, 0.85, 307), (100, 0.88, 0.83, 308), (150, 0.86, 0.81, 307), (180, 0.35, None, 84), (220, 0.3, None, 63), (240, 0.41, None, 75), (260, 0.39, None, 87), (280, 0.43, 1.07, 117), (300, 0.43, 1.15, 145), (320, 0.48, 1.32, 164), (340, 0.45, 1.0, 179), (360, 0.39, 1.12, 166), (380, 0.41, 1.23, 209), (400, 0.52, 1.01, 270), (420, 0.62, 1.01, 379), (440, 0.63, 0.8, 306), (460, 0.56, 0.85, 214), (700, 0.54, 0.87, 201), (720, 0.54, 0.64, 209), (740, 0.48, 0.93, 160), (760, 0.55, 0.79, 158), (780, 0.63, 0.98, 215), (800, 0.86, 0.89, 345), (830, 0.85, 0.9, 347), (860, 0.85, 0.91, 348)]
hand IoU 0.5758 0.5335 tex 0.9234 0.9044 weighted 0.9169 nvis 22
[(0, 0.87, 0.9, 307), (50, 0.86, 0.91, 305), (100, 0.88, …[+2599 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/theta=wrap(off\\[i\\]-bphi\\[i\\])/theta=wrap(bphi[0]+off[i]-bphi[i])/\" /work/scratch/eval_yaw_variants.py
python /work/scratch/eval_yaw_variants.py"
```
<details><summary>output</summary>

```
base IoU 0.5803 0.5433 tex 0.9715 0.9852 weighted 0.968 nvis 22
[(0, 0.87, 0.92, 310), (50, 0.87, 0.93, 308), (100, 0.88, 0.93, 309), (150, 0.87, 0.92, 308), (180, 0.35, None, 85), (220, 0.29, None, 61), (240, 0.41, None, 73), (260, 0.43, None, 96), (280, 0.5, 1.03, 135), (300, 0.38, 1.0, 137), (320, 0.48, 1.12, 165), (340, 0.44, 1.29, 175), (360, 0.39, 1.02, 166), (380, 0.41, 0.91, 209), (400, 0.52, 0.82, 269), (420, 0.6, 1.1, 367), (440, 0.56, 0.87, 273), (460, 0.57, 1.01, 220), (700, 0.54, 0.82, 198), (720, 0.55, 0.8, 209), (740, 0.48, 0.99, 160), (760, 0.56, 0.99, 160), (780, 0.63, 0.94, 208), (800, 0.84, 1.0, 337), (830, 0.83, 0.99, 337), (860, 0.83, 0.99, 338)]
hand IoU 0.5803 0.5403 tex 0.9076 0.8421 weighted 0.8981 nvis 22
[(0, 0.87, 0.83, 310), (50, 0.87, 0.83, 308), (100, 0.88, 0 …[+2604 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/pooled_static_yaw.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR

def main():
 dev='cuda';H,W=576,768
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl'];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
 cap=cv2.VideoCapture('/work/inputs/video.mp4');keys=[0,30,60,90,120,150,800,820,840,860];frames={};i=0
 while 1:
  ok,x=cap.read()
  if not ok:break
  if i in keys:frames[i]=x
  i+=1
 cap.release()
 with h5py.File('/work/inputs/masks.h5','r') as f:obj=f['object'][:];per=f['person'][:]
 glctx=dr.RasterizeCudaContext();angs=np.arang …[+2027 chars]
```
<details><summary>output</summary>

```
init tex top [(60, 0.8271, 0.8837), (70, 0.8353, 0.8799), (55, 0.8363, 0.8842), (65, 0.8364, 0.8857), (45, 0.8396, 0.8832), (50, 0.8481, 0.8831), (75, 0.8548, 0.8825), (30, 0.8553, 0.8828), (115, 0.8579, 0.879), (90, 0.8616, 0.8768), (85, 0.8617, 0.8754), (95, 0.8645, 0.8792), (40, 0.8676, 0.882), (105, 0.8686, 0.8829), (100, 0.8694, 0.8804)]
init iou top [(-55, 1.0516, 0.8929), (-65, 1.063, 0.8928), (-40, 1.0301, 0.892), (-60, 1.054, 0.8904), (-80, 1.0588, 0.8904), (-70, 1.0728, 0.8902), (-50, 1.0561, 0.8901), (-45, 1.0606, 0.8901), (-30, 1.0303, 0.8901), (-75, 1.0801, 0.8898)]
final tex top [(-35, 0.752, 0.8546), (-15, 0.7546, 0.8532), (-45, 0.7694, 0.8524), (-10, 0.7788, 0.8543), (-50, 0.7829, 0.8497), (-20, 0.7891, 0.8534), (-30, 0.7896, 0.8545), (-25, 0.7898, 0.8557), (-5, 0.7914, 0.8 …[+379 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_yaw_hi.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def wrap(a):return (a+180)%360-180
def main():
 dev='cuda';H,W=576,768
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl'];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);Kp=r['human_keypoints_world']
 rh=Kp[:,21:42].mean(1);lh=Kp[:,42:63].mean(1);u=rh-lh;hang=np.degrees(np.unwrap(np.arctan2(u[:,2],u[:,0])));hang=gaussian_filter1d(hang,sigma=4)
 anchor=180;release=760;hd= …[+3683 chars]
```
<details><summary>output</summary>

```
raw_const IoU 0.6127 0.5783 tex 0.9833 0.9431 weighted 0.9715 angmed/p99/max [  4.19  74.25 171.22]
 tex [(0, 0.91, 711), (50, 0.91, 707), (100, 0.94, 715), (150, 0.94, 709), (200, 1.28, 245), (220, 1.09, 161), (240, 0.84, 194), (260, 1.29, 212), (280, 0.95, 340), (300, 1.03, 323), (320, 1.03, 421), (340, 1.33, 450), (360, 1.06, 430), (380, 0.92, 524), (400, 0.9, 667), (420, 1.04, 880), (440, 0.83, 663), (460, 0.88, 533), (700, 0.9, 466), (720, 0.75, 503), (740, 0.99, 384), (760, 0.91, 403), (780, 1.0, 477), (800, 0.96, 732), (830, 0.94, 744), (860, 0.94, 747)]
raw_hand IoU 0.6151 0.5711 tex 0.9382 0.8644 weighted 0.9227 angmed/p99/max [  4.28  73.98 169.91]
 tex [(0, 0.86, 706), (50, 0.87, 699), (100, 0.83, 710), (150, 0.81, 705), (200, 1.22, 242), (220, 1.07, 178), (240, 0.98, 198), (260 …[+4338 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/eval_yaw_hi.npz');r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];X=z['smooth_hand'];d=np.einsum('fji,fjk->fik',X[:-1,:3,:3],X[1:,:3,:3]);a=np.degrees(R.from_matrix(d).magnitude())
for i in np.argsort(a)[-30:][::-1]:
 print(i,round(a[i],1),'base',round(np.degrees(R.from_matrix(T[i,:3,:3].T@T[i+1,:3,:3]).magnitude()),1),'ay',np.round(T[i,:3,1],3),np.round(T[i+1,:3,1],3))
PY"
```
<details><summary>output</summary>

```
608 169.9 base 85.6 ay [-0.001 -0.999 -0.038] [ 0.01  -1.    -0.026]
654 169.5 base 85.0 ay [ 0.146 -0.985  0.094] [ 0.016 -0.999  0.028]
619 163.6 base 97.8 ay [-0.01  -0.996 -0.084] [ 0.082 -0.995  0.057]
537 140.9 base 71.9 ay [-0.225 -0.973  0.046] [ 0.017 -0.999  0.051]
446 139.1 base 110.3 ay [ 0.025 -0.997 -0.072] [ 0.052 -0.998 -0.026]
437 105.8 base 52.0 ay [-0.006 -0.995 -0.1  ] [ 0.005 -0.994 -0.106]
155 90.6 base 45.3 ay [ 0.008 -1.    -0.012] [ 0.025 -0.999 -0.032]
293 89.3 base 45.1 ay [ 0.048 -0.995 -0.086] [ 0.025 -0.993 -0.115]
424 80.8 base 39.6 ay [ 0.003 -0.995 -0.096] [-0.007 -0.995 -0.1  ]
294 70.8 base 36.1 ay [ 0.025 -0.993 -0.115] [ 0.033 -0.991 -0.129]
161 70.6 base 35.4 ay [ 0.004 -1.    -0.019] [ 0.045 -0.998 -0.04 ]
326 63.5 base 34.6 ay [-0.106 -0.991  0.078]  …[+1247 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/eval_yaw_hi.npz');T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];X=z['smooth_hand']
for i in [154,155,160,436,437,445,446,522,523,536,537,607,608,618,619,653,654]:
 for n,Z in [('T',T),('X',X)]:
  bx=Z[i,:3,0];phi=np.degrees(np.arctan2(bx[2],bx[0]));ay=Z[i,:3,1]
  print(i,n,'phi',round(phi,1),'ay',np.round(ay,3),'R00',round(Z[i,0,0],3),'R20',round(Z[i,2,0],3))
 d=np.einsum('fji,fjk->fik',X[max(0,i-1):i,:3,:3],X[i:i+1,:3,:3]);print('delta from prev',np.degrees(R.from_matrix(d).magnitude()))
PY"
```
<details><summary>output</summary>

```
154 T phi -81.9 ay [ 0.008 -1.    -0.012] R00 0.14 R20 -0.99
154 X phi -141.9 ay [-0.006 -1.    -0.013] R00 -0.787 R20 -0.616
delta from prev [6.87235983e-15]
155 T phi -81.9 ay [ 0.008 -1.    -0.012] R00 0.14 R20 -0.99
155 X phi -141.9 ay [-0.006 -1.    -0.013] R00 -0.787 R20 -0.616
delta from prev [6.87235983e-15]
160 T phi -122.2 ay [ 0.019 -1.    -0.016] R00 -0.533 R20 -0.846
160 X phi 137.6 ay [-0.02  -1.    -0.015] R00 -0.738 R20 0.675
delta from prev [1.09852915]
436 T phi 128.5 ay [-0.006 -0.995 -0.1  ] R00 -0.62 R20 0.781
436 X phi -61.6 ay [ 0.023 -0.995  0.097] R00 0.475 R20 -0.877
delta from prev [2.14759996]
437 T phi 128.5 ay [-0.006 -0.995 -0.1  ] R00 -0.62 R20 0.781
437 X phi -63.6 ay [ 0.027 -0.995  0.096] R00 0.443 R20 -0.893
delta from prev [2.03835853]
445 T phi 59.7 ay …[+1840 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
for p in ['/work/scratch/eval_yaw_hi.py','/work/scratch/eval_yaw_variants.py']:
 s=open(p).read()
 s=s.replace('theta=wrap(target-bphi[i])','theta=wrap(bphi[i]-target)')
 s=s.replace('theta=wrap(bphi[0]+off[i]-bphi[i])','theta=wrap(bphi[i]-(bphi[0]+off[i]))')
 open(p,'w').write(s)
PY
python /work/scratch/eval_yaw_hi.py"
```
<details><summary>output</summary>

```
raw_const IoU 0.6057 0.5826 tex 0.9656 0.9642 weighted 0.9574 angmed/p99/max [ 0.76 11.53 27.11]
 tex [(0, 0.91, 711), (50, 0.91, 707), (100, 0.94, 715), (150, 0.94, 709), (200, 1.18, 244), (220, None, 123), (240, 1.0, 155), (260, 0.98, 200), (280, 0.98, 303), (300, 1.17, 346), (320, 1.0, 417), (340, 1.12, 454), (360, 0.93, 430), (380, 0.83, 524), (400, 0.85, 668), (420, 0.9, 911), (440, 0.98, 654), (460, 0.81, 528), (700, 0.93, 468), (720, 0.88, 500), (740, 1.1, 387), (760, 0.88, 402), (780, 0.98, 482), (800, 0.96, 733), (830, 1.0, 744), (860, 0.99, 747)]
raw_hand IoU 0.6113 0.5772 tex 1.0309 1.0312 weighted 1.0179 angmed/p99/max [ 0.98 11.64 27.17]
 tex [(0, 1.05, 699), (50, 1.06, 702), (100, 1.05, 714), (150, 1.05, 708), (200, 1.33, 245), (220, None, 136), (240, 1.3, 169), (260, 1.19, 2 …[+4307 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_final2.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR
from scipy.ndimage import gaussian_filter1d

def wrap(a):return (a+180)%360-180
def sstep(x):x=np.clip(x,0,1);return x*x*(3-2*x)
def main():
 dev='cuda';H,W=576,768
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl'];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);Kp=r['human_keypoints_world']
 rh=Kp[:,21:42].mean(1);lh=Kp[:,42:63].mean(1);u=rh-lh;hang=np.degrees(np.unwrap(np.arctan2(u[:,2],u[:,0])));hang=gaussian_filter1d(hang,sigma=4)
 # choose static  …[+4358 chars]
```
<details><summary>output</summary>

```
baseline IoU 0.6195 0.5717 tex 0.8847 0.8647 weighted 0.8736 ang [  2.38  37.26 110.26]
raw_linear IoU 0.6064 0.5714 tex 0.8409 0.8235 weighted 0.8359 ang [ 0.68 11.52 27.11]
raw_handres IoU 0.6066 0.5704 tex 0.8599 0.8509 weighted 0.8521 ang [ 0.94 11.55 27.13]
raw_const IoU 0.6085 0.5761 tex 0.8558 0.8544 weighted 0.8521 ang [ 0.67 11.89 91.72]
smooth_linear IoU 0.5999 0.5725 tex 0.83 0.8414 weighted 0.8265 ang [ 0.68 11.52 27.11]
smooth_handres IoU 0.6003 0.5725 tex 0.858 0.8578 weighted 0.8496 ang [ 0.94 11.55 27.13]
smooth_const IoU 0.6013 0.574 tex 0.8645 0.8567 weighted 0.8596 ang [ 0.67 11.89 91.72]
yawonly_linear IoU 0.6078 0.5721 tex 0.8911 0.8643 weighted 0.8756 ang [ 0.1   0.24 76.24]
yawonly_handres IoU 0.6089 0.5714 tex 0.9317 0.9172 weighted 0.9197 ang [ 0.15  2.29 76.13]
ya …[+194 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_final2.py';s=open(p).read()
# replace targets construction block
old=\"\"\"targets={}
 targets['linear']=phis+total*sstep(np.clip((np.arange(870)-200)/(580),0,1))
 targets['handres']=phis-h+residual*beta
 targets['const']=np.full(870,phis)
 for n in targets:targets[n][:200]=phis;targets[n][790:]=phie\"\"\"
new=\"\"\"targets={}
 lin=phis+total*sstep(np.clip((np.arange(870)-200)/(580),0,1))
 for w in [0,.25,.5,.75,1.0]:
  targets[f'w{w:g}']=phis+total*beta+w*(-h+42.421690526219614*beta)
 targets['const']=np.full(870,phis)
 for n in targets:targets[n][:200]=phis;targets[n][790:]=phie\"\"\"
assert old in s;s=s.replace(old,new)
# remove yawonly block to save
start=s.index(\" # yaw-only version\");end=s.index(\" keys=\",start);s=s[:start]+s[end:]
open('/work/scratch/eval_blend.py','w').write(s)
PY
python /work/scratch/eval_blend.py"
```
<details><summary>output</summary>

```
baseline IoU 0.6195 0.5717 tex 0.8847 0.8647 weighted 0.8736 ang [  2.38  37.26 110.26]
raw_w0 IoU 0.6064 0.5714 tex 0.8409 0.8235 weighted 0.8359 ang [ 0.68 11.52 27.11]
raw_w0.25 IoU 0.6065 0.5708 tex 0.8402 0.8216 weighted 0.8361 ang [ 0.72 11.52 27.11]
raw_w0.5 IoU 0.6066 0.5708 tex 0.8351 0.8213 weighted 0.8317 ang [ 0.79 11.52 27.11]
raw_w0.75 IoU 0.6065 0.5711 tex 0.8381 0.8272 weighted 0.8335 ang [ 0.88 11.53 27.12]
raw_w1 IoU 0.6066 0.5704 tex 0.8599 0.8509 weighted 0.8521 ang [ 0.94 11.55 27.13]
raw_const IoU 0.6085 0.5761 tex 0.8558 0.8544 weighted 0.8521 ang [ 0.67 11.89 91.72]
smooth_w0 IoU 0.5999 0.5725 tex 0.83 0.8414 weighted 0.8265 ang [ 0.68 11.52 27.11]
smooth_w0.25 IoU 0.6004 0.5737 tex 0.8381 0.8412 weighted 0.8334 ang [ 0.72 11.52 27.11]
smooth_w0.5 IoU 0.6003 0.5722  …[+424 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_blend.py';s=open(p).read()
# inject helper after imports
s=s.replace(\"from scipy.ndimage import gaussian_filter1d\", \"from scipy.ndimage import gaussian_filter1d\")
# replace pose construction loop with only w0.5 plus tilt variants
start=s.index(\" bx=T[:,:3,0];bphi=\")
end=s.index(\" keys=\",start)
new=''' bx=T[:,:3,0];bphi=np.degrees(np.arctan2(bx[:,2],bx[:,0]))
 targets={}
 total_hand=42.421690526219614
 target=phis+total*beta+0.5*(-h+total_hand*beta);target[:200]=phis;target[790:]=phie
 # baseline tilt with yaw removed
 Rt=[];qs=[]
 for i in range(870):
  Rti=SR.from_euler('y',bphi[i],degrees=True).as_matrix()@T[i,:3,:3]
  Rt.append(Rti);qs.append(SR.from_matrix(Rti).as_quat())
 qs=np.array(qs)
 for i in range(1,len(qs)):
  if np.dot(qs[i-1],qs[i])<0:qs[i]*=-1
 poses={'baseline':T}
 for trname,pos in [('raw',praw),('smooth',psm)]:
  for sig in [0,1,2,4,8]:
   if sig==0:q=qs.copy()
   else:
    q=gaussian_filter1d(qs,sig,axis=0,mode='nearest')
   q/=np.linalg.norm(q,axis=1,keepdims=True);Rs=SR.from_quat(q).as_matrix()
   X=T.copy();X[:,:3,3]=pos
   for i in range(870):
    X[i,:3,:3]=SR.from_euler('y',-target[i],degrees=True).as_matrix()@Rs[i]
   X[:200,:3,:3]=Rstart;X[790:,:3,:3]=Rend;X[:200,:3,3]=p0;X[790:,:3,3]=p1
   poses[trname+f'_tilt{sig}']=X
'''
s=s[:start]+new+s[end:]
s=s.replace(\"eval_final2.npz\",\"eval_tilt.npz\")
open('/work/scratch/eval_tilt.py','w').write(s)
PY
python /work/scratch/eval_tilt.py"
```
<details><summary>output</summary>

```
baseline IoU 0.6195 0.5717 tex 0.8847 0.8647 weighted 0.8736 ang [  2.38  37.26 110.26]
raw_tilt0 IoU 0.6066 0.5708 tex 0.8351 0.8213 weighted 0.8317 ang [ 0.79 11.52 27.11]
raw_tilt1 IoU 0.6072 0.5705 tex 0.8418 0.8363 weighted 0.8398 ang [0.48 4.14 8.17]
raw_tilt2 IoU 0.6076 0.5693 tex 0.8418 0.8437 weighted 0.8394 ang [0.34 2.39 3.91]
raw_tilt4 IoU 0.6083 0.5709 tex 0.8479 0.85 weighted 0.8419 ang [0.26 1.34 3.52]
raw_tilt8 IoU 0.6092 0.5679 tex 0.8508 0.85 weighted 0.844 ang [0.21 1.25 3.08]
smooth_tilt0 IoU 0.6003 0.5722 tex 0.8356 0.8234 weighted 0.831 ang [ 0.79 11.52 27.11]
smooth_tilt1 IoU 0.6011 0.5716 tex 0.8278 0.8356 weighted 0.8249 ang [0.48 4.14 8.17]
smooth_tilt2 IoU 0.601 0.571 tex 0.8405 0.8453 weighted 0.8182 ang [0.34 2.39 3.91]
smooth_tilt4 IoU 0.6023 0.57 tex 0.8312 0 …[+234 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_tilt.py';s=open(p).read()
old=\"\"\"for trname,pos in [('raw',praw),('smooth',psm)]:
  for sig in [0,1,2,4,8]:
   if sig==0:q=qs.copy()
   else:
    q=gaussian_filter1d(qs,sig,axis=0,mode='nearest')
   q/=np.linalg.norm(q,axis=1,keepdims=True);Rs=SR.from_quat(q).as_matrix()
   X=T.copy();X[:,:3,3]=pos
   for i in range(870):
    X[i,:3,:3]=SR.from_euler('y',-target[i],degrees=True).as_matrix()@Rs[i]
   X[:200,:3,:3]=Rstart;X[790:,:3,:3]=Rend;X[:200,:3,3]=p0;X[790:,:3,3]=p1
   poses[trname+f'_tilt{sig}']=X\"\"\"
new=\"\"\"q=gaussian_filter1d(qs,2,axis=0,mode='nearest');q/=np.linalg.norm(q,axis=1,keepdims=True);Rs=SR.from_quat(q).as_matrix()
 for w in [0,.2,.3,.4,.5,.6,.7,.8,1.0]:
  tg=phis+total*beta+w*(-h+total_hand*beta);tg[:200]=phis;tg[790:]=phie
  X=T.copy();X[:,:3,3]=psm
  for i in range(870):X[i,:3,:3]=SR.from_euler('y',-tg[i],degrees=True).as_matrix()@Rs[i]
  X[:200,:3,:3]=Rstart;X[790:,:3,:3]=Rend;X[:200,:3,3]=p0;X[790:,:3,3]=p1
  poses[f'w{w:g}']=X\"\"\"
assert old in s;s=s.replace(old,new).replace(\"eval_tilt.npz\",\"eval_wfine.npz\")
open('/work/scratch/eval_wfine.py','w').write(s)
PY
python /work/scratch/eval_wfine.py"
```
<details><summary>output</summary>

```
baseline IoU 0.6195 0.5717 tex 0.8847 0.8647 weighted 0.8736 ang [  2.38  37.26 110.26]
w0 IoU 0.6007 0.572 tex 0.8259 0.8106 weighted 0.805 ang [0.29 2.4  3.9 ]
w0.2 IoU 0.6007 0.5713 tex 0.837 0.817 weighted 0.8152 ang [0.31 2.39 3.9 ]
w0.3 IoU 0.6009 0.571 tex 0.8385 0.8255 weighted 0.8165 ang [0.32 2.39 3.9 ]
w0.4 IoU 0.6008 0.5701 tex 0.8294 0.813 weighted 0.8085 ang [0.33 2.39 3.91]
w0.5 IoU 0.601 0.571 tex 0.8405 0.8453 weighted 0.8182 ang [0.34 2.39 3.91]
w0.6 IoU 0.6011 0.5709 tex 0.8317 0.8178 weighted 0.8104 ang [0.35 2.39 3.92]
w0.7 IoU 0.6013 0.5712 tex 0.835 0.8411 weighted 0.8332 ang [0.37 2.39 3.92]
w0.8 IoU 0.6017 0.5716 tex 0.8312 0.8266 weighted 0.8288 ang [0.39 2.39 3.93]
w1 IoU 0.6015 0.5703 tex 0.8501 0.85 weighted 0.846 ang [0.41 2.5  3.95]
targets -141.9395419289214 …[+79 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_wfine.py';s=open(p).read()
old=\"for w in [0,.2,.3,.4,.5,.6,.7,.8,1.0]:\\n  tg=phis+total*beta+w*(-h+total_hand*beta);tg[:200]=phis;tg[790:]=phie\"
new=\"traj={}\\n traj['linear']=phis+total*beta\\n for w in [.2,.4,.6,.8]:traj[f'blend{w:g}']=phis+total*beta+w*(-h+total_hand*beta)\\n traj['handnorm']=phis+total*(h/total_hand);traj['handnorm_clip']=phis+total*np.clip(h/total_hand,0,1)\\n for wname,tg in traj.items():\\n  tg[:200]=phis;tg[790:]=phie\"
assert old in s
s=s.replace(old,new).replace(\"poses[f'w{w:g}']=X\",\"poses[wname]=X\").replace(\"eval_wfine.npz\",\"eval_wshape.npz\")
open('/work/scratch/eval_wshape.py','w').write(s)
PY
python /work/scratch/eval_wshape.py"
```
<details><summary>output</summary>

```
baseline IoU 0.6195 0.5717 tex 0.8847 0.8647 weighted 0.8736 ang [  2.38  37.26 110.26]
linear IoU 0.6007 0.572 tex 0.8259 0.8106 weighted 0.805 ang [0.29 2.4  3.9 ]
blend0.2 IoU 0.6007 0.5713 tex 0.837 0.817 weighted 0.8152 ang [0.31 2.39 3.9 ]
blend0.4 IoU 0.6008 0.5701 tex 0.8294 0.813 weighted 0.8085 ang [0.33 2.39 3.91]
blend0.6 IoU 0.6011 0.5709 tex 0.8317 0.8178 weighted 0.8104 ang [0.35 2.39 3.92]
blend0.8 IoU 0.6017 0.5716 tex 0.8312 0.8266 weighted 0.8288 ang [0.39 2.39 3.93]
handnorm IoU 0.6021 0.5712 tex 0.9378 0.8845 weighted 0.9335 ang [0.58 4.47 5.08]
handnorm_clip IoU 0.6011 0.5704 tex 0.8455 0.84 weighted 0.8406 ang [0.28 3.22 4.11]
targets -141.93954192892144 126.36659350138741 -91.69386456969113 42.421690526219614 -49.272174043471516

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_tilt.py';s=open(p).read()
# replace pose construction section entirely
start=s.index(\" # baseline tilt with yaw removed\")
end=s.index(\" keys=\",start)
new=''' # baseline tilt with yaw removed
 Rt=[];qs=[]
 for i in range(870):
  Rti=SR.from_euler('y',bphi[i],degrees=True).as_matrix()@T[i,:3,:3]
  Rt.append(Rti);qs.append(SR.from_matrix(Rti).as_quat())
 qs=np.array(qs)
 for i in range(1,len(qs)):
  if np.dot(qs[i-1],qs[i])<0:qs[i]*=-1
 q=gaussian_filter1d(qs,2,axis=0,mode='nearest');q/=np.linalg.norm(q,axis=1,keepdims=True);Rs=SR.from_quat(q).as_matrix()
 target=phis+total*beta;target[:200]=phis;target[790:]=phie
 poses={'baseline':T}
 for sig in [0,1,2,3,5,8]:
  ps=gaussian_filter1d(praw,sig,axis=0,mode='nearest') if sig>0 else praw
  p0=np.median(praw[:195],0);p1=np.median(praw[790:],0)
  pos=ps.copy();pos[:195]=p0;pos[810:]=p1
  a=sstep((np.arange(870)-195)/25);pos[:220]=p0[None]*(1-a[:220,None])+pos[:220]*a[:220,None]
  a=sstep((np.arange(870)-760)/50);pos[760:]=pos[760:]*(1-a[760:,None])+p1[None]*a[760:,None]
  X=T.copy();X[:,:3,3]=pos
  for i in range(870):X[i,:3,:3]=SR.from_euler('y',-target[i],degrees=True).as_matrix()@Rs[i]
  X[:200,:3,:3]=Rstart;X[790:,:3,:3]=Rend;X[:200,:3,3]=p0;X[790:,:3,3]=p1
  poses[f'psig{sig}']=X
'''
s=s[:start]+new+s[end:];s=s.replace(\"eval_tilt.npz\",\"eval_trans.npz\")
open('/work/scratch/eval_trans.py','w').write(s)
PY
python /work/scratch/eval_trans.py"
```
<details><summary>output</summary>

```
baseline IoU 0.6195 0.5717 tex 0.8847 0.8647 weighted 0.8736 ang [  2.38  37.26 110.26]
psig0 IoU 0.6008 0.5702 tex 0.8251 0.819 weighted 0.8231 ang [0.29 2.4  3.9 ]
psig1 IoU 0.5994 0.5736 tex 0.8321 0.8133 weighted 0.8291 ang [0.29 2.4  3.9 ]
psig2 IoU 0.6007 0.572 tex 0.8259 0.8106 weighted 0.805 ang [0.29 2.4  3.9 ]
psig3 IoU 0.6017 0.5707 tex 0.8283 0.8305 weighted 0.8252 ang [0.29 2.4  3.9 ]
psig5 IoU 0.6029 0.5694 tex 0.8224 0.836 weighted 0.8013 ang [0.29 2.4  3.9 ]
psig8 IoU 0.6002 0.558 tex 0.8347 0.85 weighted 0.8282 ang [0.29 2.4  3.9 ]
targets -141.93954192892144 126.36659350138741 -91.69386456969113 42.421690526219614 -49.272174043471516

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_local_score.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR
from scipy.ndimage import gaussian_filter1d

def wrap(a):return (a+180)%360-180
def sstep(x):x=np.clip(x,0,1);return x*x*(3-2*x)
def main():
 dev='cuda';H,W=576,768
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl'];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float)
 Rstart=SR.from_euler('y',60,degrees=True).as_matrix()@T[0,:3,:3];Rend=SR.from_euler('y',-30,degrees=True).as_matrix()@T[-1,:3,:3]
 phis=np.degrees(np.arctan2(Rstart[2,0],Rstart[0,0]));phie …[+3331 chars]
```
<details><summary>output</summary>

```
0 n 715 tex [(10, 0.828, 0.879), (-5, 0.844, 0.877), (-10, 0.846, 0.877), (5, 0.847, 0.88), (15, 0.847, 0.877)] comb [(10, 0.828, 0.879), (5, 0.847, 0.88), (-5, 0.844, 0.877), (-10, 0.846, 0.877), (15, 0.847, 0.877)]
50 n 708 tex [(10, 0.838, 0.879), (-10, 0.854, 0.875), (-5, 0.855, 0.875), (15, 0.855, 0.878), (5, 0.855, 0.88)] comb [(10, 0.838, 0.879), (5, 0.855, 0.88), (15, 0.855, 0.878), (-10, 0.854, 0.875), (-5, 0.855, 0.875)]
100 n 707 tex [(-10, 0.824, 0.883), (-5, 0.832, 0.88), (-25, 0.833, 0.88), (10, 0.836, 0.882), (-55, 0.848, 0.879)] comb [(-10, 0.824, 0.883), (-5, 0.832, 0.88), (10, 0.836, 0.882), (-25, 0.833, 0.88), (5, 0.849, 0.883)]
150 n 701 tex [(10, 0.794, 0.866), (-10, 0.795, 0.867), (-5, 0.8, 0.867), (-25, 0.804, 0.868), (5, 0.805, 0.867)] comb [(-10, 0.795, 0.867), (10 …[+5417 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.ndimage import gaussian_filter1d
z=np.load('/work/scratch/yaw_local_score.npz');keys=z['keys'];ad=z['adeltas'];R=z['results'] # (nkeys,nang,4)
est=[];conf=[]
for k,r in zip(keys,R):
 s=r[:,1];n=r[:,3];v=np.isfinite(s)
 if n.max()<150 or v.sum()<5:est.append(0);conf.append(0);continue
 mn=np.nanmin(s);w=np.exp(-(s-mn)/0.05);w[~v]=0;e=(w*ad).sum()/w.sum();est.append(e);conf.append(min(1,n.max()/500))
print('raw est',[(k,round(e,1),round(c,2)) for k,e,c in zip(keys,est,conf)])
# interpolate full and smooth sigma values
for sig in [10,20,30,40,60]:
 x=np.interp(np.arange(870),keys,est);c=np.interp(np.arange(870),keys,conf);x=gaussian_filter1d(x,sig)
 print('sig',sig,'samples',[(k,round(x[k],1)) for k in keys])
 np.save(f'/work/scratch/adj_sig{sig}.npy',x)
PY"
```
<details><summary>output</summary>

```
raw est [(0, -1.4, 1), (50, -4.6, 1), (100, -9.4, 1), (150, -6.9, 1), (180, -33.0, 0.45), (200, 36.0, 0.48), (220, -45.8, 0.35), (240, -22.7, 0.35), (260, -29.0, 0.42), (280, -3.2, 0.61), (300, -33.5, 0.7), (320, -16.5, 0.83), (340, -38.2, 0.91), (360, -11.4, 0.86), (380, -15.6, 1), (400, 9.3, 1), (420, -19.5, 1), (440, -17.8, 1), (460, -11.3, 1), (660, 0, 0), (700, 6.8, 0.97), (720, 23.9, 1), (740, -12.7, 0.78), (760, 3.1, 0.8), (780, -3.4, 1), (800, -2.1, 1), (830, 0.3, 1), (860, -2.0, 1)]
sig 10 samples [(0, -1.9), (50, -4.7), (100, -8.8), (150, -10.6), (180, -16.4), (200, 6.7), (220, -25.6), (240, -28.0), (260, -22.9), (280, -14.0), (300, -24.4), (320, -23.8), (340, -28.8), (360, -17.3), (380, -10.2), (400, -1.1), (420, -13.6), (440, -16.8), (460, -12.3), (660, 0.5), (700, 9.3), (720,  …[+1635 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_trans.py';s=open(p).read()
# replace pose loop after q calculations
start=s.index(\" poses={'baseline':T}\")
end=s.index(\" keys=\",start)
new=''' poses={'baseline':T}
 adj=np.load('/work/scratch/adj_sig30.npy');adj[:200]=0;adj[670:]=0
 for name,add in [('linear',np.zeros(870)),('adj30',adj)]:
  tg=target+add;tg[:200]=phis;tg[790:]=phie
  p0=np.median(praw[:195],0);p1=np.median(praw[790:],0);ps=gaussian_filter1d(praw,2,axis=0,mode='nearest');pos=ps.copy();pos[:195]=p0;pos[810:]=p1
  a=sstep((np.arange(870)-195)/25);pos[:220]=p0[None]*(1-a[:220,None])+pos[:220]*a[:220,None];a=sstep((np.arange(870)-760)/50);pos[760:]=pos[760:]*(1-a[760:,None])+p1[None]*a[760:,None]
  X=T.copy();X[:,:3,3]=pos
  for i in range(870):X[i,:3,:3]=SR.from_euler('y',-tg[i],degrees=True).as_matrix()@Rs[i]
  X[:200,:3,:3]=Rstart;X[790:,:3,:3]=Rend;X[:200,:3,3]=p0;X[790:,:3,3]=p1
  poses[name]=X
'''
s=s[:start]+new+s[end:]
# replace keys with heldout
s=s.replace(\"keys=[0,50,100,150,200,220,240,260,280,300,320,340,360,380,400,420,440,460,700,720,740,760,780,800,830,860]\",\"keys=[0,50,100,150,210,250,270,290,310,330,350,370,390,410,430,450,650,680,690,710,730,750,770,800,830,860]\")
s=s.replace(\"eval_trans.npz\",\"eval_adj.npz\")
open('/work/scratch/eval_adj.py','w').write(s)
PY
python /work/scratch/eval_adj.py"
```
<details><summary>output</summary>

```
baseline IoU 0.5707 0.5545 tex 0.868 0.8764 weighted 0.8562 ang [  2.38  37.26 110.26]
linear IoU 0.5581 0.558 tex 0.8274 0.818 weighted 0.8254 ang [0.29 2.4  3.9 ]
adj30 IoU 0.5568 0.558 tex 0.8129 0.7965 weighted 0.8079 ang [ 0.28  2.53 13.55]
targets -141.93954192892144 126.36659350138741 -91.69386456969113 42.421690526219614 -49.272174043471516

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_adj.py';s=open(p).read()
old=\"\"\"adj=np.load('/work/scratch/adj_sig30.npy');adj[:200]=0;adj[670:]=0
 for name,add in [('linear',np.zeros(870)),('adj30',adj)]:\"\"\"
new=\"\"\"trajs={'linear':np.zeros(870)}
 t=np.arange(870)
 for cen in [300,350,400]:
  for sig in [80,120]:trajs[f'p{cen}s{sig}']=-23*np.exp(-0.5*((t-cen)/sig)**2)
 for name,add in trajs.items():\"\"\"
assert old in s;s=s.replace(old,new).replace(\"eval_adj.npz\",\"eval_pulse.npz\")
open('/work/scratch/eval_pulse.py','w').write(s)
PY
python /work/scratch/eval_pulse.py"
```
<details><summary>output</summary>

```
baseline IoU 0.5707 0.5545 tex 0.868 0.8764 weighted 0.8562 ang [  2.38  37.26 110.26]
linear IoU 0.5581 0.558 tex 0.8274 0.818 weighted 0.8254 ang [0.29 2.4  3.9 ]
p300s80 IoU 0.5568 0.558 tex 0.8101 0.797 weighted 0.8071 ang [ 0.27  2.4  11.23]
p300s120 IoU 0.5569 0.5586 tex 0.8078 0.7932 weighted 0.8032 ang [ 0.26  2.4  16.72]
p350s80 IoU 0.5568 0.558 tex 0.8105 0.7969 weighted 0.8055 ang [0.28 2.4  5.57]
p350s120 IoU 0.5566 0.5579 tex 0.8054 0.7971 weighted 0.8007 ang [ 0.27  2.39 11.23]
p400s80 IoU 0.5573 0.558 tex 0.7989 0.7924 weighted 0.7967 ang [0.29 2.39 4.03]
p400s120 IoU 0.557 0.5582 tex 0.8064 0.7914 weighted 0.801 ang [0.28 2.39 6.94]
targets -141.93954192892144 126.36659350138741 -91.69386456969113 42.421690526219614 -49.272174043471516

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_pulse.py';s=open(p).read()
old=\"\"\"trajs={'linear':np.zeros(870)}
 t=np.arange(870)
 for cen in [300,350,400]:
  for sig in [80,120]:trajs[f'p{cen}s{sig}']=-23*np.exp(-0.5*((t-cen)/sig)**2)
 for name,add in trajs.items():\"\"\"
new=\"\"\"trajs={'linear':np.zeros(870)}
 t=np.arange(870)
 for amp in [-10,-15,-20,-25,-30,-35]:trajs[f'a{amp}']=amp*np.exp(-0.5*((t-400)/80)**2)
 for name,add in trajs.items():\"\"\"
assert old in s;s=s.replace(old,new).replace(\"eval_pulse.npz\",\"eval_amp.npz\")
open('/work/scratch/eval_amp.py','w').write(s)
PY
python /work/scratch/eval_amp.py"
```
<details><summary>output</summary>

```
baseline IoU 0.5707 0.5545 tex 0.868 0.8764 weighted 0.8562 ang [  2.38  37.26 110.26]
linear IoU 0.5581 0.558 tex 0.8274 0.818 weighted 0.8254 ang [0.29 2.4  3.9 ]
a-10 IoU 0.5575 0.558 tex 0.8119 0.7945 weighted 0.8082 ang [0.29 2.4  3.92]
a-15 IoU 0.5575 0.558 tex 0.8146 0.7927 weighted 0.8085 ang [0.29 2.39 3.95]
a-20 IoU 0.5572 0.558 tex 0.8103 0.794 weighted 0.8046 ang [0.29 2.39 4.  ]
a-25 IoU 0.5573 0.558 tex 0.794 0.7944 weighted 0.7913 ang [0.29 2.39 4.05]
a-30 IoU 0.5569 0.558 tex 0.7956 0.7923 weighted 0.7917 ang [0.29 2.39 4.12]
a-35 IoU 0.5564 0.558 tex 0.8039 0.7906 weighted 0.7972 ang [0.29 2.39 4.19]
targets -141.93954192892144 126.36659350138741 -91.69386456969113 42.421690526219614 -49.272174043471516

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_amp.py';s=open(p).read()
# keep only linear + a-25
s=s.replace(\"for amp in [-10,-15,-20,-25,-30,-35]:trajs[f'a{amp}']=amp*np.exp(-0.5*((t-400)/80)**2)\",\"trajs['pulse25']=-25*np.exp(-0.5*((t-400)/80)**2)\")
s=s.replace(\"keys=[0,50,100,150,210,250,270,290,310,330,350,370,390,410,430,450,650,680,690,710,730,750,770,800,830,860]\",\"keys=[0,50,100,150,205,225,245,265,285,305,325,345,365,385,405,425,445,465,650,670,690,710,730,750,770,800,830,860]\")
s=s.replace(\"eval_amp.npz\",\"eval_pulse_holdout.npz\")
open('/work/scratch/eval_pulse_holdout.py','w').write(s)
PY
python /work/scratch/eval_pulse_holdout.py"
```
<details><summary>output</summary>

```
baseline IoU 0.5517 0.5291 tex 0.8961 0.8937 weighted 0.8867 ang [  2.38  37.26 110.26]
linear IoU 0.5354 0.5276 tex 0.8207 0.8079 weighted 0.8207 ang [0.29 2.4  3.9 ]
pulse25 IoU 0.5344 0.5251 tex 0.8137 0.8265 weighted 0.813 ang [0.29 2.39 4.05]
targets -141.93954192892144 126.36659350138741 -91.69386456969113 42.421690526219614 -49.272174043471516

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/outputs
cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent world-space 6-DoF trajectory for the bowl.

The baseline (CoCoNet/FoundationPose + calibrated depth) supplies metric
translation and per-frame bowl-axis tilt.  Its yaw about the bowl'"'s near
symmetry axis is unstable, so yaw is rebuilt from two pooled static anchors
and a smooth manipulation-phase interpolation.  The interpolation is lightly
timed by the reconstructed two-hand grasp direction.

Run:
    python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
\"\"\"
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def wrap_deg(x: np.ndarray | float) -> np.ndarray | float:
    return (x + 180.0) % 360.0 - 180.0


def yaw_matrix(angle_deg: np.ndarray | float) -> np.ndarray:
    return Rotation.from_euler(\"y\", angle_deg, degrees=True).as_matrix()


def synthetic_yaw(phi_deg: np.ndarray) -> np.ndarray:
    \"\"\"Rotation about world +y having the same convention as phi=atan2(z,x).\"\"\"
    # Net rotation is about -y because R_y(a) maps atan2(z,x) to phi-a.
    return np.stack([yaw_matrix(-p) for p in phi_deg], axis=0)


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: floa …[+7359 chars]
```
<details><summary>output</summary>

```
Wrote /work/outputs/object_poses_world.npz with T_world_mesh=(870, 4, 4)
['T_world_mesh'] (870, 4, 4) float64 True
orth err 8.881784197001252e-16 det 0.9999999999999994 1.0000000000000009

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/validate_final.py <<'PY'
import numpy as np,torch,trimesh,cv2,h5py,json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as SR

def main():
 dev='cuda';H,W=576,768
 sc=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=sc.geometry['redbowl'];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));tex=torch.tensor(tex,device=dev,dtype=torch.float)[None]/255
 v=torch.tensor(m.vertices,device=dev,dtype=torch.float);fc=torch.tensor(m.faces,device=dev,dtype=torch.int32);uv=torch.tensor(m.visual.uv.copy(),device=dev,dtype=torch.float);uv[:,1]=1-uv[:,1];V=torch.cat([v,torch.ones(len(v),1,device=dev)],1)
 with open('/work/inputs/camera.json') as f:c=json.load(f)
 K=np.array(c['K']);K[0]*=W/1536;K[1]*=H/1152;nc=.001;far=100.;dz=far-nc;q=-(far+nc)/dz;qn=-2*far*nc/dz
 proj=np.array([[2*K[0,0]/W,-2*K[0,1]/W,(-2*K[0,2]+W)/W,0],[0,2*K[1,1]/H,(2*K[1,2]-H)/H,0],[0,0,q,qn],[0,0,-1,0]],np.float32);gl=np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],np.float32);Ti=np.linalg.inv(np.array(c['T_world_cam']))
 rb=np.load('/work/inputs/reconstruction.npz');base=rb['T_world_mesh'].astype(float);res=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
 poses={'baseline':base,'result':res}
 keys=np.unique(np.r_[np.arange(0,870,15),np.array([0,100,150,199,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740,760,780,789,790,800,830,860,8 …[+4274 chars]
```
<details><summary>output</summary>

```
PHASE METRICS
rest_on_chair
baseline n 16 IoU mean 0.7778 median 0.8856 texture weighted 0.9185 visible pix median 710
result n 16 IoU mean 0.766 median 0.8787 texture weighted 0.8634 visible pix median 702
lift_and_carry
baseline n 26 IoU mean 0.4996 median 0.4941 texture weighted 0.8714 visible pix median 423
result n 26 IoU mean 0.4482 median 0.4671 texture weighted 0.8362 visible pix median 420
lower_slide_release
baseline n 34 IoU mean 0.2395 median 0.1157 texture weighted 0.8291 visible pix median 55
result n 34 IoU mean 0.2378 median 0.0825 texture weighted 0.7575 visible pix median 35
rest_on_floor
baseline n 10 IoU mean 0.8506 median 0.8526 texture weighted 0.8092 visible pix median 796
result n 10 IoU mean 0.8532 median 0.8545 texture weighted 0.792 visible pix median 796

OVERAL …[+1180 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s#r\\['T_world_mesh'\\]; T=r\\['T_world_mesh'\\]#r=np.load('/work/outputs/object_poses_world.npz'); T=r['T_world_mesh']#\" /work/scratch/overlay_baseline.py > /work/scratch/overlay_result.py
# The original line is r=np.load(...); T=r[...] on same line; inspect and force replace
python - <<'PY'
p='/work/scratch/overlay_result.py';s=open(p).read();s=s.replace(\"r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']\",\"r=np.load('/work/outputs/object_poses_world.npz'); T=r['T_world_mesh']\").replace(\"baseline_overlay.jpg\",\"result_overlay.jpg\");open(p,'w').write(s)
PY
grep -n \"np.load('/work/outputs\\|result_overlay\" /work/scratch/overlay_result.py
python /work/scratch/overlay_result.py"
```
<details><summary>output</summary>

```
(1920, 1280, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffmpeg -version | head -1; python --version; git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true; find /work/tools -maxdepth 2 -type f 2>/dev/null | head'
```
<details><summary>output</summary>

```
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
Python 3.11.10
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "/usr/bin/time -f '%e seconds %M KB' python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,h5py,PIL,cv2,torch,trimesh,nvdiffrast
for n,x in [('numpy',numpy),('scipy',scipy),('h5py',h5py),('Pillow',PIL),('opencv-python',cv2),('torch',torch),('trimesh',trimesh),('nvdiffrast',nvdiffrast)]:print(n,getattr(x,'__version__','unknown'))
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
h5py 3.16.0
Pillow 10.2.0
opencv-python 4.11.0
torch 2.5.1+cu124
trimesh 5.1.1
nvdiffrast 0.4.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "TIMEFORMAT='%3R seconds'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
Wrote /work/outputs/object_poses_world.npz with T_world_mesh=(870, 4, 4)
0.421 seconds

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
b=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);r=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
dR=np.einsum('fji,fjk->fik',b[:,:3,:3],r[:,:3,:3]);ang=np.degrees(R.from_matrix(dR).magnitude());dt=np.linalg.norm(r[:,:3,3]-b[:,:3,3],axis=1)
for a,bnd in [(0,199),(200,459),(460,789),(790,869)]:
 print(a,bnd,'rot diff med/95/max',np.quantile(ang[a:bnd+1],[.5,.95,1]),'trans diff med/95/max',np.quantile(dt[a:bnd+1],[.5,.95,1]))
print('overall',np.quantile(ang,[.5,.95,1]),np.quantile(dt,[.5,.95,1]))
# final pose details and static median
for i in [0,100,199,200,400,600,760,789,790,860]:
 print(i,'p',np.round(r[i,:3,3],4),'local y',np.round(r[i,:3,1],4))
PY"
```
<details><summary>output</summary>

```
0 199 rot diff med/95/max [55.         77.74606725 85.37047271] trans diff med/95/max [0.02236146 0.04915315 0.06277287]
200 459 rot diff med/95/max [ 81.58127005 166.18340733 179.94484408] trans diff med/95/max [0.00876101 0.10743842 0.13078888]
460 789 rot diff med/95/max [ 69.32219115 149.53176653 167.24391904] trans diff med/95/max [0.00773381 0.02634231 0.05346831]
790 869 rot diff med/95/max [30. 30. 30.] trans diff med/95/max [0.00378044 0.04873335 0.05997262]
overall [ 55.         151.28709498 179.94484408] [0.00886998 0.04844501 0.13078888]
0 p [0.1269 0.6788 3.1422] local y [-0.0053 -0.9999 -0.0137]
100 p [0.1269 0.6788 3.1422] local y [-0.0053 -0.9999 -0.0137]
199 p [0.1263 0.6776 3.1431] local y [-0.0053 -0.9999 -0.0137]
200 p [0.1259 0.6766 3.1435] local y [-0.0053 -0.9999 -0. …[+390 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).geometry['redbowl']
v=m.vertices[::20]
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);d=json.load(open('/work/inputs/ground_plane.json'))['plane'][3]
for name,path in [('base','/work/inputs/reconstruction.npz'),('result','/work/outputs/object_poses_world.npz')]:
 T=np.load(path)['T_world_mesh'][-1];p=(T[:3,:3]@v.T).T+T[:3,3];sd=p@n+d
 print(name,'plane signed dist min/q1/median/max',np.quantile(sd,[0,.01,.05,.1,.5,1]),'num below',np.sum(sd<0))
PY"
```
<details><summary>output</summary>

```
base plane signed dist min/q1/median/max [-0.02609978 -0.02546268 -0.02301615 -0.02042334  0.041987    0.10722794] num below 4152
result plane signed dist min/q1/median/max [-0.03122798 -0.02966151 -0.02632333 -0.02385292  0.03841616  0.1079238 ] num below 4532

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /work/outputs/object_poses_world.npz > /work/scratch/hash1 && python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz >/dev/null && sha256sum /work/outputs/object_poses_world.npz > /work/scratch/hash2 && cat /work/scratch/hash1 /work/scratch/hash2 && cmp /work/scratch/hash1 /work/scratch/hash2 && echo DETERMINISTIC'
```
<details><summary>output</summary>

```
5dbb4b4de989d0d7fbae2b1f031a3924bdff3b4a00b6d8c5ef60467c134b4a02  /work/outputs/object_poses_world.npz
5dbb4b4de989d0d7fbae2b1f031a3924bdff3b4a00b6d8c5ef60467c134b4a02  /work/outputs/object_poses_world.npz
DETERMINISTIC

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Bowl Trajectory Recovery Report

## Summary

I reconstructed a temporally consistent world-space trajectory for all 870 frames and wrote it to
"'`object_poses_world.npz` under `T_world_mesh` (float64, shape `(870,4,4)`).

The baseline translations are already depth-calibrated and were retained as the primary metric
translation evidence. The dominant failure is yaw about the bowl'"'s near-symmetry axis: the baseline
changes yaw abruptly (up to 110.3 degrees between adjacent frames) while the cup remains upright and
the hands move smoothly. I replaced that yaw with a static-anchor + smooth manipulation trajectory,
kept a low-passed version of the baseline's measured upright-axis tilt, smoothed translation, and
pooled each static phase to one pose.

No ground-truth annotation or challenge trajectory data was used.

## Phases found

| Phase | Frames | Evidence and treatment |
|---|---:|---|
| Rest on chair | 0-199 | Bowl mask and projected pose are effectively stationary. Both-hand contact begins around frame 166, but the bowl is still resting and does not move until about frame 200. All poses are set to one pooled pose. |
| Lift and carry | 200-459 | Contact probabilities are 1 for both hands, object mask moves steadily, and the person carries the bowl away from the chair. Translation is smoothed with a short Gaussian filter; yaw follows a smooth anchored path. |
| Lower, slide, and release | 460-789 | Heavy hand/body occl …[+6851 chars]
```
<details><summary>output</summary>

```
   95 /work/outputs/REPORT.md
   39 /work/outputs/TOOLS.md
  196 /work/outputs/solve.py
  330 total
total 136K
-rw-rw-r-- 1 hanzhou hanzhou 6.1K Oct 20 20:47 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.9K Oct 20 20:47 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 110K Oct 20 20:47 object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou 8.2K Oct 20 20:45 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py /work/outputs/REPORT.md /work/outputs/TOOLS.md && python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz'
z=np.load(p)
assert z.files==['T_world_mesh']
T=z['T_world_mesh']
assert T.shape==(870,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
assert np.allclose(T[:,3,:],np.array([0,0,0,1.]))
print('FINAL OUTPUT OK')
PY"
```
<details><summary>output</summary>

```

```
</details>

**Agent:** Completed the task. Required outputs are in `/work/outputs`:

- `object_poses_world.npz` — `T_world_mesh`, float64, shape `(870, 4, 4)`
- `solve.py` — deterministic offline solver
- `REPORT.md` — phases, method, validation tables, confidence, runtime
- `TOOLS.md` — tools, versions, revisions, and runtime dependencies

Final solver runtime: approximately **0.42 seconds** on CPU. Output validation passed, including finite values, orthonormal rotations, and deterministic regeneration.
