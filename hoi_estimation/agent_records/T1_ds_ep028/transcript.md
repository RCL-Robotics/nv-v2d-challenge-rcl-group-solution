# T1_ds_ep028

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T13:41:15Z

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
and translation both matter; a  …[+9786 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,180p' /work/AGENT_TASK.md"
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
and translation both matter; a  …[+9786 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '35,95p'"
```
<details><summary>output</summary>

```
    35	   compare with the image: silhouette agreement with the object mask (excluding pixels covered by the person),
    36	   edges / texture / colour agreement, consistency with the MoGe-2 depth map; optical flow or feature tracks between
    37	   neighbouring frames (a correct trajectory must explain the image motion of the object's texture).
    38	3. **Several hypotheses per frame, not one**: the baseline, the FoundationPose and CoCoNet estimates, their 180 deg
    39	   symmetry flips about each principal axis of the mesh, poses propagated from neighbouring frames, and fresh
    40	   registrations (e.g. render-and-compare optimisation from several initialisations, or FoundationPose
    41	   re-registration on chosen frames).
    42	4. **Global decoding over time**: choose one hyp …[+5055 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%p %s bytes\\n' | sort"
```
<details><summary>output</summary>

```
/work/inputs/.preexported 0 bytes
/work/inputs/camera.json 836 bytes
/work/inputs/episode.json 63 bytes
/work/inputs/frame0_reference_human.npz 191193 bytes
/work/inputs/ground_plane.json 398 bytes
/work/inputs/masks.h5 8809464 bytes
/work/inputs/object_mesh.glb 10939880 bytes
/work/inputs/reconstruction.npz 49772662 bytes
/work/inputs/task.txt 184 bytes
/work/inputs/video.mp4 28512527 bytes

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s bytes\\n' | head -100"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1848 bytes
/pipeline/.stages/01_depth.json 1744 bytes
/pipeline/.stages/02_export.json 1948 bytes
/pipeline/.stages/03_human_initialization.json 2148 bytes
/pipeline/.stages/05_object_pose.json 1773 bytes
/pipeline/.stages/06_coconet.json 2829 bytes
/pipeline/.stages/07_refinement.json 2133 bytes
/pipeline/.stages/08_render.json 2526 bytes
/pipeline/depth/moge2/depth_report.json 1810 bytes
/pipeline/depth/moge2/aligned.h5 1069407056 bytes
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0 bytes
/pipeline/depth/moge2/intrinsics.pkl 105135 bytes
/pipeline/depth/moge2/raw.npy 3496476800 bytes
/pipeline/export/episode_000028/edex 561 bytes
/pipeline/export/episode_000028/wild_export.json 2752 bytes
/pipeline/foundationpose/episode_000028.pkl 28071 …[+724 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work -maxdepth 2 -type f -not -path '/work/inputs/*' -printf '%p %s bytes\\n' | sort | head -100"
```
<details><summary>output</summary>

```
/work/.codex_home/.sandbox_migration 3 bytes
/work/.codex_home/config.toml 361 bytes
/work/.codex_home/goals_1.sqlite 4096 bytes
/work/.codex_home/goals_1.sqlite-shm 32768 bytes
/work/.codex_home/goals_1.sqlite-wal 78312 bytes
/work/.codex_home/installation_id 36 bytes
/work/.codex_home/logs_2.sqlite 49152 bytes
/work/.codex_home/logs_2.sqlite-shm 32768 bytes
/work/.codex_home/logs_2.sqlite-wal 119512 bytes
/work/.codex_home/memories_1.sqlite 4096 bytes
/work/.codex_home/memories_1.sqlite-shm 32768 bytes
/work/.codex_home/memories_1.sqlite-wal 61832 bytes
/work/.codex_home/model_catalog.json 3042 bytes
/work/.codex_home/queue_1.sqlite 4096 bytes
/work/.codex_home/queue_1.sqlite-shm 32768 bytes
/work/.codex_home/queue_1.sqlite-wal 82432 bytes
/work/.codex_home/state_5.sqlite 4096 bytes
/wor …[+488 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/inputs/task.txt; echo; cat /work/inputs/episode.json; echo; cat /work/inputs/camera.json; echo; cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk over to the squeegee leaning on the tall bar stool and using both hands to pick up the squeegee proceed to hold the squeegee infront of you and perform sweeping motions away you.

{
 "episode_index": 28,
 "object": "squeegee",
 "frames": 988
}
{
 "K": [
  [
   819.9974365234375,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9974365234375,
   560.5000610351562
  ],
  [
   0.0,
   0.0,
   1.0
  ]
 ],
 "K_pipeline": [
  [
   831.686767578125,
   0.0,
   768.0
  ],
  [
   0.0,
   831.686767578125,
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
   -0.9943792090216272,
   -0.008646568795513294,
   -0.10552357799841676,
   0.40723788173817244
  ],
  [
   -0.021599372796232016,
   0.9922666067814282,
   0.12223112599203007,
   -0.31638782107800895
  ],
  [
   0. …[+684 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 /work/inputs/video.mp4'
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=32.933333
nb_frames=988

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json
r=np.load('/work/inputs/reconstruction.npz')
print('keys',r.files)
for k in r.files:
 a=r[k]; print(k,a.shape,a.dtype, 'range', (np.nanmin(a), np.nanmax(a)) if a.size else None)
with h5py.File('/work/inputs/masks.h5','r') as f:
 print('mask keys',list(f.keys()))
 for k in f: print(k,f[k].shape,f[k].dtype, int(f[k][:].sum()))
PY"
```
<details><summary>output</summary>

```
keys ['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (988, 4, 4) float32 range (-0.99992925, 3.689256)
T_cam_mesh_foundationpose (988, 4, 4) float32 range (-1.2955557, 5.4856887)
T_cam_mesh_coconet (988, 4, 4) float32 range (-0.9991856, 3.2717617)
human_vertices_world (988, 4610, 3) float32 range (-0.5250845, 4.224431)
human_keypoints_world (988, 70, 3) float32 range (-0.43395978, 4.2247105)
contact_prob_right_left (988, 2) float32 range (1.1364342e-11, 1.0)
mask keys ['object', 'person']
object (988, 1152, 1536) bool 4036679
person (988, 1152, 1536) bool 43236434

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/scratch /work/outputs /work/tools && python - <<'PY'
import numpy as np, trimesh
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m), len(m))
for i,g in enumerate(m):
 print(i,g.vertices.shape,g.faces.shape,g.visual.kind, 'bounds',g.bounds)
 print('extents',g.extents,'mass',g.mass)
P=T[:,:3,3]
R=T[:,:3,:3]
print('translation first/last',P[0],P[-1])
print('translation range',P.min(0),P.max(0))
dP=np.linalg.norm(np.diff(P,axis=0),axis=1)
dR=np.array([np.degrees(np.arccos(np.clip((np.trace(R[i].T@R[i+1])-1)/2,-1,1))) for i in range(len(T)-1)])
print('dP percentiles',np.percentile(dP,[0,10,25,50,75,90,95,99,100]))
print('dR percentiles',np.percentile(dR,[0,10,25,50,75,90,95,99,100]))
for i in np.argsort(dP)[-20:]: print('jump',i,round(dP[i],4),round(dR[i],2),P[i],P[i+1])
# contact
c=r['contact_prob_right_left']
print('contact > .5 counts', (c>.5).sum(0), 'ranges',[(np.where(c[:,i]>.5)[0].min(),np.where(c[:,i]>.5)[0].max()) for i in range(2)])
# centering/mesh bounds center trajectory
# object mask area/bbox
import h5py
with h5py.File('/work/inputs/masks.h5','r') as f:
 o=f['object'][:]
 p=f['person'][:]
area=o.sum((1,2))
print('obj area pct',np.percentile(area,[0,10,25,50,75,90,99,100]))
print('empty',np.where(area==0)[0][:50],len(np.where(area==0)[0]))
print('person area',n …[+42 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
TypeError: object of type 'Trimesh' has no len()

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, trimesh, h5py
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m), 'vertices',m.vertices.shape,'faces',m.faces.shape,'visual',m.visual.kind)
print('bounds',m.bounds,'extents',m.extents,'mass',m.mass)
P=T[:,:3,3]; R=T[:,:3,:3]
print('translation first/last',P[0],P[-1])
print('translation range',P.min(0),P.max(0))
dP=np.linalg.norm(np.diff(P,axis=0),axis=1)
dR=np.array([np.degrees(np.arccos(np.clip((np.trace(R[i].T@R[i+1])-1)/2,-1,1))) for i in range(len(T)-1)])
print('dP percentiles',np.percentile(dP,[0,10,25,50,75,90,95,99,100]))
print('dR percentiles',np.percentile(dR,[0,10,25,50,75,90,95,99,100]))
for i in np.argsort(dP)[-20:]: print('jump',i,round(dP[i],4),round(dR[i],2),np.round(P[i],3),np.round(P[i+1],3))
c=r['contact_prob_right_left']
print('contact > .5 counts', (c>.5).sum(0))
for i in range(2):
 ix=np.where(c[:,i]>.5)[0]; print(i, ix.min() if len(ix) else None,ix.max() if len(ix) else None)
with h5py.File('/work/inputs/masks.h5','r') as f:
 o=f['object'][:]; p=f['person'][:]
area=o.sum((1,2))
print('obj area pct',np.percentile(area,[0,10,25,50,75,90,99,100]))
print('empty',np.where(area==0)[0][:100],len(np.where(area==0)[0]))
print('person area',np.percentile(p.sum((1,2)),[0,50,100]))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> vertices (286679, 3) faces (95575, 3) visual texture
bounds [[-0.22546926 -0.02946046 -0.51843989]
 [ 0.21180968  0.02330784  1.01889873]] extents [0.43727894 0.0527683  1.53733861] mass 0.0009574803576061292
translation first/last [0.36760977 0.72022837 2.90097427] [-0.03300745  0.69735634  3.68549395]
translation range [-0.47805455  0.24226244  2.79088473] [0.83234495 0.9706291  3.68925595]
dP percentiles [0.         0.00661124 0.01182674 0.02037286 0.03325963 0.05805859
 0.09069126 0.1796144  0.28325719]
dR percentiles [  0.           0.19663436   1.00924907   1.80814084   3.2577669
   6.39279665  11.68021969  40.43425974 119.37290556]
jump 766 0.1252 28.07 [-0.045  0.702  3.326] [-0.056  0.581  3.296]
jump 777 0.1269 29.31 [-0.129  0.686  3.348] [-0.102   …[+1318 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, numpy as np, os
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[]
for i in range(988):
 ok,im=cap.read()
 if not ok: break
 frames.append(im)
print(len(frames))
idxs=list(range(0,988,50))
if idxs[-1]"'!=987:idxs.append(987)
thumbs=[]
for i in idxs:
 im=frames[i].copy()
 cv2.putText(im,f'"'{i}',(30,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
 thumbs.append(cv2.resize(im,(384,288)))
while len(thumbs)%4:thumbs.append(np.zeros_like(thumbs[0]))
rows=[np.hstack(thumbs[i:i+4]) for i in range(0,len(thumbs),4)]
cv2.imwrite('/work/scratch/contact_50.jpg',np.vstack(rows))
# dense 680-800 every 10
idxs=list(range(680,801,10)); thumbs=[]
for i in idxs:
 im=frames[i].copy(); cv2.putText(im,f'{i}',(30,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
 thumbs.append(cv2.resize(im,(384,288)))
while len(thumbs)%4:thumbs.append(np.zeros_like(thumbs[0]))
rows=[np.hstack(thumbs[i:i+4]) for i in range(0,len(thumbs),4)]
cv2.imwrite('/work/scratch/contact_dense.jpg',np.vstack(rows))
# first 120 every 10
idxs=list(range(0,131,10)); thumbs=[]
for i in idxs:
 im=frames[i].copy(); cv2.putText(im,f'{i}',(30,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
 thumbs.append(cv2.resize(im,(384,288)))
rows=[np.hstack(thumbs[i:i+4]) for i in range(0,len(thumbs),4)]
cv2.imwrite('/work/scratch/contact_start.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```
988
Traceback (most recent call last):
  File "<stdin>", line 33, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/shape_base.py", line 289, in vstack
    return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1536 and the array at index 3 has size 768

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
for i in range(988):
 ok,im=cap.read(); frames.append(im)
def montage(idxs,name):
 thumbs=[]
 for i in idxs:
  im=frames[i].copy(); cv2.putText(im,str(i),(30,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
  thumbs.append(cv2.resize(im,(384,288)))
 rows=[]
 for j in range(0,len(thumbs),4):
  row=thumbs[j:j+4]
  while len(row)<4: row.append(np.zeros_like(thumbs[0]))
  rows.append(np.hstack(row))
 cv2.imwrite(name,np.vstack(rows))
montage(list(range(680,801,10)),'/work/scratch/contact_dense.jpg')
montage(list(range(0,131,10)),'/work/scratch/contact_start.jpg')
montage(list(range(130,681,25)),'/work/scratch/contact_mid.jpg')
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
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(float)
Tf=r['T_cam_mesh_foundationpose'].astype(float); Tc=r['T_cam_mesh_coconet'].astype(float)
import json
cam=json.load(open('/work/inputs/camera.json')); W=np.array(cam['T_world_cam'])
for name,X in [('fp',W@Tf),('coco',W@Tc)]:
 R1=T[:,:3,:3];R2=X[:,:3,:3]
 dr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',R1,R2)-1)/2,-1,1)))
 dt=np.linalg.norm(T[:,:3,3]-X[:,:3,3],axis=1)
 print(name,'rot diff pct',np.percentile(dr,[0,10,25,50,75,90,95,99,100]))
 print(name,'trans diff pct',np.percentile(dt,[0,10,25,50,75,90,95,99,100]))
 print('first rows',np.round(dr[:10],1),np.round(dt[:10],2))
# Compare fp and coco camera frame
R1=Tf[:,:3,:3];R2=Tc[:,:3,:3]
dr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',R1,R2)-1)/2,-1,1)))
dt=np.linalg.norm(Tf[:,:3,3]-Tc[:,:3,3],axis=1)
print('fp-coco rot',np.percentile(dr,[0,10,50,90,99,100]),'trans',np.percentile(dt,[0,10,50,90,99,100]))
# baseline relative motion per ranges
for a,b in [(0,100),(100,180),(180,500),(500,650),(650,700),(700,790),(790,900),(900,988)]:
 dR=[];dP=[]
 for i in range(a,b-1):
  R1=T[i,:3,:3];R2=T[i+1,:3,:3]
  dR.append(np.degrees(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1))))
  dP.append(np.linalg.norm(T[i,:3,3]-T[i+1,:3,3]))
 print(a,b,'rot med/p90/max',np.percentile(dR,[50,90,100]),'trans',np.percentile(dP,[50,90,100]))
PY"
```
<details><summary>output</summary>

```
fp rot diff pct [  0.8363695    4.37649279   6.76198245  11.09730908  16.93571573
  27.12557607  71.40419154 176.16298838 179.87910627]
fp trans diff pct [0.03194085 0.08411427 0.11408957 0.17317242 0.34109802 0.54533632
 0.68009386 1.78493086 2.70826841]
first rows [ 4.1 11.5 14.4 15.9 14.9 16.9 15.8 16.3 14.6 15.9] [0.17 0.16 0.17 0.17 0.16 0.16 0.15 0.14 0.14 0.14]
coco rot diff pct [1.2288605  1.2862931  1.31556502 1.34046116 1.38533059 1.38966093
 1.39060726 1.39166242 1.39256013]
coco trans diff pct [0.06238971 0.06435347 0.06796285 0.07716867 0.08300293 0.10470581
 0.11670159 0.12209806 0.12483262]
first rows [1.3 1.3 1.3 1.3 1.3 1.3 1.3 1.3 1.3 1.3] [0.08 0.08 0.08 0.08 0.08 0.08 0.08 0.08 0.08 0.08]
fp-coco rot [  0.42779513   3.82484419  10.36869227  26.41695775 175.23373935
 179 …[+910 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "for i in 0 50 100 130 200 300 400 500 600 680 700 720 740 760 780 800 850 900 950 987; do ffmpeg -loglevel error -y -i /work/inputs/video.mp4 -vf \"select=eq(n\\,"'$i)" -vframes 1 /work/scratch/frame_$i.png; done'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/.stages/05_object_pose.json; echo; cat /pipeline/.stages/06_coconet.json; echo; cat /pipeline/.stages/07_refinement.json'
```
<details><summary>output</summary>

```
{
  "name": "05_object_pose",
  "outputs": [
    {
      "mtime_ns": 1791249282970632326,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000028/foundationpose/episode_000028.pkl",
      "size": 280719
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py",
      "/home/hanzhou/Projects/v2d/run/out/episode_000028/export/episode_000028",
      "--depth-root",
      "/home/hanzhou/Projects/v2d/run/out/episode_000028/depth/moge2/aligned.h5",
      "--out-file",
      "/home/hanzhou/Projects/v2d/run/out/episode_000028/foundationpose/episode_000028.pkl",
      "--camera-id",
      "0",
      "--iteration",
      "5",
      "--max-attempts",
      "5",
 …[+5937 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"T_world_mesh\\|depth_correction\\|camera.json\\|ground_plane\" -n /code/reconstruction | head -100"
```
<details><summary>output</summary>

```
grep: /code/reconstruction/.venv/bin/python3.12_object_reconstruction/mesh_to_usd/runtime/recorded_support.py:416:        ground_plane_file_sha256=ground_plane_file_sha256,
/code/reconstruction/modules/v2d_hoi_object_reconstruction/mesh_to_usd/runtime/recorded_support.py:446:    ground_plane: np.ndarray,
/code/reconstruction/modules/v2d_hoi_object_reconstruction/mesh_to_usd/runtime/recorded_support.py:451:    ground_plane_file_sha256: str,
/code/reconstruction/modules/v2d_hoi_object_reconstruction/mesh_to_usd/runtime/recorded_support.py:484:                ground_plane,
/code/reconstruction/modules/v2d_hoi_object_reconstruction/mesh_to_usd/runtime/recorded_support.py:490:                ground_plane_file_sha256=ground_plane_file_sha256,
/code/reconstruction/modules/v2d_hoi_object_reconstru …[+7479 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
try:
 import nvdiffrast; print('nvdiffrast',nvdiffrast.__file__)
 import torch
 print('torch',torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
except Exception as e: print(type(e),e)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 True NVIDIA RTX 6000 Ada Generation

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth','/pipeline/foundationpose/episode_000028.pkl']:
 print('\\n',p)
 try:
  x=torch.load(p,map_location='cpu',weights_only=False)
  print(type(x))
  if isinstance(x,dict):
   print(list(x.keys())[:100])
   for k,v in x.items():
    if hasattr(v,'shape'): print(' ',k,tuple(v.shape),v.dtype)
    elif isinstance(v,(list,tuple)): print(' ',k,type(v),len(v), type(v[0]) if len(v) else '')
    elif isinstance(v,dict): print(' ',k,'dict',list(v.keys())[:20])
    else: print(' ',k,type(v),str(v)[:100])
  else: print(x)
 except Exception as e: print('ERR',e)
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth
<class 'dict'>
['schema', 'frames', 'frame_meta', 'kid', 'checkpoint', 'mhr_inference', 'config', 'mesh_diameter', 'mhr_neutral_height_init', 'mhr_spatial_scale', 'K_rois', 'bboxes', 'faces', 'pr', 'pr_initial', 'in', 'gt', 'observations', 'raw', 'metadata']
  schema <class 'str'> cari4d.mhr_wild_inference.v1
  frames <class 'list'> 988 <class 'str'>
  frame_meta <class 'list'> 988 <class 'dict'>
  kid <class 'int'> 0
  checkpoint dict ['path', 'step']
  mhr_inference dict ['schema', 'checkpoint', 'wandbRun', 'wandbConfigSha256', 'supervisionContract', 'configSource', 'xyzAnchorContract', 'spatialNormalizationContract']
  config dict ['rgb_files', 'render_files', 'split_file', 'val_split_file', 'render_root', 'rgb_root', 'packed_root', 'packed_format', 'ex …[+5146 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,numpy as np,pprint
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for sec in ['pr','pr_initial','in','raw']:
 print('\\n',sec)
 for k,v in x[sec].items():
  if hasattr(v,'shape'): print(k,tuple(v.shape),v.dtype,'minmax',float(v.min()),float(v.max()))
  else: print(k,type(v),str(v)[:200])
print('\\nmetadata'); pprint.pp(x['metadata'])
print('\\npostopt keys'); pprint.pp(x['postopt']['optimized_parameters']); print(x['postopt']['frame_indices'][:20],x['postopt']['frame_indices'][-20:])
print('postopt config'); pprint.pp(x['postopt']['config'])
PY"
```
<details><summary>output</summary>

```

 pr
pose_abs (988, 4, 4) float32 minmax -0.9999752044677734 3.3414595127105713
pose_abs_1st_delta (988, 4, 4) float32 minmax -0.9999752044677734 3.3586883544921875
mhr_global_rot6d (988, 6) float32 minmax -0.99772047996521 0.999995231628418
mhr_trans (988, 3) float32 minmax -0.9297337532043457 3.7761552333831787
mhr_body_pose_cont (988, 260) float32 minmax -0.7249239087104797 1.1434167623519897
mhr_hand (988, 108) float32 minmax -0.7737778425216675 0.6352624297142029
mhr_shape (988, 45) float32 minmax -1.2175124883651733 1.0405677556991577
mhr_scale (988, 28) float32 minmax -2.6215901374816895 3.2895078659057617
mhr_face (988, 72) float32 minmax 0.0 0.0
contact_logits (988, 2) float32 minmax -25.200347900390625 23.372316360473633
pose_abs_postopt (988, 4, 4) float32 minmax -0.999975204467 …[+5702 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_check.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py
import nvdiffrast.torch as dr

work='/work'
r=np.load(f'{work}/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open(f'{work}/inputs/camera.json')); K=np.array(cam['K'],np.float32); Wc=np.array(cam['T_world_cam'],np.float32)
Twc=np.linalg.inv(Wc)
mesh=trimesh.load(f'{work}/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
# face centers for rendering? full mesh
H,W=1152,1536
ctx=dr.RasterizeCudaContext()
def render(Twm):
    Tcm=Twc@Twm
    A=torch.tensor(Tcm,dtype=torch.float32,device='cuda')[:3,:]
    vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
    vc=(A@vh.T).T
    ones=torch.ones((len(V),1),device='cuda')
    # standard screen projection with clip w=z
    P=torch.tensor([[K[0,0],0,K[0,2],0],[0,K[1,1],K[1,2],0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device='cuda')
    vclip=(P@torch.cat([vc,ones],1).T).T
    rast,_=dr.rasterize(ctx,vclip[None],F,resolution=[H,W])
    mask=(rast[0,...,3]>0).cpu().numpy().astype(np.uint8)
    return mask

cap=cv2.VideoCapture(f'{work}/inputs/video.mp4'); frames=[]
for i in range(988):
 ok,im=cap.read(); frames.append(im)
with h5py.File(f'{work}/inputs/masks.h5','r') as h: om=h['object'][:]
idxs=[0,50,100,130,200,300,400,500,600,680,700,720,740,760,780,800,8 …[+695 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_check.py", line 33, in <module>
    im=frames[i].copy(); mask=render(T[i])
                              ^^^^^^^^^^^^
  File "/work/scratch/render_check.py", line 22, in render
    rast,_=dr.rasterize(ctx,vclip[None],F,resolution=[H,W])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^ …[+443 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/vclip\\[None\\],F,/vclip[None].contiguous(),F.contiguous(),/' /work/scratch/render_check.py && python /work/scratch/render_check.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
from scipy.ndimage import binary_erosion
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(float); c=r['contact_prob_right_left']
with h5py.File('/work/inputs/masks.h5','r') as f: o=f['object'][:]
y,x=np.indices(o.shape[1:])
cent=np.array([(x[m].mean(),y[m].mean()) for m in o]); area=o.sum((1,2))
iou=np.array([ (o[i]&o[i+1]).sum()/(o[i]|o[i+1]).sum() for i in range(len(o)-1)])
sp=np.linalg.norm(np.diff(cent,axis=0),axis=1)
for a,b in [(0,50),(0,90),(0,100),(90,120),(100,180),(180,500),(500,650),(650,700),(700,790),(790,900),(900,930),(930,988)]:
 print(a,b,'cent range',cent[a:b].min(0),cent[a:b].max(0),'sp med/p90',np.percentile(sp[a:b-1],[50,90]),'iou med',np.median(iou[a:b-1]),'area',np.median(area[a:b]),'contact r/l',np.median(c[a:b],0).round(2))
print('first cent every10',np.round(cent[0:121:10],1))
print('last cent every10',np.round(cent[900:988:10],1))
print('spikes sp',[(i,round(v,2)) for i,v in enumerate(sp) if v>5][:100])
print('contact transitions')
for i in range(1,988):
 if (c[i]>0.5)"'!=(c[i-1]>0.5): print(i,np.round(c[i],2))
PY'
```
<details><summary>output</summary>

```
0 50 cent range [624.68464903 758.63462699] [627.75688719 760.38245033] sp med/p90 [0.29305243 1.46367294] iou med 0.9889048695294843 area 4811.5 contact r/l [0. 0.]
0 90 cent range [624.68464903 758.49215598] [627.75688719 760.38245033] sp med/p90 [0.33122671 1.62009868] iou med 0.9854627354627354 area 4801.0 contact r/l [0. 0.]
0 100 cent range [624.68464903 758.49215598] [627.75688719 760.38245033] sp med/p90 [0.33305884 1.62009868] iou med 0.985982316152685 area 4798.0 contact r/l [0. 0.]
90 120 cent range [625.19365499 755.68465845] [628.35300338 759.94738006] sp med/p90 [0.50688001 2.35884611] iou med 0.9767733835530445 area 4622.5 contact r/l [0.38 0.98]
100 180 cent range [598.6874614  643.43192675] [660.95132651 759.78011981] sp med/p90 [3.94345288 9.58071612] iou med 0.5465061004 …[+3113 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
with h5py.File('/work/inputs/masks.h5','r') as f:o=f['object'][:]
y,x=np.indices(o.shape[1:]); cent=np.array([(x[m].mean(),y[m].mean()) for m in o])
ref=o[0]; iou=np.array([(ref&m).sum()/(ref|m).sum() for m in o])
print('first 130 ihou and displacement')
for i in range(0,140):
 if i%5==0 or iou[i]<.95: print(i,round(iou[i],4),np.round(cent[i]-cent[0],2))
# end relative 987
ref=o[987]; iou=np.array([(ref&m).sum()/(ref|m).sum() for m in o])
print('last 200')
for i in range(780,988):
 if i%10==0 or iou[i]>.98: print(i,round(iou[i],4),np.round(cent[i]-cent[987],2))
PY"
```
<details><summary>output</summary>

```
first 130 ihou and displacement
0 1.0 [0. 0.]
1 0.9269 [ 1.31 -0.63]
2 0.9137 [ 2.26 -1.09]
3 0.9134 [ 2.2  -1.04]
4 0.9265 [ 0.77 -0.19]
5 0.9158 [ 2.19 -1.05]
6 0.9146 [ 2.13 -1.  ]
7 0.9151 [ 2.04 -0.96]
8 0.9186 [ 0.39 -0.27]
9 0.9249 [ 0.59 -0.1 ]
10 0.9272 [ 0.65 -0.15]
11 0.9154 [ 0.2  -0.18]
12 0.9164 [ 0.28 -0.22]
13 0.9166 [ 0.36 -0.32]
14 0.9146 [-0.05 -0.03]
15 0.9216 [ 0.53 -0.11]
16 0.9196 [ 0.18 -0.16]
17 0.9227 [0.41 0.04]
18 0.9167 [ 0.81 -0.21]
19 0.9134 [ 0.18 -0.05]
20 0.9155 [-0.26  0.32]
21 0.9113 [0.54 0.09]
22 0.9173 [-0.68  0.65]
23 0.915 [-0.77  0.63]
24 0.914 [-0.4   0.38]
25 0.9114 [ 0.24 -0.1 ]
26 0.918 [ 1.61 -0.82]
27 0.9148 [ 0.26 -0.14]
28 0.9111 [-0.33  0.22]
29 0.9098 [-0.25  0.12]
30 0.9082 [-0.11  0.04]
31 0.9117 [-0.25  0.2 ]
32 0.9094 [-0.21  0.18]
33 …[+3189 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz'); c=r['contact_prob_right_left']
for th in [.1,.5,.9]:
 print('th',th)
 for j in range(2):
  ix=np.where(c[:,j]>th)[0]
  # contiguous runs
  runs=[]
  for i in ix:
   if not runs or i>runs[-1][-1]+1:runs.append([i])
   else:runs[-1].append(i)
  print(j,[(x[0],x[-1],len(x)) for x in runs if len(x)>2][:20])
print('c 85:125'); print(np.round(c[85:125],2))
print('c 890:950'); print(np.round(c[890:950],2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "for i in 860 870 880 890 910 920 930 940; do ffmpeg -loglevel error -y -i /work/inputs/video.mp4 -vf \"select=eq(n\\,"'$i)" -vframes 1 /work/scratch/frame_$i.png; done'
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float)
ref=T[950]
P=T[:,:3,3];R=T[:,:3,:3]
dR=np.degrees(np.arccos(np.clip((np.einsum('ij,fij->f',ref[:3,:3],R)-1)/2,-1,1)))
dP=np.linalg.norm(P-ref[:3,3],axis=1)
for i in range(760,988,5): print(i,round(dR[i],2),round(dP[i],3),np.round(P[i],3))
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
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);P=T[:,:3,3];R=T[:,:3,:3]
for refidx,a,b in [(0,0,120),(950,900,988),(870,800,988)]:
 ref=T[refidx]; dr=np.degrees(np.arccos(np.clip((np.einsum('ij,fij->f',ref[:3,:3],R)-1)/2,-1,1)));dp=np.linalg.norm(P-ref[:3,3],axis=1)
 print('\\nref',refidx)
 for i in range(a,b,5):print(i,round(dr[i],2),round(dp[i],3))
PY"
```
<details><summary>output</summary>

```

ref 0
0 0.03 0.0
5 24.01 0.079
10 20.6 0.08
15 19.81 0.078
20 18.9 0.102
25 21.04 0.09
30 21.04 0.084
35 21.04 0.082
40 21.04 0.082
45 21.04 0.084
50 23.47 0.083
55 28.04 0.074
60 31.38 0.084
65 29.8 0.08
70 34.05 0.079
75 31.86 0.055
80 31.47 0.043
85 33.26 0.048
90 36.49 0.037
95 34.82 0.056
100 25.25 0.036
105 23.23 0.026
110 28.3 0.029
115 31.23 0.023

ref 950
900 2.43 0.185
905 4.21 0.247
910 5.05 0.155
915 3.19 0.131
920 2.22 0.112
925 1.96 0.059
930 4.87 0.046
935 3.07 0.048
940 4.23 0.034
945 5.53 0.049
950 0.09 0.0
955 2.91 0.03
960 3.2 0.008
965 3.2 0.008
970 3.2 0.008
975 3.2 0.009
980 13.5 0.027
985 10.75 0.045

ref 870
800 7.94 0.18
805 7.77 0.177
810 9.08 0.169
815 9.19 0.091
820 15.05 0.053
825 11.6 0.026
830 16.01 0.052
835 16.19 0.083
840 9.82 0.05
845 10.29 0.053
850 7.6 …[+412 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(m.visual)
for a in dir(m.visual):
 if a.startswith('_'):continue
 try:
  v=getattr(m.visual,a)
  if not callable(v): print(a,type(v),getattr(v,'shape',None),str(v)[:200])
 except Exception as e: pass
print('uv',m.visual.uv,m.visual.uv.shape,m.visual.uv.dtype,np.min(m.visual.uv,axis=0),np.max(m.visual.uv,axis=0))
print('material',m.visual.material, type(m.visual.material))
for a in ['baseColorTexture','image','baseColorFactor','metallicFactor','roughnessFactor']:
 v=getattr(m.visual.material,a,None)
 if v is not None: print(a,type(v),getattr(v,'shape',None),getattr(v,'size',None))
PY"
```
<details><summary>output</summary>

```
<trimesh.visual.texture.TextureVisuals object at 0x78fbf2f02890>
defined <class 'bool'> None True
face_materials <class 'NoneType'> None None
kind <class 'str'> None texture
material <class 'trimesh.visual.material.PBRMaterial'> None <trimesh.visual.material.PBRMaterial object at 0x78fbf2f02950>
mesh <class 'trimesh.base.Trimesh'> None <trimesh.Trimesh(vertices.shape=(286679, 3), faces.shape=(95575, 3))>
uv <class 'trimesh.caching.TrackedArray'> (286679, 2) [[0.114384   0.36529499]
 [0.114384   0.36529499]
 [0.114384   0.36529499]
 ...
 [0.40553239 0.81350255]
 [0.40553239 0.81350255]
 [0.40553239 0.81350255]]
vertex_attributes <class 'trimesh.caching.DataStore'> None <trimesh.caching.DataStore object at 0x78fbf2f02810>
uv [[0.114384   0.36529499]
 [0.114384   0.36529499]
 [0.114384   0.36 …[+377 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
m.visual.material.baseColorTexture.save('/work/scratch/object_texture.jpg')
print(m.visual.material.baseColorFactor,m.visual.material.metallicFactor,m.visual.material.roughnessFactor)
PY"
```
<details><summary>output</summary>

```
None 0.0 None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);R=T[:,:3,:3]
R0=R[0]
for i in [5,10,20,30,40,50,60,70,80,90,100,110,120,130]:
 d=R0.T@R[i]; rv=Rotation.from_matrix(d).as_rotvec(); ang=np.degrees(np.linalg.norm(rv)); ax=rv/(np.linalg.norm(rv)+1e-9)
 print(i,round(ang,2),np.round(ax,2), 'mesh ax world',np.round(R[i][:,2],2))
# world long axis direction
for i in [0,50,100,130,500,700,800,900,960]:
 print('T',i, 'R',np.round(R[i],2),'P',np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
5 24.01 [-0.34  0.02 -0.94] mesh ax world [0.78 0.49 0.39]
10 20.6 [-0.38  0.05 -0.92] mesh ax world [0.78 0.5  0.39]
20 18.9 [-0.38  0.03 -0.92] mesh ax world [0.78 0.49 0.38]
30 21.04 [-0.39  0.02 -0.92] mesh ax world [0.78 0.48 0.39]
40 21.04 [-0.39  0.02 -0.92] mesh ax world [0.78 0.48 0.39]
50 23.47 [-0.3   0.06 -0.95] mesh ax world [0.77 0.51 0.38]
60 31.38 [-0.14  0.04 -0.99] mesh ax world [0.78 0.53 0.34]
70 34.05 [-0.06  0.04 -1.  ] mesh ax world [0.78 0.55 0.3 ]
80 31.47 [-0.09  0.02 -1.  ] mesh ax world [0.79 0.53 0.31]
90 36.49 [-0.18 -0.01 -0.98] mesh ax world [0.79 0.5  0.37]
100 25.25 [-0.32  0.01 -0.95] mesh ax world [0.78 0.49 0.39]
110 28.3 [-0.12 -0.02 -0.99] mesh ax world [0.8  0.51 0.31]
120 35.71 [ 0.02 -0.04 -1.  ] mesh ax world [0.82 0.52 0.25]
130 53.25 [ 0.31 -0.0 …[+898 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_texture_test.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py
import nvdiffrast.torch as dr
work='/work'; dev='cuda'
r=np.load(f'{work}/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open(f'{work}/inputs/camera.json')); K=np.array(cam['K'],np.float32); Wc=np.array(cam['T_world_cam'],np.float32); Twc=np.linalg.inv(Wc).astype(np.float32)
mesh=trimesh.load(f'{work}/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev); F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=dev)
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255.
print('tex',tex.shape,tex.min(),tex.max())
TEX=torch.tensor(tex,device=dev)
H,W=1152,1536; ctx=dr.RasterizeCudaContext()
P=torch.tensor([[K[0,0],0,K[0,2],0],[0,K[1,1],K[1,2],0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=dev)
def render(Twm, flip=False):
 Tcm=torch.tensor(Twc,device=dev)@torch.tensor(Twm,device=dev)
 A=Tcm[:3,:]
 vh=torch.cat([V,torch.ones_like(V[:,:1])],1); vc=(A@vh.T).T
 vclip=(P@torch.cat([vc,torch.ones_like(vc[:,:1])],1).T).T
 rast,_=dr.rasterize(ctx,vclip[None].contiguous(),F.contiguous(),resolution=[H,W])
 uv,_=dr.interpolate(UV[None],rast,F)
 if flip: uv=torch.stack([uv[...,0],1-uv[...,1]],-1)
 rgb=dr.texture(TEX[None][...,:3],uv.contiguous(),filter_mode='linear',boundary_mode='wra …[+1053 chars]
```
<details><summary>output</summary>

```
tex (3385, 3286, 3) 0.0 1.0
Traceback (most recent call last):
  File "/work/scratch/render_texture_test.py", line 28, in <module>
    with h5py.File(f'{work}/inputs/masks.h5','r') as h: om=h['object'][50], pm=h['person'][50]
                                                           ^^^^^^^^^^^^^^^^^^^
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py'
s=open(p).read().replace(\"with h5py.File(f'{work}/inputs/masks.h5','r') as h: om=h['object'][50], pm=h['person'][50]\\nom,pm=om,pm[0] if isinstance(pm,tuple) else pm\",\"with h5py.File(f'{work}/inputs/masks.h5','r') as h:\\n om=h['object'][50]; pm=h['person'][50]\")
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py"
```
<details><summary>output</summary>

```
tex (3385, 3286, 3) 0.0 1.0
flip False
/work/scratch/render_texture_test.py:35: RuntimeWarning: Mean of empty slice.
  print('valid',valid.sum(),'mean rgb',a.mean(0),'obs',b.mean(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
valid 0 mean rgb [nan nan nan] obs [nan nan nan]
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:520: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis, **keepdims_kw)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2889: RuntimeWarning: Degrees of freedom <= 0 for slice
  c = cov(x, y, rowvar, dtype=dtype)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2748: RuntimeWarning: divide by zero encountered in  …[+319 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/work/inputs/masks.h5','r') as h:
 print(h['object'][50].sum(),h['person'][50].sum(),(h['object'][50]&h['person'][50]).sum())
PY"
```
<details><summary>output</summary>

```
4975 30700 65

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/print('flip',flip);pscore/print('flip',flip,'mask sum',mask.sum().item());pscore/\" /work/scratch/render_texture_test.py && python /work/scratch/render_texture_test.py"
```
<details><summary>output</summary>

```
tex (3385, 3286, 3) 0.0 1.0
flip False mask sum 0
/work/scratch/render_texture_test.py:35: RuntimeWarning: Mean of empty slice.
  print('valid',valid.sum(),'mean rgb',a.mean(0),'obs',b.mean(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
valid 0 mean rgb [nan nan nan] obs [nan nan nan]
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:520: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis, **keepdims_kw)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2889: RuntimeWarning: Degrees of freedom <= 0 for slice
  c = cov(x, y, rowvar, dtype=dtype)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2748: RuntimeWarning: divide by zero enco …[+341 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py';s=open(p).read();s=s.replace(\"rast,_=dr.rasterize\", \"print('vc range',vc.min(0)[0].item(),vc.max(0)[0].item(),torch.isfinite(vclip).all().item(),vclip[:3]);rast,_=dr.rasterize\")
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py"
```
<details><summary>output</summary>

```
tex (3385, 3286, 3) 0.0 1.0
Traceback (most recent call last):
  File "/work/scratch/render_texture_test.py", line 40, in <module>
    rgb,mask=render(T[50],flip)
             ^^^^^^^^^^^^^^^^^^
  File "/work/scratch/render_texture_test.py", line 19, in render
    print('vc range',vc.min(0)[0].item(),vc.max(0)[0].item(),torch.isfinite(vclip).all().item(),vclip[:3]);rast,_=dr.rasterize(ctx,vclip[None].contiguous(),F.contiguous(),resolution=[H,W])
                     ^^^^^^^^^^^^^^^^^^^
RuntimeError: a Tensor with 3 elements cannot be converted to Scalar

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py';s=open(p).read();s=s.replace(\"vc.min(0)[0].item(),vc.max(0)[0].item()\",\"vc.min(0)[0],vc.max(0)[0]\")
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py"
```
<details><summary>output</summary>

```
tex (3385, 3286, 3) 0.0 1.0
vc range tensor([-1.0622,  0.1761,  2.8267], device='cuda:0') tensor([0.1954, 1.2070, 3.4722], device='cuda:0') True tensor([[2588.0847, 2112.4961,    3.4195,    3.4195],
        [2588.0847, 2112.4961,    3.4195,    3.4195],
        [2588.0847, 2112.4961,    3.4195,    3.4195]], device='cuda:0')
flip False mask sum 0
/work/scratch/render_texture_test.py:35: RuntimeWarning: Mean of empty slice.
  print('valid',valid.sum(),'mean rgb',a.mean(0),'obs',b.mean(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
valid 0 mean rgb [nan nan nan] obs [nan nan nan]
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:520: RuntimeWarning: Mean of empty slice.
  avg = …[+935 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py';s=open(p).read();s=s.replace(\"print('vc range'\", \"print('P',P,'K',K,'vc0',vc[0],'vclip0',vclip[0]);print('vc range'\")
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py | head -20"
```
<details><summary>output</summary>

```
/work/scratch/render_texture_test.py:35: RuntimeWarning: Mean of empty slice.
  print('valid',valid.sum(),'mean rgb',a.mean(0),'obs',b.mean(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:520: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis, **keepdims_kw)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2889: RuntimeWarning: Degrees of freedom <= 0 for slice
  c = cov(x, y, rowvar, dtype=dtype)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2748: RuntimeWarning: divide by zero encountered in divide
  c *= np.true_divide(1, fact)
/opt/conda/lib/python3.11/site-packages/numpy/lib/ …[+1886 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,45p' /work/scratch/render_check.py"
```
<details><summary>output</summary>

```
import numpy as np, torch, trimesh, cv2, json, h5py
import nvdiffrast.torch as dr

work='/work'
r=np.load(f'{work}/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open(f'{work}/inputs/camera.json')); K=np.array(cam['K'],np.float32); Wc=np.array(cam['T_world_cam'],np.float32)
Twc=np.linalg.inv(Wc)
mesh=trimesh.load(f'{work}/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda'); F=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda')
# face centers for rendering? full mesh
H,W=1152,1536
ctx=dr.RasterizeCudaContext()
def render(Twm):
    Tcm=Twc@Twm
    A=torch.tensor(Tcm,dtype=torch.float32,device='cuda')[:3,:]
    vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
    vc=(A@vh.T).T
   …[+1320 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/render_check.py /work/scratch/rc2.py; python - <<'PY'
p='/work/scratch/rc2.py';s=open(p).read();s=s.replace(\"cv2.imwrite(f'{work}/scratch/baseline_overlay.jpg',np.vstack(rows))\",\"print('final mask sum', mask.sum())\")
open(p,'w').write(s)
PY
python /work/scratch/rc2.py"
```
<details><summary>output</summary>

```
final mask sum 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py';s=open(p).read()
s=s.replace(\"P=torch.tensor([[K[0,0],0,K[0,2],0],[0,K[1,1],K[1,2],0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=dev)\",\"P=torch.tensor([[2*K[0,0]/W,0,2*K[0,2]/W-1,0],[0,-2*K[1,1]/H,1-2*K[1,2]/H,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=dev)\")
s=s.replace(\"print('P',P,'K',K,'vc0',vc[0],'vclip0',vclip[0]);print('vc range'\", \"print('vc range'\")
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py"
```
<details><summary>output</summary>

```
tex (3385, 3286, 3) 0.0 1.0
vc range tensor([-1.0622,  0.1761,  2.8267], device='cuda:0') tensor([0.1954, 1.2070, 3.4722], device='cuda:0') True tensor([[-0.0496, -0.2480,  3.4195,  3.4195],
        [-0.0496, -0.2480,  3.4195,  3.4195],
        [-0.0496, -0.2480,  3.4195,  3.4195]], device='cuda:0')
flip False mask sum 4240
/work/scratch/render_texture_test.py:35: RuntimeWarning: Mean of empty slice.
  print('valid',valid.sum(),'mean rgb',a.mean(0),'obs',b.mean(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
valid 0 mean rgb [nan nan nan] obs [nan nan nan]
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:520: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis, **keepd …[+893 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py';s=open(p).read();s=s.replace(\"print('valid',valid.sum()\", \"ys,xs=np.where(mask.cpu().numpy()); print('mask bbox',xs.min(),xs.max(),ys.min(),ys.max());print('valid',valid.sum()\")
# disable pscore expensive prints? fine
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py | head -70"
```
<details><summary>output</summary>

```
/work/scratch/render_texture_test.py:35: RuntimeWarning: Mean of empty slice.
  ys,xs=np.where(mask.cpu().numpy()); print('mask bbox',xs.min(),xs.max(),ys.min(),ys.max());print('valid',valid.sum(),'mean rgb',a.mean(0),'obs',b.mean(0))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:520: RuntimeWarning: Mean of empty slice.
  avg = a.mean(axis, **keepdims_kw)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2889: RuntimeWarning: Degrees of freedom <= 0 for slice
  c = cov(x, y, rowvar, dtype=dtype)
/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py:2748: RuntimeWarning: divide by zero encountered  …[+1036 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float); cam=json.load(open('/work/inputs/camera.json'));W=np.array(cam['T_world_cam']);K=np.array(cam['K']);Twc=np.linalg.inv(W)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene').to_geometry(); X=m.vertices
Tc=Twc@T[50]
xc=(Tc[:3,:3]@X.T+Tc[:3,3:4]).T
uv=(K@xc.T).T; uv=uv[:,:2]/uv[:,2:]
print(uv.min(0),uv.max(0),uv.mean(0))
print('camera bounds',xc.min(0),xc.max(0))
PY"
```
<details><summary>output</summary>

```
[450.36575246 602.78848374] [804.61219657 908.96456013] [666.77707154 733.22799564]
camera bounds [-1.062188    0.1761212   2.82669052] [0.19535211 1.20697087 3.47218447]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py';s=open(p).read();s=s.replace(\"[0,-2*K[1,1]/H,1-2*K[1,2]/H,0]\",\"[0,2*K[1,1]/H,2*K[1,2]/H-1,0]\")
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py | head -40"
```
<details><summary>output</summary>

```
tex (3385, 3286, 3) 0.0 1.0
vc range tensor([-1.0622,  0.1761,  2.8267], device='cuda:0') tensor([0.1954, 1.2070, 3.4722], device='cuda:0') True tensor([[-0.0496,  0.2480,  3.4195,  3.4195],
        [-0.0496,  0.2480,  3.4195,  3.4195],
        [-0.0496,  0.2480,  3.4195,  3.4195]], device='cuda:0')
flip False mask sum 4240
mask bbox 450 804 603 908
valid 2003 mean rgb [0.14725846 0.10264866 0.07191788] obs [0.23849377 0.17164652 0.14543241]
0 0.3216753720079536
1 0.1546624458444108
2 0.13714376942971354
luma corr 0.20676296755740334
vc range tensor([-1.0622,  0.1761,  2.8267], device='cuda:0') tensor([0.1954, 1.2070, 3.4722], device='cuda:0') True tensor([[-0.0496,  0.2480,  3.4195,  3.4195],
        [-0.0496,  0.2480,  3.4195,  3.4195],
        [-0.0496,  0.2480,  3.4195,  3.4195]], devi …[+257 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_scores.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py, time
import nvdiffrast.torch as dr
work='/work'; dev='cuda'; H,W=1152,1536
r=np.load(f'{work}/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open(f'{work}/inputs/camera.json')); K=np.array(cam['K'],np.float32); Wc=np.array(cam['T_world_cam'],np.float32); Twc=np.linalg.inv(Wc).astype(np.float32)
mesh=trimesh.load(f'{work}/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev); F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
uv=np.asarray(mesh.visual.uv).copy(); uv[:,1]=1-uv[:,1]; UV=torch.tensor(uv,dtype=torch.float32,device=dev)
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255.; TEX=torch.tensor(tex[...,:3],device=dev)
ctx=dr.RasterizeCudaContext()
P=torch.tensor([[2*K[0,0]/W,0,2*K[0,2]/W-1,0],[0,2*K[1,1]/H,2*K[1,2]/H-1,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=dev)
def render(Twm):
 Tcm=torch.tensor(Twc,device=dev)@torch.tensor(Twm,device=dev)
 A=Tcm[:3,:]; vh=torch.cat([V,torch.ones_like(V[:,:1])],1); vc=(A@vh.T).T
 vclip=(P@torch.cat([vc,torch.ones_like(vc[:,:1])],1).T).T
 rast,_=dr.rasterize(ctx,vclip[None].contiguous(),F.contiguous(),resolution=[H,W])
 uvi,_=dr.interpolate(UV[None],rast,F)
 rgb=dr.texture(TEX[None],uvi.contiguous(),filter_mode='linear',boundary_mode='wrap')[0]
 return rgb.cpu().numpy(),(rast[0 …[+2153 chars]
```
<details><summary>output</summary>

```
obs object 4647 orange 542 median area
rank 1.3308 68 0.5629 0.5582 4.0860 0.4005319 3148
rank 1.1853 74 0.5213 0.5325 4.4798 0.3130254 3005
rank 1.0901 76 0.5489 0.6458 8.1688 0.44370317 3117
rank 1.0689 70 0.5803 0.5146 5.9549 0.36488068 3189
rank 1.0164 80 0.5015 0.4596 4.5945 0.3672258 2950
rank 0.8612 78 0.4286 0.4815 5.7992 0.33032486 2640
rank 0.7293 66 0.5473 0.5015 8.6833 0.31503943 3086
rank 0.5928 62 0.3624 0.3724 5.5181 0.24963504 2321
rank 0.4767 86 0.2508 0.3860 5.7988 0.22504747 1786
rank 0.4059 72 0.4890 0.4917 11.1733 0.3391066 2877
rank 0.3955 96 0.4353 0.2786 6.2460 0.18368103 2707
rank 0.3935 52 0.3195 0.3342 6.2218 0.18473192 2140
rank 0.3623 64 0.3232 0.3771 7.4570 0.20314139 2115
rank 0.3582 50 0.2584 0.3351 5.9768 0.18140484 1825
rank 0.3211 88 0.3615 0.3579 7.9623  …[+3262 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/overlay_fixed.py <<'PY'
import numpy as np, torch, trimesh, cv2, json, h5py
import nvdiffrast.torch as dr
work='/work';dev='cuda';H,W=1152,1536
r=np.load(f'{work}/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32)
cam=json.load(open(f'{work}/inputs/camera.json'));K=np.array(cam['K'],np.float32);Wc=np.array(cam['T_world_cam'],np.float32);Twc=np.linalg.inv(Wc).astype(np.float32)
m=trimesh.load(f'{work}/inputs/object_mesh.glb',force='scene').to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device=dev);F=torch.tensor(m.faces,dtype=torch.int32,device=dev)
ctx=dr.RasterizeCudaContext();P=torch.tensor([[2*K[0,0]/W,0,2*K[0,2]/W-1,0],[0,2*K[1,1]/H,2*K[1,2]/H-1,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=dev)
def render(Twm):
 A=(torch.tensor(Twc,device=dev)@torch.tensor(Twm,device=dev))[:3,:];vh=torch.cat([V,torch.ones_like(V[:,:1])],1);vc=(A@vh.T).T;vclip=(P@torch.cat([vc,torch.ones_like(vc[:,:1])],1).T).T
 rast,_=dr.rasterize(ctx,vclip[None].contiguous(),F.contiguous(),resolution=[H,W]);return (rast[0,...,3]>0).cpu().numpy().astype(np.uint8)
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(988):ok,im=cap.read();frames.append(im)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:om=h['object'][:]
idxs=[0,20,50,68,80,90,110,130,200,300,400,500,600,680,700,720,740,760,780,800,850,900,950,987]
thumbs=[]
for i in idxs:
 im=frames[i].copy();mask=render(T[i]);cnt,_=cv2.findContours(mask,cv2.RE …[+568 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/compare_tex.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import cv2
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4')
for i in range(69):ok,im=cap.read()
obs=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
# crop
x0,x1,y0,y1=400,850,560,950
outs=[]; labels=['obs']+[str(i) for i in [0,20,50,68,74,80,100]]
outs.append(obs[y0:y1,x0:x1])
for i in [0,20,50,68,74,80,100]:
 rgb,mask=render(T[i]);
 # simple contrast/brightness gain fit per channel using obj mask only
 with h5py.File(f'{work}/inputs/masks.h5','r') as h: om=h['object'][i]
 valid=mask&om
 b=obs[valid].astype(float)/255; a=rgb[valid]
 for c in range(3):
  A=np.stack([a[:,c],np.ones(len(a))],1); coef=np.linalg.lstsq(A,b[:,c],rcond=None)[0]
  rgb[...,c]=rgb[...,c]*coef[0]+coef[1]
 rgb=np.clip(rgb,0,1); vis=(rgb*255).astype(np.uint8); vis[~mask]=0
 # composite over obs darkened mask
 comp=obs.copy();comp[mask]=(0.5*comp[mask]+0.5*vis[mask]).astype(np.uint8)
 outs.append(comp[y0:y1,x0:x1])
thumbs=[]
for lab,im in zip(labels,outs):
 im=cv2.resize(im,(450,390)); cv2.putText(im,lab,(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2); thumbs.append(im)
rows=[np.hstack(thumbs[i:i+4]) for i in range(0,8,4)]
cv2.imwrite(f'{work}/scratch/compare_tex.jpg',cv2.cvtColor(np.vstack(rows),cv2.COLOR_RGB2BGR))
PY
python /work/scratch/compare_tex.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_scores.py';s=open(p).read()
# rewrite only tail data range and filenames minimally via exec custom? Generate second script from prefix
prefix=s.split('# load frames')[0]
tail=r'''
# load frames 900..988 and masks
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4'); frames=[]
for i in range(988):
 ok,im=cap.read()
 if i>=900: frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
frames=np.array(frames)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:
 om=h['object'][900:]; pm=h['person'][900:]
med=np.median(frames[::2],axis=0)
obs_obj=(om.mean(0)>.5)&(pm.mean(0)<.5)
m=obs_obj; rgbim=med.astype(np.float32)/255
orange_obs=m&(rgbim[...,0]>rgbim[...,1]*1.25)&(rgbim[...,0]>rgbim[...,2]*1.25)&(rgbim[...,0]>.28)
print('obs object',obs_obj.sum(),'orange',orange_obs.sum())
rows=[]
for fi in range(900,988,2):
 i=fi
 rgb,mask=render(T[i])
 inter=(mask&obs_obj).sum(); union=(mask|obs_obj).sum(); iou=inter/union
 orange_ren=mask&(rgb[...,0]>rgb[...,1]*1.25)&(rgb[...,0]>rgb[...,2]*1.25)&(rgb[...,0]>.18)
 oi=(orange_ren&orange_obs).sum()/max(1,(orange_ren|orange_obs).sum())
 yy,xx=np.where(orange_obs); cobs=np.array([xx.mean(),yy.mean()])
 yy,xx=np.where(orange_ren); cren=np.array([xx.mean(),yy.mean()]) if len(xx) else np.array([np.nan,np.nan])
 cd=np.linalg.norm(cobs-cren)
 valid=mask&obs_obj; a=rgb[valid]; b=rgbim[valid]; aa=a-a.mean(0);bb=b-b.mean(0)
 ncc=((aa*bb).sum(0)/(np.sqrt((aa*aa).sum(0)*(bb*bb).sum(0))+1e-9)).mean()
 score=iou+ …[+472 chars]
```
<details><summary>output</summary>

```
obs object 5328 orange 583
rank -12.3944 952 0.1613 0.0000 125.6581 0.06769428 1372
rank -13.3298 948 0.1257 0.0000 134.6620 0.071287535 1115
rank -13.3957 946 0.0865 0.0000 134.8420 0.013548389 794
rank -13.6015 986 0.0793 0.0000 136.6035 -0.13594432 703
rank -14.3005 954 0.1383 0.0000 144.6235 0.15679464 1221
rank -14.3783 940 0.2580 0.0000 146.3081 -0.03644899 1946
rank -14.4583 928 0.4117 0.0000 148.5395 -0.10702843 2812
rank -14.4793 926 0.2812 0.0000 147.4648 -0.09338245 2150
rank -14.4915 932 0.3756 0.0000 148.5853 -0.057198778 2638
rank -14.5279 930 0.2744 0.0000 147.9049 -0.078322716 2103
rank -14.5443 938 0.3103 0.0000 148.5692 0.015919087 2257
rank -14.6173 974 0.1525 0.0000 147.7277 0.020277927 1318
rank -14.6173 972 0.1525 0.0000 147.7277 0.020277927 1318
rank -14.6173 970 0.1 …[+3295 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/debug_end.py <<'PY'
exec(open('/work/scratch/static_scores_end.py').read().split('# load frames')[0])
import cv2
# recreate data
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(988):
 ok,im=cap.read()
 if i>=900:frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
frames=np.array(frames)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:om=h['object'][900:];pm=h['person'][900:]
med=np.median(frames[::2],axis=0); obs_obj=(om.mean(0)>.5)&(pm.mean(0)<.5); rgbim=med.astype(np.float32)/255
orange_obs=obs_obj&(rgbim[...,0]>rgbim[...,1]*1.25)&(rgbim[...,0]>rgbim[...,2]*1.25)&(rgbim[...,0]>.28)
for i in [0,20,50,60,68,80,900,920,940,950,952,960,975,980,987]:
 rgb,mask=render(T[i]); orange=mask&(rgb[...,0]>rgb[...,1]*1.25)&(rgb[...,0]>rgb[...,2]*1.25)&(rgb[...,0]>.18)
 ys,xs=np.where(mask); print(i,'mask',mask.sum(),'bbox',xs.min(),xs.max(),ys.min(),ys.max(),'orange',orange.sum(),'cent',xs.mean(),ys.mean())
print('obs obj',obs_obj.sum(),'orange',orange_obs.sum())
ys,xs=np.where(orange_obs);print('obs orange cent',xs.mean(),ys.mean(),'bbox',xs.min(),xs.max(),ys.min(),ys.max())
# write orange obs median and renders
cv2.imwrite(f'{work}/scratch/end_median.jpg',cv2.cvtColor(med.astype(np.uint8),cv2.COLOR_RGB2BGR))
PY
python /work/scratch/debug_end.py"
```
<details><summary>output</summary>

```
0 mask 4592 bbox 444 816 608 910 orange 468 cent 658.4091898954704 743.3294860627178
20 mask 4265 bbox 448 804 608 906 orange 453 cent 655.2919109026964 742.188511137163
50 mask 4240 bbox 450 804 603 908 orange 446 cent 655.9622641509434 739.1186320754717
60 mask 4081 bbox 449 800 603 909 orange 424 cent 655.605243812791 738.0066160254839
68 mask 4093 bbox 447 800 599 904 orange 449 cent 655.7935499633521 733.8174932812118
80 mask 4185 bbox 444 802 600 906 orange 452 cent 655.5132616487455 735.5376344086021
900 mask 4103 bbox 797 867 651 1043 orange 395 cent 835.3546185717768 893.5432610285158
920 mask 4516 bbox 791 860 631 1030 orange 470 cent 824.8452170062002 883.2528786536758
940 mask 4161 bbox 783 865 613 1018 orange 373 cent 830.9947128094208 859.1934631098294
950 mask 4708 bbox 786  …[+602 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_scores_end.py';s=open(p).read();s=s.replace(\"range(988):\\n ok,im=cap.read()\\n if i>=900: frames.append\",\"range(988):\\n ok,im=cap.read()\\n if i>=955: frames.append\").replace(\"h['object'][900:]\",\"h['object'][955:]\").replace(\"h['person'][900:]\",\"h['person'][955:]\").replace(\"for fi in range(900,988,2):\",\"for fi in range(920,988,2):\")
open('/work/scratch/static_scores_final.py','w').write(s)
PY
python /work/scratch/static_scores_final.py | head -80"
```
<details><summary>output</summary>

```
obs object 5504 orange 584
rank -12.4811 952 0.1596 0.0000 126.5322 0.08342037 1384
rank -13.4164 948 0.1239 0.0000 135.5361 0.088876754 1120
rank -13.4827 946 0.0845 0.0000 135.7099 0.025019856 791
rank -13.6853 986 0.0818 0.0000 137.4823 -0.12583973 737
rank -14.3881 954 0.1364 0.0000 145.5016 0.17110538 1227
rank -14.4716 940 0.2508 0.0000 147.2061 -0.012201966 1938
rank -14.5588 928 0.4019 0.0000 149.4301 -0.11814881 2815
rank -14.5754 926 0.2748 0.0000 148.3538 -0.09904794 2150
rank -14.5901 932 0.3655 0.0000 149.4790 -0.051209822 2633
rank -14.6235 930 0.2687 0.0000 148.7918 -0.08663255 2106
rank -14.6398 938 0.3010 0.0000 149.4642 0.037327316 2246
rank -14.7036 974 0.1505 0.0000 148.6037 0.042147625 1326
rank -14.7036 972 0.1505 0.0000 148.6037 0.042147625 1326
rank -14.7036 970 0.1 …[+2761 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# use debug_end prefix
exec(open('/work/scratch/static_scores_end.py').read().split('# load frames')[0])
import h5py
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:]
for i in [0,20,50,68,80,900,920,940,950,952,960,975,980,987]:
 rgb,mask=render(T[i]); inter=(mask&om[i]).sum();uni=(mask|om[i]).sum();print(i,'iou',inter/uni,'areas',mask.sum(),om[i].sum(),'inter',inter)
PY"
```
<details><summary>output</summary>

```
0 iou 0.1477033552061409 areas 4592 4678 inter 1193
20 iou 0.2542208690838638 areas 4265 4798 inter 1837
50 iou 0.28414158305462656 areas 4240 4975 inter 2039
68 iou 0.5652479774885684 areas 4093 4807 inter 3214
80 iou 0.5014206919605549 areas 4185 4798 inter 3000
900 iou 0.28402691937920616 areas 4103 5246 inter 2068
920 iou 0.18254456054087279 areas 4516 5104 inter 1485
940 iou 0.2662276575729069 areas 4161 5261 inter 1981
950 iou 0.10617619395966141 areas 4708 5877 inter 1016
952 iou 0.15947727535561465 areas 4551 5475 inter 1379
960 iou 0.15111412736115823 areas 4644 5533 inter 1336
975 iou 0.14968152866242038 areas 4622 5486 inter 1316
980 iou 0.2380130654505115 areas 4560 5484 inter 1931
987 iou 0.4038946483608854 areas 4177 5844 inter 2883

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
im=cv2.imread('/work/scratch/overlay_fixed.jpg')
# montage thumbnails: positions row 5? indices list, 950 index 22 => row5 col2? row height288; x col2=768, y5=1440
cv2.imwrite('/work/scratch/overlay950.jpg',im[1440:1728,768:1152])
cv2.imwrite('/work/scratch/overlay0.jpg',im[0:288,0:384])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_align_scores.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import cv2,itertools
# static image consensus start
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(120):
 ok,im=cap.read();frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
frames=np.array(frames)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:om=h['object'][:120];pm=h['person'][:120]
med=np.median(frames[::2],axis=0); obs_obj=(om.mean(0)>.55)&(pm.mean(0)<.2); rgbim=med.astype(np.float32)/255
# crop
x0,x1,y0,y1=400,850,540,960
ob=obs_obj[y0:y1,x0:x1]; oi=rgbim[y0:y1,x0:x1]
# mesh local centroid and 180 flips
import trimesh
meshcenter=mesh.vertices.mean(0)
def flipT(Twm,axis):
 R=np.eye(4);R[:3,:3]=-np.eye(3);R[:3,:3]+=2*np.outer(axis,axis)
 C=np.eye(4);C[:3,3]=meshcenter
 Ci=np.eye(4);Ci[:3,3]=-meshcenter
 return Twm@C@R@Ci
axes=np.eye(3)
# candidate base list
cands=[]
for i in range(0,111,2): cands.append((f'b{i}',T[i].astype(float)))
for i in [0,20,40,60,68,80,100]:
 for ai,ax in enumerate(axes): cands.append((f'b{i} flip{ai}',flipT(T[i].astype(float),ax)))
# score helper

def score_candidate(Twm,search=True):
 rgb,mask=render(Twm.astype(np.float32)); m=mask[y0:y1,x0:x1];r=rgb[y0:y1,x0:x1]
 best=None
 scales=[0.97,0.985,1.0,1.015,1.03] if search else [1.]
 shifts=range(-18,19,2) if search else [0]
 # centroid of candidate crop
 yy,xx=np.where(m); cc=np.array([xx.mean(),yy.mean()])
 for scale in scale …[+1720 chars]
```
<details><summary>output</summary>

```
obs 4486
1.1133 b60 0.5990 1.0150 -4 -2 0.31046245 0.4289 0.5229 3259
1.1056 b86 0.5895 1.0000 -6 -2 0.3319966 0.4480 0.4977 3244
1.0993 b78 0.6041 1.0000 -6 2 0.33663294 0.4460 0.4536 3253
1.0929 b72 0.6046 1.0000 -4 0 0.3223981 0.4353 0.4574 3240
1.0897 b52 0.6298 1.0000 -4 -2 0.31605545 0.4156 0.4176 3353
1.0892 b58 0.6051 1.0150 -6 -2 0.2871882 0.4060 0.4949 3274
1.0875 b50 0.6332 1.0000 -6 -2 0.30299047 0.4172 0.4159 3383
1.0781 b76 0.5920 1.0000 -4 2 0.32930508 0.4439 0.4423 3211
1.0757 b68 0.6087 1.0000 -4 2 0.3040701 0.4253 0.4357 3246
1.0742 b74 0.5711 1.0000 -2 0 0.32842723 0.4469 0.4754 3129
1.0582 b6 0.6211 1.0150 -6 -4 0.2604216 0.3543 0.4532 3365
1.0531 b70 0.5961 1.0150 -4 2 0.2867578 0.4193 0.4329 3234
1.0476 b84 0.5986 0.9850 -2 -4 0.24597728 0.3527 0.4896 3202
1.0449 b56  …[+1642 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_align_scores.py';s=open(p).read()
# create end version by replacing data section and candidate loop
s=s.replace(\"for i in range(120):\", \"for i in range(988):\")
s=s.replace(\"frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))\\nframes=np.array(frames)\\nwith h5py.File(f'{work}/inputs/masks.h5','r') as h:om=h['object'][:120];pm=h['person'][:120]\", \"if i>=950: frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))\\nframes=np.array(frames)\\nwith h5py.File(f'{work}/inputs/masks.h5','r') as h:om=h['object'][950:];pm=h['person'][950:]\")
s=s.replace(\"x0,x1,y0,y1=400,850,540,960\", \"x0,x1,y0,y1=720,920,560,1080\")
s=s.replace(\"for i in range(0,111,2): cands.append((f'b{i}',T[i].astype(float)))\", \"for i in range(900,988,2): cands.append((f'b{i}',T[i].astype(float)))\")
s=s.replace(\"for i in [0,20,40,60,68,80,100]:\", \"for i in [900,920,940,950,960,980]:\")
s=s.replace(\"score=iou+0.4*max(0,ncc)+0.3*max(0,hn)+0.5*orange\", \"score=iou+0.4*max(0,ncc)+0.5*max(0,hn)\")
s=s.replace(\"shifts=range(-18,19,2)\",\"shifts=range(-14,15,2)\")
s=s.replace(\"np.savez(f'{work}/scratch/static_align_scores.npz'\", \"np.savez(f'{work}/scratch/static_align_scores_end.npz'\")
open('/work/scratch/static_align_end.py','w').write(s)
PY
python /work/scratch/static_align_end.py"
```
<details><summary>output</summary>

```
  File "/work/scratch/static_align_end.py", line 6
    ok,im=cap.read();if i>=950: frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                     ^^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/ok,im=cap.read();if i>=950:/ok,im=cap.read()\\n if i>=950:/' /work/scratch/static_align_end.py && sed -n '1,20p' /work/scratch/static_align_end.py && python /work/scratch/static_align_end.py"
```
<details><summary>output</summary>

```
obs 5499
0.6376 b940 flip0 0.4690 0.9850 -2 8 0.13022687 0.2329 0.0000 3074
0.6029 b940 flip1 0.2782 0.9850 0 10 0.29570147 0.4129 0.0000 2054
0.4883 b986 0.4883 1.0300 12 14 -0.071457855 -0.0909 0.0000 3284
0.4657 b980 flip0 0.3525 0.9850 10 14 0.08901116 0.1551 0.0000 2556
0.4335 b932 0.4335 1.0300 0 14 -0.119420685 -0.1229 0.0000 3049
0.4280 b928 0.4280 1.0300 2 14 -0.11709436 -0.1218 0.0000 3011
0.4263 b940 0.4263 1.0300 -4 12 -0.089546084 -0.1047 0.0000 2964
0.4158 b982 0.4158 1.0300 4 14 -0.13128337 -0.1452 0.0000 2970
0.4140 b938 0.4140 1.0300 -2 12 -0.091670565 -0.1003 0.0000 2916
0.3959 b984 0.3959 1.0300 6 14 -0.12663665 -0.1307 0.0000 2862
0.3931 b936 0.3931 1.0300 -2 14 -0.08190601 -0.0906 0.0000 2814
0.3895 b900 flip0 0.2579 1.0300 -4 -14 0.108288474 0.1766 0.0000 1957
0.3810  …[+1806 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,json,trimesh
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:
 for i in [0,60,68,900,920,940,950,952,960,975,980,987]:
  y,x=np.where(h['object'][i]);print(i,'obs bbox',x.min(),x.max(),y.min(),y.max(),'cent',x.mean(),y.mean(),'area',len(x))
PY"
```
<details><summary>output</summary>

```
0 obs bbox 447 798 631 902 cent 625.4959384352287 759.7289439931594 area 4678
60 obs bbox 447 797 632 902 cent 625.5034714916894 759.6642120765832 area 4753
68 obs bbox 447 797 632 902 cent 625.5529436238818 759.600790513834 area 4807
900 obs bbox 771 873 654 1050 cent 825.1277163553183 849.5089592070149 area 5246
920 obs bbox 767 873 654 1050 cent 825.5762147335423 850.9204545454545 area 5104
940 obs bbox 739 873 652 1050 cent 824.9621744915415 843.4983843375784 area 5261
950 obs bbox 729 873 651 1050 cent 821.6559469116896 831.5858431172367 area 5877
952 obs bbox 728 873 652 1050 cent 821.7042922374429 836.1041095890411 area 5475
960 obs bbox 727 873 652 1050 cent 821.3770106632929 835.7169709018616 area 5533
975 obs bbox 727 873 652 1050 cent 821.5432008749544 836.5067444403937 area 548 …[+164 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_baseline.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py,time
with h5py.File(f'{work}/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
ious=[];areas=[]
for i in range(len(T)):
 rgb,mask=render(T[i]);m=om[i] # use full obj mask
 inter=(mask&m).sum();uni=(mask|m).sum();ious.append(inter/max(1,uni));areas.append(mask.sum())
np.savez(f'{work}/scratch/baseline_eval.npz',iou=np.array(ious),area=np.array(areas))
print('iou pct',np.percentile(ious,[0,1,5,10,25,50,75,90,95,99,100]))
for i,v in sorted(enumerate(ious),key=lambda x:x[1])[:80]: print('low',i,round(v,3),areas[i],om[i].sum())
PY
python /work/scratch/eval_baseline.py"
```
<details><summary>output</summary>

```
iou pct [0.         0.         0.01921908 0.05125868 0.15037898 0.23869501
 0.34045469 0.43700974 0.50576244 0.57816574 0.65554314]
low 638 0.0 3449 2348
low 700 0.0 3892 1775
low 710 0.0 2274 1648
low 713 0.0 2395 1442
low 722 0.0 2068 1601
low 724 0.0 3229 1614
low 725 0.0 3256 1680
low 726 0.0 3287 1511
low 727 0.0 3282 1670
low 729 0.0 2729 1486
low 730 0.0 2821 1366
low 731 0.0 3924 1389
low 732 0.0 4008 1450
low 733 0.0 2093 1510
low 734 0.0 2124 1629
low 745 0.0 3801 2340
low 746 0.0 3802 2442
low 533 0.0 4120 1954
low 748 0.001 4367 2580
low 637 0.002 3452 2246
low 639 0.003 3445 2377
low 770 0.004 3089 3244
low 633 0.005 3774 2298
low 625 0.005 3588 2531
low 634 0.005 3705 2279
low 635 0.006 3714 2222
low 640 0.008 3455 2431
low 632 0.01 3475 2317
low 636 0.01 3717 2190
low 137 0. …[+1209 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2
from scipy.spatial.transform import Rotation
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:]
def pc(m,top=1.0):
 y,x=np.where(m); pts=np.c_[x,y].astype(float); c=pts.mean(0); q=pts-c; cov=q.T@q/len(q);ev,evec=np.linalg.eigh(cov);v=evec[:,-1];
 if v[0]<0:v=-v
 ang=np.degrees(np.arctan2(v[1],v[0]));proj=q@v; return c,ang,np.percentile(proj,[1,99]),np.sqrt(ev[-1]),np.sqrt(ev[0])
# get render bbox metrics from eval? recompute subset quickly using prefix
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
for i in [0,50,68,100,120,130,137,140,200,300,400,500,533,550,600,630,638,650,680,700,710,720,730,740,750,760,770,780,800,837,850,900,950,987]:
 rgb,m=render(T[i]);c1,a1,p1,s1,w1=pc(o[i]);c2,a2,p2,s2,w2=pc(m)
 # correct angle diff mod 180
 da=(a2-a1+90)%180-90
 print(i,'obs',np.round(c1,1),round(a1,1),np.round(p1,1),round(w1,1),'ren',np.round(c2,1),round(a2,1),np.round(p2,1),round(w2,1),'dc',np.round(c2-c1,1),'da',round(da,1),'lendiff',round(p2[1]-p2[0]-(p1[1]-p1[0]),1))
PY"
```
<details><summary>output</summary>

```
0 obs [625.5 759.7] -37.8 [-220.8  183.6] 8.8 ren [658.4 743.3] -37.4 [-260.6  157.3] 17.0 dc [ 32.9 -16.4] da 0.4 lendiff 13.5
50 obs [626.4 759.4] -37.6 [-221.3  183.2] 9.3 ren [656.  739.1] -39.2 [-256.9  153.9] 16.5 dc [ 29.6 -20.3] da -1.6 lendiff 6.4
68 obs [625.6 759.6] -37.7 [-221.6  183.2] 9.0 ren [655.8 733.8] -39.3 [-260.   153.3] 16.2 dc [ 30.2 -25.8] da -1.6 lendiff 8.5
100 obs [625.6 759.6] -37.7 [-221.5  183.2] 9.0 ren [649.3 744.1] -38.1 [-262.4  157.6] 16.6 dc [ 23.7 -15.5] da -0.4 lendiff 15.4
120 obs [628.1 752.9] -36.3 [-220.3  179.2] 9.5 ren [656.5 737.9] -36.6 [-269.   157.4] 16.4 dc [ 28.5 -15. ] da -0.3 lendiff 26.8
130 obs [608.5 735.7] -30.5 [-235.9  197.4] 9.3 ren [654.9 709.3] -33.0 [-262.2  145.9] 15.2 dc [ 46.4 -26.4] da -2.5 lendiff -25.1
137 obs [619.8 703.6 …[+3580 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/mask_refine_test.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import cv2,h5py
from scipy.ndimage import distance_transform_edt
#import trimesh
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
# camera rotation z

def feat(mask):
 ys,xs=np.where(mask); pts=np.c_[xs,ys].astype(float); c=pts.mean(0); q=pts-c
 cov=q.T@q/max(1,len(q)); ev,evec=np.linalg.eigh(cov);v=evec[:,-1]
 if v[0]<0:v=-v
 ang=np.arctan2(v[1],v[0]); proj=q@v
 # robust length 2-98
 L=np.percentile(proj,98)-np.percentile(proj,2)
 return c,ang,L,len(xs)

def eval_mask(M,O,do=None,dr=None):
 if do is None: do=distance_transform_edt(~O)
 if dr is None: dr=distance_transform_edt(~M)
 prec=do[M].mean() if M.any() else 999
 rec=dr[O].mean() if O.any() else 999
 inter=(M&O).sum();uni=(M|O).sum();iou=inter/max(1,uni)
 areadiff=abs(M.sum()-O.sum())/max(1,O.sum())
 return prec+rec+10*areadiff-2*iou,prec,rec,iou,areadiff

def cam_adjust(Tcam,theta=0,scalez=1.0,dx=0,dy=0):
 c,s=np.cos(theta),np.sin(theta);Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
 R=Rz@Tcam[:3,:3];t=Rz@Tcam[:3,3]
 tx,ty,tz=t[0],t[1],t[2];tz=tz/scalez
 tx+=dx*tz/K[0,0];ty+=dy*tz/K[1,1]
 Tn=np.eye(4);Tn[:3,:3]=R;Tn[:3,3]=[tx,ty,tz];return Tn

def refine(i,verbose=False):
 O=OM[i]; O=O&~PM[i] # don't remove person? object mask likely separate
 do=distance_transform_edt(~O)
 Tcam=(Twc@T[i]).astype(float)
 rgb,M=render((Wc@Tcam).astype(np.float32) …[+2615 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/mask_refine_test.py", line 70, in <module>
    b=refine(i,True);res[i]=b
      ^^^^^^^^^^^^^^
  File "/work/scratch/mask_refine_test.py", line 61, in refine
    rgb2,M2=render((Wc@Tt).astype(np.float32));c1,a1,L1,_=feat(M2)
                    ~~^~~
ValueError: matmul: Input operand 1 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/mask_refine_test.py';s=open(p).read()
s=s.replace(\"b=best; Tc=b[5]\",\"b=best; Tc=b[7]\")
s=s.replace(\"np.linspace(b[3]-5,b[3]+5,11)\",\"np.linspace(b[5]-5,b[5]+5,11)\")
s=s.replace(\"np.linspace(b[4]*.9,b[4]*1.1,7)\",\"np.linspace(b[6]*.9,b[6]*1.1,7)\")
s=s.replace(\"Tt=b[5]\",\"Tt=b[7]\")
s=s.replace(\"b=(*eval_mask(M2,O,do),b[3],b[4],Tt,M2,rgb2)\",\"b=(*eval_mask(M2,O,do),b[5],b[6],Tt,M2,rgb2)\")
s=s.replace(\"'theta',b[3],'scalez',b[4]\",\"'theta',b[5],'scalez',b[6]\")
s=s.replace(\"res[i][6].astype\",\"res[i][8].astype\")
s=s.replace(\"res[i][5] for i in idxs\",\"res[i][7] for i in idxs\")
open(p,'w').write(s)
PY
python /work/scratch/mask_refine_test.py"
```
<details><summary>output</summary>

```
frame 0 base d -0.40845660938898615 best (9.128643661396664, 7.2907102670707165, 0.6903961142858364, 0.45583238958097394, 0.2059202059202059) theta -1.4084566093889848 scalez 1.100616739078049 orig (array([658.50664923, 743.25419664]), -0.6532956717585053, 412.13475270297016) obs (array([625.73230373, 759.57636208]), -0.6604245844438756, 399.83794063935693)
frame 50 base d 1.646047190062128 best (8.742406852962452, 7.259871444903874, 2.0826259760927703, 0.35910842047330765, 0.011812627291242363) theta 0.6460471900621272 scalez 1.076390346530947 orig (array([655.97429245, 739.11179245]), -0.6841267905204024, 404.51007103714085) obs (array([627.1287169 , 758.81588595]), -0.6553978474106172, 399.6032141955484)
frame 100 base d 0.38086665792712787 best (8.618330121593283, 6.521130081063933, 0. …[+2787 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_dynamic_montage.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import cv2,h5py
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(988):ok,im=cap.read();frames.append(im)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
idxs=list(range(690,811,6));thumbs=[]
for i in idxs:
 im=frames[i].copy();rgb,M=render(T[i]);M=M.astype(np.uint8)
 cnt,_=cv2.findContours(M,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),3)
 cnt,_=cv2.findContours(om[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,255,0),2)
 # crop around union bbox
 u=M.astype(bool)|om[i];y,x=np.where(u);x0=max(0,x.min()-70);x1=min(1536,x.max()+70);y0=max(0,y.min()-70);y1=min(1152,y.max()+70)
 im=im[y0:y1,x0:x1];im=cv2.resize(im,(300,360));cv2.putText(im,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);thumbs.append(im)
rows=[]
for j in range(0,len(thumbs),5):
 row=thumbs[j:j+5]
 while len(row)<5:row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite(f'{work}/scratch/dynamic_montage.jpg',np.vstack(rows))
PY
python /work/scratch/make_dynamic_montage.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/mask_refine_test.py';s=open(p).read()
s=s.replace(\"O=OM[i]; O=O&~PM[i] # don't remove person? object mask likely separate\", \"O=OM[i]&~PM[i]\")
s=s.replace(\"prec=do[M].mean() if M.any() else 999\", \"Mv=M&~PM_CUR\\n prec=do[Mv].mean() if Mv.any() else 999\")
# Need pass PM? simplify eval_mask uses global PM_CUR set in refine
s=s.replace(\"def refine(i,verbose=False):\\n O=OM[i]&~PM[i]\", \"def refine(i,verbose=False):\\n global PM_CUR\\n PM_CUR=PM[i]\\n O=OM[i]&~PM[i]\")
s=s.replace(\"scalez=max(.5,min(2.,Lm/Lo))*dz_spread\", \"scalez=max(.5,min(2.,Lo/Lm))*dz_spread\")
# For final eval need PM_CUR still same
# idxs reduce
s=s.replace(\"idxs=[0,50,100,130,137,533,630,700,710,720,730,740,750,760,770,780,800,850,900,950,987]\", \"idxs=[690,696,702,708,714,720,726,732,738,744,750,756,762,768,774,780,786,798,810]\")
s=s.replace(\"mask_refine_test.jpg\",\"mask_refine_dyn.jpg\").replace(\"mask_refine_test.npz\",\"mask_refine_dyn.npz\")
open('/work/scratch/mask_refine_dyn.py','w').write(s)
PY
python /work/scratch/mask_refine_dyn.py"
```
<details><summary>output</summary>

```
frame 690 base d 3.655794383083803 best (8.3097203620136, 0.4504698329171928, 6.60080020173504, 0.3340775886128942, 0.1926605504587156) theta 4.655794383083801 scalez 0.9605063584653177 orig (array([667.15203145, 681.34895151]), -0.8005603971177239, 265.73352244697907) obs (array([607.84770642, 726.03990826]), -0.7367547483569966, 274.4502558792576)
frame 696 base d 5.886944322648156 best (4.434483054211992, 1.4712580886625555, 3.018541833790326, 0.2624920936116382, 0.046966731898238745) theta 6.886944322648155 scalez 0.7130023873857771 orig (array([657.08729365, 672.10555278]), -0.7015375725701087, 391.0497498588296) obs (array([610.92759295, 695.13796477]), -0.598791011258311, 261.3931924087092)
frame 702 base d 17.572503812002914 best (9.724544330585166, 3.7371733631394317, 3.3368291621 …[+4845 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/debug_adjust.py <<'PY'
exec(open('/work/scratch/mask_refine_dyn.py').read().split('idxs=')[0])
import cv2,h5py
from scipy.ndimage import distance_transform_edt
for i,deg,scalez in [(744,0.3,0.973),(720,6,0.78),(726,-83.6,.688),(756,4.86,1.06)]:
 PM_CUR=PM[i];O=OM[i]&~PM[i];do=distance_transform_edt(~O);fil=feat(O);Tcam=(Twc@T[i]).astype(float)
 Tc=cam_adjust(Tcam,np.radians(deg),scalez);rgb,M=render((Wc@Tc).astype(np.float32));c1,_,_,_=feat(M);print('\\ni',i,'obs',fil[0],fil[1],fil[2],'before',c1,feat(M)[1:3]);print('dxdy',fil[0]-c1)
 Tc=cam_adjust(Tc,0,1,*(fil[0]-c1));rgb,M=render((Wc@Tc).astype(np.float32));c2=feat(M);print('after',c2[0],c2[1],c2[2],eval_mask(M,O,do));print('Tcam',Tcam[:3,3],'new',Tc[:3,3])
PY
python /work/scratch/debug_adjust.py"
```
<details><summary>output</summary>

```

i 744 obs [748.60971223 750.75134892] 1.241569492524655 182.80953051097777 before [696.85125448 772.92249104] (1.3114638538983958, 175.06484191246213)
dxdy [ 51.75845775 -22.17114212]
after [752.67294751 750.87213997] 1.1247942534390676 181.90920652076298 (16.90708137056376, 10.747407000017928, 6.7306693509156394, 0.2967384973791497, 0.0022482014388489208)
Tcam [-0.24644712  0.68809283  2.94592857] new [-0.05893886  0.60493052  3.02767582]

i 720 obs [679.40023966 646.44218095] 0.42727944840314536 229.28605294679238 before [702.59358289 644.33511586] (0.4800331894512985, 177.07720677867158)
dxdy [-23.19334322   2.10706508]
after [678.94061758 647.15558195] 0.4960444107182939 172.15165620907356 (14.831325073712337, 1.2819361814350174, 13.606808984455666, 0.07364713416586616, 0.008987417615 …[+971 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/debug_overlay_one.py <<'PY'
exec(open('/work/scratch/debug_adjust.py').read().split('for i,deg')[0])
import cv2,h5py
i=744;PM_CUR=PM[i];O=OM[i]&~PM[i];fil=feat(O);Tcam=(Twc@T[i]).astype(float);Tc=cam_adjust(Tcam,0,.973);rgb,M=render((Wc@Tc).astype(np.float32));c1,_,_,_=feat(M);Tc=cam_adjust(Tc,0,1,*(fil[0]-c1));rgb,M=render((Wc@Tc).astype(np.float32))
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4')
for j in range(745):ok,im=cap.read()
for name,mm in [('base',render((Wc@(Twc@T[i])).astype(np.float32))[1]),('adj',M)]:
 vis=im.copy();cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(vis,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(O.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(vis,cnt,-1,(255,255,0),2)
 y,x=np.where(M|O);x0=x.min()-30;x1=x.max()+30;y0=y.min()-30;y1=y.max()+30
 vis=cv2.resize(vis[y0:y1,x0:x1],None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST);cv2.putText(vis,name,(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);cv2.imwrite(f'{work}/scratch/debug_{name}.jpg',vis)
print('areas',O.sum(),M.sum(),'bbox O',np.where(O)[1].min(),np.where(O)[1].max(),np.where(O)[0].min(),np.where(O)[0].max(),'M',np.where(M)[1].min(),np.where(M)[1].max(),np.where(M)[0].min(),np.where(M)[0].max())
PY
python /work/scratch/debug_overlay_one.py"
```
<details><summary>output</summary>

```
areas 2224 2218 bbox O 713 782 657 839 M 697 835 635 795

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/debug_overlay_one.py';s=open(p).read();s=s.replace(\"Tc=cam_adjust(Tcam,0,.973)\",\"Tc=cam_adjust(Tcam,np.radians(7),.973)\")
open('/work/scratch/debug_overlay_one2.py','w').write(s.replace(\"debug_{name}\",\"debug2_{name}\"))
PY
python /work/scratch/debug_overlay_one2.py"
```
<details><summary>output</summary>

```
areas 2224 2228 bbox O 713 782 657 839 M 697 835 631 801

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/roll_search_test.py <<'PY'
exec(open('/work/scratch/mask_refine_dyn.py').read().split('idxs=')[0])
import cv2,h5py
from scipy.ndimage import distance_transform_edt
# mesh local rot helper
meshcenter=mesh.vertices.mean(0).astype(float)
def rollworld(Tw,d):
 C=np.eye(4);C[:3,3]=meshcenter;Ci=np.eye(4);Ci[:3,3]=-meshcenter
 c,s=np.cos(np.radians(d)),np.sin(np.radians(d));Rz=np.eye(4);Rz[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
 return Tw@C@Rz@Ci

def search(i,verbose=False):
 PM_CUR=PM[i]; O=OM[i]&~PM[i]; do=distance_transform_edt(~O);co,ao,Lo,no=feat(OM[i]); Tbase=T[i].astype(float)
 best=None
 for roll in range(-180,180,15):
  Tr=rollworld(Tbase,roll);Tcam=(Twc@Tr).astype(float)
  # base camera z pca correction after render
  _,M0=render((Wc@Tcam).astype(np.float32));cm,am,Lm,nm=feat(M0);bd=(ao-am+np.pi/2)%np.pi-np.pi/2
  for dtheta in np.linspace(np.degrees(bd)-5,np.degrees(bd)+5,3):
   for sfac in [0.9,1.0,1.1,1.2]:
    scalez=max(.5,min(2.,Lo/Lm))*sfac
    Tc=cam_adjust(Tcam,np.radians(dtheta),scalez)
    _,M1=render((Wc@Tc).astype(np.float32));c1=feat(M1)[0];Tc=cam_adjust(Tc,0,1,co[0]-c1[0],co[1]-c1[1]);rgb,M=render((Wc@Tc).astype(np.float32));sc=eval_mask(M,O,do)
    if best is None or sc[0]<best[0]:best=(*sc,roll,dtheta,scalez,Tc,M,rgb)
 if verbose:print('frame',i,'best score prec rec iou ad roll dtheta scale',best[:8],'obsfeat',ao,Lo,'orig',feat(OM[i])[:3])
 return best
for i in [690,700,710,720,726,732,738,744,750,756,762,768,774,780,786, …[+753 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/roll_search_test.py", line 27, in <module>
    b=search(i,True)
      ^^^^^^^^^^^^^^
  File "/work/scratch/roll_search_test.py", line 22, in search
    _,M1=render((Wc@Tc).astype(np.float32));c1=feat(M1)[0];Tc=cam_adjust(Tc,0,1,co[0]-c1[0],co[1]-c1[1]);rgb,M=render((Wc@Tc).astype(np.float32));sc=eval_mask(M,O,do)
                                                                                                                                                     ^^^^^^^^^^^^^^^^^
  File "<string>", line 20, in eval_mask
NameError: name 'PM_CUR' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/range(690,811,6)/range(600,691,6)/; s/dynamic_montage.jpg/dynamic_montage_600.jpg/' /work/scratch/make_dynamic_montage.py > /work/scratch/make_dynamic_montage_600.py && python /work/scratch/make_dynamic_montage_600.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/roll_icp_test.py <<'PY'
exec(open('/work/scratch/mask_refine_dyn.py').read().split('idxs=')[0])
import cv2,h5py
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
meshcenter=mesh.vertices.mean(0).astype(float)
def rollworld(Tw,d):
 C=np.eye(4);C[:3,3]=meshcenter;Ci=np.eye(4);Ci[:3,3]=-meshcenter
 c,s=np.cos(np.radians(d)),np.sin(np.radians(d));Rz=np.eye(4);Rz[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
 return Tw@C@Rz@Ci

def sample_coords(mask,maxn=5000):
 y,x=np.where(mask); pts=np.c_[x,y].astype(np.float64)
 if len(pts)>maxn: pts=pts[np.linspace(0,len(pts)-1,maxn).astype(int)]
 return pts

def bilinear(D,p):
 x=np.clip(p[:,0],0,D.shape[1]-1.001);y=np.clip(p[:,1],0,D.shape[0]-1.001);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);fx=x-x0;fy=y-y0
 return D[y0,x0]*(1-fx)*(1-fy)+D[y0,x0+1]*fx*(1-fy)+D[y0+1,x0]*(1-fx)*fy+D[y0+1,x0+1]*fx*fy

def align2d(M,O):
 Do=distance_transform_edt(~O);Dm=distance_transform_edt(~M)
 P=sample_coords(M);Q=sample_coords(O);cm=P.mean(0);co=Q.mean(0)
 _,am,lm,_=feat(M);_,ao,lo,_=feat(O);th0=(ao-am+np.pi/2)%np.pi-np.pi/2
 # initial in degrees, dx,dy,log scale
 def obj(z):
  th,s=np.radians(z[0]),np.exp(z[3]);c,sn=np.cos(th),np.sin(th);R=np.array([[c,-sn],[sn,c]])
  pp=(P-cm)@R.T*s+cm+z[:2]; qq=(Q-co-z[:2])@R/s+cm
  prec=bilinear(Do,pp).mean();rec=bilinear(Dm,qq).mean()
  # penalize scale away from length-derived and moderate area mismatch
  lpred=lm*s; areapred=len(P)*s*s
  return …[+1974 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/roll_icp_test.py", line 51, in <module>
    b=search(i,True);res[i]=b
      ^^^^^^^^^^^^^^
  File "/work/scratch/roll_icp_test.py", line 42, in search
    Tc=cam_adjust(Tcam,np.radians(th),scalez);_,M1=render((Wc@Tc).astype(np.float32));c1=feat(M1)[0];Tc=cam_adjust(Tc,0,1,co:=feat(O)[0][0]-c1[0],feat(O)[0][1]-c1[1]);rgb,M2=render((Wc@Tc).astype(np.float32));sc=eval_mask(M2,O)
                                                                                                                                                                                                                    ^^^^^^^^^^^^^^^
  File "<string>", line 20, in eval_mask
NameError: name 'PM_CUR' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/roll_search_test.py';s=open(p).read();s=s.replace(\"PM_CUR=PM[i]; O=OM[i]&~PM[i]\",\"global PM_CUR\\n PM_CUR=PM[i]; O=OM[i]&~PM[i]\").replace(\"O=OM[i];M=b[9].astype\",\"O=OM[i];M=b[9].astype\")
open(p,'w').write(s)
PY
python /work/scratch/roll_search_test.py"
```
<details><summary>output</summary>

```
frame 690 best score prec rec iou ad roll dtheta scale (9.351629177915653, 0.09786546781424739, 8.909617326107796, 0.4150827713059473, 0.11743119266055047, 0, 3.6187037213928437, 0.9310675415906868) obsfeat -0.7374021025252473 274.9065082921582 orig (array([609.17247623, 724.78542327]), -0.7374021025252473, 274.9065082921582)
frame 700 best score prec rec iou ad roll dtheta scale (2.857656331928508, 1.8121169201874228, 1.5127515238658371, 0.27412974683544306, 0.008104738154613466, -150, 2.7372694886436895, 0.6623710902600944) obsfeat -0.49794645615812116 259.3110760274128 orig (array([616.67830986, 676.33915493]), -0.49794645615812116, 259.3110760274128)
frame 710 best score prec rec iou ad roll dtheta scale (16.515256170663573, 3.6514125235213792, 2.5700858525714843, 0.1732430539341583, 1 …[+4735 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fp_candidate_test.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py,cv2
from scipy.ndimage import distance_transform_edt
r2=np.load(f'{work}/inputs/reconstruction.npz');Tf=r2['T_cam_mesh_foundationpose'].astype(float);Tcoco=r2['T_cam_mesh_coconet'].astype(float)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
# candidate camera poses for each frame: baseline world->cam, fp rot + base trans; fp full scaled to base projected length; coco etc
def camera_pose(i):return Twc@T[i].astype(float)
def score_mask(M,O):
 Do=distance_transform_edt(~O);Mv=M&~PM[i]
 if not Mv.any():return 999
 Dm=distance_transform_edt(~Mv)
 return Do[Mv].mean()+Dm[O].mean()+5*abs(Mv.sum()-O.sum())/max(1,O.sum())-2*(Mv&O).sum()/max(1,(Mv|O).sum())
rows=[]
for i in range(650,851,5):
 O=OM[i]&~PM[i];Tb=camera_pose(i)
 cands={'base':Tb}
 # fp rotation at base translation
 for mode in ['baseT','scaleLen']:
  Tfp=Tf[i].copy()
  if mode=='baseT':Tfp[:3,3]=Tb[:3,3]
  else:
   # render at fp, scale camera z based on projected length vs base
   _,Mb=render((Wc@Tb).astype(np.float32));_,Mf=render((Wc@Tfp).astype(np.float32));
   # choose depth ratio via bbox max pixel extent
   yb,xb=np.where(Mb);yf,xf=np.where(Mf);Lb=np.hypot(xb.max()-xb.min(),yb.max()-yb.min());Lf=np.hypot(xf.max()-xf.min(),yf.max()-yf.min());Tfp[:3,3]*=Lf/max(1,Lb)
  cands['fp_'+mode]=Tfp
 for name,Tc in cands.items():
  _ …[+1003 chars]
```
<details><summary>output</summary>

```
(2.4396521639129247, 655, 'base', 3362, 2374)
(2.480864418750256, 800, 'fp_baseT', 3151, 3101)
(2.871263106281791, 685, 'base', 3338, 2352)
(3.2196186941652383, 800, 'base', 3346, 3101)
(3.5951768972804823, 805, 'fp_baseT', 3401, 3217)
(3.6209743750523167, 810, 'fp_baseT', 3407, 3287)
(3.635395986478003, 790, 'fp_baseT', 3068, 3056)
(3.7486663771603794, 810, 'fp_scaleLen', 4287, 3287)
(3.8243581059558855, 805, 'fp_scaleLen', 4703, 3217)
(4.517083596166142, 820, 'fp_baseT', 3739, 3293)
(4.522497503654485, 825, 'base', 3779, 3256)
(4.6432978667242395, 790, 'base', 3143, 3056)
(4.992503720837515, 830, 'fp_baseT', 3730, 3492)
(5.1065388189020435, 790, 'fp_scaleLen', 3154, 3056)
(5.154888350502589, 795, 'fp_baseT', 3299, 2972)
(5.273147786094564, 815, 'fp_baseT', 3469, 3000)
(5.275686898672432, …[+641 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);K=r['human_keypoints_world'];c=r['contact_prob_right_left']
R=T[:,:3,:3]
rh=K[:,21:42].mean(1);lh=K[:,42:63].mean(1);hv=lh-rh;hv/=np.linalg.norm(hv,axis=1,keepdims=True)+1e-9
ax=R[:,:,2]
dot=np.sum(hv*ax,1)
for i in range(0,988,25):
 if max(c[i])>.5:print(i,'contact',np.round(c[i],2),'dot',round(dot[i],2),'handdist',round(np.linalg.norm(lh[i]-rh[i]),3),'mid',np.round((rh[i]+lh[i])/2,2),'objT',np.round(T[i,:3,3],2))
# stats where contact
m=c.max(1)>.5
print('dot pct',np.percentile(np.abs(dot[m]),[0,10,25,50,75,90,100]))
PY"
```
<details><summary>output</summary>

```
contact [0.89 0.65] dot 0.95 handdist 0.365 mid [-0.03  0.55  3.28] objT [-0.15  0.87  3.34]
825 contact [0.11 0.88] dot 0.8 handdist 0.331 mid [-0.01  0.63  3.34] objT [-0.1   0.93  3.49]
850 contact [0.01 1.  ] dot 0.69 handdist 0.25 mid [-0.03  0.64  3.32] objT [-0.07  0.88  3.52]
875 contact [1.   0.35] dot 0.96 handdist 0.283 mid [-0.04  0.65  3.27] objT [-0.08  0.93  3.53]
dot pct [0.00889245 0.60115186 0.80816345 0.92725371 0.9629259  0.98340988
 0.99872663]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:]
rows=[]
for i in range(988):
 ok,im=cap.read();hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=om[i]&(hsv[...,0]<20)&(hsv[...,0]>0)&(hsv[...,1]>120)&(hsv[...,2]>80)
 y,x=np.where(m)
 if len(x):rows.append((i,len(x),x.mean(),y.mean(),x.min(),x.max(),y.min(),y.max()))
 else:rows.append((i,0,np.nan,np.nan,0,0,0,0))
for i in range(0,988,20):
 row=rows[i];print(i,row[1:4])
for i in range(650,900,10):print(rows[i])
PY"
```
<details><summary>output</summary>

```
0 (451, 752.7893569844789, 652.9667405764967)
20 (442, 753.4298642533937, 652.7782805429864)
40 (444, 753.3986486486486, 652.6824324324324)
60 (446, 753.5739910313902, 652.5605381165919)
80 (438, 754.8493150684932, 651.5799086757991)
100 (442, 751.4298642533937, 654.4298642533937)
120 (445, 760.6696629213483, 646.3932584269663)
140 (467, 774.4047109207709, 610.8244111349036)
160 (491, 786.6578411405295, 670.5152749490835)
180 (553, 693.6889692585895, 777.1157323688969)
200 (545, 623.4018348623853, 842.1522935779816)
220 (606, 614.3514851485148, 876.9389438943895)
240 (599, 626.3906510851419, 883.5442404006678)
260 (643, 677.9253499222395, 903.9284603421462)
280 (669, 771.2421524663677, 914.0717488789238)
300 (678, 648.9174041297935, 885.9852507374632)
320 (606, 666.1221122112212, 907.18976 …[+3192 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/roll_icp_test.py';s=open(p).read();s=s.replace(\"def search(i,verbose=False):\\n PM_CUR=PM[i]\", \"def search(i,verbose=False):\\n global PM_CUR\\n PM_CUR=PM[i]\")
s=s.replace(\"Tc=cam_adjust(Tc,0,1,co:=feat(O)[0][0]-c1[0],feat(O)[0][1]-c1[1])\", \"co=feat(O)[0];Tc=cam_adjust(Tc,0,1,co[0]-c1[0],co[1]-c1[1])\")
open(p,'w').write(s)
PY
python /work/scratch/roll_icp_test.py"
```
<details><summary>output</summary>

```
frame 690 score fun roll th dx dy s | eval (39.373362784079475, 25.977983100724384, 15, 3.547397413904427, -5.774603267645061, 355.8946490810504, 0.9487121036548488, (13.395379683355094, 3.7351307043813056, 6.518250183971216, 0.23863363002165022, 0.36192660550458716)) [[-0.74645408  0.4946957   0.44506416 -0.58351242]
 [ 0.28892497  0.84343923 -0.4529156   0.63478865]
 [-0.59943997 -0.20949062 -0.77251958  2.83629726]
 [ 0.          0.          0.          1.        ]]
frame 700 score fun roll th dx dy s | eval (130.31057075927862, 24.536098450088588, 45, 1.169596351776312, -2.2865494296176365, 91.7209810472552, 0.713969379130529, (105.77447230919003, 75.94625541091142, 0.699714763809025, 0.15831751131882577, 2.9445137157107233)) [[ 0.03866359  0.54324947  0.83868084 -0.56123724]
 [ 0.5082 …[+7343 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/roll_orange_test.py <<'PY'
exec(open('/work/scratch/mask_refine_dyn.py').read().split('idxs=')[0])
import cv2,h5py
from scipy.ndimage import distance_transform_edt
meshcenter=mesh.vertices.mean(0).astype(float)
def rollworld(Tw,d):
 C=np.eye(4);C[:3,3]=meshcenter;Ci=np.eye(4);Ci[:3,3]=-meshcenter
 c,s=np.cos(np.radians(d)),np.sin(np.radians(d));Rz=np.eye(4);Rz[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
 return Tw@C@Rz@Ci
# Load frames orange masks
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');OR=[]
for i in range(988):
 ok,im=cap.read();hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);m=OM[i]&(hsv[...,0]<20)&(hsv[...,0]>0)&(hsv[...,1]>120)&(hsv[...,2]>80);OR.append(m)
OR=np.array(OR)
def orange_render(rgb,mask):
 # unlit texture, use relative red dominance
 return mask&(rgb[...,0]>rgb[...,1]*1.35)&(rgb[...,0]>rgb[...,2]*1.25)&(rgb[...,0]>.10)

def search(i):
 PM_CUR=PM[i];O=OM[i]&~PM[i];Oraw=OM[i];orobs=OR[i];co=feat(Oraw)[0];do=distance_transform_edt(~O);rows=[]
 for roll in range(-180,180,10):
  Tcam=(Twc@rollworld(T[i].astype(float),roll)).astype(float);rgb,M=render((Wc@Tcam).astype(np.float32));
  # rotate camera z to PCA angle match based on full masks, then translate full centroid
  _,am,_,_=feat(M);_,ao,_,_=feat(Oraw);dt=(ao-am+np.pi/2)%np.pi-np.pi/2;Tc=cam_adjust(Tcam,dt);rgb,M=render((Wc@Tc).astype(np.float32));cm=feat(M)[0];Tc=cam_adjust(Tc,0,1,co[0]-cm[0],co[1]-cm[1]);rgb,M=render((Wc@Tc).astype(np.float32));Mv=M&~PM[i]
  if not Mv.any():conti …[+1546 chars]
```
<details><summary>output</summary>

```

frame 700 obsorange 138
  (36.387, 50, 28.235, 1.183, 0.27, 283, 138, 115.658, 0.0)
  (36.5, 40, 28.535, 1.151, 0.273, 263, 138, 122.341, 0.0)
  (36.614, 30, 29.697, 1.141, 0.267, 236, 138, 89.042, 0.0)
  (38.08, 20, 31.332, 1.156, 0.265, 200, 138, 92.843, 0.0)
  (38.844, -170, 32.557, 1.312, 0.246, 170, 138, 78.212, 0.0)

frame 710 obsorange 191
  (9.893, 100, 4.303, 4.931, 0.315, 187, 191, 26.739, 0.013)
  (10.149, 90, 3.469, 5.241, 0.202, 155, 191, 20.775, 0.003)
  (10.238, -90, 3.916, 5.022, 0.227, 155, 191, 27.097, 0.0)
  (11.953, -100, 4.639, 6.336, 0.195, 173, 191, 30.461, 0.0)
  (12.361, -80, 5.449, 5.851, 0.271, 195, 191, 28.543, 0.032)

frame 720 obsorange 250
  (12.817, 30, 1.806, 6.654, 0.343, 106, 250, 52.997, 0.0)
  (12.939, -150, 2.198, 6.762, 0.357, 62, 250, 36.299, 0.0)
  …[+4650 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File('/work/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
def sc(M,O,P):
 V=M&~P;Do=distance_transform_edt(~O);Dm=distance_transform_edt(~V);return Do[V].mean()+Dm[O].mean()+5*abs(V.sum()-O.sum())/max(1,O.sum())-2*(V&O).sum()/max(1,(V|O).sum())
for i in [800,820,840,850,860,880,900,920,940,950,960,980,987]:
 O=OM[i]&~PM[i];vals={}
 for name,j in [('base',i),('T850',850),('T900',900),('T960',960),('med')]:
  if name=='med':
   # approximate median pose orientation via scipy? use T960 trans
   continue
  _,M=render(T[j]);vals[name]=round(sc(M,O,PM[i]),2)
 print(i,'area',O.sum(),vals)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File('/work/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
def sc(M,O,P):
 V=M&~P;Do=distance_transform_edt(~O);Dm=distance_transform_edt(~V);return Do[V].mean()+Dm[O].mean()+5*abs(V.sum()-O.sum())/max(1,O.sum())-2*(V&O).sum()/max(1,(V|O).sum())
for i in [800,820,840,850,860,880,900,920,940,950,960,980,987]:
 O=OM[i]&~PM[i];vals={}
 for name,j in [('base',i),('T850',850),('T900',900),('T960',960)]:
  _,M=render(T[j]);vals[name]=round(sc(M,O,PM[i]),2)
 print(i,'area',O.sum(),vals)
PY"
```
<details><summary>output</summary>

```
800 area 3101 {'base': 3.22, 'T850': 30.92, 'T900': 16.13, 'T960': 39.86}
820 area 3293 {'base': 5.28, 'T850': 7.13, 'T900': 5.06, 'T960': 25.59}
840 area 3606 {'base': 13.96, 'T850': 4.05, 'T900': 5.76, 'T960': 14.51}
850 area 3485 {'base': 8.61, 'T850': 8.61, 'T900': 6.39, 'T960': 24.42}
860 area 3588 {'base': 8.09, 'T850': 14.08, 'T900': 5.59, 'T960': 22.83}
880 area 3667 {'base': 7.5, 'T850': 14.66, 'T900': 5.12, 'T960': 22.47}
900 area 5242 {'base': 14.5, 'T850': 16.37, 'T900': 14.5, 'T960': 14.19}
920 area 5104 {'base': 11.22, 'T850': 16.56, 'T900': 7.45, 'T960': 14.01}
940 area 5261 {'base': 8.61, 'T850': 18.07, 'T900': 8.09, 'T960': 14.92}
950 area 5877 {'base': 22.11, 'T850': 20.53, 'T900': 10.01, 'T960': 16.8}
960 area 5533 {'base': 16.56, 'T850': 20.23, 'T900': 10.42, 'T960': 16 …[+155 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_pooled_end.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
# cache DT and scores
for a,b,step in [(840,988,4),(880,988,4),(900,988,4),(920,988,4)]:
 inds=list(range(a,b,step));dt=[]
 for i in inds:dt.append(distance_transform_edt(~(OM[i]&~PM[i])))
 rows=[]
 for j in range(850,988,2):
  _,M=render(T[j]);ss=[]
  for ii,i in zip(dt,inds):
   V=M&~PM[i];Dm=distance_transform_edt(~V)
   s=ii[V].mean()+Dm[OM[i]&~PM[i]].mean()+5*abs(V.sum()-(OM[i]&~PM[i]).sum())/max(1,(OM[i]&~PM[i]).sum())-2*(V&(OM[i]&~PM[i])).sum()/max(1,(V|(OM[i]&~PM[i])).sum())
   ss.append(s)
  rows.append((np.mean(ss),np.median(ss),j,np.percentile(ss,90)))
 print('\\nrange',a,b)
 for x in sorted(rows)[:20]:print(x)
PY
python /work/scratch/static_pooled_end.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_pooled_end.py", line 11, in <module>
    _,M=render(T[j]);ss=[]
        ^^^^^^^^^^^^
  File "<string>", line 14, in render
TypeError: ones_like(): argument 'input' (position 1) must be Tensor, not numpy.ndarray

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/static_scores.py | sed -n '1,25p'"
```
<details><summary>output</summary>

```
     1	import numpy as np, torch, trimesh, cv2, json, h5py, time
     2	import nvdiffrast.torch as dr
     3	work='/work'; dev='cuda'; H,W=1152,1536
     4	r=np.load(f'{work}/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float32)
     5	cam=json.load(open(f'{work}/inputs/camera.json')); K=np.array(cam['K'],np.float32); Wc=np.array(cam['T_world_cam'],np.float32); Twc=np.linalg.inv(Wc).astype(np.float32)
     6	mesh=trimesh.load(f'{work}/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
     7	V=torch.tensor(mesh.vertices,dtype=torch.float32,device=dev); F=torch.tensor(mesh.faces,dtype=torch.int32,device=dev)
     8	uv=np.asarray(mesh.visual.uv).copy(); uv[:,1]=1-uv[:,1]; UV=torch.tensor(uv,dtype=torch.float32,device=dev)
     9	tex=np.array(mesh.visual.material …[+1095 chars]
```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_pooled_end.py';s=open(p).read();s=s.replace(\"V=M&~PM[i];Dm=distance_transform_edt(~V)\",\"VM=M&~PM[i];Dm=distance_transform_edt(~VM)\").replace(\"s=ii[V].mean()+Dm[OM[i]&~PM[i]].mean()+5*abs(V.sum()-(OM[i]&~PM[i]).sum())/max(1,(OM[i]&~PM[i]).sum())-2*(V&(OM[i]&~PM[i])).sum()/max(1,(V|(OM[i]&~PM[i])).sum())\",\"s=ii[VM].mean()+Dm[OM[i]&~PM[i]].mean()+5*abs(VM.sum()-(OM[i]&~PM[i]).sum())/max(1,(OM[i]&~PM[i]).sum())-2*(VM&(OM[i]&~PM[i])).sum()/max(1,(VM|(OM[i]&~PM[i])).sum())\")
open(p,'w').write(s)
PY
python /work/scratch/static_pooled_end.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_pooled_end.py", line 13, in <module>
    VM=M&~PM[i];Dm=distance_transform_edt(~VM)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/ndimage/_morphology.py", line 2585, in distance_transform_edt
    _nd_image.euclidean_feature_transform(input, sampling, ft)
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_pooled_end2.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM0=h['object'][:];PM0=h['person'][:]
x0,x1,y0,y1=680,930,560,1080
OM=OM0[:,y0:y1,x0:x1];PM=PM0[:,y0:y1,x0:x1]
for a,b,step in [(840,988,8),(900,988,4),(920,988,4)]:
 inds=list(range(a,b,step));data=[]
 for i in inds:
  O=OM[i]&~PM[i];data.append((i,O,distance_transform_edt(~O)))
 rows=[]
 for j in range(850,988,4):
  _,Mfull=render(T[j]);M=Mfull[y0:y1,x0:x1];ss=[]
  for i,O,do in data:
   VM=M&~PM[i]
   if not VM.any():continue
   Dm=distance_transform_edt(~VM);area=O.sum()
   s=do[VM].mean()+Dm[O].mean()+5*abs(VM.sum()-area)/max(1,area)-2*(VM&O).sum()/max(1,(VM|O).sum())
   ss.append(s)
  rows.append((np.mean(ss),np.median(ss),j,np.percentile(ss,90)))
 print('\\nrange',a,b)
 for x in sorted(rows)[:20]:print(x)
PY
python /work/scratch/static_pooled_end2.py"
```
<details><summary>output</summary>

```

range 840 988
(6.1054075450805065, 5.415535102239965, 898, 7.897427316606547)
(6.901153508208919, 7.0396110300902395, 874, 9.163118226578181)
(7.136601797060087, 7.6257620166140265, 894, 10.742965180056835)
(7.826002369023588, 8.078525109624612, 886, 10.327984636578527)
(7.940666511657738, 7.650664778639971, 902, 10.67839277708712)
(7.958494279615442, 7.966308483931503, 866, 11.530214119432177)
(8.353718373334576, 8.862800296496435, 858, 12.93183710627127)
(8.532080230925663, 7.831181578413442, 870, 10.880069423152893)
(8.737398448341551, 9.226912400413855, 890, 12.243552252333911)
(8.882387015921003, 9.456132154862544, 906, 12.765813939686074)
(8.93668017122493, 8.44419335708369, 882, 11.088463474574294)
(9.409886539426605, 9.831365848716269, 854, 14.7868934109305)
(10.012926562870241, 9 …[+3112 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/smooth_eval.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
# Preprocess masks crop based union whole image maybe downsample none
# candidate static poses
Tstart=T[60].astype(float).copy() # b60 texture-selected
Tend=T[898].astype(float).copy()
# translation anchor median 0:110 and 850:988 except not used orientation
Tstart[:3,3]=np.median(T[0:111,:3,3],0)
Tend[:3,3]=np.median(T[850:988,:3,3],0)
q=Rotation.from_matrix(T[:,:3,:3]).as_quat()
P=T[:,:3,3].copy()
def smooth_rot(w,thr):
 qo=np.empty_like(q);h=w//2
 for i in range(len(q)):
  a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();s=np.sign(qq@q[i]);s[s==0]=1;qq*=s[:,None]
  dots=np.clip(np.abs(qq@q[i]),-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq))
  bad=ang>np.radians(thr);ww[bad]=np.radians(thr)/ang[bad]
  m=(qq*ww[:,None]).sum(0);qo[i]=m/np.linalg.norm(m)
 return Rotation.from_quat(qo).as_matrix()
def smooth_trans(w):return np.array([np.median(P[max(0,i-w//2):min(len(P),i+w//2+1)],0) for i in range(len(P))])
# eval indices
idxs=list(range(110,841,6))
# evaluate variants including baseline
variants=[('base',T)]
for w in [3,5,7,9,13,17,21,31]:
 for thr in [8,15,30,180]:
  Rs=smooth_rot(w,thr);Tv=T.copy();Tv[:,:3,:3]=Rs
  if w>1:Tv[:,:3,3]=smooth_trans(w) …[+903 chars]
```
<details><summary>output</summary>

```
variants 33 frames 122
Traceback (most recent call last):
  File "/work/scratch/smooth_eval.py", line 48, in <module>
    _,M=render(Tv[i].astype(np.float32));VM=M&~PM[i]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 15, in render
TypeError: unsupported operand type(s) for @: 'numpy.ndarray' and 'Tensor'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'grep -n "''^V'"\\| V=\" /work/scratch/smooth_eval.py | head"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/P=T\\[:,:3,3\\]/POS=T[:,:3,3]/; s/P\\[max(0/POS[max(0/g; s/P,0)/POS,0)/g' /work/scratch/smooth_eval.py && grep -n 'POS\\|P\\[' /work/scratch/smooth_eval.py | head && python /work/scratch/smooth_eval.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/smooth_eval.py", line 31, in <module>
    if w>1:Tv[:,:3,3]=smooth_trans(w)
           ~~^^^^^^^^
ValueError: could not broadcast input array from shape (4,3) into shape (988,3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/len(P)/len(POS)/g' /work/scratch/smooth_eval.py && python /work/scratch/smooth_eval.py"
```
<details><summary>output</summary>

```
variants 33 frames 122
0 base (19.562651749855828, 8.61113519438432, 0.28976299696916213, 0.2107305765606726)
1 w3t8 (18.743953925670027, 7.742697364611989, 0.2889874915768452, 0.20958268823721646)
2 w3t15 (18.182669069784886, 7.742697364611989, 0.28923496065759907, 0.2081866615299227)
3 w3t30 (18.111339306260962, 7.742697364611989, 0.2894821315124999, 0.20614644772202603)
4 w3t180 (18.365860510785936, 7.742697364611989, 0.2896549986589685, 0.20572801384410228)
5 w5t8 (19.569114780421238, 7.565328254337761, 0.28844776648998394, 0.21117061438243362)
6 w5t15 (19.09588943802481, 7.61137031488499, 0.28890604833517447, 0.20957514010686887)
7 w5t30 (18.58313193369963, 7.653445788229693, 0.288886208565531, 0.20850144963717235)
8 w5t180 (18.56710321587658, 7.653445788229693, 0.2887520409195231, 0. …[+5159 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py,json
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt
r=np.load('/work/inputs/reconstruction.npz');Tf=r['T_cam_mesh_foundationpose'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
def sc(Tc,i):
 _,M=render((Wc@Tc).astype(np.float32));VM=M&~PM[i];O=OM[i]&~PM[i]
 if not VM.any():return 999
 Do=distance_transform_edt(~O);Dm=distance_transform_edt(~VM);a=max(1,O.sum())
 return Do[VM].mean()+Dm[O].mean()+5*abs(VM.sum()-a)/a
for i in range(740,851,5):
 Tb=Twc@T[i].astype(float);Tfp=Tf[i].copy();Tfp[:3,3]=Tb[:3,3]
 print(i,round(sc(Tb,i),2),round(sc(Tfp,i),2))
PY"
```
<details><summary>output</summary>

```
740 47.42 48.99
745 205.82 74.49
750 121.78 100.87
755 25.44 30.41
760 78.01 125.81
765 52.92 55.27
770 80.53 44.3
775 36.12 17.83
780 9.43 13.13
785 7.02 7.75
790 5.44 4.57
795 7.79 5.97
800 4.09 3.34
805 12.34 4.44
810 12.05 4.43
815 6.63 5.81
820 5.95 5.23
825 5.23 6.51
830 7.72 5.96
835 13.83 8.96
840 14.16 12.31
845 6.44 9.62
850 9.23 12.52

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_static_trans.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]

def consensus(a,b):
 ob=OM[a:b].mean(0)>.18; pe=PM[a:b].mean(0)<.35; return ob&pe

def refine(Tpose,a,b,crop,name):
 x0,x1,y0,y1=crop;O=consensus(a,b)[y0:y1,x0:x1];Do=distance_transform_edt(~O)
 Tcam=(Twc@Tpose.astype(float)).astype(float);best=None
 # coarse around current camera translation
 for dz in np.arange(-.08,.081,.02):
  for dy in np.arange(-.04,.041,.01):
   for dx in np.arange(-.04,.041,.01):
    Tc=Tcam.copy();Tc[:3,3]+=[dx,dy,dz]
    _,M=render((Wc@Tc).astype(np.float32));M=M[y0:y1,x0:x1]
    if not M.any():continue
    Dm=distance_transform_edt(~M);area=O.sum()
    s=Do[M].mean()+Dm[O].mean()+5*abs(M.sum()-area)/area-2*(M&O).sum()/max(1,(M|O).sum())
    if best is None or s<best[0]:best=(s,dx,dy,dz,M,Tc)
 print(name,'coarse',best[:4])
 s,dx,dy,dz,_,Tc=best
 # fine
 for ddz in np.arange(dz-.015,dz+.0151,.005):
  for ddy in np.arange(dy-.0075,dy+.0076,.0025):
   for ddx in np.arange(dx-.0075,dx+.0076,.0025):
    Tt=Tcam.copy();Tt[:3,3]+=[ddx,ddy,ddz]
    _,M=render((Wc@Tt).astype(np.float32));M=M[y0:y1,x0:x1]
    if not M.any():continue
    Dm=distance_transform_edt(~M);area=O.sum()
    ss=Do[M].mean()+Dm[O].mean()+5*abs(M.sum()-area)/area-2*(M&O).sum()/max(1,(M|O).sum())
  …[+478 chars]
```
<details><summary>output</summary>

```
start coarse (2.068123128507878, -0.03, -0.009999999999999995, -0.01999999999999999)
start fine (2.055021246518177, -0.024999999999999988, -0.014999999999999994, -0.03499999999999999)
end coarse (3.967156480544851, 6.938893903907228e-18, -0.009999999999999995, -0.08)
end fine (3.891580185016651, 0.002500000000000009, -0.002499999999999992, -0.095)
world start [[ 0.51182032  0.42105631 -0.74883385 -0.33807258]
 [ 0.82730986 -0.00667255  0.56170607  0.63238634]
 [ 0.23151282 -0.90701024 -0.35175887  3.20230702]
 [ 0.          0.          0.          1.        ]]
world end [[ 0.23509925 -0.9707539  -0.04863203  0.19933995]
 [ 0.86353413  0.2315749  -0.44797484  0.83482282]
 [ 0.44613536  0.0633232   0.89272199  2.43324148]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/check_static_refined.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py,cv2
z=np.load(f'{work}/scratch/static_refined.npz');Ts=z['Ts'].astype(np.float32);Te=z['Te'].astype(np.float32)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(988):ok,im=cap.read();frames.append(im)
idxs=[0,20,50,80,100,110,840,860,880,900,920,950,975,987];thumbs=[]
for i in idxs:
 Tw=Ts if i<=110 else Te;rgb,M=render(Tw);O=OM[i];im=frames[i].copy();cnt,_=cv2.findContours(M.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(O.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,255,0),2);u=M|O;y,x=np.where(u);x0=max(0,x.min()-30);x1=min(1536,x.max()+30);y0=max(0,y.min()-30);y1=min(1152,y.max()+30);vis=cv2.resize(im[y0:y1,x0:x1],(300,360));cv2.putText(vis,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);thumbs.append(vis)
rows=[]
for j in range(0,len(thumbs),5):
 row=thumbs[j:j+5]
 while len(row)<5:row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite(f'{work}/scratch/static_refined_check.jpg',np.vstack(rows))
PY
python /work/scratch/check_static_refined.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep '"'^P='"' /work/scratch/static_scores.py; grep '"'^P='"' /work/scratch/mask_refine_dyn.py; grep '"'^P='"' /work/scratch/static_scores_end.py"
```
<details><summary>output</summary>

```
P=torch.tensor([[2*K[0,0]/W,0,2*K[0,2]/W-1,0],[0,2*K[1,1]/H,2*K[1,2]/H-1,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=dev)
P=torch.tensor([[2*K[0,0]/W,0,2*K[0,2]/W-1,0],[0,2*K[1,1]/H,2*K[1,2]/H-1,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=dev)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_static_trans.py';s=open(p).read();s=s.replace(\"np.savez(f'{work}/scratch/static_refined.npz',Ts=bs[5],Te=be[5])\",\"np.savez(f'{work}/scratch/static_refined.npz',Ts=(Wc@bs[5]).astype(np.float32),Te=(Wc@be[5]).astype(np.float32))\")
open(p,'w').write(s)
PY
python /work/scratch/refine_static_trans.py >/work/scratch/refine_static_log.txt && python /work/scratch/check_static_refined.py && tail -5 /work/scratch/refine_static_log.txt"
```
<details><summary>output</summary>

```
 [ 0.          0.          0.          1.        ]]
world end [[ 0.23509925 -0.9707539  -0.04863203  0.19933995]
 [ 0.86353413  0.2315749  -0.44797484  0.83482282]
 [ 0.44613536  0.0633232   0.89272199  2.43324148]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_final_candidate.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);P=T[:,:3,3].copy();q=Rotation.from_matrix(T[:,:3,:3]).as_quat()
z=np.load('/work/scratch/static_refined.npz');Ts=z['Ts'].astype(float);Te=z['Te'].astype(float)
def smooth_rot(w,thr):
 qo=np.empty_like(q);h=w//2
 for i in range(len(q)):
  a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();ss=np.sign(qq@q[i]);ss[ss==0]=1;qq*=ss[:,None];dots=np.clip(np.abs(qq@q[i]),-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq));bad=ang>np.radians(thr);ww[bad]=np.radians(thr)/ang[bad];m=(qq*ww[:,None]).sum(0);qo[i]=m/np.linalg.norm(m)
 return Rotation.from_quat(qo).as_matrix()
def smooth_trans(w):return np.array([np.median(P[max(0,i-w//2):min(len(P),i+w//2+1)],0) for i in range(len(P))])
R=smooth_rot(13,8);PP=smooth_trans(13)
Tv=T.copy();Tv[:,:3,:3]=R;Tv[:,:3,3]=PP
# anchors and simple boundary blending in pose vector? use slerp
Tv[:111]=Ts;Tv[840:]=Te
# optional blend from Ts at 110 to smooth at 130, and smooth at 810 to Te at 840, via quaternion and trans
for lo,hi,A,B in [(110,130,Ts,Tv[130]),(810,840,Tv[810],Te)]:
 qa=Rotation.from_matrix(A[:3,:3]).as_quat();qb=Rotation.from_matrix(B[:3,:3]).as_quat();
 if qa@qb<0:qb=-qb
 for i in range(lo+1,hi):
  u=(i-lo)/(hi-lo);w=u*u*(3-2*u);qq=qa*(1-w)+qb*w;qq/=np.linalg.norm(qq);Tv[i,:3,:3]=Rotation.from_quat(qq).as_matrix();Tv[i,:3,3]=A[:3,3]* …[+712 chars]
```
<details><summary>output</summary>

```
base rot pct [  1.80814084   6.39279665  11.68021969  40.43425974 119.37290556] trans [0.02037286 0.05805859 0.09069126 0.1796144  0.28325719]
top rot [(744, 119.37290555661362), (732, 109.82274165682207), (727, 109.70522301778799), (723, 105.88590699481905), (734, 82.06981288298024), (708, 52.98327942168976), (746, 52.55349220964192), (709, 51.77982974062086), (778, 42.675860407250866), (702, 40.99392746575793), (730, 40.34315104339662), (755, 37.751564805284346), (752, 35.03893252573905), (765, 34.49057598513744), (721, 33.25825381384475)]
top trans [(708, 0.2832571931368536), (704, 0.2572582699576405), (734, 0.25241391059694185), (707, 0.24959281867056413), (732, 0.23404380723672144), (709, 0.2172521605679538), (727, 0.2066218882841704), (722, 0.2025133640893889), (778, 0.20200041832276 …[+869 chars]
```
</details>

**shell** (exit 2)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/smooth_eval.py';s=open(p).read()
old=\"qq=q[a:b].copy();s=np.sign(qq@q[i]);s[s==0]=1;qq*=s[:,None]\\n  dots=np.clip(np.abs(qq@q[i]),-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq));bad=ang>np.radians(thr);ww[bad]=np.radians(thr)/ang[bad]\"
new=\"qq=q[a:b].copy();ref=qq[np.argmax(np.abs(qq@q[a:b]).sum(1))];s=np.sign(qq@ref);s[s==0]=1;qq*=s[:,None];ref=ref*np.sign(ref@q[i])\\n  dots=np.clip(qq@ref,-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq));bad=ang>np.radians(thr);ww[bad]=np.radians(thr)/ang[bad]\"
assert old in s
s=s.replace(old,new)
# variants fewer
s=s.replace(\"for w in [3,5,7,9,13,17,21,31]:\",\"for w in [5,9,13,17,21]:\")
open('/work/scratch/smooth_eval_medoid.py','w').write(s)
PY
python /work/scratch/smooth_eval_medoid.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
AssertionError
python: can't open file '/work/scratch/smooth_eval_medoid.py': [Errno 2] No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '15,24p' /work/scratch/smooth_eval.py"
```
<details><summary>output</summary>

```
def smooth_rot(w,thr):
 qo=np.empty_like(q);h=w//2
 for i in range(len(q)):
  a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();s=np.sign(qq@q[i]);s[s==0]=1;qq*=s[:,None]
  dots=np.clip(np.abs(qq@q[i]),-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq))
  bad=ang>np.radians(thr);ww[bad]=np.radians(thr)/ang[bad]
  m=(qq*ww[:,None]).sum(0);qo[i]=m/np.linalg.norm(m)
 return Rotation.from_quat(qo).as_matrix()
def smooth_trans(w):return np.array([np.median(POS[max(0,i-w//2):min(len(POS),i+w//2+1)],0) for i in range(len(POS))])
# eval indices

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/smooth_eval.py';s=open(p).read()
s=s.replace(\"a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();s=np.sign(qq@q[i]);s[s==0]=1;qq*=s[:,None]\\n  dots=np.clip(np.abs(qq@q[i]),-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq))\", \"a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();ref=qq[np.argmax(np.abs(qq@q[a:b]).sum(1))];s=np.sign(qq@ref);s[s==0]=1;qq*=s[:,None];ref=ref*np.sign(ref@q[i])\\n  dots=np.clip(qq@ref,-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq))\")
s=s.replace(\"for w in [3,5,7,9,13,17,21,31]:\",\"for w in [5,9,13,17,21]:\")
open('/work/scratch/smooth_eval_medoid.py','w').write(s)
PY
python /work/scratch/smooth_eval_medoid.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/smooth_eval_medoid.py", line 30, in <module>
    Rs=smooth_rot(w,thr);Tv=T.copy();Tv[:,:3,:3]=Rs
       ^^^^^^^^^^^^^^^^^
  File "/work/scratch/smooth_eval_medoid.py", line 18, in smooth_rot
    a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();ref=qq[np.argmax(np.abs(qq@q[a:b]).sum(1))];s=np.sign(qq@ref);s[s==0]=1;qq*=s[:,None];ref=ref*np.sign(ref@q[i])
                                                                              ~~^~~~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/np.abs(qq@q\\[a:b\\])/np.abs(qq@qq.T)/' /work/scratch/smooth_eval_medoid.py && python /work/scratch/smooth_eval_medoid.py"
```
<details><summary>output</summary>

```
variants 21 frames 122
0 base (19.562651749855828, 8.61113519438432, 0.28976299696916213, 0.2107305765606726)
1 w5t8 (18.653376124370983, 7.5868646114316, 0.2894972491639594, 0.20612551795780545)
2 w5t15 (18.62596047944465, 7.653445788229693, 0.28877275591893053, 0.20663190428425296)
3 w5t30 (18.19072068544143, 7.653445788229693, 0.2887724229599869, 0.20702924305820247)
4 w5t180 (18.56710321587658, 7.653445788229693, 0.2887520409195231, 0.20374494360501183)
5 w9t8 (16.822864633262565, 7.234302760143107, 0.29717193439265527, 0.1960846928123474)
6 w9t15 (16.803130918503857, 7.234302760143107, 0.29727913901199426, 0.19440890795105098)
7 w9t30 (16.799169046476372, 7.368828087283783, 0.2976996089578631, 0.19521795178173926)
8 w9t180 (17.618071464459494, 7.368828087283783, 0.2969448043858821, 0. …[+3005 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_final_candidate2.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);POS=T[:,:3,3].copy();q=Rotation.from_matrix(T[:,:3,:3]).as_quat()
z=np.load('/work/scratch/static_refined.npz');Ts=z['Ts'].astype(float);Te=z['Te'].astype(float)
def smooth_rot(w,thr):
 qo=np.empty_like(q);h=w//2
 for i in range(len(q)):
  a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();ref=qq[np.argmax(np.abs(qq@qq.T).sum(1))];ss=np.sign(qq@ref);ss[ss==0]=1;qq*=ss[:,None];ref=ref*np.sign(ref@q[i]);dots=np.clip(qq@ref,-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq));bad=ang>np.radians(thr);ww[bad]=np.radians(thr)/ang[bad];m=(qq*ww[:,None]).sum(0);qo[i]=m/np.linalg.norm(m)
 return Rotation.from_quat(qo).as_matrix()
def smooth_trans(w):return np.array([np.median(POS[max(0,i-w//2):min(len(POS),i+w//2+1)],0) for i in range(len(POS))])
Rm=smooth_rot(13,8);Pm=smooth_trans(13)
Tv=T.copy();Tv[:,:3,:3]=Rm;Tv[:,:3,3]=Pm;Tv[:111]=Ts;Tv[840:]=Te
# blend dynamic endpoints into static anchors
for lo,hi,A,B in [(110,130,Ts,Tv[130]),(810,840,Tv[810],Te)]:
 qa=Rotation.from_matrix(A[:3,:3]).as_quat();qb=Rotation.from_matrix(B[:3,:3]).as_quat();
 if qa@qb<0:qb=-qb
 for i in range(lo+1,hi):
  u=(i-lo)/(hi-lo);w=u*u*(3-2*u);qq=qa*(1-w)+qb*w;qq/=np.linalg.norm(qq);Tv[i,:3,:3]=Rotation.from_quat(qq).as_matrix();Tv[i,:3,3]=A[:3,3]*(1-w)+B[:3,3]*w
for name,XX in [('base',T),('smooth',Tv)] …[+455 chars]
```
<details><summary>output</summary>

```
base rot pct [  1.80814084   6.39279665  11.68021969  40.43425974 119.37290556] trans [0.02037286 0.05805859 0.09069126 0.1796144  0.28325719]
top rot [(744, 119.4), (732, 109.8), (727, 109.7), (723, 105.9), (734, 82.1), (708, 53.0), (746, 52.6), (709, 51.8), (778, 42.7), (702, 41.0), (730, 40.3), (755, 37.8), (752, 35.0), (765, 34.5), (721, 33.3)]
smooth rot pct [  0.48210827   2.01032721   2.94023895  18.51458372 114.48886104] trans [0.00695319 0.02497886 0.0322794  0.06461068 0.11200286]
top rot [(743, 114.5), (735, 66.1), (711, 45.3), (728, 41.8), (707, 38.7), (727, 37.7), (723, 34.9), (732, 30.7), (753, 25.1), (750, 23.6), (755, 17.7), (714, 15.8), (729, 14.9), (133, 13.1), (708, 11.5)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
z=np.load('/work/scratch/final_candidate2.npz');S=z['T'];r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float)
def dr(a,b):return np.degrees(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
for i in range(730,760):
 print(i,'base-prev',round(dr(B[i-1],B[i]),1),'base-next',round(dr(B[i],B[i+1]),1),'smooth-base',round(dr(S[i],B[i]),1),'smooth-prev',round(dr(S[i-1],S[i]),1),'P',np.round(B[i,:3,3],2))
PY"
```
<details><summary>output</summary>

```
730 base-prev 0.0 base-next 40.3 smooth-base 71.8 smooth-prev 14.9 P [0.29 0.48 2.92]
731 base-prev 40.3 base-next 0.0 smooth-base 72.5 smooth-prev 7.0 P [0.42 0.56 2.98]
732 base-prev 0.0 base-next 109.8 smooth-base 76.8 smooth-prev 7.2 P [0.4  0.58 3.01]
733 base-prev 109.8 base-next 0.0 smooth-base 17.8 smooth-prev 30.7 P [0.48 0.38 2.92]
734 base-prev 0.0 base-next 82.1 smooth-base 12.8 smooth-prev 8.3 P [0.45 0.42 2.91]
735 base-prev 82.1 base-next 0.1 smooth-base 69.5 smooth-prev 4.8 P [0.32 0.59 3.05]
736 base-prev 0.1 base-next 0.1 smooth-base 3.5 smooth-prev 66.1 P [0.29 0.7  2.95]
737 base-prev 0.1 base-next 0.1 smooth-base 2.9 smooth-prev 1.4 P [0.27 0.68 3.03]
738 base-prev 0.1 base-next 0.1 smooth-base 2.7 smooth-prev 1.4 P [0.21 0.72 3.04]
739 base-prev 0.1 base-next 0.1 smoo …[+1745 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);q=Rotation.from_matrix(T[:,:3,:3]).as_quat();i=744;w=13;h=6;a=i-h;b=i+h+1;qq=q[a:b].copy();scores=np.abs(qq@qq.T).sum(1);ref=qq[np.argmax(scores)];print('med idx',a+np.argmax(scores),scores.round(1));print('angles to ref',np.degrees(2*np.arccos(np.clip(np.abs(qq@ref),-1,1))).round(1));print('angles to qi',np.degrees(2*np.arccos(np.clip(np.abs(qq@q[i]),-1,1))).round(1))
PY"
```
<details><summary>output</summary>

```
med idx 745 [8.  8.  8.  8.  8.  7.9 8.1 8.3 8.3 6.3 6.3 6.3 6.3]
angles to ref [138.8 138.8 138.8 138.8 138.8 124.7 119.4   0.    0.   52.6  52.6  52.5
  52.5]
angles to qi [ 33.3  33.3  33.3  33.3  33.3  15.9   0.  119.4 119.4 171.9 171.9 171.9
 171.9]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/roll_table.py <<'PY'
exec(open('/work/scratch/roll_orange_test.py').read().split('for i in')[0])
from scipy.ndimage import distance_transform_edt
for i in [720,744,750,756,768,800]:
 PM_CUR=PM[i];O=OM[i]&~PM[i];Oraw=OM[i];co=feat(Oraw)[0];do=distance_transform_edt(~O);rows=[]
 for roll in range(-180,180,15):
  Tcam=(Twc@rollworld(T[i].astype(float),roll)).astype(float);rgb,M=render((Wc@Tcam).astype(np.float32));_,am,_,_=feat(M);_,ao,_,_=feat(Oraw);dt=(ao-am+np.pi/2)%np.pi-np.pi/2;Tc=cam_adjust(Tcam,dt);rgb,M=render((Wc@Tc).astype(np.float32));cm=feat(M)[0];Tc=cam_adjust(Tc,0,1,co[0]-cm[0],co[1]-cm[1]);rgb,M=render((Wc@Tc).astype(np.float32));Mv=M&~PM[i];Dm=distance_transform_edt(~Mv);prec=do[Mv].mean();rec=Dm[O].mean();mi=(Mv&O).sum()/max(1,(Mv|O).sum());rows.append((roll,Mv.sum(),prec,rec,mi))
 print('\\n',i,'obs',O.sum())
 for r in rows:print('%4d area%5d prec%6.2f rec%6.2f iou%.2f'%r)
PY
python /work/scratch/roll_table.py"
```
<details><summary>output</summary>

```

 720 obs 1669
-180 area  763 prec  7.78 rec 11.10 iou0.10
-165 area  869 prec  3.79 rec  7.68 iou0.17
-150 area  968 prec  2.20 rec  6.76 iou0.36
-135 area 1044 prec  3.90 rec  9.62 iou0.20
-120 area  984 prec  8.31 rec 15.45 iou0.09
-105 area  668 prec 12.85 rec 21.91 iou0.03
 -90 area  361 prec  9.16 rec 31.80 iou0.02
 -75 area  961 prec 26.71 rec 30.32 iou0.03
 -60 area 1124 prec 19.40 rec 27.53 iou0.05
 -45 area  796 prec  9.49 rec 25.32 iou0.07
 -30 area  823 prec 13.52 rec 22.09 iou0.09
 -15 area  816 prec 12.23 rec 17.41 iou0.10
   0 area  799 prec  8.25 rec 11.99 iou0.11
  15 area  867 prec  3.40 rec  7.65 iou0.20
  30 area  901 prec  1.81 rec  6.65 iou0.34
  45 area  896 prec  3.72 rec  9.42 iou0.19
  60 area  960 prec  6.91 rec 13.29 iou0.15
  75 area 1180 prec 14.07 rec 15.79 i …[+5626 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/roll_select_all.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py,time
from scipy.ndimage import distance_transform_edt
meshcenter=mesh.vertices.mean(0).astype(float)
def rollworld(Tw,d):
 C=np.eye(4);C[:3,3]=meshcenter;Ci=np.eye(4);Ci[:3,3]=-meshcenter
 c,s=np.cos(np.radians(d)),np.sin(np.radians(d));Rz=np.eye(4);Rz[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
 return Tw@C@Rz@Ci
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
Tout=T.copy();rolls=np.zeros(len(T));scores=np.zeros(len(T));base_scores=np.zeros(len(T))
def eval_candidate(i,roll):
 PM_CUR=PM[i];O=OM[i]&~PM[i];Oraw=OM[i];co=feat(Oraw)[0];do=distance_transform_edt(~O)
 Tcam=(Twc@rollworld(T[i].astype(float),roll)).astype(float);rgb,M=render((Wc@Tcam).astype(np.float32));
 _,am,_,_=feat(M);_,ao,_,_=feat(Oraw);dt=(ao-am+np.pi/2)%np.pi-np.pi/2;Tc=cam_adjust(Tcam,dt);rgb,M=render((Wc@Tc).astype(np.float32));cm=feat(M)[0];Tc=cam_adjust(Tc,0,1,co[0]-cm[0],co[1]-cm[1]);rgb,M=render((Wc@Tc).astype(np.float32));VM=M&~PM[i]
 if not VM.any():return 999,Tc,M,rgb
 Dm=distance_transform_edt(~VM);area=max(1,O.sum());prec=do[VM].mean();rec=Dm[O].mean();ad=abs(VM.sum()-area)/area;iou=(VM&O).sum()/max(1,(VM|O).sum())
 # texture ncc in visible overlap, mean-centered channels
 valid=VM&O;ncc=0
 if valid.sum()>50:
  rgbim=frames_cur[valid];a=rgb[valid];aa=a-a.mean(0);bb=rgbim-bbmean if False else rgbim-rgbim.mean(0)
  ncc= …[+1255 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/roll_select_all.py", line 30, in <module>
    bs,_,_,_=eval_candidate(i,0);base_scores[i]=bs
             ^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/roll_select_all.py", line 12, in eval_candidate
    PM_CUR=PM[i];O=OM[i]&~PM[i];Oraw=OM[i];co=feat(Oraw)[0];do=distance_transform_edt(~O)
                                              ^^^^
NameError: name 'feat' is not defined. Did you mean: 'float'?

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/roll_select_all.py';s=open(p).read();s=s.replace(\"exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])\",\"exec(open('/work/scratch/mask_refine_dyn.py').read().split('idxs=')[0])\")
open(p,'w').write(s)
PY
python /work/scratch/roll_select_all.py"
```
<details><summary>output</summary>

```
frame 120
Traceback (most recent call last):
  File "/work/scratch/roll_select_all.py", line 37, in <module>
    sc,Tc,M,rgb=eval_candidate(i,roll)
^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/ndimage/_morphology.py", line 2589, in distance_transform_edt
    dt = dt.astype(np.float64)
         ^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/roll_anchor_select.py <<'PY'
exec(open('/work/scratch/mask_refine_dyn.py').read().split('idxs=')[0])
import h5py
from scipy.ndimage import distance_transform_edt
meshcenter=mesh.vertices.mean(0).astype(float)
x0,x1,y0,y1=350,1050,500,1120
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
def rollworld(Tw,d):
 C=np.eye(4);C[:3,3]=meshcenter;Ci=np.eye(4);Ci[:3,3]=-meshcenter;c,s=np.cos(np.radians(d)),np.sin(np.radians(d));Rz=np.eye(4);Rz[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];return Tw@C@Rz@Ci
def eval_c(i,roll,im):
 Oall=OM[i];O=Oall&~PM[i];oc=Oall[y0:y1,x0:x1];ov=O[y0:y1,x0:x1];do=distance_transform_edt(~ov);co=feat(oc)[0];ao=feat(oc)[1]
 Tcam=(Twc@rollworld(T[i].astype(float),roll)).astype(float);_,M0=render((Wc@Tcam).astype(np.float32));mc=M0[y0:y1,x0:x1];cm=feat(mc)[0];am=feat(mc)[1];dt=(ao-am+np.pi/2)%np.pi-np.pi/2;Tc=cam_adjust(Tcam,dt);_,M1=render((Wc@Tc).astype(np.float32));c1=feat(M1[y0:y1,x0:x1])[0];Tc=cam_adjust(Tc,0,1,co[0]-c1[0],co[1]-c1[1]);rgb,M=render((Wc@Tc).astype(np.float32));VM=(M&~PM[i])[y0:y1,x0:x1]
 if not VM.any():return 999,(roll,0,0,0,0,0),Tc,M,rgb
 Dm=distance_transform_edt(~VM);area=max(1,ov.sum());prec=do[VM].mean();rec=Dm[ov].mean();ad=abs(VM.sum()-area)/area;iou=(VM&ov).sum()/max(1,(VM|ov).sum())
 valid=VM&ov;ncc=0
 if valid.sum()>50:
  a=rgb[y0:y1,x0:x1][valid];b=im[y0:y1,x0:x1][valid];aa=a-a.mean(0);bb=b-b.mean(0);ncc=((aa*bb).sum(0)/(np.sqrt((aa*aa).sum(0)*(bb*bb).sum(0))+1e …[+1171 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/roll_anchor_select.py", line 31, in <module>
    if s2<sc:sc,roll,e,Tc,M,rgb=s2,rr,e2,T2,rgb2
             ^^^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 6, got 5)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/roll_anchor_select.py';s=open(p).read();s=s.replace(\"s2,e2,T2,M2,rgb2=eval_c(i,rr,im)\",\"sc2,e2,T2,M2,rgb2=eval_c(i,rr,im)\").replace(\"if s2<sc:sc,roll,e,Tc,M,rgb=s2,rr,e2,T2,rgb2\",\"if sc2<sc:sc,roll,e,Tc,M,rgb=sc2,rr,e2,T2,rgb2\")
open(p,'w').write(s)
PY
python /work/scratch/roll_anchor_select.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/roll_anchor_select.py", line 31, in <module>
    if sc2<sc:sc,roll,e,Tc,M,rgb=sc2,rr,e2,T2,rgb2
              ^^^^^^^^^^^^^^^^^^
ValueError: not enough values to unpack (expected 6, got 5)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/sc,roll,e,Tc,M,rgb=sc2,rr,e2,T2,rgb2/sc,roll,e,Tc,M,rgb=sc2,rr,e2,T2,M2,rgb2/' /work/scratch/roll_anchor_select.py && python /work/scratch/roll_anchor_select.py"
```
<details><summary>output</summary>

```
690 best roll 50 sc 15.33 base 20.02 e (50, 0.1, -49.54, 57.17, 8.02, 12.96, 0.39, 0.17, 0.1)
700 best roll 45 sc 29.19 base 33.26 e (45, 0.03, -53.94, 19.18, 28.04, 1.16, 0.37, 0.27, 0.17)
<string>:9: RuntimeWarning: Mean of empty slice.
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
Traceback (most recent call last):
  File "/work/scratch/roll_anchor_select.py", line 26, in <module>
    sc,e,Tc,M,rgb=eval_c(i,roll,im);cands.append((sc,roll,e,Tc,M,rgb))
                  ^^^^^^^^^^^^^^^^^
  File "/work/scratch/roll_anchor_select.py", line 11, in eval_c
    Tcam=(Twc@rollworld(T[i].astype(float),roll)).astype(float);_,M0=render((Wc@Tcam).astype(np.float32));mc=M0[y0:y1,x0:x1];cm=feat(mc)[0];am= …[+1464 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/roll_anchor_select.py';s=open(p).read();s=s.replace(\"Tcam=(Twc@rollworld(T[i].astype(float),roll)).astype(float);_,M0=render((Wc@Tcam).astype(np.float32));mc=M0[y0:y1,x0:x1];cm=feat(mc)[0]\", \"Tcam=(Twc@rollworld(T[i].astype(float),roll)).astype(float);_,M0=render((Wc@Tcam).astype(np.float32));mc=M0[y0:y1,x0:x1]\\n if not mc.any():return 999,(roll,0,0,0,0,0,0,0,0),Tcam,M0,None\\n cm=feat(mc)[0]\")
open(p,'w').write(s)
PY
python /work/scratch/roll_anchor_select.py"
```
<details><summary>output</summary>

```
690 best roll 50 sc 15.33 base 20.02 e (50, 0.1, -49.54, 57.17, 8.02, 12.96, 0.39, 0.17, 0.1)
700 best roll 45 sc 29.19 base 33.26 e (45, 0.03, -53.94, 19.18, 28.04, 1.16, 0.37, 0.27, 0.17)
<string>:9: RuntimeWarning: Mean of empty slice.
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide
  ret = um.true_divide(
Traceback (most recent call last):
  File "/work/scratch/roll_anchor_select.py", line 28, in <module>
    sc,e,Tc,M,rgb=eval_c(i,roll,im);cands.append((sc,roll,e,Tc,M,rgb))
                  ^^^^^^^^^^^^^^^^^
  File "/work/scratch/roll_anchor_select.py", line 13, in eval_c
    cm=feat(mc)[0];am=feat(mc)[1];dt=(ao-am+np.pi/2)%np.pi-np.pi/2;Tc=cam_adjust(Tcam,dt);_,M1=render((Wc@Tc).astype(np.float32));c1=feat(M1[y0 …[+1222 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/roll_anchor_select.py';s=open(p).read()
s=s.replace(\"co=feat(oc)[0];ao=feat(oc)[1]\",\"co=feat(Oall)[0];ao=feat(Oall)[1]\")
s=s.replace(\"mc=M0[y0:y1,x0:x1]\\n if not mc.any():return 999,(roll,0,0,0,0,0,0,0,0),Tcam,M0,None\\n cm=feat(mc)[0];am=feat(mc)[1]\",\"cm=feat(M0)[0];am=feat(M0)[1]\")
s=s.replace(\"c1=feat(M1[y0:y1,x0:x1])[0]\",\"c1=feat(M1)[0]\")
open(p,'w').write(s)
PY
python /work/scratch/roll_anchor_select.py"
```
<details><summary>output</summary>

```
690 best roll 50 sc 15.33 base 20.02 e (50, 0.1, -49.54, 57.17, 8.02, 12.96, 0.39, 0.17, 0.1)
700 best roll 45 sc 29.19 base 33.26 e (45, 0.03, -53.94, 19.18, 28.04, 1.16, 0.37, 0.27, 0.17)
710 best roll -90 sc 6.33 base 23.13 e (-90, -1.4, -120.07, -42.65, 3.92, 5.02, 0.11, 0.23, 0.32)
720 best roll -150 sc 6.12 base 15.59 e (-150, -0.08, -33.61, -35.97, 2.2, 6.76, 0.42, 0.36, 0.0)
730 best roll -160 sc 5.29 base 6.42 e (-160, 0.35, 28.3, 32.18, 3.99, 2.16, 0.31, 0.36, 0.27)
740 best roll -10 sc 10.19 base 10.78 e (-10, 0.69, 133.38, 68.38, 9.81, 1.59, 0.09, 0.34, 0.22)
750 best roll -90 sc 7.39 base 25.85 e (-90, -0.53, -64.49, -39.65, 5.28, 3.99, 0.16, 0.19, -0.02)
760 best roll -185 sc 11.69 base 10.48 e (-185, 0.37, 84.68, 120.97, 5.78, 11.55, 0.17, 0.19, -0.01)
770 best roll -190 sc  …[+753 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/view_anchor_selected.py <<'PY'
exec(open('/work/scratch/mask_refine_dyn.py').read().split('idxs=')[0])
import h5py,cv2
z=np.load(f'{work}/scratch/roll_anchor.npz');idxs=z['idx'];Tcs=z['Tc'];roll=z['roll']
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(988):ok,im=cap.read();frames.append(im)
thumbs=[]
for k,i in enumerate(idxs):
 im=frames[i].copy();rgb,M=render(Tcs[k].astype(np.float32));O=OM[i];cnt,_=cv2.findContours(M.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(O.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,255,0),2);u=M|O;y,x=np.where(u);x0=max(0,x.min()-30);x1=min(1536,x.max()+30);y0=max(0,y.min()-30);y1=min(1152,y.max()+30);vis=cv2.resize(im[y0:y1,x0:x1],(300,360));cv2.putText(vis,f'{i} r{roll[k]}',(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);thumbs.append(vis)
rows=[]
for j in range(0,len(thumbs),5):
 row=thumbs[j:j+5]
 while len(row)<5:row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite(f'{work}/scratch/anchor_selected.jpg',np.vstack(rows))
PY
python /work/scratch/view_anchor_selected.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/roll_anchor_select.py';s=open(p).read();s=s.replace(\"Tc=np.stack([z[4] for z in out])\",\"Tc=np.stack([Wc@z[4] for z in out])\")
open(p,'w').write(s)
PY
python /work/scratch/roll_anchor_select.py >/work/scratch/roll_anchor_log.txt && python /work/scratch/view_anchor_selected.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_anchor_traj.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float)
z=np.load('/work/scratch/roll_anchor.npz');idxs=list(map(int,z['idx']));Ta=z['Tc'].astype(float)
za=np.load('/work/scratch/static_refined.npz');Te=za['Te'].astype(float)
# keyframes, drop 760 and 780 (interpolate across), end at Te
keys={i:Ta[k] for k,i in enumerate(idxs) if i not in [760,780,840]}
keys[840]=Te
ki=sorted(keys)
T=B.copy()
def slerp(A,B,u):
 qa=Rotation.from_matrix(A[:3,:3]).as_quat();qb=Rotation.from_matrix(B[:3,:3]).as_quat();
 if qa@qb<0:qb=-qb
 q=qa*(1-u)+qb*u;q/=np.linalg.norm(q);T=np.eye(4);T[:3,:3]=Rotation.from_quat(q).as_matrix();T[:3,3]=A[:3,3]*(1-u)+B[:3,3]*u;return T
for a,b in zip(ki[:-1],ki[1:]):
 for i in range(a,b+1):
  u=(i-a)/(b-a);w=u*u*(3-2*u);T[i]=slerp(keys[a],keys[b],w)
np.savez('/work/scratch/anchor_traj.npz',T=T)
# metrics
import sys
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
def ev(i,X):
 _,M=render(X[i].astype(np.float32));VM=M&~PM[i];O=OM[i]&~PM[i];Do=distance_transform_edt(~O);Dm=distance_transform_edt(~VM);a=max(1,O.sum());return Do[VM].mean()+Dm[O].mean()+5*abs(VM.sum()-a)/a,(VM&O).sum()/max(1,(VM|O).sum())
for a,b in [(0,111),(111,690),(690,840) …[+697 chars]
```
<details><summary>output</summary>

```
0 111 base cham/iou 5.6660130920894 0.31056656621720735 new 5.6660130920894 0.31056656621720735
111 690 base cham/iou 12.481809885793291 0.310918567144505 new 12.481809885793291 0.310918567144505
690 840 base cham/iou 45.61443385606923 0.16660126462544336 new 45.61443385606923 0.16660126462544336
840 988 base cham/iou 11.456692463552786 0.2520519998191029 new 11.456692463552786 0.2520519998191029
base rot max/p99/med 119.37290555661362 40.43425974252719 1.8081408424122474 trans max/p99/med 0.2832571931368536 0.17961440337246168 0.020372861783935436
top [(744, 119.4), (732, 109.8), (727, 109.7), (723, 105.9), (734, 82.1), (708, 53.0), (746, 52.6), (709, 51.8), (778, 42.7), (702, 41.0)]
new rot max/p99/med 119.3729 40.434253692626946 1.8079687 trans max/p99/med 0.2832572 0.17961439430713635  …[+151 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_anchor_traj.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
B=T.copy();N=np.load('/work/scratch/anchor_traj.npz')['T'].astype(np.float32)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
def ev(i,X):
 _,M=render(X[i].astype(np.float32));VM=M&~PM[i];O=OM[i]&~PM[i];Do=distance_transform_edt(~O);Dm=distance_transform_edt(~VM);a=max(1,O.sum());return Do[VM].mean()+Dm[O].mean()+5*abs(VM.sum()-a)/a,(VM&O).sum()/max(1,(VM|O).sum())
for a,b in [(690,840)]:
 for name,X in [('base',B),('new',N)]:
  vals=[];ious=[]
  for i in range(a,b,3):s,j=ev(i,X);vals.append(s);ious.append(j)
  print(name,np.mean(vals),np.mean(ious),np.percentile(vals,[50,90,100]))
for name,X in [('base',B),('new',N)]:
 dtt=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);drr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',X[:-1,:3,:3],X[1:,:3,:3])-1)/2,-1,1)))
 print(name,'rot',drr.max(),np.percentile(drr,99),np.median(drr),'trans',dtt.max(),np.percentile(dtt,99),np.median(dtt));print([(i,round(v,1)) for i,v in sorted(enumerate(drr),key=lambda q:-q[1])[:10]])
PY
python /work/scratch/eval_anchor_traj.py"
```
<details><summary>output</summary>

```
base 45.61443385606923 0.16660126462544336 [ 31.69273402  92.95943953 215.54183393]
new 18.678529752593555 0.28447769032291814 [  9.62239677  41.88402057 126.29945054]
base rot 119.3729 40.434253692626946 1.8079687 trans 0.2832572 0.17961439430713635 0.020372862
[(744, 119.4), (732, 109.8), (727, 109.7), (723, 105.9), (734, 82.1), (708, 53.0), (746, 52.6), (709, 51.8), (778, 42.7), (702, 41.0)]
new rot 44.898453 19.539180221557555 1.7925321 trans 0.26934826 0.09968922898173332 0.018390045
[(689, 44.9), (834, 32.0), (835, 32.0), (533, 29.1), (704, 28.7), (705, 28.7), (833, 25.6), (836, 25.6), (703, 23.6), (706, 23.6)]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/build_final3.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation
r=np.load('/work/inputs/reconstruction.npz');B=r['T_world_mesh'].astype(float);POS=B[:,:3,3].copy();q=Rotation.from_matrix(B[:,:3,:3]).as_quat()
zs=np.load('/work/scratch/static_refined.npz');Ts=zs['Ts'].astype(float);Te=zs['Te'].astype(float)
za=np.load('/work/scratch/roll_anchor.npz');idxs=list(map(int,za['idx']));Ta=za['Tc'].astype(float);amap={i:Ta[k] for k,i in enumerate(idxs)}
def smooth_rot(w,thr):
 qo=np.empty_like(q);h=w//2
 for i in range(len(q)):
  a=max(0,i-h);b=min(len(q),i+h+1);qq=q[a:b].copy();ref=qq[np.argmax(np.abs(qq@qq.T).sum(1))];ss=np.sign(qq@ref);ss[ss==0]=1;qq*=ss[:,None];ref=ref*np.sign(ref@q[i]);dots=np.clip(qq@ref,-1,1);ang=2*np.arccos(dots);ww=np.ones(len(qq));bad=ang>np.radians(thr);ww[bad]=np.radians(thr)/ang[bad];m=(qq*ww[:,None]).sum(0);qo[i]=m/np.linalg.norm(m)
 return Rotation.from_quat(qo).as_matrix()
def smooth_trans(w):return np.array([np.median(POS[max(0,i-w//2):min(len(POS),i+w//2+1)],0) for i in range(len(POS))])
Rsm=smooth_rot(13,8);Psm=smooth_trans(13);T=B.copy();T[:,:3,:3]=Rsm;T[:,:3,3]=Psm
# static phases
T[:111]=Ts;T[840:]=Te
# interpolate selected manipulated-phase anchors, including a smoothed pre-anchor at 670
keys={670:T[670].copy(),690:amap[690],700:amap[700],710:amap[710],720:amap[720],730:amap[730],740:amap[740],750:amap[750],770:amap[770],790:amap[790],840:Te}
def slerp(A,B,u):
 qa=Rotation.from_matrix …[+1143 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/build_final3.py", line 32, in <module>
    dtt=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);drr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',X[:-1,:3,:3],X[1:,:3,:3])-1)/2,-1,1)))
                               ~^^^^^^^^
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/for name,X in \\[(\\x27base\\x27,B),(\\x27new\\x27,T)\\]:/for name,XX in [(\\x27base\\x27,B),(\\x27new\\x27,T)]:/; s/X\\[:,:3,3\\]/XX[:,:3,3]/g; s/X\\[:-1,:3,:3\\]/XX[:-1,:3,:3]/g; s/X\\[1:,:3,:3\\]/XX[1:,:3,:3]/g' /work/scratch/build_final3.py && python /work/scratch/build_final3.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/build_final3.py", line 32, in <module>
    dtt=np.linalg.norm(np.diff(XX[:,:3,3],axis=0),axis=1);drr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',XX[:-1,:3,:3],XX[1:,:3,:3])-1)/2,-1,1)))
                               ~~^^^^^^^^
IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nl -ba /work/scratch/build_final3.py | tail -10'
```
<details><summary>output</summary>

```
    24	  u=(i-a)/(b-a);w=u*u*(3-2*u);T[i]=slerp(keys[a],keys[b],w)
    25	# blend Ts into smooth pose over 110..130
    26	A=Ts;B=T[130]
    27	qa=Rotation.from_matrix(A[:3,:3]).as_quat();qb=Rotation.from_matrix(B[:3,:3]).as_quat();qb*=np.sign(qa@qb)
    28	for i in range(111,130):
    29	 u=(i-110)/20;w=u*u*(3-2*u);qq=qa*(1-w)+qb*w;qq/=np.linalg.norm(qq);T[i,:3,:3]=Rotation.from_quat(qq).as_matrix();T[i,:3,3]=A[:3,3]*(1-w)+B[:3,3]*w
    30	np.savez('/work/scratch/final3.npz',T=T)
    31	for name,XX in [('base',B),('new',T)]:
    32	 dtt=np.linalg.norm(np.diff(XX[:,:3,3],axis=0),axis=1);drr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',XX[:-1,:3,:3],XX[1:,:3,:3])-1)/2,-1,1)))
    33	 print(name,'rot pct',np.percentile(drr,[50,90,95,99,100]),'trans',np.percentile(dtt,[50,90,95,99,100 …[+86 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/A=Ts;B=T\\[130\\]/A=Ts;Bend=T[130]/; s/qb=Rotation.from_matrix(B\\[:3,:3\\])/qb=Rotation.from_matrix(Bend[:3,:3])/; s/B\\[:3,3\\]/Bend[:3,3]/g' /work/scratch/build_final3.py && python /work/scratch/build_final3.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/build_final3.py", line 24, in <module>
    u=(i-a)/(b-a);w=u*u*(3-2*u);T[i]=slerp(keys[a],keys[b],w)
                                     ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/build_final3.py", line 18, in slerp
    qa=Rotation.from_matrix(A[:3,:3]).as_quat();qb=Rotation.from_matrix(Bend[:3,:3]).as_quat();
                                                                        ^^^^
NameError: name 'Bend' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_final3.py';s=open(p).read();s=s.replace(\"def slerp(A,B,u):\\n qa=Rotation.from_matrix(A[:3,:3]).as_quat();qb=Rotation.from_matrix(Bend[:3,:3]).as_quat();\\n if qa@qb<0:qb=-qb\\n q=qa*(1-u)+qb*u;q/=np.linalg.norm(q);O=np.eye(4);O[:3,:3]=Rotation.from_quat(q).as_matrix();O[:3,3]=A[:3,3]*(1-u)+Bend[:3,3]*u;return O\", \"def slerp(X1,X2,u):\\n qa=Rotation.from_matrix(X1[:3,:3]).as_quat();qb=Rotation.from_matrix(X2[:3,:3]).as_quat();\\n if qa@qb<0:qb=-qb\\n q=qa*(1-u)+qb*u;q/=np.linalg.norm(q);O=np.eye(4);O[:3,:3]=Rotation.from_quat(q).as_matrix();O[:3,3]=X1[:3,3]*(1-u)+X2[:3,3]*u;return O\")
open(p,'w').write(s)
PY
python /work/scratch/build_final3.py"
```
<details><summary>output</summary>

```
base rot pct [  1.80814084   6.39279665  11.68021969  40.43425974 119.37290556] trans [0.02037286 0.05805859 0.09069126 0.1796144  0.28325719]
[(744, 119.4), (732, 109.8), (727, 109.7), (723, 105.9), (734, 82.1), (708, 53.0), (746, 52.6), (709, 51.8), (778, 42.7), (702, 41.0), (730, 40.3), (755, 37.8)]
new rot pct [ 0.53663696  2.96815872  4.24339454  9.72216635 28.66271032] trans [0.0068366  0.02496526 0.03047293 0.05456769 0.08447959]
[(705, 28.7), (704, 28.7), (703, 23.6), (706, 23.6), (707, 16.5), (702, 16.5), (133, 13.1), (137, 10.9), (715, 10.3), (714, 10.3), (708, 9.6), (701, 9.6)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_anchor_traj.py';s=open(p).read().replace(\"anchor_traj.npz\",\"final3.npz\");s=s.replace(\"for a,b in [(690,840)]:\",\"for a,b in [(0,111),(111,670),(670,840),(840,988),(0,988)]:\");open('/work/scratch/eval_final3.py','w').write(s)
PY
python /work/scratch/eval_final3.py"
```
<details><summary>output</summary>

```
base 5.6660130920894 0.31056656621720735 [6.18610498 6.79261515 8.88602843]
new 3.146666969929292 0.61705210663678 [3.16253099 3.33120378 3.41607531]
base 12.619445656793735 0.31001411733543843 [  7.07030948  31.12940383 106.54775599]
new 11.08999647553717 0.3260616507360608 [ 7.18403445 24.63974415 68.7782402 ]
/work/scratch/eval_final3.py:7: RuntimeWarning: Mean of empty slice.
  _,M=render(X[i].astype(np.float32));VM=M&~PM[i];O=OM[i]&~PM[i];Do=distance_transform_edt(~O);Dm=distance_transform_edt(~VM);a=max(1,O.sum());return Do[VM].mean()+Dm[O].mean()+5*abs(VM.sum()-a)/a,(VM&O).sum()/max(1,(VM|O).sum())
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dtype.type(ret / rcount)
base nan 0.203941308029 …[+884 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File('/work/inputs/masks.h5','r') as h:OM=h['object'];PM=h['person']
x0,x1,y0,y1=680,930,560,1080
o=OM[850:988].mean(0)[y0:y1,x0:x1]>.18;p=PM[850:988].mean(0)[y0:y1,x0:x1]<.35;O=o&p;D=distance_transform_edt(~O)
rows=[]
for j in range(850,988,2):
 _,M=render(T[j]);M=M[y0:y1,x0:x1];Dm=distance_transform_edt(~M);a=O.sum();s=D[M].mean()+Dm[O].mean()+5*abs(M.sum()-a)/a-2*(M&O).sum()/max(1,(M|O).sum());rows.append((s,j))
print(sorted(rows)[:20])
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 870, in __getitem__
    if self._fast_read_ok and (new_dtype is None):
       ^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/base.py", line 534, in __get__
    value = obj.__dict__[self.func.__name__] = self.func(obj)
                                               ^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/dataset.py", line 849, in _fast_read_ok
    self._extent_type == h5s.SIMPLE
    ^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packa …[+794 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File('/work/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
x0,x1,y0,y1=680,930,560,1080
o=OM[850:988].mean(0)[y0:y1,x0:x1]>.18;p=PM[850:988].mean(0)[y0:y1,x0:x1]<.35;O=o&p;D=distance_transform_edt(~O)
rows=[]
for j in range(850,988,2):
 _,M=render(T[j]);M=M[y0:y1,x0:x1];Dm=distance_transform_edt(~M);a=O.sum();s=D[M].mean()+Dm[O].mean()+5*abs(M.sum()-a)/a-2*(M&O).sum()/max(1,(M|O).sum());rows.append((s,j))
print(sorted(rows)[:20])
PY"
```
<details><summary>output</summary>

```
[(5.073630563052127, 898), (6.076017132782416, 864), (6.420936008151379, 874), (6.463157892966699, 896), (6.668707271763918, 932), (6.670024129304761, 928), (7.003584948597699, 880), (7.130329306979223, 868), (7.317128329392248, 938), (7.403194326806683, 894), (7.457091875737541, 924), (7.542121901176145, 876), (7.617374378369705, 886), (7.779691730577932, 892), (7.88849213668035, 922), (7.965356334148579, 900), (8.190381598696074, 936), (8.19539571265064, 902), (8.22536019830489, 870), (8.326892346815841, 940)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
with h5py.File('/work/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
x0,x1,y0,y1=380,850,580,960;o=OM[:111].mean(0)[y0:y1,x0:x1]>.18;p=PM[:111].mean(0)[y0:y1,x0:x1]<.35;O=o&p;D=distance_transform_edt(~O)
rows=[]
for j in range(0,111,2):
 _,M=render(T[j]);M=M[y0:y1,x0:x1];Dm=distance_transform_edt(~M);a=O.sum();s=D[M].mean()+Dm[O].mean()+5*abs(M.sum()-a)/a-2*(M&O).sum()/max(1,(M|O).sum());rows.append((s,j))
print(sorted(rows)[:20])
PY"
```
<details><summary>output</summary>

```
[(2.379278278972852, 70), (2.491943627192815, 68), (2.531669665432166, 66), (2.7368200596307286, 76), (2.7571780276940157, 80), (2.8613896694453738, 74), (2.87242668133785, 72), (2.9425918345848543, 96), (3.1908738533125742, 98), (3.559132928788793, 82), (3.595518533658063, 78), (3.628705938710201, 108), (3.653192824840926, 100), (3.78444327195342, 62), (3.7880426880797122, 104), (3.837921541360769, 88), (4.088602813941166, 52), (4.313628102633719, 64), (4.415476879109861, 60), (4.551089194052007, 58)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh']
for j in [60,66,68,70,72,74,76,80]:
 d=np.degrees(np.arccos(np.clip((np.trace(T[60][:3,:3].T@T[j][:3,:3])-1)/2,-1,1)));print(j,d,np.linalg.norm(T[60][:3,3]-T[j][:3,3]))
PY"
```
<details><summary>output</summary>

```
60 0.0 0.0
66 1.7175999187027025 0.013186567
68 2.2323046611405584 0.020466104
70 3.760152365302081 0.017741797
72 2.354679832357207 0.028086128
74 2.2868740384099344 0.03837121
76 2.7076154459048754 0.047157846
80 1.9339104886528287 0.050275706

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Deterministic trajectory reconstruction for the squeegee episode.

The released reconstruction is used as a frame-wise proposal.  The method then
uses three forms of temporal evidence that the proposal does not enforce:
  * one constant pose during the two visually stationary phases;
  * robust medoid smoothing to remove isolated tracker inversion jumps;
  * mask-aligned roll hypotheses on unambiguous keyframes around the difficult
    carry/return segment, interpolated in SE(3).

Only the supplied inputs are used.  The renderer is silhouette/UV only; it is
used to score hypotheses, not to train anything.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

import nvdiffrast.torch as dr


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pca_features(mask: np.ndarray):
    ys, xs = np.where(mask)
    pts = np.column_stack((xs, ys)).astype(np.float64)
    center = pts.mean(0)
    q = pts - center
    cov = q.T @ q / max(1, len(q))
    _, vecs = np.linalg.eigh(cov)
    axis = vec …[+16972 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cd /work && python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
static anchors: start frame 70 (score 2.380), end frame 898 (score 5.076)
anchor 690: roll=-120 deg, mask score=14.020
anchor 700: roll=  60 deg, mask score=30.922
anchor 710: roll=  90 deg, mask score=6.151
anchor 720: roll=  30 deg, mask score=5.803
anchor 730: roll=   0 deg, mask score=6.378
anchor 740: roll=-180 deg, mask score=9.099
anchor 750: roll=  90 deg, mask score=5.413
anchor 770: roll=   0 deg, mask score=14.539
anchor 790: roll=   0 deg, mask score=2.696
wrote /work/outputs/object_poses_world.npz in 36.9s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_roll_flip.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py,cv2
S=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
C=np.eye(4);C[:3,3]=mesh.vertices.mean(0);Ci=np.eye(4);Ci[:3,3]=-mesh.vertices.mean(0);R=np.eye(4);R[:3,:3]=-np.eye(3);R[:3,:3]+=2*np.outer([0,0,1],[0,0,1]);FLIP=C@R@Ci
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(988):ok,im=cap.read();frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
def orange_obs(i):
 hsv=cv2.cvtColor(frames[i],cv2.COLOR_RGB2HSV);return OM[i]&(hsv[...,0]<20)&(hsv[...,0]>0)&(hsv[...,1]>120)&(hsv[...,2]>80)
def score(i,Tw):
 rgb,M=render(Tw.astype(np.float32));valid=M&OM[i]&~PM[i];a=rgb[valid];b=frames[i][valid].astype(float)/255;aa=a-a.mean(0);bb=b-b.mean(0);ncc=((aa*bb).sum(0)/(np.sqrt((aa*aa).sum(0)*(bb*bb).sum(0))+1e-9)).mean()
 # fit per-channel affine and rmse
 rmse=[]
 for ch in range(3):
  X=np.stack([a[:,ch],np.ones(len(a))],1);coef=np.linalg.lstsq(X,b[:,ch],rcond=None)[0];rmse.append(np.sqrt(np.mean((X@coef-b[:,ch])**2)))
 ro=M&(rgb[...,0]>rgb[...,1]*1.35)&(rgb[...,0]>rgb[...,2]*1.25)&(rgb[...,0]>.10);oo=orange_obs(i);oi=(ro&oo).sum()/max(1,(ro|oo).sum())
 if ro.any() and oo.any():
  y,x=np.where(ro);yc,xc=y.mean(),x.mean();y,x=np.where(oo);cd=np.hypot(xc-x.mean(),yc-y.mean());cnt=abs(ro.sum()-oo.sum() …[+1125 chars]
```
<details><summary>output</summary>

```
690 cur ncc 0.389 rmse 0.103 orange iou 0.112 cd 8.9 cnt 0.89 areas 214 113
690 alt ncc 0.137 rmse 0.114 orange iou 0.022 cd 12.8 cnt 0.47 areas 166 113
700 cur ncc 0.068 rmse 0.108 orange iou 0.0 cd 121.7 cnt 1.36 areas 326 138
700 alt ncc 0.051 rmse 0.086 orange iou 0.0 cd 136.1 cnt 2.07 areas 423 138
710 cur ncc -0.048 rmse 0.161 orange iou 0.0 cd 21.9 cnt 0.19 areas 227 191
710 alt ncc 0.465 rmse 0.149 orange iou 0.0 cd 21.8 cnt 0.4 areas 267 191
720 cur ncc -0.357 rmse 0.114 orange iou 0.0 cd 56.9 cnt 0.25 areas 187 250
720 alt ncc -0.019 rmse 0.111 orange iou 0.0 cd 59.7 cnt 0.14 areas 216 250
730 cur ncc 0.271 rmse 0.094 orange iou 0.0 cd 38.2 cnt 2.18 areas 261 82
730 alt ncc 0.158 rmse 0.098 orange iou 0.061 cd 26.6 cnt 1.4 areas 197 82
740 cur ncc 0.217 rmse 0.061 orange iou 0.0  …[+555 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
S=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
def dr(a,b):return np.degrees(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
print('anchors');print([(i,round(dr(S[i-1],S[i]),1)) for i in [690,700,710,720,730,740,750,770,790,840]])
for i in [680,689,690,699,700,709,710,719,720,729,730,739,740,749,750,759,760,769,770,779,780,789,790,799,800,819,820,839,840]:
 if i%10==0 or i in [689,699,709,719,729,739,749,759,769,779,789,799]:print(i,np.round(S[i,:3,3],2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_overlay.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py,cv2
S=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
cap=cv2.VideoCapture(f'{work}/inputs/video.mp4');frames=[]
for i in range(988):ok,im=cap.read();frames.append(im)
idxs=[0,50,100,110,130,160,200,300,400,500,600,660,690,700,710,720,730,740,750,760,770,780,790,800,820,840,880,920,960,987];thumbs=[]
for i in idxs:
 im=frames[i].copy();rgb,M=render(S[i].astype(np.float32));O=OM[i];cnt,_=cv2.findContours(M.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(0,255,0),2);cnt,_=cv2.findContours(O.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cnt,-1,(255,255,0),2);u=M|O;y,x=np.where(u);x0=max(0,x.min()-25);x1=min(1536,x.max()+25);y0=max(0,y.min()-25);y1=min(1152,y.max()+25);vis=cv2.resize(im[y0:y1,x0:x1],(300,360));cv2.putText(vis,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);thumbs.append(vis)
rows=[]
for j in range(0,len(thumbs),5):
 row=thumbs[j:j+5]
 while len(row)<5:row.append(np.zeros_like(thumbs[0]))
 rows.append(np.hstack(row))
cv2.imwrite(f'{work}/scratch/final_overlay.jpg',np.vstack(rows))
PY
python /work/scratch/final_overlay.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_validation.py <<'PY'
exec(open('/work/scratch/static_scores.py').read().split('# load frames')[0])
import h5py
from scipy.ndimage import distance_transform_edt
B=T.astype(np.float64)
N=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(np.float64)
with h5py.File(f'{work}/inputs/masks.h5','r') as h:OM=h['object'][:];PM=h['person'][:]
def ev(i,X):
 _,M=render(X[i].astype(np.float32));V=M&~PM[i];O=OM[i]&~PM[i]
 if not V.any():return np.nan,np.nan
 D1=distance_transform_edt(~O);D2=distance_transform_edt(~V);a=max(1,O.sum())
 return D1[V].mean()+D2[O].mean()+5*abs(V.sum()-a)/a,(V&O).sum()/max(1,(V|O).sum())
phases=[('rest_start',0,111),('pickup_sweep_1',111,400),('sweep_2',400,670),('carry_place',670,840),('rest_end',840,988),('overall',0,988)]
print('phase,base_chamfer,new_chamfer,base_iou,new_iou,improved_frac')
for name,a,b in phases:
 bs=[];ns=[];bi=[];ni=[];imp=[]
 for i in range(a,b,5):
  x,y=ev(i,B);u,v=ev(i,N)
  if np.isfinite(x) and np.isfinite(u):bs.append(x);ns.append(u);imp.append(u<x)
  if np.isfinite(y) and np.isfinite(v):bi.append(y);ni.append(v)
 print(f'{name},{np.mean(bs):.4f},{np.mean(ns):.4f},{np.mean(bi):.4f},{np.mean(ni):.4f},{np.mean(imp):.4f}')
def vel(X):
 dtt=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);dr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',X[:-1,:3,:3],X[1:,:3,:3])-1)/2,-1,1)));return dr,dtt
bd,bt=vel(B);nd,nt=vel(N)
for name,d,t in [('base',bd,bt),('new',nd,nt)]:
 prin …[+542 chars]
```
<details><summary>output</summary>

```
phase,base_chamfer,new_chamfer,base_iou,new_iou,improved_frac
rest_start,5.8561,3.2439,0.2988,0.6099,1.0000
pickup_sweep_1,11.7540,11.3607,0.3228,0.3331,0.5690
sweep_2,13.3597,10.8786,0.3299,0.3252,0.5370
carry_place,43.7482,17.8720,0.1968,0.2491,0.6471
rest_end,11.9306,6.3671,0.2304,0.4277,0.8667
overall,16.7658,10.6954,0.2887,0.3604,0.6616
base rot p50/p90/p99/max [  1.80814084   6.39279665  40.43425974 119.37290556] trans p50/p90/p99/max [0.02037286 0.05805859 0.1796144  0.28325719]
new rot p50/p90/p99/max [ 0.4783936   2.57243744 13.06353248 30.46115449] trans p50/p90/p99/max [0.00673968 0.02469687 0.05316376 0.08447959]
spread start base trans median/max 0.023350727683827184 0.0949697611651703 rot median/max 24.580193722911183 39.08714596840051
spread start new trans median/max 4.1377 …[+323 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ls -lh /work/outputs/object_poses_world.npz /work/outputs/solve.py; python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
print(p.shape,p.dtype,np.isfinite(p).all(),np.max(np.abs(p[:,3,:]-[0,0,0,1])))
R=p[:,:3,:3];print('orth err',np.abs(np.einsum('fji,fjk->fik',R,R)-np.eye(3)).max(),'det range',np.linalg.det(R).min(),np.linalg.det(R).max())
PY"
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 124K Oct 20 18:51 /work/outputs/object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  18K Oct 20 18:50 /work/outputs/solve.py
(988, 4, 4) float64 True 0.0
orth err 8.881784197001252e-16 det range 0.9999999999999988 1.000000000000001

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffmpeg -version | head -1; python --version; uname -srm'
```
<details><summary>output</summary>

```
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
Python 3.11.10
Linux 6.8.0-138-generic x86_64

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy, scipy, cv2, h5py, torch, trimesh, nvdiffrast
for m in [numpy,scipy,cv2,h5py,torch,trimesh,nvdiffrast]:
 print(m.__name__,getattr(m,'__version__','unknown'),getattr(m,'__file__',''))
print('cuda',torch.version.cuda,'device',torch.cuda.get_device_name(0))
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3 /opt/conda/lib/python3.11/site-packages/numpy/__init__.py
scipy 1.16.3 /opt/conda/lib/python3.11/site-packages/scipy/__init__.py
cv2 4.11.0 /opt/conda/lib/python3.11/site-packages/cv2/__init__.py
h5py 3.16.0 /opt/conda/lib/python3.11/site-packages/h5py/__init__.py
torch 2.5.1+cu124 /opt/conda/lib/python3.11/site-packages/torch/__init__.py
trimesh 5.1.1 /opt/conda/lib/python3.11/site-packages/trimesh/__init__.py
nvdiffrast 0.4.0 /opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py
cuda 12.4 device NVIDIA RTX 6000 Ada Generation

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);N=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
dr=np.degrees(np.arccos(np.clip((np.einsum('fij,fij->f',B[:,:3,:3],N[:,:3,:3])-1)/2,-1,1)))
dt=np.linalg.norm(B[:,:3,3]-N[:,:3,3],axis=1)
print('rot diff pct',np.percentile(dr,[50,75,90,95,99,100]),'mean',dr.mean())
print('trans diff pct',np.percentile(dt,[50,75,90,95,99,100]),'mean',dt.mean())
for a,b in [(0,111),(111,400),(400,670),(670,840),(840,988)]:print(a,b,'rot med/mean',np.median(dr[a:b]),dr[a:b].mean(),'trans',np.median(dt[a:b]),dt[a:b].mean())
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Squeegee 6-DoF Trajectory Report

## Result

- Output: "'`object_poses_world.npz`
- Key: `T_world_mesh`, shape `(988, 4, 4)`, `float64`
- Reproduce with:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The released per-frame reconstruction was used as the proposal source. The final trajectory keeps its pose convention and world frame, but replaces/regularizes parts that are inconsistent with rigid-object temporal evidence.

## Phases and changes

| Frames | Phase | What changed | Confidence |
|---|---|---|---|
| 0-110 | At rest, leaning against the stool | Replaced all per-frame estimates with one pooled static pose. The start orientation was selected from the stable proposal cluster using pooled person-excluded object masks; translation was refined against the pooled silhouette. | High |
| 111-399 | Pickup and first sweeping motion | Applied 13-frame robust medoid smoothing in SO(3) and translation median filtering, with a smooth transition from the start static pose. | Medium-high |
| 400-669 | Continued sweeping/manipulation | Same robust temporal smoothing. Image evidence was retained; no static constraint was applied. | Medium |
| 670-839 | Turn, lower, and place squeegee back against stool | Added mask-aligned roll hypotheses on keyframes (690-790), interpolated selected poses in SE(3), and skipped heavily occluded frames as direct anchors. This removes  …[+6021 chars]
```
<details><summary>output</summary>

```
total 160K
-rw-rw-r-- 1 hanzhou hanzhou 5.8K Oct 20 18:55 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.5K Oct 20 18:55 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 18:50 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 124K Oct 20 18:51 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  18K Oct 20 18:50 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -nE '/work/scratch|/pipeline|/weights|http|random' /work/outputs/solve.py"
```
<details><summary>output</summary>

```
18:import random
34:    random.seed(seed)
35:    np.random.seed(seed)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz')
print(p.files)
a=p['T_world_mesh']
print(a.shape,a.dtype,np.isfinite(a).all())
print('frame0\\n',a[0]);print('frame987\\n',a[-1])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "du -sh /work/outputs; find /work/outputs -maxdepth 1 -type f -printf '%f %s\\n' | sort"
```
<details><summary>output</summary>

```
200K	/work/outputs
REPORT.md 5923
TOOLS.md 1470
object_poses_world.npz 126742
solve.py 18271

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import shutil, os
p='/work/outputs/__pycache__'
if os.path.isdir(p): shutil.rmtree(p)
print(sorted(os.listdir('/work/outputs')))
PY"
```
<details><summary>output</summary>

```
['REPORT.md', 'TOOLS.md', 'object_poses_world.npz', 'solve.py']

```
</details>

**Agent:** Completed the task. Required outputs are in `/work/outputs`:

- `object_poses_world.npz` — `T_world_mesh`, shape `(988, 4, 4)`, `float64`
- `solve.py` — deterministic offline reconstruction script
- `REPORT.md` — phases, method, validation table, confidence, runtime
- `TOOLS.md` — tool and dependency details

Validation summary: mean mask chamfer improved from `16.77` to `10.70`, static-phase pose spread was eliminated, and maximum per-frame rotation jump dropped from `119.4°` to `30.5°`. Final reproducible runtime was `36.9 s`.
