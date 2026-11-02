# T1_ds_ep014

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T12:03:24Z

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
and translation both matter; a  …[+9686 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,220p' /work/AGENT_TASK.md"
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
and translation both matter; a  …[+9686 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '45,105p'"
```
<details><summary>output</summary>

```
    45	   smoothness / constant-velocity terms on SE(3), and rigid hand-object attachment while carried.
    46	5. **Anchors and propagation**: find frames where the pose is unambiguous (object large, little occluded, distinctive
    47	   view, high evidence margin between hypotheses) and propagate forward and backward from them through the ambiguous
    48	   segments, rather than trusting every frame equally.
    49	
    50	## Validate without ground truth
    51	You have no ground truth and must not try to get it. Build your own checks and make decisions with them:
    52	measures of how well a trajectory explains the video (held-out frames, mask IoU, photometric / flow residuals,
    53	temporal smoothness, physical plausibility such as no floating or penetrating the floor/human), wit …[+4806 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' | sort && echo '---' && cat /work/inputs/task.txt && echo '---' && cat /work/inputs/episode.json && echo '---' && cat /work/inputs/camera.json && echo '---' && cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
camera.json 822 bytes
episode.json 66 bytes
frame0_reference_human.npz 192895 bytes
ground_plane.json 401 bytes
masks.h5 9372584 bytes
object_mesh.glb 15473916 bytes
reconstruction.npz 50642104 bytes
task.txt 221 bytes
video.mp4 26089675 bytes
---
"Walk to the round table and use both hands to lower it onto its side. Flip the table so the wheels are facing upward, then rotate it again onto its side, and continue rotating it until it returns to an upright position.
---
{
 "episode_index": 14,
 "object": "round_table",
 "frames": 993
}---
{
 "K": [
  [
   819.9995727539062,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   819.9995727539062,
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
   830.857666015625,
   0.0,
   768.0
  ],
  [
   0.0,
   830.8576049804688,
   576 …[+970 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p='/work/inputs/reconstruction.npz'
z=np.load(p)
for k in z.files:
 a=z[k]
 print(k,a.shape,a.dtype, 'min/max', (float(np.nanmin(a)),float(np.nanmax(a))) if np.issubdtype(a.dtype,np.number) and a.size else '')
PY"
```
<details><summary>output</summary>

```
T_world_mesh (993, 4, 4) float32 min/max (-0.9999772310256958, 3.8094642162323)
T_cam_mesh_foundationpose (993, 4, 4) float32 min/max (-0.9992298483848572, 3.6934826374053955)
T_cam_mesh_coconet (993, 4, 4) float32 min/max (-0.9999278783798218, 3.5584990978240967)
human_vertices_world (993, 4610, 3) float32 min/max (-0.7169375419616699, 3.875502586364746)
human_keypoints_world (993, 70, 3) float32 min/max (-0.7103738784790039, 3.8723623752593994)
contact_prob_right_left (993, 2) float32 min/max (7.356014262343072e-14, 1.0)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1912
/pipeline/.stages/02_export.json 2149
/pipeline/.stages/03_human_initialization.json 2296
/pipeline/.stages/04_depth_alignment.json 2058
/pipeline/.stages/05_object_pose.json 1899
/pipeline/.stages/06_coconet.json 3081
/pipeline/.stages/07_refinement.json 2301
/pipeline/.stages/08_render.json 2820
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 1071799060
/pipeline/depth/moge2/depth_report.json 1875
/pipeline/depth/moge2/intrinsics.pkl 105665
/pipeline/depth/moge2/raw.npy 3514171520
/pipeline/export/episode_000014/edex 562
/pipeline/export/episode_000014/wild_export.json 2764
/pipeline/foundationpose/episode_000014.pkl 282209
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 75938 …[+483 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/work/inputs/masks.h5'
with h5py.File(p,'r') as f:
 def pv(n,o):
  if hasattr(o,'shape'): print(n,o.shape,o.dtype, 'sum',o[:].sum() if o.ndim>=2 else '')
 f.visititems(pv)
PY"
```
<details><summary>output</summary>

```
object (993, 1152, 1536) bool sum 31192784
person (993, 1152, 1536) bool sum 28151198

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/outputs /work/scratch /work/tools && python - <<'PY'
import numpy as np, cv2, json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; F=len(T)
p=T[:,:3,3]; rot=R.from_matrix(T[:,:3,:3])
# rel angle consecutive
ang=np.degrees(rot[:-1].inv()*rot[1:]).magnitude()
dist=np.linalg.norm(np.diff(p,axis=0),axis=1)
# pair frame separation 2/5
for d in [1,2,5,10,20,50]:
 a=np.degrees(rot[:-d].inv()*rot[d:]).magnitude(); q=np.linalg.norm(p[d:]-p[:-d],axis=1)
 print('d',d,'rot median/p90/p99/max',np.percentile(a,[50,90,99,100]),'trans',np.percentile(q,[50,90,99,100]))
print('position range',p.min(0),p.max(0))
print('contact stats',np.percentile(z['contact_prob_right_left'],[0,10,25,50,75,90,100],axis=0))
# print jumps
idx=np.argsort(ang)[-80:][::-1]
print('top rot jumps frame angle distance')
for i in idx[:40]: print(i,round(ang[i],1),round(dist[i],3), 'p',np.round(p[i],2))
# per 25 frame summary
print('frame contactR contactL pos angle_abs_from0 relrot step')
for i in range(0,F,20):
 a=np.degrees((rot[0].inv()*rot[i]).magnitude())
 print(i, *(f'{v:.2f}' for v in z['contact_prob_right_left'][i]), np.round(p[i],2), f'{a:.1f}', f'{ang[i] if i<F-1 else 0:.1f}')
PY"
```
<details><summary>output</summary>

```
AttributeError: 'scipy.spatial.transform._rotation.Rotation' object has no attribute 'degrees'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
TypeError: loop of ufunc does not support argument 0 of type scipy.spatial.transform._rotation.Rotation which has no callable degrees method

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; F=len(T)
p=T[:,:3,3]; rot=R.from_matrix(T[:,:3,:3])
ang=np.degrees((rot[:-1].inv()*rot[1:]).magnitude())
dist=np.linalg.norm(np.diff(p,axis=0),axis=1)
for d in [1,2,5,10,20,50]:
 a=np.degrees((rot[:-d].inv()*rot[d:]).magnitude()); q=np.linalg.norm(p[d:]-p[:-d],axis=1)
 print('d',d,'rot median/p90/p99/max',np.round(np.percentile(a,[50,90,99,100]),2),'trans',np.round(np.percentile(q,[50,90,99,100]),3))
print('position range',p.min(0),p.max(0))
print('contact stats',np.percentile(z['contact_prob_right_left'],[0,10,25,50,75,90,100],axis=0))
idx=np.argsort(ang)[-80:][::-1]
print('top rot jumps frame angle distance p')
for i in idx[:50]: print(i,round(ang[i],1),round(dist[i],3), np.round(p[i],2),np.round(p[i+1],2))
print('frame contactR contactL pos absang step')
for i in range(0,F,20):
 a=np.degrees((rot[0].inv()*rot[i]).magnitude())
 print(i, *(f'{v:.2f}' for v in z['contact_prob_right_left'][i]), np.round(p[i],2), f'{a:.1f}', f'{ang[i] if i<F-1 else 0:.1f}')
PY"
```
<details><summary>output</summary>

```
d 1 rot median/p90/p99/max [ 0.81  2.64 10.51 28.84] trans [0.008 0.022 0.047 0.103]
d 2 rot median/p90/p99/max [ 1.25  3.84 13.21 42.8 ] trans [0.012 0.03  0.062 0.107]
d 5 rot median/p90/p99/max [ 2.47  7.51 24.19 47.7 ] trans [0.021 0.052 0.091 0.124]
d 10 rot median/p90/p99/max [ 4.61 13.95 35.76 48.05] trans [0.035 0.093 0.133 0.173]
d 20 rot median/p90/p99/max [ 9.54 25.26 57.75 74.28] trans [0.063 0.158 0.205 0.257]
d 50 rot median/p90/p99/max [24.77 50.85 81.31 87.29] trans [0.161 0.298 0.378 0.412]
position range [-0.25122246  0.44146514  2.543314  ] [0.6676629 1.1185822 3.8094642]
contact stats [[7.35601426e-14 8.86692250e-14]
 [2.30191450e-08 1.86399179e-10]
 [9.80316043e-01 6.64886892e-01]
 [9.99996066e-01 9.99989390e-01]
 [9.99999940e-01 9.99999702e-01]
 [1.00000000e+00 9.9999 …[+4607 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2, h5py, numpy as np, os
cap=cv2.VideoCapture('/work/inputs/video.mp4')
frames=[]; inds=list(range(0,993,25))
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
 if not ok: break
 im=cv2.resize(im,(384,288))
 cv2.putText(im,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
 frames.append(im)
rows=[]
for j in range(0,len(frames),5):
 row=frames[j:j+5]
 while len(row)<5: row.append(np.zeros_like(frames[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/contact.jpg',np.vstack(rows))
# masks contact overlays - load in chunks
with h5py.File('/work/inputs/masks.h5','r') as f:
 frames=[]
 for i in inds:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read()
  om=f['object'][i]; pm=f['person'][i]
  over=im.copy(); over[om]=(0.4*over[om]+0.6*np.array([0,0,255])).astype(np.uint8); over[pm]=(0.4*over[pm]+0.6*np.array([255,0,0])).astype(np.uint8)
  over=cv2.resize(over,(384,288));cv2.putText(over,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2);frames.append(over)
rows=[]
for j in range(0,len(frames),5):
 row=frames[j:j+5]
 while len(row)<5: row.append(np.zeros_like(frames[0]))
 rows.append(np.hstack(row))
cv2.imwrite('/work/scratch/contact_masks.jpg',np.vstack(rows))
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
from scipy.spatial.transform import Rotation as Rot
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; R=Rot.from_matrix(T[:,:3,:3]); R0=R[0]
for i in range(0,993,20):
 rv=(R0.inv()*R[i]).as_rotvec(); d=np.degrees(np.linalg.norm(rv)); y=np.degrees(rv[1]); planar=np.degrees(np.hypot(rv[0],rv[2]));
 print(i, 'tot',round(d,1),'aboutY',round(y,1),'xz',round(planar,1),'rv',np.round(np.degrees(rv),1))
PY"
```
<details><summary>output</summary>

```
0 tot 0.0 aboutY 0.0 xz 0.0 rv [0. 0. 0.]
20 tot 0.1 aboutY -0.0 xz 0.1 rv [ 0.  -0.   0.1]
40 tot 0.2 aboutY 0.0 xz 0.2 rv [-0.   0.   0.2]
60 tot 0.4 aboutY 0.0 xz 0.4 rv [-0.1  0.   0.4]
80 tot 0.5 aboutY -0.1 xz 0.5 rv [ 0.2 -0.1  0.4]
100 tot 0.5 aboutY 0.0 xz 0.5 rv [ 0.3  0.  -0.4]
120 tot 0.4 aboutY 0.2 xz 0.4 rv [0.4 0.2 0.1]
140 tot 0.4 aboutY -0.2 xz 0.3 rv [-0.1 -0.2  0.3]
160 tot 1.5 aboutY 0.5 xz 1.4 rv [0.4 0.5 1.3]
180 tot 6.2 aboutY 2.5 xz 5.7 rv [0.8 2.5 5.6]
200 tot 12.5 aboutY 10.6 xz 6.7 rv [ 2.1 10.6  6.3]
220 tot 22.4 aboutY 21.3 xz 6.9 rv [ 3.6 21.3  5.8]
240 tot 28.2 aboutY 27.2 xz 7.3 rv [ 5.2 27.2  5.1]
260 tot 37.2 aboutY 36.4 xz 7.9 rv [ 7.9 36.4  1.1]
280 tot 62.9 aboutY 61.5 xz 13.4 rv [13.4 61.5  1. ]
300 tot 76.5 aboutY 76.1 xz 7.4 rv [ 6.6 76.1  3.2]
320 t …[+1976 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
p='/work/inputs/object_mesh.glb'
s=trimesh.load(p,force='scene',process=False); g=s.to_geometry()
print(type(g),len(g))
for i,x in enumerate(g): print(i,x.vertices.shape,x.faces.shape,x.bounds, x.visual.kind if hasattr(x.visual,'kind') else type(x.visual))
print('scene bounds',s.bounds)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
TypeError: object of type 'Trimesh' has no len()

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; p=T[:,:3,3]
# floor dist centroid and estimated support? use each pose transform mesh later
n=np.array([-0.021970802584377374,-0.9980358822147039,-0.05866567689642774]); d=1.3898362825184551
s=p@n+d
print('centroid plane signed dist quantiles',np.round(np.percentile(s,[0,1,5,10,25,50,75,90,95,99,100]),3))
for a,b in [(0,90),(90,100),(100,250),(250,500),(500,600),(600,900),(930,993)]: print(a,b,np.round(np.percentile(s[a:b],[0,25,50,75,100]),3))
PY"
```
<details><summary>output</summary>

```
centroid plane signed dist quantiles [0.069 0.077 0.095 0.104 0.337 0.375 0.646 0.668 0.685 0.719 0.73 ]
0 90 [0.639 0.646 0.647 0.648 0.653]
90 100 [0.637 0.637 0.638 0.638 0.639]
100 250 [0.625 0.631 0.644 0.681 0.687]
250 500 [0.186 0.351 0.361 0.387 0.683]
500 600 [0.073 0.094 0.097 0.102 0.184]
600 900 [0.069 0.313 0.347 0.414 0.73 ]
930 993 [0.639 0.644 0.647 0.651 0.663]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False)
print(type(s), s.geometry.keys())
for k,x in s.geometry.items(): print(k,x.vertices.shape,x.faces.shape,'bounds',x.bounds,'visual',type(x.visual))
print('scene bounds',s.bounds)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.scene.scene.Scene'> odict_keys(['finaltable.001'])
finaltable.001 (240494, 3) (80196, 3) bounds [[-0.38281685 -0.38470846 -0.12187351]
 [ 0.40190151  0.38721192  0.66425836]] visual <class 'trimesh.visual.texture.TextureVisuals'>
scene bounds [[-0.38281685 -0.38470846 -0.12187351]
 [ 0.40190151  0.38721192  0.66425836]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json,numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); cam=json.load(open('/work/inputs/camera.json')); Wc=np.array(cam['T_world_cam']); cw=np.linalg.inv(Wc)
T=z['T_world_mesh']; Td=np.einsum('ij,njk->nik',cw,T); fp=z['T_cam_mesh_foundationpose']; co=z['T_cam_mesh_coconet']
for name,A,B in [('base-fp',Td,fp),('base-co',Td,co),('fp-co',fp,co)]:
 ra=R.from_matrix(A[:,:3,:3]);rb=R.from_matrix(B[:,:3,:3]); ang=np.degrees((ra.inv()*rb).magnitude()); dp=np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1)
 print(name,'rot q',np.round(np.percentile(ang,[0,10,25,50,75,90,100]),1),'trans q',np.round(np.percentile(dp,[0,10,25,50,75,90,100]),3))
 # frame sparse
 print([(i,round(ang[i],1),round(dp[i],3)) for i in range(0,993,50)])
PY"
```
<details><summary>output</summary>

```
base-fp rot q [ 0.7  2.   2.5  3.7  5.3  7.4 24.5] trans q [0.058 0.089 0.108 0.135 0.167 0.221 0.43 ]
[(0, 2.5, 0.152), (50, 2.4, 0.15), (100, 3.1, 0.092), (150, 3.9, 0.223), (200, 3.0, 0.126), (250, 1.2, 0.172), (300, 11.8, 0.13), (350, 9.7, 0.274), (400, 6.8, 0.245), (450, 3.1, 0.117), (500, 6.5, 0.11), (550, 6.3, 0.138), (600, 5.2, 0.136), (650, 6.5, 0.112), (700, 3.5, 0.08), (750, 5.3, 0.308), (800, 4.6, 0.131), (850, 5.0, 0.221), (900, 2.4, 0.144), (950, 2.3, 0.142)]
base-co rot q [1.3 1.3 1.3 1.3 1.4 1.4 1.4] trans q [0.058 0.065 0.072 0.08  0.093 0.123 0.218]
[(0, 1.3, 0.089), (50, 1.3, 0.089), (100, 1.3, 0.08), (150, 1.3, 0.096), (200, 1.3, 0.09), (250, 1.3, 0.098), (300, 1.3, 0.123), (350, 1.3, 0.188), (400, 1.3, 0.177), (450, 1.4, 0.081), (500, 1.4, 0.072), (550, 1.3, 0.08), (60 …[+623 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/pipeline_report.json && echo '---stages pose' && cat /pipeline/.stages/05_object_pose.json && echo '---coconet' && cat /pipeline/.stages/06_coconet.json && echo '---refine' && cat /pipeline/.stages/07_refinement.json && echo '---render' && cat /pipeline/.stages/08_render.json"
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
    "mtime_ns": 1792065993430039348,
    "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000014/inference/episode_000014_step200000_before_after.mp4",
    "size": 70191738
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.0 …[+15587 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json
for p in ['/pipeline/.stages/05_object_pose.json','/pipeline/.stages/06_coconet.json','/pipeline/.stages/07_refinement.json']:
 d=json.load(open(p)); print('\\n',p); print(d.get('signature',{}).get('command')); print('outputs', [x['path'] for x in d.get('outputs',[])])
PY"
```
<details><summary>output</summary>

```

 /pipeline/.stages/05_object_pose.json
['/opt/conda/bin/python', '/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py', '/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000014/export/episode_000014', '--depth-root', '/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000014/depth/moge2/aligned.h5', '--out-file', '/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000014/foundationpose/episode_000014.pkl', '--camera-id', '0', '--iteration', '5', '--max-attempts', '5', '--register-first-then-track', '--no-first-usable-frame-gt-rotation-oracle']
outputs ['/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000014/foundationpose/episode_000014.pkl']

 /pipeline/.stages/06_coconet.json
['/opt/conda/bin/python', '/workspac …[+2436 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
for name in ['episode_000014_step200000_reconstruction.mp4','episode_000014_step200000_before_after.mp4']:
 p='/pipeline/inference/'+name; cap=cv2.VideoCapture(p); print(name,cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FPS))
 inds=list(range(0,993,25)); ims=[]
 for i in inds:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();
  if not ok:break
  im=cv2.resize(im,(384,288));cv2.putText(im,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2);ims.append(im)
 rows=[]
 for j in range(0,len(ims),5):
  r=ims[j:j+5]
  while len(r)<5:r.append(np.zeros_like(ims[0]))
  rows.append(np.hstack(r))
 cv2.imwrite('/work/scratch/'+name.replace('.mp4','.jpg'),np.vstack(rows))
PY"
```
<details><summary>output</summary>

```
episode_000014_step200000_reconstruction.mp4 993.0 3072.0 1152.0 30.0
episode_000014_step200000_before_after.mp4 993.0 4608.0 1236.0 30.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5']:
 print('\\n',p)
 with h5py.File(p,'r') as f:
  def show(name,o):
   if hasattr(o,'shape'): print(name,o.shape,o.dtype)
  f.visititems(show)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
aligned/front_stereo_camera_left (993,) object
alignment/front_stereo_camera_left/scale (993,) float32
alignment/front_stereo_camera_left/shift (993,) float32
alignment/front_stereo_camera_left/valid_count (993,) int32
frame_names/front_stereo_camera_left (993,) object
raw/front_stereo_camera_left (993,) object

 /pipeline/inference/materialized_inputs.h5
K_rois (993, 3, 3) float32
bboxes (993, 4) float32
identity_json () object
input_rgbs (993, 3, 224, 224) float32
input_xyz (993, 6, 224, 224) float32
observed_human_masks (993, 224, 224) bool
observed_object_masks (993, 224, 224) bool
render_rgbs (993, 3, 224, 224) float32
render_xyz (993, 6, 224, 224) float32

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['nvdiffrast','torch','cv2','trimesh','open3d','pytorch3d','kornia','pyrender','scipy','skimage','h5py']
for m in mods:
 try:
  x=__import__(m); print(m,getattr(x,'__version__','ok'))
 except Exception as e: print(m,'ERR',type(e).__name__,str(e)[:120])
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124
cv2 4.11.0
trimesh 5.1.1
open3d 0.20.0
pytorch3d 0.7.9
kornia 0.8.3
pyrender 0.1.45
scipy 1.16.3
skimage 0.26.0
h5py 3.16.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,pickle
for p in ['/pipeline/inference/refined.pth','/pipeline/inference/coconet.pth','/pipeline/foundationpose/episode_000014.pkl','/pipeline/mhr_init/episode_000014.pkl']:
 print('\\n',p)
 try:
  if p.endswith('.pkl'):
   x=pickle.load(open(p,'rb')); print(type(x)); print(x.keys() if hasattr(x,'keys') else len(x))
   if hasattr(x,'keys'):
    for k,v in x.items(): print(' ',k,type(v),getattr(v,'shape',None))
  else:
   x=torch.load(p,map_location='cpu',weights_only=False); print(type(x)); print(x.keys() if hasattr(x,'keys') else '')
   if hasattr(x,'keys'):
    for k,v in x.items(): print(' ',k,type(v),getattr(v,'shape',None))
 except Exception as e: print('ERR',e)
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/refined.pth
<class 'dict'>
dict_keys(['schema', 'frames', 'frame_meta', 'kid', 'checkpoint', 'mhr_inference', 'config', 'mesh_diameter', 'mhr_neutral_height_init', 'mhr_spatial_scale', 'K_rois', 'bboxes', 'faces', 'pr', 'pr_initial', 'in', 'gt', 'observations', 'raw', 'metadata', 'postopt'])
  schema <class 'str'> None
  frames <class 'list'> None
  frame_meta <class 'list'> None
  kid <class 'int'> None
  checkpoint <class 'dict'> None
  mhr_inference <class 'dict'> None
  config <class 'dict'> None
  mesh_diameter <class 'numpy.ndarray'> (993,)
  mhr_neutral_height_init <class 'numpy.ndarray'> (993,)
  mhr_spatial_scale <class 'numpy.ndarray'> (993,)
  K_rois <class 'numpy.ndarray'> (993, 3, 3)
  bboxes <class 'numpy.ndarray'> (993, 4)
  faces <class 'numpy.ndarray' …[+4794 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
def desc(d,ind=0):
 for k,v in d.items():
  print(' '*ind,k,type(v).__name__,getattr(v,'shape',''),getattr(v,'dtype',''))
  if isinstance(v,dict) and ind<2: desc(v,ind+2)
print('pr');desc(x['pr'])
print('in');desc(x['in'])
print('raw');desc(x['raw'])
print('obs');desc(x['observations'])
print('postopt');desc(x['postopt'])
print('frame0',x['frames'][0])
print('meta0',x['frame_meta'][0])
PY"
```
<details><summary>output</summary>

```
pr
 pose_abs ndarray (993, 4, 4) float32
 pose_abs_1st_delta ndarray (993, 4, 4) float32
 mhr_global_rot6d ndarray (993, 6) float32
 mhr_trans ndarray (993, 3) float32
 mhr_body_pose_cont ndarray (993, 260) float32
 mhr_hand ndarray (993, 108) float32
 mhr_shape ndarray (993, 45) float32
 mhr_scale ndarray (993, 28) float32
 mhr_face ndarray (993, 72) float32
 contact_logits ndarray (993, 2) float32
 pose_abs_postopt ndarray (993, 4, 4) float32
in
 pose_abs ndarray (993, 4, 4) float32
 mhr_global_rot6d ndarray (993, 6) float32
 mhr_trans ndarray (993, 3) float32
 mhr_body_pose_cont ndarray (993, 260) float32
 mhr_hand ndarray (993, 108) float32
 mhr_shape ndarray (993, 45) float32
 mhr_scale ndarray (993, 28) float32
 mhr_face ndarray (993, 72) float32
raw
 trans ndarray (993, 3) float32
  …[+3170 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'rg -n "pose_abs_postopt|pose_abs.*S|oriented_bounds|T_world_mesh|fp_poses_cam|obj_pose_world" /code/reconstruction /workspace/v2d_cari4d 2>/dev/null | head -200'
```
<details><summary>output</summary>

```
/workspace/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py:553:    object_poses = np.asarray(fp["obj_pose_world"], dtype=np.float32)
/workspace/v2d_cari4d/lib/cari4d/learning/datasets/video_data.py:237:            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
/code/reconstruction/modules/v2d_foundation_pose/lib/mv_videos_to_poses.py:308:    _, obb_extents = trimesh.bounds.oriented_bounds(tm)
/code/reconstruction/modules/v2d_foundation_pose/lib/tests/test_multiview_recovery.py:88:        oriented_bounds=lambda mesh: (np.eye(4), np.ones(3)),
/code/reconstruction/modules/v2d_foundation_pose/lib/tests/test_multiview_recovery.py:677:        oriented_bounds=lambda mesh: (np.eye(4), np.ones(3)),
/code/reconstruction/modules/v2d_foundation_pose/lib/multiview_tracker.py:161:       …[+3603 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/export/episode_000014/wild_export.json && echo --- && python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
S,trans=trimesh.bounds.oriented_bounds(m)
print('S',S); print('trans',trans); print('mesh bounds',m.bounds)
m2=trimesh.load('/pipeline/export/episode_000014/object_mesh/output_aligned.glb',force='scene',process=False).to_geometry()
print('aligned bounds',m2.bounds,'n',len(m2.vertices),len(m2.faces),m2.vertices.shape)
print('first vertices',m.vertices[:3],m2.vertices[:3])
PY"
```
<details><summary>output</summary>

```
S [[ 1.22464680e-16  1.00000000e+00  0.00000000e+00 -1.25172734e-03]
 [ 1.00000000e+00 -1.22464680e-16  0.00000000e+00 -9.54233110e-03]
 [ 0.00000000e+00  0.00000000e+00 -1.00000000e+00  2.71192424e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
trans [0.77192038 0.78471836 0.78613187]
mesh bounds [[-0.38281685 -0.38470846 -0.12187351]
 [ 0.40190151  0.38721192  0.66425836]]
aligned bounds [[-0.38596019 -0.39235918 -0.39306593]
 [ 0.38596019  0.39235918  0.39306594]] n 240494 80196 (240494, 3)
first vertices [[-0.12094075 -0.36083236 -0.10394342]
 [-0.12094075 -0.36083236 -0.10394342]
 [-0.12094075 -0.36083236 -0.10394342]] [[-0.36208409 -0.13048308  0.37513585]
 [-0.36208409 -0.13048308  0.37513585]
 [-0.36208409 -0.13048308  0.37513585]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast.torch as dr, inspect
print(dr.__file__)
print(dr.RasterizeCudaContext)
print(dr.rasterize.__doc__)
PY"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/__init__.py
<class 'nvdiffrast.torch.ops.RasterizeCudaContext'>
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
        resolution: Output resolutio …[+698 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_eval.py <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
DEV='cuda'
W,H=1536,1152
# test render
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV)
f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Wc=np.array(cam['T_world_cam']); cw=np.linalg.inv(Wc)
gl=np.ones((H,W,4),dtype=np.float32); gl[:,:,0]=np.linspace(0,1,W); gl[:,:,1]=np.linspace(0,1,H)[:,None]
gl=torch.tensor(gl,device=DEV)
with h5py.File('/work/inputs/masks.h5','r') as hf:
 obj=hf['object'][0]; per=hf['person'][0]
print('masks',obj.sum(),per.sum(),'overlap',np.logical_and(obj,per).sum())
ctx=dr.RasterizeCudaContext()
scale=0.5
test_inds=[0,100,200,300,400,500,600,700,800,900,950]
for i in test_inds:
 Tw=np.eye(4);Tw[:3,:3]=T[i,:3,:3];Tw[:3,3]=T[i,:3,3]
 vc=(cw@Tw)[None,:3,:3]@v.T + (cw@Tw)[:3,3,None]
 zc=vc[2]; pix=v @ torch.tensor(K.T,dtype=torch.float32,device=DEV) # wrong
 # correct
 uv=(torch.tensor(K[:2,:2],device=DEV)@vc[:2]+torch.tensor(K[:2,2],device=DEV)[:,None])/zc
 ndc=torch.stack([uv[0]/W*2-1, 1-uv[1]/H*2, torch.zeros_like(zc), torch.ones_like(zc)],1)
 out,_=dr.rasterize(ctx,ndc[None],f,resolution=(H,W))
 mask=out[0,:,:,3]>0
 #  …[+523 chars]
```
<details><summary>output</summary>

```
masks 12545 30943 overlap 0
Traceback (most recent call last):
  File "/work/scratch/render_eval.py", line 22, in <module>
    vc=(cw@Tw)[None,:3,:3]@v.T + (cw@Tw)[:3,3,None]
       ~~~~~~~~~~~~~~~~~~~^~~~
TypeError: unsupported operand type(s) for @: 'numpy.ndarray' and 'Tensor'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_eval.py'
s=open(p).read();s=s.replace(\"Tw=np.eye(4);Tw[:3,:3]=T[i,:3,:3];Tw[:3,3]=T[i,:3,3]\\n vc=(cw@Tw)[None,:3,:3]@v.T + (cw@Tw)[:3,3,None]\\n zc=vc[2]; pix=v @ torch.tensor(K.T,dtype=torch.float32,device=DEV) # wrong\\n # correct\\n uv=(torch.tensor(K[:2,:2],device=DEV)@vc[:2]+torch.tensor(K[:2,2],device=DEV)[:,None])/zc\\n ndc=torch.stack([uv[0]/W*2-1, 1-uv[1]/H*2, torch.zeros_like(zc), torch.ones_like(zc)],1)\",\"Tw=np.eye(4,dtype=np.float32);Tw[:3,:3]=T[i,:3,:3];Tw[:3,3]=T[i,:3,3]\\n Tt=torch.tensor(cw@Tw,dtype=torch.float32,device=DEV)\\n vc=Tt[:3,:3]@v.T + Tt[:3,3,None]\\n zc=vc[2]\\n uv=(torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV)@vc[:2]+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[:,None])/zc\\n ndc=torch.stack([uv[0]/W*2-1, 1-uv[1]/H*2, torch.zeros_like(zc), torch.ones_like(zc)],1)\")
open(p,'w').write(s)
PY
python /work/scratch/render_eval.py"
```
<details><summary>output</summary>

```
masks 12545 30943 overlap 0
0 target 12545 pred 16624 IoU 0.0 visIoU 0.0 bias outside 16624 miss 12545
100 target 12545 pred 15969 IoU 0.0 visIoU 0.0 bias outside 15969 miss 12545
200 target 12545 pred 21502 IoU 0.0 visIoU 0.0 bias outside 21502 miss 12545
300 target 12545 pred 40433 IoU 0.0 visIoU 0.0 bias outside 40433 miss 12545
400 target 12545 pred 32491 IoU 0.0 visIoU 0.0 bias outside 32491 miss 12545
500 target 12545 pred 43868 IoU 0.0 visIoU 0.0 bias outside 43868 miss 12545
600 target 12545 pred 43775 IoU 0.0 visIoU 0.0 bias outside 43775 miss 12545
700 target 12545 pred 55148 IoU 0.005 visIoU 0.005 bias outside 54782 miss 12179
800 target 12545 pred 40931 IoU 0.0 visIoU 0.0 bias outside 40931 miss 12545
900 target 12545 pred 46215 IoU 0.0 visIoU 0.0 bias outside 46215 miss 12545
 …[+77 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py,trimesh
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [0,100,300,500,700,900]:
  Tw=T[i]; vc=(cw@Tw)[:3,:3]@v.T+(cw@Tw)[:3,3,None]; uv=(K[:2,:2]@vc[:2]+K[:2,2,None])/vc[2]; print(i,'proj bbox',uv.min(1),uv.max(1),'center',np.median(uv,1),'depth',np.percentile(vc[2],[0,50,100]))
  for name in ['object','person']:
   mask=f[name][i]; ys,xs=np.where(mask);print(name,len(xs), 'bbox', (xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None,'cent',np.mean(xs),np.mean(ys))
PY"
```
<details><summary>output</summary>

```
0 proj bbox [ -6.8432864  218.45444898] [181.77905754 479.2485566 ] center [ 88.01860892 266.12094805] depth [3.04701346 3.44737951 3.83700567]
object 12545 bbox (531, 629, 716, 860) cent 623.0357114388203 722.0810681546433
person 30943 bbox (785, 441, 917, 837) cent 857.4262676534273 633.4141485958052
100 proj bbox [ -7.35094473 216.67964929] [178.54897894 472.99271207] center [ 85.73130788 263.30639723] depth [3.0967282  3.49672152 3.88668702]
object 12308 bbox (531, 629, 709, 860) cent 621.7309879753006 721.87398440039
person 24341 bbox (656, 443, 814, 815) cent 743.755474302617 637.9460580912863
300 proj bbox [ 28.84132622 256.2431351 ] [285.38391162 516.40842681] center [176.04985906 380.27634073] depth [2.69410493 2.90598232 3.68237379]
object 38616 bbox (535, 656, 775, 887) cent 671 …[+1031 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py
z=np.load('/work/inputs/reconstruction.npz'); cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Wc=np.array(cam['T_world_cam']); cw=np.linalg.inv(Wc)
for i in [0,100,300,500,700,900]:
 v=z['human_vertices_world'][i];vc=cw[:3,:3]@v.T+cw[:3,3,None];uv=(K[:2,:2]@vc[:2]+K[:2,2,None])/vc[2]; print(i,'proj bbox',np.nanmin(uv,1),np.nanmax(uv,1),'cent',np.median(uv,1),'depth',np.percentile(vc[2],[0,50,100]))
 with h5py.File('/work/inputs/masks.h5','r') as f:
  m=f['person'][i];ys,xs=np.where(m);print(' mask', (xs.min(),ys.min(),xs.max(),ys.max()), np.mean(xs),np.mean(ys))
PY"
```
<details><summary>output</summary>

```
0 proj bbox [231.52620693  38.56809427] [381.44806417 438.83212576] cent [308.26115143 221.6393439 ] depth [3.33604289 3.57886296 3.85266238]
 mask (785, 441, 917, 837) 857.4262676534273 633.4141485958052
100 proj bbox [98.74004871 42.8413132 ] [249.15688688 402.39767948] cent [174.04024533 202.23257375] depth [3.4331562  3.70335365 3.96272846]
 mask (656, 443, 814, 815) 743.755474302617 637.9460580912863
300 proj bbox [137.40921202 153.52891822] [317.40379493 492.64157093] cent [253.81379533 277.16451083] depth [2.78775941 2.96730822 3.66526864]
 mask (650, 516, 826, 872) 768.2744936461812 655.780110161773
500 proj bbox [195.82074176 114.30330742] [404.27028812 506.82695306] cent [297.71556864 290.40924178] depth [2.46371589 2.89493381 3.47225155]
 mask (725, 468, 873, 872) 802.3590056800 …[+436 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Wc=np.array(cam['T_world_cam'])
for mode in ['inv','direct']:
 print(mode)
 for i in [0,300,700]:
  v=z['human_vertices_world'][i]; M=np.linalg.inv(Wc) if mode=='inv' else Wc
  vc=M[:3,:3]@v.T+M[:3,3,None];uv=(K[:2,:2]@vc[:2]+K[:2,2,None])/vc[2]; print(i,np.nanmin(uv,1),np.nanmax(uv,1),np.nanmedian(uv,1))
PY"
```
<details><summary>output</summary>

```
inv
0 [231.52620693  38.56809427] [381.44806417 438.83212576] [308.26115143 221.6393439 ]
300 [137.40921202 153.52891822] [317.40379493 492.64157093] [253.81379533 277.16451083]
700 [226.4464149  202.30646884] [571.70041852 512.49026256] [384.94475692 319.14840365]
direct
0 [242.38946785  63.76038321] [392.65473969 468.45679736] [318.97998198 246.92646806]
300 [143.89383969 181.68571272] [317.71566425 520.37531452] [252.33950295 303.05118538]
700 [229.39166822 233.39098012] [564.7036224  547.95846758] [382.92165192 350.23789919]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); v=z['human_vertices_world']; T=z['T_world_mesh'];
print('human frame0 bbox/mean',v[0].min(0),v[0].max(0),v[0].mean(0))
print('table frame0 pos',T[0,:3,3])
cam=json.load(open('/work/inputs/camera.json'));Wc=np.array(cam['T_world_cam'])
print('cam',Wc[:3,3])
print('object-center in cam',np.linalg.inv(Wc)[:3,:3]@T[0,:3,3]+np.linalg.inv(Wc)[:3,3])
PY"
```
<details><summary>output</summary>

```
human frame0 bbox/mean [-0.65612257 -0.398755    2.2162576 ] [-0.10837828  1.3007077   2.7753823 ] [-0.3881723  0.3257578  2.495157 ]
table frame0 pos [0.5965219  0.57728606 2.6447043 ]
cam [ 0.15850489 -0.28262919  6.02695573]
object-center in cam [-0.56585542  0.47404512  3.43889902]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc)
for signs in [(1,1,1),(-1,1,1),(1,-1,1),(1,1,-1),(-1,-1,1)]:
 S=np.diag([*signs,1]);
 print('signs',signs)
 for name,p in [('obj',z['T_world_mesh'][0,:3,3]),('hum',z['human_vertices_world'][0].mean(0))]:
  q=S[:3,:3]@p; c=cw[:3,:3]@q+cw[:3,3]; uv=K[:2,:2]@c[:2]/c[2]+K[:2,2]; print(name,np.round(uv,1),round(c[2],2))
PY"
```
<details><summary>output</summary>

```
obj [623.2 673.5] 3.44
hum [852.9 609.4] 3.59
signs (-1, 1, 1)
obj [905.6 674.2] 3.48
hum [675.3 608.5] 3.57
signs (1, -1, 1)
obj [621.1 393.7] 3.31
hum [856.6 459.7] 3.52
signs (1, 1, -1)
obj [687.2 549.6] 8.69
hum [781.2 527.7] 8.55
signs (-1, -1, 1)
obj [914.4 398. ] 3.35
hum [675.3 457.6] 3.49

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_eval.py';s=open(p).read();s=s.replace(\"uv=(torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV)@vc[:2]+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[:,None])/zc\",\"uv=torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV)@vc[:2]/zc+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[:,None]\")
# load current mask each frame
s=s.replace(\"obj=hf['object'][0]; per=hf['person'][0]\",\"obj=hf['object'][0]; per=hf['person'][0]\")
s=s.replace(\"for i in test_inds:\\n Tw=\", \"for i in test_inds:\\n with h5py.File('/work/inputs/masks.h5','r') as hf:\\n  obj=hf['object'][i]; per=hf['person'][i]\\n Tw=\")
open(p,'w').write(s)
PY
python /work/scratch/render_eval.py"
```
<details><summary>output</summary>

```
masks 12545 30943 overlap 0
0 target 12545 pred 11130 IoU 0.0 visIoU 0.0 bias outside 11130 miss 12545
100 target 12308 pred 10809 IoU 0.0 visIoU 0.0 bias outside 10809 miss 12308
200 target 17013 pred 16523 IoU 0.0 visIoU 0.0 bias outside 16523 miss 17013
300 target 38616 pred 40563 IoU 0.0 visIoU 0.0 bias outside 40563 miss 38616
400 target 38086 pred 39329 IoU 0.0 visIoU 0.0 bias outside 39329 miss 38086
500 target 25531 pred 25083 IoU 0.0 visIoU 0.0 bias outside 25083 miss 25531
600 target 31266 pred 31990 IoU 0.0 visIoU 0.0 bias outside 31990 miss 31266
700 target 51530 pred 50615 IoU 0.0 visIoU 0.0 bias outside 50615 miss 51530
800 target 55025 pred 54089 IoU 0.0 visIoU 0.0 bias outside 54089 miss 55025
900 target 28109 pred 27751 IoU 0.0 visIoU 0.0 bias outside 27751 miss 28109
950  …[+73 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_eval.py';s=open(p).read();
s=s.replace(\"m=mask.cpu().numpy(); vis=m & ~per\",\"m=mask.cpu().numpy(); vis=m & ~per\\n if i==300:\\n  cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read(); ov=im.copy();ov[m]=(0.3*ov[m]+0.7*np.array([0,0,255])).astype(np.uint8);ov[obj]=(0.3*ov[obj]+0.7*np.array([0,255,0])).astype(np.uint8);cv2.imwrite('/work/scratch/overlay300.jpg',ov)\")
open(p,'w').write(s)
PY
python /work/scratch/render_eval.py >/dev/null"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
cam=json.load(open('/work/inputs/camera.json'));Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc);T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
for i in [0,300]:
 p=T[i,:3,3];print(Wc);print('inv',cw);print('p',p,'cam',cw[:3,:3]@p+cw[:3,3])
PY"
```
<details><summary>output</summary>

```
[[-0.99932338 -0.00716846 -0.03607473  0.15850489]
 [-0.01114394  0.99373132  0.1112379  -0.28262919]
 [ 0.03505118  0.11156465 -0.99313883  6.02695573]
 [ 0.          0.          0.          1.        ]]
inv [[-0.99932338 -0.01114394  0.03505118 -0.0560039 ]
 [-0.00716846  0.99373132  0.11156465 -0.3904015 ]
 [-0.03607473  0.1112379  -0.99313883  6.02276088]
 [ 0.          0.          0.          1.        ]]
p [0.5965219  0.57728606 2.6447043 ] cam [-0.56585542  0.47404512  3.43889902]
[[-0.99932338 -0.00716846 -0.03607473  0.15850489]
 [-0.01114394  0.99373132  0.1112379  -0.28262919]
 [ 0.03505118  0.11156465 -0.99313883  6.02695573]
 [ 0.          0.          0.          1.        ]]
inv [[-0.99932338 -0.01114394  0.03505118 -0.0560039 ]
 [-0.00716846  0.99373132  0.11156465 -0.390401 …[+186 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/scratch/render_eval.py | sed -n '18,45p'"
```
<details><summary>output</summary>

```
    18	scale=0.5
    19	test_inds=[0,100,200,300,400,500,600,700,800,900,950]
    20	for i in test_inds:
    21	 with h5py.File('/work/inputs/masks.h5','r') as hf:
    22	  obj=hf['object'][i]; per=hf['person'][i]
    23	 Tw=np.eye(4,dtype=np.float32);Tw[:3,:3]=T[i,:3,:3];Tw[:3,3]=T[i,:3,3]
    24	 Tt=torch.tensor(cw@Tw,dtype=torch.float32,device=DEV)
    25	 vc=Tt[:3,:3]@v.T + Tt[:3,3,None]
    26	 zc=vc[2]
    27	 uv=torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV)@vc[:2]/zc+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[:,None]
    28	 ndc=torch.stack([uv[0]/W*2-1, 1-uv[1]/H*2, torch.zeros_like(zc), torch.ones_like(zc)],1)
    29	 out,_=dr.rasterize(ctx,ndc[None],f,resolution=(H,W))
    30	 mask=out[0,:,:,3]>0
    31	 # visible target excluding person
    32	 m=mask.cpu().num …[+772 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_eval.py';s=open(p).read();s=s.replace(\"print(i,'target',obj.sum(),'pred',m.sum(),'IoU'\", \"ys,xs=np.where(m); print('pred bbox',xs.min(),ys.min(),xs.max(),ys.max()) if i==300 else None\\n print(i,'target',obj.sum(),'pred',m.sum(),'IoU'\")
open(p,'w').write(s)
PY
python /work/scratch/render_eval.py | head -25"
```
<details><summary>output</summary>

```
masks 12545 30943 overlap 0
0 target 12545 pred 11130 IoU 0.0 visIoU 0.0 bias outside 11130 miss 12545
100 target 12308 pred 10809 IoU 0.0 visIoU 0.0 bias outside 10809 miss 12308
200 target 17013 pred 16523 IoU 0.0 visIoU 0.0 bias outside 16523 miss 17013
pred bbox 566 278 786 510
300 target 38616 pred 40563 IoU 0.0 visIoU 0.0 bias outside 40563 miss 38616
400 target 38086 pred 39329 IoU 0.0 visIoU 0.0 bias outside 39329 miss 38086
500 target 25531 pred 25083 IoU 0.0 visIoU 0.0 bias outside 25083 miss 25531
600 target 31266 pred 31990 IoU 0.0 visIoU 0.0 bias outside 31990 miss 31266
700 target 51530 pred 50615 IoU 0.0 visIoU 0.0 bias outside 50615 miss 51530
800 target 55025 pred 54089 IoU 0.0 visIoU 0.0 bias outside 54089 miss 55025
900 target 28109 pred 27751 IoU 0.0 visIoU 0.0 bias out …[+99 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_eval.py').read().split('for i in test_inds:')[0])
i=300;Tw=np.eye(4,dtype=np.float32);Tw[:3,:3]=T[i,:3,:3];Tw[:3,3]=T[i,:3,3];Tt=torch.tensor(cw@Tw,dtype=torch.float32,device=DEV);vc=Tt[:3,:3]@v.T+Tt[:3,3,None];zc=vc[2];uv=torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV)@vc[:2]/zc+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[:,None]
print('uv bbox',uv.min(1).values.cpu(),uv.max(1).values.cpu(),'z',zc.min().item(),zc.max().item())
ndc=torch.stack([uv[0]/W*2-1, 1-uv[1]/H*2, torch.zeros_like(zc), torch.ones_like(zc)],1)
print('ndc y',ndc[:,1].min().item(),ndc[:,1].max().item())
out,_=dr.rasterize(ctx,ndc[None],f,resolution=(H,W));m=out[0,:,:,3]>0;ys,xs=torch.where(m);print('out bbox',xs.min().item(),ys.min().item(),xs.max().item(),ys.max().item())
# alternate
for yy in [1-uv[1]/H*2,uv[1]/H*2-1]:
 ndc2=torch.stack([uv[0]/W*2-1, yy, torch.zeros_like(zc), torch.ones_like(zc)],1);o,_=dr.rasterize(ctx,ndc2[None],f,resolution=(H,W));mm=o[0,:,:,3]>0; y,x=torch.where(mm);print('alt bbox',x.min().item(),y.min().item(),x.max().item(),y.max().item())
PY"
```
<details><summary>output</summary>

```
masks 12545 30943 overlap 0
uv bbox tensor([565.5143, 641.3365]) tensor([786.9451, 873.8246]) z 2.6941049098968506 3.6823737621307373
ndc y -0.5170565843582153 -0.11343157291412354
out bbox 566 278 786 510
alt bbox 566 278 786 510
alt bbox 566 641 786 873

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/1-uv\\[1\\]\\/H\\*2/uv[1]\\/H*2-1/' /work/scratch/render_eval.py && python /work/scratch/render_eval.py"
```
<details><summary>output</summary>

```
masks 12545 30943 overlap 0
0 target 12545 pred 11131 IoU 0.768 visIoU 0.768 bias outside 847 miss 2261
100 target 12308 pred 10807 IoU 0.767 visIoU 0.773 bias outside 773 miss 2274
200 target 17013 pred 16526 IoU 0.826 visIoU 0.845 bias outside 1356 miss 1843
pred bbox 566 641 786 873
300 target 38616 pred 40563 IoU 0.826 visIoU 0.876 bias outside 4745 miss 2798
400 target 38086 pred 39327 IoU 0.922 visIoU 0.939 bias outside 2189 miss 948
500 target 25531 pred 25083 IoU 0.803 visIoU 0.818 bias outside 2546 miss 2994
600 target 31266 pred 31992 IoU 0.804 visIoU 0.819 bias outside 3806 miss 3080
700 target 51530 pred 50616 IoU 0.877 visIoU 0.886 bias outside 2891 miss 3805
800 target 55025 pred 54090 IoU 0.721 visIoU 0.726 bias outside 8362 miss 9297
900 target 28109 pred 27751 IoU 0.719 vi …[+118 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_all.py <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh, time
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
DEV='cuda'; W,H=768,576
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV); f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; F=len(T)
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Wc=np.array(cam['T_world_cam']); cw=np.linalg.inv(Wc)
ctx=dr.RasterizeCudaContext(); ious=[]; prec=[]; rec=[]; masks=[]
t=time.time()
with h5py.File('/work/inputs/masks.h5','r') as hf:
 for s in range(0,F,24):
  e=min(F,s+24); B=e-s
  Tw=torch.tensor(T[s:e],dtype=torch.float32,device=DEV)
  R=Tt=torch.tensor(cw,dtype=torch.float32,device=DEV)[None]@Tw
  vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3]
  zc=vc[:,:,2]; uv=torch.einsum('ij,bvj->bvi',torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV),vc[:,:,:2])/zc[:,:,None]+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[None,None]
  ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2)
  out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W)); m=out[:,:,:,3]>0
  for j in range(B):
   gt=cv2.resize(hf['object'][s+j].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
   pm=m[j].cpu().numpy(); in …[+583 chars]
```
<details><summary>output</summary>

```
0
24
48
72
96
120
144
168
192
216
240
264
288
312
336
360
384
408
432
456
480
504
528
552
576
600
624
648
672
696
720
744
768
792
816
840
864
888
912
936
960
984
time 49.49493455886841 iou q [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
0 100 mean 0.0 med 0.0 p10 0.0
100 930 mean 0.0 med 0.0 p10 0.0
930 993 mean 0.0 med 0.0 p10 0.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/eval_all.py').read().split('for s in range')[0])
s=0;e=24;B=24;Tw=torch.tensor(T[s:e],dtype=torch.float32,device=DEV);Tt=torch.tensor(cw,dtype=torch.float32,device=DEV)[None]@Tw
vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV),vc[:,:,:2])/zc[:,:,None]+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[None,None]
print(vc.shape,uv.min((1,2)).values.cpu(),uv.max((1,2)).values.cpu())
ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2);print(ndc.shape,ndc[0,:5]);
out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));print(out.shape,out[:,:,:,3].min().item(),out[:,:,:,3].max().item(),(out[:,:,:,3]>0).sum((1,2)))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<string>", line 12
    
    ^
IndentationError: expected an indented block after 'with' statement on line 11

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh
import nvdiffrast.torch as dr
DEV='cuda';W,H=768,576
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);cw=np.linalg.inv(np.array(cam['T_world_cam']));ctx=dr.RasterizeCudaContext()
Tw=torch.tensor(T[:24],dtype=torch.float32,device=DEV);Tt=torch.tensor(cw,dtype=torch.float32,device=DEV)[None]@Tw
vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV),vc[:,:,:2])/zc[:,:,None]+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[None,None]
print(vc.shape,uv.min((1,2)).values.cpu(),uv.max((1,2)).values.cpu())
ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2);print(ndc.shape,ndc.is_contiguous(),ndc[0,:5]);
out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));print(out.shape,out[:,:,:,3].min().item(),out[:,:,:,3].max().item(),(out[:,:,:,3]>0).sum((1,2)))
with h5py.File('/work/inputs/masks.h5','r') as hf:
 gt=cv2.resize(hf['object'][0].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST);print('gt',gt.sum());ys,xs=np.where(out[0,:,:,3].cpu()>0);print(xs.min(),ys.min(),xs.max( …[+15 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
TypeError: min() received an invalid combination of arguments - got (tuple), but expected one of:
 * ()
      didn't match because some of the arguments have invalid types: (!tuple of (int, int)!)
 * (Tensor other)
      didn't match because some of the arguments have invalid types: (!tuple of (int, int)!)
 * (int dim, bool keepdim = False)
 * (name dim, bool keepdim = False)


```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh
import nvdiffrast.torch as dr
DEV='cuda';W,H=768,576
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);cw=np.linalg.inv(np.array(cam['T_world_cam']));ctx=dr.RasterizeCudaContext()
Tw=torch.tensor(T[:24],dtype=torch.float32,device=DEV);Tt=torch.tensor(cw,dtype=torch.float32,device=DEV)[None]@Tw
vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV),vc[:,:,:2])/zc[:,:,None]+torch.tensor(K[:2,2],dtype=torch.float32,device=DEV)[None,None]
print(vc.shape,uv.amin((1,2)).cpu(),uv.amax((1,2)).cpu())
ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2);print(ndc.shape,ndc.is_contiguous(),ndc[0,:5]);
out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));print(out.shape,out[:,:,:,3].min().item(),out[:,:,:,3].max().item(),(out[:,:,:,3]>0).sum((1,2)))
with h5py.File('/work/inputs/masks.h5','r') as hf:
 gt=cv2.resize(hf['object'][0].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST);print('gt',gt.sum());ys,xs=np.where(out[0,:,:,3].cpu()>0);print(xs.min(),ys.min(),xs.max(),ys.max())
 …[+3 chars]
```
<details><summary>output</summary>

```
torch.Size([24, 240494, 3]) tensor([528.3929, 528.3951, 528.0627, 528.3605, 528.1205, 528.3361, 528.2848,
        528.5132, 528.4101, 528.0138, 528.5089, 527.9993, 528.0248, 528.5400,
        528.2983, 528.3184, 528.7916, 528.3068, 528.6031, 528.5590, 528.7615,
        528.9512, 528.6749, 528.7422]) tensor([861.3938, 861.5345, 861.2544, 861.1278, 861.0303, 861.0560, 861.1875,
        860.9351, 861.1968, 861.3018, 861.0508, 860.9807, 861.0197, 860.9807,
        861.4288, 861.0570, 861.2239, 861.4244, 861.2809, 861.4723, 861.5366,
        861.3617, 861.5126, 861.5393])
torch.Size([24, 240494, 4]) True tensor([[0.3935, 1.2748, 0.0000, 1.0000],
        [0.3935, 1.2748, 0.0000, 1.0000],
        [0.3935, 1.2748, 0.0000, 1.0000],
        [0.3935, 1.2748, 0.0000, 1.0000],
        [0.3928, 1.2742,  …[+570 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_all.py';s=open(p).read();s=s.replace(\"cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Wc=np.array(cam['T_world_cam']); cw=np.linalg.inv(Wc)\",\"cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); K[0]*=W/1536; K[1]*=H/1152; Wc=np.array(cam['T_world_cam']); cw=np.linalg.inv(Wc)\")
open(p,'w').write(s)
PY
python /work/scratch/eval_all.py"
```
<details><summary>output</summary>

```
0
24
48
72
96
120
144
168
192
216
240
264
288
312
336
360
384
408
432
456
480
504
528
552
576
600
624
648
672
696
720
744
768
792
816
840
864
888
912
936
960
984
time 46.97381091117859 iou q [0.40039286 0.57299944 0.69819568 0.73723733 0.76725434 0.79962488
 0.84825295 0.89992451 0.91240966 0.93546807 0.94617059]
0 100 mean 0.8029709255792875 med 0.8010970624849402 p10 0.7873811902925472
100 930 mean 0.8058042105079178 med 0.8086505300454739 p10 0.7314170169460076
930 993 mean 0.7819011234645242 med 0.7835220125786163 p10 0.7655180174831289

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/baseline_eval.npz');q=z['ious']
for a,b in [(0,100),(100,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,800),(800,900),(900,930),(930,993)]: print(a,b,round(q[a:b].mean(),3),round(q[a:b].min(),3))
i=np.argsort(q)[:50];print('low',[(int(j),round(float(q[j]),3)) for j in i])
# every 10
print('seq',[(j,round(float(q[j]),3)) for j in range(0,993,20)])
PY"
```
<details><summary>output</summary>

```
0 100 0.803 0.779
100 200 0.772 0.723
200 300 0.865 0.652
300 400 0.909 0.752
400 500 0.767 0.513
500 600 0.831 0.756
600 700 0.818 0.748
700 800 0.769 0.586
800 900 0.73 0.4
900 930 0.752 0.727
930 993 0.782 0.751
low [(881, 0.4), (882, 0.425), (880, 0.468), (883, 0.488), (879, 0.496), (479, 0.513), (478, 0.527), (474, 0.531), (473, 0.543), (884, 0.553), (885, 0.575), (475, 0.578), (878, 0.584), (758, 0.586), (757, 0.587), (480, 0.59), (759, 0.594), (886, 0.601), (877, 0.609), (756, 0.61), (477, 0.618), (755, 0.62), (752, 0.62), (887, 0.628), (751, 0.632), (876, 0.632), (750, 0.634), (476, 0.639), (753, 0.642), (754, 0.646), (283, 0.652), (481, 0.656), (482, 0.658), (875, 0.663), (760, 0.672), (769, 0.672), (472, 0.673), (768, 0.674), (873, 0.68), (770, 0.68), (287, 0.683), (761, 0.687),  …[+803 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/overlays.py <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh, sys
import nvdiffrast.torch as dr
DEV='cuda'; W,H=768,576; SC=0.5
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);K[0]*=SC;K[1]*=SC;cw=np.linalg.inv(np.array(cam['T_world_cam']));ctx=dr.RasterizeCudaContext()
inds=[0,120,200,283,300,400,479,500,600,700,758,800,881,900,950,980]
cap=cv2.VideoCapture('/work/inputs/video.mp4'); outs=[]
with h5py.File('/work/inputs/masks.h5','r') as hf:
 for i in inds:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.resize(im,(W,H))
  Tw=torch.tensor(T[i],dtype=torch.float32,device=DEV);Tt=torch.tensor(cw,dtype=torch.float32,device=DEV)@Tw;vc=Tt[:3,:3]@v.T+Tt[:3,3,None];zc=vc[2];uv=torch.tensor(K[:2,:2],device=DEV)@vc[:2]/zc+torch.tensor(K[:2,2],device=DEV)[:,None]
  ndc=torch.stack([uv[0]/W*2-1,uv[1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],1);out,_=dr.rasterize(ctx,ndc[None],f,resolution=(H,W));m=out[0,:,:,3].cpu().numpy()>0
  gt=cv2.resize(hf['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
  ov=im.copy();ov[m]=(0.35*ov[m]+0.65*np.array([0,0,255])).astype(np.uint8);ov[gt]=(0.35*ov[gt]+0.65*np.array([ …[+338 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/overlays.py", line 11, in <module>
    Tw=torch.tensor(T[i],dtype=torch.float32,device=DEV);Tt=torch.tensor(cw,dtype=torch.float32,device=DEV)@Tw;vc=Tt[:3,:3]@v.T+Tt[:3,3,None];zc=vc[2];uv=torch.tensor(K[:2,:2],device=DEV)@vc[:2]/zc+torch.tensor(K[:2,2],device=DEV)[:,None]
                                                                                                                                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: double != float

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/K=np.array(cam\\['K'\\])/K=np.array(cam['K'],dtype=np.float32)/\" /work/scratch/overlays.py && python /work/scratch/overlays.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
q=np.load('/work/scratch/baseline_eval.npz')['ious']
for a,b in [(270,310),(370,410),(460,510),(590,630),(740,790),(870,900)]:
 print('\\n',a,b)
 for i in range(a,b,2): print(i,round(float(q[i]),3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```

 270 310
270 0.885; 272 0.907; 274 0.906; 276 0.903; 278 0.89; 280 0.914; 282 0.909; 284 0.741; 286 0.7; 288 0.868; 290 0.872; 292 0.899; 294 0.857; 296 0.873; 298 0.854; 300 0.823; 302 0.78; 304 0.833; 306 0.946; 308 0.945; 

 370 410
370 0.91; 372 0.912; 374 0.908; 376 0.89; 378 0.893; 380 0.91; 382 0.907; 384 0.888; 386 0.907; 388 0.892; 390 0.904; 392 0.907; 394 0.901; 396 0.898; 398 0.916; 400 0.917; 402 0.901; 404 0.893; 406 0.887; 408 0.856; 

 460 510
460 0.768; 462 0.746; 464 0.748; 466 0.749; 468 0.744; 470 0.708; 472 0.673; 474 0.531; 476 0.639; 478 0.527; 480 0.59; 482 0.658; 484 0.724; 486 0.73; 488 0.751; 490 0.764; 492 0.801; 494 0.794; 496 0.812; 498 0.819; 500 0.82; 502 0.8; 504 0.78; 506 0.781; 508 0.776; 

 590 630
590 0.857; 592 0.849; 594 0.824; 596 0.828; 598 0.813;  …[+619 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
inds=list(range(270,311,4))+list(range(370,411,4))+list(range(450,511,4))+list(range(580,631,4))+list(range(735,791,4))+list(range(860,901,4))
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in inds:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.resize(im,(384,288));cv2.putText(im,str(i),(8,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2);ims.append(im)
rows=[]
for j in range(0,len(ims),6):
 r=ims[j:j+6]
 while len(r)<6:r.append(np.zeros_like(ims[0]))
 rows.append(np.hstack(r))
cv2.imwrite('/work/scratch/jump_frames.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/smooth_eval.py <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh, time, os
from scipy.signal import savgol_filter, medfilt
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr
DEV='cuda'; W,H=384,288; SC=0.25
z=np.load('/work/inputs/reconstruction.npz'); T0=z['T_world_mesh'].astype(np.float64); F=len(T0)
# cache gt masks
cache='/work/scratch/gt_masks_384.npy'
if os.path.exists(cache): GT=np.load(cache)
else:
 GT=np.empty((F,H,W),bool)
 with h5py.File('/work/inputs/masks.h5','r') as hf:
  for i in range(F): GT[i]=cv2.resize(hf['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)
 np.save(cache,GT)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);K[0]*=SC;K[1]*=SC;cw=np.linalg.inv(np.array(cam['T_world_cam'])).astype(np.float32);ctx=dr.RasterizeCudaContext()
Kt=torch.tensor(K[:2,:2],device=DEV);ct=torch.tensor(K[:2,2],device=DEV);cwt=torch.tensor(cw,device=DEV)
def evalT(T):
 ious=np.empty(F)
 with torch.no_grad():
  for s in range(0,F,24):
   e=min(F,s+24);B=e-s;Tw=torch.tensor(T[s:e],dtype=torch.float32,device=DEV);Tt=cwt[None]@Tw
   vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',Kt,vc[:, …[+3001 chars]
```
<details><summary>output</summary>

```
base time 0.5 all 0.8102 0-100 0.8276 100-930 0.8092 930+ 0.7958 p10 0.745
sg5 time 0.3 all 0.8146 0-100 0.8326 100-930 0.8134 930+ 0.8011 p10 0.745
sg9 time 0.4 all 0.815 0-100 0.8334 100-930 0.814 930+ 0.7997 p10 0.745
sg13 time 0.4 all 0.8152 0-100 0.8324 100-930 0.8143 930+ 0.8001 p10 0.747
sg17 time 0.4 all 0.8155 0-100 0.8326 100-930 0.8146 930+ 0.7997 p10 0.747
sg25 time 0.4 all 0.8154 0-100 0.8316 100-930 0.8147 930+ 0.8 p10 0.744
rob_sg9 time 0.3 all 0.8121 0-100 0.8334 100-930 0.8104 930+ 0.7997 p10 0.745
rob_sg15 time 0.3 all 0.8096 0-100 0.8326 100-930 0.8076 930+ 0.7997 p10 0.746
rob_sg25 time 0.3 all 0.8052 0-100 0.8316 100-930 0.8024 930+ 0.8 p10 0.74
med5_sg9 time 0.4 all 0.8109 0-100 0.829 100-930 0.8093 930+ 0.8034 p10 0.743
med9_sg9 time 0.4 all 0.8115 0-100 0.829 100-93 …[+29 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/pose_opt.py <<'PY'
import numpy as np, torch, json, cv2, os, time
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr
DEV='cuda'; W,H=384,288; SC=0.25
z=np.load('/work/inputs/reconstruction.npz'); T0=z['T_world_mesh'].astype(np.float64); F=len(T0)
GT=np.load('/work/scratch/gt_masks_384.npy')
import trimesh
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
cam=json.load(open('/work/inputs/camera.json'));Kfull=np.array(cam['K'],dtype=np.float64);K=Kfull.copy();K[0]*=SC;K[1]*=SC;Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc);ctx=dr.RasterizeCudaContext()
Kt=torch.tensor(K[:2,:2],dtype=torch.float32,device=DEV);ct=torch.tensor(K[:2,2],dtype=torch.float32,device=DEV);cwt=torch.tensor(cw,dtype=torch.float32,device=DEV);Rwc=Wc[:3,:3]
def evalT(T):
 ious=np.empty(F);masks=[]
 with torch.no_grad():
  for s in range(0,F,24):
   e=min(F,s+24);B=e-s;Tw=torch.tensor(T[s:e],dtype=torch.float32,device=DEV);Tt=cwt[None]@Tw
   vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',Kt,vc[:,:,:2])/zc[:,:,None]+ct[None,None]
   ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2)
   out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));m=out[:,:,: …[+2415 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/pose_opt.py", line 43, in <module>
iter 0 base 0.8153559547275979
    C=T.copy();pc=cw[:3,:3]@C[:,:3,3]+cw[:3,3];zcam=pc[:,2]
                  ~~~~~~~~~^~~~~~~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 993 is different from 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/cw\\[:3,:3\\]@C\\[:,:3,3\\]/C[:,:3,3]@cw[:3,:3].T/' /work/scratch/pose_opt.py && python /work/scratch/pose_opt.py"
```
<details><summary>output</summary>

```
iter 0 base 0.8153559547275979
tx [0.7472 0.789  0.8154 0.8053 0.7668] -> 0.8256
ty [0.7702 0.8175 0.8256 0.7812 0.7222] -> 0.8393
ray [0.8007 0.8273 0.8359 0.8393 0.8381 0.8335 0.8107] -> 0.8492
rx [0.7032 0.7901 0.8293 0.8492 0.8299 0.7929 0.7137] -> 0.8514
ry [0.7799 0.8243 0.8418 0.8514 0.844  0.8276 0.7848] -> 0.8538
rz [0.6552 0.7627 0.8219 0.8538 0.8198 0.7596 0.6518] -> 0.854
yaw [0.7724 0.823  0.8429 0.854  0.8422 0.8218 0.7734] -> 0.8543
iter 1 base 0.8543476394059297
tx [0.7718 0.8238 0.8543 0.8309 0.7813] -> 0.8551
ty [0.7579 0.8211 0.8551 0.8122 0.7502] -> 0.856
ray [0.809  0.8388 0.8499 0.856  0.8512 0.8434 0.8174] -> 0.8574
rx [0.706  0.7949 0.8349 0.8574 0.8347 0.7954 0.7136] -> 0.8577
ry [0.7818 0.8282 0.8465 0.8577 0.8476 0.8292 0.7864] -> 0.8581
rz [0.6562 0.7641 0.8241  …[+1147 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/opt_raw.npz');T=z['T'];q=z['ious'];T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
r=R.from_matrix(T[:,:3,:3]);r0=R.from_matrix(T0[:,:3,:3]);ang=np.degrees((r0.inv()*r).magnitude());dist=np.linalg.norm(T[:,:3,3]-T0[:,:3,3],axis=1)
vel=np.degrees((r[:-1].inv()*r[1:]).magnitude());td=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
print('diff from base q',np.percentile(ang,[0,10,25,50,75,90,100]),np.percentile(dist,[0,10,25,50,75,90,100]))
print('T vel',np.percentile(vel,[50,75,90,95,99,100]),np.percentile(td,[50,75,90,95,99,100]))
print('static rot',np.ptp(ang[:100]),np.ptp(ang[930:]),'iou',q[:100].mean(),q[930:].mean())
for i in range(0,993,25):
 print(i, round(float(q[i]),3),round(ang[i],1),round(dist[i],3),round(vel[min(i,991)],1))
PY"
```
<details><summary>output</summary>

```
diff from base q [3.62509930e-02 1.73297937e-01 4.75944404e-01 1.45106371e+00
 2.33534702e+00 3.95842337e+00 3.98088731e+01] [0.00202356 0.01498236 0.02126288 0.03471838 0.05469226 0.08121868
 0.18673293]
T vel [ 1.32882863  2.03574527  3.25944857  4.73442765 14.08810084 31.42097187] [0.01532434 0.02260923 0.03940539 0.04489384 0.069999   0.08383347]
static rot 1.754655261619444 1.9218569594488475 iou 0.8379419889038385 0.8192316005491568
0 0.833 1.5 0.039 0.0
25 0.843 0.1 0.016 0.0
50 0.851 0.0 0.021 1.4
75 0.853 1.3 0.022 1.5
100 0.818 0.1 0.058 0.0
125 0.807 0.0 0.084 0.1
150 0.789 0.1 0.035 3.0
175 0.808 0.1 0.012 0.3
200 0.864 0.4 0.045 1.6
225 0.896 0.3 0.025 0.5
250 0.911 0.1 0.059 1.7
275 0.955 4.3 0.12 3.0
300 0.964 10.0 0.082 12.2
325 0.951 2.2 0.049 0.1
350 0.938 3.3 0.051 4.6
3 …[+592 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_variants.py <<'PY'
import numpy as np, torch, json, cv2, trimesh, time
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr
DEV='cuda'; W,H=384,288; SC=0.25; F=993
GT=np.load('/work/scratch/gt_masks_384.npy')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);K[0]*=SC;K[1]*=SC;Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc).astype(np.float32);ctx=dr.RasterizeCudaContext();Kt=torch.tensor(K[:2,:2],device=DEV);ct=torch.tensor(K[:2,2],device=DEV);cwt=torch.tensor(cw,device=DEV)
def evalT(T):
 ious=np.empty(F)
 with torch.no_grad():
  for s in range(0,F,24):
   e=min(F,s+24);B=e-s;Tw=torch.tensor(T[s:e],dtype=torch.float32,device=DEV);Tt=cwt[None]@Tw
   vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',Kt,vc[:,:,:2])/zc[:,:,None]+ct[None,None]
   ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2)
   out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));m=out[:,:,:,3]>0
   for j in range(B):
    pm=m[j].cpu().numpy();gt=GT[s+j];inter=np.logical_and(pm,gt).sum();ious[s+j]=inter/max(np.logical_or(pm,gt).sum(),1)
 return ious
def smooth(T,win …[+1342 chars]
```
<details><summary>output</summary>

```
opt 0.8592 0.8379 0.8648 0.8192 0.8
sg3 0.8592 0.8379 0.8648 0.8192 0.8
sg5 0.8598 0.8362 0.8655 0.8208 0.801
sg7 0.8604 0.8373 0.8661 0.8215 0.801
sg9 0.8606 0.837 0.8664 0.8226 0.801
sg13 0.861 0.8369 0.8667 0.8242 0.802
sg17 0.8611 0.8367 0.8668 0.8246 0.801
sg25 0.8602 0.8365 0.8657 0.8252 0.803
sg33 0.8581 0.8365 0.8632 0.8249 0.798
sg5_stat 0.8599 0.8376 0.8655 0.8213 0.801
sg9_stat 0.861 0.839 0.8664 0.8253 0.801
sg13_stat 0.8616 0.839 0.8667 0.8304 0.802
sg17_stat 0.8618 0.8412 0.8668 0.8279 0.801
sg25_stat 0.8608 0.8399 0.8657 0.8294 0.803

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
T0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);T=np.load('/work/scratch/opt_raw.npz')['T']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;n=np.array([-0.021970802584377374,-0.9980358822147039,-0.05866567689642774]);d=1.3898362825184551
cam=json.load(open('/work/inputs/camera.json'));campos=np.array(cam['T_world_cam'])[:3,3]
def stats(T,a,b):
 mins=[];cent=[]
 for i in range(a,b):
  vv=T[i,:3,:3]@v.T+T[i,:3,3,None]; mins.append((vv.T@n+d).min());cent.append((T[i,:3,3]@n+d))
 print(a,b,'minplane',np.percentile(mins,[0,25,50,75,100]),'cent',np.percentile(cent,[0,50,100]))
stats(T0,0,100);stats(T,0,100);stats(T0,930,993);stats(T,930,993)
# ray shift compare baseline to opt
ray=(T[:,:3,3]-campos);ray0=(T0[:,:3,3]-campos);dist=np.sum((T[:,:3,3]-T0[:,:3,3])*ray0/np.linalg.norm(ray0,axis=1,keepdims=True),axis=1)
for a,b in [(0,100),(100,930),(930,993)]:print(a,b,'radial signed',np.percentile(dist[a:b],[0,10,25,50,75,90,100]))
PY"
```
<details><summary>output</summary>

```
0 100 minplane [-0.02852269 -0.01898295 -0.0175111  -0.01657311 -0.01236282] cent [0.63667479 0.64707335 0.65282008]
0 100 minplane [-0.0210224  -0.0139268  -0.01354483 -0.01317283 -0.00933125] cent [0.64372227 0.65092421 0.65511069]
930 993 minplane [-0.02724132 -0.02298128 -0.02072867 -0.01665931 -0.00329439] cent [0.63925997 0.64713566 0.66321545]
930 993 minplane [-0.03151973 -0.01465672 -0.01220715 -0.00816112 -0.00458693] cent [0.6366852  0.65626922 0.66141317]
0 100 radial signed [-0.08939498 -0.05641952 -0.03562781 -0.02154463 -0.01519477 -0.01190443
  0.01005014]
100 930 radial signed [-0.11214515 -0.06326384 -0.02666916  0.0137755   0.03678491  0.0633092
  0.17310431]
930 993 radial signed [-0.04307245 -0.03163503 -0.00887849  0.00242309  0.0159517   0.03531346
  0.06776868]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 for k in ['aligned/front_stereo_camera_left','alignment/front_stereo_camera_left/scale','raw/front_stereo_camera_left']:
  x=f[k];print(k,type(x),getattr(x,'shape',None),getattr(x,'dtype',None))
  if isinstance(x,h5py.Group):
   print(' keys',list(x.keys()))
   for kk,v in x.items():print(' ',kk,v.shape,v.dtype)
  break
 x=f['aligned/front_stereo_camera_left/0'];print(x.shape,x.dtype,np.nanpercentile(x,[0,1,10,50,90,99,100]))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (993,) object
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 407, in __getitem__
    return self._get(name)
           ^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 421, in _get
    oid = h5o.open(self.id, self._e(name), lapl=lapl)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5 …[+108 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 d=f['aligned/front_stereo_camera_left'];print(type(d[0]),d[0])
 x=f[d[0]];print(x,x.shape,x.dtype,np.nanpercentile(x,[0,1,10,50,90,99,100]))
 print('scale',f['alignment/front_stereo_camera_left/scale'][:5],f['alignment/front_stereo_camera_left/shift'][:5])
 print('rawref',f['raw/front_stereo_camera_left'][0],type(f[f['raw/front_stereo_camera_left'][0]]))
PY"
```
<details><summary>output</summary>

```
<class 'numpy.ndarray'> [137  80  78 ...  66  96 130]
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 407, in __getitem__
    return self._get(name)
           ^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 423, in _get
    raise TypeError("Accessing a group is done with bytes or str, "
TypeError: Accessing a group is done with bytes or str, not <class 'numpy.ndarray'>

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  d=f[k];print(k,d[0].shape,d[0].dtype,d[0].min(),d[0].max())
  print(np.percentile(d[0],[0,1,10,50,90,99,100]))
 print(f['alignment/front_stereo_camera_left/scale'][:10])
PY"
```
<details><summary>output</summary>

```
[  0.   2.  26. 128. 230. 253. 255.]
raw/front_stereo_camera_left (503829,) uint8 0 255
[  0.   1.  25. 128. 231. 253. 255.]
[1.1265537 1.1194471 1.123547  1.1253167 1.1241337 1.1281657 1.1241521
 1.1243519 1.1239895 1.1267309]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 x=f['aligned/front_stereo_camera_left'][0];print(x[:100]);print(x[-20:])
PY"
```
<details><summary>output</summary>

```
[137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0   0   0   4 128  16   0   0   0   0  76  58   5   5   0   1   0
   0  73  68  65  84 120 218 180 253 235 150  28 185 178  36   6 155  35
 178  50 235  66 238 189 207  92  36 141 142 164  25 105 206  72  75 210
  90 210 251 127 143  53  58 187 187  73  86 192 191  31 129  64 248  21
  64  86 239  33  87  55 201 170 172 188]
[240 145 116  40  11 249 203 159   0   0   0   0  73  69  78  68 174  66
  96 130]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  b=f[k][0].tobytes();im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED);print(k,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[0,1,10,50,90,99,100]))
  scale=f['alignment/front_stereo_camera_left/scale'][0];shift=f['alignment/front_stereo_camera_left/shift'][0]
  for form in [(im.astype(float)/1000),(im.astype(float)*scale+shift),(im.astype(float)/65535*scale+shift)]:
   print(' ',np.percentile(form,[1,10,50,90,99]))
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (1152, 1536) uint16 1806 8132 [1806. 1899. 2267. 3869. 5991. 7629. 8132.]
  [1.899 2.267 3.869 5.991 7.629]
  [2139.32539022 2553.89713514 4358.63608992 6749.18294513 8594.47783148]
  [0.03264401 0.03896997 0.06650852 0.10298593 0.13114333]
raw/front_stereo_camera_left (1152, 1536) uint16 1604 7219 [1604. 1686. 2013. 3435. 5318. 6772. 7219.]
  [1.686 2.013 3.435 5.318 6.772]
  [1899.36946177 2267.75250685 3869.71180379 5991.01233554 7629.02134943]
  [0.02898252 0.03460369 0.05904802 0.09141699 0.1164114 ]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/depth_eval.py <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh, time
import nvdiffrast.torch as dr
DEV='cuda';W,H=384,288;SC=.25;F=993
GT=np.load('/work/scratch/gt_masks_384.npy')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);K[0]*=SC;K[1]*=SC;Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc).astype(np.float32);ctx=dr.RasterizeCudaContext();Kt=torch.tensor(K[:2,:2],device=DEV);ct=torch.tensor(K[:2,2],device=DEV);cwt=torch.tensor(cw,device=DEV)
def evalT(T):
 med=np.full(F,np.nan);mae=np.full(F,np.nan);sq=np.full(F,np.nan)
 with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as hf, torch.no_grad():
  for s in range(0,F,24):
   e=min(F,s+24);B=e-s;Tw=torch.tensor(T[s:e],dtype=torch.float32,device=DEV);Tt=cwt[None]@Tw
   vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',Kt,vc[:,:,:2])/zc[:,:,None]+ct[None,None]
   ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2)
   out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));tri=out[:,:,:,3].long()-1;u=out[:,:,:,0];vbar=out[:,:,:,1]
   valid=tri>=0;tf=f[tri.clamp(min=0)];z0=torch.gather(zc,1,tf[:,:,:,0]);z1=torch.gather(zc,1,tf[:,:,:,1]);z2=torch.g …[+1285 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_eval.py", line 27, in <module>
    med,mae,sq=evalT(T);np.savez(f'/work/scratch/depth_{name}.npz',med=med,mae=mae,sq=sq)
               ^^^^^^^^
  File "/work/scratch/depth_eval.py", line 15, in evalT
    valid=tri>=0;tf=f[tri.clamp(min=0)];z0=torch.gather(zc,1,tf[:,:,:,0]);z1=torch.gather(zc,1,tf[:,:,:,1]);z2=torch.gather(zc,1,tf[:,:,:,2]);zd=z0*(1-u-vbar)+z1*u+z2*vbar
                                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: gather(): Expected dtype int64 for index

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/tf=f\\[tri.clamp(min=0)\\]/tf=f[tri.clamp(min=0)].long()/' /work/scratch/depth_eval.py && python /work/scratch/depth_eval.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_eval.py", line 27, in <module>
    med,mae,sq=evalT(T);np.savez(f'/work/scratch/depth_{name}.npz',med=med,mae=mae,sq=sq)
               ^^^^^^^^
  File "/work/scratch/depth_eval.py", line 15, in evalT
    valid=tri>=0;tf=f[tri.clamp(min=0)].long();z0=torch.gather(zc,1,tf[:,:,:,0]);z1=torch.gather(zc,1,tf[:,:,:,1]);z2=torch.gather(zc,1,tf[:,:,:,2]);zd=z0*(1-u-vbar)+z1*u+z2*vbar
                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Index tensor must have the same number of dimensions as input tensor

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/depth_eval.py';s=open(p).read();s=s.replace(\"tf=f[tri.clamp(min=0)].long();z0=torch.gather(zc,1,tf[:,:,:,0]);z1=torch.gather(zc,1,tf[:,:,:,1]);z2=torch.gather(zc,1,tf[:,:,:,2]);zd=z0*(1-u-vbar)+z1*u+z2*vbar\",\"tf=f[tri.clamp(min=0)].long().reshape(B,-1,3);zce=zc[:,None,:].expand(-1,H*W,-1);z0=torch.gather(zce,2,tf[:,:,0:1]).squeeze(2);z1=torch.gather(zce,2,tf[:,:,1:2]).squeeze(2);z2=torch.gather(zce,2,tf[:,:,2:3]).squeeze(2);zd=(z0*(1-u.reshape(B,-1)-vbar.reshape(B,-1))+z1*u.reshape(B,-1)+z2*vbar.reshape(B,-1)).reshape(B,H,W)\")
open(p,'w').write(s)
PY
python /work/scratch/depth_eval.py"
```
<details><summary>output</summary>

```
base med diff q [-0.32255924 -0.190277   -0.14573765 -0.1060245  -0.04480147  0.04143883
  0.2592566 ] mae [0.06218095 0.12814844 0.21937451] rmse [0.15259971 0.20263085 0.32831724]
  0 100 medabs 0.15852415561676025 rmse 0.28734906017780304
  100 930 medabs 0.10499215126037598 rmse 0.19916729629039764
  930 993 medabs 0.13755559921264648 rmse 0.17343519628047943
opt med diff q [-0.35925436 -0.20285969 -0.1679709  -0.11577582 -0.05315995  0.05497522
  0.30750012] mae [0.06700044 0.13768697 0.22965262] rmse [0.14962838 0.21022306 0.32162772]
  0 100 medabs 0.18251079320907593 rmse 0.3103056699037552
  100 930 medabs 0.11262392997741699 rmse 0.2058054506778717
  930 993 medabs 0.16381096839904785 rmse 0.19430096447467804

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];c=z['contact_prob_right_left'];rr=R.from_matrix(T[:,:3,:3]);vel=np.degrees((rr[:-1].inv()*rr[1:]).magnitude());td=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
for i in range(60,130): print(i,round(float(c[i,0]),2),round(float(c[i,1]),2),round(float(vel[min(i,991)]),2),round(float(td[min(i,991)]),3))
print('END')
for i in range(890,950): print(i,round(float(c[i,0]),2),round(float(c[i,1]),2),round(float(vel[min(i,991)]),2),round(float(td[min(i,991)]),3))
PY"
```
<details><summary>output</summary>

```
60 0.0 0.0 0.12 0.009
61 0.009
89 0.0 0.0 0.28 0.007
90 0.0 0.0 0.14 0.01
91 0.0 0.0 0.1 0.013
92 0.0 0.0 0.14 0.002
93 0.0 0.0 0.21 0.001
94 0.0 0.0 0.12 0.006
95 0.0 0.0 0.4 0.005
96 0.83 0.0 0.31 0.007
97 0.81 0.0 0.07 0.004
98 0.78 0.0 0.16 0.003
99 0.99 0.0 0.15 0.003
100 1.0 0.0 0.23 0.002
101 1.0 0.0 0.38 0.012
102 1.0 0.0 0.19 0.003
103 1.0 0.0 0.34 0.005
104 1.0 0.0 0.09 0.003
105 1.0 0.13 0.33 0.014
106 1.0 0.9 0.2 0.005
107 1.0 1.0 0.4 0.013
108 1.0 1.0 0.2 0.004
109 1.0 1.0 0.32 0.004
110 1.0 1.0 0.22 0.007
111 1.0 1.0 0.39 0.002
112 1.0 1.0 0.37 0.005
113 1.0 1.0 0.71 0.002
114 1.0 1.0 0.21 0.004
115 1.0 1.0 0.19 0.002
116 1.0 1.0 0.08 0.003
117 1.0 1.0 0.27 0.002
118 1.0 1.0 0.42 0.003
119 1.0 1.0 0.3 0.002
120 1.0 1.0 0.34 0.009
121 1.0 1.0 0.26 0.01
122 1.0 1.0 0.13 0.006
1 …[+860 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];c=z['contact_prob_right_left'];rr=R.from_matrix(T[:,:3,:3]);
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in list(range(85,106))+list(range(920,951)):
  m=f['object'][i];ys,xs=np.where(m);ang=np.degrees((rr[i-1].inv()*rr[i]).magnitude()) if i else 0
  print(i,'con',round(float(c[i,0]),2),round(float(c[i,1]),2),'area',m.sum(),'cent',round(float(xs.mean()),1),round(float(ys.mean()),1),'p',np.round(T[i,:3,3],3),'step',round(ang,2),round(float(np.linalg.norm(T[i,:3,3]-T[i-1,:3,3])),3))
PY"
```
<details><summary>output</summary>

```
85 con 0.0 0.0 area 11989 cent 622.9 720.6 p [0.596 0.578 2.619] step 0.61 0.023
86 con 0.0 0.0 area 12000 cent 623.0 720.6 p [0.598 0.581 2.61 ] step 0.04 0.01
87 con 0.0 0.0 area 12531 cent 622.9 721.8 p [0.599 0.584 2.607] step 0.09 0.004
88 con 0.0 0.0 area 12456 cent 622.8 721.7 p [0.598 0.585 2.601] step 0.35 0.006
89 con 0.0 0.0 area 12443 cent 622.8 721.6 p [0.599 0.586 2.603] step 0.41 0.003
90 con 0.0 0.0 area 12450 cent 622.7 721.7 p [0.601 0.588 2.597] step 0.28 0.007
91 con 0.0 0.0 area 12427 cent 622.8 721.2 p [0.602 0.585 2.606] step 0.14 0.01
92 con 0.0 0.0 area 12403 cent 622.6 721.2 p [0.602 0.589 2.594] step 0.1 0.013
93 con 0.0 0.0 area 12213 cent 621.9 720.6 p [0.601 0.588 2.593] step 0.14 0.002
94 con 0.0 0.0 area 11954 cent 622.6 720.0 p [0.602 0.588 2.592] step 0.21 …[+3547 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/make_candidate.py <<'PY'
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
F=993;T=np.load('/work/scratch/opt_raw.npz')['T'].copy()
for name,a,b in [('sg','all',None)]:pass
q=Rot.from_matrix(T[:,:3,:3]).as_quat()
for i in range(1,F):
 if np.dot(q[i],q[i-1])<0:q[i]*=-1
q=savgol_filter(q,17,2,axis=0);q/=np.linalg.norm(q,axis=1,keepdims=True);p=savgol_filter(T[:,:3,3],17,2,axis=0)
T[:,:3,:3]=Rot.from_quat(q).as_matrix();T[:,:3,3]=p
# Pooled static poses: phase boundaries from contact: object untouched through 95; released from 932 on.
for a,b in [(0,96),(932,993)]:
 qq=Rot.from_matrix(T[a:b,:3,:3]).as_quat()
 for i in range(1,len(qq)):
  if np.dot(qq[i],qq[i-1])<0:qq[i]*=-1
 qm=np.mean(qq,axis=0);qm/=np.linalg.norm(qm);pm=np.median(T[a:b,:3,3],axis=0)
 T[a:b,:3,:3]=Rot.from_quat(qm).as_matrix();T[a:b,:3,3]=pm
np.savez('/work/scratch/candidate.npz',T=T)
PY
python /work/scratch/make_candidate.py
python - <<'PY'
# evaluate candidate with helper without executing bottom: quick import-like
exec(open('/work/scratch/eval_variants.py').read().split('Topt=')[0])
T=np.load('/work/scratch/candidate.npz')['T'];q=evalT(T);print(q.mean(),q[:96].mean(),q[96:932].mean(),q[932:].mean(),np.percentile(q,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
0.8616939357292185 0.8417722312549961 0.8665351121073153 0.8266982009330929 [0.60795455 0.80127822 0.8596094  0.93303455 0.96306186]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();vis=m.visual
print(type(vis),vis.kind)
print('uv',vis.uv.shape,vis.uv.min(0),vis.uv.max(0))
print('mat',type(vis.material));print(vars(vis.material).keys());
for k,v in vars(vis.material).items(): print(k,type(v),getattr(v,'shape',None),v if isinstance(v,(str,int,float,tuple)) else '')
print('image',getattr(vis.material,'image',None))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> texture
uv (240494, 2) [0.000699   0.00099498] [0.99930102 0.99900502]
mat <class 'trimesh.visual.material.PBRMaterial'>
dict_keys(['_data'])
_data <class 'dict'> None 
image None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.visual
print(v.material._data)
print(v.material.baseColorTexture,v.material.baseColorFactor,v.material.metallicFactor,v.material.roughnessFactor)
PY"
```
<details><summary>output</summary>

```
{'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=5007x3516 at 0x74351737FE90>, 'doubleSided': True, 'name': '_texture.001'}
<PIL.Image.Image image mode=RGB size=5007x3516 at 0x74351737FE90> None 0.0 None

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
z=np.load('/work/inputs/reconstruction.npz');kp=z['human_keypoints_world'];cp=z['contact_prob_right_left'];
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
# subsample maybe all
for name in ['base','cand']:
 T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'] if name=='base' else np.load('/work/scratch/candidate.npz')['T']
 d=np.full((993,2),np.nan)
 for i in range(993):
  vv=T[i,:3,:3]@v.T+T[i,:3,3,None]; tree=cKDTree(vv.T)
  for h,a,b in [(0,21,42),(1,42,63)]: d[i,h]=tree.query(kp[i,a:b])[0].min()
 active=cp>.5
 for h in range(2):
  idx=active[:,h]
  print(name,'hand',h,'active',idx.sum(),'dist q',np.percentile(d[idx,h],[0,10,25,50,75,90,100]))
 print('all active mean',np.mean(np.where(active,d,np.nan)))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
z=np.load('/work/inputs/reconstruction.npz');kp=z['human_keypoints_world'];cp=z['contact_prob_right_left'];
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::8]
for name in ['base','cand']:
 T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'] if name=='base' else np.load('/work/scratch/candidate.npz')['T']
 d=np.full((993,2),np.nan)
 for i in range(993):
  vv=T[i,:3,:3]@v.T+T[i,:3,3,None]; tree=cKDTree(vv.T)
  for h,a,b in [(0,21,42),(1,42,63)]: d[i,h]=tree.query(kp[i,a:b])[0].min()
 for h in range(2):
  idx=cp[:,h]>.5
  print(name,'hand',h,'n',idx.sum(),'q',np.round(np.percentile(d[idx,h],[0,10,25,50,75,90,100]),3))
PY"
```
<details><summary>output</summary>

```
base hand 0 n 789 q [0.    0.002 0.003 0.004 0.011 0.056 0.268]
base hand 1 n 758 q [0.    0.002 0.003 0.005 0.012 0.04  0.264]
cand hand 0 n 789 q [0.    0.002 0.003 0.005 0.014 0.037 0.27 ]
cand hand 1 n 758 q [0.    0.002 0.003 0.006 0.013 0.031 0.23 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/flip_eval.py <<'PY'
exec(open('/work/scratch/eval_variants.py').read().split('Topt=')[0])
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/scratch/candidate.npz')['T'];R0=Rot.from_matrix(T[:,:3,:3])
axes={'localx':np.array([1.,0,0]),'localy':np.array([0,1.,0]),'localz':np.array([0,0,1.]),'worldx':np.array([1.,0,0]),'worldy':np.array([0,1.,0]),'worldz':np.array([0,0,1.])}
base=evalT(T);print('base',base.mean(),base[:96].mean(),base[96:932].mean(),base[932:].mean())
for n,a in axes.items():
 C=T.copy();dev=Rot.from_rotvec(np.pi*a).as_matrix()
 if n.startswith('local'):C[:,:3,:3]=T[:,:3,:3]@dev
 else:C[:,:3,:3]=dev@T[:,:3,:3]
 q=evalT(C);print(n,round(q.mean(),4),round(q[:96].mean(),4),round(q[96:932].mean(),4),round(q[932:].mean(),4),'better>0.05',int((q>base+.05).sum()),'worse',int((q<base-.05).sum()),'maxrel',round(float((q-base).max()),3))
PY
python /work/scratch/flip_eval.py"
```
<details><summary>output</summary>

```
base 0.8616939357292185 0.8417722312549961 0.8665351121073153 0.8266982009330929
localx 0.2903 0.0569 0.332 0.0859 better>0.05 0 worse 993 maxrel -0.079
localy 0.2905 0.0603 0.3312 0.0946 better>0.05 0 worse 993 maxrel -0.085
localz 0.6796 0.5514 0.7003 0.5984 better>0.05 0 worse 957 maxrel -0.01
worldx 0.3329 0.0711 0.3801 0.0972 better>0.05 0 worse 993 maxrel -0.098
worldy 0.4585 0.4802 0.4526 0.5046 better>0.05 0 worse 993 maxrel -0.138
worldz 0.2904 0.0623 0.3293 0.117 better>0.05 0 worse 993 maxrel -0.136

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');fp=R.from_matrix(z['T_cam_mesh_foundationpose'][:,:3,:3]);Tb=z['T_world_mesh'];Tc=np.load('/work/scratch/candidate.npz')['T'];
for name,T in [('base',Tb),('cand',Tc)]:
 rb=R.from_matrix(T[:,:3,:3]); ang=np.degrees((rb.inv()*fp).magnitude())
 print(name,np.round(np.percentile(ang,[0,10,25,50,75,90,95,99,100]),2), 'mean',ang.mean())
 for a,b in [(0,96),(96,932),(932,993)]:print(a,b,np.round(np.percentile(ang[a:b],[25,50,75]),2))
PY"
```
<details><summary>output</summary>

```
base [167.52 174.   176.2  177.2  177.84 178.49 178.89 179.67 179.97] mean 176.725531323788
0 96 [177.15 177.31 177.5 ]
96 932 [175.8  177.04 177.96]
932 993 [177.04 177.52 177.71]
cand [151.17 172.94 175.58 177.24 178.03 178.72 179.26 179.86 179.98] mean 176.43132755481307
0 96 [177.78 177.92 178.01]
96 932 [174.87 176.88 177.94]
932 993 [177.72 177.94 178.16]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');Wc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);fp=R.from_matrix(np.einsum('ij,njk->nik',Wc[:3,:3],z['T_cam_mesh_foundationpose'][:,:3,:3]));Tb=z['T_world_mesh'];Tc=np.load('/work/scratch/candidate.npz')['T'];
for name,T in [('base',Tb),('cand',Tc)]:
 rb=R.from_matrix(T[:,:3,:3]); ang=np.degrees((rb.inv()*fp).magnitude())
 print(name,np.round(np.percentile(ang,[0,10,25,50,75,90,95,99,100]),2), 'mean',round(ang.mean(),2))
PY"
```
<details><summary>output</summary>

```
base [ 0.66  2.    2.48  3.72  5.34  7.44  9.42 12.83 24.51] mean 4.36
cand [ 0.66  2.09  2.5   3.92  5.79  8.84 12.33 19.27 31.12] mean 4.86

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/pose_opt.py';s=open(p).read()
# replace fixed mode with env/mode
s=s.replace(\"import numpy as np, torch, json, cv2, os, time\",\"import numpy as np, torch, json, cv2, os, time, sys\")
s=s.replace(\"# camera center\\ncampos=Wc[:3,3]\\nfor it in range(4):\",\"# camera center\\ncampos=Wc[:3,3]\\nmode=sys.argv[1] if len(sys.argv)>1 else 'all'\\nfor it in range(4):\")
s=s.replace(\"for axis,name in [(0,'tx'),(1,'ty')]:\",\"for axis,name in ([(0,'tx'),(1,'ty')] if mode in ('all','trans','xy') else []):\")
s=s.replace(\"# depth along ray\\n best=base.copy();bestT=T.copy();vals=[]\\n for a in ray_m:\",\"# depth along ray\\n best=base.copy();bestT=T.copy();vals=[]\\n for a in (ray_m if mode in ('all','trans','ray') else []):\")
s=s.replace(\"for ax,name in [('cx','rx'),('cy','ry'),('cz','rz'),('wy','yaw')]:\",\"for ax,name in ([('cx','rx'),('cy','ry'),('cz','rz'),('wy','yaw')] if mode in ('all','rot') else []):\")
s=s.replace(\"np.savez('/work/scratch/opt_raw.npz',T=T,ious=base)\",\"np.savez(f'/work/scratch/opt_{mode}.npz',T=T,ious=base)\")
open('/work/scratch/pose_opt_modes.py','w').write(s)
PY
python /work/scratch/pose_opt_modes.py xy"
```
<details><summary>output</summary>

```
iter 0 base 0.8153559547275979
tx [0.7472 0.789  0.8154 0.8053 0.7668] -> 0.8256
ty [0.7702 0.8175 0.8256 0.7812 0.7222] -> 0.8393
ray [] -> 0.8393
iter 1 base 0.8392842991771239
tx [0.7702 0.8171 0.8393 0.8171 0.7704] -> 0.8414
ty [0.7493 0.81   0.8414 0.8092 0.7512] -> 0.842
ray [] -> 0.842
iter 2 base 0.8419746041797675
tx [0.7693 0.8179 0.842  0.8201 0.7736] -> 0.8423
ty [0.7502 0.8109 0.8423 0.8096 0.7512] -> 0.8423
ray [] -> 0.8423
iter 3 base 0.8423169946591861
tx [0.7686 0.8177 0.8423 0.8206 0.7743] -> 0.8423
ty [0.7503 0.811  0.8423 0.8096 0.7511] -> 0.8423
ray [] -> 0.8423
FINAL 0.842327714298366 0.8298993243122331 0.8457282547634383 0.8172545462444482 [0.54488518 0.77260896 0.84363178 0.91466808 0.94837172]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/pose_opt_modes.py trans'
```
<details><summary>output</summary>

```
iter 0 base 0.8153559547275979
tx [0.7472 0.789  0.8154 0.8053 0.7668] -> 0.8256
ty [0.7702 0.8175 0.8256 0.7812 0.7222] -> 0.8393
ray [0.8007 0.8273 0.8359 0.8393 0.8381 0.8335 0.8107] -> 0.8492
iter 1 base 0.8491712185007058
tx [0.7708 0.8209 0.8492 0.8264 0.7777] -> 0.8507
ty [0.7546 0.8168 0.8507 0.8113 0.7517] -> 0.8518
ray [0.8062 0.8354 0.8459 0.8518 0.8477 0.8402 0.8151] -> 0.8528
iter 2 base 0.8528012106669152
tx [0.773  0.824  0.8528 0.8282 0.7787] -> 0.8529
ty [0.7557 0.8182 0.8529 0.8123 0.7519] -> 0.853
ray [0.8081 0.8371 0.8472 0.853  0.8478 0.8399 0.8142] -> 0.8531
iter 3 base 0.8531237698602364
tx [0.7729 0.8241 0.8531 0.8286 0.7789] -> 0.8531
ty [0.7559 0.8185 0.8531 0.8123 0.7518] -> 0.8531
ray [0.8083 0.8373 0.8473 0.8531 0.8478 0.8398 0.8141] -> 0.8531
FINAL 0.853132328 …[+120 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/pose_opt_modes.py rot'
```
<details><summary>output</summary>

```
iter 0 base 0.8153559547275979
ray [] -> 0.8154
rx [0.6957 0.7739 0.8038 0.8154 0.8028 0.7718 0.6998] -> 0.8204
ry [0.7642 0.7978 0.8114 0.8204 0.8196 0.8108 0.7754] -> 0.8273
rz [0.6525 0.757  0.8086 0.8273 0.7925 0.7379 0.6408] -> 0.8285
yaw [0.7632 0.8049 0.8207 0.8285 0.8199 0.8023 0.7578] -> 0.8295
iter 1 base 0.8295090269925353
ray [] -> 0.8295
rx [0.693  0.7767 0.8112 0.8295 0.8144 0.7816 0.7049] -> 0.8304
ry [0.7691 0.8073 0.8218 0.8304 0.8229 0.8079 0.7695] -> 0.8309
rz [0.648  0.7519 0.8059 0.8309 0.7985 0.7444 0.6451] -> 0.8309
yaw [0.7637 0.8065 0.8224 0.8309 0.8211 0.8027 0.7579] -> 0.8311
iter 2 base 0.8310564137672446
ray [] -> 0.8311
rx [0.6905 0.7747 0.8108 0.8311 0.8168 0.7845 0.7077] -> 0.8312
ry [0.7704 0.8082 0.8228 0.8312 0.8229 0.8078 0.7686] -> 0.8313
rz [0.6481 0.7 …[+558 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_tex.py <<'PY'
import numpy as np, torch, json, cv2, trimesh, h5py
import nvdiffrast.torch as dr
from PIL import Image
DEV='cuda';W,H=768,576;SC=.5
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();img=np.array(m.visual.material.baseColorTexture);tex=torch.tensor(img,dtype=torch.float32,device=DEV).permute(2,0,1)[None]/255
uv=torch.tensor(m.visual.uv,dtype=torch.float32,device=DEV)
v=torch.tensor(m.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(m.faces,dtype=torch.int32,device=DEV)
# sample both variants
samples=[]
for flip in [False,True]:
 u=uv.clone();u[:,0]=u[:,0]*2-1
 u[:,1]=(1-u[:,1] if flip else u[:,1])*2-1
 col=torch.nn.functional.grid_sample(tex,u[None,None],mode='bilinear',align_corners=False)[0,:,0].T
 samples.append(col)
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);K[0]*=SC;K[1]*=SC;cw=np.linalg.inv(np.array(cam['T_world_cam'])).astype(np.float32);ctx=dr.RasterizeCudaContext();cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,300,700,950]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.resize(im,(W,H));outs=[]
 for col in samples:
  Tw=torch.tensor(T[i],dtype=torch.float32,device=DEV);Tt=torch.tensor(cw,device=DEV)@Tw;vc=Tt[:3,:3]@v.T+Tt[:3,3,None];zc=vc[2];uvp=torch.tensor(K[:2,:2],device=DEV)@vc[:2]/zc+torch.tensor(K[:2,2],device=DEV)[:,None]
  ndc=to …[+501 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_tex.py", line 20, in <module>
    ndc=torch.stack([uvp[0]/W*2-1,uvp[1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],1);rast,_=dr.rasterize(ctx,ndc[None],f,resolution=(H,W));rgb,_=dr.interpolate(col[None],rast,f);mask=rast[0,:,:,3]>0;c=(rgb[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
                                                                                                                                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/a …[+458 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/rgb,_=dr.interpolate(col\\[None\\],rast,f)/rgb,_=dr.interpolate(col[None].contiguous(),rast.contiguous(),f.contiguous())/\" /work/scratch/render_tex.py && python /work/scratch/render_tex.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_tex.py", line 21, in <module>
    ov=im.copy();ov[mask]=(.3*ov[mask]+.7*c[mask]).astype(np.uint8);cv2.putText(ov,f'{i}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2);outs.append(ov)
                              ~~^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/mask=rast\\[0,:,:,3\\]>0;c=/mask=(rast[0,:,:,3]>0).cpu().numpy();c=/\" /work/scratch/render_tex.py && python /work/scratch/render_tex.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/photo_eval.py <<'PY'
import numpy as np, torch, json, cv2, trimesh, time
import nvdiffrast.torch as dr
DEV='cuda';W,H=384,288;SC=.25;F=993
GT=np.load('/work/scratch/gt_masks_384.npy')
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(m.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(m.faces,dtype=torch.int32,device=DEV)
tex=np.array(m.visual.material.baseColorTexture);tex=torch.tensor(tex,dtype=torch.float32,device=DEV).permute(2,0,1)[None]/255
uv=torch.tensor(m.visual.uv,dtype=torch.float32,device=DEV)
def getcol(flip):
 u=uv.clone();u[:,0]=u[:,0]*2-1;u[:,1]=(1-u[:,1] if flip else u[:,1])*2-1
 return torch.nn.functional.grid_sample(tex,u[None,None],mode='bilinear',align_corners=False)[0,:,0].T.contiguous()
cols=[getcol(False),getcol(True)]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);K[0]*=SC;K[1]*=SC;Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc).astype(np.float32);ctx=dr.RasterizeCudaContext();Kt=torch.tensor(K[:2,:2],device=DEV);ct=torch.tensor(K[:2,2],device=DEV);cwt=torch.tensor(cw,device=DEV)
def load_video():
 cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
 while True:
  ok,im=cap.read()
  if not ok:break
  ims.append(cv2.resize(im,(W,H),interpolation=cv2.INTER_AREA))
 return np.stack(ims)
IMS=load_video();print('video',IMS.shape)
def ncc(a,b,m):
 a=a[m];b=b[m]
 if len(a)<30:return np.nan
 a=a-a.mean();b=b-b.mean();r …[+1681 chars]
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
0 base ncc [-0.34381526 -0.277807   -0.04527559  0.30898033  0.37779028] grad [-0.14247252 -0.04658804  0.09139235  0.29173898  0.34306745]
0 cand ncc [-0.36614047 -0.31052329 -0.03253579  0.30648581  0.38751342] grad [-0.11783506 -0.03034116  0.18205601  0.30962301  0.3353285 ]
1 base ncc [0.07915754 0.14774113 0.227704   0.33128137 0.45349918] grad [-0.17958456 -0.01314179  0.28207246  0.54112594  0.60295393]
1 cand ncc [0.09548452 0.14027372 0.2235946  0.3478357  0.42266684] grad [-0.08937849  0.02884012  0.44454381  0.53465529  0.55681737]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
for flip in [1]:
 for name in ['base','cand']:
  z=np.load(f'/work/scratch/photo_{flip}_{name}.npz');
  for k in ['ncc','grad']:
   a=z[k]
   print(name,k,'all',np.nanmean(a),'0-96',np.nanmean(a[:96]),'96-932',np.nanmean(a[96:932]),'932+',np.nanmean(a[932:]),'nan',np.isnan(a).sum())
PY"
```
<details><summary>output</summary>

```
base ncc all 0.24085529786586235 0-96 0.4000067674525796 96-932 0.22228236938524834 932+ 0.24492787375878813 nan 0
base grad all 0.25191405350639673 0-96 0.574900963121057 96-932 0.2024541598469701 932+ 0.4214505744289097 nan 0
cand ncc all 0.24204013585917722 0-96 0.40059596218774546 96-932 0.22403608155317326 932+ 0.23925374360141932 nan 0
cand grad all 0.30670449099336555 0-96 0.5407122980402842 96-932 0.2653136096362765 932+ 0.5056885457150432 nan 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/photo_1_base.npz');c=np.load('/work/scratch/photo_1_cand.npz');qi={n:np.load(f'/work/scratch/ious_{n}.npy') for n in ['base','sg17']}
# candidate q computed not saved; compute load eval? Use candidate from earlier run not
print('grad delta mean by base grad quartile')
d=c['grad']-b['grad']
for lo,hi in [(-1,.05),(.05,.2),(.2,.4),(.4,.6),(.6,1)]:
 m=(b['grad']>=lo)&(b['grad']<hi);print(lo,hi,m.sum(),np.nanmean(d[m]),np.nanmean(b['grad'][m]),np.nanmean(c['grad'][m]))
print('ncc delta',np.nanmean(c['ncc']-b['ncc']))
PY"
```
<details><summary>output</summary>

```
grad delta mean by base grad quartile
-1 0.05 296 0.10571673272393188 -0.14377971069705645 -0.038062977973124565
0.05 0.2 81 0.1712006541816621 0.13259468861365098 0.30379534279531306
0.2 0.4 216 0.09619693381713669 0.289344679730964 0.3855416135481007
0.4 0.6 295 -0.0030634674834729993 0.5211862008924437 0.5181227334089707
0.6 1 105 -0.10121253568871812 0.625913329860151 0.5247007941714328
ncc delta 0.0011848379933148629

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/photo_eval.py';s=open(p).read();s=s.replace(\"def evalT(T,flip=0):\\n nccs=np.full(F,np.nan);grads=np.full(F,np.nan)\",\"def evalT(T,flip=0):\\n nccs=np.full(F,np.nan);grads=np.full(F,np.nan);ious=np.full(F,np.nan)\")
s=s.replace(\"mask=GT[i]&pm[j]\\n    nccs[i]\",\"mask=GT[i]&pm[j]\\n    ious[i]=np.logical_and(GT[i],pm[j]).sum()/max(np.logical_or(GT[i],pm[j]).sum(),1)\\n    nccs[i]\")
s=s.replace(\"return nccs,grads\",\"return ious,nccs,grads\")
s=s.replace(\"a,b=evalT(T,flip)\",\"a,bb,b=evalT(T,flip)\")
s=s.replace(\"np.savez(f'/work/scratch/photo_{flip}_{name}.npz',ncc=a,grad=b)\",\"np.savez(f'/work/scratch/photo_{flip}_{name}.npz',ncc=bb,grad=b,iou=a)\")
s=s.replace(\"print(flip,name,'ncc',np.nanpercentile(a,[10,25,50,75,90]),'grad',np.nanpercentile(b,[10,25,50,75,90]),flush=True)\",\"print(flip,name,'iou',np.nanmean(a),'ncc',np.nanpercentile(bb,[10,25,50,75,90]),'grad',np.nanpercentile(b,[10,25,50,75,90]),flush=True)\")
open(p,'w').write(s)
PY
python /work/scratch/photo_eval.py"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
0 base iou 0.8131712106938656 ncc [-0.34381526 -0.277807   -0.04527559  0.30898033  0.37779028] grad [-0.14247252 -0.04658804  0.09139235  0.29173898  0.34306745]
0 cand iou 0.8616939357292185 ncc [-0.36614047 -0.31052329 -0.03253579  0.30648581  0.38751342] grad [-0.11783506 -0.03034116  0.18205601  0.30962301  0.3353285 ]
1 base iou 0.8131712106938656 ncc [0.07915754 0.14774113 0.227704   0.33128137 0.45349918] grad [-0.17958456 -0.01314179  0.28207246  0.54112594  0.60295393]
1 cand iou 0.8616939357292185 ncc [0.09548452 0.14027372 0.2235946  0.3478357  0.42266684] grad [-0.08937849  0.02884012  0.44454381  0.53465529  0.55681737]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_temporal.py <<'PY'
import sys, numpy as np, time
sys.path.insert(0,'/work/scratch')
import photo_eval as pe
from scipy.spatial.transform import Rotation as Rot
F=993; T=np.load('/work/scratch/candidate.npz')['T'].copy(); T0=T.copy()
static=np.ones(F,bool);static[96:932]=False
# camera parameters
import json
cam=json.load(open('/work/inputs/camera.json'));Kfull=np.array(cam['K']);Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc);Rwc=Wc[:3,:3];campos=Wc[:3,3];SC=.25
def q_of(X):return Rot.from_matrix(X[:,:3,:3]).as_quat()
def scores(X):
 iou,ncc,grad=pe.evalT(X,1)
 data=iou+0.15*np.clip(grad,-.2,1.)*iou+0.03*np.clip(ncc,-.5,1.)*iou
 q=q_of(X)
 for i in range(1,F):
  if np.dot(q[i],q[i-1])<0:q[i]*=-1
 p=X[:,:3,3]
 # compare candidate pose to smooth neighbor center; d0 for first/last
 dr=np.zeros(F);dt=np.zeros(F)
 for i in range(1,F-1):
  qm=q[i-1]+q[i+1];qm/=np.linalg.norm(qm)+1e-12
  if np.dot(q[i],qm)<0: qm=-qm
  dr[i]=np.degrees((Rot.from_quat(q[i]).inv()*Rot.from_quat(qm)).magnitude())
  dt[i]=np.linalg.norm(p[i]-.5*(p[i-1]+p[i+1]))
 pen=2.0*dt**2+0.0015*dr**2
 score=data-pen
 score[static]=-1e9
 return score,iou,ncc,grad,data,pen
score,*_=scores(T);print('start',np.nanmean(score[~static]),flush=True)
trans_px=[-12,-6,0,6,12];ray_m=[-.03,-.015,0,.015,.03];rot_deg=[-2,-1,0,1,2]
for it in range(3):
 base=score.copy();print('iter',it,flush=True)
 for axis,name in [(0,'tx'),(1,'ty')]:
  best=base.copy();bestT=T.copy();vals=[]
  fo …[+1631 chars]
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
start 0.9067824028215813
iter 0
tx [0.8864 0.9033 0.9068 0.8927 0.8679] 0.9092679074709342
ty [0.881  0.9041 0.9093 0.8877 0.856 ] 0.9119559407969604
ray [0.9013 0.9083 0.9119 0.9076 0.9002] 0.9141264168164439
rx [0.8798 0.9026 0.9138 0.8962 0.8687] 0.9146141945860655
ry [0.8909 0.9049 0.914  0.9072 0.8951] 0.9155980913130285
rz [0.8553 0.8931 0.9144 0.8934 0.8552] 0.9158464409593681
yaw [0.8892 0.9045 0.9145 0.9053 0.8913] 0.9160187692431813
iter 1
tx [0.8805 0.9013 0.9146 0.9015 0.8803] 0.9151871398123547
ty [0.8713 0.8977 0.915  0.8968 0.8665] 0.9153144694847822
ray [0.9017 0.9096 0.9151 0.9089 0.9008] 0.9157236092032237
rx [0.8781 0.9016 0.9154 0.8972 0.8695] 0.9158767022753594
ry [0.8906 0.9052 0.9154 0.9069 0.8943] 0.916060619986229
rz [0.8555 0.8935 0.9154 0 …[+868 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/temporal_refined.npz');T=z['T'];T0=np.load('/work/scratch/candidate.npz')['T']
for name,X in [('cand',T0),('ref',T)]:
 r=R.from_matrix(X[:,:3,:3]);vel=np.degrees((r[:-1].inv()*r[1:]).magnitude());td=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1)
 acc=np.degrees((r[:-2].inv()*r[1:-1]).inv()*(r[1:-1].inv()*r[2:])).magnitude()
 # actually acceleration distance not correct; rel vec differences
 rv=(r[:-1].inv()*r[1:]).as_rotvec();ra=np.linalg.norm(np.diff(rv,axis=0),axis=1);ta=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1)
 print(name,'rotvel',np.percentile(vel,[50,75,90,95,99,100]),'transvel',np.percentile(td,[50,75,90,95,99,100]),'rotacc',np.percentile(np.degrees(ra),[50,75,90,95,99,100]),'transacc',np.percentile(ta,[50,75,90,95,99,100]))
PY"
```
<details><summary>output</summary>

```
AttributeError: 'scipy.spatial.transform._rotation.Rotation' object has no attribute 'degrees'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
TypeError: loop of ufunc does not support argument 0 of type scipy.spatial.transform._rotation.Rotation which has no callable degrees method

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/temporal_refined.npz');T=z['T'];T0=np.load('/work/scratch/candidate.npz')['T']
for name,X in [('cand',T0),('ref',T)]:
 r=R.from_matrix(X[:,:3,:3]);vel=np.degrees((r[:-1].inv()*r[1:]).magnitude());td=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1)
 rv=(r[:-1].inv()*r[1:]).as_rotvec();ra=np.linalg.norm(np.diff(rv,axis=0),axis=1);ta=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1)
 print(name,'rotvel',np.round(np.percentile(vel,[50,75,90,95,99,100]),2),'transvel',np.round(np.percentile(td,[50,75,90,95,99,100]),3),'rotacc',np.round(np.percentile(np.degrees(ra),[50,75,90,95,99,100]),2),'transacc',np.round(np.percentile(ta,[50,75,90,95,99,100]),3))
PY"
```
<details><summary>output</summary>

```
cand rotvel [0.46 0.89 1.33 1.78 3.91 8.13] transvel [0.004 0.007 0.01  0.011 0.014 0.018] rotacc [0.15 0.27 0.47 0.77 1.95 3.28] transacc [0.004 0.007 0.01  0.011 0.014 0.018]
ref rotvel [ 1.    1.63  2.35  2.95  5.11 13.58] transvel [0.011 0.018 0.028 0.033 0.046 0.071] rotacc [ 1.16  2.29  3.65  4.62  6.99 11.23] transacc [0.011 0.018 0.028 0.033 0.046 0.071]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_refined_smooth.py <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');import photo_eval as pe
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
F=993;T0=np.load('/work/scratch/temporal_refined.npz')['T']
def smooth(T,win,static=True):
 T=T.copy();q=Rot.from_matrix(T[:,:3,:3]).as_quat()
 for i in range(1,F):
  if np.dot(q[i],q[i-1])<0:q[i]*=-1
 q=savgol_filter(q,win,2,axis=0);q/=np.linalg.norm(q,axis=1,keepdims=True);p=savgol_filter(T[:,:3,3],win,2,axis=0)
 T[:,:3,:3]=Rot.from_quat(q).as_matrix();T[:,:3,3]=p
 if static:
  for a,b in [(0,96),(932,993)]:
   qq=Rot.from_matrix(T[a:b,:3,:3]).as_quat()
   for i in range(1,len(qq)):
    if np.dot(qq[i],qq[i-1])<0:qq[i]*=-1
   qm=np.mean(qq,0);qm/=np.linalg.norm(qm);pm=np.median(T[a:b,:3,3],0);T[a:b,:3,:3]=Rot.from_quat(qm).as_matrix();T[a:b,:3,3]=pm
 return T
for win in [3,5,7,9,13,17,25]:
 T=smooth(T0,win);iou,ncc,grad=pe.evalT(T,1);data=iou+0.15*np.clip(grad,-.2,1)*iou+0.03*np.clip(ncc,-.5,1)*iou
 r=Rot.from_matrix(T[:,:3,:3]);rv=(r[:-1].inv()*r[1:]).as_rotvec();acc=np.degrees(np.linalg.norm(np.diff(rv,axis=0),axis=1))
 print(win,'iou',round(np.nanmean(iou),4),'ncc',round(np.nanmean(ncc),4),'grad',round(np.nanmean(grad),4),'data',round(np.nanmean(data),4),'rotacc95',round(np.percentile(acc,95),2),flush=True)
 np.savez(f'/work/scratch/refsm{win}.npz',T=T)
PY
python /work/scratch/eval_refined_smooth.py"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
3 iou 0.8637 ncc 0.2635 grad 0.3581 data 0.917 rotacc95 4.62
5 iou 0.8636 ncc 0.2538 grad 0.3497 data 0.9154 rotacc95 1.66
7 iou 0.8639 ncc 0.2536 grad 0.3499 data 0.9158 rotacc95 1.02
9 iou 0.8641 ncc 0.2524 grad 0.3495 data 0.9159 rotacc95 0.81
13 iou 0.864 ncc 0.252 grad 0.3489 data 0.9156 rotacc95 0.56
17 iou 0.8638 ncc 0.2514 grad 0.348 data 0.9153 rotacc95 0.42
25 iou 0.8625 ncc 0.2534 grad 0.346 data 0.9137 rotacc95 0.3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for f in ['candidate.npz','refsm13.npz']:
 T=np.load('/work/scratch/'+f)['T'];r=R.from_matrix(T[:,:3,:3]);rv=(r[:-1].inv()*r[1:]).as_rotvec();acc=np.degrees(np.linalg.norm(np.diff(rv,axis=0),axis=1));vel=np.degrees((r[:-1].inv()*r[1:]).magnitude());
 print(f,'acc q',np.percentile(acc,[50,90,95,99,100]),'vel',np.percentile(vel,[50,90,95,99,100]))
PY"
```
<details><summary>output</summary>

```
candidate.npz acc q [0.15248956 0.4653147  0.77483384 1.94962005 3.2846506 ] vel [0.4620305  1.33094721 1.78332882 3.90847979 8.13435164]
refsm13.npz acc q [0.15324155 0.40479416 0.55612214 1.22679545 2.08459667] vel [0.46136953 1.30573813 1.68392358 3.16741807 9.5219907 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/scratch/candidate.npz')['T'];r=R.from_matrix(z[:,:3,:3]);
for i in [0,50,95,932,950,992]:
 rv=np.degrees((r[0].inv()*r[i]).as_rotvec());print(i,np.round(rv,2),round(np.linalg.norm(rv),2))
# floor normal expressed axis angle
n=np.array([-0.021970802584377374,-0.9980358822147039,-0.05866567689642774]);print('n',n)
PY"
```
<details><summary>output</summary>

```
0 [0. 0. 0.] 0.0
50 [0. 0. 0.] 0.0
95 [0. 0. 0.] 0.0
932 [ -0.25   0.25 -38.5 ] 38.5
950 [ -0.25   0.25 -38.5 ] 38.5
992 [ -0.25   0.25 -38.5 ] 38.5
n [-0.0219708  -0.99803588 -0.05866568]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_opt.py <<'PY'
import sys, numpy as np, json, torch, cv2, time
sys.path.insert(0,'/work/scratch');import photo_eval as pe
from scipy.spatial.transform import Rotation as Rot
F=993;T=np.load('/work/scratch/refsm13.npz')['T'].copy();Torig=T.copy()
cam=json.load(open('/work/inputs/camera.json'));Kfull=np.array(cam['K']);Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc);Rwc=Wc[:3,:3];campos=Wc[:3,3];SC=.25
DEV='cuda';W,H=384,288
Kt=pe.Kt;ct=pe.ct;cwt=pe.cwt;ctx=pe.ctx;v=pe.v;f=pe.f;col=pe.cols[1].contiguous();GT=pe.GT;IMS=pe.IMS
def eval_phase(P,inds):
 B=len(inds);Tw=torch.tensor(P,dtype=torch.float32,device=DEV).expand(B,-1,-1);Tt=cwt[None]@Tw
 vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',Kt,vc[:,:,:2])/zc[:,:,None]+ct[None,None]
 ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2)
 rast,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));rgb,_=dr.interpolate(col[None].expand(B,-1,-1).contiguous(),rast.contiguous(),f.contiguous());pm=(rast[:,:,:,3]>0).cpu().numpy();rg=rgb.clamp(0,1).cpu().numpy()
 ious=[];nccs=[];grads=[];ms=[]
 def ncc(a,b,m):
  a=a[m];b=b[m]
  if len(a)<30:return np.nan
  a=a-a.mean();b=b-b.mean();return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-8))
 for j,i in enumerate(inds):
  obs=IMS[i].astype(np.float32)/255;ob=cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY);pr=cv2.cvtColor(rg[j],cv2.COLOR_RGB2GRAY);ob=cv2.GaussianBlu …[+2279 chars]
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
phase 0 96 start (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
x (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
y (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
ray (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
yaw (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
rx (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
ry (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
x (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
y (0.9507579007756274, 0.8417722312549961, 0.410513813070736, 0.5447299738340561)
ray (0.950757900 …[+3768 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/depth_eval.py';s=open(p).read();start=s.index(\"for name,path in [\")
s=s[:start]+\"for name,path in [('base','/work/inputs/reconstruction.npz'),('cand','/work/scratch/candidate.npz'),('refsm13','/work/scratch/refsm13.npz')]:\\n T=np.load(path)['T_world_mesh'] if name=='base' else np.load(path)['T']\\n med,mae,sq=evalT(T);np.savez(f'/work/scratch/depth_{name}.npz',med=med,mae=mae,sq=sq)\\n print(name,'med diff q',np.nanpercentile(med,[0,10,25,50,75,90,100]),'mae',np.nanpercentile(mae,[10,50,90]),'rmse',np.nanpercentile(sq,[10,50,90]),flush=True)\\n for a,b in [(0,96),(96,932),(932,993)]:print(' ',a,b,'medabs',np.nanmedian(np.abs(med[a:b])),'rmse',np.nanmedian(sq[a:b]))\\n\"
open('/work/scratch/depth_eval3.py','w').write(s)
PY
python /work/scratch/depth_eval3.py"
```
<details><summary>output</summary>

```
base med diff q [-0.32255924 -0.190277   -0.14573765 -0.1060245  -0.04480147  0.04143883
  0.2592566 ] mae [0.06218095 0.12814844 0.21937451] rmse [0.15259971 0.20263085 0.32831724]
  0 96 medabs 0.1588282585144043 rmse 0.28944043815135956
  96 932 medabs 0.1045997142791748 rmse 0.19926808029413223
  932 993 medabs 0.13631105422973633 rmse 0.1724654734134674
cand med diff q [-0.3583889  -0.19488494 -0.16656184 -0.11872292 -0.05433631  0.05576968
  0.29529154] mae [0.06853902 0.13817263 0.23266625] rmse [0.14956221 0.20644741 0.31284259]
  0 96 medabs 0.1837868094444275 rmse 0.3101241737604141
  96 932 medabs 0.11457055807113647 rmse 0.20141441375017166
  932 993 medabs 0.15757524967193604 rmse 0.19249799847602844
refsm13 med diff q [-0.38593793 -0.2080287  -0.17100763 -0.11994886 -0.054538 …[+288 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/nodepth_eval.py <<'PY'
import sys,numpy as np,json
sys.path.insert(0,'/work/scratch');import photo_eval as pe
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
F=993;z=np.load('/work/inputs/reconstruction.npz');Tb=z['T_world_mesh'].astype(float);cam=json.load(open('/work/inputs/camera.json'));campos=np.array(cam['T_world_cam'])[:3,3]
for src in ['candidate','refsm13']:
 T=np.load(f'/work/scratch/{src}.npz')['T'].copy();ray=Tb[:,:3,3]-campos;ray/=np.linalg.norm(ray,axis=1,keepdims=True);d=T[:,:3,3]-Tb[:,:3,3];d-=np.sum(d*ray,axis=1,keepdims=True)*ray;T[:,:3,3]=Tb[:,:3,3]+d
 # smooth 13 and staticize
 q=Rot.from_matrix(T[:,:3,:3]).as_quat()
 for i in range(1,F):
  if np.dot(q[i],q[i-1])<0:q[i]*=-1
 q=savgol_filter(q,13,2,axis=0);q/=np.linalg.norm(q,axis=1,keepdims=True);p=savgol_filter(T[:,:3,3],13,2,axis=0);T[:,:3,:3]=Rot.from_quat(q).as_matrix();T[:,:3,3]=p
 for a,b in [(0,96),(932,993)]:
  qq=Rot.from_matrix(T[a:b,:3,:3]).as_quat()
  for i in range(1,len(qq)):
   if np.dot(qq[i],qq[i-1])<0:qq[i]*=-1
  qm=np.mean(qq,0);qm/=np.linalg.norm(qm);pm=np.median(T[a:b,:3,3],0);T[a:b,:3,:3]=Rot.from_quat(qm).as_matrix();T[a:b,:3,3]=pm
 iou,ncc,grad=pe.evalT(T,1);data=iou+0.15*np.clip(grad,-.2,1)*iou+0.03*np.clip(ncc,-.5,1)*iou
 # depth radial exact check
 ray2=Tb[:,:3,3]-campos;ray2/=np.linalg.norm(ray2,axis=1,keepdims=True);rad=np.sum((T[:,:3,3]-Tb[:,:3,3])*ray2,axis=1)
 print(src,'iou',np.nanmean(iou),'gr …[+194 chars]
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
candidate iou 0.8486318209720441 grad 0.27767271169961644 ncc 0.2317747137399756 data 0.889954659938605 rad [-0.05292838 -0.00019194  0.04027235]
refsm13 iou 0.8509141137112116 grad 0.31380857050182387 ncc 0.24365042128720368 data 0.8971088659603375 rad [-0.05292838 -0.00019439  0.04027235]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/frame0_reference_human.npz');print(z.files)
for k in z.files:print(k,z[k].shape,z[k].dtype)
PY"
```
<details><summary>output</summary>

```
['vertices_world']
vertices_world (18056, 3) float32

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
n=np.array([-0.021970802584377374,-0.9980358822147039,-0.05866567689642774]);d=1.3898362825184551;m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
for fn in ['/work/inputs/reconstruction.npz','/work/scratch/candidate.npz','/work/scratch/refsm13.npz','/work/scratch/nodepth_refsm13.npz']:
 T=np.load(fn)['T_world_mesh'] if 'reconstruction' in fn else np.load(fn)['T']
 print(fn)
 for a,b in [(0,96),(932,993)]:
  T0=T[a];vv=T0[:3,:3]@v.T+T0[:3,3,None];dist=vv.T@n+d
  print(a,b,'min',dist.min(),'maxbottom?',np.percentile(dist,[0,1,2,5,10,50]))
PY"
```
<details><summary>output</summary>

```
/work/inputs/reconstruction.npz
0 96 min -0.019246987277755867 maxbottom? [-1.92469873e-02  3.86055479e-04  1.27032787e-02  3.57755114e-02
  8.08489105e-02  7.32750773e-01]
932 993 min -0.011625452216499799 maxbottom? [-0.01162545  0.01178122  0.02438859  0.04821899  0.09195911  0.74146482]
/work/scratch/candidate.npz
0 96 min -0.013200698648644149 maxbottom? [-0.0132007   0.00583244  0.01829774  0.04127608  0.08658433  0.73862518]
932 993 min -0.012293425413027137 maxbottom? [-0.01229343  0.00991106  0.02237301  0.04645159  0.09037808  0.74072252]
/work/scratch/refsm13.npz
0 96 min -0.013202480990760268 maxbottom? [-0.01320248  0.00582536  0.01829067  0.04128107  0.0865849   0.73862484]
932 993 min -0.0122927798323329 maxbottom? [-0.01229278  0.00990844  0.0223737   0.04645267  0.09037725 …[+283 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/blend_eval.py <<'PY'
import sys,numpy as np,json,trimesh
sys.path.insert(0,'/work/scratch');import photo_eval as pe
from scipy.spatial.transform import Rotation as Rot
F=993;z=np.load('/work/inputs/reconstruction.npz');Tb=z['T_world_mesh'].astype(float);Tr=np.load('/work/scratch/refsm13.npz')['T'].copy();campos=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])[:3,3];ray=Tb[:,:3,3]-campos;ray/=np.linalg.norm(ray,axis=1,keepdims=True);delta=Tr[:,:3,3]-Tb[:,:3,3];rad=np.sum(delta*ray,axis=1,keepdims=True)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();vv=m.vertices;n=np.array([-0.021970802584377374,-0.9980358822147039,-0.05866567689642774]);dplane=1.3898362825184551
for a in [0,.25,.5,.75,1.0]:
 T=Tr.copy();T[:,:3,3]=Tb[:,:3,3]+delta+(a-1)*rad
 # re-staticize translated phase mean (rotation stays)
 for s,e in [(0,96),(932,993)]:T[s:e,:3,3]=np.median(T[s:e,:3,3],0)
 iou,ncc,grad=pe.evalT(T,1);data=iou+0.15*np.clip(grad,-.2,1)*iou+0.03*np.clip(ncc,-.5,1)*iou
 mins=[]
 for i in [0,932]:mins.append(((T[i,:3,:3]@vv.T+T[i,:3,3,None]).T@n+dplane).min())
 print(a,'iou',np.nanmean(iou),'grad',np.nanmean(grad),'ncc',np.nanmean(ncc),'data',np.nanmean(data),'floor',mins,flush=True)
 np.savez(f'/work/scratch/blend{int(a*100)}.npz',T=T)
PY
python /work/scratch/blend_eval.py"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
0 iou 0.6012375667254644 grad 0.12708361706787313 ncc 0.09504597477787996 data 0.6194412086832352 floor [-0.03581036783105507, -0.01000789431116722]
0.25 iou 0.6518354267279464 grad 0.17023096884503894 ncc 0.11332882877149933 data 0.674239405339166 floor [-0.03015839612098148, -0.01057911569145853]
0.5 iou 0.7129309300087274 grad 0.2182650037366817 ncc 0.15157601301625317 data 0.7417977872507718 floor [-0.024506424410907668, -0.01115033707175006]
0.75 iou 0.7922245375240642 grad 0.27463410051096904 ncc 0.20197503205692321 data 0.830697137235945 floor [-0.01885445270083408, -0.011721558452041592]
1.0 iou 0.8639764258234578 grad 0.3489364134235579 ncc 0.25196987324505626 data 0.9156436066763215 floor [-0.013202480990760268, -0.0122927798323329]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
A=np.load('/work/scratch/blend0.npz')['T'];B=np.load('/work/scratch/nodepth_refsm13.npz')['T'];print(np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1).min(),np.median(np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1)),np.max(np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1)))
for i in [0,50,100,300,500,900,950]:print(i,A[i,:3,3],B[i,:3,3])
PY"
```
<details><summary>output</summary>

```
0.00044209772039598675 0.05787106435805849 0.31978339018915697
0 [0.61417336 0.59139048 2.68513529] [0.59589736 0.57569515 2.64343404]
50 [0.61417336 0.59139048 2.68513529] [0.59589736 0.57569515 2.64343404]
100 [0.65144758 0.63139816 2.71512264] [0.59809019 0.58573778 2.59420679]
300 [0.26764712 0.72897595 3.00432458] [0.35873994 0.79700418 3.17602946]
500 [0.01928873 1.0148307  3.67786169] [-0.0223478   0.99108201  3.60899512]
900 [-0.17756118  0.50528071  3.6832793 ] [-0.14260766  0.52513998  3.74370112]
950 [-0.14948801  0.51765826  3.74218437] [-0.14708625  0.51901124  3.74645763]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz');Tb=z['T_world_mesh'];Tr=np.load('/work/scratch/refsm13.npz')['T'];cam=json.load(open('/work/inputs/camera.json'));Wc=np.array(cam['T_world_cam']);Rwc=Wc[:3,:3];campos=Wc[:3,3]
for i in [0,50,100,200,300,400,500,700,900,950]:
 d=Tr[i,:3,3]-Tb[i,:3,3];dc=Rwc.T@d;ray=Tb[i,:3,3]-campos;rayu=ray/np.linalg.norm(ray);rad=d@rayu
 print(i,'d',np.round(d,3),'dc',np.round(dc,3),'rad',round(rad,3))
PY"
```
<details><summary>output</summary>

```
0 d [-0.003 -0.007  0.019] dc [ 0.004 -0.005 -0.02 ] rad -0.021
50 d [-0.003 -0.005  0.022] dc [ 0.004 -0.002 -0.022] rad -0.023
100 d [-0.007 -0.017  0.058] dc [ 0.009 -0.01  -0.059] rad -0.061
200 d [-0.009 -0.011  0.03 ] dc [ 0.01  -0.008 -0.031] rad -0.033
300 d [ 0.017  0.079 -0.072] dc [-0.021  0.07   0.08 ] rad 0.095
400 d [-0.007  0.005  0.   ] dc [0.007 0.005 0.001] rad 0.001
500 d [ 0.006 -0.021  0.033] dc [-0.005 -0.018 -0.035] rad -0.039
700 d [ 0.    -0.008 -0.004] dc [-0.    -0.008  0.004] rad 0.001
900 d [ 0.003  0.002 -0.032] dc [-0.004 -0.002  0.032] rad 0.031
950 d [-0.008 -0.009 -0.005] dc [ 0.008 -0.01   0.004] rad 0.003

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
b=np.load('/work/scratch/depth_base.npz');c=np.load('/work/scratch/depth_cand.npz');r=np.load('/work/scratch/depth_refsm13.npz');Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Tc=np.load('/work/scratch/refsm13.npz')['T'];Wc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);campos=Wc[:3,3];ray=Tb[:,:3,3]-campos;ray/=np.linalg.norm(ray,axis=1,keepdims=True);rad=np.sum((Tc[:,:3,3]-Tb[:,:3,3])*ray,axis=1)
for name,z in [('cand',c),('ref',r)]:
 d=np.abs(z['med'])-np.abs(b['med']);print(name,'depth abs improved frames',(d<0).sum(),'mean delta',np.nanmean(d),'med',np.nanmedian(d))
 for a,b2 in [(0,96),(96,300),(300,500),(500,700),(700,932),(932,993)]:print(a,b2,'delta',np.nanmedian(d[a:b2]),'improved',np.sum(d[a:b2]<0),'n',b2-a)
 # corr radial with depth delta
 m=np.isfinite(d);print('corr rad,dbetter',np.corrcoef(rad[m],d[m])[0,1])
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
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Tr=np.load('/work/scratch/refsm13.npz')['T']
for name,T in [('base',Tb),('ref',Tr)]:
 print(name)
 for a,b in [(0,96),(932,993)]:
  q=R.from_matrix(T[a:b,:3,:3]).as_quat()
  for i in range(1,len(q)):
   if np.dot(q[i],q[i-1])<0:q[i]*=-1
  qm=q.mean(0);qm/=np.linalg.norm(qm);pm=np.median(T[a:b,:3,3],0);print(a,'pm',pm,'qm',qm)
 print('rel',np.degrees((R.from_matrix(T[0,:3,:3]).inv()*R.from_matrix(T[-1,:3,:3])).as_rotvec()))
PY"
```
<details><summary>output</summary>

```
base
0 pm [0.5962226 0.5754905 2.6441321] qm [-0.53558002  0.42947861  0.43946708  0.57928478]
932 pm [-0.14080457  0.5272274   3.7482178 ] qm [-0.6497872   0.22908542  0.22317702  0.68955672]
rel [ -0.47235435   0.65475334 -38.89808291]
ref
0 pm [0.59321436 0.57043149 2.66417629] qm [-0.53757496  0.42694106  0.43649455  0.58155568]
932 pm [-0.14736977  0.5197765   3.74430261] qm [-0.65042153  0.22612172  0.22012653  0.69091614]
rel [ -0.24724736   0.24439494 -38.50114643]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
sys.path.insert(0,'/work/scratch');import photo_eval as pe
from scipy.spatial.transform import Rotation as R
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].copy()
for a,b in [(0,96),(932,993)]:
 q=R.from_matrix(Tb[a:b,:3,:3]).as_quat()
 for i in range(1,len(q)):
  if np.dot(q[i],q[i-1])<0:q[i]*=-1
 qm=q.mean(0);qm/=np.linalg.norm(qm);pm=np.median(Tb[a:b,:3,3],0);Tb[a:b,:3,:3]=R.from_quat(qm).as_matrix();Tb[a:b,:3,3]=pm
iou,ncc,grad=pe.evalT(Tb,1);data=iou+0.15*np.clip(grad,-.2,1)*iou+0.03*np.clip(ncc,-.5,1)*iou
print('all',np.nanmean(iou),np.nanmean(ncc),np.nanmean(grad),np.nanmean(data))
for a,b in [(0,96),(96,932),(932,993)]:print(a,b,np.nanmean(iou[a:b]),np.nanmean(ncc[a:b]),np.nanmean(grad[a:b]),np.nanmean(data[a:b]))
np.savez('/work/scratch/base_static.npz',T=Tb)
PY"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
all 0.813572742493526 0.2329711523394417 0.25183291713739314 0.8519953610236453
0 96 0.8310162793670832 0.31848140213486437 0.568403778809833 0.9098077577746903
96 932 0.8126460500896298 0.22228236938524834 0.2024541598469701 0.8450458199101581
932 993 0.7988208623262417 0.2448865387221489 0.4303548576986955 0.8562548082822505

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_hi.py <<'PY'
import numpy as np, torch, json, cv2, h5py, trimesh, time, sys
import nvdiffrast.torch as dr
DEV='cuda'; W,H=768,576;F=993
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],dtype=np.float32);K[0]*=W/1536;K[1]*=H/1152;cw=np.linalg.inv(np.array(cam['T_world_cam'])).astype(np.float32);ctx=dr.RasterizeCudaContext();Kt=torch.tensor(K[:2,:2],device=DEV);ct=torch.tensor(K[:2,2],device=DEV);cwt=torch.tensor(cw,device=DEV)
def ev(T):
 ious=np.empty(F);prec=np.empty(F);rec=np.empty(F)
 with h5py.File('/work/inputs/masks.h5','r') as hf,torch.no_grad():
  for s in range(0,F,12):
   e=min(F,s+12);B=e-s;Tw=torch.tensor(T[s:e],dtype=torch.float32,device=DEV);Tt=cwt[None]@Tw;vc=torch.einsum('bij,vj->bvi',Tt[:,:3,:3],v)+Tt[:,None,:3,3];zc=vc[:,:,2];uv=torch.einsum('ij,bvj->bvi',Kt,vc[:,:,:2])/zc[:,:,None]+ct[None,None];ndc=torch.stack([uv[:,:,0]/W*2-1,uv[:,:,1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],2);out,_=dr.rasterize(ctx,ndc,f,resolution=(H,W));m=(out[:,:,:,3]>0).cpu().numpy()
   for j in range(B):
    i=s+j;gt=cv2.resize(hf['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);pm=m[j];inter=np.logical_and(pm,gt).sum();ious[i]=inter/max(np.logical_or(pm,gt).sum(),1);prec[i]= …[+613 chars]
```
<details><summary>output</summary>

```
base 0.8040023746805293 [0.40039286 0.73723733 0.76725434 0.79962488 0.84825295 0.89992451
 0.94617059]
 phase [(0, 96, 0.8038), (96, 932, 0.8056), (932, 993, 0.7818)]
prec rec 0.9032886099017841 0.8777739274134331
refsm13 0.8564365339331063 [0.62294211 0.79472785 0.81065324 0.86459559 0.89936388 0.93241579
 0.96696597]
 phase [(0, 96, 0.7971), (96, 932, 0.8654), (932, 993, 0.8275)]
prec rec 0.9372569967425778 0.9072642119549643
base_static 0.8042014821775986 [0.40039286 0.73723733 0.76913876 0.80440098 0.84825295 0.89992451
 0.94617059]
 phase [(0, 96, 0.8066), (96, 932, 0.8056), (932, 993, 0.7806)]
prec rec 0.9034579228427572 0.8778604774272778

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_variants.py <<'PY'
import numpy as np,torch,json,cv2,h5py,trimesh,sys
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'/work/scratch');import photo_eval as pe
DEV='cuda';F=993
Tb=np.load('/work/scratch/base_static.npz')['T'];Tr=np.load('/work/scratch/refsm13.npz')['T']
# mesh
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();vv=mesh.vertices;ff=mesh.faces;v=torch.tensor(vv,dtype=torch.float32,device=DEV);f=torch.tensor(ff,dtype=torch.int32,device=DEV)
cam=json.load(open('/work/inputs/camera.json'));Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc);ctx=dr.RasterizeCudaContext()
def iou_hi(P,inds):
 W,H=768,576;K=np.array(cam['K'],dtype=np.float32);K[0]*=W/1536;K[1]*=H/1152;cwt=torch.tensor(cw,dtype=torch.float32,device=DEV);Kt=torch.tensor(K[:2,:2],device=DEV);ct=torch.tensor(K[:2,2],device=DEV)
 Tt=cwt@torch.tensor(P,dtype=torch.float32,device=DEV);vc=Tt[:3,:3]@v.T+Tt[:3,3,None];zc=vc[2];uv=Kt@vc[:2]/zc+ct[:,None];ndc=torch.stack([uv[0]/W*2-1,uv[1]/H*2-1,torch.zeros_like(zc),torch.ones_like(zc)],1);out,_=dr.rasterize(ctx,ndc[None],f,resolution=(H,W));pm=(out[0,:,:,3]>0).cpu().numpy();ious=[]
 with h5py.File('/work/inputs/masks.h5','r') as hf:
  for i in inds:
   gt=cv2.resize(hf['object'][i].astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST).astype(bool);ious.append(np.logical_and(pm,gt).sum()/np.logical_or(pm,gt).sum())
 return np.m …[+2247 chars]
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
PHASE 0 96
base hi 0.8066 ncc 0.3185 grad 0.5684 floor -0.0172
ref hi 0.7971 ncc 0.4105 grad 0.5447 floor -0.0132
refrot_basepos hi 0.8089 ncc 0.3826 grad 0.5779 floor -0.0171
baserot_refpos hi 0.7971 ncc 0.3852 grad 0.5513 floor -0.0132
blend0.25 hi 0.8068 ncc 0.368 grad 0.5807 floor -0.0162
blend0.5 hi 0.8017 ncc 0.3855 grad 0.5707 floor -0.0152
blend0.75 hi 0.8008 ncc 0.3996 grad 0.5692 floor -0.0142
PHASE 932 993
base hi 0.7806 ncc 0.2449 grad 0.4304 floor -0.0204
ref hi 0.8275 ncc 0.2385 grad 0.5055 floor -0.0123
refrot_basepos hi 0.7776 ncc 0.2095 grad 0.3867 floor -0.0201
baserot_refpos hi 0.8289 ncc 0.2067 grad 0.5159 floor -0.0126
blend0.25 hi 0.8033 ncc 0.1696 grad 0.4499 floor -0.0183
blend0.5 hi 0.8162 ncc 0.2939 grad 0.5003 floor -0.0163
blend0.75 hi 0 …[+43 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
Tb=np.load('/work/scratch/base_static.npz')['T'];Tr=np.load('/work/scratch/refsm13.npz')['T'];T=Tr.copy()
# initial: improved pooled rotation with reliable baseline pooled translation
T[:96,:3,:3]=Tr[0,:3,:3];T[:96,:3,3]=Tb[0,:3,3]
# final: 75% pooled mask/texture pose, 25% baseline pooled pose
T[932:,:3,3]=.25*Tb[932,:3,3]+.75*Tr[932,:3,3]
q0=R.from_matrix(Tb[932,:3,:3]);q1=R.from_matrix(Tr[932,:3,:3]);dq=(q0.inv()*q1);T[932:,:3,:3]=(q0*R.from_rotvec(.75*dq.as_rotvec())).as_matrix()
np.savez('/work/scratch/final_hybrid.npz',T=T)
PY
python - <<'PY'
import numpy as np,sys
sys.path.insert(0,'/work/scratch');import photo_eval as pe
T=np.load('/work/scratch/final_hybrid.npz')['T'];iou,ncc,grad=pe.evalT(T,1);data=iou+0.15*np.clip(grad,-.2,1)*iou+0.03*np.clip(ncc,-.5,1)*iou
print('all',np.nanmean(iou),np.nanmean(ncc),np.nanmean(grad),np.nanmean(data))
for a,b in [(0,96),(96,932),(932,993)]:print(a,b,np.nanmean(iou[a:b]),np.nanmean(ncc[a:b]),np.nanmean(grad[a:b]),np.nanmean(data[a:b]))
PY"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
all 0.8633497405546899 0.2535201175264266 0.35240014376590223 0.9154240917274039
0 96 0.8309621457529923 0.38262302024211997 0.5779493823416764 0.9125377940466032
96 932 0.8692462516570517 0.23474351888641487 0.31502999119494574 0.9165225124922546
932 993 0.8335091802167954 0.30767352412221805 0.5095922855043493 0.9049126953002187

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];T=np.load('/work/scratch/final_hybrid.npz')['T'];rb=R.from_matrix(Tb[:,:3,:3]);r=R.from_matrix(T[:,:3,:3]);ang=np.degrees((rb.inv()*r).magnitude());dist=np.linalg.norm(T[:,:3,3]-Tb[:,:3,3],axis=1)
print('rot diff q',np.percentile(ang,[0,10,25,50,75,90,95,99,100]))
print('trans diff q',np.percentile(dist,[0,10,25,50,75,90,95,99,100]))
print('largest rot',[(int(i),round(float(ang[i]),1),round(float(dist[i]),3)) for i in np.argsort(ang)[-20:][::-1]])
PY"
```
<details><summary>output</summary>

```
rot diff q [ 0.1190645   0.40017581  0.62239836  1.11181653  2.20332387  3.779794
  5.48502932 13.77131396 35.39138938]
trans diff q [0.00095204 0.00890122 0.01639003 0.03139767 0.0536608  0.08556021
 0.10437169 0.13255682 0.1652473 ]
largest rot [(605, 35.4, 0.009), (606, 26.5, 0.013), (613, 22.1, 0.017), (608, 20.7, 0.024), (612, 19.8, 0.012), (607, 18.5, 0.026), (287, 17.0, 0.162), (297, 16.9, 0.106), (614, 16.4, 0.038), (611, 14.9, 0.02), (283, 13.7, 0.152), (377, 13.4, 0.044), (284, 13.0, 0.122), (286, 12.7, 0.165), (285, 12.2, 0.144), (395, 12.2, 0.009), (288, 11.9, 0.12), (296, 11.7, 0.085), (376, 10.9, 0.04), (291, 10.9, 0.124)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/final_hybrid.npz')['T'];rb=R.from_matrix(T[:,:3,:3]);z=np.array([0,0,1.]);up=-np.array([-0.021970802584377374,-0.9980358822147039,-0.05866567689642774]);up/=np.linalg.norm(up)
tilt=np.degrees(np.arccos(np.clip((rb.apply(z))@up,-1,1)))
# yaw twist relative local z maybe project
py=T[:,:3,3]@up
for i in range(0,993,20):print(i,'tilt',round(float(tilt[i]),1),'height',round(float(py[i]),3))
PY"
```
<details><summary>output</summary>

```
0 tilt 0.3 height 0.743
20 tilt 0.3 height 0.743
40 tilt 0.3 height 0.743
60 tilt 0.3 height 0.743
80 tilt 0.3 height 0.743
100 tilt 0.6 height 0.738
120 tilt 0.6 height 0.736
140 tilt 0.0 height 0.739
160 tilt 0.9 height 0.738
180 tilt 2.6 height 0.734
200 tilt 10.9 height 0.709
220 tilt 21.1 height 0.692
240 tilt 27.8 height 0.703
260 tilt 37.6 height 0.721
280 tilt 61.7 height 0.86
300 tilt 78.4 height 1.013
320 tilt 85.8 height 1.047
340 tilt 86.1 height 1.044
360 tilt 86.9 height 1.056
380 tilt 87.6 height 1.058
400 tilt 92.6 height 1.02
420 tilt 115.2 height 0.988
440 tilt 127.7 height 1.009
460 tilt 133.3 height 1.02
480 tilt 148.8 height 1.093
500 tilt 164.9 height 1.187
520 tilt 179.6 height 1.289
540 tilt 179.8 height 1.296
560 tilt 179.2 height 1.293
580 tilt 179.3 height 1.298
 …[+541 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/hi_base.npz')['iou'];r=np.load('/work/scratch/hi_refsm13.npz')['iou'];d=r-b
print('improved',np.sum(d>0),'worse',np.sum(d<0),'med delta',np.median(d), 'mean',d.mean())
print('top worse',[(int(i),round(float(b[i]),3),round(float(r[i]),3),round(float(d[i]),3)) for i in np.argsort(d)[:30]])
print('top better',[(int(i),round(float(b[i]),3),round(float(r[i]),3),round(float(d[i]),3)) for i in np.argsort(d)[-20:][::-1]])
for a,b2 in [(0,96),(96,176),(176,400),(400,520),(520,600),(600,770),(770,830),(830,932),(932,993)]:print(a,b2,round(float(b[a:b2].mean()),3),round(float(r[a:b2].mean()),3),round(float(d[a:b2].mean()),3))
PY"
```
<details><summary>output</summary>

```
improved 921 worse 72 med delta 0.04277747133507648 mean 0.05243415925257702
top worse [(70, 0.845, 0.798, -0.047), (74, 0.837, 0.797, -0.04), (75, 0.837, 0.799, -0.038), (71, 0.832, 0.798, -0.034), (73, 0.831, 0.796, -0.034), (72, 0.829, 0.797, -0.033), (69, 0.826, 0.796, -0.029), (76, 0.825, 0.797, -0.028), (77, 0.826, 0.798, -0.027), (67, 0.821, 0.796, -0.025), (53, 0.825, 0.8, -0.025), (85, 0.815, 0.792, -0.022), (66, 0.82, 0.798, -0.022), (78, 0.817, 0.796, -0.02), (54, 0.819, 0.799, -0.02), (82, 0.814, 0.795, -0.02), (68, 0.815, 0.796, -0.019), (63, 0.817, 0.799, -0.018), (65, 0.817, 0.8, -0.018), (81, 0.813, 0.795, -0.018), (64, 0.815, 0.797, -0.018), (188, 0.816, 0.798, -0.017), (50, 0.816, 0.799, -0.017), (79, 0.813, 0.796, -0.017), (49, 0.814, 0.797, -0.016), (62, 0.815, 0.799, - …[+914 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_hi.py';s=open(p).read();s=s.replace(\"for name,path in [('base','/work/inputs/reconstruction.npz'),('refsm13','/work/scratch/refsm13.npz'),('base_static','/work/scratch/base_static.npz')]:\",\"for name,path in [('final_hybrid','/work/scratch/final_hybrid.npz')]:\")
s=s.replace(\"T=np.load(path)['T_world_mesh'] if name=='base' else np.load(path)['T']\",\"T=np.load(path)['T_world_mesh'] if name=='base' else np.load(path)['T']\")
open('/work/scratch/eval_hi_final.py','w').write(s)
PY
python /work/scratch/eval_hi_final.py"
```
<details><summary>output</summary>

```
final_hybrid 0.8574181072875741 [0.62294211 0.79787176 0.81265509 0.86459559 0.89936388 0.93241579
 0.96696597]
 phase [(0, 96, 0.8089), (96, 932, 0.8654), (932, 993, 0.8249)]
prec rec 0.9382002634803986 0.907589957628076

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'df -h /work && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader && git -C /work status --short 2>/dev/null || true'
```
<details><summary>output</summary>

```
Filesystem                                               Size  Used Avail Use% Mounted on
10.127.102.71:/Group/hanzhou/v2d/agent_runs/T1_ds_ep014  139T   49T   90T  36% /work
NVIDIA RTX 6000 Ada Generation, 49140 MiB, 14415 MiB
NVIDIA RTX 6000 Ada Generation, 49140 MiB, 14309 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import scipy,h5py,PIL,cv2,torch,trimesh,nvdiffrast
for m in [scipy,h5py,PIL,cv2,torch,trimesh,nvdiffrast]:print(m.__name__,getattr(m,'__version__','?'))
PY"
```
<details><summary>output</summary>

```
scipy 1.16.3
h5py 3.16.0
PIL 10.2.0
cv2 4.11.0
torch 2.5.1+cu124
trimesh 5.1.1
nvdiffrast 0.4.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,240p' /work/scratch/pose_opt.py"
```
<details><summary>output</summary>

```
import numpy as np, torch, json, cv2, os, time
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
import nvdiffrast.torch as dr
DEV='cuda'; W,H=384,288; SC=0.25
z=np.load('/work/inputs/reconstruction.npz'); T0=z['T_world_mesh'].astype(np.float64); F=len(T0)
GT=np.load('/work/scratch/gt_masks_384.npy')
import trimesh
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=torch.tensor(mesh.vertices,dtype=torch.float32,device=DEV);f=torch.tensor(mesh.faces,dtype=torch.int32,device=DEV)
cam=json.load(open('/work/inputs/camera.json'));Kfull=np.array(cam['K'],dtype=np.float64);K=Kfull.copy();K[0]*=SC;K[1]*=SC;Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc);ctx=dr.RasterizeCudaContext()
Kt=torch.tensor(K[:2,:2],dt …[+3020 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,260p' /work/scratch/refine_temporal.py"
```
<details><summary>output</summary>

```
import sys, numpy as np, time
sys.path.insert(0,'/work/scratch')
import photo_eval as pe
from scipy.spatial.transform import Rotation as Rot
F=993; T=np.load('/work/scratch/candidate.npz')['T'].copy(); T0=T.copy()
static=np.ones(F,bool);static[96:932]=False
# camera parameters
import json
cam=json.load(open('/work/inputs/camera.json'));Kfull=np.array(cam['K']);Wc=np.array(cam['T_world_cam']);cw=np.linalg.inv(Wc);Rwc=Wc[:3,:3];campos=Wc[:3,3];SC=.25
def q_of(X):return Rot.from_matrix(X[:,:3,:3]).as_quat()
def scores(X):
 iou,ncc,grad=pe.evalT(X,1)
 data=iou+0.15*np.clip(grad,-.2,1.)*iou+0.03*np.clip(ncc,-.5,1.)*iou
 q=q_of(X)
 for i in range(1,F):
  if np.dot(q[i],q[i-1])<0:q[i]*=-1
 p=X[:,:3,3]
 # compare candidate pose to smooth neighbor center; d0 for first/last
 dr=np.zeros(F);dt=np.zer …[+2220 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,180p' /work/scratch/eval_refined_smooth.py && sed -n '1,120p' /work/scratch/make_candidate.py"
```
<details><summary>output</summary>

```
import sys,numpy as np
sys.path.insert(0,'/work/scratch');import photo_eval as pe
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
F=993;T0=np.load('/work/scratch/temporal_refined.npz')['T']
def smooth(T,win,static=True):
 T=T.copy();q=Rot.from_matrix(T[:,:3,:3]).as_quat()
 for i in range(1,F):
  if np.dot(q[i],q[i-1])<0:q[i]*=-1
 q=savgol_filter(q,win,2,axis=0);q/=np.linalg.norm(q,axis=1,keepdims=True);p=savgol_filter(T[:,:3,3],win,2,axis=0)
 T[:,:3,:3]=Rot.from_quat(q).as_matrix();T[:,:3,3]=p
 if static:
  for a,b in [(0,96),(932,993)]:
   qq=Rot.from_matrix(T[a:b,:3,:3]).as_quat()
   for i in range(1,len(qq)):
    if np.dot(qq[i],qq[i-1])<0:qq[i]*=-1
   qm=np.mean(qq,0);qm/=np.linalg.norm(qm);pm=np.median(T[a:b,:3,3],0);T[a:b,:3,:3]=Rot.from_qua …[+1447 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');import photo_eval as pe
T=np.load('/work/scratch/final_hybrid.npz')['T'];iou,ncc,grad=pe.evalT(T,1);np.savez('/work/scratch/photo_1_final.npz',iou=iou,ncc=ncc,grad=grad)
print(iou.mean(),ncc.mean(),grad.mean())
PY"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
0.8633497405546899 0.2535201175264266 0.35240014376590223

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/depth_eval3.py').read();i=s.index('for name,path in [');s=s[:i]+\"for name,path in [('final','/work/scratch/final_hybrid.npz')]:\\n T=np.load(path)['T'];med,mae,sq=evalT(T);np.savez(f'/work/scratch/depth_{name}.npz',med=med,mae=mae,sq=sq);print(name,np.nanmedian(np.abs(med)),np.nanmedian(sq));\\n for a,b in [(0,96),(96,932),(932,993)]:print(a,b,np.nanmedian(np.abs(med[a:b])),np.nanmedian(sq[a:b]))\\n\"
open('/work/scratch/depth_final.py','w').write(s)
PY
python /work/scratch/depth_final.py"
```
<details><summary>output</summary>

```
final 0.128373384475708 0.20817673206329346
0 96 0.16022121906280518 0.29649879038333893
96 932 0.11671066284179688 0.20355276763439178
932 993 0.15648412704467773 0.19160524010658264

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
z=np.load('/work/inputs/reconstruction.npz');kp=z['human_keypoints_world'];cp=z['contact_prob_right_left'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::8]
for name,path in [('base','/work/inputs/reconstruction.npz'),('final','/work/scratch/final_hybrid.npz')]:
 T=(z['T_world_mesh'] if name=='base' else np.load(path)['T']);d=np.full((993,2),np.nan)
 for i in range(993):
  vv=T[i,:3,:3]@v.T+T[i,:3,3,None];tree=cKDTree(vv.T)
  for h,a,b in [(0,21,42),(1,42,63)]:d[i,h]=tree.query(kp[i,a:b])[0].min()
 print(name,[round(float(np.median(d[cp[:,h]>.5,h])),4) for h in range(2)],[round(float(np.mean(d[cp[:,h]>.5,h])),4) for h in range(2)])
PY"
```
<details><summary>output</summary>

```
base [0.0042, 0.005] [0.018, 0.014]
final [0.005, 0.0061] [0.0169, 0.0141]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Tf=np.load('/work/scratch/final_hybrid.npz')['T']
for name,T in [('base',Tb),('final',Tf)]:
 r=R.from_matrix(T[:,:3,:3]);vel=np.degrees((r[:-1].inv()*r[1:]).magnitude());rv=(r[:-1].inv()*r[1:]).as_rotvec();acc=np.degrees(np.linalg.norm(np.diff(rv,axis=0),axis=1));tv=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
 print(name,'rot step mean/p95/max',vel.mean(),np.percentile(vel,95),vel.max(),'acc p95/max',np.percentile(acc,95),acc.max(),'trans p95/max',np.percentile(tv,95),tv.max())
 for a,b in [(0,96),(96,176),(176,400),(400,520),(520,600),(600,770),(770,830),(830,932),(932,993)]:print(a,b,round(float(vel[a:b].mean()),3),round(float(np.percentile(vel[a:b],95)),3),round(float(np.percentile(acc[a:b],95)),3))
PY"
```
<details><summary>output</summary>

```
base rot step mean/p95/max 1.3793281993757296 4.482991692645272 28.839774381963185 acc p95/max 7.0352984725684955 32.6196924819364 trans p95/max 0.027780543733388157 0.10260339
0 96 0.212 0.445 0.804
96 176 0.298 0.66 1.062
176 400 2.44 8.698 12.663
400 520 1.791 4.39 5.676
520 600 1.127 2.554 4.851
600 770 1.713 3.791 6.518
770 830 1.044 1.935 2.931
830 932 0.989 2.116 2.084
932 993 0.297 0.671 0.897
final rot step mean/p95/max 0.6266507836179226 1.6839235791679588 9.52199070238072 acc p95/max 0.5561221421864231 2.0845966712931165 trans p95/max 0.01223260535498288 0.023525403940615187
0 96 0.003 0.0 0.0
96 176 0.171 0.456 0.462
176 400 0.732 1.909 0.604
400 520 0.954 1.924 0.413
520 600 0.259 0.598 0.615
600 770 1.099 4.136 1.244
770 830 0.633 1.101 0.316
830 932 0.821 1.907 0.397
932 993 …[+13 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
phases=[(0,96,'Static upright start'),(96,176,'Engage / roll'),(176,400,'Lower to side'),(400,520,'Flip wheels up'),(520,600,'Upside-down hold'),(600,770,'Roll to side'),(770,830,'Side hold'),(830,932,'Return upright'),(932,993,'Static upright end')]
hb=np.load('/work/scratch/hi_base.npz');hf=np.load('/work/scratch/hi_final_hybrid.npz');pb=np.load('/work/scratch/photo_1_base.npz');pf=np.load('/work/scratch/photo_1_final.npz');db=np.load('/work/scratch/depth_base.npz');df=np.load('/work/scratch/depth_final.npz')
print('| phase | frames | mask IoU base/final | grad-NCC base/final | depth MAE base/final (cm) |')
for a,b,n in phases:
 print(f'| {n} | {a}-{b-1} | {hb[\"iou\"][a:b].mean():.3f} / {hf[\"iou\"][a:b].mean():.3f} | {pb[\"grad\"][a:b].mean():.3f} / {pf[\"grad\"][a:b].mean():.3f} | {np.nanmedian(np.abs(db[\"med\"][a:b]))*100:.1f} / {np.nanmedian(np.abs(df[\"med\"][a:b]))*100:.1f} |')
print('overall',hb['iou'].mean(),hf['iou'].mean(),pb['grad'].mean(),pf['grad'].mean(),np.nanmedian(np.abs(db['med']))*100,np.nanmedian(np.abs(df['med']))*100)
print('floor')
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_blend_phases.py <<'PY'
import sys,numpy as np,json
sys.path.insert(0,'/work/scratch');import photo_eval as pe
Tb=np.load('/work/scratch/base_static.npz')['T'];Tr=np.load('/work/scratch/final_hybrid.npz')['T'];campos=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])[:3,3];ray=Tb[:,:3,3]-campos;ray/=np.linalg.norm(ray,axis=1,keepdims=True);delta=Tr[:,:3,3]-Tb[:,:3,3];rad=np.sum(delta*ray,axis=1,keepdims=True)
variants={
'all':[],
'B':[(96,176),(400,520),(770,830)],
'C':[(96,176),(400,520)],
'D':[(96,176),(400,520),(770,830)],
}
for name,ps in variants.items():
 T=Tr.copy()
 for a,b in ps:
  frac=.5 if name=='D' else 0
  T[a:b,:3,3]=Tb[a:b,:3,3]+delta[a:b]+(frac-1)*rad[a:b]
 # ensure static
 for a,b in [(0,96),(932,993)]:T[a:b,:3,3]=np.median(T[a:b,:3,3],0)
 iou,ncc,grad=pe.evalT(T,1);data=iou+0.15*np.clip(grad,-.2,1)*iou+0.03*np.clip(ncc,-.5,1)*iou
 print(name,'all',np.nanmean(iou),np.nanmean(grad),np.nanmean(ncc),np.nanmean(data))
 for a,b in [(0,96),(96,176),(176,400),(400,520),(520,600),(600,770),(770,830),(830,932),(932,993)]:print(a,b,round(float(np.nanmean(iou[a:b])),3),round(float(np.nanmean(grad[a:b])),3),round(float(np.nanmean(data[a:b])),3))
 np.savez(f'/work/scratch/depthblend_{name}.npz',T=T)
PY
python /work/scratch/depth_blend_phases.py"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
all all 0.8633497405546899 0.35240014376590223 0.2535201175264266 0.9154240917274039
0 96 0.831 0.578 0.913
96 176 0.833 0.563 0.914
176 400 0.921 0.556 1.003
400 520 0.817 0.293 0.86
520 600 0.883 0.289 0.928
600 770 0.891 0.017 0.897
770 830 0.855 0.029 0.863
830 932 0.808 0.302 0.85
932 993 0.834 0.51 0.905
B all 0.7625921425937654 0.2622722395023199 0.1793746410704917 0.8038492790686458
0 96 0.831 0.578 0.913
96 176 0.216 -0.109 0.219
176 400 0.921 0.556 1.003
400 520 0.476 0.031 0.486
520 600 0.883 0.289 0.928
600 770 0.891 0.017 0.897
770 830 0.693 -0.041 0.692
830 932 0.808 0.302 0.85
932 993 0.834 0.51 0.905
C all 0.7724263314147395 0.26654044604935107 0.18473506096887266 0.8142311082147582
0 96 0.831 0.578 0.913
96 176 0.216 -0.109 0.219
176 400 0.921 0.55 …[+471 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Deterministic trajectory recovery for the round-table monocular episode.

The method starts from the released frame-wise reconstruction, robustly smooths
it, performs a mask-driven local rigid-pose search, then performs a second
search with a texture-gradient term and a temporal acceleration penalty.
Finally, the two unambiguous static phases are pooled over all their frames.

Only the supplied inputs are used.  Randomness is not used.
"""

import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

import nvdiffrast.torch as dr

# The optimization renderer uses a quarter-resolution preview.  This is enough
# for a 0.75 m object that occupies roughly 100-350 pixels in the source video.
EVAL_W, EVAL_H = 384, 288
EVAL_SCALE = 0.25
CHUNK = 24
DEVICE = "cuda"
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_inputs(input_dir):
    with open(os.path.join(input_dir, "camera.json"), "r") as f:
        camera = json.load(f)
    with open(os.path.join(input_dir, "episode.json"), "r") as f:
        episode = json.load(f)
    rec = np.load(os.path.join(input_dir, "reconstruction.npz"))
    return camera, episode, r …[+16513 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "/usr/bin/time -f 'ELAPSED=%e MAXRSS_MB=%M' python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
[pose] iteration 1/4 mask IoU=0.8153
[pose] iteration 2/4 mask IoU=0.8542
[pose] iteration 3/4 mask IoU=0.8584
[pose] iteration 4/4 mask IoU=0.8593
[appearance] iteration 1/3
[appearance] iteration 2/3
[appearance] iteration 3/3
wrote /work/outputs/object_poses_world.npz in 315.3s

real	5m17.742s
user	5m44.641s
sys	1m34.328s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
A=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/scratch/final_hybrid.npz')['T']
print(A.shape,A.dtype)
print('max abs diff',np.max(np.abs(A-B)),'mean',np.mean(np.abs(A-B)))
print('finite',np.isfinite(A).all())
print('rigid err',np.max(np.abs(np.einsum('nij,nkj->nik',A[:,:3,:3],A[:,:3,:3])-np.eye(3)[None])))
print('last row err',np.max(np.abs(A[:,3]-np.array([0,0,0,1]))))
print('sample',A[0]);print(A[-1])
PY"
```
<details><summary>output</summary>

```
(993, 4, 4) float64
max abs diff 0.02312949264967301 mean 0.00029431205225809057
finite True
rigid err 1.1102230246251565e-15
last row err 0.0
sample [[ 0.24967977 -0.96795523  0.02688289  0.59622258]
 [ 0.049085    0.04037797  0.9979781   0.5754905 ]
 [-0.96708359 -0.2478554   0.05759364  2.64413202]
 [ 0.          0.          0.          1.        ]]
[[ 0.8067321  -0.5903344   0.02624164 -0.14181452]
 [ 0.00897785  0.05664783  0.99835386  0.52213809]
 [-0.59084915 -0.80516851  0.05099956  3.7438126 ]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -lh /work/outputs && sha256sum /work/outputs/solve.py /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
total 152K
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 16:34 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 125K Oct 20 16:39 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  18K Oct 20 16:34 solve.py
c09f5f346e8f074c5a0810ff28af709d56a1a5d8af2eb1be5c281d9c40427077  /work/outputs/solve.py
137fdc0b0a28c9de98a73d58c862bb1b89ae21c0906d342e05c555f4a4ed2983  /work/outputs/object_poses_world.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/eval_hi.py').read();i=s.index('for name,path in [');s=s[:i]+\"for name,path in [('solve','/work/outputs/object_poses_world.npz')]:\\n T=np.load(path)['T_world_mesh'];q,p,r=ev(T);np.savez('/work/scratch/hi_solve.npz',iou=q,prec=p,rec=r);print(name,np.mean(q),np.percentile(q,[0,10,25,50,75,90,100]));print(' phase',[(a,b,round(float(np.mean(q[a:b])),4)) for a,b in [(0,96),(96,932),(932,993)]]);print('prec rec',np.mean(p),np.mean(r),flush=True)\\n\"
open('/work/scratch/eval_hi_solve.py','w').write(s)
PY
python /work/scratch/eval_hi_solve.py"
```
<details><summary>output</summary>

```
solve 0.8575291977846544 [0.62294211 0.79807411 0.81265985 0.86459559 0.89936388 0.93241579
 0.96696597]
 phase [(0, 96, 0.8085), (96, 932, 0.8654), (932, 993, 0.8265)]
prec rec 0.9383681050836623 0.9075751006268188

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for name,X in [('baseline',Tb),('result',T)]:
 r=R.from_matrix(X[:,:3,:3]);vel=np.degrees((r[:-1].inv()*r[1:]).magnitude());rv=(r[:-1].inv()*r[1:]).as_rotvec();acc=np.degrees(np.linalg.norm(np.diff(rv,axis=0),axis=1));tv=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1)
 print(name,'rot_step_mean',vel.mean(),'rot_step_p95',np.percentile(vel,95),'rot_step_max',vel.max(),'rot_acc_p95',np.percentile(acc,95),'trans_step_p95',np.percentile(tv,95),'trans_step_max',tv.max())
PY"
```
<details><summary>output</summary>

```
baseline rot_step_mean 1.3793281993757296 rot_step_p95 4.482991692645272 rot_step_max 28.839774381963185 rot_acc_p95 7.0352984725684955 trans_step_p95 0.027780543733388157 trans_step_max 0.10260339
result rot_step_mean 0.6280767265803878 rot_step_p95 1.6839235791679537 rot_step_max 9.521990702380702 rot_acc_p95 0.556122142186377 trans_step_p95 0.01223260535498288 trans_step_max 0.02178573003898436

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');import photo_eval as pe
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];iou,ncc,grad=pe.evalT(T,1);np.savez('/work/scratch/photo_1_solve.npz',iou=iou,ncc=ncc,grad=grad);print(iou.mean(),ncc.mean(),grad.mean())
PY"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
0.8630036334723832 0.24942717365968914 0.35185648374818945

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/depth_eval3.py').read();i=s.index('for name,path in [');s=s[:i]+\"for name,path in [('solve','/work/outputs/object_poses_world.npz')]:\\n T=np.load(path)['T_world_mesh'];med,mae,sq=evalT(T);np.savez('/work/scratch/depth_solve.npz',med=med,mae=mae,sq=sq);print(name,np.nanmedian(np.abs(med)),np.nanmedian(sq));\\n for a,b in [(0,96),(96,932),(932,993)]:print(a,b,np.nanmedian(np.abs(med[a:b])),np.nanmedian(sq[a:b]))\\n\"
open('/work/scratch/depth_solve.py','w').write(s)
PY
python /work/scratch/depth_solve.py"
```
<details><summary>output</summary>

```
solve 0.12854862213134766 0.20986409485340118
0 96 0.15907573699951172 0.30212707817554474
96 932 0.11671066284179688 0.20417437702417374
932 993 0.15538418292999268 0.19461669027805328

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/depth_base.npz');s=np.load('/work/scratch/depth_solve.npz')
for a,e in [(0,96),(96,932),(932,993)]:print(a,e,np.nanmedian(np.abs(b['med'][a:e])),np.nanmedian(np.abs(s['med'][a:e])))
PY"
```
<details><summary>output</summary>

```
0 96 0.1588282585144043 0.15907573699951172
96 932 0.1045997142791748 0.11671066284179688
932 993 0.13631105422973633 0.15538418292999268

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
phases=[(0,96,'Static upright (start)'),(96,176,'Engage / initial roll'),(176,400,'Lower onto side'),(400,520,'Flip wheels upward'),(520,600,'Upside-down hold'),(600,770,'Rotate back onto side'),(770,830,'Side hold'),(830,932,'Return upright / release'),(932,993,'Static upright (end)')]
hb=np.load('/work/scratch/hi_base.npz');hs=np.load('/work/scratch/hi_solve.npz');pb=np.load('/work/scratch/photo_1_base.npz');ps=np.load('/work/scratch/photo_1_solve.npz');db=np.load('/work/scratch/depth_base.npz');ds=np.load('/work/scratch/depth_solve.npz')
for a,b,n in phases:
 print(f'| {n} | {a}-{b-1} | {hb[\"iou\"][a:b].mean():.3f} / {hs[\"iou\"][a:b].mean():.3f} | {pb[\"grad\"][a:b].mean():.3f} / {ps[\"grad\"][a:b].mean():.3f} | {np.nanmedian(np.abs(db[\"med\"][a:b]))*100:.1f} / {np.nanmedian(np.abs(ds[\"med\"][a:b]))*100:.1f} |')
print('all',hb['iou'].mean(),hs['iou'].mean(),pb['grad'].mean(),ps['grad'].mean(),np.nanmedian(np.abs(db['med']))*100,np.nanmedian(np.abs(ds['med']))*100)
PY"
```
<details><summary>output</summary>

```
| Static upright (start) | 0-95 | 0.804 / 0.808 | 0.575 / 0.568 | 15.9 / 15.9 |
| Engage / initial roll | 96-175 | 0.761 / 0.807 | 0.607 / 0.563 | 5.1 / 17.3 |
| Lower onto side | 176-399 | 0.879 / 0.917 | 0.457 / 0.556 | 12.5 / 12.0 |
| Flip wheels upward | 400-519 | 0.772 / 0.816 | 0.186 / 0.293 | 8.2 / 13.4 |
| Upside-down hold | 520-599 | 0.840 / 0.874 | 0.250 / 0.289 | 12.8 / 12.8 |
| Rotate back onto side | 600-769 | 0.800 / 0.888 | -0.047 / 0.017 | 7.4 / 7.7 |
| Side hold | 770-829 | 0.751 / 0.866 | -0.309 / 0.029 | 2.7 / 6.3 |
| Return upright / release | 830-931 | 0.734 / 0.811 | 0.024 / 0.301 | 15.1 / 15.5 |
| Static upright (end) | 932-992 | 0.782 / 0.826 | 0.421 / 0.517 | 13.6 / 15.5 |
all 0.8040023746805293 0.8575291977846544 0.25191405350639673 0.35185648374818945 11.44971847 …[+27 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# evaluate 180-degree flips of final output on quarter-res masks
import sys,numpy as np
sys.path.insert(0,'/work/scratch');import photo_eval as pe
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];base=pe.evalT(T,1)[0];print('base',base.mean())
for name,a in [('mesh_x',[1,0,0]),('mesh_y',[0,1,0]),('mesh_z',[0,0,1]),('world_y',[0,1,0])]:
 C=T.copy();D=R.from_rotvec(np.pi*np.array(a)).as_matrix()
 if name.startswith('mesh'):C[:,:3,:3]=T[:,:3,:3]@D
 else:C[:,:3,:3]=D@T[:,:3,:3]
 q=pe.evalT(C,1)[0];print(name,q.mean(),'better>0.02',int((q>base+.02).sum()),'max_delta',float((q-base).max()))
PY"
```
<details><summary>output</summary>

```
video (993, 288, 384, 3)
base 0.8630036334723832
mesh_x 0.2899370554490331 better>0.02 0 max_delta -0.09192252204114348
mesh_y 0.2899250698002018 better>0.02 0 max_delta -0.09741072926245509
mesh_z 0.6765920626909524 better>0.02 0 max_delta -0.011351841435875087
world_y 0.4579912800699358 better>0.02 0 max_delta -0.14393794056536813

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
ph=[(0,96),(96,176),(176,400),(400,520),(520,600),(600,770),(770,830),(830,932),(932,993)]
with h5py.File('/work/inputs/masks.h5','r') as f:
 for a,b in ph:
  ar=[];diag=[]
  for i in range(a,b):
   m=f['object'][i];ys,xs=np.where(m);ar.append(m.sum());diag.append(np.hypot(xs.max()-xs.min(),ys.max()-ys.min()))
  print(a,b,'area med',np.median(ar),'equiv diameter',round(float(np.sqrt(np.median(ar)/np.pi)*2),1),'bboxdiag',round(float(np.median(diag)),1))
PY"
```
<details><summary>output</summary>

```
0 96 area med 12245.5 equiv diameter 124.9 bboxdiag 295.3
96 176 area med 11967.5 equiv diameter 123.4 bboxdiag 289.2
176 400 area med 36266.0 equiv diameter 214.9 bboxdiag 339.1
400 520 area med 23756.5 equiv diameter 173.9 bboxdiag 429.5
520 600 area med 35522.5 equiv diameter 212.7 bboxdiag 421.3
600 770 area med 48888.5 equiv diameter 249.5 bboxdiag 487.6
770 830 area med 52772.0 equiv diameter 259.2 bboxdiag 565.5
830 932 area med 29124.0 equiv diameter 192.6 bboxdiag 447.4
932 993 area med 29217.0 equiv diameter 192.9 bboxdiag 442.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;n=np.array([-0.021970802584377374,-0.9980358822147039,-0.05866567689642774]);d=1.3898362825184551
for name,path in [('base','/work/inputs/reconstruction.npz'),('solve','/work/outputs/object_poses_world.npz')]:
 T=np.load(path)['T_world_mesh'];
 for i in [0,95,932,992]:
  vv=T[i,:3,:3]@v.T+T[i,:3,3,None];ds=vv.T@n+d;print(name,i,round(float(ds.min()),4),round(float(np.percentile(ds,1)),4))
PY"
```
<details><summary>output</summary>

```
base 0 -0.0192 0.0004
base 95 -0.0279 -0.0093
base 932 -0.0116 0.0118
base 992 -0.0211 0.0042
solve 0 -0.0171 0.002
solve 95 -0.0171 0.002
solve 932 -0.0149 0.0075
solve 992 -0.0149 0.0075

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
from scipy.spatial import cKDTree
z=np.load('/work/inputs/reconstruction.npz');kp=z['human_keypoints_world'];cp=z['contact_prob_right_left'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::8]
for name,path in [('base','/work/inputs/reconstruction.npz'),('solve','/work/outputs/object_poses_world.npz')]:
 T=z['T_world_mesh'] if name=='base' else np.load(path)['T_world_mesh'];d=np.full((993,2),np.nan)
 for i in range(993):
  vv=T[i,:3,:3]@v.T+T[i,:3,3,None];tree=cKDTree(vv.T)
  for h,a,b in [(0,21,42),(1,42,63)]:d[i,h]=tree.query(kp[i,a:b])[0].min()
 print(name,'median',[round(float(np.median(d[cp[:,h]>.5,h])),4) for h in range(2)],'mean',[round(float(np.mean(d[cp[:,h]>.5,h])),4) for h in range(2)])
PY"
```
<details><summary>output</summary>

```
base median [0.0042, 0.005] mean [0.018, 0.014]
solve median [0.005, 0.0061] mean [0.0169, 0.0142]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Round-table 6-DoF trajectory recovery

## Summary

The final trajectory is produced by "'`/work/outputs/solve.py` and stored as
`/work/outputs/object_poses_world.npz` under `T_world_mesh` with shape
`(993, 4, 4)` and dtype `float64`.

The main failure mode in the supplied reconstruction is frame-wise angular
jitter, including a 28.8 degree one-frame rotation jump. The result robustly
smooths the released trajectory, searches locally for the pose that best
explains the object silhouette, refines it with rendered texture gradients
under a temporal acceleration prior, and pools the two untouched static
phases. No ground truth or challenge annotations were used.

## Phases and confidence

The static boundaries are derived from the supplied contact signal: first
touch is frame 96 and the last contact is frame 931. The object is upright and
untouched before/after these boundaries. Within the manipulation interval, the
phase boundaries below were selected from the object'"'s tilt profile and checked
visually against the video.

| Frames | Phase | Confidence | Reason |
|---|---:|---|---|
| 0-95 | Static upright, start | High | 96-frame pooled pose; clear silhouette and stable appearance |
| 96-175 | Engage / initial roll | Medium | Mask improves clearly, but texture gradient and MoGe depth partially disagree |
| 176-399 | Lower onto side | High | Mask and texture improve; MoGe residual also improves |
| 400-519 | Flip wheels up …[+6724 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/TOOLS.md <<'MD'
# Tools and dependencies

No external repositories, model weights, or pretrained checkpoints were
downloaded for this solution. The deterministic solver needs only the supplied
inputs and the preinstalled software below.

| Tool | Version | Used by "'`solve.py` | Purpose |
|---|---:|---:|---|
| Python | 3.11 | Yes | Runtime |
| PyTorch | 2.5.1+cu124 | Yes | GPU tensors and mesh transforms |
| nvdiffrast | 0.4.0 | Yes | Rasterization for object masks and textured appearance |
| trimesh | 5.1.1 | Yes | GLB mesh and UV/texture loading |
| OpenCV (`cv2`) | 4.11.0 | Yes | Video/mask I/O, resizing, Sobel and NCC preprocessing |
| NumPy | installed with environment | Yes | Numerical arrays and SE(3) trajectory processing |
| SciPy | 1.16.3 | Yes | Savitzky-Golay filtering and rotation utilities |
| h5py | 3.16.0 | Yes | Reading `masks.h5` |
| Pillow (`PIL`) | 10.2.0 | Yes | Accessing the GLB base-color texture |

Additional supplied data inspected during validation, but not required by the
final offline solver:

- `/pipeline/depth/moge2/aligned.h5`: aligned MoGe-2 depth used only for the
  reported depth diagnostic.
- `/pipeline/inference/*.pth` and `/pipeline/foundationpose/*.pkl`: inspected
  to understand the released reconstruction; `solve.py` does not read a new
  pose output from them.

GPU use: one selected CUDA device; final measured solver runtime 315.3 s,
well below the 45-minute budget.

Downloaded tools/weights: none. …[+284 chars]
```
<details><summary>output</summary>

```
total 160K
-rw-rw-r-- 1 hanzhou hanzhou 8.0K Oct 20 16:44 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.5K Oct 20 16:44 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 125K Oct 20 16:39 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  18K Oct 20 16:34 solve.py
26f04bf6ec6fb0949eea690c00bfaa0d7277fd5860ec3716b9b8dac5d96bf834  /work/outputs/REPORT.md
1028a35b31bceb200fffe2121dfb543167ab935593ee05a5dfb37834ea13a2fb  /work/outputs/TOOLS.md
c09f5f346e8f074c5a0810ff28af709d56a1a5d8af2eb1be5c281d9c40427077  /work/outputs/solve.py
137fdc0b0a28c9de98a73d58c862bb1b89ae21c0906d342e05c555f4a4ed2983  /work/outputs/object_poses_world.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md'
s=open(p).read().replace('| 830-932 | Return upright / release |','| 830-931 | Return upright / release |')
open(p,'w').write(s)
PY
python - <<'PY'
import numpy as np, os
p='/work/outputs/object_poses_world.npz';z=np.load(p);T=z['T_world_mesh']
assert T.shape==(993,4,4) and T.dtype==np.float64 and np.isfinite(T).all()
R=T[:,:3,:3];err=np.max(np.abs(np.einsum('nij,nkj->nik',R,R)-np.eye(3)[None]));det=np.linalg.det(R)
print('keys',z.files,'shape',T.shape,'dtype',T.dtype,'orth_err',err,'det',det.min(),det.max())
for f in ['solve.py','REPORT.md','TOOLS.md','object_poses_world.npz']: print(f,os.path.getsize('/work/outputs/'+f))
PY"
```
<details><summary>output</summary>

```
TOOLS.md 1445
object_poses_world.npz 127382

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace('gt = np.empty((n_frames, EVAL_H, EVAL_W), dtype=bool)','obs_masks = np.empty((n_frames, EVAL_H, EVAL_W), dtype=bool)')
s=s.replace('gt[i] = cv2.resize(','obs_masks[i] = cv2.resize(')
s=s.replace('return gt, images','return obs_masks, images')
s=s.replace('gt, video = preload_observations(','obs_masks, video = preload_observations(')
s=s.replace('gt[i]','obs_masks[i]')
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
rg -n \"\\bgt\\b|ground truth\" /work/outputs/solve.py || true"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
[pose] iteration 1/4 mask IoU=0.8153
[pose] iteration 2/4 mask IoU=0.8542
[pose] iteration 3/4 mask IoU=0.8584
[pose] iteration 4/4 mask IoU=0.8593
[appearance] iteration 1/3
[appearance] iteration 2/3
[appearance] iteration 3/3
wrote /work/outputs/object_poses_world.npz in 274.4s

real	4m36.822s
user	5m0.411s
sys	1m38.979s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/outputs/object_poses_world.npz');T=z['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all());R=T[:,:3,:3];print(np.max(np.abs(np.einsum('nij,nkj->nik',R,R)-np.eye(3)[None])),np.linalg.det(R).min(),np.linalg.det(R).max())
PY
sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py"
```
<details><summary>output</summary>

```
137fdc0b0a28c9de98a73d58c862bb1b89ae21c0906d342e05c555f4a4ed2983  /work/outputs/object_poses_world.npz
c820cc1fb2a0f8fbf525b8d587e7ea73283de1d11a1c7303a8cc89e25abfb05d  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
for p in ['/work/outputs/REPORT.md','/work/outputs/TOOLS.md']:
 s=open(p).read().replace('315.3 seconds','274.4 seconds').replace('315.3 s','274.4 s')
 open(p,'w').write(s)
PY
rg -n \"runtime|run time|274|315\" /work/outputs/REPORT.md /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
/work/outputs/TOOLS.md:28:GPU use: one selected CUDA device; final measured solver runtime 274.4 s,
/work/outputs/REPORT.md:147:Measured runtime: 274.4 seconds on one RTX 6000 Ada GPU, with deterministic

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/outputs -maxdepth 2 -printf '%P %y %s\\n' | sort"
```
<details><summary>output</summary>

```
 d 4096
REPORT.md f 8156
TOOLS.md f 1445
__pycache__ d 4096
__pycache__/solve.cpython-311.pyc f 31063
object_poses_world.npz f 127382
solve.py f 17881

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
p=Path('/work/outputs/__pycache__')
if p.exists():
 for f in p.iterdir(): f.unlink()
 p.rmdir()
PY
find /work/outputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort"
```
<details><summary>output</summary>

```
REPORT.md 8156 bytes
TOOLS.md 1445 bytes
object_poses_world.npz 127382 bytes
solve.py 17881 bytes

```
</details>

**Agent:** Completed the task and generated all required outputs:

- `/work/outputs/object_poses_world.npz`
  - Key: `T_world_mesh`
  - Shape: `(993, 4, 4)`
  - Dtype: `float64`
- `/work/outputs/solve.py`
- `/work/outputs/REPORT.md`
- `/work/outputs/TOOLS.md`

Validation highlights:

- Mean object-mask IoU: **0.804 → 0.858**
- Mean texture-gradient NCC: **0.252 → 0.352**
- Rotation jump maximum: **28.84° → 9.52°**
- 95th-percentile rotation acceleration: **7.04° → 0.56°**
- 180° symmetry flips did not improve any frame by more than 0.02 IoU.
- Final deterministic solve runtime: **274.4 seconds** on one GPU.
- Output rigid-transform orthogonality error: **1.1e-15**.
