# T1_ds_ep024

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T18:32:19Z

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
/usr/bin/bash -c "wc -l /work/AGENT_TASK.md && sed -n '1,220p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
120 /work/AGENT_TASK.md
# Task: recover the full 6-DoF object trajectory of one monocular human-object-interaction video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose; no symmetry handling, no alignment), averaged over thresholds 1-9 cm. Rotation
and tra …[+9816 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '45,75p' /work/AGENT_TASK.md"
```
<details><summary>output</summary>

```
   smoothness / constant-velocity terms on SE(3), and rigid hand-object attachment while carried.
5. **Anchors and propagation**: find frames where the pose is unambiguous (object large, little occluded, distinctive
   view, high evidence margin between hypotheses) and propagate forward and backward from them through the ambiguous
   segments, rather than trusting every frame equally.

## Validate without ground truth
You have no ground truth and must not try to get it. Build your own checks and make decisions with them:
measures of how well a trajectory explains the video (held-out frames, mask IoU, photometric / flow residuals,
temporal smoothness, physical plausibility such as no floating or penetrating the floor/human), with a table baseline
vs. your result per phase. Adopt a change on …[+1712 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort && printf '\\nTask: '; cat /work/inputs/task.txt; printf '\\nEpisode: '; cat /work/inputs/episode.json"
```
<details><summary>output</summary>

```
camera.json 821 bytes
episode.json 67 bytes
frame0_reference_human.npz 193251 bytes
ground_plane.json 395 bytes
masks.h5 5749282 bytes
object_mesh.glb 8903692 bytes
reconstruction.npz 37835005 bytes
task.txt 130 bytes
video.mp4 20461923 bytes

Task: Walk over to the watering can on the ground and kneel next to the watering can and tip it forward as if pouring using right hand.

Episode: {
 "episode_index": 24,
 "object": "watering_can",
 "frames": 748
}
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -la /work && ls -la /work/outputs 2>/dev/null || true && ls -la /work/scratch 2>/dev/null || true && ls -la /work/tools 2>/dev/null || true'
```
<details><summary>output</summary>

```
total 40
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 22:26 .
drwxr-xr-x 1 hanzhou hanzhou   200 Oct 20 22:32 ..
drwxrwxr-x 8 hanzhou hanzhou  4096 Oct 20 22:26 .codex_home
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 22:26 .net
-rw-rw-r-- 1 hanzhou hanzhou 10592 Oct 20 22:26 AGENT_TASK.md
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 17:59 inputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 22:26 logs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 22:26 outputs
drwxrwxr-x 2 hanzhou hanzhou  4096 Oct 20 22:26 scratch
total 8
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 20 22:26 .
drwxrwxr-x 8 hanzhou hanzhou 4096 Oct 20 22:26 ..
total 8
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 20 22:26 .
drwxrwxr-x 8 hanzhou hanzhou 4096 Oct 20 22:26 ..

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import json
for f in ['camera.json','ground_plane.json']:
 p='/work/inputs/'+f
 print('\\n###',f)
 d=json.load(open(p)); print(json.dumps(d,indent=2)[:10000])
PY"
```
<details><summary>output</summary>

```

### camera.json
{
  "K": [
    [
      819.9990844726562,
      0.0,
      758.0999755859375
    ],
    [
      0.0,
      819.9990844726562,
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
      830.61328125,
      0.0,
      768.0
    ],
    [
      0.0,
      830.61328125,
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
      0.9978466339093989,
      -0.005199675795528573,
      0.06538393202693832,
      -0.04590448602684517
    ],
    [
      0.0034368592586444427,
      0.9996283187207317,
      0.027044637361207494,
      0.0051501422582485145
    ],
    [
      -0.06550025338972487,
      -0.02676168498402791,
      0.9974936235498938,
      0.040141001279571153
    ],
    [
      0.0,
       …[+609 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np, os
p='/work/inputs/masks.h5'
with h5py.File(p) as f:
 print(f.keys())
 for k in f.keys(): print(k,f[k].shape,f[k].dtype)
p='/work/inputs/reconstruction.npz'
z=np.load(p)
for k in z.files: print(k,z[k].shape,z[k].dtype)
PY"
```
<details><summary>output</summary>

```
<KeysViewHDF5 ['object', 'person']>
object (748, 1152, 1536) bool
person (748, 1152, 1536) bool
T_world_mesh (748, 4, 4) float32
T_cam_mesh_foundationpose (748, 4, 4) float32
T_cam_mesh_coconet (748, 4, 4) float32
human_vertices_world (748, 4610, 3) float32
human_keypoints_world (748, 70, 3) float32
contact_prob_right_left (748, 2) float32

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames,duration,r_frame_rate -of default=noprint_wrappers=1 /work/inputs/video.mp4'
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=24.933333
nb_frames=748

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1961
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1847
/pipeline/.stages/05_object_pose.json 1772
/pipeline/.stages/06_coconet.json 2828
/pipeline/.stages/07_refinement.json 2130
/pipeline/.stages/08_render.json 2525
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 895975256
/pipeline/depth/moge2/depth_report.json 1807
/pipeline/depth/moge2/intrinsics.pkl 79695
/pipeline/depth/moge2/raw.npy 2647130240
/pipeline/export/episode_000024/edex 553
/pipeline/export/episode_000024/wild_export.json 2792
/pipeline/foundationpose/episode_000024.pkl 214239
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 5721369 …[+548 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz')
for k in ['T_world_mesh','T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
 T=z[k]; print('\\n',k)
 c=T[:,:3,3]; print('center first/last/min/max',c[0],c[-1],c.min(0),c.max(0))
 # frame to frame transforms diffs
 d=np.linalg.norm(np.diff(c,axis=0),axis=1)
 print('trans step pct',np.percentile(d,[0,50,90,95,99,100]))
 # rot angle steps
 R=T[:,:3,:3]; Rd=np.einsum('nij,nkj->nik',R[:-1],R[1:]); ang=np.arccos(np.clip((np.trace(Rd,axis1=1,axis2=2)-1)/2,-1,1)); print('rot step deg pct',np.percentile(np.rad2deg(ang),[0,50,90,95,99,100]))
print('contacts',np.percentile(z['contact_prob_right_left'],[0,25,50,75,90,99,100],axis=0))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
p='/work/inputs/object_mesh.glb'
s=trimesh.load(p,force='scene',process=False)
g=s.to_geometry()
print(type(g),g)
print('bounds',g.bounds,'centroid',g.centroid,'extents',g.extents,'volume',g.volume,'area',g.area)
print('verts',len(g.vertices),'faces',len(g.faces),'visual',type(g.visual),g.visual.material)
print('T',s.graph.get('world'))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> <trimesh.Trimesh(vertices.shape=(297436, 3), faces.shape=(99174, 3))>
bounds [[-0.0929729  -0.05118706 -0.10288549]
 [ 0.08365388  0.0466293   0.17518532]] centroid [3.84139646e-11 3.57153647e-12 2.71758489e-11] extents [0.17662679 0.09781636 0.27807081] volume 0.001321769166252129 area 0.08691697042955163
verts 297436 faces 99174 visual <class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.material.PBRMaterial object at 0x7427a7f03390>
T (array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]]), None)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 3 -type f \\( -name '*.py' -o -name '*.sh' \\) -printf '%p\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/code/reconstruction/.venv/bin/activate_this.py
/code/reconstruction/.venv/bin/jp.py
/code/reconstruction/modules/v2d_bundlesdf/__init__.py
/code/reconstruction/modules/v2d_common/__init__.py
/code/reconstruction/modules/v2d_common/broadcast.py
/code/reconstruction/modules/v2d_common/datatypes.py
/code/reconstruction/modules/v2d_common/ffv1_sidecar.py
/code/reconstruction/modules/v2d_common/hdf5_transcode.py
/code/reconstruction/modules/v2d_common/result_bundle.py
/code/reconstruction/modules/v2d_common/utils.py
/code/reconstruction/modules/v2d_common/video.py
/code/reconstruction/modules/v2d_cusfm/__init__.py
/code/reconstruction/modules/v2d_docker/__init__.py
/code/reconstruction/modules/v2d_docker/container.py
/code/reconstruction/modules/v2d_ego_hand_reconstruction/__init__.py
/code/re …[+2033 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "for f in /pipeline/.stages/*.json /pipeline/pipeline_report.json /pipeline/depth/moge2/depth_report.json; do echo '###' "'$f; cat $f; echo; done'
```
<details><summary>output</summary>

```
### /pipeline/.stages/01_depth.json
{
  "name": "01_depth",
  "outputs": [
    {
      "mtime_ns": 1791240316053666532,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000024/depth/moge2/raw.npy",
      "size": 2647130240
    },
    {
      "mtime_ns": 1791240316143666531,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000024/depth/moge2/intrinsics.pkl",
      "size": 79695
    },
    {
      "mtime_ns": 1791240318291666523,
      "path": "/home/hanzhou/Projects/v2d/run/out/episode_000024/depth/moge2/depth_report.json",
      "size": 1807
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "/workspace/v2d_cari4d/lib/cari4d/prep/mhr_wild_depth.py",
      "--video",
      "/home/hanzhou/Projects/v2d/run/wor …[+24294 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
for p in ['/pipeline/foundationpose/episode_000024.pkl','/pipeline/mhr_init/episode_000024.pkl']:
 print('\\n###',p)
 x=pickle.load(open(p,'rb')); print(type(x))
 if isinstance(x,dict):
  for k,v in x.items():
   try: print(k,type(v),getattr(v,'shape',len(v) if hasattr(v,'__len__') else ''))
   except: print(k,type(v))
 else: print(x)
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
body_model <class 'str'> 3
frames <class 'list'> 748
kids <class 'list'> 1
metadata <class 'dict'> 16
mhr_global_rot6d <class 'numpy.ndarray'> (748, 6)
mhr_trans <class 'numpy.ndarray'> (748, 3)
mhr_body_pose_cont <class 'numpy.ndarray'> (748, 260)
mhr_hand <class 'numpy.ndarray'> (748, 108)
mhr_shape <class 'numpy.ndarray'> (748, 45)
mhr_scale <class 'numpy.ndarray'> (748, 28)
mhr_face <class 'numpy.ndarray'> (748, 72)
mhr_joints <class 'numpy.ndarray'> (748, 127, 3)
mhr_keypoints <class 'numpy.ndarray'> (748, 70, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/inference/materialized_inputs.h5','/pipeline/work/masks_k0.h5']:
 print('\\n###',p)
 with h5py.File(p) as f:
  def visit(n,o):
   if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype)
  f.visititems(visit)
PY"
```
<details><summary>output</summary>

```
K_rois (748, 3, 3) float32
bboxes (748, 4) float32
identity_json () object
input_rgbs (748, 3, 224, 224) float32
input_xyz (748, 6, 224, 224) float32
observed_human_masks (748, 224, 224) bool
observed_object_masks (748, 224, 224) bool
render_rgbs (748, 3, 224, 224) float32
render_xyz (748, 6, 224, 224) float32

### /pipeline/work/masks_k0.h5
episode_000024/000000-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000024/000000-k0.person_mask.png (1152, 1536) uint8
episode_000024/000001-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000024/000001-k0.person_mask.png (1152, 1536) uint8
episode_000024/000002-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000024/000002-k0.person_mask.png (1152, 1536) uint8
episode_000024/000003-k0.obj_rend_mask.png (1152, 1536) uint8
episode_000024/000003-k0.per …[+90800 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
K=np.array(json.load(open('/work/inputs/camera.json'))['K']); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); Tc=np.linalg.inv(Twc)
for i in range(0,748,10):
 P=Tc@T[i]; c=P[:3,3]; uv=K@c; uv=uv[:2]/uv[2]
 R=P[:3,:3]; ang=np.rad2deg(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
 print(i,'world',np.round(T[i,:3,3],3),'cam',np.round(c,3),'uv',np.round(uv,1),'rotmag',round(ang,1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as h:
 m=h['object'][:]; p=h['person'][:]
# stats every 10 frames
for i in range(0,748,10):
 ys,xs=np.where(m[i]); 
 print(i,m[i].sum(), (xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None, 'person',p[i].sum(),'ov',np.logical_and(m[i],p[i]).sum())
PY"
```
<details><summary>output</summary>

```
0 2541 (700, 828, 772, 886) person 31841 ov 0
10 2561 (700, 829, 773, 886) person 31265 ov 0
20 2540 (700, 829, 772, 886) person 31070 ov 0
30 2531 (700, 829, 772, 886) person 29873 ov 1
40 2547 (700, 829, 772, 886) person 31977 ov 0
50 2543 (700, 829, 772, 886) person 34191 ov 1
60 2549 (700, 829, 772, 886) person 34605 ov 0
70 2519 (700, 829, 773, 886) person 35447 ov 3
80 2579 (700, 829, 774, 886) person 34320 ov 1
90 2505 (700, 829, 773, 886) person 32587 ov 0
100 2516 (700, 828, 773, 886) person 31857 ov 1
110 2345 (701, 831, 773, 886) person 31206 ov 8
120 2283 (701, 831, 773, 886) person 30388 ov 11
130 2364 (701, 831, 774, 886) person 30144 ov 16
140 2100 (705, 830, 766, 885) person 30536 ov 33
150 2065 (704, 826, 765, 880) person 30590 ov 37
160 2145 (696, 816, 767, 870) person 30 …[+2856 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/vis && python - <<'PY'
import cv2,h5py,numpy as np,os
v=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
with h5py.File('/work/inputs/masks.h5') as h:
 masks=h['object'][:]
 persons=h['person'][:]
 idx=list(range(0,748,30))+[747]
 ims=[]
 for i in idx:
  v.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=v.read()
  m=masks[i].astype(np.uint8); p=persons[i].astype(np.uint8)
  overlay=im.copy(); overlay[m>0]=(0.35*overlay[m>0]+0.65*np.array([0,0,255])).astype(np.uint8)
  # object contour in yellow, person contour green
  cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(overlay,cs,-1,(0,255,255),2)
  # label on 576x432
  thumb=cv2.resize(overlay,(384,288)); cv2.putText(thumb,f'{i} a={m.sum()}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
  ims.append(thumb)
 # grid 5 cols
 rows=[]
 for j in range(0,len(ims),5):
  r=ims[j:j+5]
  while len(r)<5:r.append(np.zeros_like(ims[0]))
  rows.append(np.concatenate(r,axis=1))
 out=np.concatenate(rows,axis=0)
 cv2.imwrite('/work/scratch/vis/contact.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,92])
print('wrote',out.shape)
PY"
```
<details><summary>output</summary>

```
wrote (1728, 1920, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
v=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as h: masks=h['object'][:]; persons=h['person'][:]
idx=list(range(0,748,20))
ims=[]
for i in idx:
 v.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=v.read()
 ys,xs=np.where(masks[i]);
 if len(xs):
  cx=int(xs.mean());cy=int(ys.mean()); w=120;hgt=120
  x0=max(0,cx-w//2);y0=max(0,cy-hgt//2); crop=im[y0:y0+hgt,x0:x0+w]
 else: crop=im[720:920,620:860]
 crop=cv2.resize(crop,(300,300),interpolation=cv2.INTER_CUBIC)
 # show crop coords
 cv2.putText(crop,f'{i}',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
 ims.append(crop)
rows=[]
for j in range(0,len(ims),6):
 r=ims[j:j+6]
 while len(r)<6:r.append(np.zeros_like(ims[0]))
 rows.append(np.concatenate(r,1))
out=np.concatenate(rows,0)
cv2.imwrite('/work/scratch/vis/crops.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])
print(out.shape)
PY"
```
<details><summary>output</summary>

```
(2100, 1800, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 print(list(f.keys())[:50])
 def visit(n,o):
  if isinstance(o,h5py.Dataset): print(n,o.shape,o.dtype)
 f.visititems(visit)
PY"
```
<details><summary>output</summary>

```
['aligned', 'alignment', 'frame_names', 'raw']
aligned/front_stereo_camera_left (748,) object
alignment/front_stereo_camera_left/scale (748,) float32
alignment/front_stereo_camera_left/shift (748,) float32
alignment/front_stereo_camera_left/valid_count (748,) int32
frame_names/front_stereo_camera_left (748,) object
raw/front_stereo_camera_left (748,) object

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import pandas as pd, pickle
x=pickle.load(open('/pipeline/foundationpose/episode_000024.pkl','rb'))
for k in ['fp_selected_iou','fp_selected_visibility','fp_reliable','fp_oracle_rotation_error_deg','fp_selected_translation_distance_m','gt_object_visibility_ratio','fp_full_image_iou','fp_tolerant_precision','fp_tolerant_recall','fp_observed_visible_pixels','fp_rendered_visible_pixels']:
 v=x[k]; print(k,'min med max',float(v.min()),float(v.min()+0*v.min()) if False else float(v.mean()),float(v.max()),'pct',__import__('numpy').percentile(v,[5,25,50,75,95]))
print('metadata')
for k,v in x['metadata'].items(): print(k,v)
PY"
```
<details><summary>output</summary>

```
fp_selected_iou min med max 0.688040018081665 0.7911907434463501 0.8617810606956482 pct [0.73792382 0.77115767 0.79517692 0.81284662 0.8340427 ]
fp_selected_visibility min med max 0.7484489679336548 0.9024457335472107 1.0 pct [0.79180082 0.82680841 0.90255687 0.99185474 1.        ]
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py", line 4283, in percentile
    return _quantile_unchecked(
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py", line 4555, in _quantile_unchecked
    return _ureduce(a,
           ^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/function_base.py", line 3823, in _ureduce
    r = func(a, **kwa …[+664 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['nvdiffrast','pyrender','pytorch3d','open3d','trimesh','cv2','torch','scipy','skimage']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'ERR',e)
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
pytorch3d OK 0.7.9
open3d OK 0.20.0
trimesh OK 5.1.1
cv2 OK 4.11.0
torch OK 2.5.1+cu124
scipy OK 1.16.3
skimage OK 0.26.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  g=f[p]; print(p,type(g),g.shape,g.dtype,g)
  for i in [0,100,500]:
   x=g[i]; print(i,type(x),getattr(x,'shape',None),getattr(x,'dtype',None),np.asarray(x).shape,np.asarray(x).dtype,np.asarray(x).min() if np.asarray(x).size else None,np.asarray(x).max() if np.asarray(x).size else None)
PY"
```
<details><summary>output</summary>

```
0 <class 'numpy.ndarray'> (630083,) uint8 (630083,) uint8 0 255
100 <class 'numpy.ndarray'> (623865,) uint8 (623865,) uint8 0 255
500 <class 'numpy.ndarray'> (617460,) uint8 (617460,) uint8 0 255
raw/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (748,) object <HDF5 dataset "front_stereo_camera_left": shape (748,), type "|O">
0 <class 'numpy.ndarray'> (589267,) uint8 (589267,) uint8 0 255
100 <class 'numpy.ndarray'> (584615,) uint8 (584615,) uint8 0 255
500 <class 'numpy.ndarray'> (580810,) uint8 (580810,) uint8 0 255

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
x=pickle.load(open('/pipeline/foundationpose/episode_000024.pkl','rb'))
for k in ['fp_selected_iou','fp_selected_visibility','fp_reliable','fp_oracle_rotation_error_deg','fp_selected_translation_distance_m','gt_object_visibility_ratio','fp_full_image_iou','fp_tolerant_precision','fp_tolerant_recall','fp_observed_visible_pixels','fp_rendered_visible_pixels']:
 v=x[k].astype(float); print(k,'mean %.4g min %.4g max %.4g pct'%(v.mean(),v.min(),v.max()),np.round(np.percentile(v,[5,25,50,75,95]),4))
print('metadata')
for k,v in x['metadata'].items(): print(k,repr(v)[:1000])
PY"
```
<details><summary>output</summary>

```
fp_selected_iou mean 0.7912 min 0.688 max 0.8618 pct [0.7379 0.7712 0.7952 0.8128 0.834 ]
fp_selected_visibility mean 0.9024 min 0.7484 max 1 pct [0.7918 0.8268 0.9026 0.9919 1.    ]
fp_reliable mean 1 min 1 max 1 pct [1. 1. 1. 1. 1.]
fp_oracle_rotation_error_deg mean nan min nan max nan pct [nan nan nan nan nan]
fp_selected_translation_distance_m mean nan min nan max nan pct [nan nan nan nan nan]
gt_object_visibility_ratio mean nan min nan max nan pct [nan nan nan nan nan]
fp_full_image_iou mean 0.7912 min 0.688 max 0.8618 pct [0.7379 0.7712 0.7952 0.8128 0.834 ]
fp_tolerant_precision mean 0.9993 min 0.961 max 1 pct [0.9965 1.     1.     1.     1.    ]
fp_tolerant_recall mean 0.9702 min 0.8783 max 1 pct [0.9373 0.9613 0.9731 0.9826 0.9925]
fp_observed_visible_pixels mean 2092 min 1567 max …[+5628 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/export/episode_000024/wild_export.json && echo && cat /pipeline/work/prompts.json && echo && cat /work/inputs/episode.json'
```
<details><summary>output</summary>

```
{
  "aligned_object_extents_m": [
    0.09789853543043137,
    0.17661064863204956,
    0.2777229845523834
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
  "frame_count": 748,
  "height": 1152,
  "intrinsics": [
    [
      830.61328125,
      0.0,
      768.0
    ],
    [
      0.0,
      830.61328125,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "object_mesh_file": "/home/hanzhou/Projects/v2d/run/out/episode_000024/export/episode_000024/object_mesh/output_aligned.glb",
  "object_mesh_to_training_transform": [
  …[+2654 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_mask_test.py <<'PY'
import numpy as np, torch, trimesh, h5py, json, cv2, time
import nvdiffrast.torch as dr
D=torch.device('cuda')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D)
F=torch.tensor(mesh.faces,dtype=torch.int32,device=D)
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json')); K=np.array(j['K']); Twc=np.array(j['T_world_cam']); Tc=np.linalg.inv(Twc)
H,W=1152,1536
# camera -> clip
fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
P=torch.tensor([[2*fx/W,0,2*cx/W-1,0],[0,-2*fy/H,1-2*cy/H,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=D)
Tcc=torch.tensor(Tc,dtype=torch.float32,device=D)
glctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5') as h: masks=h['object'][:]
for i in [0,50,100,120,150,200,300,400,500,550,600,650,700,747]:
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 clip=((P@(Tcc@T[i]@Vh.T)).T)[None]
 rast,_=dr.rasterize(glctx,clip,F,resolution=[H,W])
 pred=(rast[0,:,:,3]>0).flip(0).cpu().numpy()
 gt=masks[i]
 inter=np.logical_and(pred,gt).sum(); uni=np.logical_or(pred,gt).sum()
 print(i,'iou',inter/uni,'pred',pred.sum(),'gt',gt.sum())
PY
python /work/scratch/render_mask_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_mask_test.py", line 19, in <module>
    rast,_=dr.rasterize(glctx,clip,F,resolution=[H,W])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 78, in forward
    out, out_db = _nvdiffrast_c …[+292 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/clip=((P@(Tcc@T\\[i\\]@Vh.T)).T)\\[None\\]/clip=((P@(Tcc@T[i]@Vh.T)).T)[None].contiguous()/\" /work/scratch/render_mask_test.py && python /work/scratch/render_mask_test.py"
```
<details><summary>output</summary>

```
0 iou 0.6707452725250278 pred 1965 gt 2541
50 iou 0.7396197128443927 pred 1940 gt 2543
100 iou 0.7283418267346139 pred 1893 gt 2516
120 iou 0.7412790697674418 pred 1910 gt 2283
150 iou 0.672967032967033 pred 1741 gt 2065
200 iou 0.6684611933602512 pred 1720 gt 1999
300 iou 0.5654911838790933 pred 1770 gt 1959
400 iou 0.5730684326710817 pred 1640 gt 1923
500 iou 0.48169477581242287 pred 1683 gt 1919
550 iou 0.5903420523138833 pred 1760 gt 2192
600 iou 0.7040643522438611 pred 1788 gt 2237
650 iou 0.7392062799825556 pred 1740 gt 2248
700 iou 0.6912807676261994 pred 1724 gt 2330
747 iou 0.6882352941176471 pred 1712 gt 2306

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2,json,trimesh
# rerun modified capture pred bounds via import? just extract from script quickly
exec(open('/work/scratch/render_mask_test.py').read().replace('print(i,', 'print(i,').replace(\"pred=(rast[0,:,:,3]>0).flip(0).cpu().numpy()\", \"pred=(rast[0,:,:,3]>0).flip(0).cpu().numpy(); ys,xs=np.where(pred); print(' pred bbox',xs.min(),ys.min(),xs.max(),ys.max(),'cent',xs.mean(),ys.mean())\"))
PY"
```
<details><summary>output</summary>

```
 pred bbox 704 825 775 878 cent 728.7699745547073 851.9236641221374
0 iou 0.6707452725250278 pred 1965 gt 2541
 pred bbox 705 828 775 882 cent 729.8092783505155 855.7592783505155
50 iou 0.7396197128443927 pred 1940 gt 2543
 pred bbox 706 828 773 881 cent 729.6439513998944 855.1278394083465
100 iou 0.7283418267346139 pred 1893 gt 2516
 pred bbox 706 829 774 883 cent 729.8340314136126 856.3382198952879
120 iou 0.7412790697674418 pred 1910 gt 2283
 pred bbox 707 818 766 875 cent 726.4480183802413 847.807007466973
150 iou 0.672967032967033 pred 1741 gt 2065
 pred bbox 707 776 772 830 cent 729.4697674418604 806.163953488372
200 iou 0.6684611933602512 pred 1720 gt 1999
 pred bbox 697 779 751 854 cent 720.6418079096045 807.8853107344632
300 iou 0.5654911838790933 pred 1770 gt 1959
 pred bbox 696  …[+764 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual
print(v.uv.shape,v.uv.min(0),v.uv.max(0),v.uv.dtype)
mat=v.material
print(type(mat),mat)
for k in ['baseColorTexture','baseColorFactor','metallicFactor','roughnessFactor','image','name']:
 print(k,getattr(mat,k,None))
print('image',type(mat.image),getattr(mat.image,'size',None),getattr(mat.image,'mode',None))
print('face uv',v.uv.shape,'verts',len(m.vertices),'faces',len(m.faces))
print('scene?',m)
PY"
```
<details><summary>output</summary>

```
(297436, 2) [0.003587   0.00575203] [0.99592298 0.99668199] float64
<class 'trimesh.visual.material.PBRMaterial'> <trimesh.visual.material.PBRMaterial object at 0x7f47a17d4850>
baseColorTexture <PIL.Image.Image image mode=RGB size=1388x1428 at 0x7F47A41C1250>
baseColorFactor None
metallicFactor 0.0
roughnessFactor None
image None
name _texture
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.c_[m.vertices,np.ones(len(m.vertices))]
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']); n,d=pl[:3],pl[3]
for i in [0,50,100,120,140,160,180,200,240,300,400,500,520,550,600,650,700,747]:
 W=(T[i]@V.T).T[:,:3]; sd=W@n+d
 print(i,'plane dist min q5 med max',np.round([sd.min(),np.percentile(sd,5),np.median(sd),sd.max()],4),'centerd',round(n@T[i,:3,3]+d,4))
PY"
```
<details><summary>output</summary>

```
0 plane dist min q5 med max [0.0051 0.0098 0.1011 0.1846] centerd 0.0906
50 plane dist min q5 med max [-0.0084 -0.0048  0.0868  0.1698] centerd 0.0763
100 plane dist min q5 med max [-0.026  -0.0233  0.0685  0.1511] centerd 0.0581
120 plane dist min q5 med max [-0.0171 -0.0146  0.0772  0.161 ] centerd 0.0669
140 plane dist min q5 med max [-0.0295 -0.0255  0.0658  0.1536] centerd 0.0559
160 plane dist min q5 med max [0.0373 0.0516 0.1392 0.2364] centerd 0.1321
180 plane dist min q5 med max [0.0808 0.0947 0.1829 0.2789] centerd 0.1753
200 plane dist min q5 med max [0.118  0.1354 0.212  0.3261] centerd 0.2142
240 plane dist min q5 med max [0.0684 0.1117 0.2143 0.3242] centerd 0.212
300 plane dist min q5 med max [0.0599 0.1069 0.215  0.3236] centerd 0.2126
400 plane dist min q5 med max [0.0547  …[+575 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; kp=z['human_keypoints_world']; cp=z['contact_prob_right_left']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
# sample vertices every 10
V=m.vertices[::10]; Vh=np.c_[V,np.ones(len(V))]
for i in range(0,748,20):
 W=(T[i]@Vh.T).T[:,:3]
 # nearest any hand kp vs object
 dh=np.linalg.norm(kp[i,21:41,None,:]-W[None,:,:],axis=2).min()
 dl=np.linalg.norm(kp[i,42:62,None,:]-W[None,:,:],axis=2).min()
 # nearest hand bones? min kp to object vertex already
 cent=T[i,:3,3]
 dr=np.linalg.norm(kp[i,21:41]-cent,axis=1).min(); dlc=np.linalg.norm(kp[i,42:62]-cent,axis=1).min()
 print(i,'contact',np.round(cp[i],3),'right/left surf',round(dh,3),round(dl,3),'cent',round(drc if False else dr,3),round(dlc,3))
PY"
```
<details><summary>output</summary>

```
0 contact [0. 0.] right/left surf 0.291 0.725 cent 0.374 0.784
20 contact [0. 0.] right/left surf 0.176 0.646 cent 0.27 0.705
40 contact [0. 0.] right/left surf 0.175 0.642 cent 0.281 0.704
60 contact [0. 0.] right/left surf 0.159 0.621 cent 0.265 0.685
80 contact [0. 0.] right/left surf 0.152 0.602 cent 0.256 0.665
100 contact [0. 0.] right/left surf 0.045 0.61 cent 0.149 0.674
120 contact [0.    0.997] right/left surf 0.025 0.623 cent 0.1 0.687
140 contact [0. 1.] right/left surf 0.01 0.614 cent 0.09 0.683
160 contact [0. 1.] right/left surf 0.001 0.569 cent 0.074 0.631
180 contact [0. 1.] right/left surf 0.002 0.53 cent 0.08 0.588
200 contact [0. 1.] right/left surf 0.001 0.46 cent 0.07 0.51
220 contact [0. 1.] right/left surf 0.001 0.508 cent 0.053 0.558
240 contact [0. 1.] right/left  …[+1665 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_texture_test.py <<'PY'
import numpy as np, torch, trimesh, h5py, json, cv2, time
import nvdiffrast.torch as dr
D=torch.device('cuda')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D); F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous()
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=D)
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255
tex=torch.tensor(tex,device=D).permute(2,0,1)[None]
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json')); K=np.array(j['K']); Twc=np.array(j['T_world_cam']); Tc=np.linalg.inv(Twc)
H,W=1152,1536; fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
P=torch.tensor([[2*fx/W,0,2*cx/W-1,0],[0,-2*fy/H,1-2*cy/H,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=D)
Tcc=torch.tensor(Tc,dtype=torch.float32,device=D)
glctx=dr.RasterizeCudaContext()
def render(i,vflip=True):
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 clip=((P@(Tcc@T[i]@Vh.T)).T)[None].contiguous()
 rast,_=dr.rasterize(glctx,clip,F,resolution=[H,W])
 uv,_=dr.interpolate(UV[None],rast,F)
 mask=(rast[0,:,:,3]>0)
 # flip image y
 rgb=torch.nn.functional.grid_sample(tex,uv.permute(0,2,1)[:,:,None,:],mode='bilinear',align_corners=False)
 rgb=rgb[0].permute(1,2,0)
 rgb=rgb.flip(0); mask=mask.flip(0); uv=uv.flip( …[+1090 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_texture_test.py", line 31, in <module>
    v.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=v.read(); rend,m=render(i)
                                                             ^^^^^^^^^
  File "/work/scratch/render_texture_test.py", line 22, in render
    rgb=torch.nn.functional.grid_sample(tex,uv.permute(0,2,1)[:,:,None,:],mode='bilinear',align_corners=False)
                                            ^^^^^^^^^^^^^^^^^
RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 4 is not equal to len(dims) = 3

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_texture_test.py'
s=open(p).read();s=s.replace(\"rgb=torch.nn.functional.grid_sample(tex,uv.permute(0,2,1)[:,:,None,:],mode='bilinear',align_corners=False)\",\"grid=uv*2-1; rgb=torch.nn.functional.grid_sample(tex,grid,mode='bilinear',align_corners=False)\")
s=s.replace(\"rgb=rgb[0].permute(1,2,0)\\nrgb=rgb.flip(0); mask=mask.flip(0); uv=uv.flip(1) # HACK x? no\",\"rgb=rgb[0].permute(1,2,0)\\nrgb=rgb.flip(0); mask=mask.flip(0)\")
open(p,'w').write(s)
PY
python /work/scratch/render_texture_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_texture_test.py", line 33, in <module>
    x0=max(0,cx0-rad);y0=max(0,cy0-rad); x1=min(W,cx0+rad);y1=min(H,cy0+rad)
                               ^^^
NameError: name 'cy0' is not defined. Did you mean: 'cy'?

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/cy=int(ys.mean())/cy0=int(ys.mean())/' /work/scratch/render_texture_test.py && python /work/scratch/render_texture_test.py"
```
<details><summary>output</summary>

```
(1040, 1560, 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image
im=Image.open('/work/inputs/object_mesh.glb')
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/opt/conda/lib/python3.11/site-packages/PIL/Image.py", line 3309, in open
    raise UnidentifiedImageError(msg)
PIL.UnidentifiedImageError: cannot identify image file '/work/inputs/object_mesh.glb'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
m.visual.material.baseColorTexture.save('/work/scratch/vis/texture.png')
print(m.visual.material.baseColorTexture.size)
PY"
```
<details><summary>output</summary>

```
(1388, 1428)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as h: m=h['object'][:]
for a,b in [(0,100),(0,110),(0,120),(650,748),(640,748),(660,748)]:
 avg=m[a:b].mean(0); med=(avg>.5)
 ious=[]
 for i in range(a,b):
  x=m[i]; ious.append(np.logical_and(x,med).sum()/np.logical_or(x,med).sum())
 print(a,b,'area med',med.sum(),'ious',np.percentile(ious,[0,5,50,95,100]))
PY"
```
<details><summary>output</summary>

```
0 100 area med 2541 ious [0.94945713 0.96152274 0.99003557 0.99411083 0.99607227]
0 110 area med 2540 ious [0.92806604 0.95133663 0.98977786 0.99333318 0.99489596]
0 120 area med 2538 ious [0.89685925 0.91251766 0.98869175 0.99295447 0.99489395]
650 748 area med 2271 ious [0.96003401 0.96597384 0.97338812 0.98654223 0.99036356]
640 748 area med 2274 ious [0.96047599 0.96563192 0.97379278 0.98628248 0.98993435]
660 748 area med 2270 ious [0.96044236 0.96676877 0.97584239 0.98627046 0.98992554]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as h: m=h['object'][:]; p=h['person'][:]
areas=m.sum((1,2)); ov=(m&p).sum((1,2));
cent=np.array([[np.where(x)[1].mean(),np.where(x)[0].mean()] if x.any() else [np.nan,np.nan] for x in m])
# rolling area/cent
for i in range(0,748,10):
 s=slice(max(0,i-5),min(748,i+6)); print(i,round(areas[s].mean(),1),round(ov[s].mean(),1),np.round(cent[s].mean(0),2))
PY"
```
<details><summary>output</summary>

```
0 2631.7 0.0 [729.08 856.08]
10 2579.9 0.5 [728.8  856.23]
20 2561.0 0.4 [728.73 856.3 ]
30 2532.8 1.7 [728.46 856.35]
40 2536.9 0.5 [728.42 856.41]
50 2542.2 0.9 [728.42 856.4 ]
60 2551.5 1.1 [728.65 856.36]
70 2531.9 2.3 [728.6  856.33]
80 2554.5 1.0 [728.89 856.3 ]
90 2535.3 0.4 [728.71 856.27]
100 2512.1 1.4 [728.4 856.4]
110 2396.0 12.6 [729.47 857.58]
120 2331.3 17.6 [730.12 858.18]
130 2335.1 15.0 [730.53 858.13]
140 2135.1 27.5 [729.81 857.11]
150 2085.0 45.1 [727.25 852.51]
160 2179.4 36.0 [722.21 843.18]
170 2217.6 47.7 [721.4  833.77]
180 2037.8 27.2 [726.65 822.89]
190 2036.2 36.5 [729.5  816.77]
200 1967.9 72.3 [728.99 812.61]
210 2128.9 125.4 [724.52 813.65]
220 2122.0 62.7 [720.61 815.03]
230 2106.7 104.2 [719.22 812.69]
240 2040.8 33.1 [716.84 811.57]
250 1947.7 52.5 [716.3 …[+1586 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  x=f[p][0]; print(p,x[:40],len(x)); print('unique',np.unique(x)[:20])
 print('scale',f['alignment/front_stereo_camera_left/scale'][:5],f['alignment/front_stereo_camera_left/shift'][:5])
 print('frame name',f['frame_names/front_stereo_camera_left'][:2])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0   0   0   4 128  16   0   0   0   0  76  58   5   5   0   1   0
   0  73  68  65] 630083
unique [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
raw/front_stereo_camera_left [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0   0   0   4 128  16   0   0   0   0  76  58   5   5   0   1   0
   0  73  68  65] 589267
unique [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
scale [1.0810959 1.0802394 1.0752708 1.0688887 1.0756558] [0. 0. 0. 0. 0.]
frame name [b'000000' b'000001']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  x=f[p][0]; im=cv2.imdecode(x,cv2.IMREAD_UNCHANGED); print(p,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[0,1,50,99,100]))
  cv2.imwrite('/work/scratch/vis/'+p.split('/')[0]+'.png',im)
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (1152, 1536) uint16 1793 9807 [1793. 1880. 4085. 8848. 9807.]
raw/front_stereo_camera_left (1152, 1536) uint16 1659 9072 [1659. 1739. 3779. 8185. 9072.]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); Tc=np.linalg.inv(np.array(cam['T_world_cam']))
with h5py.File('/work/inputs/masks.h5') as h: masks=h['object'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f: deps=f['aligned/front_stereo_camera_left']
for i in [0,50,100,150,200,300,400,500,550,600,650,700,747]:
 dep=cv2.imdecode(deps[i],cv2.IMREAD_UNCHANGED).astype(float)/1000
 m=masks[i]; md=dep[m]; c=(Tc@T[i])[:3,3]
 print(i,'depth pct',np.round(np.percentile(md,[5,25,50,75,95]),3),'pose z',round(c[2],3),'mean',round(md.mean(),3))
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
import h5py,cv2,numpy as np,json
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; cam=json.load(open('/work/inputs/camera.json')); Tc=np.linalg.inv(np.array(cam['T_world_cam']))
for i in [0,50,100,150,200,300,400,500,550,600,650,700,747]:
 with h5py.File('/work/inputs/masks.h5') as f: m=f['object'][i]
 with h5py.File('/pipeline/depth/moge2/aligned.h5') as f: dep=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
 md=dep[m]; c=(Tc@T[i])[:3,3]
 print(i,'depth pct',np.round(np.percentile(md,[5,25,50,75,95]),3),'pose z',round(c[2],3),'mean',round(md.mean(),3))
PY"
```
<details><summary>output</summary>

```
0 depth pct [3.002 3.015 3.035 3.081 3.337] pose z 2.953 mean 3.077
50 depth pct [2.857 2.872 2.896 2.959 3.162] pose z 2.953 mean 2.933
100 depth pct [2.921 2.938 2.96  3.011 3.218] pose z 2.999 mean 2.994
150 depth pct [2.928 2.941 2.963 3.    3.179] pose z 3.011 mean 2.99
200 depth pct [2.92  2.946 2.994 3.088 3.286] pose z 3.059 mean 3.036
300 depth pct [3.018 3.05  3.085 3.138 3.216] pose z 3.051 mean 3.098
400 depth pct [2.954 3.002 3.044 3.105 3.191] pose z 3.114 mean 3.057
500 depth pct [3.013 3.116 3.15  3.195 3.263] pose z 3.107 mean 3.15
550 depth pct [2.874 2.889 2.909 2.975 3.179] pose z 2.994 mean 2.951
600 depth pct [2.875 2.886 2.903 2.934 3.164] pose z 2.973 mean 2.935
650 depth pct [2.859 2.875 2.897 2.941 3.115] pose z 2.988 mean 2.925
700 depth pct [3.064 3.079 3.099 3. …[+104 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh']; c=T[:,:3,3]; R=T[:,:3,:3]
Rd=np.einsum('nij,nkj->nik',R[:-1],R[1:]); ang=np.rad2deg(np.arccos(np.clip((np.trace(Rd,axis1=1,axis2=2)-1)/2,-1,1))); dt=np.linalg.norm(np.diff(c,axis=0),axis=1)
print('top rot')
for i in np.argsort(ang)[-40:][::-1]: print(i,i+1,round(ang[i],2),round(dt[i],4))
print('top trans')
for i in np.argsort(dt)[-40:][::-1]: print(i,i+1,round(dt[i],4),round(ang[i],2))
# smoothed speed windows and values chunks
for a in range(0,748,10):
 print(a, 'mean rot %.2f max %.2f mean dt %.4f max %.4f'%(ang[a:min(a+10,len(ang))].mean(),ang[a:min(a+10,len(ang))].max(),dt[a:min(a+10,len(dt))].mean(),dt[a:min(a+10,len(dt))].max()))
PY"
```
<details><summary>output</summary>

```
top rot
216 217 10.94 0.0923
549 550 10.09 0.0151
513 514 8.67 0.0099
220 221 7.48 0.007
511 512 7.11 0.0134
163 164 6.65 0.067
510 511 6.5 0.0101
176 177 6.26 0.0469
567 568 5.6 0.0595
222 223 5.47 0.018
539 540 5.46 0.0152
512 513 5.28 0.0146
547 548 5.21 0.0164
686 687 5.15 0.0589
161 162 5.14 0.0467
568 569 5.07 0.0404
154 155 4.94 0.0039
224 225 4.89 0.0515
546 547 4.86 0.0221
711 712 4.83 0.0085
152 153 4.7 0.0204
685 686 4.59 0.0816
710 711 4.42 0.01
707 708 4.4 0.0044
223 224 4.4 0.0576
508 509 4.25 0.0116
9 10 4.17 0.0895
535 536 4.15 0.0101
151 152 4.11 0.0178
225 226 4.09 0.0124
173 174 4.04 0.0604
217 218 3.92 0.0167
666 667 3.9 0.0029
505 506 3.83 0.0057
519 520 3.81 0.0042
556 557 3.78 0.0306
419 420 3.77 0.0426
532 533 3.77 0.0704
517 518 3.75 0.0135
150 151 3.67 0.0081
top  …[+4753 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cp /work/scratch/render_texture_test.py /work/scratch/texflip.py && python - <<'PY'
p='/work/scratch/texflip.py';s=open(p).read();
# replace loop and grid
s=s.replace(\"grid=uv*2-1; rgb=torch.nn.functional.grid_sample(tex,grid,mode='bilinear',align_corners=False)\",\"grid=uv*2-1; grid=torch.stack([grid[...,0],1-grid[...,1]],-1); rgb=torch.nn.functional.grid_sample(tex,grid,mode='bilinear',align_corners=False)\")
s=s.replace(\"for i in [0,100,120,160,200,300,400,500,550,600,650,700]:\",\"for i in [0,50,100,160,300,400,500,650]:\")
s=s.replace(\"cv2.imwrite('/work/scratch/vis/render_compare.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,96])\",\"cv2.imwrite('/work/scratch/vis/render_compare_flip.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,96])\")
open(p,'w').write(s)
PY
python /work/scratch/texflip.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/texflip.py", line 43, in <module>
    out=np.concatenate(rows,0)
        ^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 1, the array at index 0 has size 1560 and the array at index 2 has size 1040

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/for i in \\[0,50,100,160,300,400,500,650\\]:/for i in [0,50,100,160,300,400,500,650,747]:/\" /work/scratch/texflip.py && python /work/scratch/texflip.py"
```
<details><summary>output</summary>

```
(780, 1560, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_test.py <<'PY'
import numpy as np, torch, trimesh, h5py, json, cv2, time, math
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
D=torch.device('cuda'); torch.manual_seed(0)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
# decimate? no
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D)
F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous()
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=D)
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255
tex=torch.tensor(tex,device=D).permute(2,0,1)[None]
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json')); K=np.array(j['K']); Twc=np.array(j['T_world_cam']); Tc=np.linalg.inv(Twc); Tcc=torch.tensor(Tc,dtype=torch.float32,device=D)
H,W=1152,1536; fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
glctx=dr.RasterizeCudaContext()

def rodrigues(r):
 th=torch.linalg.norm(r)+1e-12; k=r/th; Kx=torch.zeros(3,3,device=r.device); Kx[0,1]=-k[2];Kx[0,2]=k[1];Kx[1,0]=k[2];Kx[1,2]=-k[0];Kx[2,0]=-k[1];Kx[2,1]=k[0]
 return torch.eye(3,device=r.device)+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)

def render_pose(Tw,x0,y0,w,h,ss=2):
 R=Tw[:3,:3]; t=Tw[:3,3]
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 P=torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,-2*fy/h,1-2*(cy-y0)/h,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,dev …[+3205 chars]
```
<details><summary>output</summary>

```

FRAME 0
0 dice 0.8039900064468384 photo 0.010397854261100292 rdeg 0.39695626717213994 dt [ 0.004 -0.004  0.004]
20 dice 0.7984803318977356 photo -0.03800579905509949 rdeg 0.8435986921006847 dt [ 0.01271885  0.00946889 -0.01308525]
40 dice 0.6810745596885681 photo -0.0667925551533699 rdeg 1.5156088382637802 dt [ 0.02480609  0.02835374 -0.03242811]
60 dice 0.6618925333023071 photo -0.005584642756730318 rdeg 2.8772011796120043 dt [ 0.01900167  0.03685536 -0.03840255]
79 dice 0.5625882148742676 photo 0.0034432392567396164 rdeg 4.157829419034249 dt [ 0.0315903   0.04940902 -0.04892084]
FINAL 0 0.5653564929962158 r 4.157829419034249 dt [ 0.0315903   0.04940902 -0.04892084] Tw [[ 0.05024425  0.6237328   0.78002083  0.07066949]
 [ 0.996037   -0.08868335  0.00675573  1.1832713 ]
 [ 0.07338863  0.7 …[+7169 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import opt_test
for fr in [0,50,100,160,300,400,500,650,700]:
 print('\\nFRAME',fr); opt_test.optimize(fr,steps=100,lr=0.001,use_photo=False)
PY"
```
<details><summary>output</summary>

```

FRAME 0
0 dice 0.8039900064468384 photo 0.010397854261100292 rdeg 0.0 dt [0. 0. 0.]
20 dice 0.8039900064468384 photo 0.010397854261100292 rdeg 0.0 dt [0. 0. 0.]
40 dice 0.8039900064468384 photo 0.010397854261100292 rdeg 0.0 dt [0. 0. 0.]
60 dice 0.8039900064468384 photo 0.010397854261100292 rdeg 0.0 dt [0. 0. 0.]
80 dice 0.8039900064468384 photo 0.010397854261100292 rdeg 0.0 dt [0. 0. 0.]
99 dice 0.8039900064468384 photo 0.010397854261100292 rdeg 0.0 dt [0. 0. 0.]
FINAL 0 0.8039465546607971 r 0.0 dt [0. 0. 0.] Tw [[-0.01129406  0.6496513   0.7601482   0.03907919]
 [ 0.99559885 -0.06342328  0.06899623  1.1338623 ]
 [ 0.09303457  0.75758207 -0.6460758   2.9643936 ]
 [ 0.          0.          0.          1.        ]]

FRAME 50
0 dice 0.8556941747665405 photo -0.034561432898044586 rdeg 0.0 dt …[+5868 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_test.py';s=open(p).read();
s=s.replace(\"rast,_=dr.rasterize(glctx,clip,F,resolution=[Hr,Wr])\\n uv,_=dr.interpolate(UV[None],rast,F)\",\"rast,_=dr.rasterize(glctx,clip,F,resolution=[Hr,Wr])\\n alpha=dr.antialias(rast[...,3:4],clip,F,[Hr,Wr])\\n uv,_=dr.interpolate(UV[None],rast,F)\")
s=s.replace(\"mask=(rast[0,:,:,3]>0).float()\",\"mask=alpha[0,:,:,0]\")
# note mask flips rows, alpha and rgb same
open('/work/scratch/opt_test_aa.py','w').write(s)
PY
python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import opt_test_aa as o
for fr in [0,50,100,160,300,400,500,650,700]:
 print('\\nFRAME',fr);o.optimize(fr,steps=80,lr=0.001,use_photo=False)
PY"
```
<details><summary>output</summary>

```

FRAME 0
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/work/scratch/opt_test_aa.py", line 65, in optimize
    rgb,alpha,_=render_pose(Tw,x0,y0,w,h)
                ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_test_aa.py", line 28, in render_pose
    alpha=dr.antialias(rast[...,3:4],clip,F,[Hr,Wr])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 517, in antialias
    assert all(isinstance(x, torch.Tensor) for x in (color, rast, pos, tri))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/dr.antialias(rast\\[...,3:4\\],clip,F,\\[Hr,Wr\\])/dr.antialias(rast[...,3:4],rast,clip,F,[Hr,Wr])/' /work/scratch/opt_test_aa.py && python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import opt_test_aa as o
for fr in [0,50,100,160,300,400,500,650,700]:
 print('\\nFRAME',fr);o.optimize(fr,steps=80,lr=0.001,use_photo=False)
PY"
```
<details><summary>output</summary>

```

FRAME 0
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/work/scratch/opt_test_aa.py", line 65, in optimize
    rgb,alpha,_=render_pose(Tw,x0,y0,w,h)
                ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_test_aa.py", line 28, in render_pose
    alpha=dr.antialias(rast[...,3:4],rast,clip,F,[Hr,Wr])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 521, in antialias
    assert isinstance(topology_hash, _nvdiffrast_c.TopologyHashWrapper)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.signature(dr.antialias));print(inspect.getsource(dr.antialias))
PY"
```
<details><summary>output</summary>

```
(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0)
def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
    """Perform antialiasing.

    All input tensors must be contiguous and reside in GPU memory. The output tensor
    will be contiguous and reside in GPU memory.

    Note that silhouette edge determination is based on vertex indices in the triangle
    tensor. For it to work properly, a vertex belonging to multiple triangles must be
    referred to using the same vertex index in each triangle. Otherwise, nvdiffrast will always
    classify the adjacent edges as silhouette edges, which leads to bad performance and
    potentially incorrect gradients. If you are unsure whether your data is good, check
    which pixels are modified by the ant …[+1248 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/,F,\\[Hr,Wr\\])/,F)/' /work/scratch/opt_test_aa.py && python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import opt_test_aa as o
for fr in [0,50,100,160,300,400,500,650,700]:
 print('\\nFRAME',fr);o.optimize(fr,steps=80,lr=0.001,use_photo=False)
PY"
```
<details><summary>output</summary>

```

FRAME 0
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
  File "/work/scratch/opt_test_aa.py", line 65, in optimize
    rgb,alpha,_=render_pose(Tw,x0,y0,w,h)
                ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_test_aa.py", line 28, in render_pose
    alpha=dr.antialias(rast[...,3:4],rast,clip,F)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 526, in antialias
    return _antialias_func.apply(color, rast, pos, tri, topology_hash, pos_gradient_boost)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kw …[+445 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/rast\\[...,3:4\\],rast,clip,F/rast[...,3:4].contiguous(),rast,clip,F/' /work/scratch/opt_test_aa.py && python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import opt_test_aa as o
for fr in [0,50,100,160,300,400,500,650,700]:
 print('\\nFRAME',fr);o.optimize(fr,steps=80,lr=0.001,use_photo=False)
PY"
```
<details><summary>output</summary>

```

FRAME 0
0 dice 1.8744666576385498 photo 0.0005904000136069953 rdeg 0.0992391935250606 dt [-0.001  0.001 -0.001]
20 dice 1.9971203804016113 photo 0.062211014330387115 rdeg 1.582049183249962 dt [-0.00968674  0.01364466 -0.01352814]
40 dice 1.99960458278656 photo 0.10817285627126694 rdeg 2.235314942453015 dt [-0.00678145  0.01351167 -0.01288666]
60 dice 1.9999600648880005 photo 0.10158612579107285 rdeg 2.5608940519316556 dt [-0.00752924  0.01299416 -0.01196873]
79 dice 1.9999603033065796 photo 0.09369564056396484 rdeg 2.6095290542469662 dt [-0.00764073  0.01286632 -0.01168307]
FINAL 0 1.9999603033065796 r 2.6095290542469662 dt [-0.00764073  0.01286632 -0.01168307] Tw [[ 0.02937987  0.6643014   0.7468869   0.03143845]
 [ 0.9947337  -0.0928336   0.04343945  1.1467285 ]
 [ 0.09819307  0.7416774 …[+7361 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch;sys.path.insert(0,'/work/scratch');import opt_test_aa as o
for fr in [0,50,100,160,300,400,500,650,700]:
 Tw=o.optimize(fr,steps=60,lr=.001,use_photo=False)
 rgb,a,_=o.render_pose(Tw,650,745,210,180)
 a=a.detach().cpu().numpy(); y,x=np.where(a>.5); gt=o.prepare(fr)[1]
 # render return is not flipped? render function flips; okay
 pred=(a>.5)
 inter=(pred&gt).sum();uni=(pred|gt).sum()
 print('CHECK',fr,'pred area',pred.sum(),'gt',gt.sum(),'iou',inter/uni,'bbox',(x.min(),y.min(),x.max(),y.max()),'alpha',a.min(),a.max(),a.mean())
PY"
```
<details><summary>output</summary>

```
0 dice 1.8744666576385498 photo 0.0005904000136069953 rdeg 0.0992391935250606 dt [-0.001  0.001 -0.001]
20 dice 1.9971203804016113 photo 0.062211014330387115 rdeg 1.582049183249962 dt [-0.00968674  0.01364466 -0.01352814]
40 dice 1.99960458278656 photo 0.10817285627126694 rdeg 2.235314942453015 dt [-0.00678145  0.01351167 -0.01288666]
59 dice 1.99996018409729 photo 0.09855955839157104 rdeg 2.5542166882443222 dt [-0.00751123  0.01300512 -0.01199123]
FINAL 0 1.9999600648880005 r 2.5542166882443222 dt [-0.00751123  0.01300512 -0.01199123] Tw [[ 0.02849505  0.66392046  0.74725986  0.03156795]
 [ 0.99471414 -0.09261127  0.04435149  1.1468674 ]
 [ 0.09865054  0.7420463  -0.66305006  2.9524024 ]
 [ 0.          0.          0.          1.        ]]
CHECK 0 pred area 2075 gt 2541 iou 0.8166076347894 …[+7210 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np;sys.path.insert(0,'/work/scratch');import opt_test_aa as o
rgb,a,_=o.render_pose(o.T[0],650,745,210,180); a=a.detach().cpu().numpy();print(np.percentile(a,[0,50,90,95,99,99.9,99.99,100]),a.sum()); print(np.percentile(a[a>0],[0,1,10,50,90,99,100]))
PY"
```
<details><summary>output</summary>

```
[    0.             0.             0.         27998.834375
 77827.2903125  89656.14257813 94835.37665156 97758.9765625 ] 125093120.0
[  963.48614502  6249.31298828 29745.32578125 63772.54296875
 82384.8484375  90870.68875    97758.9765625 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# make side by side original and rendered at 4x nearest
import sys,cv2,numpy as np,torch
sys.path.insert(0,'/work/scratch');import opt_test_aa as o
v=cv2.VideoCapture('/work/inputs/video.mp4')
ims=[]
for fr in [0,50,100,120,160,200,300,400,500,550,650,700,747]:
 v.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=v.read(); rgb,m,_=o.render_pose(o.T[fr],650,745,210,180); rgb=rgb.detach().cpu().numpy();m=m.detach().cpu().numpy()
 # alpha normalization flawed; get mask from raster via threshold? just use rgb * (rgb>0)
 rr=(np.clip(rgb,0,1)*255).astype(np.uint8)
 # actual object mask crop
 import h5py
 with h5py.File('/work/inputs/masks.h5') as h: om=h['object'][fr,745:925,650:860]
 # crop around gt
 ys,xs=np.where(om);cx=int(xs.mean());cy=int(ys.mean())
 x0=max(0,cx-50);y0=max(0,cy-50);x1=x0+100;y1=y0+100
 a=im[745+y0:745+y1,650+x0:650+x1];b=rr[y0:y1,x0:x1]
 pair=np.concatenate([a,b],1);pair=cv2.resize(pair,(400,200),interpolation=cv2.INTER_NEAREST);cv2.putText(pair,str(fr),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2);ims.append(pair)
# 4 columns
while len(ims)%4:ims.append(np.zeros_like(ims[0]))
rows=[np.concatenate(ims[i:i+4],1) for i in range(0,len(ims),4)]
out=np.concatenate(rows,0);cv2.imwrite('/work/scratch/vis/raw_render_compare.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97]);print(out.shape)
PY"
```
<details><summary>output</summary>

```
(800, 1600, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import nvdiffrast,os
print(os.path.dirname(nvdiffrast.__file__))
PY
rg -n \"antialias\\(\" /opt/conda/lib/python3.11/site-packages/nvdiffrast /workspace 2>/dev/null | head -30"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/nvdiffrast
/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py:489:def antialias(color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0):
/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py:533:    known to remain constant. This avoids reconstructing it every time `antialias()` is called.
/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py:541:        `antialias()` in the `topology_hash` argument.
/workspace/third_party/nvdiffrast/docs/index.html:983:<div class="apifunc"><h4><code>nvdiffrast.torch.antialias(<em>color</em>, <em>rast</em>, <em>pos</em>, <em>tri</em>, <em>topology_hash</em>=<span class="defarg">None</span>, <em>pos_gradient_boost</em>=<span class="defarg">1.0</span>)</code>&nbsp;<span class …[+1739 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,160p' /workspace/third_party/nvdiffrast/samples/torch/pose.py && sed -n '1,80p' /workspace/third_party/nvdiffrast/samples/torch/cube.py"
```
<details><summary>output</summary>

```
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pathlib
import numpy as np
import torch
import imageio

import util

import nvdiffrast.torch as dr

#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------

# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, …[+8495 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_test_aa.py';s=open(p).read();
old=\"rast,_=dr.rasterize(glctx,clip,F,resolution=[Hr,Wr])\\n alpha=dr.antialias(rast[...,3:4].contiguous(),rast,clip,F)\\n uv,_=dr.interpolate(UV[None],rast,F)\\n grid=uv*2-1; grid=torch.stack([grid[...,0],1-grid[...,1]],-1)\\n rgb=Fn.grid_sample(tex,grid,mode='bilinear',align_corners=False)[0].permute(1,2,0)\\n mask=alpha[0,:,:,0]\"
new=\"rast,_=dr.rasterize(glctx,clip,F,resolution=[Hr,Wr])\\n uv,_=dr.interpolate(UV[None],rast,F)\\n grid=uv*2-1; grid=torch.stack([grid[...,0],1-grid[...,1]],-1)\\n rgb=Fn.grid_sample(tex,grid,mode='bilinear',align_corners=False)[0].permute(1,2,0)\\n alpha_hard=(rast[0,:,:,3]>0).float()[...,None]\\n rgba=torch.cat([rgb*alpha_hard,alpha_hard],-1)[None]\\n rgba=dr.antialias(rgba.contiguous(),rast,clip,F)\\n rgb=rgba[0,:,:,:3]; mask=rgba[0,:,:,3]\"
assert old in s
open('/work/scratch/opt_test_aa2.py','w').write(s.replace(old,new))
PY
python - <<'PY'
import sys,numpy as np;sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
rgb,a,_=o.render_pose(o.T[0],650,745,210,180);a=a.detach().cpu().numpy();print(np.percentile(a,[0,50,75,90,95,99,99.9,100]));print('area', (a>.5).sum())
PY"
```
<details><summary>output</summary>

```
[0.         0.         0.         0.         0.53173372 1.
 1.         1.        ]
area 1918

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
for fr in [0,50,100,160,300,400,500,650,700]:
 print('\\nFRAME',fr);o.optimize(fr,steps=100,lr=.002,use_photo=False)
PY"
```
<details><summary>output</summary>

```

FRAME 0
0 dice 0.7920665144920349 photo 0.016687333583831787 rdeg 0.1984783870501212 dt [-0.002  0.002 -0.002]
20 dice 0.8700067400932312 photo 0.002834154525771737 rdeg 2.8900532477524408 dt [-0.00489182  0.01486883 -0.02208113]
40 dice 0.8703240752220154 photo 0.06228860095143318 rdeg 5.340106136059483 dt [-0.003486    0.00993571 -0.03262179]
60 dice 0.8847377300262451 photo 0.018196165561676025 rdeg 7.110466836377871 dt [-0.00386376  0.00398044 -0.04548696]
80 dice 0.8835180401802063 photo 0.03820515424013138 rdeg 8.59337091462663 dt [-0.00496971 -0.00255547 -0.05854489]
99 dice 0.8924300074577332 photo 0.02584347315132618 rdeg 9.3149069759049 dt [-0.00535771 -0.007351   -0.07051801]
FINAL 0 0.8938403129577637 r 9.3149069759049 dt [-0.00535771 -0.007351   -0.07051801] Tw [[-0.02265721  …[+8400 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/vis_opt.py <<'PY'
import sys,cv2,h5py,numpy as np,torch
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
v=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for fr in [0,50,100,160,200,300,400,500,550,650,700,747]:
 print('OPT',fr);Tw=o.optimize(fr,steps=80,lr=.002,use_photo=False)
 v.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=v.read()
 with h5py.File('/work/inputs/masks.h5') as h: om=h['object'][fr]
 color=np.zeros_like(im)
 for pose,col in [(o.T[fr],(255,0,255)),(Tw,(0,255,255))]:
  rgb,a,_=o.render_pose(pose,650,745,210,180);m=(a>.5).detach().cpu().numpy().astype(np.uint8)
  cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  for c in cs: c[:,:,0]+=650;c[:,:,1]+=745
  cv2.drawContours(im,cs,-1,col,1)
 # crop actual
 ys,xs=np.where(om);cx=int(xs.mean());cy=int(ys.mean());rad=55;x0=cx-rad;y0=cy-rad
 crop=im[y0:y0+2*rad,x0:x0+2*rad];crop=cv2.resize(crop,(330,330),interpolation=cv2.INTER_CUBIC);cv2.putText(crop,str(fr),(8,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2);ims.append(crop)
while len(ims)%4:ims.append(np.zeros_like(ims[0]))
rows=[np.concatenate(ims[i:i+4],1) for i in range(0,len(ims),4)];out=np.concatenate(rows,0);cv2.imwrite('/work/scratch/vis/opt_contours.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97]);print(out.shape)
PY
python /work/scratch/vis_opt.py"
```
<details><summary>output</summary>

```
OPT 0
0 dice 0.7920665144920349 photo 0.016687333583831787 rdeg 0.1984783870501212 dt [-0.002  0.002 -0.002]
20 dice 0.8700067400932312 photo 0.0028341582510620356 rdeg 2.8900532477524408 dt [-0.00489182  0.01486883 -0.02208113]
40 dice 0.8703240752220154 photo 0.06228860095143318 rdeg 5.340106136059483 dt [-0.003486    0.00993571 -0.03262179]
60 dice 0.8847377300262451 photo 0.018196165561676025 rdeg 7.110466836377871 dt [-0.00386376  0.00398044 -0.04548696]
79 dice 0.8839160799980164 photo 0.05003012344241142 rdeg 8.531870183792543 dt [-0.00485193 -0.00217485 -0.05797331]
FINAL 0 0.883492648601532 r 8.531870183792543 dt [-0.00485193 -0.00217485 -0.05797331] Tw [[-0.02475759  0.53855586  0.8422258   0.03422726]
 [ 0.9990114  -0.01778953  0.04074176  1.1316874 ]
 [ 0.0369245   0.842402   - …[+9893 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# extract minimal via rerun optimized quick then floor distances for representative
import sys,numpy as np,json,trimesh
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
m=o.mesh; V=np.c_[m.vertices,np.ones(len(m.vertices))]; pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n,d=pl[:3],pl[3]
for fr in [0,50,100,160,200,300,400,500,550,650,700,747]:
 Tw=o.optimize(fr,steps=60,lr=.002,use_photo=False);W=(Tw.cpu().numpy()@V.T).T[:,:3];sd=W@n+d
 Tb=o.T[fr].cpu().numpy();Wb=(Tb@V.T).T[:,:3];sdb=Wb@n+d
 print('FLOOR',fr,'base min/q5',round(sdb.min(),3),round(np.percentile(sdb,5),3),'opt',round(sd.min(),3),round(np.percentile(sd,5),3))
PY"
```
<details><summary>output</summary>

```
0 dice 0.7920665144920349 photo 0.016687333583831787 rdeg 0.1984783870501212 dt [-0.002  0.002 -0.002]
20 dice 0.8700067400932312 photo 0.0028341561555862427 rdeg 2.8900532477524408 dt [-0.00489182  0.01486883 -0.02208113]
40 dice 0.8703240752220154 photo 0.06228860095143318 rdeg 5.340106136059483 dt [-0.003486    0.00993571 -0.03262179]
59 dice 0.8847560882568359 photo 0.02024257183074951 rdeg 7.027626460142548 dt [-0.00338427  0.00436287 -0.04491958]
FINAL 0 0.8847125768661499 r 7.027626460142548 dt [-0.00338427  0.00436287 -0.04491958] Tw [[-0.00309026  0.5607776   0.82796043  0.03569492]
 [ 0.9990847  -0.03359299  0.02648152  1.1382251 ]
 [ 0.0426639   0.82728463 -0.5601606   2.9194741 ]
 [ 0.          0.          0.          1.        ]]
FLOOR 0 base min/q5 0.005 0.01 opt 0.004 0.008
 …[+8932 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/opt_all.py <<'PY'
import sys,os,time,numpy as np,torch,cv2,h5py
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
# preload ROI video and masks
x0,y0,w,h=650,745,210,180
cap=cv2.VideoCapture('/work/inputs/video.mp4'); imgs=[]
while True:
 ok,im=cap.read()
 if not ok: break
 imgs.append(im[y0:y0+h,x0:x0+w].copy())
cap.release(); imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:
 objs=f['object'][:,y0:y0+h,x0:x0+w]
 persons=f['person'][:,y0:y0+h,x0:x0+w]
print('loaded',imgs.shape,objs.shape)
N=len(imgs); out=[]; diags=[]
t0=time.time()
for fr in range(N):
 img=torch.tensor(imgs[fr].astype(np.float32)/255,device=o.D)
 target=torch.tensor(objs[fr],device=o.D).float()
 person=torch.tensor(persons[fr],device=o.D).float()
 person=torch.nn.functional.max_pool2d(person[None,None],5,1,2)[0,0]
 vis=(1-person).clamp(0,1);target=target*vis
 T0=o.T[fr].clone(); r=torch.zeros(3,device=o.D,requires_grad=True); dt=torch.zeros(3,device=o.D,requires_grad=True)
 opt=torch.optim.Adam([r,dt],lr=.002)
 for it in range(45):
  opt.zero_grad();Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=o.render_pose(Tw,x0,y0,w,h)
  pv=alpha*vis
  dice=(2*(pv*target).sum()+1)/(pv.sum()+target.sum()+1)
  loss=3*(1-dice)+0.02*(r*r).sum()+1.0*(dt*dt).sum()
  loss.backward();opt.step()
 with torch.no_grad():
  Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=o.render_pose(T …[+775 chars]
```
<details><summary>output</summary>

```
loaded (748, 180, 210, 3) (748, 180, 210)
Traceback (most recent call last):
  File "/work/scratch/opt_all.py", line 40, in <module>
    if fr%10==0: print(fr,'dice',round(float(dice0),4),'->',round(float(dice),4),'rot',round(float(np.rad2deg(r.detach().norm())),2),'dt',np.round(dt.detach().cpu().numpy(),3),'elapsed',round(time.time()-t0,1),flush=True)
                                                                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,time
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
for n in [10000,20000,40000]:
 t=time.time();s=m.simplify_quadric_decimation(face_count=n);print(n,len(s.faces),len(s.vertices),time.time()-t)
PY"
```
<details><summary>output</summary>

```
10000 10000 20933 0.24564170837402344
20000 20000 32721 0.1830759048461914
40000 40000 43892 0.19653010368347168

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/mask_opt.py <<'PY'
import numpy as np, torch, trimesh, json, cv2, h5py
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
D=torch.device('cuda')
_mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
_mesh=_mesh.simplify_quadric_decimation(face_count=12000)
V=torch.tensor(_mesh.vertices,dtype=torch.float32,device=D)
F=torch.tensor(_mesh.faces,dtype=torch.int32,device=D).contiguous()
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json')); K=np.array(j['K']); Twc=np.array(j['T_world_cam']); Tc=np.linalg.inv(Twc); Tcc=torch.tensor(Tc,dtype=torch.float32,device=D)
H,W=1152,1536; fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]
glctx=dr.RasterizeCudaContext()
def rodrigues(r):
 th=torch.linalg.norm(r)+1e-12; k=r/th; Kx=torch.zeros(3,3,device=r.device); Kx[0,1]=-k[2];Kx[0,2]=k[1];Kx[1,0]=k[2];Kx[1,2]=-k[0];Kx[2,0]=-k[1];Kx[2,1]=k[0]
 return torch.eye(3,device=r.device)+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)
def render_pose(Tw,x0,y0,w,h,ss=2):
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 P=torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,-2*fy/h,1-2*(cy-y0)/h,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=D)
 clip=((P@(Tcc@Tw@Vh.T)).T)[None].contiguous();Hr,Wr=int(h*ss),int(w*ss)
 rast,_=dr.rasterize(glctx,clip,F,resolution=[Hr,Wr])
 ah=(rast[0,:,:,3]>0).float()[...,None]
 rgba=torch.cat([ah,ah,ah,ah] …[+1816 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/work/scratch/mask_opt.py", line 32, in optimize
    alpha=render_pose(Tw,x0,y0,w,h);pv=alpha*vis
          ^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: render_pose() missing 1 required positional argument: 'ss'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'def render_pose\\|alpha=render' /work/scratch/mask_opt.py && python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import mask_opt,inspect
print(inspect.signature(mask_opt.render_pose));print(inspect.signature(mask_opt.optimize))
PY"
```
<details><summary>output</summary>

```
(Tw, x0, y0, w, h, ss=2)
(frame, img, om, pm, x0=650, y0=745, w=210, h=180, steps=35, lr=0.002)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,time,cv2,h5py,numpy as np;sys.path.insert(0,'/work/scratch');import mask_opt as m
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);_,im=cap.read()
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][0,745:925,650:860];pm=f['person'][0,745:925,650:860]
for steps in [25,35,45]:
 t=time.time();Tw,d0,d,r,dt=m.optimize(0,im,om,pm,steps=steps);print('time',time.time()-t,'dice',d0,d,'rot',np.rad2deg(r.norm().item()),'dt',dt.cpu().numpy())
PY"
```
<details><summary>output</summary>

```
time 1.8472168445587158 dice 0.11594928056001663 0.11170609295368195 rot 0.9336860057670968 dt [ 0.03323264  0.00528367 -0.0088389 ]
time 0.11512947082519531 dice 0.11594928056001663 0.10823628306388855 rot 1.348166537317506 dt [ 0.0375892   0.01277589 -0.0170585 ]
time 0.14354419708251953 dice 0.11594928056001663 0.08409743010997772 rot 1.4335559614954188 dt [ 0.05359297  0.0034143  -0.00903792]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,torch,numpy as np;sys.path.insert(0,'/work/scratch');import mask_opt as m
print('V',m.V.shape,'F',m.F.shape,'bounds',m.V.min(0).values,m.V.max(0).values)
a=m.render_pose(m.T[0],650,745,210,180);print('area',float((a>.5).sum()),'sum',float(a.sum()),'max',float(a.max()),'nonzero',int((a>0).sum()))
PY"
```
<details><summary>output</summary>

```
V torch.Size([23811, 3]) F torch.Size([12000, 3]) bounds tensor([-0.0930, -0.0512, -0.1029], device='cuda:0') tensor([0.0836, 0.0466, 0.1752], device='cuda:0')
area 27.0 sum 171.38296508789062 max 1.0 nonzero 1043

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/mask_opt.py';s=open(p).read();s=s.replace('rgba=torch.cat([ah,ah,ah,ah],-1)[None]','ones=torch.ones_like(ah); zeros=torch.zeros_like(ah); rgba=torch.cat([ones,zeros,zeros,ah],-1)[None]');open(p,'w').write(s)
PY
python - <<'PY'
import sys,torch,numpy as np;sys.path.insert(0,'/work/scratch');import mask_opt as m
a=m.render_pose(m.T[0],650,745,210,180);print('area',float((a>.5).sum()),'sum',float(a.sum()),'max',float(a.max()),'nonzero',int((a>0).sum()), np.percentile(a.detach().cpu().numpy(),[50,90,95,99]))
PY"
```
<details><summary>output</summary>

```
area 27.0 sum 171.38296508789062 max 1.0 nonzero 1043 [0.         0.         0.         0.17388294]

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
for n in [None,12000,20000]:
 s=m if n is None else m.simplify_quadric_decimation(face_count=n)
 print(n,'water',s.is_watertight,'winding',s.is_winding_consistent,'euler',s.euler_number,'bodies',len(s.split(only_watertight=False)),'components',len(s.split(only_watertight=False)))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
  File "/opt/conda/lib/python3.11/site-packages/trimesh/base.py", line 1454, in split
    return graph.split(self, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trimesh/graph.py", line 391, in split
    return mesh.submesh(
           ^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trimesh/base.py", line 2957, in submesh
    return util.submesh(
           ^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trimesh/util.py", line 1705, in submesh
    visuals.append(mesh.visual.face_subset(index))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/trimesh/visual/texture.py", line 163, in face_subset …[+560 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
for n in [None,12000,20000]:
 s=m if n is None else m.simplify_quadric_decimation(face_count=n)
 print(n,'water',s.is_watertight,'winding',s.is_winding_consistent,'euler',s.euler_number)
PY"
```
<details><summary>output</summary>

```
None water False winding True euler 99131
12000 water False winding True euler 2944
20000 water False winding True euler 575

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pytorch3d,os
print(os.path.dirname(pytorch3d.__file__))
PY
rg -n \"SoftSilhouetteShader|PerspectiveCameras\\(\" /opt/conda/lib/python3.11/site-packages/pytorch3d 2>/dev/null | head -40"
```
<details><summary>output</summary>

```
/opt/conda/lib/python3.11/site-packages/pytorch3d
/opt/conda/lib/python3.11/site-packages/pytorch3d/datasets/shapenet_base.py:140:                SoftSilhouetteShader.
/opt/conda/lib/python3.11/site-packages/pytorch3d/datasets/shapenet_base.py:157:        cameras = kwargs.get("cameras", FoVPerspectiveCameras()).to(device)
/opt/conda/lib/python3.11/site-packages/pytorch3d/vis/plotly_vis.py:217:        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
/opt/conda/lib/python3.11/site-packages/pytorch3d/vis/plotly_vis.py:237:        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
/opt/conda/lib/python3.11/site-packages/pytorch3d/vis/plotly_vis.py:251:        cameras = PerspectiveCameras(...)
/opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/points/pulsar/unified.py:366: …[+2330 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '120,190p' /opt/conda/lib/python3.11/site-packages/pytorch3d/datasets/shapenet_base.py; sed -n '260,330p' /opt/conda/lib/python3.11/site-packages/pytorch3d/renderer/mesh/shader.py"
```
<details><summary>output</summary>

```
        """
        If a list of model_ids are supplied, render all the objects by the given model_ids.
        If no model_ids are supplied, but categories and sample_nums are specified, randomly
        select a number of objects (number specified in sample_nums) in the given categories
        and render these objects. If instead a list of idxs is specified, check if the idxs
        are all valid and render models by the given idxs. Otherwise, randomly select a number
        (first number in sample_nums, default is set to be 1) of models from the loaded dataset
        and render these models.

        Args:
            model_ids: List[str] of model_ids of models intended to be rendered.
            categories: List[str] of categories intended to be rendered. categories
               …[+5483 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/p3d_test.py <<'PY'
import numpy as np,torch,trimesh,json,cv2,h5py,time
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, BlendParams
from pytorch3d.structures import Meshes
D=torch.device('cuda')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D);F=torch.tensor(mesh.faces,dtype=torch.int64,device=D)
z=np.load('/work/inputs/reconstruction.npz');T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json'));K=np.array(j['K']);Twc=np.array(j['T_world_cam']);Tc=torch.tensor(np.linalg.inv(Twc),dtype=torch.float32,device=D)
x0,y0,w,h=650,745,210,180; fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
# p3d camera: x left, y up, z forward
S=torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]],dtype=torch.float32,device=D)
focal=torch.tensor([[2*fx/w,2*fy/h]],device=D)
pp=torch.tensor([[2*(cx-x0)/w-1, 1-2*(cy-y0)/h]],device=D)
cams=PerspectiveCameras(focal_length=focal,principal_point=pp,image_size=((h,w),),device=D)
raster_settings=RasterizationSettings(image_size=(h,w),blur_radius=1e-4,faces_per_pixel=10,bin_size=0)
renderer=MeshRenderer(MeshRasterizer(cameras=cams,raster_settings=raster_settings),SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4,gamma=1e-4)))
def render_pose(Tw):
 P=Tc@Tw
 R=S@P[:3,:3];t=S@P[:3,3]
 verts=torch.matmul(V,R.T)+ …[+600 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/p3d_test.py", line 25, in <module>
    a=render_pose(T[0]); print(a.shape,a.min(),a.max(),a.mean()); pred=(a>.5);print('pred area',pred.sum().item(),'gt',om.sum().item(),'iou',(pred&om.bool()).sum().item()/(pred|om.bool()).sum().item())
      ^^^^^^^^^^^^^^^^^
  File "/work/scratch/p3d_test.py", line 22, in render_pose
    out=renderer(meshes) # [1,h,w,4]
        ^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^ …[+1760 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/p3d_test.py';s=open(p).read();s=s.replace(\"focal=torch.tensor([[2*fx/w,2*fy/h]],device=D)\",\"focal=torch.tensor([[2*fx/w,2*fy/h]],device=D,dtype=torch.float32)\");s=s.replace(\"pp=torch.tensor([[2*(cx-x0)/w-1, 1-2*(cy-y0)/h]],device=D)\",\"pp=torch.tensor([[2*(cx-x0)/w-1, 1-2*(cy-y0)/h]],device=D,dtype=torch.float32)\");s=s.replace(\"image_size=((h,w),),device=D)\",\"image_size=((h,w),),R=torch.eye(3,device=D)[None],T=torch.zeros(1,3,device=D),device=D)\");open(p,'w').write(s)
PY
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
torch.Size([180, 210]) tensor(0., device='cuda:0') tensor(0.9984, device='cuda:0') tensor(0.0505, device='cuda:0')
pred area 1945 gt 2541.0 iou 0.6165765765765766

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat >> /work/scratch/p3d_test.py <<'PY'
def rod(r):
 th=torch.linalg.norm(r)+1e-12;k=r/th;Kx=torch.zeros(3,3,device=D);Kx[0,1]=-k[2];Kx[0,2]=k[1];Kx[1,0]=k[2];Kx[1,2]=-k[0];Kx[2,0]=-k[1];Kx[2,1]=k[0];return torch.eye(3,device=D)+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)
def opt(T0,target,steps=100,lr=.002):
 r=torch.zeros(3,device=D,requires_grad=True);dt=torch.zeros(3,device=D,requires_grad=True);op=torch.optim.Adam([r,dt],lr=lr)
 for i in range(steps):
  op.zero_grad();Rd=rod(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  a=render_pose(Tw);pv=a*target.new_tensor(1.0)
  dice=2*(pv*target).sum()/(pv.sum()+target.sum()+1)
  loss=1-dice+0.02*(r*r).sum()+(dt*dt).sum();loss.backward();op.step()
 return r.detach(),dt.detach(),Tw.detach()
if __name__=='__main__':
 t=time.time();r,dt,Tw=opt(T[0],om,100);print('TIME',time.time()-t,'r',np.rad2deg(r.norm().item()),'dt',dt.cpu().numpy());a=render_pose(Tw);print('new',a.mean().item(),(a>.5).sum().item())
PY
python /work/scratch/p3d_test.py"
```
<details><summary>output</summary>

```
torch.Size([180, 210]) tensor(0., device='cuda:0') tensor(0.9984, device='cuda:0') tensor(0.0505, device='cuda:0')
pred area 1945 gt 2541.0 iou 0.6165765765765766
TIME 7.9912800788879395 r 14.112606018745693 dt [-0.0531646 -0.0089072 -0.0247014]
new 0.04839961975812912 1863

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/hard_mask.py <<'PY'
import numpy as np,torch,trimesh,json
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
D=torch.device('cuda')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().simplify_quadric_decimation(face_count=12000)
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D);F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous()
z=np.load('/work/inputs/reconstruction.npz');T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json'));K=np.array(j['K']);Tc=np.linalg.inv(np.array(j['T_world_cam']));Tcc=torch.tensor(Tc,dtype=torch.float32,device=D)
H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];glctx=dr.RasterizeCudaContext()
def render(Tw,x0=650,y0=745,w=210,h=180,ss=2):
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1);P=torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,-2*fy/h,1-2*(cy-y0)/h,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=D)
 clip=((P@(Tcc@Tw@Vh.T)).T)[None].contiguous();r,_=dr.rasterize(glctx,clip,F,resolution=[int(h*ss),int(w*ss)]);a=(r[0,:,:,3]>0).float()[None,None].flip(2)
 if ss>1:a=Fn.avg_pool2d(a,ss)
 return a[0,0]
PY
python - <<'PY'
import sys,numpy as np;sys.path.insert(0,'/work/scratch');import hard_mask as h
for fr in [0,50,100,160,300,400,500,550,650,700,747]:
 a=h.render(h.T[fr]);print(fr,float((a>.5).sum()),float(a.sum()))
PY"
```
<details><summary>output</summary>

```
0 62.0 354.25
50 50.0 363.5
100 55.0 343.75
160 54.0 376.0
300 99.0 382.25
400 86.0 361.0
500 94.0 381.0
550 58.0 343.25
650 57.0 356.0
700 57.0 332.5
747 71.0 343.75

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/hard_mask.py';s=open(p).read();s2=s.replace(\".simplify_quadric_decimation(face_count=12000)\",\"\")
open('/work/scratch/hard_mask_orig.py','w').write(s2)
PY
python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import hard_mask_orig as h,hard_mask as hs
for fr in [0,50,400]:
 a=h.render(h.T[fr]);b=hs.render(hs.T[fr]);print(fr,'orig',float((a>.5).sum()),float(a.sum()),'simp',float((b>.5).sum()),float(b.sum()))
PY"
```
<details><summary>output</summary>

```
0 orig 1924.0 1969.25 simp 62.0 354.25
50 orig 1908.0 1945.75 simp 50.0 363.5
400 orig 1612.0 1636.5 simp 86.0 361.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry()
print('processed',len(m.vertices),len(m.faces),m.bounds,m.area,m.volume)
s=m.simplify_quadric_decimation(face_count=12000);print('simp',len(s.vertices),len(s.faces),s.bounds,s.area,s.volume)
PY"
```
<details><summary>output</summary>

```
processed 54500 99174 [[-0.0929729  -0.05118706 -0.10288549]
 [ 0.08365388  0.0466293   0.17518532]] 0.08691697042955163 0.001321769166252129
simp 7211 12000 [[-0.09297083 -0.05134587 -0.1029038 ]
 [ 0.08374797  0.04660102  0.17510235]] 0.08358276050938447 0.001284478194776007

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/force='scene',process=False/force='scene',process=True/\" /work/scratch/hard_mask.py && python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import hard_mask as h
for fr in [0,50,100,160,300,400,500,550,650,700,747]:
 a=h.render(h.T[fr]);print(fr,float((a>.5).sum()),float(a.sum()))
PY"
```
<details><summary>output</summary>

```
0 1911.0 1951.5
50 1888.0 1932.5
100 1842.0 1871.25
160 1881.0 1911.25
300 1729.0 1752.0
400 1604.0 1632.0
500 1643.0 1668.5
550 1720.0 1745.5
650 1693.0 1724.5
700 1684.0 1715.5
747 1678.0 1707.25

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,time,torch;sys.path.insert(0,'/work/scratch');import hard_mask as h
with torch.no_grad():
 for ss in [1,2,4]:
  t=time.time()
  for i in range(100):a=h.render(h.T[0],ss=ss)
  print('ss',ss,'per',(time.time()-t)/100)
PY"
```
<details><summary>output</summary>

```
ss 1 per 0.0018017959594726562
ss 2 per 0.0003691840171813965
ss 4 per 0.0003563427925109863

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/pattern_opt.py <<'PY'
import numpy as np,torch,time,json,h5py
import hard_mask as hm
D=hm.D
# convex hull vertices for floor and sampled vertices for hand
import trimesh
_hull=hm.mesh.convex_hull
HV=torch.tensor(_hull.vertices,dtype=torch.float32,device=D)
# For hand use a deterministic spatial subset of hull plus all original verts subset
HVall=torch.tensor(hm.mesh.vertices,dtype=torch.float32,device=D)[::20]
pl=json.load(open('/work/inputs/ground_plane.json'))['plane'];nplane=torch.tensor(pl[:3],dtype=torch.float32,device=D);dplane=float(pl[3])
kp=np.load('/work/inputs/reconstruction.npz')['human_keypoints_world']; contact=np.load('/work/inputs/reconstruction.npz')['contact_prob_right_left'][:,1]

def se3(T0,r,dt):
 Rd=hm.dr  # no
 th=torch.linalg.norm(r)+1e-12;k=r/th;Kx=torch.zeros(3,3,device=D);Kx[0,1]=-k[2];Kx[0,2]=k[1];Kx[1,0]=k[2];Kx[1,2]=-k[0];Kx[2,0]=-k[1];Kx[2,1]=k[0]
 Rd=torch.eye(3,device=D)+torch.sin(th)*Kx+(1-torch.cos(th))*(Kx@Kx)
 Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
 return Tw

def eval_pose(fr,y,T0,target,vis,phase,prior_weight=0.005):
 r=torch.tensor(y[:3],dtype=torch.float32,device=D);dt=torch.tensor(y[3:],dtype=torch.float32,device=D)
 Tw=se3(T0,r,dt)
 a=hm.render(Tw,ss=2)
 pv=a*vis
 inter=(pv*target).sum();iou=inter/(pv.sum()+target.sum()-inter+1e-9)
 with torch.no_grad():
  # floor soft penalty only at rest-ish phases and object likely near floor (allow 2 cm penetration)
  Vw=HV@Tw[:3,:3].T+Tw[: …[+2286 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
TypeError: 'float' object is not subscriptable

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/b,_=p.eval_pose(fr,np.zeros(6),hm.T\\[fr\\],target,vis,'')\\n print(fr,'iou %.4f -> %.4f'%(b\\[1\\]/b=p.eval_pose(fr,np.zeros(6),hm.T[fr],target,vis,'')\\n print(fr,'iou %.4f -> %.4f'%(b[1]/\" /dev/null 2>/dev/null || true
python - <<'PY'
p='/work/scratch/test_pattern.py'
PY
# easier rewrite command in file
cat > /work/scratch/test_pattern.py <<'PY'
import sys,time,cv2,h5py,numpy as np,torch;sys.path.insert(0,'/work/scratch');import pattern_opt as p,hard_mask as hm
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for fr in [0,50,100,120,160,200,300,400,500,550,650,700,747]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=cap.read()
 with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][fr,745:925,650:860];pm=f['person'][fr,745:925,650:860]
 target=torch.tensor(om,device=p.D).float();person=torch.tensor(pm,device=p.D).float();person=torch.nn.functional.max_pool2d(person[None,None],5,1,2)[0,0];vis=(1-person).clamp(0,1);target=target*vis
 t=time.time();y,val=p.optimize(fr,hm.T[fr],target,vis,'');dt=time.time()-t
 b=p.eval_pose(fr,np.zeros(6),hm.T[fr],target,vis,'')[1]
 print(fr,'iou %.4f -> %.4f'%(b,p.eval_pose(fr,y,hm.T[fr],target,vis,'')[1]),'rotdeg',np.rad2deg(np.linalg.norm(y[:3])),'trans',np.round(y[3:],3),'time',round(dt,2),flush=True)
PY
python /work/scratch/test_pattern.py"
```
<details><summary>output</summary>

```
0 iou 0.6674 -> 0.8093 rotdeg 10.169666920774821 trans [-0.005  0.01  -0.022] time 0.76
50 iou 0.7434 -> 0.8235 rotdeg 10.85954278315862 trans [-0.006 -0.    -0.018] time 0.82
100 iou 0.7287 -> 0.8102 rotdeg 15.881676389921886 trans [-0.007  0.01   0.001] time 0.84
120 iou 0.7718 -> 0.8528 rotdeg 12.15952335654287 trans [-0.006  0.006 -0.007] time 0.94
160 iou 0.7343 -> 0.8588 rotdeg 8.210032666381824 trans [-0.01  0.01 -0.01] time 0.47
200 iou 0.8009 -> 0.8604 rotdeg 5.913938654581754 trans [ 0.     0.01  -0.017] time 0.51
300 iou 0.7486 -> 0.8357 rotdeg 3.276240273045105 trans [-0.01  0.01 -0.01] time 0.35
400 iou 0.7428 -> 0.8596 rotdeg 18.74599110951691 trans [-0.01   0.007 -0.041] time 0.92
500 iou 0.5747 -> 0.8308 rotdeg 10.027049578628505 trans [-0.01   0.022 -0.047] time 0.77
550 i …[+349 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,time,torch;sys.path.insert(0,'/work/scratch');import hard_mask_orig as h
with torch.no_grad():
 for ss in [1,2]:
  t=time.time()
  for i in range(100): a=h.render(h.T[0],ss=ss)
  print(ss,(time.time()-t)/100)
PY"
```
<details><summary>output</summary>

```
1 0.0017876935005187989
2 0.00046731233596801757

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/import hard_mask as hm/import hard_mask_orig as hm/' /work/scratch/pattern_opt.py > /work/scratch/pattern_opt_orig.py
sed 's/import pattern_opt as p,hard_mask as hm/import pattern_opt_orig as p,hard_mask_orig as hm/' /work/scratch/test_pattern.py > /work/scratch/test_pattern_orig.py
python /work/scratch/test_pattern_orig.py"
```
<details><summary>output</summary>

```
0 iou 0.6722 -> 0.8210 rotdeg 9.420147342081119 trans [-0.004  0.004 -0.045] time 1.03
50 iou 0.7477 -> 0.8294 rotdeg 14.098868502989085 trans [-0.007 -0.002 -0.019] time 0.81
100 iou 0.7343 -> 0.8175 rotdeg 16.762369668308686 trans [-0.007  0.007 -0.004] time 0.84
120 iou 0.7786 -> 0.8546 rotdeg 39.395512884740356 trans [-0.001  0.001 -0.021] time 0.97
160 iou 0.7344 -> 0.8645 rotdeg 6.599519748031091 trans [-0.01   0.01  -0.008] time 0.8
200 iou 0.8036 -> 0.8642 rotdeg 6.927081496233155 trans [-0.002  0.009 -0.02 ] time 0.86
300 iou 0.7493 -> 0.8583 rotdeg 10.521941452849443 trans [-0.01   0.01  -0.011] time 0.96
400 iou 0.7428 -> 0.8576 rotdeg 19.722630039018618 trans [-0.01   0.01  -0.031] time 0.8
500 iou 0.5757 -> 0.8275 rotdeg 10.182544795915335 trans [-0.01   0.022 -0.034] time 0.7 …[+355 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/powell_test.py <<'PY'
import sys,time,cv2,h5py,numpy as np,torch;sys.path.insert(0,'/work/scratch');import pattern_opt_orig as p,hard_mask_orig as hm
from scipy.optimize import minimize
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for fr in [0,50,100,120,160,200,300,400,500,550,650,700,747]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=cap.read()
 with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][fr,745:925,650:860];pm=f['person'][fr,745:925,650:860]
 target=torch.tensor(om,device=p.D).float();person=torch.tensor(pm,device=p.D).float();person=torch.nn.functional.max_pool2d(person[None,None],5,1,2)[0,0];vis=(1-person).clamp(0,1);target=target*vis
 ev=lambda y:p.eval_pose(fr,np.clip(y,[-np.deg2rad(30)]*3+[-.12]*3,[np.deg2rad(30)]*3+[.12]*3),hm.T[fr],target,vis,'')[0]
 # Note clip inside only, Powell may wander; use bounds
 t=time.time();res=minimize(lambda y:-ev(y),np.zeros(6),method='Powell',bounds=[(-.52,.52)]*3+[(-.12,.12)]*3,options={'maxfev':1200,'xtol':1e-4,'ftol':1e-5,'disp':False})
 y=res.x;dt=time.time()-t
 b=p.eval_pose(fr,np.zeros(6),hm.T[fr],target,vis,'')[1]; a=p.eval_pose(fr,y,hm.T[fr],target,vis,'')[1]
 print(fr,'iou %.4f -> %.4f'%(b,a),'rotdeg',np.rad2deg(np.linalg.norm(y[:3])),'trans',np.round(y[3:],3),'nfev',res.nfev,'time',round(dt,2),flush=True)
PY
python /work/scratch/powell_test.py"
```
<details><summary>output</summary>

```
0 iou 0.6722 -> 0.8115 rotdeg 12.060084482150717 trans [-0.003  0.012 -0.017] nfev 1126 time 1.34
50 iou 0.7477 -> 0.8166 rotdeg 19.600864114342134 trans [-0.005  0.011  0.013] nfev 492 time 0.52
100 iou 0.7343 -> 0.8180 rotdeg 20.796832342487313 trans [-0.005  0.007 -0.005] nfev 553 time 0.57
120 iou 0.7786 -> 0.8554 rotdeg 18.28587914686958 trans [-0.005  0.007 -0.002] nfev 468 time 0.83
160 iou 0.7344 -> 0.8630 rotdeg 7.878753076927884 trans [-0.01   0.012 -0.004] nfev 372 time 0.59
200 iou 0.8036 -> 0.8568 rotdeg 4.527492372553325 trans [-0.001  0.012 -0.002] nfev 281 time 0.46
300 iou 0.7493 -> 0.8490 rotdeg 7.613768555677531 trans [-0.006  0.014 -0.006] nfev 374 time 0.71
400 iou 0.7428 -> 0.8490 rotdeg 20.521782650492774 trans [-0.01   0.017 -0.016] nfev 475 time 0.56
500 iou 0.5757 …[+475 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_opt.py <<'PY'
import sys,cv2,h5py,numpy as np,torch,time
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
cap=cv2.VideoCapture('/work/inputs/video.mp4');res={}
for fr in list(range(0,121,10))+list(range(650,748,10))+[747]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=cap.read()
 with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][fr,745:925,650:860];pm=f['person'][fr,745:925,650:860]
 Tw,_,_,r,dt=o.optimize(fr,steps=70,lr=.0015,use_photo=False)
 res[fr]=Tw.detach().cpu().numpy();print(fr,np.rad2deg(r.norm().item()),dt.cpu().numpy(),flush=True)
np.savez('/work/scratch/static_opt.npz',**{str(k):v for k,v in res.items()})
# pairwise rotations against median-ish first
R0=res[0][:3,:3]
for fr in res:
 Rd=res[fr][:3,:3]@R0.T;ang=np.rad2deg(np.arccos(np.clip((np.trace(Rd)-1)/2,-1,1)));print('rel0',fr,round(ang,2),'trans',np.round(res[fr][:3,3]-res[0][:3,3],3))
PY
python /work/scratch/static_opt.py"
```
<details><summary>output</summary>

```
0 dice 0.7920665144920349 photo 0.016687333583831787 rdeg 0.14885878361748428 dt [-0.0015  0.0015 -0.0015]
20 dice 0.8677639961242676 photo 0.040789857506752014 rdeg 1.9223078633588409 dt [-0.00393859  0.01532763 -0.01959305]
40 dice 0.8729472160339355 photo 0.06888802349567413 rdeg 3.868535902964713 dt [-0.00277092  0.01036659 -0.0251834 ]
60 dice 0.8742677569389343 photo 0.0735531598329544 rdeg 5.699036418925062 dt [-0.00368457  0.00639311 -0.03446411]
69 dice 0.882270097732544 photo 0.05563028156757355 rdeg 6.4839771262685595 dt [-0.00371482  0.00454451 -0.03906394]
FINAL 0 0.8812373876571655 r 6.4839771262685595 dt [-0.00371482  0.00454451 -0.03906394] Tw [[-0.02072538  0.56807     0.822719    0.03536436]
 [ 0.9987609  -0.02547693  0.04275139  1.1384068 ]
 [ 0.04524613  0.82258576 -0.5 …[+307 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/opt_test_aa2.py'
```
<details><summary>output</summary>

```
  den=m.sum().clamp_min(1)
  # zero mean each over mask
  hpz=hp-(hp*m[...,None]).sum((0,1),keepdim=True)/den
  iz=img_hp-(img_hp*m[...,None]).sum((0,1),keepdim=True)/den
  cos=(hpz*iz*m[...,None]).sum()/torch.sqrt((hpz*hpz*m[...,None]).sum()*(iz*iz*m[...,None]).sum()+1e-6)
  loss_p=1-cos
  loss=3*loss_d+(0.2*loss_p if use_photo else 0)+0.02*(r*r).sum()+1.0*(dt*dt).sum()
  loss.backward();opt.step()
  if it%20==0 or it==steps-1:
   print(it,'dice',dice.item(),'photo',cos.item(),'rdeg',np.rad2deg(r.detach().norm().item()),'dt',dt.detach().cpu().numpy())
 with torch.no_grad():
  Rd=rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=render_pose(Tw,x0,y0,w,h);pv=alpha*vis
  inter=(pv*target).sum(); dice=2*inter/(pv.sum()+target.sum())
 print('FINAL',frame,dice …[+242 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_opt.py';s=open(p).read();s=s.replace(\"Tw,_,_,r,dt=o.optimize(fr,steps=70,lr=.0015,use_photo=False)\\n res[fr]=Tw.detach().cpu().numpy();print(fr,np.rad2deg(r.norm().item()),dt.cpu().numpy(),flush=True)\",\"Tw=o.optimize(fr,steps=70,lr=.0015,use_photo=False)\\n res[fr]=Tw.detach().cpu().numpy();print('STORE',fr,Tw.cpu().numpy(),flush=True)\")
open(p,'w').write(s)
PY
python /work/scratch/static_opt.py > /work/scratch/static_opt.log 2>&1; tail -30 /work/scratch/static_opt.log"
```
<details><summary>output</summary>

```
 [-0.01368814  0.6210362  -0.7836602   2.980913  ]
 [ 0.          0.          0.          1.        ]]
STORE 747 [[-0.05775312  0.7819355   0.6206772   0.02815257]
 [ 0.998237    0.05375187  0.02516494  1.1521486 ]
 [-0.01368814  0.6210362  -0.7836602   2.980913  ]
 [ 0.          0.          0.          1.        ]]
rel0 0 0.03 trans [0. 0. 0.]
rel0 10 2.21 trans [ 0.    -0.004 -0.019]
rel0 20 4.16 trans [ 0.001  0.002 -0.005]
rel0 30 3.71 trans [ 0.   -0.   -0.01]
rel0 40 5.04 trans [0.    0.005 0.001]
rel0 50 4.16 trans [ 0.001 -0.002 -0.014]
rel0 60 4.12 trans [-0.001  0.004 -0.001]
rel0 70 7.1 trans [ 0.     0.002 -0.004]
rel0 80 3.33 trans [0.002 0.013 0.022]
rel0 90 5.71 trans [-0.001 -0.011 -0.036]
rel0 100 5.6 trans [0.001 0.018 0.038]
rel0 110 6.21 trans [-0.001 -0.007 -0.027]
rel …[+520 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,torch,numpy as np,time;sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
for ss in [1,2,3]:
 a=o.render_pose(o.T[0],650,745,210,180,ss=ss)[1];print(ss,float((a>.5).sum()),float(a.sum()),float(a.max()))
 t=time.time()
 for i in range(20):o.render_pose(o.T[0],650,745,210,180,ss=ss)
 print('t',time.time()-t)
PY"
```
<details><summary>output</summary>

```
1 1910.0 1839.2958984375 1.0
t 0.010428190231323242
2 1918.0 1902.647705078125 1.0
t 0.010943889617919922
3 1953.0 1926.3289794921875 1.0
t 0.014636754989624023

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
for fr in [0,50,100,160,300,400,500,650,700]:
 print('\\nFRAME',fr);Tw=o.optimize(fr,steps=70,lr=.0015,use_photo=True)
PY"
```
<details><summary>output</summary>

```

FRAME 0
0 dice 0.7920665144920349 photo 0.016687333583831787 rdeg 0.14885878361748428 dt [-0.0015  0.0015 -0.0015]
20 dice 0.8673858046531677 photo 0.09575270861387253 rdeg 1.7152755448273513 dt [-0.00187255  0.01267265 -0.01677445]
40 dice 0.8733795881271362 photo 0.08841633051633835 rdeg 3.7442943350753577 dt [-0.00613161  0.00966292 -0.02272403]
60 dice 0.8724857568740845 photo 0.09916527569293976 rdeg 5.084576806378259 dt [-0.00529983  0.00539066 -0.02944761]
69 dice 0.8734917044639587 photo 0.1287156194448471 rdeg 5.4746711079924895 dt [-0.00794916  0.00418154 -0.03341584]
FINAL 0 0.873411238193512 r 5.4746711079924895 dt [-0.00794916  0.00418154 -0.03341584] Tw [[-0.01220668  0.57986194  0.8146231   0.03113002]
 [ 0.9983434  -0.03874377  0.04253807  1.1380438 ]
 [ 0.05622777  0.8137 …[+7218 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_all_photo.py <<'PY'
import sys,time,numpy as np,torch,cv2,h5py
import torch.nn.functional as Fn
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
x0,y0,w,h=650,745,210,180
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs=[]
while True:
 ok,im=cap.read()
 if not ok: break
 imgs.append(im[y0:y0+h,x0:x0+w].copy())
cap.release();imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:
 objs=f['object'][:,y0:y0+h,x0:x0+w];persons=f['person'][:,y0:y0+h,x0:x0+w]
N=len(imgs);print('loaded',imgs.shape,flush=True)
out=[];diags=[];t0=time.time()
for fr in range(N):
 img=torch.tensor(imgs[fr].astype(np.float32)/255,device=o.D)
 target=torch.tensor(objs[fr],device=o.D).float();person=torch.tensor(persons[fr],device=o.D).float()
 person=Fn.max_pool2d(person[None,None],5,1,2)[0,0];vis=(1-person).clamp(0,1);target=target*vis
 img_hp=img-o.blur(img[None])[0]
 T0=o.T[fr].clone();r=torch.zeros(3,device=o.D,requires_grad=True);dt=torch.zeros(3,device=o.D,requires_grad=True)
 opt=torch.optim.Adam([r,dt],lr=.0015)
 for it in range(55):
  opt.zero_grad();Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=o.render_pose(Tw,x0,y0,w,h)
  pv=alpha*vis
  dice=(2*(pv*target).sum()+1)/(pv.sum()+target.sum()+1)
  hp=rgb-o.blur(rgb[None])[0];m=pv*target;den=m.sum().clamp_min(1)
  hpz=hp-(hp*m[...,None]).sum((0,1),keepdim=True)/den
  iz=img_hp-(img_hp*m[...,None]).sum((0,1),keepdim=True)/den
  cos=(hpz*iz*m[ …[+1271 chars]
```
<details><summary>output</summary>

```
loaded (748, 180, 210, 3)
Traceback (most recent call last):
  File "/work/scratch/opt_all_photo.py", line 38, in <module>
    out.append(Tw.cpu().numpy().astype(np.float64));diags.append([fr,dice.item(),cos.item(),np.rad2deg(r.norm().item()),*(dt.cpu().numpy())])
                                                                                                                          ^^^^^^^^^^^^^^^^
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dt.cpu().numpy()/dt.detach().cpu().numpy()/g' /work/scratch/opt_all_photo.py && python /work/scratch/opt_all_photo.py > /work/scratch/opt_all_photo.log 2>&1; tail -20 /work/scratch/opt_all_photo.log"
```
<details><summary>output</summary>

```
loaded (748, 180, 210, 3)
Traceback (most recent call last):
  File "/work/scratch/opt_all_photo.py", line 39, in <module>
    if fr%10==0: print(fr,'dice',round(float(dice),4),'cos',round(float(cos),4),'rot',round(float(np.rad2deg(r.norm())),2),'dt',np.round(dt.detach().cpu().numpy(),3),'elapsed',round(time.time()-t0,1),flush=True)
                                                                                                  ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'dt' /work/scratch/opt_all_photo.py | tail -5"
```
<details><summary>output</summary>

```
23:  opt.zero_grad();Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
31:  loss=3*(1-dice)+0.2*(1-cos)+0.02*(r*r).sum()+1.0*(dt*dt).sum()
34:  Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
38: out.append(Tw.cpu().numpy().astype(np.float64));diags.append([fr,dice.item(),cos.item(),np.rad2deg(r.norm().item()),*(dt.detach().cpu().numpy())])
39: if fr%10==0: print(fr,'dice',round(float(dice),4),'cos',round(float(cos),4),'rot',round(float(np.rad2deg(r.norm())),2),'dt',np.round(dt.detach().cpu().numpy(),3),'elapsed',round(time.time()-t0,1),flush=True)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_all_photo.py';s=open(p).read();s=s.replace(\"out.append(Tw.cpu().numpy().astype(np.float64));diags.append([fr,dice.item(),cos.item(),np.rad2deg(r.norm().item()),*(dt.detach().cpu().numpy())])\",\"out.append(Tw.detach().cpu().numpy().astype(np.float64)); dd=dt.detach().cpu(); diags.append([fr,dice.item(),cos.item(),float(np.rad2deg(r.detach().norm().item())),float(dd[0]),float(dd[1]),float(dd[2])])\");s=s.replace(\"'dt',np.round(dt.detach().cpu().numpy(),3)\",\"'dt',[round(float(dt.detach().cpu()[i]),3) for i in range(3)]\");open(p,'w').write(s)
PY
python /work/scratch/opt_all_photo.py > /work/scratch/opt_all_photo.log 2>&1; tail -20 /work/scratch/opt_all_photo.log"
```
<details><summary>output</summary>

```
loaded (748, 180, 210, 3)
Traceback (most recent call last):
  File "/work/scratch/opt_all_photo.py", line 39, in <module>
    if fr%10==0: print(fr,'dice',round(float(dice),4),'cos',round(float(cos),4),'rot',round(float(np.rad2deg(r.norm())),2),'dt',[round(float(dt.detach().cpu()[i]),3) for i in range(3)],'elapsed',round(time.time()-t0,1),flush=True)
                                                                                                  ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -20 /work/scratch/opt_all_photo.log'
```
<details><summary>output</summary>

```
10 dice 0.8875 cos 0.0725 rot 4.6 dt [-0.006, 0.005, -0.027] elapsed 3.7
20 dice 0.8821 cos 0.0742 rot 4.59 dt [-0.005, 0.001, -0.024] elapsed 6.1
30 dice 0.8842 cos 0.0851 rot 4.93 dt [-0.006, -0.004, -0.028] elapsed 8.5
40 dice 0.8663 cos 0.0814 rot 4.89 dt [-0.004, -0.003, -0.024] elapsed 10.8
50 dice 0.8738 cos 0.1051 rot 5.14 dt [-0.008, -0.005, -0.026] elapsed 13.8
60 dice 0.8776 cos 0.1009 rot 5.68 dt [-0.004, -0.009, -0.033] elapsed 16.9
70 dice 0.8803 cos 0.0814 rot 5.09 dt [-0.005, -0.007, -0.036] elapsed 20.0
80 dice 0.8636 cos 0.0984 rot 5.31 dt [-0.005, -0.001, -0.031] elapsed 23.0
90 dice 0.8939 cos 0.0802 rot 5.04 dt [-0.007, 0.001, -0.021] elapsed 26.0
100 dice 0.8714 cos 0.0968 rot 4.76 dt [-0.006, -0.002, -0.028] elapsed 31.2
110 dice 0.9075 cos 0.0782 rot 4.41 dt [-0.005 …[+713 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -15 /work/scratch/opt_all_photo.log'
```
<details><summary>output</summary>

```
190 dice 0.9022 cos 0.0353 rot 4.11 dt [-0.005, 0.012, -0.024] elapsed 57.8
200 dice 0.9108 cos 0.0033 rot 2.35 dt [-0.001, 0.009, -0.019] elapsed 60.1
210 dice 0.9235 cos 0.0084 rot 0.77 dt [0.003, 0.013, -0.019] elapsed 62.3
220 dice 0.9106 cos 0.0382 rot 1.12 dt [-0.001, 0.015, -0.022] elapsed 64.5
230 dice 0.9017 cos 0.1244 rot 4.02 dt [-0.006, 0.006, -0.021] elapsed 66.8
240 dice 0.895 cos 0.1049 rot 4.47 dt [-0.006, 0.002, -0.018] elapsed 69.1
250 dice 0.8896 cos 0.1355 rot 3.39 dt [-0.006, 0.003, -0.02] elapsed 71.8
260 dice 0.9163 cos 0.1308 rot 4.06 dt [-0.003, 0.005, -0.014] elapsed 74.6
270 dice 0.8997 cos 0.1673 rot 4.36 dt [-0.005, 0.0, -0.011] elapsed 77.5
280 dice 0.9068 cos 0.1492 rot 3.64 dt [-0.004, 0.004, -0.012] elapsed 80.2
290 dice 0.8976 cos 0.1637 rot 4.07 dt [-0.00 …[+333 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/opt_all_photo.log'
```
<details><summary>output</summary>

```
370 dice 0.8977 cos 0.1238 rot 4.65 dt [-0.004, 0.006, -0.017] elapsed 105.9
380 dice 0.9111 cos 0.1563 rot 4.37 dt [0.0, 0.005, -0.012] elapsed 108.7
390 dice 0.8905 cos 0.1369 rot 4.73 dt [-0.008, 0.004, -0.016] elapsed 111.4
400 dice 0.8995 cos 0.1857 rot 4.02 dt [-0.003, 0.004, -0.014] elapsed 113.9
410 dice 0.8929 cos 0.175 rot 4.52 dt [-0.002, 0.003, -0.013] elapsed 116.1
420 dice 0.8888 cos 0.0482 rot 3.78 dt [-0.006, 0.011, -0.02] elapsed 118.4
430 dice 0.9113 cos 0.1657 rot 3.72 dt [-0.002, 0.006, -0.019] elapsed 120.7
440 dice 0.9063 cos 0.1312 rot 2.46 dt [0.001, 0.008, -0.016] elapsed 123.5
450 dice 0.8987 cos 0.0058 rot 2.54 dt [-0.006, 0.008, -0.02] elapsed 126.5
460 dice 0.9035 cos 0.0702 rot 2.01 dt [-0.001, 0.014, -0.022] elapsed 129.4

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/opt_all_photo.log'
```
<details><summary>output</summary>

```
470 dice 0.8812 cos -0.0745 rot 1.13 dt [-0.008, 0.016, -0.023] elapsed 132.6
480 dice 0.8933 cos -0.0486 rot 3.23 dt [-0.011, 0.015, -0.023] elapsed 135.5
490 dice 0.8997 cos -0.0004 rot 3.54 dt [-0.008, 0.018, -0.022] elapsed 138.5
500 dice 0.8917 cos -0.0588 rot 1.75 dt [-0.006, 0.023, -0.028] elapsed 141.4
510 dice 0.9081 cos -0.009 rot 1.24 dt [-0.01, 0.017, -0.02] elapsed 144.5
520 dice 0.9048 cos 0.0483 rot 1.89 dt [-0.004, 0.009, -0.02] elapsed 147.4
530 dice 0.9297 cos 0.0514 rot 5.04 dt [-0.006, 0.008, -0.017] elapsed 150.3
540 dice 0.8874 cos 0.0763 rot 4.11 dt [-0.004, 0.018, -0.029] elapsed 153.2
550 dice 0.891 cos 0.0522 rot 5.33 dt [-0.001, 0.015, -0.028] elapsed 156.2
560 dice 0.893 cos 0.0201 rot 4.99 dt [-0.003, 0.015, -0.03] elapsed 161.3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/opt_all_photo.log'
```
<details><summary>output</summary>

```
580 dice 0.8786 cos -0.0195 rot 4.0 dt [-0.006, 0.018, -0.029] elapsed 171.1
590 dice 0.8889 cos 0.0699 rot 4.22 dt [-0.007, 0.002, -0.022] elapsed 175.5
600 dice 0.8922 cos 0.0379 rot 5.46 dt [-0.007, 0.003, -0.026] elapsed 180.1
610 dice 0.9173 cos 0.028 rot 5.25 dt [-0.009, 0.006, -0.017] elapsed 182.4
620 dice 0.8989 cos 0.0728 rot 5.25 dt [-0.009, 0.003, -0.022] elapsed 184.5
630 dice 0.9019 cos -0.0172 rot 5.53 dt [-0.007, 0.008, -0.025] elapsed 186.8
640 dice 0.8865 cos 0.0597 rot 4.57 dt [-0.008, 0.003, -0.023] elapsed 189.6
650 dice 0.8961 cos 0.0816 rot 5.1 dt [-0.005, 0.001, -0.024] elapsed 192.4
660 dice 0.88 cos 0.0568 rot 5.74 dt [-0.003, -0.006, -0.03] elapsed 195.7
670 dice 0.8923 cos 0.0925 rot 5.0 dt [-0.014, -0.005, -0.032] elapsed 198.8

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/round(float(np.rad2deg(r.norm())),2)/round(float(r.detach().norm())*57.29577951308232,2)/' /work/scratch/opt_all_photo.py && python /work/scratch/opt_all_photo.py > /work/scratch/opt_all_photo.log 2>&1; tail -20 /work/scratch/opt_all_photo.log"
```
<details><summary>output</summary>

```
560 dice 0.893 cos 0.0201 rot 4.99 dt [-0.003, 0.015, -0.03] elapsed 161.3
570 dice 0.8869 cos 0.0187 rot 2.87 dt [-0.001, 0.016, -0.027] elapsed 166.5
580 dice 0.8786 cos -0.0195 rot 4.0 dt [-0.006, 0.018, -0.029] elapsed 171.1
590 dice 0.8889 cos 0.0699 rot 4.22 dt [-0.007, 0.002, -0.022] elapsed 175.5
600 dice 0.8922 cos 0.0379 rot 5.46 dt [-0.007, 0.003, -0.026] elapsed 180.1
610 dice 0.9173 cos 0.028 rot 5.25 dt [-0.009, 0.006, -0.017] elapsed 182.4
620 dice 0.8989 cos 0.0728 rot 5.25 dt [-0.009, 0.003, -0.022] elapsed 184.5
630 dice 0.9019 cos -0.0172 rot 5.53 dt [-0.007, 0.008, -0.025] elapsed 186.8
640 dice 0.8865 cos 0.0597 rot 4.57 dt [-0.008, 0.003, -0.023] elapsed 189.6
650 dice 0.8961 cos 0.0816 rot 5.1 dt [-0.005, 0.001, -0.024] elapsed 192.4
660 dice 0.88 cos 0.0568 rot 5.74 …[+681 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/scratch/opt_all_photo55.npz');T=z['T_world_mesh'];d=z['diags']
print(T.shape,d.shape)
R=T[:,:3,:3];c=T[:,:3,3]
Rd=np.einsum('nij,nkj->nik',R[:-1],R[1:]);ang=np.rad2deg(np.arccos(np.clip((np.trace(Rd,axis1=1,axis2=2)-1)/2,-1,1)));dt=np.linalg.norm(np.diff(c,axis=0),axis=1)
print('rot steps pct',np.percentile(ang,[0,50,90,95,99,100]),'trans',np.percentile(dt,[0,50,90,95,99,100]))
for i in np.argsort(ang)[-30:][::-1]:print('jump',i,i+1,round(ang[i],2),round(dt[i],4))
print('every 10')
for i in range(0,748,10): print(i,'c',np.round(c[i],3),'rotstep',round(ang[i],2) if i<747 else 0,'dice',round(d[i,1],2),'cos',round(d[i,2],2),'corrrot',round(d[i,3],1),'dt',np.round(d[i,4:7],3))
print('ranges c',c.min(0),c.max(0))
PY"
```
<details><summary>output</summary>

```
[-0.016  1.039  3.122] rotstep 1.62 dice 0.89 cos 0.05 corrrot 3.8 dt [-0.006  0.011 -0.02 ]
430 c [-0.018  1.031  3.109] rotstep 2.17 dice 0.91 cos 0.17 corrrot 3.7 dt [-0.002  0.006 -0.019]
440 c [-0.013  1.037  3.121] rotstep 2.73 dice 0.91 cos 0.13 corrrot 2.5 dt [ 0.001  0.008 -0.016]
450 c [-0.023  1.054  3.155] rotstep 1.14 dice 0.9 cos 0.01 corrrot 2.5 dt [-0.006  0.008 -0.02 ]
460 c [-0.022  1.032  3.076] rotstep 0.91 dice 0.9 cos 0.07 corrrot 2.0 dt [-0.001  0.014 -0.022]
470 c [-0.015  1.057  3.124] rotstep 1.24 dice 0.88 cos -0.07 corrrot 1.1 dt [-0.008  0.016 -0.023]
480 c [1.000e-03 1.034e+00 3.065e+00] rotstep 1.06 dice 0.89 cos -0.05 corrrot 3.2 dt [-0.011  0.015 -0.023]
490 c [0.005 1.03  3.059] rotstep 0.89 dice 0.9 cos -0.0 corrrot 3.5 dt [-0.008  0.018 -0.022]
500 c [3. …[+2478 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2,h5py,time;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
z=np.load('/work/scratch/opt_all_photo55.npz');To=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device='cuda')
x0,y0,w,h=650,745,210,180
with h5py.File('/work/inputs/masks.h5') as f: objs=f['object'][:,y0:y0+h,x0:x0+w];pers=f['person'][:,y0:y0+h,x0:x0+w]
res=[]
with torch.no_grad():
 for i in range(748):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];vis=(1-p).clamp(0,1);tar=t*vis
  vals=[]
  for T in [hm.T[i],To[i]]:
   a=hm.render(T,ss=2);pr=a*vis;inter=(pr*tar).sum();iou=inter/(pr.sum()+tar.sum()-inter+1e-9);vals.append(float(iou))
  res.append(vals)
res=np.array(res)
np.save('/work/scratch/iou_compare.npy',res)
for a,b in [(0,110),(110,140),(140,520),(520,640),(640,748),(0,748)]:
 r=res[a:b];print(a,b,'base',r[:,0].mean(),'opt',r[:,1].mean(),'delta',(r[:,1]-r[:,0]).mean(),'median',np.median(r[:,1]-r[:,0]),'worse',np.mean(r[:,1]<r[:,0]))
print('overall pct base',np.percentile(res[:,0],[0,25,50,75,100]),'opt',np.percentile(res[:,1],[0,25,50,75,100]))
# list worst opt improvements
for i in np.argsort(res[:,1]-res[:,0])[:20]:print('worse',i,res[i],res[i,1]-res[i,0])
PY"
```
<details><summary>output</summary>

```
0 110 base 0.7381813195618716 opt 0.8093643876639279 delta 0.07118306810205632 median 0.06556394696235657 worse 0.0
110 140 base 0.7760121643543243 opt 0.8352750102678935 delta 0.05926284591356913 median 0.05821087956428528 worse 0.0
140 520 base 0.7260655908208145 opt 0.8376598558927837 delta 0.11159426507196929 median 0.10230913758277893 worse 0.0
520 640 base 0.7053893655538559 opt 0.8286269883314769 delta 0.12323762277762094 median 0.103748619556427 worse 0.0
640 748 base 0.7006326803454647 opt 0.7981321441906469 delta 0.09749946384518235 median 0.0861007571220398 worse 0.0
0 748 base 0.722861347510853 opt 0.8262467758859543 delta 0.10338542837510134 median 0.09271472692489624 worse 0.0
overall pct base [0.56172717 0.69888544 0.73413202 0.75592034 0.81991231] opt [0.75789922 0.81134544 …[+1109 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh
z=np.load('/work/inputs/reconstruction.npz');kp=z['human_keypoints_world'];cp=z['contact_prob_right_left'][:,1]
o=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];b=z['T_world_mesh']
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(m.vertices[::20],dtype=torch.float32,device='cuda');K=torch.tensor(kp[:,21:41],dtype=torch.float32,device='cuda')
for name,T in [('base',b),('opt',o)]:
 T=torch.tensor(T,dtype=torch.float32,device='cuda');ds=[]
 for i in range(748):
  W=V@T[i,:3,:3].T+T[i,:3,3];d=torch.cdist(K[i],W).min().item();ds.append(d)
 ds=np.array(ds)
 print(name)
 for a,c in [(0,110),(110,140),(140,520),(520,640),(640,748)]: print(a,c,'mean',ds[a:c].mean(),'med',np.median(ds[a:c]),'p90',np.percentile(ds[a:c],90))
PY"
```
<details><summary>output</summary>

```
base
0 110 mean 0.15249049399386752 med 0.15681812167167664 p90 0.19935343712568285
110 140 mean 0.020941374901061256 med 0.02116992138326168 p90 0.031043891049921515
140 520 mean 0.010182514008882112 med 0.005167482886463404 p90 0.02611277811229229
520 640 mean 0.032296268427550484 med 0.029021216556429863 p90 0.05932712927460676
640 748 mean 0.42896493068999714 med 0.49331483244895935 p90 0.6674146890640259
opt
0 110 mean 0.16781254666434092 med 0.1720607727766037 p90 0.22265375256538392
110 140 mean 0.006164380306533227 med 0.005734649719670415 p90 0.010004873014986518
140 520 mean 0.004222646481847685 med 0.0029296875 p90 0.009427700564265253
520 640 mean 0.02403352033676735 med 0.018386843614280224 p90 0.049460845068097205
640 748 mean 0.4331047919199423 med 0.4979902058839798 p90 0.6 …[+16 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/pooled_static.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,json,trimesh
import torch.nn.functional as Fn
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
D=o.D;x0,y0,w,h=650,745,210,180
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs=[]
while True:
 ok,im=cap.read()
 if not ok:break
 imgs.append(im[y0:y0+h,x0:x0+w])
cap.release();imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w];pers=f['person'][:,y0:y0+h,x0:x0+w]
zo=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh']
# convex hull for floor
mm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry().convex_hull
HV=torch.tensor(mm.vertices,dtype=torch.float32,device=D)
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);pn=torch.tensor(pl[:3],dtype=torch.float32,device=D);pd=float(pl[3])
def rav(Rs):
 q=Rotation.from_matrix(Rs).as_quat();q0=q[0].copy()
 for i in range(len(q)):
  if np.dot(q[i],q0)<0:q[i]*=-1
 qm=q.mean(0);qm/=np.linalg.norm(qm);return Rotation.from_quat(qm).as_matrix()
def pool(a,b,steps=200):
 img=np.median(imgs[a:b].astype(np.float32)/255,axis=0)
 # visible consensus mask: average object outside dilated person per frame not needed; object static and >95% visible
 target=objs[a:b].mean(0).astype(np.float32)
 im=torch.tensor(img,device=D);tar=torch.tensor(target,device=D)
 imhp=im-o.blur(im[None])[0]
 R …[+1740 chars]
```
<details><summary>output</summary>

```

POOL initial
it 0 dice 0.8675107359886169 cos 0.08677927404642105 floor -0.004623293876647949 rdeg 0.14886973542161286 dt [-0.0014999997802078724, 0.0014999996637925506, -0.0014999997802078724]
it 40 dice 0.8787373900413513 cos 0.08850719779729843 floor 0.0060797929763793945 rdeg 3.160351065918803 dt [-0.0024195851292461157, -0.010240857489407063, -0.01834098994731903]
it 80 dice 0.8808138370513916 cos 0.07738633453845978 floor 0.017223834991455078 rdeg 4.9940817900002 dt [-0.004311253782361746, -0.020523222163319588, -0.039462748914957047]
it 120 dice 0.8898718357086182 cos 0.043870922178030014 floor 0.024074792861938477 rdeg 5.750992083549499 dt [-0.004285620525479317, -0.025852905586361885, -0.06017294526100159]
it 160 dice 0.895662784576416 cos 0.05429166182875633 floor 0.031608343124 …[+1540 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
x0,y0,w,h=650,745,210,180;zo=torch.tensor(np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'],dtype=torch.float32,device='cuda')
Ti=torch.tensor(np.load('/work/scratch/pooled_initial.npy'),dtype=torch.float32,device='cuda');Tf=torch.tensor(np.load('/work/scratch/pooled_final.npy'),dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w];pers=f['person'][:,y0:y0+h,x0:x0+w]
with torch.no_grad():
 for name,a,b,Tp in [('initial',0,110,Ti),('final',640,748,Tf)]:
  vals=[]
  for i in range(a,b):
   t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];vis=(1-p).clamp(0,1);tar=t*vis
   row=[]
   for T in [hm.T[i],zo[i],Tp]:
    al=hm.render(T,ss=2)*vis;inter=(al*tar).sum();row.append(float(inter/(al.sum()+tar.sum()-inter+1e-9)))
   vals.append(row)
  r=np.array(vals);print(name,'base/opt/pool mean',r.mean(0),'pct pool-base',np.percentile(r[:,2]-r[:,0],[0,25,50,75,100]),'pool-vs-opt',np.percentile(r[:,2]-r[:,1],[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
initial base/opt/pool mean [0.73818132 0.80936439 0.83937894] pct pool-base [0.06944728 0.08784582 0.09551382 0.11377884 0.16916555] pool-vs-opt [-0.00218552  0.02553318  0.03025752  0.03513947  0.05120587]
final base/opt/pool mean [0.70063268 0.79813214 0.83269666] pct pool-base [0.06596726 0.1065096  0.12794772 0.14749278 0.28172332] pool-vs-opt [-0.00234222  0.0218567   0.03971609  0.04817611  0.06361228]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Ti=np.load('/work/scratch/pooled_initial.npy');Tf=np.load('/work/scratch/pooled_final.npy')
def ang(A,B):return np.rad2deg(np.arccos(np.clip((np.trace(A[:3,:3]@B[:3,:3].T)-1)/2,-1,1)))
for i in list(range(105,121))+list(range(630,651)):
 ref=Ti if i<300 else Tf
 print(i,'to static deg',round(ang(T[i],ref),2),'dt',np.round(T[i,:3,3]-ref[:3,3],3))
PY"
```
<details><summary>output</summary>

```
105 to static deg 6.74 dt [0.008 0.048 0.112]
106 to static deg 5.91 dt [0.006 0.05  0.11 ]
107 to static deg 7.13 dt [0.009 0.045 0.098]
108 to static deg 6.57 dt [0.005 0.05  0.111]
109 to static deg 6.9 dt [0.007 0.05  0.112]
110 to static deg 7.76 dt [0.002 0.028 0.06 ]
111 to static deg 6.3 dt [0.005 0.034 0.076]
112 to static deg 7.25 dt [0.007 0.053 0.119]
113 to static deg 6.62 dt [0.006 0.046 0.099]
114 to static deg 6.6 dt [0.007 0.04  0.086]
115 to static deg 9.38 dt [0.002 0.026 0.053]
116 to static deg 9.17 dt [0.003 0.029 0.054]
117 to static deg 8.38 dt [0.008 0.047 0.106]
118 to static deg 8.56 dt [0.004 0.029 0.058]
119 to static deg 9.05 dt [0.007 0.045 0.092]
120 to static deg 8.57 dt [0.005 0.048 0.101]
630 to static deg 5.07 dt [0.    0.028 0.048]
631 to static deg 5.8 …[+898 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
Ti=torch.tensor(np.load('/work/scratch/pooled_initial.npy'),dtype=torch.float32,device='cuda');Tf=torch.tensor(np.load('/work/scratch/pooled_final.npy'),dtype=torch.float32,device='cuda');T=torch.tensor(np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'],dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:,745:925,650:860]
for name,Tx,a,b in [('initpool',Ti,0,110),('finalpool',Tf,640,748)]:
 al=hm.render(Tx,ss=2);print(name,'area',float((al>.5).sum()),'mean target',o[a:b].mean())
for i in [0,50,100,650,700,747]:
 print(i,'perframe area',float((hm.render(T[i],ss=2)>.5).sum()),'gt',o[i].sum())
PY"
```
<details><summary>output</summary>

```
initpool area 2148.0 mean target 0.06726503126503126
finalpool area 1945.0 mean target 0.06040711346266902
0 perframe area 2017.0 gt 2541
50 perframe area 1982.0 gt 2543
100 perframe area 1933.0 gt 2516
650 perframe area 1806.0 gt 2248
700 perframe area 1786.0 gt 2330
747 perframe area 1801.0 gt 2306

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/vis_pooled.py <<'PY'
import sys,cv2,h5py,numpy as np,torch
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
v=cv2.VideoCapture('/work/inputs/video.mp4');Ti=torch.tensor(np.load('/work/scratch/pooled_initial.npy'),dtype=torch.float32,device='cuda');Tf=torch.tensor(np.load('/work/scratch/pooled_final.npy'),dtype=torch.float32,device='cuda');Tp=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];T=torch.tensor(Tp,dtype=torch.float32,device='cuda')
ims=[]
for fr in [0,20,50,80,100,109,640,660,680,700,720,747]:
 v.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=v.read()
 with h5py.File('/work/inputs/masks.h5') as h:om=h['object'][fr]
 P=Ti if fr<200 else Tf
 for pose,col in [(o.T[fr],(255,0,255)),(T[fr],(0,255,0)),(P,(0,255,255))]:
  _,a,_=o.render_pose(pose,650,745,210,180);m=(a>.5).detach().cpu().numpy().astype(np.uint8);cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  for c in cs:c[:,:,0]+=650;c[:,:,1]+=745
  cv2.drawContours(im,cs,-1,col,1)
 ys,xs=np.where(om);cx=int(xs.mean());cy=int(ys.mean());rad=52;crop=im[cy-rad:cy+rad,cx-rad:cx+rad];crop=cv2.resize(crop,(330,330),interpolation=cv2.INTER_CUBIC);cv2.putText(crop,str(fr),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);ims.append(crop)
while len(ims)%4:ims.append(np.zeros_like(ims[0]))
out=np.concatenate([np.concatenate(ims[i:i+4],1) for i in range(0,len(ims),4)],0);cv2.imwrite('/work/scratch/vis/pooled_contours.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97]);pri …[+52 chars]
```
<details><summary>output</summary>

```
(990, 1320, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
Ti=np.load('/work/scratch/pooled_initial.npy');Tf=np.load('/work/scratch/pooled_final.npy')
Rd=Ti[:3,:3].T@Tf[:3,:3];a=np.rad2deg(np.arccos(np.clip((np.trace(Rd)-1)/2,-1,1)));print('rel init-final deg',a,'trans',Tf[:3,3]-Ti[:3,3]);print('centers',Ti[:3,3],Tf[:3,3])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/smooth_eval.py <<'PY'
import sys,numpy as np,torch,h5py,time
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'].copy();Ti=np.load('/work/scratch/pooled_initial.npy');Tf=np.load('/work/scratch/pooled_final.npy')
# orientation continuity
q=Rot.from_matrix(T[:,:3,:3]).as_quat()
for i in range(1,len(q)):
 if np.dot(q[i],q[i-1])<0:q[i]*=-1
t=T[:,:3,3].copy()
def build(w,poly,method='sg'):
 qq=q.copy();tt=t.copy()
 if method=='sg':
  qq=savgol_filter(qq,w,poly,axis=0,mode='interp');tt=savgol_filter(tt,w,poly,axis=0,mode='interp')
 elif method=='gauss':
  # gaussian kernel smooth
  sig=w/5;xx=np.arange(-3*w,3*w+1);k=np.exp(-xx*xx/(2*sig*sig));k/=k.sum()
  for j in range(3):
   tt[:,j]=np.convolve(np.pad(t[:,j],(3*w,3*w),mode='edge'),k,mode='same')[3*w:-3*w]
   qq[:,j]=np.convolve(np.pad(q[:,j],(3*w,3*w),mode='edge'),k,mode='same')[3*w:-3*w]
 qq/=np.linalg.norm(qq,axis=1,keepdims=True)
 O=np.tile(np.eye(4),(len(T),1,1));O[:,:3,:3]=Rot.from_quat(qq).as_matrix();O[:,:3,3]=tt
 O[:110]=Ti;O[640:]=Tf
 return O
# load targets
x0,y0,w,h=650,745,210,180
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w];pers=f['person'][:,y0:y0+h,x0:x0+w]
targets=[]
for i in range(748):
 tgt=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float( …[+1186 chars]
```
<details><summary>output</summary>

```
orig (0.7130219706949198, 1.9886794446342761, 3.3413815865425964, 0.020239575634440718, 0.05100101847544578)
sg 5 2 (0.7130286066037304, 1.1919501458426962, 2.2598709523557514, 0.011116139881011821, 0.027448383228086445)
sg 7 2 (0.7130133865014562, 0.938440579630951, 1.996590687013526, 0.007503213610546583, 0.014845056234193358)
sg 9 2 (0.7130103909744406, 0.8356059030221288, 1.8090459926258953, 0.0063444468194639855, 0.012664996189108298)
sg 11 2 (0.7131029457416175, 0.7517364908497397, 1.7341353155088501, 0.005160235707348502, 0.010306533393826018)
sg 7 3 (0.7130133865014562, 0.9384405796309937, 1.9965906870128682, 0.0075032136105466045, 0.014845056234193625)
sg 9 3 (0.7130103909744406, 0.8356059030218721, 1.8090459926264593, 0.006344446819463972, 0.012664996189108298)
gauss 5 0 (0.71309 …[+323 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/scratch/hard_mask_orig.py'
```
<details><summary>output</summary>

```
import numpy as np,torch,trimesh,json
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
D=torch.device('cuda')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D);F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous()
z=np.load('/work/inputs/reconstruction.npz');T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json'));K=np.array(j['K']);Tc=np.linalg.inv(np.array(j['T_world_cam']));Tcc=torch.tensor(Tc,dtype=torch.float32,device=D)
H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];glctx=dr.RasterizeCudaContext()
def render(Tw,x0=650,y0=745,w=210,h=180,ss=2):
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1);P=torch.ten …[+312 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/scratch/iou_compare.npy');print(r[110:640,1].mean(),r[110:640,0].mean())
PY"
```
<details><summary>output</summary>

```
0.8354796870699469 0.7242113458660414

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Ts=torch.tensor(T,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:,745:925,650:860];p=f['person'][:,745:925,650:860]
for i in [110,200,300,400,500,600,639]:
 tgt=torch.tensor(o[i],device='cuda').float();pp=torch.tensor(p[i],device='cuda').float();pp=torch.nn.functional.max_pool2d(pp[None,None],5,1,2)[0,0];vis=(1-pp).clamp(0,1);tar=tgt*vis
 al=hm.render(Ts[i],ss=2)*vis;inter=(al*tar).sum();iou=float(inter/(al.sum()+tar.sum()-inter+1e-9))
 print(i,iou,float(tar.sum()),float(al.sum()))
PY"
```
<details><summary>output</summary>

```
110 0.8527438640594482 2269.0 1942.75
200 0.8538329601287842 1738.0 1502.5
300 0.8416721820831299 1496.0 1301.5
400 0.8330147862434387 1428.0 1209.25
500 0.8228228092193604 1579.0 1304.25
600 0.8256796002388 2155.0 1790.75
639 0.8467872738838196 2139.0 1841.75

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_candidates.py <<'PY'
import sys,numpy as np,torch,h5py
sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
x0,y0,w,h=650,745,210,180
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w];pers=f['person'][:,y0:y0+h,x0:x0+w]
def evalT(T):
 Ts=torch.tensor(T,dtype=torch.float32,device='cuda');ious=[]
 with torch.no_grad():
  for i in range(110,640):
   t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=t*(1-p).clamp(0,1)
   al=hm.render(Ts[i],ss=2);inter=(al*tar).sum();ious.append(float(inter/(al.sum()+tar.sum()-inter+1e-9)))
 Rs=T[:,:3,:3];Rd=np.einsum('nij,nkj->nik',Rs[:-1],Rs[1:]);ang=np.rad2deg(np.arccos(np.clip((np.trace(Rd,axis1=1,axis2=2)-1)/2,-1,1)));dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
 return np.array(ious),ang,dt
if __name__=='__main__':
 for f in ['opt_all_photo55.npz','smooth_sg7.npz','smooth_sg9.npz','smooth_g9.npz']:
  z=np.load('/work/scratch/'+f);T=z['T_world_mesh'];io,an,dt=evalT(T);print(f,'iou',io.mean(),'rot',an[110:639].mean(),np.percentile(an[110:639],90),'dt',dt[110:639].mean(),np.percentile(dt[110:639],90),'first',io[:5])
PY
python /work/scratch/eval_candidates.py"
```
<details><summary>output</summary>

```
opt_all_photo55.npz iou 0.7130219706949198 rot 1.9886794446342761 3.3413815865425964 dt 0.020239575634440718 0.05100101847544578 first [0.81372654 0.80250156 0.78089714 0.78744411 0.79415435]
smooth_sg7.npz iou 0.7130133865014562 rot 0.938440579630951 1.996590687013526 dt 0.007503213610546583 0.014845056234193358 first [0.80815423 0.80657393 0.78861451 0.78518057 0.79448205]
smooth_sg9.npz iou 0.7130103909744406 rot 0.8356059030221288 1.8090459926258953 dt 0.0063444468194639855 0.012664996189108298 first [0.80495256 0.80356389 0.7908659  0.79199505 0.79647952]
smooth_g9.npz iou 0.7127413001825225 rot 1.1561074944878313 2.290977880322933 dt 0.00395890395096955 0.007668067920782088 first [0.80438447 0.80523044 0.79082263 0.79483455 0.7917968 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys,torch,h5py
sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
r=np.load('/work/scratch/iou_compare.npy');T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Ts=torch.tensor(T,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
for i in [110,111,200,300,400,500,600,639]:
 t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=t*(1-p).clamp(0,1)
 al=hm.render(Ts[i],ss=2);inter=(al*tar).sum();iou=float(inter/(al.sum()+tar.sum()-inter+1e-9))
 print(i,'saved',r[i],'now',iou)
PY"
```
<details><summary>output</summary>

```
110 saved [0.78834826 0.85274386] now 0.813726544380188
111 saved [0.79323226 0.84076643] now 0.8025015592575073
200 saved [0.80364519 0.85383296] now 0.7439252138137817
300 saved [0.74927771 0.84167218] now 0.6516309976577759
400 saved [0.74284744 0.83301479] now 0.6447881460189819
500 saved [0.57565689 0.82282281] now 0.6591542363166809
600 saved [0.73298192 0.8256796 ] now 0.7878587245941162
639 saved [0.76668209 0.84678727] now 0.8404512405395508

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py,time;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Ts=torch.tensor(T,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][200,745:925,650:860];p=f['person'][200,745:925,650:860]
t=torch.tensor(o,device='cuda').float();pp=torch.tensor(p,device='cuda').float();pp=torch.nn.functional.max_pool2d(pp[None,None],5,1,2)[0,0];tar=t*(1-pp).clamp(0,1)
for j in range(10):
 al=hm.render(Ts[200],ss=2);inter=(al*tar).sum();iou=float(inter/(al.sum()+tar.sum()-inter+1e-9));print(j,iou,float(al.sum()),float((al>.5).sum()))
PY"
```
<details><summary>output</summary>

```
0 0.7439252138137817 1760.75 1736.0
1 0.7439252138137817 1760.75 1736.0
2 0.7439252138137817 1760.75 1736.0
3 0.7439252138137817 1760.75 1736.0
4 0.7439252138137817 1760.75 1736.0
5 0.7439252138137817 1760.75 1736.0
6 0.7439252138137817 1760.75 1736.0
7 0.7439252138137817 1760.75 1736.0
8 0.7439252138137817 1760.75 1736.0
9 0.7439252138137817 1760.75 1736.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o,hard_mask_orig as hm
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Tw=torch.tensor(T[200],dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][200,745:925,650:860];pm=f['person'][200,745:925,650:860]
tar=torch.tensor(om,device='cuda').float();p=torch.tensor(pm,device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=tar*(1-p).clamp(0,1)
a=o.render_pose(Tw,650,745,210,180)[1];b=hm.render(Tw,ss=2)
for name,x in [('soft',a),('hard',b)]:
 for thr in [.01,.1,.5,.9]:
  pr=x>thr;inter=(pr&(tar>.5)).sum();uni=(pr|(tar>.5)).sum();print(name,thr,float(inter/uni),float(pr.sum()),float(x.sum()))
print('diff',float((a-b).abs().sum()),float(a.max()),float(b.max()))
PY"
```
<details><summary>output</summary>

```
soft 0.01 0.7564792037010193 1854.0 1706.18212890625
soft 0.1 0.7561214566230774 1848.0 1706.18212890625
soft 0.5 0.7407035231590271 1726.0 1706.18212890625
soft 0.9 0.6958096027374268 1540.0 1706.18212890625
hard 0.01 0.7564792037010193 1854.0 1760.75
hard 0.1 0.7564792037010193 1854.0 1760.75
hard 0.5 0.7413533926010132 1736.0 1760.75
hard 0.9 0.7246819138526917 1651.0 1760.75
diff 54.814292907714844 1.0 1.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Tw=torch.tensor(T[200],dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][200,745:925,650:860];pm=f['person'][200,745:925,650:860]
tar=torch.tensor(om,device='cuda').float();p=torch.tensor(pm,device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=tar*(1-p).clamp(0,1)
a=o.render_pose(Tw,650,745,210,180)[1];pv=a*(1-p).clamp(0,1);inter=(pv*tar).sum();print('dice',float(2*inter/(pv.sum()+tar.sum()+1)), 'inter',float(inter),'pv',float(pv.sum()),'tar',float(tar.sum()))
PY"
```
<details><summary>output</summary>

```
dice 0.9104937314987183 inter 1458.758544921875 pv 1465.3243408203125 tar 1738.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Ts=torch.tensor(T,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
for i in [0,50,100,110,160,200,300,400,500,600,650,700,747]:
 t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=(t*(1-p).clamp(0,1))>.5
 vals=[]
 for X in [hm.T[i],Ts[i]]:
  pr=hm.render(X,ss=2)>.5;inter=(pr&tar).sum();vals.append(float(inter/(pr|tar).sum()))
 print(i,vals)
PY"
```
<details><summary>output</summary>

```
0 [0.6617044806480408, 0.7895563244819641]
50 [0.7299960851669312, 0.7902125716209412]
100 [0.7137739062309265, 0.7891295552253723]
110 [0.738753616809845, 0.7976391315460205]
160 [0.6469843983650208, 0.7785778641700745]
200 [0.6840550899505615, 0.7413533926010132]
300 [0.5577478408813477, 0.6591026186943054]
400 [0.553398072719574, 0.6496745944023132]
500 [0.4121503531932831, 0.6588713526725769]
600 [0.6810159087181091, 0.7795031070709229]
650 [0.7206405401229858, 0.8203950524330139]
700 [0.6828451752662659, 0.7589743733406067]
747 [0.6781802773475647, 0.7748487591743469]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
Ti=torch.tensor(np.load('/work/scratch/pooled_initial.npy'),dtype=torch.float32,device='cuda');Tf=torch.tensor(np.load('/work/scratch/pooled_final.npy'),dtype=torch.float32,device='cuda');To=torch.tensor(np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'],dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
for name,a,b,P in [('i',0,110,Ti),('f',640,748,Tf)]:
 vals=[]
 for i in range(a,b):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5)
  row=[]
  for X in [hm.T[i],To[i],P]:
   pr=hm.render(X,ss=2)>.5;row.append(float((pr&tar).sum()/(pr|tar).sum()))
  vals.append(row)
 r=np.array(vals);print(name,'mean base/opt/pool',r.mean(0),'pool-base',np.percentile(r[:,2]-r[:,0],[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
i mean base/opt/pool [0.71184409 0.79408743 0.81453654] pool-base [0.0581013  0.08326711 0.09798744 0.12155148 0.16922092]
f mean base/opt/pool [0.68801308 0.7848938  0.81792611] pool-base [0.06055248 0.10578468 0.12831956 0.14631446 0.27809018]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
for f in ['smooth_sg7.npz','smooth_sg9.npz']:
 T=np.load('/work/scratch/'+f)['T_world_mesh'];R=T[:,:3,:3];Rd=np.einsum('nij,nkj->nik',R[:-1],R[1:]);ang=np.rad2deg(np.arccos(np.clip((np.trace(Rd,axis1=1,axis2=2)-1)/2,-1,1)));dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
 print(f)
 for i in [0,50,100,108,109,110,111,112,120,130,638,639,640,641,642,650,700,746]:print(i,'rotnext',round(ang[i],2) if i<747 else 0,'dt',round(dt[i],4) if i<747 else 0,'c',np.round(T[i,:3,3],3))
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
T=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Ti=np.load('/work/scratch/pooled_initial.npy');Tf=np.load('/work/scratch/pooled_final.npy')
def ang(A,B):return np.rad2deg(np.arccos(np.clip((np.trace(A[:3,:3]@B[:3,:3].T)-1)/2,-1,1)))
print('init')
for i in range(105,151,5):print(i,round(ang(T[i],Ti),1),np.round(T[i,:3,3]-Ti[:3,3],3))
print('final')
for i in range(580,651,5):print(i,round(ang(T[i],Tf),1),np.round(T[i,:3,3]-Tf[:3,3],3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
To=torch.tensor(np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');Ti=torch.tensor(np.load('/work/scratch/pooled_initial.npy'),dtype=torch.float32,device='cuda');Tf=torch.tensor(np.load('/work/scratch/pooled_final.npy'),dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
for a,b in [(0,110),(0,130),(110,130),(130,150),(150,580),(580,610),(610,640),(640,748)]:
 vals=[]
 for i in range(a,b):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5)
  row=[]
  for X in [To[i],Ti if i<300 else Tf]:
   pr=hm.render(X,ss=2)>.5;row.append(float((pr&tar).sum()/(pr|tar).sum()))
  vals.append(row)
 r=np.array(vals);print(a,b,'opt',r[:,0].mean(),'static',r[:,1].mean(),'delta',np.mean(r[:,1]-r[:,0]))
PY"
```
<details><summary>output</summary>

```
0 110 opt 0.7940874294801192 static 0.8145365379073403 delta 0.020449108427221124
0 130 opt 0.7926073202720055 static 0.8104108159358685 delta 0.017803495663862964
110 130 opt 0.7844667196273803 static 0.7877193450927734 delta 0.0032526254653930664
130 150 opt 0.7864326566457749 static 0.7372687071561813 delta -0.0491639494895935
150 580 opt 0.6957175376803376 static 0.12344228497998745 delta -0.5722752527003502
580 610 opt 0.7816974182923635 static 0.7406911055246989 delta -0.041006312767664595
610 640 opt 0.8005181392033894 static 0.8059927125771841 delta 0.005474573373794556
640 748 opt 0.7848938046782105 static 0.8179261066295482 delta 0.033032301951337745

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/build_candidate.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
S=np.load('/work/scratch/smooth_sg9.npz')['T_world_mesh'].copy();Ti=np.load('/work/scratch/pooled_initial.npy');Tf=np.load('/work/scratch/pooled_final.npy')
R=Rot.from_matrix(S[:,:3,:3])
def lerp_pose(A,B,w):
 r=Rot.from_matrix(A[:3,:3]).slerp(Rot.from_matrix(B[:3,:3]),w).as_matrix();t=(1-w)*A[:3,3]+w*B[:3,3];T=np.eye(4);T[:3,:3]=r;T[:3,3]=t;return T
def smoothstep(x):return x*x*(3-2*x)
O=S.copy()
for i in range(129):
 O[i]=Ti
for i in range(129,146):
 w=smoothstep((i-128)/(146-128));O[i]=lerp_pose(Ti,S[i],w)
for i in range(595,615):
 w=smoothstep((i-594)/(615-594));O[i]=lerp_pose(S[i],Tf,w)
for i in range(615,748): O[i]=Tf
np.save('/work/scratch/candidate_v1.npy',O)
print('saved')
PY
python /work/scratch/build_candidate.py
python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
T=np.load('/work/scratch/candidate_v1.npy');Ts=torch.tensor(T,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
ious=[]
for i in range(748):
 t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5)
 pr=hm.render(Ts[i],ss=2)>.5;ious.append(float((pr&tar).sum()/(pr|tar).sum()))
ious=np.array(i …[+575 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/build_candidate.py", line 12, in <module>
    w=smoothstep((i-128)/(146-128));O[i]=lerp_pose(Ti,S[i],w)
                                         ^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/build_candidate.py", line 6, in lerp_pose
    r=Rot.from_matrix(A[:3,:3]).slerp(Rot.from_matrix(B[:3,:3]),w).as_matrix();t=(1-w)*A[:3,3]+w*B[:3,3];T=np.eye(4);T[:3,:3]=r;T[:3,3]=t;return T
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'scipy.spatial.transform._rotation.Rotation' object has no attribute 'slerp'
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                 …[+131 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_candidate.py';s=open(p).read();s=s.replace('from scipy.spatial.transform import Rotation as Rot','from scipy.spatial.transform import Rotation as Rot, Slerp');s=s.replace(\"r=Rot.from_matrix(A[:3,:3]).slerp(Rot.from_matrix(B[:3,:3]),w).as_matrix();t=(1-w)*A[:3,3]+w*B[:3,3];T=np.eye(4);T[:3,:3]=r;T[:3,3]=t;return T\",\"r=Slerp([0,1],Rot.from_matrix(np.stack([A[:3,:3],B[:3,:3]])))(w).as_matrix();t=(1-w)*A[:3,3]+w*B[:3,3];T=np.eye(4);T[:3,:3]=r;T[:3,3]=t;return T\");open(p,'w').write(s)
PY
python /work/scratch/build_candidate.py
python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
T=np.load('/work/scratch/candidate_v1.npy');Ts=torch.tensor(T,dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
ious=[]
for i in range(748):
 t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5)
 pr=hm.render(Ts[i],ss=2)>.5;ious.append(float((pr&tar).sum()/(pr|tar).sum()))
ious=np.array(ious);np.save('/work/scratch/iou_candidate.npy',ious)
for a,b in [(0,110),(110,129),(129,146),(146,520),(520,595),(595,615),(615,640),(640,748),(0,748)]:print(a,b,ious[a:b].mean())
R=T[:,:3,:3];Rd=np.einsum('nij,nkj->nik',R[:-1],R[1:]);ang=np.rad2deg(np.arccos(np.clip((np.trace(Rd …[+295 chars]
```
<details><summary>output</summary>

```
saved
0 110 0.8145365379073403
110 129 0.7883507076062655
129 146 0.7929220304769629
146 520 0.6833621467498534
520 595 0.7735352166493734
595 615 0.7906778007745743
615 640 0.8084702277183533
640 748 0.8179261066295482
0 748 0.74333057819522
motion pct rot [0.         0.35890217 1.62694341 2.15225507 2.69402583 3.4527645 ] dt [0.         0.00274037 0.01108621 0.01451656 0.02000975 0.02323192]
127 0.007 0.0
128 0.076 0.0008
129 0.217 0.0026
130 0.359 0.005
145 0.672 0.0073
146 0.981 0.0032
147 1.271 0.004
594 0.947 0.002
595 0.638 0.0045
596 0.396 0.0028
614 0.034 0.0005
615 0.0 0.0
616 0.0 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');O=torch.tensor(np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');C=torch.tensor(np.load('/work/scratch/candidate_v1.npy'),dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
res=[]
for i in range(748):
 t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);row=[]
 for X in [B[i],O[i],C[i]]:
  pr=hm.render(X,ss=2)>.5;row.append(float((pr&tar).sum()/(pr|tar).sum()))
 res.append(row)
res=np.array(res);np.save('/work/scratch/iou_all3.npy',res)
for a,b in [(0,110),(110,129),(129,146),(146,520),(520,595),(595,615),(615,640),(640,748),(0,748)]:
 r=res[a:b];print(a,b,'base %.4f opt %.4f cand %.4f gain %.4f'%(r[:,0].mean(),r[:,1].mean(),r[:,2].mean(),(r[:,2]-r[:,0]).mean()))
PY"
```
<details><summary>output</summary>

```
0 110 base 0.7118 opt 0.7941 cand 0.8145 gain 0.1027
110 129 base 0.7173 opt 0.7847 cand 0.7884 gain 0.0710
129 146 base 0.7096 opt 0.7868 cand 0.7929 gain 0.0833
146 520 base 0.5649 opt 0.6842 cand 0.6834 gain 0.1185
520 595 base 0.6185 opt 0.7738 cand 0.7735 gain 0.1551
595 615 base 0.6924 opt 0.7896 cand 0.7907 gain 0.0983
615 640 base 0.7031 opt 0.8016 cand 0.8085 gain 0.1054
640 748 base 0.6880 opt 0.7849 cand 0.8179 gain 0.1299
0 748 base 0.6248 opt 0.7355 cand 0.7433 gain 0.1185

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_test.py <<'PY'
import sys,time,cv2,h5py,numpy as np,torch
import torch.nn.functional as Fn
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
T0all=torch.tensor(np.load('/work/scratch/candidate_v1.npy'),dtype=torch.float32,device='cuda')
def optimize(fr,steps=40,lr=.0008):
 cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=cap.read();cap.release();img=torch.tensor(im[745:925,650:860].astype(np.float32)/255,device='cuda')
 with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][fr,745:925,650:860];pm=f['person'][fr,745:925,650:860]
 target=torch.tensor(om,device='cuda').float();person=torch.tensor(pm,device='cuda').float();person=Fn.max_pool2d(person[None,None],5,1,2)[0,0];vis=(1-person).clamp(0,1);target=target*vis
 imhp=img-o.blur(img[None])[0];T0=T0all[fr].clone();r=torch.zeros(3,device='cuda',requires_grad=True);dt=torch.zeros(3,device='cuda',requires_grad=True);opt=torch.optim.Adam([r,dt],lr=lr)
 for it in range(steps):
  opt.zero_grad();Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=o.render_pose(Tw,650,745,210,180);pv=alpha*vis;inter=(pv*target).sum();union=(pv*pv).sum()+(target*target).sum()-inter;iou=inter/(union+1e-9)
  hp=rgb-o.blur(rgb[None])[0];m=pv*target;den=m.sum().clamp_min(1);hpz=hp-(hp*m[...,None]).sum((0,1),keepdim=True)/den;iz=imhp-(imhp*m[...,None]).sum((0,1),keepdim=True)/den;cos=(hpz*iz*m[...,None]).sum()/torch.sqrt( …[+517 chars]
```
<details><summary>output</summary>

```
0 time 1.902531385421753 delta [-0.00014871 -0.00405586 -0.01301599]
50 time 0.40286707878112793 delta [ 0.00175916  0.00061882 -0.00735331]
100 time 0.4521007537841797 delta [ 0.00133835  0.00072408 -0.00615501]
150 time 0.5193068981170654 delta [-0.00039194 -0.00433028 -0.01088786]
200 time 0.6281092166900635 delta [ 0.00224362 -0.00073445 -0.00404191]
300 time 0.4300670623779297 delta [-0.00251598 -0.00017822 -0.00923395]
400 time 0.6216435432434082 delta [ 0.00110428 -0.00226712 -0.01124167]
500 time 0.7992925643920898 delta [-0.00189332 -0.00344765 -0.01154447]
600 time 0.4620664119720459 delta [ 0.0009369  -0.00386178 -0.01231027]
650 time 0.5818376541137695 delta [-0.0007795   0.0019511  -0.00810027]
700 time 0.6283516883850098 delta [ 0.00077127 -0.00571394 -0.01564074]
747 time 0. …[+61 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,time,cv2,h5py,numpy as np,torch;sys.path.insert(0,'/work/scratch');import refine_test as r,hard_mask_orig as hm
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
for fr in [0,50,100,150,200,300,400,500,600,650,700,747]:
 Tw=r.optimize(fr,steps=35)
 t=torch.tensor(objs[fr],device='cuda').float();p=torch.tensor(pers[fr],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5)
 vals=[]
 for X in [r.T0all[fr],Tw]:
  pr=hm.render(X,ss=2)>.5;vals.append(float((pr&tar).sum()/(pr|tar).sum()))
 print(fr,vals)
PY"
```
<details><summary>output</summary>

```
0 [0.8309254050254822, 0.8361295461654663]
50 [0.8174322843551636, 0.8306899070739746]
100 [0.824791431427002, 0.8408179879188538]
150 [0.7946256995201111, 0.8073967099189758]
200 [0.7382146716117859, 0.7475000023841858]
300 [0.6585239171981812, 0.6733509302139282]
400 [0.6428187489509583, 0.6583106517791748]
500 [0.6740359663963318, 0.6769388914108276]
600 [0.7694346308708191, 0.789496898651123]
650 [0.8118374347686768, 0.8441030383110046]
700 [0.8137462735176086, 0.8212766051292419]
747 [0.8166666626930237, 0.8209083080291748]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/refine_all.log'
```
<details><summary>output</summary>

```
140 iou 0.869619607925415 cos -0.02122959867119789 rdeg 1.335381238348782 dt [-0.0005, -0.0039, -0.008] elapsed 4.2
160 iou 0.8691983222961426 cos 0.032404232770204544 rdeg 1.9262460473924874 dt [-0.0015, -0.0016, -0.0056] elapsed 8.6
180 iou 0.8878247737884521 cos -0.008561406284570694 rdeg 0.9983814563602209 dt [-0.0007, -0.0021, -0.0032] elapsed 13.1
200 iou 0.8637168407440186 cos -0.030967602506279945 rdeg 1.6050499258562922 dt [0.0018, -0.0006, -0.0042] elapsed 17.6
220 iou 0.8617144227027893 cos 0.026380056515336037 rdeg 0.9388316944241524 dt [-0.0022, -0.002, -0.0099] elapsed 22.1
240 iou 0.8349009156227112 cos 0.14016492664813995 rdeg 2.0750586558133364 dt [-0.0011, -0.0026, -0.0128] elapsed 26.5

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/refine_all.log'
```
<details><summary>output</summary>

```
240 iou 0.8349009156227112 cos 0.14016492664813995 rdeg 2.0750586558133364 dt [-0.0011, -0.0026, -0.0128] elapsed 26.5
260 iou 0.8656861186027527 cos 0.14014950394630432 rdeg 1.972267409786582 dt [-0.0025, -0.0015, -0.0085] elapsed 30.8
280 iou 0.8474343419075012 cos 0.1828753650188446 rdeg 2.1715780455619096 dt [-0.0006, -0.0025, -0.0107] elapsed 35.3
300 iou 0.8450275659561157 cos 0.13369983434677124 rdeg 2.373297354206443 dt [-0.0013, -0.0004, -0.0085] elapsed 39.8
320 iou 0.8377535939216614 cos 0.15003585815429688 rdeg 2.2768808517605064 dt [-0.0009, -0.0012, -0.0095] elapsed 43.9
340 iou 0.8692497611045837 cos -0.03443116694688797 rdeg 2.022455494478345 dt [-0.003, -0.0014, -0.014] elapsed 47.4
360 iou 0.8606614470481873 cos 0.1540801078081131 rdeg 2.0746861696243286 dt [-0.0017, -0.0 …[+381 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_all.py <<'PY'
import sys,time,numpy as np,torch,cv2,h5py
import torch.nn.functional as Fn
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
x0,y0,w,h=650,745,210,180
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs=[]
while True:
 ok,im=cap.read()
 if not ok:break
 imgs.append(im[y0:y0+h,x0:x0+w].copy())
cap.release();imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w];persons=f['person'][:,y0:y0+h,x0:x0+w]
T=np.load('/work/scratch/candidate_v1.npy');t0=time.time()
for fr in range(129,615):
 T0=torch.tensor(T[fr],dtype=torch.float32,device=o.D);img=torch.tensor(imgs[fr].astype(np.float32)/255,device=o.D);target=torch.tensor(objs[fr],device=o.D).float();person=torch.tensor(persons[fr],device=o.D).float();person=Fn.max_pool2d(person[None,None],5,1,2)[0,0];vis=(1-person).clamp(0,1);target=target*vis
 imhp=img-o.blur(img[None])[0];r=torch.zeros(3,device=o.D,requires_grad=True);dt=torch.zeros(3,device=o.D,requires_grad=True);opt=torch.optim.Adam([r,dt],lr=.0008)
 for it in range(40):
  opt.zero_grad();Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=o.render_pose(Tw,x0,y0,w,h);pv=alpha*vis;inter=(pv*target).sum();union=(pv*pv).sum()+(target*target).sum()-inter;iou=inter/(union+1e-9)
  hp=rgb-o.blur(rgb[None])[0];m=pv*target;den=m.sum().clamp_min(1);hpz=hp-(hp*m[...,None]).sum((0,1),keepdim=True)/den;iz=imhp-(imhp*m[...,None]).sum((0,1) …[+710 chars]
```
<details><summary>output</summary>

```
240 iou 0.8349009156227112 cos 0.14016492664813995 rdeg 2.0750586558133364 dt [-0.0011, -0.0026, -0.0128] elapsed 26.5
260 iou 0.8656861186027527 cos 0.14014950394630432 rdeg 1.972267409786582 dt [-0.0025, -0.0015, -0.0085] elapsed 30.8
280 iou 0.8474343419075012 cos 0.1828753650188446 rdeg 2.1715780455619096 dt [-0.0006, -0.0025, -0.0107] elapsed 35.3
300 iou 0.8450275659561157 cos 0.13369983434677124 rdeg 2.373297354206443 dt [-0.0013, -0.0004, -0.0085] elapsed 39.8
320 iou 0.8377535939216614 cos 0.15003585815429688 rdeg 2.2768808517605064 dt [-0.0009, -0.0012, -0.0095] elapsed 43.9
340 iou 0.8692497611045837 cos -0.03443116694688797 rdeg 2.022455494478345 dt [-0.003, -0.0014, -0.014] elapsed 47.4
360 iou 0.8606614470481873 cos 0.1540801078081131 rdeg 2.0746861696243286 dt [-0.0017, -0.0 …[+1467 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_v2.py <<'PY'
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot,Slerp
T=np.load('/work/scratch/refined_v1.npy').copy();Ti=np.load('/work/scratch/pooled_initial.npy');Tf=np.load('/work/scratch/pooled_final.npy')
q=Rot.from_matrix(T[:,:3,:3]).as_quat()
for i in range(1,len(q)):
 if np.dot(q[i],q[i-1])<0:q[i]*=-1
def smoothstep(x):return x*x*(3-2*x)
def lerp_pose(A,B,w):
 r=Slerp([0,1],Rot.from_matrix(np.stack([A[:3,:3],B[:3,:3]])))(w).as_matrix();O=np.eye(4);O[:3,:3]=r;O[:3,3]=(1-w)*A[:3,3]+w*B[:3,3];return O
def build(win):
 qq=savgol_filter(q,win,2,axis=0,mode='interp');qq/=np.linalg.norm(qq,axis=1,keepdims=True)
 S=np.tile(np.eye(4),(len(T),1,1));S[:,:3,:3]=Rot.from_quat(qq).as_matrix();S[:,:3,3]=savgol_filter(T[:,:3,3],win,2,axis=0,mode='interp')
 O=S.copy();O[:129]=Ti
 for i in range(129,146):O[i]=lerp_pose(Ti,S[i],smoothstep((i-128)/(146-128)))
 for i in range(595,615):O[i]=lerp_pose(S[i],Tf,smoothstep((i-594)/(615-594)))
 O[615:]=Tf
 np.save(f'/work/scratch/candidate_v2_sg{win}.npy',O)
for win in [5,7,9,11]:build(win)
PY
python /work/scratch/build_v2.py
python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
def ev(T):
 Ts=torch.tensor(T,dtype=torch.float32,device='cuda');ious=[]
 for i in range(748 …[+915 chars]
```
<details><summary>output</summary>

```
refined_v1.npy iou all 0.7496 manip 0.7152 rot 1.292 p90 2.361 max 4.48 dt 0.0075 p90 0.0151 max 0.0247
candidate_v2_sg5.npy iou all 0.7485 manip 0.7135 rot 0.936 p90 2.003 max 3.86 dt 0.0058 p90 0.0121 max 0.0214
candidate_v2_sg7.npy iou all 0.7483 manip 0.7131 rot 0.852 p90 1.902 max 3.45 dt 0.0052 p90 0.0106 max 0.0184
candidate_v2_sg9.npy iou all 0.7481 manip 0.7129 rot 0.807 p90 1.836 max 3.26 dt 0.0048 p90 0.0097 max 0.0170
candidate_v2_sg11.npy iou all 0.7480 manip 0.7127 rot 0.756 p90 1.766 max 3.13 dt 0.0043 p90 0.0086 max 0.0174

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/vis_final.py <<'PY'
import sys,cv2,h5py,numpy as np,torch
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
T=np.load('/work/scratch/candidate_v2_sg7.npy');Ts=torch.tensor(T,dtype=torch.float32,device='cuda');v=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for fr in list(range(0,748,25))+[747]:
 v.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=v.read()
 with h5py.File('/work/inputs/masks.h5') as h:om=h['object'][fr]
 for pose,col in [(o.T[fr],(255,0,255)),(Ts[fr],(0,255,255))]:
  rgb,a,_=o.render_pose(pose,650,745,210,180);m=(a>.5).detach().cpu().numpy().astype(np.uint8);cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  for c in cs:c[:,:,0]+=650;c[:,:,1]+=745
  cv2.drawContours(im,cs,-1,col,1)
 ys,xs=np.where(om);cx=int(xs.mean());cy=int(ys.mean());rad=48;crop=im[max(0,cy-rad):cy+rad,max(0,cx-rad):cx+rad];crop=cv2.resize(crop,(260,260),interpolation=cv2.INTER_CUBIC);cv2.putText(crop,str(fr),(5,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2);ims.append(crop)
while len(ims)%6:ims.append(np.zeros_like(ims[0]))
out=np.concatenate([np.concatenate(ims[i:i+6],1) for i in range(0,len(ims),6)],0);cv2.imwrite('/work/scratch/vis/final_contours.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97]);print(out.shape)
PY
python /work/scratch/vis_final.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/vis_final.py", line 2, in <module>
    sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
                                       ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/opt_test_aa2.py", line 7, in <module>
    V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
```
<details><summary>output</summary>

```
36641 MiB, 49140 MiB
5167 MiB, 49140 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
print(torch.cuda.mem_get_info())
x=torch.zeros(1000,device='cuda');print(x)
PY"
```
<details><summary>output</summary>

```
(47676522496, 50856656896)
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0 …[+3588 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/vis_final.py'
```
<details><summary>output</summary>

```
(1560, 1560, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/pooled_refine.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,json,trimesh
import torch.nn.functional as Fn
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
x0,y0,w,h=650,745,210,180
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs=[]
while True:
 ok,im=cap.read()
 if not ok:break
 imgs.append(im[y0:y0+h,x0:x0+w])
cap.release();imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w]
mm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry().convex_hull;HV=torch.tensor(mm.vertices,dtype=torch.float32,device=o.D)
pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);pn=torch.tensor(pl[:3],dtype=torch.float32,device=o.D);pd=float(pl[3])
def opt(P,a,b,steps=120):
 img=torch.tensor(np.median(imgs[a:b].astype(np.float32)/255,axis=0),device=o.D);tar=torch.tensor(objs[a:b].mean(0),device=o.D);imhp=img-o.blur(img[None])[0]
 T0=torch.tensor(P,dtype=torch.float32,device=o.D);r=torch.zeros(3,device=o.D,requires_grad=True);dt=torch.zeros(3,device=o.D,requires_grad=True);op=torch.optim.Adam([r,dt],lr=.0007)
 for it in range(steps):
  op.zero_grad();Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=o.render_pose(Tw,x0,y0,w,h);pv=alpha;inter=(pv*tar).sum();union=(pv*pv).sum()+(tar*tar).sum()-inter;iou=inter/(union+1e-9)
  hp=rgb-o.blur(rgb[None])[0];m=pv*tar;den=m.sum().clamp_min(1);hpz=hp-(hp*m[...,None]).sum …[+944 chars]
```
<details><summary>output</summary>

```
initial 0.8672710516378902 0.0429309452522293 0.04920506477355957 [[ 0.02479898  0.5992415   0.8001841   0.03259078]
 [ 0.9977282  -0.06498832  0.01774725  1.0996404 ]
 [ 0.0626375   0.79792607 -0.59949183  2.8273904 ]
 [ 0.          0.          0.          1.        ]]
final 0.8641216551482137 0.018482651265984768 0.030870914459228516 [[-1.8972980e-02  7.7701622e-01  6.2919462e-01  2.5004828e-02]
 [ 9.9981844e-01  1.5869435e-02  1.0551166e-02  1.1130841e+00]
 [-1.7865272e-03  6.2928045e-01 -7.7717626e-01  2.8866639e+00]
 [ 0.0000000e+00  0.0000000e+00  0.0000000e+00  1.0000000e+00]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
Ti=torch.tensor(np.load('/work/scratch/pooled2_initial.npy'),dtype=torch.float32,device='cuda');Tf=torch.tensor(np.load('/work/scratch/pooled2_final.npy'),dtype=torch.float32,device='cuda');Oi=torch.tensor(np.load('/work/scratch/pooled_initial.npy'),dtype=torch.float32,device='cuda');Of=torch.tensor(np.load('/work/scratch/pooled_final.npy'),dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
for name,a,b,P,Q in [('i',0,129,Ti,Oi),('f',615,748,Tf,Of)]:
 vals=[]
 for i in range(a,b):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);row=[]
  for X in [Q,P]:
   pr=hm.render(X,ss=2)>.5;row.append(float((pr&tar).sum()/(pr|tar).sum()))
  vals.append(row)
 r=np.array(vals);print(name,'old/new',r.mean(0),'delta',np.mean(r[:,1]-r[:,0]),'worse',np.mean(r[:,1]<r[:,0]),'pct',np.percentile(r[:,1]-r[:,0],[0,25,50,75,100]))
PY"
```
<details><summary>output</summary>

```
i old/new [0.81067971 0.8349363 ] delta 0.024256590262863986 worse 0.0 pct [0.01140636 0.02343768 0.02514476 0.02615869 0.02750027]
f old/new [0.81614869 0.82801774] delta 0.01186905140267279 worse 0.0 pct [0.00642371 0.0097782  0.01197708 0.01403934 0.01774168]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/refine_all_b.log'
```
<details><summary>output</summary>

```
200 iou 0.8619112372398376 cos -0.011065579019486904 rdeg 1.6615964245051145 dt [0.0013, 0.0002, -0.0047] elapsed 33.2
220 iou 0.8633581399917603 cos 0.0025651706382632256 rdeg 1.0628644723445178 dt [-0.0022, 0.0, -0.0118] elapsed 38.2
240 iou 0.8359692096710205 cos 0.1452515721321106 rdeg 2.1484774980694055 dt [-0.0012, -0.0038, -0.0166] elapsed 43.4
260 iou 0.8652321100234985 cos 0.12234720587730408 rdeg 1.9840057410299776 dt [-0.003, -0.0013, -0.0086] elapsed 50.9
280 iou 0.8473961353302002 cos 0.1408526599407196 rdeg 2.236992597207427 dt [-0.0013, -0.0024, -0.0098] elapsed 57.2

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/refine_all_b.log'
```
<details><summary>output</summary>

```
340 iou 0.8701929450035095 cos -0.02662702277302742 rdeg 2.0692004833370445 dt [-0.0029, -0.0017, -0.0144] elapsed 71.9
360 iou 0.8581435084342957 cos 0.1642727553844452 rdeg 1.983761116862297 dt [-0.0016, -0.0005, -0.0066] elapsed 76.7
380 iou 0.8518709540367126 cos 0.18276727199554443 rdeg 2.242803381755948 dt [0.0012, -0.0012, -0.0052] elapsed 81.5
400 iou 0.8403952121734619 cos 0.1707661896944046 rdeg 2.1551081791520117 dt [0.002, -0.0021, -0.0102] elapsed 86.2
420 iou 0.821427583694458 cos -0.01663467288017273 rdeg 1.5624518075957894 dt [-0.0042, -0.0016, -0.0121] elapsed 90.8

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/refine_all_b.log'
```
<details><summary>output</summary>

```
500 iou 0.8396185040473938 cos -0.08439082652330399 rdeg 2.0203614603728055 dt [-0.0013, -0.0011, -0.0129] elapsed 110.4
520 iou 0.8628275394439697 cos 0.01166708767414093 rdeg 1.0719200491905212 dt [-0.0034, -0.0034, -0.0051] elapsed 115.0
540 iou 0.8356013298034668 cos 0.008319574408233166 rdeg 1.8542385969310997 dt [0.0013, -0.0069, -0.0165] elapsed 119.7
560 iou 0.8359050154685974 cos 0.010707848705351353 rdeg 1.7704769125208257 dt [0.0014, -0.0051, -0.0159] elapsed 124.3
580 iou 0.8205791115760803 cos 0.008273793384432793 rdeg 1.7764301810413599 dt [-0.0005, -0.0054, -0.0141] elapsed 128.8

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_candidate.py';s=open(p).read().replace(\"pooled_initial.npy\",\"pooled2_initial.npy\").replace(\"pooled_final.npy\",\"pooled2_final.npy\").replace(\"candidate_v1.npy\",\"candidate_v1b.npy\");open('/work/scratch/build_candidate_b.py','w').write(s)
p='/work/scratch/refine_all.py';s=open(p).read().replace(\"candidate_v1.npy\",\"candidate_v1b.npy\").replace(\"refined_v1.npy\",\"refined_v1b.npy\");open('/work/scratch/refine_all_b.py','w').write(s)
p='/work/scratch/build_v2.py';s=open(p).read().replace(\"refined_v1.npy\",\"refined_v1b.npy\").replace(\"pooled_initial.npy\",\"pooled2_initial.npy\").replace(\"pooled_final.npy\",\"pooled2_final.npy\").replace(\"candidate_v2_sg\",\"candidate_v2b_sg\");open('/work/scratch/build_v2_b.py','w').write(s)
PY
python /work/scratch/build_candidate_b.py && python /work/scratch/refine_all_b.py > /work/scratch/refine_all_b.log 2>&1 && python /work/scratch/build_v2_b.py && tail -5 /work/scratch/refine_all_b.log"
```
<details><summary>output</summary>

```
saved
540 iou 0.8356013298034668 cos 0.008319574408233166 rdeg 1.8542385969310997 dt [0.0013, -0.0069, -0.0165] elapsed 119.7
560 iou 0.8359050154685974 cos 0.010707848705351353 rdeg 1.7704769125208257 dt [0.0014, -0.0051, -0.0159] elapsed 124.3
580 iou 0.8205791115760803 cos 0.008273793384432793 rdeg 1.7764301810413599 dt [-0.0005, -0.0054, -0.0141] elapsed 128.8
600 iou 0.8377822637557983 cos 0.0038882875815033913 rdeg 2.142663938552141 dt [0.0013, -0.0023, -0.0092] elapsed 133.6
done 136.94086980819702

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ls -l /work/scratch/candidate_v2b_sg*.npy && python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
def ev(T):
 Ts=torch.tensor(T,dtype=torch.float32,device='cuda');ious=[]
 for i in range(748):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);pr=hm.render(Ts[i],ss=2)>.5;ious.append(float((pr&tar).sum()/(pr|tar).sum()))
 ious=np.array(ious);R=T[:,:3,:3];Rd=np.einsum('nij,nkj->nik',R[:-1],R[1:]);ang=np.rad2deg(np.arccos(np.clip((np.trace(Rd,axis1=1,axis2=2)-1)/2,-1,1)));dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);return ious,ang,dt
for f in ['refined_v1b.npy','candidate_v2b_sg5.npy','candidate_v2b_sg7.npy','candidate_v2b_sg9.npy','candidate_v2b_sg11.npy']:
 T=np.load('/work/scratch/'+f);io,an,dt=ev(T);print(f,'iou all %.4f manip %.4f phases'% (io.mean(),io[129:615].mean()),[round(io[a:b].mean(),4) for a,b in [(0,129),(129,146),(146,520),(520,595),(595,615),(615,748)]],'rot %.3f p90 %.3f max %.2f dt %.4f p90 %.4f max %.4f'%(an[129:614].mean(),np.percentile(an[129:614],90),an[129:614].max(),dt[129:614].mean(),np.percentile(dt[129:614],90),dt[129:614].max()))
PY"
```
<details><summary>output</summary>

```
refined_v1b.npy iou all 0.7561 manip 0.7155 phases [0.8349, 0.8111, 0.6927, 0.7812, 0.8132, 0.828] rot 1.276 p90 2.372 max 4.50 dt 0.0077 p90 0.0157 max 0.0256
candidate_v2b_sg5.npy iou all 0.7551 manip 0.7139 phases [0.8349, 0.801, 0.6917, 0.78, 0.8079, 0.828] rot 0.933 p90 1.945 max 3.80 dt 0.0060 p90 0.0121 max 0.0217
candidate_v2b_sg7.npy iou all 0.7549 manip 0.7137 phases [0.8349, 0.8016, 0.6914, 0.78, 0.8071, 0.828] rot 0.848 p90 1.938 max 3.43 dt 0.0053 p90 0.0107 max 0.0204
candidate_v2b_sg9.npy iou all 0.7549 manip 0.7136 phases [0.8349, 0.801, 0.6913, 0.7797, 0.8068, 0.828] rot 0.803 p90 1.848 max 3.30 dt 0.0049 p90 0.0098 max 0.0203
candidate_v2b_sg11.npy iou all 0.7548 manip 0.7135 phases [0.8349, 0.8012, 0.6913, 0.7794, 0.8074, 0.828] rot 0.752 p90 1.764 max 3.06 dt 0.0045 p90 …[+19 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/vis_pool2_texture.py <<'PY'
import sys,cv2,h5py,numpy as np,torch
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
Ti=torch.tensor(np.load('/work/scratch/pooled2_initial.npy'),dtype=torch.float32,device='cuda');Tf=torch.tensor(np.load('/work/scratch/pooled2_final.npy'),dtype=torch.float32,device='cuda');v=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for fr in [0,20,50,80,100,125,620,640,660,680,700,747]:
 v.set(cv2.CAP_PROP_POS_FRAMES,fr);_,im=v.read();P=Ti if fr<300 else Tf
 rgb,a,_=o.render_pose(P,650,745,210,180);rr=(np.clip(rgb.detach().cpu().numpy(),0,1)*255).astype(np.uint8);mm=(a>.5).detach().cpu().numpy()
 with h5py.File('/work/inputs/masks.h5') as h:om=h['object'][fr,745:925,650:860]
 ys,xs=np.where(om);cx=int(xs.mean());cy=int(ys.mean());rad=48;x0=cx-rad;y0=cy-rad;x1=x0+96;y1=y0+96
 crop=im[745+y0:745+y1,650+x0:650+x1];rend=rr[y0:y1,x0:x1];ov=crop.copy();ov[mm[y0:y1,x0:x1]]=(0.25*ov[mm[y0:y1,x0:x1]]+0.75*rend[mm[y0:y1,x0:x1]]).astype(np.uint8)
 pair=np.concatenate([crop,ov],1);pair=cv2.resize(pair,(400,200),interpolation=cv2.INTER_NEAREST);cv2.putText(pair,str(fr),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2);ims.append(pair)
while len(ims)%4:ims.append(np.zeros_like(ims[0]))
out=np.concatenate([np.concatenate(ims[i:i+4],1) for i in range(0,len(ims),4)],0);cv2.imwrite('/work/scratch/vis/pool2_texture.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,97]);print(out.shape)
PY
python /work/scratch/vis_pool2_texture.py"
```
<details><summary>output</summary>

```
(600, 1600, 3)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py,cv2,json,trimesh
import nvdiffrast.torch as dr
D='cuda';mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D);F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous();j=json.load(open('/work/inputs/camera.json'));K=np.array(j['K']);Tc=torch.tensor(np.linalg.inv(np.array(j['T_world_cam'])),dtype=torch.float32,device=D);x0,y0,w,h=650,745,210,180;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];P=torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,-2*fy/h,1-2*(cy-y0)/h,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=D);ctx=dr.RasterizeCudaContext()
B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device=D);C=torch.tensor(np.load('/work/scratch/candidate_v2b_sg7.npy'),dtype=torch.float32,device=D)
with h5py.File('/work/inputs/masks.h5') as f:masks=f['object'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:deps=f['aligned/front_stereo_camera_left']
def predz(T):
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1);Pc=Tc@T;Vc=(Pc@Vh.T).T;clip=((P@Vc.T).T)[None].contiguous();r,_=dr.rasterize(ctx,clip,F,resolution=[h,w]);z,_=dr.interpolate(Vc[:,2:3][None],r,F);m=r[0,:,:,3]>0;return z[0,:,:,0].flip(0)[m],m.flip(0)
for i in [0,50,100,160,200,300,400,500,600,650,700,747]:
 dep=cv2.imdecode(deps[i],cv2.IMREAD_UNCHANGED).astype(float)/1000;m=masks[i,745:925,650:860];tgt=dep[745:925, …[+292 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
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
  File "/opt/conda/lib/python3.11/site-pack …[+795 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py,cv2,json,trimesh
import nvdiffrast.torch as dr
D='cuda';mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D);F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous();j=json.load(open('/work/inputs/camera.json'));K=np.array(j['K']);Tc=torch.tensor(np.linalg.inv(np.array(j['T_world_cam'])),dtype=torch.float32,device=D);x0,y0,w,h=650,745,210,180;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];P=torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,-2*fy/h,1-2*(cy-y0)/h,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=D);ctx=dr.RasterizeCudaContext()
B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device=D);C=torch.tensor(np.load('/work/scratch/candidate_v2b_sg7.npy'),dtype=torch.float32,device=D)
with h5py.File('/work/inputs/masks.h5') as f:masks=f['object'][:,y0:y0+h,x0:x0+w]
def predz(T):
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1);Pc=Tc@T;Vc=(Pc@Vh.T).T;clip=((P@Vc.T).T)[None].contiguous();r,_=dr.rasterize(ctx,clip,F,resolution=[h,w]);z,_=dr.interpolate(Vc[:,2:3][None],r,F);m=r[0,:,:,3]>0;return z[0,:,:,0].flip(0)[m],m.flip(0)
for i in [0,50,100,160,200,300,400,500,600,650,700,747]:
 with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:dep=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
 m=masks[i];tgt=dep[m]
 vals=[]
 for  …[+278 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1152 but corresponding boolean dimension is 180

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py,cv2,json,trimesh
import nvdiffrast.torch as dr
D='cuda';mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D);F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous();j=json.load(open('/work/inputs/camera.json'));K=np.array(j['K']);Tc=torch.tensor(np.linalg.inv(np.array(j['T_world_cam'])),dtype=torch.float32,device=D);x0,y0,w,h=650,745,210,180;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];P=torch.tensor([[2*fx/w,0,2*(cx-x0)/w-1,0],[0,-2*fy/h,1-2*(cy-y0)/h,0],[0,0,1,0],[0,0,1,0]],dtype=torch.float32,device=D);ctx=dr.RasterizeCudaContext()
B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device=D);C=torch.tensor(np.load('/work/scratch/candidate_v2b_sg7.npy'),dtype=torch.float32,device=D)
with h5py.File('/work/inputs/masks.h5') as f:masks=f['object'][:,y0:y0+h,x0:x0+w]
def predz(T):
 Vh=torch.cat([V,torch.ones_like(V[:,:1])],1);Pc=Tc@T;Vc=(Pc@Vh.T).T;clip=((P@Vc.T).T)[None].contiguous();r,_=dr.rasterize(ctx,clip,F,resolution=[h,w]);z,_=dr.interpolate(Vc[:,2:3][None],r,F);m=r[0,:,:,3]>0;return z[0,:,:,0].flip(0)[m]
for i in [0,50,100,160,200,300,400,500,600,650,700,747]:
 with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:dep=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)/1000
 m=masks[i];tgt=dep[y0:y0+h,x0:x0+w][m];vals=[] …[+274 chars]
```
<details><summary>output</summary>

```
0 target 3.035 base/diff 0.0 -3.035 final/diff 0.0 -3.035
50 target 2.896 base/diff 0.0 -2.896 final/diff 0.0 -2.896
100 target 2.96 base/diff 0.0 -2.96 final/diff 0.0 -2.96
160 target 2.907 base/diff 2.957 0.05 final/diff 2.913 0.006
200 target 2.994 base/diff 0.0 -2.994 final/diff 0.0 -2.994
300 target 3.085 base/diff 0.0 -3.085 final/diff 0.0 -3.085
400 target 3.044 base/diff 0.0 -3.044 final/diff 0.0 -3.044
500 target 3.15 base/diff 0.0 -3.15 final/diff 0.0 -3.15
600 target 2.903 base/diff 0.0 -2.903 final/diff 0.0 -2.903
650 target 2.897 base/diff 0.0 -2.897 final/diff 0.0 -2.897
700 target 3.099 base/diff 0.0 -3.099 final/diff 0.0 -3.099
747 target 3.068 base/diff 0.0 -3.068 final/diff 0.0 -3.068

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# quickly inspect predz on frame0 using code imported? no
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_depth_grid.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,json,trimesh
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
x0,y0,w,h=650,745,210,180;cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs=[]
while True:
 ok,im=cap.read()
 if not ok:break
 imgs.append(im[y0:y0+h,x0:x0+w])
cap.release();imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w]
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Tc=torch.tensor(np.linalg.inv(np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])),dtype=torch.float32,device=o.D)
mm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry().convex_hull;HV=torch.tensor(mm.vertices,dtype=torch.float32,device=o.D);pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);pn=torch.tensor(pl[:3],dtype=torch.float32,device=o.D);pd=float(pl[3])
def run(name,a,b,P0,lam):
 img=torch.tensor(np.median(imgs[a:b].astype(np.float32)/255,axis=0),device=o.D);tar=torch.tensor(objs[a:b].mean(0),device=o.D);imhp=img-o.blur(img[None])[0]
 T0=torch.tensor(P0,dtype=torch.float32,device=o.D);r=torch.zeros(3,device=o.D,requires_grad=True);dt=torch.zeros(3,device=o.D,requires_grad=True);op=torch.optim.Adam([r,dt],lr=.0007)
 z0=float((Tc@T0)[2,3])
 for it in range(130):
  op.zero_grad();Rd=o.rodrigues(r);Tw=T0.clone();Tw[:3,:3]=Rd@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt
  rgb,alpha,_=o.render_pose(Tw,x0,y0,w,h) …[+1175 chars]
```
<details><summary>output</summary>

```
initial 0 iou 0.8703697007973845 cos 0.031970524485293424 floor 0.0561223030090332 z 2.7992310523986816 2.8149960041046143
initial 1 iou 0.8690445636361069 cos 0.04464779227579596 floor 0.055910348892211914 z 2.800886631011963 2.8149960041046143
initial 3 iou 0.8663028636844738 cos 0.05520221368402338 floor 0.05493009090423584 z 2.8027431964874268 2.8149960041046143
initial 10 iou 0.8652772686239355 cos 0.05090318500789438 floor 0.05404841899871826 z 2.805401563644409 2.8149960041046143
initial 30 iou 0.8657623917557637 cos 0.04915672689492902 floor 0.05166327953338623 z 2.8103744983673096 2.8149960041046143
final 0 iou 0.8729248076040735 cos 0.01761284575020213 floor 0.04400312900543213 z 2.8420357704162598 2.873988389968872
final 1 iou 0.8741743305284029 cos 0.008258621193212546 floor 0. …[+422 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
flips=[]
for ax in range(3):
 R=torch.eye(3,device='cuda');R[(ax+1)%3,(ax+1)%3]=-1;R[(ax+2)%3,(ax+2)%3]=-1;flips.append(R)
for fr in [0,50,100,160,300,400,500,650,700,747]:
 t=torch.tensor(objs[fr],device='cuda').float();p=torch.tensor(pers[fr],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);vals=[]
 for Fi in [None]+flips:
  T=B[fr].clone()
  if Fi is not None:T[:3,:3]=T[:3,:3]@Fi
  pr=hm.render(T,ss=2)>.5;vals.append(float((pr&tar).sum()/(pr|tar).sum()))
 print(fr,np.round(vals,3))
PY"
```
<details><summary>output</summary>

```
0 [0.662 0.577 0.514 0.492]
50 [0.73  0.64  0.545 0.511]
100 [0.714 0.612 0.529 0.504]
160 [0.647 0.577 0.475 0.457]
300 [0.558 0.515 0.374 0.373]
400 [0.553 0.509 0.371 0.358]
500 [0.412 0.444 0.376 0.313]
650 [0.721 0.649 0.528 0.494]
700 [0.683 0.612 0.502 0.463]
747 [0.678 0.6   0.49  0.459]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_metrics.py <<'PY'
import sys,numpy as np,torch,h5py,json,trimesh
sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];C=np.load('/work/scratch/candidate_v2b_sg7.npy')
x0,y0,w,h=650,745,210,180
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w];pers=f['person'][:,y0:y0+h,x0:x0+w]
cts=[]
for i in range(748):
 t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);cts.append(tar)
def iou_centroid(T):
 Ts=torch.tensor(T,dtype=torch.float32,device='cuda');ious=[];ce=[]
 for i in range(748):
  pr=hm.render(Ts[i],ss=2)>.5;tar=cts[i];inter=(pr&tar).sum();ious.append(float(inter/(pr|tar).sum()))
  yy,xx=torch.where(pr);ty,tx=torch.where(tar)
  if len(xx) and len(tx):ce.append(float(torch.sqrt((xx.float().mean()-tx.float().mean())**2+(yy.float().mean()-ty.float().mean())**2)))
  else:ce.append(float('nan'))
 return np.array(ious),np.array(ce)
bi,bc=iou_centroid(B);ci,cc=iou_centroid(C)
# rotation and translation smoothness
def smooth(T):
 R=T[:,:3,:3];Rd=np.einsum('nij,nkj->nik',R[:-1],R[1:]);rot=np.rad2deg(np.arccos(np.clip((np.trace(Rd,axis1=1,axis2=2)-1)/2,-1,1)));tr=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1);return rot,tr
br,bt=smooth(B);cr,ct=smooth(C)
# floor hull
m=trimesh.load('/work/inputs/object_mesh.glb' …[+2921 chars]
```
<details><summary>output</summary>

```
{
  "rest_before": {
    "frames": [
      0,
      128
    ],
    "baseline_iou": 0.7126501248788464,
    "result_iou": 0.8349363004514413,
    "baseline_centroid_px": 2.2293238334877548,
    "result_centroid_px": 1.4015061799869981,
    "baseline_rot_step_deg": 1.378971815109253,
    "result_rot_step_deg": 0.01827218688007443,
    "baseline_trans_step_m": 0.01730170100927353,
    "result_trans_step_m": 0.0,
    "baseline_floor_min_cm": -0.8394360542297363,
    "result_floor_min_cm": 4.920506477355957,
    "baseline_hand_dist_cm": 15.424123406410217,
    "result_hand_dist_cm": 22.933630645275116
  },
  "grasp_transition": {
    "frames": [
      129,
      159
    ],
    "baseline_iou": 0.6720009080825313,
    "result_iou": 0.7939396077586759,
    "baseline_centroid_px": 4.865290795603106 …[+3060 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,140p' /work/scratch/opt_test_aa2.py && sed -n '1,180p' /work/scratch/pooled_static.py && sed -n '1,160p' /work/scratch/pooled_refine.py"
```
<details><summary>output</summary>

```
import numpy as np, torch, trimesh, h5py, json, cv2, time, math
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
D=torch.device('cuda'); torch.manual_seed(0)
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
# decimate? no
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=D)
F=torch.tensor(mesh.faces,dtype=torch.int32,device=D).contiguous()
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=D)
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255
tex=torch.tensor(tex,device=D).permute(2,0,1)[None]
z=np.load('/work/inputs/reconstruction.npz'); T=torch.tensor(z['T_world_mesh'],dtype=torch.float32,device=D)
j=json.load(open('/work/inputs/camera.json')); K=np.array(j['K']); Twc=np.array(j['T_world_cam']); …[+9286 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF watering-can trajectory.

The method uses the supplied FoundationPose/CoCoNet world trajectory only as an
initialisation.  It fits the exact textured mesh to the object mask and to
gradient/texture detail in a fixed image ROI, pools the two rest phases into a
single rigid pose, then smooths and re-fits the manipulation interval.  No
network access and no episode-specific pose table is used.

Usage:
    python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
import nvdiffrast.torch as dr

# ---------------------------------------------------------------------------
# Fixed choices verified from the video/masks (see REPORT.md).
# ROI contains the object in every frame.  Frame boundaries are annotated
# explicitly because decisions made from visual inspection must be reproducible.
# ---------------------------------------------------------------------------
ROI_X0, ROI_Y0, ROI_W, ROI_H = 650, 745, 210, 180
INITIAL_REST_END = 129       # hand begins lifting/turning the can after this frame
INITIAL_BLEND_END = 146
FINAL_ …[+16576 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -10 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
loaded 748 frames, mesh=297436 vertices
per-frame 0/748 (3.6s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
loaded 748 frames, mesh=297436 vertices
per-frame 0/748 (3.6s)
per-frame 25/748 (279.2s)
per-frame 50/748 (292.4s)
per-frame 75/748 (305.0s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/solve_run.log && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```
<details><summary>output</summary>

```
per-frame 100/748 (314.8s)
per-frame 125/748 (320.3s)
per-frame 150/748 (326.1s)
per-frame 175/748 (332.3s)
per-frame 200/748 (339.6s)
1409 MiB, 49140 MiB
14325 MiB, 49140 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
per-frame 200/748 (339.6s)
per-frame 225/748 (347.7s)
per-frame 250/748 (356.0s)
per-frame 275/748 (364.2s)
per-frame 300/748 (372.4s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
per-frame 350/748 (388.1s)
per-frame 375/748 (394.0s)
per-frame 400/748 (399.8s)
per-frame 425/748 (405.4s)
per-frame 450/748 (412.8s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
per-frame 425/748 (405.4s)
per-frame 450/748 (412.8s)
per-frame 475/748 (420.0s)
per-frame 500/748 (427.5s)
per-frame 525/748 (439.1s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -5 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
per-frame 525/748 (439.1s)
per-frame 550/748 (451.5s)
per-frame 575/748 (464.0s)
per-frame 600/748 (470.7s)
per-frame 625/748 (476.8s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -8 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
per-frame 575/748 (464.0s)
per-frame 600/748 (470.7s)
per-frame 625/748 (476.8s)
per-frame 650/748 (483.2s)
per-frame 675/748 (490.7s)
per-frame 700/748 (499.4s)
per-frame 725/748 (507.5s)
pooled initial (516.4s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -8 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
pooled initial (516.4s)
pooled final (517.7s)
refine 150/615 (525.0s)
refine 175/615 (529.9s)
refine 200/615 (534.8s)
refine 225/615 (540.9s)
refine 250/615 (545.2s)
refine 275/615 (550.3s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -8 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
refine 250/615 (545.2s)
refine 275/615 (550.3s)
refine 300/615 (555.6s)
refine 325/615 (561.2s)
refine 350/615 (566.7s)
refine 375/615 (572.1s)
refine 400/615 (577.6s)
refine 425/615 (584.6s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'tail -8 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
refine 400/615 (577.6s)
refine 425/615 (584.6s)
refine 450/615 (592.4s)
refine 475/615 (599.4s)
refine 500/615 (603.6s)
refine 525/615 (608.2s)
refine 550/615 (614.0s)
refine 575/615 (620.4s)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cd /work && python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > /work/scratch/solve_run.log 2>&1; tail -30 /work/scratch/solve_run.log'
```
<details><summary>output</summary>

```
per-frame 550/748 (451.5s)
per-frame 575/748 (464.0s)
per-frame 600/748 (470.7s)
per-frame 625/748 (476.8s)
per-frame 650/748 (483.2s)
per-frame 675/748 (490.7s)
per-frame 700/748 (499.4s)
per-frame 725/748 (507.5s)
pooled initial (516.4s)
pooled final (517.7s)
refine 150/615 (525.0s)
refine 175/615 (529.9s)
refine 200/615 (534.8s)
refine 225/615 (540.9s)
refine 250/615 (545.2s)
refine 275/615 (550.3s)
refine 300/615 (555.6s)
refine 325/615 (561.2s)
refine 350/615 (566.7s)
refine 375/615 (572.1s)
refine 400/615 (577.6s)
refine 425/615 (584.6s)
refine 450/615 (592.4s)
refine 475/615 (599.4s)
refine 500/615 (603.6s)
refine 525/615 (608.2s)
refine 550/615 (614.0s)
refine 575/615 (620.4s)
refine 600/615 (626.2s)
wrote /work/outputs/object_poses_world.npz shape=(748, 4, 4) in 630.4s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
A=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/scratch/candidate_v2b_sg7.npy')
print('shapes',A.shape,A.dtype)
print('finite',np.isfinite(A).all(),'bottom',np.max(np.abs(A[:,3]-[0,0,0,1])))
c=np.linalg.norm(A[:,:3,3]-B[:,:3,3],axis=1);dd=[]
for i in range(len(A)):
 Rd=A[i,:3,:3]@B[i,:3,:3].T;dd.append(np.rad2deg(np.arccos(np.clip((np.trace(Rd)-1)/2,-1,1))))
dd=np.array(dd)
print('diff trans max/mean/med',c.max(),c.mean(),np.median(c),'rot max/mean/med',dd.max(),dd.mean(),np.median(dd))
for i in [0,128,129,145,146,150,300,595,614,615,700,747]:print(i,'dt',c[i],'rd',dd[i])
PY"
```
<details><summary>output</summary>

```
shapes (748, 4, 4) float64
finite True bottom 0.0
diff trans max/mean/med 0.03303916725597839 0.010702804276149317 0.0030416076052352525 rot max/mean/med 3.0283519490654216 0.6833878156454087 0.6311712591377777
0 dt 0.01601043928225362 rd 0.6311712591377777
128 dt 0.01601043928225362 rd 0.6311712591377777
129 dt 0.016004761916327204 rd 0.6335359165091355
145 dt 0.001841436340525694 rd 0.26175237157216347
146 dt 0.0029087778656911247 rd 0.5129581654893021
150 dt 0.0008549280126040998 rd 0.1112734196681395
300 dt 0.0030710870739456505 rd 0.21788358674707053
595 dt 0.0060244829259933535 rd 1.006942655953209
614 dt 0.03303216594737787 rd 1.4083469570241622
615 dt 0.03303916725597839 rd 1.410144103253559
700 dt 0.03303916725597839 rd 1.410144103253559
747 dt 0.03303916725597839 rd 1.41014410325 …[+5 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -lh /work/outputs/object_poses_world.npz /work/outputs/solve.py && sha256sum /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 94K Oct 20 23:10 /work/outputs/object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 18K Oct 20 23:00 /work/outputs/solve.py
aec523ac087ab284481fa65a8f460d7272439459b971e712b0195b57b9ebe81c  /work/outputs/object_poses_world.npz

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed \"s#candidate_v2b_sg7.npy#../../outputs/object_poses_world.npz#; s#np.load('/work/scratch/../../outputs/object_poses_world.npz')#np.load('/work/outputs/object_poses_world.npz')#; s#if __name__=='__main__':#if True:#\" /work/scratch/final_metrics.py > /work/scratch/final_metrics_out.py
# simpler sed path line
sed -i \"s#C=np.load('/work/scratch/candidate_v2b_sg7.npy')#C=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']#\" /work/scratch/final_metrics_out.py
python /work/scratch/final_metrics_out.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/final_metrics_out.py", line 17, in <module>
    bi,bc=iou_centroid(B);ci,cc=iou_centroid(C)
                                ^^^^^^^^^^^^^^^
  File "/work/scratch/final_metrics_out.py", line 10, in iou_centroid
    Ts=torch.tensor(T,dtype=torch.float32,device='cuda');ious=[];ce=[]
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: could not determine the shape of object type 'NpzFile'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'head -5 /work/scratch/final_metrics_out.py'
```
<details><summary>output</summary>

```
import sys,numpy as np,torch,h5py,json,trimesh
sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];C=np.load('/work/outputs/object_poses_world.npz')
x0,y0,w,h=650,745,210,180
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w];pers=f['person'][:,y0:y0+h,x0:x0+w]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s#C=np.load('/work/outputs/object_poses_world.npz')#C=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']#\" /work/scratch/final_metrics_out.py && python /work/scratch/final_metrics_out.py"
```
<details><summary>output</summary>

```
{
  "rest_before": {
    "frames": [
      0,
      128
    ],
    "baseline_iou": 0.7126501248788464,
    "result_iou": 0.8355593695196994,
    "baseline_centroid_px": 2.2293238334877548,
    "result_centroid_px": 1.5472509870233462,
    "baseline_rot_step_deg": 1.378971815109253,
    "result_rot_step_deg": 0.0,
    "baseline_trans_step_m": 0.01730170100927353,
    "result_trans_step_m": 0.0,
    "baseline_floor_min_cm": -0.8394360542297363,
    "result_floor_min_cm": 5.548489093780518,
    "baseline_hand_dist_cm": 15.424123406410217,
    "result_hand_dist_cm": 23.97097945213318
  },
  "grasp_transition": {
    "frames": [
      129,
      159
    ],
    "baseline_iou": 0.6720009080825313,
    "result_iou": 0.794172800356342,
    "baseline_centroid_px": 4.865290795603106,
    "result_cent …[+3056 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '/work/scratch\\|candidate\\|opt_all\\|pooled_' /work/outputs/solve.py || true"
```
<details><summary>output</summary>

```
448:    candidate = apply_rest_constraints(smooth9, initial_pose, final_pose)
452:    refined = candidate.copy()
456:            candidate[i],

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz'
z=np.load(p)
print(z.files)
T=z['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all())
print('rotation determinants',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
print('orthogonality max',np.max(np.abs(np.einsum('nij,nkj->nik',T[:,:3,:3],T[:,:3,:3])-np.eye(3))))
PY"
```
<details><summary>output</summary>

```
['T_world_mesh']
(748, 4, 4) float64 True
rotation determinants 0.9999999265039464 1.000000128517751
orthogonality max 1.937930371731511e-07

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
c=T[:,:3,3];print('world translation range',c.min(0),c.max(0));print('baseline range',B[:,:3,3].min(0),B[:,:3,3].max(0))
print('static unique rotation spread')
for a,b in [(0,129),(615,748)]:
 R=T[a:b,:3,:3];print(a,b,'trans std',T[a:b,:3,3].std(0),'rot max angle dev',np.rad2deg(np.arccos(np.clip((np.trace(np.einsum('nij,nkj->nik',R,R[0].T),axis1=1,axis2=2)-1)/2,-1,1))).max())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/einsumfunc.py", line 1371, in einsum
world translation range [-0.02978142  0.99803842  2.81230299] [0.12846276 1.16073314 3.13672502]
baseline range [-0.02518946  0.98880756  2.8643103 ] [0.13276309 1.1761553  3.1956618 ]
static unique rotation spread
    return c_einsum(*operands, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: einstein sum subscripts string contains too many subscripts for operand 1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for a,b in [(0,129),(615,748)]:
 R=T[a:b,:3,:3]; print(a,b,'trans std',T[a:b,:3,3].std(0),'max center diff',np.abs(T[a:b,:3,3]-T[a,:3,3]).max(),'max R diff',np.abs(T[a:b,:3,:3]-T[a,:3,:3]).max())
PY"
```
<details><summary>output</summary>

```
0 129 trans std [0. 0. 0.] max center diff 0.0 max R diff 0.0
615 748 trans std [0. 0. 0.] max center diff 0.0 max R diff 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import importlib.metadata as m
for p in ['torch','numpy','scipy','opencv-python','opencv-python-headless','h5py','trimesh','nvdiffrast','pytorch3d','open3d','Pillow']:
 try: print(p,m.version(p))
 except Exception as e: print(p,'not-installed')
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124
numpy 1.26.3
scipy 1.16.3
opencv-python 4.11.0.86
opencv-python-headless not-installed
h5py 3.16.0
trimesh 5.1.1
nvdiffrast 0.4.0
pytorch3d 0.7.9
open3d 0.20.0
Pillow 10.2.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def ang(A,C):return np.rad2deg(np.arccos(np.clip((np.trace(A[:3,:3]@C[:3,:3].T)-1)/2,-1,1)))
Ti=T[0];Tf=T[-1];Bi=B[0];Bf=B[-1]
print('initial center',Ti[:3,3],'baseline',Bi[:3,3],'delta',Ti[:3,3]-Bi[:3,3],'rot delta',ang(Ti,Bi))
print('final center',Tf[:3,3],'baseline',Bf[:3,3],'delta',Tf[:3,3]-Bf[:3,3],'rot delta',ang(Tf,Bf))
print('init-final rel',np.linalg.norm(Tf[:3,3]-Ti[:3,3]),ang(Tf,Ti))
# per-frame correction stats vs baseline
ds=np.linalg.norm(T[:,:3,3]-B[:,:3,3],axis=1);rs=np.array([ang(T[i],B[i]) for i in range(748)])
print('corr trans',np.percentile(ds,[0,25,50,75,100]),'rot',np.percentile(rs,[0,25,50,75,100]))
print('overall path rotation total',sum(np.rad2deg(np.arccos(np.clip((np.trace(T[i,:3,:3]@T[i-1,:3,:3].T)-1)/2,-1,1))) for i in range(1,748)))
PY"
```
<details><summary>output</summary>

```
initial center [0.03182116 1.09427416 2.81232572] baseline [0.03907919 1.1338623  2.9643936 ] delta [-0.00725803 -0.03958809 -0.1520679 ] rot delta 4.1362135709654355
final center [0.02287424 1.10168767 2.85572577] baseline [0.03875534 1.157751   3.0313237 ] delta [-0.0158811  -0.05606329 -0.17559791] rot delta 5.422136590282535
init-final rel 0.04492851831099169 15.373254953451685
corr trans [0.0107834  0.02868564 0.04776129 0.15497326 0.22548852] rot [ 1.79890274  5.11706008  6.03403417  6.83699261 13.19588737]
overall path rotation total 412.2981680125899

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/fixed_depth_test.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,json,trimesh
import torch.nn.functional as Fn
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
from scipy.spatial.transform import Rotation
x0,y0,w,h=650,745,210,180
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs=[]
while True:
 ok,im=cap.read()
 if not ok:break
 imgs.append(im[y0:y0+h,x0:x0+w])
cap.release();imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w]
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];P=np.load('/work/scratch/opt_all_photo55.npz')['T_world_mesh'];Tc=torch.tensor(o.Tc,dtype=torch.float32,device='cuda');Rwc=torch.tensor(o.Tc[:3,:3],device='cuda')
mm=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry().convex_hull;HV=torch.tensor(mm.vertices,dtype=torch.float32,device='cuda');pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);pn=torch.tensor(pl[:3],dtype=torch.float32,device='cuda');pd=float(pl[3])
def rav(Rs):
 q=Rotation.from_matrix(Rs).as_quat();q0=q[0].copy()
 for i in range(len(q)):
  if np.dot(q[i],q0)<0:q[i]*=-1
 q=q.mean(0);q/=np.linalg.norm(q);return Rotation.from_quat(q).as_matrix()
def run(a,b,dz,steps=180):
 img=torch.tensor(np.median(imgs[a:b].astype(np.float32)/255,axis=0),device='cuda');tar=torch.tensor(objs[a:b].mean(0),device='cuda');imhp=img-o.blur(img[None])[0]
 # start from average optimized rotation, medi …[+1827 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fixed_depth_test.py", line 38, in <module>
    T,iou,cos,fl,z0=run(a,b,dz);np.save(f'/work/scratch/fd_{name}_{dz}.npy',T);print(name,dz,'z0',z0,'iou',iou,'cos',cos,'floor',fl,'cent',T[:3,3],flush=True)
                    ^^^^^^^^^^^
  File "/work/scratch/fixed_depth_test.py", line 28, in run
    opt.zero_grad();dt=Rwc@torch.cat([duv,torch.zeros(1,device='cuda')]);Tw=o.rodrigues(r)@T0;Tw[:3,3]=T0[:3,3]+dt
                       ~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RuntimeError: expected scalar type Double but found Float

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s#Rwc=torch.tensor(o.Tc\\[:3,:3\\],device='cuda')#Rwc=torch.tensor(o.Tc[:3,:3],device='cuda',dtype=torch.float32)#\" /work/scratch/fixed_depth_test.py && python /work/scratch/fixed_depth_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fixed_depth_test.py", line 38, in <module>
    T,iou,cos,fl,z0=run(a,b,dz);np.save(f'/work/scratch/fd_{name}_{dz}.npy',T);print(name,dz,'z0',z0,'iou',iou,'cos',cos,'floor',fl,'cent',T[:3,3],flush=True)
                    ^^^^^^^^^^^
  File "/work/scratch/fixed_depth_test.py", line 28, in run
    opt.zero_grad();dt=Rwc@torch.cat([duv,torch.zeros(1,device='cuda')]);Tw=o.rodrigues(r)@T0;Tw[:3,3]=T0[:3,3]+dt
                                                                            ~~~~~~~~~~~~~~^~~
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x3 and 4x4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/fixed_depth_test.py';s=open(p).read();s=s.replace(\"Tw=o.rodrigues(r)@T0;Tw[:3,3]=T0[:3,3]+dt\",\"Tw=T0.clone();Tw[:3,:3]=o.rodrigues(r)@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt\");open(p,'w').write(s)
PY
python /work/scratch/fixed_depth_test.py"
```
<details><summary>output</summary>

```
initial -0.1 z0 2.8588593218156815 iou 0.8440169453327884 cos -0.0010647474964905557 floor 0.027679085731506348 cent [0.0345097 1.1177745 2.869068 ]
initial -0.06 z0 2.8988593218156815 iou 0.8303815414177295 cos 0.041295150290070304 floor 0.009775280952453613 cent [0.03563314 1.1331866  2.9092543 ]
initial -0.03 z0 2.9288593218156818 iou 0.8164471479482602 cos 0.07053555264331073 floor -0.0029189586639404297 cent [0.0351543 1.1437491 2.9392803]
initial 0 z0 2.9588593218156816 iou 0.8046499210745516 cos 0.0664746990286284 floor -0.01597273349761963 cent [0.03658091 1.1547699  2.9694438 ]
initial 0.03 z0 2.9888593218156814 iou 0.7887632755079794 cos 0.08072261973497506 floor -0.02830803394317627 cent [0.03718648 1.165526   2.9995468 ]
final -0.1 z0 2.9125345899922612 iou 0.8467743498999817 c …[+678 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];O=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(m.vertices[::20],dtype=torch.float32,device='cuda')
def dist(T):
 T=torch.tensor(T,dtype=torch.float32,device='cuda');out=[]
 for i in range(748):out.append(float(torch.cdist(kp[i],V@T[i,:3,:3].T+T[i,:3,3]).min()))
 return np.array(out)
for name,T in [('base',B),('out',O)]:
 d=dist(T)
 print(name,[(a,b,round(d[a:b].mean()*100,2),round(d[a:b].max()*100,2)) for a,b in [(0,100),(100,110),(110,120),(120,129),(129,146),(146,160)]])
PY"
```
<details><summary>output</summary>

```
base [(0, 100, 16.5, 29.14), (100, 110, 2.76, 4.52), (110, 120, 2.36, 3.22), (120, 129, 2.76, 3.46), (129, 146, 1.34, 1.85), (146, 160, 0.47, 1.77)]
out [(0, 100, 25.12, 37.22), (100, 110, 14.34, 18.09), (110, 120, 7.86, 10.93), (120, 129, 8.6, 11.08), (129, 146, 5.38, 10.1), (146, 160, 0.69, 3.16)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];O=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(m.vertices[::20],dtype=torch.float32,device='cuda')
for name,T in [('base',B),('out',O)]:
 T=torch.tensor(T,dtype=torch.float32,device='cuda');ds=[]
 for i in range(748):ds.append(float(torch.cdist(kp[i],V@T[i,:3,:3].T+T[i,:3,3]).min()))
 ds=np.array(ds);print(name,[(a,b,round(ds[a:b].mean()*100,2),round(ds[a:b].max()*100,2)) for a,b in [(580,600),(600,615),(615,625),(625,635),(635,645),(645,660)]])
PY"
```
<details><summary>output</summary>

```
base [(580, 600, 1.52, 2.69), (600, 615, 2.4, 3.4), (615, 625, 4.09, 5.87), (625, 635, 6.94, 9.39), (635, 645, 10.2, 10.62), (645, 660, 8.94, 13.11)]
out [(580, 600, 0.71, 1.52), (600, 615, 2.17, 5.31), (615, 625, 4.42, 7.98), (625, 635, 5.92, 7.94), (635, 645, 10.26, 11.74), (645, 660, 11.43, 14.13)]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_hand_test.py <<'PY'
import sys,numpy as np,torch,cv2,h5py,json,trimesh
sys.path.insert(0,'/work/scratch');import opt_test_aa2 as o
x0,y0,w,h=650,745,210,180
cap=cv2.VideoCapture('/work/inputs/video.mp4');imgs=[]
while True:
 ok,im=cap.read()
 if not ok:break
 imgs.append(im[y0:y0+h,x0:x0+w])
cap.release();imgs=np.stack(imgs)
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,y0:y0+h,x0:x0+w]
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');cp=torch.tensor(z['contact_prob_right_left'][:,1],dtype=torch.float32,device='cuda')
mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(mesh.vertices[::20],dtype=torch.float32,device='cuda');HV=torch.tensor(mesh.convex_hull.vertices,dtype=torch.float32,device='cuda');pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);pn=torch.tensor(pl[:3],dtype=torch.float32,device='cuda');pd=float(pl[3])
a,b=0,129;img=torch.tensor(np.median(imgs[a:b].astype(np.float32)/255,axis=0),device='cuda');tar=torch.tensor(objs[a:b].mean(0),device='cuda');imhp=img-o.blur(img[None])[0]
P0=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'][0]
for hw in [0,1,3,10,30,100]:
 T0=torch.tensor(P0,dtype=torch.float32,device='cuda');r=torch.zeros(3,device='cuda',requires_grad=True);dt=torch.zeros(3,device='cuda',requires_grad=True);opt=torch.optim.Ad …[+1567 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/static_hand_test.py", line 18, in <module>
    opt.zero_grad();Tw=o.rodrigues(r)@T0;Tw[:3,3]=T0[:3,3]+dt;rgb,alpha,_=o.render_pose(Tw,x0,y0,w,h);pv=alpha;inter=(pv*tar).sum();union=(pv*pv).sum()+(tar*tar).sum()-inter;iou=inter/(union+1e-9)
                       ~~~~~~~~~~~~~~^~~
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x3 and 4x4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_hand_test.py';s=open(p).read();s=s.replace(\"Tw=o.rodrigues(r)@T0;Tw[:3,3]=T0[:3,3]+dt;rgb\",\"Tw=T0.clone();Tw[:3,:3]=o.rodrigues(r)@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt;rgb\");s=s.replace(\"Tw=o.rodrigues(r)@T0;Tw[:3,3]=T0[:3,3]+dt;W=\",\"Tw=T0.clone();Tw[:3,:3]=o.rodrigues(r)@T0[:3,:3];Tw[:3,3]=T0[:3,3]+dt;W=\");open(p,'w').write(s)
PY
python /work/scratch/static_hand_test.py"
```
<details><summary>output</summary>

```
hw 0 iou 0.8756762759842274 cos 0.041293522225497634 floor 0.059551119804382324 hand110-128 mean/max 0.08778800031072215 0.11672679334878922 center [0.03184905 1.0905944  2.80549   ] delta [ 2.7891248e-05 -3.6797523e-03 -6.8356991e-03]
hw 1 iou 0.8730220968491504 cos 0.024407629734148506 floor 0.05856502056121826 hand110-128 mean/max 0.0852182758481879 0.1140865758061409 center [0.03200947 1.0914319  2.8083382 ] delta [ 0.00018832 -0.00284231 -0.00398755]
hw 3 iou 0.871178487311211 cos 0.050423751217732515 floor 0.05579495429992676 hand110-128 mean/max 0.08125738917212737 0.10999666154384613 center [0.03200406 1.0938573  2.8133485 ] delta [ 0.0001829  -0.00041687  0.00102282]
hw 10 iou 0.8659869838513063 cos 0.041899986097057834 floor 0.04591965675354004 hand110-128 mean/max 0.062522986119 …[+556 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/for hw in \\[0,1,3,10,30,100\\]:/for hw in [300,1000]:/' /work/scratch/static_hand_test.py && python /work/scratch/static_hand_test.py"
```
<details><summary>output</summary>

```
hw 300 iou 0.8479196249318128 cos 0.07425403797727959 floor 0.03212547302246094 hand110-128 mean/max 0.035817807466772045 0.06260672211647034 center [0.03393123 1.1136576  2.8635378 ] delta [0.00211008 0.01938343 0.05121207]
hw 1000 iou 0.8359615787271585 cos 0.05732257253957268 floor 0.0344313383102417 hand110-128 mean/max 0.034720894292389094 0.06211736053228378 center [0.03259659 1.1107212  2.8632712 ] delta [0.00077543 0.01644707 0.05094552]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
T=torch.tensor(np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device='cuda')
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:,745:925,650:860];p=f['person'][:,745:925,650:860]
for i in [0,50,100,650,700,747]:
 t=torch.tensor(o[i],device='cuda').float();pp=torch.tensor(p[i],device='cuda').float();pp=torch.nn.functional.max_pool2d(pp[None,None],5,1,2)[0,0];tar=((t*(1-pp).clamp(0,1))>.5);print(i,'target',int(tar.sum()),end=' ')
 for X in [B[i],T[i]]:
  pr=hm.render(X,ss=2)>.5;inter=(pr&tar).sum();print('area',int(pr.sum()),'iou',round(float(inter/(pr|tar).sum()),3),end=' ')
 print()
PY"
```
<details><summary>output</summary>

```
0 target 2541 area 1924 iou 0.662 area 2176 iou 0.845 
50 target 2481 area 1908 iou 0.73 area 2176 iou 0.842 
100 target 2445 area 1860 iou 0.714 area 2176 iou 0.848 
650 target 2157 area 1711 iou 0.721 area 1990 iou 0.837 
700 target 2330 area 1692 iou 0.683 area 1990 iou 0.835 
747 target 2306 area 1678 iou 0.678 area 1990 iou 0.837 

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
v=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:m=h['object'][:]
for i in [0,50,100,160,300,500,650,700,747]:
 v.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=v.read();im=im.astype(np.float32);g=im[:,:,1]-np.maximum(im[:,:,0],im[:,:,2]);
 for th in [10,20,30,40,50]:
  c=g>th
  inter=(c&m[i]).sum();uni=(c|m[i]).sum();print(i,th,'area',c.sum(),'iou',round(inter/uni,3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
NameError: name 'h' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
v=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:masks=f['object'][:]
for i in [0,50,100,160,300,500,650,700,747]:
 v.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=v.read();im=im.astype(np.float32);g=im[:,:,1]-np.maximum(im[:,:,0],im[:,:,2]);m=masks[i]
 print(i,end=' ')
 for th in [10,20,30,40,50]:
  c=g>th; inter=(c&m).sum();uni=(c|m).sum();print(th,'area',int(c.sum()),'iou',round(float(inter/uni),3),end='; ')
 print()
PY"
```
<details><summary>output</summary>

```
0 10 area 3036 iou 0.78; 20 area 2232 iou 0.865; 30 area 1888 iou 0.743; 40 area 1496 iou 0.589; 50 area 416 iou 0.164; 
50 10 area 3107 iou 0.746; 20 area 2195 iou 0.829; 30 area 1864 iou 0.73; 40 area 1504 iou 0.591; 50 area 380 iou 0.149; 
100 10 area 3173 iou 0.732; 20 area 2204 iou 0.84; 30 area 1844 iou 0.73; 40 area 1440 iou 0.572; 50 area 372 iou 0.148; 
160 10 area 2729 iou 0.717; 20 area 1780 iou 0.797; 30 area 1400 iou 0.653; 40 area 624 iou 0.291; 50 area 28 iou 0.013; 
300 10 area 2313 iou 0.705; 20 area 1491 iou 0.738; 30 area 1152 iou 0.588; 40 area 648 iou 0.331; 50 area 20 iou 0.01; 
500 10 area 2322 iou 0.672; 20 area 1420 iou 0.706; 30 area 1084 iou 0.562; 40 area 412 iou 0.215; 50 area 4 iou 0.002; 
650 10 area 2821 iou 0.725; 20 area 1892 iou 0.818; 30 area 1620 iou 0. …[+301 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/color_validation.py <<'PY'
import sys,numpy as np,torch,cv2,h5py
sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');C=torch.tensor(np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'],dtype=torch.float32,device='cuda')
cap=cv2.VideoCapture('/work/inputs/video.mp4');res=[]
for i in range(748):
 ok,im=cap.read(); crop=im[745:925,650:860].astype(np.float32); green=crop[:,:,1]-np.maximum(crop[:,:,0],crop[:,:,2]); obj=green>20
 # require vicinity of supplied object mask to avoid any green background/object elsewhere
 with h5py.File('/work/inputs/masks.h5') as f: sam=f['object'][i,745:925,650:860]
 obj=obj & cv2.dilate(sam.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
 row=[]
 for T in [B[i],C[i]]:
  pr=hm.render(T,ss=2)>.5;inter=(pr&obj).sum();row.append(float(inter/(pr|obj).sum()))
 res.append(row)
res=np.array(res);np.save('/work/scratch/iou_color.npy',res)
for a,b in [(0,129),(129,160),(160,510),(510,615),(615,748),(0,748)]:print(a,b,'color baseline/result',res[a:b,0].mean(),res[a:b,1].mean(),'delta',(res[a:b,1]-res[a:b,0]).mean())
PY
python /work/scratch/color_validation.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/color_validation.py", line 12, in <module>
    pr=hm.render(T,ss=2)>.5;inter=(pr&obj).sum();row.append(float(inter/(pr|obj).sum()))
                                   ~~^~~~
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/obj=obj & cv2.dilate/obj_t=torch.tensor(obj \\& cv2.dilate/; s/astype(bool))/astype(bool),device=\"cuda\")/; s/(pr&obj).sum();row.append(float(inter\\/(pr|obj).sum()))/(pr\\&obj_t).sum();row.append(float(inter\\/(pr|obj_t).sum()))/' /work/scratch/color_validation.py
# inspect syntax
grep -n 'obj' /work/scratch/color_validation.py"
```
<details><summary>output</summary>

```
3:B=torch.tensor(np.load('/work/inputs/reconstruction.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');C=torch.tensor(np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'],dtype=torch.float32,device='cuda')
6: ok,im=cap.read(); crop=im[745:925,650:860].astype(np.float32); green=crop[:,:,1]-np.maximum(crop[:,:,0],crop[:,:,2]); obj=green>20
7: # require vicinity of supplied object mask to avoid any green background/object elsewhere
8: with h5py.File('/work/inputs/masks.h5') as f: sam=f['object'][i,745:925,650:860]
9: obj_t=torch.tensor(obj & cv2.dilate(sam.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
12:  pr=hm.render(T,ss=2)>.5;inter=(pr&obj_t).sum();row.append(float(inter/(pr|obj_t).sum()))

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/color_validation.py';s=open(p).read();s=s.replace('obj_t=torch.tensor(obj & cv2.dilate(sam.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)','obj_t=torch.tensor(obj & cv2.dilate(sam.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool),device=\"cuda\")');open(p,'w').write(s)
PY
python /work/scratch/color_validation.py"
```
<details><summary>output</summary>

```
0 129 color baseline/result 0.7642816124960433 0.8621199089427327 delta 0.09783829644668934
129 160 color baseline/result 0.727258566887148 0.7992478955176568 delta 0.07198932863050891
160 510 color baseline/result 0.5908069398573467 0.7159713445390974 delta 0.12516440468175072
510 615 color baseline/result 0.6536596786408198 0.7885659694671631 delta 0.13490629082634337
615 748 color baseline/result 0.7287112277253229 0.861936666911706 delta 0.13322543918638302
0 748 color baseline/result 0.659722770160851 0.780771560289643 delta 0.12104879012879204

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
for f in ['/work/inputs/reconstruction.npz','/work/outputs/object_poses_world.npz']:
 z=np.load(f);T=z['T_world_mesh'];R=T[:,:3,:3];ang=np.rad2deg(np.arccos(np.clip((np.trace(np.einsum('nij,nkj->nik',R[:-1],R[1:]),axis1=1,axis2=2)-1)/2,-1,1)))
 print(f,'sum',ang.sum(),'dynamic sum',ang[129:614].sum(),'max',ang.max(),'mean',ang[129:614].mean())
PY"
```
<details><summary>output</summary>

```
/work/inputs/reconstruction.npz sum 1307.1945 dynamic sum 940.1477 max 10.944652 mean 1.9384489
/work/outputs/object_poses_world.npz sum 412.298168012583 dynamic sum 409.3821884400337 max 3.3386997343628715 mean 0.8440869864742963

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];R0=T[0,:3,:3]
for i in range(0,748,10):
 d=T[i,:3,:3]@R0.T;a=np.rad2deg(np.arccos(np.clip((np.trace(d)-1)/2,-1,1)));print(i,round(a,1),np.round(T[i,:3,3],3))
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
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];R0=T[0,:3,:3]
for i in range(0,748,25):
 d=T[i,:3,:3]@R0.T;a=np.rad2deg(np.arccos(np.clip((np.trace(d)-1)/2,-1,1)));print(i,round(a,1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Watering-can 6-DoF trajectory — method and validation

## Result

"'`object_poses_world.npz` contains key `T_world_mesh` as a `(748, 4, 4)`
`float64` array. The same deterministic script reproduces it offline with:

```bash
python /work/outputs/solve.py --inputs /work/inputs '"\\
  --out /work/outputs/object_poses_world.npz
"'```

The final end-to-end run took **630.4 s** on the available RTX 6000 Ada GPU.
Random seeds are fixed, and no network, challenge annotation, or ground-truth
data was used.

## Video interpretation and phases

| Phase | Frames | What happens | Confidence |
|---|---:|---|---|
| Rest before interaction | 0–128 | Can rests on the floor; the right hand approaches during the last ~20 frames but the can does not visibly move. | High |
| Grasp transition | 129–159 | Right hand closes on the handle and begins lifting/tilting. | Medium-high |
| Tip / pour | 160–509 | Can is tipped forward by about 75–80° relative to its initial pose and held there. | Medium |
| Return / release | 510–614 | Can returns toward upright and the hand releases. | Medium-high |
| Rest after interaction | 615–747 | Can is stationary again. Relative to frame 0 it has settled with about a 15.4° rotation and 4.5 cm translation. | High |

The right-hand keypoints are `21:41`; in this episode the second supplied
contact-probability column follows those keypoints. Both rest intervals were
visually checked to be truly static, not just u …[+7813 chars]
```
<details><summary>output</summary>

```
total 132K
-rw-rw-r-- 1 hanzhou hanzhou 7.2K Oct 20 23:15 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 2.0K Oct 20 23:15 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 23:00 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  94K Oct 20 23:10 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  18K Oct 20 23:00 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /work/outputs/solve.py /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
8d6a04e155a39075fd5a428e94b28896b90031e62131d2bab07ab4e2ac28dddd  /work/outputs/solve.py
aec523ac087ab284481fa65a8f460d7272439459b971e712b0195b57b9ebe81c  /work/outputs/object_poses_world.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/94c246e6b08ca5038ddc3ce38145ecec8525abc28d0cbc09a73c8ee340986b9e/8d6a04e155a39075fd5a428e94b28896b90031e62131d2bab07ab4e2ac28dddd/' /work/outputs/TOOLS.md
python - <<'PY'
import shutil,os
p='/work/outputs/__pycache__'
if os.path.isdir(p): shutil.rmtree(p)
PY
ls -lh /work/outputs"
```
<details><summary>output</summary>

```
total 128K
-rw-rw-r-- 1 hanzhou hanzhou 7.2K Oct 20 23:15 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 2.0K Oct 20 23:15 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou  94K Oct 20 23:10 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  18K Oct 20 23:00 solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'hands\\|V=' /work/scratch/final_metrics_out.py | head -20 && python - <<'PY'
import numpy as np,torch,trimesh
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');O=torch.tensor(np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(m.vertices[::20],dtype=torch.float32,device='cuda')
for a,b in [(129,146),(146,160),(160,510),(510,615)]:
 dd=[]
 for i in range(a,b):
  W=V@O[i,:3,:3].T+O[i,:3,3];dd.append(float(torch.cdist(kp[i],W).min()))
 print(a,b,np.mean(dd),np.median(dd),np.max(dd))
PY"
```
<details><summary>output</summary>

```
129 146 0.05384830060168443 0.05617569014430046 0.10095030069351196
146 160 0.006902241513931325 0.005393061554059386 0.03164424002170563
160 510 0.004510178175821368 0.002583741443231702 0.048504788428545
510 615 0.012346140342941951 0.009160968475043774 0.053121767938137054

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '30,40p' /work/scratch/final_metrics_out.py"
```
<details><summary>output</summary>

```
bf=floors(B);cf=floors(C)
# hand distance (right hand keypoints) on sample verts
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(mesh.vertices[::20],dtype=torch.float32,device='cuda')
def hands(T):
 Ts=torch.tensor(T,dtype=torch.float32,device='cuda');out=[]
 for i in range(748):
  W=V@Ts[i,:3,:3].T+Ts[i,:3,3];out.append(float(torch.cdist(kp[i],W).min()))
 return np.array(out)
bh=hands(B);ch=hands(C)
phases=[('rest_before',0,129),('grasp_transition',129,160),('tip_pour',160,510),('return_release',510,615),('rest_after',615,748)]
out={}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');O=torch.tensor(np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'],dtype=torch.float32,device='cuda');m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(m.vertices[::20],dtype=torch.float32,device='cuda')
d=[]
for i in range(129,160):
 W=V@O[i,:3,:3].T+O[i,:3,3];d.append(float(torch.cdist(kp[i],W).min()))
print(d);print(np.median(d),np.mean(d))
PY"
```
<details><summary>output</summary>

```
[0.0810546875, 0.08588755130767822, 0.06613282114267349, 0.10073751956224442, 0.10095030069351196, 0.09655630588531494, 0.04869121313095093, 0.046105749905109406, 0.0814654529094696, 0.0615156851708889, 0.05617569014430046, 0.0455019436776638, 0.024705294519662857, 0.01143037062138319, 0.0027621358167380095, 0.0013810679083690047, 0.004367320332676172, 0.006550980266183615, 0.006694975309073925, 0.005348853301256895, 0.002183660166338086, 0.001953125, 0.0070420922711491585, 0.011048543266952038, 0.00811193697154522, 0.002183660166338086, 0.0054372698068618774, 0.03164424002170563, 0.003088161814957857, 0.004367320332676172, 0.0009765625]
0.011048543266952038 0.032646854562054

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,torch,trimesh
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(m.vertices[::20],dtype=torch.float32,device='cuda')
for name,p in [('base','/work/inputs/reconstruction.npz'),('res','/work/outputs/object_poses_world.npz')]:
 T=torch.tensor(np.load(p)['T_world_mesh'],dtype=torch.float32,device='cuda');d=[]
 for i in range(748):
  W=V@T[i,:3,:3].T+T[i,:3,3];d.append(float(torch.cdist(kp[i],W).min()))
 d=np.array(d)
 print(name,[(a,b,round(d[a:b].mean()*100,2),round(np.median(d[a:b])*100,2),round(float((d[a:b]<.035).mean()),3)) for a,b in [(129,160),(160,510),(510,615)]])
PY"
```
<details><summary>output</summary>

```
base [(129, 160, 0.94, 1.18, 1.0), (160, 510, 1.03, 0.51, 0.977), (510, 615, 2.27, 2.22, 0.752)]
res [(129, 160, 3.26, 1.1, 0.613), (160, 510, 0.45, 0.26, 0.991), (510, 615, 1.23, 0.92, 0.962)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/for hw in \\[300,1000\\]:/for hw in [3000,10000,30000]:/' /work/scratch/static_hand_test.py && python /work/scratch/static_hand_test.py"
```
<details><summary>output</summary>

```
hw 3000 iou 0.5226035000482342 cos -0.019482049380990074 floor 0.07567238807678223 hand110-128 mean/max 0.0283428712895042 0.056901004165410995 center [0.02774905 1.0698278  2.861067  ] delta [-0.00407211 -0.02444637  0.04874134]
hw 10000 iou 0.484288171451344 cos 0.0026894408791297286 floor 0.08190083503723145 hand110-128 mean/max 0.027697816821991614 0.056691113859415054 center [0.02637022 1.06383    2.8611898 ] delta [-0.00545093 -0.03044415  0.04886413]
hw 30000 iou 0.4682451197639822 cos -0.013108159681743269 floor 0.08478808403015137 hand110-128 mean/max 0.027493420069252972 0.056590091437101364 center [0.02581622 1.0610025  2.861248  ] delta [-0.00600494 -0.03327167  0.0489223 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys,torch,h5py
sys.path.insert(0,'/work/scratch');from build_candidate import lerp_pose,smoothstep
T0=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];Ti=T0[0].copy();Tend=T0[146].copy()
for start in [105,110,115,120,125,129]:
 T=T0.copy()
 for i in range(start,146):
  w=smoothstep((i-(start-1))/(146-(start-1)));T[i]=lerp_pose(Ti,Tend,w)
 for i in range(start):T[i]=Ti
 np.save(f'/work/scratch/earlystart_{start}.npy',T)
PY
python - <<'PY'
import sys,numpy as np,torch,h5py,trimesh;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
z=np.load('/work/inputs/reconstruction.npz');kp=torch.tensor(z['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda');mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(mesh.vertices[::20],dtype=torch.float32,device='cuda')
def ev(T):
 Ts=torch.tensor(T,dtype=torch.float32,device='cuda');ious=[];hands=[]
 for i in range(110,160):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);pr=hm.render(Ts[i],ss=2)>.5;ious.append(float((pr&tar).sum()/(pr|tar).sum()));W=V@Ts[i,:3,:3].T+Ts[i,:3,3];hands.append(float(torch.cdist(kp[i],W).min()))
 return np.mean(ious[:19]),np. …[+165 chars]
```
<details><summary>output</summary>

```
saved
105 (0.7800216486579493, 0.7718311679363251, 0.035116561308601184, 0.017321730204857885, 0.00662297778762877)
110 (0.7917757630348206, 0.7780640053749085, 0.048867743413307164, 0.0223527937871404, 0.005994125036522746)
115 (0.801666353878222, 0.7841341984272003, 0.06243816749682944, 0.02741376855177805, 0.007577014621347189)
120 (0.8081395406472055, 0.7901377403736114, 0.07363535120691124, 0.03283037384739146, 0.010182178113609552)
125 (0.8113937189704493, 0.7955772542953491, 0.08063523863491259, 0.038711576550267635, 0.029859581030905247)
129 (0.8122102963296991, 0.7987806129455567, 0.0820639786359511, 0.043471876478288325, 0.03580589219927788)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
P={dz:torch.tensor(np.load(f'/work/scratch/fd_initial_{dz}.npy'),dtype=torch.float32,device='cuda') for dz in [-.1,-.06,-.03,0]}
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
# precompute color masks
cap=cv2.VideoCapture('/work/inputs/video.mp4');colors=[]
for i in range(129):
 ok,im=cap.read();crop=im[745:925,650:860].astype(np.float32);g=crop[:,:,1]-np.maximum(crop[:,:,0],crop[:,:,2]);colors.append(torch.tensor((g>20)&cv2.dilate(objs[i].astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool),device='cuda'))
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=True).to_geometry();V=torch.tensor(m.vertices[::20],dtype=torch.float32,device='cuda');kp=torch.tensor(np.load('/work/inputs/reconstruction.npz')['human_keypoints_world'][:,21:41],dtype=torch.float32,device='cuda')
for dz,T in P.items():
 sai=[];coi=[];hd=[]
 for i in range(129):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);pr=hm.render(T,ss=2)>.5;sai.append(float((pr&tar).sum()/(pr|tar).sum()))
  col=colors[i];coi.append(float((pr&col).sum()/(pr|col).sum()))
  if i>=110:
   W=V@T[:3,:3].T+T[:3,3];hd.append(float(torch.cdist(kp[i],W).min()))
 print(dz,'S …[+87 chars]
```
<details><summary>output</summary>

```
-0.1 SAM 0.8233533931332965 color 0.8497287912886272 hand110 0.0337766989281303 median 0.0331600122153759
-0.06 SAM 0.8100816870844642 color 0.8380000374113866 hand110 0.01016050534886553 median 0.005074367392808199
-0.03 SAM 0.7963732564172079 color 0.8492971612501514 hand110 0.009499382484998358 median 0.004256737418472767
0 SAM 0.7834374909253083 color 0.8426240174345268 hand110 0.023454397941302312 median 0.016341015696525574

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np,torch,cv2,h5py;sys.path.insert(0,'/work/scratch');import hard_mask_orig as hm
P={dz:torch.tensor(np.load(f'/work/scratch/fd_final_{dz}.npy'),dtype=torch.float32,device='cuda') for dz in [-.1,-.06,-.03,0]}
with h5py.File('/work/inputs/masks.h5') as f:objs=f['object'][:,745:925,650:860];pers=f['person'][:,745:925,650:860]
cap=cv2.VideoCapture('/work/inputs/video.mp4');colors={}
for i in range(748):
 ok,im=cap.read()
 if i>=615:
  crop=im[745:925,650:860].astype(np.float32);g=crop[:,:,1]-np.maximum(crop[:,:,0],crop[:,:,2]);colors[i]=torch.tensor((g>20)&cv2.dilate(objs[i].astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool),device='cuda')
for dz,T in P.items():
 sai=[];coi=[]
 for i in range(615,748):
  t=torch.tensor(objs[i],device='cuda').float();p=torch.tensor(pers[i],device='cuda').float();p=torch.nn.functional.max_pool2d(p[None,None],5,1,2)[0,0];tar=((t*(1-p).clamp(0,1))>.5);pr=hm.render(T,ss=2)>.5;sai.append(float((pr&tar).sum()/(pr|tar).sum()));col=colors[i];coi.append(float((pr&col).sum()/(pr|col).sum()))
 print(dz,'SAM',np.mean(sai),'color',np.mean(coi))
PY"
```
<details><summary>output</summary>

```
-0.1 SAM 0.8114694361399887 color 0.8461746456927823
-0.06 SAM 0.8026811547745439 color 0.8517821081598899
-0.03 SAM 0.7893343209324026 color 0.841060837856809
0 SAM 0.7747745352580135 color 0.8314782116646157

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read()
s=s.replace('| Phase | Rotation step (deg/frame) baseline → result | Translation step (cm/frame) baseline → result | Median floor minimum (cm) baseline → result | Median right-hand distance (cm) baseline → result |','| Phase | Rotation step (deg/frame) baseline → result | Translation step (cm/frame) baseline → result | Median floor minimum (cm) baseline → result | Mean right-hand distance (cm; <3.5 cm coverage) baseline → result |')
s=s.replace('| Grasp transition | 2.37 → **1.63** | 2.07 → **0.94** | −1.69 → 1.43 | 1.18 → **1.10** |','| Grasp transition | 2.37 → **1.63** | 2.07 → **0.94** | −1.69 → 1.43 | 0.94 (100%) → 3.26 (61%) |')
s=s.replace('| Tip / pour | 1.77 → **0.64** | 2.13 → **0.47** | 5.63 → 4.84 | 0.51 → **0.26** |','| Tip / pour | 1.77 → **0.64** | 2.13 → **0.47** | 5.63 → 4.84 | 1.03 (98%) → **0.45 (99%)** |')
s=s.replace('| Return / release | 2.37 → **1.28** | 1.74 → **0.58** | 3.96 → **3.85** | 2.22 → **0.92** |','| Return / release | 2.37 → **1.28** | 1.74 → **0.58** | 3.96 → **3.85** | 2.27 (75%) → **1.23 (96%)** |')
s=s.replace('Median right-hand surface distance\\nfalls from 0.51 cm to 0.26 cm during pouring and from 2.22 cm to 0.92 cm during\\nreturn/release. Static phase poses are exactly constant.','Mean right-hand surface distance falls from 1.03 cm to 0.45 cm during\\npouring and from 2.27 cm to 1.23 cm during return/release. The earliest part\\nof the grasp transition (r …[+989 chars]
```
<details><summary>output</summary>

```
22:| Grasp transition | 129–159 | Right hand closes on the handle and begins lifting/tilting. | Medium-high |
74:| Grasp transition | 0.672 → **0.794** | 0.727 → **0.799** | 4.87 → **2.61** |
90:| Phase | Rotation step (deg/frame) baseline → result | Translation step (cm/frame) baseline → result | Median floor minimum (cm) baseline → result | Mean right-hand distance (cm; <3.5 cm coverage) baseline → result |
93:| Grasp transition | 2.37 → **1.63** | 2.07 → **0.94** | −1.69 → 1.43 | 0.94 (100%) → 3.26 (61%) |
100:from 5.37 cm to 1.09 cm in translation. Mean right-hand surface distance falls from 1.03 cm to 0.45 cm during
101:pouring and from 2.27 cm to 1.23 cm during return/release. The earliest part
123:silhouette-driven fit can over-correct distance. Here depth is supported by both the S …[+31 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/outputs/TOOLS.md'
```
<details><summary>output</summary>

```
# Tools and dependencies

No external repositories were cloned and no model weights were downloaded for
the final solution. The final script depends only on the supplied inputs and
the installed software listed below.

## Used by `solve.py`

| Tool/package | Version | Role |
|---|---:|---|
| PyTorch | 2.5.1+cu124 | SE(3) optimization and differentiable rasterization graphs |
| nvdiffrast | 0.4.0 | Mesh rasterization, UV interpolation, and antialiased silhouette |
| NumPy | 1.26.3 | Array/pose math |
| SciPy | 1.16.3 | Rotation averaging/interpolation and Savitzky–Golay smoothing |
| OpenCV (`opencv-python`) | 4.11.0.86 | Video decoding and ROI handling |
| h5py | 3.16.0 | Reading supplied object/person masks |
| trimesh | 5.1.1 | GLB mesh, UV texture, and convex-hull geometry |
| Pillow |  …[+1198 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,180p' /work/outputs/REPORT.md"
```
<details><summary>output</summary>

```
# Watering-can 6-DoF trajectory — method and validation

## Result

`object_poses_world.npz` contains key `T_world_mesh` as a `(748, 4, 4)`
`float64` array. The same deterministic script reproduces it offline with:

```bash
python /work/outputs/solve.py --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The final end-to-end run took **630.4 s** on the available RTX 6000 Ada GPU.
Random seeds are fixed, and no network, challenge annotation, or ground-truth
data was used.

## Video interpretation and phases

| Phase | Frames | What happens | Confidence |
|---|---:|---|---|
| Rest before interaction | 0–128 | Can rests on the floor; the right hand approaches during the last ~20 frames but the can does not visibly move. | High |
| Grasp transition | 129–159 | Right hand  …[+6879 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,os
p='/work/outputs/object_poses_world.npz';z=np.load(p);assert z.files==['T_world_mesh'];T=z['T_world_mesh'];assert T.shape==(748,4,4) and T.dtype==np.float64 and np.isfinite(T).all();print('output OK',T.shape,T.dtype,T.nbytes)
for f in ['/work/outputs/solve.py','/work/outputs/REPORT.md','/work/outputs/TOOLS.md']:print(f,os.path.getsize(f))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read();s=s.replace('Both independent appearance measures improve in every phase and on every\\nframe. The improvement is largest during the heavily occluded tip/pour phase.','Both independent appearance measures improve in every phase. The improvement\\nis largest during the heavily occluded tip/pour phase.');s=s.replace('The result also reduces p90 dynamic jitter from 3.36° to 1.88° in rotation and\\nfrom 5.37 cm to 1.09 cm in translation. Mean right-hand surface distance falls from 1.03 cm to 0.45 cm during\\npouring','The result also reduces p90 dynamic jitter from 3.36° to 1.88° in rotation and\\nfrom 5.37 cm to 1.09 cm in translation. Mean right-hand surface distance falls\\nfrom 1.03 cm to 0.45 cm during pouring');s=s.replace('Here depth is supported by both the SAM and independent green-color\\nsilhouettes and by the floor-plane soft constraint; hand proximity gives','Here depth is supported by both the SAM and independent green-color\\nsilhouettes and by the floor-plane soft constraint; hand proximity gives');s=s.replace('reconstructed hand is less consistent with the silhouette during the initial\\nreach, and MoGe object depth is noisy at this object size, so neither was used\\nas a hard constraint. Confidence is consequently **medium** for absolute depth and\\n**high** for temporal consistency and relative motion.','reconstructed hand is less consistent with the silhouette during the initial\\nr …[+246 chars]
```
<details><summary>output</summary>

```

```
</details>

**Agent:** Completed the task. Required outputs are in `/work/outputs`:

- `object_poses_world.npz` — `T_world_mesh`, shape `(748, 4, 4)`, `float64`
- `solve.py` — deterministic offline reproduction script
- `REPORT.md` — phases, method, validation, confidence, and runtime
- `TOOLS.md` — dependencies and tools used

Validation highlights:

- Mean SAM mask IoU: **0.625 → 0.757**
- Independent green-color mask IoU: **0.660 → 0.781**
- Dynamic p90 rotation jitter: **3.36° → 1.88°**
- Dynamic p90 translation jitter: **5.37 cm → 1.09 cm**
- Static phases now use exactly constant world poses
- End-to-end reproduction runtime: **630.4 s**
