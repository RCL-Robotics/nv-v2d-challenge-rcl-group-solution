# T1_ds_ep002

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T15:11:29Z

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
and translation both matter; a  …[+9792 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from pathlib import Path
s=Path('/work/AGENT_TASK.md').read_text()
print(len(s))
for i in range(0,len(s),4000):
 print(f'---CHUNK {i}---')
 print(s[i:i+4000])
PY"
```
<details><summary>output</summary>

```
10592
---CHUNK 0---
# Task: recover the full 6-DoF object trajectory of one monocular human-object-interaction video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose; no symmetry handling, no alignment), averaged over thresholds 1-9 cm. Rotation
and transla …[+9849 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '35,65p' | fold -w 160"
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
    42	4. **Global decoding over time**: choose one hyp …[+2201 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%f %s bytes\\n' && echo '--- task' && cat /work/inputs/task.txt && echo '--- episode' && cat /work/inputs/episode.json && echo '--- camera' && cat /work/inputs/camera.json && echo '--- ground' && cat /work/inputs/ground_plane.json"
```
<details><summary>output</summary>

```
video.mp4 30288483 bytes
object_mesh.glb 9153188 bytes
ground_plane.json 398 bytes
task.txt 291 bytes
episode.json 72 bytes
camera.json 839 bytes
reconstruction.npz 49439935 bytes
frame0_reference_human.npz 194239 bytes
masks.h5 7970917 bytes
--- task
Walk over to the blue basket on the round table and pick up the blue basket using your right hand and place the basket in a different area within the bounding area. Then pick up the basket and handover to the left hand and place the blue basket in a different area within the bounding area.
--- episode
{
 "episode_index": 2,
 "object": "blue_easter_basket",
 "frames": 980
}--- camera
{
 "K": [
  [
   820.0018310546875,
   0.0,
   758.1000366210938
  ],
  [
   0.0,
   820.0018310546875,
   560.4999389648438
  ],
  [
   0.0,
   0.0,
   1.0
  ]
  …[+1086 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | head -100; echo '--- code'; find /code/reconstruction -maxdepth 2 -type f | head -80"
```
<details><summary>output</summary>

```
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/01_depth.json 1744
/pipeline/.stages/02_export.json 1986
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2132
/pipeline/.stages/08_render.json 2525
/pipeline/depth/moge2/depth_report.json 1809
/pipeline/depth/moge2/aligned.h5 1186607560
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/intrinsics.pkl 104287
/pipeline/depth/moge2/raw.npy 3468165248
/pipeline/export/episode_000002/edex 557
/pipeline/export/episode_000002/wild_export.json 2804
/pipeline/foundationpose/episode_000002.pkl 278503
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 74944 …[+2102 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, trimesh
r=np.load('/work/inputs/reconstruction.npz')
print('recon keys/shapes:')
for k in r.files: print(k, r[k].shape, r[k].dtype)
print('contact stats', np.percentile(r['contact_prob_right_left'],[0,5,25,50,75,95,100],axis=0))
with h5py.File('/work/inputs/masks.h5') as f:
 print('masks', {k:(f[k].shape, f[k].dtype) for k in f})
print('frame0', np.load('/work/inputs/frame0_reference_human.npz').files)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print('mesh',type(m),m.vertices.shape,m.faces.shape,'bounds',m.bounds,'extents',m.extents)
PY"
```
<details><summary>output</summary>

```
recon keys/shapes:
T_world_mesh (980, 4, 4) float32
T_cam_mesh_foundationpose (980, 4, 4) float32
T_cam_mesh_coconet (980, 4, 4) float32
human_vertices_world (980, 4610, 3) float32
human_keypoints_world (980, 70, 3) float32
contact_prob_right_left (980, 2) float32
contact stats [[6.57975723e-17 1.25305260e-14]
 [1.79715427e-13 1.20996977e-12]
 [5.39738718e-12 7.48804126e-11]
 [1.32015060e-07 1.52334576e-06]
 [9.99522313e-01 1.11617334e-02]
 [9.99999404e-01 1.00000000e+00]
 [1.00000000e+00 1.00000000e+00]]
masks {'object': ((980, 1152, 1536), dtype('bool')), 'person': ((980, 1152, 1536), dtype('bool'))}
frame0 ['vertices_world']
mesh <class 'trimesh.base.Trimesh'> (263964, 3) (88020, 3) bounds [[-0.11327863 -0.06662077 -0.12387678]
 [ 0.12049808  0.08906516  0.1194884 ]] extents [0.23377671 …[+24 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader; python - <<'PY'
import torch,cv2,trimesh,open3d
print('torch',torch.__version__,torch.cuda.is_available())
print('cv2',cv2.__version__,'trimesh',trimesh.__version__,'open3d',open3d.__version__)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 True
cv2 4.11.0 trimesh 5.1.1 open3d 0.20.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch /work/outputs && python - <<'PY'
import cv2,json,numpy as np,h5py
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh']; c=r['contact_prob_right_left']; hk=r['human_keypoints_world']
F=len(T)
# frame video props
cap=cv2.VideoCapture('/work/inputs/video.mp4')
print('video',cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_FPS),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# stats arrays
motion=np.linalg.norm(T[:,:3,3]-np.r_[T[1:,:3,3],T[-1:,:3,3]],axis=1)
relR=[]
for i in range(F-1):
 rr=T[i+1,:3,:3]@T[i,:3,:3].T
 relR.append(R.from_matrix(rr).magnitude())
relR=np.r_[relR,0]
# pose center and z
Tc=np.linalg.inv(np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']))
pc=T[:,:3,3]
pcc=pc@Tc[:3,:3].T+Tc[:3,3]
print('trans xyz percentiles')
for j in range(3): print(j,np.percentile(pc[:,j],[0,1,5,25,50,75,95,99,100]))
print('cam z',np.percentile(pcc[:,2],[0,1,5,25,50,75,95,99,100]))
print('rot step deg',np.percentile(np.degrees(relR),[0,50,75,90,95,99,100]), 'max frames',np.argsort(relR)[-20:])
print('trans step',np.percentile(motion,[0,50,75,90,95,99,100]), 'max frames',np.argsort(motion)[-20:])
# contacts phase runs threshold
for thr in [.5,.9,.99]:
 mask=np.any(c>thr,axis=1)
 runs=[]; st=None
 for i,v in enumerate(mask):
  if v and st is None: st=i
  if st is not None and (not v or i==F-1):
   en=i if v else i-1; runs.append((st,en)); s …[+302 chars]
```
<details><summary>output</summary>

```
video 980.0 30.0 1536.0 1152.0
trans xyz percentiles
0 [-0.06643914 -0.0542079  -0.02158693  0.13673757  0.4075049   0.72445652
  1.12949011  1.42312915  1.43914735]
1 [0.13541691 0.14708837 0.20180091 0.3069157  0.46920232 1.07440907
 1.09362471 1.14436108 1.18767917]
2 [2.49719906 2.51461665 2.57067707 3.12803167 3.37750423 3.93994665
 4.08610697 5.00621205 5.07185411]
cam z [2.46353019 2.47915855 2.53976885 3.12307124 3.33433074 3.89797165
 4.04636557 5.02799644 5.08992454]
rot step deg [  0.           1.4704317    7.14368377  17.32820553  26.40934214
  55.7295028  158.99178732] max frames [647 871 825 835 876 216 861 798 732 914 273 963 962 923 812 766 826 123
 842 807]
trans step [0.         0.01409609 0.02884402 0.05253638 0.0714509  0.14860982
 0.92019099] max frames [822 434 809 21 …[+1904 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/contact.py <<'PY'
import cv2,json,h5py,numpy as np,trimesh,os,sys
root='/work/inputs'
r=np.load(root+'/reconstruction.npz'); T=r['T_world_mesh']
cam=json.load(open(root+'/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
mesh=trimesh.load(root+'/object_mesh.glb',force='scene',process=False).to_geometry(); V=np.c_[mesh.vertices,np.ones(len(mesh.vertices))]
F=len(T)
h5=h5py.File(root+'/masks.h5','r')
frames=eval(sys.argv[1]); out=sys.argv[2]
cap=cv2.VideoCapture(root+'/video.mp4')
thumb=[]
for fi in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ok,im=cap.read(); assert ok
 om=np.array(h5['object'][fi]); pm=np.array(h5['person'][fi])
 # overlay object blue and person green transparently
 ov=im.copy(); ov[om]=(0.4*ov[om]+0.6*np.array([255,80,0])).astype(np.uint8); ov[pm]=(0.55*ov[pm]+0.45*np.array([0,255,0])).astype(np.uint8)
 # projected all mesh outline/bbox
 pc=(Tcw@T[fi]@V.T).T[:,:3]; uv=(K@pc.T).T; uv=uv[:,:2]/uv[:,2:3]
 valid=(pc[:,2]>0)&np.isfinite(uv).all(1)
 if valid.any():
  x0,y0=np.floor(uv[valid].min(0)).astype(int); x1,y1=np.ceil(uv[valid].max(0)).astype(int)
  cv2.rectangle(ov,(x0,y0),(x1,y1),(255,255,0),3)
 # crop based on union / resize? full
 ov=cv2.resize(ov,(768,576))
 cv2.putText(ov,f'{fi}',(10,35),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),3)
 cv2.putText(ov,f'{fi}',(10,35),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),1)
 thumb.append(ov)
cols=4; rows=(len(thumb)+cols-1)//cols
canvas= …[+392 chars]
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 1.7M Oct 20 19:06 /work/scratch/contact_0_480.jpg
-rw-rw-r-- 1 hanzhou hanzhou 1.9M Oct 20 19:06 /work/scratch/contact_480_980.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for f in /pipeline/.stages/*.json /pipeline/pipeline_report.json /pipeline/depth/moge2/depth_report.json /pipeline/export/episode_000002/wild_export.json; do echo ==== $f; cat $f; done'
```
<details><summary>output</summary>

```
{
  "aligned_object_extents_m": [
    0.1550884246826172,
    0.23449096083641052,
    0.2436237335205078
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "depth_source_commit": "925b8ed835a7a9cdb7578ba15c658a0afc969030",
  "frame_count": 980,
  "height": 1152,
  "intrinsics": [
    [
      813.6884765625,
      0.0,
      768.0
    ],
    [
      0.0,
      813.6884765625,
      576.0
    ],
    [
      0.0,
      0.0,
      1.0
    ]
  ],
  "object_mesh_file": "/home/hanzhou/Projects/v2d/run/out/episode_000002/export/episode_000002/object_mesh/output_aligned.glb",
  "object_mesh_to_training_transform":  …[+2004 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,pprint,numpy as np
for p in ['/pipeline/foundationpose/episode_000002.pkl','/pipeline/mhr_init/episode_000002.pkl']:
 print('\\n====',p)
 d=pickle.load(open(p,'rb'))
 print(type(d))
 if isinstance(d,dict):
  for k,v in d.items(): print(k, type(v), getattr(v,'shape',None), (repr(v)[:300] if not hasattr(v,'shape') else ''))
 else: print(repr(d)[:2000])
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
frames <class 'list'> None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029',
camera_id <class 'int'> None 0
fp_poses_cam <class 'numpy.ndarray'> (980, 1, 4, 4) 
fp_candidate_count <class 'numpy.ndarray'> (980,) 
fp_selected_candidate_index <class 'numpy.ndarray'> (980,) 
fp_selected_iou <class 'numpy.ndarray'> (980,) 
fp_selected_visibility <class 'numpy.ndarray'> (980,) 
fp_retry_count <class 'numpy.ndarray'> (980,) 
fp_reliable <class 'numpy.ndarray'> (980,) 
fp_oracle_rotation_error_deg <class 'numpy.ndarray'> (980,) 
fp_tran …[+2847 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
for p in ['/pipeline/inference/materialized_inputs.h5','/pipeline/work/masks_k0.h5','/pipeline/mhr_init/episode_000002.sam3d.h5']:
 print('\\n====',p)
 with h5py.File(p,'r') as f:
  def show(name,obj):
   if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype, obj.attrs.get('__doc__'))
  f.visititems(show)
PY"
```
<details><summary>output</summary>

```

==== /pipeline/inference/materialized_inputs.h5
K_rois (980, 3, 3) float32 None
bboxes (980, 4) float32 None
identity_json () object None
input_rgbs (980, 3, 224, 224) float32 None
input_xyz (980, 6, 224, 224) float32 None
observed_human_masks (980, 224, 224) bool None
observed_object_masks (980, 224, 224) bool None
render_rgbs (980, 3, 224, 224) float32 None
render_xyz (980, 6, 224, 224) float32 None

==== /pipeline/work/masks_k0.h5
episode_000002/000000-k0.obj_rend_mask.png (1152, 1536) uint8 None
episode_000002/000000-k0.person_mask.png (1152, 1536) uint8 None
episode_000002/000001-k0.obj_rend_mask.png (1152, 1536) uint8 None
episode_000002/000001-k0.person_mask.png (1152, 1536) uint8 None
episode_000002/000002-k0.obj_rend_mask.png (1152, 1536) uint8 None
episode_000002/000002-k0.perso …[+126970 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
for p in ['/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth']:
 d=torch.load(p,map_location='cpu',weights_only=False)
 print('\\n',p,type(d))
 if isinstance(d,dict):
  for k,v in d.items(): print(k,type(v),getattr(v,'shape',None),repr(v)[:200] if not hasattr(v,'shape') else '')
 else: print(repr(d)[:1000])
PY"
```
<details><summary>output</summary>

```

 /pipeline/inference/coconet.pth <class 'dict'>
schema <class 'str'> None 'cari4d.mhr_wild_inference.v1'
frames <class 'list'> None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019',
frame_meta <class 'list'> None [{'frame': '000000', 'src_frame': 0, 'kid': 0}, {'frame': '000001', 'src_frame': 1, 'kid': 0}, {'frame': '000002', 'src_frame': 2, 'kid': 0}, {'frame': '000003', 'src_frame': 3, 'kid': 0}, {'frame': '
kid <class 'int'> None 0
checkpoint <class 'dict'> None {'path': '/mnt/nas/Group/hanzhou/v2d/weights/cari4d/2026-08-25-09-35-57/step200000.pth', 'step': 200000}
mhr_inference <class 'dict'> None {'schema': 'cari4d.mhr_inference_proven …[+5157 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/plot_diag.py <<'PY'
import numpy as np,pickle,h5py,cv2,json,matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; c=r['contact_prob_right_left']; F=len(T)
fp=pickle.load(open('/pipeline/foundationpose/episode_000002.pkl','rb'))
obj=[]
with h5py.File('/work/inputs/masks.h5','r') as h5:
 for i in range(F):
  m=h5['object'][i]; ys,xs=np.where(m); obj.append((m.sum(),xs.mean() if len(xs) else np.nan,ys.mean() if len(ys) else np.nan, xs.ptp() if len(xs) else 0,ys.ptp() if len(ys) else 0))
obj=np.array(obj)
stepR=np.zeros(F); stept=np.zeros(F)
for i in range(F-1):
 stepR[i]=np.degrees(R.from_matrix(T[i+1,:3,:3]@T[i,:3,:3].T).magnitude())
 stept[i]=np.linalg.norm(T[i+1,:3,3]-T[i,:3,3])
fig,axs=plt.subplots(5,1,figsize=(20,14),sharex=True)
x=np.arange(F)
axs[0].plot(x,T[:,0,3],label='x');axs[0].plot(x,T[:,1,3],label='y');axs[0].plot(x,T[:,2,3],label='z');axs[0].legend();axs[0].set_ylabel('world pos')
axs[1].plot(x,stepR);axs[1].set_ylabel('rot step deg');axs[1].set_ylim(0,50)
axs[2].plot(x,stept*100);axs[2].set_ylabel('trans step cm');axs[2].set_ylim(0,20)
axs[3].plot(x,c[:,0],label='right');axs[3].plot(x,c[:,1],label='left');axs[3].legend();axs[3].set_ylabel('contact')
axs[4].plot(x,obj[:,0],label='area'); axs[4].set_ylabel('object mask px'); axs4b=axs[4].twinx();axs4b.plot(x,obj[:,1],label='cx',alpha=.5);axs4b.plot(x,obj[:,2],label='cy',alpha=.5);axs4b.legen …[+828 chars]
```
<details><summary>output</summary>

```
fp_selected_iou [0.         0.         0.         0.90847485 0.94314808]
fp_selected_visibility [0.         0.00370711 0.96823114 1.         1.        ]
fp_reliable (array([ True]), array([980]))
fp_observed_visible_pixels (array([   0,   62,   80,  120,  157,  159,  195,  234,  240,  247,  263,
        351,  376,  386,  412,  438,  457,  489,  492,  504,  519,  521,
        560,  587,  614,  619,  629,  637,  639,  642,  666,  670,  697,
        717,  721,  723,  731,  734,  736,  737,  745,  748,  762,  764,
        783,  787,  814,  821,  841,  845,  879,  882,  897,  930,  943,
        949,  954,  956,  978,  995, 1025, 1030, 1031, 1034, 1037, 1044,
       1047, 1058, 1087, 1088, 1102, 1112, 1174, 1180, 1182, 1193, 1215,
       1248, 1274, 1296, 1316, 1319, 1340, 1372, 1387, 1390, 1397 …[+6394 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,numpy as np
fp=pickle.load(open('/pipeline/foundationpose/episode_000002.pkl','rb'))
for k in fp['metadata']: print(k,repr(fp['metadata'][k])[:1000])
print('sources',[(s,int((fp['fp_emitted_pose_source']==s).sum())) for s in np.unique(fp['fp_emitted_pose_source'])])
# compare fp pose with recon
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'];
print('fp pose first',fp['fp_poses_cam'][0,0]);print('recon',T[0])
PY"
```
<details><summary>output</summary>

```
source 'foundationpose_register_first_then_track_metric_depth'
depth_root '/home/hanzhou/Projects/v2d/run/out/episode_000002/depth/moge2/aligned.h5'
depth_source 'metric_depth'
depth_input_identity {'path': '/home/hanzhou/Projects/v2d/run/out/episode_000002/depth/moge2/aligned.h5', 'size': 1186607560, 'mtime_ns': 1791201684890813898}
mesh '/home/hanzhou/Projects/v2d/run/out/episode_000002/export/episode_000002/object_mesh/output_aligned.glb'
mask_identities {'human': {'raw': {'storage': 'h5', 'source': {'path': '/home/hanzhou/Projects/v2d/run/out/episode_000002/export/episode_000002/human_masks/front_stereo_camera_left.h5', 'size': 23678439, 'mtime_ns': 1791200572091818143}}, 'effective': None}, 'object': {'raw': {'storage': 'h5', 'source': {'path': '/home/hanzhou/Projects/v2d/run/out/epis …[+5131 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
try:
 import pyrender; print('pyrender',pyrender.__version__)
except Exception as e: print('no pyrender',e)
try:
 import nvdiffrast; print('nvdiffrast ok',nvdiffrast.__file__)
except Exception as e: print('no nvdiffrast',e)
PY"
```
<details><summary>output</summary>

```
pyrender 0.1.45
nvdiffrast ok /opt/conda/lib/python3.11/site-packages/nvdiffrast/__init__.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,pickle
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'];
for a,b in [(0,94),(260,418),(846,979),(500,700)]:
 Ts=T[a:b+1]
 p=Ts[:,:3,3]; print('\\n',a,b,'pos mean',p.mean(0),'std cm',p.std(0)*100,'range cm',(p.max(0)-p.min(0))*100)
 # rotation dispersion around mean quaternion chordal
 q=R.from_matrix(Ts[:,:3,:3]).as_quat(); q*=np.sign(q[:,0,None]); qm=q.mean(0); qm/=np.linalg.norm(qm)
 ds=np.degrees(2*np.arccos(np.clip(np.abs(q@qm),-1,1)))
 print('rot relative median/max',np.median(ds),np.max(ds))
 print('first pose',Ts[0]); print('last pose',Ts[-1])
PY"
```
<details><summary>output</summary>

```

 0 94 pos mean [0.4756381  0.30670503 3.3474185 ] std cm [0.06602655 0.04069935 0.43188334] range cm [0.31548738 0.19447207 2.063632  ]
rot relative median/max 0.0 0.0
first pose [[ 0.42274916 -0.00584769  0.90622675  0.47602487]
 [-0.05988644 -0.9979724   0.02149707  0.30694348]
 [ 0.9042653  -0.06335848 -0.42224336  3.3499467 ]
 [ 0.          0.          0.          1.        ]]
last pose [[ 0.42274916 -0.00584769  0.90622675  0.47752225]
 [-0.05988644 -0.9979724   0.02149707  0.3078665 ]
 [ 0.9042653  -0.06335848 -0.42224336  3.3597412 ]
 [ 0.          0.          0.          1.        ]]

 260 418 pos mean [0.1364407 1.081743  3.9300146] std cm [0.13398737 0.8001356  2.8308785 ] range cm [ 0.98758936  4.3762803  13.686991  ]
rot relative median/max 1.9141989487859286 80.9616761226722
 …[+1682 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh']; f=r['T_cam_mesh_foundationpose']; c=r['T_cam_mesh_coconet']; cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
for name,X in [('fp',f),('coco',c)]:
 # world equivalent direct using tf world cam (not depth scale)
 W=Twc@X
 rd=np.degrees(R.from_matrix(T[:,:3,:3].transpose(0,2,1)@W[:,:3,:3]).magnitude())
 td=np.linalg.norm(T[:,:3,3]-W[:,:3,3],axis=1)
 print(name,'rot diffs',np.percentile(rd,[0,10,25,50,75,90,95,99,100]),'transdiff',np.percentile(td,[0,10,50,90,100]))
 # cam z scaling ratios
 p=(np.linalg.inv(Twc)@T)[:,:3,3]; q=X[:,:3,3]
 print('z ratio',np.percentile(q[:,2]/p[:,2],[0,10,50,90,100]))
 print('rot step',np.percentile(np.degrees(R.from_matrix(X[1:,:3,:3]@X[:-1,:3,:3].transpose(0,2,1)).magnitude()),[0,50,90,99,100]))
# Which pipeline pose corresponds to reconstruction? mapping likely centered S
import pickle
fp=pickle.load(open('/pipeline/foundationpose/episode_000002.pkl','rb'))
print('fp_poses same T_cam fp?',np.max(np.abs(fp['fp_poses_cam'][:,0]-f)),np.max(np.abs(fp['obj_pose_world']-T)))
for i in [0,100,200,300,400,500,600,700,800,900]:
 print(i,'T',np.round(T[i],3).tolist(),'fpobj',np.round(fp['obj_pose_world'][i],3).tolist())
PY"
```
<details><summary>output</summary>

```
fp rot diffs [  0.36499815   2.03146891   4.15706015  19.92314387  50.01193054
 118.24189887 148.10954594 174.32133877 179.40627913] transdiff [0.03664126 0.08260773 0.15443599 1.40025191 2.92402566]
z ratio [0.83562893 0.96111323 1.01307337 1.49547813 1.95402981]
rot step [1.21212837e-01 6.80763456e+00 3.08503667e+01 1.63473628e+02
 1.77738315e+02]
coco rot diffs [0.95606796 1.00526368 1.15480685 1.16258889 1.23744727 1.2772536
 1.28383965 1.29034161 1.29144682] transdiff [0.05282809 0.06516567 0.10298844 0.16860143 0.22253609]
z ratio [0.95747529 0.96681638 0.98200434 0.99557154 1.00299659]
rot step [9.95662663e-16 1.48287218e+00 1.73376183e+01 5.57579307e+01
 1.58996236e+02]
fp_poses same T_cam fp? 5.0371695 5.568636
0 T [[0.4230000078678131, -0.006000000052154064, 0.906000018119812, 0. …[+5576 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"T_world_mesh\\|T_cam_mesh_coconet\\|object_poses_world\\|reconstruction.npz\" -n /code/reconstruction /workspace 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_mv_postprocess/lib/interaction_trim.py:322:    object_poses_world_from_object: np.ndarray,
/code/reconstruction/modules/v2d_mv_postprocess/lib/interaction_trim.py:330:    poses = np.asarray(object_poses_world_from_object)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"depth correction\\|depth_correction\\|obj_pose_world\\|pose_abs\" -n /code/reconstruction/modules /workspace/v2d_cari4d 2>/dev/null | head -200"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:520:        total = len(bundle["pr"]["pose_abs"])
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:555:        self.pose_init = _as_tensor(self.pr["pose_abs"], device=self.device, dtype=self.dtype)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:556:        self.pose_ref = _as_tensor(self.ref.get("pose_abs", self.pr["pose_abs"]), device=self.device, dtype=self.dtype)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:803:        result["pr"]["pose_abs"] = _to_numpy(pose).astype(np.float32)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:804 …[+6482 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '500,720p' /workspace/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py"
```
<details><summary>output</summary>

```
        if self.profiler is not None:
            self.profiler.record("persistent_input_cache_write", started)
        return path


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _numpy(value: Any) -> np.ndarray:
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def write_wild_inference_bundle(bundle: Mapping[str, Any], output: str | Path) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(bundle, temporary, pickle_protocol=4)
    os.replace( …[+16058 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '600,675p' /workspace/v2d_cari4d/lib/cari4d/tools/run_mhr_wild_inference.py"
```
<details><summary>output</summary>

```
                batch_cpu, records = builder.build(indices)
                if profiler is not None:
                    profiler.record("window_input_materialization", input_started)
                transfer_started = profiler.start() if profiler is not None and device.type != "cuda" else 0.0
                transfer_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if profiler is not None and device.type == "cuda" else None
                if transfer_events is not None:
                    transfer_events[0].record()
                batch = _to_device(batch_cpu, device)
                if transfer_events is not None:
                    transfer_events[1].record()
                elif profiler is not None:
                    profiler.record("host_to_ …[+6501 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print(type(m.visual),m.visual)
print(vars(m.visual).keys())
print('uv',getattr(m.visual,'uv',None).shape if getattr(m.visual,'uv',None) is not None else None)
print('material',m.visual.material)
if hasattr(m.visual.material,'image'): print('image',m.visual.material.image.size,m.visual.material.image.mode)
print('vertex colors',getattr(m.visual,'vertex_colors',None).shape if getattr(m.visual,'vertex_colors',None) is not None else None)
print('face colors',getattr(m.visual,'face_colors',None).shape if getattr(m.visual,'face_colors',None) is not None else None)
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> <trimesh.visual.texture.TextureVisuals object at 0x7fe8b9d0b610>
dict_keys(['vertex_attributes', '_cache', 'material', 'face_materials', 'mesh'])
uv (263964, 2)
material <trimesh.visual.material.PBRMaterial object at 0x7fe8b9d0b850>
vertex colors None
face colors None

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); mat=m.visual.material
for k in ['baseColorTexture','image','baseColorFactor','metallicFactor','roughnessFactor','normalTexture']:
 try:
  v=getattr(mat,k); print(k,type(v),getattr(v,'shape',None),getattr(v,'size',None),repr(v)[:200])
 except Exception as e: print(k,e)
print(mat.to_pbr())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'to_pbr'
baseColorTexture <class 'PIL.Image.Image'> None (3066, 2181) <PIL.Image.Image image mode=RGB size=3066x2181 at 0x7CDC1ED57250>
image 'PBRMaterial' object has no attribute 'image'
baseColorFactor <class 'NoneType'> None None None
metallicFactor <class 'float'> None None 0.0
roughnessFactor <class 'NoneType'> None None None
normalTexture <class 'NoneType'> None None None

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_overlay.py <<'PY'
import os,cv2,json,numpy as np,torch,trimesh,h5py
import nvdiffrast.torch as dr
from PIL import Image
root='/work/inputs'; cam=json.load(open(root+'/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc).astype(np.float32)
mesh=trimesh.load(root+'/object_mesh.glb',force='scene',process=False).to_geometry(); V=mesh.vertices.astype(np.float32); Fa=mesh.faces.astype(np.int32); UV=mesh.visual.uv.astype(np.float32)
tex=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255.; th,tw=tex.shape[:2]
dev='cuda'; ctx=dr.RasterizeCudaContext()
Vt=torch.tensor(V,device=dev); Ft=torch.tensor(Fa,device=dev); UVt=torch.tensor(UV,device=dev); text=torch.tensor(tex,device=dev)
r=np.load(root+'/reconstruction.npz'); T=r['T_world_mesh']
def render(i):
 h,w=1152,1536; fx,fy=K[0,0],K[1,1]; cx,cy=K[0,2],K[1,2]; zn,zf=.01,100
 P=np.array([[2*fx/w,0,1-2*cx/w,0],[0,2*fy/h,1-2*cy/h,0],[0,0,-(zf+zn)/(zf-zn),-2*zf*zn/(zf-zn)],[0,0,1,0]],np.float32)
 M=(P@Tcw@T[i]).astype(np.float32)
 vh=np.c_[V,np.ones(len(V),np.float32)]
 clip=torch.tensor(vh@M.T,device=dev)[None]
 rast,rdb=dr.rasterize(ctx,clip,Ft,resolution=[h,w])
 uv,uvd=dr.interpolate(UVt[None],rast,Ft)
 # flip y? sample bilinear
 pix=torch.stack([uv[...,0]*(tw-1), (1-uv[...,1])*(th-1)],-1)
 pix=torch.stack([pix[...,0],pix[...,1]],-1)
 col=dr.texture(text[None],pix[None],filter_mode='linear')[0]
 mask=(rast[0,.. …[+1740 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_overlay.py", line 29, in <module>
    om=np.array(h5['object'][i]); pm=np.array(h5['person'][i]); rm,rgb=render(i)
                                                                       ^^^^^^^^^
  File "/work/scratch/render_overlay.py", line 21, in render
    col=dr.texture(text[None],pix[None],filter_mode='linear')[0]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 439, in texture
    return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py" …[+437 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_overlay.py'
s=open(p).read().replace(\"pix=torch.stack([pix[...,0],pix[...,1]],-1)\\n col=dr.texture(text[None],pix[None]\", \"pix=torch.stack([pix[...,0],pix[...,1]],-1)[:, :, :, 0, :]\\n col=dr.texture(text[None],pix[None]\")
open(p,'w').write(s)
PY
python /work/scratch/render_overlay.py '[0,50,90,100,120,150,180,220,260,300,360,400,430,470,500,550,600,650,700,740,780,820,850,880,920,960]' /work/scratch/pose_overlay.jpg"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_overlay.py", line 29, in <module>
    om=np.array(h5['object'][i]); pm=np.array(h5['person'][i]); rm,rgb=render(i)
                                                                       ^^^^^^^^^
  File "/work/scratch/render_overlay.py", line 20, in render
    pix=torch.stack([pix[...,0],pix[...,1]],-1)[:, :, :, 0, :]
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
IndexError: too many indices for tensor of dimension 4

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_overlay.py';s=open(p).read()
a=\" pix=torch.stack([uv[...,0]*(tw-1), (1-uv[...,1])*(th-1)],-1)\\n pix=torch.stack([pix[...,0],pix[...,1]],-1)[:, :, :, 0, :]\\n\"
b=\" pix=torch.stack([uv[...,0]*(tw-1), (1-uv[...,1])*(th-1)],-1).reshape(1,h,w,2)\\n\"
assert a in s;s=s.replace(a,b);open(p,'w').write(s)
PY
python /work/scratch/render_overlay.py '[0,50,90,100,120,150,180,220,260,300,360,400,430,470,500,550,600,650,700,740,780,820,850,880,920,960]' /work/scratch/pose_overlay.jpg"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_overlay.py", line 28, in <module>
    om=np.array(h5['object'][i]); pm=np.array(h5['person'][i]); rm,rgb=render(i)
                                                                       ^^^^^^^^^
  File "/work/scratch/render_overlay.py", line 20, in render
    col=dr.texture(text[None],pix[None],filter_mode='linear')[0]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 439, in texture
    return _texture_func.apply(filter_mode, tex, uv, filter_mode_enum, boundary_mode_enum)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py" …[+437 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dr.texture(text\\[None\\],pix\\[None\\]/dr.texture(text[None],pix/' /work/scratch/render_overlay.py
python /work/scratch/render_overlay.py '[0,50,90,100,120,150,180,220,260,300,360,400,430,470,500,550,600,650,700,740,780,820,850,880,920,960]' /work/scratch/pose_overlay.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_mesh_views.py <<'PY'
import torch,numpy as np,trimesh,nvdiffrast.torch as dr,cv2
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); V=m.vertices;F=m.faces;UV=m.visual.uv
texture=np.array(m.visual.material.baseColorTexture)/255.;H,W=texture.shape[:2]
ctx=dr.RasterizeCudaContext();dev='cuda'; vt=torch.tensor(V,dtype=torch.float32,device=dev);ft=torch.tensor(F,dtype=torch.int32,device=dev);uvt=torch.tensor(UV,dtype=torch.float32,device=dev);tt=torch.tensor(texture,dtype=torch.float32,device=dev)
center=(V.min(0)+V.max(0))/2;radius=np.linalg.norm(V-center,axis=1).max(); ims=[]
for j,a in enumerate(np.arange(0,360,45)):
 th=np.radians(a);el=np.radians(-15);d=np.array([np.cos(el)*np.sin(th),-np.sin(el),np.cos(el)*np.cos(th)]);eye=center+d*radius*2.8
 zaxis=(eye-center);zaxis/=np.linalg.norm(zaxis);xaxis=np.cross([0,-1,0],zaxis);xaxis/=np.linalg.norm(xaxis);yaxis=np.cross(zaxis,xaxis)
 R=np.stack([xaxis,yaxis,zaxis]); t=-R@eye
 size=512; f=size*2.4;cx=cy=size/2;zn,zf=.01,100
 P=np.array([[2*f/size,0,1-2*cx/size,0],[0,2*f/size,1-2*cy/size,0],[0,0,-(zf+zn)/(zf-zn),-2*zf*zn/(zf-zn)],[0,0,1,0]],np.float32)
 M=P@np.r_[np.c_[R,t],[[0,0,0,1]]]
 clip=torch.tensor(np.c_[V,np.ones(len(V))]@M.T,dtype=torch.float32,device=dev)[None]
 rast,rdb=dr.rasterize(ctx,clip,ft,resolution=[size,size]);uv,_=dr.interpolate(uvt[None],rast,ft)
 pix=torch.stack([uv[...,0]*(W-1),(1-uv[...,1])*(H-1)],-1).reshape(1,size,size,2)
 c …[+494 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_mesh_views.py').read().replace(\"cv2.imwrite('/work/scratch/mesh_views.jpg',can)\",\"print(clip.min(1),clip.max(1),rast.max(),mask.sum(),col.min(),col.max())\"))
PY"
```
<details><summary>output</summary>

```
torch.return_types.min(
values=tensor([[-0.5852, -0.4979,  0.2488, -0.5113]], device='cuda:0'),
indices=tensor([[256161, 255814,  75885,  26942]], device='cuda:0')) torch.return_types.max(
values=tensor([[ 0.5632,  0.4716,  0.4914, -0.2688]], device='cuda:0'),
indices=tensor([[139924, 103311,  26942,  75885]], device='cuda:0')) tensor(0., device='cuda:0') 0 0.0 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_mesh_views.py';s=open(p).read();s=s.replace(\"zaxis=(eye-center);zaxis/=np.linalg.norm(zaxis);xaxis=np.cross([0,-1,0],zaxis);xaxis/=np.linalg.norm(xaxis);yaxis=np.cross(zaxis,xaxis)\",\"zaxis=(center-eye);zaxis/=np.linalg.norm(zaxis);xaxis=np.cross([0,-1,0],zaxis);xaxis/=np.linalg.norm(xaxis);yaxis=np.cross(zaxis,xaxis)\")
open(p,'w').write(s)
PY
python /work/scratch/render_mesh_views.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_mesh_views.py').read().replace(\"cv2.imwrite('/work/scratch/mesh_views.jpg',can)\",\"print(clip.min(1),clip.max(1),rast.max(),mask.sum(),col.min(),col.max())\"))
PY"
```
<details><summary>output</summary>

```
torch.return_types.min(
values=tensor([[-0.5632, -0.4979, -0.5314,  0.2688]], device='cuda:0'),
indices=tensor([[139924, 255814,  26942,  75885]], device='cuda:0')) torch.return_types.max(
values=tensor([[ 0.5852,  0.4716, -0.2888,  0.5113]], device='cuda:0'),
indices=tensor([[256161, 103311,  75885,  26942]], device='cuda:0')) tensor(0., device='cuda:0') 0 0.0 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'zaxis\\|xaxis\\|P=\\|M=' /work/scratch/render_mesh_views.py"
```
<details><summary>output</summary>

```
8: zaxis=(center-eye);zaxis/=np.linalg.norm(zaxis);xaxis=np.cross([0,-1,0],zaxis);xaxis/=np.linalg.norm(xaxis);yaxis=np.cross(zaxis,xaxis)
9: R=np.stack([xaxis,yaxis,zaxis]); t=-R@eye
11: P=np.array([[2*f/size,0,1-2*cx/size,0],[0,2*f/size,1-2*cy/size,0],[0,0,-(zf+zn)/(zf-zn),-2*zf*zn/(zf-zn)],[0,0,1,0]],np.float32)
12: M=P@np.r_[np.c_[R,t],[[0,0,0,1]]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
for p in ['/work/scratch/render_overlay.py','/work/scratch/render_mesh_views.py']:
 s=open(p).read();s=s.replace(\"[0,0,-(zf+zn)/(zf-zn),-2*zf*zn/(zf-zn)],[0,0,1,0]\",\"[0,0,0.5,0],[0,0,1,0]\")
 open(p,'w').write(s)
PY
python /work/scratch/render_mesh_views.py
python /work/scratch/render_overlay.py '[0,50,90,100,120,150,180,220,260,300,360,400,430,470,500,550,600,650,700,740,780,820,850,880,920,960]' /work/scratch/pose_overlay.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/object_crops.py <<'PY'
exec(open('/work/scratch/render_overlay.py').read().split('frames=eval')[0])
import os
frames=eval(os.sys.argv[1]);out=os.sys.argv[2]; cols=4; tiles=[]
cap=cv2.VideoCapture(root+'/video.mp4')
with h5py.File(root+'/masks.h5','r') as h5:
 for i in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok
  om=np.array(h5['object'][i]);rm,rgb=render(i); pm=np.array(h5['person'][i])
  ys,xs=np.where(om|rm)
  if len(xs):
   cx,cy=int(np.median(xs)),int(np.median(ys)); sz=max(60,int(max(xs.ptp(),ys.ptp())*2.3));x0=max(0,min(1535-sz,cx-sz//2));y0=max(0,min(1151-sz,cy-sz//2));
  else: sz=100;x0=y0=0
  a=im[y0:y0+sz,x0:x0+sz].copy(); r=(rgb[y0:y0+sz,x0:x0+sz]*255).astype(np.uint8); rm2=rm[y0:y0+sz,x0:x0+sz];om2=om[y0:y0+sz,x0:x0+sz]
  ren=np.zeros_like(a);ren[rm2]=r[rm2]
  ov=a.copy();ov[rm2]=(0.3*ov[rm2]+0.7*r[rm2]).astype(np.uint8)
  # object mask boundary green, rendered cyan
  co,_=cv2.findContours(om2.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,co,-1,(0,255,0),1)
  cr,_=cv2.findContours(rm2.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ov,cr,-1,(255,255,0),1)
  szv=300
  row=np.concatenate([cv2.resize(a,(szv,szv)),cv2.resize(ren,(szv,szv)),cv2.resize(ov,(szv,szv))],1)
  cv2.putText(row,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);tiles.append(row)
# one row per frame, 3 panels
can=np.zeros((len(tiles)*300,900,3),np.uint8)
for  …[+220 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from PIL import Image
import numpy as np
im=np.array(Image.open('/work/inputs/object_mesh.glb').convert('RGB')) if False else np.array(__import__('trimesh').load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().visual.material.baseColorTexture)
print(im.shape,np.percentile(im.reshape(-1,3),[0,10,25,50,75,90,100],axis=0))
from collections import Counter
c=Counter(map(tuple,im[::20,::20].reshape(-1,3)));print(c.most_common(20))
PY"
```
<details><summary>output</summary>

```
(2181, 3066, 3) [[  0.   0.   0.]
 [  0.   0.   0.]
 [  0.   0.   0.]
 [  0.   0.   0.]
 [ 41. 111. 116.]
 [ 52. 125. 128.]
 [248. 255. 255.]]
[((0, 0, 0), 9583), ((0, 1, 0), 181), ((1, 0, 0), 131), ((0, 0, 2), 105), ((1, 0, 2), 37), ((3, 0, 0), 32), ((0, 2, 0), 27), ((54, 129, 125), 25), ((55, 130, 126), 20), ((49, 130, 133), 17), ((0, 1, 2), 16), ((47, 110, 115), 16), ((56, 131, 127), 16), ((45, 114, 119), 14), ((48, 112, 114), 14), ((48, 113, 117), 13), ((48, 109, 114), 13), ((49, 129, 128), 13), ((58, 129, 125), 13), ((3, 0, 2), 12)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,cv2,numpy as np
im=np.array(trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry().visual.material.baseColorTexture)[:,:,::-1]
cv2.imwrite('/work/scratch/texture.jpg',cv2.resize(im,(768,546)))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np
sys.path.insert(0,'/work/scratch');from renderer import Renderer
R=Renderer('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
for size in [(1536,1152),(224,224)]:
 K2=K.copy()
 if size[0]==224:
  K2*=224/123;K2[0,2]=(K[0,2]-900)*224/123;K2[1,2]=(K[1,2]-210)*224/123
 m,rgb,d=R.render([Tcw@r['T_world_mesh'][0]],K2[None],np.array([size],np.float32))[0]
 print(size,m.shape,m.sum(),rgb.max(),d.min(),d.max())
PY"
```
<details><summary>output</summary>

```
(1536, 1152) (1152, 1536) 2261 0.63407636 -199.68361 60.411842
(224, 224) (224, 224) 0 0.0 0.0 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np
sys.path.insert(0,'/work/scratch');from renderer import Renderer
R=Renderer('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
x0,y0,sz,lim=773,549,123,224;s=lim/sz;K2=K.copy();K2[0,0]*=s;K2[1,1]*=s;K2[0,2]=(K[0,2]-x0)*s;K2[1,2]=(K[1,2]-y0)*s
print(K2)
m,rgb,d=R.render([Tcw@r['T_world_mesh'][0]],K2[None],np.array([[lim,lim]],np.float32))[0]
print(m.sum(),rgb.max(),d.min(),d.max())
# projected coords
pc=(Tcw@r['T_world_mesh'][0]@np.c_[R.V[:,:3],np.ones(len(R.V),np.float32)].T).T[:,:3];uv=(K@pc.T).T;uv=uv[:,:2]/uv[:,2:3];print(uv.min(0),uv.max(0),uv.mean(0))
PY"
```
<details><summary>output</summary>

```
[[ 1.4933367e+03  0.0000000e+00 -2.7134893e+01]
 [ 0.0000000e+00  1.4933367e+03  2.0942978e+01]
 [ 0.0000000e+00  0.0000000e+00  1.0000000e+00]]
0 0.0 0.0 0.0
[803.8799 587.3609] [864.47144 632.81604] [829.07117 609.99994]

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "cat > /work/scratch/renderer.py <<'PY'
import numpy as np,torch,trimesh,nvdiffrast.torch as dr
class Renderer:
 def __init__(self,mesh_path,device='cuda'):
  self.device=device; self.ctx=dr.RasterizeCudaContext()
  mesh=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
  self.V=np.c_[mesh.vertices,np.ones(len(mesh.vertices),np.float32)].astype(np.float32)
  self.F=mesh.faces.astype(np.int32)
  self.UV=mesh.visual.uv.astype(np.float32)
  self.texture=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255.
  self.th,self.tw=self.texture.shape[:2]
  self.vt=torch.tensor(self.V[:,:3],dtype=torch.float32,device=device)
  self.vh=torch.tensor(self.V,dtype=torch.float32,device=device)
  self.ft=torch.tensor(self.F,dtype=torch.int32,device=device)
  self.uvt=torch.tensor(self.UV,dtype=torch.float32,device=device)
  self.tt=torch.tensor(self.texture,dtype=torch.float32,device=device)
 def render(self,poses,K,sizes):
  # poses Bx4x4 world->mesh ; K Bx3x3 ; sizes Bx2 (w,h). returns mask/rgb/depth per frame list
  B=len(poses); outs=[]
  for j in range(B):
   h,w=int(sizes[j,1]),int(sizes[j,0]);fx,fy=K[j,0,0],K[j,1,1];cx,cy=K[j,0,2],K[j,1,2]
   P=np.array([[2*fx/w,0,1-2*cx/w,0],[0,2*fy/h,1-2*cy/h,0],[0,0,0.5,0],[0,0,1,0]],np.float32)
   M=torch.tensor((P@poses[j]).astype(np.float32),device=self.device)
   clip=self.vh@M.T
   rast,rdb=dr.rasterize(self.ctx,clip[None],self.ft,resolution=[h,w])
   uv,_=dr.interpolate(self.uvt[None],rast,self …[+3966 chars]
```
<details><summary>output</summary>

```
0 [0, 2371, 50803, 0, 0.0, 0, 0.0, nan, nan, 773, 549, 123]
100 [100, 2252, 35310, 0, 0.0, 0, 0.0, nan, nan, 774, 550, 121]
Traceback (most recent call last):
  File "/work/scratch/metrics.py", line 26, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); assert ok
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/1-2\\*cx\\/w/2*cx\\/w-1/g; s/1-2\\*cy\\/h/2*cy\\/h-1/g' /work/scratch/renderer.py /work/scratch/render_overlay.py /work/scratch/render_mesh_views.py
python - <<'PY'
import sys,json,numpy as np
sys.path.insert(0,'/work/scratch');from renderer import Renderer
R=Renderer('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
x0,y0,sz,lim=773,549,123,224;s=lim/sz;K2=K.copy();K2[0,0]*=s;K2[1,1]*=s;K2[0,2]=(K[0,2]-x0)*s;K2[1,2]=(K[1,2]-y0)*s
for sizes,Kt in [([(1536,1152)],K), ([(224,224)],K2)]:
 m,rgb,d=R.render([Tcw@r['T_world_mesh'][0]],Kt[None],np.array(sizes,np.float32))[0];print(m.sum(),rgb.max(),d.min(),d.max())
PY"
```
<details><summary>output</summary>

```
2260 0.63570535 -49.882904 437.59366
7485 0.87741315 -473.79977 259.12735

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c 'python /work/scratch/metrics.py'
```
<details><summary>output</summary>

```
0 [0, 2371, 50803, 7485, 0.8806245389722154, 0.9569806279225117, 0.9169226830517153, -0.002463682507393947, 0.32372686, 773, 549, 123]
100 [100, 2252, 35310, 7638, 0.8604735883424408, 0.9294333683105981, 0.9206184227621151, 0.0027734354306941265, 0.30441654, 774, 550, 121]
Traceback (most recent call last):
  File "/work/scratch/metrics.py", line 26, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); assert ok
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/f=size\\*2.4/f=size*1.1/' /work/scratch/render_mesh_views.py && python /work/scratch/render_mesh_views.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/object_crops.py';s=open(p).read();s=s.replace('r=(rgb[y0:y0+sz,x0:x0+sz]*255).astype(np.uint8)','r=(rgb[y0:y0+sz,x0:x0+sz][...,::-1]*255).astype(np.uint8)');open(p,'w').write(s)
PY
python /work/scratch/object_crops.py '[0,90,300,360,400,500,600,650,700]' /work/scratch/object_crops2.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/renderer.py /work/scratch/renderer_flip.py
sed -i 's/(1-uv\\[...,1\\])\\*(self.th-1)/uv[...,1]*(self.th-1)/' /work/scratch/renderer.py
python - <<'PY'
# quick patch object crops imports renderer and render_overlay split defines own render; inspect both
for p in ['/work/scratch/render_overlay.py']:
 s=open(p).read().replace('(1-uv[...,1])*(th-1)','uv[...,1]*(th-1)')
 open(p,'w').write(s)
PY
python /work/scratch/object_crops.py '[0,300,600,650,700]' /work/scratch/object_crops3.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();uv=m.visual.uv
print(uv.shape,np.percentile(uv,[0,1,5,25,50,75,95,99,100],axis=0))
print('zero uv',np.mean(np.all(uv<1e-6,1)),'nan',np.isnan(uv).sum())
# per vertex color true
tex=np.array(m.visual.material.baseColorTexture);H,W=tex.shape[:2];pix=tex[np.clip((uv[:,1]*(H-1)).astype(int),0,H-1),np.clip((uv[:,0]*(W-1)).astype(int),0,W-1)]
print('tex quantiles',np.percentile(pix,[0,10,25,50,75,90,100],axis=0),'black frac',np.mean(pix.sum(1)<10))
PY"
```
<details><summary>output</summary>

```
(263964, 2) [[0.001142   0.00199395]
 [0.02053756 0.05049199]
 [0.065155   0.08298597]
 [0.232894   0.32021105]
 [0.53098744 0.50839502]
 [0.65903188 0.63756573]
 [0.91534299 0.92664737]
 [0.96925098 0.96966621]
 [0.99885798 0.99839503]]
zero uv 0.0 nan 0
tex quantiles [[  0.   0.   0.]
 [  0.   0.   0.]
 [  0.   0.   0.]
 [ 29.  78.  54.]
 [ 48. 113. 118.]
 [ 74. 125. 129.]
 [208. 231. 233.]] black frac 0.4244101468382052

```
</details>

**shell** (exit 143)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/metrics.py';s=open(p).read();s=s.replace(\"for i in range(F):\\n  obj=\", \"for i in range(F):\\n  ok,im=cap.read(); assert ok\\n  obj=\")
s=s.replace(\"  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); assert ok\\n  imr=\", \"  imr=\")
open(p,'w').write(s)
PY
python /work/scratch/metrics.py"
```
<details><summary>output</summary>

```
0 [0, 2371, 50803, 7485, 0.8806245389722154, 0.9569806279225117, 0.9169226830517153, -0.002463682507393947, 0.32372686, 773, 549, 123]
100 [100, 2252, 35310, 7638, 0.8604735883424408, 0.9294333683105981, 0.9206184227621151, 0.0027734354306941265, 0.30441654, 774, 550, 121]
200 [200, 1495, 21799, 6064, 0.6431393922877776, 0.8400071187043958, 0.7329192546583851, 0.014425117906831791, 0.2071015, 686, 682, 108]
300 [300, 2142, 28532, 8512, 0.7934361014420687, 0.9372650375939849, 0.837937191471484, -0.006169277297763969, 0.20948334, 689, 712, 106]
400 [400, 2234, 20964, 8708, 0.7965543941153698, 0.9451079467156638, 0.835193829916785, -0.008193812797662731, 0.21055259, 690, 711, 107]
500 [500, 1188, 24080, 7544, 0.8420963364993216, 0.934324426044411, 0.8950784207679827, 0.0011877712748797314, 0. …[+357 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];X=r['T_cam_mesh_coconet'];cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);W=Twc@X
D=np.einsum('bij,bjk->bik',T[:,:3,:3],np.transpose(W[:,:3,:3],(0,2,1)))
print('rotation delta euler deg mean/std')
e=R.from_matrix(D).as_euler('xyz',degrees=True);print(e.mean(0),e.std(0),np.percentile(e,[0,50,100],axis=0))
# relative coco vs baseline rotations after fitting Delta frame0? angle
for i in [0,100,200,300,400,500,600,700,800,900]:
 dd=T[i,:3,:3].T@W[i,:3,:3];print(i,R.from_matrix(dd).as_euler('xyz',degrees=True))
# baseline pos vs coco world pos difference in camera coordinates
Tcw=np.linalg.inv(Twc);p=(Tcw@T)[:,:3,3];q=(Tcw@W)[:,:3,3]
# compare directions and depths
qn=q/np.linalg.norm(q,axis=1)[:,None];pn=p/np.linalg.norm(p,axis=1)[:,None];print('dir angle deg',np.percentile(np.degrees(np.arccos(np.clip((qn*pn).sum(1),-1,1))),[0,50,90,99,100]))
print('depth ratios',np.percentile(q[:,2]/p[:,2],[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
rotation delta euler deg mean/std
[-0.97986962  0.62797687  0.01867979] [0.07525628 0.05827741 0.13024695] [[-1.08806321  0.51113886 -0.15279096]
 [-0.98677686  0.65063022  0.0367537 ]
 [-0.80418641  0.71221721  0.26852621]]
0 [0.39814933 0.64325378 0.98151946]
100 [0.39814933 0.64325378 0.98151946]
200 [ 0.31786129  0.88598369 -0.71367306]
300 [-0.77713336  0.65407727 -0.5538535 ]
400 [-0.77713336  0.65407727 -0.5538535 ]
500 [ 0.81980063  0.75706809 -0.6157738 ]
600 [0.95021843 0.70577059 0.41679294]
700 [ 0.13258541  0.57026836 -0.99673455]
800 [ 0.36834563  0.43260776 -1.00898766]
900 [-0.43736479  0.52632272  0.73127807]
dir angle deg [0.95606903 1.1625887  1.27725354 1.29034183 1.29144614]
depth ratios [0.95747529 0.96681638 0.98200434 0.99557154 1.00299659]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
for lo,hi,name in [(0,94,'init'),(259,418,'static2'),(846,979,'final')]:
 q=R.from_matrix(T[lo:hi+1,:3,:3]).as_quat(); # align signs to first
 q=q*np.sign(q@q[0])[:,None]; med=np.median(q,0);med/=np.linalg.norm(med)
 ang=np.degrees(2*np.arccos(np.clip(np.abs(q@med),-1,1)))
 print('\\n',name,'ang percentiles',np.percentile(ang,[0,25,50,75,90,95,99,100]))
 # runs >5,10,20
 for th in [2,5,10,20,45]:
  m=ang>th;runs=[];st=None
  for k,v in enumerate(m):
   if v and st is None:st=k
   if st is not None and (not v or k==len(m)-1):
    en=k if v else k-1;runs.append((lo+st,lo+en,round(float(ang[st:en+1].max()),1)));st=None
  if len(runs)<30: print('thr',th,runs)
PY"
```
<details><summary>output</summary>

```

 init ang percentiles [0. 0. 0. 0. 0. 0. 0. 0.]
thr 2 []
thr 5 []
thr 10 []
thr 20 []
thr 45 []

 static2 ang percentiles [ 0.          0.          0.          0.         19.81609436 49.99031726
 66.59355581 79.05193552]
thr 2 [(259, 273, 79.1), (415, 418, 24.9)]
thr 5 [(259, 273, 79.1), (415, 418, 24.9)]
thr 10 [(259, 273, 79.1), (415, 418, 24.9)]
thr 20 [(259, 273, 79.1), (418, 418, 24.9)]
thr 45 [(259, 262, 79.1), (265, 267, 62.3), (270, 271, 49.8), (273, 273, 58.2)]

 final ang percentiles [  1.31188381  13.27545039  14.99313718  91.00045717 139.01881822
 152.77218884 158.56342349 168.85306037]
thr 2 [(846, 911, 148.5), (913, 926, 74.3), (928, 979, 168.9)]
thr 5 [(846, 911, 148.5), (913, 926, 74.3), (928, 979, 168.9)]
thr 10 [(846, 910, 148.5), (913, 925, 74.3), (928, 948, 16.1), (950 …[+251 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/scratch/object_crops.py 'list(range(840,980,10))+[979]' /work/scratch/final_crops.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');K=r['human_keypoints_world'];T=r['T_world_mesh'];c=r['contact_prob_right_left']
for i in [90,100,150,200,250,420,450,500,520,550,600,650,700,750,800,840,850,900]:
 p=T[i,:3,3]
 dr=np.linalg.norm(K[i,21:42]-p,axis=1);dl=np.linalg.norm(K[i,42:63]-p,axis=1)
 print(i,'contact',np.round(c[i],2),'Rmin med',dr.min(),np.median(dr),'Lmin med',dl.min(),np.median(dl),'centers',K[i,21:42].mean(0),K[i,42:63].mean(0),'obj',p)
PY"
```
<details><summary>output</summary>

```
[0.3064264  0.24357203 2.9282825 ] obj [0.26452932 0.27302623 2.872218  ]
650 contact [1. 0.] Rmin med 0.552004 0.5887365 Lmin med 0.023990927 0.06706392 centers [-0.04141296  0.5226115   2.5123115 ] [0.5235004  0.32908568 2.57483   ] obj [0.5263176  0.36780187 2.519519  ]
700 contact [1. 0.] Rmin med 0.5353491 0.5758313 Lmin med 0.08040789 0.10604581 centers [0.34845114 0.52551323 2.1787212 ] [0.62814933 0.28567058 2.5003362 ] obj [0.7058418  0.30509818 2.567603  ]
750 contact [1.   0.01] Rmin med 0.46985358 0.5278412 Lmin med 0.10878839 0.15787375 centers [0.6603348  0.53127366 2.1882977 ] [0.6378205 0.3546924 2.6485538] obj [0.7246168  0.24059562 2.6153862 ]
800 contact [1. 0.] Rmin med 2.4652605 2.4932542 Lmin med 1.916799 1.9494733 centers [0.74134    0.48929527 2.3096619 ] [0.7774614 …[+645 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];h=r['human_keypoints_world'];c=r['contact_prob_right_left']
for i in range(740,871,5):
 p=T[i,:3,3];lh=h[i,42:63].mean(0);rh=h[i,21:42].mean(0);print(i,'obj',np.round(p,3),'L',np.round(lh,3),'R',np.round(rh,3),'dL',round(np.linalg.norm(p-lh),3),'dR',round(np.linalg.norm(p-rh),3),'c',np.round(c[i],2))
PY"
```
<details><summary>output</summary>

```
740 obj [0.723 0.233 2.698] L [0.647 0.316 2.582] R [0.609 0.536 2.118] dL 0.161 dR 0.664 c [1.   0.21]
745 obj [0.703 0.228 2.637] L [0.63  0.334 2.628] R [0.63  0.534 2.146] dL 0.129 dR 0.583 c [1.   0.09]
750 obj [0.725 0.241 2.615] L [0.638 0.355 2.649] R [0.66  0.531 2.188] dL 0.147 dR 0.521 c [1.   0.01]
755 obj [0.8   0.242 2.728] L [0.646 0.34  2.727] R [0.684 0.524 2.221] dL 0.183 dR 0.591 c [1. 0.]
760 obj [0.745 0.198 2.645] L [0.659 0.318 2.784] R [0.717 0.517 2.269] dL 0.203 dR 0.494 c [1.   0.01]
765 obj [0.812 0.241 2.781] L [0.669 0.338 2.794] R [0.735 0.508 2.29 ] dL 0.173 dR 0.564 c [1.   0.02]
770 obj [0.825 0.219 3.024] L [0.692 0.331 2.851] R [0.762 0.506 2.337] dL 0.246 dR 0.748 c [1. 0.]
775 obj [1.247 0.466 4.405] L [0.71  0.337 2.86 ] R [0.775 0.502 2.341] dL 1.64  …[+1941 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/hand_analysis.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world'];C=r['contact_prob_right_left']
def hand_frames(idx):
 pts=K[:,idx]
 # pca basis, signs arbitrary; use axis nearest previous and cross
 out=np.zeros((len(K),3,3));prev=None
 for t in range(len(K)):
  p=pts[t];c=p.mean(0);X=p-c;w,V=np.linalg.eigh(X.T@X);V=V[:,::-1] # cols primary, secondary, normal
  # enforce right handed
  V[:,2]=np.cross(V[:,0],V[:,1]);V[:,1]=np.cross(V[:,2],V[:,0])
  if prev is not None:
   # orient each axis to prev; if ambiguous, flip first then re-cross
   if V[:,0]@prev[:,0]<0: V[:,0]*=-1;V[:,2]*=-1
   if V[:,1]@prev[:,1]<0: V[:,1]*=-1;V[:,2]*=-1
   V[:,2]=np.cross(V[:,0],V[:,1])
  out[t]=V;prev=V
 return out
for label,idx,contactcol in [('right',np.arange(21,42),0),('left',np.arange(42,63),1)]:
 H=hand_frames(idx);rel=[]
 for t in range(len(T)):
  rel.append(H[t].T@T[t,:3,:3])
 rel=np.array(rel)
 print('\\n',label)
 for lo,hi in [(96,258),(419,520),(550,750),(700,770),(770,850)]:
  q=R.from_matrix(rel[lo:hi+1]).as_quat();q*=np.sign(q@q[0])[:,None];qm=q.mean(0);qm/=np.linalg.norm(qm);a=np.degrees(2*np.arccos(np.clip(np.abs(q@qm),-1,1)))
  print(lo,hi,'contact',C[lo:hi+1,contactcol].mean(),'relang med/90/max',np.percentile(a,[50,90,100]))
# print relative euler every 10 in 550-750
H=hand_frames(np.arange(42,63));rel=np.array([H[t].T@ …[+199 chars]
```
<details><summary>output</summary>

```

 right
96 258 contact 0.00010842253 relang med/90/max [ 83.34965129 164.26241181 179.36155605]
419 520 contact 0.781236 relang med/90/max [ 59.31030353  82.43696903 117.73125651]
550 750 contact 0.99265105 relang med/90/max [ 62.06435141  99.33392147 132.53533632]
700 770 contact 0.9997979 relang med/90/max [43.93768843 54.08108337 67.55866953]
770 850 contact 0.9256592 relang med/90/max [102.10081124 173.2651812  179.90907595]

 left
96 258 contact 0.9993905 relang med/90/max [135.24109885 170.54218028 179.56080658]
419 520 contact 0.1254783 relang med/90/max [ 43.20709975  84.9498084  151.82736633]
550 750 contact 0.060644664 relang med/90/max [ 52.21907844  96.53939668 125.22064906]
700 770 contact 0.17382592 relang med/90/max [40.38540245 56.92450579 79.3477094 ]
770 850 contact 0.001 …[+998 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];fp=r['T_cam_mesh_foundationpose'];co=r['T_cam_mesh_coconet'];cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam'])
for name,X in [('fp',fp),('co',co)]:
 W=Twc@X
 # baseline delta angle but fix constant camera delta fit? direct
 ang=np.degrees(R.from_matrix(T[:,:3,:3].transpose(0,2,1)@W[:,:3,:3]).magnitude())
 print(name,np.percentile(ang,[0,10,25,50,75,90,95,99,100]))
 # print every 10 650 onwards
 print([(i,round(float(ang[i]),0)) for i in range(640,980,10)])
# fp-co differences
W1=Twc@fp;W2=Twc@co
ang=np.degrees(R.from_matrix(W1[:,:3,:3].transpose(0,2,1)@W2[:,:3,:3]).magnitude());print('fp-co',np.percentile(ang,[0,25,50,75,90,95,99,100]))
PY"
```
<details><summary>output</summary>

```
fp [  0.36499815   2.03146891   4.15706015  19.92314387  50.01193054
 118.24189887 148.10954594 174.32133877 179.40627913]
[(640, 5.0), (650, 6.0), (660, 9.0), (670, 63.0), (680, 16.0), (690, 19.0), (700, 19.0), (710, 5.0), (720, 20.0), (730, 27.0), (740, 77.0), (750, 65.0), (760, 120.0), (770, 97.0), (780, 8.0), (790, 16.0), (800, 8.0), (810, 74.0), (820, 94.0), (830, 94.0), (840, 97.0), (850, 48.0), (860, 34.0), (870, 130.0), (880, 56.0), (890, 152.0), (900, 99.0), (910, 91.0), (920, 47.0), (930, 139.0), (940, 122.0), (950, 57.0), (960, 89.0), (970, 53.0)]
co [0.95606796 1.00526368 1.15480685 1.16258889 1.23744727 1.2772536
 1.28383965 1.29034161 1.29144682]
[(640, 1.0), (650, 1.0), (660, 1.0), (670, 1.0), (680, 1.0), (690, 1.0), (700, 1.0), (710, 1.0), (720, 1.0), (730, 1.0), (740, 1.0) …[+407 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/candidate_masks.py <<'PY'
import sys,os,json,cv2,numpy as np,h5py,torch,time
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from renderer import Renderer
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K0=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
r=np.load(root+'/reconstruction.npz');T=r['T_world_mesh'].astype(np.float32);F=len(T)
fpw=np.array(cam['T_world_cam'],np.float32)@r['T_cam_mesh_foundationpose'].astype(np.float32)
# align FP world rotation to baseline by median small correction (rotation only)
Ds=np.array([T[i,:3,:3]@fpw[i,:3,:3].T for i in range(F)])
q=Rot.from_matrix(Ds).as_quat();qm=q.mean(0);qm/=np.linalg.norm(qm);Rcorr=Rot.from_quat(qm).as_matrix().astype(np.float32);print('Rcorr',Rot.from_matrix(Rcorr).as_euler('xyz',degrees=True))
flips=[np.eye(3),Rot.from_euler('y',180,degrees=True).as_matrix(),Rot.from_euler('x',180,degrees=True).as_matrix(),Rot.from_euler('z',180,degrees=True).as_matrix()]
ren=Renderer(root+'/object_mesh.glb');lim=160
out=[];start=time.time()
with h5py.File(root+'/masks.h5','r') as h5:
 for i in range(F):
  obj=np.array(h5['object'][i]);per=np.array(h5['person'][i]);ys,xs=np.where(obj)
  # baseline full projection bbox
  pc=(Tcw@T[i]@ren.V.T).T[:,:3];uv=(K0@pc.T).T;uv=uv[:,:2]/uv[:,2:3];valid=(pc[:,2]>0)&np.isfinite(uv).all(1)
  bx0,bx1,by0,by1=uv[valid,0].min(),uv[valid,0].max(),uv[val …[+1446 chars]
```
<details><summary>output</summary>

```
Rcorr [ 0.77722008 12.69515043  5.6661021 ]
Traceback (most recent call last):
  File "/work/scratch/candidate_masks.py", line 25, in <module>
    Tp=np.eye(4,np.float32);Tp[:3,:3]=Rc;Tp[:3,3]=t;poses.append(Tcw@Tp)
       ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/twodim_base.py", line 211, in eye
    m = zeros((N, M), dtype=dtype, order=order)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'type' object cannot be interpreted as an integer

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/candidate_masks.py';s=open(p).read();s=s.replace(\"q=Rot.from_matrix(Ds).as_quat();qm=q.mean(0);qm/=np.linalg.norm(qm);Rcorr=Rot.from_quat(qm).as_matrix().astype(np.float32)\",\"q=Rot.from_matrix(Ds).as_quat();q*=np.sign(q@q[0])[:,None];qm=np.median(q,0);qm/=np.linalg.norm(qm);Rcorr=Rot.from_quat(qm).as_matrix().astype(np.float32)\")
s=s.replace(\"np.eye(4,np.float32)\",\"np.eye(4,dtype=np.float32)\")
s=s.replace(\"obj[y0:y0+sz,x0:x1*0+x0+sz]\",\"obj[y0:y0+sz,x0:x0+sz]\")
open(p,'w').write(s)
PY
python /work/scratch/candidate_masks.py"
```
<details><summary>output</summary>

```
Rcorr [-0.90692526  0.39837509 -0.39069423]
Traceback (most recent call last):
  File "/work/scratch/candidate_masks.py", line 26, in <module>
    masks,_,_=ren.render_batch(np.array(poses,np.float32),Kr[None],np.array([[lim,lim]],np.float32))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/renderer.py", line 38, in render_batch
    rast,rdb=dr.rasterize(self.ctx,clip,tris,resolution=[h,w])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Fil …[+614 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/renderer.py';s=open(p).read();s=s.replace(\"tris=self.ft[None].expand(B,-1,-1)\",\"tris=self.ft[None].expand(B,-1,-1).contiguous()\").replace(\"uv,_=dr.interpolate(self.uvt[None].expand(B,-1,-1),rast,tris)\",\"uv,_=dr.interpolate(self.uvt[None].expand(B,-1,-1).contiguous(),rast,tris)\").replace(\"col=dr.texture(self.tt[None].expand(B,-1,-1,-1),pix,filter_mode='linear')\",\"col=dr.texture(self.tt[None].expand(B,-1,-1,-1).contiguous(),pix,filter_mode='linear')\")
open(p,'w').write(s)
PY
python /work/scratch/candidate_masks.py"
```
<details><summary>output</summary>

```
Rcorr [-0.90692526  0.39837509 -0.39069423]
Traceback (most recent call last):
  File "/work/scratch/candidate_masks.py", line 26, in <module>
    masks,_,_=ren.render_batch(np.array(poses,np.float32),Kr[None],np.array([[lim,lim]],np.float32))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/renderer.py", line 38, in render_batch
    rast,rdb=dr.rasterize(self.ctx,clip,tris,resolution=[h,w])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 135, in rasterize
    return _rasterize_func.apply(glctx, pos, tri, resolution, ranges, grad_db, -1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  Fil …[+569 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,100p' /work/scratch/renderer.py"
```
<details><summary>output</summary>

```
import numpy as np,torch,trimesh,nvdiffrast.torch as dr
class Renderer:
 def __init__(self,mesh_path,device='cuda'):
  self.device=device; self.ctx=dr.RasterizeCudaContext()
  mesh=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
  self.V=np.c_[mesh.vertices,np.ones(len(mesh.vertices),np.float32)].astype(np.float32)
  self.F=mesh.faces.astype(np.int32)
  self.UV=mesh.visual.uv.astype(np.float32)
  self.texture=np.array(mesh.visual.material.baseColorTexture).astype(np.float32)/255.
  self.th,self.tw=self.texture.shape[:2]
  self.vt=torch.tensor(self.V[:,:3],dtype=torch.float32,device=device)
  self.vh=torch.tensor(self.V,dtype=torch.float32,device=device)
  self.ft=torch.tensor(self.F,dtype=torch.int32,device=device)
  self.uvt=torch.tensor(self.UV,dtype=torch.float32,devic …[+1733 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np
sys.path.insert(0,'/work/scratch');from renderer import Renderer
R=Renderer('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
x0,y0,sz,lim=776,552,117,160;s=lim/sz;Kr=K.copy();Kr[0,0]*=s;Kr[1,1]*=s;Kr[0,2]=(K[0,2]-x0)*s;Kr[1,2]=(K[1,2]-y0)*s
p=(Tcw@r['T_world_mesh'][0]).astype(np.float32)
a=R.render([p],Kr[None],np.array([[lim,lim]],np.float32))[0][0].sum();b=R.render_batch(np.array([p]),Kr[None],np.array([[lim,lim]],np.float32))[0][0].sum().item();print(a,b)
PY"
```
<details><summary>output</summary>

```
4209 1052

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,json,numpy as np,torch,nvdiffrast.torch as dr
sys.path.insert(0,'/work/scratch');from renderer import Renderer
R=Renderer('/work/inputs/object_mesh.glb');r=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32)
x0,y0,sz,lim=776,552,117,160;s=lim/sz;Kr=K.copy();Kr[0,0]*=s;Kr[1,1]*=s;Kr[0,2]=(K[0,2]-x0)*s;Kr[1,2]=(K[1,2]-y0)*s
p=(Tcw@r['T_world_mesh'][0]).astype(np.float32)
P=np.array([[2*Kr[0,0]/lim,0,2*Kr[0,2]/lim-1,0],[0,2*Kr[1,1]/lim,2*Kr[1,2]/lim-1,0],[0,0,.5,0],[0,0,1,0]],np.float32)
M=P@p;clip=R.vh@torch.tensor(M,device='cuda').T
rast,_=dr.rasterize(R.ctx,clip[None],R.ft,resolution=[lim,lim]);print(rast.shape,(rast[...,3]>0).sum().item())
mm=np.einsum('ij,bjk->bik',P,np.array([p]));print(np.max(abs(mm[0]-M)))
clip2=torch.einsum('bnj,bjk->bnk',R.vh[None],torch.tensor(mm,device='cuda'))
print(torch.max(abs(clip2[0]-clip)))
rast2,_=dr.rasterize(R.ctx,clip2,R.ft,resolution=[lim,lim]);print(rast2.shape,(rast2[...,3]>0).sum().item())
PY"
```
<details><summary>output</summary>

```
torch.Size([1, 160, 160, 4]) 4209
1.9744039e-07
tensor(3.5130, device='cuda:0')
torch.Size([1, 160, 160, 4]) 1052

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/torch.einsum('bnj,bjk->bnk',self.vh\\[None\\].expand(B,-1,-1),mats)/torch.einsum('bnj,bkj->bnk',self.vh[None].expand(B,-1,-1),mats)/\" /work/scratch/renderer.py
# stop old bad run
"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/renderer.py';s=open(p).read();s=s.replace(\"tris=self.ft[None].expand(B,-1,-1).contiguous()\",\"tris=self.ft\").replace(\"uv,_=dr.interpolate(self.uvt[None].expand(B,-1,-1).contiguous(),rast,tris)\",\"uv,_=dr.interpolate(self.uvt[None].expand(B,-1,-1).contiguous(),rast,self.ft)\")
open(p,'w').write(s)
PY
python /work/scratch/candidate_masks.py"
```
<details><summary>output</summary>

```
Rcorr [-0.90692526  0.39837509 -0.39069423]
0 [0, 2371, 50803, 776, 552, 117, 0.21191470652890745, 0.9163498098859315, 0.216095045953822, 1052, 0.23112996873604288, 0.9838403041825095, 0.23201075991930062, 1052, 0.21915081822202565, 0.9420152091254753, 0.22214750056041246, 1052, 0.22620106761565836, 0.9667300380228137, 0.22797579018157363, 1052, 0.4577691811734365, 0.917312661498708, 0.47747141896435774, 2322]
100 [100, 2252, 35310, 777, 553, 115, 0.21851187808157777, 0.9019426456984274, 0.2238292011019284, 1081, 0.22713930909911945, 0.9323447636700649, 0.23094582185491278, 1079, 0.23153602174898053, 0.9462962962962963, 0.23461891643709826, 1080, 0.2406392694063927, 0.9777365491651205, 0.24196510560146925, 1078, 0.7358570820927265, 0.9090909090909091, 0.7943067033976124, 3830]
200 [200, 14 …[+1666 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));W=np.array(cam['T_world_cam'])@r['T_cam_mesh_foundationpose']
D=np.array([T[i,:3,:3]@W[i,:3,:3].T for i in range(len(T))]);q=R.from_matrix(D).as_quat();
for i in range(5):print(q[i],q[i]@q[0],np.sign(q[i]@q[0]))
q*=np.sign(q@q[0])[:,None];print('q percentiles',np.percentile(q,[0,25,50,75,100],axis=0),'mean',q.mean(0));print('norm mean',np.linalg.norm(q.mean(0)))
qm=np.median(q,0);qm/=np.linalg.norm(qm);A=R.from_quat(qm).as_matrix();print(R.from_matrix(A).as_euler('xyz',degrees=True),np.degrees(R.from_matrix(D[0]).magnitude()))
PY"
```
<details><summary>output</summary>

```
[ 0.01084341  0.8829512   0.07409318 -0.4634542 ] 1.0 1.0
[ 0.01018907  0.88044958  0.06548852 -0.4694848 ] 0.9999414527837904 1.0
[ 0.00992846  0.92418509  0.06284494 -0.37660836] 0.9953151031657625 1.0
[ 0.00393313  0.99363536  0.08133665 -0.07783087] 0.9194717205346493 1.0
[0.00590111 0.98965602 0.06603026 0.12722477] 0.8198114969018567 1.0
q percentiles [[-0.98012555 -0.46276951 -0.80840207 -0.99999493]
 [-0.00407538 -0.05551542 -0.01829738 -0.99934205]
 [ 0.00778359 -0.00345066  0.00333096 -0.984924  ]
 [ 0.02673569  0.19567854  0.02371538 -0.90626301]
 [ 0.97851137  0.99744674  0.95951204  0.88321185]] mean [ 0.04201987  0.08671469  0.01797914 -0.80381882]
norm mean 0.8097734522343446
[-0.90692542  0.39837513 -0.39069451] 124.77954679842694

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];X=r['T_cam_mesh_coconet'];cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);W=Twc@X
for i in [0,1]:
 D=T[i,:3,:3]@W[i,:3,:3].T
 E=T[i,:3,:3].T@W[i,:3,:3]
 print(i,'T',T[i,:3,:3]);print('W',W[i,:3,:3]);print('D',D,'angles D',R.from_matrix(D).magnitude()*180/np.pi,'E',R.from_matrix(E).magnitude()*180/np.pi)
PY"
```
<details><summary>output</summary>

```
0 T [[ 0.42274916 -0.00584769  0.90622675]
 [-0.05988644 -0.9979724   0.02149707]
 [ 0.9042653  -0.06335848 -0.42224336]]
W [[ 0.41238642 -0.00675827  0.9109828 ]
 [-0.07720934 -0.99663275  0.0275578 ]
 [ 0.90773077 -0.08170076 -0.41152074]]
D [[ 0.99993252 -0.00183857  0.01128908]
 [ 0.00163167  0.99982821  0.01832785]
 [-0.01132152 -0.01830869  0.99976776]] angles D 1.2374472665018168 E 1.2374472665017329
1 T [[ 0.42274916 -0.00584769  0.90622675]
 [-0.05988644 -0.9979724   0.02149707]
 [ 0.9042653  -0.06335848 -0.42224336]]
W [[ 0.41238642 -0.00675827  0.9109828 ]
 [-0.07720934 -0.99663275  0.0275578 ]
 [ 0.90773077 -0.08170076 -0.41152074]]
D [[ 0.99993252 -0.00183857  0.01128908]
 [ 0.00163167  0.99982821  0.01832785]
 [-0.01132152 -0.01830869  0.99976776]] angles D 1.2374472665018168 …[+22 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/candidate_overlay.py <<'PY'
exec(open('/work/scratch/candidate_masks.py').read().split('out=[]')[0])
import os,time
frames=eval(sys.argv[1]);outpath=sys.argv[2];cap=cv2.VideoCapture(root+'/video.mp4');cols=['baseline','yaw180','x180','z180','FPcorr'];colors=[(0,255,255),(255,0,255),(0,0,255),(255,0,0),(0,255,0)]
tiles=[]
with h5py.File(root+'/masks.h5','r') as h5:
 for i in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok
  obj=np.array(h5['object'][i]);per=np.array(h5['person'][i]);ys,xs=np.where(obj)
  pc=(Tcw@T[i]@ren.V.T).T[:,:3];uv=(K0@pc.T).T;uv=uv[:,:2]/uv[:,2:3];valid=(pc[:,2]>0)&np.isfinite(uv).all(1);bx0,bx1,by0,by1=uv[valid,0].min(),uv[valid,0].max(),uv[valid,1].min(),uv[valid,1].max()
  if len(xs):bx0=min(bx0,xs.min());bx1=max(bx1,xs.max());by0=min(by0,ys.min());by1=max(by1,ys.max())
  sz=int(np.ceil(max((bx1-bx0)*1.9,(by1-by0)*1.9,80)));cx=int(round((bx0+bx1)/2));cy=int(round((by0+by1)/2));x0=max(0,min(1536-sz,cx-sz//2));y0=max(0,min(1152-sz,cy-sz//2));sz=min(sz,1536-x0,1152-y0);scale=lim/sz;Kr=K0.copy();Kr[0,0]*=scale;Kr[1,1]*=scale;Kr[0,2]=(K0[0,2]-x0)*scale;Kr[1,2]=(K0[1,2]-y0)*scale
  Ro=T[i,:3,:3];t=T[i,:3,3];Rcands=[Ro@f for f in flips]+[Rcorr@fpw[i,:3,:3]];poses=[]
  for Rc in Rcands:
   Tp=np.eye(4,dtype=np.float32);Tp[:3,:3]=Rc;Tp[:3,3]=t;poses.append(Tcw@Tp)
  masks,_,_=ren.render_batch(np.array(poses,np.float32),Kr[None],np.array([[lim,lim]],np.float32));masks=masks.cpu().numpy()
  crop=im[y0 …[+819 chars]
```
<details><summary>output</summary>

```
Rcorr [-0.90692526  0.39837509 -0.39069423]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
im=cv2.imread('/work/scratch/candidate_overlay.jpg')
# split tiles and arrange selected as grid 4x2, upscale 1.2
idx=[0,1,6,9,12,13,16,19]
sel=[im[i*480:(i+1)*480] for i in idx]
can=np.zeros((480*2,480*4,3),np.uint8)
for j,a in enumerate(sel):can[j//4*480:(j//4+1)*480,j%4*480:(j%4+1)*480]=a
cv2.imwrite('/work/scratch/candidate_grid.jpg',can)
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
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];q=R.from_matrix(T[:,:3,:3]).as_quat()
for i in range(760,851,5):
 d=R.from_matrix(T[i,:3,:3]@T[i-5,:3,:3].T).magnitude()*180/np.pi
 print(i,'step5',round(d,1),'euler',np.round(R.from_matrix(T[i,:3,:3]).as_euler('xyz',degrees=True),0),'pos',np.round(T[i,:3,3],2))
# against 770
q0=q[770];dd=np.degrees(2*np.arccos(np.clip(np.abs(q[760:851]@q0),-1,1)))
print('rel770',[(760+i,round(float(dd[i]),0)) for i in range(0,91,5)])
PY"
```
<details><summary>output</summary>

```
760 step5 21.1 euler [89. 63. -8.] pos [0.74 0.2  2.64]
765 step5 21.1 euler [ 54.  60. -51.] pos [0.81 0.24 2.78]
770 step5 92.0 euler [  1.  44. 179.] pos [0.83 0.22 3.02]
775 step5 0.0 euler [  1.  44. 179.] pos [1.25 0.47 4.4 ]
780 step5 16.4 euler [ -1.  61. 178.] pos [1.29 0.48 4.54]
785 step5 22.9 euler [-17.  83. 164.] pos [1.32 0.48 4.61]
790 step5 4.5 euler [  20.   85. -161.] pos [1.36 0.49 4.7 ]
795 step5 40.7 euler [ -3.  45. 174.] pos [1.36 0.5  4.69]
800 step5 53.0 euler [176.  83. -11.] pos [1.38 0.51 4.72]
805 step5 13.1 euler [ -2.  85. 175.] pos [1.42 0.6  4.98]
810 step5 156.5 euler [-48. -66.  81.] pos [1.4  0.55 4.93]
815 step5 68.8 euler [-162.  -35. -160.] pos [1.44 0.66 5.03]
820 step5 0.0 euler [-162.  -35. -160.] pos [1.41 0.74 5.03]
825 step5 17.0 euler [-176.   …[+599 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
R0=T[0,:3,:3]
def fityaw(M,lo=-180,hi=180):
 f=lambda d: (R.from_euler('y',d,degrees=True).as_matrix().T@M)[1,1] # maximize dot? 
 # brute then opt
 ds=np.linspace(lo,hi,721);vals=[]
 for d in ds:
  err=((R.from_euler('y',d,degrees=True).as_matrix()@R0)-M)
  vals.append(np.linalg.norm(err))
 k=np.argmin(vals);res=minimize_scalar(lambda d:np.linalg.norm((R.from_euler('y',d,degrees=True).as_matrix()@R0)-M),bounds=(ds[max(0,k-2)],ds[min(len(ds)-1,k+2)]),method='bounded')
 Y=R.from_euler('y',res.x,degrees=True).as_matrix()@R0
 return res.x,np.degrees(R.from_matrix(M@Y.T).magnitude())
for name,a,b in [('init',0,94),('static2',274,414),('finalA',846,911),('finalB',913,926),('finalC',928,962),('finalD',963,979),('final850',846,870),('final900',890,940),('final940',930,979)]:
 M=T[a:b+1,:3,:3].mean(0);# project-ish
 U,s,Vt=np.linalg.svd(M);M=U@Vt
 print(name,fit_yaw:=fit_yaw if False else fit_yaw if False else '', end=' ')
 print(fit_yaw if False else '')
PY"
```
<details><summary>output</summary>

```
init  
static2  
finalA  
finalB  
finalC  
finalD  
final850  
final900  
final940  

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
from scipy.optimize import minimize_scalar
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];R0=T[0,:3,:3]
def fit(M):
 ds=np.linspace(-180,180,721);vals=[np.linalg.norm(RR.from_euler('y',d,degrees=True).as_matrix()@R0-M) for d in ds];k=int(np.argmin(vals))
 res=minimize_scalar(lambda d:np.linalg.norm(RR.from_euler('y',d,degrees=True).as_matrix()@R0-M),bounds=(ds[max(0,k-2)],ds[min(720,k+2)]),method='bounded');Y=RR.from_euler('y',res.x,degrees=True).as_matrix()@R0
 return res.x,np.degrees(RR.from_matrix(M@Y.T).magnitude())
for name,a,b in [('init',0,94),('static2',274,414),('finalA',846,911),('finalB',913,926),('finalC',928,962),('finalD',963,979),('final850',846,870),('final900',890,940),('final940',930,979)]:
 M=T[a:b+1,:3,:3].mean(0);U,s,Vt=np.linalg.svd(M);M=U@Vt;print(name,fit(M), 'det',np.linalg.det(M))
PY"
```
<details><summary>output</summary>

```
init (-1.3517142208240537e-05, 1.9206547660146127e-06) det 1.0000001
static2 (-162.5903002548678, 8.280745577264685) det 1.0
finalA (-16.196497319555597, 20.366326627968732) det 1.0
finalB (-84.32606485535068, 23.69351526774851) det 1.0000001
finalC (-51.808071279677776, 7.595307585049486) det 1.0
finalD (95.28048541845574, 11.50645386242151) det 1.0
final850 (59.883392226046595, 62.43878345684411) det 1.0000001
final900 (-61.27833100593973, 3.9592096912622243) det 0.9999999
final940 (-29.459196221532693, 9.00741037320159) det 1.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/candidate_masks.py'
```
<details><summary>output</summary>

```
Rcorr [-0.90692526  0.39837509 -0.39069423]
0 [0, 2371, 50803, 776, 552, 117, 0.870953819594303, 0.9588976003801378, 0.9047298811925577, 4209, 0.910939226519337, 0.9847109412326803, 0.9240080699394755, 4186, 0.6737107405971307, 0.8329338446788112, 0.7789733243667339, 4172, 0.6600647495715102, 0.8143796992481203, 0.7769558394978704, 4256, 0.9082285462166336, 0.9828121269992839, 0.9228872450123291, 4189]
100 [100, 2252, 35310, 777, 553, 115, 0.8642718864271887, 0.9320343307817212, 0.9224058769513315, 4323, 0.8998013683513573, 0.9588428974600188, 0.9359504132231405, 4279, 0.6923831959167648, 0.8269636576787808, 0.8096877869605142, 4265, 0.6760915560684747, 0.8065626434144103, 0.8069329660238751, 4358, 0.871239991343865, 0.9382428338382661, 0.9242424242424242, 4304]
200 [200, 1495, 21799, 689, …[+2458 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 a=np.array([f['object'][i].sum() for i in range(980)])
for i in range(760,980): print(i,int(a[i]))
PY"
```
<details><summary>output</summary>

```
760 0
761 0
762 0
763 0
764 0
765 0
766 0
767 0
768 0
769 0
770 0
771 0
772 0
773 0
774 0
775 0
776 0
777 0
778 0
779 0
780 0
781 0
782 0
783 0
784 0
785 0
786 0
787 0
788 0
789 0
790 0
791 0
792 0
793 0
794 0
795 0
796 0
797 0
798 0
799 0
800 0
801 0
802 0
803 0
804 0
805 0
806 0
807 0
808 0
809 0
810 0
811 0
812 0
813 0
814 0
815 0
816 0
817 0
818 0
819 0
820 0
821 0
822 0
823 0
824 0
825 0
826 0
827 0
828 0
829 0
830 0
831 0
832 0
833 0
834 0
835 0
836 0
837 0
838 0
839 0
840 0
841 0
842 0
843 0
844 0
845 0
846 0
847 713
848 672
849 709
850 762
851 899
852 792
853 793
854 757
855 721
856 723
857 720
858 704
859 685
860 681
861 677
862 674
863 648
864 593
865 498
866 389
867 250
868 0
869 0
870 0
871 0
872 0
873 0
874 0
875 0
876 0
877 0
878 0
879 0
880 0
881 0
882 0
883 0
884 0
885 0
88 …[+562 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(847,868):
  m=np.array(f['object'][i]);p=np.array(f['person'][i]);ys,xs=np.where(m);print(i,m.sum(),(xs.min(),ys.min(),xs.max(),ys.max()),(xs.mean(),ys.mean()),'overlap person',np.logical_and(m,p).sum())
PY"
```
<details><summary>output</summary>

```
847 713 (898, 818, 915, 878) (908.4263674614306, 851.6704067321178) overlap person 301
848 672 (900, 814, 914, 876) (908.0074404761905, 849.6860119047619) overlap person 215
849 709 (900, 818, 914, 875) (907.425952045134, 848.5768688293371) overlap person 122
850 762 (898, 818, 914, 874) (906.4448818897638, 847.3280839895014) overlap person 258
851 899 (896, 819, 914, 879) (905.547274749722, 848.6840934371523) overlap person 339
852 792 (896, 819, 913, 875) (905.1388888888889, 846.0037878787879) overlap person 75
853 793 (895, 819, 913, 875) (904.7679697351829, 846.1841109709962) overlap person 70
854 757 (895, 820, 913, 876) (904.8018494055482, 847.2021136063408) overlap person 128
855 721 (895, 820, 912, 875) (904.6699029126214, 846.8404993065187) overlap person 102
856 723 (895, 820, 91 …[+978 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('/work/scratch/candidate_masks.npy');names=['base','yaw180','x180','z180','fp']
for i in [0,90,100,200,250,300,360,400,420,470,500,600,650,700,740,780,800,820,840,845,847,850,855,860,865,867,880,900,920,940,960,979]:
 row=a[i];print(i,'area',int(row[1]),'person',int(row[2]),' '.join(f'{names[j]} {row[6+4*j]:.3f}/{row[7+4*j]:.2f}' for j in range(5)))
PY"
```
<details><summary>output</summary>

```
0 area 2371 person 50803 base 0.871/0.96 yaw180 0.911/0.98 x180 0.674/0.83 z180 0.660/0.81 fp 0.908/0.98
90 area 2280 person 39274 base 0.885/0.94 yaw180 0.934/0.97 x180 0.669/0.81 z180 0.655/0.79 fp 0.884/0.95
100 area 2252 person 35310 base 0.864/0.93 yaw180 0.900/0.96 x180 0.692/0.83 z180 0.676/0.81 fp 0.871/0.94
200 area 1495 person 21799 base 0.642/0.84 yaw180 0.661/0.85 x180 0.536/0.71 z180 0.518/0.70 fp 0.637/0.77
250 area 1859 person 20350 base 0.789/0.94 yaw180 0.791/0.93 x180 0.774/0.89 z180 0.804/0.92 fp 0.750/0.98
300 area 2142 person 28532 base 0.793/0.94 yaw180 0.766/0.92 x180 0.729/0.89 z180 0.760/0.92 fp 0.796/0.94
360 area 2224 person 28066 base 0.791/0.96 yaw180 0.779/0.95 x180 0.745/0.92 z180 0.774/0.95 fp 0.792/0.97
400 area 2234 person 20964 base 0.795/0.94 yaw180 0.78 …[+2581 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
from scipy.optimize import minimize_scalar
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];R0=T[0,:3,:3]
def fit(M):
 ds=np.linspace(-180,180,721);k=int(np.argmin([np.linalg.norm(RR.from_euler('y',d,degrees=True).as_matrix()@R0-M) for d in ds]));res=minimize_scalar(lambda d:np.linalg.norm(RR.from_euler('y',d,degrees=True).as_matrix()@R0-M),bounds=(ds[max(0,k-2)],ds[min(720,k+2)]),method='bounded');Y=RR.from_euler('y',res.x,degrees=True).as_matrix()@R0;return res.x,np.degrees(RR.from_matrix(M@Y.T).magnitude())
for i in range(840,881,2): print(i,fit(T[i,:3,:3]),'eul',np.round(RR.from_matrix(T[i,:3,:3]).as_euler('xyz',degrees=True),0))
PY"
```
<details><summary>output</summary>

```
840 (-140.4638739112284, 161.00513774682918) eul [-161.   12.  174.]
842 (-140.5253344203829, 160.9925464683958) eul [-161.   12.  174.]
844 (76.39358123670728, 76.82155843254988) eul [-111.   -8.   31.]
846 (76.37254999109746, 76.814643861251) eul [-111.   -8.   31.]
848 (87.91565389879503, 101.56650400363222) eul [-86. -12.  35.]
850 (72.85456093838107, 75.4122601592924) eul [-113.  -10.   30.]
852 (72.84988663438338, 75.41298686609446) eul [-113.  -10.   30.]
854 (79.85274163188478, 81.76273137350408) eul [-112.  -11.   41.]
856 (85.71626349642253, 83.68512284595327) eul [-110.   -9.   45.]
858 (72.8512118420786, 75.41937850018529) eul [-113.  -10.   30.]
860 (72.85430777723215, 75.41500903289896) eul [-113.  -10.   30.]
862 (56.41793423681026, 30.85685296606378) eul [-151.   -9.    7.] …[+605 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=pl[:3];d=pl[3]
for i in [0,90,300,360,400,847,850,855,860,865,867,900,960]:
 W=(T[i]@V.T).T[:,:3];dist=W@n+d
 print(i,'center',np.round(T[i,:3,3],3),'plane dist min/max',dist.min(),dist.max(),'bottom y',W[:,1].max(),'floor yobj',-(n[0]*T[i,0,3]+n[2]*T[i,2,3]+d)/n[1])
PY"
```
<details><summary>output</summary>

```
0 center [0.476 0.307 3.35 ] plane dist min/max 0.8177892601280983 0.9747879662078524 bottom y 0.37839836370315605 floor yobj 1.1931847876486619
90 center [0.476 0.307 3.352] plane dist min/max 0.8175186118807087 0.9745173179604627 bottom y 0.3785636771854406 floor yobj 1.1930789578489647
300 center [0.137 1.084 3.935] plane dist min/max 0.01196313358061274 0.17597076010079205 bottom y 1.1569959743255 floor yobj 1.1661307703285355
360 center [0.137 1.086 3.94 ] plane dist min/max 0.010135966180646827 0.1741435927008259 bottom y 1.1585126741164546 floor yobj 1.1658169607024327
400 center [0.133 1.062 3.856] plane dist min/max 0.03846284100499453 0.20247046752517361 bottom y 1.1349987609618526 floor yobj 1.170681734209346
847 center [0.791 1.14  3.063] plane dist min/max -0.04685459712526052 …[+1115 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world'];cam=json.load(open('/work/inputs/camera.json'));KK=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
def proj(p):return KK@(Tcw[:3,:3]@p+Tcw[:3,3]);u=proj
for i in range(845,868,2):
 o=proj(T[i,:3,3]);ol=proj(K[i,42:63].mean(0));orr=proj(K[i,21:42].mean(0));print(i,'obj',np.round(o[:2]/o[2],1),'L',np.round(ol[:2]/ol[2],1),'R',np.round(orr[:2]/orr[2],1),'worldL',np.round(K[i,42:63].mean(0),2))
PY"
```
<details><summary>output</summary>

```
845 obj [960. 775.] L [909.1 840.6] R [983.9 751.7] worldL [0.67 1.07 2.89]
847 obj [929.2 841.7] L [902.4 836.9] R [991.  749.9] worldL [0.64 1.04 2.85]
849 obj [926.  842.1] L [900.  827.2] R [986.6 748.5] worldL [0.63 1.01 2.85]
851 obj [927.3 841. ] L [893.3 816.8] R [987.4 744.4] worldL [0.6  0.95 2.8 ]
853 obj [927.3 842. ] L [889.7 804. ] R [989.1 742.3] worldL [0.58 0.9  2.79]
855 obj [931.3 840.8] L [887.1 784.6] R [989.3 734.1] worldL [0.57 0.83 2.77]
857 obj [931.5 841.9] L [886.7 768.6] R [989.3 729.7] worldL [0.55 0.76 2.71]
859 obj [931.7 842.4] L [886.3 748.9] R [987.1 723. ] worldL [0.56 0.7  2.74]
861 obj [927.7 843.8] L [883.4 733.7] R [989.4 718.7] worldL [0.54 0.65 2.72]
863 obj [929.5 852.7] L [882.5 717.7] R [993.4 714.5] worldL [0.54 0.6  2.73]
865 obj [930.3 845.9]  …[+134 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world'];C=r['contact_prob_right_left']
def frames(idx):
 out=np.zeros((len(K),3,3));prev=None
 for t,p in enumerate(K[:,idx]):
  c=p.mean(0);X=p-c;w,V=np.linalg.eigh(X.T@X);V=V[:,::-1];V[:,2]=np.cross(V[:,0],V[:,1]);V[:,1]=np.cross(V[:,2],V[:,0])
  if prev is not None:
   if V[:,0]@prev[:,0]<0:V[:,0]*=-1;V[:,2]*=-1
   if V[:,1]@prev[:,1]<0:V[:,1]*=-1;V[:,2]*=-1
   V[:,2]=np.cross(V[:,0],V[:,1])
  out[t]=V;prev=V
 return out
HR=frames(np.arange(21,42));HL=frames(np.arange(42,63))
for t in range(540,771,10):
 p=T[t,:3,3];pr=K[t,21:42].mean(0);pl=K[t,42:63].mean(0)
 # nearest, choose contact high
 j=0 if np.linalg.norm(p-pr)<np.linalg.norm(p-pl) else 1; h=pr if j==0 else pl; H=HR[t] if j==0 else HL[t];o=H.T@(p-h)
 print(t,'hand',j,'dist',np.linalg.norm(p-h),'localoff',np.round(o,3))
PY"
```
<details><summary>output</summary>

```
540 hand 1 dist 0.09661174 localoff [-0.04   0.085 -0.023]
550 hand 1 dist 0.072601214 localoff [-0.058  0.043 -0.007]
560 hand 1 dist 0.08630841 localoff [-0.084  0.01  -0.019]
570 hand 1 dist 0.07352895 localoff [-0.072  0.013 -0.005]
580 hand 1 dist 0.03213697 localoff [-0.008  0.001  0.031]
590 hand 1 dist 0.054571453 localoff [-0.041 -0.017  0.032]
600 hand 1 dist 0.07593521 localoff [-0.05  -0.041  0.04 ]
610 hand 1 dist 0.07826582 localoff [-0.059 -0.036  0.036]
620 hand 1 dist 0.076421924 localoff [-0.056 -0.045  0.026]
630 hand 1 dist 0.068035185 localoff [-0.061 -0.025  0.017]
640 hand 1 dist 0.05884385 localoff [-0.048 -0.028  0.019]
650 hand 1 dist 0.06757354 localoff [-0.057 -0.034  0.012]
660 hand 1 dist 0.066200614 localoff [-0.059 -0.031 -0.001]
670 hand 1 dist 0.08970911 l …[+620 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# construct hand frames, median offset 580:700, predict 760:850
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];K=r['human_keypoints_world']
def frames(idx):
 out=np.zeros((len(K),3,3));prev=None
 for t,p in enumerate(K[:,idx]):
  c=p.mean(0);X=p-c;w,V=np.linalg.eigh(X.T@X);V=V[:,::-1];V[:,2]=np.cross(V[:,0],V[:,1]);V[:,1]=np.cross(V[:,2],V[:,0])
  if prev is not None:
   if V[:,0]@prev[:,0]<0:V[:,0]*=-1;V[:,2]*=-1
   if V[:,1]@prev[:,1]<0:V[:,1]*=-1;V[:,2]*=-1
   V[:,2]=np.cross(V[:,0],V[:,1])
  out[t]=V;prev=V
 return out
HL=frames(np.arange(42,63));offs=[]
for t in range(580,701):
 p=T[t,:3,3];h=K[t,42:63].mean(0);offs.append(HL[t].T@(p-h))
o=np.median(offs,axis=0);print('off',o,'std',np.std(offs,axis=0))
for t in range(760,851,5):
 h=K[t,42:63].mean(0);pred=h+HL[t]@o;print(t,'pred',np.round(pred,2),'base',np.round(T[t,:3,3],2),'h',np.round(h,2))
PY"
```
<details><summary>output</summary>

```
off [-0.04698968 -0.02269889  0.01363323] std [0.02724188 0.05889938 0.04287486]
760 pred [0.7  0.35 2.78] base [0.74 0.2  2.64] h [0.66 0.32 2.78]
765 pred [0.71 0.37 2.78] base [0.81 0.24 2.78] h [0.67 0.34 2.79]
770 pred [0.74 0.36 2.84] base [0.83 0.22 3.02] h [0.69 0.33 2.85]
775 pred [0.75 0.37 2.85] base [1.25 0.47 4.4 ] h [0.71 0.34 2.86]
780 pred [0.77 0.37 2.91] base [1.29 0.48 4.54] h [0.72 0.34 2.91]
785 pred [0.76 0.38 2.92] base [1.32 0.48 4.61] h [0.72 0.35 2.93]
790 pred [0.75 0.42 2.91] base [1.36 0.49 4.7 ] h [0.71 0.38 2.92]
795 pred [0.79 0.43 2.9 ] base [1.36 0.5  4.69] h [0.75 0.39 2.91]
800 pred [0.82 0.52 2.86] base [1.38 0.51 4.72] h [0.78 0.48 2.87]
805 pred [0.87 0.64 2.88] base [1.42 0.6  4.98] h [0.83 0.61 2.88]
810 pred [0.92 0.79 2.87] base [1.4  0.55 4.93] h …[+554 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
for p in [[.7,1.12,2.89],[.62,1.04,2.85],[.7,1.13,2.9],[.7,1.15,2.85]]:
 q=Tcw[:3,:3]@p+Tcw[:3,3];u=K@q;print(p,u[:2]/u[2],q[2])
PY"
```
<details><summary>output</summary>

```
[0.7, 1.12, 2.89] [916.93040491 854.94589912] 2.8850678903385565
[0.62, 1.04, 2.85] [897.11566477 837.15624459] 2.8380549456545467
[0.7, 1.13, 2.9] [916.20577634 856.64961648] 2.8953396166187577
[0.7, 1.15, 2.85] [919.81666563 867.9599474 ] 2.8460368821398854

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_yaw_scan.py <<'PY'
import sys,json,cv2,numpy as np,h5py
from scipy.spatial.transform import Rotation as RR
sys.path.insert(0,'/work/scratch');from renderer import Renderer
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K0=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);R0=np.load(root+'/reconstruction.npz')['T_world_mesh'][0,:3,:3]
ren=Renderer(root+'/object_mesh.glb');lim=160
frames=[849,850,851,852,853,854,855,856,857,858,859,860,861]
with h5py.File(root+'/masks.h5','r') as h5:
 objs=[np.array(h5['object'][i]) for i in frames];pers=[np.array(h5['person'][i]) for i in frames]
x0,y0,sz=835,790,180;scale=lim/sz;Kr=K0.copy();Kr[0,0]*=scale;Kr[1,1]*=scale;Kr[0,2]=(K0[0,2]-x0)*scale;Kr[1,2]=(K0[1,2]-y0)*scale
objsr=[cv2.resize(o[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0 for o in objs];persr=[cv2.resize(p[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0 for p in pers]
# translation candidates around hand-attached/floor plausible
translations=[[.70,1.13,2.88],[.68,1.13,2.88],[.72,1.13,2.88],[.70,1.15,2.88],[.70,1.11,2.88],[.70,1.13,2.93],[.70,1.13,2.83],[.66,1.13,2.88],[.74,1.13,2.88]]
thetas=np.arange(-180,180,3)
best=[]
for t in translations:
 poses=[]
 for th in thetas:
  R=RR.from_euler('y',th,degrees=True).as_matrix()@R0;Tp=np.eye(4,dtype=np.float32);Tp[:3,:3]=R;Tp[:3,3]=t;poses.append(Tcw@T …[+583 chars]
```
<details><summary>output</summary>

```
t [0.7, 1.13, 2.88] best theta -51 cost 0.44738501549153015
t [0.68, 1.13, 2.88] best theta -27 cost 0.3547005299255312
t [0.72, 1.13, 2.88] best theta -45 cost 0.5882331021058735
t [0.7, 1.15, 2.88] best theta -39 cost 0.4005069621398706
t [0.7, 1.11, 2.88] best theta -24 cost 0.4752357363282558
t [0.7, 1.13, 2.93] best theta -48 cost 0.4177673101204572
t [0.7, 1.13, 2.83] best theta -42 cost 0.4584234483119418
t [0.66, 1.13, 2.88] best theta -27 cost 0.2914955094641051
t [0.74, 1.13, 2.88] best theta -114 cost 0.7326027840554237
BEST (0.7326027840554237, [0.74, 1.13, 2.88], -114)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [847,849,851,853,855,857,859,861,863,865,867]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();c=im[790:900,870:950].copy();c=cv2.resize(c,(400,550),interpolation=cv2.INTER_CUBIC);c=np.clip(c.astype(np.float32)*2.2,0,255).astype(np.uint8);cv2.putText(c,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);ims.append(c)
can=np.concatenate(ims,axis=1);cv2.imwrite('/work/scratch/final_visible.jpg',can)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [847,853,859,865]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();c=im[800:895,880:935].copy();c=cv2.resize(c,(440,760),interpolation=cv2.INTER_CUBIC);c=np.clip(c.astype(np.float32)*2.4,0,255).astype(np.uint8);cv2.putText(c,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);ims.append(c)
cv2.imwrite('/work/scratch/final_zoom.jpg',np.concatenate(ims,axis=1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_pose_view.py <<'PY'
import sys,json,cv2,numpy as np
from scipy.spatial.transform import Rotation as RR
sys.path.insert(0,'/work/scratch');from renderer import Renderer
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K0=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);R0=np.load(root+'/reconstruction.npz')['T_world_mesh'][0,:3,:3]
ren=Renderer(root+'/object_mesh.glb');frames=[847,853,859,865]
# options translation, theta, color
opts=[([.70,1.13,2.88],-51,(0,255,255)),([.72,1.13,2.88],-45,(255,0,255)),([.74,1.13,2.88],-114,(0,255,0)),([.74,1.13,2.88],-27,(0,0,255)),([.74,1.15,2.88],-114,(255,255,0))]
cap=cv2.VideoCapture(root+'/video.mp4');tiles=[]
for i in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();assert ok;crop=im[800:895,880:935].copy();crop=cv2.resize(crop,(440,760),interpolation=cv2.INTER_CUBIC)
 for (t,th,col) in opts:
  R=RR.from_euler('y',th,degrees=True).as_matrix()@R0;Tp=np.eye(4,dtype=np.float32);Tp[:3,:3]=R;Tp[:3,3]=t;pose=Tcw@Tp
  m,rgb,d=ren.render([pose],K0[None],np.array([[1536,1152]],np.float32))[0]
  mm=m[800:895,880:935].astype(np.uint8);mm=cv2.resize(mm,(440,760),interpolation=cv2.INTER_NEAREST);co,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,co,-1,col,3)
 cv2.putText(crop,str(i),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);tiles.append(crop)
cv2.imwrite('/work/scratch/final_po …[+86 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_fit.py <<'PY'
import sys,json,cv2,numpy as np,h5py,time
from scipy.spatial.transform import Rotation as RR
from scipy.optimize import differential_evolution, minimize
sys.path.insert(0,'/work/scratch');from renderer import Renderer
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K0=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);R0=np.load(root+'/reconstruction.npz')['T_world_mesh'][0,:3,:3]
pl=np.array(json.load(open(root+'/ground_plane.json'))['plane']);n=pl[:3];d=pl[3]
ren=Renderer(root+'/object_mesh.glb');import trimesh
mesh=trimesh.load(root+'/object_mesh.glb',force='scene',process=False).to_geometry();hull=mesh.convex_hull.vertices
frames=list(range(848,868));lim=160;x0,y0,sz=835,790,180;scale=lim/sz;Kr=K0.copy();Kr[0,0]*=scale;Kr[1,1]*=scale;Kr[0,2]=(K0[0,2]-x0)*scale;Kr[1,2]=(K0[1,2]-y0)*scale
with h5py.File(root+'/masks.h5','r') as h5:
 objs=[cv2.resize(np.array(h5['object'][i])[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0 for i in frames]
 pers=[cv2.resize(np.array(h5['person'][i])[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0 for i in frames]
# weight frames by target area, more area more info
weights=np.array([max(1,o.sum()) for o in objs],np.float32);weights/=weights.sum()
log=[]
def cost(p):
 x,y,z,yaw,lift=p
 R=RR.from_euler('y',yaw,degrees=True).as_matrix()@R0;T=np.eye(4,dtyp …[+1137 chars]
```
<details><summary>output</summary>

```
differential_evolution step 1: f(x)= 0.3337202109813955
differential_evolution step 2: f(x)= 0.3007179876183128
differential_evolution step 3: f(x)= 0.29919513598695324
differential_evolution step 4: f(x)= 0.29919513598695324
differential_evolution step 5: f(x)= 0.29919513598695324
differential_evolution step 6: f(x)= 0.29919513598695324
differential_evolution step 7: f(x)= 0.2654816786974757
differential_evolution step 8: f(x)= 0.2597757355665512
differential_evolution step 9: f(x)= 0.2597757355665512
differential_evolution step 10: f(x)= 0.24517261785352995
differential_evolution step 11: f(x)= 0.24517261785352995
differential_evolution step 12: f(x)= 0.24517261785352995
differential_evolution step 13: f(x)= 0.24517261785352995
differential_evolution step 14: f(x)= 0.24517261785352995
di …[+833 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/show_final_fit.py <<'PY'
import sys,json,cv2,numpy as np
from scipy.spatial.transform import Rotation as RR
sys.path.insert(0,'/work/scratch');from renderer import Renderer
root='/work/inputs';r=np.load('/work/inputs/reconstruction.npz');cam=json.load(open(root+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);R0=r['T_world_mesh'][0,:3,:3]
p=np.load('/work/scratch/final_fit_res.npy');x,y,z,yaw,lift,_=p;R=RR.from_euler('y',yaw,degrees=True).as_matrix()@R0;T=np.eye(4);T[:3,:3]=R;T[:3,3]=[x,y,z];ren=Renderer(root+'/object_mesh.glb');cap=cv2.VideoCapture(root+'/video.mp4');ims=[]
for i in [847,850,853,856,859,862,865]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();m,rgb,d=ren.render([Tcw@T],K[None],np.array([[1536,1152]],np.float32))[0];crop=im[795:900,875:945].copy();crop=cv2.resize(crop,(420,630),interpolation=cv2.INTER_CUBIC);mm=cv2.resize(m[795:900,875:945].astype(np.uint8),(420,630),interpolation=cv2.INTER_NEAREST);co,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,co,-1,(0,255,255),3);cv2.putText(crop,str(i),(8,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);ims.append(crop)
cv2.imwrite('/work/scratch/final_fit_view.jpg',np.concatenate(ims,axis=1));print('pose',p)
PY
python /work/scratch/show_final_fit.py"
```
<details><summary>output</summary>

```
pose [ 7.52174160e-01  1.13285545e+00  2.93383320e+00 -1.43910438e+02
  2.73215705e-02  2.36405598e-01]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
with h5py.File('/work/inputs/masks.h5','r') as h5:
 for i in [853,859,865]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();om=np.array(h5['object'][i]);pm=np.array(h5['person'][i]);crop=im[805:890,878:925].copy();oc=om[805:890,878:925];pc=pm[805:890,878:925];ov=crop.copy();ov[oc]=(.3*ov[oc]+.7*np.array([0,0,255])).astype(np.uint8);ov[pc]=(.7*ov[pc]+.3*np.array([0,255,0])).astype(np.uint8);dark=(crop.mean(2)<65)&~pc;ov[dark]=(.2*ov[dark]+.8*np.array([255,0,0])).astype(np.uint8);ov=cv2.resize(ov,(376,680),interpolation=cv2.INTER_NEAREST);cv2.putText(ov,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);ims.append(ov)
cv2.imwrite('/work/scratch/final_masks_view.jpg',np.concatenate(ims,axis=1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
print('components',m.split(only_watertight=False).__len__())
for j,c in enumerate(m.split(only_watertight=False)):print(j,len(c.vertices),len(c.faces),c.bounds,c.extents)
# vertex y slices counts & xz range
for yi in np.linspace(-.0666,.089,11):
 mask=np.abs(m.vertices[:,1]-yi)<.02;print('y',round(yi,3),mask.sum(),m.vertices[mask,0].min() if mask.any() else 0,m.vertices[mask,0].max() if mask.any() else 0,m.vertices[mask,2].min() if mask.any() else 0,m.vertices[mask,2].max() if mask.any() else 0)
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
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
import trimesh,numpy as np
from trimesh.graph import connected_components
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
comps=connected_components(m.edges,min_len=10)
print('ncomp',len(comps))
for j,c in enumerate(sorted(comps,key=len,reverse=True)[:20]):
 v=m.vertices[c];print(j,len(c),v.min(0),v.max(0),v.mean(0))
PY"
```
<details><summary>output</summary>

```
ncomp 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
from trimesh.graph import connected_components
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
e=np.vstack([m.faces[:,[0,1]],m.faces[:,[1,2]],m.faces[:,[2,0]]]);e=np.sort(e,axis=1);e=np.unique(e,axis=0);print(e.shape)
comps=connected_components(e,min_len=10);print('ncomp',len(comps))
for j,c in enumerate(sorted(comps,key=len,reverse=True)[:20]):
 v=m.vertices[c];print(j,len(c),np.round(v.min(0),3),np.round(v.max(0),3),np.round(v.mean(0),3))
PY"
```
<details><summary>output</summary>

```
(264012, 2)
ncomp 0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
for yi in np.linspace(v[:,1].min(),v[:,1].max(),12):
 mask=np.abs(v[:,1]-yi)<.012;w=v[mask];print('y',round(yi,3),mask.sum(),'x',np.round([w[:,0].min(),w[:,0].max()],3),'z',np.round([w[:,2].min(),w[:,2].max()],3))
# histogram at extreme x
for lo,hi in [(.08,.12),(.04,.08),(0,.04),(-.04,0),(-.08,-.04),(-.113,-.08)]:
 mask=(v[:,0]>=lo)&(v[:,0]<hi);w=v[mask];print('x',lo,hi,mask.sum(),'y',np.round([w[:,1].min(),w[:,1].max()],3),'z',np.round([w[:,2].min(),w[:,2].max()],3))
PY"
```
<details><summary>output</summary>

```
y -0.067 44154 x [-0.086  0.095] z [-0.087  0.094]
y -0.052 30382 x [-0.089  0.097] z [-0.089  0.096]
y -0.038 40386 x [-0.113  0.099] z [-0.109  0.098]
y -0.024 59379 x [-0.113  0.101] z [-0.121  0.1  ]
y -0.01 30902 x [-0.113  0.103] z [-0.124  0.102]
y 0.004 23605 x [-0.113  0.105] z [-0.124  0.106]
y 0.018 27528 x [-0.111  0.108] z [-0.124  0.107]
y 0.032 24012 x [-0.106  0.11 ] z [-0.121  0.108]
y 0.047 15434 x [-0.104  0.112] z [-0.117  0.111]
y 0.061 19396 x [-0.107  0.114] z [-0.106  0.112]
y 0.075 79286 x [-0.113  0.12 ] z [-0.113  0.119]
y 0.089 65324 x [-0.113  0.12 ] z [-0.113  0.119]
x 0.08 0.12 35526 y [-0.065  0.089] z [-0.081  0.093]
x 0.04 0.08 40163 y [-0.065  0.089] z [-0.119  0.115]
x 0 0.04 32094 y [-0.066  0.089] z [-0.124  0.119]
x -0.04 0 28495 y [-0.066  0.088] z [ …[+128 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();uv=m.visual.uv;tex=np.array(m.visual.material.baseColorTexture);H,W=tex.shape[:2];pix=tex[np.clip((uv[:,1]*(H-1)).astype(int),0,H-1),np.clip((uv[:,0]*(W-1)).astype(int),0,W-1)]
for name,mask in [('tan',(pix[:,0]>pix[:,2]*1.3)&(pix[:,0]>70)),('dark',pix.max(1)<60),('cyan',(pix[:,1]>pix[:,0]*1.4)&(pix[:,1]>50)),('white',pix.min(1)>150)]:
 v=m.vertices[mask];print(name,mask.sum(),'bbox',np.round(v.min(0),3) if len(v) else None,np.round(v.max(0),3) if len(v) else None,'mean',np.round(v.mean(0),3) if len(v) else None)
PY"
```
<details><summary>output</summary>

```
tan 21881 bbox [-0.113 -0.065 -0.103] [0.118 0.083 0.075] mean [-0.08  -0.03  -0.059]
dark 120902 bbox [-0.113 -0.067 -0.124] [0.12  0.089 0.119] mean [-0.002  0.026  0.006]
cyan 114004 bbox [-0.113 -0.067 -0.124] [0.12  0.089 0.119] mean [-0.014  0.009 -0.03 ]
white 148 bbox [ 0.07  -0.065 -0.022] [ 0.093 -0.059  0.018] mean [ 0.09  -0.063 -0.006]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
with h5py.File('/work/inputs/masks.h5','r') as h5:
 for i in [847,850,853,856,859,862,865,867]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();om=np.array(h5['object'][i]);pm=np.array(h5['person'][i]);gray=im.mean(2);dil=cv2.dilate(om.astype(np.uint8),np.ones((13,13),np.uint8))>0;comp=om|((gray<75)&~pm&dil)
  y0,y1,x0,x1=800,895,875,945;crop=im[y0:y1,x0:x1].copy();cc=comp[y0:y1,x0:x1];crop[cc]=(.25*crop[cc]+.75*np.array([255,0,0])).astype(np.uint8);crop=cv2.resize(crop,(280,380),interpolation=cv2.INTER_NEAREST);cv2.putText(crop,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);ims.append(crop)
cv2.imwrite('/work/scratch/composite_masks.jpg',np.concatenate(ims,axis=1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as h5:
 for i in [853,859,865]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();om=np.array(h5['object'][i]);pm=np.array(h5['person'][i]);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ys,xs=np.where(om);cx,cy=int(xs.mean()),int(ys.mean());roi=(gray<80)&~pm
  # limit 60 px around object
  yy,xx=np.ogrid[:1152,:1536];roi &= ((xx-cx)**2+(yy-cy)**2<60**2)
  n,lab,stats,cent=cv2.connectedComponentsWithStats(roi.astype(np.uint8),8)
  arr=[]
  for k in range(1,n):
   x,y,w,h,a=stats[k];
   if a>=3:arr.append((a,x,y,w,h,np.round(cent[k],1)))
  print(i,'objbbox',xs.min(),xs.max(),ys.min(),ys.max(),'comps',sorted(arr,reverse=True)[:20])
PY"
```
<details><summary>output</summary>

```
853 objbbox 895 913 819 875 comps [(4423, 845, 811, 68, 95, array([879.4, 857.5])), (105, 865, 793, 17, 15, array([873.1, 800.3])), (59, 898, 810, 10, 11, array([903.4, 814.6])), (19, 897, 806, 7, 5, array([899.7, 807.3])), (14, 899, 787, 7, 3, array([901.8, 788. ])), (8, 878, 792, 3, 4, array([878.9, 793.4])), (7, 903, 802, 4, 2, array([904.3, 802.6])), (6, 902, 794, 4, 2, array([903.5, 794.5])), (4, 898, 798, 2, 3, array([898.2, 799. ])), (4, 879, 802, 2, 2, array([879.5, 802.5]))]
859 objbbox 895 912 821 875 comps [(4220, 844, 811, 68, 96, array([877.5, 858.5])), (248, 865, 794, 41, 22, array([884.9, 804.7])), (113, 898, 823, 12, 17, array([904.3, 830.7])), (40, 878, 792, 11, 7, array([883.4, 795.2])), (26, 890, 797, 10, 5, array([894.7, 798.9])), (15, 902, 816, 4, 5, array([903.3, 818. …[+797 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 ims=[]
 for i in [847,853,859,865]:
  m=np.array(f['object'][i])[800:895,875:945].astype(np.uint8)*255;m=cv2.resize(m,(420,570),interpolation=cv2.INTER_NEAREST);cv2.putText(m,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(200),2);ims.append(m)
cv2.imwrite('/work/scratch/objectmask_zoom.jpg',np.concatenate(ims,axis=1))
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
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in range(640,761): print(i,int(f['object'][i].sum()))
PY"
```
<details><summary>output</summary>

```
640 4185
641 4168
642 4211
643 4306
644 4438
645 4337
646 4368
647 4401
648 4381
649 4440
650 4449
651 4523
652 4479
653 4434
654 4393
655 4454
656 4486
657 4517
658 4397
659 4397
660 4423
661 4346
662 4271
663 4294
664 4262
665 4231
666 4155
667 4081
668 4146
669 4154
670 4154
671 4151
672 4260
673 4266
674 4302
675 4300
676 4311
677 4213
678 4182
679 4296
680 4146
681 4164
682 4151
683 4104
684 4106
685 4106
686 4094
687 4072
688 4040
689 4012
690 4006
691 4010
692 4031
693 4043
694 4056
695 4054
696 4043
697 4013
698 3983
699 4077
700 3994
701 4028
702 3990
703 3958
704 3941
705 3917
706 3898
707 3932
708 3860
709 3870
710 3858
711 3877
712 3991
713 3888
714 3877
715 3898
716 3981
717 3968
718 3855
719 3891
720 3697
721 3719
722 3593
723 3425
724 3174
725 3124
726 2991
727 2865
728 2689 …[+233 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('/work/scratch/candidate_masks.npy');names=['base','yaw','x','z','fp']
for i in range(700,746):
 r=a[i];print(i,int(r[1]),' '.join(f'{names[j]}:{r[6+4*j]:.2f}' for j in range(5)))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [600,650,680,700,710,720,730,740,750]:
  m=np.array(f['object'][i]);ys,xs=np.where(m);p=T[i,:3,3];q=Tcw[:3,:3]@p+Tcw[:3,3];u=K@q;print(i,'mask',round(xs.mean(),1),round(ys.mean(),1),'pred',np.round(u[:2]/u[2],1),'diff',np.round(u[:2]/u[2]-[xs.mean(),ys.mean()],1),'p',np.round(p,2))
PY"
```
<details><summary>output</summary>

```
600 mask 797.2 619.3 pred [793.9 616.3] diff [-3.3 -3. ] p [0.26 0.27 2.87]
650 mask 897.1 668.8 pred [892.  658.1] diff [ -5.1 -10.6] p [0.53 0.37 2.52]
680 mask 953.5 660.9 pred [940.5 638.9] diff [-13. -22.] p [0.69 0.32 2.58]
700 mask 957.3 657.0 pred [945.9 635. ] diff [-11.4 -22. ] p [0.71 0.31 2.57]
710 mask 968.7 651.2 pred [944.  630.5] diff [-24.6 -20.8] p [0.7  0.29 2.56]
720 mask 973.2 650.9 pred [951.8 624. ] diff [-21.4 -26.9] p [0.73 0.27 2.57]
730 mask 971.6 641.0 pred [953.3 617.9] diff [-18.4 -23.1] p [0.75 0.26 2.62]
740 mask 972.0 633.1 pred [939.5 608.2] diff [-32.5 -24.9] p [0.72 0.23 2.7 ]
<stdin>:5: RuntimeWarning: Mean of empty slice.
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret …[+102 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/trans_opt.py <<'PY'
import sys,json,cv2,numpy as np,h5py
from scipy.optimize import minimize,differential_evolution
sys.path.insert(0,'/work/scratch');from renderer import Renderer
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K0=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);r=np.load(root+'/reconstruction.npz');T=r['T_world_mesh'];h=r['human_keypoints_world']
ren=Renderer(root+'/object_mesh.glb');lim=192
with h5py.File(root+'/masks.h5','r') as h5:
 for i in eval(sys.argv[1]):
  obj=np.array(h5['object'][i]);per=np.array(h5['person'][i]);ys,xs=np.where(obj);sz=int(max(xs.ptp(),ys.ptp())*2.6);cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;x0=max(0,min(1536-sz,cx-sz//2));y0=max(0,min(1152-sz,cy-sz//2));scale=lim/sz;Kr=K0.copy();Kr[0,0]*=scale;Kr[1,1]*=scale;Kr[0,2]=(K0[0,2]-x0)*scale;Kr[1,2]=(K0[1,2]-y0)*scale;objr=cv2.resize(obj[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0;perr=cv2.resize(per[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0
  R=T[i,:3,:3];t0=T[i,:3,3]
  def cost(t):
   Tp=np.eye(4,dtype=np.float32);Tp[:3,:3]=R;Tp[:3,3]=t;m,_,_=ren.render([Tcw@Tp],Kr[None],np.array([[lim,lim]],np.float32))[0];eff=m&~perr;inter=(eff&objr).sum();uni=(eff|objr).sum();return 1-inter/uni if uni else 1
  res=differential_evolution(cost,[(t0[0]-.2,t0[0]+.2),(t0[1]-.2,t0[1]+.2),(t0[2]-.3,t0[2]+.4)],po …[+268 chars]
```
<details><summary>output</summary>

```
600 t0 [0.265 0.273 2.872] best [0.26  0.284 2.729] iou 0.9438909281594127 cost0 0.3122383649347451
650 t0 [0.526 0.368 2.52 ] best [0.519 0.398 2.408] iou 0.9539568345323741 cost0 0.4420428056791693
700 t0 [0.706 0.305 2.568] best [0.75  0.393 2.579] iou 0.9322453016815034 cost0 0.6923755735968937
730 t0 [0.746 0.258 2.625] best [0.893 0.394 2.985] iou 0.8673524634542501 cost0 0.6965620328849028

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/scratch/trans_opt.py '[300,350,400,500,550]'"
```
<details><summary>output</summary>

```
300 t0 [0.137 1.084 3.935] best [0.117 1.024 3.664] iou 0.9499220576773187 cost0 0.21083990005765907
350 t0 [0.137 1.086 3.94 ] best [0.12  1.016 3.63 ] iou 0.94750499001996 cost0 0.21192842942345924
400 t0 [0.133 1.062 3.856] best [0.119 1.02  3.646] iou 0.9515967061658968 cost0 0.2090163934426229
500 t0 [-0.048  0.401  3.937] best [-0.045  0.386  3.739] iou 0.866847157272951 cost0 0.15602655771195095
550 t0 [0.121 0.192 3.45 ] best [0.108 0.183 3.162] iou 0.9391727493917275 cost0 0.15843828715365238

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))];T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=pl[:3];d=pl[3]
for i,t in [(300,[.117,1.024,3.664]),(350,[.12,1.016,3.63]),(400,[.119,1.02,3.646]),(300,T[300,:3,3]),(700,[.75,.393,2.579]),(730,[.893,.394,2.985])]:
 R=T[i,:3,:3];W=(R@V[:,:3].T).T+t;dist=W@n+d;print(i,'t',t,'mindist maxdist',dist.min(),dist.max(),'bottomy',W[:,1].max(),'floor_at_center',-(n@t+d)/n[1])
PY"
```
<details><summary>output</summary>

```
300 t [0.117, 1.024, 3.664] mindist maxdist 0.08780903897566916 0.2518166654958487 bottomy 1.0969177816146236 floor_at_center 0.15803721048351974
350 t [0.12, 1.016, 3.63] mindist maxdist 0.09768500966433802 0.2616926361845173 bottomy 1.0889177816146236 floor_at_center 0.16793124501397727
400 t [0.119, 1.02, 3.646] mindist maxdist 0.09279475528067116 0.25680238180085024 bottomy 1.0929177816146236 floor_at_center 0.16303204601268448
300 t [0.13669571 1.0840782  3.9345546 ] mindist maxdist 0.01196313358061274 0.17597076010079205 bottomy 1.1569959743255 floor_at_center 0.08205257761765913
700 t [0.75, 0.393, 2.579] mindist maxdist 0.7689579213814183 0.9300160783304366 bottomy 0.4671010850482289 floor_at_center 0.8391862177915632
730 t [0.893, 0.394, 2.985] mindist maxdist 0.7278312445310259 0 …[+80 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/trans_xy_opt.py <<'PY'
exec(open('/work/scratch/trans_opt.py').read().split(\"for i in eval\")[0])
with h5py.File(root+'/masks.h5','r') as h5:
 for i in eval(sys.argv[1]):
  obj=np.array(h5['object'][i]);per=np.array(h5['person'][i]);ys,xs=np.where(obj);sz=int(max(xs.ptp(),ys.ptp())*2.6);cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;x0=max(0,min(1536-sz,cx-sz//2));y0=max(0,min(1152-sz,cy-sz//2));scale=lim/sz;Kr=K0.copy();Kr[0,0]*=scale;Kr[1,1]*=scale;Kr[0,2]=(K0[0,2]-x0)*scale;Kr[1,2]=(K0[1,2]-y0)*scale;objr=cv2.resize(obj[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0;perr=cv2.resize(per[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0
  R=T[i,:3,:3];t0=T[i,:3,3]
  def cost(d):
   t=t0.copy();t[:2]+=d;Tp=np.eye(4,dtype=np.float32);Tp[:3,:3]=R;Tp[:3,3]=t;m,_,_=ren.render([Tcw@Tp],Kr[None],np.array([[lim,lim]],np.float32))[0];eff=m&~perr;inter=(eff&objr).sum();uni=(eff|objr).sum();return 1-inter/uni if uni else 1
  res=differential_evolution(cost,[(-.2,.2),(-.2,.2)],popsize=7,maxiter=15,seed=2,polish=False,disp=False);res2=minimize(cost,res.x,method='Nelder-Mead');print(i,'t0',np.round(t0,3),'delta',np.round(res2.x,3),'best',np.round(t0+np.r_[res2.x,0],3),'iou',1-res2.fun,'cost0',cost([0,0]))
PY
python /work/scratch/trans_xy_opt.py '[300,350,400,600,650,700,730]'"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/trans_xy_opt.py", line 1, in <module>
    exec(open('/work/scratch/trans_opt.py').read().split("for i in eval")[0])
  File "<string>", line 7
    
    ^
IndentationError: expected an indented block after 'with' statement on line 6

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/trans_xy_opt.py';s=open(p).read().replace(\"exec(open('/work/scratch/trans_opt.py').read().split(\\\"for i in eval\\\")[0])\",\"exec(open('/work/scratch/trans_opt.py').read().split(\\\"with h5py.File\\\")[0])\")
open(p,'w').write(s)
PY
python /work/scratch/trans_xy_opt.py '[300,350,400,600,650,700,730]'"
```
<details><summary>output</summary>

```
300 t0 [0.137 1.084 3.935] delta [-0.014  0.019] best [0.122 1.103 3.935] iou 0.877739794892419 cost0 0.21083990005765907
350 t0 [0.137 1.086 3.94 ] delta [-0.007  0.019] best [0.13  1.105 3.94 ] iou 0.855761316872428 cost0 0.21192842942345924
400 t0 [0.133 1.062 3.856] delta [-0.008  0.019] best [0.125 1.082 3.856] iou 0.8916752312435766 cost0 0.2090163934426229
600 t0 [0.265 0.273 2.872] delta [0.013 0.025] best [0.278 0.298 2.872] iou 0.8890080428954423 cost0 0.3122383649347451
650 t0 [0.526 0.368 2.52 ] delta [0.022 0.05 ] best [0.548 0.418 2.52 ] iou 0.918097106424718 cost0 0.4420428056791693
700 t0 [0.706 0.305 2.568] delta [0.041 0.086] best [0.747 0.391 2.568] iou 0.9313653136531366 cost0 0.6923755735968937
730 t0 [0.746 0.258 2.625] delta [0.014 0.088] best [0.76  0.346 2.625] iou …[+45 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as f:
 a=np.array([f['object'][i].sum() for i in range(980)])
for lo,hi in [(0,94),(95,258),(259,418),(419,745),(746,846),(847,867),(868,979)]:
 b=a[lo:hi+1];print(lo,hi,'zero',sum(b==0),'<100',sum(b<100),'min med max',b.min(),np.median(b),b.max())
# list zero runs all
m=a==0;r=[];st=None
for i,v in enumerate(m):
 if v and st is None:st=i
 if st is not None and (not v or i==979):
  en=i if v else i-1;r.append((st,en));st=None
print(r)
PY"
```
<details><summary>output</summary>

```
0 94 zero 0 <100 0 min med max 2267 2336.0 2371
95 258 zero 0 <100 0 min med max 129 1867.0 2453
259 418 zero 0 <100 0 min med max 1907 2173.5 2256
419 745 zero 0 <100 1 min med max 63 2605.0 4523
746 846 zero 101 <100 101 min med max 0 0.0 0
847 867 zero 0 <100 0 min med max 250 704.0 899
868 979 zero 112 <100 112 min med max 0 0.0 0
[(746, 846), (868, 979)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];q=R.from_matrix(T[:,:3,:3]).as_quat();ang=np.zeros(979)
for i in range(979):ang[i]=np.degrees(2*np.arccos(np.clip(abs(q[i+1]@q[i]),-1,1)))
for lo,hi in [(90,260),(415,746),(746,867)]:
 print('\\n',lo,hi)
 for i in range(lo,hi):
  if ang[i]>30:print(i,'->',i+1,round(float(ang[i]),1))
PY"
```
<details><summary>output</summary>

```

 90 260
123 -> 124 136.1
140 -> 141 35.3
194 -> 195 35.2
212 -> 213 32.3
216 -> 217 46.8
218 -> 219 36.6
252 -> 253 38.0

 415 746
439 -> 440 31.9
647 -> 648 40.2
732 -> 733 53.5
735 -> 736 31.7

 746 867
766 -> 767 91.4
775 -> 776 38.0
798 -> 799 53.4
807 -> 808 159.0
812 -> 813 87.7
825 -> 826 42.3
826 -> 827 98.6
835 -> 836 43.4
842 -> 843 145.2
861 -> 862 47.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/scratch/object_crops.py '[110,115,118,120,122,124,126,128,130,135,140,150,180,200,220,240,250,258]' /work/scratch/seg1_crops.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/dp_branch.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];A=np.load('/work/scratch/candidate_masks.npy')
yawflip=R.from_euler('y',180,degrees=True).as_matrix()
# anchors
R_A=T[0,:3,:3]
from scipy.optimize import minimize_scalar
# static2 mode median quaternion
q=R.from_matrix(T[274:415,:3,:3]).as_quat();q*=np.sign(q@q[0])[:,None];qm=np.median(q,0);qm/=np.linalg.norm(qm);R_C=R.from_quat(qm).as_matrix()
# final anchor from fit
pf=np.load('/work/scratch/final_fit_res.npy');R_E=R.from_euler('y',pf[3],degrees=True).as_matrix()@R_A

def angle(A,B):return np.degrees(R.from_matrix(A.T@B).magnitude())
def decode(lo,hi,start,end,lam,anchorw):
 Rs=[];base=[]
 for i in range(lo,hi+1):
  rb=T[i,:3,:3];Rs.append([rb,rb@yawflip]);base.append([1-A[i,6],1-A[i,7]])
 Rs=np.array(Rs);base=np.array(base)
 n=len(Rs);dp=np.full((n,2),1e9);back=np.zeros((n,2),int);dp[0]=base[0]+np.array([angle(Rs[0,0],start),angle(Rs[0,1],start)])*anchorw
 for t in range(1,n):
  for s in range(2):
   for p in range(2):
    c=dp[t-1,p]+lam*(angle(Rs[t-1,p],Rs[t,s])/180)**2+base[t,s]
    if c<dp[t,s]:dp[t,s]=c;back[t,s]=p
 dp[-1]+=np.array([angle(Rs[-1,0],end),angle(Rs[-1,1],end)])*anchorw
 s=int(np.argmin(dp[-1]));states=[s]
 for t in range(n-1,0,-1):states.append(back[t,states[-1]])
 states=states[::-1]
 return np.array(states),dp[-1]
for lam in [.5,1,2,5,10,20]:
 for aw in [.02,.05,.1]:
 …[+290 chars]
```
<details><summary>output</summary>

```
lam 0.5 aw 0.02 B flip 0.9878048780487805 runs 2 D flip 0.9484777517564403 runs 5 sB end 0 sD end 1
lam 0.5 aw 0.05 B flip 0.9878048780487805 runs 2 D flip 0.9484777517564403 runs 5 sB end 0 sD end 1
lam 0.5 aw 0.1 B flip 0.9878048780487805 runs 2 D flip 0.9484777517564403 runs 5 sB end 0 sD end 1
lam 1 aw 0.02 B flip 0.9878048780487805 runs 2 D flip 0.9484777517564403 runs 5 sB end 0 sD end 1
lam 1 aw 0.05 B flip 0.9878048780487805 runs 2 D flip 0.9484777517564403 runs 5 sB end 0 sD end 1
lam 1 aw 0.1 B flip 0.9878048780487805 runs 2 D flip 0.9484777517564403 runs 5 sB end 0 sD end 1
lam 2 aw 0.02 B flip 0.823170731707317 runs 1 D flip 0.9484777517564403 runs 5 sB end 1 sD end 1
lam 2 aw 0.05 B flip 0.8170731707317073 runs 2 D flip 0.9484777517564403 runs 5 sB end 0 sD end 1
lam 2 aw 0.1  …[+961 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i,box in [(0,(900,570,1040,660)),(300,(660,690,760,780)),(650,(850,620,970,730))]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();x0,y0,x1,y1=box;c=im[y0:y1,x0:x1].copy();c=cv2.resize(c,(600,450),interpolation=cv2.INTER_CUBIC);c=cv2.detailEnhance(c,sigma_s=10,sigma_r=.15);cv2.putText(c,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);ims.append(c)
cv2.imwrite('/work/scratch/basket_compare.jpg',np.concatenate(ims,axis=1))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
NameError: name 'np' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '1s/import cv2/import cv2,numpy as np/' /dev/stdin 2>/dev/null || true
python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i,box in [(0,(900,570,1040,660)),(300,(660,690,760,780)),(650,(850,620,970,730))]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();x0,y0,x1,y1=box;c=im[y0:y1,x0:x1].copy();c=cv2.resize(c,(600,450),interpolation=cv2.INTER_CUBIC);c=cv2.detailEnhance(c,sigma_s=10,sigma_r=.15);cv2.putText(c,str(i),(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2);ims.append(c)
cv2.imwrite('/work/scratch/basket_compare.jpg',np.concatenate(ims,axis=1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();mat=m.visual.material
print(vars(mat))
PY"
```
<details><summary>output</summary>

```
{'_data': {'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=3066x2181 at 0x77979DC230D0>, 'doubleSided': True, 'name': '_texture'}}

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_mesh_views.py';s=open(p).read();s=s.replace(\"col=dr.texture(tt[None],pix,filter_mode='linear')[0].cpu().numpy();mask=\", \"col=dr.texture(tt[None],pix,filter_mode='linear')[0].cpu().numpy()[...,::-1];mask=\")
open(p,'w').write(s)
PY
python /work/scratch/render_mesh_views.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/renderer.py';s=open(p).read();s=s.replace(\"pix=torch.stack([uv[...,0]*(self.tw-1),uv[...,1]*(self.th-1)],-1).reshape(1,h,w,2)\",\"pix=uv.reshape(1,h,w,2)\")
s=s.replace(\"pix=torch.stack([uv[...,0]*(self.tw-1),(1-uv[...,1])*(self.th-1)],-1).reshape(B,h,w,2)\",\"pix=uv.reshape(B,h,w,2)\")
open(p,'w').write(s)
# patch standalone scripts similarly
for p in ['/work/scratch/render_overlay.py','/work/scratch/render_mesh_views.py']:
 s=open(p).read();s=s.replace(\"pix=torch.stack([uv[...,0]*(tw-1), uv[...,1]*(th-1)],-1).reshape(1,h,w,2)\",\"pix=uv.reshape(1,h,w,2)\")
 s=s.replace(\"pix=torch.stack([uv[...,0]*(tw-1), uv[...,1]*(th-1)],-1).reshape(1,size,size,2)\",\"pix=uv.reshape(1,size,size,2)\")
 open(p,'w').write(s)
PY
python /work/scratch/render_mesh_views.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'pix=' /work/scratch/render_mesh_views.py /work/scratch/renderer.py /work/scratch/render_overlay.py"
```
<details><summary>output</summary>

```
/work/scratch/render_mesh_views.py:15: pix=torch.stack([uv[...,0]*(W-1),(1-uv[...,1])*(H-1)],-1).reshape(1,size,size,2)
/work/scratch/renderer.py:26:   pix=uv.reshape(1,h,w,2)
/work/scratch/renderer.py:40:  pix=torch.stack([uv[...,0]*(self.tw-1),uv[...,1]*(self.th-1)],-1).reshape(B,h,w,2)
/work/scratch/render_overlay.py:19: pix=uv.reshape(1,h,w,2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_mesh_views.py';s=open(p).read();s=s.replace(\"pix=torch.stack([uv[...,0]*(W-1),(1-uv[...,1])*(H-1)],-1).reshape(1,size,size,2)\",\"pix=uv.reshape(1,size,size,2)\");open(p,'w').write(s)
p='/work/scratch/renderer.py';s=open(p).read();s=s.replace(\"pix=torch.stack([uv[...,0]*(self.tw-1),uv[...,1]*(self.th-1)],-1).reshape(B,h,w,2)\",\"pix=uv.reshape(B,h,w,2)\");open(p,'w').write(s)
PY
python /work/scratch/render_mesh_views.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/scratch/object_crops.py '[0,300,650,700]' /work/scratch/object_crops4.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as f:
 for i in [847,853,859,865]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();om=np.array(f['object'][i]);pm=np.array(f['person'][i]);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);dil=cv2.dilate(om.astype(np.uint8),np.ones((19,19),np.uint8))>0;add=(gray<85)&~pm&dil&~om
  print(i,'obj',om.sum(),'add',add.sum())
  ys,xs=np.where(add);print((xs.min(),xs.max(),ys.min(),ys.max()) if len(xs) else None)
PY"
```
<details><summary>output</summary>

```
847 obj 713 add 448
(889, 913, 847, 887)
853 obj 793 add 777
(886, 912, 810, 884)
859 obj 685 add 772
(886, 911, 812, 884)
865 obj 498 add 663
(886, 907, 813, 880)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace/v2d_sam2 -maxdepth 3 -type f | head -60; find /weights/sam2 -maxdepth 3 -type f | head -30; sed -n '1,200p' /workspace/v2d_sam2/README.md 2>/dev/null || true"
```
<details><summary>output</summary>

```
/workspace/v2d_sam2/.dockerignore
/workspace/v2d_sam2/.gitignore
/workspace/v2d_sam2/LICENSE
/workspace/v2d_sam2/__init__.py
/workspace/v2d_sam2/assets/test_prompts.json
/workspace/v2d_sam2/assets/test_video.mp4
/workspace/v2d_sam2/docker/Dockerfile
/workspace/v2d_sam2/docker/__init__.py
/workspace/v2d_sam2/docker/_config.py
/workspace/v2d_sam2/docker/build.py
/workspace/v2d_sam2/docker/pyproject.toml
/workspace/v2d_sam2/docker/run_annotate.py
/workspace/v2d_sam2/docker/run_download_weights.py
/workspace/v2d_sam2/docker/run_mv_videos_to_masks.py
/workspace/v2d_sam2/docker/run_shell.py
/workspace/v2d_sam2/docker/run_video_to_masks.py
/workspace/v2d_sam2/docker/v2d_sam2_docker.egg-info/PKG-INFO
/workspace/v2d_sam2/docker/v2d_sam2_docker.egg-info/SOURCES.txt
/workspace/v2d_sam2/docker/v2d_sam …[+1369 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
from scipy.spatial.transform import Rotation as RR
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();uv=m.visual.uv;tex=np.array(m.visual.material.baseColorTexture);H,W=tex.shape[:2];pix=tex[np.clip((uv[:,1]*(H-1)).astype(int),0,H-1),np.clip((uv[:,0]*(W-1)).astype(int),0,W-1)]
mask=(pix[:,0]>pix[:,2]*1.3)&(pix[:,0]>70);h=m.vertices[mask].mean(0);print('handle',h)
R0=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][0,:3,:3];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
t=np.array([.752,1.133,2.934]);
for th in range(-180,180,10):
 R=RR.from_euler('y',th,degrees=True).as_matrix()@R0;p1=Tcw[:3,:3]@(R@h+t)+Tcw[:3,3];p0=Tcw[:3,:3]@t+Tcw[:3,3];u1=K@p1;u0=K@p0;du=(u1[:2]/u1[2]-u0[:2]/u0[2]);
 if th in [-150,-140,-130,-120,-110,-100,-60,-50,-40,-30,-20,0,50,90,140,180]:print(th,np.round(du,1))
PY"
```
<details><summary>output</summary>

```
handle [-0.08009544 -0.02950805 -0.05874228]
-150 [27.3  8.8]
-140 [28.1 10.7]
-130 [28.1 12.6]
-120 [27.2 14.4]
-110 [25.5 16.1]
-100 [22.9 17.5]
-60 [ 6.4 20.4]
-50 [ 1.4 20.3]
-40 [-3.6 19.8]
-30 [-8.5 19. ]
-20 [-13.2  17.9]
0 [-20.9  14.9]
50 [-27.6   5.7]
90 [-18.6   0.1]
140 [ 3.3 -0.9]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/handle_sel.py <<'PY'
import trimesh,numpy as np,cv2,torch,nvdiffrast.torch as dr
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=m.vertices;F=m.faces;uv=m.visual.uv;tex=np.array(m.visual.material.baseColorTexture).astype(np.float32)/255;H,W=tex.shape[:2]
uvf=uv[F];pix=tex[np.clip((uvf[...,1]*(H-1)).astype(int),0,H-1),np.clip((uvf[...,0]*(W-1)).astype(int),0,W-1)];col=pix.mean(1);rgb=pix.mean(0) if False else pix.mean(1) # note mean over vertices
# face texture avg actually average vertex colors
for name,sel in [('redbrown',(rgb[:,0]>rgb[:,2]*1.15)&(rgb[:,0]>.28)),('darkbrown',(rgb[:,0]>rgb[:,2]*1.05)&(rgb[:,0]<.5)&(rgb.sum(1)<1.2))]:
 fs=F[sel];print(name,sel.sum(),'bbox',V[np.unique(fs)].min(0),V[np.unique(fs)].max(0))
 # render colored points/faisl
 dev='cuda';ctx=dr.RasterizeCudaContext();size=512;center=(V.min(0)+V.max(0))/2;radius=np.linalg.norm(V-center,axis=1).max();ims=[]
 for a in [0,90,180,270]:
  th=np.radians(a);el=np.radians(-15);d=np.array([np.cos(el)*np.sin(th),-np.sin(el),np.cos(el)*np.cos(th)]);eye=center+d*radius*3.2;z=(center-eye);z/=np.linalg.norm(z);x=np.cross([0,-1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x);R=np.stack([x,y,z]);t=-R@eye
  f=size*1.2;P=np.array([[2*f/size,0,-1,0],[0,2*f/size,-1,0],[0,0,.5,0],[0,0,1,0]],np.float32);M=P@np.r_[np.c_[R,t],[[0,0,0,1]]];clip=torch.tensor(np.c_[V,np.ones(len(V))]@M.T,dtype=torch.float32,device=dev)[None];img=np.zeros((size,size,3),np …[+515 chars]
```
<details><summary>output</summary>

```
redbrown 7500 bbox [-0.11289903 -0.06594164 -0.11764763] [0.11919521 0.08401395 0.07574921]
darkbrown 9035 bbox [-0.11327863 -0.06662077 -0.12147305] [0.1203083  0.08816706 0.11938796]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_texture_scan.py <<'PY'
import sys,json,cv2,numpy as np,h5py
from scipy.spatial.transform import Rotation as RR
sys.path.insert(0,'/work/scratch');from renderer import Renderer
root='/work/inputs';r=np.load(root+'/reconstruction.npz');cam=json.load(open(root+'/camera.json'));K0=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);R0=r['T_world_mesh'][0,:3,:3]
ren=Renderer(root+'/object_mesh.glb');lim=160;x0,y0,sz=835,790,180;scale=lim/sz;Kr=K0.copy();Kr[0,0]*=scale;Kr[1,1]*=scale;Kr[0,2]=(K0[0,2]-x0)*scale;Kr[1,2]=(K0[1,2]-y0)*scale
frames=list(range(849,862));cap=cv2.VideoCapture(root+'/video.mp4');ims=[];imgs=[];objs=[];pers=[];rng=0
with h5py.File(root+'/masks.h5','r') as h5:
 for i in range(980):
  ok,im=cap.read();assert ok
  if i in frames:
   imgs.append(cv2.resize(im[y0:y0+sz,x0:x0+sz],(lim,lim),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255.);objs.append(cv2.resize(np.array(h5['object'][i])[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0);pers.append(cv2.resize(np.array(h5['person'][i])[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0)
thetas=np.arange(-180,180,3);t=np.array([.752,1.133,2.934],np.float32);scores=[];edges=[]
for th in thetas:
 R=RR.from_euler('y',th,degrees=True).as_matrix()@R0;T=np.eye(4,dtype=np.float32);T[:3,:3]=R;T[:3,3]=t;m,rgb,d=ren.render([Tcw@T],Kr[None],np.array([[li …[+968 chars]
```
<details><summary>output</summary>

```
cc 15 0.18675269955964274 0.06665020474926056
cc 6 0.18617036216036184 0.0766266539698912
cc -135 0.1743898007802862 0.09222098773783134
cc 93 0.16854169247451092 0.014414712418387486
cc 165 0.15774354439250976 0.08096752448938928
cc 18 0.14330304218833237 0.09242429454049174
cc 3 0.14191776484586818 0.06886666071693787
cc 96 0.1407057690378305 -0.05800384057645317
cc 111 0.1363116579584135 -0.1508127335554518
cc 168 0.13613264120286656 0.1295177790401835
cc 12 0.13424593711761004 0.059473737162796386
cc -153 0.12069658809227139 -0.09585884403918965
--- edge
edge 27 0.16841119357438497 0.06542545857385972
edge 24 0.14482880215482388 -0.005235476243996959
edge 21 0.1419860779760724 0.08413331006623641
edge -159 0.13277116658554694 0.09292343968211823
edge 168 0.1295177790401835 0.1361326412 …[+347 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/final_pose_view.py';s=open(p).read();s=s.replace(\"opts=[([.70,1.13,2.88],-51,(0,255,255)),([.72,1.13,2.88],-45,(255,0,255)),([.74,1.13,2.88],-114,(0,255,0)),([.74,1.13,2.88],-27,(0,0,255)),([.74,1.15,2.88],-114,(255,255,0))]\",\"opts=[([.752,1.133,2.934],-144,(0,255,255)),([.752,1.133,2.934],15,(255,0,255)),([.752,1.133,2.934],60,(0,255,0)),([.752,1.133,2.934],90,(0,0,255)),([.752,1.133,2.934],-45,(255,255,0))]\")
open('/work/scratch/final_pose_view2.py','w').write(s)
PY
python /work/scratch/final_pose_view2.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_comb_scan.py <<'PY'
exec(open('/work/scratch/final_texture_scan.py').read().split('scores=[]')[0])
# reuse loaded frames and images
th_arr=np.arange(-180,180,3);ious=[];ccs=[];floors=[]
for th in th_arr:
 R=RR.from_euler('y',th,degrees=True).as_matrix()@R0;T=np.eye(4,dtype=np.float32);T[:3,:3]=R;T[:3,3]=t;m,rgb,d=ren.render([Tcw@T],Kr[None],np.array([[lim,lim]],np.float32))[0];iv=[];cv=[]
 for im,o,p in zip(imgs,objs,pers):
  eff=m&~p;inter=(eff&o).sum();uni=(eff|o).sum();iv.append(inter/uni if uni else 0)
  msk=o&(rgb.max(2)>.05)&~p
  if msk.sum()<20:cv.append(0)
  else:
   a=im[msk].mean(1);b=rgb[msk].mean(1)
   cv.append(np.corrcoef(a,b)[0,1] if a.std()>1e-5 and b.std()>1e-5 else 0)
 ious.append(np.mean(iv));ccs.append(np.mean(cv))
ious=np.array(ious);ccs=np.array(ccs)
for k in np.argsort(ious)[-15:][::-1]:print('IOU',th_arr[k],round(ious[k],3),'cc',round(ccs[k],3))
print('--- combined')
comb=ious+.4*ccs
for k in np.argsort(comb)[-20:][::-1]:print('COMB',th_arr[k],round(comb[k],3),round(ious[k],3),round(ccs[k],3))
np.savez('/work/scratch/final_scan.npz',theta=th_arr,iou=ious,cc=ccs)
PY
python /work/scratch/final_comb_scan.py"
```
<details><summary>output</summary>

```
IOU -141 0.776 cc 0.068
IOU -144 0.775 cc 0.006
IOU -135 0.775 cc 0.174
IOU -138 0.774 cc 0.052
IOU -132 0.773 cc 0.095
IOU -147 0.773 cc -0.133
IOU -150 0.772 cc 0.056
IOU -159 0.772 cc 0.093
IOU -156 0.772 cc -0.107
IOU -129 0.771 cc -0.049
IOU -126 0.771 cc 0.11
IOU -153 0.771 cc 0.121
IOU -162 0.771 cc 0.079
IOU -123 0.77 cc 0.038
IOU -120 0.77 cc 0.007
--- combined
COMB -135 0.845 0.775 0.174
COMB 165 0.821 0.758 0.158
COMB -153 0.819 0.771 0.121
COMB -126 0.815 0.771 0.11
COMB 168 0.814 0.759 0.136
COMB -132 0.811 0.773 0.095
COMB -159 0.809 0.772 0.093
COMB 174 0.806 0.762 0.11
COMB -117 0.804 0.767 0.094
COMB -141 0.803 0.776 0.068
COMB -162 0.802 0.771 0.079
COMB -165 0.796 0.769 0.066
COMB -138 0.795 0.774 0.052
COMB -150 0.795 0.772 0.056
COMB 171 0.794 0.762 0.082
COMB -177 0.7 …[+121 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# calc hand quaternion delta angle and propagation impact
import numpy as np
from scipy.spatial.transform import Rotation as RR
r=np.load('/work/inputs/reconstruction.npz');K=r['human_keypoints_world'];T=r['T_world_mesh']
def frames(idx):
 out=np.zeros((len(K),3,3));prev=None
 for t,p in enumerate(K[:,idx]):
  c=p.mean(0);X=p-c;w,V=np.linalg.eigh(X.T@X);V=V[:,::-1];V[:,2]=np.cross(V[:,0],V[:,1]);V[:,1]=np.cross(V[:,2],V[:,0])
  if prev is not None:
   if V[:,0]@prev[:,0]<0:V[:,0]*=-1;V[:,2]*=-1
   if V[:,1]@prev[:,1]<0:V[:,1]*=-1;V[:,2]*=-1
   V[:,2]=np.cross(V[:,0],V[:,1])
  out[t]=V;prev=V
 return out
H=frames(np.arange(42,63))
for i in [745,760,780,800,820,840,846,850]:
 d=H[i]@H[745].T;print(i,RR.from_matrix(d).magnitude()*180/np.pi,RR.from_matrix(d).as_euler('xyz',degrees=True))
PY"
```
<details><summary>output</summary>

```
745 4.239903290272146e-15 [ 2.12546430e-15  0.00000000e+00 -3.49545776e-15]
760 10.754638119834203 [  2.15434417 -10.42493794  -1.74573382]
780 11.979386049979484 [  0.42627795 -11.90387272  -1.32127021]
800 22.033817618050904 [-12.50081204 -17.48219536   7.29815734]
820 54.0979438894212 [-30.24413458 -29.58009528  43.9006495 ]
840 66.04315211350693 [-48.44982894 -39.6289634   48.95758766]
846 70.21718974938229 [-44.72506361 -41.06761238  59.51873973]
850 69.65449574324072 [-38.67123326 -33.70583337  62.78548987]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];R0=T[0,:3,:3];pf=np.load('/work/scratch/final_fit_res.npy');Rf=RR.from_euler('y',-135,degrees=True).as_matrix()@R0
q=T[745,:3,:3];print('q745 to final135',RR.from_matrix(q.T@Rf).magnitude()*180/np.pi,'q745 to fit',RR.from_matrix(q.T@(RR.from_euler('y',pf[3],degrees=True).as_matrix()@R0)).magnitude()*180/np.pi)
PY"
```
<details><summary>output</summary>

```
q745 to final135 100.71143683593355 q745 to fit 95.95246964834365

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/centroid_align.py <<'PY'
exec(open('/work/scratch/trans_opt.py').read().split('with h5py.File')[0])
with h5py.File(root+'/masks.h5','r') as h5:
 for i in eval(sys.argv[1]):
  obj=np.array(h5['object'][i]);per=np.array(h5['person'][i]);ys,xs=np.where(obj);sz=int(max(xs.ptp(),ys.ptp())*2.8);cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;x0=max(0,min(1536-sz,cx-sz//2));y0=max(0,min(1152-sz,cy-sz//2));scale=lim/sz;Kr=K0.copy();Kr[0,0]*=scale;Kr[1,1]*=scale;Kr[0,2]=(K0[0,2]-x0)*scale;Kr[1,2]=(K0[1,2]-y0)*scale;objr=cv2.resize(obj[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0;perr=cv2.resize(per[y0:y0+sz,x0:x0+sz].astype(np.uint8),(lim,lim),interpolation=cv2.INTER_NEAREST)>0
  R=T[i,:3,:3];t0=T[i,:3,3];t=t0.copy();oy,ox=np.where(objr);tc=np.array([ox.mean(),oy.mean()])
  for it in range(5):
   Tp=np.eye(4,dtype=np.float32);Tp[:3,:3]=R;Tp[:3,3]=t;m,_,_=ren.render([Tcw@Tp],Kr[None],np.array([[lim,lim]],np.float32))[0];eff=m&~perr
   py,px=np.where(eff)
   if len(px)<10:break
   pc=np.array([px.mean(),py.mean()]);du=(tc-pc)/scale; # original image pixels
   q=Tcw[:3,:3]@t+Tcw[:3,3];Z=q[2];dx=du[0]*Z/K0[0,0];dy=du[1]*Z/K0[1,1];dworld=Tcw[:3,:3].T@np.array([dx,dy,0]);t+=dworld
  Tp=np.eye(4,dtype=np.float32);Tp[:3,:3]=R;Tp[:3,3]=t;m,_,_=ren.render([Tcw@Tp],Kr[None],np.array([[lim,lim]],np.float32))[0];eff=m&~perr;inter=(eff&objr).sum();uni=(eff|objr).sum();print(i,'delta',np.round(t-t0,3),'iou',inter/uni if un …[+84 chars]
```
<details><summary>output</summary>

```
300 delta [-0.01   0.017  0.   ] iou 0.8814079848806993
600 delta [ 0.01   0.025 -0.001] iou 0.8790740176667682
650 delta [ 0.018  0.048 -0.002] iou 0.9237629459148446
700 delta [ 0.037  0.084 -0.005] iou 0.9120816789563244
730 delta [ 0.014  0.094 -0.004] iou 0.8337278106508875
847 delta [ 0.013  0.019 -0.001] iou 0.25052854122621565
859 delta [-0.016  0.031  0.   ] iou 0.6850533807829181

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/scratch/centroid_align.py '[0,20,50,90]'"
```
<details><summary>output</summary>

```
0 delta [-0.009  0.003  0.   ] iou 0.9157001414427157
20 delta [-0.01   0.003  0.001] iou 0.926661152467604
50 delta [-0.01   0.003  0.001] iou 0.9279626065438549
90 delta [-0.007  0.003  0.   ] iou 0.939684329199549

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/proto_solve.py <<'PY'
import argparse,json,os,sys,time
import numpy as np,cv2,h5py,trimesh,torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import differential_evolution,minimize
from scipy.ndimage import gaussian_filter1d

np.random.seed(0); torch.manual_seed(0)
RES=160

class Renderer:
 def __init__(self,mesh_path,device='cuda'):
  self.device=device; self.ctx=dr.RasterizeCudaContext()
  m=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
  self.mesh=m; self.V=np.c_[m.vertices,np.ones(len(m.vertices),np.float32)].astype(np.float32)
  self.F=m.faces.astype(np.int32);self.UV=m.visual.uv.astype(np.float32)
  self.tex=np.array(m.visual.material.baseColorTexture).astype(np.float32)/255.
  self.vh=torch.tensor(self.V,device=device);self.ft=torch.tensor(self.F,dtype=torch.int32,device=device);self.uvt=torch.tensor(self.UV,device=device);self.tt=torch.tensor(self.tex,device=device)
  # sample of vertices for floor checks
  self.hull=m.convex_hull.vertices.astype(np.float32)
 def render_batch(self,poses,Ks,sizes):
  B=len(poses);h,w=int(sizes[0,1]),int(sizes[0,0]);fx,fy=Ks[0,0,0],Ks[0,1,1];cx,cy=Ks[0,0,2],Ks[0,1,2]
  P=np.array([[2*fx/w,0,2*cx/w-1,0],[0,2*fy/h,2*cy/h-1,0],[0,0,.5,0],[0,0,1,0]],np.float32)
  mats=torch.tensor(np.einsum('ij,bjk->bik',P,poses).astype(np.float32),device=self.device)
  clip=torch.einsum('bnj,bkj->bnk',self.vh[None].expand(B,-1,-1),mats)
  rast …[+10676 chars]
```
<details><summary>output</summary>

```
/work/scratch/proto_solve.py:74: RuntimeWarning: invalid value encountered in divide
  sig=1.0;sm=np.stack([gaussian_filter1d(seg[:,j],sig,mode='nearest') for j in range(4)],1);sm/=np.linalg.norm(sm,axis=1)[:,None]
Traceback (most recent call last):
  File "/work/scratch/proto_solve.py", line 190, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/work/scratch/proto_solve.py", line 136, in main
    RB=robust_orientations(T,95,258,start_R=R0,end_R=RC)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/proto_solve.py", line 77, in robust_orientations
    return Rot.from_matrix(Rot.from_quat(sm).as_matrix()).as_matrix()
                           ^^^^^^^^^^^^^^^^^
  File "scipy/spatial/transform/_rotation.pyx", line 1012, in  …[+217 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/proto_solve.py';s=open(p).read();s=s.replace(\"sig=1.0;sm=np.stack([gaussian_filter1d(seg[:,j],sig,mode='nearest') for j in range(4)],1);sm/=np.linalg.norm(sm,axis=1)[:,None]\",\"sig=1.0;sm=np.stack([gaussian_filter1d(seg[:,j],sig,mode='nearest') for j in range(4)],1);nrm=np.linalg.norm(sm,axis=1);bad=nrm<1e-8;sm[bad]=seg[bad];nrm[bad]=np.linalg.norm(sm[bad],axis=1);sm/=nrm[:,None]\")
s=s.replace(\"print('static',tA,iouA,RC,tC,iouC,'final',rf.x,1-rf.fun)\",\"print('static',tA,iouA,RC,tC,iouC,'final',rf.x,1-rf.fun,flush=True)\")
open(p,'w').write(s)
PY
python /work/scratch/proto_solve.py"
```
<details><summary>output</summary>

```
/work/scratch/proto_solve.py:74: RuntimeWarning: invalid value encountered in divide
  sig=1.0;sm=np.stack([gaussian_filter1d(seg[:,j],sig,mode='nearest') for j in range(4)],1);nrm=np.linalg.norm(sm,axis=1);bad=nrm<1e-8;sm[bad]=seg[bad];nrm[bad]=np.linalg.norm(sm[bad],axis=1);sm/=nrm[:,None]
Traceback (most recent call last):
  File "/work/scratch/proto_solve.py", line 190, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/work/scratch/proto_solve.py", line 136, in main
    RB=robust_orientations(T,95,258,start_R=R0,end_R=RC)
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/proto_solve.py", line 77, in robust_orientations
    return Rot.from_matrix(Rot.from_quat(sm).as_matrix()).as_matrix()
                           ^^ …[+295 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');import proto_solve as ps
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];lo,hi=95,258
q=Rot.from_matrix(T[:,:3,:3]).as_quat().copy();seg=q[lo:hi+1].copy()
for it in range(2):
 out=seg.copy()
 for i in range(len(seg)):
  ids=np.arange(max(0,i-4),min(len(seg),i+5));qs=seg[ids].copy();qs*=np.sign(qs@seg[i])[:,None];w=np.exp(-.5*((ids-i)/2.7)**2);D=1-np.abs(qs@qs.T);score=(D*w[:,None]).sum(0);k=int(np.argmin(score));out[i]=ps.quat_slerp(seg[i],qs[k],.72)
 seg=out;print('pass',it,'norms',np.linalg.norm(seg,axis=1).min(),np.isnan(seg).sum())
for i in range(1,len(seg)):
 if seg[i]@seg[i-1]<0:seg[i]*=-1
print('after hemi',np.linalg.norm(seg,axis=1).min(),np.max(np.abs(seg)))
sm=np.stack([__import__('scipy.ndimage').ndimage.gaussian_filter1d(seg[:,j],1.0,mode='nearest') for j in range(4)],1);nrm=np.linalg.norm(sm,axis=1);print('nrm',nrm.min(),nrm.max(),np.where(nrm<1e-8)[0],seg[nrm<1e-8]);
PY"
```
<details><summary>output</summary>

```
pass 0 norms 0.0 0
pass 1 norms 0.0 0
after hemi 0.0 0.9967941605966978
nrm 0.0 0.9999984738985341 [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24 138 139 140 141 142 143 144 145 146 147 148
 149 150 151 152 153] [[ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [ 0. -0.  0. -0.]
 [-0. -0.  0.  0.]
 [-0. -0.  0.  0.]
 [-0. -0.  0.  0.]
  …[+247 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/proto_solve.py';s=open(p).read();s=s.replace(\"th=np.arccos(np.clip(q0@q1,-1,1));return (np.sin((1-t)*th)*q0+np.sin(t*th)*q1)/max(np.sin(th),1e-9)\",\"th=np.arccos(np.clip(q0@q1,-1,1))\\n if th<1e-7:return q0/np.linalg.norm(q0)\\n return (np.sin((1-t)*th)*q0+np.sin(t*th)*q1)/np.sin(th)\")
open(p,'w').write(s)
PY
python /work/scratch/proto_solve.py"
```
<details><summary>output</summary>

```
static [0.46459642 0.30967867 3.3481545 ] 0.9375525106885776 [[-0.67397416  0.01625692 -0.738576  ]
 [ 0.05218286 -0.99621296 -0.06954633]
 [-0.7369096  -0.08541343  0.6705735 ]] [0.1257212 1.1070312 3.938467 ] 0.8621875676068 final [   0.7516895     1.1341983     2.92303422 -120.37643023] 0.8861875049909845
Traceback (most recent call last):
  File "/work/scratch/proto_solve.py", line 192, in <module>
    if __name__=='__main__':main()
                            ^^^^^^
  File "/work/scratch/proto_solve.py", line 164, in main
    for j,i in enumerate(range(95,259)):tB[j]=align_xy(ren,Tcw,K,objm[i],perm[i],RB[j],T[i,:3,3])
                                                                 ~~~~^^^
KeyError: 97

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/proto_solve.py';s=open(p).read();s=s.replace(\"frames_needed=set(range(0,980,2))|set([x for x in range(259,419,5)])|set(range(415,868))|set([95,100,150,200,250,258,745,846,979])\",\"frames_needed=set(range(0,980,2))|set(range(95,259))|set([x for x in range(259,419,5)])|set(range(415,868))|set([745,846,979])\")
open(p,'w').write(s)
PY
python /work/scratch/proto_solve.py"
```
<details><summary>output</summary>

```
static [0.46459642 0.30967867 3.3481545 ] 0.9375525106885776 [[-0.67397416  0.01625692 -0.738576  ]
 [ 0.05218286 -0.99621296 -0.06954633]
 [-0.7369096  -0.08541343  0.6705735 ]] [0.1257212 1.1070312 3.938467 ] 0.8621875676068 final [   0.7516895     1.1341983     2.92303422 -120.37643023] 0.8861875049909845
done 123.78460955619812

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/evaluate.py <<'PY'
import sys,json,h5py,cv2,numpy as np,time
sys.path.insert(0,'/work/scratch');from proto_solve import Renderer,make_crop,resized_masks,mask_iou,RES
root='/work/inputs';r=np.load(root+'/reconstruction.npz');Tb=r['T_world_mesh'].astype(np.float32);Tp=np.load('/work/scratch/proto_poses.npy').astype(np.float32);cam=json.load(open(root+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);ren=Renderer(root+'/object_mesh.glb');rows=[]
with h5py.File(root+'/masks.h5','r') as f:
 for i in range(0,980,2):
  obj=np.array(f['object'][i]);
  if obj.sum()<50:continue
  per=np.array(f['person'][i]);poseB=Tcw@Tb[i];poseP=Tcw@Tp[i]
  # crop includes both
  ys,xs=np.where(obj);mnB,mxB=__import__('proto_solve').projected_bbox(ren,poseB,K);mnP,mxP=__import__('proto_solve').projected_bbox(ren,poseP,K);x0=min(xs.min(),mnB[0],mnP[0]);x1=max(xs.max(),mxB[0],mxP[0]);y0=min(ys.min(),mnB[1],mnP[1]);y1=max(ys.max(),mxB[1],mxP[1]);sz=int(np.ceil(max((x1-x0)*1.8,(y1-y0)*1.8,64)));cx=int((x0+x1)/2);cy=int((y0+y1)/2);sx=max(0,min(1536-sz,cx-sz//2));sy=max(0,min(1152-sz,cy-sz//2));sz=min(sz,1536-sx,1152-sy);Kr=__import__('proto_solve').crop_K(K,sx,sy,sz,RES);o,p=resized_masks(obj,per,sx,sy,sz);mm,_,_=ren.render_batch(np.array([poseB,poseP],np.float32),Kr[None],np.array([[RES,RES]],np.float32));mm=mm.cpu().numpy();rows.append([i,obj.sum(),mask_iou(mm[0],o,p),mask_iou(mm[1],o,p)])
a …[+980 chars]
```
<details><summary>output</summary>

```
A static n 48 baseline 0.874 result 0.933 delta 0.059 improved% 100.0 worse% 0.0
B carry/work/scratch/evaluate.py:13: RuntimeWarning: Mean of empty slice.
  b=a[(a[:,0]>=lo)&(a[:,0]<=hi)];print(name,'n',len(b),'baseline',round(b[:,2].mean(),3),'result',round(b[:,3].mean(),3),'delta',round((b[:,3]-b[:,2]).mean(),3),'improved%',round(100*(b[:,3]>b[:,2]+.01).mean(),1),'worse%',round(100*(b[:,3]<b[:,2]-.01).mean(),1))
 n 82 baseline 0.757 result 0.834 delta 0.076 improved% 89.0 worse% 1.2
C static n 78 baseline 0.783 result 0.866 delta 0.083 improved% 100.0 worse% 0.0
D visible carry n 165 baseline 0.622 result 0.855 delta 0.233 improved% 83.6 worse% 2.4
D occluded n 0 baseline nan result nan delta nan improved% nan worse% nan
E placement visible n 10 baseline 0.401 result 0.756 delta 0.356 im …[+807 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/show_result.py <<'PY'
import sys,json,h5py,cv2,numpy as np
sys.path.insert(0,'/work/scratch');import proto_solve as ps
root='/work/inputs';r=np.load(root+'/reconstruction.npz');Tb=r['T_world_mesh'].astype(np.float32);Tp=np.load('/work/scratch/proto_poses.npy').astype(np.float32);cam=json.load(open(root+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);ren=ps.Renderer(root+'/object_mesh.glb');cap=cv2.VideoCapture(root+'/video.mp4');frames=eval(sys.argv[1]);ims=[]
with h5py.File(root+'/masks.h5','r') as f:
 for i in range(980):
  ok,im=cap.read();assert ok
  if i not in frames:continue
  obj=np.array(f['object'][i]);per=np.array(f['person'][i]);poses=[Tcw@Tb[i],Tcw@Tp[i]];mn,mx=ps.projected_bbox(ren,poses[0],K);mn2,mx2=ps.projected_bbox(ren,poses[1],K);ys,xs=np.where(obj);x0=min(mn[0],mn2[0],xs.min() if len(xs) else mn[0]);x1=max(mx[0],mx2[0],xs.max() if len(xs) else mx[0]);y0=min(mn[1],mn2[1],ys.min() if len(ys) else mn[1]);y1=max(mx[1],mx2[1],ys.max() if len(ys) else mx[1]);sz=int(max((x1-x0)*1.8,(y1-y0)*1.8,80));cx=int((x0+x1)/2);cy=int((y0+y1)/2);sx=max(0,min(1536-sz,cx-sz//2));sy=max(0,min(1152-sz,cy-sz//2));sz=min(sz,1536-sx,1152-sy);lim=320;Kr=ps.crop_K(K,sx,sy,sz,lim);mm,_,_=ren.render_batch(np.array(poses,np.float32),Kr[None],np.array([[lim,lim]],np.float32));mm=mm.cpu().numpy();crop=im[sy:sy+sz,sx:sx+sz].copy();crop=cv2.resize(crop,(lim,lim),interpolation …[+725 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as RR
Tp=np.load('/work/scratch/proto_poses.npy');Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
for name,T in [('base',Tb),('res',Tp)]:
 for i in [745,775,800,820,840,850,860,920]:
  p=T[i,:3,3];q=Tcw[:3,:3]@p+Tcw[:3,3];u=K@q;print(name,i,'p',np.round(p,3),'uv',np.round(u[:2]/u[2],1))
 print()
PY"
```
<details><summary>output</summary>

```
base 745 p [0.703 0.228 2.637] uv [938.9 608.3]
base 775 p [1.247 0.466 4.405] uv [946.4 622.9]
base 800 p [1.381 0.514 4.722] uv [953.4 625.2]
base 820 p [1.405 0.741 5.029] uv [942.3 656.6]
base 840 p [1.04  0.865 3.179] uv [984.8 758.9]
base 850 p [0.786 1.133 3.066] uv [927.6 839.6]
base 860 p [0.794 1.13  3.055] uv [930.6 839.8]
base 920 p [0.804 1.034 3.197] uv [923.5 801.8]

res 745 p [0.686 0.306 2.635] uv [933.8 632.7]
res 775 p [0.76  0.419 2.888] uv [934.6 656.5]
res 800 p [0.83  0.568 2.879] uv [954.8 699. ]
res 820 p [0.725 1.119 2.894] uv [923.8 854.2]
res 840 p [0.74  1.138 2.912] uv [926.6 857.3]
res 850 p [0.752 1.134 2.923] uv [928.9 854.9]
res 860 p [0.752 1.134 2.923] uv [928.9 854.9]
res 920 p [0.752 1.134 2.923] uv [928.9 854.9]


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/staticC_upright.py <<'PY'
import sys,json,cv2,numpy as np,h5py
from scipy.spatial.transform import Rotation as RR
from scipy.optimize import differential_evolution,minimize
sys.path.insert(0,'/work/scratch');import proto_solve as ps
root='/work/inputs';r=np.load(root+'/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open(root+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);pl=np.array(json.load(open(root+'/ground_plane.json'))['plane']);n=pl[:3];d=pl[3];R0=T[0,:3,:3];ren=ps.Renderer(root+'/object_mesh.glb');lim=ps.RES
frames=[290,310,330,350,370,390,410];x0,y0,sz,Kr=690,710,105,ps.crop_K(K,690,710,105);vals=[];cap=cv2.VideoCapture(root+'/video.mp4');imgs={}
with h5py.File(root+'/masks.h5','r') as f:
 for i in range(980):
  ok,im=cap.read();assert ok
  if i in frames:
   o=np.array(f['object'][i]);p=np.array(f['person'][i]);vals.append(ps.resized_masks(o,p,x0,y0,sz));imgs[i]=cv2.resize(im[y0:y0+sz,x0:x0+sz],(lim,lim),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255.
def cost(par):
 x,y,z,yaw=par;R=RR.from_euler('y',yaw,degrees=True).as_matrix()@R0;P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=[x,y,z];poses=np.repeat((Tcw@P)[None],len(vals),axis=0);mm,rgb,_=ren.render_batch(poses.astype(np.float32),Kr[None],np.array([[lim,lim]],np.float32));mm=mm.cpu().numpy();rgb=rgb.cpu().numpy();ious=[];cc=[]
 for j,(o,p) in enumerate(vals):
  ious.appen …[+602 chars]
```
<details><summary>output</summary>

```
[ 0.12254446  1.07862691  3.83846095 48.48716998] 0.9354205668091213

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# use saved? no; repeat evaluate candidate quickly with class
import sys,json,cv2,numpy as np,h5py
from scipy.spatial.transform import Rotation as RR
sys.path.insert(0,'/work/scratch');import proto_solve as ps
root='/work/inputs';r=np.load(root+'/reconstruction.npz');T=r['T_world_mesh'];cam=json.load(open(root+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);R0=T[0,:3,:3];ren=ps.Renderer(root+'/object_mesh.glb');frames=[290,310,330,350,370,390,410];x0,y0,sz=690,710,105;Kr=ps.crop_K(K,x0,y0,sz);par=[.12254446,1.07862691,3.83846095,48.48716998];R=RR.from_euler('y',par[3],degrees=True).as_matrix()@R0;P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=par[:3];poses=np.repeat((Tcw@P)[None],len(frames),axis=0);mm,_,_=ren.render_batch(poses.astype(np.float32),Kr[None],np.array([[ps.RES,ps.RES]],np.float32));mm=mm.cpu().numpy();ious=[]
with h5py.File(root+'/masks.h5','r') as f:
 for j,i in enumerate(frames):
  o,p=ps.resized_masks(np.array(f['object'][i]),np.array(f['person'][i]),x0,y0,sz);ious.append(ps.mask_iou(mm[j],o,p))
pl=np.array(json.load(open(root+'/ground_plane.json'))['plane']);n=pl[:3];d=pl[3];pd=((ren.hull@R.T)@n).min()+np.dot(n,par[:3])+d
print('ious',ious,'mean',np.mean(ious),'floor',pd)
PY"
```
<details><summary>output</summary>

```
ious [0.8706929926442122, 0.8861691542288557, 0.8833069620253164, 0.8657950394154971, 0.8845320197044335, 0.8865795724465558, 0.8649480569449788] mean 0.8774319710585499 floor 0.02353198011500579

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/proto_solve.py';s=open(p).read()
start=s.index(\" # ---------- static C: largest rotational mode in core ----------\")
end=s.index(\" # ---------- dynamic orientations ----------\")
new=''' # ---------- static C: upright on floor after first placement ----------
 fitC=[290,310,330,350,370,390,410]
 objC={};perC={}
 with h5py.File(root+'/masks.h5','r') as f:
  for i in fitC:
   objC[i]=np.array(f['object'][i]);perC[i]=np.array(f['person'][i])
 xc0,yc0,szc=690,710,105;Krc=crop_K(K,xc0,yc0,szc);valsC=[resized_masks(objC[i],perC[i],xc0,yc0,szc) for i in fitC]
 imgsC={};capC=cv2.VideoCapture(root+'/video.mp4')
 for i in range(980):
  ok,im=capC.read();assert ok
  if i in fitC:imgsC[i]=cv2.resize(im[yc0:yc0+szc,xc0:xc0+szc],(RES,RES),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255.
 def costC(par):
  x,y,z,yaw=par;R=Rot.from_euler('y',yaw,degrees=True).as_matrix()@R0;P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=[x,y,z];poses=np.repeat((Tcw@P)[None],len(valsC),axis=0);mm,rgb,_=ren.render_batch(poses.astype(np.float32),Krc[None],np.array([[RES,RES]],np.float32));mm=mm.cpu().numpy();rgb=rgb.cpu().numpy();ious=[];ccs=[]
  for j,(o,p) in enumerate(valsC):
   ious.append(mask_iou(mm[j],o,p));msk=o&(rgb[j].max(2)>.05)&~p;a=imgsC[fitC[j]][msk].mean(1);b=rgb[j][msk].mean(1);ccs.append(np.corrcoef(a,b)[0,1] if msk.sum()>20 and a.std()>1e-4 and b.std()>1e-4 else 0)
  pd=((ren.hull@R.T)@n).min()+np.dot(n,[x,y,z])+d;return 1-n …[+479 chars]
```
<details><summary>output</summary>

```
static [0.46459642 0.30967867 3.3481545 ] 0.9375525106885776 [[ 0.9373614  -0.04881942  0.34491833]
 [-0.05988644 -0.99797243  0.02149707]
 [ 0.34317076 -0.04080633 -0.9383857 ]] [0.11866937 1.0606955  3.78332   ] 0.9577977567639505 final [   0.7516895     1.1341983     2.92303422 -120.37643023] 0.8861875049909845
done 122.93008422851562

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/evaluate.py'
```
<details><summary>output</summary>

```
/work/scratch/evaluate.py:13: RuntimeWarning: Mean of empty slice.
  b=a[(a[:,0]>=lo)&(a[:,0]<=hi)];print(name,'n',len(b),'baseline',round(b[:,2].mean(),3),'result',round(b[:,3].mean(),3),'delta',round((b[:,3]-b[:,2]).mean(),3),'improved%',round(100*(b[:,3]>b[:,2]+.01).mean(),1),'worse%',round(100*(b[:,3]<b[:,2]-.01).mean(),1))
A static n 48 baseline 0.874 result 0.933 delta 0.059 improved% 100.0 worse% 0.0
B carry n 82 baseline 0.757 result 0.834 delta 0.076 improved% 89.0 worse% 1.2
C static n 78 baseline 0.783 result 0.901 delta 0.118 improved% 100.0 worse% 0.0
D visible carry n 165 baseline 0.622 result 0.855 delta 0.233 improved% 83.6 worse% 2.4
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret = ret.dt …[+805 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/scratch/proto_poses.npy')
for i in list(range(740,751))+list(range(835,851)):
 if i==0:continue
 d=RR.from_matrix(T[i,:3,:3]@T[i-1,:3,:3].T).magnitude()*180/np.pi
 print(i,round(d,1),np.round(RR.from_matrix(T[i,:3,:3]).as_euler('xyz',degrees=True),0))
PY"
```
<details><summary>output</summary>

```
740 1.7 [ 59.  64. -47.]
741 0.4 [ 60.  64. -46.]
742 0.0 [ 60.  64. -46.]
743 0.0 [ 60.  64. -46.]
744 0.1 [ 60.  64. -46.]
745 0.2 [ 60.  64. -46.]
746 5.3 [ 50.  63. -53.]
747 2.1 [ 53.  63. -48.]
748 3.1 [ 55.  60. -49.]
749 1.2 [ 54.  61. -49.]
750 0.8 [ 54.  62. -48.]
835 3.1 [  11.   20. -146.]
836 4.2 [   7.   20. -149.]
837 2.9 [   7.   18. -152.]
838 3.1 [   6.   17. -155.]
839 2.7 [   5.   16. -157.]
840 3.1 [   4.   14. -160.]
841 2.9 [   4.   13. -163.]
842 3.0 [   3.   12. -165.]
843 3.4 [   2.   10. -168.]
844 2.9 [   2.    9. -171.]
845 3.3 [   1.    7. -174.]
846 3.2 [   2.    5. -177.]
847 0.0 [   2.    5. -177.]
848 0.0 [   2.    5. -177.]
849 0.0 [   2.    5. -177.]
850 0.0 [   2.    5. -177.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/scratch/proto_poses.npy');ds=np.degrees(RR.from_matrix(T[1:,:3,:3]@T[:-1,:3,:3].transpose(0,2,1)).magnitude())
for k in np.argsort(ds)[-20:][::-1]:print(k,k+1,ds[k])
PY"
```
<details><summary>output</summary>

```
415 416 172.1871298150406
257 258 146.372223591489
123 124 56.29098887083202
122 123 30.00524563420109
124 125 29.643374458933465
732 733 22.116308946080114
252 253 15.277071842750559
647 648 14.050554061329674
731 732 13.906499930282036
733 734 13.843680178970494
797 798 13.146988717627254
155 156 12.909396669368638
156 157 12.232432386191016
218 219 11.885818372637583
154 155 11.359152406587839
648 649 11.281327296001814
215 216 10.795157385573194
466 467 10.787344929350732
598 599 10.782237827087194
795 796 10.777253836610331

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '60,85p' /work/scratch/proto_solve.py"
```
<details><summary>output</summary>

```
 return (np.sin((1-t)*th)*q0+np.sin(t*th)*q1)/np.sin(th)

def matrix_slerp(A,B,t):return Rot.from_quat(quat_slerp(Rot.from_matrix(A).as_quat(),Rot.from_matrix(B).as_quat(),t)).as_matrix()

def robust_orientations(T,lo,hi,start_R=None,end_R=None):
 q=Rot.from_matrix(T[:,:3,:3]).as_quat().copy();seg=q[lo:hi+1].copy()
 # two robust rolling medoid passes
 for _ in range(2):
  out=seg.copy()
  for i in range(len(seg)):
   ids=np.arange(max(0,i-4),min(len(seg),i+5));qs=seg[ids].copy();qs*=np.sign(qs@seg[i])[:,None];w=np.exp(-.5*((ids-i)/2.7)**2)
   D=1-np.abs(qs@qs.T);score=(D*w[:,None]).sum(0);k=int(np.argmin(score));out[i]=quat_slerp(seg[i],qs[k],.72)
  seg=out
 # hemisphere-consistent light smoothing
 for i in range(1,len(seg)):
  if seg[i]@seg[i-1]<0:seg[i]*=-1
 sig=1.0;sm=np.stack([gaussian …[+647 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/test_unwrap.py <<'PY'
import sys,numpy as np
from scipy.spatial.transform import Rotation as RR
sys.path.insert(0,'/work/scratch');import proto_solve as ps
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];R0=T[0,:3,:3];RC=None
# load current RC from proto poses static frame300? not current fit; compute script output known not saved. use proto T300? wrong. Instead use Tp[259]? Tp saved with old? It's new after rerun yes Tp static C.
Tp=np.load('/work/scratch/proto_poses.npy');RC=Tp[300,:3,:3]
qflip=RR.from_euler('y',180,degrees=True).as_quat()
def smooth(lo,hi,start,end):
 q=RR.from_matrix(T[lo:hi+1,:3,:3]).as_quat();prev=RR.from_matrix(start).as_quat();out=[]
 for x in q:
  c=[x,x* qflip] # WRONG quaternion multiplication maybe
  # use matrices robust
  R=RR.from_quat(x).as_matrix();cands=[R,R@RR.from_euler('y',180,degrees=True).as_matrix()];ang=[RR.from_matrix(prev.T@c).magnitude() for c in cands];k=np.argmin(ang);out.append(cands[k]);prev=cands[k]
 out=np.array(out)
 # smooth matrices as quats
 qq=RR.from_matrix(out).as_quat();sm=np.stack([__import__('scipy.ndimage').ndimage.gaussian_filter1d(qq[:,j],1,mode='nearest') for j in range(4)],1);sm/=np.linalg.norm(sm,axis=1)[:,None]
 return out,sm
for lo,hi,st,en in [(95,258,R0,RC),(415,745,RC,RC)]:
 out,sm=smooth(lo,hi,st,en);ds=np.degrees(RR.from_matrix(out[1:]@out[:-1].transpose(0,2,1)).magnitude());print(lo,hi,'maxstep',ds.max(),'median',np.median(ds),'start/end angles',RR.fro …[+135 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_unwrap.py", line 19, in <module>
    out,sm=smooth(lo,hi,st,en);ds=np.degrees(RR.from_matrix(out[1:]@out[:-1].transpose(0,2,1)).magnitude());print(lo,hi,'maxstep',ds.max(),'median',np.median(ds),'start/end angles',RR.from_matrix(st.T@out[0]).magnitude()*180/np.pi,RR.from_matrix(en.T@out[-1]).magnitude()*180/np.pi)
           ^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/test_unwrap.py", line 13, in smooth
    R=RR.from_quat(x).as_matrix();cands=[R,R@RR.from_euler('y',180,degrees=True).as_matrix()];ang=[RR.from_matrix(prev.T@c).magnitude() for c in cands];k=np.argmin(ang);out.append(cands[k]);prev=cands[k]
                                                                                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ …[+562 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sed -i "s/prev=RR.from_matrix(start).as_quat()/prev=RR.from_matrix(start).as_matrix()/" /work/scratch/test_unwrap.py
python /work/scratch/test_unwrap.py'
```
<details><summary>output</summary>

```
95 258 maxstep 46.81118486769821 median 3.785202667050031 start/end angles 1.652078003166389e-14 34.99796011199789
415 745 maxstep 53.53954975707812 median 4.193203697662571 start/end angles 16.252022699216432 104.42419601932133

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];RC=np.load('/work/scratch/proto_poses.npy')[300,:3,:3]
for i in [415,416,417,257,258]:
 R=T[i,:3,:3];cands={'raw':R,'yawlocal':R@RR.from_euler('y',180,degrees=True).as_matrix(),'xlocal':R@RR.from_euler('x',180,degrees=True).as_matrix(),'zlocal':R@RR.from_euler('z',180,degrees=True).as_matrix()}
 print(i,[(k,round(RR.from_matrix(RC.T@v).magnitude()*180/np.pi,1)) for k,v in cands.items()])
PY"
```
<details><summary>output</summary>

```
415 [('raw', 164.3), ('yawlocal', 16.3), ('xlocal', 178.6), ('zlocal', 176.1)]
416 [('raw', 172.3), ('yawlocal', 9.0), ('xlocal', 178.6), ('zlocal', 175.7)]
417 [('raw', 170.7), ('yawlocal', 10.1), ('xlocal', 178.4), ('zlocal', 176.6)]
257 [('raw', 132.6), ('yawlocal', 47.7), ('xlocal', 177.2), ('zlocal', 176.4)]
258 [('raw', 145.0), ('yawlocal', 35.0), ('xlocal', 179.7), ('zlocal', 178.5)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/proto_solve.py /work/outputs/solve.py
python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
# import argparse already there
# replace main signature/root
s=s.replace(\"def main():\\n tstart=time.time();root='/work/inputs';r=np.load(root+'/reconstruction.npz');\", \"def main():\\n ap=argparse.ArgumentParser();ap.add_argument('--inputs',default='/work/inputs');ap.add_argument('--out',default='/work/outputs/object_poses_world.npz');args=ap.parse_args();tstart=time.time();root=args.inputs;r=np.load(root+'/reconstruction.npz');\")
# Add ramp function before main
needle=\"def main():\\n\"
ramp='''def anchor_ramp(Rseq,start_R=None,end_R=None,n=12):
 if len(Rseq)<2:return Rseq
 def ss(x):return x*x*(3-2*x)
 if start_R is not None:
  Rseq[0]=start_R
  for k in range(1,min(n,len(Rseq))):
   Rseq[k]=matrix_slerp(start_R,Rseq[k],ss(k/(n-1)))
 if end_R is not None:
  Rseq[-1]=end_R
  for k in range(1,min(n,len(Rseq))):
   Rseq[-1-k]=matrix_slerp(end_R,Rseq[-1-k],ss(k/(n-1)))
 return Rseq

def main():\\n'''
# But main signature replacement already changed, do insert before final def main
idx=s.rfind(\"def main():\\n\");s=s[:idx]+ramp+s[idx+len(\"def main():\\n\"):]
# robust calls with ramps
s=s.replace(\"RB=robust_orientations(T,95,258,start_R=R0,end_R=RC)\\n RD=robust_orientations(T,415,745,start_R=RC)\", \"RB=robust_orientations(T,95,258,start_R=R0,end_R=RC)\\n RD=robust_orientations(T,415,745,start_R=RC)\\n RB=anchor_ramp(RB,start_R=R0,end_R=RC,n …[+446 chars]
```
<details><summary>output</summary>

```
115:def anchor_ramp(Rseq,start_R=None,end_R=None,n=12):
128:def main():
158: RB=robust_orientations(T,95,258,start_R=R0,end_R=RC)
160: RB=anchor_ramp(RB,start_R=R0,end_R=RC,n=12)
161: RD=anchor_ramp(RD,start_R=RC,n=12)
212: np.savez(args.out,T_world_mesh=Tout.astype(np.float64))

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '108,175p' /work/outputs/solve.py; tail -20 /work/outputs/solve.py"
```
<details><summary>output</summary>

```
  orr,prr=resized_masks(o,p,x0,y0,sz);vals.append((j,orr,prr))
 sizes=np.array([[RES,RES]],np.float32)
 def cost(d):
  t=t0.copy();t[:2]+=d;P=np.eye(4,dtype=np.float32);P[:3,:3]=R;P[:3,3]=t;poses=np.repeat((Tcw@P)[None],len(vals),axis=0)
  mm,_,_=ren.render_batch(poses.astype(np.float32),Kr[None],sizes);mm=mm.cpu().numpy();return 1-np.mean([mask_iou(m,o,p) for m,(_,o,p) in zip(mm,vals)])
 res=differential_evolution(cost,[(-bounds,bounds),(-bounds,bounds)],seed=0,popsize=5,maxiter=9,tol=.005,polish=False,disp=False);r2=minimize(cost,res.x,method='Nelder-Mead',options={'maxiter':80});t=t0.copy();t[:2]+=r2.x;return t,1-r2.fun

def anchor_ramp(Rseq,start_R=None,end_R=None,n=12):
 if len(Rseq)<2:return Rseq
 def ss(x):return x*x*(3-2*x)
 if start_R is not None:
  Rseq[0]=start_R
  for k in rang …[+5643 chars]
```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "/usr/bin/time -f 'wall %E maxrss %MKB' python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "TIMEFORMAT='wall %3R'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
static [0.46459642 0.30967867 3.3481545 ] 0.9375525106885776 [[ 0.9373614  -0.04881942  0.34491833]
 [-0.05988644 -0.99797243  0.02149707]
 [ 0.34317076 -0.04080633 -0.9383857 ]] [0.11866937 1.0606955  3.78332   ] 0.9577977567639505 final [   0.7516895     1.1341983     2.92303422 -120.37643023] 0.8861875049909845
wrote /work/outputs/object_poses_world.npz frames 980 runtime_s 129.6
wall 132.520

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed 's#/work/scratch/proto_poses.npy#/work/outputs/object_poses_world.npz#; s#Tp=np.load(.*proto.*).astype#Tp=np.load(\\\"/work/outputs/object_poses_world.npz\\\")[\\\"T_world_mesh\\\"].astype#' /work/scratch/evaluate.py > /work/scratch/evaluate_final.py
# verify line
grep 'Tp=' /work/scratch/evaluate_final.py
python /work/scratch/evaluate_final.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/evaluate_final.py", line 3, in <module>
    root='/work/inputs';r=np.load(root+'/reconstruction.npz');Tb=r['T_world_mesh'].astype(np.float32);Tp=np.load('/work/outputs/object_poses_world.npz').astype(np.float32);cam=json.load(open(root+'/camera.json'));K=np.array(cam['K'],np.float32);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],np.float32)).astype(np.float32);ren=Renderer(root+'/object_mesh.glb');rows=[]
                                                                                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'NpzFile' object has no attribute 'astype'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/evaluate_final.py';s=open(p).read();s=s.replace(\"Tp=np.load('/work/outputs/object_poses_world.npz').astype(np.float32)\",\"Tp=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(np.float32)\");open(p,'w').write(s)
PY
python /work/scratch/evaluate_final.py"
```
<details><summary>output</summary>

```
A static n 48 baseline 0.874 result 0.933 delta 0.059 improved% 100.0 worse% 0.0
B carry n 82 baseline 0.757 result 0.834 delta 0.077 improved% 89.0 worse% 1.2
C static n 78 baseline 0.783 result 0.901 delta 0.118 improved% 100.0 worse% 0.0
D visible carry n 165 baseline 0.622 result 0.854 delta 0.232 improved% 83.6 worse% 2.4
/work/scratch/evaluate_final.py:13: RuntimeWarning: Mean of empty slice.
  b=a[(a[:,0]>=lo)&(a[:,0]<=hi)];print(name,'n',len(b),'baseline',round(b[:,2].mean(),3),'result',round(b[:,3].mean(),3),'delta',round((b[:,3]-b[:,2]).mean(),3),'improved%',round(100*(b[:,3]>b[:,2]+.01).mean(),1),'worse%',round(100*(b[:,3]<b[:,2]-.01).mean(),1))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret =  …[+815 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];ds=np.degrees(RR.from_matrix(T[1:,:3,:3]@T[:-1,:3,:3].transpose(0,2,1)).magnitude())
for k in np.argsort(ds)[-20:][::-1]:print(k,k+1,ds[k])
PY"
```
<details><summary>output</summary>

```
251 252 137.87653373291434
123 124 56.29098851675625
252 253 30.687763004830565
122 123 30.005246330706452
124 125 29.64337585153364
420 421 24.42143645373985
253 254 24.062897617017807
419 420 23.63113696920354
421 422 23.418626230087696
732 733 22.116309376304812
422 423 20.942840477267044
418 419 20.899835149171636
250 251 18.837736097779885
254 255 18.47165286528527
423 424 16.934485889771413
417 418 16.678683339704552
249 250 16.22416198100559
255 256 14.518129396372531
647 648 14.0505541100057
731 732 13.906499914590176

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];y=RR.from_euler('y',180,degrees=True).as_matrix()
for i in range(245,260):
 R=T[i,:3,:3];print(i,'prev',None if i==245 else round(RR.from_matrix(T[i-1,:3,:3].T@R).magnitude()*180/np.pi,1),'yaw flips prev',None if i==245 else round(RR.from_matrix(T[i-1,:3,:3].T@(R@y)).magnitude()*180/np.pi,1))
PY"
```
<details><summary>output</summary>

```
245 prev None yaw flips prev None
246 prev 0.0 yaw flips prev 180.0
247 prev 0.0 yaw flips prev 180.0
248 prev 4.0 yaw flips prev 176.0
249 prev 10.9 yaw flips prev 169.1
250 prev 16.2 yaw flips prev 163.8
251 prev 18.8 yaw flips prev 161.2
252 prev 137.9 yaw flips prev 42.2
253 prev 30.7 yaw flips prev 149.3
254 prev 24.1 yaw flips prev 155.9
255 prev 18.5 yaw flips prev 161.5
256 prev 14.5 yaw flips prev 165.5
257 prev 9.6 yaw flips prev 170.4
258 prev 3.4 yaw flips prev 176.6
259 prev 0.0 yaw flips prev 180.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for i in [250,251,252,253,254]:print(i,RR.from_matrix(T[251,:3,:3].T@T[i,:3,:3]).magnitude()*180/np.pi)
print('251-253',RR.from_matrix(T[251,:3,:3].T@T[253,:3,:3]).magnitude()*180/np.pi)
PY"
```
<details><summary>output</summary>

```
250 18.83773609777988
251 1.115008564079626e-14
252 137.8765337329143
253 168.55676408827713
254 167.38047386378838
251-253 168.55676408827713

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];RC=T[259,:3,:3]
for i in [235,240,245,248,250,251]:print(i,RR.from_matrix(T[i,:3,:3].T@RC).magnitude()*180/np.pi)
PY"
```
<details><summary>output</summary>

```
235 171.43794292863166
240 171.43794292863166
245 171.43794292863166
248 167.44502177557138
250 140.27707280561268
251 121.44034973014163

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as RR
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];RC=T[259,:3,:3];y=RR.from_euler('y',180,degrees=True).as_matrix()
for i in [235,240,245,248,250,251,252,258]:print(i,'raw',round(RR.from_matrix(T[i,:3,:3].T@RC).magnitude()*180/np.pi,1),'flip',round(RR.from_matrix((T[i,:3,:3]@y).T@RC).magnitude()*180/np.pi,1))
PY"
```
<details><summary>output</summary>

```
235 raw 171.4 flip 9.5
240 raw 171.4 flip 9.5
245 raw 171.4 flip 9.5
248 raw 167.4 flip 13.2
250 raw 140.3 flip 39.9
251 raw 121.4 flip 58.7
252 raw 100.7 flip 79.4
258 raw 0.0 flip 180.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
a=np.load('/work/scratch/eval_rows.npy') # this was from final eval
for i in range(230,259):
 k=np.where(a[:,0]==i)[0]
 # odd not in stride2; check
 if len(k):print(i,a[k[0],2],a[k[0],3])
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
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];d=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)
for k in np.argsort(d)[-20:][::-1]:print(k,k+1,d[k]*100)
print('percentiles cm',np.percentile(d*100,[50,75,90,95,99,100]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
# add vec ramp after anchor_ramp
needle=\"def main():\\n\"
insert='''def vec_anchor_ramp(P,start=None,end=None,n=10):
 P=np.array(P,copy=True)
 if len(P)<2:return P
 def ss(x):return x*x*(3-2*x)
 if start is not None:
  P[0]=start
  for k in range(1,min(n,len(P))):
   s=ss(k/(n-1));P[k]=(1-s)*np.asarray(start)+s*P[k]
 if end is not None:
  P[-1]=end
  for k in range(1,min(n,len(P))):
   s=ss(k/(n-1));P[-1-k]=(1-s)*np.asarray(end)+s*P[-1-k]
 return P

'''
idx=s.rfind(needle);s=s[:idx]+insert+s[idx:]
# after RB/RD ramps add B ending bridge (last 12) and temporal translation smoothing/ramps after tB/tD loop
s=s.replace(\"RD=anchor_ramp(RD,start_R=RC,n=12)\\n # hand frames\", \"RD=anchor_ramp(RD,start_R=RC,n=12)\\n # The baseline flips to the opposite orientation branch during the final placement; bridge the last 12 frames smoothly to the static pose instead of accepting a one-frame 138 degree jump.\\n nbridge=12\\n for k in range(len(RB)-nbridge,len(RB)):\\n  sfrac=(k-(len(RB)-nbridge))/(nbridge-1);sfrac=sfrac*sfrac*(3-2*sfrac);RB[k]=matrix_slerp(RB[len(RB)-nbridge],RC,sfrac)\\n # hand frames\")
s=s.replace(\"for j,i in enumerate(range(415,746)):tD[j]=align_xy(ren,Tcw,K,objm[i],perm[i],RD[j],T[i,:3,3])\\n # hand-anchored\", \"for j,i in enumerate(range(415,746)):tD[j]=align_xy(ren,Tcw,K,objm[i],perm[i],RD[j],T[i,:3,3])\\n # Light temporal smoothing removes per-frame mask-centroid jitter; hard anchors k …[+537 chars]
```
<details><summary>output</summary>

```
177: nbridge=12
178: for k in range(len(RB)-nbridge,len(RB)):
179:  sfrac=(k-(len(RB)-nbridge))/(nbridge-1);sfrac=sfrac*sfrac*(3-2*sfrac);RB[k]=matrix_slerp(RB[len(RB)-nbridge],RC,sfrac)
207: tB=gaussian_filter1d(tB,1.0,axis=0);tD=gaussian_filter1d(tD,1.0,axis=0)
218: pred=np.array(pred);pred=gaussian_filter1d(pred,2.0,axis=0);pred=vec_anchor_ramp(pred,start=tD[-1],n=10)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "TIMEFORMAT='wall %3R'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
static [0.46459642 0.30967867 3.3481545 ] 0.9375525106885776 [[ 0.9373614  -0.04881942  0.34491833]
 [-0.05988644 -0.99797243  0.02149707]
 [ 0.34317076 -0.04080633 -0.9383857 ]] [0.11866937 1.0606955  3.78332   ] 0.9577977567639505 final [   0.7516895     1.1341983     2.92303422 -120.37643023] 0.8861875049909845
wrote /work/outputs/object_poses_world.npz frames 980 runtime_s 135.6
wall 138.035

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/evaluate_final.py'
```
<details><summary>output</summary>

```
A static n 48 baseline 0.874 result 0.933 delta 0.059 improved% 100.0 worse% 0.0
B carry n 82 baseline 0.757 result 0.835 delta 0.077 improved% 89.0 worse% 1.2
C static n 78 baseline 0.783 result 0.901 delta 0.118 improved% 100.0 worse% 0.0
D visible carry n 165 baseline 0.622 result 0.856 delta 0.233 improved% 85.5 worse% 3.0
/work/scratch/evaluate_final.py:13: RuntimeWarning: Mean of empty slice.
  b=a[(a[:,0]>=lo)&(a[:,0]<=hi)];print(name,'n',len(b),'baseline',round(b[:,2].mean(),3),'result',round(b[:,3].mean(),3),'delta',round((b[:,3]-b[:,2]).mean(),3),'improved%',round(100*(b[:,3]>b[:,2]+.01).mean(),1),'worse%',round(100*(b[:,3]<b[:,2]-.01).mean(),1))
/opt/conda/lib/python3.11/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
  ret =  …[+812 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];d=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)
print(np.percentile(d*100,[50,75,90,95,99,100]));
for k in np.argsort(d)[-10:][::-1]:print(k,k+1,d[k]*100)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as RR
r=np.load('/work/inputs/reconstruction.npz');Tb=r['T_world_mesh'];Tp=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];K=r['human_keypoints_world'];pl=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane']);n=pl[:3];d=pl[3]
import trimesh
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=np.c_[m.vertices,np.ones(len(m.vertices))]
def floor(i,T):
 W=(T[i]@V.T).T[:,:3];return (W@n+d).min(),(W@n+d).max()
for i in [300,400,850,860,920,979]:
 print('frame',i,'base dist',floor(i,Tb),'res',floor(i,Tp))
# hand distance carried occlusion
for name,T in [('base',Tb),('res',Tp)]:
 ds=[]
 for i in range(746,846):
  h=K[i,42:63].mean(0);ds.append(np.linalg.norm(T[i,:3,3]-h))
 print(name,'occlusion hand dist cm median/p90/max',np.percentile(np.array(ds)*100,[50,90,100]))
# static translation changes
for lo,hi,name in [(0,94,'A'),(259,414,'C'),(846,979,'E')]:
 print(name,'base med',np.median(Tb[lo:hi+1,:3,3],0),'res',Tp[(lo+hi)//2,:3,3],'delta cm',(Tp[(lo+hi)//2,:3,3]-np.median(Tb[lo:hi+1,:3,3],0))*100)
# file info
a=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];print(a.shape,a.dtype,np.isfinite(a).all(),np.max(abs(a[:,3]-[0,0,0,1])),np.linalg.det(a[:,:3,:3]).min(),np.linalg.det(a[:,:3,:3]).max())
# saved metrics
b=np.load('/work/scratch/eval_rows.npy');print('eval',b.shape)
PY"
```
<details><summary>output</summary>

```
frame 300 base dist (0.01196313358061274, 0.17597076010079205) res (0.04497476924528265, 0.20736232387927456)
frame 400 base dist (0.03846284100499453, 0.20247046752517361) res (0.04497476924528265, 0.20736232387927456)
frame 850 base dist (-0.04037999377989254, 0.2044415761204854) res (0.0027611954671040717, 0.1796099856493043)
frame 860 base dist (-0.037025146550502264, 0.20779307520710844) res (0.0027611954671040717, 0.1796099856493043)
frame 920 base dist (0.054287522595313886, 0.3091830877885968) res (0.0027611954671040717, 0.1796099856493043)
frame 979 base dist (0.09772334320762388, 0.28527171746533364) res (0.0027611954671040717, 0.1796099856493043)
base occlusion hand dist cm median/p90/max [110.11966324 222.24414825 233.82434082]
res occlusion hand dist cm median/p90/max [ 9.7986 …[+512 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import importlib.metadata as m,sys,scipy,numpy,cv2,h5py,trimesh,torch
print(sys.version.split()[0])
for p in ['numpy','scipy','opencv-python','opencv-python-headless','h5py','trimesh','torch','nvdiffrast','pyrender','Pillow']:
 try: print(p,m.version(p))
 except Exception as e: print(p,'(not installed)' if p"'!='"'nvdiffrast' else getattr(__import__('nvdiffrast'),'__version__','unknown'))
PY
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1"
```
<details><summary>output</summary>

```
3.11.10
numpy 1.26.3
scipy 1.16.3
opencv-python 4.11.0.86
opencv-python-headless (not installed)
h5py 3.16.0
trimesh 5.1.1
torch 2.5.1+cu124
nvdiffrast 0.4.0
pyrender 0.1.45
Pillow 10.2.0
NVIDIA RTX 6000 Ada Generation, 49140 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Monocular 6-DoF Basket Trajectory Recovery

## Result

"'`object_poses_world.npz` contains `T_world_mesh` with shape `(980, 4, 4)`, dtype
`float64`. All matrices are finite rigid transforms and map the original GLB mesh
frame to the released world frame.

## Recovered phases

The frame boundaries below were selected from the object masks, the hand/object
contact signal, and visual inspection of the full-resolution video. They are hard
coded in `solve.py` because they are episode-level decisions, not per-frame pose
values.

| Phase | Frames | Description | Treatment |
|---|---:|---|---|
| A | 0-94 | Basket at rest on the round table | Constant pose, upright orientation |
| B | 95-258 | First right-hand pick-up, carry, and placement | Per-frame translation alignment, robust SO(3) smoothing |
| C | 259-414 | Basket at rest on the floor | Constant upright pose fitted to all static observations |
| D1 | 415-745 | Second pick-up and manipulation, basket still visible | Per-frame mask alignment, robust temporal rotation filtering |
| D2 | 746-845 | Basket carried/placed while fully hidden by the person | Hand-attached translation and SO(3) propagation between visible anchors |
| E | 846-979 | Basket released and stationary | One constant final pose for all 134 frames |

The two long mask gaps are exact and data-driven: frames **746-846** and
**868-979** have zero object-mask pixels. Frame 846 is included in phase E;
frames 847 …[+7250 chars]
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 6.9K Oct 20 20:02 /work/outputs/REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 1.6K Oct 20 20:02 /work/outputs/TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 123K Oct 20 20:00 /work/outputs/object_poses_world.npz
-rw-rw-r-- 1 hanzhou hanzhou  15K Oct 20 19:58 /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n '/work/scratch\\|/pipeline\\|/weights\\|http' /work/outputs/solve.py || true; grep -n 'xc0\\|fitC\\|fitframes\\|off=' /work/outputs/solve.py"
```
<details><summary>output</summary>

```
155: fitC=[290,310,330,350,370,390,410]
158:  for i in fitC:
160: xc0,yc0,szc=690,710,105;Krc=crop_K(K,xc0,yc0,szc);valsC=[resized_masks(objC[i],perC[i],xc0,yc0,szc) for i in fitC]
164:  if i in fitC:imgsC[i]=cv2.resize(im[yc0:yc0+szc,xc0:xc0+szc],(RES,RES),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.float32)/255.
168:   ious.append(mask_iou(mm[j],o,p));msk=o&(rgb[j].max(2)>.05)&~p;a=imgsC[fitC[j]][msk].mean(1);b=rgb[j][msk].mean(1);ccs.append(np.corrcoef(a,b)[0,1] if msk.sum()>20 and a.std()>1e-4 and b.std()>1e-4 else 0)
184: fitframes=list(range(849,862));fitpairs=[(i,objm[i],perm[i]) for i in fitframes];x0,y0,sz,Kr=835,790,180,crop_K(K,835,790,180);vals=[]
190:  if i in fitframes:imgs[i]=cv2.resize(im[y0:y0+sz,x0:x0+sz],(RES,RES),interpolation=cv2.INTER_AREA)[:,:,::-1].astype(np.f …[+193 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"ren=Renderer(root+'/object_mesh.glb');R0=T[0,:3,:3].copy()\\n # load selected masks\", \"ren=Renderer(root+'/object_mesh.glb');R0=T[0,:3,:3].copy()\\n # Episode-level phase decisions, verified from full-resolution video, object/person masks,\\n # and the contact signal. A/C/E are periods in which the object is at rest.\\n A_END=94; B_START=95; B_END=258; C_START=259; C_END=414; D_START=415; D_END=845; E_START=846\\n # load selected masks\")
s=s.replace(\" # ---------- static C: upright on floor after first placement ----------\\n fitC=[290,310,330,350,370,390,410]\", \" # ---------- static C: upright on floor after first placement ----------\\n # These are stable, unoccluded observations from the visually verified rest interval.\\n fitC=[290,310,330,350,370,390,410]\")
s=s.replace(\" xc0,yc0,szc=690,710,105;Krc=crop_K\", \" # Fixed crop encloses every phase-C basket mask and render while remaining small enough to resolve the object.\\n xc0,yc0,szc=690,710,105;Krc=crop_K\")
s=s.replace(\" # final fit: upright rotation, translation constrained near floor using visible frames\\n fitframes=list(range(849,862));fitpairs=\", \" # final fit: upright rotation, translation constrained near floor using visible frames.\\n # Frames 849-861 are the stable part of the short 847-867 window in which SAM still sees the released basket.\\n fitframes=list(range(849,862));fitpairs=\")
s=s.replace(\" x0,y0 …[+386 chars]
```
<details><summary>output</summary>

```
145: # Episode-level phase decisions, verified from full-resolution video, object/person masks,
158: # These are stable, unoccluded observations from the visually verified rest interval.
164: # Fixed crop encloses every phase-C basket mask and render while remaining small enough to resolve the object.
189: # Frames 849-861 are the stable part of the short 847-867 window in which SAM still sees the released basket.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '186,193p' /work/outputs/solve.py"
```
<details><summary>output</summary>

```
 HR=hand_frames(r['human_keypoints_world'],np.arange(21,42));HL=hand_frames(r['human_keypoints_world'],np.arange(42,63))
 Hfin=HL
 # final fit: upright rotation, translation constrained near floor using visible frames.
 # Frames 849-861 are the stable part of the short 847-867 window in which SAM still sees the released basket.
 fitframes=list(range(849,862));fitpairs=[(i,objm[i],perm[i]) for i in fitframes];x0,y0,sz,Kr=835,790,180,crop_K(K,835,790,180);vals=[]
 for i,o,p in fitpairs:vals.append((*resized_masks(o,p,x0,y0,sz),))
 # actual color crops for texture term
 imgs={};cap=cv2.VideoCapture(root+'/video.mp4')

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"fitframes=list(range(849,862));fitpairs=\", \" # Fixed 180 px crop centered on the visible final basket; chosen by eye from frames 847-867.\\n fitframes=list(range(849,862));fitpairs=\");open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sha256sum /work/outputs/object_poses_world.npz > /work/scratch/hash_before.txt
TIMEFORMAT='wall %3R'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
sha256sum /work/outputs/object_poses_world.npz > /work/scratch/hash_after.txt
cat /work/scratch/hash_before.txt /work/scratch/hash_after.txt"
```
<details><summary>output</summary>

```
static [0.46459642 0.30967867 3.3481545 ] 0.9375525106885776 [[ 0.9373614  -0.04881942  0.34491833]
 [-0.05988644 -0.99797243  0.02149707]
 [ 0.34317076 -0.04080633 -0.9383857 ]] [0.11866937 1.0606955  3.78332   ] 0.9577977567639505 final [   0.7516895     1.1341983     2.92303422 -120.37643023] 0.8861875049909845
wrote /work/outputs/object_poses_world.npz frames 980 runtime_s 126.1
wall 128.483
e8691e6306fa7561014798b1ae4af28c47c909916e634963df5259b482f11002  /work/outputs/object_poses_world.npz
e8691e6306fa7561014798b1ae4af28c47c909916e634963df5259b482f11002  /work/outputs/object_poses_world.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read();s=s.replace(\"**135.6 s** internal (**138.0 s** wall)\",\"**126.1 s** internal (**128.5 s** wall)\");open(p,'w').write(s)
PY
tail -8 /work/outputs/REPORT.md
python - <<'PY'
import numpy as np,os
p='/work/outputs/object_poses_world.npz';a=np.load(p);print(a.files);T=a['T_world_mesh'];print(T.shape,T.dtype,np.isfinite(T).all(),T[0],T[-1],os.path.getsize(p))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import importlib.metadata as m
for p in ['matplotlib','open3d']:
 print(p,m.version(p))
PY"
```
<details><summary>output</summary>

```
matplotlib 3.11.2
open3d 0.20.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/| Matplotlib | installed | Diagnostic plots only |/| Matplotlib | 3.11.2 | Diagnostic plots only |/' /work/outputs/TOOLS.md
grep -n 'Matplotlib\\|Open3D' /work/outputs/TOOLS.md"
```
<details><summary>output</summary>

```
26:| Open3D | 0.20.0 | Environment availability check only |
27:| Matplotlib | 3.11.2 | Diagnostic plots only |

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/outputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort && echo '--- validation summary' && python - <<'PY'
import numpy as np
b=np.load('/work/scratch/eval_rows.npy');print('visible-frame rendering IoU:',round(b[:,2].mean(),3),'->',round(b[:,3].mean(),3),'over',len(b),'samples')
PY
echo '--- output checksum' && sha256sum /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
visible-frame rendering IoU: 0.71 -> 0.868 over 383 samples
--- output checksum
e8691e6306fa7561014798b1ae4af28c47c909916e634963df5259b482f11002  /work/outputs/object_poses_world.npz

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, `(980, 4, 4)`, `float64`
- `/work/outputs/solve.py` — deterministic offline reconstruction script
- `/work/outputs/REPORT.md` — phases, methodology, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — dependencies and tools used

Validation on 383 sampled visible frames improved rendered-mask IoU from **0.710 to 0.868**. Static phases are now constant, and the unobservable hand-carried segment has median object-to-hand distance reduced from **110.1 cm to 9.8 cm**.

Final solver runtime was **126.1 seconds**. Re-running produced an identical output checksum:

`e8691e6306fa7561014798b1ae4af28c47c909916e634963df5259b482f11002`
