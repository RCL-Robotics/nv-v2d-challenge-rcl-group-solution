# T1_ds_ep006

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T16:11:36Z

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
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '1,80p'"
```
<details><summary>output</summary>

```
     1	# Task: recover the full 6-DoF object trajectory of one monocular human-object-interaction video
     2	
     3	You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
     4	put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
     5	`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.
     6	
     7	## Goal
     8	Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
     9	possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
    10	your pose and under the true pose; no symmetry handling, no alignment), ave …[+6675 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '81,160p'"
```
<details><summary>output</summary>

```
    81	    tracking, and its refinement by the CARI4D network), mesh-file frame, for `K_pipeline`. Their depth scale is not
    82	    calibrated to the world; `T_world_mesh` already includes a per-frame depth correction.
    83	  - `human_vertices_world` (F,V,3, every 4th vertex), `human_keypoints_world` (F,70,3; MHR keypoints, right hand
    84	    21-41, left hand 42-62) of the reconstructed human in world; `contact_prob_right_left` (F,2).
    85	- `masks.h5` - `object` and `person` (F,1152,1536) boolean SAM2 masks (may be imperfect; you may recompute them).
    86	
    87	## Tools you may use
    88	- This container's software (Python, torch + CUDA, nvdiffrast, opencv, trimesh, open3d, ...). The CARI4D pipeline
    89	  modules are installed under `/workspace` (v2d_sam2, v2d_moge, v2d_ …[+3157 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 1 -type f -printf '%f %s bytes\\n' | sort && printf '\\nPipeline dirs:\\n' && find /pipeline -maxdepth 2 -type f | head -50"
```
<details><summary>output</summary>

```
camera.json 825 bytes
episode.json 61 bytes
frame0_reference_human.npz 193793 bytes
ground_plane.json 400 bytes
masks.h5 5055381 bytes
object_mesh.glb 10935340 bytes
reconstruction.npz 27418335 bytes
task.txt 255 bytes
video.mp4 15484106 bytes

Pipeline dirs:
/pipeline/.stages/03_human_initialization.json
/pipeline/.stages/01_depth.json
/pipeline/.stages/02_export.json
/pipeline/.stages/04_depth_alignment.json
/pipeline/.stages/05_object_pose.json
/pipeline/.stages/06_coconet.json
/pipeline/.stages/07_refinement.json
/pipeline/.stages/08_render.json
/pipeline/mhr_init/episode_000006.stats.pkl
/pipeline/mhr_init/episode_000006.sam3d.h5
/pipeline/mhr_init/episode_000006.pkl
/pipeline/foundationpose/episode_000006.pkl
/pipeline/inference/.materialized_inputs.h5.lock
/pipeline/inference/refine …[+286 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat inputs/episode.json; echo; cat inputs/task.txt; echo; cat inputs/camera.json; echo; cat inputs/ground_plane.json'
```
<details><summary>output</summary>

```
{
 "episode_index": 6,
 "object": "monitor",
 "frames": 546
}
Walk over to the monitor on the round table. Using only the base of the monitor, rotate and reposition the monitor around the surface of the table without touching or moving the screen itself. When finished, leave the monitor resting on top of the table.

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
   817.865478515625,
   0.0,
   768.0
  ],
  [
   0.0,
   817.865478515625,
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
   0.9979089832158867,
   0.003886161652594123,
   0.06451789646791505,
   -0.046856469271318854
  ],
  [
   -0.008000604419330558,
   0.9979410521065133,
   0.0 …[+744 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1911
/pipeline/.stages/02_export.json 2132
/pipeline/.stages/03_human_initialization.json 2295
/pipeline/.stages/04_depth_alignment.json 2057
/pipeline/.stages/05_object_pose.json 1898
/pipeline/.stages/06_coconet.json 3080
/pipeline/.stages/07_refinement.json 2299
/pipeline/.stages/08_render.json 2820
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 678400800
/pipeline/depth/moge2/depth_report.json 1877
/pipeline/depth/moge2/intrinsics.pkl 58274
/pipeline/depth/moge2/raw.npy 1932263552
/pipeline/export/episode_000006/edex 561
/pipeline/export/episode_000006/wild_export.json 2876
/pipeline/foundationpose/episode_000006.pkl 158381
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 4177536 …[+480 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/pipeline_report.json; echo; for f in /pipeline/.stages/*.json; do echo ==== $f; cat "$f"; done'
```
<details><summary>output</summary>

```
{
  "name": "07_refinement",
  "outputs": [
    {
      "mtime_ns": 1792061522558932754,
      "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000006/inference/refined.pth",
      "size": 417801696
    },
    {
      "mtime_ns": 1792061515115932577,
      "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000006/inference/refinement_checkpoint.pth",
      "size": 1724642
    }
  ],
  "schema": "v2d.cari4d.stage.v1",
  "signature": {
    "command": [
      "/opt/conda/bin/python",
      "-m",
      "learning.training.mhr_opt_refineout",
      "--bundle",
      "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000006/inference/coconet.pth",
      "--object-mesh",
      "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episo …[+4357 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json, trimesh, os
p='inputs/reconstruction.npz'
z=np.load(p)
for k in z.files:
 print(k, z[k].shape, z[k].dtype)
with h5py.File('inputs/masks.h5','r') as f:
 for k in f.keys(): print('mask',k,f[k].shape,f[k].dtype)
print('mesh load...')
s=trimesh.load('inputs/object_mesh.glb', force='scene', process=False)
g=s.to_geometry(); print(type(g), len(g.vertices),len(g.faces),g.bounds, g.extents, g.is_watertight)
PY"
```
<details><summary>output</summary>

```
T_world_mesh (546, 4, 4) float32
T_cam_mesh_foundationpose (546, 4, 4) float32
T_cam_mesh_coconet (546, 4, 4) float32
human_vertices_world (546, 4610, 3) float32
human_keypoints_world (546, 70, 3) float32
contact_prob_right_left (546, 2) float32
mask object (546, 1152, 1536) bool
mask person (546, 1152, 1536) bool
mesh load...
<class 'trimesh.base.Trimesh'> 294042 98048 [[-0.25416997 -0.08496192 -0.26776227]
 [ 0.26815489  0.1019434   0.27105448]] [0.52232486 0.18690532 0.53881675] False

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "mkdir -p /work/scratch /work/outputs
cat > /work/scratch/analyze.py <<'PY'
import numpy as np, h5py, json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh'].astype(float); F=len(T)
with h5py.File('/work/inputs/masks.h5','r') as h:
 m=h['object'][:]; p=h['person'][:]
for name,mask in [('obj',m),('person',p)]:
 area=mask.sum((1,2)); ys,xs=np.where(mask)
 print(name,'area min/med/max',area.min(),np.median(area),area.max(),'union bbox',xs.min(),xs.max(),ys.min(),ys.max())
# mask centroid and bbox each frame
for i in list(range(0,F,10))+[F-1]:
 ys,xs=np.where(m[i]);
 print(i,'area',m[i].sum(),'bbox', (xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None,'cent',(xs.mean(),ys.mean()) if len(xs) else None,'parea',p[i].sum(),'inter',np.logical_and(m[i],p[i]).sum())
# baseline translations, rotation magnitude deltas
trans=T[:,:3,3]; q=R.from_matrix(T[:,:3,:3]).as_quat()
print('trans xyz range',trans.min(0),trans.max(0),'median',np.median(trans,0))
for lag in [1,2,5,10,20]:
 dt=np.linalg.norm(trans[lag:]-trans[:-lag],axis=1)
 dr=np.linalg.norm((R.from_matrix(T[lag:,:3,:3])*R.from_matrix(T[:-lag,:3,:3]).inv()).as_rotvec(),axis=1)
 print('lag',lag,'dt med/p95/max',np.median(dt),np.percentile(dt,95),dt.max(),'dr deg med/p95/max',np.degrees(np.median(dr)),np.degrees(np.percentile(dr,95)),np.degrees(dr.max()))
# top jumps
for lag in [1,5]:
 dt=np.linalg.norm(trans[lag:]-trans[:-lag],axis=1)
 dr=np.degrees( …[+919 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/analyze.py", line 8, in <module>
    area=mask.sum((1,2)); ys,xs=np.where(mask)
                          ^^^^^
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/analyze.py'
s=open(p).read().replace(\"area=mask.sum((1,2)); ys,xs=np.where(mask)\",\"area=mask.sum((1,2)); ys,xs=np.where(mask.any(0))\")
open(p,'w').write(s)
PY
python /work/scratch/analyze.py"
```
<details><summary>output</summary>

```
obj area min/med/max 4482 13635.0 21873 union bbox 748 1000 492 681
person area min/med/max 14576 32365.0 52749 union bbox 594 1101 386 928
0 area 15642 bbox (757, 499, 890, 681) cent (820.6092571282445, 571.8269402889656) parea 34312 inter 0
10 area 15704 bbox (757, 499, 890, 681) cent (820.6763245033112, 571.8479368313806) parea 34546 inter 0
20 area 15774 bbox (757, 499, 890, 681) cent (820.6425129960695, 572.1366806136681) parea 35541 inter 0
30 area 15680 bbox (757, 499, 889, 681) cent (820.7162627551021, 571.781887755102) parea 36429 inter 0
40 area 15564 bbox (757, 499, 890, 681) cent (820.6906964790543, 571.2022616294012) parea 38283 inter 0
50 area 15533 bbox (757, 499, 890, 681) cent (820.7558745895834, 570.974827786004) parea 38259 inter 0
60 area 15158 bbox (757, 499, 889, 680) …[+7464 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/contact.py <<'PY'
import cv2, numpy as np, h5py
from PIL import Image,ImageDraw,ImageFont
cap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]
while True:
 ok,fr=cap.read()
 if not ok: break
 frames.append(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
print('frames',len(frames))
with h5py.File('/work/inputs/masks.h5','r') as h:
 om=h['object'][:]; pm=h['person'][:]
idxs=list(range(0,len(frames),10))+[len(frames)-1]
thumbs=[]
for i in idxs:
 im=frames[i].copy()
 ov=im.copy(); ov[pm[i]]=(0.35*ov[pm[i]]+0.65*np.array([0,150,255])).astype(np.uint8); ov[om[i]]=(0.4*ov[om[i]]+0.6*np.array([255,30,30])).astype(np.uint8)
 im=cv2.addWeighted(im,.55,ov,.45,0)
 # crop broad region
 crop=im[350:750,550:1100]
 crop=cv2.resize(crop,(550,400))
 pil=Image.fromarray(crop); d=ImageDraw.Draw(pil); d.rectangle((0,0,70,25),fill='black'); d.text((5,4),str(i),fill='white')
 thumbs.append(pil)
cols=6; rows=(len(thumbs)+cols-1)//cols
out=Image.new('RGB',(cols*550,rows*400),'black')
for j,p in enumerate(thumbs): out.paste(p,((j%cols)*550,(j//cols)*400))
out.save('/work/scratch/contact_masks.jpg',quality=92)
# raw contact sheet only every 10
out2=Image.new('RGB',(cols*384,rows*288))
for j,i in enumerate(idxs):
 im=cv2.resize(frames[i],(384,288)); p=Image.fromarray(im); d=ImageDraw.Draw(p); d.rectangle((0,0,70,25),fill='black');d.text((5,4),str(i),fill='white');out2.paste(p,((j%cols)*384,(j//cols)*288))
out2.save('/work/scratch/contact_raw.jpg',quality=90)
PY
python /wor …[+21 chars]
```
<details><summary>output</summary>

```
frames 546

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 3 -type f | head -100; echo '---'; find /workspace -maxdepth 2 -type d | head -100"
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
/code/reconstructio …[+5419 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, pprint
p='/pipeline/foundationpose/episode_000006.pkl'
x=pickle.load(open(p,'rb'))
print(type(x))
if isinstance(x,dict):
 for k,v in x.items():
  try: print(k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None),v if getattr(v,'size',0)<20 else '')
  except: print(k,type(v))
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
frames <class 'list'> None None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029', '000030', '000031', '000032', '000033', '000034', '000035', '000036', '000037', '000038', '000039', '000040', '000041', '000042', '000043', '000044', '000045', '000046', '000047', '000048', '000049', '000050', '000051', '000052', '000053', '000054', '000055', '000056', '000057', '000058', '000059', '000060', '000061', '000062', '000063', '000064', '000065', '000066', '000067', '000068', '000069', '000070', '000071', '000072', '000073', '000074', '0 …[+13829 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['nvdiffrast','pyrender','open3d','torch','trimesh','cv2','scipy','skimage']
for m in mods:
 try:
  x=__import__(m); print(m,getattr(x,'__version__','ok'))
 except Exception as e: print(m,'ERR',e)
PY"
```
<details><summary>output</summary>

```
pyrender 0.1.45
open3d 0.20.0
torch 2.5.1+cu124
trimesh 5.1.1
cv2 4.11.0
scipy 1.16.3
skimage 0.26.0

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, numpy as np
x=pickle.load(open('/pipeline/foundationpose/episode_000006.pkl','rb'))
for k,v in x.items():
 a=np.asarray(v)
 if a.ndim==0 or a.size<20: print(k,repr(v)[:1000])
 else: print(k,a.shape,a.dtype,'minmedmax',np.nanmin(a),np.nanmedian(a),np.nanmax(a))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/nanfunctions.py", line 343, in nanmin
    res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
numpy.core._exceptions._UFuncNoLoopError: ufunc 'fmin' did not contain a loop with signature matching types (dtype('<U6'), dtype('<U6')) -> None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle, numpy as np
x=pickle.load(open('/pipeline/foundationpose/episode_000006.pkl','rb'))
for k,v in x.items():
 try:a=np.asarray(v)
 except: print(k,type(v));continue
 if np.issubdtype(a.dtype,np.number) and a.size>=20: print(k,a.shape,a.dtype,'minmedmax',np.nanmin(a),np.nanmedian(a),np.nanmax(a))
 elif a.size<20: print(k,repr(v)[:1000])
 else: print(k,a.shape,a.dtype)
PY"
```
<details><summary>output</summary>

```
<stdin>:6: RuntimeWarning: All-NaN slice encountered
frames (546,) <U6
camera_id 0
fp_poses_cam (546, 1, 4, 4) float32 minmedmax -0.99997574 0.0 5.928474
fp_candidate_count (546,) int32 minmedmax 1 1.0 252
fp_selected_candidate_index (546,) int32 minmedmax -1 -1.0 192
fp_selected_iou (546,) float32 minmedmax 0.04255319 0.7565594 0.8862336
fp_selected_visibility (546,) float32 minmedmax 0.6470359 0.972574 1.0
fp_retry_count (546,) int32 minmedmax 0 0.0 5
fp_reliable (546,) bool
fp_oracle_rotation_error_deg (546,) float32 minmedmax nan nan nan
fp_translation_threshold_m (546,) float32 minmedmax 0.2 0.2 0.2
fp_selected_translation_distance_m (546,) float32 minmedmax 0.0009157059 0.013156987 2.7922385
gt_object_visibility_ratio (546,) float32 minmedmax nan nan nan
gt_object_visibility_valid (5 …[+2286 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); C=z['T_cam_mesh_coconet'].astype(float); F=z['T_cam_mesh_foundationpose'].astype(float)
X=T@np.linalg.inv(C)
print('T @ inv(C) transl minmax',X[:,:3,3].min(0),X[:,:3,3].max(0)); print('rot rel vec std deg',np.degrees(np.std(R.from_matrix(X[:,:3,:3]).as_rotvec(axis=1),axis=0)))
print(X[0]); print(X[100]); print(X[-1])
# pose angle/trans each 5 around motion
def mat_angle(A,B): return np.degrees(np.linalg.norm((R.from_matrix(A)*R.from_matrix(B).inv()).as_rotvec()))
for i in range(0,546,5):
 if 140<=i<=460:
  print(i,'Txyz',np.round(T[i,:3,3],3),'Cxyz',np.round(C[i,:3,3],3),'Fxyz',np.round(F[i,:3,3],3),'T-C rot',round(mat_angle(T[i,:3,:3],C[i,:3,:3]),2),'C-F rot',round(mat_angle(C[i,:3,:3],F[i,:3,:3]),2))
PY"
```
<details><summary>output</summary>

```
T @ inv(C) transl minmax [-0.05030891 -0.1253815  -0.00622162] [-0.01348884 -0.110992    0.13176005]
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
  File "scipy/spatial/transform/_rotation.pyx", line 1895, in scipy.spatial.transform._rotation.Rotation.as_rotvec
TypeError: as_rotvec() got an unexpected keyword argument 'axis'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh'].astype(float); C=z['T_cam_mesh_coconet'].astype(float); F=z['T_cam_mesh_foundationpose'].astype(float)
X=T@np.linalg.inv(C)
print('T @ inv(C) transl minmax',X[:,:3,3].min(0),X[:,:3,3].max(0)); print('rot rel vec mean std deg',np.degrees(np.mean(R.from_matrix(X[:,:3,:3]).as_rotvec(),0)),np.degrees(np.std(R.from_matrix(X[:,:3,:3]).as_rotvec(),0)))
print(X[0]); print(X[100]); print(X[-1])
def mat_angle(A,B): return np.degrees(np.linalg.norm((R.from_matrix(A)*R.from_matrix(B).inv()).as_rotvec()))
for i in range(0,546,5):
 if 140<=i<=460:
  print(i,'Txyz',np.round(T[i,:3,3],3),'Cxyz',np.round(C[i,:3,3],3),'Fxyz',np.round(F[i,:3,3],3),'T-C rot',round(mat_angle(T[i,:3,:3],C[i,:3,:3]),2),'C-F rot',round(mat_angle(C[i,:3,:3],F[i,:3,:3]),2))
PY"
```
<details><summary>output</summary>

```
T @ inv(C) transl minmax [-0.05030891 -0.1253815  -0.00622162] [-0.01348884 -0.110992    0.13176005]
rot rel vec mean std deg [-4.71122599  4.35385987 -0.19233408] [0.02272196 0.02024533 0.066996  ]
[[ 0.99707936  0.00136173  0.07635996 -0.0500467 ]
 [-0.00766101  0.99658131  0.08226246 -0.1253168 ]
 [-0.07598686 -0.0826072   0.99368103 -0.00466417]
 [ 0.          0.          0.          1.        ]]
[[ 0.99707852  0.00146494  0.0763687  -0.0473492 ]
 [-0.00776409  0.99658133  0.08225193 -0.12367435]
 [-0.07598708 -0.08260456  0.99368122  0.01360198]
 [ 0.          0.          0.          1.        ]]
[[ 9.97143231e-01 -9.07890816e-04  7.55283210e-02 -2.35753702e-02]
 [-5.26709384e-03  9.96657958e-01  8.15179156e-02 -1.15106004e-01]
 [-7.53498749e-02 -8.16828533e-02  9.93805974e-01  9.7032 …[+6945 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/mosaic.py <<'PY'
import cv2, numpy as np, h5py, sys
from PIL import Image,ImageDraw
start,end,step,cropx0,cropy0,cropx1,cropy1=map(int,sys.argv[1:9])
outp=sys.argv[9]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
idxs=list(range(start,end,step))
ims=[]
with h5py.File('/work/inputs/masks.h5','r') as h:
 om=h['object']; pm=h['person']
 for i in idxs:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read(); fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
  fr=fr[cropy0:cropy1,cropx0:cropx1]
  fr=cv2.resize(fr,((cropx1-cropx0)*2,(cropy1-cropy0)*2))
  p=Image.fromarray(fr);d=ImageDraw.Draw(p);d.rectangle((0,0,70,30),fill='black');d.text((5,5),str(i),fill='white');ims.append(p)
cols=5; rows=(len(ims)+cols-1)//cols; w,h=ims[0].size
out=Image.new('RGB',(w*cols,h*rows))
for j,im in enumerate(ims):out.paste(im,((j%cols)*w,(j//cols)*h))
out.save(outp,quality=93)
PY
python /work/scratch/mosaic.py 300 420 4 700 420 1020 700 /work/scratch/motion_300_420.jpg"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/mosaic.py", line 3, in <module>
    start,end,step,cropx0,cropy0,cropx1,cropy1=map(int,sys.argv[1:9])
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: '/work/scratch/motion_300_420.jpg'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/start,end,step,cropx0,cropy0,cropx1,cropy1=map(int,sys.argv\\[1:9\\])/start,end,step,cropx0,cropy0,cropx1,cropy1=map(int,sys.argv[1:8])/' /work/scratch/mosaic.py
python /work/scratch/mosaic.py 300 420 4 700 420 1020 700 /work/scratch/motion_300_420.jpg"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/mosaic.py", line 4, in <module>
    outp=sys.argv[9]
         ~~~~~~~~^^^
IndexError: list index out of range

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/outp=sys.argv\\[9\\]/outp=sys.argv[8]/' /work/scratch/mosaic.py
python /work/scratch/mosaic.py 300 420 4 700 420 1020 700 /work/scratch/motion_300_420.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=s.to_geometry()
print(type(g),g.visual)
print('uv',g.visual.uv.shape if hasattr(g.visual,'uv') and g.visual.uv is not None else None)
print('mat',g.visual.material)
print('image',g.visual.material.image.size if g.visual.material.image else None)
print('face normals range')
print(g.face_normals.min(0),g.face_normals.max(0))
print('volume area',g.volume,g.area)
print('components',len(g.split(only_watertight=False)))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> <trimesh.visual.texture.TextureVisuals object at 0x77b1445cb190>
uv (294042, 2)
mat <trimesh.visual.material.PBRMaterial object at 0x77b186faabd0>
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);g=s.to_geometry();m=g.visual.material
print(type(m),m.__dict__.keys())
for k,v in m.__dict__.items(): print(k,type(v),getattr(v,'shape',None),getattr(v,'size',None))
print('uv range',g.visual.uv.min(0),g.visual.uv.max(0))
print('baseColorTexture',getattr(m,'baseColorTexture',None))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.material.PBRMaterial'> dict_keys(['_data'])
_data <class 'dict'> None None
uv range [0.001383   0.00108302] [0.99861699 0.99891698]
baseColorTexture <PIL.Image.Image image mode=RGB size=2531x3232 at 0x741EC23ED250>

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 def visit(name,obj):
  if isinstance(obj,h5py.Dataset): print(name,obj.shape,obj.dtype,obj.chunks,obj.compression)
 f.visititems(visit)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle,pprint
p='/pipeline/depth/moge2/intrinsics.pkl';x=pickle.load(open(p,'rb'));print(type(x));
if isinstance(x,dict):
 for k,v in x.items():print(k,type(v),getattr(v,'shape',None),v if getattr(v,'size',1)<30 else '')
else:pprint.pp(x)
PY"
```
<details><summary>output</summary>

```
<class 'dict'>
fx <class 'float'> None 817.865478515625
fy <class 'float'> None 817.865478515625
cx <class 'float'> None 768.0
cy <class 'float'> None 576.0
focals <class 'list'> None [817.865478515625, 815.69970703125, 812.26806640625, 814.73095703125, 813.8219604492188, 817.5750732421875, 816.0789184570312, 817.3055419921875, 813.4839477539062, 815.4505615234375, 817.8015747070312, 816.6614990234375, 816.85791015625, 812.396484375, 813.4191284179688, 812.6585083007812, 810.9226684570312, 815.1932373046875, 814.8528442382812, 813.943359375, 816.655517578125, 812.835693359375, 814.0968017578125, 817.36572265625, 815.1220703125, 815.435791015625, 815.258056640625, 815.6092529296875, 816.6721801757812, 815.9307861328125, 814.912109375, 816.9364013671875, 818.6116333007812, 818.223388671875,  …[+55560 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /workspace/v2d_foundation_pose/lib -maxdepth 4 -type f | head -100; find /workspace/v2d_cari4d/lib -maxdepth 5 -type f | grep -E 'render|object|refine' | head -100"
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
/workspace/v2d_foundation_pose/lib/FoundationPose/bundlesdf/config_li …[+5736 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, numpy as np
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
 d=f[p]; print(p,d.shape,d.dtype)
 for i in [0,100,300,400]:
  a=d[i]; print(i,type(a),a.shape,a.dtype,a.min(),a.max(),a.mean())
print(f['alignment/front_stereo_camera_left/scale'][:10])
print(f['alignment/front_stereo_camera_left/shift'][:10])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (546,) object
0 <class 'numpy.ndarray'> (651081,) uint8 0 255 127.16955033244712
100 <class 'numpy.ndarray'> (655135,) uint8 0 255 127.07339555969381
300 <class 'numpy.ndarray'> (648786,) uint8 0 255 127.32277052834063
400 <class 'numpy.ndarray'> (651415,) uint8 0 255 127.60130331662612
raw/front_stereo_camera_left (546,) object
0 <class 'numpy.ndarray'> (586719,) uint8 0 255 127.50834215356926
100 <class 'numpy.ndarray'> (590520,) uint8 0 255 127.52476630766104
300 <class 'numpy.ndarray'> (587323,) uint8 0 255 127.56158536273907
400 <class 'numpy.ndarray'> (594195,) uint8 0 255 127.44535548094481
[1.1572324 1.1627696 1.1707243 1.1664408 1.1624843 1.1586977 1.1593212
 1.1523722 1.1554984 1.1593244]
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,240p' /workspace/v2d_cari4d/lib/cari4d/render_h5_codec.py; grep -R \"aligned/front\" -n /workspace/v2d_cari4d/lib /workspace/v2d_foundation_pose/lib | head -50"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "PYTHONPATH=/workspace/v2d_cari4d/lib python - <<'PY'
import h5py
from cari4d.render_h5_codec import load_pickled_dataset
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r')
for p in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
 d=f[p]; print(p,d.shape,d.dtype,d.attrs.keys())
 for i in [0,100,300,400]:
  a=load_pickled_dataset(d[i]); print(i,type(a),a.shape,a.dtype,a.min(),a.max(),a.mean())
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left (546,) object <KeysViewHDF5 []>
Traceback (most recent call last):
  File "<stdin>", line 7, in <module>
  File "/workspace/v2d_cari4d/lib/cari4d/render_h5_codec.py", line 67, in load_pickled_dataset
    return decode_pickled_payload(bytes(node[()]), node.attrs.get(RECORD_CODEC_ATTR))
                                                   ^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'attrs'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py, pickle, numpy as np
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r');d=f['aligned/front_stereo_camera_left']
print(d.dtype,d.id.get_type())
x=d[0];print(type(x),x.shape,x.dtype,repr(x)[:300]);
y=x.item();print(type(y),getattr(y,'shape',None),getattr(y,'dtype',None),repr(y)[:300])
if isinstance(y,(bytes,np.bytes_)):
 try:
  q=pickle.loads(y);print('unpickled',type(q),q.shape,q.dtype)
 except Exception as e:print(e)
PY"
```
<details><summary>output</summary>

```
object <h5py.h5t.TypeVlenID object at 0x716822353bf0>
<class 'numpy.ndarray'> (651081,) uint8 array([137,  80,  78, ...,  66,  96, 130], dtype=uint8)
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
ValueError: can only convert an array of size 1 to a Python scalar

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r');d=f['aligned/front_stereo_camera_left']
for i in [0,100,300]:
 b=d[i]; print(i,b.shape,b[:20]); im=cv2.imdecode(b,cv2.IMREAD_UNCHANGED); print(im.shape,im.dtype,im.min(),im.max(),im.mean())
PY"
```
<details><summary>output</summary>

```
0 (651081,) [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0]
(1152, 1536) uint16 1792 10715 4285.707664207176
100 (655135,) [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0]
(1152, 1536) uint16 1812 10209 4291.272423073097
300 (648786,) [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0]
(1152, 1536) uint16 1817 10117 4283.644133956344

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import numpy as np, trimesh, torch, nvdiffrast.torch as dr, cv2, json
from PIL import Image,ImageDraw
import h5py
# setup
device='cuda'
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False); mesh=s.to_geometry()
V=torch.tensor(mesh.vertices,dtype=torch.float32,device=device)
Faces=torch.tensor(mesh.faces,dtype=torch.int32,device=device)
UV=torch.tensor(mesh.visual.uv,dtype=torch.float32,device=device)
tex=torch.tensor(np.array(mesh.visual.material.baseColorTexture),dtype=torch.float32,device=device).permute(2,0,1)[None]/255.
ctx=dr.RasterizeCudaContext()
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);Twc=np.array(cam['T_world_cam'],float);Tcw=np.linalg.inv(Twc)
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5','r') as h: om=h['object']

def render(i):
 H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
 Tcm=Tcw@T[i]
 R=torch.tensor(Tcm[:3,:3],dtype=torch.float32,device=device);t=torch.tensor(Tcm[:3,3],dtype=torch.float32,device=device)
 v=V@R.T+t
 x,y,zc=v.unbind(1)
 X=(2*fx/W)*x+(2*cx/W-1)*zc
 Y=-(2*fy/H)*y+(1-2*cy/H)*zc
 clip=torch.stack([X,Y,torch.zeros_like(zc),zc],1)[None]
 rast=dr.rasterize(ctx,clip,Faces,resolution=[H,W])[0]
 tri=rast[...,3]
 mask=tri>0
 # interpolate uv + z
 uvz=torch.cat([UV,torch.zeros((len(UV),1),device=device),torch.ones((le …[+1585 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 42, in <module>
    ok,fr=cap.read()
          ^^^^^^^^
AttributeError: 'set' object has no attribute 'read'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read();s=s.replace(\"cap=set(indices);frames={}\\nwhile True:\\n ok,fr=cap.read()\\n if not ok:break\\n idx=len(frames)\\n# redo\\ncap=cv2.VideoCapture('/work/inputs/video.mp4'); frames=[]\\nfor i in range(546):\\n ok,fr=cap.read()\\n if i in cap: frames.append((i,cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))\",\"wanted=set(indices); frames=[]\\nfor i in range(546):\\n ok,fr=cap.read()\\n if i in wanted: frames.append((i,cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)))\")
# also variable rgb_np undefined in loop
s=s.replace('mask,rgb=render(i)\\n m=om[i]','mask,rgb=render(i); rgb_np=rgb\\n m=om[i]')
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 46, in <module>
    mask,rgb=render(i); rgb_np=rgb
             ^^^^^^^^^
  File "/work/scratch/render_test.py", line 34, in render
    rgb=torch.nn.functional.grid_sample(tex,sample,mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1,2,0).clamp(0,1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/functional.py", line 4910, in grid_sample
    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: grid_sampler(): expected 4D input and grid with sam …[+110 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read().replace(\"out=dr.interpolate(uvz,rast,Faces)[0]\\n u=out[...,0];vcoord=out[...,1]\\n sample=torch.stack([2*u-1,2*vcoord-1],-1)[None]\",\"out=dr.interpolate(uvz,rast,Faces)[0]\\n u=out[...,0];vcoord=out[...,1]\\n sample=torch.stack([2*u-1,2*vcoord-1],-1)[None]\")
# already [0], why [1,1...]? u shape maybe [1,H,W] because interpolate always batch -> [1, H,W,4], [0] should H,W. Check exactly
print('ok')
PY
grep -n 'out=dr\\|u=\\|sample=' /work/scratch/render_test.py
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 46, in <module>
    mask,rgb=render(i); rgb_np=rgb
             ^^^^^^^^^
  File "/work/scratch/render_test.py", line 34, in render
    rgb=torch.nn.functional.grid_sample(tex,sample,mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1,2,0).clamp(0,1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/functional.py", line 4910, in grid_sample
    return torch.grid_sampler(input, grid, mode_enum, padding_mode_enum, align_corners)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: grid_sampler(): expected 4D input and grid with sam …[+110 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read().replace(\"sample=torch.stack([2*u-1,2*vcoord-1],-1)[None]\",\"sample=torch.stack([2*u-1,2*vcoord-1],-1)\\n if sample.ndim==4: sample=sample[0]\\n sample=sample[None]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 49, in <module>
    m=om[i].astype(bool)
      ~~^^^
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
    ^^^ …[+853 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read().replace(\"with h5py.File('/work/inputs/masks.h5','r') as h: om=h['object']\",\"with h5py.File('/work/inputs/masks.h5','r') as h: om=h['object'][:]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 iou 0.49260696566291695 area 14743 15642
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 54, in <module>
    ar=np.zeros((1152,1536,3),np.uint8);ar[mask]=rgb[mask]
                                                 ~~~^^^^^^
IndexError: boolean index did not match indexed array along dimension 0; dimension is 1152 but corresponding boolean dimension is 1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read().replace(\"rgb_np=(rgb.detach().cpu().numpy()*255).astype(np.uint8);mask_np=mask.cpu().numpy()\",\"rgb_np=(rgb.detach().cpu().numpy()*255).astype(np.uint8)\\n if rgb_np.ndim==4: rgb_np=rgb_np[0]\\n mask_np=mask.cpu().numpy()\\n if mask_np.ndim==3: mask_np=mask_np[0]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 iou 0.49260696566291695 area 14743 15642
100 iou 0.5359520272872311 area 14306 13613
200 iou 0.5342312008978676 area 14621 14086
300 iou 0.46240012847793793 area 18886 17538
350 iou 0.20221147201105735 area 12537 4859
400 iou 0.40854398940280384 area 13154 12366
450 iou 0.4613771199137053 area 12414 11972
545 iou 0.4936761722944756 area 13347 11690

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/Y=-(2\\*fy\\/H)\\*y+(1-2\\*cy\\/H)\\*zc/Y=(2*fy\\/H)*y+(1-2*cy\\/H)*zc/' /work/scratch/render_test.py > /work/scratch/render_test_flip.py
python /work/scratch/render_test_flip.py"
```
<details><summary>output</summary>

```
0 iou 0.5361747307750645 area 14742 15642
100 iou 0.4538117058946053 area 14306 13613
200 iou 0.46334998470792127 area 14622 14086
300 iou 0.4605854049719326 area 18889 17538
350 iou 0.3373049427319548 area 12538 4859
400 iou 0.34857052264440097 area 13153 12366
450 iou 0.3258317025440313 area 12418 11972
545 iou 0.35408328826392643 area 13347 11690

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -name '*.glb' -o -name 'output_aligned*' -printf '%p %s\\n'; ls -la /pipeline/export/episode_000006"
```
<details><summary>output</summary>

```
total 32
drwxrwxr-x 6 hanzhou hanzhou 4096 Oct 15 13:55 .
drwxrwxr-x 3 hanzhou hanzhou 4096 Oct 15 13:55 ..
-rw-rw-r-- 1 hanzhou hanzhou  561 Oct 15 13:55 edex
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 15 13:56 human_masks
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 15 13:56 images
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 15 14:02 object_masks
drwxrwxr-x 2 hanzhou hanzhou 4096 Oct 15 13:55 object_mesh
-rw-rw-r-- 1 hanzhou hanzhou 2876 Oct 15 13:55 wild_export.json

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -la /pipeline/export/episode_000006/object_mesh /pipeline/export/episode_000006/images | head; cat /pipeline/export/episode_000006/wild_export.json'
```
<details><summary>output</summary>

```
/pipeline/export/episode_000006/images:
total 487660
drwxrwxr-x 2 hanzhou hanzhou      4096 Oct 15 13:56 .
drwxrwxr-x 6 hanzhou hanzhou      4096 Oct 15 13:55 ..
-rw-rw-r-- 1 hanzhou hanzhou         0 Oct 15 13:56 .front_stereo_camera_left.h5.storage-repack.lock
-rw-rw-r-- 1 hanzhou hanzhou 499347648 Oct 15 13:55 front_stereo_camera_left.h5

/pipeline/export/episode_000006/object_mesh:
total 10688
drwxrwxr-x 2 hanzhou hanzhou     4096 Oct 15 13:55 .
{
  "aligned_object_extents_m": [
    0.18680381774902344,
    0.5231255292892456,
    0.5388104319572449
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
  "de …[+2530 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_align.py <<'PY'
import numpy as np, trimesh, torch, nvdiffrast.torch as dr, cv2, json
import h5py
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);mesh=s.to_geometry();V=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');Faces=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext()
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K'],float);Tcw=np.linalg.inv(np.array(cam['T_world_cam'],float));z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform'])
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:]

def iou(i,X,sign):
 H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];Tcm=Tcw@(X@T[i]);R=torch.tensor(Tcm[:3,:3],dtype=torch.float32,device='cuda');t=torch.tensor(Tcm[:3,3],dtype=torch.float32,device='cuda');v=V@R.T+t;x,y,zc=v.unbind(1);clx=(2*fx/W)*x+(2*cx/W-1)*zc;cly=sign*(2*fy/H)*y+(1-2*cy/H)*zc;clip=torch.stack([clx,cly,torch.zeros_like(zc),zc],1)[None];rast=dr.rasterize(ctx,clip,Faces,resolution=[H,W])[0];mask=(rast[...,3]>0).cpu().numpy();m=om[i];return (mask&m).sum()/(mask|m).sum()
for name,X in [('I',np.eye(4)),('T@A',A),('T@iA',np.linalg.inv(A)),('A@T',None),('iA@T',None)]:
 for sign in [-1,1]:
  vals=[]
  for i in [0,100,200,300,350,400,450,545]:
   if name=='A@T': q=A@T[i]
   elif na …[+721 chars]
```
<details><summary>output</summary>

```
I -1 [0.493 0.536 0.534 0.462 0.202 0.409 0.461 0.494] 0.4488748844934995
I 1 [0.536 0.454 0.463 0.461 0.337 0.349 0.326 0.354] 0.40996403531672965
T@A -1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
T@A 1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
T@iA -1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
T@iA 1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
A@T -1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
A@T 1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
iA@T -1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
iA@T 1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_signs.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,cv2,json,h5py
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=s.to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:]
for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
 vals=[]
 for i in [0,100,200,300,350,400,450,545]:
  H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];Tcm=Tcw@T[i];R=torch.tensor(Tcm[:3,:3],dtype=torch.float32,device='cuda');t=torch.tensor(Tcm[:3,3],dtype=torch.float32,device='cuda');v=V@R.T+t;x,y,z=v.unbind(1);X=sx*(2*fx/W)*x+(sx*2*cx/W-1)*z;Y=sy*(2*fy/H)*y+(1-sy*2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,resolution=[H,W])[0];mk=(rast[...,3]>0).cpu().numpy();vals.append((mk&om[i]).sum()/(mk|om[i]).sum())
 print(sx,sy,np.round(vals,3),np.mean(vals))
PY
python /work/scratch/test_signs.py"
```
<details><summary>output</summary>

```
-1 -1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
-1 1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
1 -1 [0. 0. 0. 0. 0. 0. 0. 0.] 0.0
1 1 [0.536 0.454 0.463 0.461 0.337 0.349 0.326 0.354] 0.40996403531672965

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# compute projection bbox from vertices exact using cv2
import numpy as np,trimesh,json,h5py
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);v=s.to_geometry().vertices
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:]
for i in [0,100,200,300,400,545]:
 vc=(Tcw@T[i])[:3,:3]@v.T+(Tcw@T[i])[:3,3:4];p=K@vc;p=p[:2]/p[2];ys,xs=np.where(om[i]);print(i,'proj bbox',np.round([p[0].min(),p[1].min(),p[0].max(),p[1].max()],1),'cent',np.round(p[:2].mean(1),1),'mask',[xs.min(),ys.min(),xs.max(),ys.max()],np.round([xs.mean(),ys.mean()],1),'z range',vc[2].min(),vc[2].max())
PY"
```
<details><summary>output</summary>

```
0 proj bbox [759.6 501.3 892.4 673.4] cent [824.8 593.9] mask [757, 499, 890, 681] [820.6 571.8] z range 2.370611468698286 2.787003668375556
100 proj bbox [759.3 506.  889.3 676. ] cent [822.8 597.8] mask [756, 500, 889, 626] [821.1 558.8] z range 2.406287508194589 2.8259184876827086
200 proj bbox [774.  504.9 898.8 675.7] cent [833. 596.] mask [771, 499, 898, 626] [832.6 558.5] z range 2.4203020084777043 2.8089603529637865
300 proj bbox [803.9 499.2 964.9 669.6] cent [883.5 589.6] mask [799, 495, 965, 612] [882.5 552. ] z range 2.443800786149242 2.811287934827263
400 proj bbox [ 874.7  503.8 1002.2  680.7] cent [944.4 598.9] mask [864, 497, 995, 645] [923.6 563.3] z range 2.316115563413077 2.7841983068913416
545 proj bbox [867.9 507.3 991.7 683.8] cent [937.5 603.3] mask [862, 499, 989, 6 …[+63 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_diag.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split('indices=')[0])
from PIL import Image,ImageDraw
for i in [0,100,300,400,545]:
 mask,rgb=render(i)
 if mask.ndim==3:mask=mask[0]
 if rgb.ndim==4:rgb=rgb[0]
 m=om[i]
 cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 x0,y0,x1,y1=730,470,1020,700
 a=im.copy();a[m]=np.array([255,0,0]);a[mask]=np.array([0,0,255]);a[m&mask]=np.array([255,0,255])
 diff=np.zeros_like(a);diff[m&~mask]=[255,0,0];diff[mask&~m]=[0,0,255];diff[m&mask]=[255,255,255]
 stack=np.concatenate([im[y0:y1,x0:x1],a[y0:y1,x0:x1],diff[y0:y1,x0:x1]],1)
 stack=cv2.resize(stack,(870,230))
 Image.fromarray(stack).save(f'/work/scratch/diag_{i}.jpg',quality=95)
PY
python /work/scratch/render_diag.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_align2.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,cv2,json,h5py
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=s.to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform']);Ai=np.linalg.inv(A)
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:]
def score(P,sx=1,sy=1):
 vals=[];areas=[]
 for i in [0,100,200,300,350,400,450,545]:
  H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];Tcm=Tcw@P[i];R=torch.tensor(Tcm[:3,:3],dtype=torch.float32,device='cuda');t=torch.tensor(Tcm[:3,3],dtype=torch.float32,device='cuda');v=V@R.T+t;x,y,z=v.unbind(1);X=sx*(2*fx/W)*x+(sx*2*cx/W-1)*z;Y=sy*(2*fy/H)*y+(1-sy*2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,resolution=[H,W])[0];mk=(rast[...,3]>0).cpu().numpy();vals.append((mk&om[i]).sum()/(mk|om[i]).sum());areas.append(mk.sum())
 return np.mean(vals),vals,areas
names=[('I',T),('T@A',T@A),('T@Ai',T@Ai),('A@T',A@T),('Ai@T',Ai@T)]
for name,P in names:
 for sx,sy in [(1,1),(1,-1),(- …[+146 chars]
```
<details><summary>output</summary>

```
I 1 1 0.41 [0.536 0.454 0.463 0.461 0.337 0.349 0.326 0.354] [14742, 14306]
I 1 -1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
I -1 1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
I -1 -1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
T@A 1 1 0.1317 [0.15  0.117 0.12  0.107 0.1   0.178 0.144 0.138] [9178, 8878]
T@A 1 -1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
T@A -1 1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
T@A -1 -1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
T@Ai 1 1 0.1374 [0.156 0.126 0.129 0.116 0.1   0.182 0.149 0.142] [9127, 8853]
T@Ai 1 -1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
T@Ai -1 1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
T@Ai -1 -1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
A@T 1 1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
A@T 1 -1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
A@T -1 1 0.0 [0. 0. 0. 0. 0. 0. 0. 0.] [0, 0]
A@T -1 -1 0.0 [0 …[+219 chars]
```
</details>

**shell** (exit 137)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=s.to_geometry();parts=m.split(only_watertight=False)
print('parts',len(parts))
for j,p in enumerate(sorted(parts,key=lambda x:-len(x.faces))[:20]):
 print(j,len(p.vertices),len(p.faces),'bounds',np.round(p.bounds,3).tolist(),'cent',np.round(p.centroid,3),'area',round(p.area,4))
PY"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 7:  3794 Killed                  python - <<'PY'
import trimesh,numpy as np
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=s.to_geometry();parts=m.split(only_watertight=False)
print('parts',len(parts))
for j,p in enumerate(sorted(parts,key=lambda x:-len(x.faces))[:20]):
 print(j,len(p.vertices),len(p.faces),'bounds',np.round(p.bounds,3).tolist(),'cent',np.round(p.centroid,3),'area',round(p.area,4))
PY


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
for ax in range(3):
 h,e=np.histogram(v[:,ax],bins=30);print('axis',ax,'top bins',[(round(e[i],3),round(e[i+1],3),int(h[i])) for i in np.argsort(h)[-10:][::-1]])
# y slices, bounds of others and counts
for lo,hi in [(-.09,-.06),(-.06,-.02),(-.02,.02),(.02,.06),(.06,.11)]:
 q=v[(v[:,1]>=lo)&(v[:,1]<hi)];print('yslice',lo,hi,len(q), 'bounds',np.round(q.min(0),3) if len(q) else None,np.round(q.max(0),3) if len(q) else None)
PY"
```
<details><summary>output</summary>

```
axis 0 top bins [(0.233, 0.251, 40589), (0.251, 0.268, 34028), (-0.254, -0.237, 21234), (0.077, 0.094, 20376), (0.024, 0.042, 20322), (0.042, 0.059, 18922), (-0.01, 0.007, 16037), (0.059, 0.077, 15898), (-0.237, -0.219, 15470), (0.007, 0.024, 14579)]
axis 1 top bins [(-0.048, -0.041, 25286), (-0.041, -0.035, 19694), (0.033, 0.04, 19330), (-0.054, -0.048, 18919), (-0.035, -0.029, 18296), (0.027, 0.033, 16830), (-0.029, -0.023, 15902), (0.021, 0.027, 13201), (-0.01, -0.004, 11779), (-0.023, -0.016, 10984)]
axis 2 top bins [(0.038, 0.056, 20737), (0.02, 0.038, 19931), (-0.016, 0.002, 18499), (-0.034, -0.016, 18383), (-0.052, -0.034, 16164), (0.002, 0.02, 15731), (-0.088, -0.07, 12444), (0.253, 0.271, 12433), (-0.124, -0.106, 12062), (-0.106, -0.088, 11919)]
yslice -0.09 -0.06 14459 bounds [ 0 …[+342 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/mesh_views.py <<'PY'
import trimesh,numpy as np,cv2,os
os.environ['PYOPENGL_PLATFORM']='egl'
import pyrender
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
# ensure texture material
scene=pyrender.Scene(ambient_light=[.5,.5,.5],bg_color=[255,255,255])
mesh=pyrender.Mesh.from_trimesh(m,smooth=False)
scene.add(mesh)
def look_at(eye,target,up):
 eye=np.asarray(eye,float);target=np.asarray(target,float);up=np.asarray(up,float)
 z=(eye-target);z/=np.linalg.norm(z);x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x);M=np.eye(4);M[:3,0]=x;M[:3,1]=y;M[:3,2]=z;M[:3,3]=eye;return M
cam=pyrender.IntrinsicsCamera(800,800,640,480,znear=.01,zfar=10)
r=pyrender.OffscreenRenderer(640,480)
views=[('neg_y',(0,-1,0),(0,0,0),(0,0,1)),('pos_y',(0,1,0),(0,0,0),(0,0,1)),('pos_z',(0,0,1),(0,0,0),(0,-1,0)),('neg_z',(0,0,-1),(0,0,0),(0,-1,0)),('pos_x',(1,0,0),(0,0,0),(0,-1,0)),('neg_x',(-1,0,0),(0,0,0),(0,-1,0))]
ims=[]
for n,e,t,u in views:
 c=pyrender.Node(camera=cam)
 sc=pyrender.Scene(ambient_light=[.8,.8,.8],bg_color=[255,255,255]);sc.add(mesh);sc.add(c) if False else None
 # pyrender doesn't accept node? add camera directly
 sc.add(cam,pose=look_at(e,t,u))
 color,depth=r.render(sc)
 cv2.putText(color,n,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
 ims.append(color)
r.delete()
out=np.concatenate([np.concatenate(ims[:3],1),np.concatenate(ims[3:],1)],0)
Image.fromarray(out).save('/work/scratch/m …[+65 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/mesh_views.py", line 23, in <module>
    cv2.putText(color,n,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'putText'
> Overload resolution failed:
>  - img marked as output argument, but provided NumPy array marked as readonly
>  - Expected Ptr<cv::UMat> for argument 'img'


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/color,depth=r.render(sc)/color,depth=r.render(sc);color=color.copy()/' /work/scratch/mesh_views.py
python /work/scratch/mesh_views.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for i in [0,100,300,400,545]:
 R=T[i,:3,:3]
 print(i,'axes mesh->world x',np.round(R[:,0],3),'y',np.round(R[:,1],3),'z',np.round(R[:,2],3),'world up dot meshz',R[1,2])
PY"
```
<details><summary>output</summary>

```
0 axes mesh->world x [ 0.009  1.    -0.007] y [-0.685  0.011  0.729] z [ 0.729 -0.002  0.685] world up dot meshz -0.0019081899663433433
100 axes mesh->world x [ 0.     1.    -0.013] y [-0.687  0.01   0.727] z [0.727 0.009 0.686] world up dot meshz 0.00866006687283516
300 axes mesh->world x [0.032 0.998 0.052] y [ 0.618 -0.061  0.783] z [ 0.785  0.007 -0.619] world up dot meshz 0.00726312818005681
400 axes mesh->world x [-0.001  0.998  0.064] y [ 0.672  0.049 -0.739] z [-0.741  0.042 -0.67 ] world up dot meshz 0.04224245995283127
545 axes mesh->world x [0.055 0.997 0.045] y [ 0.644 -0.001 -0.765] z [-0.763  0.071 -0.643] world up dot meshz 0.07109414041042328

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();n=m.face_normals;a=m.area_faces
for ax in range(3):
 for s in [1,-1]:
  sel=(n[:,ax]*s>.9);print(ax,s,'faces',sel.sum(),'area',a[sel].sum())
# normals histogram dominant directions
from collections import Counter
key=np.round(n,1);c=Counter(map(tuple,key));print(c.most_common(20))
PY"
```
<details><summary>output</summary>

```
0 1 faces 7491 area 0.045632158206655025
0 -1 faces 7728 area 0.05234692588637257
1 1 faces 18465 area 0.19752276738590063
1 -1 faces 20411 area 0.2136326024635135
2 1 faces 4438 area 0.018537327778007307
2 -1 faces 3406 area 0.018872340998088856
[((-0.2, -1.0, 0.0), 5918), ((-0.2, -1.0, -0.1), 2360), ((-0.1, -1.0, 0.0), 2012), ((0.2, 1.0, 0.0), 1251), ((-1.0, -0.0, 0.0), 1137), ((-0.1, -1.0, -0.1), 1131), ((0.3, 1.0, -0.0), 1045), ((0.2, 1.0, 0.1), 923), ((0.1, 1.0, 0.0), 902), ((-0.0, 1.0, -0.0), 812), ((1.0, 0.0, 0.0), 746), ((0.1, 1.0, 0.1), 651), ((-0.1, 1.0, 0.0), 631), ((1.0, 0.1, 0.0), 626), ((-0.0, -1.0, -0.1), 548), ((0.3, 1.0, 0.1), 545), ((-1.0, -0.1, 0.0), 540), ((0.0, 1.0, 0.1), 515), ((0.4, 0.9, 0.0), 496), ((0.0, -0.1, 1.0), 480)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();n=m.face_normals;a=m.area_faces
for ax,s in [(1,1),(1,-1)]:
 sel=(n[:,ax]*s>.95);fi=m.faces[sel];tri=m.vertices[fi]
 print('normal',ax,s,'faces',sel.sum(),'area',a[sel].sum())
 for j,axn in enumerate('xyz'):
  q=tri[:,:,j];print(axn,q.min(),np.percentile(q,[1,5,50,95,99]),q.max())
 # per-face centers distribution
 c=tri.mean(1)
 print('cent bounds',c.min(0),c.max(0),'med',np.median(c,0))
PY"
```
<details><summary>output</summary>

```
normal 1 1 faces 12692 area 0.1497545772482524
x -0.25316905975341797 [-0.24485731 -0.237914    0.00652881  0.24691214  0.25878796] 0.2623472511768341
y -0.06418386846780777 [-0.05640106 -0.05200212  0.00194649  0.08640891  0.09629242] 0.10194340348243713
z -0.26383382081985474 [-0.26055628 -0.2143058  -0.02981131  0.2015263   0.2610971 ] 0.26623299717903137
cent bounds [-0.2518244  -0.06404918 -0.26351108] [0.26195375 0.10190435 0.26580539] med [ 0.00688419  0.001972   -0.02935356]
normal 1 -1 faces 17319 area 0.2041517204879131
x -0.2523677945137024 [-0.24716724 -0.2423903  -0.02142149  0.18108657  0.25334835] 0.26803699135780334
y -0.0849619209766388 [-0.08384999 -0.05889321 -0.0200294   0.0416818   0.09123793] 0.09357850253582001
z -0.266518771648407 [-0.25951484 -0.2259869   0.0122848 …[+173 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr,trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda');ctx=dr.RasterizeCudaContext()
v=V.clone();# simple orthographic NDC
clip=torch.stack([v[:,1]/1.0,v[:,2]/1.0,torch.zeros(len(v),device='cuda'),torch.ones(len(v),device='cuda')],1)[None]
r=dr.rasterize(ctx,clip,Fa,resolution=[512,512])[0]
print('rast',r.shape)
for attr in [UV[None],torch.cat([UV,torch.zeros(len(UV),1,device='cuda')],1)[None],torch.cat([UV,torch.zeros(len(UV),1,device='cuda'),torch.ones(len(UV),1,device='cuda')],1)[None]]:
 try:
  o=dr.interpolate(attr,r,Fa);print(attr.shape,o.shape,o[0,250,250])
 except Exception as e:print('err',e)
PY"
```
<details><summary>output</summary>

```
rast torch.Size([1, 512, 512, 4])
err 'tuple' object has no attribute 'shape'
err 'tuple' object has no attribute 'shape'
err 'tuple' object has no attribute 'shape'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr,trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda');ctx=dr.RasterizeCudaContext();clip=torch.stack([V[:,1],V[:,2],torch.zeros(len(V),device='cuda'),torch.ones(len(V),device='cuda')],1)[None];r=dr.rasterize(ctx,clip,Fa,resolution=[512,512])[0]
for attr in [UV[None],torch.cat([UV,torch.zeros(len(UV),1,device='cuda')],1)[None],torch.cat([UV,torch.zeros(len(UV),1,device='cuda'),torch.ones(len(UV),1,device='cuda')],1)[None]]:
 try:
  o=dr.interpolate(attr,r,Fa);print(attr.shape,type(o),len(o),[x.shape for x in o if torch.is_tensor(x)],o[0][0,250,250])
 except Exception as e:print('err',e)
PY"
```
<details><summary>output</summary>

```
torch.Size([1, 294042, 2]) <class 'tuple'> 2 [torch.Size([1, 512, 512, 2]), torch.Size([1, 512, 512, 0])] tensor([0.1740, 0.2647], device='cuda:0')
torch.Size([1, 294042, 3]) <class 'tuple'> 2 [torch.Size([1, 512, 512, 3]), torch.Size([1, 512, 512, 0])] tensor([0.1740, 0.2647, 0.0000], device='cuda:0')
torch.Size([1, 294042, 4]) <class 'tuple'> 2 [torch.Size([1, 512, 512, 4]), torch.Size([1, 512, 512, 0])] tensor([0.1740, 0.2647, 0.0000, 1.0000], device='cuda:0')

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_canonical.py <<'PY'
import trimesh,numpy as np,torch,nvdiffrast.torch as dr,cv2
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda');ctx=dr.RasterizeCudaContext();tex=torch.tensor(np.array(m.visual.material.baseColorTexture),dtype=torch.float32,device='cuda').permute(2,0,1)[None]/255.
def rend(view,flipv):
 # map view axis to z, use x horiz and z vert; camera looking along sign? clip x=x, y=z (screen row down -> y down)
 if view=='posy': x=V[:,0]; y=-V[:,2]; z=-V[:,1]
 elif view=='negy': x=V[:,0]; y=-V[:,2]; z=V[:,1]
 elif view=='posx': x=V[:,1]; y=-V[:,2]; z=-V[:,0]
 elif view=='negx': x=V[:,1]; y=-V[:,2]; z=V[:,0]
 elif view=='posz': x=V[:,0]; y=-V[:,1]; z=-V[:,2]
 elif view=='negz': x=V[:,0]; y=-V[:,1]; z=V[:,2]
 clip=torch.stack([x/.3,y/.3,torch.zeros(len(V),device='cuda'),torch.ones(len(V),device='cuda')],1)[None]
 rast=dr.rasterize(ctx,clip,Fa,resolution=[600,600])[0];uv=dr.interpolate(UV[None],rast,Fa)[0];u=uv[...,0];v=uv[...,1];
 if flipv:v=1-v
 sample=torch.stack([2*u-1,2*v-1],-1)[None];rgb=torch.nn.functional.grid_sample(tex,sample,mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1,2,0).clamp(0,1).cpu().numpy();mk=(rast[...,3]>0).cpu().numpy();
 if mk.ndim==3:mk=mk[0 …[+463 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_canonical.py", line 20, in <module>
    for fl in [False,True]:ims.append(rend(view,fl))
                                      ^^^^^^^^^^^^^
  File "/work/scratch/render_canonical.py", line 15, in rend
    sample=torch.stack([2*u-1,2*v-1],-1)[None];rgb=torch.nn.functional.grid_sample(tex,sample,mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1,2,0).clamp(0,1).cpu().numpy();mk=(rast[...,3]>0).cpu().numpy();
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/nn/functional.py", line 4910, in grid_sample
    return torch.grid_sampler(input, grid, mode_enum …[+297 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/uv=dr.interpolate(UV\\[None\\],rast,Fa)\\[0\\]/uv=dr.interpolate(UV[None],rast,Fa)[0][0]/\" /work/scratch/render_canonical.py
python /work/scratch/render_canonical.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np
ctx=dr.RasterizeCudaContext()
# triangle covering y negative side? vertices (-.5,.8),(.5,.8),(0,0) in clip
for y in [.8,-.8]:
 v=torch.tensor([[-.5,y,0,1],[.5,y,0,1],[0,0,0,1]],dtype=torch.float32,device='cuda')[None];f=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda');r=dr.rasterize(ctx,v,f,resolution=[100,100])[0];m=r[...,3]>0;ys,xs=torch.where(m);print(y,ys.min().item(),ys.max().item())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
ValueError: too many values to unpack (expected 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,nvdiffrast.torch as dr,numpy as np
ctx=dr.RasterizeCudaContext()
for y in [.8,-.8]:
 v=torch.tensor([[-.5,y,0,1],[.5,y,0,1],[0,0,0,1]],dtype=torch.float32,device='cuda')[None];f=torch.tensor([[0,1,2]],dtype=torch.int32,device='cuda');r=dr.rasterize(ctx,v,f,resolution=[100,100])[0];m=r[...,3]>0;print(m.shape);m=m.reshape(100,100);ys,xs=torch.where(m);print(y,ys.min().item(),ys.max().item())
PY"
```
<details><summary>output</summary>

```
torch.Size([1, 100, 100])
0.8 51 89
torch.Size([1, 100, 100])
-0.8 10 48

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_coord.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split('indices=')[0])
from PIL import Image,ImageDraw
# Build coordinate color attr
cols=torch.zeros((len(V),3),device=device)
cols[:,0]=(V[:,0]-V[:,0].min())/(V[:,0].max()-V[:,0].min())
cols[:,1]=(V[:,1]-V[:,1].min())/(V[:,1].max()-V[:,1].min())
cols[:,2]=(V[:,2]-V[:,2].min())/(V[:,2].max()-V[:,2].min())
def rend(i):
 H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];Tcm=Tcw@T[i];R=torch.tensor(Tcm[:3,:3],dtype=torch.float32,device=device);t=torch.tensor(Tcm[:3,3],dtype=torch.float32,device=device);v=V@R.T+t;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Faces,[H,W])[0];out=dr.interpolate(cols[None],rast,Faces)[0][0].clamp(0,1);mk=rast[...,3]>0;mk=mk.reshape(H,W);return mk,out.cpu().numpy()
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [0,100,300,400,545]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);mk,rgb=rend(i);a=im.copy();a[mk]=(rgb[mk]*255).astype(np.uint8);x0,y0,x1,y1=730,470,1020,700;z=np.concatenate([im[y0:y1,x0:x1],a[y0:y1,x0:x1]],1);z=cv2.resize(z,(580,230));p=Image.fromarray(z);ImageDraw.Draw(p).text((5,5),str(i),fill='white');ims.append(p)
out=Image.new('RGB',(580,230*len(ims)))
for j,p in enumerate(ims):out.paste(p,(0,j*230))
out.save('/work/scratch/coord.jpg',quality=94)
PY
python /wor …[+26 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_coord.py", line 12, in <module>
    cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);mk,rgb=rend(i);a=im.copy();a[mk]=(rgb[mk]*255).astype(np.uint8);x0,y0,x1,y1=730,470,1020,700;z=np.concatenate([im[y0:y1,x0:x1],a[y0:y1,x0:x1]],1);z=cv2.resize(z,(580,230));p=Image.fromarray(z);ImageDraw.Draw(p).text((5,5),str(i),fill='white');ims.append(p)
                                                                                                                               ~~~^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^^^^^^^^^^^
TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to …[+39 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/return mk,out.cpu().numpy()/return mk.cpu().numpy(),out.cpu().numpy()/' /work/scratch/render_coord.py
python /work/scratch/render_coord.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();im=m.visual.material.baseColorTexture;im.save('/work/scratch/texture.png');print(im.mode,im.size)
PY"
```
<details><summary>output</summary>

```
RGB (2531, 3232)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,100,300,400,545]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();cv2.imwrite(f'/work/scratch/frame_{i}.jpg',im[450:720,720:1030])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_baseline.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2,time
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=s.to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
H,W=1152,1536;fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
def render(i):
 Tcm=Tcw@T[i];R=torch.tensor(Tcm[:3,:3],dtype=torch.float32,device='cuda');t=torch.tensor(Tcm[:3,3],dtype=torch.float32,device='cuda');v=V@R.T+t;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();return mk
ious=[]
t0=time.time()
for i in range(len(T)):
 mk=render(i);vis=mk&~pm[i];ius=(vis&om[i]).sum()/max(1,(vis|om[i]).sum());ious.append(ius)
 if i%50==0:print(i,ius,time.time()-t0)
ious=np.array(ious);np.save('/work/scratch/base_iou.npy',ious)
print('time',time.time()-t0,'iou min med mean max',ious.min(),np.median(ious),ious.mean(),ious.max())
for lo in range(0,546,20):print(lo,lo+19,round(ious[lo:lo+20].mean( …[+158 chars]
```
<details><summary>output</summary>

```
0 0.5361747307750645 0.10502266883850098
50 0.4992176063802938 0.3363533020019531
100 0.47466622893412125 0.5528268814086914
150 0.5122598870056497 0.7744309902191162
200 0.4865125240847784 0.9850907325744629
250 0.5116036203295428 1.2079811096191406
300 0.5124062834701892 1.4252018928527832
350 0.44789221190160255 1.6523125171661377
400 0.392128886510909 1.8449797630310059
450 0.33463599821348816 2.064635753631592
500 0.3540860525555876 2.276271104812622
time 2.478548049926758 iou min med mean max 0.021127465705417042 0.48931314373021384 0.4534824146079515 0.5740740740740741
0 19 0.56 0.528
20 39 0.514 0.498
40 59 0.508 0.499
60 79 0.498 0.469
80 99 0.457 0.442
100 119 0.488 0.466
120 139 0.492 0.486
140 159 0.502 0.494
160 179 0.49 0.483
180 199 0.485 0.48
200 219 0.493 0.487
220 239 0.5 …[+892 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform'])
va=v@A[:3,:3].T+A[:3,3]
print('source bounds',v.min(0),v.max(0),v.max(0)-v.min(0))
print('Acol bounds',va.min(0),va.max(0),va.max(0)-va.min(0))
vr=v@A[:3,:3]+A[:3,3]
print('Arow bounds',vr.min(0),vr.max(0),vr.max(0)-vr.min(0))
# inverse
Ai=np.linalg.inv(A); vi=v@Ai[:3,:3].T+Ai[:3,3]
print('inv bounds',vi.min(0),vi.max(0),vi.max(0)-vi.min(0))
PY"
```
<details><summary>output</summary>

```
source bounds [-0.25416997 -0.08496192 -0.26776227] [0.26815489 0.1019434  0.27105448] [0.52232486 0.18690532 0.53881675]
Acol bounds [-0.09340191 -0.26156277 -0.26940521] [0.09340191 0.26156277 0.26940522] [0.18680381 0.52312555 0.53881043]
Arow bounds [-0.0934011  -0.26160309 -0.26948544] [0.09343833 0.26164624 0.26933335] [0.18683943 0.52324933 0.53881879]
inv bounds [-0.09715431 -0.2578082  -0.26948896] [0.08968513 0.26544116 0.26932984] [0.18683943 0.52324935 0.53881881]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:];p=h['person'][:]
for i in [0,100,300,400,545]:
 cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();
 mk=np.zeros_like(im);mk[o[i]]=im[o[i]];mk2=np.zeros_like(im);mk2[o[i]]=[255,255,255];mk2[p[i]]=[0,0,255]
 cv2.imwrite(f'/work/scratch/maskonly_{i}.jpg',np.concatenate([im[450:720,720:1030],mk[450:720,720:1030],mk2[450:720,720:1030]],1))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_fp_aligned.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,pickle
s=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False);m=s.to_geometry();A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform']);Va=(m.vertices@A[:3,:3].T+A[:3,3]).astype(np.float32);Fa=m.faces
V=torch.tensor(Va,device='cuda');Faces=torch.tensor(Fa,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
x=pickle.load(open('/pipeline/foundationpose/episode_000006.pkl','rb'));P=x['fp_poses_cam'][:,0].astype(float)
rec=np.load('/work/inputs/reconstruction.npz'); R=rec['T_cam_mesh_foundationpose'].astype(float)
def rend(T):
 v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Faces,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
for name,poses in [('fp_pkl',P),('rec_fp',R)]:
 vals=[];valsv=[]
 for i in [0,100,200,300,350,400,450,545]:
  mk=rend(poses[i]);vals.append((mk&om[i]).sum()/(mk|om[i]).sum());vis=mk&~pm[i]; …[+257 chars]
```
<details><summary>output</summary>

```
fp_pkl raw [0.74  0.763 0.793 0.787 0.313 0.606 0.552 0.639] 0.6492019822725066 vis [0.74  0.767 0.821 0.811 0.313 0.676 0.553 0.638] 0.6648836012193495
internal [0.73970765 0.77102166 0.8207794  0.8122549  0.31342715 0.67605335
 0.55343217 0.6414398 ]
rec_fp raw [0.29  0.307 0.329 0.345 0.356 0.202 0.185 0.202] 0.27689051335915627 vis [0.29  0.332 0.362 0.356 0.356 0.222 0.203 0.218] 0.29249095252784707
internal [0.73970765 0.77102166 0.8207794  0.8122549  0.31342715 0.67605335
 0.55343217 0.6414398 ]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_rec_input.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,pickle
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Faces=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
r=np.load('/work/inputs/reconstruction.npz')
def rend(T):
 v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Faces,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
for name,poses in [('fp',r['T_cam_mesh_foundationpose'].astype(float)),('coco',r['T_cam_mesh_coconet'].astype(float))]:
 vals=[];valsv=[]
 for i in [0,100,200,300,350,400,450,545]:
  mk=rend(poses[i]);vals.append((mk&om[i]).sum()/(mk|om[i]).sum());vis=mk&~pm[i];valsv.append((vis&om[i]).sum()/(vis|om[i]).sum())
 print(name,'raw',np.round(vals,3),np.mean(vals),'vis',np.round(valsv,3),np.mean(valsv))
PY
python /work/scratch/eval_rec_input.py"
```
<details><summary>output</summary>

```
fp raw [0.74  0.763 0.793 0.787 0.313 0.606 0.552 0.639] 0.6492019822725066 vis [0.74  0.767 0.821 0.811 0.313 0.676 0.553 0.638] 0.6648836012193495
coco raw [0.825 0.782 0.795 0.813 0.249 0.585 0.575 0.658] 0.6603566923477061 vis [0.825 0.787 0.825 0.849 0.305 0.673 0.58  0.659] 0.6881136692374559

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Kp=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);r=np.load('/work/inputs/reconstruction.npz')
for i in [0,100,300,400,545]:
 print('\\nframe',i)
 for name,P,Kk in [('world',r['T_world_mesh'][i].astype(float),K),('Twc@fp',Twc@r['T_cam_mesh_foundationpose'][i].astype(float),Kp),('Twc@coco',Twc@r['T_cam_mesh_coconet'][i].astype(float),Kp),('Twc@fpK',Twc@r['T_cam_mesh_foundationpose'][i].astype(float),K),('Tcw@worldKp',Tcw@r['T_world_mesh'][i].astype(float),Kp)]:
  M=Tcw@P if name.startswith('Tcw') else P
  vc=M[:3,:3]@v.T+M[:3,3:4];p=Kk@vc;p=p[:2]/p[2];print(name,'bbox',np.round([p[0].min(),p[1].min(),p[0].max(),p[1].max()],1),'cent',np.round(p.mean(1),1))
PY"
```
<details><summary>output</summary>

```

frame 0
world bbox [796.1 511.8 933.2 685.2] cent [863.5 607. ]
Twc@fp bbox [795.4 525.4 929.3 681.4] cent [862.5 610.7]
Twc@coco bbox [796.8 513.3 931.8 684.1] cent [863.3 607.1]
Twc@fpK bbox [785.6 509.8 919.8 666.2] cent [852.9 595.2]
Tcw@worldKp bbox [733.  503.1 862.1 676.8] cent [796.4 596.2]

frame 100
world bbox [796.1 517.  930.3 688.5] cent [861.8 611.4]
Twc@fp bbox [792.7 520.7 929.8 688.1] cent [861.  614.1]
Twc@coco bbox [796.2 517.6 929.3 687.7] cent [861.4 611.3]
Twc@fpK bbox [782.9 505.  920.3 672.9] cent [851.3 598.7]
Tcw@worldKp bbox [732.4 508.2 858.9 678.8] cent [794.1 599.5]

frame 300
world bbox [ 843.4  510.  1003.8  682.9] cent [923.4 603.1]
Twc@fp bbox [ 842.   509.9 1005.5  682.5] cent [923.5 605.9]
Twc@coco bbox [ 842.6  508.9 1003.4  682.5] cent [922.8 602.7]
T …[+757 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float)
for idx in [0,100]:
 M=T[idx];vc=M[:3,:3]@v.T+M[:3,3:4];p=K@vc;p=p[:2]/p[2];print('direct',np.round([p[0].min(),p[1].min(),p[0].max(),p[1].max()],1))
 M2=Tcw@T[idx];vc=M2[:3,:3]@v.T+M2[:3,3:4];p=K@vc;p=p[:2]/p[2];print('Tcw@world',np.round([p[0].min(),p[1].min(),p[0].max(),p[1].max()],1))
 print('T',T[idx,:3,3], 'X',M[:3,3])
PY"
```
<details><summary>output</summary>

```
direct [796.1 511.8 933.2 685.2]
Tcw@world [759.6 501.3 892.4 673.4]
T [0.33102539 0.1097873  2.58335042] X [0.33102539 0.1097873  2.58335042]
direct [796.1 517.  930.3 688.5]
Tcw@world [759.3 506.  889.3 676. ]
T [0.33027282 0.12599939 2.61917639] X [0.33027282 0.12599939 2.61917639]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# quick modify eval_rec_input to both Ks
s=open('/work/scratch/eval_rec_input.py').read();s=s.replace(\"for name,poses in [('fp',r['T_cam_mesh_foundationpose'].astype(float)),('coco',r['T_cam_mesh_coconet'].astype(float))]:\",\"for K,ktag in [(np.array(cam['K_pipeline']),'Kp'),(np.array(cam['K']),'Kcam')]:\\n fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]\\n for name,poses in [('fp',r['T_cam_mesh_foundationpose'].astype(float)),('coco',r['T_cam_mesh_coconet'].astype(float))]:\")
# indentation broken likely
open('/work/scratch/eval_rec_both.py','w').write(s)
PY
sed -n '1,80p' /work/scratch/eval_rec_both.py"
```
<details><summary>output</summary>

```
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,pickle
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Faces=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
r=np.load('/work/inputs/reconstruction.npz')
def rend(T):
 v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y, …[+648 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_rec_both.py';s=open(p).read();prefix=s.split('for K,ktag')[0]
tail='''for Ktag in ['Kp','Kcam']:
 K=np.array(cam[Ktag])
 fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
 for name,poses in [('fp',r['T_cam_mesh_foundationpose'].astype(float)),('coco',r['T_cam_mesh_coconet'].astype(float))]:
  vals=[];valsv=[]
  for i in [0,100,200,300,350,400,450,545]:
   mk=rend(poses[i]);vals.append((mk&om[i]).sum()/(mk|om[i]).sum());vis=mk&~pm[i];valsv.append((vis&om[i]).sum()/(vis|om[i]).sum())
  print(Ktag,name,'raw',np.round(vals,3),np.mean(vals),'vis',np.round(valsv,3),np.mean(valsv))
'''
open(p,'w').write(prefix+tail)
PY
python /work/scratch/eval_rec_both.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_rec_both.py", line 9, in <module>
    K=np.array(cam[Ktag])
               ~~~^^^^^^
KeyError: 'Kp'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/for Ktag in \\['Kp','Kcam'\\]:/for Ktag,ky in [('Kp','K_pipeline'),('Kcam','K')]:/; s/K=np.array(cam\\[Ktag\\])/K=np.array(cam[ky])/\" /work/scratch/eval_rec_both.py
python /work/scratch/eval_rec_both.py"
```
<details><summary>output</summary>

```
Kp fp raw [0.74  0.763 0.793 0.787 0.313 0.606 0.552 0.639] 0.6492019822725066 vis [0.74  0.767 0.821 0.811 0.313 0.676 0.553 0.638] 0.6648836012193495
Kp coco raw [0.825 0.782 0.795 0.813 0.249 0.585 0.575 0.658] 0.6603566923477061 vis [0.825 0.787 0.825 0.849 0.305 0.673 0.58  0.659] 0.6881136692374559
Kcam fp raw [0.612 0.506 0.543 0.588 0.229 0.575 0.52  0.548] 0.5151787764331716 vis [0.612 0.557 0.594 0.629 0.229 0.628 0.52  0.566] 0.5419065588517908
Kcam coco raw [0.663 0.543 0.57  0.601 0.274 0.582 0.555 0.551] 0.5423319173189152 vis [0.663 0.587 0.611 0.642 0.326 0.638 0.555 0.582] 0.5756267608770891

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_world_ks.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Faces=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext()
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cam=json.load(open('/work/inputs/camera.json'));Tcw=np.linalg.inv(np.array(cam['T_world_cam']));T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);H,W=1152,1536
def eval(Ktag):
 K=np.array(cam[Ktag]);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];vals=[];valsv=[]
 for i in [0,100,200,300,350,400,450,545]:
  M=Tcw@T[i];v=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Faces,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vals.append((mk&om[i]).sum()/(mk|om[i]).sum());vis=mk&~pm[i];valsv.append((vis&om[i]).sum()/(vis|om[i]).sum())
 print(Ktag,np.round(vals,3),np.mean(vals),np.round(valsv,3),np.mean(valsv))
for k in ['K','K_pipeline']:eval(k)
PY
python /work/scratch/eval_world_ks.py"
```
<details><summary>output</summary>

```
K [0.536 0.454 0.463 0.461 0.337 0.349 0.326 0.354] 0.40996403531672965 [0.536 0.475 0.487 0.512 0.448 0.392 0.335 0.369] 0.4441650213093583
K_pipeline [0.615 0.557 0.542 0.564 0.31  0.383 0.365 0.407] 0.46785851944854967 [0.615 0.572 0.559 0.629 0.421 0.462 0.396 0.412] 0.5082570477491388

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;f=m.faces;uv=m.visual.uv;tex=np.array(m.visual.material.baseColorTexture);n=m.face_normals
# face uv centroid sample
uvc=uv[f].mean(1);px=np.clip((uvc[:,0]*tex.shape[1]).astype(int),0,tex.shape[1]-1);py=np.clip((uvc[:,1]*tex.shape[0]).astype(int),0,tex.shape[0]-1);col=tex[py,px]
for name,sel in [('yfront?', (n[:,1]>.9)),('yback?',(n[:,1]<-.9)),('xplus',(n[:,0]>.9)),('xminus',(n[:,0]<-.9))]:
 print(name,sel.sum(),'rgb mean med',col[sel].mean(0),np.median(col[sel],0),'bright frac',(col[sel].mean(1)>180).mean())
 # y coordinate face center
 print(' yc med',np.median(v[f[sel]].mean(1)[:,1]))
PY"
```
<details><summary>output</summary>

```
yfront? 18465 rgb mean med [44.66547522 54.31546168 52.45020309] [51. 66. 64.] bright frac 0.003520173300839426
 yc med -0.009032081502179304
yback? 20411 rgb mean med [44.18578218 50.65043359 47.76929107] [46. 53. 49.] bright frac 0.00044093870951937683
 yc med -0.02118561292688052
xplus 7491 rgb mean med [35.0989187  40.63542918 38.26525164] [33. 43. 40.] bright frac 0.007475637431584568
 yc med -0.011035824504991373
xminus 7728 rgb mean med [32.78377329 40.2801501  38.35998965] [10.5 13.5 11. ] bright frac 0.00012939958592132506
 yc med 0.021214076628287636

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for i in [0,100,180,220,260,300,320,340,360,380,400,450,545]:
 Q=(R.from_matrix(T[i,:3,:3])*R.from_matrix(T[0,:3,:3]).inv());rv=Q.as_rotvec();print(i,'rel angle',np.degrees(np.linalg.norm(rv)),'axis',np.round(rv/(np.linalg.norm(rv)+1e-9),3),'t',np.round(T[i,:3,3],3))
# relative world translation/rotation speeds in motion
for i in range(320,401,5):
 d=(R.from_matrix(T[i,:3,:3])*R.from_matrix(T[i-5,:3,:3]).inv()).as_rotvec();print(i,'5f dr',np.degrees(np.linalg.norm(d)),'axis',np.round(d/(np.linalg.norm(d)+1e-9),2),'dt',np.linalg.norm(T[i,:3,3]-T[i-5,:3,3]))
PY"
```
<details><summary>output</summary>

```
0 rel angle 0.0 axis [0. 0. 0.] t [0.331 0.11  2.583]
100 rel angle 0.6302794327274714 axis [-0.552 -0.236  0.8  ] t [0.33  0.126 2.619]
180 rel angle 1.8377206348714008 axis [ 0.982 -0.007  0.189] t [0.353 0.123 2.636]
220 rel angle 6.408943036517293 axis [0.66  0.646 0.382] t [0.362 0.116 2.615]
260 rel angle 45.89715839364766 axis [0.125 0.992 0.019] t [0.414 0.114 2.619]
300 rel angle 81.56541162448909 axis [0.055 0.998 0.009] t [0.524 0.101 2.625]
320 rel angle 104.1107605777607 axis [ 0.039  0.999 -0.005] t [0.626 0.104 2.646]
340 rel angle 133.02423670078983 axis [ 0.038  0.999 -0.016] t [0.666 0.096 2.604]
360 rel angle 90.8035315459332 axis [-0.145  0.76  -0.633] t [0.663 0.056 2.669]
380 rel angle 171.66024084268747 axis [-0.037  0.999  0.029] t [0.717 0.111 2.483]
400 rel angle  …[+1527 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);Q=R.from_matrix(T[:,:3,:3]*0+T[:,:3,:3]) # no
R0=R.from_matrix(T[0,:3,:3])
for i in range(150,421,5):
 d=(R.from_matrix(T[i,:3,:3])*R0.inv());rv=d.as_rotvec();ang=np.degrees(np.linalg.norm(rv));axis=rv/(np.linalg.norm(rv)+1e-12);signed=np.degrees(rv[1]);print(i,'signedY',round(signed,1),'angle',round(ang,1),'axis',np.round(axis,2))
PY"
```
<details><summary>output</summary>

```
150 signedY -0.3 angle 1.8 axis [ 0.16 -0.16 -0.97]
155 signedY -0.4 angle 1.8 axis [ 0.84 -0.21 -0.51]
160 signedY -0.1 angle 1.1 axis [ 0.65 -0.12 -0.75]
165 signedY 0.0 angle 1.1 axis [ 0.22  0.03 -0.98]
170 signedY 0.5 angle 1.1 axis [ 0.5   0.45 -0.74]
175 signedY 0.4 angle 1.4 axis [ 0.87  0.3  -0.38]
180 signedY -0.0 angle 1.8 axis [ 0.98 -0.01  0.19]
185 signedY -0.2 angle 1.2 axis [ 0.99 -0.14 -0.04]
190 signedY -0.4 angle 2.2 axis [ 0.98 -0.18  0.08]
195 signedY -0.3 angle 4.3 axis [ 0.91 -0.07  0.42]
200 signedY 0.7 angle 5.5 axis [0.85 0.12 0.52]
205 signedY 0.2 angle 5.4 axis [0.85 0.04 0.52]
210 signedY 0.4 angle 5.4 axis [0.85 0.08 0.52]
215 signedY 1.7 angle 5.5 axis [0.84 0.31 0.43]
220 signedY 4.1 angle 6.4 axis [0.66 0.65 0.38]
225 signedY 11.3 angle 12.2 axis [0.35 0.93 …[+2053 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_all_camera.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,time
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
r=np.load('/work/inputs/reconstruction.npz')
def run(key,out):
 P=r[key].astype(float);ious=[];areas=[];t0=time.time()
 for i in range(len(P)):
  T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];ious.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()));areas.append(mk.sum())
 ious=np.array(ious);np.save(out,ious);print(key,'time',time.time()-t0,'min med mean max',ious.min(),np.median(ious),ious.mean(),ious.max())
 for lo in range(0,546,20):print(lo,lo+19,round(ious[lo:lo+20].mean(),3),round(ious[lo:lo+20].min(),3))
 print('worst',[(int(i),round(float(ious[i]),3)) for i in np.argsort(ious)[:30]])
run('T_ca …[+149 chars]
```
<details><summary>output</summary>

```
T_cam_mesh_coconet time 2.708872079849243 min med mean max 0.18336771251764195 0.8105209064497116 0.737432367592342 0.8882735261808862
0 19 0.807 0.786
20 39 0.841 0.834
40 59 0.84 0.826
60 79 0.818 0.787
80 99 0.779 0.769
100 119 0.802 0.779
120 139 0.82 0.809
140 159 0.819 0.813
160 179 0.824 0.814
180 199 0.821 0.811
200 219 0.83 0.822
220 239 0.835 0.808
240 259 0.844 0.797
260 279 0.866 0.829
280 299 0.846 0.823
300 319 0.847 0.832
320 339 0.756 0.551
340 359 0.33 0.197
360 379 0.428 0.183
380 399 0.662 0.64
400 419 0.68 0.661
420 439 0.694 0.671
440 459 0.614 0.577
460 479 0.609 0.585
480 499 0.632 0.61
500 519 0.64 0.626
520 539 0.651 0.627
540 559 0.664 0.659
worst [(376, 0.183), (351, 0.197), (366, 0.236), (347, 0.246), (355, 0.248), (356, 0.252), (346, 0.254), (353, 0.26), (377,  …[+1409 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/diag_coco.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
from PIL import Image,ImageDraw
A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform']);m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices@A[:3,:3].T+A[:3,3],dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536;r=np.load('/work/inputs/reconstruction.npz');P=r['T_cam_mesh_coconet'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cols=torch.zeros((len(V),3),device='cuda');cols[:,0]=(V[:,0]-V[:,0].min())/(V[:,0].max()-V[:,0].min());cols[:,1]=(V[:,1]-V[:,1].min())/(V[:,1].max()-V[:,1].min());cols[:,2]=(V[:,2]-V[:,2].min())/(V[:,2].max()-V[:,2].min())
def rend(i):
 T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();rgb=dr.interpolate(cols[None],rast,Fa)[0][0].cpu().numpy();return mk,rgb
cap=cv2.Video …[+752 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/depth_eval.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform']);m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices@A[:3,:3].T+A[:3,3],dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
r=np.load('/work/inputs/reconstruction.npz');P=r['T_cam_mesh_coconet'].astype(float)
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r');dd=f['aligned/front_stereo_camera_left']
def rend(i):
 T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W);zb=dr.interpolate(z[:,None,None].expand(-1,1,1),rast,Fa)[0][0,...,0];return mk,zb
for i in [0,100,200,300,340,350,360,370,380,400,450,545]:
 mk,zb=rend(i);d=cv2.imdecode(dd[i],cv2.IMREAD_UNCHANGED).astype(float)/1000.;sel=(mk.cpu().numpy()&om[i]&~pm[i]);res=(zb …[+266 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/depth_eval.py", line 9, in <module>
    mk,zb=rend(i);d=cv2.imdecode(dd[i],cv2.IMREAD_UNCHANGED).astype(float)/1000.;sel=(mk.cpu().numpy()&om[i]&~pm[i]);res=(zb.cpu().numpy()-d)[sel];print(i,'pix',sel.sum(),'depth obj',np.median(d[om[i]]),'render',np.median(zb.cpu().numpy()[mk.cpu().numpy()]),'res med mean p10 p90',np.median(res),res.mean(),np.percentile(res,10),np.percentile(res,90))
          ^^^^^^^
  File "/work/scratch/depth_eval.py", line 7, in rend
    T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa …[+1224 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/zb=dr.interpolate(z\\[:,None,None\\].expand(-1,1,1),rast,Fa)\\[0\\]\\[0,...,0\\]/zb=dr.interpolate(z[:,None].contiguous(),rast,Fa)[0][0,...,0]/\" /work/scratch/depth_eval.py
python /work/scratch/depth_eval.py"
```
<details><summary>output</summary>

```
0 pix 5547 depth obj 2.763 render 2.5601354 res med mean p10 p90 -0.1501489963531495 -0.19204505131716165 -0.5136749774932861 0.07087789039611839
100 pix 5393 depth obj 2.637 render 2.578671 res med mean p10 p90 0.013128810882568231 -0.016726842561944965 -0.34359652633667004 0.23577398490905785
200 pix 6009 depth obj 2.561 render 2.5289216 res med mean p10 p90 0.025110551834106598 0.00557155397764001 -0.30812648658752434 0.2375323513031004
300 pix 7089 depth obj 2.606 render 2.5575132 res med mean p10 p90 0.029564765930175874 0.0022734504354775227 -0.24418988571166986 0.23188949356079103
340 pix 1876 depth obj 2.765 render 2.5586832 res med mean p10 p90 -0.08155808162689215 -0.09868761898662998 -0.2921338558197022 0.09589663124084469
350 pix 2601 depth obj 4.504 render 2.681623 res med mea …[+965 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/texture_pose.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
from PIL import Image,ImageDraw
A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform']);m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices@A[:3,:3].T+A[:3,3],dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda');ctx=dr.RasterizeCudaContext();tex=torch.tensor(np.array(m.visual.material.baseColorTexture),dtype=torch.float32,device='cuda').permute(2,0,1)[None]/255.;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536;r=np.load('/work/inputs/reconstruction.npz');P=r['T_cam_mesh_coconet'].astype(float)
def rend(i,flipv):
 T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];uv=dr.interpolate(UV[None],rast,Fa)[0][0];u=uv[...,0];vv=uv[...,1];vv=1-vv if flipv else vv;s=torch.stack([2*u-1,2*vv-1],-1)[None];rgb=torch.nn.functional.grid_sample(tex,s,mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1, …[+805 chars]
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
r=np.load('/work/inputs/reconstruction.npz')
for key in ['T_cam_mesh_coconet','T_world_mesh','T_cam_mesh_foundationpose']:
 P=r[key].astype(float)
 print('\\n',key)
 for name,a,b in [('init',0,180),('late',390,546),('final',450,546)]:
  Rm=R.from_matrix(P[a:b,:3,:3]);mean=Rm.mean().as_matrix();rel=np.degrees(np.linalg.norm((Rm*R.from_matrix(mean).inv()).as_rotvec(),axis=1));print(name,'t std',np.round(P[a:b,:3,3].std(0),4),'med',np.round(np.median(P[a:b,:3,3],0),4),'rot std/relmax',np.round(np.std(Rm.as_rotvec(0),axis=0),4),rel.max())
PY"
```
<details><summary>output</summary>

```

 T_cam_mesh_coconet
init t std [0.0041 0.0057 0.024 ] med [0.1782 0.0278 2.6211] rot std/relmax [0.0129 0.0118 0.0117] 2.2632806531892147
late t std [0.0125 0.0088 0.0388] med [0.5271 0.0489 2.4915] rot std/relmax [0.0324 0.0372 0.0264] 5.753463435122293
final t std [0.0085 0.0053 0.0479] med [0.5229 0.0535 2.4994] rot std/relmax [0.0265 0.0334 0.0245] 5.8820731584842605

 T_world_mesh
init t std [0.0063 0.0079 0.0309] med [0.3328 0.1223 2.6391] rot std/relmax [0.0125 0.0118 0.012 ] 2.2599685275182892
late t std [0.0119 0.0095 0.0335] med [0.689  0.1334 2.5133] rot std/relmax [0.0325 0.0382 0.0253] 5.751978088636479
final t std [0.0099 0.0049 0.0382] med [0.6854 0.1373 2.5361] rot std/relmax [0.0257 0.0354 0.0227] 5.8803271406160444

 T_cam_mesh_foundationpose
init t std [0.0051 0.0092 0. …[+323 chars]
```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,time
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
A=np.array(json.load(open('/pipeline/export/episode_000006/wild_export.json'))['source_object_mesh_to_aligned_transform']);m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices@A[:3,:3].T+A[:3,3],dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float)
def rotvec_mat(v):return Rot.from_rotvec(v).as_matrix()
def render(R,t):
 Rt=torch.tensor(R,dtype=torch.float32,device='cuda');tt=torch.tensor(t,dtype=torch.float32,device='cuda');v=V@Rt.T+tt;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
def optimize(name,idxs,R0,t0,bounds):
 n=[0];tstart=time.time()
 def fun(x):
  n[0]+=1;R0d=R0@rotvec_mat(x[:3]);t=t0+x[3:];mk=render(R0d,t);vals=[]
  for i in idxs:
   vis=mk&~pm[i];vals.append(( …[+830 chars]
```
<details><summary>output</summary>

```
init robust [[ 0.00442628 -0.73912995  0.67354831]
 [ 0.99692596 -0.04942725 -0.06079124]
 [ 0.07822427  0.67174687  0.73663906]] [0.17838228 0.02738629 2.61934042]
init 50 cost 0.6441920459310819 x [ 0.03694452 -0.14756081 -0.14756059  0.          0.          0.        ] meanIoU 0.35580795406891813
init 100 cost 0.6200248395563261 x [ 0.01981419 -0.07914023 -0.07917961 -0.04235437 -0.01058347 -0.07073362] meanIoU 0.37997516044367385
init 150 cost 0.5630711261262431 x [ 0.14364746 -0.14512163 -0.14771458 -0.07901476 -0.01974413 -0.13195806] meanIoU 0.43692887387375695
init 200 cost 0.5563855414248424 x [ 0.14364746 -0.14596806 -0.10491503 -0.07901476 -0.02153678 -0.14827722] meanIoU 0.4436144585751575
init 250 cost 0.5560119621483102 x [ 0.14533533 -0.14574169 -0.10401867 -0.0789082  -0.02 …[+1940 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float)
idx=list(range(0,181,5));Rm=Rot.from_matrix(C[idx,:3,:3]);mean=Rm.mean()
print('C0\\n',C[0,:3,:3]);print('mean\\n',mean.as_matrix());print('rel mean deg',np.degrees(np.linalg.norm((Rm*mean.inv()).as_rotvec(),axis=1)).max())
print('quat avg direct',Rot.from_quat(np.mean(Rm.as_quat(),0)).as_matrix())
PY"
```
<details><summary>output</summary>

```
C0
 [[ 0.00190316 -0.73803514  0.67475957]
 [ 0.99709642 -0.04996593 -0.05746378]
 [ 0.07612527  0.67290968  0.73579711]]
mean
 [[ 0.00442628 -0.73912995  0.67354831]
 [ 0.99692596 -0.04942725 -0.06079124]
 [ 0.07822427  0.67174687  0.73663906]]
rel mean deg 1.835127811491396
quat avg direct [[ 0.00442637 -0.7391299   0.67354836]
 [ 0.99692596 -0.04942717 -0.06079128]
 [ 0.07822424  0.67174693  0.73663901]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# Load relevant code then evaluate mean iou at robust pose for init and final
exec(open('/work/scratch/opt_static.py').read().split('def optimize')[0])
for name,idxs in [('init',list(range(0,181,5))),('final',list(range(400,546,5)))]:
 R0=Rot.from_matrix(C[idxs,:3,:3]).mean().as_matrix();t0=np.median(C[idxs,:3,3],0);mk=render(R0,t0);v=[]
 for i in idxs:
  vis=mk&~pm[i];v.append((vis&om[i]).sum()/(vis|om[i]).sum())
 print(name,np.mean(v),np.percentile(v,[0,10,50,90,100]))
PY"
```
<details><summary>output</summary>

```
init 0.33123607670989463 [0.29140749 0.2923223  0.34773572 0.35676308 0.36316291]
final 0.22649153782224332 [0.20778001 0.21469574 0.22790965 0.2343394  0.23665468]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/opt_static.py').read().split('def optimize')[0])
idx=0
for name,R,t in [('C0',C[0,:3,:3],C[0,:3,3]),('rob',Rot.from_matrix(C[list(range(0,181,5)),:3,:3]).mean().as_matrix(),np.median(C[list(range(0,181,5)),:3,3],0))]:
 mk=render(R,t);vis=mk&~pm[idx];print(name,(vis&om[idx]).sum()/(vis|om[idx]).sum(),mk.sum(),R,t)
PY"
```
<details><summary>output</summary>

```
C0 0.2932903294030561 8930 [[ 0.00190316 -0.73803514  0.67475957]
 [ 0.99709642 -0.04996593 -0.05746378]
 [ 0.07612527  0.67290968  0.73579711]] [0.18150283 0.02103061 2.62010002]
rob 0.2914074855034265 8987 [[ 0.00442628 -0.73912995  0.67354831]
 [ 0.99692596 -0.04942725 -0.06079124]
 [ 0.07822427  0.67174687  0.73663906]] [0.17838228 0.02738629 2.61934042]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/V=torch.tensor(m.vertices@A\\[:3,:3\\].T+A\\[:3,3\\],dtype=torch.float32,device='cuda')/V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda')/\" /work/scratch/opt_static.py
python /work/scratch/opt_static.py"
```
<details><summary>output</summary>

```
init robust [[ 0.00442628 -0.73912995  0.67354831]
 [ 0.99692596 -0.04942725 -0.06079124]
 [ 0.07822427  0.67174687  0.73663906]] [0.17838228 0.02738629 2.61934042]
init 50 cost 0.16569279203957354 x [ 0.0355128  -0.0159829  -0.0112245  -0.00701188  0.          0.        ] meanIoU 0.8343072079604265
init 100 cost 0.15208250082706054 x [ 0.01379106 -0.0159829  -0.0112245  -0.002122   -0.01102571 -0.02649594] meanIoU 0.8479174991729395
init 150 cost 0.14482571795692922 x [ 0.02400276 -0.01562517 -0.00948502 -0.00629422 -0.01102571 -0.02649594] meanIoU 0.8551742820430708
init 200 cost 0.13774028142293093 x [ 0.00626813 -0.0149245  -0.00607803 -0.01396856 -0.00576976 -0.08324116] meanIoU 0.8622597185770691
init 250 cost 0.13205912635340378 x [ 0.01071856  0.01590091  0.02197372 -0.01396856 -0. …[+4255 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/diag_coco_input.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
from PIL import Image,ImageDraw
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536;r=np.load('/work/inputs/reconstruction.npz');P=r['T_cam_mesh_coconet'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cols=torch.zeros((len(V),3),device='cuda');cols[:,0]=(V[:,0]-V[:,0].min())/(V[:,0].max()-V[:,0].min());cols[:,1]=(V[:,1]-V[:,1].min())/(V[:,1].max()-V[:,1].min());cols[:,2]=(V[:,2]-V[:,2].min())/(V[:,2].max()-V[:,2].min())
def rend(i):
 T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();rgb=dr.interpolate(cols[None],rast,Fa)[0][0].cpu().numpy();return mk,rgb
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [0,200,300,330,340,345,350,355,360,365,370,375,380,385,390,400,450,545]:
 cap.set(cv2. …[+632 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
c=np.load('/work/scratch/coco_iou.npy');f=np.load('/work/scratch/fp_iou.npy')
for i in range(300,401):print(i,round(float(c[i]),3),round(float(f[i]),3))
PY"
```
<details><summary>output</summary>

```
300 0.849 0.811
301 0.851 0.806
302 0.84 0.79
303 0.852 0.796
304 0.856 0.81
305 0.86 0.812
306 0.846 0.801
307 0.85 0.804
308 0.843 0.802
309 0.843 0.804
310 0.842 0.798
311 0.832 0.791
312 0.842 0.811
313 0.84 0.801
314 0.853 0.81
315 0.863 0.826
316 0.84 0.809
317 0.841 0.811
318 0.841 0.811
319 0.854 0.84
320 0.849 0.835
321 0.855 0.843
322 0.834 0.832
323 0.841 0.839
324 0.851 0.83
325 0.851 0.837
326 0.842 0.819
327 0.846 0.817
328 0.844 0.828
329 0.849 0.842
330 0.733 0.749
331 0.731 0.742
332 0.721 0.73
333 0.718 0.738
334 0.703 0.704
335 0.653 0.685
336 0.631 0.673
337 0.644 0.688
338 0.564 0.613
339 0.551 0.597
340 0.551 0.607
341 0.45 0.413
342 0.484 0.333
343 0.409 0.267
344 0.401 0.261
345 0.277 0.043
346 0.254 0.247
347 0.246 0.248
348 0.264 0.266
349 0.319 0.283
350 0.305 0. …[+793 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_pose_one.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2,time
from scipy.spatial.transform import Rotation as Rot
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float)
def render(R,t):
 Rt=torch.tensor(R,dtype=torch.float32,device='cuda');tt=torch.tensor(t,dtype=torch.float32,device='cuda');v=V@Rt.T+tt;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
def score(R,t,i):
 mk=render(R,t);vis=mk&~pm[i];return (vis&om[i]).sum()/max(1,(vis|om[i]).sum())
def exp(v):return Rot.from_rotvec(v).as_matrix()
def interp_pose(i,a,b):
 u=(i-a)/(b-a);Ra=Rot.from_matrix(C[a,:3,:3]);Rb=Rot.from_matrix(C[b,:3,:3]);R=(Ra*Rot.from_rotvec(u*(Rb*Ra.inv()).as_rotvec())).as_matrix();t=(1-u)*C[a,:3,3]+u*C[b,:3,3];return R,t
def cd(i,R,t,name):
 best=score(R,t,i);print(name,'start',best)
 steps …[+907 chars]
```
<details><summary>output</summary>

```
coco start 0.7333233622494765
coco step 0.12 0 1 0.7424388793936242
coco step 0.12 1 -1 0.790786665685481
coco level 0.12 0.790786665685481
coco level 0.1 0.790786665685481
coco level 0.06 0.790786665685481
coco level 0.035 0.790786665685481
coco step 0.02 4 -1 0.7960473146719566
coco step 0.02 5 1 0.7969483568075117
coco step 0.02 0 -1 0.7998968160377359
coco step 0.02 1 1 0.8032507050616001
coco step 0.02 2 1 0.8041428465364912
coco step 0.02 0 -1 0.8063217533919785
coco step 0.02 0 -1 0.8073010173548774
coco level 0.02 0.8073010173548774
coco step 0.01 5 1 0.8101938477658858
coco step 0.01 5 1 0.8115735888606079
coco step 0.01 5 1 0.8116907417109498
coco step 0.01 0 -1 0.8125046855086588
coco step 0.01 1 1 0.8135248984504287
coco step 0.01 0 -1 0.8138658628485306
coco step 0.01 2 1 0.81 …[+28481 chars]
```
</details>

**shell** (exit 130)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_scan.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,time
from scipy.spatial.transform import Rotation as Rot
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float)
st=np.load('/work/scratch/static_init.npz');R0=st['R'];t0=st['t']
Twc=np.array(cam['T_world_cam']);axis=Twc[:3,:3].T@np.array([0.,1.,0.]);axis/=np.linalg.norm(axis);print('axis',axis,'R0',R0,'t0',t0)
def render(R,t):
 Rt=torch.tensor(R,dtype=torch.float32,device='cuda');tt=torch.tensor(t,dtype=torch.float32,device='cuda');v=V@Rt.T+tt;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
def score(R,t,i):
 mk=render(R,t);vis=mk&~pm[i];return (vis&om[i]).sum()/max(1,(vis|om[i]).sum())
def align(R,t,i):
 best=score(R,t,i)
 for step in [.08,.04,.02,.01,.005]:
  for j in range(3):
   for it in range(2):
    moved=False …[+1024 chars]
```
<details><summary>output</summary>

```
axis [-0.0080006   0.99794105  0.06363684] R0 [[-0.01176176 -0.74877179  0.66272352]
 [ 0.99365625 -0.08288363 -0.07601025]
 [ 0.11184326  0.65762536  0.74499663]] t0 [0.16239873 0.02507527 2.48967726]
FRAME 190 best 0.8798613785136696 32 [0.109 0.023 2.515] time 13.55941891670227
 top [(32, 0.88), (18, 0.876), (4, 0.873), (44, 0.872), (56, 0.863), (68, 0.841), (200, 0.838), (88, 0.832)]
FRAME 220 best 0.8850045440775522 8 [0.2   0.014 2.552] time 15.163619995117188
 top [(8, 0.885), (42, 0.882), (28, 0.881), (54, 0.873), (70, 0.864), (200, 0.862), (82, 0.859), (188, 0.837)]
FRAME 260 best 0.8832344676865439 60 [0.251 0.017 2.496] time 13.942174196243286
 top [(60, 0.883), (48, 0.879), (72, 0.862), (36, 0.857), (200, 0.829), (24, 0.82), (84, 0.809), (12, 0.753)]
FRAME 300 best 0.8873528174 …[+991 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/yaw_depth.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2,time
from scipy.spatial.transform import Rotation as Rot
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);st=np.load('/work/scratch/static_init.npz');R0=st['R'];Twc=np.array(cam['T_world_cam']);axis=Twc[:3,:3].T@np.array([0.,1.,0.]);axis/=np.linalg.norm(axis);dd=h5py.File('/pipeline/depth/moge2/aligned.h5','r')['aligned/front_stereo_camera_left']
def render(R,t):
 Rt=torch.tensor(R,dtype=torch.float32,device='cuda');tt=torch.tensor(t,dtype=torch.float32,device='cuda');v=V@Rt.T+tt;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();zb=dr.interpolate(z[:,None].contiguous(),rast,Fa)[0][0,...,0].cpu().numpy();return mk,zb
def metrics(R,t,i,target):
 mk,zb=render(R,t);vis=mk&~pm[i];iou=(vis&om[i]).sum()/max(1,(vis|om[i] …[+1492 chars]
```
<details><summary>output</summary>

```

FRAME 320
iou best [(70, 0.888, 0.12, 0.077), (65, 0.888, 0.126, 0.084), (75, 0.887, 0.115, 0.07), (60, 0.886, 0.129, 0.091), (80, 0.883, 0.147, 0.062), (90, 0.883, 0.07, 0.043), (85, 0.882, 0.12, 0.053), (55, 0.882, 0.163, 0.098), (50, 0.878, 0.17, 0.104), (95, 0.874, 0.137, 0.036)]
mae best [(105, 0.869, 0.038, 0.034), (100, 0.873, 0.048, 0.031), (90, 0.883, 0.07, 0.043), (125, 0.579, 0.079, 0.078), (120, 0.644, 0.085, 0.069), (130, 0.51, 0.089, 0.086), (135, 0.437, 0.103, 0.097), (110, 0.83, 0.107, 0.043), (75, 0.887, 0.115, 0.07), (140, 0.339, 0.116, 0.104)]
bias best [(150, 0.29, 0.323, 0.14), (155, 0.379, 0.295, 0.144), (160, 0.469, 0.28, 0.148), (175, 0.705, 0.23, 0.149), (185, 0.832, 0.233, 0.152), (50, 0.878, 0.17, 0.104), (55, 0.882, 0.163, 0.098), (165, 0.563, 0.209, 0.146), (4 …[+12904 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
a=np.load('/work/scratch/static_init.npz');b=np.load('/work/scratch/static_final.npz');R0=a['R'];Rf=b['R'];d=Rot.from_matrix(Rf@R0.T);rv=d.as_rotvec();print('relative angle',np.degrees(np.linalg.norm(rv)),'axis',rv/np.linalg.norm(rv),'deg vec',np.degrees(rv));print('t0',a['t'],'tf',b['t'])
# in world
import json
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);dw=Rot.from_matrix(Twc[:3,:3]@Rf@R0.T@Twc[:3,:3].T);print('world rel',np.degrees(dw.as_rotvec()),dw.as_rotvec()/np.linalg.norm(dw.as_rotvec()))
PY"
```
<details><summary>output</summary>

```
relative angle 176.71484195422653 axis [-0.04031138 -0.99465147 -0.09509708] deg vec [  -7.12361884 -175.7696766   -16.80506501]
t0 [0.16239873 0.02507527 2.48967726] tf [0.51242841 0.03582844 2.4505696 ]
world rel [  -8.87602006 -176.4202039    -5.02626024] [-0.05022793 -0.99833269 -0.02844277]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float)
for i in range(320,396,2):print(i,np.round(C[i,:3,3],3))
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
r=np.load('/work/inputs/reconstruction.npz');K=r['human_keypoints_world'];C=r['contact_prob_right_left'];cam=json.load(open('/work/inputs/camera.json'));Tcw=np.linalg.inv(np.array(cam['T_world_cam']));T=r['T_world_mesh'].astype(float)
for i in range(300,401,5):
 hr=K[i,21:42].mean(0);hl=K[i,42:63].mean(0);print(i,'handRW',np.round(hr,3),'handLW',np.round(hl,3),'prob',np.round(C[i],2),'objW',np.round(T[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
300 handRW [0.479 0.343 2.732] handLW [0.632 0.363 2.598] prob [1. 1.] objW [0.524 0.101 2.625]
305 handRW [0.516 0.349 2.702] handLW [0.63  0.353 2.544] prob [1. 1.] objW [0.542 0.095 2.579]
310 handRW [0.548 0.359 2.786] handLW [0.687 0.359 2.61 ] prob [1. 1.] objW [0.568 0.104 2.634]
315 handRW [0.559 0.331 2.703] handLW [0.706 0.362 2.585] prob [1. 1.] objW [0.589 0.101 2.617]
320 handRW [0.601 0.339 2.729] handLW [0.747 0.372 2.642] prob [1. 1.] objW [0.626 0.104 2.646]
325 handRW [0.662 0.372 2.728] handLW [0.74  0.365 2.587] prob [1. 1.] objW [0.644 0.106 2.588]
330 handRW [0.711 0.402 2.798] handLW [0.767 0.385 2.588] prob [1. 1.] objW [0.665 0.114 2.597]
335 handRW [0.734 0.407 2.765] handLW [0.758 0.392 2.559] prob [1. 1.] objW [0.67  0.113 2.594]
340 handRW [0.766 0.41  2.682] h …[+1220 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float)
for i in [0,300,330,340,350,360,370,380,545]:
 R=C[i,:3,:3]
 print(i,'normal -y',np.round(R@np.array([0,-1,0.]),3),'normal +y',np.round(R@np.array([0,1,0.]),3),'mesh x',np.round(R[:,0],3),'mesh z',np.round(R[:,2],3))
PY"
```
<details><summary>output</summary>

```
0 normal -y [ 0.738  0.05  -0.673] normal +y [-0.738 -0.05   0.673] mesh x [0.002 0.997 0.076] mesh z [ 0.675 -0.057  0.736]
300 normal -y [-0.558  0.125 -0.821] normal +y [ 0.558 -0.125  0.821] mesh x [0.021 0.99  0.136] mesh z [ 0.83   0.058 -0.555]
330 normal -y [-0.941  0.086 -0.328] normal +y [ 0.941 -0.086  0.328] mesh x [0.045 0.991 0.129] mesh z [ 0.336  0.107 -0.936]
340 normal -y [-0.995  0.065 -0.074] normal +y [ 0.995 -0.065  0.074] mesh x [0.059 0.995 0.085] mesh z [ 0.079  0.08  -0.994]
350 normal -y [-0.859 -0.159 -0.487] normal +y [0.859 0.159 0.487] mesh x [ 0.396  0.399 -0.827] mesh z [ 0.326 -0.903 -0.28 ]
360 normal -y [-0.564 -0.205 -0.8  ] normal +y [0.564 0.205 0.8  ] mesh x [ 0.559  0.618 -0.552] mesh z [ 0.607 -0.759 -0.234]
370 normal -y [-0.217 -0.296 -0.93 ] nor …[+344 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:];p=h['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');gray=[];rgb=[]
for i in range(300,401):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);sel=o[i]&~p[i];v=im[sel];gray.append(v.mean());rgb.append(np.median(v,0));
 if i%5==0:print(i,'area',sel.sum(),'mean',np.round(v.mean(),1),'med',np.round(np.median(v,0),0),'bright>140',round((v.mean(1)>140).mean(),2))
PY"
```
<details><summary>output</summary>

```
300 area 17511 mean 158.9 med [178. 179. 173.] bright>140 0.87
305 area 16921 mean 156.0 med [177. 178. 172.] bright>140 0.85
310 area 16493 mean 157.0 med [177. 178. 170.] bright>140 0.86
315 area 15727 mean 155.5 med [177. 178. 171.] bright>140 0.84
320 area 13951 mean 152.0 med [176. 177. 171.] bright>140 0.81
325 area 12053 mean 149.0 med [174. 175. 169.] bright>140 0.78
330 area 12633 mean 133.9 med [172. 172. 164.] bright>140 0.66
335 area 11171 mean 123.2 med [169. 168. 161.] bright>140 0.58
340 area 7531 mean 114.0 med [122. 123. 121.] bright>140 0.48
345 area 4482 mean 54.3 med [50. 50. 48.] bright>140 0.02
350 area 4859 mean 52.7 med [46. 46. 45.] bright>140 0.03
355 area 5492 mean 49.0 med [39. 39. 38.] bright>140 0.03
360 area 6465 mean 41.7 med [26. 27. 26.] bright>140 0.02
36 …[+464 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float)
for i in range(338,351):
 R=C[i,:3,:3];print(i,np.round(R@np.array([0,1.,0]),3),np.round(R@np.array([0,-1.,0]),3),np.round(C[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
338 [ 0.992 -0.068  0.104] [-0.992  0.068 -0.104] [0.513 0.024 2.593]
339 [ 0.995 -0.06   0.076] [-0.995  0.06  -0.076] [0.508 0.024 2.588]
340 [ 0.995 -0.065  0.074] [-0.995  0.065 -0.074] [0.505 0.006 2.596]
341 [ 0.998 -0.057 -0.006] [-0.998  0.057  0.006] [ 0.489 -0.004  2.552]
342 [ 0.997 -0.079 -0.015] [-0.997  0.079  0.015] [ 0.491 -0.009  2.582]
343 [ 0.996 -0.079 -0.043] [-0.996  0.079  0.043] [ 0.485 -0.013  2.573]
344 [ 0.996 -0.071 -0.045] [-0.996  0.071  0.045] [0.493 0.009 2.58 ]
345 [ 0.954 -0.087  0.288] [-0.954  0.087 -0.288] [ 0.434 -0.063  2.627]
346 [ 0.936 -0.089  0.342] [-0.936  0.089 -0.342] [ 0.45  -0.07   2.711]
347 [ 0.936 -0.103  0.338] [-0.936  0.103 -0.338] [ 0.457 -0.09   2.687]
348 [0.862 0.12  0.492] [-0.862 -0.12  -0.492] [ 0.506 -0.033  2.685]
349 [0.889 0 …[+128 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_static_poses.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Faces=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float)
def rend(R,t):
 Rt=torch.tensor(R,dtype=torch.float32,device='cuda');tt=torch.tensor(t,dtype=torch.float32,device='cuda');v=V@Rt.T+tt;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Faces,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
def sc(R,t,i):
 mk=rend(R,t);vis=mk&~pm[i];return (vis&om[i]).sum()/(vis|om[i]).sum()
si=np.load('/work/scratch/static_init.npz');sf=np.load('/work/scratch/static_final.npz')
for name,R,t in [('init_opt',si['R'],si['t']),('final_opt',sf['R'],sf['t'])]:
 print(name)
 for a,b in [(0,190),(180,220),(300,345),(340,381),(380,450),(390,546)]:
  vals=[sc(R,t,i) for i in range(a,b,5)];print(a,b,round(np.mean(vals),3),round(np.min(vals),3),round(np.max(vals),3))
 print('vs coco') …[+174 chars]
```
<details><summary>output</summary>

```
init_opt
0 190 0.878 0.83 0.925
180 220 0.809 0.784 0.837
300 345 0.139 0.0 0.326
340 381 0.0 0.0 0.002
380 450 0.021 0.002 0.03
390 546 0.028 0.014 0.035
vs coco
180 opt 0.83 coco 0.82
200 opt 0.787 coco 0.825
220 opt 0.782 coco 0.836
300 opt 0.326 coco 0.849
340 opt 0.0 coco 0.551
380 opt 0.002 coco 0.662
400 opt 0.024 coco 0.673
450 opt 0.029 coco 0.58
545 opt 0.035 coco 0.659
final_opt
0 190 0.034 0.031 0.058
180 220 0.071 0.054 0.09
300 345 0.612 0.406 0.785
340 381 0.409 0.218 0.62
380 450 0.729 0.62 0.775
390 546 0.714 0.679 0.775
vs coco
180 opt 0.058 coco 0.82
200 opt 0.082 coco 0.825
220 opt 0.093 coco 0.836
300 opt 0.478 coco 0.849
340 opt 0.406 coco 0.551
380 opt 0.62 coco 0.662
400 opt 0.74 coco 0.673
450 opt 0.688 coco 0.58
545 opt 0.721 coco 0.659

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/k_compare.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float);H,W=1152,1536
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [0,300,400,545]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);row=[]
 for kk in ['K','K_pipeline']:
  K=np.array(cam[kk]);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];T=C[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();
  a=im.copy();a[mk]=[0,0,255];b=im.copy();b[om[i]]=[255,0,0];b[mk]=[0,0,255];b[om[i]&mk]=[255,0,255];x0,y0,x1,y1=700,450,1050,710;q=np.concatenate([b[y0:y1,x0:x1]],1);q=cv2.resize(q,(350,260));cv2.putText(q,kk,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2);row.append(q)
 ims.app …[+149 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed 's/C=np.load.*/T=np.load(\"\\/work\\/inputs\\/reconstruction.npz\")[\"T_world_mesh\"].astype(float);Tcm=np.linalg.inv(np.array(cam[\"T_world_cam\"]));C=np.einsum(\"ij,fjk->fik\",Tcm,T)/' /work/scratch/k_compare.py | sed 's/for kk in \\['\"'\"'K'\"'\"','\"'\"'K_pipeline'\"'\"'\\]:/for kk in ['\"'\"'K'\"'\"','\"'\"'K_pipeline'\"'\"']:/' > /work/scratch/kcompare_world.py
python /work/scratch/kcompare_world.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/kcompare_world.py", line 9, in <module>
    K=np.array(cam[kk]);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];T=C[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();
                                                                                                                                                                                                                        ^
NameError: name 'W' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/kcompare_world.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
from PIL import Image
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));H,W=1152,1536;T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));C=np.array([Tcw@p for p in T])
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [0,300,400,545]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);row=[]
 for kk in ['K','K_pipeline']:
  K=np.array(cam[kk]);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];P=C[i];v=V@torch.tensor(P[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(P[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();b=im.copy();b[om[i]]=[255,0,0];b[mk]=[0,0,255];b[om[i]&mk]=[255,0,255];x0,y0,x1,y1=700,450,1050,710;q=cv2.resize(b[y0:y1,x0:x1],(350,260));cv2.putText(q,kk,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),2) …[+183 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# calculate iou for raw coco transformed by Twc with K/Kp and baseline world
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));H,W=1152,1536;Tcw=np.linalg.inv(np.array(cam['T_world_cam']));Twc=np.array(cam['T_world_cam']);r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);T=r['T_world_mesh'].astype(float)
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:];p=h['person'][:]
poses={'base':T,'TwcC':np.array([Twc@c for c in C])}
for name,P in poses.items():
 for kk in ['K','K_pipeline']:
  K=np.array(cam[kk]);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];vals=[];valsv=[]
  for i in [0,100,300,340,380,400,450,545]:
   M=Tcw@P[i];v=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vals.append((mk&o[i]).sum()/(mk|o[i]).sum());vis=mk&~p[i];valsv.append((vis&o[i]).sum()/(vis|o[i]).sum())
  print(name,kk,'raw',np.round(vals,3),np.mean(vals),'vi …[+40 chars]
```
<details><summary>output</summary>

```
base K raw [0.536 0.454 0.461 0.413 0.292 0.349 0.326 0.354] 0.3979791995408603 vis [0.536 0.475 0.512 0.447 0.332 0.392 0.335 0.369] 0.424719287641261
base K_pipeline raw [0.615 0.557 0.564 0.315 0.328 0.383 0.365 0.407] 0.44166097242086455 vis [0.615 0.572 0.629 0.353 0.404 0.462 0.396 0.412] 0.48038750216900694
TwcC K raw [0.663 0.543 0.601 0.528 0.528 0.582 0.555 0.551] 0.5688112382710873 vis [0.663 0.587 0.642 0.559 0.589 0.638 0.555 0.582] 0.6018778445520614
TwcC K_pipeline raw [0.825 0.782 0.813 0.513 0.562 0.585 0.575 0.658] 0.6642201544355568 vis [0.825 0.787 0.849 0.551 0.662 0.673 0.58  0.659] 0.698372512407285

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');_,im=cap.read();
for x,y in [(800,750),(700,800),(1000,820),(600,700),(900,900),(750,880)]:
 b,g,r=im[y,x];h,s,v=cv2.cvtColor(np.uint8([[[b,g,r]]]),cv2.COLOR_BGR2HSV)[0,0];print((x,y),'bgr',(b,g,r),'hsv',(h,s,v))
PY"
```
<details><summary>output</summary>

```
(800, 750) bgr (99, 109, 108) hsv (33, 23, 109)
(700, 800) bgr (83, 88, 86) hsv (42, 14, 88)
(1000, 820) bgr (110, 118, 117) hsv (34, 17, 118)
(600, 700) bgr (86, 94, 91) hsv (41, 22, 94)
(900, 900) bgr (82, 87, 87) hsv (30, 15, 87)
(750, 880) bgr (78, 83, 81) hsv (42, 15, 83)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/table_plane.py <<'PY'
import cv2,h5py,numpy as np,json,open3d as o3d
cap=cv2.VideoCapture('/work/inputs/video.mp4');_,im=cap.read();hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][0];p=h['person'][0]
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r');d=cv2.imdecode(f['aligned/front_stereo_camera_left'][0],cv2.IMREAD_UNCHANGED).astype(float)/1000.
H,W=d.shape;ys,xs=np.indices((H,W));
roi=(xs>550)&(xs<1150)&(ys>600)&(ys<950)&~o&~p
color=(hsv[...,1]<55)&(hsv[...,0]>20)&(hsv[...,0]<50)&(hsv[...,2]>65)&(hsv[...,2]<160)
sel=roi&color&(d>0.5)&(d<6)
print('sel',sel.sum());cv2.imwrite('/work/scratch/tablemask.png',(sel*255).astype('uint8'))
cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);u=xs[sel];v=ys[sel];z=d[sel];P=np.stack([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z],1);Pw=P@Twc[:3,:3].T+Twc[:3,3]
pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw));plane,inds=pc.segment_plane(.015,3,1000);print('plane',plane,'inliers',len(inds))
a,b,c,dd=plane;n=np.array([a,b,c]);print('norm',n,'d',dd,'height y at camera axis?')
# fit only inliers stats
q=Pw[inds];print('bounds',q.min(0),q.max(0),'med',np.median(q,0))
# save cloud
pc.paint_uniform_color([.5,.5,.5]);pc.select_by_index(inds).paint_uniform_color([1,0,0]);o3d.io.write_point_cloud('/work/scratch/table_cloud.ply',pc)
PY
python /work/scratch/table_plane.py"
```
<details><summary>output</summary>

```
sel 119677
plane [ 0.02893444  0.99852279  0.04598957 -1.36474226] inliers 110305
norm [0.02893444 0.99852279 0.04598957] d -1.3647422622255074 height y at camera axis?
bounds [-1.15941901  1.00797948  2.45556145] [3.12085137 1.2684321  6.01011764] med [0.67170325 1.18505732 3.53613313]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/table_plane.py';s=open(p).read().replace(\"pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw));plane,inds=pc.segment_plane(.015,3,1000)\",\"Pw=Pw[Pw[:,1]<1.05];\\npc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw));plane,inds=pc.segment_plane(.012,3,1000)\")
open('/work/scratch/table_plane2.py','w').write(s)
PY
python /work/scratch/table_plane2.py"
```
<details><summary>output</summary>

```
sel 119677
plane [ 0.02611623  0.99854307  0.04721943 -1.35954455] inliers 2708
norm [0.02611623 0.99854307 0.04721943] d -1.3595445460425135 height y at camera axis?
bounds [1.87970515 1.00797948 5.06254962] [3.12085137 1.04997313 5.83192822] med [2.58702688 1.03499187 5.51856021]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "for th in 0.9 0.75 0.6; do sed \"s/Pw=Pw\\[Pw\\[:,1\\]<1.05\\]/Pw=Pw[Pw[:,1]<"'$th]/" /work/scratch/table_plane.py > /work/scratch/tp.py; python /work/scratch/tp.py | head -4; done'
```
<details><summary>output</summary>

```
sel 119677
plane [ 0.0290603   0.99852802  0.04579631 -1.36418679] inliers 110153
norm [0.0290603  0.99852802 0.04579631] d -1.3641867887512835 height y at camera axis?
bounds [-1.11225128  1.00797948  2.45556145] [3.12085137 1.2684321  6.00569762] med [0.67668795 1.1851281  3.53501892]
sel 119677
plane [ 0.02880172  0.99852008  0.04613135 -1.36515693] inliers 109718
norm [0.02880172 0.99852008 0.04613135] d -1.3651569257830443 height y at camera axis?
bounds [-1.19936566  1.01221158  2.45556145] [3.03915301 1.2684321  6.03068534] med [0.66911901 1.18535332 3.53114253]
sel 119677
plane [ 0.02906961  0.99852873  0.04577491 -1.36410696] inliers 110143
norm [0.02906961 0.99852873 0.04577491] d -1.3641069558267411 height y at camera axis?
bounds [-1.11225128  1.00797948  2.45556145] [3.1208513 …[+64 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'Pw=' /work/scratch/table_plane.py /work/scratch/tp.py"
```
<details><summary>output</summary>

```
/work/scratch/table_plane.py:10:cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);u=xs[sel];v=ys[sel];z=d[sel];P=np.stack([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z],1);Pw=P@Twc[:3,:3].T+Twc[:3,3]
/work/scratch/tp.py:10:cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);u=xs[sel];v=ys[sel];z=d[sel];P=np.stack([(u-K[0,2])/K[0,0]*z,(v-K[1,2])/K[1,1]*z,z],1);Pw=P@Twc[:3,:3].T+Twc[:3,3]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/table_plane.py').read();s=s.replace(\"Pw=P@Twc[:3,:3].T+Twc[:3,3]\",\"Pw=P@Twc[:3,:3].T+Twc[:3,3];Pw=Pw[(Pw[:,1]<0.9)&(Pw[:,1]>0.0)]\")
open('/work/scratch/tp.py','w').write(s)
PY
python /work/scratch/tp.py"
```
<details><summary>output</summary>

```
sel 119677
plane [ 0.21787547 -0.28708357  0.93279864 -2.19651535] inliers 1193
norm [ 0.21787547 -0.28708357  0.93279864] d -2.1965153522323666 height y at camera axis?
bounds [0.08203871 0.35351553 2.26611842] [0.76384554 0.82884205 2.52172338] med [0.41984944 0.64883324 2.45614385]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/table_plane.py').read();s=s.replace(\"roi=(xs>550)&(xs<1150)&(ys>600)&(ys<950)&~o&~p\",\"roi=((xs-855)/260)**2+((ys-805)/120)**2<1;roi=roi&~o&~p\")
open('/work/scratch/tp_ellipse.py','w').write(s)
PY
python /work/scratch/tp_ellipse.py"
```
<details><summary>output</summary>

```
sel 72327
plane [ 0.0277944   0.9986787   0.04322417 -1.3553051 ] inliers 65007
norm [0.0277944  0.9986787  0.04322417] d -1.3553050962905329 height y at camera axis?
bounds [-0.62982755  1.07224085  2.64176663] [2.00509628 1.23691176 5.80478266] med [0.67681026 1.17799669 3.67198336]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np,json
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r');d=cv2.imdecode(f['aligned/front_stereo_camera_left'][0],cv2.IMREAD_UNCHANGED).astype(float)/1000.;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam'])
for x,y in [(700,700),(800,700),(900,700),(1000,700),(700,800),(800,800),(900,800),(1000,800),(800,900),(900,900),(855,815),(855,750),(650,820),(1050,820)]:
 z=d[y,x];p=np.array([(x-K[0,2])/K[0,0]*z,(y-K[1,1])/K[1,1]*z,z]);w=Twc[:3,:3]@p+Twc[:3,3];print((x,y),'z',round(z,3),'world',np.round(w,3))
PY"
```
<details><summary>output</summary>

```
(700, 700) z 5.756 world [-0.156 -0.581  5.833]
(800, 700) z 2.358 world [ 0.196 -0.313  2.381]
(900, 700) z 2.354 world [ 0.483 -0.315  2.359]
(1000, 700) z 4.819 world [ 1.625 -0.521  4.773]
(700, 800) z 3.888 world [-0.119  0.042  3.915]
(800, 800) z 3.875 world [0.354 0.038 3.872]
(900, 800) z 3.879 world [0.828 0.034 3.845]
(1000, 800) z 3.84 world [1.288 0.029 3.777]
(800, 900) z 2.86 world [0.25  0.344 2.84 ]
(900, 900) z 2.821 world [0.591 0.335 2.779]
(855, 815) z 3.666 world [0.579 0.094 3.644]
(855, 750) z 4.133 world [ 0.657 -0.206  4.127]
(650, 820) z 3.563 world [-0.33   0.117  3.598]
(1050, 820) z 3.596 world [1.422 0.105 3.518]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/inputs/reconstruction.npz');h=r['human_vertices_world'];k=r['human_keypoints_world'];ref=np.load('/work/inputs/frame0_reference_human.npz')['vertices_world']
for name,a in [('human',h[0]),('ref',ref),('kp',k[0])]:print(name,a.min(0),a.max(0),np.median(a,0))
print('floor eq y',1.387290344/.9982268)
PY"
```
<details><summary>output</summary>

```
human [-0.4500471  -0.48185655  3.1102095 ] [0.0534122 1.2092949 3.5723925] [-0.18924832  0.3632819   3.301343  ]
ref [-0.45405295 -0.44103312  3.16835   ] [0.04395172 1.1999127  3.6020515 ] [-0.18486907  0.43791085  3.3334646 ]
kp [-0.4472796 -0.3624939  3.119799 ] [0.04287797 1.1964829  3.5512574 ] [-0.23893324  0.47352564  3.3120015 ]
floor eq y 1.3897546569577175

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np,json
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r');d=cv2.imdecode(f['aligned/front_stereo_camera_left'][0],cv2.IMREAD_UNCHANGED).astype(float)/1000.;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam']);print(K,Twc)
x,y=800,800;z=d[y,x];p=np.array([(x-K[0,2])/K[0,0]*z,(y-K[1,1])/K[1,1]*z,z]);w=Twc[:3,:3]@p+Twc[:3,3];print(z,p,w)
PY"
```
<details><summary>output</summary>

```
[[817.86547852   0.         768.        ]
 [  0.         817.86547852 576.        ]
 [  0.           0.           1.        ]] [[ 0.99790898  0.00388616  0.0645179  -0.04685647]
 [-0.0080006   0.99794105  0.06363684 -0.12334862]
 [-0.06413775 -0.06401995  0.99588543  0.01700147]
 [ 0.          0.          0.          1.        ]]
3.875 [ 0.15161418 -0.08464562  3.875     ] [0.35411858 0.03755978 3.87175233]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/(v-K\\[1,1\\])\\/K\\[1,1\\]/(v-K[1,2])\\/K[1,1]/' /work/scratch/table_plane.py
python /work/scratch/table_plane.py"
```
<details><summary>output</summary>

```
sel 119677
plane [ 0.02902816  0.99852797  0.04581766 -1.36423797] inliers 110250
norm [0.02902816 0.99852797 0.04581766] d -1.3642379690630961 height y at camera axis?
bounds [-1.12128385  1.00797948  2.45556145] [3.12085137 1.2684321  6.00569762] med [0.67382151 1.18508546 3.53577621]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/table_plane.py').read().replace(\"roi=(xs>550)&(xs<1150)&(ys>600)&(ys<950)&~o&~p\",\"roi=((xs-855)/260)**2+((ys-805)/120)**2<1;roi=roi&~o&~p\")
open('/work/scratch/tp_ellipse.py','w').write(s)
PY
python /work/scratch/tp_ellipse.py"
```
<details><summary>output</summary>

```
sel 72327
plane [ 0.02782137  0.9986685   0.043442   -1.35612593] inliers 65102
norm [0.02782137 0.9986685  0.043442  ] d -1.3561259279434121 height y at camera axis?
bounds [-0.62982755  1.07224085  2.65784188] [2.00509628 1.23691176 5.76318131] med [0.67944784 1.17798556 3.67250539]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# execute setup from table_plane until before pc
s=open('/work/scratch/tp_ellipse.py').read().split('pc=o3d')[0];exec(s)
print('y percentiles',np.percentile(Pw[:,1],[0,1,5,10,25,50,75,90,95,99,100]))
for lo,hi in [(0,.4),(.4,.7),(.7,.9),(.9,1.05),(1.05,1.2),(1.2,1.5),(1.5,2)]:
 q=Pw[(Pw[:,1]>=lo)&(Pw[:,1]<hi)];print(lo,hi,len(q),np.round(q.min(0),2) if len(q) else None,np.round(q.max(0),2) if len(q) else None)
PY"
```
<details><summary>output</summary>

```
sel 72327
y percentiles [0.43113314 0.63603538 1.09511176 1.11489108 1.14183711 1.17311111
 1.19788282 1.21492556 1.22438093 1.23196437 1.23691176]
0 0.4 0 None None
0.4 0.7 1013 [0.06 0.43 2.24] [1.18 0.7  3.47]
0.7 0.9 710 [0.08 0.7  2.44] [1.47 0.9  4.5 ]
0.9 1.05 263 [-0.23  0.9   2.61] [1.72 1.05 5.21]
1.05 1.2 53864 [-0.63  1.05  2.47] [2.01 1.2  5.85]
1.2 1.5 16477 [-0.63  1.2   2.64] [1.17 1.24 4.2 ]
1.5 2 0 None None

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/tp_ellipse.py').read().replace(\"pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw));plane,inds=pc.segment_plane(.015,3,1000)\",\"Pw=Pw[Pw[:,1]<1.27];\\npc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw));plane,inds=pc.segment_plane(.008,3,1000)\")
open('/work/scratch/tp_table.py','w').write(s)
PY
python /work/scratch/tp_table.py"
```
<details><summary>output</summary>

```
sel 72327
plane [ 0.02755207  0.99861485  0.04482486 -1.36079363] inliers 62037
norm [0.02755207 0.99861485 0.04482486] d -1.360793629834887 height y at camera axis?
bounds [-0.62767829  1.07224085  2.67264972] [2.00509628 1.23691176 5.51591036] med [0.69722185 1.17934849 3.64410285]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);T=r['T_world_mesh'].astype(float);Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);
for i in [0,190,340,400,545]:
 for name,P in [('raw',Twc@C[i]),('base',T[i])]:
  vw=v@P[:3,:3].T+P[:3,3];print(i,name,'yminmax',np.round([vw[:,1].min(),vw[:,1].max()],3),'t',np.round(P[:3,3],3),'bottom z/x median',np.round(np.median(vw[vw[:,1]>np.percentile(vw[:,1],95)],0),3))
PY"
```
<details><summary>output</summary>

```
0 raw yminmax [-0.195  0.329] t [0.303 0.063 2.613] bottom z/x median [0.336 0.322 2.604]
0 base yminmax [-0.145  0.377] t [0.331 0.11  2.583] bottom z/x median [0.375 0.369 2.585]
190 raw yminmax [-0.198  0.336] t [0.308 0.067 2.562] bottom z/x median [0.329 0.327 2.539]
190 base yminmax [-0.14   0.388] t [0.348 0.121 2.624] bottom z/x median [0.376 0.381 2.605]
340 raw yminmax [-0.215  0.318] t [0.624 0.044 2.569] bottom z/x median [0.592 0.306 2.538]
340 base yminmax [-0.159  0.368] t [0.666 0.096 2.604] bottom z/x median [0.632 0.358 2.578]
400 raw yminmax [-0.199  0.338] t [0.659 0.068 2.466] bottom z/x median [0.608 0.329 2.432]
400 base yminmax [-0.144  0.389] t [0.703 0.12  2.514] bottom z/x median [0.652 0.38  2.487]
545 raw yminmax [-0.191  0.362] t [0.644 0.084 2.498] bottom z/x …[+121 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np,json
f=h5py.File('/pipeline/depth/moge2/aligned.h5','r');d=cv2.imdecode(f['aligned/front_stereo_camera_left'][0],cv2.IMREAD_UNCHANGED).astype(float)/1000.;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);Twc=np.array(cam['T_world_cam'])
for x,y in [(810,675),(810,690),(780,700),(840,700),(800,720),(750,680),(900,680),(850,710),(750,750),(850,750)]:
 z=d[y,x];p=np.array([(x-K[0,2])/K[0,0]*z,(y-K[1,2])/K[1,1]*z,z]);w=Twc[:3,:3]@p+Twc[:3,3];print((x,y),'z',round(z,3),'w',np.round(w,3))
PY"
```
<details><summary>output</summary>

```
(810, 675) z 2.659 w [0.262 0.366 2.636]
(810, 690) z 2.494 w [0.243 0.381 2.47 ]
(780, 700) z 2.356 w [0.141 0.383 2.338]
(840, 700) z 2.364 w [0.315 0.383 2.335]
(800, 720) z 2.188 w [0.181 0.4   2.166]
(750, 680) z 2.626 w [0.066 0.377 2.615]
(900, 680) z 2.618 w [0.545 0.372 2.576]
(850, 710) z 2.251 w [0.325 0.386 2.221]
(750, 750) z 4.623 w [0.154 1.153 4.565]
(850, 750) z 4.601 w [0.714 1.143 4.507]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);C=r['T_cam_mesh_coconet'].astype(float);Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for a,b in [(0,190),(190,400),(400,546)]:
 for name,P in [('base',T),('raw',np.array([Twc@c for c in C]))]:
  mn=[]
  for i in range(a,b,5):mn.append((v@P[i,:3,:3].T+P[i,:3,3])[:,1].min())
  print(name,a,b,'miny med/range',round(np.median(mn),3),round(min(mn),3),round(max(mn),3))
PY"
```
<details><summary>output</summary>

```
base 0 190 miny med/range -0.135 -0.16 -0.127
raw 0 190 miny med/range -0.191 -0.211 -0.177
base 190 400 miny med/range -0.15 -0.308 -0.138
raw 190 400 miny med/range -0.206 -0.362 -0.192
base 400 546 miny med/range -0.135 -0.147 -0.117
raw 400 546 miny med/range -0.193 -0.202 -0.173

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);C=r['T_cam_mesh_coconet'].astype(float);Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for a,b in [(0,190),(190,400),(400,546)]:
 for name,P in [('base',T),('raw',np.array([Twc@c for c in C]))]:
  mn=[]
  for i in range(a,b,5):mn.append((v@P[i,:3,:3].T+P[i,:3,3])[:,1].max())
  print(name,a,b,'maxy med/range',round(np.median(mn),3),round(min(mn),3),round(max(mn),3))
PY"
```
<details><summary>output</summary>

```
base 0 190 maxy med/range 0.388 0.364 0.399
raw 0 190 maxy med/range 0.336 0.319 0.348
base 190 400 maxy med/range 0.38 0.301 0.392
raw 190 400 maxy med/range 0.329 0.248 0.342
base 400 546 maxy med/range 0.41 0.385 0.423
raw 400 546 maxy med/range 0.357 0.335 0.367

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_clean.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);F=len(C)
# robust means
Ri=Rot.from_matrix(C[:191,:3,:3]).mean();ti=np.median(C[:191,:3,3],0)
Rf=Rot.from_matrix(C[400:,:3,:3]).mean();tf=np.median(C[400:,:3,3],0)
# yaw relative camera axes from initial
# fixed world up in camera computed using camera world transform
import json
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);axis=Twc[:3,:3].T@np.array([0.,1.,0.]);axis/=np.linalg.norm(axis)
def yaw(R):return np.degrees((Rot.from_matrix(R@Ri.as_matrix().T)).as_rotvec()@axis)
th=np.array([yaw(C[i,:3,:3]) for i in range(F)])
# stats around glitch
for i in range(330,391):print(i,round(th[i],1),np.round(C[i,:3,3],3))
# smooth t excluding glitch via savgol raw for now
ts=[];Rs=[]
# 191-344 preserve coco rot, blend from init by210
for i in range(F):
 if i<=190: R=Ri.as_matrix();t=ti
 elif i<=344:
  R=C[i,:3,:3];t=C[i,:3,3].copy()
  u=min(1,(i-190)/20);R=(Ri*Rot.from_rotvec(u*(Rot.from_matrix(R)*Ri.inv()).as_rotvec())).as_matrix();t=(1-u)*ti+u*t
 elif i<=379:
  u=(i-344)/36
  # smoothstep
  s=u*u*(3-2*u)
  Rstart=C[344,:3,:3];Rend=C[380,:3,:3]
  # desired yaw interpolate; compute raw yaw and correct
  target=(1-s)*th[344]+s*th[380]
  corr=Rot.from_rotvec(axis*np.radians(target-th[i])).as_matrix();R=corr@C[i, …[+616 chars]
```
<details><summary>output</summary>

```
330 117.9 [0.505 0.024 2.591]
331 119.7 [0.503 0.022 2.584]
332 121.0 [0.505 0.021 2.588]
333 121.6 [0.508 0.018 2.602]
334 121.7 [0.505 0.013 2.605]
335 125.4 [0.509 0.024 2.587]
336 125.9 [0.51  0.026 2.595]
337 128.5 [0.51  0.023 2.595]
338 131.2 [0.513 0.024 2.593]
339 132.8 [0.508 0.024 2.588]
340 133.1 [0.505 0.006 2.596]
341 137.7 [ 0.489 -0.004  2.552]
342 138.0 [ 0.491 -0.009  2.582]
343 139.7 [ 0.485 -0.013  2.573]
344 139.9 [0.493 0.009 2.58 ]
345 120.5 [ 0.434 -0.063  2.627]
346 117.4 [ 0.45  -0.07   2.711]
347 117.4 [ 0.457 -0.09   2.687]
348 89.1 [ 0.506 -0.033  2.685]
349 87.9 [ 0.516 -0.031  2.693]
350 87.8 [ 0.525 -0.035  2.682]
351 90.3 [ 0.466 -0.055  2.668]
352 91.2 [ 0.467 -0.07   2.663]
353 111.8 [ 0.503 -0.045  2.639]
354 106.4 [ 0.51  -0.03   2.677]
355 101.4 [ 0.52 …[+1127 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_clean.py <<'PY'
exec(open('/work/scratch/eval_all_camera.py').read().split(\"r=np.load\")[0])
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam.npz');P=np.zeros_like(C);P[:,:3,:3]=cl['R'];P[:,:3,3]=cl['t'];P[:,3,3]=1
def run(P):
 ious=[]
 for i in range(len(P)):
  T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];ious.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
 return np.array(ious)
a=run(C);b=run(P);print('coco',a.mean(),np.median(a));print('clean',b.mean(),np.median(b));
for lo in range(0,546,20):print(lo,lo+19,round(a[lo:lo+20].mean(),3),round(b[lo:lo+20].mean(),3))
print('worst clean',[(int(i),round(float(b[i]),3)) for i in np.argsort(b)[:30]])
np.save('/work/scratch/clean_iou.npy',b)
PY
python /work/scratch/eval_clean.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.6670605290313023 0.6609410759593921
0 19 0.807 0.839
20 39 0.841 0.839
40 59 0.84 0.837
60 79 0.818 0.83
80 99 0.779 0.811
100 119 0.802 0.808
120 139 0.82 0.818
140 159 0.819 0.818
160 179 0.824 0.814
180 199 0.821 0.807
200 219 0.83 0.842
220 239 0.835 0.627
240 259 0.844 0.46
260 279 0.866 0.447
280 299 0.846 0.427
300 319 0.847 0.534
320 339 0.756 0.543
340 359 0.33 0.351
360 379 0.428 0.487
380 399 0.662 0.546
400 419 0.68 0.718
420 439 0.694 0.706
440 459 0.614 0.637
460 479 0.609 0.616
480 499 0.632 0.615
500 519 0.64 0.616
520 539 0.651 0.628
540 559 0.664 0.633
worst clean [(364, 0.193), (359, 0.209), (344, 0.236), (363, 0.24), (343, 0.256), (342, 0.281), (356, 0.312), (341, 0.328), (355, 0.328), (353, 0.329), (352, 0.344), (349, 0 …[+254 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam.npz');R=cl['R'];t=cl['t']
for i in [200,220,240,260,280,300,320,340]:
 print(i,'t',np.round(t[i],3),'C',np.round(C[i,:3,3],3),'rot',np.degrees(np.linalg.norm((Rot.from_matrix(R[i])*Rot.from_matrix(C[i,:3,:3]).inv()).as_rotvec())))
PY"
```
<details><summary>output</summary>

```
200 t [0.192 0.027 2.6  ] C [0.207 0.029 2.584] rot 3.957142155285904
220 t [0.207 0.023 2.596] C [0.205 0.024 2.592] rot 5.08357070559633
240 t [0.23  0.02  2.574] C [0.228 0.02  2.576] rot 46.70481540471672
260 t [0.255 0.02  2.556] C [0.251 0.022 2.546] rot 61.87100894857425
280 t [0.306 0.015 2.638] C [0.311 0.013 2.656] rot 90.40434255290381
300 t [0.368 0.008 2.633] C [0.367 0.009 2.626] rot 108.23858873845903
320 t [0.462 0.011 2.616] C [0.465 0.011 2.643] rot 134.53335343416126
340 t [0.5   0.01  2.585] C [0.505 0.006 2.596] rot 161.111822049992

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float);Ri=Rot.from_matrix(C[:191,:3,:3]).mean()
for i in [220,240,260,300,340]:
 R=C[i,:3,:3];Q=Rot.from_matrix(R)*Ri.inv();D=Rot.from_rotvec(Q.as_rotvec());Rrec=(Ri*D).as_matrix();print(i,np.degrees(Q.magnitude()),np.degrees(np.linalg.norm((Rot.from_matrix(Rrec)*Rot.from_matrix(R).inv()).as_rotvec())))
PY"
```
<details><summary>output</summary>

```
220 6.447531081265678 5.08357070559633
240 33.17197252932722 46.70481540471672
260 46.00878096870813 61.87100894857425
300 81.69694415167257 108.23858873845903
340 133.17192093984045 161.111822049992

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_clean.py';s=open(p).read();s=s.replace(\"R=(Ri*Rot.from_rotvec(u*(Rot.from_matrix(R)*Ri.inv()).as_rotvec())).as_matrix()\",\"R=(Rot.from_rotvec(u*(Rot.from_matrix(R)*Ri.inv()).as_rotvec())*Ri).as_matrix()\")
s=s.replace(\"R=(Rot.from_matrix(C[380,:3,:3])*Rot.from_rotvec(s*(Rf*Rot.from_matrix(C[380,:3,:3]).inv()).as_rotvec())).as_matrix()\",\"R=(Rot.from_rotvec(s*(Rf*Rot.from_matrix(C[380,:3,:3]).inv()).as_rotvec())*Rot.from_matrix(C[380,:3,:3])).as_matrix()\")
open(p,'w').write(s)
PY
python /work/scratch/build_clean.py >/work/scratch/build_clean.log
python /work/scratch/eval_clean.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7434170095742434 0.8123517419751152
0 19 0.807 0.839
20 39 0.841 0.839
40 59 0.84 0.837
60 79 0.818 0.83
80 99 0.779 0.811
100 119 0.802 0.808
120 139 0.82 0.818
140 159 0.819 0.818
160 179 0.824 0.814
180 199 0.821 0.805
200 219 0.83 0.827
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.763
340 359 0.33 0.392
360 379 0.428 0.487
380 399 0.662 0.645
400 419 0.68 0.718
420 439 0.694 0.706
440 459 0.614 0.637
460 479 0.609 0.616
480 499 0.632 0.615
500 519 0.64 0.616
520 539 0.651 0.628
540 559 0.664 0.633
worst clean [(364, 0.193), (359, 0.209), (363, 0.24), (356, 0.312), (355, 0.328), (353, 0.329), (352, 0.344), (349, 0.355), (351, 0.363), (346, 0.365), (345, 0.375), (375,  …[+254 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:];p=h['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [340,345,350,355,360,365,370,375,380]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();mk=np.zeros_like(im);mk[o[i]]=[255,255,255];mk[p[i]]=[0,0,255];cv2.putText(mk,str(i),(5,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2);cv2.imwrite(f'/work/scratch/m_{i}.jpg',np.concatenate([im[450:710,700:1050],mk[450:710,700:1050]],1))
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
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:];p=h['person'][:]
for i in range(330,381):
 m=o[i]&~p[i];ys,xs=np.where(m)
 print(i,'area',m.sum(),'bbox',xs.min(),xs.max(),ys.min(),ys.max(),'w',xs.max()-xs.min()+1,'h',ys.max()-ys.min()+1)
PY"
```
<details><summary>output</summary>

```
330 area 12633 bbox 858 973 495 653 w 116 h 159
331 area 12316 bbox 858 971 495 654 w 114 h 160
332 area 11960 bbox 858 969 495 654 w 112 h 160
333 area 11505 bbox 858 966 495 654 w 109 h 160
334 area 11880 bbox 854 964 496 654 w 111 h 159
335 area 11171 bbox 857 962 496 654 w 106 h 159
336 area 10439 bbox 856 958 497 655 w 103 h 159
337 area 9145 bbox 860 957 497 655 w 98 h 159
338 area 8799 bbox 859 954 496 655 w 96 h 160
339 area 8270 bbox 860 951 496 655 w 92 h 160
340 area 7531 bbox 858 947 496 655 w 90 h 160
341 area 6633 bbox 858 943 496 656 w 86 h 161
342 area 5802 bbox 859 940 496 656 w 82 h 161
343 area 5185 bbox 864 937 496 656 w 74 h 161
344 area 4750 bbox 864 935 497 656 w 72 h 160
345 area 4482 bbox 859 935 498 657 w 77 h 160
346 area 4563 bbox 858 935 495 657 w 78 h 163
347  …[+1568 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/scan_clean.py <<'PY'
exec(open('/work/scratch/eval_all_camera.py').read().split(\"r=np.load\")[0])
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam.npz');R0=cl['R'];T0=cl['t'];basey=cl['yaw']
def render_score(R,t,i):
 v=V@torch.tensor(R,dtype=torch.float32,device='cuda').T+torch.tensor(t,dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];return (vis&om[i]).sum()/max(1,(vis|om[i]).sum())
for i in range(335,391):
 vals=[]
 rawyaw=cl['yaw'][i]
 for d in np.arange(-35,35.1,5):
  yaw=rawyaw+d;corr=Rot.from_rotvec(axis*np.radians(yaw-cl['yaw'][i])).as_matrix();R=corr@R0[i];vals.append((yaw,render_score(R,T0[i],i)))
 vals.sort(key=lambda x:-x[1]);print('FRAME',i,'base',round(basey[i],1),'best',[(round(a,1),round(float(b),3)) for a,b in vals[:8]],flush=True)
PY
# Need axis defined in exec? eval file defines no axis. add
sed -i \"/from scipy.spatial.transform import Rotation as Rot/a import json\\naxis=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])[:3,:3].T@np.array([0.,1.,0.]);axis/=np.linalg.norm(axis)\" /work/scratch/scan_clean.py
python /work/scratch/scan_clean.py"
```
<details><summary>output</summary>

```
FRAME 335 base 125.4 best [(115.4, 0.737), (110.4, 0.72), (120.4, 0.709), (105.4, 0.695), (100.4, 0.672), (125.4, 0.667), (95.4, 0.652), (90.4, 0.635)]
FRAME 336 base 125.9 best [(120.9, 0.694), (115.9, 0.685), (110.9, 0.662), (125.9, 0.661), (105.9, 0.641), (100.9, 0.621), (130.9, 0.606), (95.9, 0.604)]
FRAME 337 base 128.5 best [(123.5, 0.706), (128.5, 0.683), (118.5, 0.667), (163.5, 0.644), (113.5, 0.642), (108.5, 0.618), (133.5, 0.611), (103.5, 0.597)]
FRAME 338 base 131.2 best [(126.2, 0.678), (131.2, 0.637), (121.2, 0.62), (166.2, 0.598), (116.2, 0.594), (161.2, 0.593), (111.2, 0.575), (106.2, 0.558)]
FRAME 339 base 132.8 best [(127.8, 0.648), (132.8, 0.605), (122.8, 0.591), (162.8, 0.575), (117.8, 0.552), (167.8, 0.538), (112.8, 0.535), (107.8, 0.519)]
FRAME 340 base 133.1 best [(13 …[+7672 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/scan_clean.py').read();s=s.replace(\"rawyaw=cl['yaw'][i]\",\"rawyaw=np.degrees((Rot.from_matrix(R0[i]@cl['R'][0].T)).as_rotvec()@axis)\") # wrong ref use Ri? R0[0] okay
s=s.replace(\"corr=Rot.from_rotvec(axis*np.radians(yaw-cl['yaw'][i])).as_matrix();R=corr@R0[i]\",\"corr=Rot.from_rotvec(axis*np.radians(yaw-rawyaw)).as_matrix();R=corr@R0[i]\")
open('/work/scratch/scan_clean2.py','w').write(s)
PY
python /work/scratch/scan_clean2.py | sed -n '1,220p'"
```
<details><summary>output</summary>

```
FRAME 335 base 125.4 best [(115.4, 0.737), (110.4, 0.72), (120.4, 0.709), (105.4, 0.695), (100.4, 0.672), (125.4, 0.667), (95.4, 0.652), (90.4, 0.635)]
FRAME 336 base 125.9 best [(120.9, 0.694), (115.9, 0.685), (110.9, 0.662), (125.9, 0.661), (105.9, 0.641), (100.9, 0.621), (130.9, 0.606), (95.9, 0.604)]
FRAME 337 base 128.5 best [(123.5, 0.706), (128.5, 0.683), (118.5, 0.667), (163.5, 0.644), (113.5, 0.642), (108.5, 0.618), (133.5, 0.611), (103.5, 0.597)]
FRAME 338 base 131.2 best [(126.2, 0.678), (131.2, 0.637), (121.2, 0.62), (166.2, 0.598), (116.2, 0.594), (161.2, 0.593), (111.2, 0.575), (106.2, 0.558)]
FRAME 339 base 132.8 best [(127.8, 0.648), (132.8, 0.605), (122.8, 0.591), (162.8, 0.575), (117.8, 0.552), (167.8, 0.538), (112.8, 0.535), (107.8, 0.519)]
FRAME 340 base 133.1 best [(13 …[+7822 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/final_yaw_scan.py <<'PY'
exec(open('/work/scratch/eval_static_poses.py').read().split(\"for name,R,t\")[0])
# R0 initial and base final rotation/translation from clean
cl=np.load('/work/scratch/clean_cam.npz');Rbase=cl['R'][400];tbase=cl['t'][400];idx=list(range(400,546,3))
axis=np.array(cam['T_world_cam'])[:3,:3].T@np.array([0.,1.,0.]);axis/=np.linalg.norm(axis)
def eval_pose(R,t):
 vals=[]
 for i in idx:
  mk=rend(R,t);vis=mk&~pm[i];vals.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
 return np.mean(vals),vals
def align(R,t):
 best,_=eval_pose(R,t);t=t.copy()
 for step in [.04,.02,.01,.005]:
  moved=True
  while moved:
   moved=False
   for j in range(3):
    for sg in [1,-1]:
     tt=t.copy();tt[j]+=sg*step;s,_=eval_pose(R,tt)
     if s>best+1e-4:best=s;t=tt;moved=True;break
 return best,t
for deg in range(155,211,5):
 R=Rot.from_rotvec(axis*np.radians(deg)).as_matrix()@Rbase;s,t=align(R,tbase);print(deg,round(s,4),np.round(t,3),flush=True)
PY
python /work/scratch/final_yaw_scan.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/final_yaw_scan.py", line 22, in <module>
    R=Rot.from_rotvec(axis*np.radians(deg)).as_matrix()@Rbase;s,t=align(R,tbase);print(deg,round(s,4),np.round(t,3),flush=True)
      ^^^
NameError: name 'Rot' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '1i from scipy.spatial.transform import Rotation as Rot' /work/scratch/final_yaw_scan.py
python /work/scratch/final_yaw_scan.py"
```
<details><summary>output</summary>

```
155 0.4598 [0.381 0.03  2.179]
160 0.5351 [0.406 0.03  2.294]
165 0.5696 [0.411 0.035 2.344]
170 0.5827 [0.431 0.035 2.484]
175 0.6135 [0.436 0.035 2.459]
180 0.636 [0.466 0.025 2.469]
185 0.6519 [0.486 0.02  2.554]
190 0.655 [0.496 0.015 2.579]
195 0.6523 [0.526 0.    2.759]
200 0.649 [ 0.556 -0.01   2.919]
205 0.6389 [ 0.556 -0.01   2.954]
210 0.57 [0.531 0.04  2.404]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/diag_clean.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
from PIL import Image,ImageDraw
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536;cl=np.load('/work/scratch/clean_cam.npz');P=np.zeros((546,4,4));P[:,:3,:3]=cl['R'];P[:,:3,3]=cl['t'];P[:,3,3]=1
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
cols=torch.zeros((len(V),3),device='cuda');cols[:,0]=(V[:,0]-V[:,0].min())/(V[:,0].max()-V[:,0].min());cols[:,1]=(V[:,1]-V[:,1].min())/(V[:,1].max()-V[:,1].min());cols[:,2]=(V[:,2]-V[:,2].min())/(V[:,2].max()-V[:,2].min())
def rend(i):
 T=P[i];v=V@torch.tensor(T[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(T[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();rgb=dr.interpolate(cols[None],rast,Fa)[0][0].cpu().numpy();return mk,rgb
cap=cv2.VideoCapture('/work/inputs/video.mp4');ims=[]
for i in [190,220,260,300,330,335,340,345,350,355,360,365,370,375,380,385 …[+603 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_clean_t.py <<'PY'
exec(open('/work/scratch/eval_static_poses.py').read().split(\"for name,R,t\")[0])
cl=np.load('/work/scratch/clean_cam.npz');Rall=cl['R'];tall=cl['t']
def score(R,t,i):
 mk=rend(R,t);vis=mk&~pm[i];return (vis&om[i]).sum()/max(1,(vis|om[i]).sum())
def cd(R,t,i):
 best=score(R,t,i)
 for step in [.12,.06,.03,.015,.0075]:
  for j in range(3):
   for it in range(4):
    moved=False
    for sg in [1,-1]:
     tt=t.copy();tt[j]+=sg*step;s=score(R,tt,i)
     if s>best+1e-5:best=s;t=tt;moved=True;break
    if not moved:break
 return best,t
out={}
for i in range(335,391):
 s,t=cd(Rall[i],tall[i].copy(),i);out[i]=(s,t);print(i,'base',round(score(Rall[i],tall[i],i),3),'best',round(s,3),'dt',np.round(t-tall[i],3),'t',np.round(t,3),flush=True)
np.save('/work/scratch/opt_clean_t.npy',np.array([[i,*out[i][0:1],*out[i][1]] for i in out]))
PY
python /work/scratch/opt_clean_t.py"
```
<details><summary>output</summary>

```
335 base 0.667 best 0.691 dt [-0.007 -0.015  0.03 ] t [0.501 0.005 2.62 ]
336 base 0.661 best 0.722 dt [-0.008 -0.015  0.06 ] t [0.5   0.003 2.649]
337 base 0.683 best 0.767 dt [ 0.    -0.015  0.075] t [0.506 0.    2.665]
338 base 0.637 best 0.749 dt [ 0.    -0.015  0.105] t [ 5.040e-01 -2.000e-03  2.695e+00]
339 base 0.605 best 0.753 dt [ 0.    -0.015  0.135] t [ 0.502 -0.004  2.723]
340 base 0.562 best 0.68 dt [ 0.    -0.022  0.158] t [ 0.5   -0.013  2.742]
341 base 0.44 best 0.577 dt [ 0.015 -0.015  0.285] t [ 0.514 -0.007  2.868]
342 base 0.475 best 0.5 dt [ 0.    -0.008  0.06 ] t [ 0.497 -0.     2.641]
343 base 0.424 best 0.435 dt [0.   0.   0.06] t [0.495 0.006 2.639]
344 base 0.404 best 0.406 dt [ 0.     0.    -0.008] t [0.494 0.005 2.57 ]
345 base 0.375 best 0.4 dt [0.    0.112 0.0 …[+3409 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/opt_clean_t.py').read();s=s.replace(\"for step in [.12,.06,.03,.015,.0075]:\\n  for j in range(3):\",\"for step in [.12,.06,.03,.015,.0075]:\\n  for j in range(3):\")
# limit z delta around base
s=s.replace(\"tt=t.copy();tt[j]+=sg*step;s=score(R,tt,i)\",\"tt=t.copy();tt[j]+=sg*step\\n     if abs(tt[2]-tall[i,2])>.10: continue\\n     s=score(R,tt,i)\")
open('/work/scratch/opt_clean_t_lim.py','w').write(s)
PY
python /work/scratch/opt_clean_t_lim.py"
```
<details><summary>output</summary>

```
335 base 0.667 best 0.691 dt [-0.007 -0.015  0.03 ] t [0.501 0.005 2.62 ]
336 base 0.661 best 0.722 dt [-0.008 -0.015  0.06 ] t [0.5   0.003 2.649]
337 base 0.683 best 0.767 dt [ 0.    -0.015  0.075] t [0.506 0.    2.665]
338 base 0.637 best 0.755 dt [-0.008 -0.015  0.068] t [ 4.970e-01 -2.000e-03  2.658e+00]
339 base 0.605 best 0.754 dt [-0.008 -0.015  0.09 ] t [ 0.495 -0.004  2.678]
340 base 0.562 best 0.667 dt [-0.015 -0.015  0.09 ] t [ 0.485 -0.005  2.675]
341 base 0.44 best 0.549 dt [-0.03  -0.03   0.067] t [ 0.469 -0.022  2.651]
342 base 0.475 best 0.5 dt [ 0.    -0.008  0.06 ] t [ 0.497 -0.     2.641]
343 base 0.424 best 0.435 dt [0.   0.   0.06] t [0.495 0.006 2.639]
344 base 0.404 best 0.406 dt [ 0.     0.    -0.008] t [0.494 0.005 2.57 ]
345 base 0.375 best 0.4 dt [0.    0.112 0. …[+3383 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/build_clean2.py <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);F=len(C)
si=np.load('/work/scratch/static_init.npz');sf=np.load('/work/scratch/static_final.npz');Ri=si['R'];ti=si['t'];Rfinal=sf['R'];tfinal=sf['t']
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);axis=Twc[:3,:3].T@np.array([0.,1.,0.]);axis/=np.linalg.norm(axis)
Rref=Ri
def yaw(R):return np.degrees((Rot.from_matrix(R@Rref.T)).as_rotvec()@axis)
th=np.array([yaw(C[i,:3,:3]) for i in range(F)])
# Smooth baseline translations before corruption, and create a smooth transition using anchors.
# Savgol on pre-glitch baseline up to 344
ts=C[:,:3,3].copy()
for j in range(3):ts[190:345,j]=savgol_filter(C[190:345,:3,3][:,j],21,2)
# anchors at last reliable and final settled pose; interpolate smoothly through occluded rotation
for i in range(345,380):
 u=(i-344)/36;s=u*u*(3-2*u)
 # translation follows smooth path, avoid tracker's long detour
 for j in range(3):ts[i,j]=(1-s)*ts[344,j]+s*ts[380,j]
# blend to optimized camera frame static final by 400
for i in range(380,400):
 u=(i-380)/20;s=u*u*(3-2*u);ts[i]=(1-s)*ts[380]+s*tfinal
ts[400:]=tfinal
for i in range(0,191):ts[i]=ti
Rs=[]
for i in range(F):
 if i<=190:R=Ri
 elif i<=344:
  R=C[i,:3,:3];u=min(1,(i-190)/ …[+1124 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/build_clean2.py", line 29, in <module>
    R=C[i,:3,:3];u=min(1,(i-190)/20);R=(Rot.from_rotvec(u*(Rot.from_matrix(R)*Ri.inv()).as_rotvec())*Ri).as_matrix()
                                                                              ^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'inv'
Traceback (most recent call last):
  File "/work/scratch/eval_clean2.py", line 2, in <module>
    r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam2.npz');P=np.zeros_like(C);P[:,:3,:3]=cl['R'];P[:,:3,3]=cl['t'];P[:,3,3]=1
                                                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/co …[+276 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_clean2.py';s=open(p).read();s=s.replace(\"Rref=Ri\",\"RiR=Rot.from_matrix(Ri);Rref=Ri\");s=s.replace(\"Ri.inv()\",\"RiR.inv()\").replace(\"*Ri).as_matrix()\",\"*RiR).as_matrix()\")
# static final blend line uses Rfinal matrix okay
open(p,'w').write(s)
PY
python /work/scratch/build_clean2.py
python /work/scratch/eval_clean2.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/build_clean2.py", line 36, in <module>
    R=(Rot.from_rotvec(s*(Rfinal@C[380,:3,:3].T).as_rotvec())@C[380,:3,:3]).as_matrix()
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'numpy.ndarray' object has no attribute 'as_rotvec'
Traceback (most recent call last):
  File "/work/scratch/eval_clean2.py", line 2, in <module>
    r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam2.npz');P=np.zeros_like(C);P[:,:3,:3]=cl['R'];P[:,:3,3]=cl['t'];P[:,3,3]=1
                                                                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio. …[+227 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_clean2.py';s=open(p).read();s=s.replace(\" elif i<=399:\\n  u=(i-380)/20;s=u*u*(3-2*u)\\n  R=(Rot.from_rotvec(s*(Rfinal@C[380,:3,:3].T).as_rotvec())@C[380,:3,:3]).as_matrix()\\n  # note Rfinal and C are different coordinate frames; this blend is only coarse, use q slerp via camera correction of final\\n\", \" elif i<=399:\\n  R=C[i,:3,:3]\\n\")
open(p,'w').write(s)
PY
python /work/scratch/build_clean2.py
python /work/scratch/eval_clean2.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7816018233205195 0.8360306188842301
0 19 0.807 0.923
20 39 0.841 0.922
40 59 0.84 0.916
60 79 0.818 0.876
80 99 0.779 0.841
100 119 0.802 0.85
120 139 0.82 0.861
140 159 0.819 0.868
160 179 0.824 0.859
180 199 0.821 0.825
200 219 0.83 0.829
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.395
360 379 0.428 0.49
380 399 0.662 0.683
400 419 0.68 0.737
420 439 0.694 0.765
440 459 0.614 0.702
460 479 0.609 0.683
480 499 0.632 0.69
500 519 0.64 0.699
520 539 0.651 0.713
540 559 0.664 0.717
worst clean [(364, 0.212), (359, 0.235), (363, 0.252), (352, 0.333), (356, 0.343), (349, 0.346), (346, 0.349), (345, 0.351), (351, 0.352), (355, 0.354), (353, 0.367), (344, 0. …[+255 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_world.py <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);T=r['T_world_mesh'].astype(float);cl=np.load('/work/scratch/clean_cam2.npz');Rc=cl['R'];F=len(T)
# relative rotation from camera to world is essentially constant
X=[T[i,:3,:3]@C[i,:3,:3].T for i in range(F)];Roff=Rot.from_matrix(X).mean().as_matrix();print('off angle spread',np.degrees(np.max(np.linalg.norm((Rot.from_matrix(X)*Rot.from_matrix(Roff).inv()).as_rotvec(),axis=1))))
Rw=np.array([Roff@Rc[i] for i in range(F)])
# world translations: preserve baseline depth calibration, smooth jitter, repair tracker detour
tw=T[:,:3,3].copy()
for j in range(3):tw[190:345,j]=savgol_filter(T[190:345,:3,3][:,j],21,2)
for i in range(345,380):
 u=(i-344)/36;s=u*u*(3-2*u)
 tw[i]=(1-s)*tw[344]+s*tw[380]
for i in range(380,400):
 u=(i-380)/20;s=u*u*(3-2*u);tw[i]=(1-s)*tw[380]+s*np.median(T[400:,:3,3],0)
tw[:191]=np.median(T[:191,:3,3],0);tw[400:]=np.median(T[400:,:3,3],0)
# enforce constant rotation static
Rwi=Rot.from_matrix(Rw[:191]).mean().as_matrix();Rwf=Rot.from_matrix(Rw[400:]).mean().as_matrix();Rw[:191]=Rwi;Rw[400:]=Rwf
# blend rotation into static at boundaries
for i in range(191,211):
 u=(i-190)/20;s=u*u*(3-2*u);Rw[i]=(Rot.from_rotvec(s*(Rot.from_matrix(Rw[i])*Rot.from_matrix(Rwi).inv()).as_rotvec())*Rot.from_matrix(Rwi)).as_m …[+474 chars]
```
<details><summary>output</summary>

```
off angle spread 0.14768159936673148
static t std [0. 0. 0.] [0. 0. 0.]
rel final angle 176.71484195422636

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_world_candidate.py <<'PY'
exec(open('/work/scratch/eval_all_camera.py').read().split(\"r=np.load\")[0])
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);P=np.load('/work/scratch/candidate_world.npz')['T_world_mesh'];Tcw=np.linalg.inv(np.array(cam['T_world_cam']));H,W=1152,1536
def run(P,K):
 fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ious=[]
 for i in range(len(P)):
  Tt=Tcw@P[i];v=V@torch.tensor(Tt[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(Tt[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];ious.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
 return np.array(ious)
for kk in ['K','K_pipeline']:
 a=run(T,np.array(cam[kk]));b=run(P,np.array(cam[kk]));print(kk,'base',a.mean(),np.median(a),'cand',b.mean(),np.median(b));
 for lo in range(0,546,20):print(lo,round(a[lo:lo+20].mean(),3),round(b[lo:lo+20].mean(),3))
PY
python /work/scratch/eval_world_candidate.py"
```
<details><summary>output</summary>

```
K base 0.4534824146079515 0.48931314373021384 cand 0.45489189607306246 0.4868634153490793
0 0.56 0.5
20 0.514 0.5
40 0.508 0.504
60 0.498 0.497
80 0.457 0.471
100 0.488 0.488
120 0.492 0.491
140 0.502 0.496
160 0.49 0.487
180 0.485 0.478
200 0.493 0.492
220 0.51 0.51
240 0.524 0.524
260 0.525 0.525
280 0.517 0.518
300 0.504 0.504
320 0.494 0.497
340 0.375 0.353
360 0.342 0.346
380 0.386 0.378
400 0.394 0.423
420 0.407 0.422
440 0.358 0.398
460 0.353 0.39
480 0.354 0.379
500 0.368 0.37
520 0.371 0.37
540 0.367 0.365
K_pipeline base 0.5266313302489372 0.5683275775764658 cand 0.5322739369956834 0.5688533521577155
0 0.632 0.612
20 0.632 0.612
40 0.617 0.61
60 0.576 0.582
80 0.568 0.568
100 0.584 0.569
120 0.575 0.567
140 0.569 0.571
160 0.564 0.586
180 0.567 0.588
200 0.57 0.574
220 0.598 0.59 …[+255 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/contact_eval.py <<'PY'
import numpy as np,trimesh,json
from scipy.spatial import cKDTree
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);C=r['T_cam_mesh_coconet'].astype(float);K=r['human_keypoints_world'];Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);cand=np.load('/work/scratch/candidate_world.npz')['T_world_mesh'];raw=np.array([Twc@c for c in C])
for name,P in [('baseline',T),('raw',raw),('cand',cand)]:
 ds=[]
 for i in range(546):
  vw=v@P[i,:3,:3].T+P[i,:3,3];tree=cKDTree(vw);h=np.concatenate([K[i,21:42],K[i,42:63]]);d=tree.query(h)[0];ds.append(d.min())
 ds=np.array(ds);print(name,'contact frames 100-450 med',np.median(ds[100:450]),'p90',np.percentile(ds[100:450],90),'<5cm',np.mean(ds[100:450]<.05))
 for a,b in [(0,100),(100,200),(200,300),(300,400),(400,450),(450,546)]:print(a,b,round(np.median(ds[a:b]),3),round(np.mean(ds[a:b]<.05),2))
PY
python /work/scratch/contact_eval.py"
```
<details><summary>output</summary>

```
baseline contact frames 100-450 med 0.0019205813019067567 p90 0.03469479469256977 <5cm 0.9371428571428572
0 100 0.484 0.0
100 200 0.002 1.0
200 300 0.002 1.0
300 400 0.003 0.78
400 450 0.002 1.0
450 546 0.502 0.01
raw contact frames 100-450 med 0.005969232819810405 p90 0.048416161467384984 <5cm 0.9028571428571428
0 100 0.477 0.02
100 200 0.003 1.0
200 300 0.009 1.0
300 400 0.031 0.67
400 450 0.004 0.98
450 546 0.568 0.0
cand contact frames 100-450 med 0.0019041352502781749 p90 0.00509965573094553 <5cm 0.9714285714285714
0 100 0.467 0.02
100 200 0.002 1.0
200 300 0.002 1.0
300 400 0.002 0.9
400 450 0.002 1.0
450 546 0.497 0.02

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,cv2,h5py
from scipy.spatial.transform import Rotation as Rot
cl=np.load('/work/scratch/clean_cam2.npz');C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float);r=np.load('/work/inputs/reconstruction.npz');
# brightness
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:];p=h['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4');bright=np.zeros(546)
for i in range(330,391):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);sel=o[i]&~p[i];bright[i]=im[sel].mean()
for i in range(335,391):
 R=cl['R'][i];front=R@np.array([0,1.,0]);back=-front;t=cl['t'][i];# view from object to cam direction approx -t normalized
 view=-t/np.linalg.norm(t);print(i,'frontdot',round(float(front@view),2),'backdot',round(float(back@view),2),'bright',round(bright[i],1))
PY"
```
<details><summary>output</summary>

```
335 frontdot -0.38 backdot 0.38 bright 123.2
336 frontdot -0.38 backdot 0.38 bright 122.6
337 frontdot -0.34 backdot 0.34 bright 127.6
338 frontdot -0.29 backdot 0.29 bright 123.2
339 frontdot -0.26 backdot 0.26 bright 119.1
340 frontdot -0.26 backdot 0.26 bright 114.0
341 frontdot -0.18 backdot 0.18 bright 106.8
342 frontdot -0.17 backdot 0.17 bright 95.7
343 frontdot -0.14 backdot 0.14 bright 81.2
344 frontdot -0.14 backdot 0.14 bright 65.2
345 frontdot -0.14 backdot 0.14 bright 54.3
346 frontdot -0.14 backdot 0.14 bright 52.2
347 frontdot -0.13 backdot 0.13 bright 52.4
348 frontdot 0.17 backdot -0.17 bright 52.4
349 frontdot 0.26 backdot -0.26 bright 52.3
350 frontdot 0.22 backdot -0.22 bright 52.7
351 frontdot 0.29 backdot -0.29 bright 52.7
352 frontdot 0.34 backdot -0.34 bright 52.4
3 …[+1663 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
from scipy.spatial.transform import Rotation as Rot
import numpy as np,json
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam2.npz');th=cl['yaw'];axis=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])[:3,:3].T@np.array([0.,1.,0.]);axis/=np.linalg.norm(axis);R344=C[344,:3,:3];t=C[344,:3,3]
for i in range(340,381):
 u=np.clip((i-344)/36,0,1);s=u*u*(3-2*u);target=(1-s)*th[344]+s*th[380]
 R=Rot.from_rotvec(axis*np.radians(target-th[344])).as_matrix()@R344;front=R@np.array([0,1.,0]);view=-t/np.linalg.norm(t);print(i,round(target,1),round(float(front@view),2))
PY"
```
<details><summary>output</summary>

```
340 140.7 -0.14
341 140.7 -0.14
342 140.7 -0.14
343 140.7 -0.14
344 140.7 -0.14
345 140.8 -0.14
346 141.0 -0.14
347 141.4 -0.13
348 141.8 -0.12
349 142.4 -0.11
350 143.1 -0.1
351 143.8 -0.09
352 144.7 -0.07
353 145.6 -0.06
354 146.7 -0.04
355 147.8 -0.02
356 148.9 -0.0
357 150.1 0.02
358 151.3 0.04
359 152.6 0.06
360 153.9 0.08
361 155.2 0.11
362 156.5 0.13
363 157.8 0.15
364 159.1 0.17
365 160.4 0.2
366 161.6 0.22
367 162.9 0.24
368 164.0 0.26
369 165.2 0.28
370 166.3 0.3
371 167.3 0.31
372 168.2 0.33
373 169.1 0.34
374 169.9 0.36
375 170.6 0.37
376 171.1 0.38
377 171.6 0.38
378 171.9 0.39
379 172.1 0.39
380 172.2 0.39

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float)
for i in [0,100,200,300,330,340,380,400,545]:
 R=C[i,:3,:3];f=R@np.array([0,1.,0]);t=C[i,:3,3];view=-t/np.linalg.norm(t);print(i,'frontdot',round(float(f@view),2))
PY"
```
<details><summary>output</summary>

```
0 frontdot -0.62
100 frontdot -0.62
200 frontdot -0.61
300 frontdot -0.89
330 frontdot -0.5
340 frontdot -0.26
380 frontdot 0.35
400 frontdot 0.51
545 frontdot 0.55

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
sf=np.load('/work/scratch/static_final.npz');si=np.load('/work/scratch/static_init.npz');
for name,R,t in [('init',si['R'],si['t']),('final',sf['R'],sf['t'])]:
 f=R@np.array([0,1.,0]);v=-t/np.linalg.norm(t);print(name,'+ydot',f@v,'-ydot',-f@v)
PY"
```
<details><summary>output</summary>

```
init +ydot -0.606629198865831 -ydot 0.606629198865831
final +ydot 0.5492387108764916 -ydot -0.5492387108764916

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/build_clean2.py').read()
old=\"\"\" elif i<=379:
  u=(i-344)/36;s=u*u*(3-2*u)
  target=(1-s)*th[344]+s*th[380]
  corr=Rot.from_rotvec(axis*np.radians(target-th[i])).as_matrix();R=corr@C[i,:3,:3]
\"\"\"
new=\"\"\" elif i<=379:
  u=(i-344)/36;s=u*u*(3-2*u)
  target=(1-s)*th[344]+s*th[380]
  # propagate the last reliable orientation rigidly about the table-normal axis
  R=(Rot.from_rotvec(axis*np.radians(target-th[344])).as_matrix()@C[344,:3,:3])
\"\"\"
s=s.replace(old,new)
# final blend should be from frame380 clean base rather than raw C380
s=s.replace(\"Rs[i]=(Rot.from_rotvec(s*(Rot.from_matrix(Rfinal)*Rot.from_matrix(C[380,:3,:3]).inv()).as_rotvec())*Rot.from_matrix(C[380,:3,:3])).as_matrix()\",\"Rs[i]=(Rot.from_rotvec(s*(Rot.from_matrix(Rfinal)*Rot.from_matrix(Rs[380]).inv()).as_rotvec())*Rot.from_matrix(Rs[380])).as_matrix()\")
s=s.replace(\"clean_cam2.npz\",\"clean_cam3.npz\")
open('/work/scratch/build_clean3.py','w').write(s)
PY
python /work/scratch/build_clean3.py
sed 's/clean_cam2.npz/clean_cam3.npz/' /work/scratch/eval_clean2.py > /work/scratch/eval_clean3.py
python /work/scratch/eval_clean3.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7853332051075033 0.8360306188842301
0 19 0.807 0.923
20 39 0.841 0.922
40 59 0.84 0.916
60 79 0.818 0.876
80 99 0.779 0.841
100 119 0.802 0.85
120 139 0.82 0.861
140 159 0.819 0.868
160 179 0.824 0.859
180 199 0.821 0.825
200 219 0.83 0.829
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.37
360 379 0.428 0.617
380 399 0.662 0.683
400 419 0.68 0.737
420 439 0.694 0.765
440 459 0.614 0.702
460 479 0.609 0.683
480 499 0.632 0.69
500 519 0.64 0.699
520 539 0.651 0.713
540 559 0.664 0.717
worst clean [(354, 0.258), (355, 0.263), (356, 0.283), (353, 0.29), (357, 0.308), (352, 0.328), (358, 0.339), (351, 0.358), (345, 0.358), (346, 0.37), (359, 0.377), (344, 0.37 …[+254 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/clean_cam.npz/clean_cam3.npz/' /work/scratch/diag_clean.py > /work/scratch/diag_clean3.py
python /work/scratch/diag_clean3.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# use functions from diag_clean3 perhaps load huge then compute bbox
s=open('/work/scratch/diag_clean3.py').read().split('cap=cv2.VideoCapture')[0];exec(s)
for i in range(335,381,2):
 mk,rgb=rend(i);vis=mk&~pm[i];ys,xs=np.where(vis);m=om[i]&~pm[i];yt,xt=np.where(m);print(i,'render',xs.min(),xs.max(),ys.min(),ys.max(),'target',xt.min(),xt.max(),yt.min(),yt.max(),'cent',round(xs.mean(),1),round(xt.mean(),1),round(ys.mean(),1),round(yt.mean(),1))
PY"
```
<details><summary>output</summary>

```
335 render 885 965 499 676 target 857 962 496 654 cent 925.8 916.7 567.8 572.2
337 render 888 964 498 675 target 860 957 497 655 cent 926.6 918.4 569.1 571.5
339 render 892 958 498 673 target 860 951 496 655 cent 926.9 917.1 571.5 574.8
341 render 897 955 493 669 target 858 943 496 656 cent 927.0 914.6 571.5 579.2
343 render 898 953 491 667 target 864 937 496 656 cent 925.7 913.6 573.8 585.4
345 render 897 952 489 666 target 859 935 498 657 cent 924.9 913.0 574.3 590.7
347 render 897 952 490 666 target 859 935 495 657 cent 925.7 913.0 574.8 587.4
349 render 899 953 490 666 target 859 936 495 657 cent 927.1 913.4 575.9 585.3
351 render 901 955 490 667 target 860 938 493 657 cent 929.1 914.3 577.4 583.8
353 render 904 956 491 668 target 864 941 494 656 cent 931.8 915.6 579.4 581.6
355 render …[+1017 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/build_clean3.py').read();s=s.replace(\"for i in range(345,379):\\n u=(i-344)/36;s=u*u*(3-2*u)\\n # translation follows smooth path, avoid tracker's long detour\\n for j in range(3):ts[i,j]=(1-s)*ts[344,j]+s*ts[380,j]\", \"for i in range(345,379):\\n u=(i-344)/36;s=u*u*(3-2*u)\\n # translation follows smooth path, avoid tracker's long detour\\n for j in range(3):ts[i,j]=(1-s)*ts[344,j]+s*ts[380,j]\\n # image-space alignment correction during the edge-on/occlusion interval\\n bump=np.sin(np.pi*u)\\n ts[i]+=np.array([-0.05,-0.025,0.08])*bump\")
s=s.replace('clean_cam3.npz','clean_cam4.npz');open('/work/scratch/build_clean4.py','w').write(s)
PY
python /work/scratch/build_clean4.py
sed 's/clean_cam3.npz/clean_cam4.npz/' /work/scratch/eval_clean3.py > /work/scratch/eval_clean4.py
python /work/scratch/eval_clean4.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7853332051075033 0.8360306188842301
0 19 0.807 0.923
20 39 0.841 0.922
40 59 0.84 0.916
60 79 0.818 0.876
80 99 0.779 0.841
100 119 0.802 0.85
120 139 0.82 0.861
140 159 0.819 0.868
160 179 0.824 0.859
180 199 0.821 0.825
200 219 0.83 0.829
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.37
360 379 0.428 0.617
380 399 0.662 0.683
400 419 0.68 0.737
420 439 0.694 0.765
440 459 0.614 0.702
460 479 0.609 0.683
480 499 0.632 0.69
500 519 0.64 0.699
520 539 0.651 0.713
540 559 0.664 0.717
worst clean [(354, 0.258), (355, 0.263), (356, 0.283), (353, 0.29), (357, 0.308), (352, 0.328), (358, 0.339), (351, 0.358), (345, 0.358), (346, 0.37), (359, 0.377), (344, 0.37 …[+254 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n 'bump\\|345,379' /work/scratch/build_clean4.py; python - <<'PY'
import numpy as np
a=np.load('/work/scratch/clean_cam3.npz');b=np.load('/work/scratch/clean_cam4.npz');print(np.max(np.abs(a['t']-b['t'])),b['t'][350]-a['t'][350])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '12,28p' /work/scratch/build_clean3.py"
```
<details><summary>output</summary>

```
# Savgol on pre-glitch baseline up to 344
ts=C[:,:3,3].copy()
for j in range(3):ts[190:345,j]=savgol_filter(C[190:345,:3,3][:,j],21,2)
# anchors at last reliable and final settled pose; interpolate smoothly through occluded rotation
for i in range(345,380):
 u=(i-344)/36;s=u*u*(3-2*u)
 # translation follows smooth path, avoid tracker's long detour
 for j in range(3):ts[i,j]=(1-s)*ts[344,j]+s*ts[380,j]
# blend to optimized camera frame static final by 400
for i in range(380,400):
 u=(i-380)/20;s=u*u*(3-2*u);ts[i]=(1-s)*ts[380]+s*tfinal
ts[400:]=tfinal
for i in range(0,191):ts[i]=ti
Rs=[]
for i in range(F):
 if i<=190:R=Ri
 elif i<=344:

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_clean4.py';s=open(p).read();needle=\" for j in range(3):ts[i,j]=(1-s)*ts[344,j]+s*ts[380,j]\\n# blend\"
repl=\" for j in range(3):ts[i,j]=(1-s)*ts[344,j]+s*ts[380,j]\\n bump=np.sin(np.pi*u)\\n ts[i]+=np.array([-0.05,-0.025,0.08])*bump\\n# blend\"
assert needle in s;s=s.replace(needle,repl);open(p,'w').write(s)
PY
python /work/scratch/build_clean4.py
python /work/scratch/eval_clean4.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7793061025655924 0.8360306188842301
0 19 0.807 0.923
20 39 0.841 0.922
40 59 0.84 0.916
60 79 0.818 0.876
80 99 0.779 0.841
100 119 0.802 0.85
120 139 0.82 0.861
140 159 0.819 0.868
160 179 0.824 0.859
180 199 0.821 0.825
200 219 0.83 0.829
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.359
360 379 0.428 0.463
380 399 0.662 0.683
400 419 0.68 0.737
420 439 0.694 0.765
440 459 0.614 0.702
460 479 0.609 0.683
480 499 0.632 0.69
500 519 0.64 0.699
520 539 0.651 0.713
540 559 0.664 0.717
worst clean [(357, 0.292), (359, 0.295), (358, 0.298), (360, 0.301), (356, 0.304), (355, 0.306), (361, 0.306), (354, 0.315), (353, 0.318), (352, 0.321), (362, 0.322), (351, 0 …[+255 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/clean_cam2.npz/clean_cam3.npz/; s/candidate_world.npz/candidate_world3.npz/' /work/scratch/build_world.py > /work/scratch/build_world3.py
python /work/scratch/build_world3.py
sed 's/candidate_world.npz/candidate_world3.npz/' /work/scratch/eval_world_candidate.py > /work/scratch/eval_world_candidate3.py
python /work/scratch/eval_world_candidate3.py"
```
<details><summary>output</summary>

```
off angle spread 0.14768159936673148
static t std [0. 0. 0.] [0. 0. 0.]
rel final angle 176.71484195422636
K base 0.4534824146079515 0.48931314373021384 cand 0.45501415907891646 0.4868634153490793
0 0.56 0.5
20 0.514 0.5
40 0.508 0.504
60 0.498 0.497
80 0.457 0.471
100 0.488 0.488
120 0.492 0.491
140 0.502 0.496
160 0.49 0.487
180 0.485 0.478
200 0.493 0.492
220 0.51 0.51
240 0.524 0.524
260 0.525 0.525
280 0.517 0.518
300 0.504 0.504
320 0.494 0.497
340 0.375 0.33
360 0.342 0.372
380 0.386 0.378
400 0.394 0.423
420 0.407 0.422
440 0.358 0.398
460 0.353 0.39
480 0.354 0.379
500 0.368 0.37
520 0.371 0.37
540 0.367 0.365
K_pipeline base 0.5266313302489372 0.5683275775764658 cand 0.5326607314760654 0.5688533521577155
0 0.632 0.612
20 0.632 0.612
40 0.617 0.61
60 0.576 0.582
80 0.568 0.568
100 …[+361 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;P=np.load('/work/scratch/candidate_world3.npz')['T_world_mesh'];T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
for name,A in [('base',T),('cand',P)]:
 for i in [0,100,190,300,340,360,380,400,450,545]:
  vw=v@A[i,:3,:3].T+A[i,:3,3];print(name,i,'maxy',round(vw[:,1].max(),3),'t',np.round(A[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
base 0 maxy 0.377 t [0.331 0.11  2.583]
base 100 maxy 0.395 t [0.33  0.126 2.619]
base 190 maxy 0.388 t [0.348 0.121 2.624]
base 300 maxy 0.374 t [0.524 0.101 2.625]
base 340 maxy 0.368 t [0.666 0.096 2.604]
base 360 maxy 0.323 t [0.663 0.056 2.669]
base 380 maxy 0.376 t [0.717 0.111 2.483]
base 400 maxy 0.389 t [0.703 0.12  2.514]
base 450 maxy 0.413 t [0.68  0.139 2.491]
base 545 maxy 0.416 t [0.695 0.139 2.566]
cand 0 maxy 0.39 t [0.333 0.122 2.638]
cand 100 maxy 0.39 t [0.333 0.122 2.638]
cand 190 maxy 0.39 t [0.333 0.122 2.638]
cand 300 maxy 0.373 t [0.525 0.1   2.631]
cand 340 maxy 0.37 t [0.661 0.098 2.593]
cand 360 maxy 0.365 t [0.675 0.094 2.542]
cand 380 maxy 0.376 t [0.717 0.111 2.483]
cand 400 maxy 0.414 t [0.689 0.134 2.511]
cand 450 maxy 0.414 t [0.689 0.134 2.511]
cand 545 m …[+32 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;P=np.load('/work/scratch/candidate_world3.npz')['T_world_mesh'];n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n)
def support(M):
 vw=v@M[:3,:3].T+M[:3,3];s=vw@n;return np.percentile(s,0.1),s.min()
for i in [0,100,190,400,450,545]:print(i,support(P[i]))
print('n',n)
PY"
```
<details><summary>output</summary>

```
0 (-0.5458844233387172, -0.5491940826703624)
100 (-0.5458844233387172, -0.5491940826703624)
190 (-0.5458844233387172, -0.5491940826703624)
400 (-0.5602859861534178, -0.5639232287987042)
450 (-0.5602859861534178, -0.5639232287987042)
545 (-0.5602859861534178, -0.5639232287987042)
n [-0.01947626 -0.99822682 -0.05624843]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for path in ['/work/inputs/reconstruction.npz','/work/scratch/candidate_world3.npz']:
 z=np.load(path);T=z['T_world_mesh'].astype(float)
 print(path)
 for lag in [1,5,10]:
  dr=np.degrees(np.linalg.norm((R.from_matrix(T[lag:,:3,:3])*R.from_matrix(T[:-lag,:3,:3]).inv()).as_rotvec(),axis=1));dt=np.linalg.norm(T[lag:,:3,3]-T[:-lag,:3,3],axis=1);print(lag,'rot p95 max',np.percentile(dr,95),dr.max(),'dt p95 max',np.percentile(dt,95),dt.max())
PY"
```
<details><summary>output</summary>

```
/work/inputs/reconstruction.npz
1 rot p95 max 12.438203242448735 118.4074555173418 dt p95 max 0.03888795331355261 0.12147565028495645
5 rot p95 max 47.012096232421534 179.72062962751164 dt p95 max 0.061143157293975285 0.17946210832712825
10 rot p95 max 68.86574723723324 177.18725345502304 dt p95 max 0.1078569758647406 0.20505676873584128
/work/scratch/candidate_world3.npz
1 rot p95 max 1.9599058107501173 11.152029323353622 dt p95 max 0.005323938957468977 0.020123712469490348
5 rot p95 max 8.33589557068573 12.791332939441897 dt p95 max 0.026290718126737958 0.03585045819743905
10 rot p95 max 15.25389612508486 18.581223685665133 dt p95 max 0.05044308795953831 0.06437394614588715

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial import cKDTree
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::20];r=np.load('/work/inputs/reconstruction.npz');K=r['human_keypoints_world'];P=np.load('/work/scratch/candidate_world3.npz')['T_world_mesh'].copy();P2=P.copy();n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n);P2[400:,:3,3]+=(-.0144)*n
for name,A in [('cand',P),('shift',P2)]:
 ds=[]
 for i in range(100,451,5):
  vw=v@A[i,:3,:3].T+A[i,:3,3];tree=cKDTree(vw);h=np.concatenate([K[i,21:42],K[i,42:63]]);ds.append(tree.query(h)[0].min())
 ds=np.array(ds);print(name,'med',np.median(ds),'p90',np.percentile(ds,90),'<5cm',np.mean(ds<.05),'<2cm',np.mean(ds<.02))
PY"
```
<details><summary>output</summary>

```
cand med 0.002983876670780885 p90 0.005216522251426721 <5cm 1.0 <2cm 0.971830985915493
shift med 0.002774030831038663 p90 0.005279528907841168 <5cm 1.0 <2cm 0.971830985915493

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/cd_static.py <<'PY'
exec(open('/work/scratch/opt_static.py').read().split('def optimize')[0])
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

def score(x,R0,t0,idxs):
 R=R0@rotvec_mat(x[:3]);t=t0+x[3:];mk=render(R,t);vals=[]
 for i in idxs:
  vis=mk&~pm[i];vals.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
 return np.mean(vals),mk
for name,idxs in [('init',list(range(0,181,5))),('final',list(range(400,546,5)))]:
 R0=Rot.from_matrix(C[idxs,:3,:3]).mean().as_matrix();t0=np.median(C[idxs,:3,3],0);x=np.zeros(6);best,_=score(x,R0,t0,idxs);print(name,'start',best)
 for step in [.12,.08,.05,.03,.02,.012,.008,.005,.003,.002]:
  for it in range(10):
   moved=False
   # alternate trans/rot; test all coords best improvement
   for grp in [[3,4,5],[0,1,2]]:
    for j in grp:
     cand=[]
     for sg in [1,-1]:
      xx=x.copy();xx[j]+=sg*step;s,_=score(xx,R0,t0,idxs);cand.append((s,xx))
     s,xx=max(cand,key=lambda z:z[0])
     if s>best+1e-6:best=s;x=xx;moved=True
   if not moved:break
  print(name,'step',step,'best',best,'x',np.round(x,4),flush=True)
 R=R0@rotvec_mat(x[:3]);t=t0+x[3:];np.savez('/work/scratch/cd_static_'+name+'.npz',R=R,t=t,x=x,cost=1-best)
PY
python /work/scratch/cd_static.py"
```
<details><summary>output</summary>

```
init start 0.8249867564794157
init step 0.12 best 0.8277034441173068 x [ 0.    0.    0.    0.    0.   -0.12]
init step 0.08 best 0.8358497890385675 x [ 0.    0.    0.    0.    0.   -0.04]
init step 0.05 best 0.8395979292159775 x [ 0.   -0.05  0.    0.    0.   -0.04]
init step 0.03 best 0.8395979292159775 x [ 0.   -0.05  0.    0.    0.   -0.04]
init step 0.02 best 0.8408953683339777 x [ 0.02 -0.03 -0.02  0.    0.   -0.04]
init step 0.012 best 0.8419799029441091 x [ 0.008 -0.03  -0.02   0.     0.    -0.052]
init step 0.008 best 0.8660637614172899 x [ 0.008 -0.006 -0.004 -0.008 -0.008 -0.068]
init step 0.005 best 0.8660637614172899 x [ 0.008 -0.006 -0.004 -0.008 -0.008 -0.068]
init step 0.003 best 0.8693160951199853 x [ 0.005 -0.009 -0.004 -0.008 -0.005 -0.077]
init step 0.002 best 0.86931609 …[+921 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for name in ['init','final']:
 a=np.load('/work/scratch/cd_static_'+name+'.npz');print(name,a['x'],a['cost'])
i=np.load('/work/scratch/cd_static_init.npz');f=np.load('/work/scratch/cd_static_final.npz');q=R.from_matrix(f['R']@i['R'].T);print('rel',np.degrees(q.magnitude()),q.as_rotvec())
PY"
```
<details><summary>output</summary>

```
init [ 0.005 -0.009 -0.004 -0.008 -0.005 -0.077] 0.1306839048800147
final [ 0.039  0.013 -0.01   0.    -0.027  0.072] 0.2982013808277859
rel 177.80553409791156 [-0.06330357 -3.08309742 -0.34774152]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/build_clean3.py').read().replace(\"si=np.load('/work/scratch/static_init.npz');sf=np.load('/work/scratch/static_final.npz')\",\"si=np.load('/work/scratch/cd_static_init.npz');sf=np.load('/work/scratch/cd_static_final.npz')\").replace('clean_cam3.npz','clean_cam_cd.npz')
# critical: final blend uses Rfinal, tfinal arrays fine
open('/work/scratch/build_clean_cd.py','w').write(s)
PY
python /work/scratch/build_clean_cd.py
sed 's/clean_cam3.npz/clean_cam_cd.npz/' /work/scratch/eval_clean3.py > /work/scratch/eval_clean_cd.py
python /work/scratch/eval_clean_cd.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7789884293749623 0.8369721806959637
0 19 0.807 0.891
20 39 0.841 0.891
40 59 0.84 0.888
60 79 0.818 0.871
80 99 0.779 0.849
100 119 0.802 0.851
120 139 0.82 0.861
140 159 0.819 0.866
160 179 0.824 0.86
180 199 0.821 0.831
200 219 0.83 0.831
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.37
360 379 0.428 0.619
380 399 0.662 0.669
400 419 0.68 0.713
420 439 0.694 0.739
440 459 0.614 0.68
460 479 0.609 0.667
480 499 0.632 0.698
500 519 0.64 0.707
520 539 0.651 0.706
540 559 0.664 0.706
worst clean [(354, 0.258), (355, 0.265), (356, 0.285), (353, 0.289), (357, 0.311), (352, 0.327), (358, 0.342), (351, 0.357), (345, 0.358), (346, 0.37), (344, 0.379), (359, 0.3 …[+251 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam_cd.npz')
for i in [0,200,300,330,340,344,345,350,355,356,360,370,380,390,400,545]:
 for nm,R in [('C',C[i,:3,:3]),('cl',cl['R'][i])]:
  f=R@np.array([0,1.,0]);t=C[i,:3,3];v=-t/np.linalg.norm(t);print(i,nm,round(float(f@v),2),end='; ')
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
s=open('/work/scratch/build_clean_cd.py').read()
# replace only rotation section second occurrence line in i<=379: currently s=u*u*(3-2*u) target...
old=\"\"\" elif i<=379:
  u=(i-344)/36;s=u*u*(3-2*u)
  target=(1-s)*th[344]+s*th[380]
  # propagate the last reliable orientation rigidly about the table-normal axis
  R=(Rot.from_rotvec(axis*np.radians(target-th[344])).as_matrix()@C[344,:3,:3])\"\"\"
new=\"\"\" elif i<=379:
  u=(i-344)/36
  # Fast initial turn observed as the display crosses edge-on near frame 345,
  # followed by a slow settle into the final resting orientation.
  s=1-(1-u)**2
  target=(1-s)*th[344]+s*th[380]
  # propagate the last reliable orientation rigidly about the table-normal axis
  R=(Rot.from_rotvec(axis*np.radians(target-th[344])).as_matrix()@C[344,:3,:3])\"\"\"
assert old in s;s=s.replace(old,new).replace('clean_cam_cd.npz','clean_cam_ease.npz');open('/work/scratch/build_clean_ease.py','w').write(s)
PY
python /work/scratch/build_clean_ease.py
sed 's/clean_cam_cd.npz/clean_cam_ease.npz/' /work/scratch/eval_clean_cd.py > /work/scratch/eval_clean_ease.py
python /work/scratch/eval_clean_ease.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7817092036959673 0.8369721806959637
0 19 0.807 0.891
20 39 0.841 0.891
40 59 0.84 0.888
60 79 0.818 0.871
80 99 0.779 0.849
100 119 0.802 0.851
120 139 0.82 0.861
140 159 0.819 0.866
160 179 0.824 0.86
180 199 0.821 0.831
200 219 0.83 0.831
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.418
360 379 0.428 0.645
380 399 0.662 0.669
400 419 0.68 0.713
420 439 0.694 0.739
440 459 0.614 0.68
460 479 0.609 0.667
480 499 0.632 0.698
500 519 0.64 0.707
520 539 0.651 0.706
540 559 0.664 0.706
worst clean [(347, 0.3), (348, 0.3), (349, 0.338), (346, 0.346), (345, 0.349), (350, 0.373), (344, 0.379), (351, 0.381), (352, 0.393), (353, 0.407), (343, 0.43), (354, 0.432) …[+249 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet'].astype(float);cl=np.load('/work/scratch/clean_cam_ease.npz');
for i in range(338,383,2):
 R=cl['R'][i];f=R@np.array([0,1.,0]);t=cl['t'][i];v=-t/np.linalg.norm(t);print(i,round(float(f@v),2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's/s=1-(1-u)\\*\\*2/s=np.sqrt(u)/; s/clean_cam_ease.npz/clean_cam_sqrt.npz/' /work/scratch/build_clean_ease.py > /work/scratch/build_clean_sqrt.py
python /work/scratch/build_clean_sqrt.py
sed 's/clean_cam_ease.npz/clean_cam_sqrt.npz/' /work/scratch/eval_clean_ease.py > /work/scratch/eval_clean_sqrt.py
python /work/scratch/eval_clean_sqrt.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7817847691458389 0.8369721806959637
0 19 0.807 0.891
20 39 0.841 0.891
40 59 0.84 0.888
60 79 0.818 0.871
80 99 0.779 0.849
100 119 0.802 0.851
120 139 0.82 0.861
140 159 0.819 0.866
160 179 0.824 0.86
180 199 0.821 0.831
200 219 0.83 0.831
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.418
360 379 0.428 0.647
380 399 0.662 0.669
400 419 0.68 0.713
420 439 0.694 0.739
440 459 0.614 0.68
460 479 0.609 0.667
480 499 0.632 0.698
500 519 0.64 0.707
520 539 0.651 0.706
540 559 0.664 0.706
worst clean [(345, 0.292), (346, 0.317), (347, 0.341), (348, 0.359), (349, 0.361), (350, 0.367), (351, 0.376), (344, 0.379), (352, 0.387), (353, 0.403), (354, 0.426), (343, 0 …[+254 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
cl=np.load('/work/scratch/clean_cam_sqrt.npz');
for i in range(338,381,2):
 R=cl['R'][i];f=R@np.array([0,1.,0]);t=cl['t'][i];print(i,round(float(f@(-t/np.linalg.norm(t))),2))
PY"
```
<details><summary>output</summary>

```
338 -0.29
340 -0.26
342 -0.17
344 -0.14
346 -0.01
348 0.04
350 0.08
352 0.11
354 0.14
356 0.17
358 0.19
360 0.21
362 0.23
364 0.25
366 0.26
368 0.28
370 0.29
372 0.31
374 0.32
376 0.34
378 0.35
380 0.35

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_final_candidate.py <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as Rot
from scipy.signal import savgol_filter
r=np.load('/work/inputs/reconstruction.npz');C=r['T_cam_mesh_coconet'].astype(float);T=r['T_world_mesh'].astype(float);cl=np.load('/work/scratch/clean_cam_sqrt.npz');Rc=cl['R'];F=len(T)
# constant world<-camera rotation aligned to baseline
X=np.array([T[i,:3,:3]@C[i,:3,:3].T for i in range(F)]);Roff=Rot.from_matrix(X).mean().as_matrix()
Rw=np.array([Roff@Rc[i] for i in range(F)])
# translations: preserve baseline world depth calibration, smooth and repair occlusion detour
tw=T[:,:3,3].copy()
for j in range(3):tw[190:345,j]=savgol_filter(T[190:345,:3,3][:,j],21,2)
for i in range(345,380):
 u=(i-344)/36;s=u*u*(3-2*u);tw[i]=(1-s)*tw[344]+s*tw[380]
# final static translation / rotation pooled
ti=np.median(T[:191,:3,3],0);tf=np.median(T[400:,:3,3],0)
Rwi=Rot.from_matrix(Rw[:191]).mean().as_matrix();Rwf=Rot.from_matrix(Rw[400:]).mean().as_matrix()
# Match support height on the same table: initial pose is untouched and gives the table contact.
n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n)
from trimesh import load
mesh=load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=mesh.vertices
def support(R,t):return np.percentile((v@R.T+t)@n,0.1)
si=support(Rwi,ti);sf=support(Rwf,tf);tf=tf+(si-sf)*n
print('support',si,sf,'shift',si-s …[+878 chars]
```
<details><summary>output</summary>

```
support -0.5441824339426368 -0.5559261772794512 shift 0.011743743336814383 tf [0.68836305 0.1220624  2.51049563]
K base 0.4534824146079515 0.48931314373021384 cand 0.4627910486472911 0.48763464143688845
0 0.56 0.496
20 0.514 0.496
40 0.508 0.499
60 0.498 0.492
80 0.457 0.47
100 0.488 0.488
120 0.492 0.491
140 0.502 0.496
160 0.49 0.487
180 0.485 0.481
200 0.493 0.493
220 0.51 0.51
240 0.524 0.524
260 0.525 0.525
280 0.517 0.518
300 0.504 0.504
320 0.494 0.497
340 0.375 0.42
360 0.342 0.414
380 0.386 0.384
400 0.394 0.44
420 0.407 0.436
440 0.358 0.414
460 0.353 0.405
480 0.354 0.386
500 0.368 0.375
520 0.371 0.377
540 0.367 0.374
K_pipeline base 0.5266313302489372 0.5683275775764658 cand 0.5424674286096096 0.5798009222953293
0 0.632 0.614
20 0.632 0.613
40 0.617 0.611
60 0.576 0.588
80 0.5 …[+368 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_rot_variants.py <<'PY'
exec(open('/work/scratch/eval_all_camera.py').read().split(\"r=np.load\")[0])
r=np.load('/work/inputs/reconstruction.npz');T=r['T_world_mesh'].astype(float);C=r['T_cam_mesh_coconet'].astype(float);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));H,W=1152,1536
# world R offset
from scipy.spatial.transform import Rotation as Rot
X=np.array([T[i,:3,:3]@C[i,:3,:3].T for i in range(546)]);Roff=Rot.from_matrix(X).mean().as_matrix()
variants={}
variants['base_per']=T.copy()
P=T.copy();P[:191,:3,:3]=Rot.from_matrix(T[:191,:3,:3]).mean().as_matrix();P[400:,:3,:3]=Rot.from_matrix(T[400:,:3,:3]).mean().as_matrix();variants['base_pool']=P
si=np.load('/work/scratch/cd_static_init.npz');sf=np.load('/work/scratch/cd_static_final.npz');P=T.copy();P[:191,:3,:3]=Roff@si['R'];P[400:,:3,:3]=Roff@sf['R'];variants['cd']=P
si=np.load('/work/scratch/static_init.npz');sf=np.load('/work/scratch/static_final.npz');P=T.copy();P[:191,:3,:3]=Roff@si['R'];P[400:,:3,:3]=Roff@sf['R'];variants['powell']=P
for name,P in variants.items():
 for kk in ['K','K_pipeline']:
  K=np.array(cam[kk]);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];ious=[]
  for i in range(546):
   M=Tcw@P[i];v=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);Xc=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([Xc,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H, …[+305 chars]
```
<details><summary>output</summary>

```
base_per K init 0.5001 final 0.3718 overall 0.4535
base_per K_pipeline init 0.5895 final 0.4238 overall 0.5266
base_pool K init 0.4986 final 0.3719 overall 0.453
base_pool K_pipeline init 0.5894 final 0.4257 overall 0.5271
cd K init 0.4982 final 0.3809 overall 0.4552
cd K_pipeline init 0.5912 final 0.4373 overall 0.5308
powell K init 0.4971 final 0.4022 overall 0.4606
powell K_pipeline init 0.5828 final 0.4606 overall 0.5341

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s#si=np.load('/work/scratch/cd_static_init.npz');sf=np.load('/work/scratch/cd_static_final.npz')#si=np.load('/work/scratch/static_init.npz');sf=np.load('/work/scratch/static_final.npz')#; s/clean_cam_sqrt.npz/clean_cam_powell_sqrt.npz/\" /work/scratch/build_clean_sqrt.py > /work/scratch/build_clean_powell_sqrt.py
python /work/scratch/build_clean_powell_sqrt.py
sed 's/clean_cam_sqrt.npz/clean_cam_powell_sqrt.npz/' /work/scratch/eval_clean_sqrt.py > /work/scratch/eval_clean_powell_sqrt.py
python /work/scratch/eval_clean_powell_sqrt.py
# build world using this and eval
sed \"s#clean_cam_sqrt.npz#clean_cam_powell_sqrt.npz#; s#candidate_final.npz#candidate_powell.npz#\" /work/scratch/build_final_candidate.py > /work/scratch/build_candidate_powell.py
python /work/scratch/build_candidate_powell.py
sed 's/candidate_world.npz/candidate_powell.npz/' /work/scratch/eval_world_candidate.py > /work/scratch/eval_candidate_powell.py
python /work/scratch/eval_candidate_powell.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7882174222683928 0.8360306188842301
0 19 0.807 0.923
20 39 0.841 0.922
40 59 0.84 0.916
60 79 0.818 0.876
80 99 0.779 0.841
100 119 0.802 0.85
120 139 0.82 0.861
140 159 0.819 0.868
160 179 0.824 0.859
180 199 0.821 0.825
200 219 0.83 0.829
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.419
360 379 0.428 0.647
380 399 0.662 0.683
400 419 0.68 0.737
420 439 0.694 0.765
440 459 0.614 0.702
460 479 0.609 0.683
480 499 0.632 0.69
500 519 0.64 0.699
520 539 0.651 0.713
540 559 0.664 0.717
worst clean [(345, 0.292), (346, 0.317), (347, 0.341), (348, 0.36), (349, 0.361), (350, 0.367), (351, 0.375), (344, 0.379), (352, 0.388), (353, 0.403), (354, 0.427), (343, 0. …[+1423 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF monitor trajectory.

The method uses the released CARI4D/CoCoNet camera-frame track as a proposal
track.  It (1) pools all frames in the two static phases to estimate one pose
per phase, (2) repairs the long, physically impossible yaw reversal during the
fast edge-on turn by propagating yaw about the table normal, and (3) smooths or
replaces tracker translation detours while retaining the baseline'"'s human/floor
calibrated world depth.  Silhouette scoring uses nvdiffrast and the SAM2 masks;
no ground-truth data is used.
\"\"\"
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# Phase boundaries were verified visually from the video and contact signal.
# The monitor is untouched through frame 190, is turned quickly while edge-on
# around frames 345-360, and is resting again from frame 400 onward.
STATIC_INITIAL_END = 190
TURN_START = 344
TURN_END = 380
SETTLE_END = 399
STATIC_FINAL_START = 400


def robust_mean_rotation(mats: np.ndarray) -> Rotation:
    return Rotation.from_matrix(mats).mean()


def load_inputs(inputs: Path):
    episode = json.loads((inputs / \"episode.json\").read_text())
    camera = json.loa …[+10107 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "/usr/bin/time -f '\\nELAPSED=%e MAXRSS_KB=%M' python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'start=$(date +%s); python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz; rc=$?; end=$(date +%s); echo ELAPSED=$((end-start))s; exit $rc'
```
<details><summary>output</summary>

```
static initial pooled mask IoU: 0.8771
static final pooled mask IoU:   0.7134
ELAPSED=142s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz');print(p.files)
T=p['T_world_mesh'];print(T.shape,T.dtype,'finite',np.isfinite(T).all())
print('det range',np.linalg.det(T[:,:3,:3]).min(),np.linalg.det(T[:,:3,:3]).max())
print('last row unique',np.unique(T[:,3,:],axis=0))
for i in [0,190,191,300,340,344,345,360,380,399,400,545]:print(i,np.round(T[i],4))
q=np.load('/work/scratch/candidate_powell.npz')['T_world_mesh'];print('diff candidate',np.abs(T-q).max())
PY"
```
<details><summary>output</summary>

```
['T_world_mesh']
(546, 4, 4) float64 finite True
det range 0.9999943328880727 1.0000000000000016
last row unique [[0. 0. 0. 1.]]
0 [[-1.7000e-03 -6.9930e-01  7.1480e-01  3.3330e-01]
 [ 9.9950e-01 -2.4400e-02 -2.1600e-02  1.2250e-01]
 [ 3.2600e-02  7.1440e-01  6.9900e-01  2.6384e+00]
 [ 0.0000e+00  0.0000e+00  0.0000e+00  1.0000e+00]]
190 [[-1.7000e-03 -6.9930e-01  7.1480e-01  3.3330e-01]
 [ 9.9950e-01 -2.4400e-02 -2.1600e-02  1.2250e-01]
 [ 3.2600e-02  7.1440e-01  6.9900e-01  2.6384e+00]
 [ 0.0000e+00  0.0000e+00  0.0000e+00  1.0000e+00]]
191 [[-1.7000e-03 -6.9930e-01  7.1480e-01  3.4950e-01]
 [ 9.9950e-01 -2.4400e-02 -2.1600e-02  1.2170e-01]
 [ 3.2600e-02  7.1440e-01  6.9900e-01  2.6264e+00]
 [ 0.0000e+00  0.0000e+00  0.0000e+00  1.0000e+00]]
300 [[ 0.032   0.6184  0.7852  0.5255]
 [ 0.99 …[+1405 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "git -C /workspace/v2d_cari4d rev-parse HEAD 2>/dev/null || true; git -C /workspace/v2d_foundation_pose rev-parse HEAD 2>/dev/null || true; python - <<'PY'
import importlib.metadata as m
for p in ['nvdiffrast','torch','trimesh','opencv-python','opencv-python-headless','scipy','numpy','h5py','open3d','pyrender']:
 try:print(p,m.version(p))
 except:pass
PY"
```
<details><summary>output</summary>

```
nvdiffrast 0.4.0
torch 2.5.1+cu124
trimesh 5.1.1
opencv-python 4.11.0.86
scipy 1.16.3
numpy 1.26.3
h5py 3.16.0
open3d 0.20.0
pyrender 0.1.45

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];q=np.load('/work/scratch/candidate_powell.npz')['T_world_mesh']
d=np.abs(p-q);idx=np.unravel_index(np.argmax(d),d.shape);print(idx,d[idx],p[idx[0]],q[idx[0]])
for i in [0,190,191,300,340,344,345,360,380,399,400,545]:print(i,'rdiff',np.max(np.abs(p[i]-q[i])),'p t',p[i,:3,3],'q t',q[i,:3,3])
PY"
```
<details><summary>output</summary>

```
[[-0.08625985  0.77850897 -0.62167758  0.71680361]
 [ 0.99464735  0.1029245  -0.00912126  0.11143062]
 [ 0.05688488 -0.61913676 -0.78322014  2.4829967 ]
 [ 0.          0.          0.          1.        ]]
0 rdiff 0.0026294228469957437 p t [0.33332717 0.1224723  2.63836551] q t [0.33332717 0.1224723  2.63836551]
190 rdiff 0.0026294228469957437 p t [0.33332717 0.1224723  2.63836551] q t [0.33332717 0.1224723  2.63836551]
191 rdiff 0.00263217161559659 p t [0.34947561 0.1217276  2.62638063] q t [0.34947561 0.1217276  2.62638063]
300 rdiff 6.661338147750939e-16 p t [0.5254875  0.09996682 2.6310934 ] q t [0.5254875  0.09996682 2.6310934 ]
340 rdiff 2.220446049250313e-16 p t [0.6613607  0.09840877 2.59344552] q t [0.6613607  0.09840877 2.59344552]
344 rdiff 3.885780586188048e-16 p t [0.64518769 0 …[+719 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"elif i<=399\\|for i in range(380,400)\" /work/scratch/build_clean_powell_sqrt.py | head; sed -n '31,52p' /work/scratch/build_clean_powell_sqrt.py"
```
<details><summary>output</summary>

```
21:for i in range(380,400):
38: elif i<=399:
44:for i in range(380,400):
  u=(i-344)/36
  # Fast initial turn observed as the display crosses edge-on near frame 345,
  # followed by a slow settle into the final resting orientation.
  s=np.sqrt(u)
  target=(1-s)*th[344]+s*th[380]
  # propagate the last reliable orientation rigidly about the table-normal axis
  R=(Rot.from_rotvec(axis*np.radians(target-th[344])).as_matrix()@C[344,:3,:3])
 elif i<=399:
  R=C[i,:3,:3]
 else:R=Rfinal
 Rs.append(R)
Rs=np.array(Rs)
# Better 380-400: derive final *camera* rotation consistent with optimized static camera pose directly (Rfinal is camera pose)
for i in range(380,400):
 u=(i-380)/20;s=u*u*(3-2*u);Rs[i]=(Rot.from_rotvec(s*(Rot.from_matrix(Rfinal)*Rot.from_matrix(Rs[380]).inv()).as_rotvec())*Rot.from_ma …[+96 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/build_clean_powell_sqrt.py';s=open(p).read().replace(\" elif i<=399:\\n  R=C[i,:3,:3]\",\" elif i<=399:\\n  R=(Rot.from_rotvec(axis*np.radians(th[380]-th[344])).as_matrix()@C[344,:3,:3])\");s=s.replace('clean_cam_powell_sqrt.npz','clean_cam_powell_sqrt2.npz');open('/work/scratch/build_clean_powell_sqrt2.py','w').write(s)
PY
python /work/scratch/build_clean_powell_sqrt2.py
sed 's/clean_cam_powell_sqrt.npz/clean_cam_powell_sqrt2.npz/' /work/scratch/eval_clean_powell_sqrt.py > /work/scratch/eval_clean_powell_sqrt2.py
python /work/scratch/eval_clean_powell_sqrt2.py"
```
<details><summary>output</summary>

```
coco 0.737432367592342 0.8105209064497116
clean 0.7889606371920355 0.8360306188842301
0 19 0.807 0.923
20 39 0.841 0.922
40 59 0.84 0.916
60 79 0.818 0.876
80 99 0.779 0.841
100 119 0.802 0.85
120 139 0.82 0.861
140 159 0.819 0.868
160 179 0.824 0.859
180 199 0.821 0.825
200 219 0.83 0.829
220 239 0.835 0.834
240 259 0.844 0.844
260 279 0.866 0.866
280 299 0.846 0.845
300 319 0.847 0.847
320 339 0.756 0.76
340 359 0.33 0.419
360 379 0.428 0.647
380 399 0.662 0.703
400 419 0.68 0.737
420 439 0.694 0.765
440 459 0.614 0.702
460 479 0.609 0.683
480 499 0.632 0.69
500 519 0.64 0.699
520 539 0.651 0.713
540 559 0.664 0.717
worst clean [(345, 0.292), (346, 0.317), (347, 0.341), (348, 0.36), (349, 0.361), (350, 0.367), (351, 0.375), (344, 0.379), (352, 0.388), (353, 0.403), (354, 0.427), (343, 0. …[+256 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s#clean_cam_sqrt.npz#clean_cam_powell_sqrt2.npz#; s#candidate_final.npz#candidate_powell2.npz#\" /work/scratch/build_final_candidate.py > /work/scratch/build_candidate_powell2.py
python /work/scratch/build_candidate_powell2.py
sed 's/candidate_world.npz/candidate_powell2.npz/' /work/scratch/eval_world_candidate.py > /work/scratch/eval_candidate_powell2.py
python /work/scratch/eval_candidate_powell2.py"
```
<details><summary>output</summary>

```
support -0.5458844233387172 -0.5602859861534178 shift 0.01440156281470062 tf [0.68831129 0.11940929 2.51034614]
K base 0.4534824146079515 0.48931314373021384 cand 0.47164041009704444 0.4868634153490793
0 0.56 0.5
20 0.514 0.5
40 0.508 0.504
60 0.498 0.497
80 0.457 0.471
100 0.488 0.488
120 0.492 0.491
140 0.502 0.496
160 0.49 0.487
180 0.485 0.478
200 0.493 0.492
220 0.51 0.51
240 0.524 0.524
260 0.525 0.525
280 0.517 0.518
300 0.504 0.504
320 0.494 0.497
340 0.375 0.42
360 0.342 0.413
380 0.386 0.413
400 0.394 0.465
420 0.407 0.463
440 0.358 0.439
460 0.353 0.43
480 0.354 0.416
500 0.368 0.407
520 0.371 0.408
540 0.367 0.404
K_pipeline base 0.5266313302489372 0.5683275775764658 cand 0.5488019935351084 0.5688533521577155
0 0.632 0.612
20 0.632 0.612
40 0.617 0.61
60 0.576 0.582
80 0.568 0. …[+367 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace(\"    rotations = np.empty((F, 3, 3), dtype=np.float64)\\n    for i in range(F):\", \"    turn_end_R = (\\n        Rotation.from_rotvec(axis * np.radians(yaw[TURN_END] - yaw[TURN_START])).as_matrix()\\n        @ C[TURN_START, :3, :3]\\n    )\\n\\n    rotations = np.empty((F, 3, 3), dtype=np.float64)\\n    for i in range(F):\")
old=\"\"\"        elif i <= SETTLE_END:
            u = (i - TURN_END) / 20.0
            s = u * u * (3.0 - 2.0 * u)
            delta = s * (Rotation.from_matrix(Rf) * Rotation.from_matrix(rotations[TURN_END]).inv()).as_rotvec()
            R = (Rotation.from_rotvec(delta) * Rotation.from_matrix(rotations[TURN_END])).as_matrix()
\"\"\"
new=\"\"\"        elif i <= SETTLE_END:
            u = (i - TURN_END) / 20.0
            s = u * u * (3.0 - 2.0 * u)
            delta = s * (Rotation.from_matrix(Rf) * Rotation.from_matrix(turn_end_R).inv()).as_rotvec()
            R = (Rotation.from_rotvec(delta) * Rotation.from_matrix(turn_end_R)).as_matrix()
\"\"\"
assert old in s;s=s.replace(old,new);open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'start=$(date +%s); python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz; rc=$?; end=$(date +%s); echo ELAPSED=$((end-start))s; exit $rc'
```
<details><summary>output</summary>

```
static initial pooled mask IoU: 0.8771
static final pooled mask IoU:   0.7134
ELAPSED=137s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];q=np.load('/work/scratch/candidate_powell2.npz')['T_world_mesh'];d=np.abs(p-q);print('max',d.max(),'mean',d.mean());idx=np.unravel_index(np.argmax(d),d.shape);print(idx,p[idx[0],:3,:3],q[idx[0],:3,:3],p[idx[0],:3,3],q[idx[0],:3,3]);
for i in [0,190,300,340,344,345,360,379,380,381,399,400,545]:print(i,np.max(np.abs(p[i]-q[i])))
PY"
```
<details><summary>output</summary>

```
max 0.004756745794427364 mean 0.0004090599598073912
(205, 1, 1) [[-0.02980339 -0.6858559   0.72712684]
 [ 0.9976419  -0.06540579 -0.02080219]
 [ 0.06182561  0.72479222  0.6861879 ]] [[-0.02718701 -0.68643129  0.72668628]
 [ 0.9979597  -0.06064904 -0.01995339]
 [ 0.05776946  0.72466115  0.68667962]] [0.36354875 0.11841846 2.61665069] [0.36354875 0.11841846 2.61665069]
0 0.0026294228469957437
190 0.0026294228469957437
300 6.661338147750939e-16
340 2.220446049250313e-16
344 3.885780586188048e-16
345 2.1772504944617133e-05
360 7.88302173825306e-05
379 0.0001041282665767973
380 0.00010497457921465614
381 0.00010323501791043022
399 0.00018291149504461035
400 0.0001842472878817586
545 0.0001842472878817586

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/outputs/evaluate.py <<'PY'
import json
import numpy as np
import trimesh, torch, nvdiffrast.torch as dr, h5py
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from pathlib import Path
inputs=Path('/work/inputs'); result=Path('/work/outputs/object_poses_world.npz')
ep=json.load(open(inputs/'episode.json')); cam=json.load(open(inputs/'camera.json')); rec=np.load(inputs/'reconstruction.npz'); Tout=np.load(result)['T_world_mesh']; Tbase=rec['T_world_mesh'].astype(float)
with h5py.File(inputs/'masks.h5','r') as h: om=h['object'][:]; pm=h['person'][:]
mesh=trimesh.load(inputs/'object_mesh.glb',force='scene',process=False).to_geometry();v=mesh.vertices
V=torch.tensor(v,dtype=torch.float32,device='cuda');Fa=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();Tcw=np.linalg.inv(np.array(cam['T_world_cam']));H,W=1152,1536
phases=[('P0 untouched/static',0,190),('P1 approach+turn',191,339),('P2 edge-on flip',340,379),('P3 settle',380,399),('P4 resting/static',400,545)]
def ious(P,K):
 fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];out=[]
 for i in range(len(P)):
  M=Tcw@P[i];x,y,z=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[ …[+2598 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/outputs/evaluate.py", line 20, in <module>
    K=np.array(cam[kn]);a=ious(Tbase,K);b=ious(Tout,K);ratings[kn]=(a,b)
                          ^^^^^^^^^^^^^
  File "/work/outputs/evaluate.py", line 16, in ious
    M=Tcw@P[i];x,y,z=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];out.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
               ^^^^^
ValueError: too many values to unpack (expected 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/evaluate.py';s=open(p).read().replace(\"M=Tcw@P[i];x,y,z=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');X=\", \"M=Tcw@P[i];vv=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=vv.unbind(1);X=\")
open(p,'w').write(s)
PY
python /work/outputs/evaluate.py"
```
<details><summary>output</summary>

```
{
  "phase_iou_K": {
    "P0 untouched/static": [
      0.5000899001771729,
      0.4905832123021194
    ],
    "P1 approach+turn": [
      0.5082481076804987,
      0.5081856526873164
    ],
    "P2 edge-on flip": [
      0.35844345723453164,
      0.4162570129900285
    ],
    "P3 settle": [
      0.3863968693410472,
      0.41363275229010765
    ],
    "P4 resting/static": [
      0.37184646384592496,
      0.43186901962310464
    ]
  },
  "phase_iou_Kpipeline": {
    "P0 untouched/static": [
      0.5895497685309555,
      0.5866563021238438
    ],
    "P1 approach+turn": [
      0.6046379651237529,
      0.6052133769507351
    ],
    "P2 edge-on flip": [
      0.35094130494888437,
      0.4112839996076906
    ],
    "P3 settle": [
      0.44676074475311145,
      0.4810468290471485
   …[+1448 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for name,path in [('baseline','/work/inputs/reconstruction.npz'),('result','/work/outputs/object_poses_world.npz')]:
 P=np.load(path)['T_world_mesh'].astype(float);print(name)
 for a,b in [(0,191),(400,546)]:
  tr=np.linalg.norm(np.ptp(P[a:b,:3,3],axis=0));r=np.degrees(np.linalg.norm((Rot.from_matrix(P[a:b,:3,:3])*Rot.from_matrix(P[a,:3,:3]).inv()).as_rotvec(),axis=1));print(a,b,'t span',tr,'rot max',r.max())
PY"
```
<details><summary>output</summary>

```
baseline
0 191 t span 0.15293952447119907 rot max 2.2192249893000944
400 546 t span 0.18479732099981142 rot max 7.29301200828927
result
0 191 t span 0.0 rot max 0.0
400 546 t span 0.0 rot max 0.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'for d in /code/reconstruction /workspace/v2d_cari4d /workspace/v2d_foundation_pose /workspace/third_party/nvdiffrast /workspace/third_party/kaolin; do echo $d; git -C "$d" rev-parse HEAD 2>/dev/null || echo no-git; done'
```
<details><summary>output</summary>

```
/code/reconstruction
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
/workspace/v2d_cari4d
no-git
/workspace/v2d_foundation_pose
no-git
/workspace/third_party/nvdiffrast
253ac4fcea7de5f396371124af597e6cc957bfae
/workspace/third_party/kaolin
06ffb7d955ca26b608c60a9e862327c56b226921

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import importlib.metadata as m
for p in ['pillow','nvdiffrast','torch','trimesh','opencv-python','opencv-python-headless','scipy','numpy','h5py','open3d','pyrender']:
 try: print(p,m.version(p))
 except Exception as e: print(p,'not found')
PY"
```
<details><summary>output</summary>

```
pillow 10.2.0
nvdiffrast 0.4.0
torch 2.5.1+cu124
trimesh 5.1.1
opencv-python 4.11.0.86
opencv-python-headless not found
scipy 1.16.3
numpy 1.26.3
h5py 3.16.0
open3d 0.20.0
pyrender 0.1.45

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'du -h /work/outputs/object_poses_world.npz /work/outputs/solve.py; sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py'
```
<details><summary>output</summary>

```
72K	/work/outputs/object_poses_world.npz
12K	/work/outputs/solve.py
cf9f0c24d52cffed83fb336f8d9e025a9aec5cb4423e4da8d79527e4842362fb  /work/outputs/object_poses_world.npz
9f9ab05b0ad2974bc5811ab42e345aeec74c7dc46565166819508e37e01bd014  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Monitor 6-DoF trajectory recovery

## Summary

- Input episode: "'`episode_000006`, object `monitor`, 546 frames.
- Output: `object_poses_world.npz`, key `T_world_mesh`, shape `(546, 4, 4)`, `float64`.
- The released `T_world_mesh` is used as the depth/world-calibration fallback. Its camera-frame CoCoNet track supplies the object proposal in every frame.
- I did not use ground truth or challenge annotations. Validation uses only the supplied video, masks, reconstructed human, mesh, and camera/floor metadata.

The main correction is temporal. The baseline'"'s yaw reverses direction for about 30 frames during the fast edge-on turn and then jumps back near the final 180-degree orientation. The corrected track keeps one rigid object, turns monotonically about the table-normal axis, pools the two static phases into single poses, removes tracker translation detours, and preserves the baseline's human/floor-calibrated world depth.

## Phases

| Phase | Frames | Interpretation | Confidence |
|---|---:|---|---|
| P0 | 0-190 | Monitor untouched on the table; static world pose. | High |
| P1 | 191-339 | Person approaches and begins carrying/turning the monitor; baseline pose is mostly reliable. | High |
| P2 | 340-379 | Fast front-to-back turn. The display crosses edge-on near frame 345 and is heavily occluded by the person. Baseline yaw oscillates/reverses here. | Medium |
| P3 | 380-399 | Monitor settles toward its final 180-deg …[+6799 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py'
```
<details><summary>output</summary>

```
cf9f0c24d52cffed83fb336f8d9e025a9aec5cb4423e4da8d79527e4842362fb  /work/outputs/object_poses_world.npz
9f9ab05b0ad2974bc5811ab42e345aeec74c7dc46565166819508e37e01bd014  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/TOOLS.md <<'MD'
# Tools and provenance

No external repository, package, or model weight was downloaded during this task. The final solver is self-contained and uses only software already present in the container plus the supplied inputs.

## Required by "'`solve.py`

| Tool | Version / revision | Use | Provenance |
|---|---|---|---|
| Python | 3.11 | Runtime | Container |
| NumPy | 1.26.3 | Array math, NPZ I/O | Installed |
| SciPy | 1.16.3 | Rotation averaging, Powell optimization, Savitzky-Golay filter | Installed |
| h5py | 3.16.0 | Read SAM2 masks | Installed |
| trimesh | 5.1.1 | Load exact GLB mesh | Installed |
| PyTorch | 2.5.1+cu124 | GPU tensor/mesh projection | Installed |
| nvdiffrast | 0.4.0, commit `253ac4fcea7de5f396371124af597e6cc957bfae` | Differentiable/rasterized mesh silhouettes | `/workspace/third_party/nvdiffrast` |
| CUDA runtime / one GPU | Container-provided | Projection and rasterization | Container |

`/usr/bin/time` was not available, so runtime was measured with the shell `date` command.

## Existing reconstruction software and intermediate outputs

The supplied baseline and camera-frame proposals were produced by the container'"'s CARI4D/FoundationPose stack. No such model was rerun for the final solution.

| Tool | Revision / identifier | Role |
|---|---|---|
| CARI4D reconstruction repository | "'`7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | Source of supplied CARI4D pipeline behavior/intermediates |
| CARI …[+1645 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/final_texture_scan.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,cv2
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation as Rot
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');UV=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda');ctx=dr.RasterizeCudaContext();tex=torch.tensor(np.array(m.visual.material.baseColorTexture),dtype=torch.float32,device='cuda').permute(2,0,1)[None]/255.;cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K_pipeline']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536;Tcw=np.linalg.inv(np.array(cam['T_world_cam']));P=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n)
with h5py.File('/work/inputs/masks.h5','r') as h:o=h['object'][:];p=h['person'][:]
def rend(M):
 Pt=Tcw@M;v=V@torch.tensor(Pt[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(Pt[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();uv=dr.interpolate(UV[None],rast,Fa)[0][0];uu=uv[...,0];vv=1-uv[...,1];s=torch …[+923 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed \"s/K=np.array(cam\\['K_pipeline'\\])/K=np.array(cam['K'])/; s/final_texture_scan.jpg/final_texture_scan_K.jpg/\" /work/scratch/final_texture_scan.py > /work/scratch/final_texture_scan_K.py
python /work/scratch/final_texture_scan_K.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static_worldK.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py,time
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=torch.tensor(m.vertices,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536;Tcw=np.linalg.inv(np.array(cam['T_world_cam']))
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);P=np.array([Tcw@t for t in T])
def rotmat(v):return Rot.from_rotvec(v).as_matrix()
def render(R,t):
 Rt=torch.tensor(R,dtype=torch.float32,device='cuda');tt=torch.tensor(t,dtype=torch.float32,device='cuda');v=V@Rt.T+tt;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
def optimize(name,idxs):
 R0=Rot.from_matrix(P[idxs,:3,:3]).mean().as_matrix();t0=np.median(P[idxs,:3,3],0);n=[0]
 def fun(x):
  n[0]+=1;R=R0@rotmat(x[:3]);t=t0+x[3:];mk=render(R,t);v=[]
  for i in idxs:
   vis=mk&~pm[i];v.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
 …[+493 chars]
```
<details><summary>output</summary>

```
init 100 0.21933132557293744
init 200 0.2319179089068537
init 300 0.1571659299489827
init 400 0.14128022030817655
init 500 0.16322335948848088
init 600 0.12713597400781995
init 700 0.12436455295543991
init 800 0.12370182767026394
init IoU 0.8767272619504552 x [-0.01448521 -0.04609036  0.03006207 -0.01360803 -0.1017877  -0.14168037] R [[ 0.01285866 -0.74124015  0.67111675]
 [ 0.99329776 -0.06763223 -0.09373064]
 [ 0.11486603  0.66782402  0.73540252]] t [ 0.19701859 -0.02344217  2.5085349 ]
final 100 0.3810276630818632
final 200 0.3592630101416867
final 300 0.31868118666784084
final 400 0.44385616161786534
final 500 0.28377320008622986
final 600 0.280016560675482
final 700 0.2793564473000374
final 800 0.27844470378342145
final IoU 0.7216162837574941 x [-0.00876908  0.03115716  0.00823215 -0. …[+194 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
from scipy.spatial import cKDTree
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;r=np.load('/work/inputs/reconstruction.npz');Kp=r['human_keypoints_world'];T=r['T_world_mesh'].astype(float);Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n);vs=v[::20]
for name in ['init','final']:
 a=np.load('/work/scratch/static_worldK_'+name+'.npz');P=Twc@np.block([[a['R'],a['t'][:,None]],[np.zeros((1,3)),np.ones((1,1))]]);print(name,'cam R',a['R'],'t',a['t'],'world t',P[:3,3],'support',np.percentile((v@P[:3,:3].T+P[:3,3])@n,.1))
 # contact static range
 ds=[]
 for i in (range(0,191,5) if name=='init' else range(400,546,5)):
  vw=vs@P[:3,:3].T+P[:3,3];tree=cKDTree(vw);hh=np.concatenate([Kp[i,21:42],Kp[i,42:63]]);ds.append(tree.query(hh)[0].min())
 print('contact',np.median(ds),np.percentile(ds,90))
PY"
```
<details><summary>output</summary>

```
init cam R [[ 0.01285866 -0.74124015  0.67111675]
 [ 0.99329776 -0.06763223 -0.09373064]
 [ 0.11486603  0.66782402  0.73540252]] t [ 0.19701859 -0.02344217  2.5085349 ] world t [0.31150445 0.01131644 2.50407927] support -0.42296452675383317
contact 0.18947844052808466 0.7764366629172931
final cam R [[ 0.05166192  0.73451566 -0.67662234]
 [ 0.99090378  0.04660066  0.12624606]
 [ 0.12426076 -0.67698975 -0.72542687]] t [ 0.52059842 -0.00344325  2.37946863] world t [0.6261583  0.02047197 2.35351003] support -0.43552171566803444
contact 0.42573822559892144 0.774762752387348

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
for name in ['init','final']:
 a=np.load('/work/scratch/static_worldK_'+name+'.npz');P=Twc@np.block([[a['R'],a['t'][:,None]],[np.zeros((1,3)),np.ones((1,1))]]);w=v@P[:3,:3].T+P[:3,3];print(name,'ymax',w[:,1].max(),'bottom median',np.median(w[w[:,1]>np.percentile(w[:,1],99)],0))
PY"
```
<details><summary>output</summary>

```
init ymax 0.2813623950954998 bottom median [0.28602833 0.27565505 2.41790288]
final ymax 0.2985252847842508 bottom median [0.52553545 0.29059968 2.32897036]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static_worldK_fixdepth.py <<'PY'
exec(open('/work/scratch/opt_static_worldK.py').read().split(\"def optimize\")[0])
from scipy.optimize import minimize
# existing defs P,R rotmat, render loaded
def optimize(name,idxs,zlim):
 R0=Rot.from_matrix(P[idxs,:3,:3]).mean().as_matrix();t0=np.median(P[idxs,:3,3],0);n=[0]
 def fun(x):
  n[0]+=1;R=R0@rotmat(x[:3]);t=t0+x[3:];mk=render(R,t);v=[]
  for i in idxs:
   vis=mk&~pm[i];v.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
  return 1-np.mean(v)
 # bounds rot +-0.2, tx ty +-.2, tz small
 bounds=[(-.2,.2)]*5+[(-zlim,zlim)];res=minimize(fun,np.zeros(6),method='Powell',bounds=bounds,options={'xtol':1e-4,'ftol':1e-5,'maxiter':8})
 R=R0@rotmat(res.x[:3]);t=t0+res.x[3:];print(name,'iou',1-res.fun,'x',res.x,'t',t,'R',R);np.savez('/work/scratch/static_worldKfix_'+name+'.npz',R=R,t=t,iou=1-res.fun)
optimize('init',list(range(0,191,5)),.02);optimize('final',list(range(400,546,5)),.02)
PY
python /work/scratch/opt_static_worldK_fixdepth.py"
```
<details><summary>output</summary>

```
init iou 0.8487121523861699 x [ 0.05265391 -0.02325043  0.04251482 -0.00651696 -0.10836439 -0.01928122] t [ 0.20410965 -0.03001886  2.63093405] R [[-0.01020479 -0.69412469  0.71978245]
 [ 0.99420413 -0.08408465 -0.06699189]
 [ 0.10702339  0.71492705  0.6909597 ]]
final iou 0.7043347728826848 x [ 0.06983931  0.03594855  0.12499299 -0.01071532 -0.115064   -0.015005  ] t [ 0.56025383 -0.0151709   2.53095509] R [[ 0.13744161  0.66924465 -0.7302201 ]
 [ 0.98956331 -0.06055277  0.13075861]
 [ 0.04329265 -0.74057069 -0.67058243]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n)
for name in ['init','final']:
 a=np.load('/work/scratch/static_worldKfix_'+name+'.npz');P=Twc@np.block([[a['R'],a['t'][:,None]],[np.zeros((1,3)),np.ones((1,1))]]);w=v@P[:3,:3].T+P[:3,3];print(name,'iou',a['iou'],'world t',P[:3,3],'maxy',w[:,1].max(),'support',np.percentile(w@n,.1))
PY"
```
<details><summary>output</summary>

```
init iou 0.8487121523861699 world t [0.32645206 0.01248565 2.62594103] maxy 0.2806276418287513 support -0.434324344352317
final iou 0.7043347728826848 world t [0.6754588  0.01809132 2.50258058] maxy 0.30312545923802514 support -0.4486639748866405

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);Tcw=np.linalg.inv(np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']));P=Tcw@T[0];w=v@P[:3,:3].T+P[:3,3];print('cam t',P[:3,3],'maxy',w[:,1].max(),'world t',T[0,:3,3]);
PY"
```
<details><summary>output</summary>

```
cam t [0.21062662 0.06982687 2.59500571] maxy 0.33996717856239955 world t [0.33102539 0.1097873  2.58335042]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static_worldK_support.py <<'PY'
import numpy as np,trimesh,torch,nvdiffrast.torch as dr,json,h5py
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();verts=m.vertices;V=torch.tensor(verts,dtype=torch.float32,device='cuda');Fa=torch.tensor(m.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();cam=json.load(open('/work/inputs/camera.json'));K=np.array(cam['K']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536;Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc)
with h5py.File('/work/inputs/masks.h5','r') as h:om=h['object'][:];pm=h['person'][:]
T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);n/=np.linalg.norm(n)
# tangent basis
u=np.array([1.,0.,0.]);u=u-n*(u@n);u/=np.linalg.norm(u);vv=np.cross(n,u);B=np.stack([u,vv,n],1)
def render_world(M):
 C=Tcw@M;Rt=torch.tensor(C[:3,:3],dtype=torch.float32,device='cuda');tt=torch.tensor(C[:3,3],dtype=torch.float32,device='cuda');v=V@Rt.T+tt;x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];return (rast[...,3]>0).reshape(H,W).cpu().numpy()
def support(R,t):return np.percentile((verts@R.T+t)@n,.1)
def optimize(name,idxs,support_target): …[+1245 chars]
```
<details><summary>output</summary>

```
init start 0.5373014176967037 support target -0.5296360619883147
Traceback (most recent call last):
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_minimize.py", line 1114, in _validate_bounds
    bounds.lb = np.broadcast_to(bounds.lb, x0.shape)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/stride_tricks.py", line 413, in broadcast_to
    return _broadcast_to(array, shape, subok=subok, readonly=True)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/stride_tricks.py", line 349, in _broadcast_to
    it = np.nditer(
         ^^^^^^^^^^
ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (5,)  an …[+1107 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/n0=np.zeros(6)/n0=np.zeros(5)/' /work/scratch/opt_static_worldK_support.py
python /work/scratch/opt_static_worldK_support.py"
```
<details><summary>output</summary>

```
init start 0.5373014176967037 support target -0.5296360619883147
init iou 0.663505963128807 x [-0.01253591  0.19105301  0.1733726   0.01370703  0.19431808] t [0.34899847 0.07878725 2.82929668] support -0.5296360619883147 R [[-0.16208993 -0.52829685  0.83344424]
 [ 0.98677461 -0.08536869  0.13779712]
 [-0.00164774  0.84475714  0.53514732]]
final start 0.4299998364080184 support target -0.5296360619883147
final iou 0.6087685238897202 x [-0.12243177 -0.06229283  0.10882201 -0.01219436  0.19771426] t [0.67603731 0.0851852  2.70487469] support -0.5296360619883147 R [[-0.07227117  0.72758772 -0.68219717]
 [ 0.99402516 -0.00355127 -0.10909341]
 [-0.08179769 -0.68600546 -0.72298385]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_static_worldK_rotonly.py <<'PY'
exec(open('/work/scratch/opt_static_worldK.py').read().split(\"def optimize\")[0])
from scipy.optimize import minimize
def optimize(name,idxs):
 R0=Rot.from_matrix(P[idxs,:3,:3]).mean().as_matrix();t=np.median(P[idxs,:3,3],0)
 def fun(x):
  R=Rot.from_rotvec(x).as_matrix()@R0;mk=render(R,t);v=[]
  for i in idxs:
   vis=mk&~pm[i];v.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
  return 1-np.mean(v)
 print(name,'start',1-fun(np.zeros(3)))
 res=minimize(fun,np.zeros(3),method='Powell',bounds=[(-.25,.25)]*3,options={'xtol':1e-5,'ftol':1e-6,'maxiter':10});R=Rot.from_rotvec(res.x).as_matrix()@R0;print(name,'iou',1-res.fun,'x',res.x,'t',t,'R',R);np.savez('/work/scratch/static_worldKrot_'+name+'.npz',R=R,t=t,iou=1-res.fun)
optimize('init',list(range(0,191,5)));optimize('final',list(range(400,546,5)))
PY
python /work/scratch/opt_static_worldK_rotonly.py"
```
<details><summary>output</summary>

```
init start 0.4942510221727101
init iou 0.5155723581803675 x [-0.197683    0.13979017  0.13959592] t [0.21062662 0.07834553 2.65021527] R [[-0.13994653 -0.62650751  0.76674853]
 [ 0.98203023  0.01119993  0.18839104]
 [-0.12661593  0.77933491  0.61368192]]
final start 0.3638380218899758
final iou 0.4649587831503562 x [-0.21723966  0.24748848  0.12341639] t [0.57096915 0.0998931  2.54596009] R [[-0.09850233  0.53668413 -0.83801398]
 [ 0.98997868 -0.03278223 -0.13735917]
 [-0.10119045 -0.84314617 -0.52807672]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
a=np.load('/work/scratch/static_worldKrot_init.npz');b=np.load('/work/scratch/static_worldKrot_final.npz');q=R.from_matrix(b['R']@a['R'].T);print('rot-only rel',np.degrees(q.magnitude()),q.as_rotvec())
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];q=R.from_matrix(T[500,:3,:3]@T[0,:3,:3].T);print('candidate rel',np.degrees(q.magnitude()),q.as_rotvec())
PY"
```
<details><summary>output</summary>

```
rot-only rel 173.56647125536324 [ 0.36348912 -2.98795729  0.3415906 ]
candidate rel 176.51262469867905 [-0.14909426 -3.07693425 -0.03349846]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_rot_support.py <<'PY'
exec(open('/work/scratch/eval_all_camera.py').read().split(\"r=np.load\")[0])
import json,trimesh
from scipy.spatial.transform import Rotation as Rot
Tbase=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);g=json.load(open('/work/inputs/ground_plane.json'));n=np.array(g['plane'][:3]);n/=np.linalg.norm(n);mesh=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();vv=mesh.vertices
H,W=1152,1536;K=np.array(cam['K']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2]
def iou_world(P):
 vals=[]
 for i in range(546):
  M=Tcw@P[i];v=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];vals.append((vis&om[i]).sum()/(vis|om[i]).sum())
 return np.array(vals)
target=np.percentile((vv@Tbase[0,:3,:3].T+Tbase[0,:3,3])@n,.1)
for name in ['init','final']:
 a=np.load('/work/scratch/static_worldKrot_'+name+'.npz');C=np.block([[a['R'],a['t'][:,None]],[np.zeros((1,3)),np.ones((1,1))]]);Wp=Twc@C;s=np.percentile((vv@Wp[:3,:3].T+Wp[:3,3])@n,.1);Wp[:3,3]+=(target-s)*n
 # evaluate only phase
 b=0 if name=='init' else 400;e=191 if name=='init' else 546;P=n …[+191 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_rot_support.py", line 15, in <module>
    b=0 if name=='init' else 400;e=191 if name=='init' else 546;P=np.repeat(Tbase[0:1],e-b,axis=0);P[:]=Wp;print(name,'iou',iou_world(P).mean(),'support',np.percentile((vv@Wp[:3,:3].T+Wp[:3,3])@n,.1),'t',Wp[:3,3])
                                                                                                                            ^^^^^^^^^^^^
  File "/work/scratch/eval_rot_support.py", line 9, in iou_world
    M=Tcw@P[i];v=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip …[+202 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_rot_support.py';s=open(p).read();s=s.replace(\"def iou_world(P):\\n vals=[]\\n for i in range(546):\", \"def iou_world(P,inds):\\n vals=[]\\n for i in inds:\").replace(\"b=0 if name=='init' else 400;e=191 if name=='init' else 546;P=np.repeat(Tbase[0:1],e-b,axis=0);P[:]=Wp;print(name,'iou',iou_world(P).mean()\", \"inds=list(range(0,191,5)) if name=='init' else list(range(400,546,5));P=np.repeat(Tbase[0:1],len(inds),axis=0);P[:]=Wp;print(name,'iou',iou_world(P,inds).mean()\")
open(p,'w').write(s)
PY
python /work/scratch/eval_rot_support.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/eval_rot_support.py", line 15, in <module>
    inds=list(range(0,191,5)) if name=='init' else list(range(400,546,5));P=np.repeat(Tbase[0:1],len(inds),axis=0);P[:]=Wp;print(name,'iou',iou_world(P,inds).mean(),'support',np.percentile((vv@Wp[:3,:3].T+Wp[:3,3])@n,.1),'t',Wp[:3,3])
                                                                                                                                            ^^^^^^^^^^^^^^^^^
  File "/work/scratch/eval_rot_support.py", line 9, in iou_world
    M=Tcw@P[i];v=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like …[+242 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_rot_support.py';s=open(p).read().replace(\"for i in inds:\\n  M=Tcw@P[i]\",\"for j,i in enumerate(inds):\\n  M=Tcw@P[j]\");open(p,'w').write(s)
PY
python /work/scratch/eval_rot_support.py"
```
<details><summary>output</summary>

```
init iou 0.6126596126060507 support -0.5296360619883147 t [0.3339537  0.08762605 2.63586171]
final iou 0.5449297259598165 support -0.5296360619883147 t [0.68669211 0.08894823 2.50694358]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_rot_candidate.py <<'PY'
exec(open('/work/scratch/eval_all_camera.py').read().split(\"r=np.load\")[0])
import json
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float);Tcw=np.linalg.inv(np.array(cam['T_world_cam']));K=np.array(cam['K']);fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];H,W=1152,1536
from scipy.spatial.transform import Rotation as Rot
def score(R,t,i):
 v=V@torch.tensor(R,dtype=torch.float32,device='cuda').T+torch.tensor(t,dtype=torch.float32,device='cuda');x,y,z=v.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];return (vis&om[i]).sum()/max(1,(vis|om[i]).sum())
def opt(i,R,t):
 best=score(R,t,i)
 for step in [.12,.06,.03,.015,.0075,.00375]:
  for j in range(3):
   for it in range(3):
    moved=False
    for sg in [1,-1]:
     rr=Rot.from_rotvec(np.eye(3)[j]*sg*step).as_matrix()@R;s=score(rr,t,i)
     if s>best+1e-5:best=s;R=rr;moved=True;break
    if not moved:break
 return best,R
for i in [190,220,260,300,320,330,340,344,345,350,355,360,365,370,375,380,390,400,450,545]:
 C=Tcw@T[i];b,R=opt(i,C[:3,:3],C[:3,3]);print(i,'base',round(score(C[:3,:3],C[:3,3],i),3),'best',round(b,3),'delta_deg',round(np.degrees(np.linalg.norm((Rot.from_matrix(R)*Rot.from_matrix(C[:3,:3]).inv()).as_rotvec())),1),flush=True)
PY
python /work/scratch/opt_rot_candid …[+7 chars]
```
<details><summary>output</summary>

```
190 base 0.475 best 0.509 delta_deg 16.9
220 base 0.503 best 0.531 delta_deg 13.2
260 base 0.533 best 0.543 delta_deg 15.1
300 base 0.519 best 0.543 delta_deg 36.4
320 base 0.502 best 0.536 delta_deg 14.3
330 base 0.522 best 0.655 delta_deg 22.1
340 base 0.45 best 0.594 delta_deg 18.5
344 base 0.432 best 0.487 delta_deg 14.3
345 base 0.263 best 0.487 delta_deg 40.9
350 base 0.429 best 0.442 delta_deg 3.3
355 base 0.46 best 0.467 delta_deg 2.4
360 base 0.468 best 0.494 delta_deg 7.1
365 base 0.422 best 0.527 delta_deg 16.0
370 base 0.395 best 0.536 delta_deg 27.7
375 base 0.401 best 0.514 delta_deg 18.1
380 base 0.382 best 0.546 delta_deg 29.6
390 base 0.421 best 0.552 delta_deg 29.3
400 base 0.452 best 0.544 delta_deg 18.1
450 base 0.432 best 0.523 delta_deg 16.3
545 base 0.409 best 0.444  …[+15 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_rot_all.py <<'PY'
exec(open('/work/scratch/opt_rot_candidate.py').read().split('for i in [190')[0])
Rs=np.zeros((546,3,3));scores=np.zeros(546)
for i in range(191,400):
 C=Tcw@T[i];b,R=opt(i,C[:3,:3],C[:3,3]);Rs[i]=R;scores[i]=b
 print(i,round(score(C[:3,:3],C[:3,3],i),3),round(b,3),flush=True)
np.savez('/work/scratch/rotopt_motion.npz',R=Rs,score=scores)
PY
python /work/scratch/opt_rot_all.py > /work/scratch/rotopt.log"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];Tcw=np.linalg.inv(np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']));op=np.load('/work/scratch/rotopt_motion.npz');R=op['R'];sc=op['score'];
for i in range(190,401,5):
 C=Tcw@T[i];Cb=Tcw@T[0];q=Rot.from_matrix(R[i]@Cb[:3,:3].T);rv=q.as_rotvec();dq=Rot.from_matrix(R[i]@C[:3,:3].T);print(i,'yaw',round(np.degrees(np.linalg.norm(rv)),1),'delta current',round(np.degrees(dq.magnitude()),1),'score',round(float(sc[i]),3))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
  File "scipy/spatial/transform/_rotation.pyx", line 1143, in scipy.spatial.transform._rotation.Rotation.from_matrix
ValueError: Non-positive determinant (left-handed or null coordinate frame) in rotation matrix 0: [[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]].

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];Tcw=np.linalg.inv(np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']));op=np.load('/work/scratch/rotopt_motion.npz');R=np.array([(Tcw@t)[:3,:3] for t in T]);R[191:400]=op['R'][191:400];sc=op['score'];Cb=(Tcw@T[0])[:3,:3]
for i in range(190,401,5):
 q=Rot.from_matrix(R[i]@Cb.T);rv=q.as_rotvec();dq=Rot.from_matrix(R[i]@(Tcw@T[i])[:3,:3].T);print(i,'yawang',round(np.degrees(np.linalg.norm(rv)),1),'rv',np.round(np.degrees(rv),1),'delta',round(np.degrees(dq.magnitude()),1),'score',round(float(sc[i]),3))
PY"
```
<details><summary>output</summary>

```
190 yawang 0.0 rv [0. 0. 0.] delta 0.0 score 0.0
195 yawang 9.3 rv [-5.4  6.9  3.2] delta 9.3 score 0.513
200 yawang 11.9 rv [ 2.3 11.4 -2.7] delta 11.6 score 0.509
205 yawang 7.9 rv [0.2 7.8 1.6] delta 7.0 score 0.513
210 yawang 15.3 rv [-12.3   7.8   4.7] delta 16.0 score 0.519
215 yawang 10.4 rv [-3.2  9.7  1.7] delta 8.9 score 0.517
220 yawang 15.9 rv [-9.4 12.   4.6] delta 13.2 score 0.531
225 yawang 19.7 rv [-5.5 18.9  1.5] delta 9.7 score 0.528
230 yawang 26.8 rv [-4.  26.5 -0.7] delta 10.5 score 0.527
235 yawang 61.9 rv [-5.7 61.3 -6.1] delta 37.2 score 0.54
240 yawang 55.4 rv [-2.3 55.4 -1.6] delta 21.5 score 0.548
245 yawang 53.5 rv [-2.6 53.3 -3.2] delta 15.3 score 0.535
250 yawang 55.6 rv [-3.1 54.8 -9. ] delta 17.1 score 0.533
255 yawang 53.7 rv [-2.2 53.4 -4.9] delta 12.7 sco …[+1817 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n)
s=[]
for i in range(0,191,5):s.append(np.percentile((v@T[i,:3,:3].T+T[i,:3,3])@n,.1))
print(np.percentile(s,[0,25,50,75,100]),np.mean(s))
PY"
```
<details><summary>output</summary>

```
[-0.55636123 -0.54706965 -0.5440496  -0.54090137 -0.5158202 ] -0.5422948127426247

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_candidate_staticrot.py <<'PY'
import numpy as np,trimesh,json
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'].astype(float).copy();Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tbase=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float);m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;n=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n)
s=[]; 
for i in range(0,191,5):s.append(np.percentile((v@Tbase[i,:3,:3].T+Tbase[i,:3,3])@n,.1))
target=np.median(s);print('target',target)
poses=[]
for name in ['init','final']:
 a=np.load('/work/scratch/static_worldKrot_'+name+'.npz');C=np.block([[a['R'],a['t'][:,None]],[np.zeros((1,3)),np.ones((1,1))]]);W=Twc@C;sup=np.percentile((v@W[:3,:3].T+W[:3,3])@n,.1);W[:3,3]+=(target-sup)*n;poses.append(W);print(name,W[:3,3],np.percentile((v@W[:3,:3].T+W[:3,3])@n,.1))
Wi,Wf=poses
P=T.copy();P[:191]=Wi;P[400:]=Wf
for i in range(191,211):
 u=(i-190)/20;s=u*u*(3-2*u)
 q=Rot.from_matrix(P[i,:3,:3])*Rot.from_matrix(Wi[:3,:3]).inv();P[i,:3,:3]=(Rot.from_rotvec(s*q.as_rotvec())*Rot.from_matrix(Wi[:3,:3])).as_matrix();P[i,:3,3]=(1-s)*Wi[:3,3]+s*P[i,:3,3]
for i in range(380,400):
 u=(i-380)/20;s=u*u*(3-2*u)
 q=Rot.from_matrix(Wf[:3,:3])*Rot.from_matrix(P[i,:3,:3]).inv();P[i,:3,:3]=(Rot.from_rotvec(s*q.as_rotv …[+386 chars]
```
<details><summary>output</summary>

```
target -0.5440495991219778
init [0.33423443 0.10201403 2.63667245] -0.5440495991219777
final [0.68697283 0.10333621 2.50775431] -0.5440495991219778
K base 0.4534824146079515 0.48931314373021384 cand 0.5279197555493468 0.5422690356985692
0 0.56 0.577
20 0.514 0.577
40 0.508 0.582
60 0.498 0.579
80 0.457 0.552
100 0.488 0.566
120 0.492 0.565
140 0.502 0.569
160 0.49 0.568
180 0.485 0.555
200 0.493 0.501
220 0.51 0.51
240 0.524 0.524
260 0.525 0.525
280 0.517 0.518
300 0.504 0.504
320 0.494 0.497
340 0.375 0.42
360 0.342 0.413
380 0.386 0.49
400 0.394 0.56
420 0.407 0.591
440 0.358 0.557
460 0.353 0.537
480 0.354 0.499
500 0.368 0.467
520 0.371 0.468
540 0.367 0.464
K_pipeline base 0.5266313302489372 0.5683275775764658 cand 0.5927836160212985 0.619880526077369
0 0.632 0.636
20 0.632 0.636
40  …[+405 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
P=np.load('/work/scratch/candidate_staticrot.npz')['T_world_mesh'];q=Rot.from_matrix(P[500,:3,:3]@P[0,:3,:3].T);print('rel',np.degrees(q.magnitude()),q.as_rotvec());print('t init/final',P[0,:3,3],P[500,:3,3]);print('+y init',P[0,:3,:3]@np.array([0,1.,0]),'final',P[500,:3,:3]@np.array([0,1.,0]))
PY"
```
<details><summary>output</summary>

```
rel 173.56647125536324 [ 0.37315608 -2.96297563  0.50816061]
t init/final [0.33423443 0.10201403 2.63667245] [0.68697283 0.10333621 2.50775431]
+y init [-0.57487289  0.06578371  0.81559405] final [ 0.4810365  -0.09066368 -0.87199999]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
# Insert K rotation optimizers before build_camera_trajectory
needle='''def build_camera_trajectory(rec, camera, static_init, static_final):
'''
insert='''def optimize_static_rotation_world_K(renderer, rec, camera, ground, mesh, indices, support_target):
    \"\"\"Refine only rotation under the released camera K, then place the pose on the table.

    CoCoNet and FoundationPose use K_pipeline, but the released evaluation frame
    uses camera.json K.  Optimising all six DoF under K can trade a good mask fit
    for an object hovering above the table.  Here translation is held at the
    pooled baseline value during silhouette optimisation, and only the component
    normal to the table is adjusted afterwards to preserve table support.
    \"\"\"
    T = rec[\"T_world_mesh\"].astype(np.float64)
    T_world_cam = np.asarray(camera[\"T_world_cam\"], dtype=np.float64)
    T_cam_world = np.linalg.inv(T_world_cam)
    cam_pose = np.array([T_cam_world @ T[i] for i in range(len(T))])
    R0 = robust_mean_rotation(cam_pose[indices, :3, :3]).as_matrix()
    t0 = np.median(cam_pose[indices, :3, 3], axis=0)

    def objective(x):
        R = Rotation.from_rotvec(x).as_matrix() @ R0
        return 1.0 - mean_visible_iou(renderer, R, t0, indices, object_masks, person_masks)

    result = minimize(
        objective,
        np.zeros(3, dtype=np.float64),
        method=\"Powell\",
        bounds=[(-0.25, 0.25)] * …[+3819 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'start=$(date +%s); python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz; rc=$?; end=$(date +%s); echo ELAPSED=$((end-start))s; exit $rc'
```
<details><summary>output</summary>

```
static initial pooled mask IoU: 0.8771
static final pooled mask IoU:   0.7134
K-space static initial rotation IoU: 0.5156
K-space static final rotation IoU:   0.4650
ELAPSED=216s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];q=np.load('/work/scratch/candidate_staticrot.npz')['T_world_mesh'];print(p.shape,p.dtype,'maxdiff',np.abs(p-q).max(),'mean',np.abs(p-q).mean());
for i in [0,100,190,191,200,300,340,360,380,399,400,500,545]:print(i,np.max(np.abs(p[i]-q[i])),p[i,:3,3])
PY"
```
<details><summary>output</summary>

```
(546, 4, 4) float64 maxdiff 0.0 mean 0.0
0 0.0 [0.33423443 0.10201403 2.63667245]
100 0.0 [0.33423443 0.10201403 2.63667245]
190 0.0 [0.33423443 0.10201403 2.63667245]
191 0.0 [0.33434492 0.10215695 2.63659783]
200 0.0 [0.34838829 0.11078881 2.62738841]
300 0.0 [0.5254875  0.09996682 2.6310934 ]
340 0.0 [0.6613607  0.09840877 2.59344552]
360 0.0 [0.67505222 0.09440655 2.54238858]
380 0.0 [0.71680361 0.11143062 2.4829967 ]
399 0.0 [0.68698401 0.103451   2.50777159]
400 0.0 [0.68697283 0.10333621 2.50775431]
500 0.0 [0.68697283 0.10333621 2.50775431]
545 0.0 [0.68697283 0.10333621 2.50775431]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py'
```
<details><summary>output</summary>

```
a6d3ca81d75051ca484dfdceb58627907c915398e55e0ca408ac860a94c86efc  /work/outputs/object_poses_world.npz
bcc2e8f24f600037109ba6925ddc69deddfb58bf423c0d0c6564ce5bf2ea3e76  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/outputs/evaluate.py'
```
<details><summary>output</summary>

```
{
  "phase_iou_K": {
    "P0 untouched/static": [
      0.5000899001771729,
      0.5703852330045199
    ],
    "P1 approach+turn": [
      0.5082481076804987,
      0.5132109152626229
    ],
    "P2 edge-on flip": [
      0.35844345723453164,
      0.4162570129900285
    ],
    "P3 settle": [
      0.3863968693410472,
      0.49002283581977046
    ],
    "P4 resting/static": [
      0.37184646384592496,
      0.523160571342142
    ]
  },
  "phase_iou_Kpipeline": {
    "P0 untouched/static": [
      0.5895497685309555,
      0.6291306498325324
    ],
    "P1 approach+turn": [
      0.6046379651237529,
      0.6136136647942071
    ],
    "P2 edge-on flip": [
      0.35094130494888437,
      0.4112839996076906
    ],
    "P3 settle": [
      0.44676074475311145,
      0.5662564057039756
     …[+1467 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/REPORT.md';s=open(p).read()
s=s.replace('The corrected track keeps one rigid object, turns monotonically about the table-normal axis, pools the two static phases into single poses, removes tracker translation detours, and preserves the baseline\\'s human/floor-calibrated world depth.','The corrected track keeps one rigid object, turns monotonically about the table-normal axis, pools the two static phases into single poses, removes tracker translation detours, preserves the baseline\\'s human/floor-calibrated support height, and refines static rotation under the released camera "'`K` without sacrificing table contact.'"')
s=s.replace('4. **World-depth preservation.** Camera rotations are transferred to the released world frame through the nearly constant rotation "'`R_world_mesh @ R_cam_mesh''^-1`. World translations retain the baseline'"\\'s human/floor depth calibration rather than replacing depth with the less reliable object-silhouette scale.\\n5. **Same-table support constraint.** Both static phases are on the same table. The final pooled pose is translated along the released ground-plane normal so its lowest support percentile matches the untouched initial pose. This removes a 14.4 mm support-height disagreement without snapping the object to the floor.','4. **World-depth and support preservation.** Camera rotations are transferred to the released world frame through the nearly constant rotation "'`R_world_mesh @ R_cam_m …[+4632 chars]
```
<details><summary>output</summary>

```
48:| P0 static | 0-190 | 0.5001 -> **0.5704** | 0.5895 -> **0.6291** |
53:| Overall | 0-545 | 0.4535 -> **0.5279** | 0.5266 -> **0.5928** |
84:## Reproducibility and runtime
92:The script is deterministic, seeds NumPy/PyTorch, performs no network access, and reads only the supplied inputs. Final measured runtime in this container was 216 seconds on one GPU. It uses nvdiffrast only for silhouette scoring and does not reload any external model weights. Output file SHA-256 at submission time:
95:a6d3ca81d75051ca484dfdceb58627907c915398e55e0ca408ac860a94c86efc  object_poses_world.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/The final run took 137 seconds and used one GPU/The final run took 216 seconds and used one GPU/' /work/outputs/TOOLS.md
# update solve docstring to mention K-space static support refinement
python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace(\"(2) repairs the long, physically impossible yaw reversal during the\\nfast edge-on turn by propagating yaw about the table normal, and (3) smooths or\\nreplaces tracker translation detours while retaining the baseline's human/floor\\ncalibrated world depth.\",\"(2) repairs the long, physically impossible yaw reversal during the\\nfast edge-on turn by propagating yaw about the table normal, and (3) smooths or\\nreplaces tracker translation detours while retaining the baseline's human/floor\\ncalibrated world depth.  Static rotations are finally refined under camera K and\\nplaced on the shared table-support plane.\")
open(p,'w').write(s)
PY
python -m py_compile /work/outputs/solve.py
sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py"
```
<details><summary>output</summary>

```
a6d3ca81d75051ca484dfdceb58627907c915398e55e0ca408ac860a94c86efc  /work/outputs/object_poses_world.npz
e9ce982d7807f68034dc735b7eb86d346a63f17a17bd5d6bf82d638bf584b73c  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/outputs/evaluate.py > /work/outputs/validation.json
ls -lah /work/outputs
python - <<'PY'
import json,numpy as np
x=json.load(open('/work/outputs/validation.json'));print('overall K',x['overall_iou_K']);print('overall Kp',x['overall_iou_Kpipeline']);print('contact',x['contact']);print('support',x['support'])
p=np.load('/work/outputs/object_poses_world.npz');T=p['T_world_mesh'];assert T.shape==(546,4,4) and T.dtype==np.float64 and np.isfinite(T).all();print('output OK')
PY"
```
<details><summary>output</summary>

```
total 124K
drwxrwxr-x 3 hanzhou hanzhou 4.0K Oct 20 21:20 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 20:05 ..
-rw-rw-r-- 1 hanzhou hanzhou 8.7K Oct 20 21:20 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 3.1K Oct 20 21:20 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:20 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou 4.0K Oct 20 21:00 evaluate.py
-rw-rw-r-- 1 hanzhou hanzhou  69K Oct 20 21:20 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  16K Oct 20 21:20 solve.py
-rw-rw-r-- 1 hanzhou hanzhou 2.3K Oct 20 21:21 validation.json
overall K [0.4534824146079515, 0.5279197555493468]
overall Kp [0.5266313302489372, 0.5927836160212985]
contact {'baseline': {'median_min_hand_distance_m': 0.0027771020222580093, 'p90_m': 0.0404563151734771, 'fraction_under_5cm': 0.9436619718309859}, 'result': {'median_m …[+359 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import shutil
shutil.rmtree('/work/outputs/__pycache__', ignore_errors=True)
PY
ls -lah /work/outputs"
```
<details><summary>output</summary>

```
total 120K
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:21 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 20:05 ..
-rw-rw-r-- 1 hanzhou hanzhou 8.7K Oct 20 21:20 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 3.1K Oct 20 21:20 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou 4.0K Oct 20 21:00 evaluate.py
-rw-rw-r-- 1 hanzhou hanzhou  69K Oct 20 21:20 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  16K Oct 20 21:20 solve.py
-rw-rw-r-- 1 hanzhou hanzhou 2.3K Oct 20 21:21 validation.json

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -nE '/work/scratch|/pipeline|http|requests|urllib' /work/outputs/solve.py || true
python - <<'PY'
import numpy as np
p=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
print(p.shape,p.dtype,np.isfinite(p).all(),p[:,3,:].min(),p[:,3,:].max())
PY"
```
<details><summary>output</summary>

```

```
</details>

**Agent:** Completed the trajectory recovery task.

Outputs created:

- `/work/outputs/object_poses_world.npz` — `T_world_mesh`, shape `(546, 4, 4)`, `float64`
- `/work/outputs/REPORT.md` — phases, method, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — tools, versions, revisions, provenance
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/validation.json` — evaluation metrics
- `/work/outputs/evaluate.py` — validation script

Key validation improvements over baseline:

- Overall visible-mask IoU with released `K`: `0.4535 -> 0.5279`
- Overall visible-mask IoU with `K_pipeline`: `0.5266 -> 0.5928`
- Hand-object contact under 5 cm: `94.4% -> 100%`
- P95 one-frame rotation change: `12.44° -> 2.07°`
- Maximum one-frame rotation change: `118.41° -> 5.25°`
- Static-phase pose drift removed
- Initial/final table-support height aligned

Final solver runtime was 216 seconds on one GPU.
