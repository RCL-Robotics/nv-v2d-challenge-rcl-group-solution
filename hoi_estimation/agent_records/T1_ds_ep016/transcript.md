# T1_ds_ep016

Codex DeepSeek 完整轨迹（Tier 1 输入）  
模型: deepseek-flash api=api.deepseek.com:443 effort=high net=open  
日期: 2026-10-20T17:28:20Z

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
/usr/bin/bash -c "nl -ba /work/AGENT_TASK.md | sed -n '45,95p'"
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
    53	temporal smoothness, physical plausibility such as no floating or penetrating the floor/human), wit …[+3943 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work/inputs -maxdepth 2 -type f -printf '%p %s\\n' | sort"
```
<details><summary>output</summary>

```
/work/inputs/camera.json 824
/work/inputs/episode.json 67
/work/inputs/frame0_reference_human.npz 190402
/work/inputs/ground_plane.json 397
/work/inputs/masks.h5 4569555
/work/inputs/object_mesh.glb 8634552
/work/inputs/reconstruction.npz 23984227
/work/inputs/task.txt 117
/work/inputs/video.mp4 12977144

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1911
/pipeline/.stages/02_export.json 2150
/pipeline/.stages/03_human_initialization.json 2295
/pipeline/.stages/04_depth_alignment.json 2057
/pipeline/.stages/05_object_pose.json 1898
/pipeline/.stages/06_coconet.json 3080
/pipeline/.stages/07_refinement.json 2298
/pipeline/.stages/08_render.json 2819
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 508264620
/pipeline/depth/moge2/depth_report.json 1879
/pipeline/depth/moge2/intrinsics.pkl 50960
/pipeline/depth/moge2/raw.npy 1688076416
/pipeline/export/episode_000016/edex 563
/pipeline/export/episode_000016/wild_export.json 2879
/pipeline/foundationpose/episode_000016.pkl 139259
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 3650187 …[+480 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "nvidia-smi; python - <<'PY'
import torch, cv2, trimesh, h5py, numpy as np
print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print('cv2',cv2.__version__,'trimesh',trimesh.__version__,'numpy',np.__version__)
PY"
```
<details><summary>output</summary>

```
torch 2.5.1+cu124 True NVIDIA RTX 6000 Ada Generation
cv2 4.11.0 trimesh 5.1.1 numpy 1.26.3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /work/inputs/task.txt; echo; cat /work/inputs/episode.json; echo; cat /work/inputs/camera.json; echo; cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
Walk to the spray bottle on the white desk, pick up the spray bottle with your left hand and place it on the ground.

{
 "episode_index": 16,
 "object": "spray_bottle",
 "frames": 477
}
{
 "K": [
  [
   820.0003662109375,
   0.0,
   758.0999755859375
  ],
  [
   0.0,
   820.0003662109375,
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
   819.3696899414062,
   0.0,
   768.0
  ],
  [
   0.0,
   819.3696899414062,
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
   -0.9988628505074858,
   -0.00045903805249741716,
   -0.04767384146601045,
   0.19096887694024275
  ],
  [
   -0.00573470773997912,
   0.9938502190458774,
   0.11058415451409849,
   -0.3128998428047386
  ],
  [
   0.0473298954488277,
   0.11073179934656237,
   -0.9927226952227272,
   6.12 …[+608 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,json
p='/work/inputs/reconstruction.npz'
z=np.load(p)
for k in z.files: print(k,z[k].shape,z[k].dtype)
print('masks')
with h5py.File('/work/inputs/masks.h5') as f:
 def v(n,x): print(n,x.shape,x.dtype)
 f.visititems(v)
PY"
```
<details><summary>output</summary>

```
T_world_mesh (477, 4, 4) float32
T_cam_mesh_foundationpose (477, 4, 4) float32
T_cam_mesh_coconet (477, 4, 4) float32
human_vertices_world (477, 4610, 3) float32
human_keypoints_world (477, 70, 3) float32
contact_prob_right_left (477, 2) float32
masks
object (477, 1152, 1536) bool
person (477, 1152, 1536) bool

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/pipeline_report.json; echo; cat /pipeline/depth/moge2/depth_report.json'
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
    "mtime_ns": 1792069255611117124,
    "path": "/mnt/nas/Group/hanzhou/v2d/experiments/M1_prompt_v2/out/episode_000016/inference/episode_000016_step200000_before_after.mp4",
    "size": 32638922
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.0 …[+7418 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ffprobe -v error -show_entries stream=width,height,r_frame_rate,nb_frames,duration -of json /work/inputs/video.mp4; du -sh /work/*'
```
<details><summary>output</summary>

```
{
    "programs": [

    ],
    "streams": [
        {
            "width": 1536,
            "height": 1152,
            "r_frame_rate": "30/1",
            "duration": "15.900000",
            "nb_frames": "477"
        }
    ]
}
12K	/work/AGENT_TASK.md
49M	/work/inputs
60K	/work/logs
4.0K	/work/outputs
12K	/work/scratch

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, h5py, json, trimesh
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh']; K=np.array(json.load(open('/work/inputs/camera.json'))['K']); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tcw=np.linalg.inv(Twc)
cent=T[:,:3,3]; rot=T[:,:3,:3]
rel=np.einsum('nji,njk->nik',rot[:-1],rot[1:])
rvec=R.from_matrix(rel).as_rotvec(); ang=np.linalg.norm(rvec,axis=1)*180/np.pi
tr=np.linalg.norm(np.diff(cent,axis=0),axis=1)
# cam coords
cc=np.einsum('ij,nj->ni',Tcw[:3,:3],cent)+Tcw[:3,3]
uv=(np.einsum('ij,nj->ni',K,cc)); uv=uv[:,:2]/uv[:,2,None]
cp=z['contact_prob_right_left']
# masks areas and centroids
with h5py.File('/work/inputs/masks.h5') as f:
 obj=f['object'][:]; per=f['person'][:]
area=obj.sum((1,2)); pa=per.sum((1,2)); inter=(obj&per).sum((1,2))
# object mask centroids
ys,xs=np.meshgrid(np.arange(1152),np.arange(1536),indexing='ij')
cent_obj=np.stack([(obj*xs).sum((1,2))/np.maximum(area,1),(obj*ys).sum((1,2))/np.maximum(area,1)],1)
print('centers',cent[0],cent[-1]); print('cam z range',cc[:,2].min(),cc[:,2].max())
print('uv range',uv.min(0),uv.max(0)); print('area object',area.min(),np.median(area),area.max()); print('person area',pa.min(),np.median(pa),pa.max()); print('obj/person overlap frac',np.percentile(inter/np.maximum(area,1),[0,50,90,100]))
print('rot step deg percentiles',np.percentile(ang,[0,50,90,95,99,100]))
print('trans step m perc …[+428 chars]
```
<details><summary>output</summary>

```
centers [0.16693087 0.19702162 3.1192625 ] [0.10238931 1.1328211  3.0923452 ]
cam z range 1.7872483264589736 3.247447798308045
uv range [487.76689677 593.20259481] [1041.7145858   847.29449213]
area object 0 1957.0 4308
person area 13518 37542.0 71336
obj/person overlap frac [0.         0.         0.01309682 0.26373626]
rot step deg percentiles [  0.17839811   2.41820519   9.11531872  21.59114694 163.84673821
 175.78038997]
trans step m percentiles [7.54371518e-04 1.78590901e-02 3.83328125e-02 5.25148287e-02
 1.27825089e-01 1.43371534e+00]
top rot jumps
332 333 175.8 0.054
368 369 174.3 0.024
336 337 170.9 0.028
326 327 170.7 0.014
338 339 165.8 0.144
325 326 163.2 0.031
335 336 145.3 0.093
344 345 129.4 0.028
327 328 89.1 0.027
268 269 83.5 0.025
328 329 80.8 0.04
329 330 74.3 0.018
323 3 …[+2901 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5','/pipeline/mhr_init/episode_000016.sam3d.h5']:
 print('\\n',p)
 with h5py.File(p) as f:
  def v(n,x): print(n,x.shape,x.dtype, dict(x.attrs))
  f.visititems(v)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 771, in visititems
    return h5o.visit(self.id, proxy)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5o.pyx", line 420, in h5py.h5o.visit
  File "h5py/h5o.pyx", line 367, in h5py.h5o.cb_obj_simple
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 770, in proxy
    return func(name, self[name])
           ^^^^^^^^^^^^^^^^^^^^^^
  File "<stdin>", line 5, in v
AttributeError: 'Group' object has no attribute 'shape'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 3 -type f | head -100; grep -R \"T_world_mesh\\|T_cam_mesh_coconet\\|pose_abs\" -n /code/reconstruction | head -100"
```
<details><summary>output</summary>

```
grep: /code/reconstruction/.venv/bin/python3.12: No such file or directory
grep: /code/reconstruction/.venv/bin/python3: No such file or directory
grep: /code/reconstruction/.venv/bin/python: No such file or directory
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:520:        total = len(bundle["pr"]["pose_abs"])
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:555:        self.pose_init = _as_tensor(self.pr["pose_abs"], device=self.device, dtype=self.dtype)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:556:        self.pose_ref = _as_tensor(self.ref.get("pose_abs", self.pr["pose_abs"]), device=self.device, dtype=self.dtype)
/code/reconstruction/modules/v2d_cari4d …[+2019 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
for p in ['/pipeline/foundationpose/episode_000016.pkl','/pipeline/inference/coconet.pth','/pipeline/inference/refined.pth','/pipeline/inference/refinement_checkpoint.pth','/pipeline/mhr_init/episode_000016.pkl']:
 print('\\n',p)
 try:
  x=torch.load(p,map_location='cpu',weights_only=False)
  print(type(x))
  if isinstance(x,dict):
   print(list(x.keys()))
   for k,v in x.items():
    if hasattr(v,'shape'): print(' ',k,v.shape,v.dtype)
    elif isinstance(v,(dict,list)): print(' ',k,type(v),len(v))
    else: print(' ',k,type(v),repr(v)[:200])
 except Exception as e: print('ERR',e)
PY"
```
<details><summary>output</summary>

```

 /pipeline/foundationpose/episode_000016.pkl
ERR Invalid magic number; corrupt file?

 /pipeline/inference/coconet.pth
<class 'dict'>
['schema', 'frames', 'frame_meta', 'kid', 'checkpoint', 'mhr_inference', 'config', 'mesh_diameter', 'mhr_neutral_height_init', 'mhr_spatial_scale', 'K_rois', 'bboxes', 'faces', 'pr', 'pr_initial', 'in', 'gt', 'observations', 'raw', 'metadata']
  schema <class 'str'> 'cari4d.mhr_wild_inference.v1'
  frames <class 'list'> 477
  frame_meta <class 'list'> 477
  kid <class 'int'> 0
  checkpoint <class 'dict'> 2
  mhr_inference <class 'dict'> 8
  config <class 'dict'> 291
  mesh_diameter (477,) float32
  mhr_neutral_height_init (477,) float32
  mhr_spatial_scale (477,) float32
  K_rois (477, 3, 3) float32
  bboxes (477, 4) float32
  faces (36874, 3) int32
  pr <c …[+1894 chars]
```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c 'file /pipeline/foundationpose/episode_000016.pkl /pipeline/mhr_init/episode_000016.pkl; head -c 200 /pipeline/foundationpose/episode_000016.pkl | xxd; head -c 200 /pipeline/mhr_init/episode_000016.pkl | xxd'
```
<details><summary>output</summary>

```
/pipeline/foundationpose/episode_000016.pkl: data
/pipeline/mhr_init/episode_000016.pkl:       data
/usr/bin/bash: line 1: xxd: command not found
/usr/bin/bash: line 1: xxd: command not found

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5','/pipeline/mhr_init/episode_000016.sam3d.h5']:
 print('\\n',p)
 with h5py.File(p) as f:
  def v(n,x):
   if isinstance(x,h5py.Dataset): print(n,x.shape,x.dtype, dict(x.attrs))
   else: print(n+'/',dict(x.attrs))
  f.visititems(v)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
aligned/ {}
aligned/front_stereo_camera_left (477,) object {}
alignment/ {}
alignment/front_stereo_camera_left/ {}
alignment/front_stereo_camera_left/scale (477,) float32 {}
alignment/front_stereo_camera_left/shift (477,) float32 {}
alignment/front_stereo_camera_left/valid_count (477,) int32 {}
frame_names/ {}
frame_names/front_stereo_camera_left (477,) object {}
raw/ {}
raw/front_stereo_camera_left (477,) object {}

 /pipeline/inference/materialized_inputs.h5
K_rois (477, 3, 3) float32 {}
bboxes (477, 4) float32 {}
identity_json () object {}
input_rgbs (477, 3, 224, 224) float32 {}
input_xyz (477, 6, 224, 224) float32 {}
observed_human_masks (477, 224, 224) bool {}
observed_object_masks (477, 224, 224) bool {}
render_rgbs (477, 3, 224, 224) float32 {}
re …[+449 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"def .*pose\\|pose_abs\\|object_translation\\|T_world\" -n /code/reconstruction/modules/v2d_cari4d/lib/cari4d | grep -E \"wild|export|foundation|run_mhr|postopt\" | head -200"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:804:        result["pr"]["pose_abs_postopt"] = _to_numpy(pose).astype(np.float32)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:1261:        result["pr"]["pose_abs_postopt"] = _to_numpy(pose).astype(np.float32)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/mhr_opt_refineout.py:1438:        cfg = MHRParityPostOptConfig(num_steps=args.num_steps, batch_size=args.batch_size, device=args.device, frame_start=args.frame_start, frame_limit=args.frame_limit, lr=args.lr, warmup_steps=args.num_steps // 10, schedule_steps=int(args.num_steps * 1.5), object_surface_samples=1000 if args.object_surface_samples is None else args.object_surface_samples, …[+16115 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'od -An -tx1 -N64 /pipeline/foundationpose/episode_000016.pkl; od -An -tc -N64 /pipeline/foundationpose/episode_000016.pkl; cat /pipeline/export/episode_000016/wild_export.json'
```
<details><summary>output</summary>

```
 80 04 95 ee 59 01 00 00 00 00 00 7d 94 28 8c 06
 66 72 61 6d 65 73 94 5d 94 28 8c 06 30 30 30 30
 30 30 94 8c 06 30 30 30 30 30 31 94 8c 06 30 30
 30 30 30 32 94 8c 06 30 30 30 30 30 33 94 8c 06
 200 004 225 356   Y 001  \0  \0  \0  \0  \0   } 224   ( 214 006
   f   r   a   m   e   s 224   ] 224   ( 214 006   0   0   0   0
   0   0 224 214 006   0   0   0   0   0   1 224 214 006   0   0
   0   0   0   2 224 214 006   0   0   0   0   0   3 224 214 006
{
  "aligned_object_extents_m": [
    0.08719000965356827,
    0.08752718567848206,
    0.2962729036808014
  ],
  "camera_id": 0,
  "depth_backend": "moge2",
  "depth_camera_policy": "first_frame_export_per_frame_depth",
  "depth_model_id": "Ruicheng/moge-2-vitl-normal",
  "depth_model_revision": "b135031bae30b5ac2ae141a0e68717795ce38340",
   …[+2535 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
for p in ['/pipeline/foundationpose/episode_000016.pkl','/pipeline/mhr_init/episode_000016.pkl']:
 print('\\n',p)
 with open(p,'rb') as f: x=pickle.load(f)
 print(type(x))
 if isinstance(x,dict):
  print(list(x.keys()))
  for k,v in x.items():
   if hasattr(v,'shape'): print(' ',k,v.shape,v.dtype)
   elif isinstance(v,(dict,list)): 
    print(' ',k,type(v),len(v))
    if isinstance(v,dict): print('   keys',list(v.keys())[:30])
   else: print(' ',k,type(v),repr(v)[:300])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/frames /work/outputs; ffmpeg -v error -i /work/inputs/video.mp4 -q:v 2 /work/scratch/frames/%04d.jpg; python - <<'PY'
import cv2, numpy as np, os
idxs=list(range(0,477,15))
ims=[]
for i in idxs:
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')
 im=cv2.resize(im,(384,288))
 cv2.putText(im,f'{i:03d}',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),4,cv2.LINE_AA)
 cv2.putText(im,f'{i:03d}',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA)
 ims.append(im)
while len(ims)%8: ims.append(np.zeros_like(ims[0]))
rows=[np.hstack(ims[j:j+8]) for j in range(0,len(ims),8)]
cv2.imwrite('/work/scratch/contact.jpg',np.vstack(rows))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,pickle
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz')
Tw=z['T_world_mesh']; F=z['T_cam_mesh_foundationpose']; C=z['T_cam_mesh_coconet']
def relang(A,B):
 r=np.einsum('nji,njk->nik',A[:,:3,:3],B[:,:3,:3]); return np.linalg.norm(R.from_matrix(r).as_rotvec(),axis=1)*180/np.pi
def steps(T):
 r=np.einsum('nji,njk->nik',T[:-1,:3,:3],T[1:,:3,:3]); return np.linalg.norm(R.from_matrix(r).as_rotvec(),axis=1)*180/np.pi
for n,T in [('world',Tw),('fp_cam',F),('coco_cam',C)]:
 s=steps(T); print(n,'steps percentiles',np.percentile(s,[50,90,95,99,100]),'big>30',np.where(s>30)[0][:100])
print('rel F-C',np.percentile(relang(F,C),[0,50,90,95,99,100]))
print('rel world-F',np.percentile(relang(Tw,F),[0,50,90,95,99,100]))
print('rel world-C',np.percentile(relang(Tw,C),[0,50,90,95,99,100]))
p=pickle.load(open('/pipeline/foundationpose/episode_000016.pkl','rb'))
P=p['fp_poses_cam'][:,0]
print('pipeline fp shape',P.shape,'rel F-P',np.percentile(relang(F,P),[0,50,90,95,99,100]))
print('pipeline world vs Tw rel',np.percentile(relang(p['obj_pose_world'],Tw),[0,50,90,95,99,100]))
print('pipeline world vs P rel',np.percentile(relang(p['obj_pose_world'],P),[0,50,90,95,99,100]))
# print per frame rel angles in problematic stages
for i in list(range(90,130,5))+list(range(250,375,5))+list(range(380,477,10)):
 print(i, 'F-C',relang(F[i:i+1],C[i:i+1])[0], 'W-F',relang(Tw[i:i+1],F[i:i+1])[0], 'W-C',relang(Tw[i:i+ …[+20 chars]
```
<details><summary>output</summary>

```
world steps percentiles [  2.41820519   9.11531872  21.59114694 163.84673821 175.78038997] big>30 [262 263 268 323 325 326 327 328 329 332 335 336 338 339 340 341 342 344
 349 368 370]
fp_cam steps percentiles [  2.92488091  10.27909818  16.32840219 161.88311123 179.32057941] big>30 [323 326 329 332 335 338 341 344 348 349 356 368 370]
coco_cam steps percentiles [  2.41746139   9.11641247  21.59011358 163.84562298 175.77764226] big>30 [262 263 268 323 325 326 327 328 329 332 335 336 338 339 340 341 342 344
 349 368 370]
rel F-C [  0.56813543   5.19003489  18.19170861  48.64276176  93.1313719
 101.18013387]
rel world-F [106.71691485 174.58749541 177.81649348 178.72362391 179.54603525
 179.98117966]
rel world-C [176.58290488 176.6424414  176.66482928 176.66908998 176.67802672
 176.67937045]
 …[+3415 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); T=z['T_world_mesh']; C=z['T_cam_mesh_coconet']; F=z['T_cam_mesh_foundationpose']
def ang(A):
 r=np.einsum('nji,njk->nik',A[:-1,:3,:3],A[1:,:3,:3]); return np.linalg.norm(R.from_matrix(r).as_rotvec(),axis=1)*180/np.pi
def tr(A):return np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)
for a,b in [(0,105),(0,100),(370,477),(375,477),(380,477),(0,120),(120,370)]:
 print('\\n',a,b)
 for n,A in [('world',T),('coco',C),('fp',F)]:
  aa=ang(A)[a:b-1]; tt=tr(A)[a:b-1]
  print(n,'rot',np.round(np.percentile(aa,[50,75,90,95,100]),2),'trans',np.round(np.percentile(tt,[50,75,90,95,100]),4))
# mean transforms static
for a,b in [(0,100),(375,477),(390,477)]:
 for n,A in [('world',T),('coco',C),('fp',F)]:
  M=A[a:b]
  # rotation variation to mean quaternion? use first
  r=np.einsum('nji,jk->nik',M[:,:3,:3],M[0,:3,:3]); aa=np.linalg.norm(R.from_matrix(r).as_rotvec(),axis=1)*180/np.pi
  print(a,b,n,'std center',M[:,:3,3].std(0),'angles to first p',np.percentile(aa,[50,90,100]))
PY"
```
<details><summary>output</summary>

```

 0 105
world rot [1.68 2.26 3.14 3.35 5.95] trans [0.0114 0.0185 0.0246 0.0313 0.0605]
coco rot [1.68 2.26 3.14 3.35 5.95] trans [0.0109 0.0184 0.0239 0.0298 0.0599]
fp rot [1.76 2.93 5.59 6.51 9.58] trans [0.0161 0.027  0.039  0.047  0.0644]

 0 100
world rot [1.69 2.24 3.11 3.32 5.29] trans [0.0115 0.0188 0.0249 0.0319 0.0605]
coco rot [1.69 2.24 3.11 3.32 5.29] trans [0.011  0.0187 0.0241 0.0306 0.0599]
fp rot [1.69 2.78 5.48 6.4  9.58] trans [0.0163 0.0277 0.0394 0.0473 0.0644]

 370 477
world rot [ 2.    2.95  5.14  5.88 30.28] trans [0.0083 0.0127 0.021  0.0231 0.0351]
coco rot [ 2.    2.95  5.14  5.88 30.27] trans [0.0097 0.0155 0.0227 0.0263 0.0367]
fp rot [ 2.06  3.    4.51  6.55 31.86] trans [0.0129 0.023  0.0311 0.0341 0.048 ]

 375 477
world rot [1.98 2.93 5.08 5.59 9.78] tran …[+1985 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
p='/work/inputs/object_mesh.glb'; s=trimesh.load(p,force='scene',process=False); g=s.to_geometry(); print(type(g),len(g) if hasattr(g,'__len__') else '')
if isinstance(g,dict): geoms=list(g.values())
else: geoms=[g]
for i,m in enumerate(geoms): print(i,m.vertices.shape,m.faces.shape,m.visual.kind, 'bounds',m.bounds, 'extents',m.extents, 'center',m.centroid)
print('scene bounds',s.bounds, 'bounds center',s.bounds.mean(0))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.base.Trimesh'> 
0 (297065, 3) (99076, 3) texture bounds [[-0.04442554 -0.04279278 -0.1302719 ]
 [ 0.04542691  0.04405301  0.16690788]] extents [0.08985245 0.0868458  0.29717977] center [-2.95402643e-11  2.44770762e-11 -1.26793660e-12]
scene bounds [[-0.04442554 -0.04279278 -0.1302719 ]
 [ 0.04542691  0.04405301  0.16690788]] bounds center [0.00050068 0.00063012 0.01831799]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"nvdiffrast\\|MeshRasterizer\\|render_mesh\\|texture\" -n /code/reconstruction/modules/v2d_cari4d/lib/cari4d | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/predict_score.py:73:    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra)
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/training_config.py:32:    render_backend:Optional[str] = 'nvdiffrast'
/code/reconstruction/modules/v2d_cari4d/lib/cari4d/learning/training/predict_pose_refine.py:43:    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bb …[+17912 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
mods=['nvdiffrast','pyrender','pytorch3d','open3d','scipy','skimage','romatch','kornia','cotracker','torchvision']
for m in mods:
 try:
  x=__import__(m); print(m,'OK',getattr(x,'__version__',''))
 except Exception as e: print(m,'NO',type(e).__name__,str(e)[:100])
PY"
```
<details><summary>output</summary>

```
pyrender OK 0.1.45
pytorch3d OK 0.7.9
open3d OK 0.20.0
scipy OK 1.16.3
skimage OK 0.26.0
romatch NO ModuleNotFoundError No module named 'romatch'
kornia OK 0.8.3
cotracker NO ModuleNotFoundError No module named 'cotracker'
torchvision OK 0.20.1+cu124

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,310p' /code/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py"
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


import os, sys, torch, trimesh, logging, importlib, scipy
import nvdiffrast.torch as dr
import torch.nn.functional as F
import torchvision
from functools import lru_cache
import open3d as o3d
import cv2
import numpy as np
from collections import defaultdict
from transformations import *
from collections import OrderedDict
import ruamel.yaml
yaml = ruamel.yaml.YAML()
cod …[+11620 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '120,245p' /code/reconstruction/modules/v2d_cari4d/lib/cari4d/Utils.py"
```
<details><summary>output</summary>

```
def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None, output_size=None, use_light=False, light_color=None, light_dir=np.array([0,0,1]), light_pos=np.array([0,0,0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
  '''Just plain rendering, not support any gradient
  @K: (3,3) np array
  @ob_in_cams: (N,4,4) torch tensor, openCV camera
  @projection_mat: np array (4,4)
  @output_size: (height, width)
  @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
  @light_dir: in cam space
  @light_pos: in cam space
  '''
  if glctx is None:
    if context == 'gl':
      glctx = dr.RasterizeGLContext()
    elif context=='cuda':
      glctx = dr.RasterizeCudaContext()
    else:
 …[+5001 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,cv2,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.visual
print(type(v),type(v.material),v.material)
img=v.material.image
print(img.size,img.mode)
cv2.imwrite('/work/scratch/texture.png',cv2.cvtColor(np.array(img.convert('RGB')),cv2.COLOR_RGB2BGR))
# vertex color/texture UV stats
print('uv',v.uv.shape,v.uv.min(0),v.uv.max(0))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> <class 'trimesh.visual.material.PBRMaterial'> <trimesh.visual.material.PBRMaterial object at 0x741cd9177390>
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
AttributeError: 'PBRMaterial' object has no attribute 'image'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=m.visual
print(v.material.__dict__.keys())
for k,x in v.material.__dict__.items(): print(k,type(x),getattr(x,'shape',None),repr(x)[:300])
print(v.uv.shape,v.uv.min(0),v.uv.max(0))
PY"
```
<details><summary>output</summary>

```
dict_keys(['_data'])
_data <class 'dict'> None {'metallicFactor': 0.0, 'baseColorTexture': <PIL.Image.Image image mode=RGB size=1425x1557 at 0x70CD355D8B90>, 'doubleSided': True, 'name': '_texture'}
(297065, 2) [0.002625   0.00231862] [0.99754399 0.99775201]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,cv2,numpy as np
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); img=m.visual.material.baseColorTexture
cv2.imwrite('/work/scratch/texture.png',cv2.cvtColor(np.array(img.convert('RGB')),cv2.COLOR_RGB2BGR)); print(img.size)
PY"
```
<details><summary>output</summary>

```
(1425, 1557)

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/test_render.py <<'PY'
import sys,os,json
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib/cari4d')
import numpy as np, torch, trimesh, cv2
from Utils import nvdiffrast_render, make_mesh_tensors
from transformations import transform_pts
z=np.load('/work/inputs/reconstruction.npz'); Tw=z['T_world_mesh']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
# patch make tensors for material
img=np.array(m.visual.material.baseColorTexture.convert('RGB'))
mt={}; mt['tex']=torch.as_tensor(img,device='cuda',dtype=torch.float)[None]/255.
mt['uv_idx']=torch.as_tensor(m.faces,device='cuda',dtype=torch.int)
uv=torch.as_tensor(m.visual.uv,device='cuda',dtype=torch.float); uv[:,1]=1-uv[:,1]; mt['uv']=uv
mt['pos']=torch.tensor(m.vertices,device='cuda',dtype=torch.float); mt['faces']=torch.tensor(m.faces,device='cuda',dtype=torch.int)
mt['vnormals']=torch.tensor(m.vertex_normals,device='cuda',dtype=torch.float)
gl=None
inds=[0,60,110,180,260,300,330,360,400,470]
out=[]
for i in inds:
 rgb=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')
 Tcm=Tcw@Tw[i]
 color,depth,_=nvdiffrast_render(K=K,H=1152,W=1536,ob_in_cams=torch.tensor(Tcm[None],device='cuda',dtype=torch.float),glctx=gl,mesh_tensors=mt,output_size=(288,384),use_light=False,extra={})
 c=(color[0].detac …[+387 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/test_render.py", line 5, in <module>
    from transformations import transform_pts
ImportError: cannot import name 'transform_pts' from 'transformations' (/opt/conda/lib/python3.11/site-packages/transformations/__init__.py)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '/from transformations/d' /work/scratch/test_render.py; python /work/scratch/test_render.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz'); W=z['T_world_mesh']; C=z['T_cam_mesh_coconet']; F=z['T_cam_mesh_foundationpose']
A=W[0]@np.linalg.inv(C[0]);
for name,X in [('C',C),('F',F)]:
 Y=np.einsum('ij,njk->nik',A,X)
 d=np.linalg.norm(Y[:,:3,3]-W[:,:3,3],axis=1)
 print(name,'trans diff',np.percentile(d,[0,50,90,95,99,100]))
 print('first/last',Y[0],W[0])
# Determine A rotation
print('A',A)
# cam world check
import json
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); print('Twc',Twc)
PY"
```
<details><summary>output</summary>

```
C trans diff [2.38883786e-07 3.12713794e-02 1.88985229e-01 2.03042522e-01
 2.05313433e-01 2.06721485e-01]
first/last [[ 0.9968962   0.06409964 -0.04570589  0.16693087]
 [-0.04277655 -0.046352   -0.9980088   0.1970216 ]
 [-0.06609049  0.9968665  -0.04346609  3.1192622 ]
 [ 0.          0.          0.          1.        ]] [[ 0.99689627  0.06409964 -0.04570589  0.16693087]
 [-0.04277656 -0.04635199 -0.9980088   0.19702162]
 [-0.06609049  0.9968665  -0.04346608  3.1192625 ]
 [ 0.          0.          0.          1.        ]]
F trans diff [0.01450818 0.15909177 0.30787598 0.45884765 1.0048713  2.3221159 ]
first/last [[ 0.9979965   0.04639158 -0.04302189  0.1737688 ]
 [-0.03941445 -0.07605535 -0.99632424  0.23175949]
 [-0.04949299  0.9960239  -0.0740744   2.9244382 ]
 [ 0.          0.          0 …[+757 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch,h5py,numpy as np
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
for sec in ['pr','pr_initial','in','observations','raw','postopt','metadata','config']:
 print('\\n###',sec,type(x[sec]))
 if isinstance(x[sec],dict):
  for k,v in x[sec].items():
   if hasattr(v,'shape'): print(k,tuple(v.shape),v.dtype)
   else: print(k,type(v),repr(v)[:250])
with h5py.File('/pipeline/inference/materialized_inputs.h5') as f:
 print('input rgb range',f['input_rgbs'][0].min(),f['input_rgbs'][0].max())
 print('render rgb range',f['render_rgbs'][0].min(),f['render_rgbs'][0].max())
 print('masks',f['observed_object_masks'][0].sum(),f['observed_human_masks'][0].sum())
 print('K0',f['K_rois'][0], 'bbox',f['bboxes'][0])
PY"
```
<details><summary>output</summary>

```

### pr <class 'dict'>
pose_abs (477, 4, 4) float32
pose_abs_1st_delta (477, 4, 4) float32
mhr_global_rot6d (477, 6) float32
mhr_trans (477, 3) float32
mhr_body_pose_cont (477, 260) float32
mhr_hand (477, 108) float32
mhr_shape (477, 45) float32
mhr_scale (477, 28) float32
mhr_face (477, 72) float32
contact_logits (477, 2) float32
pose_abs_postopt (477, 4, 4) float32

### pr_initial <class 'dict'>
pose_abs (477, 4, 4) float32
pose_abs_1st_delta (477, 4, 4) float32
mhr_global_rot6d (477, 6) float32
mhr_trans (477, 3) float32
mhr_body_pose_cont (477, 260) float32
mhr_hand (477, 108) float32
mhr_shape (477, 45) float32
mhr_scale (477, 28) float32
mhr_face (477, 72) float32
contact_logits (477, 2) float32

### in <class 'dict'>
pose_abs (477, 4, 4) float32
mhr_global_rot6d (477, 6) float32
mhr …[+16924 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/diag_hyp.py <<'PY'
import sys,os,json
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib/cari4d')
import numpy as np, torch, trimesh, cv2
from Utils import nvdiffrast_render
import h5py
z=np.load('/work/inputs/reconstruction.npz'); W=z['T_world_mesh']; C=z['T_cam_mesh_coconet']; F=z['T_cam_mesh_foundationpose']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc)
A=W[0]@np.linalg.inv(C[0]); FW=np.einsum('ij,njk->nik',A,F)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
img=np.array(m.visual.material.baseColorTexture.convert('RGB'))
mt={}; mt['tex']=torch.as_tensor(img,device='cuda',dtype=torch.float)[None]/255.
mt['uv_idx']=torch.as_tensor(m.faces,device='cuda',dtype=torch.int)
uv=torch.as_tensor(m.visual.uv,device='cuda',dtype=torch.float); uv[:,1]=1-uv[:,1]; mt['uv']=uv
mt['pos']=torch.tensor(m.vertices,device='cuda',dtype=torch.float); mt['faces']=torch.tensor(m.faces,device='cuda',dtype=torch.int); mt['vnormals']=torch.tensor(m.vertex_normals,device='cuda',dtype=torch.float)
gl=None
Rflips={
 'z180':np.diag([-1.,-1.,1.]),
 'x180':np.diag([1.,-1.,-1.]),
 'y180':np.diag([-1.,1.,-1.]),
 'z90':np.array([[0,-1,0],[1,0,0],[0,0,1.]],float),
 'z270':np.array([[0,1,0],[-1,0,0],[0,0,1.]],float),
}
def proj(T,pts):
 cc=pts@T[:3,:3].T+T[:3,3]; uv=cc@K.T; return uv[:,:2]/uv[:,2:3]
# object mask bboxe …[+1985 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/diag_hyp.py", line 59, in <module>
    for nm,T in poses[:5]: ims.append(label(ov(im,render(i,T,bb)),nm))
                                                  ^^^^^^^^^^^^^^
  File "/work/scratch/diag_hyp.py", line 39, in render
    x0,y0,s=bb; Kc=K.copy(); Kc[0,2]-=x0; Kc[1,2]-=y0
    ^^^^^^^
ValueError: not enough values to unpack (expected 3, got 2)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/diag_hyp.py'
s=open(p).read().replace(\"x0,y0,s=bb; Kc=K.copy()\", \"im0,(x0,y0,s)=bb; Kc=K.copy()\")
open(p,'w').write(s)
PY
python /work/scratch/diag_hyp.py"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created c …[+6657 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/diag_hyp.py'; s=open(p).read();
# modify final loop to custom set
s=s.split(\"for batch,inds in enumerate(\")[0]
s += r'''
inds=[240,255,260,262,263,264,265,266,268,269,270,275,300,315,320,323,325,327,330,335,340,345,350,355,360,365,368,370,375,400,470]
rows=[]
for i in inds:
 bb=crop_image(i); im=bb[0]
 poses=[('base',W[i]),('FP',FW[i])]
 for nm in ['z180','x180','y180','z90','z270']:
  T=W[i].copy(); T[:3,:3]=W[i][:3,:3]@Rflips[nm]; poses.append((nm,T))
 ims=[label(cv2.resize(im,(256,256)),str(i)+' orig')]
 for nm,T in poses: ims.append(label(ov(im,render(i,T,bb)),nm))
 rows.append(np.hstack(ims[:6]))
cv2.imwrite('/work/scratch/hyp3.jpg',np.vstack(rows))
'''
open('/work/scratch/diag_hyp3.py','w').write(s)
PY
python /work/scratch/diag_hyp3.py >/work/scratch/diag3.log 2>&1; tail -5 /work/scratch/diag3.log"
```
<details><summary>output</summary>

```
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/score_rots.py <<'PY'
import sys,os,json
sys.path.insert(0,'/code/reconstruction/modules/v2d_cari4d/lib/cari4d')
import numpy as np, torch, trimesh, cv2, h5py
from Utils import nvdiffrast_render
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); W=z['T_world_mesh']; C=z['T_cam_mesh_coconet']; F=z['T_cam_mesh_foundationpose']
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K'],np.float32); Twc=np.array(cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc)
A=W[0]@np.linalg.inv(C[0]); FW=np.einsum('ij,njk->nik',A,F)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); verts=np.asarray(m.vertices)
imgtex=np.array(m.visual.material.baseColorTexture.convert('RGB'))
mt={'tex':torch.as_tensor(imgtex,device='cuda',dtype=torch.float)[None]/255.,'uv_idx':torch.as_tensor(m.faces,device='cuda',dtype=torch.int)}
uv=torch.as_tensor(m.visual.uv,device='cuda',dtype=torch.float); uv[:,1]=1-uv[:,1]; mt['uv']=uv
mt['pos']=torch.tensor(m.vertices,device='cuda',dtype=torch.float); mt['faces']=torch.tensor(m.faces,device='cuda',dtype=torch.int); mt['vnormals']=torch.tensor(m.vertex_normals,device='cuda',dtype=torch.float)
gl=None
with h5py.File('/work/inputs/masks.h5') as h: OM=h['object'][:]; PM=h['person'][:]
def proj(T,pts):
 cc=pts@T[:3,:3].T+T[:3,3]; uv=cc@K.T; return uv[:,:2]/uv[:,2:3]
def cropbox(i,s=192):
 uv=proj(Tcw@W[i],verts)
 if OM[i].sum()>20:
  ys,x …[+2672 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
0 BASE (0.702, 0.378, 0.111) FP (0.662, 0.379, 0.206) TOP [('base', 0.702, 0.378, 0.111), ('z0', 0.702, 0.378, 0.111), ('z330', 0.703, 0.267, 0.058), ('FP', 0.662, 0.379, 0.206)]
60 BASE (0.811, 0.577, 0.299) FP (0.71, 0.436, 0.192) TOP [('base', 0.811, 0.577, 0.299), ('z0', 0.811, 0.577, 0.299), ('z30', 0.788, 0.346, 0.103), ('z330', 0.804, 0.176, 0.162)]
100 BASE (0.796, 0.448, 0.239) FP (0.778, 0.392, 0.214) TOP [('base', 0.796, 0.448, 0.239), ('z0', 0.796, 0.448, 0.239), ('z330', 0.806, 0.304, 0.113), ('FP', 0.778, 0.392, 0.214)]
240 BASE (0.818, 0.312, …[+3548 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/single_diag.py <<'PY'
exec(open('/work/scratch/diag_hyp.py').read().split('for batch,inds')[0])
import sys
inds=[int(x) for x in sys.argv[1:]]
for i in inds:
 bb=crop_image(i,size=220); im=bb[0]
 poses=[('base',W[i]),('FP',FW[i])]
 for nm in ['z180','z90','z270']:
  T=W[i].copy();T[:3,:3]=W[i][:3,:3]@Rflips[nm];poses.append((nm,T))
 ims=[label(cv2.resize(im,(420,420)),str(i)+' orig')]
 for nm,T in poses: ims.append(label(ov(im,render(i,T,bb,outsize=420)),nm))
 cv2.imwrite(f'/work/scratch/single_{i}.jpg',np.hstack(ims))
PY
python /work/scratch/single_diag.py 260 320 325 330 335 340 345 350 355 360 365 370 400 470 >/work/scratch/single.log 2>&1; tail -3 /work/scratch/single.log"
```
<details><summary>output</summary>

```
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 45, in ov
IndexError: boolean index did not match indexed array along dimension 0; dimension is 256 but corresponding boolean dimension is 420

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/diag_hyp.py';s=open(p).read().replace(\"def ov(im,r):\\n im=cv2.resize(im,(256,256)); alpha=(r.sum(2)>5)\",\"def ov(im,r):\\n im=cv2.resize(im,(r.shape[1],r.shape[0])); alpha=(r.sum(2)>5)\")
open(p,'w').write(s)
PY
python /work/scratch/single_diag.py 260 320 325 330 335 340 345 350 355 360 365 370 400 470 >/work/scratch/single.log 2>&1; tail -3 /work/scratch/single.log"
```
<details><summary>output</summary>

```
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:
 o=f['object'][:]
for i in [300,315,318,319,320,321,322,323,324,325,326,327,328,329,330,335,340,345]:
 ys,xs=np.where(o[i]); print(i,len(xs), (xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None)
PY"
```
<details><summary>output</summary>

```
300 2410 (469, 663, 508, 758)
315 1788 (486, 652, 520, 736)
318 1213 (495, 653, 521, 733)
319 962 (500, 654, 522, 731)
320 741 (503, 652, 522, 726)
321 394 (506, 653, 520, 709)
322 91 (509, 652, 519, 665)
323 0 None
324 0 None
325 0 None
326 0 None
327 0 None
328 0 None
329 0 None
330 0 None
335 0 None
340 0 None
345 1214 (634, 729, 656, 790)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:]
for i in [320,330,340,345]:
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg'); ys,xs=np.where(o[i]); print(i,im.shape)
 if len(xs):
  x0,x1,y0,y1=xs.min()-60,xs.max()+60,ys.min()-60,ys.max()+60
  im=im[max(0,y0):y1,max(0,x0):x1]
 cv2.imwrite(f'/work/scratch/maskcrop_{i}.jpg',cv2.resize(im,(im.shape[1]*3,im.shape[0]*3)))
PY"
```
<details><summary>output</summary>

```
320 (1152, 1536, 3)
330 (1152, 1536, 3)
340 (1152, 1536, 3)
345 (1152, 1536, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/diag_hyp.py').read().split('for batch,inds')[0])
for i in [260,320,330,340,345,400]:
 bb=crop_image(i,size=220); print(i,bb[1])
PY"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
260 (620, 634, 235)
320 (402, 579, 220)
330 (462, 572, 220)
340 (471, 583, 220)
345 (535, 650, 220)
400 (626, 730, 220)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];H=z['human_vertices_world'];K=z['human_keypoints_world'];cp=z['contact_prob_right_left'];c=T[:,:3,3]
# nearest distance center to hands keypoints (all 21 points)
for i in range(0,477,10):
 d1=np.linalg.norm(K[i,21:42]-c[i],axis=1).min(); d2=np.linalg.norm(K[i,42:63]-c[i],axis=1).min()
 # nearest human vertices to center
 dv=np.linalg.norm(H[i]-c[i],axis=1).min()
 print(i,round(d1,3),round(d2,3),round(dv,3),np.round(cp[i],2),np.round(c[i],2))
PY"
```
<details><summary>output</summary>

```
0 0.757 1.101 0.743 [0. 0.] [0.17 0.2  3.12]
10 0.702 1.149 0.655 [0. 0.] [0.17 0.19 3.17]
20 0.709 1.166 0.642 [0. 0.] [0.17 0.19 3.16]
30 0.788 1.189 0.682 [0. 0.] [0.17 0.19 3.19]
40 0.898 1.363 0.844 [0. 0.] [0.17 0.18 3.18]
50 0.851 1.318 0.82 [0. 0.] [0.18 0.17 3.29]
60 0.808 1.028 0.798 [0. 0.] [0.17 0.18 3.23]
70 0.78 0.541 0.538 [0. 0.] [0.17 0.17 3.28]
80 0.838 0.352 0.348 [0. 0.] [0.17 0.17 3.26]
90 0.86 0.227 0.222 [0. 0.] [0.17 0.17 3.26]
100 0.844 0.124 0.119 [0. 0.] [0.17 0.17 3.26]
110 0.773 0.094 0.088 [0.74 0.  ] [0.17 0.17 3.29]
120 0.76 0.127 0.12 [0.99 0.  ] [0.17 0.16 3.3 ]
130 0.751 0.089 0.082 [1. 0.] [0.09 0.15 3.29]
140 0.617 0.109 0.105 [1. 0.] [-0.11  0.14  3.24]
150 0.606 0.123 0.112 [1. 0.] [-0.34  0.14  3.32]
160 0.649 0.069 0.058 [1. 0.] [-0.56  0.18  3.31]
 …[+1496 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];K=z['human_keypoints_world']; c=T[:,:3,3]
idx=[];ds=[];off=[]
for i in range(105,366):
 d=np.linalg.norm(K[i,42:63]-c[i],axis=1); j=d.argmin();idx.append(j+42);ds.append(d[j]);off.append(c[i]-K[i,j+42])
print('idx counts')
from collections import Counter
print(Counter(idx).most_common(20))
for i in range(105,366,5): print(i,idx[i-105],round(ds[i-105],3),np.round(off[i-105],3))
PY"
```
<details><summary>output</summary>

```
idx counts
[(59, 106), (60, 54), (58, 22), (61, 15), (56, 13), (52, 12), (57, 8), (50, 7), (55, 4), (51, 4), (53, 4), (62, 3), (46, 2), (49, 2), (44, 2), (47, 1), (45, 1), (42, 1)]
105 59 0.102 [0.065 0.031 0.072]
110 59 0.094 [0.05  0.038 0.07 ]
115 60 0.122 [0.053 0.045 0.1  ]
120 60 0.127 [0.056 0.046 0.105]
125 60 0.114 [0.039 0.067 0.083]
130 60 0.089 [0.028 0.075 0.039]
135 59 0.09 [0.012 0.089 0.007]
140 58 0.109 [ 0.013  0.098 -0.046]
145 58 0.142 [ 0.035  0.084 -0.109]
150 58 0.123 [ 0.039  0.086 -0.078]
155 58 0.085 [ 0.016  0.056 -0.062]
160 61 0.069 [-0.012  0.061 -0.029]
165 60 0.058 [ 0.     0.056 -0.016]
170 59 0.069 [-0.002  0.067 -0.015]
175 60 0.066 [-0.016  0.064  0.001]
180 60 0.073 [-0.018  0.071  0.001]
185 59 0.061 [-0.022  0.057 -0.006]
190 60 0.066 [-0.015  0.064 - …[+1253 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];K=z['human_keypoints_world'];cp=z['contact_prob_right_left']; c=T[:,:3,3]
def ang(i):
 r=T[i-1,:3,:3].T@T[i,:3,:3];return np.linalg.norm(R.from_matrix(r).as_rotvec())*180/np.pi
for i in range(305,378):
 d=np.linalg.norm(K[i,42:63]-c[i],axis=1);j=d.argmin()+42
 st=np.linalg.norm(c[i]-c[i-1])
 print(i,np.round(c[i],3),'step',round(st,3),'rot',round(ang(i),1),'hand',j,np.round(K[i,j],3),'d',round(d.min(),3),'cp',round(float(cp[i,0]),2))
PY"
```
<details><summary>output</summary>

```
305 [0.905 0.449 3.604] step 0.002 rot 4.6 hand 56 [0.835 0.315 3.632] d 0.154 cp 1.0
306 [0.91  0.439 3.6  ] step 0.012 rot 3.7 hand 56 [0.823 0.323 3.612] d 0.145 cp 1.0
307 [0.92  0.453 3.535] step 0.067 rot 3.3 hand 56 [0.815 0.354 3.522] d 0.144 cp 1.0
308 [0.914 0.444 3.547] step 0.017 rot 3.2 hand 60 [0.808 0.345 3.574] d 0.147 cp 1.0
309 [0.906 0.437 3.545] step 0.01 rot 4.4 hand 60 [0.816 0.342 3.555] d 0.131 cp 1.0
310 [0.904 0.438 3.549] step 0.004 rot 4.5 hand 60 [0.819 0.346 3.535] d 0.127 cp 1.0
311 [0.915 0.456 3.481] step 0.07 rot 3.8 hand 60 [0.83  0.382 3.46 ] d 0.115 cp 1.0
312 [0.911 0.451 3.479] step 0.007 rot 2.4 hand 61 [0.824 0.363 3.466] d 0.124 cp 1.0
313 [0.906 0.454 3.47 ] step 0.011 rot 2.5 hand 61 [0.811 0.381 3.458] d 0.12 cp 1.0
314 [0.903 0.451 3.451] step  …[+5525 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,h5py
idxs=list(range(315,376))
ims=[]
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:]
for i in idxs:
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[580:860,450:750]
 im=cv2.resize(im,(200,187))
 cv2.putText(im,str(i),(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3)
 cv2.putText(im,str(i),(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
 ims.append(im)
while len(ims)%8:ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/carry_315_375.jpg',np.vstack([np.hstack(ims[j:j+8]) for j in range(0,len(ims),8)]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/single_diag.py 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 >/work/scratch/single2.log 2>&1; tail -3 /work/scratch/single2.log'
```
<details><summary>output</summary>

```
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');W=z['T_world_mesh'][:,:3,:3]; C=z['T_cam_mesh_coconet'][:,:3,:3]
# intrinsic angles: rotation relative to first
for ref in [0,240,340]:
 print('\\nREF',ref)
 for i in range(ref if ref else 0,477,10):
  rr=R.from_matrix(W[ref].T@W[i]); rv=rr.as_rotvec(); ang=np.linalg.norm(rv)*180/np.pi
  print(i, np.round(rv*180/np.pi,1), round(ang,1))
PY"
```
<details><summary>output</summary>

```

REF 0
0 [-0. -0.  0.] 0.0
10 [-0.1  0.4 -0.7] 0.8
20 [ 0.6  1.  -5.3] 5.4
30 [-0.7  1.8 -3.6] 4.1
40 [-2.   0.6 -4.6] 5.0
50 [ 1.3  2.3 -2.2] 3.4
60 [-0.1  1.4 -2.6] 3.0
70 [-1.1  1.5 -2.9] 3.4
80 [ 0.2  1.2 -4.5] 4.6
90 [ 0.1  1.7 -3.1] 3.5
100 [ 4.5  1.7 -1.6] 5.0
110 [ 3.8  1.6 -6.5] 7.7
120 [ 2.5  2.4 -5.3] 6.4
130 [ 1.2  7.9 10.3] 13.0
140 [-9.2  8.4 18.6] 22.4
150 [-3.9  5.3 63. ] 63.3
160 [-5.1  4.9 92.7] 92.9
170 [-12.    4.4 101.5] 102.3
180 [-12.2   3.7 105.4] 106.2
190 [-13.3   3.3 117.8] 118.6
200 [-15.1   3.4 126.7] 127.7
210 [-10.4   2.  136.7] 137.1
220 [-11.3   4.5 143.2] 143.7
230 [ -7.3  -1.8 152. ] 152.2
240 [ -9.5  -5.5 122.4] 122.9
250 [ -8.9  -1.7 110.3] 110.6
260 [-1.  -3.  85.4] 85.4
270 [ 0.1 -0.7 69.8] 69.8
280 [-5.9  1.8 58.9] 59.3
290 [-5.4 10.6 50.6] 52.0
300  …[+1659 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:]
for i in [400,430,470]:
 ys,xs=np.where(o[i]);print(i,len(xs),(xs.min(),ys.min(),xs.max(),ys.max()),'cent',xs.mean(),ys.mean())
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg'); print('nonbg around',im[ys.mean().astype(int),xs.mean().astype(int)])
PY"
```
<details><summary>output</summary>

```
400 1554 (726, 798, 747, 882) cent 737.989703989704 842.1634491634492
nonbg around [90 83 68]
430 1546 (726, 798, 747, 882) cent 737.7723156532988 842.1934023285899
nonbg around [94 87 72]
470 1573 (726, 798, 747, 882) cent 737.6039415130324 841.3363000635728
nonbg around [102  97  82]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/diag_hyp.py').read().split('for batch,inds')[0])
for i in [400,470]: print(i,crop_image(i,size=220)[1])
PY"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
400 (626, 730, 220)
470 (626, 730, 220)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/bbox_metrics.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
for i in [0,60,100,240,260,300,320,345,350,355,360,365,370,380,400,430,470]:
 box=cropbox(i); x0,y0,s=box
 rgb,dep=render_pose(i,W[i],box)
 rm=dep>0.01; om=OM[i,y0:y0+s,x0:x0+s]
 def bb(m):
  ys,xs=np.where(m);return (xs.min(),ys.min(),xs.max(),ys.max()) if len(xs) else None
 rb=bb(rm);ob=bb(om)
 print(i,'render',rb,'mask',ob,'iou',round((rm&om).sum()/max(1,(rm|om).sum()),3),'areas',rm.sum(),om.sum())
PY
python /work/scratch/bbox_metrics.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
0 render (86, 57, 113, 137) mask (83, 53, 109, 139) iou 0.702 areas 1552 2049
60 render (85, 55, 110, 138) mask (83, 53, 109, 139) iou 0.811 areas 1676 2049
100 render (85, 54, 111, 138) mask (83, 53, 109, 139) iou 0.796 areas 1724 2003
240 render (84, 47, 143, 176) mask (80, 45, 140, 176) iou 0.818 areas 3851 4308
260 render (92, 32, 133, 162) mask (91, 44, 131, 179) iou 0.717 areas 3513 3910
300 render (78, 49, 119, 148) mask (77, 49, 116, 144) iou 0.81 areas 2348 2410
320 render (85, 48, 125, 136) mask (87, 59, 106, 133) iou 0.322 areas 1904 741
345 rend …[+790 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/pool_static.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def avg_pose(ids):
 # quaternion average
 qs=R.from_matrix(W[ids,:3,:3]).as_quat(); q0=qs[0].copy()
 for _ in range(10):
  ref=qs.mean(0); ref/=np.linalg.norm(ref); qs=np.where((qs@ref)[:,None]<0,-qs,qs)
 rot=R.from_quat(qs.mean(0)/np.linalg.norm(qs.mean(0))).as_matrix()
 T=np.eye(4);T[:3,:3]=rot;T[:3,3]=np.median(W[ids,:3,3],axis=0); return T
for a,b in [(0,100),(380,470),(390,470),(10,90)]:
 ids=np.arange(a,b,2); T0=avg_pose(ids); print('\\nPHASE',a,b,'center',T0[:3,3])
 for mode,offs in [('z',range(-180,180,10))]:
  rows=[]
  for off in offs:
   Roff=R.from_euler('z',off,degrees=True).as_matrix();T=T0.copy();T[:3,:3]=T0[:3,:3]@Roff
   ious=[];gns=[];ggs=[]
   for i in ids:
    iou,gn,nvis,gg=scores(i,T,cropbox(i));ious.append(iou);gns.append(gn);ggs.append(gg)
   rows.append((off,np.mean(ious),np.mean(gns),np.mean(ggs)))
  top=sorted(rows,key=lambda x:-(x[2]+.2*x[3]))[:12]
  print('TOP',[(x[0],round(x[1],3),round(x[2],3),round(x[3],3)) for x in top])
  print('base',next((round(x[1],3),round(x[2],3),round(x[3],3)) for x in rows if x[0]==0))
PY
python /work/scratch/pool_static.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

PHASE 0 100 center [0.17033726 0.17890538 3.22635818]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; ps aux | grep pool_static | grep -v grep'
```
<details><summary>output</summary>

```
4173, 1404 MiB
hanzhou     4171  0.0  0.0   4788  3460 ?        SNs  21:38   0:00 /usr/bin/bash -c cat > /work/scratch/pool_static_batch.py <<'PY' exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0]) from scipy.spatial.transform import Rotation as R  def avg_pose(ids):  qs=R.from_matrix(W[ids,:3,:3]).as_quat()  for _ in range(20):   ref=qs.mean(0);ref/=np.linalg.norm(ref);qs=np.where((qs@ref)[:,None]<0,-qs,qs)  rot=R.from_quat(qs.mean(0)/np.linalg.norm(qs.mean(0))).as_matrix();T=np.eye(4);T[:3,:3]=rot;T[:3,3]=np.median(W[ids,:3,3],axis=0);return T def render_batch(i,Ts,box):  x0,y0,s=box;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0  Tb=torch.tensor(np.stack([Tcw@T for T in Ts]),device='cuda',dtype=torch.float)  cols,deps,_=nvdiffrast_render(K=Kc,H=s,W=s,ob_in_cams=Tb,g …[+1765 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/pool_static_batch.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
from scipy.spatial.transform import Rotation as R

def avg_pose(ids):
 qs=R.from_matrix(W[ids,:3,:3]).as_quat()
 for _ in range(20):
  ref=qs.mean(0);ref/=np.linalg.norm(ref);qs=np.where((qs@ref)[:,None]<0,-qs,qs)
 rot=R.from_quat(qs.mean(0)/np.linalg.norm(qs.mean(0))).as_matrix();T=np.eye(4);T[:3,:3]=rot;T[:3,3]=np.median(W[ids,:3,3],axis=0);return T
def render_batch(i,Ts,box):
 x0,y0,s=box;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0
 Tb=torch.tensor(np.stack([Tcw@T for T in Ts]),device='cuda',dtype=torch.float)
 cols,deps,_=nvdiffrast_render(K=Kc,H=s,W=s,ob_in_cams=Tb,glctx=gl,mesh_tensors=mt,output_size=(s,s),use_light=False,extra={})
 return cols.detach().cpu().numpy(),deps.detach().cpu().numpy()
def metric(i,Ts,box):
 x0,y0,s=box;im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s]; x=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255
 cols,deps=render_batch(i,Ts,box);om=OM[i,y0:y0+s,x0:x0+s];pm=PM[i,y0:y0+s,x0:x0+s];out=[]
 for rgb,dep in zip(cols,deps):
  rr=np.clip(rgb,0,1);rm=dep>0.01;vis=om&~pm&rm
  iou=(rm&om).sum()/max(1,(rm|om).sum())
  if vis.sum()<10: out.append((iou,0,0));continue
  a=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)[vis];b=cv2.cvtColor(rr,cv2.COLOR_RGB2GRAY)[vis]
  gn=np.corrcoef(a,b)[0,1] if a.std()>1e-5 and b.std()>1e-5 else 0
  g1=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)[vis]+cv2.Sobel(x,cv2.CV_32 …[+876 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

PHASE 0 100 center [0.1703 0.1805 3.2218]
TOP [(-180, 0.776, 0.061, 0.971), (-165, 0.778, 0.077, 0.883), (-150, 0.769, 0.107, nan), (-135, 0.748, 0.079, nan), (-120, 0.725, 0.089, nan), (-105, 0.723, 0.105, nan), (-90, 0.743, 0.035, nan), (-75, 0.768, 0.054, nan), (-60, 0.787, 0.088, nan), (-45, 0.789, 0.146, nan), (-30, 0.792, 0.278, nan), (-15, 0.795, 0.486, nan), (0, 0.798, 0.577, 0.773), (15, 0.797, 0.405, 0.709), (30, 0.783, 0.314, 0.847)]
base [0.798 0.577 0.773]

PHASE 380 470 center [0.1053 1.1311 3.0909]
TOP [(-180, 0.692, 0.209, nan), (-165, 0.69 …[+1330 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/single_diag.py'
# create specialized
s=open(p).read().split('inds=')[0]
s += r'''
from scipy.spatial.transform import Rotation as R
for i in [400,430,470]:
 bb=crop_image(i,size=220); im=bb[0]; poses=[]
 for a in [-180,-150,-120,-90,-75,-60,-30,0,30,60,90,120,150]:
  T=W[i].copy();T[:3,:3]=W[i][:3,:3]@R.from_euler('z',a,degrees=True).as_matrix();poses.append((str(a),T))
 ims=[label(cv2.resize(im,(320,320)),str(i)+' orig')]
 for nm,T in poses:ims.append(label(ov(im,render(i,T,bb,outsize=320)),nm))
 cv2.imwrite(f'/work/scratch/zgrid_{i}.jpg',np.vstack([np.hstack(ims[j:j+7]) for j in range(0,len(ims),7)]))
'''
open('/work/scratch/zgrid.py','w').write(s)
PY
python /work/scratch/zgrid.py >/work/scratch/zgrid.log 2>&1; tail -2 /work/scratch/zgrid.log"
```
<details><summary>output</summary>

```
[nvdiffrast_render()] created context
[nvdiffrast_render()] created context

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
for i in [400,470]:
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg'); crop=im[770:910,700:780]; cv2.imwrite(f'/work/scratch/final_{i}.png',cv2.resize(crop,None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC))
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
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:]
for i in range(360,376):
 ys,xs=np.where(o[i]); print(i,len(xs),(xs.min(),xs.max(),ys.min(),ys.max()) if len(xs) else None)
PY"
```
<details><summary>output</summary>

```
360 1392 (697, 724, 787, 854)
361 1411 (701, 728, 788, 856)
362 1397 (704, 731, 789, 858)
363 1449 (706, 735, 788, 861)
364 1457 (710, 737, 790, 862)
365 1511 (713, 741, 791, 866)
366 1628 (717, 744, 788, 868)
367 1711 (721, 746, 788, 871)
368 1679 (723, 747, 786, 871)
369 1583 (725, 747, 788, 873)
370 1371 (727, 747, 794, 875)
371 1277 (728, 747, 799, 877)
372 1281 (729, 747, 800, 880)
373 1305 (729, 747, 798, 880)
374 1237 (727, 747, 800, 881)
375 1156 (727, 747, 805, 881)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');cp=z['contact_prob_right_left'];T=z['T_world_mesh'];c=T[:,:3,3]
for i in range(95,121):print(i,np.round(cp[i],2))
print('---')
for i in range(360,386):print(i,np.round(cp[i],2),np.round(c[i],3))
PY"
```
<details><summary>output</summary>

```
95 [0. 0.]
96 [0. 0.]
97 [0. 0.]
98 [0. 0.]
99 [0. 0.]
100 [0. 0.]
101 [0. 0.]
102 [0. 0.]
103 [0. 0.]
104 [0. 0.]
105 [0.06 0.  ]
106 [0.01 0.  ]
107 [0.21 0.  ]
108 [0.81 0.  ]
109 [0.88 0.  ]
110 [0.74 0.  ]
111 [0.96 0.  ]
112 [0.96 0.  ]
113 [0.97 0.  ]
114 [0.98 0.  ]
115 [1. 0.]
116 [0.96 0.  ]
117 [0.99 0.  ]
118 [0.95 0.  ]
119 [0.98 0.  ]
120 [0.99 0.  ]
---
360 [0.98 0.  ] [0.205 1.01  3.123]
361 [0.97 0.  ] [0.193 1.009 3.145]
362 [0.9 0. ] [0.183 1.022 3.139]
363 [0.51 0.  ] [0.17  1.036 3.129]
364 [0.99 0.  ] [0.163 1.052 3.116]
365 [0.83 0.  ] [0.146 1.051 3.142]
366 [0.53 0.01] [0.135 1.015 3.223]
367 [0.28 0.  ] [0.12  1.062 3.092]
368 [0. 0.] [0.11  1.069 3.095]
369 [0.01 0.  ] [0.118 1.09  3.087]
370 [0.01 0.  ] [0.116 1.112 3.066]
371 [0. 0.] [0.11  1.118 3.073]
372 [0. …[+441 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/traj_exp.py <<'PY'
import numpy as np, json
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter, medfilt
z=np.load('/work/inputs/reconstruction.npz');T0=z['T_world_mesh'].astype(np.float64);F=len(T0)
# phases
p0,p1=0,106; d0,d1=107,378; f0,f1=379,F-1
# rotations
Rs=T0[:,:3,:3].copy(); qs=R.from_matrix(Rs).as_quat(); qs=np.where((qs[1:]@qs[:-1])[:,None]<0,-qs[1:],qs[1:]).tolist();
qs=[R.from_matrix(Rs[0]).as_quat()]+qs;qs=np.array(qs)
# quaternion sign continuity
for i in range(1,F):
 if qs[i]@qs[i-1]<0: qs[i]*=-1
# component median/savgol
Q=qs.copy()
# repair random spikes via median of rotations? use savgol then evaluate branch
for _ in range(1):
 Q2=np.stack([savgol_filter(Q[:,j],7,2,mode='interp') for j in range(4)],1);Q2/=np.linalg.norm(Q2,axis=1,keepdims=True)
 # only replace if raw local accel huge? use all
 Q=Q2
Rs_s=R.from_quat(Q).as_matrix()
# static mean rotations quaternion mean
def qmean(ids):
 q=qs[ids].copy()
 for _ in range(20):
  m=q.mean(0);m/=np.linalg.norm(m);q=np.where((q@m)[:,None]<0,-q,q)
 return q.mean(0)/np.linalg.norm(q.mean(0))
for a,b in [(p0,p1),(f0,f1)]:
 q=qmean(np.arange(a,b+1)); Rs_s[a:b+1]=R.from_quat(q).as_matrix()
# translations
t=T0[:,:3,3].copy()
# robust interp gross jumps
sp=np.linalg.norm(np.diff(t,axis=0),axis=1);bad=np.where(sp>.2)[0]+1
print('gross translation outliers',bad)
for i in bad:
 t[i]=np.nan
# interpolate all nans per coord
for j in range(3):
 ii=np. …[+1140 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/traj_exp.py", line 8, in <module>
    Rs=T0[:,:3,:3].copy(); qs=R.from_matrix(Rs).as_quat(); qs=np.where((qs[1:]@qs[:-1])[:,None]<0,-qs[1:],qs[1:]).tolist();
                                                                        ~~~~~~^~~~~~~~
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 476 is different from 4)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/traj_exp.py';s=open(p).read().replace(\"qs=np.where((qs[1:]@qs[:-1])[:,None]<0,-qs[1:],qs[1:]).tolist();\\nqs=[R.from_matrix(Rs[0]).as_quat()]+qs;qs=np.array(qs)\",\"qs=[R.from_matrix(Rs[0]).as_quat()]+list(qs[1:]);qs=np.array(qs)\")
open(p,'w').write(s)
PY
python /work/scratch/traj_exp.py"
```
<details><summary>output</summary>

```
gross translation outliers [343 344]
base rot [  2.42   9.12  21.59 163.85 175.78] trans [0.0179 0.0383 0.0525 0.1278 1.4337]
smooth rot [ 0.77  7.91 26.45 68.14 82.66] trans [0.0098 0.0283 0.0335 0.0472 0.0659]
static rot max offsets base/smooth
0 106 base r [3.62149994 6.56417225 9.75747774] tstd [0.00319105 0.00887768 0.05205875]
0 106 smooth r [0. 0. 0.] tstd [5.55111512e-17 3.33066907e-16 5.77315973e-15]
379 476 base r [ 6.50781273  8.01838788 10.3834709 ] tstd [0.00261523 0.00538779 0.01246317]
379 476 smooth r [0. 0. 0.] tstd [2.77555756e-17 4.44089210e-16 0.00000000e+00]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/unwrap_test.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'].astype(float);Rs=T[:,:3,:3];F=len(T)
Rz=np.diag([-1.,-1.,1.]); Rc=[]
def ad(a,b): return np.linalg.norm(R.from_matrix(a.T@b).as_rotvec())*180/np.pi
Rc=[Rs[0]]
fl=[]
for i in range(1,F):
 d=ad(Rc[-1],Rs[i]); df=ad(Rc[-1],Rs[i]@Rz)
 if df+25<d:
  Rc.append(Rs[i]@Rz);fl.append((i,round(d,1),round(df,1),'z'))
 else:Rc.append(Rs[i])
print('flips',fl)
aa=[ad(Rc[i-1],Rc[i]) for i in range(1,F)]
print('step pct',np.percentile(aa,[50,90,95,99,100]))
print('big')
for i in np.where(np.array(aa)>25)[0]: print(i+1,round(aa[i],1))
# now median quaternion components then savgol
q=R.from_matrix(np.array(Rc)).as_quat()
for i in range(1,F):
 if q[i]@q[i-1]<0:q[i]*=-1
for filt in [('sg',lambda x:savgol_filter(x,7,2,mode='interp')),('med',lambda x:medfilt(x,5))]:
 import scipy.signal as ss
 Q=np.stack([filt[1](q[:,j]) for j in range(4)],1);Q/=np.linalg.norm(Q,axis=1,keepdims=True);S=R.from_quat(Q).as_matrix();aa=[ad(S[i-1],S[i]) for i in range(1,F)];print(filt[0],np.percentile(aa,[50,90,95,99,100]))
PY
python /work/scratch/unwrap_test.py"
```
<details><summary>output</summary>

```
flips [(327, 170.7, 15.1, 'z'), (328, 162.3, 89.1, 'z'), (329, 148.2, 80.8, 'z'), (330, 106.8, 74.3, 'z'), (331, 172.6, 7.9, 'z'), (332, 174.3, 7.7, 'z'), (336, 145.3, 35.5, 'z'), (345, 129.4, 51.6, 'z'), (346, 173.1, 7.8, 'z'), (347, 168.7, 11.3, 'z'), (348, 177.1, 7.5, 'z'), (349, 167.9, 12.7, 'z'), (350, 146.0, 36.0, 'z'), (351, 160.2, 20.3, 'z'), (352, 163.9, 16.2, 'z'), (353, 157.0, 25.7, 'z'), (354, 170.3, 13.4, 'z'), (355, 172.0, 9.8, 'z'), (356, 168.5, 11.7, 'z'), (357, 162.1, 18.3, 'z'), (358, 169.0, 11.1, 'z'), (359, 179.0, 1.2, 'z'), (360, 178.5, 2.5, 'z'), (361, 173.5, 7.3, 'z'), (362, 176.0, 4.1, 'z'), (363, 167.4, 12.8, 'z'), (364, 175.5, 5.2, 'z'), (365, 165.8, 14.9, 'z'), (366, 175.1, 7.6, 'z'), (367, 176.7, 6.3, 'z'), (368, 171.8, 12.5, 'z')]
step pct [  2.41820523   8.872 …[+1170 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');W=z['T_world_mesh'][:,:3,:3];ref=W[320];Rz=np.diag([-1.,-1.,1.])
def a(A,B):return np.linalg.norm(R.from_matrix(A.T@B).as_rotvec())*180/np.pi
for i in range(315,381):
 d0=a(ref,W[i]);d1=a(ref,W[i]@Rz)
 print(i,round(d0,1),round(d1,1),'choice',0 if d0<d1 else 180,'step',round(a(W[i-1],W[i]),1))
PY"
```
<details><summary>output</summary>

```
315 19.9 161.5 choice 0 step 11.5
316 13.9 166.9 choice 0 step 6.6
317 10.1 172.5 choice 0 step 6.4
318 10.4 169.9 choice 0 step 7.9
319 9.0 179.1 choice 0 step 12.4
320 0.0 180.0 choice 0 step 9.0
321 8.4 176.9 choice 0 step 8.4
322 8.2 172.5 choice 0 step 6.5
323 9.9 175.9 choice 0 step 13.0
324 56.3 127.7 choice 0 step 62.5
325 56.4 126.2 choice 0 step 6.1
326 170.6 173.5 choice 0 step 163.2
327 175.3 178.0 choice 0 step 170.7
328 118.7 131.5 choice 0 step 89.1
329 179.3 168.5 choice 180 step 80.8
330 168.3 180.0 choice 0 step 74.3
331 167.1 178.2 choice 0 step 7.9
332 162.3 176.2 choice 0 step 7.7
333 83.8 102.0 choice 0 step 175.8
334 101.5 81.8 choice 180 step 21.9
335 106.9 75.8 choice 180 step 6.9
336 41.5 140.3 choice 0 step 145.3
337 161.2 176.8 choice 0 step 170.9
338 151.1 176. …[+1469 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
z=np.load('/work/inputs/reconstruction.npz');A=z['T_world_mesh'][:,:3,:3];F=len(A)
def ang(a,b):return np.linalg.norm(R.from_matrix(a.T@b).as_rotvec())*180/np.pi
print('leave-one-out residual >10')
for i in range(1,F-1):
 rr=R.from_matrix(A[i-1:i+2]); sl=Slerp([0,2],rr)(1).as_matrix(); d=ang(A[i],sl)
 if d>10 and (i<315 or i>368): print(i,round(d,1))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 8, in <module>
  File "scipy/spatial/transform/_rotation.pyx", line 3773, in scipy.spatial.transform._rotation.Slerp.__init__
leave-one-out residual >10
ValueError: Expected number of rotations to be equal to number of timestamps given, got 3 rotations and 2 timestamps.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');A=z['T_world_mesh'][:,:3,:3];F=len(A)
def ang(a,b):return np.linalg.norm(R.from_matrix(a.T@b).as_rotvec())*180/np.pi
print('midpoint residual >10')
for i in range(1,F-1):
 rr=R.from_matrix(A[[i-1,i+1]]); sl=rr[0]*R.from_rotvec(rr[0].inv().apply(rr[1].as_rotvec())/2) # wrong
 # do delta log
 d=R.from_matrix(A[i-1].T@A[i+1]).as_rotvec()/2; sl=R.from_matrix(A[i-1]@R.from_rotvec(d).as_matrix()).as_matrix(); dis=ang(A[i],sl)
 if dis>10 and (i<315 or i>368): print(i,round(dis,1))
PY"
```
<details><summary>output</summary>

```
midpoint residual >10
237 11.2
238 10.3
256 13.0
257 11.2
262 19.0
264 21.8
268 39.4
269 43.3
369 103.1
371 14.4

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
idxs=list(range(250,281));ims=[]
for i in idxs:
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')
 # use object mask bbox center + fixed 180
 import h5py
 im=im[600:830,500:780]; im=cv2.resize(im,(200,164))
 cv2.putText(im,str(i),(3,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3);cv2.putText(im,str(i),(3,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);ims.append(im)
while len(ims)%8:ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/carry_250_280.jpg',np.vstack([np.hstack(ims[j:j+8]) for j in range(0,len(ims),8)]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/make_candidate.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter, medfilt
z=np.load('/work/inputs/reconstruction.npz');T0=z['T_world_mesh'].astype(float);F=len(T0)
p0,p1=0,106;d0,d1=107,378;f0,f1=379,F-1
Rs=T0[:,:3,:3].copy();Rz=np.diag([-1.,-1.,1.])
# per-frame quaternions
def qmean(ids):
 q=R.from_matrix(Rs[ids]).as_quat()
 for _ in range(30):
  m=q.mean(0);m/=np.linalg.norm(m);q=np.where((q@m)[:,None]<0,-q,q)
 m=q.mean(0);return m/np.linalg.norm(m)
def slerp_q(qa,qb,n):
 # t 1..n-1 interior? return n samples from a to b
 a=R.from_quat(qa);b=R.from_quat(qb);d=(a.inv()*b).as_rotvec();return R.from_rotvec(np.outer(np.linspace(0,1,n),d)).as_quat()
# corrected rotation matrices
Rc=Rs.copy()
# final orientation branch: x180/z180 flip correction relative to carry branch
Rf=Rz
Rc[f0:f1+1]=Rs[f0:f1+1]@Rf
qf=qmean(np.arange(f0,f1+1))
# outlier/interruption fill
for a,b,ea,eb in [(262,269,261,270),(324,344,323,345),(369,378,368,None)]:
 qa=R.from_matrix(Rc[ea]).as_quat()
 qb=qf if eb is None else R.from_matrix(Rc[eb]).as_quat()
 Rc[a:b+1]=R.from_quat(slerp_q(qa,qb,b-a+2)[1:-1]).as_matrix()
# smooth dynamic quaternions with SG; static constant
q=R.from_matrix(Rc).as_quat()
for i in range(1,F):
 if q[i]@q[i-1]<0:q[i]*=-1
Qs=q.copy()
for j in range(4):
 Qs[d0-2:d1+3,j]=savgol_filter(q[d0-2:d1+3,j],7,2,mode='interp')
Qs/=np.linalg.norm(Qs,axis=1,keepdims=True)
Rout=R.from_quat(Qs).as …[+892 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/make_candidate.py", line 26, in <module>
    Rc[a:b+1]=R.from_quat(slerp_q(qa,qb,b-a+2)[1:-1]).as_matrix()
    ~~^^^^^^^
ValueError: could not broadcast input array from shape (7,3,3) into shape (8,3,3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/b-a+2)\\[1:-1\\]/b-a+3)[1:-1]/' /work/scratch/make_candidate.py; python /work/scratch/make_candidate.py"
```
<details><summary>output</summary>

```
translation outliers [343 344]
step pct [  0.77252323   7.36947082  13.98406238  34.04815281 122.75469213]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/traj_candidate.npy');W=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def a(i):
 r=T[i-1,:3,:3].T@T[i,:3,:3];return np.linalg.norm(R.from_matrix(r).as_rotvec())*180/np.pi
for i in range(1,len(T)):
 if a(i)>15:print(i,round(a(i),1))
PY"
```
<details><summary>output</summary>

```
261 33.1
262 43.1
263 33.9
264 15.4
269 32.4
270 41.9
271 31.3
323 23.8
324 28.0
325 23.4
344 24.6
345 28.9
346 23.3
367 15.4
368 34.5
369 40.5
370 29.3
372 17.7
376 22.4
378 28.8
379 122.8

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
# exec modified to inspect stages by making a debug copy quickly
s=open('/work/scratch/make_candidate.py').read();
s=s.replace(\"np.save('/work/scratch/traj_candidate.npy',T)\",\"np.save('/work/scratch/traj_candidate.npy',T)\\nnp.save('/work/scratch/Rc.npy',Rc);np.save('/work/scratch/Qs.npy',Qs)\")
open('/work/scratch/make_candidate_dbg.py','w').write(s)
PY
python /work/scratch/make_candidate_dbg.py
python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
for nm in ['/work/scratch/Rc.npy','/work/scratch/Qs.npy']:
 A=np.load(nm);print(nm)
 for i in list(range(258,274))+list(range(365,382)):
  r=A[i-1].T@A[i];a=np.linalg.norm(R.from_matrix(r).as_rotvec())*180/np.pi
  if a>10:print(i,round(a,1))
PY"
```
<details><summary>output</summary>

```
translation outliers [343 344]
step pct [  0.77252323   7.36947082  13.98406238  34.04815281 122.75469213]
/work/scratch/Rc.npy
262 120.8
270 118.1
365 14.9
368 12.5
369 104.1
370 12.3
371 12.3
372 12.3
373 12.3
374 12.3
375 12.3
376 12.3
377 12.3
378 12.3
379 149.6
/work/scratch/Qs.npy
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
  File "scipy/spatial/transform/_rotation.pyx", line 1128, in scipy.spatial.transform._rotation.Rotation.from_matrix
ValueError: Expected `matrix` to have shape (3, 3) or (N, 3, 3), got ()

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
W=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][:,:3,:3];Rz=np.diag([-1,-1,1.])
def a(A,B):return np.linalg.norm(R.from_matrix(A.T@B).as_rotvec())*180/np.pi
for i in range(255,276):print(i,round(a(W[261],W[i]),1),round(a(W[261],W[i]@Rz),1))
PY"
```
<details><summary>output</summary>

```
255 19.4 161.8
256 23.3 157.2
257 7.4 176.7
258 5.5 176.4
259 7.5 176.5
260 5.8 174.9
261 0.0 180.0
262 4.9 175.8
263 28.4 152.7
264 67.7 112.3
265 64.5 115.5
266 65.4 114.6
267 64.7 115.5
268 59.6 120.4
269 24.3 156.3
270 21.0 159.3
271 23.0 157.0
272 29.8 150.3
273 31.1 149.4
274 32.8 147.6
275 29.3 151.8

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
W=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][:,:3,:3]
def slerp_q(qa,qb,n):
 a=R.from_quat(qa);b=R.from_quat(qb);d=(a.inv()*b).as_rotvec();print('d',d,np.linalg.norm(d)*180/np.pi);return R.from_rotvec(np.outer(np.linspace(0,1,n),d)).as_quat()
q=slerp_q(R.from_matrix(W[261]).as_quat(),R.from_matrix(W[270]).as_quat(),10)
A=R.from_quat(q).as_matrix()
for i in range(1,len(A)):
 print(i,np.linalg.norm(R.from_matrix(A[i-1].T@A[i]).as_rotvec())*180/np.pi)
print(np.linalg.norm(R.from_matrix(A[0].T@W[261]).as_rotvec()))
PY"
```
<details><summary>output</summary>

```
d [ 0.01157019  0.06133941 -0.36199791] 21.047047688501504
1 2.3385608542779446
2 2.3385608542779455
3 2.3385608542779446
4 2.3385608542779455
5 2.3385608542779455
6 2.3385608542779446
7 2.3385608542779415
8 2.3385608542779472
9 2.338560854277947
2.084987199802595

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/make_candidate.py';s=open(p).read()
s=s.replace(\"def qmean(ids):\\n q=R.from_matrix(Rs[ids]).as_quat()\", \"def qmean_mats(mats):\\n q=R.from_matrix(mats).as_quat()\")
s=s.replace(\"def slerp_q(qa,qb,n):\\n # t 1..n-1 interior? return n samples from a to b\\n a=R.from_quat(qa);b=R.from_quat(qb);d=(a.inv()*b).as_rotvec();return R.from_rotvec(np.outer(np.linspace(0,1,n),d)).as_quat()\", \"def slerp_q(qa,qb,n):\\n a=R.from_quat(qa);b=R.from_quat(qb);d=(a.inv()*b).as_rotvec();return (a*R.from_rotvec(np.outer(np.linspace(0,1,n),d))).as_quat()\")
s=s.replace(\"qf=qmean(np.arange(f0,f1+1))\", \"qf=qmean_mats(Rc[f0:f1+1])\")
s=s.replace(\"Rout[p0:p1+1]=R.from_quat(qmean(np.arange(p0,p1+1))).as_matrix()\", \"Rout[p0:p1+1]=R.from_quat(qmean_mats(Rs[p0:p1+1])).as_matrix()\")
open(p,'w').write(s)
PY
python /work/scratch/make_candidate.py"
```
<details><summary>output</summary>

```
translation outliers [343 344]
step pct [ 0.72174212  3.80937751  4.7454306   8.07999998 14.54803445]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_traj.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
Tb=W.astype(float); Tc=np.load('/work/scratch/traj_candidate.npy')
ids=np.arange(0,477,2)
from scipy.signal import find_peaks
res=[]
for i in ids:
 box=cropbox(i); x0,y0,s=box
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s]
 Ts=[Tb[i],Tc[i]]; out=[]
 for T in Ts:
  iou,gn,nvis,gg=scores(i,T,cropbox(i),im);out.append((iou,gn,gg,nvis,(OM[i].sum())))
 res.append((i,*out[0],*out[1]))
res=np.array(res)
# columns i bi bgn bgg bnvis area ci cgn cgg cnvis
import json
for a,b,name in [(0,106,'initial'),(107,323,'carry_visible'),(324,344,'occluded'),(345,368,'lower_place'),(369,378,'release'),(379,476,'final')]:
 m=(res[:,0]>=a)&(res[:,0]<=b); valid=m&(res[:,4]>100)
 print(name,'n',m.sum(),'valid',valid.sum())
 for tag,off in [('base',1),('cand',6)]:
  print(' ',tag,'IoU',round(res[m,off].mean(),4),'visIoU',round(res[valid,off].mean(),4),'NCC',round(res[valid,off+1].mean(),4),'Edge',np.nanmean(res[valid,off+2]))
# Hand center distances
H=Kkp=z['human_keypoints_world'];cp=z['contact_prob_right_left']
for name,T in [('base',Tb),('cand',Tc)]:
 c=T[:,:3,3];d=[]
 for i in range(477):d.append(np.linalg.norm(H[i,42:63]-c[i],axis=1).min())
 print(name,'handdist carry median/p90/max',np.percentile(np.array(d)[107:369],[50,90,100]))
np.save('/work/scratch/eval_res.npy',res)
PY
python /work/scratch/eval_traj.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
initial n 54 valid 54
  base IoU 0.7906 visIoU 0.7906 NCC 0.4847 Edge 0.2434558195016965
  cand IoU 0.8044 visIoU 0.8044 NCC 0.5192 Edge 0.125197929531665
carry_visible n 108 valid 107
  base IoU 0.7818 visIoU 0.789 NCC 0.3088 Edge 0.15578964745813495
  cand IoU 0.7858 visIoU 0.7932 NCC 0.3169 Edge 0.16758075095094685
occluded n 11 valid 2
  base IoU 0.0938 visIoU 0.516 NCC -0.1249 Edge 0.0009056538910644091
  cand IoU 0.0777 visIoU 0.4275 NCC 0.0371 Edge 0.15890914474297835
lower_place n 12 valid 12
  base IoU 0.6333 visIoU 0.6333 NCC 0.127 Edge 0.07061754 …[+527 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/scratch/eval_res.npy')
for x in r[(r[:,0]>=340)&(r[:,0]<=380)]:print(int(x[0]),'biou',round(x[1],3),'ciou',round(x[6],3),'bncc',round(x[2],3),'cncc',round(x[7],3))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/trans_test.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
from scipy.signal import savgol_filter,medfilt
Tc=np.load('/work/scratch/traj_candidate.npy')
t0=W[:,:3,3].astype(float);F=len(t0)
# replace gross
t=t0.copy();sp=np.linalg.norm(np.diff(t,axis=0),axis=1);bad=np.where(sp>.2)[0]+1
for i in bad:t[i]=np.nan
for j in range(3):
 ii=np.arange(F);g=~np.isnan(t[:,j]);t[:,j]=np.interp(ii,ii[g],t[g,j])
variants={'raw':t}
for name,w,p in [('m3',3,None),('m5',5,None),('m5s5',5,5),('s5',None,5),('s7',None,7)]:
 x=t.copy()
 for j in range(3):
  if w:x[:,j]=medfilt(x[:,j],w)
  if p:x[:,j]=savgol_filter(x[:,j],p,2,mode='interp')
 variants[name]=x
ids=list(range(340,371,2)); boxcache={i:cropbox(i) for i in ids}
res={k:[] for k in variants}
for i in ids:
 box=boxcache[i];x0,y0,s=box;im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s]
 for k,tv in variants.items():
  T=Tc[i].copy();T[:3,3]=tv[i];res[k].append(scores(i,T,box,im))
for k,v in res.items():
 v=np.array(v);print(k,'iou',round(v[:,0].mean(),3),'ncc',round(v[:,1].mean(),3),'edge',np.nanmean(v[:,3]))
PY
python /work/scratch/trans_test.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
raw iou 0.604 ncc 0.097 edge 0.08376143274958578
m3 iou 0.601 ncc 0.105 edge 0.09554780074894914
m5 iou 0.601 ncc 0.115 edge 0.12192589382919206
m5s5 iou 0.553 ncc 0.119 edge 0.10016960183215601
s5 iou 0.544 ncc 0.114 edge 0.09279223008841389
s7 iou 0.557 ncc 0.116 edge 0.10586021528476955

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/rot_test.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
from scipy.signal import savgol_filter,medfilt
# recreate intermediate matrix pre-smoothing from candidate generator by exec until smoothing? easier rerun with debug saved Rc
exec(open('/work/scratch/make_candidate.py').read())
# make translations m5
Tc=np.load('/work/scratch/traj_candidate.npy');t=W[:,:3,3].astype(float);F=len(t);sp=np.linalg.norm(np.diff(t,axis=0),axis=1);bad=np.where(sp>.2)[0]+1
for i in bad:t[i]=np.nan
for j in range(3):
 ii=np.arange(F);g=~np.isnan(t[:,j]);t[:,j]=np.interp(ii,ii[g],t[g,j]);t[:,j]=medfilt(t[:,j],5)
# candidate smoothed in Tc, and raw corrected Rc? make_candidate local Rc save debug wasn't saved; rebuild not simple. use smoothed as base.
q=R.from_matrix(Tc[:,:3,:3]).as_quat()
variants={'sg7':Tc[:,:3,:3]}
for name,w,p in [('med3',3,None),('med5',5,None),('sg5',None,5),('sg9',None,9)]:
 Q=q.copy()
 for j in range(4):
  if w:Q[:,j]=medfilt(Q[:,j],w)
  if p:Q[:,j]=savgol_filter(Q[:,j],p,2,mode='interp')
 Q/=np.linalg.norm(Q,axis=1,keepdims=True);variants[name]=R.from_quat(Q).as_matrix()
ids=list(range(340,371,2));boxcache={i:cropbox(i) for i in ids};res={k:[] for k in variants}
for i in ids:
 box=boxcache[i];x0,y0,s=box;im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s]
 for k,Rt in variants.items():
  T=np.eye(4);T[:3,:3]=Rt[i];T[:3,3]=t[i];res[k].append(scores(i,T,box,im))
for k,v in r …[+168 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
translation outliers [343 344]
step pct [ 0.72174212  3.80937751  4.7454306   8.07999998 14.54803445]
sg7 iou 0.601 ncc 0.115 edge 0.12192589382919206
med3 iou 0.6 ncc 0.115 edge 0.1200255464866728
med5 iou 0.601 ncc 0.12 edge 0.12285379140125033
sg5 iou 0.602 ncc 0.117 edge 0.12086818632991511
sg9 iou 0.601 ncc 0.116 edge 0.12087337633525269

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p) as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  x=f[k];print(k,type(x),x.shape,x.dtype)
  for i in [0,100,300,400,470]:
   if isinstance(x,h5py.Group): print(i,list(x.keys())[:3],x[str(i)].shape,x[str(i)].dtype) if str(i) in x else None
   else: print(i,x[i].shape,x[i].dtype)
 print('align',f['alignment/front_stereo_camera_left/scale'][:3],f['alignment/front_stereo_camera_left/shift'][:3])
PY"
```
<details><summary>output</summary>

```
aligned/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (477,) object
0 (568041,) uint8
100 (562981,) uint8
300 (572276,) uint8
400 (568781,) uint8
470 (555808,) uint8
raw/front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (477,) object
0 (498228,) uint8
100 (499728,) uint8
300 (509045,) uint8
400 (499192,) uint8
470 (502319,) uint8
align [1.1956384 1.190106  1.19718  ] [0. 0. 0.]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 for k in ['aligned/front_stereo_camera_left','raw/front_stereo_camera_left']:
  d=f[k]
  for i in [0]:
   print(k,d[i].attrs, int(d[i].shape[0]),d[i][:20])
  print('ds attrs',dict(d.attrs));print('file attrs',dict(f.attrs))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 6, in <module>
AttributeError: 'numpy.ndarray' object has no attribute 'attrs'

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"aligned/front_stereo_camera_left\\|compression.*aligned\\|aligned.h5\" -n /code/reconstruction/modules/v2d_cari4d /code/reconstruction/modules/v2d_moge | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/modules/v2d_cari4d/lib/run_inference.py:116:    aligned_depth = output_root / "depth/moge2/aligned.h5"

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"alignment/.*scale\\|front_stereo_camera_left\" -n /code/reconstruction /workspace 2>/dev/null | head -100"
```
<details><summary>output</summary>

```
/code/reconstruction/workflows/mv_hoi/orchestration/campaign_controller.py:2904:        "back_stereo_camera_left", "front_stereo_camera_left",
/code/reconstruction/workflows/mv_hoi/orchestration/campaign_controller.py:3118:            "back_stereo_camera_left", "front_stereo_camera_left",
/code/reconstruction/workflows/mv_hoi/orchestration/campaign_inventory.py:39:    "back_stereo_camera_left", "front_stereo_camera_left",
/code/reconstruction/workflows/mv_hoi/orchestration/submit.py:34:    "front_stereo_camera_left",
/code/reconstruction/workflows/mv_hoi/orchestration/submit.py:1050:                "front_stereo_camera_left.json",
/code/reconstruction/workflows/mv_hoi/tests/test_campaign_controller.py:1575:            "back_stereo_camera_left", "front_stereo_camera_left",
/code/reconstruct …[+14983 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
x=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r'); print(x.shape,x.dtype)
for i in [0,100,400,470]:
 a=x[i]; print(i,a.min(),a.max(),np.percentile(a,[1,50,99]),a[800,740],a[840,738])
PY"
```
<details><summary>output</summary>

```
(477, 1152, 1536) uint16
0 1555 7207 [1643. 3440. 6651.] 3140 3060
100 1552 7260 [1635. 3444. 6666.] 3164 3071
400 1601 7369 [1687. 3529. 6903.] 2809 2806
470 1581 7285 [1670. 3444. 6811.] 2786 2766

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py
D=np.load('/pipeline/depth/moge2/raw.npy',mmap_mode='r')
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][:]
for i in [0,10,50,100,110,150,200,240,260,300,320,345,350,360,370,400,430,470]:
 m=o[i]
 if m.sum():
  vals=D[i][m];print(i,'area',m.sum(),'depth median',np.median(vals)/1000,'p10/90',np.percentile(vals,[10,90])/1000)
PY"
```
<details><summary>output</summary>

```
0 area 2049 depth median 2.654 p10/90 [2.628  2.9718]
10 area 1984 depth median 2.678 p10/90 [2.656  2.8557]
50 area 2057 depth median 2.605 p10/90 [2.571  2.8898]
100 area 2003 depth median 2.534 p10/90 [2.514  2.7002]
110 area 1946 depth median 2.541 p10/90 [2.524  2.6095]
150 area 1865 depth median 2.582 p10/90 [2.559 2.635]
200 area 2341 depth median 2.448 p10/90 [2.387 2.963]
240 area 4308 depth median 1.778 p10/90 [1.731  1.8293]
260 area 3910 depth median 2.11 p10/90 [1.9859 2.182 ]
300 area 2410 depth median 2.131 p10/90 [2.116  2.3493]
320 area 741 depth median 2.311 p10/90 [2.167 5.123]
345 area 1214 depth median 2.663 p10/90 [2.612  3.3054]
350 area 1189 depth median 3.1 p10/90 [3.049  3.3272]
360 area 1392 depth median 2.489 p10/90 [2.437 2.877]
370 area 1371 depth median 2.809 …[+188 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,zlib,gzip
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 x=f['aligned/front_stereo_camera_left'][0]
 print(len(x),x[:50],x[:8])
 for w in [15,-15,31,-31]:
  try:
   y=zlib.decompress(x,w);print('zlib',w,len(y),y[:20])
  except Exception as e:pass
 try:
  y=np.frombuffer(zlib.decompress(x),np.float32);print('float',y.shape,y[:10])
 except Exception as e:print('err',e)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np,h5py
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 b=f['aligned/front_stereo_camera_left'][0];a=cv2.imdecode(b,cv2.IMREAD_UNCHANGED);print(a.shape,a.dtype,a.min(),a.max(),np.percentile(a,[1,50,99]))
 b=f['raw/front_stereo_camera_left'][0];r=cv2.imdecode(b,cv2.IMREAD_UNCHANGED);print(r.shape,r.dtype,r.min(),r.max(),np.percentile(r,[1,50,99]))
 print('ratio',np.percentile(a.astype(float)/np.maximum(r,1),[1,50,99]))
PY"
```
<details><summary>output</summary>

```
(1152, 1536) uint16 1859 8616 [1964. 4112. 7952.]
(1152, 1536) uint16 1555 7207 [1643. 3440. 6651.]
ratio [1.19512195 1.19549812 1.19563823]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,h5py,cv2
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];T=np.load('/work/scratch/traj_candidate.npy')
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']);Tcw=np.linalg.inv(Twc)
with h5py.File('/pipeline/depth/moge2/aligned.h5') as f:
 def ad(i):return cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED)
 for i in [0,50,100,200,240,300,350,400,470]:
  cc=Tcw[:3,:3]@T[i,:3,3]+Tcw[:3,3]
  print(i,'camcenter',np.round(cc,3),'depth',round(cc[2],3))
PY"
```
<details><summary>output</summary>

```
0 camcenter [-0.118  0.168  2.908] depth 2.908
50 camcenter [-0.118  0.168  2.908] depth 2.908
100 camcenter [-0.118  0.168  2.908] depth 2.908
200 camcenter [0.925 0.342 2.768] depth 2.768
240 camcenter [0.436 0.457 1.973] depth 1.973
300 camcenter [-0.823  0.484  2.523] depth 2.523
350 camcenter [-0.363  0.887  3.243] depth 3.243
400 camcenter [-0.066  1.098  3.169] depth 3.169
470 camcenter [-0.066  1.098  3.169] depth 3.169

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');W=z['T_world_mesh'];C=z['T_cam_mesh_coconet'];F=z['T_cam_mesh_foundationpose']
for i in [0,100,200,240,300,350,400,470]:
 print(i,'Wcam??','Cz',C[i,2,3],'Fz',F[i,2,3],'Cxy',C[i,:2,3],'Wc',W[i,:3,3])
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
T=np.load('/work/scratch/traj_candidate.npy');W=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];gp=json.load(open('/work/inputs/ground_plane.json'));n=np.array(gp['plane'][:3]);d=gp['plane'][3]
for i in [0,240,350,400,470]:
 for nm,A in [('base',W),('cand',T)]:
  R=A[i,:3,:3];c=A[i,:3,3]
  # mesh bounds along z: world positions c + R[:,2]*[-.1303,.1669]
  ends=[c+R[:,2]*z for z in [-.1303,.1669,0]]
  dist=[n@p+d for p in ends]
  print(i,nm,'center',np.round(c,3),'zaxis',np.round(R[:,2],2),'end dist',np.round(dist,3))
PY"
```
<details><summary>output</summary>

```
0 base center [0.167 0.197 3.119] zaxis [-0.05 -1.   -0.04] end dist [0.885 1.182 1.015]
0 cand center [0.17  0.177 3.249] zaxis [-0.02 -1.   -0.06] end dist [0.898 1.195 1.028]
240 base center [-0.336  0.357  4.242] zaxis [-0.2  -0.98 -0.04] end dist [0.675 0.967 0.803]
240 cand center [-0.339  0.357  4.235] zaxis [-0.2  -0.98 -0.02] end dist [0.675 0.967 0.803]
350 base center [0.382 0.923 2.982] zaxis [-0.01 -0.99 -0.16] end dist [0.164 0.459 0.293]
350 cand center [0.399 0.929 2.985] zaxis [ 0.03 -0.97 -0.23] end dist [0.159 0.451 0.287]
400 base center [0.108 1.131 3.079] zaxis [-0.03 -0.99 -0.13] end dist [-0.044  0.252  0.086]
400 cand center [0.106 1.129 3.096] zaxis [-0.01 -0.99 -0.1 ] end dist [-0.043  0.254  0.087]
470 base center [0.107 1.132 3.095] zaxis [ 0.01 -1.   -0.08] en …[+124 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][470];p=f['person'][470]
im=cv2.imread('/work/scratch/frames/471.jpg'); crop=im[770:910,700:780].copy(); oc=o[770:910,700:780]
ov=crop.copy();ov[oc]=(0,0,255)
cv2.imwrite('/work/scratch/final_mask.png',cv2.resize(np.hstack([crop,ov]),None,fx=5,fy=5,interpolation=cv2.INTER_NEAREST))
print('mask region colors',crop[oc].mean(0))
PY"
```
<details><summary>output</summary>

```
[ WARN:0@0.097] global loadsave.cpp:268 findDecoder imread_('/work/scratch/frames/471.jpg'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
TypeError: 'NoneType' object is not subscriptable

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls /work/scratch/frames | tail; ls /work/scratch/frames/0471.jpg'
```
<details><summary>output</summary>

```
0468.jpg
0469.jpg
0470.jpg
0471.jpg
0472.jpg
0473.jpg
0474.jpg
0475.jpg
0476.jpg
0477.jpg
/work/scratch/frames/0471.jpg

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,h5py,numpy as np
with h5py.File('/work/inputs/masks.h5') as f:o=f['object'][470]
im=cv2.imread('/work/scratch/frames/0471.jpg'); crop=im[770:910,700:780].copy(); oc=o[770:910,700:780]
ov=crop.copy();ov[oc]=(0,0,255)
cv2.imwrite('/work/scratch/final_mask.png',cv2.resize(np.hstack([crop,ov]),None,fx=5,fy=5,interpolation=cv2.INTER_NEAREST))
print('mask region colors',crop[oc].mean(0))
PY"
```
<details><summary>output</summary>

```
mask region colors [103.9179911  102.78766688  94.78067387]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh,numpy as np,torch
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
S,T=trimesh.bounds.oriented_bounds(m);print('S',S)
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False)
C=np.load('/work/inputs/reconstruction.npz')['T_cam_mesh_coconet']
for key in ['pr','pr_initial','in']:
 for pk in ['pose_abs','pose_abs_postopt']:
  if pk in x[key]:
   P=np.array(x[key][pk]);P=P@S
   # rotation error to C
   from scipy.spatial.transform import Rotation as R
   rr=np.einsum('nji,njk->nik',C[:,:3,:3],P[:,:3,:3]);a=np.linalg.norm(R.from_matrix(rr).as_rotvec(),axis=1)*180/np.pi
   dt=np.linalg.norm(C[:,:3,3]-P[:,:3,3],axis=1)
   print(key,pk,'rot',np.percentile(a,[0,50,90,100]),'trans',np.percentile(dt,[0,50,90,100]))
PY"
```
<details><summary>output</summary>

```
S [[ 0.95957785 -0.28091073 -0.01730673 -0.00122253]
 [ 0.2807624   0.9597196  -0.01052462 -0.00135753]
 [ 0.01956608  0.00524012  0.99979483 -0.01803285]
 [ 0.          0.          0.          1.        ]]
pr pose_abs rot [1.03542673e-08 2.19280313e-07 6.54732619e-07 1.26062114e-06] trans [0.00046328 0.00544589 0.01505497 0.24085063]
pr pose_abs_postopt rot [1.03542673e-08 2.19280313e-07 6.54732619e-07 1.26062114e-06] trans [0.00046328 0.00544589 0.01505497 0.24085063]
pr_initial pose_abs rot [1.03542673e-08 2.19280313e-07 6.54732619e-07 1.26062114e-06] trans [1.01513627e-09 5.79974540e-08 1.05982504e-07 1.25508493e-07]
in pose_abs rot [  0.56813535   5.19003449  18.19170842 101.18013459] trans [0.00677466 0.12312848 0.30836375 2.18940744]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,trimesh,torch
from scipy.spatial.transform import Rotation as R
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();S,_=trimesh.bounds.oriented_bounds(m)
z=np.load('/work/inputs/reconstruction.npz');W=z['T_world_mesh'];F=z['T_cam_mesh_foundationpose'];C=z['T_cam_mesh_coconet']; A=W[0,:3,:3]@C[0,:3,:3].T
x=torch.load('/pipeline/inference/refined.pth',map_location='cpu',weights_only=False);I=np.array(x['in']['pose_abs'])@S
# in rot to world via A
Irot=np.einsum('ij,njk->nik',A,I[:,:3,:3])
Rz=np.diag([-1,-1,1.]);ref=W[320,:3,:3]
def a(A,B):return np.linalg.norm(R.from_matrix(A.T@B).as_rotvec())*180/np.pi
for i in range(315,369,2):
 print(i,'base',round(a(ref,W[i,:3,:3]),1),'float',round(a(ref,Irot[i]),1),'fp',round(a(ref,F[i,:3,:3]),1))
# step stats methods
for n,X in [('base',W[:,:3,:3]),('in',Irot),('fp',F[:,:3,:3])]:
 ss=[]
 for i in range(315,369):ss.append(a(X[i-1],X[i]))
 print(n,'steps',np.round(np.percentile(ss,[50,90,95,100]),1),[(i+315,round(v,1)) for i,v in enumerate(ss) if v>40][:20])
PY"
```
<details><summary>output</summary>

```
315 base 19.9 float 22.3 fp 156.3
317 base 10.1 float 14.0 fp 165.9
319 base 9.0 float 11.9 fp 169.2
321 base 8.4 float 14.1 fp 169.7
323 base 9.9 float 17.1 fp 165.3
325 base 56.4 float 136.3 fp 123.3
327 base 175.3 float 110.5 fp 136.2
329 base 179.3 float 113.1 fp 127.1
331 base 167.1 float 153.3 fp 104.4
333 base 83.8 float 110.8 fp 136.1
335 base 106.9 float 111.6 fp 130.3
337 base 161.2 float 91.2 fp 125.3
339 base 120.0 float 179.6 fp 88.8
341 base 118.6 float 175.8 fp 87.1
343 base 151.1 float 171.4 fp 94.6
345 base 25.8 float 16.5 fp 168.1
347 base 28.5 float 22.9 fp 163.5
349 base 43.5 float 39.6 fp 162.7
351 base 30.4 float 24.2 fp 156.5
353 base 39.4 float 21.1 fp 157.5
355 base 37.2 float 26.2 fp 151.6
357 base 18.4 float 23.8 fp 179.9
359 base 19.9 float 41.4 fp 178.9
361 bas …[+674 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_variants.py <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter, medfilt
z=np.load('/work/inputs/reconstruction.npz');T0=z['T_world_mesh'].astype(float);F=len(T0)
def qmean_mats(mats):
 q=R.from_matrix(mats).as_quat()
 for _ in range(30):
  m=q.mean(0);m/=np.linalg.norm(m);q=np.where((q@m)[:,None]<0,-q,q)
 m=q.mean(0);return m/np.linalg.norm(m)
def slerp_q(qa,qb,n):
 a=R.from_quat(qa);b=R.from_quat(qb);d=(a.inv()*b).as_rotvec();return (a*R.from_rotvec(np.outer(np.linspace(0,1,n),d))).as_quat()
def build(win):
 p0,p1=0,106;d0,d1=107,378;f0,f1=379,F-1
 Rs=T0[:,:3,:3].copy();Rz=np.diag([-1.,-1.,1.]);Rc=Rs.copy();Rc[f0:f1+1]=Rs[f0:f1+1]@Rz
 qf=qmean_mats(Rc[f0:f1+1])
 for a,b,ea,eb in [(262,269,261,270),(324,344,323,345),(369,378,368,None)]:
  qa=R.from_matrix(Rc[ea]).as_quat();qb=qf if eb is None else R.from_matrix(Rc[eb]).as_quat()
  Rc[a:b+1]=R.from_quat(slerp_q(qa,qb,b-a+3)[1:-1]).as_matrix()
 q=R.from_matrix(Rc).as_quat()
 for i in range(1,F):
  if q[i]@q[i-1]<0:q[i]*=-1
 Q=q.copy()
 for j in range(4):Q[d0-2:d1+3,j]=savgol_filter(q[d0-2:d1+3,j],win,2,mode='interp')
 Q/=np.linalg.norm(Q,axis=1,keepdims=True);Rout=R.from_quat(Q).as_matrix()
 Rout[p0:p1+1]=R.from_quat(qmean_mats(Rs[p0:p1+1])).as_matrix();Rout[f0:f1+1]=R.from_quat(qf).as_matrix()
 t=T0[:,:3,3].copy();sp=np.linalg.norm(np.diff(t,axis=0),axis=1);bad=np.where(sp>.2)[0]+1
 for i in bad:t[i]=np.nan
 for j in ran …[+374 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_variants.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
W=W.astype(float);Ts={w:np.load(f'/work/scratch/traj_w{w}.npy') for w in [3,5,7,9]}
ids=sorted(set(list(range(0,477,4))+list(range(315,380))))
def render_batch(i,Ts,box):
 x0,y0,s=box;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0
 Tb=torch.tensor(np.stack([Tcw@T for T in Ts]),device='cuda',dtype=torch.float)
 cols,deps,_=nvdiffrast_render(K=Kc,H=s,W=s,ob_in_cams=Tb,glctx=gl,mesh_tensors=mt,output_size=(s,s),use_light=False,extra={})
 return cols.detach().cpu().numpy(),deps.detach().cpu().numpy()
def metric(i,pairs,box):
 x0,y0,s=box;im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s]; x=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255
 cols,deps=render_batch(i,[T for n,T in pairs],box);om=OM[i,y0:y0+s,x0:x0+s];pm=PM[i,y0:y0+s,x0:x0+s];out=[]
 for rgb,dep in zip(cols,deps):
  rr=np.clip(rgb,0,1);rm=dep>0.01;vis=om&~pm&rm;iou=(rm&om).sum()/max(1,(rm|om).sum())
  if vis.sum()<10:out.append((iou,0,0));continue
  a=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)[vis];b=cv2.cvtColor(rr,cv2.COLOR_RGB2GRAY)[vis]
  gn=np.corrcoef(a,b)[0,1] if a.std()>1e-5 and b.std()>1e-5 else 0
  g1=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)[vis]+cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)[vis]
  g2=cv2.Sobel(rr,cv2.CV_32F,1,0,ksize=3)[vis]+cv2.Sobel(rr,cv2.CV_32F,0,1,ksize=3)[vis]
  gg=np.corrcoef(g1,g2)[0,1] if g1.std()>1e-5 and g2.std()>1e-5 else 0
  out.append((iou …[+943 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
initial frames 27 valid 27
  base IoU 0.7919 vis 0.7919 NCC 0.5187 edge 0.8883
  w3 IoU 0.8061 vis 0.8061 NCC 0.5767 edge 0.493
  w5 IoU 0.8061 vis 0.8061 NCC 0.5767 edge 0.493
  w7 IoU 0.8061 vis 0.8061 NCC 0.5767 edge 0.493
  w9 IoU 0.8061 vis 0.8061 NCC 0.5767 edge 0.493
carry_visible frames 61 valid 59
  base IoU 0.7369 vis 0.7618 NCC 0.3141 edge 0.634
  w3 IoU 0.7386 vis 0.7636 NCC 0.3229 edge 0.7234
  w5 IoU 0.7381 vis 0.7631 NCC 0.3242 edge 0.7676
  w7 IoU 0.7381 vis 0.763 NCC 0.3237 edge 0.7505
  w9 IoU 0.7383 vis 0.7633 NCC 0.3231 edge 0.7476
occlu …[+1110 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_fit.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
Tc=np.load('/work/scratch/traj_w5.npy');W=W.astype(float);F=len(W)
# majority masks
for a,b,name,T in [(0,106,'init',None),(379,476,'final',Tc)]:
 M=OM[a:b+1].mean(0)>=.5
 ys,xs=np.where(M);print(name,'majority area',M.sum(),'bbox',xs.min(),ys.min(),xs.max(),ys.max())
 # render candidate static using full crop and get mask bbox
 # choose crop around majority
 cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;s=180;x0=int(cx-s/2);y0=int(cy-s/2)
 box=(x0,y0,s)
 Tr=Tc if T is not None else np.load('/work/scratch/traj_w5.npy')
 for nm,TT in [('baseavg',None)]:pass
 # evaluate candidate and base first frame
 for nm,TT in [('base',W[a]),('cand',Tr[a])]:
  rgb,dep=render_pose(a,TT,box);rm=dep>.01
  yy,xx=np.where(rm); print(' ',nm,'bboxlocal',xx.min(),yy.min(),xx.max(),yy.max(),'iou', (rm&M[y0:y0+s,x0:x0+s]).sum()/(rm|M[y0:y0+s,x0:x0+s]).sum())
PY
python /work/scratch/static_fit.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
init majority area 1991 bbox 710 559 736 645
  base bboxlocal 79 51 106 131 iou 0.7308255984367367
  cand bboxlocal 79 49 105 133 iou 0.8121927236971485
final majority area 1567 bbox 726 798 747 882
  base bboxlocal 83 54 106 133 iou 0.6648351648351648
  cand bboxlocal 83 50 105 131 iou 0.693900391717963

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
idxs=list(range(360,380));ims=[]
for i in idxs:
 im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[750:900,690:780]
 im=cv2.resize(im,(270,450),interpolation=cv2.INTER_CUBIC)
 cv2.putText(im,str(i),(3,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3);cv2.putText(im,str(i),(3,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);ims.append(im)
cv2.imwrite('/work/scratch/release_zoom.jpg',np.hstack(ims))
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
z=np.load('/work/inputs/reconstruction.npz');O=z['T_world_mesh'][:,:3,:3];K=z['human_keypoints_world'];T=z['T_world_mesh'][:,:3,3]
# Kabsch hand orientation relative ref, using all 21 points
ref=K[160,42:63];refc=ref-ref.mean(0)
H=[]
for i in range(477):
 p=K[i,42:63];pc=p-p.mean(0);U,S,Vt=np.linalg.svd(pc.T@refc);Rd=Vt.T@U.T
 if np.linalg.det(Rd)<0:Vt[-1]*=-1;Rd=Vt.T@U.T
 H.append(Rd)
H=np.array(H)
# relative O wrt hand
rel=np.einsum('nji,njk->nik',H,O)
def aa(A):return np.linalg.norm(R.from_matrix(A).as_rotvec())*180/np.pi
for i in range(100,370,10):print(i,round(aa(H[i-1].T@H[i]),1),round(aa(rel[i-1].T@rel[i]),1),np.round(R.from_matrix(rel[i]).as_rotvec()*180/np.pi,0))
PY"
```
<details><summary>output</summary>

```
100 2.5 7.4 [ 91.  93. -50.]
110 1.2 2.9 [ 84.  93. -69.]
120 3.9 5.1 [ 79.  96. -79.]
130 6.1 9.6 [ 98.  70. -62.]
140 3.9 3.0 [102.  42. -38.]
150 6.1 15.3 [ 98. -34.  20.]
160 4.8 5.3 [ 65. -67.  68.]
170 3.5 4.1 [ 39. -83.  72.]
180 4.1 3.4 [ 39. -76.  82.]
190 1.4 3.9 [ 29. -80.  93.]
200 1.8 1.4 [ 18. -87.  96.]
210 2.8 2.7 [ 20. -85.  96.]
220 2.0 1.8 [ 34. -72.  99.]
230 3.7 12.6 [ 26. -63.  94.]
240 8.8 11.6 [ 39. -43.  58.]
250 6.1 6.2 [ 39. -34.  43.]
260 5.4 12.1 [54.  0. 12.]
270 6.0 9.3 [59. 21. -9.]
280 1.6 2.3 [ 61.  44. -21.]
290 7.0 6.4 [ 71.  71. -28.]
300 4.0 5.8 [ 49. 116. -50.]
310 1.3 5.3 [  -3. -161.   66.]
320 5.1 12.5 [  41. -130.   41.]
330 3.8 71.4 [-65.  27. -26.]
340 6.4 58.3 [ 72.  77. -86.]
350 11.1 41.1 [ 82. -92.  20.]
360 8.1 9.0 [ 99. -82.  24.]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz');O=z['T_world_mesh'][:,:3,:3];K=z['human_keypoints_world'];ref=K[160,42:63];rc=ref-ref.mean(0);H=[]
for i in range(477):
 p=K[i,42:63];pc=p-p.mean(0);U,S,Vt=np.linalg.svd(pc.T@rc);Rd=Vt.T@U.T
 if np.linalg.det(Rd)<0:Vt[-1]*=-1;Rd=Vt.T@U.T
 H.append(Rd)
H=np.array(H);rel=np.einsum('nji,njk->nik',H,O)
for i in list(range(315,369)):
 print(i,np.round(R.from_matrix(rel[i]).as_rotvec()*180/np.pi,0))
PY"
```
<details><summary>output</summary>

```
315 [  24. -142.   39.]
316 [  30. -140.   41.]
317 [  35. -137.   31.]
318 [  33. -133.   43.]
319 [  42. -119.   51.]
320 [  41. -130.   41.]
321 [  35. -135.   46.]
322 [  32. -137.   58.]
323 [  44. -130.   46.]
324 [  10.  142. -102.]
325 [ -11. -150.   97.]
326 [ -63.  -49. -122.]
327 [-39.  56.  55.]
328 [42. 10. 56.]
329 [-30.  50.  37.]
330 [-65.  27. -26.]
331 [-64.  26. -33.]
332 [-58.  41. -33.]
333 [103. -42.   7.]
334 [125. -36. -18.]
335 [131. -44. -28.]
336 [  -5. -133.   42.]
337 [ -65.  -33. -122.]
338 [ -56.  -40. -110.]
339 [-104. -120.   71.]
340 [ 72.  77. -86.]
341 [ -95. -124.   78.]
342 [134.  66. -95.]
343 [-114. -107.   67.]
344 [ -86. -106.  116.]
345 [ 50. -94.  36.]
346 [  62. -100.   27.]
347 [  51. -103.   40.]
348 [ 45. -92.  47.]
349 [ 45. -98.  53.]
350 [ …[+409 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/hand_interp_test.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
from scipy.spatial.transform import Rotation as R
W=W.astype(float);Tc=np.load('/work/scratch/traj_w5.npy')
# hand orientations
ref=z['human_keypoints_world'][160,42:63];rc=ref-ref.mean(0);H=[]
for i in range(477):
 p=z['human_keypoints_world'][i,42:63];pc=p-p.mean(0);U,S,Vt=np.linalg.svd(pc.T@rc);Rd=Vt.T@U.T
 if np.linalg.det(Rd)<0:Vt[-1]*=-1;Rd=Vt.T@U.T
 H.append(Rd)
H=np.array(H)
# relative at endpoints
ra=H[323].T@Tc[323,:3,:3];rb=H[345].T@Tc[345,:3,:3];d=R.from_matrix(ra.T@rb).as_rotvec()
for i in range(324,345):
 q=R.from_rotvec(d*((i-323)/(345-323))).as_matrix();Tc[i,:3,:3]=H[i]@ra@q
np.save('/work/scratch/traj_hand.npy',Tc)
# compare angles and scores at 338-344, 345 transition
def a(A,B):return np.linalg.norm(R.from_matrix(A.T@B).as_rotvec())*180/np.pi
for i in range(338,346): print('angle',i,round(a(np.load('/work/scratch/traj_w5.npy')[i,:3,:3],Tc[i,:3,:3]),1))
Td=np.load('/work/scratch/traj_w5.npy');ids=list(range(336,347))
for nm,T in [('direct',Td),('hand',Tc)]:
 out=[]
 for i in ids:
  box=cropbox(i);x0,y0,s=box;im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s]
  out.append(scores(i,T[i],box,im))
 out=np.array(out);print(nm,'iou',round(out[:,0].mean(),3),'ncc',round(out[:,1].mean(),3),'edge',np.nanmean(out[:,3]))
 for i,x in zip(ids,out):print(i,round(x[0],2),round(x[1],2))
PY
python /work/scratch …[+33 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
angle 338 27.4
angle 339 30.6
angle 340 32.6
angle 341 27.9
angle 342 25.9
angle 343 9.0
angle 344 6.1
angle 345 0.0
direct iou 0.273 ncc -0.026 edge 0.006006686432781192
336 0.0 0.0
337 0.0 0.0
338 0.0 0.0
339 0.0 0.0
340 0.0 0.0
341 0.0 0.0
342 0.59 -0.01
343 0.77 -0.08
344 0.62 -0.07
345 0.51 -0.07
346 0.51 -0.05
hand iou 0.265 ncc -0.007 edge 0.0012712799286132672
336 0.0 0.0
337 0.0 0.0
338 0.0 0.0
339 0.0 0.0
340 0.0 0.0
341 0.0 0.0
342 0.55 0.15
343 0.74 -0.07
344 0.61 -0.04
345 0.51 -0.07
346 0.51 -0.05

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
for w in [3,5,7,9]:
 T=np.load(f'/work/scratch/traj_w{w}.npy');print(w,T[0,:3,3],T[470,:3,3])
PY"
```
<details><summary>output</summary>

```
 [0.1055852  1.13040078 3.09397316]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_opt.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
import json
cam=json.load(open('/work/inputs/camera.json'));Kreal=np.array(cam['K']);Twc=np.array(cam['T_world_cam']);F=len(W)
T=np.load('/work/scratch/traj_w5.npy').astype(float)
# majority masks
maj={}
for a,b,name in [(0,106,'init'),(379,476,'final')]:maj[name]=OM[a:b+1].mean(0)>=.5
def projuv(TT,pts):
 cc=pts@(Tcw@TT)[:3,:3].T+(Tcw@TT)[:3,3];u=cc@K.T;u=u[:,:2]/u[:,2:];return u
def fit(name,T0i,amp=0.10):
 a,b=({'init':(0,106),'final':(379,476)})[name];M=maj[name]
 ys,xs=np.where(M);cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;sz=180;x0=int(cx-sz/2);y0=int(cy-sz/2);box=(x0,y0,sz);Mt=M[y0:y0+sz,x0:x0+sz]
 base=T[T0i].copy();cc=Tcw[:3,:3]@base[:3,3]+Tcw[:3,3];zc=cc[2]
 ks=np.arange(1-amp,1+amp+1e-6,.01);Ts=[]
 for k in ks:
  TT=base.copy();TT[:3,3]+=Twc[:3,2]*zc*(1/k-1);Ts.append(TT)
 x0,y0,s=box;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;Tb=torch.tensor(np.stack([Tcw@TT for TT in Ts]),device='cuda',dtype=torch.float)
 cols,deps,_=nvdiffrast_render(K=Kc,H=s,W=s,ob_in_cams=Tb,glctx=gl,mesh_tensors=mt,output_size=(s,s),use_light=False,extra={})
 rms=deps.detach().cpu().numpy()>.01
 best=(-1,None)
 for j,k in enumerate(ks):
  for dy in range(-6,7):
   for dx in range(-6,7):
    rm=np.roll(np.roll(rms[j],dy,0),dx,1);iou=(rm&Mt).sum()/max(1,(rm|Mt).sum())
    if iou>best[0]:best=(iou,(float(k),dx,dy))
 print(name,'base iou',(rms[10]&Mt).sum() …[+397 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
init base iou 0.8121927236971485 best (0.8786069651741294, (1.0300000000000002, 0, -2))
final base iou 0.693900391717963 best (0.82031708749266, (1.02, -2, -6))
Ti [0.17441955 0.16115361 3.32505332] Tf [0.11628391 1.10051379 3.15275373]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/static_opt_eval.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
T=np.load('/work/scratch/traj_w5.npy');Ti=np.load('/work/scratch/static_opt_init.npy');Tf=np.load('/work/scratch/static_opt_final.npy')
for name,a,b,To in [('init',0,106,Ti),('final',379,476,Tf)]:
 for tag,TT in [('cand',T),('opt',To if name=='final' else Ti)]:
  out=[]
  for i in range(a,b+1,3):
   # use candidate pos T but override static opt
   U=T[i].copy();U[:3,3]=TT[:3,3];U[:3,:3]=TT[:3,:3]
   out.append(scores(i,U,cropbox(i)))
  out=np.array(out);print(name,tag,'iou',round(out[:,0].mean(),4),'ncc',round(out[:,1].mean(),4),'edge',round(np.nanmean(out[:,3]),4))
PY
python /work/scratch/static_opt_eval.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/static_opt_eval.py 2>&1 | tail -30'
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
Traceback (most recent call last):
  File "/work/scratch/static_opt_eval.py", line 8, in <module>
    U=T[i].copy();U[:3,3]=TT[:3,3];U[:3,:3]=TT[:3,:3]
                  ~^^^^^^
ValueError: could not broadcast input array from shape (3,4) into shape (3,)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
for p in ['/work/scratch/traj_w5.npy','/work/scratch/static_opt_init.npy','/work/scratch/static_opt_final.npy']: print(p,np.load(p).shape)
PY
nl -ba /work/scratch/static_opt_eval.py | head -15"
```
<details><summary>output</summary>

```
     1	exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
     2	T=np.load('/work/scratch/traj_w5.npy');Ti=np.load('/work/scratch/static_opt_init.npy');Tf=np.load('/work/scratch/static_opt_final.npy')
     3	for name,a,b,To in [('init',0,106,Ti),('final',379,476,Tf)]:
     4	 for tag,TT in [('cand',T),('opt',To if name=='final' else Ti)]:
     5	  out=[]
     6	  for i in range(a,b+1,3):
     7	   # use candidate pos T but override static opt
     8	   U=T[i].copy();U[:3,3]=TT[:3,3];U[:3,:3]=TT[:3,:3]
     9	   out.append(scores(i,U,cropbox(i)))
    10	  out=np.array(out);print(name,tag,'iou',round(out[:,0].mean(),4),'ncc',round(out[:,1].mean(),4),'edge',round(np.nanmean(out[:,3]),4))

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/static_opt_eval.py';s=open(p).read();s=s.replace(\"for tag,TT in [('cand',T),('opt',To if name=='final' else Ti)]:\",\"for tag in ['cand','opt']:\").replace(\"U=T[i].copy();U[:3,3]=TT[:3,3];U[:3,:3]=TT[:3,:3]\",\"U=T[i].copy()\\n   TT=T[i] if tag=='cand' else To\\n   U[:3,3]=TT[:3,3];U[:3,:3]=TT[:3,:3]\")
open(p,'w').write(s)
PY
python /work/scratch/static_opt_eval.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
init cand iou 0.8048 ncc 0.5443 edge 0.205
init opt iou 0.8713 ncc 0.6325 edge 0.4753
final cand iou 0.6923 ncc 0.1732 edge 0.2083
final opt iou 0.8145 ncc 0.191 edge -0.076

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/scratch/traj_w5.npy');T[0:107]=np.load('/work/scratch/static_opt_init.npy');T[379:]=np.load('/work/scratch/static_opt_final.npy')
np.save('/work/scratch/traj_finaltmp.npy',T)
for i in [104,105,106,107,108,377,378,379,380]:
 print(i,T[i,:3,3])
def a(i):return np.linalg.norm(R.from_matrix(T[i-1,:3,:3].T@T[i,:3,:3]).as_rotvec())*180/np.pi
for i in range(1,477):
 st=np.linalg.norm(T[i,:3,3]-T[i-1,:3,3])
 if st>.1 or a(i)>20:print('jump',i,'t',round(st,3),'r',round(a(i),1))
PY"
```
<details><summary>output</summary>

```
104 [0.17441955 0.16115361 3.32505332]
105 [0.17441955 0.16115361 3.32505332]
106 [0.17441955 0.16115361 3.32505332]
107 [0.17040145 0.16587226 3.28545213]
108 [0.17005308 0.16587226 3.28545213]
377 [0.10407408 1.15284479 3.05928063]
378 [0.10407408 1.15242863 3.06584597]
379 [0.11628391 1.10051379 3.15275373]
380 [0.11628391 1.10051379 3.15275373]
jump 342 t 0.168 r 1.6
jump 379 t 0.102 r 4.1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/scratch/traj_w5.npy')
for i in range(338,348):print(i,T[i,:3,3],np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]))
PY"
```
<details><summary>output</summary>

```
338 [0.68156105 0.49446607 3.12940216] 0.004088342189788818
339 [0.69511753 0.52285659 3.10881162] 0.0376001417976792
340 [0.68156105 0.56706214 3.10396481] 0.04649086008500605
341 [0.68156105 0.59982902 3.09084249] 0.03529679334284099
342 [0.51989198 0.64396173 3.09084249] 0.16758456378071876
343 [0.5101708  0.69116779 3.08836325] 0.04826034012655644
344 [0.50044962 0.73837386 3.07276169] 0.05065888229641738
345 [0.49072844 0.78557992 3.05716014] 0.050658882296417365
346 [0.47330797 0.82206827 3.02750683] 0.050141715492686444
347 [0.45825484 0.84991294 3.0030365 ] 0.04000898697790216

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');K=z['human_keypoints_world'];H=z['human_vertices_world']
for i in range(335,347):
 hc=K[i,42:63].mean(0); allc=K[i].mean(0)
 print(i,'handmean',np.round(hc,3),'step',round(np.linalg.norm(hc-K[i-1,42:63].mean(0)),3),'body',np.round(allc,3))
PY"
```
<details><summary>output</summary>

```
335 handmean [0.706 0.534 3.123] step 0.135 body [0.59  0.483 3.303]
336 handmean [0.706 0.545 3.11 ] step 0.016 body [0.591 0.488 3.304]
337 handmean [0.697 0.546 3.146] step 0.037 body [0.59  0.485 3.335]
338 handmean [0.691 0.541 3.198] step 0.052 body [0.586 0.473 3.389]
339 handmean [0.703 0.577 3.114] step 0.092 body [0.6   0.501 3.309]
340 handmean [0.685 0.587 3.089] step 0.032 body [0.597 0.503 3.319]
341 handmean [0.655 0.608 3.085] step 0.036 body [0.591 0.512 3.313]
342 handmean [0.647 0.623 3.137] step 0.054 body [0.593 0.52  3.33 ]
343 handmean [0.547 0.632 3.146] step 0.101 body [0.567 0.533 3.327]
344 handmean [0.481 0.649 3.143] step 0.068 body [0.549 0.544 3.338]
345 handmean [0.452 0.68  3.154] step 0.044 body [0.541 0.553 3.351]
346 handmean [0.431 0.694 3.194] step 0.0 …[+28 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');t=z['T_world_mesh'][:,:3,3].astype(float)
from scipy.signal import medfilt,savgol_filter
x=t.copy();
for j in range(3):x[:,j]=savgol_filter(medfilt(x[:,j],5),9,2,mode='interp')
for i in range(335,347):print(i,np.round(x[i],3),round(np.linalg.norm(x[i]-x[i-1]),3))
PY"
```
<details><summary>output</summary>

```
335 [0.679 0.493 3.117] 0.016
336 [0.676 0.493 3.126] 0.009
337 [0.678 0.498 3.13 ] 0.006
338 [0.698 0.508 3.124] 0.023
339 [0.692 0.524 3.115] 0.019
340 [0.664 0.537 3.113] 0.031
341 [0.621 0.575 3.108] 0.058
342 [0.569 0.625 3.097] 0.072
343 [0.52  0.677 3.085] 0.073
344 [0.481 0.731 3.066] 0.069
345 [0.461 0.783 3.044] 0.061
346 [0.472 0.829 3.021] 0.052

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
r=np.load('/work/scratch/eval_variants.npy')
# 0 i,1 area, 2 baseiou,3 base ncc,4 edge, 5 w3iou...
for x in r[(r[:,0]>=336)&(r[:,0]<=348)]:
 print(int(x[0]),'area',int(x[1]),'base',round(x[2],2),'w5iou',round(x[5+3*2],2),'w5ncc',round(x[6+3*2],2))
PY"
```
<details><summary>output</summary>

```
336 area 0 base 0.0 w5iou 0.0 w5ncc 0.0
337 area 0 base 0.0 w5iou 0.0 w5ncc 0.0
338 area 0 base 0.0 w5iou 0.0 w5ncc 0.0
339 area 0 base 0.0 w5iou 0.0 w5ncc 0.0
340 area 0 base 0.0 w5iou 0.0 w5ncc 0.0
341 area 797 base 0.0 w5iou 0.0 w5ncc 0.0
342 area 1428 base 0.51 w5iou 0.59 w5ncc -0.02
343 area 1626 base 0.13 w5iou 0.77 w5ncc -0.07
344 area 1311 base 0.52 w5iou 0.62 w5ncc -0.07
345 area 1214 base 0.51 w5iou 0.51 w5ncc -0.07
346 area 1155 base 0.5 w5iou 0.52 w5ncc -0.09
347 area 1188 base 0.46 w5iou 0.47 w5ncc -0.04
348 area 1129 base 0.36 w5iou 0.29 w5ncc 0.02

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
Tf=np.load('/work/scratch/static_opt_final.npy');gp=json.load(open('/work/inputs/ground_plane.json'));n=np.array(gp['plane'][:3]);d=gp['plane'][3];R=Tf[:3,:3];c=Tf[:3,3]
for z in [-.13027,.1669]:print(z,n@(c+R[:,2]*z)+d)
PY"
```
<details><summary>output</summary>

```
-0.13027 -0.018068125524610235
0.1669 0.2787449693876698

```
</details>

**shell** (exit 127)
```
/usr/bin/bash -c "mkdir -p /work/outputs && cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent world-space spray-bottle trajectory.

The input reconstruction is accurate locally but contains three kinds of
frame-by-frame failures: 180-degree spin-branch flips, short tracking spikes,
and jitter in long static phases.  This script repairs those failures with
SO(3)-aware temporal constraints, static-pose pooling, and a final render/mask
refinement of the two static poses.  It is deterministic and does not use
network access.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.spatial.transform import Rotation as Rot


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def quat_mean(mats: np.ndarray) -> np.ndarray:
    """Sign-invariant mean quaternion for a stack of rotation matrices."""
    q = Rot.from_matrix(mats).as_quat()
    for _ in range(30):
        m = q.mean(axis=0)
        m /= np.linalg.norm(m)
        q = np.where((q @ m)[:, None] < 0.0, -q, q)
    m = q.mean(axis=0)
    return m / np.linalg.norm(m)


def slerp_quats(qa: np.ndarray, qb: np.ndarr …[+13308 chars]
```
<details><summary>output</summary>

```
/usr/bin/bash: line 351: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
static[0:106] mask IoU 0.8786, depth_scale=1.03, shift=(0,-2)px
warning: static image refinement skipped: Expected `matrix` to have shape (3, 3) or (N, 3, 3), got (4, 4)
static[379:476] mask IoU 0.8203, depth_scale=1.02, shift=(-2,-6)px
wrote /work/outputs/object_poses_world.npz with shape (477, 4, 4)

real	0m9.203s
user	0m12.160s
sys	0m7.014s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read().replace(\"q_release = Rot.from_matrix(T[carry_end]).as_quat()\",\"q_release = Rot.from_matrix(T[carry_end, :3, :3]).as_quat()\")
open(p,'w').write(s)
PY
time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
static[0:106] mask IoU 0.8786, depth_scale=1.03, shift=(0,-2)px
static[379:476] mask IoU 0.8203, depth_scale=1.02, shift=(-2,-6)px
wrote /work/outputs/object_poses_world.npz with shape (477, 4, 4)

real	0m9.177s
user	0m11.701s
sys	0m8.517s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
old='''        # Bridge the release interval into the refined static pose, avoiding a
        # one-frame jump at the static boundary.
        q_release = Rot.from_matrix(T[carry_end, :3, :3]).as_quat()
        q_final = Rot.from_matrix(T_final[:3, :3]).as_quat()
        q_bridge = slerp_quats(q_release, q_final, final_start - carry_end + 1)
        T[carry_end + 1:final_start, :3, :3] = Rot.from_quat(q_bridge[1:-1]).as_matrix()
        t_release = T[carry_end, :3, 3]
        t_final = T_final[:3, 3]
        frac = np.linspace(0.0, 1.0, final_start - carry_end + 1)[1:-1, None]
        T[carry_end + 1:final_start, :3, 3] = t_release + frac * (t_final - t_release)
'''
new='''        # Bridge the 10-frame release/settling interval (369..378, verified
        # from masks and hand motion) into the refined static pose, avoiding a
        # one-frame jump at the static boundary.
        bridge_first = 369
        n_bridge = final_start - bridge_first + 1
        q_release = Rot.from_matrix(T[bridge_first - 1, :3, :3]).as_quat()
        q_final = Rot.from_matrix(T_final[:3, :3]).as_quat()
        q_bridge = slerp_quats(q_release, q_final, n_bridge)
        T[bridge_first:final_start, :3, :3] = Rot.from_quat(q_bridge[1:-1]).as_matrix()
        t_release = T[bridge_first - 1, :3, 3]
        t_final = T_final[:3, 3]
        frac = np.linspace(0.0, 1.0, n_bridge)[1:-1, None]
        T[bridge_first:final_start, …[+216 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
static[0:106] mask IoU 0.8786, depth_scale=1.03, shift=(0,-2)px
static[379:476] mask IoU 0.8203, depth_scale=1.02, shift=(-2,-6)px
warning: static image refinement skipped: could not broadcast input array from shape (9,3,3) into shape (10,3,3)
wrote /work/outputs/object_poses_world.npz with shape (477, 4, 4)

real	0m9.216s
user	0m11.663s
sys	0m7.813s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/n_bridge = final_start - bridge_first + 1/n_bridge = final_start - bridge_first + 2/' /work/outputs/solve.py
time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
static[0:106] mask IoU 0.8786, depth_scale=1.03, shift=(0,-2)px
static[379:476] mask IoU 0.8203, depth_scale=1.02, shift=(-2,-6)px
wrote /work/outputs/object_poses_world.npz with shape (477, 4, 4)

real	0m9.086s
user	0m12.734s
sys	0m8.670s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def steps(X):
 rr=np.einsum('nji,njk->nik',X[:-1,:3,:3],X[1:,:3,:3]);a=np.linalg.norm(R.from_matrix(rr).as_rotvec(),axis=1)*180/np.pi;t=np.linalg.norm(np.diff(X[:,:3,3],axis=0),axis=1);return a,t
for n,X in [('base',B),('result',T)]:
 a,t=steps(X);print(n,'rot pct',np.round(np.percentile(a,[50,90,95,99,100]),2),'trans pct',np.round(np.percentile(t,[50,90,95,99,100]),4))
 print('static init center std',X[:107,:3,3].std(0),'final',X[379:,:3,3].std(0))
for i in range(1,len(T)):
 if np.linalg.norm(T[i,:3,3]-T[i-1,:3,3])>.15: print('trans jump',i,np.linalg.norm(T[i,:3,3]-T[i-1,:3,3]))
print('frames 366-381')
for i in range(365,382):print(i,np.round(T[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```
base rot pct [  2.42   9.12  21.59 163.85 175.78] trans pct [0.0179 0.0383 0.0525 0.1278 1.4337]
static init center std [0.00319105 0.00887769 0.05205875] final [0.00261523 0.00538779 0.01246317]
result rot pct [ 0.99  4.25  6.15 11.05 15.76] trans pct [0.0062 0.0317 0.0417 0.0578 0.1676]
static init center std [3.33066907e-16 0.00000000e+00 5.77315973e-15] final [9.71445147e-17 1.33226763e-15 2.22044605e-15]
trans jump 342 0.16758456378071876
frames 366-381
365 [0.1462 1.0507 3.1287]
366 [0.1349 1.0524 3.1159]
367 [0.1199 1.0624 3.0952]
368 [0.1176 1.0692 3.0919]
369 [0.1175 1.072  3.0975]
370 [0.1174 1.0749 3.103 ]
371 [0.1172 1.0777 3.1085]
372 [0.1171 1.0806 3.1141]
373 [0.117  1.0834 3.1196]
374 [0.1169 1.0863 3.1251]
375 [0.1168 1.0891 3.1306]
376 [0.1166 1.092  3.1362]
377 [0.1165 1 …[+122 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_final_all.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
Tb=W.astype(float);Tr=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
ids=np.arange(477);res=[]
def render_batch(i,Ts,box):
 x0,y0,s=box;Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0
 Tb=torch.tensor(np.stack([Tcw@T for T in Ts]),device='cuda',dtype=torch.float)
 cols,deps,_=nvdiffrast_render(K=Kc,H=s,W=s,ob_in_cams=Tb,glctx=gl,mesh_tensors=mt,output_size=(s,s),use_light=False,extra={})
 return cols.detach().cpu().numpy(),deps.detach().cpu().numpy()
for ii,i in enumerate(ids):
 box=cropbox(i);x0,y0,s=box;im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s];x=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255
 cols,deps=render_batch(i,[Tb[i],Tr[i]],box);om=OM[i,y0:y0+s,x0:x0+s];pm=PM[i,y0:y0+s,x0:x0+s];vals=[]
 for rgb,dep in zip(cols,deps):
  rr=np.clip(rgb,0,1);rm=dep>0.01;vis=om&~pm&rm;iou=(rm&om).sum()/max(1,(rm|om).sum())
  if vis.sum()<10:vals.append((iou,0,0));continue
  a=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)[vis];b=cv2.cvtColor(rr,cv2.COLOR_RGB2GRAY)[vis]
  gn=np.corrcoef(a,b)[0,1] if a.std()>1e-5 and b.std()>1e-5 else 0
  g1=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3)[vis]+cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)[vis]
  g2=cv2.Sobel(rr,cv2.CV_32F,1,0,ksize=3)[vis]+cv2.Sobel(rr,cv2.CV_32F,0,1,ksize=3)[vis]
  gg=np.corrcoef(g1,g2)[0,1] if g1.std()>1e-5 and g2.std()>1e-5 else 0
  vals.append((iou,gn,gg))
 res.app …[+1368 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
frame 0
frame 50
frame 100
frame 150
frame 200
frame 250
frame 300
frame 350
frame 400
frame 450

 initial_static frames 107 valid 107
baseline IoU all/valid 0.7918 0.7918 NCC 0.518 edge 0.8123
result IoU all/valid 0.8739 0.8739 NCC 0.6698 edge 0.9699

 carry_visible frames 217 valid 215
baseline IoU all/valid 0.7781 0.7853 NCC 0.3401 edge 0.821
result IoU all/valid 0.7767 0.7839 NCC 0.343 edge 0.8496

 occluded frames 21 valid 4
baseline IoU all/valid 0.0553 0.2901 NCC -0.0595 edge 0.7315
result IoU all/valid 0.0944 0.4954 NCC -0.0452 edge 0.6198

 lower_p …[+1507 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/release_eval.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
Tb=W.astype(float);Tw5=np.load('/work/scratch/traj_w5.npy');Tf=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];Ts=np.load('/work/scratch/static_opt_final.npy')
for i in range(365,380):
 vals=[]
 for nm,T in [('base',Tb[i]),('w5',Tw5[i]),('result',Tf[i]),('finalpose',Ts)]:
  box=cropbox(i);x0,y0,s=box;im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')[y0:y0+s,x0:x0+s];v=scores(i,T,box,im);vals.append((nm,round(v[0],3),round(v[1],3)))
 print(i,vals)
PY
python /work/scratch/release_eval.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
365 [('base', 0.73, 0.239), ('w5', 0.734, 0.266), ('result', 0.734, 0.266), ('finalpose', 0.276, 0.295)]
366 [('base', 0.758, 0.245), ('w5', 0.749, 0.101), ('result', 0.749, 0.101), ('finalpose', 0.418, 0.143)]
367 [('base', 0.695, 0.062), ('w5', 0.702, 0.113), ('result', 0.702, 0.113), ('finalpose', 0.536, -0.052)]
368 [('base', 0.684, 0.17), ('w5', 0.767, 0.105), ('result', 0.767, 0.105), ('finalpose', 0.591, -0.051)]
369 [('base', 0.812, 0.258), ('w5', 0.804, 0.234), ('result', 0.789, 0.079), ('finalpose', 0.648, -0.023)]
370 [('base', 0.809, 0.159), ('w …[+1015 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/static_window_fit.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
import json
cam=json.load(open('/work/inputs/camera.json'));Kreal=np.array(cam['K']);Twc=np.array(cam['T_world_cam'])
T0=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];base=T0[470].copy()
def fit(wa,wb):
 M=OM[wa:wb+1].mean(0)>=.5
 ys,xs=np.where(M);cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;sz=180;x0=int(cx-sz/2);y0=int(cy-sz/2);Mt=M[y0:y0+sz,x0:x0+sz]
 cc=Tcw[:3,:3]@base[:3,3]+Tcw[:3,3];zc=cc[2];ks=np.arange(.9,1.1001,.01);Ts=[]
 for k in ks:
  TT=base.copy();TT[:3,3]+=Twc[:3,2]*zc*(1/k-1);Ts.append(TT)
 Kc=K.copy();Kc[0,2]-=x0;Kc[1,2]-=y0;Tb=torch.tensor(np.stack([Tcw@TT for TT in Ts]),device='cuda',dtype=torch.float)
 _,deps,_=nvdiffrast_render(K=Kc,H=sz,W=sz,ob_in_cams=Tb,glctx=gl,mesh_tensors=mt,output_size=(sz,sz),use_light=False,extra={});rms=deps.detach().cpu().numpy()>.01
 best=(-1,None)
 for j,k in enumerate(ks):
  for dy in range(-8,9):
   for dx in range(-8,9):
    rm=np.roll(np.roll(rms[j],dy,0),dx,1);iou=(rm&Mt).sum()/(rm|Mt).sum()
    if iou>best[0]:best=(iou,(k,dx,dy))
 iou,(k,dx,dy)=best;out=base.copy();out[:3,3]+=Twc[:3,2]*zc*(1/k-1)+Twc[:3,0]*(dx*zc/Kreal[0,0])+Twc[:3,1]*(dy*zc/Kreal[1,1])
 print(wa,wb,'Marea',M.sum(),'best',best,'t',out[:3,3]);return out
outs=[]
for wa,wb in [(368,378),(369,378),(370,476),(375,476),(379,476)]:outs.append((f'{wa}_{wb}',fit(wa,wb)))
# eval each pose on e …[+410 chars]
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
368 378 Marea 1301 best (0.7885117493472585, (0.9700000000000001, 0, 8)) t [0.11168683 1.14128418 3.06067547]
369 378 Marea 1323 best (0.7890675241157556, (0.9800000000000001, 0, 6)) t [0.11324921 1.13013334 3.09229728]
370 476 Marea 1559 best (0.8220786846741045, (1.0100000000000002, 0, -3)) t [0.11775635 1.0858082  3.18204667]
375 476 Marea 1565 best (0.8234604105571848, (1.0100000000000002, 0, -3)) t [0.11775635 1.0858082  3.18204667]
379 476 Marea 1567 best (0.8235638921453693, (1.0100000000000002, 0, -3)) t [0.11775635 1.0858082  3.18204667]

 368_378
 …[+457 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/repeat.npz >/work/scratch/repeat.log 2>&1 && sha256sum /work/outputs/object_poses_world.npz /work/scratch/repeat.npz && python - <<'PY'
import numpy as np
a=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];b=np.load('/work/scratch/repeat.npz')['T_world_mesh'];print('max abs diff',np.abs(a-b).max(),'equal',np.array_equal(a,b))
PY"
```
<details><summary>output</summary>

```
65da99a48b1d710b735473b7dba9e44665b642184b597ccea9c99950b320c6ac  /work/outputs/object_poses_world.npz
65da99a48b1d710b735473b7dba9e44665b642184b597ccea9c99950b320c6ac  /work/scratch/repeat.npz
max abs diff 0.0 equal True

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/bridge_variants.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
Tout=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];Tw5=np.load('/work/scratch/traj_w5.npy')
Tbase=Tw5.copy();Tbase[:107]=Tout[:107];Tbase[379:]=Tout[379:]
variants={}
for bs in [369,373,376,378,379]:
 T=Tbase.copy()
 if bs<=378:
  n=379-bs+2;frac=np.linspace(0,1,n)[1:-1,None];t0=T[bs-1,:3,3];t1=Tout[379,:3,3];T[bs:379,:3,3]=t0+frac*(t1-t0)
  from scipy.spatial.transform import Rotation as R
  a=R.from_matrix(T[bs-1,:3,:3]);b=R.from_matrix(Tout[379,:3,:3]);d=(a.inv()*b).as_rotvec();T[bs:379,:3,:3]=R.from_rotvec(np.outer(np.linspace(0,1,n),d)[1:-1]).as_matrix()
 variants[bs]=T
for bs,T in variants.items():
 out=[]
 for i in range(369,379):out.append(scores(i,T[i],cropbox(i)))
 out=np.array(out);print(bs,'iou',round(out[:,0].mean(),3),'ncc',round(out[:,1].mean(),3),'edge',round(np.nanmean(out[:,3]),3))
 # max trans step across bridge and final
 st=np.linalg.norm(np.diff(T[365:382,:3,3],axis=0),axis=1);print(' maxstep',round(st.max(),3),[round(v,3) for v in st])
PY
python /work/scratch/bridge_variants.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
369 iou 0.522 ncc -0.008 edge 0.056
 maxstep 0.027 [0.017, 0.027, 0.008, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.0, 0.0]
373 iou 0.657 ncc 0.144 edge 0.047
 maxstep 0.027 [0.017, 0.027, 0.008, 0.022, 0.026, 0.006, 0.014, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.0, 0.0]
376 iou 0.723 ncc 0.202 edge 0.066
 maxstep 0.027 [0.017, 0.027, 0.008, 0.022, 0.026, 0.006, 0.014, 0.005, 0.018, 0.002, 0.027, 0.027, 0.027, 0.027, 0.0, 0.0]
378 iou 0.742 ncc 0.197 edge 0.054
 maxstep 0.054 [0.017, 0.027, 0.008, 0.022, 0.026, …[+227 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read()
s=s.replace('''        # Bridge the 10-frame release/settling interval (369..378, verified
        # from masks and hand motion) into the refined static pose, avoiding a
        # one-frame jump at the static boundary.
        bridge_first = 369''','''        # Keep the directly observed 369..375 poses, then blend over four
        # frames into the pooled final pose.  This preserves the stronger
        # per-frame silhouette evidence during release while avoiding the
        # 10-cm one-frame jump that a hard static boundary would create.
        bridge_first = 376''')
open(p,'w').write(s)
PY
time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0
static[0:106] mask IoU 0.8786, depth_scale=1.03, shift=(0,-2)px
static[379:476] mask IoU 0.8203, depth_scale=1.02, shift=(-2,-6)px
wrote /work/outputs/object_poses_world.npz with shape (477, 4, 4)

real	0m7.920s
user	0m13.010s
sys	0m8.008s

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_release_new.py <<'PY'
exec(open('/work/scratch/score_rots.py').read().split('# compare z rotations')[0])
res=np.load('/work/scratch/eval_final_all.npy');Tb=W.astype(float);Tr=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for i in range(369,379):
 v=scores(i,Tr[i],cropbox(i));row=res[res[:,0]==i][0];res[res[:,0]==i,5:8]=v[:3]
np.save('/work/scratch/eval_final_new.npy',res)
# print per phase and overall
for a,b,name in [(0,106,'initial_static'),(107,323,'carry_visible'),(324,344,'occluded'),(345,368,'lower_place'),(369,378,'release'),(379,476,'final_static'),(0,476,'all')]:
 m=(res[:,0]>=a)&(res[:,0]<=b);valid=m&(res[:,1]>100)
 print('\\n',name,'frames',m.sum(),'valid',valid.sum())
 for tag,off in [('baseline',2),('result',5)]:
  print(tag,'IoU all/valid',round(res[m,off].mean(),4),round(res[valid,off].mean(),4),'NCC',round(res[valid,off+1].mean(),4),'edge',round(np.nanmean(res[valid,off+2]),4))
PY
python /work/scratch/eval_release_new.py 2>/dev/null"
```
<details><summary>output</summary>

```
Warp 1.17.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA RTX 6000 Ada Generation" (47 GiB, sm_89, mempool enabled)
   Kernel cache:
     /work/scratch/.cache/warp/1.17.0

 initial_static frames 107 valid 107
baseline IoU all/valid 0.7918 0.7918 NCC 0.518 edge 0.8123
result IoU all/valid 0.8739 0.8739 NCC 0.6698 edge 0.9699

 carry_visible frames 217 valid 215
baseline IoU all/valid 0.7781 0.7853 NCC 0.3401 edge 0.821
result IoU all/valid 0.7767 0.7839 NCC 0.343 edge 0.8496

 occluded frames 21 valid 4
baseline IoU all/valid 0.0553 0.2901 NCC -0.0595 edge 0.7315
result IoU all/valid 0.0944 0.4954 NCC -0.0452 edge 0.6198

 lower_place frames 24 valid 24
baseline IoU all/valid 0.5989 0.5989 NCC 0.1132 edge 0.9186
result IoU al …[+485 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as R
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
def steps(X,a,b):
 rr=np.einsum('nji,njk->nik',X[a:b,:3,:3],X[a+1:b+1,:3,:3]);r=np.linalg.norm(R.from_matrix(rr).as_rotvec(),axis=1)*180/np.pi;t=np.linalg.norm(np.diff(X[a:b+1,:3,3],axis=0),axis=1);return r,t
for a,b,nm in [(0,106,'initial'),(107,323,'carry'),(324,344,'occluded'),(345,368,'lower'),(369,378,'release'),(379,476,'final'),(0,476,'all')]:
 print(nm)
 for n,X in [('baseline',B),('result',T)]:
  r,t=steps(X,a,b);print(n,'r90/max',round(np.percentile(r,90),2),round(r.max(),2),'t90/max',round(np.percentile(t,90),3),round(t.max(),3))
PY"
```
<details><summary>output</summary>

```
initial
baseline r90/max 3.13 5.95 t90/max 0.024 0.06
result r90/max 0.0 0.0 t90/max 0.0 0.0
carry
baseline r90/max 6.83 83.5 t90/max 0.038 0.07
result r90/max 4.83 10.4 t90/max 0.035 0.064
occluded
baseline r90/max 170.74 175.78 t90/max 0.297 1.434
result r90/max 1.62 2.75 t90/max 0.049 0.168
lower
baseline r90/max 19.93 35.96 t90/max 0.076 0.14
result r90/max 13.87 15.76 t90/max 0.053 0.07
release
baseline r90/max 22.54 30.28 t90/max 0.024 0.031
result r90/max 4.08 4.08 t90/max 0.027 0.027
final
baseline r90/max 4.53 9.78 t90/max 0.021 0.035
result r90/max 0.0 0.0 t90/max 0.0 0.0
all
baseline r90/max 9.12 175.78 t90/max 0.038 1.434
result r90/max 4.29 15.76 t90/max 0.032 0.168

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json,trimesh
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];gp=json.load(open('/work/inputs/ground_plane.json'));n=np.array(gp['plane'][:3]);d=gp['plane'][3]
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices
for i in [370,379,400,470]:
 p=v@T[i,:3,:3].T+T[i,:3,3];dist=p@n+d
 print(i,'min/max',dist.min(),dist.max(),'bottom pts within 2cm',np.mean(np.abs(dist)<.02))
PY"
```
<details><summary>output</summary>

```
370 min/max -0.02775796474675496 0.26872863259310864 bottom pts within 2cm 0.11321764596973727
379 min/max -0.019305260269745217 0.27797821357925656 bottom pts within 2cm 0.22743170686550082
400 min/max -0.019305260269745217 0.27797821357925656 bottom pts within 2cm 0.22743170686550082
470 min/max -0.019305260269745217 0.27797821357925656 bottom pts within 2cm 0.22743170686550082

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'git -C /code/reconstruction rev-parse HEAD; git -C /code/reconstruction status --short | head; ffmpeg -version | head -1; python --version'
```
<details><summary>output</summary>

```
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
D  ../.claude/skills/ego-reconstruction-setup/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-doctor/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-doctor/agents/openai.yaml
D  ../.claude/skills/hoi-object-reconstruction-run/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-run/agents/openai.yaml
D  ../.claude/skills/hoi-object-reconstruction-setup/SKILL.md
D  ../.claude/skills/hoi-object-reconstruction-setup/agents/openai.yaml
D  ../.claude/skills/hoi-object-reconstruction-setup/scripts/preflight_input.py
D  ../.claude/skills/ingestion_agent_doctor/SKILL.md
D  ../.claude/skills/ingestion_agent_onboard/SKILL.md
ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers
Python 3.11.10

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -l /work/outputs; sha256sum /work/outputs/object_poses_world.npz /work/outputs/solve.py'
```
<details><summary>output</summary>

```
total 76
-rw-rw-r-- 1 hanzhou hanzhou 61334 Oct 20 21:51 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou 14770 Oct 20 21:51 solve.py
52ac0d2b6819784546b7338ee4663c0c94119ec4e1dfcdf78f6683c2ca638711  /work/outputs/object_poses_world.npz
a287dda91dc6cc9a503fec670f40401bf8462f74a47dbe3b756059f62f0c8d84  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy,scipy,torch,trimesh,cv2,h5py,nvdiffrast,PIL,warp
print('python',sys.version.split()[0])
for m in [numpy,scipy,torch,trimesh,cv2,h5py,nvdiffrast,PIL,warp]:print(m.__name__,getattr(m,'__version__','unknown'))
PY"
```
<details><summary>output</summary>

```
python 3.11.10
numpy 1.26.3
scipy 1.16.3
torch 2.5.1+cu124
trimesh 5.1.1
cv2 4.11.0
h5py 3.16.0
nvdiffrast 0.4.0
PIL 10.2.0
warp 1.17.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/REPORT.md <<'MD'
# Spray-bottle 6-DoF trajectory recovery

## Summary

The output trajectory is a temporally constrained repair of the supplied
"'`T_world_mesh` baseline.  The main changes are:

1. **Phase segmentation from hand contact.**  The confident hand-contact run
   starts at frame 108 and ends at frame 366.  This gives a table-rest phase
   (0-106), a carried phase (107-368), and a floor-rest phase (379-476), plus a
   short release/settling interval (369-378).
2. **SO(3)-aware branch repair.**  The baseline contains exact-looking
   180-degree spin-flip failures.  The final segment'"'s branch is selected
   automatically by continuity with the pre-release pose, then pooled.  This
   removes the 174-degree jump at frame 369.  Short discontinuous intervals at
   262-269 and 324-344 are replaced by shortest-path SO(3) interpolation.
3. **Temporal filtering.**  Rotations are smoothed as sign-consistent
   quaternions with a 5-frame Savitzky-Golay filter.  Translations use a
   5-frame median filter; the two 1.3-1.4 m tracking spikes at 343-344 are
   interpolated before filtering.
4. **Static-pose pooling.**  Every frame in each rest phase is forced to one
   rigid pose.  The pooled pose is then aligned to the phase's majority object
   mask using the textured mesh renderer.  A coarse search over +/-10% depth
   and +/-6 pixels image-plane translation improved the majority-mask IoU from
   0.812 to 0.879 for the initial rest pose an …[+5417 chars]
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
52ac0d2b6819784546b7338ee4663c0c94119ec4e1dfcdf78f6683c2ca638711  /work/outputs/object_poses_world.npz
a287dda91dc6cc9a503fec670f40401bf8462f74a47dbe3b756059f62f0c8d84  /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -n \"bridge_first\\|Keep the directly\" /work/outputs/solve.py"
```
<details><summary>output</summary>

```
316:        # Keep the directly observed 369..375 poses, then blend over four
320:        bridge_first = 376
321:        n_bridge = final_start - bridge_first + 2
322:        q_release = Rot.from_matrix(T[bridge_first - 1, :3, :3]).as_quat()
325:        T[bridge_first:final_start, :3, :3] = Rot.from_quat(q_bridge[1:-1]).as_matrix()
326:        t_release = T[bridge_first - 1, :3, 3]
329:        T[bridge_first:final_start, :3, 3] = t_release + frac * (t_final - t_release)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
for i in range(369,380):print(i,np.round(T[i,:3,3],4))
PY"
```
<details><summary>output</summary>

```
369 [0.1158 1.0903 3.0875]
370 [0.1099 1.1124 3.0755]
371 [0.1099 1.1184 3.0732]
372 [0.1095 1.1304 3.0658]
373 [0.1095 1.1352 3.064 ]
374 [0.1072 1.1518 3.0586]
375 [0.1047 1.1524 3.0586]
376 [0.1076 1.1394 3.0821]
377 [0.1105 1.1265 3.1057]
378 [0.1134 1.1135 3.1292]
379 [0.1163 1.1005 3.1528]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/TOOLS.md <<'MD'
# Tools and external dependencies

No model weights or repositories were downloaded for this solution.  The
supplied reconstruction, mesh, masks, camera metadata, and existing installed
software were used.  "'`solve.py` requires no network access.

## Required by `solve.py`

| Tool | Version / revision | Role | Needed at solve time |
|---|---|---|---|
| Python | 3.11.10 | runtime | yes |
| NumPy | 1.26.3 | arrays, phase construction, SE(3) operations | yes |
| SciPy | 1.16.3 | quaternion rotations, Savitzky-Golay and median filters | yes |
| h5py | 3.16.0 | read supplied `masks.h5` for static-pose refinement | yes |
| trimesh | 5.1.1 | load `object_mesh.glb` and texture | yes |
| PyTorch | 2.5.1+cu124 | GPU tensors for mask rendering | yes |
| nvdiffrast | 0.4.0 | rasterize mesh for static-pose mask fitting | yes |
| Pillow | 10.2.0 | decode GLB base-color texture via trimesh | yes |
| CARI4D `Utils.py` | `/code/reconstruction` git `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | batched nvdiffrast render helper used by the static-pose fitter | yes |
| NVIDIA Warp | 1.17.0 | imported indirectly by CARI4D `Utils.py` | yes (indirect) |
| OpenCV | 4.11.0 | imported indirectly by CARI4D `Utils.py` | yes (indirect) |
| Open3D | 0.20.0 | imported indirectly by CARI4D `Utils.py` | yes (indirect) |
| Kornia | 0.8.3 | optional import inside CARI4D `Utils.py` | not used |

The GPU renderer in `solve.py` uses less than about 1.4 GB and c …[+1272 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python -m py_compile /work/outputs/solve.py && python - <<'PY'
import numpy as np
p='/work/outputs/object_poses_world.npz';z=np.load(p);T=z['T_world_mesh']
print('keys',z.files,'shape',T.shape,'dtype',T.dtype,'finite',np.isfinite(T).all())
R=T[:,:3,:3];orth=np.abs(np.einsum('nji,njk->nik',R,R)-np.eye(3)).max();det=np.linalg.det(R)
print('orthogonality max error',orth,'det range',det.min(),det.max(),'bottom rows unique',np.unique(T[:,3,:],axis=0))
PY"
```
<details><summary>output</summary>

```
keys ['T_world_mesh'] shape (477, 4, 4) dtype float64 finite True
orthogonality max error 1.1102230246251565e-15 det range 0.9999999999999992 1.0000000000000013 bottom rows unique [[0. 0. 0. 1.]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'ls -lh /work/outputs; wc -l /work/outputs/REPORT.md /work/outputs/TOOLS.md /work/outputs/solve.py; sha256sum /work/outputs/object_poses_world.npz'
```
<details><summary>output</summary>

```
total 92K
-rw-rw-r-- 1 hanzhou hanzhou 6.7K Oct 20 21:52 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 2.7K Oct 20 21:52 TOOLS.md
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:52 __pycache__
-rw-rw-r-- 1 hanzhou hanzhou  60K Oct 20 21:51 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  15K Oct 20 21:51 solve.py
  112 /work/outputs/REPORT.md
   51 /work/outputs/TOOLS.md
  351 /work/outputs/solve.py
  514 total
52ac0d2b6819784546b7338ee4663c0c94119ec4e1dfcdf78f6683c2ca638711  /work/outputs/object_poses_world.npz

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "grep -R \"work/scratch\\|http://\\|https://\\|pipeline\\|weights\" -n /work/outputs/solve.py || true"
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
ls -lah /work/outputs"
```
<details><summary>output</summary>

```
total 96K
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 21:52 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 21:22 ..
-rw-rw-r-- 1 hanzhou hanzhou 6.7K Oct 20 21:52 REPORT.md
-rw-rw-r-- 1 hanzhou hanzhou 2.7K Oct 20 21:52 TOOLS.md
-rw-rw-r-- 1 hanzhou hanzhou  60K Oct 20 21:51 object_poses_world.npz
-rwxrwxr-x 1 hanzhou hanzhou  15K Oct 20 21:51 solve.py

```
</details>

**Agent:** Completed the task and produced all required outputs:

- `/work/outputs/object_poses_world.npz` — 477 float64 `(4,4)` world-space poses
- `/work/outputs/solve.py` — deterministic offline solver
- `/work/outputs/REPORT.md` — phases, methodology, validation, confidence, runtime
- `/work/outputs/TOOLS.md` — tools, versions, and dependency details

Validation summary:

- Mean visible-mask IoU improved from **0.756** to **0.799**
- Mean visible-object photometric NCC improved from **0.332** to **0.373**
- Maximum rotation step reduced from **175.8°** to **15.8°**
- Maximum translation step reduced from **1.434 m** to **0.168 m**
- Initial and final static phases are now exactly rigid
- Final solver runtime is approximately **8 seconds**
- Repeated runs are bit-identical; output SHA-256 is `52ac0d2b6819784546b7338ee4663c0c94119ec4e1dfcdf78f6683c2ca638711`
