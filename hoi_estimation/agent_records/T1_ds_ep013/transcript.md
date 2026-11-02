# T1_ds_ep013

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
/usr/bin/bash -c "sed -n '1,140p' /work/AGENT_TASK.md"
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
/usr/bin/bash -c 'ls -lah /work/inputs && cat /work/inputs/task.txt && cat /work/inputs/episode.json && cat /work/inputs/camera.json && cat /work/inputs/ground_plane.json'
```
<details><summary>output</summary>

```
total 93M
drwxrwxr-x 2 hanzhou hanzhou 4.0K Oct 20 17:35 .
drwxrwxr-x 8 hanzhou hanzhou 4.0K Oct 20 17:35 ..
-rw-rw-r-- 1 hanzhou hanzhou    0 Oct 20 17:35 .preexported
-rw-rw-r-- 1 hanzhou hanzhou  830 Oct 20 15:53 camera.json
-rw-rw-r-- 1 hanzhou hanzhou   64 Oct 20 15:53 episode.json
-rw-rw-r-- 1 hanzhou hanzhou 188K Oct 20 15:54 frame0_reference_human.npz
-rw-rw-r-- 1 hanzhou hanzhou  398 Oct 20 15:53 ground_plane.json
-rw-rw-r-- 1 hanzhou hanzhou 7.4M Oct 20 15:54 masks.h5
-rw-rw-r-- 1 hanzhou hanzhou  16M Oct 20 15:53 object_mesh.glb
-rw-rw-r-- 1 hanzhou hanzhou  45M Oct 20 15:54 reconstruction.npz
-rw-rw-r-- 1 hanzhou hanzhou  174 Oct 20 15:53 task.txt
-rw-rw-r-- 1 hanzhou hanzhou  26M Oct 20 15:53 video.mp4
Walk over to the snack box on the white desk. Put the snack box on the grou …[+1391 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /work -maxdepth 2 -type f -printf '%p %s\\n' | sort | head -100 && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"
```
<details><summary>output</summary>

```
/work/.codex_home/.sandbox_migration 3
/work/.codex_home/config.toml 361
/work/.codex_home/goals_1.sqlite 4096
/work/.codex_home/goals_1.sqlite-shm 32768
/work/.codex_home/goals_1.sqlite-wal 78312
/work/.codex_home/installation_id 36
/work/.codex_home/logs_2.sqlite 49152
/work/.codex_home/logs_2.sqlite-shm 32768
/work/.codex_home/logs_2.sqlite-wal 119512
/work/.codex_home/memories_1.sqlite 4096
/work/.codex_home/memories_1.sqlite-shm 32768
/work/.codex_home/memories_1.sqlite-wal 61832
/work/.codex_home/model_catalog.json 3042
/work/.codex_home/queue_1.sqlite 4096
/work/.codex_home/queue_1.sqlite-shm 32768
/work/.codex_home/queue_1.sqlite-wal 82432
/work/.codex_home/state_5.sqlite 4096
/work/.codex_home/state_5.sqlite-shm 32768
/work/.codex_home/state_5.sqlite-wal 2484392
/work/.codex_home/ …[+760 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /pipeline -maxdepth 3 -type f -printf '%p %s\\n' | sort | head -200"
```
<details><summary>output</summary>

```
/pipeline/.stages/01_depth.json 1743
/pipeline/.stages/02_export.json 1951
/pipeline/.stages/03_human_initialization.json 2148
/pipeline/.stages/04_depth_alignment.json 1848
/pipeline/.stages/05_object_pose.json 1773
/pipeline/.stages/06_coconet.json 2829
/pipeline/.stages/07_refinement.json 2133
/pipeline/.stages/08_render.json 2526
/pipeline/depth/moge2/.aligned.h5.storage-repack.lock 0
/pipeline/depth/moge2/aligned.h5 1133571304
/pipeline/depth/moge2/depth_report.json 1811
/pipeline/depth/moge2/intrinsics.pkl 98139
/pipeline/depth/moge2/raw.npy 3262906496
/pipeline/export/episode_000013/edex 563
/pipeline/export/episode_000013/wild_export.json 2740
/pipeline/foundationpose/episode_000013.pkl 262437
/pipeline/inference/.materialized_inputs.h5.lock 0
/pipeline/inference/coconet.pth 705120 …[+549 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np, json
p='/work/inputs/reconstruction.npz'
z=np.load(p)
print(z.files)
for k in z.files:
 a=z[k]
 print(k,a.shape,a.dtype, 'minmax', (np.nanmin(a),np.nanmax(a)) if np.issubdtype(a.dtype,np.number) else '')
print('contacts sample',zoom:=z['contact_prob_right_left'][::50])
PY"
```
<details><summary>output</summary>

```
['T_world_mesh', 'T_cam_mesh_foundationpose', 'T_cam_mesh_coconet', 'human_vertices_world', 'human_keypoints_world', 'contact_prob_right_left']
T_world_mesh (922, 4, 4) float32 minmax (-0.99984026, 3.6959171)
T_cam_mesh_foundationpose (922, 4, 4) float32 minmax (-1.4173024, 7.333685)
T_cam_mesh_coconet (922, 4, 4) float32 minmax (-0.99904436, 3.5280752)
human_vertices_world (922, 4610, 3) float32 minmax (-0.89827895, 3.6723018)
human_keypoints_world (922, 70, 3) float32 minmax (-0.8807008, 3.5946677)
contact_prob_right_left (922, 2) float32 minmax (6.6739976e-17, 1.0)
contacts sample [[2.6043399e-09 1.6452394e-09]
 [4.3528034e-10 2.0607998e-09]
 [4.3730126e-04 6.3268113e-01]
 [9.9999994e-01 1.0000000e+00]
 [9.9999988e-01 1.0000000e+00]
 [1.0000000e+00 1.0000000e+00]
 [9.9999779e-01 9.99999 …[+381 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "ffprobe -v error -show_entries stream=width,height,nb_frames,r_frame_rate,duration -of default=nw=1 /work/inputs/video.mp4 && python - <<'PY'
import h5py, numpy as np
with h5py.File('/work/inputs/masks.h5') as f:
 print(f.keys())
 for k in f: 
  d=f[k]; print(k,d.shape,d.dtype, np.sum(d[0]) if d.ndim==3 else '')
PY"
```
<details><summary>output</summary>

```
width=1536
height=1152
r_frame_rate=30/1
duration=30.733333
nb_frames=922
<KeysViewHDF5 ['object', 'person']>
object (922, 1152, 1536) bool 5956
person (922, 1152, 1536) bool 54708

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
p='/work/inputs/object_mesh.glb'
s=trimesh.load(p,force='scene',process=False)
print(s, s.geometry.keys())
g=s.to_geometry()
print(type(g),g.vertices.shape,g.faces.shape,g.bounds,g.extents,g.centroid, g.visual, type(g.visual))
print('watertight',g.is_watertight,'volume',g.volume,'area',g.area)
print('scene graph',s.graph.nodes_geometry)
PY"
```
<details><summary>output</summary>

```
<trimesh.Scene(len(geometry)=1)> odict_keys(['snackbox.001'])
<class 'trimesh.base.Trimesh'> (293289, 3) (97862, 3) [[-0.17087026 -0.10298643 -0.20244676]
 [ 0.16922666  0.10250368  0.20245828]] [0.34009692 0.2054901  0.40490504] [ 1.03784936e-11  3.33431084e-11 -4.42337201e-11] <trimesh.visual.texture.TextureVisuals object at 0x7e6d7f47fdd0> <class 'trimesh.visual.texture.TextureVisuals'>
watertight False volume 0.026817061730150655 area 0.5597393261592739
scene graph ['snackbox.001']

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'cat /pipeline/pipeline_report.json && for f in /pipeline/.stages/*.json; do echo ===$f; cat $f; done'
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
    "mtime_ns": 1791296647940451643,
    "path": "/home/hanzhou/Projects/v2d/run/out/episode_000013/inference/episode_000013_step200000_before_after.mp4",
    "size": 67508313
  },
  "defaults": {
    "coconet_supervision": "checkpoint_embedded_offline",
    "depth": "moge2",
    "foundationpose": "register_first_then_track",
    "human_prompt": "sam2_bbox_mask",
    "postopt_contact_activation_distance_m": 0.05,
    "postopt_frame …[+22284 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "find /code/reconstruction -maxdepth 3 -type f | sed -n '1,200p'"
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
/code/reconstructio …[+7037 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import pickle
for p in ['/pipeline/foundationpose/episode_000013.pkl','/pipeline/mhr_init/episode_000013.stats.pkl']:
 with open(p,'rb') as f: x=pickle.load(f)
 print('\\n',p,type(x))
 if isinstance(x,dict):
  for k,v in x.items():
   try: print(k,type(v),getattr(v,'shape',None),getattr(v,'dtype',None),str(v)[:300])
   except: pass
 else: print(x)
PY"
```
<details><summary>output</summary>

```

 /pipeline/foundationpose/episode_000013.pkl <class 'dict'>
frames <class 'list'> None None ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029',
camera_id <class 'int'> None None 0
fp_poses_cam <class 'numpy.ndarray'> (922, 1, 4, 4) float32 [[[[ 0.02237438 -0.9942518   0.10470331 -0.18150255]
   [ 0.9899095   0.00737462 -0.14150855  0.02689826]
   [ 0.13992295  0.10681292  0.98438436  3.3325582 ]
   [ 0.          0.          0.          1.        ]]]


 [[[ 0.02254655 -0.9941371   0.10574971 -0.18147984]
   [ 0.9900774   0.00752798 -0.
fp_candid …[+12270 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "wc -l /work/AGENT_TASK.md; nl -ba /work/AGENT_TASK.md | sed -n '45,130p'"
```
<details><summary>output</summary>

```
120 /work/AGENT_TASK.md
    45	   smoothness / constant-velocity terms on SE(3), and rigid hand-object attachment while carried.
    46	5. **Anchors and propagation**: find frames where the pose is unambiguous (object large, little occluded, distinctive
    47	   view, high evidence margin between hypotheses) and propagate forward and backward from them through the ambiguous
    48	   segments, rather than trusting every frame equally.
    49	
    50	## Validate without ground truth
    51	You have no ground truth and must not try to get it. Build your own checks and make decisions with them:
    52	measures of how well a trajectory explains the video (held-out frames, mask IoU, photometric / flow residuals,
    53	temporal smoothness, physical plausibility such as no floating or penetrati …[+6424 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "mkdir -p /work/scratch/video_frames /work/outputs && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"fps=2,scale=384:288,tile=10x7\" -frames:v 1 /work/scratch/contact_2fps.jpg && ffmpeg -hide_banner -loglevel error -i /work/inputs/video.mp4 -vf \"select='not(mod(n,30))',scale=512:384\" -vsync vfr /work/scratch/video_frames/f%04d.jpg && ls -lh /work/scratch/contact_2fps.jpg && ls /work/scratch/video_frames | wc -l"
```
<details><summary>output</summary>

```
-rw-rw-r-- 1 hanzhou hanzhou 387K Oct 20 17:35 /work/scratch/contact_2fps.jpg
31

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/diag.py <<'PY'
import numpy as np,h5py,cv2,json
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_erosion
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh'].astype(float); Tf=z['T_cam_mesh_foundationpose'].astype(float); Tc=z['T_cam_mesh_coconet'].astype(float)
h=np.load('/work/inputs/human_keypoints_world.npz') if False else None
kp=z['human_keypoints_world']; cp=z['contact_prob_right_left']
with json.load(open('/work/inputs/camera.json')) as c: K=np.array(c['K']); Twc=np.array(c['T_world_cam']); Kp=np.array(c['K_pipeline'])
with h5py.File('/work/inputs/masks.h5') as f:
 om=f['object'][:]; pm=f['person'][:]
areas=om.sum((1,2)); personareas=pm.sum((1,2))
ys,xs=np.where(om.reshape(len(om),-1))
# np.where flattened but per frame need calculate
cent=np.zeros((len(om),2)); bb=np.zeros((len(om),4),int)
for i in range(len(om)):
 y,x=np.where(om[i]); cent[i]=[x.mean(),y.mean()]; bb[i]=[x.min(),y.min(),x.max(),y.max()]
# projections baseline center origin t and bbox corners
cen=T[:,:3,3]; pc=(Twc[:3,:3]@cen.T+Twc[:3,3:4]).T; uv=(K@pc.T).T; uv=uv[:,:2]/uv[:,2:]
# world/cam compare translations
fc=(Twc[:3,:3]@Tf[:,:3,3].T+Twc[:3,3:4]).T
cc=(Twc[:3,:3]@Tc[:,:3,3].T+Twc[:3,3:4]).T
# rot difference baseline vs coconet world
Rb=T[:,:3,:3]; Rc=(Twc[:3,:3]@Tc[:,:3,:3])
angs=np.degrees(np.arccos(np.clip((np.trace(np.einsum('nij,nkj->nik',Rb,Rc),axis1=1,axis2=2)-1)/2,-1,1)))
td=np.linalg.norm(T[:,:3,3]-cc,axis …[+1755 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/diag.py", line 8, in <module>
    with json.load(open('/work/inputs/camera.json')) as c: K=np.array(c['K']); Twc=np.array(c['T_world_cam']); Kp=np.array(c['K_pipeline'])
TypeError: 'dict' object does not support the context manager protocol

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i \"s/with json.load(open('\\/work\\/inputs\\/camera.json')) as c:/with open('\\/work\\/inputs\\/camera.json') as cf:\\n c=json.load(cf)/\" /work/scratch/diag.py && python /work/scratch/diag.py"
```
<details><summary>output</summary>

```
  File "/work/scratch/diag.py", line 9
    c=json.load(cf) K=np.array(c['K']); Twc=np.array(c['T_world_cam']); Kp=np.array(c['K_pipeline'])
                    ^
SyntaxError: invalid syntax

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i '9s/c=json.load(cf) /c=json.load(cf); /' /work/scratch/diag.py && python /work/scratch/diag.py"
```
<details><summary>output</summary>

```
   0-  50 area    5995 foot   718.6,  583.8 t [0.02513096 0.20371383 3.33364175] dR mean/max    0.1/   0.1 dT mean/max 0.0004/0.0022 contact [5.9432609e-10 5.6757976e-10] angcoco    1.3 tdcoco 0.0812 rhd 0.838 lhd 0.872
  50- 100 area    4933 foot   723.0,  580.2 t [0.02670849 0.20780598 3.37330749] dR mean/max    0.5/   2.3 dT mean/max 0.0110/0.0390 contact [2.38876910e-05 1.53738335e-02] angcoco    1.3 tdcoco 0.1001 rhd 0.497 lhd 0.581
 100- 200 area    2924 foot   700.4,  578.9 t [-0.09580586  0.18881235  3.48761515] dR mean/max    1.4/   8.9 dT mean/max 0.0159/0.0400 contact [0.9012032  0.96803933] angcoco    1.3 tdcoco 0.1353 rhd 0.239 lhd 0.260
 200- 300 area    2454 foot   624.7,  720.9 t [-0.45799212  0.75730891  3.55075786] dR mean/max    9.4/ 112.8 dT mean/max 0.0436/0.1565 conta …[+7437 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/crops.py <<'PY'
import cv2,numpy as np,h5py
cap=cv2.VideoCapture('/work/inputs/video.mp4')
D=np.load('/work/scratch/diag.npz'); bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][:]
out=[]
for i in range(len(om)):
 ok,im=cap.read();
 if i%10: continue
 x0,y0,x1,y1=bb[i]; mx,my=(x1-x0)//2,(y1-y0)//2; cx=(x0+x1)//2;cy=(y0+y1)//2
 pad=max(80, int(max(x1-x0,y1-y0)*1.3)); pad=min(pad,180)
 a=max(0,cx-pad); b=min(im.shape[1],cx+pad); c=max(0,cy-pad); d=min(im.shape[0],cy+pad)
 crop=im[c:d,a:b].copy(); m=om[i,c:d,a:b]
 cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop,cnt,-1,(0,0,255),1)
 crop=cv2.resize(crop,(256,256),interpolation=cv2.INTER_CUBIC)
 cv2.rectangle(crop,(0,0),(70,22),(0,0,0),-1); cv2.putText(crop,f'{i}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
 out.append(crop)
cap.release()
while len(out)%10: out.append(np.zeros_like(out[0]))
rows=[]
for j in range(0,len(out),10): rows.append(np.hstack(out[j:j+10]))
sheet=np.vstack(rows); cv2.imwrite('/work/scratch/object_crops.jpg',sheet); print(sheet.shape)
PY
python /work/scratch/crops.py"
```
<details><summary>output</summary>

```
(2560, 2560, 3)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat /pipeline/export/episode_000013/wild_export.json; python - <<'PY'
import h5py
for p in ['/pipeline/depth/moge2/aligned.h5','/pipeline/inference/materialized_inputs.h5']:
 print('\\n',p)
 with h5py.File(p,'r') as f:
  print(f.keys())
  def visit(name,obj):
   if hasattr(obj,'shape'): print(name,obj.shape,obj.dtype)
  f.visititems(visit)
PY"
```
<details><summary>output</summary>

```

 /pipeline/depth/moge2/aligned.h5
<KeysViewHDF5 ['aligned', 'alignment', 'frame_names', 'raw']>
aligned/front_stereo_camera_left (922,) object
alignment/front_stereo_camera_left/scale (922,) float32
alignment/front_stereo_camera_left/shift (922,) float32
alignment/front_stereo_camera_left/valid_count (922,) int32
frame_names/front_stereo_camera_left (922,) object
raw/front_stereo_camera_left (922,) object

 /pipeline/inference/materialized_inputs.h5
<KeysViewHDF5 ['K_rois', 'bboxes', 'identity_json', 'input_rgbs', 'input_xyz', 'observed_human_masks', 'observed_object_masks', 'render_rgbs', 'render_xyz']>
K_rois (922, 3, 3) float32
bboxes (922, 4) float32
identity_json () object
input_rgbs (922, 3, 224, 224) float32
input_xyz (922, 6, 224, 224) float32
observed_human_masks (922, 224, 224)  …[+125 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import trimesh, numpy as np
from PIL import Image
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=g.visual
print(type(v),v.uv.shape if v.uv is not None else None)
print(v.material)
for k in dir(v.material):
 if not k.startswith('_'):
  try:
   x=getattr(v.material,k)
   if isinstance(x,(np.ndarray,Image.Image,str,int,float,type(None))): print(k,type(x),getattr(x,'shape',None),str(x)[:100])
  except: pass
if v.material.baseColorTexture:
 im=np.asarray(v.material.baseColorTexture); print('tex',im.shape,im.dtype,im.min(),im.max()); Image.fromarray(im).save('/work/scratch/texture.png')
print('uv range',v.uv.min(0),v.uv.max(0))
PY"
```
<details><summary>output</summary>

```
<class 'trimesh.visual.texture.TextureVisuals'> (293289, 2)
<trimesh.visual.material.PBRMaterial object at 0x7c7384f43010>
alphaCutoff <class 'NoneType'> None None
alphaMode <class 'NoneType'> None None
baseColorFactor <class 'NoneType'> None None
baseColorTexture <class 'PIL.Image.Image'> None <PIL.Image.Image image mode=RGB size=2996x3064 at 0x7C7384F09F90>
doubleSided <class 'bool'> None True
emissiveFactor <class 'NoneType'> None None
emissiveTexture <class 'NoneType'> None None
main_color <class 'numpy.ndarray'> (4,) [102 102 102 255]
metallicFactor <class 'float'> None 0.0
metallicRoughnessTexture <class 'NoneType'> None None
name <class 'str'> None _texture.001
normalTexture <class 'NoneType'> None None
occlusionTexture <class 'NoneType'> None None
roughnessFactor <class 'NoneType'> …[+96 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import torch
print(torch.__version__,torch.cuda.is_available())
try:
 import nvdiffrast.torch as dr; print('nvdiffrast ok')
except Exception as e: print('nvdiffrast fail',e)
try:
 import pytorch3d; print('pytorch3d',pytorch3d.__version__)
except Exception as e: print('no pytorch3d',e)
try:
 import kornia; print('kornia',kornia.__version__)
except Exception as e: print('no kornia',e)
PY"
```
<details><summary>output</summary>

```
2.5.1+cu124 True
nvdiffrast ok
pytorch3d 0.7.9
kornia 0.8.3

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/render_test.py <<'PY'
import numpy as np, torch, trimesh, cv2, json
import nvdiffrast.torch as dr
from PIL import Image
torch.manual_seed(0); np.random.seed(0)
dev='cuda'; H,W=1152,1536
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(geo.vertices,dtype=torch.float32,device=dev); F=torch.tensor(geo.faces,dtype=torch.int32,device=dev)
UV=torch.tensor(geo.visual.uv,dtype=torch.float32,device=dev)
tex=torch.tensor(np.asarray(geo.visual.material.baseColorTexture),dtype=torch.float32,device=dev)/255
ctx=dr.RasterizeCudaContext()
with open('/work/inputs/camera.json') as f: cam=json.load(f)
K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
Z=np.load('/work/inputs/reconstruction.npz'); Ts=Z['T_world_mesh'].astype(np.float32)
def render(i):
 T=torch.tensor(Ts[i],device=dev)
 vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
 w=(T@vh.T).T
 pc=(torch.tensor(Tcw,dtype=torch.float32,device=dev)@w.T).T
 x=pc[:,0]/pc[:,2]; y=pc[:,1]/pc[:,2]; z=pc[:,2]
 clip=torch.stack([x*(2*K[0,0]/W), y*(2*K[1,1]/H), z, torch.ones_like(z)],1)[None]
 rast,rdb=dr.rasterize(ctx,clip,F,resolution=[H,W])
 uv,uvdb=dr.interp(UV[None],rast,rdb)
 mask=rast[0,...,3]>0
 texn=torch.stack([uv[0,...,0]*2-1, uv[0,...,1]*2-1],-1)[None]
 rgb=torch.nn.functional.grid_sample(tex.permute(2,0,1)[None],texn,align_corners=False,mode='bilinear',padding_mode='border')[0].permute(1,2,0)
 # zero outside
 rgb=rgb*mas …[+703 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 32, in <module>
    m,r=render(i)
        ^^^^^^^^^
  File "/work/scratch/render_test.py", line 22, in render
    uv,uvdb=dr.interp(UV[None],rast,rdb)
            ^^^^^^^^^
AttributeError: module 'nvdiffrast.torch' has no attribute 'interp'

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/dr.interp(/dr.interpolate(/' /work/scratch/render_test.py && python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/render_test.py", line 32, in <module>
    m,r=render(i)
        ^^^^^^^^^
  File "/work/scratch/render_test.py", line 22, in render
    uv,uvdb=dr.interpolate(UV[None],rast,rdb)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _ …[+180 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import inspect,nvdiffrast.torch as dr
print(inspect.signature(dr.interpolate))
print(inspect.getsource(dr.interpolate)[:1000])
PY"
```
<details><summary>output</summary>

```
(attr, rast, tri, rast_db=None, diff_attrs=None)
def interpolate(attr, rast, tri, rast_db=None, diff_attrs=None):
    """Interpolate vertex attributes.

    All input tensors must be contiguous and reside in GPU memory. The output tensors
    will be contiguous and reside in GPU memory.

    Args:
        attr: Attribute tensor with dtype `torch.float32`. 
              Shape is [num_vertices, num_attributes] in range mode, or 
              [minibatch_size, num_vertices, num_attributes] in instanced mode.
              Broadcasting is supported along the minibatch axis.
        rast: Main output tensor from `rasterize()`.
        tri: Triangle tensor with shape [num_triangles, 3] and dtype `torch.int32`.
        rast_db: (Optional) Tensor containing image-space derivatives of barycentrics …[+250 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dr.interpolate(UV\\[None\\],rast,rdb)/dr.interpolate(UV[None],rast,F,rdb)/' /work/scratch/render_test.py && python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 0 baseline
100 0 baseline
200 0 baseline
240 0 baseline
300 0 baseline
400 0 baseline
500 0 baseline
600 0 baseline
700 0 baseline
740 0 baseline
775 0 baseline
800 0 baseline
850 0 baseline
921 0 baseline

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py'
s=open(p).read()
s=s.replace(\"clip=torch.stack([x*(2*K[0,0]/W), y*(2*K[1,1]/H), z, torch.ones_like(z)],1)[None]\", \"clip=torch.stack([x*(2*K[0,0]/W), y*(2*K[1,1]/H), 0.5*z+0.1, z],1)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 631 baseline
100 486 baseline
200 477 baseline
240 869 baseline
300 961 baseline
400 948 baseline
500 1270 baseline
600 1013 baseline
700 1060 baseline
740 797 baseline
775 1100 baseline
800 996 baseline
850 836 baseline
921 850 baseline

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
exec(open('/work/scratch/render_test.py').read().split('cap=cv2.VideoCapture')[0])
i=0;T=torch.tensor(Ts[i],device=dev);vh=torch.cat([V,torch.ones_like(V[:,:1])],1);w=(T@vh.T).T;pc=(torch.tensor(Tcw,dtype=torch.float32,device=dev)@w.T).T
u=K[0,0]*pc[:,0]/pc[:,2]+K[0,2]; v=K[1,1]*pc[:,1]/pc[:,2]+K[1,2]
print(pc[:,2].min(),pc[:,2].max(),u.min(),u.max(),v.min(),v.max())
mask,r=render(0); ys,xs=np.where(mask);print('render bbox',xs.min(),xs.max(),ys.min(),ys.max(),mask.sum())
PY"
```
<details><summary>output</summary>

```
tensor(3.0545, device='cuda:0') tensor(3.5147, device='cuda:0') tensor(671.5619, device='cuda:0') tensor(770.9609, device='cuda:0') tensor(550.6785, device='cuda:0') tensor(615.7924, device='cuda:0')
render bbox 740 771 573 593 631

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/render_test.py';s=open(p).read()
s=s.replace(\"x=pc[:,0]/pc[:,2]; y=pc[:,1]/pc[:,2]; z=pc[:,2]\\n clip=torch.stack([x*(2*K[0,0]/W), y*(2*K[1,1]/H), 0.5*z+0.1, z],1)[None]\", \"x=pc[:,0]; y=pc[:,1]; z=pc[:,2]\\n clip=torch.stack([x*(2*K[0,0]/W), y*(2*K[1,1]/H), 0.5*z+0.1, z],1)[None]\")
open(p,'w').write(s)
PY
python /work/scratch/render_test.py"
```
<details><summary>output</summary>

```
0 6130 baseline
100 5419 baseline
200 5739 baseline
240 8154 baseline
300 8423 baseline
400 8539 baseline
500 10194 baseline
600 9192 baseline
700 9316 baseline
740 9301 baseline
775 10550 baseline
800 9998 baseline
850 9365 baseline
921 9429 baseline

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np,glob
ims=[]
for i in [0,100,200,240,300,400,500,600,700,740,775,800,850,921]:
 im=cv2.imread(f'/work/scratch/overlay_{i}.jpg'); ims.append(cv2.copyMakeBorder(im,0,0,0,0,cv2.BORDER_CONSTANT))
while len(ims)%5: ims.append(np.zeros_like(ims[0]))
cv2.imwrite('/work/scratch/overlays.jpg',np.vstack([np.hstack(ims[i:i+5]) for i in range(0,len(ims),5)]))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_compare.py <<'PY'
exec(open('/work/scratch/render_test.py').read().split('cap=cv2.VideoCapture')[0])
cap=cv2.VideoCapture('/work/inputs/video.mp4'); D=np.load('/work/scratch/diag.npz'); bb=D['bb']
ims=[]
for i in [0,50,100,150,200,240,300,400,500,600,700,740,775,800,850,921]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read(); m,r=render(i)
 x0,y0,x1,y1=bb[i]; pad=70;a=max(0,x0-pad);b=min(W,x1+pad);c=max(0,y0-pad);d=min(H,y1+pad)
 orig=im[c:d,a:b].copy(); ren=np.zeros_like(orig); mr=m[c:d,a:b]; ren[mr]=(r[c:d,a:b][mr]*255).astype(np.uint8)
 # add silhouette line
 cnt,_=cv2.findContours(mr.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ren,cnt,-1,(0,255,0),1)
 oo=cv2.resize(orig,(300,300));rr=cv2.resize(ren,(300,300));cv2.putText(oo,str(i),(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
 ims.append(np.hstack([oo,rr]))
cv2.imwrite('/work/scratch/render_compare.jpg',np.vstack([np.hstack(ims[i:i+4]) for i in range(0,len(ims),4)]))
PY
python /work/scratch/render_compare.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/render_compare.py /work/scratch/render_compare_flip.py && sed -i 's/uv\\[0,...,1\\]\\*2-1/1-uv[0,...,1]*2/' /work/scratch/render_test.py && python /work/scratch/render_compare.py && cp /work/scratch/render_compare.jpg /work/scratch/render_compare_flipy.jpg"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np
p='/pipeline/depth/moge2/aligned.h5'
with h5py.File(p,'r') as f:
 for g in ['aligned','raw','alignment']:
  print(g,dict(f[g].attrs))
  print('group keys',list(f[g].keys()))
  for k in f[g]:
   x=f[g][k];print(' ',k,type(x),getattr(x,'shape',None),getattr(x,'dtype',None),dict(x.attrs))
   if k=='front_stereo_camera_left':
    print('   first',x[0] if not hasattr(x[0],'shape') else (x[0].shape,x[0].dtype))
PY"
```
<details><summary>output</summary>

```
aligned {}
group keys ['front_stereo_camera_left']
  front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (922,) object {}
   first ((649789,), dtype('uint8'))
raw {}
group keys ['front_stereo_camera_left']
  front_stereo_camera_left <class 'h5py._hl.dataset.Dataset'> (922,) object {}
   first ((598854,), dtype('uint8'))
alignment {}
group keys ['front_stereo_camera_left']
  front_stereo_camera_left <class 'h5py._hl.group.Group'> None None {}
Traceback (most recent call last):
  File "<stdin>", line 10, in <module>
  File "/opt/conda/lib/python3.11/site-packages/h5py/_hl/group.py", line 407, in __getitem__
    return self._get(name)
           ^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects …[+247 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,io
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 x=f['aligned/front_stereo_camera_left'][0]; print(type(x),x.shape,x[:20])
 for typ in ['npy','npz']:
  try:
   if typ=='npy': a=np.load(io.BytesIO(x.tobytes()))
   else: a=np.load(io.BytesIO(x.tobytes()))
   print(typ,a.shape,a.dtype,a.min(),a.max());break
  except Exception as e: print(typ,e)
 print('align keys',list(f['alignment/front_stereo_camera_left'].keys()))
 for k in f['alignment/front_stereo_camera_left']:
  print(k,f['alignment/front_stereo_camera_left'][k].shape,f['alignment/front_stereo_camera_left'][k][0])
PY"
```
<details><summary>output</summary>

```
<class 'numpy.ndarray'> (649789,) [137  80  78  71  13  10  26  10   0   0   0  13  73  72  68  82   0   0
   6   0]
npy Cannot load file containing pickled data when allow_pickle=False
npz Cannot load file containing pickled data when allow_pickle=False
align keys ['scale', 'shift', 'valid_count']
scale (922,) 1.1033392
shift (922,) 0.0
valid_count (922,) 45696

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,numpy as np,cv2
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:
 for g in ['aligned','raw']:
  x=f[g+'/front_stereo_camera_left'][0]; im=cv2.imdecode(x,cv2.IMREAD_UNCHANGED);print(g,im.shape,im.dtype,im.min(),im.max(),np.percentile(im,[1,50,99]))
PY"
```
<details><summary>output</summary>

```
aligned (1152, 1536) uint16 1767 9901 [1850. 3933. 8986.]
raw (1152, 1536) uint16 1602 8974 [1677. 3565. 8145.]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/pose_utils.py <<'PY'
import numpy as np, torch, trimesh, cv2, json
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
DEV='cuda'; H,W=1152,1536
_geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=torch.tensor(_geo.vertices,dtype=torch.float32,device=DEV)
F=torch.tensor(_geo.faces,dtype=torch.int32,device=DEV)
UV=torch.tensor(_geo.visual.uv,dtype=torch.float32,device=DEV)
TEX=torch.tensor(np.asarray(_geo.visual.material.baseColorTexture),dtype=torch.float32,device=DEV).permute(2,0,1)[None]/255
ctx=dr.RasterizeCudaContext()
with open('/work/inputs/camera.json') as f: _cam=json.load(f)
K=np.array(_cam['K'],np.float32); Twc=np.array(_cam['T_world_cam'],np.float32); Tcw=np.linalg.inv(Twc).astype(np.float32)

def render_batch(Ts, x0,y0,w,h):
    # Ts [N,4,4] world_mesh
    Ts=torch.as_tensor(Ts,dtype=torch.float32,device=DEV)
    N=len(Ts)
    vh=torch.cat([V,torch.ones_like(V[:,:1])],1)
    pw=(Ts@vh.T).transpose(1,2) # N,V,4
    Tcw_t=torch.tensor(Tcw,device=DEV)
    pc=(Tcw_t[None]@pw).transpose(1,2)
    X,Y,Z=pc[:,:,0],pc[:,:,1],pc[:,:,2]
    clip=torch.stack([
        2*(K[0,0]*X+(K[0,2]-x0)*Z)/w-Z,
        2*(K[1,1]*Y+(K[1,2]-y0)*Z)/h-Z,
        0.5*Z+0.1,
        Z],-1)
    rast,rb=dr.rasterize(ctx,clip,F,resolution=[h,w])
    uv,_=dr.interpolate(UV[None].expand(N,-1,-1),rast,F,rb)
    mask=rast[:,:,:,3]>0
    texn=torch.stack([uv[...,0]*2-1, 1-uv[...,1]*2],-1)
     …[+3601 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_candidates.py", line 7, in <module>
    Hs=[np.eye(4,np.float32)]
        ^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/twodim_base.py", line 211, in eye
    m = zeros((N, M), dtype=dtype, order=order)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'type' object cannot be interpreted as an integer

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/np.eye(4,np.float32)/np.eye(4,dtype=np.float32)/g' /work/scratch/test_candidates.py && python /work/scratch/test_candidates.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/test_candidates.py", line 24, in <module>
    s,d=score(Ts[i]@H,im,oms[i],pms[i],bb[i],True); vals.append((s,n,d['iou'],d['inten'],d['grad']))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_utils.py", line 51, in score
    mm,rr,_=render_batch(Ts,x0,y0,w,h)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_utils.py", line 21, in render_batch
    pc=(Tcw_t[None]@pw).transpose(1,2)
        ~~~~~~~~~~~^~~
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [1, 4] but got: [1, 293289].

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/pc=(Tcw_t\\[None\\]@pw).transpose(1,2)/pc=pw@Tcw_t.T/' /work/scratch/pose_utils.py && python /work/scratch/test_candidates.py"
```
<details><summary>output</summary>

```
0 flipxy:1.88(iou0.92,i-0.05,g0.04) | flipyz:1.87(iou0.92,i-0.00,g0.04) | flipxz:1.86(iou0.92,i-0.11,g0.03) | flipz:1.86(iou0.92,i-0.05,g0.03) | flipy:1.86(iou0.92,i-0.05,g0.03) | flipx:1.86(iou0.92,i-0.05,g0.03) | id:1.85(iou0.92,i-0.05,g0.02)
50 flipyz:1.87(iou0.92,i-0.02,g0.04) | flipxy:1.86(iou0.91,i-0.06,g0.03) | id:1.85(iou0.91,i-0.03,g0.03) | flipxz:1.84(iou0.91,i-0.11,g0.03) | flipz:1.84(iou0.91,i-0.06,g0.02) | flipy:1.84(iou0.91,i-0.06,g0.02) | flipx:1.84(iou0.91,i-0.06,g0.02)
100 flipyz:1.71(iou0.82,i0.10,g0.02) | flipz:1.70(iou0.82,i0.07,g0.03) | flipy:1.70(iou0.82,i0.07,g0.03) | flipx:1.70(iou0.82,i0.07,g0.03) | flipxy:1.67(iou0.82,i0.02,g0.02) | id:1.67(iou0.81,i0.04,g0.03) | flipxz:1.63(iou0.80,i-0.03,g0.03)
150 flipyz:1.84(iou0.90,i0.07,g0.01) | flipxz:1.81(iou0.90,i-0.13,g0 …[+3152 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,json
from scipy.spatial.transform import Rotation as R
z=np.load('/work/inputs/reconstruction.npz'); B=z['T_world_mesh'].astype(float); F=z['T_cam_mesh_foundationpose'].astype(float); C=z['T_cam_mesh_coconet'].astype(float)
cam=json.load(open('/work/inputs/camera.json'));Twc=np.array(cam['T_world_cam']);Tcw=np.linalg.inv(Twc);K=np.array(cam['K_pipeline'])
Bc=Tcw@B
for name,A in [('base',Bc),('fp',F),('coco',C)]:
 print(name,'z',np.percentile(A[:,2,3],[0,5,25,50,75,95,100]))
# rot diff helper
def rd(A,B):
 Rm=np.einsum('nij,nkj->nik',A[:,:3,:3],B[:,:3,:3]);return np.degrees(np.arccos(np.clip((np.trace(Rm,axis1=1,axis2=2)-1)/2,-1,1)))
for a,b,n in [(Bc,F,'bf'),(Bc,C,'bc'),(F,C,'fc')]:
 d=rd(a,b); print(n,'ang',np.percentile(d,[0,25,50,75,90,95,99,100]))
for i in range(0,922,50):
 print(i,'basecam',np.round(Bc[i,:3,3],3),'fp',np.round(F[i,:3,3],3),'coco',np.round(C[i,:3,3],3),'angs',round(rd(Bc[i:i+1],F[i:i+1])[0],1),round(rd(Bc[i:i+1],C[i:i+1])[0],1))
# relative frame to previous rotation/depth for fp
for n,A in [('base',Bc),('fp',F),('coco',C)]:
 dr=np.r_[0,rd(A[1:],A[:-1])];dt=np.r_[0,np.linalg.norm(np.diff(A[:,:3,3],axis=0),axis=1)]
 print('\\n',n,'dR percent',np.percentile(dr,[50,75,90,95,99,100]),'dT',np.percentile(dt,[50,75,90,95,99,100]))
 print('max dr idx',np.argsort(dr)[-15:][::-1],np.round(dr[np.argsort(dr)[-15:][::-1]],1))
PY"
```
<details><summary>output</summary>

```
base z [3.09972652 3.17695242 3.27726248 3.32616402 3.41996651 3.53852068
 3.6775984 ]
fp z [2.85291171 3.20111494 3.32695866 3.38429117 3.5370059  4.10913732
 7.33368492]
coco z [3.06958842 3.13247924 3.215478   3.2891686  3.32564533 3.42255017
 3.52807522]
bf ang [  0.36376266   1.55734881   2.3569984    7.74249877  20.95643069
  63.69238326 175.06218857 179.6585098 ]
bc ang [0.95935571 0.97338704 1.10600067 1.30755627 1.31838289 1.35361454
 1.38580884 1.39173976]
fc ang [  0.19210019   1.22573621   1.96543763   8.26303202  20.93825656
  63.26727193 174.99101914 179.88834408]
0 basecam [-0.136  0.084  3.285] fp [-0.182  0.027  3.333] coco [-0.177  0.023  3.264] angs 1.2 1.3
50 basecam [-0.137  0.085  3.302] fp [-0.178  0.027  3.227] coco [-0.177  0.023  3.264] angs 1.2 1.3
100 basecam [- …[+2655 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/crops_range.py <<'PY'
import cv2,numpy as np,h5py,sys
cap=cv2.VideoCapture('/work/inputs/video.mp4'); D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f: om=f['object'][:]
def make(a,b,step,name):
 out=[];cap.set(cv2.CAP_PROP_POS_FRAMES,a)
 for i in range(a,b):
  ok,im=cap.read()
  if (i-a)%step: continue
  x0,y0,x1,y1=bb[i];cx=(x0+x1)//2;cy=(y0+y1)//2;pad=max(55,int(max(x1-x0,y1-y0)*.7))
  xa=max(0,cx-pad);xb=min(im.shape[1],cx+pad);ya=max(0,cy-pad);yb=min(im.shape[0],cy+pad)
  cr=im[ya:yb,xa:xb].copy();m=om[i,ya:yb,xa:xb];cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(cr,cnt,-1,(0,0,255),1)
  cr=cv2.resize(cr,(200,200));cv2.rectangle(cr,(0,0),(50,20),(0,0,0),-1);cv2.putText(cr,str(i),(3,16),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);out.append(cr)
 while len(out)%10:out.append(np.zeros_like(out[0]))
 rows=[np.hstack(out[j:j+10]) for j in range(0,len(out),10)];cv2.imwrite('/work/scratch/'+name+'.jpg',np.vstack(rows));print(name,len(out))
make(180,321,4,'dyn_180_320');make(700,821,3,'dyn_700_820');make(80,181,3,'dyn_80_180');make(560,701,4,'dyn_560_700');make(800,922,6,'late_800_921')
PY
python /work/scratch/crops_range.py"
```
<details><summary>output</summary>

```
dyn_180_320 40
dyn_700_820 50
dyn_80_180 40
dyn_560_700 40
late_800_921 30

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/montage_flips.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
Z=np.load('/work/inputs/reconstruction.npz');T=Z['T_world_mesh'].astype(np.float32);D=np.load('/work/scratch/diag.npz');bb=D['bb']
Hs=[];names=[]
for mask in range(8):
 H=np.eye(4,dtype=np.float32)
 for j in range(3):
  if mask>>j&1:H[j,j]=-1
 Hs.append(H); names.append(''.join('xyz'[j] for j in range(3) if mask>>j&1) or 'id')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,240,321,500,692,740,800,850,921]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 arr=[]
 for n,H in zip(names,Hs):
  s,d=score(T[i]@H,im,oms[i],pms[i],bb[i],True);m=d['mask'];r=d['rgb'];x0,y0,w,h=d['crop']
  ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(ren,cnt,-1,(0,255,0),1)
  ren=cv2.resize(ren,(160,160));cv2.rectangle(ren,(0,0),(100,20),(0,0,0),-1);cv2.putText(ren,f'{n} {s:.2f}',(2,15),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);arr.append(ren)
 # original
 x0,y0,w,h=crop_box(bb[i]);orig=im[y0:y0+h,x0:x0+w];orig=cv2.resize(orig,(160,160));cv2.putText(orig,str(i),(3,15),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1)
 cv2.imwrite(f'/work/scratch/flips_{i}.jpg',np.hstack([orig]+arr))
PY
python /work/s …[+214 chars]
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2,numpy as np
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,921);ok,im=cap.read();D=np.load('/work/scratch/diag.npz');bb=D['bb'][921];print(bb)
x0,y0,x1,y1=bb;pad=50;cv2.imwrite('/work/scratch/frame921_crop.jpg',im[y0-pad:y1+pad,x0-pad:x1+pad])
PY"
```
<details><summary>output</summary>

```
[680 552 806 619]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('/work/scratch/diag.npz')
for i in [0,240,321,500,692,740,800,850,921]:print(i,'mask',D['cent'][i],D['bb'][i],'uv',D['uv'][i],'t',D['cen'][i],'area',D['areas'][i])
PY"
```
<details><summary>output</summary>

```
0 mask [718.74697784 583.93804567] [671 552 769 616] uv [803.99464116 639.56701618] t [0.02511301 0.20299922 3.32312417] area 5956
240 mask [595.28582202 693.16968326] [577 654 615 744] uv [651.94181762 754.46189689] t [-0.62682486  0.7071622   3.49576855] area 1326
321 mask [668.31591007 804.94271803] [611 591 721 848] uv [752.2407501  855.44132709] t [-0.19255896  1.11887634  3.4620676 ] area 6983
500 mask [811.32051028 811.21934335] [746 763 877 862] uv [895.22665081 869.24228009] t [0.36940372 1.06705928 3.15547848] area 10112
692 mask [810.22146267 814.13997396] [744 765 878 862] uv [893.33260392 872.09671968] t [0.38094348 1.13016236 3.31367707] area 9216
740 mask [848.20269815 750.7539629 ] [803 692 896 802] uv [955.94922652 808.74161438] t [0.62087601 0.86186832 3.25694013] area 59 …[+402 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');from pose_utils import *
Z=np.load('/work/inputs/reconstruction.npz');T=Z['T_world_mesh']
for i in [0,921]:
 pw=T[i]@np.r_[V.cpu().numpy().mean(0),1];pc=Tcw@pw;u=K[0,0]*pc[0]/pc[2]+K[0,2];v=K[1,1]*pc[1]/pc[2]+K[1,2]
 print(i,'meshmean',V.cpu().numpy().mean(0),'Tt',T[i,:3,3],'pw',pw,'pc',pc,'uv',u,v)
PY"
```
<details><summary>output</summary>

```
0 meshmean [-0.00199131  0.00799177 -0.00497137] Tt [0.02511301 0.20299922 3.3231242 ] pw [0.02609082 0.19541332 3.3172894  1.        ] pc [-0.1345934   0.07703781  3.2786744   1.        ] uv 724.4381 579.7672
921 meshmean [-0.00199131  0.00799177 -0.00497137] Tt [0.12337284 0.19974597 3.3872325 ] pw [0.12607001 0.20386742 3.378969   1.        ] pc [-0.03737001  0.08280793  3.3449013   1.        ] uv 748.93884 580.8003

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,25p' /work/scratch/diag.py"
```
<details><summary>output</summary>

```
import numpy as np,h5py,cv2,json
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import binary_erosion
z=np.load('/work/inputs/reconstruction.npz')
T=z['T_world_mesh'].astype(float); Tf=z['T_cam_mesh_foundationpose'].astype(float); Tc=z['T_cam_mesh_coconet'].astype(float)
h=np.load('/work/inputs/human_keypoints_world.npz') if False else None
kp=z['human_keypoints_world']; cp=z['contact_prob_right_left']
with open('/work/inputs/camera.json') as cf:
 c=json.load(cf); K=np.array(c['K']); Twc=np.array(c['T_world_cam']); Kp=np.array(c['K_pipeline'])
with h5py.File('/work/inputs/masks.h5') as f:
 om=f['object'][:]; pm=f['person'][:]
areas=om.sum((1,2)); personareas=pm.sum((1,2))
ys,xs=np.where(om.reshape(len(om),-1))
# np.where flattened but per frame need calculate
cent=np. …[+620 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/render_flip_big.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
Z=np.load('/work/inputs/reconstruction.npz');T=Z['T_world_mesh'].astype(np.float32);D=np.load('/work/scratch/diag.npz');bb=D['bb']
i=921;cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
arr=[]
for mask in range(8):
 H=np.eye(4,dtype=np.float32)
 for j in range(3):
  if mask>>j&1:H[j,j]=-1
 s,d=score(T[i]@H,im,*[np.array([])]*0) if False else (None,None)
 # directly render into larger crop
 x0,y0,w,h=crop_box(bb[i],pad=65,min_size=300)
 mm,rr,_=render_batch((T[i]@H)[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy()
 ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);ren=cv2.cvtColor(ren,cv2.COLOR_RGB2BGR);cv2.putText(ren,''.join('xyz'[j] for j in range(3) if mask>>j&1) or 'id',(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2);arr.append(ren)
orig=im[y0:y0+h,x0:x0+w];orig=cv2.cvtColor(orig,cv2.COLOR_RGB2BGR);cv2.putText(orig,str(i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
cv2.imwrite('/work/scratch/flip_big_921.jpg',np.hstack([orig]+arr))
PY
python /work/scratch/render_flip_big.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/depth_search.py <<'PY'
import numpy as np,h5py,cv2,sys,json
sys.path.insert(0,'/work/scratch');from pose_utils import *
Z=np.load('/work/inputs/reconstruction.npz');Ts=Z['T_world_mesh'].astype(np.float32);D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,321,500,692,800,850,921]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 T=Ts[i].copy();cam=Tcw@T; center=cam[:3,3]; z0=center[2]
 vals=[]
 for z in np.arange(z0-0.4,z0+0.9,0.02):
  Tn=T.copy();xc=center[:2]*z/z0;Tn[:3,3]=(Twc@np.r_[xc,z,1])[:3]
  s,d=score(Tn,im,oms[i],pms[i],bb[i],True); vals.append((z,d['iou'],s,d['inten'],d['grad']))
 best=max(vals,key=lambda x:x[1]); print(i,'basez',z0,'best',best,'top',sorted(vals,key=lambda x:x[1],reverse=True)[:5])
PY
python /work/scratch/depth_search.py"
```
<details><summary>output</summary>

```
0 basez 3.2847762 best (3.2447762107849125, 0.9207671749724886, 1.869764292490272, -0.047714688398454295, 0.02822994254529476) top [(3.2447762107849125, 0.9207671749724886, 1.869764292490272, -0.047714688398454295, 0.02822994254529476), (3.2647762107849125, 0.9202715503631197, 1.864170827087395, -0.05685042490085576, 0.02362772636115551), (3.2847762107849126, 0.9159677164108245, 1.853149642318624, -0.05170424774348956, 0.021214209496974945), (3.3647762107849126, 0.9152212676284649, 1.8438680633287505, -0.04818483941919394, 0.013425528071820736), (3.3047762107849126, 0.9147002551020408, 1.852867134523635, -0.02412031346020933, 0.023466624319553375)]
321 basez 3.4527137 best (3.17271372795105, 0.8459093421779988, 1.7029293496326443, -0.2446901569432724, 0.011110665276646614) top [(3.17271372 …[+3811 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/fit_mask.py <<'PY'
import numpy as np,h5py,cv2,sys,json
from scipy.optimize import differential_evolution, minimize
sys.path.insert(0,'/work/scratch');from pose_utils import *
from scipy.spatial.transform import Rotation as Rot
Z=np.load('/work/inputs/reconstruction.npz');Ts=Z['T_world_mesh'].astype(np.float32);D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
# bbox local corners
lo,hi=_geo.bounds
corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
def pose_from(R,t):
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=t;return T
def project_corners(R,z,du,dv,center0,uv0):
 # center in cam along ray from baseline and pixel offset
 X=(uv0[0]+du-K[0,2])*z/K[0,0];Y=(uv0[1]+dv-K[1,2])*z/K[1,1];tcam=np.array([X,Y,z])
 Tcam=pose_from(R,tcam);Tw=Twc@Tcam
 pc=(Tcw@(Tw@corners.T)).T
 u=K[0,0]*pc[:,0]/pc[:,2]+K[0,2];v=K[1,1]*pc[:,1]/pc[:,2]+K[1,2]
 return np.c_[u,v]
def raster(poly,x0,y0,w,h):
 m=np.zeros((h,w),np.uint8);p=np.round(poly-[x0,y0]).astype(np.int32);cv2.fillConvexPoly(m,cv2.convexHull(p),1);return m.astype(bool)
def fit(i,ret=False):
 bb_i=bb[i];x0,y0,w,h=crop_box(bb_i,pad=25,min_size=240);o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w]
 # suppress person
 vis=~cv2.dilate(p.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
 T=Ts[i];R0=T[:3,:3].copy();cam=Tcw@T;center=cam[:3,3];uv0=np.array([K[0,0]*center[0]/center[2]+K[0,2 …[+1120 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/fit_mask.py", line 8, in <module>
    lo,hi=_geo.bounds
          ^^^^
NameError: name '_geo' is not defined

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/fit_mask.py';s=open(p).read();s=s.replace(\"lo,hi=_geo.bounds\", \"import trimesh\\ngeo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()\\nlo,hi=geo.bounds\")
open(p,'w').write(s)
PY
python /work/scratch/fit_mask.py"
```
<details><summary>output</summary>

```
fit 0 0.08016219588271989 0.9198378041172801 [ 3.40359552e+00  1.95727670e-03  2.01512997e-03  5.67061018e-02
  3.37911320e-04 -1.30167233e-03]
fit 321 0.10659144893111638 0.8934085510688836 [ 3.25142583e+00  4.44699113e-04  1.96201847e-03 -1.54991806e-04
 -2.78593509e-01  6.29521578e-05]
fit 500 0.046961594555177455 0.9530384054448225 [ 3.23759657e+00  8.84808767e-04  1.48422839e-03  4.13914175e-04
 -8.26281708e-02  4.37378054e-04]
fit 692 0.030208449448222097 0.9697915505517779 [ 3.20189525e+00 -1.14645646e-05  3.10076227e-03  3.16309494e-05
 -6.50586452e-02  1.11252518e-03]
fit 800 0.45698510529019 0.54301489470981 [ 3.40611555e+00  5.10798537e-04  8.03993937e-04  1.34933496e-03
  1.07572076e-03 -9.58724354e-02]
fit 850 0.3086139896373057 0.6913860103626943 [ 3.72416260e+00  8.42227818e …[+215 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat >> /work/scratch/fit_mask.py <<'PY'
PY
python - <<'PY'
# reuse definitions by exec excluding loop
s=open('/work/scratch/fit_mask.py').read().split('for i in [')[0]
exec(s)
import cv2
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [321,500,692,850,921]:
 Tnew,b=fit(i,True);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 s,d=score(Tnew,im,oms[i],pms[i],bb[i],True)
 print('fullscore',i,s,d['iou'],d['inten'],d['grad'])
 mm,rr,_=render_batch(Tnew[None],*d['crop']);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();x0,y0,w,h=d['crop'];ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);cv2.imwrite(f'/work/scratch/fitted_{i}.jpg',cv2.cvtColor(np.hstack([cv2.cvtColor(im[y0:y0+h,x0:x0+w],cv2.COLOR_RGB2BGR),cv2.cvtColor(ren,cv2.COLOR_RGB2BGR)]),cv2.COLOR_RGB2BGR))
PY"
```
<details><summary>output</summary>

```
fit 321 0.10659144893111638 0.8934085510688836 [ 3.25142583e+00  4.44699113e-04  1.96201847e-03 -1.54991806e-04
 -2.78593509e-01  6.29521578e-05]
fullscore 321 1.7689873133344154 0.8795620437956204 -0.2417661498265524 0.009863225743174553
fit 500 0.046961594555177455 0.9530384054448225 [ 3.23759657e+00  8.84808767e-04  1.48422839e-03  4.13914175e-04
 -8.26281708e-02  4.37378054e-04]
fullscore 500 1.8584021592451103 0.927432059866089 -0.0874584658354785 0.0035380395129323006
fit 692 0.030208449448222097 0.9697915505517779 [ 3.20189525e+00 -1.14645646e-05  3.10076227e-03  3.16309494e-05
 -6.50586452e-02  1.11252518e-03]
fullscore 692 1.9158970875954595 0.946969696969697 0.017032152110891277 0.013441617600619793
fit 850 0.3086139896373057 0.6913860103626943 [ 3.72416260e+00  8.42227818e-05  2 …[+395 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
for i in [500,692,321]:
 a=cv2.imread(f'/work/scratch/fitted_{i}.jpg')
 cv2.imwrite(f'/work/scratch/check_{i}.png',a)
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/texflip_test.py <<'PY'
import numpy as np,torch,cv2,sys,trimesh
sys.path.insert(0,'/work/scratch');import pose_utils as pu
Z=np.load('/work/inputs/reconstruction.npz');T=Z['T_world_mesh'].astype(np.float32);D=np.load('/work/scratch/diag.npz');bb=D['bb'];i=500
cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=pu.crop_box(bb[i],pad=30,min_size=260)
arr=[]
for fu in [0,1]:
 for fv in [0,1]:
  Ts=torch.tensor(T[i:i+1],device=pu.DEV)
  vh=torch.cat([pu.V,torch.ones_like(pu.V[:,:1])],1);pw=(Ts@vh.T).transpose(1,2);pc=pw@torch.tensor(pu.Tcw,device=pu.DEV).T
  X,Y,Zz=pc[:,:,0],pc[:,:,1],pc[:,:,2]
  clip=torch.stack([2*(pu.K[0,0]*X+(pu.K[0,2]-x0)*Zz)/w-Zz,2*(pu.K[1,1]*Y+(pu.K[1,2]-y0)*Zz)/h-Zz,.5*Zz+.1,Zz],-1)
  rast,rb=pu.dr.rasterize(pu.ctx,clip,pu.F,resolution=[h,w]);uv,_=pu.dr.interpolate(pu.UV[None],rast,pu.F,rb);m=rast[0,:,:,3]>0
  uu=uv[...,0]*2-1;vv=uv[...,1]*2-1
  if fu:uu=-uu
  if fv:vv=-vv
  texn=torch.stack([uu,vv],-1);rgb=torch.nn.functional.grid_sample(pu.TEX,texn,align_corners=False,mode='bilinear',padding_mode='border')[0].permute(1,2,0).cpu().numpy();r=np.zeros((h,w,3),np.uint8);r[m]=(rgb[m]*255).astype(np.uint8);r=cv2.cvtColor(r,cv2.COLOR_RGB2BGR);cv2.putText(r,f'u{fu}v{fv}',(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2);arr.append(r)
orig=cv2.cvtColor(im[y0:y0+h,x0:x0+w],cv2.COLOR_RGB2BGR);cv2.putText(orig,str(i),(5,25),cv2.FONT_HER …[+133 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/texflip_test.py", line 16, in <module>
    texn=torch.stack([uu,vv],-1);rgb=torch.nn.functional.grid_sample(pu.TEX,texn,align_corners=False,mode='bilinear',padding_mode='border')[0].permute(1,2,0).cpu().numpy();r=np.zeros((h,w,3),np.uint8);r[m]=(rgb[m]*255).astype(np.uint8);r=cv2.cvtColor(r,cv2.COLOR_RGB2BGR);cv2.putText(r,f'u{fu}v{fv}',(5,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2);arr.append(r)
                                                                                                                                                                                                               ~~~^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/_tensor.py", line 1149, in __array__
    return self.numpy()
           ^^ …[+130 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/;m=rast\\[0,:,:,3\\]>0/;m=rast[0,:,:,3].cpu().numpy()>0/' /work/scratch/texflip_test.py && python /work/scratch/texflip_test.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import struct,json
p='/work/inputs/object_mesh.glb';b=open(p,'rb').read();magic,ver,n=struct.unpack_from('<III',b,0);off=12
for i in range(n):
 ln,typ=struct.unpack_from('<II',b,off);off+=8;data=b[off:off+ln];off+=ln
 if typ==0x4E4F534A:
  j=json.loads(data); print(json.dumps({k:j[k] for k in j if k not in ['buffers','bufferViews','accessors']},indent=1)[:20000])
PY"
```
<details><summary>output</summary>

```
{
 "scene": 0,
 "scenes": [
  {
   "nodes": [
    0
   ]
  }
 ],
 "asset": {
  "version": "2.0",
  "generator": "https://github.com/mikedh/trimesh"
 },
 "meshes": [
  {
   "name": "snackbox.001",
   "extras": {
    "units": "meters",
    "from_gltf_primitive": false,
    "name": "snackbox.001",
    "node": "snackbox"
   },
   "primitives": [
    {
     "attributes": {
      "POSITION": 1,
      "TEXCOORD_0": 2
     },
     "indices": 0,
     "mode": 4,
     "material": 0
    }
   ]
  }
 ],
 "images": [
  {
   "bufferView": 2,
   "mimeType": "image/png"
  }
 ],
 "textures": [
  {
   "source": 0
  }
 ],
 "materials": [
  {
   "pbrMetallicRoughness": {
    "baseColorTexture": {
     "index": 0
    },
    "metallicFactor": 0.0
   },
   "name": "_texture.001",
   "doubleSided": true
  }
 ],
 "n …[+337 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
from scipy.spatial.transform import Rotation as Rot
Z=np.load('/work/inputs/reconstruction.npz');T=Z['T_world_mesh'];D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:om=f['object'][:];pm=f['person'][:]
cap=cv2.VideoCapture('/work/inputs/video.mp4')
for i in [0,500,850,921]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 for zadd,rz in [(0,0),(.3,0),(.3,.56)]:
  Tn=T[i].copy(); cam=Tcw@Tn;z=cam[2,3]; x=cam[0,3]*(z+zadd)/z;y=cam[1,3]*(z+zadd)/z;Tn[:3,3]=(Twc@np.r_[x,y,z+zadd,1])[:3]
  if rz:
   H=np.eye(4);H[:3,:3]=Rot.from_rotvec([0,0,rz]).as_matrix();Tn=Tn@H
  mm,rr,_=render_batch(Tn[None],*crop_box(bb[i],pad=20,min_size=220));m=mm[0].cpu().numpy();ys,xs=np.where(m);ys2,xs2=np.where(om[i])
  print(i,'dz',zadd,'rz',rz,'render',m.sum(),(xs.min(),ys.min(),xs.max(),ys.max()),(xs.max()-xs.min()+1,ys.max()-ys.min()+1),'obs',om[i].sum(),(xs2.min(),ys2.min(),xs2.max(),ys2.max()),(xs2.max()-xs2.min()+1,ys2.max()-ys2.min()+1))
PY"
```
<details><summary>output</summary>

```
0 dz 0 rz 0 render 6151 (64, 79, 162, 143) (99, 65) obs 5956 (671, 552, 769, 616) (99, 65)
0 dz 0.3 rz 0 render 5160 (68, 81, 158, 140) (91, 60) obs 5956 (671, 552, 769, 616) (99, 65)
0 dz 0.3 rz 0.56 render 5152 (63, 66, 164, 156) (102, 91) obs 5956 (671, 552, 769, 616) (99, 65)
500 dz 0 rz 0 render 10201 (46, 60, 178, 163) (133, 104) obs 10112 (746, 763, 877, 862) (132, 100)
500 dz 0.3 rz 0 render 8485 (52, 63, 172, 157) (121, 95) obs 10112 (746, 763, 877, 862) (132, 100)
500 dz 0.3 rz 0.56 render 8933 (48, 52, 177, 169) (130, 118) obs 10112 (746, 763, 877, 862) (132, 100)
850 dz 0 rz 0 render 9361 (63, 51, 167, 165) (105, 115) obs 7607 (680, 552, 806, 619) (127, 68)
850 dz 0.3 rz 0 render 7889 (67, 55, 163, 160) (97, 106) obs 7607 (680, 552, 806, 619) (127, 68)
850 dz 0.3 rz 0.56 render …[+370 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fit_mask_global.py <<'PY'
import numpy as np,h5py,cv2,sys,json,trimesh
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from pose_utils import *
Z=np.load('/work/inputs/reconstruction.npz');Ts=Z['T_world_mesh'].astype(np.float32);D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();lo,hi=geo.bounds
corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
def make_T(R,z,du,dv,uv0):
 tcam=np.array([(uv0[0]+du-K[0,2])*z/K[0,0],(uv0[1]+dv-K[1,2])*z/K[1,1],z]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=tcam;return Twc@T
def raster(poly,x0,y0,w,h):
 m=np.zeros((h,w),np.uint8);p=np.round(poly-[x0,y0]).astype(np.int32);cv2.fillConvexPoly(m,cv2.convexHull(p),1);return m.astype(bool)
def setup(i):
 T=Ts[i];cam=Tcw@T;c=cam[:3,3];uv0=np.array([K[0,0]*c[0]/c[2]+K[0,2],K[1,1]*c[1]/c[2]+K[1,2]]);x0,y0,w,h=crop_box(bb[i],pad=25,min_size=240);o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];vis=~cv2.dilate(p.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);return T,uv0,x0,y0,w,h,o,vis
def fitglobal(i,seed=None):
 T,uv0,x0,y0,w,h,o,vis=setup(i);z0=(Tcw@T)[2,3]
 def f(x):
  z,du,dv,rx,ry,rz=x;R=Rot.from_rotvec([rx,ry,rz]).as_matrix().astype(np.float32);Tw=make …[+745 chars]
```
<details><summary>output</summary>

```
global 921 0.9653350515463918 [ 3.45850138 -4.95391567  4.18317817 -1.42974164 -0.15690658  2.63898518]
global 850 0.9662877809351589 [ 3.45923076 -4.87680385  4.03730266 -1.43287794 -0.16495268  2.63238871]
global 800 0.9260596844706329 [  3.24409554 -12.85919407  -9.22104359  -2.47258989   0.21225909
   2.00175024]
global 321 0.9314964225909576 [ 3.64836719 -7.44021108  4.39967416  1.09151805  0.89525467  2.10463859]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/view_global.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
arr=[]
for i in [321,500,692,800,850,921]:
 if i==500:T=np.load('/work/scratch/globalT_321.npy') if False else None
 elif i in [800]:T=np.load('/work/scratch/globalT_800.npy')
 elif i==850:T=np.load('/work/scratch/globalT_850.npy')
 elif i==921:T=np.load('/work/scratch/globalT_921.npy')
 else:
  continue
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 s,d=score(T,im,oms[i],pms[i],bb[i],True);print(i,s,d['iou'],d['inten'],d['grad'])
 mm,rr,_=render_batch(T[None],*d['crop']);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();x0,y0,w,h=d['crop'];ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);ren=cv2.cvtColor(ren,cv2.COLOR_RGB2BGR);orig=cv2.cvtColor(im[y0:y0+h,x0:x0+w],cv2.COLOR_RGB2BGR);arr.append(np.hstack([orig,ren]))
cv2.imwrite('/work/scratch/global_compare.jpg',np.vstack(arr))
PY
python /work/scratch/view_global.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/view_global.py", line 14, in <module>
    s,d=score(T,im,oms[i],pms[i],bb[i],True);print(i,s,d['iou'],d['inten'],d['grad'])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_utils.py", line 51, in score
    mm,rr,_=render_batch(Ts,x0,y0,w,h)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_utils.py", line 19, in render_batch
    pw=(Ts@vh.T).transpose(1,2) # N,V,4
        ~~^~~~~
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1 and 4x293289)

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "head -25 /work/scratch/pose_utils.py; python - <<'PY'
import sys;sys.path.insert(0,'/work/scratch');import pose_utils as p;print(p.V.shape,p.F.shape)
PY"
```
<details><summary>output</summary>

```
torch.Size([293289, 3]) torch.Size([97862, 3])

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/view_global.py';s=open(p).read();s=s.replace(\"for i in [321,500,692,800,850,921]:\\n if i==500:T=np.load('/work/scratch/globalT_321.npy') if False else None\", \"for i in [800,850,921]:\\n if i==500:T=np.load('/work/scratch/globalT_321.npy') if False else None\")
open(p,'w').write(s)
PY
python /work/scratch/view_global.py"
```
<details><summary>output</summary>

```
800 1.7863535272763897 0.8912875867386276 -0.13998411311643905 0.003778353799134493
850 1.864798385841096 0.9291977489857348 -0.03663086840738647 0.006402887869626284
921 1.8500720790075442 0.9233986928104575 -0.03239073039590028 0.003274693386629224

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/global_flips.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
for i in [800,850,921]:
 T=np.load(f'/work/scratch/globalT_{i}.npy');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 vals=[];arr=[]
 for mask in range(8):
  H=np.eye(4,dtype=np.float32)
  for j in range(3):
   if mask>>j&1:H[j,j]=-1
  s,d=score(T@H,im,oms[i],pms[i],bb[i],True);vals.append((s,''.join('xyz'[j] for j in range(3) if mask>>j&1) or 'id',d))
 print(i,sorted([(round(s,3),n,round(d['iou'],3),round(d['inten'],3),round(d['grad'],3)) for s,n,d in vals],reverse=True))
 # montage in sorted order
 for s,n,d in sorted(vals,reverse=True):
  m=d['mask'];r=d['rgb'];x0,y0,w,h=d['crop'];ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);ren=cv2.resize(cv2.cvtColor(ren,cv2.COLOR_RGB2BGR),(180,180));cv2.putText(ren,n,(3,18),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1);arr.append(ren)
 x0,y0,w,h=crop_box(bb[i]);orig=cv2.resize(cv2.cvtColor(im[y0:y0+h,x0:x0+w],cv2.COLOR_RGB2BGR),(180,180));cv2.putText(orig,str(i),(3,18),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1);cv2.imwrite(f'/work/scratch/globalflips_{i}.jpg',np.hstack([orig]+arr))
PY
python /work/scratch/global_flips.py"
```
<details><summary>output</summary>

```
800 [(1.833, 'xy', 0.898, 0.051, 0.012), (1.811, 'z', 0.899, -0.065, 0.014), (1.796, 'xyz', 0.895, -0.114, 0.006), (1.794, 'x', 0.89, -0.139, 0.015), (1.786, 'id', 0.891, -0.14, 0.004), (1.784, 'yz', 0.888, -0.002, 0.009), (1.782, 'xz', 0.883, -0.017, 0.017), (1.764, 'y', 0.882, -0.103, -0.002)]
850 [(1.897, 'y', 0.937, 0.018, 0.014), (1.888, 'xy', 0.926, 0.049, 0.012), (1.886, 'yz', 0.932, -0.02, 0.023), (1.877, 'xz', 0.934, -0.009, 0.008), (1.874, 'x', 0.93, -0.073, 0.013), (1.868, 'xyz', 0.928, -0.028, 0.011), (1.865, 'id', 0.929, -0.037, 0.006), (1.853, 'z', 0.923, -0.017, 0.007)]
921 [(1.882, 'y', 0.932, 0.013, 0.012), (1.874, 'yz', 0.928, -0.02, 0.018), (1.874, 'xy', 0.92, 0.048, 0.01), (1.866, 'xz', 0.929, -0.021, 0.007), (1.86, 'x', 0.927, -0.079, 0.007), (1.851, 'xyz', 0.923, -0.0 …[+84 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/uvnormals.py <<'PY'
import trimesh,numpy as np,cv2
from PIL import Image
g=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();V=g.vertices;F=g.faces;UV=g.visual.uv
n=g.face_normals
# dominant axes
lab=np.argmax(np.abs(n),axis=1);sg=np.sign(n[np.arange(len(n)),lab]).astype(int)
cols=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255]],np.uint8)
tex=np.asarray(g.visual.material.baseColorTexture).copy(); over=tex.copy()
for k in range(3):
 for s in [-1,1]:
  idx=(lab==k)&(sg==s); c=cols[k*2+(s<0)]
  uv=UV[F[idx]]; pts=(uv*[tex.shape[1]-1,tex.shape[0]-1]).astype(np.int32)
  for tri in pts: cv2.fillConvexPoly(over,tri,c)
  ys,xs=pts[:,:,1],pts[:,:,0];print(k,s,idx.sum(),'uv bbox',xs.min(),xs.max(),ys.min(),ys.max())
out=(tex*.55+over*.45).astype(np.uint8);Image.fromarray(out).save('/work/scratch/texture_normals.jpg')
PY
python /work/scratch/uvnormals.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/uvnormals.py", line 13, in <module>
    for tri in pts: cv2.fillConvexPoly(over,tri,c)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.11.0) :-1: error: (-5:Bad argument) in function 'fillConvexPoly'
> Overload resolution failed:
>  - Scalar value for argument 'color' is not numeric
>  - Scalar value for argument 'color' is not numeric


```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/cv2.fillConvexPoly(over,tri,c)/cv2.fillConvexPoly(over,tri,tuple(map(int,c)))/' /work/scratch/uvnormals.py && python /work/scratch/uvnormals.py"
```
<details><summary>output</summary>

```
0 -1 10165 uv bbox 308 2990 5 3058
0 1 8162 uv bbox 4 2976 4 3057
1 -1 25613 uv bbox 5 2974 7 3058
1 1 33798 uv bbox 4 2989 6 2216
2 -1 10260 uv bbox 399 2989 7 2213
2 1 9864 uv bbox 65 2987 6 1525

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/opt_photo.py <<'PY'
import numpy as np,torch,h5py,cv2,sys,time
sys.path.insert(0,'/work/scratch');from pose_utils import *
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0)
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
def so3exp(w):
 th=torch.linalg.norm(w)+1e-12; k=w/th;K=torch.zeros(3,3,device=w.device);K[0,1]=-k[2];K[0,2]=k[1];K[1,0]=k[2];K[1,2]=-k[0];K[2,0]=-k[1];K[2,1]=k[0];return torch.eye(3,device=w.device)+torch.sin(th)*K+(1-torch.cos(th))*(K@K)
def opt(i,Tseed,niter=150,lr=.003):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB).astype(np.float32)/255
 x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);tar=torch.tensor(im[y0:y0+h,x0:x0+w],device=DEV).permute(2,0,1)[None]
 o=torch.tensor(oms[i,y0:y0+h,x0:x0+w],device=DEV,dtype=torch.bool);p=torch.tensor(pms[i,y0:y0+h,x0:x0+w],device=DEV,dtype=torch.bool);vis=~p
 T0=torch.tensor(Tseed,dtype=torch.float32,device=DEV)
 dw=torch.zeros(3,device=DEV,requires_grad=True);dt=torch.zeros(3,device=DEV,requires_grad=True)
 opt=torch.optim.Adam([dw,dt],lr=lr)
 best=(1e9,None)
 for it in range(niter):
  R=so3exp(dw);Tn=T0.clone();Tn[:3,:3]=R@T0[:3,:3];Tn[:3,3]=T0[:3,3]+dt
  m,r,_=render_batch(Tn[None],x0,y0,w,h);m=m[0];r=r[0]
  mm=(m&o)[:,:,None].float()
  # NCC per channel over render/observed overlap, plu …[+2577 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_photo.py", line 51, in <module>
    T,b,m,r,crop=opt(i,T0@H,200)
                 ^^^^^^^^^^^^^^^
  File "/work/scratch/opt_photo.py", line 22, in opt
    a=tar*mm.permute(2,0,1);b=r*mm.permute(2,0,1);n=mm.sum()*3+1e-6
                              ~^~~~~~~~~~~~~~~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (304) at non-singleton dimension 2

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_photo.py';s=open(p).read()
s=s.replace(\"mm=(m&o)[:,:,None].float()\\n  # NCC per channel over render/observed overlap, plus sobel gradients\\n  a=tar*mm.permute(2,0,1);b=r*mm.permute(2,0,1);n=mm.sum()*3+1e-6\\n  am=a.sum((1,2),keepdim=True)/n;bm=b.sum((1,2),keepdim=True)/n\\n  aa=((a-am)**2*mm.permute(2,0,1)).sum();bbv=((b-bm)**2*mm.permute(2,0,1)).sum();ab=((a-am)*(b-bm)*mm.permute(2,0,1)).sum()\", \"ro=r.permute(2,0,1)[None]\\n  wgt=(m&o).float()[None,None]\\n  # NCC per channel over render/observed overlap, plus sobel gradients\\n  a=tar*wgt;b=ro*wgt;n=wgt.sum()*3+1e-6\\n  am=a.sum((1,2),keepdim=True)/n;bm=b.sum((1,2),keepdim=True)/n\\n  aa=((a-am)**2*wgt).sum();bbv=((b-bm)**2*wgt).sum();ab=((a-am)*(b-bm)*wgt).sum()\")
s=s.replace(\"ga=torch.nn.functional.avg_pool2d(tar,3,1,1);gb=torch.nn.functional.avg_pool2d(r,3,1,1);gm=(m&o).float()[:,None]\", \"ga=torch.nn.functional.avg_pool2d(tar,3,1,1);gb=torch.nn.functional.avg_pool2d(ro,3,1,1);gm=(m&o).float()[None,None]\")
s=s.replace(\"gm2=(m&o).float()[:,None,1:,1:]\", \"gm2=(m&o).float()[None,None,1:,1:]\")
open(p,'w').write(s)
PY
python /work/scratch/opt_photo.py 921 id y z xy xz yz xyz x 2>&1 | tee /work/scratch/opt921.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_photo.py", line 52, in <module>
    T,b,m,r,crop=opt(i,T0@H,200)
                 ^^^^^^^^^^^^^^^
  File "/work/scratch/opt_photo.py", line 31, in opt
    dot=(gxa*gxb+gya*gyb)*gm2; mag=(torch.sqrt((gxa*gxa+gya*gya)*(gxb*gxb+gyb*gyb))+1e-6)*gm2
         ~~~~~~~^~~~~~~~
RuntimeError: The size of tensor a (303) must match the size of tensor b (304) at non-singleton dimension 3

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/opt_photo.py';s=open(p).read();old=\"\"\"gxa=ga[:,:,:,1:]-ga[:,:,:,:-1];gxb=gb[:,:,:,1:]-gb[:,:,:,:-1];gya=ga[:,:,1:,:]-ga[:,:,:-1,:];gyb=gb[:,:,1:,:]-gb[:,:,:-1,:]
  gm2=(m&o).float()[None,None,1:,1:]
  dot=(gxa*gxb+gya*gyb)*gm2; mag=(torch.sqrt((gxa*gxa+gya*gya)*(gxb*gxb+gyb*gyb))+1e-6)*gm2
  gscore=(dot/(mag+1e-6)).sum()/(gm2.sum()*3+1e-6)\"\"\"
new=\"\"\"kx=torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]],device=DEV).view(1,1,3,3);ky=kx.transpose(2,3)
  gxa=torch.nn.functional.conv2d(ga,kx,padding=1);gxb=torch.nn.functional.conv2d(gb,kx,padding=1);gya=torch.nn.functional.conv2d(ga,ky,padding=1);gyb=torch.nn.functional.conv2d(gb,ky,padding=1)
  gm2=(m&o).float()[None,None]
  dot=(gxa*gxb+gya*gyb)*gm2; mag=(torch.sqrt(gxa*gxa+gya*gya)*torch.sqrt(gxb*gxb+gyb*gyb)+1e-6)*gm2
  gscore=(dot/(mag+1e-6)).sum()/(gm2.sum()*3+1e-6)\"\"\"
assert old in s;s=s.replace(old,new);open(p,'w').write(s)
PY
python /work/scratch/opt_photo.py 921 id y z xy xz yz xyz x 2>&1 | tee /work/scratch/opt921.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_photo.py", line 53, in <module>
    T,b,m,r,crop=opt(i,T0@H,200)
                 ^^^^^^^^^^^^^^^
  File "/work/scratch/opt_photo.py", line 30, in opt
    gxa=torch.nn.functional.conv2d(ga,kx,padding=1);gxb=torch.nn.functional.conv2d(gb,kx,padding=1);gya=torch.nn.functional.conv2d(ga,ky,padding=1);gyb=torch.nn.functional.conv2d(gb,ky,padding=1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Given groups=1, weight of size [1, 1, 3, 3], expected input[1, 3, 304, 304] to have 1 channels, but got 3 channels instead

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/padding=1)/padding=1,groups=3)/g' /work/scratch/opt_photo.py && python /work/scratch/opt_photo.py 921 id y z xy xz yz xyz x 2>&1 | tee /work/scratch/opt921.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/opt_photo.py", line 53, in <module>
    T,b,m,r,crop=opt(i,T0@H,200)
                 ^^^^^^^^^^^^^^^
  File "/work/scratch/opt_photo.py", line 30, in opt
    gxa=torch.nn.functional.conv2d(ga,kx,padding=1,groups=3);gxb=torch.nn.functional.conv2d(gb,kx,padding=1,groups=3);gya=torch.nn.functional.conv2d(ga,ky,padding=1,groups=3);gyb=torch.nn.functional.conv2d(gb,ky,padding=1,groups=3)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Given groups=3, expected weight to be at least 3 at dimension 0, but got weight of size [1, 1, 3, 3] instead

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/\\.view(1,1,3,3)/.view(1,1,3,3).repeat(3,1,1,1)/\" /work/scratch/opt_photo.py && python /work/scratch/opt_photo.py 921 id y z xy xz yz xyz x 2>&1 | tee /work/scratch/opt921.log"
```
<details><summary>output</summary>

```
921 0 loss -0.9158 ncc 0.915 g 0.001 iou 0.923 dw [-0.003 -0.003 -0.003] dt [ 0.003 -0.003  0.003]
921 25 loss nan ncc 0.0 g 0.0 iou 0.0 dw [nan nan nan] dt [nan nan nan]
921 50 loss nan ncc 0.0 g 0.0 iou 0.0 dw [nan nan nan] dt [nan nan nan]
921 75 loss nan ncc 0.0 g 0.0 iou 0.0 dw [nan nan nan] dt [nan nan nan]
921 100 loss nan ncc 0.0 g 0.0 iou 0.0 dw [nan nan nan] dt [nan nan nan]
921 125 loss nan ncc 0.0 g 0.0 iou 0.0 dw [nan nan nan] dt [nan nan nan]
921 150 loss nan ncc 0.0 g 0.0 iou 0.0 dw [nan nan nan] dt [nan nan nan]
921 175 loss nan ncc 0.0 g 0.0 iou 0.0 dw [nan nan nan] dt [nan nan nan]
FINAL 921 -0.9163080453872681 iou 0.9176255230125523
921 0 loss -0.9298 ncc 0.9 g 0.043 iou 0.932 dw [ 0.003 -0.003 -0.003] dt [-0.003 -0.003  0.003]
921 25 loss -0.8897 ncc 0.895 g -0.007 iou  …[+5629 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/global_big.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];i=921;cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
T0=np.load(f'/work/scratch/globalT_{i}.npy');arr=[]
for mask in [0,1,2,3,4,5,6,7]:
 H=np.eye(4,dtype=np.float32)
 for j in range(3):
  if mask>>j&1:H[j,:3]*=-1
 s,d=score(T0@H,im,*[np.array([])]*0) if False else (0,None)
 # render 500
 x0,y0,w,h=crop_box(bb[i],pad=55,min_size=400)
 mm,rr,_=render_batch((T0@H)[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);ren=cv2.cvtColor(ren,cv2.COLOR_RGB2BGR)
 cv2.rectangle(ren,(0,0),(70,28),(0,0,0),-1);cv2.putText(ren,''.join('xyz'[j] for j in range(3) if mask>>j&1) or 'id',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2);arr.append(ren)
orig=cv2.cvtColor(im[y0:y0+h,x0:x0+w],cv2.COLOR_RGB2BGR)
cv2.imwrite('/work/scratch/global_big_921a.jpg',np.vstack([np.hstack([orig,arr[0],arr[1],arr[2]]),np.hstack([arr[3],arr[4],arr[5],arr[6]])]))
cv2.imwrite('/work/scratch/global_big_921b.jpg',arr[7])
PY
python /work/scratch/global_big.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/color_scores.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
def metrics(im,r,m,o,p):
 vis=~cv2.dilate(p.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);g=m&o&vis
 a=im.astype(np.float32)/255;b=r
 out={}
 for c in range(3):out['ncc'+'rgb'[c]]=ncc(a[:,:,c],b[:,:,c],g)
 ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
 h1=cv2.cvtColor(a,cv2.COLOR_RGB2HSV);h2=cv2.cvtColor(b,cv2.COLOR_RGB2HSV)
 out['sat']=ncc(h1[:,:,1],h2[:,:,1],g)
 out['hue']=ncc(h1[:,:,0],h2[:,:,0],g) if g.sum() else -1
 # LAB chroma
 l1=cv2.cvtColor(a,cv2.COLOR_RGB2LAB);l2=cv2.cvtColor(b,cv2.COLOR_RGB2LAB)
 out['a']=ncc(l1[:,:,1],l2[:,:,1],g);out['bb']=ncc(l1[:,:,2],l2[:,:,2],g)
 # pixel rgb corr vectorized concatenated
 out['rgb']=ncc(a.reshape(-1,3),b.reshape(-1,3),[g.reshape(-1,1).repeat(3,1)]) if False else 0
 # saliency weighted edge alignment
 so1=cv2.Laplacian(ga,cv2.CV_32F);so2=cv2.Laplacian(gb,cv2.CV_32F);out['lap']=ncc(so1,so2,g)
 return out
for i in [800,850,921]:
 T0=np.load(f'/work/scratch/globalT_{i}.npy');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 print('\\nframe',i)
 for mask in range(8):
  H=np.eye(4,dtype=np.float32)
  for j in range(3):
   if mask …[+337 chars]
```
<details><summary>output</summary>

```

frame 800
id nccr:-0.015 nccg:-0.148 nccb:-0.266 sat:-0.299 hue:0.150 a:-0.119 bb:-0.057 rgb:0.000 lap:0.035
x nccr:-0.025 nccg:-0.156 nccb:-0.202 sat:-0.196 hue:0.229 a:-0.154 bb:0.027 rgb:0.000 lap:0.017
y nccr:-0.050 nccg:-0.108 nccb:-0.143 sat:-0.156 hue:0.202 a:-0.101 bb:-0.033 rgb:0.000 lap:0.023
xy nccr:0.059 nccg:0.043 nccb:0.041 sat:0.008 hue:0.330 a:-0.011 bb:-0.052 rgb:0.000 lap:0.015
z nccr:-0.027 nccg:-0.060 nccb:-0.076 sat:-0.056 hue:0.229 a:-0.012 bb:0.066 rgb:0.000 lap:0.042
xz nccr:0.047 nccg:-0.023 nccb:-0.074 sat:-0.058 hue:0.260 a:-0.060 bb:0.093 rgb:0.000 lap:0.048
yz nccr:0.037 nccg:-0.012 nccb:-0.039 sat:-0.049 hue:0.244 a:-0.042 bb:-0.053 rgb:0.000 lap:-0.010
xyz nccr:-0.047 nccg:-0.125 nccb:-0.165 sat:-0.167 hue:0.137 a:-0.126 bb:-0.047 rgb:0.000 lap:0.032

frame  …[+1577 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh'];D=np.load('/work/scratch/diag.npz')
for i in range(0,201,10):
 print(i,'t',np.round(T[i,:3,3],4),'mask',D['cent'][i].round(1),'area',D['areas'][i],'cp',D['cp'][i].round(2) if 'cp' in D else '')
PY"
```
<details><summary>output</summary>

```
0 t [0.0251 0.203  3.3231] mask [718.7 583.9] area 5956 cp 
10 t [0.0251 0.2031 3.325 ] mask [718.9 583.9] area 6069 cp 
20 t [0.0251 0.2034 3.3287] mask [718.4 583.8] area 5931 cp 
30 t [0.0251 0.2041 3.3398] mask [718.4 583.8] area 5924 cp 
40 t [0.0251 0.2043 3.3426] mask [718.7 583.8] area 6052 cp 
50 t [0.0251 0.2042 3.3408] mask [718.5 583.7] area 5941 cp 
60 t [0.0251 0.2042 3.3408] mask [718.7 584.1] area 6086 cp 
70 t [0.0214 0.2109 3.3447] mask [721.1 581.3] area 5431 cp 
80 t [0.0282 0.2096 3.3541] mask [725.7 578.1] area 4662 cp 
90 t [0.0324 0.2122 3.4393] mask [727.4 575.4] area 3755 cp 
100 t [0.0315 0.2132 3.4873] mask [725.7 580.7] area 3300 cp 
110 t [0.0315 0.2113 3.454 ] mask [724.1 584.3] area 3168 cp 
120 t [0.029  0.2258 3.4728] mask [724.1 586.1] area 3067 cp 
130 t …[+521 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fit_anchors.py <<'PY'
exec(open('/work/scratch/fit_mask_global.py').read().split(\"if __name__=='__main__'\")[0])
frames=[0,20,40,60,150,180,320,350,400,450,500,550,600,650,680,700,800,840,860,880,900,921]
for i in frames:
 T,res=fitglobal(i,0);np.save(f'/work/scratch/globalT_{i}.npy',T)
PY
python /work/scratch/fit_anchors.py 2>&1 | tee /work/scratch/fit_anchors.log"
```
<details><summary>output</summary>

```
global 0 0.9621299818091615 [ 3.96299864 -4.16190682  1.43452602  2.48476448 -0.38740866  2.19953416]
global 20 0.9588905398712234 [ 3.90417053 -5.04391615  1.07621331 -2.13645015  0.32391049 -1.98266618]
global 40 0.9619556418973612 [ 3.90306717 -4.27729967  1.17480838  2.20620663  0.33278803 -2.52572683]
global 60 0.9519872813990461 [ 3.72642223 -6.95733839  3.80182276 -0.26351318 -1.69051839  0.23624506]
global 150 0.9091450700029788 [ 3.21022659e+00 -6.68843859e+00  2.19738330e+00  2.99173459e+00
 -5.65708454e-03  2.50148251e-01]
global 180 0.8438755020080321 [ 4.12827574 -4.23977595  1.24780471 -2.41083547  0.60540791  2.21528848]
global 320 0.9304754481683554 [ 3.66697367 -6.4821945   4.03905948 -1.77738345 -1.37568604 -3.01202782]
global 350 0.9204647006255585 [ 3.70766712 -5.928960 …[+1521 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/eval_sym.py <<'PY'
import numpy as np,h5py,cv2,sys,itertools
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb']
# 24 proper signed axis permutations
SYMS=[];NAMES=[]
for p in itertools.permutations(range(3)):
 for signs in itertools.product([-1,1],repeat=3):
  H=np.zeros((4,4),np.float32);H[3,3]=1
  for j in range(3):H[p[j],j]=signs[j]
  if np.linalg.det(H[:3,:3])>0:
   SYMS.append(H);NAMES.append(str(tuple(p))+'/'+''.join('+' if s>0 else '-' for s in signs))
SYMS=np.array(SYMS);print('syms',len(SYMS))
def measure(im,r,m,o,p):
 vis=~cv2.dilate(p.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);g=m&o&vis
 a=im.astype(np.float32)/255
 h1=cv2.cvtColor(a,cv2.COLOR_RGB2HSV);h2=cv2.cvtColor(r,cv2.COLOR_RGB2HSV)
 l1=cv2.cvtColor(a,cv2.COLOR_RGB2LAB);l2=cv2.cvtColor(r,cv2.COLOR_RGB2LAB)
 ga=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY);gb=cv2.cvtColor(r,cv2.COLOR_RGB2GRAY)
 gx1=cv2.Sobel(ga,cv2.CV_32F,1,0,ksize=3);gy1=cv2.Sobel(ga,cv2.CV_32F,0,1,ksize=3);gx2=cv2.Sobel(gb,cv2.CV_32F,1,0,ksize=3);gy2=cv2.Sobel(gb,cv2.CV_32F,0,1,ksize=3)
 dot=(gx1*gx2+gy1*gy2)[g];mag=(np.sqrt(gx1*gx1+gy1*gy1)*np.sqrt(gx2*gx2+gy2*gy2))[g];grad=(dot/(mag+1e-6)).mean() if g.sum() else -1
 inter=(m&o&vis).sum();union=((m|o)&vis).sum();iou=inter/max(1,union)
 sat=ncc(h1[:,:,1],h2[:,:,1],g);ca=ncc(l1[:,:,1],l2[:,:,1],g);cb=ncc(l1[:,:,2],l2[:,:,2],g);gray=ncc(ga,gb,g)
 ch …[+1082 chars]
```
<details><summary>output</summary>

```
syms 24
Traceback (most recent call last):
  File "/work/scratch/eval_sym.py", line 39, in <module>
    vals,ms,rs,crop=evaluate(i,T0,im,oms[i],pms[i]);print('\n',i)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/eval_sym.py", line 29, in evaluate
    mm,rr,_=render_batch(Ts,x0,y0,w,h);ms=mm.cpu().numpy();rs=rr.cpu().numpy();a=im[y0:y0+h,x0:x0+w];o=om[y0:y0+h,x0:x0+w];p=pm[y0:y0+h,x0:x0+w]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/pose_utils.py", line 29, in render_batch
    uv,_=dr.interpolate(UV[None].expand(N,-1,-1),rast,F,rb)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
        …[+558 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/UV\\[None\\].expand(N,-1,-1)/UV[None].expand(N,-1,-1).contiguous()/' /work/scratch/pose_utils.py && python /work/scratch/eval_sym.py 921 850 500 0"
```
<details><summary>output</summary>

```
syms 24

 921
2.146 (2, 1, 0)/++- 23 iou:0.904 sat:0.160 ca:0.151 cb:0.222 gray:0.068 grad:0.012 chrom:0.318
2.028 (0, 1, 2)/-+- 1 iou:0.929 sat:0.108 ca:0.032 cb:0.130 gray:-0.021 grad:0.008 chrom:0.165
1.807 (0, 1, 2)/+++ 3 iou:0.923 sat:-0.039 ca:-0.072 cb:0.031 gray:-0.032 grad:0.022 chrom:-0.050
1.724 (2, 1, 0)/+-+ 22 iou:0.909 sat:-0.029 ca:-0.026 cb:-0.090 gray:0.013 grad:-0.023 chrom:-0.084
1.685 (1, 2, 0)/-+- 13 iou:0.603 sat:0.192 ca:0.358 cb:0.212 gray:0.079 grad:0.032 chrom:0.447
1.683 (0, 1, 2)/--+ 0 iou:0.920 sat:-0.036 ca:-0.071 cb:-0.206 gray:0.048 grad:0.023 chrom:-0.178
1.683 (0, 1, 2)/+-- 2 iou:0.928 sat:-0.059 ca:-0.147 cb:-0.134 gray:-0.020 grad:0.046 chrom:-0.196
1.645 (2, 1, 0)/-++ 21 iou:0.897 sat:-0.111 ca:-0.095 cb:-0.055 gray:0.054 grad:0.001 chrom:-0.160
1.619 ( …[+3119 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
D=np.load('/work/scratch/diag.npz');z=np.load('/work/inputs/reconstruction.npz');T=z['T_world_mesh']
for i in range(300,922,10):
 if D['cp'].shape[0] if 'cp' in D else False: pass
 print(f\"{i:3} cent {D['cent'][i,0]:.1f},{D['cent'][i,1]:.1f} bb x{D['bb'][i,0]}-{D['bb'][i,2]} y{D['bb'][i,1]}-{D['bb'][i,3]} A{D['areas'][i]:5d} t {T[i,0,3]: .3f},{T[i,1,3]: .3f},{T[i,2,3]: .3f} cp {z['contact_prob_right_left'][i,0]:.2f}/{z['contact_prob_right_left'][i,1]:.2f} dR {D['dR'][i]:.1f} dT {D['dT'][i]:.3f}\")
PY"
```
<details><summary>output</summary>

```

410 cent 711.1,802.4 bb x658-770 y756-848 A 7356 t -0.001, 1.112, 3.387 cp 0.00/0.00 dR 0.9 dT 0.015
420 cent 744.6,805.6 bb x694-791 y759-851 A 6556 t  0.116, 1.107, 3.352 cp 0.00/0.00 dR 1.7 dT 0.012
430 cent 761.8,805.1 bb x706-807 y759-851 A 6805 t  0.171, 1.093, 3.316 cp 0.00/0.00 dR 0.5 dT 0.005
440 cent 773.8,804.5 bb x715-818 y759-853 A 6943 t  0.215, 1.091, 3.282 cp 0.00/0.00 dR 0.7 dT 0.015
450 cent 782.9,804.2 bb x738-830 y758-854 A 7822 t  0.253, 1.075, 3.254 cp 0.00/0.00 dR 0.4 dT 0.015
460 cent 785.3,803.7 bb x738-836 y757-854 A 8143 t  0.261, 1.061, 3.214 cp 0.00/0.00 dR 1.4 dT 0.005
470 cent 795.6,805.6 bb x739-855 y757-855 A 9331 t  0.303, 1.058, 3.182 cp 0.00/0.00 dR 3.1 dT 0.007
480 cent 809.6,810.8 bb x742-878 y762-862 A10577 t  0.357, 1.047, 3.095 cp 0.00/0.00 dR 0.9  …[+4457 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python /work/scratch/eval_sym.py 550 650 680 20 40 180 320 350 400 450 600 700 800 840 860 880 900 > /work/scratch/eval_sym_all.txt && python - <<'PY'
import re
s=open('/work/scratch/eval_sym_all.txt').read().split('\\n\\n')
for chunk in s:
 lines=chunk.strip().splitlines()
 if lines: print(lines[0], '\\n  '+'\\n  '.join(lines[1:5]))
PY"
```
<details><summary>output</summary>

```
syms 24 
  
550 
  2.048 (0, 1, 2)/--+ 0 iou:0.944 sat:0.118 ca:0.131 cb:-0.028 gray:-0.085 grad:0.043 chrom:0.139
  2.022 (0, 1, 2)/+-- 2 iou:0.942 sat:0.052 ca:0.050 cb:0.112 gray:-0.016 grad:0.026 chrom:0.125
  1.823 (0, 1, 2)/-+- 1 iou:0.933 sat:0.021 ca:-0.032 cb:-0.108 gray:0.021 grad:0.029 chrom:-0.062
  1.803 (2, 1, 0)/+-+ 22 iou:0.890 sat:0.048 ca:0.065 cb:-0.054 gray:-0.026 grad:-0.032 chrom:0.040
650 
  2.045 (0, 1, 2)/+++ 3 iou:0.936 sat:0.229 ca:0.115 cb:-0.110 gray:0.043 grad:0.002 chrom:0.163
  2.032 (2, 1, 0)/-++ 21 iou:0.897 sat:0.263 ca:0.150 cb:-0.042 gray:-0.010 grad:-0.013 chrom:0.243
  1.893 (2, 1, 0)/++- 23 iou:0.894 sat:0.115 ca:0.040 cb:0.024 gray:-0.003 grad:-0.023 chrom:0.115
  1.823 (1, 2, 0)/-+- 13 iou:0.770 sat:0.073 ca:0.183 cb:0.268 gray:-0.223 grad:-0.034 c …[+5971 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/check_consistency.py <<'PY'
import numpy as np,sys,itertools
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS,NAMES
sel={0:0,20:0,40:0,60:0,550:0,650:3,680:2,700:0,840:23,850:23,860:0,880:23,900:1,921:23}
Ts={}
for i,k in sel.items():Ts[i]=np.load(f'/work/scratch/globalT_{i}.npy')@SYMS[k]
def rd(A,B):return np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude())
for grp in [[0,20,40,60],[550,650,680,700],[840,850,860,880,900,921]]:
 print('\\nGrp')
 for i in grp:
  print(i,'t',np.round(Ts[i][:3,3],3),'Rvec',np.round(Rot.from_matrix(Ts[i][:3,:3]).as_rotvec(),3))
 for i in grp:
  print(i,[(j,round(rd(Ts[i],Ts[j]),1),round(np.linalg.norm(Ts[i][:3,3]-Ts[j][:3,3]),3)) for j in grp if j"'!=i])
PY
python /work/scratch/check_consistency.py'
```
<details><summary>output</summary>

```
syms 24

Grp
0 t [0.006 0.256 4.001] Rvec [0.236 1.723 0.185]
20 t [2.000e-03 2.500e-01 3.943e+00] Rvec [0.222 1.677 0.196]
40 t [0.006 0.251 3.941] Rvec [ 0.174 -1.386 -0.223]
60 t [-0.006  0.25   3.765] Rvec [ 2.123 -0.372 -1.935]
0 [(20, 2.8, 0.059), (40, 179.3, 0.06), (60, 179.6, 0.237)]
20 [(0, 2.8, 0.059), (40, 176.8, 0.004), (60, 180.0, 0.178)]
40 [(0, 179.3, 0.06), (20, 176.8, 0.004), (60, 145.1, 0.177)]
60 [(0, 179.6, 0.237), (20, 180.0, 0.178), (40, 145.1, 0.177)]

Grp
550 t [0.365 1.078 3.171] Rvec [ 0.022  0.725 -0.   ]
650 t [0.359 1.075 3.159] Rvec [-0.063 -0.607 -0.16 ]
680 t [0.362 1.079 3.166] Rvec [0.004 0.721 0.015]
700 t [0.364 1.084 3.178] Rvec [ 0.011 -2.477  0.015]
550 [(650, 77.0, 0.014), (680, 1.4, 0.005), (700, 176.5, 0.009)]
650 [(550, 77.0, 0.014), (680, 76.8, 0 …[+1094 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/agg_static_eval.py <<'PY'
import numpy as np,h5py,cv2,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS,NAMES,measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
def avgT(Ts):
 Ts=np.array(Ts);R=Rot.from_matrix(Ts[:,:3,:3]);rv=R.as_rotvec();rv=rv.mean(0); T=np.eye(4);T[:3,:3]=Rot.from_rotvec(rv).as_matrix();T[:3,3]=Ts[:,:3,3].mean(0);return T
cands={}
# final clusters
cands['A']=avgT([np.load(f'/work/scratch/globalT_{i}.npy')@SYMS[k] for i,k in [(850,23),(880,23),(921,23),(840,23)]])
cands['B']=avgT([np.load(f'/work/scratch/globalT_{i}.npy')@SYMS[k] for i,k in [(860,0),(900,1)]])
# ground: consistency 550/680 same cluster, candidates
cands['G2']=avgT([np.load(f'/work/scratch/globalT_{i}.npy')@SYMS[k] for i,k in [(550,0),(680,2),(700,0)]])
cands['G1']=avgT([np.load(f'/work/scratch/globalT_{i}.npy')@SYMS[k] for i,k in [(320,3),(350,3),(400,0)]])
for name,T in cands.items():
 print('\\n',name,'R',Rot.from_matrix(T[:3,:3]).as_rotvec(),'t',T[:3,3])
 for phase,(a,b) in {'initial':(0,70),'ground1':(330,385),'ground2':(500,580),'ground2b':(660,685),'final':(840,922)}.items():
  iis=list(range(a,b,10)); vals=[]
  for i in iis:
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
   x0,y0 …[+399 chars]
```
<details><summary>output</summary>

```
syms 24

 A R [ 0.99215351 -0.2095253   2.92816872] t [0.10705204 0.22486342 3.50363858]
initial mean score/iou/chrom [1.235 0.613 0.017] success 0.0
ground1 mean score/iou/chrom [-2.3  0.  -1.8] success 0.0
ground2 mean score/iou/chrom [-2.3  0.  -1.8] success 0.0
ground2b mean score/iou/chrom [-2.3  0.  -1.8] success 0.0
final mean score/iou/chrom [2.136 0.905 0.314] success 1.0

 B R [-0.00169209  0.0040466  -0.00110646] t [0.10916893 0.20991014 3.31964824]
initial mean score/iou/chrom [1.143 0.498 0.113] success 0.0
ground1 mean score/iou/chrom [-2.3  0.  -1.8] success 0.0
ground2 mean score/iou/chrom [-2.3  0.  -1.8] success 0.0
ground2b mean score/iou/chrom [-2.3  0.  -1.8] success 0.0
final mean score/iou/chrom [ 1.086  0.671 -0.254] success 0.0

 G2 R [ 0.01256269 -0.34353837  0.00 …[+714 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/cluster_static.py <<'PY'
import numpy as np,h5py,cv2,sys,itertools
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS,NAMES,measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
def avgT(Ts):
 Ts=np.array(Ts);rv=Rot.from_matrix(Ts[:,:3,:3]).as_rotvec();rv=rv.mean(0);T=np.eye(4);T[:3,:3]=Rot.from_rotvec(rv).as_matrix();T[:3,3]=Ts[:,:3,3].mean(0);return T
def rd(A,B):return np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude())
def clusters(anchors,thrR=8,thrT=.12):
 Ts=[];meta=[]
 for i in anchors:
  G=np.load(f'/work/scratch/globalT_{i}.npy')
  for k,S in enumerate(SYMS):Ts.append(G@S);meta.append((i,k,NAMES[k]))
 n=len(Ts);par=list(range(n))
 def find(a):
  while par[a]"'!=a:par[a]=par[par[a]];a=par[a]
  return a
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:par[b]=a
 for a in range(n):
  for b in range(a):
   if rd(Ts[a],Ts[b])<thrR and np.linalg.norm(Ts[a][:3,3]-Ts[b][:3,3])<thrT:union(a,b)
 groups={}
 for a in range(n):groups.setdefault(find(a),[]).append(a)
 out=[]
 for g in groups.values():
  if len(g)>=max(2,len(anchors)): # candidate represented across at least 2 anchors
   out.append((avgT([Ts[a] for a in g]),[meta[a] for a in g]))
 return out
def evaluate(T,valf):
 vals=[]
 for i in valf:
  cap.set(cv2.CAP …[+1355 chars]
```
<details><summary>output</summary>

```
syms 24

PHASE initial clusters 0
Traceback (most recent call last):
  File "/work/scratch/cluster_static.py", line 46, in <module>
    best=max(ranked,key=lambda x:x[0]);np.save(f'/work/scratch/staticT_{name}.npy',best[4])
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: max() arg is an empty sequence

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys,itertools
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
def rd(A,B):return np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude())
Ts={i:np.load(f'/work/scratch/globalT_{i}.npy') for i in [0,20,40,60]}
for a,b in itertools.combinations(Ts,2):
 ds=[]
 for ka,sa in enumerate(SYMS):
  for kb,sb in enumerate(SYMS):
   Ta=Ts[a]@sa;Tb=Ts[b]@sb;ds.append((rd(Ta,Tb),np.linalg.norm(Ta[:3,3]-Tb[:3,3]),ka,kb,Ta[:3,3],Tb[:3,3]))
 print(a,b,min(ds,key=lambda x:x[0]))
PY"
```
<details><summary>output</summary>

```
syms 24
0 20 (2.790062533954644, 0.05894001008857458, 12, 12, array([0.00616895, 0.25597673, 4.00115995]), array([2.17276001e-03, 2.50181026e-01, 3.94264188e+00]))
0 40 (0.782931508246715, 0.060042180765075344, 0, 2, array([0.00616895, 0.25597673, 4.00115995]), array([0.0058229 , 0.25056999, 3.9413627 ]))
0 60 (38.292608251247536, 0.2366211551224125, 12, 15, array([0.00616895, 0.25597673, 4.00115995]), array([-0.00573459,  0.25026819,  3.76490735]))
20 40 (3.51548427244974, 0.0038872967556211635, 4, 7, array([2.17276001e-03, 2.50181026e-01, 3.94264188e+00]), array([0.0058229 , 0.25056999, 3.9413627 ]))
20 60 (37.29183773162396, 0.17791035357127083, 4, 5, array([2.17276001e-03, 2.50181026e-01, 3.94264188e+00]), array([-0.00573459,  0.25026819,  3.76490735]))
40 60 (38.37672428643079, 0.1768 …[+116 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/if len(g)>=max(2,len(anchors)):/if len(set(meta[a][0] for a in g))>=2:/' /work/scratch/cluster_static.py && python /work/scratch/cluster_static.py 2>&1 | tee /work/scratch/cluster_static.log"
```
<details><summary>output</summary>

```
syms 24

PHASE initial clusters 24
1.912 iou0.916 ch0.056 succ1.00 t[0.005 0.252 3.962] rv[0.23 1.71 0.19] n3 anchors[0, 20, 40]
1.906 iou0.914 ch0.070 succ1.00 t[0.005 0.252 3.962] rv[ 0.18 -1.41 -0.22] n3 anchors[0, 20, 40]
1.826 iou0.929 ch-0.032 succ1.00 t[0.005 0.252 3.962] rv[-2.25  0.3  -1.93] n3 anchors[0, 20, 40]
1.788 iou0.831 ch0.136 succ1.00 t[0.005 0.252 3.962] rv[ 0.26  0.15 -0.03] n3 anchors[0, 20, 40]
1.742 iou0.914 ch-0.093 succ1.00 t[0.005 0.252 3.962] rv[-1.91 -0.24  2.22] n3 anchors[0, 20, 40]
1.718 iou0.830 ch0.088 succ1.00 t[0.005 0.252 3.962] rv[-2.87  0.04  0.22] n3 anchors[0, 20, 40]
1.703 iou0.762 ch0.123 succ0.00 t[0.005 0.252 3.962] rv[ 1.27 -0.98 -1.49] n3 anchors[0, 20, 40]
1.673 iou0.827 ch0.016 succ1.00 t[0.005 0.252 3.962] rv[ 0.24 -0.41  3.08] n3 anchors[0 …[+2524 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS,NAMES
for a,b in [('initial','ground1'),('ground1','ground2'),('ground2','final'),('initial','final')]:
 A=np.load(f'/work/scratch/staticT_{a}.npy');B=np.load(f'/work/scratch/staticT_{b}.npy');ds=[]
 for k,S in enumerate(SYMS):
  d=np.degrees(Rot.from_matrix(A[:3,:3]@S[:3,:3]@B[:3,:3].T).magnitude());ds.append((d,k,NAMES[k]))
 print(a,b,sorted(ds)[:5])
 print('tdist',np.linalg.norm(A[:3,3]-B[:3,3]),'t',A[:3,3],B[:3,3])
PY"
```
<details><summary>output</summary>

```
syms 24
initial ground1 [(37.29541473432505, 10, '(1, 0, 2)/+-+'), (67.29851353991535, 16, '(2, 0, 1)/--+'), (74.70047197070028, 14, '(1, 2, 0)/+--'), (80.10707908013227, 0, '(0, 1, 2)/--+'), (83.77037489807144, 20, '(2, 1, 0)/---')]
tdist 1.0304512001219392 t [0.00472153 0.25224258 3.96172151] [-0.23753164  1.21760954  3.69487004]
ground1 ground2 [(49.82517054569453, 9, '(1, 0, 2)/-++'), (59.90427905291835, 0, '(0, 1, 2)/--+'), (67.16315857055821, 19, '(2, 0, 1)/+++'), (78.69562639102999, 5, '(0, 2, 1)/-++'), (87.3526971760809, 22, '(2, 1, 0)/+-+')]
tdist 0.8087479167228929 t [-0.23753164  1.21760954  3.69487004] [0.36345947 1.08044187 3.17135509]
ground2 final [(8.518450956308529, 0, '(0, 1, 2)/--+'), (82.03670239218101, 4, '(0, 2, 1)/---'), (87.37758194910467, 20, '(2, 1, 0)/---'), (90. …[+496 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/view_static_candidates.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from cluster_static import clusters,evaluate
from eval_sym import measure
from pose_utils import *
# Not rerun clustering; use stored top and generate second by rerunning cluster? Instead output all candidates to files from cluster static would require reload; recompute quick.
# import current static plus manually obtain ranked list using same functions
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
phases={'initial':([0,20,40],[0,20,40]),'ground1':([320,350],[340,380]),'ground2':([550,680,700],[510,555,680]),'final':([840,850,880,921],[850,921])}
for name,(anchors,valf) in phases.items():
 cs=clusters(anchors);rank=[]
 for T,meta in cs:
  vals=evaluate(T,valf);rank.append((np.mean([v[0] for v in vals]),T,meta))
 rank=sorted(rank,reverse=True,key=lambda x:x[0])
 arr=[]
 for rnk,(sc,T,meta) in enumerate(rank[:4]):
  i=valf[-1];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=35,min_size=320);mm,rr,_=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);ren=cv2.cvtColor(ren,cv2.COLOR_RGB2BGR);cv2.putText(ren,f'{rnk} {sc:.2f}',(3,20),cv2.FONT_HERSHEY_SIMPLEX,.5 …[+326 chars]
```
<details><summary>output</summary>

```
syms 24

PHASE initial clusters 24
1.912 iou0.916 ch0.056 succ1.00 t[0.005 0.252 3.962] rv[0.23 1.71 0.19] n3 anchors[0, 20, 40]
1.906 iou0.914 ch0.070 succ1.00 t[0.005 0.252 3.962] rv[ 0.18 -1.41 -0.22] n3 anchors[0, 20, 40]
1.826 iou0.929 ch-0.032 succ1.00 t[0.005 0.252 3.962] rv[-2.25  0.3  -1.93] n3 anchors[0, 20, 40]
1.788 iou0.831 ch0.136 succ1.00 t[0.005 0.252 3.962] rv[ 0.26  0.15 -0.03] n3 anchors[0, 20, 40]
1.742 iou0.914 ch-0.093 succ1.00 t[0.005 0.252 3.962] rv[-1.91 -0.24  2.22] n3 anchors[0, 20, 40]
1.718 iou0.830 ch0.088 succ1.00 t[0.005 0.252 3.962] rv[-2.87  0.04  0.22] n3 anchors[0, 20, 40]
1.703 iou0.762 ch0.123 succ0.00 t[0.005 0.252 3.962] rv[ 1.27 -0.98 -1.49] n3 anchors[0, 20, 40]
1.673 iou0.827 ch0.016 succ1.00 t[0.005 0.252 3.962] rv[ 0.24 -0.41  3.08] n3 anchors[0 …[+2524 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import cv2
for n in ['initial','ground1','ground2','final']:
 a=cv2.imread(f'/work/scratch/static_cand_{n}.jpg');h,w=a.shape[:2];cv2.imwrite(f'/work/scratch/static_cand_{n}_big.png',cv2.resize(a,(w*2,h*2),interpolation=cv2.INTER_NEAREST))
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
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,im=cap.read();x0,y0,x1,y1=bb[0];cv2.imwrite('/work/scratch/frame0_crop.jpg',im[y0-30:y1+30,x0-30:x1+30])
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import h5py,cv2,numpy as np,json
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f, h5py.File('/work/inputs/masks.h5','r') as m:
 for i in [0,20,40,330,350,380,510,550,680,700,840,860,900,921]:
  d=cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(float)
  o=m['object'][i]; v=o&np.isfinite(d)&(d>0)
  print(i,'n',v.sum(),'depth pixels percent',np.percentile(d[v],[5,25,50,75,95]),'mean',d[v].mean())
PY"
```
<details><summary>output</summary>

```
0 n 5956 depth pixels percent [3123. 3133. 3144. 3164. 3333.] mean 3194.6749496306247
20 n 5931 depth pixels percent [3157. 3168. 3179. 3198. 3387.] mean 3233.8760748609
40 n 6052 depth pixels percent [3023. 3033. 3045. 3063. 3278.] mean 3103.399372108394
330 n 8442 depth pixels percent [3207.   3250.   3301.   3399.   3606.95] mean 3343.525467898602
350 n 8639 depth pixels percent [3252.  3294.5 3342.  3461.  3679.1] mean 3397.0316008797313
380 n 8686 depth pixels percent [3152. 3193. 3240. 3353. 3548.] mean 3288.8583928160256
510 n 10335 depth pixels percent [3116. 3176. 3256. 3336. 3503.] mean 3274.7011127237542
550 n 10407 depth pixels percent [3009. 3069. 3144. 3220. 3385.] mean 3162.6100701450946
680 n 9870 depth pixels percent [3082.   3142.   3219.   3295.   3489.55] mean 3240.2919 …[+477 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/pose_utils.py';s=open(p).read()
s=s.replace(\"uv,_=dr.interpolate(UV[None].expand(N,-1,-1).contiguous(),rast,F,rb)\\n    mask=rast[:,:,:,3]>0\", \"uv,_=dr.interpolate(UV[None].expand(N,-1,-1).contiguous(),rast,F,rb)\\n    depth,_=dr.interpolate(Z[...,None],rast,F,rb)\\n    mask=rast[:,:,:,3]>0\")
s=s.replace(\"return mask, rgb.permute(0,2,3,1), rast[...,2]\", \"return mask, rgb.permute(0,2,3,1), depth[...,0]\")
open(p,'w').write(s)
PY
python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');from pose_utils import *
for i in [0,921]:
 T=np.load(f'/work/scratch/globalT_{i}.npy');m,r,d=render_batch(T[None],600,450,300,300);print(i,m.sum(),d[m].min().item(),d[m].median().item(),d[m].max().item())
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
  File "/work/scratch/pose_utils.py", line 30, in render_batch
    depth,_=dr.interpolate(Z[...,None],rast,F,rb)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 291, in interpolate
    return _interpolate_func.apply(attr, rast, tri)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/nvdiffrast/torch/ops.py", line 230, in forward
    out, out_da = _nvdiffrast_c.interpolate_fwd(attr, rast, tri) …[+152 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/dr.interpolate(Z\\[...,None\\],rast,F,rb)/dr.interpolate(Z[...,None].contiguous(),rast,F,rb)/' /work/scratch/pose_utils.py && python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');from pose_utils import *
for i in [0,921]:
 T=np.load(f'/work/scratch/globalT_{i}.npy');m,r,d=render_batch(T[None],600,450,300,300);print(i,m.sum(),d[m].min().item(),d[m].median().item(),d[m].max().item())
PY"
```
<details><summary>output</summary>

```
0 tensor(5581, device='cuda:0') 3.802354335784912 4.100579261779785 4.172307968139648
921 tensor(7100, device='cuda:0') 3.245129108428955 3.5617682933807373 3.71822190284729

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fit_anchors_z.py <<'PY'
exec(open('/work/scratch/fit_mask_global.py').read().split(\"if __name__=='__main__'\")[0])
for i in [0,20,40,330,350,380,510,550,680,700,840,850,860,880,900,921]:
 T,res=fitglobal(i,0);np.save(f'/work/scratch/globalTz_{i}.npy',T)
PY
# alter bounds locally in generated script by replacing z span
sed -i \"s/(max(1,z0-.8),z0+.9)/(max(1,z0-.16),z0+.16)/\" /work/scratch/fit_mask_global.py
python /work/scratch/fit_anchors_z.py 2>&1 | tee /work/scratch/fit_anchors_z.log"
```
<details><summary>output</summary>

```
global 0 0.951509312780989 [ 3.24471577e+00 -7.30580606e+00  4.31124192e+00 -1.50124987e-01
 -1.44007597e-01 -5.02354666e-03]
global 20 0.9646590343454455 [ 3.44133289 -4.03529036  0.79521253  0.12280612 -0.200799    3.13035367]
global 40 0.9514033680834002 [ 3.42594848 -7.73876663  3.93104829 -0.33685291  0.28817279  3.09841842]
global 330 0.9550641101046936 [  3.48941528 -11.5689049    3.54534572  -1.89452479  -1.17157923
  -2.894214  ]
global 350 0.9580619357787682 [ 3.38036905 -3.65815926  1.43254838  0.08321053  0.22258486  0.02836253]
global 380 0.9465905273765536 [ 3.63040502 -6.62558811  1.82560297  0.14813476 -1.04930275 -0.34887363]
global 510 0.9600943396226415 [ 3.19726602 -4.21347695  2.07132451  0.23206983  2.4548271  -0.00591125]
global 550 0.9775653161681181 [ 3.16673126 -2 …[+891 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/cluster_static_z.py <<'PY'
import numpy as np,h5py,cv2,sys,itertools
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS,NAMES,measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEP={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
def avgT(Ts):
 Ts=np.array(Ts);rv=Rot.from_matrix(Ts[:,:3,:3]).as_rotvec();rv=rv.mean(0);T=np.eye(4);T[:3,:3]=Rot.from_rotvec(rv).as_matrix();T[:3,3]=Ts[:,:3,3].mean(0);return T
def rd(A,B):return np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude())
def clusters(anchors,thrR=10,thrT=.18):
 Ts=[];meta=[]
 for i in anchors:
  G=np.load(f'/work/scratch/globalTz_{i}.npy')
  for k,S in enumerate(SYMS):Ts.append(G@S);meta.append((i,k,NAMES[k]))
 n=len(Ts);par=list(range(n))
 def find(a):
  while par[a]"'!=a:par[a]=par[par[a]];a=par[a]
  return a
 def union(a,b):
  a=find(a);b=find(b)
  if a!=b:par[b]=a
 for a in range(n):
  for b in range(a):
   if rd(Ts[a],Ts[b])<thrR and np.linalg.norm(Ts[a][:3,3]-Ts[b][:3,3])<thrT:union(a,b)
 groups={}
 for a in range(n):groups.setdefault(find(a),[]).append(a)
 return [(avgT([Ts[a] for a in g]),[meta[a] for a in g]) for g in …[+2099 chars]
```
<details><summary>output</summary>

```
syms 24

PHASE initial clusters 0
Traceback (most recent call last):
  File "/work/scratch/cluster_static_z.py", line 45, in <module>
    best=sorted(rank,reverse=True,key=lambda x:x[0])[0];np.save(f'/work/scratch/staticTz_{name}.npy',best[1]);np.savez(f'/work/scratch/staticTz_{name}_all.npz',T=np.array([r[1] for r in sorted(rank,reverse=True,key=lambda x:x[0])[:10]]),score=np.array([r[0] for r in sorted(rank,reverse=True,key=lambda x:x[0])[:10]]))
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
IndexError: list index out of range

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
T=np.load('/work/scratch/staticTz_initial.npy');print(T)
m,r,d=render_batch(T[None],600,450,300,300);print(d[m].min().item(),np.percentile(d[m].cpu().numpy(),[5,25,50,75,95]),d[m].max().item())
# corners depths
import trimesh
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();lo,hi=geo.bounds;C=np.array([[x,y,z,1] for x in[lo[0],hi[0]] for y in[lo[1],hi[1]] for z in[lo[2],hi[2]]]);pc=C@Tcw.T@T.T
print('center cam',(Tcw@T)[:3,3], 'corner z sorted',np.sort(pc[:,2]))
PY"
```
<details><summary>output</summary>

```
[[ 0.99490336 -0.00859704  0.10046559 -0.00385035]
 [ 0.01077236 -0.98159417 -0.19067484  0.21735523]
 [ 0.10025568  0.19078529 -0.97649872  3.28363318]
 [ 0.          0.          0.          1.        ]]
3.0508639812469482 [3.11361694 3.33192992 3.42821169 3.44576311 3.46396179] 3.4775540828704834
center cam [-0.16308182  0.10040204  3.24471573] corner z sorted [3.1018309  3.12155383 3.13251855 3.15224149 3.50150618 3.52122911
 3.53219383 3.55191676]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/pose_utils.py';s=open(p).read()
s=s.replace(\"        0.5*Z+0.1,\\n        Z],-1)\", \"        (100.0+0.1)*Z-20.0,\\n        Z],-1)\")
open(p,'w').write(s)
PY
python - <<'PY'
import numpy as np,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
T=np.load('/work/scratch/staticTz_initial.npy');m,r,d=render_batch(T[None],600,450,300,300);print('depthz',d[m].min().item(),np.percentile(d[m].cpu().numpy(),[5,25,50,75,95]),d[m].max().item())
PY
python /work/scratch/eval_sym.py 921 850 500 0"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 3, in <module>
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
syms 24

 921
-2.3 (2, 1, 0)/--- 20 iou:0.000 sat:-1.000 ca:-1.000 cb:-1.000 gray:-1.000 grad:-1.000 chrom:-1.800
-2.3 (2, 1, 0)/-++ 21 iou:0.000 sat:-1.000 ca:-1.000 cb:-1.000 gray:-1.000 grad:-1.000 chrom:-1.800
-2.3 (2, 1, 0)/+-+ 22 iou:0.000 sat:-1.000 ca:-1.000 cb:-1.000 gray:-1.000 grad:-1.000 chrom:-1.800
-2.3 (2, 1, 0)/++- 23 iou:0.000 sat:-1.000 ca:-1.000 cb:-1.000 gray:-1.000 grad:-1.000 chrom:-1.800
-2.3 (2, 0, 1)/--+ 16 iou:0.000 sat:-1.000 ca:-1.000 cb:-1.000 gray:-1.000 grad:-1.000 chrom:-1.800
-2.3 (2, 0, 1)/-+- 17 iou:0.000 sat:-1.000 ca:-1.000 cb:-1.000 gray:-1.000 gra …[+3438 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/pose_utils.py';s=open(p).read();s=s.replace(\"(100.0+0.1)*Z-20.0\", \"(100.0+0.01)/99.99*Z-2*100.0*0.01/99.99\")
open(p,'w').write(s)
PY
python - <<'PY'
import numpy as np,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
T=np.load('/work/scratch/staticTz_initial.npy');m,r,d=render_batch(T[None],600,450,300,300);print(m.sum(),d[m].min().item(),np.percentile(d[m].cpu().numpy(),[5,25,50,75,95]),d[m].max().item())
PY
python /work/scratch/eval_sym.py 921 850 500 0"
```
<details><summary>output</summary>

```
tensor(5805, device='cuda:0') 3.0108466148376465 [3.02232952 3.03878593 3.05361533 3.07032228 3.28029981] 3.4380452632904053
syms 24

 921
2.597 (2, 1, 0)/++- 23 iou:0.904 sat:0.472 ca:0.330 cb:0.484 gray:0.092 grad:-0.017 chrom:0.778
2.499 (0, 1, 2)/-+- 1 iou:0.929 sat:0.368 ca:0.301 cb:0.330 gray:0.121 grad:0.021 chrom:0.605
2.04 (2, 1, 0)/-++ 21 iou:0.897 sat:0.117 ca:-0.021 cb:0.327 gray:-0.060 grad:-0.009 chrom:0.250
1.853 (2, 1, 0)/--- 20 iou:0.915 sat:0.010 ca:-0.037 cb:0.083 gray:-0.102 grad:-0.017 chrom:0.033
1.779 (0, 1, 2)/+-- 2 iou:0.928 sat:-0.052 ca:-0.020 cb:-0.076 gray:0.073 grad:-0.007 chrom:-0.089
1.758 (1, 0, 2)/--- 8 iou:0.652 sat:0.334 ca:0.148 cb:0.234 gray:0.091 grad:-0.015 chrom:0.444
1.623 (2, 0, 1)/--+ 16 iou:0.536 sat:0.323 ca:0.355 cb:0.232 gray:-0.006 grad:0.00 …[+3218 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/cluster_static_z.py';s=open(p).read();s=s.replace(\"return [(avgT([Ts[a] for a in g]),[meta[a] for a in g]) for g in groups.values() if len(set(meta[a][0] for a in g))>=2]\", \"out=[(avgT([Ts[a] for a in g]),[meta[a] for a in g]) for g in groups.values() if len(set(meta[a][0] for a in g))>=2]\\n if not out: out=[(Ts[a],[meta[a]]) for a in range(n)]\\n return out\")
open(p,'w').write(s)
PY
python /work/scratch/cluster_static_z.py 2>&1 | tee /work/scratch/cluster_static_z.log"
```
<details><summary>output</summary>

```
syms 24

PHASE initial clusters 72
0 1.233 t[-0.004  0.217  3.284] rv[2.95 0.   0.15] iou0.917 ch0.564 grad0.069 dz0.314 n1 anchors[0]
1 0.941 t[-0.004  0.217  3.284] rv[ 0.16 -0.3  -3.12] iou0.913 ch0.230 grad0.032 dz0.315 n1 anchors[0]
2 0.820 t[-0.007  0.229  3.465] rv[ 2.9  -0.    0.25] iou0.914 ch0.491 grad0.044 dz0.483 n1 anchors[40]
3 0.737 t[-0.004  0.217  3.284] rv[-0.    3.03 -0.29] iou0.918 ch-0.017 grad0.006 dz0.315 n1 anchors[0]
4 0.682 t[-0.004  0.217  3.284] rv[ 2.22 -0.2  -2.01] iou0.828 ch0.019 grad0.000 dz0.288 n1 anchors[0]
5 0.650 t[-0.004  0.217  3.284] rv[2.01 0.21 2.23] iou0.812 ch0.027 grad-0.022 dz0.289 n1 anchors[0]
6 0.640 t[-0.004  0.217  3.284] rv[ 1.14 -1.37 -1.02] iou0.635 ch0.177 grad0.012 dz0.229 n1 anchors[0]
7 0.629 t[-0.004  0.217  3.284] rv[-0.19 -0.1   …[+3685 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/cluster_static_z.py 2>&1 | tee /work/scratch/cluster_static_z_fixed.log'
```
<details><summary>output</summary>

```
syms 24

PHASE initial clusters 72
0 1.905 t[0.008 0.217 3.48 ] rv[ 0.19 -0.13  3.13] iou0.915 ch0.881 grad0.187 dz0.133 n1 anchors[20]
1 1.460 t[-0.004  0.217  3.284] rv[ 0.16 -0.3  -3.12] iou0.913 ch0.267 grad0.034 dz0.070 n1 anchors[0]
2 1.341 t[-0.004  0.217  3.284] rv[-0.    3.03 -0.29] iou0.918 ch0.107 grad0.040 dz0.070 n1 anchors[0]
3 1.322 t[-0.007  0.229  3.465] rv[-0.27  0.35  3.11] iou0.915 ch0.210 grad0.077 dz0.129 n1 anchors[40]
4 1.280 t[0.008 0.217 3.48 ] rv[-3.05  0.01  0.19] iou0.914 ch0.162 grad0.090 dz0.132 n1 anchors[20]
5 1.244 t[-0.007  0.229  3.465] rv[ 0.    2.95 -0.34] iou0.915 ch0.136 grad0.039 dz0.129 n1 anchors[40]
6 1.234 t[-0.004  0.217  3.284] rv[2.95 0.   0.15] iou0.917 ch-0.007 grad0.009 dz0.070 n1 anchors[0]
7 1.194 t[-0.004  0.217  3.284] rv[-0.15 -1.67   …[+3679 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/view_staticz.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:]
arr=[]
for name,i in [('initial',0),('ground1',350),('ground2',550),('final',921)]:
 T=np.load(f'/work/scratch/staticTz_{name}.npy');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=45,min_size=360);mm,rr,dd=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);ren=cv2.cvtColor(ren,cv2.COLOR_RGB2BGR);orig=cv2.cvtColor(im[y0:y0+h,x0:x0+w],cv2.COLOR_RGB2BGR);cv2.putText(orig,name,(3,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1);arr.append(np.hstack([orig,ren]))
cv2.imwrite('/work/scratch/staticz_compare.jpg',np.vstack(arr))
PY
python /work/scratch/view_staticz.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/fit_dynamic.py <<'PY'
import numpy as np,h5py,cv2,sys,trimesh,time
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from pose_utils import *
from eval_sym import SYMS,NAMES,measure
D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();lo,hi=geo.bounds
corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
def raster(poly,x0,y0,w,h):
 m=np.zeros((h,w),np.uint8);p=np.round(poly-[x0,y0]).astype(np.int32);cv2.fillConvexPoly(m,cv2.convexHull(p),1);return m.astype(bool)
def fit_single(i,debug=False):
 cap=cv2.VideoCapture('/work/inputs/video.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);cap.release()
 Tbase=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'][i].astype(np.float32);cam=Tcw@Tbase;c0=cam[:3,3];z0=c0[2];uv0=np.array([K[0,0]*c0[0]/c0[2]+K[0,2],K[1,1]*c0[1]/c0[2]+K[1,2]])
 x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];vis=~cv2.dilate(p.astype(np.uint …[+1839 chars]
```
<details><summary>output</summary>

```
syms 24
frame 240 global mask 0.9696106362773029 best 0.6432099822552391 (1, 0, 2)/++- iou 0.8367003367003367 ch 0.44659354447665933 grad 0.036968112 dz -0.4779644 t [-0.60162353  0.607416    3.47743385]
  0.643 (1, 0, 2)/++- 0.837 0.447 0.037 -0.478
  0.364 (1, 0, 2)/+-+ 0.842 0.08 0.026 -0.454
  0.335 (1, 0, 2)/-++ 0.832 0.168 -0.028 -0.486
  0.287 (1, 0, 2)/--- 0.844 0.069 -0.0 -0.483
  0.269 (0, 1, 2)/--+ 0.94 0.024 0.065 -0.564
frame 300 global mask 0.9706701317987748 best 1.3373010941315226 (0, 1, 2)/-+- iou 0.9152510733619563 ch 0.34641209244994764 grad 0.046381056 dz -0.15103102 t [-0.28292705  1.03994427  3.25921818]
  1.337 (0, 1, 2)/-+- 0.915 0.346 0.046 -0.151
  1.189 (2, 1, 0)/++- 0.813 0.412 -0.065 -0.145
  0.956 (1, 0, 2)/++- 0.733 0.166 -0.008 -0.101
  0.9 (0, 1, 2)/--+ 0.9 …[+3872 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/view_dyn.py <<'PY'
import numpy as np,h5py,cv2,sys,os
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:]
arr=[]
for i in [240,300,330,400,450,600,740,760,800,820]:
 T=np.load(f'/work/scratch/dynT_{i}.npy');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=35,min_size=320);mm,rr,dd=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);ren=cv2.cvtColor(ren,cv2.COLOR_RGB2BGR);orig=cv2.cvtColor(im[y0:y0+h,x0:x0+w],cv2.COLOR_RGB2BGR);cv2.putText(orig,str(i),(3,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1);arr.append(np.hstack([orig,ren]))
# grid 2 cols
rows=[]
for j in range(0,len(arr),2):rows.append(np.hstack(arr[j:j+2]))
cv2.imwrite('/work/scratch/dyn_compare.jpg',np.vstack(rows))
PY
python /work/scratch/view_dyn.py"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/view_dyn.py", line 10, in <module>
    for j in range(0,len(arr),2):rows.append(np.hstack(arr[j:j+2]))
                                             ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/core/shape_base.py", line 359, in hstack
    return _nx.concatenate(arrs, 1, dtype=dtype, casting=casting)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 320 and the array at index 1 has size 336

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/view_dyn.py';s=open(p).read();s=s.replace(\"arr.append(np.hstack([orig,ren]))\", \"arr.append(cv2.resize(np.hstack([orig,ren]),(800,400)))\")
s=s.replace(\"rows.append(np.hstack(arr[j:j+2]))\", \"rows.append(np.vstack(arr[j:j+2]) if j+1>=len(arr) else np.vstack(arr[j:j+2]))\")
# simpler vertical stack all
s=s.replace(\"rows=[]\\nfor j in range(0,len(arr),2):rows.append(np.vstack(arr[j:j+2]) if j+1>=len(arr) else np.vstack(arr[j:j+2]))\\ncv2.imwrite('/work/scratch/dyn_compare.jpg',np.vstack(rows))\", \"cv2.imwrite('/work/scratch/dyn_compare.jpg',np.vstack(arr))\")
open(p,'w').write(s)
PY
python /work/scratch/view_dyn.py"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,glob,os
from scipy.spatial.transform import Rotation as Rot
for i in [240,300,330,400,450,600,740,760,780,800,820]:
 p=f'/work/scratch/dynT_{i}.npy'
 if os.path.exists(p):
  T=np.load(p);print(i,'t',np.round(T[:3,3],3),'rv',np.round(Rot.from_matrix(T[:3,:3]).as_rotvec(),3))
for n in ['initial','ground1','ground2','final']:
 T=np.load(f'/work/scratch/staticTz_{n}.npy');print(n,'t',np.round(T[:3,3],3),'rv',np.round(Rot.from_matrix(T[:3,:3]).as_rotvec(),3))
PY"
```
<details><summary>output</summary>

```
240 t [-0.602  0.607  3.477] rv [-1.125 -1.714  1.35 ]
300 t [-0.283  1.04   3.259] rv [-2.577  0.081 -1.677]
330 t [-0.239  1.192  3.643] rv [-2.454  0.393 -1.604]
400 t [-0.219  1.182  3.623] rv [-1.692  1.522 -1.1  ]
450 t [0.275 1.114 3.346] rv [-0.213  0.165 -3.056]
600 t [0.397 1.141 3.395] rv [-2.836  0.072 -1.21 ]
740 t [0.604 0.884 3.375] rv [1.15  0.464 1.98 ]
760 t [0.676 0.509 3.181] rv [ 2.697 -0.024  0.236]
780 t [0.65  0.149 3.517] rv [2.264 2.023 0.215]
800 t [0.214 0.171 3.284] rv [ 1.32  -0.122  2.767]
820 t [0.116 0.218 3.42 ] rv [ 1.013 -0.184  2.905]
initial t [0.008 0.217 3.48 ] rv [ 0.19  -0.134  3.126]
ground1 t [-0.205  1.108  3.391] rv [-0.414  0.065 -3.092]
ground2 t [0.357 1.085 3.184] rv [ 1.333 -0.113  2.675]
final t [0.108 0.219 3.435] rv [ 0.923 -0.164  2.97 …[+3 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
for i,n in [(330,'ground1'),(820,'final')]:
 A=np.load(f'/work/scratch/dynT_{i}.npy');B=np.load(f'/work/scratch/staticTz_{n}.npy');ds=[]
 for k,S in enumerate(SYMS):ds.append((np.degrees(Rot.from_matrix(A[:3,:3]@S[:3,:3]@B[:3,:3].T).magnitude()),k))
 print(i,n,sorted(ds)[:5], 'td',np.linalg.norm(A[:3,3]-B[:3,3]))
PY"
```
<details><summary>output</summary>

```
syms 24
330 ground1 [(18.463374873009176, 21), (73.96319856704966, 17), (83.26798311914581, 1), (88.93942885532742, 15), (93.95658544971933, 12)] td 0.2676670110937333
820 final [(4.421176221039196, 3), (86.19733867714096, 21), (88.30598329078833, 10), (88.7397855451405, 7), (91.41511470328591, 6)] td 0.016583369689713287

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/fit_dynamic.py 80 100 120 140 150 160 170 180 190 200 210 220 230 250 260 270 280 290 310 320 390 410 420 430 440 460 470 480 490 500 690 700 710 720 730 750 770 790 810 830 2>&1 | tee /work/scratch/fit_dynamic_more.log'
```
<details><summary>output</summary>

```
syms 24
frame 80 global mask 0.938585345192715 best 1.7330424970342801 (2, 1, 0)/-++ iou 0.8448618784530387 ch 0.8996161621456458 grad 0.0958524 dz 0.31815863 t [0.00998426 0.2210181  3.57474698]
  1.733 (2, 1, 0)/-++ 0.845 0.9 0.096 0.318
  1.035 (0, 1, 2)/+++ 0.912 0.111 0.005 0.343
  1.029 (2, 1, 0)/++- 0.84 0.179 -0.002 0.318
  1.028 (0, 1, 2)/-+- 0.908 0.103 0.014 0.342
  0.85 (2, 1, 0)/+-+ 0.852 -0.05 0.018 0.319
frame 100 global mask 0.9667989895344641 best 2.273009805082852 (0, 1, 2)/+++ iou 0.8786106032906764 ch 0.9900627938188529 grad 0.14450116 dz 0.03196287 t [0.01173048 0.21443549 3.42145921]
  2.273 (0, 1, 2)/+++ 0.879 0.99 0.145 0.032
  1.627 (0, 1, 2)/-+- 0.884 0.293 0.099 0.03
  1.276 (2, 1, 0)/++- 0.752 0.085 0.114 0.05
  1.272 (0, 1, 2)/--+ 0.88 -0.076 0.056 0.032
  1.25 …[+16252 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,glob,os,re
from scipy.spatial.transform import Rotation as Rot
def rd(A,B):return np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude())
for seg in [range(80,330,10),range(390,501,10),range(690,831,10)]:
 arr=[]
 for i in seg:
  p=f'/work/scratch/dynT_{i}.npy'
  if os.path.exists(p):arr.append((i,np.load(p)))
 print('\\nSEG')
 for (i,A),(j,B) in zip(arr,arr[1:]):
  print(f'{i:3}->{j:3} dR {rd(A,B):6.1f} dT {np.linalg.norm(A[:3,3]-B[:3,3]):.3f} t {np.round(B[:3,3],2)}')
PY"
```
<details><summary>output</summary>

```

SEG
 80->100 dR    6.5 dT 0.153 t [0.01 0.21 3.42]
100->120 dR  166.2 dT 0.274 t [-0.01  0.25  3.69]
120->140 dR  162.4 dT 0.111 t [-0.    0.22  3.58]
140->150 dR  159.8 dT 0.116 t [-0.04  0.23  3.69]
150->160 dR    5.7 dT 0.081 t [-0.08  0.19  3.64]
160->170 dR    4.8 dT 0.095 t [-0.18  0.18  3.66]
170->180 dR   88.1 dT 0.339 t [-0.32  0.17  3.35]
180->190 dR    3.3 dT 0.189 t [-0.42  0.18  3.52]
190->200 dR  158.5 dT 0.175 t [-0.55  0.16  3.63]
200->210 dR  109.7 dT 0.116 t [-0.61  0.26  3.66]
210->220 dR  162.6 dT 0.183 t [-0.63  0.43  3.71]
220->230 dR  159.2 dT 0.171 t [-0.65  0.52  3.57]
230->240 dR   65.1 dT 0.134 t [-0.6   0.61  3.48]
240->250 dR  126.4 dT 0.545 t [-0.53  1.07  3.76]
250->260 dR  160.8 dT 0.117 t [-0.44  1.13  3.81]
260->270 dR   96.8 dT 0.097 t [-0.38  1.14  3.74 …[+1443 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/candidate_metrics.py <<'PY'
import numpy as np,h5py,cv2,sys,glob,re,os
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS,NAMES,measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
frames=sorted(int(re.search(r'dynT_(\\d+)\\.npy',p).group(1)) for p in glob.glob('/work/scratch/dynT_*.npy'))
out={}
for i in frames:
 T0=np.load(f'/work/scratch/dynT_{i}.npy');cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);Ts=T0[None]@SYMS;mm,rr,dd=render_batch(Ts,x0,y0,w,h);ms=mm.cpu().numpy();rs=rr.cpu().numpy();ds=dd.cpu().numpy();aa=im[y0:y0+h,x0:x0+w];o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];dep=DEPS[i][y0:y0+h,x0:x0+w]
 ious=[];chroms=[];grads=[];grays=[];sat=[];cas=[];cbs=[];dzs=[];scores=[]
 for k in range(24):
  sc,d=measure(aa,rs[k],ms[k],o,p);mask=ms[k]&o;dz=np.median(ds[k][mask])-np.median(dep[mask]) if mask.sum()>20 else 1
  ious.append(d['iou']);chroms.append(d['chrom']);grads.append(d['grad']);grays.append(d['gray']);sat.append(d['sat']);cas.append(d['ca']);cbs.append(d['cb']);dzs.append(dz)
  scor …[+573 chars]
```
<details><summary>output</summary>

```
syms 24
80 best 3 1.733 next 1.035
100 best 3 2.273 next 1.627
120 best 3 0.885 next 0.779
140 best 3 1.841 next 1.105
150 best 3 0.981 next 0.6
160 best 3 1.12 next 1.036
170 best 3 0.987 next 0.928
180 best 3 1.197 next 1.164
190 best 3 -4.556 next -4.779
200 best 3 -0.836 next -0.861
210 best 3 -4.522 next -4.568
220 best 3 -1.113 next -1.366
230 best 3 1.226 next 1.177
240 best 3 0.643 next 0.364
250 best 3 0.712 next 0.556
260 best 3 1.648 next 1.314
270 best 3 1.35 next 1.21
280 best 3 1.443 next 1.339
290 best 3 1.517 next 1.464
300 best 3 1.337 next 1.189
310 best 3 1.473 next 1.343
320 best 3 1.374 next 1.141
330 best 3 1.612 next 1.534
390 best 3 1.424 next 1.323
400 best 3 1.218 next 1.066
410 best 3 1.145 next 1.014
420 best 3 1.994 next 1.286
430 best 3 1.436 next 1.356
440 be …[+635 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/viterbi.py <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
Z=np.load('/work/scratch/candidate_metrics.npz')
frames=sorted(set(int(k.split('_')[0]) for k in Z.files));print('frames',frames)
def get(i,k):return Z[f'{i}_{k}']
def rd(A,B):return Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude()
def seg_vit(frames,startT,endT,lam=0.8,hi=2.0,maxstep=.45):
 # nodes first and last virtual states
 states=[];scores=[]
 states.append([startT]);scores.append(np.array([0.]))
 for i in frames:
  states.append([Z[f'{i}_T'][k] for k in range(24)]);scores.append(1.6*get(i,'iou')+0.9*get(i,'chrom')+0.5*get(i,'grad')-2*np.abs(get(i,'dz')-.08))
 states.append([endT]);scores.append(np.array([0.]))
 # Viterbi
 dp=scores[0].copy();backs=[]
 for j in range(1,len(states)):
  ndp=np.full(len(states[j]),1e9);bk=np.zeros(len(states[j]),int)
  for b,Tb in enumerate(states[j]):
   costs=[]
   for a,Ta in enumerate(states[j-1]):
    ang=rd(Ta,Tb);tr=lam*ang+hi*max(0,ang-maxstep)**2
    costs.append(dp[a]-tr)
   bk[b]=int(np.argmax(costs));ndp[b]=scores[j][b]+max(costs)
  dp=ndp;backs.append(bk)
 path=[int(np.argmax(dp))]
 for bk in backs[::-1]:path.append(int(bk[bk[path[-1]]]))
 path=path[::-1]
 return path
Tinit=np.load('/work/scratch/staticTz_initial.npy');Tg1=np.load('/work/scratch/staticTz_ground1.npy');Tg2=np.load('/work/scratch/staticTz_ground2.npy');Tfin=np.load('/work/ …[+654 chars]
```
<details><summary>output</summary>

```
syms 24
frames [80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830]

LAMBDA 0.2
Traceback (most recent call last):
  File "/work/scratch/viterbi.py", line 35, in <module>
    p=seg_vit(fs,s,e,lam=lam,hi=2.0,maxstep=.5)
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/viterbi.py", line 27, in seg_vit
    for bk in backs[::-1]:path.append(int(bk[bk[path[-1]]]))
                                          ~~^^^^^^^^^^^^^^
IndexError: index 4 is out of bounds for axis 0 with size 1

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/viterbi.py';s=open(p).read();s=s.replace(\"path=[int(np.argmax(dp))]\\n for bk in backs[::-1]:path.append(int(bk[bk[path[-1]]]))\\n path=path[::-1]\", \"path=[int(np.argmax(dp))]\\n idx=path[0]\\n for bk in backs[::-1]:\\n  idx=int(bk[idx]);path.append(idx)\\n path=path[::-1]\")
open(p,'w').write(s)
PY
python /work/scratch/viterbi.py"
```
<details><summary>output</summary>

```
syms 24
frames [80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830]

LAMBDA 0.2
A
 150: state 1 dR  0.1 dT0.081
 160: state 1 dR  0.1 dT0.095
 170: state 1 dR  0.2 dT0.339
 180: state21 dR  0.1 dT0.189
 190: state21 dR  0.7 dT0.175
 200: state23 dR  0.7 dT0.116
 210: state 3 dR  0.6 dT0.183
 220: state 1 dR  0.7 dT0.171
 230: state10 dR  0.7 dT0.134
 240: state18 dR  0.3 dT0.545
 250: state 1 dR  0.7 dT0.117
 260: state 3 dR  0.5 dT0.097
 270: state21 dR  0.5 dT0.084
 280: state 8 dR  1.2 dT0.098
 290: state 4 dR  0.9 dT0.329
 300: state21 dR  0.3 dT0.470
 310: state21 dR  0.8 dT0.062
 320:  …[+6040 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for a,b in [('initial','ground1'),('ground1','ground2'),('ground2','final')]:
 A=np.load(f'/work/scratch/staticTz_{a}.npy');B=np.load(f'/work/scratch/staticTz_{b}.npy');print(a,b,np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude()),np.linalg.norm(A[:3,3]-B[:3,3]))
PY"
```
<details><summary>output</summary>

```
initial ground1 8.826003159385897 0.9205823767235872
ground1 ground2 39.01631996659701 0.5990631396112743
ground2 final 19.823321931791163 0.9357136529431748

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/viterbi.py';s=open(p).read();s=s.replace(\"for j,i in enumerate(fs):\\n   a=rd(seq[j+1],seq[j+2]); t=np.linalg.norm(seq[j+1][:3,3]-seq[j+2][:3,3]);print(f' {i}: state{p[j+1]:2} dR{a:5.1f} dT{t:.3f}')\", \"for j,i in enumerate(fs):\\n   a=rd(seq[j+1],seq[j]); t=np.linalg.norm(seq[j+1][:3,3]-seq[j][:3,3]);print(f' {i}: state{p[j+1]:2} dR{a:5.1f} dT{t:.3f}')\")
s=s.replace(\"print(' end dR',round(rd(seq[-2],seq[-1]),1))\", \"print(' end dR',round(rd(seq[-1],seq[-2]),1))\")
open(p,'w').write(s)
PY
python /work/scratch/viterbi.py | sed -n '1,120p'"
```
<details><summary>output</summary>

```
syms 24
frames [80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830]

LAMBDA 0.2
A
 150: state 1 dR  0.5 dT0.219
 160: state 1 dR  0.1 dT0.081
 170: state 1 dR  0.1 dT0.095
 180: state21 dR  0.2 dT0.339
 190: state21 dR  0.1 dT0.189
 200: state23 dR  0.7 dT0.175
 210: state 3 dR  0.7 dT0.116
 220: state 1 dR  0.6 dT0.183
 230: state10 dR  0.7 dT0.171
 240: state18 dR  0.7 dT0.134
 250: state 1 dR  0.3 dT0.545
 260: state 3 dR  0.7 dT0.117
 270: state21 dR  0.5 dT0.097
 280: state 8 dR  0.5 dT0.084
 290: state 4 dR  1.2 dT0.098
 300: state21 dR  0.9 dT0.329
 310: state21 dR  0.3 dT0.470
 320:  …[+2562 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
# import funcs without runs
s=open('/work/scratch/viterbi.py').read().split(\"Tinit=\")[0]
exec(s)
from scipy.spatial.transform import Rotation as Rot
for lam in [.2,.5,2.0]:
 fs=[i for i in range(400,500,10)];Tg1=np.load('/work/scratch/staticTz_ground1.npy');Tg2=np.load('/work/scratch/staticTz_ground2.npy');p=seg_vit(fs,Tg1,Tg2,lam=lam,hi=2,maxstep=.5)
 seq=[Tg1]+[Z[f'{i}_T'][p[j+1]] for j,i in enumerate(fs)]+[Tg2]
 print(lam,p,len(seq),'startend',np.degrees(Rot.from_matrix(seq[0][:3,:3]@seq[-1][:3,:3].T).magnitude()),'sum',sum(np.degrees(Rot.from_matrix(a[:3,:3]@b[:3,:3].T).magnitude()) for a,b in zip(seq,seq[1:])))
 for i in range(0,len(seq)-1):print(i,round(np.degrees(Rot.from_matrix(seq[i][:3,:3]@seq[i+1][:3,:3].T).magnitude()),1),end=' ')
 print()
PY"
```
<details><summary>output</summary>

```
syms 24
frames [80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830]
0.2 [0, 17, 17, 3, 8, 8, 3, 5, 3, 22, 22, 0] 12 startend 39.01631996659701 sum 371.3033853849319
0 18.9 1 55.3 2 49.6 3 19.2 4 8.3 5 26.6 6 39.5 7 46.2 8 51.4 9 14.9 10 41.3 
0.5 [0, 17, 17, 3, 8, 8, 3, 5, 3, 22, 22, 0] 12 startend 39.01631996659701 sum 371.3033853849319
0 18.9 1 55.3 2 49.6 3 19.2 4 8.3 5 26.6 6 39.5 7 46.2 8 51.4 9 14.9 10 41.3 
2.0 [0, 17, 17, 3, 8, 8, 3, 5, 3, 22, 22, 0] 12 startend 39.01631996659701 sum 371.3033853849319
0 18.9 1 55.3 2 49.6 3 19.2 4 8.3 5 26.6 6 39.5 7 46.2 8 51.4 9 14.9 10 41.3 

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/viterbi.py').read().split(\"Tinit=\")[0];exec(s)
import numpy as np
from scipy.spatial.transform import Rotation as Rot
Tinit=np.load('/work/scratch/staticTz_initial.npy');Tg1=np.load('/work/scratch/staticTz_ground1.npy');Tg2=np.load('/work/scratch/staticTz_ground2.npy');Tfin=np.load('/work/scratch/staticTz_final.npy')
segs=[('A',[i for i in range(150,321,10)],Tinit,Tg1),('B',[i for i in range(400,500,10)],Tg1,Tg2),('C',[i for i in range(700,831,10)],Tg2,Tfin)]
for lam in [3,5,10,20]:
 print('\\nL',lam)
 for name,fs,a,b in segs:
  p=seg_vit(fs,a,b,lam=lam,hi=10,maxstep=.15);seq=[a]+[Z[f'{i}_T'][p[j+1]] for j,i in enumerate(fs)]+[b]
  ds=[np.degrees(Rot.from_matrix(seq[j][:3,:3]@seq[j+1][:3,:3].T).magnitude()) for j in range(len(seq)-1)]
  scores=[1.6*get(i,'iou')[p[j+1]]+.9*get(i,'chrom')[p[j+1]]+.5*get(i,'grad')[p[j+1]]-2*np.abs(get(i,'dz')[p[j+1]]-.08) for j,i in enumerate(fs)]
  print(name,'states',p[1:-1],'dR',np.round(ds,1).tolist(),'meanData',round(np.mean(scores),2))
PY"
```
<details><summary>output</summary>

```
syms 24
frames [80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830]

L 3
A states [1, 1, 1, 21, 21, 23, 3, 1, 10, 18, 1, 3, 21, 8, 4, 21, 21, 4] dR [27.5, 5.7, 4.8, 8.9, 3.3, 37.5, 41.7, 34.5, 38.0, 38.2, 18.8, 40.7, 25.9, 27.5, 70.2, 49.5, 16.2, 47.0, 46.1] meanData -0.04
B states [17, 17, 3, 8, 8, 3, 5, 3, 22, 22] dR [18.9, 55.3, 49.6, 19.2, 8.3, 26.6, 39.5, 46.2, 51.4, 14.9, 41.3] meanData 1.29
C states [3, 22, 3, 0, 3, 3, 21, 5, 16, 3, 3, 3, 3, 3] dR [16.5, 40.6, 28.9, 27.6, 42.6, 9.9, 24.1, 52.4, 31.4, 25.2, 8.3, 6.7, 6.1, 27.0, 29.9] meanData 0.67

L 5
A states [1, 1, 1, 21, 21, 23, 3, …[+1440 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
Z=np.load('/work/scratch/candidate_metrics.npz')
for i in [400,410,420,430,440,450,460,470,480,490]:
 s=1.6*Z[f'{i}_iou']+.9*Z[f'{i}_chrom']+.5*Z[f'{i}_grad']-2*np.abs(Z[f'{i}_dz']-.08)
 print(i,np.round(np.sort(s)[::-1][:10],2))
PY"
```
<details><summary>output</summary>

```
400 [1.22 1.07 1.03 0.95 0.92 0.9  0.86 0.84 0.83 0.76]
410 [1.15 1.01 1.01 1.   0.95 0.94 0.89 0.88 0.87 0.85]
420 [1.99 1.29 1.16 1.05 0.91 0.9  0.88 0.85 0.84 0.84]
430 [1.44 1.36 1.15 1.12 1.07 1.06 0.97 0.89 0.86 0.85]
440 [1.62 1.43 1.41 1.36 1.34 1.27 1.27 1.22 1.08 1.06]
450 [1.8  1.32 1.28 1.26 1.24 1.21 1.21 1.19 1.18 1.11]
460 [1.51 1.47 1.46 1.46 1.45 1.41 1.32 1.31 1.3  1.28]
470 [2.1  1.53 1.42 1.35 1.34 1.34 1.3  1.28 1.24 1.24]
480 [1.36 1.   0.95 0.93 0.89 0.87 0.81 0.81 0.8  0.79]
490 [1.36 1.05 0.98 0.94 0.91 0.89 0.89 0.87 0.86 0.81]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
Z=np.load('/work/scratch/candidate_metrics.npz')
def get(i,k):return Z[f'{i}_{k}']
def score(i,k):return 1.6*get(i,'iou')[k]+.9*get(i,'chrom')[k]+.5*get(i,'grad')[k]-2*np.abs(get(i,'dz')[k]-.08)
def rd(a,b):return np.degrees(Rot.from_matrix(a[:3,:3]@b[:3,:3].T).magnitude())
for lam in [0,5,20,100]:
 vals=[]
 for k in range(24):vals.append((score(410,k)-lam*np.radians(rd(Z['400_T'][17],Z['410_T'][k])),k,score(410,k),rd(Z['400_T'][17],Z['410_T'][k])))
 vals.sort(reverse=True);print(lam,vals[:10])
PY"
```
<details><summary>output</summary>

```
syms 24
0 [(1.1452594061348271, 3, 1.1452594061348271, 78.34437204230716), (1.0140177988204728, 10, 1.0140177988204728, 158.05469302915893), (1.006874381786007, 16, 1.006874381786007, 152.05242334271117), (1.001629998692893, 18, 1.001629998692893, 176.82494061883912), (0.94989585438237, 8, 0.94989585438237, 162.61157422226194), (0.9431279253046153, 1, 0.9431279253046153, 138.79696213865967), (0.8887985178613876, 19, 0.8887985178613876, 133.38313448310896), (0.8792977170414745, 9, 0.8792977170414745, 50.07518034923678), (0.8690836668834961, 17, 0.8690836668834961, 55.339308524871754), (0.8477914281509672, 21, 0.8477914281509672, 74.31227121177606)]
5 [(-3.490586136079255, 9, 0.8792977170414745, 50.07518034923678), (-3.9601820307963957, 17, 0.8690836668834961, 55.339308524871754), (-5.637177 …[+1821 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/view_vit.py <<'PY'
import numpy as np,sys,cv2,h5py
sys.path.insert(0,'/work/scratch');from viterbi import seg_vit,get,Z
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:]
Tinit=np.load('/work/scratch/staticTz_initial.npy');Tg1=np.load('/work/scratch/staticTz_ground1.npy');Tg2=np.load('/work/scratch/staticTz_ground2.npy');Tfin=np.load('/work/scratch/staticTz_final.npy')
segs=[('A',[i for i in range(150,321,10)],Tinit,Tg1),('B',[i for i in range(400,500,10)],Tg1,Tg2),('C',[i for i in range(700,831,10)],Tg2,Tfin)]
for name,fs,s,e in segs:
 p=seg_vit(fs,s,e,lam=.5,hi=2,maxstep=.5);arr=[]
 for j,i in enumerate(fs):
  T=Z[f'{i}_T'][p[j+1]];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=280);mm,rr,dd=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();ren=np.zeros((h,w,3),np.uint8);ren[m]=(r[m]*255).astype(np.uint8);orig=im[y0:y0+h,x0:x0+w];sep=np.full((h,4,3),255,np.uint8);row=np.hstack([orig,sep,ren]);row=cv2.resize(row,(420,210));row=cv2.cvtColor(row,cv2.COLOR_RGB2BGR);cv2.putText(row,f'{i} s{p[j+1]}',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);arr.append(row)
 while len(arr)%2:arr.append(np.zeros_like(arr[0]))
 rows=[np.vstack(arr[j:j+2]) for j in range(0,len(arr),2)];cv2.imwrite(f'/work/scratch/vit_{n …[+63 chars]
```
<details><summary>output</summary>

```
syms 24
frames [80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830]

LAMBDA 0.2
A
 150: state 1 dR  0.5 dT0.219
 160: state 1 dR  0.1 dT0.081
 170: state 1 dR  0.1 dT0.095
 180: state21 dR  0.2 dT0.339
 190: state21 dR  0.1 dT0.189
 200: state23 dR  0.7 dT0.175
 210: state 3 dR  0.7 dT0.116
 220: state 1 dR  0.6 dT0.183
 230: state10 dR  0.7 dT0.171
 240: state18 dR  0.7 dT0.134
 250: state 1 dR  0.3 dT0.545
 260: state 3 dR  0.7 dT0.117
 270: state21 dR  0.5 dT0.097
 280: state 8 dR  0.5 dT0.084
 290: state 4 dR  1.2 dT0.098
 300: state21 dR  0.9 dT0.329
 310: state21 dR  0.3 dT0.470
 320:  …[+6040 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,os
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
for seg in [range(150,321,10),range(390,501,10),range(690,831,10)]:
 print('\\n')
 for i in seg:
  p=f'/work/scratch/dynT_{i}.npy'
  if os.path.exists(p):
   T=np.load(p);print(i,'key',np.round(T[:3,3],3),'base',np.round(B[i,:3,3],3),'diff',np.round(T[:3,3]-B[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
700 key [0.366 1.093 3.214] base [0.382 1.124 3.299] diff [-0.016 -0.031 -0.086]
710 key [0.397 1.07  3.165] base [0.417 1.125 3.365] diff [-0.02  -0.054 -0.199]
720 key [0.427 1.046 3.213] base [0.477 1.054 3.247] diff [-0.05  -0.008 -0.034]
730 key [0.511 0.994 3.226] base [0.541 1.009 3.289] diff [-0.03  -0.016 -0.063]
740 key [0.604 0.884 3.375] base [0.621 0.862 3.257] diff [-0.017  0.023  0.118]
750 key [0.645 0.683 3.317] base [0.67  0.689 3.262] diff [-0.025 -0.006  0.054]
760 key [0.676 0.509 3.181] base [0.746 0.517 3.295] diff [-0.07  -0.008 -0.114]
770 key [0.615 0.216 3.389] base [0.563 0.123 3.228] diff [0.052 0.093 0.161]
780 key [0.65  0.149 3.517] base [0.473 0.023 3.323] diff [0.177 0.125 0.194]
790 key [0.412 0.107 3.315] base [ 0.323 -0.024  3.366] diff [ 0.089  0.131 - …[+325 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_intermediate.py <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch')
# load definitions without executing viterbi main
s=open('/work/scratch/viterbi.py').read().split('Tinit=')[0];exec(s)
Tinit=np.load('/work/scratch/staticTz_initial.npy');Tg1=np.load('/work/scratch/staticTz_ground1.npy');Tg2=np.load('/work/scratch/staticTz_ground2.npy');Tfin=np.load('/work/scratch/staticTz_final.npy')
segs=[('A',[i for i in range(150,321,10)],Tinit,Tg1),('B',[i for i in range(400,500,10)],Tg1,Tg2),('C',[i for i in range(700,831,10)],Tg2,Tfin)]
keys={0:Tinit.copy(),149:Tinit.copy(),325:Tg1.copy(),389:Tg1.copy(),390:Tg1.copy(),499:Tg2.copy(),500:Tg2.copy(),689:Tg2.copy(),690:Tg2.copy(),830:Tfin.copy(),921:Tfin.copy()}
for name,fs,sT,eT in segs:
 p=seg_vit(fs,sT,eT,lam=.5,hi=2,maxstep=.5)
 for j,i in enumerate(fs):keys[i]=Z[f'{i}_T'][p[j+1]].copy()
 print(name,p)
def slerpT(A,B,alpha):
 TA=Rot.from_matrix(A[:3,:3]);TB=Rot.from_matrix(B[:3,:3]);rA=TA.as_quat();rB=TB.as_quat();
 # ensure shortest
 if np.dot(rA,rB)<0:rB=-rB
 # scipy slerp unavailable? use Rotation.from_rotvec rotvec delta
 d=(TB*TA.inv()).as_rotvec();R=Rot.from_rotvec(d*alpha)*TA
 T=np.eye(4);T[:3,:3]=R.as_matrix();T[:3,3]=(1-alpha)*A[:3,3]+alpha*B[:3,3];return T
kf=sorted(keys);Ts=np.zeros((922,4,4))
for i in range(922):
 if i<kf[0]:Ts[i]=keys[kf[0]]
 elif i>=kf[-1]:Ts[i]=keys[kf[-1]]
 else:
  j=np.searchsorted(kf,i,side='r …[+342 chars]
```
<details><summary>output</summary>

```
syms 24
frames [80, 100, 120, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830]
A [0, 1, 1, 1, 21, 21, 23, 3, 1, 10, 18, 1, 3, 21, 8, 4, 21, 21, 4, 0]
B [0, 17, 17, 3, 8, 8, 3, 5, 3, 22, 22, 0]
C [0, 3, 22, 3, 0, 3, 3, 21, 5, 16, 3, 3, 3, 3, 3, 0]
0 [0.008 0.217 3.48 ] [ 0.19 -0.13  3.13]
149 [0.008 0.217 3.48 ] [ 0.19 -0.13  3.13]
150 [-0.035  0.234  3.694] [-0.21  0.5   3.08]
200 [-0.546  0.163  3.632] [-0.72 -0.42  2.94]
240 [-0.602  0.607  3.477] [ 0.33  0.06 -2.71]
300 [-0.283  1.04   3.259] [-0.65  0.13 -3.06]
320 [-0.237  1.196  3.635] [ 0.31  0.82 -2.65]
325 [-0.205  1.108  3.391] [-0.41  0.07 - …[+578 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_traj.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];A=np.load('/work/scratch/T_intermediate.npy')
def ev(Ts,step=5):
 rows=[]
 for i in range(0,922,step):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(Ts[i][None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();d=dd[0].cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];sc,z=measure(im[y0:y0+h,x0:x0+w],r,m,o,p);mask=m&o;dz=np.median(d[mask])-np.median(DEPS[i][y0:y0+h,x0:x0+w][mask]) if mask.sum()>20 else np.nan;rows.append((i,z['iou'],z['chrom'],z['grad'],dz,z['gray']))
 return np.array(rows)
for name,Ts in [('baseline',B),('intermediate',A)]:
 R=ev(Ts);np.savez(f'/work/scratch/eval_{name}.npz',R=R);print(name)
 for lo,hi in [(0,150),(150,325),(325,390),(390,500),(500,690),(690,830),(830,922)]:
  Q=R[(R[:,0]>=lo)&(R[:,0]<hi)];print(lo,hi,'iou',np.nanme …[+205 chars]
```
<details><summary>output</summary>

```
syms 24
baseline
0 150 iou 0.884 chrom 0.995 grad 0.168 dz 0.101 gray 0.398
150 325 iou 0.672 chrom 0.15 grad 0.019 dz 0.538 gray 0.012
325 390 iou 0.88 chrom -0.012 grad -0.01 dz 0.082 gray -0.05
390 500 iou 0.846 chrom -0.017 grad -0.004 dz 0.106 gray -0.045
500 690 iou 0.924 chrom -0.077 grad 0.008 dz 0.141 gray -0.082
690 830 iou 0.64 chrom 0.027 grad -0.022 dz 0.498 gray -0.058
830 922 iou 0.598 chrom -0.169 grad 0.009 dz 0.1 gray -0.039
intermediate
0 150 iou 0.877 chrom 0.914 grad 0.174 dz 0.164 gray 0.439
150 325 iou 0.712 chrom 0.127 grad -0.001 dz 0.611 gray 0.022
325 390 iou 0.922 chrom 1.014 grad 0.12 dz 0.094 gray 0.151
390 500 iou 0.729 chrom 0.353 grad 0.016 dz 0.078 gray 0.047
500 690 iou 0.884 chrom 0.728 grad -0.021 dz 0.184 gray 0.015
690 830 iou 0.82 chrom 0.605 grad 0. …[+85 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];A=np.load('/work/scratch/staticTz_initial.npy')
for i in [0,20,40,60,80,100,120,140]:
 print(i,'rd',np.degrees(Rot.from_matrix(A[:3,:3]@B[i,:3,:3].T).magnitude()),'dt',np.linalg.norm(A[:3,3]-B[i,:3,3]),'t',np.round(B[i,:3,3],3))
PY"
```
<details><summary>output</summary>

```
0 rd 2.268804665318086 dt 0.15846798128660552 t [0.025 0.203 3.323]
20 rd 2.268804665318086 dt 0.15292928769717434 t [0.025 0.203 3.329]
40 rd 2.268804665318086 dt 0.13911551813968712 t [0.025 0.204 3.343]
60 rd 2.268804665318086 dt 0.1408742134199243 t [0.025 0.204 3.341]
80 rd 1.7099888182728298 dt 0.12771378033233735 t [0.028 0.21  3.354]
100 rd 0.7900513535871586 dt 0.02442303070137568 t [0.031 0.213 3.487]
120 rd 3.496134163985996 dt 0.023546115220443184 t [0.029 0.226 3.473]
140 rd 6.595527696420984 dt 0.01618940049558445 t [0.022 0.223 3.486]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
b=np.load('/work/scratch/eval_baseline.npz')['R'];a=np.load('/work/scratch/eval_intermediate.npz')['R']
for i in range(0,len(b)):
 if (150<=b[i,0]<325) or (390<=b[i,0]<500) or (690<=b[i,0]<830):
  print(int(b[i,0]),'B iou/ch/dz',round(b[i,1],2),round(b[i,2],2),round(b[i,4],2),'A',round(a[i,1],2),round(a[i,2],2),round(a[i,4],2))
PY"
```
<details><summary>output</summary>

```

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/add_baseline_candidates.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
Z=np.load('/work/scratch/candidate_metrics.npz');B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float32)
frames=sorted(set(int(k.split('_')[0]) for k in Z.files));out={k:Z[k] for k in Z.files}
for i in frames:
 T=B[i];cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();dep=dd[0].cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];sc,d=measure(im[y0:y0+h,x0:x0+w],r,m,o,p);mask=m&o;dz=np.median(dep[mask])-np.median(DEPS[i][y0:y0+h,x0:x0+w][mask]) if mask.sum()>20 else 1;score=1.6*d['iou']+0.9*d['chrom']+0.5*d['grad']-2*abs(dz-.08)
 def ap(k,v):out[f'{i}_{k}']=np.append(Z[f'{i}_{k}'],v)
 ap('T',T);ap('score',score);ap('iou',d['iou']);ap('chrom',d['chrom']);ap('grad',d['grad']);ap('gray',d['gray']);ap('sat',d['sat']);ap('ca',d['c …[+268 chars]
```
<details><summary>output</summary>

```
syms 24
80 base 2.43 dynbest 1.73
100 base 2.26 dynbest 2.27
120 base 1.79 dynbest 0.88
140 base 1.65 dynbest 1.84
150 base 1.83 dynbest 0.98
160 base 2.16 dynbest 1.12
170 base 2.15 dynbest 0.99
180 base 1.59 dynbest 1.2
190 base -5.38 dynbest -4.56
200 base 0.53 dynbest -0.84
210 base -5.1 dynbest -4.52
220 base -3.13 dynbest -1.11
230 base 1.22 dynbest 1.23
240 base 0.17 dynbest 0.64
250 base 0.46 dynbest 0.71
260 base 0.49 dynbest 1.65
270 base 1.11 dynbest 1.35
280 base 0.96 dynbest 1.44
290 base 0.77 dynbest 1.52
300 base 1.19 dynbest 1.34
310 base 1.2 dynbest 1.47
320 base 1.09 dynbest 1.37
330 base 1.18 dynbest 1.61
390 base 1.35 dynbest 1.42
400 base 1.24 dynbest 1.22
410 base 1.08 dynbest 1.15
420 base 0.86 dynbest 1.99
430 base 0.97 dynbest 1.44
440 base 0.96 dynbest 1.62
450 ba …[+585 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/viterbi_aug.py <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
Z=np.load('/work/scratch/candidate_metrics_aug.npz')
def get(i,k):return Z[f'{i}_{k}']
def rd(A,B):return Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude()
def score(i,k):return 1.6*get(i,'iou')[k]+.9*get(i,'chrom')[k]+.5*get(i,'grad')[k]-2*np.abs(get(i,'dz')[k]-.08)
def seg_vit(frames,startT,endT,lam=.5,hi=3.0,maxstep=.5):
 states=[[startT]]+[[Z[f'{i}_T'][k] for k in range(len(Z[f'{i}_T']))] for i in frames]+[[endT]]
 scores=[np.array([0.])]+[1.6*get(i,'iou')+.9*get(i,'chrom')+.5*get(i,'grad')-2*np.abs(get(i,'dz')-.08) for i in frames]+[np.array([0.])]
 dp=scores[0].copy();backs=[]
 for j in range(1,len(states)):
  ndp=np.full(len(states[j]),-1e18);bk=np.zeros(len(states[j]),int)
  for b,Tb in enumerate(states[j]):
   best=-1e18;bi=0
   for a,Ta in enumerate(states[j-1]):
    ang=rd(Ta,Tb);tr=lam*ang+hi*max(0,ang-maxstep)**2
    val=dp[a]-tr
    if val>best:best=val;bi=a
   ndp[b]=scores[j][b]+best;bk[b]=bi
  dp=ndp;backs.append(bk)
 path=[int(np.argmax(dp))];idx=path[0]
 for bk in backs[::-1]:idx=int(bk[idx]);path.append(idx)
 return path[::-1]
def avgT(Ts):
 Ts=np.array(Ts);T=np.eye(4);T[:3,:3]=Rot.from_matrix(Ts[:,:3,:3]).mean().as_matrix();T[:3,3]=np.median(Ts[:,:3,3],0);return T
if __name__=='__main__':
 B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Tinit=avgT(B[:66]);Tg1=np.load('/work/scratch/staticTz_ground1.npy');Tg2=np …[+782 chars]
```
<details><summary>output</summary>

```

L 0.2
Traceback (most recent call last):
  File "/work/scratch/viterbi_aug.py", line 32, in <module>
    p=seg_vit(fs,s,e,lam=lam,hi=3,maxstep=mx);seq=[s]+[Z[f'{i}_T'][p[j+1]] for j,i in enumerate(fs)]+[e]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/viterbi_aug.py", line 16, in seg_vit
    ang=rd(Ta,Tb);tr=lam*ang+hi*max(0,ang-maxstep)**2
        ^^^^^^^^^
  File "/work/scratch/viterbi_aug.py", line 5, in rd
    def rd(A,B):return Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude()
                                                ~^^^^^^^
IndexError: invalid index to scalar variable.

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
Z=np.load('/work/scratch/candidate_metrics_aug.npz')
for k in ['150_T','150_iou','400_T']:print(k,Z[k].shape,Z[k].dtype)
PY"
```
<details><summary>output</summary>

```
150_T (400,) float64
150_iou (25,) float64
400_T (400,) float64

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/viterbi_aug.py';s=open(p).read();s=s.replace(\"def get(i,k):return Z[f'{i}_{k}']\", \"def get(i,k):\\n v=Z[f'{i}_{k}']\\n return v.reshape(25,4,4) if k=='T' else v\")
s=s.replace(\"Z[f'{i}_T'][k]\", \"get(i,'T')[k]\")
open(p,'w').write(s)
PY
python /work/scratch/viterbi_aug.py"
```
<details><summary>output</summary>

```

L 0.2
Traceback (most recent call last):
  File "/work/scratch/viterbi_aug.py", line 34, in <module>
    p=seg_vit(fs,s,e,lam=lam,hi=3,maxstep=mx);seq=[s]+[Z[f'{i}_T'][p[j+1]] for j,i in enumerate(fs)]+[e]
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/viterbi_aug.py", line 10, in seg_vit
    states=[[startT]]+[[get(i,'T')[k] for k in range(len(Z[f'{i}_T']))] for i in frames]+[[endT]]
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/viterbi_aug.py", line 10, in <listcomp>
    states=[[startT]]+[[get(i,'T')[k] for k in range(len(Z[f'{i}_T']))] for i in frames]+[[endT]]
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/scratch/viterbi_aug.py", line 10, in <listcomp>
    sta …[+191 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i \"s/range(len(Z\\[f'{i}_T'\\]))/range(25)/g; s/Z\\[f'{i}_T'\\]\\[p\\[j+1\\]\\]/get(i,'T')[p[j+1]]/\" /work/scratch/viterbi_aug.py && python /work/scratch/viterbi_aug.py"
```
<details><summary>output</summary>

```

L 0.2
A states [24, 24, 24, 24, 24, 24, 3, 1, 10, 18, 1, 3, 21, 8, 4, 21, 21, 4]
 dR [3.9, 4.6, 3.9, 5.4, 19.6, 3.9, 35.5, 34.5, 38.0, 38.2, 18.8, 40.7, 25.9, 27.5, 70.2, 49.5, 16.2, 47.0, 46.1]
 score [1.83, 2.16, 2.15, 1.59, -5.38, 0.53, -4.52, -1.63, 1.08, -0.24, 0.56, 1.65, 1.09, 0.52, 1.17, 1.19, 1.34, 0.83] mean 0.33
B states [17, 17, 3, 8, 8, 3, 5, 3, 22, 22]
 dR [18.9, 55.3, 49.6, 19.2, 8.3, 26.6, 39.5, 46.2, 51.4, 14.9, 41.3]
 score [0.95, 0.87, 1.99, 0.82, 1.05, 1.8, 1.47, 2.1, 0.93, 0.89] mean 1.29
C states [3, 22, 3, 0, 3, 3, 21, 1, 21, 3, 3, 3, 3, 3]
 dR [16.5, 40.6, 28.9, 27.6, 42.6, 9.9, 24.1, 94.0, 31.4, 89.4, 8.3, 6.7, 6.1, 27.0, 29.9]
 score [2.31, 1.09, 1.99, 1.25, 1.62, 1.9, 1.33, -5.64, -0.47, 1.77, 1.94, 1.96, 1.98, 1.45] mean 1.03

L 0.5
A states [24, 24, 24, 24, 24 …[+2252 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_intermediate2.py <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from viterbi_aug import seg_vit,get,avgT
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Tinit=avgT(B[:66]);Tg1=np.load('/work/scratch/staticTz_ground1.npy');Tg2=np.load('/work/scratch/staticTz_ground2.npy');Tfin=np.load('/work/scratch/staticTz_final.npy')
segs=[('A',[i for i in range(150,321,10)],Tinit,Tg1,.55),('B',[i for i in range(400,500,10)],Tg1,Tg2,.9),('C',[i for i in range(700,831,10)],Tg2,Tfin,.75)]
keys={0:Tinit.copy(),149:Tinit.copy(),325:Tg1.copy(),389:Tg1.copy(),390:Tg1.copy(),499:Tg2.copy(),500:Tg2.copy(),689:Tg2.copy(),690:Tg2.copy(),830:Tfin.copy(),921:Tfin.copy()}
for name,fs,sT,eT,mx in segs:
 p=seg_vit(fs,sT,eT,lam=.5,hi=3,maxstep=mx)
 for j,i in enumerate(fs):keys[i]=get(i,'T')[p[j+1]].copy()
 print(name,p)
def slerpT(A,B,alpha):
 TA=Rot.from_matrix(A[:3,:3]);TB=Rot.from_matrix(B[:3,:3]);d=(TB*TA.inv()).as_rotvec();R=Rot.from_rotvec(d*alpha)*TA
 T=np.eye(4);T[:3,:3]=R.as_matrix();T[:3,3]=(1-alpha)*A[:3,3]+alpha*B[:3,3];return T
kf=sorted(keys);Ts=np.zeros((922,4,4))
for i in range(922):
 if i<=kf[0]:Ts[i]=keys[kf[0]]
 elif i>=kf[-1]:Ts[i]=keys[kf[-1]]
 else:
  j=np.searchsorted(kf,i,side='right')-1;a,b=kf[j],kf[j+1];Ts[i]=slerpT(keys[a],keys[b],(i-a)/(b-a))
np.save('/work/scratch/T_intermediate2.npy',Ts)
print('Tinit',Tinit)
PY
python /work/scratch/build_intermediate2 …[+4 chars]
```
<details><summary>output</summary>

```
A [0, 24, 24, 24, 24, 24, 24, 3, 1, 10, 18, 1, 3, 21, 8, 4, 21, 21, 4, 0]
B [0, 17, 17, 3, 8, 8, 3, 5, 3, 22, 22, 0]
C [0, 3, 22, 3, 0, 3, 3, 21, 5, 16, 3, 3, 3, 3, 3, 0]
Tinit [[-0.9873783  -0.0260899   0.1562159   0.02513988]
 [ 0.01345305 -0.99658982 -0.08141092  0.20413534]
 [ 0.15780718 -0.0782818   0.98436216  3.33984566]
 [ 0.          0.          0.          1.        ]]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed 's#T_intermediate.npy#T_intermediate2.npy#; s#eval_intermediate.npz#eval_intermediate2.npz#' /work/scratch/eval_traj.py > /work/scratch/eval_traj2.py && python /work/scratch/eval_traj2.py"
```
<details><summary>output</summary>

```
syms 24
baseline
0 150 iou 0.884 chrom 0.995 grad 0.168 dz 0.101 gray 0.398
150 325 iou 0.672 chrom 0.15 grad 0.019 dz 0.538 gray 0.012
325 390 iou 0.88 chrom -0.012 grad -0.01 dz 0.082 gray -0.05
390 500 iou 0.846 chrom -0.017 grad -0.004 dz 0.106 gray -0.045
500 690 iou 0.924 chrom -0.077 grad 0.008 dz 0.141 gray -0.082
690 830 iou 0.64 chrom 0.027 grad -0.022 dz 0.498 gray -0.058
830 922 iou 0.598 chrom -0.169 grad 0.009 dz 0.1 gray -0.039
intermediate
0 150 iou 0.885 chrom 1.049 grad 0.275 dz 0.054 gray 0.474
150 325 iou 0.719 chrom 0.273 grad 0.021 dz 0.554 gray 0.047
325 390 iou 0.922 chrom 1.014 grad 0.12 dz 0.094 gray 0.151
390 500 iou 0.729 chrom 0.353 grad 0.016 dz 0.078 gray 0.047
500 690 iou 0.884 chrom 0.728 grad -0.021 dz 0.184 gray 0.015
690 830 iou 0.82 chrom 0.605 grad 0.0 …[+84 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "cat > /work/scratch/refine_static_mask.py <<'PY'
import numpy as np,h5py,cv2,sys,trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();lo,hi=geo.bounds
corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
def raster(poly,x0,y0,w,h):
 m=np.zeros((h,w),np.uint8);p=np.round(poly-[x0,y0]).astype(np.int32);cv2.fillConvexPoly(m,cv2.convexHull(p),1);return m.astype(bool)
def refine(name,frames):
 T0=np.load(f'/work/scratch/staticTz_{name}.npy');cam=Tcw@T0;c0=cam[:3,3];z0=c0[2];uv0=np.array([K[0,0]*c0[0]/c0[2]+K[0,2],K[1,1]*c0[1]/c0[2]+K[1,2]])
 crops=[]
 for i in frames:
  x0,y0,w,h=crop_box(bb[i],pad=25,min_size=240);o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];vis=~cv2.dilate(p.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool);crops.append((o,vis,x0,y0,w,h))
 def make(x):
  z,du,dv,rx,ry,rz=x;R=Rot.from_rotvec([rx,ry,rz]).as_matrix().astype(np.float32)@T0[:3,:3];tcam=np.array([(uv0[0]+du-K[0,2])*z/K[0,0],(uv0[1]+dv-K[1,2])*z/K[1,1],z]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=tcam;return Twc@T
 def f(x):
  T=make(x);pc=(Tcw@(T@corners.T)).T;u=K[0,0]*pc[:,0]/pc[:,2]+K[0,2];v=K[1,1]*pc[:,1]/pc[:,2]+K[1, …[+940 chars]
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/scratch/refine_static_mask.py", line 33, in <module>
    refine(name,frames)
  File "/work/scratch/refine_static_mask.py", line 27, in refine
    r=minimize(f,seed,method='Nelder-Mead',options={'maxiter':2000,'xatol':1e-6,'fatol':1e-8})
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_minimize.py", line 772, in minimize
    res = _minimize_neldermead(fun, x0, args, callback, bounds=bounds,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/scipy/optimize/_optimize.py", line 851, in _minimize_neldermead
    fsim[k] = func(sim[k])
              ^^^^^^^^^^^^
  File "/op …[+390 chars]
```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "sed -i 's/(z-z0)\\*\\*2/(x[0]-z0)**2/; s/(du\\/100)\\*\\*2/(x[1]\\/100)**2/; s/(dv\\/100)\\*\\*2/(x[2]\\/100)**2/; s/(rx\\*rx+ry\\*ry+rz\\*rz)/(x[3]**2+x[4]**2+x[5]**2)/' /work/scratch/refine_static_mask.py && python /work/scratch/refine_static_mask.py"
```
<details><summary>output</summary>

```
ground1 oldfun 0.059861190158116195 new 0.05316542795751182 x [ 3.32333868e+00 -1.72135765e-03 -1.74984018e-03  4.35251388e-03
 -2.46953766e-02  2.00355414e-03] T [-0.2015072   1.08888143  3.3341099 ]
Traceback (most recent call last):
  File "/work/scratch/refine_static_mask.py", line 33, in <module>
    refine(name,frames)
  File "/work/scratch/refine_static_mask.py", line 12, in refine
    T0=np.load(f'/work/scratch/staticTz_{name}.npy');cam=Tcw@T0;c0=cam[:3,3];z0=c0[2];uv0=np.array([K[0,0]*c0[0]/c0[2]+K[0,2],K[1,1]*c0[1]/c0[2]+K[1,2]])
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/conda/lib/python3.11/site-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^ …[+97 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/refine_static_mask.py';s=open(p).read();s=s.replace(\"for name,a,b in [('ground1',330,386),('ground2a',510,581),('ground2',660,686),('final',835,922)]:\", \"for name,a,b in [('ground2',510,686),('final',835,922)]:\")
open(p,'w').write(s)
PY
python /work/scratch/refine_static_mask.py"
```
<details><summary>output</summary>

```
ground2 oldfun 0.12145198270748325 new 0.0818009604275707 x [ 3.20475445e+00  1.03781825e-03 -4.94199502e-04 -5.82248874e-04
  2.57750828e-03  9.89794837e-02] T [0.35744275 1.08775559 3.1912576 ]
final oldfun 0.06723245552698409 new 0.039598837967706545 x [3.39930879e+00 6.03953593e-05 1.07395074e-03 2.39620857e-02
 6.59751255e-04 1.29712556e-03] T [0.10799951 0.2185874  3.43363296]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/check_ref.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
for name,fs in [('ground1',range(330,386,6)),('ground2',list(range(510,581,6))+list(range(660,686,4))),('final',range(835,922,6))]:
 for kind in ['staticTz','staticRef']:
  T=np.load(f'/work/scratch/{kind}_{name}.npy');vals=[]
  for i in fs:
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();sc,z=measure(im[y0:y0+h,x0:x0+w],r,m,oms[i,y0:y0+h,x0:x0+w],pms[i,y0:y0+h,x0:x0+w]);vals.append([z['iou'],z['chrom'],z['grad']])
  print(name,kind,np.round(np.mean(vals,0),3))
PY
python /work/scratch/check_ref.py"
```
<details><summary>output</summary>

```
syms 24
ground1 staticTz [0.921 1.025 0.119]
ground1 staticRef [0.924 1.084 0.138]
ground2 staticTz [ 0.888  0.718 -0.018]
ground2 staticRef [0.893 0.719 0.048]
final staticTz [0.919 0.846 0.089]
final staticRef [0.913 0.809 0.102]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cp /work/scratch/T_intermediate2.npy /dev/null; python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'];Ts=B[:66];T=np.eye(4);T[:3,:3]=Rot.from_matrix(Ts[:,:3,:3]).mean().as_matrix();T[:3,3]=np.median(Ts[:,:3,3],0);np.save('/work/scratch/staticTz_initialbase.npy',T);print(T)
PY
python - <<'PY'
p='/work/scratch/refine_static_mask.py';s=open(p).read();s=s.replace(\"for name,a,b in [('ground2',510,686),('final',835,922)]:\",\"for name,a,b in [('initialbase',0,66)]:\")
open('/work/scratch/refine_initial.py','w').write(s)
PY
python /work/scratch/refine_initial.py"
```
<details><summary>output</summary>

```
[[-0.9873783  -0.0260899   0.1562159   0.02513988]
 [ 0.01345305 -0.99658982 -0.08141092  0.20413534]
 [ 0.15780718 -0.0782818   0.98436216  3.33984566]
 [ 0.          0.          0.          1.        ]]
initialbase oldfun 0.11246461812918795 new 0.0774925543593843 x [ 3.23480465e+00  2.36532902e-03 -4.80107851e-04 -2.87367493e-04
 -9.66595998e-02 -1.55731373e-03] T [0.02503545 0.19960585 3.27320828]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f:dep={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(66)}
for fn in ['staticTz_initialbase','staticRef_initialbase']:
 T=np.load('/work/scratch/'+fn+'.npy');v=[]
 for i in range(0,66,5):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();d=dd[0].cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=np.zeros_like(o);s,z=measure(im[y0:y0+h,x0:x0+w],r,m,o,p);m2=m&o;dz=np.median(d[m2])-np.median(dep[i][y0:y0+h,x0:x0+w][m2]);v.append([z['iou'],z['chrom'],z['grad'],dz])
 print(fn,np.mean(v,0))
PY"
```
<details><summary>output</summary>

```
syms 24
staticTz_initialbase [0.9145325  1.06253334 0.26853427 0.00389767]
staticRef_initialbase [ 0.90126832  0.94886688  0.01631106 -0.06662621]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_intermediate3.py <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from viterbi_aug import seg_vit,get
Tinit=np.load('/work/scratch/staticTz_initialbase.npy');Tg1=np.load('/work/scratch/staticRef_ground1.npy');Tg2=np.load('/work/scratch/staticRef_ground2.npy');Tfin=np.load('/work/scratch/staticRef_final.npy')
segs=[('A',[i for i in range(150,321,10)],Tinit,Tg1,.55),('B',[i for i in range(400,500,10)],Tg1,Tg2,.9),('C',[i for i in range(700,831,10)],Tg2,Tfin,.75)]
keys={0:Tinit.copy(),149:Tinit.copy(),325:Tg1.copy(),389:Tg1.copy(),390:Tg1.copy(),499:Tg2.copy(),500:Tg2.copy(),689:Tg2.copy(),690:Tg2.copy(),830:Tfin.copy(),921:Tfin.copy()}
for name,fs,sT,eT,mx in segs:
 p=seg_vit(fs,sT,eT,lam=.5,hi=3,maxstep=mx)
 for j,i in enumerate(fs):keys[i]=get(i,'T')[p[j+1]].copy()
 print(name,p)
def slerpT(A,B,alpha):
 TA=Rot.from_matrix(A[:3,:3]);TB=Rot.from_matrix(B[:3,:3]);d=(TB*TA.inv()).as_rotvec();R=Rot.from_rotvec(d*alpha)*TA;T=np.eye(4);T[:3,:3]=R.as_matrix();T[:3,3]=(1-alpha)*A[:3,3]+alpha*B[:3,3];return T
kf=sorted(keys);Ts=np.zeros((922,4,4))
for i in range(922):
 if i<=kf[0]:Ts[i]=keys[kf[0]]
 elif i>=kf[-1]:Ts[i]=keys[kf[-1]]
 else:
  j=np.searchsorted(kf,i,side='right')-1;a,b=kf[j],kf[j+1];Ts[i]=slerpT(keys[a],keys[b],(i-a)/(b-a))
np.save('/work/scratch/T_intermediate3.npy',Ts)
PY
python /work/scratch/build_intermediate3.py"
```
<details><summary>output</summary>

```
A [0, 24, 24, 24, 24, 24, 24, 3, 1, 10, 18, 1, 3, 21, 8, 4, 21, 21, 4, 0]
B [0, 17, 17, 3, 8, 8, 3, 5, 3, 22, 22, 0]
C [0, 3, 22, 3, 0, 3, 3, 21, 5, 16, 3, 3, 3, 3, 3, 0]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_translation.py <<'PY'
import numpy as np,h5py,cv2,sys,trimesh
from scipy.optimize import minimize
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();lo,hi=geo.bounds
corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
def raster(poly,x0,y0,w,h):
 m=np.zeros((h,w),np.uint8);p=np.round(poly-[x0,y0]).astype(np.int32);cv2.fillConvexPoly(m,cv2.convexHull(p),1);return m.astype(bool)
T=np.load('/work/scratch/T_intermediate3.npy');out=T.copy();areas=oms.sum((1,2))
for i in range(150,830):
 if not (150<=i<=324 or 390<=i<=499 or 690<=i<=829):continue
 Tc=T[i];cam=Tcw@Tc;c0=cam[:3,3];z0=c0[2];uv0=np.array([K[0,0]*c0[0]/c0[2]+K[0,2],K[1,1]*c0[1]/c0[2]+K[1,2]])
 x0,y0,w,h=crop_box(bb[i],pad=25,min_size=240);o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];vis=~cv2.dilate(p.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
 wgt=float(np.clip(areas[i]/7000,0.08,1.0))
 def make(x):
  z,du,dv=x;tcam=np.array([(uv0[0]+du-K[0,2])*z/K[0,0],(uv0[1]+dv-K[1,2])*z/K[1,1],z]);Tn=Tc.copy();Tn[:3,3]=(Twc@np.r_[tcam,1])[:3];return Tn
 def f(x):
  Tn=make(x);pc=(Tcw@(Tn@corners.T)).T;u=K[0,0]*pc[:,0]/pc[:,2]+K[0,2];v=K[1,1]*pc[:,1]/pc[:,2]+K[1,2];m=raster(np.c_[u,v],x0,y0,w,h)&vis;i …[+449 chars]
```
<details><summary>output</summary>

```
160 area 3617 w 0.52 x [ 3.387 -0.019  0.054] loss 0.0989
180 area 2222 w 0.32 x [3.491 0.    0.   ] loss 0.0876
200 area 1137 w 0.16 x [ 3.571  0.    -0.   ] loss 0.0708
220 area 1610 w 0.23 x [3.655 0.    0.   ] loss 0.0098
240 area 1326 w 0.19 x [3.429 0.    0.   ] loss 0.0126
260 area 2782 w 0.4 x [ 3.794  0.02  -0.014] loss 0.0552
280 area 3318 w 0.47 x [ 3.645  0.    -0.   ] loss 0.2763
300 area 5937 w 0.85 x [3.243 0.    0.   ] loss 0.1414
320 area 6843 w 0.98 x [ 3.633  0.    -0.   ] loss 0.3046
400 area 7671 w 1.0 x [3.615 0.    0.   ] loss 0.2819
420 area 6556 w 0.94 x [3.341 0.    0.   ] loss 0.1973
440 area 6943 w 0.99 x [ 3.407  0.099 -0.092] loss 0.51
460 area 8143 w 1.0 x [3.426 0.    0.   ] loss 0.2569
480 area 10577 w 1.0 x [3.168 0.    0.   ] loss 0.0539
700 area 8747 w 1 …[+377 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_dyn_compare.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
Ts=[np.load('/work/scratch/T_intermediate3.npy'),np.load('/work/scratch/T_translated.npy')]
for lo,hi in [(150,325),(390,500),(690,830)]:
 print('\\nPHASE',lo,hi)
 for ti,T in enumerate(Ts):
  vals=[]
  for i in range(lo,hi,5):
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(T[i][None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();d=dd[0].cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];sc,z=measure(im[y0:y0+h,x0:x0+w],r,m,o,p);mask=m&o;dz=np.median(d[mask])-np.median(DEPS[i][y0:y0+h,x0:x0+w][mask]) if mask.sum()>20 else np.nan;vals.append([z['iou'],z['chrom'],z['grad'],dz])
  print(ti,np.nanmean(vals,0))
PY
python /work/scratch/eval_dyn_compare.py"
```
<details><summary>output</summary>

```
syms 24

PHASE 150 325
0 [ 0.71862589  0.27319331  0.02133637 -0.26320908]
1 [ 0.71898269  0.27484898  0.01893657 -0.26233593]

PHASE 390 500
0 [ 0.72732682  0.35045535  0.01340896 -0.00402869]
1 [ 0.7276499   0.35097766  0.01443057 -0.00543537]

PHASE 690 830
0 [ 0.8201764   0.6045211   0.03324136 -0.35822957]
1 [ 0.82047849  0.60361639  0.03333348 -0.35944912]

```
</details>

**shell** (exit 1)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for name,p in [('baseline','/work/inputs/reconstruction.npz'),('inter3','/work/scratch/T_intermediate3.npy'),('trans','/work/scratch/T_translated.npy')]:
 T=np.load(p)['T_world_mesh'].astype(float) if p.endswith('.npz') else np.load(p)
 R=Rot.from_matrix(T[:,:3,:3]);dv=(R[1:]*R[:-1].inv()).as_rotvec();dr=np.degrees(np.linalg.norm(dv,axis=1));dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
 accR=np.degrees(np.linalg.norm(np.diff(dv,axis=0),axis=1));accT=np.linalg.norm(np.diff(T[:,:3,3],n=2,axis=0),axis=1)
 print(name,'dR',np.percentile(dr,[50,90,95,99,100]),'dT',np.percentile(dt,[50,90,95,99,100]),'aR',np.percentile(accR,[50,90,99,100]),'aT',np.percentile(accT,[50,90,99,100]))
 for seg in [(150,325),(390,500),(690,830)]:
  a,b=seg;s=slice(a-1,b-1);print(' ',seg,'dR',np.percentile(dr[s],[50,90,99,100]),'aR',np.percentile(accR[s-1],[50,90,99,100]),'dT',np.percentile(dt[s],[50,90,99,100]))
PY"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
TypeError: unsupported operand type(s) for -: 'slice' and 'int'
baseline dR [  0.58778094   6.51910826  13.34983524  29.9398902  112.82357703] dT [0.01026534 0.04405159 0.07225686 0.12696664 0.2355822 ] aR [  0.95964032  13.71854097  46.21523041 112.82847594] aT [0.01385379 0.07647594 0.19372279 0.29413597]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for name,p in [('baseline','/work/inputs/reconstruction.npz'),('inter3','/work/scratch/T_intermediate3.npy'),('trans','/work/scratch/T_translated.npy')]:
 T=np.load(p)['T_world_mesh'].astype(float) if p.endswith('.npz') else np.load(p)
 R=Rot.from_matrix(T[:,:3,:3]);dv=(R[1:]*R[:-1].inv()).as_rotvec();dr=np.degrees(np.linalg.norm(dv,axis=1));dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
 accR=np.degrees(np.linalg.norm(np.diff(dv,axis=0),axis=1));accT=np.linalg.norm(np.diff(T[:,:3,3],n=2,axis=0),axis=1)
 print(name,'dR',np.percentile(dr,[50,90,95,99,100]),'dT',np.percentile(dt,[50,90,95,99,100]),'aR',np.percentile(accR,[50,90,99,100]),'aT',np.percentile(accT,[50,90,99,100]))
 for a,b in [(150,325),(390,500),(690,830)]:
  s=np.arange(a-1,b-1);ss=np.arange(a-1,b-2);print(' ',a,b,'dR',np.percentile(dr[s],[50,90,99,100]),'aR',np.percentile(accR[ss],[50,90,99,100]),'dT',np.percentile(dt[s],[50,90,99,100]))
PY"
```
<details><summary>output</summary>

```
baseline dR [  0.58778094   6.51910826  13.34983524  29.9398902  112.82357703] dT [0.01026534 0.04405159 0.07225686 0.12696664 0.2355822 ] aR [  0.95964032  13.71854097  46.21523041 112.82847594] aT [0.01385379 0.07647594 0.19372279 0.29413597]
  150 325 dR [  1.41675178  22.01214785  39.90645113 112.82357703] aR [  2.88485207  30.94631107  77.71818274 112.82847594] dT [0.02437202 0.08348649 0.14578167 0.15652778]
  390 500 dR [0.77295751 1.57352303 2.83126064 3.14167322] aR [1.01354895 1.81410479 3.17306934 4.05999511] dT [0.00966982 0.01900804 0.02898739 0.03170365]
  690 830 dR [ 3.87174416 14.30010018 31.54180819 44.20084193] aR [ 7.1411215  25.06805402 42.28842162 50.63570207] dT [0.02642702 0.08609456 0.15594008 0.2355822 ]
inter3 dR [0.30027619 4.25784802 4.96205137 7.01648705 9.668 …[+1533 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_single_static.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS,NAMES,measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
def evalcand(T0,valf):
 Tsall=T0[None]@SYMS;vals=np.zeros(24);ious=np.zeros(24);chs=np.zeros(24);grs=np.zeros(24)
 for i in valf:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(Tsall,x0,y0,w,h);ms=mm.cpu().numpy();rs=rr.cpu().numpy();ds=dd.cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];dep=DEPS[i][y0:y0+h,x0:x0+w]
  for k in range(24):
   sc,d=measure(im[y0:y0+h,x0:x0+w],rs[k],ms[k],o,p);mask=ms[k]&o;dz=np.median(ds[k][mask])-np.median(dep[mask]) if mask.sum()>20 else 1
   vals[k]+=1.6*d['iou']+.9*d['chrom']+.5*d['grad']-2*abs(dz-.08);ious[k]+=d['iou'];chs[k]+=d['chrom'];grs[k]+=d['grad']
 return sorted([(vals[k]/len(valf),k,ious[k]/len(valf),chs[k]/len(valf),grs[k]/len(valf)) for k in range(24)],reverse=True)
for name,anchor,valf in [('g1',350,[330,340,360,380]),('g2',550,[510,530,555,665,680 …[+254 chars]
```
<details><summary>output</summary>

```
syms 24
g1 350
2.105 0 0.923 1.02 0.122
1.316 22 0.858 0.262 0.026
1.284 2 0.92 0.175 0.015
1.268 15 0.818 0.187 0.001
1.18 5 0.688 0.311 0.004
1.177 12 0.819 0.076 0.004
g2 550
2.146 3 0.945 1.214 0.212
1.333 1 0.943 0.388 0.084
1.268 23 0.887 0.442 0.016
0.888 0 0.947 -0.081 0.027
0.829 13 0.731 0.196 -0.008
0.803 8 0.693 0.237 0.032
fin 850
2.298 3 0.938 0.99 0.181
1.864 23 0.911 0.621 0.007
1.58 1 0.932 0.279 0.046
1.408 0 0.941 0.082 0.023
1.359 13 0.654 0.426 -0.011
1.356 5 0.534 0.607 -0.006

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
for name,anchor,k,cur in [('ground1',350,0,'staticTz_ground1'),('ground2',550,3,'staticTz_ground2'),('final',850,3,'staticTz_final')]:
 T=np.load(f'/work/scratch/globalTz_{anchor}.npy')@SYMS[k];C=np.load(f'/work/scratch/{cur}.npy');print(name,'rd',np.degrees(Rot.from_matrix(T[:3,:3]@C[:3,:3].T).magnitude()),'dt',np.linalg.norm(T[:3,3]-C[:3,3]),'t',T[:3,3],C[:3,3])
PY"
```
<details><summary>output</summary>

```
syms 24
ground1 rd 2.517825581320396e-13 dt 0.0 t [-0.2052929   1.10791649  3.39060328] [-0.2052929   1.10791649  3.39060328]
ground2 rd 14.604900481831116 dt 0.03352358924184586 t [0.36103107 1.0726447  3.15364876] [0.35669998 1.08533608 3.18437337]
final rd 4.256282818588156 dt 0.11094354324168261 t [0.10760694 0.20974956 3.323916  ] [0.10802192 0.21864403 3.43450165]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
for name,anchor,k in [('ground1',350,0),('ground2',550,3),('final',850,3)]:
 np.save(f'/work/scratch/direct_{name}.npy',np.load(f'/work/scratch/globalTz_{anchor}.npy')@SYMS[k])
PY
cat > /work/scratch/check_direct.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEP={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
for name,fs in [('ground1',range(330,386,6)),('ground2',list(range(510,581,6))+list(range(660,686,4))),('final',range(835,922,6))]:
 print(name)
 for fn in [f'direct_{name}',f'staticTz_{name}',f'staticRef_{name}']:
  T=np.load(f'/work/scratch/{fn}.npy');v=[]
  for i in fs:
   cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(T[None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();d=dd[0].cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];s,z=measure(im[y0:y0+h,x0:x0+w],r,m,o,p);mask=m&o;dz=np.median(d[mask])-np.median(DEP[i][y0:y0+h,x0:x0 …[+142 chars]
```
<details><summary>output</summary>

```
syms 24
syms 24
ground1
direct_ground1 [0.921 1.025 0.119 0.152]
staticTz_ground1 [0.921 1.025 0.119 0.152]
staticRef_ground1 [0.924 1.084 0.138 0.21 ]
ground2
direct_ground2 [0.945 1.214 0.212 0.279]
staticTz_ground2 [ 0.888  0.718 -0.018  0.239]
staticRef_ground2 [0.893 0.719 0.048 0.237]
final
direct_final [0.937 0.987 0.178 0.081]
staticTz_final [0.919 0.846 0.089 0.043]
staticRef_final [0.913 0.809 0.102 0.044]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
s=open('/work/scratch/test_single_static.py').read().split(\"for name,anchor,valf\")[0];exec(s)
import numpy as np
T0=np.load('/work/scratch/globalTz_0.npy');r=evalcand(T0,[0,10,20,30,40,50,60]);print(r[:10])
PY"
```
<details><summary>output</summary>

```
syms 24
[(1.4671752918029966, 0, 0.9123623120301748, 0.270948297263557, 0.03608083751584802), (1.3262593019041042, 1, 0.9169171302100689, 0.10540989857935458, 0.039819931877510886), (1.21142822633241, 23, 0.8169209635608657, 0.10547469902113901, 0.009700514974870853), (1.2106887833224056, 2, 0.9162591181943526, -0.005227598360264836, 0.009641156989216273), (1.2002472124825734, 21, 0.8008394015155035, 0.11605826123029768, 0.024126030238611356), (1.1558730609964134, 3, 0.9161384475183542, -0.08926337249155106, 0.0488842422408717), (1.1313626084137742, 12, 0.642094977698286, 0.24023240266475082, 0.028062396815844944), (1.0531178169942363, 20, 0.8277780961315281, -0.08186109197682075, -0.0028077576503814533), (1.0113372130022795, 22, 0.8117030499879928, -0.08324362814725939, -0.029915604474289 …[+93 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/build_intermediate4.py <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from viterbi_aug import seg_vit,get
Tinit=np.load('/work/scratch/staticTz_initialbase.npy');Tg1=np.load('/work/scratch/direct_ground1.npy');Tg2=np.load('/work/scratch/direct_ground2.npy');Tfin=np.load('/work/scratch/direct_final.npy')
segs=[('A',[i for i in range(150,321,10)],Tinit,Tg1,.55),('B',[i for i in range(400,500,10)],Tg1,Tg2,.9),('C',[i for i in range(700,831,10)],Tg2,Tfin,.75)]
keys={0:Tinit.copy(),149:Tinit.copy(),325:Tg1.copy(),389:Tg1.copy(),390:Tg1.copy(),499:Tg2.copy(),500:Tg2.copy(),689:Tg2.copy(),690:Tg2.copy(),830:Tfin.copy(),921:Tfin.copy()}
for name,fs,sT,eT,mx in segs:
 p=seg_vit(fs,sT,eT,lam=.5,hi=3,maxstep=mx)
 for j,i in enumerate(fs):keys[i]=get(i,'T')[p[j+1]].copy()
 print(name,p)
def slerpT(A,B,alpha):
 TA=Rot.from_matrix(A[:3,:3]);TB=Rot.from_matrix(B[:3,:3]);d=(TB*TA.inv()).as_rotvec();R=Rot.from_rotvec(d*alpha)*TA;T=np.eye(4);T[:3,:3]=R.as_matrix();T[:3,3]=(1-alpha)*A[:3,3]+alpha*B[:3,3];return T
kf=sorted(keys);Ts=np.zeros((922,4,4))
for i in range(922):
 if i<=kf[0]:Ts[i]=keys[kf[0]]
 elif i>=kf[-1]:Ts[i]=keys[kf[-1]]
 else:
  j=np.searchsorted(kf,i,side='right')-1;a,b=kf[j],kf[j+1];Ts[i]=slerpT(keys[a],keys[b],(i-a)/(b-a))
np.save('/work/scratch/T_intermediate4.npy',Ts)
PY
python /work/scratch/build_intermediate4.py
sed 's#T_intermediate3.npy#T_intermediate4.npy#; s#T_ …[+157 chars]
```
<details><summary>output</summary>

```
A [0, 24, 24, 24, 24, 24, 24, 3, 1, 10, 18, 1, 3, 21, 8, 4, 21, 21, 4, 0]
B [0, 17, 17, 3, 8, 8, 3, 5, 3, 22, 22, 0]
C [0, 3, 22, 3, 0, 3, 3, 21, 5, 16, 3, 3, 3, 3, 3, 0]
160 area 3617 w 0.52 x [ 3.387 -0.019  0.054] loss 0.0989
180 area 2222 w 0.32 x [3.491 0.    0.   ] loss 0.0876
200 area 1137 w 0.16 x [ 3.571  0.    -0.   ] loss 0.0708
220 area 1610 w 0.23 x [3.655 0.    0.   ] loss 0.0098
240 area 1326 w 0.19 x [3.429 0.    0.   ] loss 0.0126
260 area 2782 w 0.4 x [ 3.794  0.02  -0.014] loss 0.0552
280 area 3318 w 0.47 x [ 3.645  0.    -0.   ] loss 0.2763
300 area 5937 w 0.85 x [3.243 0.    0.   ] loss 0.1414
320 area 6843 w 0.98 x [ 3.633  0.    -0.   ] loss 0.3046
400 area 7671 w 1.0 x [3.615 0.    0.   ] loss 0.2819
420 area 6556 w 0.94 x [3.341 0.    0.   ] loss 0.1973
440 area 69 …[+548 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_finaltraj.py <<'PY'
import numpy as np,h5py,cv2,sys
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
S=np.load('/work/scratch/T_translated4.npy');B=np.load('/work/inputs/reconstruction.npz')['T_world_mesh']
def ev(T):
 rows=[]
 for i in range(0,922,5):
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(T[i][None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();d=dd[0].cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];sc,z=measure(im[y0:y0+h,x0:x0+w],r,m,o,p);mask=m&o;dz=np.median(d[mask])-np.median(DEPS[i][y0:y0+h,x0:x0+w][mask]) if mask.sum()>20 else np.nan;rows.append([i,z['iou'],z['chrom'],z['grad'],dz,z['gray']])
 return np.array(rows)
for name,T in [('baseline',B),('result',S)]:
 R=ev(T);np.savez(f'/work/scratch/eval_final_{name}.npz',R=R);print(name)
 for lo,hi in [(0,150),(150,325),(325,390),(390,500),(500,690),(690,830),(830,922)]:
  Q=R[(R[:,0]>=lo)&(R[:,0]<hi)];print(f'{lo}-{hi} iou {np.nanmean(Q[ …[+184 chars]
```
<details><summary>output</summary>

```
syms 24
baseline
0-150 iou 0.884 chrom 0.995 grad 0.168 |dz| 0.101 gray 0.398
150-325 iou 0.672 chrom 0.150 grad 0.019 |dz| 0.538 gray 0.012
325-390 iou 0.880 chrom -0.012 grad -0.010 |dz| 0.082 gray -0.050
390-500 iou 0.846 chrom -0.017 grad -0.004 |dz| 0.106 gray -0.045
500-690 iou 0.924 chrom -0.077 grad 0.008 |dz| 0.141 gray -0.082
690-830 iou 0.640 chrom 0.027 grad -0.022 |dz| 0.498 gray -0.058
830-922 iou 0.598 chrom -0.169 grad 0.009 |dz| 0.100 gray -0.039
result
0-150 iou 0.885 chrom 1.049 grad 0.275 |dz| 0.054 gray 0.474
150-325 iou 0.719 chrom 0.275 grad 0.019 |dz| 0.555 gray 0.046
325-390 iou 0.922 chrom 1.014 grad 0.120 |dz| 0.094 gray 0.151
390-500 iou 0.729 chrom 0.359 grad 0.015 |dz| 0.078 gray 0.046
500-690 iou 0.944 chrom 1.219 grad 0.217 |dz| 0.223 gray 0.259
690-830 iou  …[+114 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/refine_pose_mask.py <<'PY'
import numpy as np,h5py,cv2,sys,trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();lo,hi=geo.bounds
corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
def raster(poly,x0,y0,w,h):
 m=np.zeros((h,w),np.uint8);p=np.round(poly-[x0,y0]).astype(np.int32);cv2.fillConvexPoly(m,cv2.convexHull(p),1);return m.astype(bool)
T=np.load('/work/scratch/T_translated4.npy');out=T.copy();areas=oms.sum((1,2))
def optframe(i,lamrot=.3):
 Tc=T[i];cam=Tcw@Tc;c0=cam[:3,3];z0=c0[2];uv0=np.array([K[0,0]*c0[0]/c0[2]+K[0,2],K[1,1]*c0[1]/c0[2]+K[1,2]])
 x0,y0,w,h=crop_box(bb[i],pad=25,min_size=240);o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];vis=~cv2.dilate(p.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
 wgt=float(np.clip(areas[i]/7000,.06,1.0))
 def make(x):
  z,du,dv,rx,ry,rz=x;R=Rot.from_rotvec([rx,ry,rz]).as_matrix().astype(np.float32)@Tc[:3,:3];tcam=np.array([(uv0[0]+du-K[0,2])*z/K[0,0],(uv0[1]+dv-K[1,2])*z/K[1,1],z]);Tn=np.eye(4);Tn[:3,:3]=R;Tn[:3,3]=tcam;return Twc@Tn
 def f(x):
  Tn=make(x);pc=(Tcw@(Tn@corners.T)).T;u=K[0,0]*pc[:,0]/pc[:,2]+K[0,2];v=K …[+931 chars]
```
<details><summary>output</summary>

```
160 area 3617 loss 0.1062 x [ 3.387e+00 -2.000e-03  2.500e-02  3.800e-02 -3.700e-02 -4.500e-02]
180 area 2222 loss 0.0777 x [ 3.492  0.    -0.     0.    -0.077 -0.   ]
200 area 1137 loss 0.0547 x [ 3.568e+00  1.000e-03 -6.000e-03  1.140e-01 -7.100e-02  4.100e-02]
220 area 1610 loss 0.0116 x [ 3.654e+00 -3.000e-03  3.000e-03  3.700e-02 -5.000e-03  3.000e-03]
240 area 1326 loss 0.0111 x [ 3.428e+00  3.000e-03  6.000e-03  6.000e-03 -6.200e-02 -9.000e-03]
260 area 2782 loss 0.0361 x [ 3.773  0.009 -0.015  0.03   0.043 -0.016]
280 area 3318 loss 0.254 x [ 3.645  0.     0.     0.075 -0.    -0.   ]
300 area 5937 loss 0.0866 x [ 3.228e+00 -5.000e-03 -3.000e-03 -6.600e-02 -7.000e-03  1.000e-02]
320 area 6843 loss 0.2843 x [ 3.642e+00  0.000e+00  0.000e+00 -6.400e-02 -0.000e+00  1.000e-03]
400 area  …[+1086 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/scratch/eval_dyn_compare.py';s=open(p).read();s=s.replace(\"Ts=[np.load('/work/scratch/T_intermediate3.npy'),np.load('/work/scratch/T_translated.npy')]\", \"Ts=[np.load('/work/scratch/T_translated4.npy'),np.load('/work/scratch/T_poseRefined.npy')]\")
open('/work/scratch/eval_poseRef.py','w').write(s)
PY
python /work/scratch/eval_poseRef.py"
```
<details><summary>output</summary>

```
syms 24

PHASE 150 325
0 [ 0.71898269  0.27484898  0.01893657 -0.26233593]
1 [ 0.75274646  0.33431879  0.05808016 -0.3169482 ]

PHASE 390 500
0 [ 0.72921185  0.35905798  0.01490354 -0.0018517 ]
1 [0.7643754  0.40729511 0.00954842 0.00195244]

PHASE 690 830
0 [ 0.82326247  0.63362003  0.04301004 -0.36208844]
1 [ 0.82962081  0.59996617  0.04020954 -0.36380353]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np
from scipy.spatial.transform import Rotation as Rot
for name,p in [('base','/work/inputs/reconstruction.npz'),('res','/work/scratch/T_poseRefined.npy')]:
 T=np.load(p)['T_world_mesh'].astype(float) if p.endswith('.npz') else np.load(p);R=Rot.from_matrix(T[:,:3,:3]);dv=(R[1:]*R[:-1].inv()).as_rotvec();dr=np.degrees(np.linalg.norm(dv,axis=1));dt=np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)
 print(name,'all dR',np.percentile(dr,[50,90,99,100]),'dT',np.percentile(dt,[50,90,99,100]))
 for a,b in [(150,325),(390,500),(690,830)]:
  s=np.arange(a-1,b-1);print(a,b,'dR',np.percentile(dr[s],[50,90,99,100]),'dT',np.percentile(dt[s],[50,90,99,100]))
PY"
```
<details><summary>output</summary>

```
base all dR [  0.58778094   6.51910826  29.9398902  112.82357703] dT [0.01026534 0.04405159 0.12696664 0.2355822 ]
150 325 dR [  1.41675178  22.01214785  39.90645113 112.82357703] dT [0.02437202 0.08348649 0.14578167 0.15652778]
390 500 dR [0.77295751 1.57352303 2.83126064 3.14167322] dT [0.00966982 0.01900804 0.02898739 0.03170365]
690 830 dR [ 3.87174416 14.30010018 31.54180819 44.20084193] dT [0.02642702 0.08609456 0.15594008 0.2355822 ]
res all dR [ 0.34906732  6.67126942 16.42592258 28.85789462] dT [0.00128156 0.0307995  0.06023349 0.14107159]
150 325 dR [ 3.8267936  10.24139322 18.02631625 18.57117672] dT [0.01674207 0.05455071 0.07633862 0.14107159]
390 500 dR [ 4.94696037 13.41692921 28.2775073  28.85789462] dT [0.02100893 0.04194994 0.05281467 0.06560621]
690 830 dR [ 3.19202507   …[+85 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/smooth_traj.py <<'PY'
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot
T=np.load('/work/scratch/T_poseRefined.npy').copy();O=T.copy()
for a,b in [(150,325),(390,500),(690,830)]:
 rv=Rot.from_matrix(T[a:b,:3,:3]).as_rotvec()
 # unwrap components (simple)
 for k in range(3):rv[:,k]=np.unwrap(rv[:,k])
 rvs=savgol_filter(rv,5,2,axis=0,mode='interp');ts=savgol_filter(T[a:b,:3,3],5,2,axis=0,mode='interp')
 # preserve endpoints exactly
 rvs[0]=rv[0];rvs[-1]=rv[-1];ts[0]=T[a,:3,3];ts[-1]=T[b-1,:3,3]
 for i in range(a,b):T[i,:3,:3]=Rot.from_rotvec(rvs[i-a]).as_matrix();T[i,:3,3]=ts[i-a]
np.save('/work/scratch/T_smoothed.npy',T)
PY
python /work/scratch/smooth_traj.py
python - <<'PY'
p='/work/scratch/eval_dyn_compare.py';s=open(p).read().replace(\"Ts=[np.load('/work/scratch/T_intermediate3.npy'),np.load('/work/scratch/T_translated.npy')]\", \"Ts=[np.load('/work/scratch/T_poseRefined.npy'),np.load('/work/scratch/T_smoothed.npy')]\")
open('/work/scratch/eval_smoothed.py','w').write(s)
PY
python /work/scratch/eval_smoothed.py"
```
<details><summary>output</summary>

```
syms 24

PHASE 150 325
0 [ 0.75274646  0.33431879  0.05808016 -0.3169482 ]
1 [ 0.68447167  0.23523751  0.06469704 -0.31766599]

PHASE 390 500
0 [0.7643754  0.40729511 0.00954842 0.00195244]
1 [0.74982437 0.32670026 0.01550452 0.00153552]

PHASE 690 830
0 [ 0.82962081  0.59996617  0.04020954 -0.36380353]
1 [ 0.69157506  0.22333524  0.01140215 -0.35368994]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "stat -c '%y %n' /work/scratch/pose_utils.py /work/scratch/candidate_metrics.py /work/scratch/candidate_metrics.npz /work/scratch/eval_sym.py"
```
<details><summary>output</summary>

```
2026-10-20 18:03:01.813205964 +0400 /work/scratch/pose_utils.py
2026-10-20 18:12:06.427218948 +0400 /work/scratch/candidate_metrics.py
2026-10-20 18:43:59.254264554 +0400 /work/scratch/candidate_metrics.npz
2026-10-20 17:50:09.798187558 +0400 /work/scratch/eval_sym.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'python /work/scratch/candidate_metrics.py 2>&1 | tee /work/scratch/candidate_metrics_fixed.log && python /work/scratch/add_baseline_candidates.py 2>&1 | tee /work/scratch/add_baseline_fixed.log'
```
<details><summary>output</summary>

```
syms 24
80 best 3 1.733 next 1.035
100 best 3 2.273 next 1.627
120 best 3 0.885 next 0.779
140 best 3 1.841 next 1.105
150 best 3 0.981 next 0.6
160 best 3 1.12 next 1.036
170 best 3 0.987 next 0.928
180 best 3 1.197 next 1.164
190 best 3 -4.556 next -4.779
200 best 3 -0.836 next -0.861
210 best 3 -4.522 next -4.568
220 best 3 -1.113 next -1.366
230 best 3 1.226 next 1.177
240 best 3 0.643 next 0.364
250 best 3 0.712 next 0.556
260 best 3 1.648 next 1.314
270 best 3 1.35 next 1.21
280 best 3 1.443 next 1.339
290 best 3 1.517 next 1.464
300 best 3 1.337 next 1.189
310 best 3 1.473 next 1.343
320 best 3 1.374 next 1.141
330 best 3 1.612 next 1.534
390 best 3 1.424 next 1.323
400 best 3 1.218 next 1.066
410 best 3 1.145 next 1.014
420 best 3 1.994 next 1.286
430 best 3 1.436 next 1.356
440 be …[+2020 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c 'git -C /code/reconstruction rev-parse HEAD 2>/dev/null || true; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
```
<details><summary>output</summary>

```
7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80
887 MiB
12865 MiB

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy,scipy,cv2,trimesh,torch,h5py,nvdiffrast,PIL
for x in [numpy,scipy,cv2,trimesh,torch,h5py,PIL]:print(x.__name__,getattr(x,'__version__','?'))
import importlib.metadata as m
for p in ['nvdiffrast','opencv-python','trimesh','numpy','scipy','torch','h5py','Pillow']:
 try:print(p,m.version(p))
 except Exception as e:print(p,e)
PY"
```
<details><summary>output</summary>

```
numpy 1.26.3
scipy 1.16.3
cv2 4.11.0
trimesh 5.1.1
torch 2.5.1+cu124
h5py 3.16.0
PIL 10.2.0
nvdiffrast 0.4.0
opencv-python 4.11.0.86
trimesh 5.1.1
numpy 1.26.3
scipy 1.16.3
torch 2.5.1+cu124
h5py 3.16.0
Pillow 10.2.0

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/outputs/solve.py <<'PY'
#"'!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF snack-box trajectory.

The method uses the exact textured mesh, SAM2 masks, the released camera, MoGe-2
depth, the CARI4D baseline, and a temporal hypothesis decoder.  It is deliberately
self contained: all pose estimates below are recomputed from the inputs.
"""
import argparse
import itertools
import json
import os
import time
from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation as Rot

import nvdiffrast.torch as dr

# ------------------------------ deterministic setup ------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# These boundaries were verified by eye on the video.  They are fixed, commented
# constants (rather than per-frame fitted data): the object is stationary while
# untouched/resting, and moves only in the three manipulation intervals.
PHASES = {
    "desk_initial": (0, 149),       # box rests on the white desk
    "pickup_to_floor": (150, 324),  # carried down and released
    "floor_initial": (325, 389),    # box rests on the floor
    "foot_push": (390, 499),        # foot pushes/tumbles box through desk
    "floor_final": (500, 689),      # box rests again on the floor
    "l …[+25273 chars]
```
<details><summary>output</summary>

```
674 /work/outputs/solve.py

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cd /work && /usr/bin/time -f 'ELAPSED %e sec MAXRSS %M KB' python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/outputs/solve.log"
```
<details><summary>output</summary>

```
/usr/bin/bash: line 1: /usr/bin/time: No such file or directory

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "TIMEFORMAT='ELAPSED %R sec'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/outputs/solve.log"
```
<details><summary>output</summary>

```
Traceback (most recent call last):
  File "/work/outputs/solve.py", line 674, in <module>
    main()
  File "/work/outputs/solve.py", line 666, in main
    solver = Solver(args.inputs, args.out)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/work/outputs/solve.py", line 270, in __init__
    self.both[i] = [x.min(), x.max() + 1, y.min(), y.max() + 1]
    ~~~~~~~~~^^^
ValueError: could not broadcast input array from shape (4,) into shape (2,)
ELAPSED 7.863 sec

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -i 's/self.both = np.zeros((self.F, 2), np.int32)/self.both = np.zeros((self.F, 4), np.int32)/' /work/outputs/solve.py
TIMEFORMAT='ELAPSED %R sec'; time python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz 2>&1 | tee /work/outputs/solve.log"
```
<details><summary>output</summary>

```
[1/5] estimating static phase poses
    desk_initial [0.0251 0.2041 3.3398]
    floor_initial [-0.2053  1.1079  3.3906]
    floor_final [0.3635 1.0831 3.1523]
    desk_final [0.0761 0.1528 3.2978]
[2/5] fitting and decoding dynamic keyframe hypotheses
    A [0, 24, 24, 24, 24, 2, 24, 19, 3, 3, 20, 1, 21, 3, 10, 3, 20, 3, 21, 0]
    B [0, 19, 0, 3, 1, 2, 7, 6, 14, 13, 3, 0]
    C [0, 23, 0, 2, 24, 24, 24, 24, 5, 5, 24, 24, 24, 24, 24, 0]
[3/5] interpolating the selected trajectory
[4/5] per-frame silhouette translation alignment
[5/5] local rigid-pose refinement from image evidence
wrote /work/outputs/object_poses_world.npz (922 frames) in 9.11 minutes
ELAPSED 558.089 sec

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/eval_output.py <<'PY'
import numpy as np,h5py,cv2,sys,time
sys.path.insert(0,'/work/scratch');from eval_sym import measure
from pose_utils import *
D=np.load('/work/scratch/diag.npz');bb=D['bb'];cap=cv2.VideoCapture('/work/inputs/video.mp4')
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
with h5py.File('/pipeline/depth/moge2/aligned.h5','r') as f: DEPS={i:cv2.imdecode(f['aligned/front_stereo_camera_left'][i],cv2.IMREAD_UNCHANGED).astype(np.float32)/1000 for i in range(922)}
T=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];print('shape',T.shape,T.dtype,'finite',np.isfinite(T).all())
rows=[]
for i in range(0,922,5):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);x0,y0,w,h=crop_box(bb[i],pad=30,min_size=300);mm,rr,dd=render_batch(T[i][None],x0,y0,w,h);m=mm[0].cpu().numpy();r=rr[0].cpu().numpy();d=dd[0].cpu().numpy();o=oms[i,y0:y0+h,x0:x0+w];p=pms[i,y0:y0+h,x0:x0+w];s,z=measure(im[y0:y0+h,x0:x0+w],r,m,o,p);mask=m&o;dz=np.median(d[mask])-np.median(DEPS[i][y0:y0+h,x0:x0+w][mask]) if mask.sum()>20 else np.nan;rows.append([i,z['iou'],z['chrom'],z['grad'],dz,z['gray']])
R=np.array(rows);np.savez('/work/scratch/eval_output.npz',R=R)
for lo,hi in [(0,150),(150,325),(325,390),(390,500),(500,690),(690,830),(830,922)]:
 Q=R[(R[:,0]>=lo)&(R[:,0]<hi)];print(f'{lo}-{hi} iou {np.nanmean(Q[:,1]):.3f} chrom {np.nanmean(Q[:,2]):.3f} grad {np.nanmean(Q[:,3]):.3f} | …[+420 chars]
```
<details><summary>output</summary>

```
syms 24
shape (922, 4, 4) float64 finite True
0-150 iou 0.885 chrom 1.049 grad 0.275 |dz| 0.054 gray 0.474
150-325 iou 0.775 chrom 0.226 grad 0.056 |dz| 0.577 gray 0.094
325-390 iou 0.922 chrom 1.014 grad 0.120 |dz| 0.094 gray 0.151
390-500 iou 0.783 chrom 0.107 grad 0.016 |dz| 0.103 gray 0.114
500-690 iou 0.884 chrom 0.424 grad 0.010 |dz| 0.206 gray 0.109
690-830 iou 0.641 chrom -0.123 grad -0.068 |dz| 0.302 gray -0.059
830-922 iou 0.623 chrom -0.165 grad -0.009 |dz| 0.031 gray -0.089
dR pct [ 0.40225101  6.96965081 15.40174484 28.34858647] dT [0.00083043 0.02913374 0.06227034 0.13970695]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
S=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];
for i,ref in [(350,'ground1'),(550,'ground2'),(921,'final'),(820,'dyn820')]:
 p=f'/work/scratch/'+('direct_'+ref+'.npy' if ref in ['ground1','ground2','final'] else 'dynT_820.npy')
 T=np.load(p);ds=[]
 for k,H in enumerate(SYMS):ds.append((np.degrees(Rot.from_matrix(S[i,:3,:3]@H[:3,:3]@T[:3,:3].T).magnitude()),k))
 print(i,'out',S[i,:3,3],Rot.from_matrix(S[i,:3,:3]).as_rotvec(),'ref',T[:3,3],Rot.from_matrix(T[:3,:3]).as_rotvec(),'nearestsym',sorted(ds)[:3])
PY"
```
<details><summary>output</summary>

```
syms 24
350 out [-0.20529278  1.10791636  3.39060307] [-0.41386779  0.06508307 -3.09167483] ref [-0.2052929   1.10791649  3.39060328] [-0.41386777  0.06508307 -3.09167483] nearestsym [(5.565354584709946e-07, 3), (89.99999944731654, 21), (89.99999993569307, 7)]
550 out [0.36352712 1.08312237 3.15229654] [-0.72812918 -0.72708544  0.16547821] ref [0.36103107 1.0726447  3.15364876] [ 1.11791857 -0.02458803  2.93544244] nearestsym [(42.7854035342471, 22), (60.951894336049925, 12), (70.8619431569393, 18)]
921 out [0.07609792 0.15284303 3.29780555] [-1.04787538  1.15773957 -1.09246675] ref [0.10760694 0.20974956 3.323916  ] [ 0.84562052 -0.10047795  3.02132675] nearestsym [(37.74299799311625, 17), (53.50720148252751, 9), (89.77878227319907, 4)]
820 out [0.06611888 0.20002884 3.24158384] [-0.54957 …[+191 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
p='/work/outputs/solve.py';s=open(p).read();s=s.replace('def bbox_pose_fit(self, i, zspan=0.16, popsize=12, maxiter=180):','def bbox_pose_fit(self, i, zspan=0.16, popsize=12, maxiter=180, seed=SEED):').replace('            seed=SEED,\\n            popsize=popsize,','            seed=seed,\\n            popsize=popsize,')
open(p,'w').write(s)
PY
cat > /work/scratch/test_multiseed.py <<'PY'
import sys,numpy as np,time
sys.path.insert(0,'/work/outputs');import solve
S=solve.Solver('/work/inputs','/work/scratch/x.npz')
for phase,anchor,vals in [('g2',550,[510,530,555,665,680]),('fin',850,[840,860,880,900,921])]:
 print('\\n',phase)
 for seed in range(6):
  t=time.time();T0=S.bbox_pose_fit(anchor,zspan=.16,popsize=12,maxiter=180,seed=seed);T,score,det=S.best_symmetry(T0,vals)
  print(seed,'sec',round(time.time()-t,1),'score',round(score,3),'t',np.round(T[:3,3],3),'rv',np.round(solve.Rot.from_matrix(T[:3,:3]).as_rotvec(),2),det,flush=True)
S.close()
PY
python /work/scratch/test_multiseed.py"
```
<details><summary>output</summary>

```

 g2
0 sec 4.7 score 1.272 t [0.364 1.083 3.152] rv [-0.73 -0.73  0.17] {'iou': 0.8868105160045735, 'chrom': 0.41255106942698133, 'grad': 0.014448452880606056, 'dz': -0.18281254768371583}
1 sec 3.6 score 2.247 t [0.362 1.07  3.143] rv [-1.13  0.03 -2.93] {'iou': 0.9521148117572216, 'chrom': 1.2944842922917257, 'grad': 0.28756662607192995, 'dz': -0.21237764358520508}
2 sec 4.4 score 1.244 t [0.361 1.08  3.145] rv [-0.72 -0.74  0.17] {'iou': 0.889582213535667, 'chrom': 0.40535484380640047, 'grad': 0.0005848939705174417, 'dz': -0.19201102256774902}
3 sec 3.5 score 2.154 t [0.362 1.074 3.159] rv [-1.12  0.02 -2.92] {'iou': 0.9419021927265344, 'chrom': 1.2220089003909336, 'grad': 0.20136524736881256, 'dz': -0.196882963180542}
4 sec 4.4 score 1.182 t [0.35  1.075 3.125] rv [-0.64 -0.49  0.24] {' …[+1392 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_more_seeds.py <<'PY'
import sys,numpy as np,time
sys.path.insert(0,'/work/outputs');import solve
S=solve.Solver('/work/inputs','/work/scratch/x.npz')
for seed in range(6,16):
 t=time.time();T0=S.bbox_pose_fit(850,zspan=.16,popsize=12,maxiter=180,seed=seed);T,score,det=S.best_symmetry(T0,[840,860,880,900,921])
 print(seed,round(score,3),'t',np.round(T[:3,3],3),'rv',np.round(solve.Rot.from_matrix(T[:3,:3]).as_rotvec(),2),det,flush=True)
S.close()
PY
python /work/scratch/test_more_seeds.py"
```
<details><summary>output</summary>

```
6 1.343 t [0.07  0.172 3.455] rv [-1.39  1.62 -0.87] {'iou': 0.9736767842658305, 'chrom': 0.06412273349723965, 'grad': 0.010894728405401111, 'dz': 0.21908936500549317}
7 2.121 t [0.093 0.17  3.344] rv [ 0.71 -0.51  2.96] {'iou': 0.9555974071633224, 'chrom': 0.7595999444978683, 'grad': 0.10167693048715591, 'dz': 0.008944034576416016}
8 1.364 t [0.04  0.143 3.454] rv [-1.71  0.66  0.45] {'iou': 0.9352776760251593, 'chrom': 0.06330435450900439, 'grad': 0.03448421470820904, 'dz': 0.18343677520751953}
9 1.49 t [0.093 0.161 3.298] rv [-0.87  1.57 -0.63] {'iou': 0.9307198055518168, 'chrom': 0.013748060399808327, 'grad': 0.027694259956479072, 'dz': 0.08547697067260743}
10 1.488 t [0.038 0.135 3.438] rv [-0.07  1.75 -2.58] {'iou': 0.9374844567190099, 'chrom': 0.14294601056117062, 'grad': 0.00134225 …[+887 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
S=solve.Solver('/work/inputs','/work/scratch/x.npz')
T=np.load('/work/scratch/direct_final.npy')
for i in [840,860,880,900,921]:
 sc=S.candidate_scores(T[None],i)[0];print(i,sc[0],sc[1],sc[2])
S.close()
PY"
```
<details><summary>output</summary>

```
840 2.268944266331166 {'iou': 0.9582637729549248, 'chrom': 0.84788446274439, 'grad': 0.21813137829303741, 'gray': 0.48700842972437536, 'sat': 0.6025740498568983, 'ca': 0.24519490002330835, 'cb': 0.529500786966803} 0.011780261993408203
860 2.178393112192838 {'iou': 0.9524400330851944, 'chrom': 0.8295006140592067, 'grad': 0.215690016746521, 'gray': 0.4924066741784896, 'sat': 0.5886701552415464, 'ca': 0.2450323037141118, 'cb': 0.5139340697224775} -0.019953250885009766
880 2.18454658625462 {'iou': 0.9528243327126009, 'chrom': 0.878931595265608, 'grad': 0.21382789313793182, 'gray': 0.4867169338105494, 'sat': 0.5999276668317496, 'ca': 0.3174353612034525, 'cb': 0.5170777814936082} -0.038962364196777344
900 2.1759093123113282 {'iou': 0.9530312435340368, 'chrom': 0.8341392341605816, 'grad': 0.20848 …[+381 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
sys.path.insert(0,'/work/outputs');import solve
S=solve.Solver('/work/inputs','/work/scratch/x.npz')
for seed in [0,1,7,12]:
 T0=S.bbox_pose_fit(850,zspan=.16,popsize=12,maxiter=180,seed=seed);print('seed',seed,'t',T0[:3,3],'rv',solve.Rot.from_matrix(T0[:3,:3]).as_rotvec())
S.close()
T=np.load('/work/scratch/globalTz_850.npy');print('scratch','t',T[:3,3],'rv',solve.Rot.from_matrix(T[:3,:3]).as_rotvec())
PY"
```
<details><summary>output</summary>

```
seed 0 t [0.07609792 0.15284303 3.2978055 ] rv [1.21490312 1.09961438 1.42144127]
seed 1 t [0.04179421 0.14500923 3.4851122 ] rv [ 0.76165566 -0.09252281  0.18786924]
seed 7 t [0.09311534 0.16993514 3.3436794 ] rv [ 0.70877702 -0.51333701  2.96140186]
seed 12 t [0.1190942  0.19651628 3.2742858 ] rv [ 0.11341772  0.70504527 -0.04405994]
scratch t [0.10760694 0.20974956 3.323916  ] rv [ 0.84562052 -0.10047795  3.02132675]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/scratch');import fit_mask_global as f
T,r=f.fitglobal(850,0);print('now scratch',T[:3,3],f.Rot.from_matrix(T[:3,:3]).as_rotvec(),r.fun)
PY"
```
<details><summary>output</summary>

```
global 850 0.9800285306704708 [ 3.28940164 -3.44069282  3.5297262   0.78350192 -0.16485564  3.04854417]
now scratch [0.10760694 0.20974956 3.323916  ] [ 0.84562052 -0.10047795  3.02132675] 0.0199714693295292

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "sed -n '1,70p' /work/scratch/fit_mask_global.py; sed -n '300,380p' /work/outputs/solve.py"
```
<details><summary>output</summary>

```
import numpy as np,h5py,cv2,sys,json,trimesh
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from pose_utils import *
Z=np.load('/work/inputs/reconstruction.npz');Ts=Z['T_world_mesh'].astype(np.float32);D=np.load('/work/scratch/diag.npz');bb=D['bb']
with h5py.File('/work/inputs/masks.h5') as f:oms=f['object'][:];pms=f['person'][:]
geo=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry();lo,hi=geo.bounds
corners=np.array([[x,y,z,1] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
def make_T(R,z,du,dv,uv0):
 tcam=np.array([(uv0[0]+du-K[0,2])*z/K[0,0],(uv0[1]+dv-K[1,2])*z/K[1,1],z]);T=np.eye(4);T[:3,:3]=R;T[:3,3]=tcam;return Twc@T
def raste …[+4137 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import sys,numpy as np
sys.path.insert(0,'/work/outputs');import solve
s=solve.Solver('/work/inputs','/work/scratch/x.npz')
print(s.renderer.corners)
s.close()
sys.path.insert(0,'/work/scratch');import fit_mask_global as f
print(f.corners)
PY"
```
<details><summary>output</summary>

```
[[-0.17087026 -0.10298643 -0.20244676  1.        ]
 [-0.17087026 -0.10298643  0.20245828  1.        ]
 [-0.17087026  0.10250368 -0.20244676  1.        ]
 [-0.17087026  0.10250368  0.20245828  1.        ]
 [ 0.16922666 -0.10298643 -0.20244676  1.        ]
 [ 0.16922666 -0.10298643  0.20245828  1.        ]
 [ 0.16922666  0.10250368 -0.20244676  1.        ]
 [ 0.16922666  0.10250368  0.20245828  1.        ]]
[[-0.17087026 -0.10298643 -0.20244676  1.        ]
 [-0.17087026 -0.10298643  0.20245828  1.        ]
 [-0.17087026  0.10250368 -0.20244676  1.        ]
 [-0.17087026  0.10250368  0.20245828  1.        ]
 [ 0.16922666 -0.10298643 -0.20244676  1.        ]
 [ 0.16922666 -0.10298643  0.20245828  1.        ]
 [ 0.16922666  0.10250368 -0.20244676  1.        ]
 [ 0.16922666  0.10250368  0.20245 …[+18 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "cat > /work/scratch/test_final_more.py <<'PY'
import sys,numpy as np,time
sys.path.insert(0,'/work/outputs');import solve
S=solve.Solver('/work/inputs','/work/scratch/x.npz')
best=(0,None)
for seed in range(16,41):
 T0=S.bbox_pose_fit(850,zspan=.16,popsize=12,maxiter=180,seed=seed);T,score,det=S.best_symmetry(T0,[840,860,880,900,921])
 print(seed,round(score,3),'t',np.round(T[:3,3],3),'rv',np.round(solve.Rot.from_matrix(T[:3,:3]).as_rotvec(),2),det,flush=True)
 if score>best[0]:best=(score,T)
print('BEST',best[0],best[1][:3,3],solve.Rot.from_matrix(best[1][:3,:3]).as_rotvec())
np.save('/work/scratch/best_final_seedsearch.npy',best[1])
S.close()
PY
python /work/scratch/test_final_more.py 2>&1 | tee /work/scratch/test_final_more.log"
```
<details><summary>output</summary>

```
16 1.234 t [0.004 0.089 3.351] rv [-1.12  1.54 -0.77] {'iou': 0.6395407334589767, 'chrom': 0.36614548101310723, 'grad': -0.03796611800789833, 'dz': 0.1301197052001953}
17 1.495 t [0.047 0.13  3.402] rv [ 0.42  1.44 -2.36] {'iou': 0.938733702134732, 'chrom': 0.07150615160951632, 'grad': 0.026626498252153397, 'dz': 0.12229356765747071}
18 1.611 t [0.059 0.146 3.404] rv [ 0.2   1.05 -2.77] {'iou': 0.9141251336277756, 'chrom': 0.2193375393957572, 'grad': -0.02127706091850996, 'dz': 0.09737577438354492}
19 1.444 t [0.034 0.142 3.514] rv [-0.31  0.67 -3.03] {'iou': 0.759232928673016, 'chrom': 0.4508933090592267, 'grad': 0.017268733587116003, 'dz': 0.17268171310424804}
20 1.133 t [0.068 0.177 3.524] rv [-1.02  1.16 -1.06] {'iou': 0.9491705996045414, 'chrom': 0.019844504305994028, 'grad': -0.00113 …[+3515 chars]
```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
A=np.load('/work/scratch/direct_final.npy');B=np.load('/work/scratch/best_final_seedsearch.npy');print(np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude()),np.linalg.norm(A[:3,3]-B[:3,3]))
# nearest proper symmetry
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
print(sorted([(np.degrees(Rot.from_matrix(A[:3,:3]@S[:3,:3]@B[:3,:3].T).magnitude()),i) for i,S in enumerate(SYMS)])[:5])
PY"
```
<details><summary>output</summary>

```
17.960366038287766 0.15592257907906995
syms 24
[(17.960366038287766, 3), (78.10976421177777, 21), (78.99618918236251, 6), (87.00721898526135, 10), (95.6267443218083, 9)]

```
</details>

**shell** (exit 0)
```
/usr/bin/bash -c "python - <<'PY'
import numpy as np,sys
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0,'/work/scratch');from eval_sym import SYMS
A=np.load('/work/scratch/direct_final.npy')
# seed7 T not saved; generate from earlier? load test? no
for f in ['/work/scratch/best_final_seedsearch.npy']:
 B=np.load(f);print(f,np.degrees(Rot.from_matrix(A[:3,:3]@B[:3,:3].T).magnitude()))
PY"
```
<details><summary>output</summary>

```
syms 24
/work/scratch/best_final_seedsearch.npy 17.960366038287766

```
</details>
